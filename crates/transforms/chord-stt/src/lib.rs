//! chord-stt — speech-to-text plug-in (`audio -> text`).
//!
//! Backed by whisper.cpp through the `whisper-rs` binding crate. The plug-in
//! decodes the incoming audio to the 16 kHz mono f32 whisper expects, runs
//! inference, and writes the transcript.
//!
//! Options (via `-o key=value`):
//!   - `model` — model file path, or a short name resolved to
//!     `<XDG data>/chord/models/ggml-<name>.bin` (default `large-v3-turbo`;
//!     also honored via `$CHORD_STT_MODEL`).
//!   - `lang`  — language code (e.g. `en`, `de`); omitted = whisper default.
//!   - `threads` — CPU threads (default 4).

use std::io::{self, BufRead, Read, Write};
use std::path::{Path, PathBuf};

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use chord_core::{ChordError, Kind, OptionSpec, Options, ResourceSpec, Result, Unary};

/// The default model, declared so `chord pull stt` works without the host
/// hardcoding it (R8). The URL basename matches what `resolve_model_spec`
/// expects in `models_dir()` for the default "large-v3-turbo" spec.
const RESOURCES: &[ResourceSpec] = &[ResourceSpec {
    key: "model",
    spec: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin",
    describe: "whisper large-v3-turbo (~1.5 GB)",
}];

const OPTS: &[OptionSpec] = &[
    OptionSpec {
        key: "model",
        help: "whisper model path or name (default large-v3-turbo)",
        takes_value: true,
    },
    OptionSpec {
        key: "lang",
        help: "language code, e.g. en, de (default auto)",
        takes_value: true,
    },
    OptionSpec {
        key: "threads",
        help: "CPU threads (default 4)",
        takes_value: true,
    },
];
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

pub struct Stt;

impl Unary for Stt {
    fn name(&self) -> &str {
        "stt"
    }
    fn from(&self) -> Kind {
        Kind::Audio
    }
    fn to(&self) -> Kind {
        Kind::Text
    }
    fn describe(&self) -> &str {
        "speech-to-text (whisper.cpp)"
    }
    fn backend(&self) -> &str {
        "whisper.cpp"
    }
    fn options(&self) -> &'static [OptionSpec] {
        OPTS
    }

    fn resources(&self) -> &'static [ResourceSpec] {
        RESOURCES
    }

    fn apply(&self, input: &mut dyn Read, output: &mut dyn Write, opts: &Options) -> Result<()> {
        // Route whisper.cpp/ggml logs into hooks that go nowhere (we don't
        // enable the log/tracing backends), silencing the model-load spam on
        // stderr. Idempotent.
        whisper_rs::install_logging_hooks();

        let model = resolve_model(opts)?;

        let mut bytes = Vec::new();
        input.read_to_end(&mut bytes)?;
        let samples = decode_wav_to_16k_mono(&bytes)?;

        let ctx = context_for(&model)?;

        let threads: i32 = opts.get_or("threads", "4").parse().unwrap_or(4);
        let text = transcribe_one(&ctx, &samples, opts.get("lang"), threads)?;
        writeln!(output, "{}", text.trim())?;
        Ok(())
    }
}

/// Loaded whisper contexts, memoized per process by model path: the first
/// apply pays the load; every later one (the `--each` batch loop) reuses it.
/// Friedman-Wise forcing semantics — evaluate once, store, never re-evaluate
/// (rule R5 in chord's docs/theory/THEORY.md).
static CONTEXTS: OnceLock<Mutex<HashMap<PathBuf, Arc<WhisperContext>>>> = OnceLock::new();

fn context_for(model: &Path) -> Result<Arc<WhisperContext>> {
    let cache = CONTEXTS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| ChordError::Engine("whisper context cache poisoned".into()))?;
    if let Some(ctx) = cache.get(model) {
        return Ok(ctx.clone());
    }
    let ctx = Arc::new(WhisperContext::new_with_params(
        model.to_str().ok_or("model path is not valid UTF-8")?,
        WhisperContextParameters::default(),
    )?);
    cache.insert(model.to_path_buf(), ctx.clone());
    Ok(ctx)
}

/// Transcribe one clip with an already-loaded context (the model load is the
/// expensive part, so batch callers load `ctx` once and reuse it).
fn transcribe_one(
    ctx: &WhisperContext,
    samples: &[f32],
    lang: Option<&str>,
    threads: i32,
) -> Result<String> {
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_n_threads(threads);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    if let Some(l) = lang {
        params.set_language(Some(l));
    }
    let mut state = ctx.create_state()?;
    state.full(params, samples)?;
    let mut text = String::new();
    for segment in state.as_iter() {
        text.push_str(&segment.to_string());
    }
    Ok(text)
}

/// `chord-stt --batch`: load the whisper model ONCE, then transcribe a list of
/// files read from stdin (one `path` or `path<TAB>lang` per line), printing one
/// transcript line per input (in order). This removes the per-call model-load
/// cost that dominates per-segment pipelines. Flags are parsed by the binary's
/// clap front-end (see `main.rs`) and passed in typed.
pub fn run_batch(model: Option<String>, default_lang: Option<String>, threads: i32) -> Result<()> {
    whisper_rs::install_logging_hooks();
    let model = resolve_model_spec(model)?;

    let ctx = WhisperContext::new_with_params(
        model.to_str().ok_or("model path is not valid UTF-8")?,
        WhisperContextParameters::default(),
    )?;

    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = stdout.lock();
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let mut it = line.splitn(2, '\t');
        let path = it.next().unwrap_or("").trim();
        let lang = it
            .next()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .or_else(|| default_lang.clone());
        let bytes = std::fs::read(path)?;
        let samples = decode_wav_to_16k_mono(&bytes)?;
        let text = transcribe_one(&ctx, &samples, lang.as_deref(), threads)?;
        writeln!(out, "{}", text.replace('\n', " ").trim())?;
    }
    Ok(())
}

/// Resolve the model: an existing path is used as-is; otherwise the value is a
/// short catalog name resolved to `~/models/ggml-<name>.bin`.
fn resolve_model(opts: &Options) -> Result<PathBuf> {
    resolve_model_spec(opts.get("model").map(str::to_string))
}

/// Resolve a model spec (None -> $CHORD_STT_MODEL -> default) to a file path.
fn resolve_model_spec(spec: Option<String>) -> Result<PathBuf> {
    let spec = spec
        .or_else(|| std::env::var("CHORD_STT_MODEL").ok())
        .unwrap_or_else(|| "large-v3-turbo".to_string());

    if chord_hf::is_hf(&spec) {
        return chord_hf::resolve(&spec);
    }

    let direct = Path::new(&spec);
    if direct.exists() {
        return Ok(direct.to_path_buf());
    }

    let candidate = chord_core::dirs::models_dir().join(format!("ggml-{spec}.bin"));
    if candidate.exists() {
        return Ok(candidate);
    }

    Err(ChordError::ModelMissing {
        what: format!("whisper model {spec:?} (looked at {})", candidate.display()),
        hint: "run `chord pull stt` (models now live under the XDG data dir; move \
               any legacy ~/models/* there), or set --model to a ggml model path"
            .to_string(),
    }
    .into())
}

/// Decode WAV bytes to mono f32 samples at 16 kHz (what whisper expects).
/// Patch the RIFF/data sizes of a *streaming* WAV (0xFFFFFFFF — the
/// unknown-length convention of live sources) to the true sizes computed
/// from the buffer we hold. No-op when sizes are already exact.
fn normalize_streaming_wav(bytes: &mut [u8]) {
    if bytes.len() < 44 || &bytes[..4] != b"RIFF" {
        return;
    }
    if bytes[4..8] == [0xFF; 4] {
        let riff = (bytes.len() as u32).saturating_sub(8).to_le_bytes();
        bytes[4..8].copy_from_slice(&riff);
    }
    // Walk the chunk list to the `data` chunk and patch its size.
    let mut off = 12;
    while off + 8 <= bytes.len() {
        let size = u32::from_le_bytes(bytes[off + 4..off + 8].try_into().unwrap());
        if &bytes[off..off + 4] == b"data" {
            if size == u32::MAX {
                let real = (bytes.len() - off - 8) as u32;
                bytes[off + 4..off + 8].copy_from_slice(&real.to_le_bytes());
            }
            return;
        }
        if size == u32::MAX {
            return; // malformed: an unknown-size non-data chunk
        }
        off += 8 + size as usize + (size as usize & 1);
    }
}

fn decode_wav_to_16k_mono(bytes: &[u8]) -> Result<Vec<f32>> {
    // A streaming producer (chord-tts, ffmpeg pipes) writes unknown-length
    // sizes; we hold the complete stream, so patch them before hound parses.
    let mut owned;
    let bytes = if bytes.len() >= 12 && &bytes[..4] == b"RIFF" && bytes[4..8] == [0xFF; 4] {
        owned = bytes.to_vec();
        normalize_streaming_wav(&mut owned);
        &owned[..]
    } else {
        bytes
    };
    let mut reader = hound::WavReader::new(std::io::Cursor::new(bytes))?;
    let spec = reader.spec();
    let channels = spec.channels.max(1) as usize;

    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<std::result::Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / scale))
                .collect::<std::result::Result<_, _>>()?
        }
    };

    let mono: Vec<f32> = if channels <= 1 {
        interleaved
    } else {
        interleaved
            .chunks(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect()
    };

    Ok(resample_linear(&mono, spec.sample_rate, 16_000))
}

/// Linear-interpolation resampler. whisper is robust to this; a higher-quality
/// resampler can be swapped in later without changing the plug-in contract.
fn resample_linear(input: &[f32], from: u32, to: u32) -> Vec<f32> {
    if input.is_empty() || from == to {
        return input.to_vec();
    }
    let ratio = to as f64 / from as f64;
    let out_len = ((input.len() as f64) * ratio).round() as usize;
    let last = input.len() - 1;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src = i as f64 / ratio;
        let idx = src.floor() as usize;
        let frac = (src - idx as f64) as f32;
        let a = input[idx.min(last)];
        let b = input[(idx + 1).min(last)];
        out.push(a + (b - a) * frac);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal streaming WAV: unknown-size header + n silent 16-bit samples.
    fn streaming_wav(sample_rate: u32, n_samples: usize) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(b"RIFF");
        b.extend_from_slice(&[0xFF; 4]);
        b.extend_from_slice(b"WAVE");
        b.extend_from_slice(b"fmt ");
        b.extend_from_slice(&16u32.to_le_bytes());
        b.extend_from_slice(&1u16.to_le_bytes()); // PCM
        b.extend_from_slice(&1u16.to_le_bytes()); // mono
        b.extend_from_slice(&sample_rate.to_le_bytes());
        b.extend_from_slice(&(sample_rate * 2).to_le_bytes());
        b.extend_from_slice(&2u16.to_le_bytes());
        b.extend_from_slice(&16u16.to_le_bytes());
        b.extend_from_slice(b"data");
        b.extend_from_slice(&[0xFF; 4]);
        b.extend(std::iter::repeat_n(0u8, n_samples * 2));
        b
    }

    #[test]
    fn streaming_wav_sizes_are_patched_to_true_values() {
        let mut bytes = streaming_wav(16000, 1600);
        normalize_streaming_wav(&mut bytes);
        let riff = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        assert_eq!(riff as usize, bytes.len() - 8);
        let data = u32::from_le_bytes(bytes[40..44].try_into().unwrap());
        assert_eq!(data as usize, 1600 * 2);
    }

    #[test]
    fn streaming_wav_decodes_after_normalization() {
        // The full decode path must accept a streaming (unknown-size) WAV —
        // this is what keeps `chord tts :: stt` working once tts streams.
        let bytes = streaming_wav(16000, 1600);
        let samples = decode_wav_to_16k_mono(&bytes).unwrap();
        assert_eq!(samples.len(), 1600);
        assert!(samples.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn exact_size_wav_is_left_untouched() {
        let mut bytes = streaming_wav(16000, 100);
        normalize_streaming_wav(&mut bytes); // patch once
        let before = bytes.clone();
        normalize_streaming_wav(&mut bytes); // idempotent on exact sizes
        assert_eq!(bytes, before);
    }
}
