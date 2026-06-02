//! chord-stt — speech-to-text plug-in (`audio -> text`).
//!
//! Backed by whisper.cpp through the `whisper-rs` binding crate. The plug-in
//! decodes the incoming audio to the 16 kHz mono f32 whisper expects, runs
//! inference, and writes the transcript.
//!
//! Options (via `-o key=value`):
//!   - `model` — model file path, or a short name resolved to
//!     `~/models/ggml-<name>.bin` (default `large-v3-turbo`;
//!     also honored via `$CHORD_STT_MODEL`).
//!   - `lang`  — language code (e.g. `en`, `de`); omitted = whisper default.
//!   - `threads` — CPU threads (default 4).

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use chord_core::{ChordError, Kind, OptionSpec, Options, Result, Transform};

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

impl Transform for Stt {
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
    fn options(&self) -> &'static [OptionSpec] {
        OPTS
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

        let ctx = WhisperContext::new_with_params(
            model.to_str().ok_or("model path is not valid UTF-8")?,
            WhisperContextParameters::default(),
        )?;

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        let threads: i32 = opts.get_or("threads", "4").parse().unwrap_or(4);
        params.set_n_threads(threads);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        if let Some(lang) = opts.get("lang") {
            params.set_language(Some(lang));
        }

        let mut state = ctx.create_state()?;
        state.full(params, &samples)?;

        let mut text = String::new();
        for segment in state.as_iter() {
            text.push_str(&segment.to_string());
        }
        writeln!(output, "{}", text.trim())?;
        Ok(())
    }
}

/// Resolve the model: an existing path is used as-is; otherwise the value is a
/// short catalog name resolved to `~/models/ggml-<name>.bin`.
fn resolve_model(opts: &Options) -> Result<PathBuf> {
    let spec = opts
        .get("model")
        .map(str::to_string)
        .or_else(|| std::env::var("CHORD_STT_MODEL").ok())
        .unwrap_or_else(|| "large-v3-turbo".to_string());

    if chord_hf::is_hf(&spec) {
        return chord_hf::resolve(&spec);
    }

    let direct = Path::new(&spec);
    if direct.exists() {
        return Ok(direct.to_path_buf());
    }

    let home = std::env::var("HOME").unwrap_or_default();
    let candidate = Path::new(&home)
        .join("models")
        .join(format!("ggml-{spec}.bin"));
    if candidate.exists() {
        return Ok(candidate);
    }

    Err(ChordError::ModelMissing {
        what: format!("whisper model {spec:?} (looked at {})", candidate.display()),
        hint: "run `chord pull stt`, or set --model to a ggml model path".to_string(),
    }
    .into())
}

/// Decode WAV bytes to mono f32 samples at 16 kHz (what whisper expects).
fn decode_wav_to_16k_mono(bytes: &[u8]) -> Result<Vec<f32>> {
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
