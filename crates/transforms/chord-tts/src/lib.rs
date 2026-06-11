//! chord-tts — text-to-speech plug-in (`text -> audio`).
//!
//! Runs the Supertonic ONNX pipeline (duration predictor -> text encoder ->
//! vector estimator (flow-matching denoise loop) -> vocoder) via ONNX Runtime
//! (`ort`). Since chord synthesizes one text at a time, this runs at batch
//! size 1 (so the attention/latent masks are all-ones).
//!
//! Output: 16-bit mono WAV at the model's sample rate (44.1 kHz).
//!
//! Options: `assets` (dir, default <XDG data>/chord/tts/assets or $CHORD_TTS_ASSETS),
//! `voice` (preset name or path, default M1), `lang` (default en),
//! `steps` (denoise steps, default 8), `speed` (default 1.05),
//! `silence` (seconds between chunks, default 0.3).

mod preprocess;

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use chord_core::{ChordError, Kind, OptionSpec, Options, Result, Unary};
use ort::session::Session;

const OPTS: &[OptionSpec] = &[
    OptionSpec {
        key: "voice",
        help: "voice preset name or JSON path (default M1)",
        takes_value: true,
    },
    OptionSpec {
        key: "lang",
        help: "language code (default en)",
        takes_value: true,
    },
    OptionSpec {
        key: "steps",
        help: "denoising steps 5-12 (default 8)",
        takes_value: true,
    },
    OptionSpec {
        key: "speed",
        help: "speech speed 0.7-2.0 (default 1.05)",
        takes_value: true,
    },
    OptionSpec {
        key: "silence",
        help: "seconds between chunks (default 0.3)",
        takes_value: true,
    },
    OptionSpec {
        key: "assets",
        help: "Supertonic assets directory",
        takes_value: true,
    },
    OptionSpec {
        key: "chunk",
        help: "max characters per synthesis chunk; smaller starts audio sooner (default 300, ko/ja 120)",
        takes_value: true,
    },
];
use ort::value::Tensor;
use serde::Deserialize;

pub struct Tts;

impl Unary for Tts {
    fn name(&self) -> &str {
        "tts"
    }
    fn from(&self) -> Kind {
        Kind::Text
    }
    fn to(&self) -> Kind {
        Kind::Audio
    }
    fn describe(&self) -> &str {
        "text-to-speech (Supertonic / ONNX)"
    }
    fn backend(&self) -> &str {
        "onnxruntime"
    }
    fn options(&self) -> &'static [OptionSpec] {
        OPTS
    }

    fn apply(&self, input: &mut dyn Read, output: &mut dyn Write, opts: &Options) -> Result<()> {
        let lang = opts.get_or("lang", "en").to_string();
        let steps: usize = opts.get_or("steps", "8").parse().unwrap_or(8);
        let speed: f32 = opts.get_or("speed", "1.05").parse().unwrap_or(1.05);
        let silence: f32 = opts.get_or("silence", "0.3").parse().unwrap_or(0.3);

        let engine = engine_for(opts)?;
        let mut engine = engine
            .lock()
            .map_err(|_| ChordError::Engine("tts engine lock poisoned".into()))?;
        let sr = engine.cfg.ae.sample_rate;

        let default_len = if lang == "ko" || lang == "ja" {
            120
        } else {
            300
        };
        let max_len = opts
            .get("chunk")
            .and_then(|v| v.parse().ok())
            .filter(|&n: &usize| n > 0)
            .unwrap_or(default_len);

        // Incremental: synthesize each complete sentence as it arrives, so an
        // upstream token stream (chord chat) overlaps with synthesis instead
        // of waiting for its EOF. `raw` holds bytes whose UTF-8 may be split
        // mid-character by the pipe; only the valid prefix moves to `pending`.
        let mut raw: Vec<u8> = Vec::new();
        let mut pending = String::new();
        let mut emitted = 0usize;
        let mut buf = [0u8; 4096];
        loop {
            let n = input.read(&mut buf)?;
            if n == 0 {
                break;
            }
            raw.extend_from_slice(&buf[..n]);
            let valid_up_to = match std::str::from_utf8(&raw) {
                Ok(_) => raw.len(),
                Err(e) => e.valid_up_to(),
            };
            if valid_up_to > 0 {
                pending.push_str(std::str::from_utf8(&raw[..valid_up_to]).unwrap());
                raw.drain(..valid_up_to);
            }
            while let Some(chunk) = preprocess::take_stream_chunk(&mut pending, max_len) {
                synth_chunk(
                    &mut engine,
                    output,
                    &chunk,
                    &lang,
                    steps,
                    speed,
                    sr,
                    silence,
                    emitted == 0,
                )?;
                emitted += 1;
            }
        }
        // EOF: whatever remains goes through the offline chunker.
        pending.push_str(&String::from_utf8_lossy(&raw));
        for chunk in preprocess::chunk_text(pending.trim(), max_len) {
            synth_chunk(
                &mut engine,
                output,
                &chunk,
                &lang,
                steps,
                speed,
                sr,
                silence,
                emitted == 0,
            )?;
            emitted += 1;
        }
        if emitted == 0 {
            return Err(ChordError::BadInput("no input text".to_string()).into());
        }
        Ok(())
    }
}

// ---- config & assets --------------------------------------------------------

#[derive(Deserialize)]
struct Cfg {
    ae: Ae,
    ttl: Ttl,
}
#[derive(Deserialize)]
struct Ae {
    sample_rate: usize,
    base_chunk_size: usize,
}
#[derive(Deserialize)]
struct Ttl {
    chunk_compress_factor: usize,
    latent_dim: usize,
}

#[derive(Deserialize)]
struct StyleJson {
    style_ttl: TensorJson,
    style_dp: TensorJson,
}
#[derive(Deserialize)]
struct TensorJson {
    data: Vec<Vec<Vec<f64>>>,
    dims: Vec<i64>,
}

impl TensorJson {
    /// Flatten a [1, d1, d2] style tensor to (dims, row-major f32 data).
    fn flatten(&self) -> (Vec<i64>, Vec<f32>) {
        let mut flat = Vec::new();
        for b in &self.data {
            for row in b {
                for v in row {
                    flat.push(*v as f32);
                }
            }
        }
        (self.dims.clone(), flat)
    }
}

struct Engine {
    cfg: Cfg,
    indexer: Vec<i64>,
    ttl: (Vec<i64>, Vec<f32>),
    dp_style: (Vec<i64>, Vec<f32>),
    dp: Session,
    text_enc: Session,
    vector_est: Session,
    vocoder: Session,
}

/// Loaded engines, memoized per process by (assets dir, voice): the first
/// apply pays the ONNX session setup; every later one (the `--each` batch
/// loop) reuses it (rule R5 in chord's docs/theory/THEORY.md).
type EngineKey = (PathBuf, String);
static ENGINES: OnceLock<Mutex<HashMap<EngineKey, Arc<Mutex<Engine>>>>> = OnceLock::new();

fn engine_for(opts: &Options) -> Result<Arc<Mutex<Engine>>> {
    let key = (resolve_assets(opts), opts.get_or("voice", "M1").to_string());
    let cache = ENGINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| ChordError::Engine("tts engine cache poisoned".into()))?;
    if let Some(e) = cache.get(&key) {
        return Ok(e.clone());
    }
    let e = Arc::new(Mutex::new(Engine::load(opts)?));
    cache.insert(key, e.clone());
    Ok(e)
}

impl Engine {
    fn load(opts: &Options) -> Result<Engine> {
        let assets = resolve_assets(opts);
        let onnx = assets.join("onnx");

        let cfg: Cfg = serde_json::from_str(&read(onnx.join("tts.json"))?)?;
        let indexer: Vec<i64> = serde_json::from_str(&read(onnx.join("unicode_indexer.json"))?)?;

        let voice = resolve_voice(opts.get_or("voice", "M1"), &assets);
        let style: StyleJson = serde_json::from_str(&read(&voice)?)?;
        let ttl = style.style_ttl.flatten();
        let dp_style = style.style_dp.flatten();

        let session = |name: &str| -> Result<Session> {
            Ok(Session::builder()?.commit_from_file(onnx.join(name))?)
        };

        Ok(Engine {
            cfg,
            indexer,
            ttl,
            dp_style,
            dp: session("duration_predictor.onnx")?,
            text_enc: session("text_encoder.onnx")?,
            vector_est: session("vector_estimator.onnx")?,
            vocoder: session("vocoder.onnx")?,
        })
    }

    /// Synthesize one text chunk (batch size 1). Returns the full waveform and
    /// the predicted duration in seconds.
    fn infer(
        &mut self,
        text: &str,
        lang: &str,
        steps: usize,
        speed: f32,
    ) -> Result<(Vec<f32>, f32)> {
        let ids = preprocess::text_to_ids(text, lang, &self.indexer);
        let l = ids.len() as i64;
        let text_mask: Vec<f32> = vec![1.0; ids.len()];

        // Duration predictor -> seconds.
        let dur_out = {
            let ids_t = Tensor::from_array((vec![1, l], ids.clone()))?;
            let style_t = Tensor::from_array((self.dp_style.0.clone(), self.dp_style.1.clone()))?;
            let mask_t = Tensor::from_array((vec![1, 1, l], text_mask.clone()))?;
            let outputs = self.dp.run(ort::inputs![
                "text_ids" => ids_t, "style_dp" => style_t, "text_mask" => mask_t
            ])?;
            let (_, data) = outputs["duration"].try_extract_tensor::<f32>()?;
            data.to_vec()
        };
        let dur = dur_out[0] / speed;

        // Text encoder -> text embedding (constant across denoise steps).
        let (emb_shape, emb_data) = {
            let ids_t = Tensor::from_array((vec![1, l], ids.clone()))?;
            let style_t = Tensor::from_array((self.ttl.0.clone(), self.ttl.1.clone()))?;
            let mask_t = Tensor::from_array((vec![1, 1, l], text_mask.clone()))?;
            let outputs = self.text_enc.run(ort::inputs![
                "text_ids" => ids_t, "style_ttl" => style_t, "text_mask" => mask_t
            ])?;
            let (shape, data) = outputs["text_emb"].try_extract_tensor::<f32>()?;
            (shape.to_vec(), data.to_vec())
        };

        // Sample noisy latent ~ N(0,1). Masks are all-ones at batch size 1.
        let chunk_size = self.cfg.ae.base_chunk_size * self.cfg.ttl.chunk_compress_factor;
        let wav_len_max = (dur as f64) * self.cfg.ae.sample_rate as f64;
        let latent_len =
            ((wav_len_max + chunk_size as f64 - 1.0) / chunk_size as f64).floor() as usize;
        let latent_dim = self.cfg.ttl.latent_dim * self.cfg.ttl.chunk_compress_factor;
        let latent_mask: Vec<f32> = vec![1.0; latent_len];

        let mut rng = Rng::seeded();
        let mut xt: Vec<f32> = (0..latent_dim * latent_len)
            .map(|_| rng.gaussian())
            .collect();
        let latent_shape = vec![1, latent_dim as i64, latent_len as i64];

        // Flow-matching denoise loop.
        for step in 0..steps {
            let noisy = Tensor::from_array((latent_shape.clone(), xt.clone()))?;
            let emb = Tensor::from_array((emb_shape.clone(), emb_data.clone()))?;
            let style_t = Tensor::from_array((self.ttl.0.clone(), self.ttl.1.clone()))?;
            let lmask = Tensor::from_array((vec![1, 1, latent_len as i64], latent_mask.clone()))?;
            let tmask = Tensor::from_array((vec![1, 1, l], text_mask.clone()))?;
            let cur = Tensor::from_array((vec![1], vec![step as f32]))?;
            let tot = Tensor::from_array((vec![1], vec![steps as f32]))?;

            let outputs = self.vector_est.run(ort::inputs![
                "noisy_latent" => noisy,
                "text_emb" => emb,
                "style_ttl" => style_t,
                "latent_mask" => lmask,
                "text_mask" => tmask,
                "current_step" => cur,
                "total_step" => tot
            ])?;
            let (_, data) = outputs["denoised_latent"].try_extract_tensor::<f32>()?;
            xt = data.to_vec();
        }

        // Vocoder -> waveform.
        let wav = {
            let latent = Tensor::from_array((latent_shape.clone(), xt.clone()))?;
            let outputs = self.vocoder.run(ort::inputs!["latent" => latent])?;
            let (_, data) = outputs["wav_tts"].try_extract_tensor::<f32>()?;
            data.to_vec()
        };

        Ok((wav, dur))
    }
}

fn resolve_assets(opts: &Options) -> PathBuf {
    if let Some(a) = opts.get("assets") {
        return PathBuf::from(a);
    }
    if let Ok(a) = std::env::var("CHORD_TTS_ASSETS") {
        if !a.is_empty() {
            return PathBuf::from(a);
        }
    }
    chord_core::dirs::data_dir().join("tts").join("assets")
}

fn resolve_voice(voice: &str, assets: &Path) -> PathBuf {
    if voice.contains('/') || voice.ends_with(".json") {
        return PathBuf::from(voice);
    }
    assets.join("voice_styles").join(format!("{voice}.json"))
}

fn read(path: impl AsRef<Path>) -> Result<String> {
    let p = path.as_ref();
    std::fs::read_to_string(p).map_err(|e| format!("reading {}: {e}", p.display()).into())
}

// ---- WAV output -------------------------------------------------------------

/// Synthesize one chunk and stream it: the unknown-length WAV header before
/// the first chunk, an inter-chunk silence gap before later ones, then the
/// clip's PCM — flushed, so a downstream player starts immediately.
#[allow(clippy::too_many_arguments)]
fn synth_chunk(
    engine: &mut Engine,
    output: &mut dyn Write,
    chunk: &str,
    lang: &str,
    steps: usize,
    speed: f32,
    sr: usize,
    silence: f32,
    first: bool,
) -> Result<()> {
    if first {
        write_wav_stream_header(output, sr as u32)?;
    } else {
        let gap = (silence * sr as f32) as usize;
        write_pcm(output, &vec![0.0_f32; gap])?;
    }
    let (wav, dur) = engine.infer(chunk, lang, steps, speed)?;
    let n = ((sr as f32) * dur) as usize;
    write_pcm(output, &wav[..n.min(wav.len())])?;
    output.flush()?;
    Ok(())
}

/// WAV header for a stream of unknown length: RIFF and data sizes are
/// 0xFFFFFFFF (the ffmpeg/sox live-source convention). 16-bit mono PCM.
fn write_wav_stream_header(output: &mut dyn Write, sample_rate: u32) -> Result<()> {
    let byte_rate = sample_rate * 2;
    output.write_all(b"RIFF")?;
    output.write_all(&[0xFF; 4])?;
    output.write_all(b"WAVE")?;
    output.write_all(b"fmt ")?;
    output.write_all(&16u32.to_le_bytes())?;
    output.write_all(&1u16.to_le_bytes())?; // PCM
    output.write_all(&1u16.to_le_bytes())?; // mono
    output.write_all(&sample_rate.to_le_bytes())?;
    output.write_all(&byte_rate.to_le_bytes())?;
    output.write_all(&2u16.to_le_bytes())?; // block align
    output.write_all(&16u16.to_le_bytes())?; // bits per sample
    output.write_all(b"data")?;
    output.write_all(&[0xFF; 4])?;
    Ok(())
}

/// Append f32 samples as 16-bit little-endian PCM.
fn write_pcm(output: &mut dyn Write, samples: &[f32]) -> Result<()> {
    let mut buf = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        buf.extend_from_slice(&((s.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes());
    }
    output.write_all(&buf)?;
    Ok(())
}

// ---- tiny Gaussian RNG (SplitMix64 + Box-Muller) ----------------------------

struct Rng(u64);

impl Rng {
    fn seeded() -> Self {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x9E3779B97F4A7C15);
        Rng(seed | 1)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn gaussian(&mut self) -> f32 {
        let u1 = self.next_f64().max(1e-10);
        let u2 = self.next_f64();
        ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_header_is_a_valid_unknown_length_wav_preamble() {
        // The streaming convention (ffmpeg/sox): RIFF and data sizes are
        // 0xFFFFFFFF because a live source can't know its length. 44 bytes,
        // 16-bit mono PCM at the given rate.
        let mut buf = Vec::new();
        write_wav_stream_header(&mut buf, 44100).unwrap();
        assert_eq!(buf.len(), 44);
        assert_eq!(&buf[..4], b"RIFF");
        assert_eq!(&buf[4..8], &[0xFF; 4]); // unknown RIFF size
        assert_eq!(&buf[8..12], b"WAVE");
        assert_eq!(&buf[12..16], b"fmt ");
        assert_eq!(u16::from_le_bytes(buf[22..24].try_into().unwrap()), 1); // mono
        assert_eq!(
            u32::from_le_bytes(buf[24..28].try_into().unwrap()),
            44100 // sample rate
        );
        assert_eq!(u16::from_le_bytes(buf[34..36].try_into().unwrap()), 16); // bits
        assert_eq!(&buf[36..40], b"data");
        assert_eq!(&buf[40..44], &[0xFF; 4]); // unknown data size
    }

    #[test]
    fn write_pcm_converts_f32_to_i16_le() {
        let mut buf = Vec::new();
        write_pcm(&mut buf, &[0.0, 1.0, -1.0]).unwrap();
        assert_eq!(buf.len(), 6);
        assert_eq!(i16::from_le_bytes(buf[0..2].try_into().unwrap()), 0);
        assert_eq!(i16::from_le_bytes(buf[2..4].try_into().unwrap()), 32767);
        assert_eq!(i16::from_le_bytes(buf[4..6].try_into().unwrap()), -32767);
    }
}
