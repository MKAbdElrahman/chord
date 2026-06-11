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
        let mut text = String::new();
        input.read_to_string(&mut text)?;
        let text = text.trim();
        if text.is_empty() {
            return Err(ChordError::BadInput("no input text".to_string()).into());
        }

        let lang = opts.get_or("lang", "en").to_string();
        let steps: usize = opts.get_or("steps", "8").parse().unwrap_or(8);
        let speed: f32 = opts.get_or("speed", "1.05").parse().unwrap_or(1.05);
        let silence: f32 = opts.get_or("silence", "0.3").parse().unwrap_or(0.3);

        let engine = engine_for(opts)?;
        let mut engine = engine
            .lock()
            .map_err(|_| ChordError::Engine("tts engine lock poisoned".into()))?;
        let sr = engine.cfg.ae.sample_rate;

        let max_len = if lang == "ko" || lang == "ja" {
            120
        } else {
            300
        };
        let chunks = preprocess::chunk_text(text, max_len);

        let mut wav_all: Vec<f32> = Vec::new();
        for (i, chunk) in chunks.iter().enumerate() {
            let (wav, dur) = engine.infer(chunk, &lang, steps, speed)?;
            let n = ((sr as f32) * dur) as usize;
            let clip = &wav[..n.min(wav.len())];
            if i > 0 {
                let gap = (silence * sr as f32) as usize;
                wav_all.extend(std::iter::repeat_n(0.0_f32, gap));
            }
            wav_all.extend_from_slice(clip);
        }

        write_wav(output, &wav_all, sr as u32)
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
static ENGINES: OnceLock<Mutex<HashMap<(PathBuf, String), Arc<Mutex<Engine>>>>> = OnceLock::new();

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

fn write_wav(output: &mut dyn Write, samples: &[f32], sample_rate: u32) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut buf = std::io::Cursor::new(Vec::<u8>::new());
    {
        let mut writer = hound::WavWriter::new(&mut buf, spec)?;
        for &s in samples {
            let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
            writer.write_sample(v)?;
        }
        writer.finalize()?;
    }
    output.write_all(&buf.into_inner())?;
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
