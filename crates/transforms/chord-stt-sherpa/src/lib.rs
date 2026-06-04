//! chord-stt-sherpa — `stt` (audio -> text) backed by **sherpa-onnx**, running
//! NVIDIA NeMo transducer models, in particular **Parakeet-TDT-0.6B-v3** — the
//! top *multilingual* model on the Open ASR Leaderboard (25 languages, beats
//! Whisper-large-v3, very fast on CPU).
//!
//! The model is four ONNX files (encoder/decoder/joiner) + a tokens file,
//! resolved as `hf:` references; sherpa-onnx handles features + TDT decoding.

use std::io::{Read, Write};

use chord_core::{ChordError, Kind, OptionSpec, Options, Result, Unary};
use sherpa_rs::transducer::{TransducerConfig, TransducerRecognizer};

const REPO: &str = "csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8";
const DEF_ENCODER: &str =
    "hf:csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8:encoder.int8.onnx";
const DEF_DECODER: &str =
    "hf:csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8:decoder.int8.onnx";
const DEF_JOINER: &str =
    "hf:csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8:joiner.int8.onnx";
const DEF_TOKENS: &str = "hf:csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8:tokens.txt";

const OPTS: &[OptionSpec] = &[
    OptionSpec {
        key: "encoder",
        help: "encoder ONNX (path or hf: ref)",
        takes_value: true,
    },
    OptionSpec {
        key: "decoder",
        help: "decoder ONNX (path or hf: ref)",
        takes_value: true,
    },
    OptionSpec {
        key: "joiner",
        help: "joiner ONNX (path or hf: ref)",
        takes_value: true,
    },
    OptionSpec {
        key: "tokens",
        help: "tokens file (path or hf: ref)",
        takes_value: true,
    },
    OptionSpec {
        key: "threads",
        help: "CPU threads (default 4)",
        takes_value: true,
    },
    OptionSpec {
        key: "chunk",
        help: "seconds per chunk for long audio (default 300; Parakeet caps ~13min/pass)",
        takes_value: true,
    },
];

pub struct SttSherpa;

impl Unary for SttSherpa {
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
        "speech-to-text via NeMo transducer (sherpa-onnx; Parakeet)"
    }
    fn backend(&self) -> &str {
        "sherpa-onnx"
    }
    fn options(&self) -> &'static [OptionSpec] {
        OPTS
    }

    fn apply(&self, input: &mut dyn Read, output: &mut dyn Write, opts: &Options) -> Result<()> {
        let encoder = chord_hf::resolve(opts.get_or("encoder", DEF_ENCODER))?;
        let decoder = chord_hf::resolve(opts.get_or("decoder", DEF_DECODER))?;
        let joiner = chord_hf::resolve(opts.get_or("joiner", DEF_JOINER))?;
        let tokens = chord_hf::resolve(opts.get_or("tokens", DEF_TOKENS))?;
        let threads: i32 = opts.get_or("threads", "4").parse().unwrap_or(4);

        let mut wav = Vec::new();
        input.read_to_end(&mut wav)?;
        if wav.is_empty() {
            return Err(ChordError::BadInput("no audio on input".to_string()).into());
        }
        let (samples, rate) = decode_wav_to_mono_f32(&wav)?;

        let to_str = |p: &std::path::Path| -> Result<String> {
            p.to_str().map(str::to_string).ok_or_else(|| {
                ChordError::Engine(format!("non-UTF-8 path: {}", p.display())).into()
            })
        };

        let config = TransducerConfig {
            encoder: to_str(&encoder)?,
            decoder: to_str(&decoder)?,
            joiner: to_str(&joiner)?,
            tokens: to_str(&tokens)?,
            // NeMo transducer (Parakeet/Canary); 80-dim log-mel @ 16 kHz, greedy.
            model_type: "nemo_transducer".to_string(),
            sample_rate: 16_000,
            feature_dim: 80,
            decoding_method: "greedy_search".to_string(),
            num_threads: threads,
            ..Default::default()
        };

        let mut recognizer = TransducerRecognizer::new(config)
            .map_err(|e| ChordError::Engine(format!("sherpa-onnx ({REPO}): {e}")))?;

        // Parakeet's offline encoder caps input length (~13 min), so transcribe
        // long audio in fixed windows and concatenate. One window for short clips.
        let chunk_secs: usize = opts.get_or("chunk", "300").parse().unwrap_or(300);
        let window = chunk_secs.max(1) * rate as usize;
        let mut full = String::new();
        let mut start = 0;
        while start < samples.len() {
            let end = (start + window).min(samples.len());
            let part = recognizer.transcribe(rate, &samples[start..end]);
            let part = part.trim();
            if !part.is_empty() {
                if !full.is_empty() {
                    full.push(' ');
                }
                full.push_str(part);
            }
            start = end;
        }

        writeln!(output, "{}", full.trim())?;
        Ok(())
    }
}

/// Decode WAV bytes to mono f32, returning (samples, sample_rate). sherpa-onnx
/// resamples to its configured 16 kHz internally.
fn decode_wav_to_mono_f32(bytes: &[u8]) -> Result<(Vec<f32>, u32)> {
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
            .map(|f| f.iter().sum::<f32>() / channels as f32)
            .collect()
    };
    Ok((mono, spec.sample_rate))
}
