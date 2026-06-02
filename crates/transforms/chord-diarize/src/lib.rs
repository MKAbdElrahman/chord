//! chord-diarize — speaker diarization (`audio -> text`): *who spoke when*.
//!
//! Backed by **sherpa-onnx** (pyannote segmentation, a speaker-embedding model,
//! and clustering), the same engine as the `sherpa-onnx` stt backend. Output is
//! a line per speaker turn with timestamps — pair it with `stt` for a
//! speaker-attributed transcript.

use std::io::{Read, Write};

use chord_core::{ChordError, Kind, OptionSpec, Options, Result, Transform};
use sherpa_rs::diarize::{Diarize, DiarizeConfig};

const DEF_SEG: &str = "hf:csukuangfj/sherpa-onnx-pyannote-segmentation-3-0:model.onnx";
const DEF_EMB: &str =
    "hf:csukuangfj/speaker-embedding-models:3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx";

const OPTS: &[OptionSpec] = &[
    OptionSpec {
        key: "segmentation",
        help: "pyannote segmentation ONNX (path or hf: ref)",
        takes_value: true,
    },
    OptionSpec {
        key: "embedding",
        help: "speaker-embedding ONNX (path or hf: ref)",
        takes_value: true,
    },
    OptionSpec {
        key: "speakers",
        help: "fixed number of speakers; omit/-1 to auto-detect",
        takes_value: true,
    },
    OptionSpec {
        key: "threshold",
        help: "clustering threshold when auto-detecting (default 0.5)",
        takes_value: true,
    },
];

pub struct Diarizer;

impl Transform for Diarizer {
    fn name(&self) -> &str {
        "diarize"
    }
    fn from(&self) -> Kind {
        Kind::Audio
    }
    fn to(&self) -> Kind {
        Kind::Text
    }
    fn describe(&self) -> &str {
        "speaker diarization: who spoke when (sherpa-onnx / pyannote)"
    }
    fn backend(&self) -> &str {
        "sherpa-onnx"
    }
    fn options(&self) -> &'static [OptionSpec] {
        OPTS
    }

    fn apply(&self, input: &mut dyn Read, output: &mut dyn Write, opts: &Options) -> Result<()> {
        let segmentation = chord_hf::resolve(opts.get_or("segmentation", DEF_SEG))?;
        let embedding = chord_hf::resolve(opts.get_or("embedding", DEF_EMB))?;
        let speakers: i32 = opts.get_or("speakers", "-1").parse().unwrap_or(-1);
        let threshold: f32 = opts.get_or("threshold", "0.5").parse().unwrap_or(0.5);

        let mut wav = Vec::new();
        input.read_to_end(&mut wav)?;
        if wav.is_empty() {
            return Err(ChordError::BadInput("no audio on input".to_string()).into());
        }
        let (samples, src_rate) = decode_wav_to_mono_f32(&wav)?;
        // sherpa-onnx diarization expects 16 kHz mono.
        let samples = resample_linear(&samples, src_rate, 16_000);

        let config = DiarizeConfig {
            // num_clusters > 0 fixes the speaker count; <= 0 auto-detects via threshold.
            num_clusters: Some(speakers),
            threshold: Some(threshold),
            ..Default::default()
        };

        let mut diarizer = Diarize::new(&segmentation, &embedding, config)
            .map_err(|e| ChordError::Engine(format!("sherpa-onnx diarization: {e}")))?;
        let segments = diarizer
            .compute(samples, None)
            .map_err(|e| ChordError::Engine(format!("diarize: {e}")))?;

        for s in &segments {
            writeln!(
                output,
                "[{:>8.2} - {:>8.2}] speaker {}",
                s.start, s.end, s.speaker
            )?;
        }
        Ok(())
    }
}

/// Decode WAV bytes to mono f32, returning (samples, sample_rate).
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

/// Linear-interpolation resampler.
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
