//! chord-langid — spoken language identification (`audio -> text`): prints the
//! detected language code (e.g. `de`, `en`), via sherpa-onnx's Whisper-based
//! language ID. An atomic building block — run it on a VAD segment to decide
//! which model/language to transcribe that section with.

use std::io::{Read, Write};

use chord_core::{ChordError, Kind, OptionSpec, Options, Result, Transform};
use sherpa_rs::language_id::{SpokenLanguageId, SpokenLanguageIdConfig};

const DEF_ENCODER: &str = "hf:csukuangfj/sherpa-onnx-whisper-tiny:tiny-encoder.onnx";
const DEF_DECODER: &str = "hf:csukuangfj/sherpa-onnx-whisper-tiny:tiny-decoder.onnx";
const RATE: u32 = 16_000;

const OPTS: &[OptionSpec] = &[
    OptionSpec {
        key: "encoder",
        help: "Whisper encoder ONNX (path or hf: ref)",
        takes_value: true,
    },
    OptionSpec {
        key: "decoder",
        help: "Whisper decoder ONNX (path or hf: ref)",
        takes_value: true,
    },
];

pub struct LangId;

impl Transform for LangId {
    fn name(&self) -> &str {
        "langid"
    }
    fn from(&self) -> Kind {
        Kind::Audio
    }
    fn to(&self) -> Kind {
        Kind::Text
    }
    fn describe(&self) -> &str {
        "spoken language identification (sherpa-onnx / Whisper)"
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

        let mut wav = Vec::new();
        input.read_to_end(&mut wav)?;
        if wav.is_empty() {
            return Err(ChordError::BadInput("no audio on input".to_string()).into());
        }
        let (samples, src_rate) = decode_wav_to_mono_f32(&wav)?;
        let samples = resample_linear(&samples, src_rate, RATE);

        let config = SpokenLanguageIdConfig {
            encoder: encoder
                .to_str()
                .ok_or("encoder path not UTF-8")?
                .to_string(),
            decoder: decoder
                .to_str()
                .ok_or("decoder path not UTF-8")?
                .to_string(),
            debug: false,
            provider: None,
            num_threads: None,
        };
        let mut lid = SpokenLanguageId::new(config);
        let lang = lid
            .compute(samples, RATE)
            .map_err(|e| ChordError::Engine(format!("language id: {e}")))?;

        writeln!(output, "{}", lang.trim())?;
        Ok(())
    }
}

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
