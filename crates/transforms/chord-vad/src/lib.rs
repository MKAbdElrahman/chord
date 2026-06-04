//! chord-vad — voice activity detection (`audio -> text`): emits one
//! `START END` (seconds) line per detected speech segment, via sherpa-onnx's
//! Silero VAD. An atomic building block — compose it yourself, e.g. slice each
//! segment out and run `stt`/`langid` on it.

use std::io::{Read, Write};

use chord_core::{ChordError, Kind, OptionSpec, Options, Result, Unary};
use sherpa_rs::silero_vad::{SileroVad, SileroVadConfig};

const DEF_MODEL: &str = "hf:csukuangfj/vad:silero_vad.onnx";
const RATE: u32 = 16_000;

const OPTS: &[OptionSpec] = &[
    OptionSpec {
        key: "model",
        help: "Silero VAD ONNX (path or hf: ref)",
        takes_value: true,
    },
    OptionSpec {
        key: "threshold",
        help: "speech probability threshold (default 0.5)",
        takes_value: true,
    },
    OptionSpec {
        key: "min_silence",
        help: "min silence seconds to split (default 0.5)",
        takes_value: true,
    },
    OptionSpec {
        key: "min_speech",
        help: "min speech seconds to keep (default 0.25)",
        takes_value: true,
    },
    OptionSpec {
        key: "max_speech",
        help: "max speech seconds per segment (default 30)",
        takes_value: true,
    },
];

pub struct Vad;

impl Unary for Vad {
    fn name(&self) -> &str {
        "vad"
    }
    fn from(&self) -> Kind {
        Kind::Audio
    }
    fn to(&self) -> Kind {
        Kind::Text
    }
    fn describe(&self) -> &str {
        "voice activity detection: speech segment timestamps (sherpa-onnx / Silero)"
    }
    fn backend(&self) -> &str {
        "sherpa-onnx"
    }
    fn options(&self) -> &'static [OptionSpec] {
        OPTS
    }

    fn apply(&self, input: &mut dyn Read, output: &mut dyn Write, opts: &Options) -> Result<()> {
        let model = chord_hf::resolve(opts.get_or("model", DEF_MODEL))?;
        let config = SileroVadConfig {
            model: model.to_str().ok_or("model path not UTF-8")?.to_string(),
            threshold: opts.get_or("threshold", "0.5").parse().unwrap_or(0.5),
            min_silence_duration: opts.get_or("min_silence", "0.5").parse().unwrap_or(0.5),
            min_speech_duration: opts.get_or("min_speech", "0.25").parse().unwrap_or(0.25),
            max_speech_duration: opts.get_or("max_speech", "30").parse().unwrap_or(30.0),
            sample_rate: RATE,
            window_size: 512,
            provider: None,
            num_threads: Some(1),
            debug: false,
        };

        let mut wav = Vec::new();
        input.read_to_end(&mut wav)?;
        if wav.is_empty() {
            return Err(ChordError::BadInput("no audio on input".to_string()).into());
        }
        let (samples, src_rate) = decode_wav_to_mono_f32(&wav)?;
        let samples = resample_linear(&samples, src_rate, RATE);

        let mut vad = SileroVad::new(config, 60.0)
            .map_err(|e| ChordError::Engine(format!("silero vad: {e}")))?;

        let mut emit = |vad: &mut SileroVad| -> Result<()> {
            while !vad.is_empty() {
                let seg = vad.front();
                let start = seg.start as f64 / RATE as f64;
                let end = (seg.start as usize + seg.samples.len()) as f64 / RATE as f64;
                writeln!(output, "{start:.3} {end:.3}")?;
                vad.pop();
            }
            Ok(())
        };

        // Feed in windows and drain completed segments as they appear — Silero
        // is a streaming detector; feeding everything at once loses segments.
        let win = 512;
        let mut i = 0;
        while i < samples.len() {
            let end = (i + win).min(samples.len());
            vad.accept_waveform(samples[i..end].to_vec());
            emit(&mut vad)?;
            i = end;
        }
        vad.flush();
        emit(&mut vad)?;
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
