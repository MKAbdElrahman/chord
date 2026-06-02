//! chord-stt-llama — `stt` (audio -> text) backed by **llama.cpp's `mtmd` audio**
//! path, i.e. audio-LLM ASR. This is the same multimodal pipeline as `see`, but
//! the media is audio (`MtmdBitmap::from_audio_data`) instead of an image.
//!
//! It lets chord run recent audio-LLM speech models (Voxtral, Ultravox,
//! Qwen-audio) that whisper.cpp can't, via the engine chord already links. Model
//! + audio projector are GGUF, resolvable as `hf:` references.

use std::io::{IsTerminal, Read, Write};
use std::num::NonZeroU32;
use std::time::Duration;

use chord_core::{ChordError, Kind, OptionSpec, Options, Result, Transform};
use indicatif::{ProgressBar, ProgressStyle};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::mtmd::{MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputText};
use llama_cpp_2::openai::OpenAIChatTemplateParams;
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::{send_logs_to_tracing, LogOptions};

// Default model: Mistral's Voxtral-Mini-3B (audio LLM) + its audio projector.
const DEFAULT_MODEL: &str =
    "hf:ggml-org/Voxtral-Mini-3B-2507-GGUF:Voxtral-Mini-3B-2507-Q4_K_M.gguf";
const DEFAULT_MMPROJ: &str =
    "hf:ggml-org/Voxtral-Mini-3B-2507-GGUF:mmproj-Voxtral-Mini-3B-2507-Q8_0.gguf";

const OPTS: &[OptionSpec] = &[
    OptionSpec {
        key: "model",
        help: "audio-LLM GGUF (path or hf: ref)",
        takes_value: true,
    },
    OptionSpec {
        key: "mmproj",
        help: "audio projector GGUF (path or hf: ref)",
        takes_value: true,
    },
    OptionSpec {
        key: "prompt",
        help: "instruction (default: transcribe verbatim)",
        takes_value: true,
    },
    OptionSpec {
        key: "max_tokens",
        help: "max output tokens (default 448)",
        takes_value: true,
    },
    OptionSpec {
        key: "temperature",
        help: "sampling temperature; 0 = greedy (default 0)",
        takes_value: true,
    },
    OptionSpec {
        key: "n_ctx",
        help: "context window tokens (default 4096)",
        takes_value: true,
    },
    OptionSpec {
        key: "threads",
        help: "CPU threads (default 4)",
        takes_value: true,
    },
];

pub struct SttLlama;

impl Transform for SttLlama {
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
        "speech-to-text via audio-LLM (llama.cpp mtmd)"
    }
    fn backend(&self) -> &str {
        "llama.cpp"
    }
    fn options(&self) -> &'static [OptionSpec] {
        OPTS
    }

    fn apply(&self, input: &mut dyn Read, output: &mut dyn Write, opts: &Options) -> Result<()> {
        send_logs_to_tracing(LogOptions::default().with_logs_enabled(false));
        unsafe {
            llama_cpp_sys_2::mtmd_helper_log_set(Some(silent_mtmd_log), std::ptr::null_mut())
        };

        let model_path = chord_hf::resolve(opts.get_or("model", DEFAULT_MODEL))?;
        let mmproj_path = chord_hf::resolve(opts.get_or("mmproj", DEFAULT_MMPROJ))?;
        let prompt = opts
            .get_or(
                "prompt",
                "Transcribe the audio verbatim, with no extra commentary.",
            )
            .to_string();
        let max_tokens: i32 = opts.get_or("max_tokens", "448").parse().unwrap_or(448);
        let temperature: f32 = opts.get_or("temperature", "0").parse().unwrap_or(0.0);
        let n_ctx: u32 = opts.get_or("n_ctx", "4096").parse().unwrap_or(4096);
        let threads: i32 = opts.get_or("threads", "4").parse().unwrap_or(4);

        let mut wav = Vec::new();
        input.read_to_end(&mut wav)?;
        if wav.is_empty() {
            return Err(ChordError::BadInput("no audio on input".to_string()).into());
        }
        let (samples, src_rate) = decode_wav_to_mono_f32(&wav)?;

        let pb = spinner(&format!(
            "loading {}…",
            model_path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("model")
        ));

        let backend = LlamaBackend::init()?;
        let model = LlamaModel::load_from_file(
            &backend,
            &model_path,
            &LlamaModelParams::default().with_n_gpu_layers(0),
        )?;

        let mtmd_params = MtmdContextParams {
            use_gpu: false,
            print_timings: false,
            n_threads: threads,
            ..Default::default()
        };
        let marker = mtmd_params
            .media_marker
            .to_str()
            .unwrap_or("<__media__>")
            .to_string();
        let mtmd_ctx = MtmdContext::init_from_file(
            mmproj_path.to_str().ok_or("mmproj path not UTF-8")?,
            &model,
            &mtmd_params,
        )?;
        if !mtmd_ctx.support_audio() {
            return Err("model/mmproj does not support audio input".into());
        }

        // Resample to the rate the audio encoder expects (usually 16 kHz).
        let target_rate = mtmd_ctx.get_audio_sample_rate().unwrap_or(16_000);
        let samples = resample_linear(&samples, src_rate, target_rate);
        let bitmap = MtmdBitmap::from_audio_data(&samples)?;

        let user = format!("{marker}\n{prompt}");
        let messages_json =
            serde_json::Value::Array(vec![serde_json::json!({"role":"user","content":user})])
                .to_string();
        let template = model.chat_template(None)?;
        let tmpl_params = OpenAIChatTemplateParams {
            messages_json: &messages_json,
            tools_json: None,
            tool_choice: None,
            json_schema: None,
            grammar: None,
            reasoning_format: None,
            chat_template_kwargs: None,
            add_generation_prompt: true,
            use_jinja: true,
            parallel_tool_calls: false,
            enable_thinking: false,
            add_bos: false,
            add_eos: false,
            parse_tool_calls: false,
        };
        let rendered = model
            .apply_chat_template_oaicompat(&template, &tmpl_params)?
            .prompt;

        let chunks = mtmd_ctx.tokenize(
            MtmdInputText {
                text: rendered,
                add_special: true,
                parse_special: true,
            },
            &[&bitmap],
        )?;

        let ctx_params =
            LlamaContextParams::default().with_n_ctx(Some(NonZeroU32::new(n_ctx).unwrap()));
        let mut ctx = model.new_context(&backend, ctx_params)?;

        pb.set_message("encoding audio…");
        let n_past = chunks.eval_chunks(&mtmd_ctx, &ctx, 0, 0, 512, true)?;

        let mut sampler = if temperature <= 0.0 {
            LlamaSampler::greedy()
        } else {
            LlamaSampler::chain_simple([
                LlamaSampler::top_k(40),
                LlamaSampler::top_p(0.95, 1),
                LlamaSampler::temp(temperature),
                LlamaSampler::dist(1234),
            ])
        };

        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut answer = String::new();
        let mut batch = LlamaBatch::new(512, 1);
        let mut n_cur = n_past;
        let mut produced = 0;

        pb.set_message("transcribing…");
        let mut token = sampler.sample(&ctx, -1);
        while produced < max_tokens {
            sampler.accept(token);
            if model.is_eog_token(token) {
                break;
            }
            answer.push_str(&model.token_to_piece(token, &mut decoder, false, None)?);
            batch.clear();
            batch.add(token, n_cur, &[0], true)?;
            n_cur += 1;
            produced += 1;
            pb.set_message(format!("{produced} tok │ …{}", rolling_tail(&answer, 64)));
            ctx.decode(&mut batch)?;
            token = sampler.sample(&ctx, batch.n_tokens() - 1);
        }
        pb.finish_and_clear();

        writeln!(output, "{}", answer.trim())?;
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

/// Linear-interpolation resampler (the audio encoder is robust to it).
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

unsafe extern "C" fn silent_mtmd_log(
    _level: llama_cpp_sys_2::ggml_log_level,
    _text: *const std::os::raw::c_char,
    _user_data: *mut std::os::raw::c_void,
) {
}

fn rolling_tail(s: &str, max: usize) -> String {
    let flat: String = s.chars().map(|c| if c == '\n' { ' ' } else { c }).collect();
    let n = flat.chars().count();
    flat.chars().skip(n.saturating_sub(max)).collect()
}

fn spinner(msg: &str) -> ProgressBar {
    if std::io::stderr().is_terminal() {
        let pb = ProgressBar::new_spinner();
        pb.set_style(ProgressStyle::with_template("{spinner:.cyan} {msg}").unwrap());
        pb.enable_steady_tick(Duration::from_millis(120));
        pb.set_message(msg.to_string());
        pb
    } else {
        ProgressBar::hidden()
    }
}
