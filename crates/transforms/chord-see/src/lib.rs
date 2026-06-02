//! chord-see — vision plug-in (`image -> text`).
//!
//! Backed by llama.cpp's multimodal stack (`mtmd` + CLIP) through `llama-cpp-2`.
//! It loads a vision-language model plus its `mmproj` projector, feeds the image
//! and a prompt through the model, and writes the model's description to stdout.
//!
//! Options (config or `-o`):
//! - `model` — VLM GGUF path (default gemma-4-26B under ~/.kronk/models, or $CHORD_SEE_MODEL)
//! - `mmproj` — multimodal projector GGUF (default: the model's sibling mmproj)
//! - `prompt` — instruction (default "Describe this image.")
//! - `max_tokens` (default 256), `temperature` (default 0.3), `n_ctx` (default 4096)

use std::io::{IsTerminal, Read, Write};
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::time::Duration;

use chord_core::{Kind, OptionSpec, Options, Registry, Result, Transform};
use indicatif::{ProgressBar, ProgressStyle};

const OPTS: &[OptionSpec] = &[
    OptionSpec { key: "prompt", help: "instruction/question about the image", takes_value: true },
    OptionSpec { key: "model", help: "vision GGUF model path", takes_value: true },
    OptionSpec { key: "mmproj", help: "multimodal projector GGUF path", takes_value: true },
    OptionSpec { key: "max_tokens", help: "max reply tokens (default 256)", takes_value: true },
    OptionSpec { key: "temperature", help: "sampling temperature (default 0.3)", takes_value: true },
    OptionSpec { key: "n_ctx", help: "context window tokens (default 4096)", takes_value: true },
];
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::mtmd::{MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputText};
use llama_cpp_2::openai::OpenAIChatTemplateParams;
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::{send_logs_to_tracing, LogOptions};

pub fn register(reg: &mut Registry) {
    reg.register(Box::new(See));
}

struct See;

impl Transform for See {
    fn name(&self) -> &str {
        "see"
    }
    fn from(&self) -> Kind {
        Kind::Image
    }
    fn to(&self) -> Kind {
        Kind::Text
    }
    fn describe(&self) -> &str {
        "vision: describe/read an image (llama.cpp mtmd)"
    }
    fn options(&self) -> &'static [OptionSpec] {
        OPTS
    }

    fn apply(&self, input: &mut dyn Read, output: &mut dyn Write, opts: &Options) -> Result<()> {
        // Silence llama/ggml logs AND the CLIP/mtmd logger (separate channel —
        // it has its own callback, so send_logs_to_tracing alone misses the
        // "clip_model_loader: tensor[...]" dump). The spinner still draws.
        send_logs_to_tracing(LogOptions::default().with_logs_enabled(false));
        // mtmd_helper_log_set silences the mtmd-helper logger (image encode/decode
        // progress) AND calls mtmd_log_set internally (the CLIP loader dump).
        unsafe { llama_cpp_sys_2::mtmd_helper_log_set(Some(silent_mtmd_log), std::ptr::null_mut()) };

        let (model_path, mmproj_path) = resolve_models(opts)?;
        let prompt = opts.get_or("prompt", "Describe this image.").to_string();
        let max_tokens: i32 = opts.get_or("max_tokens", "256").parse().unwrap_or(256);
        let temperature: f32 = opts.get_or("temperature", "0.3").parse().unwrap_or(0.3);
        let n_ctx: u32 = opts.get_or("n_ctx", "4096").parse().unwrap_or(4096);

        let mut image_bytes = Vec::new();
        input.read_to_end(&mut image_bytes)?;
        if image_bytes.is_empty() {
            return Err("no image on input".into());
        }

        let pb = spinner(&format!(
            "loading {}…",
            model_path.file_name().and_then(|s| s.to_str()).unwrap_or("model")
        ));

        let backend = LlamaBackend::init()?;
        let model = LlamaModel::load_from_file(
            &backend,
            &model_path,
            &LlamaModelParams::default().with_n_gpu_layers(0),
        )?;

        // Multimodal context (CLIP projector + text model).
        let mut mtmd_params = MtmdContextParams::default();
        mtmd_params.use_gpu = false;
        mtmd_params.print_timings = false;
        mtmd_params.n_threads = opts.get_or("threads", "4").parse().unwrap_or(4);
        let marker = mtmd_params.media_marker.to_str().unwrap_or("<__media__>").to_string();
        let mtmd_ctx = MtmdContext::init_from_file(
            mmproj_path.to_str().ok_or("mmproj path not UTF-8")?,
            &model,
            &mtmd_params,
        )?;
        if !mtmd_ctx.support_vision() {
            return Err("model/mmproj does not support vision input".into());
        }

        let bitmap = MtmdBitmap::from_buffer(&mtmd_ctx, &image_bytes)?;

        // Render the chat prompt with the media marker in the user turn, then
        // let mtmd tokenize text + image into chunks.
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
        let rendered = model.apply_chat_template_oaicompat(&template, &tmpl_params)?.prompt;

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

        pb.set_message("encoding image…");
        // Evaluate text+image chunks into the context; logits on the last token.
        let n_past = chunks.eval_chunks(&mtmd_ctx, &ctx, 0, 0, 512, true)?;

        // Generate the description.
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

        pb.set_message("describing…");
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

/// Resolve (model, mmproj) GGUF paths. Defaults to gemma-4-26B + its sibling
/// mmproj under ~/.kronk/models; override with `model`/`mmproj` or
/// $CHORD_SEE_MODEL / $CHORD_SEE_MMPROJ.
fn resolve_models(opts: &Options) -> Result<(PathBuf, PathBuf)> {
    let home = std::env::var("HOME").unwrap_or_default();
    let dir = PathBuf::from(&home).join(".kronk/models/unsloth/gemma-4-26B-A4B-it-GGUF");

    let model = opts
        .get("model")
        .map(PathBuf::from)
        .or_else(|| std::env::var("CHORD_SEE_MODEL").ok().map(PathBuf::from))
        .unwrap_or_else(|| dir.join("gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"));
    let mmproj = opts
        .get("mmproj")
        .map(PathBuf::from)
        .or_else(|| std::env::var("CHORD_SEE_MMPROJ").ok().map(PathBuf::from))
        .unwrap_or_else(|| dir.join("mmproj-gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"));

    if !model.exists() {
        return Err(format!("vision model not found: {} (set -o model=…)", model.display()).into());
    }
    if !mmproj.exists() {
        return Err(format!("mmproj not found: {} (set -o mmproj=…)", mmproj.display()).into());
    }
    Ok((model, mmproj))
}

/// No-op mtmd/CLIP log callback — drops the vision-loader's stderr output.
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
