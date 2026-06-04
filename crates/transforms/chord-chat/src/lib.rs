//! chord-chat — chat / text-generation plug-in (`text -> text`).
//!
//! Backed by llama.cpp through the `llama-cpp-2` binding crate — the LLM
//! sibling of `chord-stt`'s whisper.cpp binding. Fully local and in-process:
//! it loads a GGUF model, applies the model's own chat template to a
//! (system + user) conversation, generates a reply, and writes it to stdout.
//!
//! Options (via config or `-o`):
//! - `model` — GGUF path (default `$CHORD_CHAT_MODEL` or the Qwen3-30B-A3B GGUF under the XDG models dir)
//! - `system` — optional system prompt
//! - `max_tokens` — reply length cap (default 512)
//! - `temperature` — sampling temperature; 0 = greedy (default 0.7)
//! - `n_ctx` — context window in tokens (default 4096)
//! - `think` — enable reasoning/thinking via the chat template's
//!   `enable_thinking` (default false; set `true` for reasoning models)

use std::io::{IsTerminal, Read, Write};
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::time::Duration;

use chord_core::{ChordError, Kind, OptionSpec, Options, Result, Transform};
use indicatif::{ProgressBar, ProgressStyle};

const OPTS: &[OptionSpec] = &[
    OptionSpec {
        key: "model",
        help: "GGUF model path or name",
        takes_value: true,
    },
    OptionSpec {
        key: "system",
        help: "system prompt",
        takes_value: true,
    },
    OptionSpec {
        key: "think",
        help: "enable reasoning (default off)",
        takes_value: false,
    },
    OptionSpec {
        key: "max_tokens",
        help: "max reply tokens (default 512)",
        takes_value: true,
    },
    OptionSpec {
        key: "temperature",
        help: "sampling temperature (default 0.7)",
        takes_value: true,
    },
    OptionSpec {
        key: "n_ctx",
        help: "context window tokens (default 4096)",
        takes_value: true,
    },
];
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::openai::OpenAIChatTemplateParams;
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::{send_logs_to_tracing, LogOptions};

pub struct Chat;

impl Transform for Chat {
    fn name(&self) -> &str {
        "chat"
    }
    fn from(&self) -> Kind {
        Kind::Text
    }
    fn to(&self) -> Kind {
        Kind::Text
    }
    fn describe(&self) -> &str {
        "chat / text generation (llama.cpp)"
    }
    fn backend(&self) -> &str {
        "llama.cpp"
    }
    fn options(&self) -> &'static [OptionSpec] {
        OPTS
    }

    fn apply(&self, input: &mut dyn Read, output: &mut dyn Write, opts: &Options) -> Result<()> {
        let mut user = String::new();
        input.read_to_string(&mut user)?;
        let user = user.trim();
        if user.is_empty() {
            return Err(ChordError::BadInput("no input prompt".to_string()).into());
        }

        // Silence llama.cpp / ggml stderr logging (model-load "repack" spam etc.).
        send_logs_to_tracing(LogOptions::default().with_logs_enabled(false));

        let model_path = resolve_model(opts)?;
        let max_tokens: i32 = opts.get_or("max_tokens", "512").parse().unwrap_or(512);
        let temperature: f32 = opts.get_or("temperature", "0.7").parse().unwrap_or(0.7);
        let n_ctx: u32 = opts.get_or("n_ctx", "4096").parse().unwrap_or(4096);

        let model_label = model_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("model");
        let pb = spinner(&format!("loading {model_label}…"));

        let backend = LlamaBackend::init()?;

        // CPU inference (no GPU offload).
        let model_params = LlamaModelParams::default().with_n_gpu_layers(0);
        let model = LlamaModel::load_from_file(&backend, &model_path, &model_params)?;

        // Reasoning ("thinking") is controlled the proper llama.cpp way: the
        // model's Jinja chat template is rendered with enable_thinking, instead
        // of injecting model-specific tokens like Qwen's /no_think. Default off
        // for clean, fast pipeline output; `-o think=true` enables reasoning.
        let think: bool = opts.get_or("think", "false").parse().unwrap_or(false);

        // We pass enable_thinking through the template kwargs below (the
        // standard llama.cpp control). Some templates (e.g. Qwen3) only honor it
        // via their soft-switch token, so when thinking is off we also append
        // `/no_think` to the user turn — kept inside the engine so callers just
        // set `think`, never type model-specific tokens.
        let user_content = if think {
            user.to_string()
        } else {
            format!("{user} /no_think")
        };

        // OpenAI-format messages JSON (serde_json handles escaping).
        let mut msgs: Vec<serde_json::Value> = Vec::new();
        if let Some(system) = opts.get("system") {
            msgs.push(serde_json::json!({ "role": "system", "content": system }));
        }
        msgs.push(serde_json::json!({ "role": "user", "content": user_content }));
        let messages_json = serde_json::Value::Array(msgs).to_string();

        // Pass enable_thinking into the Jinja template via chat_template_kwargs
        // (the channel the template variable actually reads — same as
        // llama.cpp's `--chat-template-kwargs '{"enable_thinking":false}'`).
        let kwargs = format!("{{\"enable_thinking\": {think}}}");

        let template = model.chat_template(None)?;
        let params = OpenAIChatTemplateParams {
            messages_json: &messages_json,
            tools_json: None,
            tool_choice: None,
            json_schema: None,
            grammar: None,
            // "auto" makes the parser extract <think>…</think> into
            // reasoning_content instead of leaving it inline in content.
            reasoning_format: Some("auto"),
            chat_template_kwargs: Some(&kwargs),
            add_generation_prompt: true,
            use_jinja: true,
            parallel_tool_calls: false,
            enable_thinking: think,
            add_bos: false,
            add_eos: false,
            parse_tool_calls: false,
        };
        let rendered = model.apply_chat_template_oaicompat(&template, &params)?;
        let prompt = rendered.prompt.clone();
        // llama.cpp's model-aware parser: separates `content` (the answer) from
        // `reasoning_content` (thinking) in the generated text — the same split
        // llama-server exposes. We emit only content to stdout.
        let mut parser = rendered.streaming_state_oaicompat()?;

        let ctx_params =
            LlamaContextParams::default().with_n_ctx(Some(NonZeroU32::new(n_ctx).unwrap()));
        let mut ctx = model.new_context(&backend, ctx_params)?;

        let tokens = model.str_to_token(&prompt, AddBos::Always)?;

        let mut batch = LlamaBatch::new(512.max(tokens.len()), 1);
        let last = (tokens.len() - 1) as i32;
        for (i, token) in (0_i32..).zip(tokens) {
            batch.add(token, i, &[0], i == last)?;
        }
        ctx.decode(&mut batch)?;

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
        let mut answer = String::new(); // content only -> stdout
        let mut preview = String::new(); // reasoning + content -> spinner
        let mut n_cur = batch.n_tokens();
        let mut produced = 0;

        pb.set_message("generating…");
        while produced < max_tokens {
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(token);
            if model.is_eog_token(token) {
                break;
            }
            let piece = model.token_to_piece(token, &mut decoder, false, None)?;
            for delta in parser.update(&piece, true)? {
                let (content, reasoning) = split_delta(&delta);
                answer.push_str(&content);
                preview.push_str(&reasoning);
                preview.push_str(&content);
            }

            batch.clear();
            batch.add(token, n_cur, &[0], true)?;
            n_cur += 1;
            produced += 1;
            // Live preview shows reasoning+answer streaming (stderr, TTY only);
            // stdout gets only the parsed answer.
            pb.set_message(format!("{produced} tok │ …{}", rolling_tail(&preview, 64)));
            ctx.decode(&mut batch)?;
        }
        // Finalize the parser (flush any buffered content).
        for delta in parser.update("", false)? {
            let (content, _) = split_delta(&delta);
            answer.push_str(&content);
        }
        pb.finish_and_clear();

        writeln!(output, "{}", answer.trim())?;
        Ok(())
    }
}

/// Extract `(content, reasoning_content)` from one OpenAI-compatible delta JSON
/// emitted by the chat parser. Handles both a flat delta object and one nested
/// under `choices[0].delta`.
fn split_delta(json: &str) -> (String, String) {
    let v: serde_json::Value = serde_json::from_str(json).unwrap_or(serde_json::Value::Null);
    let delta = v
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("delta"))
        .unwrap_or(&v);
    let field = |k: &str| {
        delta
            .get(k)
            .and_then(serde_json::Value::as_str)
            .unwrap_or("")
            .to_string()
    };
    (field("content"), field("reasoning_content"))
}

/// The last `max` characters of `s`, with newlines flattened to spaces, for a
/// single-line streaming preview.
fn rolling_tail(s: &str, max: usize) -> String {
    let flat: String = s.chars().map(|c| if c == '\n' { ' ' } else { c }).collect();
    let n = flat.chars().count();
    flat.chars().skip(n.saturating_sub(max)).collect()
}

/// A stderr spinner that only renders on an interactive terminal, so piped or
/// redirected output (the data plane) is never touched. Auto-ticks on its own
/// thread; clear it with `finish_and_clear`.
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

/// Resolve the GGUF model path: `model` option, else `$CHORD_CHAT_MODEL`, else
/// the Qwen3-30B-A3B GGUF under the XDG models dir.
fn resolve_model(opts: &Options) -> Result<PathBuf> {
    if let Some(m) = opts.get("model") {
        return chord_hf::resolve(m);
    }
    if let Ok(m) = std::env::var("CHORD_CHAT_MODEL") {
        if !m.is_empty() {
            return chord_hf::resolve(&m);
        }
    }
    let default = chord_core::dirs::models_dir()
        .join("unsloth/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf");
    if default.exists() {
        return Ok(default);
    }
    Err(ChordError::ModelMissing {
        what: "chat model".to_string(),
        hint: "run `chord pull chat`, or set --model to a GGUF path (or $CHORD_CHAT_MODEL)"
            .to_string(),
    }
    .into())
}
