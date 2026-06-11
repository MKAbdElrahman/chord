//! chord-chat — the multimodal chat plug-in (`text,image,audio -> text`).
//!
//! This is the general case of the chord [`Transform`] contract: an ordered
//! [`Message`] of typed parts in, a `Message` out. It collects every text,
//! image, and audio part (whether piped in as a framed message, e.g. from
//! `chord pack`, or supplied via the `--image`/`--audio` convenience flags),
//! hands them to a multimodal model through mistral.rs, and emits the reply as
//! a single text part.
//!
//! The mono-modal `text` engine (llama.cpp) is the text-only restriction of
//! this; `chat` is where the full multimodality lives.

use std::io::{Read, Write};

use chord_core::{
    ChordError, Kind, Message, OptionSpec, Options, Part, Result, Signature, Transform,
};
use mistralrs::{
    AudioInput, IsqBits, ModelBuilder, MultimodalMessages, PagedAttentionMetaBuilder, Response,
    TextMessageRole,
};

/// Default model: Gemma 4 E4B, the small multimodal (text+image+audio) model.
/// Override with `--model <hf-id-or-path>`.
const DEFAULT_MODEL: &str = "google/gemma-4-E4B-it";

const OPTS: &[OptionSpec] = &[
    OptionSpec {
        key: "model",
        help: "model: HF repo id or local path (default google/gemma-4-E4B-it)",
        takes_value: true,
    },
    OptionSpec {
        key: "system",
        help: "system prompt",
        takes_value: true,
    },
    OptionSpec {
        key: "image",
        help: "image file to include in the prompt (path)",
        takes_value: true,
    },
    OptionSpec {
        key: "audio",
        help: "audio file (wav) to include in the prompt (path)",
        takes_value: true,
    },
    OptionSpec {
        key: "think",
        help: "enable reasoning/thinking (default off)",
        takes_value: false,
    },
    OptionSpec {
        key: "mtp_model",
        help: "MTP assistant model (HF id or path) for speculative decoding; requires PagedAttention (GPU)",
        takes_value: true,
    },
    OptionSpec {
        key: "mtp_n_predict",
        help: "tokens the MTP assistant proposes per step (default: assistant config, else 6)",
        takes_value: true,
    },
];

pub struct Chat;

impl Transform for Chat {
    fn name(&self) -> &str {
        "chat"
    }

    fn signature(&self) -> Signature {
        Signature::new(vec![Kind::Text, Kind::Image, Kind::Audio], vec![Kind::Text])
    }

    fn describe(&self) -> &str {
        "multimodal chat (mistral.rs): text/image/audio -> text"
    }

    fn backend(&self) -> &str {
        "mistral.rs"
    }

    fn options(&self) -> &'static [OptionSpec] {
        OPTS
    }

    fn apply(&self, input: Message, opts: &Options) -> Result<Message> {
        // Collect the prompt: text parts are concatenated in order; image/audio
        // parts and the --image/--audio flag files become model inputs.
        let mut text = String::new();
        let mut images = Vec::new();
        let mut audios = Vec::new();

        for part in &input.parts {
            match part.kind {
                Kind::Text => {
                    let s = String::from_utf8_lossy(part.as_bytes());
                    let s = s.trim();
                    if !s.is_empty() {
                        if !text.is_empty() {
                            text.push('\n');
                        }
                        text.push_str(s);
                    }
                }
                Kind::Image => images.push(decode_image(part.as_bytes())?),
                Kind::Audio => audios.push(decode_audio(part.as_bytes())?),
                Kind::Video => {
                    return Err(ChordError::BadInput(
                        "video input is not yet supported by chord chat".into(),
                    )
                    .into())
                }
            }
        }

        if let Some(path) = opts.get("image") {
            images.push(decode_image(&read_file(path)?)?);
        }
        if let Some(path) = opts.get("audio") {
            audios.push(decode_audio(&read_file(path)?)?);
        }

        if text.is_empty() && images.is_empty() && audios.is_empty() {
            return Err(ChordError::BadInput(
                "no input: provide a prompt and/or --image/--audio".into(),
            )
            .into());
        }

        let cfg = ChatCfg::from(opts);
        let mut reply = String::new();
        run(cfg, text, images, audios, |tok| {
            reply.push_str(tok);
            Ok(())
        })?;
        Ok(Message::one(Part::text(reply)))
    }

    /// Raw-singleton fast path: the input stream is the text prompt, and each
    /// generated token is written (and flushed) the moment the model emits it
    /// — so `chord chat | chord tts --chunk 80 | aplay` speaks the first
    /// sentence while the model is still generating (rule R2, engine half).
    fn apply_raw(
        &self,
        input: &mut dyn Read,
        output: &mut dyn Write,
        opts: &Options,
    ) -> Result<()> {
        let mut text = String::new();
        input.read_to_string(&mut text)?;
        let text = text.trim().to_string();

        let mut images = Vec::new();
        let mut audios = Vec::new();
        if let Some(path) = opts.get("image") {
            images.push(decode_image(&read_file(path)?)?);
        }
        if let Some(path) = opts.get("audio") {
            audios.push(decode_audio(&read_file(path)?)?);
        }
        if text.is_empty() && images.is_empty() && audios.is_empty() {
            return Err(ChordError::BadInput(
                "no input: provide a prompt and/or --image/--audio".into(),
            )
            .into());
        }

        run(ChatCfg::from(opts), text, images, audios, |tok| {
            output.write_all(tok.as_bytes())?;
            output.flush()?;
            Ok(())
        })
    }
}

/// The option-derived knobs of one chat turn.
struct ChatCfg {
    model_id: String,
    system: Option<String>,
    think: bool,
    mtp: Option<Mtp>,
}

impl ChatCfg {
    fn from(opts: &Options) -> Self {
        ChatCfg {
            model_id: opts.get("model").unwrap_or(DEFAULT_MODEL).to_string(),
            system: opts.get("system").map(str::to_string),
            think: opts.get("think").is_some(),
            // Optional MTP (multi-token-prediction) speculative decoding drafter.
            mtp: opts.get("mtp_model").map(|m| Mtp {
                model: m.to_string(),
                n_predict: opts.get("mtp_n_predict").and_then(|s| s.parse().ok()),
            }),
        }
    }
}

/// An MTP (multi-token-prediction) speculative-decoding drafter.
struct Mtp {
    model: String,
    n_predict: Option<usize>,
}

/// Run one multimodal turn, streaming: each token delta the model emits is
/// handed to `on_token` as it arrives. Both apply paths share this core —
/// buffered `apply` collects the deltas, `apply_raw` pipes them out live.
/// (Blocks on a private tokio runtime; the engine binary is synchronous.)
fn run<F>(
    cfg: ChatCfg,
    text: String,
    images: Vec<image::DynamicImage>,
    audios: Vec<AudioInput>,
    mut on_token: F,
) -> Result<()>
where
    F: FnMut(&str) -> Result<()>,
{
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| ChordError::Engine(format!("tokio runtime: {e}")))?;

    rt.block_on(async move {
        let mut builder = ModelBuilder::new(&cfg.model_id).with_auto_isq(IsqBits::Four);
        // MTP speculative decoding: a drafter proposes several tokens that the
        // target verifies. It requires PagedAttention on the target, so enable
        // that alongside it (only when MTP is requested).
        if let Some(mtp) = cfg.mtp {
            builder = builder
                .with_mtp_model(mtp.model, mtp.n_predict)
                .with_paged_attn(PagedAttentionMetaBuilder::default().build().map_err(|e| {
                    ChordError::Engine(format!("paged attention (required for MTP): {e}"))
                })?);
        }
        let model = builder
            .build()
            .await
            .map_err(|e| ChordError::Engine(format!("loading model {}: {e}", cfg.model_id)))?;

        let mut msgs = MultimodalMessages::new();
        if let Some(sys) = cfg.system {
            msgs = msgs.add_message(TextMessageRole::System, sys);
        }
        // One user turn carrying the text plus all images and audio. mistral.rs
        // applies the model's own layout/prefix-token rules per modality.
        msgs = msgs
            .add_multimodal_message(TextMessageRole::User, text, images, audios, vec![])
            .enable_thinking(cfg.think);

        let mut stream = model
            .stream_chat_request(msgs)
            .await
            .map_err(|e| ChordError::Engine(format!("inference: {e}")))?;

        while let Some(resp) = stream.next().await {
            match resp {
                Response::Chunk(chunk) => {
                    for choice in &chunk.choices {
                        if let Some(content) = &choice.delta.content {
                            on_token(content)?;
                        }
                    }
                }
                Response::Done(_) => break,
                Response::ModelError(msg, _) => {
                    return Err(ChordError::Engine(format!("inference: {msg}")).into());
                }
                Response::InternalError(e) | Response::ValidationError(e) => {
                    return Err(ChordError::Engine(format!("inference: {e}")).into());
                }
                _ => {}
            }
        }
        Ok::<(), chord_core::Error>(())
    })
}

fn read_file(path: &str) -> Result<Vec<u8>> {
    std::fs::read(path).map_err(|e| ChordError::BadInput(format!("reading {path}: {e}")).into())
}

fn decode_image(bytes: &[u8]) -> Result<image::DynamicImage> {
    image::load_from_memory(bytes)
        .map_err(|e| ChordError::BadInput(format!("decoding image: {e}")).into())
}

fn decode_audio(bytes: &[u8]) -> Result<AudioInput> {
    AudioInput::from_bytes(bytes)
        .map_err(|e| ChordError::BadInput(format!("decoding audio: {e}")).into())
}
