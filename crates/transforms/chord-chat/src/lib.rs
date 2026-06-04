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

use chord_core::{
    ChordError, Kind, Message, OptionSpec, Options, Part, Result, Signature, Transform,
};
use mistralrs::{
    AudioInput, IsqBits, ModelBuilder, MultimodalMessages, PagedAttentionMetaBuilder,
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

        let model_id = opts.get("model").unwrap_or(DEFAULT_MODEL).to_string();
        let system = opts.get("system").map(str::to_string);
        let think = opts.get("think").is_some();
        // Optional MTP (multi-token-prediction) speculative decoding drafter.
        let mtp = opts.get("mtp_model").map(|m| Mtp {
            model: m.to_string(),
            n_predict: opts.get("mtp_n_predict").and_then(|s| s.parse().ok()),
        });

        let reply = run(model_id, system, text, images, audios, think, mtp)?;
        Ok(Message::one(Part::text(reply)))
    }
}

/// An MTP (multi-token-prediction) speculative-decoding drafter.
struct Mtp {
    model: String,
    n_predict: Option<usize>,
}

/// Run one multimodal turn synchronously (block on a private tokio runtime).
fn run(
    model_id: String,
    system: Option<String>,
    text: String,
    images: Vec<image::DynamicImage>,
    audios: Vec<AudioInput>,
    think: bool,
    mtp: Option<Mtp>,
) -> Result<String> {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| ChordError::Engine(format!("tokio runtime: {e}")))?;

    rt.block_on(async move {
        let mut builder = ModelBuilder::new(&model_id).with_auto_isq(IsqBits::Four);
        // MTP speculative decoding: a drafter proposes several tokens that the
        // target verifies. It requires PagedAttention on the target, so enable
        // that alongside it (only when MTP is requested).
        if let Some(mtp) = mtp {
            builder = builder
                .with_mtp_model(mtp.model, mtp.n_predict)
                .with_paged_attn(PagedAttentionMetaBuilder::default().build().map_err(|e| {
                    ChordError::Engine(format!("paged attention (required for MTP): {e}"))
                })?);
        }
        let model = builder
            .build()
            .await
            .map_err(|e| ChordError::Engine(format!("loading model {model_id}: {e}")))?;

        let mut msgs = MultimodalMessages::new();
        if let Some(sys) = system {
            msgs = msgs.add_message(TextMessageRole::System, sys);
        }
        // One user turn carrying the text plus all images and audio. mistral.rs
        // applies the model's own layout/prefix-token rules per modality.
        msgs = msgs
            .add_multimodal_message(TextMessageRole::User, text, images, audios, vec![])
            .enable_thinking(think);

        let resp = model
            .send_chat_request(msgs)
            .await
            .map_err(|e| ChordError::Engine(format!("inference: {e}")))?;

        let content = resp
            .choices
            .first()
            .and_then(|c| c.message.content.clone())
            .unwrap_or_default();
        Ok::<String, chord_core::Error>(content)
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
