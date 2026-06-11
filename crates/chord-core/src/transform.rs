use std::collections::HashMap;
use std::io::{Cursor, Read, Write};

use crate::message::{Message, Part};
use crate::{Kind, Result};

/// Per-transform configuration (e.g. `voice=M1`, `model=tiny`, `lang=de`).
///
/// The kernel passes this through untyped; each plug-in reads only the keys it
/// understands and ignores the rest. This keeps the contract stable as engines
/// are added.
#[derive(Debug, Default, Clone)]
pub struct Options(HashMap<String, String>);

impl Options {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.0.insert(key.into(), value.into());
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.0.get(key).map(String::as_str)
    }

    /// Return the value for `key`, or `default` if absent.
    pub fn get_or<'a>(&'a self, key: &str, default: &'a str) -> &'a str {
        self.get(key).unwrap_or(default)
    }
}

/// Declares one option a transform accepts, so the host can expose it as a real
/// CLI flag (`--key value`, or a boolean `--key`) instead of `-o key=value`.
#[derive(Debug, Clone, Copy)]
pub struct OptionSpec {
    /// Option key — both the `--flag` name and the [`Options`] key the
    /// transform reads.
    pub key: &'static str,
    /// One-line help shown in `--help`.
    pub help: &'static str,
    /// `true`: takes a value (`--key value`); `false`: a boolean flag
    /// (`--key` sets the value to `"true"`).
    pub takes_value: bool,
}

/// The kinds a transform consumes and produces. Coarse on purpose: the kernel
/// type-checks on modality (`accepts ∩ next.emits`), while exact codecs/order
/// are an engine/adapter concern. A mono-modal transform has one of each.
#[derive(Debug, Clone)]
pub struct Signature {
    pub accepts: Vec<Kind>,
    pub emits: Vec<Kind>,
}

impl Signature {
    pub fn new(accepts: Vec<Kind>, emits: Vec<Kind>) -> Self {
        Signature { accepts, emits }
    }

    /// The 1→1 signature: one `from` part in, one `to` part out.
    pub fn unary(from: Kind, to: Kind) -> Self {
        Signature {
            accepts: vec![from],
            emits: vec![to],
        }
    }

    /// The primary input kind — the default a raw (unframed) input decodes to.
    pub fn primary_in(&self) -> Kind {
        self.accepts.first().copied().unwrap_or(Kind::Text)
    }

    /// The primary output kind — the default a raw (unframed) child output
    /// decodes to.
    pub fn primary_out(&self) -> Kind {
        self.emits.first().copied().unwrap_or(Kind::Text)
    }

    /// True if this transform can consume a `kind` input.
    pub fn accepts_kind(&self, kind: Kind) -> bool {
        self.accepts.contains(&kind)
    }

    /// True if at least one kind this transform emits is accepted by `next` —
    /// the modality check for adjacent pipeline stages. Possibility, not
    /// certainty: which kind actually flows is runtime data. An empty side
    /// (an engine that declared no kinds) is treated as unchecked.
    pub fn connects_to(&self, next: &Signature) -> bool {
        self.emits.is_empty()
            || next.accepts.is_empty()
            || self.emits.iter().any(|k| next.accepts_kind(*k))
    }

    /// Human rendering for `chord ls` / `--help`, e.g. `"text,image,audio -> text"`.
    pub fn display(&self) -> String {
        let join = |ks: &[Kind]| ks.iter().map(Kind::as_str).collect::<Vec<_>>().join(",");
        format!("{} -> {}", join(&self.accepts), join(&self.emits))
    }
}

/// A `Transform` is the chord plug-in contract in full generality: a function
/// from a [`Message`] (ordered typed parts) to a `Message`. Composition is the
/// shell pipe; the kernel does no routing.
///
/// Contract every implementation MUST honor:
/// 1. **Stateless across calls** — the value holds config, not per-request state.
/// 2. **Signature honesty** — [`signature`](Transform::signature) must be accurate.
/// 3. **Options are advisory** — ignore unknown keys; default missing ones.
///
/// Mono-modal engines should implement [`Unary`] instead — the blanket impl
/// below lifts them into this trait for free.
pub trait Transform: Send + Sync {
    /// Unique CLI verb for this plug-in (e.g. `"chat"`, `"stt"`).
    fn name(&self) -> &str;

    /// The kinds this transform consumes and produces.
    fn signature(&self) -> Signature;

    /// One-line human summary, shown by `chord ls`.
    fn describe(&self) -> &str;

    /// The inference backend this transform requires (e.g. `"whisper.cpp"`,
    /// `"llama.cpp"`, `"mistral.rs"`); empty for engine-less transforms.
    fn backend(&self) -> &str {
        ""
    }

    /// The options this transform accepts, so the host can render them as CLI
    /// flags. Defaults to none.
    fn options(&self) -> &'static [OptionSpec] {
        &[]
    }

    /// Transform an input message into an output message.
    fn apply(&self, input: Message, opts: &Options) -> Result<Message>;
}

/// A 1→1 transform: exactly one input part of one [`Kind`], exactly one output
/// part of another. This is the common, mono-modal case (`stt`, `tts`, `text`,
/// `see`, `draw`, `redact`, …); implementing it keeps the familiar
/// read-stream/write-stream body. The blanket impl below makes every `Unary`
/// a [`Transform`].
pub trait Unary: Send + Sync {
    /// Unique CLI verb for this plug-in.
    fn name(&self) -> &str;

    /// Input modality.
    fn from(&self) -> Kind;

    /// Output modality.
    fn to(&self) -> Kind;

    /// One-line human summary.
    fn describe(&self) -> &str;

    /// Inference backend; empty for engine-less transforms.
    fn backend(&self) -> &str {
        ""
    }

    /// Options this transform accepts.
    fn options(&self) -> &'static [OptionSpec] {
        &[]
    }

    /// Read the single input part's bytes, write the single output part's bytes.
    fn apply(&self, input: &mut dyn Read, output: &mut dyn Write, opts: &Options) -> Result<()>;
}

/// Every [`Unary`] is a [`Transform`]: take the one input part of `from()`,
/// run the stream-in/stream-out body into a buffer, and wrap the buffer as the
/// one output part of `to()`.
impl<T: Unary> Transform for T {
    fn name(&self) -> &str {
        Unary::name(self)
    }

    fn signature(&self) -> Signature {
        Signature::unary(Unary::from(self), Unary::to(self))
    }

    fn describe(&self) -> &str {
        Unary::describe(self)
    }

    fn backend(&self) -> &str {
        Unary::backend(self)
    }

    fn options(&self) -> &'static [OptionSpec] {
        Unary::options(self)
    }

    fn apply(&self, input: Message, opts: &Options) -> Result<Message> {
        let from = Unary::from(self);
        let to = Unary::to(self);
        let bytes = input.single(from)?.as_bytes().to_vec();
        let mut rd = Cursor::new(bytes);
        let mut out = Vec::new();
        Unary::apply(self, &mut rd, &mut out, opts)?;
        Ok(Message::one(Part::new(to, out)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mismatched_unary_stages_do_not_connect() {
        // draw (text -> image) piped into stt (audio -> text) can never work.
        let draw = Signature::unary(Kind::Text, Kind::Image);
        let stt = Signature::unary(Kind::Audio, Kind::Text);
        assert!(!draw.connects_to(&stt));
    }

    #[test]
    fn matching_unary_stages_connect() {
        // stt (audio -> text) into tts (text -> audio).
        let stt = Signature::unary(Kind::Audio, Kind::Text);
        let tts = Signature::unary(Kind::Text, Kind::Audio);
        assert!(stt.connects_to(&tts));
    }

    #[test]
    fn multimodal_consumer_connects_on_any_shared_kind() {
        // chat (text,image,audio -> text) into tts (text -> audio), and back:
        // tts emits audio, which chat also accepts.
        let chat = Signature::new(vec![Kind::Text, Kind::Image, Kind::Audio], vec![Kind::Text]);
        let tts = Signature::unary(Kind::Text, Kind::Audio);
        assert!(chat.connects_to(&tts));
        assert!(tts.connects_to(&chat));
    }

    #[test]
    fn all_kinds_signature_is_transparent() {
        // pack/unpack declare every kind on both sides; they must never block a chain.
        let all = Signature::new(Kind::all().to_vec(), Kind::all().to_vec());
        let draw = Signature::unary(Kind::Text, Kind::Image);
        assert!(all.connects_to(&draw));
        assert!(draw.connects_to(&all));
    }

    #[test]
    fn empty_side_is_unchecked() {
        // A manifest that declared no kinds is treated as unchecked, not incompatible.
        let none = Signature::new(vec![], vec![]);
        let stt = Signature::unary(Kind::Audio, Kind::Text);
        assert!(none.connects_to(&stt));
        assert!(stt.connects_to(&none));
    }
}
