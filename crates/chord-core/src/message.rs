//! The chord data plane: a [`Message`] is an **ordered list of typed [`Part`]s**.
//!
//! This is the one currency that flows across a chord pipe. A transform is, in
//! full generality, a function `Message -> Message`. Mono-modal transforms
//! (`stt`, `tts`, `text`, `see`, `draw`, …) are the special case of exactly one
//! input part and one output part — see [`crate::transform::Unary`].
//!
//! ## Wire format & the singleton-raw rule
//!
//! The kernel's reason for existing is that chord composes with ordinary Unix
//! tools (`aplay`, `ffmpeg`, `> file.png`). So the encoding has one principled
//! special case:
//!
//! - A `Message` of **exactly one inline part** serializes as its **bare payload
//!   bytes** — no header. This is what keeps `chord tts | aplay` and
//!   `chord draw > x.png` byte-identical to a world without messages.
//! - Anything else (multi-part, or a by-reference part) serializes **framed**,
//!   behind a magic prefix.
//!
//! Decoding mirrors it: peek the magic; if present, parse a framed message;
//! otherwise the whole stream is a single part whose [`Kind`] is supplied by the
//! consumer (its primary accepted kind). The magic starts with a NUL byte, which
//! no text payload and none of the common media magics (`RIFF`, `\x89PNG`,
//! `\xff\xd8`, `ID3`, `ftyp`) begin with, so the discriminator is unambiguous in
//! practice.

use std::collections::BTreeMap;
use std::io::{Read, Write};

use crate::{ChordError, Kind, Result};

/// Frame magic for a multi-part / by-reference message. Leading NUL makes it
/// disjoint from text and from common media file signatures.
const MAGIC: [u8; 6] = [0x00, b'C', b'H', b'R', b'D', 0x01];

/// Length of the frame discriminator: how many leading bytes a consumer must
/// peek to tell a framed message from a raw payload.
pub const MAGIC_LEN: usize = MAGIC.len();

/// True when `prefix` — the first bytes of a stream — begins a framed
/// message. A prefix shorter than the magic is necessarily raw, so peeking
/// [`MAGIC_LEN`] bytes (or hitting EOF first) always decides.
pub fn is_framed(prefix: &[u8]) -> bool {
    prefix.len() >= MAGIC_LEN && prefix[..MAGIC_LEN] == MAGIC
}

/// Where a part's bytes live.
#[derive(Debug, Clone)]
pub enum Body {
    /// The bytes themselves, carried inline on the wire.
    Inline(Vec<u8>),
    /// A reference to bytes stored out-of-band (content id or path). Reserved
    /// for large payloads (video) that shouldn't be re-copied across stages;
    /// not yet produced by any engine, but the wire format reserves a tag.
    Ref(String),
}

/// One typed element of a [`Message`]: a modality, an exact MIME type, open
/// metadata, and a body.
#[derive(Debug, Clone)]
pub struct Part {
    pub kind: Kind,
    pub mime: String,
    pub meta: BTreeMap<String, String>,
    pub body: Body,
}

impl Part {
    /// A part with inline bytes and the kind's default MIME type.
    pub fn new(kind: Kind, bytes: Vec<u8>) -> Self {
        Part {
            kind,
            mime: kind.default_mime().to_string(),
            meta: BTreeMap::new(),
            body: Body::Inline(bytes),
        }
    }

    /// A part with an explicit MIME type.
    pub fn with_mime(kind: Kind, mime: impl Into<String>, bytes: Vec<u8>) -> Self {
        Part {
            kind,
            mime: mime.into(),
            meta: BTreeMap::new(),
            body: Body::Inline(bytes),
        }
    }

    /// A `text/plain` part.
    pub fn text(s: impl Into<String>) -> Self {
        Part::new(Kind::Text, s.into().into_bytes())
    }

    /// The inline bytes of this part, or an empty slice for a by-reference part.
    pub fn as_bytes(&self) -> &[u8] {
        match &self.body {
            Body::Inline(b) => b,
            Body::Ref(_) => &[],
        }
    }

    /// Attach a metadata key/value (builder style).
    pub fn with_meta(mut self, key: impl Into<String>, val: impl Into<String>) -> Self {
        self.meta.insert(key.into(), val.into());
        self
    }
}

/// An ordered list of typed parts — the chord data plane.
#[derive(Debug, Clone, Default)]
pub struct Message {
    pub parts: Vec<Part>,
}

impl Message {
    /// An empty message (no parts).
    pub fn empty() -> Self {
        Message { parts: Vec::new() }
    }

    /// A single-part message.
    pub fn one(part: Part) -> Self {
        Message { parts: vec![part] }
    }

    /// True when there are no parts.
    pub fn is_empty(&self) -> bool {
        self.parts.is_empty()
    }

    /// Require exactly one part of `kind` and return it. The contract a 1→1
    /// (`Unary`) transform relies on; errors as [`ChordError::BadInput`] otherwise.
    pub fn single(&self, kind: Kind) -> Result<&Part> {
        match self.parts.as_slice() {
            [p] if p.kind == kind => Ok(p),
            [] => Err(ChordError::BadInput(format!("expected one {kind} part, got none")).into()),
            [p] => {
                Err(ChordError::BadInput(format!("expected a {kind} part, got {}", p.kind)).into())
            }
            parts => Err(ChordError::BadInput(format!(
                "expected one {kind} part, got {}",
                parts.len()
            ))
            .into()),
        }
    }

    /// Map every part of `kind` through `f`, passing parts of other kinds through
    /// unchanged and in place. The "operate on what I understand, leave the rest
    /// alone" combinator that makes heterogeneous messages composable.
    pub fn map_kind(mut self, kind: Kind, mut f: impl FnMut(Part) -> Part) -> Self {
        for p in self.parts.iter_mut() {
            if p.kind == kind {
                let taken = std::mem::replace(p, Part::text(""));
                *p = f(taken);
            }
        }
        self
    }
}

// ---------------------------------------------------------------------------
// Wire codec
// ---------------------------------------------------------------------------

/// Serialize a message to `w`. A lone inline part is written raw (singleton-raw
/// rule); anything else is framed behind [`MAGIC`].
pub fn encode(msg: &Message, w: &mut dyn Write) -> Result<()> {
    if let [Part {
        body: Body::Inline(bytes),
        ..
    }] = msg.parts.as_slice()
    {
        w.write_all(bytes)?;
        return Ok(());
    }
    // Empty message → empty stream (no frame), so a no-op stage stays clean.
    if msg.parts.is_empty() {
        return Ok(());
    }

    w.write_all(&MAGIC)?;
    for p in &msg.parts {
        w.write_all(&[p.kind.tag()])?;
        write_str(w, &p.mime)?;
        write_meta(w, &p.meta)?;
        match &p.body {
            Body::Inline(bytes) => {
                w.write_all(&[0u8])?;
                write_u64(w, bytes.len() as u64)?;
                w.write_all(bytes)?;
            }
            Body::Ref(s) => {
                w.write_all(&[1u8])?;
                write_str_u64(w, s)?;
            }
        }
    }
    Ok(())
}

/// Deserialize a message from `r`. Unframed input becomes a single part of
/// `default_kind` (the consuming transform's primary accepted kind).
pub fn decode(r: &mut dyn Read, default_kind: Kind) -> Result<Message> {
    // Read the whole stream; messages are buffered in this version.
    let mut buf = Vec::new();
    r.read_to_end(&mut buf)?;

    if buf.is_empty() {
        return Ok(Message::empty());
    }
    if buf.len() < MAGIC.len() || buf[..MAGIC.len()] != MAGIC {
        // Raw singleton: the whole stream is one part of the expected kind.
        return Ok(Message::one(Part::new(default_kind, buf)));
    }

    let mut cur = &buf[MAGIC.len()..];
    let mut parts = Vec::new();
    while !cur.is_empty() {
        let tag = take(&mut cur, 1)?[0];
        let kind = Kind::from_tag(tag)
            .ok_or_else(|| ChordError::BadInput(format!("bad part kind tag {tag}")))?;
        let mime = read_str(&mut cur)?;
        let meta = read_meta(&mut cur)?;
        let body_tag = take(&mut cur, 1)?[0];
        let body = match body_tag {
            0 => {
                let len = read_u64(&mut cur)? as usize;
                Body::Inline(take(&mut cur, len)?.to_vec())
            }
            1 => Body::Ref(read_str_u64(&mut cur)?),
            other => {
                return Err(ChordError::BadInput(format!("bad body tag {other}")).into());
            }
        };
        parts.push(Part {
            kind,
            mime,
            meta,
            body,
        });
    }
    Ok(Message { parts })
}

// ---- little-endian-free primitives (all big-endian) ----

fn write_u16(w: &mut dyn Write, v: u16) -> Result<()> {
    w.write_all(&v.to_be_bytes())?;
    Ok(())
}
fn write_u32(w: &mut dyn Write, v: u32) -> Result<()> {
    w.write_all(&v.to_be_bytes())?;
    Ok(())
}
fn write_u64(w: &mut dyn Write, v: u64) -> Result<()> {
    w.write_all(&v.to_be_bytes())?;
    Ok(())
}
/// String with a u16 length prefix (for short fields like MIME and meta keys).
fn write_str(w: &mut dyn Write, s: &str) -> Result<()> {
    write_u16(w, s.len() as u16)?;
    w.write_all(s.as_bytes())?;
    Ok(())
}
/// String with a u64 length prefix (for ref bodies, which can be long).
fn write_str_u64(w: &mut dyn Write, s: &str) -> Result<()> {
    write_u64(w, s.len() as u64)?;
    w.write_all(s.as_bytes())?;
    Ok(())
}
fn write_meta(w: &mut dyn Write, meta: &BTreeMap<String, String>) -> Result<()> {
    write_u16(w, meta.len() as u16)?;
    for (k, v) in meta {
        write_str(w, k)?;
        write_u32(w, v.len() as u32)?;
        w.write_all(v.as_bytes())?;
    }
    Ok(())
}

fn take<'a>(cur: &mut &'a [u8], n: usize) -> Result<&'a [u8]> {
    if cur.len() < n {
        return Err(ChordError::BadInput("truncated message frame".into()).into());
    }
    let (head, tail) = cur.split_at(n);
    *cur = tail;
    Ok(head)
}
fn read_u16(cur: &mut &[u8]) -> Result<u16> {
    Ok(u16::from_be_bytes(take(cur, 2)?.try_into().unwrap()))
}
fn read_u32(cur: &mut &[u8]) -> Result<u32> {
    Ok(u32::from_be_bytes(take(cur, 4)?.try_into().unwrap()))
}
fn read_u64(cur: &mut &[u8]) -> Result<u64> {
    Ok(u64::from_be_bytes(take(cur, 8)?.try_into().unwrap()))
}
fn read_str(cur: &mut &[u8]) -> Result<String> {
    let len = read_u16(cur)? as usize;
    let bytes = take(cur, len)?;
    String::from_utf8(bytes.to_vec())
        .map_err(|_| ChordError::BadInput("bad utf8 in message frame".into()).into())
}
fn read_str_u64(cur: &mut &[u8]) -> Result<String> {
    let len = read_u64(cur)? as usize;
    let bytes = take(cur, len)?;
    String::from_utf8(bytes.to_vec())
        .map_err(|_| ChordError::BadInput("bad utf8 in message frame".into()).into())
}
fn read_meta(cur: &mut &[u8]) -> Result<BTreeMap<String, String>> {
    let n = read_u16(cur)? as usize;
    let mut meta = BTreeMap::new();
    for _ in 0..n {
        let k = read_str(cur)?;
        let vlen = read_u32(cur)? as usize;
        let v = String::from_utf8(take(cur, vlen)?.to_vec())
            .map_err(|_| ChordError::BadInput("bad utf8 in message meta".to_string()))?;
        meta.insert(k, v);
    }
    Ok(meta)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(msg: &Message, default_kind: Kind) -> Message {
        let mut buf = Vec::new();
        encode(msg, &mut buf).unwrap();
        decode(&mut buf.as_slice(), default_kind).unwrap()
    }

    #[test]
    fn singleton_text_is_raw() {
        let msg = Message::one(Part::text("hello"));
        let mut buf = Vec::new();
        encode(&msg, &mut buf).unwrap();
        // No frame: the bytes are exactly the payload.
        assert_eq!(buf, b"hello");
    }

    #[test]
    fn singleton_binary_is_raw_and_roundtrips() {
        let png = vec![0x89, b'P', b'N', b'G', 1, 2, 3];
        let msg = Message::one(Part::new(Kind::Image, png.clone()));
        let mut buf = Vec::new();
        encode(&msg, &mut buf).unwrap();
        assert_eq!(buf, png); // raw passthrough
        let back = decode(&mut buf.as_slice(), Kind::Image).unwrap();
        assert_eq!(back.parts.len(), 1);
        assert_eq!(back.parts[0].kind, Kind::Image);
        assert_eq!(back.parts[0].as_bytes(), png.as_slice());
    }

    #[test]
    fn unframed_decodes_to_default_kind() {
        let raw = b"some audio bytes".to_vec();
        let back = decode(&mut raw.as_slice(), Kind::Audio).unwrap();
        assert_eq!(back.parts.len(), 1);
        assert_eq!(back.parts[0].kind, Kind::Audio);
        assert_eq!(back.parts[0].as_bytes(), raw.as_slice());
    }

    #[test]
    fn multipart_roundtrips() {
        let msg = Message {
            parts: vec![
                Part::with_mime(Kind::Image, "image/png", vec![1, 2, 3]).with_meta("role", "user"),
                Part::text("describe this and transcribe the audio"),
                Part::with_mime(Kind::Audio, "audio/wav", vec![9, 8, 7, 6]),
            ],
        };
        let back = roundtrip(&msg, Kind::Text);
        assert_eq!(back.parts.len(), 3);
        assert_eq!(back.parts[0].kind, Kind::Image);
        assert_eq!(back.parts[0].mime, "image/png");
        assert_eq!(
            back.parts[0].meta.get("role").map(String::as_str),
            Some("user")
        );
        assert_eq!(back.parts[1].kind, Kind::Text);
        assert_eq!(
            back.parts[1].as_bytes(),
            b"describe this and transcribe the audio"
        );
        assert_eq!(back.parts[2].kind, Kind::Audio);
        assert_eq!(back.parts[2].as_bytes(), &[9, 8, 7, 6]);
    }

    #[test]
    fn empty_roundtrips() {
        let back = roundtrip(&Message::empty(), Kind::Text);
        assert!(back.is_empty());
    }

    #[test]
    fn framed_prefix_is_detected() {
        let msg = Message {
            parts: vec![Part::text("a"), Part::text("b")],
        };
        let mut buf = Vec::new();
        encode(&msg, &mut buf).unwrap();
        assert!(is_framed(&buf));
        assert!(is_framed(&MAGIC));
    }

    #[test]
    fn raw_and_short_prefixes_are_not_framed() {
        assert!(!is_framed(b"hello, this is plain text"));
        assert!(!is_framed(b""));
        assert!(!is_framed(&MAGIC[..3])); // shorter than the magic is necessarily raw
        assert!(!is_framed(&[0x89, b'P', b'N', b'G'])); // media magic, not ours
    }

    #[test]
    fn ref_body_roundtrips() {
        let msg = Message {
            parts: vec![
                Part::text("a"),
                Part {
                    kind: Kind::Video,
                    mime: "video/mp4".into(),
                    meta: BTreeMap::new(),
                    body: Body::Ref("cid:sha256:deadbeef".into()),
                },
            ],
        };
        let back = roundtrip(&msg, Kind::Text);
        assert_eq!(back.parts.len(), 2);
        match &back.parts[1].body {
            Body::Ref(s) => assert_eq!(s, "cid:sha256:deadbeef"),
            _ => panic!("expected ref body"),
        }
    }
}
