//! chord-unpack — inspect or extract parts from a multimodal [`Message`].
//!
//! The inverse of `chord pack`. With `--manifest` it prints a JSON description
//! of the parts (kind, mime, byte size) without dumping any binary; with
//! `--part N` it writes the Nth part's raw bytes (1-based). With neither, a
//! single-part message passes through and a multi-part one is an error (you
//! must choose what to extract).
//!
//! ```text
//! chord pack --image a.png --text hi | chord unpack --manifest
//! chord pack --image a.png          | chord unpack --part 1 > a_out.png
//! ```

use chord_core::{ChordError, Kind, Message, OptionSpec, Options, Part, Result, Signature, Transform};

const ALL_KINDS: [Kind; 4] = [Kind::Text, Kind::Audio, Kind::Image, Kind::Video];

const OPTS: &[OptionSpec] = &[
    OptionSpec {
        key: "part",
        help: "extract the Nth part's raw bytes (1-based)",
        takes_value: true,
    },
    OptionSpec {
        key: "manifest",
        help: "print a JSON description of the parts instead of extracting",
        takes_value: false,
    },
];

pub struct Unpack;

impl Transform for Unpack {
    fn name(&self) -> &str {
        "unpack"
    }

    fn signature(&self) -> Signature {
        Signature::new(ALL_KINDS.to_vec(), ALL_KINDS.to_vec())
    }

    fn describe(&self) -> &str {
        "inspect (--manifest) or extract (--part N) parts of a multimodal message"
    }

    fn options(&self) -> &'static [OptionSpec] {
        OPTS
    }

    fn apply(&self, input: Message, opts: &Options) -> Result<Message> {
        if opts.get("manifest").is_some() {
            return Ok(Message::one(Part::text(manifest_json(&input))));
        }

        if let Some(n) = opts.get("part") {
            let idx: usize = n
                .parse()
                .map_err(|_| ChordError::BadInput(format!("--part: not a number: {n:?}")))?;
            if idx == 0 || idx > input.parts.len() {
                return Err(ChordError::BadInput(format!(
                    "--part {idx} out of range (message has {} part(s))",
                    input.parts.len()
                ))
                .into());
            }
            return Ok(Message::one(input.parts[idx - 1].clone()));
        }

        match input.parts.len() {
            0 => Err(ChordError::BadInput("unpack: empty message".into()).into()),
            1 => Ok(input),
            n => Err(ChordError::BadInput(format!(
                "unpack: message has {n} parts; use --part N or --manifest"
            ))
            .into()),
        }
    }
}

/// A small JSON array describing each part: index, kind, mime, byte size.
fn manifest_json(msg: &Message) -> String {
    let mut out = String::from("[");
    for (i, p) in msg.parts.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&format!(
            "{{\"index\":{},\"kind\":\"{}\",\"mime\":\"{}\",\"bytes\":{}}}",
            i + 1,
            p.kind.as_str(),
            p.mime,
            p.as_bytes().len()
        ));
    }
    out.push(']');
    out.push('\n');
    out
}
