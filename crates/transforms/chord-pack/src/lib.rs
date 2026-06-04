//! chord-pack — assemble a multimodal [`Message`] for a downstream transform.
//!
//! `pack` is a source/merge transform: it takes any parts piped into it and
//! appends parts named by the `--text`/`--image`/`--audio`/`--video` flags,
//! emitting one framed multi-part message. It's the explicit way to build the
//! input for `chord chat` when you want several files in one prompt:
//!
//! ```text
//! chord pack --image cat.png --text "what is this and what's said?" --audio q.wav | chord chat
//! ```

use std::path::Path;

use chord_core::{
    ChordError, Kind, Message, OptionSpec, Options, Part, Result, Signature, Transform,
};

const OPTS: &[OptionSpec] = &[
    OptionSpec {
        key: "text",
        help: "a text part to add",
        takes_value: true,
    },
    OptionSpec {
        key: "image",
        help: "an image file to add as an image part",
        takes_value: true,
    },
    OptionSpec {
        key: "audio",
        help: "an audio file to add as an audio part",
        takes_value: true,
    },
    OptionSpec {
        key: "video",
        help: "a video file to add as a video part",
        takes_value: true,
    },
];

pub struct Pack;

impl Transform for Pack {
    fn name(&self) -> &str {
        "pack"
    }

    fn signature(&self) -> Signature {
        // Accepts anything piped in (merged into the output) and emits a mix.
        Signature::new(
            vec![Kind::Text, Kind::Image, Kind::Audio, Kind::Video],
            vec![Kind::Text, Kind::Image, Kind::Audio, Kind::Video],
        )
    }

    fn describe(&self) -> &str {
        "assemble a multimodal message from --text/--image/--audio/--video"
    }

    fn options(&self) -> &'static [OptionSpec] {
        OPTS
    }

    fn apply(&self, input: Message, opts: &Options) -> Result<Message> {
        // Start with whatever was piped in, then append the flag parts. Media
        // before text matches the layout most vision/audio models prefer; the
        // model's own adapter has the final say on ordering.
        let mut parts = input.parts;

        if let Some(p) = opts.get("image") {
            parts.push(file_part(Kind::Image, p)?);
        }
        if let Some(p) = opts.get("audio") {
            parts.push(file_part(Kind::Audio, p)?);
        }
        if let Some(p) = opts.get("video") {
            parts.push(file_part(Kind::Video, p)?);
        }
        if let Some(t) = opts.get("text") {
            parts.push(Part::text(t));
        }

        if parts.is_empty() {
            return Err(ChordError::BadInput(
                "pack: nothing to pack (give --text/--image/--audio/--video or pipe input)".into(),
            )
            .into());
        }
        Ok(Message { parts })
    }
}

/// Read a file into a part of `kind`, guessing the MIME type from its extension.
fn file_part(kind: Kind, path: &str) -> Result<Part> {
    let bytes =
        std::fs::read(path).map_err(|e| ChordError::BadInput(format!("reading {path}: {e}")))?;
    Ok(Part::with_mime(kind, mime_for(path, kind), bytes))
}

fn mime_for(path: &str, kind: Kind) -> String {
    let ext = Path::new(path)
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let m = match ext.as_str() {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        "gif" => "image/gif",
        "bmp" => "image/bmp",
        "wav" => "audio/wav",
        "mp3" => "audio/mpeg",
        "flac" => "audio/flac",
        "ogg" => "audio/ogg",
        "mp4" => "video/mp4",
        "mov" => "video/quicktime",
        "webm" => "video/webm",
        _ => return kind.default_mime().to_string(),
    };
    m.to_string()
}
