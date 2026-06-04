use std::fmt;

use serde::{Deserialize, Serialize};

/// A data kind (modality) that plug-ins consume and produce.
///
/// Deliberately coarse: the kernel reasons about modalities, not codecs. The
/// difference between WAV and MP3, or PNG and JPEG, is a plug-in's concern, not
/// the core's.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    Text,
    Audio,
    Image,
    Video,
}

impl Kind {
    /// The canonical lowercase name of the kind.
    pub fn as_str(&self) -> &'static str {
        match self {
            Kind::Text => "text",
            Kind::Audio => "audio",
            Kind::Image => "image",
            Kind::Video => "video",
        }
    }

    /// Parse a user-supplied string (accepts a few common aliases).
    pub fn parse(s: &str) -> Option<Kind> {
        match s.trim().to_ascii_lowercase().as_str() {
            "text" | "txt" => Some(Kind::Text),
            "audio" | "speech" | "sound" => Some(Kind::Audio),
            "image" | "img" | "picture" => Some(Kind::Image),
            "video" | "movie" | "clip" => Some(Kind::Video),
            _ => None,
        }
    }

    /// A default MIME type for this kind, used when a part carries no explicit one.
    pub fn default_mime(&self) -> &'static str {
        match self {
            Kind::Text => "text/plain",
            Kind::Audio => "audio/wav",
            Kind::Image => "image/png",
            Kind::Video => "video/mp4",
        }
    }

    /// The wire tag byte for this kind (used by the [`crate::message`] codec).
    pub fn tag(&self) -> u8 {
        match self {
            Kind::Text => 0,
            Kind::Audio => 1,
            Kind::Image => 2,
            Kind::Video => 3,
        }
    }

    /// Parse a wire tag byte back into a kind.
    pub fn from_tag(tag: u8) -> Option<Kind> {
        match tag {
            0 => Some(Kind::Text),
            1 => Some(Kind::Audio),
            2 => Some(Kind::Image),
            3 => Some(Kind::Video),
            _ => None,
        }
    }

    /// All known kinds, for listing.
    pub fn all() -> [Kind; 4] {
        [Kind::Text, Kind::Audio, Kind::Image, Kind::Video]
    }
}

impl fmt::Display for Kind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}
