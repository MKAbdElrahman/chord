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
}

impl Kind {
    /// The canonical lowercase name of the kind.
    pub fn as_str(&self) -> &'static str {
        match self {
            Kind::Text => "text",
            Kind::Audio => "audio",
            Kind::Image => "image",
        }
    }

    /// Parse a user-supplied string (accepts a few common aliases).
    pub fn parse(s: &str) -> Option<Kind> {
        match s.trim().to_ascii_lowercase().as_str() {
            "text" | "txt" => Some(Kind::Text),
            "audio" | "speech" | "sound" => Some(Kind::Audio),
            "image" | "img" | "picture" => Some(Kind::Image),
            _ => None,
        }
    }

    /// All known kinds, for listing.
    pub fn all() -> [Kind; 3] {
        [Kind::Text, Kind::Audio, Kind::Image]
    }
}

impl fmt::Display for Kind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}
