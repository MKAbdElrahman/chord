//! The plug-in manifest — the host↔engine wire protocol.
//!
//! Every engine already carries all of its metadata in its [`Transform`] impl.
//! A [`Manifest`] is just a serializable snapshot of that metadata. The engine
//! prints it (as JSON) when invoked with `--chord-manifest`; the host reads it
//! to discover what the engine is and which flags it accepts — so the engine's
//! own `Transform` impl is the single source of truth, and the metadata can no
//! longer drift between the engine and a hand-maintained copy in the host.

use serde::{Deserialize, Serialize};

use crate::{Kind, OptionSpec, Transform};

/// One option an engine accepts, in owned form so the host can deserialize it
/// (the in-process [`OptionSpec`] uses `&'static str` and can't be deserialized).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestOption {
    pub key: String,
    pub help: String,
    pub takes_value: bool,
}

impl From<&OptionSpec> for ManifestOption {
    fn from(s: &OptionSpec) -> Self {
        ManifestOption {
            key: s.key.to_string(),
            help: s.help.to_string(),
            takes_value: s.takes_value,
        }
    }
}

/// A serializable description of one transform plug-in.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    /// CLI verb (e.g. `"stt"`).
    pub name: String,
    /// Input modality.
    pub from: Kind,
    /// Output modality.
    pub to: Kind,
    /// Inference backend (e.g. `"whisper.cpp"`); empty for engine-less transforms.
    pub backend: String,
    /// One-line human summary.
    pub describe: String,
    /// Flags the engine accepts.
    pub options: Vec<ManifestOption>,
}

impl Manifest {
    /// Snapshot a live transform's metadata into a serializable manifest.
    pub fn of(t: &dyn Transform) -> Self {
        Manifest {
            name: t.name().to_string(),
            from: t.from(),
            to: t.to(),
            backend: t.backend().to_string(),
            describe: t.describe().to_string(),
            options: t.options().iter().map(ManifestOption::from).collect(),
        }
    }
}
