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
///
/// `version` lets the host ignore manifest-cache entries written by an older
/// schema (e.g. the pre-`Message` `from`/`to` shape) without a manual cache wipe.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    /// Manifest schema version. Bump when the fields below change shape.
    #[serde(default)]
    pub version: u32,
    /// CLI verb (e.g. `"stt"`).
    pub name: String,
    /// Input modalities this transform consumes.
    pub accepts: Vec<Kind>,
    /// Output modalities this transform produces.
    pub emits: Vec<Kind>,
    /// Inference backend (e.g. `"whisper.cpp"`); empty for engine-less transforms.
    pub backend: String,
    /// One-line human summary.
    pub describe: String,
    /// Flags the engine accepts.
    pub options: Vec<ManifestOption>,
}

/// Current manifest schema version. Entries with a different version are
/// re-queried by the host's discovery cache.
pub const MANIFEST_VERSION: u32 = 2;

impl Manifest {
    /// Snapshot a live transform's metadata into a serializable manifest.
    pub fn of(t: &dyn Transform) -> Self {
        let sig = t.signature();
        Manifest {
            version: MANIFEST_VERSION,
            name: t.name().to_string(),
            accepts: sig.accepts,
            emits: sig.emits,
            backend: t.backend().to_string(),
            describe: t.describe().to_string(),
            options: t.options().iter().map(ManifestOption::from).collect(),
        }
    }
}
