//! The plug-in manifest — the host↔engine wire protocol.
//!
//! Every engine already carries all of its metadata in its [`Transform`] impl.
//! A [`Manifest`] is just a serializable snapshot of that metadata. The engine
//! prints it (as JSON) when invoked with `--chord-manifest`; the host reads it
//! to discover what the engine is and which flags it accepts — so the engine's
//! own `Transform` impl is the single source of truth, and the metadata can no
//! longer drift between the engine and a hand-maintained copy in the host.

use serde::{Deserialize, Serialize};

use crate::{Kind, OptionSpec, ResourceSpec, Transform};

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

/// One fetchable resource an engine declares, in owned form (the in-process
/// [`ResourceSpec`] uses `&'static str` and can't be deserialized).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestResource {
    /// The option key that overrides this resource (e.g. `"model"`).
    pub key: String,
    /// Resolvable reference: an `hf:` ref or a direct `https:` URL.
    pub spec: String,
    /// Human label for download prompts and progress.
    pub describe: String,
}

impl From<&ResourceSpec> for ManifestResource {
    fn from(r: &ResourceSpec) -> Self {
        ManifestResource {
            key: r.key.to_string(),
            spec: r.spec.to_string(),
            describe: r.describe.to_string(),
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
    /// Fetchable resources (default models/assets) the engine needs.
    /// Additive field: absent in pre-resources manifests, so it defaults.
    #[serde(default)]
    pub resources: Vec<ManifestResource>,
    /// Protocol capabilities, announced git-handshake style (e.g. `"each"`:
    /// the binary can loop over a delimited multi-message stream). The
    /// runner — not the engine — appends runner-provided capabilities.
    #[serde(default)]
    pub caps: Vec<String>,
}

/// Current manifest schema version. Entries with a different version are
/// re-queried by the host's discovery cache. Additive `#[serde(default)]`
/// fields (e.g. `resources`) do NOT bump this — only shape changes do.
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
            resources: t.resources().iter().map(ManifestResource::from).collect(),
            caps: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Kind, Message, Options, ResourceSpec, Result, Signature};

    struct WithResources;

    impl Transform for WithResources {
        fn name(&self) -> &str {
            "fake"
        }
        fn signature(&self) -> Signature {
            Signature::unary(Kind::Text, Kind::Text)
        }
        fn describe(&self) -> &str {
            "test"
        }
        fn resources(&self) -> &'static [ResourceSpec] {
            &[ResourceSpec {
                key: "model",
                spec: "hf:org/repo:file.onnx",
                describe: "test model",
            }]
        }
        fn apply(&self, input: Message, _o: &Options) -> Result<Message> {
            Ok(input)
        }
    }

    #[test]
    fn manifest_carries_declared_resources() {
        // R8: the engine's Transform impl is the single source of truth for
        // its resources; the host learns them only through the manifest.
        let m = Manifest::of(&WithResources);
        assert_eq!(m.resources.len(), 1);
        assert_eq!(m.resources[0].key, "model");
        assert_eq!(m.resources[0].spec, "hf:org/repo:file.onnx");
    }

    #[test]
    fn caps_field_is_additive_and_defaults_empty() {
        // Capability negotiation, git-handshake style: the host streams
        // multi-message batches only to engines that advertise "each".
        let old = r#"{"version":2,"name":"x","accepts":["text"],"emits":["text"],"backend":"","describe":"d","options":[]}"#;
        let m: Manifest = serde_json::from_str(old).unwrap();
        assert!(m.caps.is_empty());

        let mut m = Manifest::of(&WithResources);
        m.caps.push("each".to_string());
        let back: Manifest = serde_json::from_str(&serde_json::to_string(&m).unwrap()).unwrap();
        assert_eq!(back.caps, vec!["each"]);
    }

    #[test]
    fn manifest_without_resources_field_still_parses() {
        // Additive compatibility: a v2 manifest emitted before `resources`
        // existed must parse with an empty resource list (no version bump).
        let old = r#"{"version":2,"name":"x","accepts":["text"],"emits":["text"],"backend":"","describe":"d","options":[]}"#;
        let m: Manifest = serde_json::from_str(old).unwrap();
        assert!(m.resources.is_empty());
    }

    #[test]
    fn resources_roundtrip_through_json() {
        let m = Manifest::of(&WithResources);
        let json = serde_json::to_string(&m).unwrap();
        let back: Manifest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.resources[0].describe, "test model");
    }
}
