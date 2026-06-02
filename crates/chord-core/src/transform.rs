use std::collections::HashMap;
use std::io::{Read, Write};

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

/// A `Transform` converts one data [`Kind`] into another. It is a self-contained
/// Unix filter: read `input`, write `output`.
///
/// Contract every implementation MUST honor:
/// 1. **Stateless across calls** — the value holds config, not per-request state.
/// 2. **Stream, don't hoard** — read `input`, write `output`; never close
///    `output`, and never write logs/progress to it (stdout is the data plane;
///    diagnostics go to stderr).
/// 3. **Kind honesty** — [`from`](Transform::from)/[`to`](Transform::to) must be
///    accurate.
/// 4. **Options are advisory** — ignore unknown keys; default missing ones.
pub trait Transform: Send + Sync {
    /// Unique CLI verb for this plug-in (e.g. `"stt"`, `"tts"`).
    fn name(&self) -> &str;

    /// Input modality.
    fn from(&self) -> Kind;

    /// Output modality.
    fn to(&self) -> Kind;

    /// One-line human summary, shown by `chord ls`.
    fn describe(&self) -> &str;

    /// The options this transform accepts, so the host can render them as CLI
    /// flags. Defaults to none.
    fn options(&self) -> &'static [OptionSpec] {
        &[]
    }

    /// Read the input stream, write the output stream.
    fn apply(&self, input: &mut dyn Read, output: &mut dyn Write, opts: &Options) -> Result<()>;
}
