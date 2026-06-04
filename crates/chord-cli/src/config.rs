//! Configuration loading — a **host concern**, deliberately outside the kernel.
//!
//! The user describes their preferred model and option defaults in a YAML
//! file with one section per transform (keyed by the transform's name). The
//! host turns the matching section into the [`Options`] the transform already
//! reads, so pipelines stay free of inline option flags:
//!
//! ```yaml
//! stt:  { model: large-v3-turbo, lang: en }
//! tts:  { voice: M1, lang: en }
//! chat: { model: ~/models/qwen3-8b.gguf, system: "Be concise." }
//! ```
//!
//! Precedence is applied by the caller: named CLI flags override config, which
//! overrides the transform's built-in defaults.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use chord_core::{Error, Options};

/// Parsed configuration: transform-name -> (key -> value), values already
/// `${ENV}`-expanded.
pub struct Config {
    pub path: Option<PathBuf>,
    sections: HashMap<String, HashMap<String, String>>,
}

impl Config {
    /// Load config from `explicit` if given, else from the first of
    /// `$CHORD_CONFIG`, `./chord.yaml`, `~/.config/chord/config.yaml`. A missing
    /// auto-discovered file yields an empty config (transforms use their
    /// defaults); a missing *explicit* file is an error.
    pub fn load(explicit: Option<&str>) -> Result<Config, Error> {
        let path = match explicit {
            Some(p) => Some(PathBuf::from(p)),
            None => discover(),
        };

        let Some(path) = path else {
            return Ok(Config {
                path: None,
                sections: HashMap::new(),
            });
        };

        let text = std::fs::read_to_string(&path)
            .map_err(|e| format!("reading config {}: {e}", path.display()))?;
        let raw: HashMap<String, HashMap<String, serde_yaml_ng::Value>> =
            serde_yaml_ng::from_str(&text)
                .map_err(|e| format!("parsing config {}: {e}", path.display()))?;

        let sections = raw
            .into_iter()
            .map(|(section, kvs)| {
                let kvs = kvs
                    .into_iter()
                    .filter_map(|(k, v)| value_to_string(&v).map(|s| (k, expand_env(&s))))
                    .collect();
                (section, kvs)
            })
            .collect();

        Ok(Config {
            path: Some(path),
            sections,
        })
    }

    /// The Options for a transform, drawn from its config section (empty if the
    /// section is absent).
    pub fn options_for(&self, name: &str) -> Options {
        let mut opts = Options::new();
        if let Some(section) = self.sections.get(name) {
            for (k, v) in section {
                opts.insert(k.clone(), v.clone());
            }
        }
        opts
    }

    /// Sections, for `chord config` display.
    pub fn sections(&self) -> &HashMap<String, HashMap<String, String>> {
        &self.sections
    }
}

fn discover() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("CHORD_CONFIG") {
        if !p.is_empty() {
            return Some(PathBuf::from(p));
        }
    }
    let cwd = Path::new("chord.yaml");
    if cwd.exists() {
        return Some(cwd.to_path_buf());
    }
    // The XDG config dir (`$XDG_CONFIG_HOME/chord`), via the shared `dirs` module.
    let p = chord_core::dirs::config_dir().join("config.yaml");
    p.exists().then_some(p)
}

/// Render a scalar YAML value as a string; non-scalars (maps/sequences/null) are
/// dropped, since Options are flat string key-values.
fn value_to_string(v: &serde_yaml_ng::Value) -> Option<String> {
    match v {
        serde_yaml_ng::Value::String(s) => Some(s.clone()),
        serde_yaml_ng::Value::Bool(b) => Some(b.to_string()),
        serde_yaml_ng::Value::Number(n) => Some(n.to_string()),
        _ => None,
    }
}

/// Replace every `${VAR}` with the environment variable's value (empty if
/// unset). Keeps secrets like API keys out of the file.
fn expand_env(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(pos) = rest.find("${") {
        out.push_str(&rest[..pos]);
        rest = &rest[pos + 2..];
        match rest.find('}') {
            Some(end) => {
                let var = &rest[..end];
                out.push_str(&std::env::var(var).unwrap_or_default());
                rest = &rest[end + 1..];
            }
            None => {
                out.push_str("${");
                break;
            }
        }
    }
    out.push_str(rest);
    out
}
