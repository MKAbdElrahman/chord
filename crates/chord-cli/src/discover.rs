//! Plug-in discovery — the host finds engines the way `git`/`cargo` find their
//! subcommands: any executable named `chord-<…>` next to the host (or on `$PATH`)
//! is a candidate, and each one *advertises itself* via `--chord-manifest`.
//!
//! There is no hardcoded list of engines and no hand-copied option tables: the
//! registry is whatever is installed, described by each engine's own manifest.
//! Manifests are cached in `cache_dir()/manifests.json` keyed by (path, mtime),
//! so after the first run discovery is just a few `stat`s — no re-spawning.

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;
use std::time::UNIX_EPOCH;

use chord_core::{dirs, Kind, Manifest, OptionSpec, MANIFEST_VERSION};
use serde::{Deserialize, Serialize};

/// One installed engine, with all its metadata pre-leaked to `'static` so the
/// host can hand out cheap `Copy` proxies that borrow from it. (`accepts`/`emits`
/// are owned `Vec`s; the `Engine` itself lives `'static` in the discovery cache,
/// so borrows of its fields are `'static` too.)
pub struct Engine {
    pub name: &'static str,
    pub accepts: Vec<Kind>,
    pub emits: Vec<Kind>,
    pub describe: &'static str,
    pub backend: &'static str,
    pub opts: &'static [OptionSpec],
    pub bin: PathBuf,
    /// True when this binary is named exactly `chord-<name>` (the default
    /// backend), as opposed to a `chord-<name>-<variant>` alternate.
    pub is_default: bool,
}

/// All discovered engines (computed once per process).
pub fn engines() -> &'static [Engine] {
    static ENGINES: OnceLock<Vec<Engine>> = OnceLock::new();
    ENGINES.get_or_init(discover)
}

fn discover() -> Vec<Engine> {
    let mut cache = Cache::load();
    let mut seen = HashSet::new();
    let mut engines = Vec::new();

    // Exe dir first, then PATH: a binary beside the host wins over one on PATH.
    for dir in search_dirs() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Some(fname) = path.file_name().and_then(|s| s.to_str()) else {
                continue;
            };
            if !fname.starts_with("chord-") || !is_executable(&path) {
                continue;
            }
            // First occurrence of a given binary name wins.
            if !seen.insert(fname.to_string()) {
                continue;
            }
            if let Some(manifest) = cache.manifest_for(&path) {
                let is_default = fname == format!("chord-{}", manifest.name);
                engines.push(engine_from(manifest, path, is_default));
            }
        }
    }
    cache.save();
    engines
}

/// Build a `'static`-stringed [`Engine`] from a manifest by leaking its owned
/// strings (bounded: a handful of engines, each with a few options, leaked once
/// for the life of the process — the same pattern the host uses for clap names).
fn engine_from(m: Manifest, bin: PathBuf, is_default: bool) -> Engine {
    let opts: Vec<OptionSpec> = m
        .options
        .iter()
        .map(|o| OptionSpec {
            key: leak(&o.key),
            help: leak(&o.help),
            takes_value: o.takes_value,
        })
        .collect();
    Engine {
        name: leak(&m.name),
        accepts: m.accepts,
        emits: m.emits,
        describe: leak(&m.describe),
        backend: leak(&m.backend),
        opts: Box::leak(opts.into_boxed_slice()),
        bin,
        is_default,
    }
}

fn leak(s: &str) -> &'static str {
    Box::leak(s.to_string().into_boxed_str())
}

fn search_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            dirs.push(parent.to_path_buf());
        }
    }
    if let Some(path) = std::env::var_os("PATH") {
        dirs.extend(std::env::split_paths(&path));
    }
    dirs
}

#[cfg(unix)]
fn is_executable(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    std::fs::metadata(path)
        .map(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
        .unwrap_or(false)
}

#[cfg(not(unix))]
fn is_executable(path: &Path) -> bool {
    path.is_file()
}

/// Ask one binary to describe itself. `None` means it isn't a chord engine
/// (didn't respond to `--chord-manifest` with valid JSON).
fn query(bin: &Path) -> Option<Manifest> {
    let out = Command::new(bin).arg("--chord-manifest").output().ok()?;
    if !out.status.success() {
        return None;
    }
    serde_json::from_slice(&out.stdout).ok()
}

/// The (path, mtime) → manifest cache, so steady-state discovery never spawns.
#[derive(Default, Serialize, Deserialize)]
struct Cache {
    entries: BTreeMap<String, Entry>,
    #[serde(skip)]
    dirty: bool,
}

#[derive(Serialize, Deserialize, Clone)]
struct Entry {
    mtime_ns: u128,
    manifest: Manifest,
}

impl Cache {
    fn path() -> PathBuf {
        dirs::cache_dir().join("manifests.json")
    }

    fn load() -> Cache {
        std::fs::read(Self::path())
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or_default()
    }

    /// Return `bin`'s manifest, from cache if its mtime is unchanged, else by
    /// querying the binary and updating the cache.
    fn manifest_for(&mut self, bin: &Path) -> Option<Manifest> {
        let mtime_ns = mtime_ns(bin)?;
        let key = bin.to_string_lossy().into_owned();
        if let Some(entry) = self.entries.get(&key) {
            // Ignore entries written by an older manifest schema (e.g. the
            // pre-Message `from`/`to` shape) so the host re-queries the engine.
            if entry.mtime_ns == mtime_ns && entry.manifest.version == MANIFEST_VERSION {
                return Some(entry.manifest.clone());
            }
        }
        let manifest = query(bin)?;
        self.entries.insert(
            key,
            Entry {
                mtime_ns,
                manifest: manifest.clone(),
            },
        );
        self.dirty = true;
        Some(manifest)
    }

    fn save(&self) {
        if !self.dirty {
            return;
        }
        let path = Self::path();
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(bytes) = serde_json::to_vec(self) {
            let _ = std::fs::write(path, bytes);
        }
    }
}

fn mtime_ns(path: &Path) -> Option<u128> {
    let modified = std::fs::metadata(path).ok()?.modified().ok()?;
    Some(modified.duration_since(UNIX_EPOCH).ok()?.as_nanos())
}
