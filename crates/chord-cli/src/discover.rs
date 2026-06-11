//! Plug-in discovery — the host finds engines the way `git`/`cargo` find their
//! subcommands: any executable named `chord-<…>` next to the host (or on `$PATH`)
//! is a candidate, and each one *advertises itself* via `--chord-manifest`.
//!
//! There is no hardcoded list of engines and no hand-copied option tables: the
//! registry is whatever is installed, described by each engine's own manifest.
//! Manifests are cached in `cache_dir()/manifests.json` keyed by (path, mtime),
//! so after the first run discovery is just a few `stat`s — no re-spawning.

use std::collections::{BTreeMap, HashSet};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::OnceLock;
use std::time::{Duration, Instant, UNIX_EPOCH};

use chord_core::{dirs, Kind, Manifest, OptionSpec, ResourceSpec, MANIFEST_VERSION};
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
    /// Fetchable resources the engine declared in its manifest (R8: the
    /// host fetches what engines declare; it knows no engine specifics).
    pub resources: &'static [ResourceSpec],
    /// Protocol capabilities the binary announced (e.g. `"each"`).
    pub caps: &'static [&'static str],
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
        // read_dir order is filesystem-dependent; sort so discovery order —
        // and everything keyed off it (`chord ls`, default fallback) — is
        // deterministic across machines.
        let mut entries: Vec<_> = entries.flatten().collect();
        entries.sort_by_key(|e| e.file_name());
        for entry in entries {
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
        // `__`-prefixed keys are host-reserved (e.g. the __format relay); an
        // engine manifest cannot inject into that namespace.
        .filter(|o| !o.key.starts_with("__"))
        .map(|o| OptionSpec {
            key: leak(&o.key),
            help: leak(&o.help),
            takes_value: o.takes_value,
        })
        .collect();
    let resources: Vec<ResourceSpec> = m
        .resources
        .iter()
        .map(|r| ResourceSpec {
            key: leak(&r.key),
            spec: leak(&r.spec),
            describe: leak(&r.describe),
        })
        .collect();
    Engine {
        name: leak(&m.name),
        accepts: m.accepts,
        emits: m.emits,
        describe: leak(&m.describe),
        backend: leak(&m.backend),
        opts: Box::leak(opts.into_boxed_slice()),
        resources: Box::leak(resources.into_boxed_slice()),
        caps: Box::leak(
            m.caps
                .iter()
                .map(|c| leak(c))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        ),
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

/// How long a `--chord-manifest` probe may run before we kill it. Generous —
/// printing JSON takes milliseconds — this only guards against a hung or
/// non-chord binary. Kahn restriction (ii): a line transmits in *finite*
/// time; one stuck binary on PATH must not brick every chord invocation.
const PROBE_TIMEOUT: Duration = Duration::from_secs(2);

/// Ask one binary to describe itself. `None` means it isn't a usable chord
/// engine: no/invalid manifest, a hung probe, or a manifest version this
/// host doesn't speak (which is reported, not silent).
fn query(bin: &Path) -> Option<Manifest> {
    let out = probe(bin, PROBE_TIMEOUT)?;
    match parse_manifest(&out) {
        Ok(m) => Some(m),
        Err(ManifestIssue::VersionMismatch(v)) => {
            eprintln!(
                "chord: ignoring {}: manifest version {v}, host speaks {MANIFEST_VERSION} \
                 (rebuild the engine or update chord)",
                bin.display()
            );
            None
        }
        Err(ManifestIssue::Invalid) => None,
    }
}

/// Why a manifest payload was rejected.
#[derive(Debug)]
enum ManifestIssue {
    /// Not JSON / not a manifest at all — an ordinary non-chord binary.
    Invalid,
    /// A chord manifest, but from a different schema version.
    VersionMismatch(u32),
}

/// Parse a manifest, surfacing version skew. The loose first pass extracts
/// `version` even when the rest of the schema changed shape, so a skewed
/// engine is diagnosable instead of silently vanishing from `chord ls`.
fn parse_manifest(bytes: &[u8]) -> std::result::Result<Manifest, ManifestIssue> {
    let v: serde_json::Value = serde_json::from_slice(bytes).map_err(|_| ManifestIssue::Invalid)?;
    let version = v.get("version").and_then(|x| x.as_u64()).unwrap_or(0) as u32;
    if version != MANIFEST_VERSION {
        return Err(ManifestIssue::VersionMismatch(version));
    }
    serde_json::from_value(v).map_err(|_| ManifestIssue::Invalid)
}

/// Run `bin --chord-manifest` and return its stdout, killing the child if it
/// outlives `timeout`. A thread drains stdout so a chatty child can't block
/// on a full pipe while we wait.
fn probe(bin: &Path, timeout: Duration) -> Option<Vec<u8>> {
    let mut child = Command::new(bin)
        .arg("--chord-manifest")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let mut stdout = child.stdout.take()?;
    let reader = std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = stdout.read_to_end(&mut buf);
        buf
    });
    let deadline = Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let out = reader.join().ok()?;
                return status.success().then_some(out);
            }
            Ok(None) if Instant::now() >= deadline => {
                let _ = child.kill();
                let _ = child.wait();
                // Do NOT join the reader: an orphaned grandchild (e.g. a
                // shell's spawned sleep) can inherit the pipe and keep it
                // open long after the probe is dead. The thread parks
                // harmlessly until that fd closes.
                return None;
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(15)),
            Err(_) => return None,
        }
    }
}

/// Write `bytes` to `path` via a temp file + rename, so a concurrent reader
/// never observes a torn write — the same discipline as `pull`'s `.part`
/// download files.
fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension(format!("tmp{}", std::process::id()));
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)
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
        if let Ok(bytes) = serde_json::to_vec(self) {
            let _ = write_atomic(&Self::path(), &bytes);
        }
    }
}

fn mtime_ns(path: &Path) -> Option<u128> {
    let modified = std::fs::metadata(path).ok()?.modified().ok()?;
    Some(modified.duration_since(UNIX_EPOCH).ok()?.as_nanos())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest_json(version: u32) -> String {
        format!(
            r#"{{"version":{version},"name":"fake","accepts":["text"],"emits":["text"],"backend":"","describe":"d","options":[{{"key":"__format","help":"h","takes_value":true}},{{"key":"model","help":"h","takes_value":true}}],"resources":[{{"key":"model","spec":"hf:org/repo:m.bin","describe":"m"}}],"caps":["each"]}}"#
        )
    }

    #[test]
    fn parse_manifest_accepts_current_version() {
        assert!(parse_manifest(manifest_json(MANIFEST_VERSION).as_bytes()).is_ok());
    }

    #[test]
    fn parse_manifest_surfaces_version_mismatch_instead_of_silence() {
        // A version-skewed engine must be reportable, not silently invisible.
        match parse_manifest(manifest_json(99).as_bytes()) {
            Err(ManifestIssue::VersionMismatch(99)) => {}
            other => panic!("expected VersionMismatch(99), got {other:?}"),
        }
    }

    #[test]
    fn parse_manifest_rejects_garbage() {
        assert!(matches!(
            parse_manifest(b"not json"),
            Err(ManifestIssue::Invalid)
        ));
    }

    #[test]
    fn engine_from_filters_host_reserved_option_keys() {
        // `__`-prefixed keys are the host's namespace (e.g. __format); an
        // engine manifest must not be able to collide with it.
        let m: Manifest = serde_json::from_str(&manifest_json(MANIFEST_VERSION)).unwrap();
        let e = engine_from(m, PathBuf::from("/bin/chord-fake"), true);
        assert!(
            e.opts.iter().all(|o| !o.key.starts_with("__")),
            "{:?}",
            e.opts
        );
        assert!(e.opts.iter().any(|o| o.key == "model"));
    }

    #[test]
    fn engine_from_carries_declared_resources() {
        // R8: resource knowledge travels engine -> manifest -> host, never
        // the other way.
        let m: Manifest = serde_json::from_str(&manifest_json(MANIFEST_VERSION)).unwrap();
        let e = engine_from(m, PathBuf::from("/bin/chord-fake"), true);
        assert_eq!(e.resources.len(), 1);
        assert_eq!(e.resources[0].spec, "hf:org/repo:m.bin");
        assert_eq!(e.resources[0].key, "model");
        assert_eq!(e.caps, ["each"]);
    }

    #[test]
    fn write_atomic_writes_content_and_leaves_no_temp_file() {
        let dir = std::env::temp_dir().join(format!("chord-wa-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("cache.json");
        write_atomic(&path, b"hello").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"hello");
        let leftovers: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .filter(|e| e.path() != path)
            .collect();
        assert!(leftovers.is_empty(), "{leftovers:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Serializes the script-spawning tests. Writing a script in one test
    /// thread while another test's `Command::spawn` forks can leave the
    /// script's fd open in the forked child for a moment, making the first
    /// test's exec fail with ETXTBSY.
    #[cfg(unix)]
    static SPAWN_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[cfg(unix)]
    fn script(dir: &Path, name: &str, body: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let p = dir.join(name);
        std::fs::write(&p, format!("#!/bin/sh\n{body}\n")).unwrap();
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
        p
    }

    #[cfg(unix)]
    #[test]
    fn query_returns_manifest_from_a_responsive_engine() {
        let _serial = SPAWN_LOCK.lock().unwrap();
        let dir = std::env::temp_dir().join(format!("chord-q-ok-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let bin = script(
            &dir,
            "chord-ok",
            &format!("echo '{}'", manifest_json(MANIFEST_VERSION)),
        );
        assert!(query(&bin).is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn query_kills_a_hung_probe_within_the_deadline() {
        // Kahn restriction (ii): a line transmits in finite time. One hung
        // binary on PATH must not brick every chord invocation.
        let _serial = SPAWN_LOCK.lock().unwrap();
        let dir = std::env::temp_dir().join(format!("chord-q-hang-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let bin = script(&dir, "chord-hang", "sleep 30");
        let t0 = std::time::Instant::now();
        assert!(query(&bin).is_none());
        assert!(
            t0.elapsed() < std::time::Duration::from_secs(5),
            "probe took {:?}",
            t0.elapsed()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
