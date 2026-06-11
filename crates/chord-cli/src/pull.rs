//! `chord pull <transform>` — fetch a transform's declared resources.
//!
//! Explicit by design: engine models range from ~1 GB (whisper) to 20-30 GB
//! (the chat/see GGUFs), so chord never downloads silently. A missing model
//! errors with a hint to run this command.
//!
//! The host hardcodes **no engine knowledge** here (rule R8 in
//! `docs/theory/THEORY.md`): engines declare their default resources in
//! their manifests; this module only understands the reference *schemes*
//! (`hf:` → the shared HF cache, `https:` → `models_dir()/<basename>`) and
//! owns the UX — the one-prompt-then-download flow and the progress bar.

use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::PathBuf;

use chord_core::{dirs, ChordError, Options, ResourceSpec, Result, Transform};
use indicatif::{ProgressBar, ProgressStyle};

/// Make every resource a transform needs present before it runs: `hf:`
/// references the user supplied as option values, plus the engine's declared
/// default resources (skipped when their overriding option is set).
///
/// If one is missing: at an interactive terminal, ask once and download it
/// (so "download and run" is a single command); otherwise (a pipe, a script)
/// stay explicit and error with a `chord pull` hint — never a surprise download.
pub fn ensure(t: &dyn Transform, opts: &Options) -> Result<()> {
    // User-supplied hf: option values must be cached before the engine runs.
    for spec in t.options() {
        if let Some(value) = opts.get(spec.key) {
            if chord_hf::is_hf(value) && chord_hf::cached_path(value)?.is_none() {
                offer_download(value, value, &format!("run `chord pull --model {value}`"))?;
            }
        }
    }
    // Engine-declared defaults (R8: the engine declares, the host fetches).
    let hint = format!("run `chord pull {}`", t.name());
    for r in pending(t.resources(), opts, is_present) {
        offer_download(r.spec, r.describe, &hint)?;
    }
    Ok(())
}

/// Which of `resources` actually need fetching: those whose overriding
/// option is unset (the engine default is in play) and whose spec isn't
/// already present locally.
fn pending<'a>(
    resources: &'a [ResourceSpec],
    opts: &Options,
    present: impl Fn(&str) -> bool,
) -> Vec<&'a ResourceSpec> {
    resources
        .iter()
        .filter(|r| opts.get(r.key).is_none() && !present(r.spec))
        .collect()
}

/// True when `spec` is already fetched: an `hf:` ref in the shared HF cache,
/// or a URL whose basename exists in `models_dir()`.
fn is_present(spec: &str) -> bool {
    if chord_hf::is_hf(spec) {
        return matches!(chord_hf::cached_path(spec), Ok(Some(_)));
    }
    match url_file_name(spec) {
        Some(name) => dirs::models_dir().join(name).exists(),
        None => false,
    }
}

/// The basename a URL download lands under in `models_dir()`.
fn url_file_name(url: &str) -> Option<&str> {
    url.rsplit('/').next().filter(|s| !s.is_empty())
}

/// Fetch one resource by scheme: `hf:` into the shared HF cache, `https:`
/// into `models_dir()/<basename>`.
fn fetch(spec: &str, label: &str) -> Result<()> {
    if chord_hf::is_hf(spec) {
        let path = chord_hf::download(spec)?;
        eprintln!("ready: {}", path.display());
        return Ok(());
    }
    if spec.starts_with("https://") {
        let name = url_file_name(spec).ok_or_else(|| {
            ChordError::BadInput(format!("resource URL has no file name: {spec}"))
        })?;
        return download(spec, dirs::models_dir().join(name), label);
    }
    Err(ChordError::BadInput(format!(
        "unsupported resource scheme (expected hf: or https:): {spec}"
    ))
    .into())
}

fn offer_download(spec: &str, label: &str, hint: &str) -> Result<()> {
    // Prompt on the controlling terminal (not stdin/stdout — those carry data).
    if let Ok(tty) = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open("/dev/tty")
    {
        let mut w = &tty;
        write!(
            w,
            "chord: {label} ({spec})\n      isn't downloaded yet. Download it now? [Y/n] "
        )?;
        w.flush()?;
        let mut answer = String::new();
        BufReader::new(&tty).read_line(&mut answer)?;
        let a = answer.trim().to_lowercase();
        if a.is_empty() || a == "y" || a == "yes" {
            return fetch(spec, label);
        }
        return Err(ChordError::ModelMissing {
            what: label.to_string(),
            hint: "download declined".to_string(),
        }
        .into());
    }
    // No terminal: keep it explicit and scriptable.
    Err(ChordError::ModelMissing {
        what: label.to_string(),
        hint: hint.to_string(),
    }
    .into())
}

/// Dispatch `chord pull`: either an explicit `--model hf:…` reference, or a
/// transform's manifest-declared resources.
pub fn run(m: &clap::ArgMatches) -> Result<()> {
    if let Some(spec) = m.get_one::<String>("model") {
        if chord_hf::is_hf(spec) {
            let path = chord_hf::download(spec)?;
            eprintln!("ready: {}", path.display());
            return Ok(());
        }
        return Err(ChordError::BadInput(format!(
            "--model expects an hf: reference (e.g. hf:org/repo:Q4_K_M); got {spec:?}"
        ))
        .into());
    }
    match m.get_one::<String>("transform").map(String::as_str) {
        Some(t) => pull_declared(t),
        None => Err(ChordError::BadInput(
            "nothing to pull: give a transform (e.g. `chord pull stt`) or `--model hf:org/repo:QUANT`"
                .to_string(),
        )
        .into()),
    }
}

/// Fetch every resource `transform` declared in its manifest.
fn pull_declared(transform: &str) -> Result<()> {
    let Some(proxy) = crate::proxy::resolve(transform) else {
        return Err(ChordError::BadInput(format!(
            "unknown transform {transform:?} (see `chord ls`)"
        ))
        .into());
    };
    let resources = Transform::resources(&proxy);
    if resources.is_empty() {
        return Err(ChordError::Engine(format!(
            "{transform} declares no fetchable resources — it downloads on demand \
             or takes a user-supplied model (see `chord {transform} --help`)"
        ))
        .into());
    }
    for r in resources {
        if is_present(r.spec) {
            eprintln!("already present: {}", r.spec);
            continue;
        }
        fetch(r.spec, r.describe)?;
    }
    Ok(())
}

/// Stream `url` to `dest` with a progress bar. Writes to a `.part` file and
/// renames on success, so an interrupted download never looks complete.
fn download(url: &str, dest: PathBuf, label: &str) -> Result<()> {
    if dest.exists() {
        eprintln!("already present: {}", dest.display());
        return Ok(());
    }
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)?;
    }
    eprintln!("downloading {label}\n  -> {}", dest.display());

    let resp = ureq::get(url)
        .call()
        .map_err(|e| ChordError::Engine(format!("download failed: {e}")))?;
    let total: u64 = resp
        .header("Content-Length")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.cyan} {bytes}/{total_bytes} [{bar:30}] {bytes_per_sec}",
        )
        .unwrap(),
    );

    let mut reader = resp.into_reader();
    let tmp = dest.with_extension("part");
    let mut file = fs::File::create(&tmp)?;
    let mut buf = [0u8; 65536];
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])?;
        pb.inc(n as u64);
    }
    pb.finish_and_clear();
    fs::rename(&tmp, &dest)?;
    eprintln!("saved {}", dest.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use chord_core::ResourceSpec;

    const RES: &[ResourceSpec] = &[
        ResourceSpec {
            key: "model",
            spec: "hf:org/repo:a.onnx",
            describe: "a",
        },
        ResourceSpec {
            key: "assets",
            spec: "https://example.com/voices.tar",
            describe: "v",
        },
    ];

    #[test]
    fn unset_and_absent_resources_are_pending() {
        let p = pending(RES, &Options::new(), |_| false);
        assert_eq!(p.len(), 2);
    }

    #[test]
    fn an_overridden_resource_is_not_fetched() {
        // The user supplied their own --model; its default is not ours to pull.
        let mut opts = Options::new();
        opts.insert("model", "/my/own.onnx");
        let p = pending(RES, &opts, |_| false);
        assert_eq!(p.len(), 1);
        assert_eq!(p[0].key, "assets");
    }

    #[test]
    fn already_present_resources_are_skipped() {
        let p = pending(RES, &Options::new(), |spec| spec.starts_with("hf:"));
        assert_eq!(p.len(), 1);
        assert_eq!(p[0].spec, "https://example.com/voices.tar");
    }

    #[test]
    fn url_file_name_extracts_the_basename() {
        assert_eq!(
            url_file_name("https://x.test/a/b/model.bin"),
            Some("model.bin")
        );
        assert_eq!(url_file_name("https://x.test/a/"), None);
    }
}
