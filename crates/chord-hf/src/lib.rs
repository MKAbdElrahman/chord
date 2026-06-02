//! Resolve `hf:` model references against the Hugging Face Hub.
//!
//! Reference form (mirrors llama.cpp's `-hf` and Ollama's `hf.co/...`):
//!
//! ```text
//! hf:<org>/<repo>[@<revision>][:<quant-or-file>]
//! hf:bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M     # quant tag
//! hf:ggml-org/whisper.cpp:ggml-base.en.bin           # explicit file
//! hf:org/repo                                         # default file
//! ```
//!
//! Uses HF's official `hf-hub` crate: `HF_TOKEN`/`HF_HOME` env vars, the shared
//! `~/.cache/huggingface` cache, and revisions all work. Downloads are explicit
//! (`chord pull`); engines only ever read the cache, never auto-download.

use std::path::PathBuf;

use chord_core::{ChordError, Result};
use hf_hub::api::sync::ApiBuilder;
use hf_hub::{Cache, Repo, RepoType};

/// True if `spec` is an `hf:` reference (vs. a plain filesystem path).
pub fn is_hf(spec: &str) -> bool {
    spec.starts_with("hf:")
}

struct HfRef {
    repo: String,
    revision: String,
    selector: Selector,
}

enum Selector {
    /// An explicit filename within the repo.
    File(String),
    /// A quantization tag (e.g. `Q4_K_M`) matched against `.gguf` filenames.
    Quant(String),
    /// Pick a sensible default `.gguf`.
    Default,
}

fn parse(spec: &str) -> Result<HfRef> {
    let rest = spec
        .strip_prefix("hf:")
        .ok_or_else(|| ChordError::BadInput(format!("not an hf: reference: {spec}")))?;

    // The selector is the segment after the last ':' (the repo id has no ':').
    let (repo_part, selector) = match rest.rsplit_once(':') {
        Some((r, sel)) if !sel.is_empty() && !sel.contains('/') => (r, Some(sel.to_string())),
        _ => (rest, None),
    };
    // Optional `@revision`.
    let (repo, revision) = match repo_part.split_once('@') {
        Some((r, rev)) => (r.to_string(), rev.to_string()),
        None => (repo_part.to_string(), "main".to_string()),
    };
    if !repo.contains('/') {
        return Err(
            ChordError::BadInput(format!("hf reference needs <org>/<repo>: {spec}")).into(),
        );
    }

    let selector = match selector {
        None => Selector::Default,
        // A selector with a file extension is an explicit filename (e.g.
        // `encoder.int8.onnx`, `tokens.txt`, `model.gguf`); a bare tag like
        // `Q4_K_M` is a quantization to match against the repo's `.gguf` files.
        Some(s) if s.contains('.') => Selector::File(s),
        Some(s) => Selector::Quant(s),
    };
    Ok(HfRef {
        repo,
        revision,
        selector,
    })
}

fn repo_of(r: &HfRef) -> Repo {
    Repo::with_revision(r.repo.clone(), RepoType::Model, r.revision.clone())
}

/// Choose a filename from the repo's file list given the selector.
fn pick_file(names: &[String], selector: &Selector) -> Result<String> {
    if let Selector::File(f) = selector {
        return Ok(f.clone());
    }
    let ggufs: Vec<&String> = names
        .iter()
        .filter(|f| f.to_lowercase().ends_with(".gguf"))
        .collect();
    let chosen = match selector {
        Selector::Quant(q) => {
            let needle = q.to_lowercase();
            ggufs
                .iter()
                .find(|f| f.to_lowercase().contains(&needle))
                .copied()
        }
        // Default: prefer a Q4_K_M build, else the first .gguf.
        _ => ggufs
            .iter()
            .find(|f| f.to_lowercase().contains("q4_k_m"))
            .copied()
            .or_else(|| ggufs.first().copied()),
    };
    chosen.cloned().ok_or_else(|| {
        let avail: Vec<&str> = ggufs.iter().map(|s| s.as_str()).collect();
        ChordError::ModelMissing {
            what: "a matching .gguf".to_string(),
            hint: if avail.is_empty() {
                "the repo has no .gguf files".to_string()
            } else {
                format!("available files: {}", avail.join(", "))
            },
        }
        .into()
    })
}

/// Resolve the target filename, querying repo metadata if the selector is a
/// quant tag or the default (an explicit file needs no network).
fn resolve_filename(r: &HfRef) -> Result<String> {
    if let Selector::File(f) = &r.selector {
        return Ok(f.clone());
    }
    let api = ApiBuilder::from_env()
        .build()
        .map_err(|e| ChordError::Engine(format!("hf api init: {e}")))?;
    let info = api
        .repo(repo_of(r))
        .info()
        .map_err(|e| ChordError::Engine(format!("hf repo info {}: {e}", r.repo)))?;
    let names: Vec<String> = info.siblings.into_iter().map(|s| s.rfilename).collect();
    pick_file(&names, &r.selector)
}

/// Resolve a model spec to a usable path: an `hf:` reference → its cached path
/// (erroring with a `chord pull` hint if not yet downloaded); any other string
/// → a filesystem path, unchanged. The one call every engine uses.
pub fn resolve(spec: &str) -> Result<PathBuf> {
    if is_hf(spec) {
        cached_path(spec)?.ok_or_else(|| {
            ChordError::ModelMissing {
                what: format!("HF model {spec}"),
                hint: format!("run `chord pull --model {spec}`"),
            }
            .into()
        })
    } else {
        Ok(PathBuf::from(spec))
    }
}

/// Resolve an `hf:` spec to a local cached path **without downloading**.
/// `None` means it isn't in the cache yet (the caller should suggest `chord pull`).
pub fn cached_path(spec: &str) -> Result<Option<PathBuf>> {
    let r = parse(spec)?;
    let file = resolve_filename(&r)?;
    Ok(Cache::from_env().repo(repo_of(&r)).get(&file))
}

/// Download an `hf:` spec (with hf-hub's progress bar) and return the local path.
pub fn download(spec: &str) -> Result<PathBuf> {
    let r = parse(spec)?;
    let api = ApiBuilder::from_env()
        .build()
        .map_err(|e| ChordError::Engine(format!("hf api init: {e}")))?;
    let repo = api.repo(repo_of(&r));
    let file = match &r.selector {
        Selector::File(f) => f.clone(),
        _ => {
            let info = repo
                .info()
                .map_err(|e| ChordError::Engine(format!("hf repo info {}: {e}", r.repo)))?;
            let names: Vec<String> = info.siblings.into_iter().map(|s| s.rfilename).collect();
            pick_file(&names, &r.selector)?
        }
    };
    eprintln!("pulling {}/{} …", r.repo, file);
    repo.get(&file)
        .map_err(|e| ChordError::Engine(format!("hf download {}/{}: {e}", r.repo, file)).into())
}
