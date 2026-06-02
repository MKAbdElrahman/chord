//! `chord pull <transform>` — fetch a transform's default model.
//!
//! Explicit by design: engine models range from ~1 GB (whisper) to 20-30 GB
//! (the chat/see GGUFs), so chord never downloads silently. A missing model
//! errors with a hint to run this command.
//!
//! Only engines with a single canonical, public download are wired here. The
//! large local GGUFs (chat/see) and the Supertonic asset bundle (tts) are
//! user-supplied — for those, point the engine at a path via `--model`/config.

use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::PathBuf;

use chord_core::{ChordError, Options, Result, Transform};
use indicatif::{ProgressBar, ProgressStyle};

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap_or_default())
}

/// Make every `hf:` model option a transform needs present before it runs.
///
/// If one is missing: at an interactive terminal, ask once and download it
/// (so "download and run" is a single command); otherwise (a pipe, a script)
/// stay explicit and error with a `chord pull` hint — never a surprise download.
pub fn ensure(t: &dyn Transform, opts: &Options) -> Result<()> {
    for spec in t.options() {
        if let Some(value) = opts.get(spec.key) {
            if chord_hf::is_hf(value) && chord_hf::cached_path(value)?.is_none() {
                offer_download(value)?;
            }
        }
    }
    Ok(())
}

fn offer_download(spec: &str) -> Result<()> {
    // Prompt on the controlling terminal (not stdin/stdout — those carry data).
    if let Ok(tty) = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open("/dev/tty")
    {
        let mut w = &tty;
        write!(
            w,
            "chord: {spec}\n      isn't downloaded yet. Download it now? [Y/n] "
        )?;
        w.flush()?;
        let mut answer = String::new();
        BufReader::new(&tty).read_line(&mut answer)?;
        let a = answer.trim().to_lowercase();
        if a.is_empty() || a == "y" || a == "yes" {
            chord_hf::download(spec)?;
            return Ok(());
        }
        return Err(ChordError::ModelMissing {
            what: format!("HF model {spec}"),
            hint: "download declined".to_string(),
        }
        .into());
    }
    // No terminal: keep it explicit and scriptable.
    Err(ChordError::ModelMissing {
        what: format!("HF model {spec}"),
        hint: format!("run `chord pull --model {spec}`"),
    }
    .into())
}

/// Dispatch `chord pull`: either an explicit `--model hf:…` reference, or a
/// transform's built-in default.
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
        Some(t) => pull_default(t),
        None => Err(ChordError::BadInput(
            "nothing to pull: give a transform (e.g. `chord pull stt`) or `--model hf:org/repo:QUANT`"
                .to_string(),
        )
        .into()),
    }
}

/// Fetch the built-in default model for `transform`.
fn pull_default(transform: &str) -> Result<()> {
    match transform {
        "stt" => download(
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin",
            home().join("models").join("ggml-large-v3-turbo.bin"),
            "whisper large-v3-turbo (~1.5 GB)",
        ),
        "draw" => {
            eprintln!("draw downloads its model automatically on first use — just run `chord draw \"…\"`.");
            Ok(())
        }
        "chat" | "see" | "tts" => Err(ChordError::Engine(format!(
            "no built-in download for `{transform}`: point it at a local model \
             (chat/see: --model <GGUF> or config; tts: --assets <dir>)"
        ))
        .into()),
        other => Err(ChordError::BadInput(format!(
            "unknown transform {other:?} (try: stt, tts, chat, see, draw)"
        ))
        .into()),
    }
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
