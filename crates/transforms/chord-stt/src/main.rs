//! chord-stt binary — the speech-to-text engine as a standalone Unix filter.
//! The `chord` host spawns this out-of-process (see chord-cli's exec-proxy).
//!
//! A power-user `--batch` mode loads the whisper model ONCE and transcribes a
//! stdin list of `path` or `path<TAB>lang` lines (one transcript per line),
//! removing the per-call model-load cost. The normal single-stream filter path
//! goes through chord-runner (which also serves `--chord-manifest`).

use std::process::ExitCode;

use chord_core::ChordError;
use clap::{Arg, ArgAction, Command};

fn main() -> ExitCode {
    if std::env::args().any(|a| a == "--batch") {
        return run_batch();
    }
    chord_runner::run(&chord_stt::Stt)
}

/// Parse the `--batch` flags with clap and run batch transcription, adopting the
/// shared categorized exit codes (2 bad input, 3 missing model, else 1).
fn run_batch() -> ExitCode {
    let m = Command::new("chord-stt")
        .about("batch transcribe: load the model once, read `path` or `path<TAB>lang` lines from stdin")
        .arg(
            Arg::new("batch")
                .long("batch")
                .action(ArgAction::SetTrue)
                .help("enable batch mode"),
        )
        .arg(
            Arg::new("model")
                .long("model")
                .help("whisper model path or name (default large-v3-turbo)"),
        )
        .arg(
            Arg::new("lang")
                .long("lang")
                .help("default language code when a line omits one"),
        )
        .arg(
            Arg::new("threads")
                .long("threads")
                .help("CPU threads (default 4)"),
        )
        .get_matches();

    let model = m.get_one::<String>("model").cloned();
    let lang = m.get_one::<String>("lang").cloned();
    let threads: i32 = m
        .get_one::<String>("threads")
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);

    match chord_stt::run_batch(model, lang, threads) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            let code = e
                .downcast_ref::<ChordError>()
                .map(ChordError::exit_code)
                .unwrap_or(1);
            eprintln!("chord-stt: {e}");
            ExitCode::from(code as u8)
        }
    }
}
