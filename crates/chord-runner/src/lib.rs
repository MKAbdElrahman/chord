//! chord-runner — drive a [`Transform`] as a standalone Unix-filter binary.
//!
//! Every engine ships as its own executable (so incompatible native libraries
//! never share one address space, and each can stream/crash independently). The
//! engine still implements the kernel's [`Transform`] contract; this helper
//! turns that contract into a CLI: a positional `input` file (or stdin) plus a
//! named flag per declared option, writing the result to stdout.
//!
//! ```ignore
//! fn main() -> std::process::ExitCode {
//!     chord_runner::run(&chord_stt::Stt)
//! }
//! ```
//!
//! The `chord` host doesn't link these binaries; it spawns them (see the
//! exec-proxy in `chord-cli`). Flag names here therefore mirror the options the
//! proxy forwards.

use std::io::{self, IsTerminal};
use std::process::ExitCode;

use chord_core::events;
use chord_core::{ChordError, Manifest, Options, Transform};
use clap::{Arg, ArgAction, Command};

/// Run `t` as a filter: parse argv, read the input file (or stdin), apply, and
/// write to stdout. Returns a process exit code.
pub fn run(t: &dyn Transform) -> ExitCode {
    // Self-description protocol: when the host asks `chord-<name> --chord-manifest`,
    // print this engine's metadata as JSON and exit. This is how the host
    // discovers the plug-in, so the engine's own `Transform` impl is the single
    // source of truth for its name, kinds, backend, and options.
    if std::env::args().any(|a| a == "--chord-manifest") {
        match serde_json::to_string(&Manifest::of(t)) {
            Ok(json) => {
                println!("{json}");
                return ExitCode::SUCCESS;
            }
            Err(e) => {
                eprintln!("chord-{}: manifest: {e}", t.name());
                return ExitCode::FAILURE;
            }
        }
    }

    init_tracing();

    let matches = build_command(t).get_matches();
    let jsonl = matches.get_one::<String>("format").map(String::as_str) == Some("jsonl");

    if jsonl {
        emit(&events::Start::new(t.name()));
    }

    let mut opts = Options::new();
    for spec in t.options() {
        if spec.takes_value {
            if let Some(v) = matches.get_one::<String>(spec.key) {
                opts.insert(spec.key, v.clone());
            }
        } else if matches.get_flag(spec.key) {
            opts.insert(spec.key, "true");
        }
    }

    let stdout = io::stdout();
    let mut out = stdout.lock();
    // A raw (unframed) input decodes to a single part of this transform's
    // primary input kind; the singleton-raw rule keeps mono-modal pipes clean.
    let default_kind = t.signature().primary_in();

    let result = (|| -> chord_core::Result<()> {
        let input = match matches.get_one::<String>("input") {
            Some(path) if path != "-" => {
                let mut f = std::fs::File::open(path)
                    .map_err(|e| -> chord_core::Error { format!("opening {path}: {e}").into() })?;
                chord_core::decode(&mut f, default_kind)?
            }
            _ => {
                let stdin = io::stdin();
                let mut input = stdin.lock();
                chord_core::decode(&mut input, default_kind)?
            }
        };
        let output = t.apply(input, &opts)?;
        chord_core::encode(&output, &mut out)?;
        Ok(())
    })();

    match result {
        Ok(()) => {
            if jsonl {
                emit(&events::Done::new(t.name()));
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            // Categorized errors carry a meaningful exit code (2 bad input,
            // 3 missing model, …); anything else is a generic failure (1).
            let ce = e.downcast_ref::<ChordError>();
            let code = ce.map(ChordError::exit_code).unwrap_or(1);
            if jsonl {
                emit(&events::Error::new(
                    t.name(),
                    code,
                    ce.map(ChordError::kind).unwrap_or("engine"),
                    e.to_string(),
                ));
            } else {
                eprintln!("chord-{}: {e}", t.name());
            }
            ExitCode::from(code as u8)
        }
    }
}

/// Initialize structured logging once: engine/native-library diagnostics go to
/// stderr, filtered by `RUST_LOG` (default: quiet), and ANSI is used only when
/// stderr is a terminal so pipes stay clean. Idempotent across engines.
fn init_tracing() {
    use tracing_subscriber::{fmt, EnvFilter};
    let _ = fmt()
        .with_writer(io::stderr)
        .with_ansi(io::stderr().is_terminal())
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn")),
        )
        .try_init();
}

/// Emit one NDJSON event on stderr (the machine-readable status channel; the
/// data plane stays on stdout).
fn emit(event: &impl serde::Serialize) {
    if let Ok(line) = serde_json::to_string(event) {
        eprintln!("{line}");
    }
}

/// Build the clap command: program name `chord-<name>`, a positional input, and
/// a flag per declared option (value flag, or boolean for `takes_value = false`).
fn build_command(t: &dyn Transform) -> Command {
    // clap needs a 'static program name; this binary lives for the whole run,
    // so leaking one small string is fine.
    let name: &'static str = Box::leak(format!("chord-{}", t.name()).into_boxed_str());
    let mut cmd = Command::new(name).about(t.describe().to_owned()).arg(
        Arg::new("input")
            .index(1)
            .help("input file (default: stdin)")
            .value_parser(clap::value_parser!(String)),
    );
    for spec in t.options() {
        let mut arg = Arg::new(spec.key).long(spec.key).help(spec.help);
        if spec.takes_value {
            arg = arg.value_parser(clap::value_parser!(String));
        } else {
            arg = arg.action(ArgAction::SetTrue);
        }
        cmd = cmd.arg(arg);
    }
    cmd.arg(
        Arg::new("format")
            .long("format")
            .value_parser(["text", "jsonl"])
            .default_value("text")
            .help("output format: text, or jsonl for lifecycle/error events on stderr"),
    )
}
