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

use std::io::{self, Cursor, IsTerminal, Read, Write};
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
        let mut manifest = Manifest::of(t);
        // The per-message loop lives in this runner, so every engine built
        // with it can serve a delimited multi-message stream.
        manifest.caps.push("each".to_string());
        match serde_json::to_string(&manifest) {
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
    let each = matches.get_flag("each");

    let started = std::time::Instant::now();
    let result = (|| -> chord_core::Result<FilterStats> {
        match matches.get_one::<String>("input") {
            Some(path) if path != "-" => {
                let mut f = std::fs::File::open(path)
                    .map_err(|e| -> chord_core::Error { format!("opening {path}: {e}").into() })?;
                if each {
                    run_each(t, &mut f, &mut out, &opts)
                } else {
                    filter(t, &mut f, &mut out, &opts)
                }
            }
            _ => {
                let stdin = io::stdin();
                let mut input = stdin.lock();
                if each {
                    run_each(t, &mut input, &mut out, &opts)
                } else {
                    filter(t, &mut input, &mut out, &opts)
                }
            }
        }
    })();
    let duration_ms = started.elapsed().as_millis() as u64;

    match result {
        Ok(stats) => {
            if jsonl {
                emit(&events::Done::new(
                    t.name(),
                    duration_ms,
                    stats.bytes_in,
                    stats.bytes_out,
                ));
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            // Categorized errors carry a meaningful exit code (2 bad input,
            // 3 missing model, …); anything else is a generic failure (1).
            let ce = e.downcast_ref::<ChordError>();
            let code = ce.map(ChordError::exit_code).unwrap_or(1);
            if jsonl {
                emit(
                    &events::Error::new(
                        t.name(),
                        code,
                        ce.map(ChordError::kind).unwrap_or("engine"),
                        e.to_string(),
                    )
                    .with_duration(duration_ms),
                );
            } else {
                eprintln!("chord-{}: {e}", t.name());
            }
            ExitCode::from(code as u8)
        }
    }
}

/// Exact byte counts from one [`filter`] run — the per-item measurements
/// Little's finite-window theorem turns into λ, W, and L for a stage.
#[derive(Debug, Clone, Copy, Default)]
pub struct FilterStats {
    pub bytes_in: u64,
    pub bytes_out: u64,
}

/// Counts bytes flowing through a reader.
struct CountingReader<R> {
    inner: R,
    count: u64,
}

impl<R: Read> Read for CountingReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.count += n as u64;
        Ok(n)
    }
}

/// Counts bytes flowing through a writer.
struct CountingWriter<W> {
    inner: W,
    count: u64,
}

impl<W: Write> Write for CountingWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.count += n as u64;
        Ok(n)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

/// Run `t` as a persistent worker over a delimited multi-message stream:
/// read messages until clean EOF, applying each and writing its delimited
/// reply **immediately** (flushed per message, so downstream stages see
/// items as they finish, not at process exit). Returns the message count.
///
/// This is the model-load amortization seam (rule R5 in
/// `docs/theory/THEORY.md`): whatever an engine loads on the first `apply`
/// stays in process memory for all subsequent ones — Friedman-Wise
/// memoization made structural.
pub fn filter_each(
    t: &dyn Transform,
    input: &mut dyn Read,
    output: &mut dyn Write,
    opts: &Options,
) -> chord_core::Result<u64> {
    let mut n = 0;
    while let Some(msg) = chord_core::read_delimited(input)? {
        let reply = t.apply(msg, opts)?;
        chord_core::write_delimited(&reply, output)?;
        output.flush()?;
        n += 1;
    }
    Ok(n)
}

/// [`filter_each`] with byte accounting, for the `--format jsonl` Done event.
fn run_each(
    t: &dyn Transform,
    input: &mut dyn Read,
    output: &mut dyn Write,
    opts: &Options,
) -> chord_core::Result<FilterStats> {
    let mut input = CountingReader {
        inner: input,
        count: 0,
    };
    let mut output = CountingWriter {
        inner: output,
        count: 0,
    };
    filter_each(t, &mut input, &mut output, opts)?;
    Ok(FilterStats {
        bytes_in: input.count,
        bytes_out: output.count,
    })
}

/// Run `t` over one input stream: peek the frame discriminator, then either
/// hand a raw stream to the transform's streaming path ([`Transform::apply_raw`])
/// or decode a framed message (buffered). The glue holds only the
/// discriminator prefix, never the stream — rule R2 in `docs/theory/THEORY.md`.
/// Returns exact byte counts for `--format jsonl` instrumentation.
pub fn filter(
    t: &dyn Transform,
    input: &mut dyn Read,
    output: &mut dyn Write,
    opts: &Options,
) -> chord_core::Result<FilterStats> {
    let mut input = CountingReader {
        inner: input,
        count: 0,
    };
    let mut output = CountingWriter {
        inner: output,
        count: 0,
    };

    // Peek exactly enough leading bytes to tell framed from raw. EOF before
    // a full magic means the stream is raw (or empty).
    let mut head = [0u8; chord_core::MAGIC_LEN];
    let mut n = 0;
    while n < head.len() {
        match input.read(&mut head[n..]) {
            Ok(0) => break,
            Ok(r) => n += r,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e.into()),
        }
    }
    let mut rest = Cursor::new(head[..n].to_vec()).chain(&mut input);
    if n == 0 || chord_core::is_framed(&head[..n]) {
        // Framed (multi-part) — or empty, which must stay an empty Message so
        // source transforms keep their flags-only behavior. Buffered path:
        // decode the whole message, apply, encode.
        let msg = chord_core::decode(&mut rest, t.signature().primary_in())?;
        let out = t.apply(msg, opts)?;
        chord_core::encode(&out, &mut output)?;
    } else {
        // Raw singleton: stream straight through. Unary engines never buffer
        // in the glue; whole-message transforms fall back to the buffered
        // default impl of apply_raw.
        t.apply_raw(&mut rest, &mut output, opts)?;
    }
    Ok(FilterStats {
        bytes_in: input.count,
        bytes_out: output.count,
    })
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
    .arg(
        Arg::new("each")
            .long("each")
            .action(ArgAction::SetTrue)
            .help("persistent-worker mode: serve a stream of delimited messages until EOF"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use chord_core::{encode, Kind, Message, Part, Result, Signature};
    use std::io::{Cursor, Read, Write};

    /// A streaming engine: uppercases whatever flows through.
    struct Upper;

    impl chord_core::Unary for Upper {
        fn name(&self) -> &str {
            "upper"
        }
        fn from(&self) -> Kind {
            Kind::Text
        }
        fn to(&self) -> Kind {
            Kind::Text
        }
        fn describe(&self) -> &str {
            "test: uppercase"
        }
        fn apply(&self, input: &mut dyn Read, output: &mut dyn Write, _o: &Options) -> Result<()> {
            let mut s = String::new();
            input.read_to_string(&mut s)?;
            output.write_all(s.to_uppercase().as_bytes())?;
            Ok(())
        }
    }

    /// A whole-message engine: replies with its input's part count.
    struct CountParts;

    impl Transform for CountParts {
        fn name(&self) -> &str {
            "count"
        }
        fn signature(&self) -> Signature {
            Signature::unary(Kind::Text, Kind::Text)
        }
        fn describe(&self) -> &str {
            "test: count parts"
        }
        fn apply(&self, input: Message, _o: &Options) -> Result<Message> {
            Ok(Message::one(Part::text(input.parts.len().to_string())))
        }
    }

    fn run_filter(t: &dyn Transform, input: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        filter(
            t,
            &mut Cursor::new(input.to_vec()),
            &mut out,
            &Options::new(),
        )
        .unwrap();
        out
    }

    #[test]
    fn raw_input_takes_the_streaming_path() {
        assert_eq!(run_filter(&Upper, b"hello"), b"HELLO");
    }

    #[test]
    fn raw_input_shorter_than_the_magic_still_works() {
        assert_eq!(run_filter(&Upper, b"hi"), b"HI");
    }

    #[test]
    fn framed_input_decodes_every_part() {
        let msg = Message {
            parts: vec![Part::text("a"), Part::text("b"), Part::text("c")],
        };
        let mut framed = Vec::new();
        encode(&msg, &mut framed).unwrap();
        assert_eq!(run_filter(&CountParts, &framed), b"3");
    }

    #[test]
    fn framed_singleton_reaches_a_unary_through_apply() {
        // A framed one-part message must still feed a Unary correctly
        // (via the buffered decode -> apply path, not apply_raw).
        let msg = Message {
            parts: vec![
                Part::text("ok").with_meta("origin", "test"),
                Part::text("go"),
            ],
        };
        let mut framed = Vec::new();
        encode(&msg, &mut framed).unwrap();
        // Two text parts -> Unary's single() contract rejects it cleanly.
        let mut out = Vec::new();
        let err = filter(&Upper, &mut Cursor::new(framed), &mut out, &Options::new()).unwrap_err();
        assert!(err.to_string().contains("expected one text part"), "{err}");
    }

    #[test]
    fn empty_input_is_an_empty_message_not_an_empty_part() {
        // Source transforms (pack) build output from flags alone; empty stdin
        // must reach apply as Message::empty().
        assert_eq!(run_filter(&CountParts, b""), b"0");
    }

    #[test]
    fn each_loop_processes_every_message_and_delimits_output() {
        // The persistent-worker loop: N delimited messages in, N delimited
        // messages out, on one process — the model-load amortization seam.
        use chord_core::{read_delimited, write_delimited};
        let mut input = Vec::new();
        for text in ["alpha", "beta", "gamma"] {
            write_delimited(&Message::one(Part::text(text)), &mut input).unwrap();
        }
        let mut out = Vec::new();
        let n = filter_each(&Upper, &mut Cursor::new(input), &mut out, &Options::new()).unwrap();
        assert_eq!(n, 3);

        let mut r = out.as_slice();
        let mut replies = Vec::new();
        while let Some(m) = read_delimited(&mut r).unwrap() {
            replies.push(String::from_utf8(m.parts[0].as_bytes().to_vec()).unwrap());
        }
        assert_eq!(replies, ["ALPHA", "BETA", "GAMMA"]);
    }

    #[test]
    fn each_loop_on_empty_stream_is_zero_messages() {
        let mut out = Vec::new();
        let n = filter_each(
            &Upper,
            &mut Cursor::new(Vec::new()),
            &mut out,
            &Options::new(),
        )
        .unwrap();
        assert_eq!(n, 0);
        assert!(out.is_empty());
    }

    #[test]
    fn filter_reports_exact_byte_counts_for_raw_input() {
        // Little's Law instrumentation: byte counts must be exact.
        let mut out = Vec::new();
        let stats = filter(
            &Upper,
            &mut Cursor::new(b"hello".to_vec()),
            &mut out,
            &Options::new(),
        )
        .unwrap();
        assert_eq!(stats.bytes_in, 5);
        assert_eq!(stats.bytes_out, 5);
    }

    #[test]
    fn filter_reports_exact_byte_counts_for_framed_input() {
        let msg = Message {
            parts: vec![Part::text("a"), Part::text("b")],
        };
        let mut framed = Vec::new();
        encode(&msg, &mut framed).unwrap();
        let framed_len = framed.len() as u64;

        let mut out = Vec::new();
        let stats = filter(
            &CountParts,
            &mut Cursor::new(framed),
            &mut out,
            &Options::new(),
        )
        .unwrap();
        assert_eq!(stats.bytes_in, framed_len);
        assert_eq!(stats.bytes_out, 1); // the "2" reply
    }
}
