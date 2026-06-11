//! chord CLI — the host and composition root.
//!
//! This is the only crate that knows the concrete set of plug-ins. It builds a
//! [`Registry`], turns each registered [`Transform`] into a Unix-filter
//! subcommand, and dispatches. Composition between transforms is the shell
//! pipe's job: `cat q.wav | chord stt | chord chat | chord tts > a.wav`.

mod config;
mod discover;
mod proxy;
mod pull;

use std::fs::File;
use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process::{exit, Child, ChildStdout, Stdio};

use chord_core::{ChordError, Kind, Registry, Result, Signature, Transform};
use clap::{Arg, ArgAction, ArgMatches, Command};

use config::Config;

/// Build the registry — the composition root. Every engine runs out-of-process:
/// it ships as its own `chord-<name>` binary, *discovered* on disk and described
/// by its own `--chord-manifest` (see `discover.rs`). The host hardcodes no
/// engine list and links no engine code (which also sidesteps native-library
/// symbol clashes, e.g. the two copies of ggml). See proxy.rs.
fn build_registry() -> Registry {
    let mut reg = Registry::new();
    for proxy in proxy::default_proxies() {
        reg.register(Box::new(proxy));
    }
    reg
}

fn main() {
    let reg = build_registry();
    let matches = build_cli(&reg).get_matches();

    let config = match Config::load(matches.get_one::<String>("config").map(String::as_str)) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("chord: config: {e}");
            exit(1);
        }
    };

    let result = match matches.subcommand() {
        Some(("ls", _)) => {
            print_ls(&reg);
            Ok(())
        }
        Some(("kinds", _)) => {
            print_kinds();
            Ok(())
        }
        Some(("config", _)) => {
            print_config(&config);
            Ok(())
        }
        Some(("pull", sub)) => pull::run(sub),
        Some(("pipeline", sub)) => run_pipeline(sub, &config),
        Some((name, sub)) => match sub.get_one::<String>("backend").map(String::as_str) {
            // An explicit --backend: use an alternate engine if one exists, else
            // the default if it already is that backend.
            Some(b) => {
                if let Some(alt) = proxy::alternate(name, b) {
                    run_filter(&alt, sub, &config)
                } else if let Some(t) = reg.get(name).filter(|t| t.backend() == b) {
                    run_filter(t, sub, &config)
                } else {
                    eprintln!("chord: {name} has no backend {b:?} (see `chord ls`)");
                    exit(2);
                }
            }
            None => match reg.get(name) {
                Some(t) => run_filter(t, sub, &config),
                None => {
                    eprintln!("chord: unknown command: {name}");
                    exit(2);
                }
            },
        },
        None => {
            eprintln!("chord: no command (try `chord --help`)");
            exit(2);
        }
    };

    if let Err(err) = result {
        // Adopt a categorized exit code when present. A `Child` error means a
        // spawned engine already printed the message, so don't repeat it.
        let (code, reported) = err
            .downcast_ref::<chord_core::ChordError>()
            .map(|c| (c.exit_code(), c.already_reported()))
            .unwrap_or((1, false));
        if !reported {
            eprintln!("chord: {err}");
        }
        exit(code);
    }
}

/// Assemble the CLI: static `ls`/`kinds` plus one filter subcommand per plug-in.
fn build_cli(reg: &Registry) -> Command {
    let mut cmd = Command::new("chord")
        .about("Compose AI models as Unix filters (format in -> format out)")
        .subcommand_required(true)
        .arg_required_else_help(true)
        .arg(
            Arg::new("config")
                .short('c')
                .long("config")
                .global(true)
                .value_name("FILE")
                .value_parser(clap::value_parser!(String))
                .help("config YAML (default: $CHORD_CONFIG, ./chord.yaml, ~/.config/chord/config.yaml)"),
        )
        .arg(
            Arg::new("format")
                .long("format")
                .global(true)
                .value_parser(["text", "jsonl"])
                .default_value("text")
                .help("output format: text (default), or jsonl for machine-readable lifecycle/error events on stderr"),
        )
        .arg(
            Arg::new("backend")
                .long("backend")
                .global(true)
                .value_parser(clap::value_parser!(String))
                .help("select a transform's inference backend (see `chord ls`); default if omitted"),
        )
        .subcommand(Command::new("ls").about("List available transforms"))
        .subcommand(
            Command::new("pull")
                .about("Download a model: a transform's default, or any hf: reference")
                .arg(
                    Arg::new("transform")
                        .required(false)
                        .help("a transform whose default model to fetch (e.g. stt)"),
                )
                .arg(
                    Arg::new("model")
                        .long("model")
                        .help("a Hugging Face reference to download, e.g. hf:org/repo:Q4_K_M"),
                ),
        )
        .subcommand(Command::new("kinds").about("List the data kinds chord understands"))
        .subcommand(Command::new("config").about("Show the resolved configuration"))
        .subcommand(
            Command::new("pipeline")
                .about("Run several transforms in one process; separate stages with '::'")
                .long_about(
                    "Run a chain of transforms in one process. Separate stages with '::':\n\
                     \n  chord pipeline see --prompt \"what is this\" :: chat --system \"translate to german\" :: tts\n\
                     \nThe first stage reads its file arg or stdin; each later stage reads the\nprevious stage's output; the last writes stdout. Adjacent stages must\nconnect by kind: each must emit a kind the next accepts (`chord ls`\nshows every transform's signature).",
                )
                .arg(
                    Arg::new("stages")
                        .help("stage transforms and flags, separated by '::'")
                        .num_args(1..)
                        .trailing_var_arg(true)
                        .allow_hyphen_values(true)
                        .value_parser(clap::value_parser!(String)),
                ),
        );

    for t in reg.all() {
        cmd = cmd.subcommand(transform_command(t));
    }
    cmd
}

/// Build the clap subcommand for one transform: a positional input plus a named
/// flag for each option the transform declares.
fn transform_command(t: &dyn Transform) -> Command {
    // clap needs 'static strings; the registry lives for the whole program, so
    // leaking these few small strings is fine in the host.
    let name: &'static str = Box::leak(t.name().to_owned().into_boxed_str());
    let mut cmd = Command::new(name)
        .about(format!("{}  ({})", t.describe(), t.signature().display()))
        .arg(
            Arg::new("input")
                .help("input file, or inline text for text inputs (default: stdin)")
                .value_parser(clap::value_parser!(String))
                .index(1),
        );
    // Accept options from the default backend AND any alternate backends, so
    // `--backend X` flags parse regardless of which engine runs. Each engine
    // reads only the keys it declares; dedup by key.
    let alternates = proxy::alternates(t.name());
    let mut seen = std::collections::HashSet::new();
    for spec in t
        .options()
        .iter()
        .chain(alternates.iter().flat_map(|p| p.options()))
    {
        if !seen.insert(spec.key) {
            continue;
        }
        let mut arg = Arg::new(spec.key).long(spec.key).help(spec.help);
        if spec.takes_value {
            arg = arg.value_parser(clap::value_parser!(String));
        } else {
            arg = arg.action(ArgAction::SetTrue);
        }
        cmd = cmd.arg(arg);
    }
    cmd
}

/// Build the Options for a transform: the config-file section (persistent
/// defaults), overridden by the named CLI flags.
fn opts_from(t: &dyn Transform, m: &ArgMatches, config: &Config) -> chord_core::Options {
    let mut opts = config.options_for(t.name());
    for spec in t.options() {
        if spec.takes_value {
            if let Some(v) = m.get_one::<String>(spec.key) {
                opts.insert(spec.key, v.clone());
            }
        } else if m.get_flag(spec.key) {
            opts.insert(spec.key, "true");
        }
    }
    // The global `--format` flag isn't an engine option; stash it under a
    // reserved key so the exec-proxy can forward it to the engine binary.
    // `try_get_one` (not `get_one`) because pipeline stages are parsed by a
    // command that doesn't define `format` — looking it up there would panic.
    if let Ok(Some(fmt)) = m.try_get_one::<String>("format") {
        if fmt != "text" {
            opts.insert("__format", fmt.clone());
        }
    }
    opts
}

/// Run a transform as a filter: read the file argument (or stdin), write stdout.
///
/// Options precedence: config-file section (lowest), then named CLI flags.
fn run_filter(t: &dyn Transform, m: &ArgMatches, config: &Config) -> Result<()> {
    let opts = opts_from(t, m, config);
    pull::ensure(t, &opts)?;

    let stdout = io::stdout();
    let mut out = stdout.lock();
    let default_kind = t.signature().primary_in();

    // Build the input message, run the transform, encode the output message.
    let input = match m.get_one::<String>("input") {
        Some(arg) if arg != "-" => {
            if std::path::Path::new(arg).is_file() {
                let mut f = File::open(arg)?;
                chord_core::decode(&mut f, default_kind)?
            } else if t.signature().accepts_kind(Kind::Text) {
                // Not a file, and this transform consumes text: treat the
                // argument as the literal input, so `chord chat "hi"` works.
                chord_core::Message::one(chord_core::Part::text(arg.clone()))
            } else {
                return Err(chord_core::ChordError::BadInput(format!(
                    "input file not found: {arg:?}"
                ))
                .into());
            }
        }
        _ => {
            let stdin = io::stdin();
            // No file arg and nothing piped in: don't block reading an interactive
            // terminal. Source transforms (e.g. `pack`) build their output from
            // flags alone; others surface a clean "no input" error.
            if stdin.is_terminal() {
                chord_core::Message::empty()
            } else {
                let mut input = stdin.lock();
                chord_core::decode(&mut input, default_kind)?
            }
        }
    };
    let output = t.apply(input, &opts)?;
    chord_core::encode(&output, &mut out)?;
    Ok(())
}

/// Reject a chain that can never type-check: each stage must emit at least one
/// kind its successor accepts ([`Signature::connects_to`]). Checked before any
/// model is pulled, so an impossible chain fails in milliseconds, not minutes.
fn check_pipeline_kinds(stages: &[(&str, Signature)]) -> Result<()> {
    for w in stages.windows(2) {
        let ((a_name, a_sig), (b_name, b_sig)) = (&w[0], &w[1]);
        if !a_sig.connects_to(b_sig) {
            let kinds = |ks: &[Kind]| ks.iter().map(Kind::as_str).collect::<Vec<_>>().join(",");
            return Err(ChordError::BadInput(format!(
                "pipeline: {a_name} emits {} but {b_name} accepts {} (see `chord ls`)",
                kinds(&a_sig.emits),
                kinds(&b_sig.accepts),
            ))
            .into());
        }
    }
    Ok(())
}

/// Run a multi-stage pipeline as a true streaming chain. Stages are separated by
/// `::`; each stage is `transform [flags…]`. Every stage's engine process is
/// spawned at once and wired with OS pipes (stage N's stdout *is* stage N+1's
/// stdin), so bytes flow kernel-to-kernel and the stages run concurrently —
/// exactly like a shell `a | b | c`. The first stage reads its file arg, literal
/// text, or our stdin; the last writes our stdout.
fn run_pipeline(m: &ArgMatches, config: &Config) -> Result<()> {
    let tokens: Vec<String> = m
        .get_many::<String>("stages")
        .map(|v| v.cloned().collect())
        .unwrap_or_default();

    // Split tokens into stages on the literal "::".
    let mut stages: Vec<Vec<String>> = Vec::new();
    let mut cur: Vec<String> = Vec::new();
    for tok in tokens {
        if tok == "::" {
            stages.push(std::mem::take(&mut cur));
        } else {
            cur.push(tok);
        }
    }
    stages.push(cur);
    if stages.iter().any(|s| s.is_empty()) {
        return Err("pipeline: empty stage (check the '::' separators)".into());
    }

    struct Stage {
        proxy: proxy::ExecProxy,
        opts: chord_core::Options,
        input: Option<String>,
    }

    let mut plan: Vec<Stage> = Vec::with_capacity(stages.len());
    for stage in &stages {
        let name = stage[0].as_str();
        let p =
            proxy::resolve(name).ok_or_else(|| format!("pipeline: unknown transform {name:?}"))?;
        let argv = std::iter::once(name.to_string()).chain(stage[1..].iter().cloned());
        let matches = transform_command(&p)
            .try_get_matches_from(argv)
            .map_err(|e| format!("pipeline stage {name:?}: {e}"))?;
        let opts = opts_from(&p, &matches, config);
        let input = matches.get_one::<String>("input").cloned();
        plan.push(Stage {
            proxy: p,
            opts,
            input,
        });
    }

    // The chain must type-check on kinds before anything heavier happens.
    let sigs: Vec<(&str, Signature)> = plan
        .iter()
        .map(|s| (s.proxy.name(), s.proxy.signature()))
        .collect();
    check_pipeline_kinds(&sigs)?;

    // Make sure every stage's models are present before the chain starts, so a
    // missing download is resolved up front (one prompt) rather than mid-run.
    for stage in &plan {
        pull::ensure(&stage.proxy, &stage.opts)?;
    }

    // Resolve the first stage's input source: a file arg, literal text (for a
    // text-input first stage), or our stdin.
    enum FirstInput {
        Stdin,
        File(PathBuf),
        Literal(Vec<u8>),
    }
    let first_input = match plan[0].input.as_deref() {
        Some(arg) if arg != "-" => {
            if Path::new(arg).is_file() {
                FirstInput::File(PathBuf::from(arg))
            } else if plan[0].proxy.signature().accepts_kind(Kind::Text) {
                FirstInput::Literal(arg.as_bytes().to_vec())
            } else {
                return Err(ChordError::BadInput(format!("input file not found: {arg:?}")).into());
            }
        }
        _ => FirstInput::Stdin,
    };

    // Spawn every stage, wiring each one's stdout into the next one's stdin.
    let last = plan.len() - 1;
    let mut children: Vec<Child> = Vec::with_capacity(plan.len());
    let mut prev_stdout: Option<ChildStdout> = None;
    let mut writer: Option<std::thread::JoinHandle<()>> = None;

    for (i, stage) in plan.iter().enumerate() {
        let mut cmd = stage.proxy.command(&stage.opts);

        if i == 0 {
            match &first_input {
                FirstInput::Stdin => {
                    cmd.stdin(Stdio::inherit());
                }
                FirstInput::File(p) => {
                    let f = File::open(p).map_err(|e| format!("opening {}: {e}", p.display()))?;
                    cmd.stdin(Stdio::from(f));
                }
                // Literal text is fed in after spawn, on a thread (below).
                FirstInput::Literal(_) => {
                    cmd.stdin(Stdio::piped());
                }
            }
        } else {
            cmd.stdin(Stdio::from(
                prev_stdout.take().expect("previous stage stdout"),
            ));
        }

        // The last stage writes straight to our stdout; the rest pipe onward.
        if i == last {
            cmd.stdout(Stdio::inherit());
        } else {
            cmd.stdout(Stdio::piped());
        }
        // stderr is inherited (default) so spinners/errors reach the terminal.

        let mut child = cmd
            .spawn()
            .map_err(|e| format!("cannot run pipeline stage {:?}: {e}", stage.proxy.name()))?;

        if i == 0 {
            if let FirstInput::Literal(bytes) = &first_input {
                let mut stdin = child.stdin.take().expect("piped stdin");
                let data = bytes.clone();
                writer = Some(std::thread::spawn(move || {
                    let _ = stdin.write_all(&data);
                    // stdin drops here, closing the pipe (EOF for the stage).
                }));
            }
        }
        if i != last {
            prev_stdout = child.stdout.take();
        }
        children.push(child);
    }

    // Wait for every stage, collecting the exit codes of those that failed.
    let mut failures: Vec<i32> = Vec::new();
    for child in &mut children {
        let status = child
            .wait()
            .map_err(|e| format!("waiting on pipeline stage: {e}"))?;
        if !status.success() {
            failures.push(status.code().unwrap_or(1));
        }
    }
    if let Some(handle) = writer {
        let _ = handle.join();
    }

    // Adopt one stage's code (the engine already wrote its own categorized
    // message to the inherited stderr, so don't repeat it). Prefer a *categorized*
    // failure (2 bad-input, 3 model-missing) over a generic 1 — when one stage
    // dies, an adjacent stage often fails with a generic broken-pipe error that
    // would otherwise mask the real cause; else take the first failure.
    let code = failures
        .iter()
        .copied()
        .find(|c| *c != 1)
        .or_else(|| failures.first().copied());
    match code {
        Some(code) => Err(ChordError::Child(code).into()),
        None => Ok(()),
    }
}

fn print_ls(reg: &Registry) {
    println!(
        "{:<8} {:<18} {:<22} DESCRIPTION",
        "NAME", "KINDS", "BACKEND"
    );
    for t in reg.all() {
        let backend = if t.backend().is_empty() {
            "-"
        } else {
            t.backend()
        };
        println!(
            "{:<8} {:<18} {:<22} {}",
            t.name(),
            t.signature().display(),
            backend,
            t.describe()
        );
    }
}

fn print_kinds() {
    for k in Kind::all() {
        println!("{k}");
    }
}

fn print_config(config: &Config) {
    match &config.path {
        Some(p) => println!("config: {}", p.display()),
        None => {
            println!("config: (none found — transforms use built-in defaults)");
            return;
        }
    }
    let sections = config.sections();
    if sections.is_empty() {
        println!("(empty)");
        return;
    }
    let mut names: Vec<&String> = sections.keys().collect();
    names.sort();
    for name in names {
        println!("\n[{name}]");
        let section = &sections[name];
        let mut keys: Vec<&String> = section.keys().collect();
        keys.sort();
        for k in keys {
            println!("  {k} = {}", section[k]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chord_core::Signature;

    fn sig(accepts: &[Kind], emits: &[Kind]) -> Signature {
        Signature::new(accepts.to_vec(), emits.to_vec())
    }

    #[test]
    fn compatible_chain_passes() {
        // stt :: chat :: tts — every joint shares a kind.
        let stages = [
            ("stt", sig(&[Kind::Audio], &[Kind::Text])),
            (
                "chat",
                sig(&[Kind::Text, Kind::Image, Kind::Audio], &[Kind::Text]),
            ),
            ("tts", sig(&[Kind::Text], &[Kind::Audio])),
        ];
        assert!(check_pipeline_kinds(&stages).is_ok());
    }

    #[test]
    fn incompatible_joint_is_bad_input_naming_both_stages() {
        // draw :: stt — image into an audio consumer can never work.
        let stages = [
            ("draw", sig(&[Kind::Text], &[Kind::Image])),
            ("stt", sig(&[Kind::Audio], &[Kind::Text])),
        ];
        let err = check_pipeline_kinds(&stages).unwrap_err();
        let ce = err
            .downcast_ref::<ChordError>()
            .expect("categorized BadInput error");
        assert_eq!(ce.exit_code(), 2);
        let msg = err.to_string();
        assert!(msg.contains("draw"), "missing stage name in: {msg}");
        assert!(msg.contains("stt"), "missing stage name in: {msg}");
        assert!(msg.contains("image"), "missing emitted kind in: {msg}");
        assert!(msg.contains("audio"), "missing accepted kind in: {msg}");
    }

    #[test]
    fn single_stage_has_no_joints_to_check() {
        let stages = [("stt", sig(&[Kind::Audio], &[Kind::Text]))];
        assert!(check_pipeline_kinds(&stages).is_ok());
    }
}
