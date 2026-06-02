//! chord CLI — the host and composition root.
//!
//! This is the only crate that knows the concrete set of plug-ins. It builds a
//! [`Registry`], turns each registered [`Transform`] into a Unix-filter
//! subcommand, and dispatches. Composition between transforms is the shell
//! pipe's job: `cat q.wav | chord stt | chord chat | chord tts > a.wav`.

mod config;
mod proxy;

use std::fs::File;
use std::io::{self, Read};
use std::process::exit;

use chord_core::{Kind, Registry, Result, Transform};
use clap::{Arg, ArgAction, ArgMatches, Command};

use config::Config;

/// Build the registry — the composition root. Engine plug-ins (stt/tts/chat/see)
/// will be registered here as they are implemented.
fn build_registry() -> Registry {
    let mut reg = Registry::new();
    chord_stt::register(&mut reg);
    chord_tts::register(&mut reg);
    chord_chat::register(&mut reg);
    chord_see::register(&mut reg);
    // `draw` (text->image, stable-diffusion.cpp) runs out-of-process: its bundled
    // ggml can't statically co-link with llama.cpp's, so it ships as the separate
    // `chord-draw` binary and we proxy to it. See proxy.rs.
    reg.register(Box::new(proxy::ExecProxy::draw()));
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
        Some(("pipeline", sub)) => run_pipeline(&reg, sub, &config),
        Some((name, sub)) => match reg.get(name) {
            Some(t) => run_filter(t, sub, &config),
            None => {
                eprintln!("chord: unknown command: {name}");
                exit(2);
            }
        },
        None => {
            eprintln!("chord: no command (try `chord --help`)");
            exit(2);
        }
    };

    if let Err(err) = result {
        eprintln!("chord: {err}");
        exit(1);
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
        .subcommand(Command::new("ls").about("List available transforms"))
        .subcommand(Command::new("kinds").about("List the data kinds chord understands"))
        .subcommand(Command::new("config").about("Show the resolved configuration"))
        .subcommand(
            Command::new("pipeline")
                .about("Run several transforms in one process; separate stages with '::'")
                .long_about(
                    "Run a chain of transforms in one process. Separate stages with '::':\n\
                     \n  chord pipeline see --prompt \"what is this\" :: chat --system \"translate to german\" :: tts\n\
                     \nThe first stage reads its file arg or stdin; each later stage reads the\nprevious stage's output; the last writes stdout.",
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
        .about(format!("{}  ({} -> {})", t.describe(), t.from(), t.to()))
        .arg(
            Arg::new("input")
                .help("input file (default: stdin)")
                .value_parser(clap::value_parser!(String))
                .index(1),
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
    opts
}

/// Run a transform as a filter: read the file argument (or stdin), write stdout.
///
/// Options precedence: config-file section (lowest), then named CLI flags.
fn run_filter(t: &dyn Transform, m: &ArgMatches, config: &Config) -> Result<()> {
    let opts = opts_from(t, m, config);

    let stdout = io::stdout();
    let mut out = stdout.lock();

    match m.get_one::<String>("input") {
        Some(path) if path != "-" => {
            let mut f = File::open(path)?;
            t.apply(&mut f, &mut out, &opts)
        }
        _ => {
            let stdin = io::stdin();
            let mut input = stdin.lock();
            t.apply(&mut input, &mut out, &opts)
        }
    }
}

/// Run a multi-stage pipeline in one process. Stages are separated by `::`;
/// each stage is `transform [flags…]`. The first stage reads its file arg or
/// stdin; each later stage reads the previous stage's output; the last writes
/// stdout.
fn run_pipeline(reg: &Registry, m: &ArgMatches, config: &Config) -> Result<()> {
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

    struct Stage<'a> {
        t: &'a dyn Transform,
        opts: chord_core::Options,
        input: Option<String>,
    }

    let mut plan: Vec<Stage> = Vec::with_capacity(stages.len());
    for stage in &stages {
        let name = stage[0].as_str();
        let t = reg
            .get(name)
            .ok_or_else(|| format!("pipeline: unknown transform {name:?}"))?;
        let argv = std::iter::once(name.to_string()).chain(stage[1..].iter().cloned());
        let matches = transform_command(t)
            .try_get_matches_from(argv)
            .map_err(|e| format!("pipeline stage {name:?}: {e}"))?;
        let opts = opts_from(t, &matches, config);
        let input = matches.get_one::<String>("input").cloned();
        plan.push(Stage { t, opts, input });
    }

    // Initial input: the first stage's file arg, else stdin.
    let mut data: Vec<u8> = Vec::new();
    match plan[0].input.as_deref() {
        Some(p) if p != "-" => data = std::fs::read(p)?,
        _ => {
            io::stdin().lock().read_to_end(&mut data)?;
        }
    }

    let stdout = io::stdout();
    let last = plan.len() - 1;
    for (i, stage) in plan.iter().enumerate() {
        let mut reader = io::Cursor::new(data);
        if i == last {
            let mut out = stdout.lock();
            return stage.t.apply(&mut reader, &mut out, &stage.opts);
        }
        let mut buf: Vec<u8> = Vec::new();
        stage
            .t
            .apply(&mut reader, &mut buf, &stage.opts)
            .map_err(|e| format!("{}: {e}", stage.t.name()))?;
        data = buf;
    }
    Ok(())
}

fn print_ls(reg: &Registry) {
    println!("{:<10} {:<16} DESCRIPTION", "NAME", "KINDS");
    for t in reg.all() {
        println!(
            "{:<10} {:<16} {}",
            t.name(),
            format!("{} -> {}", t.from(), t.to()),
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
