//! chord-draw — the image-generation engine (stable-diffusion.cpp via
//! diffusion-rs), shipped as a *separate binary* because its bundled `ggml`
//! cannot be statically linked alongside llama.cpp's in one executable.
//!
//! It's a plain filter: reads a prompt (stdin or file arg), writes PNG bytes to
//! stdout. The main `chord` binary drives it out-of-process via a `draw` proxy,
//! so users still type `chord draw …`.
//!
//! Flags: --model (sd-turbo|sdxl-turbo|sd1.5), --steps, --seed, --width, --height.

use std::io::{IsTerminal, Read, Write};
use std::process::exit;
use std::time::Duration;

use clap::{Arg, ArgMatches, Command};
use diffusion_rs::api::gen_img;
use diffusion_rs::preset::{Preset, PresetBuilder};
use indicatif::{ProgressBar, ProgressStyle};

fn main() {
    let m = cli().get_matches();
    let jsonl = m.get_one::<String>("format").map(String::as_str) == Some("jsonl");
    if jsonl {
        eprintln!(
            "{}",
            serde_json::json!({ "event": "start", "transform": "draw" })
        );
    }
    match run(&m) {
        Ok(()) => {
            if jsonl {
                eprintln!(
                    "{}",
                    serde_json::json!({ "event": "done", "transform": "draw" })
                );
            }
        }
        Err(e) => {
            // Match the shared exit-code convention (2 bad input, 3 missing
            // model, 1 otherwise) so the proxy can propagate it like any engine.
            let ce = e.downcast_ref::<chord_core::ChordError>();
            let code = ce.map(chord_core::ChordError::exit_code).unwrap_or(1);
            if jsonl {
                eprintln!(
                    "{}",
                    serde_json::json!({
                        "event": "error",
                        "transform": "draw",
                        "code": code,
                        "kind": ce.map(chord_core::ChordError::kind).unwrap_or("engine"),
                        "message": e.to_string(),
                    })
                );
            } else {
                eprintln!("chord-draw: {e}");
            }
            exit(code);
        }
    }
}

fn cli() -> Command {
    Command::new("chord-draw")
        .about("text -> image (stable-diffusion.cpp)")
        .arg(
            Arg::new("input")
                .index(1)
                .help("prompt file (default: stdin)"),
        )
        .arg(
            Arg::new("model")
                .long("model")
                .help("sd-turbo | sdxl-turbo | sd1.5"),
        )
        .arg(Arg::new("steps").long("steps"))
        .arg(Arg::new("seed").long("seed"))
        .arg(Arg::new("width").long("width"))
        .arg(Arg::new("height").long("height"))
        .arg(
            Arg::new("format")
                .long("format")
                .value_parser(["text", "jsonl"])
                .default_value("text")
                .help("output format: text, or jsonl for events on stderr"),
        )
}

fn run(m: &ArgMatches) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Prompt from file arg, else stdin.
    let prompt = match m.get_one::<String>("input") {
        Some(p) if p != "-" => std::fs::read_to_string(p)?,
        _ => {
            let mut s = String::new();
            std::io::stdin().read_to_string(&mut s)?;
            s
        }
    };
    let prompt = prompt.trim().to_string();
    if prompt.is_empty() {
        return Err(chord_core::ChordError::BadInput("no prompt on input".to_string()).into());
    }

    let model = m
        .get_one::<String>("model")
        .map(String::as_str)
        .unwrap_or("sd-turbo");
    let preset = match model {
        "sd-turbo" => Preset::SDTurbo,
        "sdxl-turbo" => Preset::SDXLTurbo1_0,
        "sd1.5" | "sd15" => Preset::StableDiffusion1_5,
        other => {
            return Err(
                format!("unknown model preset {other:?} (sd-turbo, sdxl-turbo, sd1.5)").into(),
            )
        }
    };
    let steps = m
        .get_one::<String>("steps")
        .and_then(|s| s.parse::<i32>().ok());
    let seed = m
        .get_one::<String>("seed")
        .and_then(|s| s.parse::<i64>().ok());
    let width = m
        .get_one::<String>("width")
        .and_then(|s| s.parse::<i32>().ok());
    let height = m
        .get_one::<String>("height")
        .and_then(|s| s.parse::<i32>().ok());

    let out_path = std::env::temp_dir().join(format!("chord-draw-{}.png", std::process::id()));

    let pb = spinner("preparing model (first run downloads it)…");
    let modifier_path = out_path.clone();
    let (config, mut model_config) = PresetBuilder::default()
        .preset(preset)
        .prompt(prompt)
        .with_modifier(move |(mut cb, mb)| {
            cb.output(modifier_path.clone());
            if let Some(s) = steps {
                cb.steps(s);
            }
            if let Some(s) = seed {
                cb.seed(s);
            }
            if let Some(w) = width {
                cb.width(w);
            }
            if let Some(h) = height {
                cb.height(h);
            }
            Ok((cb, mb))
        })
        .build()
        .map_err(|e| format!("build config: {e}"))?;

    pb.set_message("generating image…");
    {
        // stable-diffusion.cpp prints its logs and sampling progress bar to
        // stdout (fd 1) — that would corrupt the PNG bytes we stream there. We
        // keep our own indicatif spinner on stderr as the progress UI, so mute
        // the engine's fd-1 chatter for the duration of generation; the guard
        // restores stdout on drop, before we write the image.
        let _muted = StdoutMute::new();
        gen_img(&config, &mut model_config).map_err(|e| format!("generate: {e}"))?;
    }
    pb.finish_and_clear();

    let png = std::fs::read(&out_path).map_err(|e| format!("reading rendered image: {e}"))?;
    std::io::stdout().write_all(&png)?;
    let _ = std::fs::remove_file(&out_path);
    Ok(())
}

/// RAII guard that redirects the process's stdout (fd 1) to /dev/null and
/// restores it on drop. Used to silence the C engine's direct writes to fd 1
/// without losing our own stdout (the PNG) afterwards.
struct StdoutMute {
    saved: i32,
}

impl StdoutMute {
    fn new() -> Option<Self> {
        unsafe {
            let saved = libc::dup(1);
            if saved < 0 {
                return None;
            }
            let devnull = libc::open(c"/dev/null".as_ptr(), libc::O_WRONLY);
            if devnull < 0 {
                libc::close(saved);
                return None;
            }
            libc::dup2(devnull, 1);
            libc::close(devnull);
            Some(StdoutMute { saved })
        }
    }
}

impl Drop for StdoutMute {
    fn drop(&mut self) {
        unsafe {
            libc::dup2(self.saved, 1);
            libc::close(self.saved);
        }
    }
}

fn spinner(msg: &str) -> ProgressBar {
    if std::io::stderr().is_terminal() {
        let pb = ProgressBar::new_spinner();
        pb.set_style(ProgressStyle::with_template("{spinner:.cyan} {msg}").unwrap());
        pb.enable_steady_tick(Duration::from_millis(120));
        pb.set_message(msg.to_string());
        pb
    } else {
        ProgressBar::hidden()
    }
}
