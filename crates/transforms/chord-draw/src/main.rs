//! chord-draw — the image-generation engine (stable-diffusion.cpp via
//! diffusion-rs), shipped as a *separate binary* because its bundled `ggml`
//! cannot be statically linked alongside llama.cpp's in one executable.
//!
//! It's a plain filter: reads a prompt (stdin or file arg), writes PNG bytes to
//! stdout. The main `chord` binary drives it out-of-process via a `draw` proxy,
//! so users still type `chord draw …`.
//!
//! Flags: --model (sd-turbo|sdxl-turbo|sd1.5|flux-schnell), --steps, --seed,
//! --width, --height, --lora, --lora-weight.

use std::io::{IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::process::exit;
use std::time::Duration;

use clap::{Arg, ArgMatches, Command};
use diffusion_rs::api::{
    gen_img, Config, ConfigBuilder, LoraSpec, ModelConfig, ModelConfigBuilder,
};
use diffusion_rs::preset::{Preset, PresetBuilder};
use diffusion_rs::util::download_file_hf_hub;
use indicatif::{ProgressBar, ProgressStyle};

fn main() {
    // draw is a bespoke filter (no chord_core::Transform impl), so it answers the
    // discovery protocol by hand instead of through chord-runner. Keep this
    // manifest in step with `cli()` below.
    if std::env::args().any(|a| a == "--chord-manifest") {
        print_manifest();
        return;
    }
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

/// Emit the discovery manifest (draw has no `Transform` impl, so it's built by
/// hand). The option set mirrors `cli()`.
fn print_manifest() {
    let opt = |key: &str, help: &str| chord_core::ManifestOption {
        key: key.to_string(),
        help: help.to_string(),
        takes_value: true,
    };
    let manifest = chord_core::Manifest {
        version: chord_core::MANIFEST_VERSION,
        name: "draw".to_string(),
        accepts: vec![chord_core::Kind::Text],
        emits: vec![chord_core::Kind::Image],
        backend: "stable-diffusion.cpp".to_string(),
        describe: "text-to-image (stable-diffusion.cpp)".to_string(),
        options: vec![
            opt(
                "model",
                "preset: sd-turbo | sdxl-turbo | sd1.5 | flux-schnell (default sd-turbo)",
            ),
            opt("steps", "sampling steps (preset default)"),
            opt("seed", "RNG seed (<0 = random; default 42)"),
            opt("width", "image width in px"),
            opt("height", "image height in px"),
            opt("lora", "path to a LoRA .safetensors to apply (flux-schnell)"),
            opt("lora-weight", "LoRA strength multiplier (default 1.0)"),
        ],
    };
    match serde_json::to_string(&manifest) {
        Ok(json) => println!("{json}"),
        Err(e) => {
            eprintln!("chord-draw: manifest: {e}");
            exit(1);
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
                .help("sd-turbo | sdxl-turbo | sd1.5 | flux-schnell"),
        )
        .arg(Arg::new("steps").long("steps"))
        .arg(Arg::new("seed").long("seed"))
        .arg(Arg::new("width").long("width"))
        .arg(Arg::new("height").long("height"))
        .arg(
            Arg::new("lora")
                .long("lora")
                .help("path to a LoRA .safetensors to apply (flux-schnell)"),
        )
        .arg(
            Arg::new("lora-weight")
                .long("lora-weight")
                .help("LoRA strength multiplier (default 1.0)"),
        )
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

    // Optional LoRA: stable-diffusion.cpp loads it from a directory by file
    // stem, so split the path into (dir, stem) up front. Default strength 1.0.
    let lora = match m.get_one::<String>("lora") {
        Some(p) => {
            let path = PathBuf::from(p);
            let dir = path
                .parent()
                .filter(|d| !d.as_os_str().is_empty())
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("."));
            let stem = path
                .file_stem()
                .and_then(|s| s.to_str())
                .ok_or_else(|| format!("invalid --lora path {p:?}"))?
                .to_string();
            let weight = m
                .get_one::<String>("lora-weight")
                .and_then(|s| s.parse::<f32>().ok())
                .unwrap_or(1.0);
            Some((dir, stem, weight))
        }
        None => None,
    };

    let out_path = std::env::temp_dir().join(format!("chord-draw-{}.png", std::process::id()));

    let pb = spinner("preparing model (first run downloads it)…");

    // FLUX.1-schnell is assembled by hand from ungated GGUF mirrors (see
    // `build_flux_schnell`); the SD presets use diffusion-rs's bundled preset +
    // a modifier closure for the per-run knobs.
    let (config, mut model_config) = if matches!(model, "flux-schnell" | "flux") {
        build_flux_schnell(&prompt, &out_path, steps, seed, width, height, lora.as_ref())
            .map_err(|e| format!("build flux config: {e}"))?
    } else {
        let preset = match model {
            "sd-turbo" => Preset::SDTurbo,
            "sdxl-turbo" => Preset::SDXLTurbo1_0,
            "sd1.5" | "sd15" => Preset::StableDiffusion1_5,
            other => {
                return Err(format!(
                    "unknown model preset {other:?} (sd-turbo, sdxl-turbo, sd1.5, flux-schnell)"
                )
                .into())
            }
        };
        let modifier_path = out_path.clone();
        PresetBuilder::default()
            .preset(preset)
            .prompt(prompt)
            .with_modifier(move |(mut cb, mut mb)| {
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
                if let Some((dir, stem, weight)) = &lora {
                    mb.lora_models(
                        dir,
                        vec![LoraSpec {
                            file_name: stem.clone(),
                            is_high_noise: false,
                            multiplier: *weight,
                        }],
                    );
                }
                Ok((cb, mb))
            })
            .build()
            .map_err(|e| format!("build config: {e}"))?
    };

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

/// Assemble FLUX.1-schnell from ungated GGUF mirrors and return ready configs.
///
/// diffusion-rs's own `Flux1Schnell` preset sources its VAE from the gated
/// `black-forest-labs/FLUX.1-schnell` repo (HTTP 401 without an HF token), so we
/// bypass the preset and pin every file to a public mirror — the Q4_K base from
/// `leejet`, the T5/VAE GGUFs from `Green-Sky`, and `clip_l` from
/// `comfyanonymous`. Result: `chord draw --model flux-schnell` needs no HF login.
/// Optional `lora` is applied on top (matched from its directory by file stem).
#[allow(clippy::too_many_arguments)]
fn build_flux_schnell(
    prompt: &str,
    out_path: &Path,
    steps: Option<i32>,
    seed: Option<i64>,
    width: Option<i32>,
    height: Option<i32>,
    lora: Option<&(PathBuf, String, f32)>,
) -> Result<(Config, ModelConfig), Box<dyn std::error::Error + Send + Sync>> {
    let base = download_file_hf_hub("leejet/FLUX.1-schnell-gguf", "flux1-schnell-q4_k.gguf")?;
    let t5xxl = download_file_hf_hub("Green-Sky/flux.1-schnell-GGUF", "t5xxl_q4_k.gguf")?;
    let clip_l = download_file_hf_hub("comfyanonymous/flux_text_encoders", "clip_l.safetensors")?;
    let vae = download_file_hf_hub("Green-Sky/flux.1-schnell-GGUF", "ae-f16.gguf")?;

    let mut mb = ModelConfigBuilder::default();
    mb.diffusion_model(base)
        .t5xxl(t5xxl)
        .clip_l(clip_l)
        .vae(vae)
        .vae_tiling(true);
    if let Some((dir, stem, weight)) = lora {
        mb.lora_models(
            dir,
            vec![LoraSpec {
                file_name: stem.clone(),
                is_high_noise: false,
                multiplier: *weight,
            }],
        );
    }
    let model_config = mb.build().map_err(|e| format!("flux model config: {e}"))?;

    // schnell is distilled: cfg_scale 1, 4 steps, 1024² — matching the preset.
    let mut cb = ConfigBuilder::default();
    cb.prompt(prompt.to_string())
        .output(out_path.to_path_buf())
        .cfg_scale(1.0)
        .steps(steps.unwrap_or(4))
        .width(width.unwrap_or(1024))
        .height(height.unwrap_or(1024));
    if let Some(s) = seed {
        cb.seed(s);
    }
    let config = cb.build().map_err(|e| format!("flux config: {e}"))?;
    Ok((config, model_config))
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
