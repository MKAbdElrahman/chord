//! Out-of-process plug-in proxy.
//!
//! Some engines can't be statically linked into the main `chord` binary (e.g.
//! stable-diffusion.cpp vendors its own `ggml`, which collides with llama.cpp's).
//! Such an engine ships as a separate binary; this proxy makes it look like any
//! other [`Transform`]: it forwards the declared options as `--flags` and pipes
//! bytes through the child process. `chord ls`, `chord <name>`, and
//! `chord pipeline` therefore work unchanged.

use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};

use chord_core::{Kind, OptionSpec, Options, Result, Transform};

/// A transform backed by an external binary (stdin -> stdout).
pub struct ExecProxy {
    name: &'static str,
    from: Kind,
    to: Kind,
    describe: &'static str,
    bin: &'static str,
    opts: &'static [OptionSpec],
}

const DRAW_OPTS: &[OptionSpec] = &[
    OptionSpec { key: "model", help: "preset: sd-turbo | sdxl-turbo | sd1.5 (default sd-turbo)", takes_value: true },
    OptionSpec { key: "steps", help: "sampling steps (preset default)", takes_value: true },
    OptionSpec { key: "seed", help: "RNG seed (<0 = random; default 42)", takes_value: true },
    OptionSpec { key: "width", help: "image width in px", takes_value: true },
    OptionSpec { key: "height", help: "image height in px", takes_value: true },
];

impl ExecProxy {
    /// The `draw` transform, backed by the `chord-draw` binary.
    pub fn draw() -> Self {
        ExecProxy {
            name: "draw",
            from: Kind::Text,
            to: Kind::Image,
            describe: "text-to-image (stable-diffusion.cpp, external)",
            bin: "chord-draw",
            opts: DRAW_OPTS,
        }
    }
}

impl Transform for ExecProxy {
    fn name(&self) -> &str {
        self.name
    }
    fn from(&self) -> Kind {
        self.from
    }
    fn to(&self) -> Kind {
        self.to
    }
    fn describe(&self) -> &str {
        self.describe
    }
    fn options(&self) -> &'static [OptionSpec] {
        self.opts
    }

    fn apply(&self, input: &mut dyn Read, output: &mut dyn Write, opts: &Options) -> Result<()> {
        let mut cmd = Command::new(locate(self.bin));
        for spec in self.opts {
            if let Some(v) = opts.get(spec.key) {
                if spec.takes_value {
                    cmd.arg(format!("--{}", spec.key)).arg(v);
                } else if v == "true" {
                    cmd.arg(format!("--{}", spec.key));
                }
            }
        }
        // stderr is inherited so the child's spinner/errors reach the terminal.
        cmd.stdin(Stdio::piped()).stdout(Stdio::piped());
        let mut child = cmd
            .spawn()
            .map_err(|e| format!("cannot run {} (is it installed next to chord?): {e}", self.bin))?;

        // Feed input, then stream the child's output through. The inputs here
        // are small (a prompt), so write-then-read won't deadlock.
        let mut buf = Vec::new();
        input.read_to_end(&mut buf)?;
        child.stdin.take().expect("piped stdin").write_all(&buf)?;

        let mut child_out = child.stdout.take().expect("piped stdout");
        std::io::copy(&mut child_out, output)?;

        let status = child.wait()?;
        if !status.success() {
            return Err(format!("{} exited with {status}", self.bin).into());
        }
        Ok(())
    }
}

/// Find the helper binary next to the current executable (so an installed
/// `chord` finds the installed `chord-draw` beside it), else rely on PATH.
fn locate(bin: &str) -> PathBuf {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let candidate = dir.join(bin);
            if candidate.exists() {
                return candidate;
            }
        }
    }
    PathBuf::from(bin)
}
