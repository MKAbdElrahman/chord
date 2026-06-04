//! Out-of-process plug-in proxy.
//!
//! Every engine ships as its own binary (`chord-stt`, `chord-tts`, …) so
//! incompatible native libraries — e.g. the two copies of `ggml` vendored by
//! llama.cpp and stable-diffusion.cpp — never share one address space, and each
//! engine can stream and crash independently. The `chord` binary links *none* of
//! them.
//!
//! A single generic [`ExecProxy`] adapts any discovered engine (see
//! `discover.rs`) into a [`Transform`]: it spawns the matching binary, forwards
//! the options the engine declared in its manifest as `--flags`, and pipes bytes
//! through it. So `chord ls`, `chord <name>`, and `chord pipeline` work exactly
//! as if the engine were in-process — with no hardcoded engine list or
//! hand-maintained option tables anywhere in the host.

use std::collections::HashSet;
use std::io::{Read, Write};
use std::process::{Command, Stdio};

use chord_core::{ChordError, Kind, Message, OptionSpec, Options, Result, Signature, Transform};

use crate::discover::{self, Engine};

/// A transform backed by an external engine binary (stdin -> stdout). Cheap to
/// copy: it just borrows the discovered, `'static` [`Engine`].
#[derive(Clone, Copy)]
pub struct ExecProxy {
    engine: &'static Engine,
}

impl ExecProxy {
    fn new(engine: &'static Engine) -> Self {
        ExecProxy { engine }
    }

    /// Build the engine command (program + the options forwarded as `--flags`),
    /// with **no stdio wired**. The single source of truth for how an engine is
    /// invoked: [`apply`](Self::apply) wires piped stdio onto it, and the
    /// streaming `chord pipeline` wires inter-stage OS pipes onto it.
    pub fn command(&self, opts: &Options) -> Command {
        let mut cmd = Command::new(&self.engine.bin);
        for spec in self.engine.opts {
            if let Some(v) = opts.get(spec.key) {
                if spec.takes_value {
                    cmd.arg(format!("--{}", spec.key)).arg(v);
                } else if v == "true" {
                    cmd.arg(format!("--{}", spec.key));
                }
            }
        }
        // Forward the global output format (stashed under a reserved key by the
        // host) so the engine emits jsonl events on its (inherited) stderr.
        if let Some(fmt) = opts.get("__format") {
            cmd.arg("--format").arg(fmt);
        }
        cmd
    }
}

/// The default proxy for `name`, if an engine for it is installed. Used by
/// `chord pipeline` to resolve each stage.
pub fn resolve(name: &str) -> Option<ExecProxy> {
    chosen_default(name).map(ExecProxy::new)
}

/// The default proxy for each distinct transform name (one per `chord-<name>`),
/// in discovery order — what the host registers and lists.
pub fn default_proxies() -> Vec<ExecProxy> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for e in discover::engines() {
        if seen.insert(e.name) {
            if let Some(def) = chosen_default(e.name) {
                out.push(ExecProxy::new(def));
            }
        }
    }
    out
}

/// The proxy for a specific backend of `name`, if such an engine is installed.
pub fn alternate(name: &str, backend: &str) -> Option<ExecProxy> {
    discover::engines()
        .iter()
        .find(|e| e.name == name && e.backend == backend)
        .map(ExecProxy::new)
}

/// The non-default backends of `name` (used to union their flags so
/// `--backend X`'s options always parse).
pub fn alternates(name: &str) -> Vec<ExecProxy> {
    let default_bin = chosen_default(name).map(|e| &e.bin);
    discover::engines()
        .iter()
        .filter(|e| e.name == name && Some(&e.bin) != default_bin)
        .map(ExecProxy::new)
        .collect()
}

/// Pick the default engine for `name`: the one named exactly `chord-<name>` if
/// present, else the first discovered backend (so `chord <name>` still works
/// when only an alternate backend is installed).
fn chosen_default(name: &str) -> Option<&'static Engine> {
    let mut first = None;
    for e in discover::engines() {
        if e.name != name {
            continue;
        }
        if e.is_default {
            return Some(e);
        }
        first.get_or_insert(e);
    }
    first
}

impl Transform for ExecProxy {
    fn name(&self) -> &str {
        self.engine.name
    }
    fn signature(&self) -> Signature {
        Signature::new(self.engine.accepts.clone(), self.engine.emits.clone())
    }
    fn describe(&self) -> &str {
        self.engine.describe
    }
    fn backend(&self) -> &str {
        self.engine.backend
    }
    fn options(&self) -> &'static [OptionSpec] {
        self.engine.opts
    }

    fn apply(&self, input: Message, opts: &Options) -> Result<Message> {
        let mut cmd = self.command(opts);
        // stderr is inherited so the child's spinner/errors reach the terminal.
        cmd.stdin(Stdio::piped()).stdout(Stdio::piped());
        let mut child = cmd.spawn().map_err(|e| {
            format!(
                "cannot run {} (is it installed next to chord?): {e}",
                self.engine.bin.display()
            )
        })?;

        // Encode the input message, feed the child's stdin on a separate thread
        // while we read its stdout. Decoupling the two directions means a child
        // that emits output before consuming all input can't deadlock us. The
        // singleton-raw rule keeps a lone inline part on the wire as bare bytes.
        let mut buf = Vec::new();
        chord_core::encode(&input, &mut buf)?;
        let mut child_stdin = child.stdin.take().expect("piped stdin");
        let writer = std::thread::spawn(move || {
            let _ = child_stdin.write_all(&buf);
            // child_stdin drops here, closing the pipe (EOF for the child).
        });

        let mut child_out = child.stdout.take().expect("piped stdout");
        let mut out_buf = Vec::new();
        let read_res = child_out.read_to_end(&mut out_buf);
        let _ = writer.join();
        read_res?;

        let status = child.wait()?;
        if !status.success() {
            // The engine already wrote its own (categorized) message to the
            // inherited stderr; carry its exit code up without repeating it.
            return Err(ChordError::Child(status.code().unwrap_or(1)).into());
        }

        // A raw (unframed) child output decodes to one part of the engine's
        // primary output kind; framed output (multi-part) decodes faithfully.
        let default_kind = self.engine.emits.first().copied().unwrap_or(Kind::Text);
        chord_core::decode(&mut out_buf.as_slice(), default_kind)
    }
}
