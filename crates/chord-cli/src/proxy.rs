//! Out-of-process plug-in proxies.
//!
//! Every engine ships as its own binary (`chord-stt`, `chord-tts`, `chord-chat`,
//! `chord-see`, `chord-draw`) so incompatible native libraries — e.g. the two
//! copies of `ggml` vendored by llama.cpp and stable-diffusion.cpp — never share
//! one address space, and each engine can stream and crash independently.
//!
//! The `chord` binary links *none* of them. Each [`ExecProxy`] is a [`Transform`]
//! that spawns the matching engine binary, forwards the declared options as
//! `--flags`, and pipes bytes through it. So `chord ls`, `chord <name>`, and
//! `chord pipeline` work exactly as if the engine were in-process.

use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};

use chord_core::{ChordError, Kind, OptionSpec, Options, Result, Transform};

/// A transform backed by an external binary (stdin -> stdout).
pub struct ExecProxy {
    name: &'static str,
    from: Kind,
    to: Kind,
    describe: &'static str,
    /// The inference backend (engine) this transform declares it needs.
    backend: &'static str,
    /// The engine binary that provides that backend.
    bin: &'static str,
    opts: &'static [OptionSpec],
}

const STT_OPTS: &[OptionSpec] = &[
    OptionSpec {
        key: "model",
        help: "whisper model path or name (default large-v3-turbo)",
        takes_value: true,
    },
    OptionSpec {
        key: "lang",
        help: "language code, e.g. en, de (default auto)",
        takes_value: true,
    },
    OptionSpec {
        key: "threads",
        help: "CPU threads (default 4)",
        takes_value: true,
    },
];

const TTS_OPTS: &[OptionSpec] = &[
    OptionSpec {
        key: "voice",
        help: "voice preset name or JSON path (default M1)",
        takes_value: true,
    },
    OptionSpec {
        key: "lang",
        help: "language code (default en)",
        takes_value: true,
    },
    OptionSpec {
        key: "steps",
        help: "denoising steps 5-12 (default 8)",
        takes_value: true,
    },
    OptionSpec {
        key: "speed",
        help: "speech speed 0.7-2.0 (default 1.05)",
        takes_value: true,
    },
    OptionSpec {
        key: "silence",
        help: "seconds between chunks (default 0.3)",
        takes_value: true,
    },
    OptionSpec {
        key: "assets",
        help: "Supertonic assets directory",
        takes_value: true,
    },
];

const CHAT_OPTS: &[OptionSpec] = &[
    OptionSpec {
        key: "model",
        help: "GGUF model path or name",
        takes_value: true,
    },
    OptionSpec {
        key: "system",
        help: "system prompt",
        takes_value: true,
    },
    OptionSpec {
        key: "think",
        help: "enable reasoning (default off)",
        takes_value: false,
    },
    OptionSpec {
        key: "max_tokens",
        help: "max reply tokens (default 512)",
        takes_value: true,
    },
    OptionSpec {
        key: "temperature",
        help: "sampling temperature (default 0.7)",
        takes_value: true,
    },
    OptionSpec {
        key: "n_ctx",
        help: "context window tokens (default 4096)",
        takes_value: true,
    },
];

const SEE_OPTS: &[OptionSpec] = &[
    OptionSpec {
        key: "prompt",
        help: "instruction/question about the image",
        takes_value: true,
    },
    OptionSpec {
        key: "model",
        help: "vision GGUF model path",
        takes_value: true,
    },
    OptionSpec {
        key: "mmproj",
        help: "multimodal projector GGUF path",
        takes_value: true,
    },
    OptionSpec {
        key: "max_tokens",
        help: "max reply tokens (default 256)",
        takes_value: true,
    },
    OptionSpec {
        key: "temperature",
        help: "sampling temperature (default 0.3)",
        takes_value: true,
    },
    OptionSpec {
        key: "n_ctx",
        help: "context window tokens (default 4096)",
        takes_value: true,
    },
];

// Options for the alternate llama.cpp (mtmd-audio) stt backend.
const STT_LLAMA_OPTS: &[OptionSpec] = &[
    OptionSpec {
        key: "model",
        help: "audio-LLM GGUF (path or hf: ref)",
        takes_value: true,
    },
    OptionSpec {
        key: "mmproj",
        help: "audio projector GGUF (path or hf: ref)",
        takes_value: true,
    },
    OptionSpec {
        key: "prompt",
        help: "instruction (default: transcribe verbatim)",
        takes_value: true,
    },
    OptionSpec {
        key: "max_tokens",
        help: "max output tokens (default 448)",
        takes_value: true,
    },
    OptionSpec {
        key: "temperature",
        help: "sampling temperature; 0 = greedy",
        takes_value: true,
    },
    OptionSpec {
        key: "n_ctx",
        help: "context window tokens (default 4096)",
        takes_value: true,
    },
    OptionSpec {
        key: "threads",
        help: "CPU threads (default 4)",
        takes_value: true,
    },
];

// Options for the alternate sherpa-onnx (NeMo transducer / Parakeet) stt backend.
const STT_SHERPA_OPTS: &[OptionSpec] = &[
    OptionSpec {
        key: "encoder",
        help: "encoder ONNX (path or hf: ref)",
        takes_value: true,
    },
    OptionSpec {
        key: "decoder",
        help: "decoder ONNX (path or hf: ref)",
        takes_value: true,
    },
    OptionSpec {
        key: "joiner",
        help: "joiner ONNX (path or hf: ref)",
        takes_value: true,
    },
    OptionSpec {
        key: "tokens",
        help: "tokens file (path or hf: ref)",
        takes_value: true,
    },
    OptionSpec {
        key: "threads",
        help: "CPU threads (default 4)",
        takes_value: true,
    },
    OptionSpec {
        key: "chunk",
        help: "seconds per chunk for long audio (default 300)",
        takes_value: true,
    },
];

const DIARIZE_OPTS: &[OptionSpec] = &[
    OptionSpec {
        key: "segmentation",
        help: "pyannote segmentation ONNX (path or hf: ref)",
        takes_value: true,
    },
    OptionSpec {
        key: "embedding",
        help: "speaker-embedding ONNX (path or hf: ref)",
        takes_value: true,
    },
    OptionSpec {
        key: "speakers",
        help: "fixed number of speakers; omit/-1 to auto-detect",
        takes_value: true,
    },
    OptionSpec {
        key: "threshold",
        help: "clustering threshold when auto-detecting (default 0.5)",
        takes_value: true,
    },
];

const VAD_OPTS: &[OptionSpec] = &[
    OptionSpec {
        key: "model",
        help: "Silero VAD ONNX (path or hf: ref)",
        takes_value: true,
    },
    OptionSpec {
        key: "threshold",
        help: "speech probability threshold (default 0.5)",
        takes_value: true,
    },
    OptionSpec {
        key: "min_silence",
        help: "min silence seconds to split (default 0.5)",
        takes_value: true,
    },
    OptionSpec {
        key: "min_speech",
        help: "min speech seconds to keep (default 0.25)",
        takes_value: true,
    },
    OptionSpec {
        key: "max_speech",
        help: "max speech seconds per segment (default 30)",
        takes_value: true,
    },
];

const LANGID_OPTS: &[OptionSpec] = &[
    OptionSpec {
        key: "encoder",
        help: "Whisper encoder ONNX (path or hf: ref)",
        takes_value: true,
    },
    OptionSpec {
        key: "decoder",
        help: "Whisper decoder ONNX (path or hf: ref)",
        takes_value: true,
    },
];

const DRAW_OPTS: &[OptionSpec] = &[
    OptionSpec {
        key: "model",
        help: "preset: sd-turbo | sdxl-turbo | sd1.5 (default sd-turbo)",
        takes_value: true,
    },
    OptionSpec {
        key: "steps",
        help: "sampling steps (preset default)",
        takes_value: true,
    },
    OptionSpec {
        key: "seed",
        help: "RNG seed (<0 = random; default 42)",
        takes_value: true,
    },
    OptionSpec {
        key: "width",
        help: "image width in px",
        takes_value: true,
    },
    OptionSpec {
        key: "height",
        help: "image height in px",
        takes_value: true,
    },
];

impl ExecProxy {
    pub fn stt() -> Self {
        ExecProxy {
            name: "stt",
            from: Kind::Audio,
            to: Kind::Text,
            describe: "speech-to-text (whisper.cpp)",
            backend: "whisper.cpp",
            bin: "chord-stt",
            opts: STT_OPTS,
        }
    }
    pub fn tts() -> Self {
        ExecProxy {
            name: "tts",
            from: Kind::Text,
            to: Kind::Audio,
            describe: "text-to-speech (Supertonic / ONNX)",
            backend: "onnxruntime",
            bin: "chord-tts",
            opts: TTS_OPTS,
        }
    }
    pub fn chat() -> Self {
        ExecProxy {
            name: "chat",
            from: Kind::Text,
            to: Kind::Text,
            describe: "chat / text generation (llama.cpp)",
            backend: "llama.cpp",
            bin: "chord-chat",
            opts: CHAT_OPTS,
        }
    }
    pub fn see() -> Self {
        ExecProxy {
            name: "see",
            from: Kind::Image,
            to: Kind::Text,
            describe: "vision: describe/read an image (llama.cpp mtmd)",
            backend: "llama.cpp",
            bin: "chord-see",
            opts: SEE_OPTS,
        }
    }
    pub fn draw() -> Self {
        ExecProxy {
            name: "draw",
            from: Kind::Text,
            to: Kind::Image,
            describe: "text-to-image (stable-diffusion.cpp)",
            backend: "stable-diffusion.cpp",
            bin: "chord-draw",
            opts: DRAW_OPTS,
        }
    }
    /// Alternate `stt` backend: audio-LLM ASR via llama.cpp mtmd-audio.
    pub fn stt_llama() -> Self {
        ExecProxy {
            name: "stt",
            from: Kind::Audio,
            to: Kind::Text,
            describe: "speech-to-text via audio-LLM (llama.cpp mtmd)",
            backend: "llama.cpp",
            bin: "chord-stt-llama",
            opts: STT_LLAMA_OPTS,
        }
    }
    /// Voice activity detection (audio -> speech segment timestamps) via sherpa-onnx.
    pub fn vad() -> Self {
        ExecProxy {
            name: "vad",
            from: Kind::Audio,
            to: Kind::Text,
            describe: "voice activity detection: speech segment timestamps (sherpa-onnx / Silero)",
            backend: "sherpa-onnx",
            bin: "chord-vad",
            opts: VAD_OPTS,
        }
    }
    /// Spoken language identification (audio -> language code) via sherpa-onnx.
    pub fn langid() -> Self {
        ExecProxy {
            name: "langid",
            from: Kind::Audio,
            to: Kind::Text,
            describe: "spoken language identification (sherpa-onnx / Whisper)",
            backend: "sherpa-onnx",
            bin: "chord-langid",
            opts: LANGID_OPTS,
        }
    }
    /// Speaker diarization (audio -> speaker turns) via sherpa-onnx.
    pub fn diarize() -> Self {
        ExecProxy {
            name: "diarize",
            from: Kind::Audio,
            to: Kind::Text,
            describe: "speaker diarization: who spoke when (sherpa-onnx / pyannote)",
            backend: "sherpa-onnx",
            bin: "chord-diarize",
            opts: DIARIZE_OPTS,
        }
    }
    /// Alternate `stt` backend: NeMo transducer (Parakeet) via sherpa-onnx.
    pub fn stt_sherpa() -> Self {
        ExecProxy {
            name: "stt",
            from: Kind::Audio,
            to: Kind::Text,
            describe: "speech-to-text via NeMo transducer (sherpa-onnx; Parakeet)",
            backend: "sherpa-onnx",
            bin: "chord-stt-sherpa",
            opts: STT_SHERPA_OPTS,
        }
    }
}

/// All alternate (non-default) backends for a transform, selectable via
/// `--backend`. Empty if the transform has only its default backend.
pub fn alternates(name: &str) -> Vec<ExecProxy> {
    match name {
        "stt" => vec![ExecProxy::stt_llama(), ExecProxy::stt_sherpa()],
        _ => vec![],
    }
}

/// The alternate backend matching `backend`, if any.
pub fn alternate(name: &str, backend: &str) -> Option<ExecProxy> {
    alternates(name).into_iter().find(|p| p.backend == backend)
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
    fn backend(&self) -> &str {
        self.backend
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
        // Forward the global output format (stashed under a reserved key by the
        // host) so the engine emits jsonl events on its (inherited) stderr.
        if let Some(fmt) = opts.get("__format") {
            cmd.arg("--format").arg(fmt);
        }
        // stderr is inherited so the child's spinner/errors reach the terminal.
        cmd.stdin(Stdio::piped()).stdout(Stdio::piped());
        let mut child = cmd.spawn().map_err(|e| {
            format!(
                "cannot run {} (is it installed next to chord?): {e}",
                self.bin
            )
        })?;

        // Read the input fully, then feed the child's stdin on a separate thread
        // while we stream its stdout through. Decoupling the two directions means
        // a child that emits output before consuming all input can't deadlock us.
        let mut buf = Vec::new();
        input.read_to_end(&mut buf)?;
        let mut child_stdin = child.stdin.take().expect("piped stdin");
        let writer = std::thread::spawn(move || {
            let _ = child_stdin.write_all(&buf);
            // child_stdin drops here, closing the pipe (EOF for the child).
        });

        let mut child_out = child.stdout.take().expect("piped stdout");
        let copy_res = std::io::copy(&mut child_out, output);
        let _ = writer.join();
        copy_res?;

        let status = child.wait()?;
        if !status.success() {
            // The engine already wrote its own (categorized) message to the
            // inherited stderr; carry its exit code up without repeating it.
            return Err(ChordError::Child(status.code().unwrap_or(1)).into());
        }
        Ok(())
    }
}

/// Find the helper binary next to the current executable (so an installed
/// `chord` finds the installed `chord-*` engines beside it), else rely on PATH.
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
