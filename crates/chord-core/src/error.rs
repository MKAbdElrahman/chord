//! A categorized error, so the CLI can choose a meaningful process exit code.
//!
//! Engines return these for known conditions; any other error (I/O, an engine
//! SDK failure, …) boxes into [`crate::Error`] and is treated as a generic
//! failure (exit code 1). Keeping this in the kernel lets every engine and the
//! host agree on the same categories and codes.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ChordError {
    /// A required model or asset isn't present. `hint` tells the user how to
    /// get it (e.g. "run `chord pull stt`").
    #[error("{what} not found — {hint}")]
    ModelMissing { what: String, hint: String },
    /// The input was empty or malformed.
    #[error("{0}")]
    BadInput(String),
    /// The engine failed at runtime.
    #[error("{0}")]
    Engine(String),
    /// A spawned engine process already reported its own error to stderr; this
    /// just carries its exit code up so the host can exit with the same one.
    #[error("engine exited with code {0}")]
    Child(i32),
}

impl ChordError {
    /// The process exit code for this category: 1 engine failure, 2 bad input,
    /// 3 missing model. `Child` carries the spawned engine's own code through.
    pub fn exit_code(&self) -> i32 {
        match self {
            ChordError::Engine(_) => 1,
            ChordError::BadInput(_) => 2,
            ChordError::ModelMissing { .. } => 3,
            ChordError::Child(code) => *code,
        }
    }

    /// True when the error was already printed by a child process, so the host
    /// shouldn't print it again — only adopt the exit code.
    pub fn already_reported(&self) -> bool {
        matches!(self, ChordError::Child(_))
    }

    /// A stable machine-readable category tag (used in `--format jsonl` events).
    pub fn kind(&self) -> &'static str {
        match self {
            ChordError::ModelMissing { .. } => "model_missing",
            ChordError::BadInput(_) => "bad_input",
            ChordError::Engine(_) => "engine",
            ChordError::Child(_) => "child",
        }
    }
}
