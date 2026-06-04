//! The `--format jsonl` event schema.
//!
//! When a transform runs with `--format jsonl`, it emits one of these as NDJSON
//! on **stderr** (the machine-readable status channel — stdout stays pure data).
//! Typed structs keep the wire format stable and shared between the runner that
//! produces them and any consumer that parses them.

use serde::{Deserialize, Serialize};

/// Emitted when a transform starts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Start<'a> {
    pub event: &'a str,
    pub transform: &'a str,
}

impl<'a> Start<'a> {
    pub fn new(transform: &'a str) -> Self {
        Start {
            event: "start",
            transform,
        }
    }
}

/// Emitted when a transform finishes successfully.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Done<'a> {
    pub event: &'a str,
    pub transform: &'a str,
}

impl<'a> Done<'a> {
    pub fn new(transform: &'a str) -> Self {
        Done {
            event: "done",
            transform,
        }
    }
}

/// Emitted when a transform fails, carrying the exit code and error category.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Error<'a> {
    pub event: &'a str,
    pub transform: &'a str,
    pub code: i32,
    pub kind: &'a str,
    pub message: String,
}

impl<'a> Error<'a> {
    pub fn new(transform: &'a str, code: i32, kind: &'a str, message: String) -> Self {
        Error {
            event: "error",
            transform,
            code,
            kind,
            message,
        }
    }
}
