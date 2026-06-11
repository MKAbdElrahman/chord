//! The `--format jsonl` event schema.
//!
//! When a transform runs with `--format jsonl`, it emits one of these as NDJSON
//! on **stderr** (the machine-readable status channel — stdout stays pure data).
//! Typed structs keep the wire format stable and shared between the runner that
//! produces them and any consumer that parses them.
//!
//! Every event carries a wall-clock timestamp and the emitting pid; `Done`
//! carries the apply duration and byte counts. Together these are exactly the
//! per-item arrival/departure data Little's finite-window theorem needs, so a
//! consumer can compute each stage's λ, W, and L *exactly* from a log of one
//! run — no statistics required (see `docs/theory/THEORY.md`).

use serde::{Deserialize, Serialize};

/// Milliseconds since the Unix epoch — the event timestamp base.
fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Emitted when a transform starts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Start<'a> {
    pub event: &'a str,
    pub transform: &'a str,
    /// Wall-clock ms since the Unix epoch.
    pub ts_ms: u64,
    /// The emitting process — disambiguates pipeline stages running the same
    /// transform.
    pub pid: u32,
}

impl<'a> Start<'a> {
    pub fn new(transform: &'a str) -> Self {
        Start {
            event: "start",
            transform,
            ts_ms: now_ms(),
            pid: std::process::id(),
        }
    }
}

/// Emitted when a transform finishes successfully.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Done<'a> {
    pub event: &'a str,
    pub transform: &'a str,
    pub ts_ms: u64,
    pub pid: u32,
    /// Time from input open to output written, in ms (the item's W).
    pub duration_ms: u64,
    /// Bytes consumed from the input stream.
    pub bytes_in: u64,
    /// Bytes written to the output stream.
    pub bytes_out: u64,
    /// 1-based item index within an `--each` batch; absent for the
    /// whole-run summary event.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub item: Option<u64>,
}

impl<'a> Done<'a> {
    pub fn new(transform: &'a str, duration_ms: u64, bytes_in: u64, bytes_out: u64) -> Self {
        Done {
            event: "done",
            transform,
            ts_ms: now_ms(),
            pid: std::process::id(),
            duration_ms,
            bytes_in,
            bytes_out,
            item: None,
        }
    }

    /// Tag this event with its batch item index (builder style).
    pub fn with_item(mut self, item: u64) -> Self {
        self.item = Some(item);
        self
    }
}

/// Emitted when a transform fails, carrying the exit code and error category.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Error<'a> {
    pub event: &'a str,
    pub transform: &'a str,
    pub ts_ms: u64,
    pub pid: u32,
    /// Time from input open to the failure, in ms. Zero when unknown.
    pub duration_ms: u64,
    pub code: i32,
    pub kind: &'a str,
    pub message: String,
}

impl<'a> Error<'a> {
    pub fn new(transform: &'a str, code: i32, kind: &'a str, message: String) -> Self {
        Error {
            event: "error",
            transform,
            ts_ms: now_ms(),
            pid: std::process::id(),
            duration_ms: 0,
            code,
            kind,
            message,
        }
    }

    /// Attach the elapsed time (builder style).
    pub fn with_duration(mut self, duration_ms: u64) -> Self {
        self.duration_ms = duration_ms;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Little's Law (the finite-window theorem is numerically exact on a
    // sample path) needs arrival/departure times per item and a way to tell
    // stages apart — so every event carries a wall-clock timestamp and the
    // emitting pid, and Done carries duration and byte counts.

    #[test]
    fn start_carries_timestamp_and_pid() {
        let e = Start::new("stt");
        assert!(e.ts_ms > 1_500_000_000_000); // sane wall-clock epoch ms
        assert_eq!(e.pid, std::process::id());
        let json = serde_json::to_string(&e).unwrap();
        assert!(json.contains("\"ts_ms\""), "{json}");
        assert!(json.contains("\"pid\""), "{json}");
    }

    #[test]
    fn done_carries_duration_and_byte_counts() {
        let e = Done::new("stt", 1234, 10, 20);
        assert_eq!(e.duration_ms, 1234);
        assert_eq!(e.bytes_in, 10);
        assert_eq!(e.bytes_out, 20);
        let json = serde_json::to_string(&e).unwrap();
        for key in ["ts_ms", "pid", "duration_ms", "bytes_in", "bytes_out"] {
            assert!(json.contains(&format!("\"{key}\"")), "{key} in {json}");
        }
    }

    #[test]
    fn done_item_index_is_optional_and_serialized_when_set() {
        // Per-message events in --each batches carry an item index; the
        // batch-summary Done omits it entirely (no noisy null).
        let summary = serde_json::to_string(&Done::new("stt", 1, 2, 3)).unwrap();
        assert!(!summary.contains("\"item\""), "{summary}");
        let per_item = serde_json::to_string(&Done::new("stt", 1, 2, 3).with_item(7)).unwrap();
        assert!(per_item.contains("\"item\":7"), "{per_item}");
    }

    #[test]
    fn error_carries_duration() {
        let e = Error::new("stt", 2, "bad_input", "boom".into()).with_duration(7);
        assert_eq!(e.duration_ms, 7);
        let json = serde_json::to_string(&e).unwrap();
        assert!(json.contains("\"duration_ms\""), "{json}");
        assert!(json.contains("\"ts_ms\""), "{json}");
    }
}
