//! chord-core — the chord microkernel.
//!
//! This crate is the stable, minimal **core system** of the microkernel
//! architecture (see ARCHITECTURE.md). It defines:
//!
//! - [`Kind`]: the coarse data modalities plug-ins consume and produce.
//! - [`Transform`]: the plug-in contract (a self-contained Unix filter).
//! - [`Registry`]: a name -> plug-in directory for CLI dispatch.
//!
//! It has **no model/engine dependencies** (standard library only). Engines
//! live in separate plug-in crates that depend on this one — never the reverse.

mod error;
mod kind;
mod registry;
mod transform;

pub use error::ChordError;
pub use kind::Kind;
pub use registry::Registry;
pub use transform::{OptionSpec, Options, Transform};

/// Boxed error type used across the kernel. Keeps the core dependency-free
/// while letting plug-ins return any `std::error::Error`.
pub type Error = Box<dyn std::error::Error + Send + Sync + 'static>;

/// Result alias for transform operations.
pub type Result<T> = std::result::Result<T, Error>;
