//! chord-core — the chord microkernel.
//!
//! This crate is the stable, minimal **core system** of the microkernel
//! architecture. It defines:
//!
//! - [`Kind`]: the coarse data modalities plug-ins consume and produce.
//! - [`Message`]: the data plane — an ordered list of typed [`Part`]s.
//! - [`Transform`]: the plug-in contract (`Message -> Message`); [`Unary`] is
//!   the mono-modal special case.
//! - [`Registry`]: a name -> plug-in directory for CLI dispatch.
//! - [`Manifest`]: the serializable host↔engine wire description of a plug-in.
//! - [`dirs`]: the single source of truth for XDG paths.
//!
//! It links **no model/engine code** (the rule that keeps the dual-ggml symbol
//! clash out of the host); its only deps are tiny pure-Rust utility crates with
//! no native build scripts. Engines live in separate plug-in crates that depend
//! on this one — never the reverse.

pub mod dirs;
mod error;
pub mod events;
mod kind;
mod manifest;
pub mod message;
mod registry;
mod transform;

pub use error::ChordError;
pub use kind::Kind;
pub use manifest::{Manifest, ManifestOption, ManifestResource, MANIFEST_VERSION};
pub use message::{
    decode, encode, is_framed, read_delimited, write_delimited, Body, Message, Part, MAGIC_LEN,
};
pub use registry::Registry;
pub use transform::{OptionSpec, Options, ResourceSpec, Signature, Transform, Unary};

/// Boxed error type used across the kernel. Keeps the core dependency-free
/// while letting plug-ins return any `std::error::Error`.
pub type Error = Box<dyn std::error::Error + Send + Sync + 'static>;

/// Result alias for transform operations.
pub type Result<T> = std::result::Result<T, Error>;
