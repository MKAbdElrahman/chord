//! XDG base directories for chord — the single source of truth for paths.
//!
//! Every path chord uses (config, models, caches) derives from the platform's
//! standard base directories via the `directories` crate, so chord honors
//! `$XDG_CONFIG_HOME`, `$XDG_DATA_HOME`, and `$XDG_CACHE_HOME` on Linux instead
//! of scattering files across ad-hoc dotdirs. On Linux this resolves to:
//!
//! ```text
//! config  ~/.config/chord
//! data    ~/.local/share/chord   (models live here, under data/models)
//! cache   ~/.cache/chord
//! ```

use std::path::PathBuf;

use directories::ProjectDirs;

/// The chord `ProjectDirs`. Falls back to `.`-relative paths only in the
/// pathological case where no home directory can be determined.
fn project() -> Option<ProjectDirs> {
    ProjectDirs::from("", "", "chord")
}

/// Configuration directory (`$XDG_CONFIG_HOME/chord`).
pub fn config_dir() -> PathBuf {
    project()
        .map(|p| p.config_dir().to_path_buf())
        .unwrap_or_else(|| PathBuf::from(".config/chord"))
}

/// Data directory (`$XDG_DATA_HOME/chord`) — durable, user-generated state.
pub fn data_dir() -> PathBuf {
    project()
        .map(|p| p.data_dir().to_path_buf())
        .unwrap_or_else(|| PathBuf::from(".local/share/chord"))
}

/// Cache directory (`$XDG_CACHE_HOME/chord`) — regenerable scratch.
pub fn cache_dir() -> PathBuf {
    project()
        .map(|p| p.cache_dir().to_path_buf())
        .unwrap_or_else(|| PathBuf::from(".cache/chord"))
}

/// Where downloaded models live (`data_dir()/models`). The home for every
/// engine's weights, replacing the old scattered `~/models`, `~/.kronk/models`,
/// and `~/.yapper` locations.
pub fn models_dir() -> PathBuf {
    data_dir().join("models")
}
