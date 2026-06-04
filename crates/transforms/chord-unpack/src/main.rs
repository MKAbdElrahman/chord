//! chord-unpack binary — inspect/extract message parts as a standalone Unix
//! filter. The `chord` host spawns this out-of-process (see chord-cli's exec-proxy).

fn main() -> std::process::ExitCode {
    chord_runner::run(&chord_unpack::Unpack)
}
