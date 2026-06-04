//! chord-pack binary — assemble a multimodal message as a standalone Unix
//! filter. The `chord` host spawns this out-of-process (see chord-cli's exec-proxy).

fn main() -> std::process::ExitCode {
    chord_runner::run(&chord_pack::Pack)
}
