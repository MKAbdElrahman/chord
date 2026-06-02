//! chord-tts binary — the text-to-speech engine as a standalone Unix filter.
//! The `chord` host spawns this out-of-process (see chord-cli's exec-proxy).

fn main() -> std::process::ExitCode {
    chord_runner::run(&chord_tts::Tts)
}
