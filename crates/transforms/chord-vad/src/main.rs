//! chord-vad binary — Silero VAD via sherpa-onnx; the `vad` transform.

fn main() -> std::process::ExitCode {
    chord_runner::run(&chord_vad::Vad)
}
