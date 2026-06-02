//! chord-langid binary — spoken language ID via sherpa-onnx; the `langid` transform.

fn main() -> std::process::ExitCode {
    chord_runner::run(&chord_langid::LangId)
}
