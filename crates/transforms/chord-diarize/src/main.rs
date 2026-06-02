//! chord-diarize binary — speaker diarization via sherpa-onnx. The `chord` host
//! spawns it out-of-process (exec-proxy) as the `diarize` transform.

fn main() -> std::process::ExitCode {
    chord_runner::run(&chord_diarize::Diarizer)
}
