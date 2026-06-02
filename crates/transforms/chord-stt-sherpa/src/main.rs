//! chord-stt-sherpa binary — speech-to-text via sherpa-onnx (NeMo transducer,
//! e.g. Parakeet-TDT multilingual). An alternate `stt` backend; the host selects
//! it with `chord stt --backend sherpa-onnx`.

fn main() -> std::process::ExitCode {
    chord_runner::run(&chord_stt_sherpa::SttSherpa)
}
