//! chord-stt-llama binary — speech-to-text via llama.cpp's `mtmd` audio path,
//! i.e. audio-LLM ASR (Voxtral, Ultravox, Qwen-audio). An alternate `stt`
//! backend; the host selects it with `chord stt --backend llama.cpp`.

fn main() -> std::process::ExitCode {
    chord_runner::run(&chord_stt_llama::SttLlama)
}
