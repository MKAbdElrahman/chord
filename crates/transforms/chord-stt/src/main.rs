//! chord-stt binary — the speech-to-text engine as a standalone Unix filter.
//! The `chord` host spawns this out-of-process (see chord-cli's exec-proxy).

fn main() -> std::process::ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--batch") {
        // Batch: load the model once, transcribe a list of files from stdin.
        return match chord_stt::run_batch(&args) {
            Ok(()) => std::process::ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("chord-stt: {e}");
                std::process::ExitCode::FAILURE
            }
        };
    }
    chord_runner::run(&chord_stt::Stt)
}
