//! Explicit source-bound independent TTS replay; no audio device or native fallback.
#![forbid(unsafe_code)]

use std::process::ExitCode;

fn main() -> ExitCode {
    bonsai_speech_service::replay_harness::run_cli_main(
        || {
            #[cfg(windows)]
            {
                bonsai_speech_tts::replay_consumer::run()
            }
            #[cfg(not(windows))]
            {
                Err(std::io::Error::new(
                    std::io::ErrorKind::Unsupported,
                    "Native Windows ARM64 is required.",
                ))
            }
        },
        "TTS replay request or report failed.",
    )
}
