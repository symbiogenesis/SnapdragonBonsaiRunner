//! Explicit source-bound independent ASR packet consumer; no native fallback.
#![forbid(unsafe_code)]

#[cfg(windows)]
#[path = "replay/consumer.rs"]
mod consumer;

use std::process::ExitCode;

fn main() -> ExitCode {
    bonsai_speech_service::replay_harness::run_cli_main(
        || {
            #[cfg(windows)]
            {
                consumer::run()
            }
            #[cfg(not(windows))]
            {
                Err(std::io::Error::new(
                    std::io::ErrorKind::Unsupported,
                    "Native Windows ARM64 is required.",
                ))
            }
        },
        "ASR replay request or report failed.",
    )
}
