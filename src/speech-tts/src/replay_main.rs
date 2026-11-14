//! Explicit source-bound independent TTS replay; no audio device or native fallback.
#![forbid(unsafe_code)]

#[cfg(windows)]
#[path = "replay/consumer.rs"]
mod consumer;

use std::{
    io::{self, Write as _},
    process::ExitCode,
};

fn run() -> io::Result<bool> {
    #[cfg(windows)]
    {
        consumer::run()
    }
    #[cfg(not(windows))]
    {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "Native Windows ARM64 is required.",
        ))
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(_) => {
            let _written = writeln!(io::stderr().lock(), "TTS replay request or report failed.");
            ExitCode::FAILURE
        }
    }
}
