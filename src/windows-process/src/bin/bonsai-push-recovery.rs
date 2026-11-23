//! One bounded, offline rebuild of the source-sanity checker.
#![forbid(unsafe_code)]

use std::process::ExitCode;

fn main() -> ExitCode {
    #[cfg(windows)]
    {
        bonsai_windows_process::push_recovery::main_exit_code()
    }
    #[cfg(not(windows))]
    {
        let _ = std::io::Write::write_fmt(
            &mut std::io::stderr().lock(),
            format_args!("Push checker recovery requires native Windows ARM64.\n"),
        );
        ExitCode::FAILURE
    }
}
