//! Regenerates the native application icon from its bounded two-leaf SVG source.
#![forbid(unsafe_code)]

fn main() -> std::process::ExitCode {
    bonsai_tools::assets::run_cli()
}
