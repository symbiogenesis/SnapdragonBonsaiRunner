//! Consolidated integration tests for `bonsai-ci`.
#![forbid(unsafe_code)]

#[path = "integration/cli.rs"]
mod cli;
#[cfg(all(windows, target_arch = "aarch64"))]
#[path = "integration/push_format.rs"]
mod push_format;
