//! Consolidated integration tests for `bonsai-tools`.
#![forbid(unsafe_code)]

#[path = "integration/cli.rs"]
mod cli;
#[path = "integration/credentials.rs"]
mod credentials;
#[cfg(all(windows, target_arch = "aarch64"))]
#[path = "integration/maintenance.rs"]
mod maintenance;
#[cfg(windows)]
#[path = "integration/provisioning.rs"]
mod provisioning;
#[cfg(windows)]
#[path = "integration/release_lock.rs"]
mod release_lock;
#[cfg(all(windows, target_arch = "aarch64"))]
#[path = "integration/runtime_handles.rs"]
mod runtime_handles;
