//! Consolidated runtime lifetime and offline provisioning regressions.
#![cfg(windows)]
#![forbid(unsafe_code)]

#[path = "staging/lifetime.rs"]
mod lifetime;
#[path = "staging/provisioning.rs"]
mod provisioning;
