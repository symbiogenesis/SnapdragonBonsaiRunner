#![expect(
    clippy::unwrap_used,
    reason = "shared test runtime setup must panic loudly on failure"
)]
#![allow(
    dead_code,
    reason = "shared helpers are used selectively across independent integration test targets"
)]

use std::time::Duration;
use tokio::time::Instant;

pub(crate) const CLIENT_KEY: &str =
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
pub(crate) const BACKEND_KEY: &str =
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

pub(crate) fn run(test: impl Future<Output = ()>) {
    tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .start_paused(true)
        .build()
        .unwrap()
        .block_on(test);
}

pub(crate) fn deadline() -> Instant {
    Instant::now() + Duration::from_mins(1)
}
