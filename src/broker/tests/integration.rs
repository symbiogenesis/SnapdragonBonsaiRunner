#![forbid(unsafe_code)]
//! Consolidated broker integration tests.
#![expect(
    clippy::unwrap_used,
    reason = "test setup and successful-operation assertions must fail loudly"
)]

mod support;

#[path = "integration/backend_http.rs"]
mod backend_http;
#[path = "integration/control_tokens.rs"]
mod control_tokens;
#[path = "integration/fragmentation.rs"]
mod fragmentation;
#[path = "integration/http_reception.rs"]
mod http_reception;
#[path = "integration/http_response.rs"]
mod http_response;
#[path = "integration/launch_policy.rs"]
mod launch_policy;
#[path = "integration/local_service.rs"]
mod local_service;
#[cfg(windows)]
#[path = "integration/markdown_worker.rs"]
mod markdown_worker;
#[path = "integration/policy_boundaries.rs"]
mod policy_boundaries;
#[path = "integration/runtime.rs"]
mod runtime;
#[path = "integration/scheduler.rs"]
mod scheduler;
#[path = "integration/scheduler_simulation.rs"]
mod scheduler_simulation;
#[path = "integration/supervisor.rs"]
mod supervisor;
