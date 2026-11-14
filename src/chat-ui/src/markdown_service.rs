//! Offline process ownership and bounded transport for untrusted Markdown parsing.
#![forbid(unsafe_code)]
#![cfg_attr(
    test,
    allow(
        dead_code,
        reason = "Headless presentation fixtures use validated parser packets without launching a real MXC process; the transport has separate framing controls."
    )
)]

use eframe::egui;
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, RecvTimeoutError, SyncSender},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use bonsai_windows_process::OwnedProcess;

mod ownership;
mod protocol;

const MAX_SOURCE_BYTES: usize = 256 * 1024;
const MAX_OUTPUT_BYTES: usize = 2 * 1024 * 1024;
const MAX_REQUESTS: usize = 16;
const POLL: Duration = Duration::from_millis(25);
const STARTUP_LIMIT: Duration = Duration::from_secs(10);
const REQUEST_LIMIT: Duration = Duration::from_secs(2);
const CLEANUP_LIMIT: Duration = Duration::from_secs(5);

static ACTIVE: AtomicBool = AtomicBool::new(false);
static ADMISSION_CLOSED: AtomicBool = AtomicBool::new(false);

/// Session-local identity for a bounded parse request; zero is an immediate failure.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct Ticket(u64);

impl Ticket {
    /// Whether admission allocated a live request identity.
    pub(crate) const fn admitted(self) -> bool {
        self.0 != 0
    }
}

/// Untrusted transport bytes still require the caller's independent typed validation.
#[derive(Clone)]
pub(crate) enum Status {
    /// Parsing has not completed; the UI keeps its current source presentation.
    Pending,
    /// A bounded response belonging to the requested source revision.
    Ready(Arc<[u8]>),
    /// Admission, transport, worker execution or cleanup failed for this session.
    Failed,
}

struct Request {
    ticket: Ticket,
    source: Arc<str>,
}

#[derive(Default)]
struct State {
    next: u64,
    failed: bool,
    results: BTreeMap<Ticket, Status>,
}

#[derive(Default)]
struct Shared {
    cancelled: AtomicBool,
    cleanup_failed: AtomicBool,
    state: Mutex<State>,
}

impl Shared {
    fn fail(&self) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.failed = true;
        for status in state.results.values_mut() {
            if matches!(status, Status::Pending) {
                *status = Status::Failed;
            }
        }
    }

    fn pending(&self, ticket: Ticket) -> bool {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .results
            .get(&ticket)
            .is_some_and(|status| matches!(status, Status::Pending))
    }

    fn complete(&self, ticket: Ticket, bytes: Vec<u8>) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.failed || self.cancelled.load(Ordering::Acquire) {
            return;
        }
        if let Some(status) = state.results.get_mut(&ticket) {
            *status = Status::Ready(bytes.into());
        }
    }
}

/// Context-owned, lazy, single-worker Markdown transport with no parser in the UI.
pub(crate) struct Service {
    shared: Arc<Shared>,
    requests: SyncSender<Request>,
    worker: Mutex<Option<JoinHandle<()>>>,
}

impl Service {
    /// Supplies the verified headless image after native startup prepares it.
    pub(crate) fn configure(context: &egui::Context, image: &Path) {
        context.data_mut(|data| {
            data.insert_temp(egui::Id::unique("offline-markdown-image"), image.to_owned());
        });
    }

    /// Separates a terminal session failure from transient queue backpressure.
    pub(crate) fn unavailable(&self) -> bool {
        let state = self
            .shared
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let unavailable = state.failed || state.next == u64::MAX;
        drop(state);
        unavailable
            || self.shared.cancelled.load(Ordering::Acquire)
            || ADMISSION_CLOSED.load(Ordering::Acquire)
    }
    /// Reuses one service for the native UI context; process setup stays off its thread.
    pub(crate) fn get(context: &egui::Context) -> Option<Arc<Self>> {
        context.data_mut(|data| {
            let identity = egui::Id::unique("offline-markdown-service");
            if let Some(service) = data.get_temp::<Arc<Self>>(identity) {
                return Some(service);
            }
            // Source launches prepare the broker after the first UI frame.
            // Restored messages wait without poisoning the parser session.
            let image = data.get_temp::<PathBuf>(egui::Id::unique("offline-markdown-image"))?;
            let service = Self::new(image);
            data.insert_temp(identity, Arc::clone(&service));
            Some(service)
        })
    }

    fn new(image: PathBuf) -> Arc<Self> {
        let shared = Arc::new(Shared::default());
        let (requests, pending) = mpsc::sync_channel(MAX_REQUESTS);
        let state = Arc::clone(&shared);
        let worker = thread::Builder::new()
            .name("bonsai-markdown-supervisor".to_owned())
            .spawn(move || supervise(&state, &pending, &image));
        let worker = worker.map_or_else(
            |_| {
                shared.fail();
                None
            },
            Some,
        );
        Arc::new(Self {
            shared,
            requests,
            worker: Mutex::new(worker),
        })
    }

    /// Admits at most sixteen source/result owners without blocking the event loop.
    pub(crate) fn request(&self, source: Arc<str>) -> Ticket {
        if source.len() > MAX_SOURCE_BYTES || self.shared.cancelled.load(Ordering::Acquire) {
            return Ticket(0);
        }
        let mut state = self
            .shared
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.failed || state.results.len() >= MAX_REQUESTS {
            return Ticket(0);
        }
        let Some(next) = state.next.checked_add(1) else {
            return Ticket(0);
        };
        state.next = next;
        let ticket = Ticket(next);
        state.results.insert(ticket, Status::Pending);
        if self.requests.try_send(Request { ticket, source }).is_err() {
            state.results.remove(&ticket);
            return Ticket(0);
        }
        drop(state);
        ticket
    }

    /// Polls a result without waiting; absent or released identities fail closed.
    pub(crate) fn poll(&self, ticket: Ticket) -> Status {
        self.shared
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .results
            .get(&ticket)
            .cloned()
            .unwrap_or(Status::Failed)
    }

    /// Releases obsolete pending/results; queued obsolete sources are skipped before IPC.
    pub(crate) fn release(&self, ticket: Ticket) {
        self.shared
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .results
            .remove(&ticket);
    }

    /// Closes the parsing session after independently detecting malformed output.
    pub(crate) fn reject(&self) {
        self.shared.cancelled.store(true, Ordering::Release);
        self.shared.fail();
    }

    /// Reports retained uncertain process/storage cleanup separately from a parse failure.
    pub(crate) fn cleanup_failed(&self) -> bool {
        self.shared.cleanup_failed.load(Ordering::Acquire)
    }
}

impl Drop for Service {
    fn drop(&mut self) {
        self.shared.cancelled.store(true, Ordering::Release);
        if let Some(worker) = self
            .worker
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
        {
            let _joined = worker.join();
        }
    }
}

struct Active;

impl Active {
    fn acquire() -> Option<Self> {
        if ADMISSION_CLOSED.load(Ordering::Acquire) {
            return None;
        }
        ACTIVE
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .ok()
            .map(|_| Self)
    }
}

impl Drop for Active {
    fn drop(&mut self) {
        ACTIVE.store(false, Ordering::Release);
    }
}

fn next_request(shared: &Shared, requests: &Receiver<Request>) -> Option<Request> {
    while !shared.cancelled.load(Ordering::Acquire) {
        match requests.recv_timeout(POLL) {
            Ok(request) if shared.pending(request.ticket) => return Some(request),
            Ok(_) | Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => return None,
        }
    }
    None
}

fn supervise(shared: &Shared, requests: &Receiver<Request>, image: &Path) {
    let Some(first) = next_request(shared, requests) else {
        return;
    };
    let Some(_active) = Active::acquire() else {
        shared.fail();
        return;
    };
    let startup = Instant::now() + STARTUP_LIMIT;
    let Ok(mut prepared) = ownership::Prepared::new(image) else {
        shared.fail();
        return;
    };
    let (process, input, output) = match OwnedProcess::start_sandboxed_piped_minimal(
        &prepared.specification,
        &prepared.policy,
    ) {
        Ok(started) => started,
        Err(error) => {
            if bonsai_windows_process::process_start_cleanup_unconfirmed(&error) {
                ownership::quarantine(prepared, None);
                shared.cleanup_failed.store(true, Ordering::Release);
            } else if prepared.cleanup().is_err() {
                shared.cleanup_failed.store(true, Ordering::Release);
                ADMISSION_CLOSED.store(true, Ordering::Release);
            }
            shared.fail();
            return;
        }
    };
    let (commands, commands_rx) = mpsc::sync_channel(1);
    let (events_tx, events) = mpsc::sync_channel(1);
    let reader = thread::Builder::new()
        .name("bonsai-markdown-pipe".to_owned())
        .spawn(move || protocol::exchange(input, output, &commands_rx, &events_tx));
    let Ok(reader) = reader else {
        shared.fail();
        if process.stop(CLEANUP_LIMIT).is_err() {
            ownership::quarantine(prepared, None);
            shared.cleanup_failed.store(true, Ordering::Release);
        } else if prepared.cleanup().is_err() {
            shared.cleanup_failed.store(true, Ordering::Release);
            ADMISSION_CLOSED.store(true, Ordering::Release);
        }
        return;
    };
    let ready = await_event(shared, &process, &events, startup)
        .is_some_and(|event| matches!(event, protocol::Event::Ready));
    if ready {
        let mut request = Some(first);
        while let Some(current) = request {
            if !shared.cancelled.load(Ordering::Acquire) && shared.pending(current.ticket) {
                let ticket = current.ticket;
                if commands.try_send(current).is_err() {
                    break;
                }
                let deadline = Instant::now() + REQUEST_LIMIT;
                let Some(protocol::Event::Response(actual, bytes)) =
                    await_event(shared, &process, &events, deadline)
                else {
                    break;
                };
                if actual != ticket {
                    break;
                }
                shared.complete(ticket, bytes);
            }
            request = next_request(shared, requests);
        }
    }
    shared.fail();
    // Release channel waiters before terminating the exact process: a hostile
    // worker may have stalled either a pipe read or a pipe write.
    drop(commands);
    drop(events);
    let cleanup = Instant::now() + CLEANUP_LIMIT;
    let stopped = process.stop(CLEANUP_LIMIT).is_ok();
    while stopped && !reader.is_finished() && Instant::now() < cleanup {
        thread::sleep(POLL.min(cleanup.saturating_duration_since(Instant::now())));
    }
    if !stopped || !reader.is_finished() {
        ownership::quarantine(prepared, Some(reader));
        shared.cleanup_failed.store(true, Ordering::Release);
    } else {
        let _joined = reader.join();
        if prepared.cleanup().is_err() {
            shared.cleanup_failed.store(true, Ordering::Release);
            ADMISSION_CLOSED.store(true, Ordering::Release);
        }
    }
}

fn await_event(
    shared: &Shared,
    process: &OwnedProcess,
    events: &Receiver<protocol::Event>,
    deadline: Instant,
) -> Option<protocol::Event> {
    while !shared.cancelled.load(Ordering::Acquire) && Instant::now() < deadline {
        match events.recv_timeout(POLL.min(deadline.saturating_duration_since(Instant::now()))) {
            Ok(event) => return Some(event),
            Err(RecvTimeoutError::Disconnected) => return None,
            Err(RecvTimeoutError::Timeout) => {}
        }
        if !matches!(process.try_wait(), Ok(None)) {
            return None;
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;

    #[test]
    fn markdown_waits_for_prepared_image_and_reuses_one_context_service() -> io::Result<()> {
        let context = egui::Context::default();
        assert!(Service::get(&context).is_none());
        Service::configure(&context, Path::new("C:\\prepared\\broker.exe"));
        let first = Service::get(&context).ok_or_else(|| io::Error::other("Missing service"))?;
        let second = Service::get(&context).ok_or_else(|| io::Error::other("Missing service"))?;
        assert!(Arc::ptr_eq(&first, &second));
        assert!(!first.unavailable());
        // No request is admitted here, so the supervisor never launches a child.
        Ok(())
    }

    #[test]
    fn markdown_queue_backpressure_is_temporary_and_rejection_stays_terminal() -> io::Result<()> {
        // Inert channels exercise admission without a supervisor or native child.
        let (requests, pending) = mpsc::sync_channel(MAX_REQUESTS);
        let service = Service {
            shared: Arc::new(Shared::default()),
            requests,
            worker: Mutex::new(None),
        };
        for _ in 0..MAX_REQUESTS {
            assert!(service.request(Arc::from("界")).admitted());
        }
        assert!(!service.request(Arc::from("busy")).admitted());
        assert!(!service.unavailable());
        let released = pending.try_recv().map_err(io::Error::other)?;
        service.release(released.ticket);
        let ticket = service.request(Arc::from("available"));
        assert!(ticket.admitted());
        service.reject();
        service.shared.complete(ticket, b"late response".to_vec());
        assert!(matches!(service.poll(ticket), Status::Failed));
        assert!(service.unavailable());
        assert!(!service.request(Arc::from("closed")).admitted());
        Ok(())
    }
}
