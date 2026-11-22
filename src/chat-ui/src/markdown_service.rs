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
        Arc, Mutex, MutexGuard,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, RecvTimeoutError, SyncSender},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use crate::isolated_worker::WorkerSlot;
use bonsai_windows_process::OwnedProcess;

mod ownership;
mod protocol;
mod raster;

pub(crate) use raster::{DecodedRaster, MAX_RASTER_DIMENSION, decode_raster};

const MAX_SOURCE_BYTES: usize = 256 * 1024;
const MAX_SVG_BYTES: usize = 256 * 1024;
const MAX_OUTPUT_BYTES: usize = 2 * 1024 * 1024;
const MAX_REQUESTS: usize = 32;
const POLL: Duration = Duration::from_millis(25);
const STARTUP_LIMIT: Duration = Duration::from_secs(10);
const REQUEST_LIMIT: Duration = Duration::from_secs(2);
const CLEANUP_LIMIT: Duration = Duration::from_secs(5);

static SLOT: WorkerSlot = WorkerSlot::new();

/// Session-local identity for a bounded parse or raster request; zero is an immediate failure.
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
    /// Parsing or rasterization has not completed; the UI keeps its current presentation.
    Pending,
    /// A bounded response belonging to the requested revision.
    Ready(Arc<[u8]>),
    /// Admission, transport, worker execution or cleanup failed for this session.
    Failed,
}

enum RequestKind {
    Markdown(Arc<str>),
    Rasterize {
        max_width: u32,
        max_height: u32,
        svg: Arc<str>,
    },
}

struct Request {
    ticket: Ticket,
    kind: RequestKind,
}

#[derive(Default)]
struct State {
    next: u64,
    ready: bool,
    failed: bool,
    results: BTreeMap<Ticket, Status>,
}

#[derive(Default)]
struct Shared {
    cancelled: AtomicBool,
    warmup: AtomicBool,
    cleanup_failed: AtomicBool,
    state: Mutex<State>,
}

impl Shared {
    fn state(&self) -> MutexGuard<'_, State> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn ready(&self) {
        self.state().ready = true;
    }

    fn fail(&self) {
        let mut state = self.state();
        state.failed = true;
        for status in state.results.values_mut() {
            if matches!(status, Status::Pending) {
                *status = Status::Failed;
            }
        }
    }

    fn pending(&self, ticket: Ticket) -> bool {
        self.state()
            .results
            .get(&ticket)
            .is_some_and(|status| matches!(status, Status::Pending))
    }

    fn complete(&self, ticket: Ticket, bytes: Vec<u8>) {
        let mut state = self.state();
        if state.failed || self.cancelled.load(Ordering::Acquire) {
            return;
        }
        if let Some(status) = state.results.get_mut(&ticket) {
            *status = Status::Ready(bytes.into());
        }
    }
}

/// Context-owned single-worker Markdown transport with no parser in the UI.
pub(crate) struct Service {
    shared: Arc<Shared>,
    requests: SyncSender<Request>,
    worker: Mutex<Option<JoinHandle<()>>>,
}

impl Service {
    #[cfg(test)]
    fn inert(context: &egui::Context) -> (Arc<Self>, Receiver<Request>) {
        let (requests, pending) = mpsc::sync_channel(MAX_REQUESTS);
        let service = Arc::new(Self {
            shared: Arc::new(Shared::default()),
            requests,
            worker: Mutex::new(None),
        });
        context.data_mut(|data| {
            data.insert_temp(
                egui::Id::unique("offline-markdown-service"),
                Arc::clone(&service),
            );
        });
        (service, pending)
    }

    #[cfg(test)]
    pub(crate) fn custom_responder<T: FnMut(&str, Option<(u32, u32)>) -> Vec<u8>>(
        context: &egui::Context,
        mut reply: T,
    ) -> (
        Arc<Self>,
        impl FnMut() -> std::io::Result<Arc<str>> + use<T>,
    ) {
        let (service, pending) = Self::inert(context);
        service.shared.ready();
        let responder_service = Arc::clone(&service);
        let step = move || {
            let request = pending.try_recv().map_err(std::io::Error::other)?;
            match request.kind {
                RequestKind::Markdown(source) => {
                    let payload = reply(&source, None);
                    responder_service.shared.complete(request.ticket, payload);
                    Ok(source)
                }
                RequestKind::Rasterize {
                    max_width,
                    max_height,
                    svg,
                } => {
                    let payload = reply(&svg, Some((max_width, max_height)));
                    responder_service.shared.complete(request.ticket, payload);
                    Ok(svg)
                }
            }
        };
        (service, step)
    }

    #[cfg(test)]
    pub(crate) fn fixture(
        context: &egui::Context,
    ) -> impl Fn() -> std::io::Result<Arc<str>> + use<> {
        let (service, pending) = Self::inert(context);
        service.shared.ready();
        move || {
            let request = pending.try_recv().map_err(std::io::Error::other)?;
            match request.kind {
                RequestKind::Markdown(source) => {
                    service
                        .shared
                        .complete(request.ticket, bonsai_markdown::parse(&source));
                    Ok(source)
                }
                RequestKind::Rasterize {
                    max_width,
                    max_height,
                    svg,
                } => {
                    service.shared.complete(
                        request.ticket,
                        bonsai_markdown::rasterize(&svg, max_width, max_height),
                    );
                    Ok(svg)
                }
            }
        }
    }

    /// Starts the sandbox without a parse request; true means handshake pending.
    /// Terminal formatting failure lets independent model preparation continue.
    pub(crate) fn warm_up(&self) -> bool {
        self.shared.warmup.store(true, Ordering::Release);
        let state = self.shared.state();
        !state.ready
            && !state.failed
            && !self.shared.cancelled.load(Ordering::Acquire)
            && !SLOT.is_closed()
    }

    /// Supplies the verified headless image after native startup prepares it.
    pub(crate) fn configure(context: &egui::Context, image: &Path) {
        context.data_mut(|data| {
            data.insert_temp(egui::Id::unique("offline-markdown-image"), image.to_owned());
        });
    }

    /// Separates a terminal session failure from transient queue backpressure.
    pub(crate) fn unavailable(&self) -> bool {
        let state = self.shared.state();
        let unavailable = state.failed || state.next == u64::MAX;
        drop(state);
        unavailable || self.shared.cancelled.load(Ordering::Acquire) || SLOT.is_closed()
    }
    /// Reuses one service for the native UI context; process setup stays off its thread.
    pub(crate) fn get(context: &egui::Context) -> Option<Arc<Self>> {
        context.data_mut(|data| {
            let identity = egui::Id::unique("offline-markdown-service");
            if let Some(service) = data.get_temp::<Arc<Self>>(identity) {
                return Some(service);
            }
            // Legacy launchers can prepare the broker after the first UI frame.
            // Restored messages wait without poisoning the parser session.
            let image = data.get_temp::<PathBuf>(egui::Id::unique("offline-markdown-image"))?;
            let service = Self::new(image);
            data.insert_temp(identity, Arc::clone(&service));
            Some(service)
        })
    }

    /// Borrows a cached service handle or resolves and caches it from `context`.
    pub(crate) fn get_or_init<'a>(
        slot: &'a mut Option<Arc<Self>>,
        context: &egui::Context,
    ) -> Option<&'a Self> {
        // Polonius lets a warm slot lend directly while a cold slot ends the
        // borrow to resolve and cache the context service.
        if let Some(service) = slot.as_deref() {
            return Some(service);
        }
        *slot = Self::get(context);
        slot.as_deref()
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

    /// Admits a bounded Markdown parse request without blocking the event loop.
    pub(crate) fn request(&self, source: Arc<str>) -> Ticket {
        if source.len() > MAX_SOURCE_BYTES {
            return Ticket(0);
        }
        self.admit(RequestKind::Markdown(source))
    }

    /// Admits a bounded offline SVG rasterization request without blocking the event loop.
    pub(crate) fn request_raster(&self, svg: Arc<str>, max_width: u32, max_height: u32) -> Ticket {
        if svg.is_empty()
            || svg.len() > MAX_SVG_BYTES
            || !(1..=MAX_RASTER_DIMENSION).contains(&max_width)
            || !(1..=MAX_RASTER_DIMENSION).contains(&max_height)
        {
            return Ticket(0);
        }
        self.admit(RequestKind::Rasterize {
            max_width,
            max_height,
            svg,
        })
    }

    fn admit(&self, kind: RequestKind) -> Ticket {
        if self.shared.cancelled.load(Ordering::Acquire) {
            return Ticket(0);
        }
        let mut state = self.shared.state();
        if state.failed || state.results.len() >= MAX_REQUESTS {
            return Ticket(0);
        }
        let Some(next) = state.next.checked_add(1) else {
            return Ticket(0);
        };
        state.next = next;
        let ticket = Ticket(next);
        state.results.insert(ticket, Status::Pending);
        if self.requests.try_send(Request { ticket, kind }).is_err() {
            state.results.remove(&ticket);
            return Ticket(0);
        }
        drop(state);
        ticket
    }

    /// Polls a result without waiting; absent or released identities fail closed.
    pub(crate) fn poll(&self, ticket: Ticket) -> Status {
        self.shared
            .state()
            .results
            .get(&ticket)
            .cloned()
            .unwrap_or(Status::Failed)
    }

    /// Releases obsolete pending/results; queued obsolete sources are skipped before IPC.
    pub(crate) fn release(&self, ticket: Ticket) {
        self.shared.state().results.remove(&ticket);
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

fn finish_prepared_cleanup(
    mut prepared: ownership::Prepared,
    reader: Option<JoinHandle<()>>,
    need_quarantine: bool,
    shared: &Shared,
) {
    if need_quarantine {
        ownership::quarantine(prepared, reader);
        shared.cleanup_failed.store(true, Ordering::Release);
    } else {
        if let Some(reader) = reader {
            let _joined = reader.join();
        }
        if prepared.cleanup().is_err() {
            shared.cleanup_failed.store(true, Ordering::Release);
            SLOT.close_admission();
        }
    }
}

fn next_request(
    shared: &Shared,
    requests: &Receiver<Request>,
    allow_warmup: bool,
) -> Option<Request> {
    while !shared.cancelled.load(Ordering::Acquire) {
        if allow_warmup && shared.warmup.load(Ordering::Acquire) {
            return None;
        }
        match requests.recv_timeout(POLL) {
            Ok(request) if shared.pending(request.ticket) => return Some(request),
            Ok(_) | Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => return None,
        }
    }
    None
}

fn supervise(shared: &Shared, requests: &Receiver<Request>, image: &Path) {
    let first = next_request(shared, requests, true);
    if shared.cancelled.load(Ordering::Acquire)
        || (first.is_none() && !shared.warmup.load(Ordering::Acquire))
    {
        return;
    }
    let Ok(_active) = SLOT.acquire() else {
        shared.fail();
        return;
    };
    let startup = Instant::now() + STARTUP_LIMIT;
    let Ok(prepared) = ownership::Prepared::new(image) else {
        shared.fail();
        return;
    };
    let (process, input, output) = match OwnedProcess::start_sandboxed_piped_minimal(
        &prepared.specification,
        &prepared.policy,
    ) {
        Ok(started) => started,
        Err(error) => {
            finish_prepared_cleanup(
                prepared,
                None,
                bonsai_windows_process::process_start_cleanup_unconfirmed(&error),
                shared,
            );
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
        finish_prepared_cleanup(prepared, None, process.stop(CLEANUP_LIMIT).is_err(), shared);
        return;
    };
    let ready = await_event(shared, &process, &events, startup)
        .is_some_and(|event| matches!(event, protocol::Event::Ready));
    if ready {
        shared.ready();
        let mut request = first.or_else(|| next_request(shared, requests, false));
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
            request = next_request(shared, requests, false);
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
    let quarantine = !stopped || !reader.is_finished();
    finish_prepared_cleanup(prepared, Some(reader), quarantine, shared);
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
    fn markdown_warmup_preserves_queued_source_without_a_synthetic_request() -> io::Result<()> {
        let context = egui::Context::default();
        let (service, pending) = Service::inert(&context);
        assert!(service.warm_up());
        assert!(pending.try_recv().is_err());
        let source: Arc<str> = Arc::from("**A restored answer**");
        let ticket = service.request(Arc::clone(&source));
        assert!(ticket.admitted());
        assert!(next_request(&service.shared, &pending, true).is_none());
        let request = next_request(&service.shared, &pending, false)
            .ok_or_else(|| io::Error::other("Warmup consumed the queued source"))?;
        assert_eq!(request.ticket, ticket);
        assert!(
            matches!(request.kind, RequestKind::Markdown(ref queued) if Arc::ptr_eq(queued, &source))
        );
        service.shared.ready();
        assert!(!service.warm_up());
        service.reject();
        assert!(!service.warm_up());
        assert!(matches!(service.poll(ticket), Status::Failed));
        Ok(())
    }

    #[test]
    fn markdown_waits_for_prepared_image_and_reuses_one_context_service() -> io::Result<()> {
        let context = egui::Context::default();
        let mut slot = None;
        assert!(Service::get(&context).is_none());
        assert!(Service::get_or_init(&mut slot, &context).is_none());
        Service::configure(&context, Path::new("C:\\prepared\\broker.exe"));
        let first = Service::get(&context).ok_or_else(|| io::Error::other("Missing service"))?;
        let second = Service::get(&context).ok_or_else(|| io::Error::other("Missing service"))?;
        assert!(Arc::ptr_eq(&first, &second));
        let cached = Service::get_or_init(&mut slot, &context)
            .ok_or_else(|| io::Error::other("Missing service from get_or_init"))?;
        assert!(std::ptr::eq(cached, Arc::as_ptr(&first)));
        let reused = Service::get_or_init(&mut slot, &context)
            .ok_or_else(|| io::Error::other("Missing cached service from get_or_init"))?;
        assert!(std::ptr::eq(reused, Arc::as_ptr(&first)));
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

    #[test]
    fn svg_raster_admission_enforces_bounds_and_dispatches_to_queue() -> io::Result<()> {
        let context = egui::Context::default();
        let (service, pending) = Service::inert(&context);
        let svg: Arc<str> = Arc::from(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="8" height="8"><rect width="8" height="8" fill="red"/></svg>"#,
        );
        assert!(!service.request_raster(Arc::from(""), 64, 64).admitted());
        assert!(
            !service
                .request_raster(Arc::from("x".repeat(MAX_SVG_BYTES + 1)), 64, 64)
                .admitted()
        );
        assert!(!service.request_raster(Arc::clone(&svg), 0, 64).admitted());
        assert!(!service.request_raster(Arc::clone(&svg), 64, 0).admitted());
        assert!(
            !service
                .request_raster(Arc::clone(&svg), MAX_RASTER_DIMENSION + 1, 64)
                .admitted()
        );
        assert!(
            !service
                .request_raster(Arc::clone(&svg), 64, MAX_RASTER_DIMENSION + 1)
                .admitted()
        );
        let ticket = service.request_raster(Arc::clone(&svg), 128, 64);
        assert!(ticket.admitted());
        let request = pending.try_recv().map_err(io::Error::other)?;
        assert_eq!(request.ticket, ticket);
        assert!(matches!(
            request.kind,
            RequestKind::Rasterize {
                max_width: 128,
                max_height: 64,
                ref svg,
            } if svg.starts_with("<svg")
        ));
        Ok(())
    }
}
