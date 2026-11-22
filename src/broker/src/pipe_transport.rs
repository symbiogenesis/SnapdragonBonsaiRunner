//! Parent-side framed HTTP/1.1 stream multiplexer over anonymous child pipes.
//!
//! Used by natively contained broker-pipe inference children running in an
//! offline `AppContainer` without loopback or external network capability.
//! Completes the `SB2PIPE1`/`SB2OKAY1` in-memory bearer handshake, waits for
//! `KIND_SERVER_READY` emitted after worker initialization, and multiplexes
//! sequential logical streams over `(child_stdin, child_stdout)` with isolated
//! `KIND_RESET` cancellation for abandoned generation requests.

use crate::{
    backend::{BackendError, BackendPolicy},
    supervisor::BackendConnector,
};
use std::{
    collections::VecDeque,
    fs::File,
    io::{self, Read, Write},
    pin::Pin,
    sync::{
        Arc, Condvar, Mutex, MutexGuard,
        atomic::{AtomicBool, AtomicU32, Ordering},
    },
    task::{Context, Poll, Waker},
};
use tokio::{
    io::{AsyncRead, AsyncWrite, ReadBuf},
    sync::{mpsc, watch},
    time::{Instant, timeout_at},
};

/// Fixed host-header authority port paired with broker-pipe requests.
pub const BROKER_PIPE_HOST_PORT: u16 = 1024;

const HANDSHAKE_REQUEST_MAGIC: &[u8; 8] = b"SB2PIPE1";
const HANDSHAKE_RESPONSE_MAGIC: &[u8; 8] = b"SB2OKAY1";
const KEY_HEX_BYTES: usize = 64;
const FRAME_HEADER_BYTES: usize = 9;
const MAX_FRAME_PAYLOAD: usize = 16_384;
const FREE_LIST_CAPACITY: usize = 32;
const CHANNEL_CAPACITY: usize = 64;
const MAX_TRACKED_RESETS: usize = 16;

struct BufferPool(Mutex<Vec<Vec<u8>>>);

impl BufferPool {
    fn new() -> Arc<Self> {
        Arc::new(Self(Mutex::new(Vec::with_capacity(FREE_LIST_CAPACITY))))
    }

    fn take_raw(&self) -> Vec<u8> {
        lock_unpoisoned(&self.0)
            .pop()
            .unwrap_or_else(|| Vec::with_capacity(MAX_FRAME_PAYLOAD))
    }

    fn take_zeroed(&self, len: usize) -> Vec<u8> {
        let mut buf = self.take_raw();
        buf.resize(len, 0);
        buf
    }

    fn take_from_slice(&self, bytes: &[u8]) -> Vec<u8> {
        let mut buf = self.take_raw();
        buf.clear();
        buf.extend_from_slice(bytes);
        buf
    }

    fn recycle(&self, mut buf: Vec<u8>) {
        if !(MAX_FRAME_PAYLOAD..=MAX_FRAME_PAYLOAD * 2).contains(&buf.capacity()) {
            return;
        }
        crate::scrub_bytes(&mut buf);
        buf.clear();
        let mut list = lock_unpoisoned(&self.0);
        if list.len() < FREE_LIST_CAPACITY {
            list.push(buf);
        }
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        lock_unpoisoned(&self.0).len()
    }
}

pub(crate) const KIND_OPEN: u8 = 1;
pub(crate) const KIND_DATA: u8 = 2;
pub(crate) const KIND_EOF: u8 = 3;
pub(crate) const KIND_RESET: u8 = 4;
pub(crate) const KIND_SERVER_READY: u8 = 5;

/// Control frame kind carrying one parent-verified read-only section attestation (`stream_id == 0`).
pub const KIND_VERIFIED_SECTION: u8 = 6;

/// Maximum number of verified read-only section attestations sent during pipe startup.
pub const MAX_VERIFIED_SECTIONS: usize = 8;

const VERIFIED_SECTION_PAYLOAD_BYTES: usize = 52;

/// Attestation metadata for one parent-verified read-only file section inherited by the worker.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VerifiedSectionAttestation {
    /// Zero-based section slot index matching the inherited handle order.
    pub slot: u32,
    /// Inherited Windows `HANDLE` value valid in the child process.
    pub child_handle: u64,
    /// Nonzero byte length of the verified regular file backed by the section.
    pub bytes: u64,
    /// Expected SHA-256 digest of the verified section contents.
    pub sha256: [u8; 32],
}

/// Encodes one [`VerifiedSectionAttestation`] into its canonical 52-byte little-endian payload.
///
/// Wire layout: `slot (4 LE) || child_handle (8 LE) || bytes (8 LE) || sha256 (32)`.
///
/// # Errors
/// Returns [`BackendError::Configuration`] when `child_handle == 0` or `bytes == 0`.
pub fn encode_verified_section_payload(
    attestation: &VerifiedSectionAttestation,
) -> Result<[u8; VERIFIED_SECTION_PAYLOAD_BYTES], BackendError> {
    if attestation.child_handle == 0 || attestation.bytes == 0 {
        return Err(BackendError::Configuration);
    }
    let mut payload = [0_u8; VERIFIED_SECTION_PAYLOAD_BYTES];
    payload[..4].copy_from_slice(&attestation.slot.to_le_bytes());
    payload[4..12].copy_from_slice(&attestation.child_handle.to_le_bytes());
    payload[12..20].copy_from_slice(&attestation.bytes.to_le_bytes());
    payload[20..52].copy_from_slice(&attestation.sha256);
    Ok(payload)
}

fn encode_verified_sections(
    sections: &[VerifiedSectionAttestation],
) -> Result<Vec<[u8; VERIFIED_SECTION_PAYLOAD_BYTES]>, BackendError> {
    if sections.len() > MAX_VERIFIED_SECTIONS {
        return Err(BackendError::Configuration);
    }
    let mut seen_slots = [0_u32; MAX_VERIFIED_SECTIONS];
    let mut encoded = Vec::with_capacity(sections.len());
    for (index, section) in sections.iter().enumerate() {
        if seen_slots[..index].contains(&section.slot) {
            return Err(BackendError::Configuration);
        }
        seen_slots[index] = section.slot;
        encoded.push(encode_verified_section_payload(section)?);
    }
    Ok(encoded)
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum ReadyState {
    Pending,
    Ready,
    Failed,
}

fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

fn wait_unpoisoned<'a, T>(condvar: &Condvar, guard: MutexGuard<'a, T>) -> MutexGuard<'a, T> {
    match condvar.wait(guard) {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

struct ScrubHandshake(Box<[u8; KEY_HEX_BYTES]>);

impl Drop for ScrubHandshake {
    fn drop(&mut self) {
        crate::scrub_bytes(self.0.as_mut_slice());
    }
}

struct OutgoingFrame {
    stream_id: u32,
    kind: u8,
    payload: Vec<u8>,
}

struct TxState {
    queue: VecDeque<OutgoingFrame>,
    queued_data_frames: usize,
    enqueued_seq: u64,
    flushed_seq: u64,
    ack_received: bool,
    closed: bool,
    failed: bool,
    reset_streams: VecDeque<u32>,
    dropped_reset_data: VecDeque<u32>,
    write_waker: Option<Waker>,
    flush_waker: Option<Waker>,
}

impl TxState {
    const fn new() -> Self {
        Self {
            queue: VecDeque::new(),
            queued_data_frames: 0,
            enqueued_seq: 0,
            flushed_seq: 0,
            ack_received: false,
            closed: false,
            failed: false,
            reset_streams: VecDeque::new(),
            dropped_reset_data: VecDeque::new(),
            write_waker: None,
            flush_waker: None,
        }
    }

    fn push_frame(&mut self, stream_id: u32, kind: u8, payload: Vec<u8>) -> u64 {
        self.enqueued_seq = self.enqueued_seq.saturating_add(1);
        if kind == KIND_DATA {
            self.queued_data_frames = self.queued_data_frames.saturating_add(1);
        }
        self.queue.push_back(OutgoingFrame {
            stream_id,
            kind,
            payload,
        });
        self.enqueued_seq
    }

    fn fail_and_drain(&mut self, pool: &BufferPool) {
        self.failed = true;
        for frame in self.queue.drain(..) {
            pool.recycle(frame.payload);
        }
        self.queued_data_frames = 0;
        self.wake_io();
    }

    fn is_reset(&self, stream_id: u32) -> bool {
        self.reset_streams.contains(&stream_id)
    }

    fn has_dropped_data(&self, stream_id: u32) -> bool {
        self.dropped_reset_data.contains(&stream_id)
    }

    fn record_dropped_data(&mut self, stream_id: u32) {
        push_bounded(&mut self.dropped_reset_data, stream_id);
    }

    fn record_reset(&mut self, stream_id: u32, pool: &BufferPool) {
        push_bounded(&mut self.reset_streams, stream_id);
        let mut retained = VecDeque::with_capacity(self.queue.len());
        let mut removed_data = 0_usize;
        for frame in self.queue.drain(..) {
            if frame.stream_id == stream_id && matches!(frame.kind, KIND_DATA | KIND_EOF) {
                if frame.kind == KIND_DATA {
                    removed_data = removed_data.saturating_add(1);
                }
                pool.recycle(frame.payload);
            } else {
                retained.push_back(frame);
            }
        }
        self.queue = retained;
        self.queued_data_frames = self.queued_data_frames.saturating_sub(removed_data);
        if removed_data > 0 {
            self.record_dropped_data(stream_id);
        }
    }

    fn wake_io(&mut self) {
        if let Some(waker) = self.write_waker.take() {
            waker.wake();
        }
        if let Some(waker) = self.flush_waker.take() {
            waker.wake();
        }
    }
}

fn push_bounded(queue: &mut VecDeque<u32>, stream_id: u32) {
    if !queue.contains(&stream_id) {
        if queue.len() >= MAX_TRACKED_RESETS {
            let _ = queue.pop_front();
        }
        queue.push_back(stream_id);
    }
}

struct TxShared {
    state: Mutex<TxState>,
    writer_ready: Condvar,
}

impl TxShared {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(TxState::new()),
            writer_ready: Condvar::new(),
        })
    }
}

struct TxCloser(Arc<TxShared>);

impl Drop for TxCloser {
    fn drop(&mut self) {
        let mut state = lock_unpoisoned(&self.0.state);
        state.closed = true;
        state.wake_io();
        drop(state);
        self.0.writer_ready.notify_one();
    }
}

type ActiveRx = (u32, mpsc::Sender<io::Result<Vec<u8>>>, Arc<AtomicBool>);

struct RxState {
    active: Option<ActiveRx>,
    closed: bool,
    failed: bool,
}

impl RxState {
    const fn new() -> Self {
        Self {
            active: None,
            closed: false,
            failed: false,
        }
    }

    fn take_if_active(&mut self, stream_id: u32) -> Option<ActiveRx> {
        if self
            .active
            .as_ref()
            .is_some_and(|(active_id, _, _)| *active_id == stream_id)
        {
            self.active.take()
        } else {
            None
        }
    }
}

/// Parent-side broker pipe connector, carrying HTTP/1.1 requests over anonymous pipes.
pub struct PipeBackend {
    policy: BackendPolicy,
    tx: Arc<TxShared>,
    rx_state: Arc<Mutex<RxState>>,
    ready_rx: watch::Receiver<ReadyState>,
    next_stream_id: AtomicU32,
    closer: Arc<TxCloser>,
    pool: Arc<BufferPool>,
}

impl PipeBackend {
    /// Connects over the child process's parent-owned `(stdin, stdout)` pipe handles.
    ///
    /// # Errors
    /// Returns [`BackendError::Configuration`] for an invalid 64-byte hex credential,
    /// or [`BackendError::Io`] if background pipe threads cannot start.
    pub fn connect_stdio(
        child_stdin: File,
        child_stdout: File,
        backend_key: &str,
    ) -> Result<Self, BackendError> {
        Self::connect_stdio_with_sections(child_stdin, child_stdout, backend_key, &[])
    }

    /// Connects over the child's `(stdin, stdout)` pipe handles and attests inherited sections.
    ///
    /// Validates `sections` eagerly before starting background I/O threads, completes the
    /// `SB2PIPE1`/`SB2OKAY1` bearer handshake, and emits one `KIND_VERIFIED_SECTION` control
    /// frame (`stream_id == 0`, 52-byte payload) per attested section before `KIND_SERVER_READY`.
    ///
    /// # Errors
    /// Returns [`BackendError::Configuration`] for an invalid 64-byte hex credential, more than
    /// [`MAX_VERIFIED_SECTIONS`] attestations, duplicate `slot`, or zero `child_handle`/`bytes`;
    /// returns [`BackendError::Io`] if background pipe threads cannot start.
    pub fn connect_stdio_with_sections(
        child_stdin: File,
        child_stdout: File,
        backend_key: &str,
        sections: &[VerifiedSectionAttestation],
    ) -> Result<Self, BackendError> {
        Self::from_io_with_sections(child_stdout, child_stdin, backend_key, sections)
    }

    /// Connects over arbitrary blocking reader/writer streams using the `SB2PIPE1` protocol.
    ///
    /// # Errors
    /// Returns [`BackendError::Configuration`] for an invalid 64-byte hex credential,
    /// or [`BackendError::Io`] if background pipe threads cannot start.
    pub fn from_io<R, W>(reader: R, writer: W, backend_key: &str) -> Result<Self, BackendError>
    where
        R: Read + Send + 'static,
        W: Write + Send + 'static,
    {
        Self::from_io_with_sections(reader, writer, backend_key, &[])
    }

    /// Connects over blocking reader/writer streams and attests inherited read-only sections.
    ///
    /// Validates `sections` synchronously before spawning background threads, then transmits
    /// one `KIND_VERIFIED_SECTION` control frame (`stream_id == 0`, 52-byte payload) per entry
    /// immediately after `SB2OKAY1` is observed and before `KIND_SERVER_READY` completes.
    ///
    /// # Errors
    /// Returns [`BackendError::Configuration`] for an invalid 64-byte hex credential, more than
    /// [`MAX_VERIFIED_SECTIONS`] attestations, duplicate `slot`, or zero `child_handle`/`bytes`;
    /// returns [`BackendError::Io`] if background pipe threads cannot start.
    pub fn from_io_with_sections<R, W>(
        reader: R,
        writer: W,
        backend_key: &str,
        sections: &[VerifiedSectionAttestation],
    ) -> Result<Self, BackendError>
    where
        R: Read + Send + 'static,
        W: Write + Send + 'static,
    {
        let policy = BackendPolicy::new(BROKER_PIPE_HOST_PORT, backend_key)?;
        let raw_key = backend_key.as_bytes();
        if raw_key.len() != KEY_HEX_BYTES {
            return Err(BackendError::Configuration);
        }
        let pending_sections = encode_verified_sections(sections)?;
        let mut boxed_key = Box::new([0_u8; KEY_HEX_BYTES]);
        boxed_key.copy_from_slice(raw_key);
        let handshake_key = ScrubHandshake(boxed_key);

        let tx = TxShared::new();
        let closer = Arc::new(TxCloser(Arc::clone(&tx)));
        let rx_state = Arc::new(Mutex::new(RxState::new()));
        let (ready_tx, ready_rx) = watch::channel(ReadyState::Pending);
        let pool = BufferPool::new();

        let tx_worker = Arc::clone(&tx);
        let rx_fail = Arc::clone(&rx_state);
        let ready_tx_fail = ready_tx.clone();
        let tx_pool = Arc::clone(&pool);
        std::thread::Builder::new()
            .name("broker-pipe-client-tx".to_owned())
            .spawn(move || {
                run_writer(
                    writer,
                    handshake_key,
                    pending_sections,
                    &tx_worker,
                    &rx_fail,
                    &ready_tx_fail,
                    &tx_pool,
                );
            })
            .map_err(|_| BackendError::Io)?;

        let rx_tx = Arc::clone(&tx);
        let rx_worker = Arc::clone(&rx_state);
        let rx_pool = Arc::clone(&pool);
        std::thread::Builder::new()
            .name("broker-pipe-client-rx".to_owned())
            .spawn(move || run_reader(reader, &rx_tx, &rx_worker, &ready_tx, &rx_pool))
            .map_err(|_| BackendError::Io)?;

        Ok(Self {
            policy,
            tx,
            rx_state,
            ready_rx,
            next_stream_id: AtomicU32::new(1),
            closer,
            pool,
        })
    }

    /// Waits for handshake acknowledgement (`SB2OKAY1`) and `KIND_SERVER_READY` before `deadline`.
    ///
    /// # Errors
    /// Returns [`BackendError::Deadline`] on timeout or [`BackendError::Io`] if the pipe closes
    /// or fails before readiness arrives.
    pub async fn wait_server_ready(&self, deadline: Instant) -> Result<(), BackendError> {
        let mut rx = self.ready_rx.clone();
        let wait = async {
            loop {
                match *rx.borrow_and_update() {
                    ReadyState::Ready => return Ok(()),
                    ReadyState::Failed => return Err(BackendError::Io),
                    ReadyState::Pending => {}
                }
                if rx.changed().await.is_err() {
                    return Err(BackendError::Io);
                }
            }
        };
        timeout_at(deadline, wait)
            .await
            .map_err(|_| BackendError::Deadline)?
    }
}

impl BackendConnector for PipeBackend {
    type Stream = PipeStream;

    fn policy(&self) -> &BackendPolicy {
        &self.policy
    }

    async fn connect(&self, _deadline: Instant) -> Result<Self::Stream, BackendError> {
        if *self.ready_rx.borrow() != ReadyState::Ready {
            return Err(BackendError::Io);
        }
        let stream_id = self
            .next_stream_id
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(1)
            })
            .map_err(|_| BackendError::Io)?;
        let (chunk_tx, chunk_rx) = mpsc::channel(CHANNEL_CAPACITY);
        let eof_received = Arc::new(AtomicBool::new(false));
        {
            let mut rx = lock_unpoisoned(&self.rx_state);
            if rx.closed || rx.failed || rx.active.is_some() {
                return Err(BackendError::Io);
            }
            rx.active = Some((stream_id, chunk_tx, Arc::clone(&eof_received)));
        }
        {
            let mut state = lock_unpoisoned(&self.tx.state);
            if state.closed || state.failed {
                drop(state);
                let _ = lock_unpoisoned(&self.rx_state).take_if_active(stream_id);
                return Err(BackendError::Io);
            }
            state.push_frame(stream_id, KIND_OPEN, Vec::new());
            drop(state);
            self.tx.writer_ready.notify_one();
        }
        Ok(PipeStream {
            stream_id,
            rx: chunk_rx,
            buffered: Vec::new(),
            offset: 0,
            eof_received,
            rx_state: Arc::clone(&self.rx_state),
            tx: Arc::clone(&self.tx),
            _closer: Arc::clone(&self.closer),
            pool: Arc::clone(&self.pool),
            eof_sent: false,
            pending_data_seq: 0,
            pending_eof_seq: 0,
        })
    }
}

/// One sequential HTTP/1.1 logical stream over the parent's broker pipe.
pub struct PipeStream {
    stream_id: u32,
    rx: mpsc::Receiver<io::Result<Vec<u8>>>,
    buffered: Vec<u8>,
    offset: usize,
    eof_received: Arc<AtomicBool>,
    rx_state: Arc<Mutex<RxState>>,
    tx: Arc<TxShared>,
    _closer: Arc<TxCloser>,
    pool: Arc<BufferPool>,
    eof_sent: bool,
    pending_data_seq: u64,
    pending_eof_seq: u64,
}

impl Drop for PipeStream {
    fn drop(&mut self) {
        let _ = lock_unpoisoned(&self.rx_state).take_if_active(self.stream_id);
        let leftover = std::mem::take(&mut self.buffered);
        self.pool.recycle(leftover);
        self.rx.close();
        while let Ok(item) = self.rx.try_recv() {
            if let Ok(chunk) = item {
                self.pool.recycle(chunk);
            }
        }
        if !self.eof_sent {
            let mut state = lock_unpoisoned(&self.tx.state);
            if !state.closed && !state.failed && !state.is_reset(self.stream_id) {
                let kind = if self.eof_received.load(Ordering::Acquire) {
                    KIND_EOF
                } else {
                    state.record_reset(self.stream_id, &self.pool);
                    KIND_RESET
                };
                state.push_frame(self.stream_id, kind, Vec::new());
                drop(state);
                self.tx.writer_ready.notify_one();
            }
        }
    }
}

impl AsyncRead for PipeStream {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        if buf.remaining() == 0 {
            return Poll::Ready(Ok(()));
        }
        while self.offset >= self.buffered.len() {
            if self.buffered.capacity() >= MAX_FRAME_PAYLOAD {
                let finished = std::mem::take(&mut self.buffered);
                self.pool.recycle(finished);
            } else {
                self.buffered.clear();
            }
            self.offset = 0;
            match self.rx.poll_recv(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(None) => return Poll::Ready(Ok(())),
                Poll::Ready(Some(Err(error))) => return Poll::Ready(Err(error)),
                Poll::Ready(Some(Ok(chunk))) => {
                    if chunk.is_empty() {
                        self.pool.recycle(chunk);
                    } else {
                        self.buffered = chunk;
                    }
                }
            }
        }
        let available = &self.buffered[self.offset..];
        let count = available.len().min(buf.remaining());
        buf.put_slice(&available[..count]);
        self.offset += count;
        if self.offset >= self.buffered.len() {
            let finished = std::mem::take(&mut self.buffered);
            self.pool.recycle(finished);
            self.offset = 0;
        }
        Poll::Ready(Ok(()))
    }
}

impl AsyncWrite for PipeStream {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        if buf.is_empty() {
            return Poll::Ready(Ok(0));
        }
        if self.eof_sent {
            return Poll::Ready(Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "write after broker-pipe shutdown",
            )));
        }
        let (count, seq) = {
            let mut state = lock_unpoisoned(&self.tx.state);
            if state.closed || state.failed || state.is_reset(self.stream_id) {
                return Poll::Ready(Err(io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "broker-pipe stream closed",
                )));
            }
            if state.queued_data_frames >= CHANNEL_CAPACITY {
                state.write_waker = Some(cx.waker().clone());
                return Poll::Pending;
            }
            let count = buf.len().min(MAX_FRAME_PAYLOAD);
            let payload = self.pool.take_from_slice(&buf[..count]);
            let seq = state.push_frame(self.stream_id, KIND_DATA, payload);
            drop(state);
            self.tx.writer_ready.notify_one();
            (count, seq)
        };
        self.pending_data_seq = seq;
        Poll::Ready(Ok(count))
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        let mut state = lock_unpoisoned(&self.tx.state);
        if state.failed || state.has_dropped_data(self.stream_id) {
            return Poll::Ready(Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "broker-pipe flush failed",
            )));
        }
        let target_seq = if state.is_reset(self.stream_id) {
            self.pending_data_seq
        } else {
            self.pending_data_seq.max(self.pending_eof_seq)
        };
        if state.flushed_seq >= target_seq {
            return Poll::Ready(Ok(()));
        }
        if state.closed {
            return Poll::Ready(Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "broker-pipe closed before flush",
            )));
        }
        state.flush_waker = Some(cx.waker().clone());
        Poll::Pending
    }

    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        if !self.eof_sent {
            let mut state = lock_unpoisoned(&self.tx.state);
            if state.failed || state.closed || state.has_dropped_data(self.stream_id) {
                drop(state);
                self.eof_sent = true;
                return Poll::Ready(Err(io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "broker-pipe shutdown failed",
                )));
            }
            if state.is_reset(self.stream_id) {
                drop(state);
                self.eof_sent = true;
                return self.poll_flush(cx);
            }
            let seq = state.push_frame(self.stream_id, KIND_EOF, Vec::new());
            drop(state);
            self.pending_eof_seq = seq;
            self.eof_sent = true;
            self.tx.writer_ready.notify_one();
        }
        self.poll_flush(cx)
    }
}

fn encode_header(stream_id: u32, kind: u8, payload_len: u32) -> [u8; FRAME_HEADER_BYTES] {
    let mut header = [0_u8; FRAME_HEADER_BYTES];
    header[..4].copy_from_slice(&stream_id.to_be_bytes());
    header[4] = kind;
    header[5..9].copy_from_slice(&payload_len.to_be_bytes());
    header
}

fn decode_header(header: [u8; FRAME_HEADER_BYTES]) -> (u32, u8, usize) {
    let stream_id = u32::from_be_bytes([header[0], header[1], header[2], header[3]]);
    let kind = header[4];
    let len = u32::from_be_bytes([header[5], header[6], header[7], header[8]]);
    let len_usize = usize::try_from(len).unwrap_or(usize::MAX);
    (stream_id, kind, len_usize)
}

enum HeaderRead {
    Frame([u8; FRAME_HEADER_BYTES]),
    CleanEof,
}

fn read_header<R: Read>(reader: &mut R) -> io::Result<HeaderRead> {
    let mut header = [0_u8; FRAME_HEADER_BYTES];
    let mut read = 0_usize;
    while read < FRAME_HEADER_BYTES {
        match reader.read(&mut header[read..]) {
            Ok(0) if read == 0 => return Ok(HeaderRead::CleanEof),
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "truncated broker-pipe frame header",
                ));
            }
            Ok(count) => read += count,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
    Ok(HeaderRead::Frame(header))
}

fn write_raw_frame(
    writer: &mut impl Write,
    stream_id: u32,
    kind: u8,
    payload: &[u8],
) -> io::Result<()> {
    let payload_len = u32::try_from(payload.len()).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "broker-pipe frame exceeds u32 bound",
        )
    })?;
    writer
        .write_all(&encode_header(stream_id, kind, payload_len))
        .and_then(|()| writer.write_all(payload))
}

#[cfg(test)]
pub(crate) fn write_test_frame(
    writer: &mut impl Write,
    stream_id: u32,
    kind: u8,
    payload: &[u8],
) -> io::Result<()> {
    write_raw_frame(writer, stream_id, kind, payload)?;
    writer.flush()
}

#[cfg(test)]
pub(crate) fn read_test_frame_opt(
    reader: &mut impl Read,
) -> io::Result<Option<(u32, u8, Vec<u8>)>> {
    let HeaderRead::Frame(header) = read_header(reader)? else {
        return Ok(None);
    };
    let (stream_id, kind, len) = decode_header(header);
    let mut payload = vec![0_u8; len];
    if len > 0 {
        reader.read_exact(&mut payload)?;
    }
    Ok(Some((stream_id, kind, payload)))
}

fn fail_transport(
    tx: &Arc<TxShared>,
    rx_state: &Arc<Mutex<RxState>>,
    ready_tx: &watch::Sender<ReadyState>,
    pool: &BufferPool,
) {
    {
        let mut rx = lock_unpoisoned(rx_state);
        rx.failed = true;
        if let Some((_, sender, _)) = rx.active.take() {
            let _ = sender.try_send(Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "broker-pipe failed",
            )));
        }
    }
    let mut state = lock_unpoisoned(&tx.state);
    state.fail_and_drain(pool);
    drop(state);
    tx.writer_ready.notify_one();
    let _ = ready_tx.send(ReadyState::Failed);
}

fn handle_reader_frame<R: Read>(
    reader: &mut R,
    tx: &Arc<TxShared>,
    rx_state: &Arc<Mutex<RxState>>,
    ready_tx: &watch::Sender<ReadyState>,
    pool: &BufferPool,
    header: [u8; FRAME_HEADER_BYTES],
) -> io::Result<()> {
    let (stream_id, kind, len) = decode_header(header);
    match kind {
        KIND_SERVER_READY => {
            if stream_id != 0 || len != 0 || *ready_tx.borrow() != ReadyState::Pending {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "unexpected broker-pipe server ready frame",
                ));
            }
            let _ = ready_tx.send(ReadyState::Ready);
        }
        KIND_DATA => {
            if stream_id == 0 || !(1..=MAX_FRAME_PAYLOAD).contains(&len) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "invalid data frame",
                ));
            }
            let mut payload = pool.take_zeroed(len);
            if let Err(error) = reader.read_exact(&mut payload) {
                pool.recycle(payload);
                return Err(error);
            }
            let sender = {
                let rx = lock_unpoisoned(rx_state);
                rx.active
                    .as_ref()
                    .filter(|(active_id, _, _)| *active_id == stream_id)
                    .map(|(_, sender, _)| sender.clone())
            };
            if let Some(sender) = sender {
                if let Err(mpsc::error::SendError(Ok(returned))) = sender.blocking_send(Ok(payload))
                {
                    pool.recycle(returned);
                }
            } else {
                pool.recycle(payload);
            }
        }
        KIND_EOF => {
            if stream_id == 0 || len != 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "invalid eof frame",
                ));
            }
            if let Some((_, sender, eof_received)) =
                lock_unpoisoned(rx_state).take_if_active(stream_id)
            {
                eof_received.store(true, Ordering::Release);
                drop(sender);
            }
        }
        KIND_RESET => {
            if stream_id == 0 || len != 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "invalid reset frame",
                ));
            }
            {
                let mut state = lock_unpoisoned(&tx.state);
                state.record_reset(stream_id, pool);
                state.wake_io();
            }
            if let Some((_, sender, _)) = lock_unpoisoned(rx_state).take_if_active(stream_id) {
                let _ = sender.try_send(Err(io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "backend reset broker-pipe stream",
                )));
            }
        }
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid broker-pipe frame kind",
            ));
        }
    }
    Ok(())
}

fn run_reader<R: Read>(
    mut reader: R,
    tx: &Arc<TxShared>,
    rx_state: &Arc<Mutex<RxState>>,
    ready_tx: &watch::Sender<ReadyState>,
    pool: &BufferPool,
) {
    let mut ack = [0_u8; HANDSHAKE_RESPONSE_MAGIC.len()];
    if reader.read_exact(&mut ack).is_err() || &ack != HANDSHAKE_RESPONSE_MAGIC {
        fail_transport(tx, rx_state, ready_tx, pool);
        return;
    }
    {
        let mut state = lock_unpoisoned(&tx.state);
        state.ack_received = true;
    }
    tx.writer_ready.notify_one();
    loop {
        let header = match read_header(&mut reader) {
            Ok(HeaderRead::CleanEof) => {
                let mut rx = lock_unpoisoned(rx_state);
                rx.closed = true;
                rx.active = None;
                drop(rx);
                if *ready_tx.borrow() == ReadyState::Pending {
                    let _ = ready_tx.send(ReadyState::Failed);
                }
                lock_unpoisoned(&tx.state).fail_and_drain(pool);
                break;
            }
            Ok(HeaderRead::Frame(header)) => header,
            Err(_) => {
                fail_transport(tx, rx_state, ready_tx, pool);
                break;
            }
        };
        if handle_reader_frame(&mut reader, tx, rx_state, ready_tx, pool, header).is_err() {
            fail_transport(tx, rx_state, ready_tx, pool);
            break;
        }
    }
}

fn run_writer<W: Write>(
    mut writer: W,
    handshake_key: ScrubHandshake,
    pending_sections: Vec<[u8; VERIFIED_SECTION_PAYLOAD_BYTES]>,
    tx: &Arc<TxShared>,
    rx_state: &Arc<Mutex<RxState>>,
    ready_tx: &watch::Sender<ReadyState>,
    pool: &BufferPool,
) {
    let handshake_result = writer
        .write_all(HANDSHAKE_REQUEST_MAGIC)
        .and_then(|()| writer.write_all(handshake_key.0.as_slice()))
        .and_then(|()| writer.flush());
    drop(handshake_key);
    if handshake_result.is_err() {
        fail_transport(tx, rx_state, ready_tx, pool);
        return;
    }

    if !pending_sections.is_empty() {
        let should_emit = {
            let mut state = lock_unpoisoned(&tx.state);
            while !state.ack_received && !state.failed && !state.closed {
                state = wait_unpoisoned(&tx.writer_ready, state);
            }
            state.ack_received && !state.failed && !state.closed
        };
        if !should_emit {
            return;
        }
        let mut emit_result = Ok(());
        for payload in pending_sections {
            if let Err(error) = write_raw_frame(&mut writer, 0, KIND_VERIFIED_SECTION, &payload) {
                emit_result = Err(error);
                break;
            }
        }
        if emit_result.is_ok() {
            emit_result = writer.flush();
        }
        if emit_result.is_err() {
            fail_transport(tx, rx_state, ready_tx, pool);
            return;
        }
    }

    let mut batch = Vec::new();
    loop {
        let batch_seq = {
            let mut state = lock_unpoisoned(&tx.state);
            while state.queue.is_empty() && !state.closed && !state.failed {
                state = wait_unpoisoned(&tx.writer_ready, state);
            }
            if state.failed || (state.queue.is_empty() && state.closed) {
                break;
            }
            batch.clear();
            let mut flushed_data = 0_usize;
            while let Some(frame) = state.queue.pop_front() {
                if frame.kind == KIND_DATA {
                    flushed_data = flushed_data.saturating_add(1);
                }
                if matches!(frame.kind, KIND_DATA | KIND_EOF) && state.is_reset(frame.stream_id) {
                    if frame.kind == KIND_DATA {
                        state.record_dropped_data(frame.stream_id);
                    }
                    pool.recycle(frame.payload);
                    continue;
                }
                batch.push(frame);
            }
            state.queued_data_frames = state.queued_data_frames.saturating_sub(flushed_data);
            let waker = if state.queued_data_frames < CHANNEL_CAPACITY {
                state.write_waker.take()
            } else {
                None
            };
            let seq = state.enqueued_seq;
            drop(state);
            if let Some(waker) = waker {
                waker.wake();
            }
            seq
        };

        let mut io_result = Ok(());
        for frame in batch.drain(..) {
            let write_result =
                write_raw_frame(&mut writer, frame.stream_id, frame.kind, &frame.payload);
            pool.recycle(frame.payload);
            if let Err(error) = write_result {
                io_result = Err(error);
                break;
            }
        }
        for frame in batch.drain(..) {
            pool.recycle(frame.payload);
        }
        if io_result.is_ok() {
            io_result = writer.flush();
        }

        if io_result.is_err() {
            fail_transport(tx, rx_state, ready_tx, pool);
            break;
        }

        let mut state = lock_unpoisoned(&tx.state);
        state.flushed_seq = state.flushed_seq.max(batch_seq);
        if let Some(waker) = state.flush_waker.take() {
            waker.wake();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    const VALID_KEY: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    struct PipeQueue {
        bytes: VecDeque<u8>,
        closed: bool,
    }

    struct MemoryStream {
        shared: Arc<(Mutex<PipeQueue>, Condvar)>,
    }

    impl MemoryStream {
        fn buffered_bytes(&self) -> usize {
            lock_unpoisoned(&self.shared.0).bytes.len()
        }
    }

    fn memory_channel() -> (MemoryStream, MemoryStream) {
        let shared = Arc::new((
            Mutex::new(PipeQueue {
                bytes: VecDeque::new(),
                closed: false,
            }),
            Condvar::new(),
        ));
        (
            MemoryStream {
                shared: Arc::clone(&shared),
            },
            MemoryStream { shared },
        )
    }

    impl Drop for MemoryStream {
        fn drop(&mut self) {
            let mut guard = lock_unpoisoned(&self.shared.0);
            guard.closed = true;
            drop(guard);
            self.shared.1.notify_all();
        }
    }

    impl Read for MemoryStream {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            let mut guard = lock_unpoisoned(&self.shared.0);
            while guard.bytes.is_empty() && !guard.closed {
                guard = wait_unpoisoned(&self.shared.1, guard);
            }
            let count = guard.bytes.len().min(buf.len());
            for slot in &mut buf[..count] {
                *slot = guard.bytes.pop_front().unwrap_or(0);
            }
            Ok(count)
        }
    }

    impl Write for MemoryStream {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            let mut guard = lock_unpoisoned(&self.shared.0);
            guard.bytes.extend(buf.iter().copied());
            drop(guard);
            self.shared.1.notify_all();
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    fn write_frame(
        writer: &mut impl Write,
        stream_id: u32,
        kind: u8,
        payload: &[u8],
    ) -> io::Result<()> {
        write_test_frame(writer, stream_id, kind, payload)
    }

    fn read_frame(reader: &mut impl Read) -> io::Result<(u32, u8, Vec<u8>)> {
        read_test_frame_opt(reader)?
            .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "eof"))
    }

    fn run_async(future: impl Future<Output = io::Result<()>>) -> io::Result<()> {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?
            .block_on(future)
    }

    fn complete_handshake(p_to_c_r: &mut impl Read, c_to_p_w: &mut impl Write) -> io::Result<()> {
        let mut handshake = [0_u8; 8 + KEY_HEX_BYTES];
        p_to_c_r.read_exact(&mut handshake)?;
        assert_eq!(&handshake[..8], HANDSHAKE_REQUEST_MAGIC);
        assert_eq!(&handshake[8..], VALID_KEY.as_bytes());
        c_to_p_w.write_all(HANDSHAKE_RESPONSE_MAGIC)?;
        c_to_p_w.flush()
    }

    async fn ready_memory_backend() -> io::Result<(PipeBackend, MemoryStream, MemoryStream)> {
        let (p_to_c_w, mut p_to_c_r) = memory_channel();
        let (mut c_to_p_w, c_to_p_r) = memory_channel();
        let backend = PipeBackend::from_io(c_to_p_r, p_to_c_w, VALID_KEY)
            .map_err(|_| io::Error::other("PipeBackend::from_io failed"))?;
        complete_handshake(&mut p_to_c_r, &mut c_to_p_w)?;
        write_frame(&mut c_to_p_w, 0, KIND_SERVER_READY, &[])?;
        backend
            .wait_server_ready(Instant::now() + Duration::from_secs(2))
            .await
            .map_err(|_| io::Error::other("wait_server_ready failed"))?;
        Ok((backend, p_to_c_r, c_to_p_w))
    }

    #[test]
    fn handshake_readiness_large_exchanges_and_drop_reset_work_end_to_end() -> io::Result<()> {
        run_async(async {
            let (p_to_c_w, mut p_to_c_r) = memory_channel();
            let (mut c_to_p_w, c_to_p_r) = memory_channel();

            let backend = PipeBackend::from_io(c_to_p_r, p_to_c_w, VALID_KEY)
                .map_err(|_| io::Error::other("PipeBackend::from_io failed"))?;
            complete_handshake(&mut p_to_c_r, &mut c_to_p_w)?;

            // Calling wait_server_ready concurrently before KIND_SERVER_READY arrives must wait and succeed.
            let server_ready_sender = std::thread::spawn(move || -> io::Result<MemoryStream> {
                std::thread::sleep(Duration::from_millis(20));
                write_frame(&mut c_to_p_w, 0, KIND_SERVER_READY, &[])?;
                Ok(c_to_p_w)
            });
            backend
                .wait_server_ready(Instant::now() + Duration::from_secs(2))
                .await
                .map_err(|_| io::Error::other("wait_server_ready failed"))?;
            let mut c_to_p_w = server_ready_sender
                .join()
                .map_err(|_| io::Error::other("server ready thread panicked"))??;

            // Calling wait_server_ready again after readiness already arrived returns immediately.
            backend
                .wait_server_ready(Instant::now() + Duration::from_secs(1))
                .await
                .map_err(|_| io::Error::other("second wait_server_ready failed"))?;

            // Exchange 1: >16 KiB write + server EOF + drop emits KIND_EOF.
            let payload: Vec<u8> = (0..=255_u8).cycle().take(MAX_FRAME_PAYLOAD + 512).collect();
            let mut s1 = backend
                .connect(Instant::now() + Duration::from_secs(1))
                .await
                .map_err(|_| io::Error::other("connect 1 failed"))?;
            s1.write_all(&payload).await?;
            s1.flush().await?;
            assert_eq!(read_frame(&mut p_to_c_r)?, (1, KIND_OPEN, Vec::new()));
            let (id_a, kind_a, data_a) = read_frame(&mut p_to_c_r)?;
            let (id_b, kind_b, data_b) = read_frame(&mut p_to_c_r)?;
            assert_eq!(
                (id_a, kind_a, data_a.len()),
                (1, KIND_DATA, MAX_FRAME_PAYLOAD)
            );
            assert_eq!((id_b, kind_b, data_b.len()), (1, KIND_DATA, 512));
            assert_eq!([data_a, data_b].concat(), payload);

            write_frame(&mut c_to_p_w, 1, KIND_DATA, b"HTTP/1.1 200 OK\r\n\r\n")?;
            write_frame(&mut c_to_p_w, 1, KIND_EOF, &[])?;
            let mut reply = Vec::new();
            s1.read_to_end(&mut reply).await?;
            assert_eq!(reply, b"HTTP/1.1 200 OK\r\n\r\n");
            drop(s1);
            assert_eq!(read_frame(&mut p_to_c_r)?, (1, KIND_EOF, Vec::new()));

            // Exchange 2: early drop without server EOF sends KIND_RESET and safely ignores trailing server frames.
            let mut s2 = backend
                .connect(Instant::now() + Duration::from_secs(1))
                .await
                .map_err(|_| io::Error::other("connect 2 failed"))?;
            s2.write_all(b"POST /v1/chat/completions").await?;
            s2.flush().await?;
            assert_eq!(read_frame(&mut p_to_c_r)?, (2, KIND_OPEN, Vec::new()));
            assert_eq!(
                read_frame(&mut p_to_c_r)?,
                (2, KIND_DATA, b"POST /v1/chat/completions".to_vec())
            );
            drop(s2);
            assert_eq!(read_frame(&mut p_to_c_r)?, (2, KIND_RESET, Vec::new()));
            // Trailing server frame on dropped stream 2 must not break subsequent stream 3.
            write_frame(&mut c_to_p_w, 2, KIND_EOF, &[])?;

            let mut s3 = backend
                .connect(Instant::now() + Duration::from_secs(1))
                .await
                .map_err(|_| io::Error::other("connect 3 failed"))?;
            assert_eq!(read_frame(&mut p_to_c_r)?, (3, KIND_OPEN, Vec::new()));
            write_frame(&mut c_to_p_w, 3, KIND_DATA, b"ok")?;
            write_frame(&mut c_to_p_w, 3, KIND_EOF, &[])?;
            let mut s3_reply = Vec::new();
            s3.read_to_end(&mut s3_reply).await?;
            assert_eq!(s3_reply, b"ok");
            Ok(())
        })
    }

    #[test]
    fn buffer_pool_recycles_16kib_buffers_respects_bound_and_scrubs_bytes() {
        let pool = BufferPool::new();
        assert_eq!(pool.len(), 0);

        let first = pool.take_from_slice(b"secret-payload");
        assert!(first.capacity() >= MAX_FRAME_PAYLOAD);
        let first_ptr = first.as_ptr();
        pool.recycle(first);
        assert_eq!(pool.len(), 1);

        // Reusing a recycled buffer preserves its allocation, starts zeroed for RX,
        // and carries no residual payload bytes past the requested read slice.
        let second = pool.take_zeroed(16);
        assert_eq!(pool.len(), 0);
        assert_eq!(second.as_ptr(), first_ptr);
        assert!(second.iter().all(|byte| *byte == 0));
        pool.recycle(second);

        // Undersized and oversized buffers are never retained in the free-list.
        pool.recycle(Vec::with_capacity(64));
        pool.recycle(Vec::with_capacity(MAX_FRAME_PAYLOAD * 3));
        assert_eq!(pool.len(), 1);

        // The free-list enforces FREE_LIST_CAPACITY.
        let mut held = Vec::with_capacity(FREE_LIST_CAPACITY + 4);
        for _ in 0..(FREE_LIST_CAPACITY + 4) {
            held.push(pool.take_zeroed(MAX_FRAME_PAYLOAD));
        }
        for buf in held {
            pool.recycle(buf);
        }
        assert_eq!(pool.len(), FREE_LIST_CAPACITY);
    }

    #[test]
    fn pipe_transport_recycles_buffers_across_frames_and_resets() -> io::Result<()> {
        run_async(async {
            let (backend, mut p_to_c_r, mut c_to_p_w) = ready_memory_backend().await?;

            // Stream 1 writes and reads multiple frames, then consumes until EOF.
            let mut s1 = backend
                .connect(Instant::now() + Duration::from_secs(1))
                .await
                .map_err(|_| io::Error::other("connect 1 failed"))?;
            let outbound = vec![0xab_u8; MAX_FRAME_PAYLOAD + 128];
            s1.write_all(&outbound).await?;
            s1.flush().await?;
            assert_eq!(read_frame(&mut p_to_c_r)?, (1, KIND_OPEN, Vec::new()));
            let (_, _, f1) = read_frame(&mut p_to_c_r)?;
            let (_, _, f2) = read_frame(&mut p_to_c_r)?;
            assert_eq!([f1, f2].concat(), outbound);

            let inbound = vec![0xcd_u8; MAX_FRAME_PAYLOAD];
            write_frame(&mut c_to_p_w, 1, KIND_DATA, &inbound)?;
            write_frame(&mut c_to_p_w, 1, KIND_DATA, b"tail")?;
            write_frame(&mut c_to_p_w, 1, KIND_EOF, &[])?;
            let mut reply = Vec::new();
            s1.read_to_end(&mut reply).await?;
            assert_eq!(&reply[..MAX_FRAME_PAYLOAD], inbound.as_slice());
            assert_eq!(&reply[MAX_FRAME_PAYLOAD..], b"tail");
            drop(s1);
            assert_eq!(read_frame(&mut p_to_c_r)?, (1, KIND_EOF, Vec::new()));
            assert!(backend.pool.len() >= 1);

            // Stream 2 writes a queued data frame then resets early; unread RX frame is also recycled.
            let mut s2 = backend
                .connect(Instant::now() + Duration::from_secs(1))
                .await
                .map_err(|_| io::Error::other("connect 2 failed"))?;
            s2.write_all(b"aborted").await?;
            write_frame(&mut c_to_p_w, 2, KIND_DATA, b"unread-rx")?;
            // Let reader enqueue the RX chunk before dropping s2.
            tokio::time::sleep(Duration::from_millis(20)).await;
            drop(s2);
            assert!(backend.pool.len() >= 1);
            assert!(backend.pool.len() <= FREE_LIST_CAPACITY);
            Ok(())
        })
    }

    fn sample_attestation(slot: u32, child_handle: u64, bytes: u64) -> VerifiedSectionAttestation {
        let mut sha256 = [0_u8; 32];
        for (index, byte) in sha256.iter_mut().enumerate() {
            *byte = u8::try_from(index).unwrap_or(0) ^ u8::try_from(slot & 0xff).unwrap_or(0);
        }
        VerifiedSectionAttestation {
            slot,
            child_handle,
            bytes,
            sha256,
        }
    }

    #[test]
    fn verified_section_payload_encodes_52_le_bytes_and_rejects_invalid_bounds()
    -> Result<(), BackendError> {
        let attestation = sample_attestation(3, 0x0123_4567_89ab_cdef, 0x1020_3040_5060_7080);
        let payload = encode_verified_section_payload(&attestation)?;
        assert_eq!(payload.len(), 52);
        assert_eq!(&payload[0..4], &3_u32.to_le_bytes());
        assert_eq!(&payload[4..12], &0x0123_4567_89ab_cdef_u64.to_le_bytes());
        assert_eq!(&payload[12..20], &0x1020_3040_5060_7080_u64.to_le_bytes());
        assert_eq!(&payload[20..52], &attestation.sha256);

        assert_eq!(
            encode_verified_section_payload(&VerifiedSectionAttestation {
                child_handle: 0,
                ..attestation
            }),
            Err(BackendError::Configuration)
        );
        assert_eq!(
            encode_verified_section_payload(&VerifiedSectionAttestation {
                bytes: 0,
                ..attestation
            }),
            Err(BackendError::Configuration)
        );

        let exact_max: Vec<VerifiedSectionAttestation> = (0..8_u32)
            .map(|slot| sample_attestation(slot, u64::from(slot) + 100, u64::from(slot) + 4096))
            .collect();
        assert_eq!(
            encode_verified_sections(&exact_max).map(|encoded| encoded.len()),
            Ok(MAX_VERIFIED_SECTIONS)
        );

        let mut too_many = exact_max;
        too_many.push(sample_attestation(8, 200, 4096));
        assert_eq!(
            encode_verified_sections(&too_many),
            Err(BackendError::Configuration)
        );

        assert_eq!(
            encode_verified_sections(&[
                sample_attestation(2, 100, 512),
                sample_attestation(2, 104, 1024),
            ]),
            Err(BackendError::Configuration)
        );
        Ok(())
    }

    #[test]
    fn from_io_with_sections_rejects_invalid_attestations_before_spawning_threads() {
        let invalid_cases = [
            (0..=u32::try_from(MAX_VERIFIED_SECTIONS).unwrap_or(8))
                .map(|slot| sample_attestation(slot, u64::from(slot) + 10, 64))
                .collect::<Vec<_>>(),
            vec![
                sample_attestation(1, 0x10, 64),
                sample_attestation(1, 0x20, 128),
            ],
            vec![sample_attestation(0, 0, 64)],
            vec![sample_attestation(0, 0x10, 0)],
        ];
        for sections in invalid_cases {
            let (p_to_c_w, mut p_to_c_r) = memory_channel();
            let (_c_to_p_w, c_to_p_r) = memory_channel();
            assert!(matches!(
                PipeBackend::from_io_with_sections(c_to_p_r, p_to_c_w, VALID_KEY, &sections),
                Err(BackendError::Configuration)
            ));
            let mut drained = Vec::new();
            assert!(p_to_c_r.read_to_end(&mut drained).is_ok());
            assert!(drained.is_empty());
        }
    }

    fn assert_verified_section(
        reader: &mut impl Read,
        expected: &VerifiedSectionAttestation,
    ) -> io::Result<()> {
        let (stream_id, kind, payload) = read_frame(reader)?;
        assert_eq!(stream_id, 0);
        assert_eq!(kind, KIND_VERIFIED_SECTION);
        let expected_payload = encode_verified_section_payload(expected)
            .map_err(|_| io::Error::other("encode_verified_section_payload failed"))?;
        assert_eq!(payload, expected_payload.as_slice());
        Ok(())
    }

    #[test]
    fn from_io_with_sections_emits_frames_after_sb2okay1_and_before_server_ready() -> io::Result<()>
    {
        run_async(async {
            let sections = [
                sample_attestation(0, 0x110, 8192),
                sample_attestation(3, 0x220, 16384),
            ];
            let (p_to_c_w, mut p_to_c_r) = memory_channel();
            let (mut c_to_p_w, c_to_p_r) = memory_channel();

            let backend =
                PipeBackend::from_io_with_sections(c_to_p_r, p_to_c_w, VALID_KEY, &sections)
                    .map_err(|_| io::Error::other("from_io_with_sections failed"))?;

            let mut handshake = [0_u8; 8 + KEY_HEX_BYTES];
            p_to_c_r.read_exact(&mut handshake)?;
            assert_eq!(&handshake[..8], HANDSHAKE_REQUEST_MAGIC);
            assert_eq!(&handshake[8..], VALID_KEY.as_bytes());

            // No KIND_VERIFIED_SECTION frame may be emitted before SB2OKAY1 arrives.
            std::thread::sleep(Duration::from_millis(25));
            assert_eq!(p_to_c_r.buffered_bytes(), 0);

            c_to_p_w.write_all(HANDSHAKE_RESPONSE_MAGIC)?;
            c_to_p_w.flush()?;

            for expected in &sections {
                assert_verified_section(&mut p_to_c_r, expected)?;
            }

            // wait_server_ready stays pending until KIND_SERVER_READY is sent after attestation.
            assert!(
                matches!(
                    backend
                        .wait_server_ready(Instant::now() + Duration::from_millis(25))
                        .await,
                    Err(BackendError::Deadline)
                ),
                "wait_server_ready completed before KIND_SERVER_READY"
            );

            write_frame(&mut c_to_p_w, 0, KIND_SERVER_READY, &[])?;
            backend
                .wait_server_ready(Instant::now() + Duration::from_secs(2))
                .await
                .map_err(|_| io::Error::other("wait_server_ready failed"))?;

            let mut stream = backend
                .connect(Instant::now() + Duration::from_secs(1))
                .await
                .map_err(|_| io::Error::other("connect after verified sections failed"))?;
            stream.write_all(b"GET /v1/models HTTP/1.1\r\n\r\n").await?;
            stream.flush().await?;
            assert_eq!(read_frame(&mut p_to_c_r)?, (1, KIND_OPEN, Vec::new()));
            assert_eq!(
                read_frame(&mut p_to_c_r)?,
                (1, KIND_DATA, b"GET /v1/models HTTP/1.1\r\n\r\n".to_vec())
            );
            write_frame(&mut c_to_p_w, 1, KIND_DATA, b"HTTP/1.1 200 OK\r\n\r\n")?;
            write_frame(&mut c_to_p_w, 1, KIND_EOF, &[])?;
            let mut reply = Vec::new();
            stream.read_to_end(&mut reply).await?;
            assert_eq!(reply, b"HTTP/1.1 200 OK\r\n\r\n");
            drop(stream);
            assert_eq!(read_frame(&mut p_to_c_r)?, (1, KIND_EOF, Vec::new()));
            Ok(())
        })
    }

    #[test]
    fn from_io_with_sections_suppresses_attestations_when_ack_fails() -> io::Result<()> {
        run_async(async {
            let sections = [sample_attestation(0, 0x330, 4096)];
            let (p_to_c_w, mut p_to_c_r) = memory_channel();
            let (mut c_to_p_w, c_to_p_r) = memory_channel();

            let backend =
                PipeBackend::from_io_with_sections(c_to_p_r, p_to_c_w, VALID_KEY, &sections)
                    .map_err(|_| io::Error::other("from_io_with_sections failed"))?;

            let mut handshake = [0_u8; 8 + KEY_HEX_BYTES];
            p_to_c_r.read_exact(&mut handshake)?;
            c_to_p_w.write_all(b"BADACK01")?;
            c_to_p_w.flush()?;

            assert!(matches!(
                backend
                    .wait_server_ready(Instant::now() + Duration::from_secs(2))
                    .await,
                Err(BackendError::Io)
            ));
            let mut trailing = Vec::new();
            p_to_c_r.read_to_end(&mut trailing)?;
            assert!(trailing.is_empty());
            Ok(())
        })
    }

    #[test]
    fn connect_stdio_with_sections_exchanges_attestations_over_os_pipes() -> io::Result<()> {
        use std::os::windows::io::OwnedHandle;

        run_async(async {
            let (mut p_to_c_r, p_to_c_w) = io::pipe()?;
            let (c_to_p_r, mut c_to_p_w) = io::pipe()?;
            let sections = [sample_attestation(0, 0x440, 12_288)];

            let backend = PipeBackend::connect_stdio_with_sections(
                File::from(OwnedHandle::from(p_to_c_w)),
                File::from(OwnedHandle::from(c_to_p_r)),
                VALID_KEY,
                &sections,
            )
            .map_err(|_| io::Error::other("connect_stdio_with_sections failed"))?;

            complete_handshake(&mut p_to_c_r, &mut c_to_p_w)?;
            assert_verified_section(&mut p_to_c_r, &sections[0])?;

            write_frame(&mut c_to_p_w, 0, KIND_SERVER_READY, &[])?;
            backend
                .wait_server_ready(Instant::now() + Duration::from_secs(2))
                .await
                .map_err(|_| io::Error::other("wait_server_ready failed"))?;
            Ok(())
        })
    }
}
