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
const CHANNEL_CAPACITY: usize = 64;
const MAX_TRACKED_RESETS: usize = 16;

const KIND_OPEN: u8 = 1;
const KIND_DATA: u8 = 2;
const KIND_EOF: u8 = 3;
const KIND_RESET: u8 = 4;
const KIND_SERVER_READY: u8 = 5;

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
            closed: false,
            failed: false,
            reset_streams: VecDeque::new(),
            dropped_reset_data: VecDeque::new(),
            write_waker: None,
            flush_waker: None,
        }
    }

    fn is_reset(&self, stream_id: u32) -> bool {
        self.reset_streams.contains(&stream_id)
    }

    fn has_dropped_data(&self, stream_id: u32) -> bool {
        self.dropped_reset_data.contains(&stream_id)
    }

    fn record_dropped_data(&mut self, stream_id: u32) {
        if !self.has_dropped_data(stream_id) {
            if self.dropped_reset_data.len() >= MAX_TRACKED_RESETS {
                let _ = self.dropped_reset_data.pop_front();
            }
            self.dropped_reset_data.push_back(stream_id);
        }
    }

    fn record_reset(&mut self, stream_id: u32) {
        if !self.is_reset(stream_id) {
            if self.reset_streams.len() >= MAX_TRACKED_RESETS {
                let _ = self.reset_streams.pop_front();
            }
            self.reset_streams.push_back(stream_id);
        }
        let mut retained = VecDeque::with_capacity(self.queue.len());
        let mut removed_data = 0_usize;
        for frame in self.queue.drain(..) {
            if frame.stream_id == stream_id && matches!(frame.kind, KIND_DATA | KIND_EOF) {
                if frame.kind == KIND_DATA {
                    removed_data = removed_data.saturating_add(1);
                }
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
}

/// Parent-side broker pipe connector, carrying HTTP/1.1 requests over anonymous pipes.
pub struct PipeBackend {
    policy: BackendPolicy,
    tx: Arc<TxShared>,
    rx_state: Arc<Mutex<RxState>>,
    ready_rx: watch::Receiver<ReadyState>,
    next_stream_id: AtomicU32,
    closer: Arc<TxCloser>,
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
        Self::from_io(child_stdout, child_stdin, backend_key)
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
        let policy = BackendPolicy::new(BROKER_PIPE_HOST_PORT, backend_key)?;
        let raw_key = backend_key.as_bytes();
        if raw_key.len() != KEY_HEX_BYTES {
            return Err(BackendError::Configuration);
        }
        let mut boxed_key = Box::new([0_u8; KEY_HEX_BYTES]);
        boxed_key.copy_from_slice(raw_key);
        let handshake_key = ScrubHandshake(boxed_key);

        let tx = TxShared::new();
        let closer = Arc::new(TxCloser(Arc::clone(&tx)));
        let rx_state = Arc::new(Mutex::new(RxState::new()));
        let (ready_tx, ready_rx) = watch::channel(ReadyState::Pending);

        let tx_worker = Arc::clone(&tx);
        let rx_fail = Arc::clone(&rx_state);
        let ready_tx_fail = ready_tx.clone();
        std::thread::Builder::new()
            .name("broker-pipe-client-tx".to_owned())
            .spawn(move || {
                run_writer(writer, handshake_key, &tx_worker, &rx_fail, &ready_tx_fail);
            })
            .map_err(|_| BackendError::Io)?;

        let rx_tx = Arc::clone(&tx);
        let rx_worker = Arc::clone(&rx_state);
        std::thread::Builder::new()
            .name("broker-pipe-client-rx".to_owned())
            .spawn(move || run_reader(reader, &rx_tx, &rx_worker, &ready_tx))
            .map_err(|_| BackendError::Io)?;

        Ok(Self {
            policy,
            tx,
            rx_state,
            ready_rx,
            next_stream_id: AtomicU32::new(1),
            closer,
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
                let mut rx = lock_unpoisoned(&self.rx_state);
                if rx
                    .active
                    .as_ref()
                    .is_some_and(|(active_id, _, _)| *active_id == stream_id)
                {
                    rx.active = None;
                }
                return Err(BackendError::Io);
            }
            state.enqueued_seq = state.enqueued_seq.saturating_add(1);
            state.queue.push_back(OutgoingFrame {
                stream_id,
                kind: KIND_OPEN,
                payload: Vec::new(),
            });
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
    eof_sent: bool,
    pending_data_seq: u64,
    pending_eof_seq: u64,
}

impl Drop for PipeStream {
    fn drop(&mut self) {
        {
            let mut rx = lock_unpoisoned(&self.rx_state);
            if rx
                .active
                .as_ref()
                .is_some_and(|(active_id, _, _)| *active_id == self.stream_id)
            {
                rx.active = None;
            }
        }
        if !self.eof_sent {
            let mut state = lock_unpoisoned(&self.tx.state);
            if !state.closed && !state.failed && !state.is_reset(self.stream_id) {
                let kind = if self.eof_received.load(Ordering::Acquire) {
                    KIND_EOF
                } else {
                    state.record_reset(self.stream_id);
                    KIND_RESET
                };
                state.enqueued_seq = state.enqueued_seq.saturating_add(1);
                state.queue.push_back(OutgoingFrame {
                    stream_id: self.stream_id,
                    kind,
                    payload: Vec::new(),
                });
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
            self.buffered.clear();
            self.offset = 0;
            match self.rx.poll_recv(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(None) => return Poll::Ready(Ok(())),
                Poll::Ready(Some(Err(error))) => return Poll::Ready(Err(error)),
                Poll::Ready(Some(Ok(chunk))) => {
                    if !chunk.is_empty() {
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
            self.buffered.clear();
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
            state.enqueued_seq = state.enqueued_seq.saturating_add(1);
            let seq = state.enqueued_seq;
            state.queued_data_frames = state.queued_data_frames.saturating_add(1);
            state.queue.push_back(OutgoingFrame {
                stream_id: self.stream_id,
                kind: KIND_DATA,
                payload: buf[..count].to_vec(),
            });
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
            state.enqueued_seq = state.enqueued_seq.saturating_add(1);
            let seq = state.enqueued_seq;
            state.queue.push_back(OutgoingFrame {
                stream_id: self.stream_id,
                kind: KIND_EOF,
                payload: Vec::new(),
            });
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

fn fail_transport(
    tx: &Arc<TxShared>,
    rx_state: &Arc<Mutex<RxState>>,
    ready_tx: &watch::Sender<ReadyState>,
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
    state.failed = true;
    state.queue.clear();
    state.queued_data_frames = 0;
    state.wake_io();
    drop(state);
    tx.writer_ready.notify_one();
    let _ = ready_tx.send(ReadyState::Failed);
}

fn handle_reader_frame<R: Read>(
    reader: &mut R,
    tx: &Arc<TxShared>,
    rx_state: &Arc<Mutex<RxState>>,
    ready_tx: &watch::Sender<ReadyState>,
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
            let mut payload = vec![0_u8; len];
            reader.read_exact(&mut payload)?;
            let sender = {
                let rx = lock_unpoisoned(rx_state);
                rx.active
                    .as_ref()
                    .filter(|(active_id, _, _)| *active_id == stream_id)
                    .map(|(_, sender, _)| sender.clone())
            };
            if let Some(sender) = sender {
                let _ = sender.blocking_send(Ok(payload));
            }
        }
        KIND_EOF => {
            if stream_id == 0 || len != 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "invalid eof frame",
                ));
            }
            let mut rx = lock_unpoisoned(rx_state);
            if rx
                .active
                .as_ref()
                .is_some_and(|(active_id, _, _)| *active_id == stream_id)
                && let Some((_, sender, eof_received)) = rx.active.take()
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
                state.record_reset(stream_id);
                state.wake_io();
            }
            let mut rx = lock_unpoisoned(rx_state);
            if rx
                .active
                .as_ref()
                .is_some_and(|(active_id, _, _)| *active_id == stream_id)
                && let Some((_, sender, _)) = rx.active.take()
            {
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
) {
    let mut ack = [0_u8; HANDSHAKE_RESPONSE_MAGIC.len()];
    if reader.read_exact(&mut ack).is_err() || &ack != HANDSHAKE_RESPONSE_MAGIC {
        fail_transport(tx, rx_state, ready_tx);
        return;
    }
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
                let mut state = lock_unpoisoned(&tx.state);
                state.failed = true;
                state.wake_io();
                break;
            }
            Ok(HeaderRead::Frame(header)) => header,
            Err(_) => {
                fail_transport(tx, rx_state, ready_tx);
                break;
            }
        };
        if handle_reader_frame(&mut reader, tx, rx_state, ready_tx, header).is_err() {
            fail_transport(tx, rx_state, ready_tx);
            break;
        }
    }
}

fn run_writer<W: Write>(
    mut writer: W,
    handshake_key: ScrubHandshake,
    tx: &Arc<TxShared>,
    rx_state: &Arc<Mutex<RxState>>,
    ready_tx: &watch::Sender<ReadyState>,
) {
    let handshake_result = writer
        .write_all(HANDSHAKE_REQUEST_MAGIC)
        .and_then(|()| writer.write_all(handshake_key.0.as_slice()))
        .and_then(|()| writer.flush());
    drop(handshake_key);
    if handshake_result.is_err() {
        fail_transport(tx, rx_state, ready_tx);
        return;
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
        for frame in &batch {
            let Ok(payload_len) = u32::try_from(frame.payload.len()) else {
                io_result = Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "broker-pipe frame exceeds u32 bound",
                ));
                break;
            };
            let header = encode_header(frame.stream_id, frame.kind, payload_len);
            if let Err(error) = writer
                .write_all(&header)
                .and_then(|()| writer.write_all(&frame.payload))
            {
                io_result = Err(error);
                break;
            }
        }
        if io_result.is_ok() {
            io_result = writer.flush();
        }

        if io_result.is_err() {
            fail_transport(tx, rx_state, ready_tx);
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
        let len = u32::try_from(payload.len()).map_err(io::Error::other)?;
        writer.write_all(&encode_header(stream_id, kind, len))?;
        writer.write_all(payload)?;
        writer.flush()
    }

    fn read_frame(reader: &mut impl Read) -> io::Result<(u32, u8, Vec<u8>)> {
        let HeaderRead::Frame(header) = read_header(reader)? else {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "eof"));
        };
        let (stream_id, kind, len) = decode_header(header);
        let mut payload = vec![0_u8; len];
        reader.read_exact(&mut payload)?;
        Ok((stream_id, kind, payload))
    }

    #[test]
    fn handshake_readiness_large_exchanges_and_drop_reset_work_end_to_end() -> io::Result<()> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        runtime.block_on(async {
            let (p_to_c_w, mut p_to_c_r) = memory_channel();
            let (mut c_to_p_w, c_to_p_r) = memory_channel();

            let backend = PipeBackend::from_io(c_to_p_r, p_to_c_w, VALID_KEY)
                .map_err(|_| io::Error::other("PipeBackend::from_io failed"))?;

            let mut handshake = [0_u8; 8 + KEY_HEX_BYTES];
            p_to_c_r.read_exact(&mut handshake)?;
            assert_eq!(&handshake[..8], HANDSHAKE_REQUEST_MAGIC);
            assert_eq!(&handshake[8..], VALID_KEY.as_bytes());
            c_to_p_w.write_all(HANDSHAKE_RESPONSE_MAGIC)?;
            c_to_p_w.flush()?;

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
}
