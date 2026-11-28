//! Shared bounded loopback HTTP/1.1 transport, header scanning, and body framing.

#![forbid(unsafe_code)]
#![expect(
    clippy::redundant_pub_crate,
    reason = "The native HTTP adapters share this crate-private transport layer."
)]

use crate::scrub::ScrubbingBytes;
use std::{
    io::{self, BufRead as _, BufReader, IoSlice, Read, Write},
    net::{Ipv4Addr, SocketAddr, SocketAddrV4, TcpStream},
    time::{Duration, Instant},
};

/// Maximum HTTP/1.1 response header block admitted from a loopback peer.
pub(crate) const MAX_HEAD: usize = 16 * 1024;
const MAX_CHUNK_LINE: usize = 32;
const CONNECT_TRANSIENT_BACKOFF_MS: [u64; 2] = [50, 150];

/// Shared bounded HTTP/1.1 transport and framing failures.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TransportError {
    /// Failed to establish a loopback TCP connection.
    Connect,
    /// Failed to apply required loopback socket options or timeouts.
    Configure,
    /// Upload size exceeded the transport limit or slice budget.
    UploadSize,
    /// Peer closed the connection while uploading the request.
    UploadClosed,
    /// Socket error while uploading the request.
    UploadInterrupted,
    /// Response header exceeded `MAX_HEAD` bytes.
    HeadOverflow,
    /// Malformed HTTP/1.1 status line or header syntax.
    InvalidHead,
    /// Duplicate header name in the response head.
    DuplicateHeader,
    /// Unsupported `Content-Encoding` header.
    UnsupportedEncoding,
    /// Unexpected or malformed `Content-Type` header.
    UnsupportedMedia,
    /// Missing, invalid, or oversized `Content-Length` header.
    InvalidLength,
    /// Unsupported, missing, or conflicting body framing headers.
    UnsupportedFraming,
    /// Peer closed before completing headers or declared body framing.
    ResponseClosed,
    /// Socket error while reading the response.
    ResponseInterrupted,
    /// Peer sent more payload bytes than the declared `Content-Length`.
    LengthExceeded,
    /// Chunk-size or trailer line exceeded the bounded metadata buffer.
    LineOverflow,
    /// Invalid CRLF delimiter following a chunk payload.
    InvalidChunkEnding,
    /// Malformed chunk framing or missing terminal data.
    InvalidChunkFraming,
    /// Invalid hex digit or alignment constraint on a chunk size.
    InvalidChunkSize,
    /// Streamed body exceeded the declared total limit.
    BodyLimitExceeded,
    /// Chunked response included unsupported trailing headers.
    UnsupportedTrailers,
    /// Unexpected trailing bytes appeared after the terminal `0` chunk.
    TrailingBytes,
    /// Failed to reserve bounded buffer space for the response body.
    Allocation,
    /// Recognized HTTP error status returned by a local companion.
    Status(u16),
    /// Unrecognized HTTP status code returned by the local broker.
    UnexpectedStatus,
}

/// Transport error carrying an adapter-specific cancellation/sink failure `E`.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Error<E> {
    /// Request or callback interrupted/cancelled via caller control signal.
    Interrupted(E),
    /// Bounded HTTP/1.1 transport or protocol failure.
    Transport(TransportError),
}

impl<E> From<TransportError> for Error<E> {
    fn from(error: TransportError) -> Self {
        Self::Transport(error)
    }
}

/// Returns `true` when an I/O error warrants polling for cancellation and retrying.
pub(crate) fn retry_poll(error: &io::Error) -> bool {
    matches!(
        error.kind(),
        io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut | io::ErrorKind::Interrupted
    )
}

fn retry_transient_connect(error: &io::Error) -> bool {
    matches!(
        error.kind(),
        io::ErrorKind::ConnectionRefused
            | io::ErrorKind::ConnectionReset
            | io::ErrorKind::AddrNotAvailable
    )
}

fn wait_interruptible_with<E>(
    delay: Duration,
    deadline: Instant,
    poll_step: &mut impl FnMut() -> Duration,
    check: &mut impl FnMut() -> Result<(), E>,
    now_fn: &mut impl FnMut() -> Instant,
    sleep: &mut impl FnMut(Duration),
) -> Result<bool, Error<E>> {
    let wait_until = (now_fn() + delay).min(deadline);
    loop {
        check().map_err(Error::Interrupted)?;
        let now = now_fn();
        if now >= deadline {
            return Ok(false);
        }
        if now >= wait_until {
            return Ok(true);
        }
        let step = poll_step().min(wait_until.saturating_duration_since(now));
        if step.is_zero() {
            return Ok(false);
        }
        sleep(step);
    }
}

/// Connects only to `127.0.0.1:port` while polling caller cancellation checks.
pub(crate) fn connect_loopback<E>(
    port: u16,
    connect_timeout: Duration,
    mut poll_step: impl FnMut() -> Duration,
    mut check: impl FnMut() -> Result<(), E>,
) -> Result<TcpStream, Error<E>> {
    check().map_err(Error::Interrupted)?;
    let address = SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::LOCALHOST, port));
    connect_socket(address, connect_timeout, &mut poll_step, &mut check)
}

/// Connects to `address` using bounded `connect_timeout` slices without spawning a helper thread.
pub(crate) fn connect_socket<E>(
    address: SocketAddr,
    connect_timeout: Duration,
    mut poll_step: impl FnMut() -> Duration,
    mut check: impl FnMut() -> Result<(), E>,
) -> Result<TcpStream, Error<E>> {
    connect_socket_with(
        address,
        connect_timeout,
        &mut poll_step,
        &mut check,
        &mut |addr, slice| TcpStream::connect_timeout(addr, slice),
        &mut Instant::now,
        &mut std::thread::sleep,
    )
}

fn connect_socket_with<E>(
    address: SocketAddr,
    connect_timeout: Duration,
    poll_step: &mut impl FnMut() -> Duration,
    check: &mut impl FnMut() -> Result<(), E>,
    connect: &mut impl FnMut(&SocketAddr, Duration) -> io::Result<TcpStream>,
    now_fn: &mut impl FnMut() -> Instant,
    sleep: &mut impl FnMut(Duration),
) -> Result<TcpStream, Error<E>> {
    check().map_err(Error::Interrupted)?;
    let deadline = now_fn() + connect_timeout;
    let mut transient_retry = 0_usize;
    loop {
        check().map_err(Error::Interrupted)?;
        let remaining = deadline.saturating_duration_since(now_fn());
        if remaining.is_zero() {
            return Err(Error::Transport(TransportError::Connect));
        }
        let slice = poll_step().min(remaining);
        if slice.is_zero() {
            return Err(Error::Transport(TransportError::Connect));
        }
        match connect(&address, slice) {
            Ok(stream) => {
                check().map_err(Error::Interrupted)?;
                return Ok(stream);
            }
            Err(error) if retry_poll(&error) => {
                check().map_err(Error::Interrupted)?;
                if now_fn() >= deadline {
                    return Err(Error::Transport(TransportError::Connect));
                }
            }
            Err(error)
                if retry_transient_connect(&error)
                    && transient_retry < CONNECT_TRANSIENT_BACKOFF_MS.len() =>
            {
                let delay = Duration::from_millis(CONNECT_TRANSIENT_BACKOFF_MS[transient_retry]);
                transient_retry += 1;
                if !wait_interruptible_with(delay, deadline, poll_step, check, now_fn, sleep)? {
                    return Err(Error::Transport(TransportError::Connect));
                }
            }
            Err(_) => {
                check().map_err(Error::Interrupted)?;
                return Err(Error::Transport(TransportError::Connect));
            }
        }
    }
}

/// Polling hook invoked before, after, and on retryable socket conditions.
#[derive(Clone, Copy)]
pub(crate) enum PollPhase {
    Check,
    Wait,
}

/// Socket adapter that surfaces caller cancellation `E` cleanly out of `std::io`.
pub(crate) struct PollStream<F, E> {
    stream: TcpStream,
    poll: F,
    interrupted: Option<E>,
}

impl<F, E> PollStream<F, E>
where
    F: FnMut(PollPhase) -> Result<(), E>,
{
    /// Wrap a configured loopback stream with a cancellation/poll hook.
    pub(crate) const fn new(stream: TcpStream, poll: F) -> Self {
        Self {
            stream,
            poll,
            interrupted: None,
        }
    }

    /// Run a non-socket cancellation check, retaining any first interruption.
    pub(crate) fn check(&mut self) -> io::Result<()> {
        if self.interrupted.is_some() {
            return Err(aborted_error());
        }
        if let Err(reason) = (self.poll)(PollPhase::Check) {
            self.interrupted = Some(reason);
            return Err(aborted_error());
        }
        Ok(())
    }

    /// Extract the retained interruption reason, if any.
    pub(crate) fn take_interrupted(&mut self) -> Option<E> {
        self.interrupted.take()
    }

    fn wait(&mut self) -> io::Result<()> {
        if self.interrupted.is_some() {
            return Err(aborted_error());
        }
        if let Err(reason) = (self.poll)(PollPhase::Wait) {
            self.interrupted = Some(reason);
            return Err(aborted_error());
        }
        self.check()
    }

    fn run_retryable<T>(
        &mut self,
        mut op: impl FnMut(&mut TcpStream) -> io::Result<T>,
    ) -> io::Result<T> {
        loop {
            self.check()?;
            match op(&mut self.stream) {
                Err(error) if retry_poll(&error) => self.wait()?,
                result => {
                    self.check()?;
                    return result;
                }
            }
        }
    }
}

fn aborted_error() -> io::Error {
    io::Error::new(
        io::ErrorKind::ConnectionAborted,
        "Local HTTP request interrupted",
    )
}

impl<F, E> Read for PollStream<F, E>
where
    F: FnMut(PollPhase) -> Result<(), E>,
{
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        self.run_retryable(|stream| stream.read(bytes))
    }
}

impl<F, E> Write for PollStream<F, E>
where
    F: FnMut(PollPhase) -> Result<(), E>,
{
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.run_retryable(|stream| stream.write(bytes))
    }

    fn write_vectored(&mut self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        self.run_retryable(|stream| stream.write_vectored(bufs))
    }

    fn flush(&mut self) -> io::Result<()> {
        self.stream.flush()
    }
}

/// Builds an HTTP/1.1 loopback request head in pre-sized memory scrubbed on drop.
pub(crate) fn bearer_request_head(
    method: &str,
    route: &str,
    port: u16,
    key: &str,
    content_type: Option<&str>,
    accept_json_identity: bool,
    content_length: usize,
) -> ScrubbingBytes {
    // Reserve the full formatted line upfront so the bearer secret never
    // triggers a realloc copy before drop scrubbing clears the buffer.
    let capacity = 192
        + method.len()
        + route.len()
        + key.len()
        + content_type.map_or(0, |value| 16 + value.len());
    let mut head = Vec::with_capacity(capacity);
    let _written = write!(
        head,
        "{method} {route} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nAuthorization: Bearer {key}\r\n"
    );
    if accept_json_identity {
        let _written = write!(
            head,
            "Accept: application/json\r\nAccept-Encoding: identity\r\n"
        );
    }
    if let Some(media) = content_type {
        let _written = write!(head, "Content-Type: {media}\r\n");
    }
    let _written = write!(
        head,
        "Content-Length: {content_length}\r\nConnection: close\r\n\r\n"
    );
    ScrubbingBytes::new(head)
}

/// Sends gathered slices completely while polling `check` and `wait` on partial/retry writes.
pub(crate) fn write_all_vectored<E>(
    stream: &mut impl Write,
    mut buffers: &mut [IoSlice<'_>],
    mut wait: impl FnMut() -> Result<(), E>,
    mut check: impl FnMut() -> Result<(), E>,
) -> Result<(), Error<E>> {
    IoSlice::advance_slices(&mut buffers, 0);
    while !buffers.is_empty() {
        check().map_err(Error::Interrupted)?;
        let result = stream.write_vectored(buffers);
        check().map_err(Error::Interrupted)?;
        match result {
            Ok(0) => return Err(Error::Transport(TransportError::UploadClosed)),
            Ok(count) => IoSlice::advance_slices(&mut buffers, count),
            Err(error) if retry_poll(&error) => wait().map_err(Error::Interrupted)?,
            Err(_) => return Err(Error::Transport(TransportError::UploadInterrupted)),
        }
    }
    check().map_err(Error::Interrupted)
}

/// Locates the first HTTP header ending without rescanning an unchanged prefix.
#[derive(Default)]
pub(super) struct HeaderEnd {
    next: usize,
    #[cfg(test)]
    comparisons: usize,
}

impl HeaderEnd {
    /// Receives the same growing buffer until its first complete header ending.
    pub(super) fn find(&mut self, bytes: &[u8]) -> Option<usize> {
        let window = &bytes[self.next..];
        if let Some(offset) = memchr::memmem::find(window, b"\r\n\r\n") {
            #[cfg(test)]
            {
                self.comparisons += offset + 1;
            }
            return Some(self.next + offset + 4);
        }
        #[cfg(test)]
        {
            self.comparisons += window.len().saturating_sub(3);
        }
        // Only these trailing bytes can begin a delimiter spanning the next read.
        self.next = bytes.len().saturating_sub(3);
        None
    }
}

/// Accept the expected media type with at most one explicit UTF-8 charset.
pub(super) fn utf8_media_type(value: &[u8], expected: &str) -> bool {
    let Ok(value) = std::str::from_utf8(value) else {
        return false;
    };
    let mut parts = value.split(';');
    if !parts.next().is_some_and(|media| {
        media
            .trim_matches([' ', '\t'])
            .eq_ignore_ascii_case(expected)
    }) {
        return false;
    }
    let Some(parameter) = parts.next() else {
        return true;
    };
    let Some((name, charset)) = parameter.split_once('=') else {
        return false;
    };
    // HTTP optional whitespace is SP / HTAB, not Unicode White_Space.
    let charset = charset.trim_matches([' ', '\t']);
    name.trim_matches([' ', '\t'])
        .eq_ignore_ascii_case("charset")
        && (charset.eq_ignore_ascii_case("utf-8") || charset.eq_ignore_ascii_case("\"utf-8\""))
        && parts.next().is_none()
}

/// Framing mode validated on a parsed HTTP/1.1 response head.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Framing {
    /// `Transfer-Encoding: chunked` streaming payload.
    Chunked,
    /// Exact `Content-Length` payload.
    Length(usize),
}

/// Framing constraint enforced during `ResponseHead::parse`.
#[derive(Clone, Copy)]
pub(crate) enum AllowedFraming {
    /// Permit either `Content-Length` or `Transfer-Encoding: chunked`.
    LengthOrChunked,
    /// Require an explicit `Content-Length`.
    LengthOnly,
    /// Require `Transfer-Encoding: chunked`.
    ChunkedOnly,
}

/// Explicit per-endpoint framing and header rules for HTTP/1.1 responses.
#[derive(Clone, Copy)]
pub(crate) struct HeadPolicy {
    /// Framing mode permitted on this response.
    pub(crate) framing: AllowedFraming,
    /// Require explicit `Connection: close`.
    pub(crate) require_connection_close: bool,
    /// Allow HTAB (`0x09`) inside header values alongside printable bytes.
    pub(crate) allow_htab: bool,
}

/// Validated `Retry-After` metadata without retaining borrowed header bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RetryAfter {
    /// No `Retry-After` header was present.
    Missing,
    /// Exact `Retry-After: 1` header value.
    OneSecond,
    /// Any other `Retry-After` value.
    Other,
}

/// Validated HTTP/1.1 response head metadata.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ResponseHead {
    /// HTTP status code (e.g. `200`).
    pub(crate) code: u16,
    /// Validated `Retry-After` classification.
    pub(crate) retry_after: RetryAfter,
    /// Validated framing for the response body.
    pub(crate) framing: Framing,
    /// Maximum body size allowed for this response status.
    pub(crate) max_body: usize,
}

impl ResponseHead {
    /// Parses and strictly validates one bounded HTTP/1.1 response head block.
    pub(crate) fn parse(
        head: &[u8],
        policy: HeadPolicy,
        mut status_limit: impl FnMut(u16) -> Result<usize, TransportError>,
        mut accept_media: impl FnMut(u16, &[u8]) -> bool,
    ) -> Result<Self, TransportError> {
        if !head.starts_with(b"HTTP/1.1 ") || has_bare_cr_or_lf(head) {
            return Err(TransportError::InvalidHead);
        }
        let mut headers = [httparse::EMPTY_HEADER; 64];
        let mut response = httparse::Response::new(&mut headers);
        if response.parse(head).ok() != Some(httparse::Status::Complete(head.len()))
            || response.version != Some(1)
        {
            return Err(TransportError::InvalidHead);
        }
        let code = response.code.ok_or(TransportError::InvalidHead)?;
        let limit = status_limit(code)?;

        let mut content_type = None;
        let mut length = None;
        let mut transfer = None;
        let mut retry_after = None;
        let mut connection_close = false;
        for (index, header) in response.headers.iter().enumerate() {
            if response.headers[..index]
                .iter()
                .any(|prev| prev.name.eq_ignore_ascii_case(header.name))
            {
                return Err(TransportError::DuplicateHeader);
            }
            if ["trailer", "upgrade"]
                .iter()
                .any(|forbidden| header.name.eq_ignore_ascii_case(forbidden))
            {
                return Err(TransportError::UnsupportedFraming);
            }
            if header.name.eq_ignore_ascii_case("content-encoding") {
                return Err(TransportError::UnsupportedEncoding);
            }
            let invalid_ctl = header
                .value
                .iter()
                .any(|&byte| byte == 127 || (byte < 32 && !(policy.allow_htab && byte == b'\t')));
            if invalid_ctl {
                return Err(TransportError::InvalidHead);
            }
            if header.name.eq_ignore_ascii_case("content-type") {
                content_type = Some(header.value);
            } else if header.name.eq_ignore_ascii_case("content-length") {
                length = Some(header.value);
            } else if header.name.eq_ignore_ascii_case("transfer-encoding") {
                transfer = Some(header.value);
            } else if header.name.eq_ignore_ascii_case("retry-after") {
                retry_after = Some(header.value);
            } else if header.name.eq_ignore_ascii_case("connection") {
                if !header.value.eq_ignore_ascii_case(b"close") {
                    return Err(TransportError::UnsupportedFraming);
                }
                connection_close = true;
            }
        }
        if policy.require_connection_close && !connection_close {
            return Err(TransportError::UnsupportedFraming);
        }
        if !content_type.is_some_and(|value| accept_media(code, value)) {
            return Err(TransportError::UnsupportedMedia);
        }

        let framing = match (policy.framing, length, transfer) {
            (AllowedFraming::LengthOrChunked | AllowedFraming::ChunkedOnly, None, Some(value))
                if value.eq_ignore_ascii_case(b"chunked") =>
            {
                Framing::Chunked
            }
            (AllowedFraming::LengthOrChunked | AllowedFraming::LengthOnly, Some(value), None)
                if !value.is_empty() && value.iter().all(u8::is_ascii_digit) =>
            {
                let parsed = std::str::from_utf8(value)
                    .ok()
                    .and_then(|text| text.parse::<usize>().ok())
                    .filter(|&value| value <= limit)
                    .ok_or(TransportError::InvalidLength)?;
                Framing::Length(parsed)
            }
            _ => return Err(TransportError::UnsupportedFraming),
        };
        let retry_after = match retry_after {
            None => RetryAfter::Missing,
            Some(value) if value == b"1" => RetryAfter::OneSecond,
            Some(_) => RetryAfter::Other,
        };
        Ok(Self {
            code,
            retry_after,
            framing,
            max_body: limit,
        })
    }
}

fn has_bare_cr_or_lf(bytes: &[u8]) -> bool {
    bytes.iter().enumerate().any(|(index, &byte)| {
        (byte == b'\n' && (index == 0 || bytes[index - 1] != b'\r'))
            || (byte == b'\r' && bytes.get(index + 1) != Some(&b'\n'))
    })
}

/// Configuration for streaming Content-Length or chunked HTTP bodies.
#[derive(Clone, Copy)]
pub(crate) struct BodyPolicy {
    /// Maximum chunk-size line length including CRLF (at most `32`).
    pub(crate) chunk_line_limit: usize,
    /// Total streamed byte budget across chunks (used for chunked framing).
    pub(crate) max_body: usize,
    /// Require each non-zero chunk length to be even (PCM16 sample alignment).
    pub(crate) require_even_chunks: bool,
    /// Reject empty `0`-chunk streams with zero payload bytes.
    pub(crate) require_non_empty: bool,
}

/// Bounded streaming HTTP/1.1 body decoder supporting `Content-Length` and `chunked`.
pub(crate) struct BodyReader<R> {
    reader: BufReader<R>,
    framing: Framing,
    policy: BodyPolicy,
    remaining: usize,
    total: usize,
    done: bool,
    chunk_ending: bool,
    borrowed: usize,
}

impl<R: Read> BodyReader<R> {
    /// Wraps `reader` in a fixed-capacity buffer, parses response headers, and prepares framing.
    pub(crate) fn new(
        reader: R,
        read_capacity: usize,
        head_policy: HeadPolicy,
        status_limit: impl FnMut(u16) -> Result<usize, TransportError>,
        accept_media: impl FnMut(u16, &[u8]) -> bool,
        mut body_policy: BodyPolicy,
    ) -> Result<(ResponseHead, Self), TransportError> {
        let mut buffered = BufReader::with_capacity(read_capacity, reader);
        let head = read_head(&mut buffered, head_policy, status_limit, accept_media)?;
        body_policy.max_body = body_policy.max_body.min(head.max_body);
        let body = Self {
            reader: buffered,
            framing: head.framing,
            policy: body_policy,
            remaining: 0,
            total: 0,
            done: false,
            chunk_ending: false,
            borrowed: 0,
        };
        Ok((head, body))
    }

    /// Borrows the next ready payload slice directly from the reader buffer.
    pub(crate) fn next_payload(&mut self) -> Result<Option<&[u8]>, TransportError> {
        self.reader.consume(self.borrowed);
        self.borrowed = 0;
        if self.done {
            return Ok(None);
        }
        let remaining = match self.framing {
            Framing::Length(length) => length - self.total,
            Framing::Chunked => {
                self.prepare_chunk()?;
                self.remaining
            }
        };
        if remaining == 0 {
            self.done = true;
            return Ok(None);
        }
        // Release the ready prefix immediately without waiting to fill the buffer.
        let available = self
            .reader
            .fill_buf()
            .map_err(|_| TransportError::ResponseInterrupted)?;
        if available.is_empty() {
            return Err(TransportError::ResponseClosed);
        }
        let count = available.len().min(remaining);
        self.total += count;
        if matches!(self.framing, Framing::Chunked) {
            self.remaining -= count;
            self.chunk_ending = self.remaining == 0;
        }
        self.borrowed = count;
        Ok(Some(&available[..count]))
    }

    /// Confirms the peer reached clean EOF immediately after the terminal `0` chunk.
    pub(crate) fn verify_eof(&mut self) -> Result<(), TransportError> {
        self.reader.consume(self.borrowed);
        self.borrowed = 0;
        let remaining = self
            .reader
            .fill_buf()
            .map_err(|_| TransportError::ResponseInterrupted)?;
        if remaining.is_empty() {
            Ok(())
        } else {
            Err(TransportError::TrailingBytes)
        }
    }

    /// Mutably borrows the underlying transport reader.
    pub(crate) fn inner_mut(&mut self) -> &mut R {
        self.reader.get_mut()
    }

    #[cfg(test)]
    pub(crate) fn buffer_ptr(&self) -> *const u8 {
        self.reader.buffer().as_ptr()
    }

    fn prepare_chunk(&mut self) -> Result<(), TransportError> {
        if self.remaining == 0 {
            if self.chunk_ending {
                let mut ending = [0; 2];
                read_exact_unbuffered(&mut self.reader, &mut ending)?;
                if ending != *b"\r\n" {
                    return Err(TransportError::InvalidChunkEnding);
                }
                self.chunk_ending = false;
            }
            let mut line_buf = [0; MAX_CHUNK_LINE];
            let limit = self.policy.chunk_line_limit.min(MAX_CHUNK_LINE);
            let line = read_line(&mut self.reader, &mut line_buf[..limit])?;
            if line.is_empty() || !line.iter().all(u8::is_ascii_hexdigit) {
                return Err(TransportError::InvalidChunkFraming);
            }
            let text = std::str::from_utf8(line).map_err(|_| TransportError::InvalidChunkSize)?;
            let length =
                usize::from_str_radix(text, 16).map_err(|_| TransportError::InvalidChunkSize)?;
            if length > self.policy.max_body - self.total {
                return Err(TransportError::BodyLimitExceeded);
            }
            if length == 0 {
                if !read_line(&mut self.reader, &mut [0; 2])?.is_empty() {
                    return Err(TransportError::UnsupportedTrailers);
                }
                if self.policy.require_non_empty && self.total == 0 {
                    return Err(TransportError::InvalidChunkFraming);
                }
                self.done = true;
            } else if self.policy.require_even_chunks && !length.is_multiple_of(2) {
                return Err(TransportError::InvalidChunkFraming);
            }
            self.remaining = length;
        }
        Ok(())
    }
}

fn read_head<R: Read>(
    reader: &mut BufReader<R>,
    policy: HeadPolicy,
    status_limit: impl FnMut(u16) -> Result<usize, TransportError>,
    accept_media: impl FnMut(u16, &[u8]) -> bool,
) -> Result<ResponseHead, TransportError> {
    let available = reader
        .fill_buf()
        .map_err(|_| TransportError::ResponseInterrupted)?;
    if available.is_empty() {
        return Err(TransportError::ResponseClosed);
    }
    let window = &available[..available.len().min(MAX_HEAD)];
    if let Some(offset) = memchr::memmem::find(window, b"\r\n\r\n") {
        let end = offset + 4;
        let parsed = ResponseHead::parse(&window[..end], policy, status_limit, accept_media)?;
        reader.consume(end);
        return Ok(parsed);
    }
    if available.len() >= MAX_HEAD {
        return Err(TransportError::HeadOverflow);
    }
    let mut head = available.to_vec();
    let consumed = head.len();
    reader.consume(consumed);
    while !head.ends_with(b"\r\n\r\n") {
        if head.len() == MAX_HEAD {
            return Err(TransportError::HeadOverflow);
        }
        head.push(read_byte(reader)?);
    }
    ResponseHead::parse(&head, policy, status_limit, accept_media)
}

/// Validates and reads an HTTP head alongside coalesced prefix bytes into a single buffer.
pub(crate) fn read_head_bytes<E>(
    mut read: impl FnMut(&mut [u8]) -> Result<usize, Error<E>>,
) -> Result<(Vec<u8>, usize), Error<E>> {
    let mut received = Vec::new();
    let mut buffer = [0; 4096];
    let mut header_end = HeaderEnd::default();
    let end = loop {
        if let Some(end) = header_end.find(&received) {
            break end;
        }
        if received.len() >= MAX_HEAD {
            return Err(Error::Transport(TransportError::HeadOverflow));
        }
        let size = buffer.len().min(MAX_HEAD - received.len());
        let count = read(&mut buffer[..size])?;
        received.extend_from_slice(&buffer[..count]);
    };
    Ok((received, end))
}

/// Reads an exact `Content-Length` body directly into `received`, reusing the header buffer.
pub(crate) fn read_exact_body<E>(
    mut received: Vec<u8>,
    end: usize,
    length: usize,
    mut read: impl FnMut(&mut [u8]) -> Result<usize, Error<E>>,
) -> Result<Vec<u8>, Error<E>> {
    let prefix = received.len() - end;
    if prefix > length {
        return Err(Error::Transport(TransportError::LengthExceeded));
    }
    received.copy_within(end.., 0);
    received.truncate(prefix);
    received
        .try_reserve_exact(length - prefix)
        .map_err(|_| Error::Transport(TransportError::Allocation))?;
    received.resize(length, 0);
    let mut filled = prefix;
    while filled < length {
        let count = read(&mut received[filled..])?;
        if count == 0 {
            return Err(Error::Transport(TransportError::ResponseClosed));
        }
        filled += count;
    }
    Ok(received)
}

fn read_exact_unbuffered(reader: &mut impl Read, bytes: &mut [u8]) -> Result<(), TransportError> {
    // Do not call `Read::read_exact`: it loops on `Interrupted` and could hide
    // caller Stop/Shutdown cancellation until another socket read arrives.
    let mut filled = 0;
    while filled < bytes.len() {
        let count = reader
            .read(&mut bytes[filled..])
            .map_err(|_| TransportError::ResponseInterrupted)?;
        if count == 0 {
            return Err(TransportError::ResponseClosed);
        }
        filled += count;
    }
    Ok(())
}

fn read_byte(reader: &mut impl Read) -> Result<u8, TransportError> {
    let mut byte = [0];
    read_exact_unbuffered(reader, &mut byte)?;
    Ok(byte[0])
}

fn read_line<'a>(reader: &mut impl Read, bytes: &'a mut [u8]) -> Result<&'a [u8], TransportError> {
    let mut length = 0;
    loop {
        if length == bytes.len() {
            return Err(TransportError::LineOverflow);
        }
        bytes[length] = read_byte(reader)?;
        length += 1;
        if bytes[..length].ends_with(b"\r\n") {
            return Ok(&bytes[..length - 2]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{HeaderEnd, bearer_request_head, utf8_media_type};

    #[test]
    fn fragmented_headers_scan_each_window_once_and_preserve_coalesced_payload() {
        let mut head = vec![b'x'; 16_384 - 4];
        head.extend_from_slice(b"\r\n\r\n");
        for fragment in [1, 2, 3, 7, 4096, 16_384] {
            let mut scan = HeaderEnd::default();
            let mut received = Vec::new();
            for chunk in head.chunks(fragment) {
                received.extend_from_slice(chunk);
                let expected = (received.len() == head.len()).then_some(head.len());
                assert_eq!(scan.find(&received), expected);
            }
            assert_eq!(scan.comparisons, head.len() - 3);
        }
        for split in 1..=4 {
            let mut scan = HeaderEnd::default();
            assert_eq!(scan.find(&head[..head.len() - split]), None);
            let mut wire = head.clone();
            wire.extend_from_slice(b"body\r\n\r\n");
            assert_eq!(scan.find(&wire), Some(head.len()));
            assert_eq!(&wire[head.len()..], b"body\r\n\r\n");
            assert_eq!(scan.comparisons, head.len() - 3);
        }
    }

    #[test]
    fn local_media_types_share_the_same_utf8_parameter_subset() {
        for media in ["application/json", "text/event-stream"] {
            for suffix in [
                "",
                "; CHARSET=UTF-8",
                " ; charset = \"utf-8\"",
                "\t;\tcharset\t=\tutf-8\t",
            ] {
                assert!(utf8_media_type(
                    format!("{media}{suffix}").as_bytes(),
                    media
                ));
            }
            for suffix in [
                "-extra",
                ";",
                "; charset=latin1",
                "; charset=utf-8; charset=utf-8",
                "; charset=\"utf-8",
                "; other=utf-8",
            ] {
                assert!(!utf8_media_type(
                    format!("{media}{suffix}").as_bytes(),
                    media
                ));
            }
            assert!(!utf8_media_type(&[0xff], media));
            for whitespace in [
                '\u{00a0}', '\u{2003}', '\u{0085}', '\u{000b}', '\u{000c}', '\r', '\n',
            ] {
                for value in [
                    format!("{whitespace}{media}"),
                    format!("{media}{whitespace}"),
                    format!("{media};{whitespace}charset=utf-8"),
                    format!("{media};charset{whitespace}=utf-8"),
                    format!("{media};charset={whitespace}utf-8"),
                    format!("{media};charset=utf-8{whitespace}"),
                ] {
                    assert!(!utf8_media_type(value.as_bytes(), media), "{value:?}");
                }
            }
        }
        assert!(!utf8_media_type(b"application/json", "text/event-stream"));
        assert!(!utf8_media_type(b"text/event-stream", "application/json"));
    }

    #[test]
    fn bearer_request_head_matches_exact_local_wire_format_without_reallocation() {
        let secret = "0123456789abcdef".repeat(4);
        let head = bearer_request_head(
            "POST",
            "/v1/chat/completions",
            8080,
            &secret,
            Some("application/json"),
            false,
            42,
        );
        assert_eq!(
            &head[..],
            format!(
                "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:8080\r\nAuthorization: Bearer {secret}\r\nContent-Type: application/json\r\nContent-Length: 42\r\nConnection: close\r\n\r\n"
            )
            .as_bytes()
        );
    }

    #[test]
    fn connect_loopback_uses_bounded_slices_and_honors_cancellation() -> std::io::Result<()> {
        let listener = std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))?;
        let port = listener.local_addr()?.port();
        let connected = super::connect_loopback(
            port,
            std::time::Duration::from_millis(200),
            || std::time::Duration::from_millis(25),
            || Ok::<(), &'static str>(()),
        );
        assert!(connected.is_ok());

        let zero_step = super::connect_loopback(
            port,
            std::time::Duration::from_millis(200),
            || std::time::Duration::ZERO,
            || Ok::<(), &'static str>(()),
        );
        assert_eq!(
            zero_step.err(),
            Some(super::Error::Transport(super::TransportError::Connect))
        );

        let addr = std::net::SocketAddr::V4(std::net::SocketAddrV4::new(
            std::net::Ipv4Addr::LOCALHOST,
            port,
        ));
        drop(listener);
        let mut checks = 0_usize;
        let cancel_clock = std::cell::Cell::new(std::time::Instant::now());
        let mut cancel_sleeps = Vec::new();
        let cancelled = super::connect_socket_with(
            addr,
            std::time::Duration::from_millis(200),
            &mut || std::time::Duration::from_millis(25),
            &mut || {
                checks += 1;
                if checks >= 4 { Err("stopped") } else { Ok(()) }
            },
            &mut |_, _| {
                Err(std::io::Error::new(
                    std::io::ErrorKind::ConnectionRefused,
                    "refused",
                ))
            },
            &mut || cancel_clock.get(),
            &mut |delay| {
                cancel_sleeps.push(delay);
                cancel_clock.set(cancel_clock.get() + delay);
            },
        );
        assert_eq!(cancelled.err(), Some(super::Error::Interrupted("stopped")));
        assert_eq!(cancel_sleeps, vec![std::time::Duration::from_millis(25)]);

        // Transient ConnectionRefused retries within connect_timeout until peer binds.
        let delayed_listener = std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))?;
        let retry_addr = delayed_listener.local_addr()?;
        let mut attempts = 0_usize;
        let retry_clock = std::cell::Cell::new(std::time::Instant::now());
        let mut retry_sleeps = Vec::new();
        let retried = super::connect_socket_with(
            retry_addr,
            std::time::Duration::from_secs(2),
            &mut || std::time::Duration::from_millis(20),
            &mut || Ok::<(), &'static str>(()),
            &mut |addr, slice| {
                attempts += 1;
                if attempts == 1 {
                    Err(std::io::Error::new(
                        std::io::ErrorKind::ConnectionRefused,
                        "not yet bound",
                    ))
                } else {
                    std::net::TcpStream::connect_timeout(addr, slice)
                }
            },
            &mut || retry_clock.get(),
            &mut |delay| {
                retry_sleeps.push(delay);
                retry_clock.set(retry_clock.get() + delay);
            },
        );
        assert!(retried.is_ok());
        assert_eq!(attempts, 2);
        assert_eq!(
            retry_sleeps,
            vec![
                std::time::Duration::from_millis(20),
                std::time::Duration::from_millis(20),
                std::time::Duration::from_millis(10),
            ]
        );

        // Direct checks for retry_transient_connect classification and wait_interruptible.
        for kind in [
            std::io::ErrorKind::ConnectionRefused,
            std::io::ErrorKind::ConnectionReset,
            std::io::ErrorKind::AddrNotAvailable,
        ] {
            assert!(super::retry_transient_connect(&std::io::Error::new(
                kind,
                "transient"
            )));
        }
        for kind in [
            std::io::ErrorKind::PermissionDenied,
            std::io::ErrorKind::TimedOut,
        ] {
            assert!(!super::retry_transient_connect(&std::io::Error::new(
                kind,
                "non-transient"
            )));
        }

        let wait_base = std::time::Instant::now();
        let wait_clock = std::cell::Cell::new(wait_base);
        let future = wait_base + std::time::Duration::from_secs(1);
        assert!(matches!(
            super::wait_interruptible_with(
                std::time::Duration::from_millis(10),
                future,
                &mut || std::time::Duration::from_millis(5),
                &mut || Ok::<(), &'static str>(()),
                &mut || wait_clock.get(),
                &mut |delay| wait_clock.set(wait_clock.get() + delay),
            ),
            Ok(true)
        ));

        let expired = wait_base
            .checked_sub(std::time::Duration::from_millis(1))
            .unwrap_or(wait_base);
        assert!(matches!(
            super::wait_interruptible_with(
                std::time::Duration::from_millis(50),
                expired,
                &mut || std::time::Duration::from_millis(5),
                &mut || Ok::<(), &'static str>(()),
                &mut || wait_base,
                &mut |_| {},
            ),
            Ok(false)
        ));
        assert!(matches!(
            super::wait_interruptible_with(
                std::time::Duration::from_millis(50),
                wait_base + std::time::Duration::from_secs(1),
                &mut || std::time::Duration::ZERO,
                &mut || Ok::<(), &'static str>(()),
                &mut || wait_base,
                &mut |_| {},
            ),
            Ok(false)
        ));

        let mut polls = 0_usize;
        let mut int_sleeps = Vec::new();
        let int_clock = std::cell::Cell::new(wait_base);
        let interrupted = super::wait_interruptible_with(
            std::time::Duration::from_secs(5),
            wait_base + std::time::Duration::from_secs(5),
            &mut || std::time::Duration::from_millis(5),
            &mut || {
                polls += 1;
                if polls >= 2 { Err("cancelled") } else { Ok(()) }
            },
            &mut || int_clock.get(),
            &mut |delay| {
                int_sleeps.push(delay);
                int_clock.set(int_clock.get() + delay);
            },
        );
        assert_eq!(
            interrupted.err(),
            Some(super::Error::Interrupted("cancelled"))
        );
        assert_eq!(int_sleeps, vec![std::time::Duration::from_millis(5)]);
        Ok(())
    }
}
