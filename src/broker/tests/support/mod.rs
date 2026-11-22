#![expect(
    clippy::unwrap_used,
    reason = "shared test runtime setup must panic loudly on failure"
)]
#![allow(
    dead_code,
    reason = "shared helpers are used selectively across independent integration test targets"
)]

use bonsai_broker::{AdmittedChat, PendingChat, RequestPolicy, backend::BackendPolicy};
use http::{Request, Version, request::Parts};
use std::{
    cell::RefCell,
    io,
    pin::Pin,
    rc::Rc,
    task::{Context, Poll},
    time::Duration,
};
use tokio::{
    io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, ReadBuf},
    net::{TcpListener, TcpStream},
    task::LocalSet,
    time::{Instant, Sleep, sleep, timeout},
};

pub(crate) const CLIENT_KEY: &str =
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
pub(crate) const BACKEND_KEY: &str =
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
pub(crate) const COUNT_BODY: &[u8] = br#"{"input_tokens":100,"object":"response.input_tokens"}"#;

pub(crate) fn run(test: impl Future<Output = ()>) {
    tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .start_paused(true)
        .build()
        .unwrap()
        .block_on(test);
}

pub(crate) fn run_local_timeout(limit: Duration, test: impl Future<Output = ()>) {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(LocalSet::new().run_until(async {
            timeout(limit, test).await.unwrap();
        }));
}

pub(crate) fn deadline() -> Instant {
    Instant::now() + Duration::from_mins(1)
}

pub(crate) fn chat_policy() -> RequestPolicy {
    RequestPolicy::new(8080, CLIENT_KEY).unwrap()
}

pub(crate) fn backend_policy() -> BackendPolicy {
    BackendPolicy::new(8081, BACKEND_KEY).unwrap()
}

pub(crate) fn chat_head(size: usize) -> Parts {
    Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .version(Version::HTTP_11)
        .header("host", "127.0.0.1:8080")
        .header("origin", "http://127.0.0.1:8080")
        .header("authorization", format!("Bearer {CLIENT_KEY}"))
        .header("content-type", "application/json")
        .header("content-length", size.to_string())
        .body(())
        .unwrap()
        .into_parts()
        .0
}

pub(crate) fn pending_chat(body: &[u8]) -> PendingChat {
    chat_policy()
        .check_head(&chat_head(body.len()))
        .unwrap()
        .prepare(body)
        .unwrap()
}

pub(crate) fn admitted_chat(body: &[u8]) -> AdmittedChat {
    pending_chat(body).admit(|_| Ok(100)).unwrap()
}

pub(crate) fn http_post_request(
    port: u16,
    key: &str,
    origin_port: Option<u16>,
    body: &[u8],
) -> Vec<u8> {
    let origin =
        origin_port.map_or_else(String::new, |p| format!("Origin: http://127.0.0.1:{p}\r\n"));
    let mut bytes = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n{origin}Authorization: Bearer {key}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
        body.len()
    )
    .into_bytes();
    bytes.extend_from_slice(body);
    bytes
}

fn response_content_type(stream: bool) -> &'static str {
    if stream {
        "text/event-stream"
    } else {
        "application/json"
    }
}

pub(crate) fn http_length_response(body: &[u8], stream: bool) -> Vec<u8> {
    let mut bytes = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: {}\r\nContent-Length: {}\r\n\r\n",
        response_content_type(stream),
        body.len()
    )
    .into_bytes();
    bytes.extend_from_slice(body);
    bytes
}

pub(crate) fn http_chunked_head(stream: bool) -> Vec<u8> {
    format!(
        "HTTP/1.1 200 OK\r\nContent-Type: {}\r\nTransfer-Encoding: chunked\r\n\r\n",
        response_content_type(stream)
    )
    .into_bytes()
}

pub(crate) fn http_chunked_response(body: &[u8], stream: bool) -> Vec<u8> {
    let mut bytes = http_chunked_head(stream);
    bytes.extend_from_slice(body);
    bytes
}

pub(crate) async fn loopback_listener() -> TcpListener {
    TcpListener::bind("127.0.0.1:0").await.unwrap()
}

pub(crate) async fn read_http_request(stream: &mut TcpStream) -> (String, Vec<u8>) {
    let mut raw = Vec::new();
    loop {
        assert!(raw.len() <= 2 * 1024 * 1024);
        let mut buffer = [0; 4096];
        let count = stream.read(&mut buffer).await.unwrap();
        assert_ne!(count, 0);
        raw.extend_from_slice(&buffer[..count]);
        if let Some(end) = raw.array_windows().position(|part| part == b"\r\n\r\n") {
            let head = String::from_utf8(raw[..end + 4].to_vec()).unwrap();
            let length: usize = head
                .lines()
                .find_map(|line| line.strip_prefix("Content-Length: "))
                .unwrap()
                .parse()
                .unwrap();
            if raw.len() >= end + 4 + length {
                return (head, raw[end + 4..end + 4 + length].to_vec());
            }
        }
    }
}

pub(crate) async fn write_json_response(stream: &mut TcpStream, body: &[u8]) {
    let head = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
        body.len()
    );
    stream.write_all(head.as_bytes()).await.unwrap();
    stream.write_all(body).await.unwrap();
}

pub(crate) async fn serve_count_once(listener: &TcpListener) -> (String, Vec<u8>) {
    let (mut stream, _) = listener.accept().await.unwrap();
    let request = read_http_request(&mut stream).await;
    write_json_response(&mut stream, COUNT_BODY).await;
    request
}

#[derive(Default)]
pub(crate) struct MockStreamState {
    pub(crate) read_polls: usize,
    pub(crate) read_bytes: usize,
    pub(crate) largest_read: usize,
    pub(crate) write_polls: usize,
    pub(crate) written: Vec<u8>,
    pub(crate) flushes: usize,
    pub(crate) flush_offsets: Vec<usize>,
    pub(crate) shutdowns: usize,
    pub(crate) dropped: bool,
}

#[expect(
    clippy::struct_excessive_bools,
    reason = "independent fault-injection switches exercise each I/O failure mode"
)]
pub(crate) struct MockStream {
    pub(crate) input: Vec<u8>,
    pub(crate) position: usize,
    pub(crate) read_fragment: usize,
    pub(crate) stall_after_input: bool,
    pub(crate) fail_read_after_input: Option<io::ErrorKind>,
    pub(crate) read_delay: Option<Duration>,
    pub(crate) delayed_after: Option<(usize, Pin<Box<Sleep>>)>,
    read_timer: Option<Pin<Box<Sleep>>>,
    pub(crate) write_fragment: usize,
    pub(crate) write_delay: Option<Duration>,
    pub(crate) delay_write_in_body: bool,
    write_timer: Option<Pin<Box<Sleep>>>,
    pub(crate) stall_write: bool,
    pub(crate) fail_write: bool,
    pub(crate) zero_write: bool,
    pub(crate) stall_flush: bool,
    pub(crate) fail_flush: bool,
    pub(crate) stall_shutdown: bool,
    pub(crate) state: Rc<RefCell<MockStreamState>>,
}

impl MockStream {
    pub(crate) fn new(input: impl Into<Vec<u8>>, read_fragment: usize) -> Self {
        Self {
            input: input.into(),
            position: 0,
            read_fragment,
            stall_after_input: false,
            fail_read_after_input: None,
            read_delay: None,
            delayed_after: None,
            read_timer: None,
            write_fragment: usize::MAX,
            write_delay: None,
            delay_write_in_body: false,
            write_timer: None,
            stall_write: false,
            fail_write: false,
            zero_write: false,
            stall_flush: false,
            fail_flush: false,
            stall_shutdown: false,
            state: Rc::new(RefCell::new(MockStreamState::default())),
        }
    }

    pub(crate) fn full(input: impl Into<Vec<u8>>) -> Self {
        Self::new(input, usize::MAX)
    }

    pub(crate) fn writer(write_fragment: usize) -> Self {
        let mut stream = Self::new(Vec::new(), usize::MAX);
        stream.write_fragment = write_fragment;
        stream.delay_write_in_body = true;
        stream
    }

    pub(crate) fn stalled() -> Self {
        let mut stream = Self::new(Vec::new(), usize::MAX);
        stream.stall_after_input = true;
        stream
    }
}

impl AsyncRead for MockStream {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buffer: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        {
            let mut state = self.state.borrow_mut();
            state.read_polls += 1;
            state.largest_read = state.largest_read.max(buffer.remaining());
        }
        if let Some(delay) = self.read_delay {
            let timer = self
                .read_timer
                .get_or_insert_with(|| Box::pin(sleep(delay)));
            if timer.as_mut().poll(cx).is_pending() {
                return Poll::Pending;
            }
            self.read_timer = None;
        }
        let position = self.position;
        let max_before_delay = match &mut self.delayed_after {
            Some((after, timer)) if position >= *after => {
                if timer.as_mut().poll(cx).is_pending() {
                    return Poll::Pending;
                }
                usize::MAX
            }
            Some((after, _)) => *after - position,
            None => usize::MAX,
        };
        if self.position == self.input.len() {
            if let Some(kind) = self.fail_read_after_input {
                return Poll::Ready(Err(kind.into()));
            }
            if self.stall_after_input {
                return Poll::Pending;
            }
        }
        let count = buffer
            .remaining()
            .min(self.read_fragment)
            .min(self.input.len() - self.position)
            .min(max_before_delay);
        buffer.put_slice(&self.input[self.position..self.position + count]);
        self.position += count;
        self.state.borrow_mut().read_bytes += count;
        Poll::Ready(Ok(()))
    }
}

impl AsyncWrite for MockStream {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        bytes: &[u8],
    ) -> Poll<io::Result<usize>> {
        self.state.borrow_mut().write_polls += 1;
        if self.fail_write {
            return Poll::Ready(Err(io::ErrorKind::BrokenPipe.into()));
        }
        if self.zero_write {
            return Poll::Ready(Ok(0));
        }
        if self.stall_write {
            return Poll::Pending;
        }
        if let Some(delay) = self.write_delay {
            let should_delay = !self.delay_write_in_body
                || self
                    .state
                    .borrow()
                    .written
                    .array_windows()
                    .any(|part| part == b"\r\n\r\n");
            if should_delay {
                let timer = self
                    .write_timer
                    .get_or_insert_with(|| Box::pin(sleep(delay)));
                if timer.as_mut().poll(cx).is_pending() {
                    return Poll::Pending;
                }
                self.write_timer = None;
            }
        }
        let count = self.write_fragment.min(bytes.len());
        self.state
            .borrow_mut()
            .written
            .extend_from_slice(&bytes[..count]);
        Poll::Ready(Ok(count))
    }

    fn poll_flush(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<io::Result<()>> {
        self.state.borrow_mut().flushes += 1;
        if self.fail_flush {
            Poll::Ready(Err(io::ErrorKind::BrokenPipe.into()))
        } else if self.stall_flush {
            Poll::Pending
        } else {
            let mut state = self.state.borrow_mut();
            let count = state.written.len();
            state.flush_offsets.push(count);
            Poll::Ready(Ok(()))
        }
    }

    fn poll_shutdown(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<io::Result<()>> {
        self.state.borrow_mut().shutdowns += 1;
        if self.stall_shutdown {
            Poll::Pending
        } else {
            Poll::Ready(Ok(()))
        }
    }
}

impl Drop for MockStream {
    fn drop(&mut self) {
        self.state.borrow_mut().dropped = true;
    }
}
