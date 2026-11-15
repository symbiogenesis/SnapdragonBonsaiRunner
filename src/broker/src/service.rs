//! Explicit development-only listener integration. Nothing starts automatically.
//!
//! This module is not a launcher or qualified daily inference boundary. Backend
//! process identity, bypass prevention, OS memory/filesystem/egress restrictions,
//! and real-model behavior must be qualified before exposing daily chat.

use crate::{
    RequestPolicy,
    backend::{BackendError, BackendPolicy},
    deadline::RequestDeadline,
    scheduler::{GateError, RequestGate, Reservation},
    supervisor::{BackendConnector, ServeError, serve_chat_with_budget},
    transport::LoopbackBackend,
};
use std::{
    future::poll_fn,
    io,
    net::{Ipv4Addr, SocketAddrV4},
    pin::pin,
    sync::atomic::{AtomicBool, Ordering},
    task::Poll,
    time::Duration,
};
use subtle::ConstantTimeEq;
use tokio::{
    io::ReadBuf,
    net::{TcpListener, TcpSocket, TcpStream},
    time::{Instant, timeout_at},
};

/// Maximum time an accepted socket may wait for the single active request slot.
pub const QUEUE_TIMEOUT: Duration = Duration::from_secs(30);
/// Maximum time allowed for backend settlement and `POST /reset` after client cancellation.
pub const RESET_TIMEOUT: Duration = Duration::from_secs(15);

mod burst;
pub(crate) use burst::BurstBudget;

/// Reserve numeric IPv4 loopback with exclusive Windows address ownership.
pub(crate) fn bound_loopback_socket(port: u16) -> io::Result<TcpSocket> {
    let socket = TcpSocket::new_v4()?;
    // Windows requires this option before bind. Disabling SO_REUSEADDR alone
    // does not request exclusive ownership of the transport address.
    #[cfg(windows)]
    bonsai_windows_process::require_exclusive_address(&socket)?;
    socket.bind(SocketAddrV4::new(Ipv4Addr::LOCALHOST, port).into())?;
    Ok(socket)
}

/// Value-free service configuration, admission, transport or request error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ServiceError {
    /// Ports or credentials are invalid, shared, or unsuitable for loopback use.
    Configuration,
    /// Binding or accepting the numeric loopback listener failed.
    Io,
    /// No reservation is available, the queue expired, or shutdown was requested.
    Gate(GateError),
    /// The admitted request failed; the connection may contain a partial response.
    Request(ServeError),
}

/// Numeric-loopback listener, immutable backend and bounded admission gate.
///
/// Bind is explicit and requires a Tokio runtime with I/O and time enabled.
/// No task, process, key file, UI, retry or restart is created. The application
/// must poll accepted requests concurrently to use the bounded waiting capacity.
pub struct LocalChatService {
    listener: TcpListener,
    policy: RequestPolicy,
    backend: LoopbackBackend,
    gate: RequestGate,
    port: u16,
    capacity: usize,
    burst: BurstBudget,
    reset_on_cancel: bool,
}

impl LocalChatService {
    /// Bind only `127.0.0.1`, using distinct frontend/backend ports and keys.
    ///
    /// Port zero requests an OS-assigned frontend port; other frontend ports
    /// and the backend port must be unprivileged. Keys are never generated or
    /// logged here. The caller supplies an already running development backend.
    ///
    /// # Errors
    /// Returns a value-free [`ServiceError`] for invalid policy or bind failure.
    #[expect(
        clippy::unused_async,
        clippy::unused_async_trait_impl,
        reason = "Preserve the existing asynchronous service-binding API."
    )]
    pub async fn bind(
        port: u16,
        frontend_key: &str,
        backend_port: u16,
        backend_key: &str,
        max_waiting: usize,
    ) -> Result<Self, ServiceError> {
        if (port != 0 && port < 1024)
            || port == backend_port
            || bool::from(frontend_key.as_bytes().ct_eq(backend_key.as_bytes()))
        {
            return Err(ServiceError::Configuration);
        }
        // Validate every non-I/O input before binding a socket.
        RequestPolicy::new(1024, frontend_key).map_err(|_| ServiceError::Configuration)?;
        let backend = LoopbackBackend::new(backend_port, backend_key)
            .map_err(|_| ServiceError::Configuration)?;
        let gate = RequestGate::new(max_waiting).map_err(ServiceError::Gate)?;
        let listener = bound_loopback_socket(port)
            .and_then(|socket| socket.listen(1024))
            .map_err(|_| ServiceError::Io)?;
        let port = listener.local_addr().map_err(|_| ServiceError::Io)?.port();
        if port < 1024 || port == backend_port {
            return Err(ServiceError::Configuration);
        }
        Ok(Self {
            listener,
            policy: RequestPolicy::new(port, frontend_key)
                .map_err(|_| ServiceError::Configuration)?,
            backend,
            gate,
            port,
            capacity: max_waiting + 1,
            burst: BurstBudget::new(Instant::now()),
            reset_on_cancel: false,
        })
    }

    /// Numeric frontend port, including an OS-assigned port after binding zero.
    pub const fn port(&self) -> u16 {
        self.port
    }

    /// Actual prompt, completion and template budget of this service's model.
    pub const fn context_tokens(&self) -> usize {
        self.policy.context_tokens
    }

    /// Whether this service admits bounded multimodal image parts alongside text.
    #[must_use]
    pub const fn vision_enabled(&self) -> bool {
        self.policy.vision
    }

    /// Enable bounded multimodal image parts on this service's frontend policy.
    pub const fn enable_vision(&mut self) {
        self.policy.vision = true;
    }

    /// The owned launcher binds admission to the same context used by the child.
    #[cfg(windows)]
    pub(crate) fn select_context_tokens(
        &mut self,
        context_tokens: usize,
    ) -> Result<(), ServiceError> {
        if !(1..=bonsai_cpu_policy::context::MODEL_CONTEXT_TOKENS).contains(&context_tokens) {
            return Err(ServiceError::Configuration);
        }
        self.policy.context_tokens = context_tokens;
        Ok(())
    }

    /// Immutable total reservation bound, including the one active request.
    ///
    /// Pending accepts and accepted handles share this bound. This is not a
    /// count of currently available slots and does not change after shutdown.
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    #[cfg(windows)]
    pub(crate) const fn require_backend_process(&mut self, process_id: u32) {
        self.backend.require_process(process_id);
    }

    /// Only the owned launcher can enable the already verified pinned adapter.
    #[cfg(windows)]
    pub(crate) const fn enable_pinned_abliteration(&mut self) {
        self.policy.pinned_abliteration = true;
    }

    /// Enable in-place `POST /reset` recovery when a client cancels active generation.
    ///
    /// By default (`false`), any interrupted generation attempt permanently closes
    /// admission. Callers may enable this only when the owned backend implements
    /// bounded cancellation settlement and authenticated `POST /reset`.
    pub const fn enable_reset_on_cancel(&mut self) {
        self.reset_on_cancel = true;
    }

    async fn reset_backend(&self) -> Result<(), BackendError> {
        let deadline = Instant::now() + RESET_TIMEOUT;
        let mut shutdown = pin!(self.gate.closed());
        let mut resetting = pin!(async {
            let stream = self.backend.connect(deadline).await?;
            crate::backend::reset_backend(stream, self.backend.policy(), deadline.into()).await
        });
        poll_fn(|cx| {
            if shutdown.as_mut().poll(cx).is_ready() {
                return Poll::Ready(Err(BackendError::Io));
            }
            resetting.as_mut().poll(cx)
        })
        .await
    }

    /// Stop admission and wake queued, accepting and active operations.
    ///
    /// Polling operations observe closure and drop their owned streams. No
    /// backend-process cancellation acknowledgement or termination is implied.
    /// The listener remains bound until this service is dropped.
    pub fn close(&self) {
        self.gate.close();
    }

    /// Reserve capacity before accepting a socket or allocating request buffers.
    ///
    /// At most one active plus the configured waiting count may be reserved,
    /// including pending accept calls and returned but unpolled requests. Kernel
    /// TCP backlog is separate. A full gate returns immediately; the caller must
    /// avoid a busy retry loop. Dropping this future releases its reservation.
    ///
    /// # Errors
    /// Returns [`ServiceError`] for capacity, shutdown or socket failure.
    pub async fn accept(&self) -> Result<AcceptedChat<'_>, ServiceError> {
        let reservation = self.gate.reserve().map_err(ServiceError::Gate)?;
        let mut shutdown = pin!(self.gate.closed());
        let mut accepting = pin!(self.listener.accept());
        let (stream, peer) = poll_fn(|cx| {
            if shutdown.as_mut().poll(cx).is_ready() {
                return Poll::Ready(Err(ServiceError::Gate(GateError::Closed)));
            }
            accepting
                .as_mut()
                .poll(cx)
                .map(|result| result.map_err(|_| ServiceError::Io))
        })
        .await?;
        if !peer.ip().is_loopback() || !peer.is_ipv4() {
            return Err(ServiceError::Configuration);
        }
        stream.set_nodelay(true).map_err(|_| ServiceError::Io)?;
        let accepted = Instant::now();
        Ok(AcceptedChat {
            stream,
            reservation,
            service: self,
            accepted,
        })
    }
}

/// One reserved, owned socket. Drop closes it and releases capacity.
/// No Debug implementation: peer streams and policy are intentionally private.
pub struct AcceptedChat<'a> {
    stream: TcpStream,
    reservation: Reservation<'a>,
    service: &'a LocalChatService,
    accepted: Instant,
}

impl AcceptedChat<'_> {
    /// Wait fairly for the single active slot, then serve one owned request.
    ///
    /// Queue expiry and shutdown drop this socket without reading a body.
    /// Active generation has no total elapsed-time cap; errors never append a second
    /// HTTP response. Client half-close is treated as abandonment by the
    /// supervisor, and disconnecting closes the backend transport. Once a
    /// generation connection is attempted, any failure or caller cancellation
    /// closes the entire admission gate unless [`LocalChatService::enable_reset_on_cancel`]
    /// is enabled and an authenticated backend `POST /reset` succeeds after a
    /// client-initiated disconnect while still holding the active slot. Recreating
    /// a closed service requires a stopped or freshly owned backend first.
    ///
    /// # Errors
    /// Returns [`ServiceError`] for queue/deadline/shutdown or request failure.
    pub async fn serve(self) -> Result<(), ServiceError> {
        let Self {
            stream,
            reservation,
            service,
            accepted,
        } = self;
        let queue_deadline = accepted + QUEUE_TIMEOUT;
        if Instant::now() >= queue_deadline {
            return Err(ServiceError::Gate(GateError::Deadline));
        }
        let active = {
            let mut shutdown = pin!(service.gate.closed());
            let mut entering = pin!(reservation.enter(queue_deadline));
            let mut peek_enabled = true;
            let mut peek_buf = [0_u8; 1];
            timeout_at(
                queue_deadline,
                poll_fn(|cx| {
                    if shutdown.as_mut().poll(cx).is_ready() {
                        return Poll::Ready(Err(ServiceError::Gate(GateError::Closed)));
                    }
                    if peek_enabled {
                        let mut read_buf = ReadBuf::new(&mut peek_buf);
                        match stream.poll_peek(cx, &mut read_buf) {
                            Poll::Ready(Ok(0) | Err(_)) => {
                                return Poll::Ready(Err(ServiceError::Request(
                                    ServeError::ClientAbandoned,
                                )));
                            }
                            Poll::Ready(Ok(_)) => {
                                // Request bytes are already queued; stop peeking so we do not
                                // busy-poll while continuing to await gate admission or closure.
                                peek_enabled = false;
                            }
                            Poll::Pending => {}
                        }
                    }
                    entering
                        .as_mut()
                        .poll(cx)
                        .map(|result| result.map_err(ServiceError::Gate))
                }),
            )
            .await
            .map_err(|_| ServiceError::Gate(GateError::Deadline))??
        };
        if Instant::now() >= queue_deadline {
            return Err(ServiceError::Gate(GateError::Deadline));
        }
        active.check().map_err(ServiceError::Gate)?;
        let result = {
            let backend = GenerationAttempt {
                backend: &service.backend,
                gate: &service.gate,
                counted: AtomicBool::new(false),
                generation: AtomicBool::new(false),
                completed: AtomicBool::new(false),
            };
            let result = {
                let mut shutdown = pin!(service.gate.closed());
                let mut serving = pin!(serve_chat_with_budget(
                    stream,
                    &service.policy,
                    &backend,
                    RequestDeadline::NONE,
                    &service.burst
                ));
                poll_fn(|cx| {
                    if shutdown.as_mut().poll(cx).is_ready() {
                        return Poll::Ready(Err(ServiceError::Gate(GateError::Closed)));
                    }
                    serving
                        .as_mut()
                        .poll(cx)
                        .map(|result| result.map_err(ServiceError::Request))
                })
                .await
            };
            let client_cancelled = matches!(
                result,
                Err(ServiceError::Request(
                    ServeError::ClientAbandoned
                        | ServeError::Response(crate::response::ResponseError::ClientIo)
                ))
            );
            if result.is_ok() {
                backend.completed.store(true, Ordering::Relaxed);
            } else if service.reset_on_cancel
                && client_cancelled
                && !service.gate.is_closed()
                && backend.generation.load(Ordering::Relaxed)
                && service.reset_backend().await.is_ok()
                && !service.gate.is_closed()
            {
                backend.completed.store(true, Ordering::Relaxed);
            }
            if service.gate.is_closed() {
                Err(ServiceError::Gate(GateError::Closed))
            } else {
                result
            }
        };
        // Retain both permits through stream closure and cancellation latching.
        // The admission deadline must not expire active generation.
        drop(active);
        result
    }
}

// The supervisor makes exactly one count connection, then one generation
// connection. Mark before the second attempt, including cancellation of connect.
// Drop runs even when the caller cancels AcceptedChat::serve externally.
struct GenerationAttempt<'a> {
    backend: &'a LoopbackBackend,
    gate: &'a RequestGate,
    counted: AtomicBool,
    generation: AtomicBool,
    completed: AtomicBool,
}

impl BackendConnector for GenerationAttempt<'_> {
    type Stream = TcpStream;

    fn policy(&self) -> &BackendPolicy {
        self.backend.policy()
    }

    async fn connect(&self, deadline: Instant) -> Result<TcpStream, BackendError> {
        if self.counted.swap(true, Ordering::Relaxed) {
            self.generation.store(true, Ordering::Relaxed);
        }
        self.backend.connect(deadline).await
    }
}

impl Drop for GenerationAttempt<'_> {
    fn drop(&mut self) {
        if self.generation.load(Ordering::Relaxed) && !self.completed.load(Ordering::Relaxed) {
            self.gate.close();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    const FRONTEND_KEY: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    const BACKEND_KEY: &str = "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210";

    async fn read_http_head(stream: &mut TcpStream) -> io::Result<String> {
        let mut raw = Vec::new();
        let mut chunk = [0_u8; 512];
        let head_end = loop {
            if let Some(pos) = raw.windows(4).position(|window| window == b"\r\n\r\n") {
                break pos + 4;
            }
            let count = stream.read(&mut chunk).await?;
            if count == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "incomplete HTTP head",
                ));
            }
            raw.extend_from_slice(&chunk[..count]);
        };
        let head = String::from_utf8(raw[..head_end].to_vec()).map_err(io::Error::other)?;
        let content_length = head
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.trim()
                    .eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse::<usize>().ok())
                    .flatten()
            })
            .unwrap_or(0);
        let already = raw.len() - head_end;
        if already < content_length {
            let mut body = vec![0_u8; content_length - already];
            stream.read_exact(&mut body).await?;
        }
        Ok(head)
    }

    fn client_request(port: u16) -> String {
        let body = r#"{"model":"bonsai","messages":[{"role":"user","content":"hi"}],"stream":true,"max_completion_tokens":32}"#;
        format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nAuthorization: Bearer {FRONTEND_KEY}\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        )
    }

    #[test]
    fn reset_on_cancel_recovers_in_place_and_default_still_closes_gate() -> io::Result<()> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_io()
            .enable_time()
            .build()?;
        runtime.block_on(async {
            tokio::time::timeout(Duration::from_secs(10), async {
            // 1. Default service closes gate when active generation is abandoned.
            let backend_listener = bound_loopback_socket(0)?.listen(16)?;
            let backend_port = backend_listener.local_addr()?.port();
            let service =
                LocalChatService::bind(0, FRONTEND_KEY, backend_port, BACKEND_KEY, 1)
                    .await
                    .map_err(|_| io::Error::other("bind"))?;
            let frontend_port = service.port();
            let backend_task = tokio::spawn(async move {
                let (mut count_conn, _) = backend_listener.accept().await?;
                let _ = read_http_head(&mut count_conn).await?;
                let count_body = r#"{"object":"response.input_tokens","input_tokens":8}"#;
                count_conn
                    .write_all(
                        format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{count_body}",
                            count_body.len()
                        )
                        .as_bytes(),
                    )
                    .await?;
                drop(count_conn);

                let (mut gen_conn, _) = backend_listener.accept().await?;
                let _ = read_http_head(&mut gen_conn).await?;
                let sse_chunk = "data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n\n";
                gen_conn
                    .write_all(
                        format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n{:x}\r\n{sse_chunk}\r\n",
                            sse_chunk.len()
                        )
                        .as_bytes(),
                    )
                    .await?;
                gen_conn.flush().await?;
                let mut eof = [0_u8; 16];
                let _ = gen_conn.read(&mut eof).await?;
                Ok::<(), io::Error>(())
            });
            let client_task = tokio::spawn(async move {
                let mut client =
                    TcpStream::connect((Ipv4Addr::LOCALHOST, frontend_port)).await?;
                client
                    .write_all(client_request(frontend_port).as_bytes())
                    .await?;
                let mut buf = [0_u8; 256];
                let _ = client.read(&mut buf).await?;
                drop(client);
                Ok::<(), io::Error>(())
            });
            let accepted = service.accept().await.map_err(|_| io::Error::other("accept"))?;
            assert!(accepted.serve().await.is_err());
            client_task.await.map_err(io::Error::other)??;
            backend_task.await.map_err(io::Error::other)??;
            assert!(matches!(
                service.accept().await,
                Err(ServiceError::Gate(GateError::Closed))
            ));

            // 2. Opt-in reset_on_cancel resets the backend before admitting a queued turn.
            let backend_listener = bound_loopback_socket(0)?.listen(16)?;
            let backend_port = backend_listener.local_addr()?.port();
            let mut service =
                LocalChatService::bind(0, FRONTEND_KEY, backend_port, BACKEND_KEY, 1)
                    .await
                    .map_err(|_| io::Error::other("bind"))?;
            service.enable_reset_on_cancel();
            let frontend_port = service.port();
            let (first_streaming_tx, first_streaming_rx) = tokio::sync::oneshot::channel();
            let backend_task = tokio::spawn(async move {
                // First request count + abandoned generation.
                let (mut count_conn, _) = backend_listener.accept().await?;
                let head = read_http_head(&mut count_conn).await?;
                assert!(head.starts_with("POST /v1/chat/completions/input_tokens "));
                let count_body = r#"{"object":"response.input_tokens","input_tokens":8}"#;
                count_conn
                    .write_all(
                        format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{count_body}",
                            count_body.len()
                        )
                        .as_bytes(),
                    )
                    .await?;
                drop(count_conn);

                let (mut gen_conn, _) = backend_listener.accept().await?;
                let head = read_http_head(&mut gen_conn).await?;
                assert!(head.starts_with("POST /v1/chat/completions "));
                let sse_chunk = "data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n\n";
                gen_conn
                    .write_all(
                        format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n{:x}\r\n{sse_chunk}\r\n",
                            sse_chunk.len()
                        )
                        .as_bytes(),
                    )
                    .await?;
                gen_conn.flush().await?;
                let _ = first_streaming_tx.send(());
                let mut eof = [0_u8; 16];
                assert_eq!(gen_conn.read(&mut eof).await?, 0);
                drop(gen_conn);

                // Reset must arrive BEFORE the queued second request's count!
                let (mut reset_conn, _) = backend_listener.accept().await?;
                let reset_head = read_http_head(&mut reset_conn).await?;
                assert!(reset_head.starts_with("POST /reset HTTP/1.1\r\n"));
                assert!(!reset_head.to_ascii_lowercase().contains("content-type:"));
                reset_conn
                    .write_all(
                        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 15\r\nConnection: close\r\n\r\n{\"status\":\"ok\"}",
                    )
                    .await?;
                drop(reset_conn);

                // Queued second request count + complete generation.
                let (mut count2, _) = backend_listener.accept().await?;
                let head2 = read_http_head(&mut count2).await?;
                assert!(head2.starts_with("POST /v1/chat/completions/input_tokens "));
                count2
                    .write_all(
                        format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{count_body}",
                            count_body.len()
                        )
                        .as_bytes(),
                    )
                    .await?;
                drop(count2);

                let (mut gen2, _) = backend_listener.accept().await?;
                let head2 = read_http_head(&mut gen2).await?;
                assert!(head2.starts_with("POST /v1/chat/completions "));
                let done = "data: {\"choices\":[{\"delta\":{\"content\":\"lo\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
                gen2.write_all(
                    format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n{:x}\r\n{done}\r\n0\r\n\r\n",
                        done.len()
                    )
                    .as_bytes(),
                )
                .await?;
                drop(gen2);
                Ok::<(), io::Error>(())
            });

            let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
            let clients = tokio::spawn(async move {
                let mut client1 =
                    TcpStream::connect((Ipv4Addr::LOCALHOST, frontend_port)).await?;
                client1
                    .write_all(client_request(frontend_port).as_bytes())
                    .await?;
                let _ = first_streaming_rx.await;
                let mut partial = [0_u8; 256];
                let _ = client1.read(&mut partial).await?;

                let mut client2 =
                    TcpStream::connect((Ipv4Addr::LOCALHOST, frontend_port)).await?;
                client2
                    .write_all(client_request(frontend_port).as_bytes())
                    .await?;
                // Abandon client1 while client2 is waiting in the broker queue.
                drop(client1);

                let mut full = Vec::new();
                client2.read_to_end(&mut full).await?;
                let text = String::from_utf8(full).map_err(io::Error::other)?;
                assert!(text.contains("HTTP/1.1 200 OK"));
                assert!(text.contains("[DONE]"));
                let _ = shutdown_tx.send(());
                Ok::<(), io::Error>(())
            });

            let report = crate::runtime::run(&service, async move {
                let _ = shutdown_rx.await;
                Ok(())
            })
            .await;
            clients.await.map_err(io::Error::other)??;
            backend_task.await.map_err(io::Error::other)??;
            assert_eq!(report.exit, crate::runtime::RunExit::Shutdown);
            assert_eq!(report.completed, 1);
            assert_eq!(report.failed, 1);
            Ok::<(), io::Error>(())
            })
            .await
            .map_err(|_| io::Error::new(io::ErrorKind::TimedOut, "service test timed out"))?
        })
    }
}
