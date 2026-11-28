//! Owned, bounded HTTPS with native TLS and cancellation of blocked sockets.
#![expect(
    clippy::redundant_pub_crate,
    reason = "Startup and update modules share an explicit crate-local transport API."
)]

mod read_poll;

use std::{
    cell::RefCell,
    collections::BTreeMap,
    fs::File,
    io::{Read as _, Seek as _, SeekFrom, Write as _},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use ureq::{
    Agent, Body,
    http::{self, Uri},
    unversioned::{
        resolver::DefaultResolver,
        transport::{ConnectProxyConnector, ConnectionDetails, Connector},
    },
};

thread_local! {
    // Separate preparation workers must never cancel each other's connections.
    static AGENT: RefCell<Option<Agent>> = const { RefCell::new(None) };
}

pub(crate) struct Response {
    pub(crate) status: u16,
    pub(crate) headers: BTreeMap<String, String>,
    pub(crate) body: Vec<u8>,
}

pub(crate) fn validate_uri(uri: &str) -> crate::Result<()> {
    validate_target(uri, false)
}

fn validate_target(uri: &str, loopback: bool) -> crate::Result<()> {
    let parsed: Uri = uri.parse().map_err(|_| "Invalid HTTPS artifact URL.")?;
    let authority = parsed.authority().ok_or("Missing HTTPS authority.")?;
    let scheme_valid = parsed.scheme_str() == Some("https")
        || (loopback && parsed.scheme_str() == Some("http") && parsed.host() == Some("127.0.0.1"));
    if !scheme_valid
        || parsed.host().is_none_or(str::is_empty)
        || authority.as_str().contains(['@', '%', '\\'])
        || uri.contains('#')
        || uri.chars().any(|c| c.is_whitespace() || c.is_control())
    {
        return Err("A credential-free, unambiguous HTTPS artifact URL is required.".into());
    }
    Ok(())
}

// This small adapter is the sole use of ureq's unversioned connector API. The
// exact dependency pin and redirect regression protect this upgrade boundary.
// Pool keys include the complete authority: a credentialed redirect cannot
// bypass validation by borrowing a credential-free connection.
#[derive(Debug)]
struct CredentialFree {
    proxy: Option<Uri>,
    loopback: bool,
}

impl Connector for CredentialFree {
    type Out = ();

    fn connect(
        &self,
        details: &ConnectionDetails<'_>,
        chained: Option<()>,
    ) -> Result<Option<()>, ureq::Error> {
        if details.config.proxy().is_some_and(|proxy| {
            !matches!(
                proxy.protocol(),
                ureq::ProxyProtocol::Http | ureq::ProxyProtocol::Https
            )
        }) {
            return Err(ureq::Error::BadUri(
                "Native HTTPS supports HTTP and HTTPS CONNECT proxies only.".into(),
            ));
        }
        // CONNECT recursively invokes the chain for the explicitly configured
        // proxy. Its credentials/scheme belong to proxy policy, not artifact URLs.
        if !(self.proxy.as_ref() == Some(details.uri) && details.config.proxy().is_none()) {
            validate_target(&details.uri.to_string(), self.loopback)
                .map_err(ureq::Error::BadUri)?;
        }
        Ok(chained)
    }
}

fn new_agent(loopback: bool) -> Agent {
    let config = Agent::config_builder()
        .https_only(!loopback)
        .http_status_as_error(false)
        .max_response_header_size(64 * 1024)
        .timeout_resolve(Some(Duration::from_secs(15)))
        .timeout_connect(Some(Duration::from_secs(15)))
        .timeout_recv_response(Some(Duration::from_secs(60)))
        .timeout_per_read(Some(Duration::from_secs(60)))
        .timeout_per_write(Some(Duration::from_secs(60)))
        .user_agent(format!(
            "SnapdragonBonsaiRunner/{}",
            env!("CARGO_PKG_VERSION")
        ));
    #[cfg(windows)]
    let config = config.tls_config(
        ureq::tls::TlsConfig::builder()
            .provider(ureq::tls::TlsProvider::NativeTls)
            .root_certs(ureq::tls::RootCerts::PlatformVerifier)
            .build(),
    );
    let config = if loopback { config.proxy(None) } else { config }.build();
    let guard = CredentialFree {
        proxy: config.proxy().map(|p| p.uri().clone()),
        loopback,
    };
    // Poll raw reads below TLS: Schannel must never see the short internal
    // timeouts, and CONNECT's recursive chain must retain the same protection.
    let connector = guard
        .chain(ConnectProxyConnector::default())
        .chain(read_poll::ReadPollConnector);
    #[cfg(windows)]
    let connector = connector.chain(ureq::unversioned::transport::NativeTlsConnector::default());
    #[cfg(not(windows))]
    let connector = connector.chain(ureq::unversioned::transport::RustlsConnector::default());
    Agent::with_parts(config, connector, DefaultResolver::default())
}

struct RequestGuard {
    agent: Agent,
    abort: bool,
}

struct UnwindShutdown<'a> {
    agent: Agent,
    cancelled: &'a AtomicBool,
}

impl Drop for UnwindShutdown<'_> {
    fn drop(&mut self) {
        if thread::panicking() {
            self.cancelled.store(true, Ordering::Release);
            self.agent.clone().shutdown();
        }
    }
}

impl Drop for RequestGuard {
    fn drop(&mut self) {
        if self.abort {
            self.agent.clone().shutdown();
            AGENT.with(|cached| *cached.borrow_mut() = None);
        }
    }
}

fn remaining(until: Instant) -> crate::Result<Duration> {
    until
        .checked_duration_since(Instant::now())
        .filter(|d| !d.is_zero())
        .ok_or_else(|| "Native HTTPS operation exceeded its aggregate allowance.".into())
}

fn active(until: Instant, cancelled: &AtomicBool) -> crate::Result<()> {
    if cancelled.load(Ordering::Acquire) {
        return Err("Native HTTPS operation cancelled.".into());
    }
    remaining(until).map(|_| ())
}

fn owned<T: Send>(
    agent: Agent,
    until: Instant,
    checkpoint: &mut (impl FnMut() -> crate::Result<()> + ?Sized),
    operation: impl FnOnce(&Agent, &AtomicBool) -> crate::Result<T> + Send,
) -> crate::Result<T> {
    owned_with_abort(agent, until, checkpoint, operation, |_| false)
}

fn owned_with_abort<T: Send>(
    agent: Agent,
    until: Instant,
    checkpoint: &mut (impl FnMut() -> crate::Result<()> + ?Sized),
    operation: impl FnOnce(&Agent, &AtomicBool) -> crate::Result<T> + Send,
    should_abort: impl FnOnce(&T) -> bool,
) -> crate::Result<T> {
    checkpoint()?;
    remaining(until)?;
    let mut guard = RequestGuard { agent, abort: true };
    let cancelled = AtomicBool::new(false);
    let result = thread::scope(|scope| {
        // Drop during owner unwinding BEFORE scope joins the worker. An outer
        // guard alone would close sockets only after the implicit scope join.
        let _unwind = UnwindShutdown {
            agent: guard.agent.clone(),
            cancelled: &cancelled,
        };
        let (send, receive) = mpsc::sync_channel(1);
        let worker_agent = guard.agent.clone();
        let stopped = &cancelled;
        let worker = thread::Builder::new()
            .name("bonsai-https".into())
            .spawn_scoped(scope, move || {
                let result = operation(&worker_agent, stopped);
                let _sent = send.send(());
                result
            })
            .map_err(|e| format!("Cannot start HTTPS worker: {e}"))?;
        let observed = loop {
            if let Err(error) = checkpoint().and_then(|()| remaining(until).map(|_| ())) {
                cancelled.store(true, Ordering::Release);
                guard.agent.clone().shutdown();
                break Err(error);
            }
            match receive.recv_timeout(Duration::from_millis(50)) {
                Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => break Ok(()),
                Err(mpsc::RecvTimeoutError::Timeout) => {}
            }
        };
        let result = worker.join().map_err(|_| "Native HTTPS worker failed.")?;
        observed?;
        checkpoint()?;
        remaining(until)?;
        result
    });
    guard.abort = match &result {
        Ok(value) => should_abort(value),
        Err(_) => true,
    };
    result
}

fn agent() -> Agent {
    AGENT.with(|cached| {
        cached
            .borrow_mut()
            .get_or_insert_with(|| new_agent(false))
            .clone()
    })
}

pub(crate) const OFFLINE_PREFIX: &str = "No internet connection:";

fn is_network_offline_io(error: &std::io::Error) -> bool {
    if matches!(
        error.kind(),
        std::io::ErrorKind::NetworkUnreachable
            | std::io::ErrorKind::HostUnreachable
            | std::io::ErrorKind::NetworkDown
            | std::io::ErrorKind::AddrNotAvailable
            | std::io::ErrorKind::NotConnected
            | std::io::ErrorKind::TimedOut
            | std::io::ErrorKind::ConnectionRefused
            | std::io::ErrorKind::ConnectionAborted
            | std::io::ErrorKind::ConnectionReset
    ) {
        return true;
    }
    // Windows `getaddrinfo` failures arrive as `ErrorKind::Uncategorized` with
    // localized OS text, so match the Winsock error codes directly.
    matches!(
        error.raw_os_error(),
        Some(
            10050
                | 10051
                | 10052
                | 10053
                | 10054
                | 10057
                | 10060
                | 10061
                | 10064
                | 10065
                | 11001
                | 11002
                | 11003
                | 11004
        )
    )
}

fn is_network_offline_error(error: &ureq::Error) -> bool {
    match error {
        ureq::Error::HostNotFound
        | ureq::Error::ConnectionFailed
        | ureq::Error::ConnectProxyFailed(_)
        | ureq::Error::Timeout(_) => true,
        ureq::Error::Io(io_error) => is_network_offline_io(io_error),
        _ => false,
    }
}

pub(crate) fn is_offline_message(error: &str) -> bool {
    error.starts_with(OFFLINE_PREFIX)
        || error == "Native HTTPS operation exceeded its aggregate allowance."
}

fn read_error(error: &std::io::Error) -> String {
    if is_network_offline_io(error) {
        format!("{OFFLINE_PREFIX} Native HTTPS read failed: {error}")
    } else {
        error.to_string()
    }
}

fn format_ureq_error(error: &ureq::Error) -> String {
    if is_network_offline_error(error) {
        format!("{OFFLINE_PREFIX} Native HTTPS request failed: {error}")
    } else {
        format!("Native HTTPS request failed: {error}")
    }
}

fn call_raw(
    agent: &Agent,
    uri: &str,
    headers: &[(String, String)],
    until: Instant,
    redirects: u32,
) -> crate::Result<Result<http::Response<Body>, ureq::Error>> {
    let mut request = agent
        .get(uri)
        .config()
        .timeout_global(Some(remaining(until)?))
        .max_redirects(redirects)
        .build();
    for (name, value) in headers {
        request = request.header(name, value);
    }
    Ok(request.call())
}

fn get(
    agent: &Agent,
    uri: &str,
    headers: &[(String, String)],
    until: Instant,
    redirects: u32,
) -> crate::Result<http::Response<Body>> {
    call_raw(agent, uri, headers, until, redirects)?.map_err(|e| format_ureq_error(&e))
}

const MAX_HTTP_RETRIES: usize = 4;
const BASE_RETRY_DELAY: Duration = Duration::from_millis(250);
const MAX_RETRY_DELAY: Duration = Duration::from_secs(30);

pub(crate) fn is_retryable_http_status(status: u16) -> bool {
    status == 408 || status == 429 || (500..=599).contains(&status)
}

pub(crate) fn is_transient_io_error(error: &std::io::Error) -> bool {
    is_network_offline_io(error)
        || matches!(
            error.kind(),
            std::io::ErrorKind::UnexpectedEof
                | std::io::ErrorKind::Interrupted
                | std::io::ErrorKind::BrokenPipe
                | std::io::ErrorKind::WouldBlock
        )
}

pub(crate) fn is_transient_ureq_error(error: &ureq::Error) -> bool {
    match error {
        ureq::Error::HostNotFound
        | ureq::Error::ConnectionFailed
        | ureq::Error::ConnectProxyFailed(_)
        | ureq::Error::Timeout(_) => true,
        ureq::Error::Io(io_error) => is_transient_io_error(io_error),
        _ => false,
    }
}

fn parse_http_date(value: &str) -> Option<u64> {
    let words: Vec<_> = value.split_ascii_whitespace().collect();
    if words.len() != 6 || words[5] != "GMT" {
        return None;
    }
    let day: u64 = words[1].parse().ok()?;
    let month = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    .iter()
    .position(|month| *month == words[2])?;
    let year: u64 = words[3].parse().ok()?;
    let clock: Vec<u64> = words[4]
        .split(':')
        .map(str::parse)
        .collect::<Result<_, _>>()
        .ok()?;
    let [hour, minute, second] = clock.as_slice() else {
        return None;
    };
    if !(1970..=9999).contains(&year) || day == 0 || *hour > 23 || *minute > 59 || *second > 59 {
        return None;
    }
    let leap = |year: u64| {
        year.is_multiple_of(4) && (!year.is_multiple_of(100) || year.is_multiple_of(400))
    };
    let months = [
        31,
        if leap(year) { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    if day > months[month] {
        return None;
    }
    let leaps = |year: u64| year / 4 - year / 100 + year / 400;
    let days = (year - 1970) * 365 + leaps(year - 1) - leaps(1969)
        + months[..month].iter().sum::<u64>()
        + day
        - 1;
    Some(days * 86_400 + hour * 3600 + minute * 60 + second)
}

pub(crate) fn parse_retry_after(value: &str) -> Option<Duration> {
    let trimmed = value.trim();
    if let Ok(seconds) = trimmed.parse::<u64>() {
        return Some(Duration::from_secs(seconds));
    }
    let target = parse_http_date(trimmed)?;
    let now = SystemTime::now().duration_since(UNIX_EPOCH).ok()?.as_secs();
    Some(Duration::from_secs(target.saturating_sub(now)))
}

pub(crate) enum RetryOutcome<T> {
    Done(T),
    Retry {
        error: String,
        retry_after: Option<Duration>,
    },
    Fatal(String),
}

fn compute_retry_delay(base: Duration, attempt: usize, retry_after: Option<Duration>) -> Duration {
    let shift = u32::try_from(attempt.min(5)).unwrap_or(5);
    let exp = base.saturating_mul(1_u32 << shift);
    exp.max(retry_after.unwrap_or(Duration::ZERO))
        .min(MAX_RETRY_DELAY)
}

pub(crate) fn run_with_retry<T, C: FnMut() -> crate::Result<()> + ?Sized>(
    until: Instant,
    checkpoint: &mut C,
    attempt_fn: impl FnMut(&mut C) -> RetryOutcome<T>,
) -> crate::Result<T> {
    run_with_retry_budget(
        MAX_HTTP_RETRIES,
        BASE_RETRY_DELAY,
        until,
        checkpoint,
        attempt_fn,
    )
}

pub(crate) fn run_with_retry_budget<T, C: FnMut() -> crate::Result<()> + ?Sized>(
    max_retries: usize,
    base_delay: Duration,
    until: Instant,
    checkpoint: &mut C,
    mut attempt_fn: impl FnMut(&mut C) -> RetryOutcome<T>,
) -> crate::Result<T> {
    for attempt in 0..=max_retries {
        checkpoint()?;
        remaining(until)?;
        match attempt_fn(checkpoint) {
            RetryOutcome::Done(value) => return Ok(value),
            RetryOutcome::Fatal(error) => return Err(error),
            RetryOutcome::Retry { error, retry_after } => {
                if attempt == max_retries {
                    return Err(error);
                }
                let delay = compute_retry_delay(base_delay, attempt, retry_after);
                let rem = remaining(until)?;
                if delay >= rem {
                    return Err(error);
                }
                let wake = Instant::now()
                    .checked_add(delay)
                    .ok_or("Invalid HTTPS retry delay.")?;
                loop {
                    checkpoint()?;
                    let rem = remaining(until)?;
                    let Some(wait) = wake.checked_duration_since(Instant::now()) else {
                        break;
                    };
                    if wait.is_zero() {
                        break;
                    }
                    thread::sleep(wait.min(rem).min(Duration::from_millis(if cfg!(test) {
                        5
                    } else {
                        50
                    })));
                }
            }
        }
    }
    Err("Native HTTPS retry budget exhausted.".into())
}

fn response_headers(response: &http::Response<Body>) -> crate::Result<BTreeMap<String, String>> {
    let mut headers = BTreeMap::new();
    for (name, value) in response.headers() {
        if !matches!(
            name.as_str(),
            "content-length"
                | "content-range"
                | "content-encoding"
                | "etag"
                | "last-modified"
                | "retry-after"
                | "x-ratelimit-remaining"
                | "x-ratelimit-reset"
        ) {
            continue;
        }
        let value = value
            .to_str()
            .map_err(|_| "Invalid HTTP response header.")?;
        // Ambiguous integrity/framing headers must never be silently overwritten.
        if headers.insert(name.as_str().into(), value.into()).is_some() {
            return Err("Duplicate HTTP response header.".into());
        }
    }
    Ok(headers)
}

fn content_length(headers: &BTreeMap<String, String>) -> crate::Result<Option<u64>> {
    headers
        .get("content-length")
        .map(|v| v.parse().map_err(|_| "Invalid HTTP Content-Length.".into()))
        .transpose()
}

fn metadata_with(
    agent: &Agent,
    uri: &str,
    headers: &[(String, String)],
    until: Instant,
    maximum: u64,
    cancelled: &AtomicBool,
) -> crate::Result<Response> {
    metadata_with_redirects(agent, uri, headers, until, maximum, cancelled, 3)
}

fn metadata_with_redirects(
    agent: &Agent,
    uri: &str,
    headers: &[(String, String)],
    until: Instant,
    maximum: u64,
    cancelled: &AtomicBool,
    redirects: u32,
) -> crate::Result<Response> {
    match metadata_attempt(
        agent, uri, headers, until, maximum, cancelled, redirects, false,
    )? {
        RetryOutcome::Done(response) => Ok(response),
        RetryOutcome::Retry { error, .. } | RetryOutcome::Fatal(error) => Err(error),
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "Single metadata attempt shares framing, size-cap and cancellation checks across retried and unretried paths."
)]
fn metadata_attempt(
    agent: &Agent,
    uri: &str,
    headers: &[(String, String)],
    until: Instant,
    maximum: u64,
    cancelled: &AtomicBool,
    redirects: u32,
    retry_transient_http: bool,
) -> crate::Result<RetryOutcome<Response>> {
    let mut response = match call_raw(agent, uri, headers, until, redirects)? {
        Ok(response) => response,
        Err(error) => {
            let message = format_ureq_error(&error);
            return Ok(if is_transient_ureq_error(&error) {
                RetryOutcome::Retry {
                    error: message,
                    retry_after: None,
                }
            } else {
                RetryOutcome::Fatal(message)
            });
        }
    };
    active(until, cancelled)?;
    let status = response.status().as_u16();
    let headers = response_headers(&response)?;
    if retry_transient_http && is_retryable_http_status(status) {
        let retry_after = headers
            .get("retry-after")
            .and_then(|value| parse_retry_after(value));
        return Ok(RetryOutcome::Retry {
            error: format!("Native HTTPS request returned HTTP {status}."),
            retry_after,
        });
    }
    let mut body = Vec::new();
    // Preserve Retry-After and rate-limit headers without consuming error bodies.
    if status == 200 {
        let declared = content_length(&headers)?;
        if declared.is_some_and(|n| n > maximum) {
            return Ok(RetryOutcome::Fatal(
                "Update metadata exceeds its byte limit.".into(),
            ));
        }
        let mut reader = response.body_mut().as_reader();
        let mut buffer = [0; 16 * 1024];
        loop {
            active(until, cancelled)?;
            let count = match reader.read(&mut buffer) {
                Ok(count) => count,
                Err(error) => {
                    let message = read_error(&error);
                    return Ok(if is_transient_io_error(&error) {
                        RetryOutcome::Retry {
                            error: message,
                            retry_after: None,
                        }
                    } else {
                        RetryOutcome::Fatal(message)
                    });
                }
            };
            active(until, cancelled)?;
            if count == 0 {
                break;
            }
            let next = u64::try_from(body.len())
                .map_err(|e| e.to_string())?
                .saturating_add(u64::try_from(count).map_err(|e| e.to_string())?);
            if next > maximum {
                return Ok(RetryOutcome::Fatal(
                    "Update metadata exceeds its byte limit.".into(),
                ));
            }
            if declared.is_some_and(|expected| next > expected) {
                return Ok(RetryOutcome::Fatal(
                    "Update metadata length differs from Content-Length.".into(),
                ));
            }
            body.extend_from_slice(&buffer[..count]);
        }
        if let Some(expected) = declared
            && u64::try_from(body.len()).map_err(|e| e.to_string())? != expected
        {
            return Ok(RetryOutcome::Retry {
                error: "Update metadata response ended before Content-Length.".into(),
                retry_after: None,
            });
        }
    }
    Ok(RetryOutcome::Done(Response {
        status,
        headers,
        body,
    }))
}

pub(crate) fn metadata_with_retry(
    uri: &str,
    headers: &[(String, String)],
    until: Instant,
    maximum: u64,
    checkpoint: &mut (impl FnMut() -> crate::Result<()> + ?Sized),
) -> crate::Result<Response> {
    validate_uri(uri)?;
    run_with_retry(until, checkpoint, |checkpoint| {
        match owned_with_abort(
            agent(),
            until,
            checkpoint,
            |agent, cancelled| {
                metadata_attempt(agent, uri, headers, until, maximum, cancelled, 3, true)
            },
            |outcome| matches!(outcome, RetryOutcome::Retry { .. }),
        ) {
            Ok(outcome) => outcome,
            Err(error) => RetryOutcome::Fatal(error),
        }
    })
}

pub(crate) fn metadata(
    uri: &str,
    headers: &[(String, String)],
    timeout: Duration,
    maximum: u64,
) -> crate::Result<Response> {
    validate_uri(uri)?;
    let until = Instant::now()
        .checked_add(timeout)
        .ok_or("Invalid HTTPS allowance.")?;
    owned(
        agent(),
        until,
        &mut crate::startup_cancel::checkpoint,
        |agent, cancelled| metadata_with(agent, uri, headers, until, maximum, cancelled),
    )
}

pub(crate) fn metadata_no_redirects(
    uri: &str,
    timeout: Duration,
    maximum: u64,
) -> crate::Result<Response> {
    validate_uri(uri)?;
    let until = Instant::now()
        .checked_add(timeout)
        .ok_or("Invalid HTTPS allowance.")?;
    owned(
        agent(),
        until,
        &mut crate::startup_cancel::checkpoint,
        |agent, cancelled| metadata_with_redirects(agent, uri, &[], until, maximum, cancelled, 0),
    )
}

fn response_offset(
    status: u16,
    headers: &BTreeMap<String, String>,
    offset: u64,
    bytes: u64,
) -> crate::Result<u64> {
    if headers
        .get("content-encoding")
        .is_some_and(|v| !v.eq_ignore_ascii_case("identity"))
    {
        return Err("Pinned artifacts require identity content encoding.".into());
    }
    let start = match status {
        200 if !headers.contains_key("content-range") => 0,
        206 => {
            let expected = format!("bytes {offset}-{}/{bytes}", bytes - 1);
            if headers.get("content-range") != Some(&expected) {
                return Err("Pinned download Content-Range differs from its requested offset and locked length.".into());
            }
            offset
        }
        _ => return Err(format!("Pinned download returned HTTP {status}.")),
    };
    if content_length(headers)?.is_some_and(|n| n != bytes - start) {
        return Err("Pinned download Content-Length differs from its locked length.".into());
    }
    Ok(start)
}

fn download_with(
    agent: &Agent,
    uri: &str,
    file: &mut File,
    offset: u64,
    bytes: u64,
    until: Instant,
    cancelled: &AtomicBool,
) -> crate::Result<()> {
    download_with_mode(
        agent,
        uri,
        file,
        (offset, false),
        bytes,
        until,
        cancelled,
        None,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "Single download helper shares framing, range validation, and optional progress callbacks."
)]
fn download_with_mode(
    agent: &Agent,
    uri: &str,
    file: &mut File,
    mode: (u64, bool),
    bytes: u64,
    until: Instant,
    cancelled: &AtomicBool,
    mut on_progress: Option<&mut (dyn FnMut(u64, u64) + Send)>,
) -> crate::Result<()> {
    let (offset, fresh) = mode;
    if bytes == 0 || offset >= bytes || file.metadata().map_err(|e| e.to_string())?.len() != offset
    {
        return Err("Invalid pinned download length or changed partial offset.".into());
    }
    let mut headers = vec![("Accept-Encoding".into(), "identity".into())];
    if offset != 0 {
        headers.push(("Range".into(), format!("bytes={offset}-")));
    }
    let mut response = if fresh {
        let mut request = agent
            .get(uri)
            .config()
            .timeout_global(Some(remaining(until)?))
            .timeout_recv_response(Some(Duration::from_secs(30)))
            .timeout_per_read(Some(Duration::from_secs(30)))
            .max_redirects(10)
            .build();
        for (name, value) in &headers {
            request = request.header(name, value);
        }
        let response = request.call().map_err(|error| error.to_string())?;
        if response.status() != 200 {
            return Err("Fresh reference intake requires HTTP 200.".into());
        }
        response
    } else {
        get(agent, uri, &headers, until, 10)?
    };
    active(until, cancelled)?;
    let headers = response_headers(&response)?;
    let start = response_offset(response.status().as_u16(), &headers, offset, bytes)?;
    // Validate status, range, encoding and length before altering the partial.
    if start != offset {
        file.set_len(0).map_err(|e| e.to_string())?;
    }
    file.seek(SeekFrom::Start(start))
        .map_err(|e| e.to_string())?;
    let mut written = start;
    if let Some(progress) = on_progress.as_deref_mut() {
        progress(written, bytes);
    }
    let mut buffer = vec![0; 64 * 1024];
    let mut reader = response.body_mut().as_reader();
    loop {
        active(until, cancelled)?;
        let count = reader.read(&mut buffer).map_err(|e| read_error(&e))?;
        active(until, cancelled)?;
        if count == 0 {
            break;
        }
        let end = written
            .checked_add(u64::try_from(count).map_err(|e| e.to_string())?)
            .ok_or("Download length overflow.")?;
        if end > bytes {
            return Err("Pinned download exceeded its locked byte length.".into());
        }
        file.write_all(&buffer[..count])
            .map_err(|e| e.to_string())?;
        written = end;
        if let Some(progress) = on_progress.as_deref_mut() {
            progress(written, bytes);
        }
    }
    if written != bytes {
        return Err("Pinned download ended before its locked byte length.".into());
    }
    file.sync_all().map_err(|e| e.to_string())?;
    active(until, cancelled)
}

pub(crate) fn download(
    uri: &str,
    file: &mut File,
    offset: u64,
    bytes: u64,
    until: Instant,
    checkpoint: &mut (impl FnMut() -> crate::Result<()> + ?Sized),
) -> crate::Result<()> {
    validate_uri(uri)?;
    owned(agent(), until, checkpoint, |agent, cancelled| {
        download_with(agent, uri, file, offset, bytes, until, cancelled)
    })
}

pub(crate) fn download_with_progress(
    uri: &str,
    file: &mut File,
    offset: u64,
    bytes: u64,
    until: Instant,
    checkpoint: &mut (impl FnMut() -> crate::Result<()> + ?Sized),
    on_progress: &mut (dyn FnMut(u64, u64) + Send),
) -> crate::Result<()> {
    validate_uri(uri)?;
    owned(agent(), until, checkpoint, |agent, cancelled| {
        download_with_mode(
            agent,
            uri,
            file,
            (offset, false),
            bytes,
            until,
            cancelled,
            Some(on_progress),
        )
    })
}

/// Fresh-only public-reference intake retains its original HTTP-200 and 30-second idle bounds.
pub(crate) fn download_fresh(
    uri: &str,
    file: &mut File,
    bytes: u64,
    until: Instant,
    checkpoint: &mut (impl FnMut() -> crate::Result<()> + ?Sized),
) -> crate::Result<()> {
    validate_uri(uri)?;
    owned(agent(), until, checkpoint, |agent, cancelled| {
        download_with_mode(agent, uri, file, (0, true), bytes, until, cancelled, None)
    })
}

fn range_expected_len(start: u64, end_inclusive: u64, max_bytes: u64) -> crate::Result<u64> {
    if start > end_inclusive {
        return Err("Range start must not exceed end.".into());
    }
    let expected_len = end_inclusive
        .checked_sub(start)
        .and_then(|diff| diff.checked_add(1))
        .ok_or("Range length overflow.")?;
    if expected_len == 0 || expected_len > max_bytes {
        return Err("Range request exceeds byte limit.".into());
    }
    Ok(expected_len)
}

fn parse_content_range_total(
    headers: &BTreeMap<String, String>,
    start: u64,
    end_inclusive: u64,
) -> crate::Result<u64> {
    let header = headers
        .get("content-range")
        .ok_or("HTTP 206 response omitted Content-Range.")?;
    let expected_prefix = format!("bytes {start}-{end_inclusive}/");
    let total_str = header
        .strip_prefix(&expected_prefix)
        .ok_or("HTTP 206 Content-Range differs from requested slice.")?;
    if total_str == "*" || total_str.is_empty() || !total_str.bytes().all(|b| b.is_ascii_digit()) {
        return Err("HTTP 206 Content-Range total is invalid or unknown.".into());
    }
    let total: u64 = total_str
        .parse()
        .map_err(|_| "Invalid HTTP 206 Content-Range total.")?;
    if total <= end_inclusive {
        return Err("HTTP 206 Content-Range total does not cover requested range.".into());
    }
    Ok(total)
}

#[cfg(test)]
fn fetch_range_with(
    agent: &Agent,
    uri: &str,
    start: u64,
    end_inclusive: u64,
    max_bytes: u64,
    until: Instant,
    cancelled: &AtomicBool,
) -> crate::Result<(Vec<u8>, u64)> {
    match fetch_range_attempt(
        agent,
        uri,
        start,
        end_inclusive,
        max_bytes,
        until,
        cancelled,
    )? {
        RetryOutcome::Done(result) => Ok(result),
        RetryOutcome::Retry { error, .. } | RetryOutcome::Fatal(error) => Err(error),
    }
}

fn fetch_range_attempt(
    agent: &Agent,
    uri: &str,
    start: u64,
    end_inclusive: u64,
    max_bytes: u64,
    until: Instant,
    cancelled: &AtomicBool,
) -> crate::Result<RetryOutcome<(Vec<u8>, u64)>> {
    let expected_len = range_expected_len(start, end_inclusive, max_bytes)?;
    let headers = [
        ("Accept-Encoding".to_owned(), "identity".to_owned()),
        ("Range".to_owned(), format!("bytes={start}-{end_inclusive}")),
    ];
    let mut response = match call_raw(agent, uri, &headers, until, 10)? {
        Ok(response) => response,
        Err(error) => {
            let message = format_ureq_error(&error);
            return Ok(if is_transient_ureq_error(&error) {
                RetryOutcome::Retry {
                    error: message,
                    retry_after: None,
                }
            } else {
                RetryOutcome::Fatal(message)
            });
        }
    };
    active(until, cancelled)?;
    let status = response.status().as_u16();
    let response_headers = response_headers(&response)?;
    if response_headers
        .get("content-encoding")
        .is_some_and(|v| !v.eq_ignore_ascii_case("identity"))
    {
        return Ok(RetryOutcome::Fatal(
            "Range fetches require identity content encoding.".into(),
        ));
    }
    if is_retryable_http_status(status) {
        let retry_after = response_headers
            .get("retry-after")
            .and_then(|value| parse_retry_after(value));
        return Ok(RetryOutcome::Retry {
            error: format!("Range fetch expected HTTP 206, received HTTP {status}."),
            retry_after,
        });
    }
    if status != 206 {
        return Ok(RetryOutcome::Fatal(format!(
            "Range fetch expected HTTP 206, received HTTP {status}."
        )));
    }
    let total = parse_content_range_total(&response_headers, start, end_inclusive)?;
    if content_length(&response_headers)?.is_some_and(|n| n != expected_len) {
        return Ok(RetryOutcome::Fatal(
            "Range fetch Content-Length differs from requested slice.".into(),
        ));
    }
    let expected_usize = usize::try_from(expected_len).map_err(|e| e.to_string())?;
    let mut body = Vec::new();
    body.try_reserve_exact(expected_usize)
        .map_err(|e| format!("Range buffer allocation failed: {e}"))?;
    let mut reader = response.body_mut().as_reader();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        active(until, cancelled)?;
        let count = match reader.read(&mut buffer) {
            Ok(count) => count,
            Err(error) => {
                let message = read_error(&error);
                return Ok(if is_transient_io_error(&error) {
                    RetryOutcome::Retry {
                        error: message,
                        retry_after: None,
                    }
                } else {
                    RetryOutcome::Fatal(message)
                });
            }
        };
        active(until, cancelled)?;
        if count == 0 {
            break;
        }
        let next = u64::try_from(body.len())
            .map_err(|e| e.to_string())?
            .checked_add(u64::try_from(count).map_err(|e| e.to_string())?)
            .ok_or("Range response length overflow.")?;
        if next > expected_len {
            return Ok(RetryOutcome::Fatal(
                "Range fetch exceeded requested slice length.".into(),
            ));
        }
        body.extend_from_slice(&buffer[..count]);
    }
    if u64::try_from(body.len()).map_err(|e| e.to_string())? != expected_len {
        return Ok(RetryOutcome::Retry {
            error: "Range fetch ended before requested slice length.".into(),
            retry_after: None,
        });
    }
    active(until, cancelled)?;
    Ok(RetryOutcome::Done((body, total)))
}

/// Fetches an exact bounded byte range (`start..=end_inclusive`) with bounded retry on
/// transient transport errors and HTTP 408/429/5xx, honoring `Retry-After`.
pub(crate) fn fetch_range_with_retry(
    uri: &str,
    start: u64,
    end_inclusive: u64,
    max_bytes: u64,
    until: Instant,
    checkpoint: &mut (impl FnMut() -> crate::Result<()> + ?Sized),
) -> crate::Result<(Vec<u8>, u64)> {
    validate_uri(uri)?;
    let _ = range_expected_len(start, end_inclusive, max_bytes)?;
    run_with_retry(until, checkpoint, |checkpoint| {
        match owned_with_abort(
            agent(),
            until,
            checkpoint,
            |agent, cancelled| {
                fetch_range_attempt(
                    agent,
                    uri,
                    start,
                    end_inclusive,
                    max_bytes,
                    until,
                    cancelled,
                )
            },
            |outcome| matches!(outcome, RetryOutcome::Retry { .. }),
        ) {
            Ok(outcome) => outcome,
            Err(error) => RetryOutcome::Fatal(error),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fs::{self, OpenOptions},
        net::{TcpListener, TcpStream},
        path::PathBuf,
        sync::Arc,
    };

    struct Partial {
        path: PathBuf,
        file: File,
    }

    impl Partial {
        fn new(bytes: &[u8]) -> crate::Result<Self> {
            let path = std::env::temp_dir().join(format!(
                "bonsai-native-https-{}",
                crate::startup_process::unique()
            ));
            let mut file = OpenOptions::new()
                .create_new(true)
                .read(true)
                .write(true)
                .open(&path)
                .map_err(|e| e.to_string())?;
            file.write_all(bytes).map_err(|e| e.to_string())?;
            Ok(Self { path, file })
        }

        fn bytes(&mut self) -> crate::Result<Vec<u8>> {
            self.file
                .seek(SeekFrom::Start(0))
                .map_err(|e| e.to_string())?;
            let mut bytes = Vec::new();
            self.file
                .read_to_end(&mut bytes)
                .map_err(|e| e.to_string())?;
            Ok(bytes)
        }
    }

    impl Drop for Partial {
        fn drop(&mut self) {
            let _removed = fs::remove_file(&self.path);
        }
    }

    fn request(stream: &mut TcpStream) -> crate::Result<String> {
        stream.set_nonblocking(false).map_err(|e| e.to_string())?;
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .map_err(|e| e.to_string())?;
        stream
            .set_write_timeout(Some(Duration::from_secs(5)))
            .map_err(|e| e.to_string())?;
        let mut request = Vec::new();
        while !request.ends_with(b"\r\n\r\n") && request.len() < 8192 {
            let mut byte = [0];
            if stream.read(&mut byte).map_err(|e| e.to_string())? == 0 {
                return Err("Fixture request truncated.".into());
            }
            request.push(byte[0]);
        }
        if !request.ends_with(b"\r\n\r\n") {
            return Err("Fixture request exceeded its bound.".into());
        }
        String::from_utf8(request).map_err(|e| e.to_string())
    }

    fn fixture<T>(
        response: &[u8],
        operation: impl FnOnce(Agent, &str) -> crate::Result<T>,
    ) -> crate::Result<(T, String)> {
        let listener = TcpListener::bind("127.0.0.1:0").map_err(|e| e.to_string())?;
        listener.set_nonblocking(true).map_err(|e| e.to_string())?;
        let uri = format!(
            "http://{}/artifact",
            listener.local_addr().map_err(|e| e.to_string())?
        );
        thread::scope(|scope| {
            let server = scope.spawn(move || {
                let until = Instant::now() + Duration::from_secs(5);
                loop {
                    match listener.accept() {
                        Ok((mut stream, _)) => {
                            let request = request(&mut stream)?;
                            stream.write_all(response).map_err(|e| e.to_string())?;
                            return Ok::<_, String>(request);
                        }
                        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                            remaining(until)?;
                            thread::sleep(Duration::from_millis(1));
                        }
                        Err(e) => return Err(e.to_string()),
                    }
                }
            });
            let result = operation(new_agent(true), &uri);
            let request = server.join().map_err(|_| "HTTPS fixture panicked.")??;
            Ok((result?, request))
        })
    }

    #[test]
    fn native_https_rejects_ambiguous_urls_and_mismatched_ranges() {
        for uri in [
            "http://example.com/a",
            "https://user:pass@example.com/a",
            "https://example.com/a b",
            "https://ex%61mple.com/a",
            "https://",
            "https://example.com/a#b",
        ] {
            assert!(validate_uri(uri).is_err());
        }
        assert!(validate_uri("https://example.com/a?token=123").is_ok());
        let mut headers = BTreeMap::from([
            ("content-range".into(), "bytes 5-9/10".into()),
            ("content-length".into(), "5".into()),
        ]);
        assert_eq!(response_offset(206, &headers, 5, 10), Ok(5));
        for bad in [
            "bytes 0-4/10",
            "bytes 5-9/*",
            "bytes 5-9/11",
            "bytes 5-8/10",
        ] {
            headers.insert("content-range".into(), bad.into());
            assert!(response_offset(206, &headers, 5, 10).is_err());
        }
        headers.remove("content-range");
        assert!(response_offset(200, &headers, 5, 10).is_err());
        headers.insert("content-length".into(), "10".into());
        assert_eq!(response_offset(200, &headers, 5, 10), Ok(0));
        headers.insert("content-encoding".into(), "gzip".into());
        assert!(response_offset(200, &headers, 5, 10).is_err());
    }

    #[test]
    fn native_https_loopback_resumes_restarts_and_retains_bounded_partials() -> crate::Result<()> {
        for (response, expected, success) in [
            (b"HTTP/1.1 206 Partial Content\r\nContent-Length: 5\r\nContent-Range: bytes 5-9/10\r\nConnection: close\r\n\r\nworld".as_slice(), b"helloworld".as_slice(), true),
            (b"HTTP/1.1 200 OK\r\nContent-Length: 10\r\nSet-Cookie: a=1\r\nSet-Cookie: b=2\r\nConnection: close\r\n\r\n0123456789", b"0123456789", true),
            (b"HTTP/1.1 206 Partial Content\r\nContent-Length: 5\r\nContent-Range: bytes 4-8/10\r\nConnection: close\r\n\r\nworld", b"hello", false),
            (b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\nB\r\n01234567890\r\n0\r\n\r\n", b"", false),
            (b"HTTP/1.1 206 Partial Content\r\nContent-Length: 5\r\nContent-Range: bytes 5-9/10\r\nConnection: close\r\n\r\nwo", b"hellowo", false),
        ] {
            let mut partial = Partial::new(b"hello")?;
            let (result, request) = fixture(response, |agent, uri| {
                Ok(download_with(&agent, uri, &mut partial.file, 5, 10, Instant::now() + Duration::from_secs(5), &AtomicBool::new(false)))
            })?;
            assert_eq!(result.is_ok(), success, "{result:?}");
            assert!(request.to_ascii_lowercase().contains("range: bytes=5-"));
            assert_eq!(partial.bytes()?, expected);
        }
        Ok(())
    }

    #[test]
    fn fresh_reference_download_rejects_partial_status_before_writing() -> crate::Result<()> {
        for (response,success) in [
            (b"HTTP/1.1 200 OK\r\nContent-Length: 4\r\nConnection: close\r\n\r\ndata".as_slice(),true),
            (b"HTTP/1.1 206 Partial Content\r\nContent-Length: 4\r\nContent-Range: bytes 0-3/4\r\nConnection: close\r\n\r\ndata".as_slice(),false),
        ] {
            let mut partial=Partial::new(b"")?;
            let (result,_) = fixture(response,|agent,uri|Ok(download_with_mode(&agent,uri,&mut partial.file,(0,true),4,Instant::now()+Duration::from_secs(5),&AtomicBool::new(false),None)))?;
            assert_eq!(result.is_ok(),success);
            assert_eq!(partial.bytes()?,if success{b"data".as_slice()}else{b"".as_slice()});
        }
        Ok(())
    }

    #[test]
    fn native_https_metadata_bounds_status_and_redirect_credentials() -> crate::Result<()> {
        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
        encoder.write_all(b"metadata").map_err(|e| e.to_string())?;
        let compressed = encoder.finish().map_err(|e| e.to_string())?;
        let mut raw = format!("HTTP/1.1 200 OK\r\nContent-Encoding: gzip\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", compressed.len()).into_bytes();
        raw.extend_from_slice(&compressed);
        let (response, request) = fixture(&raw, |agent, uri| {
            metadata_with(
                &agent,
                uri,
                &[],
                Instant::now() + Duration::from_secs(5),
                1024,
                &AtomicBool::new(false),
            )
        })?;
        assert!(request.to_ascii_lowercase().contains(&format!(
            "user-agent: snapdragonbonsairunner/{}",
            env!("CARGO_PKG_VERSION")
        )));
        assert_eq!(response.body, compressed);
        assert_eq!(
            response.headers.get("content-encoding").map(String::as_str),
            Some("gzip")
        );
        let (response, _) = fixture(b"HTTP/1.1 429 Too Many Requests\r\nRetry-After: 90\r\nContent-Length: 999999\r\nConnection: close\r\n\r\n", |agent, uri| {
            metadata_with(&agent, uri, &[], Instant::now() + Duration::from_secs(5), 4, &AtomicBool::new(false))
        })?;
        assert_eq!(response.status, 429);
        assert_eq!(
            response.headers.get("retry-after").map(String::as_str),
            Some("90")
        );
        assert_eq!(response.body, Vec::<u8>::new());
        let (response, _) = fixture(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n5\r\nhello\r\n0\r\n\r\n", |agent, uri| {
            Ok(metadata_with(&agent, uri, &[], Instant::now() + Duration::from_secs(5), 4, &AtomicBool::new(false)))
        })?;
        assert!(response.is_err());
        let (response, _) = fixture(b"HTTP/1.1 302 Found\r\nLocation: http://user:secret@127.0.0.1:9/steal\r\nContent-Length: 0\r\nConnection: close\r\n\r\n", |agent, uri| {
            Ok(metadata_with(&agent, uri, &[], Instant::now() + Duration::from_secs(5), 4, &AtomicBool::new(false)))
        })?;
        assert!(
            response
                .err()
                .is_some_and(|e| e.contains("credential-free"))
        );
        let (response, _) = fixture(
            b"HTTP/1.1 200 OK\r\nContent-Length: 10\r\nConnection: close\r\n\r\nhello",
            |agent, uri| {
                Ok(metadata_with(
                    &agent,
                    uri,
                    &[],
                    Instant::now() + Duration::from_secs(5),
                    1024,
                    &AtomicBool::new(false),
                ))
            },
        )?;
        assert!(response.is_err());
        Ok(())
    }

    #[test]
    fn native_https_rejects_duplicate_integrity_headers_before_restarting_partial()
    -> crate::Result<()> {
        for header in [
            "Content-Range: bytes 5-9/10",
            "Content-Encoding: identity",
            "ETag: \"version\"",
            "Last-Modified: Sat, 14 Nov 2026 00:00:00 GMT",
        ] {
            // Identical repetitions are still ambiguous. A 200 response would
            // otherwise discard the existing prefix before downloading afresh.
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: 10\r\n{header}\r\n{header}\r\nConnection: close\r\n\r\n0123456789"
            );
            let mut partial = Partial::new(b"hello")?;
            let (result, _) = fixture(response.as_bytes(), |agent, uri| {
                Ok(download_with(
                    &agent,
                    uri,
                    &mut partial.file,
                    5,
                    10,
                    Instant::now() + Duration::from_secs(5),
                    &AtomicBool::new(false),
                ))
            })?;
            assert_eq!(result, Err("Duplicate HTTP response header.".into()));
            assert_eq!(partial.bytes()?, b"hello");
        }
        Ok(())
    }

    #[test]
    fn native_https_metadata_checks_declared_length_before_reading_body() -> crate::Result<()> {
        let (result, _) = fixture(
            b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\nConnection: close\r\n\r\n",
            |agent, uri| {
                Ok(metadata_with(
                    &agent,
                    uri,
                    &[],
                    Instant::now() + Duration::from_secs(5),
                    4,
                    &AtomicBool::new(false),
                ))
            },
        )?;
        // The response deliberately contains no body: an attempted body read
        // would produce truncation instead of the declared-length rejection.
        assert!(matches!(result, Err(error) if error == "Update metadata exceeds its byte limit."));
        let (response, _) = fixture(
            b"HTTP/1.1 200 OK\r\nContent-Length: 4\r\nConnection: close\r\n\r\nokay",
            |agent, uri| {
                metadata_with(
                    &agent,
                    uri,
                    &[],
                    Instant::now() + Duration::from_secs(5),
                    4,
                    &AtomicBool::new(false),
                )
            },
        )?;
        assert_eq!(response.body, b"okay");
        Ok(())
    }

    #[test]
    fn native_https_expired_allowance_never_starts_worker() {
        let started = AtomicBool::new(false);
        let result = owned(new_agent(true), Instant::now(), &mut || Ok(()), |_, _| {
            started.store(true, Ordering::Release);
            Ok(())
        });
        assert_eq!(
            result,
            Err("Native HTTPS operation exceeded its aggregate allowance.".into())
        );
        assert!(!started.load(Ordering::Acquire));
    }

    #[derive(Clone, Copy)]
    enum Stop {
        Cancel,
        Panic,
        Deadline,
    }

    #[test]
    fn native_https_cancellation_shuts_down_stalled_body_and_joins_worker() -> crate::Result<()> {
        for stop in [Stop::Cancel, Stop::Panic] {
            stalled_body(stop)?;
        }
        Ok(())
    }

    #[test]
    fn native_https_deadline_shuts_down_stalled_body_and_joins_worker() -> crate::Result<()> {
        stalled_body(Stop::Deadline)
    }

    fn cached_metadata(
        uri: &str,
        stop: &AtomicBool,
        timeout: Duration,
    ) -> (crate::Result<Response>, bool) {
        // Use the production per-owner cache with loopback-only fixture transport.
        AGENT.with(|cached| *cached.borrow_mut() = Some(new_agent(true)));
        let until = Instant::now() + timeout;
        let response = owned(
            agent(),
            until,
            &mut || {
                if stop.load(Ordering::Acquire) {
                    Err("Synthetic owner cancellation.".into())
                } else {
                    Ok(())
                }
            },
            |agent, cancelled| metadata_with(agent, uri, &[], until, 5, cancelled),
        );
        (response, AGENT.with(|cached| cached.borrow().is_some()))
    }

    fn pending_metadata(listener: &TcpListener) -> crate::Result<TcpStream> {
        listener.set_nonblocking(true).map_err(|e| e.to_string())?;
        let until = Instant::now() + Duration::from_secs(5);
        let mut stream = loop {
            match listener.accept() {
                Ok((stream, _)) => break stream,
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    remaining(until)?;
                    thread::sleep(Duration::from_millis(1));
                }
                Err(e) => return Err(e.to_string()),
            }
        };
        request(&mut stream)?;
        stream
            .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\n")
            .map_err(|e| e.to_string())?;
        Ok(stream)
    }

    #[test]
    fn native_https_cancelled_cached_owner_preserves_peer_connection() -> crate::Result<()> {
        let cancelled_listener = TcpListener::bind("127.0.0.1:0").map_err(|e| e.to_string())?;
        let peer_listener = TcpListener::bind("127.0.0.1:0").map_err(|e| e.to_string())?;
        let cancelled_uri = format!(
            "http://{}/cancelled",
            cancelled_listener.local_addr().map_err(|e| e.to_string())?
        );
        let peer_uri = format!(
            "http://{}/peer",
            peer_listener.local_addr().map_err(|e| e.to_string())?
        );
        let stop = AtomicBool::new(false);
        let peer_stop = AtomicBool::new(false);
        thread::scope(|scope| {
            let cancelled_owner =
                scope.spawn(|| cached_metadata(&cancelled_uri, &stop, Duration::from_secs(1)));
            let peer_owner =
                scope.spawn(|| cached_metadata(&peer_uri, &peer_stop, Duration::from_secs(5)));
            let mut cancelled_socket = pending_metadata(&cancelled_listener)
                .map_err(|e| format!("Cancelled owner fixture: {e}"))?;
            let mut peer_socket =
                pending_metadata(&peer_listener).map_err(|e| format!("Peer owner fixture: {e}"))?;
            // Both owners now have a live response body. Keep the peer blocked
            // until cancellation has shut down and joined the other owner.
            stop.store(true, Ordering::Release);
            let (cancelled, retained) = cancelled_owner
                .join()
                .map_err(|_| "Cancelled HTTPS owner panicked.")?;
            assert!(matches!(cancelled, Err(error) if error == "Synthetic owner cancellation."));
            assert!(!retained, "An aborted owner retained its shut-down agent.");
            let mut byte = [0];
            match cancelled_socket.read(&mut byte) {
                Ok(0) => {}
                Err(e) if e.kind() == std::io::ErrorKind::ConnectionReset => {}
                result => return Err(format!("Cancelled owner retained its socket: {result:?}")),
            }
            peer_socket.write_all(b"alive").map_err(|e| e.to_string())?;
            let (peer, retained) = peer_owner
                .join()
                .map_err(|_| "Peer HTTPS owner panicked.")?;
            assert_eq!(peer?.body, b"alive");
            assert!(retained, "Successful peer lost its reusable agent.");
            Ok(())
        })
    }

    fn stalled_body(stop: Stop) -> crate::Result<()> {
        let listener = TcpListener::bind("127.0.0.1:0").map_err(|e| e.to_string())?;
        let uri = format!(
            "http://{}/artifact",
            listener.local_addr().map_err(|e| e.to_string())?
        );
        listener.set_nonblocking(true).map_err(|e| e.to_string())?;
        let headers_sent = Arc::new(AtomicBool::new(false));
        thread::scope(|scope| {
            let started = Arc::clone(&headers_sent);
            let server = scope.spawn(move || {
                let until = Instant::now() + Duration::from_secs(5);
                let mut stream = loop {
                    match listener.accept() {
                        Ok((stream, _)) => break stream,
                        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                            remaining(until)?;
                            thread::sleep(Duration::from_millis(1));
                        }
                        Err(e) => return Err(e.to_string()),
                    }
                };
                request(&mut stream)?;
                stream
                    .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\n")
                    .map_err(|e| e.to_string())?;
                started.store(true, Ordering::Release);
                let mut byte = [0];
                match stream.read(&mut byte) {
                    Ok(0) => Ok(()),
                    Err(e) if e.kind() == std::io::ErrorKind::ConnectionReset => Ok(()),
                    _ => Err("Cancelled native HTTPS worker did not release its socket.".into()),
                }
            });
            let started = Instant::now();
            let until = started
                + if matches!(stop, Stop::Deadline) {
                    Duration::from_millis(150)
                } else {
                    Duration::from_secs(5)
                };
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                owned(
                    new_agent(true),
                    until,
                    &mut || {
                        if headers_sent.load(Ordering::Acquire) && !matches!(stop, Stop::Deadline) {
                            if matches!(stop, Stop::Panic) {
                                std::panic::resume_unwind(Box::new("Synthetic caller panic."));
                            }
                            Err("Synthetic caller cancellation.".into())
                        } else {
                            Ok(())
                        }
                    },
                    |agent, cancelled| metadata_with(agent, &uri, &[], until, 5, cancelled),
                )
            }));
            // Joining at the request deadline used to satisfy the socket/exit
            // checks despite cancellation doing no useful work. Require the
            // cancelled and unwinding owners to finish before that deadline.
            assert!(started.elapsed() < Duration::from_secs(2));
            let released = server
                .join()
                .map_err(|_| "Stalled HTTPS fixture panicked.")?;
            assert!(headers_sent.load(Ordering::Acquire));
            if matches!(stop, Stop::Panic) {
                assert!(result.is_err());
            } else {
                let expected = if matches!(stop, Stop::Deadline) {
                    "Native HTTPS operation exceeded its aggregate allowance."
                } else {
                    "Synthetic caller cancellation."
                };
                assert!(
                    result
                        .map_err(|_| "Unexpected owner panic.")?
                        .err()
                        .is_some_and(|e| e == expected)
                );
            }
            released
        })
    }

    #[test]
    fn native_https_fetch_range_validates_partial_content_and_bounds() -> crate::Result<()> {
        assert!(range_expected_len(10, 5, 100).is_err());
        assert!(range_expected_len(0, 100, 100).is_err());

        let (result, request) = fixture(
            b"HTTP/1.1 206 Partial Content\r\nContent-Length: 5\r\nContent-Range: bytes 2-6/20\r\nConnection: close\r\n\r\nworld",
            |agent, uri| {
                fetch_range_with(
                    &agent,
                    uri,
                    2,
                    6,
                    16,
                    Instant::now() + Duration::from_secs(5),
                    &AtomicBool::new(false),
                )
            },
        )?;
        assert_eq!(result, (b"world".to_vec(), 20));
        let lower = request.to_ascii_lowercase();
        assert!(lower.contains("range: bytes=2-6\r\n"));
        assert!(lower.contains("accept-encoding: identity\r\n"));

        for response in [
            b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\nConnection: close\r\n\r\nworld".as_slice(),
            b"HTTP/1.1 206 Partial Content\r\nContent-Length: 5\r\nContent-Range: bytes 0-4/20\r\nConnection: close\r\n\r\nworld",
            b"HTTP/1.1 206 Partial Content\r\nContent-Length: 5\r\nContent-Range: bytes 2-6/*\r\nConnection: close\r\n\r\nworld",
            b"HTTP/1.1 206 Partial Content\r\nContent-Length: 5\r\nContent-Range: bytes 2-6/6\r\nConnection: close\r\n\r\nworld",
            b"HTTP/1.1 206 Partial Content\r\nContent-Encoding: gzip\r\nContent-Length: 5\r\nContent-Range: bytes 2-6/20\r\nConnection: close\r\n\r\nworld",
            b"HTTP/1.1 206 Partial Content\r\nContent-Length: 5\r\nContent-Range: bytes 2-6/20\r\nConnection: close\r\n\r\nwo",
            b"HTTP/1.1 206 Partial Content\r\nContent-Length: 4\r\nContent-Range: bytes 2-6/20\r\nConnection: close\r\n\r\nworl",
        ] {
            let (err, _) = fixture(response, |agent, uri| {
                Ok(fetch_range_with(
                    &agent,
                    uri,
                    2,
                    6,
                    16,
                    Instant::now() + Duration::from_secs(5),
                    &AtomicBool::new(false),
                ))
            })?;
            assert!(err.is_err(), "Expected failure for range response");
        }
        Ok(())
    }

    #[test]
    fn offline_network_errors_are_classified_without_masking_http_or_tls_failures() {
        assert!(is_network_offline_error(&ureq::Error::HostNotFound));
        assert!(is_network_offline_error(&ureq::Error::ConnectionFailed));
        assert!(is_network_offline_error(&ureq::Error::ConnectProxyFailed(
            "proxy unreachable".into()
        )));
        assert!(is_network_offline_error(&ureq::Error::Timeout(
            ureq::Timeout::Global
        )));
        for code in [11001, 10051, 10065] {
            let io_error = std::io::Error::from_raw_os_error(code);
            assert!(is_network_offline_io(&io_error));
            assert!(is_network_offline_error(&ureq::Error::Io(io_error)));
        }
        assert!(is_network_offline_io(&std::io::Error::new(
            std::io::ErrorKind::NetworkUnreachable,
            "unreachable"
        )));
        assert!(!is_network_offline_io(&std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "corrupt payload"
        )));
        assert!(!is_network_offline_error(&ureq::Error::StatusCode(403)));
        assert!(!is_network_offline_error(&ureq::Error::BadUri(
            "bad".into()
        )));
        assert!(!is_network_offline_error(&ureq::Error::Tls(
            "handshake failed"
        )));

        assert!(is_offline_message(
            "No internet connection: Native HTTPS request failed: host not found"
        ));
        assert!(is_offline_message(
            "Native HTTPS operation exceeded its aggregate allowance."
        ));
        assert!(!is_offline_message(
            "Native HTTPS request failed: tls handshake failed"
        ));
        assert!(!is_offline_message("Native HTTPS operation cancelled."));
        assert!(!is_offline_message(
            "GitHub update check failed (HTTP 403). Your installed app is unchanged."
        ));
    }

    fn sequence_server(
        responses: Vec<&'static [u8]>,
    ) -> crate::Result<(String, thread::JoinHandle<crate::Result<Vec<String>>>)> {
        let listener = TcpListener::bind("127.0.0.1:0").map_err(|e| e.to_string())?;
        let uri = format!(
            "http://{}/artifact",
            listener.local_addr().map_err(|e| e.to_string())?
        );
        let handle = thread::spawn(move || -> crate::Result<Vec<String>> {
            let mut requests = Vec::with_capacity(responses.len());
            for payload in responses {
                let (mut stream, _) = listener.accept().map_err(|e| e.to_string())?;
                requests.push(request(&mut stream)?);
                stream.write_all(payload).map_err(|e| e.to_string())?;
            }
            Ok(requests)
        });
        Ok((uri, handle))
    }

    #[test]
    fn retry_after_and_exponential_backoff_are_bounded_and_honored() {
        assert_eq!(parse_retry_after("0"), Some(Duration::from_secs(0)));
        assert_eq!(parse_retry_after(" 12 "), Some(Duration::from_secs(12)));
        assert_eq!(
            parse_retry_after("Thu, 01 Jan 1970 00:00:01 GMT"),
            Some(Duration::from_secs(0))
        );
        assert_eq!(parse_retry_after("invalid-header"), None);

        assert_eq!(
            compute_retry_delay(Duration::from_millis(250), 0, None),
            Duration::from_millis(250)
        );
        assert_eq!(
            compute_retry_delay(Duration::from_millis(250), 2, None),
            Duration::from_secs(1)
        );
        assert_eq!(
            compute_retry_delay(Duration::from_millis(250), 0, Some(Duration::from_secs(5))),
            Duration::from_secs(5)
        );
        assert_eq!(
            compute_retry_delay(
                Duration::from_millis(250),
                0,
                Some(Duration::from_secs(120))
            ),
            MAX_RETRY_DELAY
        );

        let mut attempts = 0usize;
        let err = run_with_retry_budget(
            3,
            Duration::from_millis(1),
            Instant::now() + Duration::from_secs(2),
            &mut || Ok(()),
            |_| -> RetryOutcome<()> {
                attempts += 1;
                RetryOutcome::Retry {
                    error: "too many requests".into(),
                    retry_after: Some(Duration::from_secs(10)),
                }
            },
        );
        assert_eq!(attempts, 1);
        assert_eq!(err.err().as_deref(), Some("too many requests"));

        let mut fatal_attempts = 0usize;
        let fatal = run_with_retry_budget(
            3,
            Duration::from_millis(1),
            Instant::now() + Duration::from_secs(2),
            &mut || Ok(()),
            |_| -> RetryOutcome<()> {
                fatal_attempts += 1;
                RetryOutcome::Fatal("bad status 404".into())
            },
        );
        assert_eq!(fatal_attempts, 1);
        assert_eq!(fatal.err().as_deref(), Some("bad status 404"));
    }

    #[test]
    fn metadata_and_range_retry_transient_http_and_truncation_without_retrying_404()
    -> crate::Result<()> {
        let (uri, server) = sequence_server(vec![
            b"HTTP/1.1 429 Too Many Requests\r\nRetry-After: 0\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            b"HTTP/1.1 206 Partial Content\r\nContent-Length: 5\r\nContent-Range: bytes 2-6/20\r\nConnection: close\r\n\r\nwo",
            b"HTTP/1.1 206 Partial Content\r\nContent-Length: 5\r\nContent-Range: bytes 2-6/20\r\nConnection: close\r\n\r\nworld",
        ])?;
        let until = Instant::now() + Duration::from_secs(10);
        let (bytes, total) = run_with_retry_budget(
            4,
            Duration::from_millis(1),
            until,
            &mut || Ok(()),
            |checkpoint| match owned_with_abort(
                new_agent(true),
                until,
                checkpoint,
                |agent, cancelled| fetch_range_attempt(agent, &uri, 2, 6, 16, until, cancelled),
                |outcome| matches!(outcome, RetryOutcome::Retry { .. }),
            ) {
                Ok(outcome) => outcome,
                Err(error) => RetryOutcome::Fatal(error),
            },
        )?;
        assert_eq!(bytes, b"world");
        assert_eq!(total, 20);
        assert_eq!(
            server
                .join()
                .map_err(|_| "Sequence server panicked.")??
                .len(),
            4
        );

        let (not_found_uri, not_found_server) = sequence_server(vec![
            b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
        ])?;
        let not_found = run_with_retry_budget(
            4,
            Duration::from_millis(1),
            until,
            &mut || Ok(()),
            |checkpoint| match owned_with_abort(
                new_agent(true),
                until,
                checkpoint,
                |agent, cancelled| {
                    fetch_range_attempt(agent, &not_found_uri, 0, 3, 16, until, cancelled)
                },
                |outcome| matches!(outcome, RetryOutcome::Retry { .. }),
            ) {
                Ok(outcome) => outcome,
                Err(error) => RetryOutcome::Fatal(error),
            },
        );
        assert!(not_found.is_err());
        assert_eq!(
            not_found_server
                .join()
                .map_err(|_| "Sequence server panicked.")??
                .len(),
            1
        );

        // metadata_attempt retries 429 and truncated 200, succeeds on valid 200,
        // and rejects oversize & 404 fatally without retrying.
        let (meta_uri, meta_server) = sequence_server(vec![
            b"HTTP/1.1 429 Too Many Requests\r\nRetry-After: 0\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            b"HTTP/1.1 200 OK\r\nContent-Length: 10\r\nConnection: close\r\n\r\nab",
            b"HTTP/1.1 200 OK\r\nContent-Length: 8\r\nConnection: close\r\n\r\n{\"ok\":1}",
        ])?;
        let meta_res = run_with_retry_budget(
            4,
            Duration::from_millis(1),
            until,
            &mut || Ok(()),
            |checkpoint| match owned_with_abort(
                new_agent(true),
                until,
                checkpoint,
                |agent, cancelled| {
                    metadata_attempt(agent, &meta_uri, &[], until, 1024, cancelled, 0, true)
                },
                |outcome| matches!(outcome, RetryOutcome::Retry { .. }),
            ) {
                Ok(outcome) => outcome,
                Err(error) => RetryOutcome::Fatal(error),
            },
        )?;
        assert_eq!(meta_res.body, b"{\"ok\":1}");
        assert_eq!(
            meta_server
                .join()
                .map_err(|_| "Meta server panicked.")??
                .len(),
            3
        );

        let (over_uri, over_server) = sequence_server(vec![
            b"HTTP/1.1 200 OK\r\nContent-Length: 2048\r\nConnection: close\r\n\r\n",
        ])?;
        let over_res = run_with_retry_budget(
            3,
            Duration::from_millis(1),
            until,
            &mut || Ok(()),
            |checkpoint| match owned_with_abort(
                new_agent(true),
                until,
                checkpoint,
                |agent, cancelled| {
                    metadata_attempt(agent, &over_uri, &[], until, 1024, cancelled, 0, true)
                },
                |outcome| matches!(outcome, RetryOutcome::Retry { .. }),
            ) {
                Ok(outcome) => outcome,
                Err(error) => RetryOutcome::Fatal(error),
            },
        );
        assert!(over_res.is_err());
        assert_eq!(
            over_server
                .join()
                .map_err(|_| "Oversize server panicked.")??
                .len(),
            1
        );

        // is_retryable_http_status, is_transient_io_error, parse_http_date, and retry cancellation/exhaustion
        for s in [408_u16, 429, 500, 599] {
            assert!(is_retryable_http_status(s));
        }
        for s in [200_u16, 400, 403, 404, 499, 600] {
            assert!(!is_retryable_http_status(s));
        }
        for kind in [
            std::io::ErrorKind::UnexpectedEof,
            std::io::ErrorKind::Interrupted,
            std::io::ErrorKind::BrokenPipe,
            std::io::ErrorKind::WouldBlock,
        ] {
            assert!(is_transient_io_error(&std::io::Error::from(kind)));
        }
        assert!(!is_transient_io_error(&std::io::Error::from(
            std::io::ErrorKind::InvalidData
        )));

        assert!(parse_http_date("Thu, 29 Feb 2024 12:34:56 GMT").is_some());
        assert!(parse_http_date("Sat, 29 Feb 2025 12:34:56 GMT").is_none());
        assert!(parse_http_date("Mon, 29 Feb 2100 12:34:56 GMT").is_none());
        assert!(parse_http_date("Tue, 29 Feb 2000 12:34:56 GMT").is_some());
        assert!(parse_http_date("Thu, 01 Foo 2024 12:34:56 GMT").is_none());
        assert!(parse_http_date("Thu, 01 Jan 2024 25:00:00 GMT").is_none());
        assert!(parse_http_date("Thu, 01 Jan 2024 12:34:56 UTC").is_none());
        assert!(parse_retry_after("Wed, 01 Jan 2099 00:00:00 GMT").is_some_and(|d| !d.is_zero()));

        // run_with_retry_budget cancellation during chunked 50ms sleep wakes promptly
        let mut sleep_checks = 0_usize;
        let started_cancel = Instant::now();
        let cancelled_res = run_with_retry_budget(
            3,
            Duration::from_millis(50),
            Instant::now() + Duration::from_secs(10),
            &mut || {
                sleep_checks += 1;
                if sleep_checks >= 3 {
                    Err("user cancelled".to_owned())
                } else {
                    Ok(())
                }
            },
            |_| -> RetryOutcome<()> {
                RetryOutcome::Retry {
                    error: "transient".into(),
                    retry_after: Some(Duration::from_millis(200)),
                }
            },
        );
        assert_eq!(cancelled_res.err().as_deref(), Some("user cancelled"));
        assert!(started_cancel.elapsed() < Duration::from_millis(500));

        // Exhaustion runs max_retries + 1 attempts
        let mut exh = 0_usize;
        let exh_res = run_with_retry_budget(
            2,
            Duration::from_millis(1),
            Instant::now() + Duration::from_secs(5),
            &mut || Ok(()),
            |_| -> RetryOutcome<()> {
                exh += 1;
                RetryOutcome::Retry {
                    error: "always 503".into(),
                    retry_after: Some(Duration::from_millis(1)),
                }
            },
        );
        assert_eq!(exh, 3);
        assert_eq!(exh_res.err().as_deref(), Some("always 503"));
        Ok(())
    }
}
