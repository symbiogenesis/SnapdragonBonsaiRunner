//! Conditional public metadata requests with bounded gzip decoding and storage.

use std::{
    collections::BTreeMap,
    fs,
    io::Read as _,
    path::Path,
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};

use super::super::policy::Asset;
use super::{MAX_METADATA, external};

const MAX_HEADERS: u64 = 64 * 1024;
const API_VERSION: &str = "2022-11-28";

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Cached {
    url: String,
    sha256: String,
    etag: Option<String>,
    modified: Option<String>,
}

#[derive(Default, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Cooldown {
    until: u64,
    failures: u32,
    #[serde(default)]
    offline: bool,
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn header_value(value: &str) -> bool {
    !value.is_empty() && value.len() <= 1024 && value.bytes().all(|byte| (32..127).contains(&byte))
}

fn slot(url: &str) -> crate::Result<&'static str> {
    let api = format!("https://api.github.com/repos/{}/", super::super::REPOSITORY);
    for (page, slot) in [(1, "releases-1"), (2, "releases-2"), (3, "releases-3")] {
        if url == format!("{api}releases?per_page=100&page={page}") {
            return Ok(slot);
        }
    }
    if let Some(version) = url.strip_prefix(&format!("{api}commits/v")) {
        super::super::policy::Version::parse(version)?;
        return Ok("source");
    }
    let prefix = format!(
        "https://github.com/{}/releases/download/v",
        super::super::REPOSITORY
    );
    if let Some((version, name)) = url
        .strip_prefix(&prefix)
        .and_then(|tail| tail.split_once('/'))
    {
        super::super::policy::Version::parse(version)?;
        return match name {
            "release-manifest.json" => Ok("manifest"),
            "SHA256SUMS.txt" => Ok("checksums"),
            _ => Err("Unsupported update metadata asset URL.".into()),
        };
    }
    Err("Unsupported update metadata URL.".into())
}

fn cached(root: &Path, slot: &str, url: &str) -> Option<(Cached, Vec<u8>)> {
    let descriptor = super::read_bounded(&root.join(format!("{slot}.json")), MAX_HEADERS).ok()?;
    let entry: Cached = serde_json::from_slice(&descriptor).ok()?;
    if entry.url != url
        || !crate::identity::is_lower_hex(&entry.sha256, 64)
        || entry
            .etag
            .as_deref()
            .is_some_and(|value| !header_value(value))
        || entry
            .modified
            .as_deref()
            .is_some_and(|value| !header_value(value))
    {
        return None;
    }
    let body = super::read_bounded(&root.join(format!("{slot}.body")), MAX_METADATA).ok()?;
    (crate::identity::sha256(&body) == entry.sha256).then_some((entry, body))
}

fn discovery_cache(
    root: &Path,
    slot: &str,
    url: &str,
    conditional: bool,
) -> Option<(Cached, Vec<u8>)> {
    conditional.then(|| cached(root, slot, url)).flatten()
}

fn content_encoding(headers: &BTreeMap<String, String>) -> crate::Result<Option<&str>> {
    let mut found = None;
    for (name, value) in headers {
        if name.eq_ignore_ascii_case("content-encoding") {
            if found.is_some() {
                return Err("Ambiguous update metadata content encoding.".into());
            }
            found = Some(value.as_str());
        }
    }
    Ok(found)
}

fn decoded(bytes: &[u8], encoding: Option<&str>) -> crate::Result<Vec<u8>> {
    if u64::try_from(bytes.len()).map_err(|error| error.to_string())? > MAX_METADATA {
        return Err("Update metadata exceeds its byte limit.".into());
    }
    let encoding = encoding.map(str::trim).unwrap_or("identity");
    if encoding.eq_ignore_ascii_case("identity") {
        Ok(bytes.to_vec())
    } else if encoding.eq_ignore_ascii_case("gzip") {
        let mut decoder = flate2::bufread::GzDecoder::new(bytes);
        let mut body = Vec::new();
        decoder
            .by_ref()
            .take(MAX_METADATA + 1)
            .read_to_end(&mut body)
            .map_err(|error| error.to_string())?;
        if u64::try_from(body.len()).map_err(|error| error.to_string())? > MAX_METADATA {
            return Err("Expanded update metadata exceeds its byte limit.".into());
        }
        if !decoder.into_inner().is_empty() {
            return Err("Update metadata gzip stream has unexpected trailing bytes.".into());
        }
        Ok(body)
    } else {
        Err("Unsupported update metadata content encoding.".into())
    }
}

// GitHub documents numeric Retry-After. Also accept the standard IMF-fixdate
// form without adding a date/time runtime or interpreting local timezone data.
fn http_date(value: &str) -> Option<u64> {
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

fn cooldown_delay(headers: &BTreeMap<String, String>, now: u64, failures: u32) -> u64 {
    let retry = headers.get("retry-after").and_then(|value| {
        value
            .parse::<u64>()
            .ok()
            .map(|seconds| now.saturating_add(seconds))
            .or_else(|| http_date(value))
    });
    let reset = (headers.get("x-ratelimit-remaining").map(String::as_str) == Some("0"))
        .then(|| {
            headers
                .get("x-ratelimit-reset")
                .and_then(|value| value.parse::<u64>().ok())
        })
        .flatten();
    // No immediate retries. Repeated failures grow from 15 minutes to six hours.
    let default = now.saturating_add(900_u64.saturating_mul(1_u64 << failures.min(5)).min(21_600));
    default.max(retry.unwrap_or(0)).max(reset.unwrap_or(0))
}

fn load_cooldown(root: &Path) -> Cooldown {
    super::read_bounded(&root.join("cooldown.json"), MAX_HEADERS)
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap_or_default()
}

fn store_cooldown(root: &Path, cooldown: &Cooldown) -> crate::Result<()> {
    super::write_current(
        &root.join("cooldown.json"),
        &serde_json::to_vec(cooldown).map_err(|error| error.to_string())?,
    )
}

fn request_headers(cached: Option<&Cached>) -> Vec<(String, String)> {
    let mut headers = vec![
        ("Accept".into(), "application/vnd.github+json".into()),
        ("X-GitHub-Api-Version".into(), API_VERSION.into()),
        (
            "User-Agent".into(),
            format!("SnapdragonBonsaiRunner/{}", env!("CARGO_PKG_VERSION")),
        ),
        // The shared client disables automatic decoding. Bound compressed and
        // expanded metadata independently before accepting a cache entry.
        ("Accept-Encoding".into(), "gzip, identity".into()),
    ];
    if let Some(cached) = cached {
        if let Some(etag) = &cached.etag {
            headers.push(("If-None-Match".into(), etag.clone()));
        } else if let Some(modified) = &cached.modified {
            headers.push(("If-Modified-Since".into(), modified.clone()));
        }
    }
    headers
}

#[derive(Clone, Copy)]
pub(in super::super) struct MetadataPolicy {
    pub(in super::super) conditional: bool,
    pub(in super::super) bypass_offline_cooldown: bool,
    pub(in super::super) retry_transient: bool,
}

fn format_wait_duration(seconds: u64) -> String {
    if seconds < 60 {
        "for less than a minute".into()
    } else {
        let minutes = seconds.div_ceil(60);
        if minutes < 60 {
            if minutes == 1 {
                "for about 1 minute".into()
            } else {
                format!("for about {minutes} minutes")
            }
        } else {
            let hours = seconds.div_ceil(3600);
            if hours == 1 {
                "for about 1 hour".into()
            } else {
                format!("for about {hours} hours")
            }
        }
    }
}

fn cooldown_blocked(cooldown: &Cooldown, now_secs: u64, bypass_offline: bool) -> bool {
    cooldown.until > now_secs && !(bypass_offline && cooldown.offline)
}

fn cooldown_error_at(cooldown: &Cooldown, now_secs: u64) -> String {
    let wait = format_wait_duration(cooldown.until.saturating_sub(now_secs));
    if cooldown.offline {
        format!(
            "{} GitHub update requests are paused {wait} after a network failure. Your installed app is unchanged.",
            crate::https_transfer::OFFLINE_PREFIX,
        )
    } else {
        format!(
            "GitHub update requests are paused {wait} after a server or network failure. Your installed app is unchanged.",
        )
    }
}

fn cooldown_error(cooldown: &Cooldown) -> String {
    cooldown_error_at(cooldown, now())
}

fn record_failure(
    root: &Path,
    headers: &BTreeMap<String, String>,
    previous: &Cooldown,
    offline: bool,
) -> crate::Result<()> {
    store_cooldown(
        root,
        &Cooldown {
            until: cooldown_delay(headers, now(), previous.failures),
            failures: previous.failures.saturating_add(1),
            offline,
        },
    )
}

fn response_body(
    response: &crate::https_transfer::Response,
    cached: Option<&(Cached, Vec<u8>)>,
) -> crate::Result<Vec<u8>> {
    if response.status == 304 {
        let (_, body) = cached
            .filter(|(entry, _)| entry.etag.is_some() || entry.modified.is_some())
            .ok_or("GitHub returned 304 without a valid conditional cache entry.")?;
        return Ok(body.clone());
    }
    super::require_successful_http(response.status)?;
    decoded(&response.body, content_encoding(&response.headers)?)
}

fn is_resilient_retry_status(status: u16) -> bool {
    matches!(status, 408 | 500 | 502 | 503 | 504)
}

fn is_resilient_retry_error(error: &str) -> bool {
    if crate::https_transfer::is_offline_message(error) {
        return true;
    }
    let lower = error.to_ascii_lowercase();
    [
        "connection reset",
        "connection aborted",
        "unexpected eof",
        "ended before content-length",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
}

fn wait_resilient_backoff(delay: Duration, deadline: Instant) -> crate::Result<bool> {
    let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
        return Ok(false);
    };
    if delay >= remaining {
        return Ok(false);
    }
    let wake = Instant::now() + delay;
    loop {
        crate::startup_cancel::checkpoint()?;
        let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
            return Ok(false);
        };
        let Some(wait) = wake.checked_duration_since(Instant::now()) else {
            return Ok(true);
        };
        if wait.is_zero() {
            return Ok(true);
        }
        thread::sleep(wait.min(remaining).min(Duration::from_millis(50)));
    }
}

fn fetch_metadata_once(
    url: &str,
    headers: &[(String, String)],
) -> crate::Result<crate::https_transfer::Response> {
    crate::https_transfer::metadata(url, headers, Duration::from_secs(10), MAX_METADATA)
}

fn fetch_metadata_resilient(
    url: &str,
    headers: &[(String, String)],
) -> crate::Result<crate::https_transfer::Response> {
    let deadline = Instant::now() + Duration::from_secs(15);
    let mut last_error: Option<String> = None;
    for attempt in 0..=2_usize {
        crate::startup_cancel::checkpoint()?;
        let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
            break;
        };
        if remaining.is_zero() {
            break;
        }
        let timeout = remaining.min(Duration::from_secs(10));
        match crate::https_transfer::metadata(url, headers, timeout, MAX_METADATA) {
            Ok(response) => {
                if attempt < 2 && is_resilient_retry_status(response.status) {
                    let delay = Duration::from_millis(250_u64 * (1_u64 << attempt));
                    if !wait_resilient_backoff(delay, deadline)? {
                        return Ok(response);
                    }
                    continue;
                }
                return Ok(response);
            }
            Err(error) => {
                if attempt < 2 && is_resilient_retry_error(&error) {
                    last_error = Some(error);
                    let delay = Duration::from_millis(250_u64 * (1_u64 << attempt));
                    if !wait_resilient_backoff(delay, deadline)? {
                        break;
                    }
                    continue;
                }
                return Err(error);
            }
        }
    }
    Err(last_error.unwrap_or_else(|| {
        "GitHub update request exceeded its retry allowance. Your installed app is unchanged."
            .into()
    }))
}

pub(in super::super) fn metadata_with_policy(
    url: &str,
    _working: &Path,
    asset: Option<&Asset>,
    policy: MetadataPolicy,
) -> crate::Result<Vec<u8>> {
    let slot = slot(url)?;
    let root = super::super::cache_root()?.join("http");
    external(&root)?;
    fs::create_dir_all(&root).map_err(|error| error.to_string())?;
    external(&root)?;

    let cached = {
        let _lock = crate::startup_cache::lock_file(&root.join("cache.lock"))?;
        // User-writable cache is only an efficiency hint for read-only discovery.
        // Every mutating gate requires new unconditional HTTPS publication bytes.
        let cached = discovery_cache(&root, slot, url, policy.conditional);
        if let (Some(asset), Some((entry, body))) = (asset, &cached)
            && entry.sha256 == asset.sha256
            && u64::try_from(body.len()).map_err(|error| error.to_string())? == asset.bytes
        {
            // The freshly revalidated catalog authenticates these immutable bytes.
            return Ok(body.clone());
        }
        let cooldown = load_cooldown(&root);
        if cooldown_blocked(&cooldown, now(), policy.bypass_offline_cooldown) {
            return Err(cooldown_error(&cooldown));
        }
        cached
    };

    let headers = request_headers(cached.as_ref().map(|(entry, _)| entry));
    let result = if policy.retry_transient && !policy.conditional {
        fetch_metadata_resilient(url, &headers)
    } else {
        fetch_metadata_once(url, &headers)
    };
    let response = match result {
        Ok(response) => response,
        Err(error) => {
            let _lock = crate::startup_cache::lock_file(&root.join("cache.lock"))?;
            let latest_cooldown = load_cooldown(&root);
            record_failure(
                &root,
                &BTreeMap::new(),
                &latest_cooldown,
                crate::https_transfer::is_offline_message(&error),
            )?;
            return Err(error);
        }
    };
    let body = match response_body(&response, cached.as_ref()) {
        Ok(body) => body,
        Err(error) => {
            let _lock = crate::startup_cache::lock_file(&root.join("cache.lock"))?;
            let latest_cooldown = load_cooldown(&root);
            record_failure(&root, &response.headers, &latest_cooldown, false)?;
            return Err(error);
        }
    };

    let _lock = crate::startup_cache::lock_file(&root.join("cache.lock"))?;
    let latest_cooldown = load_cooldown(&root);
    if response.status == 304 {
        if latest_cooldown.failures != 0 || latest_cooldown.until != 0 {
            store_cooldown(&root, &Cooldown::default())?;
        }
        return Ok(body);
    }
    let descriptor = Cached {
        url: url.into(),
        sha256: crate::identity::sha256(&body),
        etag: response
            .headers
            .get("etag")
            .filter(|value| header_value(value))
            .cloned(),
        modified: response
            .headers
            .get("last-modified")
            .filter(|value| header_value(value))
            .cloned(),
    };
    super::write_current(&root.join(format!("{slot}.body")), &body)?;
    super::write_current(
        &root.join(format!("{slot}.json")),
        &serde_json::to_vec(&descriptor).map_err(|error| error.to_string())?,
    )?;
    if latest_cooldown.failures != 0 || latest_cooldown.until != 0 {
        store_cooldown(&root, &Cooldown::default())?;
    }
    Ok(body)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;

    #[test]
    fn update_http_cache_admits_only_exact_bounded_slots_and_conditional_headers() {
        assert_eq!(
            super::super::super::REPOSITORY,
            "symbiogenesis/SnapdragonBonsaiRunner"
        );
        let api = format!(
            "https://api.github.com/repos/{}/",
            super::super::super::REPOSITORY
        );
        for page in 1..=3 {
            assert!(slot(&format!("{api}releases?per_page=100&page={page}")).is_ok());
        }
        assert!(slot(&format!("{api}releases?per_page=100&page=4")).is_err());
        assert!(
            slot("https://api.github.com/repos/symbiogenesis/SwiftBonsai2Runner/releases?per_page=100&page=1")
                .is_err()
        );
        assert!(
            slot("https://api.github.com/repos/symbiogenesis/SwiftBonsai2Runner/commits/v0.3.6")
                .is_err()
        );
        assert!(
            slot("https://github.com/symbiogenesis/SwiftBonsai2Runner/releases/download/v0.3.6/release-manifest.json")
                .is_err()
        );
        assert!(slot("https://evil.invalid/releases").is_err());
        assert!(!header_value("\"etag\"\r\nAuthorization: secret"));
        let entry = Cached {
            url: String::new(),
            sha256: String::new(),
            etag: Some("W/\"version\"".into()),
            modified: Some("date".into()),
        };
        let headers = request_headers(Some(&entry));
        assert!(headers.iter().any(|(name, value)| name == "User-Agent"
            && *value == format!("SnapdragonBonsaiRunner/{}", env!("CARGO_PKG_VERSION"))));
        assert!(
            headers
                .iter()
                .any(|(name, value)| name == "If-None-Match" && value == "W/\"version\"")
        );
        assert!(!headers.iter().any(|(name, _)| name == "If-Modified-Since"));
        let fallback = Cached {
            etag: None,
            ..entry
        };
        assert!(
            request_headers(Some(&fallback))
                .iter()
                .any(|(name, value)| name == "If-Modified-Since" && value == "date")
        );
    }

    #[test]
    fn update_http_gzip_bounds_expansion_and_rejects_corruption() -> crate::Result<()> {
        let compress = |bytes: &[u8]| -> crate::Result<Vec<u8>> {
            let mut encoder =
                flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
            encoder
                .write_all(bytes)
                .map_err(|error| error.to_string())?;
            encoder.finish().map_err(|error| error.to_string())
        };
        let gzip = Some("gzip");
        assert_eq!(decoded(&compress(b"[]")?, gzip)?, b"[]");
        assert_eq!(decoded(&compress(b"[]")?, Some("GZip"))?, b"[]");
        assert_eq!(decoded(b"[]", Some("Identity"))?, b"[]");
        let mut with_trailing = compress(b"[]")?;
        with_trailing.push(0);
        assert!(decoded(&with_trailing, gzip).is_err());
        let mut concatenated = compress(b"[")?;
        concatenated.extend_from_slice(&compress(b"]")?);
        assert!(decoded(&concatenated, gzip).is_err());
        assert!(decoded(b"corrupt", gzip).is_err());
        let oversized =
            vec![0; usize::try_from(MAX_METADATA).map_err(|error| error.to_string())? + 1];
        assert!(decoded(&compress(&oversized)?, gzip).is_err());
        for unsupported in ["br", "Br", "DEFLATE"] {
            assert!(decoded(b"[]", Some(unsupported)).is_err());
        }
        let mixed = BTreeMap::from([("Content-Encoding".into(), "GZip".into())]);
        assert_eq!(content_encoding(&mixed)?, Some("GZip"));
        let mixed_bad = BTreeMap::from([("Content-Encoding".into(), "Br".into())]);
        assert!(decoded(b"[]", content_encoding(&mixed_bad)?).is_err());
        let duplicate = BTreeMap::from([
            ("Content-Encoding".into(), "gzip".into()),
            ("content-encoding".into(), "identity".into()),
        ]);
        assert!(content_encoding(&duplicate).is_err());
        Ok(())
    }

    #[test]
    fn update_http_backoff_honors_server_deadlines_and_grows_without_retrying() {
        let mut headers = BTreeMap::from([("retry-after".into(), "7200".into())]);
        assert_eq!(cooldown_delay(&headers, 100, 0), 7300);
        headers.insert("x-ratelimit-remaining".into(), "0".into());
        headers.insert("x-ratelimit-reset".into(), "9000".into());
        assert_eq!(cooldown_delay(&headers, 100, 0), 9000);
        assert_eq!(cooldown_delay(&BTreeMap::new(), 100, 0), 1000);
        assert_eq!(cooldown_delay(&BTreeMap::new(), 100, 1), 1900);
        assert_eq!(cooldown_delay(&BTreeMap::new(), 100, 30), 21_700);
        assert_eq!(http_date("Thu, 01 Jan 1970 00:00:01 GMT"), Some(1));
        assert_eq!(
            http_date("Sat, 29 Feb 2020 00:00:00 GMT"),
            Some(1_582_934_400)
        );
        assert!(http_date("Tue, 29 Feb 2022 00:00:00 GMT").is_none());
    }

    #[test]
    fn update_http_failed_or_invalid_responses_never_reuse_cached_success() -> crate::Result<()> {
        let cached = (
            Cached {
                url: "https://example.invalid".into(),
                sha256: crate::identity::sha256(b"cached catalog"),
                etag: Some("\"validator\"".into()),
                modified: None,
            },
            b"cached catalog".to_vec(),
        );
        let mut response = crate::https_transfer::Response {
            status: 429,
            headers: BTreeMap::from([("retry-after".into(), "7200".into())]),
            body: Vec::new(),
        };
        assert!(response_body(&response, Some(&cached)).is_err());
        assert_eq!(cooldown_delay(&response.headers, 100, 0), 7300);
        response.status = 200;
        response
            .headers
            .insert("content-encoding".into(), "gzip".into());
        response.body = b"corrupt gzip".to_vec();
        assert!(response_body(&response, Some(&cached)).is_err());
        response.status = 304;
        assert_eq!(response_body(&response, Some(&cached))?, b"cached catalog");
        assert!(response_body(&response, None).is_err());
        let cached_without_validator = (
            Cached {
                etag: None,
                ..cached.0
            },
            cached.1,
        );
        assert!(response_body(&response, Some(&cached_without_validator)).is_err());
        Ok(())
    }

    #[test]
    fn update_http_fresh_gates_ignore_forged_cache_and_discovery_rejects_partial_pairs()
    -> crate::Result<()> {
        let root = std::env::temp_dir().join(format!(
            "Bonsai-Http-Cache-{}",
            crate::startup_process::unique()
        ));
        fs::create_dir(&root).map_err(|error| error.to_string())?;
        let url = format!(
            "https://api.github.com/repos/{}/releases?per_page=100&page=1",
            super::super::super::REPOSITORY
        );
        let body = b"locally forged publication bytes";
        let descriptor = Cached {
            url: url.clone(),
            sha256: crate::identity::sha256(body),
            etag: Some("\"genuine-server-validator\"".into()),
            modified: None,
        };
        super::super::write_current(&root.join("releases-1.body"), body)?;
        super::super::write_current(
            &root.join("releases-1.json"),
            &serde_json::to_vec(&descriptor).map_err(|error| error.to_string())?,
        )?;
        assert!(discovery_cache(&root, "releases-1", &url, true).is_some());
        // A new HTTPS request must authenticate every authoritative byte. The
        // writable body/hash/validator triple cannot authorize a mutating gate.
        let fresh = discovery_cache(&root, "releases-1", &url, false);
        assert!(fresh.is_none());
        let headers = request_headers(fresh.as_ref().map(|(entry, _)| entry));
        assert!(!headers.iter().any(|(name, _)| name.starts_with("If-")));
        assert!(discovery_cache(&root, "releases-1", "different-url", true).is_none());
        super::super::write_current(&root.join("releases-1.body"), b"partial publication")?;
        assert!(discovery_cache(&root, "releases-1", &url, true).is_none());
        fs::remove_file(root.join("releases-1.body")).map_err(|error| error.to_string())?;
        fs::remove_file(root.join("releases-1.json")).map_err(|error| error.to_string())?;
        fs::remove_dir(root).map_err(|error| error.to_string())
    }

    #[test]
    fn cooldown_preserves_legacy_schema_and_offline_classification() -> crate::Result<()> {
        let legacy: Cooldown = serde_json::from_str(r#"{"until":1234,"failures":2}"#)
            .map_err(|error| error.to_string())?;
        assert!(!legacy.offline);
        let server_message = cooldown_error(&legacy);
        assert!(!crate::https_transfer::is_offline_message(&server_message));

        let offline = Cooldown {
            until: 1234,
            failures: 1,
            offline: true,
        };
        let offline_message = cooldown_error(&offline);
        assert!(crate::https_transfer::is_offline_message(&offline_message));
        assert!(!offline_message.contains("Unix time"));

        // Manual and mutating probes bypass stale offline cooldowns, while
        // honoring server/rate-limit cooldowns.
        assert!(cooldown_blocked(&offline, 1000, false));
        assert!(!cooldown_blocked(&offline, 1000, true));
        assert!(cooldown_blocked(&legacy, 1000, false));
        assert!(cooldown_blocked(&legacy, 1000, true));

        let minute_msg = cooldown_error_at(
            &Cooldown {
                until: 1090,
                failures: 1,
                offline: true,
            },
            1000,
        );
        assert!(minute_msg.contains("for about 2 minutes"));
        let hour_msg = cooldown_error_at(
            &Cooldown {
                until: 4600,
                failures: 1,
                offline: false,
            },
            1000,
        );
        assert!(hour_msg.contains("for about 1 hour"));
        Ok(())
    }
}
