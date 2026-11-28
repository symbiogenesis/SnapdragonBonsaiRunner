//! Fixed historical artifact admission and integrity replay without producer execution.

use super::{decode, digest, integer, path, read_json, require, text, write_json};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File, OpenOptions},
    io::{self, Read as _, Seek as _, SeekFrom, Write as _},
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

pub(super) const REPOSITORY: &str = "symbiogenesis/SwiftBonsai2Runner";
pub(super) const REVISION: &str = "469f57b233898fd10b97d5ebc10aaef78b940e30";
pub(super) const RUNTIME: &str = "cae16fa6998dac58a947eba9ae4398ed6a9fb863";
pub(super) const RUN: u64 = 41_568_281_287;
pub(super) const ARTIFACT: u64 = 13_200_218_351;
pub(super) const ARCHIVE_BYTES: u64 = 294_092_332;
pub(super) const ARCHIVE_SHA: &str =
    "c31de3e837f4d4d43df5d144892c0b5c4b36d747ab763799abe0556b419af8d8";
pub(super) const COMPARATOR_SHA: &str =
    "b8372072b3d77a918c5c6fa7038da95f51a50701f6927fb4b4931a591b056c08";
pub(super) const WORKFLOW: &str = ".github/workflows/native-ptq1-numerics.yml";

pub(super) fn producer(run: &Value, artifact: &Value, jobs: &Value) -> crate::Result<()> {
    require(
        run["id"] == RUN
            && run["head_sha"] == REVISION
            && run["head_branch"] == "main"
            && run["event"] == "workflow_dispatch"
            && run["run_attempt"] == 1
            && run["path"] == WORKFLOW
            && run["status"] == "completed"
            && run["conclusion"] == "success"
            && run["repository"]["full_name"] == REPOSITORY,
        "inexact original producer",
    )?;
    require(
        artifact["id"] == ARTIFACT
            && artifact["name"] == format!("native-ptq1-numerics-{RUN}-1")
            && artifact["size_in_bytes"] == ARCHIVE_BYTES
            && artifact["expired"] == false
            && artifact["digest"] == format!("sha256:{ARCHIVE_SHA}")
            && artifact["workflow_run"]["id"] == RUN
            && artifact["workflow_run"]["head_sha"] == REVISION
            && artifact["workflow_run"]["head_branch"] == "main"
            && artifact["workflow_run"]["repository_id"] == run["repository"]["id"],
        "inexact original artifact",
    )?;
    let items = jobs["jobs"]
        .as_array()
        .filter(|a| a.len() == 2)
        .ok_or("original job inventory differs")?;
    require(
        jobs["total_count"] == 2
            && items
                .iter()
                .filter_map(|j| j["name"].as_str())
                .collect::<BTreeSet<_>>()
                == BTreeSet::from(["numerics", "prepare"]),
        "original job inventory differs",
    )?;
    for item in items {
        require(
            item["status"] == "completed"
                && item["conclusion"] == "success"
                && item["run_id"] == RUN
                && item["head_sha"] == REVISION,
            "original jobs did not complete successfully",
        )?;
        if item["name"] == "numerics" {
            require(
                item["labels"] == json!([format!("ptq1-numerics-{RUN}")]),
                "inexact original owner label",
            )?;
        }
    }
    Ok(())
}

pub(super) fn redirect(url: &str) -> crate::Result<()> {
    crate::https_transfer::validate_uri(url)?;
    let parsed: ureq::http::Uri = url.parse().map_err(|_| "invalid retained redirect")?;
    let host = parsed
        .host()
        .ok_or("retained redirect host missing")?
        .to_ascii_lowercase();
    require(
        host.ends_with(".blob.core.windows.net") || host.ends_with(".githubusercontent.com"),
        "unapproved artifact redirect",
    )
}

fn redact_secret(message: &str, secret: &str) -> String {
    if secret.is_empty() {
        message.to_owned()
    } else {
        message.replace(secret, "[REDACTED]")
    }
}

fn response_retry_after(response: &ureq::http::Response<ureq::Body>) -> Option<Duration> {
    response
        .headers()
        .get("retry-after")
        .and_then(|value| value.to_str().ok())
        .and_then(crate::https_transfer::parse_retry_after)
}

fn endpoint(endpoint: &str, token: &str, destination: Option<&Path>) -> crate::Result<Value> {
    let config = ureq::Agent::config_builder()
        .https_only(true)
        .http_status_as_error(false)
        .max_redirects(0)
        .timeout_global(Some(Duration::from_secs(120)));
    #[cfg(windows)]
    let config = config.tls_config(
        ureq::tls::TlsConfig::builder()
            .provider(ureq::tls::TlsProvider::NativeTls)
            .root_certs(ureq::tls::RootCerts::PlatformVerifier)
            .build(),
    );
    let agent = config.build().new_agent();
    let api_url = format!("https://api.github.com/repos/{REPOSITORY}/{endpoint}");
    let until = Instant::now()
        .checked_add(Duration::from_secs(120))
        .ok_or("Invalid retained endpoint timeout.")?;
    endpoint_with(
        &agent,
        &api_url,
        token,
        destination,
        ARCHIVE_BYTES,
        ARCHIVE_SHA,
        until,
        redirect,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "Retained endpoint helper binds agent, credentialed API URI, redirect validator and exact archive bounds."
)]
fn endpoint_with(
    agent: &ureq::Agent,
    api_url: &str,
    token: &str,
    destination: Option<&Path>,
    archive_bytes: u64,
    archive_sha: &str,
    until: Instant,
    validate_redirect: impl Fn(&str) -> crate::Result<()>,
) -> crate::Result<Value> {
    use crate::https_transfer::{
        RetryOutcome, is_retryable_http_status, is_transient_io_error, is_transient_ureq_error,
        run_with_retry,
    };

    let maximum = if destination.is_some() {
        archive_bytes
    } else {
        4_194_304
    };
    let partial_path = destination.map(|target| {
        target.with_extension(format!("{}.partial", crate::startup_process::unique()))
    });
    let mut partial_guard = partial_path
        .as_ref()
        .map(|path| crate::fs_guard::TempFileGuard::new(path.clone()));
    let mut redirect_target: Option<String> = None;

    run_with_retry(until, &mut crate::startup_cancel::checkpoint, |_| {
        let remaining = match until.checked_duration_since(Instant::now()) {
            Some(duration) if !duration.is_zero() => duration,
            _ => {
                return RetryOutcome::Fatal(
                    "Native HTTPS operation exceeded its aggregate allowance.".into(),
                );
            }
        };
        let mut response = if let Some(target) = &redirect_target {
            match agent
                .get(target)
                .config()
                .timeout_global(Some(remaining))
                .build()
                .call()
            {
                Ok(response) => response,
                Err(error) => {
                    let message = redact_secret(&error.to_string(), token);
                    return if is_transient_ureq_error(&error) {
                        RetryOutcome::Retry {
                            error: message,
                            retry_after: None,
                        }
                    } else {
                        RetryOutcome::Fatal(message)
                    };
                }
            }
        } else {
            let initial = match agent
                .get(api_url)
                .header("Authorization", format!("Bearer {token}"))
                .header("Accept", "application/vnd.github+json")
                .header("X-GitHub-Api-Version", "2022-11-28")
                .header("User-Agent", "bonsai-retained-reference")
                .config()
                .timeout_global(Some(remaining))
                .build()
                .call()
            {
                Ok(response) => response,
                Err(error) => {
                    let message = redact_secret(&error.to_string(), token);
                    return if is_transient_ureq_error(&error) {
                        RetryOutcome::Retry {
                            error: message,
                            retry_after: None,
                        }
                    } else {
                        RetryOutcome::Fatal(message)
                    };
                }
            };
            if [301, 302, 303, 307, 308].contains(&initial.status().as_u16()) {
                if destination.is_none() {
                    return RetryOutcome::Fatal("unexpected metadata redirect".into());
                }
                let target = match initial
                    .headers()
                    .get("location")
                    .and_then(|h| h.to_str().ok())
                {
                    Some(location) => location.to_owned(),
                    None => return RetryOutcome::Fatal("artifact redirect missing".into()),
                };
                if let Err(error) = validate_redirect(&target) {
                    return RetryOutcome::Fatal(redact_secret(&error, token));
                }
                redirect_target = Some(target.clone());
                let remaining = match until.checked_duration_since(Instant::now()) {
                    Some(duration) if !duration.is_zero() => duration,
                    _ => {
                        return RetryOutcome::Fatal(
                            "Native HTTPS operation exceeded its aggregate allowance.".into(),
                        );
                    }
                };
                // A fresh credential-free request, with automatic redirects disabled.
                match agent
                    .get(&target)
                    .config()
                    .timeout_global(Some(remaining))
                    .build()
                    .call()
                {
                    Ok(response) => response,
                    Err(error) => {
                        let message = redact_secret(&error.to_string(), token);
                        return if is_transient_ureq_error(&error) {
                            RetryOutcome::Retry {
                                error: message,
                                retry_after: None,
                            }
                        } else {
                            RetryOutcome::Fatal(message)
                        };
                    }
                }
            } else {
                initial
            }
        };

        let status = response.status().as_u16();
        if is_retryable_http_status(status) {
            return RetryOutcome::Retry {
                error: format!("retained endpoint returned HTTP {status}"),
                retry_after: response_retry_after(&response),
            };
        }
        if status != 200 {
            return RetryOutcome::Fatal("retained endpoint response differs".into());
        }

        if let (Some(target), Some(partial)) = (destination, partial_path.as_ref()) {
            let mut output = match OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(partial)
            {
                Ok(file) => file,
                Err(error) => return RetryOutcome::Fatal(error.to_string()),
            };
            let copied = match io::copy(
                &mut response.body_mut().as_reader().take(maximum + 1),
                &mut output,
            ) {
                Ok(bytes) => bytes,
                Err(error) => {
                    let message = redact_secret(&error.to_string(), token);
                    return if is_transient_io_error(&error) {
                        RetryOutcome::Retry {
                            error: message,
                            retry_after: None,
                        }
                    } else {
                        RetryOutcome::Fatal(message)
                    };
                }
            };
            if let Err(error) = output.sync_all() {
                return RetryOutcome::Fatal(error.to_string());
            }
            drop(output);
            if copied < maximum {
                return RetryOutcome::Retry {
                    error: "archive raw length/hash differs".into(),
                    retry_after: None,
                };
            }
            if copied != maximum {
                return RetryOutcome::Fatal("archive raw length/hash differs".into());
            }
            match digest(partial) {
                Ok(actual) if actual == archive_sha => {}
                Ok(_) => return RetryOutcome::Fatal("archive raw length/hash differs".into()),
                Err(error) => return RetryOutcome::Fatal(error),
            }
            if let Err(error) = crate::fs_guard::rename_with_retry(partial, target) {
                return RetryOutcome::Fatal(error.to_string());
            }
            if let Some(guard) = partial_guard.as_mut() {
                guard.disarm();
            }
            RetryOutcome::Done(Value::Null)
        } else {
            let mut raw = Vec::new();
            if let Err(error) = response
                .body_mut()
                .as_reader()
                .take(maximum + 1)
                .read_to_end(&mut raw)
            {
                let message = redact_secret(&error.to_string(), token);
                return if is_transient_io_error(&error) {
                    RetryOutcome::Retry {
                        error: message,
                        retry_after: None,
                    }
                } else {
                    RetryOutcome::Fatal(message)
                };
            }
            match u64::try_from(raw.len()) {
                Ok(len) if len <= maximum => {}
                Ok(_) => return RetryOutcome::Fatal("retained response exceeds byte bound".into()),
                Err(error) => return RetryOutcome::Fatal(error.to_string()),
            }
            match decode(&raw) {
                Ok(value) => RetryOutcome::Done(value),
                Err(error) => RetryOutcome::Fatal(error),
            }
        }
    })
}

fn central(archive: &Path) -> crate::Result<Vec<(String, u64)>> {
    let mut file = File::open(archive).map_err(|error| error.to_string())?;
    let length = file.metadata().map_err(|error| error.to_string())?.len();
    let tail_len = length.min(65_557);
    file.seek(SeekFrom::Start(length - tail_len))
        .map_err(|error| error.to_string())?;
    let mut tail = vec![0; usize::try_from(tail_len).map_err(|error| error.to_string())?];
    file.read_exact(&mut tail)
        .map_err(|error| error.to_string())?;
    let u16_at = |bytes: &[u8], at: usize| -> crate::Result<u16> {
        Ok(u16::from_le_bytes(
            bytes
                .get(at..at + 2)
                .ok_or("truncated ZIP integer")?
                .try_into()
                .map_err(|_| "truncated ZIP integer")?,
        ))
    };
    let u32_at = |bytes: &[u8], at: usize| -> crate::Result<u32> {
        Ok(u32::from_le_bytes(
            bytes
                .get(at..at + 4)
                .ok_or("truncated ZIP integer")?
                .try_into()
                .map_err(|_| "truncated ZIP integer")?,
        ))
    };
    let end = (0..tail.len().saturating_sub(21))
        .rev()
        .find(|&at| {
            u32_at(&tail, at).ok() == Some(0x0605_4b50)
                && u16_at(&tail, at + 20).is_ok_and(|n| at + 22 + usize::from(n) == tail.len())
        })
        .ok_or("unsupported ZIP directory")?;
    require(
        u16_at(&tail, end + 4)? == 0 && u16_at(&tail, end + 6)? == 0,
        "unsupported ZIP directory",
    )?;
    let count = u16_at(&tail, end + 10)?;
    require(
        count == u16_at(&tail, end + 8)? && count < u16::MAX,
        "unsupported ZIP member count",
    )?;
    let bytes = u64::from(u32_at(&tail, end + 12)?);
    let offset = u64::from(u32_at(&tail, end + 16)?);
    require(
        offset + bytes
            == length - tail_len + u64::try_from(end).map_err(|error| error.to_string())?,
        "ZIP directory extent differs",
    )?;
    file.seek(SeekFrom::Start(offset))
        .map_err(|error| error.to_string())?;
    let mut records = Vec::new();
    for _ in 0..count {
        let mut header = [0; 46];
        file.read_exact(&mut header)
            .map_err(|error| error.to_string())?;
        require(
            u32_at(&header, 0)? == 0x0201_4b50,
            "ZIP central record differs",
        )?;
        let flags = u16_at(&header, 8)?;
        let method = u16_at(&header, 10)?;
        let size = u64::from(u32_at(&header, 24)?);
        let attributes = u32_at(&header, 38)?;
        require(
            flags & 1 == 0
                && [0, 8].contains(&method)
                && size <= 536_870_912
                && [0, 0x8000].contains(&((attributes >> 16) & 0xf000))
                && attributes & 0x10 == 0,
            "invalid ZIP encryption/method/extent",
        )?;
        let mut name = vec![0; usize::from(u16_at(&header, 28)?)];
        file.read_exact(&mut name)
            .map_err(|error| error.to_string())?;
        let name = String::from_utf8(name).map_err(|error| error.to_string())?;
        file.seek(SeekFrom::Current(
            i64::from(u16_at(&header, 30)?) + i64::from(u16_at(&header, 32)?),
        ))
        .map_err(|error| error.to_string())?;
        records.push((name, size));
    }
    require(
        file.stream_position().map_err(|error| error.to_string())? == offset + bytes,
        "ZIP central directory trailing bytes",
    )?;
    Ok(records)
}

pub(super) fn extract(
    archive: &Path,
    destination: &Path,
    expected: &Value,
) -> crate::Result<Value> {
    let expected = expected.as_object().ok_or("invalid retained inventory")?;
    let metadata = central(archive)?;
    require(
        metadata.len() == expected.len(),
        "archive member count differs",
    )?;
    let mut zip = zip::ZipArchive::new(File::open(archive).map_err(|error| error.to_string())?)
        .map_err(|error| error.to_string())?;
    require(zip.len() == expected.len(), "archive member count differs")?;
    let mut seen = BTreeSet::new();
    let mut manifest = BTreeMap::new();
    for (index, (central_name, central_size)) in metadata.iter().enumerate() {
        let mut member = zip.by_index(index).map_err(|error| error.to_string())?;
        let name = member
            .name()
            .map_err(|error| error.to_string())?
            .into_owned();
        require(
            name == *central_name
                && member.size() == *central_size
                && name.len() <= 200
                && !name.is_empty()
                && !name.contains([':', '\\'])
                && name
                    .split('/')
                    .all(|part| !part.is_empty() && part != "." && part != "..")
                && expected.contains_key(&name)
                && seen.insert(name.to_lowercase()),
            "unsafe unexpected or duplicate archive member",
        )?;
        let mode = member.unix_mode().unwrap_or(0) & 0xf000;
        let record = &expected[&name];
        require(
            [0, 0x8000].contains(&mode)
                && !member.is_dir()
                && member.size() == integer(record, "bytes")?
                && member.size() <= 536_870_912,
            "invalid archive member type or extent",
        )?;
        let target = destination.join(&name);
        fs::create_dir_all(target.parent().ok_or("missing archive member parent")?)
            .map_err(|error| error.to_string())?;
        let mut output = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&target)
            .map_err(|error| error.to_string())?;
        let mut hash = crate::identity::Hasher::new()?;
        let mut total = 0;
        let mut buffer = vec![0; 1_048_576];
        loop {
            let count = member
                .read(&mut buffer)
                .map_err(|error| error.to_string())?;
            if count == 0 {
                break;
            }
            total += u64::try_from(count).map_err(|error| error.to_string())?;
            require(
                total <= member.size(),
                "expanded member exceeds declared extent",
            )?;
            hash.update(&buffer[..count])?;
            output
                .write_all(&buffer[..count])
                .map_err(|error| error.to_string())?;
        }
        let sha = hash.finalize_hex()?;
        require(
            total == member.size() && sha == text(record, "sha256")?,
            "retained member hash differs",
        )?;
        manifest.insert(
            name.clone(),
            json!({"path":name,"bytes":total,"sha256":sha}),
        );
    }
    require(seen.len() == expected.len(), "missing retained member")?;
    Ok(manifest.into_values().collect::<Vec<_>>().into())
}

pub(super) fn replay(root: &Path, comparator: &Path) -> crate::Result<Value> {
    require(
        digest(comparator)? == COMPARATOR_SHA,
        "original comparator source differs",
    )?;
    let original = read_json(&root.join("comparison.json"), 1_048_576)?;
    let owner = read_json(&root.join("numerics.json"), 1_048_576)?;
    require(
        owner["Revision"] == REVISION
            && (owner["RunAttempt"] == 1 || owner["RunAttempt"] == "1")
            && text(&owner, "RunUrl")?.ends_with(&format!("/actions/runs/{RUN}"))
            && owner["Status"] == "completed-one-token-smoke"
            && owner["Phase"] == "completed"
            && owner["Comparison"] == original
            && owner["DefaultMayChange"] == false
            && owner["Plan"]["RustSource"]["commit"] == RUNTIME
            && owner["Plan"]["RustBackend"] == "neon"
            && owner["Plan"]["RustRowWorkers"] == 4
            && owner["Plan"]["AdapterCondition"] == "standard"
            && owner["Plan"]["AdapterScale"].as_f64() == Some(1.3),
        "retained owner/comparison binding differs",
    )?;
    require(
        original["status"] == "passed-bounded-one-token-smoke"
            && original["files"] == 129
            && original["elements"] == 39_504_384
            && original["byteExactFiles"] == 129
            && original["failedElements"] == 0
            && original["defaultMayChange"] == false
            && original["backendRequested"] == "neon"
            && original["rustRowWorkers"] == 4
            && original["adapterCondition"] == "standard"
            && original["adapterScale"].as_f64() == Some(1.3),
        "retained selected comparison differs",
    )?;
    let recomputed = super::native::compare(
        &json!({"native":root.join("native"),"rust":root.join("neon"),"backend":"neon","rowWorkers":4,"adapterCondition":"standard"}),
    )?;
    require(
        recomputed == original,
        "replayed comparison differs from original report",
    )?;
    Ok(recomputed)
}

struct ScratchDir(Option<PathBuf>);

impl ScratchDir {
    fn create(path: PathBuf) -> crate::Result<Self> {
        fs::create_dir(&path).map_err(|error| error.to_string())?;
        Ok(Self(Some(path)))
    }

    fn cleanup(&mut self) -> crate::Result<()> {
        if let Some(path) = self.0.take() {
            remove_dir_all_with_retry(&path)?;
        }
        Ok(())
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        if let Some(path) = self.0.take() {
            let _ = remove_dir_all_with_retry(&path);
        }
    }
}

fn is_retryable_cleanup_error(error: &io::Error) -> bool {
    error.kind() == io::ErrorKind::PermissionDenied
        || matches!(error.raw_os_error(), Some(5 | 32 | 33))
}

#[cfg(windows)]
fn clear_readonly_tree(path: &Path) {
    let Ok(metadata) = fs::symlink_metadata(path) else {
        return;
    };
    if bonsai_artifacts::is_redirect(&metadata) {
        return;
    }
    if metadata.is_dir()
        && let Ok(entries) = fs::read_dir(path)
    {
        for entry in entries.flatten() {
            clear_readonly_tree(&entry.path());
        }
    }
    if metadata.permissions().readonly() {
        let mut permissions = metadata.permissions();
        #[expect(
            clippy::permissions_set_readonly_false,
            reason = "Windows-only scratch cleanup clears the DOS readonly attribute; this branch never changes Unix permissions."
        )]
        permissions.set_readonly(false);
        let _ = fs::set_permissions(path, permissions);
    }
}

fn remove_dir_all_with_retry(path: &Path) -> crate::Result<()> {
    const MAX_ATTEMPTS: usize = 8;
    for attempt in 0..MAX_ATTEMPTS {
        match fs::remove_dir_all(path) {
            Ok(()) => return Ok(()),
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
            Err(error) if is_retryable_cleanup_error(&error) && attempt + 1 < MAX_ATTEMPTS => {
                #[cfg(windows)]
                clear_readonly_tree(path);
                let shift = u32::try_from(attempt.min(4)).unwrap_or(4);
                std::thread::sleep(Duration::from_millis(20 << shift));
            }
            Err(error) => return Err(error.to_string()),
        }
    }
    Ok(())
}

fn finalize_audit(
    output: &Path,
    mut report: Value,
    result: crate::Result<()>,
    mut scratch: Option<ScratchDir>,
) -> crate::Result<Value> {
    if let Err(ref error) = result {
        report["failureClass"] = "native-validation-or-execution".into();
        report["failureReason"] = error.clone().into();
    }
    let write_result = write_json(&output.join("audit.json"), &report);
    let cleanup_result = if let Some(mut guard) = scratch.take() {
        guard.cleanup()
    } else {
        Ok(())
    };
    result?;
    write_result?;
    cleanup_result?;
    Ok(report)
}

#[expect(
    clippy::too_many_lines,
    reason = "One failure-retaining operation encloses hosted admission, exact artifact intake, replay and scratch cleanup."
)]
pub(super) fn audit(request: &Value) -> crate::Result<Value> {
    let comparator = path(request, "comparator")?;
    let output = path(request, "outputDirectory")?;
    let repository = path(request, "repository")?;
    require(!output.exists(), "retained audit output must be fresh")?;
    fs::create_dir_all(&output).map_err(|error| error.to_string())?;
    let mut report = json!({"schemaVersion":2,"status":"failed","stage":"hosted-guard","sourceRunId":RUN.to_string(),"sourceRevision":REVISION,"artifactId":ARTIFACT.to_string(),"artifactSha256":ARCHIVE_SHA,"comparatorSourceSha256":COMPARATOR_SHA,"replayImplementationSha256":digest(&repository.join("tools/bonsai/src/native_reference/native.rs"))?,"originalComparatorExecuted":false,"auditRevision":std::env::var("GITHUB_SHA").unwrap_or_default(),"auditRunId":std::env::var("GITHUB_RUN_ID").unwrap_or_default(),"ownerDeviceExecutedByAudit":false,"modelExecutedByAudit":false,"defaultMayChange":false});
    let mut scratch = None;
    let result = (|| -> crate::Result<()> {
        for (key, value) in [
            ("GITHUB_ACTIONS", "true"),
            ("GITHUB_REPOSITORY", REPOSITORY),
            ("GITHUB_REF", "refs/heads/main"),
            ("RUNNER_ENVIRONMENT", "github-hosted"),
        ] {
            require(
                std::env::var(key).ok().as_deref() == Some(value),
                "hosted main only",
            )?;
        }
        let token = std::env::var("GH_TOKEN")
            .ok()
            .filter(|s| !s.is_empty())
            .ok_or("artifact credential unavailable")?;
        report["stage"] = "producer-metadata".into();
        let run = endpoint(&format!("actions/runs/{RUN}"), &token, None)?;
        let artifact = endpoint(&format!("actions/artifacts/{ARTIFACT}"), &token, None)?;
        let jobs = endpoint(
            &format!("actions/runs/{RUN}/jobs?filter=all&per_page=100"),
            &token,
            None,
        )?;
        producer(&run, &artifact, &jobs)?;
        write_json(
            &output.join("producer.json"),
            &json!({"run":run,"artifact":artifact,"jobs":jobs}),
        )?;
        let root = PathBuf::from(
            std::env::var_os("RUNNER_TEMP").ok_or("runner temporary directory absent")?,
        )
        .join(format!(
            "ptq1-retained-{}",
            crate::startup_process::unique()
        ));
        scratch = Some(ScratchDir::create(root.clone())?);
        let archive = root.join("evidence.zip");
        let extracted = root.join("extracted");
        report["stage"] = "archive-download".into();
        endpoint(
            &format!("actions/artifacts/{ARTIFACT}/zip"),
            &token,
            Some(&archive),
        )?;
        report["stage"] = "retained-member-integrity".into();
        let inventory = read_json(
            &repository.join(format!(
                "tools/native-reference/fixtures/retained-{RUN}.inventory.json"
            )),
            1_048_576,
        )?;
        let files = inventory["files"]
            .as_array()
            .filter(|a| a.len() == 155)
            .ok_or("expected inventory producer/extent differs")?;
        require(
            inventory["sourceRunId"].as_str() == Some("41568281287")
                && inventory["sourceRevision"] == REVISION
                && files
                    .iter()
                    .map(|f| integer(f, "bytes"))
                    .collect::<crate::Result<Vec<_>>>()?
                    .iter()
                    .sum::<u64>()
                    == 316_267_936,
            "expected inventory producer/extent differs",
        )?;
        let mut expected = serde_json::Map::new();
        for file in files {
            require(
                expected
                    .insert(text(file, "path")?.into(), file.clone())
                    .is_none(),
                "duplicate expected inventory member",
            )?;
        }
        let manifest = extract(&archive, &extracted, &Value::Object(expected))?;
        write_json(
            &output.join("manifest.json"),
            &json!({"sourceRunId":RUN.to_string(),"sourceRevision":REVISION,"files":manifest}),
        )?;
        report["stage"] = "replacement-comparator-replay".into();
        let comparison = replay(&extracted, &comparator)?;
        fs::copy(
            extracted.join("comparison.json"),
            output.join("original-comparison.json"),
        )
        .map_err(|error| error.to_string())?;
        write_json(&output.join("replayed-comparison.json"), &comparison)?;
        report["retainedFiles"] = 155.into();
        report["retainedBytes"] = 316_267_936.into();
        for (destination, source) in [
            ("comparedFiles", "files"),
            ("comparedValues", "elements"),
            ("byteExactFiles", "byteExactFiles"),
            ("modelSha256", "modelSha256"),
            ("adapterSha256", "adapterSha256"),
            ("nativeCaptureSha256", "nativeCaptureSha256"),
            ("nativeStateSha256", "nativeStateSha256"),
            ("rustStdoutSha256", "rustStdoutSha256"),
        ] {
            report[destination] = comparison[source].clone();
        }
        report["runtimeRevision"] = RUNTIME.into();
        report["originalExecutionUnchanged"] = true.into();
        report["fullContextQualified"] = false.into();
        report["scope"]="Retained byte integrity and replacement-comparator replay only; no new model/device execution, promotion, provenance, quality or full-context acceptance.".into();
        report["status"] = "passed-retained-capture-integrity-and-replay".into();
        report["stage"] = "complete".into();
        Ok(())
    })();
    finalize_audit(&output, report, result, scratch)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finalize_audit_writes_report_first_cleans_scratch_and_propagates_error() -> crate::Result<()>
    {
        let root = std::env::temp_dir().join(format!(
            "bonsai-retained-finalize-{}",
            crate::startup_process::unique()
        ));
        fs::create_dir(&root).map_err(|error| error.to_string())?;
        let output = root.join("output");
        fs::create_dir(&output).map_err(|error| error.to_string())?;
        let scratch_path = root.join("scratch");
        let scratch = ScratchDir::create(scratch_path.clone())?;
        let nested_file = scratch_path.join("evidence.zip");
        fs::write(&nested_file, b"partial").map_err(|error| error.to_string())?;
        #[cfg(windows)]
        {
            let mut perms = fs::metadata(&nested_file)
                .map_err(|error| error.to_string())?
                .permissions();
            perms.set_readonly(true);
            fs::set_permissions(&nested_file, perms).map_err(|error| error.to_string())?;
        }

        let report = json!({"schemaVersion": 2, "status": "failed", "stage": "archive-download"});
        let outcome = finalize_audit(
            &output,
            report,
            Err("simulated archive failure".into()),
            Some(scratch),
        );
        assert_eq!(outcome.err().as_deref(), Some("simulated archive failure"));
        assert!(!scratch_path.exists());

        let audit_json = read_json(&output.join("audit.json"), 65_536)?;
        assert_eq!(audit_json["status"], "failed");
        assert_eq!(audit_json["failureClass"], "native-validation-or-execution");
        assert_eq!(audit_json["failureReason"], "simulated archive failure");

        // Primary error is not masked even when writing audit.json fails.
        let non_dir_output = root.join("not-a-dir");
        fs::write(&non_dir_output, b"file").map_err(|error| error.to_string())?;
        let masked_check = finalize_audit(
            &non_dir_output,
            json!({"status": "failed"}),
            Err("primary error".into()),
            None,
        );
        assert_eq!(masked_check.err().as_deref(), Some("primary error"));

        // result=Ok + write_result=Err still cleans scratch and returns write_result
        let w_scratch_path = root.join("w-scratch");
        let w_scratch = ScratchDir::create(w_scratch_path.clone())?;
        let w_err = finalize_audit(&non_dir_output, json!({"status": "ok"}), Ok(()), Some(w_scratch));
        assert!(w_err.is_err());
        assert!(!w_scratch_path.exists());

        // is_retryable_cleanup_error raw codes 5, 32, 33 vs 2
        for code in [5_i32, 32, 33] {
            assert!(is_retryable_cleanup_error(&io::Error::from_raw_os_error(
                code
            )));
        }
        assert!(!is_retryable_cleanup_error(
            &io::Error::from_raw_os_error(2)
        ));

        remove_dir_all_with_retry(&root)?;
        Ok(())
    }

    #[test]
    fn endpoint_retries_transient_http_strips_redirect_credentials_and_cleans_partials()
    -> crate::Result<()> {
        use std::net::TcpListener;

        fn read_http_request(stream: &mut std::net::TcpStream) -> crate::Result<String> {
            let mut raw = Vec::new();
            let mut buf = [0_u8; 1024];
            loop {
                let n = stream.read(&mut buf).map_err(|e| e.to_string())?;
                if n == 0 {
                    break;
                }
                raw.extend_from_slice(&buf[..n]);
                if raw.windows(4).any(|w| w == b"\r\n\r\n") {
                    break;
                }
            }
            Ok(String::from_utf8_lossy(&raw).into_owned())
        }

        let secret = "ghp_secret_token_never_leak";
        let api_listener = TcpListener::bind("127.0.0.1:0").map_err(|e| e.to_string())?;
        let api_addr = api_listener.local_addr().map_err(|e| e.to_string())?;
        let blob_listener = TcpListener::bind("127.0.0.1:0").map_err(|e| e.to_string())?;
        let blob_addr = blob_listener.local_addr().map_err(|e| e.to_string())?;

        let payload = b"retained-archive";
        let payload_sha = crate::identity::sha256(payload);
        let payload_len = u64::try_from(payload.len()).map_err(|e| e.to_string())?;

        let redirect_url = format!("http://{blob_addr}/blob.zip");
        let redirect_response = format!(
            "HTTP/1.1 302 Found\r\nLocation: {redirect_url}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
        );
        let ok_blob_response = format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {payload_len}\r\nConnection: close\r\n\r\n{}",
            std::str::from_utf8(payload).map_err(|e| e.to_string())?
        );

        let api_thread = std::thread::spawn(move || -> crate::Result<Vec<String>> {
            let mut seen = Vec::new();
            for reply in [
                "HTTP/1.1 429 Too Many Requests\r\nRetry-After: 0\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_owned(),
                "HTTP/1.1 200 OK\r\nContent-Length: 10\r\nConnection: close\r\n\r\n{\"ok\":123}".to_owned(),
                redirect_response,
            ] {
                let (mut stream, _) = api_listener.accept().map_err(|e| e.to_string())?;
                seen.push(read_http_request(&mut stream)?);
                stream.write_all(reply.as_bytes()).map_err(|e| e.to_string())?;
            }
            Ok(seen)
        });
        let blob_thread = std::thread::spawn(move || -> crate::Result<Vec<String>> {
            let mut seen = Vec::new();
            for reply in [
                "HTTP/1.1 503 Service Unavailable\r\nRetry-After: 0\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_owned(),
                "HTTP/1.1 200 OK\r\nContent-Length: 4\r\nConnection: close\r\n\r\nhalf".to_owned(),
                ok_blob_response,
            ] {
                let (mut stream, _) = blob_listener.accept().map_err(|e| e.to_string())?;
                seen.push(read_http_request(&mut stream)?);
                stream.write_all(reply.as_bytes()).map_err(|e| e.to_string())?;
            }
            Ok(seen)
        });

        let root = std::env::temp_dir().join(format!(
            "bonsai-retained-http-{}",
            crate::startup_process::unique()
        ));
        fs::create_dir(&root).map_err(|error| error.to_string())?;
        let mut _root_guard = crate::fs_guard::TempDirGuard::new(root.clone());

        let agent = ureq::Agent::config_builder()
            .https_only(false)
            .http_status_as_error(false)
            .max_redirects(0)
            .build()
            .new_agent();
        let until = Instant::now() + Duration::from_secs(10);

        let metadata = endpoint_with(
            &agent,
            &format!("http://{api_addr}/meta"),
            secret,
            None,
            payload_len,
            &payload_sha,
            until,
            |_| Ok(()),
        )?;
        assert_eq!(metadata, json!({"ok": 123}));

        let archive_path = root.join("evidence.zip");
        endpoint_with(
            &agent,
            &format!("http://{api_addr}/zip"),
            secret,
            Some(&archive_path),
            payload_len,
            &payload_sha,
            until,
            |_| Ok(()),
        )?;
        assert_eq!(fs::read(&archive_path).map_err(|e| e.to_string())?, payload);
        let partial_count = fs::read_dir(&root)
            .map_err(|error| error.to_string())?
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "partial"))
            .count();
        assert_eq!(partial_count, 0);

        let api_requests = api_thread.join().map_err(|_| "API server panicked.")??;
        assert_eq!(api_requests.len(), 3);
        assert!(
            api_requests
                .iter()
                .all(|req| req.contains(&format!("Bearer {secret}")))
        );

        let blob_requests = blob_thread.join().map_err(|_| "Blob server panicked.")??;
        assert_eq!(blob_requests.len(), 3);
        assert!(blob_requests.iter().all(
            |req| !req.to_ascii_lowercase().contains("authorization") && !req.contains(secret)
        ));

        assert!(!redact_secret(&format!("failed with {secret}"), secret).contains(secret));
        assert_eq!(redact_secret("no secret here", ""), "no secret here");

        // Error branches in endpoint_with:
        // (a) metadata 302 redirect -> fatal without retry
        // (b) 302 missing Location -> fatal without retry
        // (c) 302 rejected validate_redirect -> fatal with secret redacted
        // (d) archive SHA-256 mismatch -> fatal without retry & .partial cleaned
        // (e) archive oversize -> fatal without retry & .partial cleaned
        // (f) 503 exhaustion across attempts
        let err_listener = TcpListener::bind("127.0.0.1:0").map_err(|e| e.to_string())?;
        let err_addr = err_listener.local_addr().map_err(|e| e.to_string())?;
        let err_replies = vec![
            "HTTP/1.1 302 Found\r\nLocation: http://127.0.0.1/other\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_owned(),
            "HTTP/1.1 302 Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_owned(),
            format!(
                "HTTP/1.1 302 Found\r\nLocation: http://untrusted.example.invalid/{secret}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
            ),
            "HTTP/1.1 200 OK\r\nContent-Length: 8\r\nConnection: close\r\n\r\nwrongzip".to_owned(),
            "HTTP/1.1 200 OK\r\nContent-Length: 32\r\nConnection: close\r\n\r\n0123456789abcdef0123456789abcdef".to_owned(),
            "HTTP/1.1 503 Service Unavailable\r\nRetry-After: 0\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_owned(),
            "HTTP/1.1 503 Service Unavailable\r\nRetry-After: 0\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_owned(),
            "HTTP/1.1 503 Service Unavailable\r\nRetry-After: 0\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_owned(),
            "HTTP/1.1 503 Service Unavailable\r\nRetry-After: 0\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_owned(),
        ];
        let err_server = std::thread::spawn(move || -> crate::Result<usize> {
            let mut count = 0_usize;
            for reply in err_replies {
                let (mut stream, _) = err_listener.accept().map_err(|e| e.to_string())?;
                let _ = read_http_request(&mut stream)?;
                stream.write_all(reply.as_bytes()).map_err(|e| e.to_string())?;
                count += 1;
            }
            Ok(count)
        });

        let bad_zip = root.join("bad.zip");
        // (a) metadata redirect
        assert!(
            endpoint_with(
                &agent,
                &format!("http://{err_addr}/a"),
                secret,
                None,
                64,
                &payload_sha,
                until,
                |_| Ok(()),
            )
            .is_err()
        );
        // (b) missing Location on 302
        assert!(
            endpoint_with(
                &agent,
                &format!("http://{err_addr}/b"),
                secret,
                Some(&bad_zip),
                64,
                &payload_sha,
                until,
                |_| Ok(()),
            )
            .is_err()
        );
        // (c) rejected validate_redirect with secret redacted
        let redir_err = endpoint_with(
            &agent,
            &format!("http://{err_addr}/c"),
            secret,
            Some(&bad_zip),
            64,
            &payload_sha,
            until,
            |_| Ok(()),
        )
        .err()
        .ok_or("expected redirect validation error")?;
        assert!(!redir_err.contains(secret));
        // (d) SHA-256 mismatch
        assert!(
            endpoint_with(
                &agent,
                &format!("http://{err_addr}/d"),
                secret,
                Some(&bad_zip),
                payload_len,
                &payload_sha,
                until,
                |_| Ok(()),
            )
            .is_err()
        );
        // (e) oversize
        assert!(
            endpoint_with(
                &agent,
                &format!("http://{err_addr}/e"),
                secret,
                Some(&bad_zip),
                16,
                &payload_sha,
                until,
                |_| Ok(()),
            )
            .is_err()
        );
        // (f) 503 exhaustion (4 attempts)
        assert!(
            endpoint_with(
                &agent,
                &format!("http://{err_addr}/f"),
                secret,
                None,
                64,
                &payload_sha,
                until,
                |_| Ok(()),
            )
            .is_err()
        );
        assert!(!bad_zip.exists());
        let leftover_partials = fs::read_dir(&root)
            .map_err(|e| e.to_string())?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "partial"))
            .count();
        assert_eq!(leftover_partials, 0);
        assert_eq!(err_server.join().map_err(|_| "err server panicked")??, 9);
        Ok(())
    }
}
