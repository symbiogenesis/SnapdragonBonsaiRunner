//! Bounded credential-checked startup log collection for native acceptance.

use std::{
    fs::{self, File, OpenOptions},
    io::Write as _,
    path::{Path, PathBuf},
};

use bonsai_artifacts::{is_redirect, validate_external_path};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

const MAX_LOG_BYTES: usize = 65_536;
const MAX_ACCEPTANCE_BYTES: usize = 1024 * 1024;
const MAX_SESSIONS: usize = 8;
const PUBLIC_FILES: [&str; 3] = [
    "startup-backend.stdout.txt",
    "startup-backend.stderr.txt",
    "startup-diagnostics.json",
];

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Request {
    evidence_directory: PathBuf,
    #[serde(default)]
    acceptance_exit_code: Option<i32>,
    #[serde(default)]
    budget_expired: bool,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct AcceptanceSummary {
    schema_version: u32,
    all_requested_workloads_passed: Option<bool>,
}

#[derive(Serialize)]
#[serde(rename_all = "PascalCase")]
struct Source {
    session: usize,
    name: &'static str,
    classification: &'static str,
    published: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "PascalCase")]
struct PublishedFile {
    name: &'static str,
    bytes: usize,
    sha256: String,
}

#[derive(Serialize)]
#[serde(rename_all = "PascalCase")]
struct Receipt {
    schema_version: u32,
    scope: &'static str,
    acceptance_process_exit_code: Option<i32>,
    aggregate_budget_expired: bool,
    maximum_log_bytes: usize,
    maximum_sessions: usize,
    classification: &'static str,
    sources: Vec<Source>,
    published_files: Vec<PublishedFile>,
}

fn ordinary_directory(path: &Path) -> Result<(), &'static str> {
    validate_external_path(path).map_err(|_| "unsafe-directory")?;
    let metadata = fs::symlink_metadata(path).map_err(|_| "unsafe-directory")?;
    if !metadata.is_dir() || is_redirect(&metadata) {
        return Err("unsafe-directory");
    }
    Ok(())
}

// Keep opened key files alive until publication ends. Windows denies write/delete
// sharing and opens the leaf reparse point itself before checking its metadata.
fn read_bounded(path: &Path, maximum: usize) -> Result<(Vec<u8>, File), &'static str> {
    validate_external_path(path).map_err(|_| "unsafe-file")?;
    let metadata = fs::symlink_metadata(path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            "missing"
        } else {
            "read-failed"
        }
    })?;
    if !metadata.is_file() || is_redirect(&metadata) {
        return Err("unsafe-file");
    }
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt as _;
        options.share_mode(1).custom_flags(0x0020_0000);
    }
    let mut file = options.open(path).map_err(|_| "read-failed")?;
    let metadata = file.metadata().map_err(|_| "read-failed")?;
    if !metadata.is_file() || is_redirect(&metadata) {
        return Err("unsafe-file");
    }
    let maximum = u64::try_from(maximum).map_err(|_| "read-bound-exceeded")?;
    if metadata.len() > maximum {
        return Err("read-bound-exceeded");
    }
    let bytes = crate::bounded_input::read(&mut file, maximum).map_err(|error| match error {
        crate::bounded_input::Error::Bound => "read-bound-exceeded",
        crate::bounded_input::Error::Io(_) => "read-failed",
    })?;
    Ok((bytes, file))
}

fn write_new(root: &Path, name: &str, bytes: &[u8]) -> Result<(), &'static str> {
    ordinary_directory(root)?;
    let path = root.join(name);
    validate_external_path(&path).map_err(|_| "unsafe-file")?;
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt as _;
        options.share_mode(0).custom_flags(0x0020_0000);
    }
    let mut file = options.open(path).map_err(|_| "write-failed")?;
    let metadata = file.metadata().map_err(|_| "write-failed")?;
    if !metadata.is_file() || is_redirect(&metadata) {
        return Err("unsafe-file");
    }
    file.write_all(bytes).map_err(|_| "write-failed")?;
    file.sync_all().map_err(|_| "write-failed")
}

fn output_preflight(root: &Path) -> Result<(), &'static str> {
    // Inspect every name before removing any stale ordinary file. A directory or
    // reparse point is never traversed, changed or recursively removed.
    for name in PUBLIC_FILES {
        match fs::symlink_metadata(root.join(name)) {
            Ok(metadata) if !metadata.is_file() || is_redirect(&metadata) => {
                return Err("unsafe-output");
            }
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(_) => return Err("collection-failed"),
        }
    }
    for name in PUBLIC_FILES {
        let path = root.join(name);
        validate_external_path(&path).map_err(|_| "unsafe-output")?;
        match fs::remove_file(path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(_) => return Err("collection-failed"),
        }
    }
    Ok(())
}

fn session_name(name: &str) -> bool {
    name.strip_prefix("SwiftBonsai-Keys-")
        .is_some_and(|suffix| crate::identity::is_lower_hex(suffix, 32))
}

fn sessions(root: &Path) -> Result<Vec<PathBuf>, &'static str> {
    let mut directories = Vec::new();
    for entry in fs::read_dir(root).map_err(|_| "collection-failed")? {
        let entry = entry.map_err(|_| "collection-failed")?;
        if entry
            .file_name()
            .to_str()
            .is_none_or(|name| !session_name(name))
        {
            continue;
        }
        // Include a matching reparse point so key validation closes publication.
        let metadata = fs::symlink_metadata(entry.path()).map_err(|_| "collection-failed")?;
        if !metadata.is_dir() && !is_redirect(&metadata) {
            continue;
        }
        directories.push(entry.path());
        if directories.len() > MAX_SESSIONS {
            return Err("session-bound-exceeded");
        }
    }
    directories.sort();
    if directories.is_empty() {
        return Err("no-protected-worker-directory");
    }
    Ok(directories)
}

fn contains_key(text: &str, keys: &[String]) -> bool {
    keys.iter().any(|key| {
        text.as_bytes()
            .windows(key.len())
            .any(|part| part.eq_ignore_ascii_case(key.as_bytes()))
    })
}

fn is_path_delimiter_before(byte: u8) -> bool {
    matches!(
        byte,
        b' ' | b'\t'
            | b'\n'
            | b'\r'
            | b'"'
            | b'\''
            | b'`'
            | b'='
            | b'('
            | b'['
            | b'{'
            | b'<'
            | b','
            | b';'
            | b':'
    )
}

fn path_value_stop(tail: &[u8], opener: Option<u8>) -> usize {
    tail.iter()
        .position(|&byte| match opener {
            Some(quote) => {
                byte == quote || matches!(byte, b'\r' | b'\n') || byte.is_ascii_control()
            }
            None => {
                byte.is_ascii_whitespace()
                    || byte.is_ascii_control()
                    || matches!(
                        byte,
                        b'"' | b'\'' | b'`' | b'<' | b'>' | b')' | b']' | b'}' | b',' | b';'
                    )
            }
        })
        .unwrap_or(tail.len())
}

fn local_path_len(bytes: &[u8], index: usize) -> Option<usize> {
    let prev = index.checked_sub(1).map(|i| bytes[i]);
    if !prev.is_none_or(is_path_delimiter_before) {
        return None;
    }
    let tail = &bytes[index..];
    let is_windows_user = tail.len() >= 8
        && tail[0].is_ascii_alphabetic()
        && tail[1] == b':'
        && matches!(tail[2], b'\\' | b'/')
        && tail[3..8].eq_ignore_ascii_case(b"users")
        && tail.get(8).is_none_or(|&byte| {
            matches!(byte, b'\\' | b'/') || byte.is_ascii_whitespace() || byte.is_ascii_control()
        });
    let is_unc = tail.len() >= 3
        && tail[0] == b'\\'
        && tail[1] == b'\\'
        && !tail[2].is_ascii_whitespace()
        && !tail[2].is_ascii_control()
        && tail[2] != b'\\';
    let is_posix_home = (tail.len() >= 7 && tail[..7].eq_ignore_ascii_case(b"/users/"))
        || (tail.len() >= 6 && tail[..6].eq_ignore_ascii_case(b"/home/"));
    if !(is_windows_user || is_unc || is_posix_home) {
        return None;
    }
    let opener = prev.filter(|&byte| byte == b'"' || byte == b'\'');
    let len = path_value_stop(tail, opener);
    (len > 0).then_some(len)
}

fn custom_key_match_len(bytes: &[u8], keys: &[String]) -> Option<usize> {
    keys.iter()
        .filter(|key| !key.is_empty())
        .filter_map(|key| {
            let key_bytes = key.as_bytes();
            (bytes.len() >= key_bytes.len()
                && bytes[..key_bytes.len()].eq_ignore_ascii_case(key_bytes))
            .then_some(key_bytes.len())
        })
        .max()
}

fn sanitize(text: &str, keys: &[String]) -> String {
    let bytes = text.as_bytes();
    let check_custom_keys = keys.iter().any(|key| {
        !key.is_empty() && (key.len() < 64 || !key.bytes().all(|byte| byte.is_ascii_hexdigit()))
    });
    let mut output = String::with_capacity(text.len());
    let mut index = 0;
    while index < bytes.len() {
        if let Some(path_len) = local_path_len(bytes, index) {
            output.push_str("[REDACTED]");
            index += path_len;
            continue;
        }
        if check_custom_keys && let Some(key_len) = custom_key_match_len(&bytes[index..], keys) {
            output.push_str("[REDACTED]");
            index += key_len;
            continue;
        }
        let byte = bytes[index];
        if byte.is_ascii_hexdigit() {
            let mut run_len = 1;
            while index + run_len < bytes.len() && bytes[index + run_len].is_ascii_hexdigit() {
                run_len += 1;
            }
            if run_len >= 64 {
                output.push_str("[REDACTED]");
                index += run_len;
                continue;
            }
            if !check_custom_keys {
                output.push_str(&text[index..index + run_len]);
                index += run_len;
                continue;
            }
        }
        if byte < 0x80 {
            if byte.is_ascii_control() && !matches!(byte, b'\t' | b'\n' | b'\r') {
                output.push('?');
            } else {
                output.push(char::from(byte));
            }
            index += 1;
        } else if let Some(character) = text[index..].chars().next() {
            output.push(character);
            index += character.len_utf8();
        } else {
            break;
        }
    }
    output
}

fn collect_logs(
    root: &Path,
    receipt: &mut Receipt,
    keys: &mut Vec<String>,
    guards: &mut Vec<File>,
) -> Result<bool, &'static str> {
    let directories = sessions(root)?;
    for directory in &directories {
        ordinary_directory(directory).map_err(|_| "key-validation-failed")?;
        for name in ["frontend.key", "backend.key"] {
            let (bytes, guard) =
                read_bounded(&directory.join(name), 64).map_err(|_| "key-validation-failed")?;
            let key = String::from_utf8(bytes).map_err(|_| "key-validation-failed")?;
            if !crate::identity::is_lower_hex(&key, 64) {
                return Err("key-validation-failed");
            }
            keys.push(key);
            guards.push(guard);
        }
    }
    for (name, output_name) in [
        ("backend.stdout", PUBLIC_FILES[0]),
        ("backend.stderr", PUBLIC_FILES[1]),
    ] {
        let mut combined = String::new();
        for (index, directory) in directories.iter().enumerate() {
            let mut source = Source {
                session: index + 1,
                name,
                classification: "not-read",
                published: false,
            };
            match read_bounded(&directory.join(name), MAX_LOG_BYTES) {
                Err(classification) => source.classification = classification,
                Ok((bytes, _guard)) => match std::str::from_utf8(&bytes) {
                    Err(_) => source.classification = "invalid-utf8",
                    Ok("") => source.classification = "empty",
                    Ok(text) if text.trim().is_empty() => {
                        source.classification = "whitespace-only";
                    }
                    Ok(text) => {
                        let text = sanitize(text, keys);
                        if contains_key(&text, keys) {
                            source.classification = "credential-check-failed";
                        } else {
                            use std::fmt::Write as _;
                            writeln!(combined, "--- worker session {} ---", index + 1)
                                .map_err(|_| "collection-failed")?;
                            combined.push_str(&text);
                            combined.push('\n');
                            source.classification = "sanitized";
                            source.published = true;
                        }
                    }
                },
            }
            receipt.sources.push(source);
        }
        if combined.is_empty() {
            continue;
        }
        if contains_key(&combined, keys) {
            return Err("credential-check-failed");
        }
        write_new(root, output_name, combined.as_bytes()).map_err(|_| "collection-failed")?;
        receipt.published_files.push(PublishedFile {
            name: output_name,
            bytes: combined.len(),
            sha256: crate::identity::sha256(combined.as_bytes()),
        });
    }
    receipt.classification = if receipt.published_files.is_empty() {
        "no-log-published"
    } else {
        "bounded-sanitized-logs-retained"
    };
    Ok(receipt.sources.iter().all(|source| {
        matches!(
            source.classification,
            "sanitized" | "empty" | "whitespace-only"
        )
    }))
}

fn publish_receipt(root: &Path, receipt: &Receipt, keys: &[String]) -> Result<(), &'static str> {
    let bytes = serde_json::to_vec_pretty(receipt).map_err(|_| "publication-check-failed")?;
    // Check the receipt before writing as well: even an output digest matching a
    // real credential must not escape through metadata.
    let text = std::str::from_utf8(&bytes).map_err(|_| "publication-check-failed")?;
    if contains_key(text, keys) {
        return Err("publication-check-failed");
    }
    write_new(root, PUBLIC_FILES[2], &bytes).map_err(|_| "publication-check-failed")
}

fn final_check(root: &Path, receipt: &Receipt, keys: &[String]) -> Result<(), &'static str> {
    // Credential replacement shrinks hex runs; control replacement preserves
    // byte length. Input plus fixed headers stays within this aggregate bound.
    for name in receipt
        .published_files
        .iter()
        .map(|file| file.name)
        .chain(std::iter::once(PUBLIC_FILES[2]))
    {
        let (bytes, _guard) = read_bounded(
            root.join(name).as_path(),
            MAX_LOG_BYTES * MAX_SESSIONS + 4096,
        )
        .map_err(|_| "publication-check-failed")?;
        let text = std::str::from_utf8(&bytes).map_err(|_| "publication-check-failed")?;
        if contains_key(text, keys) {
            return Err("publication-check-failed");
        }
    }
    Ok(())
}

fn acceptance_workloads_passed(root: &Path) -> Option<bool> {
    let (bytes, _guard) = read_bounded(&root.join("acceptance.json"), MAX_ACCEPTANCE_BYTES).ok()?;
    // Serde structs also accept tuple sequences; a receipt must be an object.
    if bytes.iter().find(|byte| !byte.is_ascii_whitespace()) != Some(&b'{') {
        return None;
    }
    let summary: AcceptanceSummary = serde_json::from_slice(&bytes).ok()?;
    (summary.schema_version == 1)
        .then_some(summary.all_requested_workloads_passed)
        .flatten()
}

/// Collects fixed worker logs and a bounded optional boolean workload result.
///
/// All collection failures suppress readiness. No filesystem error, raw log,
/// credential, or private path enters the command's JSON or stderr diagnostics.
/// The workload result remains separate from upload readiness and process success.
///
/// # Errors
/// Rejects malformed requests; collection failures return a closed upload decision.
pub fn run(request: Value) -> crate::Result<Value> {
    let request: Request = serde_json::from_value(request)
        .map_err(|_| "Invalid native startup diagnostics request.")?;
    let mut receipt = Receipt {
        schema_version: 1,
        scope: "bounded-credential-checked-native-startup-diagnostics",
        acceptance_process_exit_code: request.acceptance_exit_code,
        aggregate_budget_expired: request.budget_expired,
        maximum_log_bytes: MAX_LOG_BYTES,
        maximum_sessions: MAX_SESSIONS,
        classification: "collection-failed",
        sources: Vec::new(),
        published_files: Vec::new(),
    };
    let root = &request.evidence_directory;
    if ordinary_directory(root).is_err() {
        return Ok(
            json!({"Ready": false, "Classification": receipt.classification,
            "AllRequestedWorkloadsPassed": null}),
        );
    }
    let all_requested_workloads_passed = acceptance_workloads_passed(root);
    if let Err(classification) = output_preflight(root) {
        return Ok(json!({"Ready": false, "Classification": classification,
            "AllRequestedWorkloadsPassed": all_requested_workloads_passed}));
    }
    let mut keys = Vec::new();
    let mut guards = Vec::new();
    let complete = match collect_logs(root, &mut receipt, &mut keys, &mut guards) {
        Ok(complete) => complete,
        Err(classification) => {
            receipt.classification = classification;
            false
        }
    };
    let published = publish_receipt(root, &receipt, &keys).is_ok();
    let checked = !complete || final_check(root, &receipt, &keys).is_ok();
    let classification = if published && checked {
        receipt.classification
    } else {
        "publication-check-failed"
    };
    Ok(
        json!({"Ready": complete && published && checked, "Classification": classification,
        "AllRequestedWorkloadsPassed": all_requested_workloads_passed}),
    )
}

#[cfg(test)]
mod tests;
