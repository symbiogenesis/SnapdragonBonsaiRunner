//! Offline preservation and retrieval of an explicitly reviewed evidence selection.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File, OpenOptions},
    io::{Read as _, Write as _},
    path::{Path, PathBuf},
};

use bonsai_artifacts::{VerifiedFile, open_locked_file, validate_external_path};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::identity::{Hasher, is_lower_hex, sha256};

const MANIFEST: &str = "evidence-manifest.json";
const MANIFEST_LIMIT: u64 = 1024 * 1024;
const FILE_LIMIT: u64 = 64 * 1024 * 1024;
const TOTAL_LIMIT: u64 = 256 * 1024 * 1024;
const ENTRY_LIMIT: usize = 4096;

#[derive(Deserialize)]
#[serde(tag = "operation", rename_all = "kebab-case", deny_unknown_fields)]
enum Request {
    Seal {
        #[serde(rename = "sourceDirectory")]
        source: PathBuf,
        #[serde(rename = "outputDirectory")]
        output: PathBuf,
        #[serde(rename = "credentialFreeSelectionReviewed")]
        reviewed: bool,
        identities: Value,
        files: Vec<Entry>,
    },
    Verify {
        #[serde(rename = "directory")]
        directory: PathBuf,
        #[serde(rename = "manifestSha256")]
        digest: String,
    },
    Restore {
        #[serde(rename = "directory")]
        directory: PathBuf,
        #[serde(rename = "manifestSha256")]
        digest: String,
        #[serde(rename = "outputDirectory")]
        output: PathBuf,
    },
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Entry {
    path: String,
    bytes: u64,
    sha256: String,
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Manifest {
    schema_version: u32,
    kind: String,
    credential_free_selection_reviewed: bool,
    identities: Value,
    files: Vec<Entry>,
}

fn contains_local_path_text(text: &str) -> bool {
    let bytes = text.as_bytes();
    for (index, _) in bytes.iter().enumerate() {
        let tail = &bytes[index..];
        let prev = index.checked_sub(1).map(|i| bytes[i]);
        let drive = tail.len() >= 3
            && tail[0].is_ascii_alphabetic()
            && tail[1] == b':'
            && matches!(tail[2], b'\\' | b'/')
            && prev.is_none_or(|byte| !byte.is_ascii_alphanumeric());
        let unc = tail.len() >= 3
            && tail[0] == b'\\'
            && tail[1] == b'\\'
            && !tail[2].is_ascii_whitespace()
            && !tail[2].is_ascii_control()
            && tail[2] != b'\\';
        let file_url = tail.len() >= 7 && tail[..7].eq_ignore_ascii_case(b"file://");
        let posix_home = prev.is_none_or(|byte| {
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
                    | b'<'
                    | b','
                    | b';'
                    | b':'
            )
        }) && ((tail.len() >= 7 && tail[..7].eq_ignore_ascii_case(b"/users/"))
            || (tail.len() >= 6 && tail[..6].eq_ignore_ascii_case(b"/home/")));
        if drive || unc || file_url || posix_home {
            return true;
        }
    }
    false
}

fn contains_local_path(value: &Value) -> bool {
    match value {
        Value::String(text) => contains_local_path_text(text),
        Value::Array(items) => items.iter().any(contains_local_path),
        Value::Object(map) => map
            .iter()
            .any(|(key, item)| contains_local_path_text(key) || contains_local_path(item)),
        Value::Null | Value::Bool(_) | Value::Number(_) => false,
    }
}

fn safe_error(error: impl std::fmt::Display) -> String {
    let message = error.to_string();
    if contains_local_path_text(&message) {
        "Evidence archive operation failed without disclosing local paths.".into()
    } else {
        message
    }
}

fn external(path: &Path) -> crate::Result<()> {
    validate_external_path(path).map_err(safe_error)
}

fn safe_name(name: &str) -> bool {
    if name.is_empty() || name.len() > 240 || name.eq_ignore_ascii_case(MANIFEST) {
        return false;
    }
    let parts = name.split('/').collect::<Vec<_>>();
    parts.len() <= 16
        && parts.iter().all(|part| {
            let stem = part
                .split('.')
                .next()
                .unwrap_or_default()
                .to_ascii_lowercase();
            !part.is_empty()
                && !part.starts_with('.')
                && !part.ends_with(['.', ' '])
                && part
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
                && !matches!(stem.as_str(), "con" | "prn" | "aux" | "nul")
                && !(stem.len() == 4
                    && (stem.starts_with("com") || stem.starts_with("lpt"))
                    && matches!(stem.as_bytes()[3], b'1'..=b'9'))
                && !matches!(
                    stem.as_str(),
                    "credentials" | "keys" | "conversations" | "models" | "adapters"
                )
        })
        && !matches!(
            name.rsplit('.')
                .next()
                .unwrap_or_default()
                .to_ascii_lowercase()
                .as_str(),
            "gguf" | "safetensors" | "exe" | "dll" | "key" | "pem" | "pfx" | "wav" | "flac"
        )
}

fn validate(manifest: &Manifest) -> crate::Result<u64> {
    if manifest.schema_version != 1
        || manifest.kind != "bonsai-qualification-evidence"
        || !manifest.credential_free_selection_reviewed
        || manifest
            .identities
            .as_object()
            .is_none_or(serde_json::Map::is_empty)
        || contains_local_path(&manifest.identities)
        || manifest.files.is_empty()
        || manifest.files.len() > ENTRY_LIMIT
    {
        return Err(
            "A reviewed credential-free selection and path-free capture identities are required."
                .into(),
        );
    }
    let mut paths = BTreeSet::new();
    let mut total = 0_u64;
    for entry in &manifest.files {
        let key = entry.path.to_ascii_lowercase();
        if !safe_name(&entry.path)
            || !is_lower_hex(&entry.sha256, 64)
            || entry.bytes > FILE_LIMIT
            || !paths.insert(key)
        {
            return Err(
                "Evidence selection contains an unsafe, duplicate or oversized file.".into(),
            );
        }
        total = total
            .checked_add(entry.bytes)
            .ok_or("Evidence byte count overflow.")?;
    }
    if total > TOTAL_LIMIT {
        return Err("Evidence selection exceeds the aggregate byte bound.".into());
    }
    // A file cannot also be the ancestor of another selected file.
    for path in &paths {
        for (index, _) in path.match_indices('/') {
            if paths.contains(&path[..index]) {
                return Err("Evidence file and directory names conflict.".into());
            }
        }
    }
    Ok(total)
}

fn read_manifest(
    directory: &Path,
    digest: &str,
) -> crate::Result<(Vec<u8>, Manifest, SelectionGuards)> {
    if !is_lower_hex(digest, 64) {
        return Err("The independently retained manifest SHA-256 is required.".into());
    }
    let mut directories = BTreeMap::new();
    hold_directories(directory, &mut directories)?;
    let path = directory.join(MANIFEST);
    let input = open_locked_file(&path).map_err(safe_error)?;
    let bytes = crate::bounded_input::read(input, MANIFEST_LIMIT).map_err(|error| match error {
        crate::bounded_input::Error::Bound => {
            "Evidence manifest differs from its independently retained identity.".into()
        }
        crate::bounded_input::Error::Io(error) => safe_error(error),
    })?;
    let length = u64::try_from(bytes.len()).map_err(|_| "Evidence manifest length overflow.")?;
    if sha256(&bytes) != digest {
        return Err("Evidence manifest differs from its independently retained identity.".into());
    }
    let guard = VerifiedFile::verify(&path, length, digest).map_err(safe_error)?;
    let manifest = serde_json::from_slice(&bytes).map_err(|_| "Invalid evidence manifest.")?;
    Ok((
        bytes,
        manifest,
        SelectionGuards {
            _files: vec![guard],
            _directories: directories,
        },
    ))
}

fn inventory(directory: &Path, manifest: &Manifest) -> crate::Result<()> {
    let mut expected = BTreeSet::from([MANIFEST.to_owned()]);
    for entry in &manifest.files {
        expected.insert(entry.path.clone());
        for (index, _) in entry.path.match_indices('/') {
            expected.insert(entry.path[..index].to_owned());
        }
    }
    let mut seen = BTreeSet::new();
    let mut pending = vec![PathBuf::new()];
    while let Some(relative) = pending.pop() {
        external(&directory.join(&relative))?;
        for entry in fs::read_dir(directory.join(&relative)).map_err(safe_error)? {
            let entry = entry.map_err(safe_error)?;
            let path = relative.join(entry.file_name());
            let name = path
                .to_str()
                .ok_or("Evidence paths must be Unicode.")?
                .replace('\\', "/");
            if !expected.contains(&name) || !seen.insert(name) {
                return Err("Evidence archive contains unexpected or duplicate entries.".into());
            }
            let metadata = fs::symlink_metadata(entry.path()).map_err(safe_error)?;
            if bonsai_artifacts::is_redirect(&metadata) {
                return Err("Evidence archive contains a redirected path.".into());
            }
            if metadata.is_dir() {
                pending.push(path);
            } else if !metadata.is_file() {
                return Err("Evidence archive contains a special file.".into());
            }
        }
    }
    if seen != expected {
        return Err("Evidence archive inventory is incomplete.".into());
    }
    Ok(())
}

struct SelectionGuards {
    _files: Vec<VerifiedFile>,
    _directories: BTreeMap<PathBuf, File>,
}

fn guards(directory: &Path, manifest: &Manifest) -> crate::Result<SelectionGuards> {
    let mut directories = BTreeMap::new();
    hold_directories(directory, &mut directories)?;
    let mut files = Vec::new();
    for entry in &manifest.files {
        let path = directory.join(&entry.path);
        hold_directories(
            path.parent().ok_or("Missing evidence parent.")?,
            &mut directories,
        )?;
        files.push(VerifiedFile::verify(&path, entry.bytes, &entry.sha256).map_err(safe_error)?);
    }
    Ok(SelectionGuards {
        _files: files,
        _directories: directories,
    })
}

fn create(path: &Path) -> crate::Result<File> {
    external(path)?;
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt as _;
        options.share_mode(0).custom_flags(0x0020_0000);
    }
    options.open(path).map_err(safe_error)
}

fn hold_directories(path: &Path, handles: &mut BTreeMap<PathBuf, File>) -> crate::Result<()> {
    external(path)?;
    // Hold ancestors from the root down before opening each descendant.
    crate::bounded_input::hold_directory_chain(path, handles, true, 1).map_err(safe_error)?;
    external(path)?;
    Ok(())
}

fn separate_trees(source: &Path, output: &Path) -> bool {
    let parts = |path: &Path| {
        path.components()
            .map(|part| part.as_os_str().to_string_lossy().to_ascii_lowercase())
            .collect::<Vec<_>>()
    };
    let source = parts(source);
    let output = parts(output);
    !source.starts_with(&output) && !output.starts_with(&source)
}

fn copy(directory: &Path, output: &Path, bytes: &[u8], manifest: &Manifest) -> crate::Result<()> {
    external(directory)?;
    external(output)?;
    let parent = output.parent().ok_or("Missing output parent.")?;
    let mut directory_guards = BTreeMap::new();
    hold_directories(parent, &mut directory_guards)?;
    // Resolve existing ancestry before comparison, including Windows short-name
    // aliases and the actual filesystem's case spelling. The output leaf is new.
    let source_identity = fs::canonicalize(directory).map_err(safe_error)?;
    let output_identity = fs::canonicalize(parent)
        .map_err(safe_error)?
        .join(output.file_name().ok_or("Missing output directory name.")?);
    if !separate_trees(&source_identity, &output_identity) {
        return Err("Evidence source and destination must be separate directory trees.".into());
    }
    // A fresh directory and create_new files preserve every existing backup.
    fs::create_dir(output).map_err(safe_error)?;
    hold_directories(output, &mut directory_guards)?;
    for entry in &manifest.files {
        let mut parent = output.to_owned();
        let parts = entry.path.split('/').collect::<Vec<_>>();
        for part in &parts[..parts.len() - 1] {
            parent.push(part);
            external(&parent)?;
            match fs::create_dir(&parent) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(error) => return Err(safe_error(error)),
            }
            // Hold every new level before creating its descendants.
            hold_directories(&parent, &mut directory_guards)?;
        }
        let path = output.join(&entry.path);
        let mut source = open_locked_file(&directory.join(&entry.path)).map_err(safe_error)?;
        let mut destination = create(&path)?;
        let mut digest = Hasher::new()?;
        let mut remaining = entry.bytes;
        let mut buffer = vec![0_u8; 64 * 1024];
        loop {
            let count = source.read(&mut buffer).map_err(safe_error)?;
            if count == 0 {
                break;
            }
            remaining = remaining
                .checked_sub(u64::try_from(count).map_err(|_| "Evidence read length overflow.")?)
                .ok_or("Evidence input grew during copying.")?;
            destination
                .write_all(&buffer[..count])
                .map_err(safe_error)?;
            digest.update(&buffer[..count])?;
        }
        if remaining != 0 || digest.finalize_hex()? != entry.sha256 {
            return Err("Evidence copy differs from the selected identity.".into());
        }
        destination.sync_all().map_err(safe_error)?;
    }
    // Publish the success marker last. Failed copies remain unsealed for diagnosis.
    let mut destination = create(&output.join(MANIFEST))?;
    destination.write_all(bytes).map_err(safe_error)?;
    destination.sync_all().map_err(safe_error)
}

fn receipt(digest: &str, manifest: &Manifest) -> crate::Result<Value> {
    Ok(
        json!({"schemaVersion":1,"kind":"bonsai-qualification-evidence-integrity",
        "manifestSha256":digest,"files":manifest.files.len(),
        "bytes":validate(manifest)?,"identities":manifest.identities,"integrityVerified":true,
        "qualificationGranted":false,"defaultPromotionQualified":false,
        "scope":"Exact selected bytes and archive inventory only. The caller must retain this digest separately and arrange durable storage; this is no execution, provenance, human-review or promotion verdict."}),
    )
}

/// Seal, verify or restore an explicit credential-free qualification selection.
///
/// No capture is downloaded and no directory is selected recursively. Verification
/// requires a separately retained manifest digest. Failed writes leave an unsealed
/// new directory; existing archives are never replaced or removed.
///
/// # Errors
/// Rejects unreviewed selections, unsafe or Git paths, missing/changed bytes,
/// unanchored manifests, extra files, oversized inputs and existing destinations.
pub fn run(request: Value) -> crate::Result<Value> {
    if !cfg!(windows) {
        return Err("Evidence archival requires Windows file and ancestor guards.".into());
    }
    match serde_json::from_value(request).map_err(|error| crate::cli::bounded_json_error(&error))? {
        Request::Seal {
            source,
            output,
            reviewed,
            identities,
            files,
        } => {
            let manifest = Manifest {
                schema_version: 1,
                kind: "bonsai-qualification-evidence".into(),
                credential_free_selection_reviewed: reviewed,
                identities,
                files,
            };
            validate(&manifest)?;
            let bytes = serde_json::to_vec(&manifest).map_err(|error| error.to_string())?;
            if u64::try_from(bytes.len()).map_err(|_| "Evidence manifest length overflow.")?
                > MANIFEST_LIMIT
            {
                return Err("Evidence manifest exceeds its byte bound.".into());
            }
            let _guards = guards(&source, &manifest)?;
            copy(&source, &output, &bytes, &manifest)?;
            inventory(&output, &manifest)?;
            let _copied = guards(&output, &manifest)?;
            let digest = sha256(&bytes);
            let (_bytes, _manifest, _manifest_guard) = read_manifest(&output, &digest)?;
            receipt(&digest, &manifest)
        }
        Request::Verify { directory, digest } => {
            let (_bytes, manifest, _manifest_guard) = read_manifest(&directory, &digest)?;
            validate(&manifest)?;
            let _guards = guards(&directory, &manifest)?;
            inventory(&directory, &manifest)?;
            receipt(&digest, &manifest)
        }
        Request::Restore {
            directory,
            digest,
            output,
        } => {
            let (bytes, manifest, _manifest_guard) = read_manifest(&directory, &digest)?;
            validate(&manifest)?;
            let _guards = guards(&directory, &manifest)?;
            inventory(&directory, &manifest)?;
            copy(&directory, &output, &bytes, &manifest)?;
            let _copied = guards(&output, &manifest)?;
            inventory(&output, &manifest)?;
            let (_bytes, _manifest, _copied_manifest_guard) = read_manifest(&output, &digest)?;
            receipt(&digest, &manifest)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evidence_selection_rejects_aliases_credentials_and_conflicts() {
        for name in [
            "../capture.json",
            "a\\b",
            "a//b",
            "a/./b",
            "a/../b",
            "C:/a",
            ".git/a",
            "NUL.txt",
            "a/COM1.log",
            "evidence-manifest.json",
            "EVIDENCE-MANIFEST.JSON",
            "a./b",
            "keys/private.json",
            "conversations/a.json",
            "model.gguf",
            "x.safetensors",
            "x.key",
        ] {
            assert!(!safe_name(name), "{name}");
        }
        for names in [vec!["a.json", "A.json"], vec!["a", "a/b.json"]] {
            let manifest = Manifest {
                schema_version: 1,
                kind: "bonsai-qualification-evidence".into(),
                credential_free_selection_reviewed: true,
                identities: json!({"source":"retained"}),
                files: names
                    .into_iter()
                    .map(|path| Entry {
                        path: path.into(),
                        bytes: 0,
                        sha256: sha256(b""),
                    })
                    .collect(),
            };
            assert!(validate(&manifest).is_err());
        }
    }

    #[test]
    #[cfg(windows)]
    fn evidence_archive_rejects_overlapping_trees_with_windows_path_aliases() {
        for source in [
            r"C:\scratch\source",
            r"C:\scratch\source\",
            r"C:\scratch\\source\\",
            "C:/scratch/source/",
        ] {
            for output in [
                r"C:\scratch\source\archive",
                r"c:\SCRATCH\SOURCE",
                r"C:\scratch",
            ] {
                assert!(!separate_trees(Path::new(source), Path::new(output)));
            }
        }
        assert!(separate_trees(
            Path::new(r"C:\scratch\source"),
            Path::new(r"C:\scratch\archive")
        ));
    }

    #[cfg(windows)]
    fn seal(root: &Path) -> crate::Result<Value> {
        let source = root.join("source");
        fs::create_dir(&source).map_err(|error| error.to_string())?;
        fs::create_dir(source.join("observations")).map_err(|error| error.to_string())?;
        fs::write(
            source.join("observations/no-go.json"),
            b"{\"passed\":false}\r\n",
        )
        .map_err(|error| error.to_string())?;
        fs::write(source.join("empty.log"), []).map_err(|error| error.to_string())?;
        // An unselected file is never discovered or copied, even in the source root.
        fs::write(source.join("not-selected.txt"), b"unselected")
            .map_err(|error| error.to_string())?;
        run(
            json!({"operation":"seal","sourceDirectory":source,"outputDirectory":root.join("archive"),
            "credentialFreeSelectionReviewed":true,
            "identities":{"repositoryRevision":"frozen-source","runUrl":"retained-run","result":"no-go"},
            "files":[{"path":"observations/no-go.json","bytes":18,"sha256":sha256(b"{\"passed\":false}\r\n")},
                {"path":"empty.log","bytes":0,"sha256":sha256(b"")}]}),
        )
    }

    #[test]
    #[cfg(windows)]
    fn evidence_archive_output_directory_guards_deny_replacement() -> crate::Result<()> {
        let root = crate::bounded_input::ScratchDir::new("Bonsai-Evidence")?;
        let child = root.join("destination");
        fs::create_dir(&child).map_err(|error| error.to_string())?;
        let mut guards = BTreeMap::new();
        hold_directories(&child, &mut guards)?;
        assert!(fs::rename(&child, root.join("replaced")).is_err());
        drop(guards);
        fs::rename(child, root.join("replaced")).map_err(|error| error.to_string())?;
        Ok(())
    }

    #[test]
    #[cfg(windows)]
    fn evidence_archive_retrieves_exact_losing_bytes_without_original_runner() -> crate::Result<()>
    {
        let root = crate::bounded_input::ScratchDir::new("Bonsai-Evidence")?;
        let report = seal(root.path())?;
        let digest = report["manifestSha256"]
            .as_str()
            .ok_or("Missing manifest digest.")?;
        assert_eq!(report["integrityVerified"], true);
        assert_eq!(report["qualificationGranted"], false);
        assert_eq!(report["files"], 2);
        assert!(!root.join("archive/not-selected.txt").exists());
        let restored = run(
            json!({"operation":"restore","directory":root.join("archive"),
            "manifestSha256":digest,"outputDirectory":root.join("retrieved")}),
        )?;
        assert_eq!(restored["identities"], report["identities"]);
        // The independently retained digest works after the source capture is gone.
        fs::rename(root.join("source"), root.join("unavailable-source"))
            .map_err(|error| error.to_string())?;
        assert_eq!(
            run(
                json!({"operation":"verify","directory":root.join("retrieved"),
            "manifestSha256":digest})
            )?["manifestSha256"],
            digest
        );
        assert_eq!(
            fs::read(root.join("retrieved/observations/no-go.json"))
                .map_err(|error| error.to_string())?,
            b"{\"passed\":false}\r\n"
        );
        assert_eq!(
            fs::read(root.join("archive").join(MANIFEST)).map_err(|error| error.to_string())?,
            fs::read(root.join("retrieved").join(MANIFEST)).map_err(|error| error.to_string())?
        );
        assert!(
            run(
                json!({"operation":"restore","directory":root.join("archive"),
            "manifestSha256":digest,"outputDirectory":root.join("retrieved")})
            )
            .is_err()
        );
        Ok(())
    }

    #[test]
    #[cfg(windows)]
    fn evidence_archive_rejects_changed_missing_extra_and_unanchored_bytes() -> crate::Result<()> {
        let root = crate::bounded_input::ScratchDir::new("Bonsai-Evidence")?;
        let report = seal(root.path())?;
        let request = json!({"operation":"verify","directory":root.join("archive"),"manifestSha256":report["manifestSha256"]});
        let path = root.join("archive/observations/no-go.json");
        fs::write(&path, b"{\"passed\":true }\r\n").map_err(|error| error.to_string())?;
        assert!(run(request.clone()).is_err());
        fs::write(&path, b"{\"passed\":false}\r\n").map_err(|error| error.to_string())?;
        fs::write(root.join("archive/extra.json"), []).map_err(|error| error.to_string())?;
        assert!(run(request.clone()).is_err());
        fs::remove_file(root.join("archive/extra.json")).map_err(|error| error.to_string())?;
        fs::remove_file(root.join("archive/empty.log")).map_err(|error| error.to_string())?;
        assert!(run(request.clone()).is_err());
        fs::write(root.join("archive/empty.log"), []).map_err(|error| error.to_string())?;
        assert!(run(json!({"operation":"verify","directory":root.join("archive"),"manifestSha256":"0".repeat(64)})).is_err());
        let manifest = root.join("archive").join(MANIFEST);
        let mut bytes = fs::read(&manifest).map_err(|error| error.to_string())?;
        bytes.push(b' ');
        fs::write(manifest, bytes).map_err(|error| error.to_string())?;
        assert!(run(request).is_err());
        Ok(())
    }

    #[test]
    fn evidence_archive_rejects_local_user_and_unc_paths_in_identities_and_errors() {
        for bad_identity in [
            json!({"source": r"C:\Users\alice\evidence"}),
            json!({"nested": ["ok", r"\\server\share\logs"]}),
            json!({"macPath": "/Users/alice/tmp/run.json"}),
            json!({"linuxPath": "/home/bob/.cache/model"}),
            json!({r"C:\Users\alice": "ok"}),
            json!({"url": "file:///Users/alice/archive.json"}),
        ] {
            let manifest = Manifest {
                schema_version: 1,
                kind: "bonsai-qualification-evidence".into(),
                credential_free_selection_reviewed: true,
                identities: bad_identity,
                files: vec![Entry {
                    path: "summary.json".into(),
                    bytes: 0,
                    sha256: sha256(b""),
                }],
            };
            assert!(validate(&manifest).is_err());
        }

        let valid_url = Manifest {
            schema_version: 1,
            kind: "bonsai-qualification-evidence".into(),
            credential_free_selection_reviewed: true,
            identities: json!({"runUrl": "https://github.com/org/repo/actions/runs/123"}),
            files: vec![Entry {
                path: "summary.json".into(),
                bytes: 0,
                sha256: sha256(b""),
            }],
        };
        assert!(validate(&valid_url).is_ok());

        let receipt_value = receipt(&sha256(b""), &valid_url).ok();
        assert!(receipt_value.is_some());
        if let Some(value) = receipt_value {
            assert!(
                value
                    .as_object()
                    .is_some_and(|obj| !obj.contains_key("directory"))
            );
        }

        let redacted = safe_error(r"open failed at C:\Users\alice\secret.json");
        assert!(!redacted.contains("Alice"));
        assert!(!redacted.contains("alice"));
        assert!(!redacted.contains(r"C:\Users"));

        if cfg!(windows) {
            let err = run(json!({
                "operation": "bad-op-C:\\Users\\alice\\secret",
                "directory": r"C:\Temp\x"
            }))
            .err()
            .unwrap_or_default();
            assert!(!err.contains("alice"));
        }
    }
}
