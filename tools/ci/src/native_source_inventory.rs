//! Exact, immutable external ASR package inventory checks; no source is executed.

use std::{
    collections::{BTreeMap, BTreeSet},
    ffi::OsString,
    fs::{self, File, Metadata},
    io::Read as _,
    path::{Component, Path},
    time::Instant,
};

use bonsai_native_hash::{Hasher, hash_bytes};
use serde::Deserialize;

use crate::Result;

const MANIFEST: &str = "src/speech-asr/native-sources.json";
const USAGE: &str =
    "Usage: bonsai-ci native-source-verify --root PATH --source PATH --archive PATH";
const MAX_FILES: usize = 4096;
const MAX_FILE_BYTES: u64 = 32 * 1024 * 1024;
const MAX_TOTAL_BYTES: u64 = 128 * 1024 * 1024;
const MAX_MANIFEST_BYTES: u64 = 2 * 1024 * 1024;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Package {
    name: String,
    version: String,
    registry: String,
    archive_sha256: String,
    repository: String,
    revision: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Entry {
    path: String,
    bytes: u64,
    sha256: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Inventory {
    schema_version: u32,
    scope: String,
    package: Package,
    excluded_local_markers: Vec<String>,
    files: Vec<Entry>,
}

pub(super) fn plain_metadata(path: &Path) -> Result<Metadata> {
    let metadata = fs::symlink_metadata(path).map_err(|error| error.to_string())?;
    if crate::impact::workspace::is_link(&metadata) {
        return Err("Native source inventory rejects symbolic links and reparse points.".into());
    }
    Ok(metadata)
}

pub(super) fn plain_ancestors(path: &Path) -> Result<()> {
    for ancestor in path.ancestors() {
        if !ancestor.as_os_str().is_empty() {
            plain_metadata(ancestor)?;
        }
    }
    Ok(())
}

pub(super) fn digest(path: &Path, limit: u64) -> Result<(u64, String)> {
    digest_until(path, limit, None)
}

pub(super) fn checkpoint(deadline: Option<Instant>) -> Result<()> {
    if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
        return Err("Native source operation exceeded its original work deadline.".into());
    }
    Ok(())
}

pub(super) fn digest_until(
    path: &Path,
    limit: u64,
    deadline: Option<Instant>,
) -> Result<(u64, String)> {
    checkpoint(deadline)?;
    let before = plain_metadata(path)?;
    if !before.is_file() || before.len() > limit {
        return Err("Native inventory input is not a bounded regular file.".into());
    }
    let mut file = File::open(path).map_err(|error| error.to_string())?;
    let mut hash = Hasher::new().map_err(|error| error.to_string())?;
    let mut length = 0_u64;
    let mut buffer = [0_u8; 16 * 1024];
    loop {
        checkpoint(deadline)?;
        let count = file.read(&mut buffer).map_err(|error| error.to_string())?;
        if count == 0 {
            break;
        }
        length = length
            .checked_add(u64::try_from(count).map_err(|error| error.to_string())?)
            .filter(|length| *length <= limit)
            .ok_or("Native inventory input exceeded its byte bound.")?;
        hash.update(&buffer[..count])
            .map_err(|error| error.to_string())?;
    }
    let after = plain_metadata(path)?;
    checkpoint(deadline)?;
    if length != before.len() || length != after.len() {
        return Err("Native inventory input length changed during hashing.".into());
    }
    let digest = hash.finish().map_err(|error| error.to_string())?;
    Ok((length, hex(&digest)))
}

pub(super) fn hex(bytes: &[u8]) -> String {
    hex::encode(bytes)
}

pub(super) fn read_bounded(path: &Path, limit: u64) -> Result<Vec<u8>> {
    read_bounded_until(path, limit, None)
}

pub(super) fn read_bounded_until(
    path: &Path,
    limit: u64,
    deadline: Option<Instant>,
) -> Result<Vec<u8>> {
    checkpoint(deadline)?;
    let metadata = plain_metadata(path)?;
    if !metadata.is_file() || metadata.len() > limit {
        return Err("Native inventory input is not a bounded regular file.".into());
    }
    let file = File::open(path).map_err(|error| error.to_string())?;
    crate::impact::workspace::read_bounded_until_reader(
        file,
        limit,
        "Native inventory input grew beyond its byte bound.",
        || checkpoint(deadline),
    )
}

pub(super) use crate::impact::workspace::is_lower_hex as lower_hex;

fn checked_entries(inventory: &Inventory) -> Result<BTreeMap<&str, &Entry>> {
    if inventory.schema_version != 1
        || inventory.scope != "complete-registry-package-files-not-compiled-translation-units"
        || inventory.package.name != "transcribe-cpp-sys"
        || inventory.package.version != "0.4.1"
        || inventory.package.registry != "registry+https://github.com/rust-lang/crates.io-index"
        || inventory.package.repository != "https://github.com/handy-computer/transcribe.cpp"
        || inventory.package.revision != "4d7c1ccefd39e5a9ea36cca5f8fefc6dcc14b99b"
        || !lower_hex(&inventory.package.archive_sha256, 64)
        || inventory.excluded_local_markers != [".cargo-ok"]
        || inventory.files.is_empty()
        || inventory.files.len() > MAX_FILES
    {
        return Err("Unsupported native source inventory identity, scope or bounds.".into());
    }
    let mut entries = BTreeMap::new();
    let mut folded = BTreeSet::new();
    let mut total = 0_u64;
    let mut previous = "";
    for entry in &inventory.files {
        let path = Path::new(&entry.path);
        if entry.path.is_empty()
            || entry.path.len() > 512
            || !entry.path.is_ascii()
            || entry.path.contains(['\\', ':'])
            || entry
                .path
                .split('/')
                .any(|part| part.is_empty() || part == "." || part == "..")
            || !path
                .components()
                .all(|part| matches!(part, Component::Normal(_)))
            || !lower_hex(&entry.sha256, 64)
            || entry.bytes > MAX_FILE_BYTES
            || entry.path.as_str() <= previous
            || !folded.insert(entry.path.to_ascii_lowercase())
            || inventory.excluded_local_markers.contains(&entry.path)
        {
            return Err("Invalid, duplicate or unordered native source inventory entry.".into());
        }
        total = total
            .checked_add(entry.bytes)
            .filter(|total| *total <= MAX_TOTAL_BYTES)
            .ok_or("Native source inventory exceeds its aggregate byte bound.")?;
        previous = &entry.path;
        entries.insert(entry.path.as_str(), entry);
    }
    Ok(entries)
}

fn files(root: &Path) -> Result<BTreeSet<String>> {
    if !plain_metadata(root)?.is_dir() {
        return Err("Native source package must be a directory.".into());
    }
    let mut pending = vec![root.to_path_buf()];
    let mut found = BTreeSet::new();
    let mut count = 0_usize;
    while let Some(directory) = pending.pop() {
        for entry in fs::read_dir(directory).map_err(|error| error.to_string())? {
            let entry = entry.map_err(|error| error.to_string())?;
            let path = entry.path();
            let metadata = plain_metadata(&path)?;
            count += 1;
            if count > MAX_FILES * 2 {
                return Err("Native source package enumeration exceeded its entry bound.".into());
            }
            if metadata.is_dir() {
                pending.push(path);
            } else if metadata.is_file() {
                let relative = path
                    .strip_prefix(root)
                    .map_err(|error| error.to_string())?
                    .to_str()
                    .ok_or("Native source inventory requires UTF-8 file names.")?
                    .replace('\\', "/");
                found.insert(relative);
            } else {
                return Err("Native source package contains a nonregular entry.".into());
            }
        }
    }
    Ok(found)
}

fn verify(inventory: &Inventory, source: &Path) -> Result<u64> {
    let expected = checked_entries(inventory)?;
    let mut actual = files(source)?;
    // Cargo creates this local extraction marker; it is not part of the archive.
    actual.remove(".cargo-ok");
    if actual.iter().map(String::as_str).collect::<BTreeSet<_>>()
        != expected.keys().copied().collect::<BTreeSet<_>>()
    {
        return Err("Native package has missing or additional inventory files.".into());
    }
    let mut total = 0_u64;
    for entry in expected.values() {
        let (length, hash) = digest(&source.join(&entry.path), MAX_FILE_BYTES)?;
        if length != entry.bytes || hash != entry.sha256 {
            return Err(format!("Native source identity differs: {}", entry.path));
        }
        total += length;
    }
    let vcs: serde_json::Value =
        serde_json::from_slice(&read_bounded(&source.join(".cargo_vcs_info.json"), 4096)?)
            .map_err(|error| error.to_string())?;
    if vcs["git"]["sha1"] != inventory.package.revision {
        return Err("Native package VCS revision differs from its inventory.".into());
    }
    Ok(total)
}

fn lock_matches(root: &Path, inventory: &Inventory) -> Result<()> {
    let lock_text = String::from_utf8(read_bounded(&root.join("Cargo.lock"), 4 * 1024 * 1024)?)
        .map_err(|error| error.to_string())?;
    // Value::from_str parses an individual TOML value in the pinned toml crate;
    // Cargo.lock is a document containing tables and must use its deserializer.
    let lock: toml::Value = toml::from_str(&lock_text)
        .map_err(|error| format!("Cannot parse native inventory Cargo.lock document: {error}"))?;
    let packages = lock
        .get("package")
        .and_then(toml::Value::as_array)
        .ok_or("Cargo.lock has no package table.")?;
    let matches = packages
        .iter()
        .filter(|package| {
            package.get("name").and_then(toml::Value::as_str) == Some(&inventory.package.name)
        })
        .collect::<Vec<_>>();
    if matches.len() != 1
        || matches[0].get("version").and_then(toml::Value::as_str)
            != Some(&inventory.package.version)
        || matches[0].get("source").and_then(toml::Value::as_str)
            != Some(&inventory.package.registry)
        || matches[0].get("checksum").and_then(toml::Value::as_str)
            != Some(&inventory.package.archive_sha256)
    {
        return Err("Cargo.lock does not match the native package inventory.".into());
    }
    Ok(())
}

pub(super) fn run(arguments: &[OsString]) -> Result<String> {
    let mut cursor = super::ArgCursor::new(arguments, USAGE);
    cursor.skip_command()?;
    let mut paths = BTreeMap::new();
    while let Some(key) = cursor.next_flag()? {
        let value = cursor.next_path()?;
        if !["--root", "--source", "--archive"].contains(&key)
            || !value.is_absolute()
            || paths.insert(key, value).is_some()
        {
            return Err(USAGE.into());
        }
    }
    let root = paths.get("--root").ok_or(USAGE)?;
    let source = paths.get("--source").ok_or(USAGE)?;
    let archive = paths.get("--archive").ok_or(USAGE)?;
    for path in [root, source, archive] {
        plain_ancestors(path)?;
    }
    let manifest = root.join(MANIFEST);
    plain_ancestors(&manifest)?;
    let manifest_bytes = read_bounded(&manifest, MAX_MANIFEST_BYTES)?;
    let manifest_hash = hex(&hash_bytes(&manifest_bytes).map_err(|error| error.to_string())?);
    let inventory: Inventory =
        serde_json::from_slice(&manifest_bytes).map_err(|error| error.to_string())?;
    checked_entries(&inventory)?;
    lock_matches(root, &inventory)?;
    if digest(archive, MAX_TOTAL_BYTES)?.1 != inventory.package.archive_sha256 {
        return Err("Native package archive differs from the locked registry checksum.".into());
    }
    let total = verify(&inventory, source)?;
    Ok(serde_json::json!({
        "schemaVersion": 1, "kind": "native-source-inventory-verification", "passed": true,
        "package": inventory.package.name, "version": inventory.package.version,
        "revision": inventory.package.revision, "archiveSha256": inventory.package.archive_sha256,
        "inventorySha256": manifest_hash, "files": inventory.files.len(), "bytes": total,
        "scope": inventory.scope, "freshNativeExecution": false, "modelExecuted": false,
        "compiledTranslationUnitClosureQualified": false, "inferenceParityQualified": false
    })
    .to_string())
}

#[cfg(test)]
mod tests {
    use super::{Inventory, checked_entries, digest, files, lock_matches, verify};
    use crate::impact::workspace::ScratchDir;
    use std::fs;

    struct Scratch(ScratchDir);

    impl Scratch {
        fn new() -> Result<Self, Box<dyn std::error::Error>> {
            Ok(Self(ScratchDir::new("Bonsai-NativeInventory")?))
        }
    }

    impl std::ops::Deref for Scratch {
        type Target = std::path::PathBuf;

        fn deref(&self) -> &Self::Target {
            &self.0.0
        }
    }

    fn fixture() -> Result<Inventory, serde_json::Error> {
        serde_json::from_str(
            r#"{
            "schemaVersion":1,
            "scope":"complete-registry-package-files-not-compiled-translation-units",
            "package":{"name":"transcribe-cpp-sys","version":"0.4.1",
              "registry":"registry+https://github.com/rust-lang/crates.io-index",
              "archiveSha256":"23a837974fdb873998a59a21ad2bf5a4b7bbf721c20d03d2a3ab8c221726a444",
              "repository":"https://github.com/handy-computer/transcribe.cpp",
              "revision":"4d7c1ccefd39e5a9ea36cca5f8fefc6dcc14b99b"},
            "excludedLocalMarkers":[".cargo-ok"],
            "files":[{"path":"a.cpp","bytes":3,"sha256":"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"}]
        }"#,
        )
    }

    #[test]
    fn native_source_inventory_rejects_traversal_duplicates_and_unknown_schema()
    -> Result<(), Box<dyn std::error::Error>> {
        assert!(checked_entries(&fixture()?).is_ok());
        for path in [
            "../a.cpp",
            "/a.cpp",
            "a//b.cpp",
            "a/./b.cpp",
            "a\\b.cpp",
            "C:a.cpp",
            ".cargo-ok",
        ] {
            let mut inventory = fixture()?;
            inventory.files[0].path = path.into();
            assert!(checked_entries(&inventory).is_err(), "{path}");
        }
        let mut inventory = fixture()?;
        inventory.schema_version = 2;
        assert!(checked_entries(&inventory).is_err());
        let mut inventory = fixture()?;
        inventory.files.push(super::Entry {
            path: "a.cpp".into(),
            bytes: 0,
            sha256: "0".repeat(64),
        });
        assert!(checked_entries(&inventory).is_err());
        inventory.files[1].path = "A.cpp".into();
        assert!(checked_entries(&inventory).is_err());
        Ok(())
    }

    #[test]
    fn native_source_inventory_rejects_missing_extra_and_changed_bytes()
    -> Result<(), Box<dyn std::error::Error>> {
        let scratch = Scratch::new()?;
        let mut inventory = fixture()?;
        fs::write(scratch.0.join("a.cpp"), b"abc")?;
        assert_eq!(
            digest(&scratch.0.join("a.cpp"), 3)?.1,
            inventory.files[0].sha256
        );
        assert!(digest(&scratch.0.join("a.cpp"), 2).is_err());
        // VCS is a required independent identity entry, not a mere arbitrary source file.
        let vcs = br#"{"git":{"sha1":"4d7c1ccefd39e5a9ea36cca5f8fefc6dcc14b99b"}}"#;
        let vcs_path = scratch.0.join(".cargo_vcs_info.json");
        fs::write(&vcs_path, vcs)?;
        let (bytes, sha256) = digest(&vcs_path, 1024)?;
        inventory.files.insert(
            0,
            super::Entry {
                path: ".cargo_vcs_info.json".into(),
                bytes,
                sha256,
            },
        );
        assert_eq!(verify(&inventory, &scratch.0)?, bytes + 3);
        fs::write(scratch.0.join(".cargo-ok"), b"local extraction marker")?;
        assert!(verify(&inventory, &scratch.0).is_ok());
        fs::write(scratch.0.join("extra.hpp"), b"extra")?;
        assert!(verify(&inventory, &scratch.0).is_err());
        fs::remove_file(scratch.0.join("extra.hpp"))?;
        fs::write(scratch.0.join("a.cpp"), b"abd")?;
        assert!(verify(&inventory, &scratch.0).is_err());
        fs::remove_file(scratch.0.join("a.cpp"))?;
        assert!(verify(&inventory, &scratch.0).is_err());
        Ok(())
    }

    #[test]
    fn native_source_inventory_rejects_linked_entries() -> Result<(), Box<dyn std::error::Error>> {
        let scratch = Scratch::new()?;
        #[cfg(unix)]
        std::os::unix::fs::symlink(std::env::temp_dir(), scratch.0.join("linked"))?;
        #[cfg(windows)]
        {
            // A junction exercises reparse rejection without Developer Mode or symlink privileges.
            let output = std::process::Command::new("cmd.exe")
                .args(["/D", "/C", "mklink", "/J"])
                .arg(scratch.0.join("linked"))
                .arg(std::env::temp_dir())
                .output()?;
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        assert!(files(&scratch.0).is_err());
        Ok(())
    }

    #[test]
    fn native_source_inventory_binds_cargo_lock_and_rejects_incomplete_records()
    -> Result<(), Box<dyn std::error::Error>> {
        let scratch = Scratch::new()?;
        let inventory = fixture()?;
        let lock = format!(
            "version = 4\n[[package]]\nname = \"{}\"\nversion = \"{}\"\nsource = \"{}\"\nchecksum = \"{}\"\n",
            inventory.package.name,
            inventory.package.version,
            inventory.package.registry,
            inventory.package.archive_sha256,
        );
        fs::write(scratch.0.join("Cargo.lock"), &lock)?;
        lock_matches(&scratch.0, &inventory)?;
        for changed in [
            "version = 4".to_owned(),
            "[[package]]\nname = \"transcribe-cpp-sys\"".to_owned(),
            lock.replace(&inventory.package.archive_sha256, &"0".repeat(64)),
            lock.replace("0.4.1", "0.4.2"),
            format!("{lock}[[package]]\nname = \"transcribe-cpp-sys\"\n"),
        ] {
            fs::write(scratch.0.join("Cargo.lock"), changed)?;
            assert!(lock_matches(&scratch.0, &inventory).is_err());
        }
        Ok(())
    }
}
