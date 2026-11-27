//! Source-only Cargo workspace graph discovery.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path, PathBuf};

use toml::Value;

pub(crate) struct Package {
    pub(crate) path: PathBuf,
    pub(crate) directory: String,
    manifest: Value,
}

pub(crate) struct Workspace {
    pub(crate) packages: BTreeMap<String, Package>,
    pub(crate) reverse: BTreeMap<String, BTreeSet<String>>,
}

fn manifest(path: &Path) -> crate::Result<Value> {
    let text =
        std::fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
    toml::from_str(&text).map_err(|error| format!("{}: {error}", path.display()))
}

fn strings<'a>(value: Option<&'a Value>, name: &str) -> crate::Result<Vec<&'a str>> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    value
        .as_array()
        .ok_or_else(|| format!("{name} must be an array"))?
        .iter()
        .map(|item| {
            item.as_str()
                .ok_or_else(|| format!("{name} must contain strings"))
        })
        .collect()
}

fn relative(root: &Path, directory: &Path) -> crate::Result<String> {
    let path = directory
        .strip_prefix(root)
        .map_err(|error| error.to_string())?;
    if path.as_os_str().is_empty() {
        return Ok(".".into());
    }
    path.to_str()
        .map(|text| text.replace('\\', "/"))
        .ok_or_else(|| "Workspace path is not UTF-8".into())
}

fn excluded(root: &Path, directory: &Path, excludes: &[&str]) -> crate::Result<bool> {
    let relative = relative(root, directory)?;
    Ok(excludes
        .iter()
        .any(|pattern| super::matches(&relative, pattern)))
}

fn class_matches(character: char, pattern: &[char]) -> bool {
    let (negative, pattern) = if pattern.first() == Some(&'!') {
        (true, &pattern[1..])
    } else {
        (false, pattern)
    };
    let mut found = false;
    let mut index = 0;
    while index < pattern.len() {
        if index + 2 < pattern.len() && pattern[index + 1] == '-' {
            found |= (pattern[index]..=pattern[index + 2]).contains(&character);
            index += 3;
        } else {
            found |= pattern[index] == character;
            index += 1;
        }
    }
    found != negative
}

fn member_matches(name: &str, pattern: &str) -> bool {
    let name: Vec<_> = name.chars().collect();
    let pattern: Vec<_> = pattern.chars().collect();
    let mut previous = vec![false; name.len() + 1];
    previous[0] = true;
    let mut index = 0;
    while index < pattern.len() {
        let character = pattern[index];
        let class_end = if character == '[' {
            let start = index + 1 + usize::from(pattern.get(index + 1) == Some(&'!'));
            let start = start + usize::from(pattern.get(start) == Some(&']'));
            pattern.get(start..).and_then(|suffix| {
                suffix
                    .iter()
                    .position(|item| *item == ']')
                    .map(|offset| start + offset)
            })
        } else {
            None
        };
        let mut next = vec![false; previous.len()];
        next[0] = character == '*' && previous[0];
        for (position, &actual) in name.iter().enumerate() {
            next[position + 1] = match (character, class_end) {
                ('*', _) => previous[position + 1] || next[position],
                ('?', _) => previous[position],
                ('[', Some(end)) => {
                    previous[position] && class_matches(actual, &pattern[index + 1..end])
                }
                _ => previous[position] && actual == character,
            };
        }
        index = class_end.map_or(index + 1, |end| end + 1);
        previous = next;
    }
    previous[name.len()]
}

fn children(directory: &Path) -> crate::Result<Vec<PathBuf>> {
    let mut result = Vec::new();
    for entry in
        std::fs::read_dir(directory).map_err(|error| format!("{}: {error}", directory.display()))?
    {
        let entry = entry.map_err(|error| error.to_string())?;
        if entry.path().is_dir() {
            result.push(entry.path());
        }
    }
    result.sort();
    Ok(result)
}

fn expand(directory: &Path, parts: &[&str], output: &mut BTreeSet<PathBuf>) -> crate::Result<()> {
    let Some((part, remaining)) = parts.split_first() else {
        if directory.exists() {
            output.insert(directory.to_path_buf());
        }
        return Ok(());
    };
    if *part == "**" {
        expand(directory, remaining, output)?;
        if directory.is_dir() {
            for child in children(directory)? {
                // Path.glob does not descend through symbolic-link directories.
                if !child
                    .symlink_metadata()
                    .map_err(|error| error.to_string())?
                    .file_type()
                    .is_symlink()
                {
                    expand(&child, parts, output)?;
                }
            }
        }
    } else if part.contains(['*', '?', '[']) {
        if directory.is_dir() {
            for child in children(directory)? {
                if child
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| member_matches(name, part))
                {
                    expand(&child, remaining, output)?;
                }
            }
        }
    } else {
        expand(&directory.join(part), remaining, output)?;
    }
    Ok(())
}

fn dependencies(
    data: &Value,
    base: &Path,
    root: &Path,
    settings: Option<&Value>,
) -> crate::Result<Vec<PathBuf>> {
    let mut tables = vec![data];
    if let Some(targets) = data.get("target").and_then(Value::as_table) {
        tables.extend(targets.values());
    }
    let mut result = Vec::new();
    for table in tables {
        for kind in ["dependencies", "build-dependencies", "dev-dependencies"] {
            let Some(dependencies) = table.get(kind).and_then(Value::as_table) else {
                continue;
            };
            for (alias, dependency) in dependencies {
                if !dependency.is_table() {
                    continue;
                }
                let (dependency, directory) =
                    if dependency.get("workspace").and_then(Value::as_bool) == Some(true) {
                        (
                            settings
                                .and_then(|settings| settings.get("dependencies"))
                                .and_then(|dependencies| dependencies.get(alias))
                                .ok_or_else(|| format!("Missing workspace dependency {alias}"))?,
                            root,
                        )
                    } else {
                        (dependency, base)
                    };
                if let Some(path) = dependency.get("path") {
                    let path = path
                        .as_str()
                        .ok_or_else(|| format!("Dependency {alias} path must be a string"))?;
                    let directory = directory.join(path);
                    result.push(resolve(&directory)?);
                }
            }
        }
    }
    Ok(result)
}

fn resolve(path: &Path) -> crate::Result<PathBuf> {
    match path.canonicalize() {
        Ok(path) => return Ok(path),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => (),
        Err(error) => return Err(format!("{}: {error}", path.display())),
    }
    // Resolve existing ancestors without requiring an unrelated external path
    // dependency to be installed. Manifest discovery only follows local members.
    let parent = path
        .parent()
        .ok_or_else(|| format!("Cannot resolve {}", path.display()))?;
    let mut result = resolve(parent)?;
    match path.components().next_back() {
        Some(Component::ParentDir) => {
            result.pop();
        }
        Some(Component::CurDir) => (),
        Some(Component::Normal(name)) => result.push(name),
        _ => return Err(format!("Cannot resolve {}", path.display())),
    }
    Ok(result)
}

pub(crate) fn read(root: &Path) -> crate::Result<Workspace> {
    let root_manifest = manifest(&root.join("Cargo.toml"))?;
    let settings = root_manifest.get("workspace");
    let excludes = strings(
        settings.and_then(|value| value.get("exclude")),
        "workspace.exclude",
    )?;
    let mut manifests = Vec::new();
    if root_manifest.get("package").is_some() {
        manifests.push(root.join("Cargo.toml"));
    }
    for pattern in strings(
        settings.and_then(|value| value.get("members")),
        "workspace.members",
    )? {
        if Path::new(pattern).is_absolute() {
            return Err("Workspace members must be relative paths".into());
        }
        let mut directories = BTreeSet::new();
        expand(
            root,
            &pattern.split('/').collect::<Vec<_>>(),
            &mut directories,
        )?;
        for directory in directories {
            let directory = directory
                .canonicalize()
                .map_err(|error| error.to_string())?;
            if !excluded(root, &directory, &excludes)? {
                let candidate = directory.join("Cargo.toml");
                if !manifests.contains(&candidate) {
                    manifests.push(candidate);
                }
            }
        }
    }
    let mut packages = BTreeMap::new();
    let mut index = 0;
    while let Some(path) = manifests.get(index).cloned() {
        let data = manifest(&path)?;
        let name = data
            .get("package")
            .and_then(|package| package.get("name"))
            .and_then(Value::as_str)
            .ok_or_else(|| format!("{} has no package.name", path.display()))?
            .to_owned();
        if packages.contains_key(&name) {
            return Err(format!("Duplicate workspace package: {name}"));
        }
        let base = path
            .parent()
            .ok_or_else(|| "Manifest has no parent directory".to_owned())?;
        for directory in dependencies(&data, base, root, settings)? {
            if directory.starts_with(root) && !excluded(root, &directory, &excludes)? {
                let candidate = directory.join("Cargo.toml");
                if !manifests.contains(&candidate) {
                    manifests.push(candidate);
                }
            }
        }
        packages.insert(
            name,
            Package {
                path: base.to_path_buf(),
                directory: relative(root, base)?,
                manifest: data,
            },
        );
        index += 1;
    }
    let by_directory: BTreeMap<_, _> = packages
        .iter()
        .map(|(name, package)| (&package.path, name))
        .collect();
    let mut reverse: BTreeMap<_, BTreeSet<String>> = packages
        .keys()
        .map(|name| (name.clone(), BTreeSet::new()))
        .collect();
    for (name, package) in &packages {
        for directory in dependencies(&package.manifest, &package.path, root, settings)? {
            if let Some(dependency) = by_directory.get(&directory) {
                reverse
                    .entry((*dependency).clone())
                    .or_default()
                    .insert(name.clone());
            }
        }
    }
    Ok(Workspace { packages, reverse })
}

pub(crate) fn is_link(metadata: &std::fs::Metadata) -> bool {
    if metadata.file_type().is_symlink() {
        return true;
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt as _;
        metadata.file_attributes() & 0x400 != 0
    }
    #[cfg(not(windows))]
    {
        false
    }
}

pub(crate) fn read_bounded_until_reader<R: std::io::Read, F: FnMut() -> crate::Result<()>>(
    mut reader: R,
    limit: u64,
    exceeded: &str,
    mut checkpoint: F,
) -> crate::Result<Vec<u8>> {
    checkpoint()?;
    let mut bytes = Vec::new();
    let mut buffer = [0_u8; 16 * 1024];
    loop {
        checkpoint()?;
        let count = match reader.read(&mut buffer) {
            Ok(count) => count,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error.to_string()),
        };
        if count == 0 {
            break;
        }
        if u64::try_from(bytes.len())
            .map_err(|error| error.to_string())?
            .checked_add(u64::try_from(count).map_err(|error| error.to_string())?)
            .is_none_or(|length| length > limit)
        {
            return Err(exceeded.into());
        }
        bytes.extend_from_slice(&buffer[..count]);
    }
    checkpoint()?;
    Ok(bytes)
}

pub(crate) fn read_bounded_reader<R: std::io::Read>(
    reader: R,
    limit: u64,
    exceeded: &str,
) -> crate::Result<Vec<u8>> {
    read_bounded_until_reader(reader, limit, exceeded, || Ok(()))
}

pub(crate) fn read_bounded(path: &Path, limit: u64, exceeded: &str) -> crate::Result<Vec<u8>> {
    let file = std::fs::File::open(path).map_err(|error| error.to_string())?;
    let metadata = file.metadata().map_err(|error| error.to_string())?;
    if !metadata.is_file() {
        return Err(format!("{} is not a regular file.", path.display()));
    }
    if metadata.len() > limit {
        return Err(exceeded.into());
    }
    read_bounded_reader(file, limit, exceeded)
}

pub(crate) fn is_lower_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
}

pub(crate) fn validate_repo_relative_path(path: &str) -> bool {
    !path.is_empty()
        && path.len() <= 1024
        && path.is_ascii()
        && !path.bytes().any(|byte| byte.is_ascii_control())
        && !path.contains(['\\', ':', '<', '>', '"', '|', '?', '*'])
        && path.split('/').all(|part| {
            let stem = part.split('.').next().unwrap_or_default();
            let reserved = ["CON", "PRN", "AUX", "NUL"]
                .iter()
                .any(|name| stem.eq_ignore_ascii_case(name))
                || matches!(
                    stem.as_bytes(),
                    [prefix @ .., b'1'..=b'9']
                        if prefix.eq_ignore_ascii_case(b"COM")
                            || prefix.eq_ignore_ascii_case(b"LPT")
                );
            !part.is_empty()
                && part != "."
                && part != ".."
                && !part.ends_with(['.', ' '])
                && !reserved
        })
        && Path::new(path)
            .components()
            .all(|part| matches!(part, Component::Normal(_)))
}

pub(crate) fn is_dated_nightly_channel(channel: &str) -> bool {
    channel.len() == 18
        && channel.starts_with("nightly-")
        && channel.bytes().enumerate().skip(8).all(|(index, byte)| {
            if matches!(index, 12 | 15) {
                byte == b'-'
            } else {
                byte.is_ascii_digit()
            }
        })
}

#[cfg(windows)]
pub(crate) fn pinned_toolchain_channel(root: &Path) -> crate::Result<String> {
    let bytes = read_bounded(
        &root.join("rust-toolchain.toml"),
        16 * 1024,
        "Toolchain manifest exceeds its byte bound.",
    )?;
    let text = String::from_utf8(bytes).map_err(|error| error.to_string())?;
    let pin: Value = toml::from_str(&text).map_err(|error| error.to_string())?;
    pin.get("toolchain")
        .and_then(|value| value.get("channel"))
        .and_then(Value::as_str)
        .filter(|channel| is_dated_nightly_channel(channel))
        .map(str::to_owned)
        .ok_or_else(|| "Toolchain pin requires a dated nightly channel.".into())
}

pub(crate) fn paths_overlap(a: &Path, b: &Path) -> bool {
    let normalized = |path: &Path| {
        path.to_string_lossy()
            .replace('/', "\\")
            .trim_end_matches('\\')
            .to_uppercase()
    };
    let a = normalized(a);
    let b = normalized(b);
    a == b || a.starts_with(&(b.clone() + "\\")) || b.starts_with(&(a + "\\"))
}

#[cfg(test)]
pub(crate) struct ScratchDir(pub(crate) PathBuf);

#[cfg(test)]
impl ScratchDir {
    pub(crate) fn new(prefix: &str) -> crate::Result<Self> {
        static SEQUENCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let canonical = std::fs::canonicalize(std::env::temp_dir()).map_err(|e| e.to_string())?;
        // Use ordinary drive syntax on Windows so process controls that reject \\?\ work seamlessly.
        let base_text = canonical.to_string_lossy();
        let base = PathBuf::from(
            base_text
                .strip_prefix("\\\\?\\")
                .filter(|stripped| stripped.as_bytes().get(1) == Some(&b':'))
                .unwrap_or(&base_text),
        );
        let root = base.join(format!(
            "{prefix}-{}-{}",
            std::process::id(),
            SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir(&root).map_err(|error| error.to_string())?;
        Ok(Self(root))
    }

    pub(crate) fn path(&self) -> &Path {
        &self.0
    }
}

#[cfg(test)]
impl AsRef<Path> for ScratchDir {
    fn as_ref(&self) -> &Path {
        &self.0
    }
}

#[cfg(test)]
impl std::ops::Deref for ScratchDir {
    type Target = Path;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[cfg(test)]
impl Drop for ScratchDir {
    fn drop(&mut self) {
        if let Ok(parent) = std::fs::canonicalize(std::env::temp_dir())
            && let Ok(target) = std::fs::canonicalize(&self.0)
            && target.parent() == Some(parent.as_path())
            && std::fs::symlink_metadata(&self.0)
                .is_ok_and(|metadata| metadata.is_dir() && !is_link(&metadata))
        {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::{
        ScratchDir, is_dated_nightly_channel, is_lower_hex, paths_overlap, read_bounded,
        validate_repo_relative_path,
    };

    #[test]
    fn shared_workspace_validators_and_scratch_dir_enforce_bounds() -> crate::Result<()> {
        assert!(is_lower_hex(&"0123456789abcdef".repeat(4), 64));
        assert!(!is_lower_hex(&"0123456789ABCDEF".repeat(4), 64));
        assert!(is_dated_nightly_channel("nightly-2026-11-26"));
        assert!(!is_dated_nightly_channel("nightly-2026-11"));
        assert!(validate_repo_relative_path("tools/ci/src/lib.rs"));
        assert!(!validate_repo_relative_path("../escape.rs"));
        assert!(!validate_repo_relative_path("a/LPT1.txt"));
        assert!(paths_overlap(
            Path::new(r"C:\src\repo"),
            Path::new("c:/src/repo/target")
        ));
        assert!(!paths_overlap(
            Path::new(r"C:\src\repo"),
            Path::new(r"C:\src\repo-target")
        ));

        let scratch = ScratchDir::new("Bonsai-WorkspaceHelper")?;
        let root = scratch.path().to_path_buf();
        let file = scratch.join("bounded.txt");
        std::fs::write(&file, b"hello").map_err(|error| error.to_string())?;
        assert_eq!(read_bounded(&file, 5, "too big")?, b"hello");
        assert!(read_bounded(&file, 4, "too big").is_err());
        drop(scratch);
        assert!(!root.exists());
        Ok(())
    }

    #[cfg(windows)]
    #[test]
    fn pinned_toolchain_channel_reads_dated_nightly_channel() -> crate::Result<()> {
        let scratch = ScratchDir::new("Bonsai-ToolchainChannel")?;
        std::fs::write(
            scratch.join("rust-toolchain.toml"),
            "[toolchain]\nchannel = \"nightly-2026-11-26\"\n",
        )
        .map_err(|error| error.to_string())?;
        assert_eq!(
            super::pinned_toolchain_channel(scratch.path())?,
            "nightly-2026-11-26"
        );
        Ok(())
    }
}
