//! Bounded static ARM64 PE dependency inspection; no image is loaded or executed.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, OpenOptions};
use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::{Value, json};

use crate::Result;

type Aliases = BTreeMap<String, Vec<(String, String)>>;
const GUI_MODULES: &[&str] = &[
    "shell32.dll",
    "user32.dll",
    "gdi32.dll",
    "gdi32full.dll",
    "win32u.dll",
    "comdlg32.dll",
    "comctl32.dll",
    "dwmapi.dll",
    "uxtheme.dll",
    "imm32.dll",
];

struct Budget(Instant);

impl Budget {
    fn checkpoint(&self) -> Result<()> {
        crate::startup_cancel::checkpoint()?;
        if self.0.elapsed() >= Duration::from_secs(120) {
            return Err("Static PE audit exceeded its 120-second allowance".into());
        }
        Ok(())
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Request {
    bundle_directory: PathBuf,
    system_directory: PathBuf,
    aliases: Option<Aliases>,
}

use crate::pe_format::{PeLayout, PePolicy, extent, u32_at, usize_at as offset_at};

pub(crate) struct Image {
    data: Vec<u8>,
    layout: PeLayout,
}

fn ordinary_directory(path: &Path) -> Result<()> {
    let metadata = fs::symlink_metadata(path).map_err(|error| error.to_string())?;
    if !metadata.is_dir() || bonsai_artifacts::is_redirect(&metadata) {
        return Err("Redirected audit directory".into());
    }
    Ok(())
}

fn read_image(path: &Path, budget: &Budget) -> Result<Image> {
    budget.checkpoint()?;
    let metadata = fs::symlink_metadata(path).map_err(|error| error.to_string())?;
    if !metadata.is_file() || bonsai_artifacts::is_redirect(&metadata) {
        return Err("Dependency must be a regular file".into());
    }
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt as _;
        options.share_mode(1).custom_flags(0x0020_0000);
    }
    let file = options.open(path).map_err(|error| error.to_string())?;
    let metadata = file.metadata().map_err(|error| error.to_string())?;
    if !metadata.is_file() || bonsai_artifacts::is_redirect(&metadata) {
        return Err("Dependency must be a regular file".into());
    }
    if metadata.len() > 33_554_432 {
        return Err("PE exceeds static read bound".into());
    }
    let mut reader = file.take(33_554_433);
    let mut data = Vec::new();
    let mut buffer = vec![0; 65_536];
    loop {
        budget.checkpoint()?;
        let count = reader
            .read(&mut buffer)
            .map_err(|error| error.to_string())?;
        if count == 0 {
            break;
        }
        data.extend_from_slice(&buffer[..count]);
        if data.len() > 33_554_432 {
            return Err("PE exceeds static read bound".into());
        }
    }
    parse_image(data)
}

pub(crate) fn parse_image(data: Vec<u8>) -> Result<Image> {
    let layout = PeLayout::parse(&mut data.as_slice(), &PePolicy::STATIC_AUDIT)?;
    Ok(Image { data, layout })
}

fn mapped(image: &Image, rva: u64, size: usize) -> Result<(usize, usize)> {
    let (offset, available) = image.layout.map_rva(rva, size, false)?;
    Ok((
        usize::try_from(offset).map_err(|error| error.to_string())?,
        available,
    ))
}

fn safe_module(name: &str) -> bool {
    name.len() > 4
        && Path::new(name)
            .extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("dll"))
        && name.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || b"_.-".contains(&byte)
        })
}

fn module(image: &Image, rva: u64) -> Result<String> {
    let mut scratch = [0_u8; 260];
    let bytes = image.layout.read_c_bytes_at(
        &mut image.data.as_slice(),
        rva,
        false,
        &mut scratch,
        "Unterminated import name",
    )?;
    let name = std::str::from_utf8(bytes)
        .map_err(|_| "Unsafe import module name")?
        .to_ascii_lowercase();
    if !safe_module(&name) {
        return Err("Unsafe import module name".into());
    }
    Ok(name)
}

pub(crate) fn imports(image: &Image, delayed: bool) -> Result<Vec<String>> {
    let mut result = Vec::new();
    image.layout.visit_import_descriptors::<_, String>(
        &mut image.data.as_slice(),
        delayed,
        65_536,
        65_536,
        false,
        |_, descriptor| {
            let rva = image.layout.resolve_import_name_rva(
                delayed,
                descriptor.attributes,
                descriptor.name_raw,
                false,
            )?;
            result.push(module(image, rva)?);
            Ok(())
        },
    )?;
    Ok(result)
}

pub(crate) enum ImportSymbol<'a> {
    Name(&'a str),
    Ordinal(u16),
}

pub(crate) fn visit_import_symbols(
    image: &Image,
    delayed: bool,
    visitor: impl FnMut(&str, ImportSymbol<'_>) -> Result<()>,
) -> Result<usize> {
    symbol_imports::visit_import_symbols(image, delayed, visitor)
}

mod symbol_imports;

pub(crate) mod symbol_exports;

fn api_text(data: &[u8], offset: usize, length: usize) -> Result<String> {
    if !length.is_multiple_of(2) {
        return Err("Invalid API-set name extent".into());
    }
    let units: Vec<_> = extent(data, offset, length)?
        .as_chunks::<2>()
        .0
        .iter()
        .map(|unit| u16::from_le_bytes(*unit))
        .collect();
    String::from_utf16(&units)
        .map(|value| value.to_lowercase())
        .map_err(|error| error.to_string())
}

fn api_sets(system: &Path, budget: &Budget) -> Result<Aliases> {
    let image = read_image(&system.join("apisetschema.dll"), budget)?;
    let sections: Vec<_> = image
        .layout
        .sections()
        .iter()
        .filter(|section| section.name_bytes() == b".apiset")
        .collect();
    let [section] = sections.as_slice() else {
        return Err("Expected one API-set namespace".into());
    };
    let raw = usize::try_from(section.raw_offset).map_err(|error| error.to_string())?;
    let data = extent(&image.data, raw, section.raw_size)?;
    let size = offset_at(data, 4)?;
    if data.len() < 28 || u32_at(data, 0)? != 6 || !(28..=data.len()).contains(&size) {
        return Err("Unsupported API-set namespace".into());
    }
    let data = &data[..size];
    let count = offset_at(data, 12)?;
    let offset = offset_at(data, 16)?;
    if count > 4096 || offset < 28 {
        return Err("API-set entry table outside declared namespace".into());
    }
    extent(data, offset, count * 24)?;
    let mut result = Aliases::new();
    for index in 0..count {
        budget.checkpoint()?;
        let entry = offset + index * 24;
        let length = offset_at(data, entry + 8)?;
        let hashed_length = offset_at(data, entry + 12)?;
        if hashed_length == 0 || hashed_length > length {
            return Err("Invalid API-set hashed name extent".into());
        }
        let name_offset = offset_at(data, entry + 4)?;
        let name = api_text(data, name_offset, length)? + ".dll";
        let hashed = api_text(data, name_offset, hashed_length)? + "-*.dll";
        let values = offset_at(data, entry + 16)?;
        let host_count = offset_at(data, entry + 20)?;
        if host_count > 64 {
            return Err("API-set host table outside declared namespace".into());
        }
        extent(data, values, host_count * 20)?;
        let mut hosts = Vec::new();
        for host in 0..host_count {
            let value = values + host * 20;
            hosts.push((
                api_text(
                    data,
                    offset_at(data, value + 4)?,
                    offset_at(data, value + 8)?,
                )?,
                api_text(
                    data,
                    offset_at(data, value + 12)?,
                    offset_at(data, value + 16)?,
                )?,
            ));
        }
        result.insert(name, hosts.clone());
        if result
            .get(&hashed)
            .is_some_and(|previous| previous != &hosts)
        {
            return Err("Ambiguous API-set hashed contract".into());
        }
        result.insert(hashed, hosts);
    }
    Ok(result)
}

fn resolve<'a>(aliases: &'a Aliases, dependency: &'a str, importer: &str) -> Result<&'a str> {
    let hashed = dependency
        .rsplit_once('-')
        .map(|(prefix, _)| format!("{prefix}-*.dll"));
    let hosts = aliases
        .get(dependency)
        .or_else(|| hashed.as_ref().and_then(|key| aliases.get(key)));
    let Some(hosts) = hosts else {
        return Ok(dependency);
    };
    let mut matches: BTreeSet<_> = hosts
        .iter()
        .filter(|(name, _)| {
            name == importer || Some(name.as_str()) == importer.strip_suffix(".dll")
        })
        .map(|(_, host)| host.as_str())
        .collect();
    if matches.is_empty() {
        matches = hosts
            .iter()
            .filter(|(name, _)| name.is_empty())
            .map(|(_, host)| host.as_str())
            .collect();
    }
    if matches.len() != 1 {
        return Err(format!("Unresolved API-set host: {dependency}"));
    }
    matches
        .into_iter()
        .next()
        .filter(|host| !host.is_empty())
        .ok_or_else(|| format!("Unresolved API-set host: {dependency}"))
}

fn candidate_inventory(directory: &Path, budget: &Budget) -> Result<BTreeMap<String, PathBuf>> {
    let mut candidates = BTreeMap::new();
    for entry in fs::read_dir(directory).map_err(|error| error.to_string())? {
        budget.checkpoint()?;
        let path = entry.map_err(|error| error.to_string())?.path();
        if path.extension().is_some_and(|extension| {
            extension.eq_ignore_ascii_case("dll") || extension.eq_ignore_ascii_case("exe")
        }) {
            let name = path
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or("Non-Unicode candidate module")?
                .to_ascii_lowercase();
            if candidates.insert(name, path).is_some() {
                return Err("Ambiguous candidate module name".into());
            }
            if candidates.len() > 64 {
                return Err("Candidate inventory outside bound".into());
            }
        }
    }
    if candidates.is_empty() {
        return Err("Candidate inventory outside bound".into());
    }
    Ok(candidates)
}

fn normalize_aliases(aliases: Aliases) -> Result<Aliases> {
    if aliases.len() > 8192 || aliases.values().any(|hosts| hosts.len() > 64) {
        return Err("API-set inventory outside bound".into());
    }
    let mut result = Aliases::new();
    for (name, hosts) in aliases {
        let key = name.to_ascii_lowercase();
        if result.get(&key).is_some_and(|previous| previous != &hosts) {
            return Err("Ambiguous API-set contract casing".into());
        }
        result.insert(key, hosts);
    }
    Ok(result)
}

fn audit(request: Request) -> Result<Value> {
    let budget = Budget(Instant::now());
    ordinary_directory(&request.bundle_directory)?;
    ordinary_directory(&request.system_directory)?;
    let candidates = candidate_inventory(&request.bundle_directory, &budget)?;
    let aliases = normalize_aliases(
        request
            .aliases
            .map_or_else(|| api_sets(&request.system_directory, &budget), Ok)?,
    )?;
    let mut pending: Vec<_> = candidates
        .keys()
        .map(|name| (name.clone(), "candidate"))
        .collect();
    let mut modules = BTreeMap::new();
    let mut edges = Vec::new();
    while let Some((name, location)) = pending.pop() {
        budget.checkpoint()?;
        let key = (location, name.clone());
        if modules.contains_key(&key) {
            continue;
        }
        if GUI_MODULES.contains(&name.as_str()) {
            return Err(format!("GUI dependency: {name}"));
        }
        if modules.len() >= 128 {
            return Err("Dependency closure exceeds module bound".into());
        }
        let path = if location == "candidate" {
            candidates
                .get(&name)
                .ok_or("Missing candidate module")?
                .clone()
        } else {
            request.system_directory.join(&name)
        };
        let image = read_image(&path, &budget)?;
        let normal = imports(&image, false)?;
        let delayed = imports(&image, true)?;
        let mut hasher = crate::identity::Hasher::new()?;
        hasher.update(&image.data)?;
        let file_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or("Non-Unicode module name")?;
        modules.insert(key, json!({"name":file_name, "bytes":image.data.len(), "sha256":hasher.finalize_hex()?, "machine":"0xAA64", "normalImports":normal, "delayImports":delayed, "location":location}));
        for dependency in normal
            .iter()
            .chain(delayed.iter().filter(|_| location == "candidate"))
        {
            budget.checkpoint()?;
            let resolved = resolve(&aliases, dependency, &name)?;
            if !safe_module(resolved) {
                return Err("Unsafe resolved module name".into());
            }
            let destination = if candidates.contains_key(resolved) {
                "candidate"
            } else {
                "system"
            };
            edges.push(
                json!({"from":name,"import":dependency,"resolved":resolved,"location":destination}),
            );
            pending.push((resolved.into(), destination));
        }
    }
    budget.checkpoint()?;
    Ok(
        json!({"schemaVersion":1,"status":"candidate-and-mandatory-named-import-closure-gui-free",
        "scope":"All candidate normal/delay imports and recursive system normal imports; optional OS delays are recorded, not followed or claimed absent. No DLL was loaded.",
        "forwardedExportsResolved":false,"modules":modules.into_values().collect::<Vec<_>>(),"requiredEdges":edges,"modelExecuted":false,"workerPolicyChanged":false}),
    )
}

/// Inspect candidate normal/delay imports and recursive system normal imports.
///
/// An omitted or null `aliases` reads the system's API-set namespace. Explicit
/// aliases support independently supplied replay fixtures without loading a DLL.
///
/// # Errors
/// Rejects malformed requests, redirected inputs, unsupported images, extent or
/// inventory bounds, unresolved dependencies and mandatory GUI imports.
pub fn run(request: Value) -> Result<Value> {
    audit(serde_json::from_value(request).map_err(|error| error.to_string())?)
}

#[cfg(test)]
mod tests;
