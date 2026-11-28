//! Compiled release identity for the independently built inference server.
#![expect(
    clippy::redundant_pub_crate,
    reason = "Explicit parent scope keeps these implementation helpers private."
)]

use std::{
    path::{Path, PathBuf},
    sync::LazyLock,
};

use bonsai_artifacts::VerifiedFile;
use serde::Deserialize;

use crate::{Object, launcher::LaunchError};

const LOCK: &str = include_str!("../../../config/prism-inference.lock.json");
pub(super) const TEMPLATE_SHA256: &str =
    "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041";
pub(super) const MMPROJ_SHA256: &str =
    "6807ede61d570bb86ba34b756a0fa109edc33668604de867c6ea6d8f1d631903";

fn parse_enum<T: Copy>(
    value: Option<&std::ffi::OsStr>,
    variants: &[(&str, T)],
) -> Result<T, LaunchError> {
    let raw = value.and_then(std::ffi::OsStr::to_str);
    variants
        .iter()
        .find_map(|(name, variant)| (raw == Some(*name)).then_some(*variant))
        .ok_or(LaunchError::Arguments)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum KvCache {
    F16,
    Rsq4,
    Rsq4Kvf,
    Rsq4K3v,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum PromptReuseMode {
    #[default]
    Disabled,
    Latest,
    PromptEndAndLatest,
}

impl PromptReuseMode {
    pub(super) fn parse(value: Option<&std::ffi::OsStr>) -> Result<Self, LaunchError> {
        parse_enum(
            value,
            &[
                ("disabled", Self::Disabled),
                ("latest", Self::Latest),
                ("prompt-end-and-latest", Self::PromptEndAndLatest),
            ],
        )
    }

    pub(super) const fn as_wire(self) -> Option<&'static str> {
        match self {
            Self::Disabled => None,
            Self::Latest => Some("latest"),
            Self::PromptEndAndLatest => Some("prompt-end-and-latest"),
        }
    }

    pub(super) const fn enabled(self) -> bool {
        !matches!(self, Self::Disabled)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum CpuPrefillGdn {
    #[default]
    Sequential,
    WyF32,
}

impl CpuPrefillGdn {
    pub(super) fn parse(value: Option<&std::ffi::OsStr>) -> Result<Self, LaunchError> {
        parse_enum(
            value,
            &[("sequential", Self::Sequential), ("wy-f32", Self::WyF32)],
        )
    }

    pub(super) const fn is_wy_f32(self) -> bool {
        matches!(self, Self::WyF32)
    }

    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::Sequential => "sequential",
            Self::WyF32 => "wy-f32",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum SpeculationType {
    #[default]
    None,
    Suffix,
}

impl SpeculationType {
    pub(super) fn parse(value: Option<&std::ffi::OsStr>) -> Result<Self, LaunchError> {
        parse_enum(value, &[("none", Self::None), ("suffix", Self::Suffix)])
    }

    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Suffix => "suffix",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CpuFamily {
    X1,
    X2,
}

impl CpuFamily {
    pub(super) fn parse(value: Option<&std::ffi::OsStr>) -> Result<Self, LaunchError> {
        parse_enum(value, &[("x1", Self::X1), ("x2", Self::X2)])
    }

    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::X1 => "x1",
            Self::X2 => "x2",
        }
    }

    pub(super) const fn default_row_workers(self) -> u8 {
        match self {
            Self::X1 | Self::X2 => 8,
        }
    }

    pub(super) const fn into_snapdragon(self) -> bonsai_cpu_policy::hardware::SnapdragonFamily {
        match self {
            Self::X1 => bonsai_cpu_policy::hardware::SnapdragonFamily::XElite,
            Self::X2 => bonsai_cpu_policy::hardware::SnapdragonFamily::X2EliteExtreme,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum Placement {
    #[default]
    Auto,
    Cpu,
    Vulkan,
    Composite,
}

impl Placement {
    pub(super) fn parse(value: Option<&std::ffi::OsStr>) -> Result<Self, LaunchError> {
        parse_enum(
            value,
            &[
                ("auto", Self::Auto),
                ("cpu", Self::Cpu),
                ("vulkan", Self::Vulkan),
                ("composite", Self::Composite),
            ],
        )
    }

    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Vulkan => "vulkan",
            Self::Composite => "composite",
        }
    }
}

pub(super) fn select_placement(
    native: bool,
    requested: Option<Placement>,
) -> Result<Placement, LaunchError> {
    if !native {
        return requested.map_or(Ok(Placement::Auto), |_| Err(LaunchError::Arguments));
    }
    Ok(requested.unwrap_or_default())
}

impl KvCache {
    pub(super) fn parse(value: Option<&std::ffi::OsStr>) -> Result<Self, LaunchError> {
        parse_enum(
            value,
            &[
                ("f16", Self::F16),
                ("rsq4", Self::Rsq4),
                ("rsq4kvf", Self::Rsq4Kvf),
                ("rsq4k3v", Self::Rsq4K3v),
            ],
        )
    }

    pub(super) const fn name(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::Rsq4 => "rsq4",
            Self::Rsq4Kvf => "rsq4kvf",
            Self::Rsq4K3v => "rsq4k3v",
        }
    }
}

pub(super) fn select_kv_cache(
    native: bool,
    requested: Option<KvCache>,
) -> Result<Option<KvCache>, LaunchError> {
    if !native {
        return if requested.is_some() {
            Err(LaunchError::Arguments)
        } else {
            Ok(None)
        };
    }
    if requested == Some(KvCache::Rsq4K3v) {
        Err(LaunchError::NativeRuntime)
    } else {
        Ok(Some(requested.unwrap_or(KvCache::Rsq4)))
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ReleasePin {
    schema_version: u8,
    package: String,
    version: String,
    repository: String,
    commit: String,
    target: String,
    release: String,
    asset: Object<Asset>,
    files: Vec<Object<File>>,
    #[serde(rename = "Source")]
    source: Object<Source>,
}

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase", deny_unknown_fields)]
struct Source {
    repository: String,
    commit: String,
    url: String,
    bytes: u64,
    sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Asset {
    filename: String,
    url: String,
    bytes: u64,
    sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct File {
    name: String,
    bytes: u64,
    sha256: String,
}

fn hex(value: &str, length: usize) -> bool {
    crate::configuration::hex(value.as_bytes(), length)
}

fn release(input: &str) -> Result<ReleasePin, LaunchError> {
    let Object(release): Object<ReleasePin> =
        serde_json::from_str(input).map_err(|_| LaunchError::Artifacts)?;
    let filename = format!("prism-inference-{}-{}.zip", release.version, release.target);
    if release.schema_version != 1
        || release.package != "prism-inference"
        || release.repository != "https://github.com/symbiogenesis/prism-inference"
        || release.target != "aarch64-pc-windows-msvc"
        || release.version.is_empty()
        || !release
            .version
            .bytes()
            .all(|byte| byte.is_ascii_digit() || byte == b'.')
        || release.release.strip_prefix('v') != Some(release.version.as_str())
        || !hex(&release.commit, 40)
        || release.asset.filename != filename
        || release.asset.url
            != format!(
                "{}/releases/download/{}/{}",
                release.repository, release.release, filename
            )
        || release.asset.bytes == 0
        || !hex(&release.asset.sha256, 64)
        || release.source.repository != release.repository
        || release.source.commit != release.commit
        || release.source.url != format!("{}/archive/{}.zip", release.repository, release.commit)
        || release.source.bytes == 0
        || release.source.bytes > 256 * 1024 * 1024
        || !hex(&release.source.sha256, 64)
        || release.files.is_empty()
    {
        return Err(LaunchError::Artifacts);
    }
    let mut names = std::collections::HashSet::new();
    for Object(file) in &release.files {
        if file.bytes == 0
            || !hex(&file.sha256, 64)
            || !names.insert(file.name.as_str())
            || ![
                "prism-inference.exe",
                "bonsai-token-worker.exe",
                "bonsai-swift-worker.exe",
                "prism-inference-server.exe",
                "bonsai-gguf.exe",
                "LICENSE",
                "NOTICE",
                "PRISM-LICENSE.txt",
                "CAMELID-LICENSE.txt",
            ]
            .contains(&file.name.as_str())
        {
            return Err(LaunchError::Artifacts);
        }
    }
    if !names.contains("prism-inference-server.exe") {
        return Err(LaunchError::Artifacts);
    }
    Ok(release)
}

static PINNED_RELEASE: LazyLock<Result<ReleasePin, LaunchError>> = LazyLock::new(|| release(LOCK));

fn pinned_release() -> Result<&'static ReleasePin, LaunchError> {
    PINNED_RELEASE.as_ref().map_err(|_| LaunchError::Artifacts)
}

pub(super) fn parse_cpu_prefill_tokens(value: Option<&std::ffi::OsStr>) -> Result<u8, LaunchError> {
    parse_enum(value, &[("8", 8), ("16", 16), ("32", 32), ("64", 64)])
}

pub(super) fn select_cpu_prefill_tokens(
    native: bool,
    requested: Option<u8>,
) -> Result<Option<u8>, LaunchError> {
    let Some(tokens) = requested else {
        return Ok(None);
    };
    if !native || !matches!(tokens, 8 | 16 | 32 | 64) {
        return Err(LaunchError::Arguments);
    }
    Ok(Some(tokens))
}

pub(super) fn parse_prefill_row_workers(
    value: Option<&std::ffi::OsStr>,
) -> Result<u8, LaunchError> {
    parse_enum(
        value,
        &[("1", 1), ("4", 4), ("8", 8), ("12", 12), ("16", 16)],
    )
}

pub(super) fn select_prefill_row_workers(
    native: bool,
    requested: Option<u8>,
    family: Option<bonsai_cpu_policy::hardware::SnapdragonFamily>,
    explicit_family: Option<CpuFamily>,
) -> Result<Option<u8>, LaunchError> {
    if !native {
        return if requested.is_some() {
            Err(LaunchError::Arguments)
        } else {
            Ok(None)
        };
    }
    if let Some(workers) = requested {
        if !matches!(workers, 1 | 4 | 8 | 12 | 16) {
            return Err(LaunchError::Arguments);
        }
        return Ok(Some(workers));
    }
    let effective_family = explicit_family.map(CpuFamily::into_snapdragon).or(family);
    let (_, prefill) =
        bonsai_cpu_policy::hardware::SnapdragonFamily::default_native_row_worker_budget(
            effective_family,
        );
    Ok(Some(prefill))
}

pub(super) fn verify(data: &Path, row_workers: u8) -> Result<(PathBuf, VerifiedFile), LaunchError> {
    if !matches!(row_workers, 1 | 4 | 8 | 12) {
        return Err(LaunchError::Artifacts);
    }
    let release = pinned_release()?;
    let file = release
        .files
        .iter()
        .find(|Object(file)| file.name == "prism-inference-server.exe")
        .ok_or(LaunchError::Artifacts)?;
    let executable = data
        .join("inference")
        .join(&release.commit)
        .join(&release.target)
        .join(&file.name);
    let guard = VerifiedFile::verify(&executable, file.bytes, &file.sha256)
        .map_err(|_| LaunchError::Artifacts)?;
    Ok((executable, guard))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_placement_selection_validates_native_and_non_native() -> Result<(), LaunchError> {
        assert_eq!(select_placement(true, None)?, Placement::Auto);
        for placement in [
            Placement::Auto,
            Placement::Cpu,
            Placement::Vulkan,
            Placement::Composite,
        ] {
            assert_eq!(select_placement(true, Some(placement))?, placement);
            assert_eq!(
                select_placement(false, Some(placement)),
                Err(LaunchError::Arguments)
            );
        }
        assert_eq!(select_placement(false, None)?, Placement::Auto);
        Ok(())
    }

    #[test]
    fn bundled_kv_cache_selection_defaults_to_rsq4_and_rejects_rsq4k3v() -> Result<(), LaunchError>
    {
        assert_eq!(select_kv_cache(true, None)?, Some(KvCache::Rsq4));
        assert_eq!(
            select_kv_cache(true, Some(KvCache::F16))?,
            Some(KvCache::F16)
        );
        assert_eq!(
            select_kv_cache(true, Some(KvCache::Rsq4))?,
            Some(KvCache::Rsq4)
        );
        assert_eq!(
            select_kv_cache(true, Some(KvCache::Rsq4Kvf))?,
            Some(KvCache::Rsq4Kvf)
        );
        assert_eq!(
            select_kv_cache(true, Some(KvCache::Rsq4K3v)),
            Err(LaunchError::NativeRuntime)
        );
        assert_eq!(
            KvCache::parse(Some(std::ffi::OsStr::new("rsq4k3v")))?,
            KvCache::Rsq4K3v
        );
        assert_eq!(select_kv_cache(false, None)?, None);
        for cache in [
            KvCache::F16,
            KvCache::Rsq4,
            KvCache::Rsq4Kvf,
            KvCache::Rsq4K3v,
        ] {
            assert_eq!(
                select_kv_cache(false, Some(cache)),
                Err(LaunchError::Arguments)
            );
        }
        Ok(())
    }

    #[test]
    fn compiled_server_release_is_pinned_and_rejects_path_or_identity_drift()
    -> Result<(), serde_json::Error> {
        assert!(release(LOCK).is_ok());
        let value: serde_json::Value = serde_json::from_str(LOCK)?;
        for (field, replacement) in [
            ("schemaVersion", serde_json::json!(2)),
            ("package", serde_json::json!("other")),
            ("target", serde_json::json!("../escape")),
            ("commit", serde_json::json!("../escape")),
            ("release", serde_json::json!("latest")),
            ("files", serde_json::json!([])),
        ] {
            let mut changed = value.clone();
            changed[field] = replacement;
            assert!(
                release(&serde_json::to_string(&changed)?).is_err(),
                "{field}"
            );
        }
        let mut changed = value.clone();
        changed["files"][0]["name"] = serde_json::json!("../prism-inference-server.exe");
        assert!(release(&serde_json::to_string(&changed)?).is_err());
        for (field, replacement) in [
            ("Repository", serde_json::json!("https://example.com/prism")),
            ("Commit", serde_json::json!("../escape")),
            ("Url", serde_json::json!("https://example.com/source.zip")),
            ("Bytes", serde_json::json!(0)),
            ("Bytes", serde_json::json!(268_435_457_u64)),
            ("Sha256", serde_json::json!("not-a-hash")),
        ] {
            let mut changed = value.clone();
            changed["Source"][field] = replacement;
            assert!(
                release(&serde_json::to_string(&changed)?).is_err(),
                "Source.{field}"
            );
        }
        let mut changed = value.clone();
        changed["Source"] = serde_json::json!([]);
        assert!(release(&serde_json::to_string(&changed)?).is_err());
        let mut changed = value.clone();
        changed["Source"]["unexpected"] = serde_json::json!(true);
        assert!(release(&serde_json::to_string(&changed)?).is_err());
        let mut changed = value;
        changed["asset"]["url"] = serde_json::json!("https://example.com/runtime.zip");
        assert!(release(&serde_json::to_string(&changed)?).is_err());
        Ok(())
    }

    #[test]
    fn verify_rejects_invalid_row_workers_before_artifact_lookup() {
        let missing_dir = std::env::temp_dir().join("Bonsai-inference-verify-nonexistent");
        for workers in [0, 2, 3, 5, 16, 255] {
            assert_eq!(
                verify(&missing_dir, workers).err(),
                Some(LaunchError::Artifacts)
            );
        }
        for workers in [1, 4, 8, 12] {
            assert_eq!(
                verify(&missing_dir, workers).err(),
                Some(LaunchError::Artifacts)
            );
        }
    }

    #[test]
    fn compiled_mmproj_digest_matches_policy_locks() -> Result<(), String> {
        let data = std::env::temp_dir();
        let mmproj = bonsai_cpu_policy::swift_mmproj_artifact(&data)?;
        assert_eq!(MMPROJ_SHA256, mmproj.sha256);
        Ok(())
    }

    #[test]
    fn session_enums_validate_exact_tokens_and_reject_aliases() -> Result<(), LaunchError> {
        use std::ffi::OsStr;

        assert_eq!(
            PromptReuseMode::parse(Some(OsStr::new("disabled")))?,
            PromptReuseMode::Disabled
        );
        assert!(!PromptReuseMode::Disabled.enabled());
        assert_eq!(
            PromptReuseMode::parse(Some(OsStr::new("latest")))?,
            PromptReuseMode::Latest
        );
        assert!(PromptReuseMode::Latest.enabled());
        assert_eq!(
            PromptReuseMode::parse(Some(OsStr::new("prompt-end-and-latest")))?,
            PromptReuseMode::PromptEndAndLatest
        );
        assert!(PromptReuseMode::PromptEndAndLatest.enabled());
        assert!(PromptReuseMode::parse(Some(OsStr::new("prompt_end"))).is_err());

        assert_eq!(
            CpuPrefillGdn::parse(Some(OsStr::new("sequential")))?,
            CpuPrefillGdn::Sequential
        );
        assert!(!CpuPrefillGdn::Sequential.is_wy_f32());
        assert_eq!(
            CpuPrefillGdn::parse(Some(OsStr::new("wy-f32")))?,
            CpuPrefillGdn::WyF32
        );
        assert!(CpuPrefillGdn::WyF32.is_wy_f32());
        assert!(CpuPrefillGdn::parse(Some(OsStr::new("wy"))).is_err());

        assert_eq!(
            SpeculationType::parse(Some(OsStr::new("none")))?,
            SpeculationType::None
        );
        assert_eq!(
            SpeculationType::parse(Some(OsStr::new("suffix")))?,
            SpeculationType::Suffix
        );
        assert!(SpeculationType::parse(Some(OsStr::new("ngram"))).is_err());

        let x1 = CpuFamily::parse(Some(OsStr::new("x1")))?;
        let x2 = CpuFamily::parse(Some(OsStr::new("x2")))?;
        assert_eq!(x1.as_str(), "x1");
        assert_eq!(x2.as_str(), "x2");
        assert_eq!(x1.default_row_workers(), 8);
        assert_eq!(x2.default_row_workers(), 8);
        assert!(CpuFamily::parse(Some(OsStr::new("x3"))).is_err());

        let vulkan = Placement::parse(Some(OsStr::new("vulkan")))?;
        let composite = Placement::parse(Some(OsStr::new("composite")))?;
        assert_eq!(vulkan, Placement::Vulkan);
        assert_eq!(composite, Placement::Composite);
        assert_eq!(Placement::Auto.as_str(), "auto");
        assert_eq!(Placement::Cpu.as_str(), "cpu");
        assert_eq!(vulkan.as_str(), "vulkan");
        assert_eq!(composite.as_str(), "composite");
        assert!(Placement::parse(Some(OsStr::new("npu"))).is_err());
        Ok(())
    }

    #[test]
    fn cpu_prefill_tokens_and_prefill_row_workers_validate_bundled_budgets()
    -> Result<(), LaunchError> {
        use bonsai_cpu_policy::hardware::SnapdragonFamily;
        use std::ffi::OsStr;

        for (raw, expected) in [("8", 8_u8), ("16", 16), ("32", 32), ("64", 64)] {
            assert_eq!(
                parse_cpu_prefill_tokens(Some(OsStr::new(raw))),
                Ok(expected)
            );
            assert_eq!(
                select_cpu_prefill_tokens(true, Some(expected)),
                Ok(Some(expected))
            );
            assert_eq!(
                select_cpu_prefill_tokens(false, Some(expected)),
                Err(LaunchError::Arguments)
            );
        }
        assert_eq!(select_cpu_prefill_tokens(true, None), Ok(None));
        assert_eq!(select_cpu_prefill_tokens(false, None), Ok(None));
        assert_eq!(
            select_cpu_prefill_tokens(true, Some(12)),
            Err(LaunchError::Arguments)
        );
        assert!(parse_cpu_prefill_tokens(Some(OsStr::new("128"))).is_err());

        for (raw, expected) in [("1", 1_u8), ("4", 4), ("8", 8), ("12", 12), ("16", 16)] {
            assert_eq!(
                parse_prefill_row_workers(Some(OsStr::new(raw))),
                Ok(expected)
            );
            assert_eq!(
                select_prefill_row_workers(true, Some(expected), None, None),
                Ok(Some(expected))
            );
            assert_eq!(
                select_prefill_row_workers(false, Some(expected), None, None),
                Err(LaunchError::Arguments)
            );
        }
        assert_eq!(
            select_prefill_row_workers(true, None, Some(SnapdragonFamily::XElite), None),
            Ok(Some(12))
        );
        assert_eq!(
            select_prefill_row_workers(true, None, Some(SnapdragonFamily::X2EliteExtreme), None),
            Ok(Some(16))
        );
        assert_eq!(
            select_prefill_row_workers(
                true,
                None,
                Some(SnapdragonFamily::XElite),
                Some(CpuFamily::X2),
            ),
            Ok(Some(16))
        );
        assert_eq!(
            select_prefill_row_workers(true, None, None, None),
            Ok(Some(4))
        );
        assert_eq!(
            select_prefill_row_workers(false, None, Some(SnapdragonFamily::XElite), None),
            Ok(None)
        );
        assert_eq!(
            select_prefill_row_workers(true, Some(2), None, None),
            Err(LaunchError::Arguments)
        );
        Ok(())
    }
}
