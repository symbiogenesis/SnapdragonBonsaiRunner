//! Compiled release identity for the independently built inference server.
#![expect(
    clippy::redundant_pub_crate,
    reason = "Explicit parent scope keeps these implementation helpers private."
)]

use std::path::{Path, PathBuf};

use bonsai_artifacts::VerifiedFile;
use serde::Deserialize;

use crate::{Object, launcher::LaunchError};

const LOCK: &str = include_str!("../../../config/prism-inference.lock.json");
pub(super) const MODEL_SHA256: &str =
    "5912bb739217cf25b4283a098c7b893d9baff36bdf2e72d3dbf4cb999d5633d5";
pub(super) const REPACKED_PTQ1_SHA256: &str =
    "e658f0f0736a03a4549535a439f9b07254ebf8ab88ae27cbe6e52d098ca5d1f5";
pub(super) const ADAPTER_SHA256: &str =
    "f1669534803d340a496015f5c45125f3437b4d13ec764f40e34488ce83967f42";
pub(super) const ADAPTER_PQ2_EXECUTION_SHA256: &str =
    "73d7c72c706b7e310bafb314f70564fd1d006d56ff742bff7372f9f47223b9e6";
pub(super) const ADAPTER_REPACKED_PTQ1_EXECUTION_SHA256: &str =
    "f7f50497a217f23cbeafc49781fedce3a8437aa629ad98740aaf53b892b4b81a";
pub(super) const TEMPLATE_SHA256: &str =
    "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041";
pub(super) const MMPROJ_SHA256: &str =
    "6807ede61d570bb86ba34b756a0fa109edc33668604de867c6ea6d8f1d631903";

pub(super) const fn adapter_execution_sha256(repacked: bool) -> &'static str {
    if repacked {
        ADAPTER_REPACKED_PTQ1_EXECUTION_SHA256
    } else {
        ADAPTER_PQ2_EXECUTION_SHA256
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum KvCache {
    F16,
    Rsq4,
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
        match value.and_then(std::ffi::OsStr::to_str) {
            Some("disabled") => Ok(Self::Disabled),
            Some("latest") => Ok(Self::Latest),
            Some("prompt-end-and-latest") => Ok(Self::PromptEndAndLatest),
            _ => Err(LaunchError::Arguments),
        }
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
        match value.and_then(std::ffi::OsStr::to_str) {
            Some("sequential") => Ok(Self::Sequential),
            Some("wy-f32") => Ok(Self::WyF32),
            _ => Err(LaunchError::Arguments),
        }
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
        match value.and_then(std::ffi::OsStr::to_str) {
            Some("none") => Ok(Self::None),
            Some("suffix") => Ok(Self::Suffix),
            _ => Err(LaunchError::Arguments),
        }
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
        match value.and_then(std::ffi::OsStr::to_str) {
            Some("x1") => Ok(Self::X1),
            Some("x2") => Ok(Self::X2),
            _ => Err(LaunchError::Arguments),
        }
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
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum Placement {
    #[default]
    Auto,
    Cpu,
}

impl Placement {
    pub(super) fn parse(value: Option<&std::ffi::OsStr>) -> Result<Self, LaunchError> {
        match value.and_then(std::ffi::OsStr::to_str) {
            Some("auto") => Ok(Self::Auto),
            Some("cpu") => Ok(Self::Cpu),
            _ => Err(LaunchError::Arguments),
        }
    }
}

// CPU placement must skip accelerator initialization and report both its
// requested placement and effective CPU backend in a genuine reviewed release.
// Source omission alone does not justify a smaller fixed memory allowance.
const CPU_PLACEMENT_RELEASE: Option<(&str, &str)> =
    Some(("0.1.32", "23e24e444a821d4426f65d56c6cf874cf568b0d9"));

fn select_placement(release: &ReleasePin, requested: Placement) -> Result<Placement, LaunchError> {
    if requested == Placement::Auto || reviewed_release_matches(release, CPU_PLACEMENT_RELEASE) {
        Ok(requested)
    } else {
        Err(LaunchError::NativeRuntime)
    }
}

pub(super) fn pinned_placement(
    native: bool,
    requested: Option<Placement>,
) -> Result<Placement, LaunchError> {
    if !native {
        return requested.map_or(Ok(Placement::Auto), |_| Err(LaunchError::Arguments));
    }
    select_placement(&release(LOCK)?, requested.unwrap_or_default())
}

impl KvCache {
    pub(super) fn parse(value: Option<&std::ffi::OsStr>) -> Result<Self, LaunchError> {
        match value.and_then(std::ffi::OsStr::to_str) {
            Some("f16") => Ok(Self::F16),
            Some("rsq4") => Ok(Self::Rsq4),
            Some("rsq4k3v") => Ok(Self::Rsq4K3v),
            _ => Err(LaunchError::Arguments),
        }
    }

    pub(super) const fn name(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::Rsq4 => "rsq4",
            Self::Rsq4K3v => "rsq4k3v",
        }
    }
}

fn reviewed_release_matches(release: &ReleasePin, identity: Option<(&str, &str)>) -> bool {
    identity.is_some_and(|(version, commit)| release.version == version && release.commit == commit)
}

fn turboquant_supported(release: &ReleasePin, memory_release: Option<(&str, &str)>) -> bool {
    // M1/M2 review includes the selected K4/V4 cache and its source-bound
    // readiness contract; delivery must retain the ordinary cache default.
    reviewed_release_matches(release, memory_release)
}

fn select_kv_cache(
    release: &ReleasePin,
    requested: Option<KvCache>,
) -> Result<Option<KvCache>, LaunchError> {
    if requested == Some(KvCache::Rsq4K3v) {
        if k4v3_context_supported(release) {
            Ok(requested)
        } else {
            Err(LaunchError::NativeRuntime)
        }
    } else if turboquant_supported(release, MEMORY_CONTEXT_RELEASE) {
        Ok(Some(requested.unwrap_or(KvCache::Rsq4)))
    } else if requested == Some(KvCache::Rsq4) {
        Err(LaunchError::NativeRuntime)
    } else {
        // Historical servers do not accept --kv-cache; their reviewed profile
        // already selects FP16. This is release selection, never retry fallback.
        Ok(None)
    }
}

pub(super) fn pinned_kv_cache(
    native: bool,
    requested: Option<KvCache>,
) -> Result<Option<KvCache>, LaunchError> {
    if native {
        select_kv_cache(&release(LOCK)?, requested)
    } else if requested.is_some() {
        Err(LaunchError::Arguments)
    } else {
        Ok(None)
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

fn row_workers_supported(
    release: &ReleasePin,
    workers: u8,
    memory_release: Option<(&str, &str)>,
) -> Result<(), LaunchError> {
    if workers == 1
        || (matches!(workers, 4 | 8 | 12) && reviewed_release_matches(release, memory_release))
    {
        Ok(())
    } else {
        Err(LaunchError::Artifacts)
    }
}

// Enable only after a reviewed source-bound release implements the paired
// context/memory flags AND the selected-cache/sequential-scratch accounting
// contract in bonsai_cpu_policy::context. A protocol-only source candidate must
// not activate smaller process allowances. A version number alone attests neither.
// This combined review also retains four row workers, K4/V4 and latest reuse;
// a later repin cannot silently lose those existing serving capabilities.
const MEMORY_CONTEXT_RELEASE: Option<(&str, &str)> =
    Some(("0.1.32", "23e24e444a821d4426f65d56c6cf874cf568b0d9"));

// K4/V3 changes precision and needs its own reviewed capacity/quality release.
// Both gates must identify the same genuine released source before this profile
// can use its smaller admission charge. Existing K4/V4 releases stay unchanged.
const K4V3_CONTEXT_RELEASE: Option<(&str, &str)> = None;

// Full-model K4/V4 context needs the separately reviewed larger live-state and
// checkpoint guards, not just the existing paired memory protocol. This exact
// source supplies the geometry-derived Swift allowance; generic graphs retain
// 4 GiB. Precision and process charges remain unchanged; this gate cannot
// authorize an unrelated memory release.
const K4V4_FULL_CONTEXT_RELEASE: Option<(&str, &str)> =
    Some(("0.1.32", "23e24e444a821d4426f65d56c6cf874cf568b0d9"));

fn full_context_release_matches(release: &ReleasePin, reviewed: Option<(&str, &str)>) -> bool {
    reviewed.is_some()
        && MEMORY_CONTEXT_RELEASE == reviewed
        && reviewed_release_matches(release, reviewed)
}

pub(super) fn full_context_supported() -> Result<bool, LaunchError> {
    let pin = release(LOCK)?;
    Ok(full_context_release_matches(
        &pin,
        K4V4_FULL_CONTEXT_RELEASE,
    ))
}

fn k4v3_context_supported(release: &ReleasePin) -> bool {
    K4V3_CONTEXT_RELEASE.is_some_and(|identity| {
        MEMORY_CONTEXT_RELEASE == Some(identity)
            && release.version == identity.0
            && release.commit == identity.1
    })
}

pub(super) fn memory_context_supported() -> Result<bool, LaunchError> {
    let pin = release(LOCK)?;
    Ok(reviewed_release_matches(&pin, MEMORY_CONTEXT_RELEASE))
}

// In-place cancellation recovery requires a reviewed release whose worker loop
// observes abandoned generation sockets, settles active computation within a
// bounded timeout, and resets live model state on authenticated `POST /reset`
// without expecting a `Content-Type` header.
const RESET_ON_CANCEL_RELEASE: Option<(&str, &str)> =
    Some(("0.1.32", "23e24e444a821d4426f65d56c6cf874cf568b0d9"));

fn reset_on_cancel_release_matches(release: &ReleasePin) -> bool {
    reviewed_release_matches(release, RESET_ON_CANCEL_RELEASE)
}

pub(super) fn reset_on_cancel_supported() -> Result<bool, LaunchError> {
    let pin = release(LOCK)?;
    Ok(reset_on_cancel_release_matches(&pin))
}

// Offline anonymous broker-pipe transport requires a separately reviewed native
// release implementing `--broker-pipe`, `AppContainer` verification, `SB2PIPE1`
// / `SB2OKAY1` handshake, `KIND_SERVER_READY`, `KIND_RESET`, and `KIND_VERIFIED_SECTION`.
const BROKER_PIPE_RELEASE: Option<(&str, &str)> =
    Some(("0.1.32", "23e24e444a821d4426f65d56c6cf874cf568b0d9"));

fn broker_pipe_release_matches(release: &ReleasePin) -> bool {
    reviewed_release_matches(release, BROKER_PIPE_RELEASE)
}

pub(super) fn broker_pipe_supported() -> Result<bool, LaunchError> {
    let pin = release(LOCK)?;
    Ok(broker_pipe_release_matches(&pin))
}

// Calibrated CoW admission, sparse branch prompt reuse, chunkwise GDN prefill,
// and suffix CPU speculation require the reviewed post-0.1.31 Prism release.
const CALIBRATED_COW_ADMISSION_RELEASE: Option<(&str, &str)> =
    Some(("0.1.32", "23e24e444a821d4426f65d56c6cf874cf568b0d9"));
const BRANCH_PROMPT_REUSE_RELEASE: Option<(&str, &str)> =
    Some(("0.1.32", "23e24e444a821d4426f65d56c6cf874cf568b0d9"));
const CPU_PREFILL_GDN_RELEASE: Option<(&str, &str)> =
    Some(("0.1.32", "23e24e444a821d4426f65d56c6cf874cf568b0d9"));
const CPU_SPECULATION_RELEASE: Option<(&str, &str)> =
    Some(("0.1.32", "23e24e444a821d4426f65d56c6cf874cf568b0d9"));

pub(super) fn calibrated_cow_admission_supported() -> Result<bool, LaunchError> {
    let pin = release(LOCK)?;
    Ok(reviewed_release_matches(
        &pin,
        CALIBRATED_COW_ADMISSION_RELEASE,
    ))
}

pub(super) fn branch_prompt_reuse_supported() -> Result<bool, LaunchError> {
    let pin = release(LOCK)?;
    Ok(reviewed_release_matches(&pin, BRANCH_PROMPT_REUSE_RELEASE))
}

pub(super) fn cpu_prefill_gdn_supported() -> Result<bool, LaunchError> {
    let pin = release(LOCK)?;
    Ok(reviewed_release_matches(&pin, CPU_PREFILL_GDN_RELEASE))
}

pub(super) fn cpu_speculation_supported() -> Result<bool, LaunchError> {
    let pin = release(LOCK)?;
    Ok(reviewed_release_matches(&pin, CPU_SPECULATION_RELEASE))
}

// Multimodal image/mmproj support requires a separately reviewed native release
// that implements vision projector loading, image embedding injection and the
// matching readiness identity fields.
const VISION_RELEASE: Option<(&str, &str)> =
    Some(("0.1.32", "23e24e444a821d4426f65d56c6cf874cf568b0d9"));
// Multi-rank DaisyRose LoRA support stays gated off for the frozen v0.1.32 pin
// until a reviewed Prism release publishes its release tuple here.
const MULTI_RANK_LORA_RELEASE: Option<(&str, &str)> = None;

// CPU tiled prefill requires a separately reviewed native release supporting
// `--cpu-prefill-tokens`, `--cpu-prefill-scratch-bytes`, and the matching
// readiness fields (`cpu_prefill_tokens`, `cpu_prefill_scratch_bytes`, `:cpu-tiled-ffn-v1:{n}`).
const CPU_PREFILL_RELEASE: Option<(&str, &str)> =
    Some(("0.1.32", "23e24e444a821d4426f65d56c6cf874cf568b0d9"));

pub(super) fn parse_cpu_prefill_tokens(value: Option<&std::ffi::OsStr>) -> Result<u8, LaunchError> {
    match value.and_then(std::ffi::OsStr::to_str) {
        Some("8") => Ok(8),
        Some("16") => Ok(16),
        Some("32") => Ok(32),
        _ => Err(LaunchError::Arguments),
    }
}

pub(super) fn pinned_cpu_prefill_tokens(
    native: bool,
    requested: Option<u8>,
) -> Result<Option<u8>, LaunchError> {
    let Some(tokens) = requested else {
        return Ok(None);
    };
    if !native || !matches!(tokens, 8 | 16 | 32) {
        return Err(LaunchError::Arguments);
    }
    let pin = release(LOCK)?;
    if reviewed_release_matches(&pin, MEMORY_CONTEXT_RELEASE)
        && reviewed_release_matches(&pin, CPU_PREFILL_RELEASE)
    {
        Ok(Some(tokens))
    } else {
        Err(LaunchError::NativeRuntime)
    }
}

fn vision_release_matches(release: &ReleasePin) -> bool {
    reviewed_release_matches(release, VISION_RELEASE)
}

pub(super) fn vision_supported() -> Result<bool, LaunchError> {
    let pin = release(LOCK)?;
    Ok(vision_release_matches(&pin))
}

fn multi_rank_lora_release_matches(release: &ReleasePin) -> bool {
    reviewed_release_matches(release, MULTI_RANK_LORA_RELEASE)
}

pub(super) fn multi_rank_lora_supported() -> Result<bool, LaunchError> {
    let pin = release(LOCK)?;
    Ok(multi_rank_lora_release_matches(&pin))
}

pub(super) fn pinned_version() -> Result<String, LaunchError> {
    release(LOCK).map(|release| release.version)
}

fn prompt_reuse_supported(release: &ReleasePin, latest: bool) -> Result<(), LaunchError> {
    // Capability is bound to reviewed source and immutable released bytes. A
    // later version must be reviewed explicitly, even if its CLI looks similar.
    if !latest || turboquant_supported(release, MEMORY_CONTEXT_RELEASE) {
        Ok(())
    } else {
        Err(LaunchError::Artifacts)
    }
}

pub(super) fn verify(
    data: &Path,
    row_workers: u8,
    latest_prompt_reuse: bool,
) -> Result<(PathBuf, VerifiedFile), LaunchError> {
    let release = release(LOCK)?;
    row_workers_supported(&release, row_workers, MEMORY_CONTEXT_RELEASE)?;
    prompt_reuse_supported(&release, latest_prompt_reuse)?;
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
    fn cpu_placement_waits_for_an_exact_reviewed_release() -> Result<(), LaunchError> {
        let mut pin = release(LOCK)?;
        assert_eq!(select_placement(&pin, Placement::Cpu)?, Placement::Cpu);
        for (version, commit) in [
            ("0.1.27", "23e24e444a821d4426f65d56c6cf874cf568b0d9"),
            ("0.1.28", "23e24e444a821d4426f65d56c6cf874cf568b0d9"),
            ("0.1.30", "23e24e444a821d4426f65d56c6cf874cf568b0d9"),
            ("0.1.31", "577966d98cf8b36021df52da194b1a5540429a6b"),
            ("0.1.33", "23e24e444a821d4426f65d56c6cf874cf568b0d9"),
            ("0.1.32", "0000000000000000000000000000000000000000"),
            ("0.1.32", "89df9aaff9e580d66679f2befe94d471bcf4fc7f"),
        ] {
            pin.version = version.into();
            pin.commit = commit.into();
            assert_eq!(select_placement(&pin, Placement::Auto)?, Placement::Auto);
            assert_eq!(
                select_placement(&pin, Placement::Cpu),
                Err(LaunchError::NativeRuntime)
            );
        }
        assert_eq!(pinned_placement(true, None)?, Placement::Auto);
        assert!(reset_on_cancel_supported()?);
        let mut pin = release(LOCK)?;
        assert!(reset_on_cancel_release_matches(&pin));
        for (version, commit) in [
            ("0.1.27", "23e24e444a821d4426f65d56c6cf874cf568b0d9"),
            ("0.1.28", "23e24e444a821d4426f65d56c6cf874cf568b0d9"),
            ("0.1.30", "23e24e444a821d4426f65d56c6cf874cf568b0d9"),
            ("0.1.31", "577966d98cf8b36021df52da194b1a5540429a6b"),
            ("0.1.32", "89df9aaff9e580d66679f2befe94d471bcf4fc7f"),
            ("0.1.32", "0000000000000000000000000000000000000000"),
        ] {
            pin.version = version.into();
            pin.commit = commit.into();
            assert!(!reset_on_cancel_release_matches(&pin));
        }
        assert_eq!(pinned_placement(false, None)?, Placement::Auto);
        for placement in [Placement::Auto, Placement::Cpu] {
            assert_eq!(
                pinned_placement(false, Some(placement)),
                Err(LaunchError::Arguments)
            );
        }
        Ok(())
    }

    #[test]
    fn kv_cache_default_and_recovery_require_the_exact_reviewed_source() -> Result<(), LaunchError>
    {
        let mut pin = release(LOCK)?;
        assert_eq!(
            MEMORY_CONTEXT_RELEASE,
            Some((pin.version.as_str(), pin.commit.as_str()))
        );
        assert!(memory_context_supported()?);
        assert!(row_workers_supported(&pin, 4, MEMORY_CONTEXT_RELEASE).is_ok());
        assert_eq!(select_kv_cache(&pin, None)?, Some(KvCache::Rsq4));
        assert_eq!(
            select_kv_cache(&pin, Some(KvCache::F16))?,
            Some(KvCache::F16)
        );
        assert_eq!(
            select_kv_cache(&pin, Some(KvCache::Rsq4))?,
            Some(KvCache::Rsq4)
        );
        for (version, commit) in [
            ("0.1.27", "23e24e444a821d4426f65d56c6cf874cf568b0d9"),
            ("0.1.28", "23e24e444a821d4426f65d56c6cf874cf568b0d9"),
            ("0.1.30", "23e24e444a821d4426f65d56c6cf874cf568b0d9"),
            ("0.1.31", "577966d98cf8b36021df52da194b1a5540429a6b"),
            ("0.1.33", "23e24e444a821d4426f65d56c6cf874cf568b0d9"),
            ("0.1.32", "89df9aaff9e580d66679f2befe94d471bcf4fc7f"),
            ("0.1.32", "0000000000000000000000000000000000000000"),
        ] {
            pin.version = version.into();
            pin.commit = commit.into();
            assert_eq!(select_kv_cache(&pin, None)?, None);
            assert_eq!(select_kv_cache(&pin, Some(KvCache::F16))?, None);
            assert_eq!(
                select_kv_cache(&pin, Some(KvCache::Rsq4)),
                Err(LaunchError::NativeRuntime)
            );
        }
        Ok(())
    }

    #[test]
    fn full_k4v4_context_requires_its_own_aligned_source_release() -> Result<(), LaunchError> {
        let mut pin = release(LOCK)?;
        assert!(full_context_supported()?);
        assert_eq!(K4V4_FULL_CONTEXT_RELEASE, MEMORY_CONTEXT_RELEASE);
        assert!(!full_context_release_matches(&pin, None));
        assert!(full_context_release_matches(&pin, MEMORY_CONTEXT_RELEASE));
        for version in ["0.1.27", "0.1.28", "0.1.30", "0.1.31", "0.1.33"] {
            pin.version = version.into();
            assert!(!full_context_release_matches(&pin, MEMORY_CONTEXT_RELEASE));
        }
        pin.version = "0.1.32".into();
        pin.commit = "0".repeat(40);
        assert!(!full_context_release_matches(&pin, MEMORY_CONTEXT_RELEASE));
        let other_source = (pin.version.as_str(), pin.commit.as_str());
        assert!(!full_context_release_matches(&pin, Some(other_source)));
        pin.version = "0.1.26".into();
        pin.commit = "ac941f2668ae93dba9824002b4597081457aa09f".into();
        assert!(!full_context_release_matches(&pin, MEMORY_CONTEXT_RELEASE));
        Ok(())
    }

    #[test]
    fn k4v3_capacity_selection_waits_for_a_separately_reviewed_memory_release()
    -> Result<(), LaunchError> {
        let mut pin = release(LOCK)?;
        assert!(!k4v3_context_supported(&pin));
        assert_eq!(
            select_kv_cache(&pin, Some(KvCache::Rsq4K3v)),
            Err(LaunchError::NativeRuntime)
        );
        assert_eq!(
            KvCache::parse(Some(std::ffi::OsStr::new("rsq4k3v")))?,
            KvCache::Rsq4K3v
        );
        for (version, commit) in [
            ("0.1.27", "23e24e444a821d4426f65d56c6cf874cf568b0d9"),
            ("0.1.28", "23e24e444a821d4426f65d56c6cf874cf568b0d9"),
            ("0.1.30", "23e24e444a821d4426f65d56c6cf874cf568b0d9"),
            ("0.1.31", "577966d98cf8b36021df52da194b1a5540429a6b"),
            ("0.1.33", "23e24e444a821d4426f65d56c6cf874cf568b0d9"),
            ("0.1.32", "0000000000000000000000000000000000000000"),
        ] {
            pin.version = version.into();
            pin.commit = commit.into();
            assert!(!k4v3_context_supported(&pin));
            assert_eq!(
                select_kv_cache(&pin, Some(KvCache::Rsq4K3v)),
                Err(LaunchError::NativeRuntime)
            );
        }
        assert_eq!(
            pinned_kv_cache(false, Some(KvCache::Rsq4K3v)),
            Err(LaunchError::Arguments)
        );
        Ok(())
    }

    #[test]
    fn latest_prompt_reuse_requires_exact_reviewed_release_source() -> Result<(), LaunchError> {
        let mut pin = release(LOCK)?;
        assert!(prompt_reuse_supported(&pin, true).is_ok());
        for version in [
            "0.1.25",
            "0.1.27",
            "0.1.28",
            "0.1.29",
            "0.1.30",
            "0.1.31",
            "0.1.33",
            "0.1.32-dev",
            "v0.1.32",
            "0.1.32.0",
            "0.1.032",
            "",
            "latest",
        ] {
            pin.version = version.into();
            assert_eq!(
                prompt_reuse_supported(&pin, true),
                Err(LaunchError::Artifacts)
            );
            assert!(prompt_reuse_supported(&pin, false).is_ok());
        }
        pin = release(LOCK)?;
        pin.commit = "0".repeat(40);
        assert_eq!(
            prompt_reuse_supported(&pin, true),
            Err(LaunchError::Artifacts)
        );
        assert!(prompt_reuse_supported(&pin, false).is_ok());
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
    fn four_row_workers_require_the_reviewed_parallel_release() -> Result<(), LaunchError> {
        let mut pin = release(LOCK)?;
        for workers in [1, 4, 8, 12] {
            assert!(row_workers_supported(&pin, workers, MEMORY_CONTEXT_RELEASE).is_ok());
        }
        for workers in [0, 2, 3, 5, 16, 255] {
            assert_eq!(
                row_workers_supported(&pin, workers, MEMORY_CONTEXT_RELEASE),
                Err(LaunchError::Artifacts)
            );
        }
        for version in [
            "0.1.8",
            "0.1.9",
            "0.1.25",
            "0.1.27",
            "0.1.28",
            "0.1.29",
            "0.1.30",
            "0.1.31",
            "0.1.33",
            "0.1.32-dev",
            "v0.1.32",
            "0.1.32.0",
            "0.1.032",
            "",
            "latest",
        ] {
            pin.version = version.into();
            assert!(row_workers_supported(&pin, 1, MEMORY_CONTEXT_RELEASE).is_ok());
            for workers in [4, 8, 12] {
                assert_eq!(
                    row_workers_supported(&pin, workers, MEMORY_CONTEXT_RELEASE),
                    Err(LaunchError::Artifacts)
                );
            }
        }
        pin = release(LOCK)?;
        pin.commit = "0".repeat(40);
        assert!(row_workers_supported(&pin, 1, MEMORY_CONTEXT_RELEASE).is_ok());
        for workers in [4, 8, 12] {
            assert_eq!(
                row_workers_supported(&pin, workers, MEMORY_CONTEXT_RELEASE),
                Err(LaunchError::Artifacts)
            );
        }
        Ok(())
    }

    #[test]
    fn reviewed_memory_release_keeps_parallel_quantized_and_reuse_prerequisites()
    -> Result<(), LaunchError> {
        let mut pin = release(LOCK)?;
        let reviewed = Some(("0.1.26", "1111111111111111111111111111111111111111"));
        pin.version = "0.1.26".into();
        pin.commit = "1111111111111111111111111111111111111111".into();
        assert!(reviewed_release_matches(&pin, reviewed));
        assert!(turboquant_supported(&pin, reviewed));
        assert!(row_workers_supported(&pin, 4, reviewed).is_ok());
        for workers in [0, 2, 3, 5, 16, 255] {
            assert_eq!(
                row_workers_supported(&pin, workers, reviewed),
                Err(LaunchError::Artifacts)
            );
        }
        for identity in [
            None,
            Some(("0.1.27", "1111111111111111111111111111111111111111")),
            Some(("0.1.26", "0000000000000000000000000000000000000000")),
        ] {
            assert!(!reviewed_release_matches(&pin, identity));
            assert!(!turboquant_supported(&pin, identity));
            assert_eq!(
                row_workers_supported(&pin, 4, identity),
                Err(LaunchError::Artifacts)
            );
            assert!(row_workers_supported(&pin, 1, identity).is_ok());
        }
        // Synthetic source review does not open the production gate or change
        // the published pin. Latest reuse shares the TurboQuant source review.
        assert!(!turboquant_supported(&pin, MEMORY_CONTEXT_RELEASE));
        assert_eq!(
            select_kv_cache(&pin, Some(KvCache::Rsq4)),
            Err(LaunchError::NativeRuntime)
        );
        assert_eq!(
            prompt_reuse_supported(&pin, true),
            Err(LaunchError::Artifacts)
        );
        assert!(prompt_reuse_supported(&pin, false).is_ok());
        assert!(vision_supported()?);
        assert!(!vision_release_matches(&pin));
        assert!(!multi_rank_lora_supported()?);
        assert!(!multi_rank_lora_release_matches(&pin));
        Ok(())
    }

    #[test]
    fn compiled_adapter_and_mmproj_digests_match_policy_locks() -> Result<(), String> {
        let data = std::env::temp_dir();
        let adapter = bonsai_cpu_policy::abliterate_lora_artifact(&data)?;
        let mmproj = bonsai_cpu_policy::swift_mmproj_artifact(&data)?;
        assert_eq!(ADAPTER_SHA256, adapter.sha256);
        assert_eq!(MMPROJ_SHA256, mmproj.sha256);
        Ok(())
    }

    #[test]
    fn adapter_execution_digests_are_pinned_lowercase_hex_and_distinct() {
        let pq2 = adapter_execution_sha256(false);
        let ptq1 = adapter_execution_sha256(true);
        assert_eq!(pq2, ADAPTER_PQ2_EXECUTION_SHA256);
        assert_eq!(ptq1, ADAPTER_REPACKED_PTQ1_EXECUTION_SHA256);
        assert!(hex(pq2, 64));
        assert!(hex(ptq1, 64));
        assert_ne!(pq2, ptq1);
        for artifact in [
            MODEL_SHA256,
            REPACKED_PTQ1_SHA256,
            ADAPTER_SHA256,
            TEMPLATE_SHA256,
            MMPROJ_SHA256,
        ] {
            assert_ne!(pq2, artifact);
            assert_ne!(ptq1, artifact);
        }
    }

    #[test]
    fn broker_pipe_and_0_1_32_capabilities_require_reviewed_release() -> Result<(), LaunchError> {
        let mut pin = release(LOCK)?;
        assert_eq!(
            BROKER_PIPE_RELEASE,
            Some(("0.1.32", "23e24e444a821d4426f65d56c6cf874cf568b0d9"))
        );
        assert!(broker_pipe_supported()?);
        assert!(broker_pipe_release_matches(&pin));
        assert!(calibrated_cow_admission_supported()?);
        assert!(branch_prompt_reuse_supported()?);
        assert!(cpu_prefill_gdn_supported()?);
        assert!(cpu_speculation_supported()?);
        for (version, commit) in [
            ("0.1.31", "577966d98cf8b36021df52da194b1a5540429a6b"),
            ("0.1.32", "577966d98cf8b36021df52da194b1a5540429a6b"),
            ("0.1.32", "0000000000000000000000000000000000000000"),
            ("0.1.33", "23e24e444a821d4426f65d56c6cf874cf568b0d9"),
        ] {
            pin.version = version.into();
            pin.commit = commit.into();
            assert!(!broker_pipe_release_matches(&pin));
            assert!(!reviewed_release_matches(
                &pin,
                CALIBRATED_COW_ADMISSION_RELEASE
            ));
            assert!(!reviewed_release_matches(&pin, BRANCH_PROMPT_REUSE_RELEASE));
            assert!(!reviewed_release_matches(&pin, CPU_PREFILL_GDN_RELEASE));
            assert!(!reviewed_release_matches(&pin, CPU_SPECULATION_RELEASE));
        }
        Ok(())
    }

    #[test]
    fn parsed_0_1_32_session_enums_validate_exact_tokens_and_reject_aliases()
    -> Result<(), LaunchError> {
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
        Ok(())
    }
}
