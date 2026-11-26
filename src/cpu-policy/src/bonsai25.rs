//! Pinned artifact policy and lexical path resolution for opt-in Ternary Bonsai 2.5 27B `GGUF` models.

use std::fmt;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::{hex, require, unexpanded_variable};

/// Embedded compiled lock document for `prism-ml/Ternary-Bonsai-2.5-27B-GGUF`.
pub const BONSAI_2_5_LOCK: &str = include_str!("../../../config/bonsai-2-5-27b.lock.json");

/// Opt-in Ternary Bonsai 2.5 27B quantization variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Bonsai25Variant {
    /// `PQ2_0` ternary quantization (`7_253_354_848` bytes).
    #[serde(rename = "pq2")]
    Pq2,
    /// `PTQ1_0` ternary quantization (`5_993_834_848` bytes).
    #[serde(rename = "ptq1")]
    Ptq1,
}

impl Bonsai25Variant {
    /// CLI and lock key spelling (`"pq2"`, `"ptq1"`).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Pq2 => "pq2",
            Self::Ptq1 => "ptq1",
        }
    }

    /// Prism `weight_representation` identifier returned by readiness responses.
    #[must_use]
    pub const fn representation_name(self) -> &'static str {
        match self {
            Self::Pq2 => "Bonsai-2.5-PQ2_0",
            Self::Ptq1 => "Bonsai-2.5-PTQ1_0",
        }
    }

    /// Readiness `id` emitted by `prism-inference-server` for this variant.
    #[must_use]
    pub const fn model_id(self) -> &'static str {
        match self {
            Self::Pq2 => "Ternary-Bonsai-2.5-27B-PQ2_0",
            Self::Ptq1 => "Ternary-Bonsai-2.5-27B-PTQ1_0",
        }
    }

    /// Exact pinned `GGUF` byte length.
    #[must_use]
    pub const fn weight_bytes(self) -> u64 {
        match self {
            Self::Pq2 => 7_253_354_848,
            Self::Ptq1 => 5_993_834_848,
        }
    }

    /// Exact pinned lowercase SHA-256 hex digest.
    #[must_use]
    pub const fn sha256(self) -> &'static str {
        match self {
            Self::Pq2 => "103a11c94324029d8bae3f8152f30bba414730064cfb77438f8589db72e1aa85",
            Self::Ptq1 => "929560ec9d37cf3cadd4b00cc87b51c570082a628dd830b4692dc0b2ed18dbbe",
        }
    }

    /// Exact immutable Hugging Face artifact filename.
    #[must_use]
    pub const fn filename(self) -> &'static str {
        match self {
            Self::Pq2 => "Ternary-Bonsai-2.5-27B-PQ2_0.gguf",
            Self::Ptq1 => "Ternary-Bonsai-2.5-27B-PTQ1_0.gguf",
        }
    }

    /// Exact expected `GGUF` tensor count (`851`).
    #[must_use]
    pub const fn tensor_count(self) -> usize {
        let _ = self;
        851
    }

    /// Exact expected `general.file_type` declaration (`141` for `PQ2_0`, `143` for `PTQ1_0`).
    #[must_use]
    pub const fn file_type(self) -> u64 {
        match self {
            Self::Pq2 => 141,
            Self::Ptq1 => 143,
        }
    }
}

impl fmt::Display for Bonsai25Variant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for Bonsai25Variant {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "pq2" => Ok(Self::Pq2),
            "ptq1" => Ok(Self::Ptq1),
            _ => Err("Unsupported Ternary Bonsai 2.5 variant; expected `pq2` or `ptq1`.".into()),
        }
    }
}

/// Maximum installed/OS-usable physical memory (`18 GiB`) classified as
/// structurally super memory-constrained for `PQ2_0` on first load, catching
/// `16 GiB` devices without affecting `24`/`32`/`48 GiB` machines.
pub const BONSAI25_PTQ1_AUTO_TOTAL_PHYSICAL_MAX_BYTES: u64 = 18 * 1024 * 1024 * 1024;

/// Host physical RAM snapshot used to select a default Bonsai 2.5 variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bonsai25MemorySnapshot {
    /// OS-usable physical memory in bytes.
    pub total_physical_bytes: u64,
    /// Physical memory currently available to the system in bytes.
    pub available_physical_bytes: u64,
}

/// On-disk presence of pinned Bonsai 2.5 `GGUF` variants in the data folder.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Bonsai25CachedVariants {
    /// Whether `Ternary-Bonsai-2.5-27B-PQ2_0.gguf` (or an active partial) is present.
    pub pq2_present: bool,
    /// Whether `Ternary-Bonsai-2.5-27B-PTQ1_0.gguf` (or an active partial) is present.
    pub ptq1_present: bool,
}

/// Returns whether a host RAM snapshot is super memory-constrained for `PQ2_0`
/// under the selected native runtime profile and calibrated options.
///
/// Hosts with `<= 18 GiB` total usable RAM or insufficient currently available
/// RAM to admit `PQ2_0`'s minimum calibrated context (`MIN_CONTEXT_TOKENS`)
/// classify as super memory-constrained. Invalid option combinations never
/// classify as memory pressure.
#[must_use]
pub fn is_super_memory_constrained_for_pq2(
    memory: Bonsai25MemorySnapshot,
    profile: crate::context::SwiftContextProfile,
    mut pq2_options: crate::context::SwiftCalibratedOptions,
    allow_tile_64: bool,
) -> bool {
    pq2_options.weight_packing = crate::context::SwiftWeightPacking::Bonsai25Pq2;
    if crate::context::SwiftContextPlan::for_calibrated_execution_with_tile_64(
        u64::MAX / 2,
        profile,
        pq2_options,
        allow_tile_64,
    )
    .is_err()
    {
        return false;
    }
    if memory.total_physical_bytes <= BONSAI25_PTQ1_AUTO_TOTAL_PHYSICAL_MAX_BYTES {
        return true;
    }
    crate::context::SwiftContextPlan::for_calibrated_execution_with_tile_64(
        memory.available_physical_bytes,
        profile,
        pq2_options,
        allow_tile_64,
    )
    .is_err()
}

/// Selects the default Bonsai 2.5 variant when the user omitted `--bonsai-2-5`.
///
/// Preserves an existing single-variant local cache so live memory changes
/// never force a multi-gigabyte re-download, while first loads (or caches with
/// both variants present) automatically select `Ptq1` when `memory` shows a
/// super memory-constrained host and `Pq2` otherwise.
#[must_use]
pub fn select_default_bonsai25_variant(
    memory: Option<Bonsai25MemorySnapshot>,
    cached: Bonsai25CachedVariants,
    profile: crate::context::SwiftContextProfile,
    pq2_options: crate::context::SwiftCalibratedOptions,
    allow_tile_64: bool,
) -> Bonsai25Variant {
    match (cached.pq2_present, cached.ptq1_present) {
        (true, false) => Bonsai25Variant::Pq2,
        (false, true) => Bonsai25Variant::Ptq1,
        (false, false) | (true, true) => {
            if memory.is_some_and(|snapshot| {
                is_super_memory_constrained_for_pq2(snapshot, profile, pq2_options, allow_tile_64)
            }) {
                Bonsai25Variant::Ptq1
            } else {
                Bonsai25Variant::Pq2
            }
        }
    }
}

/// Verified compiled provenance and lexical path for a Ternary Bonsai 2.5 27B `GGUF` artifact.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Bonsai25ModelArtifact {
    /// Lexical target file path inside `data/models/<revision>/<filename>`.
    pub path: PathBuf,
    /// Selected Ternary Bonsai 2.5 quantization variant.
    pub variant: Bonsai25Variant,
    /// Pinned Hugging Face `owner/name` repository.
    pub repository: String,
    /// Pinned 40-hex Git revision hash.
    pub revision: String,
    /// Plain `GGUF` artifact filename.
    pub filename: String,
    /// Exact locked byte length.
    pub bytes: u64,
    /// Exact lowercase 64-hex SHA-256 digest.
    pub sha256: String,
    /// Exact pinned Hugging Face download URL.
    pub url: String,
    /// Short SPDX license identifier (`"Apache-2.0"`).
    pub license: String,
    /// Exact expected `GGUF` tensor count (`851`).
    pub tensor_count: usize,
    /// Exact expected `general.file_type` value.
    pub file_type: u64,
    /// Optional fixed chunk size (`>= 4 MiB`) for parallel mapped SHA-256 verification.
    pub chunk_bytes: Option<usize>,
    /// Optional ordered lowercase 64-hex SHA-256 digests covering `0..bytes`.
    pub chunk_sha256s: Option<Vec<String>>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Bonsai25VariantLock {
    filename: String,
    model_id: String,
    representation: String,
    bytes: u64,
    sha256: String,
    url: String,
    tensor_count: usize,
    file_type: u64,
    runtime_wired: bool,
    #[serde(default, rename = "chunkBytes")]
    chunk_bytes: Option<usize>,
    #[serde(default, rename = "chunkSha256s")]
    chunk_sha256s: Option<Vec<String>>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Bonsai25VariantsMap {
    pq2: Bonsai25VariantLock,
    ptq1: Bonsai25VariantLock,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Bonsai25MmprojLock {
    filename: String,
    bytes: u64,
    sha256: String,
    url: String,
    shared_with_swift_mmproj: bool,
    runtime_wired: bool,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Bonsai25MtpReservedLock {
    filename: String,
    bytes: u64,
    sha256: String,
    url: String,
    tensor_count: usize,
    nextn_predict_layers: u32,
    runtime_wired: bool,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Bonsai25MtpReservedMap {
    #[serde(rename = "pq2-mtp")]
    pq2_mtp: Bonsai25MtpReservedLock,
    #[serde(rename = "ptq1-mtp")]
    ptq1_mtp: Bonsai25MtpReservedLock,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Bonsai25SourceInfo {
    base_model: String,
    announcement_url: String,
    published_date: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Bonsai25Qualification {
    status: String,
    notes: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Bonsai25LockFile {
    schema_version: u32,
    decision: String,
    family: String,
    repository: String,
    revision: String,
    license: String,
    architecture: String,
    source: Bonsai25SourceInfo,
    variants: Bonsai25VariantsMap,
    mmproj: Bonsai25MmprojLock,
    mtp_reserved: Bonsai25MtpReservedMap,
    qualification: Bonsai25Qualification,
}

/// Resolves and validates the pinned Ternary Bonsai 2.5 27B model artifact inside `data`.
///
/// # Errors
/// Rejects relative, NUL-containing, or unexpanded data directories, or lockfile drift.
pub fn bonsai25_model_artifact(
    data: &Path,
    variant: Bonsai25Variant,
) -> Result<Bonsai25ModelArtifact, String> {
    let data_text = data
        .to_str()
        .ok_or("DataDirectory must be valid Unicode.")?;
    require(
        data.is_absolute() && !data_text.contains('\0') && !unexpanded_variable(data_text),
        "An expanded absolute data directory is required.",
    )?;
    bonsai25_model_artifact_from_lock(data, variant, BONSAI_2_5_LOCK)
}

fn validate_mtp_entry(
    entry: &Bonsai25MtpReservedLock,
    repository: &str,
    revision: &str,
) -> Result<(), String> {
    let expected_url = format!(
        "https://huggingface.co/{repository}/resolve/{revision}/{}",
        entry.filename
    );
    require(
        !entry.runtime_wired
            && !entry.filename.is_empty()
            && entry.bytes > 0
            && hex(&entry.sha256, 64)
            && entry.tensor_count == 866
            && entry.nextn_predict_layers == 1
            && entry.url == expected_url,
        "Unexpected Bonsai 2.5 reserved MTP entry.",
    )
}

fn bonsai25_model_artifact_from_lock(
    data: &Path,
    variant: Bonsai25Variant,
    lock_json: &str,
) -> Result<Bonsai25ModelArtifact, String> {
    let parsed: Bonsai25LockFile = serde_json::from_str(lock_json)
        .map_err(|_| "Invalid Ternary Bonsai 2.5 lock JSON.".to_owned())?;
    require(
        parsed.schema_version == 1,
        "Bonsai 2.5 lock schemaVersion must be 1.",
    )?;
    require(
        !parsed.decision.is_empty()
            && parsed.family == "ternary-bonsai-2.5-27b"
            && parsed.architecture == "qwen35"
            && parsed.license == "Apache-2.0"
            && parsed.source.base_model == "Qwen/Qwen3.8-27B"
            && !parsed.source.announcement_url.is_empty()
            && !parsed.source.published_date.is_empty()
            && !parsed.qualification.status.is_empty()
            && !parsed.qualification.notes.is_empty(),
        "Unexpected Bonsai 2.5 lock metadata.",
    )?;
    require(
        parsed.repository == "prism-ml/Ternary-Bonsai-2.5-27B-GGUF" && hex(&parsed.revision, 40),
        "Unexpected Bonsai 2.5 upstream repository or revision.",
    )?;
    let expected_mmproj_url = format!(
        "https://huggingface.co/{}/resolve/{}/{}",
        parsed.repository, parsed.revision, parsed.mmproj.filename
    );
    require(
        parsed.mmproj.runtime_wired
            && parsed.mmproj.shared_with_swift_mmproj
            && parsed.mmproj.filename == "Ternary-Bonsai-2.5-27B-mmproj-Q8_0.gguf"
            && parsed.mmproj.bytes == crate::MMPROJ_BYTES
            && parsed.mmproj.sha256 == crate::MMPROJ_SHA256
            && parsed.mmproj.url == expected_mmproj_url,
        "Unexpected Bonsai 2.5 mmproj lock record.",
    )?;
    validate_mtp_entry(
        &parsed.mtp_reserved.pq2_mtp,
        &parsed.repository,
        &parsed.revision,
    )?;
    validate_mtp_entry(
        &parsed.mtp_reserved.ptq1_mtp,
        &parsed.repository,
        &parsed.revision,
    )?;

    let entry = match variant {
        Bonsai25Variant::Pq2 => parsed.variants.pq2,
        Bonsai25Variant::Ptq1 => parsed.variants.ptq1,
    };
    let expected_url = format!(
        "https://huggingface.co/{}/resolve/{}/{}",
        parsed.repository, parsed.revision, entry.filename
    );
    require(
        entry.runtime_wired
            && entry.filename == variant.filename()
            && entry.model_id == variant.model_id()
            && entry.representation == variant.representation_name()
            && entry.bytes == variant.weight_bytes()
            && entry.sha256 == variant.sha256()
            && entry.url == expected_url
            && entry.tensor_count == variant.tensor_count()
            && entry.file_type == variant.file_type(),
        "Bonsai 2.5 variant lock entry disagrees with compiled identity.",
    )?;
    let (chunk_bytes, chunk_sha256s) =
        crate::validate_chunk_manifest(entry.bytes, entry.chunk_bytes, entry.chunk_sha256s)?
            .map_or((None, None), |(cb, cs)| (Some(cb), Some(cs)));

    Ok(Bonsai25ModelArtifact {
        path: data
            .join("models")
            .join(&parsed.revision)
            .join(&entry.filename),
        variant,
        repository: parsed.repository,
        revision: parsed.revision,
        filename: entry.filename,
        bytes: entry.bytes,
        sha256: entry.sha256,
        url: entry.url,
        license: parsed.license,
        tensor_count: entry.tensor_count,
        file_type: entry.file_type,
        chunk_bytes,
        chunk_sha256s,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bonsai25_lock_resolves_pq2_and_ptq1_and_rejects_drifted_inputs() -> Result<(), String> {
        let data = Path::new(if cfg!(windows) {
            r"C:\BonsaiData"
        } else {
            "/var/lib/bonsai"
        });
        for variant in [Bonsai25Variant::Pq2, Bonsai25Variant::Ptq1] {
            let artifact = bonsai25_model_artifact(data, variant)?;
            assert_eq!(artifact.bytes, variant.weight_bytes());
            assert_eq!(artifact.sha256, variant.sha256());
            assert_eq!(artifact.filename, variant.filename());
            assert_eq!(artifact.tensor_count, 851);
            assert_eq!(artifact.file_type, variant.file_type());
            assert_eq!(artifact.license, "Apache-2.0");
            assert_eq!(artifact.chunk_bytes, None);
            assert_eq!(artifact.chunk_sha256s, None);
            assert_eq!(
                artifact.path,
                data.join("models")
                    .join(&artifact.revision)
                    .join(&artifact.filename)
            );
            assert_eq!(variant.as_str().parse::<Bonsai25Variant>()?, variant);
        }

        assert!("nariq".parse::<Bonsai25Variant>().is_err());
        assert!(bonsai25_model_artifact(Path::new("relative-data"), Bonsai25Variant::Pq2).is_err());
        assert!(
            bonsai25_model_artifact(
                Path::new(r"%LOCALAPPDATA%\SwiftBonsai2"),
                Bonsai25Variant::Ptq1
            )
            .is_err()
        );
        assert!(
            bonsai25_model_artifact_from_lock(
                data,
                Bonsai25Variant::Pq2,
                &BONSAI_2_5_LOCK.replace("7253354848", "7253354849"),
            )
            .is_err()
        );
        assert!(
            bonsai25_model_artifact_from_lock(
                data,
                Bonsai25Variant::Ptq1,
                &BONSAI_2_5_LOCK.replace("\"runtimeWired\": false", "\"runtimeWired\": true"),
            )
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn bonsai25_lock_accepts_optional_chunk_manifests_and_rejects_malformed_ones()
    -> Result<(), String> {
        let data = Path::new(if cfg!(windows) {
            r"C:\BonsaiData"
        } else {
            "/var/lib/bonsai"
        });
        let chunk_bytes: usize = 64 * 1024 * 1024;
        let chunk_bytes_u64 = u64::try_from(chunk_bytes).map_err(|_| "chunk_bytes overflow")?;
        let expected_chunks = usize::try_from(
            Bonsai25Variant::Pq2
                .weight_bytes()
                .div_ceil(chunk_bytes_u64),
        )
        .map_err(|_| "chunk count overflow")?;
        let digests = vec!["a".repeat(64); expected_chunks];

        let mut valid: serde_json::Value =
            serde_json::from_str(BONSAI_2_5_LOCK).map_err(|error| error.to_string())?;
        valid["variants"]["pq2"]["chunkBytes"] = serde_json::json!(chunk_bytes);
        valid["variants"]["pq2"]["chunkSha256s"] = serde_json::json!(digests);
        let artifact = bonsai25_model_artifact_from_lock(
            data,
            Bonsai25Variant::Pq2,
            &serde_json::to_string(&valid).map_err(|error| error.to_string())?,
        )?;
        assert_eq!(artifact.chunk_bytes, Some(chunk_bytes));
        assert_eq!(
            artifact.chunk_sha256s.as_ref().map(Vec::len),
            Some(expected_chunks)
        );

        let mut only_size = valid.clone();
        if let Some(obj) = only_size["variants"]["pq2"].as_object_mut() {
            obj.remove("chunkSha256s");
        }
        assert!(
            bonsai25_model_artifact_from_lock(
                data,
                Bonsai25Variant::Pq2,
                &serde_json::to_string(&only_size).map_err(|error| error.to_string())?,
            )
            .is_err()
        );

        let mut wrong_count = valid.clone();
        wrong_count["variants"]["pq2"]["chunkSha256s"] = serde_json::json!(["a".repeat(64)]);
        assert!(
            bonsai25_model_artifact_from_lock(
                data,
                Bonsai25Variant::Pq2,
                &serde_json::to_string(&wrong_count).map_err(|error| error.to_string())?,
            )
            .is_err()
        );

        let mut sub_4mib = valid;
        sub_4mib["variants"]["pq2"]["chunkBytes"] = serde_json::json!(1024_u64 * 1024);
        assert!(
            bonsai25_model_artifact_from_lock(
                data,
                Bonsai25Variant::Pq2,
                &serde_json::to_string(&sub_4mib).map_err(|error| error.to_string())?,
            )
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn dynamic_default_selects_ptq1_when_super_memory_constrained_and_respects_cache()
    -> Result<(), String> {
        use crate::context::{
            SwiftCalibratedOptions, SwiftCalibratedReuse, SwiftContextPlan, SwiftContextProfile,
            SwiftWeightPacking,
        };

        const GIB: u64 = 1024 * 1024 * 1024;
        let options = SwiftCalibratedOptions {
            weight_packing: SwiftWeightPacking::Bonsai25Pq2,
            request_vision: true,
            prefill_tokens: Some(16),
            chunkwise_gdn_wy_f32: true,
            speculation_tokens: None,
            prompt_reuse: SwiftCalibratedReuse::PromptEndAndLatest {
                checkpoints: 1,
                checkpoint_bytes: None,
            },
        };
        let profile = SwiftContextProfile::NativeRsq4Full;
        let min_probe_options = SwiftCalibratedOptions {
            request_vision: false,
            ..options
        };
        let min_pq2_threshold = {
            let mut probe = 15 * GIB;
            while SwiftContextPlan::for_calibrated_execution(probe, profile, min_probe_options)
                .is_ok()
            {
                probe -= 64 * 1024 * 1024;
            }
            probe += 1;
            while SwiftContextPlan::for_calibrated_execution(probe, profile, min_probe_options)
                .is_err()
            {
                probe += 1;
            }
            probe
        };
        assert!(
            SwiftContextPlan::for_calibrated_execution(
                min_pq2_threshold - 1,
                profile,
                min_probe_options,
            )
            .is_err()
        );
        assert!(
            SwiftContextPlan::for_calibrated_execution(
                min_pq2_threshold,
                profile,
                min_probe_options,
            )
            .is_ok()
        );

        // 16 GiB devices always flip to PTQ1 on first load even if available RAM looks close.
        assert_eq!(
            select_default_bonsai25_variant(
                Some(Bonsai25MemorySnapshot {
                    total_physical_bytes: 16 * GIB,
                    available_physical_bytes: 15 * GIB,
                }),
                Bonsai25CachedVariants::default(),
                profile,
                options,
                false,
            ),
            Bonsai25Variant::Ptq1
        );

        // 32 GiB X1 / 48 GiB X2 with healthy available RAM select PQ2 on first load.
        for total in [32 * GIB, 48 * GIB] {
            assert_eq!(
                select_default_bonsai25_variant(
                    Some(Bonsai25MemorySnapshot {
                        total_physical_bytes: total,
                        available_physical_bytes: 20 * GIB,
                    }),
                    Bonsai25CachedVariants::default(),
                    profile,
                    options,
                    false,
                ),
                Bonsai25Variant::Pq2
            );
        }

        // Exact boundary on >18 GiB machine: min_pq2_threshold - 1 triggers PTQ1, min_pq2_threshold stays PQ2.
        assert_eq!(
            select_default_bonsai25_variant(
                Some(Bonsai25MemorySnapshot {
                    total_physical_bytes: 32 * GIB,
                    available_physical_bytes: min_pq2_threshold - 1,
                }),
                Bonsai25CachedVariants::default(),
                profile,
                min_probe_options,
                false,
            ),
            Bonsai25Variant::Ptq1
        );
        assert_eq!(
            select_default_bonsai25_variant(
                Some(Bonsai25MemorySnapshot {
                    total_physical_bytes: 32 * GIB,
                    available_physical_bytes: min_pq2_threshold,
                }),
                Bonsai25CachedVariants::default(),
                profile,
                min_probe_options,
                false,
            ),
            Bonsai25Variant::Pq2
        );

        // Constrained 32 GiB host (e.g. heavy background apps) selects PTQ1 on first load.
        let constrained_snapshot = Bonsai25MemorySnapshot {
            total_physical_bytes: 32 * GIB,
            available_physical_bytes: 11 * GIB,
        };
        let constrained = Some(constrained_snapshot);
        assert!(is_super_memory_constrained_for_pq2(
            constrained_snapshot,
            profile,
            options,
            false
        ));
        assert_eq!(
            select_default_bonsai25_variant(
                constrained,
                Bonsai25CachedVariants::default(),
                profile,
                options,
                false,
            ),
            Bonsai25Variant::Ptq1
        );

        // Single-variant cache stickiness avoids surprise 6-7 GB re-downloads.
        assert_eq!(
            select_default_bonsai25_variant(
                constrained,
                Bonsai25CachedVariants {
                    pq2_present: true,
                    ptq1_present: false,
                },
                profile,
                options,
                false,
            ),
            Bonsai25Variant::Pq2
        );
        assert_eq!(
            select_default_bonsai25_variant(
                Some(Bonsai25MemorySnapshot {
                    total_physical_bytes: 48 * GIB,
                    available_physical_bytes: 40 * GIB,
                }),
                Bonsai25CachedVariants {
                    pq2_present: false,
                    ptq1_present: true,
                },
                profile,
                options,
                false,
            ),
            Bonsai25Variant::Ptq1
        );

        // Both cached: follow live memory constraint check.
        assert_eq!(
            select_default_bonsai25_variant(
                constrained,
                Bonsai25CachedVariants {
                    pq2_present: true,
                    ptq1_present: true,
                },
                profile,
                options,
                false,
            ),
            Bonsai25Variant::Ptq1
        );
        assert_eq!(
            select_default_bonsai25_variant(
                Some(Bonsai25MemorySnapshot {
                    total_physical_bytes: 48 * GIB,
                    available_physical_bytes: 40 * GIB,
                }),
                Bonsai25CachedVariants {
                    pq2_present: true,
                    ptq1_present: true,
                },
                profile,
                options,
                false,
            ),
            Bonsai25Variant::Pq2
        );

        // Missing memory observation defaults safely to PQ2; invalid profile never claims pressure.
        assert_eq!(
            select_default_bonsai25_variant(
                None,
                Bonsai25CachedVariants::default(),
                profile,
                options,
                false,
            ),
            Bonsai25Variant::Pq2
        );
        assert!(!is_super_memory_constrained_for_pq2(
            constrained_snapshot,
            SwiftContextProfile::PublisherF16,
            options,
            false,
        ));
        Ok(())
    }
}
