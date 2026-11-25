//! Pinned artifact policy and lexical path resolution for opt-in `NariQ` `GGUF` models.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::{hex, require, unexpanded_variable};

/// Embedded compiled lock document for `nari-labs/Qwen3.8-27B-NariQ-GGUF`.
pub const NARIQ_LOCK: &str = include_str!("../../../config/nariq-qwen38.lock.json");
/// Additional host CPU scratch headroom (`256 MiB`) budgeted for `NariQ` trellis transforms and banded heads.
pub const NARIQ_TRELLIS_SCRATCH_HEADROOM_BYTES: u64 = 256 * 1024 * 1024;

/// Opt-in `NariQ` trellis quantization bit-rate variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NariqVariant {
    /// `2.5bpw` trellis quantization (`8_443_119_712` bytes, unbanded embedding).
    #[serde(rename = "2.5bpw")]
    Bpw2_5,
    /// `3.0bpw` trellis quantization (`10_123_626_592` bytes, unbanded embedding).
    #[serde(rename = "3.0bpw")]
    Bpw3_0,
    /// `2.0bpw` trellis quantization (`6_753_954_304` bytes, banded embedding).
    #[serde(rename = "2.0bpw")]
    Bpw2_0,
}

impl NariqVariant {
    /// CLI and lock key spelling (`"2.5bpw"`, `"3.0bpw"`, `"2.0bpw"`).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Bpw2_5 => "2.5bpw",
            Self::Bpw3_0 => "3.0bpw",
            Self::Bpw2_0 => "2.0bpw",
        }
    }

    /// Prism `weight_representation` identifier returned by readiness responses.
    #[must_use]
    pub const fn representation_name(self) -> &'static str {
        match self {
            Self::Bpw2_5 => "NariQ-2.5bpw",
            Self::Bpw3_0 => "NariQ-3.0bpw",
            Self::Bpw2_0 => "NariQ-2.0bpw",
        }
    }

    /// Readiness `id` emitted by `prism-inference-server` for this variant.
    #[must_use]
    pub const fn model_id(self) -> &'static str {
        match self {
            Self::Bpw2_5 => "Qwen3.8-27B-NariQ-2.5bpw",
            Self::Bpw3_0 => "Qwen3.8-27B-NariQ-3.0bpw",
            Self::Bpw2_0 => "Qwen3.8-27B-NariQ-2.0bpw",
        }
    }

    /// Exact pinned `GGUF` byte length.
    #[must_use]
    pub const fn weight_bytes(self) -> u64 {
        match self {
            Self::Bpw2_5 => 8_443_119_712,
            Self::Bpw3_0 => 10_123_626_592,
            Self::Bpw2_0 => 6_753_954_304,
        }
    }

    /// Exact pinned lowercase SHA-256 hex digest.
    #[must_use]
    pub const fn sha256(self) -> &'static str {
        match self {
            Self::Bpw2_5 => "2445372e6de20620b940fbfab53aa718af9ba741f202da30cd2f02b5df78d05c",
            Self::Bpw3_0 => "1eaad934aa05579b8eba47ed9b26450392f9875fc2354efe8cfa8b29be8f4419",
            Self::Bpw2_0 => "3a9e4955329dba8ba0be5509a9552220e5ccdae2f66bcc72c21c4580cd49bbb2",
        }
    }

    /// Exact immutable Hugging Face artifact filename.
    #[must_use]
    pub const fn filename(self) -> &'static str {
        match self {
            Self::Bpw2_5 => "Qwen3.8-27B-NariQ-2.5bpw.gguf",
            Self::Bpw3_0 => "Qwen3.8-27B-NariQ-3.0bpw.gguf",
            Self::Bpw2_0 => "Qwen3.8-27B-NariQ-2.0bpw.gguf",
        }
    }

    /// Exact expected `GGUF` tensor count.
    #[must_use]
    pub const fn tensor_count(self) -> usize {
        match self {
            Self::Bpw2_5 | Self::Bpw3_0 => 2_710,
            Self::Bpw2_0 => 2_771,
        }
    }
}

impl std::str::FromStr for NariqVariant {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "2.5bpw" => Ok(Self::Bpw2_5),
            "3.0bpw" => Ok(Self::Bpw3_0),
            "2.0bpw" => Ok(Self::Bpw2_0),
            _ => Err("Unsupported NariQ variant; expected `2.5bpw`, `3.0bpw`, or `2.0bpw`.".into()),
        }
    }
}

/// Verified compiled provenance and lexical path for a `NariQ` `GGUF` artifact.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NariqModelArtifact {
    /// Lexical target file path inside `data/models/<revision>/<filename>`.
    pub path: PathBuf,
    /// Selected `NariQ` quantization bit-rate variant.
    pub variant: NariqVariant,
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
    /// Exact expected `GGUF` tensor count.
    pub tensor_count: usize,
    /// Number of banded output head matrices (`16`).
    pub lm_head_bands: u32,
    /// Optional number of banded token embedding matrices (`Some(16)` for `2.0bpw`).
    pub embed_bands: Option<u32>,
    /// Optional fixed chunk size (`>= 4 MiB`) for parallel mapped SHA-256 verification.
    pub chunk_bytes: Option<usize>,
    /// Optional ordered lowercase 64-hex SHA-256 digests covering `0..bytes`.
    pub chunk_sha256s: Option<Vec<String>>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct NariqVariantLock {
    filename: String,
    bytes: u64,
    sha256: String,
    url: String,
    tensor_count: usize,
    lm_head_bands: u32,
    #[serde(default)]
    embed_bands: Option<u32>,
    #[serde(default, rename = "chunkBytes")]
    chunk_bytes: Option<usize>,
    #[serde(default, rename = "chunkSha256s")]
    chunk_sha256s: Option<Vec<String>>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct NariqVariantsMap {
    #[serde(rename = "2.5bpw")]
    bpw_2_5: NariqVariantLock,
    #[serde(rename = "3.0bpw")]
    bpw_3_0: NariqVariantLock,
    #[serde(rename = "2.0bpw")]
    bpw_2_0: NariqVariantLock,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct NariqMmprojReserved {
    filename: String,
    bytes: u64,
    sha256: String,
    url: String,
    runtime_wired: bool,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct NariqLockFile {
    schema_version: u32,
    decision: String,
    repository: String,
    revision: String,
    license: String,
    architecture: String,
    variants: NariqVariantsMap,
    mmproj: NariqMmprojReserved,
}

/// Resolves and validates the pinned `NariQ` model artifact inside `data`.
///
/// # Errors
/// Rejects relative, NUL-containing, or unexpanded data directories, or lockfile drift.
pub fn nariq_model_artifact(
    data: &Path,
    variant: NariqVariant,
) -> Result<NariqModelArtifact, String> {
    let data_text = data
        .to_str()
        .ok_or("DataDirectory must be valid Unicode.")?;
    require(
        data.is_absolute() && !data_text.contains('\0') && !unexpanded_variable(data_text),
        "An expanded absolute data directory is required.",
    )?;
    nariq_model_artifact_from_lock(data, variant, NARIQ_LOCK)
}

fn nariq_model_artifact_from_lock(
    data: &Path,
    variant: NariqVariant,
    lock_json: &str,
) -> Result<NariqModelArtifact, String> {
    let parsed: NariqLockFile =
        serde_json::from_str(lock_json).map_err(|_| "Invalid NariQ lock JSON.".to_owned())?;
    require(
        parsed.schema_version == 1,
        "NariQ lock schemaVersion must be 1.",
    )?;
    require(
        !parsed.decision.is_empty()
            && parsed.architecture == "qwen35"
            && parsed.license == "Apache-2.0",
        "Unexpected NariQ lock metadata.",
    )?;
    require(
        parsed.repository == "nari-labs/Qwen3.8-27B-NariQ-GGUF" && hex(&parsed.revision, 40),
        "Unexpected NariQ upstream repository or revision.",
    )?;
    require(
        !parsed.mmproj.runtime_wired
            && !parsed.mmproj.filename.is_empty()
            && parsed.mmproj.bytes == 931_145_760
            && hex(&parsed.mmproj.sha256, 64)
            && parsed.mmproj.url.starts_with("https://huggingface.co/"),
        "Unexpected NariQ mmproj reservation record.",
    )?;

    let entry = match variant {
        NariqVariant::Bpw2_5 => parsed.variants.bpw_2_5,
        NariqVariant::Bpw3_0 => parsed.variants.bpw_3_0,
        NariqVariant::Bpw2_0 => parsed.variants.bpw_2_0,
    };
    let expected_url = format!(
        "https://huggingface.co/{}/resolve/{}/{}",
        parsed.repository, parsed.revision, entry.filename
    );
    let expected_embed_bands = matches!(variant, NariqVariant::Bpw2_0).then_some(16);
    require(
        entry.filename == variant.filename()
            && entry.bytes == variant.weight_bytes()
            && entry.sha256 == variant.sha256()
            && entry.url == expected_url
            && entry.tensor_count == variant.tensor_count()
            && entry.lm_head_bands == 16
            && entry.embed_bands == expected_embed_bands,
        "NariQ variant lock entry disagrees with compiled identity.",
    )?;
    let (chunk_bytes, chunk_sha256s) =
        crate::validate_chunk_manifest(entry.bytes, entry.chunk_bytes, entry.chunk_sha256s)?
            .map_or((None, None), |(cb, cs)| (Some(cb), Some(cs)));

    Ok(NariqModelArtifact {
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
        lm_head_bands: entry.lm_head_bands,
        embed_bands: entry.embed_bands,
        chunk_bytes,
        chunk_sha256s,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nariq_lock_resolves_all_three_variants_and_rejects_drifted_inputs() -> Result<(), String> {
        let data = Path::new(if cfg!(windows) {
            r"C:\BonsaiData"
        } else {
            "/var/lib/bonsai"
        });
        for variant in [
            NariqVariant::Bpw2_5,
            NariqVariant::Bpw3_0,
            NariqVariant::Bpw2_0,
        ] {
            let artifact = nariq_model_artifact(data, variant)?;
            assert_eq!(artifact.bytes, variant.weight_bytes());
            assert_eq!(artifact.sha256, variant.sha256());
            assert_eq!(artifact.filename, variant.filename());
            assert_eq!(artifact.tensor_count, variant.tensor_count());
            assert_eq!(artifact.lm_head_bands, 16);
            assert_eq!(
                artifact.embed_bands,
                matches!(variant, NariqVariant::Bpw2_0).then_some(16)
            );
            assert_eq!(artifact.chunk_bytes, None);
            assert_eq!(artifact.chunk_sha256s, None);
            assert!(artifact.path.ends_with(variant.filename()));
            assert_eq!(variant.as_str().parse::<NariqVariant>(), Ok(variant));
        }
        assert!("1.5bpw".parse::<NariqVariant>().is_err());
        assert!(nariq_model_artifact(Path::new("relative-data"), NariqVariant::Bpw2_5).is_err());
        assert!(nariq_model_artifact_from_lock(data, NariqVariant::Bpw2_5, "{}").is_err());
        Ok(())
    }

    #[test]
    fn nariq_lock_accepts_optional_chunk_manifests_and_rejects_malformed_ones()
    -> Result<(), String> {
        let data = Path::new(if cfg!(windows) {
            r"C:\BonsaiData"
        } else {
            "/var/lib/bonsai"
        });
        let chunk_bytes: usize = 64 * 1024 * 1024;
        let chunk_bytes_u64 = u64::try_from(chunk_bytes).map_err(|_| "chunk_bytes overflow")?;
        let expected_chunks = usize::try_from(
            NariqVariant::Bpw2_5
                .weight_bytes()
                .div_ceil(chunk_bytes_u64),
        )
        .map_err(|_| "chunk count overflow")?;
        let digests = vec!["b".repeat(64); expected_chunks];

        let mut valid: serde_json::Value =
            serde_json::from_str(NARIQ_LOCK).map_err(|error| error.to_string())?;
        valid["variants"]["2.5bpw"]["chunkBytes"] = serde_json::json!(chunk_bytes);
        valid["variants"]["2.5bpw"]["chunkSha256s"] = serde_json::json!(digests);
        let artifact = nariq_model_artifact_from_lock(
            data,
            NariqVariant::Bpw2_5,
            &serde_json::to_string(&valid).map_err(|error| error.to_string())?,
        )?;
        assert_eq!(artifact.chunk_bytes, Some(chunk_bytes));
        assert_eq!(
            artifact.chunk_sha256s.as_ref().map(Vec::len),
            Some(expected_chunks)
        );

        let mut only_hashes = valid.clone();
        if let Some(obj) = only_hashes["variants"]["2.5bpw"].as_object_mut() {
            obj.remove("chunkBytes");
        }
        assert!(
            nariq_model_artifact_from_lock(
                data,
                NariqVariant::Bpw2_5,
                &serde_json::to_string(&only_hashes).map_err(|error| error.to_string())?,
            )
            .is_err()
        );

        let mut uppercase_hex = valid;
        if let Some(first) = uppercase_hex["variants"]["2.5bpw"]["chunkSha256s"]
            .as_array_mut()
            .and_then(|items| items.first_mut())
        {
            *first = serde_json::json!("B".repeat(64));
        }
        assert!(
            nariq_model_artifact_from_lock(
                data,
                NariqVariant::Bpw2_5,
                &serde_json::to_string(&uppercase_hex).map_err(|error| error.to_string())?,
            )
            .is_err()
        );
        Ok(())
    }
}
