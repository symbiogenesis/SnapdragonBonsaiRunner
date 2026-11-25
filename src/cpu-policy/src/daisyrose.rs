//! Pinned schema, download provenance, and path resolution for DaisyRose multi-rank `LoRA` artifacts.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::{SWIFT_LOCK, hex, parse, require, unexpanded_variable};

/// Immutable remote source provenance for a pinned DaisyRose multi-rank `LoRA` GGUF artifact.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct DaisyRoseLoraSource {
    /// Upstream artifact host kind (`"huggingface"`, `"github"`, or `"github-release"`).
    pub host: String,
    /// Immutable `owner/name` repository identifier.
    pub repository: String,
    /// Immutable release tag for `"github-release"` assets; absent for `"huggingface"` / `"github"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub release: Option<String>,
    /// Lowercase 40-character hexadecimal Git revision hash.
    pub revision: String,
    /// Repository-relative POSIX path to the `.gguf` artifact (must equal `filename` for `"github-release"`).
    pub path: String,
    /// Plain `.gguf` filename matching the final segment of `path`.
    pub filename: String,
    /// Exact immutable HTTPS download URL matching `host`, `repository`, `release`/`revision`, and `path`.
    pub url: String,
    /// Short ASCII license identifier (for example `"Apache-2.0"`).
    pub license: String,
}

/// Pinned schema for a promoted stackable DaisyRose multi-rank `LoRA` GGUF artifact lock.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct DaisyRoseLoraLock {
    /// Lock schema version (`1`).
    pub schema_version: u32,
    /// Target policy envelope (`"npu-friendly"` or `"full-matrix"`).
    pub envelope: String,
    /// Factor storage dtype (`"f16"` or `"f32"`).
    pub dtype: String,
    /// Maximum rank across all verified targets (`1..=64`).
    pub max_rank: u16,
    /// Number of paired target matrices.
    pub target_count: usize,
    /// Exact locked byte length of the GGUF artifact.
    pub bytes: u64,
    /// Lowercase 64-hex SHA-256 of the GGUF artifact.
    pub sha256: String,
    /// Lowercase 64-hex SHA-256 of the compatible base model.
    pub base_model_sha256: String,
    /// Optional immutable remote provenance for automatic setup and `provision`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<DaisyRoseLoraSource>,
    /// Whether this verified artifact is approved for automatic first-run and default launch use.
    #[serde(default)]
    pub default_enabled: bool,
    /// Recommended default inference scale expressed in tenths (`1..=20` when `default_enabled`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_scale_tenths: Option<u8>,
}

/// Resolved external path and immutable identity for a verified DaisyRose multi-rank `LoRA` artifact.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DaisyRoseLoraArtifact {
    /// Lexical target file path inside `data/adapters/<revision>/<filename>`.
    pub path: PathBuf,
    /// Exact locked byte length.
    pub bytes: u64,
    /// Lowercase 64-hex SHA-256 digest of the GGUF file.
    pub sha256: String,
    /// Lowercase 64-hex SHA-256 digest of the compatible Swift base model.
    pub base_model_sha256: String,
    /// Target policy envelope (`"npu-friendly"` or `"full-matrix"`).
    pub envelope: String,
    /// Factor storage dtype (`"f16"` or `"f32"`).
    pub dtype: String,
    /// Maximum rank across all verified target matrices.
    pub max_rank: u16,
    /// Number of verified target matrix pairs.
    pub target_count: usize,
    /// Source repository identifier (`owner/name`).
    pub repository: String,
    /// Optional immutable release tag when hosted as a `"github-release"` asset.
    pub release: Option<String>,
    /// Pinned 40-hex commit revision.
    pub revision: String,
    /// Pinned `.gguf` filename.
    pub filename: String,
    /// Pinned immutable HTTPS download URL.
    pub url: String,
    /// Whether this artifact is approved for automatic setup and default-on launch.
    pub default_enabled: bool,
    /// Optional recommended default scale in tenths (`1..=20` when `default_enabled`).
    pub default_scale_tenths: Option<u8>,
}

impl DaisyRoseLoraArtifact {
    /// Effective default inference scale in tenths (`0..=20`), gated by `default_enabled`.
    ///
    /// Returns `0` when the artifact is not promoted (`default_enabled == false`) or when
    /// `default_scale_tenths` is absent or outside `1..=20`.
    #[must_use]
    pub const fn effective_default_scale_tenths(&self) -> u8 {
        if !self.default_enabled {
            return 0;
        }
        match self.default_scale_tenths {
            Some(tenths) if tenths >= 1 && tenths <= 20 => tenths,
            _ => 0,
        }
    }

    /// Effective default inference scale as `f64`, gated by `default_enabled`.
    #[must_use]
    pub fn effective_default_scale_f64(&self) -> f64 {
        f64::from(self.effective_default_scale_tenths()) / 10.0
    }

    /// Effective default inference scale as `f32`, gated by `default_enabled`.
    #[must_use]
    pub fn effective_default_scale_f32(&self) -> f32 {
        f32::from(self.effective_default_scale_tenths()) / 10.0
    }

    /// Resolve and validate a [`DaisyRoseLoraArtifact`] from an explicit lock JSON string
    /// against the base model `SHA-256` of a pinned [`crate::NariqVariant`].
    ///
    /// This performs no filesystem I/O, downloads, or model loading.
    ///
    /// # Errors
    /// Rejects invalid data directories, invalid lock JSON, missing `source` provenance,
    /// or `baseModelSha256` drift against `variant.sha256()`.
    pub fn for_nariq_lock(
        data: &Path,
        variant: crate::NariqVariant,
        lock_json: &str,
    ) -> Result<Self, String> {
        daisyrose_lora_artifact_for_base_sha256(data, lock_json, variant.sha256())
    }

    /// Resolve and validate a [`DaisyRoseLoraArtifact`] from an explicit lock JSON string
    /// against the base model `SHA-256` of a pinned [`crate::Bonsai25Variant`].
    ///
    /// This performs no filesystem I/O, downloads, or model loading.
    ///
    /// # Errors
    /// Rejects invalid data directories, invalid lock JSON, missing `source` provenance,
    /// or `baseModelSha256` drift against `variant.sha256()`.
    pub fn for_bonsai25_lock(
        data: &Path,
        variant: crate::Bonsai25Variant,
        lock_json: &str,
    ) -> Result<Self, String> {
        daisyrose_lora_artifact_for_base_sha256(data, lock_json, variant.sha256())
    }
}

fn atom(value: &str) -> bool {
    !value.is_empty()
        && value != "."
        && value != ".."
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"_.-".contains(&byte))
}

fn validate_source(source: &DaisyRoseLoraSource) -> Result<(), String> {
    let valid_repo = source
        .repository
        .split_once('/')
        .is_some_and(|(owner, name)| atom(owner) && atom(name));
    require(
        valid_repo,
        "DaisyRose source repository must be owner/name using safe ASCII atoms.",
    )?;
    require(
        hex(&source.revision, 40),
        "DaisyRose source revision must be a lowercase 40-hex commit hash.",
    )?;
    require(
        atom(&source.filename) && source.filename.ends_with(".gguf"),
        "DaisyRose source filename must be a plain .gguf filename.",
    )?;
    let valid_path = !source.path.is_empty()
        && !source.path.contains('\\')
        && !source.path.starts_with('/')
        && !source.path.ends_with('/')
        && source.path.split('/').all(atom)
        && source.path.rsplit('/').next() == Some(source.filename.as_str());
    require(
        valid_path,
        "DaisyRose source path must be a relative POSIX path ending in filename.",
    )?;
    let expected_url = match source.host.as_str() {
        "huggingface" => {
            require(
                source.release.is_none(),
                "DaisyRose source release is only valid for github-release host.",
            )?;
            format!(
                "https://huggingface.co/{}/resolve/{}/{}",
                source.repository, source.revision, source.path
            )
        }
        "github" => {
            require(
                source.release.is_none(),
                "DaisyRose source release is only valid for github-release host.",
            )?;
            format!(
                "https://raw.githubusercontent.com/{}/{}/{}",
                source.repository, source.revision, source.path
            )
        }
        "github-release" => {
            let tag = source
                .release
                .as_deref()
                .ok_or("DaisyRose github-release source requires release tag.")?;
            require(
                atom(tag) && tag.len() <= 128,
                "DaisyRose github-release tag must be a safe ASCII atom (<= 128 bytes).",
            )?;
            require(
                source.path == source.filename,
                "DaisyRose github-release source path must equal filename.",
            )?;
            format!(
                "https://github.com/{}/releases/download/{}/{}",
                source.repository, tag, source.filename
            )
        }
        _ => return Err("Unsupported DaisyRose source host.".into()),
    };
    require(
        source.url == expected_url,
        "DaisyRose source URL disagrees with host, repository, revision, and path.",
    )?;
    require(
        !source.license.trim().is_empty()
            && source.license.len() <= 64
            && source
                .license
                .bytes()
                .all(|byte| (0x20..=0x7e).contains(&byte)),
        "DaisyRose source license must be a non-empty printable ASCII string (<= 64 bytes).",
    )?;
    Ok(())
}

fn validate_data_dir(data: &Path) -> Result<(), String> {
    let data_text = data.to_str().ok_or("Invalid data directory.")?;
    require(
        data.is_absolute() && !data_text.contains('\0') && !unexpanded_variable(data_text),
        "An expanded absolute data directory is required.",
    )
}

/// Parse and validate a [`DaisyRoseLoraLock`] JSON document.
///
/// # Errors
/// Rejects malformed JSON, unsupported schema versions, invalid envelopes or dtypes,
/// out-of-range ranks, empty targets/bytes, non-lowercase-hex digests, invalid source
/// provenance, out-of-range `defaultScaleTenths`, or `defaultEnabled` without an
/// `npu-friendly` artifact, source provenance, and nonzero default scale.
pub fn parse_daisyrose_lora_lock(json: &str) -> Result<DaisyRoseLoraLock, String> {
    let lock: DaisyRoseLoraLock =
        serde_json::from_str(json).map_err(|_| "Invalid DaisyRose LoRA lock JSON.".to_owned())?;
    require(
        lock.schema_version == 1,
        "Unsupported DaisyRose lock schema version.",
    )?;
    require(
        matches!(lock.envelope.as_str(), "npu-friendly" | "full-matrix"),
        "Unsupported DaisyRose lock envelope.",
    )?;
    require(
        matches!(lock.dtype.as_str(), "f16" | "f32"),
        "Unsupported DaisyRose lock dtype.",
    )?;
    require(
        (1..=64).contains(&lock.max_rank),
        "DaisyRose lock maxRank must be within 1..=64.",
    )?;
    require(
        lock.target_count > 0 && lock.bytes > 0,
        "DaisyRose lock requires non-zero targetCount and bytes.",
    )?;
    require(
        hex(&lock.sha256, 64) && hex(&lock.base_model_sha256, 64),
        "DaisyRose lock digests must be lowercase 64-hex SHA-256 strings.",
    )?;
    if let Some(source) = &lock.source {
        validate_source(source)?;
    }
    if let Some(tenths) = lock.default_scale_tenths {
        require(
            tenths <= 20,
            "DaisyRose lock defaultScaleTenths must be within 0..=20.",
        )?;
    }
    if lock.default_enabled {
        require(
            lock.source.is_some(),
            "Default-enabled DaisyRose lock requires source provenance.",
        )?;
        require(
            lock.envelope == "npu-friendly",
            "Default-enabled DaisyRose lock must use the npu-friendly envelope.",
        )?;
        require(
            lock.default_scale_tenths
                .is_some_and(|tenths| (1..=20).contains(&tenths)),
            "Default-enabled DaisyRose lock requires defaultScaleTenths within 1..=20.",
        )?;
    }
    Ok(lock)
}

/// Resolve and validate a [`DaisyRoseLoraArtifact`] from an explicit lock JSON string against
/// an explicit lowercase 64-hex base model `SHA-256` digest.
///
/// This performs no filesystem I/O, downloads, or model loading.
///
/// # Errors
/// Rejects invalid data directories, non-64-hex `expected_base_sha256` digests, invalid lock JSON,
/// missing `source` provenance, or `baseModelSha256` drift against `expected_base_sha256`.
#[expect(
    unreachable_pub,
    reason = "src/cpu-policy/src/lib.rs is digest-pinned; callers outside the crate use DaisyRoseLoraArtifact::for_nariq_lock."
)]
pub fn daisyrose_lora_artifact_for_base_sha256(
    data: &Path,
    lock_json: &str,
    expected_base_sha256: &str,
) -> Result<DaisyRoseLoraArtifact, String> {
    validate_data_dir(data)?;
    require(
        hex(expected_base_sha256, 64),
        "Expected DaisyRose baseModelSha256 must be a lowercase 64-hex SHA-256.",
    )?;
    let lock = parse_daisyrose_lora_lock(lock_json)?;
    require(
        lock.base_model_sha256 == expected_base_sha256,
        "DaisyRose lock baseModelSha256 differs from the expected base model.",
    )?;
    let source = lock
        .source
        .ok_or("DaisyRose lock requires remote source provenance.")?;
    let path = data
        .join("adapters")
        .join(&source.revision)
        .join(&source.filename);
    Ok(DaisyRoseLoraArtifact {
        path,
        bytes: lock.bytes,
        sha256: lock.sha256,
        base_model_sha256: lock.base_model_sha256,
        envelope: lock.envelope,
        dtype: lock.dtype,
        max_rank: lock.max_rank,
        target_count: lock.target_count,
        repository: source.repository,
        release: source.release,
        revision: source.revision,
        filename: source.filename,
        url: source.url,
        default_enabled: lock.default_enabled,
        default_scale_tenths: lock.default_scale_tenths,
    })
}

/// Resolve and validate a [`DaisyRoseLoraArtifact`] from an explicit lock JSON string.
///
/// This performs no filesystem I/O, downloads, or model loading.
///
/// # Errors
/// Rejects invalid data directories, invalid lock JSON, missing `source` provenance,
/// or `baseModelSha256` drift against the compiled Swift PQ2 model lock.
pub fn daisyrose_lora_artifact_from_lock(
    data: &Path,
    lock_json: &str,
) -> Result<DaisyRoseLoraArtifact, String> {
    let swift_lock = parse(SWIFT_LOCK)?;
    let expected_base_sha256 = swift_lock["model"]["sha256"]
        .as_str()
        .ok_or("Pinned Swift model lock is missing model.sha256.")?;
    daisyrose_lora_artifact_for_base_sha256(data, lock_json, expected_base_sha256)
}

/// Resolve the compiled pinned DaisyRose multi-rank `LoRA` artifact if one has been promoted.
///
/// Remains `Ok(None)` until zero-retraining extraction / direct weight editing (`DR-E1`/`DR-W1`), `DR-G2`, and Prism `v0.1.33+` promotion
/// publish a verified `config/daisyrose-lora.lock.json` artifact.
///
/// # Errors
/// Rejects non-absolute, NUL-containing, or unexpanded data directories.
pub fn pinned_daisyrose_lora_artifact(
    data: &Path,
) -> Result<Option<DaisyRoseLoraArtifact>, String> {
    validate_data_dir(data)?;
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::{
        DaisyRoseLoraArtifact, daisyrose_lora_artifact_for_base_sha256,
        daisyrose_lora_artifact_from_lock, parse_daisyrose_lora_lock,
        pinned_daisyrose_lora_artifact,
    };
    use serde_json::{Value, json};
    use std::path::Path;

    const SWIFT_MODEL_SHA: &str =
        "5912bb739217cf25b4283a098c7b893d9baff36bdf2e72d3dbf4cb999d5633d5";

    fn sample_full_lock(host: &str, url: &str) -> Value {
        json!({
            "schemaVersion": 1,
            "envelope": "npu-friendly",
            "dtype": "f16",
            "maxRank": 16,
            "targetCount": 128,
            "bytes": 67_108_864_u64,
            "sha256": "a".repeat(64),
            "baseModelSha256": SWIFT_MODEL_SHA,
            "source": {
                "host": host,
                "repository": "symbiogenesis/DaisyRose-Swift-LoRA",
                "revision": "0123456789abcdef0123456789abcdef01234567",
                "path": "gguf/daisyrose-npu-r16.gguf",
                "filename": "daisyrose-npu-r16.gguf",
                "url": url,
                "license": "Apache-2.0"
            },
            "defaultEnabled": true,
            "defaultScaleTenths": 10
        })
    }

    #[test]
    fn validates_minimal_and_full_daisyrose_locks_and_rejects_drift() -> Result<(), String> {
        let sha = "a".repeat(64);
        let base_sha = "b".repeat(64);
        let minimal = format!(
            r#"{{
                "schemaVersion": 1,
                "envelope": "npu-friendly",
                "dtype": "f16",
                "maxRank": 16,
                "targetCount": 48,
                "bytes": 67108864,
                "sha256": "{sha}",
                "baseModelSha256": "{base_sha}"
            }}"#
        );
        let parsed = parse_daisyrose_lora_lock(&minimal)?;
        assert_eq!(parsed.max_rank, 16);
        assert_eq!(parsed.target_count, 48);
        assert!(parsed.source.is_none());
        assert!(!parsed.default_enabled);

        assert!(
            parse_daisyrose_lora_lock(&minimal.replace(r#""maxRank": 16"#, r#""maxRank": 0"#))
                .is_err()
        );
        assert!(parse_daisyrose_lora_lock(&minimal.replace("npu-friendly", "other")).is_err());
        assert!(parse_daisyrose_lora_lock(&minimal.replace(&sha, &"A".repeat(64))).is_err());

        let hf_url = "https://huggingface.co/symbiogenesis/DaisyRose-Swift-LoRA/resolve/0123456789abcdef0123456789abcdef01234567/gguf/daisyrose-npu-r16.gguf";
        let gh_url = "https://raw.githubusercontent.com/symbiogenesis/DaisyRose-Swift-LoRA/0123456789abcdef0123456789abcdef01234567/gguf/daisyrose-npu-r16.gguf";
        let gh_rel_url = "https://github.com/symbiogenesis/DaisyRose-Swift-LoRA/releases/download/v0.1.0-daisyrose-r16/daisyrose-npu-r16.gguf";
        let hf_lock = sample_full_lock("huggingface", hf_url);
        let gh_lock = sample_full_lock("github", gh_url);
        let mut gh_rel_lock = sample_full_lock("github-release", gh_rel_url);
        gh_rel_lock["source"]["release"] = json!("v0.1.0-daisyrose-r16");
        gh_rel_lock["source"]["path"] = json!("daisyrose-npu-r16.gguf");
        assert!(parse_daisyrose_lora_lock(&hf_lock.to_string()).is_ok());
        assert!(parse_daisyrose_lora_lock(&gh_lock.to_string()).is_ok());
        assert!(parse_daisyrose_lora_lock(&gh_rel_lock.to_string()).is_ok());

        for mut bad in [
            {
                let mut v = hf_lock.clone();
                v["source"]["url"] = json!(gh_url);
                v
            },
            {
                let mut v = hf_lock.clone();
                v["source"]["release"] = json!("v0.1.0");
                v
            },
            {
                let mut v = gh_lock;
                v["source"]["release"] = json!("v0.1.0");
                v
            },
            {
                let mut v = gh_rel_lock.clone();
                v["source"]
                    .as_object_mut()
                    .ok_or("Expected source object.")?
                    .remove("release");
                v
            },
            {
                let mut v = gh_rel_lock.clone();
                v["source"]["release"] = json!("../bad-tag");
                v
            },
            {
                let mut v = gh_rel_lock.clone();
                v["source"]["release"] = json!("t".repeat(129));
                v
            },
            {
                let mut v = gh_rel_lock.clone();
                v["source"]["path"] = json!("gguf/daisyrose-npu-r16.gguf");
                v
            },
            {
                let mut v = gh_rel_lock.clone();
                v["source"]["url"] = json!(hf_url);
                v
            },
            {
                let mut v = gh_rel_lock;
                v["source"]["revision"] = json!("not-hex");
                v
            },
            {
                let mut v = hf_lock.clone();
                v["source"]["path"] = json!("../escape.gguf");
                v
            },
            {
                let mut v = hf_lock.clone();
                v["source"]["filename"] = json!("wrong-name.gguf");
                v
            },
            {
                let mut v = hf_lock.clone();
                v["source"]["revision"] = json!("main");
                v
            },
            {
                let mut v = hf_lock.clone();
                v["envelope"] = json!("full-matrix");
                v
            },
            {
                let mut v = hf_lock.clone();
                v["defaultScaleTenths"] = json!(0);
                v
            },
            {
                let mut v = hf_lock.clone();
                v["defaultScaleTenths"] = json!(21);
                v
            },
            {
                let mut v = hf_lock;
                v.as_object_mut()
                    .ok_or("Expected object.")?
                    .remove("source");
                v
            },
        ] {
            let text = bad.to_string();
            assert!(
                parse_daisyrose_lora_lock(&text).is_err(),
                "Expected rejection for {bad}"
            );
            let _ = &mut bad;
        }
        Ok(())
    }

    #[test]
    fn resolves_artifact_path_and_keeps_pinned_artifact_unpromoted() -> Result<(), String> {
        let data = if cfg!(windows) {
            Path::new(r"C:\External\SwiftData")
        } else {
            Path::new("/tmp/swift-data")
        };
        assert_eq!(pinned_daisyrose_lora_artifact(data)?, None);
        assert!(pinned_daisyrose_lora_artifact(Path::new("relative-data")).is_err());
        assert!(pinned_daisyrose_lora_artifact(Path::new(r"%LOCALAPPDATA%\SwiftBonsai2")).is_err());

        let hf_url = "https://huggingface.co/symbiogenesis/DaisyRose-Swift-LoRA/resolve/0123456789abcdef0123456789abcdef01234567/gguf/daisyrose-npu-r16.gguf";
        let valid = sample_full_lock("huggingface", hf_url);
        let resolved = daisyrose_lora_artifact_from_lock(data, &valid.to_string())?;
        assert_eq!(
            resolved.path,
            data.join("adapters")
                .join("0123456789abcdef0123456789abcdef01234567")
                .join("daisyrose-npu-r16.gguf")
        );
        assert_eq!(resolved.url, hf_url);
        assert_eq!(resolved.release, None);
        assert!(resolved.default_enabled);
        assert_eq!(resolved.default_scale_tenths, Some(10));

        let gh_rel_url = "https://github.com/symbiogenesis/DaisyRose-Swift-LoRA/releases/download/v0.1.0-daisyrose-r16/daisyrose-npu-r16.gguf";
        let mut gh_rel = sample_full_lock("github-release", gh_rel_url);
        gh_rel["source"]["release"] = json!("v0.1.0-daisyrose-r16");
        gh_rel["source"]["path"] = json!("daisyrose-npu-r16.gguf");
        let resolved_gh_rel = daisyrose_lora_artifact_from_lock(data, &gh_rel.to_string())?;
        assert_eq!(
            resolved_gh_rel.path,
            data.join("adapters")
                .join("0123456789abcdef0123456789abcdef01234567")
                .join("daisyrose-npu-r16.gguf")
        );
        assert_eq!(
            resolved_gh_rel.release.as_deref(),
            Some("v0.1.0-daisyrose-r16")
        );
        assert_eq!(resolved_gh_rel.url, gh_rel_url);
        assert!(resolved_gh_rel.default_enabled);
        assert_eq!(resolved_gh_rel.default_scale_tenths, Some(10));

        let mut wrong_base = valid.clone();
        wrong_base["baseModelSha256"] = json!("c".repeat(64));
        assert!(daisyrose_lora_artifact_from_lock(data, &wrong_base.to_string()).is_err());

        let mut missing_source = valid;
        missing_source
            .as_object_mut()
            .ok_or("Expected object.")?
            .remove("source");
        missing_source["defaultEnabled"] = json!(false);
        assert!(daisyrose_lora_artifact_from_lock(data, &missing_source.to_string()).is_err());
        Ok(())
    }

    #[test]
    fn effective_default_scale_helpers_gate_on_default_enabled() -> Result<(), String> {
        let data = if cfg!(windows) {
            Path::new(r"C:\External\SwiftData")
        } else {
            Path::new("/tmp/swift-data")
        };
        let hf_url = "https://huggingface.co/symbiogenesis/DaisyRose-Swift-LoRA/resolve/0123456789abcdef0123456789abcdef01234567/gguf/daisyrose-npu-r16.gguf";
        let promoted = daisyrose_lora_artifact_from_lock(
            data,
            &sample_full_lock("huggingface", hf_url).to_string(),
        )?;
        assert_eq!(promoted.effective_default_scale_tenths(), 10);
        assert!((promoted.effective_default_scale_f64() - 1.0).abs() < 1e-12);
        assert!((promoted.effective_default_scale_f32() - 1.0).abs() < f32::EPSILON);

        let mut unpromoted = promoted.clone();
        unpromoted.default_enabled = false;
        assert_eq!(unpromoted.effective_default_scale_tenths(), 0);
        assert!((unpromoted.effective_default_scale_f64() - 0.0).abs() < 1e-12);
        assert!((unpromoted.effective_default_scale_f32() - 0.0).abs() < f32::EPSILON);

        for invalid in [None, Some(0), Some(21)] {
            let mut mutated = promoted.clone();
            mutated.default_scale_tenths = invalid;
            assert_eq!(mutated.effective_default_scale_tenths(), 0);
            assert!((mutated.effective_default_scale_f64() - 0.0).abs() < 1e-12);
            assert!((mutated.effective_default_scale_f32() - 0.0).abs() < f32::EPSILON);
        }
        Ok(())
    }

    #[test]
    fn resolves_nariq_variant_locks_and_rejects_cross_base_drift() -> Result<(), String> {
        let data = if cfg!(windows) {
            Path::new(r"C:\External\SwiftData")
        } else {
            Path::new("/tmp/swift-data")
        };
        let hf_url = "https://huggingface.co/symbiogenesis/DaisyRose-Swift-LoRA/resolve/0123456789abcdef0123456789abcdef01234567/gguf/daisyrose-npu-r16.gguf";
        let pq2_lock = sample_full_lock("huggingface", hf_url);

        for variant in [
            crate::NariqVariant::Bpw2_5,
            crate::NariqVariant::Bpw3_0,
            crate::NariqVariant::Bpw2_0,
        ] {
            let mut nariq_lock = pq2_lock.clone();
            nariq_lock["baseModelSha256"] = json!(variant.sha256());
            let resolved =
                DaisyRoseLoraArtifact::for_nariq_lock(data, variant, &nariq_lock.to_string())?;
            assert_eq!(resolved.base_model_sha256, variant.sha256());
            assert_eq!(resolved.max_rank, 16);
            assert_eq!(
                resolved.path,
                data.join("adapters")
                    .join("0123456789abcdef0123456789abcdef01234567")
                    .join("daisyrose-npu-r16.gguf")
            );

            assert!(daisyrose_lora_artifact_from_lock(data, &nariq_lock.to_string()).is_err());
        }

        assert!(
            DaisyRoseLoraArtifact::for_nariq_lock(
                data,
                crate::NariqVariant::Bpw2_5,
                &pq2_lock.to_string(),
            )
            .is_err()
        );

        let mut bpw25_lock = pq2_lock;
        bpw25_lock["baseModelSha256"] = json!(crate::NariqVariant::Bpw2_5.sha256());
        assert!(
            DaisyRoseLoraArtifact::for_nariq_lock(
                data,
                crate::NariqVariant::Bpw3_0,
                &bpw25_lock.to_string(),
            )
            .is_err()
        );

        for invalid_expected in ["", "abc", &"A".repeat(64), &format!("{}g", "a".repeat(63))] {
            assert!(
                daisyrose_lora_artifact_for_base_sha256(
                    data,
                    &bpw25_lock.to_string(),
                    invalid_expected,
                )
                .is_err()
            );
        }
        Ok(())
    }
}
