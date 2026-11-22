//! Offline verification of compiled model/runtime identities with retained guards.

use crate::launch_policy::{LaunchProfile, ModelArtifact, reviewed_pins_for_representation};
use bonsai_artifacts::{ChunkedFileSpec, RuntimeSpec, VerifiedFile, VerifiedRuntime};
use std::{
    path::{Path, PathBuf},
    sync::atomic::AtomicBool,
};

pub use bonsai_artifacts::ArtifactError;

fn default_chunk_workers() -> usize {
    #[cfg(windows)]
    {
        std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(4)
            .clamp(1, bonsai_windows_credentials::MAX_PARALLEL_HASH_WORKERS)
    }
    #[cfg(not(windows))]
    {
        1
    }
}

/// Verify an artifact against its exact byte length and SHA-256, using parallel
/// chunked verification when a complete chunk manifest is provided.
#[cfg_attr(
    not(windows),
    expect(
        dead_code,
        reason = "Used by Windows-only launcher guards in addition to verify_selected_model."
    )
)]
pub(crate) fn verify_file_with_optional_chunks(
    path: &Path,
    bytes: u64,
    sha256: &str,
    chunk_bytes: Option<usize>,
    chunk_sha256s: Option<&[String]>,
) -> Result<VerifiedFile, ArtifactError> {
    match (chunk_bytes, chunk_sha256s) {
        (Some(chunk_bytes), Some(chunk_sha256s)) => {
            let spec = ChunkedFileSpec {
                path: path.to_path_buf(),
                bytes,
                sha256: sha256.to_owned(),
                chunk_bytes,
                chunk_sha256s: chunk_sha256s.to_vec(),
            };
            VerifiedFile::verify_chunked_with(
                &spec,
                default_chunk_workers(),
                &AtomicBool::new(false),
            )
        }
        (None, None) => VerifiedFile::verify(path, bytes, sha256),
        (Some(_), None) | (None, Some(_)) => Err(ArtifactError::Configuration),
    }
}

fn model_failure_reason(path: &Path, error: ArtifactError) -> &'static str {
    match error {
        ArtifactError::Path => {
            // This is a diagnostic snapshot after rejection, never an admission
            // decision. Path errors also include denied access and unsafe
            // ancestors, so an arbitrary failed open must not imply corruption.
            if bonsai_artifacts::validate_external_path(path).is_ok()
                && std::fs::symlink_metadata(path)
                    .is_err_and(|error| error.kind() == std::io::ErrorKind::NotFound)
            {
                "missing"
            } else {
                "storage"
            }
        }
        ArtifactError::Identity => "identity",
        ArtifactError::Cancelled => "cancelled",
        ArtifactError::Configuration
        | ArtifactError::Archive
        | ArtifactError::Installation
        | ArtifactError::Architecture => "configuration",
    }
}

fn write_model_failure(
    output: &mut impl std::io::Write,
    path: &Path,
    use_repacked_ptq1: bool,
    error: ArtifactError,
) -> std::io::Result<()> {
    let representation = if use_repacked_ptq1 {
        "ptq1"
    } else {
        "original"
    };
    writeln!(
        output,
        "Selected model verification failed: reason={}, representation={representation}.",
        model_failure_reason(path, error)
    )?;
    writeln!(output, "Selected model verifier error: {error:?}.")
}

/// Verified policy paths and guards retained until the backend child exits.
pub struct VerifiedArtifacts {
    executable: PathBuf,
    model: PathBuf,
    runtime_directory: PathBuf,
    archive_path: PathBuf,
    // Retain archive/installation handles through the Windows backend lifetime.
    // The non-Windows verifier rejects before acquiring any runtime guard.
    #[cfg(windows)]
    runtime: Option<VerifiedRuntime>,
    _model: Option<VerifiedFile>,
}
impl VerifiedArtifacts {
    #[cfg(all(test, windows))]
    pub(crate) const fn synthetic_for_test() -> Self {
        Self {
            executable: PathBuf::new(),
            model: PathBuf::new(),
            runtime_directory: PathBuf::new(),
            archive_path: PathBuf::new(),
            runtime: None,
            _model: None,
        }
    }

    /// Verify the locked reference model and runtime, retaining their read handles.
    ///
    /// # Errors
    /// Rejects policy/path drift, changed artifacts and unsafe installations.
    pub fn verify(data: &Path) -> Result<Self, ArtifactError> {
        Self::verify_for_profile(data, LaunchProfile::ReferenceCpu)
    }

    /// Verify one compiled model/runtime profile through the shared verifier.
    ///
    /// # Errors
    /// Rejects policy/path drift, changed artifacts and unsafe installations.
    pub fn verify_for_profile(data: &Path, profile: LaunchProfile) -> Result<Self, ArtifactError> {
        Self::verify_for_profile_with_representation(data, profile, false)
    }

    /// Verify one compiled model/runtime profile and its explicitly selected weight representation.
    /// PTQ1 uses the pinned current-weight export receipt and retains the same runtime verification.
    ///
    /// # Errors
    /// Rejects incompatible selection, policy/path drift, changed artifacts and unsafe installations.
    pub fn verify_for_profile_with_representation(
        data: &Path,
        profile: LaunchProfile,
        use_repacked_ptq1: bool,
    ) -> Result<Self, ArtifactError> {
        if !cfg!(windows) {
            return Err(ArtifactError::Configuration);
        }
        let pins = reviewed_pins_for_representation(profile, use_repacked_ptq1)
            .map_err(|_| ArtifactError::Configuration)?;
        let spec = RuntimeSpec {
            archive_path: data.join("downloads").join(&pins.runtime.asset.filename),
            runtime_directory: data.join("runtime").join(&pins.runtime.commit),
            archive_bytes: pins.runtime.asset.bytes,
            archive_sha256: pins.runtime.asset.sha256.clone(),
        };
        let (runtime_result, model_result) = std::thread::scope(|scope| {
            let runtime_handle = scope.spawn(|| VerifiedRuntime::verify(&spec));
            let model_result = Self::verify_selected_model(data, profile, use_repacked_ptq1);
            let runtime_result = runtime_handle
                .join()
                .unwrap_or(Err(ArtifactError::Configuration));
            (runtime_result, model_result)
        });
        let runtime = runtime_result?;
        let (model, model_guard) = model_result?;
        Ok(Self {
            executable: runtime.executable().to_owned(),
            model,
            runtime_directory: spec.runtime_directory,
            archive_path: spec.archive_path,
            #[cfg(windows)]
            runtime: Some(runtime),
            _model: Some(model_guard),
        })
    }

    #[cfg(windows)]
    pub(crate) fn verify_model_for_profile(
        data: &Path,
        profile: LaunchProfile,
        use_repacked_ptq1: bool,
    ) -> Result<Self, ArtifactError> {
        let (model, model_guard) = Self::verify_selected_model(data, profile, use_repacked_ptq1)?;
        Ok(Self {
            executable: PathBuf::new(),
            model,
            runtime_directory: PathBuf::new(),
            archive_path: PathBuf::new(),
            runtime: None,
            _model: Some(model_guard),
        })
    }

    fn verify_model_artifact(
        model: &Path,
        artifact: &ModelArtifact,
        use_repacked_ptq1: bool,
    ) -> Result<VerifiedFile, ArtifactError> {
        let (chunk_bytes, chunk_sha256s) = artifact
            .chunk_manifest
            .as_ref()
            .map_or((None, None), |(bytes, digests)| {
                (Some(*bytes), Some(digests.as_slice()))
            });
        verify_file_with_optional_chunks(
            model,
            artifact.bytes,
            &artifact.sha256,
            chunk_bytes,
            chunk_sha256s,
        )
        .inspect_err(|error| {
            // Only fixed classifications enter stderr; paths, file contents
            // and inherited process output never enter the UI error text.
            let _written = write_model_failure(
                &mut std::io::stderr().lock(),
                model,
                use_repacked_ptq1,
                *error,
            );
        })
    }

    fn verify_selected_model(
        data: &Path,
        profile: LaunchProfile,
        use_repacked_ptq1: bool,
    ) -> Result<(PathBuf, VerifiedFile), ArtifactError> {
        let pins = reviewed_pins_for_representation(profile, use_repacked_ptq1)
            .map_err(|_| ArtifactError::Configuration)?;
        let model = data
            .join("models")
            .join(&pins.model.revision)
            .join(&pins.model.filename);
        let guard = Self::verify_model_artifact(&model, &pins.model, use_repacked_ptq1)?;
        Ok((model, guard))
    }

    /// Server path established by the verified archive.
    #[must_use]
    pub fn executable(&self) -> &Path {
        &self.executable
    }
    /// Whole-model path held stable by verification handles.
    #[must_use]
    pub fn model(&self) -> &Path {
        &self.model
    }
    /// Locked read-only model file handle held stable by verification handles.
    #[cfg(windows)]
    pub(crate) fn model_file(&self) -> Option<&std::fs::File> {
        self._model.as_ref().and_then(VerifiedFile::file)
    }
    /// Complete installed runtime directory.
    #[must_use]
    pub fn runtime_directory(&self) -> &Path {
        &self.runtime_directory
    }
    /// Exact installed runtime files established by the retained verifier.
    #[cfg(windows)]
    pub(crate) fn runtime_files(&self) -> Vec<PathBuf> {
        self.runtime
            .iter()
            .flat_map(VerifiedRuntime::files)
            .map(|file| file.path().to_owned())
            .collect()
    }
    /// Locked runtime ZIP path.
    #[must_use]
    pub fn archive_path(&self) -> &Path {
        &self.archive_path
    }
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;
    use crate::{launch_policy::reviewed_pins, test_util::TempDir};
    use std::fs;

    #[test]
    fn model_failure_markers_distinguish_missing_storage_and_identity_without_paths()
    -> std::io::Result<()> {
        let temp = TempDir::new("Bonsai-model-failure")?;
        let directory = temp.path();
        let path = directory.join("private-model-name.gguf");
        for (error, reason) in [
            (ArtifactError::Path, "missing"),
            (ArtifactError::Identity, "identity"),
            (ArtifactError::Configuration, "configuration"),
            (ArtifactError::Cancelled, "cancelled"),
        ] {
            for (ptq1, representation) in [(false, "original"), (true, "ptq1")] {
                let mut output = Vec::new();
                write_model_failure(&mut output, &path, ptq1, error)?;
                assert_eq!(
                    output,
                    format!(
                        "Selected model verification failed: reason={reason}, representation={representation}.\nSelected model verifier error: {error:?}.\n"
                    )
                    .into_bytes()
                );
            }
        }
        fs::write(&path, b"model bytes must never appear in diagnostics")?;
        assert_eq!(model_failure_reason(&path, ArtifactError::Path), "storage");
        fs::create_dir(directory.join(".git"))?;
        assert_eq!(
            model_failure_reason(&directory.join("missing.gguf"), ArtifactError::Path),
            "storage"
        );
        assert_eq!(directory.parent(), Some(std::env::temp_dir().as_path()));
        Ok(())
    }

    #[test]
    fn ptq1_verification_rejects_missing_or_altered_selected_file_without_pq2_fallback()
    -> std::io::Result<()> {
        let temp = TempDir::new("Bonsai-ptq1-identity")?;
        let data = temp.path();
        let profile = LaunchProfile::SwiftPq2Assessment;
        let pins = reviewed_pins(profile).map_err(std::io::Error::other)?;
        let directory = data.join("models").join(&pins.model.revision);
        fs::create_dir_all(&directory)?;
        let pq2 = directory.join(pins.model.filename);
        let ptq1 = directory.join("Swift-Bonsai-2-current-repacked-PTQ1_0.gguf");
        fs::write(&ptq1, b"altered PTQ1")?;
        assert_eq!(
            VerifiedArtifacts::verify_selected_model(data, profile, true).err(),
            Some(ArtifactError::Identity)
        );
        assert_eq!(
            VerifiedArtifacts::verify_selected_model(data, profile, false).err(),
            Some(ArtifactError::Path)
        );
        fs::write(&pq2, b"altered PQ2")?;
        fs::remove_file(&ptq1)?;
        assert_eq!(
            VerifiedArtifacts::verify_selected_model(data, profile, true).err(),
            Some(ArtifactError::Path)
        );
        assert_eq!(
            VerifiedArtifacts::verify_selected_model(data, profile, false).err(),
            Some(ArtifactError::Identity)
        );
        assert_eq!(data.parent(), Some(std::env::temp_dir().as_path()));
        Ok(())
    }

    #[test]
    fn native_profile_checks_model_without_requiring_upstream_runtime() -> std::io::Result<()> {
        let temp = TempDir::new("Bonsai-native-model")?;
        let data = temp.path();
        let profile = LaunchProfile::SwiftPq2Assessment;
        let pins = reviewed_pins(profile).map_err(std::io::Error::other)?;
        let model_directory = data.join("models").join(&pins.model.revision);
        fs::create_dir_all(&model_directory)?;
        let model = model_directory.join(&pins.model.filename);
        fs::write(&model, b"")?;
        // Identity, rather than Path, proves admission reached the model without
        // first opening a C++ archive or requiring its installed runtime tree.
        assert_eq!(
            VerifiedArtifacts::verify_model_for_profile(data, profile, false).err(),
            Some(ArtifactError::Identity)
        );
        assert!(!data.join("downloads").exists());
        assert_eq!(data.parent(), Some(std::env::temp_dir().as_path()));
        Ok(())
    }

    #[cfg(windows)]
    #[test]
    fn profile_verification_opens_only_its_locked_archive() -> std::io::Result<()> {
        let temp = TempDir::new("Bonsai-artifact-profile")?;
        let data = temp.path();
        fs::create_dir(data.join("downloads"))?;
        for profile in [
            LaunchProfile::ReferenceCpu,
            LaunchProfile::SwiftPq2Assessment,
        ] {
            let pins = reviewed_pins(profile).map_err(std::io::Error::other)?;
            fs::create_dir_all(data.join("runtime").join(&pins.runtime.commit))?;
            fs::create_dir_all(data.join("models").join(&pins.model.revision))?;
        }
        for selected in [
            LaunchProfile::ReferenceCpu,
            LaunchProfile::SwiftPq2Assessment,
        ] {
            let pins = reviewed_pins(selected).map_err(std::io::Error::other)?;
            let archive = data.join("downloads").join(&pins.runtime.asset.filename);
            fs::write(&archive, b"")?;
            for candidate in [
                LaunchProfile::ReferenceCpu,
                LaunchProfile::SwiftPq2Assessment,
            ] {
                let expected = if candidate == selected {
                    ArtifactError::Identity
                } else {
                    ArtifactError::Path
                };
                assert_eq!(
                    VerifiedArtifacts::verify_for_profile(data, candidate).err(),
                    Some(expected)
                );
            }
            assert_eq!(
                VerifiedArtifacts::verify(data).err(),
                Some(if selected == LaunchProfile::ReferenceCpu {
                    ArtifactError::Identity
                } else {
                    ArtifactError::Path
                })
            );
            fs::remove_file(&archive)?;
        }
        assert_eq!(data.parent(), Some(std::env::temp_dir().as_path()));
        Ok(())
    }

    fn lower_hex(digest: &[u8; 32]) -> String {
        let mut out = String::with_capacity(64);
        for byte in digest {
            use std::fmt::Write as _;
            let _ = write!(out, "{byte:02x}");
        }
        out
    }

    fn synthetic_gguf_bytes(len: usize) -> Vec<u8> {
        let mut payload: Vec<u8> = (0..len)
            .map(|i| u8::try_from((i.wrapping_mul(29) ^ (i >> 7) ^ 0x5c) & 0xff).unwrap_or(0))
            .collect();
        if payload.len() >= 4 {
            payload[..4].copy_from_slice(b"GGUF");
        }
        payload
    }

    #[test]
    fn chunked_model_verification_succeeds_and_catches_one_byte_corruption_in_any_chunk()
    -> std::io::Result<()> {
        let temp = TempDir::new("Bonsai-chunked-model-verify")?;
        let model_path = temp.path().join("synthetic-model.gguf");
        let chunk_bytes = usize::try_from(bonsai_windows_credentials::MIN_MAPPED_HASH_BYTES)
            .map_err(std::io::Error::other)?;
        let total_len = 2 * chunk_bytes + 4_096;
        let payload = synthetic_gguf_bytes(total_len);
        fs::write(&model_path, &payload)?;

        let whole_sha256 = lower_hex(&bonsai_windows_credentials::Sha256::digest_bytes(&payload)?);
        let locked = bonsai_artifacts::open_locked_file(&model_path)
            .map_err(|error| std::io::Error::other(format!("{error:?}")))?;
        let bytes_u64 = u64::try_from(total_len).map_err(std::io::Error::other)?;
        let raw_chunks = bonsai_windows_credentials::compute_mapped_chunk_digests(
            &locked,
            bytes_u64,
            chunk_bytes,
            4,
            &AtomicBool::new(false),
        )?
        .ok_or_else(|| std::io::Error::other("Expected mapped chunk digests."))?;
        drop(locked);
        let chunk_sha256s: Vec<String> = raw_chunks.iter().map(lower_hex).collect();
        assert_eq!(chunk_sha256s.len(), 3);

        let artifact = ModelArtifact {
            revision: "a".repeat(40),
            filename: "synthetic-model.gguf".to_owned(),
            bytes: bytes_u64,
            sha256: whole_sha256.clone(),
            chunk_manifest: Some((chunk_bytes, chunk_sha256s.clone())),
        };

        // Valid chunked verification acquires read lock and prevents write opens.
        let guard = VerifiedArtifacts::verify_model_artifact(&model_path, &artifact, false)
            .map_err(|error| std::io::Error::other(format!("{error:?}")))?;
        assert!(
            fs::OpenOptions::new()
                .write(true)
                .open(&model_path)
                .is_err()
        );
        drop(guard);

        // Whole-file fallback when chunk_manifest is omitted also succeeds.
        let whole_artifact = ModelArtifact {
            chunk_manifest: None,
            ..artifact.clone()
        };
        let whole_guard =
            VerifiedArtifacts::verify_model_artifact(&model_path, &whole_artifact, false)
                .map_err(|error| std::io::Error::other(format!("{error:?}")))?;
        drop(whole_guard);

        // 1-byte corruption in chunk 0, chunk 1, and final tail chunk must fail with Identity.
        for tamper_offset in [17_usize, chunk_bytes + 17, total_len - 1] {
            let mut tampered = payload.clone();
            tampered[tamper_offset] ^= 0x01;
            fs::write(&model_path, &tampered)?;
            assert_eq!(
                VerifiedArtifacts::verify_model_artifact(&model_path, &artifact, false).err(),
                Some(ArtifactError::Identity),
                "missed corruption at offset {tamper_offset}"
            );
            assert_eq!(
                VerifiedArtifacts::verify_model_artifact(&model_path, &whole_artifact, false).err(),
                Some(ArtifactError::Identity)
            );
        }

        // Restore valid file and verify manifest schema/identity mismatches fail closed.
        fs::write(&model_path, &payload)?;
        let mut wrong_count = artifact.clone();
        wrong_count.chunk_manifest = Some((chunk_bytes, chunk_sha256s[..2].to_vec()));
        assert_eq!(
            VerifiedArtifacts::verify_model_artifact(&model_path, &wrong_count, false).err(),
            Some(ArtifactError::Identity)
        );

        let mut wrong_chunk_bytes = artifact.clone();
        wrong_chunk_bytes.chunk_manifest = Some((chunk_bytes + 4096, chunk_sha256s.clone()));
        assert_eq!(
            VerifiedArtifacts::verify_model_artifact(&model_path, &wrong_chunk_bytes, false).err(),
            Some(ArtifactError::Identity)
        );

        let mut bad_hex = artifact;
        bad_hex.chunk_manifest = Some((
            chunk_bytes,
            vec![
                "bad".to_owned(),
                chunk_sha256s[1].clone(),
                chunk_sha256s[2].clone(),
            ],
        ));
        assert_eq!(
            VerifiedArtifacts::verify_model_artifact(&model_path, &bad_hex, false).err(),
            Some(ArtifactError::Identity)
        );

        assert_eq!(
            verify_file_with_optional_chunks(
                &model_path,
                bytes_u64,
                &whole_sha256,
                Some(chunk_bytes),
                None,
            )
            .err(),
            Some(ArtifactError::Configuration)
        );
        assert_eq!(
            verify_file_with_optional_chunks(
                &model_path,
                bytes_u64,
                &whole_sha256,
                None,
                Some(&chunk_sha256s),
            )
            .err(),
            Some(ArtifactError::Configuration)
        );
        Ok(())
    }
}
