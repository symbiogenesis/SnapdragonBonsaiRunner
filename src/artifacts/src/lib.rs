//! Offline, bounded artifact verification and archive staging for maintenance tools.
//!
//! Retain verification guards until the child has exited. On Windows, their read
//! handles deny file writes and deletion while the verified artifacts are in use.
//! Verification never downloads, writes, or executes code. Staging writes
//! only validated archive entries and never executes code.
#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
#[cfg(test)]
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    fs::{self, File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    path::{Component, Path, PathBuf},
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
};
use zip::ZipArchive;

const MAX_EXPANDED_BYTES: u64 = 512 * 1024 * 1024;
const MAX_ENTRIES: usize = 4096;
// Amortize file-read and CNG calls when verifying multi-gigabyte models. This
// remains bounded independently of artifact size and checks cancellation before
// every read; small standalone files need no full-size allocation.
const HASH_BUFFER_BYTES: usize = 1024 * 1024;

fn verification_buffer(bytes: u64) -> Vec<u8> {
    vec![
        0;
        usize::try_from(bytes)
            .unwrap_or(HASH_BUFFER_BYTES)
            .clamp(1, HASH_BUFFER_BYTES)
    ]
}

mod install;
pub use install::{RuntimeStagingError, stage_runtime, stage_runtime_cleaned};
pub mod lora_adapter;
pub use lora_adapter::{MultiRankEnvelope, VerifiedMultiRankAdapter, verify_multi_rank_lora_gguf};

/// Value-free artifact rejection; no filesystem contents enter diagnostics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArtifactError {
    /// The platform or compiled policy cannot support this verification.
    Configuration,
    /// A path redirects, enters Git storage, or cannot be opened safely.
    Path,
    /// An artifact differs from its exact locked length or SHA-256.
    Identity,
    /// ZIP metadata, names, expansion, or entries are unsupported.
    Archive,
    /// The installation has missing, unexpected or changed entries.
    Installation,
    /// A native executable or DLL is not an ARM64 PE image.
    Architecture,
    /// The caller cancelled verification at a checkpoint.
    Cancelled,
}
impl fmt::Display for ArtifactError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Configuration => {
                "Artifact verification requires the reviewed Windows configuration."
            }
            Self::Path => "Artifact storage could not be opened safely.",
            Self::Identity => "Artifact length or SHA-256 differs from the compiled lock.",
            Self::Archive => "Unsupported or unsafe runtime ZIP archive.",
            Self::Installation => "Installed runtime differs from the verified archive.",
            Self::Architecture => "Expected a native ARM64 PE image.",
            Self::Cancelled => "Artifact verification was cancelled.",
        })
    }
}
impl std::error::Error for ArtifactError {}

/// Runtime identity and installation selected by a caller's reviewed policy.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RuntimeSpec {
    /// Whole runtime ZIP path held stable for the guard's lifetime.
    pub archive_path: PathBuf,
    /// Exact directory whose contents must match the verified ZIP.
    pub runtime_directory: PathBuf,
    /// Locked whole ZIP length in bytes.
    pub archive_bytes: u64,
    /// Locked lowercase whole ZIP SHA-256.
    pub archive_sha256: String,
}

/// Fixed-chunk SHA-256 manifest specification for a locked file.
///
/// When `bytes` and `chunk_bytes` meet the Windows read-only section mapping
/// threshold, [`VerifiedFile::verify_chunked_with`] hashes every contiguous
/// chunk across `0..bytes` in parallel via `chunk_sha256s` without running a
/// redundant second sequential whole-file pass.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ChunkedFileSpec {
    /// Target file path held stable for the guard's lifetime.
    pub path: PathBuf,
    /// Locked exact file length in bytes.
    pub bytes: u64,
    /// Locked lowercase whole-file SHA-256.
    pub sha256: String,
    /// Fixed chunk size in bytes used to partition `0..bytes`.
    pub chunk_bytes: usize,
    /// Locked lowercase SHA-256 digests covering every contiguous chunk in `0..bytes`.
    pub chunk_sha256s: Vec<String>,
}

/// Artifact identity specification for bounded multi-artifact parallel verification.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum FileVerificationSpec {
    /// Verify a file by exact byte length and whole-file lowercase SHA-256.
    Whole {
        /// Target file path held stable for the guard's lifetime.
        path: PathBuf,
        /// Locked exact file length in bytes.
        bytes: u64,
        /// Locked lowercase whole-file SHA-256.
        sha256: String,
    },
    /// Verify a file by exact byte length and fixed-chunk SHA-256 manifest.
    Chunked(ChunkedFileSpec),
}

impl FileVerificationSpec {
    /// Construct a whole-file verification specification.
    #[must_use]
    pub fn whole(path: impl Into<PathBuf>, bytes: u64, sha256: impl Into<String>) -> Self {
        Self::Whole {
            path: path.into(),
            bytes,
            sha256: sha256.into(),
        }
    }

    /// Construct a chunk-manifest verification specification.
    #[must_use]
    pub const fn chunked(spec: ChunkedFileSpec) -> Self {
        Self::Chunked(spec)
    }

    /// Target file path to verify and lock.
    #[must_use]
    pub fn path(&self) -> &Path {
        match self {
            Self::Whole { path, .. } => path,
            Self::Chunked(spec) => &spec.path,
        }
    }
}

/// Identity of one installed file established by verification or installation.
/// A runtime verification guard retains file ownership; installation receipts do not.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedRuntimeFile {
    path: PathBuf,
    bytes: u64,
    sha256: String,
}
impl VerifiedRuntimeFile {
    /// Exact installed file path recorded by verification or installation.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }
    /// Verified installed file length.
    #[must_use]
    pub const fn bytes(&self) -> u64 {
        self.bytes
    }
    /// SHA-256 of the matching ZIP entry and installed file.
    #[must_use]
    pub fn sha256(&self) -> &str {
        &self.sha256
    }
}

/// Verified installed runtime and read handles that keep its files stable.
/// Retain this guard until every child using the runtime has exited.
pub struct VerifiedRuntime {
    executable: PathBuf,
    runtime_directory: PathBuf,
    archive_path: PathBuf,
    files: Vec<VerifiedRuntimeFile>,
    _handles: Vec<File>,
}
impl VerifiedRuntime {
    /// Verify the whole ZIP and installed files, including native ARM64 headers.
    ///
    /// # Errors
    /// Rejects unsupported platforms, unsafe paths, identity drift, unsafe ZIPs,
    /// and installations that differ from their verified archive.
    pub fn verify(spec: &RuntimeSpec) -> Result<Self, ArtifactError> {
        Self::verify_with(spec, || Ok(()))
    }

    /// Verify a runtime while checking caller cancellation between bounded reads.
    /// The checkpoint also runs between archive and installation entries.
    ///
    /// # Errors
    /// Returns artifact rejection or the checkpoint's cancellation error.
    pub fn verify_with(
        spec: &RuntimeSpec,
        mut checkpoint: impl FnMut() -> Result<(), ArtifactError>,
    ) -> Result<Self, ArtifactError> {
        checkpoint()?;
        if !cfg!(windows) {
            return Err(ArtifactError::Configuration);
        }
        let mut handles = Vec::new();
        let mut held_ancestors = BTreeSet::new();
        hold_ancestors(
            &spec.archive_path,
            &mut handles,
            &mut held_ancestors,
            &mut checkpoint,
        )?;
        hold_ancestors(
            &spec.runtime_directory.join("llama-server.exe"),
            &mut handles,
            &mut held_ancestors,
            &mut checkpoint,
        )?;
        let mut archive = open_locked_file(&spec.archive_path)?;
        let mut buffer = vec![0; HASH_BUFFER_BYTES];
        check_locked(
            &mut archive,
            spec.archive_bytes,
            &spec.archive_sha256,
            &mut buffer,
            &mut checkpoint,
        )?;
        let mut zip = ZipArchive::new(archive.try_clone().map_err(|_| ArtifactError::Archive)?)
            .map_err(|_| ArtifactError::Archive)?;
        let (executable, files) = verify_installation(
            &mut zip,
            &spec.runtime_directory,
            &mut handles,
            &mut held_ancestors,
            &mut buffer,
            &mut checkpoint,
        )?;
        handles.push(archive);
        Ok(Self {
            executable,
            runtime_directory: spec.runtime_directory.clone(),
            archive_path: spec.archive_path.clone(),
            files,
            _handles: handles,
        })
    }

    /// Server path established by the verified archive.
    #[must_use]
    pub fn executable(&self) -> &Path {
        &self.executable
    }
    /// Complete installed runtime directory.
    #[must_use]
    pub fn runtime_directory(&self) -> &Path {
        &self.runtime_directory
    }
    /// Locked runtime ZIP path.
    #[must_use]
    pub fn archive_path(&self) -> &Path {
        &self.archive_path
    }
    /// Whether this exact installed file path is held by the guard.
    /// Archive and directory paths are not installed files.
    #[must_use]
    pub fn contains_file(&self, path: &Path) -> bool {
        self.files.iter().any(|file| file.path == path)
    }
    /// Verified installed file identities, ordered by archive path ignoring case.
    #[must_use]
    pub fn files(&self) -> &[VerifiedRuntimeFile] {
        &self.files
    }
}

/// A whole locked file and read handles that keep it and its ancestors stable.
///
/// Retain this guard until every child reading the file has exited.
/// A contained reader retains metadata-only ancestor handles within its existing
/// authority; it requires the host owner's guards and ACL policy to remain in
/// force for its complete lifetime, including the volume root.
pub struct VerifiedFile {
    path: PathBuf,
    _handles: Vec<File>,
}
impl VerifiedFile {
    /// Verify a file's exact byte length and lowercase SHA-256.
    ///
    /// # Errors
    /// Rejects unsupported platforms, unsafe paths, and identity drift.
    pub fn verify(path: &Path, bytes: u64, hash: &str) -> Result<Self, ArtifactError> {
        Self::verify_with(path, bytes, hash, || Ok(()))
    }

    /// Verify a file while checking caller cancellation between bounded reads.
    ///
    /// # Errors
    /// Returns artifact rejection or the checkpoint's cancellation error.
    pub fn verify_with(
        path: &Path,
        bytes: u64,
        hash: &str,
        mut checkpoint: impl FnMut() -> Result<(), ArtifactError>,
    ) -> Result<Self, ArtifactError> {
        checkpoint()?;
        if !cfg!(windows) {
            return Err(ArtifactError::Configuration);
        }
        let mut handles = Vec::new();
        let mut held_ancestors = BTreeSet::new();
        hold_ancestors(path, &mut handles, &mut held_ancestors, &mut checkpoint)?;
        let mut file = open_locked_file_with_access(path, true)?;
        check_locked_preferred(&mut file, bytes, hash, &mut checkpoint)?;
        handles.push(file);
        Ok(Self {
            path: path.to_owned(),
            _handles: handles,
        })
    }

    /// Verify a file against its exact byte length and fixed-chunk SHA-256 manifest.
    ///
    /// # Errors
    /// Rejects unsupported platforms, unsafe paths, invalid chunk manifests, and
    /// identity drift.
    pub fn verify_chunked(spec: &ChunkedFileSpec) -> Result<Self, ArtifactError> {
        Self::verify_chunked_with(spec, default_chunk_workers(), &AtomicBool::new(false))
    }

    /// Verify a file against a fixed-chunk SHA-256 manifest while retaining all
    /// ancestor and file deny-write/deny-delete locks.
    ///
    /// When `spec.bytes` and `spec.chunk_bytes` meet the Windows read-only
    /// section mapping threshold, this verifies every contiguous chunk covering
    /// `0..spec.bytes` in parallel via Windows CNG (`try_verify_locked_file_chunk_digests_mapped_with`)
    /// without performing a redundant second sequential whole-file hash pass.
    /// Smaller files, smaller chunk sizes, or non-mappable files fall back to a
    /// single-pass sequential read verifying both `chunk_sha256s` and `sha256`.
    ///
    /// # Errors
    /// Rejects unsupported platforms, invalid `chunk_bytes`, unsafe paths,
    /// identity or chunk-manifest drift, and `cancel` interruption.
    pub fn verify_chunked_with(
        spec: &ChunkedFileSpec,
        workers: usize,
        cancel: &AtomicBool,
    ) -> Result<Self, ArtifactError> {
        Self::verify_chunked_until(spec, workers, &|| cancel.load(Ordering::Acquire))
    }

    fn verify_chunked_until(
        spec: &ChunkedFileSpec,
        workers: usize,
        is_cancelled: &(impl Fn() -> bool + Sync),
    ) -> Result<Self, ArtifactError> {
        if is_cancelled() {
            return Err(ArtifactError::Cancelled);
        }
        if !cfg!(windows) {
            return Err(ArtifactError::Configuration);
        }
        let chunk_bytes_u64 =
            u64::try_from(spec.chunk_bytes).map_err(|_| ArtifactError::Configuration)?;
        if chunk_bytes_u64 == 0 {
            return Err(ArtifactError::Configuration);
        }

        let mut checkpoint = || {
            if is_cancelled() {
                Err(ArtifactError::Cancelled)
            } else {
                Ok(())
            }
        };
        let mut handles = Vec::new();
        let mut held_ancestors = BTreeSet::new();
        hold_ancestors(
            &spec.path,
            &mut handles,
            &mut held_ancestors,
            &mut checkpoint,
        )?;
        let mut file = open_locked_file_with_access(&spec.path, true)?;
        checkpoint()?;

        if file.metadata().map_err(|_| ArtifactError::Identity)?.len() != spec.bytes {
            return Err(ArtifactError::Identity);
        }
        let expected_chunks = usize::try_from(spec.bytes.div_ceil(chunk_bytes_u64))
            .map_err(|_| ArtifactError::Identity)?;
        if spec.chunk_sha256s.len() != expected_chunks {
            return Err(ArtifactError::Identity);
        }
        let expected_whole_digest = parse_sha256_hex(&spec.sha256)?;
        let expected_chunk_digests: Vec<[u8; 32]> = spec
            .chunk_sha256s
            .iter()
            .map(|digest| parse_sha256_hex(digest))
            .collect::<Result<Vec<_>, _>>()?;

        #[cfg(windows)]
        if chunk_bytes_u64 >= bonsai_windows_credentials::MIN_MAPPED_HASH_BYTES {
            match bonsai_windows_credentials::try_verify_locked_file_chunk_digests_mapped_with(
                &file,
                spec.bytes,
                spec.chunk_bytes,
                &expected_chunk_digests,
                workers,
                is_cancelled,
            ) {
                Ok(Some(true)) => {
                    file.seek(SeekFrom::Start(0))
                        .map_err(|_| ArtifactError::Identity)?;
                    handles.push(file);
                    return Ok(Self {
                        path: spec.path.clone(),
                        _handles: handles,
                    });
                }
                Ok(Some(false)) => return Err(ArtifactError::Identity),
                Ok(None) => {}
                Err(error) => {
                    return Err(
                        if error.kind() == std::io::ErrorKind::Interrupted || is_cancelled() {
                            ArtifactError::Cancelled
                        } else {
                            ArtifactError::Identity
                        },
                    );
                }
            }
        }
        #[cfg(not(windows))]
        let _ = workers;

        check_locked_chunked_buffered(
            &mut file,
            spec.bytes,
            chunk_bytes_u64,
            &expected_whole_digest,
            &expected_chunk_digests,
            &mut checkpoint,
        )?;
        handles.push(file);
        Ok(Self {
            path: spec.path.clone(),
            _handles: handles,
        })
    }

    /// Exact file path held stable by this guard.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Locked read-only regular file handle held stable by this guard.
    #[must_use]
    pub fn file(&self) -> Option<&File> {
        self._handles.last()
    }
}

/// Verify multiple artifacts concurrently across up to four scoped OS worker
/// threads while retaining deterministic input-ordered [`VerifiedFile`] guards.
///
/// Sets an internal `cancel_or_error` flag upon the first peer failure or
/// external `cancel` signal so peer workers abort promptly at their next
/// checkpoint. Any guards already opened before an error or cancellation are
/// dropped before returning.
///
/// # Errors
/// Returns `ArtifactError::Cancelled` when cancelled externally, or the earliest
/// non-cancellation artifact error by input order (`0..specs.len()`).
pub fn verify_files_bounded_parallel(
    specs: &[FileVerificationSpec],
    max_workers: usize,
    cancel: &AtomicBool,
) -> Result<Vec<VerifiedFile>, ArtifactError> {
    if cancel.load(Ordering::Acquire) {
        return Err(ArtifactError::Cancelled);
    }
    if !cfg!(windows) {
        return Err(ArtifactError::Configuration);
    }
    if specs.is_empty() {
        return Ok(Vec::new());
    }

    let file_workers = max_workers.clamp(1, 4).min(specs.len());
    #[cfg(windows)]
    let chunk_workers =
        (bonsai_windows_credentials::MAX_PARALLEL_HASH_WORKERS / file_workers).max(1);
    #[cfg(not(windows))]
    let chunk_workers = 1;

    let next_index = AtomicUsize::new(0);
    let cancel_or_error = AtomicBool::new(false);
    let should_abort = || {
        if cancel.load(Ordering::Acquire) {
            cancel_or_error.store(true, Ordering::Release);
            true
        } else {
            cancel_or_error.load(Ordering::Acquire)
        }
    };

    let worker_task = || -> Vec<(usize, Result<VerifiedFile, ArtifactError>)> {
        let mut local = Vec::with_capacity(specs.len().div_ceil(file_workers));
        loop {
            if should_abort() {
                break;
            }
            let index = next_index.fetch_add(1, Ordering::Relaxed);
            let Some(spec) = specs.get(index) else {
                break;
            };
            let result = match spec {
                FileVerificationSpec::Whole {
                    path,
                    bytes,
                    sha256,
                } => VerifiedFile::verify_with(path, *bytes, sha256, || {
                    if should_abort() {
                        Err(ArtifactError::Cancelled)
                    } else {
                        Ok(())
                    }
                }),
                FileVerificationSpec::Chunked(chunked) => {
                    VerifiedFile::verify_chunked_until(chunked, chunk_workers, &should_abort)
                }
            };
            if result.is_err() {
                cancel_or_error.store(true, Ordering::Release);
                local.push((index, result));
                break;
            }
            local.push((index, result));
        }
        local
    };

    let worker_results = if file_workers == 1 {
        vec![worker_task()]
    } else {
        std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(file_workers - 1);
            for _ in 0..(file_workers - 1) {
                handles.push(scope.spawn(&worker_task));
            }
            let current = worker_task();
            let mut results = Vec::with_capacity(file_workers);
            results.push(current);
            for handle in handles {
                match handle.join() {
                    Ok(items) => results.push(items),
                    Err(_) => {
                        cancel_or_error.store(true, Ordering::Release);
                        // Ensure deterministic failure rather than ignoring a panicked worker.
                        results.push(vec![(0, Err(ArtifactError::Configuration))]);
                    }
                }
            }
            results
        })
    };

    let mut ordered: Vec<Option<Result<VerifiedFile, ArtifactError>>> =
        std::iter::repeat_with(|| None).take(specs.len()).collect();
    let mut panic_fallback = false;
    for worker_items in worker_results {
        for (index, result) in worker_items {
            if let Some(slot) = ordered.get_mut(index) {
                if slot.is_none() {
                    *slot = Some(result);
                } else if matches!(result, Err(ArtifactError::Configuration)) {
                    panic_fallback = true;
                }
            }
        }
    }

    for entry in &ordered {
        if let Some(Err(error)) = entry
            && *error != ArtifactError::Cancelled
        {
            return Err(*error);
        }
    }
    if panic_fallback {
        return Err(ArtifactError::Configuration);
    }
    if cancel.load(Ordering::Acquire) {
        return Err(ArtifactError::Cancelled);
    }

    let mut verified = Vec::with_capacity(specs.len());
    for entry in ordered {
        match entry {
            Some(Ok(guard)) => verified.push(guard),
            Some(Err(error)) => return Err(error),
            None => return Err(ArtifactError::Cancelled),
        }
    }
    Ok(verified)
}

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

fn parse_sha256_hex(digest: &str) -> Result<[u8; 32], ArtifactError> {
    if digest.len() != 64
        || !digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(ArtifactError::Identity);
    }
    let mut decoded = [0_u8; 32];
    hex::decode_to_slice(digest, &mut decoded).map_err(|_| ArtifactError::Identity)?;
    Ok(decoded)
}

fn constant_time_eq_32(lhs: &[u8; 32], rhs: &[u8; 32]) -> bool {
    let mut diff = 0_u8;
    for (&a, &b) in lhs.iter().zip(rhs.iter()) {
        diff |= std::hint::black_box(a ^ b);
    }
    diff == 0
}

fn check_locked_chunked_buffered(
    file: &mut File,
    expected_bytes: u64,
    chunk_bytes_u64: u64,
    expected_whole_digest: &[u8; 32],
    expected_chunk_digests: &[[u8; 32]],
    checkpoint: &mut impl FnMut() -> Result<(), ArtifactError>,
) -> Result<(), ArtifactError> {
    let mut buffer = verification_buffer(expected_bytes);
    let mut whole_hash =
        bonsai_native_hash::Hasher::new().map_err(|_| ArtifactError::Configuration)?;
    let mut chunk_hash: Option<bonsai_native_hash::Hasher> = None;
    let mut total = 0_u64;
    let mut chunk_index = 0_usize;
    let mut chunk_offset = 0_u64;

    loop {
        checkpoint()?;
        let count = match file.read(&mut buffer) {
            Ok(count) => count,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => return Err(ArtifactError::Identity),
        };
        if count == 0 {
            break;
        }
        let count_u64 = u64::try_from(count).map_err(|_| ArtifactError::Identity)?;
        total = total
            .checked_add(count_u64)
            .ok_or(ArtifactError::Identity)?;
        if total > expected_bytes {
            return Err(ArtifactError::Identity);
        }
        let mut slice = &buffer[..count];
        whole_hash
            .update(slice)
            .map_err(|_| ArtifactError::Identity)?;
        while !slice.is_empty() {
            let hasher = match chunk_hash.as_mut() {
                Some(hasher) => hasher,
                None => {
                    let hasher = bonsai_native_hash::Hasher::new()
                        .map_err(|_| ArtifactError::Configuration)?;
                    chunk_hash.insert(hasher)
                }
            };
            let remaining_in_chunk = chunk_bytes_u64 - chunk_offset;
            let take = usize::try_from(remaining_in_chunk)
                .unwrap_or(usize::MAX)
                .min(slice.len());
            let (head, tail) = slice.split_at(take);
            hasher.update(head).map_err(|_| ArtifactError::Identity)?;
            chunk_offset += u64::try_from(take).map_err(|_| ArtifactError::Identity)?;
            slice = tail;
            if chunk_offset == chunk_bytes_u64 {
                let finished = chunk_hash
                    .take()
                    .ok_or(ArtifactError::Identity)?
                    .finish()
                    .map_err(|_| ArtifactError::Identity)?;
                let expected_chunk = expected_chunk_digests
                    .get(chunk_index)
                    .ok_or(ArtifactError::Identity)?;
                if !constant_time_eq_32(&finished, expected_chunk) {
                    return Err(ArtifactError::Identity);
                }
                chunk_index += 1;
                chunk_offset = 0;
            }
        }
    }

    if total != expected_bytes {
        return Err(ArtifactError::Identity);
    }
    if let Some(hasher) = chunk_hash.take() {
        let finished = hasher.finish().map_err(|_| ArtifactError::Identity)?;
        let expected_chunk = expected_chunk_digests
            .get(chunk_index)
            .ok_or(ArtifactError::Identity)?;
        if !constant_time_eq_32(&finished, expected_chunk) {
            return Err(ArtifactError::Identity);
        }
        chunk_index += 1;
    }
    if chunk_index != expected_chunk_digests.len() {
        return Err(ArtifactError::Identity);
    }
    let whole_digest = whole_hash.finish().map_err(|_| ArtifactError::Identity)?;
    if !constant_time_eq_32(&whole_digest, expected_whole_digest) {
        return Err(ArtifactError::Identity);
    }
    file.seek(SeekFrom::Start(0))
        .map_err(|_| ArtifactError::Identity)?;
    Ok(())
}

fn classify_hash_error(error: ArtifactError, fallback: ArtifactError) -> ArtifactError {
    if error == ArtifactError::Cancelled {
        error
    } else {
        fallback
    }
}
/// Open an ordinary external file for reading without following a leaf redirect.
///
/// On Windows the returned handle permits only other readers, preventing writes
/// and deletion of the opened file. This does not retain ancestor directory
/// handles; retain a host-owner [`VerifiedFile`] when the complete path must
/// remain stable, including throughout the lifetime of a contained reader.
///
/// # Errors
/// Rejects unsafe external paths, redirects, non-files and failed opens.
pub fn open_locked_file(path: &Path) -> Result<File, ArtifactError> {
    open_locked_file_with_access(path, false)
}

fn open_locked_file_with_access(path: &Path, sequential: bool) -> Result<File, ArtifactError> {
    validate_read_path(path)?;
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        // FILE_SHARE_READ only; OPEN_REPARSE_POINT prevents following a leaf
        // redirect. Only whole-file verification requests SEQUENTIAL_SCAN;
        // ordinary readers, including random-access model readers, retain their
        // existing cache policy.
        options
            .share_mode(1)
            .custom_flags(0x0020_0000 | if sequential { 0x0800_0000 } else { 0 });
    }
    #[cfg(not(windows))]
    let _ = sequential;
    let file = options.open(path).map_err(|_| ArtifactError::Path)?;
    let metadata = file.metadata().map_err(|_| ArtifactError::Path)?;
    if !metadata.is_file() || is_redirect(&metadata) {
        return Err(ArtifactError::Path);
    }
    Ok(file)
}

fn hold_ancestors(
    path: &Path,
    handles: &mut Vec<File>,
    held_ancestors: &mut BTreeSet<PathBuf>,
    checkpoint: &mut impl FnMut() -> Result<(), ArtifactError>,
) -> Result<(), ArtifactError> {
    checkpoint()?;
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        let contained =
            bonsai_windows_credentials::is_app_container().map_err(|_| ArtifactError::Path)?;
        if contained {
            validate_read_path(path)?;
        } else {
            bonsai_windows_credentials::validate_external_path(path)
                .map_err(|_| ArtifactError::Path)?;
        }
        let chain: Vec<&Path> = path.ancestors().skip(1).collect();
        for ancestor in chain.into_iter().rev() {
            checkpoint()?;
            // The host owner retains its root guard. A contained reader lacks
            // root metadata rights; granting the volume root would broaden its
            // file authority. Its metadata-only handles rely on the host owner's
            // guards and ACL policy for path stability. Host guards additionally
            // request FILE_LIST_DIRECTORY: FILE_READ_ATTRIBUTES alone does not
            // make Windows enforce denied-delete sharing for ancestor renames.
            // No contained listing or volume-root authority is added.
            if (contained && ancestor.parent().is_none()) || held_ancestors.contains(ancestor) {
                continue;
            }
            let directory = OpenOptions::new()
                .read(true)
                .access_mode(if contained { 0x80 } else { 0x81 })
                // Keep directory paths stable without blocking mutable siblings.
                // Renaming a child opens its destination parent for FILE_ADD_FILE;
                // deny-write sharing would reject atomic chat recovery publication.
                // Deny-delete sharing still prevents ancestor replacement, while
                // artifact leaf handles separately deny both writes and deletion.
                .share_mode(3)
                .custom_flags(0x0220_0000)
                .open(ancestor)
                .map_err(|_| ArtifactError::Path)?;
            let metadata = directory.metadata().map_err(|_| ArtifactError::Path)?;
            if !metadata.is_dir() || is_redirect(&metadata) {
                return Err(ArtifactError::Path);
            }
            held_ancestors.insert(ancestor.to_path_buf());
            handles.push(directory);
        }
    }
    #[cfg(not(windows))]
    {
        validate_read_path(path)?;
        let _ = (handles, held_ancestors);
    }
    Ok(())
}

fn validate_read_path(path: &Path) -> Result<(), ArtifactError> {
    #[cfg(windows)]
    {
        bonsai_windows_credentials::validate_read_path(path).map_err(|_| ArtifactError::Path)
    }
    #[cfg(not(windows))]
    {
        validate_external_path(path)
    }
}

fn finish_locked_check(
    file: &mut File,
    digest: &str,
    expected_hash: &str,
) -> Result<(), ArtifactError> {
    if digest != expected_hash {
        return Err(ArtifactError::Identity);
    }
    file.seek(SeekFrom::Start(0))
        .map_err(|_| ArtifactError::Identity)?;
    Ok(())
}

fn try_check_locked_mapped(
    file: &mut File,
    bytes: u64,
    hash: &str,
    checkpoint: &mut impl FnMut() -> Result<(), ArtifactError>,
) -> Result<bool, ArtifactError> {
    checkpoint()?;
    if file.metadata().map_err(|_| ArtifactError::Identity)?.len() != bytes {
        return Err(ArtifactError::Identity);
    }
    #[cfg(windows)]
    {
        if let Some(digest) = bonsai_windows_credentials::try_hash_locked_file_mapped(
            file,
            bytes,
            checkpoint,
            |_| ArtifactError::Identity,
        )? {
            finish_locked_check(file, &digest_hex(digest), hash)?;
            return Ok(true);
        }
    }
    #[cfg(not(windows))]
    let _ = (file, bytes, hash);
    Ok(false)
}

fn check_locked_preferred(
    file: &mut File,
    bytes: u64,
    hash: &str,
    checkpoint: &mut impl FnMut() -> Result<(), ArtifactError>,
) -> Result<(), ArtifactError> {
    if try_check_locked_mapped(file, bytes, hash, checkpoint)? {
        return Ok(());
    }
    let mut buffer = verification_buffer(bytes);
    check_locked_buffered(file, bytes, hash, &mut buffer, checkpoint)
}

fn check_locked_buffered(
    file: &mut File,
    bytes: u64,
    hash: &str,
    buffer: &mut [u8],
    checkpoint: &mut impl FnMut() -> Result<(), ArtifactError>,
) -> Result<(), ArtifactError> {
    let digest = hash_reader(file, bytes, buffer, checkpoint)?;
    finish_locked_check(file, &digest, hash)
}

fn check_locked(
    file: &mut File,
    bytes: u64,
    hash: &str,
    buffer: &mut [u8],
    checkpoint: &mut impl FnMut() -> Result<(), ArtifactError>,
) -> Result<(), ArtifactError> {
    if try_check_locked_mapped(file, bytes, hash, checkpoint)? {
        return Ok(());
    }
    check_locked_buffered(file, bytes, hash, buffer, checkpoint)
}

fn digest_hex(digest: [u8; 32]) -> String {
    hex::encode(digest)
}
fn stream_and_hash_bounded<W: Write + ?Sized>(
    reader: &mut impl Read,
    mut writer: Option<&mut W>,
    expected: u64,
    buffer: &mut [u8],
    checkpoint: &mut impl FnMut() -> Result<(), ArtifactError>,
    read_error: ArtifactError,
    write_error: ArtifactError,
) -> Result<String, ArtifactError> {
    if buffer.is_empty() {
        return Err(ArtifactError::Configuration);
    }
    // Uses Windows CNG on Windows and audited portable SHA-256 elsewhere
    // via the shared native-hash owner without skipping any artifact bytes.
    let mut hash = bonsai_native_hash::Hasher::new().map_err(|_| ArtifactError::Configuration)?;
    let mut total = 0_u64;
    loop {
        checkpoint()?;
        let count = match reader.read(buffer) {
            Ok(count) => count,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => return Err(read_error),
        };
        if count == 0 {
            break;
        }
        total = total
            .checked_add(u64::try_from(count).map_err(|_| read_error)?)
            .ok_or(read_error)?;
        if total > expected {
            return Err(read_error);
        }
        let chunk = &buffer[..count];
        hash.update(chunk).map_err(|_| read_error)?;
        if let Some(output) = writer.as_deref_mut() {
            output.write_all(chunk).map_err(|_| write_error)?;
        }
    }
    if total != expected {
        return Err(read_error);
    }
    let digest = hash.finish().map_err(|_| read_error)?;
    Ok(digest_hex(digest))
}

pub(crate) fn copy_and_hash_bounded(
    source: &mut impl Read,
    output: &mut impl Write,
    expected: u64,
    buffer: &mut [u8],
    checkpoint: &mut impl FnMut() -> Result<(), ArtifactError>,
) -> Result<String, ArtifactError> {
    stream_and_hash_bounded(
        source,
        Some(output),
        expected,
        buffer,
        checkpoint,
        ArtifactError::Archive,
        ArtifactError::Installation,
    )
}

/// Hash exactly `expected` bytes from the reader's current position through EOF.
///
/// The caller supplies a reusable, nonempty read buffer and a cancellation
/// checkpoint invoked before every read. The lowercase SHA-256 covers the exact
/// bytes read, without text decoding or newline normalization. This function
/// retries interrupted reads after checking cancellation again; it neither
/// rewinds nor locks the reader.
///
/// # Errors
/// Rejects an empty buffer, unavailable native hashing, read or hash failures,
/// short or long streams, byte-count overflow, and checkpoint errors.
pub fn hash_reader(
    reader: &mut impl Read,
    expected: u64,
    buffer: &mut [u8],
    checkpoint: &mut impl FnMut() -> Result<(), ArtifactError>,
) -> Result<String, ArtifactError> {
    stream_and_hash_bounded(
        reader,
        None::<&mut std::io::Sink>,
        expected,
        buffer,
        checkpoint,
        ArtifactError::Identity,
        ArtifactError::Identity,
    )
}

struct Entry {
    index: usize,
    name: String,
    bytes: u64,
}
struct Installation {
    executable: PathBuf,
    files: BTreeMap<String, Entry>,
    directories: BTreeSet<String>,
}

fn inspect_archive<R: Read + Seek>(
    archive: &mut ZipArchive<R>,
    root: &Path,
    checkpoint: &mut impl FnMut() -> Result<(), ArtifactError>,
) -> Result<Installation, ArtifactError> {
    if archive.is_empty() || archive.len() > MAX_ENTRIES {
        return Err(ArtifactError::Archive);
    }
    let mut names = BTreeSet::new();
    let mut expected_files = BTreeMap::new();
    let mut expected_directories = BTreeSet::new();
    let mut total = 0_u64;
    let mut server = None;
    for index in 0..archive.len() {
        checkpoint()?;
        let entry = archive
            .by_index(index)
            .map_err(|_| ArtifactError::Archive)?;
        let raw_name = entry.name().map_err(|_| ArtifactError::Archive)?;
        let name = archive_name(&raw_name, entry.is_dir())?;
        if !names.insert(name.to_ascii_lowercase()) {
            return Err(ArtifactError::Archive);
        }
        if let Some(mode) = entry.unix_mode() {
            let kind = mode & 0o170_000;
            if kind != 0 && kind != if entry.is_dir() { 0o040_000 } else { 0o100_000 } {
                return Err(ArtifactError::Archive);
            }
        }
        if entry.is_dir() {
            expected_directories.insert(name.to_ascii_lowercase());
        } else {
            total = total
                .checked_add(entry.size())
                .ok_or(ArtifactError::Archive)?;
            if total > MAX_EXPANDED_BYTES {
                return Err(ArtifactError::Archive);
            }
            if name
                .rsplit('/')
                .next()
                .is_some_and(|leaf| leaf.eq_ignore_ascii_case("llama-server.exe"))
            {
                if server.is_some() {
                    return Err(ArtifactError::Archive);
                }
                server = Some(root.join(name));
            }
            expected_files.insert(
                name.to_ascii_lowercase(),
                Entry {
                    index,
                    name: name.to_owned(),
                    bytes: entry.size(),
                },
            );
        }
        let mut parent = Path::new(name).parent();
        while let Some(path) = parent {
            if !path.as_os_str().is_empty() {
                let mut directory = path.to_string_lossy().replace('\\', "/");
                directory.make_ascii_lowercase();
                expected_directories.insert(directory);
            }
            parent = path.parent();
        }
    }
    if expected_files
        .keys()
        .any(|file| expected_directories.contains(file))
    {
        return Err(ArtifactError::Archive);
    }
    let server = server.ok_or(ArtifactError::Archive)?;
    Ok(Installation {
        executable: server,
        files: expected_files,
        directories: expected_directories,
    })
}

fn verify_installation<R: Read + Seek>(
    archive: &mut ZipArchive<R>,
    root: &Path,
    handles: &mut Vec<File>,
    held_ancestors: &mut BTreeSet<PathBuf>,
    buffer: &mut [u8],
    checkpoint: &mut impl FnMut() -> Result<(), ArtifactError>,
) -> Result<(PathBuf, Vec<VerifiedRuntimeFile>), ArtifactError> {
    let installation = inspect_archive(archive, root, checkpoint)?;
    verify_installed_files(
        archive,
        root,
        installation,
        handles,
        held_ancestors,
        buffer,
        checkpoint,
    )
}

fn verify_installed_files<R: Read + Seek>(
    archive: &mut ZipArchive<R>,
    root: &Path,
    installation: Installation,
    handles: &mut Vec<File>,
    held_ancestors: &mut BTreeSet<PathBuf>,
    buffer: &mut [u8],
    checkpoint: &mut impl FnMut() -> Result<(), ArtifactError>,
) -> Result<(PathBuf, Vec<VerifiedRuntimeFile>), ArtifactError> {
    let Installation {
        executable: server,
        files: expected_files,
        directories: expected_directories,
    } = installation;
    let mut installed_files = BTreeSet::new();
    let mut installed_directories = BTreeSet::new();
    enumerate_installation(
        root,
        root,
        &mut installed_files,
        &mut installed_directories,
        checkpoint,
    )?;
    // Both collections iterate sorted, unique names; compare them directly
    // instead of cloning every expected path into a second tree.
    if !installed_files.iter().eq(expected_files.keys())
        || installed_directories != expected_directories
    {
        return Err(ArtifactError::Installation);
    }
    let mut files = Vec::with_capacity(expected_files.len());
    for entry in expected_files.values() {
        checkpoint()?;
        let path = root.join(&entry.name);
        hold_ancestors(&path, handles, held_ancestors, checkpoint)?;
        let mut installed = open_locked_file_with_access(&path, true)?;
        let mut source = archive
            .by_index(entry.index)
            .map_err(|_| ArtifactError::Archive)?;
        let expected_hash = hash_reader(&mut source, entry.bytes, buffer, checkpoint)
            .map_err(|error| classify_hash_error(error, ArtifactError::Archive))?;
        check_locked(
            &mut installed,
            entry.bytes,
            &expected_hash,
            buffer,
            checkpoint,
        )
        .map_err(|error| classify_hash_error(error, ArtifactError::Installation))?;
        if path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("exe") || ext.eq_ignore_ascii_case("dll"))
        {
            check_arm64(&mut installed)?;
        }
        files.push(VerifiedRuntimeFile {
            path,
            bytes: entry.bytes,
            sha256: expected_hash,
        });
        handles.push(installed);
    }
    Ok((server, files))
}

fn archive_name(raw: &str, directory: bool) -> Result<&str, ArtifactError> {
    let name = if directory {
        raw.strip_suffix('/').ok_or(ArtifactError::Archive)?
    } else {
        raw
    };
    if name.is_empty()
        || name.len() > 240
        || name.starts_with('/')
        || name.contains('\\')
        || !name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._ /-".contains(&b))
    {
        return Err(ArtifactError::Archive);
    }
    for part in name.split('/') {
        let stem = part
            .split('.')
            .next()
            .unwrap_or_default()
            .trim_end_matches(' ');
        if part.is_empty()
            || part == "."
            || part == ".."
            || part.ends_with('.')
            || part.ends_with(' ')
            || ["CON", "PRN", "AUX", "NUL"]
                .iter()
                .any(|reserved| stem.eq_ignore_ascii_case(reserved))
            || (stem.len() == 4
                && (stem[..3].eq_ignore_ascii_case("COM") || stem[..3].eq_ignore_ascii_case("LPT"))
                && stem.as_bytes()[3].is_ascii_digit())
        {
            return Err(ArtifactError::Archive);
        }
    }
    Ok(name)
}

fn enumerate_installation(
    root: &Path,
    directory: &Path,
    files: &mut BTreeSet<String>,
    directories: &mut BTreeSet<String>,
    checkpoint: &mut impl FnMut() -> Result<(), ArtifactError>,
) -> Result<(), ArtifactError> {
    checkpoint()?;
    validate_external_path(directory).map_err(|_| ArtifactError::Path)?;
    for entry in fs::read_dir(directory).map_err(|_| ArtifactError::Installation)? {
        checkpoint()?;
        let path = entry.map_err(|_| ArtifactError::Installation)?.path();
        let metadata = fs::symlink_metadata(&path).map_err(|_| ArtifactError::Installation)?;
        if is_redirect(&metadata) {
            return Err(ArtifactError::Path);
        }
        let mut name = path
            .strip_prefix(root)
            .map_err(|_| ArtifactError::Path)?
            .to_str()
            .ok_or(ArtifactError::Path)?
            .replace('\\', "/");
        if files.len() + directories.len() >= MAX_ENTRIES * 2 {
            return Err(ArtifactError::Installation);
        }
        if metadata.is_dir() {
            name.make_ascii_lowercase();
            if !directories.insert(name) {
                return Err(ArtifactError::Installation);
            }
            enumerate_installation(root, &path, files, directories, checkpoint)?;
        } else if metadata.is_file() {
            name.make_ascii_lowercase();
            if !files.insert(name) {
                return Err(ArtifactError::Installation);
            }
        } else {
            return Err(ArtifactError::Installation);
        }
    }
    Ok(())
}

fn check_arm64(file: &mut (impl Read + Seek)) -> Result<(), ArtifactError> {
    let length = file
        .seek(SeekFrom::End(0))
        .map_err(|_| ArtifactError::Architecture)?;
    file.seek(SeekFrom::Start(0))
        .map_err(|_| ArtifactError::Architecture)?;
    let mut dos = [0; 64];
    file.read_exact(&mut dos)
        .map_err(|_| ArtifactError::Architecture)?;
    let [b'M', b'Z', .., o0, o1, o2, o3] = dos else {
        return Err(ArtifactError::Architecture);
    };
    let offset = u64::from(u32::from_le_bytes([o0, o1, o2, o3]));
    if offset < 64 || length < 24 || offset > length - 24 {
        return Err(ArtifactError::Architecture);
    }
    file.seek(SeekFrom::Start(offset))
        .map_err(|_| ArtifactError::Architecture)?;
    let mut pe = [0; 6];
    file.read_exact(&mut pe)
        .map_err(|_| ArtifactError::Architecture)?;
    if pe != [b'P', b'E', 0, 0, 0x64, 0xaa] {
        return Err(ArtifactError::Architecture);
    }
    file.seek(SeekFrom::Start(0))
        .map_err(|_| ArtifactError::Architecture)?;
    Ok(())
}

/// Validate path spelling and existing ancestors without following redirects.
/// Native launch code additionally verifies the volume is a recognized local disk.
///
/// # Errors
/// Rejects nonabsolute, namespace, ADS, unresolved-variable, Git and reparse paths.
pub fn validate_external_path(path: &Path) -> Result<(), ArtifactError> {
    let spelling = path.to_str().ok_or(ArtifactError::Path)?;
    if !path.is_absolute()
        || spelling.contains('%')
        || spelling.contains('\0')
        || path
            .components()
            .any(|part| matches!(part, Component::ParentDir | Component::CurDir))
    {
        return Err(ArtifactError::Path);
    }
    #[cfg(windows)]
    {
        use std::path::Prefix;
        let Some(Component::Prefix(prefix)) = path.components().next() else {
            return Err(ArtifactError::Path);
        };
        if !matches!(prefix.kind(), Prefix::Disk(_))
            || spelling.get(2..).is_none_or(|rest| rest.contains(':'))
            || path.parent().is_none()
        {
            return Err(ArtifactError::Path);
        }
    }
    for ancestor in path.ancestors() {
        match fs::symlink_metadata(ancestor) {
            Ok(metadata) => {
                if is_redirect(&metadata) {
                    return Err(ArtifactError::Path);
                }
                if metadata.is_dir() {
                    match fs::symlink_metadata(ancestor.join(".git")) {
                        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                        _ => return Err(ArtifactError::Path),
                    }
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(_) => return Err(ArtifactError::Path),
        }
    }
    Ok(())
}

/// Whether metadata identifies a symbolic link or Windows reparse point.
#[must_use]
pub fn is_redirect(metadata: &fs::Metadata) -> bool {
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        metadata.file_attributes() & 0x400 != 0
    }
    #[cfg(not(windows))]
    {
        metadata.file_type().is_symlink()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn names_reject_windows_aliases_and_redirects() {
        for name in [
            "../llama-server.exe",
            "C:/llama-server.exe",
            "/llama-server.exe",
            "a\\b",
            "a//b",
            "NUL.dll",
            "NUL .txt",
            "con .dll",
            "COM1 .dll",
            "con",
            "COM1.exe",
            "x./file",
            "x /file",
            "file:stream",
            "./file",
            "a/../file",
        ] {
            assert_eq!(archive_name(name, false), Err(ArtifactError::Archive));
        }
        assert_eq!(archive_name("a/b.dll", false), Ok("a/b.dll"));
    }
    #[test]
    fn bounded_hash_rejects_short_and_long_streams() {
        let mut buffer = vec![0; HASH_BUFFER_BYTES];
        assert!(hash_reader(&mut Cursor::new(b"abc"), 2, &mut buffer, &mut || Ok(())).is_err());
        assert!(hash_reader(&mut Cursor::new(b"abc"), 4, &mut buffer, &mut || Ok(())).is_err());
        assert_eq!(
            hash_reader(&mut Cursor::new(b"abc"), 3, &mut buffer, &mut || Ok(()))
                .ok()
                .as_deref(),
            Some("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        );
    }
    #[test]
    fn bounded_hash_validates_buffer_and_cancellation_before_reading() {
        let mut reader = Cursor::new(b"abc");
        assert_eq!(
            hash_reader(&mut reader, 3, &mut [], &mut || Ok(())),
            Err(ArtifactError::Configuration)
        );
        assert_eq!(reader.position(), 0);

        let mut buffer = [0; 1];
        let mut reads = 0;
        assert_eq!(
            hash_reader(&mut reader, 3, &mut buffer, &mut || {
                reads += 1;
                if reads == 2 {
                    Err(ArtifactError::Cancelled)
                } else {
                    Ok(())
                }
            }),
            Err(ArtifactError::Cancelled)
        );
        assert_eq!(reader.position(), 1);
        assert_eq!(
            hash_reader(&mut Cursor::new(b""), 0, &mut buffer, &mut || Ok(())),
            Ok("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855".into())
        );
    }
    #[test]
    fn verification_buffer_amortizes_reads_and_preserves_the_complete_tail()
    -> Result<(), Box<dyn std::error::Error>> {
        struct RecordedReader {
            inner: Cursor<Vec<u8>>,
            requests: Vec<usize>,
        }
        impl Read for RecordedReader {
            fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
                self.requests.push(buffer.len());
                self.inner.read(buffer)
            }
        }
        let payload: Vec<_> = (0..=255_u8)
            .cycle()
            .take(2 * HASH_BUFFER_BYTES + 137)
            .collect();
        let expected = digest_hex(Sha256::digest(&payload).into());
        let length = u64::try_from(payload.len())?;
        let mut reader = RecordedReader {
            inner: Cursor::new(payload),
            requests: Vec::new(),
        };
        assert_eq!(
            hash_reader(
                &mut reader,
                length,
                &mut verification_buffer(length),
                &mut || Ok(()),
            )?,
            expected
        );
        // Two complete chunks, the partial final chunk, and the mandatory EOF
        // read. A larger file never causes artifact-sized scratch allocation.
        assert_eq!(reader.requests, vec![HASH_BUFFER_BYTES; 4]);
        assert_eq!(verification_buffer(u64::MAX).len(), HASH_BUFFER_BYTES);
        assert_eq!(verification_buffer(0).len(), 1);
        assert_eq!(verification_buffer(3).len(), 3);

        reader.inner.get_mut()[2 * HASH_BUFFER_BYTES + 136] ^= 1;
        reader.inner.set_position(0);
        assert_ne!(
            hash_reader(
                &mut reader,
                length,
                &mut verification_buffer(length),
                &mut || Ok(()),
            )?,
            expected
        );
        let _ = reader.inner.get_mut().pop();
        reader.inner.set_position(0);
        assert_eq!(
            hash_reader(
                &mut reader,
                length,
                &mut verification_buffer(length),
                &mut || Ok(()),
            ),
            Err(ArtifactError::Identity)
        );
        Ok(())
    }
    #[cfg(windows)]
    #[test]
    fn sequential_verification_retains_file_guards_and_rejects_tail_changes()
    -> Result<(), Box<dyn std::error::Error>> {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "SwiftBonsai-sequential-verification-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir(&root)?;
        let path = root.join("model.bin");
        let mut payload = vec![0xa5; HASH_BUFFER_BYTES + 137];
        let length = u64::try_from(payload.len())?;
        let digest = digest_hex(Sha256::digest(&payload).into());
        fs::write(&path, &payload)?;
        let guard = VerifiedFile::verify(&path, length, &digest)?;
        // Ordinary locked readers remain seekable while verification retains
        // deny-write/delete sharing on the same file and ancestor guards.
        let mut reader = open_locked_file(&path)?;
        reader.seek(SeekFrom::Start(length - 1))?;
        let mut tail = [0];
        reader.read_exact(&mut tail)?;
        assert_eq!(tail, [0xa5]);
        drop(reader);
        assert!(OpenOptions::new().write(true).open(&path).is_err());
        assert!(fs::remove_file(&path).is_err());
        assert!(fs::rename(&path, root.join("moved.bin")).is_err());
        drop(guard);

        payload[HASH_BUFFER_BYTES + 136] ^= 1;
        fs::write(&path, &payload)?;
        assert_eq!(
            VerifiedFile::verify(&path, length, &digest).err(),
            Some(ArtifactError::Identity)
        );
        let _ = payload.pop();
        fs::write(&path, &payload)?;
        assert_eq!(
            VerifiedFile::verify(&path, length, &digest).err(),
            Some(ArtifactError::Identity)
        );
        fs::remove_file(path)?;
        fs::remove_dir(root)?;
        Ok(())
    }
    #[cfg(windows)]
    #[test]
    fn cng_hash_matches_portable_sha256_across_binary_read_boundaries() -> Result<(), ArtifactError>
    {
        let payload: Vec<_> = (0..=255_u8)
            .cycle()
            .take(2 * HASH_BUFFER_BYTES + 137)
            .collect();
        let expected = digest_hex(Sha256::digest(&payload).into());
        for length in [63, 64, 65, HASH_BUFFER_BYTES] {
            let mut buffer = vec![0; length];
            assert_eq!(
                hash_reader(
                    &mut Cursor::new(&payload),
                    u64::try_from(payload.len()).map_err(|_| ArtifactError::Identity)?,
                    &mut buffer,
                    &mut || Ok(()),
                )?,
                expected
            );
        }
        Ok(())
    }
    #[test]
    fn bounded_hash_never_returns_a_digest_after_a_read_failure() {
        struct BrokenReader(bool);
        impl Read for BrokenReader {
            fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
                if self.0 {
                    return Err(std::io::Error::other("synthetic read failure"));
                }
                self.0 = true;
                buffer[0] = b'a';
                Ok(1)
            }
        }
        assert_eq!(
            hash_reader(&mut BrokenReader(false), 1, &mut [0; 64], &mut || Ok(())),
            Err(ArtifactError::Identity)
        );
    }
    #[test]
    fn bounded_hash_retries_interruptions_without_skipping_cancellation() {
        struct InterruptedReader {
            inner: Cursor<&'static [u8]>,
            interrupted: bool,
        }
        impl Read for InterruptedReader {
            fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
                self.interrupted = !self.interrupted;
                if self.interrupted {
                    Err(std::io::ErrorKind::Interrupted.into())
                } else {
                    self.inner.read(buffer)
                }
            }
        }
        let mut reader = InterruptedReader {
            inner: Cursor::new(b"abc"),
            interrupted: false,
        };
        assert_eq!(
            hash_reader(&mut reader, 3, &mut [0; 1], &mut || Ok(())),
            Ok("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad".into())
        );

        let mut reader = InterruptedReader {
            inner: Cursor::new(b"abc"),
            interrupted: false,
        };
        let mut checkpoints = 0;
        assert_eq!(
            hash_reader(&mut reader, 3, &mut [0; 1], &mut || {
                checkpoints += 1;
                if checkpoints == 2 {
                    Err(ArtifactError::Cancelled)
                } else {
                    Ok(())
                }
            }),
            Err(ArtifactError::Cancelled)
        );
        assert_eq!(reader.inner.position(), 0);
    }
    #[test]
    fn zipped_installation_verifies_deflated_payload_and_rejects_drift()
    -> Result<(), Box<dyn std::error::Error>> {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "SwiftBonsai-zip-verification-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir(&root)?;
        let executable = root.join("llama-server.exe");
        let mut image = vec![0; 128];
        image[..2].copy_from_slice(b"MZ");
        image[60..64].copy_from_slice(&64_u32.to_le_bytes());
        image[64..70].copy_from_slice(&[b'P', b'E', 0, 0, 0x64, 0xaa]);
        fs::write(&executable, &image)?;
        let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
        writer.start_file(
            "llama-server.exe",
            zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Deflated),
        )?;
        writer.write_all(&image)?;
        let mut archive = ZipArchive::new(writer.finish()?)?;
        let mut handles = Vec::new();
        let mut buffer = vec![0; HASH_BUFFER_BYTES];
        let mut checkpoint = || Ok(());
        assert_eq!(
            verify_installation(
                &mut archive,
                &root,
                &mut handles,
                &mut BTreeSet::new(),
                &mut buffer,
                &mut checkpoint,
            )?
            .0,
            executable
        );
        drop(handles);
        image[127] = 1;
        fs::write(&executable, &image)?;
        let mut handles = Vec::new();
        assert_eq!(
            verify_installation(
                &mut archive,
                &root,
                &mut handles,
                &mut BTreeSet::new(),
                &mut buffer,
                &mut checkpoint,
            ),
            Err(ArtifactError::Installation)
        );
        drop(handles);
        fs::remove_dir_all(root)?;
        Ok(())
    }
    #[test]
    fn native_pe_requires_bounded_arm64_header() {
        let mut image = vec![0; 128];
        image[..2].copy_from_slice(b"MZ");
        image[60..64].copy_from_slice(&64_u32.to_le_bytes());
        image[64..70].copy_from_slice(&[b'P', b'E', 0, 0, 0x64, 0xaa]);
        let mut cursor = Cursor::new(&image);
        assert!(check_arm64(&mut cursor).is_ok());
        assert_eq!(cursor.position(), 0);
        image[68] = 0x4c;
        assert_eq!(
            check_arm64(&mut Cursor::new(&image)),
            Err(ArtifactError::Architecture)
        );
        image[60..64].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(check_arm64(&mut Cursor::new(&image)).is_err());

        let mut overlapping = vec![0; 128];
        overlapping[..2].copy_from_slice(b"MZ");
        overlapping[8..14].copy_from_slice(&[b'P', b'E', 0, 0, 0x64, 0xaa]);
        overlapping[60..64].copy_from_slice(&8_u32.to_le_bytes());
        assert_eq!(
            check_arm64(&mut Cursor::new(&overlapping)),
            Err(ArtifactError::Architecture)
        );
    }

    #[test]
    fn copy_and_hash_bounded_streams_writes_and_rejects_bad_lengths() {
        let mut buffer = [0_u8; 4];
        let mut out = Vec::new();
        let digest = copy_and_hash_bounded(
            &mut Cursor::new(b"abcdef"),
            &mut out,
            6,
            &mut buffer,
            &mut || Ok(()),
        );
        assert_eq!(out, b"abcdef");
        assert_eq!(
            digest,
            Ok("bef57ec7f53a6d40beb640a780a639c83bc29ac8a9816f1fc6c5c6dcd93c4721".into())
        );
        out.clear();
        assert_eq!(
            copy_and_hash_bounded(
                &mut Cursor::new(b"abcdef"),
                &mut out,
                7,
                &mut buffer,
                &mut || Ok(()),
            ),
            Err(ArtifactError::Archive)
        );
        out.clear();
        assert_eq!(
            copy_and_hash_bounded(
                &mut Cursor::new(b"abcdef"),
                &mut out,
                5,
                &mut buffer,
                &mut || Ok(()),
            ),
            Err(ArtifactError::Archive)
        );

        struct FailingWriter;
        impl Write for FailingWriter {
            fn write(&mut self, _buf: &[u8]) -> std::io::Result<usize> {
                Err(std::io::Error::other("disk full"))
            }

            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        assert_eq!(
            copy_and_hash_bounded(
                &mut Cursor::new(b"abc"),
                &mut FailingWriter,
                3,
                &mut buffer,
                &mut || Ok(()),
            ),
            Err(ArtifactError::Installation)
        );
    }

    #[cfg(windows)]
    #[test]
    fn hold_ancestors_opens_top_down_and_validates_regular_read_paths()
    -> Result<(), Box<dyn std::error::Error>> {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "SwiftBonsai-ancestors-check-{}-{nonce}",
            std::process::id()
        ));
        let nested = root.join("nested");
        fs::create_dir_all(&nested)?;
        let first = nested.join("first.bin");
        let second = nested.join("second.bin");
        fs::write(&first, b"1")?;
        fs::write(&second, b"2")?;
        assert!(validate_read_path(&first).is_ok());
        assert_eq!(validate_read_path(&nested), Err(ArtifactError::Path));
        assert_eq!(
            validate_read_path(&nested.join("missing.bin")),
            Err(ArtifactError::Path)
        );

        let mut handles = Vec::new();
        let mut held = BTreeSet::new();
        hold_ancestors(&first, &mut handles, &mut held, &mut || Ok(()))?;
        let held_count = handles.len();
        assert!(held.contains(&nested));
        assert!(held.contains(&root));
        hold_ancestors(&second, &mut handles, &mut held, &mut || Ok(()))?;
        assert_eq!(handles.len(), held_count);

        drop(handles);
        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[cfg(windows)]
    #[test]
    fn verified_file_mapped_path_accepts_rejects_tamper_and_cancels()
    -> Result<(), Box<dyn std::error::Error>> {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "SwiftBonsai-mapped-verify-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&root)?;
        let file_path = root.join("large-artifact.gguf");
        let len = usize::try_from(bonsai_windows_credentials::MIN_MAPPED_HASH_BYTES)? + 4_096;
        let mut payload = vec![0x5a_u8; len];
        payload[0] = 0x11;
        payload[len - 1] = 0x99;
        fs::write(&file_path, &payload)?;
        let expected_hash = hex::encode(Sha256::digest(&payload));

        let mut total_checks = 0_usize;
        let guard =
            VerifiedFile::verify_with(&file_path, u64::try_from(len)?, &expected_hash, || {
                total_checks += 1;
                Ok(())
            })?;
        assert!(
            guard
                .file()
                .is_some_and(|file| file.metadata().is_ok_and(|metadata| metadata.is_file()))
        );
        drop(guard);
        assert!(total_checks >= 5);

        let mut checks = 0_usize;
        assert_eq!(
            VerifiedFile::verify_with(&file_path, u64::try_from(len)?, &expected_hash, || {
                checks += 1;
                if checks >= total_checks {
                    Err(ArtifactError::Cancelled)
                } else {
                    Ok(())
                }
            })
            .err(),
            Some(ArtifactError::Cancelled)
        );

        payload[len - 1] ^= 0x01;
        fs::write(&file_path, &payload)?;
        assert_eq!(
            VerifiedFile::verify(&file_path, u64::try_from(len)?, &expected_hash).err(),
            Some(ArtifactError::Identity)
        );

        fs::remove_dir_all(root)?;
        Ok(())
    }
}
