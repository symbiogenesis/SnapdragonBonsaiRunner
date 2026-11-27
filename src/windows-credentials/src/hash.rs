//! Streaming SHA-256 through Windows' CPU-dispatched CNG implementation.

use std::{
    ffi::c_void,
    fs::File,
    io,
    os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle},
    ptr::{self, NonNull},
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
};

use windows_sys::Win32::Security::Cryptography::{BCRYPT_SHA256_ALG_HANDLE, BCryptHash};
use windows_sys::Win32::System::Memory::{
    CreateFileMappingW, FILE_MAP_READ, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile, PAGE_READONLY,
    PrefetchVirtualMemory, UnmapViewOfFile, WIN32_MEMORY_RANGE_ENTRY,
};
use windows_sys::Win32::System::Threading::GetCurrentProcess;

/// Minimum regular-file length (`4 MiB`) that uses read-only section mapping,
/// prefetching, and chunked CNG updates instead of buffered `Read` hashing.
pub const MIN_MAPPED_HASH_BYTES: u64 = 4 * 1024 * 1024;
/// Default chunk size (`16 MiB`) for parallel mapped CNG SHA-256 verification.
pub const PARALLEL_HASH_DEFAULT_CHUNK_BYTES: usize = 16 * 1024 * 1024;
/// Maximum OS worker threads used by parallel mapped CNG SHA-256 verification.
pub const MAX_PARALLEL_HASH_WORKERS: usize = 16;
const MAPPED_HASH_CHUNK_BYTES: usize = 4 * 1024 * 1024;
const MAPPED_PREFETCH_WINDOW_BYTES: usize = 16 * 1024 * 1024;

struct MappedView(NonNull<u8>);

impl Drop for MappedView {
    fn drop(&mut self) {
        let address = MEMORY_MAPPED_VIEW_ADDRESS {
            Value: self.0.as_ptr().cast(),
        };
        // SAFETY: `self.0` was returned by `MapViewOfFile` and is unmapped once on Drop.
        unsafe { UnmapViewOfFile(address) };
    }
}

struct ReadOnlyMapping {
    // Field drop order matters: unmap view before closing the section handle.
    view: MappedView,
    _mapping: OwnedHandle,
    len: usize,
}

// SAFETY: `ReadOnlyMapping` owns a process-wide kernel section handle and a
// `PAGE_READONLY` / `FILE_MAP_READ` mapping view with no thread affinity, so
// transferring ownership across threads introduces no aliasing or race.
unsafe impl Send for ReadOnlyMapping {}

// SAFETY: `&ReadOnlyMapping` exposes only immutable `&[u8]` slices over
// `0..self.len` while the view remains mapped until `Drop`.
unsafe impl Sync for ReadOnlyMapping {}

impl ReadOnlyMapping {
    fn open(file: &File, expected_bytes: u64) -> Option<Self> {
        if expected_bytes < MIN_MAPPED_HASH_BYTES {
            return None;
        }
        let Ok(metadata) = file.metadata() else {
            return None;
        };
        if !metadata.is_file() || metadata.len() != expected_bytes {
            return None;
        }
        let len = usize::try_from(expected_bytes).ok()?;
        // SAFETY: `file` is a valid regular file handle; zero max size maps the whole file read-only.
        let handle = unsafe {
            CreateFileMappingW(
                file.as_raw_handle(),
                ptr::null(),
                PAGE_READONLY,
                0,
                0,
                ptr::null(),
            )
        };
        if handle.is_null() {
            return None;
        }
        // SAFETY: `handle` is non-null and owned by `CreateFileMappingW`.
        let mapping = unsafe { OwnedHandle::from_raw_handle(handle) };
        // SAFETY: `mapping` is a valid read-only section covering `len` bytes at offset zero.
        let view_addr = unsafe { MapViewOfFile(mapping.as_raw_handle(), FILE_MAP_READ, 0, 0, len) };
        let base = NonNull::new(view_addr.Value.cast::<u8>())?;
        Some(Self {
            view: MappedView(base),
            _mapping: mapping,
            len,
        })
    }

    const fn as_bytes(&self) -> &[u8] {
        // SAFETY: `self.view.0` points to `self.len` read-only mapped bytes valid until `Drop`.
        unsafe { std::slice::from_raw_parts(self.view.0.as_ptr(), self.len) }
    }
}

/// Safe read-only section mapping over a read-locked regular file.
///
/// Owns the Windows kernel section handle and read-only view until `Drop`,
/// exposing only an immutable byte slice over `0..expected_bytes`.
pub struct MappedLockedFile(ReadOnlyMapping);

impl MappedLockedFile {
    /// Attempt to map `file` read-only when it is a regular file of at least
    /// [`MIN_MAPPED_HASH_BYTES`] matching `expected_bytes`.
    ///
    /// Returns `None` when the file is smaller than [`MIN_MAPPED_HASH_BYTES`],
    /// its length mismatches or overflows `usize`, or section mapping fails.
    #[must_use]
    pub fn try_open(file: &File, expected_bytes: u64) -> Option<Self> {
        ReadOnlyMapping::open(file, expected_bytes).map(Self)
    }

    /// Immutable view of the mapped file bytes.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8] {
        self.0.as_bytes()
    }
}

fn prefetch_window(window: &[u8]) {
    if window.is_empty() {
        return;
    }
    let mut entry = WIN32_MEMORY_RANGE_ENTRY {
        VirtualAddress: window.as_ptr().cast_mut().cast::<c_void>(),
        NumberOfBytes: window.len(),
    };
    // SAFETY: `GetCurrentProcess` returns the pseudo-handle for the current process.
    let process = unsafe { GetCurrentProcess() };
    // SAFETY: `process` is the calling process pseudo-handle and `entry` points to one live range.
    let _ = unsafe { PrefetchVirtualMemory(process, 1, &raw mut entry, 0) };
}

fn hash_mapped_windows<E>(
    bytes: &[u8],
    checkpoint: &mut impl FnMut() -> Result<(), E>,
    mut map_io_error: impl FnMut(io::Error) -> E,
) -> Result<[u8; 32], E> {
    let mut hasher = Sha256::new().map_err(&mut map_io_error)?;
    let mut windows = bytes.chunks(MAPPED_PREFETCH_WINDOW_BYTES).peekable();
    if let Some(&first) = windows.peek() {
        checkpoint()?;
        prefetch_window(first);
    }
    while let Some(window) = windows.next() {
        if let Some(&next_window) = windows.peek() {
            checkpoint()?;
            prefetch_window(next_window);
        }
        for chunk in window.chunks(MAPPED_HASH_CHUNK_BYTES) {
            checkpoint()?;
            hasher.update(chunk).map_err(&mut map_io_error)?;
        }
    }
    hasher.finish().map_err(map_io_error)
}

/// Hash a read-locked regular file of `expected_bytes` via a read-only section
/// view, `PrefetchVirtualMemory`, and `4 MiB` chunked CNG updates, invoking
/// `checkpoint` before mapping and before each prefetch/update step.
///
/// Returns `Ok(None)` when `expected_bytes < MIN_MAPPED_HASH_BYTES`, the length
/// mismatches or overflows `usize`, or section/view creation is unsupported so
/// callers can fall back to buffered `Read` hashing without losing cancellation.
///
/// # Errors
/// Returns `checkpoint` errors immediately after releasing the mapped view or
/// CNG hashing failures converted by `map_io_error`.
pub fn try_hash_locked_file_mapped<E>(
    file: &File,
    expected_bytes: u64,
    checkpoint: &mut impl FnMut() -> Result<(), E>,
    mut map_io_error: impl FnMut(io::Error) -> E,
) -> Result<Option<[u8; 32]>, E> {
    checkpoint()?;
    let Some(mapping) = ReadOnlyMapping::open(file, expected_bytes) else {
        return Ok(None);
    };
    hash_mapped_windows(mapping.as_bytes(), checkpoint, &mut map_io_error).map(Some)
}

enum WorkerError {
    Cancelled,
    PeerAborted,
    Io(io::Error),
}

fn validate_chunk_bytes(chunk_bytes: usize) -> io::Result<u64> {
    let chunk_bytes_u64 = u64::try_from(chunk_bytes).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "Chunk size exceeds supported u64 range.",
        )
    })?;
    if chunk_bytes_u64 < MIN_MAPPED_HASH_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Chunk size must be at least MIN_MAPPED_HASH_BYTES.",
        ));
    }
    Ok(chunk_bytes_u64)
}

fn cancelled_error() -> io::Error {
    io::Error::new(io::ErrorKind::Interrupted, "Chunked hashing was cancelled.")
}

pub(crate) fn constant_time_diff<const N: usize>(lhs: &[u8; N], rhs: &[u8; N]) -> u8 {
    let mut diff = 0_u8;
    for (&a, &b) in lhs.iter().zip(rhs.iter()) {
        diff |= std::hint::black_box(a ^ b);
    }
    diff
}

fn hash_mapped_chunk(
    chunk: &[u8],
    should_cancel: &(impl Fn() -> bool + Sync),
    peer_error: &AtomicBool,
    already_prefetched: bool,
) -> Result<[u8; 32], WorkerError> {
    let mut check_abort = || {
        if should_cancel() {
            Err(WorkerError::Cancelled)
        } else if peer_error.load(Ordering::Acquire) {
            Err(WorkerError::PeerAborted)
        } else {
            Ok(())
        }
    };
    if chunk.len() <= MAPPED_PREFETCH_WINDOW_BYTES {
        check_abort()?;
        if !already_prefetched {
            prefetch_window(chunk);
            check_abort()?;
        }
        return Sha256::digest_bytes(chunk).map_err(WorkerError::Io);
    }
    hash_mapped_windows(chunk, &mut check_abort, WorkerError::Io)
}

/// Compute fixed-chunk SHA-256 digests for a read-locked regular file via a
/// single read-only section mapping and bounded parallel CNG workers, polling
/// `should_cancel` before every prefetch and hash step.
///
/// Returns `Ok(None)` when `expected_bytes < MIN_MAPPED_HASH_BYTES`, the file
/// length mismatches or overflows `usize`, or section/view creation is
/// unsupported so callers can fall back cleanly.
///
/// # Errors
/// Rejects `chunk_bytes < MIN_MAPPED_HASH_BYTES`, returns
/// `io::ErrorKind::Interrupted` when `should_cancel` triggers, or returns CNG
/// hashing failures.
pub fn compute_mapped_chunk_digests_with(
    file: &File,
    expected_bytes: u64,
    chunk_bytes: usize,
    workers: usize,
    should_cancel: &(impl Fn() -> bool + Sync),
) -> io::Result<Option<Vec<[u8; 32]>>> {
    if should_cancel() {
        return Err(cancelled_error());
    }
    let _ = validate_chunk_bytes(chunk_bytes)?;
    let Some(mapping) = ReadOnlyMapping::open(file, expected_bytes) else {
        return Ok(None);
    };
    let data = mapping.as_bytes();
    let chunk_count = data.len().div_ceil(chunk_bytes);
    let worker_count = workers.clamp(1, MAX_PARALLEL_HASH_WORKERS).min(chunk_count);
    let lookahead = worker_count;
    for prefetch_index in 0..lookahead {
        if should_cancel() {
            return Err(cancelled_error());
        }
        let start = prefetch_index
            .checked_mul(chunk_bytes)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "Chunk offset overflow."))?;
        let end = start
            .saturating_add(chunk_bytes.min(MAPPED_PREFETCH_WINDOW_BYTES))
            .min(data.len());
        if let Some(window) = data.get(start..end) {
            prefetch_window(window);
        }
    }
    let next_chunk = AtomicUsize::new(0);
    let peer_error = AtomicBool::new(false);

    let worker_task = || -> Result<Vec<(usize, [u8; 32])>, WorkerError> {
        let mut local = Vec::with_capacity(chunk_count.div_ceil(worker_count));
        loop {
            if should_cancel() {
                peer_error.store(true, Ordering::Release);
                return Err(WorkerError::Cancelled);
            }
            if peer_error.load(Ordering::Acquire) {
                return Err(WorkerError::PeerAborted);
            }
            let chunk_index = next_chunk.fetch_add(1, Ordering::Relaxed);
            if chunk_index >= chunk_count {
                break;
            }
            if let Some(ahead_index) = chunk_index
                .checked_add(lookahead)
                .filter(|&index| index < chunk_count)
            {
                if should_cancel() {
                    peer_error.store(true, Ordering::Release);
                    return Err(WorkerError::Cancelled);
                }
                if peer_error.load(Ordering::Acquire) {
                    return Err(WorkerError::PeerAborted);
                }
                let ahead_start = ahead_index.checked_mul(chunk_bytes).ok_or_else(|| {
                    peer_error.store(true, Ordering::Release);
                    WorkerError::Io(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "Chunk offset overflow.",
                    ))
                })?;
                let ahead_end = ahead_start
                    .saturating_add(chunk_bytes.min(MAPPED_PREFETCH_WINDOW_BYTES))
                    .min(data.len());
                if let Some(window) = data.get(ahead_start..ahead_end) {
                    prefetch_window(window);
                }
            }
            let start = chunk_index.checked_mul(chunk_bytes).ok_or_else(|| {
                peer_error.store(true, Ordering::Release);
                WorkerError::Io(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "Chunk offset overflow.",
                ))
            })?;
            let end = start.saturating_add(chunk_bytes).min(data.len());
            let chunk = data.get(start..end).ok_or_else(|| {
                peer_error.store(true, Ordering::Release);
                WorkerError::Io(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "Chunk bounds out of range.",
                ))
            })?;
            let digest = match hash_mapped_chunk(chunk, should_cancel, &peer_error, true) {
                Ok(digest) => digest,
                Err(error) => {
                    peer_error.store(true, Ordering::Release);
                    return Err(error);
                }
            };
            local.push((chunk_index, digest));
        }
        Ok(local)
    };

    let worker_results = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count - 1);
        for _ in 0..(worker_count - 1) {
            handles.push(scope.spawn(&worker_task));
        }
        let current = worker_task();
        let mut results = Vec::with_capacity(worker_count);
        results.push(current);
        for handle in handles {
            match handle.join() {
                Ok(result) => results.push(result),
                Err(_) => {
                    peer_error.store(true, Ordering::Release);
                    results.push(Err(WorkerError::Io(io::Error::other(
                        "Parallel hash worker thread panicked.",
                    ))));
                }
            }
        }
        results
    });

    if should_cancel() {
        return Err(cancelled_error());
    }
    let mut digests = vec![[0_u8; 32]; chunk_count];
    let mut first_io_error = None;
    let mut saw_peer_abort = false;
    for result in worker_results {
        match result {
            Ok(pairs) => {
                for (chunk_index, digest) in pairs {
                    let slot = digests
                        .get_mut(chunk_index)
                        .ok_or_else(|| io::Error::other("Computed chunk index out of range."))?;
                    *slot = digest;
                }
            }
            Err(WorkerError::Cancelled) => return Err(cancelled_error()),
            Err(WorkerError::PeerAborted) => saw_peer_abort = true,
            Err(WorkerError::Io(error)) => {
                if first_io_error.is_none() {
                    first_io_error = Some(error);
                }
            }
        }
    }
    if let Some(error) = first_io_error {
        return Err(error);
    }
    if saw_peer_abort {
        return Err(io::Error::other("Parallel hash worker aborted."));
    }
    Ok(Some(digests))
}

/// Compute fixed-chunk SHA-256 digests for a read-locked regular file via a
/// single read-only section mapping and bounded parallel CNG workers, checking
/// `cancel` before every prefetch and hash step.
///
/// Returns `Ok(None)` when `expected_bytes < MIN_MAPPED_HASH_BYTES`, the file
/// length mismatches or overflows `usize`, or section/view creation is
/// unsupported so callers can fall back cleanly.
///
/// # Errors
/// Rejects `chunk_bytes < MIN_MAPPED_HASH_BYTES`, returns
/// `io::ErrorKind::Interrupted` when `cancel` is set, or returns CNG hashing
/// failures.
pub fn compute_mapped_chunk_digests(
    file: &File,
    expected_bytes: u64,
    chunk_bytes: usize,
    workers: usize,
    cancel: &AtomicBool,
) -> io::Result<Option<Vec<[u8; 32]>>> {
    compute_mapped_chunk_digests_with(file, expected_bytes, chunk_bytes, workers, &|| {
        cancel.load(Ordering::Acquire)
    })
}

/// Verify a read-locked regular file's fixed-chunk SHA-256 manifest via a
/// single read-only section mapping and bounded parallel CNG workers, polling
/// `should_cancel` before every prefetch and hash step.
///
/// Returns `Ok(None)` when `expected_bytes < MIN_MAPPED_HASH_BYTES`, the file
/// length mismatches or overflows `usize`, or section/view creation is
/// unsupported. Returns `Ok(Some(true))` when every chunk matches in constant
/// time and `Ok(Some(false))` when any chunk differs.
///
/// # Errors
/// Rejects `chunk_bytes < MIN_MAPPED_HASH_BYTES` or a chunk-digest count
/// differing from `expected_bytes.div_ceil(chunk_bytes)`, returns
/// `io::ErrorKind::Interrupted` on cancellation, or returns CNG hashing failures.
pub fn try_verify_locked_file_chunk_digests_mapped_with(
    file: &File,
    expected_bytes: u64,
    chunk_bytes: usize,
    expected_digests: &[[u8; 32]],
    workers: usize,
    should_cancel: &(impl Fn() -> bool + Sync),
) -> io::Result<Option<bool>> {
    if should_cancel() {
        return Err(cancelled_error());
    }
    let chunk_bytes_u64 = validate_chunk_bytes(chunk_bytes)?;
    let expected_chunks =
        usize::try_from(expected_bytes.div_ceil(chunk_bytes_u64)).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "Expected chunk count exceeds usize range.",
            )
        })?;
    if expected_digests.len() != expected_chunks {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Expected chunk digest count does not match file length and chunk size.",
        ));
    }
    let Some(actual_digests) = compute_mapped_chunk_digests_with(
        file,
        expected_bytes,
        chunk_bytes,
        workers,
        should_cancel,
    )?
    else {
        return Ok(None);
    };
    if actual_digests.len() != expected_digests.len() {
        return Err(io::Error::other(
            "Computed chunk digest count differed from expected chunk count.",
        ));
    }
    let mut diff = 0_u8;
    for (actual, expected) in actual_digests.iter().zip(expected_digests.iter()) {
        diff |= constant_time_diff(actual, expected);
    }
    Ok(Some(diff == 0))
}

/// Verify a read-locked regular file's fixed-chunk SHA-256 manifest via a
/// single read-only section mapping and bounded parallel CNG workers, checking
/// `cancel` before every prefetch and hash step.
///
/// Returns `Ok(None)` when `expected_bytes < MIN_MAPPED_HASH_BYTES`, the file
/// length mismatches or overflows `usize`, or section/view creation is
/// unsupported. Returns `Ok(Some(true))` when every chunk matches in constant
/// time and `Ok(Some(false))` when any chunk differs.
///
/// # Errors
/// Rejects `chunk_bytes < MIN_MAPPED_HASH_BYTES` or a chunk-digest count
/// differing from `expected_bytes.div_ceil(chunk_bytes)`, returns
/// `io::ErrorKind::Interrupted` when `cancel` is set, or returns CNG hashing
/// failures.
pub fn try_verify_locked_file_chunk_digests_mapped(
    file: &File,
    expected_bytes: u64,
    chunk_bytes: usize,
    expected_digests: &[[u8; 32]],
    workers: usize,
    cancel: &AtomicBool,
) -> io::Result<Option<bool>> {
    try_verify_locked_file_chunk_digests_mapped_with(
        file,
        expected_bytes,
        chunk_bytes,
        expected_digests,
        workers,
        &|| cancel.load(Ordering::Acquire),
    )
}

/// A streaming SHA-256 hash using Windows' CPU-dispatched CNG implementation.
///
/// Streaming delegates to [`bonsai_native_hash::Hasher`]; one-shot slice hashing
/// uses stateless `BCryptHash`.
pub struct Sha256(bonsai_native_hash::Hasher);

impl Sha256 {
    /// Create an empty SHA-256 state without opening an algorithm provider.
    ///
    /// # Errors
    /// Returns an unavailable provider or native allocation failure.
    pub fn new() -> io::Result<Self> {
        bonsai_native_hash::Hasher::new().map(Self)
    }

    /// Append exactly the supplied bytes to this hash.
    ///
    /// # Errors
    /// Rejects an input exceeding the native length bound or a provider failure.
    pub fn update(&mut self, bytes: &[u8]) -> io::Result<()> {
        self.0.update(bytes)
    }

    /// Consume this state and return its exact 32-byte SHA-256 digest.
    ///
    /// # Errors
    /// Returns a native finalization failure. The hash is destroyed on all paths.
    pub fn finish(self) -> io::Result<[u8; 32]> {
        self.0.finish()
    }

    /// Compute the 32-byte SHA-256 digest of a slice in one call via Windows CNG.
    ///
    /// # Errors
    /// Returns a provider allocation, update or finalization failure.
    pub fn digest_bytes(bytes: &[u8]) -> io::Result<[u8; 32]> {
        if let Ok(length) = u32::try_from(bytes.len()) {
            let mut digest = [0_u8; 32];
            let input = if bytes.is_empty() {
                ptr::null()
            } else {
                bytes.as_ptr()
            };
            // SAFETY: BCRYPT_SHA256_ALG_HANDLE is the OS SHA-256 pseudo-handle,
            // input is readable for length bytes (or null when empty), and
            // digest is writable for the exact 32-byte SHA-256 output size.
            check(unsafe {
                BCryptHash(
                    BCRYPT_SHA256_ALG_HANDLE,
                    ptr::null(),
                    0,
                    input,
                    length,
                    digest.as_mut_ptr(),
                    32,
                )
            })?;
            return Ok(digest);
        }
        let mut hasher = Self::new()?;
        for chunk in bytes.chunks(0x4000_0000) {
            hasher.update(chunk)?;
        }
        hasher.finish()
    }
}

fn check(status: i32) -> io::Result<()> {
    if status >= 0 {
        Ok(())
    } else {
        Err(io::Error::other("Windows SHA-256 operation failed."))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(parts: &[&[u8]]) -> io::Result<String> {
        let mut hash = Sha256::new()?;
        for part in parts {
            hash.update(part)?;
        }
        Ok(super::super::hex(&hash.finish()?))
    }

    #[test]
    fn cng_sha256_matches_known_empty_short_and_multiblock_vectors() -> io::Result<()> {
        assert_eq!(
            digest(&[])?,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            super::super::hex(&Sha256::digest_bytes(b"")?),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            digest(&[b"", b"a", b"", b"bc", b""])?,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            super::super::hex(&Sha256::digest_bytes(b"abc")?),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            digest(&[b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"])?,
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
        assert_eq!(
            super::super::hex(&Sha256::digest_bytes(
                b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
            )?),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
        let block = [b'a'; 1000];
        let mut hash = Sha256::new()?;
        for _ in 0..1000 {
            hash.update(&block)?;
        }
        assert_eq!(
            super::super::hex(&hash.finish()?),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
        Ok(())
    }

    #[test]
    fn mapped_file_hashing_matches_cng_digest_and_honors_cancellation() -> io::Result<()> {
        use std::os::windows::fs::OpenOptionsExt as _;

        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(io::Error::other)?
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "SwiftBonsai-mapped-hash-{}-{nonce}.bin",
            std::process::id()
        ));
        let len = usize::try_from(MIN_MAPPED_HASH_BYTES).map_err(io::Error::other)? + 12_345;
        let tile: Vec<u8> = (0..65_536_usize.min(len))
            .map(|i| u8::try_from((i.wrapping_mul(31) ^ (i >> 8)) & 0xff).unwrap_or(0))
            .collect();
        let mut payload = vec![0_u8; len];
        if !tile.is_empty() {
            for chunk in payload.chunks_mut(tile.len()) {
                chunk.copy_from_slice(&tile[..chunk.len()]);
            }
        }
        std::fs::write(&path, &payload)?;

        let expected = Sha256::digest_bytes(&payload)?;
        let file = std::fs::OpenOptions::new()
            .read(true)
            .share_mode(windows_sys::Win32::Storage::FileSystem::FILE_SHARE_READ)
            .open(&path)?;

        // Below threshold returns Ok(None) for buffered fallback.
        assert_eq!(
            try_hash_locked_file_mapped(
                &file,
                MIN_MAPPED_HASH_BYTES - 1,
                &mut || Ok::<(), io::Error>(()),
                |e| e
            )?,
            None
        );

        let mut checks = 0_usize;
        let actual = try_hash_locked_file_mapped(
            &file,
            u64::try_from(len).map_err(io::Error::other)?,
            &mut || {
                checks += 1;
                Ok::<(), io::Error>(())
            },
            |e| e,
        )?;
        assert_eq!(actual, Some(expected));
        assert!(checks >= 4);

        let mut cancel_checks = 0_usize;
        let cancelled = try_hash_locked_file_mapped(
            &file,
            u64::try_from(len).map_err(io::Error::other)?,
            &mut || {
                cancel_checks += 1;
                if cancel_checks >= 3 {
                    Err(io::Error::new(io::ErrorKind::Interrupted, "cancelled"))
                } else {
                    Ok(())
                }
            },
            |e| e,
        );
        assert!(cancelled.is_err());

        drop(file);
        std::fs::remove_file(path)?;
        Ok(())
    }
}
