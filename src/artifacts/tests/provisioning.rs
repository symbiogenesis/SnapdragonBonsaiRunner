//! Focused offline runtime installation regressions with synthetic ZIPs.
#![cfg(windows)]
#![forbid(unsafe_code)]

use bonsai_artifacts::{
    ArtifactError, ChunkedFileSpec, FileVerificationSpec, RuntimeSpec, VerifiedFile,
    VerifiedRuntime, stage_runtime, stage_runtime_cleaned, verify_files_bounded_parallel,
};
use sha2::{Digest as _, Sha256};
use std::fs::{self, OpenOptions};
use std::io::{Cursor, Write as _};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use zip::{CompressionMethod, ZipWriter, write::SimpleFileOptions};

type TestResult = Result<(), Box<dyn std::error::Error>>;

struct Fixture {
    root: PathBuf,
    staging: PathBuf,
    spec: RuntimeSpec,
}

impl Fixture {
    fn new(bytes: &[u8]) -> Result<Self, Box<dyn std::error::Error>> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let stamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let root = std::env::temp_dir().join(format!(
            "SwiftBonsai-runtime-install-{}-{stamp}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root)?;
        let archive_path = root.join("runtime.zip");
        fs::write(&archive_path, bytes)?;
        Ok(Self {
            staging: root.join("runtime.staging"),
            spec: RuntimeSpec {
                archive_path,
                runtime_directory: root.join("runtime"),
                archive_bytes: u64::try_from(bytes.len())?,
                archive_sha256: digest(bytes),
            },
            root,
        })
    }

    fn remove(self) -> TestResult {
        assert!(self.root.is_absolute());
        assert_eq!(self.root.parent(), Some(std::env::temp_dir().as_path()));
        assert!(
            self.root
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("SwiftBonsai-runtime-install-"))
        );
        fs::remove_dir_all(self.root)?;
        Ok(())
    }
}

fn digest(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn image() -> Vec<u8> {
    let mut bytes = vec![0; 128];
    bytes[..2].copy_from_slice(b"MZ");
    bytes[60..64].copy_from_slice(&64_u32.to_le_bytes());
    bytes[64..70].copy_from_slice(&[b'P', b'E', 0, 0, 0x64, 0xaa]);
    bytes
}

fn archive(entries: &[(&str, &[u8])]) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut zip = ZipWriter::new(Cursor::new(Vec::new()));
    for &(name, content) in entries {
        let options = SimpleFileOptions::default().compression_method(CompressionMethod::Stored);
        if name.ends_with('/') {
            zip.add_directory(name, options)?;
        } else {
            zip.start_file(name, options)?;
            zip.write_all(content)?;
        }
    }
    Ok(zip.finish()?.into_inner())
}

fn assert_absent(path: &Path) {
    assert!(fs::symlink_metadata(path).is_err());
}

#[test]
fn runtime_stage_prepares_complete_tree_without_publishing() -> TestResult {
    let image = image();
    let bytes = archive(&[
        ("llama-server.exe", &image),
        ("bin/ggml.dll", &image),
        ("LICENSE", b"synthetic test"),
        ("empty/", b""),
    ])?;
    let fixture = Fixture::new(&bytes)?;
    let files = stage_runtime(&fixture.spec, &fixture.staging)?;
    assert_absent(&fixture.spec.runtime_directory);
    assert_eq!(files.len(), 3);
    assert!(
        files
            .iter()
            .all(|file| file.path().starts_with(&fixture.staging))
    );
    assert_eq!(fs::read(&fixture.spec.archive_path)?, bytes);
    assert_eq!(fs::read(fixture.staging.join("bin/ggml.dll"))?, image);
    assert!(fixture.staging.join("empty").is_dir());
    let staged_spec = RuntimeSpec {
        runtime_directory: fixture.staging.clone(),
        ..fixture.spec.clone()
    };
    let verified = VerifiedRuntime::verify(&staged_spec)?;
    assert_eq!(verified.files(), files);
    drop(verified);
    fixture.remove()
}

#[test]
fn runtime_stage_rejects_unsafe_archive_policy_before_staging() -> TestResult {
    let image = image();
    for entries in [
        vec![
            ("llama-server.exe", image.as_slice()),
            ("../escape", b"x".as_slice()),
        ],
        vec![("llama-server.exe", &image), ("NUL.dll", b"x")],
        vec![("llama-server.exe", &image), ("trailing./", b"")],
        vec![("llama-server.exe", &image), ("trailing /entry", b"x")],
        vec![("llama-server.exe", &image), ("bad\\entry", b"x")],
        vec![("llama-server.exe", &image), ("FILE", b"x"), ("file", b"y")],
        vec![
            ("llama-server.exe", &image),
            ("bin", b"x"),
            ("bin/file", b"y"),
        ],
        vec![
            ("llama-server.exe", &image),
            ("bin/llama-server.exe", &image),
        ],
        vec![("LICENSE", b"no server")],
    ] {
        let fixture = Fixture::new(&archive(&entries)?)?;
        assert_eq!(
            stage_runtime(&fixture.spec, &fixture.staging),
            Err(ArtifactError::Archive)
        );
        assert_absent(&fixture.spec.runtime_directory);
        assert_absent(&fixture.staging);
        assert_absent(&fixture.root.join("escape"));
        fixture.remove()?;
    }
    Ok(())
}

#[test]
fn runtime_stage_rejects_identity_architecture_and_corrupt_payloads() -> TestResult {
    let image = image();
    let bytes = archive(&[("llama-server.exe", &image)])?;
    let mut fixture = Fixture::new(&bytes)?;
    fixture.spec.archive_sha256 = "0".repeat(64);
    assert_eq!(
        stage_runtime(&fixture.spec, &fixture.staging),
        Err(ArtifactError::Identity)
    );
    assert_absent(&fixture.staging);
    assert_absent(&fixture.spec.runtime_directory);
    fixture.remove()?;

    let mut wrong = image.clone();
    wrong[68..70].copy_from_slice(&0x8664_u16.to_le_bytes());
    let fixture = Fixture::new(&archive(&[("llama-server.exe", &wrong)])?)?;
    assert_eq!(
        stage_runtime(&fixture.spec, &fixture.staging),
        Err(ArtifactError::Architecture)
    );
    assert_absent(&fixture.spec.runtime_directory);
    assert!(fixture.staging.join("llama-server.exe").is_file());
    fixture.remove()?;

    let mut corrupt = bytes;
    let position = corrupt
        .windows(image.len())
        .position(|window| window == image)
        .ok_or("Missing stored ZIP payload")?;
    corrupt[position + 127] ^= 1;
    let fixture = Fixture::new(&corrupt)?;
    assert_eq!(
        stage_runtime(&fixture.spec, &fixture.staging),
        Err(ArtifactError::Archive)
    );
    assert_absent(&fixture.spec.runtime_directory);
    assert!(fixture.staging.is_dir());
    fixture.remove()
}

#[test]
fn runtime_stage_cleanup_removes_only_its_failed_attempt() -> TestResult {
    let mut wrong = image();
    wrong[68..70].copy_from_slice(&0x8664_u16.to_le_bytes());
    let bytes = archive(&[("llama-server.exe", &wrong), ("share/LICENSE", b"fixture")])?;
    let fixture = Fixture::new(&bytes)?;
    let error = stage_runtime_cleaned(&fixture.spec, &fixture.staging)
        .err()
        .ok_or("Wrong architecture unexpectedly staged")?;
    assert_eq!(error.error(), ArtifactError::Architecture);
    assert_eq!(error.cleanup_error(), None);
    assert_absent(&fixture.staging);
    assert_eq!(fs::read(&fixture.spec.archive_path)?, bytes);
    assert!(error.to_string().contains("ARM64"));

    fs::create_dir(&fixture.staging)?;
    fs::write(fixture.staging.join("keep.txt"), b"owner data")?;
    let error = stage_runtime_cleaned(&fixture.spec, &fixture.staging)
        .err()
        .ok_or("Preexisting staging unexpectedly replaced")?;
    assert_eq!(error.error(), ArtifactError::Path);
    assert_eq!(fs::read(fixture.staging.join("keep.txt"))?, b"owner data");
    fixture.remove()
}

#[test]
fn runtime_stage_rejects_entry_and_expansion_budgets() -> TestResult {
    let image = image();
    let mut zip = ZipWriter::new(Cursor::new(Vec::new()));
    zip.start_file("llama-server.exe", SimpleFileOptions::default())?;
    zip.write_all(&image)?;
    for index in 0..4096 {
        zip.start_file(format!("entry-{index}"), SimpleFileOptions::default())?;
    }
    let fixture = Fixture::new(&zip.finish()?.into_inner())?;
    assert_eq!(
        stage_runtime(&fixture.spec, &fixture.staging),
        Err(ArtifactError::Archive)
    );
    assert_absent(&fixture.spec.runtime_directory);
    assert_absent(&fixture.staging);
    fixture.remove()?;

    let mut bytes = archive(&[("llama-server.exe", &image)])?;
    let central = bytes
        .array_windows()
        .position(|window| window == b"PK\x01\x02")
        .ok_or("Missing ZIP central header")?;
    bytes[central + 24..central + 28].copy_from_slice(&(512_u32 * 1024 * 1024 + 1).to_le_bytes());
    let fixture = Fixture::new(&bytes)?;
    assert_eq!(
        stage_runtime(&fixture.spec, &fixture.staging),
        Err(ArtifactError::Archive)
    );
    assert_absent(&fixture.spec.runtime_directory);
    assert_absent(&fixture.staging);
    fixture.remove()
}

#[test]
fn runtime_stage_preserves_existing_staging_and_destination() -> TestResult {
    let image = image();
    let bytes = archive(&[("llama-server.exe", &image)])?;
    for is_directory in [false, true] {
        let fixture = Fixture::new(&bytes)?;
        if is_directory {
            fs::create_dir(&fixture.spec.runtime_directory)?;
        } else {
            fs::write(&fixture.spec.runtime_directory, b"existing destination")?;
        }
        assert_eq!(
            stage_runtime(&fixture.spec, &fixture.staging),
            Err(ArtifactError::Path)
        );
        assert_absent(&fixture.staging);
        if is_directory {
            assert_eq!(fs::read_dir(&fixture.spec.runtime_directory)?.count(), 0);
        } else {
            assert_eq!(
                fs::read(&fixture.spec.runtime_directory)?,
                b"existing destination"
            );
        }
        fixture.remove()?;
    }
    let fixture = Fixture::new(&bytes)?;
    fs::create_dir(&fixture.staging)?;
    fs::write(fixture.staging.join("keep.txt"), b"existing staging")?;
    assert_eq!(
        stage_runtime(&fixture.spec, &fixture.staging),
        Err(ArtifactError::Path)
    );
    assert_eq!(
        fs::read(fixture.staging.join("keep.txt"))?,
        b"existing staging"
    );
    assert_absent(&fixture.spec.runtime_directory);
    fixture.remove()
}

struct TempArtifactDir {
    root: PathBuf,
}

impl TempArtifactDir {
    fn new(label: &str) -> Result<Self, Box<dyn std::error::Error>> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let stamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let root = std::env::temp_dir().join(format!(
            "SwiftBonsai-artifact-{label}-{}-{stamp}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root)?;
        Ok(Self { root })
    }

    fn path(&self) -> &Path {
        &self.root
    }

    fn remove(self) -> TestResult {
        assert!(self.root.is_absolute());
        assert_eq!(self.root.parent(), Some(std::env::temp_dir().as_path()));
        assert!(
            self.root
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("SwiftBonsai-artifact-"))
        );
        fs::remove_dir_all(self.root)?;
        Ok(())
    }
}

fn synthetic_bytes(len: usize, seed: usize) -> Vec<u8> {
    (0..len)
        .map(|i| {
            u8::try_from((i.wrapping_mul(31).wrapping_add(seed) ^ (i >> 8)) & 0xff).unwrap_or(0)
        })
        .collect()
}

fn chunk_digests(bytes: &[u8], chunk_bytes: usize) -> Vec<String> {
    bytes.chunks(chunk_bytes).map(digest).collect()
}

fn make_chunked_spec(
    path: PathBuf,
    payload: &[u8],
    chunk_bytes: usize,
) -> Result<ChunkedFileSpec, Box<dyn std::error::Error>> {
    Ok(ChunkedFileSpec {
        path,
        bytes: u64::try_from(payload.len())?,
        sha256: digest(payload),
        chunk_bytes,
        chunk_sha256s: chunk_digests(payload, chunk_bytes),
    })
}

fn assert_locked_until_drop(path: &Path, guard: VerifiedFile) -> TestResult {
    assert_eq!(guard.path(), path);
    assert!(OpenOptions::new().write(true).open(path).is_err());
    assert!(fs::remove_file(path).is_err());
    drop(guard);
    assert!(OpenOptions::new().write(true).open(path).is_ok());
    Ok(())
}

#[test]
fn chunk_manifest_verification_accepts_valid_mapped_and_buffered_files_and_retains_locks()
-> TestResult {
    let dir = TempArtifactDir::new("chunk-valid")?;

    // Mapped multi-chunk path (>= 4 MiB, 3 chunks at 4 MiB chunk size).
    let chunk_bytes = 4 * 1024 * 1024;
    let mapped_len = 2 * chunk_bytes + 19_456;
    let mapped_payload = synthetic_bytes(mapped_len, 17);
    let mapped_path = dir.path().join("swift-model.gguf");
    fs::write(&mapped_path, &mapped_payload)?;
    let mapped_spec = make_chunked_spec(mapped_path.clone(), &mapped_payload, chunk_bytes)?;
    assert_eq!(mapped_spec.chunk_sha256s.len(), 3);

    let cancel = AtomicBool::new(false);
    let guard = VerifiedFile::verify_chunked_with(&mapped_spec, 4, &cancel)?;
    assert_locked_until_drop(&mapped_path, guard)?;
    let default_guard = VerifiedFile::verify_chunked(&mapped_spec)?;
    assert_locked_until_drop(&mapped_path, default_guard)?;

    // Sequential buffered fallback (< 4 MiB with multi-chunk manifest).
    let small_payload = synthetic_bytes(100_000, 42);
    let small_path = dir.path().join("small-lora.bin");
    fs::write(&small_path, &small_payload)?;
    let small_spec = make_chunked_spec(small_path.clone(), &small_payload, 32_768)?;
    assert_eq!(small_spec.chunk_sha256s.len(), 4);
    let small_guard = VerifiedFile::verify_chunked_with(&small_spec, 4, &cancel)?;
    assert_locked_until_drop(&small_path, small_guard)?;

    // Exact chunk multiple and empty file boundary cases.
    let exact_payload = synthetic_bytes(65_536, 9);
    let exact_path = dir.path().join("exact-boundary.bin");
    fs::write(&exact_path, &exact_payload)?;
    let exact_spec = make_chunked_spec(exact_path.clone(), &exact_payload, 32_768)?;
    assert_eq!(exact_spec.chunk_sha256s.len(), 2);
    let exact_guard = VerifiedFile::verify_chunked(&exact_spec)?;
    assert_locked_until_drop(&exact_path, exact_guard)?;

    let empty_path = dir.path().join("empty.bin");
    fs::write(&empty_path, [])?;
    let empty_spec = make_chunked_spec(empty_path.clone(), &[], 32_768)?;
    assert!(empty_spec.chunk_sha256s.is_empty());
    let empty_guard = VerifiedFile::verify_chunked(&empty_spec)?;
    assert_locked_until_drop(&empty_path, empty_guard)?;

    dir.remove()
}

#[test]
fn chunk_manifest_verification_rejects_corruption_wrong_count_and_cancellation() -> TestResult {
    let dir = TempArtifactDir::new("chunk-reject")?;
    let chunk_bytes = 4 * 1024 * 1024;
    let mapped_len = 2 * chunk_bytes + 4_096;
    let mapped_payload = synthetic_bytes(mapped_len, 77);
    let mapped_path = dir.path().join("mapped-corrupt.gguf");
    fs::write(&mapped_path, &mapped_payload)?;
    let mapped_spec = make_chunked_spec(mapped_path.clone(), &mapped_payload, chunk_bytes)?;
    let cancel = AtomicBool::new(false);

    // Single-byte corruption in first, middle, and last chunks.
    for offset in [0, chunk_bytes + 55, mapped_len - 1] {
        let mut tampered = mapped_payload.clone();
        tampered[offset] ^= 0x01;
        fs::write(&mapped_path, &tampered)?;
        assert_eq!(
            VerifiedFile::verify_chunked_with(&mapped_spec, 4, &cancel).err(),
            Some(ArtifactError::Identity)
        );
        assert!(OpenOptions::new().write(true).open(&mapped_path).is_ok());
    }
    fs::write(&mapped_path, &mapped_payload)?;

    // Wrong chunk counts (too few, too many, empty).
    let mut short_chunks = mapped_spec.clone();
    short_chunks.chunk_sha256s.pop();
    assert_eq!(
        VerifiedFile::verify_chunked(&short_chunks).err(),
        Some(ArtifactError::Identity)
    );

    let mut extra_chunks = mapped_spec.clone();
    extra_chunks.chunk_sha256s.push("0".repeat(64));
    assert_eq!(
        VerifiedFile::verify_chunked(&extra_chunks).err(),
        Some(ArtifactError::Identity)
    );

    let mut no_chunks = mapped_spec.clone();
    no_chunks.chunk_sha256s.clear();
    assert_eq!(
        VerifiedFile::verify_chunked(&no_chunks).err(),
        Some(ArtifactError::Identity)
    );

    // Corrupted digest in chunk manifest and uppercase digest syntax.
    let mut bad_chunk_hex = mapped_spec.clone();
    bad_chunk_hex.chunk_sha256s[1] = "f".repeat(64);
    assert_eq!(
        VerifiedFile::verify_chunked(&bad_chunk_hex).err(),
        Some(ArtifactError::Identity)
    );

    let mut uppercase_hex = mapped_spec.clone();
    uppercase_hex.chunk_sha256s[0] = uppercase_hex.chunk_sha256s[0].to_ascii_uppercase();
    assert_eq!(
        VerifiedFile::verify_chunked(&uppercase_hex).err(),
        Some(ArtifactError::Identity)
    );

    // Length mismatch and invalid zero chunk size.
    let mut wrong_len = mapped_spec.clone();
    wrong_len.bytes -= 1;
    assert_eq!(
        VerifiedFile::verify_chunked(&wrong_len).err(),
        Some(ArtifactError::Identity)
    );

    let mut zero_chunk = mapped_spec.clone();
    zero_chunk.chunk_bytes = 0;
    assert_eq!(
        VerifiedFile::verify_chunked(&zero_chunk).err(),
        Some(ArtifactError::Configuration)
    );

    // Immediate cancellation.
    let cancelled = AtomicBool::new(true);
    assert_eq!(
        VerifiedFile::verify_chunked_with(&mapped_spec, 4, &cancelled).err(),
        Some(ArtifactError::Cancelled)
    );

    // Small sequential fallback rejections (byte corruption, wrong whole sha256, wrong chunk count).
    let small_payload = synthetic_bytes(100_000, 12);
    let small_path = dir.path().join("small-corrupt.bin");
    fs::write(&small_path, &small_payload)?;
    let small_spec = make_chunked_spec(small_path.clone(), &small_payload, 32_768)?;

    for offset in [0, 40_000, small_payload.len() - 1] {
        let mut tampered = small_payload.clone();
        tampered[offset] ^= 0x80;
        fs::write(&small_path, &tampered)?;
        assert_eq!(
            VerifiedFile::verify_chunked(&small_spec).err(),
            Some(ArtifactError::Identity)
        );
    }
    fs::write(&small_path, &small_payload)?;

    let mut bad_whole_sha = small_spec.clone();
    bad_whole_sha.sha256 = "0".repeat(64);
    assert_eq!(
        VerifiedFile::verify_chunked(&bad_whole_sha).err(),
        Some(ArtifactError::Identity)
    );

    let mut bad_small_count = small_spec;
    bad_small_count.chunk_sha256s.pop();
    assert_eq!(
        VerifiedFile::verify_chunked(&bad_small_count).err(),
        Some(ArtifactError::Identity)
    );

    dir.remove()
}

#[test]
fn bounded_parallel_verification_preserves_input_order_and_retains_all_locks() -> TestResult {
    let dir = TempArtifactDir::new("bounded-parallel-ok")?;

    let swift_bytes = synthetic_bytes(2 * 4 * 1024 * 1024 + 8_192, 1);
    let orca_bytes = synthetic_bytes(262_144, 2);
    let mmproj_bytes = synthetic_bytes(4 * 1024 * 1024 + 2_048, 3);
    let speech_bytes = synthetic_bytes(131_072, 4);

    let swift_path = dir.path().join("swift.gguf");
    let orca_path = dir.path().join("orca-lora.bin");
    let mmproj_path = dir.path().join("mmproj.bin");
    let speech_path = dir.path().join("speech.gguf");

    fs::write(&swift_path, &swift_bytes)?;
    fs::write(&orca_path, &orca_bytes)?;
    fs::write(&mmproj_path, &mmproj_bytes)?;
    fs::write(&speech_path, &speech_bytes)?;

    let specs = vec![
        FileVerificationSpec::chunked(make_chunked_spec(
            swift_path.clone(),
            &swift_bytes,
            4 * 1024 * 1024,
        )?),
        FileVerificationSpec::whole(
            orca_path.clone(),
            u64::try_from(orca_bytes.len())?,
            digest(&orca_bytes),
        ),
        FileVerificationSpec::whole(
            mmproj_path.clone(),
            u64::try_from(mmproj_bytes.len())?,
            digest(&mmproj_bytes),
        ),
        FileVerificationSpec::chunked(make_chunked_spec(
            speech_path.clone(),
            &speech_bytes,
            65_536,
        )?),
    ];

    let cancel = AtomicBool::new(false);
    assert!(verify_files_bounded_parallel(&[], 4, &cancel)?.is_empty());

    for max_workers in [0, 1, 4, 16] {
        let guards = verify_files_bounded_parallel(&specs, max_workers, &cancel)?;
        assert_eq!(guards.len(), specs.len());
        for (guard, spec) in guards.iter().zip(specs.iter()) {
            assert_eq!(guard.path(), spec.path());
            assert!(OpenOptions::new().write(true).open(spec.path()).is_err());
            assert!(fs::remove_file(spec.path()).is_err());
        }
        drop(guards);
        for spec in &specs {
            assert!(OpenOptions::new().write(true).open(spec.path()).is_ok());
        }
    }

    let cancelled = AtomicBool::new(true);
    assert_eq!(
        verify_files_bounded_parallel(&specs, 4, &cancelled).err(),
        Some(ArtifactError::Cancelled)
    );

    dir.remove()
}

#[test]
fn bounded_parallel_verification_cancels_peers_early_on_corrupt_artifact() -> TestResult {
    let dir = TempArtifactDir::new("bounded-parallel-corrupt")?;

    let valid_large = synthetic_bytes(16 * 1024 * 1024 + 4_096, 5);
    let corrupt_small = synthetic_bytes(4_096, 6);
    let trailing_large = synthetic_bytes(8 * 1024 * 1024 + 1_024, 7);

    let large_path = dir.path().join("swift-large.gguf");
    let corrupt_path = dir.path().join("corrupt-lora.bin");
    let trailing_path = dir.path().join("speech-trailing.gguf");

    fs::write(&large_path, &valid_large)?;
    fs::write(&corrupt_path, &corrupt_small)?;
    fs::write(&trailing_path, &trailing_large)?;

    let specs = vec![
        FileVerificationSpec::chunked(make_chunked_spec(
            large_path.clone(),
            &valid_large,
            4 * 1024 * 1024,
        )?),
        FileVerificationSpec::whole(
            corrupt_path.clone(),
            u64::try_from(corrupt_small.len())?,
            "0".repeat(64),
        ),
        FileVerificationSpec::whole(
            trailing_path.clone(),
            u64::try_from(trailing_large.len())?,
            digest(&trailing_large),
        ),
    ];

    let cancel = AtomicBool::new(false);
    // Corrupt peer must report Identity (never masked by peer-induced Cancelled)
    // and release all opened guards before returning.
    assert_eq!(
        verify_files_bounded_parallel(&specs, 4, &cancel).err(),
        Some(ArtifactError::Identity)
    );
    for path in [&large_path, &corrupt_path, &trailing_path] {
        assert!(OpenOptions::new().write(true).open(path).is_ok());
    }

    // Missing peer reports Path cleanly without retaining earlier valid guards.
    let missing_specs = vec![
        FileVerificationSpec::whole(
            corrupt_path.clone(),
            u64::try_from(corrupt_small.len())?,
            digest(&corrupt_small),
        ),
        FileVerificationSpec::whole(dir.path().join("missing.gguf"), 16, "0".repeat(64)),
    ];
    assert_eq!(
        verify_files_bounded_parallel(&missing_specs, 4, &cancel).err(),
        Some(ArtifactError::Path)
    );
    assert!(OpenOptions::new().write(true).open(&corrupt_path).is_ok());

    dir.remove()
}
