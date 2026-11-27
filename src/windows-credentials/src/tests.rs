use super::*;
use std::fs::OpenOptions;
use std::os::windows::fs::symlink_dir;

fn test_lane(name: &str) -> io::Result<PathBuf> {
    let parent = std::env::var_os("BONSAI_CREDENTIAL_TEST_ROOT")
        .map_or_else(std::env::temp_dir, PathBuf::from);
    validate_external_path(&parent)?;
    let mut nonce = [0; 16];
    native::random(&mut nonce)?;
    let lane = parent.join(format!("credentials-{name}-{}", hex(&nonce)));
    fs::create_dir(&lane)?;
    Ok(lane)
}

#[test]
fn read_validation_keeps_regular_file_and_repository_checks() -> io::Result<()> {
    let parent = test_lane("read-validation")?;
    let path = parent.join("input.bin");
    fs::write(&path, b"approved input")?;
    validate_read_path(&path)?;
    assert!(validate_read_path(&parent).is_err());
    assert!(validate_read_path(&parent.join("missing.bin")).is_err());
    let repository = parent.join("repository");
    fs::create_dir(&repository)?;
    fs::write(repository.join(".git"), b"gitdir: elsewhere")?;
    fs::write(repository.join("input.bin"), b"unapproved input")?;
    assert!(validate_read_path(&repository.join("input.bin")).is_err());
    Ok(())
}

#[test]
fn private_worker_storage_has_protected_acl_without_extra_credentials() -> io::Result<()> {
    let parent = test_lane("worker-storage")?;
    let directory = PrivateDirectory::create(&parent)?;
    native::Security::new(true)?.verify(&directory._reader)?;
    assert!(native::noninheritable(&directory._reader)?);
    assert_eq!(fs::read_dir(directory.path())?.count(), 0);
    fs::write(directory.path().join("scratch.txt"), b"worker output")?;
    assert!(fs::rename(directory.path(), parent.join("renamed")).is_err());
    let path = directory.path().to_path_buf();
    drop(directory);
    assert_eq!(fs::read(path.join("scratch.txt"))?, b"worker output");
    Ok(())
}

#[test]
fn retained_tokens_validate_content_and_hold_files_until_drop() -> io::Result<()> {
    let lane = test_lane("retained")?;
    let credentials = PrivateCredentials::create(&lane)?;
    let path = credentials.frontend_path().to_owned();
    let retained = RetainedCredential::open(&path)?;
    // Keep assertions value-free even when the secret differs unexpectedly.
    let same_text = retained.as_hex() == credentials.frontend_key();
    let same_bytes = retained.bytes().as_slice() == credentials.frontend_key().as_bytes();
    assert!(same_text);
    assert!(same_bytes);
    assert!(native::noninheritable(&retained._reader)?);
    drop(credentials);
    assert!(OpenOptions::new().write(true).open(&path).is_err());
    assert!(fs::remove_file(&path).is_err());
    drop(retained);
    assert!(OpenOptions::new().write(true).open(&path).is_ok());

    let malformed = lane.join("malformed.key");
    for bytes in [
        Vec::new(),
        vec![b'a'; 63],
        vec![b'a'; 65],
        vec![b'A'; 64],
        vec![b'g'; 64],
        vec![b'\n'; 64],
        vec![0xff; 64],
    ] {
        fs::write(&malformed, bytes)?;
        assert!(RetainedCredential::open(&malformed).is_err());
        // Failed validation releases its handle as well as its secret buffer.
        assert!(OpenOptions::new().write(true).open(&malformed).is_ok());
    }
    assert!(RetainedCredential::open(&lane).is_err());
    let repository = lane.join("repository");
    fs::create_dir(&repository)?;
    fs::write(repository.join(".git"), b"gitdir: elsewhere")?;
    fs::write(repository.join("token.key"), [b'a'; 64])?;
    assert!(RetainedCredential::open(&repository.join("token.key")).is_err());
    Ok(())
}

#[test]
fn credentials_have_protected_acls_distinct_keys_and_retained_readers() -> io::Result<()> {
    let lane = test_lane("creation")?;
    let credentials = PrivateCredentials::create(&lane)?;
    assert_eq!(credentials.frontend_key().len(), 64);
    assert_eq!(credentials.backend_key().len(), 64);
    assert!(
        credentials
            .frontend_key()
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
    );
    assert!(
        credentials
            .backend_key()
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
    );
    // Do not let an assertion failure debug-format secret values.
    let distinct = credentials.frontend_key() != credentials.backend_key();
    assert!(distinct);
    let directory_security = native::Security::new(true)?;
    let file_security = native::Security::new(false)?;
    directory_security.verify(&credentials._directory_handle)?;
    let backend_handle = credentials
        ._backend_handle
        .as_ref()
        .ok_or_else(|| io::Error::other("missing backend key handle"))?;
    file_security.verify(&credentials._frontend_handle)?;
    file_security.verify(backend_handle)?;
    assert!(native::noninheritable(&credentials._directory_handle)?);
    assert!(native::noninheritable(&credentials._frontend_handle)?);
    assert!(native::noninheritable(backend_handle)?);
    for path in [credentials.frontend_path(), credentials.backend_path()] {
        // ReadAllText-style readers can open while retained handles are live.
        let mut on_disk = Secret::<64>::new();
        let mut reader = native::open_reader(path, false)?;
        reader.read_exact(&mut *on_disk.0)?;
        assert_eq!(
            on_disk.0.len(),
            usize::try_from(reader.metadata()?.len()).map_err(io::Error::other)?
        );
        assert!(native::create_file(path, &file_security).is_err());
        assert!(OpenOptions::new().write(true).open(path).is_err());
        assert!(fs::remove_file(path).is_err());
    }
    let retained_directory = credentials.directory().to_path_buf();
    let retained_frontend = credentials.frontend_path().to_path_buf();
    assert!(fs::rename(&retained_directory, lane.join("moved")).is_err());
    drop(credentials);
    assert!(retained_directory.is_dir());
    assert!(retained_frontend.is_file());
    Ok(())
}

#[test]
fn protected_acl_precedes_bytes_and_create_new_never_overwrites() -> io::Result<()> {
    let lane = test_lane("exclusive")?;
    let directory = lane.join("private");
    let directory_security = native::Security::new(true)?;
    native::create_directory(&directory, &directory_security)?;
    let directory_handle = native::open_reader(&directory, true)?;
    directory_security.verify(&directory_handle)?;
    assert!(native::create_directory(&directory, &directory_security).is_err());
    let file_security = native::Security::new(false)?;
    let path = directory.join("empty.key");
    let mut writer = native::create_file(&path, &file_security)?;
    assert_eq!(writer.metadata()?.len(), 0);
    file_security.verify(&writer)?;
    writer.write_all(b"sentinel")?;
    writer.sync_all()?;
    assert!(native::create_file(&path, &file_security).is_err());
    drop(writer);
    assert_eq!(fs::read(&path)?, b"sentinel");
    Ok(())
}

#[test]
fn path_validation_rejects_aliases_git_and_reparse_ancestors() -> io::Result<()> {
    let lane = test_lane("paths")?;
    validate_external_path(Path::new(&format!("{}\\", lane.display())))?;
    validate_external_path(&lane.join("new-leaf"))?;
    validate_external_path(&lane.join("emoji-🦀"))?;
    for path in [
        "relative",
        "C:",
        "C:\\",
        "C:relative",
        "\\\\server\\share\\key",
        "\\\\?\\C:\\keys",
        "//./C:/keys",
        "C:\\keys:stream",
        "C:\\%TEMP%\\keys",
        "C:\\a\\.\\key",
        "C:\\a\\..\\key",
        "C:\\a\\key\0tail",
        "C:\\a\\CON.key",
        "C:\\a\\COM1",
        "C:\\a\\LPT9.key",
        "C:\\a\\key.",
        "C:\\a\\key ",
    ] {
        assert!(validate_external_path(Path::new(path)).is_err());
    }
    assert!(validate_external_path(&lane.join("missing").join("leaf")).is_err());
    let repository = lane.join("repository");
    fs::create_dir(&repository)?;
    fs::write(repository.join(".git"), b"gitdir: elsewhere")?;
    assert!(validate_external_path(&repository.join("new-leaf")).is_err());
    let target = lane.join("target");
    fs::create_dir(&target)?;
    let alias = lane.join("alias");
    symlink_dir(&target, &alias)?;
    assert!(validate_external_path(&alias.join("new-leaf")).is_err());
    assert!(!reserved_component("🦀"));
    Ok(())
}

#[test]
fn volatile_scrub_zeroes_the_owned_buffer() {
    let mut secret = Secret::<64>::new();
    secret.0.fill(0x5a);
    let mut bytes = [0x5a_u8; 64];
    bytes.zeroize();
    assert!(bytes.iter().all(|byte| *byte == 0));
    drop(secret);
}

#[test]
fn path_validation_rejects_win32_device_aliases_and_control_characters() -> io::Result<()> {
    let lane = test_lane("device-aliases")?;
    for component in [
        "NUL .key",
        "CONIN$",
        "CONOUT$",
        "COM\u{b9}",
        "com\u{b2}.key",
        "COM\u{b3}.log",
        "LPT\u{b9}.key",
        "lpt\u{b2}",
        "LPT\u{b3}.txt",
        "key\u{1}file",
        "key\tfile",
        "key\u{1f}file",
    ] {
        assert!(validate_external_path(&lane.join(component)).is_err());
    }
    for component in ["COM0", "COM10.key", "LPT10", "company.key", "\u{1f980}"] {
        validate_external_path(&lane.join(component))?;
        assert!(!reserved_component(component));
    }
    Ok(())
}

#[test]
fn frontend_only_credentials_generate_in_memory_backend_key_without_backend_file() -> io::Result<()>
{
    let lane = test_lane("frontend-only")?;
    let credentials = PrivateCredentials::create_frontend_only(&lane)?;
    assert!(!credentials.has_backend_file());
    assert!(credentials.frontend_path().exists());
    assert!(!credentials.backend_path().exists());
    assert_eq!(fs::read_dir(credentials.directory())?.count(), 1);
    for key in [credentials.frontend_key(), credentials.backend_key()] {
        assert_eq!(key.len(), 64);
        assert!(
            key.bytes()
                .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
        );
    }
    assert_ne!(credentials.frontend_key(), credentials.backend_key());

    let retained = RetainedCredential::open(credentials.frontend_path())?;
    assert_eq!(retained.as_hex(), credentials.frontend_key());
    drop(retained);
    drop(credentials);
    Ok(())
}

fn open_read_locked_test_file(path: &Path) -> io::Result<File> {
    use std::os::windows::fs::OpenOptionsExt as _;

    OpenOptions::new()
        .read(true)
        .share_mode(windows_sys::Win32::Storage::FileSystem::FILE_SHARE_READ)
        .custom_flags(windows_sys::Win32::Storage::FileSystem::FILE_FLAG_OPEN_REPARSE_POINT)
        .open(path)
}

fn synthetic_chunk_payload(len: usize) -> Vec<u8> {
    const TILE_LEN: usize = 131_072;
    let tile: Vec<u8> = (0..TILE_LEN.min(len))
        .map(|i| u8::try_from((i.wrapping_mul(37) ^ (i >> 9) ^ 0x5a) & 0xff).unwrap_or(0))
        .collect();
    let mut payload = vec![0_u8; len];
    if !tile.is_empty() {
        for chunk in payload.chunks_mut(tile.len()) {
            chunk.copy_from_slice(&tile[..chunk.len()]);
        }
    }
    payload
}

#[test]
fn parallel_mapped_chunk_digests_compute_and_verify_exact_multichunk() -> io::Result<()> {
    let lane = test_lane("parallel-chunk-ok")?;
    let chunk_bytes = usize::try_from(MIN_MAPPED_HASH_BYTES).map_err(io::Error::other)?;
    let len = 2 * chunk_bytes + 73_219;
    let payload = synthetic_chunk_payload(len);
    let path = lane.join("multichunk.bin");
    fs::write(&path, &payload)?;

    let expected: Vec<[u8; 32]> = payload
        .chunks(chunk_bytes)
        .map(Sha256::digest_bytes)
        .collect::<io::Result<Vec<_>>>()?;
    assert_eq!(expected.len(), 3);

    let file = open_read_locked_test_file(&path)?;
    let cancel = std::sync::atomic::AtomicBool::new(false);
    let total_u64 = u64::try_from(len).map_err(io::Error::other)?;

    for workers in [
        0,
        1,
        4,
        MAX_PARALLEL_HASH_WORKERS,
        MAX_PARALLEL_HASH_WORKERS + 1,
    ] {
        let computed =
            compute_mapped_chunk_digests(&file, total_u64, chunk_bytes, workers, &cancel)?;
        assert_eq!(computed.as_deref(), Some(expected.as_slice()));
        let verified = try_verify_locked_file_chunk_digests_mapped(
            &file,
            total_u64,
            chunk_bytes,
            &expected,
            workers,
            &cancel,
        )?;
        assert_eq!(verified, Some(true));
    }

    // Also verify default 16 MiB chunk size covering the full 8+ MiB payload as one chunk.
    let whole_expected = [Sha256::digest_bytes(&payload)?];
    assert_eq!(
        compute_mapped_chunk_digests(
            &file,
            total_u64,
            PARALLEL_HASH_DEFAULT_CHUNK_BYTES,
            4,
            &cancel
        )?
        .as_deref(),
        Some(whole_expected.as_slice())
    );
    assert_eq!(
        try_verify_locked_file_chunk_digests_mapped(
            &file,
            total_u64,
            PARALLEL_HASH_DEFAULT_CHUNK_BYTES,
            &whole_expected,
            4,
            &cancel
        )?,
        Some(true)
    );

    drop(file);
    fs::remove_dir_all(&lane)?;
    Ok(())
}

#[test]
fn parallel_mapped_chunk_verification_detects_single_byte_corruption_in_any_chunk() -> io::Result<()>
{
    let lane = test_lane("parallel-chunk-corrupt")?;
    let chunk_bytes = usize::try_from(MIN_MAPPED_HASH_BYTES).map_err(io::Error::other)?;
    let len = 2 * chunk_bytes + 1_024;
    let payload = synthetic_chunk_payload(len);
    let expected: Vec<[u8; 32]> = payload
        .chunks(chunk_bytes)
        .map(Sha256::digest_bytes)
        .collect::<io::Result<Vec<_>>>()?;
    let total_u64 = u64::try_from(len).map_err(io::Error::other)?;
    let cancel = std::sync::atomic::AtomicBool::new(false);

    // Corrupt a single byte in the first, middle, and last chunks on disk.
    let path = lane.join("corrupt.bin");
    fs::write(&path, &payload)?;
    let toggle_byte = |offset: usize| -> io::Result<()> {
        use std::io::{Read as _, Seek as _, SeekFrom, Write as _};
        let mut file = OpenOptions::new().read(true).write(true).open(&path)?;
        let pos = u64::try_from(offset).map_err(io::Error::other)?;
        file.seek(SeekFrom::Start(pos))?;
        let mut byte = [0_u8; 1];
        file.read_exact(&mut byte)?;
        byte[0] ^= 0x01;
        file.seek(SeekFrom::Start(pos))?;
        file.write_all(&byte)?;
        file.flush()
    };
    for offset in [0, chunk_bytes - 1, chunk_bytes + 123, len - 1] {
        toggle_byte(offset)?;
        let file = open_read_locked_test_file(&path)?;
        let actual = try_verify_locked_file_chunk_digests_mapped(
            &file,
            total_u64,
            chunk_bytes,
            &expected,
            MAX_PARALLEL_HASH_WORKERS,
            &cancel,
        )?;
        drop(file);
        toggle_byte(offset)?;
        assert_eq!(actual, Some(false));
    }

    // Clean payload restored on disk, but corrupt expected digest for first, middle, and last chunk.
    let file = open_read_locked_test_file(&path)?;
    for chunk_index in [0, 1, 2] {
        let mut bad_expected = expected.clone();
        bad_expected[chunk_index][31] ^= 0x80;
        assert_eq!(
            try_verify_locked_file_chunk_digests_mapped(
                &file,
                total_u64,
                chunk_bytes,
                &bad_expected,
                MAX_PARALLEL_HASH_WORKERS,
                &cancel
            )?,
            Some(false)
        );
    }

    drop(file);
    fs::remove_dir_all(&lane)?;
    Ok(())
}

#[test]
fn parallel_mapped_chunk_verification_rejects_count_chunk_size_and_length_mismatches()
-> io::Result<()> {
    let lane = test_lane("parallel-chunk-mismatch")?;
    let chunk_bytes = usize::try_from(MIN_MAPPED_HASH_BYTES).map_err(io::Error::other)?;
    let len = 2 * chunk_bytes + 512;
    let path = lane.join("mismatch.bin");
    let created = File::create(&path)?;
    created.set_len(u64::try_from(len).map_err(io::Error::other)?)?;
    drop(created);

    let expected = vec![[0_u8; 32]; 3];
    let file = open_read_locked_test_file(&path)?;
    let total_u64 = u64::try_from(len).map_err(io::Error::other)?;
    let cancel = std::sync::atomic::AtomicBool::new(false);

    // Sub-threshold chunk size must fail cleanly as InvalidInput.
    for bad_chunk_bytes in [0, 1024, chunk_bytes - 1] {
        let compute_err =
            compute_mapped_chunk_digests(&file, total_u64, bad_chunk_bytes, 4, &cancel)
                .err()
                .ok_or_else(|| io::Error::other("expected invalid chunk size rejection"))?;
        assert_eq!(compute_err.kind(), io::ErrorKind::InvalidInput);
        let verify_err = try_verify_locked_file_chunk_digests_mapped(
            &file,
            total_u64,
            bad_chunk_bytes,
            &expected,
            4,
            &cancel,
        )
        .err()
        .ok_or_else(|| io::Error::other("expected invalid chunk size rejection"))?;
        assert_eq!(verify_err.kind(), io::ErrorKind::InvalidInput);
    }

    // Mismatched expected chunk digest count must fail cleanly as InvalidInput.
    let mut too_many = expected.clone();
    too_many.push([0_u8; 32]);
    for bad_digests in [&expected[..2], &too_many[..], &[][..]] {
        let count_err = try_verify_locked_file_chunk_digests_mapped(
            &file,
            total_u64,
            chunk_bytes,
            bad_digests,
            4,
            &cancel,
        )
        .err()
        .ok_or_else(|| io::Error::other("expected chunk count mismatch rejection"))?;
        assert_eq!(count_err.kind(), io::ErrorKind::InvalidInput);
    }

    // File length mismatch returns Ok(None) rather than verifying mismatched bytes.
    assert_eq!(
        compute_mapped_chunk_digests(&file, total_u64 - 1, chunk_bytes, 4, &cancel)?,
        None
    );
    assert_eq!(
        try_verify_locked_file_chunk_digests_mapped(
            &file,
            total_u64 - 1,
            chunk_bytes,
            &expected,
            4,
            &cancel
        )?,
        None
    );
    assert_eq!(
        try_verify_locked_file_chunk_digests_mapped(
            &file,
            total_u64 + 1,
            chunk_bytes,
            &expected,
            4,
            &cancel
        )?,
        None
    );

    // Below MIN_MAPPED_HASH_BYTES returns Ok(None) for sequential fallback.
    let small_path = lane.join("small.bin");
    let small_bytes = MIN_MAPPED_HASH_BYTES - 1;
    let small_file = File::create(&small_path)?;
    small_file.set_len(small_bytes)?;
    drop(small_file);
    let small_file = open_read_locked_test_file(&small_path)?;
    assert_eq!(
        compute_mapped_chunk_digests(&small_file, small_bytes, chunk_bytes, 4, &cancel)?,
        None
    );
    assert_eq!(
        try_verify_locked_file_chunk_digests_mapped(
            &small_file,
            small_bytes,
            chunk_bytes,
            &[[0_u8; 32]],
            4,
            &cancel
        )?,
        None
    );

    drop(small_file);
    drop(file);
    fs::remove_dir_all(&lane)?;
    Ok(())
}

#[test]
fn parallel_mapped_chunk_verification_honors_immediate_cancellation() -> io::Result<()> {
    let lane = test_lane("parallel-chunk-cancel")?;
    let chunk_bytes = usize::try_from(MIN_MAPPED_HASH_BYTES).map_err(io::Error::other)?;
    let len = 2 * chunk_bytes + 512;
    let path = lane.join("cancel.bin");
    let created = File::create(&path)?;
    created.set_len(u64::try_from(len).map_err(io::Error::other)?)?;
    drop(created);

    let expected = vec![[0_u8; 32]; 3];
    let file = open_read_locked_test_file(&path)?;
    let total_u64 = u64::try_from(len).map_err(io::Error::other)?;

    let cancelled = std::sync::atomic::AtomicBool::new(true);
    let compute_err = compute_mapped_chunk_digests(&file, total_u64, chunk_bytes, 4, &cancelled)
        .err()
        .ok_or_else(|| io::Error::other("expected immediate cancellation"))?;
    assert_eq!(compute_err.kind(), io::ErrorKind::Interrupted);

    let verify_err = try_verify_locked_file_chunk_digests_mapped(
        &file,
        total_u64,
        chunk_bytes,
        &expected,
        4,
        &cancelled,
    )
    .err()
    .ok_or_else(|| io::Error::other("expected immediate cancellation"))?;
    assert_eq!(verify_err.kind(), io::ErrorKind::Interrupted);

    // Also verify mid-flight cancellation triggers at the next prefetch/hash check.
    let checks = std::sync::atomic::AtomicUsize::new(0);
    let mid_cancel = || checks.fetch_add(1, std::sync::atomic::Ordering::AcqRel) >= 2;
    let mid_err = try_verify_locked_file_chunk_digests_mapped_with(
        &file,
        total_u64,
        chunk_bytes,
        &expected,
        1,
        &mid_cancel,
    )
    .err()
    .ok_or_else(|| io::Error::other("expected mid-flight cancellation"))?;
    assert_eq!(mid_err.kind(), io::ErrorKind::Interrupted);

    drop(file);
    fs::remove_dir_all(&lane)?;
    Ok(())
}
