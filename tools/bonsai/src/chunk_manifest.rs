//! Generate and verify parallel mapped CNG chunk-SHA-256 manifests for model artifacts.

use serde_json::Value;

/// Compute or verify a fixed-chunk SHA-256 manifest (`bytes`, `sha256`, `chunkBytes`, `chunkSha256s`).
///
/// # Errors
/// Rejects unsupported platforms, malformed requests, unsafe paths, invalid chunk
/// parameters, and byte-length or digest mismatches.
pub fn run(request: Value) -> crate::Result<Value> {
    #[cfg(windows)]
    {
        windows::run(request)
    }
    #[cfg(not(windows))]
    {
        let _ = request;
        Err("Chunk manifest CNG operations require Windows.".into())
    }
}

#[cfg(windows)]
mod windows {
    use std::fs::{self, File};
    use std::io::{Read as _, Seek as _, SeekFrom};
    use std::path::{Path, PathBuf};
    use std::sync::atomic::AtomicBool;

    use bonsai_artifacts::{
        ChunkedFileSpec, VerifiedFile, open_locked_file, validate_external_path,
    };
    use bonsai_windows_credentials::{
        MAX_PARALLEL_HASH_WORKERS, MIN_MAPPED_HASH_BYTES, PARALLEL_HASH_DEFAULT_CHUNK_BYTES,
        Sha256, compute_mapped_chunk_digests, try_hash_locked_file_mapped, validate_read_path,
    };
    use serde::Deserialize;
    use serde_json::{Value, json};

    #[derive(Deserialize)]
    #[serde(tag = "operation", rename_all = "kebab-case", deny_unknown_fields)]
    enum Request {
        #[serde(rename_all = "camelCase")]
        Generate {
            #[serde(default)]
            schema_version: Option<u32>,
            path: PathBuf,
            #[serde(default)]
            chunk_bytes: Option<usize>,
            #[serde(default)]
            workers: Option<usize>,
            #[serde(default)]
            output_path: Option<PathBuf>,
        },
        #[serde(rename_all = "camelCase")]
        Verify {
            #[serde(default)]
            schema_version: Option<u32>,
            path: PathBuf,
            bytes: u64,
            sha256: String,
            #[serde(default)]
            chunk_bytes: Option<usize>,
            #[serde(default)]
            chunk_sha256s: Option<Vec<String>>,
            #[serde(default)]
            workers: Option<usize>,
        },
    }

    fn check_schema_version(version: Option<u32>) -> crate::Result<()> {
        if version.is_some_and(|v| v != 1) {
            return Err("Unsupported chunk-manifest schemaVersion.".into());
        }
        Ok(())
    }

    fn resolve_workers(requested: Option<usize>) -> crate::Result<usize> {
        match requested {
            Some(0) => Err("Chunk manifest workers must be at least 1.".into()),
            Some(workers) => Ok(workers.clamp(1, MAX_PARALLEL_HASH_WORKERS)),
            None => Ok(std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(4)
                .clamp(1, MAX_PARALLEL_HASH_WORKERS)),
        }
    }

    fn validate_input_file(path: &Path) -> crate::Result<()> {
        if !path.is_absolute() {
            return Err("Artifact path must be expanded and absolute.".into());
        }
        validate_external_path(path).map_err(|error| error.to_string())?;
        validate_read_path(path).map_err(|error| error.to_string())
    }

    fn whole_file_sha256_cng(file: &mut File, expected_bytes: u64) -> crate::Result<String> {
        if let Some(digest) =
            try_hash_locked_file_mapped(file, expected_bytes, &mut || Ok::<(), String>(()), |e| {
                e.to_string()
            })?
        {
            return Ok(hex::encode(digest));
        }
        file.seek(SeekFrom::Start(0))
            .map_err(|error| error.to_string())?;
        let (_, digest) =
            crate::identity::hash_reader_checked(file, Some(expected_bytes), None, &mut || Ok(()))?;
        Ok(digest)
    }

    fn compute_chunk_digests_buffered(
        file: &mut File,
        expected_bytes: u64,
        chunk_bytes: usize,
    ) -> crate::Result<Vec<String>> {
        file.seek(SeekFrom::Start(0))
            .map_err(|error| error.to_string())?;
        let chunk_bytes_u64 = u64::try_from(chunk_bytes).map_err(|error| error.to_string())?;
        let expected_chunks = usize::try_from(expected_bytes.div_ceil(chunk_bytes_u64))
            .map_err(|error| error.to_string())?;
        let mut digests = Vec::with_capacity(expected_chunks);
        let buffer_len = chunk_bytes.clamp(1, 1_048_576);
        let mut buffer = vec![0_u8; buffer_len];
        let mut total = 0_u64;
        let mut chunk_remaining = chunk_bytes_u64;
        let mut hasher = Sha256::new().map_err(|error| error.to_string())?;

        loop {
            let max_read = usize::try_from(chunk_remaining)
                .unwrap_or(buffer.len())
                .min(buffer.len());
            let count = match file.read(&mut buffer[..max_read]) {
                Ok(count) => count,
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(error.to_string()),
            };
            if count == 0 {
                break;
            }
            let count_u64 = u64::try_from(count).map_err(|error| error.to_string())?;
            total = total
                .checked_add(count_u64)
                .ok_or("Chunked artifact length overflowed.")?;
            if total > expected_bytes {
                return Err("Artifact grew during chunk hashing.".into());
            }
            hasher
                .update(&buffer[..count])
                .map_err(|error| error.to_string())?;
            chunk_remaining -= count_u64;
            if chunk_remaining == 0 {
                let finished = hasher.finish().map_err(|error| error.to_string())?;
                digests.push(hex::encode(finished));
                hasher = Sha256::new().map_err(|error| error.to_string())?;
                chunk_remaining = chunk_bytes_u64;
            }
        }
        if total != expected_bytes {
            return Err("Artifact length changed during chunk hashing.".into());
        }
        if chunk_remaining < chunk_bytes_u64 {
            let finished = hasher.finish().map_err(|error| error.to_string())?;
            digests.push(hex::encode(finished));
        }
        if digests.len() != expected_chunks {
            return Err("Computed chunk count differed from expected chunk count.".into());
        }
        Ok(digests)
    }

    fn compute_chunk_sha256s(
        file: &mut File,
        expected_bytes: u64,
        chunk_bytes: usize,
        workers: usize,
    ) -> crate::Result<Vec<String>> {
        let cancel = AtomicBool::new(false);
        if let Some(raw_digests) =
            compute_mapped_chunk_digests(file, expected_bytes, chunk_bytes, workers, &cancel)
                .map_err(|error| error.to_string())?
        {
            return Ok(raw_digests.into_iter().map(hex::encode).collect());
        }
        compute_chunk_digests_buffered(file, expected_bytes, chunk_bytes)
    }

    fn generate(
        schema_version: Option<u32>,
        path: &Path,
        chunk_bytes: Option<usize>,
        workers: Option<usize>,
        output_path: Option<&Path>,
    ) -> crate::Result<Value> {
        check_schema_version(schema_version)?;
        validate_input_file(path)?;
        let chunk_bytes = chunk_bytes.unwrap_or(PARALLEL_HASH_DEFAULT_CHUNK_BYTES);
        let chunk_bytes_u64 = u64::try_from(chunk_bytes).map_err(|error| error.to_string())?;
        if chunk_bytes_u64 < MIN_MAPPED_HASH_BYTES {
            return Err(format!(
                "chunkBytes must be at least {MIN_MAPPED_HASH_BYTES}."
            ));
        }
        let workers = resolve_workers(workers)?;
        if let Some(out) = output_path {
            if !out.is_absolute() {
                return Err("Output path must be expanded and absolute.".into());
            }
            validate_external_path(out).map_err(|error| error.to_string())?;
            if fs::exists(out).map_err(|error| error.to_string())? {
                return Err("Use a new candidate chunk-manifest output file.".into());
            }
            if out.parent().is_none_or(|parent| !parent.is_dir()) {
                return Err("Output parent directory must already exist.".into());
            }
        }

        let mut file = open_locked_file(path).map_err(|error| error.to_string())?;
        let bytes = file.metadata().map_err(|error| error.to_string())?.len();
        if bytes == 0 {
            return Err("Artifact must be non-empty.".into());
        }
        let sha256 = whole_file_sha256_cng(&mut file, bytes)?;
        let chunk_sha256s = compute_chunk_sha256s(&mut file, bytes, chunk_bytes, workers)?;
        let validated = bonsai_cpu_policy::validate_chunk_manifest(
            bytes,
            Some(chunk_bytes),
            Some(chunk_sha256s),
        )?
        .ok_or("Missing generated chunk manifest.")?;
        let (chunk_bytes, chunk_sha256s) = validated;
        let chunk_count = chunk_sha256s.len();

        let manifest = json!({
            "bytes": bytes,
            "sha256": sha256,
            "chunkBytes": chunk_bytes,
            "chunkSha256s": chunk_sha256s,
        });
        if let Some(out) = output_path {
            crate::bounded_input::write_pretty_json_new(out, &manifest)?;
        }

        let mut response = json!({
            "schemaVersion": 1,
            "operation": "generate",
            "path": path,
            "bytes": bytes,
            "sha256": sha256,
            "chunkBytes": chunk_bytes,
            "chunkSha256s": chunk_sha256s,
            "chunkCount": chunk_count,
        });
        if let Some(out) = output_path {
            response["outputPath"] = json!(out);
        }
        Ok(response)
    }

    fn verify(
        schema_version: Option<u32>,
        path: &Path,
        bytes: u64,
        sha256: String,
        chunk_bytes: Option<usize>,
        chunk_sha256s: Option<Vec<String>>,
        workers: Option<usize>,
    ) -> crate::Result<Value> {
        check_schema_version(schema_version)?;
        if bytes == 0 || !crate::identity::is_lower_hex(&sha256, 64) {
            return Err(
                "Chunk manifest verification requires nonzero bytes and a 64-hex sha256.".into(),
            );
        }
        let (chunk_bytes, chunk_sha256s) =
            bonsai_cpu_policy::validate_chunk_manifest(bytes, chunk_bytes, chunk_sha256s)?
                .ok_or("chunk-manifest verify requires both chunkBytes and chunkSha256s.")?;
        validate_input_file(path)?;
        let workers = resolve_workers(workers)?;
        let chunk_count = chunk_sha256s.len();

        let spec = ChunkedFileSpec {
            path: path.to_path_buf(),
            bytes,
            sha256: sha256.clone(),
            chunk_bytes,
            chunk_sha256s,
        };
        let cancel = AtomicBool::new(false);
        let guard = VerifiedFile::verify_chunked_with(&spec, workers, &cancel)
            .map_err(|error| error.to_string())?;
        let mut locked_handle = guard
            .file()
            .ok_or("Verified file guard did not retain a file handle.")?
            .try_clone()
            .map_err(|error| error.to_string())?;
        let actual_whole_sha256 = whole_file_sha256_cng(&mut locked_handle, bytes)?;
        if actual_whole_sha256 != sha256 {
            return Err("Artifact whole-file SHA-256 differs from chunk manifest.".into());
        }
        drop(guard);

        Ok(json!({
            "schemaVersion": 1,
            "operation": "verify",
            "verified": true,
            "path": path,
            "bytes": bytes,
            "sha256": sha256,
            "chunkBytes": chunk_bytes,
            "chunkCount": chunk_count,
        }))
    }

    pub(super) fn run(value: Value) -> crate::Result<Value> {
        if !value.is_object() {
            return Err("Chunk manifest request must be a JSON object.".into());
        }
        let request: Request = serde_json::from_value(value)
            .map_err(|error| crate::cli::bounded_json_error(&error))?;
        match request {
            Request::Generate {
                schema_version,
                path,
                chunk_bytes,
                workers,
                output_path,
            } => generate(
                schema_version,
                &path,
                chunk_bytes,
                workers,
                output_path.as_deref(),
            ),
            Request::Verify {
                schema_version,
                path,
                bytes,
                sha256,
                chunk_bytes,
                chunk_sha256s,
                workers,
            } => verify(
                schema_version,
                &path,
                bytes,
                sha256,
                chunk_bytes,
                chunk_sha256s,
                workers,
            ),
        }
    }
}

#[cfg(all(test, windows))]
mod tests {
    use std::ffi::OsString;
    use std::fs;
    use std::path::Path;

    use bonsai_windows_credentials::{MIN_MAPPED_HASH_BYTES, PARALLEL_HASH_DEFAULT_CHUNK_BYTES};
    use serde_json::{Value, json};

    use super::run;
    use crate::bounded_input::ScratchDir;

    fn synthetic_payload(len: usize) -> Vec<u8> {
        let mut data: Vec<u8> = (0..len)
            .map(|i| u8::try_from((i.wrapping_mul(41) ^ (i >> 5) ^ 0x3a) & 0xff).unwrap_or(0))
            .collect();
        if data.len() >= 4 {
            data[..4].copy_from_slice(b"GGUF");
        }
        data
    }

    #[test]
    fn chunk_manifest_generate_and_verify_roundtrip_multi_chunk_and_default_chunk_size()
    -> crate::Result<()> {
        let scratch = ScratchDir::new("Bonsai-ChunkManifestRoundtrip")?;
        let artifact_path = scratch.join("model.gguf");
        let chunk_bytes =
            usize::try_from(MIN_MAPPED_HASH_BYTES).map_err(|error| error.to_string())?;
        let total_len = 2 * chunk_bytes + 13_579;
        let payload = synthetic_payload(total_len);
        fs::write(&artifact_path, &payload).map_err(|error| error.to_string())?;

        // 1. Explicit 4 MiB chunks -> 3 digests, written to manifest.json.
        let out_path = scratch.join("manifest.json");
        let generated = run(json!({
            "schemaVersion": 1,
            "operation": "generate",
            "path": artifact_path,
            "chunkBytes": chunk_bytes,
            "workers": 4,
            "outputPath": out_path,
        }))?;
        assert_eq!(generated["operation"], "generate");
        assert_eq!(generated["bytes"], u64::try_from(total_len).unwrap_or(0));
        assert_eq!(generated["chunkBytes"], chunk_bytes);
        assert_eq!(generated["chunkCount"], 3);
        let digests = generated["chunkSha256s"]
            .as_array()
            .ok_or("missing chunkSha256s")?;
        assert_eq!(digests.len(), 3);
        assert_eq!(generated["outputPath"], json!(out_path));

        let saved: Value =
            serde_json::from_slice(&fs::read(&out_path).map_err(|error| error.to_string())?)
                .map_err(|error| error.to_string())?;
        assert_eq!(saved["bytes"], generated["bytes"]);
        assert_eq!(saved["sha256"], generated["sha256"]);
        assert_eq!(saved["chunkBytes"], generated["chunkBytes"]);
        assert_eq!(saved["chunkSha256s"], generated["chunkSha256s"]);

        // Existing outputPath must not be overwritten.
        assert!(
            run(json!({
                "operation": "generate",
                "path": artifact_path,
                "chunkBytes": chunk_bytes,
                "outputPath": out_path,
            }))
            .is_err()
        );

        // Verify roundtrip succeeds.
        let verified = run(json!({
            "schemaVersion": 1,
            "operation": "verify",
            "path": artifact_path,
            "bytes": saved["bytes"],
            "sha256": saved["sha256"],
            "chunkBytes": saved["chunkBytes"],
            "chunkSha256s": saved["chunkSha256s"],
        }))?;
        assert_eq!(verified["verified"], true);
        assert_eq!(verified["chunkCount"], 3);

        // Verify CLI JSON request path (`--json`) parses cleanly for `chunk-manifest`.
        let req_text = serde_json::to_string(&json!({
            "operation": "verify",
            "path": artifact_path,
            "bytes": saved["bytes"],
            "sha256": saved["sha256"],
            "chunkBytes": saved["chunkBytes"],
            "chunkSha256s": saved["chunkSha256s"],
        }))
        .map_err(|error| error.to_string())?;
        let parsed_cli = crate::cli::request(
            "chunk-manifest",
            &[OsString::from("--json"), OsString::from(req_text)],
        )?;
        assert_eq!(run(parsed_cli)?["verified"], true);

        // 2. Default chunk size (16 MiB) on an 8 MiB file -> 1 chunk digest matching whole sha256.
        let default_gen = run(json!({
            "operation": "generate",
            "path": artifact_path,
        }))?;
        assert_eq!(default_gen["chunkBytes"], PARALLEL_HASH_DEFAULT_CHUNK_BYTES);
        assert_eq!(default_gen["chunkCount"], 1);
        assert_eq!(default_gen["chunkSha256s"][0], default_gen["sha256"]);
        assert_eq!(
            run(json!({
                "operation": "verify",
                "path": artifact_path,
                "bytes": default_gen["bytes"],
                "sha256": default_gen["sha256"],
                "chunkBytes": default_gen["chunkBytes"],
                "chunkSha256s": default_gen["chunkSha256s"],
            }))?["verified"],
            true
        );

        // 3. Small file (< MIN_MAPPED_HASH_BYTES) uses buffered CNG fallback cleanly.
        let small_path = scratch.join("small.gguf");
        fs::write(&small_path, b"GGUF-small-payload").map_err(|error| error.to_string())?;
        let small_gen = run(json!({
            "operation": "generate",
            "path": small_path,
        }))?;
        assert_eq!(small_gen["bytes"], 18);
        assert_eq!(small_gen["chunkCount"], 1);
        assert_eq!(small_gen["chunkSha256s"][0], small_gen["sha256"]);
        assert_eq!(
            run(json!({
                "operation": "verify",
                "path": small_path,
                "bytes": small_gen["bytes"],
                "sha256": small_gen["sha256"],
                "chunkBytes": small_gen["chunkBytes"],
                "chunkSha256s": small_gen["chunkSha256s"],
            }))?["verified"],
            true
        );
        Ok(())
    }

    #[test]
    fn chunk_manifest_rejects_corruption_self_inconsistency_and_invalid_schemas()
    -> crate::Result<()> {
        let scratch = ScratchDir::new("Bonsai-ChunkManifestReject")?;
        let artifact_path = scratch.join("model.gguf");
        let chunk_bytes =
            usize::try_from(MIN_MAPPED_HASH_BYTES).map_err(|error| error.to_string())?;
        let total_len = chunk_bytes + 4_096;
        let payload = synthetic_payload(total_len);
        fs::write(&artifact_path, &payload).map_err(|error| error.to_string())?;

        let generated = run(json!({
            "operation": "generate",
            "path": artifact_path,
            "chunkBytes": chunk_bytes,
        }))?;
        let valid_verify = json!({
            "operation": "verify",
            "path": artifact_path,
            "bytes": generated["bytes"],
            "sha256": generated["sha256"],
            "chunkBytes": generated["chunkBytes"],
            "chunkSha256s": generated["chunkSha256s"],
        });

        // Whole-file sha256 mismatch even when chunkSha256s match must fail closed.
        let mut bad_whole_sha = valid_verify.clone();
        bad_whole_sha["sha256"] = json!("0".repeat(64));
        assert!(run(bad_whole_sha).is_err());

        // Single chunk digest mismatch must fail closed.
        let mut bad_chunk_digest = valid_verify.clone();
        bad_chunk_digest["chunkSha256s"][1] = json!("f".repeat(64));
        assert!(run(bad_chunk_digest).is_err());

        // Wrong byte length, sub-threshold chunkBytes, wrong chunk count, missing fields,
        // uppercase hex, zero workers, and unknown fields must fail closed.
        let mut wrong_bytes = valid_verify.clone();
        wrong_bytes["bytes"] = json!(total_len - 1);
        assert!(run(wrong_bytes).is_err());

        let mut small_chunk = valid_verify.clone();
        small_chunk["chunkBytes"] = json!(chunk_bytes - 1);
        assert!(run(small_chunk).is_err());

        let mut short_chunks = valid_verify.clone();
        short_chunks["chunkSha256s"] = json!([generated["chunkSha256s"][0]]);
        assert!(run(short_chunks).is_err());

        let mut missing_chunk_sha = valid_verify.clone();
        missing_chunk_sha
            .as_object_mut()
            .ok_or("object")?
            .remove("chunkSha256s");
        assert!(run(missing_chunk_sha).is_err());

        let mut missing_both = valid_verify.clone();
        let obj = missing_both.as_object_mut().ok_or("object")?;
        obj.remove("chunkBytes");
        obj.remove("chunkSha256s");
        assert!(run(missing_both).is_err());

        let mut bad_hex = valid_verify.clone();
        bad_hex["chunkSha256s"][0] = json!("A".repeat(64));
        assert!(run(bad_hex).is_err());

        let mut zero_workers = valid_verify.clone();
        zero_workers["workers"] = json!(0);
        assert!(run(zero_workers).is_err());

        let mut extra = valid_verify.clone();
        extra["unexpected"] = json!(true);
        assert!(run(extra).is_err());

        // 1-byte file tamper in second chunk must fail verification.
        let mut tampered = payload;
        tampered[chunk_bytes + 99] ^= 0x01;
        fs::write(&artifact_path, &tampered).map_err(|error| error.to_string())?;
        assert!(run(valid_verify).is_err());

        // Empty file, relative path, and .git ancestor must fail closed.
        let empty = scratch.join("empty.gguf");
        fs::write(&empty, b"").map_err(|error| error.to_string())?;
        assert!(run(json!({"operation": "generate", "path": empty})).is_err());
        assert!(run(json!({"operation": "generate", "path": Path::new("relative.gguf")})).is_err());

        let git_dir = scratch.join("repo");
        fs::create_dir_all(git_dir.join(".git")).map_err(|error| error.to_string())?;
        let git_file = git_dir.join("model.gguf");
        fs::write(&git_file, b"GGUF").map_err(|error| error.to_string())?;
        assert!(run(json!({"operation": "generate", "path": git_file})).is_err());
        Ok(())
    }
}
