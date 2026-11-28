//! Fixed native companion CLI and retained compiled-artifact verification.

use crate::{Kind, credentials::Credential};
use bonsai_artifacts::{ChunkedFileSpec, VerifiedFile};
use std::{
    ffi::OsString,
    fs, io,
    path::{Path, PathBuf},
    sync::atomic::AtomicBool,
};

/// A supported companion command, with no arbitrary runtime or model overrides.
pub enum Command {
    /// Report binary and protocol identity without touching model files.
    Version,
    /// Verify the compiled model pins and serve this explicit loopback endpoint.
    Serve(ServiceArgs),
}

/// Existing local files and the fixed companion port; credential values are private.
pub struct ServiceArgs {
    /// Absolute external path of the pinned ASR or talker GGUF.
    pub model: PathBuf,
    /// Absolute external codec GGUF; required for TTS and forbidden for ASR.
    pub codec: Option<PathBuf>,
    /// Absolute pinned Base talker GGUF used to preserve a designed speaker.
    pub voice_model: Option<PathBuf>,
    /// Absolute external protected 64-byte lowercase hexadecimal credential.
    pub token_file: PathBuf,
    /// Unprivileged numeric-loopback port, defaulting to 8766 or 8767.
    pub port: u16,
    /// Optional TTS compute team size, bounded to 1 through 256.
    pub threads: Option<usize>,
    /// Explicit pinned talker quantization; ASR always uses the default value.
    pub quantization: TtsQuantization,
}

/// Reviewed TTS talker artifact selection, independent of checkpoint identity.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum TtsQuantization {
    /// Default eight-bit talker weights.
    #[default]
    Q8,
    /// Opt-in four-bit talker evaluation weights.
    Q4KM,
}

/// Read guards for all verified model files and the startup credential.
///
/// Keep this owner in the engine adapter until serving has stopped. Native model
/// loaders may reopen or map paths; these guards deny writes, renames and mutable
/// ancestor replacement for their complete lifetime on Windows.
pub struct ModelGuards {
    _models: Vec<VerifiedFile>,
    _credential: Credential,
}

/// Require a native ARM64 process on Windows before starting model work.
///
/// # Errors
/// Rejects every build target other than native Windows ARM64.
pub fn require_native_host() -> io::Result<()> {
    if !cfg!(windows) || std::env::consts::ARCH != "aarch64" {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "Native speech requires Windows ARM64.",
        ));
    }
    Ok(())
}

/// Parse the exact fixed companion command line without opening model contents.
///
/// `--version` must appear alone. ASR accepts `--model`, `--token-file`, optional
/// `--port` and bounded `--threads`; TTS additionally requires `--codec` and
/// `--voice-model`, and accepts bounded `--threads` and pinned `--quantization Q8_0|Q4_K_M`.
/// Every supplied path must already name an external regular non-reparse file.
///
/// # Errors
/// Rejects missing, duplicate or unknown options; unsafe paths; unsupported codec
/// combinations; and ports outside 1024 through 65535. Errors contain no values.
pub fn parse(kind: Kind, arguments: impl IntoIterator<Item = OsString>) -> io::Result<Command> {
    let mut arguments = arguments.into_iter().peekable();
    if arguments.peek().is_some_and(|arg| arg == "--version") {
        let _ = arguments.next();
        if arguments.next().is_none() {
            return Ok(Command::Version);
        }
        return Err(invalid_arguments());
    }
    let (mut model, mut codec, mut token_file, mut port) = (None, None, None, None);
    let mut voice_model = None;
    let (mut threads, mut quantization) = (None, None);
    while let Some(option) = arguments.next() {
        match option.to_str() {
            Some("--model") if model.is_none() => {
                model = Some(PathBuf::from(
                    arguments.next().ok_or_else(invalid_arguments)?,
                ));
            }
            Some("--codec") if kind == Kind::QwenTts && codec.is_none() => {
                codec = Some(PathBuf::from(
                    arguments.next().ok_or_else(invalid_arguments)?,
                ));
            }
            Some("--voice-model") if kind == Kind::QwenTts && voice_model.is_none() => {
                voice_model = Some(PathBuf::from(
                    arguments.next().ok_or_else(invalid_arguments)?,
                ));
            }
            Some("--token-file") if token_file.is_none() => {
                token_file = Some(PathBuf::from(
                    arguments.next().ok_or_else(invalid_arguments)?,
                ));
            }
            Some("--port") if port.is_none() => {
                port = Some(parse_decimal_arg::<u16>(&mut arguments)?);
            }
            Some("--threads")
                if matches!(kind, Kind::Canary | Kind::QwenTts) && threads.is_none() =>
            {
                let count = parse_decimal_arg::<usize>(&mut arguments)?;
                if !(1..=256).contains(&count) {
                    return Err(invalid_arguments());
                }
                threads = Some(count);
            }
            Some("--quantization") if kind == Kind::QwenTts && quantization.is_none() => {
                quantization = Some(
                    match arguments.next().as_deref().and_then(|value| value.to_str()) {
                        Some("Q8_0") => TtsQuantization::Q8,
                        Some("Q4_K_M") => TtsQuantization::Q4KM,
                        _ => return Err(invalid_arguments()),
                    },
                );
            }
            _ => return Err(invalid_arguments()),
        }
    }
    if kind == Kind::QwenTts && (codec.is_none() || voice_model.is_none()) {
        return Err(invalid_arguments());
    }
    let arguments = ServiceArgs {
        model: model.ok_or_else(invalid_arguments)?,
        codec,
        voice_model,
        token_file: token_file.ok_or_else(invalid_arguments)?,
        port: port.unwrap_or(match kind {
            Kind::Canary => 8766,
            Kind::QwenTts => 8767,
        }),
        threads,
        quantization: quantization.unwrap_or_default(),
    };
    if arguments.port < 1024 {
        return Err(invalid_arguments());
    }
    for path in [&arguments.model, &arguments.token_file]
        .into_iter()
        .chain(arguments.codec.as_ref())
        .chain(arguments.voice_model.as_ref())
    {
        validate_file(path)?;
    }
    Ok(Command::Serve(arguments))
}

/// Exact no-model-load identity receipt for PowerShell binary provisioning.
#[must_use]
pub fn version(kind: Kind) -> serde_json::Value {
    serde_json::json!({
        "Name": match kind { Kind::Canary => "bonsai-speech-asr", Kind::QwenTts => "bonsai-speech-tts" },
        "Version": env!("CARGO_PKG_VERSION"), "Architecture": std::env::consts::ARCH,
        "Model": kind.model(), "ProtocolVersion": 1,
    })
}

/// Stage at which staged TTS verification and model loading failed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TtsStartupStage {
    /// Failed while checking credential or SHA-256 model pins.
    Verify,
    /// Failed while constructing the Rust codec or design talker.
    Load,
}

/// Failure from [`verify_and_load_tts`], attributing verification vs. model loading.
#[derive(Debug)]
pub struct TtsStartupError {
    /// Startup stage where the error occurred.
    pub stage: TtsStartupStage,
    /// Underlying value-free I/O error.
    pub error: io::Error,
}

#[cfg(not(windows))]
const MIN_CHUNK_MANIFEST_BYTES: u64 = 4 * 1024 * 1024;
const MAX_CHUNK_MANIFEST_COUNT: usize = 16_384;

struct ArtifactPin<'a> {
    path: &'a Path,
    bytes: u64,
    sha256: &'a str,
    chunks: Option<(usize, Vec<String>)>,
}

fn hash_worker_count() -> usize {
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

/// Verify only the compiled checkpoint identities and retain their file guards.
///
/// `compiled_lock` must come from `include_str!` in the companion binary. No
/// command-line hash, model identity, repository or lock-file override is accepted.
/// The guarded credential is checked before loading or hashing large artifacts.
///
/// # Errors
/// Rejects malformed pins, the wrong checkpoint family, unsafe input paths, bad
/// credentials and any byte-length or SHA-256 mismatch. Errors are value-free.
pub fn verify_models(
    kind: Kind,
    arguments: &ServiceArgs,
    compiled_lock: &str,
) -> io::Result<ModelGuards> {
    let credential = Credential::open(&arguments.token_file).map_err(|_| invalid_models())?;
    let pin = parse_lock_document(kind, compiled_lock)?;
    let workers = hash_worker_count();
    let models = match kind {
        Kind::Canary => {
            if arguments.codec.is_some() || arguments.voice_model.is_some() {
                return Err(invalid_models());
            }
            let model_pin = parse_artifact_pin(&arguments.model, &pin)?;
            vec![verify_pinned_artifact(&model_pin, workers)?]
        }
        Kind::QwenTts => {
            let [talker_pin, codec_pin, voice_pin] = parse_tts_pins(arguments, &pin, |err| err)?;
            let has_chunks = talker_pin.chunks.is_some()
                && codec_pin.chunks.is_some()
                && voice_pin.chunks.is_some();
            if has_chunks {
                // Verify chunked artifacts sequentially with full worker parallelism to avoid
                // oversubscribing CPU cores and memory bandwidth across files.
                vec![
                    verify_pinned_artifact(&talker_pin, workers)?,
                    verify_pinned_artifact(&codec_pin, workers)?,
                    verify_pinned_artifact(&voice_pin, workers)?,
                ]
            } else {
                let (talker_res, codec_res, voice_res) = std::thread::scope(|scope| {
                    let codec_job = spawn_scoped(scope, "speech-verify-codec", || {
                        verify_pinned_artifact(&codec_pin, workers)
                    });
                    let voice_job = spawn_scoped(scope, "speech-verify-voice", || {
                        verify_pinned_artifact(&voice_pin, workers)
                    });
                    let talker_res = verify_pinned_artifact(&talker_pin, workers);
                    let codec_res = join_scoped(
                        codec_job,
                        || Err(invalid_models()),
                        || verify_pinned_artifact(&codec_pin, workers),
                    );
                    let voice_res = join_scoped(
                        voice_job,
                        || Err(invalid_models()),
                        || verify_pinned_artifact(&voice_pin, workers),
                    );
                    (talker_res, codec_res, voice_res)
                });
                vec![talker_res?, codec_res?, voice_res?]
            }
        }
    };
    Ok(ModelGuards {
        _models: models,
        _credential: credential,
    })
}

/// Verify compiled TTS checkpoint pins and pipeline verified model loading safely.
///
/// No loader closure is invoked before its corresponding artifact completes full SHA-256
/// verification and lock-guard retention. Once `codec` passes verification, `load_codec`
/// runs concurrently with `talker` verification; once `talker` passes verification,
/// `load_design` runs concurrently with `voice_talker` verification.
///
/// # Errors
/// Attributes credential/verification failures to [`TtsStartupStage::Verify`] and loader
/// failures to [`TtsStartupStage::Load`].
pub fn verify_and_load_tts<C, T, Fc, Ft>(
    arguments: &ServiceArgs,
    compiled_lock: &str,
    load_codec: Fc,
    load_design: Ft,
) -> Result<(ModelGuards, C, T), TtsStartupError>
where
    C: Send,
    T: Send,
    Fc: FnOnce(&Path) -> io::Result<C> + Send,
    Ft: FnOnce(&Path) -> io::Result<T> + Send,
{
    let verify_err = |error| TtsStartupError {
        stage: TtsStartupStage::Verify,
        error,
    };
    let load_err = |error| TtsStartupError {
        stage: TtsStartupStage::Load,
        error,
    };
    let credential =
        Credential::open(&arguments.token_file).map_err(|_| verify_err(invalid_models()))?;
    let pin = parse_lock_document(Kind::QwenTts, compiled_lock).map_err(verify_err)?;
    let [talker_pin, codec_pin, voice_pin] = parse_tts_pins(arguments, &pin, verify_err)?;

    let full_workers = hash_worker_count();
    let overlap_workers = (full_workers / 2).max(4).min(full_workers);
    let codec_guard = verify_pinned_artifact(&codec_pin, full_workers).map_err(verify_err)?;

    let (codec_res, talker_guard_res, voice_guard_res, design_res) = std::thread::scope(|scope| {
        let verified_codec_path = codec_guard.path();
        let codec_job = spawn_scoped(scope, "tts-load-codec", || load_codec(verified_codec_path));

        let talker_guard_res = verify_pinned_artifact(&talker_pin, full_workers);
        let (voice_guard_res, design_res) = if talker_guard_res.is_ok() {
            let verified_design_path = talker_pin.path;
            let design_job = spawn_scoped(scope, "tts-load-design", || {
                load_design(verified_design_path)
            });
            let voice_guard_res = verify_pinned_artifact(&voice_pin, overlap_workers);
            let design_res = join_scoped(
                design_job,
                || Err(io::Error::other("TTS design loader panicked.")),
                || Err(io::Error::other("Could not spawn TTS design loader.")),
            );
            (voice_guard_res, Some(design_res))
        } else {
            (Err(invalid_models()), None)
        };

        let codec_res = join_scoped(
            codec_job,
            || Err(io::Error::other("TTS codec loader panicked.")),
            || Err(io::Error::other("Could not spawn TTS codec loader.")),
        );

        (codec_res, talker_guard_res, voice_guard_res, design_res)
    });

    let talker_guard = talker_guard_res.map_err(verify_err)?;
    let voice_guard = voice_guard_res.map_err(verify_err)?;
    let codec = codec_res.map_err(load_err)?;
    let design = design_res
        .unwrap_or_else(|| Err(invalid_models()))
        .map_err(load_err)?;

    Ok((
        ModelGuards {
            _models: vec![talker_guard, codec_guard, voice_guard],
            _credential: credential,
        },
        codec,
        design,
    ))
}

fn parse_decimal_arg<T: std::str::FromStr>(
    arguments: &mut impl Iterator<Item = OsString>,
) -> io::Result<T> {
    let value = arguments.next().ok_or_else(invalid_arguments)?;
    let text = value.to_str().ok_or_else(invalid_arguments)?;
    if text.is_empty() || !text.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(invalid_arguments());
    }
    text.parse::<T>().map_err(|_| invalid_arguments())
}

fn parse_tts_pins<'a, E>(
    arguments: &'a ServiceArgs,
    pin: &'a serde_json::Value,
    map_err: impl Fn(io::Error) -> E,
) -> Result<[ArtifactPin<'a>; 3], E> {
    let files = pin.get("files").ok_or_else(|| map_err(invalid_models()))?;
    let codec = arguments
        .codec
        .as_ref()
        .ok_or_else(|| map_err(invalid_models()))?;
    let voice_model = arguments
        .voice_model
        .as_ref()
        .ok_or_else(|| map_err(invalid_models()))?;
    let pin_for = |path: &'a Path, key: &str| {
        let entry = files.get(key).ok_or_else(|| map_err(invalid_models()))?;
        parse_artifact_pin(path, entry).map_err(&map_err)
    };
    Ok([
        pin_for(&arguments.model, "talker")?,
        pin_for(codec, "codec")?,
        pin_for(voice_model, "voice_talker")?,
    ])
}

fn spawn_scoped<'scope, 'env, T: Send + 'scope>(
    scope: &'scope std::thread::Scope<'scope, 'env>,
    name: &str,
    work: impl FnOnce() -> T + Send + 'scope,
) -> io::Result<std::thread::ScopedJoinHandle<'scope, T>> {
    std::thread::Builder::new()
        .name(name.to_owned())
        .spawn_scoped(scope, work)
}

fn join_scoped<T>(
    handle: io::Result<std::thread::ScopedJoinHandle<'_, io::Result<T>>>,
    on_panic: impl FnOnce() -> io::Result<T>,
    on_spawn_err: impl FnOnce() -> io::Result<T>,
) -> io::Result<T> {
    match handle {
        Ok(handle) => handle.join().unwrap_or_else(|_| on_panic()),
        Err(_) => on_spawn_err(),
    }
}

fn parse_lock_document(kind: Kind, compiled_lock: &str) -> io::Result<serde_json::Value> {
    let pin: serde_json::Value =
        serde_json::from_str(compiled_lock).map_err(|_| invalid_models())?;
    if pin.get("schemaVersion").and_then(serde_json::Value::as_u64) != Some(1)
        || pin.get("model").and_then(serde_json::Value::as_str) != Some(kind.model())
    {
        return Err(invalid_models());
    }
    Ok(pin)
}

fn verify_pinned_artifact(pin: &ArtifactPin<'_>, workers: usize) -> io::Result<VerifiedFile> {
    if let Some((chunk_bytes, chunk_sha256s)) = &pin.chunks {
        let spec = ChunkedFileSpec {
            path: pin.path.to_path_buf(),
            bytes: pin.bytes,
            sha256: pin.sha256.to_owned(),
            chunk_bytes: *chunk_bytes,
            chunk_sha256s: chunk_sha256s.clone(),
        };
        let cancel = AtomicBool::new(false);
        VerifiedFile::verify_chunked_with(&spec, workers, &cancel).map_err(|_| invalid_models())
    } else {
        VerifiedFile::verify(pin.path, pin.bytes, pin.sha256).map_err(|_| invalid_models())
    }
}

fn is_lower_sha256_hex(hash: &str) -> bool {
    hash.len() == 64
        && hash
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn parse_artifact_pin<'a>(
    path: &'a Path,
    pin: &'a serde_json::Value,
) -> io::Result<ArtifactPin<'a>> {
    validate_file(path)?;
    let bytes = pin
        .get("bytes")
        .and_then(serde_json::Value::as_u64)
        .filter(|bytes| *bytes > 0)
        .ok_or_else(invalid_models)?;
    let hash = pin
        .get("sha256")
        .and_then(serde_json::Value::as_str)
        .filter(|hash| is_lower_sha256_hex(hash))
        .ok_or_else(invalid_models)?;
    let chunks = match (pin.get("chunkBytes"), pin.get("chunkSha256s")) {
        (None, None) => None,
        (Some(chunk_bytes_val), Some(chunk_hashes_val)) => {
            #[cfg(windows)]
            let min_chunk_bytes = bonsai_windows_credentials::MIN_MAPPED_HASH_BYTES;
            #[cfg(not(windows))]
            let min_chunk_bytes = MIN_CHUNK_MANIFEST_BYTES;
            let chunk_bytes_u64 = chunk_bytes_val
                .as_u64()
                .filter(|size| *size >= min_chunk_bytes)
                .ok_or_else(invalid_models)?;
            let chunk_bytes = usize::try_from(chunk_bytes_u64).map_err(|_| invalid_models())?;
            let expected_chunks = usize::try_from(bytes.div_ceil(chunk_bytes_u64))
                .ok()
                .filter(|count| *count > 0 && *count <= MAX_CHUNK_MANIFEST_COUNT)
                .ok_or_else(invalid_models)?;
            let array = chunk_hashes_val.as_array().ok_or_else(invalid_models)?;
            if array.len() != expected_chunks {
                return Err(invalid_models());
            }
            let mut digests = Vec::with_capacity(expected_chunks);
            for item in array {
                let digest = item
                    .as_str()
                    .filter(|hex_str| is_lower_sha256_hex(hex_str))
                    .ok_or_else(invalid_models)?;
                digests.push(digest.to_owned());
            }
            Some((chunk_bytes, digests))
        }
        _ => return Err(invalid_models()),
    };
    Ok(ArtifactPin {
        path,
        bytes,
        sha256: hash,
        chunks,
    })
}

fn validate_file(path: &Path) -> io::Result<()> {
    #[cfg(windows)]
    bonsai_windows_credentials::validate_read_path(path).map_err(|_| invalid_arguments())?;
    if !path.is_absolute() {
        return Err(invalid_arguments());
    }
    let metadata = fs::symlink_metadata(path).map_err(|_| invalid_arguments())?;
    if !metadata.is_file() || metadata.file_type().is_symlink() {
        return Err(invalid_arguments());
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt as _;
        if metadata.file_attributes() & 0x400 != 0 {
            return Err(invalid_arguments());
        }
    }
    Ok(())
}

fn invalid_arguments() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidInput,
        "Use the fixed native speech options and existing absolute external files.",
    )
}
fn invalid_models() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        "The compiled speech model artifacts or credential could not be verified.",
    )
}

#[cfg(all(test, windows))]
mod tests {
    use super::{Command, ServiceArgs, TtsStartupStage, parse, verify_and_load_tts, verify_models};
    use crate::Kind;
    use bonsai_windows_credentials::{MIN_MAPPED_HASH_BYTES, Sha256};
    use std::{
        ffi::OsString,
        fs, io,
        path::{Path, PathBuf},
        sync::atomic::{AtomicBool, Ordering},
        time::{SystemTime, UNIX_EPOCH},
    };

    fn fixture() -> io::Result<(PathBuf, Vec<OsString>)> {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(io::Error::other)?
            .as_nanos();
        let directory = std::env::temp_dir().join(format!("Bonsai-NativeSpeechCli-{stamp}"));
        fs::create_dir(&directory)?;
        let model = directory.join("model.gguf");
        let key = directory.join("token.key");
        fs::write(&model, b"abc")?;
        fs::write(&key, [b'a'; 64])?;
        let values = vec![
            OsString::from("--model"),
            model.into_os_string(),
            OsString::from("--token-file"),
            key.into_os_string(),
        ];
        Ok((directory, values))
    }

    #[test]
    fn native_cli_requires_fixed_complete_paths_and_rejects_duplicate_overrides() -> io::Result<()>
    {
        let (directory, values) = fixture()?;
        let Command::Serve(arguments) = parse(Kind::Canary, values.clone())? else {
            return Err(io::Error::other("Missing serve arguments"));
        };
        assert_eq!(arguments.port, 8766);
        assert!(arguments.codec.is_none());
        assert!(arguments.voice_model.is_none());
        for extra in [
            vec!["--model", "C:\\replacement"],
            vec!["--hash", "override"],
            vec!["--port", "1023"],
            vec!["--port", "+8766"],
            vec!["--version"],
            vec!["--codec", "C:\\codec"],
            vec!["--voice-model", "C:\\voice"],
        ] {
            let mut invalid = values.clone();
            invalid.extend(extra.into_iter().map(OsString::from));
            assert!(parse(Kind::Canary, invalid).is_err());
        }
        assert!(parse(Kind::QwenTts, values.clone()).is_err());
        let codec = directory.join("codec.gguf");
        fs::write(&codec, b"abc")?;
        let mut tts = values;
        tts.extend([OsString::from("--codec"), codec.clone().into_os_string()]);
        assert!(parse(Kind::QwenTts, tts.clone()).is_err());
        tts.extend([OsString::from("--voice-model"), codec.into_os_string()]);
        assert!(
            matches!(parse(Kind::QwenTts, tts)?, Command::Serve(arguments) if arguments.port == 8767 && arguments.codec.is_some() && arguments.voice_model.is_some())
        );
        assert!(matches!(
            parse(Kind::QwenTts, [OsString::from("--version")])?,
            Command::Version
        ));
        Ok(())
    }

    #[test]
    fn compiled_artifact_guards_reject_changed_bytes_and_wrong_model_identity() -> io::Result<()> {
        let (_, values) = fixture()?;
        let Command::Serve(arguments) = parse(Kind::Canary, values)? else {
            return Err(io::Error::other("Missing serve arguments"));
        };
        let pin = r#"{"schemaVersion":1,"model":"Qwen/Qwen3-ASR-1.7B","bytes":3,"sha256":"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"}"#;
        let unsupported = pin.replace("Qwen/Qwen3-ASR-1.7B", "unsupported/recognizer");
        assert!(verify_models(Kind::Canary, &arguments, &unsupported).is_err());
        let guard = verify_models(Kind::Canary, &arguments, pin)?;
        assert!(
            fs::OpenOptions::new()
                .write(true)
                .open(&arguments.model)
                .is_err()
        );
        drop(guard);
        fs::write(&arguments.model, b"abd")?;
        assert!(verify_models(Kind::Canary, &arguments, pin).is_err());
        assert!(verify_models(Kind::QwenTts, &arguments, pin).is_err());
        let wrong = ServiceArgs {
            model: arguments.model,
            token_file: arguments.token_file,
            codec: None,
            voice_model: None,
            port: 8767,
            threads: None,
            quantization: super::TtsQuantization::default(),
        };
        assert!(verify_models(Kind::QwenTts, &wrong, pin).is_err());
        Ok(())
    }

    #[test]
    fn tts_guards_verify_and_retain_the_base_voice_model() -> io::Result<()> {
        let (directory, mut values) = fixture()?;
        let codec = directory.join("codec.gguf");
        let voice = directory.join("voice.gguf");
        fs::write(&codec, b"abc")?;
        fs::write(&voice, b"abc")?;
        values.extend([
            OsString::from("--codec"),
            codec.into_os_string(),
            OsString::from("--voice-model"),
            voice.clone().into_os_string(),
        ]);
        let Command::Serve(mut arguments) = parse(Kind::QwenTts, values.clone())? else {
            return Err(io::Error::other("Missing TTS serve arguments"));
        };
        let artifact = serde_json::json!({
            "bytes": 3,
            "sha256": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        });
        let pin = serde_json::json!({
            "schemaVersion": 1, "model": Kind::QwenTts.model(),
            "files": {"talker": artifact, "codec": artifact, "voice_talker": artifact},
        });
        let guard = verify_models(Kind::QwenTts, &arguments, &pin.to_string())?;
        assert!(fs::OpenOptions::new().write(true).open(&voice).is_err());
        drop(guard);
        fs::write(&voice, b"abd")?;
        assert!(verify_models(Kind::QwenTts, &arguments, &pin.to_string()).is_err());
        arguments.voice_model = None;
        assert!(verify_models(Kind::QwenTts, &arguments, &pin.to_string()).is_err());
        let mut duplicate = values.clone();
        duplicate.extend([OsString::from("--voice-model"), voice.into_os_string()]);
        assert!(parse(Kind::QwenTts, duplicate).is_err());
        let mut relative = values;
        let last = relative
            .last_mut()
            .ok_or_else(|| io::Error::other("Missing voice path"))?;
        *last = OsString::from("relative.gguf");
        assert!(parse(Kind::QwenTts, relative).is_err());
        Ok(())
    }

    #[test]
    fn native_cli_rejects_model_reparse_files() -> io::Result<()> {
        let (directory, values) = fixture()?;
        let link = directory.join("redirect.gguf");
        std::os::windows::fs::symlink_file(directory.join("model.gguf"), &link)?;
        let mut invalid = values;
        invalid[1] = link.into_os_string();
        assert!(parse(Kind::Canary, invalid).is_err());
        Ok(())
    }

    #[test]
    fn tts_cli_bounds_thread_counts_and_pinned_quantization() -> io::Result<()> {
        let (directory, mut values) = fixture()?;
        let codec = directory.join("codec.gguf");
        fs::write(&codec, b"abc")?;
        values.extend([
            OsString::from("--codec"),
            codec.clone().into_os_string(),
            OsString::from("--voice-model"),
            codec.into_os_string(),
        ]);
        let mut tuned = values.clone();
        tuned.extend(["--threads", "10", "--quantization", "Q4_K_M"].map(OsString::from));
        let Command::Serve(arguments) = parse(Kind::QwenTts, tuned)? else {
            return Err(io::Error::other("Expected TTS settings."));
        };
        assert_eq!(arguments.threads, Some(10));
        assert_eq!(arguments.quantization, super::TtsQuantization::Q4KM);
        for extra in [
            vec!["--threads", "0"],
            vec!["--threads", "257"],
            vec!["--threads", "+2"],
            vec!["--threads", "2", "--threads", "3"],
            vec!["--quantization", "unknown"],
            vec!["--quantization", "Q8_0", "--quantization", "Q4_K_M"],
        ] {
            let mut invalid = values.clone();
            invalid.extend(extra.into_iter().map(OsString::from));
            assert!(parse(Kind::QwenTts, invalid).is_err());
        }
        Ok(())
    }

    #[test]
    fn asr_cli_accepts_bounded_threads_and_rejects_invalid_thread_counts() -> io::Result<()> {
        let (_, values) = fixture()?;
        let mut tuned = values.clone();
        tuned.extend(["--threads", "8"].map(OsString::from));
        let Command::Serve(arguments) = parse(Kind::Canary, tuned)? else {
            return Err(io::Error::other("Expected ASR settings."));
        };
        assert_eq!(arguments.threads, Some(8));
        for extra in [
            vec!["--threads", "0"],
            vec!["--threads", "257"],
            vec!["--threads", "+8"],
            vec!["--threads", "4", "--threads", "8"],
        ] {
            let mut invalid = values.clone();
            invalid.extend(extra.into_iter().map(OsString::from));
            assert!(parse(Kind::Canary, invalid).is_err());
        }
        Ok(())
    }

    fn hex_lower(bytes: &[u8; 32]) -> String {
        let mut out = String::with_capacity(64);
        for byte in bytes {
            out.push_str(&format!("{byte:02x}"));
        }
        out
    }

    fn write_chunked_fixture_file(
        path: &Path,
        payload: &[u8],
        chunk_bytes: usize,
    ) -> io::Result<serde_json::Value> {
        fs::write(path, payload)?;
        let whole = hex_lower(&Sha256::digest_bytes(payload)?);
        let mut chunks = Vec::new();
        for chunk in payload.chunks(chunk_bytes) {
            chunks.push(hex_lower(&Sha256::digest_bytes(chunk)?));
        }
        Ok(serde_json::json!({
            "bytes": payload.len(),
            "sha256": whole,
            "chunkBytes": chunk_bytes,
            "chunkSha256s": chunks,
        }))
    }

    #[test]
    fn chunk_manifest_verification_accepts_valid_and_rejects_malformed_or_tampered()
    -> io::Result<()> {
        let (directory, values) = fixture()?;
        let Command::Serve(arguments) = parse(Kind::Canary, values)? else {
            return Err(io::Error::other("Missing serve arguments"));
        };
        let chunk_bytes = usize::try_from(MIN_MAPPED_HASH_BYTES).map_err(io::Error::other)?;
        let len = chunk_bytes + 4_096;
        let tile: Vec<u8> = (0..256_usize)
            .map(|i| u8::try_from((i.wrapping_mul(31) ^ 0xa5) & 0xff).unwrap_or(0))
            .collect();
        let mut payload = vec![0_u8; len];
        for chunk in payload.chunks_mut(tile.len()) {
            chunk.copy_from_slice(&tile[..chunk.len()]);
        }
        let entry = write_chunked_fixture_file(&arguments.model, &payload, chunk_bytes)?;
        let valid_pin = serde_json::json!({
            "schemaVersion": 1,
            "model": Kind::Canary.model(),
            "bytes": entry["bytes"],
            "sha256": entry["sha256"],
            "chunkBytes": entry["chunkBytes"],
            "chunkSha256s": entry["chunkSha256s"],
        });

        let guard = verify_models(Kind::Canary, &arguments, &valid_pin.to_string())?;
        assert!(
            fs::OpenOptions::new()
                .write(true)
                .open(&arguments.model)
                .is_err()
        );
        drop(guard);

        // Missing half of the chunk fields must fail closed.
        let mut missing_hashes = valid_pin.clone();
        missing_hashes
            .as_object_mut()
            .ok_or_else(|| io::Error::other("object"))?
            .remove("chunkSha256s");
        assert!(verify_models(Kind::Canary, &arguments, &missing_hashes.to_string()).is_err());

        let mut missing_size = valid_pin.clone();
        missing_size
            .as_object_mut()
            .ok_or_else(|| io::Error::other("object"))?
            .remove("chunkBytes");
        assert!(verify_models(Kind::Canary, &arguments, &missing_size.to_string()).is_err());

        // Sub-threshold chunk size must fail closed.
        let mut small_chunk = valid_pin.clone();
        small_chunk["chunkBytes"] = serde_json::json!(1024);
        assert!(verify_models(Kind::Canary, &arguments, &small_chunk.to_string()).is_err());

        // Wrong chunk count must fail closed.
        let mut short_list = valid_pin.clone();
        short_list["chunkSha256s"] = serde_json::json!([entry["chunkSha256s"][0].clone()]);
        assert!(verify_models(Kind::Canary, &arguments, &short_list.to_string()).is_err());

        // Bad hex digest must fail closed.
        let mut bad_hex = valid_pin.clone();
        bad_hex["chunkSha256s"][1] = serde_json::json!("not-hex");
        assert!(verify_models(Kind::Canary, &arguments, &bad_hex.to_string()).is_err());

        // Single-byte tamper in the second chunk must be caught.
        {
            use std::io::{Read as _, Seek as _, SeekFrom, Write as _};
            let mut file = fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(&arguments.model)?;
            let pos = u64::try_from(chunk_bytes + 17).map_err(io::Error::other)?;
            file.seek(SeekFrom::Start(pos))?;
            let mut byte = [0_u8; 1];
            file.read_exact(&mut byte)?;
            byte[0] ^= 0x01;
            file.seek(SeekFrom::Start(pos))?;
            file.write_all(&byte)?;
            file.flush()?;
        }
        assert!(verify_models(Kind::Canary, &arguments, &valid_pin.to_string()).is_err());

        let _ = fs::remove_dir_all(&directory);
        Ok(())
    }

    fn write_small_fixture_file(path: &Path, payload: &[u8]) -> io::Result<serde_json::Value> {
        fs::write(path, payload)?;
        let whole = hex_lower(&Sha256::digest_bytes(payload)?);
        Ok(serde_json::json!({
            "bytes": payload.len(),
            "sha256": whole,
        }))
    }

    #[test]
    fn staged_tts_verification_never_loads_unverified_artifact() -> io::Result<()> {
        let (directory, mut values) = fixture()?;
        let codec = directory.join("codec.gguf");
        let voice = directory.join("voice.gguf");
        let chunk_bytes = usize::try_from(MIN_MAPPED_HASH_BYTES).map_err(io::Error::other)?;
        let codec_entry = write_small_fixture_file(&codec, b"cdc")?;
        let design_entry = write_chunked_fixture_file(
            &directory.join("model.gguf"),
            &vec![0x22_u8; chunk_bytes + 7],
            chunk_bytes,
        )?;
        let voice_entry = write_small_fixture_file(&voice, b"voc")?;

        values.extend([
            OsString::from("--codec"),
            codec.clone().into_os_string(),
            OsString::from("--voice-model"),
            voice.clone().into_os_string(),
        ]);
        let Command::Serve(arguments) = parse(Kind::QwenTts, values)? else {
            return Err(io::Error::other("Missing TTS arguments"));
        };
        let lock_value = serde_json::json!({
            "schemaVersion": 1,
            "model": Kind::QwenTts.model(),
            "files": {
                "talker": design_entry,
                "codec": codec_entry,
                "voice_talker": voice_entry,
            },
        });

        // Case 1: invalid codec -> neither loader runs.
        let mut bad_codec = lock_value.clone();
        bad_codec["files"]["codec"]["sha256"] = serde_json::json!("0".repeat(64));
        let codec_called = AtomicBool::new(false);
        let design_called = AtomicBool::new(false);
        let res = verify_and_load_tts(
            &arguments,
            &bad_codec.to_string(),
            |_| {
                codec_called.store(true, Ordering::Release);
                Ok(())
            },
            |_| {
                design_called.store(true, Ordering::Release);
                Ok(())
            },
        );
        let err = res
            .err()
            .ok_or_else(|| io::Error::other("expected verify failure"))?;
        assert_eq!(err.stage, TtsStartupStage::Verify);
        assert!(!codec_called.load(Ordering::Acquire));
        assert!(!design_called.load(Ordering::Acquire));

        // Case 2: bad design talker -> design loader must NEVER run.
        let mut bad_design = lock_value.clone();
        bad_design["files"]["talker"]["chunkSha256s"][1] = serde_json::json!("f".repeat(64));
        let design_called_2 = AtomicBool::new(false);
        let res = verify_and_load_tts(
            &arguments,
            &bad_design.to_string(),
            |_| Ok(()),
            |_| {
                design_called_2.store(true, Ordering::Release);
                Ok(())
            },
        );
        let err = res
            .err()
            .ok_or_else(|| io::Error::other("expected design verify failure"))?;
        assert_eq!(err.stage, TtsStartupStage::Verify);
        assert!(!design_called_2.load(Ordering::Acquire));

        // Case 3: loader failure reports Load stage.
        let res: Result<(_, (), ()), _> = verify_and_load_tts(
            &arguments,
            &lock_value.to_string(),
            |_| Ok(()),
            |_| {
                Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "bad design weights",
                ))
            },
        );
        let err = res
            .err()
            .ok_or_else(|| io::Error::other("expected load failure"))?;
        assert_eq!(err.stage, TtsStartupStage::Load);
        assert_eq!(err.error.kind(), io::ErrorKind::InvalidData);

        // Case 4: valid artifacts return guards + both loaded parts.
        let (guards, c_ok, d_ok) = verify_and_load_tts(
            &arguments,
            &lock_value.to_string(),
            |p| Ok(p.to_path_buf()),
            |p| Ok(p.to_path_buf()),
        )
        .map_err(|failure| failure.error)?;
        assert_eq!(c_ok, codec);
        assert_eq!(d_ok, arguments.model);
        assert!(fs::OpenOptions::new().write(true).open(&voice).is_err());
        drop(guards);

        let _ = fs::remove_dir_all(&directory);
        Ok(())
    }
}
