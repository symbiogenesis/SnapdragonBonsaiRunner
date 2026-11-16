//! Retained native control engine, enabled only by the default native-engine feature.

use crate::{MAX_CLIP_SECONDS, SAMPLE_RATE, intake, protocol};
use std::{
    fs::OpenOptions,
    io::{self, Read},
    path::Path,
    sync::Once,
};
use transcribe_cpp::{Backend, Model, ModelOptions, RunOptions, Session, SessionOptions};

const MAX_SAMPLES: usize = SAMPLE_RATE * MAX_CLIP_SECONDS;
const ARCHITECTURE: &str = "qwen3_asr";
const VARIANT: &str = "qwen3-asr-1.7b";
static LOGGING: Once = Once::new();

/// A clonable, per-request cooperative cancellation handle.
///
/// Cancellation is checked before and after inference and at the native pre-run
/// boundary. Owned process teardown bounds cancellation during active inference;
/// this Qwen3-ASR backend does not promise cooperative decoder interruption.
#[derive(Clone, Debug, Default)]
pub struct Cancellation(transcribe_cpp::CancelToken);

impl Cancellation {
    /// Construct a fresh request that has not been cancelled.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Stop this request; this handle must not be reused for another request.
    pub fn cancel(&self) {
        self.0.cancel();
    }

    /// Whether cancellation has been requested.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.0.is_cancelled()
    }
}

/// A loaded Qwen3-ASR model and one reusable recognition session.
///
/// The caller must retain its verified model-file guard while this value lives:
/// the native runtime memory-maps weights from that immutable file.
pub struct Recognizer {
    session: Session,
}

impl Recognizer {
    /// Load the selected Qwen3-ASR 1.7B GGUF on CPU using default thread policy, without downloading anything.
    ///
    /// # Errors
    /// Rejects relative paths, malformed or unexpected model headers, a different
    /// model identity/audio contract, unavailable memory, and native load faults.
    pub fn load(path: &Path) -> io::Result<Self> {
        Self::load_with_threads(path, None)
    }

    /// Load the selected Qwen3-ASR 1.7B GGUF on CPU with an optional thread budget, without downloading anything.
    ///
    /// `None` selects the host-bounded ASR native session budget (up to 8 Oryon
    /// threads across two clusters on Snapdragon X1/X2); `Some(n)` validates `1..=256`.
    ///
    /// # Errors
    /// Rejects relative paths, invalid thread budgets, malformed or unexpected model
    /// headers, a different model identity/audio contract, unavailable memory, and
    /// native load faults.
    pub fn load_with_threads(path: &Path, threads: Option<usize>) -> io::Result<Self> {
        if !path.is_absolute() {
            return Err(invalid("Use an absolute, verified Qwen3-ASR model path."));
        }
        let n_threads = crate::threads::select_native_threads(
            threads,
            std::thread::available_parallelism()?.get(),
        )?;
        let mut options = OpenOptions::new();
        options.read(true);
        #[cfg(windows)]
        {
            use std::os::windows::fs::OpenOptionsExt as _;
            // Deny replacement/writes during both the Rust preflight and native
            // intake; the service's verified ancestor/file guards outlive this.
            // Combine FILE_FLAG_OPEN_REPARSE_POINT with FILE_FLAG_SEQUENTIAL_SCAN.
            options
                .share_mode(1)
                .custom_flags(0x0020_0000 | 0x0800_0000);
        }
        let mut file = options.open(path)?;
        let properties = file.metadata()?;
        if !properties.is_file() {
            return Err(invalid("The Qwen3-ASR model must be a regular file."));
        }
        #[cfg(windows)]
        {
            use std::os::windows::fs::MetadataExt as _;
            if properties.file_attributes() & 0x400 != 0 {
                return Err(invalid("The Qwen3-ASR model must not be a reparse point."));
            }
        }
        validate_header(&mut file)?;
        intake::validate(&mut file, properties.len())?;
        LOGGING.call_once(transcribe_cpp::disable_logging);
        let model = Model::load_with(
            path,
            &ModelOptions {
                backend: Backend::Cpu,
                device: None,
                no_mmap: false,
            },
        )
        .map_err(|error| native_error(&error))?;
        let capabilities = model.capabilities().map_err(|error| native_error(&error))?;
        validate_identity(
            &model.arch(),
            &model.variant(),
            capabilities.native_sample_rate,
            capabilities
                .languages
                .iter()
                .any(|language| language == "en"),
        )?;
        let session = model
            .session_with(&SessionOptions {
                n_threads,
                n_ctx: 8192,
                ..SessionOptions::default()
            })
            .map_err(|error| native_error(&error))?;
        Ok(Self { session })
    }

    /// Recognize one complete mono 16 kHz clip of at most thirty seconds.
    ///
    /// # Errors
    /// Rejects empty, oversized or nonfinite/out-of-range samples and incomplete
    /// recognition. Native diagnostics and partial transcripts are not exposed.
    pub fn transcribe(&mut self, pcm: &[f32]) -> io::Result<String> {
        self.transcribe_cancellable(pcm, &Cancellation::new())
    }

    /// Recognize a clip with an independently clonable request cancellation flag.
    ///
    /// # Errors
    /// Returns `Interrupted` for cancellation; otherwise has the same error
    /// contract as [`Self::transcribe`]. Cancelled partial text is discarded.
    pub fn transcribe_cancellable(
        &mut self,
        pcm: &[f32],
        cancellation: &Cancellation,
    ) -> io::Result<String> {
        validate_pcm(pcm)?;
        check_cancelled(cancellation)?;
        self.session.set_cancel_token(&cancellation.0);
        let result = self.session.run(pcm, &RunOptions::default());
        self.session.clear_cancel_token();
        check_cancelled(cancellation)?;
        let transcript = result.map_err(|error| native_error(&error))?;
        protocol::validate_transcript(&transcript.text)?;
        Ok(transcript.text)
    }
}

fn validate_identity(
    architecture: &str,
    variant: &str,
    sample_rate: i32,
    supports_english: bool,
) -> io::Result<()> {
    if architecture != ARCHITECTURE
        || variant != VARIANT
        || sample_rate != 16_000
        || !supports_english
    {
        return Err(invalid(
            "The loaded model is not the required Qwen3-ASR 1.7B model.",
        ));
    }
    Ok(())
}

fn check_cancelled(cancellation: &Cancellation) -> io::Result<()> {
    if cancellation.is_cancelled() {
        return Err(io::Error::new(
            io::ErrorKind::Interrupted,
            "Qwen3-ASR recognition was cancelled.",
        ));
    }
    Ok(())
}

fn validate_pcm(pcm: &[f32]) -> io::Result<()> {
    let in_range = crate::packed::pcm_in_unit_range(pcm);
    if pcm.is_empty() || pcm.len() > MAX_SAMPLES || !in_range {
        return Err(invalid(
            "Qwen3-ASR requires bounded, finite mono 16 kHz samples normalized to [-1, 1].",
        ));
    }
    Ok(())
}

fn validate_header(reader: &mut impl Read) -> io::Result<()> {
    let mut header = [0_u8; 24];
    reader.read_exact(&mut header)?;
    if &header[..8] != b"GGUF\x03\0\0\0" {
        return Err(invalid(
            "Qwen3-ASR requires the pinned GGUF version 3 model.",
        ));
    }
    // The pinned publisher's first metadata entry names the architecture. Read
    // only this bounded field before allowing the native runtime to allocate.
    if bounded_string(reader)? != "general.architecture" {
        return Err(invalid("The Qwen3-ASR GGUF identity field is missing."));
    }
    let mut kind = [0_u8; 4];
    reader.read_exact(&mut kind)?;
    if u32::from_le_bytes(kind) != 8 || bounded_string(reader)? != ARCHITECTURE {
        return Err(invalid(
            "The GGUF does not contain the selected Qwen3-ASR architecture.",
        ));
    }
    Ok(())
}

fn bounded_string(reader: &mut impl Read) -> io::Result<String> {
    let mut length = [0_u8; 8];
    reader.read_exact(&mut length)?;
    let length = usize::try_from(u64::from_le_bytes(length))
        .ok()
        .filter(|length| *length <= 128)
        .ok_or_else(|| invalid("The Qwen3-ASR GGUF identity is invalid."))?;
    let mut value = vec![0_u8; length];
    reader.read_exact(&mut value)?;
    String::from_utf8(value).map_err(|_| invalid("The Qwen3-ASR GGUF identity is not UTF-8."))
}

fn native_error(error: &transcribe_cpp::Error) -> io::Error {
    use transcribe_cpp::Error;
    match error {
        Error::Aborted { .. } => io::Error::new(
            io::ErrorKind::Interrupted,
            "Qwen3-ASR recognition was cancelled.",
        ),
        Error::OutOfMemory(_) => io::Error::new(
            io::ErrorKind::OutOfMemory,
            "The native Qwen3-ASR model exceeded available memory.",
        ),
        Error::ModelFileNotFound(_) => io::Error::new(
            io::ErrorKind::NotFound,
            "The verified Qwen3-ASR model is unavailable.",
        ),
        Error::OutputTruncated { .. } | Error::OutputRepetition { .. } => {
            invalid("Qwen3-ASR did not produce a complete transcript.")
        }
        _ => io::Error::other("The native Qwen3-ASR inference request failed."),
    }
}

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

#[cfg(test)]
mod tests {
    use super::{
        ARCHITECTURE, Cancellation, MAX_SAMPLES, Path, Recognizer, VARIANT, check_cancelled,
        validate_header, validate_identity, validate_pcm,
    };

    use std::io::{self, Cursor};

    #[test]
    fn normalized_audio_rejects_nonfinite_samples_and_clip_overflow() {
        for samples in [&[][..], &[f32::NAN], &[f32::INFINITY], &[1.01], &[-1.01]] {
            assert!(validate_pcm(samples).is_err());
        }
        assert!(validate_pcm(&[-1.0, 0.0, 1.0]).is_ok());
        assert!(validate_pcm(&vec![0.0; MAX_SAMPLES]).is_ok());
        assert!(validate_pcm(&vec![0.0; MAX_SAMPLES + 1]).is_err());
    }

    #[test]
    fn cancelled_request_handles_cannot_cancel_a_fresh_request() {
        let first = Cancellation::new();
        let worker = first.clone();
        assert!(std::thread::spawn(move || worker.cancel()).join().is_ok());
        assert_eq!(
            check_cancelled(&first).err().map(|error| error.kind()),
            Some(io::ErrorKind::Interrupted)
        );
        assert!(check_cancelled(&Cancellation::new()).is_ok());
    }

    #[test]
    fn selected_qwen_asr_identity_is_checked_before_native_loading() {
        let mut header = b"GGUF\x03\0\0\0".to_vec();
        header.extend(1574_u64.to_le_bytes());
        header.extend(67_u64.to_le_bytes());
        header.extend(20_u64.to_le_bytes());
        header.extend(b"general.architecture");
        header.extend(8_u32.to_le_bytes());
        header.extend(
            u64::try_from(ARCHITECTURE.len())
                .unwrap_or_default()
                .to_le_bytes(),
        );
        header.extend(ARCHITECTURE.as_bytes());
        assert!(validate_header(&mut Cursor::new(&header)).is_ok());
        header[0] = b'X';
        assert!(validate_header(&mut Cursor::new(&header)).is_err());
        assert!(validate_header(&mut Cursor::new(b"GGUF")).is_err());
    }

    #[test]
    fn qwen_asr_rejects_smaller_variant_and_incompatible_audio_identity() {
        assert!(validate_identity(ARCHITECTURE, VARIANT, 16_000, true).is_ok());
        for (architecture, variant, sample_rate, english) in [
            (ARCHITECTURE, "qwen3-asr-0.6b", 16_000, true),
            ("unsupported_asr", "unsupported-variant", 16_000, true),
            (ARCHITECTURE, VARIANT, 48_000, true),
            (ARCHITECTURE, VARIANT, 16_000, false),
        ] {
            assert!(validate_identity(architecture, variant, sample_rate, english).is_err());
        }
    }

    #[test]
    fn qwen_asr_header_rejects_other_families_and_unbounded_identity() {
        let mut header = b"GGUF\x03\0\0\0".to_vec();
        header.extend(1574_u64.to_le_bytes());
        header.extend(67_u64.to_le_bytes());
        header.extend(20_u64.to_le_bytes());
        header.extend(b"general.architecture");
        header.extend(8_u32.to_le_bytes());
        for architecture in ["unsupported_asr", "whisper", "qwen3"] {
            let mut wrong = header.clone();
            wrong.extend(
                u64::try_from(architecture.len())
                    .unwrap_or_default()
                    .to_le_bytes(),
            );
            wrong.extend(architecture.as_bytes());
            assert!(validate_header(&mut Cursor::new(wrong)).is_err());
        }
        header.extend(129_u64.to_le_bytes());
        assert!(validate_header(&mut Cursor::new(header)).is_err());
    }

    #[test]
    fn qwen_asr_rejects_invalid_thread_budgets_before_opening_model() {
        let path = Path::new(r"C:\missing\qwen3-asr-1.7b-q8_0.gguf");
        for invalid_threads in [0, 257] {
            assert_eq!(
                Recognizer::load_with_threads(path, Some(invalid_threads))
                    .err()
                    .map(|error| error.kind()),
                Some(io::ErrorKind::InvalidInput)
            );
        }
    }
}
