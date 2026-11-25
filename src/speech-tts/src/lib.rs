//! Native, process-owned Qwen3-TTS voice design with reusable Base conditioning.
//!
//! The caller verifies and retains the pinned GGUF model artifacts before
//! loading them. Weight intake, Talker/CodePredictor, codec and speaker inference
//! run in Rust on the CPU. This library has no network, preset-voice, external
//! inference DLL or audio-device dependency. Keep it in its supervised process.
#![cfg(windows)]

use std::{
    collections::VecDeque,
    fmt,
    fs::OpenOptions,
    io::{self, Read},
    path::{Path, PathBuf},
    sync::Mutex,
    time::SystemTime,
};

#[cfg(test)]
mod native;
mod rust_codec;
mod rust_talker;
mod weights;

pub mod codec;
#[cfg(test)]
mod decode_graph;
pub mod engine;
mod numerics;
pub mod observation;
mod philox;
pub mod replay;
/// Internal binary entrypoint module; not a public API.
#[cfg(feature = "rust-qualification")]
#[doc(hidden)]
#[path = "replay/consumer.rs"]
pub mod replay_consumer;
mod sampling;
pub mod speaker_encoder;
mod tokenizer;

/// Immutable source provenance of the Qwen implementation ported into Rust.
pub const ENGINE_REVISION: &str = "5916bf7c79dbb4e66fc059cdc507a79a23a28bb4";
/// Exact upstream checkpoint represented by the pinned quantized GGUF files.
pub const MODEL: &str = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign";
/// Default mono PCM output sampling rate (`24_000` Hz).
pub const SAMPLE_RATE: u32 = 24_000;
/// High-fidelity 48 kHz mono PCM output sampling rate (`takuma104/Qwen3-TTS-Tokenizer-12Hz-48kHz`).
pub const SAMPLE_RATE_48K: u32 = 48_000;
const MAX_TEXT: usize = 2048;
const MAX_VOICE: usize = 1024;
const MAX_SAMPLES: usize = 24_000 * 120;
const MAX_SAMPLES_48K: usize = 48_000 * 120;
const TOKEN_CEILING_SAMPLES: usize = (1500 - 2) * 1920;
const TOKEN_CEILING_SAMPLES_48K: usize = (1500 - 2) * 3840;

#[derive(Clone, Copy, Debug)]
enum SynthesisStage {
    DesignModel,
    ReferenceGeneration,
    ReferenceDecode,
    ReferenceValidation,
    BaseWeights,
    BaseMetadata,
    SpeakerWeights,
    SpeakerExtraction,
    ReferenceEncoding,
    BaseTokenizer,
    BaseModel,
    CodecPriming,
    Generation,
    CodecDecode,
    AudioValidation,
    AudioOutput,
}

impl SynthesisStage {
    const fn name(self) -> &'static str {
        match self {
            Self::DesignModel => "VoiceDesign model loading",
            Self::ReferenceGeneration => "voice reference generation",
            Self::ReferenceDecode => "voice reference audio decoding",
            Self::ReferenceValidation => "voice reference audio validation",
            Self::BaseWeights => "Base model weight intake",
            Self::BaseMetadata => "Base model metadata validation",
            Self::SpeakerWeights => "speaker encoder loading",
            Self::SpeakerExtraction => "speaker embedding extraction",
            Self::ReferenceEncoding => "voice reference audio encoding",
            Self::BaseTokenizer => "Base tokenizer loading",
            Self::BaseModel => "Base model loading",
            Self::CodecPriming => "voice reference codec priming",
            Self::Generation => "speech token generation",
            Self::CodecDecode => "speech audio decoding",
            Self::AudioValidation => "speech audio validation",
            Self::AudioOutput => "speech audio output",
        }
    }

    fn failure(self, error: io::Error) -> io::Error {
        if synthesis_failure_stage(&error).is_some() {
            return error;
        }
        io::Error::new(error.kind(), SynthesisFailure(self))
    }
}

#[derive(Debug)]
struct SynthesisFailure(SynthesisStage);

impl fmt::Display for SynthesisFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.0.name())
    }
}

impl std::error::Error for SynthesisFailure {}

/// Return a fixed, privacy-safe stage label for a synthesis failure.
///
/// Labels come exclusively from this library's stage enum. Arbitrary error text,
/// model paths, request text and voice descriptions are never returned. Obtain
/// the original category separately with [`io::Error::kind`].
#[must_use]
pub fn synthesis_failure_stage(error: &io::Error) -> Option<&'static str> {
    error
        .get_ref()?
        .downcast_ref::<SynthesisFailure>()
        .map(|failure| failure.0.name())
}

/// Returns `true` when `sample_rate` is a supported Qwen3-TTS output rate (`24_000` or `48_000` Hz).
#[must_use]
pub const fn is_supported_sample_rate(sample_rate: u32) -> bool {
    sample_rate == SAMPLE_RATE || sample_rate == SAMPLE_RATE_48K
}

/// Complete validated mono speech audio, retained only in memory.
pub struct Audio {
    /// Finite normalized floating-point PCM samples, in playback order.
    pub samples: Vec<f32>,
    /// Sampling rate (`24_000` or `48_000` Hz).
    pub sample_rate: u32,
}

/// Complete validated mono samples in one Rust-owned allocation.
///
/// Borrow with [`AsRef::as_ref`] without copying the complete waveform.
pub struct Waveform {
    samples: Vec<f32>,
    sample_rate: u32,
}

impl Waveform {
    /// Sampling rate (`24_000` or `48_000` Hz) of the retained waveform.
    #[must_use]
    pub const fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

impl AsRef<[f32]> for Waveform {
    fn as_ref(&self) -> &[f32] {
        &self.samples
    }
}

/// Exclusive ownership of one native model and its selected reusable voice.
///
/// `VoiceDesign` creates a fixed reference once per profile. Base synthesis then
/// uses that reference for every text, including separate paragraphs/messages.
/// The two model contexts never coexist, preserving the speech memory policy.
pub struct Synthesizer {
    voices: VoiceCache<Voice>,
    design: Option<rust_talker::Talker>,
    model_path: PathBuf,
    voice_model_path: PathBuf,
    codec: rust_codec::Codec,
    threads: usize,
}

// A callback can unwind as well as return an error. Keep cleanup outside the
// callback stack so neither voice nor codec snapshots survive non-cooperative failure.
struct SynthesisGuard<'a> {
    synthesizer: &'a mut Synthesizer,
    retain_warm: bool,
}

impl Drop for SynthesisGuard<'_> {
    fn drop(&mut self) {
        self.synthesizer.codec.reset();
        if !self.retain_warm {
            self.synthesizer.codec.discard_reference();
            self.synthesizer.voices.0 = None;
        }
    }
}

fn flush_pcm_chunk<C, O, S>(
    codec: &mut rust_codec::Codec,
    frames: &[[i32; engine::TOTAL_CODE_GROUPS]],
    count: &mut usize,
    cancel: &std::cell::RefCell<C>,
    observe: &std::cell::RefCell<&mut O>,
    sink: &mut S,
) -> io::Result<()>
where
    C: FnMut() -> io::Result<()>,
    O: FnMut(observation::Observation<'_>) -> io::Result<()> + ?Sized,
    S: FnMut(&[f32]) -> io::Result<()> + ?Sized,
{
    let chunk = codec
        .decode_stream_cancellable(frames, &mut || (cancel.borrow_mut())())
        .map_err(|error| SynthesisStage::CodecDecode.failure(error))?;
    *count = sample_count(&chunk, *count)
        .ok_or_else(|| SynthesisStage::AudioValidation.failure(invalid_audio()))?;
    observe.borrow_mut()(observation::Observation::Pcm {
        phase: observation::Phase::Base,
        samples: &chunk,
    })?;
    sink(&chunk).map_err(|error| SynthesisStage::AudioOutput.failure(error))
}

const MAX_DERIVED_REFERENCES: usize = 32;
const MAX_MODEL_DIGESTS: usize = 8;
const MODEL_DIGEST_MEMOIZE_MIN_BYTES: u64 = 16 * 1024 * 1024;
const HASH_READ_BUFFER_BYTES: usize = 1024 * 1024;

struct Voice {
    talker: rust_talker::Talker,
    reference: rust_talker::Reference,
}

trait VoiceParts {
    type Talker;
    fn into_talker(self) -> Self::Talker;
    fn from_parts(talker: Self::Talker, reference: rust_talker::Reference) -> Self;
}

impl VoiceParts for Voice {
    type Talker = rust_talker::Talker;

    fn into_talker(self) -> Self::Talker {
        self.talker
    }

    fn from_parts(talker: Self::Talker, reference: rust_talker::Reference) -> Self {
        Self { talker, reference }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct DerivedReferenceKey {
    voice_profile_id: Box<str>,
    model_sha256: [u8; 32],
}

#[derive(Clone, Debug, PartialEq)]
struct DerivedReference {
    speaker_embedding: Vec<f32>,
    reference_codes: Vec<[i32; engine::TOTAL_CODE_GROUPS]>,
}

struct DerivedReferenceCache {
    entries: VecDeque<(DerivedReferenceKey, DerivedReference)>,
}

impl DerivedReferenceCache {
    const fn new() -> Self {
        Self {
            entries: VecDeque::new(),
        }
    }

    fn get(&mut self, profile: &str, model_sha256: &[u8; 32]) -> Option<DerivedReference> {
        let index = self.entries.iter().position(|(key, _)| {
            key.voice_profile_id.as_ref() == profile && &key.model_sha256 == model_sha256
        })?;
        let entry = self.entries.remove(index)?;
        let result = entry.1.clone();
        self.entries.push_back(entry);
        Some(result)
    }

    fn insert(&mut self, profile: &str, model_sha256: [u8; 32], value: DerivedReference) {
        if let Some(index) = self.entries.iter().position(|(key, _)| {
            key.voice_profile_id.as_ref() == profile && key.model_sha256 == model_sha256
        }) {
            let _removed = self.entries.remove(index);
        } else {
            while self.entries.len() >= MAX_DERIVED_REFERENCES {
                let _evicted = self.entries.pop_front();
            }
        }
        self.entries.push_back((
            DerivedReferenceKey {
                voice_profile_id: Box::from(profile),
                model_sha256,
            },
            value,
        ));
    }
}

struct CachedModelDigest {
    path: PathBuf,
    len: u64,
    modified: Option<SystemTime>,
    sha256: [u8; 32],
}

struct ModelDigestCache {
    entries: VecDeque<CachedModelDigest>,
}

impl ModelDigestCache {
    const fn new() -> Self {
        Self {
            entries: VecDeque::new(),
        }
    }

    fn get(&mut self, path: &Path, len: u64, modified: Option<SystemTime>) -> Option<[u8; 32]> {
        let index = self.entries.iter().position(|entry| {
            entry.len == len && entry.modified == modified && entry.path == path
        })?;
        let entry = self.entries.remove(index)?;
        let sha256 = entry.sha256;
        self.entries.push_back(entry);
        Some(sha256)
    }

    fn insert(&mut self, path: &Path, len: u64, modified: Option<SystemTime>, sha256: [u8; 32]) {
        if let Some(index) = self.entries.iter().position(|entry| entry.path == path) {
            let _removed = self.entries.remove(index);
        } else {
            while self.entries.len() >= MAX_MODEL_DIGESTS {
                let _evicted = self.entries.pop_front();
            }
        }
        self.entries.push_back(CachedModelDigest {
            path: path.to_path_buf(),
            len,
            modified,
            sha256,
        });
    }
}

static DERIVED_REFERENCE_CACHE: Mutex<DerivedReferenceCache> =
    Mutex::new(DerivedReferenceCache::new());
static MODEL_DIGEST_CACHE: Mutex<ModelDigestCache> = Mutex::new(ModelDigestCache::new());

fn file_sha256(path: &Path, cancel: &mut impl FnMut() -> io::Result<()>) -> io::Result<[u8; 32]> {
    use std::os::windows::fs::{MetadataExt as _, OpenOptionsExt as _};

    if !path.is_absolute() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Speech model path must be absolute.",
        ));
    }
    let mut options = OpenOptions::new();
    options.read(true);
    // Read sharing only + reject reparse points + sequential scan.
    options
        .share_mode(1)
        .custom_flags(0x0020_0000 | 0x0800_0000);
    let mut file = options.open(path)?;
    let metadata = file.metadata()?;
    if !metadata.is_file() || metadata.file_attributes() & 0x400 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Speech model must be a regular non-reparse file.",
        ));
    }
    let len = metadata.len();
    let modified = metadata.modified().ok();
    if len > MODEL_DIGEST_MEMOIZE_MIN_BYTES
        && let Ok(mut cache) = MODEL_DIGEST_CACHE.lock()
        && let Some(digest) = cache.get(path, len, modified)
    {
        return Ok(digest);
    }
    let mut hasher = bonsai_native_hash::Hasher::new().map_err(io::Error::other)?;
    let mut buffer = vec![0_u8; HASH_READ_BUFFER_BYTES];
    loop {
        cancel()?;
        match file.read(&mut buffer) {
            Ok(0) => break,
            Ok(count) => hasher.update(&buffer[..count]).map_err(io::Error::other)?,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error),
        }
    }
    let digest = hasher.finish().map_err(io::Error::other)?;
    if len > MODEL_DIGEST_MEMOIZE_MIN_BYTES
        && let Ok(mut cache) = MODEL_DIGEST_CACHE.lock()
    {
        cache.insert(path, len, modified, digest);
    }
    Ok(digest)
}

fn model_identity_sha256(
    model_path: &Path,
    voice_model_path: &Path,
    cancel: &mut impl FnMut() -> io::Result<()>,
) -> io::Result<[u8; 32]> {
    let design_digest = file_sha256(model_path, cancel)?;
    let base_digest = file_sha256(voice_model_path, cancel)?;
    let mut combined = bonsai_native_hash::Hasher::new().map_err(io::Error::other)?;
    combined
        .update(b"bonsai-tts-voice-model-v1")
        .map_err(io::Error::other)?;
    combined.update(&design_digest).map_err(io::Error::other)?;
    combined.update(&base_digest).map_err(io::Error::other)?;
    combined.finish().map_err(io::Error::other)
}

// The cache key deliberately excludes utterance text, request identifiers,
// playback mode and chunk boundaries. Replacing a profile releases its model
// before constructing the next one; acquisition failures cannot fall back to
// an unrelated voice or retain two loaded models.
struct VoiceCache<T>(Option<(Box<str>, T)>);

// The waveform allocation is independent of its model context.
// Keep its owner until synchronous Base extraction returns, including failures,
// while releasing Design before allocating Base's model and workers.
fn finish_voice_bootstrap<D, B, S: AsRef<[f32]>, R>(
    design: D,
    samples: S,
    load_base: impl FnOnce() -> io::Result<B>,
    extract: impl FnOnce(&mut B, &[f32]) -> io::Result<R>,
) -> io::Result<(B, R)> {
    drop(design);
    let mut base = load_base()?;
    let reference = extract(&mut base, samples.as_ref())?;
    drop(samples);
    Ok((base, reference))
}

impl<T> VoiceCache<T> {
    #[cfg(test)]
    fn get_or_create(
        &mut self,
        profile: &str,
        create: impl FnOnce() -> io::Result<T>,
    ) -> io::Result<&mut T> {
        if let Some((key, voice)) = self.0.as_mut()
            && &**key == profile
        {
            return Ok(voice);
        }
        self.0 = None;
        let voice = create()?;
        Ok(&mut self.0.insert((Box::from(profile), voice)).1)
    }

    fn get_or_acquire<D>(
        &mut self,
        profile: &str,
        design_slot: &mut Option<D>,
        reference_cache: &Mutex<DerivedReferenceCache>,
        model_sha256: impl FnOnce() -> io::Result<[u8; 32]>,
        load_base: impl FnOnce() -> io::Result<T::Talker>,
        derive_voice: impl FnOnce(&mut Option<D>) -> io::Result<(T::Talker, rust_talker::Reference)>,
    ) -> io::Result<&mut T>
    where
        T: VoiceParts,
    {
        if self.0.as_ref().is_some_and(|(key, _)| &**key == profile)
            && let Some((_, voice)) = self.0.as_mut()
        {
            return Ok(voice);
        }
        let digest = model_sha256()?;
        let cached = reference_cache
            .lock()
            .ok()
            .and_then(|mut cache| cache.get(profile, &digest));
        if let Some(derived) = cached {
            // Release any preloaded Design model before loading/retaining Base
            // so both model contexts never coexist in memory.
            *design_slot = None;
            let talker = match self.0.take() {
                Some((_, previous)) => previous.into_talker(),
                None => load_base()?,
            };
            let voice = T::from_parts(
                talker,
                rust_talker::Reference {
                    speaker: derived.speaker_embedding,
                    codes: derived.reference_codes,
                },
            );
            return Ok(&mut self.0.insert((Box::from(profile), voice)).1);
        }
        // Release the active Base model before VoiceDesign runs on a cache miss.
        self.0 = None;
        let (talker, reference) = derive_voice(design_slot)?;
        if let Ok(mut cache) = reference_cache.lock() {
            cache.insert(
                profile,
                digest,
                DerivedReference {
                    speaker_embedding: reference.speaker.clone(),
                    reference_codes: reference.codes.clone(),
                },
            );
        }
        let voice = T::from_parts(talker, reference);
        Ok(&mut self.0.insert((Box::from(profile), voice)).1)
    }
}

fn load_talker(path: &Path, kind: &str, threads: usize) -> io::Result<rust_talker::Talker> {
    let weights = weights::Weights::open(path)?;
    if weights.string("general.architecture")? != "qwen3-tts"
        || weights.string("qwen3-tts.model_type")? != kind
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Unsupported speech model architecture.",
        ));
    }
    rust_talker::Talker::load_with_weights(&weights, threads)
}

/// Preloaded, validated Qwen3 tokenizer codec ready for [`Synthesizer::assemble`].
pub struct LoadedCodec {
    codec: rust_codec::Codec,
}

/// Preloaded, validated Qwen3 `VoiceDesign` talker ready for [`Synthesizer::assemble`].
pub struct LoadedDesign {
    talker: rust_talker::Talker,
}

impl Synthesizer {
    /// Load the verified pinned `VoiceDesign`, codec and Base GGUFs in Rust.
    ///
    /// # Errors
    /// Rejects missing, unsafe or incompatible weights and unsupported metadata.
    pub fn load(model: &Path, codec: &Path, voice_model: &Path) -> io::Result<Self> {
        Self::load_with_threads(model, codec, voice_model, None)
    }

    /// Load verified weights with an optional CPU compute team size (1?256).
    ///
    /// The default reserves two available CPUs for the UI and text runtime.
    ///
    /// # Errors
    /// Returns the same failures as [`Self::load`] or rejects an invalid team size.
    pub fn load_with_threads(
        model: &Path,
        codec: &Path,
        voice_model: &Path,
        threads: Option<usize>,
    ) -> io::Result<Self> {
        let loaded_codec = Self::load_codec_with_threads(codec, threads)?;
        let loaded_design = Self::load_design_with_threads(model, threads)?;
        Self::assemble(model, voice_model, loaded_codec, loaded_design, threads)
    }

    /// Load and validate only the verified pinned codec GGUF with an optional team size.
    ///
    /// # Errors
    /// Rejects unsafe paths, invalid team sizes, and incompatible or malformed codec weights.
    pub fn load_codec_with_threads(
        codec: &Path,
        threads: Option<usize>,
    ) -> io::Result<LoadedCodec> {
        model_path(codec)?;
        let threads = select_threads(threads, std::thread::available_parallelism()?.get())?;
        let codec_weights = weights::Weights::open(codec)?;
        let codec = rust_codec::Codec::load(&codec_weights, threads)?;
        Ok(LoadedCodec { codec })
    }

    /// Load and validate only the verified pinned `VoiceDesign` GGUF with an optional team size.
    ///
    /// # Errors
    /// Rejects unsafe paths, invalid team sizes, and incompatible or malformed talker weights.
    pub fn load_design_with_threads(
        model: &Path,
        threads: Option<usize>,
    ) -> io::Result<LoadedDesign> {
        model_path(model)?;
        let threads = select_threads(threads, std::thread::available_parallelism()?.get())?;
        let talker = load_talker(model, "voice_design", threads)?;
        Ok(LoadedDesign { talker })
    }

    /// Assemble a [`Synthesizer`] from preloaded codec and `VoiceDesign` components.
    ///
    /// # Errors
    /// Rejects unsafe paths or invalid team sizes.
    pub fn assemble(
        model: &Path,
        voice_model: &Path,
        codec: LoadedCodec,
        design: LoadedDesign,
        threads: Option<usize>,
    ) -> io::Result<Self> {
        model_path(model)?;
        model_path(voice_model)?;
        let threads = select_threads(threads, std::thread::available_parallelism()?.get())?;
        Ok(Self {
            voices: VoiceCache(None),
            design: Some(design.talker),
            model_path: model.to_owned(),
            voice_model_path: voice_model.to_owned(),
            codec: codec.codec,
            threads,
        })
    }

    /// Active mono PCM output sampling rate (24 kHz for the pinned codec).
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        self.codec.sample_rate()
    }

    fn validate_base_weights(weights: &weights::Weights) -> io::Result<()> {
        if weights
            .string("qwen3-tts.model_type")
            .map_err(|error| SynthesisStage::BaseMetadata.failure(error))?
            != "base"
        {
            return Err(SynthesisStage::BaseMetadata.failure(io::Error::new(
                io::ErrorKind::InvalidData,
                "Speech voice model must be Base.",
            )));
        }
        Ok(())
    }

    fn load_talker_from_base_weights(
        weights: &weights::Weights,
        threads: usize,
        cancel: &mut impl FnMut() -> io::Result<()>,
    ) -> io::Result<rust_talker::Talker> {
        cancel().map_err(|error| SynthesisStage::BaseTokenizer.failure(error))?;
        let tokenizer = tokenizer::Tokenizer::from_weights(weights)
            .map_err(|error| SynthesisStage::BaseTokenizer.failure(error))?;
        cancel().map_err(|error| SynthesisStage::BaseModel.failure(error))?;
        rust_talker::Talker::load(weights, tokenizer, threads)
            .map_err(|error| SynthesisStage::BaseModel.failure(error))
    }

    fn voice<C, O>(
        &mut self,
        profile: &str,
        cancel: &std::cell::RefCell<C>,
        observe: &std::cell::RefCell<&mut O>,
    ) -> io::Result<&mut Voice>
    where
        C: FnMut() -> io::Result<()>,
        O: FnMut(observation::Observation<'_>) -> io::Result<()> + ?Sized,
    {
        let Self {
            voices,
            design,
            model_path,
            voice_model_path,
            codec,
            threads,
        } = self;
        voices.get_or_acquire(
            profile,
            design,
            &DERIVED_REFERENCE_CACHE,
            || {
                model_identity_sha256(model_path, voice_model_path, &mut || {
                    (cancel.borrow_mut())()
                })
                .map_err(|error| SynthesisStage::BaseWeights.failure(error))
            },
            || {
                (cancel.borrow_mut())()
                    .map_err(|error| SynthesisStage::BaseWeights.failure(error))?;
                let weights = weights::Weights::open(voice_model_path)
                    .map_err(|error| SynthesisStage::BaseWeights.failure(error))?;
                Self::validate_base_weights(&weights)?;
                Self::load_talker_from_base_weights(&weights, *threads, &mut || {
                    (cancel.borrow_mut())()
                })
            },
            |design| {
                (cancel.borrow_mut())()
                    .map_err(|error| SynthesisStage::DesignModel.failure(error))?;
                let mut talker = match design.take() {
                    Some(talker) => talker,
                    None => load_talker(model_path, "voice_design", *threads)
                        .map_err(|error| SynthesisStage::DesignModel.failure(error))?,
                };
                let mut codes = Vec::new();
                talker
                    .generate_observed(
                        rust_talker::REFERENCE_TEXT,
                        Some(profile),
                        None,
                        &mut |frame| {
                            codes.push(frame);
                            Ok(())
                        },
                        &mut || (cancel.borrow_mut())(),
                        &mut |observation| {
                            observe.borrow_mut()(observation::Observation::Talker {
                                phase: observation::Phase::Design,
                                observation,
                            })
                        },
                    )
                    .map_err(|error| SynthesisStage::ReferenceGeneration.failure(error))?;
                let samples = codec
                    .decode_cancellable(&codes, &mut || (cancel.borrow_mut())())
                    .map_err(|error| SynthesisStage::ReferenceDecode.failure(error))?;
                let count = sample_count(&samples, 0)
                    .ok_or_else(|| SynthesisStage::ReferenceValidation.failure(invalid_audio()))?;
                if !(1920..=24_000 * 30).contains(&count) {
                    return Err(SynthesisStage::ReferenceValidation.failure(invalid_audio()));
                }
                observe.borrow_mut()(observation::Observation::Pcm {
                    phase: observation::Phase::Design,
                    samples: samples.as_ref(),
                })?;
                // This is the same ownership handoff exercised by the independent
                // drop-order control: Design is gone before Base intake, and one
                // waveform survives through both speaker and acoustic extraction.
                let (_, voice_parts) = finish_voice_bootstrap(
                    talker,
                    samples,
                    || {
                        (cancel.borrow_mut())()
                            .map_err(|error| SynthesisStage::BaseWeights.failure(error))?;
                        weights::Weights::open(voice_model_path)
                            .map_err(|error| SynthesisStage::BaseWeights.failure(error))
                    },
                    |weights, samples| {
                        Self::validate_base_weights(weights)?;
                        let speaker = rust_codec::Speaker::load(weights, *threads)
                            .map_err(|error| SynthesisStage::SpeakerWeights.failure(error))?
                            .extract_cancellable(samples, &mut || (cancel.borrow_mut())())
                            .map_err(|error| SynthesisStage::SpeakerExtraction.failure(error))?;
                        let reference_codes = codec
                            .encode_reference_cancellable(samples, &mut || (cancel.borrow_mut())())
                            .map_err(|error| SynthesisStage::ReferenceEncoding.failure(error))?;
                        let talker =
                            Self::load_talker_from_base_weights(weights, *threads, &mut || {
                                (cancel.borrow_mut())()
                            })?;
                        let reference = rust_talker::Reference {
                            speaker,
                            codes: reference_codes,
                        };
                        Ok((talker, reference))
                    },
                )?;
                Ok(voice_parts)
            },
        )
    }

    /// Generate a complete clip using the voice selected by its description.
    ///
    /// # Errors
    /// Rejects invalid input, incomplete token generation, invalid samples and
    /// inference failures. Text and voice descriptions are never logged.
    pub fn synthesize(&mut self, text: &str, voice: &str) -> io::Result<Audio> {
        self.synthesize_cancellable(text, voice, &mut || Ok(()))
    }

    /// Generate a complete clip with cooperative cancellation polling.
    ///
    /// # Errors
    /// Returns [`Self::synthesize`] failures or the first error yielded by
    /// `cancel`; cooperative `cancel` resets stream state while retaining committed warm voice/codec reuse, whereas other failures or unwinds discard it.
    pub fn synthesize_cancellable(
        &mut self,
        text: &str,
        voice: &str,
        cancel: &mut (impl FnMut() -> io::Result<()> + Send + ?Sized),
    ) -> io::Result<Audio> {
        let audio = self.synthesize_owned_cancellable(text, voice, cancel)?;
        Ok(Audio {
            samples: audio.samples,
            sample_rate: audio.sample_rate,
        })
    }

    /// Generate a complete clip in a single Rust-owned waveform buffer.
    ///
    /// # Errors
    /// Returns the same failures as [`Self::synthesize`].
    pub fn synthesize_owned(&mut self, text: &str, voice: &str) -> io::Result<Waveform> {
        self.synthesize_owned_cancellable(text, voice, &mut || Ok(()))
    }

    /// Generate a complete Rust-owned waveform buffer with cooperative
    /// cancellation polling.
    ///
    /// # Errors
    /// Returns the same failures as [`Self::synthesize_cancellable`].
    pub fn synthesize_owned_cancellable(
        &mut self,
        text: &str,
        voice: &str,
        cancel: &mut (impl FnMut() -> io::Result<()> + Send + ?Sized),
    ) -> io::Result<Waveform> {
        let mut samples = Vec::new();
        self.synthesize_stream_cancellable(
            text,
            voice,
            &mut |chunk| {
                samples.try_reserve(chunk.len()).map_err(io::Error::other)?;
                samples.extend_from_slice(chunk);
                Ok(())
            },
            cancel,
        )?;
        Ok(Waveform {
            samples,
            sample_rate: self.sample_rate(),
        })
    }

    /// Generate bounded mono PCM chunks using the selected reusable voice.
    ///
    /// Partial audio heard before an error remains partial. Success requires EOS
    /// and complete sample validation. Sink errors stop generation immediately.
    ///
    /// # Errors
    /// Returns input, inference, output validation or sink failures without
    /// exposing private text or model diagnostics.
    pub fn synthesize_stream(
        &mut self,
        text: &str,
        voice: &str,
        sink: &mut (impl FnMut(&[f32]) -> io::Result<()> + Send + ?Sized),
    ) -> io::Result<()> {
        self.synthesize_stream_cancellable(text, voice, sink, &mut || Ok(()))
    }

    /// Generate bounded mono PCM chunks with cooperative cancellation polling.
    ///
    /// Note: the out-of-process `speech-service` binary retains its explicit
    /// `exit(70)` watchdog contract when a client disconnects asynchronously.
    ///
    /// # Errors
    /// Returns [`Self::synthesize_stream`] failures or `cancel` errors;
    /// cooperative `cancel` resets transient stream state while retaining committed warm voice/codec reuse, whereas other errors or unwinds discard it.
    pub fn synthesize_stream_cancellable(
        &mut self,
        text: &str,
        voice: &str,
        sink: &mut (impl FnMut(&[f32]) -> io::Result<()> + Send + ?Sized),
        cancel: &mut (impl FnMut() -> io::Result<()> + Send + ?Sized),
    ) -> io::Result<()> {
        self.synthesize_stream_observed_cancellable(text, voice, sink, cancel, &mut |_| Ok(()))
    }

    /// Synthesize with borrowed observations of both model roles and output.
    ///
    /// Bind exact source/compiler/executable, verified model locks, input bytes,
    /// worker count and [`observation::ARITHMETIC_IDENTITY`] before capture.
    /// Independent expected values, original replay bytes and tolerances remain
    /// the caller's responsibility. The observer supplies no parity verdict.
    /// Callbacks are synchronous and copy no state in the engine. Their work
    /// shares the caller's original workload deadline. Normal serving uses a
    /// no-op observer. `Complete` is valid only after this method returns `Ok`.
    ///
    /// # Errors
    /// Returns [`Self::synthesize_stream`] failures or observer refusal. After
    /// input admission, failure and unwinding discard voice/codec reuse; emitted
    /// audio/events stay partial. Rejected input preserves the selected voice.
    pub fn synthesize_stream_observed(
        &mut self,
        text: &str,
        voice: &str,
        sink: &mut (impl FnMut(&[f32]) -> io::Result<()> + Send + ?Sized),
        observe: &mut (impl FnMut(observation::Observation<'_>) -> io::Result<()> + ?Sized),
    ) -> io::Result<()> {
        self.synthesize_stream_observed_cancellable(text, voice, sink, &mut || Ok(()), observe)
    }

    /// Synthesize with both cooperative cancellation polling and borrowed
    /// model/audio observations.
    ///
    /// # Errors
    /// Returns [`Self::synthesize_stream_observed`] failures or the first error
    /// yielded by `cancel`. Cooperative `cancel` resets transient stream state while retaining committed warm reuse; observer refusals, errors, and unwinds discard it.
    pub fn synthesize_stream_observed_cancellable(
        &mut self,
        text: &str,
        voice: &str,
        sink: &mut (impl FnMut(&[f32]) -> io::Result<()> + Send + ?Sized),
        cancel: &mut (impl FnMut() -> io::Result<()> + Send + ?Sized),
        observe: &mut (impl FnMut(observation::Observation<'_>) -> io::Result<()> + ?Sized),
    ) -> io::Result<()> {
        let text = bounded_input(text, MAX_TEXT)?;
        let profile = bounded_input(voice, MAX_VOICE)?;
        let cancelled_by_hook = std::sync::atomic::AtomicBool::new(false);
        let tracked_cancel = || {
            let res = cancel();
            if res.is_err() {
                cancelled_by_hook.store(true, std::sync::atomic::Ordering::Relaxed);
            }
            res
        };
        let cancel_ref = std::cell::RefCell::new(tracked_cancel);
        let observe = std::cell::RefCell::new(observe);
        let mut guard = SynthesisGuard {
            synthesizer: self,
            retain_warm: false,
        };
        let synthesizer = &mut *guard.synthesizer;
        let result = (|| {
            let _ = synthesizer.voice(profile, &cancel_ref, &observe)?;
            let Self { voices, codec, .. } = synthesizer;
            let (_, voice) = voices.0.as_mut().ok_or_else(invalid_audio)?;
            (cancel_ref.borrow_mut())()
                .map_err(|error| SynthesisStage::CodecPriming.failure(error))?;
            observe.borrow_mut()(observation::Observation::Speaker(&voice.reference.speaker))?;
            observe.borrow_mut()(observation::Observation::ReferenceCodes(
                &voice.reference.codes,
            ))?;
            codec
                .prime_reference_cancellable(&voice.reference.codes, &mut || {
                    (cancel_ref.borrow_mut())()
                })
                .map_err(|error| SynthesisStage::CodecPriming.failure(error))?;
            let chunk_cfg = engine::StreamChunkConfig::dyadic_ramp();
            let mut chunk_idx = 0_usize;
            let mut pending_frames: Vec<[i32; engine::TOTAL_CODE_GROUPS]> =
                Vec::with_capacity(chunk_cfg.steady_chunk_frames);
            let mut count = 0;
            voice
                .talker
                .generate_observed(
                    text,
                    None,
                    Some(&voice.reference),
                    &mut |frame| {
                        pending_frames.push(frame);
                        if pending_frames.len() >= chunk_cfg.dyadic_frames_for_chunk(chunk_idx) {
                            flush_pcm_chunk(
                                codec,
                                &pending_frames,
                                &mut count,
                                &cancel_ref,
                                &observe,
                                sink,
                            )?;
                            pending_frames.clear();
                            chunk_idx = chunk_idx.saturating_add(1);
                        }
                        Ok(())
                    },
                    &mut || (cancel_ref.borrow_mut())(),
                    &mut |observation| {
                        observe.borrow_mut()(observation::Observation::Talker {
                            phase: observation::Phase::Base,
                            observation,
                        })
                    },
                )
                .map_err(|error| SynthesisStage::Generation.failure(error))?;
            if !pending_frames.is_empty() {
                flush_pcm_chunk(
                    codec,
                    &pending_frames,
                    &mut count,
                    &cancel_ref,
                    &observe,
                    sink,
                )?;
                pending_frames.clear();
            }
            if count == 0 {
                return Err(SynthesisStage::AudioValidation.failure(invalid_audio()));
            }
            observe.borrow_mut()(observation::Observation::Complete {
                samples: count,
                sample_rate: codec.sample_rate(),
            })?;
            Ok(())
        })();
        guard.retain_warm =
            result.is_ok() || cancelled_by_hook.load(std::sync::atomic::Ordering::Relaxed);
        result
    }
}

fn invalid_audio() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        "Speech generation did not produce complete valid audio.",
    )
}

// Shared by borrowed complete output and streaming callbacks. Native output is
// admitted before copying or delivering it, with one token-completion policy.
fn sample_count(samples: &[f32], previous: usize) -> Option<usize> {
    sample_count_for_rate(samples, previous, SAMPLE_RATE)
}

fn sample_count_for_rate(samples: &[f32], previous: usize, sample_rate: u32) -> Option<usize> {
    let ceiling = if sample_rate == SAMPLE_RATE_48K {
        MAX_SAMPLES_48K.min(TOKEN_CEILING_SAMPLES_48K - 1)
    } else if sample_rate == SAMPLE_RATE {
        MAX_SAMPLES.min(TOKEN_CEILING_SAMPLES - 1)
    } else {
        return None;
    };
    bonsai_audio::finite_sample_count(samples, previous, ceiling)
}

fn select_threads(requested: Option<usize>, available: usize) -> io::Result<usize> {
    let count = requested.unwrap_or_else(|| available.saturating_sub(2).clamp(1, 256));
    if count == 0 || count > 256 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Invalid speech CPU thread count.",
        ));
    }
    Ok(count)
}

fn model_path(path: &Path) -> io::Result<()> {
    if !path.is_absolute() || !path.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Native speech needs existing absolute GGUF model files.",
        ));
    }
    let text = path.to_str().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "Native speech requires UTF-8 paths.",
        )
    })?;
    if text.as_bytes().contains(&0) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Native speech requires regular file paths.",
        ));
    }
    Ok(())
}

fn bounded_input(text: &str, maximum: usize) -> io::Result<&str> {
    if text.trim().is_empty() || text.chars().take(maximum + 1).count() > maximum {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Speech text or voice description exceeds its bound.",
        ));
    }
    if text.as_bytes().contains(&0) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Speech text and voice descriptions cannot contain NUL.",
        ));
    }
    Ok(text)
}

/// Formats a sanitized synthesis error message without leaking untrusted text.
#[doc(hidden)]
#[must_use]
pub fn format_synthesis_failure(error: &io::Error) -> String {
    let stage = synthesis_failure_stage(error).unwrap_or("request execution");
    format!(
        "Qwen TTS synthesis failed at {stage}; error kind: {:?}.",
        error.kind()
    )
}

/// Formats a sanitized startup error message without leaking untrusted text.
#[doc(hidden)]
#[must_use]
pub fn format_startup_failure(stage: &'static str, error: &io::Error) -> String {
    format!(
        "Qwen TTS startup failed at {stage}; error kind: {:?}.",
        error.kind()
    )
}

#[cfg(test)]
mod tests {
    use super::{
        MAX_TEXT, MAX_VOICE, SAMPLE_RATE, SAMPLE_RATE_48K, SynthesisStage, TOKEN_CEILING_SAMPLES,
        TOKEN_CEILING_SAMPLES_48K, VoiceCache, bounded_input, finish_voice_bootstrap,
        is_supported_sample_rate, sample_count, sample_count_for_rate, select_threads,
        synthesis_failure_stage,
    };

    #[test]
    fn synthesis_failures_keep_typed_stage_and_kind_without_private_error_text() {
        use std::io;

        for (stage, label) in [
            (SynthesisStage::BaseWeights, "Base model weight intake"),
            (
                SynthesisStage::SpeakerExtraction,
                "speaker embedding extraction",
            ),
            (SynthesisStage::Generation, "speech token generation"),
            (SynthesisStage::CodecDecode, "speech audio decoding"),
            (SynthesisStage::AudioOutput, "speech audio output"),
        ] {
            let private = "private voice, request text, model path and credential";
            let error = stage.failure(io::Error::new(io::ErrorKind::PermissionDenied, private));
            assert_eq!(synthesis_failure_stage(&error), Some(label));
            assert_eq!(error.kind(), io::ErrorKind::PermissionDenied);
            assert_eq!(error.to_string(), label);
            assert!(!format!("{error:?}").contains(private));
            let error = SynthesisStage::Generation.failure(error);
            assert_eq!(synthesis_failure_stage(&error), Some(label));
            assert_eq!(error.kind(), io::ErrorKind::PermissionDenied);
        }
        let arbitrary = io::Error::other("speech token generation");
        assert_eq!(synthesis_failure_stage(&arbitrary), None);
    }

    #[test]
    fn voice_bootstrap_keeps_one_waveform_through_model_switch_and_releases_on_every_exit()
    -> std::io::Result<()> {
        use std::{cell::Cell, io, rc::Rc};

        struct Model(Rc<Cell<usize>>);
        impl Drop for Model {
            fn drop(&mut self) {
                self.0.set(self.0.get() - 1);
            }
        }
        struct Samples {
            data: Vec<f32>,
            releases: Rc<Cell<usize>>,
        }
        impl AsRef<[f32]> for Samples {
            fn as_ref(&self) -> &[f32] {
                &self.data
            }
        }
        impl Drop for Samples {
            fn drop(&mut self) {
                self.releases.set(self.releases.get() + 1);
            }
        }

        // Success, Base initialization failure, then extraction failure.
        for failure in 0..3 {
            let models = Rc::new(Cell::new(1));
            let releases = Rc::new(Cell::new(0));
            let samples = Samples {
                data: vec![0.25, -0.5, 1.0],
                releases: Rc::clone(&releases),
            };
            let original = samples.data.as_ptr();
            let extracted = Cell::new(false);
            let result = finish_voice_bootstrap(
                Model(Rc::clone(&models)),
                samples,
                || {
                    assert_eq!(models.get(), 0, "Design must be released before Base loads");
                    assert_eq!(releases.get(), 0, "Base load still owns the waveform");
                    if failure == 1 {
                        return Err(io::Error::other("injected Base load failure"));
                    }
                    models.set(1);
                    Ok(Model(Rc::clone(&models)))
                },
                |_, samples| {
                    extracted.set(true);
                    assert_eq!(models.get(), 1);
                    assert_eq!(releases.get(), 0);
                    assert_eq!(samples.as_ptr(), original);
                    assert_eq!(samples, [0.25, -0.5, 1.0]);
                    if failure == 2 {
                        return Err(io::Error::other("injected extraction failure"));
                    }
                    Ok(42)
                },
            );
            assert_eq!(releases.get(), 1);
            assert_eq!(extracted.get(), failure != 1);
            if failure == 0 {
                let (base, reference) = result?;
                assert_eq!(reference, 42);
                assert_eq!(models.get(), 1);
                drop(base);
            } else {
                assert!(result.is_err());
            }
            assert_eq!(models.get(), 0);
            assert_eq!(releases.get(), 1);
        }
        Ok(())
    }

    #[test]
    fn selected_voice_is_reused_until_profile_changes_and_failures_never_fallback()
    -> std::io::Result<()> {
        use std::{cell::Cell, io, rc::Rc};
        struct FakeVoice {
            identity: usize,
            alive: Rc<Cell<usize>>,
        }
        impl Drop for FakeVoice {
            fn drop(&mut self) {
                self.alive.set(self.alive.get() - 1);
            }
        }
        let alive = Rc::new(Cell::new(0));
        let acquisitions = Cell::new(0);
        let acquire = || {
            assert_eq!(alive.get(), 0, "Two speech model contexts coexist");
            acquisitions.set(acquisitions.get() + 1);
            alive.set(1);
            Ok(FakeVoice {
                identity: acquisitions.get(),
                alive: Rc::clone(&alive),
            })
        };
        let mut cache = VoiceCache(None);
        // Neither new utterances, paragraphs nor complete/stream output mode
        // are cache inputs: they must all use the selected identity unchanged.
        for _utterance in [
            "First paragraph.",
            "Second paragraph.\n\nA line break.",
            "Another message.",
        ] {
            assert_eq!(cache.get_or_create("Warm narrator", acquire)?.identity, 1);
        }
        assert_eq!(acquisitions.get(), 1);
        assert_eq!(cache.get_or_create("Clear alto", acquire)?.identity, 2);
        assert!(
            cache
                .get_or_create("New profile", || Err(io::Error::other(
                    "synthetic acquisition failure"
                )))
                .is_err()
        );
        assert_eq!(alive.get(), 0);
        assert!(
            cache.0.is_none(),
            "Failed voice acquisition left an unrelated fallback"
        );
        assert_eq!(cache.get_or_create("Warm narrator", acquire)?.identity, 3);
        drop(cache);
        assert_eq!(alive.get(), 0);
        Ok(())
    }

    #[test]
    fn derived_reference_cache_avoids_rederiving_on_voice_switches_and_isolates_models()
    -> std::io::Result<()> {
        use std::{cell::Cell, fs, io, rc::Rc, sync::Mutex, time::UNIX_EPOCH};

        struct FakeTalker {
            alive: Rc<Cell<usize>>,
        }
        impl Drop for FakeTalker {
            fn drop(&mut self) {
                self.alive.set(self.alive.get() - 1);
            }
        }
        struct FakeVoice {
            talker: FakeTalker,
            reference: crate::rust_talker::Reference,
        }
        impl super::VoiceParts for FakeVoice {
            type Talker = FakeTalker;
            fn into_talker(self) -> Self::Talker {
                self.talker
            }
            fn from_parts(talker: Self::Talker, reference: crate::rust_talker::Reference) -> Self {
                Self { talker, reference }
            }
        }

        let stamp = std::time::SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(io::Error::other)?
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("Bonsai-TTS-RefCache-{stamp}"));
        fs::create_dir(&dir)?;
        let design_path = dir.join("design.gguf");
        let base_path = dir.join("base.gguf");
        fs::write(&design_path, b"design-q8-bytes")?;
        fs::write(&base_path, b"base-q8-bytes")?;
        let q8_sha = super::model_identity_sha256(&design_path, &base_path, &mut || Ok(()))?;

        let alive_base = Rc::new(Cell::new(0));
        let alive_design = Rc::new(Cell::new(1));
        let design_runs = Cell::new(0_usize);
        let base_loads = Cell::new(0_usize);

        let mut cache = VoiceCache::<FakeVoice>(None);
        let mut preloaded_design = Some(FakeTalker {
            alive: Rc::clone(&alive_design),
        });
        let ref_cache = Mutex::new(super::DerivedReferenceCache::new());

        let select = |cache: &mut VoiceCache<FakeVoice>,
                      design: &mut Option<FakeTalker>,
                      profile: &str,
                      d_path: &std::path::Path,
                      b_path: &std::path::Path|
         -> io::Result<(Vec<f32>, Vec<[i32; 16]>)> {
            let voice = cache.get_or_acquire(
                profile,
                design,
                &ref_cache,
                || super::model_identity_sha256(d_path, b_path, &mut || Ok(())),
                || {
                    assert_eq!(alive_design.get(), 0, "Design and Base never coexist");
                    assert_eq!(alive_base.get(), 0, "Two Base models never coexist");
                    base_loads.set(base_loads.get() + 1);
                    alive_base.set(alive_base.get() + 1);
                    Ok(FakeTalker {
                        alive: Rc::clone(&alive_base),
                    })
                },
                |design_slot| {
                    assert_eq!(alive_base.get(), 0, "Base must drop before Design runs");
                    let _taken = design_slot.take();
                    design_runs.set(design_runs.get() + 1);
                    let run_id = design_runs.get();
                    base_loads.set(base_loads.get() + 1);
                    alive_base.set(alive_base.get() + 1);
                    let speaker = vec![f32::from(u8::try_from(run_id).unwrap_or(1))];
                    let codes = vec![[i32::try_from(run_id).unwrap_or(1); 16]];
                    Ok((
                        FakeTalker {
                            alive: Rc::clone(&alive_base),
                        },
                        crate::rust_talker::Reference { speaker, codes },
                    ))
                },
            )?;
            Ok((
                voice.reference.speaker.clone(),
                voice.reference.codes.clone(),
            ))
        };

        // Voice A cold -> runs VoiceDesign + Base load.
        let (spk_a, codes_a) = select(
            &mut cache,
            &mut preloaded_design,
            "Warm narrator",
            &design_path,
            &base_path,
        )?;
        assert_eq!(design_runs.get(), 1);
        assert_eq!(base_loads.get(), 1);
        assert_eq!(alive_design.get(), 0);
        assert_eq!(alive_base.get(), 1);

        // Voice B cold -> drops Base, runs VoiceDesign + Base load.
        let (spk_b, codes_b) = select(
            &mut cache,
            &mut preloaded_design,
            "Clear alto",
            &design_path,
            &base_path,
        )?;
        assert_ne!(spk_a, spk_b);
        assert_ne!(codes_a, codes_b);
        assert_eq!(design_runs.get(), 2);
        assert_eq!(base_loads.get(), 2);
        assert_eq!(alive_base.get(), 1);

        // Switch back to Voice A -> hits derived cache AND reuses loaded Base Talker.
        let (spk_a2, codes_a2) = select(
            &mut cache,
            &mut preloaded_design,
            "Warm narrator",
            &design_path,
            &base_path,
        )?;
        assert_eq!(spk_a2, spk_a);
        assert_eq!(codes_a2, codes_a);
        assert_eq!(design_runs.get(), 2);
        assert_eq!(base_loads.get(), 2);
        assert_eq!(alive_base.get(), 1);

        // Clearing active voice (e.g. after failure) then requesting Voice A again skips VoiceDesign,
        // only reloading Base.
        cache.0 = None;
        assert_eq!(alive_base.get(), 0);
        let (spk_a3, codes_a3) = select(
            &mut cache,
            &mut preloaded_design,
            "Warm narrator",
            &design_path,
            &base_path,
        )?;
        assert_eq!(spk_a3, spk_a);
        assert_eq!(codes_a3, codes_a);
        assert_eq!(design_runs.get(), 2);
        assert_eq!(base_loads.get(), 3);

        // Changing model bytes (e.g. loading a new Q4 Synthesizer) changes model_sha256 -> cache misses!
        drop(cache);
        assert_eq!(alive_base.get(), 0);
        fs::write(&design_path, b"design-q4-bytes")?;
        let q4_sha = super::model_identity_sha256(&design_path, &base_path, &mut || Ok(()))?;
        assert_ne!(q4_sha, q8_sha);
        let mut q4_cache = VoiceCache::<FakeVoice>(None);
        let (spk_a_q4, _) = select(
            &mut q4_cache,
            &mut preloaded_design,
            "Warm narrator",
            &design_path,
            &base_path,
        )?;
        assert_ne!(spk_a_q4, spk_a);
        assert_eq!(design_runs.get(), 3);

        drop(q4_cache);
        assert_eq!(alive_base.get(), 0);
        fs::remove_file(&design_path)?;
        fs::remove_file(&base_path)?;
        fs::remove_dir(&dir)?;

        // Verify LRU eviction capacity bound on DerivedReferenceCache.
        let mut lru = super::DerivedReferenceCache::new();
        for i in 0..=super::MAX_DERIVED_REFERENCES {
            lru.insert(
                &format!("voice-{i}"),
                [1_u8; 32],
                super::DerivedReference {
                    speaker_embedding: vec![f32::from(u8::try_from(i).unwrap_or(0))],
                    reference_codes: vec![[i32::try_from(i).unwrap_or(0); 16]],
                },
            );
        }
        assert!(lru.get("voice-0", &[1_u8; 32]).is_none());
        assert!(
            lru.get(
                &format!("voice-{}", super::MAX_DERIVED_REFERENCES),
                &[1_u8; 32]
            )
            .is_some()
        );
        Ok(())
    }

    #[test]
    fn complete_and_streamed_samples_share_token_ceiling_and_finite_validation() {
        assert!(is_supported_sample_rate(SAMPLE_RATE));
        assert!(is_supported_sample_rate(SAMPLE_RATE_48K));
        assert!(!is_supported_sample_rate(16_000));
        assert_eq!(sample_count(&[0.0, 2.0], 0), Some(2));
        assert_eq!(
            sample_count(&[0.0], TOKEN_CEILING_SAMPLES - 2),
            Some(TOKEN_CEILING_SAMPLES - 1)
        );
        assert_eq!(sample_count(&[0.0], TOKEN_CEILING_SAMPLES - 1), None);
        assert_eq!(
            sample_count_for_rate(&[0.0], TOKEN_CEILING_SAMPLES_48K - 2, SAMPLE_RATE_48K),
            Some(TOKEN_CEILING_SAMPLES_48K - 1)
        );
        assert_eq!(
            sample_count_for_rate(&[0.0], TOKEN_CEILING_SAMPLES_48K - 1, SAMPLE_RATE_48K),
            None
        );
        assert_eq!(sample_count_for_rate(&[0.0], 0, 16_000), None);
        assert_eq!(sample_count(&[0.0], usize::MAX), None);
        for samples in [&[][..], &[f32::NAN], &[f32::INFINITY], &[f32::NEG_INFINITY]] {
            assert_eq!(sample_count(samples, 0), None);
            assert_eq!(sample_count(samples, 1), None);
        }
    }

    #[test]
    fn cpu_threads_reserve_ui_capacity_and_bound_explicit_overrides() -> std::io::Result<()> {
        assert_eq!(select_threads(None, 12)?, 10);
        assert_eq!(select_threads(None, 1)?, 1);
        assert_eq!(select_threads(Some(12), 12)?, 12);
        assert!(select_threads(Some(0), 12).is_err());
        assert_eq!(select_threads(Some(16), 12)?, 16);
        assert!(select_threads(Some(257), 512).is_err());
        Ok(())
    }

    #[test]
    fn voice_design_input_bounds_count_unicode_without_truncating() {
        for (limit, text) in [
            (MAX_TEXT, "界".repeat(MAX_TEXT)),
            (MAX_VOICE, "🎙".repeat(MAX_VOICE)),
        ] {
            assert_eq!(bounded_input(&text, limit).ok(), Some(text.as_str()));
            assert!(bounded_input(&(text + "x"), limit).is_err());
        }
        assert!(bounded_input(" \n\t", MAX_TEXT).is_err());
        assert!(bounded_input("voice\0description", MAX_VOICE).is_err());
    }

    #[test]
    fn synthesis_failure_retains_kind_without_untrusted_error_text() {
        let error = std::io::Error::new(
            std::io::ErrorKind::BrokenPipe,
            "private prompt, voice description and C:/private/credential.key",
        );
        let message = super::format_synthesis_failure(&error);
        assert!(message.contains("request execution"));
        assert!(message.contains("BrokenPipe"));
        assert!(!message.contains("private"));
        assert!(!message.contains("credential"));
    }

    #[test]
    fn startup_failure_retains_stage_and_kind_without_private_error_text() {
        let error = std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "private prompt, voice description and C:/private/credential.key",
        );
        let message = super::format_startup_failure("Rust model loading", &error);
        assert!(message.contains("Rust model loading"));
        assert!(message.contains("InvalidData"));
        assert!(!message.contains("private"));
        assert!(!message.contains("credential"));
    }
}
