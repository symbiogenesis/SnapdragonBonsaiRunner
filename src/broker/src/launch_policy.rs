//! Reviewed development-only backend policies compiled into the broker.
//!
//! This plan is not permission to launch and does not qualify the daily profile.

use serde::Deserialize;
use std::{
    ffi::OsString,
    fmt,
    path::{Path, PathBuf},
    sync::LazyLock,
};

const LOCK_JSON: &str = include_str!("../../../config/runtime.lock.json");
const PROFILE_JSON: &str = include_str!("../../../config/profiles/reference-cpu.json");
const SWIFT_LOCK_JSON: &str = include_str!("../../../config/swift-assessment.lock.json");
const PTQ1_EXPORT_JSON: &str = include_str!("../../../config/swift.ptq1-export.json");
/// Fixed process memory ceiling: 18 GiB.
pub const MEMORY_LIMIT_BYTES: usize = bonsai_cpu_policy::MEMORY_LIMIT_BYTES;
/// Required available physical memory including a 4 GiB host reserve.
pub const REQUIRED_AVAILABLE_BYTES: u64 = bonsai_cpu_policy::REQUIRED_AVAILABLE_BYTES;

/// The only reviewed development model/runtime selections. Neither qualifies daily use.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LaunchProfile {
    /// Default Qwen reference; its compiled JSON profile remains launch-disabled.
    #[default]
    ReferenceCpu,
    /// Explicit Swift PQ2 assessment, optionally with the pinned abliteration `LoRA`.
    SwiftPq2Assessment,
}
impl LaunchProfile {
    /// Exact command-line name of the compiled selection.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::ReferenceCpu => "reference-cpu",
            Self::SwiftPq2Assessment => "swift-pq2-assessment",
        }
    }

    /// Job-wide committed-memory ceiling; the physical reserve is additional.
    #[must_use]
    pub const fn memory_limit_bytes(self) -> usize {
        self.cpu_profile().memory_limit_bytes()
    }

    /// Physical admission threshold, including a separate 4 GiB host reserve.
    #[must_use]
    pub const fn required_available_bytes(self) -> u64 {
        self.cpu_profile().required_available_bytes()
    }

    const fn cpu_profile(self) -> bonsai_cpu_policy::Profile {
        match self {
            Self::ReferenceCpu => bonsai_cpu_policy::Profile::ReferenceCpu,
            Self::SwiftPq2Assessment => bonsai_cpu_policy::Profile::SwiftPq2Assessment,
        }
    }
}
impl std::str::FromStr for LaunchProfile {
    type Err = PolicyError;
    fn from_str(name: &str) -> Result<Self, Self::Err> {
        match name {
            "reference-cpu" => Ok(Self::ReferenceCpu),
            "swift-pq2-assessment" => Ok(Self::SwiftPq2Assessment),
            _ => Err(PolicyError::Configuration),
        }
    }
}

/// A value-free policy failure; inherited environment values are never retained.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolicyError {
    /// Compiled artifact identities or profile need review.
    Configuration,
    /// An inherited model/runtime override or non-Unicode name was present.
    Environment,
    /// Storage is not an absolute external path without redirects.
    Path,
    /// The backend port is privileged.
    Port,
}
impl fmt::Display for PolicyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Configuration => "Unreviewed compiled launch configuration.",
            Self::Environment => "Inherited model/runtime environment overrides are unsupported.",
            Self::Path => "An absolute external path without Git or reparse ancestors is required.",
            Self::Port => "A nonprivileged backend port is required.",
        })
    }
}
impl std::error::Error for PolicyError {}

/// Reject inherited runtime/model overrides using names alone.
///
/// # Errors
/// Returns [`PolicyError::Environment`] for forbidden or non-Unicode names.
pub fn validate_environment(names: impl IntoIterator<Item = OsString>) -> Result<(), PolicyError> {
    for name in names {
        let name = name.to_str().ok_or(PolicyError::Environment)?;
        if bonsai_cpu_policy::server::forbidden_environment_name(name) {
            return Err(PolicyError::Environment);
        }
    }
    Ok(())
}

/// Verified stackable multi-rank DaisyRose `LoRA` artifact selected for a native Swift launch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DaisyRoseLoraPlan {
    /// Verified external filesystem path to the adapter GGUF.
    pub path: PathBuf,
    /// Exact byte length of the verified GGUF file.
    pub bytes: u64,
    /// Lowercase 64-hex SHA-256 of the verified GGUF file.
    pub sha256: String,
    /// Maximum factor rank across all verified target matrices.
    pub max_rank: u16,
}

impl DaisyRoseLoraPlan {
    /// Total host memory reserve (`bytes + DAISYROSE_SCRATCH_HEADROOM_BYTES`) budgeted for this adapter.
    #[must_use]
    pub const fn reserve_bytes(&self) -> Option<u64> {
        self.bytes
            .checked_add(bonsai_cpu_policy::DAISYROSE_SCRATCH_HEADROOM_BYTES)
    }
}

/// Fixed, separately tokenized backend invocation; callers cannot add arguments.
#[derive(Debug)]
pub struct LaunchPlan {
    /// Locked installed ARM64 server path.
    pub executable: PathBuf,
    /// Arguments passed directly to the executable, never a shell.
    pub arguments: Vec<OsString>,
    /// Whole-model path verified before launch.
    pub model_path: PathBuf,
    /// Optional verified vision projector identity for Swift multimodal launch.
    pub mmproj: Option<bonsai_cpu_policy::SwiftMmprojArtifact>,
    /// Optional pinned adapter identity; never an arbitrary caller-supplied path.
    pub adapter: Option<bonsai_cpu_policy::AbliterateLoraArtifact>,
    /// Optional verified stackable DaisyRose multi-rank adapter plan.
    pub daisyrose_lora: Option<DaisyRoseLoraPlan>,
    /// Pinned archive path verified before launch.
    pub archive_path: PathBuf,
    /// Complete installed runtime directory.
    pub runtime_directory: PathBuf,
    /// Owner-private backend credential path supplied by the credential owner.
    pub key_file: PathBuf,
    /// Available memory admission threshold, including the host reserve.
    pub required_available_bytes: u64,
    /// Job private-commit memory ceiling in bytes.
    pub memory_limit_bytes: usize,
    /// Process working-set memory ceiling in bytes (including mapped read-only weight sections).
    pub working_set_limit_bytes: usize,
    /// Immutable prompt, response and template bound selected for the model.
    pub context_tokens: usize,
    /// Optional native CPU tiled-prefill selection `(tokens, scratch_bytes)`.
    pub cpu_prefill: Option<(u8, usize)>,
    /// Optional native CPU suffix-speculation selection `(draft_max, verification_scratch_bytes, checkpoint_bytes)`.
    pub cpu_speculation: Option<(u8, usize, usize)>,
}
impl LaunchPlan {
    /// Base process memory limit passed to the native child's `--memory-limit-bytes`.
    ///
    /// When `mmproj` or `daisyrose_lora` is enabled, the planner includes
    /// [`bonsai_cpu_policy::MMPROJ_RESERVE_BYTES`] and `daisyrose.reserve_bytes()` in
    /// [`Self::working_set_limit_bytes`] (and [`Self::memory_limit_bytes`]), while the native
    /// child adds its own vision and multi-rank reserves on top of `--memory-limit-bytes + 4 GiB`
    /// during startup admission.
    #[must_use]
    pub fn native_child_memory_limit_bytes(&self) -> usize {
        let mut reserve_u64 = 0_u64;
        if self.mmproj.is_some() {
            reserve_u64 = reserve_u64.saturating_add(bonsai_cpu_policy::MMPROJ_RESERVE_BYTES);
        }
        if let Some(daisyrose) = &self.daisyrose_lora
            && let Some(bytes) = daisyrose.reserve_bytes()
        {
            reserve_u64 = reserve_u64.saturating_add(bytes);
        }
        if reserve_u64 == 0 {
            self.working_set_limit_bytes
        } else {
            usize::try_from(reserve_u64)
                .ok()
                .and_then(|reserve| self.working_set_limit_bytes.checked_sub(reserve))
                .unwrap_or(self.working_set_limit_bytes)
        }
    }

    /// Record a verified stackable DaisyRose multi-rank `LoRA` adapter on this launch plan.
    ///
    /// # Errors
    /// Rejects duplicate registration or unsafe external paths.
    #[cfg(windows)]
    pub(crate) fn enable_daisyrose_lora(
        &mut self,
        adapter: DaisyRoseLoraPlan,
    ) -> Result<(), PolicyError> {
        if self.daisyrose_lora.is_some() {
            return Err(PolicyError::Configuration);
        }
        validate_external_path(&adapter.path)?;
        let _path_text = adapter.path.to_str().ok_or(PolicyError::Path)?;
        self.daisyrose_lora = Some(adapter);
        Ok(())
    }

    /// Apply a trusted memory-derived Swift context to this interactive launch.
    /// Fixed qualification argument builders remain unchanged.
    #[cfg(windows)]
    pub(crate) fn select_swift_context(
        &mut self,
        context: bonsai_cpu_policy::context::SwiftContextPlan,
    ) -> Result<(), PolicyError> {
        let index = self
            .arguments
            .iter()
            .position(|argument| argument == "--ctx-size")
            .ok_or(PolicyError::Configuration)?;
        let argument = self
            .arguments
            .get_mut(index + 1)
            .ok_or(PolicyError::Configuration)?;
        *argument = context.context_tokens.to_string().into();
        self.context_tokens = context.context_tokens;
        self.memory_limit_bytes = context.memory_limit_bytes;
        self.working_set_limit_bytes = context.memory_limit_bytes;
        self.required_available_bytes = context.required_available_bytes;
        Ok(())
    }

    /// Apply a calibrated CoW Swift context splitting private Job commit from physical working set.
    #[cfg(windows)]
    pub(crate) fn select_calibrated_swift_context(
        &mut self,
        calibrated: &bonsai_cpu_policy::context::SwiftCalibratedPlan,
    ) -> Result<(), PolicyError> {
        self.select_swift_context(calibrated.context_plan)?;
        self.working_set_limit_bytes = calibrated.working_set_limit_bytes;
        Ok(())
    }

    /// Enable the verified Swift vision projector (`mmproj`) on this interactive launch.
    #[cfg(windows)]
    pub(crate) fn enable_swift_mmproj(
        &mut self,
        mmproj: bonsai_cpu_policy::SwiftMmprojArtifact,
    ) -> Result<(), PolicyError> {
        if self.arguments.iter().any(|argument| argument == "--mmproj") {
            return Err(PolicyError::Configuration);
        }
        validate_external_path(&mmproj.path)?;
        let _path_text = mmproj.path.to_str().ok_or(PolicyError::Path)?;
        let index = self
            .arguments
            .iter()
            .position(|argument| argument == "--no-mmproj")
            .ok_or(PolicyError::Configuration)?;
        self.arguments.splice(
            index..=index,
            [
                OsString::from("--mmproj"),
                mmproj.path.as_os_str().to_owned(),
                OsString::from("--no-mmproj-offload"),
                OsString::from("--image-max-tokens"),
                OsString::from(bonsai_cpu_policy::DEFAULT_IMAGE_MAX_TOKENS.to_string()),
            ],
        );
        self.mmproj = Some(mmproj);
        Ok(())
    }

    /// Build the sole reviewed development plan. This never launches or writes.
    /// The separate launcher must require its explicit development opt-in.
    ///
    /// # Errors
    /// Rejects invalid paths, ports, inherited overrides or compiled policy drift.
    pub fn reference(data: &Path, backend_port: u16, key_file: &Path) -> Result<Self, PolicyError> {
        Self::for_profile(data, backend_port, key_file, LaunchProfile::ReferenceCpu)
    }

    /// Build one compiled development plan; no model, runtime or argument overrides exist.
    ///
    /// # Errors
    /// Rejects invalid paths, ports, inherited overrides or compiled policy drift.
    pub fn for_profile(
        data: &Path,
        backend_port: u16,
        key_file: &Path,
        profile: LaunchProfile,
    ) -> Result<Self, PolicyError> {
        Self::for_profile_with_lora(data, backend_port, key_file, profile, false)
    }

    /// Build a fixed server plan with the pinned scale-one `LoRA` when requested.
    ///
    /// # Errors
    /// Rejects `LoRA` on the reference profile, invalid paths or compiled policy drift.
    pub fn for_profile_with_lora(
        data: &Path,
        backend_port: u16,
        key_file: &Path,
        profile: LaunchProfile,
        use_abliterate_lora: bool,
    ) -> Result<Self, PolicyError> {
        Self::for_profile_with_options(
            data,
            backend_port,
            key_file,
            profile,
            use_abliterate_lora,
            false,
        )
    }

    /// Build a fixed server plan with explicit pinned adapter and weight packing selections.
    /// The PTQ1 option selects only the verified current-weight export, never the older checkpoint.
    ///
    /// # Errors
    /// Rejects Swift-only options on the reference profile, invalid paths or compiled policy drift.
    pub fn for_profile_with_options(
        data: &Path,
        backend_port: u16,
        key_file: &Path,
        profile: LaunchProfile,
        use_abliterate_lora: bool,
        use_repacked_ptq1: bool,
    ) -> Result<Self, PolicyError> {
        if use_abliterate_lora && profile != LaunchProfile::SwiftPq2Assessment {
            return Err(PolicyError::Configuration);
        }
        validate_environment(std::env::vars_os().map(|(name, _)| name))?;
        let pins = reviewed_pins_for_representation(profile, use_repacked_ptq1)?;
        validate_external_path(data)?;
        validate_external_path(key_file)?;
        if backend_port < 1024 {
            return Err(PolicyError::Port);
        }
        let runtime_directory = data.join("runtime").join(&pins.runtime.commit);
        // The pinned release archive places the server at the archive root.
        let executable = runtime_directory.join("llama-server.exe");
        let model_path = data
            .join("models")
            .join(&pins.model.revision)
            .join(&pins.model.filename);
        let archive_path = data.join("downloads").join(&pins.runtime.asset.filename);
        for path in [&executable, &model_path, &archive_path] {
            validate_external_path(path)?;
        }
        let mut arguments: Vec<OsString> = bonsai_cpu_policy::server::arguments(
            model_path.to_str().ok_or(PolicyError::Path)?,
            key_file.to_str().ok_or(PolicyError::Path)?,
            backend_port,
            profile.cpu_profile(),
        )
        .map_err(|_| PolicyError::Configuration)?
        .into_iter()
        .map(OsString::from)
        .collect();
        let adapter = if use_abliterate_lora {
            let adapter = bonsai_cpu_policy::abliterate_lora_artifact(data)
                .map_err(|_| PolicyError::Configuration)?;
            validate_external_path(&adapter.path)?;
            // --lora applies scale 1 and accepts an absolute Windows path as one
            // argument; --lora-scaled would split the drive-letter colon.
            arguments.extend([
                OsString::from("--lora"),
                adapter.path.as_os_str().to_owned(),
            ]);
            Some(adapter)
        } else {
            None
        };
        Ok(Self {
            executable,
            arguments,
            model_path,
            mmproj: None,
            adapter,
            daisyrose_lora: None,
            archive_path,
            runtime_directory,
            key_file: key_file.to_path_buf(),
            required_available_bytes: profile.required_available_bytes(),
            memory_limit_bytes: profile.memory_limit_bytes(),
            working_set_limit_bytes: profile.memory_limit_bytes(),
            context_tokens: crate::CONTEXT_TOKENS,
            cpu_prefill: None,
            cpu_speculation: None,
        })
    }

    /// Build a launch plan for an opt-in native `NariQ` `GGUF` variant (never selects Orca or Swift `mmproj`).
    ///
    /// # Errors
    /// Rejects invalid external paths, privileged ports, or compiled policy drift.
    pub fn for_swift_nariq(
        data: &Path,
        backend_port: u16,
        key_file: &Path,
        variant: bonsai_cpu_policy::NariqVariant,
    ) -> Result<Self, PolicyError> {
        validate_environment(std::env::vars_os().map(|(name, _)| name))?;
        validate_external_path(data)?;
        validate_external_path(key_file)?;
        if backend_port < 1024 {
            return Err(PolicyError::Port);
        }
        let pins = reviewed_pins(LaunchProfile::SwiftPq2Assessment)?;
        let nariq = bonsai_cpu_policy::nariq_model_artifact(data, variant)
            .map_err(|_| PolicyError::Configuration)?;
        let runtime_directory = data.join("runtime").join(&pins.runtime.commit);
        let executable = runtime_directory.join("llama-server.exe");
        let archive_path = data.join("downloads").join(&pins.runtime.asset.filename);
        for path in [&executable, &nariq.path, &archive_path] {
            validate_external_path(path)?;
        }
        let arguments: Vec<OsString> = bonsai_cpu_policy::server::arguments(
            nariq.path.to_str().ok_or(PolicyError::Path)?,
            key_file.to_str().ok_or(PolicyError::Path)?,
            backend_port,
            LaunchProfile::SwiftPq2Assessment.cpu_profile(),
        )
        .map_err(|_| PolicyError::Configuration)?
        .into_iter()
        .map(OsString::from)
        .collect();
        let fixed_runtime = bonsai_cpu_policy::context::SwiftWeightPacking::from_nariq(variant)
            .calibrated_fixed_runtime_bytes();
        let memory_limit_bytes = usize::try_from(fixed_runtime)
            .unwrap_or(LaunchProfile::SwiftPq2Assessment.memory_limit_bytes());
        Ok(Self {
            executable,
            arguments,
            model_path: nariq.path,
            mmproj: None,
            adapter: None,
            daisyrose_lora: None,
            archive_path,
            runtime_directory,
            key_file: key_file.to_path_buf(),
            required_available_bytes: fixed_runtime
                .saturating_add(bonsai_cpu_policy::context::HOST_RESERVE_BYTES),
            memory_limit_bytes,
            working_set_limit_bytes: memory_limit_bytes,
            context_tokens: crate::CONTEXT_TOKENS,
            cpu_prefill: None,
            cpu_speculation: None,
        })
    }
}

// Struct deserialization rejects duplicate and unknown fields. Object wrappers
// also reject Serde's positional-array representation at every object boundary.
// Avoid `#[serde(flatten)]` here because flattening buffers keys and weakens duplicate rejection.
fn parse_json_object<T: serde::de::DeserializeOwned>(input: &str) -> Result<T, PolicyError> {
    let super::Object(value): super::Object<T> =
        serde_json::from_str(input).map_err(|_| PolicyError::Configuration)?;
    Ok(value)
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct Pins {
    schema_version: u8,
    decision: String,
    pub(crate) model: super::Object<ModelPin>,
    pub(crate) runtime: super::Object<RuntimePin>,
    fallback: super::Object<Fallback>,
    qualification: super::Object<Qualification>,
}

#[derive(PartialEq, Eq)]
struct ModelPinFields<'a> {
    repository: &'a str,
    revision: &'a str,
    filename: &'a str,
    bytes: u64,
    sha256: &'a str,
    digest_source: &'a str,
    license: &'a str,
    architecture: &'a str,
}

fn validate_chunk_manifest(
    bytes: u64,
    chunk_bytes: Option<usize>,
    chunk_sha256s: Option<Vec<String>>,
) -> Result<Option<(usize, Vec<String>)>, PolicyError> {
    let manifest = bonsai_cpu_policy::validate_chunk_manifest(bytes, chunk_bytes, chunk_sha256s)
        .map_err(|_| PolicyError::Configuration)?;
    #[cfg(windows)]
    if let Some((chunk_bytes, _)) = &manifest {
        let chunk_bytes_u64 =
            u64::try_from(*chunk_bytes).map_err(|_| PolicyError::Configuration)?;
        if chunk_bytes_u64 < bonsai_windows_credentials::MIN_MAPPED_HASH_BYTES {
            return Err(PolicyError::Configuration);
        }
    }
    Ok(manifest)
}

macro_rules! define_model_pin {
    ($vis:vis struct $name:ident $(($fvis:vis))? { $($extra:tt)* }) => {
        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase", deny_unknown_fields)]
        $vis struct $name {
            repository: String,
            $($fvis)? revision: String,
            $($fvis)? filename: String,
            $($fvis)? bytes: u64,
            $($fvis)? sha256: String,
            digest_source: String,
            license: String,
            architecture: String,
            #[serde(default)]
            chunk_bytes: Option<usize>,
            #[serde(default)]
            chunk_sha256s: Option<Vec<String>>,
            $($extra)*
        }

        impl $name {
            fn fields(&self) -> ModelPinFields<'_> {
                ModelPinFields {
                    repository: &self.repository,
                    revision: &self.revision,
                    filename: &self.filename,
                    bytes: self.bytes,
                    sha256: &self.sha256,
                    digest_source: &self.digest_source,
                    license: &self.license,
                    architecture: &self.architecture,
                }
            }

            fn into_artifact(self) -> Result<ModelArtifact, PolicyError> {
                let chunk_manifest =
                    validate_chunk_manifest(self.bytes, self.chunk_bytes, self.chunk_sha256s)?;
                Ok(ModelArtifact {
                    revision: self.revision,
                    filename: self.filename,
                    bytes: self.bytes,
                    sha256: self.sha256,
                    chunk_manifest,
                })
            }
        }
    };
}

define_model_pin!(
    struct ModelPinCommon {}
);
define_model_pin!(pub(crate) struct ModelPin (pub(crate)) {
    expected_metadata: super::Object<Metadata>,
});

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Metadata {
    block_count_including_mtp: u32,
    mtp_blocks: u32,
    full_attention_interval: u32,
    kv_heads: u32,
    key_length: u32,
    value_length: u32,
    ssm_state_size: u32,
    ssm_inner_size: u32,
    ssm_group_count: u32,
    ssm_convolution_kernel: u32,
}

#[derive(PartialEq, Eq)]
struct RuntimePinFields<'a> {
    repository: &'a str,
    commit: &'a str,
    release: &'a str,
    license: &'a str,
    asset_filename: &'a str,
    asset_bytes: u64,
    asset_sha256: &'a str,
    asset_url: &'a str,
    asset_digest_source: &'a str,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RuntimePin {
    repository: String,
    pub(crate) commit: String,
    release: String,
    license: String,
    pub(crate) asset: super::Object<AssetPin>,
}

impl RuntimePin {
    fn fields(&self) -> RuntimePinFields<'_> {
        RuntimePinFields {
            repository: &self.repository,
            commit: &self.commit,
            release: &self.release,
            license: &self.license,
            asset_filename: &self.asset.filename,
            asset_bytes: self.asset.bytes,
            asset_sha256: &self.asset.sha256,
            asset_url: &self.asset.url,
            asset_digest_source: &self.asset.digest_source,
        }
    }
}

#[derive(Clone, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct AssetPin {
    pub(crate) filename: String,
    url: String,
    pub(crate) bytes: u64,
    pub(crate) sha256: String,
    digest_source: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Fallback {
    repository: String,
    revision: String,
    filename: String,
    bytes: u64,
    sha256: String,
    status: String,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Qualification {
    source_compatibility: String,
    model_download: String,
    full_gguf_inspection: String,
    cpu_loading: String,
    daily_security: String,
    device_performance: String,
}

impl<T> std::ops::Deref for super::Object<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}

#[derive(Clone)]
pub(crate) struct ReviewedPins {
    pub(crate) model: ModelArtifact,
    pub(crate) runtime: super::Object<RuntimePin>,
}
#[derive(Clone)]
pub(crate) struct ModelArtifact {
    pub(crate) revision: String,
    pub(crate) filename: String,
    pub(crate) bytes: u64,
    pub(crate) sha256: String,
    pub(crate) chunk_manifest: Option<(usize, Vec<String>)>,
}

pub(crate) fn reviewed_pins(profile: LaunchProfile) -> Result<ReviewedPins, PolicyError> {
    static REFERENCE_CPU: LazyLock<Result<ReviewedPins, PolicyError>> = LazyLock::new(|| {
        let pins = parse_pins(LOCK_JSON)?;
        validate_profile(PROFILE_JSON)?;
        Ok(ReviewedPins {
            model: pins.model.0.into_artifact()?,
            runtime: pins.runtime,
        })
    });
    static SWIFT_PQ2: LazyLock<Result<ReviewedPins, PolicyError>> =
        LazyLock::new(|| parse_swift_pins(SWIFT_LOCK_JSON));
    match profile {
        LaunchProfile::ReferenceCpu => REFERENCE_CPU.clone(),
        LaunchProfile::SwiftPq2Assessment => SWIFT_PQ2.clone(),
    }
}

fn select_ptq1(profile: LaunchProfile) -> Result<SelectedModel, PolicyError> {
    let lock = serde_json::from_str(SWIFT_LOCK_JSON).map_err(|_| PolicyError::Configuration)?;
    let evidence =
        serde_json::from_str(PTQ1_EXPORT_JSON).map_err(|_| PolicyError::Configuration)?;
    let selected =
        bonsai_cpu_policy::select_artifact(&lock, profile.cpu_profile(), true, Some(&evidence))
            .map_err(|_| PolicyError::Configuration)?;
    serde_json::from_value(selected).map_err(|_| PolicyError::Configuration)
}

pub(crate) fn reviewed_pins_for_representation(
    profile: LaunchProfile,
    use_repacked_ptq1: bool,
) -> Result<ReviewedPins, PolicyError> {
    static PTQ1_REFERENCE_CPU: LazyLock<Result<SelectedModel, PolicyError>> =
        LazyLock::new(|| select_ptq1(LaunchProfile::ReferenceCpu));
    static PTQ1_SWIFT_PQ2: LazyLock<Result<SelectedModel, PolicyError>> =
        LazyLock::new(|| select_ptq1(LaunchProfile::SwiftPq2Assessment));
    let mut pins = reviewed_pins(profile)?;
    if use_repacked_ptq1 {
        let selected = match profile {
            LaunchProfile::ReferenceCpu => PTQ1_REFERENCE_CPU.clone()?,
            LaunchProfile::SwiftPq2Assessment => PTQ1_SWIFT_PQ2.clone()?,
        };
        pins.model.filename = selected.filename;
        pins.model.bytes = selected.bytes;
        pins.model.sha256 = selected.sha256;
        pins.model.chunk_manifest =
            validate_chunk_manifest(selected.bytes, selected.chunk_bytes, selected.chunk_sha256s)?;
    }
    Ok(pins)
}

#[derive(Clone, Deserialize)]
#[serde(rename_all = "PascalCase", deny_unknown_fields)]
struct SelectedModel {
    filename: String,
    bytes: u64,
    sha256: String,
    #[serde(default)]
    chunk_bytes: Option<usize>,
    #[serde(default)]
    chunk_sha256s: Option<Vec<String>>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct SwiftPins {
    schema_version: u8,
    decision: String,
    model: super::Object<ModelPinCommon>,
    runtime: super::Object<RuntimePin>,
    qualification: super::Object<SwiftQualification>,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct SwiftQualification {
    purpose: String,
    model_download: String,
    full_gguf_inspection: String,
    ternary_symbol_census: String,
    cpu_loading: String,
    daily_security: String,
    device_performance: String,
}

fn parse_swift_pins(input: &str) -> Result<ReviewedPins, PolicyError> {
    let pins: SwiftPins = parse_json_object(input)?;
    let q = &pins.qualification;
    if pins.schema_version != 1
        || pins.decision != "docs/swift-memory.md"
        || pins.model.fields()
            != (ModelPinFields {
                repository: "ukisai/Swift-Bonsai-2-GGUF",
                revision: "a3bdac086bbb6b04d87d0108d18943b5f63868f7",
                filename: "Swift-Bonsai-2-PQ2_0.gguf",
                bytes: 7_206_168_928,
                sha256: "5912bb739217cf25b4283a098c7b893d9baff36bdf2e72d3dbf4cb999d5633d5",
                digest_source: "https://huggingface.co/api/models/ukisai/Swift-Bonsai-2-GGUF/revision/a3bdac086bbb6b04d87d0108d18943b5f63868f7?blobs=true",
                license: "Apache-2.0",
                architecture: "qwen35",
            })
        || pins.runtime.fields()
            != (RuntimePinFields {
                repository: "https://github.com/PrismML-Eng/llama.cpp.git",
                commit: "73bfe6cc485554822f80c473a3c0c4f25e80f41b",
                release: "prism-b10787-73bfe6c",
                license: "MIT",
                asset_filename: "llama-prism-b10787-73bfe6c-bin-win-cpu-arm64.zip",
                asset_bytes: 12_412_126,
                asset_sha256: "cc808b33d35c72a3bd5c69a0d609d422426feb9dce23868e77992b6c33b8ddb3",
                asset_url: "https://github.com/PrismML-Eng/llama.cpp/releases/download/prism-b10787-73bfe6c/llama-prism-b10787-73bfe6c-bin-win-cpu-arm64.zip",
                asset_digest_source: "https://api.github.com/repos/PrismML-Eng/llama.cpp/releases/tags/prism-b10787-73bfe6c",
            })
        || q.purpose != "owner-prioritized Swift memory and exact ternary repacking assessment"
        || q.model_download != "verified-size-and-sha256-2026-10-23"
        || q.full_gguf_inspection != "config/swift.metadata.json"
        || q.ternary_symbol_census != "config/swift.pq2-census.json"
        || q.cpu_loading
            != "bounded-16-token-native-decode-completed; PQ2/current-repacked-PTQ1 with and without approved adapter; quality and TPS unqualified"
        || q.daily_security != "not-qualified"
        || q.device_performance != "pending-self-hosted-ci"
    {
        return Err(PolicyError::Configuration);
    }
    Ok(ReviewedPins {
        model: pins.model.0.into_artifact()?,
        runtime: pins.runtime,
    })
}

fn parse_pins(input: &str) -> Result<Pins, PolicyError> {
    let pins: Pins = parse_json_object(input)?;
    let metadata = &pins.model.expected_metadata;
    let f = &pins.fallback;
    let q = &pins.qualification;
    if pins.schema_version != 1
        || pins.decision != "docs/model-selection.md"
        || pins.model.fields()
            != (ModelPinFields {
                repository: "unsloth/Qwen3.8-27B-GGUF",
                revision: "4ca720788d1e01f1bff70c033e0d0028fd02e502",
                filename: "Qwen3.8-27B-UD-Q3_K_XL.gguf",
                bytes: 13_146_393_504,
                sha256: "8c2a45ff85e7674ca185ec8eb6cdeab0e617ed9d8018caed0b64380eb2a67a5e",
                digest_source: "https://huggingface.co/api/models/unsloth/Qwen3.8-27B-GGUF/revision/4ca720788d1e01f1bff70c033e0d0028fd02e502?blobs=true",
                license: "Apache-2.0",
                architecture: "qwen35",
            })
        || [
            metadata.block_count_including_mtp,
            metadata.mtp_blocks,
            metadata.full_attention_interval,
            metadata.kv_heads,
            metadata.key_length,
            metadata.value_length,
            metadata.ssm_state_size,
            metadata.ssm_inner_size,
            metadata.ssm_group_count,
            metadata.ssm_convolution_kernel,
        ] != [65, 1, 4, 4, 256, 256, 128, 6144, 16, 4]
        || pins.runtime.fields()
            != (RuntimePinFields {
                repository: "https://github.com/ggml-org/llama.cpp.git",
                commit: "36e36e63a54fc8f85b95e9c499a022b597e478dd",
                release: "b11840",
                license: "MIT",
                asset_filename: "llama-b11840-bin-win-cpu-arm64.zip",
                asset_bytes: 12_781_980,
                asset_sha256: "9bfdb90329cd190c8286e5e53ed74391f8455c7cf7a6370725817440dd7ff085",
                asset_url: "https://github.com/ggml-org/llama.cpp/releases/download/b11840/llama-b11840-bin-win-cpu-arm64.zip",
                asset_digest_source: "https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/b11840",
            })
        || f.repository != "google/gemma-4-12B-it-qat-q4_0-gguf"
        || f.revision != "29d097773436b69ff9feafd636ab4cf873786537"
        || f.filename != "gemma-4-12b-it-qat-q4_0.gguf"
        || f.bytes != 6_975_879_296
        || f.sha256 != "93567e57a8fe10b23569b9d9ec38cd005deedf71e29477c421a4b83f418a538b"
        || f.status != "literature-nominated-only; not downloaded or selected"
        || q.source_compatibility != "passed"
        || q.model_download != "size-and-sha256-verified-2026-10-23"
        || q.full_gguf_inspection != "verified-config/model.metadata.json"
        || q.cpu_loading != "pending"
        || q.daily_security != "pending"
        || q.device_performance != "pending-self-hosted-ci"
        || bonsai_cpu_policy::validate_chunk_manifest_ref(
            pins.model.bytes,
            pins.model.chunk_bytes,
            pins.model.chunk_sha256s.as_deref(),
        )
        .is_err()
    {
        return Err(PolicyError::Configuration);
    }
    Ok(pins)
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Profile {
    schema_version: u8,
    name: String,
    status: String,
    backend: String,
    context_tokens: u32,
    parallel: u32,
    batch_tokens: u32,
    microbatch_tokens: u32,
    cache_type_k: String,
    cache_type_v: String,
    context_checkpoints: u32,
    prompt_cache_mi_b: u32,
    cache_prompt: bool,
    speculation: String,
    service_ceiling_bytes: u64,
    estimated_weight_bytes: u64,
    estimated_kv_bytes: u64,
    estimated_recurrent_bytes: u64,
    estimated_runtime_headroom_bytes: u64,
    estimate_scope: String,
    launch_enabled: bool,
    remaining_gates: Vec<String>,
}
fn validate_profile(input: &str) -> Result<(), PolicyError> {
    let p: Profile = parse_json_object(input)?;
    if p.schema_version != 1
        || p.name != "reference-cpu"
        || p.status != "development-only-unqualified"
        || p.backend != "cpu"
        || p.context_tokens != 8192
        || p.parallel != 1
        || p.batch_tokens != 256
        || p.microbatch_tokens != 128
        || p.cache_type_k != "f16"
        || p.cache_type_v != "f16"
        || p.context_checkpoints != 0
        || p.prompt_cache_mi_b != 0
        || p.cache_prompt
        || p.speculation != "none"
        || p.service_ceiling_bytes != 19_327_352_832
        || p.estimated_weight_bytes != 13_146_393_504
        || p.estimated_kv_bytes != 536_870_912
        || p.estimated_recurrent_bytes != 156_893_184
        || p.estimated_runtime_headroom_bytes != 5_487_195_232
        || p.launch_enabled
        || p.estimate_scope.trim().is_empty()
        || p.remaining_gates.is_empty()
        || p.remaining_gates.iter().any(|gate| gate.trim().is_empty())
    {
        return Err(PolicyError::Configuration);
    }
    Ok(())
}

/// Validate path spelling and existing ancestors without following redirects.
/// Native launch code additionally verifies the volume is a recognized local disk.
///
/// # Errors
/// Rejects nonabsolute, namespace, ADS, unresolved-variable, Git and reparse paths.
pub fn validate_external_path(path: &Path) -> Result<(), PolicyError> {
    bonsai_artifacts::validate_external_path(path).map_err(|_| PolicyError::Path)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn compiled_configuration_is_reviewed() {
        assert!(reviewed_pins(LaunchProfile::ReferenceCpu).is_ok());
        assert!(reviewed_pins(LaunchProfile::SwiftPq2Assessment).is_ok());
        assert!(validate_profile(PROFILE_JSON).is_ok());
    }

    #[test]
    fn repacked_identity_is_the_exact_current_weight_export_not_the_older_checkpoint()
    -> Result<(), PolicyError> {
        let profile = LaunchProfile::SwiftPq2Assessment;
        let original = reviewed_pins_for_representation(profile, false)?;
        let repacked = reviewed_pins_for_representation(profile, true)?;
        assert_eq!(original.model.filename, "Swift-Bonsai-2-PQ2_0.gguf");
        assert_eq!(
            repacked.model.filename,
            "Swift-Bonsai-2-current-repacked-PTQ1_0.gguf"
        );
        assert_eq!(repacked.model.bytes, 5_946_648_928);
        assert_eq!(
            repacked.model.sha256,
            "e658f0f0736a03a4549535a439f9b07254ebf8ab88ae27cbe6e52d098ca5d1f5"
        );
        assert_eq!(repacked.model.revision, original.model.revision);
        assert_eq!(repacked.runtime.commit, original.runtime.commit);
        assert!(reviewed_pins_for_representation(LaunchProfile::ReferenceCpu, true).is_err());
        Ok(())
    }

    #[test]
    fn profile_drift_and_coercions_fail_closed() {
        for changed in [
            PROFILE_JSON.replace("\"launchEnabled\": false", "\"launchEnabled\": true"),
            PROFILE_JSON.replace("\"contextTokens\": 8192", "\"contextTokens\": \"8192\""),
            PROFILE_JSON.replace(
                "\"schemaVersion\": 1",
                "\"schemaVersion\": 1, \"extraArgs\": []",
            ),
            PROFILE_JSON.replace("\"parallel\": 1", "\"parallel\": 1, \"parallel\": 1"),
            PROFILE_JSON.replace("\"parallel\": 1", "\"parallel\": 1.0"),
        ] {
            assert_eq!(validate_profile(&changed), Err(PolicyError::Configuration));
        }
    }
    #[test]
    fn lock_drift_and_shapes_fail_closed() {
        for changed in [
            LOCK_JSON.replace("12781980", "12781981"),
            LOCK_JSON.replace("\"bytes\": 12781980", "\"bytes\": \"12781980\""),
            LOCK_JSON.replace(
                "\"schemaVersion\": 1",
                "\"schemaVersion\": 1, \"schemaVersion\": 1",
            ),
            LOCK_JSON.replace("\"model\": {", "\"model\": {\"argument\": \"--unsafe\","),
            "[]".to_owned(),
        ] {
            assert_ne!(changed, LOCK_JSON, "lock mutation must change the fixture");
            assert!(parse_pins(&changed).is_err());
        }
    }

    #[test]
    fn swift_lock_drift_and_shapes_fail_closed() {
        for changed in [
            SWIFT_LOCK_JSON.replace("7206168928", "7206168929"),
            SWIFT_LOCK_JSON.replace("12412126", "12412127"),
            SWIFT_LOCK_JSON.replace("PQ2_0.gguf", "PTQ1_0.gguf"),
            SWIFT_LOCK_JSON.replace("ukisai/Swift-Bonsai-2-GGUF", "other/Swift-Bonsai-2-GGUF"),
            SWIFT_LOCK_JSON.replace("not-qualified", "passed"),
            SWIFT_LOCK_JSON.replace(
                "73bfe6cc485554822f80c473a3c0c4f25e80f41b",
                "36e36e63a54fc8f85b95e9c499a022b597e478dd",
            ),
            SWIFT_LOCK_JSON.replace("\"bytes\": 7206168928", "\"bytes\": \"7206168928\""),
            SWIFT_LOCK_JSON.replace("\"bytes\": 7206168928", "\"bytes\": 7206168928.0"),
            SWIFT_LOCK_JSON.replace(
                "\"schemaVersion\": 1",
                "\"schemaVersion\": 1, \"schemaVersion\": 1",
            ),
            SWIFT_LOCK_JSON.replace("\"model\": {", "\"model\": {\"lora\": \"arbitrary\","),
            SWIFT_LOCK_JSON.replace("\"runtime\": {", "\"runtime\": {\"arguments\": [],"),
            SWIFT_LOCK_JSON.replace(
                "\"qualification\": {",
                "\"qualification\": {\"unknown\": true,",
            ),
            LOCK_JSON.to_owned(),
            "[]".to_owned(),
        ] {
            assert_ne!(changed, SWIFT_LOCK_JSON, "mutation must change the fixture");
            assert!(parse_swift_pins(&changed).is_err());
        }
        assert!(parse_pins(SWIFT_LOCK_JSON).is_err());
    }

    #[cfg(windows)]
    #[test]
    fn enable_swift_mmproj_replaces_no_mmproj_with_bounded_image_tokens() -> Result<(), PolicyError>
    {
        let data = std::env::temp_dir().join("Bonsai-launch-policy-mmproj");
        let key = data.join("backend.key");
        let mut plan = LaunchPlan::for_profile_with_options(
            &data,
            8081,
            &key,
            LaunchProfile::SwiftPq2Assessment,
            false,
            false,
        )?;
        assert!(plan.mmproj.is_none());
        assert!(plan.arguments.iter().any(|arg| arg == "--no-mmproj"));
        let mmproj = bonsai_cpu_policy::swift_mmproj_artifact(&data)
            .map_err(|_| PolicyError::Configuration)?;
        let expected_path = mmproj.path.clone();
        plan.enable_swift_mmproj(mmproj.clone())?;
        assert!(!plan.arguments.iter().any(|arg| arg == "--no-mmproj"));
        assert!(
            plan.mmproj
                .as_ref()
                .is_some_and(|item| item.path == expected_path)
        );
        let expected_flag_pos = plan
            .arguments
            .iter()
            .position(|arg| arg == "--mmproj")
            .ok_or(PolicyError::Configuration)?;
        assert_eq!(
            plan.arguments.get(expected_flag_pos + 1),
            Some(expected_path.as_os_str()).map(OsString::from).as_ref()
        );
        assert_eq!(
            plan.arguments.get(expected_flag_pos + 2),
            Some(&OsString::from("--no-mmproj-offload"))
        );
        assert_eq!(
            plan.arguments.get(expected_flag_pos + 3),
            Some(&OsString::from("--image-max-tokens"))
        );
        assert_eq!(
            plan.arguments.get(expected_flag_pos + 4),
            Some(&OsString::from("1024"))
        );
        assert_eq!(
            plan.enable_swift_mmproj(mmproj),
            Err(PolicyError::Configuration)
        );
        Ok(())
    }

    #[test]
    fn model_lock_optional_chunk_manifest_validates_and_fails_closed() -> Result<(), PolicyError> {
        // Default compiled locks omit chunk manifests and fall back to whole-file verification.
        assert!(
            reviewed_pins(LaunchProfile::ReferenceCpu)?
                .model
                .chunk_manifest
                .is_none()
        );
        assert!(
            reviewed_pins(LaunchProfile::SwiftPq2Assessment)?
                .model
                .chunk_manifest
                .is_none()
        );

        // Valid Swift chunk manifest: 1 GiB chunks -> ceil(7_206_168_928 / 1 GiB) = 7.
        let chunk_bytes = 1024 * 1024 * 1024_usize;
        let digests = vec!["a".repeat(64); 7];
        let mut valid_swift: serde_json::Value =
            serde_json::from_str(SWIFT_LOCK_JSON).map_err(|_| PolicyError::Configuration)?;
        valid_swift["model"]["chunkBytes"] = serde_json::json!(chunk_bytes);
        valid_swift["model"]["chunkSha256s"] = serde_json::json!(digests);
        let parsed = parse_swift_pins(&valid_swift.to_string())?;
        assert_eq!(parsed.model.chunk_manifest, Some((chunk_bytes, digests)));

        // Valid Reference chunk manifest: 1 GiB chunks -> ceil(13_146_393_504 / 1 GiB) = 13.
        let ref_digests = vec!["b".repeat(64); 13];
        let mut valid_ref: serde_json::Value =
            serde_json::from_str(LOCK_JSON).map_err(|_| PolicyError::Configuration)?;
        valid_ref["model"]["chunkBytes"] = serde_json::json!(chunk_bytes);
        valid_ref["model"]["chunkSha256s"] = serde_json::json!(ref_digests);
        let parsed_ref = parse_pins(&valid_ref.to_string())?;
        let ref_artifact = parsed_ref.model.0.into_artifact()?;
        assert_eq!(
            ref_artifact.chunk_manifest,
            Some((chunk_bytes, ref_digests))
        );

        // Reject one-sided chunkBytes, one-sided chunkSha256s, sub-threshold chunkBytes,
        // mismatched digest count, and malformed hex in both Swift and Reference pins.
        let min_chunk = bonsai_cpu_policy::MIN_MAPPED_HASH_BYTES;
        #[cfg(windows)]
        assert_eq!(min_chunk, bonsai_windows_credentials::MIN_MAPPED_HASH_BYTES);

        for bad_model in [
            serde_json::json!({"chunkBytes": chunk_bytes}),
            serde_json::json!({"chunkSha256s": vec!["a".repeat(64); 7]}),
            serde_json::json!({"chunkBytes": min_chunk - 1, "chunkSha256s": vec!["a".repeat(64); 7]}),
            serde_json::json!({"chunkBytes": 0, "chunkSha256s": vec!["a".repeat(64); 7]}),
            serde_json::json!({"chunkBytes": chunk_bytes, "chunkSha256s": vec!["a".repeat(64); 6]}),
            serde_json::json!({"chunkBytes": chunk_bytes, "chunkSha256s": vec!["a".repeat(64); 8]}),
            serde_json::json!({"chunkBytes": chunk_bytes, "chunkSha256s": vec!["A".repeat(64); 7]}),
            serde_json::json!({"chunkBytes": chunk_bytes, "chunkSha256s": vec!["bad"; 7]}),
        ] {
            let mut bad_swift: serde_json::Value =
                serde_json::from_str(SWIFT_LOCK_JSON).map_err(|_| PolicyError::Configuration)?;
            if let Some(obj) = bad_model.as_object() {
                for (k, v) in obj {
                    bad_swift["model"][k] = v.clone();
                }
            }
            assert_eq!(
                parse_swift_pins(&bad_swift.to_string()).err(),
                Some(PolicyError::Configuration)
            );

            let mut bad_ref: serde_json::Value =
                serde_json::from_str(LOCK_JSON).map_err(|_| PolicyError::Configuration)?;
            if let Some(obj) = bad_model.as_object() {
                for (k, v) in obj {
                    bad_ref["model"][k] = v.clone();
                }
            }
            assert_eq!(
                parse_pins(&bad_ref.to_string()).err(),
                Some(PolicyError::Configuration)
            );
        }
        Ok(())
    }
}
