//! Explicit Windows development launch of a compiled, pinned CPU backend.
//!
//! This owns lifetime, credentials, artifact handles and resource monitoring.
//! The native sandbox is required unless the direct development caller explicitly
//! selects the unisolated backend. Same-user interference remains unqualified.

mod identity;
mod startup;

#[cfg(test)]
use identity::validate_native_identity;
use identity::{NativeIdentityError, WorkerReadiness, validate_native_identity_detailed};
use startup::StartupTimer;

pub use startup::{StartupPhase, StartupProgress};

use crate::{
    artifacts::VerifiedArtifacts,
    launch_policy::{LaunchPlan, LaunchProfile},
    runtime::{BackendResources, RunExit, RunReport, run},
    scheduler::MAX_WAITING,
    service::{LocalChatService, bound_loopback_socket},
    supervisor::BackendConnector,
};
use bonsai_windows_credentials::{PrivateCredentials, validate_external_path};
use bonsai_windows_process::{
    MAX_WORKER_PRIVATE_STORAGE_BYTES, MAX_WORKER_PRIVATE_STORAGE_ENTRIES, OwnedProcess,
    ProcessMemory, ProcessSpec, SandboxPolicy, available_memory_bytes, listener_process_id,
    native_sandbox_available, native_sandbox_offline_available, validate_sandbox_environment,
};
use std::{
    cell::Cell,
    ffi::OsString,
    future::poll_fn,
    io,
    net::{Ipv4Addr, SocketAddrV4},
    path::PathBuf,
    pin::pin,
    task::Poll,
    time::Duration,
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpStream,
    time::{interval, sleep, timeout},
};

/// Explicit containment selection is required even though the backend is owned.
pub const LAUNCH_USAGE: &str = concat!(
    "Windows CPU development launcher (daily isolation unqualified).\n",
    "bonsai-broker --launch-development\n",
    "  <--require-native-sandbox|--allow-unisolated-backend> [--require-broker-pipe]\n",
    "  --data-directory <existing-external-directory>\n",
    "  --evidence-parent <existing-external-directory>\n",
    "  --frontend-port <0-or-1024..65535> [--backend-port <1024..65535>]\n",
    "  [--profile <reference-cpu|swift-pq2-assessment>]\n",
    "  [--native-engine|--publisher-engine] (Swift defaults to the pinned Rust runtime; publisher is explicit rollback)\n",
    "  [--native-row-workers 1|4|8|12] (Rust defaults to 8 on Snapdragon X1/X2 or 4 otherwise; publisher supports only 1)\n",
    "  [--prompt-reuse disabled|latest] (experimental exact latest-state reuse; Rust only)\n",
    "  [--kv-cache f16|rsq4|rsq4k3v] (reviewed native runtime only; rsq4 default when supported, f16 explicit recovery)\n",
    "  [--cpu-prefill-tokens 8|16|32] (reviewed native runtime only; enables layer-major CPU FFN tile prefill)\n",
    "  [--placement auto|cpu] (native Swift only; CPU requires a reviewed compatible release)\n",
    "  [--reset-on-cancel] (reviewed native runtime only; reset live state in place on client cancel)\n",
    "  [--no-mmproj] (disable multimodal vision projector; Swift only)\n",
    "  [--use-abliterate-lora] (pinned adapter; request scale 0..2, default 1.3; Swift CPU only)\n",
    "  [--use-repacked-ptq1|--use-original-pq2] (mutually exclusive weight selections; Swift CPU only)\n",
    "  [--max-waiting <0..8>] [--check-only]\n",
    "Default: reference-cpu. Only compiled model/runtime pins and fixed CPU policies are accepted.\n",
    "Swift defaults to Rust with four workers and original PQ2 weights; PTQ1 requires explicit selection. --require-broker-pipe omits --backend-port and requires a reviewed offline native release. Full daily-use qualification remains pending.\n",
    "Creates fresh protected credentials; prints paths, never key values.\n",
);

const PHYSICAL_RESERVE: u64 = 4 * 1024 * 1024 * 1024;
const STARTUP_TIMEOUT: Duration = Duration::from_mins(10);
const STOP_TIMEOUT: Duration = Duration::from_secs(10);
const HEALTH_LIMIT: usize = 20 * 1024;

/// Value-free launcher failure, suitable for technical status without secrets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LaunchError {
    /// Missing acknowledgement, repeated or unknown option, or invalid port.
    Arguments,
    /// Storage is not an existing external local directory without redirection.
    Path,
    /// The fixed launch policy or inherited environment is invalid.
    Policy,
    /// Locked model/runtime verification failed.
    Artifacts,
    /// The selected native Rust runtime is missing, changed or unsafe to open.
    NativeRuntime,
    /// The pinned abliteration adapter is missing, altered or unsafe to open.
    Adapter,
    /// Protected credential provisioning failed.
    Credentials,
    /// Physical memory admission or a running resource guard failed.
    Memory,
    /// The requested backend port is occupied by another process.
    Port,
    /// Atomic Job-owned process creation failed.
    Process,
    /// The required native containment and loopback capabilities are unavailable.
    SandboxUnavailable,
    /// The parent environment carries credentials unsuitable for a model worker.
    SandboxEnvironment,
    /// The required sandbox policy could not be applied; no unisolated retry occurs.
    Sandbox,
    /// The owned backend exited, failed readiness, or lost its listener identity.
    Backend,
    /// Console cancellation occurred before readiness.
    Cancelled,
    /// Frontend policy validation or listener binding failed.
    Service,
    /// Termination did not confirm an empty owned Job within the cleanup limit.
    Stop,
}

enum WeightRepresentation {
    Source,
    CurrentRepackedPtq1,
}

enum BackendContainment {
    NativeSandbox,
    ExplicitUnisolated,
}

/// Parsed explicit development launch; paths and credentials have no Debug output.
#[expect(
    clippy::struct_excessive_bools,
    reason = "Explicit check, runtime, adapter and reuse policies are independent."
)]
pub struct LaunchConfig {
    data: PathBuf,
    evidence_parent: PathBuf,
    frontend_port: u16,
    backend_port: u16,
    max_waiting: usize,
    check_only: bool,
    containment: BackendContainment,
    require_broker_pipe: bool,
    profile: LaunchProfile,
    native_engine: bool,
    native_row_workers: u8,
    prompt_reuse: crate::inference::PromptReuseMode,
    prompt_reuse_checkpoints: Option<u8>,
    prompt_reuse_checkpoint_bytes: Option<usize>,
    kv_cache: Option<crate::inference::KvCache>,
    cpu_prefill_tokens: Option<u8>,
    cpu_prefill_gdn: Option<crate::inference::CpuPrefillGdn>,
    spec_type: Option<crate::inference::SpeculationType>,
    draft_max: Option<u8>,
    cpu_family: Option<crate::inference::CpuFamily>,
    placement: crate::inference::Placement,
    reset_on_cancel: bool,
    no_mmproj: bool,
    use_abliterate_lora: bool,
    weight_representation: WeightRepresentation,
}

/// Offline verification result; checking never creates credentials or a process.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaunchCheck {
    /// Whether the current physical-memory observation meets launch admission.
    pub memory_admitted: bool,
}

struct VerifiedLaunchGuards {
    native_executable: Option<(PathBuf, bonsai_artifacts::VerifiedFile)>,
    artifacts: VerifiedArtifacts,
    adapter: Option<bonsai_artifacts::VerifiedFile>,
    mmproj: Option<(
        bonsai_cpu_policy::SwiftMmprojArtifact,
        bonsai_artifacts::VerifiedFile,
    )>,
}

impl LaunchConfig {
    /// Parse the launcher arguments, including an explicit containment selection.
    ///
    /// # Errors
    /// Rejects unknown/repeated options, nondecimal ports and missing opt-ins.
    pub fn parse(args: impl IntoIterator<Item = OsString>) -> Result<Self, LaunchError> {
        Self::parse_for_family(args, None)
    }

    /// Parse the launcher arguments using the observed Windows ARM64 host family
    /// when `--native-row-workers` is omitted.
    ///
    /// # Errors
    /// Rejects unknown/repeated options, nondecimal ports and missing opt-ins.
    pub fn parse_host(args: impl IntoIterator<Item = OsString>) -> Result<Self, LaunchError> {
        let family = std::thread::available_parallelism()
            .ok()
            .is_some_and(|cores| cores.get() >= 8)
            .then(|| bonsai_windows_process::native_build_host().ok())
            .flatten()
            .and_then(|host| {
                bonsai_cpu_policy::hardware::SnapdragonFamily::from_processor_name(
                    &host.processor_name,
                )
            });
        Self::parse_for_family(args, family)
    }

    /// Parse the launcher arguments using an explicit Snapdragon host family for
    /// default `--native-row-workers` selection.
    ///
    /// # Errors
    /// Rejects unknown/repeated options, nondecimal ports and missing opt-ins.
    pub fn parse_for_family(
        args: impl IntoIterator<Item = OsString>,
        family: Option<bonsai_cpu_policy::hardware::SnapdragonFamily>,
    ) -> Result<Self, LaunchError> {
        let mut args = args.into_iter();
        let (mut launch, mut unisolated, mut require_native_sandbox, mut require_broker_pipe) =
            (false, false, false, false);
        let mut check_only = false;
        let mut native_engine = None;
        let (mut native_row_workers, mut prompt_reuse) = (None, None);
        let (mut prompt_reuse_checkpoints, mut prompt_reuse_checkpoint_bytes) = (None, None);
        let (mut kv_cache, mut cpu_prefill_tokens, mut cpu_prefill_gdn, mut placement) =
            (None, None, None, None);
        let (mut spec_type, mut draft_max, mut cpu_family) = (None, None, None);
        let mut reset_on_cancel = false;
        let mut no_mmproj = false;
        let mut use_abliterate_lora = false;
        let mut use_repacked_ptq1 = None;
        let mut profile = None;
        let (mut data, mut evidence_parent) = (None, None);
        let (mut frontend_port, mut backend_port, mut max_waiting) = (None, None, None);
        while let Some(argument) = args.next() {
            match argument.to_str() {
                Some("--launch-development") if !launch => launch = true,
                Some("--allow-unisolated-backend") if !unisolated => unisolated = true,
                Some("--require-native-sandbox") if !require_native_sandbox => {
                    require_native_sandbox = true;
                }
                Some("--require-broker-pipe") if !require_broker_pipe => {
                    require_broker_pipe = true;
                }
                Some("--check-only") if !check_only => check_only = true,
                Some("--native-engine" | "--publisher-engine") if native_engine.is_none() => {
                    native_engine = Some(argument == "--native-engine");
                }
                Some("--native-row-workers") if native_row_workers.is_none() => {
                    native_row_workers = Some(row_workers(args.next().as_deref())?);
                }
                Some("--prompt-reuse") if prompt_reuse.is_none() => {
                    prompt_reuse = Some(crate::inference::PromptReuseMode::parse(
                        args.next().as_deref(),
                    )?);
                }
                Some("--prompt-reuse-checkpoints") if prompt_reuse_checkpoints.is_none() => {
                    prompt_reuse_checkpoints = Some(decimal_range_u8(args.next(), 1..=8)?);
                }
                Some("--prompt-reuse-checkpoint-bytes")
                    if prompt_reuse_checkpoint_bytes.is_none() =>
                {
                    prompt_reuse_checkpoint_bytes = Some(positive_decimal_usize(args.next())?);
                }
                Some("--kv-cache") if kv_cache.is_none() => {
                    kv_cache = Some(crate::inference::KvCache::parse(args.next().as_deref())?);
                }
                Some("--cpu-prefill-tokens") if cpu_prefill_tokens.is_none() => {
                    cpu_prefill_tokens = Some(crate::inference::parse_cpu_prefill_tokens(
                        args.next().as_deref(),
                    )?);
                }
                Some("--cpu-prefill-gdn") if cpu_prefill_gdn.is_none() => {
                    cpu_prefill_gdn = Some(crate::inference::CpuPrefillGdn::parse(
                        args.next().as_deref(),
                    )?);
                }
                Some("--spec-type") if spec_type.is_none() => {
                    spec_type = Some(crate::inference::SpeculationType::parse(
                        args.next().as_deref(),
                    )?);
                }
                Some("--draft-max") if draft_max.is_none() => {
                    draft_max = Some(decimal_range_u8(args.next(), 1..=8)?);
                }
                Some("--cpu-family") if cpu_family.is_none() => {
                    cpu_family = Some(crate::inference::CpuFamily::parse(args.next().as_deref())?);
                }
                Some("--placement") if placement.is_none() => {
                    placement = Some(crate::inference::Placement::parse(args.next().as_deref())?);
                }
                Some("--reset-on-cancel") if !reset_on_cancel => {
                    reset_on_cancel = true;
                }
                Some("--no-mmproj") if !no_mmproj => {
                    no_mmproj = true;
                }
                Some("--use-abliterate-lora") if !use_abliterate_lora => {
                    use_abliterate_lora = true;
                }
                Some("--use-repacked-ptq1") if use_repacked_ptq1.is_none() => {
                    use_repacked_ptq1 = Some(true);
                }
                Some("--use-original-pq2") if use_repacked_ptq1.is_none() => {
                    use_repacked_ptq1 = Some(false);
                }
                Some("--profile") if profile.is_none() => {
                    profile = Some(launch_profile(args.next().as_deref())?);
                }
                Some("--data-directory") if data.is_none() => {
                    data = Some(PathBuf::from(args.next().ok_or(LaunchError::Arguments)?));
                }
                Some("--evidence-parent") if evidence_parent.is_none() => {
                    evidence_parent =
                        Some(PathBuf::from(args.next().ok_or(LaunchError::Arguments)?));
                }
                Some("--frontend-port") if frontend_port.is_none() => {
                    frontend_port = Some(number(args.next())?);
                }
                Some("--backend-port") if backend_port.is_none() => {
                    backend_port = Some(number(args.next())?);
                }
                Some("--max-waiting") if max_waiting.is_none() => {
                    max_waiting = Some(usize::from(number(args.next())?));
                }
                _ => return Err(LaunchError::Arguments),
            }
        }
        let frontend_port = frontend_port.ok_or(LaunchError::Arguments)?;
        let backend_port = match backend_port {
            Some(port) => port,
            None if require_broker_pipe => {
                if frontend_port == 8081 {
                    8082
                } else {
                    8081
                }
            }
            None => return Err(LaunchError::Arguments),
        };
        let max_waiting = max_waiting.unwrap_or(2);
        let profile = profile.unwrap_or_default();
        let native_engine = native_engine.unwrap_or(profile == LaunchProfile::SwiftPq2Assessment);
        let native_row_workers = native_row_workers.unwrap_or(if native_engine {
            if let Some(explicit_family) = cpu_family {
                explicit_family.default_row_workers()
            } else {
                bonsai_cpu_policy::hardware::SnapdragonFamily::default_native_row_workers(family)
            }
        } else {
            1
        });
        let prompt_reuse = prompt_reuse.unwrap_or_default();
        let kv_cache = crate::inference::pinned_kv_cache(native_engine, kv_cache)?;
        if (cpu_prefill_tokens.is_some()
            || cpu_prefill_gdn.is_some()
            || spec_type.is_some()
            || draft_max.is_some()
            || cpu_family.is_some()
            || placement.is_some())
            && profile != LaunchProfile::SwiftPq2Assessment
        {
            return Err(LaunchError::Arguments);
        }
        let cpu_prefill_tokens =
            crate::inference::pinned_cpu_prefill_tokens(native_engine, cpu_prefill_tokens)?;
        let placement = crate::inference::pinned_placement(native_engine, placement)?;
        let configuration = Self {
            data: data.ok_or(LaunchError::Arguments)?,
            evidence_parent: evidence_parent.ok_or(LaunchError::Arguments)?,
            frontend_port,
            backend_port,
            max_waiting,
            check_only,
            containment: containment(launch, unisolated, require_native_sandbox)?,
            require_broker_pipe,
            profile,
            native_engine,
            native_row_workers,
            prompt_reuse,
            prompt_reuse_checkpoints,
            prompt_reuse_checkpoint_bytes,
            kv_cache,
            cpu_prefill_tokens,
            cpu_prefill_gdn,
            spec_type,
            draft_max,
            cpu_family,
            placement,
            reset_on_cancel,
            no_mmproj,
            use_abliterate_lora,
            weight_representation: if use_repacked_ptq1.unwrap_or(false) {
                WeightRepresentation::CurrentRepackedPtq1
            } else {
                WeightRepresentation::Source
            },
        };
        configuration.validate_selection(use_repacked_ptq1.is_some())?;
        if configuration.reset_on_cancel && !crate::inference::reset_on_cancel_supported()? {
            return Err(LaunchError::NativeRuntime);
        }
        if configuration.require_broker_pipe && !crate::inference::broker_pipe_supported()? {
            return Err(LaunchError::NativeRuntime);
        }
        if (configuration.prompt_reuse == crate::inference::PromptReuseMode::PromptEndAndLatest
            || configuration.prompt_reuse_checkpoints.is_some()
            || configuration.prompt_reuse_checkpoint_bytes.is_some())
            && !crate::inference::branch_prompt_reuse_supported()?
        {
            return Err(LaunchError::NativeRuntime);
        }
        if configuration.cpu_prefill_gdn.is_some()
            && !crate::inference::cpu_prefill_gdn_supported()?
        {
            return Err(LaunchError::NativeRuntime);
        }
        if configuration.spec_type.is_some() && !crate::inference::cpu_speculation_supported()? {
            return Err(LaunchError::NativeRuntime);
        }
        if configuration.cpu_family.is_some() && !crate::inference::broker_pipe_supported()? {
            return Err(LaunchError::NativeRuntime);
        }
        Ok(configuration)
    }

    const fn effective_speculation_tokens(&self) -> Option<u8> {
        match self.spec_type {
            Some(crate::inference::SpeculationType::Suffix) => Some(match self.draft_max {
                Some(tokens) => tokens,
                None => 4,
            }),
            Some(crate::inference::SpeculationType::None) | None => None,
        }
    }

    fn validate_selection(&self, explicit_weight: bool) -> Result<(), LaunchError> {
        let swift = self.profile == LaunchProfile::SwiftPq2Assessment;
        if (self.frontend_port != 0 && self.frontend_port < 1024)
            || self.backend_port < 1024
            || self.backend_port == self.frontend_port
            || self.max_waiting > MAX_WAITING
            || (self.native_engine && !swift)
            || (self.require_broker_pipe
                && (!self.requires_native_sandbox() || !self.native_engine))
            || (self.native_row_workers != 1 && !self.native_engine)
            || (self.prompt_reuse.enabled() && !self.native_engine)
            || ((self.prompt_reuse_checkpoints.is_some()
                || self.prompt_reuse_checkpoint_bytes.is_some())
                && (!self.native_engine || !self.prompt_reuse.enabled()))
            || ((self.cpu_prefill_tokens.is_some()
                || self.cpu_prefill_gdn.is_some()
                || self.spec_type.is_some()
                || self.draft_max.is_some()
                || self.cpu_family.is_some())
                && !self.native_engine)
            || (self.cpu_prefill_gdn.is_some() && self.cpu_prefill_tokens.is_none())
            || (self.cpu_prefill_gdn == Some(crate::inference::CpuPrefillGdn::WyF32)
                && self.kv_cache != Some(crate::inference::KvCache::F16))
            || (self.draft_max.is_some()
                && self.spec_type != Some(crate::inference::SpeculationType::Suffix))
            || (self.spec_type == Some(crate::inference::SpeculationType::Suffix)
                && self.cpu_prefill_tokens.is_none())
            || (self.cpu_prefill_gdn == Some(crate::inference::CpuPrefillGdn::WyF32)
                && self.spec_type == Some(crate::inference::SpeculationType::Suffix))
            || (self.reset_on_cancel && !self.native_engine)
            || (self.no_mmproj && !swift)
            || (self.use_abliterate_lora && !swift)
            || (explicit_weight && !swift)
        {
            return Err(LaunchError::Arguments);
        }
        Ok(())
    }

    /// Whether the command requests offline checks without credential creation or launch.
    #[must_use]
    pub const fn is_check_only(&self) -> bool {
        self.check_only
    }

    /// Whether launch requires native containment rather than the explicit development opt-out.
    #[must_use]
    pub const fn requires_native_sandbox(&self) -> bool {
        matches!(self.containment, BackendContainment::NativeSandbox)
    }

    /// Compiled profile selected by the exact CLI name, defaulting to the reference.
    #[must_use]
    pub const fn profile(&self) -> LaunchProfile {
        self.profile
    }

    /// Whether the selected Swift child uses the pinned native Rust runtime.
    #[must_use]
    pub const fn uses_native_engine(&self) -> bool {
        self.native_engine
    }

    /// Whether the selected configuration uses offline broker-pipe containment.
    #[must_use]
    pub fn uses_broker_pipe_transport(&self) -> bool {
        self.uses_broker_pipe().unwrap_or(false)
    }

    fn uses_broker_pipe(&self) -> Result<bool, LaunchError> {
        if !self.requires_native_sandbox() || !self.native_engine {
            return if self.require_broker_pipe {
                Err(LaunchError::Arguments)
            } else {
                Ok(false)
            };
        }
        let supported = crate::inference::broker_pipe_supported()?;
        if self.require_broker_pipe && !supported {
            return Err(LaunchError::NativeRuntime);
        }
        Ok(supported)
    }

    /// Whether the explicitly selected model is the verified current-weight PTQ1 export.
    #[must_use]
    pub const fn uses_repacked_ptq1(&self) -> bool {
        matches!(
            self.weight_representation,
            WeightRepresentation::CurrentRepackedPtq1
        )
    }

    /// Whether the selected runtime has the reviewed memory-context protocol.
    /// Publisher Swift supports an explicit context; native releases require
    /// exact source and release-byte review before enabling the paired flags.
    ///
    /// # Errors
    /// Returns a failure if the compiled native release identity cannot be verified.
    pub fn memory_context_supported(&self) -> Result<bool, LaunchError> {
        Ok(self.profile == LaunchProfile::SwiftPq2Assessment
            && (!self.native_engine || crate::inference::memory_context_supported()?))
    }

    fn runtime_supports_vision(&self) -> Result<bool, LaunchError> {
        Ok(self.memory_context_supported()?
            && !self.no_mmproj
            && self.effective_speculation_tokens().is_none()
            && (!self.native_engine || crate::inference::vision_supported()?))
    }

    fn swift_context_profile(
        &self,
    ) -> Result<bonsai_cpu_policy::context::SwiftContextProfile, LaunchError> {
        use bonsai_cpu_policy::context::SwiftContextProfile;
        if self.native_engine {
            match self.kv_cache {
                Some(crate::inference::KvCache::Rsq4)
                    if crate::inference::full_context_supported()? =>
                {
                    Ok(SwiftContextProfile::NativeRsq4Full)
                }
                Some(crate::inference::KvCache::Rsq4) => Ok(SwiftContextProfile::NativeRsq4),
                Some(crate::inference::KvCache::Rsq4K3v) => Ok(SwiftContextProfile::NativeRsq4K3v),
                Some(crate::inference::KvCache::F16) | None => Ok(SwiftContextProfile::NativeF16),
            }
        } else {
            Ok(SwiftContextProfile::PublisherF16)
        }
    }

    fn calibrated_context_plan(
        &self,
        available_bytes: u64,
        request_vision: bool,
    ) -> Result<bonsai_cpu_policy::context::SwiftCalibratedPlan, LaunchError> {
        use bonsai_cpu_policy::context::{
            SwiftCalibratedOptions, SwiftCalibratedReuse, SwiftContextPlan, SwiftWeightPacking,
        };
        let weight_packing = if self.uses_repacked_ptq1() {
            SwiftWeightPacking::Ptq1_0
        } else {
            SwiftWeightPacking::Pq2_0
        };
        let checkpoints = self.prompt_reuse_checkpoints.unwrap_or(1);
        let checkpoint_bytes = self.prompt_reuse_checkpoint_bytes;
        let prompt_reuse = match self.prompt_reuse {
            crate::inference::PromptReuseMode::Disabled => SwiftCalibratedReuse::Disabled,
            crate::inference::PromptReuseMode::Latest => SwiftCalibratedReuse::Latest {
                checkpoints,
                checkpoint_bytes,
            },
            crate::inference::PromptReuseMode::PromptEndAndLatest => {
                SwiftCalibratedReuse::PromptEndAndLatest {
                    checkpoints,
                    checkpoint_bytes,
                }
            }
        };
        SwiftContextPlan::for_calibrated_execution(
            available_bytes,
            self.swift_context_profile()?,
            SwiftCalibratedOptions {
                weight_packing,
                request_vision,
                prefill_tokens: self.cpu_prefill_tokens.map(usize::from),
                chunkwise_gdn_wy_f32: self
                    .cpu_prefill_gdn
                    .is_some_and(crate::inference::CpuPrefillGdn::is_wy_f32),
                speculation_tokens: self.effective_speculation_tokens(),
                prompt_reuse,
            },
        )
        .map_err(|_| LaunchError::Memory)
    }

    // Call only after the exact released runtime's context and cache-accounting
    // contract is admitted. Cache selection remains immutable through retry.
    fn swift_context_plan(
        &self,
        available_bytes: u64,
    ) -> Result<bonsai_cpu_policy::context::SwiftContextPlan, LaunchError> {
        if self.native_engine
            && crate::inference::calibrated_cow_admission_supported().unwrap_or(false)
        {
            return Ok(self
                .calibrated_context_plan(available_bytes, false)?
                .context_plan);
        }
        let (plan, _) = bonsai_cpu_policy::context::SwiftContextPlan::for_profile_with_prefill(
            available_bytes,
            self.swift_context_profile()?,
            self.cpu_prefill_tokens.map(usize::from),
        )
        .map_err(|_| LaunchError::Memory)?;
        Ok(plan)
    }

    fn initial_memory_admission(&self, plan: &LaunchPlan) -> Result<(), LaunchError> {
        if self.memory_context_supported()? {
            self.swift_context_plan(available_memory_bytes().map_err(|_| LaunchError::Memory)?)?;
            Ok(())
        } else {
            admit(plan.required_available_bytes)
        }
    }

    /// Verify the fixed policy and complete artifacts without creating state.
    ///
    /// # Errors
    /// Returns path, policy or artifact failures; low available memory is reported
    /// in the result rather than preventing this read-only verification.
    pub fn check(&self) -> Result<LaunchCheck, LaunchError> {
        self.check_sandbox_support()?;
        self.validate_paths()?;
        let plan = LaunchPlan::for_profile_with_options(
            &self.data,
            self.backend_port,
            &self.evidence_parent.join("not-created.key"),
            self.profile,
            self.use_abliterate_lora,
            self.uses_repacked_ptq1(),
        )
        .map_err(|_| LaunchError::Policy)?;
        let _guards = self.verify_launch_artifacts(&plan)?;
        let available = available_memory_bytes().map_err(|_| LaunchError::Memory)?;
        let memory_admitted = if self.memory_context_supported()? {
            self.swift_context_plan(available).is_ok()
        } else {
            available >= plan.required_available_bytes
        };
        Ok(LaunchCheck { memory_admitted })
    }

    fn check_sandbox_support(&self) -> Result<(), LaunchError> {
        if self.uses_broker_pipe()? {
            native_sandbox_offline_available()
                .map_err(|error| Self::sandbox_support_error(error.kind()))?;
        } else if self.requires_native_sandbox() {
            native_sandbox_available()
                .map_err(|error| Self::sandbox_support_error(error.kind()))?;
        }
        Ok(())
    }

    fn check_launch_environment(&self) -> Result<(), LaunchError> {
        if self.requires_native_sandbox() && !self.uses_broker_pipe()? {
            // Real launch admits the complete V1 loopback policy without a throwaway PSEC.
            validate_sandbox_environment()
                .map_err(|error| Self::sandbox_support_error(error.kind()))?;
        }
        Ok(())
    }

    const fn sandbox_support_error(kind: io::ErrorKind) -> LaunchError {
        if matches!(kind, io::ErrorKind::InvalidInput) {
            LaunchError::SandboxEnvironment
        } else {
            LaunchError::SandboxUnavailable
        }
    }

    fn classify_sandbox_start_error(error: &io::Error) -> LaunchError {
        if bonsai_windows_process::process_start_cleanup_unconfirmed(error) {
            return LaunchError::Sandbox;
        }
        match error.kind() {
            io::ErrorKind::Unsupported => LaunchError::SandboxUnavailable,
            io::ErrorKind::InvalidInput
                if error
                    .to_string()
                    .contains("inherited credential environment variables") =>
            {
                LaunchError::SandboxEnvironment
            }
            _ => LaunchError::Sandbox,
        }
    }

    fn adapter_guard(
        plan: &LaunchPlan,
    ) -> Result<Option<bonsai_artifacts::VerifiedFile>, LaunchError> {
        plan.adapter
            .as_ref()
            .map(|adapter| {
                bonsai_artifacts::VerifiedFile::verify(
                    &adapter.path,
                    adapter.bytes,
                    &adapter.sha256,
                )
                .map_err(|_| LaunchError::Adapter)
            })
            .transpose()
    }

    fn mmproj_guard(
        &self,
    ) -> Result<
        Option<(
            bonsai_cpu_policy::SwiftMmprojArtifact,
            bonsai_artifacts::VerifiedFile,
        )>,
        LaunchError,
    > {
        if !self.runtime_supports_vision()? {
            return Ok(None);
        }
        let artifact = bonsai_cpu_policy::swift_mmproj_artifact(&self.data)
            .map_err(|_| LaunchError::Policy)?;
        bonsai_artifacts::validate_external_path(&artifact.path).map_err(|_| LaunchError::Path)?;
        match std::fs::symlink_metadata(&artifact.path) {
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(_) => return Err(LaunchError::Artifacts),
            Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_file() => {
                return Err(LaunchError::Artifacts);
            }
            Ok(_) => {}
        }
        let guard = bonsai_artifacts::VerifiedFile::verify(
            &artifact.path,
            artifact.bytes,
            &artifact.sha256,
        )
        .map_err(|_| LaunchError::Artifacts)?;
        Ok(Some((artifact, guard)))
    }

    fn verify_artifacts(&self, plan: &LaunchPlan) -> Result<VerifiedArtifacts, LaunchError> {
        if self.native_engine {
            let artifacts = VerifiedArtifacts::verify_model_for_profile(
                &self.data,
                self.profile,
                self.uses_repacked_ptq1(),
            )
            .map_err(|_| LaunchError::Artifacts)?;
            if artifacts.model() != plan.model_path {
                return Err(LaunchError::Artifacts);
            }
            return Ok(artifacts);
        }
        let artifacts = VerifiedArtifacts::verify_for_profile_with_representation(
            &self.data,
            self.profile,
            self.uses_repacked_ptq1(),
        )
        .map_err(|_| LaunchError::Artifacts)?;
        if artifacts.executable() != plan.executable
            || artifacts.model() != plan.model_path
            || artifacts.runtime_directory() != plan.runtime_directory
            || artifacts.archive_path() != plan.archive_path
        {
            return Err(LaunchError::Artifacts);
        }
        Ok(artifacts)
    }

    fn verify_launch_artifacts(
        &self,
        plan: &LaunchPlan,
    ) -> Result<VerifiedLaunchGuards, LaunchError> {
        let verify_vision = self.runtime_supports_vision()?;
        std::thread::scope(|scope| {
            let native_handle = self
                .native_engine
                .then(|| scope.spawn(|| self.native_executable_guard()));
            let adapter_handle = plan
                .adapter
                .is_some()
                .then(|| scope.spawn(|| Self::adapter_guard(plan)));
            let mmproj_handle = verify_vision.then(|| scope.spawn(|| self.mmproj_guard()));

            let artifacts_result = self.verify_artifacts(plan);

            let native_executable = match native_handle {
                Some(handle) => handle.join().map_err(|_| LaunchError::NativeRuntime)??,
                None => None,
            };
            let artifacts = artifacts_result?;
            let adapter = match adapter_handle {
                Some(handle) => handle.join().map_err(|_| LaunchError::Adapter)??,
                None => None,
            };
            let mmproj = match mmproj_handle {
                Some(handle) => handle.join().map_err(|_| LaunchError::Artifacts)??,
                None => None,
            };

            Ok(VerifiedLaunchGuards {
                native_executable,
                artifacts,
                adapter,
                mmproj,
            })
        })
    }

    fn validate_paths(&self) -> Result<(), LaunchError> {
        for path in [&self.data, &self.evidence_parent] {
            validate_external_path(path).map_err(|_| LaunchError::Path)?;
            if !path.is_dir() {
                return Err(LaunchError::Path);
            }
        }
        Ok(())
    }

    fn native_executable_guard(
        &self,
    ) -> Result<Option<(PathBuf, bonsai_artifacts::VerifiedFile)>, LaunchError> {
        if self.native_engine {
            crate::inference::verify(
                &self.data,
                self.native_row_workers,
                self.prompt_reuse.enabled(),
            )
            .map(Some)
            .map_err(|_| LaunchError::NativeRuntime)
        } else {
            Ok(None)
        }
    }

    fn native_arguments(
        &self,
        plan: &LaunchPlan,
        key_path: Option<&std::path::Path>,
    ) -> Vec<OsString> {
        let mut arguments = vec![
            OsString::from("--native-engine-child"),
            OsString::from("--model"),
            plan.model_path.as_os_str().to_owned(),
        ];
        if let Some(key_path) = key_path {
            arguments.extend([
                OsString::from("--port"),
                OsString::from(self.backend_port.to_string()),
                OsString::from("--key-file"),
                key_path.as_os_str().to_owned(),
            ]);
        } else {
            let verified_sections =
                1_usize + usize::from(plan.adapter.is_some()) + usize::from(plan.mmproj.is_some());
            arguments.extend([
                OsString::from("--broker-pipe"),
                OsString::from("--verified-sections"),
                OsString::from(verified_sections.to_string()),
            ]);
        }
        arguments.extend([OsString::from("--seed"), OsString::from("42")]);
        if crate::inference::memory_context_supported().unwrap_or(false) {
            arguments.extend([
                OsString::from("--context-tokens"),
                OsString::from(plan.context_tokens.to_string()),
                OsString::from("--memory-limit-bytes"),
                OsString::from(plan.native_child_memory_limit_bytes().to_string()),
            ]);
            if crate::inference::calibrated_cow_admission_supported().unwrap_or(false) {
                arguments.extend([
                    OsString::from("--admission-policy"),
                    OsString::from("calibrated-cow"),
                ]);
            }
        }
        if self.native_row_workers > 1 {
            arguments.extend([
                OsString::from("--row-workers"),
                OsString::from(self.native_row_workers.to_string()),
            ]);
        }
        if let Some(family) = self.cpu_family {
            arguments.extend([
                OsString::from("--cpu-family"),
                OsString::from(family.as_str()),
            ]);
        }
        if let Some(mode) = self.prompt_reuse.as_wire() {
            arguments.extend([OsString::from("--prompt-reuse"), OsString::from(mode)]);
        }
        if let Some(checkpoints) = self.prompt_reuse_checkpoints {
            arguments.extend([
                OsString::from("--prompt-reuse-checkpoints"),
                OsString::from(checkpoints.to_string()),
            ]);
        }
        if let Some(bytes) = self.prompt_reuse_checkpoint_bytes {
            arguments.extend([
                OsString::from("--prompt-reuse-checkpoint-bytes"),
                OsString::from(bytes.to_string()),
            ]);
        }
        if let Some(policy) = self.kv_cache {
            arguments.extend([OsString::from("--kv-cache"), OsString::from(policy.name())]);
        }
        if let Some((tokens, scratch_bytes)) = plan.cpu_prefill {
            arguments.extend([
                OsString::from("--cpu-prefill-tokens"),
                OsString::from(tokens.to_string()),
                OsString::from("--cpu-prefill-scratch-bytes"),
                OsString::from(scratch_bytes.to_string()),
            ]);
            if let Some(gdn) = self.cpu_prefill_gdn {
                arguments.extend([
                    OsString::from("--cpu-prefill-gdn"),
                    OsString::from(gdn.as_str()),
                ]);
            }
        }
        if let Some((draft_max, verification_scratch, checkpoint_bytes)) = plan.cpu_speculation {
            arguments.extend([
                OsString::from("--spec-type"),
                OsString::from("suffix"),
                OsString::from("--draft-max"),
                OsString::from(draft_max.to_string()),
                OsString::from("--cpu-verification-scratch-bytes"),
                OsString::from(verification_scratch.to_string()),
                OsString::from("--cpu-speculative-checkpoint-bytes"),
                OsString::from(checkpoint_bytes.to_string()),
            ]);
        } else if let Some(spec_type) = self.spec_type {
            arguments.extend([
                OsString::from("--spec-type"),
                OsString::from(spec_type.as_str()),
            ]);
        }
        if self.placement == crate::inference::Placement::Cpu {
            arguments.extend([OsString::from("--placement"), OsString::from("cpu")]);
        }
        if self.uses_repacked_ptq1() {
            arguments.push("--use-repacked-ptq1".into());
        }
        if let Some(adapter) = &plan.adapter {
            arguments.extend([
                OsString::from("--abliterate-lora"),
                adapter.path.as_os_str().to_owned(),
            ]);
        }
        if let Some(mmproj) = &plan.mmproj {
            arguments.extend([
                OsString::from("--mmproj"),
                mmproj.path.as_os_str().to_owned(),
                OsString::from("--image-max-tokens"),
                OsString::from(bonsai_cpu_policy::DEFAULT_IMAGE_MAX_TOKENS.to_string()),
            ]);
        }
        arguments
    }

    fn specification(
        &self,
        plan: LaunchPlan,
        credentials: &PrivateCredentials,
        native_executable: Option<&std::path::Path>,
        broker_pipe: bool,
    ) -> Result<ProcessSpec, LaunchError> {
        // The worker may write only this child, never the sibling keys or logs.
        // Its parent already has the credential owner's protected inheritable ACL.
        let working_directory = credentials.directory().join("worker");
        std::fs::create_dir(&working_directory).map_err(|_| LaunchError::Path)?;
        validate_external_path(&working_directory).map_err(|_| LaunchError::Path)?;
        let (executable, arguments) = if let Some(executable) = native_executable {
            let key_path = (!broker_pipe).then(|| credentials.backend_path());
            let arguments = self.native_arguments(&plan, key_path);
            (executable.to_path_buf(), arguments)
        } else {
            (plan.executable, plan.arguments)
        };
        Ok(ProcessSpec {
            executable,
            arguments,
            working_directory,
            stdout: credentials.directory().join("backend.stdout"),
            stderr: credentials.directory().join("backend.stderr"),
            memory_limit_bytes: plan.memory_limit_bytes,
            active_process_limit: 1,
        })
    }

    fn start_process(
        &self,
        specification: &ProcessSpec,
        artifacts: &VerifiedArtifacts,
        adapter: Option<&bonsai_artifacts::VerifiedFile>,
        mmproj: Option<&bonsai_artifacts::VerifiedFile>,
        credentials: &PrivateCredentials,
        broker_pipe: bool,
    ) -> Result<
        (
            OwnedProcess,
            Option<(
                std::fs::File,
                std::fs::File,
                Vec<crate::pipe_transport::VerifiedSectionAttestation>,
            )>,
        ),
        LaunchError,
    > {
        if !self.requires_native_sandbox() {
            return OwnedProcess::start(specification)
                .map(|process| (process, None))
                .map_err(|_| LaunchError::Process);
        }
        let mut readonly_paths = artifacts.runtime_files();
        readonly_paths.extend([
            specification.executable.clone(),
            artifacts.model().to_owned(),
        ]);
        if self.native_engine && self.placement == crate::inference::Placement::Auto {
            readonly_paths.extend(hexagon_runtime_readonly_paths(&self.data));
        }
        if !broker_pipe && credentials.has_backend_file() {
            readonly_paths.push(credentials.backend_path().to_owned());
        }
        readonly_paths.extend(adapter.map(|file| file.path().to_owned()));
        readonly_paths.extend(mmproj.map(|file| file.path().to_owned()));
        readonly_paths.sort();
        readonly_paths.dedup();
        let policy = SandboxPolicy {
            readonly_paths,
            writable_directory: specification.working_directory.clone(),
        };
        if broker_pipe {
            let mut section_files: Vec<&std::fs::File> = Vec::with_capacity(3);
            let mut section_digests: Vec<[u8; 32]> = Vec::with_capacity(3);
            let model_hash = if self.uses_repacked_ptq1() {
                crate::inference::REPACKED_PTQ1_SHA256
            } else {
                crate::inference::MODEL_SHA256
            };
            section_files.push(artifacts.model_file().ok_or(LaunchError::Artifacts)?);
            section_digests.push(decode_sha256_hex(model_hash)?);
            if let Some(adapter_file) = adapter {
                section_files.push(adapter_file.file().ok_or(LaunchError::Artifacts)?);
                section_digests.push(decode_sha256_hex(crate::inference::ADAPTER_SHA256)?);
            }
            if let Some(mmproj_file) = mmproj {
                section_files.push(mmproj_file.file().ok_or(LaunchError::Artifacts)?);
                section_digests.push(decode_sha256_hex(crate::inference::MMPROJ_SHA256)?);
            }
            let (process, child_in, child_out, grants) =
                OwnedProcess::start_sandboxed_broker_pipe_with_sections(
                    specification,
                    &policy,
                    &section_files,
                )
                .map_err(|error| {
                    let _result = write_sandbox_failure(&mut io::stderr().lock(), &error);
                    Self::classify_sandbox_start_error(&error)
                })?;
            if grants.len() != section_digests.len() {
                return Err(LaunchError::Process);
            }
            let mut attestations = Vec::with_capacity(grants.len());
            for (index, (grant, sha256)) in grants.into_iter().zip(section_digests).enumerate() {
                let expected_slot = u32::try_from(index).map_err(|_| LaunchError::Process)?;
                if grant.slot != expected_slot || grant.child_handle == 0 || grant.bytes == 0 {
                    return Err(LaunchError::Process);
                }
                attestations.push(crate::pipe_transport::VerifiedSectionAttestation {
                    slot: grant.slot,
                    child_handle: grant.child_handle,
                    bytes: grant.bytes,
                    sha256,
                });
            }
            Ok((process, Some((child_in, child_out, attestations))))
        } else {
            OwnedProcess::start_sandboxed(specification, &policy)
                .map(|process| (process, None))
                .map_err(|error| {
                    // The native boundary emits fixed validation reasons or Windows API
                    // errors, never arguments, file contents or credential values. Keep
                    // that technical cause in the broker's retained stderr while the UI
                    // continues to receive only the value-free launch classification.
                    let _result = write_sandbox_failure(&mut io::stderr().lock(), &error);
                    Self::classify_sandbox_start_error(&error)
                })
        }
    }

    /// Verify pinned artifacts, create credentials, launch and await authenticated readiness.
    ///
    /// The caller supplies cancellation from before startup through serving. No
    /// frontend listener is created until readiness and PID checks pass. On any
    /// startup error or future drop the owned Job is killed. The old daily
    /// profile remains launch-disabled; this is a distinct development opt-in.
    ///
    /// # Errors
    /// Returns a value-free error; technical stdout/stderr remains in the new
    /// protected evidence directory. Failed evidence is retained for inspection.
    pub async fn start(
        self,
        shutdown: impl Future<Output = io::Result<()>>,
    ) -> Result<LaunchedService, LaunchError> {
        self.start_with_progress(shutdown, |_| {}).await
    }

    /// Start with fixed credential-free observations of broker startup brackets.
    ///
    /// The observer receives only fixed phases and elapsed times. It runs at
    /// operation boundaries and should record in memory without blocking. A
    /// started phase remains unfinished on cancellation or error. Readiness
    /// includes unobserved backend verification/loading and health polling;
    /// these observations cannot attribute backend kernel or loader timings.
    /// Synchronous artifact verification remains unpreemptible by async timers.
    ///
    /// # Errors
    /// Returns the same value-free failures and preserves the same ownership,
    /// verification, cancellation and containment behavior as [`Self::start`].
    pub async fn start_with_progress(
        self,
        shutdown: impl Future<Output = io::Result<()>>,
        observe: impl FnMut(StartupProgress),
    ) -> Result<LaunchedService, LaunchError> {
        const MAX_MEMORY_STARTUP_ATTEMPTS: usize = 3;
        // Register cancellation before synchronous policy/artifact work. The CLI
        // retains this same future across startup and serving, so a signal in
        // the readiness-to-serving handoff remains observable.
        let mut shutdown = pin!(shutdown);
        if poll_fn(|cx| Poll::Ready(shutdown.as_mut().poll(cx).is_ready())).await {
            return Err(LaunchError::Cancelled);
        }
        let mut startup_timing = StartupTimer::new(observe);
        startup_timing.phase(StartupPhase::PolicyAndAdmission);
        if self.check_only {
            return Err(LaunchError::Arguments);
        }
        let broker_pipe = self.uses_broker_pipe()?;
        self.check_launch_environment()?;
        self.validate_paths()?;
        // Resolve policy before creating secrets or loading any native code.
        let pending_key = self.evidence_parent.join("not-created.key");
        let preliminary = LaunchPlan::for_profile_with_options(
            &self.data,
            self.backend_port,
            &pending_key,
            self.profile,
            self.use_abliterate_lora,
            self.uses_repacked_ptq1(),
        )
        .map_err(|_| LaunchError::Policy)?;
        self.initial_memory_admission(&preliminary)?;
        if self.native_engine {
            startup_timing.phase(StartupPhase::NativeRuntimeVerification);
        }
        startup_timing.phase(StartupPhase::ModelVerification);
        let VerifiedLaunchGuards {
            native_executable,
            artifacts,
            adapter,
            mmproj,
        } = self.verify_launch_artifacts(&preliminary)?;
        startup_timing.phase(StartupPhase::AdapterVerification);
        if poll_fn(|cx| Poll::Ready(shutdown.as_mut().poll(cx).is_ready())).await {
            return Err(LaunchError::Cancelled);
        }
        startup_timing.phase(StartupPhase::CredentialsAndRuntime);
        let credentials = if broker_pipe {
            PrivateCredentials::create_frontend_only(&self.evidence_parent)
        } else {
            PrivateCredentials::create(&self.evidence_parent)
        }
        .map_err(|_| LaunchError::Credentials)?;
        let (native_executable_path, native_executable) = match native_executable {
            Some((path, guard)) => (Some(path), Some(guard)),
            None => (None, None),
        };
        let (mmproj_artifact, mmproj_file) = match mmproj {
            Some((artifact, guard)) => (Some(artifact), Some(guard)),
            None => (None, None),
        };
        let storage_budget = self.requires_native_sandbox().then_some((
            MAX_WORKER_PRIVATE_STORAGE_BYTES,
            MAX_WORKER_PRIVATE_STORAGE_ENTRIES,
        ));
        let memory_context_supported = self.memory_context_supported()?;
        let max_attempts = if memory_context_supported {
            MAX_MEMORY_STARTUP_ATTEMPTS
        } else {
            1
        };

        for attempt in 0..max_attempts {
            if attempt > 0 {
                if poll_fn(|cx| Poll::Ready(shutdown.as_mut().poll(cx).is_ready())).await {
                    return Err(LaunchError::Cancelled);
                }
                reset_worker_attempt_files(credentials.directory())?;
            }
            let mut plan = LaunchPlan::for_profile_with_options(
                &self.data,
                self.backend_port,
                credentials.backend_path(),
                self.profile,
                self.use_abliterate_lora,
                self.uses_repacked_ptq1(),
            )
            .map_err(|_| LaunchError::Policy)?;
            // Reserve as a collision check in TCP modes before sampling free RAM so
            // socket binding cannot widen the window between context sizing and admission.
            let reservation = if broker_pipe {
                None
            } else {
                Some(bound_loopback_socket(self.backend_port).map_err(|_| LaunchError::Port)?)
            };
            if attempt == 0 {
                preflight_frontend_port(self.frontend_port)?;
            }
            let mut active_mmproj = mmproj_artifact.clone();
            let calibrated_admission = self.native_engine
                && crate::inference::calibrated_cow_admission_supported().unwrap_or(false);
            if memory_context_supported {
                let raw_available = available_memory_bytes().map_err(|_| LaunchError::Memory)?;
                let retry_margin = u64::try_from(attempt)
                    .unwrap_or(0)
                    .saturating_mul(bonsai_cpu_policy::context::STARTUP_RESERVE_BYTES);
                let available = raw_available.saturating_sub(retry_margin);
                if calibrated_admission {
                    let calibrated =
                        match self.calibrated_context_plan(available, active_mmproj.is_some()) {
                            Ok(result) => result,
                            Err(_) if attempt > 0 => self
                                .calibrated_context_plan(raw_available, active_mmproj.is_some())?,
                            Err(_) => return Err(LaunchError::Memory),
                        };
                    if !calibrated.vision_admitted {
                        active_mmproj = None;
                    }
                    plan.select_calibrated_swift_context(&calibrated)
                        .map_err(|_| LaunchError::Policy)?;
                    plan.cpu_prefill = self
                        .cpu_prefill_tokens
                        .zip(calibrated.prefill_scratch_bytes);
                    plan.cpu_speculation = self
                        .effective_speculation_tokens()
                        .zip(calibrated.speculation_reserves)
                        .map(|(tokens, (verify, checkpoint))| (tokens, verify, checkpoint));
                } else {
                    if self.effective_speculation_tokens().is_some()
                        || self.cpu_prefill_gdn.is_some()
                        || self.prompt_reuse
                            == crate::inference::PromptReuseMode::PromptEndAndLatest
                        || self.prompt_reuse_checkpoints.is_some()
                        || self.prompt_reuse_checkpoint_bytes.is_some()
                    {
                        return Err(LaunchError::NativeRuntime);
                    }
                    let profile = self.swift_context_profile()?;
                    let (context, vision_admitted, prefill_scratch) =
                        match bonsai_cpu_policy::context::SwiftContextPlan::for_profile_with_options(
                            available,
                            profile,
                            active_mmproj.is_some(),
                            self.cpu_prefill_tokens.map(usize::from),
                        ) {
                            Ok(result) => result,
                            Err(_) if attempt > 0 => {
                                // If the extra retry margin pushed below minimum context, try raw available RAM.
                                bonsai_cpu_policy::context::SwiftContextPlan::for_profile_with_options(
                                    raw_available,
                                    profile,
                                    active_mmproj.is_some(),
                                    self.cpu_prefill_tokens.map(usize::from),
                                )
                                .map_err(|_| LaunchError::Memory)?
                            }
                            Err(_) => return Err(LaunchError::Memory),
                        };
                    if !vision_admitted {
                        active_mmproj = None;
                    }
                    plan.select_swift_context(context)
                        .map_err(|_| LaunchError::Policy)?;
                    plan.cpu_prefill = self.cpu_prefill_tokens.zip(prefill_scratch);
                }
            }
            if let Some(artifact) = &active_mmproj {
                plan.enable_swift_mmproj(artifact.clone())
                    .map_err(|_| LaunchError::Policy)?;
            }
            let vision_enabled = active_mmproj.is_some();
            let active_mmproj_guard = vision_enabled.then(|| mmproj_file.as_ref()).flatten();
            let context_tokens = plan.context_tokens;
            let cpu_prefill = plan.cpu_prefill;
            let cpu_speculation = plan.cpu_speculation;
            let working_set_limit_bytes = plan.working_set_limit_bytes;
            if let Err(error) = admit(plan.required_available_bytes) {
                drop(reservation);
                if error == LaunchError::Memory && attempt + 1 < max_attempts {
                    continue;
                }
                return Err(error);
            }
            let specification = self.specification(
                plan,
                &credentials,
                native_executable_path.as_deref(),
                broker_pipe,
            )?;
            validate_external_path(&specification.executable).map_err(|_| LaunchError::Path)?;
            drop(reservation);
            let worker_root = specification.working_directory.clone();
            if attempt == 0 {
                startup_timing.phase(StartupPhase::ProcessCreation);
            }
            let (process, pipe_handles) = self.start_process(
                &specification,
                &artifacts,
                adapter.as_ref(),
                active_mmproj_guard,
                &credentials,
                broker_pipe,
            )?;
            if let Err(guard_error) = guard(
                &process,
                specification.memory_limit_bytes,
                working_set_limit_bytes,
                storage_budget,
                Some(worker_root.as_path()),
            ) {
                // Keep sandbox storage rejections distinct if the check tripped right after spawn,
                // without discarding Memory or Sandbox errors returned by guard itself.
                let storage_error =
                    check_storage_budget(&process, storage_budget, Some(worker_root.as_path()))
                        .err();
                let err = classify_initial_guard_error(guard_error, storage_error);
                let status = wait_briefly_for_exit(&process, err);
                process.stop(STOP_TIMEOUT).map_err(|_| LaunchError::Stop)?;
                let (refined, child_stage) =
                    refine_backend_startup_error(err, &specification.stderr);
                if (refined == LaunchError::Memory
                    || child_stage == Some(ChildStartupStage::Memory))
                    && attempt + 1 < max_attempts
                {
                    continue;
                }
                if refined != LaunchError::Sandbox {
                    let _result = write_startup_failure(
                        &mut io::stderr().lock(),
                        StartupStage::AuthenticatedReadiness,
                        refined,
                        &status,
                    );
                }
                return Err(refined);
            }
            if attempt == 0 {
                startup_timing.phase(StartupPhase::AuthenticatedReadiness);
            }
            let readiness = if let Some((child_stdin, child_stdout, attestations)) = pipe_handles {
                match crate::pipe_transport::PipeBackend::connect_stdio_with_sections(
                    child_stdin,
                    child_stdout,
                    credentials.backend_key(),
                    &attestations,
                ) {
                    Ok(pipe_backend) => wait_ready_models_pipe(
                        &process,
                        &pipe_backend,
                        credentials.backend_key(),
                        specification.memory_limit_bytes,
                        working_set_limit_bytes,
                        storage_budget,
                        Some(worker_root.as_path()),
                        shutdown.as_mut(),
                    )
                    .await
                    .map(|models| (Some(pipe_backend), models)),
                    Err(_) => Err(LaunchError::Backend),
                }
            } else {
                wait_ready_models(
                    &process,
                    self.backend_port,
                    credentials.backend_key(),
                    specification.memory_limit_bytes,
                    working_set_limit_bytes,
                    storage_budget,
                    Some(worker_root.as_path()),
                    shutdown.as_mut(),
                )
                .await
                .map(|models| (None, models))
            };
            let (pipe_backend, models) = match readiness {
                Ok(ready) => ready,
                Err(error) => {
                    let status = wait_briefly_for_exit(&process, error);
                    process.stop(STOP_TIMEOUT).map_err(|_| LaunchError::Stop)?;
                    let (refined, child_stage) =
                        refine_backend_startup_error(error, &specification.stderr);
                    if (refined == LaunchError::Memory
                        || child_stage == Some(ChildStartupStage::Memory))
                        && attempt + 1 < max_attempts
                    {
                        continue;
                    }
                    let _result = write_startup_failure(
                        &mut io::stderr().lock(),
                        StartupStage::AuthenticatedReadiness,
                        refined,
                        &status,
                    );
                    return Err(refined);
                }
            };
            if self.native_engine {
                startup_timing.phase(StartupPhase::NativeIdentity);
                self.confirm_native_identity(
                    &process,
                    &models,
                    context_tokens,
                    vision_enabled,
                    cpu_prefill,
                    cpu_speculation,
                )?;
            }
            startup_timing.phase(StartupPhase::FrontendBinding);
            let service = self
                .bind_frontend(
                    &process,
                    &credentials,
                    pipe_backend,
                    adapter.is_some(),
                    context_tokens,
                    vision_enabled,
                )
                .await?;
            startup_timing.finish();
            return Ok(LaunchedService {
                service,
                process,
                credentials,
                _artifacts: artifacts,
                _adapter: adapter,
                _mmproj: vision_enabled.then_some(mmproj_file).flatten(),
                _native_executable: native_executable,
                memory_limit_bytes: specification.memory_limit_bytes,
                working_set_limit_bytes,
                backend_port: if broker_pipe { 0 } else { self.backend_port },
                storage_budget,
            });
        }
        Err(LaunchError::Memory)
    }

    async fn bind_frontend(
        &self,
        process: &OwnedProcess,
        credentials: &PrivateCredentials,
        pipe_backend: Option<crate::pipe_transport::PipeBackend>,
        adapter: bool,
        context_tokens: usize,
        vision_enabled: bool,
    ) -> Result<LocalChatService, LaunchError> {
        let bound = if let Some(pipe_backend) = pipe_backend {
            LocalChatService::bind_pipe(
                self.frontend_port,
                credentials.frontend_key(),
                credentials.backend_key(),
                pipe_backend,
                self.max_waiting,
            )
            .await
        } else {
            LocalChatService::bind(
                self.frontend_port,
                credentials.frontend_key(),
                self.backend_port,
                credentials.backend_key(),
                self.max_waiting,
            )
            .await
        };
        let Ok(mut service) = bound else {
            process.stop(STOP_TIMEOUT).map_err(|_| LaunchError::Stop)?;
            return Err(LaunchError::Service);
        };
        if service.select_context_tokens(context_tokens).is_err() {
            process.stop(STOP_TIMEOUT).map_err(|_| LaunchError::Stop)?;
            return Err(LaunchError::Service);
        }
        service.require_backend_process(process.id());
        if adapter {
            service.enable_pinned_abliteration();
        }
        if vision_enabled {
            service.enable_vision();
        }
        if self.reset_on_cancel {
            service.enable_reset_on_cancel();
        }
        Ok(service)
    }

    fn confirm_native_identity(
        &self,
        process: &OwnedProcess,
        models: &[u8],
        context_tokens: usize,
        vision_enabled: bool,
        cpu_prefill: Option<(u8, usize)>,
        cpu_speculation: Option<(u8, usize, usize)>,
    ) -> Result<(), LaunchError> {
        let verification = crate::inference::pinned_version()
            .map_err(|_| NativeIdentityError::ReleasePin)
            .and_then(|version| {
                validate_native_identity_detailed(
                    models,
                    self.uses_repacked_ptq1(),
                    self.use_abliterate_lora,
                    WorkerReadiness::for_version(&version, self.native_row_workers)
                        .with_prompt_reuse_mode(self.prompt_reuse)
                        .with_prompt_reuse_checkpoints(self.prompt_reuse_checkpoints.unwrap_or(1))
                        .with_vision(vision_enabled)
                        .with_kv_cache(self.kv_cache)
                        .with_cpu_prefill(cpu_prefill)
                        .with_cpu_prefill_gdn(
                            self.cpu_prefill_gdn
                                .is_some_and(crate::inference::CpuPrefillGdn::is_wy_f32),
                        )
                        .with_cpu_speculation(cpu_speculation)
                        .with_placement(self.placement)
                        .with_context_tokens(context_tokens),
                )
            });
        if let Err(mismatch) = verification {
            note_startup_failure(StartupStage::NativeIdentity, LaunchError::Backend, process);
            let _result = write_native_identity_failure(&mut io::stderr().lock(), mismatch);
            process.stop(STOP_TIMEOUT).map_err(|_| LaunchError::Stop)?;
            return Err(LaunchError::Backend);
        }
        Ok(())
    }
}

fn preflight_frontend_port(port: u16) -> Result<(), LaunchError> {
    if port != 0 {
        drop(bound_loopback_socket(port).map_err(|_| LaunchError::Service)?);
    }
    Ok(())
}

const fn classify_initial_guard_error(
    guard_error: LaunchError,
    storage_error: Option<LaunchError>,
) -> LaunchError {
    if matches!(guard_error, LaunchError::Backend)
        && let Some(storage_error) = storage_error
    {
        storage_error
    } else {
        guard_error
    }
}

#[derive(Clone, Copy)]
enum StartupStage {
    AuthenticatedReadiness,
    NativeIdentity,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ChildStartupStage {
    Arguments,
    Environment,
    Memory,
    Key,
    Model,
    Runtime,
    Listener,
    Backend,
    Unknown,
}

impl ChildStartupStage {
    const fn launch_error(self) -> LaunchError {
        match self {
            Self::Memory => LaunchError::Memory,
            Self::Environment => LaunchError::Policy,
            Self::Key => LaunchError::Credentials,
            Self::Listener => LaunchError::Port,
            Self::Arguments | Self::Model | Self::Runtime | Self::Backend | Self::Unknown => {
                LaunchError::Backend
            }
        }
    }
}

const MAX_CHILD_STDERR_CLASSIFY_BYTES: u64 = 65_536;

fn parse_child_startup_failure(stderr: &[u8]) -> Option<ChildStartupStage> {
    for raw_line in stderr.split(|&byte| byte == b'\n') {
        let raw_line = raw_line.strip_suffix(b"\r").unwrap_or(raw_line);
        let Ok(line) = std::str::from_utf8(raw_line) else {
            continue;
        };
        let Some(rest) = line.strip_prefix("private-inference-startup-failed stage=") else {
            continue;
        };
        let Some((stage, rest)) = rest.split_once(" errorKind=") else {
            continue;
        };
        let Some((kind, os_code)) = rest.split_once(" osCode=") else {
            continue;
        };
        if kind.is_empty()
            || !kind.bytes().all(|byte| byte.is_ascii_alphanumeric())
            || os_code.is_empty()
            || !(os_code == "none"
                || os_code
                    .strip_prefix('-')
                    .unwrap_or(os_code)
                    .bytes()
                    .all(|byte| byte.is_ascii_digit()))
        {
            continue;
        }
        let parsed = match stage {
            "arguments" => ChildStartupStage::Arguments,
            "environment" => ChildStartupStage::Environment,
            "memory" => ChildStartupStage::Memory,
            "key" => ChildStartupStage::Key,
            "model" => ChildStartupStage::Model,
            "runtime" => ChildStartupStage::Runtime,
            "listener" => ChildStartupStage::Listener,
            "backend" => ChildStartupStage::Backend,
            "unknown" => ChildStartupStage::Unknown,
            _ => continue,
        };
        return Some(parsed);
    }
    None
}

fn classify_child_startup_stderr(stderr_path: &std::path::Path) -> Option<ChildStartupStage> {
    use std::io::Read as _;
    let file = std::fs::File::open(stderr_path).ok()?;
    let mut bytes = Vec::with_capacity(256);
    file.take(MAX_CHILD_STDERR_CLASSIFY_BYTES)
        .read_to_end(&mut bytes)
        .ok()?;
    parse_child_startup_failure(&bytes)
}

fn refine_backend_startup_error(
    error: LaunchError,
    stderr_path: &std::path::Path,
) -> (LaunchError, Option<ChildStartupStage>) {
    if error != LaunchError::Backend {
        return (error, None);
    }
    if let Some(stage) = classify_child_startup_stderr(stderr_path) {
        (stage.launch_error(), Some(stage))
    } else {
        (error, None)
    }
}

fn wait_briefly_for_exit(process: &OwnedProcess, error: LaunchError) -> io::Result<Option<u32>> {
    let mut status = process.try_wait();
    if error == LaunchError::Backend && matches!(status, Ok(None)) {
        for _ in 0..10 {
            std::thread::sleep(Duration::from_millis(5));
            status = process.try_wait();
            if !matches!(status, Ok(None)) {
                break;
            }
        }
    }
    status
}

fn reset_worker_attempt_files(credentials_directory: &std::path::Path) -> Result<(), LaunchError> {
    validate_external_path(credentials_directory).map_err(|_| LaunchError::Path)?;
    for file in ["backend.stdout", "backend.stderr"] {
        let path = credentials_directory.join(file);
        match std::fs::remove_file(&path) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(_) => return Err(LaunchError::Path),
        }
    }
    remove_worker_directory(&credentials_directory.join("worker"))
}

fn remove_worker_directory(path: &std::path::Path) -> Result<(), LaunchError> {
    validate_external_path(path).map_err(|_| LaunchError::Path)?;
    let metadata = match std::fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(_) => return Err(LaunchError::Path),
    };
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(LaunchError::Path);
    }
    for entry in std::fs::read_dir(path).map_err(|_| LaunchError::Path)? {
        let entry = entry.map_err(|_| LaunchError::Path)?;
        let child = entry.path();
        validate_external_path(&child).map_err(|_| LaunchError::Path)?;
        let child_meta = std::fs::symlink_metadata(&child).map_err(|_| LaunchError::Path)?;
        if child_meta.file_type().is_symlink() {
            return Err(LaunchError::Path);
        }
        if child_meta.is_dir() {
            remove_worker_directory(&child)?;
        } else {
            std::fs::remove_file(&child).map_err(|_| LaunchError::Path)?;
        }
    }
    std::fs::remove_dir(path).map_err(|_| LaunchError::Path)
}

fn note_startup_failure(stage: StartupStage, error: LaunchError, process: &OwnedProcess) {
    let _result =
        write_startup_failure(&mut io::stderr().lock(), stage, error, &process.try_wait());
}

fn write_startup_failure(
    output: &mut impl io::Write,
    stage: StartupStage,
    error: LaunchError,
    status: &io::Result<Option<u32>>,
) -> io::Result<()> {
    let stage = match stage {
        StartupStage::AuthenticatedReadiness => "authenticated-readiness",
        StartupStage::NativeIdentity => "native-identity",
    };
    let marker_error = match error {
        LaunchError::Memory | LaunchError::Cancelled => error,
        _ => LaunchError::Backend,
    };
    write!(
        output,
        "Native backend startup failure: stage={stage}, error={marker_error:?}, process="
    )?;
    match status {
        Ok(Some(code)) => writeln!(output, "exited-0x{code:08X}"),
        Ok(None) => writeln!(output, "running"),
        Err(_) => writeln!(output, "status-unavailable"),
    }
}

fn write_native_identity_failure(
    output: &mut impl io::Write,
    error: NativeIdentityError,
) -> io::Result<()> {
    writeln!(
        output,
        "Native identity verification failed: reason={}.",
        error.reason()
    )
}

fn write_sandbox_failure(output: &mut impl io::Write, error: &io::Error) -> io::Result<()> {
    writeln!(
        output,
        "Native sandbox launch failed (kind={:?}, os={:?}): {error}",
        error.kind(),
        error.raw_os_error()
    )
}

/// One ready development backend with retained artifact and credential handles.
pub struct LaunchedService {
    service: LocalChatService,
    process: OwnedProcess,
    credentials: PrivateCredentials,
    _artifacts: VerifiedArtifacts,
    _adapter: Option<bonsai_artifacts::VerifiedFile>,
    _mmproj: Option<bonsai_artifacts::VerifiedFile>,
    _native_executable: Option<bonsai_artifacts::VerifiedFile>,
    memory_limit_bytes: usize,
    working_set_limit_bytes: usize,
    backend_port: u16,
    storage_budget: Option<(u64, usize)>,
}

impl LaunchedService {
    /// Bound frontend port, available only after the owned backend is ready.
    pub const fn port(&self) -> u16 {
        self.service.port()
    }

    /// Actual immutable context shared by runtime, exact admission and UI guidance.
    pub const fn context_tokens(&self) -> usize {
        self.service.context_tokens()
    }

    /// Whether the active supervised backend admits multimodal image context.
    pub const fn vision_enabled(&self) -> bool {
        self.service.vision_enabled()
    }

    /// Protected evidence directory; contains credentials, so share only selected logs.
    pub fn evidence_directory(&self) -> &std::path::Path {
        self.credentials.directory()
    }

    /// Serve while observing process exit, listener PID and physical/process memory.
    ///
    /// Every exit closes admission, drains owned sockets, terminates the complete
    /// Job and confirms it is empty. Interrupted generation ends this instance
    /// unless `--reset-on-cancel` was selected and the owned native backend
    /// acknowledged `POST /reset`.
    ///
    /// # Errors
    /// Returns resource/backend/cleanup failure after shutting down the service.
    pub async fn run(
        self,
        shutdown: impl Future<Output = io::Result<()>>,
    ) -> Result<RunReport, LaunchError> {
        let failure = Cell::new(None);
        let resources = Cell::new(BackendResources::default());
        let mut shutdown = pin!(shutdown);
        let worker_root = self.credentials.directory().join("worker");
        let monitoring = async {
            let mut tick = interval(Duration::from_millis(250));
            loop {
                tick.tick().await;
                let status = guard(
                    &self.process,
                    self.memory_limit_bytes,
                    self.working_set_limit_bytes,
                    self.storage_budget,
                    Some(worker_root.as_path()),
                )
                .and_then(|(memory, available)| {
                    resources.set(resource_sample(resources.get(), memory, available));
                    if self.backend_port == 0 {
                        Ok(())
                    } else {
                        owner(&self.process, self.backend_port)
                    }
                });
                if let Err(error) = status {
                    failure.set(Some(error));
                    return Err(io::Error::other("Owned backend guard failed."));
                }
            }
        };
        let mut monitoring = pin!(monitoring);
        let monitor = poll_fn(|cx| {
            if let Poll::Ready(result) = shutdown.as_mut().poll(cx) {
                return Poll::Ready(result);
            }
            monitoring.as_mut().poll(cx)
        });
        let mut report = run(&self.service, monitor).await;
        // Query before stopping the process. The OS Job peak includes startup
        // and descendants; root working-set residency has a different scope.
        // An unavailable observation never bypasses or delays owned cleanup.
        let mut observed = resources.get();
        observed.os_peak_job_commit_bytes = self.process.peak_job_memory_bytes().ok();
        observed.os_peak_root_working_set_bytes = self
            .process
            .memory_usage()
            .ok()
            .map(|memory| memory.peak_working_set_bytes);
        self.process
            .stop(STOP_TIMEOUT)
            .map_err(|_| LaunchError::Stop)?;
        if let Some(error) = failure.get() {
            return Err(error);
        }
        if report.exit == RunExit::ShutdownFailure {
            return Err(LaunchError::Cancelled);
        }
        report.backend_resources = Some(observed);
        Ok(report)
    }
}

fn row_workers(value: Option<&std::ffi::OsStr>) -> Result<u8, LaunchError> {
    match value.and_then(std::ffi::OsStr::to_str) {
        Some("1") => Ok(1),
        Some("4") => Ok(4),
        Some("8") => Ok(8),
        Some("12") => Ok(12),
        _ => Err(LaunchError::Arguments),
    }
}

fn canonical_decimal_str(value: Option<OsString>) -> Result<String, LaunchError> {
    let text = value
        .and_then(|raw| raw.into_string().ok())
        .ok_or(LaunchError::Arguments)?;
    if text.is_empty()
        || (text.len() > 1 && text.starts_with('0'))
        || !text.bytes().all(|byte| byte.is_ascii_digit())
    {
        return Err(LaunchError::Arguments);
    }
    Ok(text)
}

fn decimal_range_u8(
    value: Option<OsString>,
    range: std::ops::RangeInclusive<u8>,
) -> Result<u8, LaunchError> {
    let parsed = canonical_decimal_str(value)?
        .parse::<u8>()
        .map_err(|_| LaunchError::Arguments)?;
    if !range.contains(&parsed) {
        return Err(LaunchError::Arguments);
    }
    Ok(parsed)
}

fn positive_decimal_usize(value: Option<OsString>) -> Result<usize, LaunchError> {
    let parsed = canonical_decimal_str(value)?
        .parse::<usize>()
        .map_err(|_| LaunchError::Arguments)?;
    if parsed == 0 {
        return Err(LaunchError::Arguments);
    }
    Ok(parsed)
}

fn launch_profile(value: Option<&std::ffi::OsStr>) -> Result<LaunchProfile, LaunchError> {
    value
        .and_then(std::ffi::OsStr::to_str)
        .and_then(|name| name.parse().ok())
        .ok_or(LaunchError::Arguments)
}

const fn containment(
    launch: bool,
    unisolated: bool,
    sandboxed: bool,
) -> Result<BackendContainment, LaunchError> {
    if !launch || unisolated == sandboxed {
        return Err(LaunchError::Arguments);
    }
    Ok(if sandboxed {
        BackendContainment::NativeSandbox
    } else {
        BackendContainment::ExplicitUnisolated
    })
}

fn number(value: Option<OsString>) -> Result<u16, LaunchError> {
    crate::configuration::decimal_u16(value).ok_or(LaunchError::Arguments)
}

fn decode_sha256_hex(hex: &str) -> Result<[u8; 32], LaunchError> {
    let bytes = hex.as_bytes();
    if bytes.len() != 64 {
        return Err(LaunchError::Artifacts);
    }
    let mut digest = [0_u8; 32];
    for (index, pair) in bytes.as_chunks::<2>().0.iter().enumerate() {
        let nibble = |byte: u8| -> Result<u8, LaunchError> {
            match byte {
                b'0'..=b'9' => Ok(byte - b'0'),
                b'a'..=b'f' => Ok(byte - b'a' + 10),
                _ => Err(LaunchError::Artifacts),
            }
        };
        let slot = digest.get_mut(index).ok_or(LaunchError::Artifacts)?;
        *slot = (nibble(pair[0])? << 4) | nibble(pair[1])?;
    }
    Ok(digest)
}

fn hexagon_runtime_readonly_paths(data: &std::path::Path) -> Vec<PathBuf> {
    const HEXAGON_RUNTIME_FILES: [&str; 3] = [
        "libggml-htp-v81.so",
        "libggml-htp-v73.so",
        "hexagon-runtime.receipt.json",
    ];
    let runtime_dir = data.join("hexagon").join("runtime");
    HEXAGON_RUNTIME_FILES
        .into_iter()
        .map(|name| runtime_dir.join(name))
        .filter(|path| path.is_file())
        .collect()
}

fn admit(required: u64) -> Result<(), LaunchError> {
    if available_memory_bytes().map_err(|_| LaunchError::Memory)? < required {
        return Err(LaunchError::Memory);
    }
    Ok(())
}

fn check_storage_budget(
    process: &OwnedProcess,
    storage_budget: Option<(u64, usize)>,
    worker_root: Option<&std::path::Path>,
) -> Result<(), LaunchError> {
    if let Some((max_bytes, max_entries)) = storage_budget {
        process
            .check_private_storage_budget(max_bytes, max_entries)
            .map_err(|_| LaunchError::Sandbox)?;
        if let Some(worker_root) = worker_root {
            let roots = [worker_root.to_path_buf()];
            bonsai_windows_process::check_private_storage_roots(&roots, max_bytes, max_entries)
                .map_err(|_| LaunchError::Sandbox)?;
        }
    }
    Ok(())
}

fn guard(
    process: &OwnedProcess,
    private_limit: usize,
    working_set_limit: usize,
    storage_budget: Option<(u64, usize)>,
    worker_root: Option<&std::path::Path>,
) -> Result<(ProcessMemory, u64), LaunchError> {
    if process
        .try_wait()
        .map_err(|_| LaunchError::Backend)?
        .is_some()
    {
        return Err(LaunchError::Backend);
    }
    let memory = process.memory_usage().map_err(|_| LaunchError::Memory)?;
    if memory.private_bytes > private_limit || memory.working_set_bytes > working_set_limit {
        return Err(LaunchError::Memory);
    }
    check_storage_budget(process, storage_budget, worker_root)?;
    let available = available_memory_bytes().map_err(|_| LaunchError::Memory)?;
    if available < PHYSICAL_RESERVE {
        return Err(LaunchError::Memory);
    }
    Ok((memory, available))
}

fn resource_sample(
    mut observed: BackendResources,
    memory: ProcessMemory,
    available_physical_bytes: u64,
) -> BackendResources {
    observed.sample_count = observed.sample_count.saturating_add(1);
    observed.sampled_peak_root_private_bytes = observed
        .sampled_peak_root_private_bytes
        .max(memory.private_bytes);
    observed.sampled_peak_root_working_set_bytes = observed
        .sampled_peak_root_working_set_bytes
        .max(memory.working_set_bytes);
    observed.sampled_min_available_physical_bytes = Some(
        observed
            .sampled_min_available_physical_bytes
            .map_or(available_physical_bytes, |previous| {
                previous.min(available_physical_bytes)
            }),
    );
    observed
}

fn owner(process: &OwnedProcess, port: u16) -> Result<(), LaunchError> {
    if listener_process_id(port).map_err(|_| LaunchError::Backend)? != Some(process.id()) {
        return Err(LaunchError::Backend);
    }
    Ok(())
}

async fn run_startup_with_timeout<T>(
    shutdown: impl Future<Output = io::Result<()>>,
    startup: impl Future<Output = Result<T, LaunchError>>,
) -> Result<T, LaunchError> {
    let mut shutdown = pin!(shutdown);
    let mut startup = pin!(timeout(STARTUP_TIMEOUT, startup));
    poll_fn(|cx| {
        if shutdown.as_mut().poll(cx).is_ready() {
            return Poll::Ready(Err(LaunchError::Cancelled));
        }
        startup
            .as_mut()
            .poll(cx)
            .map(|result| result.map_err(|_| LaunchError::Backend)?)
    })
    .await
}

async fn wait_ready_models(
    process: &OwnedProcess,
    port: u16,
    key: &str,
    private_limit: usize,
    working_set_limit: usize,
    storage_budget: Option<(u64, usize)>,
    worker_root: Option<&std::path::Path>,
    shutdown: impl Future<Output = io::Result<()>>,
) -> Result<Vec<u8>, LaunchError> {
    run_startup_with_timeout(shutdown, async {
        loop {
            guard(
                process,
                private_limit,
                working_set_limit,
                storage_budget,
                worker_root,
            )?;
            match listener_process_id(port).map_err(|_| LaunchError::Backend)? {
                Some(pid) if pid != process.id() => return Err(LaunchError::Backend),
                Some(_) if healthy(port, key, process.id()).await => {
                    if let Some(models) = authenticated_models(port, key, process.id()).await {
                        owner(process, port)?;
                        guard(
                            process,
                            private_limit,
                            working_set_limit,
                            storage_budget,
                            worker_root,
                        )?;
                        return Ok(models);
                    }
                }
                _ => {}
            }
            sleep(Duration::from_millis(100)).await;
        }
    })
    .await
}

async fn wait_ready_models_pipe(
    process: &OwnedProcess,
    backend: &crate::pipe_transport::PipeBackend,
    key: &str,
    private_limit: usize,
    working_set_limit: usize,
    storage_budget: Option<(u64, usize)>,
    worker_root: Option<&std::path::Path>,
    shutdown: impl Future<Output = io::Result<()>>,
) -> Result<Vec<u8>, LaunchError> {
    run_startup_with_timeout(shutdown, async {
        guard(
            process,
            private_limit,
            working_set_limit,
            storage_budget,
            worker_root,
        )?;
        {
            let ready_deadline = tokio::time::Instant::now() + STARTUP_TIMEOUT;
            let mut ready = pin!(backend.wait_server_ready(ready_deadline));
            let mut tick = interval(Duration::from_millis(250));
            poll_fn(|cx| {
                if let Poll::Ready(result) = ready.as_mut().poll(cx) {
                    return Poll::Ready(match result {
                        Ok(()) => Ok(()),
                        Err(_) => Err(guard(
                            process,
                            private_limit,
                            working_set_limit,
                            storage_budget,
                            worker_root,
                        )
                        .err()
                        .unwrap_or(LaunchError::Backend)),
                    });
                }
                while tick.poll_tick(cx).is_ready() {
                    if let Err(error) = guard(
                        process,
                        private_limit,
                        working_set_limit,
                        storage_budget,
                        worker_root,
                    ) {
                        return Poll::Ready(Err(error));
                    }
                }
                Poll::Pending
            })
            .await?;
        }
        guard(
            process,
            private_limit,
            working_set_limit,
            storage_budget,
            worker_root,
        )?;
        let health = pipe_exchange(backend, "/health", Some(key))
            .await
            .and_then(|bytes| validate_health(&bytes))
            .is_ok();
        if !health {
            return Err(guard(
                process,
                private_limit,
                working_set_limit,
                storage_budget,
                worker_root,
            )
            .err()
            .unwrap_or(LaunchError::Backend));
        }
        let models = pipe_authenticated_models(backend, key)
            .await
            .ok_or_else(|| {
                guard(
                    process,
                    private_limit,
                    working_set_limit,
                    storage_budget,
                    worker_root,
                )
                .err()
                .unwrap_or(LaunchError::Backend)
            })?;
        guard(
            process,
            private_limit,
            working_set_limit,
            storage_budget,
            worker_root,
        )?;
        Ok(models)
    })
    .await
}

#[cfg(test)]
async fn wait_ready(
    process: &OwnedProcess,
    port: u16,
    key: &str,
    limit: usize,
    shutdown: impl Future<Output = io::Result<()>>,
) -> Result<(), LaunchError> {
    wait_ready_models(process, port, key, limit, limit, None, None, shutdown)
        .await
        .map(|_| ())
}

async fn healthy(port: u16, key: &str, process_id: u32) -> bool {
    timeout(
        Duration::from_secs(1),
        health_exchange(port, key, process_id),
    )
    .await
    .is_ok_and(|result| result.is_ok())
}

async fn health_exchange(port: u16, key: &str, process_id: u32) -> io::Result<()> {
    // The native Swift worker requires the bearer credential on every route,
    // including /health; the publisher engine ignores it on public /health.
    validate_health(&exchange(port, "/health", Some(key), process_id).await?)
}

fn validate_models_pair(denied: &[u8], allowed: &[u8]) -> io::Result<()> {
    validate_response(denied, 401)?;
    let body = validate_response(allowed, 200)?;
    let models: serde_json::Value = serde_json::from_slice(body).map_err(io::Error::other)?;
    if !models.get("data").is_some_and(serde_json::Value::is_array) {
        return Err(io::Error::other("Backend authentication probe failed."));
    }
    Ok(())
}

async fn authenticated_models(port: u16, key: &str, process_id: u32) -> Option<Vec<u8>> {
    timeout(Duration::from_secs(2), async {
        // Publisher /health is public while native /health requires the key.
        // Either way, /v1/models must reject missing credentials and accept
        // this newly generated key.
        let denied = exchange(port, "/v1/models", None, process_id).await?;
        let allowed = exchange(port, "/v1/models", Some(key), process_id).await?;
        validate_models_pair(&denied, &allowed)?;
        Ok::<Vec<u8>, io::Error>(allowed)
    })
    .await
    .ok()
    .and_then(Result::ok)
}

async fn pipe_authenticated_models(
    backend: &crate::pipe_transport::PipeBackend,
    key: &str,
) -> Option<Vec<u8>> {
    let denied = pipe_exchange(backend, "/v1/models", None).await.ok()?;
    let allowed = pipe_exchange(backend, "/v1/models", Some(key)).await.ok()?;
    validate_models_pair(&denied, &allowed).ok()?;
    Some(allowed)
}

#[cfg(test)]
async fn authenticated(port: u16, key: &str, process_id: u32) -> bool {
    authenticated_models(port, key, process_id).await.is_some()
}

async fn pipe_exchange(
    backend: &crate::pipe_transport::PipeBackend,
    path: &str,
    key: Option<&str>,
) -> io::Result<Vec<u8>> {
    timeout(Duration::from_secs(2), async {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
        let stream = backend
            .connect(deadline)
            .await
            .map_err(|_| io::Error::other("Broker-pipe connect failed."))?;
        exchange_http(
            stream,
            crate::pipe_transport::BROKER_PIPE_HOST_PORT,
            path,
            key,
        )
        .await
    })
    .await
    .map_err(|_| io::Error::new(io::ErrorKind::TimedOut, "Broker-pipe readiness timed out."))?
}

async fn exchange_http<S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin>(
    mut stream: S,
    port: u16,
    path: &str,
    key: Option<&str>,
) -> io::Result<Vec<u8>> {
    let head = format!("GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n");
    stream.write_all(head.as_bytes()).await?;
    if let Some(key) = key {
        stream.write_all(b"Authorization: Bearer ").await?;
        stream.write_all(key.as_bytes()).await?;
        stream.write_all(b"\r\n").await?;
    }
    stream.write_all(b"Connection: close\r\n\r\n").await?;
    stream.flush().await?;
    let mut bytes = Vec::with_capacity(1024);
    stream
        .take(u64::try_from(HEALTH_LIMIT + 1).map_err(io::Error::other)?)
        .read_to_end(&mut bytes)
        .await?;
    Ok(bytes)
}

async fn exchange(
    port: u16,
    path: &str,
    key: Option<&str>,
    process_id: u32,
) -> io::Result<Vec<u8>> {
    let check_owner = || {
        if listener_process_id(port)? != Some(process_id) {
            return Err(io::Error::other("Readiness listener owner changed."));
        }
        Ok(())
    };
    check_owner()?;
    let stream = TcpStream::connect(SocketAddrV4::new(Ipv4Addr::LOCALHOST, port)).await?;
    check_owner()?;
    exchange_http(stream, port, path, key).await
}

fn validate_health(bytes: &[u8]) -> io::Result<()> {
    #[derive(serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Health<'a> {
        #[serde(borrow)]
        status: std::borrow::Cow<'a, str>,
    }
    let body = validate_response(bytes, 200)?;
    let crate::Object(value): crate::Object<Health<'_>> =
        serde_json::from_slice(body).map_err(io::Error::other)?;
    if value.status != "ok" {
        return Err(io::Error::other("Backend is not ready."));
    }
    Ok(())
}

fn validate_response(bytes: &[u8], status: u16) -> io::Result<&[u8]> {
    let invalid = || io::Error::other("Backend is not ready.");
    if bytes.len() > HEALTH_LIMIT {
        return Err(invalid());
    }
    let mut headers = [httparse::EMPTY_HEADER; crate::MAX_HEADERS];
    let mut response = httparse::Response::new(&mut headers);
    let httparse::Status::Complete(head) = response.parse(bytes).map_err(|_| invalid())? else {
        return Err(invalid());
    };
    if head > crate::MAX_HEADER_BYTES
        || response.code != Some(status)
        || response.version != Some(1)
    {
        return Err(invalid());
    }
    let headers = &*response.headers;
    if crate::wire::has_header(headers, "transfer-encoding") {
        return Err(invalid());
    }
    let length = crate::wire::singleton_slice(headers, "content-length")
        .map_err(|_| invalid())?
        .map(crate::wire::content_length)
        .transpose()
        .map_err(|_| invalid())?;
    let content_type =
        crate::wire::singleton_slice(headers, "content-type").map_err(|_| invalid())?;
    let body = &bytes[head..];
    if body.len() > 4096
        || length != Some(body.len())
        || content_type.is_none_or(|value| {
            value != b"application/json" && value != b"application/json; charset=utf-8"
        })
    {
        return Err(invalid());
    }
    Ok(body)
}

#[cfg(test)]
#[path = "launcher_tests.rs"]
mod lifecycle_tests;

#[cfg(test)]
mod tests {
    use super::{LaunchConfig, LaunchError, LaunchProfile, validate_health};
    use std::ffi::OsString;

    #[test]
    fn startup_failure_evidence_contains_only_fixed_stage_error_and_exit_fields()
    -> std::io::Result<()> {
        for (stage, status, expected) in [
            (
                super::StartupStage::AuthenticatedReadiness,
                Ok(Some(0xC000_0142)),
                "Native backend startup failure: stage=authenticated-readiness, error=Backend, process=exited-0xC0000142\n",
            ),
            (
                super::StartupStage::NativeIdentity,
                Ok(None),
                "Native backend startup failure: stage=native-identity, error=Backend, process=running\n",
            ),
            (
                super::StartupStage::AuthenticatedReadiness,
                Err(std::io::Error::other(
                    "private credential and path must never be formatted",
                )),
                "Native backend startup failure: stage=authenticated-readiness, error=Backend, process=status-unavailable\n",
            ),
        ] {
            let mut evidence = Vec::new();
            super::write_startup_failure(&mut evidence, stage, LaunchError::Backend, &status)?;
            assert_eq!(evidence, expected.as_bytes());
        }
        Ok(())
    }

    #[test]
    fn sandbox_failure_evidence_preserves_validation_and_windows_causes() -> std::io::Result<()> {
        for error in [
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "native sandbox requires 1..128 explicit read-only grants",
            ),
            std::io::Error::from_raw_os_error(5),
        ] {
            let mut evidence = Vec::new();
            super::write_sandbox_failure(&mut evidence, &error)?;
            let evidence = String::from_utf8(evidence).map_err(std::io::Error::other)?;
            assert!(evidence.contains(&error.to_string()));
            assert!(evidence.contains(&format!("kind={:?}", error.kind())));
            assert!(evidence.contains(&format!("os={:?}", error.raw_os_error())));
        }
        Ok(())
    }
    #[test]
    fn sandbox_preflight_distinguishes_parent_credentials_from_native_support() {
        assert_eq!(
            LaunchConfig::sandbox_support_error(std::io::ErrorKind::InvalidInput),
            LaunchError::SandboxEnvironment
        );
        for kind in [std::io::ErrorKind::Unsupported, std::io::ErrorKind::Other] {
            assert_eq!(
                LaunchConfig::sandbox_support_error(kind),
                LaunchError::SandboxUnavailable
            );
        }
    }

    #[test]
    fn sandbox_start_classification_separates_unavailable_credentials_and_policy() {
        assert_eq!(
            LaunchConfig::classify_sandbox_start_error(&std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "native PSEC containment: API set is not implemented",
            )),
            LaunchError::SandboxUnavailable
        );
        assert_eq!(
            LaunchConfig::classify_sandbox_start_error(&std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "sandbox launch rejects inherited credential environment variables",
            )),
            LaunchError::SandboxEnvironment
        );
        assert_eq!(
            LaunchConfig::classify_sandbox_start_error(&std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "native sandbox requires 1..128 explicit read-only grants",
            )),
            LaunchError::Sandbox
        );
        assert_eq!(
            LaunchConfig::classify_sandbox_start_error(&std::io::Error::from_raw_os_error(5)),
            LaunchError::Sandbox
        );
    }

    fn arguments() -> Vec<OsString> {
        [
            "--launch-development",
            "--allow-unisolated-backend",
            "--data-directory",
            "C:\\ExternalData",
            "--evidence-parent",
            "C:\\ExternalEvidence",
            "--frontend-port",
            "0",
            "--backend-port",
            "8081",
        ]
        .into_iter()
        .map(OsString::from)
        .collect()
    }

    #[test]
    fn native_sandbox_requires_an_exclusive_explicit_launch_selection() {
        let mut sandboxed = arguments();
        sandboxed[1] = "--require-native-sandbox".into();
        assert!(
            LaunchConfig::parse(sandboxed.clone())
                .is_ok_and(|config| config.requires_native_sandbox())
        );
        assert!(
            LaunchConfig::parse(arguments()).is_ok_and(|config| !config.requires_native_sandbox())
        );
        for extra in ["--allow-unisolated-backend", "--require-native-sandbox"] {
            let mut conflicting = sandboxed.clone();
            conflicting.push(extra.into());
            assert_eq!(
                LaunchConfig::parse(conflicting).err(),
                Some(LaunchError::Arguments)
            );
        }
        sandboxed.remove(0);
        assert_eq!(
            LaunchConfig::parse(sandboxed).err(),
            Some(LaunchError::Arguments)
        );
    }

    #[test]
    fn repacked_ptq1_is_explicit_swift_prism_only_and_keeps_lora_available() {
        let default = LaunchConfig::parse(arguments()).ok();
        assert!(default.is_some_and(|config| !config.uses_repacked_ptq1()));
        let mut enabled = arguments();
        enabled.extend(
            ["--profile", "swift-pq2-assessment", "--use-repacked-ptq1"].map(OsString::from),
        );
        let config = LaunchConfig::parse(enabled.clone()).ok();
        assert!(config.is_some_and(|config| config.uses_repacked_ptq1()));
        let mut with_lora = enabled.clone();
        with_lora.push("--use-abliterate-lora".into());
        assert!(LaunchConfig::parse(with_lora).is_ok());
        let mut rust = enabled.clone();
        rust.extend(["--native-engine", "--use-abliterate-lora"].map(OsString::from));
        assert!(LaunchConfig::parse(rust).is_ok());
        let mut rejected = enabled.clone();
        rejected.push("--use-repacked-ptq1".into());
        assert_eq!(
            LaunchConfig::parse(rejected).err(),
            Some(LaunchError::Arguments)
        );
        let mut reference = arguments();
        reference.push("--use-repacked-ptq1".into());
        assert_eq!(
            LaunchConfig::parse(reference).err(),
            Some(LaunchError::Arguments)
        );
    }

    #[test]
    fn original_pq2_is_explicit_swift_only_and_rejects_every_duplicate_or_conflict() {
        let mut swift = arguments();
        swift.extend(
            ["--profile", "swift-pq2-assessment", "--use-abliterate-lora"].map(OsString::from),
        );
        for native in [false, true] {
            let mut selected = swift.clone();
            selected.push(
                if native {
                    "--native-engine"
                } else {
                    "--publisher-engine"
                }
                .into(),
            );
            selected.push("--use-original-pq2".into());
            assert!(
                LaunchConfig::parse(selected).is_ok_and(|config| !config.uses_repacked_ptq1()
                    && config.use_abliterate_lora
                    && config.native_engine == native)
            );
        }
        for first in ["--use-original-pq2", "--use-repacked-ptq1"] {
            for second in ["--use-original-pq2", "--use-repacked-ptq1"] {
                let mut rejected = swift.clone();
                rejected.extend([first, second].map(OsString::from));
                assert_eq!(
                    LaunchConfig::parse(rejected).err(),
                    Some(LaunchError::Arguments)
                );
            }
        }
        let mut reference = arguments();
        reference.push("--use-original-pq2".into());
        assert_eq!(
            LaunchConfig::parse(reference).err(),
            Some(LaunchError::Arguments)
        );
    }

    #[test]
    fn lora_launch_is_explicit_swift_prism_only() {
        let default = LaunchConfig::parse(arguments()).ok();
        assert!(default.is_some_and(|config| !config.use_abliterate_lora));
        let mut enabled = arguments();
        enabled.extend(
            ["--profile", "swift-pq2-assessment", "--use-abliterate-lora"].map(OsString::from),
        );
        let config = LaunchConfig::parse(enabled.clone()).ok();
        assert!(config.is_some_and(|config| config.use_abliterate_lora));
        let mut rust = enabled.clone();
        rust.push("--native-engine".into());
        assert!(LaunchConfig::parse(rust).is_ok());
        let mut rejected = enabled.clone();
        rejected.push("--use-abliterate-lora".into());
        assert_eq!(
            LaunchConfig::parse(rejected).err(),
            Some(LaunchError::Arguments)
        );
        let mut reference = arguments();
        reference.push("--use-abliterate-lora".into());
        assert_eq!(
            LaunchConfig::parse(reference).err(),
            Some(LaunchError::Arguments)
        );
    }

    #[test]
    fn reset_on_cancel_is_explicit_native_swift_only() -> Result<(), LaunchError> {
        let default = LaunchConfig::parse(arguments())?;
        assert!(!default.reset_on_cancel);

        let mut native = arguments();
        native.extend(["--profile", "swift-pq2-assessment", "--native-engine"].map(OsString::from));
        assert!(!LaunchConfig::parse(native.clone())?.reset_on_cancel);

        let mut enabled = native;
        enabled.push("--reset-on-cancel".into());
        assert!(LaunchConfig::parse(enabled.clone())?.reset_on_cancel);

        let mut duplicate = enabled;
        duplicate.push("--reset-on-cancel".into());
        assert_eq!(
            LaunchConfig::parse(duplicate).err(),
            Some(LaunchError::Arguments)
        );

        let mut reference = arguments();
        reference.push("--reset-on-cancel".into());
        assert_eq!(
            LaunchConfig::parse(reference).err(),
            Some(LaunchError::Arguments)
        );

        let mut publisher = arguments();
        publisher.extend(
            [
                "--profile",
                "swift-pq2-assessment",
                "--publisher-engine",
                "--reset-on-cancel",
            ]
            .map(OsString::from),
        );
        assert_eq!(
            LaunchConfig::parse(publisher).err(),
            Some(LaunchError::Arguments)
        );
        Ok(())
    }

    #[test]
    fn lora_guard_rejects_missing_and_altered_adapter_and_opt_out_needs_no_file()
    -> std::io::Result<()> {
        use std::{fs, io};
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(io::Error::other)?
            .as_nanos();
        let directory = std::env::temp_dir().join(format!("Bonsai-lora-guard-{stamp}"));
        fs::create_dir(&directory)?;
        let key = directory.join("not-created.key");
        let base = super::LaunchPlan::for_profile(
            &directory,
            49152,
            &key,
            LaunchProfile::SwiftPq2Assessment,
        )
        .map_err(io::Error::other)?;
        assert!(
            LaunchConfig::adapter_guard(&base)
                .map_err(|_| io::Error::other("Unexpected base guard"))?
                .is_none()
        );
        let plan = super::LaunchPlan::for_profile_with_lora(
            &directory,
            49152,
            &key,
            LaunchProfile::SwiftPq2Assessment,
            true,
        )
        .map_err(io::Error::other)?;
        assert_eq!(
            LaunchConfig::adapter_guard(&plan).err(),
            Some(LaunchError::Adapter)
        );
        let path = &plan
            .adapter
            .as_ref()
            .ok_or_else(|| io::Error::other("Missing adapter plan"))?
            .path;
        let parent = path
            .parent()
            .ok_or_else(|| io::Error::other("Missing parent"))?;
        fs::create_dir_all(parent)?;
        fs::write(path, b"altered adapter")?;
        assert_eq!(
            LaunchConfig::adapter_guard(&plan).err(),
            Some(LaunchError::Adapter)
        );
        assert_eq!(directory.parent(), Some(std::env::temp_dir().as_path()));
        fs::remove_dir_all(directory)?;
        Ok(())
    }

    #[test]
    fn launch_requires_explicit_opt_ins_and_rejects_overrides() {
        assert!(LaunchConfig::parse(arguments()).is_ok());
        let mut missing = arguments();
        missing.remove(1);
        assert_eq!(
            LaunchConfig::parse(missing).err(),
            Some(LaunchError::Arguments)
        );
        for extra in [
            vec!["--model", "C:\\OtherModel"],
            vec!["--frontend-port", "5173"],
            vec!["--max-waiting", "9"],
            vec!["--allow-unverified-key-files"],
        ] {
            let mut args = arguments();
            args.extend(extra.into_iter().map(OsString::from));
            assert_eq!(
                LaunchConfig::parse(args).err(),
                Some(LaunchError::Arguments)
            );
        }
        let mut collision = arguments();
        collision[7] = OsString::from("8081");
        assert_eq!(
            LaunchConfig::parse(collision).err(),
            Some(LaunchError::Arguments)
        );
    }

    #[test]
    fn native_row_workers_are_explicit_bounded_and_reach_only_the_native_child()
    -> Result<(), LaunchError> {
        let mut native = arguments();
        native.extend(["--profile", "swift-pq2-assessment", "--native-engine"].map(OsString::from));
        assert_eq!(LaunchConfig::parse(native.clone())?.native_row_workers, 4);
        for invalid in ["0", "2", "3", "5", "16", "01", "+4", "4.0", " 4"] {
            let mut rejected = native.clone();
            rejected.extend(["--native-row-workers", invalid].map(OsString::from));
            assert_eq!(
                LaunchConfig::parse(rejected).err(),
                Some(LaunchError::Arguments)
            );
        }
        let mut missing = native.clone();
        missing.push("--native-row-workers".into());
        assert_eq!(
            LaunchConfig::parse(missing).err(),
            Some(LaunchError::Arguments)
        );
        let mut repeated = native.clone();
        repeated
            .extend(["--native-row-workers", "1", "--native-row-workers", "4"].map(OsString::from));
        assert_eq!(
            LaunchConfig::parse(repeated).err(),
            Some(LaunchError::Arguments)
        );
        for profile in [None, Some("swift-pq2-assessment")] {
            for workers in ["4", "8", "12"] {
                let mut publisher = arguments();
                if let Some(profile) = profile {
                    publisher.extend(["--profile", profile].map(OsString::from));
                }
                publisher.extend(
                    ["--publisher-engine", "--native-row-workers", workers].map(OsString::from),
                );
                assert_eq!(
                    LaunchConfig::parse(publisher).err(),
                    Some(LaunchError::Arguments)
                );
            }
        }
        let data = std::env::temp_dir().join("Bonsai-row-routing");
        let key = data.join("backend.key");
        for workers in ["1", "4", "8", "12"] {
            for repacked in [false, true] {
                let mut args = native.clone();
                args.extend(
                    ["--native-row-workers", workers, "--use-abliterate-lora"].map(OsString::from),
                );
                if repacked {
                    args.push("--use-repacked-ptq1".into());
                }
                let config = LaunchConfig::parse(args)?;
                let plan = super::LaunchPlan::for_profile_with_options(
                    &data,
                    8081,
                    &key,
                    LaunchProfile::SwiftPq2Assessment,
                    true,
                    repacked,
                )
                .map_err(|_| LaunchError::Policy)?;
                let child = config.native_arguments(&plan, Some(&key));
                let expected_flag = workers != "1";
                assert_eq!(
                    child.iter().filter(|arg| *arg == "--row-workers").count(),
                    usize::from(expected_flag)
                );
                if expected_flag {
                    assert!(
                        child
                            .array_windows()
                            .any(|[flag, value]| flag == "--row-workers" && value == workers)
                    );
                }
                let adapter = plan.adapter.as_ref().ok_or(LaunchError::Adapter)?;
                assert!(
                    child
                        .array_windows()
                        .any(|[flag, value]| flag == "--abliterate-lora"
                            && value == adapter.path.as_os_str())
                );
                assert_eq!(
                    child.iter().any(|arg| arg == "--use-repacked-ptq1"),
                    repacked
                );
            }
        }
        Ok(())
    }

    #[test]
    fn snapdragon_x1_host_defaults_native_row_workers_to_eight() -> Result<(), LaunchError> {
        use bonsai_cpu_policy::hardware::SnapdragonFamily;

        let mut swift = arguments();
        swift.extend(["--profile", "swift-pq2-assessment"].map(OsString::from));
        assert_eq!(
            LaunchConfig::parse_for_family(swift.clone(), Some(SnapdragonFamily::XElite))?
                .native_row_workers,
            8
        );
        assert_eq!(
            LaunchConfig::parse_for_family(swift.clone(), Some(SnapdragonFamily::X2EliteExtreme))?
                .native_row_workers,
            8
        );
        assert_eq!(
            LaunchConfig::parse_for_family(swift.clone(), None)?.native_row_workers,
            4
        );
        let mut explicit = swift.clone();
        explicit.extend(["--native-row-workers", "12"].map(OsString::from));
        assert_eq!(
            LaunchConfig::parse_for_family(explicit, Some(SnapdragonFamily::XElite))?
                .native_row_workers,
            12
        );
        let mut publisher = swift;
        publisher.push("--publisher-engine".into());
        assert_eq!(
            LaunchConfig::parse_for_family(publisher, Some(SnapdragonFamily::XElite))?
                .native_row_workers,
            1
        );
        Ok(())
    }

    #[test]
    fn prompt_reuse_is_explicit_native_only_and_reaches_the_owned_child() -> Result<(), LaunchError>
    {
        use crate::inference::PromptReuseMode;
        let mut native = arguments();
        native.extend(["--profile", "swift-pq2-assessment"].map(OsString::from));
        assert_eq!(
            LaunchConfig::parse(native.clone())?.prompt_reuse,
            PromptReuseMode::Disabled
        );
        let data = std::env::temp_dir().join("Bonsai-prefix-routing");
        let key = data.join("backend.key");
        let plan = super::LaunchPlan::for_profile_with_options(
            &data,
            8081,
            &key,
            LaunchProfile::SwiftPq2Assessment,
            true,
            false,
        )
        .map_err(|_| LaunchError::Policy)?;
        for (name, mode) in [
            ("disabled", PromptReuseMode::Disabled),
            ("latest", PromptReuseMode::Latest),
            ("prompt-end-and-latest", PromptReuseMode::PromptEndAndLatest),
        ] {
            let mut selected = native.clone();
            selected.extend(["--prompt-reuse", name].map(OsString::from));
            let config = LaunchConfig::parse(selected.clone())?;
            assert_eq!(config.prompt_reuse, mode);
            let child = config.native_arguments(&plan, Some(&key));
            assert_eq!(
                child
                    .array_windows()
                    .filter(|[flag, value]| flag == "--prompt-reuse"
                        && value == mode.as_wire().unwrap_or(""))
                    .count(),
                usize::from(mode.enabled())
            );
            selected.extend(["--prompt-reuse", name].map(OsString::from));
            assert_eq!(
                LaunchConfig::parse(selected).err(),
                Some(LaunchError::Arguments)
            );
        }
        for value in ["Latest", "true", "", "snapshot", "approximate"] {
            let mut invalid = native.clone();
            invalid.extend(["--prompt-reuse", value].map(OsString::from));
            assert_eq!(
                LaunchConfig::parse(invalid).err(),
                Some(LaunchError::Arguments)
            );
        }
        native.extend(["--publisher-engine", "--prompt-reuse", "latest"].map(OsString::from));
        assert_eq!(
            LaunchConfig::parse(native).err(),
            Some(LaunchError::Arguments)
        );
        Ok(())
    }

    #[test]
    fn kv_cache_selection_rejects_unsupported_pins_and_reaches_only_the_native_child()
    -> Result<(), LaunchError> {
        use crate::inference::KvCache;
        let mut native = arguments();
        native.extend(["--profile", "swift-pq2-assessment"].map(OsString::from));
        assert_eq!(
            LaunchConfig::parse(native.clone())?.kv_cache,
            Some(KvCache::Rsq4)
        );
        let mut selected = native.clone();
        selected.extend(["--kv-cache", "f16"].map(OsString::from));
        assert_eq!(LaunchConfig::parse(selected)?.kv_cache, Some(KvCache::F16));
        selected = native.clone();
        selected.extend(["--kv-cache", "rsq4"].map(OsString::from));
        assert_eq!(LaunchConfig::parse(selected)?.kv_cache, Some(KvCache::Rsq4));
        for suffix in [
            vec!["--kv-cache", "F16"],
            vec!["--kv-cache"],
            vec!["--kv-cache", "f16", "--kv-cache", "f16"],
            vec!["--publisher-engine", "--kv-cache", "f16"],
        ] {
            let mut rejected = native.clone();
            rejected.extend(suffix.into_iter().map(OsString::from));
            assert_eq!(
                LaunchConfig::parse(rejected).err(),
                Some(LaunchError::Arguments)
            );
        }
        let mut unsupported = native.clone();
        unsupported.extend(["--kv-cache", "rsq4k3v"].map(OsString::from));
        assert_eq!(
            LaunchConfig::parse(unsupported).err(),
            Some(LaunchError::NativeRuntime)
        );
        let data = std::env::temp_dir().join("Bonsai-kv-routing");
        let key = data.join("backend.key");
        let plan = super::LaunchPlan::for_profile_with_options(
            &data,
            8081,
            &key,
            LaunchProfile::SwiftPq2Assessment,
            true,
            false,
        )
        .map_err(|_| LaunchError::Policy)?;
        let mut config = LaunchConfig::parse(native)?;
        for policy in [KvCache::Rsq4, KvCache::F16, KvCache::Rsq4K3v] {
            config.kv_cache = Some(policy);
            let child = config.native_arguments(&plan, Some(&key));
            assert_eq!(
                child
                    .array_windows::<2>()
                    .filter(|[flag, value]| flag == "--kv-cache" && value == policy.name())
                    .count(),
                1
            );
        }
        Ok(())
    }

    #[test]
    fn placement_selection_is_native_only_gated_and_reaches_the_owned_child()
    -> Result<(), LaunchError> {
        use crate::inference::Placement;
        let mut native = arguments();
        native.extend(["--profile", "swift-pq2-assessment"].map(OsString::from));
        let mut config = LaunchConfig::parse(native.clone())?;
        assert_eq!(config.placement, Placement::Auto);
        let mut selected = native.clone();
        selected.extend(["--placement", "auto"].map(OsString::from));
        assert_eq!(LaunchConfig::parse(selected)?.placement, Placement::Auto);
        selected = native.clone();
        selected.extend(["--placement", "cpu"].map(OsString::from));
        assert_eq!(LaunchConfig::parse(selected)?.placement, Placement::Cpu);
        for suffix in [
            vec!["--placement"],
            vec!["--placement", "CPU"],
            vec!["--placement", ""],
            vec!["--placement", "npu"],
            vec!["--placement", "auto", "--placement", "cpu"],
            vec!["--publisher-engine", "--placement", "auto"],
            vec!["--publisher-engine", "--placement", "cpu"],
        ] {
            let mut rejected = native.clone();
            rejected.extend(suffix.into_iter().map(OsString::from));
            assert_eq!(
                LaunchConfig::parse(rejected).err(),
                Some(LaunchError::Arguments)
            );
        }
        for placement in ["auto", "cpu"] {
            let mut rejected = arguments();
            rejected.extend(["--placement", placement].map(OsString::from));
            assert_eq!(
                LaunchConfig::parse(rejected).err(),
                Some(LaunchError::Arguments)
            );
        }
        let data = std::env::temp_dir().join("Bonsai-placement-routing");
        let key = data.join("backend.key");
        let plan = super::LaunchPlan::for_profile_with_options(
            &data,
            8081,
            &key,
            LaunchProfile::SwiftPq2Assessment,
            true,
            false,
        )
        .map_err(|_| LaunchError::Policy)?;
        let admitted = config.swift_context_plan(20 * 1024 * 1024 * 1024)?;
        for placement in [Placement::Auto, Placement::Cpu] {
            // Model-free routing control for the exact reviewed source-bound release.
            config.placement = placement;
            let child = config.native_arguments(&plan, Some(&key));
            assert_eq!(
                child
                    .array_windows::<2>()
                    .filter(|[flag, value]| flag == "--placement" && value == "cpu")
                    .count(),
                usize::from(placement == Placement::Cpu)
            );
            assert_eq!(
                config.swift_context_plan(20 * 1024 * 1024 * 1024)?,
                admitted
            );
        }
        let pipe_child = config.native_arguments(&plan, None);
        assert!(pipe_child.iter().any(|arg| arg == "--broker-pipe"));
        assert!(
            !pipe_child
                .iter()
                .any(|arg| arg == "--port" || arg == "--key-file")
        );
        Ok(())
    }

    #[test]
    fn native_context_sizing_preserves_release_gate_and_publisher_budget() -> Result<(), LaunchError>
    {
        use crate::inference::KvCache;
        let mut args = arguments();
        args.extend(["--profile", "swift-pq2-assessment"].map(OsString::from));
        let mut config = LaunchConfig::parse(args.clone())?;
        // The exact reviewed release implements paired launch flags and the
        // selected-cache accounting contract. Other identities remain rejected.
        assert!(config.memory_context_supported()?);
        let gib = 1024 * 1024 * 1024_u64;
        let available = 20 * gib;
        // Under promoted 0.1.32 calibrated-CoW admission, full 262,144 K4/V4 context fits at 20 GiB,
        // while constrained 18 GiB scales back cleanly below 262,144.
        let packed_18g = config.swift_context_plan(18 * gib)?;
        assert!(packed_18g.context_tokens < 262_144);
        let packed = config.swift_context_plan(available)?;
        assert_eq!(packed.context_tokens, 262_144);
        assert!(crate::inference::full_context_supported()?);
        assert_eq!(config.swift_context_plan(24 * gib)?.context_tokens, 262_144);
        config.kv_cache = Some(KvCache::Rsq4K3v);
        let capacity = config.swift_context_plan(available)?;
        assert_eq!(capacity.context_tokens, 262_144);
        assert!(config.memory_context_supported()?);
        config.kv_cache = Some(KvCache::F16);
        let f16 = config.swift_context_plan(available)?;
        assert!(f16.context_tokens < packed.context_tokens);
        assert!(config.swift_context_plan(0).is_err());
        config.kv_cache = None;
        assert_eq!(config.swift_context_plan(available)?, f16);

        args.push("--publisher-engine".into());
        let publisher = LaunchConfig::parse(args)?;
        assert!(publisher.memory_context_supported()?);
        let plan = publisher.swift_context_plan(available)?;
        assert_eq!(plan.context_tokens, 37_120);
        assert_eq!(
            plan,
            bonsai_cpu_policy::context::SwiftContextPlan::for_available_memory(available)
                .map_err(|_| LaunchError::Memory)?
        );
        Ok(())
    }

    #[test]
    fn rust_is_swift_development_default_and_publisher_is_explicit_rollback() {
        let mut args = arguments();
        args.push(OsString::from("--native-engine"));
        assert_eq!(
            LaunchConfig::parse(args).err(),
            Some(LaunchError::Arguments)
        );
        let mut swift = arguments();
        swift.extend(["--profile", "swift-pq2-assessment"].map(OsString::from));
        assert!(LaunchConfig::parse(swift.clone()).is_ok_and(|config| {
            config.uses_native_engine()
                && config.native_row_workers == 4
                && !config.uses_repacked_ptq1()
        }));
        assert!(LaunchConfig::parse(arguments()).is_ok_and(|config| {
            !config.uses_native_engine() && config.native_row_workers == 1
        }));
        let mut publisher = swift.clone();
        publisher.push("--publisher-engine".into());
        assert!(LaunchConfig::parse(publisher).is_ok_and(|config| {
            !config.uses_native_engine() && config.native_row_workers == 1
        }));
        for first in ["--native-engine", "--publisher-engine"] {
            for second in ["--native-engine", "--publisher-engine"] {
                let mut rejected = swift.clone();
                rejected.extend([first, second].map(OsString::from));
                assert_eq!(
                    LaunchConfig::parse(rejected).err(),
                    Some(LaunchError::Arguments)
                );
            }
        }
        let mut args = swift;
        args.push("--native-engine".into());
        let config = LaunchConfig::parse(args).ok();
        assert!(config.is_some_and(|value| value.uses_native_engine()));
        let mut args = arguments();
        args.extend(
            [
                "--profile",
                "swift-pq2-assessment",
                "--native-engine",
                "--native-engine",
            ]
            .map(OsString::from),
        );
        assert_eq!(
            LaunchConfig::parse(args).err(),
            Some(LaunchError::Arguments)
        );
    }

    #[test]
    fn native_readiness_requires_exact_model_template_runtime_context_and_seed()
    -> std::io::Result<()> {
        use serde_json::json;
        let workers = super::WorkerReadiness::for_version("0.1.10", 1);
        let model = json!({"id":"Swift-Bonsai-2-PQ2_0","object":"model",
            "model_sha256":crate::inference::MODEL_SHA256,
            "template_sha256":crate::inference::TEMPLATE_SHA256,
            "runtime":"prism-inference","context_tokens":8192,"seed":42});
        let body = serde_json::to_string(&json!({"data":[model]}))?;
        assert!(super::validate_native_identity(&response(&body), false, false, workers).is_ok());
        for (field, changed) in [
            ("id", json!("foreign")),
            ("object", json!("list")),
            ("model_sha256", json!("0".repeat(64))),
            ("template_sha256", json!("0".repeat(64))),
            ("runtime", json!("Prism")),
            ("context_tokens", json!(8193)),
            ("seed", json!(43)),
        ] {
            let mut changed_model = model.clone();
            changed_model[field] = changed;
            let body = serde_json::to_string(&json!({"data":[changed_model]}))?;
            assert!(
                super::validate_native_identity(&response(&body), false, false, workers).is_err(),
                "{field}"
            );
        }
        for data in [
            json!([]),
            json!([model.clone(), model]),
            json!([["positional"]]),
        ] {
            let body = serde_json::to_string(&json!({"data":data}))?;
            assert!(
                super::validate_native_identity(&response(&body), false, false, workers).is_err()
            );
        }
        for field in [
            "weight_representation",
            "adapter_sha256",
            "supports_abliteration_scale",
            "abliteration_scale_range",
        ] {
            let mut changed_model = model.clone();
            changed_model[field] = json!(null);
            let body = serde_json::to_string(&json!({"data":[changed_model]}))?;
            assert!(
                super::validate_native_identity(&response(&body), false, false, workers).is_err(),
                "{field}"
            );
        }
        let duplicate = body.replace("\"seed\":42", "\"seed\":42,\"seed\":42");
        assert!(
            super::validate_native_identity(&response(&duplicate), false, false, workers).is_err()
        );
        Ok(())
    }

    #[test]
    fn native_readiness_binds_explicit_packing_and_adapter_scale_capability() -> std::io::Result<()>
    {
        use super::validate_native_identity;
        use serde_json::json;
        use std::io;
        let workers = super::WorkerReadiness::for_version("0.1.10", 1);
        for (repacked, adapter) in [(true, false), (false, true), (true, true)] {
            let mut model = json!({
                "id": if repacked { "Swift-Bonsai-2-current-repacked-PTQ1_0" } else { "Swift-Bonsai-2-PQ2_0" },
                "object":"model",
                "model_sha256": if repacked { crate::inference::REPACKED_PTQ1_SHA256 } else { crate::inference::MODEL_SHA256 },
                "template_sha256":crate::inference::TEMPLATE_SHA256,
                "runtime":"prism-inference", "context_tokens":8192, "seed":42,
                "weight_representation": if repacked { "current-repacked-PTQ1_0" } else { "PQ2_0" },
                "adapter_sha256": adapter.then_some(crate::inference::ADAPTER_SHA256),
                "supports_abliteration_scale":adapter,
            });
            if adapter {
                model["abliteration_scale_range"] = json!([0, 2]);
            }
            let body = serde_json::to_string(&json!({"data":[model]}))?;
            assert!(validate_native_identity(&response(&body), repacked, adapter, workers).is_ok());
            assert!(
                validate_native_identity(&response(&body), !repacked, adapter, workers).is_err()
            );
            assert!(
                validate_native_identity(&response(&body), repacked, !adapter, workers).is_err()
            );
            for (field, value) in [
                ("weight_representation", json!("other")),
                ("adapter_sha256", json!("0".repeat(64))),
                ("supports_abliteration_scale", json!(!adapter)),
                ("abliteration_scale_range", json!([0, 3])),
                ("abliteration_scale_range", json!(null)),
            ] {
                let mut changed = model.clone();
                changed[field] = value;
                let body = serde_json::to_string(&json!({"data":[changed]}))?;
                assert!(
                    validate_native_identity(&response(&body), repacked, adapter, workers).is_err(),
                    "{field}"
                );
            }
            for field in [
                "weight_representation",
                "adapter_sha256",
                "supports_abliteration_scale",
            ] {
                let mut changed = model.clone();
                changed
                    .as_object_mut()
                    .ok_or_else(|| io::Error::other("fixture"))?
                    .remove(field);
                let body = serde_json::to_string(&json!({"data":[changed]}))?;
                assert!(
                    validate_native_identity(&response(&body), repacked, adapter, workers).is_err(),
                    "{field}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn profiles_are_explicit_exact_unique_and_default_to_reference() {
        let default = LaunchConfig::parse(arguments());
        assert_eq!(
            default.map(|config| config.profile()),
            Ok(LaunchProfile::ReferenceCpu)
        );
        for profile in [
            LaunchProfile::ReferenceCpu,
            LaunchProfile::SwiftPq2Assessment,
        ] {
            let mut args = arguments();
            args.extend([OsString::from("--profile"), OsString::from(profile.name())]);
            assert_eq!(
                LaunchConfig::parse(args).map(|config| config.profile()),
                Ok(profile)
            );
        }
        for extra in [
            vec!["--profile"],
            vec!["--profile", ""],
            vec!["--profile", "Swift-pq2-assessment"],
            vec!["--profile", "swift-ptq1-assessment"],
            vec!["--profile", "C:\\ExternalProfile.json"],
            vec![
                "--profile",
                "swift-pq2-assessment",
                "--profile",
                "reference-cpu",
            ],
            vec![
                "--profile",
                "swift-pq2-assessment",
                "--lora",
                "C:\\Adapter.gguf",
            ],
            vec![
                "--profile",
                "swift-pq2-assessment",
                "--model",
                "C:\\Other.gguf",
            ],
            vec![
                "--profile",
                "swift-pq2-assessment",
                "--runtime",
                "C:\\Other.exe",
            ],
            vec!["--profile", "swift-pq2-assessment", "--batch-size", "256"],
        ] {
            let mut args = arguments();
            args.extend(extra.into_iter().map(OsString::from));
            assert_eq!(
                LaunchConfig::parse(args).err(),
                Some(LaunchError::Arguments)
            );
        }
    }

    fn response(body: &str) -> Vec<u8> {
        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        )
        .into_bytes()
    }

    #[test]
    fn readiness_requires_bounded_complete_unambiguous_json() {
        assert!(validate_health(&response(r#"{"status":"ok"}"#)).is_ok());
        assert!(validate_health(&response(r#"{"status":"loading model"}"#)).is_err());
        assert!(validate_health(&response(r#"{"status":"ok","extra":true}"#)).is_err());
        assert!(validate_health(&response(r#"["ok"]"#)).is_err());
        assert!(validate_health(&response(r#"{"status":"ok"}garbage"#)).is_err());
        let complete = response(r#"{"status":"ok"}"#);
        let complete = String::from_utf8_lossy(&complete);
        assert!(
            validate_health(
                complete
                    .replace("Content-Length: 15", "Content-Length: 00015")
                    .as_bytes()
            )
            .is_ok()
        );
        for length in ["+15", "15x", "99999999999999999999999999999999999999"] {
            assert!(
                validate_health(
                    complete
                        .replace("Content-Length: 15", &format!("Content-Length: {length}"))
                        .as_bytes()
                )
                .is_err()
            );
        }
        for message in [
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n",
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 15\r\nContent-Length: 15\r\n\r\n{\"status\":\"ok\"}",
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 16\r\n\r\n{\"status\":\"ok\"}",
            "HTTP/1.1 503 Busy\r\nContent-Type: application/json\r\nContent-Length: 15\r\n\r\n{\"status\":\"ok\"}",
        ] {
            assert!(validate_health(message.as_bytes()).is_err());
        }
        assert!(validate_health(&vec![b' '; 20 * 1024 + 1]).is_err());
    }

    #[test]
    fn no_mmproj_and_mmproj_guard_enforce_swift_only_fallback_and_integrity()
    -> Result<(), LaunchError> {
        let mut swift = arguments();
        swift.extend(["--profile", "swift-pq2-assessment"].map(OsString::from));
        let mut publisher_args = swift.clone();
        publisher_args.push("--publisher-engine".into());
        let publisher = LaunchConfig::parse(publisher_args.clone())?;
        assert!(!publisher.no_mmproj);
        assert!(publisher.runtime_supports_vision()?);

        let mut opt_out = publisher_args;
        opt_out.push("--no-mmproj".into());
        let parsed_opt_out = LaunchConfig::parse(opt_out.clone())?;
        assert!(parsed_opt_out.no_mmproj);
        assert!(!parsed_opt_out.runtime_supports_vision()?);
        assert!(parsed_opt_out.mmproj_guard()?.is_none());

        // Duplicate --no-mmproj and reference-cpu --no-mmproj are rejected.
        let mut dup = opt_out;
        dup.push("--no-mmproj".into());
        assert_eq!(LaunchConfig::parse(dup).err(), Some(LaunchError::Arguments));
        let mut reference = arguments();
        reference.push("--no-mmproj".into());
        assert_eq!(
            LaunchConfig::parse(reference).err(),
            Some(LaunchError::Arguments)
        );

        // Native 0.1.31 supports vision unless --no-mmproj opts out.
        let mut native_opt_out = swift.clone();
        native_opt_out.push("--no-mmproj".into());
        let native = LaunchConfig::parse(swift)?;
        assert!(native.runtime_supports_vision()?);
        let parsed_native_opt_out = LaunchConfig::parse(native_opt_out)?;
        assert!(!parsed_native_opt_out.runtime_supports_vision()?);
        assert!(parsed_native_opt_out.mmproj_guard()?.is_none());

        // Real temp data directory: missing mmproj falls back cleanly to None;
        // wrong bytes at the locked mmproj path fails closed with Artifacts.
        let temp =
            std::env::temp_dir().join(format!("Bonsai-mmproj-guard-test-{}", std::process::id()));
        let _cleanup = std::fs::remove_dir_all(&temp);
        std::fs::create_dir_all(&temp).map_err(|_| LaunchError::Path)?;
        let evidence = temp.join("evidence");
        std::fs::create_dir_all(&evidence).map_err(|_| LaunchError::Path)?;
        let mut temp_cfg = LaunchConfig::parse([
            OsString::from("--launch-development"),
            "--allow-unisolated-backend".into(),
            "--profile".into(),
            "swift-pq2-assessment".into(),
            "--publisher-engine".into(),
            "--data-directory".into(),
            temp.as_os_str().to_owned(),
            "--evidence-parent".into(),
            evidence.as_os_str().to_owned(),
            "--frontend-port".into(),
            "0".into(),
            "--backend-port".into(),
            "8081".into(),
        ])?;
        assert!(temp_cfg.mmproj_guard()?.is_none());

        let artifact =
            bonsai_cpu_policy::swift_mmproj_artifact(&temp).map_err(|_| LaunchError::Policy)?;
        if let Some(parent) = artifact.path.parent() {
            std::fs::create_dir_all(parent).map_err(|_| LaunchError::Path)?;
        }
        std::fs::write(&artifact.path, b"not the genuine mmproj").map_err(|_| LaunchError::Path)?;
        assert_eq!(temp_cfg.mmproj_guard().err(), Some(LaunchError::Artifacts));
        temp_cfg.no_mmproj = true;
        assert!(temp_cfg.mmproj_guard()?.is_none());
        let _cleanup = std::fs::remove_dir_all(&temp);
        Ok(())
    }

    #[test]
    fn require_broker_pipe_enforces_sandbox_native_engine_and_reviewed_release()
    -> Result<(), LaunchError> {
        let mut valid = [
            "--launch-development",
            "--require-native-sandbox",
            "--require-broker-pipe",
            "--profile",
            "swift-pq2-assessment",
            "--data-directory",
            "C:\\ExternalData",
            "--evidence-parent",
            "C:\\ExternalEvidence",
            "--frontend-port",
            "0",
        ]
        .into_iter()
        .map(OsString::from)
        .collect::<Vec<_>>();
        // With BROKER_PIPE_RELEASE pinned at 0.1.32, valid flag combinations succeed and enable broker pipe.
        let parsed = LaunchConfig::parse(valid.clone())?;
        assert!(parsed.uses_broker_pipe()?);

        // Duplicate flag is rejected as Arguments.
        let mut dup = valid.clone();
        dup.push("--require-broker-pipe".into());
        assert_eq!(LaunchConfig::parse(dup).err(), Some(LaunchError::Arguments));

        // Optional explicit --backend-port also succeeds with --require-broker-pipe when pinned.
        valid.extend(["--backend-port", "8081"].map(OsString::from));
        let with_port = LaunchConfig::parse(valid)?;
        assert!(with_port.uses_broker_pipe()?);

        // Invalid combinations reject with Arguments before release gating.
        for bad in [
            vec![
                "--launch-development",
                "--allow-unisolated-backend",
                "--require-broker-pipe",
                "--profile",
                "swift-pq2-assessment",
                "--data-directory",
                "C:\\ExternalData",
                "--evidence-parent",
                "C:\\ExternalEvidence",
                "--frontend-port",
                "0",
            ],
            vec![
                "--launch-development",
                "--require-native-sandbox",
                "--require-broker-pipe",
                "--profile",
                "swift-pq2-assessment",
                "--publisher-engine",
                "--data-directory",
                "C:\\ExternalData",
                "--evidence-parent",
                "C:\\ExternalEvidence",
                "--frontend-port",
                "0",
            ],
            vec![
                "--launch-development",
                "--require-native-sandbox",
                "--require-broker-pipe",
                "--data-directory",
                "C:\\ExternalData",
                "--evidence-parent",
                "C:\\ExternalEvidence",
                "--frontend-port",
                "0",
            ],
        ] {
            let args = bad.into_iter().map(OsString::from).collect::<Vec<_>>();
            assert_eq!(
                LaunchConfig::parse(args).err(),
                Some(LaunchError::Arguments)
            );
        }

        // Omitting --backend-port without --require-broker-pipe still rejects.
        let missing_port = [
            "--launch-development",
            "--require-native-sandbox",
            "--profile",
            "swift-pq2-assessment",
            "--data-directory",
            "C:\\ExternalData",
            "--evidence-parent",
            "C:\\ExternalEvidence",
            "--frontend-port",
            "0",
        ]
        .into_iter()
        .map(OsString::from)
        .collect::<Vec<_>>();
        assert_eq!(
            LaunchConfig::parse(missing_port).err(),
            Some(LaunchError::Arguments)
        );
        Ok(())
    }

    #[test]
    fn native_arguments_with_vision_pass_base_memory_limit_without_double_counting_mmproj_reserve()
    -> Result<(), LaunchError> {
        use crate::inference::KvCache;
        use bonsai_cpu_policy::{
            MMPROJ_RESERVE_BYTES,
            context::{HOST_RESERVE_BYTES, STARTUP_RESERVE_BYTES, SwiftContextPlan},
        };

        let mut args = arguments();
        args.extend(["--profile", "swift-pq2-assessment"].map(OsString::from));
        let mut config = LaunchConfig::parse(args)?;
        let data = std::env::temp_dir().join("Bonsai-vision-memory-limit-routing");
        let key = data.join("backend.key");
        let mmproj =
            bonsai_cpu_policy::swift_mmproj_artifact(&data).map_err(|_| LaunchError::Policy)?;
        let gib = 1024 * 1024 * 1024_u64;

        for policy in [KvCache::Rsq4, KvCache::F16, KvCache::Rsq4K3v] {
            config.kv_cache = Some(policy);
            let profile = config.swift_context_profile()?;
            for available in [17 * gib, 18 * gib, 20 * gib, 22 * gib, 24 * gib, 48 * gib] {
                let (vision_context, vision_admitted) =
                    SwiftContextPlan::for_profile_with_vision(available, profile, true)
                        .map_err(|_| LaunchError::Memory)?;
                assert!(vision_admitted, "expected vision admission at {available}");
                let base_context =
                    SwiftContextPlan::for_profile(available - MMPROJ_RESERVE_BYTES, profile)
                        .map_err(|_| LaunchError::Memory)?;
                assert_eq!(vision_context.context_tokens, base_context.context_tokens);

                let mut plan = super::LaunchPlan::for_profile_with_options(
                    &data,
                    8081,
                    &key,
                    LaunchProfile::SwiftPq2Assessment,
                    true,
                    false,
                )
                .map_err(|_| LaunchError::Policy)?;
                plan.select_swift_context(vision_context)
                    .map_err(|_| LaunchError::Policy)?;
                plan.enable_swift_mmproj(mmproj.clone())
                    .map_err(|_| LaunchError::Policy)?;

                // Windows Job ceiling retains the full vision reserve, while the child's
                // --memory-limit-bytes receives the base text memory limit so adding
                // 4 GiB + 768 MiB inside the child matches required_available_bytes.
                assert_eq!(plan.memory_limit_bytes, vision_context.memory_limit_bytes);
                assert_eq!(
                    plan.native_child_memory_limit_bytes(),
                    base_context.memory_limit_bytes
                );
                let child = config.native_arguments(&plan, Some(&key));
                let child_limit = child
                    .array_windows::<2>()
                    .find_map(|[flag, value]| {
                        (flag == "--memory-limit-bytes")
                            .then(|| value.to_str()?.parse::<u64>().ok())
                            .flatten()
                    })
                    .ok_or(LaunchError::Arguments)?;
                assert_eq!(
                    child_limit,
                    u64::try_from(base_context.memory_limit_bytes)
                        .map_err(|_| LaunchError::Memory)?
                );
                let child_required_available =
                    child_limit + HOST_RESERVE_BYTES + MMPROJ_RESERVE_BYTES;
                assert_eq!(child_required_available, plan.required_available_bytes);
                assert!(child_required_available + STARTUP_RESERVE_BYTES <= available);
            }
        }
        Ok(())
    }

    #[test]
    fn child_startup_stderr_classification_maps_value_free_stages_and_resets_worker_files()
    -> std::io::Result<()> {
        use super::{
            ChildStartupStage, StartupStage, parse_child_startup_failure,
            refine_backend_startup_error, reset_worker_attempt_files, write_startup_failure,
        };

        for (line, expected_stage, expected_error, expected_marker) in [
            (
                b"private-inference-startup-failed stage=memory errorKind=Other osCode=none\n"
                    .as_slice(),
                ChildStartupStage::Memory,
                LaunchError::Memory,
                "error=Memory",
            ),
            (
                b"private-inference-startup-failed stage=environment errorKind=PermissionDenied osCode=none\n".as_slice(),
                ChildStartupStage::Environment,
                LaunchError::Policy,
                "error=Backend",
            ),
            (
                b"private-inference-startup-failed stage=key errorKind=PermissionDenied osCode=5\n".as_slice(),
                ChildStartupStage::Key,
                LaunchError::Credentials,
                "error=Backend",
            ),
            (
                b"private-inference-startup-failed stage=listener errorKind=AddrInUse osCode=10048\n".as_slice(),
                ChildStartupStage::Listener,
                LaunchError::Port,
                "error=Backend",
            ),
            (
                b"private-inference-startup-failed stage=arguments errorKind=InvalidInput osCode=none\n".as_slice(),
                ChildStartupStage::Arguments,
                LaunchError::Backend,
                "error=Backend",
            ),
            (
                b"private-inference-startup-failed stage=model errorKind=InvalidData osCode=-1\n".as_slice(),
                ChildStartupStage::Model,
                LaunchError::Backend,
                "error=Backend",
            ),
        ] {
            let parsed = parse_child_startup_failure(line);
            assert_eq!(parsed, Some(expected_stage));
            assert_eq!(expected_stage.launch_error(), expected_error);
            let mut marker = Vec::new();
            write_startup_failure(
                &mut marker,
                StartupStage::AuthenticatedReadiness,
                expected_error,
                &Ok(Some(1)),
            )?;
            let marker = String::from_utf8(marker).map_err(std::io::Error::other)?;
            assert!(marker.contains(expected_marker), "{marker}");
        }

        for rejected in [
            b"".as_slice(),
            b"arbitrary stderr line\n".as_slice(),
            b"private-inference-startup-failed stage=memory\n".as_slice(),
            b"private-inference-startup-failed stage=memory errorKind=Other osCode=none extra\n"
                .as_slice(),
            b"private-inference-startup-failed stage=forged errorKind=Other osCode=none\n"
                .as_slice(),
            b"private-inference-startup-failed stage=memory errorKind=Other! osCode=none\n"
                .as_slice(),
            b"private-inference-startup-failed stage=memory errorKind=Other osCode=abc\n"
                .as_slice(),
        ] {
            assert_eq!(parse_child_startup_failure(rejected), None);
        }

        let dir = std::env::temp_dir().join(format!(
            "Bonsai-child-stderr-classify-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(std::io::Error::other)?
                .as_nanos()
        ));
        std::fs::create_dir_all(dir.join("worker/nested"))?;
        std::fs::write(dir.join("frontend.key"), b"retained-frontend-key")?;
        std::fs::write(dir.join("backend.key"), b"retained-backend-key")?;
        std::fs::write(dir.join("worker/nested/state.bin"), b"scratch")?;
        std::fs::write(dir.join("backend.stdout"), b"stdout")?;
        let stderr_path = dir.join("backend.stderr");
        std::fs::write(
            &stderr_path,
            b"private-inference-startup-failed stage=memory errorKind=Other osCode=none\r\n",
        )?;
        assert_eq!(
            refine_backend_startup_error(LaunchError::Backend, &stderr_path),
            (LaunchError::Memory, Some(ChildStartupStage::Memory))
        );
        assert_eq!(
            refine_backend_startup_error(LaunchError::Cancelled, &stderr_path),
            (LaunchError::Cancelled, None)
        );
        reset_worker_attempt_files(&dir).map_err(|_| std::io::Error::other("reset failed"))?;
        assert!(!dir.join("worker").exists());
        assert!(!dir.join("backend.stdout").exists());
        assert!(!dir.join("backend.stderr").exists());
        assert!(dir.join("frontend.key").is_file());
        assert!(dir.join("backend.key").is_file());
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn native_identity_failure_evidence_emits_closed_value_free_reasons() -> std::io::Result<()> {
        use super::{NativeIdentityError, write_native_identity_failure};

        let all = [
            (NativeIdentityError::ReleasePin, "release-pin"),
            (NativeIdentityError::Framing, "framing"),
            (NativeIdentityError::Schema, "schema"),
            (NativeIdentityError::Model, "model"),
            (NativeIdentityError::ContextTokens, "context-tokens"),
            (NativeIdentityError::RowWorkers, "row-workers"),
            (NativeIdentityError::PromptReuse, "prompt-reuse"),
            (NativeIdentityError::KvCacheGraph, "kv-cache-graph"),
            (NativeIdentityError::Placement, "placement"),
            (NativeIdentityError::StartupProjection, "startup-projection"),
            (NativeIdentityError::Adapter, "adapter"),
            (NativeIdentityError::Vision, "vision"),
        ];
        for (variant, expected_reason) in all {
            assert_eq!(variant.reason(), expected_reason);
            let mut buffer = Vec::new();
            write_native_identity_failure(&mut buffer, variant)?;
            assert_eq!(
                String::from_utf8(buffer).map_err(std::io::Error::other)?,
                format!("Native identity verification failed: reason={expected_reason}.\n")
            );
        }
        Ok(())
    }

    #[test]
    fn initial_guard_error_preserves_memory_and_sandbox_classifications() {
        use super::classify_initial_guard_error;

        assert_eq!(
            classify_initial_guard_error(LaunchError::Memory, None),
            LaunchError::Memory
        );
        assert_eq!(
            classify_initial_guard_error(LaunchError::Memory, Some(LaunchError::Sandbox)),
            LaunchError::Memory
        );
        assert_eq!(
            classify_initial_guard_error(LaunchError::Sandbox, None),
            LaunchError::Sandbox
        );
        assert_eq!(
            classify_initial_guard_error(LaunchError::Backend, Some(LaunchError::Sandbox)),
            LaunchError::Sandbox
        );
        assert_eq!(
            classify_initial_guard_error(LaunchError::Backend, None),
            LaunchError::Backend
        );
    }

    #[test]
    fn child_startup_stderr_classification_tolerates_invalid_utf8_and_long_preambles()
    -> std::io::Result<()> {
        use super::{ChildStartupStage, parse_child_startup_failure, refine_backend_startup_error};

        let mut payload = vec![0xFF_u8, 0xFE, b'\n'];
        payload.extend(std::iter::repeat_n(b'a', 12_000));
        payload.push(b'\n');
        payload.extend_from_slice(
            b"private-inference-startup-failed stage=memory errorKind=Other osCode=none\r\n",
        );
        assert_eq!(
            parse_child_startup_failure(&payload),
            Some(ChildStartupStage::Memory)
        );

        let dir = std::env::temp_dir().join(format!(
            "Bonsai-long-stderr-classify-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(std::io::Error::other)?
                .as_nanos()
        ));
        std::fs::create_dir(&dir)?;
        let stderr_path = dir.join("backend.stderr");
        std::fs::write(&stderr_path, &payload)?;
        assert_eq!(
            refine_backend_startup_error(LaunchError::Backend, &stderr_path),
            (LaunchError::Memory, Some(ChildStartupStage::Memory))
        );
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn occupied_frontend_port_is_rejected_as_service_error() -> std::io::Result<()> {
        use super::preflight_frontend_port;
        use std::net::{Ipv4Addr, TcpListener};

        assert_eq!(preflight_frontend_port(0), Ok(()));
        let occupied = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))?;
        let port = occupied.local_addr()?.port();
        assert_eq!(preflight_frontend_port(port), Err(LaunchError::Service));
        drop(occupied);
        assert_eq!(preflight_frontend_port(port), Ok(()));
        Ok(())
    }

    #[test]
    fn cpu_prefill_tokens_require_native_swift_and_emit_paired_scratch_bytes()
    -> Result<(), LaunchError> {
        use bonsai_cpu_policy::context::{
            FIXED_RUNTIME_BYTES, SwiftContextPlan, SwiftContextProfile,
            swift_tiled_prefill_scratch_bytes,
        };

        let gib = 1024 * 1024 * 1024_u64;
        for (tokens, value) in [(8_u8, "8"), (16, "16"), (32, "32")] {
            let mut args = arguments();
            args.extend(
                [
                    "--profile",
                    "swift-pq2-assessment",
                    "--cpu-prefill-tokens",
                    value,
                ]
                .map(OsString::from),
            );
            let config = LaunchConfig::parse(args.clone())?;
            assert_eq!(config.cpu_prefill_tokens, Some(tokens));
            let (plan, scratch) = SwiftContextPlan::for_profile_with_prefill(
                24 * gib,
                SwiftContextProfile::NativeRsq4Full,
                Some(usize::from(tokens)),
            )
            .map_err(|_| LaunchError::Memory)?;
            let scratch = scratch.ok_or(LaunchError::Memory)?;
            assert_eq!(
                u64::try_from(scratch).ok(),
                swift_tiled_prefill_scratch_bytes(
                    usize::from(tokens),
                    plan.context_tokens,
                    bonsai_cpu_policy::context::SwiftKvCacheFormat::Rsq4,
                )
            );
            assert!(
                u64::try_from(plan.memory_limit_bytes - scratch).ok() >= Some(FIXED_RUNTIME_BYTES)
            );
            let data = std::env::temp_dir().join("Bonsai-prefill-args");
            let key = data.join("backend.key");
            let mut launch_plan = super::LaunchPlan::for_profile_with_options(
                &data,
                8081,
                &key,
                LaunchProfile::SwiftPq2Assessment,
                true,
                false,
            )
            .map_err(|_| LaunchError::Policy)?;
            launch_plan
                .select_swift_context(plan)
                .map_err(|_| LaunchError::Policy)?;
            launch_plan.cpu_prefill = Some((tokens, scratch));
            let child_args = config.native_arguments(&launch_plan, Some(&key));
            assert!(
                child_args
                    .array_windows::<2>()
                    .any(|pair| pair[0] == "--cpu-prefill-tokens" && pair[1] == value)
            );
            assert!(child_args.array_windows::<2>().any(|pair| {
                pair[0] == "--cpu-prefill-scratch-bytes" && pair[1] == scratch.to_string().as_str()
            }));
            args.extend(["--cpu-prefill-tokens", value].map(OsString::from));
            assert_eq!(
                LaunchConfig::parse(args).err(),
                Some(LaunchError::Arguments)
            );
        }
        for invalid in ["0", "4", "12", "64", "eight"] {
            let mut args = arguments();
            args.extend(
                [
                    "--profile",
                    "swift-pq2-assessment",
                    "--cpu-prefill-tokens",
                    invalid,
                ]
                .map(OsString::from),
            );
            assert_eq!(
                LaunchConfig::parse(args).err(),
                Some(LaunchError::Arguments)
            );
        }
        let mut publisher = arguments();
        publisher.extend(
            [
                "--profile",
                "swift-pq2-assessment",
                "--publisher-engine",
                "--cpu-prefill-tokens",
                "16",
            ]
            .map(OsString::from),
        );
        assert_eq!(
            LaunchConfig::parse(publisher).err(),
            Some(LaunchError::Arguments)
        );
        let mut reference = arguments();
        reference.extend(["--cpu-prefill-tokens", "8"].map(OsString::from));
        assert_eq!(
            LaunchConfig::parse(reference).err(),
            Some(LaunchError::Arguments)
        );
        Ok(())
    }

    #[test]
    fn native_arguments_emit_verified_sections_for_broker_pipe_and_omit_for_loopback()
    -> Result<(), LaunchError> {
        let mut args = arguments();
        args.extend(["--profile", "swift-pq2-assessment"].map(OsString::from));
        let config = LaunchConfig::parse(args)?;
        let data = std::env::temp_dir().join("Bonsai-verified-sections-args");
        let key = data.join("backend.key");
        let mmproj =
            bonsai_cpu_policy::swift_mmproj_artifact(&data).map_err(|_| LaunchError::Policy)?;

        for (with_adapter, with_mmproj, expected_count) in [
            (false, false, "1"),
            (true, false, "2"),
            (false, true, "2"),
            (true, true, "3"),
        ] {
            let mut plan = super::LaunchPlan::for_profile_with_options(
                &data,
                8081,
                &key,
                LaunchProfile::SwiftPq2Assessment,
                with_adapter,
                false,
            )
            .map_err(|_| LaunchError::Policy)?;
            assert_eq!(plan.adapter.is_some(), with_adapter);
            if with_mmproj {
                plan.enable_swift_mmproj(mmproj.clone())
                    .map_err(|_| LaunchError::Policy)?;
            }

            let pipe_args = config.native_arguments(&plan, None);
            assert!(pipe_args.iter().any(|arg| arg == "--broker-pipe"));
            assert!(
                pipe_args
                    .array_windows::<2>()
                    .any(|pair| pair[0] == "--verified-sections" && pair[1] == expected_count)
            );
            assert!(!pipe_args.iter().any(|arg| arg == "--key-file"));

            let loopback_args = config.native_arguments(&plan, Some(&key));
            assert!(!loopback_args.iter().any(|arg| arg == "--broker-pipe"));
            assert!(!loopback_args.iter().any(|arg| arg == "--verified-sections"));
            assert!(loopback_args.iter().any(|arg| arg == "--key-file"));
        }
        Ok(())
    }

    #[test]
    fn hexagon_runtime_readonly_paths_selects_present_runtime_files_only() -> std::io::Result<()> {
        let root = std::env::temp_dir().join(format!(
            "Bonsai-hexagon-readonly-paths-{}",
            std::process::id()
        ));
        let _cleanup = std::fs::remove_dir_all(&root);
        assert!(super::hexagon_runtime_readonly_paths(&root).is_empty());

        let runtime_dir = root.join("hexagon").join("runtime");
        std::fs::create_dir_all(&runtime_dir)?;
        std::fs::write(runtime_dir.join("libggml-htp-v81.so"), b"so81")?;
        std::fs::write(runtime_dir.join("hexagon-runtime.receipt.json"), b"{}")?;
        std::fs::write(runtime_dir.join("unrelated.bin"), b"ignored")?;

        let selected = super::hexagon_runtime_readonly_paths(&root);
        assert_eq!(
            selected,
            vec![
                runtime_dir.join("libggml-htp-v81.so"),
                runtime_dir.join("hexagon-runtime.receipt.json"),
            ]
        );
        std::fs::remove_dir_all(&root)?;
        Ok(())
    }

    #[test]
    fn calibrated_cow_default_admission_credits_mapped_weights_and_splits_child_limits()
    -> Result<(), LaunchError> {
        let mut args = arguments();
        args.extend(["--profile", "swift-pq2-assessment"].map(OsString::from));
        let config = LaunchConfig::parse(args)?;
        let gib = 1024 * 1024 * 1024_u64;
        let calibrated = config.calibrated_context_plan(24 * gib, true)?;
        assert_eq!(calibrated.context_plan.context_tokens, 262_144);
        assert!(calibrated.vision_admitted);
        assert!(calibrated.context_plan.memory_limit_bytes < calibrated.working_set_limit_bytes);

        let data = std::env::temp_dir().join("Bonsai-calibrated-cow-args");
        let key = data.join("backend.key");
        let mmproj =
            bonsai_cpu_policy::swift_mmproj_artifact(&data).map_err(|_| LaunchError::Policy)?;
        let mut plan = super::LaunchPlan::for_profile_with_options(
            &data,
            8081,
            &key,
            LaunchProfile::SwiftPq2Assessment,
            true,
            false,
        )
        .map_err(|_| LaunchError::Policy)?;
        plan.select_calibrated_swift_context(&calibrated)
            .map_err(|_| LaunchError::Policy)?;
        plan.enable_swift_mmproj(mmproj)
            .map_err(|_| LaunchError::Policy)?;

        let mmproj_reserve = usize::try_from(bonsai_cpu_policy::MMPROJ_RESERVE_BYTES)
            .map_err(|_| LaunchError::Memory)?;
        assert_eq!(
            plan.native_child_memory_limit_bytes() + mmproj_reserve,
            plan.working_set_limit_bytes
        );

        let child = config.native_arguments(&plan, None);
        assert!(
            child
                .array_windows::<2>()
                .any(|pair| pair[0] == "--admission-policy" && pair[1] == "calibrated-cow")
        );
        assert!(child.array_windows::<2>().any(|pair| {
            pair[0] == "--memory-limit-bytes"
                && pair[1] == plan.native_child_memory_limit_bytes().to_string().as_str()
        }));
        Ok(())
    }

    #[test]
    fn session_options_0_1_32_validate_reserves_vision_suppression_and_child_arguments()
    -> Result<(), LaunchError> {
        use crate::inference::{CpuFamily, CpuPrefillGdn, PromptReuseMode, SpeculationType};
        let mut native = arguments();
        native.extend(["--profile", "swift-pq2-assessment"].map(OsString::from));

        let mut branch_args = native.clone();
        branch_args.extend(
            [
                "--prompt-reuse",
                "prompt-end-and-latest",
                "--prompt-reuse-checkpoints",
                "4",
                "--cpu-prefill-tokens",
                "16",
                "--kv-cache",
                "f16",
                "--cpu-prefill-gdn",
                "wy-f32",
                "--cpu-family",
                "x1",
            ]
            .map(OsString::from),
        );
        let branch = LaunchConfig::parse(branch_args)?;
        assert_eq!(branch.prompt_reuse, PromptReuseMode::PromptEndAndLatest);
        assert_eq!(branch.prompt_reuse_checkpoints, Some(4));
        assert_eq!(branch.cpu_prefill_tokens, Some(16));
        assert_eq!(branch.cpu_prefill_gdn, Some(CpuPrefillGdn::WyF32));
        assert_eq!(branch.cpu_family, Some(CpuFamily::X1));
        assert_eq!(branch.native_row_workers, 8);
        assert!(branch.runtime_supports_vision()?);

        let gib = 1024 * 1024 * 1024_u64;
        let calibrated_branch = branch.calibrated_context_plan(48 * gib, true)?;
        let mut branch_plan = super::LaunchPlan::for_profile_with_options(
            &branch.data,
            8081,
            &branch.evidence_parent.join("backend.key"),
            LaunchProfile::SwiftPq2Assessment,
            true,
            false,
        )
        .map_err(|_| LaunchError::Policy)?;
        branch_plan
            .select_calibrated_swift_context(&calibrated_branch)
            .map_err(|_| LaunchError::Policy)?;
        branch_plan.cpu_prefill = branch
            .cpu_prefill_tokens
            .zip(calibrated_branch.prefill_scratch_bytes);
        let branch_child = branch.native_arguments(&branch_plan, None);
        for [flag, value] in [
            ["--cpu-family", "x1"],
            ["--prompt-reuse", "prompt-end-and-latest"],
            ["--prompt-reuse-checkpoints", "4"],
            ["--cpu-prefill-gdn", "wy-f32"],
        ] {
            assert!(
                branch_child
                    .array_windows::<2>()
                    .any(|pair| pair[0] == flag && pair[1] == value),
                "missing [{flag}, {value}] in {branch_child:?}"
            );
        }

        let mut spec_args = native.clone();
        spec_args.extend(
            [
                "--cpu-prefill-tokens",
                "16",
                "--spec-type",
                "suffix",
                "--draft-max",
                "6",
                "--cpu-family",
                "x2",
                "--native-row-workers",
                "4",
            ]
            .map(OsString::from),
        );
        let spec = LaunchConfig::parse(spec_args)?;
        assert_eq!(spec.spec_type, Some(SpeculationType::Suffix));
        assert_eq!(spec.draft_max, Some(6));
        assert_eq!(spec.effective_speculation_tokens(), Some(6));
        assert_eq!(spec.native_row_workers, 4);
        assert!(
            !spec.runtime_supports_vision()?,
            "suffix speculation must automatically suppress optional mmproj"
        );
        let calibrated_spec = spec.calibrated_context_plan(24 * gib, false)?;
        let mut spec_plan = super::LaunchPlan::for_profile_with_options(
            &spec.data,
            8081,
            &spec.evidence_parent.join("backend.key"),
            LaunchProfile::SwiftPq2Assessment,
            true,
            false,
        )
        .map_err(|_| LaunchError::Policy)?;
        spec_plan
            .select_calibrated_swift_context(&calibrated_spec)
            .map_err(|_| LaunchError::Policy)?;
        spec_plan.cpu_prefill = spec
            .cpu_prefill_tokens
            .zip(calibrated_spec.prefill_scratch_bytes);
        spec_plan.cpu_speculation = spec
            .effective_speculation_tokens()
            .zip(calibrated_spec.speculation_reserves)
            .map(|(tokens, (verify, checkpoint))| (tokens, verify, checkpoint));
        let spec_child = spec.native_arguments(&spec_plan, None);
        assert!(
            spec_child
                .array_windows::<2>()
                .any(|pair| pair[0] == "--spec-type" && pair[1] == "suffix")
        );
        assert!(
            spec_child
                .array_windows::<2>()
                .any(|pair| pair[0] == "--draft-max" && pair[1] == "6")
        );
        assert!(
            spec_child
                .iter()
                .all(|flag| flag != "--cpu-speculative-tokens")
        );
        assert!(
            spec_child
                .iter()
                .any(|flag| flag == "--cpu-verification-scratch-bytes")
        );
        assert!(
            spec_child
                .iter()
                .any(|flag| flag == "--cpu-speculative-checkpoint-bytes")
        );

        for invalid in [
            vec!["--prompt-reuse-checkpoints", "2"],
            vec![
                "--prompt-reuse",
                "disabled",
                "--prompt-reuse-checkpoints",
                "2",
            ],
            vec![
                "--prompt-reuse",
                "latest",
                "--prompt-reuse-checkpoints",
                "0",
            ],
            vec![
                "--prompt-reuse",
                "latest",
                "--prompt-reuse-checkpoints",
                "9",
            ],
            vec![
                "--prompt-reuse",
                "latest",
                "--prompt-reuse-checkpoints",
                "02",
            ],
            vec![
                "--prompt-reuse",
                "latest",
                "--prompt-reuse-checkpoint-bytes",
                "0",
            ],
            vec!["--cpu-prefill-gdn", "wy-f32"],
            vec!["--cpu-prefill-tokens", "16", "--cpu-prefill-gdn", "wy-f32"],
            vec![
                "--cpu-prefill-tokens",
                "16",
                "--kv-cache",
                "rsq4",
                "--cpu-prefill-gdn",
                "wy-f32",
            ],
            vec!["--draft-max", "4"],
            vec!["--spec-type", "none", "--draft-max", "4"],
            vec!["--spec-type", "suffix"],
            vec![
                "--cpu-prefill-tokens",
                "16",
                "--kv-cache",
                "f16",
                "--cpu-prefill-gdn",
                "wy-f32",
                "--spec-type",
                "suffix",
            ],
            vec!["--cpu-family", "x3"],
            vec!["--publisher-engine", "--cpu-family", "x2"],
        ] {
            let mut rejected = native.clone();
            rejected.extend(invalid.into_iter().map(OsString::from));
            assert_eq!(
                LaunchConfig::parse(rejected).err(),
                Some(LaunchError::Arguments)
            );
        }
        Ok(())
    }
}
