//! Explicit Windows development launch of a compiled, pinned CPU backend.
//!
//! This owns lifetime, credentials, artifact handles and resource monitoring.
//! The native sandbox is required unless the direct development caller explicitly
//! selects the unisolated backend. Same-user interference remains unqualified.

mod identity;
mod startup;

use identity::{WorkerReadiness, validate_native_identity};
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
    "  [--native-row-workers 1|4|8|12] (Rust defaults to 8 on Snapdragon X1 or 4 otherwise; publisher supports only 1)\n",
    "  [--prompt-reuse disabled|latest] (experimental exact latest-state reuse; Rust only)\n",
    "  [--kv-cache f16|rsq4|rsq4k3v] (reviewed native runtime only; rsq4 default when supported, f16 explicit recovery)\n",
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
    latest_prompt_reuse: bool,
    kv_cache: Option<crate::inference::KvCache>,
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
        let (mut native_row_workers, mut latest_prompt_reuse) = (None, None);
        let (mut kv_cache, mut placement) = (None, None);
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
                Some("--prompt-reuse") if latest_prompt_reuse.is_none() => {
                    latest_prompt_reuse = Some(prompt_reuse(args.next().as_deref())?);
                }
                Some("--kv-cache") if kv_cache.is_none() => {
                    kv_cache = Some(crate::inference::KvCache::parse(args.next().as_deref())?);
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
            bonsai_cpu_policy::hardware::SnapdragonFamily::default_native_row_workers(family)
        } else {
            1
        });
        let latest_prompt_reuse = latest_prompt_reuse.unwrap_or(false);
        let kv_cache = crate::inference::pinned_kv_cache(native_engine, kv_cache)?;
        if placement.is_some() && profile != LaunchProfile::SwiftPq2Assessment {
            return Err(LaunchError::Arguments);
        }
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
            latest_prompt_reuse,
            kv_cache,
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
        Ok(configuration)
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
            || (self.latest_prompt_reuse && !self.native_engine)
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

    // Call only after the exact released runtime's context and cache-accounting
    // contract is admitted. Cache selection remains immutable through retry.
    fn swift_context_plan(
        &self,
        available_bytes: u64,
    ) -> Result<bonsai_cpu_policy::context::SwiftContextPlan, LaunchError> {
        bonsai_cpu_policy::context::SwiftContextPlan::for_profile(
            available_bytes,
            self.swift_context_profile()?,
        )
        .map_err(|_| LaunchError::Memory)
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
        let _native_executable = self.native_executable_guard()?;
        let _artifacts = self.verify_artifacts(&plan)?;
        let _adapter = Self::adapter_guard(&plan)?;
        let _mmproj = self.mmproj_guard()?;
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
                self.latest_prompt_reuse,
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
            arguments.push(OsString::from("--broker-pipe"));
        }
        arguments.extend([OsString::from("--seed"), OsString::from("42")]);
        if crate::inference::memory_context_supported().unwrap_or(false) {
            arguments.extend([
                OsString::from("--context-tokens"),
                OsString::from(plan.context_tokens.to_string()),
                OsString::from("--memory-limit-bytes"),
                OsString::from(plan.memory_limit_bytes.to_string()),
            ]);
        }
        if self.native_row_workers > 1 {
            arguments.extend([
                OsString::from("--row-workers"),
                OsString::from(self.native_row_workers.to_string()),
            ]);
        }
        if self.latest_prompt_reuse {
            arguments.extend([OsString::from("--prompt-reuse"), OsString::from("latest")]);
        }
        if let Some(policy) = self.kv_cache {
            arguments.extend([OsString::from("--kv-cache"), OsString::from(policy.name())]);
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
        native: Option<(PathBuf, bonsai_artifacts::VerifiedFile)>,
        broker_pipe: bool,
    ) -> Result<(ProcessSpec, Option<bonsai_artifacts::VerifiedFile>), LaunchError> {
        // The worker may write only this child, never the sibling keys or logs.
        // Its parent already has the credential owner's protected inheritable ACL.
        let working_directory = credentials.directory().join("worker");
        std::fs::create_dir(&working_directory).map_err(|_| LaunchError::Path)?;
        validate_external_path(&working_directory).map_err(|_| LaunchError::Path)?;
        let (executable, arguments, guard) = if let Some((executable, guard)) = native {
            let key_path = (!broker_pipe).then(|| credentials.backend_path());
            let arguments = self.native_arguments(&plan, key_path);
            (executable, arguments, Some(guard))
        } else {
            (plan.executable, plan.arguments, None)
        };
        Ok((
            ProcessSpec {
                executable,
                arguments,
                working_directory,
                stdout: credentials.directory().join("backend.stdout"),
                stderr: credentials.directory().join("backend.stderr"),
                memory_limit_bytes: plan.memory_limit_bytes,
                active_process_limit: 1,
            },
            guard,
        ))
    }

    fn start_process(
        &self,
        specification: &ProcessSpec,
        artifacts: &VerifiedArtifacts,
        adapter: Option<&bonsai_artifacts::VerifiedFile>,
        mmproj: Option<&bonsai_artifacts::VerifiedFile>,
        credentials: &PrivateCredentials,
        broker_pipe: bool,
    ) -> Result<(OwnedProcess, Option<(std::fs::File, std::fs::File)>), LaunchError> {
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
        let started = if broker_pipe {
            OwnedProcess::start_sandboxed_broker_pipe(specification, &policy)
                .map(|(process, child_in, child_out)| (process, Some((child_in, child_out))))
        } else {
            OwnedProcess::start_sandboxed(specification, &policy).map(|process| (process, None))
        };
        started.map_err(|error| {
            // The native boundary emits fixed validation reasons or Windows API
            // errors, never arguments, file contents or credential values. Keep
            // that technical cause in the broker's retained stderr while the UI
            // continues to receive only the value-free launch classification.
            let _result = write_sandbox_failure(&mut io::stderr().lock(), &error);
            LaunchError::Sandbox
        })
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
        let native_executable = self.native_executable_guard()?;
        startup_timing.phase(StartupPhase::ModelVerification);
        let artifacts = self.verify_artifacts(&preliminary)?;
        startup_timing.phase(StartupPhase::AdapterVerification);
        let adapter = Self::adapter_guard(&preliminary)?;
        let mut mmproj = self.mmproj_guard()?;
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
        let mut plan = LaunchPlan::for_profile_with_options(
            &self.data,
            self.backend_port,
            credentials.backend_path(),
            self.profile,
            self.use_abliterate_lora,
            self.uses_repacked_ptq1(),
        )
        .map_err(|_| LaunchError::Policy)?;
        if self.memory_context_supported()? {
            let available = available_memory_bytes().map_err(|_| LaunchError::Memory)?;
            let profile = self.swift_context_profile()?;
            let (context, vision_admitted) =
                bonsai_cpu_policy::context::SwiftContextPlan::for_profile_with_vision(
                    available,
                    profile,
                    mmproj.is_some(),
                )
                .map_err(|_| LaunchError::Memory)?;
            if !vision_admitted {
                mmproj = None;
            }
            plan.select_swift_context(context)
                .map_err(|_| LaunchError::Policy)?;
        }
        if let Some((artifact, _)) = &mmproj {
            plan.enable_swift_mmproj(artifact.clone())
                .map_err(|_| LaunchError::Policy)?;
        }
        let vision_enabled = mmproj.is_some();
        let mmproj_file = mmproj.map(|(_, guard)| guard);
        let context_tokens = plan.context_tokens;
        // Reserve only as a collision check in TCP modes; offline broker-pipe binds no port.
        let reservation = if broker_pipe {
            None
        } else {
            Some(bound_loopback_socket(self.backend_port).map_err(|_| LaunchError::Port)?)
        };
        admit(plan.required_available_bytes)?;
        let (specification, native_executable) =
            self.specification(plan, &credentials, native_executable, broker_pipe)?;
        validate_external_path(&specification.executable).map_err(|_| LaunchError::Path)?;
        drop(reservation);
        let storage_budget = self.requires_native_sandbox().then_some((
            MAX_WORKER_PRIVATE_STORAGE_BYTES,
            MAX_WORKER_PRIVATE_STORAGE_ENTRIES,
        ));
        let worker_root = specification.working_directory.clone();
        startup_timing.phase(StartupPhase::ProcessCreation);
        let (process, pipe_handles) = self.start_process(
            &specification,
            &artifacts,
            adapter.as_ref(),
            mmproj_file.as_ref(),
            &credentials,
            broker_pipe,
        )?;
        if guard(
            &process,
            specification.memory_limit_bytes,
            storage_budget,
            Some(worker_root.as_path()),
        )
        .is_err()
        {
            // Keep sandbox storage rejections distinct if the check tripped right after spawn.
            let err = check_storage_budget(&process, storage_budget, Some(worker_root.as_path()))
                .err()
                .unwrap_or(LaunchError::Backend);
            process.stop(STOP_TIMEOUT).map_err(|_| LaunchError::Stop)?;
            return Err(err);
        }
        startup_timing.phase(StartupPhase::AuthenticatedReadiness);
        let (pipe_backend, models) = if let Some((child_stdin, child_stdout)) = pipe_handles {
            let pipe_backend = match crate::pipe_transport::PipeBackend::connect_stdio(
                child_stdin,
                child_stdout,
                credentials.backend_key(),
            ) {
                Ok(backend) => backend,
                Err(_) => {
                    note_startup_failure(
                        StartupStage::AuthenticatedReadiness,
                        LaunchError::Backend,
                        &process,
                    );
                    process.stop(STOP_TIMEOUT).map_err(|_| LaunchError::Stop)?;
                    return Err(LaunchError::Backend);
                }
            };
            let models = match wait_ready_models_pipe(
                &process,
                &pipe_backend,
                credentials.backend_key(),
                specification.memory_limit_bytes,
                storage_budget,
                Some(worker_root.as_path()),
                shutdown,
            )
            .await
            {
                Ok(models) => models,
                Err(error) => {
                    note_startup_failure(StartupStage::AuthenticatedReadiness, error, &process);
                    process.stop(STOP_TIMEOUT).map_err(|_| LaunchError::Stop)?;
                    return Err(error);
                }
            };
            (Some(pipe_backend), models)
        } else {
            let models = match wait_ready_models(
                &process,
                self.backend_port,
                credentials.backend_key(),
                specification.memory_limit_bytes,
                storage_budget,
                Some(worker_root.as_path()),
                shutdown,
            )
            .await
            {
                Ok(models) => models,
                Err(error) => {
                    note_startup_failure(StartupStage::AuthenticatedReadiness, error, &process);
                    process.stop(STOP_TIMEOUT).map_err(|_| LaunchError::Stop)?;
                    return Err(error);
                }
            };
            (None, models)
        };
        if self.native_engine {
            startup_timing.phase(StartupPhase::NativeIdentity);
            self.confirm_native_identity(&process, &models, context_tokens, vision_enabled)?;
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
        Ok(LaunchedService {
            service,
            process,
            credentials,
            _artifacts: artifacts,
            _adapter: adapter,
            _mmproj: mmproj_file,
            _native_executable: native_executable,
            memory_limit_bytes: specification.memory_limit_bytes,
            backend_port: if broker_pipe { 0 } else { self.backend_port },
            storage_budget,
        })
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
    ) -> Result<(), LaunchError> {
        let valid = crate::inference::pinned_version().is_ok_and(|version| {
            validate_native_identity(
                models,
                self.uses_repacked_ptq1(),
                self.use_abliterate_lora,
                WorkerReadiness::for_version(&version, self.native_row_workers)
                    .with_prompt_reuse(self.latest_prompt_reuse)
                    .with_vision(vision_enabled)
                    .with_kv_cache(self.kv_cache)
                    .with_placement(self.placement)
                    .with_context_tokens(context_tokens),
            )
            .is_ok()
        });
        if !valid {
            note_startup_failure(StartupStage::NativeIdentity, LaunchError::Backend, process);
            process.stop(STOP_TIMEOUT).map_err(|_| LaunchError::Stop)?;
            return Err(LaunchError::Backend);
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum StartupStage {
    AuthenticatedReadiness,
    NativeIdentity,
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
    write!(
        output,
        "Native backend startup failure: stage={stage}, error={error:?}, process="
    )?;
    match status {
        Ok(Some(code)) => writeln!(output, "exited-0x{code:08X}"),
        Ok(None) => writeln!(output, "running"),
        Err(_) => writeln!(output, "status-unavailable"),
    }
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

fn prompt_reuse(value: Option<&std::ffi::OsStr>) -> Result<bool, LaunchError> {
    match value.and_then(std::ffi::OsStr::to_str) {
        Some("disabled") => Ok(false),
        Some("latest") => Ok(true),
        _ => Err(LaunchError::Arguments),
    }
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
    limit: usize,
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
    if memory.private_bytes > limit || memory.working_set_bytes > limit {
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
    limit: usize,
    storage_budget: Option<(u64, usize)>,
    worker_root: Option<&std::path::Path>,
    shutdown: impl Future<Output = io::Result<()>>,
) -> Result<Vec<u8>, LaunchError> {
    run_startup_with_timeout(shutdown, async {
        loop {
            guard(process, limit, storage_budget, worker_root)?;
            match listener_process_id(port).map_err(|_| LaunchError::Backend)? {
                Some(pid) if pid != process.id() => return Err(LaunchError::Backend),
                Some(_) if healthy(port, key, process.id()).await => {
                    if let Some(models) = authenticated_models(port, key, process.id()).await {
                        owner(process, port)?;
                        guard(process, limit, storage_budget, worker_root)?;
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
    limit: usize,
    storage_budget: Option<(u64, usize)>,
    worker_root: Option<&std::path::Path>,
    shutdown: impl Future<Output = io::Result<()>>,
) -> Result<Vec<u8>, LaunchError> {
    run_startup_with_timeout(shutdown, async {
        guard(process, limit, storage_budget, worker_root)?;
        {
            let ready_deadline = tokio::time::Instant::now() + STARTUP_TIMEOUT;
            let mut ready = pin!(backend.wait_server_ready(ready_deadline));
            let mut tick = interval(Duration::from_millis(250));
            poll_fn(|cx| {
                if let Poll::Ready(result) = ready.as_mut().poll(cx) {
                    return Poll::Ready(match result {
                        Ok(()) => Ok(()),
                        Err(_) => Err(guard(process, limit, storage_budget, worker_root)
                            .err()
                            .unwrap_or(LaunchError::Backend)),
                    });
                }
                while tick.poll_tick(cx).is_ready() {
                    if let Err(error) = guard(process, limit, storage_budget, worker_root) {
                        return Poll::Ready(Err(error));
                    }
                }
                Poll::Pending
            })
            .await?;
        }
        guard(process, limit, storage_budget, worker_root)?;
        let health = pipe_exchange(backend, "/health", Some(key))
            .await
            .and_then(|bytes| validate_health(&bytes))
            .is_ok();
        if !health {
            return Err(guard(process, limit, storage_budget, worker_root)
                .err()
                .unwrap_or(LaunchError::Backend));
        }
        let models = pipe_authenticated_models(backend, key)
            .await
            .ok_or_else(|| {
                guard(process, limit, storage_budget, worker_root)
                    .err()
                    .unwrap_or(LaunchError::Backend)
            })?;
        guard(process, limit, storage_budget, worker_root)?;
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
    wait_ready_models(process, port, key, limit, None, None, shutdown)
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
            4
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
        let mut native = arguments();
        native.extend(["--profile", "swift-pq2-assessment"].map(OsString::from));
        assert!(!LaunchConfig::parse(native.clone())?.latest_prompt_reuse);
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
        for (name, latest) in [("disabled", false), ("latest", true)] {
            let mut selected = native.clone();
            selected.extend(["--prompt-reuse", name].map(OsString::from));
            let config = LaunchConfig::parse(selected.clone())?;
            assert_eq!(config.latest_prompt_reuse, latest);
            let child = config.native_arguments(&plan, Some(&key));
            assert_eq!(
                child
                    .array_windows()
                    .filter(|[flag, value]| flag == "--prompt-reuse" && value == "latest")
                    .count(),
                usize::from(latest)
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
        let available = 20 * 1024 * 1024 * 1024;
        let packed = config.swift_context_plan(available)?;
        assert_eq!(packed.context_tokens, 149_760);
        // Only the exact reviewed Swift K4/V4 state scope reaches the model
        // ceiling; its ordinary process and host charges remain funded.
        assert!(crate::inference::full_context_supported()?);
        assert_eq!(
            config
                .swift_context_plan(24 * 1024 * 1024 * 1024)?
                .context_tokens,
            262_144
        );
        config.kv_cache = Some(KvCache::Rsq4K3v);
        let capacity = config.swift_context_plan(available)?;
        assert_eq!(capacity.context_tokens, 170_240);
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

        // Native 0.1.30 supports vision unless --no-mmproj opts out.
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
    fn require_broker_pipe_enforces_sandbox_native_engine_and_reviewed_release() {
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
        // While BROKER_PIPE_RELEASE is None, valid flag combinations fail closed with NativeRuntime.
        assert_eq!(
            LaunchConfig::parse(valid.clone()).err(),
            Some(LaunchError::NativeRuntime)
        );

        // Duplicate flag is rejected as Arguments.
        let mut dup = valid.clone();
        dup.push("--require-broker-pipe".into());
        assert_eq!(LaunchConfig::parse(dup).err(), Some(LaunchError::Arguments));

        // Optional explicit --backend-port still fails closed with NativeRuntime when release is unpinned.
        valid.extend(["--backend-port", "8081"].map(OsString::from));
        assert_eq!(
            LaunchConfig::parse(valid).err(),
            Some(LaunchError::NativeRuntime)
        );

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
    }
}
