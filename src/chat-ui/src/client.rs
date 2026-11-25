//! Owned, bounded loopback client for the native chat window.

#![forbid(unsafe_code)]
#![expect(
    clippy::redundant_pub_crate,
    reason = "The window and worker share a deliberately crate-private command boundary"
)]

use bonsai_windows_credentials::RetainedCredential;
use bonsai_windows_process::{OwnedProcess, ToolProcessSpec};
use serde::{Deserialize, Serialize};
use std::{
    borrow::Cow,
    collections::BTreeMap,
    ffi::OsString,
    fs::File,
    io::{self, IoSlice, Read, Write},
    net::{Ipv4Addr, SocketAddr, SocketAddrV4, TcpStream},
    path::{Path, PathBuf},
    sync::{
        Arc,
        mpsc::{Receiver, RecvTimeoutError, TryRecvError},
    },
    thread,
    time::{Duration, Instant},
};

#[cfg(test)]
mod cancellation_tests;
mod effort;
mod event_queue;
#[cfg(test)]
mod memory_tests;
mod owner_control;
mod sse;

pub(crate) use event_queue::{
    EventReceiver, EventSender, channel as event_channel,
    channel_with_repaint as event_channel_with_repaint,
};
use sse::Sse;

const POLL: Duration = Duration::from_millis(100);
const IO_POLL: Duration = Duration::from_millis(50);
const SOCKET_POLL: Duration = Duration::from_millis(25);
const STARTUP_TIMEOUT: Duration = Duration::from_mins(10);
// Ten seconds for the broker's explicit model stop, then bounded exit polling
// and outer-Job cleanup. All phases share this one absolute deadline.
const BROKER_STOP: Duration = Duration::from_secs(15);
const MAX_HEAD: usize = 16 * 1024;
const MAX_BODY: usize = 4 * 1024 * 1024;
// Keep aligned with bonsai_broker::MAX_BODY_BYTES. The pinned native runtime's
// worker frames and tokenizer retain independent 1 MiB limits; increasing only
// the desktop HTTP limit would admit requests the selected runtime cannot use.
const MAX_REQUEST: usize = 1024 * 1024;
const MAX_MULTIMODAL_REQUEST: usize = 16 * 1024 * 1024;
pub(crate) const MAX_REQUEST_IMAGES: usize = 24;
const MAX_EVENT: usize = 1024 * 1024;
const MAX_ERROR_BODY: usize = 256;
const READ_SIZE: usize = 8192;
const EVIDENCE_PREFIX: &str = "Protected credentials and technical evidence: ";
const RUNNING_MEMORY_MESSAGE: &str = "Swift stopped because its running memory requirements were no longer met. Close other apps, then select Retry model. Completed conversation turns remain in memory.";
const PUBLISHER_INIT_FAILURE_MESSAGE: &str = "The publisher CPU runtime failed Windows initialization (0xC0000142). Retry model repeats that runtime choice. Copy any conversation or drafts you want to keep before closing. Close this app and relaunch bonsai-tools chat with --native-engine to try the experimental Rust runtime, preserving your DataDirectory and the same model and adapter options.";
const LOCAL_CONNECTION_CLOSED: &str = "The connection to the model on this computer closed before a complete reply was received. The cause was not reported.";
const CONTEXT_LIMIT_MESSAGE: &str = "This conversation and the selected output cap do not fit within the model's selected context limit. Choose a smaller output cap or start a new conversation, then send again.";
const REQUEST_LIMIT_MESSAGE: &str = "This conversation is too large for the local model's request limit. Shorten the message or start a new conversation, then send again.";
const INVALID_REQUEST_MESSAGE: &str = "The local model could not accept this conversation or its response settings. Edit the message or start a new conversation, then send again.";
const TOKEN_COUNT_FAILURE_MESSAGE: &str = "The local model failed while checking this conversation before generation. No reply was started.";
const GENERATION_FAILURE_MESSAGE: &str =
    "The local model failed while starting this reply. No reply was received.";
const CONTROL_TOKEN_MESSAGE: &str = "This conversation contains reserved chat markers, such as <|im_start|> or <think>. Remove them or start a new conversation before sending again.";
const RATE_LIMIT_MESSAGE: &str =
    "Too many messages were sent in a short time. Wait a moment, then send again.";
const TEARDOWN_FAILURE_MESSAGE: &str = "The local model's owned processes could not be confirmed stopped. Copy any conversation or drafts you want to keep, then close this app before launching again.";
const NATIVE_IDENTITY_GUIDANCE: &[(&str, &str)] = &[
    (
        "Native identity verification failed: reason=release-pin.",
        "Swift's compiled Rust runtime version check failed readiness verification. Close and reopen Swift (or rebuild matching local binaries) instead of retrying. Completed conversation turns remain in memory.",
    ),
    (
        "Native identity verification failed: reason=framing.",
        "The local Rust runtime returned an unexpected readiness response format. Close and reopen Swift (or rebuild matching local binaries) instead of retrying. Completed conversation turns remain in memory.",
    ),
    (
        "Native identity verification failed: reason=schema.",
        "The local Rust runtime returned an unsupported readiness schema. Close and reopen Swift (or rebuild matching local binaries) instead of retrying. Completed conversation turns remain in memory.",
    ),
    (
        "Native identity verification failed: reason=model.",
        "The loaded local model identity did not match Swift's compiled contract. Close and reopen Swift (or rebuild matching local binaries) instead of retrying. Completed conversation turns remain in memory.",
    ),
    (
        "Native identity verification failed: reason=context-tokens.",
        "The local Rust runtime reported a different context window than Swift selected. Close and reopen Swift (or rebuild matching local binaries) instead of retrying. Completed conversation turns remain in memory.",
    ),
    (
        "Native identity verification failed: reason=row-workers.",
        "The local Rust runtime reported a different worker count than Swift selected. Close and reopen Swift (or rebuild matching local binaries) instead of retrying. Completed conversation turns remain in memory.",
    ),
    (
        "Native identity verification failed: reason=prompt-reuse.",
        "The local Rust runtime reported a different prompt-reuse setting than Swift selected. Close and reopen Swift (or rebuild matching local binaries) instead of retrying. Completed conversation turns remain in memory.",
    ),
    (
        "Native identity verification failed: reason=kv-cache-graph.",
        "The local Rust runtime's compiled KV-cache or adapter graph identity did not match Swift's contract. Close and reopen Swift (or rebuild matching local binaries) instead of retrying. Completed conversation turns remain in memory.",
    ),
    (
        "Native identity verification failed: reason=placement.",
        "The local Rust runtime reported a different accelerator placement than Swift selected. Close and reopen Swift (or rebuild matching local binaries) instead of retrying. Completed conversation turns remain in memory.",
    ),
    (
        "Native identity verification failed: reason=startup-projection.",
        "The local Rust runtime did not report the expected startup projection state. Close and reopen Swift (or rebuild matching local binaries) instead of retrying. Completed conversation turns remain in memory.",
    ),
    (
        "Native identity verification failed: reason=adapter.",
        "The local Rust runtime's loaded LoRA adapter identity did not match Swift's contract. Close and reopen Swift (or rebuild matching local binaries) instead of retrying. Completed conversation turns remain in memory.",
    ),
    (
        "Native identity verification failed: reason=vision.",
        "The local Rust runtime's vision projector identity did not match Swift's contract. Close and reopen Swift (or rebuild matching local binaries) instead of retrying. Completed conversation turns remain in memory.",
    ),
    (
        "Native identity verification failed: reason=cpu-prefill.",
        "The local Rust runtime's tiled CPU prefill configuration did not match Swift's contract. Close and reopen Swift (or rebuild matching local binaries) instead of retrying. Completed conversation turns remain in memory.",
    ),
    (
        "Native identity verification failed: reason=cpu-speculation.",
        "The local Rust runtime's speculative decoding configuration did not match Swift's contract. Close and reopen Swift (or rebuild matching local binaries) instead of retrying. Completed conversation turns remain in memory.",
    ),
];
const POLICY_FAILURE_MESSAGE: &str = "The local model launch policy rejected the current environment or configuration. Restart Swift from a clean session without model or runtime environment overrides. Completed conversation turns remain in memory.";
const PATH_OR_CREDENTIALS_FAILURE_MESSAGE: &str = "Swift could not safely access its local data or temporary credential folder without links or redirects. Check folder permissions and local storage, then close and reopen Swift. Completed conversation turns remain in memory.";
const ARGUMENTS_FAILURE_MESSAGE: &str = "The app and local broker disagreed on launch options. Close and reopen Swift (or rebuild matching local binaries) instead of retrying. Completed conversation turns remain in memory.";
const SANDBOX_REBOOT_PENDING_MESSAGE: &str = "Windows Update has a pending restart required for the native Windows sandbox (PSEC 1.1). Restart Windows, then reopen Swift. Completed conversation turns remain in memory.";
const SANDBOX_UNAVAILABLE_OFFLINE_MESSAGE: &str = "This Windows build lacks the required native Windows sandbox (PSEC 1.1). Install the latest Windows updates, restart Windows, and reopen Swift. Completed conversation turns remain in memory.";
const SANDBOX_UNAVAILABLE_LOOPBACK_MESSAGE: &str = "The required native Windows sandbox is unavailable. Update Windows to a build with process containment and container-hosted loopback support, restart Windows, and reopen Swift. Completed conversation turns remain in memory.";
const NON_RETRYABLE_STARTUP_MESSAGES: [&str; 6] = [
    POLICY_FAILURE_MESSAGE,
    PATH_OR_CREDENTIALS_FAILURE_MESSAGE,
    ARGUMENTS_FAILURE_MESSAGE,
    SANDBOX_REBOOT_PENDING_MESSAGE,
    SANDBOX_UNAVAILABLE_OFFLINE_MESSAGE,
    SANDBOX_UNAVAILABLE_LOOPBACK_MESSAGE,
];
pub(crate) const STOPPED_MESSAGE: &str = "Response stopped. Ready when you are.";
/// System instructions prepended to every model turn to enable structured SVG image generation.
pub(crate) const SYSTEM_PROMPT: &str = "\
You are Swift Bonsai, a helpful assistant running locally on the user's device. \
When the user asks you to generate, draw, illustrate, create, or edit an image, artwork, diagram, or icon, \
include a complete, self-contained `<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 WIDTH HEIGHT\">...</svg>` document \
inside a top-level ```svg fenced code block. \
Use clean vector geometry, shapes, paths, gradients, and explicit `viewBox` dimensions; \
never include `<script>` elements, event-handler attributes, or external URLs or `<image href>`. \
If the user explicitly asks to inspect raw SVG markup or source code, use a ```xml block instead. \
For all ordinary requests, respond normally without emitting an ```svg block.";

pub(crate) const DEFAULT_ABLITERATION_SCALE: f32 = 1.3;
pub(crate) const DEFAULT_DAISYROSE_SCALE: f32 = 0.0;

pub(crate) const fn default_abliteration_scale() -> f32 {
    DEFAULT_ABLITERATION_SCALE
}

pub(crate) fn valid_adapter_scale(scale: f32) -> bool {
    scale.is_finite() && (0.0..=2.0).contains(&scale)
}

pub(crate) fn valid_abliteration_scale(scale: f32) -> bool {
    valid_adapter_scale(scale)
}

pub(crate) const fn default_daisyrose_scale() -> f32 {
    DEFAULT_DAISYROSE_SCALE
}

pub(crate) fn valid_daisyrose_scale(scale: f32) -> bool {
    valid_adapter_scale(scale)
}

/// Normalize a DaisyRose `SHA-256` fingerprint to its leading 12 lowercase hex characters.
pub(crate) fn valid_daisyrose_sha256_prefix(raw: &str) -> Option<String> {
    if raw.len() < 12
        || !raw
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return None;
    }
    Some(raw[..12].to_owned())
}

/// Broker-reported DaisyRose multi-rank `LoRA` readiness, default scale, and verified identity.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct DaisyRoseReadiness {
    pub(crate) available: bool,
    pub(crate) default_scale_tenths: u8,
    pub(crate) max_rank: Option<u16>,
    pub(crate) sha256_prefix: Option<String>,
}

impl DaisyRoseReadiness {
    pub(crate) const fn default_enabled(&self) -> bool {
        self.available && self.default_scale_tenths > 0
    }

    pub(crate) fn default_scale(&self) -> f32 {
        if self.available {
            f32::from(self.default_scale_tenths) / 10.0
        } else {
            DEFAULT_DAISYROSE_SCALE
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum KvCache {
    F16,
    Rsq4,
    Rsq4Kvf,
    Rsq4K3v,
}

impl KvCache {
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::Rsq4 => "rsq4",
            Self::Rsq4Kvf => "rsq4kvf",
            Self::Rsq4K3v => "rsq4k3v",
        }
    }

    pub(crate) fn from_cli(value: &str) -> Option<Self> {
        match value {
            "f16" => Some(Self::F16),
            "rsq4" => Some(Self::Rsq4),
            "rsq4kvf" => Some(Self::Rsq4Kvf),
            "rsq4k3v" => Some(Self::Rsq4K3v),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Placement {
    Auto,
    Cpu,
    Vulkan,
    Composite,
}

impl Placement {
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Vulkan => "vulkan",
            Self::Composite => "composite",
        }
    }

    pub(crate) fn from_cli(value: &str) -> Option<Self> {
        match value {
            "auto" => Some(Self::Auto),
            "cpu" => Some(Self::Cpu),
            "vulkan" => Some(Self::Vulkan),
            "composite" => Some(Self::Composite),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum PromptReuse {
    #[default]
    Disabled,
    Latest,
    PromptEndAndLatest,
}

impl PromptReuse {
    pub(crate) const fn enabled(self) -> bool {
        !matches!(self, Self::Disabled)
    }

    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::Latest => "latest",
            Self::PromptEndAndLatest => "prompt-end-and-latest",
        }
    }

    pub(crate) fn from_cli(value: &str) -> Option<Self> {
        match value {
            "disabled" => Some(Self::Disabled),
            "latest" => Some(Self::Latest),
            "prompt-end-and-latest" => Some(Self::PromptEndAndLatest),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CpuPrefillGdn {
    Sequential,
    WyF32,
}

impl CpuPrefillGdn {
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Sequential => "sequential",
            Self::WyF32 => "wy-f32",
        }
    }

    pub(crate) fn from_cli(value: &str) -> Option<Self> {
        match value {
            "sequential" => Some(Self::Sequential),
            "wy-f32" => Some(Self::WyF32),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SpeculationType {
    None,
    Suffix,
}

impl SpeculationType {
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Suffix => "suffix",
        }
    }

    pub(crate) fn from_cli(value: &str) -> Option<Self> {
        match value {
            "none" => Some(Self::None),
            "suffix" => Some(Self::Suffix),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CpuFamily {
    X1,
    X2,
}

impl CpuFamily {
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::X1 => "x1",
            Self::X2 => "x2",
        }
    }

    pub(crate) fn from_cli(value: &str) -> Option<Self> {
        match value {
            "x1" => Some(Self::X1),
            "x2" => Some(Self::X2),
            _ => None,
        }
    }

    pub(crate) const fn default_native_row_workers(self) -> u8 {
        match self {
            Self::X1 | Self::X2 => 8,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Bonsai25Variant {
    Pq2,
    Ptq1,
}

impl Bonsai25Variant {
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Pq2 => "pq2",
            Self::Ptq1 => "ptq1",
        }
    }

    pub(crate) fn from_cli(value: &str) -> Option<Self> {
        match value {
            "pq2" => Some(Self::Pq2),
            "ptq1" => Some(Self::Ptq1),
            _ => None,
        }
    }
}

/// Fixed development launch inputs; paths and keys have no debug representation.
#[derive(Clone)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "Independent explicit runtime, adapter, weight and transport choices survive retry."
)]
pub(crate) struct Config {
    pub(crate) broker: PathBuf,
    pub(crate) data: PathBuf,
    pub(crate) evidence: PathBuf,
    pub(crate) frontend_port: u16,
    pub(crate) backend_port: u16,
    pub(crate) require_broker_pipe: bool,
    pub(crate) use_abliterate_lora: bool,
    pub(crate) daisyrose_lora: Option<PathBuf>,
    pub(crate) no_daisyrose_lora: bool,
    pub(crate) no_mmproj: bool,
    pub(crate) use_repacked_ptq1: bool,
    pub(crate) bonsai25_variant: Option<Bonsai25Variant>,
    pub(crate) native_engine: bool,
    pub(crate) native_row_workers: u8,
    pub(crate) prompt_reuse: PromptReuse,
    pub(crate) prompt_reuse_checkpoints: Option<u8>,
    pub(crate) prompt_reuse_checkpoint_bytes: Option<usize>,
    pub(crate) kv_cache: Option<KvCache>,
    pub(crate) cpu_prefill_tokens: Option<u8>,
    pub(crate) cpu_prefill_gdn: Option<CpuPrefillGdn>,
    pub(crate) spec_type: Option<SpeculationType>,
    pub(crate) draft_max: Option<u8>,
    pub(crate) cpu_family: Option<CpuFamily>,
    pub(crate) placement: Option<Placement>,
    pub(crate) canary: Option<crate::canary::Config>,
    pub(crate) qwen_tts: Option<crate::tts::Config>,
}

/// Prompt reasoning effort levels supported by cognitive-control backends.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ReasoningEffort {
    Low,
    Medium,
    #[default]
    XHigh,
}

impl ReasoningEffort {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::XHigh => "xhigh",
        }
    }

    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Low => "Low (fast)",
            Self::Medium => "Medium",
            Self::XHigh => "Extra high (default)",
        }
    }
}

/// Soft thinking-token budget preset within `max_completion_tokens`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ThinkingBudgetMode {
    #[default]
    Unbounded,
    Concise,
    Balanced,
    Deep,
}

impl ThinkingBudgetMode {
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Unbounded => "Unbounded (default)",
            Self::Concise => "Concise (~35%)",
            Self::Balanced => "Balanced (~60%)",
            Self::Deep => "Deep (~80%)",
        }
    }

    pub(crate) fn compute_budget(
        self,
        max_completion_tokens: u32,
        enable_thinking: bool,
    ) -> Option<u32> {
        if !enable_thinking {
            return None;
        }
        let pct = match self {
            Self::Unbounded => return None,
            Self::Concise => 35_u64,
            Self::Balanced => 60_u64,
            Self::Deep => 80_u64,
        };
        let scaled = ((u64::from(max_completion_tokens) * pct) / 100).max(1);
        let ceiling = max_completion_tokens
            .saturating_sub(128)
            .max(1)
            .min(max_completion_tokens);
        u32::try_from(scaled)
            .ok()
            .map(|value| value.clamp(1, ceiling))
    }
}

/// Calibrated sampling & DRY anti-repetition preset.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum SamplingPreset {
    #[default]
    Canonical,
    Precise,
    BalancedDry,
    Creative,
}

impl SamplingPreset {
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Canonical => "Canonical default",
            Self::Precise => "Precise (T=0.6)",
            Self::BalancedDry => "Balanced + DRY",
            Self::Creative => "Creative + DRY",
        }
    }
}

/// Multi-call workflow used when Extra effort is enabled.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ExtraEffortStrategy {
    #[default]
    PlanDraftReview,
    MultiDraftSynthesis,
}

impl ExtraEffortStrategy {
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::PlanDraftReview => "Plan -> Draft -> Review",
            Self::MultiDraftSynthesis => "RSA Multi-Draft Synthesis",
        }
    }
}

/// Conversation-scoped cognitive and sampling options.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CognitiveOptions {
    #[serde(default)]
    pub(crate) reasoning_effort: ReasoningEffort,
    #[serde(default)]
    pub(crate) thinking_budget: ThinkingBudgetMode,
    #[serde(default)]
    pub(crate) sampling_preset: SamplingPreset,
    #[serde(default)]
    pub(crate) extra_effort_strategy: ExtraEffortStrategy,
}

/// Commands sent from the native UI to its single background worker.
pub(crate) enum Command {
    Send {
        text: String,
        images: Arc<[Arc<str>]>,
        max_completion_tokens: u32,
        extra_effort: bool,
        enable_thinking: bool,
        abliteration_scale: f32,
        daisyrose_scale: f32,
        cognitive: CognitiveOptions,
        system_appendix: Option<Arc<str>>,
        turn_appendix: Option<Arc<str>>,
    },
    NewChat,
    RestoreChat(Vec<Message>),
    Stop,
    StopModel,
    Retry,
    Shutdown,
}

/// Text-only notifications; no credentials or raw HTTP diagnostics cross this boundary.
#[derive(Clone)]
pub(crate) enum Event {
    Starting,
    Restarting,
    ContextCapacity(usize),
    VisionAvailable(bool),
    DaisyRoseAvailable(DaisyRoseReadiness),
    CognitiveControlsAvailable(bool),
    Ready,
    Text(String),
    Reasoning(String),
    Complete(Completion),
    Committed(Vec<Message>),
    Failed(String),
    Retryable(String),
    Stopping(String),
    Stopped,
    ModelStopped,
    Rejected {
        message: String,
        text: String,
        retry_required: bool,
    },
    Stage(&'static str),
    Draft(String),
    // Extra effort restores a previously completed draft as one lifecycle
    // operation. Its aggregate body retains the separate MAX_BODY bound.
    RetainedAnswer(String),
}

/// Why a successfully framed generation ended; a token cap can leave partial prose.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Completion {
    Finished,
    TokenLimit,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Failure {
    Shutdown,
    Stopped,
    ModelStopped,
    Memory,
    NativeRuntime,
    Teardown,
    CollapsedDraft,
    Rejected(&'static str),
    Message(&'static str),
}

type Result<T> = std::result::Result<T, Failure>;

fn images_empty(images: &Arc<[Arc<str>]>) -> bool {
    images.is_empty()
}

#[derive(Clone, Serialize)]
pub(crate) struct Message {
    pub(crate) role: &'static str,
    // Completed payloads are immutable. Pending requests, UI notifications and
    // restored conversations retain independent vectors but share these bytes.
    pub(crate) content: Arc<str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) reasoning_content: Option<Arc<str>>,
    #[serde(default, skip_serializing_if = "images_empty")]
    pub(crate) images: Arc<[Arc<str>]>,
}

impl<'de> Deserialize<'de> for Message {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct StoredMessage {
            role: String,
            content: String,
            reasoning_content: Option<String>,
            #[serde(default)]
            images: Vec<String>,
        }
        let stored = StoredMessage::deserialize(deserializer)?;
        let role = match stored.role.as_str() {
            "user" => "user",
            "assistant" => "assistant",
            _ => return Err(serde::de::Error::custom("Unsupported conversation role.")),
        };
        if role != "user" && !stored.images.is_empty() {
            return Err(serde::de::Error::custom(
                "Only user messages may carry attached images.",
            ));
        }
        if stored.images.len() > MAX_REQUEST_IMAGES {
            return Err(serde::de::Error::custom(
                "Conversation message exceeds maximum attached image count.",
            ));
        }
        let images: Vec<Arc<str>> = stored.images.into_iter().map(Arc::from).collect();
        Ok(Self {
            role,
            content: stored.content.into(),
            reasoning_content: stored.reasoning_content.map(Into::into),
            images: images.into(),
        })
    }
}

struct SystemMessageRef<'a>(Option<&'a str>);

impl Serialize for SystemMessageRef<'_> {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct as _;
        let mut state = serializer.serialize_struct("Message", 2)?;
        state.serialize_field("role", "system")?;
        if let Some(appendix) = self.0.filter(|text| !text.is_empty()) {
            let mut combined = String::with_capacity(SYSTEM_PROMPT.len() + 2 + appendix.len());
            combined.push_str(SYSTEM_PROMPT);
            combined.push_str("\n\n");
            combined.push_str(appendix);
            state.serialize_field("content", &combined)?;
        } else {
            state.serialize_field("content", SYSTEM_PROMPT)?;
        }
        state.end()
    }
}

#[derive(Serialize)]
struct WireImageUrl<'a> {
    url: &'a str,
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum WireContentPart<'a> {
    Text { text: &'a str },
    ImageUrl { image_url: WireImageUrl<'a> },
}

struct WireContent<'a> {
    message: &'a Message,
    turn_appendix: Option<&'a str>,
}

fn combined_turn_text<'a>(content: &'a str, turn_appendix: Option<&'a str>) -> Cow<'a, str> {
    let Some(appendix) = turn_appendix.filter(|text| !text.is_empty()) else {
        return Cow::Borrowed(content);
    };
    if content.is_empty() {
        Cow::Borrowed(appendix)
    } else {
        let mut combined = String::with_capacity(appendix.len() + 2 + content.len());
        combined.push_str(appendix);
        combined.push_str("\n\n");
        combined.push_str(content);
        Cow::Owned(combined)
    }
}

impl Serialize for WireContent<'_> {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        use serde::ser::SerializeSeq as _;
        let text = combined_turn_text(self.message.content.as_ref(), self.turn_appendix);
        if self.message.images.is_empty() {
            return serializer.serialize_str(text.as_ref());
        }
        let has_text = !text.is_empty();
        let len = self.message.images.len().saturating_add(usize::from(has_text));
        let mut seq = serializer.serialize_seq(Some(len))?;
        for url in self.message.images.iter() {
            seq.serialize_element(&WireContentPart::ImageUrl {
                image_url: WireImageUrl { url: url.as_ref() },
            })?;
        }
        if has_text {
            seq.serialize_element(&WireContentPart::Text {
                text: text.as_ref(),
            })?;
        }
        seq.end()
    }
}

#[derive(Serialize)]
struct WireMessage<'a> {
    role: &'a str,
    content: WireContent<'a>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<&'a str>,
}

struct RequestMessages<'a> {
    history: &'a [Message],
    system_appendix: Option<&'a str>,
    turn_appendix: Option<&'a str>,
}

impl Serialize for RequestMessages<'_> {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        use serde::ser::SerializeSeq as _;
        let mut seq = serializer.serialize_seq(Some(self.history.len().saturating_add(1)))?;
        seq.serialize_element(&SystemMessageRef(self.system_appendix))?;
        let last_user_index = self
            .history
            .len()
            .checked_sub(1)
            .filter(|&index| self.history[index].role == "user");
        for (index, message) in self.history.iter().enumerate() {
            let turn_appendix = (Some(index) == last_user_index)
                .then_some(self.turn_appendix)
                .flatten();
            seq.serialize_element(&WireMessage {
                role: message.role,
                content: WireContent {
                    message,
                    turn_appendix,
                },
                reasoning_content: message.reasoning_content.as_deref(),
            })?;
        }
        seq.end()
    }
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    messages: RequestMessages<'a>,
    max_completion_tokens: u32,
    stream: bool,
    enable_thinking: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    abliteration_scale: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    daisyrose_scale: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_budget: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    min_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    repeat_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dry_multiplier: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dry_base: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dry_allowed_length: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dry_penalty_last_n: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    logprobs: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_logprobs: Option<u8>,
}

#[derive(Clone, Default)]
struct ResponseOptions {
    max_completion_tokens: u32,
    enable_thinking: bool,
    abliteration_scale: Option<f32>,
    daisyrose_scale: Option<f32>,
    cognitive: CognitiveOptions,
    cognitive_controls: bool,
    reset_on_cancel: bool,
    system_appendix: Option<Arc<str>>,
    turn_appendix: Option<Arc<str>>,
    request_logprobs: bool,
    temperature_nudge: f32,
}

/// Launch the owned broker, then serve one in-memory conversation until shutdown.
#[expect(
    clippy::needless_pass_by_value,
    reason = "The background thread owns its configuration and channels until the broker is stopped"
)]
pub(crate) fn run(config: Config, commands: Receiver<Command>, events: EventSender) {
    // Only committed turns live here, outside an individual owned broker's
    // lifetime. Explicit retries preserve them without replaying a failed turn.
    let mut history = Vec::new();
    for attempt in 0..u32::MAX {
        let attempt_config = Config {
            evidence: config.evidence.join(format!("attempt-{attempt}")),
            ..config.clone()
        };
        if std::fs::create_dir(&attempt_config.evidence).is_err() {
            let _notification = events.send(Event::Failed(
                "Could not create a fresh model startup directory. Restart the app.".to_owned(),
            ));
            return;
        }
        if notify(&events, Event::Starting).is_err() {
            return;
        }
        match worker(&attempt_config, &commands, &events, &mut history) {
            Ok(()) | Err(Failure::Shutdown) => return,
            Err(Failure::Teardown) => {
                let message = attach_broker_log_evidence(
                    TEARDOWN_FAILURE_MESSAGE.to_owned(),
                    &attempt_config.evidence,
                );
                let _notification = events.send(Event::Failed(message));
                return;
            }
            Err(Failure::Memory) => {
                let guidance = attach_broker_log_evidence(
                    "Not enough free memory. Swift needs 15.5 GiB available. Free some RAM, then select Retry model.".to_owned(),
                    &attempt_config.evidence,
                );
                if !wait_for_retry(
                    &commands,
                    &events,
                    &mut history,
                    &Event::Retryable(guidance),
                ) {
                    return;
                }
            }
            Err(Failure::Stopped) => {
                // On the publisher rollback engine (without in-place `/reset`),
                // the owned Job has confirmed teardown after a response stop;
                // restart a fresh warm attempt automatically without surfacing
                // a model failure or waiting for manual Retry.
                if notify(&events, Event::Restarting).is_err() {
                    return;
                }
            }
            Err(Failure::ModelStopped) => {
                // Explicit model teardown (for example before installing an
                // application update) parks cleanly until Retry or Shutdown.
                if !wait_for_retry(&commands, &events, &mut history, &Event::ModelStopped) {
                    return;
                }
            }
            Err(failure) => {
                let message = match failure {
                    Failure::Message(message) | Failure::Rejected(message) => message.to_owned(),
                    Failure::CollapsedDraft => {
                        "Draft generation stopped early due to sustained low token confidence."
                            .to_owned()
                    }
                    Failure::NativeRuntime => native_runtime_guidance(&config.data),
                    Failure::Shutdown
                    | Failure::Memory
                    | Failure::Stopped
                    | Failure::ModelStopped
                    | Failure::Teardown => {
                        return;
                    }
                };
                // worker has confirmed its complete Job empty before Retry
                // can be offered, including on startup and transport failures.
                let guidance =
                    attach_broker_log_evidence(retry_guidance(&message), &attempt_config.evidence);
                if !wait_for_retry(
                    &commands,
                    &events,
                    &mut history,
                    &Event::Retryable(guidance),
                ) {
                    return;
                }
            }
        }
    }
}

fn retry_guidance(message: &str) -> String {
    if message == RUNNING_MEMORY_MESSAGE
        || message == PUBLISHER_INIT_FAILURE_MESSAGE
        || NON_RETRYABLE_STARTUP_MESSAGES.contains(&message)
        || NATIVE_IDENTITY_GUIDANCE
            .iter()
            .any(|(_, guidance)| *guidance == message)
    {
        return message.to_owned();
    }
    format!(
        "{message}\nSelect Retry model to restart the local model. Completed turns remain in memory."
    )
}

fn attach_broker_log_evidence(guidance: String, evidence_directory: &Path) -> String {
    let log = evidence_directory.join("broker.stderr.log");
    let has_nonempty_log = std::fs::symlink_metadata(&log).is_ok_and(|metadata| {
        metadata.is_file() && !metadata.file_type().is_symlink() && metadata.len() > 0
    });
    if has_nonempty_log {
        format!("{guidance}\n{EVIDENCE_PREFIX}{}", log.display())
    } else {
        guidance
    }
}

fn native_runtime_guidance(data: &Path) -> String {
    let lock: serde_json::Value =
        serde_json::from_str(include_str!("../../../config/prism-inference.lock.json"))
            .unwrap_or_default();
    let version = lock["version"].as_str().unwrap_or("pinned");
    format!(
        "The selected Rust runtime (Prism {version}) is missing or failed verification in {}. Close and reopen Swift with the same data folder to check and repair its download automatically. If preparation cannot finish, select Retry setup or Copy diagnostics. Copy any unsaved conversations or drafts before closing.",
        data.display(),
    )
}

fn wait_for_retry(
    commands: &Receiver<Command>,
    events: &EventSender,
    history: &mut Vec<Message>,
    notice: &Event,
) -> bool {
    if notify(events, notice.clone()).is_err() {
        return false;
    }
    loop {
        match commands.recv_timeout(POLL) {
            Ok(Command::Retry) => return true,
            Ok(Command::Shutdown) | Err(RecvTimeoutError::Disconnected) => {
                return false;
            }
            Ok(Command::Send { text, .. }) => {
                // An idle broker can exit just as the UI submits. Restore that
                // queued prompt instead of discarding it, and retain the retry
                // state without automatically replaying the request.
                if notify(events, Event::Rejected {
                    message: "The model is unavailable. Your message is restored; select Retry model before sending again.".to_owned(),
                    text,
                    retry_required: true,
                }).is_err() || notify(events, notice.clone()).is_err() {
                    return false;
                }
            }
            // The UI clears the visible conversation immediately. Honor a
            // reset queued just before idle process death so a fresh transcript
            // cannot silently retain the previous conversation after Retry.
            Ok(Command::NewChat) => history.clear(),
            Ok(Command::RestoreChat(restored)) => *history = restored,
            Ok(Command::Stop | Command::StopModel) | Err(RecvTimeoutError::Timeout) => {}
        }
    }
}

fn runtime_arguments(config: &Config) -> Vec<String> {
    let mut arguments = Vec::with_capacity(16);
    if config.use_abliterate_lora {
        arguments.push("--use-abliterate-lora".into());
    }
    if config.no_daisyrose_lora {
        arguments.push("--no-daisyrose-lora".into());
    }
    if let Some(path) = config.daisyrose_lora.as_deref().and_then(|p| p.to_str()) {
        arguments.extend(["--daisyrose-lora".into(), path.to_owned()]);
    }
    if config.no_mmproj {
        arguments.push("--no-mmproj".into());
    }
    if let Some(variant) = config.bonsai25_variant {
        arguments.extend(["--bonsai-2-5".into(), variant.name().into()]);
    } else {
        arguments.push(
            if config.use_repacked_ptq1 {
                "--use-repacked-ptq1"
            } else {
                "--use-original-pq2"
            }
            .into(),
        );
    }
    arguments.push(
        if config.native_engine {
            "--native-engine"
        } else {
            "--publisher-engine"
        }
        .into(),
    );
    if config.native_engine {
        arguments.push("--reset-on-cancel".into());
    }
    arguments.extend([
        "--native-row-workers".into(),
        match config.native_row_workers {
            4 => "4",
            8 => "8",
            12 => "12",
            _ => "1",
        }
        .into(),
    ]);
    if let Some(family) = config.cpu_family {
        arguments.extend(["--cpu-family".into(), family.name().into()]);
    }
    if config.prompt_reuse.enabled() {
        arguments.extend(["--prompt-reuse".into(), config.prompt_reuse.name().into()]);
    }
    if let Some(checkpoints) = config.prompt_reuse_checkpoints {
        arguments.extend(["--prompt-reuse-checkpoints".into(), checkpoints.to_string()]);
    }
    if let Some(bytes) = config.prompt_reuse_checkpoint_bytes {
        arguments.extend(["--prompt-reuse-checkpoint-bytes".into(), bytes.to_string()]);
    }
    if let Some(policy) = config.kv_cache {
        arguments.extend(["--kv-cache".into(), policy.name().into()]);
    }
    if let Some(tokens) = config.cpu_prefill_tokens {
        arguments.extend([
            "--cpu-prefill-tokens".into(),
            match tokens {
                8 => "8",
                16 => "16",
                32 => "32",
                _ => "8",
            }
            .into(),
        ]);
    }
    if let Some(gdn) = config.cpu_prefill_gdn {
        arguments.extend(["--cpu-prefill-gdn".into(), gdn.name().into()]);
    }
    if let Some(spec_type) = config.spec_type {
        arguments.extend(["--spec-type".into(), spec_type.name().into()]);
    }
    if let Some(draft_max) = config.draft_max {
        arguments.extend(["--draft-max".into(), draft_max.to_string()]);
    }
    if let Some(placement) = config.placement {
        arguments.extend(["--placement".into(), placement.name().into()]);
    }
    arguments
}

fn worker(
    config: &Config,
    commands: &Receiver<Command>,
    events: &EventSender,
    history: &mut Vec<Message>,
) -> Result<()> {
    let control = owner_control::OwnerControl::create(&config.evidence)
        .map_err(|_| Failure::Message("Could not create private model shutdown storage."))?;
    let spec = broker_spec(config, control.path());
    // Own the broker before its first instruction, including any descendants
    // created before the broker establishes its separate model sandbox/Job.
    // OwnedProcess keeps a kill-on-close guard on every early-return path.
    // A failed native creation can leave release unconfirmed. It cannot admit
    // another broker attempt merely because no process owner was returned.
    let broker = OwnedProcess::start_tool(&spec).map_err(|_| Failure::Teardown)?;
    let outcome = serve_broker(&broker, config, commands, events, history, &spec);
    let deadline = Instant::now() + BROKER_STOP;
    finish_owned_attempt(outcome, || control.stop(&broker, deadline))
}

fn broker_spec(config: &Config, shutdown_directory: &Path) -> ToolProcessSpec {
    let mut arguments: Vec<OsString> = runtime_arguments(config)
        .into_iter()
        .map(OsString::from)
        .collect();
    arguments.extend([
        "--launch-development".into(),
        "--require-native-sandbox".into(),
    ]);
    if config.require_broker_pipe {
        arguments.push("--require-broker-pipe".into());
    }
    arguments.extend([
        "--profile".into(),
        "swift-pq2-assessment".into(),
        "--data-directory".into(),
        config.data.as_os_str().to_owned(),
        "--evidence-parent".into(),
        config.evidence.as_os_str().to_owned(),
        "--frontend-port".into(),
        config.frontend_port.to_string().into(),
        "--backend-port".into(),
        config.backend_port.to_string().into(),
        "--max-waiting".into(),
        if config.native_engine { "1" } else { "0" }.into(),
        "--owner-shutdown-directory".into(),
        shutdown_directory.as_os_str().to_owned(),
    ]);
    ToolProcessSpec {
        executable: config.broker.clone(),
        arguments,
        working_directory: config.evidence.clone(),
        stdout: config.evidence.join("broker.stdout.log"),
        stderr: config.evidence.join("broker.stderr.log"),
        environment: BTreeMap::new(),
    }
}

fn finish_owned_attempt(outcome: Result<()>, stop: impl FnOnce() -> io::Result<()>) -> Result<()> {
    // A failed or timed-out native stop leaves whole-tree release unknown.
    // Closing the outer Job remains a broker-tree guard. Only a broker model-
    // cleanup acknowledgment, root exit and successful outer stop may admit
    // another model attempt; PSEC may broker its model outside the outer Job.
    stop().map_err(|_| Failure::Teardown)?;
    outcome
}

fn serve_broker(
    broker: &OwnedProcess,
    config: &Config,
    commands: &Receiver<Command>,
    events: &EventSender,
    history: &mut Vec<Message>,
    spec: &ToolProcessSpec,
) -> Result<()> {
    let (key, context_tokens, vision_available, daisyrose, cognitive_controls) = wait_ready(
        broker,
        config,
        commands,
        history,
        &spec.stdout,
        &spec.stderr,
    )?;
    notify(events, Event::ContextCapacity(context_tokens))?;
    notify(events, Event::VisionAvailable(vision_available))?;
    let daisyrose_available = daisyrose.available;
    notify(events, Event::DaisyRoseAvailable(daisyrose))?;
    notify(
        events,
        Event::CognitiveControlsAvailable(cognitive_controls),
    )?;
    notify(events, Event::Ready)?;
    loop {
        if broker
            .try_wait()
            .map_err(|_| Failure::Message("Could not inspect the owned broker."))?
            .is_some()
        {
            let diagnostic = read_bounded_file(&spec.stderr, MAX_HEAD).unwrap_or_default();
            return Err(runtime_failure(&diagnostic));
        }
        match commands.recv_timeout(POLL) {
            Ok(Command::Shutdown) | Err(RecvTimeoutError::Disconnected) => return Ok(()),
            Ok(Command::StopModel) => return Err(Failure::ModelStopped),
            // If completion or a recoverable rejection finished just ahead of a
            // user Stop click, acknowledge Stop without tearing down the warm model.
            Ok(Command::Stop) => notify(events, Event::Stopped)?,
            Ok(Command::NewChat) => history.clear(),
            Ok(Command::RestoreChat(restored)) => *history = restored,
            Ok(Command::Send {
                text,
                images,
                max_completion_tokens,
                extra_effort,
                enable_thinking,
                abliteration_scale,
                daisyrose_scale,
                cognitive,
                system_appendix,
                turn_appendix,
            }) => {
                if text.trim().is_empty() && images.is_empty() {
                    continue;
                }
                let normalized_text = if text.trim().is_empty() {
                    String::new()
                } else {
                    text
                };
                let mut pending = history.clone();
                pending.push(Message {
                    role: "user",
                    content: normalized_text.as_str().into(),
                    reasoning_content: None,
                    images,
                });
                let scale = config.use_abliterate_lora.then_some(abliteration_scale);
                let daisyrose_scale = daisyrose_available.then_some(daisyrose_scale);
                let options = ResponseOptions {
                    max_completion_tokens,
                    enable_thinking,
                    abliteration_scale: scale,
                    daisyrose_scale,
                    cognitive,
                    cognitive_controls,
                    reset_on_cancel: config.native_engine,
                    system_appendix,
                    turn_appendix,
                    request_logprobs: false,
                    temperature_nudge: 0.0,
                };
                let outcome =
                    serialize_request_with_options(&pending, &options).and_then(|request| {
                        if extra_effort {
                            // Validate the original request locally; effort
                            // constructs its own stage bodies.
                            drop(request);
                            effort::run(
                                config.frontend_port,
                                key.as_hex(),
                                &pending,
                                &options,
                                commands,
                                events,
                            )
                        } else {
                            generate(
                                config.frontend_port,
                                key.as_hex(),
                                request,
                                commands,
                                events,
                            )
                        }
                    });
                finish_turn(
                    history,
                    pending,
                    &normalized_text,
                    outcome,
                    events,
                    commands,
                    config.native_engine,
                )?;
            }
            Ok(Command::Retry) | Err(RecvTimeoutError::Timeout) => {}
        }
    }
}

fn wait_ready(
    broker: &OwnedProcess,
    config: &Config,
    commands: &Receiver<Command>,
    history: &mut Vec<Message>,
    stdout_path: &Path,
    stderr_path: &Path,
) -> Result<(RetainedCredential, usize, bool, DaisyRoseReadiness, bool)> {
    let until = Instant::now() + STARTUP_TIMEOUT;
    loop {
        check_startup_commands(commands, history)?;
        let output = read_bounded_file(stdout_path, MAX_HEAD)?;
        if let Some(directory) = evidence_directory(&output)? {
            let context_tokens = context_capacity(&output)?.ok_or(Failure::Message(
                "The broker did not report its selected context capacity. Rebuild the broker and retry.",
            ))?;
            let vision_available = input_modalities(&output)?.unwrap_or(false);
            let daisyrose = daisyrose_adapter(&output)?.unwrap_or_else(|| DaisyRoseReadiness {
                available: config.daisyrose_lora.is_some(),
                ..DaisyRoseReadiness::default()
            });
            let cognitive_controls = cognitive_controls_declaration(&output)?.unwrap_or(false);
            let key = read_api_key(&config.evidence, &directory)?;
            return Ok((
                key,
                context_tokens,
                vision_available,
                daisyrose,
                cognitive_controls,
            ));
        }
        if broker
            .try_wait()
            .map_err(|_| Failure::Message("Could not inspect broker startup."))?
            .is_some()
        {
            let diagnostic = read_bounded_file(stderr_path, MAX_HEAD).unwrap_or_default();
            return Err(startup_failure(&diagnostic, config.native_engine));
        }
        if Instant::now() >= until {
            return Err(Failure::Message(
                "The local model did not become ready within ten minutes.",
            ));
        }
        thread::sleep(POLL);
    }
}

fn read_api_key(parent: &Path, reported: &Path) -> Result<RetainedCredential> {
    let parent = parent
        .canonicalize()
        .map_err(|_| Failure::Message("The app evidence directory is unavailable."))?;
    let directory = reported
        .canonicalize()
        .map_err(|_| Failure::Message("The broker credential directory is unavailable."))?;
    if directory == parent || !directory.starts_with(&parent) {
        return Err(Failure::Message(
            "The broker reported an unexpected credential location.",
        ));
    }
    // Resolve canonical short/relative segments while stripping the Windows `\\?\`
    // drive verbatim prefix so `RetainedCredential::open` admits valid local drives.
    let key_path = strip_verbatim_drive(&directory).join("frontend.key");
    RetainedCredential::open(&key_path).map_err(|error| {
        if matches!(
            error.kind(),
            io::ErrorKind::InvalidData | io::ErrorKind::InvalidInput
        ) {
            Failure::Message("The broker frontend credential has an invalid format.")
        } else {
            Failure::Message("A broker diagnostic or credential file is unavailable.")
        }
    })
}

fn strip_verbatim_drive(path: &Path) -> PathBuf {
    let Some(text) = path.to_str() else {
        return path.to_owned();
    };
    if let Some(stripped) = text.strip_prefix(r"\\?\")
        && matches!(
            stripped.as_bytes(),
            [drive, b':', b'\\', ..] if drive.is_ascii_alphabetic()
        )
    {
        return PathBuf::from(stripped);
    }
    path.to_owned()
}

fn model_verification_message(diagnostic: &str) -> Option<&'static str> {
    if !diagnostic
        .lines()
        .any(|line| line == "Development backend launch failed: Artifacts.")
    {
        return None;
    }
    diagnostic.lines().find_map(|line| match line {
        "Selected model verification failed: reason=missing, representation=original." => Some(
            "The selected Swift model file is missing from this app's data directory. Close and reopen Swift to download it automatically. If setup needs another attempt, select Retry setup.",
        ),
        "Selected model verification failed: reason=storage, representation=original." => Some(
            "Swift cannot safely read the selected model file. Close any program moving or writing it, and check that the app's data directory is an accessible local folder without links or redirects. Close and reopen Swift to retry setup.",
        ),
        "Selected model verification failed: reason=identity, representation=original." => Some(
            "The selected Swift model did not pass its size and integrity checks. Close and reopen Swift so setup can download a verified replacement. Your saved conversations are kept.",
        ),
        "Selected model verification failed: reason=missing, representation=ptq1." => Some(
            "The explicitly selected PTQ1 model export is missing from the app's data directory. Restore the verified current-weight PTQ1 export, or close and relaunch Swift with the original model selection. Setup will not substitute a different model.",
        ),
        "Selected model verification failed: reason=storage, representation=ptq1." => Some(
            "Swift cannot safely read the explicitly selected PTQ1 model export. Close any program moving or writing it, and check the data directory's permissions and local storage path. Reopen Swift to check the same selected export again.",
        ),
        "Selected model verification failed: reason=identity, representation=ptq1." => Some(
            "The explicitly selected PTQ1 export did not pass its size and integrity checks. Restore the verified current-weight PTQ1 export, or close and relaunch Swift with the original model selection. Setup will not replace it with another model.",
        ),
        "Selected model verification failed: reason=configuration, representation=original."
        | "Selected model verification failed: reason=configuration, representation=ptq1." => Some(
            "This app could not apply its selected model configuration. Reopen the complete official application download and retry setup. Your model and conversation data remain separate from the executable.",
        ),
        "Selected model verification failed: reason=cancelled, representation=original."
        | "Selected model verification failed: reason=cancelled, representation=ptq1." => Some(
            "Model verification was cancelled. Close and reopen Swift when you want to continue setup.",
        ),
        _ => None,
    })
}

fn native_identity_message(diagnostic: &str) -> Option<&'static str> {
    if !diagnostic
        .lines()
        .any(|line| line == "Development backend launch failed: Backend.")
    {
        return None;
    }
    diagnostic.lines().find_map(|line| {
        NATIVE_IDENTITY_GUIDANCE
            .iter()
            .find_map(|(marker, guidance)| (*marker == line).then_some(*guidance))
    })
}

fn startup_failure(diagnostic: &[u8], native_engine: bool) -> Failure {
    let text = String::from_utf8_lossy(diagnostic);
    if let Some(message) = model_verification_message(&text) {
        return Failure::Message(message);
    }
    if native_engine && let Some(message) = native_identity_message(&text) {
        return Failure::Message(message);
    }
    if native_engine
        && text.lines().any(|line| {
            line == "Development backend launch failed: NativeRuntime."
                || line == "Development backend launch failed: NativeRuntime. Provision the pinned Rust runtime with bonsai-tools provision, operation inference, using the same dataDirectory."
        })
    {
        Failure::NativeRuntime
    } else if !native_engine
        && text.lines().any(|line| {
            line == "Native backend startup failure: stage=authenticated-readiness, error=Backend, process=exited-0xC0000142"
        })
    {
        Failure::Message(PUBLISHER_INIT_FAILURE_MESSAGE)
    } else if text.contains("SandboxEnvironment") {
        Failure::Message(
            "The model sandbox rejected inherited service credentials. Restart the app from a terminal or app session without service credentials in its parent environment.",
        )
    } else if text.contains("SandboxUnavailable") {
        if text.contains("Windows Update restart is pending") {
            Failure::Message(SANDBOX_REBOOT_PENDING_MESSAGE)
        } else if native_engine && !text.contains("container-hosted loopback") {
            Failure::Message(SANDBOX_UNAVAILABLE_OFFLINE_MESSAGE)
        } else {
            Failure::Message(SANDBOX_UNAVAILABLE_LOOPBACK_MESSAGE)
        }
    } else if text.contains("Sandbox") {
        Failure::Message(
            "The native model sandbox could not apply its required policy. Check the retained broker diagnostics, then retry the model.",
        )
    } else if text.contains("Memory") {
        Failure::Memory
    } else if text.contains("Port") || text.contains("Service") {
        Failure::Message(
            "A local chat port is unavailable. Close any other instance of this app before retrying the model.",
        )
    } else if text.contains("Adapter") {
        Failure::Message(
            "The model's required adapter is missing or failed verification. Close and reopen Swift to check and repair its download automatically. If setup needs another attempt, select Retry setup.",
        )
    } else if native_engine && text.contains("Artifacts") {
        Failure::Message(
            "The selected Swift model could not be verified. Close and reopen Swift to check its downloads and repair setup automatically. If setup fails, select Retry setup or copy its diagnostics. An explicitly selected PTQ1 export must be prepared separately.",
        )
    } else if text.contains("Artifacts") {
        Failure::Message(
            "The selected model or publisher CPU runtime could not be verified. Close and reopen Swift to check the selected downloads again. If setup fails, select Retry setup or copy its diagnostics.",
        )
    } else if text
        .lines()
        .any(|line| line == "Development backend launch failed: Policy.")
    {
        Failure::Message(POLICY_FAILURE_MESSAGE)
    } else if text.lines().any(|line| {
        line == "Development backend launch failed: Path."
            || line == "Development backend launch failed: Credentials."
    }) {
        Failure::Message(PATH_OR_CREDENTIALS_FAILURE_MESSAGE)
    } else if text
        .lines()
        .any(|line| line == "Development backend launch failed: Arguments.")
    {
        Failure::Message(ARGUMENTS_FAILURE_MESSAGE)
    } else {
        Failure::Message("Swift could not start. Check the retained broker diagnostic logs.")
    }
}

fn runtime_failure(diagnostic: &[u8]) -> Failure {
    if String::from_utf8_lossy(diagnostic)
        .contains("Owned backend stopped after launcher failure: Memory.")
    {
        Failure::Message(RUNNING_MEMORY_MESSAGE)
    } else {
        Failure::Message("The model session ended. Select Retry model to continue.")
    }
}

fn read_bounded_file(path: &Path, limit: usize) -> Result<Vec<u8>> {
    let file = File::open(path)
        .map_err(|_| Failure::Message("A broker diagnostic or credential file is unavailable."))?;
    crate::bounded_file::read(
        file,
        u64::try_from(limit).map_err(|_| Failure::Message("The diagnostic limit is invalid."))?,
    )
    .map_err(|_| Failure::Message("Could not read a broker diagnostic or credential file."))?
    .ok_or(Failure::Message(
        "A broker diagnostic or credential file exceeded its limit.",
    ))
}

fn complete_startup_lines(output: &[u8]) -> Result<Option<&str>> {
    let Some(end) = output.iter().rposition(|byte| *byte == b'\n') else {
        return Ok(None);
    };
    let text = std::str::from_utf8(&output[..=end])
        .map_err(|_| Failure::Message("Broker startup output was not valid UTF-8."))?;
    Ok(Some(text))
}

fn parse_unique_complete_line<T>(
    output: &[u8],
    prefix: &str,
    duplicate_error: &'static str,
    mut parse: impl FnMut(&str) -> Result<T>,
) -> Result<Option<T>> {
    let Some(text) = complete_startup_lines(output)? else {
        return Ok(None);
    };
    let mut observed = None;
    for line in text.lines() {
        if let Some(value) = line.strip_prefix(prefix) {
            let parsed = parse(value)?;
            if observed.replace(parsed).is_some() {
                return Err(Failure::Message(duplicate_error));
            }
        }
    }
    Ok(observed)
}

fn context_capacity(output: &[u8]) -> Result<Option<usize>> {
    parse_unique_complete_line(
        output,
        "Context tokens: ",
        "The broker reported more than one context capacity.",
        |value| {
            value
                .parse::<usize>()
                .ok()
                .filter(|tokens| (1..=crate::context::MAX_CONTEXT_TOKENS).contains(tokens))
                .filter(|_| !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit()))
                .ok_or(Failure::Message(
                    "The broker reported an invalid context capacity.",
                ))
        },
    )
}

fn input_modalities(output: &[u8]) -> Result<Option<bool>> {
    parse_unique_complete_line(
        output,
        "Input modalities: ",
        "The broker reported input modalities more than once.",
        |value| match value {
            "text, image" => Ok(true),
            "text" => Ok(false),
            _ => Err(Failure::Message(
                "The broker reported unsupported input modalities.",
            )),
        },
    )
}

fn daisyrose_adapter(output: &[u8]) -> Result<Option<DaisyRoseReadiness>> {
    parse_unique_complete_line(
        output,
        "DaisyRose adapter: ",
        "The broker reported DaisyRose adapter status more than once.",
        |value| {
            if value == "unavailable" {
                return Ok(DaisyRoseReadiness::default());
            }
            if let Some(rest) = value.strip_prefix("available, default scale ") {
                let mut parts = rest.split(", ");
                if let Some(scale) = parts.next()
                    && let [whole, b'.', frac] = scale.as_bytes()
                    && (b'0'..=b'2').contains(whole)
                    && frac.is_ascii_digit()
                {
                    let tenths = (*whole - b'0') * 10 + (*frac - b'0');
                    if tenths <= 20 {
                        let mut max_rank = None;
                        let mut sha256_prefix = None;
                        for part in parts {
                            if let Some(rank_str) = part.strip_prefix("rank ")
                                && max_rank.is_none()
                                && let Ok(rank) = rank_str.parse::<u16>()
                                && (1..=64).contains(&rank)
                            {
                                max_rank = Some(rank);
                            } else if let Some(sha_str) = part.strip_prefix("sha256 ")
                                && sha256_prefix.is_none()
                                && sha_str.len() <= 64
                                && let Some(prefix) = valid_daisyrose_sha256_prefix(sha_str)
                            {
                                sha256_prefix = Some(prefix);
                            } else {
                                return Err(Failure::Message(
                                    "The broker reported an unsupported DaisyRose adapter status.",
                                ));
                            }
                        }
                        return Ok(DaisyRoseReadiness {
                            available: true,
                            default_scale_tenths: tenths,
                            max_rank,
                            sha256_prefix,
                        });
                    }
                }
            }
            Err(Failure::Message(
                "The broker reported an unsupported DaisyRose adapter status.",
            ))
        },
    )
}

fn cognitive_controls_declaration(output: &[u8]) -> Result<Option<bool>> {
    parse_unique_complete_line(
        output,
        "Cognitive controls: ",
        "The broker reported cognitive controls more than once.",
        |value| match value {
            "available" => Ok(true),
            "unavailable" => Ok(false),
            _ => Err(Failure::Message(
                "The broker reported an unsupported cognitive controls status.",
            )),
        },
    )
}

fn evidence_directory(output: &[u8]) -> Result<Option<PathBuf>> {
    let Some(text) = complete_startup_lines(output)? else {
        return Ok(None);
    };
    // Only complete lines are considered: a poll may observe a partial write.
    for line in text
        .split_inclusive('\n')
        .filter(|line| line.ends_with('\n'))
    {
        if let Some(path) = line
            .trim_end_matches(['\r', '\n'])
            .strip_prefix(EVIDENCE_PREFIX)
        {
            let path = PathBuf::from(path);
            if !path.is_absolute() {
                return Err(Failure::Message(
                    "The broker reported a relative credential location.",
                ));
            }
            return Ok(Some(path));
        }
    }
    Ok(None)
}

fn finish_turn(
    history: &mut Vec<Message>,
    mut pending: Vec<Message>,
    text: &str,
    outcome: Result<(Message, Completion)>,
    events: &EventSender,
    commands: &Receiver<Command>,
    in_place_cancel: bool,
) -> Result<()> {
    // Buffered parsing and local validation may finish without touching the
    // socket again. Stop takes precedence until this commit boundary.
    match check_commands(commands).and(outcome) {
        Ok((answer, completion)) => {
            pending.push(answer);
            *history = pending;
            notify(events, Event::Committed(history.clone()))?;
            notify(events, Event::Complete(completion))
        }
        Err(Failure::Shutdown) => Err(Failure::Shutdown),
        Err(Failure::ModelStopped) => {
            notify(events, Event::Stopping(text.to_owned()))?;
            Err(Failure::ModelStopped)
        }
        Err(Failure::Stopped) => {
            notify(events, Event::Stopping(text.to_owned()))?;
            if in_place_cancel {
                notify(events, Event::Stopped)
            } else {
                Err(Failure::Stopped)
            }
        }
        Err(failure) => {
            let message = match failure {
                Failure::Rejected(message) => message,
                Failure::CollapsedDraft => {
                    "Draft generation stopped early due to sustained low token confidence."
                }
                Failure::Message(_)
                | Failure::Memory
                | Failure::NativeRuntime
                | Failure::Teardown => {
                    "The response did not complete. Review the conversation and retry the model."
                }
                Failure::Shutdown | Failure::Stopped | Failure::ModelStopped => {
                    return Err(failure);
                }
            };
            notify(
                events,
                Event::Rejected {
                    message: message.to_owned(),
                    text: text.to_owned(),
                    // The UI must not enable Send during owned-process
                    // teardown; Retryable requires a confirmed whole-tree stop.
                    retry_required: !matches!(failure, Failure::Rejected(_)),
                },
            )?;
            if matches!(failure, Failure::Rejected(_)) {
                Ok(())
            } else {
                Err(failure)
            }
        }
    }
}

#[cfg(test)]
fn serialize_request(
    history: &[Message],
    max_completion_tokens: u32,
    enable_thinking: bool,
    abliteration_scale: Option<f32>,
) -> Result<Vec<u8>> {
    serialize_request_with_adapters(
        history,
        max_completion_tokens,
        enable_thinking,
        abliteration_scale,
        None,
    )
}

#[cfg(test)]
fn serialize_request_with_adapters(
    history: &[Message],
    max_completion_tokens: u32,
    enable_thinking: bool,
    abliteration_scale: Option<f32>,
    daisyrose_scale: Option<f32>,
) -> Result<Vec<u8>> {
    let options = ResponseOptions {
        max_completion_tokens,
        enable_thinking,
        abliteration_scale,
        daisyrose_scale,
        ..ResponseOptions::default()
    };
    serialize_request_with_options(history, &options)
}

#[derive(Default)]
struct WireCognitiveFields {
    reasoning_effort: Option<&'static str>,
    reasoning_budget: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
    min_p: Option<f32>,
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    repeat_penalty: Option<f32>,
    dry_multiplier: Option<f32>,
    dry_base: Option<f32>,
    dry_allowed_length: Option<u32>,
    dry_penalty_last_n: Option<u32>,
    logprobs: Option<bool>,
    top_logprobs: Option<u8>,
}

impl WireCognitiveFields {
    fn from_options(options: &ResponseOptions) -> Self {
        if !options.cognitive_controls {
            return Self::default();
        }
        let mut fields = Self::default();
        if options.cognitive.reasoning_effort != ReasoningEffort::XHigh {
            fields.reasoning_effort = Some(options.cognitive.reasoning_effort.as_str());
        }
        fields.reasoning_budget = options
            .cognitive
            .thinking_budget
            .compute_budget(options.max_completion_tokens, options.enable_thinking);
        let base_temp = match options.cognitive.sampling_preset {
            SamplingPreset::Canonical => None,
            SamplingPreset::Precise => {
                fields.top_p = Some(0.90);
                fields.top_k = Some(40);
                fields.min_p = Some(0.05);
                Some(0.6)
            }
            SamplingPreset::BalancedDry => {
                fields.top_p = Some(0.95);
                fields.top_k = Some(40);
                fields.min_p = Some(0.05);
                fields.presence_penalty = Some(0.15);
                fields.dry_multiplier = Some(0.8);
                fields.dry_base = Some(1.75);
                fields.dry_allowed_length = Some(2);
                fields.dry_penalty_last_n = Some(1024);
                Some(0.85)
            }
            SamplingPreset::Creative => {
                fields.top_p = Some(0.95);
                fields.top_k = Some(64);
                fields.min_p = Some(0.04);
                fields.presence_penalty = Some(0.25);
                fields.dry_multiplier = Some(0.8);
                fields.dry_base = Some(1.75);
                fields.dry_allowed_length = Some(2);
                fields.dry_penalty_last_n = Some(1024);
                Some(1.0)
            }
        };
        fields.temperature = match (base_temp, options.temperature_nudge > 0.0) {
            (Some(temp), true) => Some((temp + options.temperature_nudge).clamp(0.05, 1.15)),
            (None, true) => Some((1.0 + options.temperature_nudge).clamp(0.05, 1.15)),
            (Some(temp), false) => Some(temp),
            (None, false) => None,
        };
        if options.request_logprobs {
            fields.logprobs = Some(true);
            fields.top_logprobs = Some(5);
        }
        fields
    }
}

fn serialize_request_with_options(
    history: &[Message],
    options: &ResponseOptions,
) -> Result<Vec<u8>> {
    if options
        .abliteration_scale
        .is_some_and(|scale| !valid_adapter_scale(scale))
    {
        return Err(Failure::Rejected(
            "Choose an abliteration scale between 0 and 2.",
        ));
    }
    if options
        .daisyrose_scale
        .is_some_and(|scale| !valid_adapter_scale(scale))
    {
        return Err(Failure::Rejected(
            "Choose a DaisyRose scale between 0 and 2.",
        ));
    }
    if !(1..=4096).contains(&options.max_completion_tokens) {
        return Err(Failure::Rejected(
            "Choose an output budget between 1 and 4,096 tokens.",
        ));
    }
    if history.len() > 127 {
        return Err(Failure::Rejected(
            "This conversation reached the 127-message limit. Select New conversation before sending again.",
        ));
    }
    let total_images = history
        .iter()
        .map(|message| message.images.len())
        .fold(0_usize, usize::saturating_add);
    if total_images > MAX_REQUEST_IMAGES {
        return Err(Failure::Rejected(
            "This conversation reached the 24-image limit, including video keyframes. Remove media attachments or select New conversation before sending again.",
        ));
    }
    let limit = if total_images > 0 {
        MAX_MULTIMODAL_REQUEST
    } else {
        MAX_REQUEST
    };
    let cognitive = WireCognitiveFields::from_options(options);
    crate::bounded_json::to_vec(
        &ChatRequest {
            messages: RequestMessages {
                history,
                system_appendix: options.system_appendix.as_deref(),
                turn_appendix: options.turn_appendix.as_deref(),
            },
            max_completion_tokens: options.max_completion_tokens,
            stream: true,
            enable_thinking: options.enable_thinking,
            abliteration_scale: options.abliteration_scale,
            daisyrose_scale: options.daisyrose_scale,
            reasoning_effort: cognitive.reasoning_effort,
            reasoning_budget: cognitive.reasoning_budget,
            temperature: cognitive.temperature,
            top_p: cognitive.top_p,
            top_k: cognitive.top_k,
            min_p: cognitive.min_p,
            presence_penalty: cognitive.presence_penalty,
            frequency_penalty: cognitive.frequency_penalty,
            repeat_penalty: cognitive.repeat_penalty,
            dry_multiplier: cognitive.dry_multiplier,
            dry_base: cognitive.dry_base,
            dry_allowed_length: cognitive.dry_allowed_length,
            dry_penalty_last_n: cognitive.dry_penalty_last_n,
            logprobs: cognitive.logprobs,
            top_logprobs: cognitive.top_logprobs,
        },
        limit,
    )
    .map_err(|error| {
        if error.io_error_kind() == Some(io::ErrorKind::FileTooLarge) {
            if total_images > 0 {
                multimodal_request_byte_limit()
            } else {
                request_byte_limit()
            }
        } else {
            Failure::Message("Could not prepare the chat request.")
        }
    })
}

const fn request_byte_limit() -> Failure {
    Failure::Rejected(
        "This conversation exceeded the 1 MiB request limit, including history and text escaping. Shorten your message or select New conversation. Your draft is kept.",
    )
}

const fn multimodal_request_byte_limit() -> Failure {
    Failure::Rejected(
        "This conversation exceeded the 16 MiB multimodal request limit, including attached images and history. Remove some media attachments or select New conversation. Your draft is kept.",
    )
}

fn notify(events: &EventSender, event: Event) -> Result<()> {
    events.send(event).map_err(|()| Failure::Shutdown)
}

fn check_startup_commands(commands: &Receiver<Command>, history: &mut Vec<Message>) -> Result<()> {
    loop {
        match commands.try_recv() {
            Ok(Command::Shutdown) | Err(TryRecvError::Disconnected) => {
                return Err(Failure::Shutdown);
            }
            Ok(Command::StopModel) => return Err(Failure::ModelStopped),
            Ok(Command::NewChat) => history.clear(),
            Ok(Command::RestoreChat(restored)) => *history = restored,
            Ok(Command::Stop | Command::Send { .. } | Command::Retry) => {}
            Err(TryRecvError::Empty) => return Ok(()),
        }
    }
}

fn check_commands(commands: &Receiver<Command>) -> Result<()> {
    loop {
        match commands.try_recv() {
            Ok(Command::Shutdown) | Err(TryRecvError::Disconnected) => {
                return Err(Failure::Shutdown);
            }
            Ok(Command::StopModel) => return Err(Failure::ModelStopped),
            Ok(Command::Stop) => return Err(Failure::Stopped),
            Ok(
                Command::Send { .. } | Command::NewChat | Command::RestoreChat(_) | Command::Retry,
            ) => {}
            Err(TryRecvError::Empty) => return Ok(()),
        }
    }
}

// Generation has no elapsed-time cutoff. Every socket operation still observes
// the same Stop/Shutdown channel, including while no response bytes arrive.
const fn map_http_error(error: crate::http::Error<Failure>) -> Failure {
    match error {
        crate::http::Error::Interrupted(failure) => failure,
        crate::http::Error::Transport(transport) => map_transport(transport),
    }
}

const fn map_transport(error: crate::http::TransportError) -> Failure {
    use crate::http::TransportError;
    Failure::Message(match error {
        TransportError::Connect => "Could not connect to the model service on this computer.",
        TransportError::Configure => "Could not configure the local connection.",
        TransportError::UploadSize
        | TransportError::UploadClosed
        | TransportError::UploadInterrupted => {
            "Could not send your message to the model on this computer."
        }
        TransportError::HeadOverflow => "The local response header exceeded its limit.",
        TransportError::InvalidHead => {
            "The model on this computer returned a reply the app could not read."
        }
        TransportError::DuplicateHeader => "The local response repeated a framing header.",
        TransportError::UnsupportedEncoding => "Encoded local responses are unsupported.",
        TransportError::UnsupportedMedia => {
            "The local model returned a reply format the app could not read."
        }
        TransportError::InvalidLength => "The response length is invalid or exceeded its limit.",
        TransportError::UnsupportedFraming
        | TransportError::LengthExceeded
        | TransportError::TrailingBytes => {
            "The local response had unsupported or conflicting framing."
        }
        TransportError::ResponseClosed => LOCAL_CONNECTION_CLOSED,
        TransportError::ResponseInterrupted => {
            "The connection to the model on this computer was interrupted before a complete reply was received."
        }
        TransportError::LineOverflow => {
            "The local model returned reply metadata larger than the app can accept."
        }
        TransportError::InvalidChunkEnding => "The local response had invalid chunk endings.",
        TransportError::InvalidChunkFraming => "The local response had invalid chunk framing.",
        TransportError::InvalidChunkSize => "The local chunk size was invalid.",
        TransportError::BodyLimitExceeded | TransportError::Allocation => {
            "The response exceeded its four-megabyte limit."
        }
        TransportError::UnsupportedTrailers => "Response trailers are unsupported.",
        TransportError::Status(_) | TransportError::UnexpectedStatus => {
            "The local model rejected the request without reporting a recognized reason."
        }
    })
}

fn connect_interruptible(
    address: SocketAddr,
    connect_timeout: Duration,
    commands: &Receiver<Command>,
) -> Result<TcpStream> {
    crate::http::connect_socket(
        address,
        connect_timeout,
        || IO_POLL,
        || check_commands(commands),
    )
    .map_err(map_http_error)
}

fn generate(
    port: u16,
    key: &str,
    request: Vec<u8>,
    commands: &Receiver<Command>,
    events: &EventSender,
) -> Result<(Message, Completion)> {
    let (message, completion, _) =
        generate_with_options(port, key, request, false, commands, events)?;
    Ok((message, completion))
}

fn generate_with_options(
    port: u16,
    key: &str,
    request: Vec<u8>,
    stop_on_collapse: bool,
    commands: &Receiver<Command>,
    events: &EventSender,
) -> Result<(Message, Completion, Option<sse::ConfidenceSummary>)> {
    check_commands(commands)?;
    let address = SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::LOCALHOST, port));
    let connect_timeout = Duration::from_secs(1);
    let stream = connect_interruptible(address, connect_timeout, commands)?;
    check_commands(commands)?;
    let _nodelay = stream.set_nodelay(true);
    stream
        .set_nonblocking(false)
        .and_then(|()| stream.set_read_timeout(Some(SOCKET_POLL)))
        .and_then(|()| stream.set_write_timeout(Some(SOCKET_POLL)))
        .map_err(|_| Failure::Message("Could not configure the local connection."))?;
    let mut connection = crate::http::PollStream::new(stream, |phase| match phase {
        crate::http::PollPhase::Check | crate::http::PollPhase::Wait => check_commands(commands),
    });
    let outcome = exchange(
        &mut connection,
        port,
        key,
        request,
        stop_on_collapse,
        events,
        commands,
    );
    let _checked = connection.check();
    connection.take_interrupted().map_or(outcome, Err)
}

fn exchange(
    connection: &mut (impl Read + Write),
    port: u16,
    key: &str,
    request: impl AsRef<[u8]>,
    stop_on_collapse: bool,
    events: &EventSender,
    commands: &Receiver<Command>,
) -> Result<(Message, Completion, Option<sse::ConfidenceSummary>)> {
    {
        let head = crate::http::bearer_request_head(
            "POST",
            "/v1/chat/completions",
            port,
            key,
            Some("application/json"),
            false,
            request.as_ref().len(),
        );
        // Keep the sending half open: the broker interprets half-close as abandonment.
        let mut slices = [IoSlice::new(&head), IoSlice::new(request.as_ref())];
        crate::http::write_all_vectored(
            connection,
            &mut slices,
            || check_commands(commands),
            || check_commands(commands),
        )
        .map_err(map_http_error)?;
        connection.flush().map_err(|_| {
            Failure::Message("Could not send your message to the model on this computer.")
        })?;
    }
    // The canonical request is no longer needed once sent. Release its
    // allocation before response headers or streaming backpressure can wait.
    drop(request);
    let mut body = HttpBody::new(connection)?;
    let mut sse = Sse::default();
    while let Some(bytes) = body.next_payload()? {
        sse.push(bytes, &mut |text, reasoning| {
            if !text.is_empty() {
                events.stream(Event::Text(text.into_owned()), commands)?;
            }
            if !reasoning.is_empty() {
                events.stream(Event::Reasoning(reasoning.into_owned()), commands)?;
            }
            Ok(())
        })?;
        if stop_on_collapse && sse.confidence_collapsed() {
            return Err(Failure::CollapsedDraft);
        }
    }
    check_commands(commands)?;
    let completion = sse.finish()?;
    let summary = sse.confidence_summary();
    let (text, reasoning) = sse.into_answer();
    Ok((
        Message {
            role: "assistant",
            content: text.into(),
            reasoning_content: (!reasoning.is_empty()).then(|| reasoning.into()),
            images: Arc::default(),
        },
        completion,
        summary,
    ))
}

#[derive(Clone, Copy)]
enum ResponseKind {
    Stream,
    ContextError,
    RequestError,
    RateError,
    BackendError,
}

impl ResponseKind {
    fn failure(self, code: &str) -> Option<Failure> {
        match (self, code) {
            (Self::ContextError, "context_length_exceeded") => {
                Some(Failure::Rejected(CONTEXT_LIMIT_MESSAGE))
            }
            (Self::ContextError, "request_body_too_large") => {
                Some(Failure::Rejected(REQUEST_LIMIT_MESSAGE))
            }
            (Self::RequestError, "input_control_tokens_rejected") => {
                Some(Failure::Rejected(CONTROL_TOKEN_MESSAGE))
            }
            (Self::RequestError, "invalid_chat_request") => {
                Some(Failure::Rejected(INVALID_REQUEST_MESSAGE))
            }
            (Self::RateError, "rate_limit_exceeded") => Some(Failure::Rejected(RATE_LIMIT_MESSAGE)),
            (Self::BackendError, "backend_token_count_failed") => {
                Some(Failure::Message(TOKEN_COUNT_FAILURE_MESSAGE))
            }
            (Self::BackendError, "backend_generation_failed") => {
                Some(Failure::Message(GENERATION_FAILURE_MESSAGE))
            }
            _ => None,
        }
    }
}

const fn response_kind(code: u16) -> Result<ResponseKind> {
    match code {
        200 => Ok(ResponseKind::Stream),
        400 => Ok(ResponseKind::RequestError),
        413 => Ok(ResponseKind::ContextError),
        429 => Ok(ResponseKind::RateError),
        502 => Ok(ResponseKind::BackendError),
        _ => Err(Failure::Message(
            "The local model rejected the request without reporting a recognized reason.",
        )),
    }
}

const RESPONSE_HEAD_POLICY: crate::http::HeadPolicy = crate::http::HeadPolicy {
    framing: crate::http::AllowedFraming::LengthOrChunked,
    require_connection_close: false,
    allow_htab: true,
};

const fn status_limit(code: u16) -> std::result::Result<usize, crate::http::TransportError> {
    match code {
        200 => Ok(MAX_BODY),
        400 | 413 | 429 | 502 => Ok(MAX_ERROR_BODY),
        _ => Err(crate::http::TransportError::UnexpectedStatus),
    }
}

fn accept_media(code: u16, value: &[u8]) -> bool {
    match code {
        200 => crate::http::utf8_media_type(value, "text/event-stream"),
        400 | 413 | 429 | 502 => value.eq_ignore_ascii_case(b"application/json"),
        _ => false,
    }
}

fn validate_response_head(head: crate::http::ResponseHead) -> Result<ResponseKind> {
    let kind = response_kind(head.code)?;
    if matches!(kind, ResponseKind::RateError)
        && head.retry_after != crate::http::RetryAfter::OneSecond
    {
        return Err(Failure::Message(
            "The local model returned an unrecognized retry delay.",
        ));
    }
    Ok(kind)
}

struct HttpBody<R> {
    reader: crate::http::BodyReader<R>,
}

impl<R: Read> HttpBody<R> {
    fn new(reader: R) -> Result<Self> {
        let body_policy = crate::http::BodyPolicy {
            chunk_line_limit: 32,
            max_body: MAX_BODY,
            require_even_chunks: false,
            require_non_empty: false,
        };
        let (head, reader) = crate::http::BodyReader::new(
            reader,
            READ_SIZE,
            RESPONSE_HEAD_POLICY,
            status_limit,
            accept_media,
            body_policy,
        )
        .map_err(map_transport)?;
        let kind = validate_response_head(head)?;
        let mut body = Self { reader };
        if !matches!(kind, ResponseKind::Stream) {
            return Err(body.failure(kind)?);
        }
        Ok(body)
    }

    fn failure(&mut self, kind: ResponseKind) -> Result<Failure> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Envelope<'a> {
            #[serde(borrow)]
            error: ErrorCode<'a>,
        }
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct ErrorCode<'a> {
            #[serde(borrow)]
            code: Cow<'a, str>,
        }
        let invalid = Failure::Message("The local model returned an unrecognized error reply.");
        let mut error = Vec::new();
        while let Some(bytes) = self.next_payload()? {
            if bytes.len() > MAX_ERROR_BODY - error.len() {
                return Err(invalid);
            }
            error.extend_from_slice(bytes);
        }
        let envelope: Envelope<'_> = serde_json::from_slice(&error).map_err(|_| invalid)?;
        kind.failure(&envelope.error.code).ok_or(invalid)
    }

    fn next_payload(&mut self) -> Result<Option<&[u8]>> {
        self.reader.next_payload().map_err(map_transport)
    }

    #[cfg(test)]
    fn buffer_ptr(&self) -> *const u8 {
        self.reader.buffer_ptr()
    }
}

#[cfg(test)]
fn parse_head(head: &[u8]) -> Result<ResponseKind> {
    let parsed =
        crate::http::ResponseHead::parse(head, RESPONSE_HEAD_POLICY, status_limit, accept_media)
            .map_err(map_transport)?;
    validate_response_head(parsed)
}

#[cfg(test)]
mod tests {
    #[test]
    fn startup_context_capacity_uses_one_complete_bounded_declaration() {
        use super::context_capacity;
        assert_eq!(
            context_capacity(b"Context tokens: 131072\n"),
            Ok(Some(131_072))
        );
        assert_eq!(
            context_capacity(b"Context tokens: 262144\r\n"),
            Ok(Some(262_144))
        );
        assert_eq!(context_capacity(b"Context tokens: 1310"), Ok(None));
        assert_eq!(
            context_capacity(b"other output\nContext tokens: 1310"),
            Ok(None)
        );
        for invalid in [
            "0",
            "262145",
            "+8192",
            " 8192",
            "8192 ",
            "8192.0",
            "",
            "184467440737095516160",
        ] {
            assert!(context_capacity(format!("Context tokens: {invalid}\n").as_bytes()).is_err());
        }
        assert!(context_capacity(b"Context tokens: 8192\nContext tokens: 8192\n").is_err());
        assert!(context_capacity(b"Context tokens: 8192\nContext tokens: 16384\n").is_err());
    }

    use super::{
        Command, Completion, Config, Event, Failure, HttpBody, Message, READ_SIZE, Sse,
        evidence_directory, finish_turn, parse_head, runtime_arguments, serialize_request,
        startup_failure, wait_for_retry,
    };

    fn publisher_runtime_config() -> Config {
        Config {
            broker: "broker.exe".into(),
            data: "data".into(),
            evidence: "evidence".into(),
            frontend_port: 5173,
            backend_port: 8081,
            require_broker_pipe: false,
            use_abliterate_lora: true,
            daisyrose_lora: None,
            no_daisyrose_lora: false,
            no_mmproj: false,
            use_repacked_ptq1: false,
            bonsai25_variant: None,
            native_engine: false,
            native_row_workers: 1,
            prompt_reuse: super::PromptReuse::Disabled,
            prompt_reuse_checkpoints: None,
            prompt_reuse_checkpoint_bytes: None,
            kv_cache: None,
            cpu_prefill_tokens: None,
            cpu_prefill_gdn: None,
            spec_type: None,
            draft_max: None,
            cpu_family: None,
            placement: None,
            canary: None,
            qwen_tts: None,
        }
    }

    #[test]
    fn kv_cache_and_runtime_choices_preserve_model_adapter_and_publisher_rollback() {
        let mut config = publisher_runtime_config();
        assert_eq!(
            runtime_arguments(&config),
            [
                "--use-abliterate-lora",
                "--use-original-pq2",
                "--publisher-engine",
                "--native-row-workers",
                "1"
            ]
        );
        config.native_engine = true;
        config.native_row_workers = 4;
        assert_eq!(
            runtime_arguments(&config),
            [
                "--use-abliterate-lora",
                "--use-original-pq2",
                "--native-engine",
                "--reset-on-cancel",
                "--native-row-workers",
                "4"
            ]
        );
        for (workers, text) in [(8, "8"), (12, "12")] {
            config.native_row_workers = workers;
            assert_eq!(
                runtime_arguments(&config),
                [
                    "--use-abliterate-lora",
                    "--use-original-pq2",
                    "--native-engine",
                    "--reset-on-cancel",
                    "--native-row-workers",
                    text
                ]
            );
        }
        config.native_row_workers = 4;
        config.use_abliterate_lora = false;
        config.use_repacked_ptq1 = true;
        assert_eq!(
            runtime_arguments(&config),
            [
                "--use-repacked-ptq1",
                "--native-engine",
                "--reset-on-cancel",
                "--native-row-workers",
                "4"
            ]
        );
        config.native_row_workers = 1;
        assert_eq!(
            runtime_arguments(&config),
            [
                "--use-repacked-ptq1",
                "--native-engine",
                "--reset-on-cancel",
                "--native-row-workers",
                "1"
            ]
        );
        config.use_repacked_ptq1 = false;
        assert_eq!(
            runtime_arguments(&config),
            [
                "--use-original-pq2",
                "--native-engine",
                "--reset-on-cancel",
                "--native-row-workers",
                "1"
            ]
        );
        config.no_mmproj = true;
        assert!(
            runtime_arguments(&config)
                .iter()
                .any(|arg| arg == "--no-mmproj")
        );
        config.no_mmproj = false;
        for policy in [
            super::KvCache::Rsq4,
            super::KvCache::Rsq4Kvf,
            super::KvCache::F16,
            super::KvCache::Rsq4K3v,
        ] {
            config.kv_cache = Some(policy);
            let arguments = runtime_arguments(&config);
            assert_eq!(
                arguments
                    .array_windows::<2>()
                    .filter(|args| **args == ["--kv-cache", policy.name()])
                    .count(),
                1
            );
        }
        for placement in [
            super::Placement::Auto,
            super::Placement::Cpu,
            super::Placement::Vulkan,
            super::Placement::Composite,
        ] {
            config.placement = Some(placement);
            let arguments = runtime_arguments(&config);
            assert_eq!(
                arguments
                    .array_windows::<2>()
                    .filter(|args| **args == ["--placement", placement.name()])
                    .count(),
                1
            );
        }
        config.prompt_reuse = super::PromptReuse::PromptEndAndLatest;
        config.prompt_reuse_checkpoints = Some(4);
        config.prompt_reuse_checkpoint_bytes = Some(134_217_728);
        config.cpu_prefill_tokens = Some(16);
        config.kv_cache = Some(super::KvCache::Rsq4Kvf);
        config.cpu_prefill_gdn = Some(super::CpuPrefillGdn::WyF32);
        config.cpu_family = Some(super::CpuFamily::X2);
        config.placement = Some(super::Placement::Vulkan);
        let selected = runtime_arguments(&config);
        for pair in [
            ["--cpu-family", "x2"],
            ["--kv-cache", "rsq4kvf"],
            ["--cpu-prefill-tokens", "16"],
            ["--cpu-prefill-gdn", "wy-f32"],
            ["--placement", "vulkan"],
            ["--prompt-reuse", "prompt-end-and-latest"],
            ["--prompt-reuse-checkpoints", "4"],
            ["--prompt-reuse-checkpoint-bytes", "134217728"],
        ] {
            assert!(
                selected.array_windows().any(|args| *args == pair),
                "missing {pair:?} in {selected:?}"
            );
        }
        config.kv_cache = Some(super::KvCache::Rsq4);
        config.cpu_prefill_gdn = None;
        config.spec_type = Some(super::SpeculationType::Suffix);
        config.draft_max = Some(6);
        let spec_selected = runtime_arguments(&config);
        assert!(
            spec_selected
                .array_windows()
                .any(|args| *args == ["--spec-type", "suffix"])
        );
        assert!(
            spec_selected
                .array_windows()
                .any(|args| *args == ["--draft-max", "6"])
        );
        let retry = Config {
            evidence: "retry-evidence".into(),
            ..config.clone()
        };
        assert_eq!(runtime_arguments(&retry), spec_selected);
    }

    #[test]
    fn owned_broker_spec_preserves_explicit_profile_paths_ports_and_evidence() {
        let mut config = Config {
            broker: r"C:\native app\broker.exe".into(),
            data: "C:\\model data\\\u{96e8}".into(),
            evidence: r"C:\private evidence\attempt-3".into(),
            frontend_port: 5174,
            backend_port: 8082,
            require_broker_pipe: true,
            use_abliterate_lora: true,
            daisyrose_lora: None,
            no_daisyrose_lora: false,
            no_mmproj: false,
            use_repacked_ptq1: false,
            bonsai25_variant: None,
            native_engine: true,
            native_row_workers: 4,
            prompt_reuse: super::PromptReuse::Disabled,
            prompt_reuse_checkpoints: None,
            prompt_reuse_checkpoint_bytes: None,
            kv_cache: None,
            cpu_prefill_tokens: None,
            cpu_prefill_gdn: None,
            spec_type: None,
            draft_max: None,
            cpu_family: None,
            placement: None,
            canary: None,
            qwen_tts: None,
        };
        for publisher in [false, true] {
            config.native_engine = !publisher;
            config.require_broker_pipe = !publisher;
            config.native_row_workers = if publisher { 1 } else { 4 };
            let control = config.evidence.join("private-owner");
            let spec = super::broker_spec(&config, &control);
            let mut expected: Vec<std::ffi::OsString> = runtime_arguments(&config)
                .into_iter()
                .map(Into::into)
                .collect();
            expected.extend([
                "--launch-development".into(),
                "--require-native-sandbox".into(),
            ]);
            if !publisher {
                expected.push("--require-broker-pipe".into());
            }
            expected.extend([
                "--profile".into(),
                "swift-pq2-assessment".into(),
                "--data-directory".into(),
                config.data.as_os_str().to_owned(),
                "--evidence-parent".into(),
                config.evidence.as_os_str().to_owned(),
                "--frontend-port".into(),
                "5174".into(),
                "--backend-port".into(),
                "8082".into(),
                "--max-waiting".into(),
                if publisher { "0" } else { "1" }.into(),
                "--owner-shutdown-directory".into(),
                control.into_os_string(),
            ]);
            assert_eq!(spec.executable, config.broker);
            assert_eq!(spec.arguments, expected);
            assert_eq!(spec.working_directory, config.evidence);
            assert_eq!(spec.stdout, config.evidence.join("broker.stdout.log"));
            assert_eq!(spec.stderr, config.evidence.join("broker.stderr.log"));
            assert!(spec.environment.is_empty());
        }
    }

    #[test]
    fn committed_and_pending_history_share_payloads_without_sharing_turn_mutations()
    -> Result<(), Failure> {
        use std::sync::{Arc, mpsc};

        let prior = Message {
            role: "assistant",
            content: "A completed answer 🌱".repeat(8192).into(),
            reasoning_content: Some("Retained reasoning 雨".repeat(8192).into()),
            images: Arc::default(),
        };
        let mut history = vec![
            Message {
                role: "user",
                content: "The original task".into(),
                reasoning_content: None,
                images: Arc::default(),
            },
            prior.clone(),
        ];
        let mut pending = history.clone();
        assert!(Arc::ptr_eq(&history[1].content, &pending[1].content));
        assert!(matches!(
            (&history[1].reasoning_content, &pending[1].reasoning_content),
            (Some(first), Some(second)) if Arc::ptr_eq(first, second)
        ));
        pending.push(Message {
            role: "user",
            content: "A new task".into(),
            reasoning_content: None,
            images: Arc::default(),
        });
        assert_eq!(history.len(), 2);
        let (_commands, commands) = mpsc::channel();
        let (events, incoming) = crate::client::event_channel();
        finish_turn(
            &mut history,
            pending,
            "A new task",
            Ok((
                Message {
                    role: "assistant",
                    content: "The new answer".into(),
                    reasoning_content: None,
                    images: Arc::default(),
                },
                Completion::Finished,
            )),
            &events,
            &commands,
            true,
        )?;
        let Ok(Event::Committed(mut visible)) = incoming.try_recv() else {
            return Err(Failure::Message("Expected committed history."));
        };
        assert!(Arc::ptr_eq(&prior.content, &history[1].content));
        assert!(Arc::ptr_eq(&visible[1].content, &history[1].content));
        assert!(Arc::ptr_eq(&visible[3].content, &history[3].content));
        visible[1].content = "Independent replacement".into();
        visible.clear();
        assert_eq!(history.len(), 4);
        assert_eq!(history[1].content.as_ref(), prior.content.as_ref());
        Ok(())
    }

    #[test]
    fn thinking_request_selection_preserves_supplied_reasoning() -> Result<(), Failure> {
        let history = [
            Message {
                role: "user",
                content: "Write a garden scene.".into(),
                reasoning_content: None,
                images: super::Arc::default(),
            },
            Message {
                role: "assistant",
                content: "The garden grew quiet.".into(),
                reasoning_content: Some("Retained prior reasoning".into()),
                images: super::Arc::default(),
            },
            Message {
                role: "user",
                content: "Continue.".into(),
                reasoning_content: None,
                images: super::Arc::default(),
            },
        ];
        for enable_thinking in [false, true] {
            let body = serialize_request(&history, 4096, enable_thinking, None)?;
            let request: serde_json::Value = serde_json::from_slice(&body)
                .map_err(|_| Failure::Message("invalid thinking request JSON"))?;
            assert_eq!(request["enable_thinking"], enable_thinking);
            assert_eq!(request["messages"][0]["role"], "system");
            assert_eq!(request["messages"][0]["content"], super::SYSTEM_PROMPT);
            assert!(request["messages"][0].get("reasoning_content").is_none());
            assert_eq!(
                request["messages"][2]["reasoning_content"],
                "Retained prior reasoning"
            );
            assert_eq!(request["max_completion_tokens"], 4096);
            assert_eq!(request["stream"], true);
        }
        Ok(())
    }

    #[test]
    fn abliteration_requests_preserve_scale_bounds_and_omit_disabled_adapter() -> Result<(), Failure>
    {
        let history = [Message {
            role: "user",
            content: "Write a garden scene.".into(),
            reasoning_content: None,
            images: super::Arc::default(),
        }];
        for scale in [0.0, super::DEFAULT_ABLITERATION_SCALE, 2.0] {
            let body = serialize_request(&history, 4096, false, Some(scale))?;
            let request: serde_json::Value = serde_json::from_slice(&body)
                .map_err(|_| Failure::Message("invalid scale request JSON"))?;
            assert!(request["abliteration_scale"].as_f64().is_some_and(|value| {
                (value - f64::from(scale)).abs() < f64::from(f32::EPSILON)
            }));
            assert!(request.get("lora").is_none());
        }
        let body = serialize_request(&history, 4096, false, None)?;
        let request: serde_json::Value = serde_json::from_slice(&body)
            .map_err(|_| Failure::Message("invalid disabled request JSON"))?;
        assert!(request.get("abliteration_scale").is_none());
        for scale in [-0.1, 2.1, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(matches!(
                serialize_request(&history, 4096, false, Some(scale)),
                Err(Failure::Rejected(_))
            ));
            assert!(matches!(
                super::serialize_request_with_adapters(
                    &history,
                    4096,
                    false,
                    Some(1.3),
                    Some(scale)
                ),
                Err(Failure::Rejected(_))
            ));
        }
        let stacked =
            super::serialize_request_with_adapters(&history, 4096, false, Some(1.3), Some(0.5))?;
        let stacked_json: serde_json::Value = serde_json::from_slice(&stacked)
            .map_err(|_| Failure::Message("invalid stacked request JSON"))?;
        assert!(
            stacked_json["daisyrose_scale"]
                .as_f64()
                .is_some_and(|v| (v - 0.5).abs() < 1e-6)
        );
        let mut with_daisy = publisher_runtime_config();
        with_daisy.daisyrose_lora = Some(r"C:\adapters\daisyrose.gguf".into());
        assert!(
            runtime_arguments(&with_daisy)
                .array_windows()
                .any(|pair| *pair == ["--daisyrose-lora", r"C:\adapters\daisyrose.gguf"])
        );
        let mut without_daisy = publisher_runtime_config();
        without_daisy.no_daisyrose_lora = true;
        assert!(
            runtime_arguments(&without_daisy)
                .iter()
                .any(|arg| arg == "--no-daisyrose-lora")
        );
        Ok(())
    }
    use std::io::{self, Read};

    struct Fragmented<'a>(&'a [u8]);
    impl Read for Fragmented<'_> {
        fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
            let count = usize::from(!self.0.is_empty()).min(bytes.len());
            bytes[..count].copy_from_slice(&self.0[..count]);
            self.0 = &self.0[count..];
            Ok(count)
        }
    }

    #[test]
    fn conversation_workspace_restores_committed_history_during_retry_without_replay()
    -> Result<(), Failure> {
        let bytes = br#"[{"role":"user","content":"Saved question"},{"role":"assistant","content":"Saved answer","reasoning_content":"Saved reasoning"}]"#;
        let saved: Vec<Message> = serde_json::from_slice(bytes)
            .map_err(|_| Failure::Message("History roundtrip failed"))?;
        let expected = serialize_request(&saved, 4096, false, None)?;
        assert!(
            serde_json::from_slice::<Message>(br#"{"role":"system","content":"Unaccepted role"}"#)
                .is_err()
        );
        let (commands, incoming) = std::sync::mpsc::channel();
        let (events, notifications) = crate::client::event_channel();
        assert!(commands.send(Command::RestoreChat(saved)).is_ok());
        assert!(commands.send(Command::Retry).is_ok());
        let mut history = Vec::new();
        assert!(wait_for_retry(
            &incoming,
            &events,
            &mut history,
            &Event::Retryable("Retry explicitly.".to_owned())
        ));
        assert_eq!(serialize_request(&history, 4096, false, None)?, expected);
        assert!(matches!(notifications.try_recv(), Ok(Event::Retryable(_))));
        assert!(notifications.try_recv().is_err());
        Ok(())
    }
    #[test]
    fn fragmented_chunked_unicode_stream_preserves_reasoning_and_history() -> Result<(), Failure> {
        let events = concat!(
            "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"Consider 🌱\"}}]}\r\n\r\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"Rain 雨\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n"
        );
        let mut wire = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n".to_vec();
        for chunk in events.as_bytes().chunks(7) {
            wire.extend_from_slice(format!("{:X}\r\n", chunk.len()).as_bytes());
            wire.extend_from_slice(chunk);
            wire.extend_from_slice(b"\r\n");
        }
        wire.extend_from_slice(b"0\r\n\r\n");
        let mut body = HttpBody::new(Fragmented(&wire))?;
        let mut sse = Sse::default();
        let mut text = String::new();
        let mut reasoning = String::new();
        while let Some(bytes) = body.next_payload()? {
            sse.push(bytes, &mut |content, thought| {
                text.push_str(&content);
                reasoning.push_str(&thought);
                Ok(())
            })?;
        }
        sse.finish()?;
        assert_eq!(text, "Rain 雨");
        assert_eq!(reasoning, "Consider 🌱");
        let request = serialize_request(
            &[
                Message {
                    role: "user",
                    content: "Begin".into(),
                    reasoning_content: None,
                    images: super::Arc::default(),
                },
                Message {
                    role: "assistant",
                    content: text.into(),
                    reasoning_content: Some(reasoning.into()),
                    images: super::Arc::default(),
                },
                Message {
                    role: "user",
                    content: "Continue".into(),
                    reasoning_content: None,
                    images: super::Arc::default(),
                },
            ],
            4096,
            false,
            None,
        )?;
        let json: serde_json::Value =
            serde_json::from_slice(&request).map_err(|_| Failure::Message("test request JSON"))?;
        assert_eq!(json["messages"][0]["role"], "system");
        assert_eq!(json["messages"][0]["content"], super::SYSTEM_PROMPT);
        assert!(json["messages"][0].get("reasoning_content").is_none());
        assert_eq!(json["messages"][2]["reasoning_content"], "Consider 🌱");
        assert!(json["messages"][1].get("reasoning_content").is_none());
        assert_eq!(json["max_completion_tokens"], 4096);
        assert_eq!(json["stream"], true);
        assert_eq!(json.as_object().map(serde_json::Map::len), Some(4));
        Ok(())
    }

    #[test]
    fn http_body_preserves_short_final_reads_for_both_framings() -> Result<(), Failure> {
        let mut expected = vec![b'x'; READ_SIZE + 3];
        expected[READ_SIZE..].copy_from_slice(b"end");
        for chunked in [false, true] {
            let framing = if chunked {
                format!("Transfer-Encoding: chunked\r\n\r\n{:X}\r\n", expected.len())
            } else {
                format!("Content-Length: {}\r\n\r\n", expected.len())
            };
            let mut wire = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n".to_vec();
            wire.extend_from_slice(framing.as_bytes());
            wire.extend_from_slice(&expected);
            if chunked {
                wire.extend_from_slice(b"\r\n0\r\n\r\n");
            }
            let mut body = HttpBody::new(Fragmented(&wire))?;
            let mut actual = Vec::new();
            while let Some(bytes) = body.next_payload()? {
                actual.extend_from_slice(bytes);
            }
            assert_eq!(actual, expected);
            assert_eq!(body.next_payload()?, None);
        }
        Ok(())
    }

    #[test]
    fn http_payload_lends_reader_storage_and_excludes_coalesced_framing() -> Result<(), Failure> {
        for (framing, payload, offset) in [
            ("Content-Length: 3\r\n\r\n", "abcdiscarded", 0),
            (
                "Transfer-Encoding: chunked\r\n\r\n",
                "3\r\nabc\r\n2\r\nde\r\n0\r\n\r\ndiscarded",
                3,
            ),
        ] {
            let wire =
                format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n{framing}{payload}");
            let mut body = HttpBody::new(wire.as_bytes())?;
            let expected = body.buffer_ptr().wrapping_add(offset);
            let bytes = body
                .next_payload()?
                .ok_or(Failure::Message("Buffered payload was not lent"))?;
            assert_eq!(bytes, b"abc");
            assert_eq!(bytes.as_ptr(), expected);
            if offset != 0 {
                assert_eq!(body.next_payload()?, Some(b"de".as_slice()));
            }
            assert_eq!(body.next_payload()?, None);
            assert_eq!(body.next_payload()?, None);
        }
        Ok(())
    }

    struct ReadyPrefix<'a>(&'a [u8]);

    impl Read for ReadyPrefix<'_> {
        fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
            if self.0.is_empty() {
                return Err(io::Error::from(io::ErrorKind::WouldBlock));
            }
            self.0.read(bytes)
        }
    }

    #[test]
    fn http_payload_releases_ready_sse_before_the_declared_body_arrives() -> Result<(), Failure> {
        let event = b"data: {\"choices\":[{\"delta\":{\"content\":\"Ready\"}}]}\n\n";
        for framing in [
            "Content-Length: 8192\r\n\r\n",
            "Transfer-Encoding: chunked\r\n\r\n2000\r\n",
        ] {
            let mut wire = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n".to_vec();
            wire.extend_from_slice(framing.as_bytes());
            wire.extend_from_slice(event);
            let mut body = HttpBody::new(ReadyPrefix(&wire))?;
            let bytes = body
                .next_payload()?
                .ok_or(Failure::Message("Ready payload was not lent"))?;
            assert_eq!(bytes, event);
            let mut received = String::new();
            Sse::default().push(bytes, &mut |text, _| {
                received.push_str(&text);
                Ok(())
            })?;
            assert_eq!(received, "Ready");
            // The gate rejects a second underlying read. It was not needed to
            // publish the complete event, and the unfinished body cannot pass.
            assert!(body.next_payload().is_err());
        }
        Ok(())
    }

    #[test]
    fn http_payload_precedes_chunk_ending_but_invalid_endings_never_complete() -> Result<(), Failure>
    {
        for ending in [b"".as_slice(), b"\r", b"xx", b"\r\n0\r\n\r"] {
            let mut wire = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n3\r\nabc".to_vec();
            wire.extend_from_slice(ending);
            let mut body = HttpBody::new(ReadyPrefix(&wire))?;
            assert_eq!(body.next_payload()?, Some(b"abc".as_slice()));
            assert!(body.next_payload().is_err());
        }
        Ok(())
    }

    #[test]
    fn http_payload_preserves_interrupted_reads_and_bounded_chunk_metadata() -> Result<(), Failure>
    {
        struct Interrupted;
        impl Read for Interrupted {
            fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
                Err(io::Error::from(io::ErrorKind::Interrupted))
            }
        }
        let head =
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: 1\r\n\r\n";
        let mut body = HttpBody::new(head.as_slice().chain(Interrupted))?;
        assert!(body.next_payload().is_err());
        for chunk in [
            "400001\r\n", // The declared payload exceeds the body budget.
            "1;extension\r\n",
            "0000000000000000000000000000001\r\n", // Beyond 32 bytes including CRLF.
            "0\r\nX-Trailer: no\r\n\r\n",
        ] {
            let wire = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n{chunk}"
            );
            let mut body = HttpBody::new(wire.as_bytes())?;
            assert!(body.next_payload().is_err());
        }
        let wire = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n000000000000000000000000000001\r\nx\r\n0\r\n\r\n";
        let mut body = HttpBody::new(wire.as_slice())?;
        assert_eq!(body.next_payload()?, Some(b"x".as_slice()));
        assert_eq!(body.next_payload()?, None);
        Ok(())
    }

    #[test]
    fn truncated_http_and_incomplete_generation_are_rejected() -> Result<(), Failure> {
        let mut body = HttpBody::new(Fragmented(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nabc"))?;
        let mut received = Vec::new();
        loop {
            match body.next_payload() {
                Ok(Some(bytes)) => received.extend_from_slice(bytes),
                Err(_) => break,
                Ok(None) => return Err(Failure::Message("Truncated HTTP was accepted")),
            }
        }
        assert_eq!(received, b"abc");
        let mut sse = Sse::default();
        sse.push(
            b"data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n",
            &mut |_, _| Ok(()),
        )?;
        assert!(sse.finish().is_err());
        assert!(parse_head(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: 1\r\nTransfer-Encoding: chunked\r\n\r\n").is_err());
        assert!(parse_head(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: 1\r\nContent-Length: 1\r\n\r\n").is_err());
        Ok(())
    }

    fn context_response(body: &[u8], chunked: bool) -> Vec<u8> {
        rejection_response("413 Content Too Large", "", body, chunked)
    }

    fn rejection_response(status: &str, headers: &str, body: &[u8], chunked: bool) -> Vec<u8> {
        let mut wire = format!("HTTP/1.1 {status}\r\nContent-Type: application/json\r\n{headers}")
            .into_bytes();
        if chunked {
            wire.extend_from_slice(
                format!("Transfer-Encoding: chunked\r\n\r\n{:X}\r\n", body.len()).as_bytes(),
            );
            wire.extend_from_slice(body);
            wire.extend_from_slice(b"\r\n0\r\n\r\n");
        } else {
            wire.extend_from_slice(format!("Content-Length: {}\r\n\r\n", body.len()).as_bytes());
            wire.extend_from_slice(body);
        }
        wire
    }

    #[test]
    fn pre_generation_failures_require_complete_status_bound_error_codes() {
        for (status, code, failure) in [
            (
                "413 Content Too Large",
                "request_body_too_large",
                Failure::Rejected(super::REQUEST_LIMIT_MESSAGE),
            ),
            (
                "400 Bad Request",
                "invalid_chat_request",
                Failure::Rejected(super::INVALID_REQUEST_MESSAGE),
            ),
            (
                "502 Bad Gateway",
                "backend_token_count_failed",
                Failure::Message(super::TOKEN_COUNT_FAILURE_MESSAGE),
            ),
            (
                "502 Bad Gateway",
                "backend_generation_failed",
                Failure::Message(super::GENERATION_FAILURE_MESSAGE),
            ),
        ] {
            let body = format!(r#"{{"error":{{"code":"{code}"}}}}"#);
            for chunked in [false, true] {
                let wire = rejection_response(status, "", body.as_bytes(), chunked);
                assert!(
                    matches!(HttpBody::new(Fragmented(&wire)), Err(actual) if actual == failure)
                );
                assert!(matches!(
                    HttpBody::new(Fragmented(&wire[..wire.len() - 1])),
                    Err(Failure::Message(message)) if message != super::TOKEN_COUNT_FAILURE_MESSAGE
                        && message != super::GENERATION_FAILURE_MESSAGE
                ));
                let wrong_status = if status.starts_with("502") {
                    "400 Bad Request"
                } else {
                    "502 Bad Gateway"
                };
                let wire = rejection_response(wrong_status, "", body.as_bytes(), chunked);
                assert!(
                    matches!(HttpBody::new(Fragmented(&wire)), Err(actual) if actual != failure)
                );
                for invalid in [
                    format!(r#"{{"error":{{"code":"{code}","code":"{code}"}}}}"#),
                    format!(r#"{{"error":{{"code":"{code}","message":"untrusted text"}}}}"#),
                    r#"{"error":{"code":"unknown"}}"#.to_owned(),
                    "x".repeat(super::MAX_ERROR_BODY + 1),
                ] {
                    let wire = rejection_response(status, "", invalid.as_bytes(), chunked);
                    assert!(
                        matches!(HttpBody::new(Fragmented(&wire)), Err(actual) if actual != failure)
                    );
                }
            }
        }
    }

    #[test]
    fn security_rejections_require_exact_codes_retry_metadata_and_complete_framing() {
        for (status, headers, body, message) in [
            (
                "400 Bad Request",
                "",
                br#"{"error":{"code":"input_control_tokens_rejected"}}"#.as_slice(),
                super::CONTROL_TOKEN_MESSAGE,
            ),
            (
                "429 Too Many Requests",
                "Retry-After: 1\r\n",
                br#"{"error":{"code":"rate_limit_exceeded"}}"#.as_slice(),
                super::RATE_LIMIT_MESSAGE,
            ),
        ] {
            for chunked in [false, true] {
                let wire = rejection_response(status, headers, body, chunked);
                assert!(
                    matches!(HttpBody::new(Fragmented(&wire)), Err(Failure::Rejected(actual))
                        if actual == message)
                );
                assert!(matches!(
                    HttpBody::new(Fragmented(&wire[..wire.len() - 1])),
                    Err(Failure::Message(_))
                ));
                for invalid in [
                    br#"{"error":{"code":"context_length_exceeded"}}"#.as_slice(),
                    br#"{"error":{"code":"rate_limit_exceeded","message":"untrusted text"}}"#,
                    br#"{"error":{"code":"input_control_tokens_rejected","code":"input_control_tokens_rejected"}}"#,
                    b"not JSON",
                    &[b'x'; super::MAX_ERROR_BODY + 1],
                ] {
                    let wire = rejection_response(status, headers, invalid, chunked);
                    assert!(matches!(
                        HttpBody::new(Fragmented(&wire)),
                        Err(Failure::Message(_))
                    ));
                }
            }
        }
        for headers in [
            "",
            "Retry-After: 0\r\n",
            "Retry-After: 2\r\n",
            "Retry-After: tomorrow\r\n",
            "Retry-After: 1\r\nRetry-After: 1\r\n",
        ] {
            let wire = rejection_response(
                "429 Too Many Requests",
                headers,
                br#"{"error":{"code":"rate_limit_exceeded"}}"#,
                false,
            );
            assert!(matches!(
                HttpBody::new(Fragmented(&wire)),
                Err(Failure::Message(_))
            ));
        }
        for (status, body) in [
            (
                "400 Bad Request",
                br#"{"error":{"code":"rate_limit_exceeded"}}"#.as_slice(),
            ),
            (
                "429 Too Many Requests",
                br#"{"error":{"code":"input_control_tokens_rejected"}}"#.as_slice(),
            ),
        ] {
            let wire = rejection_response(status, "Retry-After: 1\r\n", body, false);
            assert!(matches!(
                HttpBody::new(Fragmented(&wire)),
                Err(Failure::Message(_))
            ));
        }
    }

    #[test]
    fn context_rejection_requires_a_complete_bounded_recognized_reply() {
        let body = br#"{"error":{"code":"context_length_exceeded"}}"#;
        for chunked in [false, true] {
            let wire = context_response(body, chunked);
            assert!(
                matches!(HttpBody::new(Fragmented(&wire)), Err(Failure::Rejected(message))
                if message == super::CONTEXT_LIMIT_MESSAGE)
            );
            // Even a complete JSON code is insufficient without the complete
            // declared body or final chunk terminator.
            assert!(matches!(
                HttpBody::new(Fragmented(&wire[..wire.len() - 1])),
                Err(Failure::Message(_))
            ));
        }
        for invalid in [
            br#"{"error":{"code":"unknown"}}"#.as_slice(),
            br#"{"error":{"code":"context_length_exceeded","message":"untrusted text"}}"#,
            br#"{"error":{"code":"context_length_exceeded","code":"context_length_exceeded"}}"#,
            b"not JSON",
            &[b'x'; super::MAX_ERROR_BODY + 1],
        ] {
            for chunked in [false, true] {
                let wire = context_response(invalid, chunked);
                assert!(matches!(
                    HttpBody::new(Fragmented(&wire)),
                    Err(Failure::Message(_))
                ));
            }
        }
        for headers in [
            "Content-Length: 44\r\nContent-Length: 44",
            "Content-Length: 44\r\nTransfer-Encoding: chunked",
            "Content-Length: 44\r\nContent-Encoding: gzip",
        ] {
            let wire = format!(
                "HTTP/1.1 413 Content Too Large\r\nContent-Type: application/json\r\n{headers}\r\n\r\n"
            );
            assert!(matches!(
                HttpBody::new(Fragmented(wire.as_bytes())),
                Err(Failure::Message(_))
            ));
        }
        let wire = context_response(body, false);
        let wrong_status = String::from_utf8_lossy(&wire)
            .replace("413 Content Too Large", "500 Internal Server Error");
        assert!(matches!(
            HttpBody::new(Fragmented(wrong_status.as_bytes())),
            Err(Failure::Message(_))
        ));
    }

    #[test]
    fn unexpected_disconnect_explains_local_connection_without_guessing_token_limits() {
        let wire = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n";
        for response in [b"".as_slice(), wire] {
            assert!(
                matches!(HttpBody::new(Fragmented(response)), Err(Failure::Message(message))
                if message == super::LOCAL_CONNECTION_CLOSED)
            );
        }
        let guidance = super::retry_guidance(super::LOCAL_CONNECTION_CLOSED);
        assert!(guidance.contains("on this computer"));
        assert!(guidance.contains("cause was not reported"));
        assert!(guidance.contains("Retry model"));
        for misleading in [
            "HTTP",
            "Restart the app",
            "output budget",
            "request was too long",
        ] {
            assert!(!guidance.contains(misleading));
        }
    }

    #[test]
    fn event_stream_conformance_handles_bom_cr_lines_and_split_fields() -> Result<(), Failure> {
        for ending in ["\r", "\n", "\r\n"] {
            let events = [
                ": keepalive",
                "id: ignored-without-reconnect",
                "retry: 1000",
                "event: message",
                // A colonless field is a field whose value is empty. The data
                // lines concatenate with LF before JSON decoding.
                "data",
                "data: {\"choices\":[{\"delta\":",
                "data: {\"content\":\"Rain 雨\"},\"index\":0,\"finish_reason\":\"stop\"}]}",
                "",
                "data: [DONE]",
                "",
                ": unterminated comment is discarded at EOF",
            ]
            .join(ending);
            let mut wire = b"\xEF\xBB\xBF".to_vec();
            wire.extend_from_slice(events.as_bytes());
            let mut sse = Sse::default();
            let mut text = String::new();
            for byte in wire {
                sse.push(&[byte], &mut |content, _| {
                    text.push_str(&content);
                    Ok(())
                })?;
            }
            assert_eq!(sse.finish()?, Completion::Finished);
            assert_eq!(text, "Rain 雨");
        }
        Ok(())
    }

    #[test]
    fn terminal_reasons_distinguish_token_limits_and_reject_false_success() -> Result<(), Failure> {
        let mut capped = Sse::default();
        capped.push(
            concat!(
                "data: {\"choices\":[{\"delta\":{\"content\":\"Partial answer\"},\"finish_reason\":\"length\"}]}\n\n",
                "data: {\"choices\":[],\"usage\":{\"completion_tokens\":4096}}\n\n",
                "data: [DONE]\n\n"
            ).as_bytes(),
            &mut |_, _| Ok(()),
        )?;
        assert_eq!(capped.finish()?, Completion::TokenLimit);
        assert_eq!(capped.into_answer().0, "Partial answer");
        for reason in [
            "null",
            "\"tool_calls\"",
            "\"content_filter\"",
            "\"unknown\"",
        ] {
            let mut sse = Sse::default();
            let events = format!(
                "data: {{\"choices\":[{{\"delta\":{{\"content\":\"partial\"}},\"finish_reason\":{reason}}}]}}\n\ndata: [DONE]\n\n"
            );
            assert!(sse.push(events.as_bytes(), &mut |_, _| Ok(())).is_err());
        }
        for invalid in [
            "data: {\"choices\":[{\"delta\":{\"content\":\"unexpected\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            "event: error\ndata: {\"message\":\"backend failed\"}\n\n",
        ] {
            let mut sse = Sse::default();
            sse.push(
                b"data: {\"choices\":[{\"delta\":{\"content\":\"answer\"},\"finish_reason\":\"stop\"}]}\n\n",
                &mut |_, _| Ok(()),
            )?;
            assert!(sse.push(invalid.as_bytes(), &mut |_, _| Ok(())).is_err());
        }
        let mut missing_done = Sse::default();
        missing_done.push(
            b"data: {\"choices\":[{\"delta\":{\"content\":\"answer\"},\"finish_reason\":\"stop\"}]}\n\n",
            &mut |_, _| Ok(()),
        )?;
        assert!(missing_done.finish().is_err());
        Ok(())
    }

    #[test]
    fn event_stream_media_type_accepts_utf8_parameters_and_rejects_ambiguity() {
        for media in [
            "text/event-stream",
            "TEXT/EVENT-STREAM",
            "Text/Event-Stream; CHARSET=UTF-8",
            "text/event-stream ; charset = \"utf-8\"",
        ] {
            let head =
                format!("HTTP/1.1 200 OK\r\nContent-Type: {media}\r\nContent-Length: 0\r\n\r\n");
            assert!(parse_head(head.as_bytes()).is_ok());
        }
        for media in [
            "text/event-stream-extra",
            "text/event-stream; charset=latin1",
            "text/event-stream; charset=utf-8; charset=utf-8",
            "text/event-stream; charset=\"utf-8",
            "application/json",
        ] {
            let head =
                format!("HTTP/1.1 200 OK\r\nContent-Type: {media}\r\nContent-Length: 0\r\n\r\n");
            assert!(parse_head(head.as_bytes()).is_err());
        }
    }

    #[test]
    fn startup_sandbox_failures_explain_support_and_policy_separately() {
        assert!(
            matches!(startup_failure(b"Development backend launch failed: SandboxEnvironment.", false),
                Failure::Message(message) if message.contains("parent environment") && !message.contains("Update Windows"))
        );
        assert!(
            matches!(startup_failure(b"Development backend launch failed: SandboxUnavailable.", false),
                Failure::Message(message) if message.contains("Update Windows") && message.contains("loopback") && super::retry_guidance(message) == message)
        );
        assert!(
            matches!(startup_failure(b"Development backend launch failed: SandboxUnavailable. Native Windows sandbox support (PSEC 1.1) is unavailable because a Windows Update restart is pending.", true),
                Failure::Message(message) if message.contains("pending restart") && message.contains("Restart Windows") && super::retry_guidance(message) == message)
        );
        assert!(
            matches!(startup_failure(b"Development backend launch failed: SandboxUnavailable. Native Windows sandbox support (PSEC 1.1) is unavailable on this Windows build.", true),
                Failure::Message(message) if message.contains("PSEC 1.1") && message.contains("Windows updates") && !message.contains("loopback") && super::retry_guidance(message) == message)
        );
        assert!(
            matches!(startup_failure(b"Development backend launch failed: Sandbox.", false),
                Failure::Message(message) if message.contains("required policy") && message.contains("diagnostics"))
        );
    }

    #[test]
    fn publisher_runtime_initialization_failure_offers_explicit_native_relaunch()
    -> Result<(), Failure> {
        let diagnostic = b"Native backend startup failure: stage=authenticated-readiness, error=Backend, process=exited-0xC0000142\r\nDevelopment backend launch failed: Backend.\r\nsynthetic-private-backend-detail\r\n";
        let Failure::Message(message) = startup_failure(diagnostic, false) else {
            return Err(Failure::Message(
                "Missing publisher initialization guidance",
            ));
        };
        assert!(message.contains("publisher CPU runtime"));
        assert!(message.contains("Windows initialization (0xC0000142)"));
        assert!(message.contains("Close this app"));
        assert!(message.contains("Retry model repeats that runtime choice"));
        assert!(message.contains("Copy any conversation or drafts"));
        assert!(message.contains("bonsai-tools chat with --native-engine"));
        assert!(message.contains("experimental Rust runtime"));
        assert!(message.contains("DataDirectory"));
        assert!(message.contains("same model and adapter options"));
        assert!(!message.contains("synthetic-private-backend-detail"));
        assert_eq!(super::retry_guidance(message), message);
        assert_eq!(
            startup_failure(diagnostic, true),
            startup_failure(b"Development backend launch failed: Backend.", true)
        );
        assert!(
            matches!(startup_failure(diagnostic, true), Failure::Message(native)
                if !native.contains("--native-engine") && !native.contains("publisher"))
        );
        assert_eq!(
            startup_failure(b"Development backend launch failed: Memory.", true),
            Failure::Memory
        );
        Ok(())
    }

    #[test]
    fn startup_initialization_guidance_requires_the_exact_fixed_marker() {
        let generic = startup_failure(b"Development backend launch failed: Backend.", false);
        for diagnostic in [
            "Native backend startup failure: stage=authenticated-readiness, error=Backend, process=exited-0xC0000005\n",
            "Native backend startup failure: stage=native-identity, error=Backend, process=exited-0xC0000142\n",
            "Native backend startup failure: stage=authenticated-readiness, error=Backend, process=exited-0xc0000142\n",
            "prefix Native backend startup failure: stage=authenticated-readiness, error=Backend, process=exited-0xC0000142\n",
            "Native backend startup failure: stage=authenticated-readiness, error=Backend, process=exited-0xC0000142 suffix\n",
            "Development backend launch failed: Backend. 0xC0000142\n",
        ] {
            assert_eq!(startup_failure(diagnostic.as_bytes(), false), generic);
        }
    }

    #[test]
    fn startup_uses_complete_lines_and_safe_memory_message() -> Result<(), Failure> {
        assert!(
            evidence_directory(b"Protected credentials and technical evidence: C:\\unfinished")?
                .is_none()
        );
        assert_eq!(
            startup_failure(b"Development backend launch failed: Memory.", false),
            Failure::Memory
        );
        let Failure::Message(message) =
            startup_failure(b"Development backend launch failed: Adapter.", false)
        else {
            return Err(Failure::Message("Missing adapter startup guidance"));
        };
        assert!(message.contains("adapter"));
        assert!(message.contains("repair its download automatically"));
        assert!(message.contains("Retry setup"));
        assert!(!message.contains("bonsai-tools"));
        Ok(())
    }

    #[test]
    fn model_verification_guidance_distinguishes_fixed_reasons_and_selected_exports()
    -> Result<(), Failure> {
        for (reason, selection, expected) in [
            ("missing", "original", "model file is missing"),
            ("storage", "original", "cannot safely read"),
            ("identity", "original", "size and integrity checks"),
            ("missing", "ptq1", "PTQ1 model export is missing"),
            ("storage", "ptq1", "cannot safely read"),
            ("identity", "ptq1", "PTQ1 export did not pass"),
            ("configuration", "original", "model configuration"),
            ("cancelled", "original", "verification was cancelled"),
        ] {
            let diagnostic = format!(
                "Selected model verification failed: reason={reason}, representation={selection}.\r\nSelected model verifier error: Path.\r\nDevelopment backend launch failed: Artifacts.\r\nsynthetic-private-backend-detail\r\n"
            );
            for native_engine in [true, false] {
                let Failure::Message(message) =
                    startup_failure(diagnostic.as_bytes(), native_engine)
                else {
                    return Err(Failure::Message(
                        "Model failure lost its actionable message",
                    ));
                };
                assert!(message.contains(expected));
                assert!(!message.contains("synthetic-private-backend-detail"));
                assert!(!message.contains("bonsai-tools"));
                assert!(!message.contains("config/"));
            }
        }
        Ok(())
    }

    #[test]
    fn model_verification_guidance_requires_complete_known_lines_and_artifact_failure() {
        const MARKER: &str =
            "Selected model verification failed: reason=missing, representation=original.";
        for diagnostic in [
            MARKER.to_owned(),
            format!("{MARKER}\nDevelopment backend launch failed: Memory."),
            format!("prefix {MARKER}\nDevelopment backend launch failed: Artifacts."),
            format!("{MARKER} suffix\nDevelopment backend launch failed: Artifacts."),
            format!(
                "{}\nDevelopment backend launch failed: Artifacts.",
                MARKER.replace("missing", "unknown")
            ),
            format!(
                "{}\nDevelopment backend launch failed: Artifacts.",
                MARKER.replace("original", "unknown")
            ),
        ] {
            assert_eq!(super::model_verification_message(&diagnostic), None);
        }
        assert!(
            super::model_verification_message(&format!(
                "{MARKER}\nDevelopment backend launch failed: Artifacts."
            ))
            .is_some_and(|message| message.contains("model file is missing"))
        );
    }

    #[test]
    fn running_memory_failure_explains_resources_without_output_budget_guidance()
    -> Result<(), Failure> {
        let Failure::Message(message) =
            super::runtime_failure(b"Owned backend stopped after launcher failure: Memory.\n")
        else {
            return Err(Failure::Message(
                "Running memory failure lost its explicit resource guidance",
            ));
        };
        let guidance = super::retry_guidance(message);
        assert!(guidance.contains("memory"));
        assert!(guidance.contains("Close other apps"));
        assert!(guidance.contains("Retry model"));
        assert!(!guidance.contains("output budget"));
        assert!(!guidance.contains("request was too long"));
        assert!(
            matches!(super::runtime_failure(b"unrelated service exit"), Failure::Message(other) if other != message)
        );
        Ok(())
    }

    #[test]
    fn native_runtime_failure_requires_an_exact_native_broker_marker() {
        for line in [
            "Development backend launch failed: NativeRuntime.",
            "Development backend launch failed: NativeRuntime. Provision the pinned Rust runtime with bonsai-tools provision, operation inference, using the same dataDirectory.",
        ] {
            assert_eq!(
                startup_failure(line.as_bytes(), true),
                Failure::NativeRuntime
            );
            assert_ne!(
                startup_failure(line.as_bytes(), false),
                Failure::NativeRuntime
            );
        }
        for diagnostic in [
            "compiler warning: Development backend launch failed: NativeRuntime.",
            "Development backend launch failed: NativeRuntime.unrecognized",
            "Development backend launch failed: Artifacts.",
        ] {
            assert_ne!(
                startup_failure(diagnostic.as_bytes(), true),
                Failure::NativeRuntime
            );
        }
        assert!(matches!(
            startup_failure(b"Development backend launch failed: Artifacts.", true),
            Failure::Message(message) if message.starts_with("The selected Swift model")
        ));
    }

    #[test]
    fn native_runtime_repair_guidance_uses_in_app_setup_and_preserves_the_selected_data_directory()
    -> Result<(), String> {
        let data = std::path::Path::new("C:\\Local data\\Owner's Bonsai \u{1f331}");
        let guidance = super::native_runtime_guidance(data);
        assert!(guidance.contains(&data.display().to_string()));
        assert!(guidance.contains("Close and reopen Swift"));
        assert!(guidance.contains("Retry setup"));
        assert!(guidance.contains("repair its download automatically"));
        assert!(!guidance.contains("PowerShell"));
        assert!(!guidance.contains("provision --"));
        let lock: serde_json::Value =
            serde_json::from_str(include_str!("../../../config/prism-inference.lock.json"))
                .map_err(|error| error.to_string())?;
        assert!(guidance.contains(&format!(
            "Prism {}",
            lock["version"].as_str().ok_or("Missing version.")?
        )));
        Ok(())
    }

    #[test]
    fn memory_retry_requires_explicit_command_and_allows_shutdown() {
        let (commands, incoming) = std::sync::mpsc::channel();
        let (events, notifications) = crate::client::event_channel();
        let mut history = vec![Message {
            role: "user",
            content: "Previous conversation must not survive New conversation".into(),
            reasoning_content: None,
            images: super::Arc::default(),
        }];
        assert_eq!(
            startup_failure(b"Development backend launch failed: Memory.", false),
            Failure::Memory
        );
        assert!(matches!(
            startup_failure(b"Development backend launch failed: Artifacts.", false),
            Failure::Message(_)
        ));
        assert!(commands.send(Command::NewChat).is_ok());
        assert!(commands.send(Command::Stop).is_ok());
        assert!(
            commands
                .send(Command::Send {
                    text: "Never automatically replay this message".into(),
                    images: super::Arc::default(),
                    max_completion_tokens: 4096,
                    extra_effort: false,
                    enable_thinking: false,
                    abliteration_scale: super::DEFAULT_ABLITERATION_SCALE,
                    daisyrose_scale: super::DEFAULT_DAISYROSE_SCALE,
                    cognitive: super::CognitiveOptions::default(),
                    system_appendix: None,
                    turn_appendix: None,
                })
                .is_ok()
        );
        assert!(commands.send(Command::Retry).is_ok());
        assert!(wait_for_retry(
            &incoming,
            &events,
            &mut history,
            &Event::Retryable("Select Retry model.".to_owned())
        ));
        assert!(history.is_empty());
        assert!(matches!(notifications.try_recv(), Ok(Event::Retryable(_))));
        assert!(
            matches!(notifications.try_recv(), Ok(Event::Rejected { text, retry_required: true, .. })
            if text == "Never automatically replay this message")
        );
        assert!(matches!(notifications.try_recv(), Ok(Event::Retryable(_))));
        assert!(notifications.try_recv().is_err());
        assert!(commands.send(Command::Shutdown).is_ok());
        assert!(!wait_for_retry(
            &incoming,
            &events,
            &mut history,
            &Event::Retryable("Select Retry model.".to_owned())
        ));
    }

    #[test]
    fn rejected_and_stopped_turns_restore_composer_without_committing_partial_history()
    -> Result<(), Failure> {
        let (_commands, commands) = std::sync::mpsc::channel();
        let (events, incoming) = crate::client::event_channel();
        let mut history = vec![
            Message {
                role: "user",
                content: "Earlier task".into(),
                reasoning_content: None,
                images: super::Arc::default(),
            },
            Message {
                role: "assistant",
                content: "Earlier completed answer".into(),
                reasoning_content: Some("Retained reasoning".into()),
                images: super::Arc::default(),
            },
        ];
        let original = serialize_request(&history, 4096, false, None)?;
        for failure in [
            Failure::Rejected("Shorten the request."),
            Failure::Rejected(super::REQUEST_LIMIT_MESSAGE),
            Failure::Rejected(super::INVALID_REQUEST_MESSAGE),
            Failure::Rejected(super::CONTROL_TOKEN_MESSAGE),
            Failure::Rejected(super::RATE_LIMIT_MESSAGE),
            Failure::Message(super::TOKEN_COUNT_FAILURE_MESSAGE),
            Failure::Message(super::GENERATION_FAILURE_MESSAGE),
            Failure::Stopped,
            Failure::Message("Malformed response"),
        ] {
            let mut pending = history.clone();
            pending.push(Message {
                role: "user",
                content: "Unfinished task 🌱".into(),
                reasoning_content: None,
                images: super::Arc::default(),
            });
            let result = finish_turn(
                &mut history,
                pending,
                "Unfinished task 🌱",
                Err(failure),
                &events,
                &commands,
                true,
            );
            assert_eq!(
                result.is_ok(),
                matches!(failure, Failure::Rejected(_) | Failure::Stopped)
            );
            assert_eq!(serialize_request(&history, 4096, false, None)?, original);
            let event = incoming.try_recv();
            if failure == Failure::Stopped {
                assert!(matches!(event, Ok(Event::Stopping(text)) if text == "Unfinished task 🌱"));
                assert!(matches!(incoming.try_recv(), Ok(Event::Stopped)));
            } else {
                assert!(
                    matches!(event, Ok(Event::Rejected { text, retry_required, .. })
                    if text == "Unfinished task 🌱" && retry_required != matches!(failure, Failure::Rejected(_)))
                );
            }
            assert!(incoming.try_recv().is_err());
        }
        // A later, explicitly submitted request is appended only after framed
        // completion; the prior rejected/stopped attempt is never replayed.
        let mut pending = history.clone();
        pending.push(Message {
            role: "user",
            content: "Reviewed task".into(),
            reasoning_content: None,
            images: super::Arc::default(),
        });
        finish_turn(
            &mut history,
            pending,
            "Reviewed task",
            Ok((
                Message {
                    role: "assistant",
                    content: "New complete answer".into(),
                    reasoning_content: None,
                    images: super::Arc::default(),
                },
                Completion::Finished,
            )),
            &events,
            &commands,
            true,
        )?;
        assert_eq!(history.len(), 4);
        assert_eq!(history[2].content.as_ref(), "Reviewed task");
        assert_eq!(
            history[1].reasoning_content.as_deref(),
            Some("Retained reasoning")
        );
        assert!(
            matches!(incoming.try_recv(), Ok(Event::Committed(committed)) if committed.len() == 4 && committed[1].reasoning_content.as_deref() == Some("Retained reasoning"))
        );
        assert!(matches!(
            incoming.try_recv(),
            Ok(Event::Complete(Completion::Finished))
        ));
        Ok(())
    }

    #[test]
    fn long_unicode_requests_preserve_complete_text_and_thinking_history() -> Result<(), Failure> {
        // These sizes exercise intake for staged context profiles. Only the
        // broker's independently supplied exact token count admits a request;
        // repetitions here are text fixtures, never claimed token counts.
        for repetitions in [65_536, 131_072] {
            let text = "\u{1f331}\"".repeat(repetitions);
            let history = [
                Message {
                    role: "user",
                    content: "Keep the approved scene.".into(),
                    reasoning_content: None,
                    images: super::Arc::default(),
                },
                Message {
                    role: "assistant",
                    content: "Understood.".into(),
                    reasoning_content: Some("Prior reasoning: \u{96e8}\\\n".into()),
                    images: super::Arc::default(),
                },
                Message {
                    role: "user",
                    content: text.as_str().into(),
                    reasoning_content: None,
                    images: super::Arc::default(),
                },
            ];
            for thinking in [false, true] {
                let body = serialize_request(&history, 1024, thinking, None)?;
                assert!(body.len() > 32_768);
                assert!(body.len() <= super::MAX_REQUEST);
                let json: serde_json::Value = serde_json::from_slice(&body)
                    .map_err(|_| Failure::Message("Long request roundtrip failed."))?;
                assert_eq!(json["messages"][0]["role"], "system");
                assert_eq!(json["messages"][3]["content"], text);
                assert_eq!(
                    json["messages"][2]["reasoning_content"],
                    history[1].reasoning_content.as_deref().unwrap_or_default()
                );
                assert_eq!(json["enable_thinking"], thinking);
                assert_eq!(json["max_completion_tokens"], 1024);
            }
            assert_eq!(history[2].content.as_ref(), text);
        }
        Ok(())
    }

    #[test]
    fn escaped_request_overflow_returns_complete_draft_without_committing_history()
    -> Result<(), Failure> {
        let mut history = vec![
            Message {
                role: "user",
                content: "Saved task".into(),
                reasoning_content: None,
                images: super::Arc::default(),
            },
            Message {
                role: "assistant",
                content: "Saved answer".into(),
                reasoning_content: Some("Saved reasoning".into()),
                images: super::Arc::default(),
            },
        ];
        let original = serialize_request(&history, 1024, false, None)?;
        // Decoded text fits the byte cap, but JSON escaping does not.
        let text = "\n".repeat(super::MAX_REQUEST / 2);
        assert!(text.len() < super::MAX_REQUEST);
        let mut pending = history.clone();
        pending.push(Message {
            role: "user",
            content: text.as_str().into(),
            reasoning_content: None,
            images: super::Arc::default(),
        });
        let failure = serialize_request(&pending, 1024, false, None);
        assert_eq!(failure.err(), Some(super::request_byte_limit()));
        let (_commands, commands) = std::sync::mpsc::channel();
        let (events, incoming) = super::event_channel();
        finish_turn(
            &mut history,
            pending,
            &text,
            Err(super::request_byte_limit()),
            &events,
            &commands,
            true,
        )?;
        assert_eq!(serialize_request(&history, 1024, false, None)?, original);
        assert!(matches!(
            incoming.try_recv(),
            Ok(Event::Rejected { text: retained, retry_required: false, .. }) if retained == text
        ));
        assert!(incoming.try_recv().is_err());
        Ok(())
    }

    #[test]
    fn request_limits_are_recoverable_and_output_budget_is_bounded() -> Result<(), Failure> {
        let history = [Message {
            role: "user",
            content: "Task".into(),
            reasoning_content: None,
            images: super::Arc::default(),
        }];
        for budget in [1, 512, 1024, 2048, 4096] {
            let body = serialize_request(&history, budget, false, None)?;
            let json: serde_json::Value =
                serde_json::from_slice(&body).map_err(|_| Failure::Message("test request JSON"))?;
            assert_eq!(json["max_completion_tokens"], budget);
        }
        for budget in [0, 4097, u32::MAX] {
            assert!(matches!(
                serialize_request(&history, budget, false, None),
                Err(Failure::Rejected(_))
            ));
        }
        assert!(serialize_request(&vec![history[0].clone(); 127], 4096, false, None).is_ok());
        assert!(matches!(
            serialize_request(&vec![history[0].clone(); 128], 4096, false, None),
            Err(Failure::Rejected(_))
        ));
        let oversized = [Message {
            role: "user",
            content: "x".repeat(super::MAX_REQUEST).into(),
            reasoning_content: None,
            images: super::Arc::default(),
        }];
        assert!(matches!(
            serialize_request(&oversized, 4096, false, None),
            Err(Failure::Rejected(_))
        ));
        Ok(())
    }

    #[test]
    fn startup_input_modalities_recognizes_text_and_vision_declarations() {
        use super::input_modalities;
        assert_eq!(
            input_modalities(b"Input modalities: text, image\n"),
            Ok(Some(true))
        );
        assert_eq!(
            input_modalities(b"Input modalities: text\r\n"),
            Ok(Some(false))
        );
        assert_eq!(input_modalities(b"Context tokens: 8192\n"), Ok(None));
        assert_eq!(
            input_modalities(b"other output\nInput modalities: text, imag"),
            Ok(None)
        );
        for invalid in ["", "image", "text, audio", "text, image, audio", "text "] {
            assert!(input_modalities(format!("Input modalities: {invalid}\n").as_bytes()).is_err());
        }
        assert!(input_modalities(b"Input modalities: text\nInput modalities: text\n").is_err());
        assert!(
            input_modalities(b"Input modalities: text\nInput modalities: text, image\n").is_err()
        );
    }

    #[test]
    fn startup_daisyrose_adapter_recognizes_available_default_scales_and_unavailable() {
        use super::{DaisyRoseReadiness, daisyrose_adapter};

        assert_eq!(
            daisyrose_adapter(b"DaisyRose adapter: unavailable\r\n"),
            Ok(Some(DaisyRoseReadiness {
                available: false,
                default_scale_tenths: 0,
                max_rank: None,
                sha256_prefix: None,
            }))
        );
        for (text, tenths) in [("0.0", 0), ("1.0", 10), ("1.3", 13), ("2.0", 20)] {
            let line = format!("DaisyRose adapter: available, default scale {text}\n");
            let parsed = daisyrose_adapter(line.as_bytes());
            assert_eq!(
                parsed,
                Ok(Some(DaisyRoseReadiness {
                    available: true,
                    default_scale_tenths: tenths,
                    max_rank: None,
                    sha256_prefix: None,
                }))
            );
        }
        let full_sha = format!("1b850494b875{}", "a".repeat(52));
        let with_meta = format!(
            "DaisyRose adapter: available, default scale 1.0, rank 16, sha256 {full_sha}\n"
        );
        assert_eq!(
            daisyrose_adapter(with_meta.as_bytes()),
            Ok(Some(DaisyRoseReadiness {
                available: true,
                default_scale_tenths: 10,
                max_rank: Some(16),
                sha256_prefix: Some("1b850494b875".to_owned()),
            }))
        );
        assert_eq!(
            daisyrose_adapter(b"DaisyRose adapter: available, default scale 0.0, rank 32\n"),
            Ok(Some(DaisyRoseReadiness {
                available: true,
                default_scale_tenths: 0,
                max_rank: Some(32),
                sha256_prefix: None,
            }))
        );
        assert_eq!(daisyrose_adapter(b"Context tokens: 8192\n"), Ok(None));
        assert_eq!(
            daisyrose_adapter(b"other\nDaisyRose adapter: available, default scale 1.0"),
            Ok(None)
        );
        for invalid in [
            "",
            "available",
            "available, default scale 2.1",
            "available, default scale -0.0",
            "available, default scale 1.00",
            "available, default scale 1",
            "available, default scale 1.0, rank 0",
            "available, default scale 1.0, rank 65",
            "available, default scale 1.0, rank 16, rank 8",
            "available, default scale 1.0, sha256 1b850494b87",
            "available, default scale 1.0, sha256 1B850494B875",
            &format!("available, default scale 1.0, sha256 {}", "a".repeat(65)),
            "available, default scale 1.0, sha256 1b850494b875, sha256 1b850494b875",
            "available, default scale 1.0, bogus 1",
            "unavailable ",
        ] {
            assert!(
                daisyrose_adapter(format!("DaisyRose adapter: {invalid}\n").as_bytes()).is_err(),
                "expected rejection for {invalid:?}"
            );
        }
        assert!(
            daisyrose_adapter(b"DaisyRose adapter: unavailable\nDaisyRose adapter: unavailable\n")
                .is_err()
        );
        assert!(
            daisyrose_adapter(
                b"DaisyRose adapter: unavailable\nDaisyRose adapter: available, default scale 1.0\n"
            )
            .is_err()
        );
    }

    #[test]
    fn startup_cognitive_controls_declaration_and_gated_serialization() -> Result<(), Failure> {
        use crate::client::{
            CognitiveOptions, ExtraEffortStrategy, ReasoningEffort, ResponseOptions,
            SamplingPreset, ThinkingBudgetMode, cognitive_controls_declaration,
            serialize_request_with_options,
        };

        assert_eq!(
            cognitive_controls_declaration(b"Cognitive controls: available\n"),
            Ok(Some(true))
        );
        assert_eq!(
            cognitive_controls_declaration(b"Cognitive controls: unavailable\r\n"),
            Ok(Some(false))
        );
        assert_eq!(
            cognitive_controls_declaration(b"Context tokens: 8192\n"),
            Ok(None)
        );
        assert!(cognitive_controls_declaration(b"Cognitive controls: maybe\n").is_err());
        let history = [Message {
            role: "user",
            content: "rain".into(),
            reasoning_content: None,
            images: super::Arc::default(),
        }];
        let baseline = serialize_request(&history, 2048, true, None)?;
        let disabled = serialize_request_with_options(
            &history,
            &ResponseOptions {
                max_completion_tokens: 2048,
                enable_thinking: true,
                cognitive_controls: false,
                cognitive: CognitiveOptions {
                    reasoning_effort: ReasoningEffort::Low,
                    thinking_budget: ThinkingBudgetMode::Concise,
                    sampling_preset: SamplingPreset::BalancedDry,
                    extra_effort_strategy: ExtraEffortStrategy::MultiDraftSynthesis,
                },
                request_logprobs: true,
                temperature_nudge: 0.1,
                ..ResponseOptions::default()
            },
        )?;
        assert_eq!(baseline, disabled);

        let enabled = serialize_request_with_options(
            &history,
            &ResponseOptions {
                max_completion_tokens: 2048,
                enable_thinking: true,
                cognitive_controls: true,
                cognitive: CognitiveOptions {
                    reasoning_effort: ReasoningEffort::Medium,
                    thinking_budget: ThinkingBudgetMode::Balanced,
                    sampling_preset: SamplingPreset::BalancedDry,
                    extra_effort_strategy: ExtraEffortStrategy::MultiDraftSynthesis,
                },
                system_appendix: Some("Pinned card".into()),
                request_logprobs: true,
                temperature_nudge: 0.1,
                ..ResponseOptions::default()
            },
        )?;
        let parsed: serde_json::Value = serde_json::from_slice(&enabled)
            .map_err(|_| Failure::Message("invalid cognitive request JSON"))?;
        assert_eq!(parsed["reasoning_effort"], "medium");
        assert_eq!(parsed["reasoning_budget"], 1228);
        assert_eq!(parsed["dry_multiplier"], serde_json::json!(0.8));
        assert_eq!(parsed["logprobs"], true);
        assert_eq!(parsed["top_logprobs"], 5);
        assert!(
            parsed["messages"][0]["content"]
                .as_str()
                .is_some_and(|s| s.ends_with("\n\nPinned card"))
        );

        let multi_turn = [
            Message {
                role: "user",
                content: "First prompt".into(),
                reasoning_content: None,
                images: super::Arc::default(),
            },
            Message {
                role: "assistant",
                content: "First answer".into(),
                reasoning_content: None,
                images: super::Arc::default(),
            },
            Message {
                role: "user",
                content: "Second prompt".into(),
                reasoning_content: None,
                images: super::Arc::default(),
            },
        ];
        let without_trigger = serialize_request_with_options(
            &multi_turn,
            &ResponseOptions {
                max_completion_tokens: 2048,
                enable_thinking: true,
                system_appendix: Some("Pinned card".into()),
                ..ResponseOptions::default()
            },
        )?;
        let with_trigger = serialize_request_with_options(
            &multi_turn,
            &ResponseOptions {
                max_completion_tokens: 2048,
                enable_thinking: true,
                system_appendix: Some("Pinned card".into()),
                turn_appendix: Some("[Context: Rust]\nUse strict types.".into()),
                ..ResponseOptions::default()
            },
        )?;
        let json_plain: serde_json::Value = serde_json::from_slice(&without_trigger)
            .map_err(|_| Failure::Message("invalid baseline JSON"))?;
        let json_trig: serde_json::Value = serde_json::from_slice(&with_trigger)
            .map_err(|_| Failure::Message("invalid triggered JSON"))?;
        assert_eq!(json_plain["messages"][0], json_trig["messages"][0]);
        assert_eq!(json_plain["messages"][1], json_trig["messages"][1]);
        assert_eq!(json_plain["messages"][2], json_trig["messages"][2]);
        assert_eq!(
            json_trig["messages"][3]["content"],
            "[Context: Rust]\nUse strict types.\n\nSecond prompt"
        );
        assert_eq!(multi_turn[2].content.as_ref(), "Second prompt");

        let multimodal_tail = [Message {
            role: "user",
            content: "Check screenshot".into(),
            reasoning_content: None,
            images: super::Arc::from([super::Arc::from("data:image/png;base64,AAAA")]),
        }];
        let multimodal_body = serialize_request_with_options(
            &multimodal_tail,
            &ResponseOptions {
                max_completion_tokens: 2048,
                enable_thinking: true,
                system_appendix: Some("Pinned card".into()),
                turn_appendix: Some("[Context: Vision]\nInspect details.".into()),
                ..ResponseOptions::default()
            },
        )?;
        let json_mm: serde_json::Value = serde_json::from_slice(&multimodal_body)
            .map_err(|_| Failure::Message("invalid multimodal trigger JSON"))?;
        assert_eq!(json_mm["messages"][0], json_plain["messages"][0]);
        assert_eq!(
            json_mm["messages"][1]["content"],
            serde_json::json!([
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                {"type": "text", "text": "[Context: Vision]\nInspect details.\n\nCheck screenshot"}
            ])
        );
        Ok(())
    }

    #[test]
    fn multimodal_request_serializes_image_parts_and_enforces_24_image_and_16_mib_limits()
    -> Result<(), Failure> {
        use super::Arc;

        let image_url: Arc<str> = Arc::from("data:image/png;base64,AAAA");
        let with_caption = [Message {
            role: "user",
            content: "Describe this".into(),
            reasoning_content: None,
            images: Arc::from([Arc::clone(&image_url)]),
        }];
        let json_bytes = serde_json::to_vec(&with_caption[0])
            .map_err(|_| Failure::Message("serialize stored multimodal message"))?;
        let restored: Message = serde_json::from_slice(&json_bytes)
            .map_err(|_| Failure::Message("deserialize stored multimodal message"))?;
        assert_eq!(restored.images.as_ref(), &[Arc::clone(&image_url)]);

        assert!(
            serde_json::from_slice::<Message>(
                br#"{"role":"assistant","content":"hi","images":["data:image/png;base64,AAAA"]}"#,
            )
            .is_err()
        );

        let body = serialize_request(&with_caption, 512, false, None)?;
        let json: serde_json::Value =
            serde_json::from_slice(&body).map_err(|_| Failure::Message("test multimodal JSON"))?;
        assert_eq!(
            json["messages"][1]["content"],
            serde_json::json!([
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                {"type": "text", "text": "Describe this"}
            ])
        );

        let image_only = [Message {
            role: "user",
            content: "".into(),
            reasoning_content: None,
            images: Arc::from([Arc::clone(&image_url)]),
        }];
        let body = serialize_request(&image_only, 512, false, None)?;
        let json: serde_json::Value =
            serde_json::from_slice(&body).map_err(|_| Failure::Message("image-only JSON"))?;
        assert_eq!(
            json["messages"][1]["content"],
            serde_json::json!([
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
            ])
        );

        let large_image: Arc<str> = Arc::from(format!(
            "data:image/png;base64,{}",
            "A".repeat(2 * 1024 * 1024)
        ));
        let two_mib_turn = [Message {
            role: "user",
            content: "Large image".into(),
            reasoning_content: None,
            images: Arc::from([large_image]),
        }];
        assert!(serialize_request(&two_mib_turn, 512, false, None).is_ok());

        let too_many_images = [Message {
            role: "user",
            content: "Too many".into(),
            reasoning_content: None,
            images: vec![Arc::clone(&image_url); super::MAX_REQUEST_IMAGES + 1].into(),
        }];
        assert!(matches!(
            serialize_request(&too_many_images, 512, false, None),
            Err(Failure::Rejected(_))
        ));

        let oversized_multimodal = [Message {
            role: "user",
            content: "".into(),
            reasoning_content: None,
            images: Arc::from([Arc::from(format!(
                "data:image/png;base64,{}",
                "A".repeat(super::MAX_MULTIMODAL_REQUEST)
            ))]),
        }];
        assert_eq!(
            serialize_request(&oversized_multimodal, 512, false, None).err(),
            Some(super::multimodal_request_byte_limit())
        );
        Ok(())
    }

    #[test]
    fn read_api_key_requires_containment_and_retained_lowercase_hex_credential() -> io::Result<()> {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(io::Error::other)?
            .as_nanos();
        let root =
            std::env::temp_dir().join(format!("Bonsai-ReadApiKey-{}-{stamp}", std::process::id()));
        let evidence = root.join("evidence");
        let outside = root.join("outside");
        std::fs::create_dir_all(&evidence)?;
        std::fs::create_dir(&outside)?;
        let creds = bonsai_windows_credentials::PrivateCredentials::create(&evidence)?;
        let opened = super::read_api_key(&evidence, creds.directory())
            .map_err(|_| io::Error::other("Valid contained credential was rejected"))?;
        assert_eq!(opened.as_hex(), creds.frontend_key());

        let escaping = super::read_api_key(&evidence, &outside);
        assert_eq!(
            escaping.err(),
            Some(Failure::Message(
                "The broker reported an unexpected credential location."
            ))
        );

        let same_dir = super::read_api_key(&evidence, &evidence);
        assert_eq!(
            same_dir.err(),
            Some(Failure::Message(
                "The broker reported an unexpected credential location."
            ))
        );

        let missing_dir = evidence.join("missing");
        std::fs::create_dir(&missing_dir)?;
        assert_eq!(
            super::read_api_key(&evidence, &missing_dir).err(),
            Some(Failure::Message(
                "A broker diagnostic or credential file is unavailable."
            ))
        );

        let malformed_dir = evidence.join("malformed");
        std::fs::create_dir(&malformed_dir)?;
        std::fs::write(malformed_dir.join("frontend.key"), b"not-a-valid-key")?;
        assert_eq!(
            super::read_api_key(&evidence, &malformed_dir).err(),
            Some(Failure::Message(
                "The broker frontend credential has an invalid format."
            ))
        );

        drop(opened);
        drop(creds);
        let _removed = std::fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn native_identity_guidance_maps_all_closed_reasons_and_rejects_partial_or_publisher_lines()
    -> Result<(), Failure> {
        let reasons = [
            ("release-pin", "version check failed"),
            ("framing", "response format"),
            ("schema", "unsupported readiness schema"),
            ("model", "model identity"),
            ("context-tokens", "context window"),
            ("row-workers", "worker count"),
            ("prompt-reuse", "prompt-reuse setting"),
            ("kv-cache-graph", "KV-cache or adapter graph identity"),
            ("placement", "accelerator placement"),
            ("startup-projection", "startup projection"),
            ("adapter", "LoRA adapter identity"),
            ("vision", "vision projector identity"),
        ];
        let mut seen = std::collections::BTreeSet::new();
        for (reason, expected) in reasons {
            let diagnostic = format!(
                "Native backend startup failure: stage=native-identity, error=Backend, process=running\r\nNative identity verification failed: reason={reason}.\r\nDevelopment backend launch failed: Backend.\r\nsynthetic-private-detail\r\n"
            );
            let Failure::Message(message) = startup_failure(diagnostic.as_bytes(), true) else {
                return Err(Failure::Message(
                    "Native identity failure lost its actionable message",
                ));
            };
            assert!(message.contains(expected));
            assert!(message.contains("instead of retrying"));
            assert!(message.contains("Completed conversation turns remain in memory."));
            assert!(!message.contains("synthetic-private-detail"));
            let guidance = super::retry_guidance(message);
            assert_eq!(guidance, message);
            assert!(!guidance.contains("Select Retry model"));
            assert!(seen.insert(message));

            let generic = startup_failure(diagnostic.as_bytes(), false);
            assert_eq!(
                generic,
                startup_failure(b"Development backend launch failed: Backend.", false)
            );
        }
        assert_eq!(seen.len(), reasons.len());

        const MARKER: &str = "Native identity verification failed: reason=kv-cache-graph.";
        for bad in [
            MARKER.to_owned(),
            format!("{MARKER}\nDevelopment backend launch failed: Memory."),
            format!("prefix {MARKER}\nDevelopment backend launch failed: Backend."),
            format!("{MARKER} suffix\nDevelopment backend launch failed: Backend."),
            "Native identity verification failed: reason=unknown.\nDevelopment backend launch failed: Backend."
                .to_owned(),
        ] {
            assert_eq!(super::native_identity_message(&bad), None);
        }
        Ok(())
    }

    #[test]
    fn startup_policy_path_credentials_and_arguments_failures_give_specific_guidance()
    -> Result<(), Failure> {
        let generic = startup_failure(b"Development backend launch failed: Backend.", true);
        let cases = [
            ("Policy", "launch policy rejected"),
            ("Path", "links or redirects"),
            ("Credentials", "links or redirects"),
            ("Arguments", "disagreed on launch options"),
        ];
        for (variant, expected) in cases {
            let diagnostic =
                format!("synthetic preamble\r\nDevelopment backend launch failed: {variant}.\r\n");
            for native_engine in [true, false] {
                let Failure::Message(message) =
                    startup_failure(diagnostic.as_bytes(), native_engine)
                else {
                    return Err(Failure::Message(
                        "Deterministic startup failure lost its actionable message",
                    ));
                };
                assert!(message.contains(expected));
                assert!(message.contains("Completed conversation turns remain in memory."));
                let guidance = super::retry_guidance(message);
                assert_eq!(guidance, message);
                assert!(!guidance.contains("Select Retry model"));
                assert_ne!(Failure::Message(message), generic);
            }
            let prefixed = format!("note: Development backend launch failed: {variant}.\n");
            assert_eq!(startup_failure(prefixed.as_bytes(), true), generic);
        }
        Ok(())
    }

    #[test]
    fn broker_log_evidence_attaches_only_nonempty_regular_stderr_log_for_details_and_copy()
    -> io::Result<()> {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(io::Error::other)?
            .as_nanos();
        let root =
            std::env::temp_dir().join(format!("Bonsai-LogEvidence-{}-{stamp}", std::process::id()));
        let attempt = root.join("attempt-0");
        let keys = attempt.join("SwiftBonsai-Keys-123");
        std::fs::create_dir_all(&keys)?;
        std::fs::write(keys.join("frontend.key"), b"00")?;

        let base = "Swift could not start.".to_owned();
        assert_eq!(
            super::attach_broker_log_evidence(base.clone(), &attempt),
            base
        );

        let log = attempt.join("broker.stderr.log");
        std::fs::write(&log, b"")?;
        assert_eq!(
            super::attach_broker_log_evidence(base.clone(), &attempt),
            base
        );

        std::fs::write(
            &log,
            b"Native backend startup failure: stage=native-identity, error=Backend, process=running\n",
        )?;
        let attached = super::attach_broker_log_evidence(base.clone(), &attempt);
        assert_eq!(
            attached,
            format!("{base}\n{}{}", super::EVIDENCE_PREFIX, log.display())
        );
        assert!(!attached.contains("SwiftBonsai-Keys"));

        let _removed = std::fs::remove_dir_all(root);
        Ok(())
    }
}
