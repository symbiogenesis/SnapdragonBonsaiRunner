//! Native chat startup policy and orchestration for the direct Rust launcher.

mod instance;
mod setup;

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use bonsai_artifacts::validate_external_path;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
enum GraphicsBackend {
    Dx12,
    Vulkan,
    Gl,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
enum PromptReuse {
    #[default]
    Disabled,
    Latest,
    PromptEndAndLatest,
}

impl PromptReuse {
    const fn enabled(self) -> bool {
        !matches!(self, Self::Disabled)
    }

    const fn name(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::Latest => "latest",
            Self::PromptEndAndLatest => "prompt-end-and-latest",
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
enum CpuPrefillGdn {
    Sequential,
    WyF32,
}

impl CpuPrefillGdn {
    const fn name(self) -> &'static str {
        match self {
            Self::Sequential => "sequential",
            Self::WyF32 => "wy-f32",
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
enum SpeculationType {
    None,
    Suffix,
}

impl SpeculationType {
    const fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Suffix => "suffix",
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
enum CpuFamily {
    X1,
    X2,
}

impl CpuFamily {
    const fn name(self) -> &'static str {
        match self {
            Self::X1 => "x1",
            Self::X2 => "x2",
        }
    }

    const fn default_native_row_workers(self) -> u8 {
        match self {
            Self::X1 | Self::X2 => 8,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
enum KvCache {
    F16,
    Rsq4,
    Rsq4K3v,
}

impl KvCache {
    const fn name(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::Rsq4 => "rsq4",
            Self::Rsq4K3v => "rsq4k3v",
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
enum Placement {
    Auto,
    Cpu,
}

impl Placement {
    const fn name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
enum PreparationStage {
    #[default]
    All,
    Model,
    Speech,
}

fn graphics_environment(backend: Option<GraphicsBackend>) -> BTreeMap<String, String> {
    let mut environment = BTreeMap::new();
    if let Some(backend) = backend {
        let value = match backend {
            GraphicsBackend::Dx12 => "dx12",
            GraphicsBackend::Vulkan => "vulkan",
            GraphicsBackend::Gl => "gl",
        };
        environment.insert("WGPU_BACKEND".into(), value.into());
    }
    environment
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "PascalCase", deny_unknown_fields)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "The launch boundary preserves independent user options."
)]
struct Request {
    data_directory: PathBuf,
    repository_root: PathBuf,
    #[serde(default)]
    working_directory: Option<PathBuf>,
    #[serde(default)]
    chat_path: Option<PathBuf>,
    #[serde(default)]
    broker_path: Option<PathBuf>,
    // Retain the checkout's runtime lock after the early formatter handoff.
    #[serde(default)]
    source_built_broker: bool,
    #[serde(default)]
    asr_path: Option<PathBuf>,
    #[serde(default)]
    tts_path: Option<PathBuf>,
    #[serde(default = "default_quantization")]
    tts_quantization: String,
    #[serde(default)]
    tts_threads: u16,
    #[serde(default)]
    use_abliterate_lora: bool,
    #[serde(default)]
    no_abliterate_lora: bool,
    #[serde(default, rename = "DaisyRoseLora")]
    daisyrose_lora: Option<PathBuf>,
    #[serde(default, rename = "NoDaisyRoseLora")]
    no_daisyrose_lora: bool,
    #[serde(default)]
    no_mmproj: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    use_repacked_ptq1: Option<bool>,
    #[serde(default, rename = "NariqVariant")]
    nariq_variant: Option<bonsai_cpu_policy::NariqVariant>,
    #[serde(default, rename = "Bonsai25Variant")]
    bonsai25_variant: Option<bonsai_cpu_policy::Bonsai25Variant>,
    #[serde(default = "default_native_engine")]
    native_engine: bool,
    #[serde(default)]
    require_broker_pipe: bool,
    #[serde(default, deserialize_with = "requested_native_row_workers")]
    native_row_workers: Option<u8>,
    #[serde(default)]
    prompt_reuse: PromptReuse,
    #[serde(
        default,
        deserialize_with = "requested_prompt_reuse_checkpoints",
        skip_serializing_if = "Option::is_none"
    )]
    prompt_reuse_checkpoints: Option<u8>,
    #[serde(
        default,
        deserialize_with = "requested_prompt_reuse_checkpoint_bytes",
        skip_serializing_if = "Option::is_none"
    )]
    prompt_reuse_checkpoint_bytes: Option<usize>,
    #[serde(
        default,
        deserialize_with = "requested_kv_cache",
        skip_serializing_if = "Option::is_none"
    )]
    kv_cache: Option<KvCache>,
    #[serde(
        default,
        deserialize_with = "requested_cpu_prefill_tokens",
        skip_serializing_if = "Option::is_none"
    )]
    cpu_prefill_tokens: Option<u8>,
    #[serde(
        default,
        deserialize_with = "requested_cpu_prefill_gdn",
        skip_serializing_if = "Option::is_none"
    )]
    cpu_prefill_gdn: Option<CpuPrefillGdn>,
    #[serde(
        default,
        deserialize_with = "requested_spec_type",
        skip_serializing_if = "Option::is_none"
    )]
    spec_type: Option<SpeculationType>,
    #[serde(
        default,
        deserialize_with = "requested_draft_max",
        skip_serializing_if = "Option::is_none"
    )]
    draft_max: Option<u8>,
    #[serde(
        default,
        deserialize_with = "requested_cpu_family",
        skip_serializing_if = "Option::is_none"
    )]
    cpu_family: Option<CpuFamily>,
    #[serde(
        default,
        deserialize_with = "requested_placement",
        skip_serializing_if = "Option::is_none"
    )]
    placement: Option<Placement>,
    #[serde(default)]
    graphics_backend: Option<GraphicsBackend>,
    #[serde(default = "default_frontend_port")]
    frontend_port: u16,
    #[serde(default = "default_backend_port")]
    backend_port: u16,
    #[serde(default)]
    canary_key_file: Option<PathBuf>,
    #[serde(default = "default_canary_port")]
    canary_port: u16,
    #[serde(default)]
    tts_key_file: Option<PathBuf>,
    #[serde(default = "default_tts_port")]
    tts_port: u16,
    #[serde(default)]
    no_tts: bool,
    #[serde(default)]
    no_speech_models: bool,
    #[serde(default)]
    speech_models_only: bool,
    #[serde(default)]
    force_build: bool,
    #[serde(default)]
    preparation_stage: PreparationStage,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    portable_executable: Option<PathBuf>,
}

fn default_quantization() -> String {
    "Q8_0".into()
}

const fn default_native_engine() -> bool {
    true
}

fn requested_native_row_workers<'de, D>(deserializer: D) -> Result<Option<u8>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    // Omission selects the engine's default; an explicit null remains invalid.
    u8::deserialize(deserializer).map(Some)
}

fn requested_kv_cache<'de, D>(deserializer: D) -> Result<Option<KvCache>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    KvCache::deserialize(deserializer).map(Some)
}

fn requested_cpu_prefill_tokens<'de, D>(deserializer: D) -> Result<Option<u8>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let tokens = u8::deserialize(deserializer)?;
    if matches!(tokens, 8 | 16 | 32) {
        Ok(Some(tokens))
    } else {
        Err(serde::de::Error::custom(
            "CpuPrefillTokens must be 8, 16 or 32.",
        ))
    }
}

fn requested_placement<'de, D>(deserializer: D) -> Result<Option<Placement>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Placement::deserialize(deserializer).map(Some)
}

fn requested_prompt_reuse_checkpoints<'de, D>(deserializer: D) -> Result<Option<u8>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let count = u8::deserialize(deserializer)?;
    if (1..=8).contains(&count) {
        Ok(Some(count))
    } else {
        Err(serde::de::Error::custom(
            "PromptReuseCheckpoints must be between 1 and 8.",
        ))
    }
}

fn requested_prompt_reuse_checkpoint_bytes<'de, D>(
    deserializer: D,
) -> Result<Option<usize>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let bytes = usize::deserialize(deserializer)?;
    if bytes > 0 {
        Ok(Some(bytes))
    } else {
        Err(serde::de::Error::custom(
            "PromptReuseCheckpointBytes must be greater than zero.",
        ))
    }
}

fn requested_cpu_prefill_gdn<'de, D>(deserializer: D) -> Result<Option<CpuPrefillGdn>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    CpuPrefillGdn::deserialize(deserializer).map(Some)
}

fn requested_spec_type<'de, D>(deserializer: D) -> Result<Option<SpeculationType>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    SpeculationType::deserialize(deserializer).map(Some)
}

fn requested_draft_max<'de, D>(deserializer: D) -> Result<Option<u8>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let count = u8::deserialize(deserializer)?;
    if (1..=8).contains(&count) {
        Ok(Some(count))
    } else {
        Err(serde::de::Error::custom(
            "DraftMax must be between 1 and 8.",
        ))
    }
}

fn requested_cpu_family<'de, D>(deserializer: D) -> Result<Option<CpuFamily>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    CpuFamily::deserialize(deserializer).map(Some)
}

impl Request {
    const fn effective_bonsai25_variant(&self) -> Option<bonsai_cpu_policy::Bonsai25Variant> {
        if let Some(variant) = self.bonsai25_variant {
            Some(variant)
        } else if self.native_engine
            && self.nariq_variant.is_none()
            && self.use_repacked_ptq1.is_none()
        {
            Some(bonsai_cpu_policy::Bonsai25Variant::Pq2)
        } else {
            None
        }
    }

    const fn effective_use_abliterate_lora(&self) -> bool {
        if self.no_abliterate_lora || self.nariq_variant.is_some() {
            false
        } else if self.use_abliterate_lora {
            true
        } else {
            self.effective_bonsai25_variant().is_none()
        }
    }

    const fn native_row_workers(&self) -> u8 {
        self.native_row_workers_for_family(None)
    }

    const fn native_row_workers_for_family(
        &self,
        family: Option<bonsai_cpu_policy::hardware::SnapdragonFamily>,
    ) -> u8 {
        match self.native_row_workers {
            Some(workers) => workers,
            None if self.native_engine => match self.cpu_family {
                Some(explicit) => explicit.default_native_row_workers(),
                None => bonsai_cpu_policy::hardware::SnapdragonFamily::default_native_row_workers(
                    family,
                ),
            },
            None => 1,
        }
    }

    fn validate_native_runtime(
        &mut self,
        family: Option<bonsai_cpu_policy::hardware::SnapdragonFamily>,
    ) -> crate::Result<()> {
        let workers = self.native_row_workers_for_family(family);
        if !matches!(workers, 1 | 4 | 8 | 12) || (workers != 1 && !self.native_engine) {
            return Err(
                "NativeRowWorkers must be 1, 4, 8 or 12; parallel workers require NativeEngine."
                    .into(),
            );
        }
        if self.require_broker_pipe && !self.native_engine {
            return Err("RequireBrokerPipe requires NativeEngine.".into());
        }
        if self.prompt_reuse.enabled() && !self.native_engine {
            return Err("PromptReuse requires NativeEngine.".into());
        }
        if (self.prompt_reuse_checkpoints.is_some() || self.prompt_reuse_checkpoint_bytes.is_some())
            && (!self.native_engine || !self.prompt_reuse.enabled())
        {
            return Err(
                "PromptReuseCheckpoints and PromptReuseCheckpointBytes require NativeEngine and enabled PromptReuse."
                    .into(),
            );
        }
        if self.kv_cache.is_some() && !self.native_engine {
            return Err("KvCache requires NativeEngine.".into());
        }
        if let Some(tokens) = self.cpu_prefill_tokens {
            if !self.native_engine {
                return Err("CpuPrefillTokens requires NativeEngine.".into());
            }
            if !matches!(tokens, 8 | 16 | 32) {
                return Err("CpuPrefillTokens must be 8, 16 or 32.".into());
            }
        }
        if self.cpu_prefill_gdn.is_some()
            && (!self.native_engine || self.cpu_prefill_tokens.is_none())
        {
            return Err("CpuPrefillGdn requires NativeEngine and CpuPrefillTokens.".into());
        }
        if self.cpu_prefill_gdn == Some(CpuPrefillGdn::WyF32) && self.kv_cache != Some(KvCache::F16)
        {
            return Err("CpuPrefillGdn wy-f32 requires KvCache f16.".into());
        }
        if (self.spec_type.is_some() || self.draft_max.is_some()) && !self.native_engine {
            return Err("CPU speculation options require NativeEngine.".into());
        }
        if self.draft_max.is_some() && self.spec_type != Some(SpeculationType::Suffix) {
            return Err("DraftMax requires SpecType suffix.".into());
        }
        if self.spec_type == Some(SpeculationType::Suffix) && self.cpu_prefill_tokens.is_none() {
            return Err("SpecType suffix requires CpuPrefillTokens.".into());
        }
        if self.spec_type == Some(SpeculationType::Suffix)
            && self.cpu_prefill_gdn == Some(CpuPrefillGdn::WyF32)
        {
            return Err("CpuPrefillGdn wy-f32 cannot be combined with SpecType suffix.".into());
        }
        if self.cpu_family.is_some() && !self.native_engine {
            return Err("CpuFamily requires NativeEngine.".into());
        }
        if self.placement.is_some() && !self.native_engine {
            return Err("Placement requires NativeEngine.".into());
        }
        if self.no_daisyrose_lora && self.daisyrose_lora.is_some() {
            return Err("NoDaisyRoseLora cannot be combined with DaisyRoseLora.".into());
        }
        if let Some(path) = &self.daisyrose_lora {
            if !self.native_engine {
                return Err("DaisyRoseLora requires NativeEngine.".into());
            }
            external(path)?;
        }
        if self.use_abliterate_lora && self.no_abliterate_lora {
            return Err("UseAbliterateLora cannot be combined with NoAbliterateLora.".into());
        }
        if self.nariq_variant.is_some() {
            if self.use_abliterate_lora {
                return Err("UseAbliterateLora cannot be combined with NariqVariant.".into());
            }
            if self.use_repacked_ptq1.is_some() {
                return Err("NariqVariant cannot be combined with UseRepackedPtq1.".into());
            }
            if !self.native_engine {
                return Err("NariqVariant requires NativeEngine.".into());
            }
            if self.bonsai25_variant.is_some() {
                return Err("NariqVariant cannot be combined with Bonsai25Variant.".into());
            }
        }
        if self.bonsai25_variant.is_some() {
            if self.use_repacked_ptq1.is_some() {
                return Err("Bonsai25Variant cannot be combined with UseRepackedPtq1.".into());
            }
            if !self.native_engine {
                return Err("Bonsai25Variant requires NativeEngine.".into());
            }
        }
        self.native_row_workers = Some(workers);
        Ok(())
    }
}

const fn default_frontend_port() -> u16 {
    5173
}

const fn default_backend_port() -> u16 {
    8081
}

const fn default_canary_port() -> u16 {
    8766
}

const fn default_tts_port() -> u16 {
    8767
}

struct Plan {
    request: Request,
    own_canary: bool,
    own_tts: bool,
    model_selection: Option<&'static str>,
    native_selection: Option<&'static str>,
    app_packages: Vec<&'static str>,
}

fn empty_path(path: &mut Option<PathBuf>) {
    if path
        .as_ref()
        .is_some_and(|value| value.as_os_str().is_empty())
    {
        *path = None;
    }
}

fn external(path: &Path) -> crate::Result<()> {
    validate_external_path(path).map_err(|error| error.to_string())
}

fn executable(path: &Path, name: &str) -> crate::Result<()> {
    external(path)?;
    if !path.is_file()
        || !path
            .extension()
            .and_then(std::ffi::OsStr::to_str)
            .is_some_and(|extension| extension.eq_ignore_ascii_case("exe"))
    {
        return Err(format!(
            "{name} must name an existing executable outside Git."
        ));
    }
    Ok(())
}

fn repository(path: &Path) -> crate::Result<()> {
    if !path.is_absolute()
        || path.to_str().is_none_or(|text| text.contains(['%', '\0']))
        || path
            .components()
            .any(|part| matches!(part, Component::CurDir | Component::ParentDir))
    {
        return Err("RepositoryRoot must be an expanded absolute path.".into());
    }
    Ok(())
}

const fn selection(canary: bool, tts: bool) -> Option<&'static str> {
    match (canary, tts) {
        (true, true) => Some("All"),
        (true, false) => Some("QwenAsr"),
        (false, true) => Some("QwenTts"),
        (false, false) => None,
    }
}

fn validate_paths(request: &Request) -> crate::Result<()> {
    repository(&request.repository_root)?;
    external(&request.data_directory)?;
    if request.data_directory.exists() && !request.data_directory.is_dir() {
        return Err("The selected data location is not a directory. Choose a writable local folder for Swift.".into());
    }
    if let Some(working) = &request.working_directory {
        external(working)?;
    }
    for (name, path) in [
        ("ChatPath", &request.chat_path),
        ("BrokerPath", &request.broker_path),
        ("AsrPath", &request.asr_path),
        ("TtsPath", &request.tts_path),
    ] {
        if let Some(path) = path {
            executable(path, name)?;
        }
    }
    for (name, key) in [
        ("CanaryKeyFile", &request.canary_key_file),
        ("TtsKeyFile", &request.tts_key_file),
    ] {
        if let Some(path) = key {
            external(path)?;
            if !path.is_file() {
                return Err(format!("{name} must name an existing external key file."));
            }
        }
    }
    Ok(())
}

fn observed_host_family() -> Option<bonsai_cpu_policy::hardware::SnapdragonFamily> {
    #[cfg(windows)]
    {
        std::thread::available_parallelism()
            .ok()
            .is_some_and(|cores| cores.get() >= 8)
            .then(|| bonsai_windows_process::native_build_host().ok())
            .flatten()
            .and_then(|host| {
                bonsai_cpu_policy::hardware::SnapdragonFamily::from_processor_name(
                    &host.processor_name,
                )
            })
    }
    #[cfg(not(windows))]
    {
        None
    }
}

#[cfg(test)]
fn prepare(request: Request, native_host: bool) -> crate::Result<Plan> {
    prepare_for_family(request, native_host, None)
}

fn prepare_for_family(
    mut request: Request,
    native_host: bool,
    family: Option<bonsai_cpu_policy::hardware::SnapdragonFamily>,
) -> crate::Result<Plan> {
    request.validate_native_runtime(family)?;
    if !request.speech_models_only && !native_host {
        return Err("Run chat in the native ARM64 executable on Windows ARM64.".into());
    }
    for path in [
        &mut request.chat_path,
        &mut request.broker_path,
        &mut request.asr_path,
        &mut request.tts_path,
        &mut request.daisyrose_lora,
        &mut request.canary_key_file,
        &mut request.tts_key_file,
    ] {
        empty_path(path);
    }
    request.tts_quantization = match request.tts_quantization.to_ascii_uppercase().as_str() {
        "Q8_0" => "Q8_0".into(),
        "Q4_K_M" => "Q4_K_M".into(),
        _ => return Err("TtsQuantization must be Q8_0 or Q4_K_M.".into()),
    };
    if request.tts_threads > 256 {
        return Err("TtsThreads must be between zero and 256.".into());
    }
    if [
        request.frontend_port,
        request.backend_port,
        request.canary_port,
        request.tts_port,
    ]
    .iter()
    .any(|port| *port < 1024)
    {
        return Err("Chat and speech ports must be between 1024 and 65535.".into());
    }
    if request.frontend_port == request.backend_port {
        return Err("FrontendPort and BackendPort must differ.".into());
    }
    if request.no_speech_models && request.speech_models_only {
        return Err("SpeechModelsOnly cannot be combined with NoSpeechModels.".into());
    }
    if request.speech_models_only && request.preparation_stage != PreparationStage::All {
        return Err("SpeechModelsOnly cannot select a chat preparation stage.".into());
    }
    if request.no_tts && request.tts_key_file.is_some() {
        return Err("NoTts cannot be combined with TtsKeyFile.".into());
    }
    validate_paths(&request)?;
    let own_canary = request.canary_key_file.is_none() && !request.no_speech_models;
    let own_tts = request.tts_key_file.is_none() && !request.no_tts && !request.no_speech_models;
    if !own_tts && (request.tts_quantization != "Q8_0" || request.tts_threads != 0) {
        return Err("TtsQuantization and TtsThreads require automatically owned TTS.".into());
    }
    let canary = own_canary || request.canary_key_file.is_some();
    let tts = own_tts || request.tts_key_file.is_some();
    let chat_ports = [request.frontend_port, request.backend_port];
    if (canary && chat_ports.contains(&request.canary_port))
        || (tts && chat_ports.contains(&request.tts_port))
        || (canary && tts && request.canary_port == request.tts_port)
    {
        return Err("Every configured speech service needs a distinct local port.".into());
    }
    if (request.asr_path.is_some() && !own_canary) || (request.tts_path.is_some() && !own_tts) {
        return Err(
            "A prebuilt speech executable requires automatic setup for that service.".into(),
        );
    }
    let model_selection = if request.speech_models_only {
        Some("All")
    } else {
        selection(own_canary, own_tts)
    };
    let native_selection = selection(
        own_canary && request.asr_path.is_none(),
        own_tts && request.tts_path.is_none(),
    );
    let mut app_packages = Vec::with_capacity(2);
    if request.chat_path.is_none() {
        app_packages.push("bonsai-chat");
    }
    if request.broker_path.is_none() {
        app_packages.push("bonsai-broker");
    }
    Ok(Plan {
        request,
        own_canary,
        own_tts,
        model_selection,
        native_selection,
        app_packages,
    })
}

trait Operations {
    fn check_sandbox(&mut self, _request: &Request) -> crate::Result<()> {
        Ok(())
    }
    fn artifacts(&mut self, request: &Request, working: &Path) -> crate::Result<()>;
    fn inference(&mut self, request: &Request, working: &Path) -> crate::Result<()>;
    fn model_apps(
        &mut self,
        request: &Request,
        packages: &[&str],
        working: &Path,
    ) -> crate::Result<Option<Value>> {
        if request.native_engine {
            self.inference(request, working)?;
        }
        if packages.is_empty() {
            Ok(None)
        } else {
            self.apps(request, packages, working).map(Some)
        }
    }
    fn models(&mut self, request: &Request, model: &str, working: &Path) -> crate::Result<Value>;
    fn apps(
        &mut self,
        request: &Request,
        packages: &[&str],
        working: &Path,
    ) -> crate::Result<Value>;
    fn native(&mut self, request: &Request, model: &str, working: &Path) -> crate::Result<Value>;
    fn keys(&mut self, parent: &Path) -> crate::Result<PathBuf>;
    fn launch(
        &mut self,
        executable: &Path,
        arguments: &[OsString],
        working: &Path,
        environment: &BTreeMap<String, String>,
    ) -> crate::Result<u32>;
}

struct Native;

fn field<'a>(value: &'a Value, key: &str) -> crate::Result<&'a str> {
    value[key]
        .as_str()
        .ok_or_else(|| format!("Startup operation omitted {key}."))
}

fn path_field(value: &Value, key: &str) -> crate::Result<PathBuf> {
    field(value, key).map(PathBuf::from)
}

#[cfg(any(windows, test))]
fn sandbox_preflight_error(
    error: &std::io::Error,
    offline_only: bool,
    reboot_pending: bool,
) -> String {
    if error.kind() == std::io::ErrorKind::InvalidInput {
        "The required native Windows sandbox rejected inherited service credentials in the parent environment. Close Swift and reopen it from a clean app or terminal session without service credentials."
            .to_owned()
    } else if reboot_pending {
        format!(
            "Required Windows sandbox (PSEC 1.1) is unavailable because a Windows Update restart is pending. Restart Windows to finish installing updates, then reopen Swift. No model files were downloaded. ({error})"
        )
    } else if offline_only {
        format!(
            "Required Windows sandbox (PSEC 1.1) is unavailable on this Windows build. Open Settings > Windows Update, install the latest Windows updates, restart Windows, and reopen Swift. No model files were downloaded. ({error})"
        )
    } else {
        format!(
            "Required Windows sandbox (PSEC 1.1 with container-hosted loopback) is unavailable on this Windows build. Open Settings > Windows Update, install the latest Windows updates, restart Windows, and reopen Swift. No model files were downloaded. ({error})"
        )
    }
}

impl Operations for Native {
    fn check_sandbox(&mut self, request: &Request) -> crate::Result<()> {
        if request.speech_models_only {
            return Ok(());
        }
        #[cfg(windows)]
        {
            if request.preparation_stage == PreparationStage::Speech {
                // Reject inherited credentials before downloading speech assets
                // without creating a throwaway PSEC container on cached launches.
                bonsai_windows_process::validate_sandbox_environment().map_err(|error| {
                    sandbox_preflight_error(
                        &error,
                        false,
                        bonsai_windows_process::windows_reboot_pending(),
                    )
                })?;
                return Ok(());
            }
            // Only probe a throwaway PSEC container on first-run setup before
            // downloading multi-gigabyte model/runtime artifacts. Cached launches
            // admit the real broker policy once without extra preflight overhead.
            if setup::kind(request)? == "first-run" {
                let offline_only = request.native_engine;
                let probe = if offline_only {
                    bonsai_windows_process::native_sandbox_offline_available()
                } else {
                    bonsai_windows_process::native_sandbox_available()
                };
                probe.map_err(|error| {
                    sandbox_preflight_error(
                        &error,
                        offline_only,
                        bonsai_windows_process::windows_reboot_pending(),
                    )
                })?;
            }
        }
        #[cfg(not(windows))]
        let _ = request;
        Ok(())
    }

    fn artifacts(&mut self, request: &Request, working: &Path) -> crate::Result<()> {
        setup::prepare(request, working)
    }

    fn inference(&mut self, request: &Request, working: &Path) -> crate::Result<()> {
        let kind = setup::kind(request)?;
        setup::runtime_status(
            working,
            kind,
            crate::startup_download::RepairActivity::Publishing,
            None,
        )?;
        crate::provision::prepare_chat_runtime_observed(inference_request(request, working),
            &mut |phase, artifact| setup::runtime_status(working, kind, phase, artifact))
            .map(|_| ())
            .map_err(|error| format!("The selected Rust runtime could not be prepared in {}: {error}. Check available disk space and folder permissions, then select Retry setup.", request.data_directory.display()))
    }

    fn model_apps(
        &mut self,
        request: &Request,
        packages: &[&str],
        working: &Path,
    ) -> crate::Result<Option<Value>> {
        if request.native_engine && !packages.is_empty() {
            // The broker build consumes the release lock, not the installed
            // runtime. Neither operation needs the other's output. Join both
            // before handing the broker to the UI; a failure cancels its peer.
            parallel_model_preparation(
                || Self.inference(request, working),
                || Self.apps(request, packages, working),
            )
            .map(Some)
        } else {
            if request.native_engine {
                self.inference(request, working)?;
            }
            if packages.is_empty() {
                Ok(None)
            } else {
                self.apps(request, packages, working).map(Some)
            }
        }
    }

    fn models(&mut self, request: &Request, model: &str, working: &Path) -> crate::Result<Value> {
        crate::native_startup::models(
            &request.data_directory,
            model,
            &request.tts_quantization,
            false,
            // Preparation owns repair before speech services start. Checking only
            // length can otherwise leave same-size corrupt files failing on every
            // service load, beyond the setup retry path.
            false,
            working,
        )
    }

    fn apps(
        &mut self,
        request: &Request,
        packages: &[&str],
        working: &Path,
    ) -> crate::Result<Value> {
        crate::source_build::run(json!({
            "operation": "chat", "repositoryRoot": request.repository_root,
            "dataDirectory": request.data_directory, "workingDirectory": working,
            "packages": packages, "forceBuild": request.force_build,
        }))
    }

    fn native(&mut self, request: &Request, model: &str, working: &Path) -> crate::Result<Value> {
        crate::native_startup::build(
            &request.repository_root,
            &request.data_directory,
            working,
            model,
            request.force_build,
        )
    }

    fn keys(&mut self, parent: &Path) -> crate::Result<PathBuf> {
        path_field(
            &crate::credentials::run(json!({"parentDirectory": parent}))?,
            "FrontendKeyFile",
        )
    }

    fn launch(
        &mut self,
        executable: &Path,
        arguments: &[OsString],
        working: &Path,
        environment: &BTreeMap<String, String>,
    ) -> crate::Result<u32> {
        let result = crate::startup_process::run(
            executable.as_os_str(),
            arguments,
            working,
            environment,
            Duration::ZERO,
            &working.join("chat.log"),
        )?;
        Ok(result.code)
    }
}

#[derive(Default)]
struct ModelPreparation {
    cancelled: Arc<AtomicBool>,
    failure: Mutex<Option<String>>,
}

impl ModelPreparation {
    fn fail(&self, error: String) {
        if let Ok(mut failure) = self.failure.lock()
            && failure.is_none()
        {
            *failure = Some(error);
        }
        self.cancelled.store(true, Ordering::Release);
    }

    fn run<T>(&self, operation: impl FnOnce() -> crate::Result<T>) -> crate::Result<T> {
        let mut guard = PreparationArm {
            group: self,
            finished: false,
        };
        let result = crate::startup_cancel::with_peer_cancellation(&self.cancelled, || {
            crate::startup_cancel::checkpoint().and_then(|()| operation())
        });
        if let Err(error) = &result {
            self.fail(error.clone());
        }
        guard.finished = true;
        result
    }
}

struct PreparationArm<'a> {
    group: &'a ModelPreparation,
    finished: bool,
}

impl Drop for PreparationArm<'_> {
    fn drop(&mut self) {
        if !self.finished {
            self.group.fail("Model preparation worker panicked.".into());
        }
    }
}

fn parallel_model_preparation(
    inference: impl FnOnce() -> crate::Result<()> + Send,
    apps: impl FnOnce() -> crate::Result<Value>,
) -> crate::Result<Value> {
    let group = ModelPreparation::default();
    let built = std::thread::scope(|scope| {
        let runtime = std::thread::Builder::new()
            .name("chat-runtime-prepare".into())
            .spawn_scoped(scope, || group.run(inference))
            .map_err(|error| format!("Cannot start runtime preparation: {error}"))?;
        let built = group.run(apps);
        match runtime.join() {
            Ok(Ok(())) => {}
            Ok(Err(error)) => group.fail(error),
            Err(_) => group.fail("Model preparation worker panicked.".into()),
        }
        built
    });
    if let Some(error) = group
        .failure
        .into_inner()
        .map_err(|_| "Model preparation result was poisoned.")?
    {
        return Err(error);
    }
    built
}

fn inference_request(request: &Request, working: &Path) -> Value {
    let mut provision = json!({
        "operation": "inference", "dataDirectory": request.data_directory,
        "workingDirectory": working,
    });
    // A source-built broker embeds this checkout's lock. Explicit prebuilt
    // brokers retain the maintenance executable's compiled package lock instead.
    if request.broker_path.is_none() || request.source_built_broker {
        provision["lockPath"] = json!(
            request
                .repository_root
                .join("config")
                .join("prism-inference.lock.json")
        );
    }
    provision
}

fn argument(arguments: &mut Vec<OsString>, name: &str, value: impl Into<OsString>) {
    arguments.push(name.into());
    arguments.push(value.into());
}

fn effective_daisyrose_lora_path(
    request: &Request,
    pinned: Option<&bonsai_cpu_policy::DaisyRoseLoraArtifact>,
) -> Option<PathBuf> {
    if request.no_daisyrose_lora || !request.native_engine {
        return None;
    }
    if let Some(explicit) = &request.daisyrose_lora {
        return Some(explicit.clone());
    }
    let expected_base_sha256 = setup::model(request).ok().map(|selected| selected.sha256);
    pinned
        .filter(|artifact| {
            artifact.default_enabled
                && expected_base_sha256.as_deref() == Some(artifact.base_model_sha256.as_str())
        })
        .map(|artifact| artifact.path.clone())
}

fn runtime_arguments(request: &Request, arguments: &mut Vec<OsString>) {
    if request.use_abliterate_lora {
        arguments.push("--use-abliterate-lora".into());
    } else if request.no_abliterate_lora || request.nariq_variant.is_some() {
        arguments.push("--no-abliterate-lora".into());
    }
    if request.no_daisyrose_lora {
        arguments.push("--no-daisyrose-lora".into());
    }
    let pinned_daisyrose = setup::model(request).ok().and_then(|selected| {
        bonsai_cpu_policy::DaisyRoseLoraArtifact::pinned_for_base_sha256(
            &request.data_directory,
            &selected.sha256,
        )
        .ok()
        .flatten()
    });
    if let Some(daisyrose_lora) = effective_daisyrose_lora_path(request, pinned_daisyrose.as_ref())
    {
        argument(arguments, "--daisyrose-lora", daisyrose_lora);
    }
    if request.no_mmproj || request.nariq_variant.is_some() {
        arguments.push("--no-mmproj".into());
    }
    if let Some(variant) = request.nariq_variant {
        argument(arguments, "--nariq", variant.as_str());
    } else if let Some(variant) = request.effective_bonsai25_variant() {
        argument(arguments, "--bonsai-2-5", variant.as_str());
    } else if request.use_repacked_ptq1 == Some(true) {
        arguments.push("--use-repacked-ptq1".into());
    } else {
        arguments.push("--use-original-pq2".into());
    }
    arguments.push(if request.native_engine {
        "--native-engine".into()
    } else {
        "--publisher-engine".into()
    });
    if request.require_broker_pipe {
        arguments.push("--require-broker-pipe".into());
    }
    argument(
        arguments,
        "--native-row-workers",
        request.native_row_workers().to_string(),
    );
    if let Some(family) = request.cpu_family {
        argument(arguments, "--cpu-family", family.name());
    }
    if request.prompt_reuse.enabled() {
        argument(arguments, "--prompt-reuse", request.prompt_reuse.name());
    }
    if let Some(checkpoints) = request.prompt_reuse_checkpoints {
        argument(
            arguments,
            "--prompt-reuse-checkpoints",
            checkpoints.to_string(),
        );
    }
    if let Some(bytes) = request.prompt_reuse_checkpoint_bytes {
        argument(
            arguments,
            "--prompt-reuse-checkpoint-bytes",
            bytes.to_string(),
        );
    }
    if let Some(policy) = request.kv_cache {
        argument(arguments, "--kv-cache", policy.name());
    }
    if let Some(tokens) = request.cpu_prefill_tokens {
        argument(arguments, "--cpu-prefill-tokens", tokens.to_string());
    }
    if let Some(gdn) = request.cpu_prefill_gdn {
        argument(arguments, "--cpu-prefill-gdn", gdn.name());
    }
    if let Some(spec_type) = request.spec_type {
        argument(arguments, "--spec-type", spec_type.name());
    }
    if let Some(draft_max) = request.draft_max {
        argument(arguments, "--draft-max", draft_max.to_string());
    }
    if let Some(placement) = request.placement {
        argument(arguments, "--placement", placement.name());
    }
}

fn launch_arguments(
    request: &Request,
    models: Option<&Value>,
    own_canary: bool,
    own_tts: bool,
    working: &Path,
) -> crate::Result<Vec<OsString>> {
    let broker = request
        .broker_path
        .as_ref()
        .ok_or("Missing chat broker executable.")?;
    let mut arguments = Vec::with_capacity(36);
    argument(&mut arguments, "--broker-path", broker);
    argument(&mut arguments, "--data-directory", &request.data_directory);
    argument(&mut arguments, "--evidence-directory", working);
    argument(
        &mut arguments,
        "--frontend-port",
        request.frontend_port.to_string(),
    );
    argument(
        &mut arguments,
        "--backend-port",
        request.backend_port.to_string(),
    );
    runtime_arguments(request, &mut arguments);
    if let Some(key) = &request.canary_key_file {
        argument(&mut arguments, "--canary-key-file", key);
        argument(
            &mut arguments,
            "--canary-port",
            request.canary_port.to_string(),
        );
    }
    if let Some(key) = &request.tts_key_file {
        argument(&mut arguments, "--tts-key-file", key);
        argument(&mut arguments, "--tts-port", request.tts_port.to_string());
    }
    if own_canary || own_tts {
        let models = models.ok_or("Missing owned speech model paths.")?;
        if own_canary {
            argument(
                &mut arguments,
                "--canary-executable",
                request
                    .asr_path
                    .as_ref()
                    .ok_or("Missing Qwen3-ASR executable.")?,
            );
            argument(
                &mut arguments,
                "--canary-model",
                field(models, "CanaryModelPath")?,
            );
        }
        if own_tts {
            argument(
                &mut arguments,
                "--tts-executable",
                request.tts_path.as_ref().ok_or("Missing TTS executable.")?,
            );
            argument(
                &mut arguments,
                "--tts-model",
                field(models, "TtsModelPath")?,
            );
            argument(
                &mut arguments,
                "--tts-codec",
                field(models, "TtsCodecPath")?,
            );
            argument(
                &mut arguments,
                "--tts-voice-model",
                field(models, "TtsVoiceModelPath")?,
            );
            argument(
                &mut arguments,
                "--tts-quantization",
                &request.tts_quantization,
            );
            if request.tts_threads != 0 {
                argument(
                    &mut arguments,
                    "--tts-threads",
                    request.tts_threads.to_string(),
                );
            }
        }
    }
    Ok(arguments)
}

fn prepare_backend(
    plan: Plan,
    operations: &mut impl Operations,
    working: &Path,
) -> crate::Result<Value> {
    let Plan {
        mut request,
        mut own_canary,
        mut own_tts,
        mut model_selection,
        mut native_selection,
        app_packages,
    } = plan;
    let stage = request.preparation_stage;
    if stage == PreparationStage::Model {
        // Chat must not wait for unrelated speech downloads or native builds.
        // External speech configurations also attach only with the speech stage.
        own_canary = false;
        own_tts = false;
        model_selection = None;
        native_selection = None;
        request.canary_key_file = None;
        request.tts_key_file = None;
    }
    if stage == PreparationStage::Speech && request.broker_path.is_none() {
        return Err("Speech preparation requires the prepared broker executable.".into());
    }
    fs::create_dir_all(&request.data_directory).map_err(|error| {
        format!("Swift cannot create its data folder {}: {error}. Choose a writable local folder and launch again.", request.data_directory.display())
    })?;
    external(&request.data_directory)?;
    operations.check_sandbox(&request)?;
    let model_apps = if stage == PreparationStage::Model && !request.speech_models_only {
        operations.model_apps(&request, &app_packages, working)?
    } else {
        None
    };
    if stage == PreparationStage::All && request.native_engine && !request.speech_models_only {
        // Run only in the owned preparation worker after the first UI frame.
        // Setup checks the exact release inventory and hashes and restores only
        // the selected pinned runtime before any model handoff.
        operations.inference(&request, working)?;
    }
    if stage != PreparationStage::Speech && !request.speech_models_only {
        operations.artifacts(&request, working)?;
    }
    let models = if let Some(model) = model_selection {
        Some(operations.models(&request, model, working)?)
    } else {
        None
    };
    if request.speech_models_only {
        return models.ok_or_else(|| "Missing explicit speech provisioning result.".into());
    }
    if stage != PreparationStage::Speech && !app_packages.is_empty() {
        let built = if let Some(built) = model_apps {
            built
        } else {
            operations.apps(&request, &app_packages, working)?
        };
        if request.chat_path.is_none() {
            request.chat_path = Some(path_field(&built, "ChatPath")?);
        }
        if request.broker_path.is_none() {
            request.broker_path = Some(path_field(&built, "BrokerPath")?);
        }
    }
    if let Some(model) = native_selection {
        let built = operations.native(&request, model, working)?;
        if own_canary && request.asr_path.is_none() {
            request.asr_path = Some(path_field(&built, "AsrPath")?);
        }
        if own_tts && request.tts_path.is_none() {
            request.tts_path = Some(path_field(&built, "TtsPath")?);
        }
    }
    for (owned, name, key) in [
        (own_canary, "canary", &mut request.canary_key_file),
        (own_tts, "tts", &mut request.tts_key_file),
    ] {
        if owned {
            let parent = working.join(format!("{name}-credentials"));
            fs::create_dir_all(&parent).map_err(|error| error.to_string())?;
            *key = Some(operations.keys(&parent)?);
        }
    }
    let arguments = launch_arguments(&request, models.as_ref(), own_canary, own_tts, working)?;
    let arguments = arguments
        .into_iter()
        .map(|argument| {
            argument
                .into_string()
                .map_err(|_| "Chat arguments must be Unicode.".to_owned())
        })
        .collect::<crate::Result<Vec<_>>>()?;
    Ok(json!({"Arguments": arguments}))
}

fn launch_pending(
    mut request: Request,
    operations: &mut impl Operations,
    working: &Path,
    tool: &Path,
) -> crate::Result<Value> {
    // Both images are needed before opening the window: the headless broker
    // also formats recovered answers while model setup is pending or fails.
    // Downloads, model preparation and speech stay in the owned UI worker.
    let mut packages = Vec::with_capacity(2);
    if request.chat_path.is_none() {
        packages.push("bonsai-chat");
    }
    if request.broker_path.is_none() {
        packages.push("bonsai-broker");
    }
    if !packages.is_empty() {
        let built = operations.apps(&request, &packages, working)?;
        if request.chat_path.is_none() {
            request.chat_path = Some(path_field(&built, "ChatPath")?);
        }
        if request.broker_path.is_none() {
            request.broker_path = Some(path_field(&built, "BrokerPath")?);
            request.source_built_broker = true;
        }
    }
    request.working_directory = Some(working.into());
    let path = working.join(format!(
        "chat-startup-{}.json",
        crate::startup_process::unique()
    ));
    external(&path)?;
    let file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .map_err(|error| error.to_string())?;
    serde_json::to_writer(file, &request).map_err(|error| error.to_string())?;
    let chat = request
        .chat_path
        .as_ref()
        .ok_or("Missing chat executable.")?;
    let mut arguments = Vec::with_capacity(6);
    argument(&mut arguments, "--startup-tool", tool);
    argument(&mut arguments, "--startup-request", &path);
    argument(
        &mut arguments,
        "--startup-data-directory",
        &request.data_directory,
    );
    let environment = graphics_environment(request.graphics_backend);
    let code = operations.launch(chat, &arguments, working, &environment)?;
    if code != 0 && code != crate::app_update::UPDATE_EXIT_CODE {
        return Err(launch_failure(code, working));
    }
    let mut result =
        json!({"Status": "chat-stopped", "ExitCode": code, "EvidenceDirectory": working});
    if code == crate::app_update::UPDATE_EXIT_CODE {
        let handoff = path.with_extension("update.json");
        if !fs::exists(&handoff).map_err(|error| error.to_string())? {
            return Err(
                "The native application requested an update without a verified handoff.".into(),
            );
        }
        result["UpdateHandoff"] = json!(handoff);
    }
    Ok(result)
}

fn launch_failure(code: u32, working: &Path) -> String {
    use std::io::Read as _;
    let mut log = String::new();
    if let Ok(file) = fs::File::open(working.join("chat.log")) {
        let _read = file.take(32 * 1024).read_to_string(&mut log);
    }
    let guidance = graphics_failure_guidance(code, &log).unwrap_or_default();
    format!(
        "Chat stopped with exit code {code}.{guidance} Evidence is retained under {}.",
        working.display()
    )
}

fn graphics_failure_guidance(code: u32, log: &str) -> Option<String> {
    let staging_failure = code == 101
        && log.contains("egui-wgpu-")
        && log.contains("renderer.rs:")
        && [
            "Failed to create staging buffer for index data.",
            "Failed to create staging buffer for vertex data.",
        ]
        .iter()
        .any(|marker| log.contains(marker));
    if !staging_failure {
        return None;
    }
    let backend = log.lines().rev().find_map(|line| {
        let entry: Value = serde_json::from_str(line).ok()?;
        if entry["kind"] == "bonsai-chat-graphics" && entry["stage"] == "adapter-selected" {
            entry["backend"].as_str().map(str::to_owned)
        } else {
            None
        }
    });
    let selection = match backend.as_deref() {
        Some("Vulkan") => {
            " Vulkan was selected. For an explicit DX12 attempt, set $env:WGPU_BACKEND='dx12'"
        }
        Some("Dx12") => {
            " DX12 was selected. For an explicit Vulkan attempt, set $env:WGPU_BACKEND='vulkan'"
        }
        _ => " For an explicit graphics backend attempt, set $env:WGPU_BACKEND='dx12' or 'vulkan'",
    };
    Some(format!(
        " The native window renderer failed to upload GUI geometry. The retained error does not diagnose inference.{selection} before rerunning the same command; restore your previous setting afterwards. See docs/development-guide.md#native-renderer-failures."
    ))
}

fn working_directory(requested: Option<&Path>) -> crate::Result<PathBuf> {
    if let Some(path) = requested {
        external(path)?;
        fs::create_dir_all(path).map_err(|error| error.to_string())?;
        return Ok(path.into());
    }
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| error.to_string())?
        .as_nanos();
    for attempt in 0..100 {
        let path = std::env::temp_dir().join(format!(
            "SwiftBonsai-Chat-{}-{nonce}-{attempt}",
            std::process::id()
        ));
        external(&path)?;
        match fs::create_dir(&path) {
            Ok(()) => return Ok(path),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(error.to_string()),
        }
    }
    Err("Cannot create a fresh external chat session directory.".into())
}

/// Opens native chat with offline formatting before preparing model and speech files.
///
/// Speech-only provisioning returns model paths without compiling or starting an
/// application. Ordinary startup checks cached speech lengths; native services
/// retain full compiled SHA-256 verification before their first model use.
///
/// # Errors
/// Rejects unsupported hosts, malformed or conflicting options, unsafe paths,
/// failed provisioning/builds, and a nonzero chat exit. Evidence remains external.
pub fn run(value: Value) -> crate::Result<Value> {
    crate::startup_cancel::checkpoint()?;
    let mut request: Request = serde_json::from_value(value).map_err(|error| error.to_string())?;
    if request.source_built_broker {
        return Err("SourceBuiltBroker is reserved for the owned chat preparation worker.".into());
    }
    // Only the running, recognized bundle can advertise portable updating.
    request.portable_executable = if crate::chat_package::bundle::current_is_bundled()? {
        Some(std::env::current_exe().map_err(|error| error.to_string())?)
    } else {
        None
    };
    if request.preparation_stage != PreparationStage::All {
        return Err("PreparationStage is reserved for the owned chat preparation worker.".into());
    }
    if !request.speech_models_only && !cfg!(all(windows, target_arch = "aarch64")) {
        return Err("Run chat in the native ARM64 executable on Windows ARM64.".into());
    }
    let _instance = if request.speech_models_only {
        None
    } else {
        let directory = std::env::var_os("LOCALAPPDATA")
            .map(PathBuf::from)
            .ok_or("LOCALAPPDATA is required for native chat admission.")?
            .join("SwiftBonsai2");
        let Some(guard) = instance::acquire(&directory)? else {
            return Ok(json!({
                "Status": "already-running",
                "Message": "Swift Bonsai 2 is already starting or running. No second instance was opened."
            }));
        };
        Some(guard)
    };
    let plan = prepare_for_family(
        request,
        cfg!(all(windows, target_arch = "aarch64")),
        observed_host_family(),
    )?;
    let working = working_directory(plan.request.working_directory.as_deref())?;
    if plan.request.speech_models_only {
        return prepare_backend(plan, &mut Native, &working);
    }
    let tool = std::env::current_exe().map_err(|error| error.to_string())?;
    let mut result = launch_pending(plan.request, &mut Native, &working, &tool)?;
    let handoff = result
        .as_object_mut()
        .and_then(|object| object.remove("UpdateHandoff"))
        .and_then(|value| value.as_str().map(PathBuf::from));
    // The native UI and its complete owned Job have already stopped. Retain
    // launch admission through installer revalidation and process creation.
    // Installer startup returns immediately; returning releases admission and
    // lets this maintenance image exit before Windows replaces its files.
    if let Some(handoff) = handoff {
        return crate::app_update::apply_after_exit(&handoff);
    }
    Ok(result)
}

/// Prepares the backend off the native UI thread and returns its validated arguments.
///
/// # Errors
/// Retains the launcher's option, artifact, build, path and credential checks.
/// The caller must own this process and its descendants until preparation stops.
pub fn prepare_run(value: Value) -> crate::Result<Value> {
    crate::startup_cancel::checkpoint()?;
    let request: Request = serde_json::from_value(value).map_err(|error| error.to_string())?;
    let plan = prepare_for_family(
        request,
        cfg!(all(windows, target_arch = "aarch64")),
        observed_host_family(),
    )?;
    if plan.request.speech_models_only {
        return Err("Use chat-launch for SpeechModelsOnly provisioning.".into());
    }
    if plan.request.chat_path.is_none() {
        return Err("Chat preparation requires the already launched chat executable.".into());
    }
    let working = working_directory(plan.request.working_directory.as_deref())?;
    prepare_backend(plan, &mut Native, &working)
}

#[cfg(test)]
mod tests;
