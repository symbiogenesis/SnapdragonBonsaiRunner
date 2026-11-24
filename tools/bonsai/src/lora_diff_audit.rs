//! Read-only BF16 weight-diff, spectral-denoising audit, and GGUF LoRA export/verification.

mod activation_edit;
mod gguf_adapter;
mod peft_import;
mod readiness;
mod reallocate;
mod safetensors;
mod svd;
#[cfg(test)]
mod tests;
mod transform;

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use self::gguf_adapter::{
    ExportDtype, ExportTensorPair, VerifyPolicy, build_balanced_factors, swift_base_dims_map,
    verify_lora_gguf, verify_lora_gguf_with_dims, write_lora_gguf,
};
use self::safetensors::{
    FloatDtype, SafetensorsIndex, ShardSource, TensorCategory, TensorDescriptor, TensorSpec,
    coalesce_small_spans, decode_scalar_f32, swift_tensor_inventory,
};
use self::svd::{
    CumulativeEnergy, ShrinkageMode, StreamingTwoSidedSketch, SvdSpectrum,
    randomized_subspace_denoised_factors_with_mode, randomized_subspace_svd,
};

const DAISYROSE_REPO: &str = "RadixArk/DaisyRose-27B";
const DAISYROSE_REVISION: &str = "64a4ec665091bb0c1931babce4e97e686eef3c5c";
const QWEN38_REPO: &str = "Qwen/Qwen3.8-27B";
const QWEN38_REVISION: &str = "37561816e66b5b9153b1b64c10968c18c1e6f4a0";
const HUGGINGFACE_BASE_URL: &str = "https://huggingface.co";
const VOCAB_CHUNK_ROWS: usize = 4_096;
const SMALL_TENSOR_MAX_BYTES: u64 = 4 * 1024 * 1024;
const STREAMING_THRESHOLD_ELEMENTS: usize = 100_000_000;

#[derive(Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
enum Operation {
    Plan,
    Audit,
    ExportDenoised,
    VerifyAdapter,
    MergeAdapters,
    EditAdapter,
    ReallocateRanks,
    FoldPatch,
    RealignFactors,
    GenerateTestAdapter,
    GenerateLock,
    DrG2Plan,
    DrG2Score,
    ImportPeftLora,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
enum TargetEnvelope {
    #[default]
    All,
    ResidualWriters,
    NpuFriendly,
    FullLinear,
}

impl TargetEnvelope {
    const fn as_str(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::ResidualWriters => "residual-writers",
            Self::NpuFriendly => "npu-friendly",
            Self::FullLinear => "full-linear",
        }
    }

    fn matches_family(self, family: &str) -> bool {
        match self {
            Self::All => true,
            Self::ResidualWriters => matches!(family, "ffn_down" | "attn_output"),
            Self::NpuFriendly => {
                matches!(
                    family,
                    "ffn_down" | "attn_output" | "attn_q" | "attn_k" | "attn_v"
                )
            }
            Self::FullLinear => {
                matches!(
                    family,
                    "ffn_gate"
                        | "ffn_up"
                        | "ffn_down"
                        | "attn_output"
                        | "attn_q"
                        | "attn_k"
                        | "attn_v"
                )
            }
        }
    }
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
enum SourceSpec {
    #[serde(rename = "pinned-daisyrose")]
    PinnedDaisyrose,
    #[serde(rename = "pinned-huggingface")]
    PinnedHuggingFace {
        #[serde(rename = "candidateRepository")]
        candidate_repository: String,
        #[serde(rename = "candidateRevision")]
        candidate_revision: String,
        #[serde(rename = "candidateLicense")]
        candidate_license: String,
        #[serde(rename = "baseRepository")]
        base_repository: String,
        #[serde(rename = "baseRevision")]
        base_revision: String,
    },
    LocalDirectories {
        #[serde(rename = "candidateDirectory")]
        candidate_directory: PathBuf,
        #[serde(rename = "baseDirectory")]
        base_directory: PathBuf,
    },
}

pub(super) fn is_safe_ascii_atom(value: &str) -> bool {
    !value.is_empty()
        && value != "."
        && value != ".."
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"_.-".contains(&byte))
}

pub(super) fn validate_hf_repository(repository: &str, label: &str) -> crate::Result<()> {
    let valid = repository
        .split_once('/')
        .is_some_and(|(owner, name)| is_safe_ascii_atom(owner) && is_safe_ascii_atom(name));
    if !valid {
        return Err(format!(
            "{label} must be owner/name using safe ASCII atoms ([A-Za-z0-9_.-])."
        ));
    }
    Ok(())
}

pub(super) fn validate_commit_revision(revision: &str, label: &str) -> crate::Result<()> {
    if !crate::identity::is_lower_hex(revision, 40) {
        return Err(format!(
            "{label} must be a lowercase 40-character hex commit hash."
        ));
    }
    Ok(())
}

pub(super) fn validate_license(license: &str, label: &str) -> crate::Result<()> {
    let valid = !license.trim().is_empty()
        && license.len() <= 64
        && license.bytes().all(|byte| (0x20..=0x7e).contains(&byte));
    if !valid {
        return Err(format!(
            "{label} must be a non-empty printable ASCII string (<= 64 bytes)."
        ));
    }
    Ok(())
}

const fn default_true() -> bool {
    true
}

const fn default_min_spike_ratio() -> f64 {
    1.0
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(default, rename_all = "camelCase", deny_unknown_fields)]
struct Scope {
    layers: Vec<u32>,
    target_envelope: Option<TargetEnvelope>,
    include_output_norm: bool,
    include_vocab: bool,
    include_svd: bool,
    #[serde(default = "default_true")]
    include_spectral_denoising: bool,
    export_dtype: ExportDtype,
    #[serde(default = "default_min_spike_ratio")]
    min_spike_ratio: f64,
    #[serde(default)]
    shrinkage_mode: ShrinkageMode,
    #[serde(default)]
    shrinkage_blend: Option<f64>,
    max_rank: usize,
    oversampling: usize,
    power_iterations: usize,
}

impl Default for Scope {
    fn default() -> Self {
        Self {
            layers: vec![0, 3, 31, 63],
            target_envelope: None,
            include_output_norm: true,
            include_vocab: true,
            include_svd: true,
            include_spectral_denoising: true,
            export_dtype: ExportDtype::F16,
            min_spike_ratio: 1.0,
            shrinkage_mode: ShrinkageMode::GavishDonoho,
            shrinkage_blend: None,
            max_rank: 64,
            oversampling: 16,
            power_iterations: 2,
        }
    }
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(default, rename_all = "camelCase", deny_unknown_fields)]
struct Controls {
    time_budget_seconds: u64,
    workers: usize,
    http_concurrency: usize,
}

impl Default for Controls {
    fn default() -> Self {
        let parallel = std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(4);
        Self {
            time_budget_seconds: 300,
            workers: parallel.clamp(1, 16),
            http_concurrency: 2,
        }
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Request {
    schema_version: u32,
    operation: Operation,
    #[serde(default)]
    source: Option<SourceSpec>,
    #[serde(default)]
    scope: Scope,
    #[serde(default)]
    controls: Controls,
    output_path: Option<PathBuf>,
    #[serde(default)]
    adapter_path: Option<PathBuf>,
    #[serde(default)]
    input_paths: Option<Vec<PathBuf>>,
    #[serde(default)]
    edit_config: Option<transform::EditAdapterConfig>,
    #[serde(default)]
    reallocate_config: Option<reallocate::ReallocateRanksConfig>,
    #[serde(default)]
    fold_patch_config: Option<activation_edit::FoldPatchConfig>,
    #[serde(default)]
    realign_config: Option<activation_edit::RealignFactorsConfig>,
    #[serde(default)]
    verify_policy: VerifyPolicy,
    #[serde(default)]
    lock_source: Option<bonsai_cpu_policy::DaisyRoseLoraSource>,
    #[serde(default)]
    default_enabled: bool,
    #[serde(default)]
    default_scale_tenths: Option<u8>,
    #[serde(default)]
    evidence_path: Option<PathBuf>,
    #[serde(default)]
    screen_config: Option<readiness::DrG2PlanConfig>,
    #[serde(default)]
    peft_config: Option<peft_import::ImportPeftConfig>,
}

#[expect(
    clippy::as_conversions,
    reason = "Scalar integer counts convert cleanly to f64 for relative error and density metrics."
)]
#[inline]
const fn u64_to_f64(value: u64) -> f64 {
    value as f64
}

#[expect(
    clippy::as_conversions,
    reason = "Scalar element counts convert cleanly to f64 for mean absolute delta."
)]
#[inline]
const fn usize_to_f64(value: usize) -> f64 {
    value as f64
}

#[expect(
    clippy::as_conversions,
    reason = "Scalar integer ranks (<= 64) convert exactly to f32 for GGUF alpha."
)]
#[inline]
const fn usize_to_f32(value: usize) -> f32 {
    value as f32
}

fn effective_envelope(request: &Request) -> crate::Result<TargetEnvelope> {
    match request.operation {
        Operation::ExportDenoised => match request.scope.target_envelope {
            None => Ok(TargetEnvelope::NpuFriendly),
            Some(TargetEnvelope::All) => Err(
                "export-denoised requires targetEnvelope in {residual-writers, npu-friendly, full-linear} so ssm_out and non-matrix tensors stay excluded.".into(),
            ),
            Some(env) => Ok(env),
        },
        Operation::GenerateTestAdapter => match request.scope.target_envelope {
            None => Ok(TargetEnvelope::NpuFriendly),
            Some(TargetEnvelope::All) => Err(
                "generate-test-adapter requires targetEnvelope in {residual-writers, npu-friendly, full-linear} so ssm_out and non-matrix tensors stay excluded.".into(),
            ),
            Some(env) => Ok(env),
        },
        Operation::Plan
        | Operation::Audit
        | Operation::VerifyAdapter
        | Operation::MergeAdapters
        | Operation::EditAdapter
        | Operation::ReallocateRanks
        | Operation::FoldPatch
        | Operation::RealignFactors
        | Operation::GenerateLock
        | Operation::DrG2Plan
        | Operation::DrG2Score
        | Operation::ImportPeftLora => Ok(request.scope.target_envelope.unwrap_or(TargetEnvelope::All)),
    }
}

fn validate_output_path(path: &Path) -> crate::Result<()> {
    bonsai_artifacts::validate_external_path(path).map_err(|e| e.to_string())?;
    if fs::exists(path).map_err(|e| e.to_string())? {
        return Err("outputPath must name a new external file that does not already exist.".into());
    }
    if path.parent().is_none_or(|parent| !parent.is_dir()) {
        return Err("The outputPath parent directory must already exist.".into());
    }
    Ok(())
}

fn validate_layers_and_rank(scope: &Scope) -> crate::Result<()> {
    let mut seen_layers = BTreeSet::new();
    for &layer in &scope.layers {
        if layer >= 64 {
            return Err("Selected layer index must be in 0..64.".into());
        }
        if !seen_layers.insert(layer) {
            return Err(format!("Duplicate layer index {layer} in scope.layers."));
        }
    }
    if !(1..=64).contains(&scope.max_rank) {
        return Err("scope.maxRank must be in 1..=64.".into());
    }
    Ok(())
}

fn validate_request(request: &Request) -> crate::Result<()> {
    if request.schema_version != 1 {
        return Err("lora-diff-audit requires schemaVersion 1.".into());
    }
    if request.operation != Operation::VerifyAdapter && !request.verify_policy.is_default() {
        return Err("verifyPolicy is only supported for verify-adapter.".into());
    }
    if request.operation != Operation::MergeAdapters && request.input_paths.is_some() {
        return Err("inputPaths is only supported for merge-adapters.".into());
    }
    if request.operation != Operation::EditAdapter && request.edit_config.is_some() {
        return Err("editConfig is only supported for edit-adapter.".into());
    }
    if request.operation != Operation::ReallocateRanks && request.reallocate_config.is_some() {
        return Err("reallocateConfig is only supported for reallocate-ranks.".into());
    }
    if request.operation != Operation::FoldPatch && request.fold_patch_config.is_some() {
        return Err("foldPatchConfig is only supported for fold-patch.".into());
    }
    if request.operation != Operation::RealignFactors && request.realign_config.is_some() {
        return Err("realignConfig is only supported for realign-factors.".into());
    }
    if request.operation != Operation::GenerateLock
        && (request.lock_source.is_some()
            || request.default_enabled
            || request.default_scale_tenths.is_some())
    {
        return Err(
            "lockSource, defaultEnabled, and defaultScaleTenths are only supported for generate-lock."
                .into(),
        );
    }
    if request.operation != Operation::DrG2Score && request.evidence_path.is_some() {
        return Err("evidencePath is only supported for dr-g2-score.".into());
    }
    if request.operation != Operation::DrG2Plan && request.screen_config.is_some() {
        return Err("screenConfig is only supported for dr-g2-plan.".into());
    }
    if request.operation != Operation::ImportPeftLora && request.peft_config.is_some() {
        return Err("peftConfig is only supported for import-peft-lora.".into());
    }

    match request.operation {
        Operation::VerifyAdapter => {
            if request.source.is_some() {
                return Err("verify-adapter does not accept a model diff source.".into());
            }
            if request.output_path.is_some() {
                return Err("verify-adapter does not accept outputPath.".into());
            }
            let adapter_path = request
                .adapter_path
                .as_ref()
                .ok_or("verify-adapter requires adapterPath.")?;
            bonsai_artifacts::validate_external_path(adapter_path).map_err(|e| e.to_string())?;
            return Ok(());
        }
        Operation::MergeAdapters => {
            if request.source.is_some() {
                return Err("merge-adapters does not accept a model diff source.".into());
            }
            if request.adapter_path.is_some() {
                return Err("merge-adapters does not accept adapterPath.".into());
            }
            let inputs = request
                .input_paths
                .as_ref()
                .ok_or("merge-adapters requires inputPaths.")?;
            if !(2..=64).contains(&inputs.len()) {
                return Err("merge-adapters requires 2..=64 chunk adapter paths.".into());
            }
            let mut unique_inputs = BTreeSet::new();
            for path in inputs {
                bonsai_artifacts::validate_external_path(path).map_err(|e| e.to_string())?;
                if !path.is_file() {
                    return Err(
                        "Each merge-adapters inputPath must exist as a regular file.".into(),
                    );
                }
                if !unique_inputs.insert(path.clone()) {
                    return Err("Duplicate path in merge-adapters inputPaths.".into());
                }
            }
            let output = request
                .output_path
                .as_ref()
                .ok_or("merge-adapters requires a fresh external outputPath.")?;
            validate_output_path(output)?;
            return Ok(());
        }
        Operation::EditAdapter => {
            if request.source.is_some() {
                return Err("edit-adapter does not accept a model diff source.".into());
            }
            let adapter_path = request
                .adapter_path
                .as_ref()
                .ok_or("edit-adapter requires adapterPath.")?;
            bonsai_artifacts::validate_external_path(adapter_path).map_err(|e| e.to_string())?;
            let output = request
                .output_path
                .as_ref()
                .ok_or("edit-adapter requires a fresh external outputPath.")?;
            validate_output_path(output)?;
            let config = request
                .edit_config
                .as_ref()
                .ok_or("edit-adapter requires editConfig.")?;
            transform::validate_edit_config(config)?;
            return Ok(());
        }
        Operation::ReallocateRanks => {
            if request.source.is_some() {
                return Err("reallocate-ranks does not accept a model diff source.".into());
            }
            let adapter_path = request
                .adapter_path
                .as_ref()
                .ok_or("reallocate-ranks requires adapterPath.")?;
            bonsai_artifacts::validate_external_path(adapter_path).map_err(|e| e.to_string())?;
            let output = request
                .output_path
                .as_ref()
                .ok_or("reallocate-ranks requires a fresh external outputPath.")?;
            validate_output_path(output)?;
            let config = request
                .reallocate_config
                .as_ref()
                .ok_or("reallocate-ranks requires reallocateConfig.")?;
            reallocate::validate_reallocate_config(config)?;
            return Ok(());
        }
        Operation::FoldPatch => {
            if request.source.is_some() {
                return Err("fold-patch does not accept a model diff source.".into());
            }
            if let Some(adapter_path) = &request.adapter_path {
                bonsai_artifacts::validate_external_path(adapter_path)
                    .map_err(|e| e.to_string())?;
            }
            let output = request
                .output_path
                .as_ref()
                .ok_or("fold-patch requires a fresh external outputPath.")?;
            validate_output_path(output)?;
            let config = request
                .fold_patch_config
                .as_ref()
                .ok_or("fold-patch requires foldPatchConfig.")?;
            activation_edit::validate_fold_patch_config(config)?;
            return Ok(());
        }
        Operation::RealignFactors => {
            if request.source.is_some() {
                return Err("realign-factors does not accept a model diff source.".into());
            }
            let adapter_path = request
                .adapter_path
                .as_ref()
                .ok_or("realign-factors requires adapterPath.")?;
            bonsai_artifacts::validate_external_path(adapter_path).map_err(|e| e.to_string())?;
            let output = request
                .output_path
                .as_ref()
                .ok_or("realign-factors requires a fresh external outputPath.")?;
            validate_output_path(output)?;
            let config = request
                .realign_config
                .as_ref()
                .ok_or("realign-factors requires realignConfig.")?;
            activation_edit::validate_realign_config(config)?;
            return Ok(());
        }
        Operation::GenerateTestAdapter => {
            if request.source.is_some() {
                return Err("generate-test-adapter does not accept a model diff source.".into());
            }
            if request.adapter_path.is_some() {
                return Err("generate-test-adapter does not accept adapterPath.".into());
            }
            validate_layers_and_rank(&request.scope)?;
            if request.scope.layers.is_empty() {
                return Err(
                    "generate-test-adapter requires at least one layer in scope.layers.".into(),
                );
            }
            let envelope = effective_envelope(request)?;
            if select_tensors(&request.scope, envelope).is_empty() {
                return Err("generate-test-adapter scope selects zero tensors.".into());
            }
            let output = request
                .output_path
                .as_ref()
                .ok_or("generate-test-adapter requires a fresh external outputPath.")?;
            validate_output_path(output)?;
            return Ok(());
        }
        Operation::GenerateLock => {
            if request.source.is_some() {
                return Err("generate-lock does not accept a model diff source.".into());
            }
            let adapter_path = request
                .adapter_path
                .as_ref()
                .ok_or("generate-lock requires adapterPath.")?;
            bonsai_artifacts::validate_external_path(adapter_path).map_err(|e| e.to_string())?;
            let output = request
                .output_path
                .as_ref()
                .ok_or("generate-lock requires a fresh external outputPath.")?;
            validate_output_path(output)?;
            if request.lock_source.is_none() {
                return Err("generate-lock requires lockSource provenance.".into());
            }
            return Ok(());
        }
        Operation::DrG2Plan => {
            if request.source.is_some() {
                return Err("dr-g2-plan does not accept a model diff source.".into());
            }
            if request.adapter_path.is_some() {
                return Err("dr-g2-plan does not accept adapterPath.".into());
            }
            if let Some(output) = &request.output_path {
                validate_output_path(output)?;
            }
            return Ok(());
        }
        Operation::DrG2Score => {
            if request.source.is_some() {
                return Err("dr-g2-score does not accept a model diff source.".into());
            }
            if request.adapter_path.is_some() {
                return Err("dr-g2-score does not accept adapterPath.".into());
            }
            let evidence_path = request
                .evidence_path
                .as_ref()
                .ok_or("dr-g2-score requires evidencePath.")?;
            bonsai_artifacts::validate_external_path(evidence_path).map_err(|e| e.to_string())?;
            if let Some(output) = &request.output_path {
                validate_output_path(output)?;
            }
            return Ok(());
        }
        Operation::ImportPeftLora => {
            if request.source.is_some() {
                return Err(
                    "import-peft-lora does not accept a model diff source; supply peftConfig.source."
                        .into(),
                );
            }
            if request.adapter_path.is_some() {
                return Err("import-peft-lora does not accept adapterPath.".into());
            }
            if !(10..=900).contains(&request.controls.time_budget_seconds) {
                return Err("controls.timeBudgetSeconds must be in 10..=900.".into());
            }
            if !(1..=32).contains(&request.controls.workers) {
                return Err("controls.workers must be in 1..=32.".into());
            }
            if !(1..=4).contains(&request.controls.http_concurrency) {
                return Err("controls.httpConcurrency must be in 1..=4.".into());
            }
            let output = request
                .output_path
                .as_ref()
                .ok_or("import-peft-lora requires a fresh external outputPath.")?;
            validate_output_path(output)?;
            let config = request
                .peft_config
                .as_ref()
                .ok_or("import-peft-lora requires peftConfig.")?;
            peft_import::validate_import_peft_config(config)?;
            return Ok(());
        }
        Operation::Plan | Operation::Audit | Operation::ExportDenoised => {}
    }

    if request.adapter_path.is_some() {
        return Err("adapterPath is only supported for verify-adapter.".into());
    }
    let source = request
        .source
        .as_ref()
        .ok_or("plan, audit, and export-denoised require source.")?;
    let envelope = effective_envelope(request)?;

    validate_layers_and_rank(&request.scope)?;
    let selects_any_tensor = if envelope == TargetEnvelope::All {
        !request.scope.layers.is_empty()
            || request.scope.include_output_norm
            || request.scope.include_vocab
    } else {
        !select_tensors(&request.scope, envelope).is_empty()
    };
    if !selects_any_tensor {
        return Err("Audit scope selects zero tensors.".into());
    }
    if request.scope.oversampling > 64 {
        return Err("scope.oversampling must be in 0..=64.".into());
    }
    if request.scope.power_iterations > 6 {
        return Err("scope.powerIterations must be in 0..=6.".into());
    }
    if !request.scope.min_spike_ratio.is_finite()
        || !(1.0..=100.0).contains(&request.scope.min_spike_ratio)
    {
        return Err("scope.minSpikeRatio must be finite and in 1.0..=100.0.".into());
    }
    match request.scope.shrinkage_mode {
        ShrinkageMode::Blended => match request.scope.shrinkage_blend {
            Some(alpha) if alpha.is_finite() && (0.0..=1.0).contains(&alpha) => {}
            Some(_) => {
                return Err(
                    "scope.shrinkageBlend must be finite and in 0.0..=1.0 when shrinkageMode is 'blended'."
                        .into(),
                );
            }
            None => {
                return Err(
                    "scope.shrinkageBlend is required when scope.shrinkageMode is 'blended'."
                        .into(),
                );
            }
        },
        ShrinkageMode::GavishDonoho | ShrinkageMode::Unshrunk => {
            if request.scope.shrinkage_blend.is_some() {
                return Err(
                    "scope.shrinkageBlend is only allowed when scope.shrinkageMode is 'blended'."
                        .into(),
                );
            }
        }
    }
    if !(10..=900).contains(&request.controls.time_budget_seconds) {
        return Err("controls.timeBudgetSeconds must be in 10..=900.".into());
    }
    if !(1..=32).contains(&request.controls.workers) {
        return Err("controls.workers must be in 1..=32.".into());
    }
    if !(1..=4).contains(&request.controls.http_concurrency) {
        return Err("controls.httpConcurrency must be in 1..=4.".into());
    }
    match source {
        SourceSpec::PinnedDaisyrose => {}
        SourceSpec::PinnedHuggingFace {
            candidate_repository,
            candidate_revision,
            candidate_license,
            base_repository,
            base_revision,
        } => {
            validate_hf_repository(candidate_repository, "source.candidateRepository")?;
            validate_commit_revision(candidate_revision, "source.candidateRevision")?;
            validate_license(candidate_license, "source.candidateLicense")?;
            validate_hf_repository(base_repository, "source.baseRepository")?;
            validate_commit_revision(base_revision, "source.baseRevision")?;
        }
        SourceSpec::LocalDirectories {
            candidate_directory,
            base_directory,
        } => {
            bonsai_artifacts::validate_external_path(candidate_directory)
                .map_err(|e| e.to_string())?;
            bonsai_artifacts::validate_external_path(base_directory).map_err(|e| e.to_string())?;
            if !candidate_directory.is_dir() || !base_directory.is_dir() {
                return Err("Local candidateDirectory and baseDirectory must both exist.".into());
            }
        }
    }
    match request.operation {
        Operation::ExportDenoised => {
            if !request.scope.include_svd || !request.scope.include_spectral_denoising {
                return Err(
                    "export-denoised requires includeSvd and includeSpectralDenoising to be true."
                        .into(),
                );
            }
            let output = request
                .output_path
                .as_ref()
                .ok_or("export-denoised requires a fresh external outputPath.")?;
            validate_output_path(output)?;
        }
        Operation::Plan | Operation::Audit => {
            if let Some(output) = &request.output_path {
                validate_output_path(output)?;
            }
        }
        Operation::VerifyAdapter
        | Operation::MergeAdapters
        | Operation::EditAdapter
        | Operation::ReallocateRanks
        | Operation::FoldPatch
        | Operation::RealignFactors
        | Operation::GenerateTestAdapter
        | Operation::GenerateLock
        | Operation::DrG2Plan
        | Operation::DrG2Score
        | Operation::ImportPeftLora => {}
    }
    Ok(())
}

fn select_tensors(scope: &Scope, envelope: TargetEnvelope) -> Vec<TensorSpec> {
    let layer_set: BTreeSet<u32> = scope.layers.iter().copied().collect();
    let inventory = swift_tensor_inventory();
    let mut selected = Vec::new();
    for spec in inventory {
        let include = if envelope == TargetEnvelope::All {
            match spec.layer {
                Some(layer) => layer_set.contains(&layer),
                None => match spec.category {
                    TensorCategory::NonMatrix => scope.include_output_norm,
                    TensorCategory::VocabMatrix => scope.include_vocab,
                    TensorCategory::GdnSmallMatrix | TensorCategory::LinearMatrix => false,
                },
            }
        } else {
            spec.layer.is_some_and(|layer| layer_set.contains(&layer))
                && spec.category == TensorCategory::LinearMatrix
                && envelope.matches_family(spec.family)
        };
        if include {
            selected.push(spec);
        }
    }
    // Order by increasing cost: non-matrix -> small GDN -> block linear -> vocab.
    selected.sort_by_key(|spec| (spec.category, spec.layer.unwrap_or(u32::MAX)));
    selected
}

fn resolved_scope_json(scope: &Scope, envelope: TargetEnvelope) -> Value {
    let mut obj = json!({
        "layers": scope.layers,
        "targetEnvelope": envelope,
        "includeOutputNorm": if envelope == TargetEnvelope::All { scope.include_output_norm } else { false },
        "includeVocab": if envelope == TargetEnvelope::All { scope.include_vocab } else { false },
        "includeSvd": scope.include_svd,
        "includeSpectralDenoising": scope.include_spectral_denoising,
        "exportDtype": scope.export_dtype,
        "minSpikeRatio": scope.min_spike_ratio,
        "shrinkageMode": scope.shrinkage_mode,
        "maxRank": scope.max_rank,
        "oversampling": scope.oversampling,
        "powerIterations": scope.power_iterations,
    });
    if let Some(blend) = scope.shrinkage_blend
        && let Some(map) = obj.as_object_mut()
    {
        map.insert("shrinkageBlend".to_owned(), json!(blend));
    }
    obj
}

fn estimated_bf16_bytes(spec: &TensorSpec) -> u64 {
    let elements: u64 = match spec.family {
        "token_embd" | "output" => 248_320 * 5_120,
        "norm" => {
            if spec.gguf_name.contains("q_norm") || spec.gguf_name.contains("k_norm") {
                256
            } else {
                5_120
            }
        }
        "ffn_gate" | "ffn_up" | "ffn_down" => 17_408 * 5_120,
        "attn_q" => 12_288 * 5_120,
        "attn_k" | "attn_v" => 1_024 * 5_120,
        "attn_output" | "attn_gate" | "ssm_out" => 6_144 * 5_120,
        "attn_qkv" => 10_240 * 5_120,
        "ssm_alpha" | "ssm_beta" => 48 * 5_120,
        "ssm_conv1d" => 10_240 * 4,
        "ssm_norm" => 128,
        "ssm_a" | "ssm_dt_bias" => 48,
        _ => 0,
    };
    elements * 2
}

fn source_summary(source: &SourceSpec) -> Value {
    match source {
        SourceSpec::PinnedDaisyrose => json!({
            "kind": "pinned-daisyrose",
            "candidate": {
                "repository": DAISYROSE_REPO,
                "revision": DAISYROSE_REVISION,
                "baseUrl": HUGGINGFACE_BASE_URL
            },
            "base": {
                "repository": QWEN38_REPO,
                "revision": QWEN38_REVISION,
                "baseUrl": HUGGINGFACE_BASE_URL
            }
        }),
        SourceSpec::PinnedHuggingFace {
            candidate_repository,
            candidate_revision,
            candidate_license,
            base_repository,
            base_revision,
        } => json!({
            "kind": "pinned-huggingface",
            "candidate": {
                "repository": candidate_repository,
                "revision": candidate_revision,
                "license": candidate_license,
                "baseUrl": HUGGINGFACE_BASE_URL
            },
            "base": {
                "repository": base_repository,
                "revision": base_revision,
                "baseUrl": HUGGINGFACE_BASE_URL
            }
        }),
        SourceSpec::LocalDirectories {
            candidate_directory,
            base_directory,
        } => json!({
            "kind": "local-directories",
            "candidateDirectory": candidate_directory,
            "baseDirectory": base_directory
        }),
    }
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct TensorReport {
    hf_name: String,
    gguf_name: String,
    layer: Option<u32>,
    family: String,
    category: String,
    dtype: String,
    shape: Vec<usize>,
    element_count: usize,
    byte_size_per_model: u64,
    base_frobenius_norm: f64,
    candidate_frobenius_norm: f64,
    delta_frobenius_energy: f64,
    delta_frobenius_norm: f64,
    relative_frobenius_delta: f64,
    mean_abs_delta: f64,
    max_abs_delta: f64,
    nonzero_delta_count: u64,
    nonzero_delta_fraction: f64,
    changed_rows: Option<usize>,
    total_rows: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    svd: Option<SvdSpectrum>,
    #[serde(skip)]
    base_frobenius_energy: f64,
}

#[derive(Default)]
struct MetricsAccum {
    tensor_count: usize,
    total_elements: u64,
    total_base_energy: f64,
    total_delta_energy: f64,
    sum_rel_delta: f64,
    max_rel_delta: f64,
    max_abs_delta: f64,
    total_nonzero: u64,
    svd_weight_energy: f64,
    weighted_r1: f64,
    weighted_r4: f64,
    weighted_r8: f64,
    weighted_r16: f64,
    weighted_r32: f64,
    weighted_r64: f64,
}

impl MetricsAccum {
    fn add(&mut self, report: &TensorReport) {
        self.tensor_count += 1;
        self.total_elements += u64::try_from(report.element_count).unwrap_or(u64::MAX);
        self.total_base_energy += report.base_frobenius_energy;
        self.total_delta_energy += report.delta_frobenius_energy;
        self.sum_rel_delta += report.relative_frobenius_delta;
        self.max_rel_delta = self.max_rel_delta.max(report.relative_frobenius_delta);
        self.max_abs_delta = self.max_abs_delta.max(report.max_abs_delta);
        self.total_nonzero += report.nonzero_delta_count;
        if let Some(svd) = &report.svd
            && report.delta_frobenius_energy > 0.0
        {
            let w = report.delta_frobenius_energy;
            self.svd_weight_energy += w;
            self.weighted_r1 += w * svd.cumulative_energy.r1;
            self.weighted_r4 += w * svd.cumulative_energy.r4;
            self.weighted_r8 += w * svd.cumulative_energy.r8;
            self.weighted_r16 += w * svd.cumulative_energy.r16;
            self.weighted_r32 += w * svd.cumulative_energy.r32;
            self.weighted_r64 += w * svd.cumulative_energy.r64;
        }
    }

    fn aggregate_relative(&self) -> f64 {
        if self.total_base_energy > 0.0 {
            (self.total_delta_energy / self.total_base_energy).sqrt()
        } else {
            0.0
        }
    }

    fn mean_relative(&self) -> f64 {
        if self.tensor_count > 0 {
            self.sum_rel_delta / usize_to_f64(self.tensor_count)
        } else {
            0.0
        }
    }

    fn weighted_svd(&self) -> Option<CumulativeEnergy> {
        if self.svd_weight_energy <= 0.0 {
            return None;
        }
        let inv = 1.0 / self.svd_weight_energy;
        Some(CumulativeEnergy {
            r1: (self.weighted_r1 * inv).clamp(0.0, 1.0),
            r4: (self.weighted_r4 * inv).clamp(0.0, 1.0),
            r8: (self.weighted_r8 * inv).clamp(0.0, 1.0),
            r16: (self.weighted_r16 * inv).clamp(0.0, 1.0),
            r32: (self.weighted_r32 * inv).clamp(0.0, 1.0),
            r64: (self.weighted_r64 * inv).clamp(0.0, 1.0),
        })
    }

    fn to_json(&self, audited_energy: f64, non_vocab_energy: f64, linear_energy: f64) -> Value {
        let nonzero_fraction = if self.total_elements > 0 {
            u64_to_f64(self.total_nonzero) / u64_to_f64(self.total_elements)
        } else {
            0.0
        };
        let share_audited = if audited_energy > 0.0 {
            self.total_delta_energy / audited_energy
        } else {
            0.0
        };
        let share_non_vocab = if non_vocab_energy > 0.0 {
            self.total_delta_energy / non_vocab_energy
        } else {
            0.0
        };
        let share_linear = if linear_energy > 0.0 {
            self.total_delta_energy / linear_energy
        } else {
            0.0
        };
        json!({
            "tensorCount": self.tensor_count,
            "totalElements": self.total_elements,
            "totalBaseFrobeniusEnergy": self.total_base_energy,
            "totalDeltaFrobeniusEnergy": self.total_delta_energy,
            "aggregateRelativeFrobeniusDelta": self.aggregate_relative(),
            "meanTensorRelativeFrobeniusDelta": self.mean_relative(),
            "maxTensorRelativeFrobeniusDelta": self.max_rel_delta,
            "maxAbsDelta": self.max_abs_delta,
            "nonzeroDeltaFraction": nonzero_fraction,
            "shareOfAuditedEnergy": share_audited,
            "shareOfNonVocabEnergy": share_non_vocab,
            "shareOfLinearMatrixEnergy": share_linear,
            "energyWeightedCumulativeEnergy": self.weighted_svd(),
        })
    }
}

struct TensorDiffStats {
    base_energy: f64,
    cand_energy: f64,
    delta_energy: f64,
    sum_abs_delta: f64,
    max_abs_delta: f64,
    nonzero_count: u64,
    changed_rows: Option<usize>,
}

#[inline]
fn diff_chunk_rows(
    dtype: FloatDtype,
    cand_bytes: &[u8],
    base_bytes: &[u8],
    rows: usize,
    cols: usize,
    delta_out: &mut [f32],
    workers: usize,
) -> crate::Result<TensorDiffStats> {
    let width = dtype.byte_width();
    let expected_bytes = rows
        .checked_mul(cols)
        .and_then(|n| n.checked_mul(width))
        .ok_or("Diff chunk byte size overflow.")?;
    if cand_bytes.len() != expected_bytes
        || base_bytes.len() != expected_bytes
        || delta_out.len() != rows * cols
    {
        return Err("Diff chunk buffer length mismatch.".into());
    }
    if rows == 0 || cols == 0 {
        return Ok(TensorDiffStats {
            base_energy: 0.0,
            cand_energy: 0.0,
            delta_energy: 0.0,
            sum_abs_delta: 0.0,
            max_abs_delta: 0.0,
            nonzero_count: 0,
            changed_rows: Some(0),
        });
    }
    let worker_count = workers.clamp(1, rows);
    let rows_per_worker = rows.div_ceil(worker_count);
    let partials: Vec<crate::Result<TensorDiffStats>> = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
        for (idx, d_part) in delta_out.chunks_mut(rows_per_worker * cols).enumerate() {
            let r_start = idx * rows_per_worker;
            let local_rows = d_part.len() / cols;
            let byte_start = r_start * cols * width;
            let byte_end = (r_start + local_rows) * cols * width;
            let c_part = &cand_bytes[byte_start..byte_end];
            let b_part = &base_bytes[byte_start..byte_end];
            handles.push(scope.spawn(move || {
                let mut base_energy = 0.0_f64;
                let mut cand_energy = 0.0_f64;
                let mut delta_energy = 0.0_f64;
                let mut sum_abs_delta = 0.0_f64;
                let mut max_abs_delta = 0.0_f64;
                let mut nonzero_count = 0_u64;
                let mut changed_rows = 0_usize;
                for r in 0..local_rows {
                    let mut row_changed = false;
                    let row_base = r * cols;
                    for c in 0..cols {
                        let elem_idx = row_base + c;
                        let byte_off = elem_idx * width;
                        let cw = decode_scalar_f32(dtype, &c_part[byte_off..byte_off + width])?;
                        let bw = decode_scalar_f32(dtype, &b_part[byte_off..byte_off + width])?;
                        let diff = cw - bw;
                        if !diff.is_finite() {
                            return Err("Weight delta overflowed F32.".into());
                        }
                        d_part[elem_idx] = diff;
                        let cf = f64::from(cw);
                        let bf = f64::from(bw);
                        let df = cf - bf;
                        base_energy += bf * bf;
                        cand_energy += cf * cf;
                        delta_energy += df * df;
                        let ad = df.abs();
                        sum_abs_delta += ad;
                        max_abs_delta = max_abs_delta.max(ad);
                        if cw != bw {
                            nonzero_count += 1;
                            row_changed = true;
                        }
                    }
                    if row_changed {
                        changed_rows += 1;
                    }
                }
                Ok(TensorDiffStats {
                    base_energy,
                    cand_energy,
                    delta_energy,
                    sum_abs_delta,
                    max_abs_delta,
                    nonzero_count,
                    changed_rows: Some(changed_rows),
                })
            }));
        }
        handles
            .into_iter()
            .map(|h| {
                h.join()
                    .unwrap_or_else(|_| Err("Diff worker thread panicked.".into()))
            })
            .collect()
    });

    let mut total = TensorDiffStats {
        base_energy: 0.0,
        cand_energy: 0.0,
        delta_energy: 0.0,
        sum_abs_delta: 0.0,
        max_abs_delta: 0.0,
        nonzero_count: 0,
        changed_rows: Some(0),
    };
    for part in partials {
        let part = part?;
        total.base_energy += part.base_energy;
        total.cand_energy += part.cand_energy;
        total.delta_energy += part.delta_energy;
        total.sum_abs_delta += part.sum_abs_delta;
        total.max_abs_delta = total.max_abs_delta.max(part.max_abs_delta);
        total.nonzero_count += part.nonzero_count;
        total.changed_rows = Some(total.changed_rows.unwrap_or(0) + part.changed_rows.unwrap_or(0));
    }
    Ok(total)
}

fn fetch_pair_slices(
    cand_src: &ShardSource,
    cand_shard: &str,
    cand_begin: u64,
    cand_len: u64,
    base_src: &ShardSource,
    base_shard: &str,
    base_begin: u64,
    base_len: u64,
    deadline: Instant,
) -> crate::Result<(Vec<u8>, Vec<u8>)> {
    std::thread::scope(|scope| {
        let c_handle =
            scope.spawn(move || cand_src.fetch_slice(cand_shard, cand_begin, cand_len, deadline));
        let b_res = base_src.fetch_slice(base_shard, base_begin, base_len, deadline);
        let c_res = c_handle
            .join()
            .unwrap_or_else(|_| Err("Candidate fetch thread panicked.".into()))?;
        let b_res = b_res?;
        Ok((c_res, b_res))
    })
}

fn fetch_coalesced_side(
    src: &ShardSource,
    items: &[(usize, &str, u64, u64)],
    deadline: Instant,
) -> crate::Result<BTreeMap<usize, Vec<u8>>> {
    let spans = coalesce_small_spans(items);
    let mut out = BTreeMap::new();
    for span in spans {
        if Instant::now() >= deadline {
            return Err("Audit time budget expired during small-tensor transfer.".into());
        }
        crate::startup_cancel::checkpoint()?;
        let buf = src.fetch_slice(
            &span.shard,
            span.abs_begin,
            span.abs_end - span.abs_begin,
            deadline,
        )?;
        for (idx, begin, end) in span.entries {
            let rel_begin = usize::try_from(begin - span.abs_begin).map_err(|e| e.to_string())?;
            let rel_end = usize::try_from(end - span.abs_begin).map_err(|e| e.to_string())?;
            out.insert(idx, buf[rel_begin..rel_end].to_vec());
        }
    }
    Ok(out)
}

fn build_tensor_report(
    spec: &TensorSpec,
    dtype: FloatDtype,
    shape: Vec<usize>,
    element_count: usize,
    byte_size: u64,
    stats: TensorDiffStats,
    svd: Option<SvdSpectrum>,
) -> TensorReport {
    let base_norm = stats.base_energy.sqrt();
    let cand_norm = stats.cand_energy.sqrt();
    let delta_norm = stats.delta_energy.sqrt();
    let rel = if base_norm > 0.0 {
        delta_norm / base_norm
    } else {
        0.0
    };
    let elem_f = usize_to_f64(element_count.max(1));
    TensorReport {
        hf_name: spec.hf_name.clone(),
        gguf_name: spec.gguf_name.clone(),
        layer: spec.layer,
        family: spec.family.to_owned(),
        category: spec.category.as_str().to_owned(),
        dtype: dtype.as_str().to_owned(),
        total_rows: (shape.len() == 2).then(|| shape[0]),
        shape,
        element_count,
        byte_size_per_model: byte_size,
        base_frobenius_norm: base_norm,
        candidate_frobenius_norm: cand_norm,
        delta_frobenius_energy: stats.delta_energy,
        delta_frobenius_norm: delta_norm,
        relative_frobenius_delta: rel,
        mean_abs_delta: stats.sum_abs_delta / elem_f,
        max_abs_delta: stats.max_abs_delta,
        nonzero_delta_count: stats.nonzero_count,
        nonzero_delta_fraction: u64_to_f64(stats.nonzero_count) / elem_f,
        changed_rows: stats.changed_rows,
        svd,
        base_frobenius_energy: stats.base_energy,
    }
}

struct PreparedTensor {
    spec: TensorSpec,
    cand_shard: String,
    base_shard: String,
    desc: TensorDescriptor,
    dtype: FloatDtype,
    element_count: usize,
    byte_size: u64,
}

fn resolve_and_validate_tensors(
    selected: Vec<TensorSpec>,
    cand_src: &ShardSource,
    cand_idx: &SafetensorsIndex,
    base_src: &ShardSource,
    base_idx: &SafetensorsIndex,
    deadline: Instant,
) -> crate::Result<Vec<PreparedTensor>> {
    let mut cand_shards = BTreeSet::new();
    let mut base_shards = BTreeSet::new();
    let mut shard_pairs = Vec::with_capacity(selected.len());
    for spec in selected {
        let cs = cand_idx
            .weight_map
            .get(&spec.hf_name)
            .ok_or_else(|| format!("Candidate index missing '{}'.", spec.hf_name))?
            .clone();
        let bs = base_idx
            .weight_map
            .get(&spec.hf_name)
            .ok_or_else(|| format!("Base index missing '{}'.", spec.hf_name))?
            .clone();
        cand_shards.insert(cs.clone());
        base_shards.insert(bs.clone());
        shard_pairs.push((spec, cs, bs));
    }

    std::thread::scope(|scope| {
        let c_handle = scope.spawn(move || -> crate::Result<()> {
            for shard in &cand_shards {
                crate::startup_cancel::checkpoint()?;
                cand_src.ensure_shard_header(shard, deadline)?;
            }
            Ok(())
        });
        for shard in &base_shards {
            crate::startup_cancel::checkpoint()?;
            base_src.ensure_shard_header(shard, deadline)?;
        }
        c_handle
            .join()
            .unwrap_or_else(|_| Err("Candidate header thread panicked.".into()))?;
        Ok::<_, String>(())
    })?;

    let mut prepared = Vec::with_capacity(shard_pairs.len());
    for (spec, cand_shard, base_shard) in shard_pairs {
        let c_desc = cand_src.descriptor(&cand_shard, &spec.hf_name)?;
        let b_desc = base_src.descriptor(&base_shard, &spec.hf_name)?;
        if c_desc.dtype != b_desc.dtype || c_desc.shape != b_desc.shape {
            return Err(format!(
                "Tensor '{}' schema mismatch: candidate {} {:?} vs base {} {:?}.",
                spec.hf_name, c_desc.dtype, c_desc.shape, b_desc.dtype, b_desc.shape
            ));
        }
        let dtype = c_desc.float_dtype()?;
        let element_count = c_desc.element_count()?;
        let byte_size = c_desc.abs_end - c_desc.abs_begin;
        if b_desc.abs_end - b_desc.abs_begin != byte_size {
            return Err(format!("Byte extent mismatch on '{}'.", spec.hf_name));
        }
        if matches!(
            spec.category,
            TensorCategory::GdnSmallMatrix
                | TensorCategory::LinearMatrix
                | TensorCategory::VocabMatrix
        ) && c_desc.shape.len() != 2
        {
            return Err(format!(
                "Expected 2D matrix shape for '{}', got {:?}.",
                spec.hf_name, c_desc.shape
            ));
        }
        prepared.push(PreparedTensor {
            spec,
            cand_shard,
            base_shard,
            desc: c_desc,
            dtype,
            element_count,
            byte_size,
        });
    }
    Ok(prepared)
}

fn evaluate_gate1(
    completed: bool,
    include_svd: bool,
    skipped_count: usize,
    all_audited: &MetricsAccum,
    non_vocab_all: &MetricsAccum,
    pure_non_matrix: &MetricsAccum,
    non_matrix_small_gdn: &MetricsAccum,
    linear_matrices: &MetricsAccum,
    ffn_and_full_attn: &MetricsAccum,
    cand_a_down_attn_o: &MetricsAccum,
    cand_b_gate_up_qkv: &MetricsAccum,
) -> Value {
    let mut reasons = Vec::new();
    let total_energy = all_audited.total_delta_energy;
    let non_vocab_energy = non_vocab_all.total_delta_energy;
    let non_matrix_small_gdn_share = if non_vocab_energy > 0.0 {
        non_matrix_small_gdn.total_delta_energy / non_vocab_energy
    } else {
        0.0
    };
    let pure_non_matrix_mean_rel = pure_non_matrix.mean_relative();
    let linear_mean_rel = linear_matrices.mean_relative();
    let ffn_attn_e32 = ffn_and_full_attn.weighted_svd().map_or(0.0, |e| e.r32);
    let ffn_attn_e16 = ffn_and_full_attn.weighted_svd().map_or(0.0, |e| e.r16);
    let cand_a_share = if linear_matrices.total_delta_energy > 0.0 {
        cand_a_down_attn_o.total_delta_energy / linear_matrices.total_delta_energy
    } else {
        0.0
    };
    let gate_up_qkv_share = if linear_matrices.total_delta_energy > 0.0 {
        cand_b_gate_up_qkv.total_delta_energy / linear_matrices.total_delta_energy
    } else {
        0.0
    };

    let (passed, verdict) = if total_energy == 0.0 {
        reasons.push("Zero difference detected across every audited tensor.".to_owned());
        (false, "no-go-zero-delta")
    } else if non_vocab_energy > 0.0 && non_matrix_small_gdn_share >= 0.50 {
        reasons.push(format!(
            "Non-matrix and small-GDN parameters hold {:.2}% (>= 50%) of non-vocab delta Frobenius energy.",
            non_matrix_small_gdn_share * 100.0
        ));
        (false, "no-go-non-matrix-dominant")
    } else if linear_mean_rel > 0.0 && pure_non_matrix_mean_rel > 5.0 * linear_mean_rel {
        reasons.push(format!(
            "Pure non-matrix mean relative Frobenius delta ({pure_non_matrix_mean_rel:.6}) exceeds 5.0x linear-matrix mean ({linear_mean_rel:.6}; ratio {:.2}x).",
            pure_non_matrix_mean_rel / linear_mean_rel
        ));
        (false, "no-go-non-matrix-dominant")
    } else if include_svd && ffn_and_full_attn.tensor_count > 0 && ffn_attn_e32 < 0.50 {
        reasons.push(format!(
            "Energy-weighted E(32) across FFN and full-attention matrices is {:.2}% (< 50% Gate 1 threshold).",
            ffn_attn_e32 * 100.0
        ));
        (false, "no-go-high-rank")
    } else if include_svd && ffn_and_full_attn.tensor_count > 0 && ffn_attn_e32 < 0.65 {
        reasons.push(format!(
            "Energy-weighted E(32) across FFN and full-attention matrices is {:.2}% (between 50% stop threshold and 65% low-rank knee; E(16)={:.2}%).",
            ffn_attn_e32 * 100.0,
            ffn_attn_e16 * 100.0
        ));
        (true, "borderline-needs-full-scan")
    } else if cand_a_share >= 0.40 || cand_a_share >= gate_up_qkv_share {
        reasons.push(format!(
            "FFN and full-attention linear matrices clear Gate 1 low-rank criteria (E(16)={:.2}%, E(32)={:.2}%) with Candidate A (ffn_down + attn_output) holding {:.2}% of block linear delta energy.",
            ffn_attn_e16 * 100.0,
            ffn_attn_e32 * 100.0,
            cand_a_share * 100.0
        ));
        (true, "pass-candidate-a")
    } else {
        reasons.push(format!(
            "FFN and full-attention linear matrices clear Gate 1 low-rank criteria (E(16)={:.2}%, E(32)={:.2}%), but input-side projections (ffn_gate/ffn_up/attn_qkv) hold {:.2}% vs {:.2}% in ffn_down+attn_output (Candidate B scope).",
            ffn_attn_e16 * 100.0,
            ffn_attn_e32 * 100.0,
            gate_up_qkv_share * 100.0,
            cand_a_share * 100.0
        ));
        (true, "pass-candidate-b")
    };

    if !completed {
        reasons.push(format!(
            "Audit stopped early before auditing {skipped_count} remaining tensor(s); evaluate skipped scope before drawing final Gate 1 conclusions."
        ));
    }

    json!({
        "passed": passed,
        "verdict": verdict,
        "reasons": reasons,
        "metrics": {
            "nonMatrixAndSmallGdnShareOfNonVocabEnergy": non_matrix_small_gdn_share,
            "pureNonMatrixMeanRelativeDelta": pure_non_matrix_mean_rel,
            "linearMatrixMeanRelativeDelta": linear_mean_rel,
            "ffnAndFullAttnEnergyWeightedE16": ffn_attn_e16,
            "ffnAndFullAttnEnergyWeightedE32": ffn_attn_e32,
            "candidateAShareOfBlockLinearEnergy": cand_a_share,
            "gateUpQkvShareOfBlockLinearEnergy": gate_up_qkv_share,
        }
    })
}

fn summarize_denoising_subset(reports: &[&TensorReport]) -> (Value, usize, bool, f64, f64, f64) {
    let mut count = 0_usize;
    let mut with_spikes = 0_usize;
    let mut any_tail_saturated = false;
    let mut sum_above_rank = 0_usize;
    let mut max_above_rank = 0_usize;
    let mut sum_ratio = 0.0_f64;
    let mut max_ratio = 0.0_f64;
    let mut raw_energy = 0.0_f64;
    let mut denoised_energy = 0.0_f64;
    let mut bulk_energy = 0.0_f64;
    let mut wr1 = 0.0_f64;
    let mut wr4 = 0.0_f64;
    let mut wr8 = 0.0_f64;
    let mut wr16 = 0.0_f64;
    let mut wr32 = 0.0_f64;
    let mut wr64 = 0.0_f64;

    for &report in reports {
        raw_energy += report.delta_frobenius_energy;
        let Some(den) = report.svd.as_ref().and_then(|s| s.denoised.as_ref()) else {
            continue;
        };
        count += 1;
        if den.above_bulk_rank > 0 {
            with_spikes += 1;
        }
        any_tail_saturated |= den.tail_saturated;
        sum_above_rank += den.above_bulk_rank;
        max_above_rank = max_above_rank.max(den.above_bulk_rank);
        sum_ratio += den.sigma_max_to_bulk_edge_ratio;
        max_ratio = max_ratio.max(den.sigma_max_to_bulk_edge_ratio);
        let w = den.denoised_signal_energy;
        denoised_energy += w;
        bulk_energy += den.estimated_bulk_energy_fraction * report.delta_frobenius_energy;
        if w > 0.0 {
            wr1 += w * den.denoised_cumulative_energy.r1;
            wr4 += w * den.denoised_cumulative_energy.r4;
            wr8 += w * den.denoised_cumulative_energy.r8;
            wr16 += w * den.denoised_cumulative_energy.r16;
            wr32 += w * den.denoised_cumulative_energy.r32;
            wr64 += w * den.denoised_cumulative_energy.r64;
        }
    }

    let mean_rank = if count > 0 {
        usize_to_f64(sum_above_rank) / usize_to_f64(count)
    } else {
        0.0
    };
    let mean_ratio = if count > 0 {
        sum_ratio / usize_to_f64(count)
    } else {
        0.0
    };
    let bulk_fraction = if raw_energy > 0.0 {
        (bulk_energy / raw_energy).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let denoised_share = if raw_energy > 0.0 {
        (denoised_energy / raw_energy).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let weighted_cum = if denoised_energy > 0.0 {
        let inv = 1.0 / denoised_energy;
        Some(CumulativeEnergy {
            r1: (wr1 * inv).clamp(0.0, 1.0),
            r4: (wr4 * inv).clamp(0.0, 1.0),
            r8: (wr8 * inv).clamp(0.0, 1.0),
            r16: (wr16 * inv).clamp(0.0, 1.0),
            r32: (wr32 * inv).clamp(0.0, 1.0),
            r64: (wr64 * inv).clamp(0.0, 1.0),
        })
    } else {
        None
    };

    let summary = json!({
        "tensorCount": count,
        "tensorsWithSpikes": with_spikes,
        "anyTailSaturated": any_tail_saturated,
        "meanAboveBulkRank": mean_rank,
        "maxAboveBulkRank": max_above_rank,
        "meanSigmaMaxToBulkEdgeRatio": mean_ratio,
        "maxSigmaMaxToBulkEdgeRatio": max_ratio,
        "rawDeltaEnergy": raw_energy,
        "denoisedSignalEnergy": denoised_energy,
        "estimatedBulkEnergyFraction": bulk_fraction,
        "denoisedShareOfRawEnergy": denoised_share,
        "energyWeightedDenoisedCumulativeEnergy": weighted_cum,
    });
    (
        summary,
        max_above_rank,
        any_tail_saturated,
        mean_ratio,
        max_ratio,
        denoised_energy,
    )
}

fn evaluate_spectral_denoising_screen(
    completed: bool,
    include_svd: bool,
    include_denoising: bool,
    skipped_count: usize,
    envelope: TargetEnvelope,
    reports: &[TensorReport],
) -> Value {
    let npu_reports: Vec<&TensorReport> = reports
        .iter()
        .filter(|r| {
            r.category == "linear-matrix" && TargetEnvelope::NpuFriendly.matches_family(&r.family)
        })
        .collect();
    let ffn_attn_reports: Vec<&TensorReport> = reports
        .iter()
        .filter(|r| {
            r.category == "linear-matrix"
                && matches!(
                    r.family.as_str(),
                    "ffn_gate"
                        | "ffn_up"
                        | "ffn_down"
                        | "attn_q"
                        | "attn_k"
                        | "attn_v"
                        | "attn_output"
                )
        })
        .collect();

    let (npu_summary, npu_max_rank, npu_sat, npu_mean_ratio, npu_max_ratio, npu_den_energy) =
        summarize_denoising_subset(&npu_reports);
    let (ffn_summary, ffn_max_rank, ffn_sat, ffn_mean_ratio, ffn_max_ratio, ffn_den_energy) =
        summarize_denoising_subset(&ffn_attn_reports);

    let use_npu = !npu_reports.is_empty();
    let (primary_max_rank, primary_sat, primary_mean_ratio, primary_max_ratio, primary_summary) =
        if use_npu {
            (
                npu_max_rank,
                npu_sat,
                npu_mean_ratio,
                npu_max_ratio,
                &npu_summary,
            )
        } else {
            (
                ffn_max_rank,
                ffn_sat,
                ffn_mean_ratio,
                ffn_max_ratio,
                &ffn_summary,
            )
        };

    let npu_share_of_denoised = if ffn_den_energy > 0.0 {
        (npu_den_energy / ffn_den_energy).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let mut reasons = Vec::new();
    let verdict = if !include_svd || !include_denoising {
        reasons.push("Spectral denoising or SVD was disabled in request scope.".to_owned());
        "denoising-disabled"
    } else if !completed {
        reasons.push(format!(
            "Audit stopped before completing {skipped_count} remaining tensor(s)."
        ));
        "incomplete-budget-exceeded"
    } else if primary_max_rank == 0 {
        reasons.push(
            "Zero singular values cleared the Rademacher-null / Gavish–Donoho bulk edge."
                .to_owned(),
        );
        "no-above-bulk-spikes"
    } else if primary_sat {
        reasons.push(format!(
            "Top singular spectrum remained saturated above the bulk edge at maxRank (maxRatio={primary_max_ratio:.2}x)."
        ));
        "saturated-high-rank-tail"
    } else {
        reasons.push(format!(
            "Non-saturated coherent spike core detected above Rademacher-null bulk edge (envelope='{}', maxRank={primary_max_rank}, maxRatio={primary_max_ratio:.2}x, meanRatio={primary_mean_ratio:.2}x).",
            envelope.as_str()
        ));
        "coherent-low-rank-core-detected"
    };

    json!({
        "verdict": verdict,
        "anyTailSaturated": primary_sat,
        "meanSigmaMaxToBulkEdgeRatio": primary_mean_ratio,
        "maxSigmaMaxToBulkEdgeRatio": primary_max_ratio,
        "meanAboveBulkRank": primary_summary["meanAboveBulkRank"],
        "maxAboveBulkRank": primary_max_rank,
        "estimatedBulkEnergyFraction": primary_summary["estimatedBulkEnergyFraction"],
        "denoisedShareOfRawEnergy": primary_summary["denoisedShareOfRawEnergy"],
        "npuFriendlyShareOfDenoisedSignalEnergy": npu_share_of_denoised,
        "npuFriendlySummary": npu_summary,
        "ffnAndFullAttnSummary": ffn_summary,
        "reasons": reasons,
    })
}

fn open_sources(
    source: &SourceSpec,
    http_concurrency: usize,
) -> crate::Result<(ShardSource, ShardSource)> {
    let per_side_http = (http_concurrency / 2).max(1);
    match source {
        SourceSpec::PinnedDaisyrose => Ok((
            ShardSource::https(
                HUGGINGFACE_BASE_URL.to_owned(),
                DAISYROSE_REPO.to_owned(),
                DAISYROSE_REVISION.to_owned(),
                per_side_http,
            )?,
            ShardSource::https(
                HUGGINGFACE_BASE_URL.to_owned(),
                QWEN38_REPO.to_owned(),
                QWEN38_REVISION.to_owned(),
                per_side_http,
            )?,
        )),
        SourceSpec::PinnedHuggingFace {
            candidate_repository,
            candidate_revision,
            candidate_license: _,
            base_repository,
            base_revision,
        } => Ok((
            ShardSource::https(
                HUGGINGFACE_BASE_URL.to_owned(),
                candidate_repository.clone(),
                candidate_revision.clone(),
                per_side_http,
            )?,
            ShardSource::https(
                HUGGINGFACE_BASE_URL.to_owned(),
                base_repository.clone(),
                base_revision.clone(),
                per_side_http,
            )?,
        )),
        SourceSpec::LocalDirectories {
            candidate_directory,
            base_directory,
        } => Ok((
            ShardSource::local(candidate_directory.clone())?,
            ShardSource::local(base_directory.clone())?,
        )),
    }
}

fn fetch_indices(
    cand_src: &ShardSource,
    base_src: &ShardSource,
    deadline: Instant,
) -> crate::Result<(SafetensorsIndex, SafetensorsIndex)> {
    std::thread::scope(|scope| {
        let c_handle = scope.spawn(|| cand_src.fetch_index(deadline));
        let b_res = base_src.fetch_index(deadline);
        let c_res = c_handle
            .join()
            .unwrap_or_else(|_| Err("Candidate index thread panicked.".into()))?;
        let b_res = b_res?;
        Ok((c_res, b_res))
    })
}

fn plan_response(request: &Request) -> crate::Result<Value> {
    let source = request
        .source
        .as_ref()
        .ok_or("plan requires a model diff source.")?;
    let envelope = effective_envelope(request)?;
    let selected = select_tensors(&request.scope, envelope);
    let est_one: u64 = selected.iter().map(estimated_bf16_bytes).sum();
    let est_pair = est_one.saturating_mul(2);
    let receipt = json!({
        "schemaVersion": 1,
        "operation": "plan",
        "status": "planned-no-io",
        "source": source_summary(source),
        "scope": resolved_scope_json(&request.scope, envelope),
        "controls": request.controls,
        "selectedTensorCount": selected.len(),
        "estimatedBf16BytesOneModel": est_one,
        "estimatedBf16BytesPair": est_pair,
        "gate1Criteria": {
            "passTargetKneeE32": 0.65,
            "hardStopE32Below": 0.50,
            "hardStopNonMatrixShareAtLeast": 0.50,
            "hardStopPureNonMatrixToLinearMeanRatioAbove": 5.0
        }
    });
    if let Some(output) = &request.output_path {
        crate::bounded_input::write_pretty_json_new(output, &receipt)?;
    }
    Ok(receipt)
}

#[expect(
    clippy::too_many_lines,
    reason = "Orchestrates read-only intake, small-tensor coalescing, block SVD, spectral denoising, power endpoints, and Gate 1 aggregation."
)]
fn run_audit(request: &Request) -> crate::Result<Value> {
    let started = Instant::now();
    let power_before = crate::host_cpu::power_snapshot();
    let deadline = started
        .checked_add(Duration::from_secs(request.controls.time_budget_seconds))
        .ok_or("Audit deadline overflow.")?;

    let source = request
        .source
        .as_ref()
        .ok_or("audit requires a model diff source.")?;
    let envelope = effective_envelope(request)?;
    let (cand_src, base_src) = open_sources(source, request.controls.http_concurrency)?;
    let (cand_idx, base_idx) = fetch_indices(&cand_src, &base_src, deadline)?;

    let selected = select_tensors(&request.scope, envelope);
    let prepared = resolve_and_validate_tensors(
        selected, &cand_src, &cand_idx, &base_src, &base_idx, deadline,
    )?;

    let mut small_indices = Vec::new();
    let mut medium_indices = Vec::new();
    let mut streaming_indices = Vec::new();
    for (idx, item) in prepared.iter().enumerate() {
        if item.element_count > STREAMING_THRESHOLD_ELEMENTS && item.desc.shape.len() == 2 {
            streaming_indices.push(idx);
        } else if item.byte_size <= SMALL_TENSOR_MAX_BYTES {
            small_indices.push(idx);
        } else {
            medium_indices.push(idx);
        }
    }

    let mut reports = Vec::with_capacity(prepared.len());
    let mut skipped_tensors = Vec::new();
    let mut budget_exceeded = false;
    let k_sketch = request
        .scope
        .max_rank
        .saturating_add(request.scope.oversampling);
    let workers = request.controls.workers;
    let include_denoising = request.scope.include_spectral_denoising;

    // Phase 1: Small coalesced tensors (non-matrix + ssm_alpha/ssm_beta).
    if !small_indices.is_empty() {
        if Instant::now() >= deadline {
            budget_exceeded = true;
            for &idx in &small_indices {
                skipped_tensors.push(prepared[idx].spec.gguf_name.clone());
            }
        } else {
            let cand_items: Vec<(usize, &str, u64, u64)> = small_indices
                .iter()
                .map(|&idx| {
                    let p = &prepared[idx];
                    (idx, p.cand_shard.as_str(), p.desc.abs_begin, p.desc.abs_end)
                })
                .collect();
            let base_items: Vec<(usize, &str, u64, u64)> = small_indices
                .iter()
                .filter_map(|&idx| {
                    let p = &prepared[idx];
                    base_src
                        .descriptor(&p.base_shard, &p.spec.hf_name)
                        .ok()
                        .map(|bd| (idx, p.base_shard.as_str(), bd.abs_begin, bd.abs_end))
                })
                .collect();
            if base_items.len() != cand_items.len() {
                return Err("Base descriptor lookup failed for small tensor batch.".into());
            }
            let (cand_bufs, base_bufs) = std::thread::scope(|scope| {
                let c_handle =
                    scope.spawn(|| fetch_coalesced_side(&cand_src, &cand_items, deadline));
                let b_res = fetch_coalesced_side(&base_src, &base_items, deadline);
                let c_res = c_handle
                    .join()
                    .unwrap_or_else(|_| Err("Candidate small-tensor thread panicked.".into()))?;
                let b_res = b_res?;
                Ok::<_, String>((c_res, b_res))
            })?;

            for &idx in &small_indices {
                let p = &prepared[idx];
                let cb = cand_bufs
                    .get(&idx)
                    .ok_or("Missing candidate small-tensor slice.")?;
                let bb = base_bufs
                    .get(&idx)
                    .ok_or("Missing base small-tensor slice.")?;
                let (rows, cols) = if p.desc.shape.len() == 2 {
                    (p.desc.shape[0], p.desc.shape[1])
                } else {
                    (1, p.element_count)
                };
                let mut delta = vec![0.0_f32; p.element_count];
                let mut stats = diff_chunk_rows(p.dtype, cb, bb, rows, cols, &mut delta, 1)?;
                if p.desc.shape.len() != 2 {
                    stats.changed_rows = None;
                }
                let svd = if request.scope.include_svd && p.desc.shape.len() == 2 {
                    Some(randomized_subspace_svd(
                        &p.spec.hf_name,
                        &delta,
                        rows,
                        cols,
                        k_sketch,
                        request.scope.power_iterations,
                        stats.delta_energy,
                        include_denoising,
                        request.scope.max_rank,
                        workers,
                    )?)
                } else {
                    None
                };
                reports.push(build_tensor_report(
                    &p.spec,
                    p.dtype,
                    p.desc.shape.clone(),
                    p.element_count,
                    p.byte_size,
                    stats,
                    svd,
                ));
            }
        }
    }

    // Phase 2: Medium block linear matrices.
    for (pos, &idx) in medium_indices.iter().enumerate() {
        if Instant::now() >= deadline {
            budget_exceeded = true;
            for &rem in &medium_indices[pos..] {
                skipped_tensors.push(prepared[rem].spec.gguf_name.clone());
            }
            break;
        }
        crate::startup_cancel::checkpoint()?;
        let p = &prepared[idx];
        let b_desc = base_src.descriptor(&p.base_shard, &p.spec.hf_name)?;
        let (cb, bb) = match fetch_pair_slices(
            &cand_src,
            &p.cand_shard,
            p.desc.abs_begin,
            p.byte_size,
            &base_src,
            &p.base_shard,
            b_desc.abs_begin,
            p.byte_size,
            deadline,
        ) {
            Ok(pair) => pair,
            Err(_) if Instant::now() >= deadline => {
                budget_exceeded = true;
                for &rem in &medium_indices[pos..] {
                    skipped_tensors.push(prepared[rem].spec.gguf_name.clone());
                }
                break;
            }
            Err(err) => return Err(err),
        };
        let (rows, cols) = if p.desc.shape.len() == 2 {
            (p.desc.shape[0], p.desc.shape[1])
        } else {
            (1, p.element_count)
        };
        let mut delta = Vec::new();
        delta
            .try_reserve_exact(p.element_count)
            .map_err(|e| format!("Cannot allocate delta matrix for '{}': {e}", p.spec.hf_name))?;
        delta.resize(p.element_count, 0.0_f32);
        let mut stats = diff_chunk_rows(p.dtype, &cb, &bb, rows, cols, &mut delta, workers)?;
        if p.desc.shape.len() != 2 {
            stats.changed_rows = None;
        }
        drop(cb);
        drop(bb);
        let svd = if request.scope.include_svd && p.desc.shape.len() == 2 {
            Some(randomized_subspace_svd(
                &p.spec.hf_name,
                &delta,
                rows,
                cols,
                k_sketch,
                request.scope.power_iterations,
                stats.delta_energy,
                include_denoising,
                request.scope.max_rank,
                workers,
            )?)
        } else {
            None
        };
        reports.push(build_tensor_report(
            &p.spec,
            p.dtype,
            p.desc.shape.clone(),
            p.element_count,
            p.byte_size,
            stats,
            svd,
        ));
    }

    // Phase 3: Huge vocabulary matrices via chunked streaming + single-pass two-sided sketch.
    for (pos, &idx) in streaming_indices.iter().enumerate() {
        if Instant::now() >= deadline {
            budget_exceeded = true;
            for &rem in &streaming_indices[pos..] {
                skipped_tensors.push(prepared[rem].spec.gguf_name.clone());
            }
            break;
        }
        let p = &prepared[idx];
        let b_desc = base_src.descriptor(&p.base_shard, &p.spec.hf_name)?;
        let rows = p.desc.shape[0];
        let cols = p.desc.shape[1];
        let row_bytes = u64::try_from(cols * p.dtype.byte_width()).map_err(|e| e.to_string())?;
        let mut sketch = if request.scope.include_svd {
            Some(StreamingTwoSidedSketch::new(
                &p.spec.hf_name,
                rows,
                cols,
                k_sketch,
            )?)
        } else {
            None
        };
        let mut total_stats = TensorDiffStats {
            base_energy: 0.0,
            cand_energy: 0.0,
            delta_energy: 0.0,
            sum_abs_delta: 0.0,
            max_abs_delta: 0.0,
            nonzero_count: 0,
            changed_rows: Some(0),
        };
        let mut aborted_mid_tensor = false;
        let mut row_start = 0_usize;
        while row_start < rows {
            if Instant::now() >= deadline {
                budget_exceeded = true;
                aborted_mid_tensor = true;
                break;
            }
            crate::startup_cancel::checkpoint()?;
            let chunk_rows = (rows - row_start).min(VOCAB_CHUNK_ROWS);
            let byte_offset = u64::try_from(row_start).map_err(|e| e.to_string())? * row_bytes;
            let chunk_bytes = u64::try_from(chunk_rows).map_err(|e| e.to_string())? * row_bytes;
            let (cb, bb) = match fetch_pair_slices(
                &cand_src,
                &p.cand_shard,
                p.desc.abs_begin + byte_offset,
                chunk_bytes,
                &base_src,
                &p.base_shard,
                b_desc.abs_begin + byte_offset,
                chunk_bytes,
                deadline,
            ) {
                Ok(pair) => pair,
                Err(_) if Instant::now() >= deadline => {
                    budget_exceeded = true;
                    aborted_mid_tensor = true;
                    break;
                }
                Err(err) => return Err(err),
            };
            let mut chunk_delta = Vec::new();
            chunk_delta
                .try_reserve_exact(chunk_rows * cols)
                .map_err(|e| format!("Cannot allocate vocab chunk buffer: {e}"))?;
            chunk_delta.resize(chunk_rows * cols, 0.0_f32);
            let chunk_stats = diff_chunk_rows(
                p.dtype,
                &cb,
                &bb,
                chunk_rows,
                cols,
                &mut chunk_delta,
                workers,
            )?;
            drop(cb);
            drop(bb);
            total_stats.base_energy += chunk_stats.base_energy;
            total_stats.cand_energy += chunk_stats.cand_energy;
            total_stats.delta_energy += chunk_stats.delta_energy;
            total_stats.sum_abs_delta += chunk_stats.sum_abs_delta;
            total_stats.max_abs_delta = total_stats.max_abs_delta.max(chunk_stats.max_abs_delta);
            total_stats.nonzero_count += chunk_stats.nonzero_count;
            total_stats.changed_rows =
                Some(total_stats.changed_rows.unwrap_or(0) + chunk_stats.changed_rows.unwrap_or(0));
            if let Some(sk) = sketch.as_mut() {
                sk.update_chunk(row_start, &chunk_delta, chunk_rows, workers)?;
            }
            row_start += chunk_rows;
        }
        if aborted_mid_tensor {
            for &rem in &streaming_indices[pos..] {
                skipped_tensors.push(prepared[rem].spec.gguf_name.clone());
            }
            break;
        }
        let svd = match sketch {
            Some(sk) => Some(sk.finish(
                total_stats.delta_energy,
                include_denoising,
                request.scope.max_rank,
                workers,
            )?),
            None => None,
        };
        reports.push(build_tensor_report(
            &p.spec,
            p.dtype,
            p.desc.shape.clone(),
            p.element_count,
            p.byte_size,
            total_stats,
            svd,
        ));
    }

    let power_after = crate::host_cpu::power_snapshot();
    let power_conditions = bonsai_cpu_policy::power::compare_endpoints(&power_before, &power_after);

    let mut all_audited = MetricsAccum::default();
    let mut non_vocab_all = MetricsAccum::default();
    let mut pure_non_matrix = MetricsAccum::default();
    let mut non_matrix_small_gdn = MetricsAccum::default();
    let mut linear_matrices = MetricsAccum::default();
    let mut ffn_and_full_attn = MetricsAccum::default();
    let mut npu_friendly = MetricsAccum::default();
    let mut cand_a_ffn_down_only = MetricsAccum::default();
    let mut cand_a_down_attn_o = MetricsAccum::default();
    let mut cand_b_all = MetricsAccum::default();
    let mut cand_b_gate_up_qkv = MetricsAccum::default();
    let mut gdn_linear = MetricsAccum::default();
    let mut vocab_matrices = MetricsAccum::default();
    let mut by_family: BTreeMap<String, (String, MetricsAccum)> = BTreeMap::new();

    for report in &reports {
        all_audited.add(report);
        let fam_entry = by_family
            .entry(report.family.clone())
            .or_insert_with(|| (report.category.clone(), MetricsAccum::default()));
        fam_entry.1.add(report);

        match report.category.as_str() {
            "vocab-matrix" => vocab_matrices.add(report),
            "non-matrix" => {
                non_vocab_all.add(report);
                pure_non_matrix.add(report);
                non_matrix_small_gdn.add(report);
            }
            "gdn-small-matrix" => {
                non_vocab_all.add(report);
                non_matrix_small_gdn.add(report);
            }
            "linear-matrix" => {
                non_vocab_all.add(report);
                linear_matrices.add(report);
                if TargetEnvelope::NpuFriendly.matches_family(&report.family) {
                    npu_friendly.add(report);
                }
                match report.family.as_str() {
                    "ffn_down" => {
                        ffn_and_full_attn.add(report);
                        cand_a_ffn_down_only.add(report);
                        cand_a_down_attn_o.add(report);
                        cand_b_all.add(report);
                    }
                    "attn_output" => {
                        ffn_and_full_attn.add(report);
                        cand_a_down_attn_o.add(report);
                        cand_b_all.add(report);
                    }
                    "ffn_gate" | "ffn_up" | "attn_q" | "attn_k" | "attn_v" => {
                        ffn_and_full_attn.add(report);
                        cand_b_all.add(report);
                        cand_b_gate_up_qkv.add(report);
                    }
                    "attn_qkv" | "attn_gate" | "ssm_out" => {
                        gdn_linear.add(report);
                        if report.family != "ssm_out" {
                            cand_b_gate_up_qkv.add(report);
                        }
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }

    let audited_energy = all_audited.total_delta_energy;
    let non_vocab_energy = non_vocab_all.total_delta_energy;
    let linear_energy = linear_matrices.total_delta_energy;

    let mut family_json = serde_json::Map::new();
    for (family, (category, accum)) in &by_family {
        let mut entry = accum.to_json(audited_energy, non_vocab_energy, linear_energy);
        entry["category"] = json!(category);
        family_json.insert(family.clone(), entry);
    }

    let completed = !budget_exceeded && skipped_tensors.is_empty();
    let gate1_screen = evaluate_gate1(
        completed,
        request.scope.include_svd,
        skipped_tensors.len(),
        &all_audited,
        &non_vocab_all,
        &pure_non_matrix,
        &non_matrix_small_gdn,
        &linear_matrices,
        &ffn_and_full_attn,
        &cand_a_down_attn_o,
        &cand_b_gate_up_qkv,
    );
    let spectral_denoising_screen = evaluate_spectral_denoising_screen(
        completed,
        request.scope.include_svd,
        include_denoising,
        skipped_tensors.len(),
        envelope,
        &reports,
    );

    let elapsed_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
    let cand_bytes_read = cand_src.transferred_bytes();
    let base_bytes_read = base_src.transferred_bytes();
    let receipt = json!({
        "schemaVersion": 1,
        "operation": "audit",
        "completed": completed,
        "budgetExceeded": budget_exceeded,
        "elapsedMilliseconds": elapsed_ms,
        "source": source_summary(source),
        "scope": resolved_scope_json(&request.scope, envelope),
        "controls": request.controls,
        "candidateIndexSha256": cand_idx.sha256,
        "baseIndexSha256": base_idx.sha256,
        "transferredBytes": {
            "candidate": cand_bytes_read,
            "base": base_bytes_read,
            "total": cand_bytes_read.saturating_add(base_bytes_read)
        },
        "powerBefore": power_before,
        "powerAfter": power_after,
        "powerConditions": power_conditions,
        "auditedTensorCount": reports.len(),
        "skippedTensors": skipped_tensors,
        "byFamily": family_json,
        "aggregates": {
            "allAudited": all_audited.to_json(audited_energy, non_vocab_energy, linear_energy),
            "nonVocabAll": non_vocab_all.to_json(audited_energy, non_vocab_energy, linear_energy),
            "pureNonMatrix": pure_non_matrix.to_json(audited_energy, non_vocab_energy, linear_energy),
            "nonMatrixAndSmallGdn": non_matrix_small_gdn.to_json(audited_energy, non_vocab_energy, linear_energy),
            "linearMatrices": linear_matrices.to_json(audited_energy, non_vocab_energy, linear_energy),
            "ffnAndFullAttnLinearMatrices": ffn_and_full_attn.to_json(audited_energy, non_vocab_energy, linear_energy),
            "npuFriendly": npu_friendly.to_json(audited_energy, non_vocab_energy, linear_energy),
            "candidateA_ffnDownOnly": cand_a_ffn_down_only.to_json(audited_energy, non_vocab_energy, linear_energy),
            "candidateA_ffnDownAndAttnOutput": cand_a_down_attn_o.to_json(audited_energy, non_vocab_energy, linear_energy),
            "candidateB_allFfnAndFullAttn": cand_b_all.to_json(audited_energy, non_vocab_energy, linear_energy),
            "candidateB_gateUpQkvOnly": cand_b_gate_up_qkv.to_json(audited_energy, non_vocab_energy, linear_energy),
            "gdnLinearMatrices": gdn_linear.to_json(audited_energy, non_vocab_energy, linear_energy),
            "vocabMatrices": vocab_matrices.to_json(audited_energy, non_vocab_energy, linear_energy),
        },
        "gate1Screen": gate1_screen,
        "spectralDenoisingScreen": spectral_denoising_screen,
        "tensors": reports,
    });
    if let Some(output) = &request.output_path {
        crate::bounded_input::write_pretty_json_new(output, &receipt)?;
    }
    Ok(receipt)
}

fn append_source_provenance<'a>(source: &'a SourceSpec, meta: &mut Vec<(&'a str, &'a str)>) {
    match source {
        SourceSpec::PinnedDaisyrose => {
            meta.push(("bonsai.composition.candidate_repo", DAISYROSE_REPO));
            meta.push(("bonsai.composition.candidate_revision", DAISYROSE_REVISION));
            meta.push(("bonsai.composition.base_repo", QWEN38_REPO));
            meta.push(("bonsai.composition.base_revision", QWEN38_REVISION));
        }
        SourceSpec::PinnedHuggingFace {
            candidate_repository,
            candidate_revision,
            candidate_license,
            base_repository,
            base_revision,
        } => {
            meta.push(("bonsai.composition.candidate_repo", candidate_repository));
            meta.push(("bonsai.composition.candidate_revision", candidate_revision));
            meta.push(("bonsai.composition.candidate_license", candidate_license));
            meta.push(("bonsai.composition.base_repo", base_repository));
            meta.push(("bonsai.composition.base_revision", base_revision));
        }
        SourceSpec::LocalDirectories { .. } => {}
    }
}

fn run_export_denoised(request: &Request) -> crate::Result<Value> {
    run_export_denoised_with_dims(request, &swift_base_dims_map())
}

#[expect(
    clippy::too_many_lines,
    reason = "Streams target matrices, extracts denoised low-rank factors, publishes atomic GGUF, and self-verifies without echoing local paths."
)]
fn run_export_denoised_with_dims(
    request: &Request,
    verify_dims: &BTreeMap<String, (usize, usize, &'static str)>,
) -> crate::Result<Value> {
    let started = Instant::now();
    let power_before = crate::host_cpu::power_snapshot();
    let deadline = started
        .checked_add(Duration::from_secs(request.controls.time_budget_seconds))
        .ok_or("Export deadline overflow.")?;

    let source = request
        .source
        .as_ref()
        .ok_or("export-denoised requires a model diff source.")?;
    let output_path = request
        .output_path
        .as_ref()
        .ok_or("export-denoised requires outputPath.")?;
    let envelope = effective_envelope(request)?;
    let (cand_src, base_src) = open_sources(source, request.controls.http_concurrency)?;
    let (cand_idx, base_idx) = fetch_indices(&cand_src, &base_src, deadline)?;

    let selected = select_tensors(&request.scope, envelope);
    let prepared = resolve_and_validate_tensors(
        selected, &cand_src, &cand_idx, &base_src, &base_idx, deadline,
    )?;

    struct ExtractedEntry {
        gguf_name: String,
        in_features: usize,
        out_features: usize,
        factors: svd::DenoisedFactors,
    }

    let k_sketch = request
        .scope
        .max_rank
        .saturating_add(request.scope.oversampling);
    let workers = request.controls.workers;
    let export_dtype = request.scope.export_dtype;

    for p in &prepared {
        if p.desc.shape.len() != 2 {
            return Err(format!(
                "export-denoised expected 2D matrix for '{}'.",
                p.spec.hf_name
            ));
        }
    }

    let mut extracted = Vec::with_capacity(prepared.len());
    let mut reports = Vec::with_capacity(prepared.len());
    let mut skipped_tensors = Vec::new();
    let mut budget_exceeded = false;

    // 1-slot HTTP Range prefetch pipeline: keep at most one next-tensor slice
    // pair fetching concurrently while computing denoised factors for the current tensor.
    std::thread::scope(|scope| -> crate::Result<()> {
        let start_fetch = |index: usize| -> crate::Result<
            std::thread::ScopedJoinHandle<'_, crate::Result<(Vec<u8>, Vec<u8>)>>,
        > {
            let item = &prepared[index];
            let b_desc = base_src.descriptor(&item.base_shard, &item.spec.hf_name)?;
            let cand_ref = &cand_src;
            let base_ref = &base_src;
            let cand_shard = &item.cand_shard;
            let base_shard = &item.base_shard;
            let cand_begin = item.desc.abs_begin;
            let base_begin = b_desc.abs_begin;
            let byte_size = item.byte_size;
            Ok(scope.spawn(move || {
                fetch_pair_slices(
                    cand_ref, cand_shard, cand_begin, byte_size, base_ref, base_shard, base_begin,
                    byte_size, deadline,
                )
            }))
        };

        let mut prefetched = (!prepared.is_empty() && Instant::now() < deadline)
            .then(|| start_fetch(0))
            .transpose()?;

        for (pos, p) in prepared.iter().enumerate() {
            if Instant::now() >= deadline {
                if let Some(handle) = prefetched.take() {
                    let _ = handle.join();
                }
                budget_exceeded = true;
                for rem in &prepared[pos..] {
                    skipped_tensors.push(rem.spec.gguf_name.clone());
                }
                break;
            }
            crate::startup_cancel::checkpoint()?;
            let rows = p.desc.shape[0];
            let cols = p.desc.shape[1];
            let fetched = match prefetched.take() {
                Some(handle) => handle
                    .join()
                    .unwrap_or_else(|_| Err("Prefetch thread panicked.".into())),
                None => start_fetch(pos)?
                    .join()
                    .unwrap_or_else(|_| Err("Fetch thread panicked.".into())),
            };
            let (cb, bb) = match fetched {
                Ok(pair) => pair,
                Err(_) if Instant::now() >= deadline => {
                    budget_exceeded = true;
                    for rem in &prepared[pos..] {
                        skipped_tensors.push(rem.spec.gguf_name.clone());
                    }
                    break;
                }
                Err(err) => return Err(err),
            };
            if pos + 1 < prepared.len() && Instant::now() < deadline {
                prefetched = Some(start_fetch(pos + 1)?);
            }

            let mut delta = Vec::new();
            delta.try_reserve_exact(p.element_count).map_err(|e| {
                format!("Cannot allocate delta matrix for '{}': {e}", p.spec.hf_name)
            })?;
            delta.resize(p.element_count, 0.0_f32);
            let stats = diff_chunk_rows(p.dtype, &cb, &bb, rows, cols, &mut delta, workers)?;
            drop(cb);
            drop(bb);

            let factors = randomized_subspace_denoised_factors_with_mode(
                &p.spec.hf_name,
                &delta,
                rows,
                cols,
                k_sketch,
                request.scope.power_iterations,
                stats.delta_energy,
                request.scope.max_rank,
                request.scope.min_spike_ratio,
                request.scope.shrinkage_mode,
                request.scope.shrinkage_blend,
                workers,
            )?;
            let report = build_tensor_report(
                &p.spec,
                p.dtype,
                p.desc.shape.clone(),
                p.element_count,
                p.byte_size,
                stats,
                Some(factors.spectrum.clone()),
            );
            reports.push(report);
            if factors.rank > 0 {
                extracted.push(ExtractedEntry {
                    gguf_name: p.spec.gguf_name.clone(),
                    in_features: cols,
                    out_features: rows,
                    factors,
                });
            }
        }
        Ok(())
    })?;

    let power_after = crate::host_cpu::power_snapshot();
    let power_conditions = bonsai_cpu_policy::power::compare_endpoints(&power_before, &power_after);
    let completed = !budget_exceeded && skipped_tensors.is_empty();
    let spectral_summary = evaluate_spectral_denoising_screen(
        completed,
        true,
        true,
        skipped_tensors.len(),
        envelope,
        &reports,
    );
    let elapsed_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
    let cand_bytes_read = cand_src.transferred_bytes();
    let base_bytes_read = base_src.transferred_bytes();

    if !completed {
        return Ok(json!({
            "schemaVersion": 1,
            "operation": "export-denoised",
            "published": false,
            "status": "budget-exceeded-not-published",
            "completed": false,
            "budgetExceeded": true,
            "elapsedMilliseconds": elapsed_ms,
            "targetEnvelope": envelope,
            "exportDtype": export_dtype,
            "skippedTensors": skipped_tensors,
            "transferredBytes": {
                "candidate": cand_bytes_read,
                "base": base_bytes_read,
                "total": cand_bytes_read.saturating_add(base_bytes_read)
            },
            "powerBefore": power_before,
            "powerAfter": power_after,
            "powerConditions": power_conditions,
            "spectralSummary": spectral_summary,
        }));
    }

    if extracted.is_empty() {
        return Ok(json!({
            "schemaVersion": 1,
            "operation": "export-denoised",
            "published": false,
            "status": "no-above-bulk-spikes",
            "completed": true,
            "budgetExceeded": false,
            "elapsedMilliseconds": elapsed_ms,
            "targetEnvelope": envelope,
            "exportDtype": export_dtype,
            "pairCount": 0,
            "transferredBytes": {
                "candidate": cand_bytes_read,
                "base": base_bytes_read,
                "total": cand_bytes_read.saturating_add(base_bytes_read)
            },
            "powerBefore": power_before,
            "powerAfter": power_after,
            "powerConditions": power_conditions,
            "spectralSummary": spectral_summary,
        }));
    }

    let max_rank = extracted
        .iter()
        .map(|entry| entry.factors.rank)
        .max()
        .unwrap_or(1);
    let global_alpha = usize_to_f32(max_rank);
    let mut pairs = Vec::with_capacity(extracted.len());
    for entry in &extracted {
        let (lora_a_f32, lora_b_f32) = build_balanced_factors(
            &entry.factors.u_cols,
            &entry.factors.v_cols,
            &entry.factors.shrunk_sigmas,
            global_alpha,
            export_dtype,
        )?;
        pairs.push(ExportTensorPair {
            gguf_base_name: entry.gguf_name.clone(),
            in_features: entry.in_features,
            out_features: entry.out_features,
            rank: entry.factors.rank,
            lora_a_f32,
            lora_b_f32,
        });
    }

    let mut meta: Vec<(&str, &str)> = vec![
        (
            "bonsai.composition.method",
            "rademacher-null-gavish-donoho-v1",
        ),
        ("bonsai.composition.envelope", envelope.as_str()),
        (
            "bonsai.composition.shrinkage_mode",
            request.scope.shrinkage_mode.as_str(),
        ),
        (
            "bonsai.composition.candidate_index_sha256",
            &cand_idx.sha256,
        ),
        ("bonsai.composition.base_index_sha256", &base_idx.sha256),
    ];
    append_source_provenance(source, &mut meta);

    let written = write_lora_gguf(output_path, export_dtype, global_alpha, &meta, &pairs)?;
    let verification = verify_lora_gguf_with_dims(
        output_path,
        &VerifyPolicy {
            expected_sha256: Some(written.sha256.clone()),
            expected_envelope: Some(envelope.as_str().to_owned()),
            require_npu_friendly: matches!(
                envelope,
                TargetEnvelope::ResidualWriters | TargetEnvelope::NpuFriendly
            ),
            require_no_grouped_ssm: true,
            max_allowed_rank: Some(request.scope.max_rank),
        },
        verify_dims,
    )?;

    Ok(json!({
        "schemaVersion": 1,
        "operation": "export-denoised",
        "published": true,
        "status": "published-and-verified",
        "completed": true,
        "budgetExceeded": false,
        "elapsedMilliseconds": elapsed_ms,
        "targetEnvelope": envelope,
        "exportDtype": export_dtype,
        "alpha": global_alpha,
        "bytes": written.bytes,
        "sha256": written.sha256,
        "tensorCount": written.tensor_count,
        "pairCount": pairs.len(),
        "candidateIndexSha256": cand_idx.sha256,
        "baseIndexSha256": base_idx.sha256,
        "transferredBytes": {
            "candidate": cand_bytes_read,
            "base": base_bytes_read,
            "total": cand_bytes_read.saturating_add(base_bytes_read)
        },
        "powerBefore": power_before,
        "powerAfter": power_after,
        "powerConditions": power_conditions,
        "spectralSummary": spectral_summary,
        "verification": verification,
        "tensors": reports,
    }))
}

fn run_verify_adapter(request: &Request) -> crate::Result<Value> {
    let adapter_path = request
        .adapter_path
        .as_ref()
        .ok_or("verify-adapter requires adapterPath.")?;
    verify_lora_gguf(adapter_path, &request.verify_policy)
}

/// Runs read-only BF16 weight-diff/spectral-denoising audits (`plan`, `audit`),
/// denoised GGUF LoRA export (`export-denoised`), or adapter verification (`verify-adapter`).
///
/// # Errors
/// Rejects unsupported schema versions, invalid scopes or budgets, unsafe or existing
/// output paths, mismatched safetensors or GGUF layouts, nonfinite weights, or failed range I/O.
pub fn run(request: Value) -> crate::Result<Value> {
    let parsed: Request = serde_json::from_value(request)
        .map_err(|e| format!("Invalid lora-diff-audit request: {e}"))?;
    validate_request(&parsed)?;
    match parsed.operation {
        Operation::Plan => plan_response(&parsed),
        Operation::Audit => run_audit(&parsed),
        Operation::ExportDenoised => run_export_denoised(&parsed),
        Operation::VerifyAdapter => run_verify_adapter(&parsed),
        Operation::MergeAdapters => transform::run_merge_adapters(&parsed),
        Operation::EditAdapter => transform::run_edit_adapter(&parsed),
        Operation::ReallocateRanks => reallocate::run_reallocate_ranks(&parsed),
        Operation::FoldPatch => activation_edit::run_fold_patch(&parsed),
        Operation::RealignFactors => activation_edit::run_realign_factors(&parsed),
        Operation::GenerateTestAdapter => readiness::run_generate_test_adapter(&parsed),
        Operation::GenerateLock => readiness::run_generate_lock(&parsed),
        Operation::DrG2Plan => readiness::run_dr_g2_plan(&parsed),
        Operation::DrG2Score => readiness::run_dr_g2_score(&parsed),
        Operation::ImportPeftLora => peft_import::run_import_peft_lora(&parsed),
    }
}

#[cfg(test)]
mod pinned_huggingface_tests {
    use super::*;

    const CAND_REV: &str = "0123456789abcdef0123456789abcdef01234567";
    const BASE_REV: &str = "fedcba9876543210fedcba9876543210fedcba98";

    #[test]
    fn pinned_huggingface_plan_provenance_and_validation() -> crate::Result<()> {
        let valid_source = json!({
            "kind": "pinned-huggingface",
            "candidateRepository": "barozp/Qwen3.8-27B-Opus-Distill",
            "candidateRevision": CAND_REV,
            "candidateLicense": "Apache-2.0",
            "baseRepository": "Qwen/Qwen3.8-27B",
            "baseRevision": BASE_REV,
        });

        let plan = run(json!({
            "schemaVersion": 1,
            "operation": "plan",
            "source": valid_source.clone(),
        }))?;
        assert_eq!(plan["status"], "planned-no-io");
        assert_eq!(plan["source"]["kind"], "pinned-huggingface");
        assert_eq!(
            plan["source"]["candidate"]["repository"],
            "barozp/Qwen3.8-27B-Opus-Distill"
        );
        assert_eq!(plan["source"]["candidate"]["revision"], CAND_REV);
        assert_eq!(plan["source"]["candidate"]["license"], "Apache-2.0");
        assert_eq!(plan["source"]["candidate"]["baseUrl"], HUGGINGFACE_BASE_URL);
        assert_eq!(plan["source"]["base"]["repository"], "Qwen/Qwen3.8-27B");
        assert_eq!(plan["source"]["base"]["revision"], BASE_REV);
        assert_eq!(plan["source"]["base"]["baseUrl"], HUGGINGFACE_BASE_URL);

        let spec: SourceSpec = serde_json::from_value(valid_source).map_err(|e| e.to_string())?;
        let _sources = open_sources(&spec, 4)?;
        let mut meta = Vec::new();
        append_source_provenance(&spec, &mut meta);
        assert_eq!(
            meta,
            vec![
                (
                    "bonsai.composition.candidate_repo",
                    "barozp/Qwen3.8-27B-Opus-Distill"
                ),
                ("bonsai.composition.candidate_revision", CAND_REV),
                ("bonsai.composition.candidate_license", "Apache-2.0"),
                ("bonsai.composition.base_repo", "Qwen/Qwen3.8-27B"),
                ("bonsai.composition.base_revision", BASE_REV),
            ]
        );

        let bad_cases = [
            json!({
                "kind": "pinned-huggingface",
                "candidateRepository": "barozp",
                "candidateRevision": CAND_REV,
                "candidateLicense": "Apache-2.0",
                "baseRepository": "Qwen/Qwen3.8-27B",
                "baseRevision": BASE_REV,
            }),
            json!({
                "kind": "pinned-huggingface",
                "candidateRepository": "../evil",
                "candidateRevision": CAND_REV,
                "candidateLicense": "Apache-2.0",
                "baseRepository": "Qwen/Qwen3.8-27B",
                "baseRevision": BASE_REV,
            }),
            json!({
                "kind": "pinned-huggingface",
                "candidateRepository": "barozp/repo/extra",
                "candidateRevision": CAND_REV,
                "candidateLicense": "Apache-2.0",
                "baseRepository": "Qwen/Qwen3.8-27B",
                "baseRevision": BASE_REV,
            }),
            json!({
                "kind": "pinned-huggingface",
                "candidateRepository": "barozp/Qwen3.8-27B-Opus-Distill",
                "candidateRevision": "0123456789ABCDEF0123456789ABCDEF01234567",
                "candidateLicense": "Apache-2.0",
                "baseRepository": "Qwen/Qwen3.8-27B",
                "baseRevision": BASE_REV,
            }),
            json!({
                "kind": "pinned-huggingface",
                "candidateRepository": "barozp/Qwen3.8-27B-Opus-Distill",
                "candidateRevision": CAND_REV,
                "candidateLicense": "   ",
                "baseRepository": "Qwen/Qwen3.8-27B",
                "baseRevision": BASE_REV,
            }),
            json!({
                "kind": "pinned-huggingface",
                "candidateRepository": "barozp/Qwen3.8-27B-Opus-Distill",
                "candidateRevision": CAND_REV,
                "candidateLicense": "A".repeat(65),
                "baseRepository": "Qwen/Qwen3.8-27B",
                "baseRevision": BASE_REV,
            }),
            json!({
                "kind": "pinned-huggingface",
                "candidateRepository": "barozp/Qwen3.8-27B-Opus-Distill",
                "candidateRevision": CAND_REV,
                "candidateLicense": "Apache-2.0\n",
                "baseRepository": "Qwen/Qwen3.8-27B",
                "baseRevision": BASE_REV,
            }),
            json!({
                "kind": "pinned-huggingface",
                "candidateRepository": "barozp/Qwen3.8-27B-Opus-Distill",
                "candidateRevision": CAND_REV,
                "candidateLicense": "Apache-2.0",
                "baseRepository": "Qwen/.",
                "baseRevision": BASE_REV,
            }),
            json!({
                "kind": "pinned-huggingface",
                "candidateRepository": "barozp/Qwen3.8-27B-Opus-Distill",
                "candidateRevision": CAND_REV,
                "candidateLicense": "Apache-2.0",
                "baseRepository": "Qwen/Qwen3.8-27B",
                "baseRevision": "short",
            }),
            json!({
                "kind": "pinned-huggingface",
                "candidateRepository": "barozp/Qwen3.8-27B-Opus-Distill",
                "candidateRevision": CAND_REV,
                "candidateLicense": "Apache-2.0",
                "baseRepository": "Qwen/Qwen3.8-27B",
                "baseRevision": BASE_REV,
                "extraField": true,
            }),
        ];
        for bad in bad_cases {
            assert!(
                run(json!({
                    "schemaVersion": 1,
                    "operation": "plan",
                    "source": bad,
                }))
                .is_err()
            );
        }

        Ok(())
    }
}
