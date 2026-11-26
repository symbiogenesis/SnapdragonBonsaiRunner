//! Exact readiness for the explicitly selected native representation and adapter.

use serde::{Deserialize, Deserializer};
use std::{borrow::Cow, ops::Deref};

/// Value-free reason why a native child's `/v1/models` readiness failed verification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum NativeIdentityError {
    ReleasePin,
    Framing,
    Schema,
    Model,
    ContextTokens,
    RowWorkers,
    PromptReuse,
    KvCacheGraph,
    Placement,
    StartupProjection,
    Adapter,
    Vision,
    CpuPrefill,
    CpuSpeculation,
}

impl NativeIdentityError {
    pub(super) const fn reason(self) -> &'static str {
        match self {
            Self::ReleasePin => "release-pin",
            Self::Framing => "framing",
            Self::Schema => "schema",
            Self::Model => "model",
            Self::ContextTokens => "context-tokens",
            Self::RowWorkers => "row-workers",
            Self::PromptReuse => "prompt-reuse",
            Self::KvCacheGraph => "kv-cache-graph",
            Self::Placement => "placement",
            Self::StartupProjection => "startup-projection",
            Self::Adapter => "adapter",
            Self::Vision => "vision",
            Self::CpuPrefill => "cpu-prefill",
            Self::CpuSpeculation => "cpu-speculation",
        }
    }
}

fn prism_zero_one_patch(version: &str) -> Option<u32> {
    let patch = version.strip_prefix("0.1.")?;
    if patch.is_empty()
        || !patch.bytes().all(|byte| byte.is_ascii_digit())
        || (patch.len() > 1 && patch.starts_with('0'))
    {
        return None;
    }
    patch.parse::<u32>().ok()
}

/// The selected private projection-worker budget, separate from execution evidence.
#[derive(Clone, Copy)]
pub(super) struct WorkerReadiness<'a> {
    pub(super) expected: u8,
    pub(super) context_tokens: usize,
    pub(super) legacy_missing_allowed: bool,
    pub(super) projection_startup_required: bool,
    pub(super) prompt_reuse: crate::inference::PromptReuseMode,
    pub(super) prompt_reuse_checkpoints: u8,
    pub(super) vision: bool,
    pub(super) kv_cache: Option<crate::inference::KvCache>,
    pub(super) cpu_prefill: Option<(u8, usize)>,
    pub(super) cpu_prefill_gdn_wy_f32: bool,
    pub(super) cpu_speculation: Option<(u8, usize, usize)>,
    pub(super) placement: crate::inference::Placement,
    pub(super) daisyrose: Option<(&'a str, u16)>,
}

impl<'a> WorkerReadiness<'a> {
    pub(super) fn for_version(version: &str, expected: u8) -> Self {
        let patch = prism_zero_one_patch(version);
        Self {
            expected,
            context_tokens: crate::CONTEXT_TOKENS,
            // Only these reviewed historical releases omit this identity. A
            // later pin must report its graph's effective worker budget.
            legacy_missing_allowed: matches!(version, "0.1.8" | "0.1.9" | "0.1.10"),
            // Reviewed 0.1.28+ releases wait for model-thread initialization and
            // always publish their completed placement/residency observation.
            projection_startup_required: patch.is_some_and(|number| number >= 28),
            prompt_reuse: crate::inference::PromptReuseMode::Disabled,
            prompt_reuse_checkpoints: 1,
            vision: false,
            kv_cache: None,
            cpu_prefill: None,
            cpu_prefill_gdn_wy_f32: false,
            cpu_speculation: None,
            placement: crate::inference::Placement::Auto,
            daisyrose: None,
        }
    }

    #[cfg(test)]
    pub(super) const fn allows_missing_row_workers(self) -> bool {
        self.legacy_missing_allowed
    }

    #[cfg(test)]
    pub(super) const fn requires_projection_startup(self) -> bool {
        self.projection_startup_required
    }

    fn accepts(self, observed: Option<u8>) -> bool {
        matches!(self.expected, 1 | 4 | 8 | 12)
            && observed.map_or(self.legacy_missing_allowed, |workers| {
                workers == self.expected
            })
    }
}

#[derive(Deserialize)]
#[serde(transparent)]
struct Text<'a>(#[serde(borrow)] Cow<'a, str>);

impl Deref for Text<'_> {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Models<'a> {
    #[serde(borrow)]
    data: [crate::Object<NativeModel<'a>>; 1],
}

fn nullable<'de, D: Deserializer<'de>, T: Deserialize<'de>>(
    input: D,
) -> Result<Option<T>, D::Error> {
    Option::<T>::deserialize(input)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeModel<'a> {
    #[serde(borrow)]
    id: Text<'a>,
    #[serde(borrow)]
    object: Text<'a>,
    #[serde(borrow)]
    model_sha256: Text<'a>,
    #[serde(borrow)]
    template_sha256: Text<'a>,
    #[serde(borrow)]
    runtime: Text<'a>,
    context_tokens: usize,
    seed: u32,
    #[serde(default, deserialize_with = "crate::deserialize_present")]
    row_workers: Option<u8>,
    #[serde(default, borrow, deserialize_with = "crate::deserialize_present")]
    weight_representation: Option<Text<'a>>,
    #[serde(default, deserialize_with = "crate::deserialize_present")]
    supports_daisyrose_scale: Option<bool>,
    #[serde(default, deserialize_with = "crate::deserialize_present")]
    daisyrose_scale_range: Option<[u8; 2]>,
    #[serde(default, borrow, deserialize_with = "crate::deserialize_present")]
    daisyrose_adapter_sha256: Option<Text<'a>>,
    #[serde(default, deserialize_with = "crate::deserialize_present")]
    daisyrose_adapter_max_rank: Option<u16>,
    #[serde(default, borrow, deserialize_with = "crate::deserialize_present")]
    prompt_reuse: Option<Text<'a>>,
    #[serde(default, deserialize_with = "crate::deserialize_present")]
    prompt_reuse_checkpoints: Option<u8>,
    #[serde(default, borrow, deserialize_with = "crate::deserialize_present")]
    kv_cache: Option<Text<'a>>,
    #[serde(default, borrow, deserialize_with = "crate::deserialize_present")]
    startup_graph_identity: Option<Text<'a>>,
    #[serde(default, borrow, deserialize_with = "crate::deserialize_present")]
    placement: Option<Text<'a>>,
    #[serde(default, borrow, deserialize_with = "crate::deserialize_present")]
    projection_backend: Option<Text<'a>>,
    #[serde(default, borrow, deserialize_with = "crate::deserialize_present")]
    startup_projection: Option<crate::Object<StartupProjection<'a>>>,
    #[serde(default, deserialize_with = "crate::deserialize_present")]
    vision: Option<bool>,
    #[serde(default, borrow, deserialize_with = "crate::deserialize_present")]
    mmproj_sha256: Option<Text<'a>>,
    #[serde(default, deserialize_with = "crate::deserialize_present")]
    image_max_tokens: Option<usize>,
    #[serde(default, deserialize_with = "crate::deserialize_present")]
    cpu_prefill_tokens: Option<u8>,
    #[serde(default, deserialize_with = "crate::deserialize_present")]
    cpu_prefill_scratch_bytes: Option<usize>,
    #[serde(default, borrow, deserialize_with = "crate::deserialize_present")]
    cpu_prefill_gdn: Option<Text<'a>>,
    #[serde(default, deserialize_with = "crate::deserialize_present")]
    cpu_speculative_tokens: Option<u8>,
    #[serde(default, deserialize_with = "crate::deserialize_present")]
    cpu_verification_scratch_bytes: Option<usize>,
    #[serde(default, deserialize_with = "crate::deserialize_present")]
    cpu_speculative_checkpoint_bytes: Option<usize>,
    #[serde(default, deserialize_with = "crate::deserialize_present")]
    supports_cognitive_controls: Option<bool>,
}

/// Reports whether a verified native `/v1/models` payload explicitly advertises
/// support for per-request cognitive, sampling, DRY, budget, and logprob controls.
pub(super) fn native_supports_cognitive_controls(bytes: &[u8]) -> bool {
    let Ok(body) = super::validate_response(bytes, 200) else {
        return false;
    };
    let Ok(crate::Object(models)) = serde_json::from_slice::<crate::Object<Models<'_>>>(body)
    else {
        return false;
    };
    let [crate::Object(model)] = models.data;
    model.supports_cognitive_controls == Some(true)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StartupProjection<'a> {
    schema_version: u8,
    #[serde(borrow)]
    requested: Text<'a>,
    #[serde(borrow)]
    effective: Text<'a>,
    #[serde(borrow, deserialize_with = "nullable")]
    identity: Option<Text<'a>>,
    resident_tensors: u64,
    resident_weight_bytes: u64,
    host_metadata_bytes: u64,
}

fn startup_projection_matches(
    model: &NativeModel<'_>,
    workers: WorkerReadiness<'_>,
    is_ptq1: bool,
    is_nariq: bool,
) -> bool {
    let Some(startup) = &model.startup_projection else {
        // Historical releases, including 0.1.27, predate this observation.
        // The reviewed 0.1.28, 0.1.30 and 0.1.31 tuples must identify completed initialization.
        return !is_nariq && !workers.projection_startup_required;
    };
    let requested = match workers.placement {
        crate::inference::Placement::Auto => "auto",
        crate::inference::Placement::Cpu => "cpu",
        crate::inference::Placement::Vulkan => "vulkan",
        crate::inference::Placement::Composite => "composite",
    };
    if startup.schema_version != 1 || &*startup.requested != requested {
        return false;
    }
    if is_nariq && &*startup.effective != "cpu" {
        return false;
    }
    let no_residency = startup.resident_tensors == 0
        && startup.resident_weight_bytes == 0
        && startup.host_metadata_bytes == 0;
    let valid_graphic_identity = startup.identity.as_deref().is_some_and(|identity| {
        !identity.trim().is_empty()
            && identity.len() <= 256
            && identity.bytes().all(|byte| byte.is_ascii_graphic())
    });
    match &*startup.effective {
        "cpu" => startup.identity.is_none() && no_residency,
        "hexagon-native-q2" => {
            requested == "auto"
                && workers.cpu_speculation.is_none()
                && !is_ptq1
                && startup.identity.as_deref()
                    == Some("hexagon-hvx-native-q2-resident-v1-experimental")
                && startup.resident_tensors == 168
                && startup.resident_weight_bytes == 2_888_663_040
                && (1..=524_288).contains(&startup.host_metadata_bytes)
        }
        "hexagon-native-q2-partial" => {
            requested == "auto"
                && workers.cpu_speculation.is_none()
                && !is_ptq1
                && matches!(
                    startup.identity.as_deref(),
                    Some(
                        "hexagon-v73-hvx-native-q2-partial-v1-experimental"
                            | "hexagon-v81-hvx-native-q2-partial-v1-experimental"
                    )
                )
                && (1..168).contains(&startup.resident_tensors)
                && (1..2_888_663_040).contains(&startup.resident_weight_bytes)
                && (1..=524_288).contains(&startup.host_metadata_bytes)
        }
        "external" => {
            requested == "auto"
                && workers.cpu_prefill.is_none()
                && workers.cpu_speculation.is_none()
                && no_residency
                && valid_graphic_identity
        }
        "adreno-vulkan" => {
            requested == "vulkan"
                && valid_graphic_identity
                && startup.resident_tensors >= 1
                && startup.resident_weight_bytes >= 1
                && startup.host_metadata_bytes == 0
        }
        "composite-hexagon-vulkan" => {
            requested == "composite"
                && valid_graphic_identity
                && startup.resident_tensors >= 2
                && startup.resident_weight_bytes >= 1
                && (1..=524_288).contains(&startup.host_metadata_bytes)
        }
        _ => false,
    }
}

fn cache_identity_matches(
    model: &NativeModel<'_>,
    workers: WorkerReadiness<'_>,
    is_nariq: bool,
) -> bool {
    let Some(policy) = workers.kv_cache else {
        return workers.cpu_prefill.is_none()
            && !workers.cpu_prefill_gdn_wy_f32
            && workers.cpu_speculation.is_none()
            && model.kv_cache.is_none()
            && model.startup_graph_identity.is_none();
    };
    if workers.cpu_prefill.is_none() && policy == crate::inference::KvCache::F16 {
        return !workers.cpu_prefill_gdn_wy_f32
            && workers.cpu_speculation.is_none()
            && model.kv_cache.is_none()
            && model.startup_graph_identity.is_none();
    }
    let expected_kv = (policy != crate::inference::KvCache::F16).then_some(policy.name());
    let cache = if policy == crate::inference::KvCache::F16 {
        "kv-f16".to_owned()
    } else {
        format!("kv-{}-d256-rht3-v1", policy.name())
    };
    let mut prefix = format!(
        "{}:qwen35-text-v1:nonflash-f16:{cache}:gdn-f32:prism-arm64-fused-v1{}",
        &*model.model_sha256,
        if is_nariq { ":trellis-v1" } else { "" }
    );
    let mut suffix = String::new();
    if let Some((tokens, _)) = workers.cpu_prefill {
        use std::fmt::Write as _;
        if write!(suffix, ":cpu-tiled-ffn-v1:{tokens}").is_err() {
            return false;
        }
        if workers.cpu_prefill_gdn_wy_f32 {
            suffix.push_str(":gdn-wy-f32-v1");
        }
        if workers.cpu_speculation.is_some() {
            suffix.push_str(":cpu-tiled-verify-v1");
        }
    } else if workers.cpu_prefill_gdn_wy_f32 || workers.cpu_speculation.is_some() {
        return false;
    }
    if model.kv_cache.as_deref() != expected_kv {
        return false;
    }
    let Some(observed) = model.startup_graph_identity.as_deref() else {
        return false;
    };
    if workers.daisyrose.is_some() {
        let Some(after_prefix) = observed.strip_prefix(&prefix) else {
            return false;
        };
        let Some(after_tag) = after_prefix.strip_prefix(":multi-rank-lora-v1:") else {
            return false;
        };
        let Some((hex_part, after_hex)) = after_tag.split_at_checked(64) else {
            return false;
        };
        if !hex_part
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return false;
        }
        let Some(after_scale) = after_hex.strip_prefix(":00000000") else {
            return false;
        };
        after_scale == suffix.as_str()
    } else {
        prefix.push_str(&suffix);
        observed == prefix.as_str()
    }
}

pub(super) fn validate_native_identity_for_representation(
    bytes: &[u8],
    representation_kind: super::WeightRepresentation,
    workers: WorkerReadiness<'_>,
) -> Result<(), NativeIdentityError> {
    let body = super::validate_response(bytes, 200).map_err(|_| NativeIdentityError::Framing)?;
    let crate::Object(models): crate::Object<Models<'_>> =
        serde_json::from_slice(body).map_err(|_| NativeIdentityError::Schema)?;
    let [crate::Object(model)] = &models.data;
    let is_ptq1 = matches!(
        representation_kind,
        super::WeightRepresentation::Bonsai25(bonsai_cpu_policy::Bonsai25Variant::Ptq1)
    );
    let is_nariq = matches!(representation_kind, super::WeightRepresentation::NariQ(_));
    let (id, hash, representation) = match representation_kind {
        super::WeightRepresentation::Source => return Err(NativeIdentityError::Model),
        super::WeightRepresentation::NariQ(variant) => (
            variant.model_id(),
            variant.sha256(),
            variant.representation_name(),
        ),
        super::WeightRepresentation::Bonsai25(variant) => (
            variant.model_id(),
            variant.sha256(),
            variant.representation_name(),
        ),
    };
    if &*model.id != id
        || &*model.object != "model"
        || &*model.model_sha256 != hash
        || &*model.template_sha256 != crate::inference::TEMPLATE_SHA256
        || &*model.runtime != "prism-inference"
        || model.seed != 42
        || model.weight_representation.as_deref() != Some(representation)
    {
        return Err(NativeIdentityError::Model);
    }
    if model.context_tokens != workers.context_tokens {
        return Err(NativeIdentityError::ContextTokens);
    }
    if !workers.accepts(model.row_workers) {
        return Err(NativeIdentityError::RowWorkers);
    }
    let expected_checkpoints =
        (workers.prompt_reuse_checkpoints > 1).then_some(workers.prompt_reuse_checkpoints);
    if model.prompt_reuse.as_deref() != workers.prompt_reuse.as_wire()
        || model.prompt_reuse_checkpoints != expected_checkpoints
    {
        return Err(NativeIdentityError::PromptReuse);
    }
    let daisyrose_matches = match workers.daisyrose {
        Some((expected_sha, expected_max_rank)) => {
            model.supports_daisyrose_scale == Some(true)
                && model.daisyrose_scale_range == Some([0, 2])
                && model.daisyrose_adapter_sha256.as_deref() == Some(expected_sha)
                && model.daisyrose_adapter_max_rank == Some(expected_max_rank)
        }
        None => {
            model.supports_daisyrose_scale.is_none()
                && model.daisyrose_scale_range.is_none()
                && model.daisyrose_adapter_sha256.is_none()
                && model.daisyrose_adapter_max_rank.is_none()
        }
    };
    if !daisyrose_matches {
        return Err(NativeIdentityError::Adapter);
    }
    if !cache_identity_matches(model, workers, is_nariq) {
        return Err(NativeIdentityError::KvCacheGraph);
    }
    let expected_prefill = workers
        .cpu_prefill
        .map_or((None, None), |(tokens, scratch)| {
            (Some(tokens), Some(scratch))
        });
    let expected_gdn = workers.cpu_prefill_gdn_wy_f32.then_some("wy-f32");
    if (model.cpu_prefill_tokens, model.cpu_prefill_scratch_bytes) != expected_prefill
        || model.cpu_prefill_gdn.as_deref() != expected_gdn
    {
        return Err(NativeIdentityError::CpuPrefill);
    }
    let expected_speculation = workers
        .cpu_speculation
        .map_or((None, None, None), |(tokens, scratch, checkpoint)| {
            (Some(tokens), Some(scratch), Some(checkpoint))
        });
    if (
        model.cpu_speculative_tokens,
        model.cpu_verification_scratch_bytes,
        model.cpu_speculative_checkpoint_bytes,
    ) != expected_speculation
    {
        return Err(NativeIdentityError::CpuSpeculation);
    }
    let (expected_placement, expected_backend) = match workers.placement {
        crate::inference::Placement::Auto => (None, None),
        crate::inference::Placement::Cpu => (Some("cpu"), Some("cpu")),
        crate::inference::Placement::Vulkan => (Some("vulkan"), Some("adreno-vulkan")),
        crate::inference::Placement::Composite => {
            (Some("composite"), Some("composite-hexagon-vulkan"))
        }
    };
    if model.placement.as_deref() != expected_placement
        || model.projection_backend.as_deref() != expected_backend
    {
        return Err(NativeIdentityError::Placement);
    }
    if !startup_projection_matches(model, workers, is_ptq1, is_nariq) {
        return Err(NativeIdentityError::StartupProjection);
    }
    let vision_match = if is_nariq {
        !workers.vision
            && !model.vision.unwrap_or(false)
            && model.mmproj_sha256.is_none()
            && model.image_max_tokens.is_none()
    } else if workers.vision {
        model.vision == Some(true)
            && model.mmproj_sha256.as_deref() == Some(crate::inference::MMPROJ_SHA256)
            && model.image_max_tokens == Some(bonsai_cpu_policy::DEFAULT_IMAGE_MAX_TOKENS)
    } else {
        !model.vision.unwrap_or(false)
            && model.mmproj_sha256.is_none()
            && model.image_max_tokens.is_none()
    };
    if !vision_match {
        return Err(NativeIdentityError::Vision);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{NativeIdentityError, WorkerReadiness};
    use bonsai_cpu_policy::Bonsai25Variant;

    fn validate_native_identity(
        bytes: &[u8],
        ptq1: bool,
        workers: WorkerReadiness<'_>,
    ) -> std::io::Result<()> {
        let variant = if ptq1 {
            Bonsai25Variant::Ptq1
        } else {
            Bonsai25Variant::Pq2
        };
        super::validate_native_identity_for_representation(
            bytes,
            super::super::WeightRepresentation::Bonsai25(variant),
            workers,
        )
        .map_err(|error| std::io::Error::other(error.reason()))
    }

    fn validate_native_identity_detailed(
        bytes: &[u8],
        ptq1: bool,
        workers: WorkerReadiness<'_>,
    ) -> Result<(), NativeIdentityError> {
        let variant = if ptq1 {
            Bonsai25Variant::Ptq1
        } else {
            Bonsai25Variant::Pq2
        };
        super::validate_native_identity_for_representation(
            bytes,
            super::super::WeightRepresentation::Bonsai25(variant),
            workers,
        )
    }

    fn kv_cache_model(ptq1: bool) -> (serde_json::Value, String) {
        kv_cache_model_for_version("0.1.35", ptq1)
    }

    fn kv_cache_model_for_version(_version: &str, ptq1: bool) -> (serde_json::Value, String) {
        use serde_json::json;
        let variant = if ptq1 {
            Bonsai25Variant::Ptq1
        } else {
            Bonsai25Variant::Pq2
        };
        let hash = variant.sha256();
        let graph = format!(
            "{hash}:qwen35-text-v1:nonflash-f16:kv-rsq4-d256-rht3-v1:gdn-f32:prism-arm64-fused-v1"
        );
        let model = json!({
            "id": variant.model_id(),
            "object": "model", "model_sha256": hash,
            "template_sha256": crate::inference::TEMPLATE_SHA256,
            "runtime": "prism-inference", "context_tokens": 8192, "seed": 42,
            "weight_representation": variant.representation_name(),
            "row_workers": 4, "kv_cache": "rsq4", "startup_graph_identity": graph,
        });
        (model, graph)
    }

    fn model_response(model: &serde_json::Value) -> Result<Vec<u8>, serde_json::Error> {
        crate::test_util::http_models_response(model)
    }

    fn projection_startup(effective: &str) -> serde_json::Value {
        serde_json::json!({"schema_version":1,"requested":"auto","effective":effective,
            "identity": if effective == "hexagon-native-q2" {
                Some("hexagon-hvx-native-q2-resident-v1-experimental")
            } else { None },
            "resident_tensors": if effective == "hexagon-native-q2" {168} else {0},
            "resident_weight_bytes": if effective == "hexagon-native-q2" {2_888_663_040_u64} else {0},
            "host_metadata_bytes": if effective == "hexagon-native-q2" {4096} else {0}})
    }

    #[test]
    fn projection_startup_binds_completed_residency_and_requested_policy()
    -> Result<(), serde_json::Error> {
        use crate::inference::{KvCache, Placement};
        let (legacy_model, _) = kv_cache_model_for_version("0.1.27", false);
        let legacy = WorkerReadiness {
            kv_cache: Some(KvCache::Rsq4),
            ..WorkerReadiness::for_version("0.1.27", 4)
        };
        assert!(validate_native_identity(&model_response(&legacy_model)?, false, legacy).is_ok());

        let (mut model, _) = kv_cache_model_for_version("0.1.31", false);
        let auto = WorkerReadiness {
            kv_cache: Some(KvCache::Rsq4),
            ..WorkerReadiness::for_version("0.1.31", 4)
        };
        assert!(validate_native_identity(&model_response(&model)?, false, auto).is_err());
        let mut legacy_with_startup = legacy_model;
        legacy_with_startup["startup_projection"] = projection_startup("cpu");
        assert_eq!(
            validate_native_identity_detailed(&model_response(&legacy_with_startup)?, false, auto),
            Ok(())
        );
        for effective in ["cpu", "hexagon-native-q2"] {
            model["startup_projection"] = projection_startup(effective);
            assert!(validate_native_identity(&model_response(&model)?, false, auto).is_ok());
        }
        let cpu = WorkerReadiness {
            placement: Placement::Cpu,
            ..auto
        };
        model["placement"] = serde_json::json!("cpu");
        model["projection_backend"] = serde_json::json!("cpu");
        assert!(validate_native_identity(&model_response(&model)?, false, cpu).is_err());
        model["startup_projection"] = projection_startup("cpu");
        model["startup_projection"]["requested"] = serde_json::json!("cpu");
        assert!(validate_native_identity(&model_response(&model)?, false, cpu).is_ok());
        let (mut ptq1, _) = kv_cache_model_for_version("0.1.31", true);
        ptq1["startup_projection"] = projection_startup("hexagon-native-q2");
        assert!(validate_native_identity(&model_response(&ptq1)?, true, auto).is_err());
        ptq1["startup_projection"] = projection_startup("cpu");
        assert!(validate_native_identity(&model_response(&ptq1)?, true, auto).is_ok());
        Ok(())
    }

    #[test]
    fn partial_hexagon_projection_startup_accepts_v73_and_v81_bounds()
    -> Result<(), serde_json::Error> {
        use crate::inference::{KvCache, Placement};
        use serde_json::json;

        let (mut model, _) = kv_cache_model_for_version("0.1.31", false);
        let auto = WorkerReadiness {
            kv_cache: Some(KvCache::Rsq4),
            ..WorkerReadiness::for_version("0.1.31", 4)
        };
        for identity in [
            "hexagon-v73-hvx-native-q2-partial-v1-experimental",
            "hexagon-v81-hvx-native-q2-partial-v1-experimental",
        ] {
            model["startup_projection"] = json!({
                "schema_version": 1,
                "requested": "auto",
                "effective": "hexagon-native-q2-partial",
                "identity": identity,
                "resident_tensors": 144,
                "resident_weight_bytes": 1_800_000_000_u64,
                "host_metadata_bytes": 4096,
            });
            assert!(
                validate_native_identity(&model_response(&model)?, false, auto).is_ok(),
                "{identity}"
            );
        }

        let cpu = WorkerReadiness {
            placement: Placement::Cpu,
            ..auto
        };
        let mut cpu_model = model.clone();
        cpu_model["placement"] = json!("cpu");
        cpu_model["projection_backend"] = json!("cpu");
        cpu_model["startup_projection"]["requested"] = json!("cpu");
        assert!(validate_native_identity(&model_response(&cpu_model)?, false, cpu).is_err());

        let (mut ptq1, _) = kv_cache_model_for_version("0.1.31", true);
        ptq1["startup_projection"] = model["startup_projection"].clone();
        assert!(validate_native_identity(&model_response(&ptq1)?, true, auto).is_err());

        for (field, value) in [
            ("identity", json!(null)),
            (
                "identity",
                json!("hexagon-v73-hvx-native-q2-v1-experimental"),
            ),
            (
                "identity",
                json!("hexagon-hvx-native-q2-resident-v1-experimental"),
            ),
            ("resident_tensors", json!(0)),
            ("resident_tensors", json!(168)),
            ("resident_weight_bytes", json!(0)),
            ("resident_weight_bytes", json!(2_888_663_040_u64)),
            ("host_metadata_bytes", json!(0)),
            ("host_metadata_bytes", json!(524_289)),
        ] {
            let mut candidate = model.clone();
            candidate["startup_projection"][field] = value;
            assert!(
                validate_native_identity(&model_response(&candidate)?, false, auto).is_err(),
                "{field}"
            );
        }
        Ok(())
    }

    #[test]
    fn projection_startup_rejects_inconsistent_unknown_and_ambiguous_facts()
    -> Result<(), serde_json::Error> {
        use crate::inference::KvCache;
        use serde_json::json;
        let (model, _) = kv_cache_model(false);
        let auto = WorkerReadiness {
            kv_cache: Some(KvCache::Rsq4),
            ..WorkerReadiness::for_version("0.1.31", 4)
        };
        for (field, value) in [
            ("schema_version", json!(2)),
            ("schema_version", json!(true)),
            ("requested", json!("cpu")),
            ("effective", json!("hexagon")),
            ("identity", json!(null)),
            ("identity", json!("changed")),
            ("resident_tensors", json!(167)),
            ("resident_tensors", json!(169)),
            ("resident_weight_bytes", json!(2_888_663_041_u64)),
            ("host_metadata_bytes", json!(0)),
            ("host_metadata_bytes", json!(524_289)),
            ("unreviewed_field", json!(true)),
        ] {
            let mut candidate = model.clone();
            candidate["startup_projection"] = projection_startup("hexagon-native-q2");
            candidate["startup_projection"][field] = value;
            assert!(
                validate_native_identity(&model_response(&candidate)?, false, auto).is_err(),
                "{field}"
            );
        }
        for invalid in [
            json!(null),
            json!({}),
            json!("cpu"),
            json!([]),
            json!([1, "auto", "cpu", null, 0, 0, 0]),
        ] {
            let mut candidate = model.clone();
            candidate["startup_projection"] = invalid;
            assert!(validate_native_identity(&model_response(&candidate)?, false, auto).is_err());
        }
        for field in [
            "schema_version",
            "requested",
            "effective",
            "identity",
            "resident_tensors",
            "resident_weight_bytes",
            "host_metadata_bytes",
        ] {
            let mut candidate = model.clone();
            candidate["startup_projection"] = projection_startup("cpu");
            if let Some(object) = candidate["startup_projection"].as_object_mut() {
                object.remove(field);
            }
            assert!(
                validate_native_identity(&model_response(&candidate)?, false, auto).is_err(),
                "missing {field}"
            );
        }
        let mut candidate = model;
        candidate["startup_projection"] = projection_startup("cpu");
        candidate["startup_projection"]["resident_weight_bytes"] = json!(1);
        assert!(validate_native_identity(&model_response(&candidate)?, false, auto).is_err());
        // Check duplicate names before a Value map could erase the ambiguity.
        candidate["startup_projection"] = projection_startup("cpu");
        let body = json!({"data":[candidate]}).to_string().replace(
            "\"schema_version\":1",
            "\"schema_version\":1,\"schema_version\":1",
        );
        let bytes = format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        )
        .into_bytes();
        assert!(validate_native_identity(&bytes, false, auto).is_err());
        Ok(())
    }

    #[test]
    fn projection_startup_external_identity_never_asserts_residency()
    -> Result<(), serde_json::Error> {
        use crate::inference::KvCache;
        use serde_json::json;
        let (mut model, _) = kv_cache_model(false);
        let auto = WorkerReadiness {
            kv_cache: Some(KvCache::Rsq4),
            ..WorkerReadiness::for_version("0.1.31", 4)
        };
        model["startup_projection"] = projection_startup("external");
        for identity in ["", " ", "bad\nidentity", &"x".repeat(257)] {
            model["startup_projection"]["identity"] = json!(identity);
            assert!(validate_native_identity(&model_response(&model)?, false, auto).is_err());
        }
        model["startup_projection"]["identity"] = json!("reviewed-external-executor-v1");
        assert!(validate_native_identity(&model_response(&model)?, false, auto).is_ok());
        model["startup_projection"]["resident_tensors"] = json!(168);
        assert!(validate_native_identity(&model_response(&model)?, false, auto).is_err());
        Ok(())
    }

    #[test]
    fn cpu_placement_readiness_requires_both_requested_and_effective_cpu()
    -> Result<(), serde_json::Error> {
        use crate::inference::{KvCache, Placement};
        use serde_json::json;

        for ptq1 in [false, true] {
            let (mut model, _) = kv_cache_model(ptq1);
            let auto = WorkerReadiness {
                kv_cache: Some(KvCache::Rsq4),
                ..WorkerReadiness::for_version("0.1.27", 4)
            };
            let cpu = WorkerReadiness {
                placement: Placement::Cpu,
                ..auto
            };
            assert!(validate_native_identity(&model_response(&model)?, ptq1, auto).is_ok());
            assert!(validate_native_identity(&model_response(&model)?, ptq1, cpu).is_err());
            model["placement"] = json!("cpu");
            model["projection_backend"] = json!("cpu");
            let bytes = model_response(&model)?;
            assert!(validate_native_identity(&bytes, ptq1, cpu).is_ok());
            assert!(validate_native_identity(&bytes, ptq1, auto).is_err());
            for field in ["placement", "projection_backend"] {
                let mut missing = model.clone();
                if let Some(object) = missing.as_object_mut() {
                    object.remove(field);
                }
                assert!(validate_native_identity(&model_response(&missing)?, ptq1, cpu).is_err());
                for value in [
                    json!("auto"),
                    json!("CPU"),
                    json!("hexagon"),
                    json!(null),
                    json!(true),
                    json!(1),
                    json!([]),
                ] {
                    let mut changed = model.clone();
                    changed[field] = value;
                    assert!(
                        validate_native_identity(&model_response(&changed)?, ptq1, cpu).is_err()
                    );
                    assert!(
                        validate_native_identity(&model_response(&changed)?, ptq1, auto).is_err()
                    );
                }
            }
        }
        let cpu = WorkerReadiness {
            placement: Placement::Cpu,
            ..WorkerReadiness::for_version("0.1.27", 4)
        };
        for extra in [
            "\"placement\":\"cpu\",\"projection_backend\":\"cpu\",\"placement\":\"cpu\"",
            "\"placement\":\"cpu\",\"projection_backend\":\"cpu\",\"projection_backend\":\"cpu\"",
        ] {
            assert!(
                validate_native_identity(&response(Some(&format!("4,{extra}"))), false, cpu)
                    .is_err()
            );
        }
        Ok(())
    }

    #[test]
    fn kv_cache_readiness_binds_profile_and_graph_identity() -> Result<(), serde_json::Error> {
        use crate::inference::KvCache;
        use serde_json::json;
        for policy in [KvCache::Rsq4, KvCache::Rsq4Kvf, KvCache::Rsq4K3v] {
            for ptq1 in [false, true] {
                let (mut model, graph) = kv_cache_model(ptq1);
                let graph = graph.replace("kv-rsq4-", &format!("kv-{}-", policy.name()));
                model["kv_cache"] = json!(policy.name());
                model["startup_graph_identity"] = json!(graph);
                let f16 = WorkerReadiness::for_version("0.1.27", 4);
                let selected = WorkerReadiness {
                    kv_cache: Some(policy),
                    ..f16
                };
                let bytes = model_response(&model)?;
                assert!(validate_native_identity(&bytes, ptq1, selected).is_ok());
                assert!(validate_native_identity(&bytes, ptq1, f16).is_err());
                let other = if policy == KvCache::Rsq4 {
                    KvCache::Rsq4K3v
                } else {
                    KvCache::Rsq4
                };
                for (field, replacement) in [
                    ("kv_cache", json!("f16")),
                    ("kv_cache", json!(other.name())),
                    ("kv_cache", json!(null)),
                    ("startup_graph_identity", json!("wrong-model-or-codec")),
                    (
                        "startup_graph_identity",
                        json!(format!("{graph}:unexpected")),
                    ),
                    (
                        "startup_graph_identity",
                        json!(graph.replace(policy.name(), other.name())),
                    ),
                    ("startup_graph_identity", json!(null)),
                ] {
                    let mut changed = model.clone();
                    changed[field] = replacement;
                    assert!(
                        validate_native_identity(&model_response(&changed)?, ptq1, selected)
                            .is_err()
                    );
                }
                for field in ["kv_cache", "startup_graph_identity"] {
                    let mut changed = model.clone();
                    if let Some(object) = changed.as_object_mut() {
                        object.remove(field);
                    }
                    assert!(
                        validate_native_identity(&model_response(&changed)?, ptq1, selected)
                            .is_err()
                    );
                }
                if let Some(object) = model.as_object_mut() {
                    object.remove("kv_cache");
                    object.remove("startup_graph_identity");
                }
                let bytes = model_response(&model)?;
                assert!(validate_native_identity(&bytes, ptq1, f16).is_ok());
                assert!(validate_native_identity(&bytes, ptq1, selected).is_err());
            }
        }
        Ok(())
    }

    #[test]
    fn cpu_prefill_readiness_binds_tokens_scratch_projection_and_graph_identity()
    -> Result<(), serde_json::Error> {
        use crate::inference::KvCache;
        use serde_json::json;

        let scratch = 11_501_568_usize;
        for kv in [KvCache::F16, KvCache::Rsq4] {
            let (mut prefill_model, _) = kv_cache_model_for_version("0.1.31", false);
            prefill_model["startup_projection"] = projection_startup("cpu");
            prefill_model["cpu_prefill_tokens"] = json!(8);
            prefill_model["cpu_prefill_scratch_bytes"] = json!(scratch);
            let base_graph = prefill_model["startup_graph_identity"]
                .as_str()
                .map_or_else(String::new, ToOwned::to_owned);
            let updated_graph = if kv == KvCache::F16 {
                prefill_model.as_object_mut().map(|o| o.remove("kv_cache"));
                format!(
                    "{}:cpu-tiled-ffn-v1:8",
                    base_graph.replace("kv-rsq4-d256-rht3-v1", "kv-f16")
                )
            } else {
                format!("{base_graph}:cpu-tiled-ffn-v1:8")
            };
            prefill_model["startup_graph_identity"] = json!(updated_graph);
            let policy = WorkerReadiness {
                kv_cache: Some(kv),
                cpu_prefill: Some((8, scratch)),
                ..WorkerReadiness::for_version("0.1.31", 4)
            };
            assert_eq!(
                validate_native_identity_detailed(&model_response(&prefill_model)?, false, policy),
                Ok(())
            );

            let mut wrong_scratch = prefill_model.clone();
            wrong_scratch["cpu_prefill_scratch_bytes"] = json!(scratch + 1);
            assert_eq!(
                validate_native_identity_detailed(&model_response(&wrong_scratch)?, false, policy),
                Err(NativeIdentityError::CpuPrefill)
            );

            // 0.1.32+ preserves mixed-phase Hexagon decode alongside CPU tiled prefill
            // whenever CPU speculation is disabled, while "external" stays excluded.
            let mut hex_active = prefill_model.clone();
            hex_active["startup_projection"] = projection_startup("hexagon-native-q2");
            assert_eq!(
                validate_native_identity_detailed(&model_response(&hex_active)?, false, policy),
                Ok(())
            );

            let mut external_active = prefill_model;
            external_active["startup_projection"] = projection_startup("external");
            assert_eq!(
                validate_native_identity_detailed(
                    &model_response(&external_active)?,
                    false,
                    policy,
                ),
                Err(NativeIdentityError::StartupProjection)
            );
        }
        Ok(())
    }

    #[test]
    fn branch_reuse_gdn_wy_f32_and_speculation_readiness_bind_all_0_1_32_fields()
    -> Result<(), serde_json::Error> {
        use crate::inference::{KvCache, PromptReuseMode};
        use serde_json::json;

        let prefill_scratch = 12_000_000_usize;
        let verify_scratch = 7_946_432_usize;
        let checkpoint_bytes = 1_073_741_824_usize;

        // Prompt-end-and-latest reuse with multi-branch checkpoints and chunkwise wy-f32 GDN prefill.
        let (mut gdn_model, _) = kv_cache_model_for_version("0.1.35", false);
        gdn_model.as_object_mut().map(|o| o.remove("kv_cache"));
        let base_f16 = gdn_model["startup_graph_identity"]
            .as_str()
            .map_or_else(String::new, ToOwned::to_owned)
            .replace("kv-rsq4-d256-rht3-v1", "kv-f16");
        gdn_model["startup_projection"] = projection_startup("hexagon-native-q2");
        gdn_model["prompt_reuse"] = json!("prompt-end-and-latest");
        gdn_model["prompt_reuse_checkpoints"] = json!(4);
        gdn_model["cpu_prefill_tokens"] = json!(16);
        gdn_model["cpu_prefill_scratch_bytes"] = json!(prefill_scratch);
        gdn_model["cpu_prefill_gdn"] = json!("wy-f32");
        gdn_model["startup_graph_identity"] =
            json!(format!("{base_f16}:cpu-tiled-ffn-v1:16:gdn-wy-f32-v1"));

        let gdn_policy = WorkerReadiness {
            kv_cache: Some(KvCache::F16),
            prompt_reuse: PromptReuseMode::PromptEndAndLatest,
            prompt_reuse_checkpoints: 4,
            cpu_prefill: Some((16, prefill_scratch)),
            cpu_prefill_gdn_wy_f32: true,
            ..WorkerReadiness::for_version("0.1.35", 8)
        };
        let mut gdn_eight = gdn_model.clone();
        gdn_eight["row_workers"] = json!(8);
        assert_eq!(
            validate_native_identity_detailed(&model_response(&gdn_eight)?, false, gdn_policy),
            Ok(())
        );

        let mut wrong_gdn = gdn_eight.clone();
        wrong_gdn
            .as_object_mut()
            .map(|o| o.remove("cpu_prefill_gdn"));
        assert_eq!(
            validate_native_identity_detailed(&model_response(&wrong_gdn)?, false, gdn_policy),
            Err(NativeIdentityError::CpuPrefill)
        );

        let mut wrong_ckpts = gdn_eight;
        wrong_ckpts["prompt_reuse_checkpoints"] = json!(2);
        assert_eq!(
            validate_native_identity_detailed(&model_response(&wrong_ckpts)?, false, gdn_policy),
            Err(NativeIdentityError::PromptReuse)
        );

        // Speculative decoding with K4/V4 cache requires CPU projection and :cpu-tiled-verify-v1.
        let (mut spec_model, base_rsq4) = kv_cache_model_for_version("0.1.35", false);
        spec_model["row_workers"] = json!(8);
        spec_model["startup_projection"] = projection_startup("cpu");
        spec_model["cpu_prefill_tokens"] = json!(16);
        spec_model["cpu_prefill_scratch_bytes"] = json!(prefill_scratch);
        spec_model["cpu_speculative_tokens"] = json!(4);
        spec_model["cpu_verification_scratch_bytes"] = json!(verify_scratch);
        spec_model["cpu_speculative_checkpoint_bytes"] = json!(checkpoint_bytes);
        spec_model["startup_graph_identity"] = json!(format!(
            "{base_rsq4}:cpu-tiled-ffn-v1:16:cpu-tiled-verify-v1"
        ));

        let spec_policy = WorkerReadiness {
            kv_cache: Some(KvCache::Rsq4),
            cpu_prefill: Some((16, prefill_scratch)),
            cpu_speculation: Some((4, verify_scratch, checkpoint_bytes)),
            ..WorkerReadiness::for_version("0.1.35", 8)
        };
        assert_eq!(
            validate_native_identity_detailed(&model_response(&spec_model)?, false, spec_policy),
            Ok(())
        );

        let mut wrong_spec = spec_model.clone();
        wrong_spec["cpu_speculative_tokens"] = json!(5);
        assert_eq!(
            validate_native_identity_detailed(&model_response(&wrong_spec)?, false, spec_policy),
            Err(NativeIdentityError::CpuSpeculation)
        );

        let mut hex_spec = spec_model;
        hex_spec["startup_projection"] = projection_startup("hexagon-native-q2");
        assert_eq!(
            validate_native_identity_detailed(&model_response(&hex_spec)?, false, spec_policy),
            Err(NativeIdentityError::StartupProjection)
        );

        Ok(())
    }

    fn response(workers: Option<&str>) -> Vec<u8> {
        let variant = Bonsai25Variant::Pq2;
        let extension =
            workers.map_or_else(String::new, |workers| format!(",\"row_workers\":{workers}"));
        let body = format!(
            "{{\"data\":[{{\"id\":\"{}\",\"object\":\"model\",\"model_sha256\":\"{}\",\"template_sha256\":\"{}\",\"runtime\":\"prism-inference\",\"weight_representation\":\"{}\",\"context_tokens\":8192,\"seed\":42{extension}}}]}}",
            variant.model_id(),
            variant.sha256(),
            crate::inference::TEMPLATE_SHA256,
            variant.representation_name(),
        );
        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        )
        .into_bytes()
    }

    #[test]
    fn worker_readiness_missing_field_is_limited_to_exact_legacy_pins() {
        for (version, legacy) in [
            ("0.1.8", true),
            ("0.1.9", true),
            ("0.1.10", true),
            ("0.1.7", false),
            ("0.1.11", false),
            ("0.1.27", false),
            ("0.1.10-dev", false),
            ("v0.1.10", false),
            ("0.1.10.0", false),
            ("0.1.010", false),
            ("1.0.0", false),
            ("", false),
        ] {
            for expected in [1, 4] {
                let policy = WorkerReadiness::for_version(version, expected);
                assert_eq!(
                    validate_native_identity(&response(None), false, policy).is_ok(),
                    legacy,
                    "{version}/{expected}"
                );
                assert!(
                    validate_native_identity(
                        &response(Some(&expected.to_string())),
                        false,
                        policy,
                    )
                    .is_ok()
                );
                let other = if expected == 1 { "4" } else { "1" };
                assert!(validate_native_identity(&response(Some(other)), false, policy).is_err());
            }
        }
    }

    #[test]
    fn prompt_reuse_readiness_must_match_the_explicit_request_exactly() {
        let reset = WorkerReadiness::for_version("0.1.27", 4);
        let latest = WorkerReadiness {
            prompt_reuse: crate::inference::PromptReuseMode::Latest,
            ..reset
        };
        assert!(validate_native_identity(&response(Some("4")), false, reset).is_ok());
        assert!(validate_native_identity(&response(Some("4")), false, latest).is_err());
        for extension in [
            "\"latest\"",
            "\"disabled\"",
            "\"Latest\"",
            "null",
            "true",
            "0",
            "{}",
            "[]",
            "\"latest\",\"prompt_reuse\":\"latest\"",
        ] {
            let bytes = response(Some(&format!("4,\"prompt_reuse\":{extension}")));
            assert!(validate_native_identity(&bytes, false, reset).is_err());
            assert_eq!(
                validate_native_identity(&bytes, false, latest).is_ok(),
                extension == "\"latest\"",
                "{extension}"
            );
        }
    }

    #[test]
    fn current_release_requires_workers_for_every_representation_profile()
    -> Result<(), serde_json::Error> {
        use serde_json::json;

        for version in ["0.1.11", "0.1.27"] {
            for ptq1 in [false, true] {
                let variant = if ptq1 {
                    Bonsai25Variant::Ptq1
                } else {
                    Bonsai25Variant::Pq2
                };
                for expected in [1, 4] {
                    let model = json!({
                        "id": variant.model_id(),
                        "object": "model",
                        "model_sha256": variant.sha256(),
                        "template_sha256": crate::inference::TEMPLATE_SHA256,
                        "runtime": "prism-inference", "context_tokens": 8192, "seed": 42,
                        "weight_representation": variant.representation_name(),
                    });
                    for observed in [None, Some(1), Some(4)] {
                        let mut candidate = model.clone();
                        if let Some(workers) = observed {
                            candidate["row_workers"] = json!(workers);
                        }
                        let body = serde_json::to_string(&json!({"data": [candidate]}))?;
                        let framed = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
                            body.len()
                        );
                        assert_eq!(
                            validate_native_identity(
                                framed.as_bytes(),
                                ptq1,
                                WorkerReadiness::for_version(version, expected)
                            )
                            .is_ok(),
                            observed == Some(expected),
                            "version={version}, ptq1={ptq1}, expected={expected}, observed={observed:?}"
                        );
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn worker_readiness_rejects_coercion_unbounded_counts_and_duplicate_fields() {
        for version in ["0.1.10", "0.1.11", "0.1.27"] {
            let policy = WorkerReadiness::for_version(version, 4);
            for invalid in [
                "null",
                "\"4\"",
                "0",
                "2",
                "16",
                "17",
                "-1",
                "256",
                "18446744073709551616",
                "4.0",
                "true",
                "[]",
                "{}",
                "4,\"row_workers\":4",
                "4,\"row_workers\":null",
            ] {
                assert!(
                    validate_native_identity(&response(Some(invalid)), false, policy).is_err(),
                    "{version}/{invalid}"
                );
            }
        }
        for expected in [0, 2, 16, 255] {
            let policy = WorkerReadiness::for_version("0.1.10", expected);
            assert!(validate_native_identity(&response(None), false, policy).is_err());
            assert!(
                validate_native_identity(&response(Some(&expected.to_string())), false, policy)
                    .is_err()
            );
        }
        for expected in [8, 12] {
            let policy = WorkerReadiness::for_version("0.1.31", expected);
            assert!(validate_native_identity(&response(None), false, policy).is_err());
            assert!(
                validate_native_identity(&response(Some(&expected.to_string())), false, policy)
                    .is_err()
            );
        }
    }

    #[test]
    fn parallel_row_worker_readiness_accepts_eight_and_twelve() -> Result<(), serde_json::Error> {
        use crate::inference::KvCache;
        use serde_json::json;

        for expected in [4_u8, 8, 12] {
            let (mut model, _) = kv_cache_model(false);
            model["row_workers"] = json!(expected);
            model["startup_projection"] = projection_startup("cpu");
            let policy = WorkerReadiness {
                kv_cache: Some(KvCache::Rsq4),
                ..WorkerReadiness::for_version("0.1.31", expected)
            };
            assert!(validate_native_identity(&model_response(&model)?, false, policy).is_ok());
            for other in [1_u8, 4, 8, 12] {
                if other != expected {
                    model["row_workers"] = json!(other);
                    assert!(
                        validate_native_identity(&model_response(&model)?, false, policy).is_err()
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn vulkan_and_composite_placements_verify_matching_startup_projection_and_reject_drift()
    -> Result<(), String> {
        use crate::inference::{KvCache, Placement};
        use serde_json::json;

        let (mut base, _) = kv_cache_model_for_version("0.1.35", false);
        base["row_workers"] = json!(8);

        let mut vulkan_model = base.clone();
        vulkan_model["placement"] = json!("vulkan");
        vulkan_model["projection_backend"] = json!("adreno-vulkan");
        vulkan_model["startup_projection"] = json!({
            "schema_version": 1,
            "requested": "vulkan",
            "effective": "adreno-vulkan",
            "identity": "adreno-vulkan-pq2-v1",
            "resident_tensors": 256,
            "resident_weight_bytes": 4_000_000_000_u64,
            "host_metadata_bytes": 0,
        });
        let vulkan_policy = WorkerReadiness {
            kv_cache: Some(KvCache::Rsq4),
            placement: Placement::Vulkan,
            ..WorkerReadiness::for_version("0.1.35", 8)
        };
        let vulkan_wire = model_response(&vulkan_model).map_err(|e| e.to_string())?;
        assert_eq!(
            validate_native_identity_detailed(&vulkan_wire, false, vulkan_policy),
            Ok(())
        );
        let mut bad_vulkan_backend = vulkan_model.clone();
        bad_vulkan_backend["projection_backend"] = json!("cpu");
        assert_eq!(
            validate_native_identity_detailed(
                &model_response(&bad_vulkan_backend).map_err(|e| e.to_string())?,
                false,
                vulkan_policy,
            ),
            Err(NativeIdentityError::Placement)
        );
        let mut bad_vulkan_meta = vulkan_model;
        bad_vulkan_meta["startup_projection"]["host_metadata_bytes"] = json!(1);
        assert_eq!(
            validate_native_identity_detailed(
                &model_response(&bad_vulkan_meta).map_err(|e| e.to_string())?,
                false,
                vulkan_policy,
            ),
            Err(NativeIdentityError::StartupProjection)
        );

        let mut composite_model = base;
        composite_model["placement"] = json!("composite");
        composite_model["projection_backend"] = json!("composite-hexagon-vulkan");
        composite_model["startup_projection"] = json!({
            "schema_version": 1,
            "requested": "composite",
            "effective": "composite-hexagon-vulkan",
            "identity": "composite-hexagon-v81-adreno-v1",
            "resident_tensors": 400,
            "resident_weight_bytes": 6_000_000_000_u64,
            "host_metadata_bytes": 4096,
        });
        let composite_policy = WorkerReadiness {
            kv_cache: Some(KvCache::Rsq4),
            placement: Placement::Composite,
            ..WorkerReadiness::for_version("0.1.35", 8)
        };
        let composite_wire = model_response(&composite_model).map_err(|e| e.to_string())?;
        assert_eq!(
            validate_native_identity_detailed(&composite_wire, false, composite_policy),
            Ok(())
        );
        let mut bad_composite_tensors = composite_model.clone();
        bad_composite_tensors["startup_projection"]["resident_tensors"] = json!(1);
        assert_eq!(
            validate_native_identity_detailed(
                &model_response(&bad_composite_tensors).map_err(|e| e.to_string())?,
                false,
                composite_policy,
            ),
            Err(NativeIdentityError::StartupProjection)
        );
        let mut bad_composite_meta = composite_model;
        bad_composite_meta["startup_projection"]["host_metadata_bytes"] = json!(0);
        assert_eq!(
            validate_native_identity_detailed(
                &model_response(&bad_composite_meta).map_err(|e| e.to_string())?,
                false,
                composite_policy,
            ),
            Err(NativeIdentityError::StartupProjection)
        );
        Ok(())
    }
}

#[cfg(test)]
mod context_tests {
    use super::WorkerReadiness;
    use bonsai_cpu_policy::Bonsai25Variant;

    fn validate_native_identity(
        bytes: &[u8],
        ptq1: bool,
        workers: WorkerReadiness<'_>,
    ) -> std::io::Result<()> {
        let variant = if ptq1 {
            Bonsai25Variant::Ptq1
        } else {
            Bonsai25Variant::Pq2
        };
        super::validate_native_identity_for_representation(
            bytes,
            super::super::WeightRepresentation::Bonsai25(variant),
            workers,
        )
        .map_err(|error| std::io::Error::other(error.reason()))
    }

    #[test]
    fn readiness_rejects_any_context_other_than_the_selected_graph() {
        let variant = Bonsai25Variant::Pq2;
        let body = serde_json::json!({"data":[{
            "id":variant.model_id(), "object":"model",
            "model_sha256":variant.sha256(),
            "template_sha256":crate::inference::TEMPLATE_SHA256,
            "runtime":"prism-inference", "weight_representation":variant.representation_name(),
            "context_tokens":16384, "seed":42, "row_workers":1
        }]})
        .to_string();
        let wire = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        );
        let selected = WorkerReadiness {
            context_tokens: 16384,
            ..WorkerReadiness::for_version("0.1.27", 1)
        };
        assert!(validate_native_identity(wire.as_bytes(), false, selected).is_ok());
        let legacy = WorkerReadiness::for_version("0.1.27", 1);
        assert!(validate_native_identity(wire.as_bytes(), false, legacy).is_err());
    }

    #[test]
    fn readiness_enforces_vision_projector_and_image_token_identity() -> Result<(), String> {
        let mmproj = bonsai_cpu_policy::swift_mmproj_artifact(&std::env::temp_dir())?;
        assert_eq!(
            crate::inference::MMPROJ_SHA256,
            "6807ede61d570bb86ba34b756a0fa109edc33668604de867c6ea6d8f1d631903"
        );
        assert_eq!(crate::inference::MMPROJ_SHA256, mmproj.sha256);

        let variant = Bonsai25Variant::Pq2;
        let base_model = serde_json::json!({
            "id": variant.model_id(), "object": "model",
            "model_sha256": variant.sha256(),
            "template_sha256": crate::inference::TEMPLATE_SHA256,
            "runtime": "prism-inference",
            "weight_representation": variant.representation_name(),
            "context_tokens": 8192, "seed": 42, "row_workers": 1,
        });
        let mut with_vision = base_model.clone();
        with_vision["vision"] = serde_json::json!(true);
        with_vision["mmproj_sha256"] = serde_json::json!(crate::inference::MMPROJ_SHA256);
        with_vision["image_max_tokens"] = serde_json::json!(1024);
        let vision_wire = crate::test_util::http_models_response(&with_vision)
            .map_err(|err| format!("{err:?}"))?;
        let expect_vision = WorkerReadiness {
            vision: true,
            ..WorkerReadiness::for_version("0.1.27", 1)
        };
        let expect_text = WorkerReadiness::for_version("0.1.27", 1);
        assert!(validate_native_identity(&vision_wire, false, expect_vision).is_ok());
        assert!(validate_native_identity(&vision_wire, false, expect_text).is_err());

        let mut mismatched_vision = with_vision;
        mismatched_vision["mmproj_sha256"] =
            serde_json::json!("28e6426fe933262e362bfbe3054f973d9deac116f49e7e342d1a7748b44c1904");
        let mismatched_wire = crate::test_util::http_models_response(&mismatched_vision)
            .map_err(|err| format!("{err:?}"))?;
        assert!(validate_native_identity(&mismatched_wire, false, expect_vision).is_err());

        let text_wire = crate::test_util::http_models_response(&base_model)
            .map_err(|err| format!("{err:?}"))?;
        assert!(validate_native_identity(&text_wire, false, expect_text).is_ok());
        assert!(validate_native_identity(&text_wire, false, expect_vision).is_err());
        Ok(())
    }

    #[test]
    fn pinned_release_version_enforces_all_current_readiness_and_start_chat_combinations()
    -> Result<(), String> {
        use super::NativeIdentityError;
        use crate::inference::{KvCache, Placement};
        use serde_json::json;

        for (version, legacy_workers, proj_required) in [
            ("0.1.8", true, false),
            ("0.1.10", true, false),
            ("0.1.11", false, false),
            ("0.1.27", false, false),
            ("0.1.28", false, true),
            ("0.1.29", false, true),
            ("0.1.30", false, true),
            ("0.1.31", false, true),
            ("0.1.32", false, true),
            ("0.1.34", false, true),
            ("0.1.35", false, true),
            ("0.1.38", false, true),
            ("1.0.0", false, false),
            ("0.1.031", false, false),
            ("0.1.31-dev", false, false),
        ] {
            let w = WorkerReadiness::for_version(version, 4);
            assert_eq!(w.allows_missing_row_workers(), legacy_workers, "{version}");
            assert_eq!(w.requires_projection_startup(), proj_required, "{version}");
        }

        let pinned = crate::inference::pinned_version().map_err(|err| format!("{err:?}"))?;
        let base_policy = WorkerReadiness::for_version(&pinned, 4);
        assert!(!base_policy.allows_missing_row_workers());
        assert!(base_policy.requires_projection_startup());

        let build_payload = |variant: Bonsai25Variant,
                             vision: bool,
                             kv_cache: KvCache,
                             placement: Placement,
                             row_workers: u8,
                             prompt_reuse: bool,
                             context_tokens: usize,
                             effective: &str| {
            let hash = variant.sha256();
            let requested = if placement == Placement::Cpu {
                "cpu"
            } else {
                "auto"
            };
            let startup_projection = match effective {
                "hexagon-native-q2" => json!({
                    "schema_version": 1,
                    "requested": requested,
                    "effective": "hexagon-native-q2",
                    "identity": "hexagon-hvx-native-q2-resident-v1-experimental",
                    "resident_tensors": 168,
                    "resident_weight_bytes": 2_888_663_040_u64,
                    "host_metadata_bytes": 4096,
                }),
                "hexagon-native-q2-partial" => json!({
                    "schema_version": 1,
                    "requested": requested,
                    "effective": "hexagon-native-q2-partial",
                    "identity": "hexagon-v73-hvx-native-q2-partial-v1-experimental",
                    "resident_tensors": 144,
                    "resident_weight_bytes": 1_800_000_000_u64,
                    "host_metadata_bytes": 4096,
                }),
                _ => json!({
                    "schema_version": 1,
                    "requested": requested,
                    "effective": "cpu",
                    "identity": null,
                    "resident_tensors": 0,
                    "resident_weight_bytes": 0,
                    "host_metadata_bytes": 0,
                }),
            };
            let mut model = json!({
                "id": variant.model_id(),
                "object": "model",
                "model_sha256": hash,
                "template_sha256": crate::inference::TEMPLATE_SHA256,
                "runtime": "prism-inference",
                "weight_representation": variant.representation_name(),
                "context_tokens": context_tokens,
                "seed": 42,
                "row_workers": row_workers,
                "startup_projection": startup_projection,
            });
            if placement == Placement::Cpu {
                model["placement"] = json!("cpu");
                model["projection_backend"] = json!("cpu");
            }
            if vision {
                model["vision"] = json!(true);
                model["mmproj_sha256"] = json!(crate::inference::MMPROJ_SHA256);
                model["image_max_tokens"] = json!(bonsai_cpu_policy::DEFAULT_IMAGE_MAX_TOKENS);
            }
            if prompt_reuse {
                model["prompt_reuse"] = json!("latest");
            }
            if kv_cache != KvCache::F16 {
                let graph = format!(
                    "{hash}:qwen35-text-v1:nonflash-f16:kv-{}-d256-rht3-v1:gdn-f32:prism-arm64-fused-v1",
                    kv_cache.name()
                );
                model["kv_cache"] = json!(kv_cache.name());
                model["startup_graph_identity"] = json!(graph);
            }
            model
        };

        let wire = |model: &serde_json::Value| -> Result<Vec<u8>, String> {
            let body = serde_json::to_string(&json!({ "data": [model] }))
                .map_err(|error| error.to_string())?;
            Ok(format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            )
            .into_bytes())
        };

        for variant in [Bonsai25Variant::Pq2, Bonsai25Variant::Ptq1] {
            let repr = super::super::WeightRepresentation::Bonsai25(variant);
            let is_ptq1 = variant == Bonsai25Variant::Ptq1;
            for vision in [false, true] {
                for kv_cache in [KvCache::F16, KvCache::Rsq4] {
                    for placement in [Placement::Auto, Placement::Cpu] {
                        for row_workers in [1_u8, 4, 8, 12] {
                            for prompt_reuse in [false, true] {
                                for context_tokens in [8_192_usize, 32_768] {
                                    for effective in
                                        ["cpu", "hexagon-native-q2", "hexagon-native-q2-partial"]
                                    {
                                        if effective != "cpu"
                                            && (placement == Placement::Cpu || is_ptq1)
                                        {
                                            continue;
                                        }
                                        let model = build_payload(
                                            variant,
                                            vision,
                                            kv_cache,
                                            placement,
                                            row_workers,
                                            prompt_reuse,
                                            context_tokens,
                                            effective,
                                        );
                                        let policy = WorkerReadiness {
                                            vision,
                                            kv_cache: Some(kv_cache),
                                            placement,
                                            prompt_reuse: if prompt_reuse {
                                                crate::inference::PromptReuseMode::Latest
                                            } else {
                                                crate::inference::PromptReuseMode::Disabled
                                            },
                                            context_tokens,
                                            ..WorkerReadiness::for_version(&pinned, row_workers)
                                        };
                                        let bytes = wire(&model)?;
                                        assert_eq!(
                                            super::validate_native_identity_for_representation(
                                                &bytes, repr, policy,
                                            ),
                                            Ok(()),
                                            "variant={variant:?} vision={vision} kv={kv_cache:?} placement={placement:?} workers={row_workers} reuse={prompt_reuse} ctx={context_tokens} eff={effective}"
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let default_variant = Bonsai25Variant::Pq2;
        let default_repr = super::super::WeightRepresentation::Bonsai25(default_variant);
        let default_policy = WorkerReadiness {
            vision: true,
            kv_cache: Some(KvCache::Rsq4),
            context_tokens: 32_768,
            ..WorkerReadiness::for_version(&pinned, 4)
        };
        let default_model = build_payload(
            default_variant,
            true,
            KvCache::Rsq4,
            Placement::Auto,
            4,
            false,
            32_768,
            "hexagon-native-q2",
        );

        assert_eq!(
            super::validate_native_identity_for_representation(
                b"HTTP/1.1 500 Internal\r\n\r\n",
                default_repr,
                default_policy,
            ),
            Err(NativeIdentityError::Framing)
        );
        assert_eq!(
            super::validate_native_identity_for_representation(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}",
                default_repr,
                default_policy,
            ),
            Err(NativeIdentityError::Schema)
        );

        let mut wrong_model = default_model.clone();
        wrong_model["seed"] = json!(99);
        assert_eq!(
            super::validate_native_identity_for_representation(
                &wire(&wrong_model)?,
                default_repr,
                default_policy
            ),
            Err(NativeIdentityError::Model)
        );

        let mut wrong_context = default_model.clone();
        wrong_context["context_tokens"] = json!(8192);
        assert_eq!(
            super::validate_native_identity_for_representation(
                &wire(&wrong_context)?,
                default_repr,
                default_policy
            ),
            Err(NativeIdentityError::ContextTokens)
        );

        let mut wrong_workers = default_model.clone();
        wrong_workers["row_workers"] = json!(8);
        assert_eq!(
            super::validate_native_identity_for_representation(
                &wire(&wrong_workers)?,
                default_repr,
                default_policy
            ),
            Err(NativeIdentityError::RowWorkers)
        );

        let mut wrong_reuse = default_model.clone();
        wrong_reuse["prompt_reuse"] = json!("latest");
        assert_eq!(
            super::validate_native_identity_for_representation(
                &wire(&wrong_reuse)?,
                default_repr,
                default_policy
            ),
            Err(NativeIdentityError::PromptReuse)
        );

        let mut wrong_placement = default_model.clone();
        wrong_placement["placement"] = json!("cpu");
        assert_eq!(
            super::validate_native_identity_for_representation(
                &wire(&wrong_placement)?,
                default_repr,
                default_policy,
            ),
            Err(NativeIdentityError::Placement)
        );

        let mut wrong_projection = default_model.clone();
        if let Some(object) = wrong_projection.as_object_mut() {
            object.remove("startup_projection");
        }
        assert_eq!(
            super::validate_native_identity_for_representation(
                &wire(&wrong_projection)?,
                default_repr,
                default_policy,
            ),
            Err(NativeIdentityError::StartupProjection)
        );

        let assert_mutated_err = |model: &serde_json::Value,
                                  policy: WorkerReadiness<'_>,
                                  expected: NativeIdentityError|
         -> Result<(), String> {
            assert_eq!(
                super::validate_native_identity_for_representation(
                    &wire(model)?,
                    default_repr,
                    policy,
                ),
                Err(expected)
            );
            Ok(())
        };

        let mut wrong_vision = default_model.clone();
        wrong_vision["vision"] = json!(false);
        assert_mutated_err(&wrong_vision, default_policy, NativeIdentityError::Vision)?;

        let daisy_sha = "cd".repeat(32);
        let mut daisy_model = default_model.clone();
        daisy_model["supports_daisyrose_scale"] = json!(true);
        daisy_model["daisyrose_scale_range"] = json!([0, 2]);
        daisy_model["daisyrose_adapter_sha256"] = json!(daisy_sha.as_str());
        daisy_model["daisyrose_adapter_max_rank"] = json!(16);
        let base_identity = daisy_model["startup_graph_identity"]
            .as_str()
            .ok_or("missing startup_graph_identity")?
            .to_owned();
        let daisy_exec_sha = "ab".repeat(32);
        let daisy_identity =
            format!("{base_identity}:multi-rank-lora-v1:{daisy_exec_sha}:00000000");
        daisy_model["startup_graph_identity"] = json!(daisy_identity);
        let daisy_policy = WorkerReadiness {
            daisyrose: Some((daisy_sha.as_str(), 16)),
            ..default_policy
        };
        assert!(
            super::validate_native_identity_for_representation(
                &wire(&daisy_model)?,
                default_repr,
                daisy_policy
            )
            .is_ok()
        );
        assert_mutated_err(&daisy_model, default_policy, NativeIdentityError::Adapter)?;
        let mut wrong_rank = daisy_model.clone();
        wrong_rank["daisyrose_adapter_max_rank"] = json!(8);
        assert_mutated_err(&wrong_rank, daisy_policy, NativeIdentityError::Adapter)?;
        let mut wrong_scale_identity = daisy_model;
        wrong_scale_identity["startup_graph_identity"] = json!(
            wrong_scale_identity["startup_graph_identity"]
                .as_str()
                .ok_or("missing identity")?
                .replace(":00000000", ":3f000000")
        );
        assert_mutated_err(
            &wrong_scale_identity,
            daisy_policy,
            NativeIdentityError::KvCacheGraph,
        )?;

        assert!(!super::native_supports_cognitive_controls(&wire(
            &default_model
        )?));
        let mut cognitive_off = default_model.clone();
        cognitive_off["supports_cognitive_controls"] = json!(false);
        assert!(!super::native_supports_cognitive_controls(&wire(
            &cognitive_off
        )?));
        assert!(
            super::validate_native_identity_for_representation(
                &wire(&cognitive_off)?,
                default_repr,
                default_policy
            )
            .is_ok()
        );
        let mut cognitive_on = default_model;
        cognitive_on["supports_cognitive_controls"] = json!(true);
        assert!(super::native_supports_cognitive_controls(&wire(
            &cognitive_on
        )?));
        assert!(
            super::validate_native_identity_for_representation(
                &wire(&cognitive_on)?,
                default_repr,
                default_policy
            )
            .is_ok()
        );

        Ok(())
    }

    #[test]
    fn nariq_identity_accepts_trellis_with_daisyrose_and_rejects_drift() -> Result<(), String> {
        use super::NativeIdentityError;
        use serde_json::json;
        let variant = bonsai_cpu_policy::NariqVariant::Bpw2_5;
        let pinned = crate::inference::pinned_version().map_err(|e| format!("{e:?}"))?;
        let daisy_sha = "cd".repeat(32);
        let daisy_exec_sha = "ab".repeat(32);
        let graph = format!(
            "{}:qwen35-text-v1:nonflash-f16:kv-rsq4-d256-rht3-v1:gdn-f32:prism-arm64-fused-v1:trellis-v1:multi-rank-lora-v1:{daisy_exec_sha}:00000000",
            variant.sha256()
        );
        let payload = json!({
            "id": variant.model_id(),
            "object": "model",
            "model_sha256": variant.sha256(),
            "template_sha256": crate::inference::TEMPLATE_SHA256,
            "runtime": "prism-inference",
            "context_tokens": 32_768,
            "seed": 42,
            "row_workers": 4,
            "weight_representation": variant.representation_name(),
            "supports_daisyrose_scale": true,
            "daisyrose_scale_range": [0, 2],
            "daisyrose_adapter_sha256": daisy_sha,
            "daisyrose_adapter_max_rank": 16,
            "kv_cache": "rsq4",
            "startup_graph_identity": graph,
            "startup_projection": {
                "schema_version": 1,
                "requested": "auto",
                "effective": "cpu",
                "identity": null,
                "resident_tensors": 0,
                "resident_weight_bytes": 0,
                "host_metadata_bytes": 0,
            },
        });
        let wire =
            crate::test_util::http_models_response(&payload).map_err(|err| format!("{err:?}"))?;
        let policy = WorkerReadiness {
            kv_cache: Some(crate::inference::KvCache::Rsq4),
            context_tokens: 32_768,
            daisyrose: Some((daisy_sha.as_str(), 16)),
            ..WorkerReadiness::for_version(&pinned, 4)
        };
        let repr = super::super::WeightRepresentation::NariQ(variant);
        assert_eq!(
            super::validate_native_identity_for_representation(&wire, repr, policy),
            Ok(())
        );

        let mut missing_trellis = payload.clone();
        missing_trellis["startup_graph_identity"] = json!(graph.replace(":trellis-v1", ""));
        let missing_trellis_wire = crate::test_util::http_models_response(&missing_trellis)
            .map_err(|err| format!("{err:?}"))?;
        assert_eq!(
            super::validate_native_identity_for_representation(&missing_trellis_wire, repr, policy),
            Err(NativeIdentityError::KvCacheGraph)
        );

        let mut gpu_proj = payload.clone();
        gpu_proj["startup_projection"]["effective"] = json!("adreno-vulkan");
        gpu_proj["startup_projection"]["identity"] = json!("vulkan-test");
        gpu_proj["startup_projection"]["resident_tensors"] = json!(1);
        gpu_proj["startup_projection"]["resident_weight_bytes"] = json!(1024);
        let gpu_wire =
            crate::test_util::http_models_response(&gpu_proj).map_err(|err| format!("{err:?}"))?;
        assert_eq!(
            super::validate_native_identity_for_representation(&gpu_wire, repr, policy),
            Err(NativeIdentityError::StartupProjection)
        );

        let mut with_vision = payload;
        with_vision["vision"] = json!(true);
        let vision_wire = crate::test_util::http_models_response(&with_vision)
            .map_err(|err| format!("{err:?}"))?;
        assert_eq!(
            super::validate_native_identity_for_representation(&vision_wire, repr, policy),
            Err(NativeIdentityError::Vision)
        );
        Ok(())
    }

    #[test]
    fn bonsai25_identity_accepts_pq2_and_ptq1_with_daisyrose_hexagon_and_vision()
    -> Result<(), String> {
        use super::NativeIdentityError;
        use serde_json::json;
        let pinned = crate::inference::pinned_version().map_err(|e| format!("{e:?}"))?;
        let daisy_sha = "cd".repeat(32);
        let daisy_exec_sha = "ab".repeat(32);

        for variant in [Bonsai25Variant::Pq2, Bonsai25Variant::Ptq1] {
            let repr = super::super::WeightRepresentation::Bonsai25(variant);
            let graph = format!(
                "{}:qwen35-text-v1:nonflash-f16:kv-rsq4-d256-rht3-v1:gdn-f32:prism-arm64-fused-v1:multi-rank-lora-v1:{daisy_exec_sha}:00000000",
                variant.sha256()
            );
            let payload = json!({
                "id": variant.model_id(),
                "object": "model",
                "model_sha256": variant.sha256(),
                "template_sha256": crate::inference::TEMPLATE_SHA256,
                "runtime": "prism-inference",
                "context_tokens": 32_768,
                "seed": 42,
                "row_workers": 4,
                "vision": true,
                "mmproj_sha256": crate::inference::MMPROJ_SHA256,
                "image_max_tokens": bonsai_cpu_policy::DEFAULT_IMAGE_MAX_TOKENS,
                "weight_representation": variant.representation_name(),
                "supports_daisyrose_scale": true,
                "daisyrose_scale_range": [0, 2],
                "daisyrose_adapter_sha256": daisy_sha,
                "daisyrose_adapter_max_rank": 16,
                "kv_cache": "rsq4",
                "startup_graph_identity": graph,
                "startup_projection": {
                    "schema_version": 1,
                    "requested": "auto",
                    "effective": "cpu",
                    "identity": null,
                    "resident_tensors": 0,
                    "resident_weight_bytes": 0,
                    "host_metadata_bytes": 0,
                },
            });
            let wire = crate::test_util::http_models_response(&payload)
                .map_err(|err| format!("{err:?}"))?;
            let policy = WorkerReadiness {
                kv_cache: Some(crate::inference::KvCache::Rsq4),
                context_tokens: 32_768,
                vision: true,
                daisyrose: Some((daisy_sha.as_str(), 16)),
                ..WorkerReadiness::for_version(&pinned, 4)
            };
            assert_eq!(
                super::validate_native_identity_for_representation(&wire, repr, policy),
                Ok(())
            );

            // Reject unexpected NariQ `:trellis-v1` tag on Bonsai 2.5.
            let mut bad_trellis = payload;
            bad_trellis["startup_graph_identity"] = json!(format!("{graph}:trellis-v1"));
            let bad_wire = crate::test_util::http_models_response(&bad_trellis)
                .map_err(|err| format!("{err:?}"))?;
            assert_eq!(
                super::validate_native_identity_for_representation(&bad_wire, repr, policy),
                Err(NativeIdentityError::KvCacheGraph)
            );
        }
        Ok(())
    }

    #[test]
    fn daisyrose_accepts_suffix_speculation_with_vulkan_and_composite_placements()
    -> Result<(), String> {
        use super::NativeIdentityError;
        use serde_json::json;

        let variant = Bonsai25Variant::Pq2;
        let pinned = crate::inference::pinned_version().map_err(|e| format!("{e:?}"))?;
        let daisy_sha = "ef".repeat(32);
        let daisy_exec_sha = "12".repeat(32);
        let graph = format!(
            "{}:qwen35-text-v1:nonflash-f16:kv-rsq4-d256-rht3-v1:gdn-f32:prism-arm64-fused-v1:multi-rank-lora-v1:{daisy_exec_sha}:00000000:cpu-tiled-ffn-v1:16:cpu-tiled-verify-v1",
            variant.sha256()
        );

        for (placement, requested, effective, host_meta_bytes, resident_tensors) in [
            (
                crate::inference::Placement::Vulkan,
                "vulkan",
                "adreno-vulkan",
                0_u64,
                1_usize,
            ),
            (
                crate::inference::Placement::Composite,
                "composite",
                "composite-hexagon-vulkan",
                4096_u64,
                2_usize,
            ),
        ] {
            let payload = json!({
                "id": variant.model_id(),
                "object": "model",
                "model_sha256": variant.sha256(),
                "template_sha256": crate::inference::TEMPLATE_SHA256,
                "runtime": "prism-inference",
                "context_tokens": 32_768,
                "seed": 42,
                "row_workers": 4,
                "weight_representation": variant.representation_name(),
                "supports_daisyrose_scale": true,
                "daisyrose_scale_range": [0, 2],
                "daisyrose_adapter_sha256": daisy_sha,
                "daisyrose_adapter_max_rank": 16,
                "kv_cache": "rsq4",
                "startup_graph_identity": graph,
                "cpu_prefill_tokens": 16,
                "cpu_prefill_scratch_bytes": 1,
                "cpu_speculative_tokens": 6,
                "cpu_verification_scratch_bytes": 1,
                "cpu_speculative_checkpoint_bytes": 1,
                "placement": requested,
                "projection_backend": effective,
                "startup_projection": {
                    "schema_version": 1,
                    "requested": requested,
                    "effective": effective,
                    "identity": format!("{effective}-test-v1"),
                    "resident_tensors": resident_tensors,
                    "resident_weight_bytes": 1_048_576_u64,
                    "host_metadata_bytes": host_meta_bytes,
                },
            });
            let wire = crate::test_util::http_models_response(&payload)
                .map_err(|err| format!("{err:?}"))?;
            let policy = WorkerReadiness {
                kv_cache: Some(crate::inference::KvCache::Rsq4),
                context_tokens: 32_768,
                cpu_prefill: Some((16, 1)),
                cpu_speculation: Some((6, 1, 1)),
                placement,
                daisyrose: Some((daisy_sha.as_str(), 16)),
                ..WorkerReadiness::for_version(&pinned, 4)
            };
            assert_eq!(
                super::validate_native_identity_for_representation(
                    &wire,
                    super::super::WeightRepresentation::Bonsai25(variant),
                    policy,
                ),
                Ok(())
            );

            let mut missing_verify = payload;
            missing_verify["startup_graph_identity"] =
                json!(graph.replace(":cpu-tiled-verify-v1", ""));
            let bad_wire = crate::test_util::http_models_response(&missing_verify)
                .map_err(|err| format!("{err:?}"))?;
            assert_eq!(
                super::validate_native_identity_for_representation(
                    &bad_wire,
                    super::super::WeightRepresentation::Bonsai25(variant),
                    policy,
                ),
                Err(NativeIdentityError::KvCacheGraph)
            );
        }
        Ok(())
    }
}
