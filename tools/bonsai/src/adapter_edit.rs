//! Deterministic weight edits to a separately published rank-one adapter.

use std::{
    collections::BTreeMap,
    fs::{self, OpenOptions},
    io::{Read as _, Write as _},
    path::{Path, PathBuf},
};

use serde::Deserialize;
use serde_json::{Value, json};

use crate::{Result, identity::Hasher};

#[cfg(test)]
mod tests;

const ADAPTER_BYTES: u64 = 9_682_464;
const ADAPTER_SHA256: &str = "f1669534803d340a496015f5c45125f3437b4d13ec764f40e34488ce83967f42";
const ALGORITHM: &str = "linear-a-magnitude-prune-stable-index-v1";
const BAND_ALGORITHM: &str = "linear-a-band-magnitude-edit-v1";
const VALID_LINEAR_FAMILIES: [&str; 3] = ["ffn_down", "attn_output", "ssm_out"];

#[derive(Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
enum Operation {
    Inspect,
    Create,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Request {
    schema_version: u32,
    operation: Operation,
    adapter_path: PathBuf,
    keep_numerator: usize,
    keep_denominator: usize,
    output_path: Option<PathBuf>,
    #[serde(default)]
    layer_min: Option<u32>,
    #[serde(default)]
    layer_max: Option<u32>,
    #[serde(default)]
    family_scales: BTreeMap<String, f64>,
    #[serde(default)]
    layer_gains: BTreeMap<u32, f64>,
}

#[derive(Clone, Debug, Default)]
struct SiteFilter<'a> {
    layer_min: Option<u32>,
    layer_max: Option<u32>,
    family_scales: Option<&'a BTreeMap<String, f64>>,
    layer_gains: Option<&'a BTreeMap<u32, f64>>,
}

impl<'a> SiteFilter<'a> {
    fn from_request(request: &'a Request) -> Self {
        Self {
            layer_min: request.layer_min,
            layer_max: request.layer_max,
            family_scales: (!request.family_scales.is_empty()).then_some(&request.family_scales),
            layer_gains: (!request.layer_gains.is_empty()).then_some(&request.layer_gains),
        }
    }

    fn is_active(&self) -> bool {
        self.layer_min.is_some()
            || self.layer_max.is_some()
            || self.family_scales.is_some()
            || self.layer_gains.is_some()
    }

    fn gain_for(&self, layer: u32, family: &str) -> f64 {
        if self.layer_min.is_some_and(|min| layer < min)
            || self.layer_max.is_some_and(|max| layer > max)
        {
            return 0.0;
        }
        let family_gain = self
            .family_scales
            .and_then(|map| map.get(family).copied())
            .unwrap_or(1.0);
        let layer_gain = self
            .layer_gains
            .and_then(|map| map.get(&layer).copied())
            .unwrap_or(1.0);
        family_gain * layer_gain
    }
}

#[expect(
    clippy::as_conversions,
    reason = "Checked finite f64 rank-one adapter coefficients narrow cleanly to f32."
)]
#[inline]
const fn f64_to_f32(value: f64) -> f32 {
    value as f32
}

fn digest(bytes: &[u8]) -> Result<String> {
    let mut hash = Hasher::new()?;
    hash.update(bytes)?;
    hash.finalize_hex()
}

fn has_site_filter(request: &Request) -> bool {
    request.layer_min.is_some()
        || request.layer_max.is_some()
        || !request.family_scales.is_empty()
        || !request.layer_gains.is_empty()
}

fn parse_site_layer_and_family(site: &str) -> Result<(u32, &'static str)> {
    let rest = site
        .strip_prefix("blk.")
        .ok_or_else(|| format!("Unexpected rank-1 linear site name: {site}"))?;
    let (layer_str, suffix) = rest
        .split_once('.')
        .ok_or_else(|| format!("Missing layer separator in rank-1 linear site: {site}"))?;
    let layer = layer_str
        .parse::<u32>()
        .map_err(|_| format!("Invalid layer index in rank-1 linear site: {site}"))?;
    if layer >= 64 {
        return Err(format!("Rank-1 linear site layer exceeds 63: {site}"));
    }
    let family = match suffix {
        "ffn_down.weight" => "ffn_down",
        "attn_output.weight" => "attn_output",
        "ssm_out.weight" => "ssm_out",
        _ => return Err(format!("Unexpected rank-1 linear site family: {site}")),
    };
    Ok((layer, family))
}

fn validate(request: &Request) -> Result<()> {
    if request.schema_version != 1
        || request.keep_numerator == 0
        || request.keep_numerator > request.keep_denominator
        || request.keep_denominator > 1024
    {
        return Err(
            "Use schema 1 and a positive keep fraction at most one, denominator <=1024.".into(),
        );
    }
    if let Some(min) = request.layer_min
        && min >= 64
    {
        return Err("layerMin must be in 0..64.".into());
    }
    if let Some(max) = request.layer_max
        && max >= 64
    {
        return Err("layerMax must be in 0..64.".into());
    }
    if let (Some(min), Some(max)) = (request.layer_min, request.layer_max)
        && min > max
    {
        return Err("layerMin cannot exceed layerMax.".into());
    }
    for (family, &scale) in &request.family_scales {
        if !VALID_LINEAR_FAMILIES.contains(&family.as_str()) {
            return Err(format!("Unsupported family '{family}' in familyScales."));
        }
        if !scale.is_finite() || !(0.0..=100.0).contains(&scale) {
            return Err(format!(
                "familyScales['{family}'] must be finite and in [0.0, 100.0]."
            ));
        }
    }
    for (&layer, &gain) in &request.layer_gains {
        if layer >= 64 {
            return Err(format!("layerGains key {layer} must be in 0..64."));
        }
        if !gain.is_finite() || !(0.0..=100.0).contains(&gain) {
            return Err(format!(
                "layerGains[{layer}] must be finite and in [0.0, 100.0]."
            ));
        }
    }
    bonsai_artifacts::validate_external_path(&request.adapter_path)
        .map_err(|error| error.to_string())?;
    match (&request.operation, &request.output_path) {
        (Operation::Inspect, None) => Ok(()),
        (Operation::Create, Some(path)) => {
            bonsai_artifacts::validate_external_path(path).map_err(|error| error.to_string())?;
            if path == &request.adapter_path
                || fs::exists(path).map_err(|error| error.to_string())?
                || path.parent().is_none_or(|parent| !parent.is_dir())
            {
                return Err("Output must be a new file in an existing external directory.".into());
            }
            if request.keep_numerator == request.keep_denominator && !has_site_filter(request) {
                return Err("Creating an unchanged adapter is not a weight edit.".into());
            }
            Ok(())
        }
        _ => Err("Inspect forbids outputPath; create requires outputPath.".into()),
    }
}

fn read_source(path: &Path) -> Result<Vec<u8>> {
    let mut file = bonsai_artifacts::open_locked_file(path).map_err(|error| error.to_string())?;
    if file.metadata().map_err(|error| error.to_string())?.len() != ADAPTER_BYTES {
        return Err("Adapter editing requires the exact approved source length.".into());
    }
    let mut bytes = Vec::new();
    (&mut file)
        .take(ADAPTER_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| error.to_string())?;
    if u64::try_from(bytes.len()).map_err(|error| error.to_string())? != ADAPTER_BYTES
        || digest(&bytes)? != ADAPTER_SHA256
    {
        return Err("Adapter editing requires the exact approved source digest.".into());
    }
    Ok(bytes)
}

struct Pruned {
    bytes: Vec<u8>,
    original_nonzero: usize,
    retained_nonzero: usize,
    changed: usize,
    original_energy: f64,
    removed_energy: f64,
}

#[derive(Default)]
struct Family {
    sites: usize,
    coefficients: usize,
    retained_nonzero: usize,
    original_energy: f64,
    removed_energy: f64,
}

fn prune(bytes: &[u8], numerator: usize, denominator: usize) -> Result<Pruned> {
    let (chunks, tail) = bytes.as_chunks::<4>();
    if chunks.is_empty() || !tail.is_empty() || chunks.len() > 65_536 {
        return Err("Linear A factors require 1..65536 F32 coefficients.".into());
    }
    let values: Vec<_> = chunks
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect();
    if values.iter().any(|value| !value.is_finite()) {
        return Err("Adapter coefficients must be finite.".into());
    }
    let keep = (values.len() * numerator).div_ceil(denominator);
    let mut indices: Vec<_> = (0..values.len()).collect();
    if keep > 0 && keep < indices.len() {
        let _ = indices.select_nth_unstable_by(keep - 1, |&left, &right| {
            values[right]
                .abs()
                .total_cmp(&values[left].abs())
                .then_with(|| left.cmp(&right))
        });
    }
    let mut retained = vec![false; values.len()];
    for &index in &indices[..keep] {
        retained[index] = true;
    }
    let mut result = Pruned {
        bytes: bytes.to_vec(),
        original_nonzero: 0,
        retained_nonzero: 0,
        changed: 0,
        original_energy: 0.0,
        removed_energy: 0.0,
    };
    for (index, &value) in values.iter().enumerate() {
        let energy = f64::from(value) * f64::from(value);
        result.original_energy += energy;
        result.original_nonzero += usize::from(value != 0.0);
        if retained[index] {
            result.retained_nonzero += usize::from(value != 0.0);
        } else {
            result.removed_energy += energy;
            if value != 0.0 {
                result.bytes[index * 4..index * 4 + 4].copy_from_slice(&0.0_f32.to_le_bytes());
                result.changed += 1;
            }
        }
    }
    Ok(result)
}

fn prune_with_gain(
    bytes: &[u8],
    numerator: usize,
    denominator: usize,
    site_gain: f64,
) -> Result<Pruned> {
    if site_gain == 1.0 {
        return prune(bytes, numerator, denominator);
    }
    let (chunks, tail) = bytes.as_chunks::<4>();
    if chunks.is_empty() || !tail.is_empty() || chunks.len() > 65_536 {
        return Err("Linear A factors require 1..65536 F32 coefficients.".into());
    }
    let values: Vec<_> = chunks
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect();
    if values.iter().any(|value| !value.is_finite()) {
        return Err("Adapter coefficients must be finite.".into());
    }
    if site_gain == 0.0 {
        let mut result = Pruned {
            bytes: bytes.to_vec(),
            original_nonzero: 0,
            retained_nonzero: 0,
            changed: 0,
            original_energy: 0.0,
            removed_energy: 0.0,
        };
        for (index, &value) in values.iter().enumerate() {
            let energy = f64::from(value) * f64::from(value);
            result.original_energy += energy;
            result.removed_energy += energy;
            if value != 0.0 {
                result.original_nonzero += 1;
                result.bytes[index * 4..index * 4 + 4].copy_from_slice(&0.0_f32.to_le_bytes());
                result.changed += 1;
            }
        }
        return Ok(result);
    }
    let pruned = prune(bytes, numerator, denominator)?;
    let mut scaled_bytes = pruned.bytes;
    let mut delta_energy = 0.0_f64;
    let mut retained_nonzero = 0_usize;
    let mut changed = 0_usize;
    for (index, &orig) in values.iter().enumerate() {
        let cur = f32::from_le_bytes([
            scaled_bytes[index * 4],
            scaled_bytes[index * 4 + 1],
            scaled_bytes[index * 4 + 2],
            scaled_bytes[index * 4 + 3],
        ]);
        let next = if cur == 0.0 {
            cur
        } else {
            let scaled = f64_to_f32(f64::from(cur) * site_gain);
            if !scaled.is_finite() {
                return Err("Scaled rank-1 linear A coefficient overflowed F32.".into());
            }
            scaled
        };
        if next != 0.0 {
            retained_nonzero += 1;
        }
        let next_bytes = next.to_le_bytes();
        let orig_bytes = orig.to_le_bytes();
        if next_bytes != orig_bytes && !(cur == 0.0 && orig == 0.0) {
            changed += 1;
        }
        scaled_bytes[index * 4..index * 4 + 4].copy_from_slice(&next_bytes);
        let diff = f64::from(orig) - f64::from(next);
        delta_energy += diff * diff;
    }
    Ok(Pruned {
        bytes: scaled_bytes,
        original_nonzero: pruned.original_nonzero,
        retained_nonzero,
        changed,
        original_energy: pruned.original_energy,
        removed_energy: delta_energy,
    })
}

fn retained_energy(original: f64, removed: f64) -> f64 {
    if original == 0.0 {
        1.0
    } else {
        ((original - removed) / original).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
fn edited(source: &[u8], numerator: usize, denominator: usize) -> Result<(Vec<u8>, Value)> {
    edited_with_filter(source, numerator, denominator, &SiteFilter::default())
}

fn edited_with_filter(
    source: &[u8],
    numerator: usize,
    denominator: usize,
    filter: &SiteFilter<'_>,
) -> Result<(Vec<u8>, Value)> {
    let mut candidate = source.to_vec();
    let ranges = crate::qualification::linear_a_ranges(source)?;
    let mut sites = Vec::new();
    let mut coefficients = 0_usize;
    let mut original_nonzero = 0_usize;
    let mut retained_nonzero = 0_usize;
    let mut changed = 0_usize;
    let mut original_energy = 0.0;
    let mut removed_energy = 0.0;
    let mut minimum_retained_fraction = 1.0_f64;
    let mut active_sites = 0_usize;
    let mut zeroed_sites = 0_usize;
    let mut families: BTreeMap<&str, Family> = BTreeMap::new();
    for (site, range) in ranges {
        let factors = &source[range.clone()];
        let (layer, family_name) = parse_site_layer_and_family(&site)?;
        let site_gain = filter.gain_for(layer, family_name);
        let pruned = prune_with_gain(factors, numerator, denominator, site_gain)?;
        if pruned.retained_nonzero == 0 {
            zeroed_sites += 1;
        } else {
            active_sites += 1;
        }
        let relative_delta = if pruned.original_energy == 0.0 {
            0.0
        } else {
            (pruned.removed_energy / pruned.original_energy).sqrt()
        };
        let fraction = retained_energy(pruned.original_energy, pruned.removed_energy);
        candidate[range].copy_from_slice(&pruned.bytes);
        coefficients += factors.len() / 4;
        original_nonzero += pruned.original_nonzero;
        retained_nonzero += pruned.retained_nonzero;
        changed += pruned.changed;
        original_energy += pruned.original_energy;
        removed_energy += pruned.removed_energy;
        minimum_retained_fraction = minimum_retained_fraction.min(fraction);
        let family = families.entry(family_name).or_default();
        family.sites += 1;
        family.coefficients += factors.len() / 4;
        family.retained_nonzero += pruned.retained_nonzero;
        family.original_energy += pruned.original_energy;
        family.removed_energy += pruned.removed_energy;
        sites.push(
            json!({"site":site,"layer":layer,"siteGain":site_gain,"coefficients":factors.len()/4,
            "originalNonzero":pruned.original_nonzero,"retainedNonzero":pruned.retained_nonzero,
            "changedCoefficients":pruned.changed,"retainedSquaredL2Fraction":fraction,
            "relativeRankOneFrobeniusError":relative_delta,
            "originalASha256":digest(factors)?,"candidateASha256":digest(&pruned.bytes)?}),
        );
    }
    let families: Vec<_> = families
        .into_iter()
        .map(|(name, family)| {
            let fraction = retained_energy(family.original_energy, family.removed_energy);
            let rel_err = if family.original_energy == 0.0 {
                0.0
            } else {
                (family.removed_energy / family.original_energy).sqrt()
            };
            json!({"family":name,"sites":family.sites,"coefficients":family.coefficients,
            "retainedNonzero":family.retained_nonzero,"retainedSquaredL2Fraction":fraction,
            "relativeAggregateAFactorL2Error":rel_err})
        })
        .collect();
    let fraction = retained_energy(original_energy, removed_energy);
    let agg_err = if original_energy == 0.0 {
        0.0
    } else {
        (removed_energy / original_energy).sqrt()
    };
    let summary = json!({"linearSites":sites.len(),"activeLinearSites":active_sites,
        "zeroedLinearSites":zeroed_sites,"coefficients":coefficients,
        "originalNonzero":original_nonzero,"retainedNonzero":retained_nonzero,
        "changedCoefficients":changed,"retainedSquaredL2Fraction":fraction,
        "relativeAggregateAFactorL2Error":agg_err,
        "minimumSiteRetainedSquaredL2Fraction":minimum_retained_fraction,
        "denseLinearAFactorPayloadBytes":coefficients*4,
        "sparseU16IndexF32ValuePayloadBytes":retained_nonzero*6,
        "families":families,"sites":sites});
    Ok((candidate, summary))
}

fn publish(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut staging_name = path
        .file_name()
        .ok_or("Output requires a file name.")?
        .to_os_string();
    staging_name.push(format!(".pending-{}", std::process::id()));
    let staging = path.with_file_name(staging_name);
    bonsai_artifacts::validate_external_path(&staging).map_err(|error| error.to_string())?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&staging)
        .map_err(|error| format!("Cannot create adapter staging file: {error}"))?;
    let published = file
        .write_all(bytes)
        .and_then(|()| file.sync_all())
        .and_then(|()| fs::hard_link(&staging, path));
    drop(file);
    let cleanup = fs::remove_file(&staging);
    match (published, cleanup) {
        (Ok(()), Ok(())) => Ok(()),
        (Ok(()), Err(error)) => Err(format!(
            "Adapter published; staging cleanup failed: {error}"
        )),
        (Err(error), Ok(())) => Err(format!("Cannot publish new adapter: {error}")),
        (Err(error), Err(cleanup)) => Err(format!(
            "Cannot publish new adapter: {error}; staging cleanup failed: {cleanup}"
        )),
    }
}

fn receipt(
    algorithm: &str,
    candidate_sha256: &str,
    candidate_bytes: usize,
    keep_numerator: usize,
    keep_denominator: usize,
    published: bool,
    weight_edit: Value,
) -> Value {
    json!({
        "schemaVersion": 1,
        "kind": "experimental-edited-rank-one-adapter",
        "algorithm": algorithm,
        "sourceAdapterSha256": ADAPTER_SHA256,
        "candidateSha256": candidate_sha256,
        "candidateBytes": candidate_bytes,
        "keepNumerator": keep_numerator,
        "keepDenominator": keep_denominator,
        "published": published,
        "weightEdit": weight_edit,
        "all129SitesRetained": true,
        "bFactorsUnchanged": true,
        "embeddingFactorsUnchanged": true,
        "metadataAndLayoutUnchanged": true,
        "diskSizeReduced": false,
        "behaviorQualified": false,
        "performanceQualified": false,
        "installed": false,
        "scope": "Edits only the existing adapter's linear F32 A coefficients; retains its rank, directions, embedding, metadata and all sites. Sparse payload estimates exclude container, allocator and runtime overhead. Weight-space error is not an accuracy bound."
    })
}

/// Inspects a deterministic magnitude edit or publishes its separately named GGUF.
///
/// # Errors
/// Rejects unsupported requests, unauthenticated source bytes, invalid factor
/// inventories, nonfinite coefficients, no-op edits and unsafe/existing outputs.
pub fn run(request: Value) -> Result<Value> {
    let request: Request = serde_json::from_value(request).map_err(|error| error.to_string())?;
    validate(&request)?;
    let source = read_source(&request.adapter_path)?;
    let filter = SiteFilter::from_request(&request);
    let (candidate, summary) = edited_with_filter(
        &source,
        request.keep_numerator,
        request.keep_denominator,
        &filter,
    )?;
    if request.operation == Operation::Create && summary["changedCoefficients"] == 0 {
        return Err("Selected edit changes no coefficients.".into());
    }
    let candidate_sha256 = digest(&candidate)?;
    if let Some(output) = &request.output_path {
        publish(output, &candidate)?;
    }
    Ok(receipt(
        if filter.is_active() {
            BAND_ALGORITHM
        } else {
            ALGORITHM
        },
        &candidate_sha256,
        candidate.len(),
        request.keep_numerator,
        request.keep_denominator,
        request.operation == Operation::Create,
        summary,
    ))
}
