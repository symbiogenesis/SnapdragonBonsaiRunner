//! Owned Rust execution of the pinned 24 kHz Qwen tokenizer and speaker graphs.
//!
//! Tensor names, geometry, padding, nonlinearities and residual order follow
//! qwentts.cpp 5916bf7c79dbb4e66fc059cdc507a79a23a28bb4. The original
//! MIT notice is retained in LICENSE.qwen-core. No native code runs.
#![forbid(unsafe_code)]
#![expect(
    clippy::redundant_pub_crate,
    reason = "Private checked graph loaders share the speech service's io::Result contract."
)]

use std::io;

use fearless_simd::{Level, dispatch, f32x4, prelude::*};

use crate::{
    SAMPLE_RATE,
    codec::{
        CausalConv1d, CausalTransConv1d, CodecDecodeScratch, CodecStreamState, ConvNeXtBlock,
        DacBlock, DacDecoder, DacResUnit, PadMode, RvqDecoder, RvqEncodeSide, RvqGroup, SnakeBeta,
        TOKENIZER_HOP_24K,
    },
    engine::{LinearProjection, rms_norm_in_place},
    numerics::usize_to_f32,
    speaker_encoder::{
        AttentiveStatPool, AudioMelConfig, EcapaSpeakerEncoder, MelSpectrogramExtractor,
        SeRes2NetBlock, SqueezeExcitation, TdnnConv1d,
    },
    weights::{MappedWeights, Weights, parallel_build},
};

trait CodecSource: Sync {
    fn parallel_load(&self) -> bool;
    fn u32_meta(&self, key: &str) -> io::Result<u32>;
    fn f32_meta(&self, key: &str) -> io::Result<f32>;
    fn u32_array_meta(&self, key: &str) -> io::Result<Vec<u32>>;
    fn tensor_shape(&self, name: &str) -> io::Result<&[usize]>;
    fn tensor_f32(&self, name: &str, shape: &[usize]) -> io::Result<Vec<f32>>;
    fn projection(&self, name: &str, in_dim: usize, out_dim: usize)
    -> io::Result<LinearProjection>;
}

impl CodecSource for Weights {
    fn parallel_load(&self) -> bool {
        self.is_mapped()
    }

    fn u32_meta(&self, key: &str) -> io::Result<u32> {
        Self::u32(self, key)
    }

    fn f32_meta(&self, key: &str) -> io::Result<f32> {
        Self::f32_metadata(self, key)
    }

    fn u32_array_meta(&self, key: &str) -> io::Result<Vec<u32>> {
        Self::u32_array(self, key)
    }

    fn tensor_shape(&self, name: &str) -> io::Result<&[usize]> {
        Self::tensor_shape(self, name)
    }

    fn tensor_f32(&self, name: &str, shape: &[usize]) -> io::Result<Vec<f32>> {
        Self::tensor_f32(self, name, shape)
    }

    fn projection(
        &self,
        name: &str,
        in_dim: usize,
        out_dim: usize,
    ) -> io::Result<LinearProjection> {
        Self::projection(self, name, in_dim, out_dim)
    }
}

#[cfg(windows)]
impl CodecSource for MappedWeights {
    fn parallel_load(&self) -> bool {
        true
    }

    fn u32_meta(&self, key: &str) -> io::Result<u32> {
        Self::u32(self, key)
    }

    fn f32_meta(&self, key: &str) -> io::Result<f32> {
        Self::f32_metadata(self, key)
    }

    fn u32_array_meta(&self, key: &str) -> io::Result<Vec<u32>> {
        Self::u32_array(self, key)
    }

    fn tensor_shape(&self, name: &str) -> io::Result<&[usize]> {
        Self::tensor_shape(self, name)
    }

    fn tensor_f32(&self, name: &str, shape: &[usize]) -> io::Result<Vec<f32>> {
        Self::tensor_f32_mapped(self, name, shape)
    }

    fn projection(
        &self,
        name: &str,
        in_dim: usize,
        out_dim: usize,
    ) -> io::Result<LinearProjection> {
        Self::projection_mapped(self, name, in_dim, out_dim, 1)
    }
}

fn invalid() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        "Unsupported speech codec graph or tensor geometry.",
    )
}

fn dimension<S: CodecSource + ?Sized>(weights: &S, key: &str) -> io::Result<usize> {
    let value = usize::try_from(weights.u32_meta(key)?).map_err(|_| invalid())?;
    if value == 0 || value > 8192 {
        return Err(invalid());
    }
    Ok(value)
}

fn count_float(value: usize) -> io::Result<f32> {
    Ok(f32::from(u16::try_from(value).map_err(|_| invalid())?))
}

fn projection<S: CodecSource + ?Sized>(
    weights: &S,
    name: &str,
    input: usize,
    output: usize,
) -> io::Result<LinearProjection> {
    weights.projection(&format!("{name}.weight"), input, output)
}

fn biased_projection<S: CodecSource + ?Sized>(
    weights: &S,
    name: &str,
    input: usize,
    output: usize,
) -> io::Result<LinearProjection> {
    let mut layer = projection(weights, name, input, output)?;
    layer.bias = Some(weights.tensor_f32(&format!("{name}.bias"), &[output])?);
    Ok(layer)
}

// GGUF omits trailing unit dimensions. A mono [K, IC, 1] convolution
// therefore has the same ordered layout as the stored [K, IC] tensor; no
// other axis may disappear or move, even if the element counts agree.
fn convolution_weight<S: CodecSource + ?Sized>(
    weights: &S,
    name: &str,
    expected: [usize; 3],
) -> io::Result<Vec<f32>> {
    fn canonical(shape: &[usize]) -> &[usize] {
        let end = shape
            .iter()
            .rposition(|&dimension| dimension != 1)
            .map_or(1, |index| index + 1);
        &shape[..end]
    }
    let actual = weights.tensor_shape(name)?.to_vec();
    if canonical(&actual) != canonical(&expected) {
        return Err(invalid());
    }
    weights.tensor_f32(name, &actual)
}

#[expect(
    clippy::too_many_arguments,
    reason = "Explicit convolution geometry is validated before loading weights."
)]
fn conv<S: CodecSource + ?Sized>(
    weights: &S,
    name: &str,
    input: usize,
    output: usize,
    kernel: usize,
    dilation: usize,
    stride: usize,
    bias: bool,
    pad_mode: PadMode,
) -> io::Result<CausalConv1d> {
    Ok(CausalConv1d {
        out_c: output,
        in_c: input,
        kernel,
        dilation,
        stride,
        pad_mode,
        weight: convolution_weight(weights, &format!("{name}.weight"), [kernel, input, output])?,
        bias: if bias {
            Some(weights.tensor_f32(&format!("{name}.bias"), &[output])?)
        } else {
            None
        },
    })
}

fn trans_conv<S: CodecSource + ?Sized>(
    weights: &S,
    name: &str,
    input: usize,
    output: usize,
    kernel: usize,
    stride: usize,
) -> io::Result<CausalTransConv1d> {
    Ok(CausalTransConv1d {
        in_c: input,
        out_c: output,
        kernel,
        stride,
        // GGUF stores the PyTorch [IC, OC, K] array without a permutation.
        weight: convolution_weight(weights, &format!("{name}.weight"), [kernel, output, input])?,
        bias: Some(weights.tensor_f32(&format!("{name}.bias"), &[output])?),
    })
}

fn snake<S: CodecSource + ?Sized>(
    weights: &S,
    name: &str,
    channels: usize,
) -> io::Result<SnakeBeta> {
    SnakeBeta::from_log_params(
        &weights.tensor_f32(&format!("{name}.alpha"), &[channels])?,
        &weights.tensor_f32(&format!("{name}.beta"), &[channels])?,
    )
}

struct TransformerLayer {
    norm: Vec<f32>,
    norm_bias: Option<Vec<f32>>,
    q: LinearProjection,
    k: LinearProjection,
    v: LinearProjection,
    output: LinearProjection,
    attn_scale: Vec<f32>,
    ffn_norm: Vec<f32>,
    ffn_bias: Option<Vec<f32>>,
    gate: Option<LinearProjection>,
    up: LinearProjection,
    down: LinearProjection,
    ffn_scale: Vec<f32>,
}

#[derive(Clone, Default)]
struct KvState {
    keys: Vec<f32>,
    values: Vec<f32>,
}

#[derive(Clone, Default)]
struct TransformerState {
    position: usize,
    layers: Vec<KvState>,
}

pub(crate) trait KvTarget {
    type Guard;
    fn commit(&mut self, guard: Self::Guard);
    fn rollback(&mut self, guard: Self::Guard);
}

pub(crate) fn rollback_kv_slices<T>(
    keys: &mut Vec<T>,
    values: &mut Vec<T>,
    (saved_keys, saved_values): (usize, usize),
) {
    keys.truncate(saved_keys);
    values.truncate(saved_values);
}

// Keep appended KV positions provisional until the enclosing step commits; rollback
// only needs original slice lengths, never a second copy of the KV tensors.
pub(crate) struct KvAppend<'a, T: KvTarget> {
    pub(crate) state: &'a mut T,
    guard: Option<T::Guard>,
}

impl<'a, T: KvTarget> KvAppend<'a, T> {
    pub(crate) const fn from_guard(state: &'a mut T, guard: T::Guard) -> Self {
        Self {
            state,
            guard: Some(guard),
        }
    }

    pub(crate) fn commit(mut self) {
        if let Some(guard) = self.guard.take() {
            self.state.commit(guard);
        }
    }
}

impl<T: KvTarget> Drop for KvAppend<'_, T> {
    fn drop(&mut self) {
        if let Some(guard) = self.guard.take() {
            self.state.rollback(guard);
        }
    }
}

struct TransformerGuard {
    lengths: Vec<(usize, usize)>,
    next: usize,
    keep: Option<usize>,
}

impl KvTarget for TransformerState {
    type Guard = TransformerGuard;

    fn commit(&mut self, guard: Self::Guard) {
        if let Some(keep) = guard.keep {
            for layer in &mut self.layers {
                let discard = layer.keys.len().saturating_sub(keep);
                layer.keys.drain(..discard);
                layer.values.drain(..discard);
            }
        }
        self.position = guard.next;
    }

    fn rollback(&mut self, guard: Self::Guard) {
        for (layer, &saved) in self.layers.iter_mut().zip(&guard.lengths) {
            rollback_kv_slices(&mut layer.keys, &mut layer.values, saved);
        }
        self.layers.truncate(guard.lengths.len());
    }
}

type TransformerAppend<'a> = KvAppend<'a, TransformerState>;

impl<'a> TransformerAppend<'a> {
    fn new(
        state: &'a mut TransformerState,
        stride: usize,
        window: Option<usize>,
    ) -> io::Result<Self> {
        let next = state.position.checked_add(1).ok_or_else(invalid)?;
        let retained = window.map_or(state.position, |window| state.position.min(window));
        let expected = retained.checked_mul(stride).ok_or_else(invalid)?;
        let keep = window
            .map(|window| window.checked_mul(stride).ok_or_else(invalid))
            .transpose()?;
        if stride == 0
            || window == Some(0)
            || (state.layers.is_empty() && state.position != 0)
            || state
                .layers
                .iter()
                .any(|layer| layer.keys.len() != expected || layer.values.len() != expected)
        {
            return Err(invalid());
        }
        let lengths = state
            .layers
            .iter()
            .map(|layer| (layer.keys.len(), layer.values.len()))
            .collect();
        Ok(Self::from_guard(
            state,
            TransformerGuard {
                lengths,
                next,
                keep,
            },
        ))
    }
}

fn zero_workspace(values: &mut Vec<f32>, length: usize) {
    values.clear();
    values.resize(length, 0.0);
}

fn zero_workspace_group(group: &mut Vec<Vec<f32>>, lengths: impl ExactSizeIterator<Item = usize>) {
    group.resize_with(lengths.len(), Vec::new);
    for (slot, length) in group.iter_mut().zip(lengths) {
        zero_workspace(slot, length);
    }
}

struct Transformer {
    // Reuse the service's projection team for both codec graphs; waveform frame
    // order and mutable attention/convolution state remain sequential.
    threads: usize,
    hidden: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    window: Option<usize>,
    theta: f32,
    eps: f32,
    input: Option<LinearProjection>,
    output: Option<LinearProjection>,
    final_norm: Option<Vec<f32>>,
    layers: Vec<TransformerLayer>,
}

fn load_transformer_layer<S: CodecSource + ?Sized>(
    weights: &S,
    prefix: &str,
    index: usize,
    hidden: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    intermediate: usize,
    decoder: bool,
) -> io::Result<TransformerLayer> {
    let name = format!("{prefix}.blk.{index}");
    Ok(TransformerLayer {
        norm: weights.tensor_f32(&format!("{name}.attn_norm.weight"), &[hidden])?,
        norm_bias: if decoder {
            None
        } else {
            Some(weights.tensor_f32(&format!("{name}.attn_norm.bias"), &[hidden])?)
        },
        q: projection(weights, &format!("{name}.attn_q"), hidden, heads * head_dim)?,
        k: projection(
            weights,
            &format!("{name}.attn_k"),
            hidden,
            kv_heads * head_dim,
        )?,
        v: projection(
            weights,
            &format!("{name}.attn_v"),
            hidden,
            kv_heads * head_dim,
        )?,
        output: projection(
            weights,
            &format!("{name}.attn_output"),
            heads * head_dim,
            hidden,
        )?,
        attn_scale: weights.tensor_f32(&format!("{name}.attn_scale"), &[hidden])?,
        ffn_norm: weights.tensor_f32(&format!("{name}.ffn_norm.weight"), &[hidden])?,
        ffn_bias: if decoder {
            None
        } else {
            Some(weights.tensor_f32(&format!("{name}.ffn_norm.bias"), &[hidden])?)
        },
        gate: if decoder {
            Some(projection(
                weights,
                &format!("{name}.ffn_gate"),
                hidden,
                intermediate,
            )?)
        } else {
            None
        },
        up: projection(weights, &format!("{name}.ffn_up"), hidden, intermediate)?,
        down: projection(weights, &format!("{name}.ffn_down"), intermediate, hidden)?,
        ffn_scale: weights.tensor_f32(&format!("{name}.ffn_scale"), &[hidden])?,
    })
}

struct TransformerGeometry {
    prefix: &'static str,
    metadata: &'static str,
    hidden: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    intermediate: usize,
    count: usize,
    theta: f32,
    eps: f32,
}

impl Transformer {
    fn parse_geometry<S: CodecSource + ?Sized>(
        weights: &S,
        decoder: bool,
    ) -> io::Result<TransformerGeometry> {
        let metadata = if decoder {
            "qwen3-tts-tokenizer.decoder"
        } else {
            "qwen3-tts-tokenizer.encoder"
        };
        let prefix = if decoder {
            "tok_dec.pre_tfm"
        } else {
            "tok_enc"
        };
        let hidden = dimension(weights, &format!("{metadata}.hidden_size"))?;
        let heads = dimension(weights, &format!("{metadata}.num_attention_heads"))?;
        let kv_heads = dimension(weights, &format!("{metadata}.num_key_value_heads"))?;
        let head_dim = dimension(weights, &format!("{metadata}.head_dim"))?;
        let intermediate = dimension(weights, &format!("{metadata}.intermediate_size"))?;
        let count = dimension(weights, &format!("{metadata}.num_hidden_layers"))?;
        let theta = weights.f32_meta(&format!("{metadata}.rope_theta"))?;
        let eps = weights.f32_meta(&format!(
            "{metadata}.{}",
            if decoder { "rms_norm_eps" } else { "norm_eps" }
        ))?;
        if count > 16
            || !heads.is_multiple_of(kv_heads)
            || !head_dim.is_multiple_of(2)
            || heads * head_dim > 8192
            || kv_heads * head_dim > 8192
            || !theta.is_finite()
            || theta <= 1.0
            || !eps.is_finite()
            || eps <= 0.0
        {
            return Err(invalid());
        }
        Ok(TransformerGeometry {
            prefix,
            metadata,
            hidden,
            heads,
            kv_heads,
            head_dim,
            intermediate,
            count,
            theta,
            eps,
        })
    }

    fn load_endpoints<S: CodecSource + ?Sized>(
        weights: &S,
        geom: &TransformerGeometry,
        decoder: bool,
    ) -> io::Result<(
        Option<LinearProjection>,
        Option<LinearProjection>,
        Option<Vec<f32>>,
        Option<usize>,
    )> {
        if decoder {
            let latent = dimension(weights, &format!("{}.latent_dim", geom.metadata))?;
            let window = dimension(weights, &format!("{}.sliding_window", geom.metadata))?;
            Ok((
                Some(biased_projection(
                    weights,
                    &format!("{}.input_proj", geom.prefix),
                    latent,
                    geom.hidden,
                )?),
                Some(biased_projection(
                    weights,
                    &format!("{}.output_proj", geom.prefix),
                    geom.hidden,
                    latent,
                )?),
                Some(weights.tensor_f32(&format!("{}.norm.weight", geom.prefix), &[geom.hidden])?),
                Some(window),
            ))
        } else {
            Ok((None, None, None, None))
        }
    }

    fn load<S: CodecSource + ?Sized>(
        weights: &S,
        decoder: bool,
        threads: usize,
    ) -> io::Result<Self> {
        let geom = Self::parse_geometry(weights, decoder)?;
        let (layers, endpoints) = if weights.parallel_load() && threads > 1 {
            let (layers_res, endpoints_res) = std::thread::scope(|scope| {
                let ep_job = scope.spawn(|| Self::load_endpoints(weights, &geom, decoder));
                let layers_res = parallel_build(geom.count, threads, |index| {
                    load_transformer_layer(
                        weights,
                        geom.prefix,
                        index,
                        geom.hidden,
                        geom.heads,
                        geom.kv_heads,
                        geom.head_dim,
                        geom.intermediate,
                        decoder,
                    )
                });
                let endpoints_res = ep_job.join().unwrap_or_else(|_| Err(invalid()));
                (layers_res, endpoints_res)
            });
            (layers_res?, endpoints_res?)
        } else {
            let layers = (0..geom.count)
                .map(|index| {
                    load_transformer_layer(
                        weights,
                        geom.prefix,
                        index,
                        geom.hidden,
                        geom.heads,
                        geom.kv_heads,
                        geom.head_dim,
                        geom.intermediate,
                        decoder,
                    )
                })
                .collect::<io::Result<Vec<_>>>()?;
            let endpoints = Self::load_endpoints(weights, &geom, decoder)?;
            (layers, endpoints)
        };
        let (input, output, final_norm, window) = endpoints;
        Ok(Self {
            threads,
            hidden: geom.hidden,
            heads: geom.heads,
            kv_heads: geom.kv_heads,
            head_dim: geom.head_dim,
            window,
            theta: geom.theta,
            eps: geom.eps,
            input,
            output,
            final_norm,
            layers,
        })
    }

    #[cfg(test)]
    fn step(&self, input: &[f32], state: &mut TransformerState) -> io::Result<Vec<f32>> {
        let mut scratch = CodecTransformerScratch::default();
        self.step_with_scratch(input, state, &mut scratch)?;
        Ok(scratch.hidden)
    }

    fn step_with_scratch<'s>(
        &self,
        input: &[f32],
        state: &mut TransformerState,
        scratch: &'s mut CodecTransformerScratch,
    ) -> io::Result<&'s [f32]> {
        let stride = self
            .kv_heads
            .checked_mul(self.head_dim)
            .ok_or_else(invalid)?;
        let append = TransformerAppend::new(state, stride, self.window)?;
        let state = &mut *append.state;
        if state.layers.is_empty() {
            let reserve = self.window.map_or(0, |w| (w + 1).saturating_mul(stride));
            state.layers.resize_with(self.layers.len(), || KvState {
                keys: Vec::with_capacity(reserve),
                values: Vec::with_capacity(reserve),
            });
        }
        if state.layers.len() != self.layers.len() {
            return Err(invalid());
        }
        if let Some(proj) = &self.input {
            proj.forward_vec_into(input, None, None, self.threads, &mut scratch.hidden)?;
        } else {
            scratch.hidden.clear();
            scratch.hidden.extend_from_slice(input);
        }
        if scratch.hidden.len() != self.hidden || input.iter().any(|value| !value.is_finite()) {
            return Err(invalid());
        }
        scratch.prepare_rope_tables(self.head_dim, state.position, self.theta)?;
        for (layer, cache) in self.layers.iter().zip(&mut state.layers) {
            normalize_into(
                &scratch.hidden,
                &layer.norm,
                layer.norm_bias.as_deref(),
                self.eps,
                &mut scratch.normalized,
            )?;
            LinearProjection::forward_triple_into(
                &layer.q,
                &layer.k,
                &layer.v,
                &scratch.normalized,
                None,
                None,
                None,
                self.threads,
                &mut scratch.q,
                &mut scratch.k,
                &mut scratch.v,
            )?;
            apply_rope_neox_with_tables(
                &mut scratch.q,
                self.heads,
                self.head_dim,
                &scratch.rope_cos,
                &scratch.rope_sin,
            )?;
            apply_rope_neox_with_tables(
                &mut scratch.k,
                self.kv_heads,
                self.head_dim,
                &scratch.rope_cos,
                &scratch.rope_sin,
            )?;
            append_attention_into(
                &scratch.q,
                &scratch.k,
                &scratch.v,
                self.heads,
                self.kv_heads,
                self.head_dim,
                self.window,
                cache,
                &mut scratch.scores,
                &mut scratch.attended,
            )?;
            layer.output.forward_vec_into(
                &scratch.attended,
                None,
                None,
                self.threads,
                &mut scratch.projected,
            )?;
            for ((dst, &value), &scale) in scratch
                .hidden
                .iter_mut()
                .zip(&scratch.projected)
                .zip(&layer.attn_scale)
            {
                *dst += value * scale;
            }
            normalize_into(
                &scratch.hidden,
                &layer.ffn_norm,
                layer.ffn_bias.as_deref(),
                self.eps,
                &mut scratch.normalized,
            )?;
            if let Some(gate) = &layer.gate {
                LinearProjection::forward_pair_into(
                    &layer.up,
                    gate,
                    &scratch.normalized,
                    None,
                    None,
                    self.threads,
                    &mut scratch.up,
                    &mut scratch.gate,
                )?;
                for (value, &gate) in scratch.up.iter_mut().zip(&scratch.gate) {
                    *value *= gate / (1.0 + (-gate).exp());
                }
            } else {
                layer.up.forward_vec_into(
                    &scratch.normalized,
                    None,
                    None,
                    self.threads,
                    &mut scratch.up,
                )?;
                for value in &mut scratch.up {
                    *value = gelu_erf(*value);
                }
            }
            layer.down.forward_vec_into(
                &scratch.up,
                None,
                None,
                self.threads,
                &mut scratch.projected,
            )?;
            for ((dst, &value), &scale) in scratch
                .hidden
                .iter_mut()
                .zip(&scratch.projected)
                .zip(&layer.ffn_scale)
            {
                *dst += value * scale;
            }
        }
        if let Some(norm) = &self.final_norm {
            rms_norm_in_place(&mut scratch.hidden, norm, self.eps)?;
        }
        if let Some(proj) = &self.output {
            proj.forward_vec_into(
                &scratch.hidden,
                None,
                None,
                self.threads,
                &mut scratch.projected,
            )?;
            std::mem::swap(&mut scratch.hidden, &mut scratch.projected);
        }
        if scratch.hidden.iter().any(|value| !value.is_finite()) {
            return Err(invalid());
        }
        append.commit();
        Ok(&scratch.hidden)
    }
}

#[derive(Default)]
struct CodecTransformerScratch {
    hidden: Vec<f32>,
    normalized: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    scores: Vec<f32>,
    attended: Vec<f32>,
    projected: Vec<f32>,
    up: Vec<f32>,
    gate: Vec<f32>,
    rope_denoms: Vec<f32>,
    rope_denoms_key: Option<(usize, u32)>,
    rope_cos: Vec<f32>,
    rope_sin: Vec<f32>,
}

impl CodecTransformerScratch {
    fn prepare_rope_tables(&mut self, head_dim: usize, pos: usize, theta: f32) -> io::Result<()> {
        if head_dim == 0 || !head_dim.is_multiple_of(2) {
            return Err(invalid());
        }
        let half = head_dim / 2;
        let key = (head_dim, theta.to_bits());
        if self.rope_denoms_key != Some(key) {
            let hd_f = usize_to_f32(head_dim);
            self.rope_denoms.clear();
            self.rope_denoms.reserve(half);
            for i in 0..half {
                let exponent = usize_to_f32(2 * i) / hd_f;
                self.rope_denoms.push(theta.powf(exponent));
            }
            self.rope_denoms_key = Some(key);
        }
        self.rope_cos.clear();
        self.rope_sin.clear();
        self.rope_cos.reserve(half);
        self.rope_sin.reserve(half);
        let pos_f = usize_to_f32(pos);
        for &denom in &self.rope_denoms {
            let freq = pos_f / denom;
            let (sin_v, cos_v) = freq.sin_cos();
            self.rope_cos.push(cos_v);
            self.rope_sin.push(sin_v);
        }
        Ok(())
    }
}

#[expect(
    clippy::inline_always,
    reason = "fearless_simd dispatch! requires inline(always) to inherit target_feature attributes."
)]
#[inline(always)]
fn rope_head_simd<S: Simd>(simd: S, lo: &mut [f32], hi: &mut [f32], cos: &[f32], sin: &[f32]) {
    let (lo_chunks, lo_tail) = lo.as_chunks_mut::<4>();
    let (hi_chunks, hi_tail) = hi.as_chunks_mut::<4>();
    let (cos_chunks, cos_tail) = cos.as_chunks::<4>();
    let (sin_chunks, sin_tail) = sin.as_chunks::<4>();
    for (((lo_c, hi_c), cos_c), sin_c) in lo_chunks
        .iter_mut()
        .zip(hi_chunks.iter_mut())
        .zip(cos_chunks.iter())
        .zip(sin_chunks.iter())
    {
        let x0 = f32x4::from_slice(simd, lo_c);
        let x1 = f32x4::from_slice(simd, hi_c);
        let cv = f32x4::from_slice(simd, cos_c);
        let sv = f32x4::from_slice(simd, sin_c);
        x0.mul_add(cv, -(x1 * sv)).store_slice(lo_c);
        x0.mul_add(sv, x1 * cv).store_slice(hi_c);
    }
    for (((x0_ref, x1_ref), &cos_v), &sin_v) in lo_tail
        .iter_mut()
        .zip(hi_tail.iter_mut())
        .zip(cos_tail.iter())
        .zip(sin_tail.iter())
    {
        let x0 = *x0_ref;
        let x1 = *x1_ref;
        *x0_ref = x0.mul_add(cos_v, -x1 * sin_v);
        *x1_ref = x0.mul_add(sin_v, x1 * cos_v);
    }
}

fn apply_rope_neox_with_tables(
    heads: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    cos: &[f32],
    sin: &[f32],
) -> io::Result<()> {
    let half = head_dim / 2;
    if head_dim == 0
        || !head_dim.is_multiple_of(2)
        || cos.len() != half
        || sin.len() != half
        || heads.len() != num_heads * head_dim
    {
        return Err(invalid());
    }
    let level = Level::new();
    dispatch!(level, simd => {
        for h in 0..num_heads {
            let slice = &mut heads[h * head_dim..(h + 1) * head_dim];
            let (lo, hi) = slice.split_at_mut(half);
            rope_head_simd(simd, lo, hi, cos, sin);
        }
    });
    Ok(())
}

#[cfg(test)]
fn normalize(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    eps: f32,
) -> io::Result<Vec<f32>> {
    let mut output = Vec::new();
    normalize_into(input, weight, bias, eps, &mut output)?;
    Ok(output)
}

fn normalize_into(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    eps: f32,
    output: &mut Vec<f32>,
) -> io::Result<()> {
    let Some(bias) = bias else {
        output.clear();
        output.extend_from_slice(input);
        return rms_norm_in_place(output, weight, eps);
    };
    if input.is_empty()
        || input.len() != weight.len()
        || input.len() != bias.len()
        || !eps.is_finite()
        || eps <= 0.0
        || input
            .iter()
            .chain(weight)
            .chain(bias)
            .any(|value| !value.is_finite())
    {
        return Err(invalid());
    }
    let count = count_float(input.len())?;
    let mean = input.iter().sum::<f32>() / count;
    let variance = input
        .iter()
        .map(|value| (value - mean) * (value - mean))
        .sum::<f32>()
        / count;
    let denominator = variance + eps;
    // A finite input can overflow the original F32 reduction. An infinite
    // variance would otherwise yield zero scale and apparently valid output.
    if !mean.is_finite() || !variance.is_finite() || !denominator.is_finite() {
        return Err(invalid());
    }
    let inv = 1.0 / denominator.sqrt();
    output.clear();
    output.extend(
        input
            .iter()
            .zip(weight)
            .zip(bias)
            .map(|((&value, &scale), &offset)| ((value - mean) * inv).mul_add(scale, offset)),
    );
    if output.iter().any(|value| !value.is_finite()) {
        return Err(invalid());
    }
    Ok(())
}

#[cfg(test)]
#[expect(
    clippy::too_many_arguments,
    reason = "Keep query/key/value geometry and owned cache explicit."
)]
fn append_attention(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    heads: usize,
    kv_heads: usize,
    width: usize,
    window: Option<usize>,
    state: &mut KvState,
) -> io::Result<Vec<f32>> {
    let mut scores = Vec::new();
    let mut output = Vec::new();
    append_attention_into(
        query,
        key,
        value,
        heads,
        kv_heads,
        width,
        window,
        state,
        &mut scores,
        &mut output,
    )?;
    Ok(output)
}

#[expect(
    clippy::too_many_arguments,
    reason = "Keep query/key/value geometry and owned cache explicit."
)]
// The enclosing TransformerAppend owns rollback and delayed window eviction.
fn append_attention_into(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    heads: usize,
    kv_heads: usize,
    width: usize,
    window: Option<usize>,
    state: &mut KvState,
    scores: &mut Vec<f32>,
    output: &mut Vec<f32>,
) -> io::Result<()> {
    let stride = kv_heads.checked_mul(width).ok_or_else(invalid)?;
    let query_width = heads.checked_mul(width).ok_or_else(invalid)?;
    if heads == 0
        || kv_heads == 0
        || width == 0
        || !heads.is_multiple_of(kv_heads)
        || query.len() != query_width
        || key.len() != stride
        || value.len() != stride
        || state.keys.len() != state.values.len()
        || !state.keys.len().is_multiple_of(stride)
        || window == Some(0)
        || query
            .iter()
            .chain(key)
            .chain(value)
            .any(|value| !value.is_finite())
    {
        return Err(invalid());
    }
    state.keys.extend_from_slice(key);
    state.values.extend_from_slice(value);
    let total = state.keys.len() / stride;
    let frames = window.map_or(total, |window| total.min(window));
    let first = total - frames;
    zero_workspace(output, query.len());
    zero_workspace(scores, frames);
    let scale = 1.0 / count_float(width)?.sqrt();
    for head in 0..heads {
        let kv_head = head / (heads / kv_heads);
        let q = &query[head * width..(head + 1) * width];
        for (frame, score) in scores.iter_mut().enumerate() {
            let offset = (first + frame) * stride + kv_head * width;
            *score = q
                .iter()
                .zip(&state.keys[offset..offset + width])
                .map(|(&a, &b)| a * b)
                .sum::<f32>()
                * scale;
        }
        let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum = scores
            .iter_mut()
            .map(|score| {
                *score = (*score - maximum).exp();
                *score
            })
            .sum::<f32>();
        if !sum.is_finite() || sum <= 0.0 {
            return Err(invalid());
        }
        for (frame, score) in scores.iter().enumerate() {
            let offset = (first + frame) * stride + kv_head * width;
            for (dst, &value) in output[head * width..(head + 1) * width]
                .iter_mut()
                .zip(&state.values[offset..offset + width])
            {
                *dst = (score / sum).mul_add(value, *dst);
            }
        }
    }
    if output.iter().any(|value| !value.is_finite()) {
        return Err(invalid());
    }
    Ok(())
}

// Mimi uses erf GELU; the DAC's ConvNeXt uses the separate tanh GELU.
// libm executes this operation in Rust without a system math DLL boundary.
fn gelu_erf(value: f32) -> f32 {
    0.5 * value * (1.0 + libm::erff(value * std::f32::consts::FRAC_1_SQRT_2))
}

struct Upsample {
    conv: CausalTransConv1d,
    block: ConvNeXtBlock,
}

struct EncoderStage {
    first: CausalConv1d,
    second: CausalConv1d,
    down: CausalConv1d,
}

struct Encoder {
    threads: usize,
    first: CausalConv1d,
    stages: Vec<EncoderStage>,
    last: CausalConv1d,
    transformer: Transformer,
    down: CausalConv1d,
    semantic: RvqEncodeSide,
    acoustic: RvqEncodeSide,
}

impl Encoder {
    fn load<S: CodecSource + ?Sized>(weights: &S, threads: usize) -> io::Result<Self> {
        let (filters, hidden, compress, kernel, residual, last_kernel, ratios) =
            Self::parse_geometry(weights)?;
        let mut stage_specs = Vec::with_capacity(4);
        let mut channels = filters;
        for (index, ratio) in ratios.into_iter().rev().enumerate() {
            let ratio = usize::try_from(ratio).map_err(|_| invalid())?;
            stage_specs.push((index, channels, ratio));
            channels *= 2;
        }
        let final_channels = channels;
        let load_first = || {
            conv(
                weights,
                "tok_enc.conv.0",
                1,
                filters,
                kernel,
                1,
                1,
                true,
                PadMode::Constant,
            )
        };
        let load_stages = |stage_threads: usize| {
            parallel_build(stage_specs.len(), stage_threads, |i| {
                let (index, ch, ratio) = stage_specs[i];
                load_encoder_stage(weights, ch, compress, residual, index, ratio)
            })
        };
        let load_tail = || Self::load_tail(weights, final_channels, hidden, last_kernel, 2048, 256);

        let (first, stages, transformer, (last, down, semantic, acoustic)) =
            if weights.parallel_load() && threads > 1 {
                let (first_res, stages_res, tfm_res, tail_res) = std::thread::scope(|scope| {
                    let stages_job = scope.spawn(|| load_stages(threads));
                    let tfm_job = scope.spawn(|| Transformer::load(weights, false, threads));
                    let tail_job = scope.spawn(load_tail);
                    let first_res = load_first();
                    let stages_res = stages_job.join().unwrap_or_else(|_| Err(invalid()));
                    let tfm_res = tfm_job.join().unwrap_or_else(|_| Err(invalid()));
                    let tail_res = tail_job.join().unwrap_or_else(|_| Err(invalid()));
                    (first_res, stages_res, tfm_res, tail_res)
                });
                (first_res?, stages_res?, tfm_res?, tail_res?)
            } else {
                (
                    load_first()?,
                    load_stages(1)?,
                    Transformer::load(weights, false, threads)?,
                    load_tail()?,
                )
            };
        Ok(Self {
            first,
            threads,
            stages,
            last,
            transformer,
            down,
            semantic,
            acoustic,
        })
    }

    fn parse_geometry<S: CodecSource + ?Sized>(
        weights: &S,
    ) -> io::Result<(usize, usize, usize, usize, usize, usize, Vec<u32>)> {
        let key = "qwen3-tts-tokenizer.encoder";
        let filters = dimension(weights, &format!("{key}.num_filters"))?;
        let hidden = dimension(weights, &format!("{key}.hidden_size"))?;
        let compress = dimension(weights, &format!("{key}.compress"))?;
        let kernel = dimension(weights, &format!("{key}.kernel_size"))?;
        let residual = dimension(weights, &format!("{key}.residual_kernel_size"))?;
        let last_kernel = dimension(weights, &format!("{key}.last_kernel_size"))?;
        let ratios = weights.u32_array_meta(&format!("{key}.upsampling_ratios"))?;
        let codebook_size = dimension(weights, &format!("{key}.codebook_size"))?;
        let codebook_dim = dimension(weights, &format!("{key}.vector_quantization_hidden_dim"))?;
        if filters != 64
            || hidden != 512
            || compress != 2
            || kernel != 7
            || residual != 3
            || last_kernel != 3
            || ratios != [8, 6, 5, 4]
            || codebook_size != 2048
            || codebook_dim != 256
        {
            return Err(invalid());
        }
        Ok((
            filters,
            hidden,
            compress,
            kernel,
            residual,
            last_kernel,
            ratios,
        ))
    }

    fn load_tail<S: CodecSource + ?Sized>(
        weights: &S,
        channels: usize,
        hidden: usize,
        last_kernel: usize,
        codebook_size: usize,
        codebook_dim: usize,
    ) -> io::Result<(CausalConv1d, CausalConv1d, RvqEncodeSide, RvqEncodeSide)> {
        Ok((
            conv(
                weights,
                "tok_enc.conv.14",
                channels,
                hidden,
                last_kernel,
                1,
                1,
                true,
                PadMode::Constant,
            )?,
            conv(
                weights,
                "tok_enc.downsample",
                hidden,
                hidden,
                4,
                1,
                2,
                false,
                PadMode::Replicate,
            )?,
            load_encode_side(
                weights,
                "tok_enc.vq_semantic",
                1,
                codebook_size,
                codebook_dim,
                hidden,
            )?,
            load_encode_side(
                weights,
                "tok_enc.vq_acoustic",
                15,
                codebook_size,
                codebook_dim,
                hidden,
            )?,
        ))
    }

    fn encode_cancellable(
        &self,
        samples: &[f32],
        cancel: &mut (impl FnMut() -> io::Result<()> + ?Sized),
    ) -> io::Result<Vec<[i32; 16]>> {
        cancel()?;
        if samples.len() < TOKENIZER_HOP_24K
            || samples.len() > 24_000 * 30
            || !samples.iter().all(|value| value.is_finite())
        {
            return Err(invalid());
        }
        // Preserve qt_extract_voice_ref's whole-frame reference alignment.
        // Generated references already contain an integral number of frames.
        let mut frames = samples.len() / TOKENIZER_HOP_24K * TOKENIZER_HOP_24K;
        let (mut hidden, _) =
            self.first
                .forward_parallel(&samples[..frames], frames, self.threads)?;
        for stage in &self.stages {
            cancel()?;
            let residual = hidden.clone();
            elu(&mut hidden);
            (hidden, _) = stage
                .first
                .forward_parallel(&hidden, frames, self.threads)?;
            elu(&mut hidden);
            (hidden, _) = stage
                .second
                .forward_parallel(&hidden, frames, self.threads)?;
            for (value, residual) in hidden.iter_mut().zip(residual) {
                *value += residual;
            }
            elu(&mut hidden);
            (hidden, frames) = stage.down.forward_parallel(&hidden, frames, self.threads)?;
        }
        cancel()?;
        elu(&mut hidden);
        (hidden, frames) = self.last.forward_parallel(&hidden, frames, self.threads)?;
        let mut state = TransformerState::default();
        let mut scratch = CodecTransformerScratch::default();
        let mut transformed = Vec::with_capacity(hidden.len());
        for frame in hidden.chunks_exact(self.transformer.hidden) {
            cancel()?;
            transformed.extend_from_slice(self.transformer.step_with_scratch(
                frame,
                &mut state,
                &mut scratch,
            )?);
        }
        cancel()?;
        (hidden, frames) = self
            .down
            .forward_parallel(&transformed, frames, self.threads)?;
        let mut flat = Vec::with_capacity(frames * 16);
        self.semantic.encode(&hidden, frames, &mut flat)?;
        cancel()?;
        self.acoustic.encode(&hidden, frames, &mut flat)?;
        codebook_major_to_frames(&flat, frames)
    }
}

fn elu(values: &mut [f32]) {
    for value in values {
        if *value < 0.0 {
            *value = value.exp_m1();
        }
    }
}

fn codebook_major_to_frames(codes: &[i32], frames: usize) -> io::Result<Vec<[i32; 16]>> {
    if codes.len() != frames.checked_mul(16).ok_or_else(invalid)? {
        return Err(invalid());
    }
    let mut output = vec![[0; 16]; frames];
    for (frame, row) in output.iter_mut().enumerate() {
        for (book, slot) in row.iter_mut().enumerate() {
            *slot = codes[book * frames + frame];
        }
    }
    Ok(output)
}

fn load_encoder_stage<S: CodecSource + ?Sized>(
    weights: &S,
    channels: usize,
    compress: usize,
    residual: usize,
    index: usize,
    ratio: usize,
) -> io::Result<EncoderStage> {
    let block = 1 + index * 3;
    Ok(EncoderStage {
        first: conv(
            weights,
            &format!("tok_enc.res.{block}.blk.1"),
            channels,
            channels / compress,
            residual,
            1,
            1,
            true,
            PadMode::Constant,
        )?,
        second: conv(
            weights,
            &format!("tok_enc.res.{block}.blk.3"),
            channels / compress,
            channels,
            1,
            1,
            1,
            true,
            PadMode::Constant,
        )?,
        down: conv(
            weights,
            &format!("tok_enc.conv.{}", block + 2),
            channels,
            channels * 2,
            ratio * 2,
            1,
            ratio,
            true,
            PadMode::Constant,
        )?,
    })
}

fn load_encode_side<S: CodecSource + ?Sized>(
    weights: &S,
    name: &str,
    count: usize,
    size: usize,
    width: usize,
    hidden: usize,
) -> io::Result<RvqEncodeSide> {
    let projection =
        weights.tensor_f32(&format!("{name}.input_proj.weight"), &[1, hidden, width])?;
    let codebooks = (0..count)
        .map(|index| weights.tensor_f32(&format!("{name}.{index}.codebook"), &[width, size]))
        .collect::<io::Result<Vec<_>>>()?;
    RvqEncodeSide::new(size, width, hidden, projection, codebooks)
}

fn load_rvq_group<S: CodecSource + ?Sized>(
    weights: &S,
    name: &str,
    count: usize,
    size: usize,
    width: usize,
    hidden: usize,
) -> io::Result<RvqGroup> {
    Ok(RvqGroup {
        num_codebooks: count,
        codebook_size: size,
        codebook_dim: width,
        hidden_dim: hidden,
        codebooks: (0..count)
            .map(|index| weights.tensor_f32(&format!("{name}.{index}.codebook"), &[width, size]))
            .collect::<io::Result<_>>()?,
        out_proj_w: weights
            .tensor_f32(&format!("{name}.output_proj.weight"), &[1, width, hidden])?,
    })
}

#[derive(Clone)]
struct DecodeState {
    audio: CodecStreamState,
    transformer: TransformerState,
}

pub(crate) struct Codec {
    threads: usize,
    weights_path: Option<std::path::PathBuf>,
    #[cfg(windows)]
    mapped_weights: Option<MappedWeights>,
    quantizer: RvqDecoder,
    pre: CausalConv1d,
    transformer: Transformer,
    upsample: Vec<Upsample>,
    dac: DacDecoder,
    encoder: Option<Encoder>,
    state: DecodeState,
    primed: Option<(Vec<[i32; 16]>, DecodeState)>,
    transformer_scratch: CodecTransformerScratch,
    decode_scratch: CodecDecodeScratch,
}

fn load_upsample_stage<S: CodecSource + ?Sized>(
    weights: &S,
    latent: usize,
    index: usize,
) -> io::Result<Upsample> {
    let name = format!("tok_dec.upsample.{index}");
    Ok(Upsample {
        conv: trans_conv(weights, &format!("{name}.conv"), latent, latent, 2, 2)?,
        block: ConvNeXtBlock {
            channels: latent,
            kernel: 7,
            dw_weight: weights.tensor_f32(&format!("{name}.dwconv.weight"), &[7, 1, latent])?,
            dw_bias: weights.tensor_f32(&format!("{name}.dwconv.bias"), &[latent])?,
            norm_w: weights.tensor_f32(&format!("{name}.norm.weight"), &[latent])?,
            norm_b: weights.tensor_f32(&format!("{name}.norm.bias"), &[latent])?,
            pwconv1_w: weights
                .tensor_f32(&format!("{name}.pwconv1.weight"), &[latent, latent * 4])?,
            pwconv1_b: weights.tensor_f32(&format!("{name}.pwconv1.bias"), &[latent * 4])?,
            pwconv2_w: weights
                .tensor_f32(&format!("{name}.pwconv2.weight"), &[latent * 4, latent])?,
            pwconv2_b: weights.tensor_f32(&format!("{name}.pwconv2.bias"), &[latent])?,
            gamma: weights.tensor_f32(&format!("{name}.gamma"), &[latent])?,
        },
    })
}

fn load_dac_block<S: CodecSource + ?Sized>(
    weights: &S,
    index: usize,
    stride: usize,
    channels: usize,
) -> io::Result<DacBlock> {
    let next = channels / 2;
    let name = format!("tok_dec.dec.{}", index + 1);
    let mut units = Vec::with_capacity(3);
    for (unit, dilation) in [1, 3, 9].into_iter().enumerate() {
        let unit_name = format!("{name}.res.{unit}");
        units.push(DacResUnit {
            act1: snake(weights, &format!("{unit_name}.act1"), next)?,
            conv1: conv(
                weights,
                &format!("{unit_name}.conv1"),
                next,
                next,
                7,
                dilation,
                1,
                true,
                PadMode::Constant,
            )?,
            act2: snake(weights, &format!("{unit_name}.act2"), next)?,
            conv2: conv(
                weights,
                &format!("{unit_name}.conv2"),
                next,
                next,
                1,
                1,
                1,
                true,
                PadMode::Constant,
            )?,
        });
    }
    Ok(DacBlock {
        snake1: snake(weights, &format!("{name}.snake"), channels)?,
        trans_conv: trans_conv(
            weights,
            &format!("{name}.conv_t"),
            channels,
            next,
            stride * 2,
            stride,
        )?,
        res_units: units.try_into().map_err(|_| invalid())?,
    })
}

impl Codec {
    #[expect(
        clippy::too_many_lines,
        reason = "Keep the complete pinned decoder tensor mapping together; each subgraph validates its own fixed geometry."
    )]
    pub(crate) fn load(weights: &Weights, threads: usize) -> io::Result<Self> {
        for (key, expected) in [
            ("input_sample_rate", SAMPLE_RATE),
            ("output_sample_rate", SAMPLE_RATE),
            ("decode_upsample_rate", 1920),
            ("encode_downsample_rate", 1920),
        ] {
            if weights.u32(&format!("qwen3-tts-tokenizer.{key}"))? != expected {
                return Err(invalid());
            }
        }
        let key = "qwen3-tts-tokenizer.decoder";
        let size = dimension(weights, &format!("{key}.codebook_size"))?;
        let width = dimension(weights, &format!("{key}.codebook_dim_internal"))?;
        let hidden = dimension(weights, &format!("{key}.vector_quantization_hidden_dim"))?;
        let latent = dimension(weights, &format!("{key}.latent_dim"))?;
        if dimension(weights, &format!("{key}.num_quantizers"))? != 16
            || dimension(weights, &format!("{key}.num_semantic_quantizers"))? != 1
            || size != 2048
            || width != 256
            || hidden != 512
            || latent != 1024
        {
            return Err(invalid());
        }
        let parallel = weights.is_mapped() && threads > 1;
        let step_threads = if parallel { threads } else { 1 };
        let load_quantizer = || {
            Ok::<_, io::Error>(RvqDecoder {
                semantic: load_rvq_group(weights, "tok_dec.vq_first", 1, size, width, hidden)?,
                acoustic: RvqGroup {
                    num_codebooks: 15,
                    codebook_size: size,
                    codebook_dim: width,
                    hidden_dim: hidden,
                    codebooks: parallel_build(15, step_threads, |idx| {
                        weights
                            .tensor_f32(&format!("tok_dec.vq_rest.{idx}.codebook"), &[width, size])
                    })?,
                    out_proj_w: weights
                        .tensor_f32("tok_dec.vq_rest.output_proj.weight", &[1, width, hidden])?,
                },
            })
        };
        let load_pre = || {
            conv(
                weights,
                "tok_dec.pre_conv",
                hidden,
                latent,
                3,
                1,
                1,
                true,
                PadMode::Constant,
            )
        };
        let load_upsample = || {
            parallel_build(2, step_threads.min(2), |index| {
                load_upsample_stage(weights, latent, index)
            })
        };
        let load_dac = || {
            let strides = [
                (0_usize, 8_usize, 1536_usize),
                (1, 5, 768),
                (2, 4, 384),
                (3, 3, 192),
            ];
            let blocks = parallel_build(4, step_threads, |i| {
                let (idx, stride, ch) = strides[i];
                load_dac_block(weights, idx, stride, ch)
            })?;
            Ok::<_, io::Error>(DacDecoder {
                conv_pre: conv(
                    weights,
                    "tok_dec.dec.0.conv",
                    latent,
                    1536,
                    7,
                    1,
                    1,
                    true,
                    PadMode::Constant,
                )?,
                blocks,
                snake_post: snake(weights, "tok_dec.dec.5.snake", 96)?,
                conv_post: conv(
                    weights,
                    "tok_dec.dec.6.conv",
                    96,
                    1,
                    7,
                    1,
                    1,
                    true,
                    PadMode::Constant,
                )?,
            })
        };
        let (quantizer, pre, transformer, upsample, dac) = if parallel {
            let (quantizer_res, pre_res, tfm_res, upsample_res, dac_res) =
                std::thread::scope(|scope| {
                    let q_job = scope.spawn(load_quantizer);
                    let tfm_job = scope.spawn(|| Transformer::load(weights, true, threads));
                    let up_job = scope.spawn(load_upsample);
                    let dac_job = scope.spawn(load_dac);
                    let pre_res = load_pre();
                    let q_res = q_job.join().unwrap_or_else(|_| Err(invalid()));
                    let tfm_res = tfm_job.join().unwrap_or_else(|_| Err(invalid()));
                    let up_res = up_job.join().unwrap_or_else(|_| Err(invalid()));
                    let dac_res = dac_job.join().unwrap_or_else(|_| Err(invalid()));
                    (q_res, pre_res, tfm_res, up_res, dac_res)
                });
            (quantizer_res?, pre_res?, tfm_res?, upsample_res?, dac_res?)
        } else {
            (
                load_quantizer()?,
                load_pre()?,
                Transformer::load(weights, true, threads)?,
                load_upsample()?,
                load_dac()?,
            )
        };
        let mut codec = Self {
            threads,
            weights_path: Some(weights.path().to_path_buf()),
            #[cfg(windows)]
            mapped_weights: weights.retain_mapped_view(),
            quantizer,
            pre,
            transformer,
            upsample,
            dac,
            encoder: None,
            state: DecodeState {
                audio: CodecStreamState::default(),
                transformer: TransformerState::default(),
            },
            primed: None,
            transformer_scratch: CodecTransformerScratch::default(),
            decode_scratch: CodecDecodeScratch::default(),
        };
        codec.reset();
        Ok(codec)
    }

    pub(crate) fn sample_rate(&self) -> u32 {
        self.dac.sample_rate()
    }

    pub(crate) fn discard_reference(&mut self) {
        self.primed = None;
    }

    pub(crate) fn reset(&mut self) {
        let audio = &mut self.state.audio;
        audio.pos = 0;
        zero_workspace(
            &mut audio.pre_conv,
            self.pre.stream_context_frames() * self.pre.in_c,
        );
        zero_workspace_group(
            &mut audio.upsample_dw,
            self.upsample
                .iter()
                .map(|stage| (stage.block.kernel - 1) * stage.block.channels),
        );
        zero_workspace(
            &mut audio.dac_pre,
            self.dac.conv_pre.stream_context_frames() * self.dac.conv_pre.in_c,
        );
        zero_workspace_group(
            &mut audio.dac_carry,
            self.dac
                .blocks
                .iter()
                .map(|block| block.trans_conv.carry_frames() * block.trans_conv.out_c),
        );
        audio
            .dac_ru
            .resize_with(self.dac.blocks.len(), Default::default);
        for (slot, block) in audio.dac_ru.iter_mut().zip(&self.dac.blocks) {
            for (buf, unit) in slot.iter_mut().zip(&block.res_units) {
                zero_workspace(buf, unit.conv1.stream_context_frames() * unit.conv1.in_c);
            }
        }
        zero_workspace(
            &mut audio.dac_post,
            self.dac.conv_post.stream_context_frames() * self.dac.conv_post.in_c,
        );
        audio.kv_k.clear();
        audio.kv_v.clear();
        self.state.transformer = TransformerState::default();
    }

    pub(crate) fn decode_cancellable(
        &mut self,
        codes: &[[i32; 16]],
        cancel: &mut (impl FnMut() -> io::Result<()> + ?Sized),
    ) -> io::Result<Vec<f32>> {
        self.reset();
        self.decode_stream_cancellable(codes, cancel)
    }

    fn step_codec_batch(
        &mut self,
        codes: &[[i32; 16]],
        run_audio: bool,
        cancel: &mut (impl FnMut() -> io::Result<()> + ?Sized),
        mut out: Option<&mut Vec<f32>>,
    ) -> io::Result<()> {
        let count = codes.len();
        if count == 0 {
            return Ok(());
        }
        // Only Qwen3-TTS `kernel == stride` upsample stages are carry-free across
        // arbitrary batch boundaries; fall back per-frame if any stage carries overlap.
        if run_audio
            && count > 1
            && self
                .upsample
                .iter()
                .any(|stage| stage.conv.carry_frames() != 0)
        {
            for row in codes {
                self.step_codec_batch(std::slice::from_ref(row), true, cancel, out.as_deref_mut())?;
            }
            return Ok(());
        }

        cancel()?;
        if count == 1 {
            self.quantizer
                .decode_into(&codes[0], 1, &mut self.decode_scratch)?;
        } else {
            let mut flat = std::mem::take(&mut self.decode_scratch.flat_codes);
            flat.clear();
            let needed_codes = count * 16;
            if flat.capacity() < needed_codes {
                flat.reserve(needed_codes - flat.capacity());
            }
            for row in codes {
                flat.extend_from_slice(row);
            }
            let decoded = self
                .quantizer
                .decode_into(&flat, count, &mut self.decode_scratch);
            self.decode_scratch.flat_codes = flat;
            decoded?;
        }
        self.pre.forward_stream_into(
            &self.decode_scratch.rvq_sem_out,
            count,
            &mut self.state.audio.pre_conv,
            self.threads,
            &mut self.decode_scratch.conv_ext,
            &mut self.decode_scratch.pre_out,
        )?;
        let pre_width = self.decode_scratch.pre_out.len() / count;
        if pre_width == 0 || pre_width * count != self.decode_scratch.pre_out.len() {
            return Err(invalid());
        }
        self.decode_scratch.latents.clear();
        for t in 0..count {
            cancel()?;
            let pre_row = &self.decode_scratch.pre_out[t * pre_width..(t + 1) * pre_width];
            let latent = self.transformer.step_with_scratch(
                pre_row,
                &mut self.state.transformer,
                &mut self.transformer_scratch,
            )?;
            if run_audio {
                if t == 0 {
                    let need = count * latent.len();
                    if self.decode_scratch.latents.capacity() < need {
                        self.decode_scratch
                            .latents
                            .reserve(need - self.decode_scratch.latents.capacity());
                    }
                }
                self.decode_scratch.latents.extend_from_slice(latent);
            }
        }
        if !run_audio {
            self.state.audio.pos = self
                .state
                .audio
                .pos
                .checked_add(count)
                .ok_or_else(invalid)?;
            return Ok(());
        }

        let mut frames = count;
        let mut has_upsampled = false;
        for (stage, context) in self.upsample.iter().zip(&mut self.state.audio.upsample_dw) {
            cancel()?;
            let input = if has_upsampled {
                self.decode_scratch.upsample_b.as_slice()
            } else {
                self.decode_scratch.latents.as_slice()
            };
            let next_frames = stage.conv.forward_into(
                input,
                frames,
                self.threads,
                &mut self.decode_scratch.trans_carry,
                &mut self.decode_scratch.upsample_a,
            )?;
            stage.block.forward_stream_into(
                &self.decode_scratch.upsample_a,
                next_frames,
                context,
                self.threads,
                &mut self.decode_scratch.conv_ext,
                &mut self.decode_scratch.upsample_b,
            )?;
            has_upsampled = true;
            frames = next_frames;
        }
        cancel()?;
        let dac_input = if has_upsampled {
            self.decode_scratch.upsample_b.as_slice()
        } else {
            self.decode_scratch.latents.as_slice()
        };
        let samples = self.dac.forward_stream_with_dac_scratch(
            dac_input,
            frames,
            &mut self.state.audio,
            self.threads,
            &mut self.decode_scratch.dac,
        )?;
        let expected_samples = count.checked_mul(TOKENIZER_HOP_24K).ok_or_else(invalid)?;
        if samples.len() != expected_samples || !samples.iter().all(|value| value.is_finite()) {
            return Err(invalid());
        }
        if let Some(output) = out {
            output.extend_from_slice(samples);
        }
        self.state.audio.pos = self
            .state
            .audio
            .pos
            .checked_add(count)
            .ok_or_else(invalid)?;
        Ok(())
    }

    pub(crate) fn decode_stream_cancellable(
        &mut self,
        codes: &[[i32; 16]],
        cancel: &mut (impl FnMut() -> io::Result<()> + ?Sized),
    ) -> io::Result<Vec<f32>> {
        cancel()?;
        if codes.is_empty()
            || codes.len() > 1500
            || codes
                .iter()
                .flatten()
                .any(|&code| !(0..2048).contains(&code))
        {
            return Err(invalid());
        }
        let mut output = Vec::with_capacity(codes.len() * TOKENIZER_HOP_24K);
        // Bound batch size per DAC call so cooperative cancellation stays responsive
        // during long reference decodes while streaming chunks run in one batch.
        for chunk in codes.chunks(16) {
            self.step_codec_batch(chunk, true, cancel, Some(&mut output))?;
        }
        Ok(output)
    }

    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "Non-cancellable convenience helper retained for tests."
        )
    )]
    pub(crate) fn prime_reference(&mut self, codes: &[[i32; 16]]) -> io::Result<()> {
        self.prime_reference_cancellable(codes, &mut || Ok(()))
    }

    pub(crate) fn prime_reference_cancellable(
        &mut self,
        codes: &[[i32; 16]],
        cancel: &mut (impl FnMut() -> io::Result<()> + ?Sized),
    ) -> io::Result<()> {
        cancel()?;
        if codes.is_empty()
            || codes.len() > (24_000 * 30) / TOKENIZER_HOP_24K
            || codes
                .iter()
                .flatten()
                .any(|&code| !(0..2048).contains(&code))
        {
            return Err(invalid());
        }
        if let Some((reference, state)) = &self.primed
            && reference == codes
        {
            self.state = state.clone();
            return Ok(());
        }
        self.reset();
        // Causal history of `upsample + dac` is strictly `< 8` DAC frames (`<= 12`
        // codec frames), so priming earlier frames only needs `quantizer + pre + transformer`
        // before running the final `<= 12` frames through the full audio graph.
        let audio_start = codes.len().saturating_sub(12);
        for chunk in codes[..audio_start].chunks(16) {
            self.step_codec_batch(chunk, false, cancel, None)?;
        }
        self.step_codec_batch(&codes[audio_start..], true, cancel, None)?;
        self.primed = Some((codes.to_vec(), self.state.clone()));
        Ok(())
    }

    pub(crate) fn encode_reference_cancellable(
        &mut self,
        samples: &[f32],
        cancel: &mut (impl FnMut() -> io::Result<()> + ?Sized),
    ) -> io::Result<Vec<[i32; 16]>> {
        if let Some(encoder) = &self.encoder {
            return encoder.encode_cancellable(samples, cancel);
        }
        #[cfg(windows)]
        if let Some(mapped) = &self.mapped_weights {
            let encoder = Encoder::load(mapped, self.threads)?;
            return encoder.encode_cancellable(samples, cancel);
        }
        let path = self.weights_path.as_deref().ok_or_else(invalid)?;
        let weights = Weights::open(path)?;
        let encoder = Encoder::load(&weights, self.threads)?;
        encoder.encode_cancellable(samples, cancel)
    }
}

pub(crate) struct Speaker {
    model: EcapaSpeakerEncoder,
    threads: usize,
}

fn tdnn<S: CodecSource + ?Sized>(
    weights: &S,
    name: &str,
    input: usize,
    output: usize,
    kernel: usize,
    dilation: usize,
) -> io::Result<TdnnConv1d> {
    Ok(TdnnConv1d {
        out_c: output,
        in_c: input,
        kernel,
        dilation,
        weight: convolution_weight(weights, &format!("{name}.weight"), [kernel, input, output])?,
        bias: weights.tensor_f32(&format!("{name}.bias"), &[output])?,
    })
}

fn load_speaker_block<S: CodecSource + ?Sized>(
    weights: &S,
    index: usize,
    dilation: usize,
) -> io::Result<SeRes2NetBlock> {
    let name = format!("spk_enc.blk.{}", index + 1);
    Ok(SeRes2NetBlock {
        tdnn1: tdnn(weights, &format!("{name}.tdnn1"), 512, 512, 1, 1)?,
        branches: (0..7)
            .map(|branch| {
                tdnn(
                    weights,
                    &format!("{name}.res2net.{branch}"),
                    64,
                    64,
                    3,
                    dilation,
                )
            })
            .collect::<io::Result<_>>()?,
        tdnn2: tdnn(weights, &format!("{name}.tdnn2"), 512, 512, 1, 1)?,
        se: SqueezeExcitation {
            conv1: tdnn(weights, &format!("{name}.se.conv1"), 512, 128, 1, 1)?,
            conv2: tdnn(weights, &format!("{name}.se.conv2"), 128, 512, 1, 1)?,
        },
        scale: 8,
    })
}

impl Speaker {
    pub(crate) fn load(weights: &Weights, threads: usize) -> io::Result<Self> {
        if weights.u32("qwen3-tts.spk_enc.embedding_length")? != 2048
            || weights.u32("qwen3-tts.spk_enc.sample_rate")? != SAMPLE_RATE
        {
            return Err(invalid());
        }
        let mel = MelSpectrogramExtractor::new(AudioMelConfig::default())?;
        let dilations = [2_usize, 3, 4];
        let load_rest = || {
            Ok::<_, io::Error>((
                tdnn(weights, "spk_enc.conv0", 128, 512, 5, 1)?,
                tdnn(weights, "spk_enc.mfa", 1536, 1536, 1, 1)?,
                AttentiveStatPool {
                    tdnn: tdnn(weights, "spk_enc.asp.tdnn", 4608, 128, 1, 1)?,
                    conv: tdnn(weights, "spk_enc.asp.conv", 128, 1536, 1, 1)?,
                },
                tdnn(weights, "spk_enc.fc", 3072, 2048, 1, 1)?,
            ))
        };
        let (blocks, (conv0, mfa, asp, fc)) = if weights.is_mapped() && threads > 1 {
            let (blocks_res, rest_res) = std::thread::scope(|scope| {
                let rest_job = scope.spawn(load_rest);
                let blocks_res = parallel_build(dilations.len(), threads, |idx| {
                    load_speaker_block(weights, idx, dilations[idx])
                });
                let rest_res = rest_job.join().unwrap_or_else(|_| Err(invalid()));
                (blocks_res, rest_res)
            });
            (blocks_res?, rest_res?)
        } else {
            let blocks = parallel_build(dilations.len(), 1, |idx| {
                load_speaker_block(weights, idx, dilations[idx])
            })?;
            (blocks, load_rest()?)
        };
        Ok(Self {
            model: EcapaSpeakerEncoder {
                mel,
                conv0,
                blocks: blocks.try_into().map_err(|_| invalid())?,
                mfa,
                asp,
                fc,
            },
            threads,
        })
    }

    pub(crate) fn extract_cancellable(
        &self,
        samples: &[f32],
        cancel: &mut (impl FnMut() -> io::Result<()> + ?Sized),
    ) -> io::Result<Vec<f32>> {
        cancel()?;
        if samples.is_empty()
            || samples.len() > 24_000 * 30
            || !samples.iter().all(|value| value.is_finite())
        {
            return Err(invalid());
        }
        let vector = self.model.extract_xvector_parallel(samples, self.threads)?;
        cancel()?;
        if vector.len() != 2048 || !vector.iter().all(|value| value.is_finite()) {
            return Err(invalid());
        }
        Ok(vector)
    }
}

#[cfg(test)]
#[path = "rust_codec/transaction_tests.rs"]
mod transaction_tests;

#[cfg(test)]
#[path = "rust_codec/independent_tests.rs"]
mod independent_tests;

#[cfg(test)]
#[path = "rust_codec/synthesis_tests.rs"]
mod synthesis_tests;

#[cfg(test)]
mod tests {
    use super::{KvState, append_attention, codebook_major_to_frames, gelu_erf, normalize};

    fn transformer_projection(
        input: usize,
        output: usize,
        salt: usize,
    ) -> std::io::Result<crate::engine::LinearProjection> {
        let mut values = Vec::with_capacity(input * output);
        for index in 0..input * output {
            let phase = u16::try_from((index * 7 + salt) % 29).map_err(std::io::Error::other)?;
            values.push(f32::from(phase).mul_add(0.001, -0.014));
        }
        crate::engine::LinearProjection::new(
            "codec-fixture.weight",
            input,
            output,
            crate::engine::WeightQuantization::F32(values),
            Some(vec![0.01; output]),
        )
    }

    fn transformer_fixture(decoder: bool) -> std::io::Result<super::Transformer> {
        let hidden = 64;
        let mut layers = Vec::new();
        for salt in 0..2 {
            layers.push(super::TransformerLayer {
                norm: vec![1.0; hidden],
                norm_bias: (!decoder).then(|| vec![0.01; hidden]),
                q: transformer_projection(hidden, hidden, salt)?,
                k: transformer_projection(hidden, hidden / 2, salt + 1)?,
                v: transformer_projection(hidden, hidden / 2, salt + 2)?,
                output: transformer_projection(hidden, hidden, salt + 3)?,
                attn_scale: vec![0.9; hidden],
                ffn_norm: vec![1.0; hidden],
                ffn_bias: (!decoder).then(|| vec![0.02; hidden]),
                gate: if decoder {
                    Some(transformer_projection(hidden, hidden * 2, salt + 4)?)
                } else {
                    None
                },
                up: transformer_projection(hidden, hidden * 2, salt + 5)?,
                down: transformer_projection(hidden * 2, hidden, salt + 6)?,
                ffn_scale: vec![0.8; hidden],
            });
        }
        Ok(super::Transformer {
            threads: 1,
            hidden,
            heads: 4,
            kv_heads: 2,
            head_dim: 16,
            window: decoder.then_some(2),
            theta: 10_000.0,
            eps: 1e-5,
            input: Some(transformer_projection(hidden, hidden, 11)?),
            output: Some(transformer_projection(hidden, hidden, 12)?),
            final_norm: decoder.then(|| vec![1.0; hidden]),
            layers,
        })
    }

    #[test]
    fn codec_transformer_workers_preserve_streaming_outputs_and_complete_kv_state()
    -> std::io::Result<()> {
        for decoder in [false, true] {
            let mut graph = transformer_fixture(decoder)?;
            let mut sequential = super::TransformerState::default();
            let mut parallel = super::TransformerState::default();
            for step in 0_u16..5 {
                let input: Vec<_> = (0_u16..64)
                    .map(|index| f32::from(index + step).mul_add(0.01, -0.3))
                    .collect();
                graph.threads = 1;
                let expected = graph.step(&input, &mut sequential)?;
                graph.threads = 4;
                let actual = graph.step(&input, &mut parallel)?;
                assert_eq!(
                    expected
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    actual
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>()
                );
                assert_eq!(sequential.position, usize::from(step) + 1);
                assert_eq!(sequential.position, parallel.position);
                assert_eq!(sequential.layers.len(), parallel.layers.len());
                for (expected, actual) in sequential.layers.iter().zip(&parallel.layers) {
                    assert_eq!(
                        expected
                            .keys
                            .iter()
                            .map(|value| value.to_bits())
                            .collect::<Vec<_>>(),
                        actual
                            .keys
                            .iter()
                            .map(|value| value.to_bits())
                            .collect::<Vec<_>>()
                    );
                    assert_eq!(
                        expected
                            .values
                            .iter()
                            .map(|value| value.to_bits())
                            .collect::<Vec<_>>(),
                        actual
                            .values
                            .iter()
                            .map(|value| value.to_bits())
                            .collect::<Vec<_>>()
                    );
                    let retained = if decoder {
                        usize::from(step).min(1) + 1
                    } else {
                        usize::from(step) + 1
                    };
                    assert_eq!(actual.keys.len(), retained * 32);
                    assert_eq!(actual.values.len(), retained * 32);
                }
            }
        }
        Ok(())
    }

    struct ConvFixture(std::path::PathBuf);

    impl Drop for ConvFixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn mono_convolution_fixture(shape: &[u64]) -> std::io::Result<ConvFixture> {
        use std::io::Write as _;
        let mut bytes = b"GGUF".to_vec();
        bytes.extend(3_u32.to_le_bytes());
        bytes.extend(2_u64.to_le_bytes());
        bytes.extend(0_u64.to_le_bytes());
        for (name, dimensions, offset) in [
            ("tok_dec.dec.6.conv.weight", shape, 0_u64),
            ("tok_dec.dec.6.conv.bias", &[1_u64][..], 7 * 96 * 4),
        ] {
            bytes.extend(
                u64::try_from(name.len())
                    .map_err(std::io::Error::other)?
                    .to_le_bytes(),
            );
            bytes.extend(name.bytes());
            bytes.extend(
                u32::try_from(dimensions.len())
                    .map_err(std::io::Error::other)?
                    .to_le_bytes(),
            );
            for dimension in dimensions {
                bytes.extend(dimension.to_le_bytes());
            }
            bytes.extend(0_u32.to_le_bytes()); // Independent ordinary F32 fixture.
            bytes.extend(offset.to_le_bytes());
        }
        bytes.resize(bytes.len().next_multiple_of(32), 0);
        for value in 0_u16..7 * 96 {
            bytes.extend(f32::from(value).to_le_bytes());
        }
        bytes.extend(0.25_f32.to_le_bytes());
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(std::io::Error::other)?
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "Bonsai-MonoConv-{}-{nonce}.gguf",
            std::process::id()
        ));
        let fixture = ConvFixture(path);
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&fixture.0)?
            .write_all(&bytes)?;
        Ok(fixture)
    }

    #[test]
    fn pinned_mono_convolution_accepts_omitted_trailing_unit_axes_without_reshaping()
    -> std::io::Result<()> {
        use crate::{codec::PadMode, weights::Weights};
        for shape in [&[7, 96][..], &[7, 96, 1], &[7, 96, 1, 1]] {
            let fixture = mono_convolution_fixture(shape)?;
            let weights = Weights::open(&fixture.0)?;
            let layer = super::conv(
                &weights,
                "tok_dec.dec.6.conv",
                96,
                1,
                7,
                1,
                1,
                true,
                PadMode::Constant,
            )?;
            assert_eq!((layer.kernel, layer.in_c, layer.out_c), (7, 96, 1));
            assert_eq!(
                layer.weight,
                (0_u16..7 * 96).map(f32::from).collect::<Vec<_>>()
            );
            assert_eq!(layer.bias, Some(vec![0.25]));
        }
        for shape in [&[7, 1, 96][..], &[3, 224], &[7, 48, 2]] {
            let fixture = mono_convolution_fixture(shape)?;
            let weights = Weights::open(&fixture.0)?;
            assert!(
                super::conv(
                    &weights,
                    "tok_dec.dec.6.conv",
                    96,
                    1,
                    7,
                    1,
                    1,
                    true,
                    PadMode::Constant
                )
                .is_err()
            );
        }
        Ok(())
    }

    #[test]
    fn codec_attention_retains_only_the_causal_window_with_independent_heads() -> std::io::Result<()>
    {
        let mut state = KvState::default();
        let query = [0.0; 4];
        for (value, expected) in [(2.0_f32, 2.0_f32), (4.0, 3.0), (8.0, 6.0)] {
            let result = append_attention(
                &query,
                &[0.0; 4],
                &[value, value + 1.0, value * 10.0, value.mul_add(10.0, 1.0)],
                2,
                2,
                2,
                Some(2),
                &mut state,
            )?;
            assert_eq!(
                result,
                [
                    expected,
                    expected + 1.0,
                    expected * 10.0,
                    expected.mul_add(10.0, 1.0)
                ]
            );
        }
        // Physical eviction belongs to the complete transformer commit. The
        // last result already excludes the first frame before that commit.
        assert_eq!(state.keys.len(), 12);
        assert_eq!(state.values.len(), 12);
        Ok(())
    }

    #[test]
    fn codec_reference_code_layout_preserves_all_sixteen_books_and_final_frame()
    -> std::io::Result<()> {
        let input: Vec<_> = (0_i32..16)
            .flat_map(|book| [book * 10, book * 10 + 1, book * 10 + 2])
            .collect();
        let output = codebook_major_to_frames(&input, 3)?;
        assert_eq!(output.len(), 3);
        for (frame, row) in output.iter().enumerate() {
            for (book, &code) in row.iter().enumerate() {
                assert_eq!(usize::try_from(code).ok(), Some(book * 10 + frame));
            }
        }
        assert!(codebook_major_to_frames(&input[..47], 3).is_err());
        Ok(())
    }

    #[test]
    fn mimi_uses_layer_norm_bias_and_erf_gelu_instead_of_decoder_rms_norm() -> std::io::Result<()> {
        let result = normalize(&[1.0, 3.0], &[2.0, 3.0], Some(&[0.5, -0.5]), 1e-5)?;
        assert!((result[0] + 1.499_99).abs() < 1e-5);
        assert!((result[1] - 2.499_985).abs() < 1e-5);
        for (input, expected) in [
            (0.0, 0.0),
            (1.0, 0.841_344_7),
            (-1.0, -0.158_655_26),
            (2.0, 1.954_499_7),
        ] {
            assert!((gelu_erf(input) - expected).abs() < 3e-7);
        }
        Ok(())
    }

    #[test]
    fn codec_transformer_reused_rope_matches_per_layer_rope_bits() -> std::io::Result<()> {
        let mut scratch = super::CodecTransformerScratch::default();
        for (head_dim, num_heads, theta) in [(6_usize, 3_usize, 10_000.0_f32), (16, 4, 1_000_000.0)]
        {
            for pos in [0_usize, 1, 7, 15, 73] {
                scratch.prepare_rope_tables(head_dim, pos, theta)?;
                let original: Vec<f32> = (0..num_heads * head_dim)
                    .map(|i| (f32::from(u16::try_from(i % 23).unwrap_or(0)) - 11.0) * 0.0625)
                    .collect();
                let mut expected = original.clone();
                crate::engine::apply_rope_neox_in_place(
                    &mut expected,
                    num_heads,
                    head_dim,
                    pos,
                    theta,
                )?;
                let mut actual = original;
                super::apply_rope_neox_with_tables(
                    &mut actual,
                    num_heads,
                    head_dim,
                    &scratch.rope_cos,
                    &scratch.rope_sin,
                )?;
                assert_eq!(
                    actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "head_dim {head_dim} pos {pos}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn tail_12_prime_and_batched_decode_match_full_decode_bits() -> std::io::Result<()> {
        use crate::codec::{
            CausalConv1d, CausalTransConv1d, CodecDecodeScratch, CodecStreamState, DacBlock,
            DacDecoder, DacResUnit, PadMode, RvqDecoder, RvqGroup, SnakeBeta,
        };
        let rvq_group = |num_codebooks: usize| RvqGroup {
            num_codebooks,
            codebook_size: 4,
            codebook_dim: 2,
            hidden_dim: 2,
            codebooks: (0..num_codebooks)
                .map(|b| {
                    vec![
                        0.1 + f32::from(u8::try_from(b).unwrap_or(0)) * 0.01,
                        -0.2,
                        0.3,
                        0.4,
                        -0.15,
                        0.25,
                        0.05,
                        -0.35,
                    ]
                })
                .collect(),
            out_proj_w: vec![1.0, 0.0, 0.0, 1.0],
        };
        let conv = |dilation: usize, gain: f32| CausalConv1d {
            out_c: 2,
            in_c: 2,
            kernel: 3,
            dilation,
            stride: 1,
            pad_mode: PadMode::Constant,
            weight: vec![gain * 0.25; 2 * 2 * 3],
            bias: Some(vec![0.01, -0.02]),
        };
        let res = |dilation: usize| -> std::io::Result<DacResUnit> {
            Ok(DacResUnit {
                act1: SnakeBeta::from_log_params(&[0.1, -0.2], &[0.0, 0.2])?,
                conv1: conv(dilation, 0.5),
                act2: SnakeBeta::from_log_params(&[0.2, 0.1], &[-0.1, 0.1])?,
                conv2: CausalConv1d {
                    out_c: 2,
                    in_c: 2,
                    kernel: 1,
                    dilation: 1,
                    stride: 1,
                    pad_mode: PadMode::Constant,
                    weight: vec![0.4, -0.1, 0.1, 0.4],
                    bias: None,
                },
            })
        };
        let dac_blocks = [8, 5, 4, 3]
            .into_iter()
            .map(|stride| -> std::io::Result<DacBlock> {
                Ok(DacBlock {
                    snake1: SnakeBeta::from_log_params(&[0.0, 0.1], &[0.0, 0.0])?,
                    trans_conv: CausalTransConv1d {
                        in_c: 2,
                        out_c: 2,
                        kernel: 2 * stride,
                        stride,
                        weight: vec![0.05; 2 * 2 * 2 * stride],
                        bias: Some(vec![0.0, 0.01]),
                    },
                    res_units: [res(1)?, res(3)?, res(9)?],
                })
            })
            .collect::<std::io::Result<Vec<_>>>()?;
        let make_codec = || -> std::io::Result<super::Codec> {
            let upsamples = (0..2)
                .map(|_| super::Upsample {
                    conv: CausalTransConv1d {
                        in_c: 2,
                        out_c: 2,
                        kernel: 2,
                        stride: 2,
                        weight: vec![0.125; 2 * 2 * 2],
                        bias: Some(vec![0.0, -0.01]),
                    },
                    block: crate::codec::ConvNeXtBlock {
                        channels: 2,
                        kernel: 3,
                        dw_weight: vec![0.05; 2 * 3],
                        dw_bias: vec![0.0; 2],
                        norm_w: vec![1.0; 2],
                        norm_b: vec![0.0; 2],
                        pwconv1_w: vec![0.02; 8 * 2],
                        pwconv1_b: vec![0.0; 8],
                        pwconv2_w: vec![0.02; 2 * 8],
                        pwconv2_b: vec![0.0; 2],
                        gamma: vec![0.1; 2],
                    },
                })
                .collect();
            Ok(super::Codec {
                threads: 2,
                weights_path: None,
                #[cfg(windows)]
                mapped_weights: None,
                quantizer: RvqDecoder {
                    semantic: rvq_group(1),
                    acoustic: rvq_group(15),
                },
                pre: conv(1, 0.5),
                transformer: super::Transformer {
                    threads: 2,
                    hidden: 2,
                    heads: 1,
                    kv_heads: 1,
                    head_dim: 2,
                    window: Some(12),
                    theta: 10_000.0,
                    eps: 1e-5,
                    input: None,
                    output: None,
                    final_norm: None,
                    layers: vec![super::TransformerLayer {
                        norm: vec![1.0; 2],
                        norm_bias: None,
                        q: crate::engine::LinearProjection::new(
                            "q",
                            2,
                            2,
                            crate::engine::WeightQuantization::F32(vec![0.2, 0.1, -0.1, 0.2]),
                            None,
                        )?,
                        k: crate::engine::LinearProjection::new(
                            "k",
                            2,
                            2,
                            crate::engine::WeightQuantization::F32(vec![0.1, -0.2, 0.2, 0.1]),
                            None,
                        )?,
                        v: crate::engine::LinearProjection::new(
                            "v",
                            2,
                            2,
                            crate::engine::WeightQuantization::F32(vec![0.3, 0.0, 0.0, 0.3]),
                            None,
                        )?,
                        output: crate::engine::LinearProjection::new(
                            "o",
                            2,
                            2,
                            crate::engine::WeightQuantization::F32(vec![0.25, 0.0, 0.0, 0.25]),
                            None,
                        )?,
                        attn_scale: vec![1.0; 2],
                        ffn_norm: vec![1.0; 2],
                        ffn_bias: None,
                        gate: Some(crate::engine::LinearProjection::new(
                            "g",
                            2,
                            2,
                            crate::engine::WeightQuantization::F32(vec![0.2, 0.0, 0.0, 0.2]),
                            None,
                        )?),
                        up: crate::engine::LinearProjection::new(
                            "u",
                            2,
                            2,
                            crate::engine::WeightQuantization::F32(vec![0.1, 0.1, -0.1, 0.1]),
                            None,
                        )?,
                        down: crate::engine::LinearProjection::new(
                            "d",
                            2,
                            2,
                            crate::engine::WeightQuantization::F32(vec![0.2, 0.0, 0.0, 0.2]),
                            None,
                        )?,
                        ffn_scale: vec![1.0; 2],
                    }],
                },
                upsample: upsamples,
                dac: DacDecoder {
                    conv_pre: conv(1, 0.4),
                    blocks: dac_blocks.clone(),
                    snake_post: SnakeBeta::from_log_params(&[0.0, 0.0], &[0.0, 0.0])?,
                    conv_post: CausalConv1d {
                        out_c: 1,
                        in_c: 2,
                        kernel: 3,
                        dilation: 1,
                        stride: 1,
                        pad_mode: PadMode::Constant,
                        weight: vec![0.1; 6],
                        bias: None,
                    },
                },
                encoder: None,
                state: super::DecodeState {
                    audio: CodecStreamState {
                        pos: 0,
                        pre_conv: vec![0.0; 4],
                        upsample_dw: vec![vec![0.0; 4], vec![0.0; 4]],
                        dac_pre: vec![0.0; 4],
                        dac_carry: [8, 5, 4, 3].into_iter().map(|s| vec![0.0; s * 2]).collect(),
                        dac_ru: (0..4)
                            .map(|_| [vec![0.0; 4], vec![0.0; 12], vec![0.0; 36]])
                            .collect(),
                        dac_post: vec![0.0; 4],
                        ..CodecStreamState::default()
                    },
                    transformer: super::TransformerState::default(),
                },
                primed: None,
                transformer_scratch: super::CodecTransformerScratch::default(),
                decode_scratch: CodecDecodeScratch::default(),
            })
        };

        let codes: Vec<[i32; 16]> = (0..20_usize)
            .map(|t| std::array::from_fn(|b| i32::try_from((t + b) % 4).unwrap_or(0)))
            .collect();
        let mut full = make_codec()?;
        let mut per_frame = Vec::new();
        for row in &codes {
            per_frame
                .extend(full.decode_stream_cancellable(std::slice::from_ref(row), &mut || Ok(()))?);
        }
        let mut batched = make_codec()?;
        let mut dyadic = Vec::new();
        for (start, len) in [(0, 1), (1, 2), (3, 4), (7, 8), (15, 5)] {
            dyadic.extend(
                batched.decode_stream_cancellable(&codes[start..start + len], &mut || Ok(()))?,
            );
        }
        assert_eq!(
            dyadic.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            per_frame.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );

        let mut primed = make_codec()?;
        primed.prime_reference(&codes)?;
        assert_eq!(primed.state.audio.pos, full.state.audio.pos);
        assert_eq!(primed.state.audio.pre_conv, full.state.audio.pre_conv);
        assert_eq!(primed.state.audio.upsample_dw, full.state.audio.upsample_dw);
        assert_eq!(primed.state.audio.dac_pre, full.state.audio.dac_pre);
        assert_eq!(primed.state.audio.dac_carry, full.state.audio.dac_carry);
        assert_eq!(primed.state.audio.dac_ru, full.state.audio.dac_ru);
        assert_eq!(primed.state.audio.dac_post, full.state.audio.dac_post);
        assert_eq!(
            primed.state.transformer.position,
            full.state.transformer.position
        );
        for (a, b) in primed
            .state
            .transformer
            .layers
            .iter()
            .zip(&full.state.transformer.layers)
        {
            assert_eq!(a.keys, b.keys);
            assert_eq!(a.values, b.values);
        }
        Ok(())
    }
}
