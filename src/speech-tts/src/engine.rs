//! Pure-Rust Qwen3-TTS engine (`Talker`, `CodePredictor`, `PromptBuilder`, `LoRA` overlays,
//! and Bonsai-style ternary `PQ2_0`/`PTQ1_0` with Hadamard folding).
//!
//! Implements the pure-Rust execution stack for `Qwen3-TTS-12Hz-1.7B-VoiceDesign` and `Base`:
//! - 28-layer `Talker` transformer (`RMSNorm`, GQA with QK-norm, 3D `MRoPE` / 1D NEOX `RoPE`, `SwiGLU`, KV cache).
//! - Weight formats: `F32`, `Q8_0`, and Bonsai-style group-128 ternary `PQ2_0` / lossless `PTQ1_0`
//!   dual-bitplane repack with block-1024 Sylvester Walsh–Hadamard + sign folding (`prism.hadamard.*`).
//! - Optional low-rank / rank-one `LoRA` adapter overlays (`TalkerLoraAdapter`) with bit-exact
//!   loaded-at-zero (`scale == 0.0`) identity.
//! - 5-layer `CodePredictor` (`mtp_proj`, 15 residual codebooks `1..=15`) with decoupled
//!   `subtalker_*` sampling parameters.
//! - `PromptBuilder` and `PromptCache` with cached `icl_codec_stream` and `icl_ref_text_stream`.
//!
//! Execution is CPU-only. Signed Hexagon dispatch, placement and shared-cache
//! ownership require a separately qualified implementation; no queue protocol
//! or device execution is supplied by this module.
#![forbid(unsafe_code)]
#![expect(
    clippy::missing_errors_doc,
    reason = "Internal tensor, quantization, and layer shape checks return io::ErrorKind::InvalidInput or InvalidData."
)]

use std::{collections::HashMap, io};

use fearless_simd::{Level, dispatch, f32x4, i32x4, prelude::*};

use crate::{
    codec::codec_ref_key,
    numerics::{axpy_simd, f32_dot_simd, usize_to_f32},
    sampling,
};

pub(crate) mod decoder_workspace;
mod kv_append;
mod packed;
pub(crate) mod predictor_workspace;
mod projection_layout;
pub(crate) mod row_workers;

/// Number of residual acoustic codebooks predicted by `CodePredictor` (`15`).
pub const NUM_ACOUSTIC_CODEBOOKS: usize = 15;
/// Default low-latency streaming chunk size in 12.5 Hz codec frames (`4` frames = `320` ms).
pub const DEFAULT_STREAM_CHUNK_FRAMES: usize = 4;
/// Minimum supported streaming chunk size in 12.5 Hz codec frames (`1` frame = `80` ms).
pub const MIN_STREAM_CHUNK_FRAMES: usize = 1;
/// Maximum supported streaming chunk size in 12.5 Hz codec frames (`32` frames = `2.56` s).
pub const MAX_STREAM_CHUNK_FRAMES: usize = 32;

/// Configurable streaming codec emission schedule (`streaming-qwen3-tts` / `qwen3-tts-rs`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamChunkConfig {
    /// Number of 12.5 Hz frames in the first emitted audio chunk (`1..=32`, default `4`).
    pub first_chunk_frames: usize,
    /// Number of 12.5 Hz frames in subsequent steady-state chunks (`1..=32`, default `4`).
    pub steady_chunk_frames: usize,
}

impl StreamChunkConfig {
    /// Low-latency 4-frame (`320` ms) streaming schedule halving first-chunk TTFA vs. 8-frame chunking.
    #[must_use]
    pub const fn low_latency_4_frame() -> Self {
        Self {
            first_chunk_frames: DEFAULT_STREAM_CHUNK_FRAMES,
            steady_chunk_frames: DEFAULT_STREAM_CHUNK_FRAMES,
        }
    }

    /// Standard 8-frame (`640` ms) streaming schedule.
    #[must_use]
    pub const fn standard_8_frame() -> Self {
        Self {
            first_chunk_frames: 8,
            steady_chunk_frames: 8,
        }
    }

    /// Dyadic latency ramp (`1 -> 2 -> 4 -> 8` frames) for `80` ms TTFA converging to `640` ms chunks.
    #[must_use]
    pub const fn dyadic_ramp() -> Self {
        Self {
            first_chunk_frames: 1,
            steady_chunk_frames: 8,
        }
    }

    /// Validate that both first and steady chunk sizes lie within `MIN_STREAM_CHUNK_FRAMES..=MAX_STREAM_CHUNK_FRAMES`.
    pub fn validate(self) -> io::Result<Self> {
        if self.first_chunk_frames < MIN_STREAM_CHUNK_FRAMES
            || self.first_chunk_frames > MAX_STREAM_CHUNK_FRAMES
            || self.steady_chunk_frames < MIN_STREAM_CHUNK_FRAMES
            || self.steady_chunk_frames > MAX_STREAM_CHUNK_FRAMES
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "StreamChunkConfig frame counts must be within 1..=32.",
            ));
        }
        Ok(self)
    }

    /// Target frame count for emission `chunk_index` (`0` for first chunk, `> 0` for steady state).
    #[must_use]
    pub const fn frames_for_chunk(self, chunk_index: usize) -> usize {
        if chunk_index == 0 {
            self.first_chunk_frames
        } else {
            self.steady_chunk_frames
        }
    }

    /// Progressive target frame count (`first -> 2*first -> 4*first -> steady`, capped at `steady`).
    #[must_use]
    pub const fn dyadic_frames_for_chunk(self, chunk_index: usize) -> usize {
        let cap = self.steady_chunk_frames;
        let scaled = match chunk_index {
            0 => self.first_chunk_frames,
            1 => self.first_chunk_frames.saturating_mul(2),
            2 => self.first_chunk_frames.saturating_mul(4),
            _ => cap,
        };
        if scaled < cap { scaled } else { cap }
    }
}
/// Total RVQ codebooks per 12.5 Hz frame (`1` semantic + `15` acoustic = `16`).
pub const TOTAL_CODE_GROUPS: usize = 16;
/// Default Hadamard block size for the 1.7B `Talker` (`1024`, dividing `hidden_size = 2048`).
pub const TALKER_HADAMARD_BLOCK_SIZE: usize = 1024;
/// Default ternary quantization group size (`128`).
pub const TERNARY_GROUP_SIZE: usize = 128;

/// Decoupled sampling parameters for `Talker` (codebook 0) vs. `CodePredictor` (codebooks `1..=15`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecoupledSamplingParams {
    /// `Talker` semantic temperature (`0.9` for `VoiceDesign`, `0.85` for `Base` synthesis).
    pub temperature: f32,
    /// `Talker` semantic top-k (`50`).
    pub top_k: i32,
    /// `Talker` semantic top-p (`1.0` for `VoiceDesign`, `0.9` for `Base` synthesis).
    pub top_p: f32,
    /// Windowed (`72`-token) repetition penalty (`1.05`).
    pub repetition_penalty: f32,
    /// `CodePredictor` acoustic temperature (`0.9` for `VoiceDesign`, `0.6` for `Base` synthesis).
    pub subtalker_temperature: f32,
    /// `CodePredictor` acoustic top-k (`50`).
    pub subtalker_top_k: i32,
    /// `CodePredictor` acoustic top-p (`1.0` for `VoiceDesign`, `0.9` for `Base` synthesis).
    pub subtalker_top_p: f32,
}

impl DecoupledSamplingParams {
    /// Expressive `VoiceDesign` reference bootstrap sampling profile.
    #[must_use]
    pub const fn voice_design() -> Self {
        Self {
            temperature: 0.9,
            top_k: 50,
            top_p: 1.0,
            repetition_penalty: 1.05,
            subtalker_temperature: 0.9,
            subtalker_top_k: 50,
            subtalker_top_p: 1.0,
        }
    }

    /// High-stability `Base` voice-clone synthesis sampling profile with cooled `CodePredictor`.
    #[must_use]
    pub const fn base_clone() -> Self {
        Self {
            temperature: 0.85,
            top_k: 50,
            top_p: 0.9,
            repetition_penalty: 1.05,
            subtalker_temperature: 0.6,
            subtalker_top_k: 50,
            subtalker_top_p: 0.9,
        }
    }
}

/// One 32-element `Q8_0` quantization block.
#[derive(Clone, Debug, PartialEq)]
pub struct Q8Block {
    /// Per-block floating-point scale.
    pub scale: f32,
    /// 32 signed 8-bit quantized values.
    pub quants: [i8; 32],
}

/// Supported weight representations for `Talker` and `CodePredictor` linear projections.
/// Shared zero-copy byte slice inside a read-only OS-mapped GGUF section.
#[derive(Clone)]
pub struct MappedSlice {
    mapped: std::sync::Arc<bonsai_windows_process::MappedFile>,
    start: usize,
    end: usize,
}

impl MappedSlice {
    /// Construct a bounds-checked slice within an OS-mapped GGUF file.
    pub fn new(
        mapped: std::sync::Arc<bonsai_windows_process::MappedFile>,
        start: usize,
        end: usize,
    ) -> io::Result<Self> {
        if start > end || end > mapped.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Mapped tensor slice exceeds mapped file bounds.",
            ));
        }
        Ok(Self { mapped, start, end })
    }

    /// Borrow the immutable bytes covering this tensor.
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        &self.mapped.as_slice()[self.start..self.end]
    }
}

impl std::fmt::Debug for MappedSlice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MappedSlice")
            .field("len", &(self.end.saturating_sub(self.start)))
            .finish()
    }
}

impl PartialEq for MappedSlice {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

/// Matrix weight representation for pure-Rust projections.
#[derive(Clone, Debug, PartialEq)]
pub enum WeightQuantization {
    /// Unquantized row-major `F32` weights `[out_dim * in_dim]`.
    F32(Vec<f32>),
    /// Original GGML `F32/F16/Q8_0/Q4_K/Q6_K` blocks, decoded one row at a time.
    Packed {
        /// GGML tensor type (0, 1, 8, 12 or 14).
        ggml_type: u32,
        /// Original little-endian tensor bytes without whole-model expansion.
        bytes: Vec<u8>,
    },
    /// Zero-copy OS-mapped GGML `F32/F16/Q8_0/Q4_K/Q6_K` blocks.
    MappedPacked {
        /// GGML tensor type (`0`, `1`, `8`, `12`, or `14`).
        ggml_type: u32,
        /// Zero-copy slice inside a read-only memory-mapped GGUF file.
        bytes: MappedSlice,
    },
    /// 8-row repacked `Q4_K` matrix tuned for `SDOT` / `I8MM` kernels (`out_dim % 8 == 0`).
    RepackedQ4K8 {
        /// Repacked 8-row `Q4_K` superblock strips (`1184` bytes per 8x256 tile).
        bytes: Vec<u8>,
    },
    /// 32-element block-quantized `Q8_0` weights.
    Q8_0 {
        /// Quantized 32-weight blocks.
        blocks: Vec<Q8Block>,
    },
    /// Bonsai-style 2-bit packed ternary (`PQ2_0`, 4 trits per byte: `0 -> -1, 1 -> 0, 2 -> +1`)
    /// with group-128 scales and input-side block Walsh–Hadamard + sign folding.
    PQ2_0 {
        /// Quantization group size (`128`).
        group_size: usize,
        /// Per-group scales `[out_dim * (in_dim / group_size)]`.
        scales: Vec<f32>,
        /// Packed 2-bit trits `[out_dim * (in_dim / 4)]`.
        packed_trits: Vec<u8>,
        /// Walsh–Hadamard transform block size (`1024`, or power-of-two dividing `in_dim`).
        hadamard_block: usize,
        /// Input-side Rademacher signs (`+1` or `-1`) `[in_dim]`.
        signs: Vec<i8>,
    },
    /// Lossless dual-bitplane (`nz_mask`, `sign_mask`) ternary repack (`PTQ1_0`).
    /// This CPU layout is not the signed DSP's tiled `HTP_TYPE_Q2_0` format.
    PTQ1_0 {
        /// Quantization group size (`128`).
        group_size: usize,
        /// Per-group scales `[out_dim * (in_dim / group_size)]`.
        scales: Vec<f32>,
        /// Non-zero bitplane (8 weights per byte; bit `b` is `1` when weight is `!= 0`).
        nz_mask: Vec<u8>,
        /// Sign bitplane (8 weights per byte; bit `b` is `1` when weight is `+1`, `0` when `-1`).
        sign_mask: Vec<u8>,
        /// Walsh–Hadamard transform block size (`1024`, or power-of-two dividing `in_dim`).
        hadamard_block: usize,
        /// Input-side Rademacher signs (`+1` or `-1`) `[in_dim]`.
        signs: Vec<i8>,
    },
}

impl WeightQuantization {
    pub(crate) fn packed_slice(&self) -> Option<(u32, &[u8])> {
        match self {
            Self::Packed { ggml_type, bytes } => Some((*ggml_type, bytes.as_slice())),
            Self::MappedPacked { ggml_type, bytes } => Some((*ggml_type, bytes.as_slice())),
            _ => None,
        }
    }
}

/// Returns `true` if `tensor_name` is an eligible 28-layer `Talker` projection for
/// `PQ2_0` / `PTQ1_0` ternary compression (`talker.blk.{l}.attn_{q,k,v,output}.weight`
/// or `talker.blk.{l}.ffn_{gate,up,down}.weight`).
#[must_use]
pub fn is_talker_ternary_eligible(tensor_name: &str) -> bool {
    let Some(rest) = tensor_name.strip_prefix("talker.blk.") else {
        return false;
    };
    let Some((layer_str, suffix)) = rest.split_once('.') else {
        return false;
    };
    if layer_str.is_empty() || !layer_str.bytes().all(|b| b.is_ascii_digit()) {
        return false;
    }
    matches!(
        suffix,
        "attn_q.weight"
            | "attn_k.weight"
            | "attn_v.weight"
            | "attn_output.weight"
            | "ffn_gate.weight"
            | "ffn_up.weight"
            | "ffn_down.weight"
    )
}

/// Enforce the Qwen3-TTS mixed-precision ternary policy: never ternarize `CodePredictor`,
/// `codec_head`, `text_embd`, `codec_embd`, `text_proj`, norms, codec, or speaker encoder.
pub fn validate_talker_ternary_policy(
    tensor_name: &str,
    quant: &WeightQuantization,
) -> io::Result<()> {
    if matches!(
        quant,
        WeightQuantization::PQ2_0 { .. } | WeightQuantization::PTQ1_0 { .. }
    ) && !is_talker_ternary_eligible(tensor_name)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Tensor is excluded from PQ2_0/PTQ1_0 ternary quantization by speech policy.",
        ));
    }
    Ok(())
}

/// Losslessly repack `PQ2_0` 2-bit trits into `PTQ1_0` dual bitplanes (`nz_mask`, `sign_mask`).
pub fn repack_pq2_to_ptq1(quant: &WeightQuantization) -> io::Result<WeightQuantization> {
    let WeightQuantization::PQ2_0 {
        group_size,
        scales,
        packed_trits,
        hadamard_block,
        signs,
    } = quant
    else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Expected PQ2_0 weight tensor for PTQ1_0 repack.",
        ));
    };
    if !packed_trits.len().is_multiple_of(2) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "PQ2_0 packed_trits length must be even (8 trits per bitplane byte).",
        ));
    }
    let out_bytes = packed_trits.len() / 2;
    let mut nz_mask = Vec::with_capacity(out_bytes);
    let mut sign_mask = Vec::with_capacity(out_bytes);

    let (pairs, _) = packed_trits.as_chunks::<2>();
    for pair in pairs {
        let mut nz = 0_u8;
        let mut sg = 0_u8;
        for (byte_idx, &b) in pair.iter().enumerate() {
            for sub in 0..4 {
                let code = (b >> (sub * 2)) & 0b11;
                let bit_pos = byte_idx * 4 + sub;
                match code {
                    0 => {
                        // -1
                        nz |= 1_u8 << bit_pos;
                    }
                    1 => {
                        // 0
                    }
                    2 => {
                        // +1
                        nz |= 1_u8 << bit_pos;
                        sg |= 1_u8 << bit_pos;
                    }
                    _ => {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "Invalid trit code 3 in PQ2_0 tensor.",
                        ));
                    }
                }
            }
        }
        nz_mask.push(nz);
        sign_mask.push(sg);
    }

    Ok(WeightQuantization::PTQ1_0 {
        group_size: *group_size,
        scales: scales.clone(),
        nz_mask,
        sign_mask,
        hadamard_block: *hadamard_block,
        signs: signs.clone(),
    })
}

#[expect(
    clippy::inline_always,
    reason = "fearless_simd dispatch! requires inline(always) to inherit target_feature attributes."
)]
#[inline(always)]
fn fwht_simd_in_place<S: Simd>(simd: S, data: &mut [f32], norm: f32) {
    let n = data.len();
    let mut len = 1_usize;
    while len < n {
        let step = len * 2;
        if len >= 4 {
            for base in (0..n).step_by(step) {
                let mut i = 0_usize;
                while i + 4 <= len {
                    let a = f32x4::from_slice(simd, &data[base + i..base + i + 4]);
                    let b = f32x4::from_slice(simd, &data[base + i + len..base + i + len + 4]);
                    (a + b).store_slice(&mut data[base + i..base + i + 4]);
                    (a - b).store_slice(&mut data[base + i + len..base + i + len + 4]);
                    i += 4;
                }
            }
        } else {
            for base in (0..n).step_by(step) {
                for i in 0..len {
                    let a = data[base + i];
                    let b = data[base + i + len];
                    data[base + i] = a + b;
                    data[base + i + len] = a - b;
                }
            }
        }
        len = step;
    }
    let norm_v = f32x4::splat(simd, norm);
    let (chunks, tail) = data.as_chunks_mut::<4>();
    for chunk in chunks {
        let v = f32x4::from_slice(simd, chunk);
        (v * norm_v).store_slice(chunk);
    }
    for val in tail {
        *val *= norm;
    }
}

/// In-place normalized Sylvester Walsh–Hadamard transform over a power-of-two slice.
pub fn sylvester_walsh_hadamard_in_place(data: &mut [f32]) -> io::Result<()> {
    let n = data.len();
    if n == 0 || !n.is_power_of_two() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Hadamard block size must be a positive power of two.",
        ));
    }
    let norm = 1.0_f32 / usize_to_f32(n).sqrt();
    let level = Level::new();
    dispatch!(level, simd => fwht_simd_in_place(simd, data, norm));
    Ok(())
}

#[expect(
    clippy::inline_always,
    reason = "fearless_simd dispatch! requires inline(always) to inherit target_feature attributes."
)]
#[inline(always)]
fn sign_flip_simd<S: Simd>(simd: S, input: &[f32], signs: &[i8], out: &mut [f32]) {
    let (in_chunks, in_tail) = input.as_chunks::<4>();
    let (s_chunks, s_tail) = signs.as_chunks::<4>();
    let (out_chunks, out_tail) = out.as_chunks_mut::<4>();
    for ((x_c, s_c), dst_c) in in_chunks
        .iter()
        .zip(s_chunks.iter())
        .zip(out_chunks.iter_mut())
    {
        let xv = f32x4::from_slice(simd, x_c);
        let sv = f32x4::from_slice(
            simd,
            &[
                if s_c[0] < 0 { -1.0_f32 } else { 1.0_f32 },
                if s_c[1] < 0 { -1.0_f32 } else { 1.0_f32 },
                if s_c[2] < 0 { -1.0_f32 } else { 1.0_f32 },
                if s_c[3] < 0 { -1.0_f32 } else { 1.0_f32 },
            ],
        );
        (xv * sv).store_slice(dst_c);
    }
    for ((&x, &s), dst) in in_tail.iter().zip(s_tail.iter()).zip(out_tail.iter_mut()) {
        *dst = if s < 0 { -x } else { x };
    }
}

#[expect(
    clippy::inline_always,
    reason = "fearless_simd dispatch! requires inline(always) to inherit target_feature attributes."
)]
#[inline(always)]
fn q8_block_dot_simd<S: Simd>(simd: S, quants: &[i8; 32], x: &[f32]) -> f32 {
    let (q_chunks, _) = quants.as_chunks::<4>();
    let (x_chunks, _) = x.as_chunks::<4>();
    let mut acc0 = f32x4::splat(simd, 0.0_f32);
    let mut acc1 = f32x4::splat(simd, 0.0_f32);
    for pair in 0..4 {
        let q0 = &q_chunks[pair * 2];
        let q1 = &q_chunks[pair * 2 + 1];
        let x0 = f32x4::from_slice(simd, &x_chunks[pair * 2]);
        let x1 = f32x4::from_slice(simd, &x_chunks[pair * 2 + 1]);
        let qv0 = f32x4::from_slice(
            simd,
            &[
                f32::from(q0[0]),
                f32::from(q0[1]),
                f32::from(q0[2]),
                f32::from(q0[3]),
            ],
        );
        let qv1 = f32x4::from_slice(
            simd,
            &[
                f32::from(q1[0]),
                f32::from(q1[1]),
                f32::from(q1[2]),
                f32::from(q1[3]),
            ],
        );
        acc0 = qv0.mul_add(x0, acc0);
        acc1 = qv1.mul_add(x1, acc1);
    }
    let sum_v = acc0 + acc1;
    let mut lanes = [0.0_f32; 4];
    sum_v.store_slice(&mut lanes);
    (lanes[0] + lanes[1]) + (lanes[2] + lanes[3])
}

const PQ2_BYTE_TRITS_F32: [[f32; 4]; 256] = {
    let mut table = [[0.0_f32; 4]; 256];
    let mut b = 0_usize;
    while b < 256 {
        let mut sub = 0_usize;
        while sub < 4 {
            let code = (b >> (sub * 2)) & 0b11;
            table[b][sub] = match code {
                0 => -1.0_f32,
                2 => 1.0_f32,
                _ => 0.0_f32,
            };
            sub += 1;
        }
        b += 1;
    }
    table
};

const PTQ1_NIBBLE_TRITS_F32: [[f32; 4]; 256] = {
    let mut table = [[0.0_f32; 4]; 256];
    let mut idx = 0_usize;
    while idx < 256 {
        let nz = idx & 0x0F;
        let sg = (idx >> 4) & 0x0F;
        let mut bit_i = 0_usize;
        while bit_i < 4 {
            if (nz >> bit_i) & 1 != 0 {
                table[idx][bit_i] = if (sg >> bit_i) & 1 != 0 {
                    1.0_f32
                } else {
                    -1.0_f32
                };
            }
            bit_i += 1;
        }
        idx += 1;
    }
    table
};

#[expect(
    clippy::inline_always,
    reason = "fearless_simd dispatch! requires inline(always) to inherit target_feature attributes."
)]
#[inline(always)]
fn pq2_group_dot_simd<S: Simd>(simd: S, byte_slice: &[u8], x_group: &[f32]) -> f32 {
    let (x_quads, _) = x_group.as_chunks::<4>();
    let (b_pairs, b_rem) = byte_slice.as_chunks::<2>();
    let (x_pairs, x_rem) = x_quads.as_chunks::<2>();
    let mut acc0 = f32x4::splat(simd, 0.0_f32);
    let mut acc1 = f32x4::splat(simd, 0.0_f32);
    for (bp, xp) in b_pairs.iter().zip(x_pairs.iter()) {
        let w0 = f32x4::from_slice(simd, &PQ2_BYTE_TRITS_F32[usize::from(bp[0])]);
        let w1 = f32x4::from_slice(simd, &PQ2_BYTE_TRITS_F32[usize::from(bp[1])]);
        let x0 = f32x4::from_slice(simd, &xp[0]);
        let x1 = f32x4::from_slice(simd, &xp[1]);
        acc0 = w0.mul_add(x0, acc0);
        acc1 = w1.mul_add(x1, acc1);
    }
    for (&b, xq) in b_rem.iter().zip(x_rem.iter()) {
        let w = f32x4::from_slice(simd, &PQ2_BYTE_TRITS_F32[usize::from(b)]);
        let x = f32x4::from_slice(simd, xq);
        acc0 = w.mul_add(x, acc0);
    }
    let sum_v = acc0 + acc1;
    let mut lanes = [0.0_f32; 4];
    sum_v.store_slice(&mut lanes);
    (lanes[0] + lanes[1]) + (lanes[2] + lanes[3])
}

#[expect(
    clippy::inline_always,
    reason = "fearless_simd dispatch! requires inline(always) to inherit target_feature attributes."
)]
#[inline(always)]
fn ptq1_group_dot_simd<S: Simd>(simd: S, nz_slice: &[u8], sg_slice: &[u8], x_group: &[f32]) -> f32 {
    let (x_octs, _) = x_group.as_chunks::<8>();
    let mut acc0 = f32x4::splat(simd, 0.0_f32);
    let mut acc1 = f32x4::splat(simd, 0.0_f32);
    for ((&nz_b, &sg_b), xo) in nz_slice.iter().zip(sg_slice.iter()).zip(x_octs.iter()) {
        if nz_b == 0 {
            continue;
        }
        let lo_idx = usize::from((nz_b & 0x0F) | ((sg_b & 0x0F) << 4));
        let hi_idx = usize::from((nz_b >> 4) | (sg_b & 0xF0));
        let w0 = f32x4::from_slice(simd, &PTQ1_NIBBLE_TRITS_F32[lo_idx]);
        let w1 = f32x4::from_slice(simd, &PTQ1_NIBBLE_TRITS_F32[hi_idx]);
        let x0 = f32x4::from_slice(simd, &xo[0..4]);
        let x1 = f32x4::from_slice(simd, &xo[4..8]);
        acc0 = w0.mul_add(x0, acc0);
        acc1 = w1.mul_add(x1, acc1);
    }
    let sum_v = acc0 + acc1;
    let mut lanes = [0.0_f32; 4];
    sum_v.store_slice(&mut lanes);
    (lanes[0] + lanes[1]) + (lanes[2] + lanes[3])
}

/// Apply input-side Rademacher sign folding (`x[i] *= signs[i]`) followed by block-wise
/// normalized Sylvester Walsh–Hadamard transform (`prism.hadamard.*`).
pub fn apply_hadamard_sign_fold(
    input: &[f32],
    signs: &[i8],
    block_size: usize,
) -> io::Result<Vec<f32>> {
    projection_layout::fold_dimensions(input.len(), signs.len(), block_size)?;
    let mut folded = vec![0.0_f32; input.len()];
    let norm = 1.0_f32 / usize_to_f32(block_size).sqrt();
    let level = Level::new();
    dispatch!(level, simd => {
        sign_flip_simd(simd, input, signs, &mut folded);
        for block in folded.chunks_exact_mut(block_size) {
            fwht_simd_in_place(simd, block, norm);
        }
    });
    Ok(folded)
}

/// Optional rank-one or low-rank `LoRA` adapter patch for a single linear projection.
#[derive(Clone, Debug)]
pub struct LoraProjectionPatch {
    /// Input feature dimension.
    pub in_dim: usize,
    /// Output feature dimension.
    pub out_dim: usize,
    /// Adapter rank (`>= 1`).
    pub rank: usize,
    /// `LoRA` scaling `alpha / rank`.
    pub alpha_over_rank: f32,
    /// Runtime adapter strength (`0.0` guarantees bit-exact base identity).
    pub scale: f32,
    /// Down-projection `[rank * in_dim]`.
    pub lora_a: Vec<f32>,
    /// Up-projection `[out_dim * rank]`.
    pub lora_b: Vec<f32>,
}

impl LoraProjectionPatch {
    pub(crate) fn validate(&self, in_dim: usize, out_dim: usize) -> io::Result<()> {
        if in_dim != self.in_dim
            || out_dim != self.out_dim
            || self.lora_a.len() != self.rank * self.in_dim
            || self.lora_b.len() != self.out_dim * self.rank
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Invalid LoRA projection patch dimensions.",
            ));
        }
        Ok(())
    }

    /// Add `scale * alpha_over_rank * (lora_b @ (lora_a @ x))` onto `output` (`[out_dim]`).
    /// When `scale == 0.0`, returns immediately without touching `output` (0-ULP identity).
    pub fn apply(&self, input: &[f32], output: &mut [f32]) -> io::Result<()> {
        self.validate(input.len(), output.len())?;
        if self.scale == 0.0_f32 {
            return Ok(());
        }
        let eff_scale = self.scale * self.alpha_over_rank;
        let mut mid = vec![0.0_f32; self.rank];
        let level = Level::new();
        dispatch!(level, simd => {
            for (r, mid_slot) in mid.iter_mut().enumerate() {
                let a_row = &self.lora_a[r * self.in_dim..(r + 1) * self.in_dim];
                *mid_slot = f32_dot_simd(simd, a_row, input) * eff_scale;
            }
            for (o, out_slot) in output.iter_mut().enumerate() {
                let b_row = &self.lora_b[o * self.rank..(o + 1) * self.rank];
                *out_slot += f32_dot_simd(simd, b_row, &mid);
            }
        });
        Ok(())
    }
}

/// Named collection of `LoRA` projection overlays attached to a shared `Base` `Talker`
/// (e.g., regional-accent or language adapters such as `aguken-ai/qwen3-tts-indic-finetuned`).
#[derive(Clone, Debug, Default)]
pub struct TalkerLoraAdapter {
    /// Adapter profile identifier.
    pub name: String,
    /// Map from GGUF tensor name (e.g. `"talker.blk.0.attn_q.weight"`) to its `LoRA` patch.
    pub patches: HashMap<String, LoraProjectionPatch>,
}

/// Matrix-vector / matrix-matrix linear projection supporting `F32`, `Q8_0`, `PQ2_0`, `PTQ1_0`,
/// and optional `LoraProjectionPatch` overlays.
#[derive(Clone, Debug)]
pub struct LinearProjection {
    /// Tensor name in the GGUF artifact.
    pub name: String,
    /// Input feature dimension.
    pub in_dim: usize,
    /// Output feature dimension.
    pub out_dim: usize,
    /// Quantized or full-precision weight storage.
    pub weight: WeightQuantization,
    /// Optional bias `[out_dim]`.
    pub bias: Option<Vec<f32>>,
}

impl LinearProjection {
    /// Construct a validated linear projection, checking the speech ternary policy.
    pub fn new(
        name: impl Into<String>,
        in_dim: usize,
        out_dim: usize,
        weight: WeightQuantization,
        bias: Option<Vec<f32>>,
    ) -> io::Result<Self> {
        Self::new_with_threads(name, in_dim, out_dim, weight, bias, 1)
    }

    /// Construct a validated linear projection using `validate_threads` for packed validation.
    pub(crate) fn new_with_threads(
        name: impl Into<String>,
        in_dim: usize,
        out_dim: usize,
        weight: WeightQuantization,
        bias: Option<Vec<f32>>,
        validate_threads: usize,
    ) -> io::Result<Self> {
        let name = name.into();
        validate_talker_ternary_policy(&name, &weight)?;
        if bias.as_ref().is_some_and(|b| b.len() != out_dim) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Invalid LinearProjection dimensions.",
            ));
        }
        projection_layout::validate(in_dim, out_dim, &weight)?;
        if let Some((ggml_type, bytes)) = weight.packed_slice() {
            crate::weights::validate_packed_parallel(ggml_type, bytes, validate_threads)?;
        }
        Ok(Self {
            name,
            in_dim,
            out_dim,
            weight,
            bias,
        })
    }

    /// Decode one embedding row without expanding the complete matrix.
    ///
    /// # Errors
    /// Rejects an out-of-range row, malformed weights or a transformed ternary
    /// projection, which cannot represent an ordinary embedding lookup.
    pub fn row(&self, index: usize) -> io::Result<Vec<f32>> {
        let mut output = vec![0.0_f32; self.in_dim];
        self.row_into(index, &mut output)?;
        Ok(output)
    }

    fn decode_row_into<const ADD: bool>(&self, index: usize, dst: &mut [f32]) -> io::Result<()> {
        if index >= self.out_dim || dst.len() != self.in_dim {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                if ADD {
                    "Embedding row or accumulator extent is out of range."
                } else {
                    "Embedding row or destination extent is out of range."
                },
            ));
        }
        let invalid = || io::Error::new(io::ErrorKind::InvalidData, "Invalid embedding weights.");
        let offset = index.checked_mul(self.in_dim).ok_or_else(invalid)?;
        let end = offset.checked_add(self.in_dim).ok_or_else(invalid)?;
        if let Some((ggml_type, bytes)) = self.weight.packed_slice() {
            let (block, size) = crate::weights::packed_layout(ggml_type)?;
            if !self.in_dim.is_multiple_of(block) || (ADD && block > 256) {
                return Err(invalid());
            }
            let start = (offset / block).checked_mul(size).ok_or_else(invalid)?;
            let stop = (end / block).checked_mul(size).ok_or_else(invalid)?;
            let row = bytes.get(start..stop).ok_or_else(invalid)?;
            let mut decoded = [0.0_f32; 256];
            for (source, destination) in row.chunks_exact(size).zip(dst.chunks_exact_mut(block)) {
                if ADD {
                    crate::weights::decode_block(ggml_type, source, &mut decoded[..block]);
                    for (acc, &add) in destination.iter_mut().zip(&decoded[..block]) {
                        *acc += add;
                    }
                } else {
                    crate::weights::decode_block(ggml_type, source, destination);
                }
            }
            return Ok(());
        }
        match &self.weight {
            WeightQuantization::F32(values) => {
                let slice = values.get(offset..end).ok_or_else(invalid)?;
                if ADD {
                    for (acc, &add) in dst.iter_mut().zip(slice) {
                        *acc += add;
                    }
                } else {
                    dst.copy_from_slice(slice);
                }
                Ok(())
            }
            WeightQuantization::Q8_0 { blocks } => {
                if !self.in_dim.is_multiple_of(32) {
                    return Err(invalid());
                }
                let row = blocks.get(offset / 32..end / 32).ok_or_else(invalid)?;
                for (block, chunk) in row.iter().zip(dst.chunks_exact_mut(32)) {
                    for (value, slot) in block.quants.iter().zip(chunk.iter_mut()) {
                        let weighted = block.scale * f32::from(*value);
                        if ADD {
                            *slot += weighted;
                        } else {
                            *slot = weighted;
                        }
                    }
                }
                Ok(())
            }
            _ => Err(invalid()),
        }
    }

    /// Decode embedding row `index` directly into pre-allocated `out` (`[in_dim]`).
    ///
    /// # Errors
    /// Returns `InvalidInput` when `index >= out_dim` or `out.len() != in_dim`, and
    /// `InvalidData` when the weight payload cannot supply that row cleanly.
    pub fn row_into(&self, index: usize, out: &mut [f32]) -> io::Result<()> {
        self.decode_row_into::<false>(index, out)
    }

    /// Decode embedding row `index` and add it elementwise into `accum` (`[in_dim]`).
    ///
    /// # Errors
    /// Returns `InvalidInput` when `index >= out_dim` or `accum.len() != in_dim`, and
    /// `InvalidData` when the weight payload cannot supply that row cleanly.
    pub fn add_row_into(&self, index: usize, accum: &mut [f32]) -> io::Result<()> {
        self.decode_row_into::<true>(index, accum)
    }

    /// Return the input-side `(signs, hadamard_block)` signature when this projection uses
    /// `PQ2_0` or `PTQ1_0` block Walsh–Hadamard folding.
    #[must_use]
    pub const fn hadamard_signature(&self) -> Option<(&[i8], usize)> {
        match &self.weight {
            WeightQuantization::PQ2_0 {
                hadamard_block,
                signs,
                ..
            }
            | WeightQuantization::PTQ1_0 {
                hadamard_block,
                signs,
                ..
            } => Some((signs.as_slice(), *hadamard_block)),
            WeightQuantization::F32(_)
            | WeightQuantization::Q8_0 { .. }
            | WeightQuantization::Packed { .. } => None,
            WeightQuantization::MappedPacked { .. } | WeightQuantization::RepackedQ4K8 { .. } => {
                None
            }
        }
    }

    /// Project one vector `input` (`[in_dim]`) into `[out_dim]`, optionally applying `lora`.
    pub fn forward_vec(
        &self,
        input: &[f32],
        lora: Option<&LoraProjectionPatch>,
    ) -> io::Result<Vec<f32>> {
        self.forward_vec_parallel(input, lora, 1)
    }

    /// Project one vector `input` (`[in_dim]`) into `[out_dim]` using `fearless_simd` NEON
    /// and up to `threads` reusable row workers.
    pub fn forward_vec_parallel(
        &self,
        input: &[f32],
        lora: Option<&LoraProjectionPatch>,
        threads: usize,
    ) -> io::Result<Vec<f32>> {
        self.forward_vec_with_pre_folded(input, None, lora, threads)
    }

    /// Project one vector `input` (`[in_dim]`) into `[out_dim]`, optionally reusing a pre-folded
    /// Hadamard activation slice (`pre_folded`) shared across sibling projections (`q/k/v` or `gate/up`).
    pub fn forward_vec_with_pre_folded(
        &self,
        input: &[f32],
        pre_folded: Option<&[f32]>,
        lora: Option<&LoraProjectionPatch>,
        threads: usize,
    ) -> io::Result<Vec<f32>> {
        let mut out = Vec::new();
        self.forward_vec_into(input, pre_folded, lora, threads, &mut out)?;
        Ok(out)
    }

    /// Overwrite request-owned projection storage without retaining a logical output.
    #[expect(
        clippy::too_many_lines,
        reason = "Dispatches across F32, Q8_0, PQ2_0, and PTQ1_0 weight representations with optional reusable row parallelism."
    )]
    pub(crate) fn forward_vec_into(
        &self,
        input: &[f32],
        pre_folded: Option<&[f32]>,
        lora: Option<&LoraProjectionPatch>,
        threads: usize,
        out: &mut Vec<f32>,
    ) -> io::Result<()> {
        if input.len() != self.in_dim
            || self
                .bias
                .as_ref()
                .is_some_and(|bias| bias.len() != self.out_dim)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "LinearProjection input or bias length mismatch.",
            ));
        }
        let layout = projection_layout::validate(self.in_dim, self.out_dim, &self.weight)?;
        decoder_workspace::resize(out, self.out_dim)?;
        if let Some(bias) = &self.bias {
            out.copy_from_slice(bias);
        } else {
            out.fill(0.0);
        }
        let worker_count = threads.clamp(1, 18).min(self.out_dim.max(1));
        let level = Level::new();

        if let Some((ggml_type, bytes)) = self.weight.packed_slice() {
            if dispatch_packed_quant_kernels(
                ggml_type,
                bytes,
                self.in_dim,
                input,
                worker_count,
                out,
            ) {
                if let Some(patch) = lora {
                    patch.apply(input, out)?;
                }
                return Ok(());
            }
            let row_bytes = layout.bytes_per_row;
            let compute_slice = |row_offset: usize, destination: &mut [f32]| {
                dispatch!(level, simd => {
                    let offset = row_offset * row_bytes;
                    packed::rows(simd, ggml_type, &bytes[offset..], input, destination);
                });
            };
            run_row_chunks(out, worker_count, compute_slice);
        } else {
            match &self.weight {
                WeightQuantization::Packed { .. } => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Unreachable packed weight branch.",
                    ));
                }
                WeightQuantization::MappedPacked { .. } => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Unreachable mapped packed weight branch.",
                    ));
                }
                WeightQuantization::RepackedQ4K8 { bytes } => {
                    if !dispatch_repacked_q4_k(bytes, self.in_dim, input, worker_count, out) {
                        return Err(io::Error::new(
                            io::ErrorKind::Unsupported,
                            "Repacked Q4_K weights require ARM64 dotprod support.",
                        ));
                    }
                }
                WeightQuantization::F32(w) => {
                    let in_dim = self.in_dim;
                    let compute_slice = |row_offset: usize, dst_slice: &mut [f32]| {
                        dispatch!(level, simd => {
                            for (local_o, dst) in dst_slice.iter_mut().enumerate() {
                                let o = row_offset + local_o;
                                let row = &w[o * in_dim..(o + 1) * in_dim];
                                *dst += f32_dot_simd(simd, row, input);
                            }
                        });
                    };
                    run_row_chunks(out, worker_count, compute_slice);
                }
                WeightQuantization::Q8_0 { blocks } => {
                    let blocks_per_row = layout.groups_per_row;
                    let compute_slice = |row_offset: usize, dst_slice: &mut [f32]| {
                        dispatch!(level, simd => {
                            for (local_o, dst) in dst_slice.iter_mut().enumerate() {
                                let o = row_offset + local_o;
                                let row_blocks = &blocks[o * blocks_per_row..(o + 1) * blocks_per_row];
                                let mut acc = *dst;
                                for (bi, blk) in row_blocks.iter().enumerate() {
                                    let x_slice = &input[bi * 32..(bi + 1) * 32];
                                    let b_sum = q8_block_dot_simd(simd, &blk.quants, x_slice);
                                    acc = blk.scale.mul_add(b_sum, acc);
                                }
                                *dst = acc;
                            }
                        });
                    };
                    run_row_chunks(out, worker_count, compute_slice);
                }
                WeightQuantization::PQ2_0 {
                    group_size,
                    scales,
                    packed_trits,
                    hadamard_block,
                    signs,
                } => {
                    let x_folded =
                        projection_layout::folded(input, pre_folded, signs, *hadamard_block)?;
                    let groups_per_row = layout.groups_per_row;
                    let bytes_per_row = layout.bytes_per_row;
                    let gsize = *group_size;
                    let gbytes = gsize / 4;
                    let compute_slice = |row_offset: usize, dst_slice: &mut [f32]| {
                        dispatch!(level, simd => {
                            for (local_o, dst) in dst_slice.iter_mut().enumerate() {
                                let o = row_offset + local_o;
                                let row_scales = &scales[o * groups_per_row..(o + 1) * groups_per_row];
                                let row_bytes = &packed_trits[o * bytes_per_row..(o + 1) * bytes_per_row];
                                let mut acc = *dst;
                                for (gi, &scale) in row_scales.iter().enumerate() {
                                    let start_col = gi * gsize;
                                    let byte_slice = &row_bytes[gi * gbytes..(gi + 1) * gbytes];
                                    let x_group = &x_folded[start_col..start_col + gsize];
                                    let g_sum = pq2_group_dot_simd(simd, byte_slice, x_group);
                                    acc = scale.mul_add(g_sum, acc);
                                }
                                *dst = acc;
                            }
                        });
                    };
                    run_row_chunks(out, worker_count, compute_slice);
                }
                WeightQuantization::PTQ1_0 {
                    group_size,
                    scales,
                    nz_mask,
                    sign_mask,
                    hadamard_block,
                    signs,
                } => {
                    let x_folded =
                        projection_layout::folded(input, pre_folded, signs, *hadamard_block)?;
                    let groups_per_row = layout.groups_per_row;
                    let bytes_per_row = layout.bytes_per_row;
                    let gsize = *group_size;
                    let gbytes = gsize / 8;
                    let compute_slice = |row_offset: usize, dst_slice: &mut [f32]| {
                        dispatch!(level, simd => {
                            for (local_o, dst) in dst_slice.iter_mut().enumerate() {
                                let o = row_offset + local_o;
                                let row_scales = &scales[o * groups_per_row..(o + 1) * groups_per_row];
                                let row_nz = &nz_mask[o * bytes_per_row..(o + 1) * bytes_per_row];
                                let row_sg = &sign_mask[o * bytes_per_row..(o + 1) * bytes_per_row];
                                let mut acc = *dst;
                                for (gi, &scale) in row_scales.iter().enumerate() {
                                    let start_col = gi * gsize;
                                    let nz_slice = &row_nz[gi * gbytes..(gi + 1) * gbytes];
                                    let sg_slice = &row_sg[gi * gbytes..(gi + 1) * gbytes];
                                    let x_group = &x_folded[start_col..start_col + gsize];
                                    let g_sum = ptq1_group_dot_simd(simd, nz_slice, sg_slice, x_group);
                                    acc = scale.mul_add(g_sum, acc);
                                }
                                *dst = acc;
                            }
                        });
                    };
                    run_row_chunks(out, worker_count, compute_slice);
                }
            }
        }

        if let Some(patch) = lora {
            patch.apply(input, out)?;
        }
        Ok(())
    }

    /// Convert eligible `Q4_K` projections (`out_dim % 8 == 0`, `in_dim % 256 == 0`) to
    /// interleaved 8-row layout tuned for ARM64 `SDOT` / `I8MM` kernels.
    pub fn repack_q4_k_if_aligned(&mut self) {
        if !bonsai_windows_process::quant_kernels::Arm64QuantCaps::detect().dotprod
            || !self.out_dim.is_multiple_of(8)
            || !self.in_dim.is_multiple_of(256)
        {
            return;
        }
        let Some((12, bytes)) = self.weight.packed_slice() else {
            return;
        };
        if let Some(repacked) = bonsai_windows_process::quant_kernels::repack_q4_k_8rows(
            bytes,
            self.in_dim,
            self.out_dim,
        ) {
            self.weight = WeightQuantization::RepackedQ4K8 { bytes: repacked };
        }
    }

    /// Fuse two sibling projections sharing `input` (e.g. `gate + up`) so activation
    /// quantization and worker barrier orchestration execute once.
    #[expect(
        clippy::too_many_arguments,
        reason = "Preserves per-projection optional LoRA patches and distinct pre-allocated output buffers."
    )]
    pub fn forward_pair_into(
        proj_a: &Self,
        proj_b: &Self,
        input: &[f32],
        lora_a: Option<&LoraProjectionPatch>,
        lora_b: Option<&LoraProjectionPatch>,
        threads: usize,
        out_a: &mut Vec<f32>,
        out_b: &mut Vec<f32>,
    ) -> io::Result<()> {
        let mut targets = [(proj_a, lora_a, out_a), (proj_b, lora_b, out_b)];
        forward_group_into(&mut targets, input, threads)
    }

    /// Fuse three sibling projections sharing `input` (`attn_q + attn_k + attn_v`) so activation
    /// quantization and worker barrier orchestration execute once.
    #[expect(
        clippy::too_many_arguments,
        reason = "Preserves per-projection optional LoRA patches and distinct pre-allocated output buffers."
    )]
    pub fn forward_triple_into(
        proj_a: &Self,
        proj_b: &Self,
        proj_c: &Self,
        input: &[f32],
        lora_a: Option<&LoraProjectionPatch>,
        lora_b: Option<&LoraProjectionPatch>,
        lora_c: Option<&LoraProjectionPatch>,
        threads: usize,
        out_a: &mut Vec<f32>,
        out_b: &mut Vec<f32>,
        out_c: &mut Vec<f32>,
    ) -> io::Result<()> {
        let mut targets = [
            (proj_a, lora_a, out_a),
            (proj_b, lora_b, out_b),
            (proj_c, lora_c, out_c),
        ];
        forward_group_into(&mut targets, input, threads)
    }

    /// Fuse multiple sibling batched projections sharing `inputs` (`[seq_len * in_dim]`) into
    /// `[seq_len * out_dim]` per target so activation quantization and worker orchestration run once.
    pub fn forward_batch_group_into(
        targets: &mut [(&Self, Option<&LoraProjectionPatch>, &mut Vec<f32>)],
        inputs: &[f32],
        seq_len: usize,
        threads: usize,
    ) -> io::Result<()> {
        if !(1..=256).contains(&threads) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Invalid CPU worker count.",
            ));
        }
        if targets.is_empty() {
            return Ok(());
        }
        for (proj, lora, _) in targets.iter() {
            let _layout = projection_layout::validate(proj.in_dim, proj.out_dim, &proj.weight)?;
            let expected_in = seq_len.checked_mul(proj.in_dim).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "Batch input length overflow.")
            })?;
            if inputs.len() != expected_in {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "Batch projection input length mismatch.",
                ));
            }
            if seq_len.checked_mul(proj.out_dim).is_none() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "Batch output length overflow.",
                ));
            }
            if let Some(bias) = &proj.bias
                && bias.len() != proj.out_dim
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Projection bias length mismatch.",
                ));
            }
            if let Some(patch) = *lora {
                patch.validate(proj.in_dim, proj.out_dim)?;
            }
        }
        if seq_len == 0 {
            for (_, _, out) in targets.iter_mut() {
                out.clear();
            }
            return Ok(());
        }
        if seq_len == 1 {
            return forward_group_into(targets, inputs, threads);
        }
        for (proj, _, out) in targets.iter_mut() {
            out.resize(seq_len * proj.out_dim, 0.0);
        }
        if quant_dispatch::dispatch_batch_group_quant_kernels(targets, inputs, seq_len, threads) {
            for (proj, lora, out) in targets.iter_mut() {
                if let Some(patch) = *lora {
                    for t in 0..seq_len {
                        patch.apply(
                            &inputs[t * proj.in_dim..(t + 1) * proj.in_dim],
                            &mut out[t * proj.out_dim..(t + 1) * proj.out_dim],
                        )?;
                    }
                }
            }
            return Ok(());
        }
        for (proj, lora, out) in targets.iter_mut() {
            proj.forward_batch_into(inputs, seq_len, *lora, threads, out)?;
        }
        Ok(())
    }

    /// Project a sequence of `seq_len` vectors (`[seq_len * in_dim]`) into `out`
    /// (`[seq_len * out_dim]`), using `I8MM` (`SMMLA`) `4x2`/`8x2` tiles on ARM64 Windows.
    pub fn forward_batch_into(
        &self,
        inputs: &[f32],
        seq_len: usize,
        lora: Option<&LoraProjectionPatch>,
        threads: usize,
        out: &mut Vec<f32>,
    ) -> io::Result<()> {
        if !(1..=256).contains(&threads) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Invalid CPU worker count.",
            ));
        }
        let expected_in = seq_len.checked_mul(self.in_dim).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "Batch input length overflow.")
        })?;
        if inputs.len() != expected_in {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Batch projection input length mismatch.",
            ));
        }
        if seq_len == 0 {
            out.clear();
            return Ok(());
        }
        if seq_len == 1 {
            return self.forward_vec_into(inputs, None, lora, threads, out);
        }
        let _layout = projection_layout::validate(self.in_dim, self.out_dim, &self.weight)?;
        let expected_out = seq_len.checked_mul(self.out_dim).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "Batch output length overflow.")
        })?;
        if let Some(bias) = &self.bias
            && bias.len() != self.out_dim
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Projection bias length mismatch.",
            ));
        }
        if let Some(patch) = lora {
            patch.validate(self.in_dim, self.out_dim)?;
        }
        out.resize(expected_out, 0.0);

        if dispatch_batch_quant_kernels(self, inputs, seq_len, threads, out)? {
            if let Some(patch) = lora {
                for t in 0..seq_len {
                    patch.apply(
                        &inputs[t * self.in_dim..(t + 1) * self.in_dim],
                        &mut out[t * self.out_dim..(t + 1) * self.out_dim],
                    )?;
                }
            }
            return Ok(());
        }

        let mut row_scratch = Vec::with_capacity(self.out_dim);
        for t in 0..seq_len {
            self.forward_vec_into(
                &inputs[t * self.in_dim..(t + 1) * self.in_dim],
                None,
                lora,
                threads,
                &mut row_scratch,
            )?;
            out[t * self.out_dim..(t + 1) * self.out_dim].copy_from_slice(&row_scratch);
        }
        Ok(())
    }
}

mod quant_dispatch {
    use std::cell::RefCell;

    use bonsai_windows_process::quant_kernels::{
        Arm64QuantCaps, Q4_K_BLOCK_BYTES, Q4_K_REPACK8_BLOCK_BYTES, Q6_K_BLOCK_BYTES,
        Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMENTS, Q8_0ActBlock, Q8_KActBlock, QK_K, dot_q4_k_4rows,
        dot_q4_k_repacked8, dot_q4_k_row, dot_q6_k_4rows, dot_q6_k_8rows, dot_q6_k_row,
        dot_q8_0_4rows, dot_q8_0_8rows, dot_q8_0_row, gemm_q4_k_4x2, gemm_q4_k_repacked8_8x2,
        gemm_q4_k_repacked8_8x4, gemm_q4_k_repacked8_8x8, gemm_q6_k_4x2, gemm_q6_k_8x2,
        gemm_q6_k_8x4, gemm_q6_k_8x8, gemm_q8_0_4x2, gemm_q8_0_8x2, gemm_q8_0_8x4, gemm_q8_0_8x8,
        quantize_row_q8_0, quantize_row_q8_k,
    };

    use super::{LinearProjection, WeightQuantization, run_row_chunks_aligned};

    thread_local! {
        static ACT_Q8_0: RefCell<Vec<Q8_0ActBlock>> = const { RefCell::new(Vec::new()) };
        static ACT_Q8_K: RefCell<Vec<Q8_KActBlock>> = const { RefCell::new(Vec::new()) };
        static GROUP_COMBINED: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    }

    pub(super) fn with_q8_0_act<R>(input: &[f32], action: impl FnOnce(&[Q8_0ActBlock]) -> R) -> R {
        let blocks = input.len() / Q8_0_BLOCK_ELEMENTS;
        ACT_Q8_0.with(|cell| {
            if let Ok(mut buf) = cell.try_borrow_mut() {
                buf.resize(blocks, Q8_0ActBlock::default());
                quantize_row_q8_0(input, &mut buf[..blocks]);
                action(&buf[..blocks])
            } else {
                let mut fallback = vec![Q8_0ActBlock::default(); blocks];
                quantize_row_q8_0(input, &mut fallback);
                action(&fallback)
            }
        })
    }

    pub(super) fn with_q8_k_act<R>(input: &[f32], action: impl FnOnce(&[Q8_KActBlock]) -> R) -> R {
        let blocks = input.len() / QK_K;
        ACT_Q8_K.with(|cell| {
            if let Ok(mut buf) = cell.try_borrow_mut() {
                buf.resize(blocks, Q8_KActBlock::default());
                quantize_row_q8_k(input, &mut buf[..blocks]);
                action(&buf[..blocks])
            } else {
                let mut fallback = vec![Q8_KActBlock::default(); blocks];
                quantize_row_q8_k(input, &mut fallback);
                action(&fallback)
            }
        })
    }

    pub(super) fn with_group_workspace<R>(
        total_out: usize,
        need_q8: bool,
        need_qk: bool,
        input: &[f32],
        action: impl FnOnce(&mut [f32], &[Q8_0ActBlock], &[Q8_KActBlock]) -> R,
    ) -> R {
        let q8_blocks = if need_q8 {
            input.len() / Q8_0_BLOCK_ELEMENTS
        } else {
            0
        };
        let qk_blocks = if need_qk { input.len() / QK_K } else { 0 };
        GROUP_COMBINED.with(|comb_cell| {
            let mut comb_fallback = Vec::new();
            let mut comb_guard = comb_cell.try_borrow_mut().ok();
            let combined = match comb_guard.as_deref_mut() {
                Some(buf) => {
                    buf.resize(total_out, 0.0);
                    &mut buf[..total_out]
                }
                None => {
                    comb_fallback.resize(total_out, 0.0);
                    &mut comb_fallback[..total_out]
                }
            };
            ACT_Q8_0.with(|q8_cell| {
                ACT_Q8_K.with(|qk_cell| {
                    let mut q8_fallback = Vec::new();
                    let mut q8_guard = need_q8.then(|| q8_cell.try_borrow_mut().ok()).flatten();
                    let q8_slice: &[Q8_0ActBlock] = if need_q8 {
                        match q8_guard.as_deref_mut() {
                            Some(buf) => {
                                buf.resize(q8_blocks, Q8_0ActBlock::default());
                                quantize_row_q8_0(input, &mut buf[..q8_blocks]);
                                &buf[..q8_blocks]
                            }
                            None => {
                                q8_fallback.resize(q8_blocks, Q8_0ActBlock::default());
                                quantize_row_q8_0(input, &mut q8_fallback);
                                &q8_fallback[..q8_blocks]
                            }
                        }
                    } else {
                        &[]
                    };
                    let mut qk_fallback = Vec::new();
                    let mut qk_guard = need_qk.then(|| qk_cell.try_borrow_mut().ok()).flatten();
                    let qk_slice: &[Q8_KActBlock] = if need_qk {
                        match qk_guard.as_deref_mut() {
                            Some(buf) => {
                                buf.resize(qk_blocks, Q8_KActBlock::default());
                                quantize_row_q8_k(input, &mut buf[..qk_blocks]);
                                &buf[..qk_blocks]
                            }
                            None => {
                                qk_fallback.resize(qk_blocks, Q8_KActBlock::default());
                                quantize_row_q8_k(input, &mut qk_fallback);
                                &qk_fallback[..qk_blocks]
                            }
                        }
                    } else {
                        &[]
                    };
                    action(combined, q8_slice, qk_slice)
                })
            })
        })
    }

    pub(super) fn add_packed_rows(
        ggml_type: u32,
        bytes: &[u8],
        in_dim: usize,
        caps: Arm64QuantCaps,
        q8_act: &[Q8_0ActBlock],
        qk_act: &[Q8_KActBlock],
        start_row: usize,
        dst: &mut [f32],
    ) {
        let (block, elem_bytes) = match ggml_type {
            8 => (Q8_0_BLOCK_ELEMENTS, Q8_0_BLOCK_BYTES),
            12 => (QK_K, Q4_K_BLOCK_BYTES),
            14 => (QK_K, Q6_K_BLOCK_BYTES),
            _ => return,
        };
        let row_bytes = (in_dim / block) * elem_bytes;
        let mut processed = 0_usize;
        if ggml_type == 8 || ggml_type == 14 {
            for oct in dst.chunks_exact_mut(8) {
                let r0 = (start_row + processed) * row_bytes;
                let row_refs: [&[u8]; 8] = std::array::from_fn(|r| {
                    let start = r0 + r * row_bytes;
                    &bytes[start..start + row_bytes]
                });
                let sums = if ggml_type == 8 {
                    dot_q8_0_8rows(caps, row_refs, q8_act)
                } else {
                    dot_q6_k_8rows(caps, row_refs, qk_act)
                };
                for (slot, add) in oct.iter_mut().zip(sums) {
                    *slot += add;
                }
                processed += 8;
            }
        }
        for quad in dst[processed..].chunks_exact_mut(4) {
            let r0 = (start_row + processed) * row_bytes;
            let row_refs = [
                &bytes[r0..r0 + row_bytes],
                &bytes[r0 + row_bytes..r0 + 2 * row_bytes],
                &bytes[r0 + 2 * row_bytes..r0 + 3 * row_bytes],
                &bytes[r0 + 3 * row_bytes..r0 + 4 * row_bytes],
            ];
            let sums = match ggml_type {
                8 => dot_q8_0_4rows(caps, row_refs, q8_act),
                12 => dot_q4_k_4rows(caps, row_refs, qk_act),
                _ => dot_q6_k_4rows(caps, row_refs, qk_act),
            };
            for (slot, add) in quad.iter_mut().zip(sums) {
                *slot += add;
            }
            processed += 4;
        }
        for (local, slot) in dst[processed..].iter_mut().enumerate() {
            let off = (start_row + processed + local) * row_bytes;
            let row = &bytes[off..off + row_bytes];
            *slot += match ggml_type {
                8 => dot_q8_0_row(caps, row, q8_act),
                12 => dot_q4_k_row(caps, row, qk_act),
                _ => dot_q6_k_row(caps, row, qk_act),
            };
        }
    }

    pub(super) fn dispatch_packed_quant_kernels(
        ggml_type: u32,
        bytes: &[u8],
        in_dim: usize,
        input: &[f32],
        worker_count: usize,
        out: &mut [f32],
    ) -> bool {
        if !matches!(ggml_type, 8 | 12 | 14) {
            return false;
        }
        let caps = Arm64QuantCaps::detect();
        if !caps.dotprod {
            return false;
        }
        if ggml_type == 8 {
            with_q8_0_act(input, |act| {
                run_row_chunks_aligned(out, worker_count, 8, |start_row, dst| {
                    add_packed_rows(8, bytes, in_dim, caps, act, &[], start_row, dst);
                });
            });
        } else {
            let align = if ggml_type == 14 { 8 } else { 4 };
            with_q8_k_act(input, |act| {
                run_row_chunks_aligned(out, worker_count, align, |start_row, dst| {
                    add_packed_rows(ggml_type, bytes, in_dim, caps, &[], act, start_row, dst);
                });
            });
        }
        true
    }

    pub(super) fn add_repacked_rows(
        bytes: &[u8],
        in_dim: usize,
        caps: Arm64QuantCaps,
        qk_act: &[Q8_KActBlock],
        start_row: usize,
        dst: &mut [f32],
    ) {
        let strip_bytes = (in_dim / QK_K) * Q4_K_REPACK8_BLOCK_BYTES;
        let mut idx = 0_usize;
        while idx < dst.len() {
            let abs_row = start_row + idx;
            let strip_idx = abs_row / 8;
            let lane = abs_row % 8;
            let strip = &bytes[strip_idx * strip_bytes..(strip_idx + 1) * strip_bytes];
            let vals = dot_q4_k_repacked8(caps, strip, qk_act);
            let take = (8 - lane).min(dst.len() - idx);
            for k in 0..take {
                dst[idx + k] += vals[lane + k];
            }
            idx += take;
        }
    }

    pub(super) fn dispatch_repacked_q4_k(
        bytes: &[u8],
        in_dim: usize,
        input: &[f32],
        worker_count: usize,
        out: &mut [f32],
    ) -> bool {
        let caps = Arm64QuantCaps::detect();
        if !caps.dotprod {
            return false;
        }
        with_q8_k_act(input, |act| {
            run_row_chunks_aligned(out, worker_count, 8, |start_row, dst| {
                add_repacked_rows(bytes, in_dim, caps, act, start_row, dst);
            });
        });
        true
    }

    #[derive(Clone, Copy)]
    pub(super) enum GroupKind<'a> {
        Packed { ggml_type: u32, bytes: &'a [u8] },
        Repacked(&'a [u8]),
    }

    pub(super) fn classify_group_proj(proj: &LinearProjection) -> Option<GroupKind<'_>> {
        if let Some((ggml_type, bytes)) = proj.weight.packed_slice()
            && matches!(ggml_type, 8 | 12 | 14)
        {
            return Some(GroupKind::Packed { ggml_type, bytes });
        }
        if let WeightQuantization::RepackedQ4K8 { bytes } = &proj.weight {
            return Some(GroupKind::Repacked(bytes));
        }
        None
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "Generic 4-row GEMM slice kernel parameterizes block geometry and activation kernels without dynamic dispatch."
    )]
    fn add_batch_4row_gemm<Act>(
        dst: &mut [f32],
        bytes: &[u8],
        in_dim: usize,
        seq_len: usize,
        local_row_start: usize,
        rows_here: usize,
        block_elems: usize,
        elem_bytes: usize,
        acts: &[Act],
        gemm_4x2: impl Fn([&[u8]; 4], &[Act], &[Act]) -> [[f32; 2]; 4],
        dot_4rows: impl Fn([&[u8]; 4], &[Act]) -> [f32; 4],
        dot_row: impl Fn(&[u8], &[Act]) -> f32,
    ) {
        let blocks_per_tok = in_dim / block_elems;
        let row_bytes = blocks_per_tok * elem_bytes;
        let mut processed = 0_usize;
        while processed + 4 <= rows_here {
            let r0 = (local_row_start + processed) * row_bytes;
            let row_refs = std::array::from_fn(|r| {
                let start = r0 + r * row_bytes;
                &bytes[start..start + row_bytes]
            });
            let mut t = 0_usize;
            while t + 1 < seq_len {
                let a0 = &acts[t * blocks_per_tok..(t + 1) * blocks_per_tok];
                let a1 = &acts[(t + 1) * blocks_per_tok..(t + 2) * blocks_per_tok];
                let tile = gemm_4x2(row_refs, a0, a1);
                for r in 0..4 {
                    let base = (processed + r) * seq_len + t;
                    dst[base] += tile[r][0];
                    dst[base + 1] += tile[r][1];
                }
                t += 2;
            }
            if t < seq_len {
                let a0 = &acts[t * blocks_per_tok..(t + 1) * blocks_per_tok];
                let col = dot_4rows(row_refs, a0);
                for r in 0..4 {
                    dst[(processed + r) * seq_len + t] += col[r];
                }
            }
            processed += 4;
        }
        for local_r in processed..rows_here {
            let off = (local_row_start + local_r) * row_bytes;
            let w_row = &bytes[off..off + row_bytes];
            let base = local_r * seq_len;
            for t in 0..seq_len {
                let a0 = &acts[t * blocks_per_tok..(t + 1) * blocks_per_tok];
                dst[base + t] += dot_row(w_row, a0);
            }
        }
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "Executes repacked 8-row Q4_K batched GEMM over an arbitrary row sub-range."
    )]
    fn add_batch_repacked_gemm(
        dst: &mut [f32],
        bytes: &[u8],
        in_dim: usize,
        seq_len: usize,
        local_row_start: usize,
        rows_here: usize,
        caps: Arm64QuantCaps,
        acts: &[Q8_KActBlock],
    ) {
        let blocks_per_tok = in_dim / QK_K;
        let strip_bytes = blocks_per_tok * Q4_K_REPACK8_BLOCK_BYTES;
        let mut idx = 0_usize;
        while idx < rows_here {
            let abs_row = local_row_start + idx;
            let strip_idx = abs_row / 8;
            let lane = abs_row % 8;
            let take = (8 - lane).min(rows_here - idx);
            let strip = &bytes[strip_idx * strip_bytes..(strip_idx + 1) * strip_bytes];
            let mut t = 0_usize;
            while t + 8 <= seq_len {
                let acts: [&[Q8_KActBlock]; 8] = std::array::from_fn(|k| {
                    &acts[(t + k) * blocks_per_tok..(t + k + 1) * blocks_per_tok]
                });
                let tile = gemm_q4_k_repacked8_8x8(caps, strip, acts);
                for k in 0..take {
                    let base = (idx + k) * seq_len + t;
                    for (col_idx, &val) in tile[lane + k].iter().enumerate() {
                        dst[base + col_idx] += val;
                    }
                }
                t += 8;
            }
            if t + 4 <= seq_len {
                let acts: [&[Q8_KActBlock]; 4] = std::array::from_fn(|k| {
                    &acts[(t + k) * blocks_per_tok..(t + k + 1) * blocks_per_tok]
                });
                let tile = gemm_q4_k_repacked8_8x4(caps, strip, acts);
                for k in 0..take {
                    let base = (idx + k) * seq_len + t;
                    for (col_idx, &val) in tile[lane + k].iter().enumerate() {
                        dst[base + col_idx] += val;
                    }
                }
                t += 4;
            }
            if t + 2 <= seq_len {
                let a0 = &acts[t * blocks_per_tok..(t + 1) * blocks_per_tok];
                let a1 = &acts[(t + 1) * blocks_per_tok..(t + 2) * blocks_per_tok];
                let tile = gemm_q4_k_repacked8_8x2(caps, strip, a0, a1);
                for k in 0..take {
                    let base = (idx + k) * seq_len + t;
                    dst[base] += tile[lane + k][0];
                    dst[base + 1] += tile[lane + k][1];
                }
                t += 2;
            }
            if t < seq_len {
                let a0 = &acts[t * blocks_per_tok..(t + 1) * blocks_per_tok];
                let col = dot_q4_k_repacked8(caps, strip, a0);
                for k in 0..take {
                    dst[(idx + k) * seq_len + t] += col[lane + k];
                }
            }
            idx += take;
        }
    }

    fn transpose_to_row_major(src: &[f32], out_dim: usize, seq_len: usize, dst: &mut [f32]) {
        use fearless_simd::{Level, dispatch, f32x4, prelude::*};

        let r_tiles = out_dim / 4;
        let r_tail = r_tiles * 4;
        let t_tiles = seq_len / 4;
        let t_tail = t_tiles * 4;
        let level = Level::new();
        dispatch!(level, simd => {
            for rt in 0..r_tiles {
                let r0 = rt * 4;
                let row0 = &src[r0 * seq_len..(r0 + 1) * seq_len];
                let row1 = &src[(r0 + 1) * seq_len..(r0 + 2) * seq_len];
                let row2 = &src[(r0 + 2) * seq_len..(r0 + 3) * seq_len];
                let row3 = &src[(r0 + 3) * seq_len..(r0 + 4) * seq_len];
                for tt in 0..t_tiles {
                    let t0 = tt * 4;
                    let cols = super::packed::transpose_4x4(
                        simd,
                        [
                            f32x4::from_slice(simd, &row0[t0..t0 + 4]),
                            f32x4::from_slice(simd, &row1[t0..t0 + 4]),
                            f32x4::from_slice(simd, &row2[t0..t0 + 4]),
                            f32x4::from_slice(simd, &row3[t0..t0 + 4]),
                        ],
                    );
                    for (dt, col) in cols.into_iter().enumerate() {
                        let base = (t0 + dt) * out_dim + r0;
                        col.store_slice(&mut dst[base..base + 4]);
                    }
                }
                for t in t_tail..seq_len {
                    let base = t * out_dim + r0;
                    dst[base] = row0[t];
                    dst[base + 1] = row1[t];
                    dst[base + 2] = row2[t];
                    dst[base + 3] = row3[t];
                }
            }
        });
        for r in r_tail..out_dim {
            let row = &src[r * seq_len..(r + 1) * seq_len];
            for (t, &val) in row.iter().enumerate() {
                dst[t * out_dim + r] = val;
            }
        }
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "Generic 8-row I8MM GEMM kernel processes 8x8/8x4/8x2/8x1 tiles and delegates tail rows to 4-row GEMM."
    )]
    fn add_batch_8row_gemm<Act>(
        dst: &mut [f32],
        bytes: &[u8],
        in_dim: usize,
        seq_len: usize,
        local_row_start: usize,
        rows_here: usize,
        block_elems: usize,
        elem_bytes: usize,
        acts_buf: &[Act],
        gemm_8x8: impl Fn([&[u8]; 8], [&[Act]; 8]) -> [[f32; 8]; 8],
        gemm_8x4: impl Fn([&[u8]; 8], [&[Act]; 4]) -> [[f32; 4]; 8],
        gemm_8x2: impl Fn([&[u8]; 8], [&[Act]; 2]) -> [[f32; 2]; 8],
        dot_8rows: impl Fn([&[u8]; 8], &[Act]) -> [f32; 8],
        gemm_4x2: impl Fn([&[u8]; 4], &[Act], &[Act]) -> [[f32; 2]; 4],
        dot_4rows: impl Fn([&[u8]; 4], &[Act]) -> [f32; 4],
        dot_row: impl Fn(&[u8], &[Act]) -> f32,
    ) {
        let blocks_per_tok = in_dim / block_elems;
        let row_bytes = blocks_per_tok * elem_bytes;
        let mut processed = 0_usize;
        while processed + 8 <= rows_here {
            let r0 = (local_row_start + processed) * row_bytes;
            let row_refs: [&[u8]; 8] = std::array::from_fn(|r| {
                let start = r0 + r * row_bytes;
                &bytes[start..start + row_bytes]
            });
            let mut t = 0_usize;
            while t + 8 <= seq_len {
                let acts: [&[Act]; 8] = std::array::from_fn(|k| {
                    &acts_buf[(t + k) * blocks_per_tok..(t + k + 1) * blocks_per_tok]
                });
                let tile = gemm_8x8(row_refs, acts);
                for (r, row_vals) in tile.iter().enumerate() {
                    let base = (processed + r) * seq_len + t;
                    for (k, &val) in row_vals.iter().enumerate() {
                        dst[base + k] += val;
                    }
                }
                t += 8;
            }
            if t + 4 <= seq_len {
                let acts: [&[Act]; 4] = std::array::from_fn(|k| {
                    &acts_buf[(t + k) * blocks_per_tok..(t + k + 1) * blocks_per_tok]
                });
                let tile = gemm_8x4(row_refs, acts);
                for (r, row_vals) in tile.iter().enumerate() {
                    let base = (processed + r) * seq_len + t;
                    for (k, &val) in row_vals.iter().enumerate() {
                        dst[base + k] += val;
                    }
                }
                t += 4;
            }
            if t + 2 <= seq_len {
                let acts: [&[Act]; 2] = std::array::from_fn(|k| {
                    &acts_buf[(t + k) * blocks_per_tok..(t + k + 1) * blocks_per_tok]
                });
                let tile = gemm_8x2(row_refs, acts);
                for (r, row_vals) in tile.iter().enumerate() {
                    let base = (processed + r) * seq_len + t;
                    dst[base] += row_vals[0];
                    dst[base + 1] += row_vals[1];
                }
                t += 2;
            }
            if t < seq_len {
                let a0 = &acts_buf[t * blocks_per_tok..(t + 1) * blocks_per_tok];
                let col = dot_8rows(row_refs, a0);
                for (r, &val) in col.iter().enumerate() {
                    dst[(processed + r) * seq_len + t] += val;
                }
            }
            processed += 8;
        }
        if processed < rows_here {
            add_batch_4row_gemm(
                &mut dst[processed * seq_len..],
                bytes,
                in_dim,
                seq_len,
                local_row_start + processed,
                rows_here - processed,
                block_elems,
                elem_bytes,
                acts_buf,
                gemm_4x2,
                dot_4rows,
                dot_row,
            );
        }
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "Dispatches batched quantized GEMM kernels over shared pre-quantized batch activations."
    )]
    fn dispatch_batch_segment(
        kind: GroupKind<'_>,
        dst: &mut [f32],
        in_dim: usize,
        seq_len: usize,
        local_row: usize,
        rows_here: usize,
        caps: Arm64QuantCaps,
        q8_act: &[Q8_0ActBlock],
        qk_act: &[Q8_KActBlock],
    ) {
        match kind {
            GroupKind::Packed {
                ggml_type: 8,
                bytes,
            } => add_batch_8row_gemm(
                dst,
                bytes,
                in_dim,
                seq_len,
                local_row,
                rows_here,
                Q8_0_BLOCK_ELEMENTS,
                Q8_0_BLOCK_BYTES,
                q8_act,
                |r, a| gemm_q8_0_8x8(caps, r, a),
                |r, a| gemm_q8_0_8x4(caps, r, a),
                |r, a| gemm_q8_0_8x2(caps, r, a),
                |r, a| dot_q8_0_8rows(caps, r, a),
                |r, a0, a1| gemm_q8_0_4x2(caps, r, a0, a1),
                |r, a0| dot_q8_0_4rows(caps, r, a0),
                |r, a0| dot_q8_0_row(caps, r, a0),
            ),
            GroupKind::Packed {
                ggml_type: 12,
                bytes,
            } => add_batch_4row_gemm(
                dst,
                bytes,
                in_dim,
                seq_len,
                local_row,
                rows_here,
                QK_K,
                Q4_K_BLOCK_BYTES,
                qk_act,
                |r, a0, a1| gemm_q4_k_4x2(caps, r, a0, a1),
                |r, a0| dot_q4_k_4rows(caps, r, a0),
                |r, a0| dot_q4_k_row(caps, r, a0),
            ),
            GroupKind::Packed { bytes, .. } => add_batch_8row_gemm(
                dst,
                bytes,
                in_dim,
                seq_len,
                local_row,
                rows_here,
                QK_K,
                Q6_K_BLOCK_BYTES,
                qk_act,
                |r, a| gemm_q6_k_8x8(caps, r, a),
                |r, a| gemm_q6_k_8x4(caps, r, a),
                |r, a| gemm_q6_k_8x2(caps, r, a),
                |r, a| dot_q6_k_8rows(caps, r, a),
                |r, a0, a1| gemm_q6_k_4x2(caps, r, a0, a1),
                |r, a0| dot_q6_k_4rows(caps, r, a0),
                |r, a0| dot_q6_k_row(caps, r, a0),
            ),
            GroupKind::Repacked(bytes) => add_batch_repacked_gemm(
                dst, bytes, in_dim, seq_len, local_row, rows_here, caps, qk_act,
            ),
        }
    }

    pub(super) fn dispatch_batch_group_quant_kernels(
        targets: &mut [(
            &LinearProjection,
            Option<&super::LoraProjectionPatch>,
            &mut Vec<f32>,
        )],
        inputs: &[f32],
        seq_len: usize,
        worker_count: usize,
    ) -> bool {
        let caps = Arm64QuantCaps::detect();
        if !caps.dotprod {
            return false;
        }
        struct Segment<'a> {
            start: usize,
            len: usize,
            in_dim: usize,
            kind: GroupKind<'a>,
        }
        let mut segments = Vec::with_capacity(targets.len());
        let mut need_q8 = false;
        let mut need_qk = false;
        let mut total_out = 0_usize;
        for (proj, _, _) in targets.iter() {
            let Some(kind) = classify_group_proj(proj) else {
                return false;
            };
            match kind {
                GroupKind::Packed { ggml_type: 8, .. } => need_q8 = true,
                _ => need_qk = true,
            }
            segments.push(Segment {
                start: total_out,
                len: proj.out_dim,
                in_dim: proj.in_dim,
                kind,
            });
            total_out += proj.out_dim;
        }
        let mut combined = vec![0.0_f32; total_out * seq_len];
        for (seg, (proj, _, _)) in segments.iter().zip(targets.iter()) {
            if let Some(bias) = &proj.bias {
                for (r, &b) in bias.iter().enumerate() {
                    let off = (seg.start + r) * seq_len;
                    combined[off..off + seq_len].fill(b);
                }
            }
        }
        let in_dim = segments[0].in_dim;
        let q8_opt = need_q8.then(|| {
            let blocks_per_tok = in_dim / Q8_0_BLOCK_ELEMENTS;
            let mut buf = vec![Q8_0ActBlock::default(); seq_len * blocks_per_tok];
            for t in 0..seq_len {
                quantize_row_q8_0(
                    &inputs[t * in_dim..(t + 1) * in_dim],
                    &mut buf[t * blocks_per_tok..(t + 1) * blocks_per_tok],
                );
            }
            buf
        });
        let qk_opt = need_qk.then(|| {
            let blocks_per_tok = in_dim / QK_K;
            let mut buf = vec![Q8_KActBlock::default(); seq_len * blocks_per_tok];
            for t in 0..seq_len {
                quantize_row_q8_k(
                    &inputs[t * in_dim..(t + 1) * in_dim],
                    &mut buf[t * blocks_per_tok..(t + 1) * blocks_per_tok],
                );
            }
            buf
        });
        let q8_slice = q8_opt.as_deref().unwrap_or(&[]);
        let qk_slice = qk_opt.as_deref().unwrap_or(&[]);
        let row_align = if segments.iter().all(|s| s.len.is_multiple_of(8)) {
            8
        } else if segments.iter().all(|s| s.len.is_multiple_of(4)) {
            4
        } else {
            1
        };
        let workers = worker_count.clamp(1, 18).min(total_out.max(1));
        run_row_chunks_aligned(
            &mut combined,
            workers,
            row_align * seq_len,
            |flat_off, dst| {
                let global_row = flat_off / seq_len;
                let rows_total = dst.len() / seq_len;
                let chunk_end = global_row + rows_total;
                for seg in &segments {
                    let seg_end = seg.start + seg.len;
                    let ov_start = global_row.max(seg.start);
                    let ov_end = chunk_end.min(seg_end);
                    if ov_start >= ov_end {
                        continue;
                    }
                    let dst_slice = &mut dst
                        [(ov_start - global_row) * seq_len..(ov_end - global_row) * seq_len];
                    dispatch_batch_segment(
                        seg.kind,
                        dst_slice,
                        seg.in_dim,
                        seq_len,
                        ov_start - seg.start,
                        ov_end - ov_start,
                        caps,
                        q8_slice,
                        qk_slice,
                    );
                }
            },
        );
        for (seg, (_, _, out)) in segments.iter().zip(targets.iter_mut()) {
            let src_slice = &combined[seg.start * seq_len..(seg.start + seg.len) * seq_len];
            transpose_to_row_major(src_slice, seg.len, seq_len, out);
        }
        true
    }

    pub(super) fn dispatch_batch_quant_kernels(
        proj: &LinearProjection,
        inputs: &[f32],
        seq_len: usize,
        worker_count: usize,
        out: &mut [f32],
    ) -> std::io::Result<bool> {
        let caps = Arm64QuantCaps::detect();
        if !caps.dotprod {
            return Ok(false);
        }
        let Some(kind) = classify_group_proj(proj) else {
            return Ok(false);
        };
        let in_dim = proj.in_dim;
        let out_dim = proj.out_dim;
        let mut transposed = vec![0.0_f32; out_dim * seq_len];
        if let Some(bias) = &proj.bias {
            for (r, row) in transposed.chunks_exact_mut(seq_len).enumerate() {
                row.fill(bias[r]);
            }
        }
        let worker_count = worker_count.clamp(1, 18).min(out_dim.max(1));
        let row_align = match kind {
            GroupKind::Repacked(_)
            | GroupKind::Packed {
                ggml_type: 8 | 14, ..
            } => 8,
            GroupKind::Packed { .. } => 4,
        };
        match kind {
            GroupKind::Packed { ggml_type: 8, .. } => {
                let blocks_per_tok = in_dim / Q8_0_BLOCK_ELEMENTS;
                let mut acts = vec![Q8_0ActBlock::default(); seq_len * blocks_per_tok];
                for t in 0..seq_len {
                    quantize_row_q8_0(
                        &inputs[t * in_dim..(t + 1) * in_dim],
                        &mut acts[t * blocks_per_tok..(t + 1) * blocks_per_tok],
                    );
                }
                run_row_chunks_aligned(
                    &mut transposed,
                    worker_count,
                    row_align * seq_len,
                    |flat_off, dst| {
                        let start_row = flat_off / seq_len;
                        let rows_here = dst.len() / seq_len;
                        dispatch_batch_segment(
                            kind,
                            dst,
                            in_dim,
                            seq_len,
                            start_row,
                            rows_here,
                            caps,
                            &acts,
                            &[],
                        );
                    },
                );
            }
            _ => {
                let blocks_per_tok = in_dim / QK_K;
                let mut acts = vec![Q8_KActBlock::default(); seq_len * blocks_per_tok];
                for t in 0..seq_len {
                    quantize_row_q8_k(
                        &inputs[t * in_dim..(t + 1) * in_dim],
                        &mut acts[t * blocks_per_tok..(t + 1) * blocks_per_tok],
                    );
                }
                run_row_chunks_aligned(
                    &mut transposed,
                    worker_count,
                    row_align * seq_len,
                    |flat_off, dst| {
                        let start_row = flat_off / seq_len;
                        let rows_here = dst.len() / seq_len;
                        dispatch_batch_segment(
                            kind,
                            dst,
                            in_dim,
                            seq_len,
                            start_row,
                            rows_here,
                            caps,
                            &[],
                            &acts,
                        );
                    },
                );
            }
        }

        transpose_to_row_major(&transposed, out_dim, seq_len, out);
        Ok(true)
    }
}

use quant_dispatch::{
    dispatch_batch_quant_kernels, dispatch_packed_quant_kernels, dispatch_repacked_q4_k,
};

fn run_row_chunks_aligned<F>(out: &mut [f32], worker_count: usize, alignment: usize, compute: F)
where
    F: Fn(usize, &mut [f32]) + Sync,
{
    row_workers::run_aligned(out, worker_count, alignment, &compute);
}

fn forward_group_into(
    targets: &mut [(
        &LinearProjection,
        Option<&LoraProjectionPatch>,
        &mut Vec<f32>,
    )],
    input: &[f32],
    threads: usize,
) -> io::Result<()> {
    if !(1..=256).contains(&threads) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Invalid CPU worker count.",
        ));
    }
    let mut total_out = 0_usize;
    for (proj, lora, out) in targets.iter_mut() {
        let _layout = projection_layout::validate(proj.in_dim, proj.out_dim, &proj.weight)?;
        if input.len() != proj.in_dim {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Projection input dimension mismatch.",
            ));
        }
        if let Some(bias) = &proj.bias
            && bias.len() != proj.out_dim
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Projection bias length mismatch.",
            ));
        }
        if let Some(patch) = *lora {
            patch.validate(proj.in_dim, proj.out_dim)?;
        }
        out.resize(proj.out_dim, 0.0);
        total_out = total_out.saturating_add(proj.out_dim);
    }

    if threads > 1 && total_out >= 8 && targets.len() <= 4 {
        let caps = bonsai_windows_process::quant_kernels::Arm64QuantCaps::detect();
        if caps.dotprod {
            #[derive(Clone, Copy)]
            struct Segment<'a> {
                start: usize,
                len: usize,
                in_dim: usize,
                kind: quant_dispatch::GroupKind<'a>,
            }
            let mut segments: [Option<Segment<'_>>; 4] = [None; 4];
            let mut seg_count = 0_usize;
            let mut need_q8 = false;
            let mut need_qk = false;
            let mut cursor = 0_usize;
            let mut all_packed = true;
            for (proj, _, _) in targets.iter() {
                let Some(kind) = quant_dispatch::classify_group_proj(proj) else {
                    all_packed = false;
                    break;
                };
                match kind {
                    quant_dispatch::GroupKind::Packed { ggml_type: 8, .. } => need_q8 = true,
                    _ => need_qk = true,
                }
                segments[seg_count] = Some(Segment {
                    start: cursor,
                    len: proj.out_dim,
                    in_dim: proj.in_dim,
                    kind,
                });
                seg_count += 1;
                cursor += proj.out_dim;
            }
            if all_packed && seg_count == targets.len() {
                let alignment = if segments[..seg_count]
                    .iter()
                    .flatten()
                    .all(|s| s.len.is_multiple_of(8))
                {
                    8
                } else if segments[..seg_count]
                    .iter()
                    .flatten()
                    .all(|s| s.len.is_multiple_of(4))
                {
                    4
                } else {
                    1
                };
                let workers = threads.clamp(1, 18).min(total_out);
                quant_dispatch::with_group_workspace(
                    total_out,
                    need_q8,
                    need_qk,
                    input,
                    |combined, q8_slice, qk_slice| -> io::Result<()> {
                        for (seg_opt, (proj, _, _)) in
                            segments[..seg_count].iter().zip(targets.iter())
                        {
                            if let Some(seg) = *seg_opt {
                                let seg_dst = &mut combined[seg.start..seg.start + seg.len];
                                if let Some(bias) = &proj.bias {
                                    seg_dst.copy_from_slice(bias);
                                } else {
                                    seg_dst.fill(0.0);
                                }
                            }
                        }
                        run_row_chunks_aligned(combined, workers, alignment, |global_row, dst| {
                            let chunk_end = global_row + dst.len();
                            for seg in segments[..seg_count].iter().flatten() {
                                let seg_end = seg.start + seg.len;
                                let ov_start = global_row.max(seg.start);
                                let ov_end = chunk_end.min(seg_end);
                                if ov_start >= ov_end {
                                    continue;
                                }
                                let dst_slice =
                                    &mut dst[ov_start - global_row..ov_end - global_row];
                                let local_row = ov_start - seg.start;
                                match seg.kind {
                                    quant_dispatch::GroupKind::Packed { ggml_type, bytes } => {
                                        quant_dispatch::add_packed_rows(
                                            ggml_type, bytes, seg.in_dim, caps, q8_slice, qk_slice,
                                            local_row, dst_slice,
                                        );
                                    }
                                    quant_dispatch::GroupKind::Repacked(bytes) => {
                                        quant_dispatch::add_repacked_rows(
                                            bytes, seg.in_dim, caps, qk_slice, local_row, dst_slice,
                                        );
                                    }
                                }
                            }
                        });
                        let mut offset = 0_usize;
                        for (proj, lora, out) in targets.iter_mut() {
                            let len = proj.out_dim;
                            out.copy_from_slice(&combined[offset..offset + len]);
                            if let Some(patch) = *lora {
                                patch.apply(input, out)?;
                            }
                            offset += len;
                        }
                        Ok(())
                    },
                )?;
                return Ok(());
            }
        }
    }

    for (proj, lora, out) in targets.iter_mut() {
        proj.forward_vec_into(input, None, *lora, threads, out)?;
    }
    Ok(())
}

fn run_row_chunks<F>(out: &mut [f32], worker_count: usize, compute_slice: F)
where
    F: Fn(usize, &mut [f32]) + Sync,
{
    row_workers::run(out, worker_count, &compute_slice);
}

#[expect(
    clippy::inline_always,
    reason = "fearless_simd dispatch! requires inline(always) to inherit target_feature attributes."
)]
#[inline(always)]
fn rms_norm_scale_simd<S: Simd>(
    simd: S,
    input: &[f32],
    weight: &[f32],
    scale: f32,
    out: &mut [f32],
) {
    let scale_v = f32x4::splat(simd, scale);
    let (in_chunks, in_tail) = input.as_chunks::<4>();
    let (w_chunks, w_tail) = weight.as_chunks::<4>();
    let (out_chunks, out_tail) = out.as_chunks_mut::<4>();
    for ((x_c, w_c), dst_c) in in_chunks
        .iter()
        .zip(w_chunks.iter())
        .zip(out_chunks.iter_mut())
    {
        let xv = f32x4::from_slice(simd, x_c);
        let wv = f32x4::from_slice(simd, w_c);
        ((xv * scale_v) * wv).store_slice(dst_c);
    }
    for ((&x, &w), dst) in in_tail.iter().zip(w_tail.iter()).zip(out_tail.iter_mut()) {
        *dst = (x * scale) * w;
    }
}

#[expect(
    clippy::inline_always,
    reason = "fearless_simd dispatch! requires inline(always) to inherit target_feature attributes."
)]
#[inline(always)]
fn rms_norm_in_place_simd<S: Simd>(simd: S, data: &mut [f32], weight: &[f32], scale: f32) {
    let scale_v = f32x4::splat(simd, scale);
    let (d_chunks, d_tail) = data.as_chunks_mut::<4>();
    let (w_chunks, w_tail) = weight.as_chunks::<4>();
    for (d_c, w_c) in d_chunks.iter_mut().zip(w_chunks.iter()) {
        let xv = f32x4::from_slice(simd, d_c);
        let wv = f32x4::from_slice(simd, w_c);
        ((xv * scale_v) * wv).store_slice(d_c);
    }
    for (x, &w) in d_tail.iter_mut().zip(w_tail.iter()) {
        *x = (*x * scale) * w;
    }
}

/// RMS normalization (`x * weight / sqrt(mean(x^2) + eps)`).
pub fn rms_norm(input: &[f32], weight: &[f32], eps: f32) -> io::Result<Vec<f32>> {
    let mut out = Vec::new();
    rms_norm_into(input, weight, eps, &mut out)?;
    Ok(out)
}

fn rms_norm_to_slice_with_level(
    level: Level,
    input: &[f32],
    weight: &[f32],
    eps: f32,
    out: &mut [f32],
) {
    let inv_n = 1.0_f32 / usize_to_f32(input.len());
    let mean_sq = input.iter().map(|&v| v * v).sum::<f32>() * inv_n;
    let scale = 1.0_f32 / (mean_sq + eps).sqrt();
    dispatch!(level, simd => rms_norm_scale_simd(simd, input, weight, scale, out));
}

pub(crate) fn rms_norm_into(
    input: &[f32],
    weight: &[f32],
    eps: f32,
    out: &mut Vec<f32>,
) -> io::Result<()> {
    if input.is_empty() || input.len() != weight.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "RMSNorm dimension mismatch.",
        ));
    }
    decoder_workspace::resize(out, input.len())?;
    rms_norm_to_slice_with_level(Level::new(), input, weight, eps, out);
    Ok(())
}

fn rms_norm_batch_into(
    inputs: &[f32],
    seq_len: usize,
    weight: &[f32],
    eps: f32,
    workers: usize,
    out: &mut Vec<f32>,
) -> io::Result<()> {
    let hidden = weight.len();
    if hidden == 0 || inputs.len() != seq_len * hidden {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "RMSNorm batch dimension mismatch.",
        ));
    }
    if seq_len == 0 {
        out.clear();
        return Ok(());
    }
    decoder_workspace::resize(out, inputs.len())?;
    let level = Level::new();
    if seq_len > 1 && workers > 1 {
        row_workers::run_aligned(out, workers, hidden, &|elem_off, dst_chunk| {
            for (row_in, row_out) in inputs[elem_off..elem_off + dst_chunk.len()]
                .chunks_exact(hidden)
                .zip(dst_chunk.chunks_exact_mut(hidden))
            {
                rms_norm_to_slice_with_level(level, row_in, weight, eps, row_out);
            }
        });
    } else {
        for (row_in, row_out) in inputs
            .chunks_exact(hidden)
            .zip(out.chunks_exact_mut(hidden))
        {
            rms_norm_to_slice_with_level(level, row_in, weight, eps, row_out);
        }
    }
    Ok(())
}

/// In-place RMS normalization (`data[i] = (data[i] * scale) * weight[i]`), avoiding per-head heap allocations.
pub fn rms_norm_in_place(data: &mut [f32], weight: &[f32], eps: f32) -> io::Result<()> {
    if data.is_empty() || data.len() != weight.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "RMSNorm dimension mismatch.",
        ));
    }
    let inv_n = 1.0_f32 / usize_to_f32(data.len());
    let mean_sq = data.iter().map(|&v| v * v).sum::<f32>() * inv_n;
    let scale = 1.0_f32 / (mean_sq + eps).sqrt();
    let level = Level::new();
    dispatch!(level, simd => rms_norm_in_place_simd(simd, data, weight, scale));
    Ok(())
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

struct RopeDivisors {
    head_dim: usize,
    theta_bits: u32,
    values: Vec<f32>,
}

#[derive(Default)]
struct RopeCache {
    divisors: [Option<RopeDivisors>; 4],
    next_slot: usize,
    step_key: Option<(usize, u32, usize)>,
    step_cos: Vec<f32>,
    step_sin: Vec<f32>,
    batch_key: Option<(usize, u32, usize, usize)>,
    batch_cos: Vec<f32>,
    batch_sin: Vec<f32>,
}

impl RopeCache {
    fn divisors_for(&mut self, head_dim: usize, theta: f32) -> &[f32] {
        let half = head_dim / 2;
        let theta_bits = theta.to_bits();
        if let Some(idx) = self.divisors.iter().position(|entry| {
            matches!(
                entry,
                Some(d) if d.head_dim == head_dim && d.theta_bits == theta_bits && d.values.len() == half
            )
        }) {
            return self.divisors[idx].as_ref().map_or(&[], |d| &d.values);
        }
        let hd_f = usize_to_f32(head_dim);
        let values: Vec<f32> = (0..half)
            .map(|i| theta.powf(usize_to_f32(2 * i) / hd_f))
            .collect();
        let slot_idx = self
            .divisors
            .iter()
            .position(Option::is_none)
            .unwrap_or_else(|| {
                let idx = self.next_slot;
                self.next_slot = (self.next_slot + 1) % self.divisors.len();
                idx
            });
        self.divisors[slot_idx] = Some(RopeDivisors {
            head_dim,
            theta_bits,
            values,
        });
        self.divisors[slot_idx].as_ref().map_or(&[], |d| &d.values)
    }

    fn ensure_step(&mut self, head_dim: usize, pos: usize, theta: f32) -> (&[f32], &[f32]) {
        let half = head_dim / 2;
        let key = (head_dim, theta.to_bits(), pos);
        if self.step_key == Some(key) && self.step_cos.len() >= half && self.step_sin.len() >= half
        {
            return (&self.step_cos[..half], &self.step_sin[..half]);
        }
        self.divisors_for(head_dim, theta);
        let theta_bits = theta.to_bits();
        let pos_f = usize_to_f32(pos);
        self.step_cos.resize(half, 0.0);
        self.step_sin.resize(half, 0.0);
        if let Some(divs) = self.divisors.iter().flatten().find(|d| {
            d.head_dim == head_dim && d.theta_bits == theta_bits && d.values.len() == half
        }) {
            for ((dst_cos, dst_sin), &div) in self.step_cos[..half]
                .iter_mut()
                .zip(&mut self.step_sin[..half])
                .zip(&divs.values)
            {
                let (sin_v, cos_v) = (pos_f / div).sin_cos();
                *dst_cos = cos_v;
                *dst_sin = sin_v;
            }
        }
        self.step_key = Some(key);
        (&self.step_cos[..half], &self.step_sin[..half])
    }

    fn ensure_batch(
        &mut self,
        head_dim: usize,
        initial_seq: usize,
        seq_len: usize,
        theta: f32,
    ) -> (&[f32], &[f32]) {
        let half = head_dim / 2;
        let needed = seq_len * half;
        let key = (head_dim, theta.to_bits(), initial_seq, seq_len);
        if self.batch_key == Some(key)
            && self.batch_cos.len() >= needed
            && self.batch_sin.len() >= needed
        {
            return (&self.batch_cos[..needed], &self.batch_sin[..needed]);
        }
        self.divisors_for(head_dim, theta);
        let theta_bits = theta.to_bits();
        self.batch_cos.resize(needed, 0.0);
        self.batch_sin.resize(needed, 0.0);
        if let Some(divs) = self.divisors.iter().flatten().find(|d| {
            d.head_dim == head_dim && d.theta_bits == theta_bits && d.values.len() == half
        }) {
            for t in 0..seq_len {
                let pos_f = usize_to_f32(initial_seq + t);
                let row_cos = &mut self.batch_cos[t * half..(t + 1) * half];
                let row_sin = &mut self.batch_sin[t * half..(t + 1) * half];
                for ((dst_cos, dst_sin), &div) in
                    row_cos.iter_mut().zip(row_sin.iter_mut()).zip(&divs.values)
                {
                    let (sin_v, cos_v) = (pos_f / div).sin_cos();
                    *dst_cos = cos_v;
                    *dst_sin = sin_v;
                }
            }
        }
        self.batch_key = Some(key);
        (&self.batch_cos[..needed], &self.batch_sin[..needed])
    }
}

thread_local! {
    static ROPE_CACHE: std::cell::RefCell<RopeCache> =
        const { std::cell::RefCell::new(RopeCache {
            divisors: [const { None }; 4],
            next_slot: 0,
            step_key: None,
            step_cos: Vec::new(),
            step_sin: Vec::new(),
            batch_key: None,
            batch_cos: Vec::new(),
            batch_sin: Vec::new(),
        }) };
}

fn compute_rope_tables_uncached(
    head_dim: usize,
    pos: usize,
    theta: f32,
    cos: &mut [f32],
    sin: &mut [f32],
) {
    let half = head_dim / 2;
    let hd_f = usize_to_f32(head_dim);
    let pos_f = usize_to_f32(pos);
    for i in 0..half {
        let exponent = usize_to_f32(2 * i) / hd_f;
        let freq = pos_f / theta.powf(exponent);
        let (sin_v, cos_v) = freq.sin_cos();
        cos[i] = cos_v;
        sin[i] = sin_v;
    }
}

fn with_rope_step_tables<R>(
    head_dim: usize,
    pos: usize,
    theta: f32,
    action: impl FnOnce(&[f32], &[f32]) -> R,
) -> R {
    let half = head_dim / 2;
    if half == 0 {
        return action(&[], &[]);
    }
    ROPE_CACHE.with(|cell| {
        if let Ok(mut cache) = cell.try_borrow_mut() {
            let (cos, sin) = cache.ensure_step(head_dim, pos, theta);
            action(cos, sin)
        } else if half <= 128 {
            let mut cos_stack = [0.0_f32; 128];
            let mut sin_stack = [0.0_f32; 128];
            compute_rope_tables_uncached(
                head_dim,
                pos,
                theta,
                &mut cos_stack[..half],
                &mut sin_stack[..half],
            );
            action(&cos_stack[..half], &sin_stack[..half])
        } else {
            let mut cos_heap = vec![0.0_f32; half];
            let mut sin_heap = vec![0.0_f32; half];
            compute_rope_tables_uncached(head_dim, pos, theta, &mut cos_heap, &mut sin_heap);
            action(&cos_heap, &sin_heap)
        }
    })
}

fn with_rope_batch_tables(
    head_dim: usize,
    initial_seq: usize,
    seq_len: usize,
    theta: f32,
    mut action: impl FnMut(usize, &[f32], &[f32]) -> io::Result<()>,
) -> io::Result<()> {
    const MAX_CACHED_BATCH_ELEMS: usize = 1_048_576;
    let half = head_dim / 2;
    if half == 0 || seq_len == 0 {
        return Ok(());
    }
    let within_cache = seq_len
        .checked_mul(half)
        .is_some_and(|n| n <= MAX_CACHED_BATCH_ELEMS);
    if within_cache {
        let handled = ROPE_CACHE.with(|cell| {
            let Ok(mut cache) = cell.try_borrow_mut() else {
                return None;
            };
            let (cos_all, sin_all) = cache.ensure_batch(head_dim, initial_seq, seq_len, theta);
            Some((0..seq_len).try_for_each(|t| {
                action(
                    t,
                    &cos_all[t * half..(t + 1) * half],
                    &sin_all[t * half..(t + 1) * half],
                )
            }))
        });
        if let Some(res) = handled {
            return res;
        }
    }
    let mut cos_stack = [0.0_f32; 128];
    let mut sin_stack = [0.0_f32; 128];
    let mut cos_heap = Vec::new();
    let mut sin_heap = Vec::new();
    let (cos_buf, sin_buf): (&mut [f32], &mut [f32]) = if half <= cos_stack.len() {
        (&mut cos_stack[..half], &mut sin_stack[..half])
    } else {
        cos_heap.resize(half, 0.0);
        sin_heap.resize(half, 0.0);
        (&mut cos_heap[..half], &mut sin_heap[..half])
    };
    for t in 0..seq_len {
        compute_rope_tables(head_dim, initial_seq + t, theta, cos_buf, sin_buf);
        action(t, cos_buf, sin_buf)?;
    }
    Ok(())
}

fn prefetch_rope_step(head_dim: usize, pos: usize, theta: f32) {
    with_rope_step_tables(head_dim, pos, theta, |_, _| {});
}

fn compute_rope_tables(head_dim: usize, pos: usize, theta: f32, cos: &mut [f32], sin: &mut [f32]) {
    let half = head_dim / 2;
    with_rope_step_tables(head_dim, pos, theta, |cached_cos, cached_sin| {
        cos[..half].copy_from_slice(cached_cos);
        sin[..half].copy_from_slice(cached_sin);
    });
}

pub(crate) fn apply_rope_neox_with_tables(
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
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Invalid RoPE head dimensions.",
        ));
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

/// Apply 1D NEOX-style half-split `RoPE` (or collapsed 3D `MRoPE` when all 3 axes share `pos`)
/// in place over `[num_heads, head_dim]`, precomputing `(sin, cos)` once across all heads.
pub fn apply_rope_neox_in_place(
    heads: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    pos: usize,
    theta: f32,
) -> io::Result<()> {
    if head_dim == 0 || !head_dim.is_multiple_of(2) || heads.len() != num_heads * head_dim {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Invalid RoPE head dimensions.",
        ));
    }
    with_rope_step_tables(head_dim, pos, theta, |cos_table, sin_table| {
        apply_rope_neox_with_tables(heads, num_heads, head_dim, cos_table, sin_table)
    })
}

fn normalize_heads_in_place(
    buf: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    weight: &[f32],
    eps: f32,
) -> io::Result<()> {
    for h in 0..num_heads {
        rms_norm_in_place(&mut buf[h * head_dim..(h + 1) * head_dim], weight, eps)?;
    }
    Ok(())
}

#[inline(always)]
fn swiglu_slice_simd<S: Simd>(simd: S, gate: &mut [f32], up: &[f32]) {
    let zero = f32x4::splat(simd, 0.0_f32);
    let one = f32x4::splat(simd, 1.0_f32);
    let n = gate.len().min(up.len());
    let (g_chunks, g_tail) = gate[..n].as_chunks_mut::<4>();
    let (u_chunks, u_tail) = up[..n].as_chunks::<4>();
    for (gc, uc) in g_chunks.iter_mut().zip(u_chunks) {
        let gv = f32x4::from_slice(simd, gc);
        let uv = f32x4::from_slice(simd, uc);
        // Evaluate exp(-|g|) <= 0 so exp_f32x4 never overflows its k <= 127 exponent range.
        let e = exp_f32x4(simd, -gv.abs());
        let num = gv.simd_lt(zero).select(gv * e, gv);
        let silu = num / (one + e);
        (silu * uv).store_slice(gc);
    }
    for (g, &u) in g_tail.iter_mut().zip(u_tail) {
        let silu = *g / (1.0_f32 + (-*g).exp());
        *g = silu * u;
    }
}

pub(crate) fn swiglu_in_place(gate: &mut [f32], up: &[f32]) {
    let level = Level::new();
    dispatch!(level, simd => swiglu_slice_simd(simd, gate, up));
}

pub(crate) fn swiglu_in_place_parallel(gate: &mut [f32], up: &[f32], workers: usize) {
    const PARALLEL_SWIGLU_THRESHOLD: usize = 4096;
    let n = gate.len().min(up.len());
    if workers > 1 && n >= PARALLEL_SWIGLU_THRESHOLD {
        let level = Level::new();
        row_workers::run_aligned(&mut gate[..n], workers, 4, &|offset, g_chunk| {
            let u_chunk = &up[offset..offset + g_chunk.len()];
            dispatch!(level, simd => swiglu_slice_simd(simd, g_chunk, u_chunk));
        });
    } else {
        swiglu_in_place(&mut gate[..n], &up[..n]);
    }
}

/// One decoder layer shared by `Talker` (28 layers) and `CodePredictor` (5 layers).
#[derive(Clone, Debug)]
pub struct QwenDecoderLayer {
    /// Pre-attention `RMSNorm` scale `[hidden_size]`.
    pub attn_norm_w: Vec<f32>,
    /// Query projection `[num_q_heads * head_dim, hidden_size]`.
    pub q_proj: LinearProjection,
    /// Key projection `[num_kv_heads * head_dim, hidden_size]`.
    pub k_proj: LinearProjection,
    /// Value projection `[num_kv_heads * head_dim, hidden_size]`.
    pub v_proj: LinearProjection,
    /// Output projection `[hidden_size, num_q_heads * head_dim]`.
    pub o_proj: LinearProjection,
    /// Per-head query `RMSNorm` `[head_dim]`.
    pub q_norm_w: Vec<f32>,
    /// Per-head key `RMSNorm` `[head_dim]`.
    pub k_norm_w: Vec<f32>,
    /// Post-attention `RMSNorm` scale `[hidden_size]`.
    pub ffn_norm_w: Vec<f32>,
    /// `SwiGLU` gate projection `[intermediate_size, hidden_size]`.
    pub gate_proj: LinearProjection,
    /// `SwiGLU` up projection `[intermediate_size, hidden_size]`.
    pub up_proj: LinearProjection,
    /// `SwiGLU` down projection `[hidden_size, intermediate_size]`.
    pub down_proj: LinearProjection,
}

/// Per-layer KV cache storing past key/value vectors (`[seq_len, num_kv_heads * head_dim]`).
#[derive(Clone, Debug, Default)]
pub struct LayerKvCache {
    /// Cached `F16` keys (`[seq_len * num_kv_heads * head_dim]`).
    pub k: Vec<u16>,
    /// Cached `F16` values (`[seq_len * num_kv_heads * head_dim]`).
    pub v: Vec<u16>,
    /// Number of cached token positions.
    pub seq_len: usize,
}

/// Widen `IEEE binary16` `u16` elements into `f32` (`dst.len() == src.len()`), using ARM64 `FCVTL`
/// on Windows when available.
pub(crate) fn widen_f16_slice(src: &[u16], dst: &mut [f32]) {
    bonsai_neon_dot::widen_f16_slice(src, dst);
}

/// Narrow `f32` values into `IEEE binary16` `u16` bits (`dst.len() == src.len()`), returning
/// `false` if any input is non-finite or overflows binary16 range (`|x| >= 65520`).
pub(crate) fn narrow_f32_slice(src: &[f32], dst: &mut [u16]) -> bool {
    bonsai_neon_dot::narrow_f32_slice_checked(src, dst)
}

impl QwenDecoderLayer {
    /// Repack eligible `Q4_K` layer projections into the aligned ARM64 kernel layout.
    pub(crate) fn repack_q4_k_if_aligned(&mut self) {
        for proj in [
            &mut self.q_proj,
            &mut self.k_proj,
            &mut self.v_proj,
            &mut self.o_proj,
            &mut self.gate_proj,
            &mut self.up_proj,
            &mut self.down_proj,
        ] {
            proj.repack_q4_k_if_aligned();
        }
    }

    /// Run a single-token step (`pos = kv.seq_len`), appending K/V into `kv` and applying optional `LoRA` patches.
    #[expect(
        clippy::too_many_arguments,
        reason = "Passes explicit GQA geometry, RoPE/RMSNorm scalars, KV cache, and optional LoRA adapter."
    )]
    pub fn forward_step(
        &self,
        input: &[f32],
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope_theta: f32,
        rms_eps: f32,
        kv: &mut LayerKvCache,
        adapter: Option<&TalkerLoraAdapter>,
    ) -> io::Result<Vec<f32>> {
        self.forward_step_parallel(
            input,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rope_theta,
            rms_eps,
            kv,
            adapter,
            1,
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "Validates the fixed GQA layer geometry before single-step or batched execution."
    )]
    fn validate_geometry(
        &self,
        hidden: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope_theta: f32,
        rms_eps: f32,
        kv: &LayerKvCache,
        workers: usize,
    ) -> io::Result<(usize, usize)> {
        if !(1..=256).contains(&workers) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Invalid CPU worker count.",
            ));
        }
        let invalid_geometry = || {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "Invalid speech decoder geometry or state.",
            )
        };
        let query_width = num_q_heads
            .checked_mul(head_dim)
            .ok_or_else(invalid_geometry)?;
        let kv_width = num_kv_heads
            .checked_mul(head_dim)
            .ok_or_else(invalid_geometry)?;
        if hidden == 0
            || num_q_heads == 0
            || num_kv_heads == 0
            || !num_q_heads.is_multiple_of(num_kv_heads)
            || head_dim == 0
            || !head_dim.is_multiple_of(2)
            || !rope_theta.is_finite()
            || rope_theta <= 0.0
            || !rms_eps.is_finite()
            || rms_eps <= 0.0
            || self.attn_norm_w.len() != hidden
            || self.ffn_norm_w.len() != hidden
            || self.q_norm_w.len() != head_dim
            || self.k_norm_w.len() != head_dim
            || self.q_proj.in_dim != hidden
            || self.q_proj.out_dim != query_width
            || self.k_proj.in_dim != hidden
            || self.k_proj.out_dim != kv_width
            || self.v_proj.in_dim != hidden
            || self.v_proj.out_dim != kv_width
            || self.o_proj.in_dim != query_width
            || self.o_proj.out_dim != hidden
            || self.gate_proj.in_dim != hidden
            || self.up_proj.in_dim != hidden
            || self.gate_proj.out_dim == 0
            || self.gate_proj.out_dim != self.up_proj.out_dim
            || self.down_proj.in_dim != self.gate_proj.out_dim
            || self.down_proj.out_dim != hidden
            || kv.seq_len.checked_mul(kv_width) != Some(kv.k.len())
            || kv.k.len() != kv.v.len()
        {
            return Err(invalid_geometry());
        }
        Ok((query_width, kv_width))
    }

    /// Run one decoder step with an explicit bounded CPU worker count.
    /// Failed steps, including unwinding, preserve the prior KV contents and position.
    #[expect(
        clippy::too_many_arguments,
        reason = "Keep GQA geometry, cache ownership and compute budget explicit."
    )]
    pub fn forward_step_parallel(
        &self,
        input: &[f32],
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope_theta: f32,
        rms_eps: f32,
        kv: &mut LayerKvCache,
        adapter: Option<&TalkerLoraAdapter>,
        workers: usize,
    ) -> io::Result<Vec<f32>> {
        self.forward_step_reusing(
            input,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rope_theta,
            rms_eps,
            kv,
            adapter,
            workers,
            &mut decoder_workspace::DecoderWorkspace::default(),
        )
    }

    /// Evaluate with utterance-owned temporaries and the same transactional KV append.
    #[expect(
        clippy::too_many_arguments,
        reason = "Keep GQA geometry, transactional cache and request-owned temporary storage explicit."
    )]
    pub(crate) fn forward_step_reusing(
        &self,
        input: &[f32],
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope_theta: f32,
        rms_eps: f32,
        kv: &mut LayerKvCache,
        adapter: Option<&TalkerLoraAdapter>,
        workers: usize,
        workspace: &mut decoder_workspace::DecoderWorkspace,
    ) -> io::Result<Vec<f32>> {
        let mut output = vec![0.0_f32; input.len()];
        self.forward_step_into(
            input,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rope_theta,
            rms_eps,
            kv,
            adapter,
            workers,
            workspace,
            &mut output,
        )?;
        Ok(output)
    }

    /// Evaluate into a caller-owned output slice with zero per-layer heap allocations.
    #[expect(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        reason = "Keep GQA geometry, transactional cache and request-owned temporary storage explicit."
    )]
    pub(crate) fn forward_step_into(
        &self,
        input: &[f32],
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope_theta: f32,
        rms_eps: f32,
        kv: &mut LayerKvCache,
        adapter: Option<&TalkerLoraAdapter>,
        workers: usize,
        workspace: &mut decoder_workspace::DecoderWorkspace,
        output: &mut [f32],
    ) -> io::Result<()> {
        if input.len() != output.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Invalid speech decoder geometry or state.",
            ));
        }
        let (query_width, _kv_width) = self.validate_geometry(
            input.len(),
            num_q_heads,
            num_kv_heads,
            head_dim,
            rope_theta,
            rms_eps,
            kv,
            workers,
        )?;
        let decoder_workspace::DecoderWorkspace {
            norm: h_norm,
            q,
            k,
            v,
            attention: attn_out,
            scores,
            projected,
            gate,
            up,
            residual: x_res,
        } = workspace;
        rms_norm_into(input, &self.attn_norm_w, rms_eps, h_norm)?;
        let patch = |proj: &LinearProjection| adapter.and_then(|a| a.patches.get(&proj.name));

        let q_sig = self.q_proj.hadamard_signature();
        let k_sig = self.k_proj.hadamard_signature();
        let v_sig = self.v_proj.hadamard_signature();
        let shared_qkv_folded = if let Some((signs, block)) = q_sig
            && (k_sig == q_sig || v_sig == q_sig)
        {
            Some((
                signs,
                block,
                apply_hadamard_sign_fold(h_norm, signs, block)?,
            ))
        } else {
            None
        };
        let qkv_fold_for = |sig: Option<(&[i8], usize)>| -> Option<&[f32]> {
            shared_qkv_folded.as_ref().and_then(|(s, b, buf)| {
                if sig == Some((*s, *b)) {
                    Some(buf.as_slice())
                } else {
                    None
                }
            })
        };

        if shared_qkv_folded.is_none()
            && !matches!(
                self.q_proj.weight,
                WeightQuantization::PQ2_0 { .. } | WeightQuantization::PTQ1_0 { .. }
            )
            && !matches!(
                self.k_proj.weight,
                WeightQuantization::PQ2_0 { .. } | WeightQuantization::PTQ1_0 { .. }
            )
            && !matches!(
                self.v_proj.weight,
                WeightQuantization::PQ2_0 { .. } | WeightQuantization::PTQ1_0 { .. }
            )
        {
            LinearProjection::forward_triple_into(
                &self.q_proj,
                &self.k_proj,
                &self.v_proj,
                h_norm,
                patch(&self.q_proj),
                patch(&self.k_proj),
                patch(&self.v_proj),
                workers,
                q,
                k,
                v,
            )?;
        } else {
            self.q_proj.forward_vec_into(
                h_norm,
                qkv_fold_for(q_sig),
                patch(&self.q_proj),
                workers,
                q,
            )?;
            self.k_proj.forward_vec_into(
                h_norm,
                qkv_fold_for(k_sig),
                patch(&self.k_proj),
                workers,
                k,
            )?;
            self.v_proj.forward_vec_into(
                h_norm,
                qkv_fold_for(v_sig),
                patch(&self.v_proj),
                workers,
                v,
            )?;
        }

        normalize_heads_in_place(q, num_q_heads, head_dim, &self.q_norm_w, rms_eps)?;
        normalize_heads_in_place(k, num_kv_heads, head_dim, &self.k_norm_w, rms_eps)?;

        let pos = kv.seq_len;
        with_rope_step_tables(head_dim, pos, rope_theta, |cos_table, sin_table| {
            apply_rope_neox_with_tables(q, num_q_heads, head_dim, cos_table, sin_table)?;
            apply_rope_neox_with_tables(k, num_kv_heads, head_dim, cos_table, sin_table)
        })?;

        // Attention may read this provisional position, but later projection,
        // adapter or numerical failure must not publish it for the next call.
        let append = kv_append::KvAppend::new(kv, k, v)?;
        let kv = append.cache();

        let total_seq = kv.seq_len;
        let kv_stride = num_kv_heads * head_dim;
        let gqa_ratio = num_q_heads / num_kv_heads;
        let scale = 1.0_f32 / usize_to_f32(head_dim).sqrt();

        decoder_workspace::resize(attn_out, query_width)?;
        attn_out.fill(0.0);
        let level = Level::new();

        attend_query_f16(
            level,
            q,
            kv,
            total_seq,
            kv_stride,
            num_q_heads,
            num_kv_heads,
            head_dim,
            gqa_ratio,
            scale,
            scores,
            attn_out,
            workers,
        )?;

        self.o_proj
            .forward_vec_into(attn_out, None, patch(&self.o_proj), workers, projected)?;
        decoder_workspace::resize(x_res, input.len())?;
        for ((dst, &a), &b) in x_res.iter_mut().zip(input.iter()).zip(projected.iter()) {
            *dst = a + b;
        }

        rms_norm_into(x_res, &self.ffn_norm_w, rms_eps, h_norm)?;
        let gate_sig = self.gate_proj.hadamard_signature();
        let up_sig = self.up_proj.hadamard_signature();
        let shared_ffn_folded = if let Some((signs, block)) = gate_sig
            && up_sig == gate_sig
        {
            Some(apply_hadamard_sign_fold(h_norm, signs, block)?)
        } else {
            None
        };
        if shared_ffn_folded.is_none()
            && !matches!(
                self.gate_proj.weight,
                WeightQuantization::PQ2_0 { .. } | WeightQuantization::PTQ1_0 { .. }
            )
            && !matches!(
                self.up_proj.weight,
                WeightQuantization::PQ2_0 { .. } | WeightQuantization::PTQ1_0 { .. }
            )
        {
            LinearProjection::forward_pair_into(
                &self.gate_proj,
                &self.up_proj,
                h_norm,
                patch(&self.gate_proj),
                patch(&self.up_proj),
                workers,
                gate,
                up,
            )?;
        } else {
            self.gate_proj.forward_vec_into(
                h_norm,
                shared_ffn_folded.as_deref(),
                patch(&self.gate_proj),
                workers,
                gate,
            )?;
            self.up_proj.forward_vec_into(
                h_norm,
                shared_ffn_folded.as_deref(),
                patch(&self.up_proj),
                workers,
                up,
            )?;
        }
        swiglu_in_place(gate, up);
        self.down_proj
            .forward_vec_into(gate, None, patch(&self.down_proj), workers, projected)?;
        dispatch!(level, simd => axpy_simd(simd, 1.0_f32, projected, x_res));
        if x_res.iter().any(|value| !value.is_finite()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Nonfinite speech decoder output.",
            ));
        }
        append.commit();
        output.copy_from_slice(x_res);
        Ok(())
    }

    /// Evaluate `seq_len` causal prompt tokens in batched GEMM mode (`seq_len >= 1`).
    ///
    /// When `final_ffn_last_only` is `true`, only the last token (`seq_len - 1`) executes the
    /// MLP/SwiGLU block because downstream logits read only that final hidden vector.
    #[expect(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        reason = "Batched prefill executes GEMM QKV/O/FFN, causal RoPE/attention, and atomic KV commits."
    )]
    pub(crate) fn forward_batch_into(
        &self,
        inputs: &[f32],
        seq_len: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope_theta: f32,
        rms_eps: f32,
        kv: &mut LayerKvCache,
        adapter: Option<&TalkerLoraAdapter>,
        workers: usize,
        final_ffn_last_only: bool,
        output: &mut Vec<f32>,
    ) -> io::Result<()> {
        let invalid_geometry =
            || io::Error::new(io::ErrorKind::InvalidInput, "Invalid GQA geometry.");
        let hidden = self.attn_norm_w.len();
        if seq_len == 0
            || hidden == 0
            || inputs.len() != seq_len.checked_mul(hidden).ok_or_else(invalid_geometry)?
        {
            return Err(invalid_geometry());
        }
        if seq_len == 1 {
            output.resize(hidden, 0.0);
            let mut ws = decoder_workspace::DecoderWorkspace::default();
            return self.forward_step_into(
                inputs,
                num_q_heads,
                num_kv_heads,
                head_dim,
                rope_theta,
                rms_eps,
                kv,
                adapter,
                workers,
                &mut ws,
                output,
            );
        }
        let (query_width, kv_width) = self.validate_geometry(
            hidden,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rope_theta,
            rms_eps,
            kv,
            workers,
        )?;
        let initial_seq = kv.seq_len;
        if initial_seq.checked_add(seq_len).is_none() || seq_len.checked_mul(kv_width).is_none() {
            return Err(invalid_geometry());
        }

        let mut h_norm = Vec::new();
        rms_norm_batch_into(
            inputs,
            seq_len,
            &self.attn_norm_w,
            rms_eps,
            workers,
            &mut h_norm,
        )?;

        let patch = |proj: &LinearProjection| adapter.and_then(|a| a.patches.get(&proj.name));
        let mut q_all = Vec::new();
        let mut k_all = Vec::new();
        let mut v_all = Vec::new();
        LinearProjection::forward_batch_group_into(
            &mut [
                (&self.q_proj, patch(&self.q_proj), &mut q_all),
                (&self.k_proj, patch(&self.k_proj), &mut k_all),
                (&self.v_proj, patch(&self.v_proj), &mut v_all),
            ],
            &h_norm,
            seq_len,
            workers,
        )?;

        with_rope_batch_tables(
            head_dim,
            initial_seq,
            seq_len,
            rope_theta,
            |t, cos_buf, sin_buf| {
                let q_tok = &mut q_all[t * query_width..(t + 1) * query_width];
                normalize_heads_in_place(q_tok, num_q_heads, head_dim, &self.q_norm_w, rms_eps)?;
                let k_tok = &mut k_all[t * kv_width..(t + 1) * kv_width];
                normalize_heads_in_place(k_tok, num_kv_heads, head_dim, &self.k_norm_w, rms_eps)?;
                apply_rope_neox_with_tables(q_tok, num_q_heads, head_dim, cos_buf, sin_buf)?;
                apply_rope_neox_with_tables(k_tok, num_kv_heads, head_dim, cos_buf, sin_buf)?;
                Ok(())
            },
        )?;

        let append = kv_append::KvAppend::new_batch(kv, seq_len, &k_all, &v_all)?;
        let kv = append.cache();

        let level = Level::new();
        let gqa_ratio = num_q_heads / num_kv_heads;
        let scale = 1.0_f32 / usize_to_f32(head_dim).sqrt();
        let mut attn_seq = vec![0.0_f32; seq_len * query_width];
        attend_batch_f16(
            level,
            &q_all,
            kv,
            initial_seq,
            seq_len,
            kv_width,
            num_q_heads,
            num_kv_heads,
            head_dim,
            gqa_ratio,
            scale,
            &mut attn_seq,
            workers,
        )?;

        let mut projected = Vec::new();
        self.o_proj.forward_batch_into(
            &attn_seq,
            seq_len,
            patch(&self.o_proj),
            workers,
            &mut projected,
        )?;
        output.resize(inputs.len(), 0.0);
        for ((dst, &inp), &proj) in output.iter_mut().zip(inputs).zip(&projected) {
            *dst = inp + proj;
        }

        if final_ffn_last_only {
            let last_start = (seq_len - 1) * hidden;
            let mut row_norm = Vec::with_capacity(hidden);
            rms_norm_into(
                &output[last_start..],
                &self.ffn_norm_w,
                rms_eps,
                &mut row_norm,
            )?;
            let mut gate = Vec::new();
            let mut up = Vec::new();
            LinearProjection::forward_pair_into(
                &self.gate_proj,
                &self.up_proj,
                &row_norm,
                patch(&self.gate_proj),
                patch(&self.up_proj),
                workers,
                &mut gate,
                &mut up,
            )?;
            swiglu_in_place(&mut gate, &up);
            self.down_proj.forward_vec_into(
                &gate,
                None,
                patch(&self.down_proj),
                workers,
                &mut projected,
            )?;
            dispatch!(level, simd => axpy_simd(simd, 1.0_f32, &projected, &mut output[last_start..]));
        } else {
            rms_norm_batch_into(
                output,
                seq_len,
                &self.ffn_norm_w,
                rms_eps,
                workers,
                &mut h_norm,
            )?;
            let mut gate_all = Vec::new();
            let mut up_all = Vec::new();
            LinearProjection::forward_batch_group_into(
                &mut [
                    (&self.gate_proj, patch(&self.gate_proj), &mut gate_all),
                    (&self.up_proj, patch(&self.up_proj), &mut up_all),
                ],
                &h_norm,
                seq_len,
                workers,
            )?;
            swiglu_in_place_parallel(&mut gate_all, &up_all, workers);
            self.down_proj.forward_batch_into(
                &gate_all,
                seq_len,
                patch(&self.down_proj),
                workers,
                &mut projected,
            )?;
            dispatch!(level, simd => axpy_simd(simd, 1.0_f32, &projected, output));
        }

        if output.iter().any(|v| !v.is_finite()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Nonfinite speech batched decoder output.",
            ));
        }
        append.commit();
        Ok(())
    }
}

#[inline(always)]
pub(crate) fn exp_f32x4<S: Simd>(simd: S, x: f32x4<S>) -> f32x4<S> {
    // Cody-Waite range reduction by ln(2) = LN2_HI + LN2_LO (15-bit HI so k * LN2_HI is exact
    // for |k| <= 512) paired with a degree-6 minimax polynomial on [-ln(2)/2, ln(2)/2] with
    // exact c0 = 1.0 and c1 = 1.0 so exp(0.0) == 1.0 exactly.
    const INV_LN2: f32 = std::f32::consts::LOG2_E;
    const LN2_HI: f32 = f32::from_bits(0x3f31_7200);
    const LN2_LO: f32 = f32::from_bits(0x35bf_be8e);
    const MIN_NORMAL_EXP_INPUT: f32 = -87.336_54_f32;
    const C2: f32 = 0.5_f32;
    const C3: f32 = 0.166_664_17_f32;
    const C4: f32 = 0.041_666_668_f32;
    const C5: f32 = 0.008_375_039_f32;
    const C6: f32 = 0.001_388_888_9_f32;

    let zero = f32x4::splat(simd, 0.0_f32);
    let one = f32x4::splat(simd, 1.0_f32);
    let min_x = f32x4::splat(simd, MIN_NORMAL_EXP_INPUT);
    let is_underflow = x.simd_lt(min_x);
    let x_safe = is_underflow.select(zero, x);

    let kf = (x_safe * f32x4::splat(simd, INV_LN2)).round_ties_even();
    let r = f32x4::splat(simd, -LN2_HI).mul_add(kf, x_safe);
    let r = f32x4::splat(simd, -LN2_LO).mul_add(kf, r);

    let p = f32x4::splat(simd, C6).mul_add(r, f32x4::splat(simd, C5));
    let p = p.mul_add(r, f32x4::splat(simd, C4));
    let p = p.mul_add(r, f32x4::splat(simd, C3));
    let p = p.mul_add(r, f32x4::splat(simd, C2));
    let poly = p.mul_add(r * r, r) + one;

    let k_i32 = i32x4::truncate_from(kf);
    let pow2 = ((k_i32 + 127) << 23_u32).bitcast::<f32x4<S>>();
    is_underflow.select(zero, poly * pow2)
}

#[expect(
    clippy::inline_always,
    reason = "fearless_simd dispatch! requires inline(always) to inherit target_feature attributes."
)]
#[inline(always)]
fn softmax_row_simd<S: Simd>(simd: S, scores: &mut [f32], max_score: f32) -> f32 {
    let max_v = f32x4::splat(simd, max_score);
    let mut sum_v = f32x4::splat(simd, 0.0_f32);
    let (chunks, tail) = scores.as_chunks_mut::<4>();
    for chunk in chunks {
        let sv = f32x4::from_slice(simd, chunk);
        let ev = exp_f32x4(simd, sv - max_v);
        ev.store_slice(chunk);
        sum_v += ev;
    }
    let mut lanes = [0.0_f32; 4];
    sum_v.store_slice(&mut lanes);
    let mut sum_exp = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
    for s in tail {
        *s = (*s - max_score).exp();
        sum_exp += *s;
    }
    1.0_f32 / sum_exp.max(1e-12_f32)
}

#[expect(
    clippy::too_many_arguments,
    reason = "Executes a contiguous range of KV head groups using caller-local scratch."
)]
fn attend_kv_group_range_f16(
    level: Level,
    q: &[f32],
    kv: &LayerKvCache,
    total_seq: usize,
    kv_stride: usize,
    kvh_start: usize,
    kvh_count: usize,
    head_dim: usize,
    gqa_ratio: usize,
    scale: f32,
    scores: &mut [f32],
    out_chunk: &mut [f32],
) {
    let mut stack_buf = [0.0_f32; 256];
    let mut heap_buf = if head_dim > stack_buf.len() {
        vec![0.0_f32; head_dim]
    } else {
        Vec::new()
    };
    let head_buf: &mut [f32] = if head_dim <= stack_buf.len() {
        &mut stack_buf[..head_dim]
    } else {
        &mut heap_buf[..head_dim]
    };

    out_chunk.fill(0.0);
    if gqa_ratio <= 8 {
        dispatch!(level, simd => {
            for local_kvh in 0..kvh_count {
                let kvh = kvh_start + local_kvh;
                let kv_head_off = kvh * head_dim;
                let qh_base = kvh * gqa_ratio;
                let mut max_scores = [f32::NEG_INFINITY; 8];
                for t in 0..total_seq {
                    let k_bits = &kv.k[t * kv_stride + kv_head_off..t * kv_stride + kv_head_off + head_dim];
                    widen_f16_slice(k_bits, head_buf);
                    for (g, max_slot) in max_scores[..gqa_ratio].iter_mut().enumerate() {
                        let qh = qh_base + g;
                        let q_vec = &q[qh * head_dim..(qh + 1) * head_dim];
                        let s = f32_dot_simd(simd, q_vec, head_buf) * scale;
                        scores[g * total_seq + t] = s;
                        if s > *max_slot {
                            *max_slot = s;
                        }
                    }
                }
                let mut inv_sums = [0.0_f32; 8];
                for g in 0..gqa_ratio {
                    inv_sums[g] = softmax_row_simd(
                        simd,
                        &mut scores[g * total_seq..(g + 1) * total_seq],
                        max_scores[g],
                    );
                }
                for t in 0..total_seq {
                    let v_bits = &kv.v[t * kv_stride + kv_head_off..t * kv_stride + kv_head_off + head_dim];
                    widen_f16_slice(v_bits, head_buf);
                    for (g, &inv_sum) in inv_sums[..gqa_ratio].iter().enumerate() {
                        let local_qh = local_kvh * gqa_ratio + g;
                        let w = scores[g * total_seq + t] * inv_sum;
                        let out_head = &mut out_chunk[local_qh * head_dim..(local_qh + 1) * head_dim];
                        axpy_simd(simd, w, head_buf, out_head);
                    }
                }
            }
        });
    } else {
        let scores = &mut scores[..total_seq];
        dispatch!(level, simd => {
            for local_kvh in 0..kvh_count {
                let kvh = kvh_start + local_kvh;
                let kv_head_off = kvh * head_dim;
                let qh_base = kvh * gqa_ratio;
                for g in 0..gqa_ratio {
                    let qh = qh_base + g;
                    let local_qh = local_kvh * gqa_ratio + g;
                    let q_vec = &q[qh * head_dim..(qh + 1) * head_dim];
                    let mut max_score = f32::NEG_INFINITY;
                    for (t, score_slot) in scores.iter_mut().enumerate() {
                        let k_bits = &kv.k[t * kv_stride + kv_head_off..t * kv_stride + kv_head_off + head_dim];
                        widen_f16_slice(k_bits, head_buf);
                        let s = f32_dot_simd(simd, q_vec, head_buf) * scale;
                        *score_slot = s;
                        if s > max_score {
                            max_score = s;
                        }
                    }
                    let inv_sum = softmax_row_simd(simd, scores, max_score);
                    let out_head = &mut out_chunk[local_qh * head_dim..(local_qh + 1) * head_dim];
                    for (t, &prob_unnorm) in scores.iter().enumerate() {
                        let w = prob_unnorm * inv_sum;
                        let v_bits = &kv.v[t * kv_stride + kv_head_off..t * kv_stride + kv_head_off + head_dim];
                        widen_f16_slice(v_bits, head_buf);
                        axpy_simd(simd, w, head_buf, out_head);
                    }
                }
            }
        });
    }
}

thread_local! {
    static ATTN_SCORES: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

fn with_attn_scores<R>(needed: usize, action: impl FnOnce(&mut [f32]) -> R) -> R {
    ATTN_SCORES.with(|cell| {
        if let Ok(mut buf) = cell.try_borrow_mut() {
            if buf.len() < needed {
                buf.resize(needed, 0.0_f32);
            }
            action(&mut buf[..needed])
        } else {
            let mut fallback = vec![0.0_f32; needed];
            action(&mut fallback)
        }
    })
}

#[expect(
    clippy::too_many_arguments,
    reason = "Shared F16 KV attention kernel used by both single-step decode and batched prefill."
)]
fn attend_query_f16(
    level: Level,
    q: &[f32],
    kv: &LayerKvCache,
    total_seq: usize,
    kv_stride: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    gqa_ratio: usize,
    scale: f32,
    scores: &mut Vec<f32>,
    attn_out: &mut [f32],
    workers: usize,
) -> io::Result<()> {
    let needed_scores = if gqa_ratio <= 8 {
        gqa_ratio * total_seq
    } else {
        total_seq
    };
    if workers > 1
        && num_kv_heads > 1
        && total_seq >= 64
        && attn_out.len() == num_q_heads * head_dim
    {
        let group_width = gqa_ratio * head_dim;
        row_workers::run_aligned(attn_out, workers, group_width, &|elem_off, out_chunk| {
            let kvh_start = elem_off / group_width;
            let kvh_count = out_chunk.len() / group_width;
            with_attn_scores(needed_scores, |local_scores| {
                attend_kv_group_range_f16(
                    level,
                    q,
                    kv,
                    total_seq,
                    kv_stride,
                    kvh_start,
                    kvh_count,
                    head_dim,
                    gqa_ratio,
                    scale,
                    local_scores,
                    out_chunk,
                );
            });
        });
    } else {
        decoder_workspace::resize(scores, needed_scores)?;
        attend_kv_group_range_f16(
            level,
            q,
            kv,
            total_seq,
            kv_stride,
            0,
            num_kv_heads,
            head_dim,
            gqa_ratio,
            scale,
            scores,
            attn_out,
        );
    }
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "Executes causal F16 KV attention across prefill tokens in one worker pass with thread-local score reuse."
)]
fn attend_batch_f16(
    level: Level,
    q_all: &[f32],
    kv: &LayerKvCache,
    initial_seq: usize,
    seq_len: usize,
    kv_stride: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    gqa_ratio: usize,
    scale: f32,
    attn_seq: &mut [f32],
    workers: usize,
) -> io::Result<()> {
    let query_width = num_q_heads * head_dim;
    let group_width = gqa_ratio * head_dim;
    let max_seq = initial_seq + seq_len;
    let max_needed = if gqa_ratio <= 8 {
        gqa_ratio * max_seq
    } else {
        max_seq
    };
    if workers > 1 && seq_len * num_kv_heads >= 2 && attn_seq.len() == seq_len * query_width {
        row_workers::run_aligned(attn_seq, workers, group_width, &|elem_off, out_chunk| {
            let first_group = elem_off / group_width;
            let groups_here = out_chunk.len() / group_width;
            with_attn_scores(max_needed, |scores| {
                for local_g in 0..groups_here {
                    let flat_group = first_group + local_g;
                    let t = flat_group / num_kv_heads;
                    let kvh = flat_group % num_kv_heads;
                    let total_seq_t = initial_seq + t + 1;
                    let needed_t = if gqa_ratio <= 8 {
                        gqa_ratio * total_seq_t
                    } else {
                        total_seq_t
                    };
                    let q_tok = &q_all[t * query_width..(t + 1) * query_width];
                    let dst_group =
                        &mut out_chunk[local_g * group_width..(local_g + 1) * group_width];
                    attend_kv_group_range_f16(
                        level,
                        q_tok,
                        kv,
                        total_seq_t,
                        kv_stride,
                        kvh,
                        1,
                        head_dim,
                        gqa_ratio,
                        scale,
                        &mut scores[..needed_t],
                        dst_group,
                    );
                }
            });
        });
    } else {
        with_attn_scores(max_needed, |scores| {
            for t in 0..seq_len {
                let total_seq_t = initial_seq + t + 1;
                let needed_t = if gqa_ratio <= 8 {
                    gqa_ratio * total_seq_t
                } else {
                    total_seq_t
                };
                let q_tok = &q_all[t * query_width..(t + 1) * query_width];
                let out_tok = &mut attn_seq[t * query_width..(t + 1) * query_width];
                attend_kv_group_range_f16(
                    level,
                    q_tok,
                    kv,
                    total_seq_t,
                    kv_stride,
                    0,
                    num_kv_heads,
                    head_dim,
                    gqa_ratio,
                    scale,
                    &mut scores[..needed_t],
                    out_tok,
                );
            }
        });
    }
    Ok(())
}

/// Pure-Rust 5-layer `CodePredictor` predicting acoustic codebooks `1..=15` per frame.
#[derive(Clone, Debug)]
pub struct CodePredictorModel {
    /// Predictor hidden size (`1024`).
    pub hidden_size: usize,
    /// Attention query heads (`16`).
    pub num_q_heads: usize,
    /// Attention KV heads (`8`).
    pub num_kv_heads: usize,
    /// Per-head dimension (`128`).
    pub head_dim: usize,
    /// Codebook vocabulary size (`2048`).
    pub vocab_size: usize,
    /// Number of acoustic codebooks (`15`).
    pub num_acoustic_codebooks: usize,
    /// `RoPE` base frequency (`1_000_000.0`).
    pub rope_theta: f32,
    /// `RMSNorm` epsilon (`1e-6`).
    pub rms_norm_eps: f32,
    /// Optional `mtp_proj` (`talker_hidden -> predictor_hidden` when `2048 != 1024`).
    pub mtp_proj: Option<LinearProjection>,
    /// 5 decoder layers.
    pub layers: Vec<QwenDecoderLayer>,
    /// Final `RMSNorm` weight `[hidden_size]`.
    pub norm_w: Vec<f32>,
    /// 15 private acoustic embedding tables (`[vocab_size, talker_hidden]`).
    pub codec_embeddings: Vec<LinearProjection>,
    /// 15 private acoustic LM heads (`[vocab_size, hidden_size]`).
    pub lm_heads: Vec<LinearProjection>,
}

impl CodePredictorModel {
    fn project_in_parallel_into(
        &self,
        raw: &[f32],
        workers: usize,
        out: &mut Vec<f32>,
    ) -> io::Result<()> {
        if let Some(proj) = &self.mtp_proj {
            proj.forward_vec_into(raw, None, None, workers, out)
        } else {
            decoder_workspace::resize(out, raw.len())?;
            out.copy_from_slice(raw);
            Ok(())
        }
    }
    /// Predict all 15 acoustic codes (`c1..=c15`) for one frame given `talker_hidden`,
    /// semantic `c0`, `talker_codec_embd`, and decoupled `subtalker_*` sampling parameters.
    /// Returns `[c0, c1, ..., c15]`.
    pub fn predict_frame(
        &self,
        talker_hidden: &[f32],
        c0: i32,
        talker_codec_embd: &LinearProjection,
        params: DecoupledSamplingParams,
        seed: i64,
        subseq_base: i64,
    ) -> io::Result<[i32; TOTAL_CODE_GROUPS]> {
        self.predict_frame_with_control(
            talker_hidden,
            c0,
            talker_codec_embd,
            params,
            seed,
            subseq_base,
            1,
            &mut || Ok(()),
        )
    }

    /// Predict one frame with explicit CPU workers and cancellation between layers/codebooks.
    #[expect(
        clippy::too_many_arguments,
        reason = "Retain the independently selected sampler, seed, worker and cancellation controls."
    )]
    pub fn predict_frame_with_control(
        &self,
        talker_hidden: &[f32],
        c0: i32,
        talker_codec_embd: &LinearProjection,
        params: DecoupledSamplingParams,
        seed: i64,
        subseq_base: i64,
        workers: usize,
        cancel: &mut (impl FnMut() -> io::Result<()> + ?Sized),
    ) -> io::Result<[i32; TOTAL_CODE_GROUPS]> {
        self.predict_frame_reusing(
            talker_hidden,
            c0,
            talker_codec_embd,
            params,
            seed,
            subseq_base,
            workers,
            cancel,
            &mut predictor_workspace::PredictorWorkspace::default(),
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "Share request-owned capacity with Talker while retaining the explicit independent sampling and cancellation inputs."
    )]
    pub(crate) fn predict_frame_reusing(
        &self,
        talker_hidden: &[f32],
        c0: i32,
        talker_codec_embd: &LinearProjection,
        params: DecoupledSamplingParams,
        seed: i64,
        subseq_base: i64,
        workers: usize,
        cancel: &mut (impl FnMut() -> io::Result<()> + ?Sized),
        workspace: &mut predictor_workspace::PredictorWorkspace,
    ) -> io::Result<[i32; TOTAL_CODE_GROUPS]> {
        self.predict_frame_observed_reusing(
            talker_hidden,
            c0,
            talker_codec_embd,
            params,
            seed,
            subseq_base,
            workers,
            cancel,
            workspace,
            &mut |_| Ok(()),
        )
    }

    #[expect(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        reason = "Preserve serving inputs while borrowing complete state for independent qualification."
    )]
    pub(crate) fn predict_frame_observed_reusing(
        &self,
        talker_hidden: &[f32],
        c0: i32,
        talker_codec_embd: &LinearProjection,
        params: DecoupledSamplingParams,
        seed: i64,
        subseq_base: i64,
        workers: usize,
        cancel: &mut (impl FnMut() -> io::Result<()> + ?Sized),
        workspace: &mut predictor_workspace::PredictorWorkspace,
        observe: &mut (
                 impl FnMut(crate::observation::PredictorObservation<'_>) -> io::Result<()> + ?Sized
             ),
    ) -> io::Result<[i32; TOTAL_CODE_GROUPS]> {
        use crate::observation::PredictorObservation;

        // Clear logical state even when admission below rejects. The guard also
        // clears partial frames on cancellation, numerical errors and unwind.
        let mut frame = workspace.frame(self.layers.len())?;
        let (kv_caches, sampler_scratch, decoder_workspace, next_raw, h_a, h_b, normed, logits) =
            frame.buffers();
        if !(1..=256).contains(&workers) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Invalid CPU worker count.",
            ));
        }
        if self.num_acoustic_codebooks != NUM_ACOUSTIC_CODEBOOKS
            || self.codec_embeddings.len() != NUM_ACOUSTIC_CODEBOOKS
            || self.lm_heads.len() != NUM_ACOUSTIC_CODEBOOKS
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Invalid CodePredictor codebook table count.",
            ));
        }
        let talker_dim = talker_hidden.len();
        if talker_codec_embd.in_dim != talker_dim {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "talker.codec_embd hidden size mismatch.",
            ));
        }
        let c0_usize = usize::try_from(c0).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidInput, "Negative semantic code c0.")
        })?;
        decoder_workspace::resize(next_raw, talker_dim)?;
        talker_codec_embd
            .row_into(c0_usize, next_raw)
            .map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "Semantic code c0 out of range for talker.codec_embd.",
                )
            })?;

        let mut codes = [0_i32; TOTAL_CODE_GROUPS];
        codes[0] = c0;

        // Step 0 prefill: position 0 = talker_hidden, position 1 = embed(c0)
        cancel()?;
        self.project_in_parallel_into(talker_hidden, workers, h_a)?;
        decoder_workspace::resize(h_b, h_a.len())?;
        if let Some(first_kv) = kv_caches.first() {
            prefetch_rope_step(self.head_dim, first_kv.seq_len, self.rope_theta);
        }
        for (layer, kv) in self.layers.iter().zip(kv_caches.iter_mut()) {
            cancel()?;
            layer.forward_step_into(
                h_a,
                self.num_q_heads,
                self.num_kv_heads,
                self.head_dim,
                self.rope_theta,
                self.rms_norm_eps,
                kv,
                None,
                workers,
                decoder_workspace,
                h_b,
            )?;
            std::mem::swap(h_a, h_b);
        }

        observe(PredictorObservation::Prefill { cache: kv_caches })?;
        for g in 0..NUM_ACOUSTIC_CODEBOOKS {
            cancel()?;
            self.project_in_parallel_into(next_raw, workers, h_a)?;
            decoder_workspace::resize(h_b, h_a.len())?;
            if let Some(first_kv) = kv_caches.first() {
                prefetch_rope_step(self.head_dim, first_kv.seq_len, self.rope_theta);
            }
            for (layer, kv) in self.layers.iter().zip(kv_caches.iter_mut()) {
                cancel()?;
                layer.forward_step_into(
                    h_a,
                    self.num_q_heads,
                    self.num_kv_heads,
                    self.head_dim,
                    self.rope_theta,
                    self.rms_norm_eps,
                    kv,
                    None,
                    workers,
                    decoder_workspace,
                    h_b,
                )?;
                std::mem::swap(h_a, h_b);
            }
            rms_norm_into(h_a, &self.norm_w, self.rms_norm_eps, normed)?;
            self.lm_heads[g].forward_vec_into(normed, None, None, workers, logits)?;
            observe(PredictorObservation::Logits {
                codebook: g,
                hidden: normed,
                logits,
                cache: kv_caches,
            })?;
            cancel()?;
            let sampled_idx = sampling::sample_top_k_p_with_scratch(
                logits,
                sampling::SampleTopKPParams {
                    temperature: params.subtalker_temperature,
                    top_k: params.subtalker_top_k,
                    top_p: params.subtalker_top_p,
                    repetition_penalty: 1.0,
                    seed,
                    philox_subseq: subseq_base
                        .wrapping_add(1)
                        .wrapping_add(i64::try_from(g).unwrap_or(0)),
                },
                &[],
                sampler_scratch,
                None,
            )
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "Acoustic code sampling failed.")
            })?;
            codes[g + 1] = i32::try_from(sampled_idx).map_err(|_| {
                io::Error::new(io::ErrorKind::InvalidData, "Invalid sampled acoustic code.")
            })?;
            observe(PredictorObservation::Decision {
                codebook: g,
                token: codes[g + 1],
            })?;

            if g + 1 < NUM_ACOUSTIC_CODEBOOKS {
                self.codec_embeddings[g]
                    .row_into(sampled_idx, next_raw)
                    .map_err(|_| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            "Acoustic code index exceeds embedding table.",
                        )
                    })?;
            }
        }
        Ok(codes)
    }
}

/// Cached ICL reference streams (`icl_codec_stream` and `icl_ref_text_stream`) in `PromptCache`.
#[derive(Clone, Debug)]
pub struct CachedIclReference {
    /// FNV-1a 64-bit content hash of the `[16, ref_codes_T]` reference codes.
    pub ref_key: u64,
    /// Pre-summed codec stream `[ (1 + ref_codes_T) * hidden_size ]` (`codec_bos` + summed 16 codebooks).
    pub icl_codec_stream: Vec<f32>,
    /// Pre-projected reference text stream `[ N_ref_text * hidden_size ]`.
    pub icl_ref_text_stream: Vec<f32>,
    /// Number of inner reference text tokens (`N_ref_text`).
    pub n_ref_text: usize,
    /// Number of reference audio frames (`ref_codes_t`).
    pub ref_codes_t: usize,
}

/// Pure-Rust `PromptCache` retaining special token projections and cached ICL reference streams.
#[derive(Clone, Debug, Default)]
pub struct PromptCache {
    /// Projected `tts_bos` `[hidden_size]`.
    pub tts_bos_emb: Vec<f32>,
    /// Projected `tts_eos` `[hidden_size]`.
    pub tts_eos_emb: Vec<f32>,
    /// Projected `tts_pad` `[hidden_size]`.
    pub tts_pad_emb: Vec<f32>,
    /// `codec_pad` embedding `[hidden_size]`.
    pub codec_pad_emb: Vec<f32>,
    /// `codec_bos` embedding `[hidden_size]`.
    pub codec_bos_emb: Vec<f32>,
    /// Most-recent cached ICL reference streams so repeated `Base` synthesis skips re-projecting `REFERENCE_TEXT` and `ref_codes`.
    pub cached_icl: Option<CachedIclReference>,
}

impl PromptCache {
    /// Return or compute and cache the ICL reference codec and text streams for `ref_codes`.
    pub fn get_or_build_icl_reference(
        &mut self,
        ref_codes: &[i32],
        ref_codes_t: usize,
        compute_codec_stream: impl FnOnce() -> io::Result<Vec<f32>>,
        compute_ref_text_stream: impl FnOnce() -> io::Result<(Vec<f32>, usize)>,
    ) -> io::Result<&CachedIclReference> {
        let key = codec_ref_key(ref_codes, TOTAL_CODE_GROUPS, ref_codes_t);
        if let Some(cached) = self.cached_icl.as_ref()
            && cached.ref_key == key
            && cached.ref_codes_t == ref_codes_t
        {
            return Ok(cached);
        }
        let icl_codec_stream = compute_codec_stream()?;
        let (icl_ref_text_stream, n_ref_text) = compute_ref_text_stream()?;
        Ok(self.cached_icl.insert(CachedIclReference {
            ref_key: key,
            icl_codec_stream,
            icl_ref_text_stream,
            n_ref_text,
            ref_codes_t,
        }))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use fearless_simd::SimdBase;

    use super::{
        CodePredictorModel, DecoupledSamplingParams, LayerKvCache, LinearProjection,
        LoraProjectionPatch, NUM_ACOUSTIC_CODEBOOKS, PromptCache, QwenDecoderLayer,
        TalkerLoraAdapter, WeightQuantization, is_talker_ternary_eligible, repack_pq2_to_ptq1,
        sylvester_walsh_hadamard_in_place, validate_talker_ternary_policy,
    };

    #[test]
    fn walsh_hadamard_is_involutory_and_pq2_ptq1_repack_is_bit_exact() -> std::io::Result<()> {
        let original = vec![0.5_f32, -1.25, 2.0, 0.75, -0.5, 1.5, -2.25, 0.125];
        let mut buf = original.clone();
        sylvester_walsh_hadamard_in_place(&mut buf)?;
        sylvester_walsh_hadamard_in_place(&mut buf)?;
        for (a, b) in original.iter().zip(buf.iter()) {
            assert!((a - b).abs() < 1e-6, "a={a} b={b}");
        }

        // 8 weights, group_size = 8, 2 output rows
        // Row 0 trits: [-1, 0, +1, +1, 0, -1, +1, 0] -> codes [0, 1, 2, 2], [1, 0, 2, 1]
        let b0 = (1_u8 << 2) | (2_u8 << 4) | (2_u8 << 6);
        let b1 = 1_u8 | (2_u8 << 4) | (1_u8 << 6);
        let pq2 = WeightQuantization::PQ2_0 {
            group_size: 8,
            scales: vec![0.5, 0.25],
            packed_trits: vec![b0, b1, b1, b0],
            hadamard_block: 8,
            signs: vec![1, -1, 1, 1, -1, 1, -1, 1],
        };
        let ptq1 = repack_pq2_to_ptq1(&pq2)?;

        let proj_pq2 = LinearProjection::new(
            "talker.blk.0.attn_q.weight",
            8,
            2,
            pq2,
            Some(vec![0.1, -0.1]),
        )?;
        let proj_ptq1 = LinearProjection::new(
            "talker.blk.0.attn_q.weight",
            8,
            2,
            ptq1,
            Some(vec![0.1, -0.1]),
        )?;
        let out_pq2 = proj_pq2.forward_vec(&original, None)?;
        let out_ptq1 = proj_ptq1.forward_vec(&original, None)?;
        assert_eq!(
            out_pq2, out_ptq1,
            "PTQ1_0 repack must be bit-exact with PQ2_0"
        );

        // Exercise 16-element SIMD group loops (group_size = 32, in_dim = 64, 2 output rows).
        let input64: Vec<f32> = (0_u8..64).map(|i| (f32::from(i) - 31.5) * 0.0625).collect();
        let packed64: Vec<u8> = (0_u8..32)
            .map(|i| {
                let c0 = i % 3;
                let c1 = (i + 1) % 3;
                let c2 = (i + 2) % 3;
                let c3 = (i * 2) % 3;
                c0 | (c1 << 2) | (c2 << 4) | (c3 << 6)
            })
            .collect();
        let signs64: Vec<i8> = (0_u8..64)
            .map(|i| if i.is_multiple_of(2) { 1 } else { -1 })
            .collect();
        let pq2_64 = WeightQuantization::PQ2_0 {
            group_size: 32,
            scales: vec![0.375, -0.625, 0.5, 0.25],
            packed_trits: packed64,
            hadamard_block: 16,
            signs: signs64,
        };
        let ptq1_64 = repack_pq2_to_ptq1(&pq2_64)?;
        let proj_pq2_64 = LinearProjection::new(
            "talker.blk.0.attn_k.weight",
            64,
            2,
            pq2_64,
            Some(vec![0.125, -0.25]),
        )?;
        let proj_ptq1_64 = LinearProjection::new(
            "talker.blk.0.attn_k.weight",
            64,
            2,
            ptq1_64,
            Some(vec![0.125, -0.25]),
        )?;
        assert_eq!(
            proj_pq2_64.forward_vec(&input64, None)?,
            proj_ptq1_64.forward_vec(&input64, None)?,
            "16-wide SIMD PTQ1_0 and PQ2_0 kernels must be bit-identical"
        );
        Ok(())
    }

    #[test]
    fn ternary_policy_admits_only_talker_block_projections() {
        let dummy_pq2 = WeightQuantization::PQ2_0 {
            group_size: 8,
            scales: vec![1.0],
            packed_trits: vec![0, 0],
            hadamard_block: 8,
            signs: vec![1; 8],
        };
        assert!(is_talker_ternary_eligible("talker.blk.0.attn_q.weight"));
        assert!(is_talker_ternary_eligible("talker.blk.27.ffn_down.weight"));
        assert!(
            validate_talker_ternary_policy("talker.blk.12.attn_output.weight", &dummy_pq2).is_ok()
        );

        for forbidden in [
            "talker.text_embd.weight",
            "talker.codec_embd.weight",
            "talker.text_proj.fc1.weight",
            "talker.codec_head.weight",
            "code_pred.blk.0.attn_q.weight",
            "code_pred.lm_head.0.weight",
            "tok_dec.pre_conv.weight",
            "spk_enc.conv0.weight",
        ] {
            assert!(!is_talker_ternary_eligible(forbidden));
            assert!(validate_talker_ternary_policy(forbidden, &dummy_pq2).is_err());
        }
    }

    #[test]
    fn lora_overlay_preserves_zero_scale_bit_identity_and_applies_nonzero_delta()
    -> std::io::Result<()> {
        let proj = LinearProjection::new(
            "talker.blk.0.attn_q.weight",
            4,
            2,
            WeightQuantization::F32(vec![
                0.5, -0.25, 0.1, 0.2, // row 0
                -0.1, 0.4, 0.3, -0.5, // row 1
            ]),
            Some(vec![0.05, -0.05]),
        )?;
        let x = [1.0_f32, -0.5, 0.25, 0.75];
        let base_out = proj.forward_vec(&x, None)?;

        let mut patch = LoraProjectionPatch {
            in_dim: 4,
            out_dim: 2,
            rank: 2,
            alpha_over_rank: 2.0,
            scale: 0.0,
            lora_a: vec![0.3, -0.2, 0.4, 0.1, -0.5, 0.2, 0.1, 0.3],
            lora_b: vec![0.2, -0.4, 0.5, 0.1],
        };
        let zero_out = proj.forward_vec(&x, Some(&patch))?;
        assert_eq!(
            base_out, zero_out,
            "Loaded-at-zero LoRA must be 0-ULP identical"
        );

        patch.scale = 1.0;
        let active_out = proj.forward_vec(&x, Some(&patch))?;
        assert_ne!(base_out, active_out);
        Ok(())
    }

    fn code_predictor_fixture() -> std::io::Result<CodePredictorModel> {
        let hidden = 4_usize;
        let vocab = 8_usize;
        let make_proj =
            |name: &str, in_d: usize, out_d: usize| -> std::io::Result<LinearProjection> {
                LinearProjection::new(
                    name,
                    in_d,
                    out_d,
                    WeightQuantization::F32(vec![0.05; out_d * in_d]),
                    None,
                )
            };
        let layer = QwenDecoderLayer {
            attn_norm_w: vec![1.0; hidden],
            q_proj: make_proj("code_pred.blk.0.attn_q.weight", hidden, hidden)?,
            k_proj: make_proj("code_pred.blk.0.attn_k.weight", hidden, hidden)?,
            v_proj: make_proj("code_pred.blk.0.attn_v.weight", hidden, hidden)?,
            o_proj: make_proj("code_pred.blk.0.attn_output.weight", hidden, hidden)?,
            q_norm_w: vec![1.0; 2],
            k_norm_w: vec![1.0; 2],
            ffn_norm_w: vec![1.0; hidden],
            gate_proj: make_proj("code_pred.blk.0.ffn_gate.weight", hidden, hidden * 2)?,
            up_proj: make_proj("code_pred.blk.0.ffn_up.weight", hidden, hidden * 2)?,
            down_proj: make_proj("code_pred.blk.0.ffn_down.weight", hidden * 2, hidden)?,
        };
        let mut lm_heads = Vec::with_capacity(NUM_ACOUSTIC_CODEBOOKS);
        for g in 0..NUM_ACOUSTIC_CODEBOOKS {
            lm_heads.push(make_proj(
                &format!("code_pred.lm_head.{g}.weight"),
                hidden,
                vocab,
            )?);
        }
        let mut codec_embeddings = Vec::with_capacity(NUM_ACOUSTIC_CODEBOOKS);
        for g in 0..NUM_ACOUSTIC_CODEBOOKS {
            codec_embeddings.push(LinearProjection::new(
                &format!("code_pred.codec_embd.{g}.weight"),
                hidden,
                vocab,
                WeightQuantization::F32(vec![0.1; vocab * hidden]),
                None,
            )?);
        }
        Ok(CodePredictorModel {
            hidden_size: hidden,
            num_q_heads: 2,
            num_kv_heads: 2,
            head_dim: 2,
            vocab_size: vocab,
            num_acoustic_codebooks: NUM_ACOUSTIC_CODEBOOKS,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
            mtp_proj: None,
            layers: vec![layer],
            norm_w: vec![1.0; hidden],
            codec_embeddings,
            lm_heads,
        })
    }

    #[test]
    fn decoder_kv_failed_step_restores_complete_state_and_fresh_replay() -> std::io::Result<()> {
        let model = code_predictor_fixture()?;
        let layer = &model.layers[0];
        let step = |layer: &QwenDecoderLayer, cache: &mut LayerKvCache, workers| {
            layer.forward_step_parallel(&[0.2; 4], 2, 2, 2, 1_000_000.0, 1e-6, cache, None, workers)
        };
        let bits = |cache: &LayerKvCache| (cache.seq_len, cache.k.clone(), cache.v.clone());
        let mut before = LayerKvCache::default();
        let _ = step(layer, &mut before, 1)?;
        let mut expected = before.clone();
        let expected_output = step(layer, &mut expected, 1)?;
        for workers in [1, 3] {
            for fault in 0..7 {
                let mut failed = layer.clone();
                let projection = match fault {
                    0 => &mut failed.o_proj,
                    1 => &mut failed.gate_proj,
                    2 => &mut failed.up_proj,
                    _ => &mut failed.down_proj,
                };
                if fault >= 5 {
                    projection.bias = Some(vec![0.0; if fault == 5 { 1 } else { 5 }]);
                } else if let WeightQuantization::F32(weights) = &mut projection.weight {
                    if fault == 4 {
                        weights.fill(f32::NAN);
                    } else {
                        let _ = weights.pop();
                    }
                }
                let mut cache = before.clone();
                assert!(step(&failed, &mut cache, workers).is_err(), "fault {fault}");
                assert_eq!(bits(&cache), bits(&before), "fault {fault}");
                let output = step(layer, &mut cache, workers)?;
                assert_eq!(bits(&cache), bits(&expected));
                assert_eq!(
                    output.iter().map(|n| n.to_bits()).collect::<Vec<_>>(),
                    expected_output
                        .iter()
                        .map(|n| n.to_bits())
                        .collect::<Vec<_>>()
                );
            }
        }
        Ok(())
    }

    #[test]
    fn decoder_kv_invalid_geometry_and_cache_reject_before_mutation() -> std::io::Result<()> {
        let model = code_predictor_fixture()?;
        let layer = &model.layers[0];
        for (heads, kv_heads, dimension) in [
            (0, 2, 2),
            (2, 0, 2),
            (2, 2, 0),
            (2, 2, 3),
            (usize::MAX, 2, 2),
            (4, 2, 2),
        ] {
            let mut cache = LayerKvCache::default();
            assert!(
                layer
                    .forward_step(
                        &[0.2; 4],
                        heads,
                        kv_heads,
                        dimension,
                        1_000_000.0,
                        1e-6,
                        &mut cache,
                        None
                    )
                    .is_err()
            );
            assert!(cache.k.is_empty() && cache.v.is_empty());
            assert_eq!(cache.seq_len, 0);
        }
        let mut cache = LayerKvCache {
            k: vec![0x8000],
            v: vec![0x3800],
            seq_len: 1,
        };
        assert!(
            layer
                .forward_step(&[0.2; 4], 2, 2, 2, 1_000_000.0, 1e-6, &mut cache, None)
                .is_err()
        );
        assert_eq!(cache.k, [0x8000_u16]);
        assert_eq!(cache.v, [0x3800_u16]);
        assert_eq!(cache.seq_len, 1);
        Ok(())
    }

    #[test]
    fn batched_prefill_matches_sequential_steps_and_final_ffn_last_row() -> std::io::Result<()> {
        let model = code_predictor_fixture()?;
        let layer = &model.layers[0];
        let inputs = [
            0.1_f32, -0.2, 0.3, 0.4, -0.5, 0.25, 0.15, -0.35, 0.2, 0.6, -0.1, 0.05,
        ];
        let seq_len = 3_usize;
        let mut seq_kv = LayerKvCache::default();
        let mut seq_out = Vec::with_capacity(inputs.len());
        for t in 0..seq_len {
            let step = layer.forward_step_parallel(
                &inputs[t * 4..(t + 1) * 4],
                2,
                2,
                2,
                1_000_000.0,
                1e-6,
                &mut seq_kv,
                None,
                3,
            )?;
            seq_out.extend_from_slice(&step);
        }

        let mut batch_kv = LayerKvCache::default();
        let mut batch_out = Vec::new();
        layer.forward_batch_into(
            &inputs,
            seq_len,
            2,
            2,
            2,
            1_000_000.0,
            1e-6,
            &mut batch_kv,
            None,
            3,
            false,
            &mut batch_out,
        )?;
        assert_eq!(batch_kv.seq_len, seq_kv.seq_len);
        assert_eq!(batch_kv.k, seq_kv.k);
        assert_eq!(batch_kv.v, seq_kv.v);
        for (a, b) in batch_out.iter().zip(&seq_out) {
            assert!((a - b).abs() < 1e-5, "batch={a} vs seq={b}");
        }

        let mut sliced_kv = LayerKvCache::default();
        let mut sliced_out = Vec::new();
        layer.forward_batch_into(
            &inputs,
            seq_len,
            2,
            2,
            2,
            1_000_000.0,
            1e-6,
            &mut sliced_kv,
            None,
            3,
            true,
            &mut sliced_out,
        )?;
        let last_start = (seq_len - 1) * 4;
        for (a, b) in sliced_out[last_start..].iter().zip(&seq_out[last_start..]) {
            assert!((a - b).abs() < 1e-5, "last-row batch={a} vs seq={b}");
        }
        Ok(())
    }

    #[test]
    fn code_predictor_and_prompt_cache_execute_cleanly() -> std::io::Result<()> {
        let hidden = 4;
        let vocab = 8;
        let cp = code_predictor_fixture()?;
        let talker_embd = LinearProjection::new(
            "talker.codec_embd.weight",
            hidden,
            vocab,
            WeightQuantization::F32(vec![0.2_f32; vocab * hidden]),
            None,
        )?;
        let codes = cp.predict_frame(
            &[0.1, -0.2, 0.3, 0.4],
            3,
            &talker_embd,
            DecoupledSamplingParams::base_clone(),
            42,
            100,
        )?;
        assert_eq!(codes[0], 3);
        assert!(codes.iter().all(|&c| (0..8).contains(&c)));
        let mut pcache = PromptCache::default();
        let ref_codes = [1_i32; 32];
        let mut codec_builds = 0_usize;
        let _ = pcache.get_or_build_icl_reference(
            &ref_codes,
            2,
            || {
                codec_builds += 1;
                Ok(vec![1.0; 12])
            },
            || Ok((vec![2.0; 8], 2)),
        )?;
        let _ = pcache.get_or_build_icl_reference(
            &ref_codes,
            2,
            || {
                codec_builds += 1;
                Ok(vec![1.0; 12])
            },
            || Ok((vec![2.0; 8], 2)),
        )?;
        assert_eq!(
            codec_builds, 1,
            "PromptCache must reuse cached ICL streams on hit"
        );

        let _ = TalkerLoraAdapter {
            name: "indic-en".to_owned(),
            patches: HashMap::new(),
        };
        Ok(())
    }

    #[test]
    fn predictor_workers_preserve_codes_and_cancellation_reaches_inner_layers()
    -> std::io::Result<()> {
        let cp = code_predictor_fixture()?;
        let talker_embd = LinearProjection::new(
            "talker.codec_embd.weight",
            4,
            8,
            WeightQuantization::F32(vec![0.2_f32; 8 * 4]),
            None,
        )?;
        let codes = cp.predict_frame(
            &[0.1, -0.2, 0.3, 0.4],
            3,
            &talker_embd,
            DecoupledSamplingParams::base_clone(),
            42,
            100,
        )?;
        let parallel = cp.predict_frame_with_control(
            &[0.1, -0.2, 0.3, 0.4],
            3,
            &talker_embd,
            DecoupledSamplingParams::base_clone(),
            42,
            100,
            4,
            &mut || Ok(()),
        )?;
        assert_eq!(codes, parallel);
        let mut polls = 0;
        let cancelled = cp.predict_frame_with_control(
            &[0.1, -0.2, 0.3, 0.4],
            3,
            &talker_embd,
            DecoupledSamplingParams::base_clone(),
            42,
            100,
            1,
            &mut || {
                polls += 1;
                if polls == 4 {
                    Err(std::io::Error::new(
                        std::io::ErrorKind::Interrupted,
                        "cancelled",
                    ))
                } else {
                    Ok(())
                }
            },
        );
        assert_eq!(polls, 4);
        assert!(cancelled.is_err_and(|error| error.kind() == std::io::ErrorKind::Interrupted));

        Ok(())
    }

    #[test]
    fn predictor_workspace_matches_fresh_frames_after_edits_cancellation_errors_and_unwind()
    -> std::io::Result<()> {
        let mut model = code_predictor_fixture()?;
        let embeddings = LinearProjection::new(
            "talker.codec_embd.weight",
            4,
            8,
            WeightQuantization::F32(vec![0.2; 8 * 4]),
            None,
        )?;
        let mut workspace = super::predictor_workspace::PredictorWorkspace::default();
        let input = [0.1, -0.2, 0.3, 0.4];
        fn run(
            model: &CodePredictorModel,
            hidden: &[f32],
            code: i32,
            sequence: i64,
            embeddings: &LinearProjection,
            workspace: &mut super::predictor_workspace::PredictorWorkspace,
            cancel: &mut impl FnMut() -> std::io::Result<()>,
        ) -> std::io::Result<[i32; super::TOTAL_CODE_GROUPS]> {
            model.predict_frame_reusing(
                hidden,
                code,
                embeddings,
                DecoupledSamplingParams::base_clone(),
                42,
                sequence,
                3,
                cancel,
                workspace,
            )
        }
        let expected = model.predict_frame(
            &input,
            3,
            &embeddings,
            DecoupledSamplingParams::base_clone(),
            42,
            100,
        )?;
        for (hidden, code, sequence) in [
            (input, 3, 100),
            ([0.4, 0.3, -0.2, 0.1], 7, 132),
            (input, 3, 100),
        ] {
            let fresh = model.predict_frame(
                &hidden,
                code,
                &embeddings,
                DecoupledSamplingParams::base_clone(),
                42,
                sequence,
            )?;
            assert_eq!(
                run(
                    &model,
                    &hidden,
                    code,
                    sequence,
                    &embeddings,
                    &mut workspace,
                    &mut || Ok(()),
                )?,
                fresh
            );
        }
        let mut polls = 0;
        let interrupted = run(
            &model,
            &input,
            3,
            100,
            &embeddings,
            &mut workspace,
            &mut || {
                polls += 1;
                if polls == 8 {
                    Err(std::io::Error::from(std::io::ErrorKind::Interrupted))
                } else {
                    Ok(())
                }
            },
        );
        assert_eq!(polls, 8);
        assert!(interrupted.is_err_and(|error| error.kind() == std::io::ErrorKind::Interrupted));
        assert_eq!(
            run(
                &model,
                &input,
                3,
                100,
                &embeddings,
                &mut workspace,
                &mut || Ok(()),
            )?,
            expected
        );
        model.lm_heads[4].bias = Some(Vec::new());
        assert!(
            run(
                &model,
                &input,
                3,
                100,
                &embeddings,
                &mut workspace,
                &mut || Ok(()),
            )
            .is_err()
        );
        model.lm_heads[4].bias = None;
        assert_eq!(
            run(
                &model,
                &input,
                3,
                100,
                &embeddings,
                &mut workspace,
                &mut || Ok(()),
            )?,
            expected
        );
        polls = 0;
        let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run(
                &model,
                &input,
                3,
                100,
                &embeddings,
                &mut workspace,
                &mut || {
                    polls += 1;
                    if polls == 8 {
                        std::panic::resume_unwind(Box::new("injected acoustic frame fault"));
                    }
                    Ok(())
                },
            )
        }));
        assert!(unwound.is_err());
        assert_eq!(polls, 8);
        assert_eq!(
            run(
                &model,
                &input,
                3,
                100,
                &embeddings,
                &mut workspace,
                &mut || Ok(()),
            )?,
            expected
        );
        Ok(())
    }

    #[test]
    fn stream_chunk_config_and_parallel_projection_match_sequential() -> std::io::Result<()> {
        let cfg = super::StreamChunkConfig::low_latency_4_frame().validate()?;
        assert_eq!(cfg.frames_for_chunk(0), 4);
        assert_eq!(cfg.frames_for_chunk(1), 4);
        assert_eq!(
            super::StreamChunkConfig::standard_8_frame().frames_for_chunk(0),
            8
        );
        let dyadic = super::StreamChunkConfig::dyadic_ramp().validate()?;
        assert_eq!(
            [0, 1, 2, 3, 4].map(|idx| dyadic.dyadic_frames_for_chunk(idx)),
            [1, 2, 4, 8, 8]
        );

        let proj_par = LinearProjection::new(
            "talker.blk.0.attn_q.weight",
            4,
            8,
            WeightQuantization::F32(
                (0..32)
                    .map(|i| f32::from(u8::try_from(i).unwrap_or(0)) * 0.1)
                    .collect(),
            ),
            Some(vec![0.05; 8]),
        )?;
        let x4 = [0.25_f32, -0.5, 0.75, 1.0];
        assert_eq!(
            proj_par.forward_vec(&x4, None)?,
            proj_par.forward_vec_parallel(&x4, None, 4)?
        );

        let q8_proj = LinearProjection::new(
            "code_pred.blk.0.attn_q.weight",
            32,
            2,
            WeightQuantization::Q8_0 {
                blocks: vec![
                    super::Q8Block {
                        scale: 0.25,
                        quants: [2_i8; 32],
                    },
                    super::Q8Block {
                        scale: -0.5,
                        quants: [-1_i8; 32],
                    },
                ],
            },
            None,
        )?;
        let q8_out = q8_proj.forward_vec(&[0.5_f32; 32], None)?;
        assert!((q8_out[0] - 8.0_f32).abs() < 1e-6);
        assert!((q8_out[1] - 8.0_f32).abs() < 1e-6);

        let mut in_place = x4;
        let out_of_place = super::rms_norm(&x4, &[1.0, 0.5, 1.5, 2.0], 1e-6)?;
        super::rms_norm_in_place(&mut in_place, &[1.0, 0.5, 1.5, 2.0], 1e-6)?;
        assert_eq!(out_of_place.as_slice(), in_place.as_slice());
        Ok(())
    }

    #[test]
    fn exp_f32x4_matches_scalar_exp_within_two_ulp_and_exact_zero() {
        let level = super::Level::new();
        let mut max_ulp = 0_u32;
        let mut samples = Vec::with_capacity(4096);
        samples.extend_from_slice(&[
            0.0_f32,
            -0.0,
            -1e-7,
            -0.346_573_6,
            -1.0,
            -20.0,
            -87.3,
            -88.0,
            -120.0,
        ]);
        for i in 0..4000_u32 {
            let x = -87.3_f32 * (f32::from(u16::try_from(i).unwrap_or(0)) / 4000.0_f32);
            samples.push(x);
        }
        for chunk in samples.chunks_exact(4) {
            let got: [f32; 4] = super::dispatch!(level, simd => {
                let v = super::f32x4::from_slice(simd, chunk);
                super::exp_f32x4(simd, v).into()
            });
            for (&x, &g) in chunk.iter().zip(&got) {
                if x == 0.0 {
                    assert_eq!(g.to_bits(), 1.0_f32.to_bits());
                } else if x < -87.336_54 {
                    assert_eq!(g.to_bits(), 0.0_f32.to_bits());
                } else {
                    let want = x.exp();
                    let ulp = g.to_bits().abs_diff(want.to_bits());
                    max_ulp = max_ulp.max(ulp);
                    assert!(ulp <= 2, "x={x} got={g} want={want} ulp={ulp}");
                }
            }
        }
        assert!(max_ulp <= 2);
    }

    #[test]
    fn attend_query_f16_parallel_matches_sequential_across_gqa_ratios() -> std::io::Result<()> {
        let level = super::Level::new();
        let head_dim = 64_usize;
        let total_seq = 97_usize;
        let num_kv_heads = 4_usize;
        for gqa_ratio in [1_usize, 2, 4, 8] {
            let num_q_heads = num_kv_heads * gqa_ratio;
            let kv_stride = num_kv_heads * head_dim;
            let scale = 1.0_f32 / super::usize_to_f32(head_dim).sqrt();
            let q: Vec<f32> = (0..num_q_heads * head_dim)
                .map(|i| {
                    let v = i32::try_from((i * 17 + 5) % 29).unwrap_or(0) - 14;
                    f32::from(i8::try_from(v).unwrap_or(0)) * 0.05
                })
                .collect();
            let mut k_f16 = vec![0_u16; total_seq * kv_stride];
            let mut v_f16 = vec![0_u16; total_seq * kv_stride];
            let k_f32: Vec<f32> = (0..total_seq * kv_stride)
                .map(|i| {
                    let v = i32::try_from((i * 13 + 3) % 23).unwrap_or(0) - 11;
                    f32::from(i8::try_from(v).unwrap_or(0)) * 0.0625
                })
                .collect();
            let v_f32: Vec<f32> = (0..total_seq * kv_stride)
                .map(|i| {
                    let v = i32::try_from((i * 19 + 7) % 31).unwrap_or(0) - 15;
                    f32::from(i8::try_from(v).unwrap_or(0)) * 0.03125
                })
                .collect();
            assert!(super::narrow_f32_slice(&k_f32, &mut k_f16));
            assert!(super::narrow_f32_slice(&v_f32, &mut v_f16));
            let kv = LayerKvCache {
                k: k_f16,
                v: v_f16,
                seq_len: total_seq,
            };

            let mut scores = Vec::new();
            let mut seq_out = vec![0.0_f32; num_q_heads * head_dim];
            super::attend_query_f16(
                level,
                &q,
                &kv,
                total_seq,
                kv_stride,
                num_q_heads,
                num_kv_heads,
                head_dim,
                gqa_ratio,
                scale,
                &mut scores,
                &mut seq_out,
                1,
            )?;

            let mut par_out = vec![f32::NAN; num_q_heads * head_dim];
            super::attend_query_f16(
                level,
                &q,
                &kv,
                total_seq,
                kv_stride,
                num_q_heads,
                num_kv_heads,
                head_dim,
                gqa_ratio,
                scale,
                &mut scores,
                &mut par_out,
                4,
            )?;

            assert_eq!(
                par_out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                seq_out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "parallel vs sequential mismatch for gqa_ratio={gqa_ratio}"
            );

            // Check numerical accuracy against exact scalar f32::exp reference.
            for qh in 0..num_q_heads {
                let kvh = qh / gqa_ratio;
                let q_vec = &q[qh * head_dim..(qh + 1) * head_dim];
                let mut ref_scores = Vec::with_capacity(total_seq);
                let mut max_s = f32::NEG_INFINITY;
                for t in 0..total_seq {
                    let mut kw = vec![0.0_f32; head_dim];
                    super::widen_f16_slice(
                        &kv.k[t * kv_stride + kvh * head_dim..t * kv_stride + (kvh + 1) * head_dim],
                        &mut kw,
                    );
                    let s = q_vec.iter().zip(&kw).map(|(a, b)| a * b).sum::<f32>() * scale;
                    max_s = max_s.max(s);
                    ref_scores.push(s);
                }
                let mut sum = 0.0_f32;
                for s in &mut ref_scores {
                    *s = (*s - max_s).exp();
                    sum += *s;
                }
                let inv = 1.0_f32 / sum.max(1e-12_f32);
                let mut ref_out = vec![0.0_f32; head_dim];
                for (t, &p) in ref_scores.iter().enumerate() {
                    let mut vw = vec![0.0_f32; head_dim];
                    super::widen_f16_slice(
                        &kv.v[t * kv_stride + kvh * head_dim..t * kv_stride + (kvh + 1) * head_dim],
                        &mut vw,
                    );
                    for (dst, &val) in ref_out.iter_mut().zip(&vw) {
                        *dst += (p * inv) * val;
                    }
                }
                for (got, want) in par_out[qh * head_dim..(qh + 1) * head_dim]
                    .iter()
                    .zip(&ref_out)
                {
                    assert!(
                        (got - want).abs() < 1e-5,
                        "gqa={gqa_ratio} qh={qh} got={got} want={want}"
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn predictor_workspace_reuses_buffers_across_multiple_frames_without_reallocating()
    -> std::io::Result<()> {
        let model = code_predictor_fixture()?;
        let embeddings = LinearProjection::new(
            "talker.codec_embd.weight",
            4,
            8,
            WeightQuantization::F32(vec![0.2; 8 * 4]),
            None,
        )?;
        let mut workspace = super::predictor_workspace::PredictorWorkspace::default();
        let input = [0.1, -0.2, 0.3, 0.4];

        let first = model.predict_frame_reusing(
            &input,
            3,
            &embeddings,
            DecoupledSamplingParams::base_clone(),
            42,
            100,
            2,
            &mut || Ok(()),
            &mut workspace,
        )?;
        let mut frame_ptrs = {
            let mut f = workspace.frame(model.layers.len())?;
            let (layers, _, decoder, next_raw, h_a, h_b, normed, logits) = f.buffers();
            let layer0 = (layers[0].k.as_ptr(), layers[0].k.capacity());
            let dec = (decoder.q.as_ptr(), decoder.q.capacity());
            let ptrs = [
                (next_raw.as_ptr(), next_raw.capacity()),
                (h_a.as_ptr(), h_a.capacity()),
                (h_b.as_ptr(), h_b.capacity()),
                (normed.as_ptr(), normed.capacity()),
                (logits.as_ptr(), logits.capacity()),
            ];
            assert!(ptrs.iter().all(|&(_, cap)| cap > 0));
            (layer0, dec, ptrs)
        };

        for frame_idx in 1..4_i64 {
            let out = model.predict_frame_reusing(
                &input,
                3,
                &embeddings,
                DecoupledSamplingParams::base_clone(),
                42,
                100 + frame_idx,
                2,
                &mut || Ok(()),
                &mut workspace,
            )?;
            let fresh = model.predict_frame_with_control(
                &input,
                3,
                &embeddings,
                DecoupledSamplingParams::base_clone(),
                42,
                100 + frame_idx,
                2,
                &mut || Ok(()),
            )?;
            assert_eq!(out, fresh);
            let mut f = workspace.frame(model.layers.len())?;
            let (layers, _, decoder, next_raw, h_a, h_b, normed, logits) = f.buffers();
            assert_eq!((layers[0].k.as_ptr(), layers[0].k.capacity()), frame_ptrs.0);
            assert_eq!((decoder.q.as_ptr(), decoder.q.capacity()), frame_ptrs.1);
            let current_ptrs = [
                (next_raw.as_ptr(), next_raw.capacity()),
                (h_a.as_ptr(), h_a.capacity()),
                (h_b.as_ptr(), h_b.capacity()),
                (normed.as_ptr(), normed.capacity()),
                (logits.as_ptr(), logits.capacity()),
            ];
            assert_eq!(current_ptrs, frame_ptrs.2);
            frame_ptrs.2 = current_ptrs;
        }
        assert_eq!(first[0], 3);
        Ok(())
    }

    #[test]
    fn swiglu_in_place_and_parallel_match_scalar_and_preserve_nonfinite_edge_cases() {
        let scalar_swiglu = |gate: &mut [f32], up: &[f32]| {
            for (g, &u) in gate.iter_mut().zip(up.iter()) {
                let silu = *g / (1.0_f32 + (-*g).exp());
                *g = silu * u;
            }
        };

        for len in [0_usize, 1, 3, 4, 5, 7, 15, 64, 4095, 4096, 4097, 8195] {
            let gate_init: Vec<f32> = (0..len)
                .map(|i| {
                    let centered = i32::try_from(i % 241).unwrap_or(0) - 120;
                    f32::from(i16::try_from(centered).unwrap_or(0)) * 0.125
                })
                .collect();
            let up: Vec<f32> = (0..len)
                .map(|i| {
                    let centered = i32::try_from((i * 7 + 3) % 101).unwrap_or(0) - 50;
                    f32::from(i16::try_from(centered).unwrap_or(0)) * 0.05
                })
                .collect();

            let mut expected = gate_init.clone();
            scalar_swiglu(&mut expected, &up);

            let mut actual = gate_init.clone();
            super::swiglu_in_place(&mut actual, &up);

            let mut parallel = gate_init;
            super::swiglu_in_place_parallel(&mut parallel, &up, 4);

            for (idx, ((&want, &got), &par)) in
                expected.iter().zip(&actual).zip(&parallel).enumerate()
            {
                assert!(
                    (got - want).abs() <= 1e-6,
                    "len={len} idx={idx} got={got} want={want}"
                );
                assert_eq!(
                    par.to_bits(),
                    got.to_bits(),
                    "len={len} idx={idx} parallel mismatch"
                );
            }
        }

        // Verify exact behavior on 0.0, -0.0, extreme finite magnitudes, +-inf, and NaN.
        let edge_gate = [
            0.0_f32,
            -0.0,
            88.0,
            100.0,
            -87.0,
            -88.0,
            -100.0,
            -1000.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            1.5,
        ];
        let edge_up = [
            1.0_f32,
            2.0,
            -0.5,
            0.25,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            f32::NAN,
        ];
        let mut edge_expected = edge_gate;
        scalar_swiglu(&mut edge_expected, &edge_up);
        let mut edge_actual = edge_gate;
        super::swiglu_in_place(&mut edge_actual, &edge_up);

        for (idx, (&want, &got)) in edge_expected.iter().zip(&edge_actual).enumerate() {
            if want.is_nan() {
                assert!(got.is_nan(), "idx={idx} expected NaN got={got}");
            } else if want.is_infinite() {
                assert_eq!(got.to_bits(), want.to_bits(), "idx={idx}");
            } else {
                assert!(
                    (got - want).abs() <= 1e-6,
                    "idx={idx} got={got} want={want}"
                );
            }
        }
    }

    fn make_packed_test_proj(
        name: &str,
        ggml_type: u32,
        in_dim: usize,
        out_dim: usize,
        seed: usize,
        bias: Option<Vec<f32>>,
    ) -> std::io::Result<LinearProjection> {
        let (block, elem_bytes) = crate::weights::packed_layout(ggml_type)?;
        let blocks = in_dim / block;
        let mut bytes = Vec::with_capacity(out_dim * blocks * elem_bytes);
        for row in 0..out_dim {
            for b in 0..blocks {
                match ggml_type {
                    8 => {
                        let scale = 0.015_f32
                            * f32::from(u8::try_from((seed + row + b) % 11 + 1).unwrap_or(1));
                        bytes
                            .extend_from_slice(&bonsai_ternary::float_to_half(scale).to_le_bytes());
                        for k in 0..32 {
                            let q = i8::try_from(
                                isize::try_from((seed * 17 + row * 7 + b * 5 + k * 3) % 199)
                                    .unwrap_or(0)
                                    - 99,
                            )
                            .unwrap_or(0);
                            bytes.push(q.cast_unsigned());
                        }
                    }
                    12 => {
                        let d = 0.01_f32
                            * f32::from(u8::try_from((seed + row + b) % 9 + 1).unwrap_or(1));
                        let dmin = 0.003_f32
                            * f32::from(u8::try_from((seed + row * 3 + b) % 7 + 1).unwrap_or(1));
                        bytes.extend_from_slice(&bonsai_ternary::float_to_half(d).to_le_bytes());
                        bytes.extend_from_slice(&bonsai_ternary::float_to_half(dmin).to_le_bytes());
                        for i in 0..12 {
                            bytes.push(
                                u8::try_from((seed * 13 + row * 29 + b * 11 + i * 41) % 251)
                                    .unwrap_or(0),
                            );
                        }
                        for i in 0..128 {
                            bytes.push(
                                u8::try_from((seed * 31 + row * 19 + b * 23 + i * 37) % 256)
                                    .unwrap_or(0),
                            );
                        }
                    }
                    _ => {
                        for i in 0..128 {
                            bytes.push(
                                u8::try_from((seed * 23 + row * 11 + b * 19 + i * 43) % 256)
                                    .unwrap_or(0),
                            );
                        }
                        for i in 0..64 {
                            bytes.push(
                                u8::try_from((seed * 19 + row * 7 + b * 17 + i * 53) % 256)
                                    .unwrap_or(0),
                            );
                        }
                        for i in 0..16 {
                            let s = i8::try_from(
                                isize::try_from((seed * 5 + row * 3 + b * 7 + i * 11) % 63)
                                    .unwrap_or(0)
                                    - 31,
                            )
                            .unwrap_or(1);
                            bytes.push(s.cast_unsigned());
                        }
                        let d = 0.008_f32
                            * f32::from(u8::try_from((seed + row + b) % 11 + 1).unwrap_or(1));
                        bytes.extend_from_slice(&bonsai_ternary::float_to_half(d).to_le_bytes());
                    }
                }
            }
        }
        LinearProjection::new(
            name,
            in_dim,
            out_dim,
            WeightQuantization::Packed { ggml_type, bytes },
            bias,
        )
    }

    #[test]
    fn forward_batch_group_into_matches_sequential_across_quantizations_and_alignments()
    -> std::io::Result<()> {
        let in_dim = 256_usize;
        let mut p_repack = make_packed_test_proj(
            "talker.blk.0.attn_q.weight",
            12,
            in_dim,
            16,
            1,
            Some(vec![0.05; 16]),
        )?;
        p_repack.repack_q4_k_if_aligned();
        let p_q4 = make_packed_test_proj(
            "talker.blk.0.attn_k.weight",
            12,
            in_dim,
            8,
            2,
            Some(vec![-0.02; 8]),
        )?;
        let p_q8_odd = make_packed_test_proj("talker.blk.0.attn_v.weight", 8, in_dim, 5, 3, None)?;
        let p_q6 = make_packed_test_proj(
            "talker.blk.0.ffn_gate.weight",
            14,
            in_dim,
            12,
            4,
            Some(vec![0.01; 12]),
        )?;
        let p_f32 = LinearProjection::new(
            "f32.proj",
            in_dim,
            6,
            WeightQuantization::F32(
                (0..6 * in_dim)
                    .map(|i| f32::from(i16::try_from(i % 29).unwrap_or(0) - 14) * 0.02)
                    .collect(),
            ),
            Some(vec![0.1; 6]),
        )?;

        let lora_zero = LoraProjectionPatch {
            in_dim,
            out_dim: 16,
            rank: 2,
            alpha_over_rank: 1.5,
            scale: 0.0,
            lora_a: vec![0.1; 2 * in_dim],
            lora_b: vec![0.2; 16 * 2],
        };
        let lora_active = LoraProjectionPatch {
            in_dim,
            out_dim: 8,
            rank: 2,
            alpha_over_rank: 1.5,
            scale: 0.5,
            lora_a: (0..2 * in_dim)
                .map(|i| f32::from(i16::try_from(i % 19).unwrap_or(0) - 9) * 0.01)
                .collect(),
            lora_b: (0..8 * 2)
                .map(|i| f32::from(i16::try_from(i % 17).unwrap_or(0) - 8) * 0.02)
                .collect(),
        };

        for seq_len in [0_usize, 1, 2, 3, 5, 9] {
            let inputs: Vec<f32> = (0..seq_len * in_dim)
                .map(|i| f32::from(i16::try_from((i * 11 + 3) % 53).unwrap_or(0) - 26) * 0.05)
                .collect();
            for threads in [1_usize, 4] {
                let mut ref_q = Vec::new();
                let mut ref_k = Vec::new();
                let mut ref_v = Vec::new();
                let mut ref_g = Vec::new();
                p_repack.forward_batch_into(
                    &inputs,
                    seq_len,
                    Some(&lora_zero),
                    threads,
                    &mut ref_q,
                )?;
                p_q4.forward_batch_into(&inputs, seq_len, Some(&lora_active), threads, &mut ref_k)?;
                p_q8_odd.forward_batch_into(&inputs, seq_len, None, threads, &mut ref_v)?;
                p_q6.forward_batch_into(&inputs, seq_len, None, threads, &mut ref_g)?;

                let mut grp_q = Vec::new();
                let mut grp_k = Vec::new();
                let mut grp_v = Vec::new();
                let mut grp_g = Vec::new();
                LinearProjection::forward_batch_group_into(
                    &mut [
                        (&p_repack, Some(&lora_zero), &mut grp_q),
                        (&p_q4, Some(&lora_active), &mut grp_k),
                        (&p_q8_odd, None, &mut grp_v),
                        (&p_q6, None, &mut grp_g),
                    ],
                    &inputs,
                    seq_len,
                    threads,
                )?;
                assert_eq!(
                    grp_q.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    ref_q.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "q seq_len={seq_len} threads={threads}"
                );
                assert_eq!(
                    grp_k.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    ref_k.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "k seq_len={seq_len} threads={threads}"
                );
                assert_eq!(
                    grp_v.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    ref_v.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "v seq_len={seq_len} threads={threads}"
                );
                assert_eq!(
                    grp_g.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    ref_g.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "g seq_len={seq_len} threads={threads}"
                );

                // Also verify fallback path when mixing F32 into the group.
                let mut ref_f = Vec::new();
                p_f32.forward_batch_into(&inputs, seq_len, None, threads, &mut ref_f)?;
                let mut grp_f = Vec::new();
                let mut grp_k2 = Vec::new();
                LinearProjection::forward_batch_group_into(
                    &mut [(&p_q4, None, &mut grp_k2), (&p_f32, None, &mut grp_f)],
                    &inputs,
                    seq_len,
                    threads,
                )?;
                assert_eq!(
                    grp_f.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    ref_f.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                );
            }
        }
        Ok(())
    }

    #[test]
    fn rms_norm_batch_into_and_attend_batch_f16_match_sequential_bits() -> std::io::Result<()> {
        for (hidden, seq_len) in [(4_usize, 3_usize), (64, 7), (256, 5)] {
            let weight: Vec<f32> = (0..hidden)
                .map(|i| 0.5 + f32::from(u8::try_from(i % 13).unwrap_or(0)) * 0.1)
                .collect();
            let inputs: Vec<f32> = (0..seq_len * hidden)
                .map(|i| f32::from(i16::try_from((i * 17) % 41).unwrap_or(0) - 20) * 0.05)
                .collect();
            let mut expected = Vec::with_capacity(inputs.len());
            let mut row = Vec::new();
            for t in 0..seq_len {
                super::rms_norm_into(
                    &inputs[t * hidden..(t + 1) * hidden],
                    &weight,
                    1e-6,
                    &mut row,
                )?;
                expected.extend_from_slice(&row);
            }
            for workers in [1_usize, 4] {
                let mut actual = Vec::new();
                super::rms_norm_batch_into(&inputs, seq_len, &weight, 1e-6, workers, &mut actual)?;
                assert_eq!(
                    actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "rms_norm_batch_into hidden={hidden} seq_len={seq_len} workers={workers}"
                );
            }
        }

        let level = super::Level::new();
        let head_dim = 64_usize;
        for (initial_seq, seq_len, num_kv_heads, gqa_ratio) in [
            (0_usize, 5_usize, 4_usize, 2_usize),
            (31, 9, 2, 4),
            (64, 6, 1, 2),
        ] {
            let num_q_heads = num_kv_heads * gqa_ratio;
            let query_width = num_q_heads * head_dim;
            let kv_width = num_kv_heads * head_dim;
            let total_kv_seq = initial_seq + seq_len;
            let scale = 1.0_f32 / super::usize_to_f32(head_dim).sqrt();

            let q_all: Vec<f32> = (0..seq_len * query_width)
                .map(|i| f32::from(i16::try_from((i * 13 + 7) % 31).unwrap_or(0) - 15) * 0.05)
                .collect();
            let k_f32: Vec<f32> = (0..total_kv_seq * kv_width)
                .map(|i| f32::from(i16::try_from((i * 19 + 3) % 29).unwrap_or(0) - 14) * 0.0625)
                .collect();
            let v_f32: Vec<f32> = (0..total_kv_seq * kv_width)
                .map(|i| f32::from(i16::try_from((i * 23 + 5) % 37).unwrap_or(0) - 18) * 0.03125)
                .collect();
            let mut k_f16 = vec![0_u16; k_f32.len()];
            let mut v_f16 = vec![0_u16; v_f32.len()];
            assert!(super::narrow_f32_slice(&k_f32, &mut k_f16));
            assert!(super::narrow_f32_slice(&v_f32, &mut v_f16));
            let kv = LayerKvCache {
                k: k_f16,
                v: v_f16,
                seq_len: total_kv_seq,
            };

            let mut expected = vec![0.0_f32; seq_len * query_width];
            let mut scores = Vec::new();
            for t in 0..seq_len {
                super::attend_query_f16(
                    level,
                    &q_all[t * query_width..(t + 1) * query_width],
                    &kv,
                    initial_seq + t + 1,
                    kv_width,
                    num_q_heads,
                    num_kv_heads,
                    head_dim,
                    gqa_ratio,
                    scale,
                    &mut scores,
                    &mut expected[t * query_width..(t + 1) * query_width],
                    1,
                )?;
            }
            for workers in [1_usize, 4] {
                let mut actual = vec![f32::NAN; seq_len * query_width];
                super::attend_batch_f16(
                    level,
                    &q_all,
                    &kv,
                    initial_seq,
                    seq_len,
                    kv_width,
                    num_q_heads,
                    num_kv_heads,
                    head_dim,
                    gqa_ratio,
                    scale,
                    &mut actual,
                    workers,
                )?;
                assert_eq!(
                    actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "attend_batch_f16 initial={initial_seq} seq_len={seq_len} workers={workers}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn forward_batch_into_repacked_layer_matches_sequential_steps() -> std::io::Result<()> {
        let hidden = 256_usize;
        let num_q_heads = 4_usize;
        let num_kv_heads = 2_usize;
        let head_dim = 64_usize;
        let intermediate = 256_usize;
        let mut layer = QwenDecoderLayer {
            attn_norm_w: vec![1.0; hidden],
            q_proj: make_packed_test_proj(
                "talker.blk.0.attn_q.weight",
                12,
                hidden,
                hidden,
                1,
                None,
            )?,
            k_proj: make_packed_test_proj(
                "talker.blk.0.attn_k.weight",
                12,
                hidden,
                num_kv_heads * head_dim,
                2,
                None,
            )?,
            v_proj: make_packed_test_proj(
                "talker.blk.0.attn_v.weight",
                12,
                hidden,
                num_kv_heads * head_dim,
                3,
                None,
            )?,
            o_proj: make_packed_test_proj(
                "talker.blk.0.attn_output.weight",
                12,
                hidden,
                hidden,
                4,
                None,
            )?,
            q_norm_w: vec![1.0; head_dim],
            k_norm_w: vec![1.0; head_dim],
            ffn_norm_w: vec![1.0; hidden],
            gate_proj: make_packed_test_proj(
                "talker.blk.0.ffn_gate.weight",
                12,
                hidden,
                intermediate,
                5,
                None,
            )?,
            up_proj: make_packed_test_proj(
                "talker.blk.0.ffn_up.weight",
                12,
                hidden,
                intermediate,
                6,
                None,
            )?,
            down_proj: make_packed_test_proj(
                "talker.blk.0.ffn_down.weight",
                12,
                intermediate,
                hidden,
                7,
                None,
            )?,
        };
        layer.repack_q4_k_if_aligned();

        let seq_len = 5_usize;
        let inputs: Vec<f32> = (0..seq_len * hidden)
            .map(|i| f32::from(i16::try_from((i * 7 + 11) % 37).unwrap_or(0) - 18) * 0.04)
            .collect();
        let mut seq_kv = LayerKvCache::default();
        let mut seq_out = Vec::with_capacity(inputs.len());
        for t in 0..seq_len {
            let step = layer.forward_step_parallel(
                &inputs[t * hidden..(t + 1) * hidden],
                num_q_heads,
                num_kv_heads,
                head_dim,
                1_000_000.0,
                1e-6,
                &mut seq_kv,
                None,
                4,
            )?;
            seq_out.extend_from_slice(&step);
        }

        let mut batch_kv = LayerKvCache::default();
        let mut batch_out = Vec::new();
        layer.forward_batch_into(
            &inputs,
            seq_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
            1_000_000.0,
            1e-6,
            &mut batch_kv,
            None,
            4,
            false,
            &mut batch_out,
        )?;
        assert_eq!(batch_kv.seq_len, seq_kv.seq_len);
        assert_eq!(batch_kv.k, seq_kv.k);
        assert_eq!(batch_kv.v, seq_kv.v);
        for (a, b) in batch_out.iter().zip(&seq_out) {
            assert!((a - b).abs() < 1e-4, "batch={a} seq={b}");
        }

        let mut sliced_kv = LayerKvCache::default();
        let mut sliced_out = Vec::new();
        layer.forward_batch_into(
            &inputs,
            seq_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
            1_000_000.0,
            1e-6,
            &mut sliced_kv,
            None,
            4,
            true,
            &mut sliced_out,
        )?;
        let last_start = (seq_len - 1) * hidden;
        for (a, b) in sliced_out[last_start..].iter().zip(&seq_out[last_start..]) {
            assert!((a - b).abs() < 1e-4, "last={a} seq={b}");
        }
        Ok(())
    }

    #[test]
    fn q8_8row_8x_gemm_and_mixed_group_reuse_match_bit_exact() -> std::io::Result<()> {
        let in_dim = 256_usize;
        let big_biased_q8 = make_packed_test_proj(
            "talker.blk.0.ffn_gate.weight",
            8,
            in_dim,
            64,
            10,
            Some(
                (0..64)
                    .map(|i| 1.25 + super::usize_to_f32(i) * 0.125)
                    .collect(),
            ),
        )?;
        let q8_unbiased =
            make_packed_test_proj("talker.blk.0.attn_q.weight", 8, in_dim, 19, 11, None)?;
        let q4_biased = make_packed_test_proj(
            "talker.blk.0.attn_k.weight",
            12,
            in_dim,
            13,
            12,
            Some(vec![-0.25_f32; 13]),
        )?;
        let q6_unbiased =
            make_packed_test_proj("talker.blk.0.attn_v.weight", 14, in_dim, 9, 13, None)?;

        // First call with large biased Q8_0 fills GROUP_COMBINED with large values.
        let warm_in: Vec<f32> = (0..in_dim)
            .map(|i| f32::from(i16::try_from((i * 5 + 1) % 31).unwrap_or(0) - 15) * 0.05)
            .collect();
        let mut warm_out_a = Vec::new();
        let mut warm_out_b = Vec::new();
        super::forward_group_into(
            &mut [
                (&big_biased_q8, None, &mut warm_out_a),
                (&big_biased_q8, None, &mut warm_out_b),
            ],
            &warm_in,
            4,
        )?;
        assert_eq!(
            warm_out_a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            warm_out_b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );

        // Now run unbiased + mixed Q8_0 + Q4_K + Q6_K group; stale GROUP_COMBINED must not leak.
        let mut ref_q8 = Vec::new();
        let mut ref_q4 = Vec::new();
        let mut ref_q6 = Vec::new();
        q8_unbiased.forward_vec_into(&warm_in, None, None, 1, &mut ref_q8)?;
        q4_biased.forward_vec_into(&warm_in, None, None, 1, &mut ref_q4)?;
        q6_unbiased.forward_vec_into(&warm_in, None, None, 1, &mut ref_q6)?;

        for threads in [1_usize, 4] {
            let mut grp_q8 = Vec::new();
            let mut grp_q4 = Vec::new();
            let mut grp_q6 = Vec::new();
            super::forward_group_into(
                &mut [
                    (&q8_unbiased, None, &mut grp_q8),
                    (&q4_biased, None, &mut grp_q4),
                    (&q6_unbiased, None, &mut grp_q6),
                ],
                &warm_in,
                threads,
            )?;
            assert_eq!(
                grp_q8.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                ref_q8.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "mixed group Q8_0 threads={threads}"
            );
            assert_eq!(
                grp_q4.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                ref_q4.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "mixed group Q4_K threads={threads}"
            );
            assert_eq!(
                grp_q6.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                ref_q6.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "mixed group Q6_K threads={threads}"
            );
        }

        // Verify Q8_0 batch GEMM across 8x8, 8x4, 8x2, 8x1 tiles and row tails (19 = 8 + 8 + 3).
        for seq_len in [1_usize, 2, 3, 5, 8, 9, 17] {
            let batch_in: Vec<f32> = (0..seq_len * in_dim)
                .map(|i| f32::from(i16::try_from((i * 13 + 7) % 43).unwrap_or(0) - 21) * 0.03125)
                .collect();
            let mut ref_batch = Vec::with_capacity(seq_len * q8_unbiased.out_dim);
            let mut step_out = Vec::new();
            for t in 0..seq_len {
                q8_unbiased.forward_vec_into(
                    &batch_in[t * in_dim..(t + 1) * in_dim],
                    None,
                    None,
                    1,
                    &mut step_out,
                )?;
                ref_batch.extend_from_slice(&step_out);
            }
            for threads in [1_usize, 4] {
                let mut got_batch = Vec::new();
                q8_unbiased.forward_batch_into(
                    &batch_in,
                    seq_len,
                    None,
                    threads,
                    &mut got_batch,
                )?;
                assert_eq!(
                    got_batch.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    ref_batch.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "Q8_0 batch GEMM seq_len={seq_len} threads={threads}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn cached_rope_divisors_and_tables_match_uncached_bit_exact() -> std::io::Result<()> {
        let configs = [
            (64_usize, 1_000_000.0_f32),
            (128, 1_000_000.0),
            (64, 10_000.0),
            (32, 10_000.0),
            (264, 1_000_000.0),
        ];
        // Run two full sweeps (exceeding the 4-entry divisor ring) and repeated pos hits.
        for _sweep in 0..2 {
            for (head_dim, theta) in configs {
                let half = head_dim / 2;
                for pos in [0_usize, 1, 7, 15, 255] {
                    let mut exp_cos = vec![0.0_f32; half];
                    let mut exp_sin = vec![0.0_f32; half];
                    super::compute_rope_tables_uncached(
                        head_dim,
                        pos,
                        theta,
                        &mut exp_cos,
                        &mut exp_sin,
                    );

                    // Twice per pos to exercise both step-cache miss (with cached divisors) and hit.
                    for _hit in 0..2 {
                        let mut got_cos = vec![0.0_f32; half];
                        let mut got_sin = vec![0.0_f32; half];
                        super::compute_rope_tables(
                            head_dim,
                            pos,
                            theta,
                            &mut got_cos,
                            &mut got_sin,
                        );
                        assert_eq!(
                            got_cos.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            exp_cos.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            "cos mismatch hd={head_dim} theta={theta} pos={pos}"
                        );
                        assert_eq!(
                            got_sin.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            exp_sin.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            "sin mismatch hd={head_dim} theta={theta} pos={pos}"
                        );
                    }

                    let num_heads = 3_usize;
                    let base_heads: Vec<f32> = (0..num_heads * head_dim)
                        .map(|i| f32::from(i16::try_from((i * 19) % 47).unwrap_or(0) - 23) * 0.1)
                        .collect();
                    let mut exp_heads = base_heads.clone();
                    super::apply_rope_neox_with_tables(
                        &mut exp_heads,
                        num_heads,
                        head_dim,
                        &exp_cos,
                        &exp_sin,
                    )?;
                    let mut got_heads = base_heads;
                    super::apply_rope_neox_in_place(
                        &mut got_heads,
                        num_heads,
                        head_dim,
                        pos,
                        theta,
                    )?;
                    assert_eq!(
                        got_heads.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        exp_heads.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        "in-place RoPE mismatch hd={head_dim} theta={theta} pos={pos}"
                    );
                }

                // Batch table cache hit/miss check.
                let initial_seq = 3_usize;
                let seq_len = 9_usize;
                for _layer in 0..3 {
                    super::with_rope_batch_tables(
                        head_dim,
                        initial_seq,
                        seq_len,
                        theta,
                        |t, cos_got, sin_got| {
                            let mut cos_ref = vec![0.0_f32; half];
                            let mut sin_ref = vec![0.0_f32; half];
                            super::compute_rope_tables_uncached(
                                head_dim,
                                initial_seq + t,
                                theta,
                                &mut cos_ref,
                                &mut sin_ref,
                            );
                            assert_eq!(
                                cos_got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                                cos_ref.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                            );
                            assert_eq!(
                                sin_got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                                sin_ref.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                            );
                            Ok(())
                        },
                    )?;
                }
            }
        }
        Ok(())
    }
}
