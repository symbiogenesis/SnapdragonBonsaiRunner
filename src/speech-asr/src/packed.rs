//! Bit-exact safe SIMD kernels for Qwen3-ASR projections, attention, and PCM validation.
#![forbid(unsafe_code)]
#![expect(
    clippy::inline_always,
    reason = "fearless_simd dispatch! requires inline(always) to inherit target_feature attributes."
)]
#![allow(
    clippy::suboptimal_flops,
    reason = "Non-fused multiply and add preserve exact left-to-right scalar f32 bit parity."
)]

#[cfg(all(windows, feature = "rust-qualification"))]
use bonsai_windows_process::quant_kernels::{self, Arm64QuantCaps, Q8_0ActBlock, Q8_KActBlock};
use std::sync::OnceLock;

use crate::row_workers;
use fearless_simd::{Level, dispatch, f32x4, prelude::*, u8x16};

static SIMD_LEVEL: OnceLock<Level> = OnceLock::new();

pub(super) fn simd_level() -> Level {
    *SIMD_LEVEL.get_or_init(Level::new)
}

/// Borrowed weight slice for row-major `F32`, `Q8_0`, `Q4_K`, or `Q6_K` matrices.
#[derive(Clone, Copy)]
pub(super) enum WeightSlice<'a> {
    Float(&'a [f32]),
    Q8(&'a [u8]),
    Q4K(&'a [u8]),
    Q6K(&'a [u8]),
}

/// Borrowed matrix geometry paired with its row-major weight slice.
#[derive(Clone, Copy)]
pub(super) struct WeightMatrix<'a> {
    pub(super) input: usize,
    pub(super) output: usize,
    pub(super) data: WeightSlice<'a>,
}

/// Four-position-packed input activations so sibling projections share transposition.
#[derive(Default)]
pub(super) struct PreparedBatch {
    input: usize,
    batch: usize,
    num_quads: usize,
    quads: Vec<[f32; 4]>,
}

impl PreparedBatch {
    #[cfg(test)]
    pub(super) fn new(raw_input: &[f32], batch: usize, input: usize) -> Self {
        let mut prepared = Self::default();
        prepared.prepare_into(raw_input, batch, input);
        prepared
    }

    pub(super) fn clear_bounded(&mut self, max_retain: usize) {
        self.input = 0;
        self.batch = 0;
        self.num_quads = 0;
        self.quads.clear();
        if self.quads.capacity() > max_retain.saturating_mul(2) {
            self.quads.shrink_to(max_retain);
        }
    }

    #[cfg(test)]
    pub(super) fn quads_capacity(&self) -> usize {
        self.quads.capacity()
    }

    pub(super) fn prepare_into(&mut self, raw_input: &[f32], batch: usize, input: usize) {
        self.input = input;
        self.batch = batch;
        self.num_quads = batch.div_ceil(4);
        if self.num_quads > 0 && input > 0 {
            self.quads.resize(self.num_quads * input, [0.0; 4]);
            let full_quads = batch / 4;
            for q in 0..full_quads {
                let p0 = &raw_input[(q * 4) * input..(q * 4 + 1) * input];
                let p1 = &raw_input[(q * 4 + 1) * input..(q * 4 + 2) * input];
                let p2 = &raw_input[(q * 4 + 2) * input..(q * 4 + 3) * input];
                let p3 = &raw_input[(q * 4 + 3) * input..(q * 4 + 4) * input];
                let dst = &mut self.quads[q * input..(q + 1) * input];
                for col in 0..input {
                    dst[col] = [p0[col], p1[col], p2[col], p3[col]];
                }
            }
            let rem = batch % 4;
            if rem > 0 {
                let base = full_quads * 4;
                let p0 = &raw_input[base * input..(base + 1) * input];
                let p1 = if rem > 1 {
                    &raw_input[(base + 1) * input..(base + 2) * input]
                } else {
                    p0
                };
                let p2 = if rem > 2 {
                    &raw_input[(base + 2) * input..(base + 3) * input]
                } else {
                    p0
                };
                let dst = &mut self.quads[full_quads * input..(full_quads + 1) * input];
                for col in 0..input {
                    dst[col] = [
                        p0[col],
                        if rem > 1 { p1[col] } else { 0.0 },
                        if rem > 2 { p2[col] } else { 0.0 },
                        0.0,
                    ];
                }
            }
        } else {
            self.quads.clear();
        }
    }

    pub(super) const fn batch(&self) -> usize {
        self.batch
    }

    pub(super) const fn input(&self) -> usize {
        self.input
    }
}

/// Return `true` if every sample is finite and within `[-1.0, 1.0]`.
pub(super) fn pcm_in_unit_range(pcm: &[f32]) -> bool {
    let level = simd_level();
    dispatch!(level, simd => pcm_in_unit_range_simd(simd, pcm))
}

#[inline(always)]
fn pcm_in_unit_range_simd<S: Simd>(simd: S, pcm: &[f32]) -> bool {
    let min_v = f32x4::splat(simd, -1.0_f32);
    let max_v = f32x4::splat(simd, 1.0_f32);
    let (chunks, tail) = pcm.as_chunks::<4>();
    for chunk in chunks {
        let v = f32x4::from_slice(simd, chunk);
        let valid = v.simd_ge(min_v) & v.simd_le(max_v);
        if !valid.all_true() {
            return false;
        }
    }
    tail.iter()
        .all(|sample| sample.is_finite() && (-1.0..=1.0).contains(sample))
}

#[inline(always)]
fn half_scale(bytes: &[u8]) -> f32 {
    bonsai_ternary::half_to_float(u16::from_le_bytes([bytes[0], bytes[1]]))
}

#[inline]
pub(super) const fn unpack_q4_k_scale_min(group: usize, packed: &[u8]) -> (u8, u8) {
    bonsai_ternary::unpack_q4_k_scale_min(group, packed)
}

/// Decode one row of `Q4_K` super-blocks (`144` bytes / `256` elements) into `out`.
pub(super) fn decode_q4_k_row(row_bytes: &[u8], out: &mut [f32]) {
    for (block, dst_block) in row_bytes
        .as_chunks::<144>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<256>().0)
    {
        let d = half_scale(&block[0..2]);
        let dmin = half_scale(&block[2..4]);
        let scales = &block[4..16];
        let qs = &block[16..144];
        for chunk_idx in 0..4 {
            let (s_lo, m_lo) = unpack_q4_k_scale_min(chunk_idx * 2, scales);
            let (s_hi, m_hi) = unpack_q4_k_scale_min(chunk_idx * 2 + 1, scales);
            let sc_lo = d * f32::from(s_lo);
            let min_lo = dmin * f32::from(m_lo);
            let sc_hi = d * f32::from(s_hi);
            let min_hi = dmin * f32::from(m_hi);
            let q_chunk = &qs[chunk_idx * 32..(chunk_idx + 1) * 32];
            let dst_lo = &mut dst_block[chunk_idx * 64..chunk_idx * 64 + 32];
            for (dst, &q) in dst_lo.iter_mut().zip(q_chunk) {
                *dst = sc_lo * f32::from(q & 0x0f) - min_lo;
            }
            let dst_hi = &mut dst_block[chunk_idx * 64 + 32..chunk_idx * 64 + 64];
            for (dst, &q) in dst_hi.iter_mut().zip(q_chunk) {
                *dst = sc_hi * f32::from(q >> 4) - min_hi;
            }
        }
    }
}

/// Decode one row of `Q6_K` super-blocks (`210` bytes / `256` elements) into `out`.
pub(super) fn decode_q6_k_row(row_bytes: &[u8], out: &mut [f32]) {
    for (block, dst_block) in row_bytes
        .as_chunks::<210>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<256>().0)
    {
        let ql = &block[0..128];
        let qh = &block[128..192];
        let scales = &block[192..208];
        let d = half_scale(&block[208..210]);
        for n128 in 0..2 {
            let ql_s = &ql[n128 * 64..];
            let qh_s = &qh[n128 * 32..];
            let sc_s = &scales[n128 * 8..];
            let dst_s = &mut dst_block[n128 * 128..];
            for l in 0..32 {
                let is = l / 16;
                let ql0 = ql_s[l];
                let ql1 = ql_s[l + 32];
                let qh_v = qh_s[l];
                let q1 = i16::from((ql0 & 0x0f) | ((qh_v & 3) << 4)) - 32;
                let q2 = i16::from((ql1 & 0x0f) | (((qh_v >> 2) & 3) << 4)) - 32;
                let q3 = i16::from((ql0 >> 4) | (((qh_v >> 4) & 3) << 4)) - 32;
                let q4 = i16::from((ql1 >> 4) | (((qh_v >> 6) & 3) << 4)) - 32;
                dst_s[l] = (d * f32::from(sc_s[is].cast_signed())) * f32::from(q1);
                dst_s[l + 32] = (d * f32::from(sc_s[is + 2].cast_signed())) * f32::from(q2);
                dst_s[l + 64] = (d * f32::from(sc_s[is + 4].cast_signed())) * f32::from(q3);
                dst_s[l + 96] = (d * f32::from(sc_s[is + 6].cast_signed())) * f32::from(q4);
            }
        }
    }
}

#[inline(always)]
fn signed_lanes<S: Simd>(simd: S, bytes: u8x16<S>) -> [f32x4<S>; 4] {
    let (lo, hi) = simd.widen_i8x16(bytes.bitcast());
    let (a, b) = simd.widen_i16x8(lo);
    let (c, d) = simd.widen_i16x8(hi);
    [a, b, c, d].map(|value| simd.cvt_f32_i32x4(value))
}

#[inline(always)]
fn transpose_4x4<S: Simd>(simd: S, rows: [f32x4<S>; 4]) -> [f32x4<S>; 4] {
    let t0 = simd.zip_low_f32x4(rows[0], rows[1]);
    let t1 = simd.zip_high_f32x4(rows[0], rows[1]);
    let t2 = simd.zip_low_f32x4(rows[2], rows[3]);
    let t3 = simd.zip_high_f32x4(rows[2], rows[3]);
    let c0: f32x4<S> = simd.zip_low_u64x2(t0.bitcast(), t2.bitcast()).bitcast();
    let c1: f32x4<S> = simd.zip_high_u64x2(t0.bitcast(), t2.bitcast()).bitcast();
    let c2: f32x4<S> = simd.zip_low_u64x2(t1.bitcast(), t3.bitcast()).bitcast();
    let c3: f32x4<S> = simd.zip_high_u64x2(t1.bitcast(), t3.bitcast()).bitcast();
    [c0, c1, c2, c3]
}

#[inline(always)]
fn single_f32_4rows<S: Simd>(simd: S, rows: [&[f32]; 4], input: &[f32], out: &mut [f32]) {
    let mut acc = f32x4::splat(simd, 0.0_f32);
    let len = input.len();
    let chunks = len / 4;
    for chunk_idx in 0..chunks {
        let base = chunk_idx * 4;
        let r0 = f32x4::from_slice(simd, &rows[0][base..base + 4]);
        let r1 = f32x4::from_slice(simd, &rows[1][base..base + 4]);
        let r2 = f32x4::from_slice(simd, &rows[2][base..base + 4]);
        let r3 = f32x4::from_slice(simd, &rows[3][base..base + 4]);
        let cols = transpose_4x4(simd, [r0, r1, r2, r3]);
        for k in 0..4 {
            let act = f32x4::splat(simd, input[base + k]);
            acc += cols[k] * act;
        }
    }
    for col in (chunks * 4)..len {
        let w = f32x4::from_slice(
            simd,
            &[rows[0][col], rows[1][col], rows[2][col], rows[3][col]],
        );
        acc += w * f32x4::splat(simd, input[col]);
    }
    acc.store_slice(out);
}

#[inline(always)]
fn single_q8_4rows<S: Simd>(simd: S, rows: [&[u8]; 4], input: &[f32], out: &mut [f32]) {
    let mut acc = f32x4::splat(simd, 0.0_f32);
    let blocks = input.len() / 32;
    for block_idx in 0..blocks {
        let b_off = block_idx * 34;
        let i_off = block_idx * 32;
        let s0 = f32x4::splat(simd, half_scale(&rows[0][b_off..b_off + 2]));
        let s1 = f32x4::splat(simd, half_scale(&rows[1][b_off..b_off + 2]));
        let s2 = f32x4::splat(simd, half_scale(&rows[2][b_off..b_off + 2]));
        let s3 = f32x4::splat(simd, half_scale(&rows[3][b_off..b_off + 2]));
        for part in 0..2 {
            let q_off = b_off + 2 + part * 16;
            let p_off = i_off + part * 16;
            let q0 = signed_lanes(simd, u8x16::from_slice(simd, &rows[0][q_off..q_off + 16]));
            let q1 = signed_lanes(simd, u8x16::from_slice(simd, &rows[1][q_off..q_off + 16]));
            let q2 = signed_lanes(simd, u8x16::from_slice(simd, &rows[2][q_off..q_off + 16]));
            let q3 = signed_lanes(simd, u8x16::from_slice(simd, &rows[3][q_off..q_off + 16]));
            for group in 0..4 {
                let col_base = p_off + group * 4;
                let w0 = s0 * q0[group];
                let w1 = s1 * q1[group];
                let w2 = s2 * q2[group];
                let w3 = s3 * q3[group];
                let cols = transpose_4x4(simd, [w0, w1, w2, w3]);
                for k in 0..4 {
                    acc += cols[k] * f32x4::splat(simd, input[col_base + k]);
                }
            }
        }
    }
    acc.store_slice(out);
}

#[inline(always)]
fn single_q8_1row<S: Simd>(simd: S, row: &[u8], input: &[f32]) -> f32 {
    let mut acc = 0.0_f32;
    for (block, activations) in row
        .as_chunks::<34>()
        .0
        .iter()
        .zip(input.as_chunks::<32>().0)
    {
        let scale = f32x4::splat(simd, half_scale(&block[..2]));
        let q_lo = signed_lanes(simd, u8x16::from_slice(simd, &block[2..18]));
        let q_hi = signed_lanes(simd, u8x16::from_slice(simd, &block[18..34]));
        for (group_idx, q) in q_lo.into_iter().chain(q_hi).enumerate() {
            let w = scale * q;
            let a = f32x4::from_slice(simd, &activations[group_idx * 4..group_idx * 4 + 4]);
            let prod: [f32; 4] = (w * a).into();
            acc += prod[0];
            acc += prod[1];
            acc += prod[2];
            acc += prod[3];
        }
    }
    acc
}

#[inline(always)]
fn single_f32_1row(row: &[f32], input: &[f32]) -> f32 {
    let mut acc = 0.0_f32;
    for (weight, activation) in row.iter().zip(input) {
        acc += *weight * *activation;
    }
    acc
}

fn dot_q4_k_row_scalar(row_bytes: &[u8], input: &[f32]) -> f32 {
    let mut sum = 0.0_f32;
    for (block, in_block) in row_bytes
        .as_chunks::<144>()
        .0
        .iter()
        .zip(input.as_chunks::<256>().0)
    {
        let d = half_scale(&block[0..2]);
        let dmin = half_scale(&block[2..4]);
        let scales = &block[4..16];
        let qs = &block[16..144];
        for chunk_idx in 0..4 {
            let (s_lo, m_lo) = unpack_q4_k_scale_min(chunk_idx * 2, scales);
            let (s_hi, m_hi) = unpack_q4_k_scale_min(chunk_idx * 2 + 1, scales);
            let sc_lo = d * f32::from(s_lo);
            let min_lo = dmin * f32::from(m_lo);
            let sc_hi = d * f32::from(s_hi);
            let min_hi = dmin * f32::from(m_hi);
            let q_chunk = &qs[chunk_idx * 32..(chunk_idx + 1) * 32];
            let in_lo = &in_block[chunk_idx * 64..chunk_idx * 64 + 32];
            for (&q, &a) in q_chunk.iter().zip(in_lo) {
                sum += (sc_lo * f32::from(q & 0x0f) - min_lo) * a;
            }
            let in_hi = &in_block[chunk_idx * 64 + 32..chunk_idx * 64 + 64];
            for (&q, &a) in q_chunk.iter().zip(in_hi) {
                sum += (sc_hi * f32::from(q >> 4) - min_hi) * a;
            }
        }
    }
    sum
}

fn dot_q6_k_row_scalar(row_bytes: &[u8], input: &[f32]) -> f32 {
    let mut sum = 0.0_f32;
    for (block, in_block) in row_bytes
        .as_chunks::<210>()
        .0
        .iter()
        .zip(input.as_chunks::<256>().0)
    {
        let ql = &block[0..128];
        let qh = &block[128..192];
        let scales = &block[192..208];
        let d = half_scale(&block[208..210]);
        for n128 in 0..2 {
            let ql_s = &ql[n128 * 64..];
            let qh_s = &qh[n128 * 32..];
            let sc_s = &scales[n128 * 8..];
            let in_s = &in_block[n128 * 128..];
            for l in 0..32 {
                let is = l / 16;
                let ql0 = ql_s[l];
                let ql1 = ql_s[l + 32];
                let qh_v = qh_s[l];
                let q1 = i16::from((ql0 & 0x0f) | ((qh_v & 3) << 4)) - 32;
                let q2 = i16::from((ql1 & 0x0f) | (((qh_v >> 2) & 3) << 4)) - 32;
                let q3 = i16::from((ql0 >> 4) | (((qh_v >> 4) & 3) << 4)) - 32;
                let q4 = i16::from((ql1 >> 4) | (((qh_v >> 6) & 3) << 4)) - 32;
                sum += ((d * f32::from(sc_s[is].cast_signed())) * f32::from(q1)) * in_s[l];
                sum += ((d * f32::from(sc_s[is + 2].cast_signed())) * f32::from(q2)) * in_s[l + 32];
                sum += ((d * f32::from(sc_s[is + 4].cast_signed())) * f32::from(q3)) * in_s[l + 64];
                sum += ((d * f32::from(sc_s[is + 6].cast_signed())) * f32::from(q4)) * in_s[l + 96];
            }
        }
    }
    sum
}

#[inline(always)]
fn single_rows_simd<S: Simd>(
    simd: S,
    weights: WeightMatrix<'_>,
    start_row: usize,
    input: &[f32],
    out: &mut [f32],
) {
    let width = weights.input;
    let mut processed = 0;
    match weights.data {
        WeightSlice::Float(data) => {
            for dst in out.as_chunks_mut::<4>().0 {
                let row0 = (start_row + processed) * width;
                let row_slices = [
                    &data[row0..row0 + width],
                    &data[row0 + width..row0 + 2 * width],
                    &data[row0 + 2 * width..row0 + 3 * width],
                    &data[row0 + 3 * width..row0 + 4 * width],
                ];
                single_f32_4rows(simd, row_slices, input, dst);
                processed += 4;
            }
            for (local_idx, dst) in out[processed..].iter_mut().enumerate() {
                let off = (start_row + processed + local_idx) * width;
                *dst = single_f32_1row(&data[off..off + width], input);
            }
        }
        WeightSlice::Q8(data) => {
            let row_bytes = width / 32 * 34;
            for dst in out.as_chunks_mut::<4>().0 {
                let row0 = (start_row + processed) * row_bytes;
                let row_slices = [
                    &data[row0..row0 + row_bytes],
                    &data[row0 + row_bytes..row0 + 2 * row_bytes],
                    &data[row0 + 2 * row_bytes..row0 + 3 * row_bytes],
                    &data[row0 + 3 * row_bytes..row0 + 4 * row_bytes],
                ];
                single_q8_4rows(simd, row_slices, input, dst);
                processed += 4;
            }
            for (local_idx, dst) in out[processed..].iter_mut().enumerate() {
                let off = (start_row + processed + local_idx) * row_bytes;
                *dst = single_q8_1row(simd, &data[off..off + row_bytes], input);
            }
        }
        WeightSlice::Q4K(data) => {
            let row_bytes = width / 256 * 144;
            for (local_idx, dst) in out.iter_mut().enumerate() {
                let off = (start_row + local_idx) * row_bytes;
                *dst = dot_q4_k_row_scalar(&data[off..off + row_bytes], input);
            }
        }
        WeightSlice::Q6K(data) => {
            let row_bytes = width / 256 * 210;
            for (local_idx, dst) in out.iter_mut().enumerate() {
                let off = (start_row + local_idx) * row_bytes;
                *dst = dot_q6_k_row_scalar(&data[off..off + row_bytes], input);
            }
        }
    }
}

#[cfg(all(windows, feature = "rust-qualification"))]
fn single_rows_quant_arm64(
    caps: Arm64QuantCaps,
    weights: WeightMatrix<'_>,
    start_row: usize,
    act_q8_0: &[Q8_0ActBlock],
    act_q8_k: &[Q8_KActBlock],
    out: &mut [f32],
) {
    let width = weights.input;
    let mut processed = 0;
    match weights.data {
        WeightSlice::Q8(data) => {
            let row_bytes = width / 32 * 34;
            for dst in out.as_chunks_mut::<4>().0 {
                let row0 = (start_row + processed) * row_bytes;
                let row_slices = [
                    &data[row0..row0 + row_bytes],
                    &data[row0 + row_bytes..row0 + 2 * row_bytes],
                    &data[row0 + 2 * row_bytes..row0 + 3 * row_bytes],
                    &data[row0 + 3 * row_bytes..row0 + 4 * row_bytes],
                ];
                *dst = quant_kernels::dot_q8_0_4rows(caps, row_slices, act_q8_0);
                processed += 4;
            }
            for (local_idx, dst) in out[processed..].iter_mut().enumerate() {
                let off = (start_row + processed + local_idx) * row_bytes;
                *dst = quant_kernels::dot_q8_0_row(caps, &data[off..off + row_bytes], act_q8_0);
            }
        }
        WeightSlice::Q4K(data) => {
            let row_bytes = width / 256 * 144;
            for dst in out.as_chunks_mut::<4>().0 {
                let row0 = (start_row + processed) * row_bytes;
                let row_slices = [
                    &data[row0..row0 + row_bytes],
                    &data[row0 + row_bytes..row0 + 2 * row_bytes],
                    &data[row0 + 2 * row_bytes..row0 + 3 * row_bytes],
                    &data[row0 + 3 * row_bytes..row0 + 4 * row_bytes],
                ];
                *dst = quant_kernels::dot_q4_k_4rows(caps, row_slices, act_q8_k);
                processed += 4;
            }
            for (local_idx, dst) in out[processed..].iter_mut().enumerate() {
                let off = (start_row + processed + local_idx) * row_bytes;
                *dst = quant_kernels::dot_q4_k_row(caps, &data[off..off + row_bytes], act_q8_k);
            }
        }
        WeightSlice::Q6K(data) => {
            let row_bytes = width / 256 * 210;
            for dst in out.as_chunks_mut::<4>().0 {
                let row0 = (start_row + processed) * row_bytes;
                let row_slices = [
                    &data[row0..row0 + row_bytes],
                    &data[row0 + row_bytes..row0 + 2 * row_bytes],
                    &data[row0 + 2 * row_bytes..row0 + 3 * row_bytes],
                    &data[row0 + 3 * row_bytes..row0 + 4 * row_bytes],
                ];
                *dst = quant_kernels::dot_q6_k_4rows(caps, row_slices, act_q8_k);
                processed += 4;
            }
            for (local_idx, dst) in out[processed..].iter_mut().enumerate() {
                let off = (start_row + processed + local_idx) * row_bytes;
                *dst = quant_kernels::dot_q6_k_row(caps, &data[off..off + row_bytes], act_q8_k);
            }
        }
        WeightSlice::Float(_) => {}
    }
}

/// Bit-exact matrix-vector multiplication across `1x4` SIMD tiles (or ARM64 `SDOT` kernels) and `row_workers`.
pub(super) fn multiply_single_into(
    weights: WeightMatrix<'_>,
    input: &[f32],
    workers: usize,
    out: &mut [f32],
) {
    #[cfg(all(windows, feature = "rust-qualification"))]
    {
        if !matches!(weights.data, WeightSlice::Float(_)) {
            let caps = Arm64QuantCaps::detect();
            let mut act_q8_0 = Vec::new();
            let mut act_q8_k = Vec::new();
            match weights.data {
                WeightSlice::Q8(_) => {
                    act_q8_0.resize(
                        weights.input / 32,
                        Q8_0ActBlock {
                            d: 0.0,
                            qs: [0; 32],
                        },
                    );
                    quant_kernels::quantize_row_q8_0(input, &mut act_q8_0);
                }
                WeightSlice::Q4K(_) | WeightSlice::Q6K(_) => {
                    act_q8_k.resize(
                        weights.input / 256,
                        Q8_KActBlock {
                            d: 0.0,
                            qs: [0; 256],
                            bsums: [0; 16],
                        },
                    );
                    quant_kernels::quantize_row_q8_k(input, &mut act_q8_k);
                }
                WeightSlice::Float(_) => {}
            }
            row_workers::run_aligned(out, workers, 4, &|start_row, slice| {
                single_rows_quant_arm64(caps, weights, start_row, &act_q8_0, &act_q8_k, slice);
            });
            return;
        }
    }
    let level = simd_level();
    row_workers::run_aligned(out, workers, 4, &|start_row, slice| {
        dispatch!(level, simd => single_rows_simd(simd, weights, start_row, input, slice));
    });
}

/// Execute multiple sibling projections sharing `input` (`query + key + value` or `gate + up`)
/// in one quantized activation pass and one worker barrier when `workers > 1`.
pub(super) fn multiply_single_group_into(
    targets: &mut [(WeightMatrix<'_>, &mut [f32])],
    input: &[f32],
    workers: usize,
    scratch: &mut Vec<f32>,
) {
    if targets.is_empty() {
        return;
    }
    if targets.len() == 1 {
        let (weights, out) = &mut targets[0];
        multiply_single_into(*weights, input, workers, out);
        return;
    }

    #[cfg(all(windows, feature = "rust-qualification"))]
    let (caps, act_q8_0, act_q8_k) = {
        let caps = Arm64QuantCaps::detect();
        let needs_q8_0 = targets
            .iter()
            .any(|(w, _)| matches!(w.data, WeightSlice::Q8(_)));
        let needs_q8_k = targets
            .iter()
            .any(|(w, _)| matches!(w.data, WeightSlice::Q4K(_) | WeightSlice::Q6K(_)));
        let mut q8_0 = Vec::new();
        let mut q8_k = Vec::new();
        if needs_q8_0 {
            q8_0.resize(
                input.len() / 32,
                Q8_0ActBlock {
                    d: 0.0,
                    qs: [0; 32],
                },
            );
            quant_kernels::quantize_row_q8_0(input, &mut q8_0);
        }
        if needs_q8_k {
            q8_k.resize(
                input.len() / 256,
                Q8_KActBlock {
                    d: 0.0,
                    qs: [0; 256],
                    bsums: [0; 16],
                },
            );
            quant_kernels::quantize_row_q8_k(input, &mut q8_k);
        }
        (caps, q8_0, q8_k)
    };

    let level = simd_level();
    if workers <= 1 {
        for (weights, out) in targets.iter_mut() {
            #[cfg(all(windows, feature = "rust-qualification"))]
            if !matches!(weights.data, WeightSlice::Float(_)) {
                single_rows_quant_arm64(caps, *weights, 0, &act_q8_0, &act_q8_k, out);
                continue;
            }
            dispatch!(level, simd => single_rows_simd(simd, *weights, 0, input, out));
        }
        return;
    }

    let mut offsets = [0_usize; 4];
    let mut padded_total = 0_usize;
    for (idx, (weights, _)) in targets.iter().enumerate() {
        offsets[idx] = padded_total;
        padded_total += weights.output.next_multiple_of(4);
    }
    scratch.resize(padded_total, 0.0);
    let matrices: [WeightMatrix<'_>; 4] = [
        targets[0].0,
        targets[1].0,
        targets.get(2).map_or(targets[0].0, |t| t.0),
        targets.get(3).map_or(targets[0].0, |t| t.0),
    ];
    let count = targets.len();
    row_workers::run_aligned(scratch, workers, 4, &|chunk_start, slice| {
        let chunk_end = chunk_start + slice.len();
        for t_idx in 0..count {
            let t_start = offsets[t_idx];
            let t_end = t_start + matrices[t_idx].output;
            let lo = chunk_start.max(t_start);
            let hi = chunk_end.min(t_end);
            if lo >= hi {
                continue;
            }
            let local_row = lo - t_start;
            let dst = &mut slice[lo - chunk_start..hi - chunk_start];
            #[cfg(all(windows, feature = "rust-qualification"))]
            if !matches!(matrices[t_idx].data, WeightSlice::Float(_)) {
                single_rows_quant_arm64(
                    caps,
                    matrices[t_idx],
                    local_row,
                    &act_q8_0,
                    &act_q8_k,
                    dst,
                );
                continue;
            }
            dispatch!(
                level,
                simd => single_rows_simd(simd, matrices[t_idx], local_row, input, dst)
            );
        }
    });
    for (idx, (_, out)) in targets.iter_mut().enumerate() {
        let start = offsets[idx];
        out.copy_from_slice(&scratch[start..start + out.len()]);
    }
}

// Compute all `num_quads` quads for a 4-row block, storing each row `r`'s quad `q` at
// `transposed_rows[r * row_stride + q * 4 .. r * row_stride + q * 4 + 4]`.
#[inline(always)]
fn batch_f32_4rows<S: Simd>(
    simd: S,
    rows: [&[f32]; 4],
    quads: &[[f32; 4]],
    num_quads: usize,
    transposed_rows: &mut [f32],
) {
    let width = rows[0].len();
    let row_stride = num_quads * 4;
    let mut q = 0;
    while q + 1 < num_quads {
        let q0 = &quads[q * width..(q + 1) * width];
        let q1 = &quads[(q + 1) * width..(q + 2) * width];
        let mut acc0 = [f32x4::splat(simd, 0.0_f32); 4];
        let mut acc1 = [f32x4::splat(simd, 0.0_f32); 4];
        for col in 0..width {
            let a0 = f32x4::from_slice(simd, &q0[col]);
            let a1 = f32x4::from_slice(simd, &q1[col]);
            for r in 0..4 {
                let w = f32x4::splat(simd, rows[r][col]);
                acc0[r] += w * a0;
                acc1[r] += w * a1;
            }
        }
        for r in 0..4 {
            let base = r * row_stride + q * 4;
            acc0[r].store_slice(&mut transposed_rows[base..base + 4]);
            acc1[r].store_slice(&mut transposed_rows[base + 4..base + 8]);
        }
        q += 2;
    }
    if q < num_quads {
        let q0 = &quads[q * width..(q + 1) * width];
        let mut acc0 = [f32x4::splat(simd, 0.0_f32); 4];
        for col in 0..width {
            let a0 = f32x4::from_slice(simd, &q0[col]);
            for r in 0..4 {
                let w = f32x4::splat(simd, rows[r][col]);
                acc0[r] += w * a0;
            }
        }
        for r in 0..4 {
            let base = r * row_stride + q * 4;
            acc0[r].store_slice(&mut transposed_rows[base..base + 4]);
        }
    }
}

#[inline(always)]
fn batch_f32_1row<S: Simd>(
    simd: S,
    row: &[f32],
    quads: &[[f32; 4]],
    num_quads: usize,
    transposed_row: &mut [f32],
) {
    let width = row.len();
    let mut q = 0;
    while q + 1 < num_quads {
        let q0 = &quads[q * width..(q + 1) * width];
        let q1 = &quads[(q + 1) * width..(q + 2) * width];
        let mut acc0 = f32x4::splat(simd, 0.0_f32);
        let mut acc1 = f32x4::splat(simd, 0.0_f32);
        for (col, &weight) in row.iter().enumerate() {
            let w = f32x4::splat(simd, weight);
            acc0 += w * f32x4::from_slice(simd, &q0[col]);
            acc1 += w * f32x4::from_slice(simd, &q1[col]);
        }
        acc0.store_slice(&mut transposed_row[q * 4..q * 4 + 4]);
        acc1.store_slice(&mut transposed_row[q * 4 + 4..q * 4 + 8]);
        q += 2;
    }
    if q < num_quads {
        let q0 = &quads[q * width..(q + 1) * width];
        let mut acc0 = f32x4::splat(simd, 0.0_f32);
        for (col, &weight) in row.iter().enumerate() {
            let w = f32x4::splat(simd, weight);
            acc0 += w * f32x4::from_slice(simd, &q0[col]);
        }
        acc0.store_slice(&mut transposed_row[q * 4..q * 4 + 4]);
    }
}

#[inline(always)]
fn batch_q8_4rows<S: Simd>(
    simd: S,
    rows: [&[u8]; 4],
    width: usize,
    quads: &[[f32; 4]],
    num_quads: usize,
    transposed_rows: &mut [f32],
) {
    let blocks = width / 32;
    let row_stride = num_quads * 4;
    let mut q = 0;
    while q + 1 < num_quads {
        let q0 = &quads[q * width..(q + 1) * width];
        let q1 = &quads[(q + 1) * width..(q + 2) * width];
        let mut acc0 = [f32x4::splat(simd, 0.0_f32); 4];
        let mut acc1 = [f32x4::splat(simd, 0.0_f32); 4];
        for block_idx in 0..blocks {
            let b_off = block_idx * 34;
            let i_off = block_idx * 32;
            let s0 = f32x4::splat(simd, half_scale(&rows[0][b_off..b_off + 2]));
            let s1 = f32x4::splat(simd, half_scale(&rows[1][b_off..b_off + 2]));
            let s2 = f32x4::splat(simd, half_scale(&rows[2][b_off..b_off + 2]));
            let s3 = f32x4::splat(simd, half_scale(&rows[3][b_off..b_off + 2]));
            for part in 0..2 {
                let q_off = b_off + 2 + part * 16;
                let p_off = i_off + part * 16;
                let qs0 = signed_lanes(simd, u8x16::from_slice(simd, &rows[0][q_off..q_off + 16]));
                let qs1 = signed_lanes(simd, u8x16::from_slice(simd, &rows[1][q_off..q_off + 16]));
                let qs2 = signed_lanes(simd, u8x16::from_slice(simd, &rows[2][q_off..q_off + 16]));
                let qs3 = signed_lanes(simd, u8x16::from_slice(simd, &rows[3][q_off..q_off + 16]));
                for group in 0..4 {
                    let col_base = p_off + group * 4;
                    let w0: [f32; 4] = (s0 * qs0[group]).into();
                    let w1: [f32; 4] = (s1 * qs1[group]).into();
                    let w2: [f32; 4] = (s2 * qs2[group]).into();
                    let w3: [f32; 4] = (s3 * qs3[group]).into();
                    for k in 0..4 {
                        let col = col_base + k;
                        let a0 = f32x4::from_slice(simd, &q0[col]);
                        let a1 = f32x4::from_slice(simd, &q1[col]);
                        let wv0 = f32x4::splat(simd, w0[k]);
                        let wv1 = f32x4::splat(simd, w1[k]);
                        let wv2 = f32x4::splat(simd, w2[k]);
                        let wv3 = f32x4::splat(simd, w3[k]);
                        acc0[0] += wv0 * a0;
                        acc1[0] += wv0 * a1;
                        acc0[1] += wv1 * a0;
                        acc1[1] += wv1 * a1;
                        acc0[2] += wv2 * a0;
                        acc1[2] += wv2 * a1;
                        acc0[3] += wv3 * a0;
                        acc1[3] += wv3 * a1;
                    }
                }
            }
        }
        for r in 0..4 {
            let base = r * row_stride + q * 4;
            acc0[r].store_slice(&mut transposed_rows[base..base + 4]);
            acc1[r].store_slice(&mut transposed_rows[base + 4..base + 8]);
        }
        q += 2;
    }
    if q < num_quads {
        let q0 = &quads[q * width..(q + 1) * width];
        let mut acc0 = [f32x4::splat(simd, 0.0_f32); 4];
        for block_idx in 0..blocks {
            let b_off = block_idx * 34;
            let i_off = block_idx * 32;
            let s0 = f32x4::splat(simd, half_scale(&rows[0][b_off..b_off + 2]));
            let s1 = f32x4::splat(simd, half_scale(&rows[1][b_off..b_off + 2]));
            let s2 = f32x4::splat(simd, half_scale(&rows[2][b_off..b_off + 2]));
            let s3 = f32x4::splat(simd, half_scale(&rows[3][b_off..b_off + 2]));
            for part in 0..2 {
                let q_off = b_off + 2 + part * 16;
                let p_off = i_off + part * 16;
                let qs0 = signed_lanes(simd, u8x16::from_slice(simd, &rows[0][q_off..q_off + 16]));
                let qs1 = signed_lanes(simd, u8x16::from_slice(simd, &rows[1][q_off..q_off + 16]));
                let qs2 = signed_lanes(simd, u8x16::from_slice(simd, &rows[2][q_off..q_off + 16]));
                let qs3 = signed_lanes(simd, u8x16::from_slice(simd, &rows[3][q_off..q_off + 16]));
                for group in 0..4 {
                    let col_base = p_off + group * 4;
                    let w0: [f32; 4] = (s0 * qs0[group]).into();
                    let w1: [f32; 4] = (s1 * qs1[group]).into();
                    let w2: [f32; 4] = (s2 * qs2[group]).into();
                    let w3: [f32; 4] = (s3 * qs3[group]).into();
                    for k in 0..4 {
                        let a0 = f32x4::from_slice(simd, &q0[col_base + k]);
                        acc0[0] += f32x4::splat(simd, w0[k]) * a0;
                        acc0[1] += f32x4::splat(simd, w1[k]) * a0;
                        acc0[2] += f32x4::splat(simd, w2[k]) * a0;
                        acc0[3] += f32x4::splat(simd, w3[k]) * a0;
                    }
                }
            }
        }
        for r in 0..4 {
            let base = r * row_stride + q * 4;
            acc0[r].store_slice(&mut transposed_rows[base..base + 4]);
        }
    }
}

#[inline(always)]
fn batch_q8_1row<S: Simd>(
    simd: S,
    row: &[u8],
    width: usize,
    quads: &[[f32; 4]],
    num_quads: usize,
    transposed_row: &mut [f32],
) {
    let blocks = width / 32;
    let mut q = 0;
    while q + 1 < num_quads {
        let q0 = &quads[q * width..(q + 1) * width];
        let q1 = &quads[(q + 1) * width..(q + 2) * width];
        let mut acc0 = f32x4::splat(simd, 0.0_f32);
        let mut acc1 = f32x4::splat(simd, 0.0_f32);
        for block_idx in 0..blocks {
            let b_off = block_idx * 34;
            let i_off = block_idx * 32;
            let scale = f32x4::splat(simd, half_scale(&row[b_off..b_off + 2]));
            let q_lo = signed_lanes(simd, u8x16::from_slice(simd, &row[b_off + 2..b_off + 18]));
            let q_hi = signed_lanes(simd, u8x16::from_slice(simd, &row[b_off + 18..b_off + 34]));
            for (group_idx, group_q) in q_lo.into_iter().chain(q_hi).enumerate() {
                let w: [f32; 4] = (scale * group_q).into();
                let col_base = i_off + group_idx * 4;
                for k in 0..4 {
                    let col = col_base + k;
                    let wv = f32x4::splat(simd, w[k]);
                    acc0 += wv * f32x4::from_slice(simd, &q0[col]);
                    acc1 += wv * f32x4::from_slice(simd, &q1[col]);
                }
            }
        }
        acc0.store_slice(&mut transposed_row[q * 4..q * 4 + 4]);
        acc1.store_slice(&mut transposed_row[q * 4 + 4..q * 4 + 8]);
        q += 2;
    }
    if q < num_quads {
        let q0 = &quads[q * width..(q + 1) * width];
        let mut acc0 = f32x4::splat(simd, 0.0_f32);
        for block_idx in 0..blocks {
            let b_off = block_idx * 34;
            let i_off = block_idx * 32;
            let scale = f32x4::splat(simd, half_scale(&row[b_off..b_off + 2]));
            let q_lo = signed_lanes(simd, u8x16::from_slice(simd, &row[b_off + 2..b_off + 18]));
            let q_hi = signed_lanes(simd, u8x16::from_slice(simd, &row[b_off + 18..b_off + 34]));
            for (group_idx, group_q) in q_lo.into_iter().chain(q_hi).enumerate() {
                let w: [f32; 4] = (scale * group_q).into();
                let col_base = i_off + group_idx * 4;
                for k in 0..4 {
                    let wv = f32x4::splat(simd, w[k]);
                    acc0 += wv * f32x4::from_slice(simd, &q0[col_base + k]);
                }
            }
        }
        acc0.store_slice(&mut transposed_row[q * 4..q * 4 + 4]);
    }
}

#[inline(always)]
fn batch_rows_simd<S: Simd>(
    simd: S,
    weights: WeightMatrix<'_>,
    start_row: usize,
    quads: &[[f32; 4]],
    num_quads: usize,
    transposed_slice: &mut [f32],
) {
    let width = weights.input;
    let row_stride = num_quads * 4;
    let mut processed = 0;
    match weights.data {
        WeightSlice::Float(data) => {
            for chunk in transposed_slice.chunks_exact_mut(4 * row_stride) {
                let row0 = (start_row + processed) * width;
                let row_slices = [
                    &data[row0..row0 + width],
                    &data[row0 + width..row0 + 2 * width],
                    &data[row0 + 2 * width..row0 + 3 * width],
                    &data[row0 + 3 * width..row0 + 4 * width],
                ];
                batch_f32_4rows(simd, row_slices, quads, num_quads, chunk);
                processed += 4;
            }
            for (local_idx, row_out) in transposed_slice[processed * row_stride..]
                .chunks_exact_mut(row_stride)
                .enumerate()
            {
                let off = (start_row + processed + local_idx) * width;
                batch_f32_1row(simd, &data[off..off + width], quads, num_quads, row_out);
            }
        }
        WeightSlice::Q8(data) => {
            let row_bytes = width / 32 * 34;
            for chunk in transposed_slice.chunks_exact_mut(4 * row_stride) {
                let row0 = (start_row + processed) * row_bytes;
                let row_slices = [
                    &data[row0..row0 + row_bytes],
                    &data[row0 + row_bytes..row0 + 2 * row_bytes],
                    &data[row0 + 2 * row_bytes..row0 + 3 * row_bytes],
                    &data[row0 + 3 * row_bytes..row0 + 4 * row_bytes],
                ];
                batch_q8_4rows(simd, row_slices, width, quads, num_quads, chunk);
                processed += 4;
            }
            for (local_idx, row_out) in transposed_slice[processed * row_stride..]
                .chunks_exact_mut(row_stride)
                .enumerate()
            {
                let off = (start_row + processed + local_idx) * row_bytes;
                batch_q8_1row(
                    simd,
                    &data[off..off + row_bytes],
                    width,
                    quads,
                    num_quads,
                    row_out,
                );
            }
        }
        WeightSlice::Q4K(data) => {
            let row_bytes = width / 256 * 144;
            let mut pos_in = vec![0.0_f32; width];
            for (local_idx, row_out) in transposed_slice.chunks_exact_mut(row_stride).enumerate() {
                let r_off = (start_row + local_idx) * row_bytes;
                let w_row = &data[r_off..r_off + row_bytes];
                for q in 0..num_quads {
                    let q_slice = &quads[q * width..(q + 1) * width];
                    for lane in 0..4 {
                        for (dst, src) in pos_in.iter_mut().zip(q_slice) {
                            *dst = src[lane];
                        }
                        row_out[q * 4 + lane] = dot_q4_k_row_scalar(w_row, &pos_in);
                    }
                }
            }
        }
        WeightSlice::Q6K(data) => {
            let row_bytes = width / 256 * 210;
            let mut pos_in = vec![0.0_f32; width];
            for (local_idx, row_out) in transposed_slice.chunks_exact_mut(row_stride).enumerate() {
                let r_off = (start_row + local_idx) * row_bytes;
                let w_row = &data[r_off..r_off + row_bytes];
                for q in 0..num_quads {
                    let q_slice = &quads[q * width..(q + 1) * width];
                    for lane in 0..4 {
                        for (dst, src) in pos_in.iter_mut().zip(q_slice) {
                            *dst = src[lane];
                        }
                        row_out[q * 4 + lane] = dot_q6_k_row_scalar(w_row, &pos_in);
                    }
                }
            }
        }
    }
}

#[inline(always)]
fn scatter_quads_simd<S: Simd>(
    simd: S,
    transposed: &[f32],
    row_stride: usize,
    output: usize,
    start_elem: usize,
    out_quads: &mut [f32],
) {
    let start_quad = start_elem / (4 * output);
    let quad_count = out_quads.len() / (4 * output);
    let row_quads = output / 4;
    for local_q in 0..quad_count {
        let q = start_quad + local_q;
        let dst_q = &mut out_quads[local_q * 4 * output..(local_q + 1) * 4 * output];
        let (p0_p1, p2_p3) = dst_q.split_at_mut(2 * output);
        let (p0, p1) = p0_p1.split_at_mut(output);
        let (p2, p3) = p2_p3.split_at_mut(output);
        for rq in 0..row_quads {
            let r = rq * 4;
            let r0 = f32x4::from_slice(
                simd,
                &transposed[r * row_stride + q * 4..r * row_stride + q * 4 + 4],
            );
            let r1 = f32x4::from_slice(
                simd,
                &transposed[(r + 1) * row_stride + q * 4..(r + 1) * row_stride + q * 4 + 4],
            );
            let r2 = f32x4::from_slice(
                simd,
                &transposed[(r + 2) * row_stride + q * 4..(r + 2) * row_stride + q * 4 + 4],
            );
            let r3 = f32x4::from_slice(
                simd,
                &transposed[(r + 3) * row_stride + q * 4..(r + 3) * row_stride + q * 4 + 4],
            );
            let cols = transpose_4x4(simd, [r0, r1, r2, r3]);
            cols[0].store_slice(&mut p0[r..r + 4]);
            cols[1].store_slice(&mut p1[r..r + 4]);
            cols[2].store_slice(&mut p2[r..r + 4]);
            cols[3].store_slice(&mut p3[r..r + 4]);
        }
        for r in (row_quads * 4)..output {
            let base = r * row_stride + q * 4;
            p0[r] = transposed[base];
            p1[r] = transposed[base + 1];
            p2[r] = transposed[base + 2];
            p3[r] = transposed[base + 3];
        }
    }
}

#[cfg(all(windows, feature = "rust-qualification"))]
fn quantize_batch_activations(
    weights: &[WeightMatrix<'_>],
    raw_input: &[f32],
    batch: usize,
    width: usize,
) -> (Arm64QuantCaps, Vec<Q8_0ActBlock>, Vec<Q8_KActBlock>) {
    let caps = Arm64QuantCaps::detect();
    let needs_q8_0 = weights.iter().any(|w| matches!(w.data, WeightSlice::Q8(_)));
    let needs_q8_k = weights
        .iter()
        .any(|w| matches!(w.data, WeightSlice::Q4K(_) | WeightSlice::Q6K(_)));
    let mut act_q8_0 = Vec::new();
    let mut act_q8_k = Vec::new();
    if needs_q8_0 {
        let blocks = width / 32;
        act_q8_0.resize(
            batch * blocks,
            Q8_0ActBlock {
                d: 0.0,
                qs: [0; 32],
            },
        );
        for t in 0..batch {
            quant_kernels::quantize_row_q8_0(
                &raw_input[t * width..(t + 1) * width],
                &mut act_q8_0[t * blocks..(t + 1) * blocks],
            );
        }
    }
    if needs_q8_k {
        let blocks = width / 256;
        act_q8_k.resize(
            batch * blocks,
            Q8_KActBlock {
                d: 0.0,
                qs: [0; 256],
                bsums: [0; 16],
            },
        );
        for t in 0..batch {
            quant_kernels::quantize_row_q8_k(
                &raw_input[t * width..(t + 1) * width],
                &mut act_q8_k[t * blocks..(t + 1) * blocks],
            );
        }
    }
    (caps, act_q8_0, act_q8_k)
}

#[cfg(all(windows, feature = "rust-qualification"))]
fn batch_rows_quant_arm64(
    caps: Arm64QuantCaps,
    weights: WeightMatrix<'_>,
    start_row: usize,
    batch: usize,
    act_q8_0: &[Q8_0ActBlock],
    act_q8_k: &[Q8_KActBlock],
    transposed_slice: &mut [f32],
) {
    let width = weights.input;
    let mut processed = 0_usize;
    match weights.data {
        WeightSlice::Q8(data) => {
            let blocks = width / 32;
            let row_bytes = blocks * 34;
            for chunk in transposed_slice.chunks_exact_mut(4 * batch) {
                let r0 = (start_row + processed) * row_bytes;
                let row_slices = [
                    &data[r0..r0 + row_bytes],
                    &data[r0 + row_bytes..r0 + 2 * row_bytes],
                    &data[r0 + 2 * row_bytes..r0 + 3 * row_bytes],
                    &data[r0 + 3 * row_bytes..r0 + 4 * row_bytes],
                ];
                let mut t = 0_usize;
                while t + 1 < batch {
                    let res = quant_kernels::gemm_q8_0_4x2(
                        caps,
                        row_slices,
                        &act_q8_0[t * blocks..(t + 1) * blocks],
                        &act_q8_0[(t + 1) * blocks..(t + 2) * blocks],
                    );
                    for (r, pair) in res.iter().enumerate() {
                        chunk[r * batch + t] = pair[0];
                        chunk[r * batch + t + 1] = pair[1];
                    }
                    t += 2;
                }
                if t < batch {
                    let res = quant_kernels::dot_q8_0_4rows(
                        caps,
                        row_slices,
                        &act_q8_0[t * blocks..(t + 1) * blocks],
                    );
                    for (r, &v) in res.iter().enumerate() {
                        chunk[r * batch + t] = v;
                    }
                }
                processed += 4;
            }
            for (local_idx, row_out) in transposed_slice[processed * batch..]
                .chunks_exact_mut(batch)
                .enumerate()
            {
                let off = (start_row + processed + local_idx) * row_bytes;
                let w_row = &data[off..off + row_bytes];
                for (t, dst) in row_out.iter_mut().enumerate() {
                    *dst = quant_kernels::dot_q8_0_row(
                        caps,
                        w_row,
                        &act_q8_0[t * blocks..(t + 1) * blocks],
                    );
                }
            }
        }
        WeightSlice::Q4K(data) => {
            let blocks = width / 256;
            let row_bytes = blocks * 144;
            for chunk in transposed_slice.chunks_exact_mut(4 * batch) {
                let r0 = (start_row + processed) * row_bytes;
                let row_slices = [
                    &data[r0..r0 + row_bytes],
                    &data[r0 + row_bytes..r0 + 2 * row_bytes],
                    &data[r0 + 2 * row_bytes..r0 + 3 * row_bytes],
                    &data[r0 + 3 * row_bytes..r0 + 4 * row_bytes],
                ];
                let mut t = 0_usize;
                while t + 1 < batch {
                    let res = quant_kernels::gemm_q4_k_4x2(
                        caps,
                        row_slices,
                        &act_q8_k[t * blocks..(t + 1) * blocks],
                        &act_q8_k[(t + 1) * blocks..(t + 2) * blocks],
                    );
                    for (r, pair) in res.iter().enumerate() {
                        chunk[r * batch + t] = pair[0];
                        chunk[r * batch + t + 1] = pair[1];
                    }
                    t += 2;
                }
                if t < batch {
                    let res = quant_kernels::dot_q4_k_4rows(
                        caps,
                        row_slices,
                        &act_q8_k[t * blocks..(t + 1) * blocks],
                    );
                    for (r, &v) in res.iter().enumerate() {
                        chunk[r * batch + t] = v;
                    }
                }
                processed += 4;
            }
            for (local_idx, row_out) in transposed_slice[processed * batch..]
                .chunks_exact_mut(batch)
                .enumerate()
            {
                let off = (start_row + processed + local_idx) * row_bytes;
                let w_row = &data[off..off + row_bytes];
                for (t, dst) in row_out.iter_mut().enumerate() {
                    *dst = quant_kernels::dot_q4_k_row(
                        caps,
                        w_row,
                        &act_q8_k[t * blocks..(t + 1) * blocks],
                    );
                }
            }
        }
        WeightSlice::Q6K(data) => {
            let blocks = width / 256;
            let row_bytes = blocks * 210;
            for chunk in transposed_slice.chunks_exact_mut(4 * batch) {
                let r0 = (start_row + processed) * row_bytes;
                let row_slices = [
                    &data[r0..r0 + row_bytes],
                    &data[r0 + row_bytes..r0 + 2 * row_bytes],
                    &data[r0 + 2 * row_bytes..r0 + 3 * row_bytes],
                    &data[r0 + 3 * row_bytes..r0 + 4 * row_bytes],
                ];
                let mut t = 0_usize;
                while t + 1 < batch {
                    let res = quant_kernels::gemm_q6_k_4x2(
                        caps,
                        row_slices,
                        &act_q8_k[t * blocks..(t + 1) * blocks],
                        &act_q8_k[(t + 1) * blocks..(t + 2) * blocks],
                    );
                    for (r, pair) in res.iter().enumerate() {
                        chunk[r * batch + t] = pair[0];
                        chunk[r * batch + t + 1] = pair[1];
                    }
                    t += 2;
                }
                if t < batch {
                    let res = quant_kernels::dot_q6_k_4rows(
                        caps,
                        row_slices,
                        &act_q8_k[t * blocks..(t + 1) * blocks],
                    );
                    for (r, &v) in res.iter().enumerate() {
                        chunk[r * batch + t] = v;
                    }
                }
                processed += 4;
            }
            for (local_idx, row_out) in transposed_slice[processed * batch..]
                .chunks_exact_mut(batch)
                .enumerate()
            {
                let off = (start_row + processed + local_idx) * row_bytes;
                let w_row = &data[off..off + row_bytes];
                for (t, dst) in row_out.iter_mut().enumerate() {
                    *dst = quant_kernels::dot_q6_k_row(
                        caps,
                        w_row,
                        &act_q8_k[t * blocks..(t + 1) * blocks],
                    );
                }
            }
        }
        WeightSlice::Float(_) => {}
    }
}

#[cfg(all(windows, feature = "rust-qualification"))]
fn scatter_rows_to_batch(transposed: &[f32], output: usize, batch: usize, out: &mut [f32]) {
    for t in 0..batch {
        let dst = &mut out[t * output..(t + 1) * output];
        for (r, slot) in dst.iter_mut().enumerate() {
            *slot = transposed[r * batch + t];
        }
    }
}

/// Bit-exact matrix multiplication for a pre-transposed batch `[batch, input]`
/// using `4x8` / `4x4` SIMD tiles (or ARM64 `I8MM`/`SDOT` kernels) and `row_workers`.
pub(super) fn multiply_batch_prepared_into(
    weights: WeightMatrix<'_>,
    prepared: &PreparedBatch,
    raw_input: &[f32],
    workers: usize,
    transposed_scratch: &mut Vec<f32>,
    out: &mut [f32],
) {
    let output = weights.output;
    let batch = prepared.batch;
    let _ = raw_input;
    if batch == 0 || output == 0 {
        return;
    }
    #[cfg(all(windows, feature = "rust-qualification"))]
    if !matches!(weights.data, WeightSlice::Float(_)) {
        if batch == 1 {
            multiply_single_into(weights, raw_input, workers, out);
            return;
        }
        let (caps, act_q8_0, act_q8_k) =
            quantize_batch_activations(&[weights], raw_input, batch, weights.input);
        let needed = output * batch;
        if transposed_scratch.len() < needed {
            transposed_scratch.resize(needed, 0.0);
        }
        let transposed = &mut transposed_scratch[..needed];
        row_workers::run_aligned(transposed, workers, 4 * batch, &|elem_off, slice| {
            let start_row = elem_off / batch;
            batch_rows_quant_arm64(caps, weights, start_row, batch, &act_q8_0, &act_q8_k, slice);
        });
        scatter_rows_to_batch(transposed, output, batch, out);
        return;
    }

    let num_quads = prepared.num_quads;
    let level = simd_level();
    if num_quads > 0 && output > 0 {
        let row_stride = num_quads * 4;
        let needed = output * row_stride;
        if transposed_scratch.len() < needed {
            transposed_scratch.resize(needed, 0.0);
        }
        let transposed = &mut transposed_scratch[..needed];
        row_workers::run_aligned(transposed, workers, 4 * row_stride, &|elem_off, slice| {
            let start_row = elem_off / row_stride;
            dispatch!(
                level,
                simd => batch_rows_simd(simd, weights, start_row, &prepared.quads, num_quads, slice)
            );
        });
        let full_quads = prepared.batch / 4;
        if full_quads > 0 {
            let quad_out = &mut out[..full_quads * 4 * output];
            row_workers::run_aligned(quad_out, workers, 4 * output, &|elem_off, slice| {
                dispatch!(
                    level,
                    simd => scatter_quads_simd(simd, transposed, row_stride, output, elem_off, slice)
                );
            });
        }
        let rem = prepared.batch % 4;
        if rem > 0 {
            let tail_quad = full_quads * 4;
            let tail_out = &mut out[tail_quad * output..prepared.batch * output];
            for pos in 0..rem {
                let dst_row = &mut tail_out[pos * output..(pos + 1) * output];
                for (r, dst) in dst_row.iter_mut().enumerate() {
                    *dst = transposed[r * row_stride + tail_quad + pos];
                }
            }
        }
    }
}

/// Compute batched sibling projections sharing `prepared` / `raw_input` (`query + key + value`
/// or `gate + up`) in one quantized activation pass and unified worker pass.
pub(super) fn multiply_batch_group_into(
    targets: &mut [(WeightMatrix<'_>, &mut [f32])],
    prepared: &PreparedBatch,
    raw_input: &[f32],
    workers: usize,
    transposed_scratch: &mut Vec<f32>,
) {
    if targets.is_empty() {
        return;
    }
    if targets.len() == 1 {
        let (weights, out) = &mut targets[0];
        multiply_batch_prepared_into(
            *weights,
            prepared,
            raw_input,
            workers,
            transposed_scratch,
            out,
        );
        return;
    }
    let batch = prepared.batch;
    if batch == 0 {
        return;
    }
    if batch == 1 {
        multiply_single_group_into(targets, raw_input, workers, transposed_scratch);
        return;
    }

    #[cfg(all(windows, feature = "rust-qualification"))]
    {
        let matrices: Vec<WeightMatrix<'_>> = targets.iter().map(|(w, _)| *w).collect();
        let all_quant = matrices
            .iter()
            .all(|w| !matches!(w.data, WeightSlice::Float(_)));
        if all_quant {
            let (caps, act_q8_0, act_q8_k) =
                quantize_batch_activations(&matrices, raw_input, batch, prepared.input);
            let mut offsets = [0_usize; 4];
            let mut padded_rows = 0_usize;
            for (idx, w) in matrices.iter().enumerate() {
                offsets[idx] = padded_rows;
                padded_rows += w.output.next_multiple_of(4);
            }
            let needed = padded_rows * batch;
            if transposed_scratch.len() < needed {
                transposed_scratch.resize(needed, 0.0);
            }
            let transposed = &mut transposed_scratch[..needed];
            row_workers::run_aligned(transposed, workers, 4 * batch, &|elem_off, slice| {
                let start_row_global = elem_off / batch;
                let end_row_global = start_row_global + slice.len() / batch;
                for (idx, w) in matrices.iter().enumerate() {
                    let t_start = offsets[idx];
                    let t_end = t_start + w.output;
                    let lo = start_row_global.max(t_start);
                    let hi = end_row_global.min(t_end);
                    if lo >= hi {
                        continue;
                    }
                    let slice_lo = (lo - start_row_global) * batch;
                    let slice_hi = (hi - start_row_global) * batch;
                    batch_rows_quant_arm64(
                        caps,
                        *w,
                        lo - t_start,
                        batch,
                        &act_q8_0,
                        &act_q8_k,
                        &mut slice[slice_lo..slice_hi],
                    );
                }
            });
            for (idx, (w, out)) in targets.iter_mut().enumerate() {
                let elem_off = offsets[idx] * batch;
                scatter_rows_to_batch(
                    &transposed[elem_off..elem_off + w.output * batch],
                    w.output,
                    batch,
                    out,
                );
            }
            return;
        }
    }

    for (weights, out) in targets.iter_mut() {
        multiply_batch_prepared_into(
            *weights,
            prepared,
            raw_input,
            workers,
            transposed_scratch,
            out,
        );
    }
}

/// Compute scaled dot products between `query` and 4 key slices in exact left-to-right order.
pub(super) fn dot_4keys_seq(query: &[f32], keys: [&[f32]; 4], divisor: f32) -> [f32; 4] {
    let level = simd_level();
    dispatch!(level, simd => dot_4keys_seq_simd(simd, query, keys, divisor))
}

#[inline(always)]
fn dot_4keys_seq_simd<S: Simd>(
    simd: S,
    query: &[f32],
    keys: [&[f32]; 4],
    divisor: f32,
) -> [f32; 4] {
    let mut acc = f32x4::splat(simd, 0.0_f32);
    let len = query.len();
    let chunks = len / 4;
    for chunk_idx in 0..chunks {
        let base = chunk_idx * 4;
        let k0 = f32x4::from_slice(simd, &keys[0][base..base + 4]);
        let k1 = f32x4::from_slice(simd, &keys[1][base..base + 4]);
        let k2 = f32x4::from_slice(simd, &keys[2][base..base + 4]);
        let k3 = f32x4::from_slice(simd, &keys[3][base..base + 4]);
        let cols = transpose_4x4(simd, [k0, k1, k2, k3]);
        for k in 0..4 {
            acc += f32x4::splat(simd, query[base + k]) * cols[k];
        }
    }
    for d in (chunks * 4)..len {
        let kv = f32x4::from_slice(simd, &[keys[0][d], keys[1][d], keys[2][d], keys[3][d]]);
        acc += f32x4::splat(simd, query[d]) * kv;
    }
    (acc / f32x4::splat(simd, divisor)).into()
}

/// Accumulate `out[d] += w * v[d]` in exact left-to-right non-FMA order for 4 value rows.
pub(super) fn weighted_value_accum_4(out: &mut [f32], values: [&[f32]; 4], weights: [f32; 4]) {
    let level = simd_level();
    dispatch!(level, simd => weighted_value_accum_4_simd(simd, out, values, weights));
}

#[inline(always)]
fn weighted_value_accum_4_simd<S: Simd>(
    simd: S,
    out: &mut [f32],
    values: [&[f32]; 4],
    weights: [f32; 4],
) {
    let w0 = f32x4::splat(simd, weights[0]);
    let w1 = f32x4::splat(simd, weights[1]);
    let w2 = f32x4::splat(simd, weights[2]);
    let w3 = f32x4::splat(simd, weights[3]);
    let chunks = out.len() / 4;
    for chunk_idx in 0..chunks {
        let base = chunk_idx * 4;
        let dst = &mut out[base..base + 4];
        let mut ov = f32x4::from_slice(simd, dst);
        ov += w0 * f32x4::from_slice(simd, &values[0][base..base + 4]);
        ov += w1 * f32x4::from_slice(simd, &values[1][base..base + 4]);
        ov += w2 * f32x4::from_slice(simd, &values[2][base..base + 4]);
        ov += w3 * f32x4::from_slice(simd, &values[3][base..base + 4]);
        ov.store_slice(dst);
    }
    for d in (chunks * 4)..out.len() {
        out[d] += weights[0] * values[0][d];
        out[d] += weights[1] * values[1][d];
        out[d] += weights[2] * values[2][d];
        out[d] += weights[3] * values[3][d];
    }
}

/// Accumulate `out[d] += weight * v[d]` in exact non-FMA order.
pub(super) fn weighted_value_accum(out: &mut [f32], value: &[f32], weight: f32) {
    let level = simd_level();
    dispatch!(level, simd => weighted_value_accum_simd(simd, out, value, weight));
}

#[inline(always)]
fn weighted_value_accum_simd<S: Simd>(simd: S, out: &mut [f32], value: &[f32], weight: f32) {
    let wv = f32x4::splat(simd, weight);
    let (out_chunks, out_tail) = out.as_chunks_mut::<4>();
    let (v_chunks, v_tail) = value.as_chunks::<4>();
    for (dst, src) in out_chunks.iter_mut().zip(v_chunks) {
        let ov = f32x4::from_slice(simd, dst);
        let vv = f32x4::from_slice(simd, src);
        (ov + wv * vv).store_slice(dst);
    }
    for (dst, src) in out_tail.iter_mut().zip(v_tail) {
        *dst += weight * *src;
    }
}

/// Convert a bounded dimension or position to `f32`.
#[allow(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    reason = "Admitted encoder/decoder dimensions and positions are bounded by 65,536 and convert exactly."
)]
pub(super) const fn scalar_f32(value: usize) -> f32 {
    value as f32
}

/// Verify every slice element is finite, mapping failures with `invalid`.
pub(super) fn ensure_finite(
    values: &[f32],
    invalid: fn() -> std::io::Error,
) -> std::io::Result<()> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(invalid())
    }
}

/// Add `add` into `dst` element-wise across row workers and validate finiteness.
pub(super) fn add_residual_batch(
    dst: &mut [f32],
    add: &[f32],
    workers: usize,
    width: usize,
    invalid: fn() -> std::io::Error,
) -> std::io::Result<()> {
    row_workers::try_run_aligned(dst, workers, width, &|elem_off, slice| {
        let end = elem_off + slice.len();
        for (out, &delta) in slice.iter_mut().zip(&add[elem_off..end]) {
            *out += delta;
        }
        ensure_finite(slice, invalid)
    })
}

/// Evaluate one attention head over cached/history rows plus an optional current key/value pair.
///
/// Preserves exact left-to-right quad dot/accumulation order and scalar tail summation.
#[expect(
    clippy::too_many_arguments,
    reason = "Shared single-head attention kernel reuses caller score/output storage without allocating."
)]
pub(super) fn attend_head(
    q_head: &[f32],
    keys: &[f32],
    values: &[f32],
    history_rows: usize,
    row_width: usize,
    offset: usize,
    head_dim: usize,
    divisor: f32,
    current_kv: Option<(&[f32], &[f32])>,
    scores: &mut [f32],
    out_head: &mut [f32],
    invalid: fn() -> std::io::Error,
) -> std::io::Result<()> {
    let full_quads = history_rows / 4;
    for quad in 0..full_quads {
        let r = quad * 4;
        let k0 = &keys[r * row_width + offset..r * row_width + offset + head_dim];
        let k1 = &keys[(r + 1) * row_width + offset..(r + 1) * row_width + offset + head_dim];
        let k2 = &keys[(r + 2) * row_width + offset..(r + 2) * row_width + offset + head_dim];
        let k3 = &keys[(r + 3) * row_width + offset..(r + 3) * row_width + offset + head_dim];
        scores[r..r + 4].copy_from_slice(&dot_4keys_seq(q_head, [k0, k1, k2, k3], divisor));
    }
    for (row, score) in scores
        .iter_mut()
        .enumerate()
        .take(history_rows)
        .skip(full_quads * 4)
    {
        let k = &keys[row * row_width + offset..row * row_width + offset + head_dim];
        let mut dot = 0.0_f32;
        for (q_elem, k_elem) in q_head.iter().zip(k) {
            dot += *q_elem * *k_elem;
        }
        *score = dot / divisor;
    }
    if let Some((k, _)) = current_kv {
        let mut dot = 0.0_f32;
        for (q_elem, k_elem) in q_head.iter().zip(k) {
            dot += *q_elem * *k_elem;
        }
        scores[history_rows] = dot / divisor;
    }
    ensure_finite(scores, invalid)?;
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for score in &mut *scores {
        *score = (*score - max).exp();
        sum += *score;
    }
    if !sum.is_finite() || sum <= 0.0 {
        return Err(invalid());
    }
    out_head.fill(0.0);
    for quad in 0..full_quads {
        let r = quad * 4;
        let v0 = &values[r * row_width + offset..r * row_width + offset + head_dim];
        let v1 = &values[(r + 1) * row_width + offset..(r + 1) * row_width + offset + head_dim];
        let v2 = &values[(r + 2) * row_width + offset..(r + 2) * row_width + offset + head_dim];
        let v3 = &values[(r + 3) * row_width + offset..(r + 3) * row_width + offset + head_dim];
        let weights = [
            scores[r] / sum,
            scores[r + 1] / sum,
            scores[r + 2] / sum,
            scores[r + 3] / sum,
        ];
        weighted_value_accum_4(out_head, [v0, v1, v2, v3], weights);
    }
    for (row, &score) in scores
        .iter()
        .enumerate()
        .take(history_rows)
        .skip(full_quads * 4)
    {
        let v = &values[row * row_width + offset..row * row_width + offset + head_dim];
        weighted_value_accum(out_head, v, score / sum);
    }
    if let Some((_, v)) = current_kv {
        weighted_value_accum(out_head, v, scores[history_rows] / sum);
    }
    Ok(())
}

/// Narrows finite `f32` to IEEE binary16 bits using round-ties-to-even, returning `None`
/// when `value` is nonfinite or rounds to infinity (`|value| >= 65520`).
#[inline]
pub(super) fn checked_float_to_half(value: f32) -> Option<u16> {
    bonsai_ternary::checked_float_to_half(value)
}

/// Narrow finite `f32` values into IEEE binary16 words (`FCVTN` SIMD on ARM64 Windows),
/// returning `false` on non-finite input, length mismatch, or overflow past finite `f16` range.
pub(super) fn narrow_f32_to_f16_slice(src: &[f32], dst: &mut [u16]) -> bool {
    bonsai_neon_dot::narrow_f32_slice_checked(src, dst)
}

/// Widen IEEE binary16 words into `f32` values (`FCVTL` SIMD on ARM64 Windows).
pub(super) fn widen_f16_slice(src: &[u16], dst: &mut [f32]) {
    bonsai_neon_dot::widen_f16_slice(src, dst);
}

/// Evaluate one attention head over packed `u16` (`F16`) cached KV storage without
/// retaining a shadow `f32` cache, producing identical bits to `attend_head`.
#[expect(
    clippy::too_many_arguments,
    reason = "Shared single-head attention kernel reuses caller score/output storage without allocations."
)]
pub(super) fn attend_head_f16(
    q_head: &[f32],
    keys: &[u16],
    values: &[u16],
    history_rows: usize,
    row_width: usize,
    offset: usize,
    head_dim: usize,
    divisor: f32,
    current_kv: Option<(&[f32], &[f32])>,
    scores: &mut [f32],
    out_head: &mut [f32],
    invalid: fn() -> std::io::Error,
) -> std::io::Result<()> {
    if head_dim > 256 || q_head.len() != head_dim || out_head.len() != head_dim {
        return Err(invalid());
    }
    let mut k_buf = [[0.0_f32; 256]; 4];
    let full_quads = history_rows / 4;
    for quad in 0..full_quads {
        let r = quad * 4;
        for row_off in 0..4 {
            let src = &keys
                [(r + row_off) * row_width + offset..(r + row_off) * row_width + offset + head_dim];
            widen_f16_slice(src, &mut k_buf[row_off][..head_dim]);
        }
        let k_refs = [
            &k_buf[0][..head_dim],
            &k_buf[1][..head_dim],
            &k_buf[2][..head_dim],
            &k_buf[3][..head_dim],
        ];
        scores[r..r + 4].copy_from_slice(&dot_4keys_seq(q_head, k_refs, divisor));
    }
    for (row, score) in scores
        .iter_mut()
        .enumerate()
        .take(history_rows)
        .skip(full_quads * 4)
    {
        let src = &keys[row * row_width + offset..row * row_width + offset + head_dim];
        widen_f16_slice(src, &mut k_buf[0][..head_dim]);
        let mut dot = 0.0_f32;
        for (q_elem, &k_elem) in q_head.iter().zip(&k_buf[0][..head_dim]) {
            dot += *q_elem * k_elem;
        }
        *score = dot / divisor;
    }
    if let Some((k, _)) = current_kv {
        let mut dot = 0.0_f32;
        for (q_elem, k_elem) in q_head.iter().zip(k) {
            dot += *q_elem * *k_elem;
        }
        scores[history_rows] = dot / divisor;
    }
    ensure_finite(scores, invalid)?;
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for score in &mut *scores {
        *score = (*score - max).exp();
        sum += *score;
    }
    if !sum.is_finite() || sum <= 0.0 {
        return Err(invalid());
    }
    out_head.fill(0.0);
    let mut v_buf = [[0.0_f32; 256]; 4];
    for quad in 0..full_quads {
        let r = quad * 4;
        for row_off in 0..4 {
            let src = &values
                [(r + row_off) * row_width + offset..(r + row_off) * row_width + offset + head_dim];
            widen_f16_slice(src, &mut v_buf[row_off][..head_dim]);
        }
        let v_refs = [
            &v_buf[0][..head_dim],
            &v_buf[1][..head_dim],
            &v_buf[2][..head_dim],
            &v_buf[3][..head_dim],
        ];
        let weights = [
            scores[r] / sum,
            scores[r + 1] / sum,
            scores[r + 2] / sum,
            scores[r + 3] / sum,
        ];
        weighted_value_accum_4(out_head, v_refs, weights);
    }
    for (row, &score) in scores
        .iter()
        .enumerate()
        .take(history_rows)
        .skip(full_quads * 4)
    {
        let src = &values[row * row_width + offset..row * row_width + offset + head_dim];
        widen_f16_slice(src, &mut v_buf[0][..head_dim]);
        weighted_value_accum(out_head, &v_buf[0][..head_dim], score / sum);
    }
    if let Some((_, v)) = current_kv {
        weighted_value_accum(out_head, v, scores[history_rows] / sum);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        PreparedBatch, WeightMatrix, WeightSlice, attend_head, attend_head_f16, decode_q4_k_row,
        decode_q6_k_row, dot_4keys_seq, multiply_batch_group_into, multiply_batch_prepared_into,
        multiply_single_group_into, multiply_single_into, narrow_f32_to_f16_slice,
        weighted_value_accum, weighted_value_accum_4, widen_f16_slice,
    };
    use std::io;

    fn scalar_multiply_reference(weights: WeightMatrix<'_>, input: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0_f32; weights.output];
        for (row, target) in output.iter_mut().enumerate() {
            match weights.data {
                WeightSlice::Float(data) => {
                    for (weight, activation) in data[row * weights.input..(row + 1) * weights.input]
                        .iter()
                        .zip(input)
                    {
                        *target += *weight * *activation;
                    }
                }
                WeightSlice::Q8(data) => {
                    let row_bytes = weights.input / 32 * 34;
                    for (block, activations) in data[row * row_bytes..(row + 1) * row_bytes]
                        .as_chunks::<34>()
                        .0
                        .iter()
                        .zip(input.as_chunks::<32>().0)
                    {
                        let scale = super::half_scale(&block[0..2]);
                        for (&byte, activation) in block[2..].iter().zip(activations) {
                            *target += (scale * f32::from(byte.cast_signed())) * *activation;
                        }
                    }
                }
                WeightSlice::Q4K(data) => {
                    let row_bytes = weights.input / 256 * 144;
                    *target = super::dot_q4_k_row_scalar(
                        &data[row * row_bytes..(row + 1) * row_bytes],
                        input,
                    );
                }
                WeightSlice::Q6K(data) => {
                    let row_bytes = weights.input / 256 * 210;
                    *target = super::dot_q6_k_row_scalar(
                        &data[row * row_bytes..(row + 1) * row_bytes],
                        input,
                    );
                }
            }
        }
        output
    }

    #[cfg(all(windows, feature = "rust-qualification"))]
    fn scalar_q8_x_q8_reference(weights: WeightMatrix<'_>, input: &[f32]) -> Vec<f32> {
        let WeightSlice::Q8(data) = weights.data else {
            return scalar_multiply_reference(weights, input);
        };
        let row_bytes = weights.input / 32 * 34;
        let mut act_blocks = Vec::with_capacity(weights.input / 32);
        for chunk in input.as_chunks::<32>().0 {
            let amax = chunk.iter().copied().fold(0.0_f32, |m, v| m.max(v.abs()));
            if amax == 0.0 || !amax.is_finite() {
                act_blocks.push((0.0_f32, [0_i32; 32]));
                continue;
            }
            let d = amax / 127.0_f32;
            let inv_d = 127.0_f32 / amax;
            let mut qs = [0_i32; 32];
            for (dst, &val) in qs.iter_mut().zip(chunk) {
                let scaled = (val * inv_d).round_ties_even().clamp(-127.0, 127.0);
                let mag = (0_u8..=127)
                    .find(|&c| f32::from(c) == scaled.abs())
                    .unwrap_or(0);
                *dst = if scaled < 0.0 {
                    -i32::from(mag)
                } else {
                    i32::from(mag)
                };
            }
            act_blocks.push((d, qs));
        }
        let mut output = vec![0.0_f32; weights.output];
        for (row, target) in output.iter_mut().enumerate() {
            let mut sum = 0.0_f32;
            for (w_blk, (act_d, act_qs)) in data[row * row_bytes..(row + 1) * row_bytes]
                .as_chunks::<34>()
                .0
                .iter()
                .zip(&act_blocks)
            {
                let w_d = super::half_scale(&w_blk[0..2]);
                let mut idot = 0_i32;
                for (&wb, &ab) in w_blk[2..].iter().zip(act_qs) {
                    idot += i32::from(wb.cast_signed()) * ab;
                }
                let hi = i16::try_from(idot.div_euclid(65_536)).unwrap_or(0);
                let lo = u16::try_from(idot.rem_euclid(65_536)).unwrap_or(0);
                let idot_f32 = f32::from(hi) * 65_536.0_f32 + f32::from(lo);
                sum += w_d * (*act_d * idot_f32);
            }
            *target = sum;
        }
        output
    }

    #[test]
    fn packed_asr_f32_and_q8_single_and_batch_match_scalar_bits() -> io::Result<()> {
        let scales = [0x3555_u16, 0xb955, 0x3c00, 0x8000, 0x3801];
        for (is_q8, input) in [(false, 37_usize), (true, 96_usize)] {
            for output in [1_usize, 3, 4, 5, 9, 11] {
                let raw_q8;
                let raw_f32;
                let weights = if is_q8 {
                    let row_bytes = input / 32 * 34;
                    let mut bytes = vec![0_u8; row_bytes * output];
                    for (blk_idx, block) in bytes.as_chunks_mut::<34>().0.iter_mut().enumerate() {
                        let [s0, s1, qs @ ..] = block;
                        [*s0, *s1] = scales[blk_idx % scales.len()].to_le_bytes();
                        for (q_idx, byte) in qs.iter_mut().enumerate() {
                            let val = i8::try_from(
                                ((blk_idx * 17 + q_idx * 13) % 251).cast_signed() - 125,
                            )
                            .unwrap_or(0);
                            *byte = val.cast_unsigned();
                        }
                    }
                    raw_q8 = bytes;
                    WeightMatrix {
                        input,
                        output,
                        data: WeightSlice::Q8(&raw_q8),
                    }
                } else {
                    raw_f32 = (0..input * output)
                        .map(|idx| {
                            if idx % 19 == 0 {
                                -0.0
                            } else {
                                (f32::from(u16::try_from(idx % 37).unwrap_or(0)) - 18.0) * 0.0625
                            }
                        })
                        .collect::<Vec<_>>();
                    WeightMatrix {
                        input,
                        output,
                        data: WeightSlice::Float(&raw_f32),
                    }
                };

                for batch in [1_usize, 2, 3, 4, 5, 7, 8, 9, 15, 16, 19] {
                    let raw_input = (0..batch * input)
                        .map(|idx| {
                            if idx % 23 == 0 {
                                -0.0
                            } else {
                                (f32::from(u16::try_from((idx * 7) % 41).unwrap_or(0)) - 20.0)
                                    * 0.03125
                            }
                        })
                        .collect::<Vec<_>>();
                    let mut expected = Vec::with_capacity(batch * output);
                    for pos in 0..batch {
                        expected.extend(scalar_multiply_reference(
                            weights,
                            &raw_input[pos * input..(pos + 1) * input],
                        ));
                    }
                    let expected_bits: Vec<u32> = expected.iter().map(|v| v.to_bits()).collect();
                    let mut worker1_batch_bits: Option<Vec<u32>> = None;

                    for workers in [1_usize, 3, 8] {
                        let prepared = PreparedBatch::new(&raw_input, batch, input);
                        let mut batch_out = vec![0.0_f32; batch * output];
                        let mut scratch = Vec::new();
                        multiply_batch_prepared_into(
                            weights,
                            &prepared,
                            &raw_input,
                            workers,
                            &mut scratch,
                            &mut batch_out,
                        );
                        let batch_bits: Vec<u32> = batch_out.iter().map(|v| v.to_bits()).collect();
                        if let Some(prev) = &worker1_batch_bits {
                            assert_eq!(
                                &batch_bits, prev,
                                "worker invariance q8={is_q8} out={output} batch={batch} workers={workers}"
                            );
                        } else {
                            worker1_batch_bits = Some(batch_bits.clone());
                        }
                        let mut all_single_bits = Vec::with_capacity(batch * output);
                        let mut single_out = vec![0.0_f32; output];
                        for pos in 0..batch {
                            multiply_single_into(
                                weights,
                                &raw_input[pos * input..(pos + 1) * input],
                                workers,
                                &mut single_out,
                            );
                            all_single_bits.extend(single_out.iter().map(|v| v.to_bits()));
                        }
                        assert_eq!(
                            all_single_bits, batch_bits,
                            "single vs batch q8={is_q8} out={output} batch={batch} workers={workers}"
                        );
                        #[cfg(not(all(windows, feature = "rust-qualification")))]
                        assert_eq!(
                            batch_bits, expected_bits,
                            "exact bits q8={is_q8} out={output} batch={batch} workers={workers}"
                        );
                        #[cfg(all(windows, feature = "rust-qualification"))]
                        if !is_q8 {
                            assert_eq!(
                                batch_bits, expected_bits,
                                "exact bits q8={is_q8} out={output} batch={batch} workers={workers}"
                            );
                        } else {
                            let mut q8_ref = Vec::with_capacity(batch * output);
                            for pos in 0..batch {
                                q8_ref.extend(scalar_q8_x_q8_reference(
                                    weights,
                                    &raw_input[pos * input..(pos + 1) * input],
                                ));
                            }
                            let q8_ref_bits: Vec<u32> =
                                q8_ref.iter().map(|v| v.to_bits()).collect();
                            assert_eq!(
                                batch_bits, q8_ref_bits,
                                "exact Q8xQ8 bits out={output} batch={batch} workers={workers}"
                            );
                            for (&got, &want) in batch_out.iter().zip(&expected) {
                                let tol = 2.5_f32.max(want.abs() * 0.02);
                                assert!(
                                    (got - want).abs() <= tol,
                                    "q8={is_q8} out={output} batch={batch} got={got} want={want}"
                                );
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn packed_q4_k_q6_k_fused_groups_and_f16_kv_match_reference() -> io::Result<()> {
        let input = 256_usize;
        let out_a = 9_usize;
        let out_b = 5_usize;
        let mut q4 = vec![0_u8; out_a * 144];
        for (row, block) in q4.chunks_exact_mut(144).enumerate() {
            block[0..2].copy_from_slice(&0x3800_u16.to_le_bytes());
            block[2..4].copy_from_slice(&0x3400_u16.to_le_bytes());
            for (i, byte) in block[4..16].iter_mut().enumerate() {
                *byte = u8::try_from((row * 7 + i * 3 + 1) % 31).unwrap_or(1);
            }
            for (i, byte) in block[16..].iter_mut().enumerate() {
                *byte = u8::try_from((row * 13 + i * 5) % 256).unwrap_or(0);
            }
        }
        let mut q6 = vec![0_u8; out_b * 210];
        for (row, block) in q6.chunks_exact_mut(210).enumerate() {
            block[208..210].copy_from_slice(&0x3800_u16.to_le_bytes());
            for (i, byte) in block[..192].iter_mut().enumerate() {
                *byte = u8::try_from((row * 11 + i * 9) % 256).unwrap_or(0);
            }
            for (i, byte) in block[192..208].iter_mut().enumerate() {
                let sc = i8::try_from(((row * 3 + i) % 15).cast_signed() - 7).unwrap_or(0);
                *byte = sc.cast_unsigned();
            }
        }

        let mut dec4 = vec![0.0_f32; input];
        decode_q4_k_row(&q4[..144], &mut dec4);
        let unit = vec![1.0_f32; input];
        assert!(
            (dec4.iter().sum::<f32>() - super::dot_q4_k_row_scalar(&q4[..144], &unit)).abs() < 1e-3
        );
        let mut dec6 = vec![0.0_f32; input];
        decode_q6_k_row(&q6[..210], &mut dec6);
        assert!(
            (dec6.iter().sum::<f32>() - super::dot_q6_k_row_scalar(&q6[..210], &unit)).abs() < 1e-3
        );

        let wm4 = WeightMatrix {
            input,
            output: out_a,
            data: WeightSlice::Q4K(&q4),
        };
        let wm6 = WeightMatrix {
            input,
            output: out_b,
            data: WeightSlice::Q6K(&q6),
        };
        let batch = 5_usize;
        let raw_in: Vec<f32> = (0..batch * input)
            .map(|i| (f32::from(u16::try_from(i % 29).unwrap_or(0)) - 14.0) * 0.03125)
            .collect();
        let prepared = PreparedBatch::new(&raw_in, batch, input);

        let mut exp_a = vec![0.0_f32; batch * out_a];
        let mut exp_b = vec![0.0_f32; batch * out_b];
        let mut scratch = Vec::new();
        multiply_batch_prepared_into(wm4, &prepared, &raw_in, 1, &mut scratch, &mut exp_a);
        multiply_batch_prepared_into(wm6, &prepared, &raw_in, 1, &mut scratch, &mut exp_b);

        for workers in [1_usize, 4] {
            let mut got_a = vec![0.0_f32; batch * out_a];
            let mut got_b = vec![0.0_f32; batch * out_b];
            multiply_batch_group_into(
                &mut [(wm4, &mut got_a), (wm6, &mut got_b)],
                &prepared,
                &raw_in,
                workers,
                &mut scratch,
            );
            assert_eq!(
                got_a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                exp_a.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
            assert_eq!(
                got_b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                exp_b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );

            let mut s_a = vec![0.0_f32; out_a];
            let mut s_b = vec![0.0_f32; out_b];
            multiply_single_group_into(
                &mut [(wm4, &mut s_a), (wm6, &mut s_b)],
                &raw_in[..input],
                workers,
                &mut scratch,
            );
            assert_eq!(
                s_a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                exp_a[..out_a]
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                s_b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                exp_b[..out_b]
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            );
        }

        // Verify F16 KV narrow/widen and attend_head_f16 match attend_head bit-for-bit.
        let src_kv: Vec<f32> = (0..120_usize)
            .map(|i| {
                if i % 11 == 0 {
                    -0.0
                } else {
                    (f32::from(u8::try_from(i % 19).unwrap_or(0)) - 9.0) * 0.125
                }
            })
            .collect();
        let mut f16_bits = vec![0_u16; src_kv.len()];
        assert!(narrow_f32_to_f16_slice(&src_kv, &mut f16_bits));
        let mut f32_cache = vec![0.0_f32; src_kv.len()];
        widen_f16_slice(&f16_bits, &mut f32_cache);
        let q_head = vec![0.25_f32; 8];
        let mut s_a = [0.0_f32; 6];
        let mut s_b = [0.0_f32; 6];
        let mut out_a = [0.0_f32; 8];
        let mut out_b = [0.0_f32; 8];
        let inv = || io::Error::other("err");
        attend_head(
            &q_head,
            &f32_cache,
            &f32_cache,
            5,
            16,
            8,
            8,
            8.0_f32.sqrt(),
            Some((&q_head, &q_head)),
            &mut s_a,
            &mut out_a,
            inv,
        )?;
        attend_head_f16(
            &q_head,
            &f16_bits,
            &f16_bits,
            5,
            16,
            8,
            8,
            8.0_f32.sqrt(),
            Some((&q_head, &q_head)),
            &mut s_b,
            &mut out_b,
            inv,
        )?;
        assert_eq!(
            out_b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            out_a.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        Ok(())
    }

    #[test]
    fn packed_attention_dots_and_values_match_scalar_bits() {
        for dim in [2_usize, 4, 6, 64] {
            let query: Vec<f32> = (0..dim)
                .map(|i| {
                    if i % 5 == 0 {
                        -0.0
                    } else {
                        (f32::from(u16::try_from(i).unwrap_or(0)) - 3.0) * 0.125
                    }
                })
                .collect();
            let keys: [Vec<f32>; 4] = std::array::from_fn(|k| {
                (0..dim)
                    .map(|i| {
                        ((f32::from(u16::try_from((i + 1) * (k + 3)).unwrap_or(0)) % 17.0) - 8.0)
                            * 0.25
                    })
                    .collect()
            });
            let divisor = f32::from(u16::try_from(dim).unwrap_or(1)).sqrt();
            let simd_dots =
                dot_4keys_seq(&query, [&keys[0], &keys[1], &keys[2], &keys[3]], divisor);
            for (k_idx, &actual) in simd_dots.iter().enumerate() {
                let mut sum = 0.0_f32;
                for (q, k) in query.iter().zip(&keys[k_idx]) {
                    sum += *q * *k;
                }
                let expected = sum / divisor;
                assert_eq!(actual.to_bits(), expected.to_bits(), "dim={dim} k={k_idx}");
            }

            let weights = [0.25_f32, -0.0, 0.375, 0.125];
            let mut out_simd = vec![-0.0_f32; dim];
            let mut out_scalar = out_simd.clone();
            weighted_value_accum_4(
                &mut out_simd,
                [&keys[0], &keys[1], &keys[2], &keys[3]],
                weights,
            );
            for (k_slice, w) in keys.iter().zip(weights) {
                for (dst, v) in out_scalar.iter_mut().zip(k_slice) {
                    *dst += w * *v;
                }
            }
            assert_eq!(
                out_simd.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                out_scalar.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "accum_4 dim={dim}"
            );

            let mut single_simd = vec![0.5_f32; dim];
            let mut single_scalar = single_simd.clone();
            weighted_value_accum(&mut single_simd, &keys[2], 0.3125);
            for (dst, v) in single_scalar.iter_mut().zip(&keys[2]) {
                *dst += 0.3125 * *v;
            }
            assert_eq!(
                single_simd.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                single_scalar
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                "accum_1 dim={dim}"
            );
        }
    }
}
