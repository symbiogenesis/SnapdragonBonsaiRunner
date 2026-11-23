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

/// Borrowed weight slice for row-major `F32`, `Q8_0`, `Q4_K`, repacked `Q4_K`, or `Q6_K` matrices.
#[derive(Clone, Copy)]
pub(super) enum WeightSlice<'a> {
    Float(&'a [f32]),
    Q8(&'a [u8]),
    Q4K(&'a [u8]),
    #[cfg(all(windows, feature = "rust-qualification"))]
    RepackedQ4K8(&'a [u8]),
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
fn unsigned_lanes<S: Simd>(simd: S, bytes: u8x16<S>) -> [f32x4<S>; 4] {
    let (lo, hi) = simd.widen_u8x16(bytes);
    let (a, b) = simd.widen_u16x8(lo);
    let (c, d) = simd.widen_u16x8(hi);
    [a, b, c, d].map(|value| simd.cvt_f32_u32x4(value))
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

#[cfg(test)]
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

#[cfg(test)]
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
fn single_q4_k_4rows<S: Simd>(simd: S, rows: [&[u8]; 4], input: &[f32], out: &mut [f32]) {
    let mut acc = f32x4::splat(simd, 0.0_f32);
    let mask_0f = u8x16::splat(simd, 0x0f);
    let blocks = input.len() / 256;
    for block_idx in 0..blocks {
        let b_off = block_idx * 144;
        let i_off = block_idx * 256;
        let d_dmin = rows.map(|row| {
            (
                half_scale(&row[b_off..b_off + 2]),
                half_scale(&row[b_off + 2..b_off + 4]),
                &row[b_off + 4..b_off + 16],
            )
        });
        for chunk_idx in 0..4 {
            let params = d_dmin.map(|(d, dmin, scales)| {
                let (s_lo, m_lo) = unpack_q4_k_scale_min(chunk_idx * 2, scales);
                let (s_hi, m_hi) = unpack_q4_k_scale_min(chunk_idx * 2 + 1, scales);
                (
                    (
                        f32x4::splat(simd, d * f32::from(s_lo)),
                        f32x4::splat(simd, dmin * f32::from(m_lo)),
                    ),
                    (
                        f32x4::splat(simd, d * f32::from(s_hi)),
                        f32x4::splat(simd, dmin * f32::from(m_hi)),
                    ),
                )
            });
            let qs_base = b_off + 16 + chunk_idx * 32;
            let in_base = i_off + chunk_idx * 64;
            for (half_idx, is_hi) in [false, true].into_iter().enumerate() {
                let p_off = in_base + half_idx * 32;
                for part in 0..2 {
                    let q_off = qs_base + part * 16;
                    let col_part = p_off + part * 16;
                    let weights: [[f32x4<S>; 4]; 4] = std::array::from_fn(|r| {
                        let raw = u8x16::from_slice(simd, &rows[r][q_off..q_off + 16]);
                        let nibs = if is_hi { raw >> 4 } else { raw & mask_0f };
                        let (sc, min) = if is_hi { params[r].1 } else { params[r].0 };
                        unsigned_lanes(simd, nibs).map(|qv| sc * qv - min)
                    });
                    for group in 0..4 {
                        let col_base = col_part + group * 4;
                        let cols = transpose_4x4(
                            simd,
                            [
                                weights[0][group],
                                weights[1][group],
                                weights[2][group],
                                weights[3][group],
                            ],
                        );
                        for k in 0..4 {
                            acc += cols[k] * f32x4::splat(simd, input[col_base + k]);
                        }
                    }
                }
            }
        }
    }
    acc.store_slice(out);
}

#[inline(always)]
fn single_q4_k_1row<S: Simd>(simd: S, row: &[u8], input: &[f32]) -> f32 {
    let mut acc = 0.0_f32;
    let mask_0f = u8x16::splat(simd, 0x0f);
    for (block, in_block) in row
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
            let sc_lo = f32x4::splat(simd, d * f32::from(s_lo));
            let min_lo = f32x4::splat(simd, dmin * f32::from(m_lo));
            let sc_hi = f32x4::splat(simd, d * f32::from(s_hi));
            let min_hi = f32x4::splat(simd, dmin * f32::from(m_hi));
            let qs_base = chunk_idx * 32;
            let in_base = chunk_idx * 64;
            for (half_idx, is_hi) in [false, true].into_iter().enumerate() {
                let (sc, min) = if is_hi {
                    (sc_hi, min_hi)
                } else {
                    (sc_lo, min_lo)
                };
                let p_off = in_base + half_idx * 32;
                for part in 0..2 {
                    let q_off = qs_base + part * 16;
                    let col_part = p_off + part * 16;
                    let raw = u8x16::from_slice(simd, &qs[q_off..q_off + 16]);
                    let nibs = if is_hi { raw >> 4 } else { raw & mask_0f };
                    for (group_idx, qv) in unsigned_lanes(simd, nibs).into_iter().enumerate() {
                        let w = sc * qv - min;
                        let col_base = col_part + group_idx * 4;
                        let a = f32x4::from_slice(simd, &in_block[col_base..col_base + 4]);
                        let prod: [f32; 4] = (w * a).into();
                        acc += prod[0];
                        acc += prod[1];
                        acc += prod[2];
                        acc += prod[3];
                    }
                }
            }
        }
    }
    acc
}

#[inline(always)]
fn single_q6_k_4rows<S: Simd>(simd: S, rows: [&[u8]; 4], input: &[f32], out: &mut [f32]) {
    let mut acc = f32x4::splat(simd, 0.0_f32);
    let mask_0f = u8x16::splat(simd, 0x0f);
    let mask_03 = u8x16::splat(simd, 0x03);
    let thirty_two = f32x4::splat(simd, 32.0_f32);
    let blocks = input.len() / 256;
    for block_idx in 0..blocks {
        let b_off = block_idx * 210;
        let i_off = block_idx * 256;
        let d = rows.map(|row| half_scale(&row[b_off + 208..b_off + 210]));
        for n128 in 0..2 {
            let ql_base = b_off + n128 * 64;
            let qh_base = b_off + 128 + n128 * 32;
            let sc_base = b_off + 192 + n128 * 8;
            let in_base = i_off + n128 * 128;
            for part in 0..2 {
                let p_off = part * 16;
                let dequant_q = |r: usize, scale_idx: usize, quants: u8x16<S>| -> [f32x4<S>; 4] {
                    let sc = f32x4::splat(
                        simd,
                        d[r] * f32::from(rows[r][sc_base + scale_idx].cast_signed()),
                    );
                    unsigned_lanes(simd, quants).map(|qv| sc * (qv - thirty_two))
                };
                let weights: [[[f32x4<S>; 4]; 4]; 4] = std::array::from_fn(|r| {
                    let ql0 =
                        u8x16::from_slice(simd, &rows[r][ql_base + p_off..ql_base + p_off + 16]);
                    let ql1 = u8x16::from_slice(
                        simd,
                        &rows[r][ql_base + 32 + p_off..ql_base + 32 + p_off + 16],
                    );
                    let qh_v =
                        u8x16::from_slice(simd, &rows[r][qh_base + p_off..qh_base + p_off + 16]);
                    let q1 = (ql0 & mask_0f) | ((qh_v & mask_03) << 4);
                    let q2 = (ql1 & mask_0f) | (((qh_v >> 2) & mask_03) << 4);
                    let q3 = (ql0 >> 4) | (((qh_v >> 4) & mask_03) << 4);
                    let q4 = (ql1 >> 4) | (((qh_v >> 6) & mask_03) << 4);
                    [
                        dequant_q(r, part, q1),
                        dequant_q(r, part + 2, q2),
                        dequant_q(r, part + 4, q3),
                        dequant_q(r, part + 6, q4),
                    ]
                });
                for group in 0..4 {
                    let l_base = p_off + group * 4;
                    let c1 = transpose_4x4(
                        simd,
                        [
                            weights[0][0][group],
                            weights[1][0][group],
                            weights[2][0][group],
                            weights[3][0][group],
                        ],
                    );
                    let c2 = transpose_4x4(
                        simd,
                        [
                            weights[0][1][group],
                            weights[1][1][group],
                            weights[2][1][group],
                            weights[3][1][group],
                        ],
                    );
                    let c3 = transpose_4x4(
                        simd,
                        [
                            weights[0][2][group],
                            weights[1][2][group],
                            weights[2][2][group],
                            weights[3][2][group],
                        ],
                    );
                    let c4 = transpose_4x4(
                        simd,
                        [
                            weights[0][3][group],
                            weights[1][3][group],
                            weights[2][3][group],
                            weights[3][3][group],
                        ],
                    );
                    for k in 0..4 {
                        let col = in_base + l_base + k;
                        acc += c1[k] * f32x4::splat(simd, input[col]);
                        acc += c2[k] * f32x4::splat(simd, input[col + 32]);
                        acc += c3[k] * f32x4::splat(simd, input[col + 64]);
                        acc += c4[k] * f32x4::splat(simd, input[col + 96]);
                    }
                }
            }
        }
    }
    acc.store_slice(out);
}

#[inline(always)]
fn single_q6_k_1row<S: Simd>(simd: S, row: &[u8], input: &[f32]) -> f32 {
    let mut acc = 0.0_f32;
    let mask_0f = u8x16::splat(simd, 0x0f);
    let mask_03 = u8x16::splat(simd, 0x03);
    let thirty_two = f32x4::splat(simd, 32.0_f32);
    for (block, in_block) in row
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
            for part in 0..2 {
                let p_off = part * 16;
                let ql0 = u8x16::from_slice(simd, &ql_s[p_off..p_off + 16]);
                let ql1 = u8x16::from_slice(simd, &ql_s[32 + p_off..32 + p_off + 16]);
                let qh_v = u8x16::from_slice(simd, &qh_s[p_off..p_off + 16]);
                let dequant = |scale_idx: usize, quants: u8x16<S>| -> [f32x4<S>; 4] {
                    let sc = f32x4::splat(simd, d * f32::from(sc_s[scale_idx].cast_signed()));
                    unsigned_lanes(simd, quants).map(|qv| sc * (qv - thirty_two))
                };
                let w1 = dequant(part, (ql0 & mask_0f) | ((qh_v & mask_03) << 4));
                let w2 = dequant(part + 2, (ql1 & mask_0f) | (((qh_v >> 2) & mask_03) << 4));
                let w3 = dequant(part + 4, (ql0 >> 4) | (((qh_v >> 4) & mask_03) << 4));
                let w4 = dequant(part + 6, (ql1 >> 4) | (((qh_v >> 6) & mask_03) << 4));
                for group in 0..4 {
                    let l_base = p_off + group * 4;
                    let p1: [f32; 4] =
                        (w1[group] * f32x4::from_slice(simd, &in_s[l_base..l_base + 4])).into();
                    let p2: [f32; 4] = (w2[group]
                        * f32x4::from_slice(simd, &in_s[l_base + 32..l_base + 36]))
                    .into();
                    let p3: [f32; 4] = (w3[group]
                        * f32x4::from_slice(simd, &in_s[l_base + 64..l_base + 68]))
                    .into();
                    let p4: [f32; 4] = (w4[group]
                        * f32x4::from_slice(simd, &in_s[l_base + 96..l_base + 100]))
                    .into();
                    for k in 0..4 {
                        acc += p1[k];
                        acc += p2[k];
                        acc += p3[k];
                        acc += p4[k];
                    }
                }
            }
        }
    }
    acc
}

#[inline(always)]
fn four_rows<T>(data: &[T], start_row: usize, stride: usize) -> [&[T]; 4] {
    let row0 = start_row * stride;
    std::array::from_fn(|r| &data[row0 + r * stride..row0 + (r + 1) * stride])
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
                single_f32_4rows(
                    simd,
                    four_rows(data, start_row + processed, width),
                    input,
                    dst,
                );
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
                single_q8_4rows(
                    simd,
                    four_rows(data, start_row + processed, row_bytes),
                    input,
                    dst,
                );
                processed += 4;
            }
            for (local_idx, dst) in out[processed..].iter_mut().enumerate() {
                let off = (start_row + processed + local_idx) * row_bytes;
                *dst = single_q8_1row(simd, &data[off..off + row_bytes], input);
            }
        }
        WeightSlice::Q4K(data) => {
            let row_bytes = width / 256 * 144;
            for dst in out.as_chunks_mut::<4>().0 {
                single_q4_k_4rows(
                    simd,
                    four_rows(data, start_row + processed, row_bytes),
                    input,
                    dst,
                );
                processed += 4;
            }
            for (local_idx, dst) in out[processed..].iter_mut().enumerate() {
                let off = (start_row + processed + local_idx) * row_bytes;
                *dst = single_q4_k_1row(simd, &data[off..off + row_bytes], input);
            }
        }
        #[cfg(all(windows, feature = "rust-qualification"))]
        WeightSlice::RepackedQ4K8(_) => {}
        WeightSlice::Q6K(data) => {
            let row_bytes = width / 256 * 210;
            for dst in out.as_chunks_mut::<4>().0 {
                single_q6_k_4rows(
                    simd,
                    four_rows(data, start_row + processed, row_bytes),
                    input,
                    dst,
                );
                processed += 4;
            }
            for (local_idx, dst) in out[processed..].iter_mut().enumerate() {
                let off = (start_row + processed + local_idx) * row_bytes;
                *dst = single_q6_k_1row(simd, &data[off..off + row_bytes], input);
            }
        }
    }
}

#[cfg(all(windows, feature = "rust-qualification"))]
fn run_single_rows_quant<A>(
    data: &[u8],
    start_row: usize,
    row_bytes: usize,
    act: &[A],
    out: &mut [f32],
    dot4: impl Fn([&[u8]; 4], &[A]) -> [f32; 4],
    dot1: impl Fn(&[u8], &[A]) -> f32,
) {
    let mut processed = 0;
    for dst in out.as_chunks_mut::<4>().0 {
        *dst = dot4(four_rows(data, start_row + processed, row_bytes), act);
        processed += 4;
    }
    for (local_idx, dst) in out[processed..].iter_mut().enumerate() {
        let off = (start_row + processed + local_idx) * row_bytes;
        *dst = dot1(&data[off..off + row_bytes], act);
    }
}

#[cfg(all(windows, feature = "rust-qualification"))]
fn run_single_rows_repacked_q4_k(
    caps: Arm64QuantCaps,
    data: &[u8],
    width: usize,
    start_row: usize,
    act: &[Q8_KActBlock],
    out: &mut [f32],
) {
    let strip_bytes = (width / quant_kernels::QK_K) * quant_kernels::Q4_K_REPACK8_BLOCK_BYTES;
    let mut idx = 0_usize;
    while idx < out.len() {
        let abs_row = start_row + idx;
        let strip_idx = abs_row / 8;
        let lane = abs_row % 8;
        let strip = &data[strip_idx * strip_bytes..(strip_idx + 1) * strip_bytes];
        let vals = quant_kernels::dot_q4_k_repacked8(caps, strip, act);
        let take = (8 - lane).min(out.len() - idx);
        out[idx..idx + take].copy_from_slice(&vals[lane..lane + take]);
        idx += take;
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
    match weights.data {
        WeightSlice::Q8(data) => run_single_rows_quant(
            data,
            start_row,
            width / 32 * 34,
            act_q8_0,
            out,
            |rows, a| quant_kernels::dot_q8_0_4rows(caps, rows, a),
            |row, a| quant_kernels::dot_q8_0_row(caps, row, a),
        ),
        WeightSlice::Q4K(data) => run_single_rows_quant(
            data,
            start_row,
            width / 256 * 144,
            act_q8_k,
            out,
            |rows, a| quant_kernels::dot_q4_k_4rows(caps, rows, a),
            |row, a| quant_kernels::dot_q4_k_row(caps, row, a),
        ),
        WeightSlice::RepackedQ4K8(data) => {
            run_single_rows_repacked_q4_k(caps, data, width, start_row, act_q8_k, out);
        }
        WeightSlice::Q6K(data) => run_single_rows_quant(
            data,
            start_row,
            width / 256 * 210,
            act_q8_k,
            out,
            |rows, a| quant_kernels::dot_q6_k_4rows(caps, rows, a),
            |row, a| quant_kernels::dot_q6_k_row(caps, row, a),
        ),
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
    if !matches!(weights.data, WeightSlice::Float(_)) {
        let (caps, act_q8_0, act_q8_k) =
            quantize_batch_activations([weights], input, 1, weights.input, workers);
        let align = if matches!(weights.data, WeightSlice::RepackedQ4K8(_)) {
            8
        } else {
            4
        };
        row_workers::run_aligned(out, workers, align, &|start_row, slice| {
            single_rows_quant_arm64(caps, weights, start_row, &act_q8_0, &act_q8_k, slice);
        });
        return;
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
    let (caps, act_q8_0, act_q8_k) = quantize_batch_activations(
        targets.iter().map(|(w, _)| *w),
        input,
        1,
        input.len(),
        workers,
    );

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

    #[cfg(all(windows, feature = "rust-qualification"))]
    let row_align = if targets
        .iter()
        .any(|(w, _)| matches!(w.data, WeightSlice::RepackedQ4K8(_)))
    {
        8_usize
    } else {
        4_usize
    };
    #[cfg(not(all(windows, feature = "rust-qualification")))]
    let row_align = 4_usize;

    let mut offsets = [0_usize; 4];
    let mut padded_total = 0_usize;
    for (idx, (weights, _)) in targets.iter().enumerate() {
        offsets[idx] = padded_total;
        padded_total += weights.output.next_multiple_of(row_align);
    }
    scratch.resize(padded_total, 0.0);
    let matrices: [WeightMatrix<'_>; 4] = [
        targets[0].0,
        targets[1].0,
        targets.get(2).map_or(targets[0].0, |t| t.0),
        targets.get(3).map_or(targets[0].0, |t| t.0),
    ];
    let count = targets.len();
    row_workers::run_aligned(scratch, workers, row_align, &|chunk_start, slice| {
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

#[inline(always)]
fn batch_f32_4rows_quads<S: Simd, const Q: usize>(
    simd: S,
    rows: [&[f32]; 4],
    quads: &[[f32; 4]],
    num_quads: usize,
    q: usize,
    transposed_rows: &mut [f32],
) {
    let width = rows[0].len();
    let row_stride = num_quads * 4;
    let mut acc: [[f32x4<S>; 4]; Q] = std::array::from_fn(|_| [f32x4::splat(simd, 0.0_f32); 4]);
    for col in 0..width {
        let act: [f32x4<S>; Q] =
            std::array::from_fn(|qi| f32x4::from_slice(simd, &quads[(q + qi) * width + col]));
        for r in 0..4 {
            let w = f32x4::splat(simd, rows[r][col]);
            for qi in 0..Q {
                acc[qi][r] += w * act[qi];
            }
        }
    }
    for r in 0..4 {
        for qi in 0..Q {
            let base = r * row_stride + (q + qi) * 4;
            acc[qi][r].store_slice(&mut transposed_rows[base..base + 4]);
        }
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
    let mut q = 0;
    while q + 1 < num_quads {
        batch_f32_4rows_quads::<S, 2>(simd, rows, quads, num_quads, q, transposed_rows);
        q += 2;
    }
    if q < num_quads {
        batch_f32_4rows_quads::<S, 1>(simd, rows, quads, num_quads, q, transposed_rows);
    }
}

#[inline(always)]
fn batch_f32_1row_quads<S: Simd, const Q: usize>(
    simd: S,
    row: &[f32],
    quads: &[[f32; 4]],
    q: usize,
    transposed_row: &mut [f32],
) {
    let width = row.len();
    let mut acc = [f32x4::splat(simd, 0.0_f32); Q];
    for (col, &weight) in row.iter().enumerate() {
        let w = f32x4::splat(simd, weight);
        for qi in 0..Q {
            acc[qi] += w * f32x4::from_slice(simd, &quads[(q + qi) * width + col]);
        }
    }
    for qi in 0..Q {
        let base = (q + qi) * 4;
        acc[qi].store_slice(&mut transposed_row[base..base + 4]);
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
    let mut q = 0;
    while q + 1 < num_quads {
        batch_f32_1row_quads::<S, 2>(simd, row, quads, q, transposed_row);
        q += 2;
    }
    if q < num_quads {
        batch_f32_1row_quads::<S, 1>(simd, row, quads, q, transposed_row);
    }
}

#[inline(always)]
fn batch_q8_4rows_quads<S: Simd, const Q: usize>(
    simd: S,
    rows: [&[u8]; 4],
    width: usize,
    quads: &[[f32; 4]],
    num_quads: usize,
    q: usize,
    transposed_rows: &mut [f32],
) {
    let blocks = width / 32;
    let row_stride = num_quads * 4;
    let mut acc: [[f32x4<S>; 4]; Q] = std::array::from_fn(|_| [f32x4::splat(simd, 0.0_f32); 4]);
    for block_idx in 0..blocks {
        let b_off = block_idx * 34;
        let i_off = block_idx * 32;
        let s = rows.map(|row| f32x4::splat(simd, half_scale(&row[b_off..b_off + 2])));
        for part in 0..2 {
            let q_off = b_off + 2 + part * 16;
            let p_off = i_off + part * 16;
            let qs: [[f32x4<S>; 4]; 4] = std::array::from_fn(|r| {
                signed_lanes(simd, u8x16::from_slice(simd, &rows[r][q_off..q_off + 16]))
            });
            for group in 0..4 {
                let col_base = p_off + group * 4;
                let w: [[f32; 4]; 4] = std::array::from_fn(|r| (s[r] * qs[r][group]).into());
                for k in 0..4 {
                    let col = col_base + k;
                    let act: [f32x4<S>; Q] = std::array::from_fn(|qi| {
                        f32x4::from_slice(simd, &quads[(q + qi) * width + col])
                    });
                    for (r, wr) in w.iter().enumerate() {
                        let wv = f32x4::splat(simd, wr[k]);
                        for qi in 0..Q {
                            acc[qi][r] += wv * act[qi];
                        }
                    }
                }
            }
        }
    }
    for r in 0..4 {
        for qi in 0..Q {
            let base = r * row_stride + (q + qi) * 4;
            acc[qi][r].store_slice(&mut transposed_rows[base..base + 4]);
        }
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
    let mut q = 0;
    while q + 1 < num_quads {
        batch_q8_4rows_quads::<S, 2>(simd, rows, width, quads, num_quads, q, transposed_rows);
        q += 2;
    }
    if q < num_quads {
        batch_q8_4rows_quads::<S, 1>(simd, rows, width, quads, num_quads, q, transposed_rows);
    }
}

#[inline(always)]
fn batch_q8_1row_quads<S: Simd, const Q: usize>(
    simd: S,
    row: &[u8],
    width: usize,
    quads: &[[f32; 4]],
    q: usize,
    transposed_row: &mut [f32],
) {
    let blocks = width / 32;
    let mut acc = [f32x4::splat(simd, 0.0_f32); Q];
    for block_idx in 0..blocks {
        let b_off = block_idx * 34;
        let i_off = block_idx * 32;
        let scale = f32x4::splat(simd, half_scale(&row[b_off..b_off + 2]));
        let q_lo = signed_lanes(simd, u8x16::from_slice(simd, &row[b_off + 2..b_off + 18]));
        let q_hi = signed_lanes(simd, u8x16::from_slice(simd, &row[b_off + 18..b_off + 34]));
        for (group_idx, group_q) in q_lo.into_iter().chain(q_hi).enumerate() {
            let w: [f32; 4] = (scale * group_q).into();
            let col_base = i_off + group_idx * 4;
            for (k, &wk) in w.iter().enumerate() {
                let col = col_base + k;
                let wv = f32x4::splat(simd, wk);
                for qi in 0..Q {
                    acc[qi] += wv * f32x4::from_slice(simd, &quads[(q + qi) * width + col]);
                }
            }
        }
    }
    for qi in 0..Q {
        let base = (q + qi) * 4;
        acc[qi].store_slice(&mut transposed_row[base..base + 4]);
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
    let mut q = 0;
    while q + 1 < num_quads {
        batch_q8_1row_quads::<S, 2>(simd, row, width, quads, q, transposed_row);
        q += 2;
    }
    if q < num_quads {
        batch_q8_1row_quads::<S, 1>(simd, row, width, quads, q, transposed_row);
    }
}

#[inline(always)]
fn dot_q4_k_4rows_simd<S: Simd, const Q: usize>(
    simd: S,
    rows: [&[u8]; 4],
    width: usize,
    quads: &[[f32; 4]],
    num_quads: usize,
    q: usize,
    transposed_rows: &mut [f32],
) {
    let blocks = width / 256;
    let row_stride = num_quads * 4;
    let mask_0f = u8x16::splat(simd, 0x0f);
    let mut acc: [[f32x4<S>; 4]; Q] = std::array::from_fn(|_| [f32x4::splat(simd, 0.0_f32); 4]);
    for block_idx in 0..blocks {
        let b_off = block_idx * 144;
        let i_off = block_idx * 256;
        let d_dmin = rows.map(|row| {
            (
                half_scale(&row[b_off..b_off + 2]),
                half_scale(&row[b_off + 2..b_off + 4]),
                &row[b_off + 4..b_off + 16],
            )
        });
        for chunk_idx in 0..4 {
            let params = d_dmin.map(|(d, dmin, scales)| {
                let (s_lo, m_lo) = unpack_q4_k_scale_min(chunk_idx * 2, scales);
                let (s_hi, m_hi) = unpack_q4_k_scale_min(chunk_idx * 2 + 1, scales);
                (
                    (
                        f32x4::splat(simd, d * f32::from(s_lo)),
                        f32x4::splat(simd, dmin * f32::from(m_lo)),
                    ),
                    (
                        f32x4::splat(simd, d * f32::from(s_hi)),
                        f32x4::splat(simd, dmin * f32::from(m_hi)),
                    ),
                )
            });
            let qs_base = b_off + 16 + chunk_idx * 32;
            let in_base = i_off + chunk_idx * 64;
            for (half_idx, is_hi) in [false, true].into_iter().enumerate() {
                let p_off = in_base + half_idx * 32;
                for part in 0..2 {
                    let q_off = qs_base + part * 16;
                    let col_part = p_off + part * 16;
                    let weights: [[f32x4<S>; 4]; 4] = std::array::from_fn(|r| {
                        let raw = u8x16::from_slice(simd, &rows[r][q_off..q_off + 16]);
                        let nibs = if is_hi { raw >> 4 } else { raw & mask_0f };
                        let (sc, min) = if is_hi { params[r].1 } else { params[r].0 };
                        unsigned_lanes(simd, nibs).map(|qv| sc * qv - min)
                    });
                    for group in 0..4 {
                        let col_base = col_part + group * 4;
                        let w: [[f32; 4]; 4] = std::array::from_fn(|r| weights[r][group].into());
                        for k in 0..4 {
                            let col = col_base + k;
                            let act: [f32x4<S>; Q] = std::array::from_fn(|qi| {
                                f32x4::from_slice(simd, &quads[(q + qi) * width + col])
                            });
                            for (r, wr) in w.iter().enumerate() {
                                let wv = f32x4::splat(simd, wr[k]);
                                for qi in 0..Q {
                                    acc[qi][r] += wv * act[qi];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    for r in 0..4 {
        for qi in 0..Q {
            let base = r * row_stride + (q + qi) * 4;
            acc[qi][r].store_slice(&mut transposed_rows[base..base + 4]);
        }
    }
}

#[inline(always)]
fn batch_q4_k_4rows<S: Simd>(
    simd: S,
    rows: [&[u8]; 4],
    width: usize,
    quads: &[[f32; 4]],
    num_quads: usize,
    transposed_rows: &mut [f32],
) {
    let mut q = 0;
    while q + 1 < num_quads {
        dot_q4_k_4rows_simd::<S, 2>(simd, rows, width, quads, num_quads, q, transposed_rows);
        q += 2;
    }
    if q < num_quads {
        dot_q4_k_4rows_simd::<S, 1>(simd, rows, width, quads, num_quads, q, transposed_rows);
    }
}

#[inline(always)]
fn batch_q4_k_1row_quads<S: Simd, const Q: usize>(
    simd: S,
    row: &[u8],
    width: usize,
    quads: &[[f32; 4]],
    q: usize,
    transposed_row: &mut [f32],
) {
    let blocks = width / 256;
    let mask_0f = u8x16::splat(simd, 0x0f);
    let mut acc = [f32x4::splat(simd, 0.0_f32); Q];
    for block_idx in 0..blocks {
        let b_off = block_idx * 144;
        let i_off = block_idx * 256;
        let d = half_scale(&row[b_off..b_off + 2]);
        let dmin = half_scale(&row[b_off + 2..b_off + 4]);
        let scales = &row[b_off + 4..b_off + 16];
        for chunk_idx in 0..4 {
            let (s_lo, m_lo) = unpack_q4_k_scale_min(chunk_idx * 2, scales);
            let (s_hi, m_hi) = unpack_q4_k_scale_min(chunk_idx * 2 + 1, scales);
            let sc_lo = f32x4::splat(simd, d * f32::from(s_lo));
            let min_lo = f32x4::splat(simd, dmin * f32::from(m_lo));
            let sc_hi = f32x4::splat(simd, d * f32::from(s_hi));
            let min_hi = f32x4::splat(simd, dmin * f32::from(m_hi));
            let qs_base = b_off + 16 + chunk_idx * 32;
            let in_base = i_off + chunk_idx * 64;
            for (half_idx, is_hi) in [false, true].into_iter().enumerate() {
                let (sc, min) = if is_hi {
                    (sc_hi, min_hi)
                } else {
                    (sc_lo, min_lo)
                };
                let p_off = in_base + half_idx * 32;
                for part in 0..2 {
                    let q_off = qs_base + part * 16;
                    let col_part = p_off + part * 16;
                    let raw = u8x16::from_slice(simd, &row[q_off..q_off + 16]);
                    let nibs = if is_hi { raw >> 4 } else { raw & mask_0f };
                    for (group_idx, qv) in unsigned_lanes(simd, nibs).into_iter().enumerate() {
                        let w: [f32; 4] = (sc * qv - min).into();
                        let col_base = col_part + group_idx * 4;
                        for (k, &wk) in w.iter().enumerate() {
                            let col = col_base + k;
                            let wv = f32x4::splat(simd, wk);
                            for qi in 0..Q {
                                acc[qi] +=
                                    wv * f32x4::from_slice(simd, &quads[(q + qi) * width + col]);
                            }
                        }
                    }
                }
            }
        }
    }
    for qi in 0..Q {
        let base = (q + qi) * 4;
        acc[qi].store_slice(&mut transposed_row[base..base + 4]);
    }
}

#[inline(always)]
fn batch_q4_k_1row<S: Simd>(
    simd: S,
    row: &[u8],
    width: usize,
    quads: &[[f32; 4]],
    num_quads: usize,
    transposed_row: &mut [f32],
) {
    let mut q = 0;
    while q + 1 < num_quads {
        batch_q4_k_1row_quads::<S, 2>(simd, row, width, quads, q, transposed_row);
        q += 2;
    }
    if q < num_quads {
        batch_q4_k_1row_quads::<S, 1>(simd, row, width, quads, q, transposed_row);
    }
}

#[inline(always)]
fn dot_q6_k_4rows_simd<S: Simd, const Q: usize>(
    simd: S,
    rows: [&[u8]; 4],
    width: usize,
    quads: &[[f32; 4]],
    num_quads: usize,
    q: usize,
    transposed_rows: &mut [f32],
) {
    let blocks = width / 256;
    let row_stride = num_quads * 4;
    let mask_0f = u8x16::splat(simd, 0x0f);
    let mask_03 = u8x16::splat(simd, 0x03);
    let thirty_two = f32x4::splat(simd, 32.0_f32);
    let mut acc: [[f32x4<S>; 4]; Q] = std::array::from_fn(|_| [f32x4::splat(simd, 0.0_f32); 4]);
    for block_idx in 0..blocks {
        let b_off = block_idx * 210;
        let i_off = block_idx * 256;
        let d = rows.map(|row| half_scale(&row[b_off + 208..b_off + 210]));
        for n128 in 0..2 {
            let ql_base = b_off + n128 * 64;
            let qh_base = b_off + 128 + n128 * 32;
            let sc_base = b_off + 192 + n128 * 8;
            let in_base = i_off + n128 * 128;
            for part in 0..2 {
                let p_off = part * 16;
                let dequant_q = |r: usize, scale_idx: usize, quants: u8x16<S>| -> [f32x4<S>; 4] {
                    let sc = f32x4::splat(
                        simd,
                        d[r] * f32::from(rows[r][sc_base + scale_idx].cast_signed()),
                    );
                    unsigned_lanes(simd, quants).map(|qv| sc * (qv - thirty_two))
                };
                let weights: [[[f32x4<S>; 4]; 4]; 4] = std::array::from_fn(|r| {
                    let ql0 =
                        u8x16::from_slice(simd, &rows[r][ql_base + p_off..ql_base + p_off + 16]);
                    let ql1 = u8x16::from_slice(
                        simd,
                        &rows[r][ql_base + 32 + p_off..ql_base + 32 + p_off + 16],
                    );
                    let qh_v =
                        u8x16::from_slice(simd, &rows[r][qh_base + p_off..qh_base + p_off + 16]);
                    let q1 = (ql0 & mask_0f) | ((qh_v & mask_03) << 4);
                    let q2 = (ql1 & mask_0f) | (((qh_v >> 2) & mask_03) << 4);
                    let q3 = (ql0 >> 4) | (((qh_v >> 4) & mask_03) << 4);
                    let q4 = (ql1 >> 4) | (((qh_v >> 6) & mask_03) << 4);
                    [
                        dequant_q(r, part, q1),
                        dequant_q(r, part + 2, q2),
                        dequant_q(r, part + 4, q3),
                        dequant_q(r, part + 6, q4),
                    ]
                });
                for group in 0..4 {
                    let l_base = p_off + group * 4;
                    let w1: [[f32; 4]; 4] = std::array::from_fn(|r| weights[r][0][group].into());
                    let w2: [[f32; 4]; 4] = std::array::from_fn(|r| weights[r][1][group].into());
                    let w3: [[f32; 4]; 4] = std::array::from_fn(|r| weights[r][2][group].into());
                    let w4: [[f32; 4]; 4] = std::array::from_fn(|r| weights[r][3][group].into());
                    for k in 0..4 {
                        let col = in_base + l_base + k;
                        let a0: [f32x4<S>; Q] = std::array::from_fn(|qi| {
                            f32x4::from_slice(simd, &quads[(q + qi) * width + col])
                        });
                        let a1: [f32x4<S>; Q] = std::array::from_fn(|qi| {
                            f32x4::from_slice(simd, &quads[(q + qi) * width + col + 32])
                        });
                        let a2: [f32x4<S>; Q] = std::array::from_fn(|qi| {
                            f32x4::from_slice(simd, &quads[(q + qi) * width + col + 64])
                        });
                        let a3: [f32x4<S>; Q] = std::array::from_fn(|qi| {
                            f32x4::from_slice(simd, &quads[(q + qi) * width + col + 96])
                        });
                        for r in 0..4 {
                            let v1 = f32x4::splat(simd, w1[r][k]);
                            let v2 = f32x4::splat(simd, w2[r][k]);
                            let v3 = f32x4::splat(simd, w3[r][k]);
                            let v4 = f32x4::splat(simd, w4[r][k]);
                            for qi in 0..Q {
                                acc[qi][r] += v1 * a0[qi];
                                acc[qi][r] += v2 * a1[qi];
                                acc[qi][r] += v3 * a2[qi];
                                acc[qi][r] += v4 * a3[qi];
                            }
                        }
                    }
                }
            }
        }
    }
    for r in 0..4 {
        for qi in 0..Q {
            let base = r * row_stride + (q + qi) * 4;
            acc[qi][r].store_slice(&mut transposed_rows[base..base + 4]);
        }
    }
}

#[inline(always)]
fn batch_q6_k_4rows<S: Simd>(
    simd: S,
    rows: [&[u8]; 4],
    width: usize,
    quads: &[[f32; 4]],
    num_quads: usize,
    transposed_rows: &mut [f32],
) {
    let mut q = 0;
    while q + 1 < num_quads {
        dot_q6_k_4rows_simd::<S, 2>(simd, rows, width, quads, num_quads, q, transposed_rows);
        q += 2;
    }
    if q < num_quads {
        dot_q6_k_4rows_simd::<S, 1>(simd, rows, width, quads, num_quads, q, transposed_rows);
    }
}

#[inline(always)]
fn batch_q6_k_1row_quads<S: Simd, const Q: usize>(
    simd: S,
    row: &[u8],
    width: usize,
    quads: &[[f32; 4]],
    q: usize,
    transposed_row: &mut [f32],
) {
    let blocks = width / 256;
    let mask_0f = u8x16::splat(simd, 0x0f);
    let mask_03 = u8x16::splat(simd, 0x03);
    let thirty_two = f32x4::splat(simd, 32.0_f32);
    let mut acc = [f32x4::splat(simd, 0.0_f32); Q];
    for block_idx in 0..blocks {
        let b_off = block_idx * 210;
        let i_off = block_idx * 256;
        let d = half_scale(&row[b_off + 208..b_off + 210]);
        for n128 in 0..2 {
            let ql_base = b_off + n128 * 64;
            let qh_base = b_off + 128 + n128 * 32;
            let sc_base = b_off + 192 + n128 * 8;
            let in_base = i_off + n128 * 128;
            for part in 0..2 {
                let p_off = part * 16;
                let ql0 = u8x16::from_slice(simd, &row[ql_base + p_off..ql_base + p_off + 16]);
                let ql1 =
                    u8x16::from_slice(simd, &row[ql_base + 32 + p_off..ql_base + 32 + p_off + 16]);
                let qh_v = u8x16::from_slice(simd, &row[qh_base + p_off..qh_base + p_off + 16]);
                let dequant = |scale_idx: usize, quants: u8x16<S>| -> [f32x4<S>; 4] {
                    let sc =
                        f32x4::splat(simd, d * f32::from(row[sc_base + scale_idx].cast_signed()));
                    unsigned_lanes(simd, quants).map(|qv| sc * (qv - thirty_two))
                };
                let w1 = dequant(part, (ql0 & mask_0f) | ((qh_v & mask_03) << 4));
                let w2 = dequant(part + 2, (ql1 & mask_0f) | (((qh_v >> 2) & mask_03) << 4));
                let w3 = dequant(part + 4, (ql0 >> 4) | (((qh_v >> 4) & mask_03) << 4));
                let w4 = dequant(part + 6, (ql1 >> 4) | (((qh_v >> 6) & mask_03) << 4));
                for group in 0..4 {
                    let l_base = p_off + group * 4;
                    let g1: [f32; 4] = w1[group].into();
                    let g2: [f32; 4] = w2[group].into();
                    let g3: [f32; 4] = w3[group].into();
                    let g4: [f32; 4] = w4[group].into();
                    for k in 0..4 {
                        let col = in_base + l_base + k;
                        let a0: [f32x4<S>; Q] = std::array::from_fn(|qi| {
                            f32x4::from_slice(simd, &quads[(q + qi) * width + col])
                        });
                        let a1: [f32x4<S>; Q] = std::array::from_fn(|qi| {
                            f32x4::from_slice(simd, &quads[(q + qi) * width + col + 32])
                        });
                        let a2: [f32x4<S>; Q] = std::array::from_fn(|qi| {
                            f32x4::from_slice(simd, &quads[(q + qi) * width + col + 64])
                        });
                        let a3: [f32x4<S>; Q] = std::array::from_fn(|qi| {
                            f32x4::from_slice(simd, &quads[(q + qi) * width + col + 96])
                        });
                        let v1 = f32x4::splat(simd, g1[k]);
                        let v2 = f32x4::splat(simd, g2[k]);
                        let v3 = f32x4::splat(simd, g3[k]);
                        let v4 = f32x4::splat(simd, g4[k]);
                        for qi in 0..Q {
                            acc[qi] += v1 * a0[qi];
                            acc[qi] += v2 * a1[qi];
                            acc[qi] += v3 * a2[qi];
                            acc[qi] += v4 * a3[qi];
                        }
                    }
                }
            }
        }
    }
    for qi in 0..Q {
        let base = (q + qi) * 4;
        acc[qi].store_slice(&mut transposed_row[base..base + 4]);
    }
}

#[inline(always)]
fn batch_q6_k_1row<S: Simd>(
    simd: S,
    row: &[u8],
    width: usize,
    quads: &[[f32; 4]],
    num_quads: usize,
    transposed_row: &mut [f32],
) {
    let mut q = 0;
    while q + 1 < num_quads {
        batch_q6_k_1row_quads::<S, 2>(simd, row, width, quads, q, transposed_row);
        q += 2;
    }
    if q < num_quads {
        batch_q6_k_1row_quads::<S, 1>(simd, row, width, quads, q, transposed_row);
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
                batch_f32_4rows(
                    simd,
                    four_rows(data, start_row + processed, width),
                    quads,
                    num_quads,
                    chunk,
                );
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
                batch_q8_4rows(
                    simd,
                    four_rows(data, start_row + processed, row_bytes),
                    width,
                    quads,
                    num_quads,
                    chunk,
                );
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
            for chunk in transposed_slice.chunks_exact_mut(4 * row_stride) {
                batch_q4_k_4rows(
                    simd,
                    four_rows(data, start_row + processed, row_bytes),
                    width,
                    quads,
                    num_quads,
                    chunk,
                );
                processed += 4;
            }
            for (local_idx, row_out) in transposed_slice[processed * row_stride..]
                .chunks_exact_mut(row_stride)
                .enumerate()
            {
                let off = (start_row + processed + local_idx) * row_bytes;
                batch_q4_k_1row(
                    simd,
                    &data[off..off + row_bytes],
                    width,
                    quads,
                    num_quads,
                    row_out,
                );
            }
        }
        #[cfg(all(windows, feature = "rust-qualification"))]
        WeightSlice::RepackedQ4K8(_) => {}
        WeightSlice::Q6K(data) => {
            let row_bytes = width / 256 * 210;
            for chunk in transposed_slice.chunks_exact_mut(4 * row_stride) {
                batch_q6_k_4rows(
                    simd,
                    four_rows(data, start_row + processed, row_bytes),
                    width,
                    quads,
                    num_quads,
                    chunk,
                );
                processed += 4;
            }
            for (local_idx, row_out) in transposed_slice[processed * row_stride..]
                .chunks_exact_mut(row_stride)
                .enumerate()
            {
                let off = (start_row + processed + local_idx) * row_bytes;
                batch_q6_k_1row(
                    simd,
                    &data[off..off + row_bytes],
                    width,
                    quads,
                    num_quads,
                    row_out,
                );
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
fn quantize_batch_activations<'a>(
    weights: impl IntoIterator<Item = WeightMatrix<'a>>,
    raw_input: &[f32],
    batch: usize,
    width: usize,
    workers: usize,
) -> (Arm64QuantCaps, Vec<Q8_0ActBlock>, Vec<Q8_KActBlock>) {
    const PARALLEL_QUANT_WORK: usize = 8_192;
    let caps = Arm64QuantCaps::detect();
    let mut needs_q8_0 = false;
    let mut needs_q8_k = false;
    for w in weights {
        match w.data {
            WeightSlice::Q8(_) => needs_q8_0 = true,
            WeightSlice::Q4K(_) | WeightSlice::RepackedQ4K8(_) | WeightSlice::Q6K(_) => {
                needs_q8_k = true;
            }
            WeightSlice::Float(_) => {}
        }
    }
    let mut act_q8_0 = Vec::new();
    let mut act_q8_k = Vec::new();
    let q8_0_blocks = width / 32;
    let q8_k_blocks = width / 256;
    if needs_q8_0 {
        act_q8_0.resize(batch * q8_0_blocks, Q8_0ActBlock::default());
    }
    if needs_q8_k {
        act_q8_k.resize(batch * q8_k_blocks, Q8_KActBlock::default());
    }

    let run_q8_0 = needs_q8_0 && q8_0_blocks > 0;
    let run_q8_k = needs_q8_k && q8_k_blocks > 0;
    let active_workers = workers.clamp(1, crate::threads::MAX_WORKERS).min(batch);
    if active_workers > 1
        && batch.saturating_mul(width) >= PARALLEL_QUANT_WORK
        && (run_q8_0 || run_q8_k)
    {
        let chunk_frames = batch.div_ceil(active_workers);
        let num_tasks = batch.div_ceil(chunk_frames);
        let q8_0_slots: Vec<std::sync::Mutex<&mut [Q8_0ActBlock]>> = if run_q8_0 {
            act_q8_0
                .chunks_mut(chunk_frames * q8_0_blocks)
                .map(std::sync::Mutex::new)
                .collect()
        } else {
            Vec::new()
        };
        let q8_k_slots: Vec<std::sync::Mutex<&mut [Q8_KActBlock]>> = if run_q8_k {
            act_q8_k
                .chunks_mut(chunk_frames * q8_k_blocks)
                .map(std::sync::Mutex::new)
                .collect()
        } else {
            Vec::new()
        };
        let mut task_tokens = [0.0_f32; crate::threads::MAX_WORKERS];
        row_workers::run_aligned(
            &mut task_tokens[..num_tasks],
            active_workers,
            1,
            &|start_task, slice| {
                for task_idx in start_task..(start_task + slice.len()) {
                    let start_t = task_idx * chunk_frames;
                    let count = (batch - start_t).min(chunk_frames);
                    if let Some(slot) = q8_0_slots.get(task_idx) {
                        let mut out_chunk = slot
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                        for local_t in 0..count {
                            let t = start_t + local_t;
                            quant_kernels::quantize_row_q8_0(
                                &raw_input[t * width..(t + 1) * width],
                                &mut out_chunk[local_t * q8_0_blocks..(local_t + 1) * q8_0_blocks],
                            );
                        }
                    }
                    if let Some(slot) = q8_k_slots.get(task_idx) {
                        let mut out_chunk = slot
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                        for local_t in 0..count {
                            let t = start_t + local_t;
                            quant_kernels::quantize_row_q8_k(
                                &raw_input[t * width..(t + 1) * width],
                                &mut out_chunk[local_t * q8_k_blocks..(local_t + 1) * q8_k_blocks],
                            );
                        }
                    }
                }
            },
        );
    } else {
        if run_q8_0 {
            for t in 0..batch {
                quant_kernels::quantize_row_q8_0(
                    &raw_input[t * width..(t + 1) * width],
                    &mut act_q8_0[t * q8_0_blocks..(t + 1) * q8_0_blocks],
                );
            }
        }
        if run_q8_k {
            for t in 0..batch {
                quant_kernels::quantize_row_q8_k(
                    &raw_input[t * width..(t + 1) * width],
                    &mut act_q8_k[t * q8_k_blocks..(t + 1) * q8_k_blocks],
                );
            }
        }
    }
    (caps, act_q8_0, act_q8_k)
}

#[cfg(all(windows, feature = "rust-qualification"))]
#[expect(
    clippy::too_many_arguments,
    reason = "Parameterizes Q8_0/Q4_K/Q6_K batched row quantization kernels without extra state structs."
)]
fn run_batch_rows_quant<A>(
    data: &[u8],
    start_row: usize,
    row_bytes: usize,
    blocks: usize,
    batch: usize,
    act: &[A],
    transposed_slice: &mut [f32],
    gemm4x2: impl Fn([&[u8]; 4], &[A], &[A]) -> [[f32; 2]; 4],
    dot4: impl Fn([&[u8]; 4], &[A]) -> [f32; 4],
    dot1: impl Fn(&[u8], &[A]) -> f32,
) {
    let mut processed = 0_usize;
    for chunk in transposed_slice.chunks_exact_mut(4 * batch) {
        let row_slices = four_rows(data, start_row + processed, row_bytes);
        let mut t = 0_usize;
        while t + 1 < batch {
            let res = gemm4x2(
                row_slices,
                &act[t * blocks..(t + 1) * blocks],
                &act[(t + 1) * blocks..(t + 2) * blocks],
            );
            for (r, pair) in res.iter().enumerate() {
                chunk[r * batch + t] = pair[0];
                chunk[r * batch + t + 1] = pair[1];
            }
            t += 2;
        }
        if t < batch {
            let res = dot4(row_slices, &act[t * blocks..(t + 1) * blocks]);
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
            *dst = dot1(w_row, &act[t * blocks..(t + 1) * blocks]);
        }
    }
}

#[cfg(all(windows, feature = "rust-qualification"))]
fn run_batch_rows_repacked_q4_k(
    caps: Arm64QuantCaps,
    data: &[u8],
    width: usize,
    start_row: usize,
    batch: usize,
    act: &[Q8_KActBlock],
    transposed_slice: &mut [f32],
) {
    let blocks = width / quant_kernels::QK_K;
    let strip_bytes = blocks * quant_kernels::Q4_K_REPACK8_BLOCK_BYTES;
    let rows_here = transposed_slice.len() / batch;
    let mut row_idx = 0_usize;
    while row_idx < rows_here {
        let abs_row = start_row + row_idx;
        let strip_idx = abs_row / 8;
        let lane = abs_row % 8;
        let strip = &data[strip_idx * strip_bytes..(strip_idx + 1) * strip_bytes];
        let take = (8 - lane).min(rows_here - row_idx);
        let mut t = 0_usize;
        while t + 1 < batch {
            let tile = quant_kernels::gemm_q4_k_repacked8_8x2(
                caps,
                strip,
                &act[t * blocks..(t + 1) * blocks],
                &act[(t + 1) * blocks..(t + 2) * blocks],
            );
            for local_r in 0..take {
                let base = (row_idx + local_r) * batch + t;
                transposed_slice[base] = tile[lane + local_r][0];
                transposed_slice[base + 1] = tile[lane + local_r][1];
            }
            t += 2;
        }
        if t < batch {
            let col =
                quant_kernels::dot_q4_k_repacked8(caps, strip, &act[t * blocks..(t + 1) * blocks]);
            for local_r in 0..take {
                transposed_slice[(row_idx + local_r) * batch + t] = col[lane + local_r];
            }
        }
        row_idx += take;
    }
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
    match weights.data {
        WeightSlice::Q8(data) => {
            let blocks = width / 32;
            run_batch_rows_quant(
                data,
                start_row,
                blocks * 34,
                blocks,
                batch,
                act_q8_0,
                transposed_slice,
                |r, a, b| quant_kernels::gemm_q8_0_4x2(caps, r, a, b),
                |r, a| quant_kernels::dot_q8_0_4rows(caps, r, a),
                |r, a| quant_kernels::dot_q8_0_row(caps, r, a),
            );
        }
        WeightSlice::Q4K(data) => {
            let blocks = width / 256;
            run_batch_rows_quant(
                data,
                start_row,
                blocks * 144,
                blocks,
                batch,
                act_q8_k,
                transposed_slice,
                |r, a, b| quant_kernels::gemm_q4_k_4x2(caps, r, a, b),
                |r, a| quant_kernels::dot_q4_k_4rows(caps, r, a),
                |r, a| quant_kernels::dot_q4_k_row(caps, r, a),
            );
        }
        WeightSlice::RepackedQ4K8(data) => {
            run_batch_rows_repacked_q4_k(
                caps,
                data,
                width,
                start_row,
                batch,
                act_q8_k,
                transposed_slice,
            );
        }
        WeightSlice::Q6K(data) => {
            let blocks = width / 256;
            run_batch_rows_quant(
                data,
                start_row,
                blocks * 210,
                blocks,
                batch,
                act_q8_k,
                transposed_slice,
                |r, a, b| quant_kernels::gemm_q6_k_4x2(caps, r, a, b),
                |r, a| quant_kernels::dot_q6_k_4rows(caps, r, a),
                |r, a| quant_kernels::dot_q6_k_row(caps, r, a),
            );
        }
        WeightSlice::Float(_) => {}
    }
}

fn scatter_transposed_to_batch(
    transposed: &[f32],
    row_stride: usize,
    output: usize,
    batch: usize,
    workers: usize,
    out: &mut [f32],
) {
    const PARALLEL_SCATTER_WORK: usize = 8_192;
    if batch == 0 || output == 0 {
        return;
    }
    let level = simd_level();
    let full_quads = batch / 4;
    if full_quads > 0 {
        let quad_out = &mut out[..full_quads * 4 * output];
        if workers > 1 && full_quads >= 2 && quad_out.len() >= PARALLEL_SCATTER_WORK {
            row_workers::run_aligned(quad_out, workers, 4 * output, &|elem_off, slice| {
                dispatch!(
                    level,
                    simd => scatter_quads_simd(simd, transposed, row_stride, output, elem_off, slice)
                );
            });
        } else {
            dispatch!(
                level,
                simd => scatter_quads_simd(simd, transposed, row_stride, output, 0, quad_out)
            );
        }
    }
    let rem = batch % 4;
    if rem > 0 {
        let tail_quad = full_quads * 4;
        let tail_out = &mut out[tail_quad * output..batch * output];
        for pos in 0..rem {
            let dst_row = &mut tail_out[pos * output..(pos + 1) * output];
            for (r, dst) in dst_row.iter_mut().enumerate() {
                *dst = transposed[r * row_stride + tail_quad + pos];
            }
        }
    }
}

#[cfg(any(test, all(windows, feature = "rust-qualification")))]
fn scatter_rows_to_batch(
    transposed: &[f32],
    output: usize,
    batch: usize,
    workers: usize,
    out: &mut [f32],
) {
    scatter_transposed_to_batch(transposed, batch, output, batch, workers, out);
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
            quantize_batch_activations([weights], raw_input, batch, weights.input, workers);
        let needed = output * batch;
        if transposed_scratch.len() < needed {
            transposed_scratch.resize(needed, 0.0);
        }
        let transposed = &mut transposed_scratch[..needed];
        let row_align = if matches!(weights.data, WeightSlice::RepackedQ4K8(_)) {
            8
        } else {
            4
        };
        row_workers::run_aligned(
            transposed,
            workers,
            row_align * batch,
            &|elem_off, slice| {
                let start_row = elem_off / batch;
                batch_rows_quant_arm64(
                    caps, weights, start_row, batch, &act_q8_0, &act_q8_k, slice,
                );
            },
        );
        scatter_rows_to_batch(transposed, output, batch, workers, out);
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
        scatter_transposed_to_batch(transposed, row_stride, output, prepared.batch, workers, out);
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
        let all_quant = targets
            .iter()
            .all(|(w, _)| !matches!(w.data, WeightSlice::Float(_)));
        if all_quant {
            let count = targets.len();
            let matrices = [
                targets[0].0,
                targets[1].0,
                targets.get(2).map_or(targets[0].0, |t| t.0),
                targets.get(3).map_or(targets[0].0, |t| t.0),
            ];
            let (caps, act_q8_0, act_q8_k) = quantize_batch_activations(
                matrices[..count].iter().copied(),
                raw_input,
                batch,
                prepared.input,
                workers,
            );
            let row_align = if matrices[..count]
                .iter()
                .any(|w| matches!(w.data, WeightSlice::RepackedQ4K8(_)))
            {
                8_usize
            } else {
                4_usize
            };
            let mut offsets = [0_usize; 4];
            let mut padded_rows = 0_usize;
            for (idx, w) in matrices[..count].iter().enumerate() {
                offsets[idx] = padded_rows;
                padded_rows += w.output.next_multiple_of(row_align);
            }
            let needed = padded_rows * batch;
            if transposed_scratch.len() < needed {
                transposed_scratch.resize(needed, 0.0);
            }
            let transposed = &mut transposed_scratch[..needed];
            row_workers::run_aligned(
                transposed,
                workers,
                row_align * batch,
                &|elem_off, slice| {
                    let start_row_global = elem_off / batch;
                    let end_row_global = start_row_global + slice.len() / batch;
                    for (idx, w) in matrices[..count].iter().enumerate() {
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
                },
            );
            for (idx, (w, out)) in targets.iter_mut().enumerate() {
                let elem_off = offsets[idx] * batch;
                scatter_rows_to_batch(
                    &transposed[elem_off..elem_off + w.output * batch],
                    w.output,
                    batch,
                    workers,
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

trait KvSource: Copy {
    type Scratch;
    fn new_scratch() -> Self::Scratch;
    fn validate(
        head_dim: usize,
        q_len: usize,
        out_len: usize,
        invalid: fn() -> std::io::Error,
    ) -> std::io::Result<()>;
    fn with_quad_rows(
        self,
        base_row: usize,
        row_width: usize,
        offset: usize,
        head_dim: usize,
        buf: &mut Self::Scratch,
        f: impl FnOnce([&[f32]; 4]),
    );
    fn with_row(
        self,
        row: usize,
        row_width: usize,
        offset: usize,
        head_dim: usize,
        buf: &mut Self::Scratch,
        f: impl FnOnce(&[f32]),
    );
}

impl KvSource for &[f32] {
    type Scratch = ();

    fn new_scratch() -> Self::Scratch {}

    fn validate(
        _head_dim: usize,
        _q_len: usize,
        _out_len: usize,
        _invalid: fn() -> std::io::Error,
    ) -> std::io::Result<()> {
        Ok(())
    }

    fn with_quad_rows(
        self,
        base_row: usize,
        row_width: usize,
        offset: usize,
        head_dim: usize,
        (): &mut Self::Scratch,
        f: impl FnOnce([&[f32]; 4]),
    ) {
        let row = |r: usize| {
            let start = (base_row + r) * row_width + offset;
            &self[start..start + head_dim]
        };
        f([row(0), row(1), row(2), row(3)]);
    }

    fn with_row(
        self,
        row: usize,
        row_width: usize,
        offset: usize,
        head_dim: usize,
        (): &mut Self::Scratch,
        f: impl FnOnce(&[f32]),
    ) {
        let start = row * row_width + offset;
        f(&self[start..start + head_dim]);
    }
}

impl KvSource for &[u16] {
    type Scratch = [[f32; 256]; 4];

    fn new_scratch() -> Self::Scratch {
        [[0.0_f32; 256]; 4]
    }

    fn validate(
        head_dim: usize,
        q_len: usize,
        out_len: usize,
        invalid: fn() -> std::io::Error,
    ) -> std::io::Result<()> {
        if head_dim > 256 || q_len != head_dim || out_len != head_dim {
            return Err(invalid());
        }
        Ok(())
    }

    fn with_quad_rows(
        self,
        base_row: usize,
        row_width: usize,
        offset: usize,
        head_dim: usize,
        buf: &mut Self::Scratch,
        f: impl FnOnce([&[f32]; 4]),
    ) {
        for (row_off, dst) in buf.iter_mut().enumerate() {
            let start = (base_row + row_off) * row_width + offset;
            widen_f16_slice(&self[start..start + head_dim], &mut dst[..head_dim]);
        }
        f([
            &buf[0][..head_dim],
            &buf[1][..head_dim],
            &buf[2][..head_dim],
            &buf[3][..head_dim],
        ]);
    }

    fn with_row(
        self,
        row: usize,
        row_width: usize,
        offset: usize,
        head_dim: usize,
        buf: &mut Self::Scratch,
        f: impl FnOnce(&[f32]),
    ) {
        let start = row * row_width + offset;
        widen_f16_slice(&self[start..start + head_dim], &mut buf[0][..head_dim]);
        f(&buf[0][..head_dim]);
    }
}

fn dot_row(q_head: &[f32], k: &[f32], divisor: f32) -> f32 {
    let mut dot = 0.0_f32;
    for (q_elem, k_elem) in q_head.iter().zip(k) {
        dot += *q_elem * *k_elem;
    }
    dot / divisor
}

#[expect(
    clippy::too_many_arguments,
    reason = "Shared single-head attention kernel reuses caller score/output storage without allocating."
)]
fn attend_head_impl<T: KvSource>(
    q_head: &[f32],
    keys: T,
    values: T,
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
    T::validate(head_dim, q_head.len(), out_head.len(), invalid)?;
    let mut buf = T::new_scratch();
    let full_quads = history_rows / 4;
    for quad in 0..full_quads {
        let r = quad * 4;
        keys.with_quad_rows(r, row_width, offset, head_dim, &mut buf, |k_refs| {
            scores[r..r + 4].copy_from_slice(&dot_4keys_seq(q_head, k_refs, divisor));
        });
    }
    for (row, score) in scores
        .iter_mut()
        .enumerate()
        .take(history_rows)
        .skip(full_quads * 4)
    {
        keys.with_row(row, row_width, offset, head_dim, &mut buf, |k| {
            *score = dot_row(q_head, k, divisor);
        });
    }
    if let Some((k, _)) = current_kv {
        scores[history_rows] = dot_row(q_head, k, divisor);
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
        let weights = [
            scores[r] / sum,
            scores[r + 1] / sum,
            scores[r + 2] / sum,
            scores[r + 3] / sum,
        ];
        values.with_quad_rows(r, row_width, offset, head_dim, &mut buf, |v_refs| {
            weighted_value_accum_4(out_head, v_refs, weights);
        });
    }
    for (row, &score) in scores
        .iter()
        .enumerate()
        .take(history_rows)
        .skip(full_quads * 4)
    {
        values.with_row(row, row_width, offset, head_dim, &mut buf, |v| {
            weighted_value_accum(out_head, v, score / sum);
        });
    }
    if let Some((_, v)) = current_kv {
        weighted_value_accum(out_head, v, scores[history_rows] / sum);
    }
    Ok(())
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
    attend_head_impl(
        q_head,
        keys,
        values,
        history_rows,
        row_width,
        offset,
        head_dim,
        divisor,
        current_kv,
        scores,
        out_head,
        invalid,
    )
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

#[cfg(test)]
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
    attend_head_impl(
        q_head,
        keys,
        values,
        history_rows,
        row_width,
        offset,
        head_dim,
        divisor,
        current_kv,
        scores,
        out_head,
        invalid,
    )
}

#[inline(always)]
fn dot_4keys_tile_simd<S: Simd, const N: usize>(
    simd: S,
    q_tile: [&[f32]; N],
    keys: [&[f32]; 4],
    head_dim: usize,
    divisor: f32,
) -> [[f32; 4]; N] {
    let mut acc = [f32x4::splat(simd, 0.0_f32); N];
    let chunks = head_dim / 4;
    for chunk_idx in 0..chunks {
        let base = chunk_idx * 4;
        let k0 = f32x4::from_slice(simd, &keys[0][base..base + 4]);
        let k1 = f32x4::from_slice(simd, &keys[1][base..base + 4]);
        let k2 = f32x4::from_slice(simd, &keys[2][base..base + 4]);
        let k3 = f32x4::from_slice(simd, &keys[3][base..base + 4]);
        let cols = transpose_4x4(simd, [k0, k1, k2, k3]);
        for k in 0..4 {
            let col = cols[k];
            for g in 0..N {
                acc[g] += f32x4::splat(simd, q_tile[g][base + k]) * col;
            }
        }
    }
    for d in (chunks * 4)..head_dim {
        let kv = f32x4::from_slice(simd, &[keys[0][d], keys[1][d], keys[2][d], keys[3][d]]);
        for g in 0..N {
            acc[g] += f32x4::splat(simd, q_tile[g][d]) * kv;
        }
    }
    let div_v = f32x4::splat(simd, divisor);
    let mut out = [[0.0_f32; 4]; N];
    for g in 0..N {
        out[g] = (acc[g] / div_v).into();
    }
    out
}

#[inline(always)]
fn weighted_value_accum_4_tile_simd<S: Simd, const N: usize>(
    simd: S,
    out_tile: [&mut [f32]; N],
    values: [&[f32]; 4],
    weights: [[f32; 4]; N],
    head_dim: usize,
) {
    let w0: [f32x4<S>; N] = std::array::from_fn(|g| f32x4::splat(simd, weights[g][0]));
    let w1: [f32x4<S>; N] = std::array::from_fn(|g| f32x4::splat(simd, weights[g][1]));
    let w2: [f32x4<S>; N] = std::array::from_fn(|g| f32x4::splat(simd, weights[g][2]));
    let w3: [f32x4<S>; N] = std::array::from_fn(|g| f32x4::splat(simd, weights[g][3]));
    let chunks = head_dim / 4;
    for chunk_idx in 0..chunks {
        let base = chunk_idx * 4;
        let v0 = f32x4::from_slice(simd, &values[0][base..base + 4]);
        let v1 = f32x4::from_slice(simd, &values[1][base..base + 4]);
        let v2 = f32x4::from_slice(simd, &values[2][base..base + 4]);
        let v3 = f32x4::from_slice(simd, &values[3][base..base + 4]);
        for g in 0..N {
            let dst = &mut out_tile[g][base..base + 4];
            let mut ov = f32x4::from_slice(simd, dst);
            ov += w0[g] * v0;
            ov += w1[g] * v1;
            ov += w2[g] * v2;
            ov += w3[g] * v3;
            ov.store_slice(dst);
        }
    }
    for d in (chunks * 4)..head_dim {
        for g in 0..N {
            out_tile[g][d] += weights[g][0] * values[0][d];
            out_tile[g][d] += weights[g][1] * values[1][d];
            out_tile[g][d] += weights[g][2] * values[2][d];
            out_tile[g][d] += weights[g][3] * values[3][d];
        }
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "Applies one shared K quad across all query heads in a GQA group in exact scalar order."
)]
fn dot_4keys_group_simd<S: Simd>(
    simd: S,
    q_group: &[f32],
    k_refs: [&[f32]; 4],
    head_dim: usize,
    gqa_group: usize,
    total_rows: usize,
    row_base: usize,
    divisor: f32,
    scores: &mut [f32],
) {
    let mut g_base = 0;
    while g_base < gqa_group {
        let rem = gqa_group - g_base;
        let q_at = |local_g: usize| -> &[f32] {
            let start = (g_base + local_g) * head_dim;
            &q_group[start..start + head_dim]
        };
        let mut store = |local_g: usize, vals: [f32; 4]| {
            let start = (g_base + local_g) * total_rows + row_base;
            scores[start..start + 4].copy_from_slice(&vals);
        };
        if rem >= 4 {
            let res = dot_4keys_tile_simd(
                simd,
                [q_at(0), q_at(1), q_at(2), q_at(3)],
                k_refs,
                head_dim,
                divisor,
            );
            for (local_g, vals) in res.into_iter().enumerate() {
                store(local_g, vals);
            }
            g_base += 4;
        } else if rem == 3 {
            let res =
                dot_4keys_tile_simd(simd, [q_at(0), q_at(1), q_at(2)], k_refs, head_dim, divisor);
            for (local_g, vals) in res.into_iter().enumerate() {
                store(local_g, vals);
            }
            g_base += 3;
        } else if rem == 2 {
            let res = dot_4keys_tile_simd(simd, [q_at(0), q_at(1)], k_refs, head_dim, divisor);
            for (local_g, vals) in res.into_iter().enumerate() {
                store(local_g, vals);
            }
            g_base += 2;
        } else {
            let res = dot_4keys_tile_simd(simd, [q_at(0)], k_refs, head_dim, divisor);
            store(0, res[0]);
            g_base += 1;
        }
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "Accumulates one shared V quad across all query heads in a GQA group in exact scalar order."
)]
fn weighted_value_accum_4_group_simd<S: Simd>(
    simd: S,
    out_group: &mut [f32],
    v_refs: [&[f32]; 4],
    scores: &[f32],
    sums: &[f32],
    head_dim: usize,
    gqa_group: usize,
    total_rows: usize,
    row_base: usize,
) {
    let mut g_base = 0;
    while g_base < gqa_group {
        let rem = gqa_group - g_base;
        let weights_at = |local_g: usize| -> [f32; 4] {
            let g = g_base + local_g;
            let sum = sums[g];
            let s = &scores[g * total_rows + row_base..g * total_rows + row_base + 4];
            [s[0] / sum, s[1] / sum, s[2] / sum, s[3] / sum]
        };
        let tile_out = &mut out_group[g_base * head_dim..];
        if rem >= 4 {
            let (o0, r1) = tile_out.split_at_mut(head_dim);
            let (o1, r2) = r1.split_at_mut(head_dim);
            let (o2, r3) = r2.split_at_mut(head_dim);
            let (o3, _) = r3.split_at_mut(head_dim);
            weighted_value_accum_4_tile_simd(
                simd,
                [o0, o1, o2, o3],
                v_refs,
                [weights_at(0), weights_at(1), weights_at(2), weights_at(3)],
                head_dim,
            );
            g_base += 4;
        } else if rem == 3 {
            let (o0, r1) = tile_out.split_at_mut(head_dim);
            let (o1, r2) = r1.split_at_mut(head_dim);
            let (o2, _) = r2.split_at_mut(head_dim);
            weighted_value_accum_4_tile_simd(
                simd,
                [o0, o1, o2],
                v_refs,
                [weights_at(0), weights_at(1), weights_at(2)],
                head_dim,
            );
            g_base += 3;
        } else if rem == 2 {
            let (o0, r1) = tile_out.split_at_mut(head_dim);
            let (o1, _) = r1.split_at_mut(head_dim);
            weighted_value_accum_4_tile_simd(
                simd,
                [o0, o1],
                v_refs,
                [weights_at(0), weights_at(1)],
                head_dim,
            );
            g_base += 2;
        } else {
            let (o0, _) = tile_out.split_at_mut(head_dim);
            weighted_value_accum_4_tile_simd(simd, [o0], v_refs, [weights_at(0)], head_dim);
            g_base += 1;
        }
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "Shared GQA KV-group attention kernel reuses caller score/output storage without allocating."
)]
fn attend_kv_group_impl<T: KvSource>(
    q_group: &[f32],
    keys: T,
    values: T,
    history_rows: usize,
    row_width: usize,
    offset: usize,
    head_dim: usize,
    gqa_group: usize,
    divisor: f32,
    current_kv: Option<(&[f32], &[f32])>,
    scores: &mut [f32],
    out_group: &mut [f32],
    invalid: fn() -> std::io::Error,
) -> std::io::Result<()> {
    let total_rows = history_rows
        .checked_add(usize::from(current_kv.is_some()))
        .ok_or_else(invalid)?;
    let group_elems = gqa_group.checked_mul(head_dim).ok_or_else(invalid)?;
    let scores_elems = gqa_group.checked_mul(total_rows).ok_or_else(invalid)?;
    if gqa_group == 0
        || head_dim == 0
        || q_group.len() != group_elems
        || out_group.len() != group_elems
        || scores.len() != scores_elems
        || current_kv.is_some_and(|(k, v)| k.len() != head_dim || v.len() != head_dim)
    {
        return Err(invalid());
    }
    if gqa_group == 1 {
        return attend_head_impl(
            q_group,
            keys,
            values,
            history_rows,
            row_width,
            offset,
            head_dim,
            divisor,
            current_kv,
            scores,
            out_group,
            invalid,
        );
    }
    T::validate(head_dim, head_dim, head_dim, invalid)?;
    let mut buf = T::new_scratch();
    let full_quads = history_rows / 4;
    let level = simd_level();
    for quad in 0..full_quads {
        let r = quad * 4;
        keys.with_quad_rows(r, row_width, offset, head_dim, &mut buf, |k_refs| {
            dispatch!(level, simd => {
                dot_4keys_group_simd(
                    simd,
                    q_group,
                    k_refs,
                    head_dim,
                    gqa_group,
                    total_rows,
                    r,
                    divisor,
                    scores,
                );
            });
        });
    }
    for row in (full_quads * 4)..history_rows {
        keys.with_row(row, row_width, offset, head_dim, &mut buf, |k| {
            for g in 0..gqa_group {
                let q_head = &q_group[g * head_dim..(g + 1) * head_dim];
                scores[g * total_rows + row] = dot_row(q_head, k, divisor);
            }
        });
    }
    if let Some((k, _)) = current_kv {
        for g in 0..gqa_group {
            let q_head = &q_group[g * head_dim..(g + 1) * head_dim];
            scores[g * total_rows + history_rows] = dot_row(q_head, k, divisor);
        }
    }
    ensure_finite(scores, invalid)?;

    let mut stack_sums = [0.0_f32; 16];
    let mut heap_sums = if gqa_group > stack_sums.len() {
        vec![0.0_f32; gqa_group]
    } else {
        Vec::new()
    };
    let sums: &mut [f32] = if gqa_group <= stack_sums.len() {
        &mut stack_sums[..gqa_group]
    } else {
        &mut heap_sums[..gqa_group]
    };
    for (g, sum_slot) in sums.iter_mut().enumerate() {
        let head_scores = &mut scores[g * total_rows..(g + 1) * total_rows];
        let max = head_scores
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0_f32;
        for score in &mut *head_scores {
            *score = (*score - max).exp();
            sum += *score;
        }
        if !sum.is_finite() || sum <= 0.0 {
            return Err(invalid());
        }
        *sum_slot = sum;
    }

    out_group.fill(0.0);
    for quad in 0..full_quads {
        let r = quad * 4;
        values.with_quad_rows(r, row_width, offset, head_dim, &mut buf, |v_refs| {
            dispatch!(level, simd => {
                weighted_value_accum_4_group_simd(
                    simd,
                    out_group,
                    v_refs,
                    scores,
                    sums,
                    head_dim,
                    gqa_group,
                    total_rows,
                    r,
                );
            });
        });
    }
    for row in (full_quads * 4)..history_rows {
        values.with_row(row, row_width, offset, head_dim, &mut buf, |v| {
            for (g, &sum) in sums.iter().enumerate() {
                let w = scores[g * total_rows + row] / sum;
                let out_head = &mut out_group[g * head_dim..(g + 1) * head_dim];
                weighted_value_accum(out_head, v, w);
            }
        });
    }
    if let Some((_, v)) = current_kv {
        for (g, &sum) in sums.iter().enumerate() {
            let w = scores[g * total_rows + history_rows] / sum;
            let out_head = &mut out_group[g * head_dim..(g + 1) * head_dim];
            weighted_value_accum(out_head, v, w);
        }
    }
    Ok(())
}

/// Evaluate all `gqa_group` query heads sharing one `F32` KV head in a single pass
/// over `keys`/`values`, producing bit-identical outputs to calling `attend_head` per head.
#[expect(
    clippy::too_many_arguments,
    reason = "Shared GQA KV-group attention kernel reuses caller score/output storage without allocating."
)]
pub(super) fn attend_kv_group_f32(
    q_group: &[f32],
    keys: &[f32],
    values: &[f32],
    history_rows: usize,
    row_width: usize,
    offset: usize,
    head_dim: usize,
    gqa_group: usize,
    divisor: f32,
    current_kv: Option<(&[f32], &[f32])>,
    scores: &mut [f32],
    out_group: &mut [f32],
    invalid: fn() -> std::io::Error,
) -> std::io::Result<()> {
    attend_kv_group_impl(
        q_group,
        keys,
        values,
        history_rows,
        row_width,
        offset,
        head_dim,
        gqa_group,
        divisor,
        current_kv,
        scores,
        out_group,
        invalid,
    )
}

/// Evaluate all `gqa_group` query heads sharing one `F16` KV head in a single pass,
/// widening each K/V vector once per position and matching `attend_head_f16` bit-for-bit.
#[expect(
    clippy::too_many_arguments,
    reason = "Shared GQA KV-group attention kernel reuses caller score/output storage without allocating."
)]
pub(super) fn attend_kv_group_f16(
    q_group: &[f32],
    keys: &[u16],
    values: &[u16],
    history_rows: usize,
    row_width: usize,
    offset: usize,
    head_dim: usize,
    gqa_group: usize,
    divisor: f32,
    current_kv: Option<(&[f32], &[f32])>,
    scores: &mut [f32],
    out_group: &mut [f32],
    invalid: fn() -> std::io::Error,
) -> std::io::Result<()> {
    attend_kv_group_impl(
        q_group,
        keys,
        values,
        history_rows,
        row_width,
        offset,
        head_dim,
        gqa_group,
        divisor,
        current_kv,
        scores,
        out_group,
        invalid,
    )
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
                #[cfg(all(windows, feature = "rust-qualification"))]
                WeightSlice::RepackedQ4K8(_) => {}
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

    #[test]
    fn packed_q4_k_q6_k_four_row_and_tail_batch_match_scalar_bits() {
        let input = 512_usize;
        for out_rows in [1_usize, 3, 4, 5, 9, 11] {
            let mut q4 = vec![0_u8; out_rows * 2 * 144];
            for (blk_idx, block) in q4.chunks_exact_mut(144).enumerate() {
                block[0..2].copy_from_slice(
                    &(0x2c00_u16 + u16::try_from(blk_idx % 5).unwrap_or(0) * 0x100).to_le_bytes(),
                );
                block[2..4].copy_from_slice(
                    &(0x2800_u16 + u16::try_from(blk_idx % 3).unwrap_or(0) * 0x100).to_le_bytes(),
                );
                for (i, byte) in block[4..16].iter_mut().enumerate() {
                    *byte = u8::try_from((blk_idx * 7 + i * 5 + 3) % 63).unwrap_or(1);
                }
                for (i, byte) in block[16..].iter_mut().enumerate() {
                    *byte = u8::try_from((blk_idx * 17 + i * 11) % 256).unwrap_or(0);
                }
            }
            let mut q6 = vec![0_u8; out_rows * 2 * 210];
            for (blk_idx, block) in q6.chunks_exact_mut(210).enumerate() {
                block[208..210].copy_from_slice(
                    &(0x2c00_u16 + u16::try_from(blk_idx % 4).unwrap_or(0) * 0x100).to_le_bytes(),
                );
                for (i, byte) in block[..192].iter_mut().enumerate() {
                    *byte = u8::try_from((blk_idx * 13 + i * 9) % 256).unwrap_or(0);
                }
                for (i, byte) in block[192..208].iter_mut().enumerate() {
                    let sc =
                        i8::try_from(((blk_idx * 3 + i * 5) % 31).cast_signed() - 15).unwrap_or(0);
                    *byte = sc.cast_unsigned();
                }
            }
            let wm4 = WeightMatrix {
                input,
                output: out_rows,
                data: WeightSlice::Q4K(&q4),
            };
            let wm6 = WeightMatrix {
                input,
                output: out_rows,
                data: WeightSlice::Q6K(&q6),
            };
            for batch in [1_usize, 2, 4, 5, 8, 9, 13] {
                let raw_in: Vec<f32> = (0..batch * input)
                    .map(|i| {
                        if i % 29 == 0 {
                            -0.0
                        } else {
                            (f32::from(u16::try_from((i * 7) % 41).unwrap_or(0)) - 20.0) * 0.03125
                        }
                    })
                    .collect();
                let prepared = PreparedBatch::new(&raw_in, batch, input);
                for (name, wm) in [("q4_k", wm4), ("q6_k", wm6)] {
                    let mut exp_batch = Vec::with_capacity(batch * out_rows);
                    for t in 0..batch {
                        exp_batch.extend(scalar_multiply_reference(
                            wm,
                            &raw_in[t * input..(t + 1) * input],
                        ));
                    }
                    #[cfg(not(all(windows, feature = "rust-qualification")))]
                    let exp_bits: Vec<u32> = exp_batch.iter().map(|v| v.to_bits()).collect();
                    let mut single_ref_bits = Vec::with_capacity(batch * out_rows);
                    let mut single_row = vec![0.0_f32; out_rows];
                    for t in 0..batch {
                        multiply_single_into(
                            wm,
                            &raw_in[t * input..(t + 1) * input],
                            1,
                            &mut single_row,
                        );
                        single_ref_bits.extend(single_row.iter().map(|v| v.to_bits()));
                    }
                    let mut prev_bits: Option<Vec<u32>> = None;
                    for workers in [1_usize, 3, 8] {
                        let mut got = vec![0.0_f32; batch * out_rows];
                        let mut scratch = Vec::new();
                        multiply_batch_prepared_into(
                            wm,
                            &prepared,
                            &raw_in,
                            workers,
                            &mut scratch,
                            &mut got,
                        );
                        let got_bits: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
                        assert_eq!(
                            got_bits, single_ref_bits,
                            "batch vs single {name} out={out_rows} batch={batch} workers={workers}"
                        );
                        if let Some(prev) = &prev_bits {
                            assert_eq!(
                                &got_bits, prev,
                                "worker invariance {name} out={out_rows} batch={batch} workers={workers}"
                            );
                        } else {
                            prev_bits = Some(got_bits.clone());
                        }
                        #[cfg(not(all(windows, feature = "rust-qualification")))]
                        assert_eq!(
                            got_bits, exp_bits,
                            "exact SIMD bits {name} out={out_rows} batch={batch} workers={workers}"
                        );
                        #[cfg(all(windows, feature = "rust-qualification"))]
                        for (&g, &w) in got.iter().zip(&exp_batch) {
                            let tol = 2.0_f32.max(w.abs() * 0.03);
                            assert!(
                                (g - w).abs() <= tol,
                                "{name} out={out_rows} batch={batch} got={g} want={w}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[cfg(all(windows, feature = "rust-qualification"))]
    #[test]
    fn parallel_quantize_batch_activations_matches_serial_bits() {
        let batch = 19_usize;
        let width = 1024_usize;
        let raw_input: Vec<f32> = (0..batch * width)
            .map(|i| (f32::from(u16::try_from((i * 11) % 67).unwrap_or(0)) - 33.0) * 0.0625)
            .collect();
        let q8_data = vec![0_u8; (width / 32) * 34];
        let q4_data = vec![0_u8; (width / 256) * 144];
        let matrices = [
            WeightMatrix {
                input: width,
                output: 1,
                data: WeightSlice::Q8(&q8_data),
            },
            WeightMatrix {
                input: width,
                output: 1,
                data: WeightSlice::Q4K(&q4_data),
            },
        ];
        let (_, seq_q8, seq_qk) =
            super::quantize_batch_activations(matrices, &raw_input, batch, width, 1);
        for workers in [2_usize, 4, 8] {
            let (_, par_q8, par_qk) =
                super::quantize_batch_activations(matrices, &raw_input, batch, width, workers);
            assert_eq!(par_q8.len(), seq_q8.len());
            for (a, b) in par_q8.iter().zip(&seq_q8) {
                assert_eq!(a.d.to_bits(), b.d.to_bits());
                assert_eq!(a.qs, b.qs);
            }
            assert_eq!(par_qk.len(), seq_qk.len());
            for (a, b) in par_qk.iter().zip(&seq_qk) {
                assert_eq!(a.d.to_bits(), b.d.to_bits());
                assert_eq!(a.qs, b.qs);
                assert_eq!(a.bsums, b.bsums);
            }
        }

        // Calling inside an active row_workers scope falls back cleanly without deadlock.
        let mut outer = [0.0_f32; 4];
        super::row_workers::run_aligned(&mut outer, 4, 1, &|_, slice| {
            let (_, nested_q8, nested_qk) =
                super::quantize_batch_activations(matrices, &raw_input, batch, width, 4);
            assert_eq!(nested_q8.len(), seq_q8.len());
            for (a, b) in nested_q8.iter().zip(&seq_q8) {
                assert_eq!(a.d.to_bits(), b.d.to_bits());
                assert_eq!(a.qs, b.qs);
            }
            assert_eq!(nested_qk.len(), seq_qk.len());
            for (a, b) in nested_qk.iter().zip(&seq_qk) {
                assert_eq!(a.d.to_bits(), b.d.to_bits());
                assert_eq!(a.qs, b.qs);
                assert_eq!(a.bsums, b.bsums);
            }
            slice.fill(1.0);
        });
        assert_eq!(outer, [1.0_f32; 4]);
    }

    #[cfg(all(windows, feature = "rust-qualification"))]
    #[test]
    fn repacked_q4_k_single_batch_and_groups_match_raw_q4_k_bits() -> io::Result<()> {
        let input = 512_usize;
        for out_rows in [8_usize, 16, 24] {
            let blocks_per_row = input / 256;
            let mut q4 = vec![0_u8; out_rows * blocks_per_row * 144];
            for (blk_idx, block) in q4.chunks_exact_mut(144).enumerate() {
                block[0..2].copy_from_slice(
                    &(0x2c00_u16 + u16::try_from(blk_idx % 5).unwrap_or(0) * 0x100).to_le_bytes(),
                );
                block[2..4].copy_from_slice(
                    &(0x2800_u16 + u16::try_from(blk_idx % 3).unwrap_or(0) * 0x100).to_le_bytes(),
                );
                for (i, byte) in block[4..16].iter_mut().enumerate() {
                    *byte = u8::try_from((blk_idx * 7 + i * 5 + 3) % 63).unwrap_or(1);
                }
                for (i, byte) in block[16..].iter_mut().enumerate() {
                    *byte = u8::try_from((blk_idx * 17 + i * 11) % 256).unwrap_or(0);
                }
            }
            let repacked = super::quant_kernels::repack_q4_k_8rows(&q4, input, out_rows)
                .ok_or_else(|| io::Error::other("expected aligned Q4_K repack to succeed"))?;
            assert_eq!(
                repacked.len(),
                (out_rows / 8) * blocks_per_row * super::quant_kernels::Q4_K_REPACK8_BLOCK_BYTES
            );

            let wm_raw = WeightMatrix {
                input,
                output: out_rows,
                data: WeightSlice::Q4K(&q4),
            };
            let wm_repack = WeightMatrix {
                input,
                output: out_rows,
                data: WeightSlice::RepackedQ4K8(&repacked),
            };

            let q8_blocks = input / 32;
            let mut q8_data = vec![0_u8; out_rows * q8_blocks * 34];
            for (blk_idx, block) in q8_data.chunks_exact_mut(34).enumerate() {
                block[0..2].copy_from_slice(&0x3555_u16.to_le_bytes());
                for (i, byte) in block[2..].iter_mut().enumerate() {
                    let v =
                        i8::try_from(((blk_idx * 5 + i * 3) % 25).cast_signed() - 12).unwrap_or(0);
                    *byte = v.cast_unsigned();
                }
            }
            let wm_q8 = WeightMatrix {
                input,
                output: out_rows,
                data: WeightSlice::Q8(&q8_data),
            };

            for batch in [1_usize, 2, 3, 4, 5, 9] {
                let raw_in: Vec<f32> = (0..batch * input)
                    .map(|i| {
                        (f32::from(u16::try_from((i * 11) % 59).unwrap_or(0)) - 29.0) * 0.03125
                    })
                    .collect();
                let prepared = PreparedBatch::new(&raw_in, batch, input);

                let mut exp_single = Vec::with_capacity(batch * out_rows);
                let mut single_buf = vec![0.0_f32; out_rows];
                for t in 0..batch {
                    multiply_single_into(
                        wm_raw,
                        &raw_in[t * input..(t + 1) * input],
                        1,
                        &mut single_buf,
                    );
                    exp_single.extend_from_slice(&single_buf);
                }
                let exp_bits: Vec<u32> = exp_single.iter().map(|v| v.to_bits()).collect();

                let mut exp_q8 = vec![0.0_f32; batch * out_rows];
                let mut scratch = Vec::new();
                multiply_batch_prepared_into(
                    wm_q8,
                    &prepared,
                    &raw_in,
                    1,
                    &mut scratch,
                    &mut exp_q8,
                );
                let exp_q8_bits: Vec<u32> = exp_q8.iter().map(|v| v.to_bits()).collect();

                for workers in [1_usize, 3, 8] {
                    let mut single_repack = Vec::with_capacity(batch * out_rows);
                    for t in 0..batch {
                        multiply_single_into(
                            wm_repack,
                            &raw_in[t * input..(t + 1) * input],
                            workers,
                            &mut single_buf,
                        );
                        single_repack.extend_from_slice(&single_buf);
                    }
                    assert_eq!(
                        single_repack
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>(),
                        exp_bits,
                        "repacked GEMV mismatch out={out_rows} batch={batch} workers={workers}"
                    );

                    scratch.resize(out_rows * batch + 16, f32::NAN);
                    let mut batch_repack = vec![f32::NAN; batch * out_rows];
                    multiply_batch_prepared_into(
                        wm_repack,
                        &prepared,
                        &raw_in,
                        workers,
                        &mut scratch,
                        &mut batch_repack,
                    );
                    assert_eq!(
                        batch_repack.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        exp_bits,
                        "repacked GEMM mismatch out={out_rows} batch={batch} workers={workers}"
                    );

                    let mut grp_repack = vec![0.0_f32; batch * out_rows];
                    let mut grp_q8 = vec![0.0_f32; batch * out_rows];
                    multiply_batch_group_into(
                        &mut [(wm_q8, &mut grp_q8), (wm_repack, &mut grp_repack)],
                        &prepared,
                        &raw_in,
                        workers,
                        &mut scratch,
                    );
                    assert_eq!(
                        grp_repack.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        exp_bits,
                        "repacked batch group mismatch out={out_rows} batch={batch} workers={workers}"
                    );
                    assert_eq!(
                        grp_q8.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        exp_q8_bits,
                        "repacked batch group Q8 sibling mismatch out={out_rows} batch={batch} workers={workers}"
                    );
                }
            }

            if out_rows >= 16 {
                let raw_in: Vec<f32> = (0..3 * input)
                    .map(|i| (f32::from(u16::try_from(i % 37).unwrap_or(0)) - 18.0) * 0.05)
                    .collect();
                let (caps, _, act_qk) =
                    super::quantize_batch_activations([wm_repack], &raw_in, 3, input, 1);
                let mut full_single = vec![0.0_f32; out_rows];
                super::run_single_rows_repacked_q4_k(
                    caps,
                    &repacked,
                    input,
                    0,
                    &act_qk[..blocks_per_row],
                    &mut full_single,
                );
                let mut sub_single = vec![0.0_f32; 7];
                super::run_single_rows_repacked_q4_k(
                    caps,
                    &repacked,
                    input,
                    3,
                    &act_qk[..blocks_per_row],
                    &mut sub_single,
                );
                assert_eq!(
                    sub_single.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    full_single[3..10]
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>()
                );

                let mut full_batch = vec![0.0_f32; out_rows * 3];
                super::run_batch_rows_repacked_q4_k(
                    caps,
                    &repacked,
                    input,
                    0,
                    3,
                    &act_qk,
                    &mut full_batch,
                );
                let mut sub_batch = vec![0.0_f32; 7 * 3];
                super::run_batch_rows_repacked_q4_k(
                    caps,
                    &repacked,
                    input,
                    3,
                    3,
                    &act_qk,
                    &mut sub_batch,
                );
                assert_eq!(
                    sub_batch.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    full_batch[3 * 3..10 * 3]
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>()
                );
            }
        }
        Ok(())
    }

    #[test]
    fn scatter_rows_to_batch_simd_and_parallel_match_scalar_bits() {
        for (output, batch) in [
            (0_usize, 5_usize),
            (5, 0),
            (1, 1),
            (3, 5),
            (5, 3),
            (7, 7),
            (9, 13),
            (1025, 17),
        ] {
            let transposed: Vec<f32> = (0..output * batch)
                .map(|i| {
                    if i % 17 == 0 {
                        -0.0
                    } else {
                        f32::from(u16::try_from((i * 13 + 5) % 401).unwrap_or(0)) * 0.125 - 25.0
                    }
                })
                .collect();
            let mut expected = vec![0.0_f32; batch * output];
            for t in 0..batch {
                for r in 0..output {
                    expected[t * output + r] = transposed[r * batch + t];
                }
            }
            let expected_bits: Vec<u32> = expected.iter().map(|v| v.to_bits()).collect();
            for workers in [1_usize, 2, 4] {
                let mut actual = vec![f32::NAN; batch * output];
                super::scatter_rows_to_batch(&transposed, output, batch, workers, &mut actual);
                assert_eq!(
                    actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    expected_bits,
                    "scatter mismatch output={output} batch={batch} workers={workers}"
                );
            }
        }
    }
}
