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

use std::sync::OnceLock;

use crate::row_workers;
use fearless_simd::{Level, dispatch, f32x4, prelude::*, u8x16};

static SIMD_LEVEL: OnceLock<Level> = OnceLock::new();

pub(super) fn simd_level() -> Level {
    *SIMD_LEVEL.get_or_init(Level::new)
}

/// Borrowed weight slice for row-major `F32` or `Q8_0` matrices.
#[derive(Clone, Copy)]
pub(super) enum WeightSlice<'a> {
    Float(&'a [f32]),
    Q8(&'a [u8]),
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

    pub(super) fn clear(&mut self) {
        self.input = 0;
        self.batch = 0;
        self.num_quads = 0;
        self.quads.clear();
    }

    pub(super) fn prepare_into(&mut self, raw_input: &[f32], batch: usize, input: usize) {
        self.input = input;
        self.batch = batch;
        self.num_quads = batch / 4;
        if self.num_quads > 0 && input > 0 {
            self.quads.resize(self.num_quads * input, [0.0; 4]);
            for q in 0..self.num_quads {
                let p0 = &raw_input[(q * 4) * input..(q * 4 + 1) * input];
                let p1 = &raw_input[(q * 4 + 1) * input..(q * 4 + 2) * input];
                let p2 = &raw_input[(q * 4 + 2) * input..(q * 4 + 3) * input];
                let p3 = &raw_input[(q * 4 + 3) * input..(q * 4 + 4) * input];
                let dst = &mut self.quads[q * input..(q + 1) * input];
                for col in 0..input {
                    dst[col] = [p0[col], p1[col], p2[col], p3[col]];
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
    half::f16::from_le_bytes([bytes[0], bytes[1]]).to_f32()
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
    }
}

/// Bit-exact matrix-vector multiplication across `1x4` SIMD tiles and `row_workers`.
pub(super) fn multiply_single_into(
    weights: WeightMatrix<'_>,
    input: &[f32],
    workers: usize,
    out: &mut [f32],
) {
    let level = simd_level();
    row_workers::run_aligned(out, workers, 4, &|start_row, slice| {
        dispatch!(level, simd => single_rows_simd(simd, weights, start_row, input, slice));
    });
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

/// Bit-exact matrix multiplication for a pre-transposed batch `[batch, input]`
/// using `4x8` / `4x4` SIMD tiles, `1x4` tail positions, and `row_workers`.
pub(super) fn multiply_batch_prepared_into(
    weights: WeightMatrix<'_>,
    prepared: &PreparedBatch,
    raw_input: &[f32],
    workers: usize,
    transposed_scratch: &mut Vec<f32>,
    out: &mut [f32],
) {
    let output = weights.output;
    let input = weights.input;
    let num_quads = prepared.num_quads;
    let level = simd_level();
    if num_quads > 0 {
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
        let quad_out = &mut out[..row_stride * output];
        row_workers::run_aligned(quad_out, workers, 4 * output, &|elem_off, slice| {
            dispatch!(
                level,
                simd => scatter_quads_simd(simd, transposed, row_stride, output, elem_off, slice)
            );
        });
    }
    let tail_start = num_quads * 4;
    for pos in tail_start..prepared.batch {
        multiply_single_into(
            weights,
            &raw_input[pos * input..(pos + 1) * input],
            workers,
            &mut out[pos * output..(pos + 1) * output],
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

#[cfg(test)]
mod tests {
    use super::{
        PreparedBatch, WeightMatrix, WeightSlice, dot_4keys_seq, multiply_batch_prepared_into,
        multiply_single_into, weighted_value_accum, weighted_value_accum_4,
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
                        let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
                        for (&byte, activation) in block[2..].iter().zip(activations) {
                            *target += (scale * f32::from(byte.cast_signed())) * *activation;
                        }
                    }
                }
            }
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

                    for workers in [1_usize, 3, 8] {
                        if batch == 1 {
                            let mut single_out = vec![0.0_f32; output];
                            multiply_single_into(weights, &raw_input, workers, &mut single_out);
                            let actual_bits: Vec<u32> =
                                single_out.iter().map(|v| v.to_bits()).collect();
                            assert_eq!(
                                actual_bits, expected_bits,
                                "single q8={is_q8} out={output} workers={workers}"
                            );
                        }
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
                        let actual_bits: Vec<u32> = batch_out.iter().map(|v| v.to_bits()).collect();
                        assert_eq!(
                            actual_bits, expected_bits,
                            "batch q8={is_q8} out={output} batch={batch} workers={workers}"
                        );
                    }
                }
            }
        }
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
