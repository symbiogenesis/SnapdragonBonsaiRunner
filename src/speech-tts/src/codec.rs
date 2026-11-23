//! Pure-Rust Qwen3-TTS 12 Hz / 48 kHz causal audio codec and streaming `CodecStateSnap` cache.
//!
//! Implements both the 24 kHz (`TOKENIZER_HOP_24K = 1920`) and 48 kHz
//! (`takuma104/Qwen3-TTS-Tokenizer-12Hz-48kHz`, `TOKENIZER_HOP_48K = 3840`)
//! causal `ConvNeXt`/`SnakeBeta` decoder pipelines, the stateful `CodecStateSnap`
//! 8-slot LRU cache for zero-redecode ICL reference priming, and the `SEANet` +
//! Mimi transformer + RVQ reference encoder.
#![forbid(unsafe_code)]
#![expect(
    clippy::missing_errors_doc,
    reason = "Internal codec and convolution shape checks return io::ErrorKind::InvalidInput."
)]

use std::{cell::RefCell, io};

use fearless_simd::{Level, dispatch, f32x4, prelude::*};

use crate::{SAMPLE_RATE, SAMPLE_RATE_48K, engine::row_workers, numerics::usize_to_f32};

thread_local! {
    static CHANNEL_TILE_OUT: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static CONVNEXT_SCRATCH: RefCell<(Vec<f32>, Vec<f32>)> =
        const { RefCell::new((Vec::new(), Vec::new())) };
}

fn with_channel_tile_out<R>(len: usize, f: impl FnOnce(&mut [f32]) -> R) -> R {
    CHANNEL_TILE_OUT.with(|slot| {
        if let Ok(mut buf) = slot.try_borrow_mut() {
            if buf.len() < len {
                buf.resize(len, 0.0_f32);
            }
            f(&mut buf[..len])
        } else {
            let mut buf = vec![0.0_f32; len];
            f(&mut buf)
        }
    })
}

fn with_convnext_scratch<R>(
    c: usize,
    mid_c: usize,
    f: impl FnOnce(&mut [f32], &mut [f32]) -> R,
) -> R {
    CONVNEXT_SCRATCH.with(|slot| {
        if let Ok(mut bufs) = slot.try_borrow_mut() {
            let (y, mid) = &mut *bufs;
            if y.len() < c {
                y.resize(c, 0.0_f32);
            }
            if mid.len() < mid_c {
                mid.resize(mid_c, 0.0_f32);
            }
            f(&mut y[..c], &mut mid[..mid_c])
        } else {
            let mut y = vec![0.0_f32; c];
            let mut mid = vec![0.0_f32; mid_c];
            f(&mut y, &mut mid)
        }
    })
}

/// Reusable `DacDecoder` frame buffers inside [`CodecDecodeScratch`].
#[derive(Default)]
pub(crate) struct DacDecodeScratch {
    conv_ext: Vec<f32>,
    dac_a: Vec<f32>,
    dac_b: Vec<f32>,
    res_buf: Vec<f32>,
    block_buf: Vec<f32>,
    audio: Vec<f32>,
}

/// Reusable frame-decoding buffers across `RvqDecoder`, `CausalConv1d`,
/// `CausalTransConv1d`, `ConvNeXtBlock`, and `DacDecoder` stages.
#[derive(Default)]
pub struct CodecDecodeScratch {
    rvq_sem_codes: Vec<i32>,
    rvq_aco_codes: Vec<i32>,
    rvq_sum_emb: Vec<f32>,
    pub(crate) rvq_sem_out: Vec<f32>,
    rvq_aco_out: Vec<f32>,
    pub(crate) conv_ext: Vec<f32>,
    pub(crate) trans_carry: Vec<f32>,
    pub(crate) pre_out: Vec<f32>,
    pub(crate) upsample_a: Vec<f32>,
    pub(crate) upsample_b: Vec<f32>,
    pub(crate) flat_codes: Vec<i32>,
    pub(crate) latents: Vec<f32>,
    pub(crate) dac: DacDecodeScratch,
}

fn frame_workers(threads: usize, frames: usize, work: usize) -> usize {
    if frames <= 1 || work < 65_536 {
        1
    } else {
        // Preserve the shared Talker/codec team size through 2/4/8-frame
        // stages; fewer jobs may use that team without recreating its workers.
        threads.clamp(1, 18)
    }
}

#[expect(
    clippy::inline_always,
    reason = "fearless_simd dispatch! requires inline(always) to inherit target_feature attributes."
)]
#[inline(always)]
fn slice_binop_in_place_simd<S: Simd, const SUB: bool>(simd: S, dst: &mut [f32], src: &[f32]) {
    let (dst_chunks, dst_tail) = dst.as_chunks_mut::<4>();
    let (src_chunks, src_tail) = src.as_chunks::<4>();
    for (d, s) in dst_chunks.iter_mut().zip(src_chunks.iter()) {
        let dv = f32x4::from_slice(simd, d);
        let sv = f32x4::from_slice(simd, s);
        (if SUB { dv - sv } else { dv + sv }).store_slice(d);
    }
    for (d, &s) in dst_tail.iter_mut().zip(src_tail.iter()) {
        if SUB {
            *d -= s;
        } else {
            *d += s;
        }
    }
}

#[expect(
    clippy::inline_always,
    reason = "fearless_simd dispatch! requires inline(always) to inherit target_feature attributes."
)]
#[inline(always)]
fn add_slice_in_place_simd<S: Simd>(simd: S, dst: &mut [f32], src: &[f32]) {
    slice_binop_in_place_simd::<S, false>(simd, dst, src);
}

#[expect(
    clippy::inline_always,
    reason = "fearless_simd dispatch! requires inline(always) to inherit target_feature attributes."
)]
#[inline(always)]
fn sub_slice_in_place_simd<S: Simd>(simd: S, dst: &mut [f32], src: &[f32]) {
    slice_binop_in_place_simd::<S, true>(simd, dst, src);
}

#[inline]
fn reduce_f32x4<S: Simd>(v: f32x4<S>) -> f32 {
    let mut a = [0.0_f32; 4];
    v.store_slice(&mut a);
    (a[0] + a[2]) + (a[1] + a[3])
}

#[expect(
    clippy::inline_always,
    reason = "fearless_simd dispatch! requires inline(always) to inherit target_feature attributes."
)]
#[inline(always)]
fn dot_f32_simd<S: Simd>(simd: S, a: &[f32], b: &[f32], init: f32) -> f32 {
    let zero = f32x4::splat(simd, 0.0);
    let mut acc0 = zero;
    let mut acc1 = zero;
    let (a8, a_rem8) = a.as_chunks::<8>();
    let (b8, b_rem8) = b.as_chunks::<8>();
    for (ac, bc) in a8.iter().zip(b8.iter()) {
        let av0 = f32x4::from_slice(simd, &ac[..4]);
        let bv0 = f32x4::from_slice(simd, &bc[..4]);
        let av1 = f32x4::from_slice(simd, &ac[4..]);
        let bv1 = f32x4::from_slice(simd, &bc[4..]);
        acc0 = av0.mul_add(bv0, acc0);
        acc1 = av1.mul_add(bv1, acc1);
    }
    let (a4, a_tail) = a_rem8.as_chunks::<4>();
    let (b4, b_tail) = b_rem8.as_chunks::<4>();
    for (ac, bc) in a4.iter().zip(b4.iter()) {
        let av = f32x4::from_slice(simd, ac);
        let bv = f32x4::from_slice(simd, bc);
        acc0 = av.mul_add(bv, acc0);
    }
    let mut sum = init + reduce_f32x4(acc0 + acc1);
    for (&av, &bv) in a_tail.iter().zip(b_tail.iter()) {
        sum = av.mul_add(bv, sum);
    }
    sum
}

#[expect(
    clippy::inline_always,
    reason = "fearless_simd dispatch! requires inline(always) to inherit target_feature attributes."
)]
#[inline(always)]
fn dot_four_rows_f32_simd<S: Simd>(
    simd: S,
    w0: &[f32],
    w1: &[f32],
    w2: &[f32],
    w3: &[f32],
    x: &[f32],
    inits: [f32; 4],
) -> [f32; 4] {
    let zero = f32x4::splat(simd, 0.0);
    let mut a0_0 = zero;
    let mut a0_1 = zero;
    let mut a1_0 = zero;
    let mut a1_1 = zero;
    let mut a2_0 = zero;
    let mut a2_1 = zero;
    let mut a3_0 = zero;
    let mut a3_1 = zero;

    let (x8, x_rem8) = x.as_chunks::<8>();
    let (w0_8, w0_rem8) = w0.as_chunks::<8>();
    let (w1_8, w1_rem8) = w1.as_chunks::<8>();
    let (w2_8, w2_rem8) = w2.as_chunks::<8>();
    let (w3_8, w3_rem8) = w3.as_chunks::<8>();

    for (idx, xc) in x8.iter().enumerate() {
        let xv0 = f32x4::from_slice(simd, &xc[..4]);
        let xv1 = f32x4::from_slice(simd, &xc[4..]);

        let r0 = &w0_8[idx];
        a0_0 = f32x4::from_slice(simd, &r0[..4]).mul_add(xv0, a0_0);
        a0_1 = f32x4::from_slice(simd, &r0[4..]).mul_add(xv1, a0_1);

        let r1 = &w1_8[idx];
        a1_0 = f32x4::from_slice(simd, &r1[..4]).mul_add(xv0, a1_0);
        a1_1 = f32x4::from_slice(simd, &r1[4..]).mul_add(xv1, a1_1);

        let r2 = &w2_8[idx];
        a2_0 = f32x4::from_slice(simd, &r2[..4]).mul_add(xv0, a2_0);
        a2_1 = f32x4::from_slice(simd, &r2[4..]).mul_add(xv1, a2_1);

        let r3 = &w3_8[idx];
        a3_0 = f32x4::from_slice(simd, &r3[..4]).mul_add(xv0, a3_0);
        a3_1 = f32x4::from_slice(simd, &r3[4..]).mul_add(xv1, a3_1);
    }

    let (x4, x_tail) = x_rem8.as_chunks::<4>();
    let (w0_4, w0_tail) = w0_rem8.as_chunks::<4>();
    let (w1_4, w1_tail) = w1_rem8.as_chunks::<4>();
    let (w2_4, w2_tail) = w2_rem8.as_chunks::<4>();
    let (w3_4, w3_tail) = w3_rem8.as_chunks::<4>();

    for (idx, xc) in x4.iter().enumerate() {
        let xv = f32x4::from_slice(simd, xc);
        a0_0 = f32x4::from_slice(simd, &w0_4[idx]).mul_add(xv, a0_0);
        a1_0 = f32x4::from_slice(simd, &w1_4[idx]).mul_add(xv, a1_0);
        a2_0 = f32x4::from_slice(simd, &w2_4[idx]).mul_add(xv, a2_0);
        a3_0 = f32x4::from_slice(simd, &w3_4[idx]).mul_add(xv, a3_0);
    }

    let mut s0 = inits[0] + reduce_f32x4(a0_0 + a0_1);
    let mut s1 = inits[1] + reduce_f32x4(a1_0 + a1_1);
    let mut s2 = inits[2] + reduce_f32x4(a2_0 + a2_1);
    let mut s3 = inits[3] + reduce_f32x4(a3_0 + a3_1);

    for (idx, &xv) in x_tail.iter().enumerate() {
        s0 = w0_tail[idx].mul_add(xv, s0);
        s1 = w1_tail[idx].mul_add(xv, s1);
        s2 = w2_tail[idx].mul_add(xv, s2);
        s3 = w3_tail[idx].mul_add(xv, s3);
    }
    [s0, s1, s2, s3]
}

#[expect(
    clippy::inline_always,
    reason = "fearless_simd dispatch! requires inline(always) to inherit target_feature attributes."
)]
#[inline(always)]
fn matvec_4rows_simd<S: Simd>(
    simd: S,
    weight: &[f32],
    input: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    let in_c = input.len();
    let (out_quads, out_rem) = out.as_chunks_mut::<4>();
    for (q_idx, quad) in out_quads.iter_mut().enumerate() {
        let o_base = q_idx * 4;
        let w_base = o_base * in_c;
        let inits = bias.map_or([0.0_f32; 4], |b| {
            [b[o_base], b[o_base + 1], b[o_base + 2], b[o_base + 3]]
        });
        *quad = dot_four_rows_f32_simd(
            simd,
            &weight[w_base..w_base + in_c],
            &weight[w_base + in_c..w_base + 2 * in_c],
            &weight[w_base + 2 * in_c..w_base + 3 * in_c],
            &weight[w_base + 3 * in_c..w_base + 4 * in_c],
            input,
            inits,
        );
    }
    let rem_base = out_quads.len() * 4;
    for (r_idx, slot) in out_rem.iter_mut().enumerate() {
        let o = rem_base + r_idx;
        let init = bias.map_or(0.0_f32, |b| b[o]);
        *slot = dot_f32_simd(simd, &weight[o * in_c..(o + 1) * in_c], input, init);
    }
}

#[expect(
    clippy::inline_always,
    reason = "fearless_simd dispatch! requires inline(always) to inherit target_feature attributes."
)]
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

#[expect(
    clippy::inline_always,
    reason = "fearless_simd dispatch! requires inline(always) to inherit target_feature attributes."
)]
#[inline(always)]
fn scatter_frame_quads_simd<S: Simd>(
    simd: S,
    transposed: &[f32],
    out_frames: usize,
    out_c: usize,
    start_elem: usize,
    out_quads: &mut [f32],
) {
    let start_quad = start_elem / (4 * out_c);
    let quad_count = out_quads.len() / (4 * out_c);
    let oc_quads = out_c / 4;
    for local_q in 0..quad_count {
        let f0 = (start_quad + local_q) * 4;
        let dst_q = &mut out_quads[local_q * 4 * out_c..(local_q + 1) * 4 * out_c];
        let (p0_p1, p2_p3) = dst_q.split_at_mut(2 * out_c);
        let (p0, p1) = p0_p1.split_at_mut(out_c);
        let (p2, p3) = p2_p3.split_at_mut(out_c);
        for rq in 0..oc_quads {
            let r = rq * 4;
            let r0 = f32x4::from_slice(
                simd,
                &transposed[r * out_frames + f0..r * out_frames + f0 + 4],
            );
            let r1 = f32x4::from_slice(
                simd,
                &transposed[(r + 1) * out_frames + f0..(r + 1) * out_frames + f0 + 4],
            );
            let r2 = f32x4::from_slice(
                simd,
                &transposed[(r + 2) * out_frames + f0..(r + 2) * out_frames + f0 + 4],
            );
            let r3 = f32x4::from_slice(
                simd,
                &transposed[(r + 3) * out_frames + f0..(r + 3) * out_frames + f0 + 4],
            );
            let cols = transpose_4x4(simd, [r0, r1, r2, r3]);
            cols[0].store_slice(&mut p0[r..r + 4]);
            cols[1].store_slice(&mut p1[r..r + 4]);
            cols[2].store_slice(&mut p2[r..r + 4]);
            cols[3].store_slice(&mut p3[r..r + 4]);
        }
        for r in (oc_quads * 4)..out_c {
            let base = r * out_frames + f0;
            p0[r] = transposed[base];
            p1[r] = transposed[base + 1];
            p2[r] = transposed[base + 2];
            p3[r] = transposed[base + 3];
        }
    }
}

fn scatter_transposed_to_time_first(
    transposed: &[f32],
    out_frames: usize,
    out_c: usize,
    threads: usize,
    out: &mut [f32],
) {
    const PARALLEL_SCATTER_WORK: usize = 8_192;
    let level = Level::new();
    let full_quads = out_frames / 4;
    if full_quads > 0 {
        let quad_out = &mut out[..full_quads * 4 * out_c];
        if threads > 1 && full_quads >= 2 && quad_out.len() >= PARALLEL_SCATTER_WORK {
            row_workers::run_aligned(quad_out, threads, 4 * out_c, &|elem_off, slice| {
                dispatch!(
                    level,
                    simd => scatter_frame_quads_simd(
                        simd,
                        transposed,
                        out_frames,
                        out_c,
                        elem_off,
                        slice,
                    )
                );
            });
        } else {
            dispatch!(
                level,
                simd => scatter_frame_quads_simd(simd, transposed, out_frames, out_c, 0, quad_out)
            );
        }
    }
    let rem = out_frames % 4;
    if rem > 0 {
        let tail_f = full_quads * 4;
        for df in 0..rem {
            let f = tail_f + df;
            let dst_row = &mut out[f * out_c..(f + 1) * out_c];
            for (r, dst) in dst_row.iter_mut().enumerate() {
                *dst = transposed[r * out_frames + f];
            }
        }
    }
}

#[expect(
    clippy::inline_always,
    reason = "fearless_simd dispatch! requires inline(always) to inherit target_feature attributes."
)]
#[inline(always)]
fn dot_4rows_4frames_k_f32_simd<S: Simd>(
    simd: S,
    w_quad: &[f32],
    ext: &[f32],
    first_frames: [usize; 4],
    (in_c, kernel, dilation): (usize, usize, usize),
    inits: [f32; 4],
) -> [[f32; 4]; 4] {
    let zero = f32x4::splat(simd, 0.0_f32);
    let mut a = [[zero; 4]; 4];
    let k_in_c = kernel * in_c;
    let quad_in_c = in_c / 4;
    let tail_start = quad_in_c * 4;
    let [f0, f1, f2, f3] = first_frames;

    if quad_in_c > 0 {
        for k in 0..kernel {
            let tap_shift = k * dilation;
            let x0_off = (f0 + tap_shift) * in_c;
            let x1_off = (f1 + tap_shift) * in_c;
            let x2_off = (f2 + tap_shift) * in_c;
            let x3_off = (f3 + tap_shift) * in_c;
            let (x0_q, _) = ext[x0_off..x0_off + tail_start].as_chunks::<4>();
            let (x1_q, _) = ext[x1_off..x1_off + tail_start].as_chunks::<4>();
            let (x2_q, _) = ext[x2_off..x2_off + tail_start].as_chunks::<4>();
            let (x3_q, _) = ext[x3_off..x3_off + tail_start].as_chunks::<4>();

            let w_k = k * in_c;
            let (w0_q, _) = w_quad[w_k..w_k + tail_start].as_chunks::<4>();
            let (w1_q, _) = w_quad[k_in_c + w_k..k_in_c + w_k + tail_start].as_chunks::<4>();
            let (w2_q, _) =
                w_quad[2 * k_in_c + w_k..2 * k_in_c + w_k + tail_start].as_chunks::<4>();
            let (w3_q, _) =
                w_quad[3 * k_in_c + w_k..3 * k_in_c + w_k + tail_start].as_chunks::<4>();

            for q in 0..quad_in_c {
                let xv0 = f32x4::from_slice(simd, &x0_q[q]);
                let xv1 = f32x4::from_slice(simd, &x1_q[q]);
                let xv2 = f32x4::from_slice(simd, &x2_q[q]);
                let xv3 = f32x4::from_slice(simd, &x3_q[q]);

                let wv0 = f32x4::from_slice(simd, &w0_q[q]);
                let wv1 = f32x4::from_slice(simd, &w1_q[q]);
                let wv2 = f32x4::from_slice(simd, &w2_q[q]);
                let wv3 = f32x4::from_slice(simd, &w3_q[q]);

                a[0][0] = wv0.mul_add(xv0, a[0][0]);
                a[0][1] = wv1.mul_add(xv0, a[0][1]);
                a[0][2] = wv2.mul_add(xv0, a[0][2]);
                a[0][3] = wv3.mul_add(xv0, a[0][3]);

                a[1][0] = wv0.mul_add(xv1, a[1][0]);
                a[1][1] = wv1.mul_add(xv1, a[1][1]);
                a[1][2] = wv2.mul_add(xv1, a[1][2]);
                a[1][3] = wv3.mul_add(xv1, a[1][3]);

                a[2][0] = wv0.mul_add(xv2, a[2][0]);
                a[2][1] = wv1.mul_add(xv2, a[2][1]);
                a[2][2] = wv2.mul_add(xv2, a[2][2]);
                a[2][3] = wv3.mul_add(xv2, a[2][3]);

                a[3][0] = wv0.mul_add(xv3, a[3][0]);
                a[3][1] = wv1.mul_add(xv3, a[3][1]);
                a[3][2] = wv2.mul_add(xv3, a[3][2]);
                a[3][3] = wv3.mul_add(xv3, a[3][3]);
            }
        }
    }

    let mut sums = [
        [
            inits[0] + reduce_f32x4(a[0][0]),
            inits[1] + reduce_f32x4(a[0][1]),
            inits[2] + reduce_f32x4(a[0][2]),
            inits[3] + reduce_f32x4(a[0][3]),
        ],
        [
            inits[0] + reduce_f32x4(a[1][0]),
            inits[1] + reduce_f32x4(a[1][1]),
            inits[2] + reduce_f32x4(a[1][2]),
            inits[3] + reduce_f32x4(a[1][3]),
        ],
        [
            inits[0] + reduce_f32x4(a[2][0]),
            inits[1] + reduce_f32x4(a[2][1]),
            inits[2] + reduce_f32x4(a[2][2]),
            inits[3] + reduce_f32x4(a[2][3]),
        ],
        [
            inits[0] + reduce_f32x4(a[3][0]),
            inits[1] + reduce_f32x4(a[3][1]),
            inits[2] + reduce_f32x4(a[3][2]),
            inits[3] + reduce_f32x4(a[3][3]),
        ],
    ];

    if tail_start < in_c {
        let tail_len = in_c - tail_start;
        for k in 0..kernel {
            let tap_shift = k * dilation;
            let x0 = &ext[(f0 + tap_shift) * in_c + tail_start..(f0 + tap_shift) * in_c + in_c];
            let x1 = &ext[(f1 + tap_shift) * in_c + tail_start..(f1 + tap_shift) * in_c + in_c];
            let x2 = &ext[(f2 + tap_shift) * in_c + tail_start..(f2 + tap_shift) * in_c + in_c];
            let x3 = &ext[(f3 + tap_shift) * in_c + tail_start..(f3 + tap_shift) * in_c + in_c];
            let w_off = k * in_c + tail_start;
            let w0 = &w_quad[w_off..w_off + tail_len];
            let w1 = &w_quad[k_in_c + w_off..k_in_c + w_off + tail_len];
            let w2 = &w_quad[2 * k_in_c + w_off..2 * k_in_c + w_off + tail_len];
            let w3 = &w_quad[3 * k_in_c + w_off..3 * k_in_c + w_off + tail_len];
            for i in 0..tail_len {
                let xv = [x0[i], x1[i], x2[i], x3[i]];
                let wv = [w0[i], w1[i], w2[i], w3[i]];
                for f in 0..4 {
                    for r in 0..4 {
                        sums[f][r] = wv[r].mul_add(xv[f], sums[f][r]);
                    }
                }
            }
        }
    }
    sums
}

#[expect(
    clippy::inline_always,
    reason = "fearless_simd dispatch! requires inline(always) to inherit target_feature attributes."
)]
#[inline(always)]
fn dot_1row_4frames_k_f32_simd<S: Simd>(
    simd: S,
    w_row: &[f32],
    ext: &[f32],
    first_frames: [usize; 4],
    (in_c, kernel, dilation): (usize, usize, usize),
    init: f32,
) -> [f32; 4] {
    let zero = f32x4::splat(simd, 0.0_f32);
    let mut a = [zero; 4];
    let quad_in_c = in_c / 4;
    let tail_start = quad_in_c * 4;
    let [f0, f1, f2, f3] = first_frames;

    if quad_in_c > 0 {
        for k in 0..kernel {
            let tap_shift = k * dilation;
            let x0_off = (f0 + tap_shift) * in_c;
            let x1_off = (f1 + tap_shift) * in_c;
            let x2_off = (f2 + tap_shift) * in_c;
            let x3_off = (f3 + tap_shift) * in_c;
            let (x0_q, _) = ext[x0_off..x0_off + tail_start].as_chunks::<4>();
            let (x1_q, _) = ext[x1_off..x1_off + tail_start].as_chunks::<4>();
            let (x2_q, _) = ext[x2_off..x2_off + tail_start].as_chunks::<4>();
            let (x3_q, _) = ext[x3_off..x3_off + tail_start].as_chunks::<4>();
            let w_k = k * in_c;
            let (w_q, _) = w_row[w_k..w_k + tail_start].as_chunks::<4>();

            for q in 0..quad_in_c {
                let wv = f32x4::from_slice(simd, &w_q[q]);
                let xv0 = f32x4::from_slice(simd, &x0_q[q]);
                let xv1 = f32x4::from_slice(simd, &x1_q[q]);
                let xv2 = f32x4::from_slice(simd, &x2_q[q]);
                let xv3 = f32x4::from_slice(simd, &x3_q[q]);
                a[0] = wv.mul_add(xv0, a[0]);
                a[1] = wv.mul_add(xv1, a[1]);
                a[2] = wv.mul_add(xv2, a[2]);
                a[3] = wv.mul_add(xv3, a[3]);
            }
        }
    }

    let mut sums = [
        init + reduce_f32x4(a[0]),
        init + reduce_f32x4(a[1]),
        init + reduce_f32x4(a[2]),
        init + reduce_f32x4(a[3]),
    ];
    if tail_start < in_c {
        let tail_len = in_c - tail_start;
        for k in 0..kernel {
            let tap_shift = k * dilation;
            let x0 = &ext[(f0 + tap_shift) * in_c + tail_start..(f0 + tap_shift) * in_c + in_c];
            let x1 = &ext[(f1 + tap_shift) * in_c + tail_start..(f1 + tap_shift) * in_c + in_c];
            let x2 = &ext[(f2 + tap_shift) * in_c + tail_start..(f2 + tap_shift) * in_c + in_c];
            let x3 = &ext[(f3 + tap_shift) * in_c + tail_start..(f3 + tap_shift) * in_c + in_c];
            let w_off = k * in_c + tail_start;
            let w_t = &w_row[w_off..w_off + tail_len];
            for i in 0..tail_len {
                let wv = w_t[i];
                sums[0] = wv.mul_add(x0[i], sums[0]);
                sums[1] = wv.mul_add(x1[i], sums[1]);
                sums[2] = wv.mul_add(x2[i], sums[2]);
                sums[3] = wv.mul_add(x3[i], sums[3]);
            }
        }
    }
    sums
}

#[expect(
    clippy::inline_always,
    reason = "fearless_simd dispatch! requires inline(always) to inherit target_feature attributes."
)]
#[inline(always)]
fn dot_4rows_1frame_k_f32_simd<S: Simd>(
    simd: S,
    w_quad: &[f32],
    ext: &[f32],
    first_frame: usize,
    (in_c, kernel, dilation): (usize, usize, usize),
    inits: [f32; 4],
) -> [f32; 4] {
    let zero = f32x4::splat(simd, 0.0_f32);
    let mut a = [zero; 4];
    let k_in_c = kernel * in_c;
    let quad_in_c = in_c / 4;
    let tail_start = quad_in_c * 4;

    if quad_in_c > 0 {
        for k in 0..kernel {
            let x_off = (first_frame + k * dilation) * in_c;
            let (x_q, _) = ext[x_off..x_off + tail_start].as_chunks::<4>();
            let w_k = k * in_c;
            let (w0_q, _) = w_quad[w_k..w_k + tail_start].as_chunks::<4>();
            let (w1_q, _) = w_quad[k_in_c + w_k..k_in_c + w_k + tail_start].as_chunks::<4>();
            let (w2_q, _) =
                w_quad[2 * k_in_c + w_k..2 * k_in_c + w_k + tail_start].as_chunks::<4>();
            let (w3_q, _) =
                w_quad[3 * k_in_c + w_k..3 * k_in_c + w_k + tail_start].as_chunks::<4>();

            for q in 0..quad_in_c {
                let xv = f32x4::from_slice(simd, &x_q[q]);
                a[0] = f32x4::from_slice(simd, &w0_q[q]).mul_add(xv, a[0]);
                a[1] = f32x4::from_slice(simd, &w1_q[q]).mul_add(xv, a[1]);
                a[2] = f32x4::from_slice(simd, &w2_q[q]).mul_add(xv, a[2]);
                a[3] = f32x4::from_slice(simd, &w3_q[q]).mul_add(xv, a[3]);
            }
        }
    }

    let mut sums = [
        inits[0] + reduce_f32x4(a[0]),
        inits[1] + reduce_f32x4(a[1]),
        inits[2] + reduce_f32x4(a[2]),
        inits[3] + reduce_f32x4(a[3]),
    ];
    if tail_start < in_c {
        let tail_len = in_c - tail_start;
        for k in 0..kernel {
            let x_off = (first_frame + k * dilation) * in_c + tail_start;
            let xt = &ext[x_off..x_off + tail_len];
            let w_off = k * in_c + tail_start;
            let w0 = &w_quad[w_off..w_off + tail_len];
            let w1 = &w_quad[k_in_c + w_off..k_in_c + w_off + tail_len];
            let w2 = &w_quad[2 * k_in_c + w_off..2 * k_in_c + w_off + tail_len];
            let w3 = &w_quad[3 * k_in_c + w_off..3 * k_in_c + w_off + tail_len];
            for i in 0..tail_len {
                let xv = xt[i];
                sums[0] = w0[i].mul_add(xv, sums[0]);
                sums[1] = w1[i].mul_add(xv, sums[1]);
                sums[2] = w2[i].mul_add(xv, sums[2]);
                sums[3] = w3[i].mul_add(xv, sums[3]);
            }
        }
    }
    sums
}

#[expect(
    clippy::inline_always,
    reason = "fearless_simd dispatch! requires inline(always) to inherit target_feature attributes."
)]
#[inline(always)]
fn dot_1row_1frame_k_f32_simd<S: Simd>(
    simd: S,
    w_row: &[f32],
    ext: &[f32],
    first_frame: usize,
    (in_c, kernel, dilation): (usize, usize, usize),
    init: f32,
) -> f32 {
    let mut a = f32x4::splat(simd, 0.0_f32);
    let quad_in_c = in_c / 4;
    let tail_start = quad_in_c * 4;

    if quad_in_c > 0 {
        for k in 0..kernel {
            let x_off = (first_frame + k * dilation) * in_c;
            let (x_q, _) = ext[x_off..x_off + tail_start].as_chunks::<4>();
            let w_k = k * in_c;
            let (w_q, _) = w_row[w_k..w_k + tail_start].as_chunks::<4>();
            for q in 0..quad_in_c {
                a = f32x4::from_slice(simd, &w_q[q]).mul_add(f32x4::from_slice(simd, &x_q[q]), a);
            }
        }
    }

    let mut sum = init + reduce_f32x4(a);
    if tail_start < in_c {
        let tail_len = in_c - tail_start;
        for k in 0..kernel {
            let x_off = (first_frame + k * dilation) * in_c + tail_start;
            let xt = &ext[x_off..x_off + tail_len];
            let w_off = k * in_c + tail_start;
            let wt = &w_row[w_off..w_off + tail_len];
            for i in 0..tail_len {
                sum = wt[i].mul_add(xt[i], sum);
            }
        }
    }
    sum
}

#[expect(
    clippy::inline_always,
    reason = "fearless_simd dispatch! requires inline(always) to inherit target_feature attributes."
)]
#[inline(always)]
fn conv1d_oc_outer_time_first_simd<S: Simd>(
    simd: S,
    w_rows: &[f32],
    bias_rows: Option<&[f32]>,
    ext: &[f32],
    (first_frame, num_frames, stride): (usize, usize, usize),
    geom: (usize, usize, usize),
    out: &mut [f32],
) {
    let (in_c, kernel, _dilation) = geom;
    let k_in_c = kernel * in_c;
    let num_rows = w_rows.len() / k_in_c;
    let quads = num_rows / 4;

    for q_idx in 0..quads {
        let oc = q_idx * 4;
        let w_quad = &w_rows[oc * k_in_c..(oc + 4) * k_in_c];
        let inits = bias_rows.map_or([0.0_f32; 4], |b| [b[oc], b[oc + 1], b[oc + 2], b[oc + 3]]);
        let mut f = 0_usize;
        while f + 4 <= num_frames {
            let ff = [
                (first_frame + f) * stride,
                (first_frame + f + 1) * stride,
                (first_frame + f + 2) * stride,
                (first_frame + f + 3) * stride,
            ];
            let dots = dot_4rows_4frames_k_f32_simd(simd, w_quad, ext, ff, geom, inits);
            for (df, dot_row) in dots.iter().enumerate() {
                let dst_off = (f + df) * num_rows + oc;
                out[dst_off..dst_off + 4].copy_from_slice(dot_row);
            }
            f += 4;
        }
        while f < num_frames {
            let dots = dot_4rows_1frame_k_f32_simd(
                simd,
                w_quad,
                ext,
                (first_frame + f) * stride,
                geom,
                inits,
            );
            let dst_off = f * num_rows + oc;
            out[dst_off..dst_off + 4].copy_from_slice(&dots);
            f += 1;
        }
    }

    for oc in (quads * 4)..num_rows {
        let w_row = &w_rows[oc * k_in_c..(oc + 1) * k_in_c];
        let init = bias_rows.map_or(0.0_f32, |b| b[oc]);
        let mut f = 0_usize;
        while f + 4 <= num_frames {
            let ff = [
                (first_frame + f) * stride,
                (first_frame + f + 1) * stride,
                (first_frame + f + 2) * stride,
                (first_frame + f + 3) * stride,
            ];
            let dots = dot_1row_4frames_k_f32_simd(simd, w_row, ext, ff, geom, init);
            for (df, &val) in dots.iter().enumerate() {
                out[(f + df) * num_rows + oc] = val;
            }
            f += 4;
        }
        while f < num_frames {
            out[f * num_rows + oc] = dot_1row_1frame_k_f32_simd(
                simd,
                w_row,
                ext,
                (first_frame + f) * stride,
                geom,
                init,
            );
            f += 1;
        }
    }
}

#[expect(
    clippy::inline_always,
    reason = "fearless_simd dispatch! requires inline(always) to inherit target_feature attributes."
)]
#[inline(always)]
fn conv1d_oc_outer_channel_first_simd<S: Simd>(
    simd: S,
    w_rows: &[f32],
    bias_rows: Option<&[f32]>,
    ext: &[f32],
    (num_frames, stride): (usize, usize),
    geom: (usize, usize, usize),
    transposed_rows: &mut [f32],
) {
    let (in_c, kernel, _dilation) = geom;
    let k_in_c = kernel * in_c;
    let num_rows = w_rows.len() / k_in_c;
    let quads = num_rows / 4;

    for q_idx in 0..quads {
        let oc = q_idx * 4;
        let w_quad = &w_rows[oc * k_in_c..(oc + 4) * k_in_c];
        let inits = bias_rows.map_or([0.0_f32; 4], |b| [b[oc], b[oc + 1], b[oc + 2], b[oc + 3]]);
        let dst_quad = &mut transposed_rows[oc * num_frames..(oc + 4) * num_frames];
        let (r0_r1, r2_r3) = dst_quad.split_at_mut(2 * num_frames);
        let (r0, r1) = r0_r1.split_at_mut(num_frames);
        let (r2, r3) = r2_r3.split_at_mut(num_frames);
        let mut f = 0_usize;
        while f + 4 <= num_frames {
            let ff = [
                f * stride,
                (f + 1) * stride,
                (f + 2) * stride,
                (f + 3) * stride,
            ];
            let dots = dot_4rows_4frames_k_f32_simd(simd, w_quad, ext, ff, geom, inits);
            for df in 0..4 {
                r0[f + df] = dots[df][0];
                r1[f + df] = dots[df][1];
                r2[f + df] = dots[df][2];
                r3[f + df] = dots[df][3];
            }
            f += 4;
        }
        while f < num_frames {
            let dots = dot_4rows_1frame_k_f32_simd(simd, w_quad, ext, f * stride, geom, inits);
            r0[f] = dots[0];
            r1[f] = dots[1];
            r2[f] = dots[2];
            r3[f] = dots[3];
            f += 1;
        }
    }

    for oc in (quads * 4)..num_rows {
        let w_row = &w_rows[oc * k_in_c..(oc + 1) * k_in_c];
        let init = bias_rows.map_or(0.0_f32, |b| b[oc]);
        let dst_row = &mut transposed_rows[oc * num_frames..(oc + 1) * num_frames];
        let mut f = 0_usize;
        while f + 4 <= num_frames {
            let ff = [
                f * stride,
                (f + 1) * stride,
                (f + 2) * stride,
                (f + 3) * stride,
            ];
            let dots = dot_1row_4frames_k_f32_simd(simd, w_row, ext, ff, geom, init);
            dst_row[f..f + 4].copy_from_slice(&dots);
            f += 4;
        }
        while f < num_frames {
            dst_row[f] = dot_1row_1frame_k_f32_simd(simd, w_row, ext, f * stride, geom, init);
            f += 1;
        }
    }
}

#[inline]
const fn trans_conv_phase_taps(kernel: usize, stride: usize, phase: usize) -> usize {
    (kernel - phase).div_ceil(stride)
}

#[inline]
const fn trans_conv_phase_tap_offset(kernel: usize, stride: usize, phase: usize) -> usize {
    let q = kernel / stride;
    let r = kernel % stride;
    phase * q + if phase < r { phase } else { r }
}

#[expect(
    clippy::inline_always,
    reason = "fearless_simd dispatch! requires inline(always) to inherit target_feature attributes."
)]
#[inline(always)]
fn trans_conv_frames_16oc_simd<S: Simd, const F: usize>(
    simd: S,
    input: &[f32],
    w_tile: &[f32],
    bias_blk: Option<&[f32]>,
    (t0_base, n_taps, in_c): (usize, usize, usize),
    (dst, dst_base, dst_stride): (&mut [f32], usize, usize),
) {
    let zero = f32x4::splat(simd, 0.0_f32);
    let mut acc = [[zero; 4]; F];
    let tap_stride = in_c * 16;

    for tap_idx in 0..n_taps {
        let m = n_taps - 1 - tap_idx;
        let t0 = t0_base - m;
        let x_rows: [&[f32]; F] =
            core::array::from_fn(|f| &input[(t0 + f) * in_c..(t0 + f + 1) * in_c]);
        let (w_chunks, _) =
            w_tile[tap_idx * tap_stride..(tap_idx + 1) * tap_stride].as_chunks::<16>();
        for (ic, w_chunk) in w_chunks.iter().enumerate() {
            let xvs: [f32x4<S>; F] = core::array::from_fn(|f| f32x4::splat(simd, x_rows[f][ic]));
            let wv0 = f32x4::from_slice(simd, &w_chunk[0..4]);
            let wv1 = f32x4::from_slice(simd, &w_chunk[4..8]);
            let wv2 = f32x4::from_slice(simd, &w_chunk[8..12]);
            let wv3 = f32x4::from_slice(simd, &w_chunk[12..16]);
            for f in 0..F {
                acc[f][0] = xvs[f].mul_add(wv0, acc[f][0]);
                acc[f][1] = xvs[f].mul_add(wv1, acc[f][1]);
                acc[f][2] = xvs[f].mul_add(wv2, acc[f][2]);
                acc[f][3] = xvs[f].mul_add(wv3, acc[f][3]);
            }
        }
    }

    if let Some(b) = bias_blk {
        let bv0 = f32x4::from_slice(simd, &b[0..4]);
        let bv1 = f32x4::from_slice(simd, &b[4..8]);
        let bv2 = f32x4::from_slice(simd, &b[8..12]);
        let bv3 = f32x4::from_slice(simd, &b[12..16]);
        for f in 0..F {
            let row = &mut dst[dst_base + f * dst_stride..dst_base + f * dst_stride + 16];
            (acc[f][0] + bv0).store_slice(&mut row[0..4]);
            (acc[f][1] + bv1).store_slice(&mut row[4..8]);
            (acc[f][2] + bv2).store_slice(&mut row[8..12]);
            (acc[f][3] + bv3).store_slice(&mut row[12..16]);
        }
    } else {
        for f in 0..F {
            let row = &mut dst[dst_base + f * dst_stride..dst_base + f * dst_stride + 16];
            acc[f][0].store_slice(&mut row[0..4]);
            acc[f][1].store_slice(&mut row[4..8]);
            acc[f][2].store_slice(&mut row[8..12]);
            acc[f][3].store_slice(&mut row[12..16]);
        }
    }
}

#[expect(
    clippy::inline_always,
    reason = "fearless_simd dispatch! requires inline(always) to inherit target_feature attributes."
)]
#[inline(always)]
fn trans_conv_frames_tail_oc_simd<S: Simd, const F: usize>(
    simd: S,
    input: &[f32],
    w_tile: &[f32],
    bias_blk: Option<&[f32]>,
    (t0_base, n_taps, in_c, lane_w): (usize, usize, usize, usize),
    (dst, dst_base, dst_stride): (&mut [f32], usize, usize),
) {
    let tap_stride = in_c * lane_w;
    let quads = lane_w / 4;

    for qg in 0..quads {
        let lane_off = qg * 4;
        let mut acc = [f32x4::splat(simd, 0.0_f32); F];
        for tap_idx in 0..n_taps {
            let m = n_taps - 1 - tap_idx;
            let t0 = t0_base - m;
            let x_rows: [&[f32]; F] =
                core::array::from_fn(|f| &input[(t0 + f) * in_c..(t0 + f + 1) * in_c]);
            let w_tap = &w_tile[tap_idx * tap_stride..(tap_idx + 1) * tap_stride];
            for ic in 0..in_c {
                let wv = f32x4::from_slice(
                    simd,
                    &w_tap[ic * lane_w + lane_off..ic * lane_w + lane_off + 4],
                );
                for f in 0..F {
                    acc[f] = f32x4::splat(simd, x_rows[f][ic]).mul_add(wv, acc[f]);
                }
            }
        }
        if let Some(b) = bias_blk {
            let bv = f32x4::from_slice(simd, &b[lane_off..lane_off + 4]);
            for f in 0..F {
                (acc[f] + bv).store_slice(
                    &mut dst[dst_base + f * dst_stride + lane_off
                        ..dst_base + f * dst_stride + lane_off + 4],
                );
            }
        } else {
            for f in 0..F {
                acc[f].store_slice(
                    &mut dst[dst_base + f * dst_stride + lane_off
                        ..dst_base + f * dst_stride + lane_off + 4],
                );
            }
        }
    }

    for lane in (quads * 4)..lane_w {
        let mut acc = [0.0_f32; F];
        for tap_idx in 0..n_taps {
            let m = n_taps - 1 - tap_idx;
            let t0 = t0_base - m;
            let x_rows: [&[f32]; F] =
                core::array::from_fn(|f| &input[(t0 + f) * in_c..(t0 + f + 1) * in_c]);
            let w_tap = &w_tile[tap_idx * tap_stride..(tap_idx + 1) * tap_stride];
            for ic in 0..in_c {
                let wv = w_tap[ic * lane_w + lane];
                for f in 0..F {
                    acc[f] = x_rows[f][ic].mul_add(wv, acc[f]);
                }
            }
        }
        if let Some(b) = bias_blk {
            let bv = b[lane];
            for f in 0..F {
                dst[dst_base + f * dst_stride + lane] = acc[f] + bv;
            }
        } else {
            for f in 0..F {
                dst[dst_base + f * dst_stride + lane] = acc[f];
            }
        }
    }
}

#[expect(
    clippy::inline_always,
    reason = "fearless_simd dispatch! requires inline(always) to inherit target_feature attributes."
)]
#[inline(always)]
fn trans_conv_edge_1frame_16oc_simd<S: Simd>(
    simd: S,
    input: &[f32],
    w_tile: &[f32],
    init_carry: Option<&[f32]>,
    bias_blk: Option<&[f32]>,
    (q, frames, n_taps, in_c): (usize, usize, usize, usize),
    dst_row: &mut [f32],
) {
    let mut acc = if let Some(c) = init_carry {
        [
            f32x4::from_slice(simd, &c[0..4]),
            f32x4::from_slice(simd, &c[4..8]),
            f32x4::from_slice(simd, &c[8..12]),
            f32x4::from_slice(simd, &c[12..16]),
        ]
    } else {
        [f32x4::splat(simd, 0.0_f32); 4]
    };

    let m_hi = q.min(n_taps - 1);
    let m_lo = (q + 1).saturating_sub(frames);
    let tap_start = (n_taps - 1) - m_hi;
    let tap_end = n_taps - m_lo;
    let tap_stride = in_c * 16;

    for tap_idx in tap_start..tap_end {
        let m = n_taps - 1 - tap_idx;
        let t = q - m;
        let x_row = &input[t * in_c..(t + 1) * in_c];
        let (w_chunks, _) =
            w_tile[tap_idx * tap_stride..(tap_idx + 1) * tap_stride].as_chunks::<16>();
        for (ic, w_chunk) in w_chunks.iter().enumerate() {
            let xv = f32x4::splat(simd, x_row[ic]);
            acc[0] = xv.mul_add(f32x4::from_slice(simd, &w_chunk[0..4]), acc[0]);
            acc[1] = xv.mul_add(f32x4::from_slice(simd, &w_chunk[4..8]), acc[1]);
            acc[2] = xv.mul_add(f32x4::from_slice(simd, &w_chunk[8..12]), acc[2]);
            acc[3] = xv.mul_add(f32x4::from_slice(simd, &w_chunk[12..16]), acc[3]);
        }
    }

    if let Some(b) = bias_blk {
        acc[0] += f32x4::from_slice(simd, &b[0..4]);
        acc[1] += f32x4::from_slice(simd, &b[4..8]);
        acc[2] += f32x4::from_slice(simd, &b[8..12]);
        acc[3] += f32x4::from_slice(simd, &b[12..16]);
    }
    acc[0].store_slice(&mut dst_row[0..4]);
    acc[1].store_slice(&mut dst_row[4..8]);
    acc[2].store_slice(&mut dst_row[8..12]);
    acc[3].store_slice(&mut dst_row[12..16]);
}

#[expect(
    clippy::inline_always,
    reason = "fearless_simd dispatch! requires inline(always) to inherit target_feature attributes."
)]
#[inline(always)]
fn trans_conv_edge_1frame_tail_oc_simd<S: Simd>(
    simd: S,
    input: &[f32],
    w_tile: &[f32],
    init_carry: Option<&[f32]>,
    bias_blk: Option<&[f32]>,
    (q, frames, n_taps, in_c, lane_w): (usize, usize, usize, usize, usize),
    dst_row: &mut [f32],
) {
    let m_hi = q.min(n_taps - 1);
    let m_lo = (q + 1).saturating_sub(frames);
    let tap_start = (n_taps - 1) - m_hi;
    let tap_end = n_taps - m_lo;
    let tap_stride = in_c * lane_w;
    let quads = lane_w / 4;

    for qg in 0..quads {
        let lane_off = qg * 4;
        let mut acc = init_carry.map_or_else(
            || f32x4::splat(simd, 0.0_f32),
            |c| f32x4::from_slice(simd, &c[lane_off..lane_off + 4]),
        );
        for tap_idx in tap_start..tap_end {
            let m = n_taps - 1 - tap_idx;
            let t = q - m;
            let x_row = &input[t * in_c..(t + 1) * in_c];
            let w_tap = &w_tile[tap_idx * tap_stride..(tap_idx + 1) * tap_stride];
            for ic in 0..in_c {
                let wv = f32x4::from_slice(
                    simd,
                    &w_tap[ic * lane_w + lane_off..ic * lane_w + lane_off + 4],
                );
                acc = f32x4::splat(simd, x_row[ic]).mul_add(wv, acc);
            }
        }
        if let Some(b) = bias_blk {
            acc += f32x4::from_slice(simd, &b[lane_off..lane_off + 4]);
        }
        acc.store_slice(&mut dst_row[lane_off..lane_off + 4]);
    }

    for lane in (quads * 4)..lane_w {
        let mut acc = init_carry.map_or(0.0_f32, |c| c[lane]);
        for tap_idx in tap_start..tap_end {
            let m = n_taps - 1 - tap_idx;
            let t = q - m;
            let x_row = &input[t * in_c..(t + 1) * in_c];
            let w_tap = &w_tile[tap_idx * tap_stride..(tap_idx + 1) * tap_stride];
            for ic in 0..in_c {
                acc = x_row[ic].mul_add(w_tap[ic * lane_w + lane], acc);
            }
        }
        if let Some(b) = bias_blk {
            acc += b[lane];
        }
        dst_row[lane] = acc;
    }
}

#[expect(
    clippy::inline_always,
    reason = "fearless_simd dispatch! requires inline(always) to inherit target_feature attributes."
)]
#[inline(always)]
fn trans_conv_run_q_range_simd<S: Simd>(
    simd: S,
    input: &[f32],
    carry: &[f32],
    w_tile: &[f32],
    bias_blk: Option<&[f32]>,
    (phase, stride, frames, n_taps, in_c, out_c, oc_base, lane_w): (
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
    ),
    (q_begin, q_end): (usize, usize),
    (dst, dst_origin_frame, dst_frame_stride, dst_oc_stride, dst_oc_off): (
        &mut [f32],
        usize,
        usize,
        usize,
        usize,
    ),
) {
    let q_count = frames + n_taps - 1;
    let interior_start = (n_taps - 1).min(q_count);
    let edge1_end = interior_start.clamp(q_begin, q_end);
    let mid_start = edge1_end;
    let mid_end = frames.clamp(mid_start, q_end);
    let edge2_start = mid_end;
    let row_step = dst_frame_stride * dst_oc_stride;

    let mut q = q_begin;
    while q < edge1_end {
        let y = q * stride + phase;
        let carry_row = &carry[y * out_c + oc_base..y * out_c + oc_base + lane_w];
        let b_opt = (q < frames).then_some(bias_blk).flatten();
        let dst_off = if dst_frame_stride == 1 {
            (q - dst_origin_frame) * dst_oc_stride + dst_oc_off
        } else {
            (y - dst_origin_frame) * dst_oc_stride + dst_oc_off
        };
        let dst_row = &mut dst[dst_off..dst_off + lane_w];
        if lane_w == 16 {
            trans_conv_edge_1frame_16oc_simd(
                simd,
                input,
                w_tile,
                Some(carry_row),
                b_opt,
                (q, frames, n_taps, in_c),
                dst_row,
            );
        } else {
            trans_conv_edge_1frame_tail_oc_simd(
                simd,
                input,
                w_tile,
                Some(carry_row),
                b_opt,
                (q, frames, n_taps, in_c, lane_w),
                dst_row,
            );
        }
        q += 1;
    }

    let mut q_mid = mid_start;
    while q_mid + 4 <= mid_end {
        let dst_base = if dst_frame_stride == 1 {
            (q_mid - dst_origin_frame) * dst_oc_stride + dst_oc_off
        } else {
            (q_mid * stride + phase - dst_origin_frame) * dst_oc_stride + dst_oc_off
        };
        if lane_w == 16 {
            trans_conv_frames_16oc_simd::<S, 4>(
                simd,
                input,
                w_tile,
                bias_blk,
                (q_mid, n_taps, in_c),
                (dst, dst_base, row_step),
            );
        } else {
            trans_conv_frames_tail_oc_simd::<S, 4>(
                simd,
                input,
                w_tile,
                bias_blk,
                (q_mid, n_taps, in_c, lane_w),
                (dst, dst_base, row_step),
            );
        }
        q_mid += 4;
    }
    let rem = mid_end - q_mid;
    if rem > 0 {
        let dst_base = if dst_frame_stride == 1 {
            (q_mid - dst_origin_frame) * dst_oc_stride + dst_oc_off
        } else {
            (q_mid * stride + phase - dst_origin_frame) * dst_oc_stride + dst_oc_off
        };
        match (lane_w == 16, rem) {
            (true, 3) => trans_conv_frames_16oc_simd::<S, 3>(
                simd,
                input,
                w_tile,
                bias_blk,
                (q_mid, n_taps, in_c),
                (dst, dst_base, row_step),
            ),
            (true, 2) => trans_conv_frames_16oc_simd::<S, 2>(
                simd,
                input,
                w_tile,
                bias_blk,
                (q_mid, n_taps, in_c),
                (dst, dst_base, row_step),
            ),
            (true, _) => trans_conv_frames_16oc_simd::<S, 1>(
                simd,
                input,
                w_tile,
                bias_blk,
                (q_mid, n_taps, in_c),
                (dst, dst_base, row_step),
            ),
            (false, 3) => trans_conv_frames_tail_oc_simd::<S, 3>(
                simd,
                input,
                w_tile,
                bias_blk,
                (q_mid, n_taps, in_c, lane_w),
                (dst, dst_base, row_step),
            ),
            (false, 2) => trans_conv_frames_tail_oc_simd::<S, 2>(
                simd,
                input,
                w_tile,
                bias_blk,
                (q_mid, n_taps, in_c, lane_w),
                (dst, dst_base, row_step),
            ),
            (false, _) => trans_conv_frames_tail_oc_simd::<S, 1>(
                simd,
                input,
                w_tile,
                bias_blk,
                (q_mid, n_taps, in_c, lane_w),
                (dst, dst_base, row_step),
            ),
        }
    }

    for q_edge in edge2_start..q_end {
        let y = q_edge * stride + phase;
        let dst_off = if dst_frame_stride == 1 {
            (q_edge - dst_origin_frame) * dst_oc_stride + dst_oc_off
        } else {
            (y - dst_origin_frame) * dst_oc_stride + dst_oc_off
        };
        let dst_row = &mut dst[dst_off..dst_off + lane_w];
        if lane_w == 16 {
            trans_conv_edge_1frame_16oc_simd(
                simd,
                input,
                w_tile,
                None,
                None,
                (q_edge, frames, n_taps, in_c),
                dst_row,
            );
        } else {
            trans_conv_edge_1frame_tail_oc_simd(
                simd,
                input,
                w_tile,
                None,
                None,
                (q_edge, frames, n_taps, in_c, lane_w),
                dst_row,
            );
        }
    }
}

#[expect(
    clippy::inline_always,
    reason = "fearless_simd dispatch! requires inline(always) to inherit target_feature attributes."
)]
#[inline(always)]
fn scatter_phase_blocks_to_raw_simd<S: Simd>(
    simd: S,
    scratch: &[f32],
    tile_elems: usize,
    stride: usize,
    out_c: usize,
    first_frame: usize,
    raw_slice: &mut [f32],
) {
    let oc_blocks = out_c.div_ceil(16);
    for (local_y, dst_row) in raw_slice.chunks_exact_mut(out_c).enumerate() {
        let y = first_frame + local_y;
        let phase = y % stride;
        let q = y / stride;
        let phase_tile_base = phase * oc_blocks;
        for oc_blk in 0..oc_blocks {
            let oc_base = oc_blk * 16;
            let lane_w = 16.min(out_c - oc_base);
            let src_off = (phase_tile_base + oc_blk) * tile_elems + q * 16;
            let src = &scratch[src_off..src_off + lane_w];
            let dst = &mut dst_row[oc_base..oc_base + lane_w];
            if lane_w == 16 {
                f32x4::from_slice(simd, &src[0..4]).store_slice(&mut dst[0..4]);
                f32x4::from_slice(simd, &src[4..8]).store_slice(&mut dst[4..8]);
                f32x4::from_slice(simd, &src[8..12]).store_slice(&mut dst[8..12]);
                f32x4::from_slice(simd, &src[12..16]).store_slice(&mut dst[12..16]);
            } else {
                dst.copy_from_slice(src);
            }
        }
    }
}

#[inline(always)]
fn snake_beta_row_simd<S: Simd>(simd: S, row: &mut [f32], a: &[f32], inv_b: &[f32]) {
    // Three Cody-Waite pi parts where PI1/PI2 retain 14 trailing mantissa zero
    // bits so `k * PI1` and `k * PI2` remain exact in `f32` up to `|k| <= 16_384`
    // (`|z| <= 32_768`). Because `sin^2(r + k * pi) == sin^2(r)` for every integer
    // `k`, no quadrant sign correction is needed before squaring `sin(r)`.
    const INV_PI: f32 = std::f32::consts::FRAC_1_PI;
    const PI1: f32 = f32::from_bits(0x4049_0000);
    const PI2: f32 = f32::from_bits(0x3a7d_a000);
    const PI3: f32 = f32::from_bits(0x3422_2169);
    const RANGE_LIMIT: f32 = 32_768.0;
    const C3: f32 = f32::from_bits(0xbe2a_aaab);
    const C5: f32 = f32::from_bits(0x3c08_8889);
    const C7: f32 = f32::from_bits(0xb950_0d01);
    const C9: f32 = f32::from_bits(0x3638_ef1d);
    const C11: f32 = f32::from_bits(0xb2d7_322b);
    const C13: f32 = f32::from_bits(0x2f30_9231);

    let inv_pi_v = f32x4::splat(simd, INV_PI);
    let neg_pi1_v = f32x4::splat(simd, -PI1);
    let neg_pi2_v = f32x4::splat(simd, -PI2);
    let neg_pi3_v = f32x4::splat(simd, -PI3);
    let c3_v = f32x4::splat(simd, C3);
    let c5_v = f32x4::splat(simd, C5);
    let c7_v = f32x4::splat(simd, C7);
    let c9_v = f32x4::splat(simd, C9);
    let c11_v = f32x4::splat(simd, C11);
    let c13_v = f32x4::splat(simd, C13);

    let (row_q, row_rem) = row.as_chunks_mut::<4>();
    let (a_q, a_rem) = a.as_chunks::<4>();
    let (ib_q, ib_rem) = inv_b.as_chunks::<4>();
    for ((rq, aq), ibq) in row_q.iter_mut().zip(a_q).zip(ib_q) {
        let x = f32x4::from_slice(simd, rq);
        let z = f32x4::from_slice(simd, aq) * x;
        let z_lanes: [f32; 4] = z.into();
        if z_lanes.iter().all(|v| v.abs() <= RANGE_LIMIT) {
            let k = (z * inv_pi_v).round_ties_even();
            let mut r = neg_pi1_v.mul_add(k, z);
            r = neg_pi2_v.mul_add(k, r);
            r = neg_pi3_v.mul_add(k, r);
            let u = r * r;
            let mut p = c13_v.mul_add(u, c11_v);
            p = p.mul_add(u, c9_v);
            p = p.mul_add(u, c7_v);
            p = p.mul_add(u, c5_v);
            p = p.mul_add(u, c3_v);
            let s = (r * u).mul_add(p, r);
            let ib = f32x4::from_slice(simd, ibq);
            (s * s).mul_add(ib, x).store_slice(rq);
        } else {
            for ((slot, &a_v), &ib_v) in rq.iter_mut().zip(aq).zip(ibq) {
                let x_s = *slot;
                let s = (a_v * x_s).sin();
                *slot = (s * s).mul_add(ib_v, x_s);
            }
        }
    }
    for ((slot, &a_v), &ib_v) in row_rem.iter_mut().zip(a_rem).zip(ib_rem) {
        let x_s = *slot;
        let s = (a_v * x_s).sin();
        *slot = (s * s).mul_add(ib_v, x_s);
    }
}

#[expect(
    clippy::inline_always,
    reason = "fearless_simd dispatch! requires inline(always) to inherit target_feature attributes."
)]
#[inline(always)]
fn tanh_f32x4<S: Simd>(simd: S, z: f32x4<S>) -> f32x4<S> {
    let zero = f32x4::splat(simd, 0.0_f32);
    let half = f32x4::splat(simd, 0.5_f32);
    let one = f32x4::splat(simd, 1.0_f32);
    let pade_limit = f32x4::splat(simd, 1.25_f32);
    let nine_half = f32x4::splat(simd, 9.5_f32);

    // [7/6] Padé approximant for |z| < 1.25 (error < 2e-8; preserves -0.0).
    let z2 = z * z;
    let p_num = z * z2.mul_add(
        z2.mul_add(
            z2 + f32x4::splat(simd, 378.0_f32),
            f32x4::splat(simd, 17_325.0_f32),
        ),
        f32x4::splat(simd, 135_135.0_f32),
    );
    let p_den = z2.mul_add(
        z2.mul_add(
            z2.mul_add(
                f32x4::splat(simd, 28.0_f32),
                f32x4::splat(simd, 3_150.0_f32),
            ),
            f32x4::splat(simd, 62_370.0_f32),
        ),
        f32x4::splat(simd, 135_135.0_f32),
    );

    // For |z| >= 1.25: exp(-2 * |z|) via [4/4] Padé on w/32 and 5 squarings.
    let az = z.abs();
    let az_clamped = az.simd_ge(nine_half).select(nine_half, az);
    let u = az_clamped * f32x4::splat(simd, -0.062_5_f32);
    let u2 = u * u;
    let pe = u2.mul_add(
        u2.mul_add(
            f32x4::splat(simd, 0.000_595_238_1_f32),
            f32x4::splat(simd, 0.107_142_86_f32),
        ),
        one,
    );
    let uq = u * u2.mul_add(f32x4::splat(simd, 0.011_904_762_f32), half);
    let mut r = (pe + uq) / (pe - uq);
    for _ in 0..5 {
        r = r * r;
    }
    let signed_diff = z.simd_lt(zero).select(r - one, one - r);
    let sum_r = one + r;

    let is_small = az.simd_lt(pade_limit);
    let num = is_small.select(p_num, signed_diff);
    let den = is_small.select(p_den, sum_r);
    num / den
}

#[inline]
fn tanh_scalar(z: f32) -> f32 {
    let az = z.abs();
    if az < 1.25_f32 {
        let z2 = z * z;
        let p_num = z * z2.mul_add(z2.mul_add(z2 + 378.0_f32, 17_325.0_f32), 135_135.0_f32);
        let p_den = z2.mul_add(
            z2.mul_add(z2.mul_add(28.0_f32, 3_150.0_f32), 62_370.0_f32),
            135_135.0_f32,
        );
        return p_num / p_den;
    }
    let az_clamped = if az >= 9.5_f32 { 9.5_f32 } else { az };
    let u = az_clamped * -0.062_5_f32;
    let u2 = u * u;
    let pe = u2.mul_add(u2.mul_add(0.000_595_238_1_f32, 0.107_142_86_f32), 1.0_f32);
    let uq = u * u2.mul_add(0.011_904_762_f32, 0.5_f32);
    let mut r = (pe + uq) / (pe - uq);
    for _ in 0..5 {
        r = r * r;
    }
    let signed_diff = if z < 0.0_f32 {
        r - 1.0_f32
    } else {
        1.0_f32 - r
    };
    signed_diff / (1.0_f32 + r)
}

#[expect(
    clippy::inline_always,
    reason = "fearless_simd dispatch! requires inline(always) to inherit target_feature attributes."
)]
#[inline(always)]
fn gelu_tanh_f32x4<S: Simd>(simd: S, val: f32x4<S>) -> f32x4<S> {
    let k_gelu = (2.0_f32 / core::f32::consts::PI).sqrt();
    let k_gelu_v = f32x4::splat(simd, k_gelu);
    let c_gelu_v = f32x4::splat(simd, 0.044_715_f32);
    let half_v = f32x4::splat(simd, 0.5_f32);
    let one_v = f32x4::splat(simd, 1.0_f32);
    let inner = k_gelu_v * c_gelu_v.mul_add(val * val * val, val);
    half_v * val * (one_v + tanh_f32x4(simd, inner))
}

#[inline]
fn gelu_tanh_scalar(val: f32) -> f32 {
    let k_gelu = (2.0_f32 / core::f32::consts::PI).sqrt();
    let inner = k_gelu * 0.044_715_f32.mul_add(val * val * val, val);
    0.5_f32 * val * (1.0_f32 + tanh_scalar(inner))
}

#[expect(
    clippy::inline_always,
    reason = "fearless_simd dispatch! requires inline(always) to inherit target_feature attributes."
)]
#[inline(always)]
fn convnext_dw_layernorm_row_simd<S: Simd>(
    simd: S,
    ext: &[f32],
    frame: usize,
    (channels, kernel, inv_c): (usize, usize, f32),
    (dw_weight, dw_bias, norm_w, norm_b): (&[f32], &[f32], &[f32], &[f32]),
    norm_row: &mut [f32],
) {
    // 1. Depthwise causal conv across `channels` channels (4 channels per f32x4 lane)
    let (y_quads, y_rem) = norm_row.as_chunks_mut::<4>();
    let (b_quads, b_rem) = dw_bias.as_chunks::<4>();
    for (q_idx, (yq, bq)) in y_quads.iter_mut().zip(b_quads.iter()).enumerate() {
        let ch_base = q_idx * 4;
        let mut acc = f32x4::from_slice(simd, bq);
        let w0 = &dw_weight[ch_base * kernel..(ch_base + 1) * kernel];
        let w1 = &dw_weight[(ch_base + 1) * kernel..(ch_base + 2) * kernel];
        let w2 = &dw_weight[(ch_base + 2) * kernel..(ch_base + 3) * kernel];
        let w3 = &dw_weight[(ch_base + 3) * kernel..(ch_base + 4) * kernel];
        for ki in 0..kernel {
            let ext_off = (frame + ki) * channels + ch_base;
            let xv = f32x4::from_slice(simd, &ext[ext_off..ext_off + 4]);
            let wv = f32x4::from_slice(simd, &[w0[ki], w1[ki], w2[ki], w3[ki]]);
            acc = wv.mul_add(xv, acc);
        }
        acc.store_slice(yq);
    }
    let rem_base = y_quads.len() * 4;
    for (r_idx, (ys, &bias)) in y_rem.iter_mut().zip(b_rem.iter()).enumerate() {
        let ch = rem_base + r_idx;
        let mut acc = bias;
        let w_row = &dw_weight[ch * kernel..(ch + 1) * kernel];
        for ki in 0..kernel {
            acc = w_row[ki].mul_add(ext[(frame + ki) * channels + ch], acc);
        }
        *ys = acc;
    }

    // 2. LayerNorm across `channels` channels
    let mean = norm_row.iter().copied().sum::<f32>() * inv_c;
    let mean_v = f32x4::splat(simd, mean);
    let mut var_acc = f32x4::splat(simd, 0.0);
    let (y_chunks, y_tail) = norm_row.as_chunks::<4>();
    for yc in y_chunks {
        let diff = f32x4::from_slice(simd, yc) - mean_v;
        var_acc = diff.mul_add(diff, var_acc);
    }
    let mut var_sum = reduce_f32x4(var_acc);
    for &val in y_tail {
        let diff = val - mean;
        var_sum = diff.mul_add(diff, var_sum);
    }
    let inv_std = 1.0_f32 / var_sum.mul_add(inv_c, 1e-6_f32).sqrt();
    let inv_std_v = f32x4::splat(simd, inv_std);
    let (y_quads, y_rem) = norm_row.as_chunks_mut::<4>();
    let (weight_quads, weight_rem) = norm_w.as_chunks::<4>();
    let (bias_quads, bias_rem) = norm_b.as_chunks::<4>();
    for ((yq, wq), bq) in y_quads
        .iter_mut()
        .zip(weight_quads.iter())
        .zip(bias_quads.iter())
    {
        let yv = (f32x4::from_slice(simd, yq) - mean_v) * inv_std_v;
        yv.mul_add(f32x4::from_slice(simd, wq), f32x4::from_slice(simd, bq))
            .store_slice(yq);
    }
    for ((ys, &nw), &nb) in y_rem.iter_mut().zip(weight_rem.iter()).zip(bias_rem.iter()) {
        *ys = ((*ys - mean) * inv_std).mul_add(nw, nb);
    }
}

#[expect(
    clippy::inline_always,
    reason = "fearless_simd dispatch! requires inline(always) to inherit target_feature attributes."
)]
#[inline(always)]
fn convnext_pwconv1_gelu_time_first_simd<S: Simd>(
    simd: S,
    w_rows: &[f32],
    b_rows: &[f32],
    norm_buf: &[f32],
    (first_frame, num_frames, in_c): (usize, usize, usize),
    mid_out: &mut [f32],
) {
    let num_rows = b_rows.len();
    let quads = num_rows / 4;
    let geom = (in_c, 1_usize, 1_usize);

    for q_idx in 0..quads {
        let oc = q_idx * 4;
        let w_quad = &w_rows[oc * in_c..(oc + 4) * in_c];
        let inits = [b_rows[oc], b_rows[oc + 1], b_rows[oc + 2], b_rows[oc + 3]];
        let mut f = 0_usize;
        while f + 4 <= num_frames {
            let ff = [
                first_frame + f,
                first_frame + f + 1,
                first_frame + f + 2,
                first_frame + f + 3,
            ];
            let dots = dot_4rows_4frames_k_f32_simd(simd, w_quad, norm_buf, ff, geom, inits);
            for (df, dot_row) in dots.iter().enumerate() {
                let off = (f + df) * num_rows + oc;
                gelu_tanh_f32x4(simd, f32x4::from_slice(simd, dot_row))
                    .store_slice(&mut mid_out[off..off + 4]);
            }
            f += 4;
        }
        while f < num_frames {
            let dots =
                dot_4rows_1frame_k_f32_simd(simd, w_quad, norm_buf, first_frame + f, geom, inits);
            let off = f * num_rows + oc;
            gelu_tanh_f32x4(simd, f32x4::from_slice(simd, &dots))
                .store_slice(&mut mid_out[off..off + 4]);
            f += 1;
        }
    }

    for oc in (quads * 4)..num_rows {
        let w_row = &w_rows[oc * in_c..(oc + 1) * in_c];
        let init = b_rows[oc];
        let mut f = 0_usize;
        while f + 4 <= num_frames {
            let ff = [
                first_frame + f,
                first_frame + f + 1,
                first_frame + f + 2,
                first_frame + f + 3,
            ];
            let dots = dot_1row_4frames_k_f32_simd(simd, w_row, norm_buf, ff, geom, init);
            let gelus: [f32; 4] = gelu_tanh_f32x4(simd, f32x4::from_slice(simd, &dots)).into();
            for (df, &val) in gelus.iter().enumerate() {
                mid_out[(f + df) * num_rows + oc] = val;
            }
            f += 4;
        }
        while f < num_frames {
            mid_out[f * num_rows + oc] = gelu_tanh_scalar(dot_1row_1frame_k_f32_simd(
                simd,
                w_row,
                norm_buf,
                first_frame + f,
                geom,
                init,
            ));
            f += 1;
        }
    }
}

#[expect(
    clippy::inline_always,
    reason = "fearless_simd dispatch! requires inline(always) to inherit target_feature attributes."
)]
#[inline(always)]
fn convnext_pwconv1_gelu_channel_first_simd<S: Simd>(
    simd: S,
    w_rows: &[f32],
    b_rows: &[f32],
    norm_buf: &[f32],
    (frames, in_c): (usize, usize),
    transposed_rows: &mut [f32],
) {
    let num_rows = b_rows.len();
    let quads = num_rows / 4;
    let geom = (in_c, 1_usize, 1_usize);

    for q_idx in 0..quads {
        let oc = q_idx * 4;
        let w_quad = &w_rows[oc * in_c..(oc + 4) * in_c];
        let inits = [b_rows[oc], b_rows[oc + 1], b_rows[oc + 2], b_rows[oc + 3]];
        let dst_quad = &mut transposed_rows[oc * frames..(oc + 4) * frames];
        let (r0_r1, r2_r3) = dst_quad.split_at_mut(2 * frames);
        let (r0, r1) = r0_r1.split_at_mut(frames);
        let (r2, r3) = r2_r3.split_at_mut(frames);
        let mut f = 0_usize;
        while f + 4 <= frames {
            let ff = [f, f + 1, f + 2, f + 3];
            let dots = dot_4rows_4frames_k_f32_simd(simd, w_quad, norm_buf, ff, geom, inits);
            let rows_r = [
                f32x4::from_slice(simd, &[dots[0][0], dots[1][0], dots[2][0], dots[3][0]]),
                f32x4::from_slice(simd, &[dots[0][1], dots[1][1], dots[2][1], dots[3][1]]),
                f32x4::from_slice(simd, &[dots[0][2], dots[1][2], dots[2][2], dots[3][2]]),
                f32x4::from_slice(simd, &[dots[0][3], dots[1][3], dots[2][3], dots[3][3]]),
            ];
            gelu_tanh_f32x4(simd, rows_r[0]).store_slice(&mut r0[f..f + 4]);
            gelu_tanh_f32x4(simd, rows_r[1]).store_slice(&mut r1[f..f + 4]);
            gelu_tanh_f32x4(simd, rows_r[2]).store_slice(&mut r2[f..f + 4]);
            gelu_tanh_f32x4(simd, rows_r[3]).store_slice(&mut r3[f..f + 4]);
            f += 4;
        }
        while f < frames {
            let dots = dot_4rows_1frame_k_f32_simd(simd, w_quad, norm_buf, f, geom, inits);
            let gelus: [f32; 4] = gelu_tanh_f32x4(simd, f32x4::from_slice(simd, &dots)).into();
            r0[f] = gelus[0];
            r1[f] = gelus[1];
            r2[f] = gelus[2];
            r3[f] = gelus[3];
            f += 1;
        }
    }

    for oc in (quads * 4)..num_rows {
        let w_row = &w_rows[oc * in_c..(oc + 1) * in_c];
        let init = b_rows[oc];
        let dst_row = &mut transposed_rows[oc * frames..(oc + 1) * frames];
        let mut f = 0_usize;
        while f + 4 <= frames {
            let ff = [f, f + 1, f + 2, f + 3];
            let dots = dot_1row_4frames_k_f32_simd(simd, w_row, norm_buf, ff, geom, init);
            gelu_tanh_f32x4(simd, f32x4::from_slice(simd, &dots))
                .store_slice(&mut dst_row[f..f + 4]);
            f += 4;
        }
        while f < frames {
            dst_row[f] = gelu_tanh_scalar(dot_1row_1frame_k_f32_simd(
                simd, w_row, norm_buf, f, geom, init,
            ));
            f += 1;
        }
    }
}

#[expect(
    clippy::inline_always,
    reason = "fearless_simd dispatch! requires inline(always) to inherit target_feature attributes."
)]
#[inline(always)]
fn convnext_pwconv2_res_time_first_simd<S: Simd>(
    simd: S,
    (w_rows, b_rows, gamma_rows): (&[f32], &[f32], &[f32]),
    (mid_buf, residual_in): (&[f32], &[f32]),
    (first_frame, num_frames, mid_c): (usize, usize, usize),
    out: &mut [f32],
) {
    let num_rows = b_rows.len();
    let quads = num_rows / 4;
    let geom = (mid_c, 1_usize, 1_usize);

    for q_idx in 0..quads {
        let oc = q_idx * 4;
        let w_quad = &w_rows[oc * mid_c..(oc + 4) * mid_c];
        let inits = [b_rows[oc], b_rows[oc + 1], b_rows[oc + 2], b_rows[oc + 3]];
        let gv = f32x4::from_slice(simd, &gamma_rows[oc..oc + 4]);
        let mut f = 0_usize;
        while f + 4 <= num_frames {
            let ff = [
                first_frame + f,
                first_frame + f + 1,
                first_frame + f + 2,
                first_frame + f + 3,
            ];
            let dots = dot_4rows_4frames_k_f32_simd(simd, w_quad, mid_buf, ff, geom, inits);
            for (df, dot_row) in dots.iter().enumerate() {
                let off = (f + df) * num_rows + oc;
                let res_off = (first_frame + f + df) * num_rows + oc;
                let dv = f32x4::from_slice(simd, dot_row);
                let rv = f32x4::from_slice(simd, &residual_in[res_off..res_off + 4]);
                dv.mul_add(gv, rv).store_slice(&mut out[off..off + 4]);
            }
            f += 4;
        }
        while f < num_frames {
            let dots =
                dot_4rows_1frame_k_f32_simd(simd, w_quad, mid_buf, first_frame + f, geom, inits);
            let off = f * num_rows + oc;
            let res_off = (first_frame + f) * num_rows + oc;
            let dv = f32x4::from_slice(simd, &dots);
            let rv = f32x4::from_slice(simd, &residual_in[res_off..res_off + 4]);
            dv.mul_add(gv, rv).store_slice(&mut out[off..off + 4]);
            f += 1;
        }
    }

    for oc in (quads * 4)..num_rows {
        let w_row = &w_rows[oc * mid_c..(oc + 1) * mid_c];
        let init = b_rows[oc];
        let g = gamma_rows[oc];
        let mut f = 0_usize;
        while f + 4 <= num_frames {
            let ff = [
                first_frame + f,
                first_frame + f + 1,
                first_frame + f + 2,
                first_frame + f + 3,
            ];
            let dots = dot_1row_4frames_k_f32_simd(simd, w_row, mid_buf, ff, geom, init);
            for (df, &val) in dots.iter().enumerate() {
                let off = (f + df) * num_rows + oc;
                let res_off = (first_frame + f + df) * num_rows + oc;
                out[off] = val.mul_add(g, residual_in[res_off]);
            }
            f += 4;
        }
        while f < num_frames {
            let off = f * num_rows + oc;
            let res_off = (first_frame + f) * num_rows + oc;
            out[off] =
                dot_1row_1frame_k_f32_simd(simd, w_row, mid_buf, first_frame + f, geom, init)
                    .mul_add(g, residual_in[res_off]);
            f += 1;
        }
    }
}

#[expect(
    clippy::inline_always,
    reason = "fearless_simd dispatch! requires inline(always) to inherit target_feature attributes."
)]
#[inline(always)]
fn convnext_pwconv2_res_channel_first_simd<S: Simd>(
    simd: S,
    (w_rows, b_rows, gamma_rows): (&[f32], &[f32], &[f32]),
    (mid_buf, residual_in): (&[f32], &[f32]),
    (frames, mid_c, total_c, oc_off): (usize, usize, usize, usize),
    transposed_rows: &mut [f32],
) {
    let num_rows = b_rows.len();
    let quads = num_rows / 4;
    let geom = (mid_c, 1_usize, 1_usize);

    for q_idx in 0..quads {
        let oc = q_idx * 4;
        let oc_abs = oc_off + oc;
        let w_quad = &w_rows[oc * mid_c..(oc + 4) * mid_c];
        let inits = [b_rows[oc], b_rows[oc + 1], b_rows[oc + 2], b_rows[oc + 3]];
        let g = [
            gamma_rows[oc],
            gamma_rows[oc + 1],
            gamma_rows[oc + 2],
            gamma_rows[oc + 3],
        ];
        let dst_quad = &mut transposed_rows[oc * frames..(oc + 4) * frames];
        let (r0_r1, r2_r3) = dst_quad.split_at_mut(2 * frames);
        let (r0, r1) = r0_r1.split_at_mut(frames);
        let (r2, r3) = r2_r3.split_at_mut(frames);
        let mut f = 0_usize;
        while f + 4 <= frames {
            let ff = [f, f + 1, f + 2, f + 3];
            let dots = dot_4rows_4frames_k_f32_simd(simd, w_quad, mid_buf, ff, geom, inits);
            for (df, dot_row) in dots.iter().enumerate() {
                let res_off = (f + df) * total_c + oc_abs;
                r0[f + df] = dot_row[0].mul_add(g[0], residual_in[res_off]);
                r1[f + df] = dot_row[1].mul_add(g[1], residual_in[res_off + 1]);
                r2[f + df] = dot_row[2].mul_add(g[2], residual_in[res_off + 2]);
                r3[f + df] = dot_row[3].mul_add(g[3], residual_in[res_off + 3]);
            }
            f += 4;
        }
        while f < frames {
            let dots = dot_4rows_1frame_k_f32_simd(simd, w_quad, mid_buf, f, geom, inits);
            let res_off = f * total_c + oc_abs;
            r0[f] = dots[0].mul_add(g[0], residual_in[res_off]);
            r1[f] = dots[1].mul_add(g[1], residual_in[res_off + 1]);
            r2[f] = dots[2].mul_add(g[2], residual_in[res_off + 2]);
            r3[f] = dots[3].mul_add(g[3], residual_in[res_off + 3]);
            f += 1;
        }
    }

    for oc in (quads * 4)..num_rows {
        let oc_abs = oc_off + oc;
        let w_row = &w_rows[oc * mid_c..(oc + 1) * mid_c];
        let init = b_rows[oc];
        let g = gamma_rows[oc];
        let dst_row = &mut transposed_rows[oc * frames..(oc + 1) * frames];
        let mut f = 0_usize;
        while f + 4 <= frames {
            let ff = [f, f + 1, f + 2, f + 3];
            let dots = dot_1row_4frames_k_f32_simd(simd, w_row, mid_buf, ff, geom, init);
            for (df, &val) in dots.iter().enumerate() {
                dst_row[f + df] = val.mul_add(g, residual_in[(f + df) * total_c + oc_abs]);
            }
            f += 4;
        }
        while f < frames {
            dst_row[f] = dot_1row_1frame_k_f32_simd(simd, w_row, mid_buf, f, geom, init)
                .mul_add(g, residual_in[f * total_c + oc_abs]);
            f += 1;
        }
    }
}

#[expect(
    clippy::inline_always,
    reason = "fearless_simd dispatch! requires inline(always) to inherit target_feature attributes."
)]
#[inline(always)]
fn rvq_argmin_simd<S: Simd>(
    simd: S,
    cb: &[f32],
    sq: &[f32],
    r: &[f32],
    codebook_size: usize,
    d: usize,
) -> Option<usize> {
    let mut best_e = 0_usize;
    let mut best_sc = f32::INFINITY;
    let quads = codebook_size / 4;
    for q_idx in 0..quads {
        let e_base = q_idx * 4;
        let w_base = e_base * d;
        let dots = dot_four_rows_f32_simd(
            simd,
            &cb[w_base..w_base + d],
            &cb[w_base + d..w_base + 2 * d],
            &cb[w_base + 2 * d..w_base + 3 * d],
            &cb[w_base + 3 * d..w_base + 4 * d],
            r,
            [0.0_f32; 4],
        );
        for (k, &dot) in dots.iter().enumerate() {
            let e = e_base + k;
            let sc = (-2.0_f32).mul_add(dot, sq[e]);
            if !sc.is_finite() {
                return None;
            }
            if sc < best_sc {
                best_sc = sc;
                best_e = e;
            }
        }
    }
    for e in (quads * 4)..codebook_size {
        let c = &cb[e * d..(e + 1) * d];
        let dot = dot_f32_simd(simd, r, c, 0.0_f32);
        let sc = (-2.0_f32).mul_add(dot, sq[e]);
        if !sc.is_finite() {
            return None;
        }
        if sc < best_sc {
            best_sc = sc;
            best_e = e;
        }
    }
    Some(best_e)
}

/// Number of RVQ codebooks (`1` semantic + `15` acoustic).
pub const TOKENIZER_NUM_CODEBOOKS: usize = 16;
/// Samples per 12.5 Hz codec frame at 24 kHz (`1920`).
pub const TOKENIZER_HOP_24K: usize = 1920;
/// Samples per 12.5 Hz codec frame at 48 kHz (`3840`).
pub const TOKENIZER_HOP_48K: usize = 3840;
/// Number of LRU slots for primed ICL reference streaming states (`8`).
pub const CODEC_SNAP_SLOTS: usize = 8;
/// Default sliding window for the 8-layer codec pre-transformer (`72` frames).
pub const CODEC_SLIDING_WINDOW: usize = 72;

/// FNV-1a 64-bit content hash over `[K, T]` reference codes with `T` folded in,
/// matching `pipeline_codec_ref_key` in `qwentts.cpp`.
#[must_use]
pub fn codec_ref_key(codes: &[i32], num_codebooks: usize, frames: usize) -> u64 {
    const OFFSET_BASIS: u64 = 1_469_598_103_934_665_603;
    const PRIME: u64 = 1_099_511_628_211;
    let count = num_codebooks.saturating_mul(frames).min(codes.len());
    let mut hash = OFFSET_BASIS;
    for &code in &codes[..count] {
        for byte in code.to_le_bytes() {
            hash = (hash ^ u64::from(byte)).wrapping_mul(PRIME);
        }
    }
    let t_u64 = u64::try_from(frames).unwrap_or(0);
    (hash ^ t_u64).wrapping_mul(PRIME)
}

/// Decimate a 48 kHz mono waveform 2:1 to 24 kHz (`ref_audio_24k`) for
/// `VoiceDesign` reference extraction when a 48 kHz codec is active.
#[must_use]
pub fn decimate_48k_to_24k(samples_48k: &[f32]) -> Vec<f32> {
    let pairs = samples_48k.len() / 2;
    let mut out = Vec::with_capacity(pairs);
    for i in 0..pairs {
        out.push(f32::midpoint(samples_48k[2 * i], samples_48k[2 * i + 1]));
    }
    out
}

/// Padding mode for causal 1D convolutions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PadMode {
    /// Zero padding on left/right boundaries (`SEANet` and DAC decoder).
    Constant,
    /// Edge replication padding (Mimi encoder `k = 4, s = 2` downsample).
    Replicate,
}

fn pack_conv_okic(weight: &[f32], out_c: usize, in_c: usize, kernel: usize) -> Vec<f32> {
    let mut okic = vec![0.0_f32; weight.len()];
    let stride_oc = in_c * kernel;
    for oc in 0..out_c {
        let src_oc = &weight[oc * stride_oc..(oc + 1) * stride_oc];
        let dst_oc = &mut okic[oc * stride_oc..(oc + 1) * stride_oc];
        for ic in 0..in_c {
            let src_ic = &src_oc[ic * kernel..(ic + 1) * kernel];
            for (k, &v) in src_ic.iter().enumerate() {
                dst_oc[k * in_c + ic] = v;
            }
        }
    }
    okic
}

/// Causal 1D convolution layer (`[out_c, in_c, kernel]` row-major weights,
/// with optional load-time `[out_c, kernel, in_c]` layout for `kernel > 1`).
#[derive(Clone, Debug)]
pub struct CausalConv1d {
    /// Output channels.
    pub out_c: usize,
    /// Input channels.
    pub in_c: usize,
    /// Kernel width.
    pub kernel: usize,
    /// Dilation factor (`>= 1`).
    pub dilation: usize,
    /// Stride (`>= 1`).
    pub stride: usize,
    /// Padding mode.
    pub pad_mode: PadMode,
    /// Weights `[out_c * in_c * kernel]`.
    pub weight: Vec<f32>,
    /// Pre-repacked `[out_c, kernel, in_c]` weights when `kernel > 1` (`None` for `kernel == 1` or raw struct literals).
    pub weight_okic: Option<Vec<f32>>,
    /// Optional bias `[out_c]`.
    pub bias: Option<Vec<f32>>,
}

impl CausalConv1d {
    /// Construct a `CausalConv1d` from `[out_c, in_c, kernel]` weights,
    /// repacking into `[out_c, kernel, in_c]` order at load time when `kernel > 1`.
    #[expect(
        clippy::too_many_arguments,
        reason = "Explicit convolution geometry and owned weights avoid an extra configuration struct."
    )]
    pub fn from_oc_ic_k(
        out_c: usize,
        in_c: usize,
        kernel: usize,
        dilation: usize,
        stride: usize,
        pad_mode: PadMode,
        weight: Vec<f32>,
        bias: Option<Vec<f32>>,
    ) -> io::Result<Self> {
        let expected = out_c
            .checked_mul(in_c)
            .and_then(|v| v.checked_mul(kernel))
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "CausalConv1d weight dimensions overflow.",
                )
            })?;
        if out_c == 0
            || in_c == 0
            || kernel == 0
            || dilation == 0
            || stride == 0
            || weight.len() != expected
            || bias.as_ref().is_some_and(|b| b.len() != out_c)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Invalid CausalConv1d dimensions.",
            ));
        }
        let weight_okic = (kernel > 1).then(|| pack_conv_okic(&weight, out_c, in_c, kernel));
        Ok(Self {
            out_c,
            in_c,
            kernel,
            dilation,
            stride,
            pad_mode,
            weight,
            weight_okic,
            bias,
        })
    }

    /// Effective dilated kernel span `(kernel - 1) * dilation + 1`.
    #[must_use]
    pub const fn kernel_eff(&self) -> usize {
        (self.kernel - 1) * self.dilation + 1
    }

    /// Left-context frame count `(kernel - 1) * dilation` required for `stride == 1` streaming.
    #[must_use]
    pub const fn stream_context_frames(&self) -> usize {
        (self.kernel - 1) * self.dilation
    }

    /// Offline causal forward pass on `[frames, in_c]` time-first input.
    pub fn forward(&self, input: &[f32], frames: usize) -> io::Result<(Vec<f32>, usize)> {
        self.forward_parallel(input, frames, 1)
    }

    /// Offline forward pass using bounded workers for independent output frames.
    pub fn forward_parallel(
        &self,
        input: &[f32],
        frames: usize,
        threads: usize,
    ) -> io::Result<(Vec<f32>, usize)> {
        let expected = self
            .out_c
            .checked_mul(self.in_c)
            .and_then(|v| v.checked_mul(self.kernel));
        if frames == 0
            || self.in_c == 0
            || self.out_c == 0
            || self.stride == 0
            || self.dilation == 0
            || self.kernel == 0
            || input.len() != frames * self.in_c
            || expected != Some(self.weight.len())
            || self
                .weight_okic
                .as_ref()
                .is_some_and(|w| w.len() != self.weight.len())
            || self.bias.as_ref().is_some_and(|b| b.len() != self.out_c)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Invalid CausalConv1d dimensions.",
            ));
        }
        let k_eff = self.kernel_eff();
        let pad_left = k_eff.saturating_sub(self.stride);
        let out_frames = (frames + pad_left - k_eff).div_ceil(self.stride) + 1;
        let ideal_len = (out_frames - 1) * self.stride + k_eff - pad_left;
        let extra_pad = ideal_len.saturating_sub(frames);
        if pad_left == 0 && extra_pad == 0 {
            let mut out = Vec::new();
            self.compute_frames_into(input, out_frames, self.stride, threads, &mut out);
            return Ok((out, out_frames));
        }
        let padded_len = pad_left + frames + extra_pad;

        let mut padded = vec![0.0_f32; padded_len * self.in_c];
        if self.pad_mode == PadMode::Replicate {
            for t in 0..pad_left {
                padded[t * self.in_c..(t + 1) * self.in_c].copy_from_slice(&input[..self.in_c]);
            }
        }
        padded[pad_left * self.in_c..(pad_left + frames) * self.in_c].copy_from_slice(input);
        if self.pad_mode == PadMode::Replicate && extra_pad > 0 {
            let last = &input[(frames - 1) * self.in_c..frames * self.in_c];
            for t in 0..extra_pad {
                let dst = (pad_left + frames + t) * self.in_c;
                padded[dst..dst + self.in_c].copy_from_slice(last);
            }
        }

        let mut out = Vec::new();
        self.compute_frames_into(&padded, out_frames, self.stride, threads, &mut out);
        Ok((out, out_frames))
    }

    /// Stateful streaming causal forward pass (`stride == 1`) updating `state` (`[L, in_c]`) in place.
    pub fn forward_stream(
        &self,
        input: &[f32],
        frames: usize,
        state: &mut [f32],
    ) -> io::Result<Vec<f32>> {
        self.forward_stream_parallel(input, frames, state, 1)
    }

    /// Stream independent output frames on bounded workers, preserving context order.
    pub fn forward_stream_parallel(
        &self,
        input: &[f32],
        frames: usize,
        state: &mut [f32],
        threads: usize,
    ) -> io::Result<Vec<f32>> {
        let mut ext = Vec::new();
        let mut out = Vec::new();
        self.forward_stream_into(input, frames, state, threads, &mut ext, &mut out)?;
        Ok(out)
    }

    pub(crate) fn forward_stream_into(
        &self,
        input: &[f32],
        frames: usize,
        state: &mut [f32],
        threads: usize,
        ext_scratch: &mut Vec<f32>,
        out: &mut Vec<f32>,
    ) -> io::Result<()> {
        let ctx_frames = self.stream_context_frames();
        let expected = self
            .out_c
            .checked_mul(self.in_c)
            .and_then(|v| v.checked_mul(self.kernel));
        if self.stride != 1
            || self.in_c == 0
            || self.out_c == 0
            || self.kernel == 0
            || self.dilation == 0
            || frames == 0
            || input.len() != frames * self.in_c
            || state.len() != ctx_frames * self.in_c
            || expected != Some(self.weight.len())
            || self
                .weight_okic
                .as_ref()
                .is_some_and(|w| w.len() != self.weight.len())
            || self.bias.as_ref().is_some_and(|b| b.len() != self.out_c)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Invalid streaming CausalConv1d state or input.",
            ));
        }
        let ext = update_stream_context(state, input, ext_scratch);
        self.compute_frames_into(ext, frames, 1, threads, out);
        Ok(())
    }

    fn compute_frames_into(
        &self,
        ext: &[f32],
        out_frames: usize,
        stride: usize,
        threads: usize,
        out: &mut Vec<f32>,
    ) {
        const OC_PARALLEL_THRESHOLD_BYTES: usize = 1_048_576;
        const OC_PARALLEL_MAX_FRAMES: usize = 320;

        let out_len = out_frames * self.out_c;
        out.clear();
        out.resize(out_len, 0.0_f32);

        let packed_fallback;
        let weight_okic: &[f32] = if let Some(okic) = self.weight_okic.as_deref() {
            okic
        } else if self.kernel == 1 {
            &self.weight
        } else {
            packed_fallback = pack_conv_okic(&self.weight, self.out_c, self.in_c, self.kernel);
            &packed_fallback
        };

        let level = Level::new();
        let bias = self.bias.as_deref();
        let geom = (self.in_c, self.kernel, self.dilation);
        let k_in_c = self.kernel * self.in_c;
        let active_threads = threads.clamp(1, 18);
        let weight_bytes = self.weight.len().saturating_mul(size_of::<f32>());
        let work = out_frames.saturating_mul(self.weight.len());

        // Low-frame / large-weight stages (`conv_pre`, Block 0, TTFA `count = 1..4`):
        // tile output channels across `row_workers` into a bounded `[out_c, out_frames]`
        // scratch buffer so every worker loads weights once across all `out_frames`,
        // then 4x4 SIMD transpose back to `[out_frames, out_c]`.
        if active_threads > 1
            && out_frames <= OC_PARALLEL_MAX_FRAMES
            && weight_bytes >= OC_PARALLEL_THRESHOLD_BYTES
            && self.out_c > 4
        {
            if out_frames == 1 {
                row_workers::run_aligned(out, active_threads, 4, &|oc_off, out_rows| {
                    let rows = out_rows.len();
                    let w_slice = &weight_okic[oc_off * k_in_c..(oc_off + rows) * k_in_c];
                    let b_slice = bias.map(|b| &b[oc_off..oc_off + rows]);
                    dispatch!(
                        level,
                        simd => conv1d_oc_outer_time_first_simd(
                            simd,
                            w_slice,
                            b_slice,
                            ext,
                            (0, 1, stride),
                            geom,
                            out_rows,
                        )
                    );
                });
                return;
            }
            with_channel_tile_out(self.out_c * out_frames, |transposed| {
                row_workers::run_aligned(
                    transposed,
                    active_threads,
                    4 * out_frames,
                    &|elem_off, slice| {
                        let oc_off = elem_off / out_frames;
                        let rows = slice.len() / out_frames;
                        let w_slice = &weight_okic[oc_off * k_in_c..(oc_off + rows) * k_in_c];
                        let b_slice = bias.map(|b| &b[oc_off..oc_off + rows]);
                        dispatch!(
                            level,
                            simd => conv1d_oc_outer_channel_first_simd(
                                simd,
                                w_slice,
                                b_slice,
                                ext,
                                (out_frames, stride),
                                geom,
                                slice,
                            )
                        );
                    },
                );
                scatter_transposed_to_time_first(
                    transposed,
                    out_frames,
                    self.out_c,
                    active_threads,
                    out,
                );
            });
            return;
        }

        let workers = frame_workers(threads, out_frames, work);
        row_workers::run_aligned(out, workers, self.out_c, &|offset, output| {
            let first_frame = offset / self.out_c;
            let worker_frames = output.len() / self.out_c;
            dispatch!(
                level,
                simd => conv1d_oc_outer_time_first_simd(
                    simd,
                    weight_okic,
                    bias,
                    ext,
                    (first_frame, worker_frames, stride),
                    geom,
                    output,
                )
            );
        });
    }
}

fn update_stream_context<'a>(
    state: &mut [f32],
    input: &'a [f32],
    ext_buf: &'a mut Vec<f32>,
) -> &'a [f32] {
    if state.is_empty() {
        return input;
    }
    ext_buf.clear();
    ext_buf.reserve(state.len() + input.len());
    ext_buf.extend_from_slice(state);
    ext_buf.extend_from_slice(input);
    state.copy_from_slice(&ext_buf[input.len()..]);
    ext_buf
}

/// Causal 1D transposed convolution (pre-packed `[phase, oc_block(16), k_tap, ic, oc_lane]` layout).
#[derive(Clone, Debug)]
pub struct CausalTransConv1d {
    /// Input channels.
    pub in_c: usize,
    /// Output channels.
    pub out_c: usize,
    /// Kernel width (`>= stride`).
    pub kernel: usize,
    /// Upsampling stride (`>= 1`).
    pub stride: usize,
    /// Weights `[kernel * in_c * out_c]` in `[phase, oc_block(16), k_tap(descending k), ic, oc_lane]` order.
    pub weight: Vec<f32>,
    /// Optional bias `[out_c]`.
    pub bias: Option<Vec<f32>>,
}

impl CausalTransConv1d {
    /// Construct a `CausalTransConv1d` from `PyTorch`/`GGUF` `[in_c, out_c, kernel]` weights,
    /// pre-packing into `[phase, oc_block(16), k_tap(descending k = ascending t), ic, oc_lane]` order
    /// for zero-spill phase-blocked gather GEMM.
    pub fn from_ic_oc_k(
        in_c: usize,
        out_c: usize,
        kernel: usize,
        stride: usize,
        weight_ic_oc_k: &[f32],
        bias: Option<Vec<f32>>,
    ) -> io::Result<Self> {
        let expected = in_c
            .checked_mul(out_c)
            .and_then(|v| v.checked_mul(kernel))
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "CausalTransConv1d weight dimensions overflow.",
                )
            })?;
        if in_c == 0
            || out_c == 0
            || kernel == 0
            || stride == 0
            || kernel < stride
            || weight_ic_oc_k.len() != expected
            || bias.as_ref().is_some_and(|b| b.len() != out_c)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Invalid CausalTransConv1d dimensions.",
            ));
        }
        let mut weight = vec![0.0_f32; expected];
        let oc_blocks = out_c.div_ceil(16);
        for phase in 0..stride {
            let n_taps = trans_conv_phase_taps(kernel, stride, phase);
            let phase_base = trans_conv_phase_tap_offset(kernel, stride, phase) * in_c * out_c;
            for oc_blk in 0..oc_blocks {
                let oc_base = oc_blk * 16;
                let lane_w = 16.min(out_c - oc_base);
                let blk_base = phase_base + oc_base * n_taps * in_c;
                for tap_idx in 0..n_taps {
                    let m = n_taps - 1 - tap_idx;
                    let k = m * stride + phase;
                    let tap_dst = blk_base + tap_idx * in_c * lane_w;
                    for ic in 0..in_c {
                        let ic_dst = tap_dst + ic * lane_w;
                        for lane in 0..lane_w {
                            let oc = oc_base + lane;
                            weight[ic_dst + lane] = weight_ic_oc_k[(ic * out_c + oc) * kernel + k];
                        }
                    }
                }
            }
        }
        Ok(Self {
            in_c,
            out_c,
            kernel,
            stride,
            weight,
            bias,
        })
    }

    /// Overlap carry frames `kernel - stride` retained across streaming chunks.
    #[must_use]
    pub const fn carry_frames(&self) -> usize {
        self.kernel.saturating_sub(self.stride)
    }

    /// Offline causal transposed convolution on `[frames, in_c]` time-first input,
    /// returning `[frames * stride, out_c]`.
    pub fn forward(&self, input: &[f32], frames: usize) -> io::Result<(Vec<f32>, usize)> {
        self.forward_parallel(input, frames, 1)
    }

    /// Offline transposed convolution using bounded workers over output windows.
    pub fn forward_parallel(
        &self,
        input: &[f32],
        frames: usize,
        threads: usize,
    ) -> io::Result<(Vec<f32>, usize)> {
        let mut carry = Vec::new();
        let mut out = Vec::new();
        let out_frames = self.forward_into(input, frames, threads, &mut carry, &mut out)?;
        Ok((out, out_frames))
    }

    pub(crate) fn forward_into(
        &self,
        input: &[f32],
        frames: usize,
        threads: usize,
        carry_scratch: &mut Vec<f32>,
        out: &mut Vec<f32>,
    ) -> io::Result<usize> {
        let carry_len = self.carry_frames() * self.out_c;
        carry_scratch.clear();
        carry_scratch.resize(carry_len, 0.0_f32);
        self.forward_stream_into(input, frames, &mut carry_scratch[..carry_len], threads, out)?;
        Ok(frames * self.stride)
    }

    /// Stateful streaming causal transposed convolution updating `carry` (`[kernel - stride, out_c]`).
    pub fn forward_stream(
        &self,
        input: &[f32],
        frames: usize,
        carry: &mut [f32],
    ) -> io::Result<Vec<f32>> {
        self.forward_stream_parallel(input, frames, carry, 1)
    }

    /// Stream disjoint output windows while retaining source-frame and FMA order.
    pub fn forward_stream_parallel(
        &self,
        input: &[f32],
        frames: usize,
        carry: &mut [f32],
        threads: usize,
    ) -> io::Result<Vec<f32>> {
        let mut raw = Vec::new();
        self.forward_stream_into(input, frames, carry, threads, &mut raw)?;
        Ok(raw)
    }

    pub(crate) fn forward_stream_into(
        &self,
        input: &[f32],
        frames: usize,
        carry: &mut [f32],
        threads: usize,
        raw: &mut Vec<f32>,
    ) -> io::Result<()> {
        const PHASE_OC_SCRATCH_MAX_ELEMS: usize = 524_288;
        const PHASE_OC_THRESHOLD_BYTES: usize = 1_048_576;

        let trim = self.carry_frames();
        if frames == 0
            || self.in_c == 0
            || self.out_c == 0
            || self.stride == 0
            || self.kernel < self.stride
            || input.len() != frames * self.in_c
            || carry.len() != trim * self.out_c
            || self.weight.len() != self.in_c * self.out_c * self.kernel
            || self.bias.as_ref().is_some_and(|b| b.len() != self.out_c)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Invalid CausalTransConv1d dimensions.",
            ));
        }
        let emit = frames * self.stride;
        let raw_frames = emit + trim;
        raw.clear();
        raw.resize(raw_frames * self.out_c, 0.0_f32);

        let level = Level::new();
        let bias_ref = self.bias.as_deref();
        let active_threads = threads.clamp(1, 18);
        let oc_blocks = self.out_c.div_ceil(16);
        let total_tiles = self.stride * oc_blocks;
        let weight_bytes = self.weight.len().saturating_mul(size_of::<f32>());
        let work = frames.saturating_mul(self.weight.len());

        // Low-frame / large-weight stages (Block 0, Upsample, Block 1-2 low count):
        // parallelize across `(phase, oc_block)` into a phase-blocked scratch buffer so
        // each core loads each weight tile once across all frames and scales to 100% of cores.
        if active_threads > 1
            && total_tiles >= 2
            && work >= 65_536
            && (frames < active_threads * 4
                || (weight_bytes >= PHASE_OC_THRESHOLD_BYTES
                    && raw_frames * self.out_c <= PHASE_OC_SCRATCH_MAX_ELEMS))
        {
            let q_max = frames + self.kernel.div_ceil(self.stride) - 1;
            let tile_elems = q_max * 16;
            with_channel_tile_out(total_tiles * tile_elems, |scratch| {
                row_workers::run_aligned(
                    scratch,
                    active_threads,
                    tile_elems,
                    &|elem_off, slice| {
                        let start_tile = elem_off / tile_elems;
                        let num_tiles = slice.len() / tile_elems;
                        for local_t in 0..num_tiles {
                            let tile_idx = start_tile + local_t;
                            let phase = tile_idx / oc_blocks;
                            let oc_blk = tile_idx % oc_blocks;
                            let n_taps = trans_conv_phase_taps(self.kernel, self.stride, phase);
                            let q_count = frames + n_taps - 1;
                            let oc_base = oc_blk * 16;
                            let lane_w = 16.min(self.out_c - oc_base);
                            let phase_base =
                                trans_conv_phase_tap_offset(self.kernel, self.stride, phase)
                                    * self.in_c
                                    * self.out_c;
                            let blk_base = phase_base + oc_base * n_taps * self.in_c;
                            let w_tile =
                                &self.weight[blk_base..blk_base + n_taps * self.in_c * lane_w];
                            let bias_blk = bias_ref.map(|b| &b[oc_base..oc_base + lane_w]);
                            let tile_out =
                                &mut slice[local_t * tile_elems..(local_t + 1) * tile_elems];
                            dispatch!(
                                level,
                                simd => trans_conv_run_q_range_simd(
                                    simd,
                                    input,
                                    carry,
                                    w_tile,
                                    bias_blk,
                                    (phase, self.stride, frames, n_taps, self.in_c, self.out_c, oc_base, lane_w),
                                    (0, q_count),
                                    (tile_out, 0, 1, 16, 0),
                                )
                            );
                        }
                    },
                );
                let scatter_workers = if raw.len() >= 8_192 {
                    active_threads
                } else {
                    1
                };
                row_workers::run_aligned(raw, scatter_workers, self.out_c, &|offset, slice| {
                    let first_frame = offset / self.out_c;
                    dispatch!(
                        level,
                        simd => scatter_phase_blocks_to_raw_simd(
                            simd,
                            scratch,
                            tile_elems,
                            self.stride,
                            self.out_c,
                            first_frame,
                            slice,
                        )
                    );
                });
            });
            if trim > 0 {
                carry.copy_from_slice(&raw[emit * self.out_c..]);
            }
            raw.truncate(emit * self.out_c);
            return Ok(());
        }

        // High-frame / small-weight stages (DAC Blocks 1-3 steady state):
        // split raw output along `stride * out_c` boundaries (`phase == 0` aligned) and
        // run `(q_block, phase)` directly into each worker's disjoint raw slice.
        let workers = frame_workers(threads, raw_frames, work);
        row_workers::run_aligned(raw, workers, self.stride * self.out_c, &|offset, output| {
            const Q_TILE: usize = 64;
            let first_frame = offset / self.out_c;
            let end_frame = first_frame + output.len() / self.out_c;
            let q_start = first_frame / self.stride;

            for phase in 0..self.stride {
                let n_taps = trans_conv_phase_taps(self.kernel, self.stride, phase);
                let q_count = frames + n_taps - 1;
                let q_end = end_frame
                    .saturating_sub(phase)
                    .div_ceil(self.stride)
                    .min(q_count);
                let q_begin = q_start.min(q_end);
                if q_begin >= q_end {
                    continue;
                }
                let phase_base = trans_conv_phase_tap_offset(self.kernel, self.stride, phase)
                    * self.in_c
                    * self.out_c;
                let mut q_chunk = q_begin;
                while q_chunk < q_end {
                    let q_chunk_end = (q_chunk + Q_TILE).min(q_end);
                    for oc_blk in 0..oc_blocks {
                        let oc_base = oc_blk * 16;
                        let lane_w = 16.min(self.out_c - oc_base);
                        let blk_base = phase_base + oc_base * n_taps * self.in_c;
                        let w_tile = &self.weight[blk_base..blk_base + n_taps * self.in_c * lane_w];
                        let bias_blk = bias_ref.map(|b| &b[oc_base..oc_base + lane_w]);
                        dispatch!(
                            level,
                            simd => trans_conv_run_q_range_simd(
                                simd,
                                input,
                                carry,
                                w_tile,
                                bias_blk,
                                (phase, self.stride, frames, n_taps, self.in_c, self.out_c, oc_base, lane_w),
                                (q_chunk, q_chunk_end),
                                (output, first_frame, self.stride, self.out_c, oc_base),
                            )
                        );
                    }
                    q_chunk = q_chunk_end;
                }
            }
        });

        if trim > 0 {
            carry.copy_from_slice(&raw[emit * self.out_c..]);
        }

        raw.truncate(emit * self.out_c);
        Ok(())
    }
}

/// Precomputed `SnakeBeta` activation parameters (`a = exp(alpha)`, `inv_b = 1 / (exp(beta) + 1e-9)`).
#[derive(Clone, Debug)]
pub struct SnakeBeta {
    /// Pre-folded `exp(alpha)` per channel `[channels]`.
    pub a: Vec<f32>,
    /// Pre-folded `1.0 / (exp(beta) + 1e-9)` per channel `[channels]`.
    pub inv_b: Vec<f32>,
}

impl SnakeBeta {
    /// Construct `SnakeBeta` from raw log-space `alpha` and `beta` tensors.
    pub fn from_log_params(alpha: &[f32], beta: &[f32]) -> io::Result<Self> {
        if alpha.is_empty() || alpha.len() != beta.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "SnakeBeta alpha/beta length mismatch.",
            ));
        }
        let a = alpha.iter().map(|&v| v.exp()).collect();
        let inv_b = beta
            .iter()
            .map(|&v| 1.0_f32 / (v.exp() + 1e-9_f32))
            .collect();
        Ok(Self { a, inv_b })
    }

    /// Apply `y = x + sin^2(a * x) * inv_b` in place over `[frames, channels]` time-first data.
    pub fn apply_in_place(&self, data: &mut [f32], frames: usize) -> io::Result<()> {
        self.apply_in_place_parallel(data, frames, 1)
    }

    /// Apply `y = x + sin^2(a * x) * inv_b` in place over `[frames, channels]` time-first data
    /// using up to `threads` reusable frame workers.
    pub fn apply_in_place_parallel(
        &self,
        data: &mut [f32],
        frames: usize,
        threads: usize,
    ) -> io::Result<()> {
        let channels = self.a.len();
        if self.inv_b.len() != channels || frames.checked_mul(channels) != Some(data.len()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "SnakeBeta input shape mismatch.",
            ));
        }
        if frames == 0 || channels == 0 {
            return Ok(());
        }
        let level = Level::new();
        let workers = if frames <= 1 || data.len() < 4096 {
            1
        } else {
            threads.clamp(1, 18)
        };
        row_workers::run_aligned(data, workers, channels, &|_, slice| {
            for row in slice.chunks_exact_mut(channels) {
                dispatch!(level, simd => snake_beta_row_simd(simd, row, &self.a, &self.inv_b));
            }
        });
        Ok(())
    }
}

/// Dilated DAC residual unit (`act1 -> conv1(k=7, dilation) -> act2 -> conv2(k=1) + skip`).
#[derive(Clone, Debug)]
pub struct DacResUnit {
    /// First `SnakeBeta` activation.
    pub act1: SnakeBeta,
    /// Dilated `k = 7` causal convolution.
    pub conv1: CausalConv1d,
    /// Second `SnakeBeta` activation.
    pub act2: SnakeBeta,
    /// Pointwise `k = 1` convolution.
    pub conv2: CausalConv1d,
}

impl DacResUnit {
    /// Execute one residual unit with streaming `conv1_state`.
    pub fn forward_stream(
        &self,
        input: &[f32],
        frames: usize,
        conv1_state: &mut [f32],
    ) -> io::Result<Vec<f32>> {
        self.forward_stream_parallel(input, frames, conv1_state, 1)
    }

    /// Execute a residual unit using bounded convolution workers.
    pub fn forward_stream_parallel(
        &self,
        input: &[f32],
        frames: usize,
        conv1_state: &mut [f32],
        threads: usize,
    ) -> io::Result<Vec<f32>> {
        let mut h = Vec::new();
        h.extend_from_slice(input);
        let mut conv_ext = Vec::new();
        let mut res_buf = Vec::new();
        let mut block_buf = Vec::new();
        self.forward_stream_into(
            &mut h,
            frames,
            conv1_state,
            threads,
            &mut conv_ext,
            &mut res_buf,
            &mut block_buf,
        )?;
        Ok(h)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "Reuses caller-owned convolution and residual scratch buffers across all DAC residual units."
    )]
    fn forward_stream_into(
        &self,
        input_and_out: &mut Vec<f32>,
        frames: usize,
        conv1_state: &mut [f32],
        threads: usize,
        conv_ext: &mut Vec<f32>,
        res_buf: &mut Vec<f32>,
        block_buf: &mut Vec<f32>,
    ) -> io::Result<()> {
        res_buf.clear();
        res_buf.extend_from_slice(input_and_out);
        self.act1
            .apply_in_place_parallel(res_buf, frames, threads)?;
        self.conv1.forward_stream_into(
            res_buf,
            frames,
            conv1_state,
            threads,
            conv_ext,
            block_buf,
        )?;
        self.act2
            .apply_in_place_parallel(block_buf, frames, threads)?;
        let mut dummy_state = [];
        self.conv2.forward_stream_into(
            block_buf,
            frames,
            &mut dummy_state,
            threads,
            conv_ext,
            res_buf,
        )?;
        let level = Level::new();
        dispatch!(level, simd => add_slice_in_place_simd(simd, res_buf, input_and_out));
        std::mem::swap(input_and_out, res_buf);
        Ok(())
    }
}

/// One DAC upsampling block (`snake1 -> CausalTransConv1d -> 3 DacResUnits`).
#[derive(Clone, Debug)]
pub struct DacBlock {
    /// Pre-upsample `SnakeBeta` activation.
    pub snake1: SnakeBeta,
    /// Causal transposed convolution (`kernel = 2 * stride`).
    pub trans_conv: CausalTransConv1d,
    /// Three dilated residual units (`dilations = [1, 3, 9]`).
    pub res_units: [DacResUnit; 3],
}

impl DacBlock {
    /// Execute one DAC block, upsampling `frames` to `frames * stride`.
    pub fn forward_stream(
        &self,
        input: &[f32],
        frames: usize,
        carry: &mut [f32],
        ru_states: &mut [Vec<f32>; 3],
    ) -> io::Result<(Vec<f32>, usize)> {
        self.forward_stream_parallel(input, frames, carry, ru_states, 1)
    }

    /// Execute a DAC block with bounded independent-frame convolution workers.
    pub fn forward_stream_parallel(
        &self,
        input: &[f32],
        frames: usize,
        carry: &mut [f32],
        ru_states: &mut [Vec<f32>; 3],
        threads: usize,
    ) -> io::Result<(Vec<f32>, usize)> {
        let mut conv_ext = Vec::new();
        let mut block_buf = Vec::new();
        let mut res_buf = Vec::new();
        let mut out = Vec::new();
        let out_frames = self.forward_stream_into(
            input,
            frames,
            carry,
            ru_states,
            threads,
            &mut conv_ext,
            &mut block_buf,
            &mut res_buf,
            &mut out,
        )?;
        Ok((out, out_frames))
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "Reuses caller-owned block and residual scratch buffers across DAC upsample stages."
    )]
    fn forward_stream_into(
        &self,
        input: &[f32],
        frames: usize,
        carry: &mut [f32],
        ru_states: &mut [Vec<f32>; 3],
        threads: usize,
        conv_ext: &mut Vec<f32>,
        block_buf: &mut Vec<f32>,
        res_buf: &mut Vec<f32>,
        out: &mut Vec<f32>,
    ) -> io::Result<usize> {
        block_buf.clear();
        block_buf.extend_from_slice(input);
        self.snake1
            .apply_in_place_parallel(block_buf, frames, threads)?;
        self.trans_conv
            .forward_stream_into(block_buf, frames, carry, threads, out)?;
        let out_frames = frames * self.trans_conv.stride;
        for (ru, st) in self.res_units.iter().zip(ru_states.iter_mut()) {
            ru.forward_stream_into(out, out_frames, st, threads, conv_ext, res_buf, block_buf)?;
        }
        Ok(out_frames)
    }
}

/// `ConvNeXt` block used in the 12 Hz codec upsample stage.
#[derive(Clone, Debug)]
pub struct ConvNeXtBlock {
    /// Number of channels (`1024`).
    pub channels: usize,
    /// Depthwise causal kernel width (`7`).
    pub kernel: usize,
    /// Depthwise weights `[channels * kernel]`.
    pub dw_weight: Vec<f32>,
    /// Depthwise bias `[channels]`.
    pub dw_bias: Vec<f32>,
    /// `LayerNorm` scale `[channels]`.
    pub norm_w: Vec<f32>,
    /// `LayerNorm` bias `[channels]`.
    pub norm_b: Vec<f32>,
    /// Pointwise expansion `[4 * channels, channels]` (`out_c, in_c`).
    pub pwconv1_w: Vec<f32>,
    /// Pointwise expansion bias `[4 * channels]`.
    pub pwconv1_b: Vec<f32>,
    /// Pointwise projection `[channels, 4 * channels]` (`out_c, in_c`).
    pub pwconv2_w: Vec<f32>,
    /// Pointwise projection bias `[channels]`.
    pub pwconv2_b: Vec<f32>,
    /// `LayerScale` `[channels]`.
    pub gamma: Vec<f32>,
}

impl ConvNeXtBlock {
    /// Run the `ConvNeXt` block on `[frames, channels]` time-first input with `dw_state` (`[kernel - 1, channels]`).
    pub fn forward_stream(
        &self,
        input: &[f32],
        frames: usize,
        dw_state: &mut [f32],
    ) -> io::Result<Vec<f32>> {
        self.forward_stream_parallel(input, frames, dw_state, 1)
    }

    /// Execute independent frames on bounded workers with unchanged normalization.
    pub fn forward_stream_parallel(
        &self,
        input: &[f32],
        frames: usize,
        dw_state: &mut [f32],
        threads: usize,
    ) -> io::Result<Vec<f32>> {
        let mut ext_buf = Vec::new();
        let mut out = Vec::new();
        self.forward_stream_into(input, frames, dw_state, threads, &mut ext_buf, &mut out)?;
        Ok(out)
    }

    pub(crate) fn forward_stream_into(
        &self,
        input: &[f32],
        frames: usize,
        dw_state: &mut [f32],
        threads: usize,
        ext_buf: &mut Vec<f32>,
        out: &mut Vec<f32>,
    ) -> io::Result<()> {
        const OC_PARALLEL_THRESHOLD_BYTES: usize = 262_144;
        const OC_PARALLEL_MAX_FRAMES: usize = 320;

        let c = self.channels;
        let k = self.kernel;
        let ctx_len = k.saturating_sub(1);
        let mid_c = self.pwconv1_b.len();
        if frames == 0
            || c == 0
            || k == 0
            || mid_c == 0
            || input.len() != frames * c
            || dw_state.len() != ctx_len * c
            || self.dw_weight.len() != c * k
            || self.dw_bias.len() != c
            || self.norm_w.len() != c
            || self.norm_b.len() != c
            || self.pwconv1_w.len() != mid_c * c
            || self.pwconv2_w.len() != c * mid_c
            || self.pwconv2_b.len() != c
            || self.gamma.len() != c
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Invalid ConvNeXtBlock dimensions.",
            ));
        }
        let ext = update_stream_context(dw_state, input, ext_buf);
        let inv_c = 1.0_f32 / usize_to_f32(c);
        let level = Level::new();
        let active_threads = threads.clamp(1, 18);
        out.clear();
        out.resize(frames * c, 0.0_f32);

        with_convnext_scratch(frames * c, frames * mid_c, |norm_buf, mid_buf| {
            // Stage 1: parallel depthwise causal conv + LayerNorm across frames.
            let stage1_workers = if frames > 1 && norm_buf.len() >= 2048 {
                active_threads
            } else {
                1
            };
            row_workers::run_aligned(norm_buf, stage1_workers, c, &|offset, slice| {
                let first_frame = offset / c;
                for (local_f, norm_row) in slice.chunks_exact_mut(c).enumerate() {
                    dispatch!(
                        level,
                        simd => convnext_dw_layernorm_row_simd(
                            simd,
                            ext,
                            first_frame + local_f,
                            (c, k, inv_c),
                            (&self.dw_weight, &self.dw_bias, &self.norm_w, &self.norm_b),
                            norm_row,
                        )
                    );
                }
            });

            // Stage 2: channel-parallel `oc`-outer x `frame_quad`-inner `pwconv1` + fused SIMD GELU.
            let pw1_bytes = self.pwconv1_w.len().saturating_mul(size_of::<f32>());
            let pw1_work = frames.saturating_mul(self.pwconv1_w.len());
            if active_threads > 1
                && frames <= OC_PARALLEL_MAX_FRAMES
                && pw1_bytes >= OC_PARALLEL_THRESHOLD_BYTES
                && mid_c > 4
            {
                if frames == 1 {
                    row_workers::run_aligned(mid_buf, active_threads, 4, &|oc_off, rows| {
                        let count = rows.len();
                        let w_slice = &self.pwconv1_w[oc_off * c..(oc_off + count) * c];
                        let b_slice = &self.pwconv1_b[oc_off..oc_off + count];
                        dispatch!(
                            level,
                            simd => convnext_pwconv1_gelu_time_first_simd(
                                simd,
                                w_slice,
                                b_slice,
                                norm_buf,
                                (0, 1, c),
                                rows,
                            )
                        );
                    });
                } else {
                    with_channel_tile_out(mid_c * frames, |transposed| {
                        row_workers::run_aligned(
                            transposed,
                            active_threads,
                            4 * frames,
                            &|elem_off, slice| {
                                let oc_off = elem_off / frames;
                                let count = slice.len() / frames;
                                let w_slice = &self.pwconv1_w[oc_off * c..(oc_off + count) * c];
                                let b_slice = &self.pwconv1_b[oc_off..oc_off + count];
                                dispatch!(
                                    level,
                                    simd => convnext_pwconv1_gelu_channel_first_simd(
                                        simd,
                                        w_slice,
                                        b_slice,
                                        norm_buf,
                                        (frames, c),
                                        slice,
                                    )
                                );
                            },
                        );
                        scatter_transposed_to_time_first(
                            transposed,
                            frames,
                            mid_c,
                            active_threads,
                            mid_buf,
                        );
                    });
                }
            } else {
                let pw1_workers = frame_workers(threads, frames, pw1_work);
                row_workers::run_aligned(mid_buf, pw1_workers, mid_c, &|offset, slice| {
                    let first_frame = offset / mid_c;
                    let worker_frames = slice.len() / mid_c;
                    dispatch!(
                        level,
                        simd => convnext_pwconv1_gelu_time_first_simd(
                            simd,
                            &self.pwconv1_w,
                            &self.pwconv1_b,
                            norm_buf,
                            (first_frame, worker_frames, c),
                            slice,
                        )
                    );
                });
            }

            // Stage 3: channel-parallel `pwconv2` + fused `* gamma + residual`.
            let pw2_bytes = self.pwconv2_w.len().saturating_mul(size_of::<f32>());
            let pw2_work = frames.saturating_mul(self.pwconv2_w.len());
            if active_threads > 1
                && frames <= OC_PARALLEL_MAX_FRAMES
                && pw2_bytes >= OC_PARALLEL_THRESHOLD_BYTES
                && c > 4
            {
                if frames == 1 {
                    row_workers::run_aligned(out, active_threads, 4, &|oc_off, rows| {
                        let count = rows.len();
                        let w_slice = &self.pwconv2_w[oc_off * mid_c..(oc_off + count) * mid_c];
                        let b_slice = &self.pwconv2_b[oc_off..oc_off + count];
                        let g_slice = &self.gamma[oc_off..oc_off + count];
                        let res_slice = &input[oc_off..oc_off + count];
                        dispatch!(
                            level,
                            simd => convnext_pwconv2_res_time_first_simd(
                                simd,
                                (w_slice, b_slice, g_slice),
                                (mid_buf, res_slice),
                                (0, 1, mid_c),
                                rows,
                            )
                        );
                    });
                } else {
                    with_channel_tile_out(c * frames, |transposed| {
                        row_workers::run_aligned(
                            transposed,
                            active_threads,
                            4 * frames,
                            &|elem_off, slice| {
                                let oc_off = elem_off / frames;
                                let count = slice.len() / frames;
                                let w_slice =
                                    &self.pwconv2_w[oc_off * mid_c..(oc_off + count) * mid_c];
                                let b_slice = &self.pwconv2_b[oc_off..oc_off + count];
                                let g_slice = &self.gamma[oc_off..oc_off + count];
                                dispatch!(
                                    level,
                                    simd => convnext_pwconv2_res_channel_first_simd(
                                        simd,
                                        (w_slice, b_slice, g_slice),
                                        (mid_buf, input),
                                        (frames, mid_c, c, oc_off),
                                        slice,
                                    )
                                );
                            },
                        );
                        scatter_transposed_to_time_first(
                            transposed,
                            frames,
                            c,
                            active_threads,
                            out,
                        );
                    });
                }
            } else {
                let pw2_workers = frame_workers(threads, frames, pw2_work);
                row_workers::run_aligned(out, pw2_workers, c, &|offset, slice| {
                    let first_frame = offset / c;
                    let worker_frames = slice.len() / c;
                    dispatch!(
                        level,
                        simd => convnext_pwconv2_res_time_first_simd(
                            simd,
                            (&self.pwconv2_w, &self.pwconv2_b, &self.gamma),
                            (mid_buf, input),
                            (first_frame, worker_frames, mid_c),
                            slice,
                        )
                    );
                });
            }
        });
        Ok(())
    }
}

/// Split RVQ group (semantic or acoustic) for the 12 Hz codec decoder.
#[derive(Clone, Debug)]
pub struct RvqGroup {
    /// Number of codebooks in this split (`1` for semantic, `15` for acoustic).
    pub num_codebooks: usize,
    /// Number of entries per codebook (`2048`).
    pub codebook_size: usize,
    /// Internal embedding dimension (`256`).
    pub codebook_dim: usize,
    /// Output hidden dimension (`512`).
    pub hidden_dim: usize,
    /// Codebook tables: `num_codebooks` tables of `[codebook_size * codebook_dim]`.
    pub codebooks: Vec<Vec<f32>>,
    /// 1x1 output projection `[hidden_dim * codebook_dim]` (`[out_c, in_c]`).
    pub out_proj_w: Vec<f32>,
}

impl RvqGroup {
    /// Decode `[frames, num_codebooks]` time-first codes into `[frames, hidden_dim]`.
    pub fn decode(&self, codes: &[i32], frames: usize) -> io::Result<Vec<f32>> {
        let mut sum_emb = Vec::new();
        let mut out = Vec::new();
        self.decode_into(codes, frames, &mut sum_emb, &mut out)?;
        Ok(out)
    }

    fn decode_into(
        &self,
        codes: &[i32],
        frames: usize,
        sum_emb: &mut Vec<f32>,
        out: &mut Vec<f32>,
    ) -> io::Result<()> {
        if codes.len() != frames * self.num_codebooks
            || self.codebooks.len() != self.num_codebooks
            || self.out_proj_w.len() != self.hidden_dim * self.codebook_dim
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Invalid RVQ group decode dimensions.",
            ));
        }
        out.clear();
        out.resize(frames * self.hidden_dim, 0.0_f32);
        sum_emb.clear();
        sum_emb.resize(self.codebook_dim, 0.0_f32);
        let level = Level::new();
        for t in 0..frames {
            sum_emb.fill(0.0_f32);
            for k in 0..self.num_codebooks {
                let idx = usize::try_from(codes[t * self.num_codebooks + k]).map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidInput, "Negative RVQ code index.")
                })?;
                if idx >= self.codebook_size {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "RVQ code index exceeds codebook_size.",
                    ));
                }
                let row =
                    &self.codebooks[k][idx * self.codebook_dim..(idx + 1) * self.codebook_dim];
                dispatch!(level, simd => add_slice_in_place_simd(simd, sum_emb, row));
            }
            let out_row = &mut out[t * self.hidden_dim..(t + 1) * self.hidden_dim];
            dispatch!(
                level,
                simd => matvec_4rows_simd(simd, &self.out_proj_w, sum_emb, None, out_row)
            );
        }
        Ok(())
    }
}

/// Split RVQ decoder (`1` semantic + `15` acoustic codebooks -> `512` channels).
#[derive(Clone, Debug)]
pub struct RvqDecoder {
    /// Semantic RVQ split (`1` codebook).
    pub semantic: RvqGroup,
    /// Acoustic RVQ split (`15` codebooks).
    pub acoustic: RvqGroup,
}

impl RvqDecoder {
    /// Decode `[frames, 16]` time-first codes into `[frames, 512]`.
    pub fn decode(&self, codes: &[i32], frames: usize) -> io::Result<Vec<f32>> {
        let mut scratch = CodecDecodeScratch::default();
        self.decode_into(codes, frames, &mut scratch)?;
        Ok(scratch.rvq_sem_out)
    }

    pub(crate) fn decode_into(
        &self,
        codes: &[i32],
        frames: usize,
        scratch: &mut CodecDecodeScratch,
    ) -> io::Result<()> {
        let k_sem = self.semantic.num_codebooks;
        let k_aco = self.acoustic.num_codebooks;
        let total_k = k_sem + k_aco;
        if codes.len() != frames * total_k || self.semantic.hidden_dim != self.acoustic.hidden_dim {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Invalid RvqDecoder input codes.",
            ));
        }
        if frames == 1 {
            self.semantic.decode_into(
                &codes[..k_sem],
                1,
                &mut scratch.rvq_sum_emb,
                &mut scratch.rvq_sem_out,
            )?;
            self.acoustic.decode_into(
                &codes[k_sem..],
                1,
                &mut scratch.rvq_sum_emb,
                &mut scratch.rvq_aco_out,
            )?;
        } else {
            scratch.rvq_sem_codes.clear();
            scratch.rvq_aco_codes.clear();
            scratch.rvq_sem_codes.reserve(frames * k_sem);
            scratch.rvq_aco_codes.reserve(frames * k_aco);
            for t in 0..frames {
                let row = &codes[t * total_k..(t + 1) * total_k];
                scratch.rvq_sem_codes.extend_from_slice(&row[..k_sem]);
                scratch.rvq_aco_codes.extend_from_slice(&row[k_sem..]);
            }
            self.semantic.decode_into(
                &scratch.rvq_sem_codes,
                frames,
                &mut scratch.rvq_sum_emb,
                &mut scratch.rvq_sem_out,
            )?;
            self.acoustic.decode_into(
                &scratch.rvq_aco_codes,
                frames,
                &mut scratch.rvq_sum_emb,
                &mut scratch.rvq_aco_out,
            )?;
        }
        let level = Level::new();
        dispatch!(
            level,
            simd => add_slice_in_place_simd(simd, &mut scratch.rvq_sem_out, &scratch.rvq_aco_out)
        );
        Ok(())
    }
}

/// Host-side L2 nearest-centroid RVQ encoder side (`input_proj` + residual codebooks).
#[derive(Clone, Debug)]
pub struct RvqEncodeSide {
    /// Number of residual layers (`1` for semantic, `15` for acoustic).
    pub num_layers: usize,
    /// Codebook vocabulary size (`2048`).
    pub codebook_size: usize,
    /// Internal codebook dimension (`256`).
    pub codebook_dim: usize,
    /// Input hidden dimension (`512`).
    pub hidden_size: usize,
    /// Input 1x1 projection `[codebook_dim, hidden_size]` (`out_c, in_c`).
    pub input_proj: Vec<f32>,
    /// Codebook centroids `[num_layers][codebook_size * codebook_dim]`.
    pub codebooks: Vec<Vec<f32>>,
    /// Precomputed `||c_e||^2` per entry `[num_layers][codebook_size]`.
    pub sqsum: Vec<Vec<f32>>,
}

impl RvqEncodeSide {
    /// Construct an RVQ encode side and precompute centroid squared norms `||c_e||^2`.
    pub fn new(
        codebook_size: usize,
        codebook_dim: usize,
        hidden_size: usize,
        input_proj: Vec<f32>,
        codebooks: Vec<Vec<f32>>,
    ) -> io::Result<Self> {
        let num_layers = codebooks.len();
        if num_layers == 0
            || codebook_size == 0
            || codebook_dim == 0
            || hidden_size == 0
            || i32::try_from(codebook_size - 1).is_err()
            || codebook_dim.checked_mul(hidden_size) != Some(input_proj.len())
            || input_proj.iter().any(|value| !value.is_finite())
            || codebooks.iter().any(|cb| {
                codebook_size.checked_mul(codebook_dim) != Some(cb.len())
                    || cb.iter().any(|value| !value.is_finite())
            })
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Invalid RvqEncodeSide dimensions.",
            ));
        }
        let level = Level::new();
        let mut sqsum = Vec::with_capacity(num_layers);
        for cb in &codebooks {
            let mut layer_sq = Vec::with_capacity(codebook_size);
            for e in 0..codebook_size {
                let row = &cb[e * codebook_dim..(e + 1) * codebook_dim];
                let norm_sq = dispatch!(level, simd => dot_f32_simd(simd, row, row, 0.0_f32));
                if !norm_sq.is_finite() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Nonfinite RVQ centroid squared norm.",
                    ));
                }
                layer_sq.push(norm_sq);
            }
            sqsum.push(layer_sq);
        }
        Ok(Self {
            num_layers,
            codebook_size,
            codebook_dim,
            hidden_size,
            input_proj,
            codebooks,
            sqsum,
        })
    }

    /// Quantize `[frames, hidden_size]` into `[num_layers, frames]` codes.
    /// Failure or unwinding preserves every preexisting output code.
    pub fn encode(
        &self,
        hidden: &[f32],
        frames: usize,
        codes_out: &mut Vec<i32>,
    ) -> io::Result<()> {
        if frames == 0
            || self.num_layers == 0
            || self.codebook_size == 0
            || self.codebook_dim == 0
            || self.hidden_size == 0
            || i32::try_from(self.codebook_size - 1).is_err()
            || frames.checked_mul(self.hidden_size) != Some(hidden.len())
            || self.codebook_dim.checked_mul(self.hidden_size) != Some(self.input_proj.len())
            || self.codebooks.len() != self.num_layers
            || self.sqsum.len() != self.num_layers
            || self
                .codebooks
                .iter()
                .any(|cb| self.codebook_size.checked_mul(self.codebook_dim) != Some(cb.len()))
            || self.sqsum.iter().any(|sq| sq.len() != self.codebook_size)
            || hidden.iter().any(|value| !value.is_finite())
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Invalid RvqEncodeSide input or geometry.",
            ));
        }
        let d = self.codebook_dim;
        let elements = frames.checked_mul(d).ok_or_else(rvq_numerical_error)?;
        let mut res = vec![0.0_f32; elements];
        let level = Level::new();
        for t in 0..frames {
            let x = &hidden[t * self.hidden_size..(t + 1) * self.hidden_size];
            let y = &mut res[t * d..(t + 1) * d];
            dispatch!(
                level,
                simd => matvec_4rows_simd(simd, &self.input_proj, x, None, y)
            );
            if y.iter().any(|value| !value.is_finite()) {
                return Err(rvq_numerical_error());
            }
        }

        let mut append = RvqCodeAppend::new(codes_out);
        for l in 0..self.num_layers {
            let cb = &self.codebooks[l];
            let sq = &self.sqsum[l];
            for t in 0..frames {
                let r = &mut res[t * d..(t + 1) * d];
                let best_e = dispatch!(
                    level,
                    simd => rvq_argmin_simd(simd, cb, sq, r, self.codebook_size, d)
                )
                .ok_or_else(rvq_numerical_error)?;
                append
                    .codes
                    .push(i32::try_from(best_e).map_err(|_| rvq_numerical_error())?);
                let best_centroid = &cb[best_e * d..(best_e + 1) * d];
                dispatch!(level, simd => sub_slice_in_place_simd(simd, r, best_centroid));
                if r.iter().any(|value| !value.is_finite()) {
                    return Err(rvq_numerical_error());
                }
            }
        }
        append.committed = true;
        Ok(())
    }
}

fn rvq_numerical_error() -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, "Invalid RVQ numerical result.")
}

#[cfg(test)]
#[path = "codec/rvq_tests.rs"]
mod rvq_tests;

#[cfg(test)]
#[path = "codec/dac_tests.rs"]
mod dac_tests;

// Avoid a duplicate code buffer while preventing partial reference publication.
struct RvqCodeAppend<'a> {
    codes: &'a mut Vec<i32>,
    original: usize,
    committed: bool,
}

impl<'a> RvqCodeAppend<'a> {
    fn new(codes: &'a mut Vec<i32>) -> Self {
        Self {
            original: codes.len(),
            codes,
            committed: false,
        }
    }
}

impl Drop for RvqCodeAppend<'_> {
    fn drop(&mut self) {
        if !self.committed {
            self.codes.truncate(self.original);
        }
    }
}

/// Complete streaming state for the 12 Hz / 48 kHz causal audio decoder.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CodecStreamState {
    /// Absolute frame cursor.
    pub pos: usize,
    /// `pre_conv` (`k = 3`) left-context `[2, 512]`.
    pub pre_conv: Vec<f32>,
    /// Upsample `ConvNeXt` depthwise left-contexts (`[dwconv_kernel - 1, channels]` per block).
    pub upsample_dw: Vec<Vec<f32>>,
    /// DAC `conv_pre` (`k = 7`) left-context `[6, 1024]`.
    pub dac_pre: Vec<f32>,
    /// DAC transposed-conv overlap carries (`[stride, out_ch]` per block, 4 blocks @ 24 kHz or 5 @ 48 kHz).
    pub dac_carry: Vec<Vec<f32>>,
    /// DAC residual-unit `conv1` (`k = 7` dilated) left-contexts (`[3][6 * dilation, out_ch]` per block).
    pub dac_ru: Vec<[Vec<f32>; 3]>,
    /// DAC `conv_post` (`k = 7`) left-context.
    pub dac_post: Vec<f32>,
    /// Sliding-window pre-transformer K ring cache.
    pub kv_k: Vec<f32>,
    /// Sliding-window pre-transformer V ring cache.
    pub kv_v: Vec<f32>,
}

/// One slot in the 8-entry `CodecStateSnap` LRU cache.
#[derive(Clone, Debug, Default)]
pub struct CodecStateSnap {
    /// FNV-1a 64-bit reference-code content key.
    pub key: u64,
    /// Monotonic LRU access stamp (`0` means empty).
    pub stamp: u64,
    /// Saved streaming decoder state after ICL reference priming.
    pub state: CodecStreamState,
}

/// 8-slot LRU cache (`CODEC_SNAP_SLOTS = 8`) of primed ICL reference codec states.
#[derive(Clone, Debug)]
pub struct CodecStateCache {
    slots: [CodecStateSnap; CODEC_SNAP_SLOTS],
    stamp: u64,
}

impl Default for CodecStateCache {
    fn default() -> Self {
        Self {
            slots: core::array::from_fn(|_| CodecStateSnap::default()),
            stamp: 0,
        }
    }
}

impl CodecStateCache {
    /// Create an empty 8-slot `CodecStateSnap` LRU cache.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Borrow the streaming state matching `key`, bumping its LRU stamp on hit.
    pub fn get(&mut self, key: u64) -> Option<&CodecStreamState> {
        for slot in &mut self.slots {
            if slot.stamp != 0 && slot.key == key {
                self.stamp = self.stamp.saturating_add(1);
                slot.stamp = self.stamp;
                return Some(&slot.state);
            }
        }
        None
    }

    /// Borrow the cached streaming state for `key`, or compute and snapshot it on a miss.
    pub fn get_or_prime(
        &mut self,
        key: u64,
        prime: impl FnOnce() -> io::Result<CodecStreamState>,
    ) -> io::Result<&CodecStreamState> {
        if let Some(state) = self.get(key) {
            return Ok(state);
        }
        let state = prime()?;
        self.stamp = self.stamp.saturating_add(1);
        let lru_idx = self
            .slots
            .iter()
            .enumerate()
            .min_by_key(|(_, slot)| slot.stamp)
            .map_or(0, |(idx, _)| idx);
        self.slots[lru_idx] = CodecStateSnap {
            key,
            stamp: self.stamp,
            state,
        };
        Ok(&self.slots[lru_idx].state)
    }

    /// Restore the streaming state matching `key`, bumping its LRU stamp on hit.
    pub fn restore(&mut self, key: u64) -> Option<CodecStreamState> {
        self.get(key).cloned()
    }

    /// Save `state` under `key`, evicting the least-recently-used slot when full.
    pub fn snapshot(&mut self, key: u64, state: CodecStreamState) {
        self.stamp = self.stamp.saturating_add(1);
        if let Some(existing) = self.slots.iter_mut().find(|s| s.stamp != 0 && s.key == key) {
            existing.state = state;
            existing.stamp = self.stamp;
            return;
        }
        let lru_idx = self
            .slots
            .iter()
            .enumerate()
            .min_by_key(|(_, slot)| slot.stamp)
            .map_or(0, |(idx, _)| idx);
        self.slots[lru_idx] = CodecStateSnap {
            key,
            stamp: self.stamp,
            state,
        };
    }
}

/// Complete DAC decoder supporting both 24 kHz (`4` blocks: `8/5/4/3`) and
/// drop-in 48 kHz (`takuma104/Qwen3-TTS-Tokenizer-12Hz-48kHz`, `5` blocks: `8/5/4/3/2`).
#[derive(Clone, Debug)]
pub struct DacDecoder {
    /// Entry causal conv (`k = 7, 1024 -> 1536`).
    pub conv_pre: CausalConv1d,
    /// Upsampling blocks (`4` blocks for 24 kHz, `5` blocks for 48 kHz).
    pub blocks: Vec<DacBlock>,
    /// Post-block `SnakeBeta` activation (`96` channels @ 24 kHz, `48` channels @ 48 kHz).
    pub snake_post: SnakeBeta,
    /// Final causal conv (`k = 7 -> 1` mono channel).
    pub conv_post: CausalConv1d,
}

impl DacDecoder {
    /// Output sample rate (`24_000` Hz for 4 blocks, `48_000` Hz when the 5th 2x block is present).
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        let total_stride: usize = self.blocks.iter().map(|b| b.trans_conv.stride).product();
        if total_stride == 960 {
            SAMPLE_RATE_48K
        } else {
            SAMPLE_RATE
        }
    }

    /// Execute streaming DAC decode over `[frames, in_c]` post-upsample latents,
    /// clamping output samples to `[-1.0, 1.0]`.
    pub fn forward_stream(
        &self,
        input: &[f32],
        frames: usize,
        state: &mut CodecStreamState,
    ) -> io::Result<Vec<f32>> {
        self.forward_stream_parallel(input, frames, state, 1)
    }

    /// Decode using bounded workers for each independent convolution stage.
    pub fn forward_stream_parallel(
        &self,
        input: &[f32],
        frames: usize,
        state: &mut CodecStreamState,
        threads: usize,
    ) -> io::Result<Vec<f32>> {
        let mut scratch = CodecDecodeScratch::default();
        self.forward_stream_with_scratch(input, frames, state, threads, &mut scratch)?;
        Ok(scratch.dac.audio)
    }

    /// Decode into `scratch` reusing all intermediate buffers across streaming frames.
    pub fn forward_stream_with_scratch<'a>(
        &self,
        input: &[f32],
        frames: usize,
        state: &mut CodecStreamState,
        threads: usize,
        scratch: &'a mut CodecDecodeScratch,
    ) -> io::Result<&'a [f32]> {
        self.forward_stream_with_dac_scratch(input, frames, state, threads, &mut scratch.dac)
    }

    pub(crate) fn forward_stream_with_dac_scratch<'a>(
        &self,
        input: &[f32],
        frames: usize,
        state: &mut CodecStreamState,
        threads: usize,
        scratch: &'a mut DacDecodeScratch,
    ) -> io::Result<&'a [f32]> {
        if state.dac_carry.len() != self.blocks.len() || state.dac_ru.len() != self.blocks.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "DAC streaming state block count mismatch.",
            ));
        }
        let DacDecodeScratch {
            conv_ext,
            dac_a,
            dac_b,
            res_buf,
            block_buf,
            audio,
        } = scratch;
        self.conv_pre.forward_stream_into(
            input,
            frames,
            &mut state.dac_pre,
            threads,
            conv_ext,
            dac_a,
        )?;
        let mut cur_frames = frames;
        for (i, blk) in self.blocks.iter().enumerate() {
            cur_frames = blk.forward_stream_into(
                dac_a,
                cur_frames,
                &mut state.dac_carry[i],
                &mut state.dac_ru[i],
                threads,
                conv_ext,
                block_buf,
                res_buf,
                dac_b,
            )?;
            std::mem::swap(dac_a, dac_b);
        }
        self.snake_post
            .apply_in_place_parallel(dac_a, cur_frames, threads)?;
        self.conv_post.forward_stream_into(
            dac_a,
            cur_frames,
            &mut state.dac_post,
            threads,
            conv_ext,
            audio,
        )?;
        // Clamping infinity would turn a failed graph into apparently valid
        // full-scale PCM before Codec's final finite-output admission.
        if audio.iter().any(|sample| !sample.is_finite()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid DAC numerical result.",
            ));
        }
        for sample in audio.iter_mut() {
            *sample = sample.clamp(-1.0_f32, 1.0_f32);
        }
        Ok(audio.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CODEC_SNAP_SLOTS, CausalConv1d, CausalTransConv1d, CodecStateCache, CodecStreamState,
        ConvNeXtBlock, DacBlock, DacDecoder, DacResUnit, PadMode, RvqDecoder, RvqEncodeSide,
        RvqGroup, SnakeBeta, codec_ref_key, decimate_48k_to_24k,
    };
    use crate::{SAMPLE_RATE, SAMPLE_RATE_48K};

    fn values(count: usize) -> Vec<f32> {
        (0..count)
            .map(|i| f32::from(i16::try_from(i % 37).unwrap_or(0) - 18) * 0.013)
            .collect()
    }

    #[test]
    fn parallel_codec_frames_preserve_bits_context_and_scatter_order() -> std::io::Result<()> {
        // Odd widths exercise SIMD tails; enough fixed work selects workers.
        let conv = CausalConv1d::from_oc_ic_k(
            17,
            19,
            7,
            3,
            1,
            PadMode::Constant,
            values(19 * 17 * 7),
            Some(values(17)),
        )?;
        let input = values(41 * 19);
        let mut serial_state = values(conv.stream_context_frames() * 19);
        let mut parallel_state = serial_state.clone();
        let serial = conv.forward_stream(&input, 41, &mut serial_state)?;
        let parallel = conv.forward_stream_parallel(&input, 41, &mut parallel_state, 4)?;
        assert_eq!(
            serial.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            parallel.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(serial_state, parallel_state);
        assert_eq!(
            conv.forward(&input, 41)?,
            conv.forward_parallel(&input, 41, 4)?
        );

        for (kernel, stride) in [(16, 8), (9, 2), (4, 4)] {
            let raw_weight = values(41 * 17 * kernel);
            let trans = CausalTransConv1d::from_ic_oc_k(
                41,
                17,
                kernel,
                stride,
                &raw_weight,
                Some(values(17)),
            )?;
            let frames = 41;
            let input = values(frames * trans.in_c);
            let initial = values(trans.carry_frames() * trans.out_c);
            let mut expected =
                vec![0.0_f32; (frames * stride + trans.carry_frames()) * trans.out_c];
            expected[..initial.len()].copy_from_slice(&initial);
            // Independent scalar replay of the original whole-window scatter.
            for t in 0..frames {
                for ic in 0..trans.in_c {
                    for oc in 0..trans.out_c {
                        for k in 0..kernel {
                            let dest = (t * stride + k) * trans.out_c + oc;
                            expected[dest] = input[t * trans.in_c + ic].mul_add(
                                raw_weight[(ic * trans.out_c + oc) * kernel + k],
                                expected[dest],
                            );
                        }
                    }
                }
            }
            let expected_carry = expected[frames * stride * trans.out_c..].to_vec();
            expected.truncate(frames * stride * trans.out_c);
            for row in expected.chunks_exact_mut(trans.out_c) {
                for (value, bias) in row.iter_mut().zip(trans.bias.as_deref().unwrap_or(&[])) {
                    *value += bias;
                }
            }
            for threads in [1, 4, 18] {
                let mut carry = initial.clone();
                let output = trans.forward_stream_parallel(&input, frames, &mut carry, threads)?;
                assert_eq!(
                    output.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
                assert_eq!(carry, expected_carry);
            }
        }

        let channels = 17;
        let mid = channels * 4;
        let block = ConvNeXtBlock {
            channels,
            kernel: 7,
            dw_weight: values(channels * 7),
            dw_bias: values(channels),
            norm_w: vec![1.0; channels],
            norm_b: values(channels),
            pwconv1_w: values(channels * mid),
            pwconv1_b: values(mid),
            pwconv2_w: values(channels * mid),
            pwconv2_b: values(channels),
            gamma: values(channels),
        };
        let input = values(41 * channels);
        let mut serial_state = values(6 * channels);
        let mut parallel_state = serial_state.clone();
        let serial = block.forward_stream(&input, 41, &mut serial_state)?;
        let parallel = block.forward_stream_parallel(&input, 41, &mut parallel_state, 4)?;
        assert_eq!(
            serial.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            parallel.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(serial_state, parallel_state);
        Ok(())
    }

    #[test]
    fn short_codec_convolutions_tile_channels_without_changing_pcm_or_context()
    -> std::io::Result<()> {
        for kernel in [1, 3] {
            for out_c in [128, 129] {
                let layer = CausalConv1d::from_oc_ic_k(
                    out_c,
                    512,
                    kernel,
                    2,
                    1,
                    PadMode::Constant,
                    values(out_c * 512 * kernel),
                    Some(values(out_c)),
                )?;
                for frames in [1, 2, 4] {
                    let input = values(frames * 512);
                    let mut serial_state = values(layer.stream_context_frames() * 512);
                    let mut parallel_state = serial_state.clone();
                    let serial =
                        layer.forward_stream_parallel(&input, frames, &mut serial_state, 1)?;
                    let parallel =
                        layer.forward_stream_parallel(&input, frames, &mut parallel_state, 8)?;
                    assert_eq!(
                        serial
                            .iter()
                            .map(|value| value.to_bits())
                            .collect::<Vec<_>>(),
                        parallel
                            .iter()
                            .map(|value| value.to_bits())
                            .collect::<Vec<_>>()
                    );
                    assert_eq!(serial_state, parallel_state);
                    // A later callback must observe exactly the same retained
                    // convolution context after the channel-tiled work joins.
                    let tail = values(512);
                    assert_eq!(
                        layer.forward_stream_parallel(&tail, 1, &mut serial_state, 1)?,
                        layer.forward_stream_parallel(&tail, 1, &mut parallel_state, 8)?
                    );
                    assert_eq!(serial_state, parallel_state);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn codec_stage_sizes_preserve_selected_worker_team() {
        for frames in [2, 4, 8, 32, 1920] {
            assert_eq!(super::frame_workers(18, frames, 65_536), 18);
        }
        assert_eq!(super::frame_workers(18, 1, 1_000_000), 1);
        assert_eq!(super::frame_workers(18, 1920, 65_535), 1);
    }

    #[test]
    fn causal_conv1d_and_trans_conv1d_streaming_match_offline_bit_for_bit() -> std::io::Result<()> {
        let conv = CausalConv1d {
            out_c: 2,
            in_c: 2,
            kernel: 3,
            dilation: 2,
            stride: 1,
            pad_mode: PadMode::Constant,
            weight: (0..12)
                .map(|i| f32::from(u8::try_from(i + 1).unwrap_or(1)) * 0.05)
                .collect(),
            weight_okic: None,
            bias: Some(vec![0.1, -0.2]),
        };
        let input: Vec<f32> = (0..16)
            .map(|i| f32::from(u8::try_from(i).unwrap_or(0)).mul_add(0.1, -0.4))
            .collect();
        let (offline, out_frames) = conv.forward(&input, 8)?;
        assert_eq!(out_frames, 8);

        let mut state = vec![0.0_f32; conv.stream_context_frames() * conv.in_c];
        let mut streamed = Vec::new();
        for chunk in input.chunks(4) {
            streamed.extend(conv.forward_stream(chunk, 2, &mut state)?);
        }
        assert_eq!(offline, streamed);

        let tconv_weights: Vec<f32> = (0..16)
            .map(|i| f32::from(u8::try_from(i + 1).unwrap_or(1)) * 0.03)
            .collect();
        let tconv =
            CausalTransConv1d::from_ic_oc_k(2, 2, 4, 2, &tconv_weights, Some(vec![0.05, -0.05]))?;
        let (t_offline, t_frames) = tconv.forward(&input, 8)?;
        assert_eq!(t_frames, 16);

        let mut carry = vec![0.0_f32; tconv.carry_frames() * tconv.out_c];
        let mut t_streamed = Vec::new();
        for chunk in input.chunks(4) {
            t_streamed.extend(tconv.forward_stream(chunk, 2, &mut carry)?);
        }
        assert_eq!(t_offline, t_streamed);
        Ok(())
    }

    #[test]
    fn rvq_encode_and_decode_round_trip_nearest_centroids() -> std::io::Result<()> {
        let side = RvqEncodeSide::new(
            4,
            2,
            2,
            vec![1.0, 0.0, 0.0, 1.0],
            vec![vec![
                -1.0, -1.0, // 0
                -1.0, 1.0, // 1
                1.0, -1.0, // 2
                1.0, 1.0, // 3
            ]],
        )?;
        let hidden = vec![0.9, -0.8, -0.95, 0.85];
        let mut codes = Vec::new();
        side.encode(&hidden, 2, &mut codes)?;
        assert_eq!(codes, vec![2, 1]);

        let dec = RvqDecoder {
            semantic: RvqGroup {
                num_codebooks: 1,
                codebook_size: 4,
                codebook_dim: 2,
                hidden_dim: 2,
                codebooks: side.codebooks.clone(),
                out_proj_w: vec![1.0, 0.0, 0.0, 1.0],
            },
            acoustic: RvqGroup {
                num_codebooks: 1,
                codebook_size: 4,
                codebook_dim: 2,
                hidden_dim: 2,
                codebooks: side.codebooks,
                out_proj_w: vec![0.5, 0.0, 0.0, 0.5],
            },
        };
        let decoded = dec.decode(&[2, 0, 1, 3], 2)?;
        assert_eq!(decoded.len(), 4);
        assert!(decoded.iter().all(|v| v.is_finite()));
        Ok(())
    }

    #[test]
    fn codec_state_cache_restores_and_evicts_lru_slot() {
        let mut cache = CodecStateCache::new();
        for i in 0..CODEC_SNAP_SLOTS {
            let codes = [i32::try_from(i).unwrap_or(0); 16];
            let key = codec_ref_key(&codes, 16, 1);
            cache.snapshot(
                key,
                CodecStreamState {
                    pos: i + 1,
                    ..CodecStreamState::default()
                },
            );
        }
        // Touch slot 0 so slot 1 becomes the LRU victim.
        let key0 = codec_ref_key(&[0_i32; 16], 16, 1);
        let key1 = codec_ref_key(&[1_i32; 16], 16, 1);
        assert_eq!(cache.restore(key0).map(|s| s.pos), Some(1));

        let key_new = codec_ref_key(&[99_i32; 16], 16, 1);
        cache.snapshot(
            key_new,
            CodecStreamState {
                pos: 99,
                ..CodecStreamState::default()
            },
        );
        assert_eq!(cache.restore(key0).map(|s| s.pos), Some(1));
        assert!(cache.restore(key1).is_none());
        assert_eq!(cache.restore(key_new).map(|s| s.pos), Some(99));
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "Verifies both 4-block 24 kHz and 5-block 48 kHz DAC decode and 2:1 decimation in one test."
    )]
    fn dac_decoder_supports_24k_and_48k_upsample_blocks_and_decimation() -> std::io::Result<()> {
        let make_res = |ch: usize, dil: usize| -> std::io::Result<DacResUnit> {
            Ok(DacResUnit {
                act1: SnakeBeta::from_log_params(&vec![0.0; ch], &vec![0.0; ch])?,
                conv1: CausalConv1d::from_oc_ic_k(
                    ch,
                    ch,
                    7,
                    dil,
                    1,
                    PadMode::Constant,
                    vec![0.01; ch * ch * 7],
                    Some(vec![0.0; ch]),
                )?,
                act2: SnakeBeta::from_log_params(&vec![0.0; ch], &vec![0.0; ch])?,
                conv2: CausalConv1d::from_oc_ic_k(
                    ch,
                    ch,
                    1,
                    1,
                    1,
                    PadMode::Constant,
                    vec![0.01; ch * ch],
                    Some(vec![0.0; ch]),
                )?,
            })
        };
        let make_block = |in_c: usize, out_c: usize, stride: usize| -> std::io::Result<DacBlock> {
            Ok(DacBlock {
                snake1: SnakeBeta::from_log_params(&vec![0.0; in_c], &vec![0.0; in_c])?,
                trans_conv: CausalTransConv1d::from_ic_oc_k(
                    in_c,
                    out_c,
                    2 * stride,
                    stride,
                    &vec![0.02; in_c * out_c * 2 * stride],
                    Some(vec![0.0; out_c]),
                )?,
                res_units: [
                    make_res(out_c, 1)?,
                    make_res(out_c, 3)?,
                    make_res(out_c, 9)?,
                ],
            })
        };
        let blocks_24k = vec![
            make_block(4, 4, 8)?,
            make_block(4, 2, 5)?,
            make_block(2, 2, 4)?,
            make_block(2, 2, 3)?,
        ];
        let mut blocks_48k = blocks_24k.clone();
        blocks_48k.push(make_block(2, 2, 2)?);

        let dec_24k = DacDecoder {
            conv_pre: CausalConv1d::from_oc_ic_k(
                4,
                2,
                7,
                1,
                1,
                PadMode::Constant,
                vec![0.02; 4 * 2 * 7],
                Some(vec![0.0; 4]),
            )?,
            blocks: blocks_24k,
            snake_post: SnakeBeta::from_log_params(&[0.0; 2], &[0.0; 2])?,
            conv_post: CausalConv1d::from_oc_ic_k(
                1,
                2,
                7,
                1,
                1,
                PadMode::Constant,
                vec![0.05; 2 * 7],
                Some(vec![0.0]),
            )?,
        };
        assert_eq!(dec_24k.sample_rate(), SAMPLE_RATE);

        let dec_48k = DacDecoder {
            blocks: blocks_48k,
            ..dec_24k
        };
        assert_eq!(dec_48k.sample_rate(), SAMPLE_RATE_48K);

        let cn = ConvNeXtBlock {
            channels: 2,
            kernel: 3,
            dw_weight: vec![0.1; 6],
            dw_bias: vec![0.0; 2],
            norm_w: vec![1.0; 2],
            norm_b: vec![0.0; 2],
            pwconv1_w: vec![0.05; 16],
            pwconv1_b: vec![0.0; 8],
            pwconv2_w: vec![0.05; 16],
            pwconv2_b: vec![0.0; 2],
            gamma: vec![0.5; 2],
        };
        let mut dw_state = vec![0.0_f32; 4];
        let up_out = cn.forward_stream(&[0.2, -0.1, 0.3, 0.4], 2, &mut dw_state)?;
        assert_eq!(up_out.len(), 4);

        let decimated = decimate_48k_to_24k(&[0.2, 0.6, -0.4, 0.0]);
        assert_eq!(decimated, vec![0.4, -0.2]);
        Ok(())
    }

    #[test]
    fn simd_wide_conv_convnext_and_rvq_preserve_streaming_and_centroid_parity()
    -> std::io::Result<()> {
        let conv = CausalConv1d {
            out_c: 10,
            in_c: 9,
            kernel: 3,
            dilation: 2,
            stride: 1,
            pad_mode: PadMode::Constant,
            weight: (0..(10 * 9 * 3))
                .map(|i| f32::from(u16::try_from((i % 29) + 1).unwrap_or(1)) * 0.01)
                .collect(),
            weight_okic: None,
            bias: Some(
                (0..10)
                    .map(|i| f32::from(u8::try_from(i).unwrap_or(0)) * 0.02)
                    .collect(),
            ),
        };
        let input: Vec<f32> = (0..(6 * 9))
            .map(|i| f32::from(u8::try_from(i % 13).unwrap_or(0)).mul_add(0.05, -0.3))
            .collect();
        let (offline, _) = conv.forward(&input, 6)?;
        let mut state = vec![0.0_f32; conv.stream_context_frames() * conv.in_c];
        let mut streamed = Vec::new();
        for chunk in input.chunks(2 * 9) {
            streamed.extend(conv.forward_stream(chunk, 2, &mut state)?);
        }
        assert_eq!(offline, streamed);

        let tconv_w: Vec<f32> = (0..(6 * 10 * 4))
            .map(|i| f32::from(u16::try_from((i % 19) + 1).unwrap_or(1)) * 0.015)
            .collect();
        let tconv = CausalTransConv1d::from_ic_oc_k(
            6,
            10,
            4,
            2,
            &tconv_w,
            Some(
                (0..10)
                    .map(|i| f32::from(u8::try_from(i).unwrap_or(0)) * -0.01)
                    .collect(),
            ),
        )?;
        let t_input: Vec<f32> = (0..(4 * 6))
            .map(|i| f32::from(u8::try_from(i % 11).unwrap_or(0)).mul_add(0.07, -0.2))
            .collect();
        let (t_offline, _) = tconv.forward(&t_input, 4)?;
        let mut carry = vec![0.0_f32; tconv.carry_frames() * tconv.out_c];
        let mut t_streamed = Vec::new();
        for chunk in t_input.chunks(2 * 6) {
            t_streamed.extend(tconv.forward_stream(chunk, 2, &mut carry)?);
        }
        assert_eq!(t_offline, t_streamed);

        let c = 10;
        let mid_c = 18;
        let cn = ConvNeXtBlock {
            channels: c,
            kernel: 3,
            dw_weight: vec![0.05; c * 3],
            dw_bias: vec![0.01; c],
            norm_w: vec![1.0; c],
            norm_b: vec![0.0; c],
            pwconv1_w: (0..(mid_c * c))
                .map(|i| f32::from(u16::try_from((i % 17) + 1).unwrap_or(1)) * 0.01)
                .collect(),
            pwconv1_b: vec![0.02; mid_c],
            pwconv2_w: (0..(c * mid_c))
                .map(|i| f32::from(u16::try_from((i % 23) + 1).unwrap_or(1)) * 0.01)
                .collect(),
            pwconv2_b: vec![-0.01; c],
            gamma: vec![0.5; c],
        };
        let cn_in: Vec<f32> = (0..(4 * c))
            .map(|i| f32::from(u8::try_from(i % 9).unwrap_or(0)).mul_add(0.1, -0.4))
            .collect();
        let mut dw_all = vec![0.0_f32; 2 * c];
        let cn_batch = cn.forward_stream(&cn_in, 4, &mut dw_all)?;
        let mut dw_step = vec![0.0_f32; 2 * c];
        let mut cn_stream = Vec::new();
        for chunk in cn_in.chunks(2 * c) {
            cn_stream.extend(cn.forward_stream(chunk, 2, &mut dw_step)?);
        }
        assert_eq!(cn_batch, cn_stream);
        assert_eq!(dw_all, dw_step);
        Ok(())
    }

    #[test]
    fn codec_state_cache_lends_and_primes_lru_slots() -> std::io::Result<()> {
        let mut cache = CodecStateCache::new();
        let make_state = |marker: f32| CodecStreamState {
            pre_conv: vec![marker],
            ..CodecStreamState::default()
        };
        assert_eq!(
            cache.get_or_prime(10, || Ok(make_state(1.0)))?.pre_conv,
            vec![1.0]
        );
        assert_eq!(
            cache
                .get_or_prime(10, || {
                    Err(std::io::Error::other("cached hit must not recompute"))
                })?
                .pre_conv,
            vec![1.0]
        );
        assert_eq!(cache.restore(10).map(|s| s.pre_conv), Some(vec![1.0]));
        Ok(())
    }

    #[test]
    fn snake_beta_parallel_matches_sequential_bits() -> std::io::Result<()> {
        let channels = 96_usize;
        let frames = 65_usize;
        let alpha: Vec<f32> = (0..channels)
            .map(|c| (f32::from(u8::try_from(c % 17).unwrap_or(0)) - 8.0) * 0.05)
            .collect();
        let beta: Vec<f32> = (0..channels)
            .map(|c| (f32::from(u8::try_from((c * 3) % 19).unwrap_or(0)) - 9.0) * 0.04)
            .collect();
        let snake = SnakeBeta::from_log_params(&alpha, &beta)?;
        let original: Vec<f32> = (0..frames * channels)
            .map(|i| (f32::from(u16::try_from(i % 97).unwrap_or(0)) - 48.0) * 0.03125)
            .collect();
        let mut seq = original.clone();
        snake.apply_in_place(&mut seq, frames)?;
        for threads in [2_usize, 4, 18] {
            let mut par = original.clone();
            snake.apply_in_place_parallel(&mut par, frames, threads)?;
            assert_eq!(
                par.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                seq.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "threads {threads}",
            );
        }
        // Verify SIMD sin^2 accuracy against f64 across signed values, non-quad tail, and fallback.
        let test_channels = 7_usize;
        let test_snake = SnakeBeta {
            a: vec![1.0, 2.5, 100.0, 0.5, 40_000.0, 3.0, 1.25],
            inv_b: vec![0.25, 0.5, 1.0, 0.75, 0.5, 0.25, 0.8],
        };
        let inputs = vec![
            0.5_f32, -0.75, 327.675, -0.0, 1.0, 10.5, -3.15, -1.5, 1.5, -10.0, 2.0, -2.0, 0.0,
            0.125,
        ];
        let mut actual = inputs.clone();
        test_snake.apply_in_place(&mut actual, 2)?;
        for frame in 0..2 {
            for ch in 0..test_channels {
                let raw_x = inputs[frame * test_channels + ch];
                let z = f64::from(test_snake.a[ch] * raw_x);
                let x = f64::from(raw_x);
                let ib = f64::from(test_snake.inv_b[ch]);
                let s = z.sin();
                let expected = s * s * ib + x;
                let got = f64::from(actual[frame * test_channels + ch]);
                let tol = 3.0e-7_f64.max(expected.abs() * 3.0e-7);
                assert!(
                    (got - expected).abs() <= tol,
                    "frame={frame} ch={ch} got={got} expected={expected}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn zero_context_convolutions_and_convnext_match_padded_bits() -> std::io::Result<()> {
        let in_c = 6_usize;
        let out_c = 5_usize;
        let frames = 7_usize;
        let conv = CausalConv1d::from_oc_ic_k(
            out_c,
            in_c,
            1,
            1,
            1,
            PadMode::Constant,
            (0..out_c * in_c)
                .map(|i| (f32::from(u16::try_from(i % 13).unwrap_or(0)) - 6.0) * 0.0625)
                .collect(),
            Some(
                (0..out_c)
                    .map(|o| (f32::from(u8::try_from(o).unwrap_or(0)) - 2.0) * 0.125)
                    .collect(),
            ),
        )?;
        assert!(conv.weight_okic.is_none());
        let input: Vec<f32> = (0..frames * in_c)
            .map(|i| (f32::from(u16::try_from(i % 19).unwrap_or(0)) - 9.0) * 0.05)
            .collect();
        let mut expected = vec![0.0_f32; frames * out_c];
        for t in 0..frames {
            for o in 0..out_c {
                let mut acc = conv.bias.as_ref().map_or(0.0_f32, |b| b[o]);
                for c in 0..in_c {
                    acc = conv.weight[o * in_c + c].mul_add(input[t * in_c + c], acc);
                }
                expected[t * out_c + o] = acc;
            }
        }
        // Verify exact bit parity across offline/streaming and serial/parallel zero-context paths.
        let (off_out, off_frames) = conv.forward(&input, frames)?;
        assert_eq!(off_frames, frames);
        let mut empty_state = [];
        let stream_out = conv.forward_stream(&input, frames, &mut empty_state)?;
        assert_eq!(
            off_out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            stream_out.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        for (got, want) in off_out.iter().zip(&expected) {
            assert!((got - want).abs() < 1e-6);
        }
        for threads in [2_usize, 4] {
            let (par_off, _) = conv.forward_parallel(&input, frames, threads)?;
            let par_stream =
                conv.forward_stream_parallel(&input, frames, &mut empty_state, threads)?;
            assert_eq!(
                par_off.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                off_out.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
            assert_eq!(
                par_stream.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                off_out.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }

        let c = 8_usize;
        let cn = ConvNeXtBlock {
            channels: c,
            kernel: 1,
            dw_weight: vec![0.25; c],
            dw_bias: vec![-0.1; c],
            norm_w: vec![1.0; c],
            norm_b: vec![0.0; c],
            pwconv1_w: vec![0.05; 4 * c * c],
            pwconv1_b: vec![0.01; 4 * c],
            pwconv2_w: vec![-0.02; c * 4 * c],
            pwconv2_b: vec![0.03; c],
            gamma: vec![0.5; c],
        };
        let cn_in: Vec<f32> = (0..frames * c)
            .map(|i| (f32::from(u16::try_from(i % 11).unwrap_or(0)) - 5.0) * 0.1)
            .collect();
        let mut empty_dw = [];
        let cn_seq = cn.forward_stream(&cn_in, frames, &mut empty_dw)?;
        let cn_par = cn.forward_stream_parallel(&cn_in, frames, &mut empty_dw, 4)?;
        assert_eq!(
            cn_par.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            cn_seq.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        Ok(())
    }

    #[test]
    fn dac_decode_with_poisoned_scratch_matches_fresh_bits() -> std::io::Result<()> {
        let dac = DacDecoder {
            conv_pre: CausalConv1d::from_oc_ic_k(
                8,
                4,
                7,
                1,
                1,
                PadMode::Constant,
                vec![0.03; 8 * 4 * 7],
                Some(vec![0.01; 8]),
            )?,
            blocks: vec![DacBlock {
                snake1: SnakeBeta::from_log_params(&[0.1; 8], &[0.0; 8])?,
                trans_conv: CausalTransConv1d::from_ic_oc_k(
                    8,
                    4,
                    4,
                    2,
                    &[0.05; 8 * 4 * 4],
                    Some(vec![-0.02; 4]),
                )?,
                res_units: std::array::from_fn(|u| DacResUnit {
                    act1: SnakeBeta {
                        a: vec![1.0; 4],
                        inv_b: vec![0.5; 4],
                    },
                    conv1: CausalConv1d {
                        out_c: 4,
                        in_c: 4,
                        kernel: 7,
                        dilation: [1, 3, 9][u],
                        stride: 1,
                        pad_mode: PadMode::Constant,
                        weight: vec![0.02; 4 * 4 * 7],
                        weight_okic: None,
                        bias: Some(vec![0.01; 4]),
                    },
                    act2: SnakeBeta {
                        a: vec![0.8; 4],
                        inv_b: vec![0.6; 4],
                    },
                    conv2: CausalConv1d {
                        out_c: 4,
                        in_c: 4,
                        kernel: 1,
                        dilation: 1,
                        stride: 1,
                        pad_mode: PadMode::Constant,
                        weight: vec![-0.03; 4 * 4],
                        weight_okic: None,
                        bias: Some(vec![0.0; 4]),
                    },
                }),
            }],
            snake_post: SnakeBeta::from_log_params(&[0.0; 4], &[0.1; 4])?,
            conv_post: CausalConv1d::from_oc_ic_k(
                1,
                4,
                7,
                1,
                1,
                PadMode::Constant,
                vec![0.04; 4 * 7],
                Some(vec![0.0]),
            )?,
        };
        let make_state = || CodecStreamState {
            pos: 0,
            pre_conv: Vec::new(),
            upsample_dw: Vec::new(),
            dac_pre: vec![0.0; 6 * 4],
            dac_carry: vec![vec![0.0; 2 * 4]],
            dac_ru: vec![[vec![0.0; 6 * 4], vec![0.0; 18 * 4], vec![0.0; 54 * 4]]],
            dac_post: vec![0.0; 6 * 4],
            kv_k: Vec::new(),
            kv_v: Vec::new(),
        };
        let mut seq_state = make_state();
        let mut scratch_state = make_state();
        let mut scratch = super::CodecDecodeScratch {
            conv_ext: vec![f32::NAN; 512],
            trans_carry: vec![f32::NAN; 64],
            dac: super::DacDecodeScratch {
                conv_ext: vec![f32::NAN; 512],
                dac_a: vec![f32::NAN; 256],
                dac_b: vec![f32::NAN; 256],
                res_buf: vec![f32::NAN; 256],
                block_buf: vec![f32::NAN; 256],
                audio: vec![f32::NAN; 64],
            },
            ..super::CodecDecodeScratch::default()
        };
        for step in 0..3_usize {
            let input: Vec<f32> = (0..2 * 4)
                .map(|i| (f32::from(u8::try_from((i + step * 3) % 11).unwrap_or(0)) - 5.0) * 0.1)
                .collect();
            let expected = dac.forward_stream_parallel(&input, 2, &mut seq_state, 2)?;
            let actual =
                dac.forward_stream_with_scratch(&input, 2, &mut scratch_state, 2, &mut scratch)?;
            assert_eq!(
                actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "step {step}"
            );
        }
        Ok(())
    }

    #[test]
    fn convnext_vectorized_tanh_gelu_matches_f64_reference() -> std::io::Result<()> {
        use fearless_simd::{Level, dispatch, f32x4, prelude::*};
        let level = Level::new();
        let mut grid: Vec<f32> = (-2000_i32..=2000)
            .map(|i| f32::from(i16::try_from(i).unwrap_or(0)) * 0.01)
            .collect();
        grid.extend([
            -0.0_f32, 0.0, 0.499, 0.5, 0.501, -0.499, -0.5, -0.501, 9.4, 9.5, 12.0,
        ]);
        for chunk in grid.chunks(4) {
            let mut buf = [0.0_f32; 4];
            for (i, &v) in chunk.iter().enumerate() {
                buf[i] = v;
            }
            let got_simd: [f32; 4] = dispatch!(level, simd => {
                super::tanh_f32x4(simd, f32x4::from_slice(simd, &buf)).into()
            });
            for (i, &z) in chunk.iter().enumerate() {
                let got_sc = super::tanh_scalar(z);
                assert_eq!(
                    got_simd[i].to_bits(),
                    got_sc.to_bits(),
                    "SIMD vs scalar tanh mismatch at z={z}"
                );
                let expected = f64::from(z).tanh();
                let err = (f64::from(got_simd[i]) - expected).abs();
                assert!(err <= 4.0e-7, "tanh error {err} at z={z}");
            }
        }

        let channels = 9_usize;
        let kernel = 3_usize;
        let frames = 6_usize;
        let mid_c = 19_usize;
        let cn = ConvNeXtBlock {
            channels,
            kernel,
            dw_weight: values(channels * kernel),
            dw_bias: values(channels),
            norm_w: vec![1.0; channels],
            norm_b: values(channels),
            pwconv1_w: values(mid_c * channels),
            pwconv1_b: values(mid_c),
            pwconv2_w: values(channels * mid_c),
            pwconv2_b: values(channels),
            gamma: values(channels),
        };
        let input = values(frames * channels);
        let mut dw_state = values((kernel - 1) * channels);
        let actual = cn.forward_stream(&input, frames, &mut dw_state)?;

        // Verify ConvNeXt output against exact f64 tanh-GELU reference.
        let mut ext = values((kernel - 1) * channels);
        ext.extend_from_slice(&input);
        let k_gelu = (2.0_f64 / core::f64::consts::PI).sqrt();
        for t in 0..frames {
            let mut norm = vec![0.0_f64; channels];
            for ch in 0..channels {
                let mut acc = f64::from(cn.dw_bias[ch]);
                for ki in 0..kernel {
                    acc += f64::from(cn.dw_weight[ch * kernel + ki])
                        * f64::from(ext[(t + ki) * channels + ch]);
                }
                norm[ch] = acc;
            }
            let mean = norm.iter().sum::<f64>() / f64::from(u8::try_from(channels).unwrap_or(1));
            let var = norm.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>()
                / f64::from(u8::try_from(channels).unwrap_or(1));
            let inv_std = 1.0_f64 / (var + 1e-6_f64).sqrt();
            for ch in 0..channels {
                norm[ch] = (norm[ch] - mean) * inv_std * f64::from(cn.norm_w[ch])
                    + f64::from(cn.norm_b[ch]);
            }
            let mut mid = vec![0.0_f64; mid_c];
            for m in 0..mid_c {
                let mut acc = f64::from(cn.pwconv1_b[m]);
                for ch in 0..channels {
                    acc += f64::from(cn.pwconv1_w[m * channels + ch]) * norm[ch];
                }
                let inner = k_gelu * (acc + 0.044_715_f64 * acc * acc * acc);
                mid[m] = 0.5_f64 * acc * (1.0_f64 + inner.tanh());
            }
            for ch in 0..channels {
                let mut acc = f64::from(cn.pwconv2_b[ch]);
                for m in 0..mid_c {
                    acc += f64::from(cn.pwconv2_w[ch * mid_c + m]) * mid[m];
                }
                let expected = f64::from(input[t * channels + ch]) + acc * f64::from(cn.gamma[ch]);
                let got = f64::from(actual[t * channels + ch]);
                assert!(
                    (got - expected).abs() <= 2.0e-6,
                    "t={t} ch={ch} got={got} expected={expected}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn causal_conv1d_four_frame_tiled_gemm_matches_single_frame_bits() -> std::io::Result<()> {
        for (kernel, dilation, in_c, out_c) in
            [(1_usize, 1_usize, 259_usize, 131_usize), (3, 2, 129, 133)]
        {
            let conv = CausalConv1d::from_oc_ic_k(
                out_c,
                in_c,
                kernel,
                dilation,
                1,
                PadMode::Constant,
                values(out_c * in_c * kernel),
                Some(values(out_c)),
            )?;
            for frames in [4_usize, 5, 9] {
                let input = values(frames * in_c);
                let init_state = values(conv.stream_context_frames() * in_c);
                let mut per_frame_state = init_state.clone();
                let mut per_frame = Vec::with_capacity(frames * out_c);
                for f in 0..frames {
                    per_frame.extend(conv.forward_stream(
                        &input[f * in_c..(f + 1) * in_c],
                        1,
                        &mut per_frame_state,
                    )?);
                }
                for threads in [1_usize, 4, 18] {
                    let mut batch_state = init_state.clone();
                    let batched =
                        conv.forward_stream_parallel(&input, frames, &mut batch_state, threads)?;
                    assert_eq!(
                        batched.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        per_frame.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        "kernel={kernel} frames={frames} threads={threads}"
                    );
                    assert_eq!(batch_state, per_frame_state);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn causal_conv1d_okic_zero_spill_matches_f64_and_parallel_streaming_bits() -> std::io::Result<()>
    {
        // Exercise both small odd dims and large >= 1 MiB weights (oc-parallel + 4x4 SIMD transpose).
        for (in_c, out_c, kernel, dilation) in
            [(19_usize, 17_usize, 7_usize, 9_usize), (131, 289, 7, 9)]
        {
            let weight = values(out_c * in_c * kernel);
            let bias = values(out_c);
            let conv = CausalConv1d::from_oc_ic_k(
                out_c,
                in_c,
                kernel,
                dilation,
                1,
                PadMode::Constant,
                weight.clone(),
                Some(bias.clone()),
            )?;
            assert!(conv.weight_okic.is_some());
            let mut conv_fallback = conv.clone();
            conv_fallback.weight_okic = None;

            let ctx_frames = conv.stream_context_frames();
            for frames in [1_usize, 2, 3, 4, 32, 256] {
                let input = values(frames * in_c);
                let (offline_t1, out_frames) = conv.forward_parallel(&input, frames, 1)?;
                assert_eq!(out_frames, frames);

                // Verify f64 reference within <= 2e-5.
                for t in 0..frames {
                    for oc in 0..out_c {
                        let mut acc = f64::from(bias[oc]);
                        for k in 0..kernel {
                            let src_t = t + k * dilation;
                            if src_t >= ctx_frames {
                                let in_t = src_t - ctx_frames;
                                for ic in 0..in_c {
                                    acc += f64::from(weight[(oc * in_c + ic) * kernel + k])
                                        * f64::from(input[in_t * in_c + ic]);
                                }
                            }
                        }
                        let got = f64::from(offline_t1[t * out_c + oc]);
                        assert!(
                            (got - acc).abs() <= 2e-5,
                            "f64 mismatch in_c={in_c} out_c={out_c} frames={frames} t={t} oc={oc}: got={got} expected={acc}"
                        );
                    }
                }

                // Verify bit-identical threads in [1, 4, 10, 16] and weight_okic=None fallback.
                let offline_t1_bits: Vec<u32> = offline_t1.iter().map(|v| v.to_bits()).collect();
                for threads in [1_usize, 4, 10, 16] {
                    let (par_off, _) = conv.forward_parallel(&input, frames, threads)?;
                    assert_eq!(
                        par_off.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        offline_t1_bits,
                        "offline threads={threads} in_c={in_c} out_c={out_c} frames={frames}"
                    );
                    let mut st = vec![0.0_f32; ctx_frames * in_c];
                    let par_stream =
                        conv.forward_stream_parallel(&input, frames, &mut st, threads)?;
                    assert_eq!(
                        par_stream.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        offline_t1_bits,
                        "stream threads={threads} in_c={in_c} out_c={out_c} frames={frames}"
                    );
                }
                let (fallback_out, _) = conv_fallback.forward_parallel(&input, frames, 16)?;
                assert_eq!(
                    fallback_out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    offline_t1_bits,
                    "fallback weight_okic=None mismatch in_c={in_c} out_c={out_c} frames={frames}"
                );

                // Verify chunked [1, 2, 4, 8] streaming vs offline bits.
                let mut chunk_state = vec![0.0_f32; ctx_frames * in_c];
                let mut streamed = Vec::with_capacity(frames * out_c);
                let mut cursor = 0_usize;
                let mut chunk_cycle = [1_usize, 2, 4, 8].into_iter().cycle();
                while cursor < frames {
                    let take = chunk_cycle.next().unwrap_or(1).min(frames - cursor);
                    streamed.extend(conv.forward_stream_parallel(
                        &input[cursor * in_c..(cursor + take) * in_c],
                        take,
                        &mut chunk_state,
                        16,
                    )?);
                    cursor += take;
                }
                assert_eq!(
                    streamed.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    offline_t1_bits,
                    "chunked [1,2,4,8] streaming mismatch in_c={in_c} out_c={out_c} frames={frames}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn causal_trans_conv1d_dac_block0_and_odd_shapes_preserve_zero_ulp_bits() -> std::io::Result<()>
    {
        // Odd out_c and kernel > 2 * stride against complete scalar reference across frames & threads.
        for (in_c, out_c, kernel, stride) in
            [(19_usize, 17_usize, 11_usize, 3_usize), (37, 23, 9, 2)]
        {
            let raw_weight = values(in_c * out_c * kernel);
            let bias = values(out_c);
            let trans = CausalTransConv1d::from_ic_oc_k(
                in_c,
                out_c,
                kernel,
                stride,
                &raw_weight,
                Some(bias.clone()),
            )?;
            let initial_carry = values(trans.carry_frames() * out_c);
            for frames in [1_usize, 2, 4, 13, 32] {
                let input = values(frames * in_c);
                let mut expected = vec![0.0_f32; (frames * stride + trans.carry_frames()) * out_c];
                expected[..initial_carry.len()].copy_from_slice(&initial_carry);
                for t in 0..frames {
                    for ic in 0..in_c {
                        for oc in 0..out_c {
                            for k in 0..kernel {
                                let dest = (t * stride + k) * out_c + oc;
                                expected[dest] = input[t * in_c + ic].mul_add(
                                    raw_weight[(ic * out_c + oc) * kernel + k],
                                    expected[dest],
                                );
                            }
                        }
                    }
                }
                let expected_carry = expected[frames * stride * out_c..].to_vec();
                expected.truncate(frames * stride * out_c);
                for row in expected.chunks_exact_mut(out_c) {
                    for (value, &b) in row.iter_mut().zip(&bias) {
                        *value += b;
                    }
                }
                let expected_bits: Vec<u32> = expected.iter().map(|v| v.to_bits()).collect();
                let expected_carry_bits: Vec<u32> =
                    expected_carry.iter().map(|v| v.to_bits()).collect();
                for threads in [1_usize, 4, 10, 16] {
                    let mut carry = initial_carry.clone();
                    let actual =
                        trans.forward_stream_parallel(&input, frames, &mut carry, threads)?;
                    assert_eq!(
                        actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        expected_bits,
                        "odd tconv mismatch in_c={in_c} out_c={out_c} k={kernel} s={stride} frames={frames} threads={threads}"
                    );
                    assert_eq!(
                        carry.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        expected_carry_bits
                    );
                }
            }
        }

        // Explicit DAC Block 0 geometry: in_c=1536, out_c=768, k=16, s=8.
        let in_c = 1536_usize;
        let out_c = 768_usize;
        let kernel = 16_usize;
        let stride = 8_usize;
        let raw_weight = values(in_c * out_c * kernel);
        let bias = values(out_c);
        let block0 = CausalTransConv1d::from_ic_oc_k(
            in_c,
            out_c,
            kernel,
            stride,
            &raw_weight,
            Some(bias.clone()),
        )?;
        let initial_carry = values(block0.carry_frames() * out_c);
        for frames in [1_usize, 4, 32] {
            let input = values(frames * in_c);
            let mut carry_ref = initial_carry.clone();
            let out_t1 = block0.forward_stream_parallel(&input, frames, &mut carry_ref, 1)?;
            let out_t1_bits: Vec<u32> = out_t1.iter().map(|v| v.to_bits()).collect();
            let carry_ref_bits: Vec<u32> = carry_ref.iter().map(|v| v.to_bits()).collect();

            // Check exact scalar replay on all output/carry frames for frames in [1, 4],
            // and representative boundary/interior/carry coordinates for frames=32.
            let check_coords: Vec<(usize, usize)> = if frames <= 4 {
                (0..frames * stride + block0.carry_frames())
                    .flat_map(|y| {
                        [0_usize, 15, 16, 17, 383, 767]
                            .into_iter()
                            .map(move |oc| (y, oc))
                    })
                    .collect()
            } else {
                [0_usize, 7, 8, 9, 127, 255, 256, 263]
                    .into_iter()
                    .flat_map(|y| {
                        [0_usize, 15, 16, 511, 767]
                            .into_iter()
                            .map(move |oc| (y, oc))
                    })
                    .collect()
            };
            for (y, oc) in check_coords {
                let mut acc = if y < block0.carry_frames() {
                    initial_carry[y * out_c + oc]
                } else {
                    0.0_f32
                };
                for t in 0..frames {
                    if y >= t * stride && y < t * stride + kernel {
                        let k = y - t * stride;
                        for ic in 0..in_c {
                            acc = input[t * in_c + ic]
                                .mul_add(raw_weight[(ic * out_c + oc) * kernel + k], acc);
                        }
                    }
                }
                if y < frames * stride {
                    acc += bias[oc];
                    assert_eq!(
                        out_t1[y * out_c + oc].to_bits(),
                        acc.to_bits(),
                        "DAC Block 0 0-ULP scalar mismatch frames={frames} y={y} oc={oc}"
                    );
                } else {
                    let cy = y - frames * stride;
                    assert_eq!(
                        carry_ref[cy * out_c + oc].to_bits(),
                        acc.to_bits(),
                        "DAC Block 0 0-ULP carry mismatch frames={frames} cy={cy} oc={oc}"
                    );
                }
            }

            for threads in [10_usize, 16] {
                let mut carry = initial_carry.clone();
                let out_par =
                    block0.forward_stream_parallel(&input, frames, &mut carry, threads)?;
                assert_eq!(
                    out_par.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    out_t1_bits,
                    "DAC Block 0 frames={frames} threads={threads}"
                );
                assert_eq!(
                    carry.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    carry_ref_bits,
                    "DAC Block 0 carry frames={frames} threads={threads}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn convnext_three_stage_parallel_matches_single_thread_bits() -> std::io::Result<()> {
        for (channels, mid_c) in [(129_usize, 517_usize), (256, 1024)] {
            let kernel = 7_usize;
            let block = ConvNeXtBlock {
                channels,
                kernel,
                dw_weight: values(channels * kernel),
                dw_bias: values(channels),
                norm_w: vec![1.05; channels],
                norm_b: values(channels),
                pwconv1_w: values(mid_c * channels),
                pwconv1_b: values(mid_c),
                pwconv2_w: values(channels * mid_c),
                pwconv2_b: values(channels),
                gamma: values(channels),
            };
            for frames in [1_usize, 2, 4, 32] {
                let input = values(frames * channels);
                let init_state = values((kernel - 1) * channels);
                let mut state_t1 = init_state.clone();
                let mut state_t16 = init_state;
                let out_t1 = block.forward_stream_parallel(&input, frames, &mut state_t1, 1)?;
                let out_t16 = block.forward_stream_parallel(&input, frames, &mut state_t16, 16)?;
                assert_eq!(
                    out_t16.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    out_t1.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "ConvNeXt threads=1 vs 16 mismatch channels={channels} mid_c={mid_c} frames={frames}"
                );
                assert_eq!(state_t16, state_t1);
            }
        }
        Ok(())
    }
}
