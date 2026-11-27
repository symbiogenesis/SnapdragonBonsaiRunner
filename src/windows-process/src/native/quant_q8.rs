//! Audited ARM64 `Q8_0` / `Q4_K` / `Q6_K` `SDOT` decode and `I8MM` (`SMMLA`) GEMM kernels.
//!
//! Every public-to-crate entry point validates slice bounds and runtime CPU capability flags
//! before executing optional ARM64 instructions, preserving exact `0-ULP` bit parity with
//! [`bonsai_neon_dot::kquants::dot_q8_0_row`], [`bonsai_neon_dot::kquants::dot_q4_k_row`], and
//! [`bonsai_neon_dot::kquants::dot_q6_k_row`].

use bonsai_neon_dot::kquants::{
    Arm64QuantCaps, Q4_K_REPACK8_BLOCK_BYTES, Q6_K_BLOCK_BYTES, Q8_0_BLOCK_BYTES, Q8_0ActBlock,
    Q8_KActBlock, dot_q6_k_4rows, dot_q6_k_row, dot_q8_0_4rows, dot_q8_0_row,
    gemm_q4_k_repacked8_8x2,
};

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::{
    float32x4_t, float64x2_t, int8x16_t, int32x4_t, uint8x8_t, uint8x16_t, uint16x8_t, vaddq_f32,
    vaddq_f64, vaddvq_s32, vand_u16, vandq_u8, vandq_u16, vcombine_f32, vcombine_s8, vcombine_s16,
    vcvt_f32_f16, vcvt_f32_f64, vcvt_f64_f32, vcvt_high_f32_f16, vcvt_high_f64_f32,
    vcvtq_f32_s32, vcvtq_f64_s64, vdotq_s32, vdup_n_s16, vdup_n_u16, vdupq_n_f32, vdupq_n_f64,
    vdupq_n_s8, vdupq_n_s32, vdupq_n_u8, vdupq_n_u16, vfmaq_f32, vget_high_s8, vget_high_u16, vget_low_f16,
    vget_low_f32, vget_low_s8, vget_low_s16, vget_low_s32, vget_low_u16, vld1_u8, vld1_u16,
    vld1q_f32, vld1q_f64, vld1q_lane_u16, vld1q_s8, vld1q_s32, vld1q_u8, vld1q_u16, vmaxv_u16,
    vmaxvq_u16, vmlal_high_s16, vmlal_s16, vmlaq_s32, vmovl_high_s32, vmovl_s32, vmovl_u8,
    vmovl_u16, vmulq_f32, vmulq_f64, vmulq_laneq_f32, vorrq_u8, vpaddq_f32, vpaddq_s32,
    vreinterpret_f16_u16, vreinterpretq_f16_u16, vreinterpretq_s8_u8, vreinterpretq_s16_u16,
    vreinterpretq_s32_u32, vshlq_n_u8, vshrq_n_u8, vst1q_f32, vst1q_f64, vst1q_s8, vst1q_s32,
    vsubq_f64, vsubq_s8, vzip1q_f32, vzip1q_s32, vzip2q_f32, vzip2q_s32,
};

#[inline]
#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Superblock f64 scale product narrows once per dot output matching K-quant reference."
)]
const fn f64_to_f32(value: f64) -> f32 {
    value as f32
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn load_u8x8(bytes: &[u8; 8]) -> uint8x8_t {
    // SAFETY: `bytes` is a valid 8-byte array reference and `vld1_u8` requires no alignment.
    unsafe { vld1_u8(bytes.as_ptr()) }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn load_u8x16(bytes: &[u8; 16]) -> uint8x16_t {
    // SAFETY: `bytes` is a valid 16-byte array reference and `vld1q_u8` requires no alignment.
    unsafe { vld1q_u8(bytes.as_ptr()) }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn load_s8x16(bytes: &[i8; 16]) -> int8x16_t {
    // SAFETY: `bytes` is a valid 16-byte array reference and `vld1q_s8` requires no alignment.
    unsafe { vld1q_s8(bytes.as_ptr()) }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn store_s8x16(vec: int8x16_t, out: &mut [i8; 16]) {
    // SAFETY: `out` is a valid mutable 16-byte array reference and `vst1q_s8` requires no alignment.
    unsafe { vst1q_s8(out.as_mut_ptr(), vec) };
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn load_f32x4(values: &[f32; 4]) -> float32x4_t {
    // SAFETY: `values` is a valid 4-element `f32` array reference and `vld1q_f32` requires no alignment.
    unsafe { vld1q_f32(values.as_ptr()) }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn store_f32x4(vec: float32x4_t, out: &mut [f32; 4]) {
    // SAFETY: `out` is a valid mutable 4-element `f32` array reference and `vst1q_f32` requires no alignment.
    unsafe { vst1q_f32(out.as_mut_ptr(), vec) };
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn prefetch_l1(ptr: *const u8) {
    // SAFETY: `prfm pldl1keep` is a non-faulting read prefetch hint on an address within a live slice.
    unsafe {
        core::arch::asm!(
            "prfm pldl1keep, [{ptr}]",
            ptr = in(reg) ptr,
            options(nostack, preserves_flags),
        );
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon,i8mm")]
fn smmla_s32(mut acc: int32x4_t, w: int8x16_t, a: int8x16_t) -> int32x4_t {
    // SAFETY: Caller verified `neon,i8mm` support; `smmla` operates strictly on vector registers.
    unsafe {
        core::arch::asm!(
            "smmla {acc:v}.4s, {w:v}.16b, {a:v}.16b",
            acc = inout(vreg) acc,
            w = in(vreg) w,
            a = in(vreg) a,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    acc
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon,dotprod")]
fn sdot_q8_block(blk: &[u8; 34], a0: int8x16_t, a1: int8x16_t, zero: int32x4_t) -> int32x4_t {
    let halves = blk[2..34].as_chunks::<16>().0;
    let w0 = vreinterpretq_s8_u8(load_u8x16(&halves[0]));
    let w1 = vreinterpretq_s8_u8(load_u8x16(&halves[1]));
    vdotq_s32(vdotq_s32(zero, w0, a0), w1, a1)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
fn dot_q8_0_8rows_dotprod(rows: [&[u8]; 8], act: &[Q8_0ActBlock]) -> [f32; 8] {
    let r_chunks: [&[[u8; 34]]; 8] = rows.map(|r| r.as_chunks::<34>().0);
    let blocks = act.len();
    let mut acc_lo = vdupq_n_f32(0.0_f32);
    let mut acc_hi = vdupq_n_f32(0.0_f32);
    let zero_i32 = vdupq_n_s32(0);

    for (idx, ab) in act.iter().enumerate() {
        if idx + 2 < blocks {
            for r in &r_chunks {
                prefetch_l1(r[idx + 2].as_ptr());
            }
        }
        let b0 = &r_chunks[0][idx];
        let b1 = &r_chunks[1][idx];
        let b2 = &r_chunks[2][idx];
        let b3 = &r_chunks[3][idx];
        let b4 = &r_chunks[4][idx];
        let b5 = &r_chunks[5][idx];
        let b6 = &r_chunks[6][idx];
        let b7 = &r_chunks[7][idx];

        let d_bits: [u16; 8] = [
            u16::from_le_bytes([b0[0], b0[1]]),
            u16::from_le_bytes([b1[0], b1[1]]),
            u16::from_le_bytes([b2[0], b2[1]]),
            u16::from_le_bytes([b3[0], b3[1]]),
            u16::from_le_bytes([b4[0], b4[1]]),
            u16::from_le_bytes([b5[0], b5[1]]),
            u16::from_le_bytes([b6[0], b6[1]]),
            u16::from_le_bytes([b7[0], b7[1]]),
        ];
        let [w_lo_arr, w_hi_arr] = bonsai_neon_dot::widen_f16_eight(&d_bits);
        let w_scale_lo = load_f32x4(&w_lo_arr);
        let w_scale_hi = load_f32x4(&w_hi_arr);
        let act_scale = vdupq_n_f32(ab.d);

        let a_halves = ab.qs.as_chunks::<16>().0;
        let a0 = load_s8x16(&a_halves[0]);
        let a1 = load_s8x16(&a_halves[1]);

        let d0 = sdot_q8_block(b0, a0, a1, zero_i32);
        let d1 = sdot_q8_block(b1, a0, a1, zero_i32);
        let d2 = sdot_q8_block(b2, a0, a1, zero_i32);
        let d3 = sdot_q8_block(b3, a0, a1, zero_i32);
        let dots_lo = vpaddq_s32(vpaddq_s32(d0, d1), vpaddq_s32(d2, d3));

        let d4 = sdot_q8_block(b4, a0, a1, zero_i32);
        let d5 = sdot_q8_block(b5, a0, a1, zero_i32);
        let d6 = sdot_q8_block(b6, a0, a1, zero_i32);
        let d7 = sdot_q8_block(b7, a0, a1, zero_i32);
        let dots_hi = vpaddq_s32(vpaddq_s32(d4, d5), vpaddq_s32(d6, d7));

        let dot_f32_lo = vcvtq_f32_s32(dots_lo);
        let dot_f32_hi = vcvtq_f32_s32(dots_hi);

        let prod_lo = vmulq_f32(w_scale_lo, vmulq_f32(act_scale, dot_f32_lo));
        let prod_hi = vmulq_f32(w_scale_hi, vmulq_f32(act_scale, dot_f32_hi));
        acc_lo = vaddq_f32(acc_lo, prod_lo);
        acc_hi = vaddq_f32(acc_hi, prod_hi);
    }

    let mut out_lo = [0.0_f32; 4];
    let mut out_hi = [0.0_f32; 4];
    store_f32x4(acc_lo, &mut out_lo);
    store_f32x4(acc_hi, &mut out_hi);
    [
        out_lo[0], out_lo[1], out_lo[2], out_lo[3], out_hi[0], out_hi[1], out_hi[2], out_hi[3],
    ]
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,i8mm")]
fn gemm_q8_0_8x_i8mm<const TOKENS: usize, const COL_PAIRS: usize>(
    rows: [&[u8]; 8],
    acts: [&[Q8_0ActBlock]; TOKENS],
) -> [[f32; TOKENS]; 8] {
    let r_chunks: [&[[u8; 34]]; 8] = rows.map(|r| r.as_chunks::<34>().0);
    let blocks = acts[0].len();
    let mut acc = [[vdupq_n_f32(0.0_f32); COL_PAIRS]; 4];

    for idx in 0..blocks {
        if idx + 2 < blocks {
            for r in &r_chunks {
                prefetch_l1(r[idx + 2].as_ptr());
            }
        }
        let b0 = &r_chunks[0][idx];
        let b1 = &r_chunks[1][idx];
        let b2 = &r_chunks[2][idx];
        let b3 = &r_chunks[3][idx];
        let b4 = &r_chunks[4][idx];
        let b5 = &r_chunks[5][idx];
        let b6 = &r_chunks[6][idx];
        let b7 = &r_chunks[7][idx];

        let d_bits: [u16; 8] = [
            u16::from_le_bytes([b0[0], b0[1]]),
            u16::from_le_bytes([b1[0], b1[1]]),
            u16::from_le_bytes([b2[0], b2[1]]),
            u16::from_le_bytes([b3[0], b3[1]]),
            u16::from_le_bytes([b4[0], b4[1]]),
            u16::from_le_bytes([b5[0], b5[1]]),
            u16::from_le_bytes([b6[0], b6[1]]),
            u16::from_le_bytes([b7[0], b7[1]]),
        ];
        let [ws_lo_arr, ws_hi_arr] = bonsai_neon_dot::widen_f16_eight(&d_bits);
        let ws_lo = load_f32x4(&ws_lo_arr);
        let ws_hi = load_f32x4(&ws_hi_arr);
        let d_w_pairs = [
            vzip1q_f32(ws_lo, ws_lo),
            vzip2q_f32(ws_lo, ws_lo),
            vzip1q_f32(ws_hi, ws_hi),
            vzip2q_f32(ws_hi, ws_hi),
        ];

        let d_act_arr: [[f32; 4]; COL_PAIRS] = std::array::from_fn(|cp| {
            let d0 = acts[cp * 2][idx].d;
            let d1 = acts[cp * 2 + 1][idx].d;
            [d0, d1, d0, d1]
        });

        let mut packed_a = [[[vdupq_n_s8(0); 2]; COL_PAIRS]; 2];
        for (half, half_a) in packed_a.iter_mut().enumerate() {
            for (cp, pair_a) in half_a.iter_mut().enumerate() {
                let a_even = load_s8x16(&acts[cp * 2][idx].qs.as_chunks::<16>().0[half]);
                let a_odd = load_s8x16(&acts[cp * 2 + 1][idx].qs.as_chunks::<16>().0[half]);
                *pair_a = [
                    vcombine_s8(vget_low_s8(a_even), vget_low_s8(a_odd)),
                    vcombine_s8(vget_high_s8(a_even), vget_high_s8(a_odd)),
                ];
            }
        }

        let blks = [b0, b1, b2, b3, b4, b5, b6, b7];
        for rp in 0..4 {
            let mut c_rp = [vdupq_n_s32(0); COL_PAIRS];
            let halves0 = blks[rp * 2][2..34].as_chunks::<16>().0;
            let halves1 = blks[rp * 2 + 1][2..34].as_chunks::<16>().0;
            for (half, half_a) in packed_a.iter().enumerate() {
                let w_even = vreinterpretq_s8_u8(load_u8x16(&halves0[half]));
                let w_odd = vreinterpretq_s8_u8(load_u8x16(&halves1[half]));
                let w_lo = vcombine_s8(vget_low_s8(w_even), vget_low_s8(w_odd));
                let w_hi = vcombine_s8(vget_high_s8(w_even), vget_high_s8(w_odd));
                for (cp, &[a_lo, a_hi]) in half_a.iter().enumerate() {
                    c_rp[cp] = smmla_s32(smmla_s32(c_rp[cp], w_lo, a_lo), w_hi, a_hi);
                }
            }
            for (cp, &c_vec) in c_rp.iter().enumerate() {
                let dot_f32 = vcvtq_f32_s32(c_vec);
                let d_act_v = load_f32x4(&d_act_arr[cp]);
                let prod = vmulq_f32(d_w_pairs[rp], vmulq_f32(d_act_v, dot_f32));
                acc[rp][cp] = vaddq_f32(acc[rp][cp], prod);
            }
        }
    }

    let mut out = [[0.0_f32; TOKENS]; 8];
    for (rp, pair_acc) in acc.iter().enumerate() {
        for (cp, &tile_v) in pair_acc.iter().enumerate() {
            let mut tile = [0.0_f32; 4];
            store_f32x4(tile_v, &mut tile);
            out[rp * 2][cp * 2] = tile[0];
            out[rp * 2][cp * 2 + 1] = tile[1];
            out[rp * 2 + 1][cp * 2] = tile[2];
            out[rp * 2 + 1][cp * 2 + 1] = tile[3];
        }
    }
    out
}

fn valid_q8_rows(rows: &[&[u8]; 8], act_blocks: usize) -> bool {
    let Some(expected) = act_blocks.checked_mul(Q8_0_BLOCK_BYTES) else {
        return false;
    };
    rows.iter().all(|row| row.len() == expected)
}

/// Computes eight `Q8_0` row dot products while sharing activations per block (`0-ULP` vs `dot_q8_0_row`).
pub(crate) fn dot_q8_0_8rows(
    caps: Arm64QuantCaps,
    rows: [&[u8]; 8],
    act: &[Q8_0ActBlock],
) -> [f32; 8] {
    if !valid_q8_rows(&rows, act.len()) {
        return [0.0; 8];
    }
    #[cfg(target_arch = "aarch64")]
    if caps.dotprod && bonsai_neon_dot::has_dotprod() {
        // SAFETY: `caps.dotprod` and `bonsai_neon_dot::has_dotprod()` verified CPU support for `neon,dotprod`.
        return unsafe { dot_q8_0_8rows_dotprod(rows, act) };
    }
    if caps.dotprod {
        let lo = dot_q8_0_4rows(caps, [rows[0], rows[1], rows[2], rows[3]], act);
        let hi = dot_q8_0_4rows(caps, [rows[4], rows[5], rows[6], rows[7]], act);
        return [lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]];
    }
    rows.map(|row| dot_q8_0_row(caps, row, act))
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
fn load_f16_lane<const LANE: i32>(blk: &[u8; 34], vec: uint16x8_t) -> uint16x8_t {
    // SAFETY: `blk` is a 34-byte Q8_0 block (`>= 2` bytes) and ARM64 `ld1` lane load requires no alignment.
    unsafe { vld1q_lane_u16::<LANE>(blk.as_ptr().cast::<u16>(), vec) }
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon,dotprod")]
fn q8_0_8rows_finite_scales_block_prod(
    b: [&[u8; 34]; 8],
    ab: &Q8_0ActBlock,
    zero_u16: uint16x8_t,
    zero_i32: int32x4_t,
) -> (float32x4_t, float32x4_t) {
    let mut d_raw = load_f16_lane::<0>(b[0], zero_u16);
    d_raw = load_f16_lane::<1>(b[1], d_raw);
    d_raw = load_f16_lane::<2>(b[2], d_raw);
    d_raw = load_f16_lane::<3>(b[3], d_raw);
    d_raw = load_f16_lane::<4>(b[4], d_raw);
    d_raw = load_f16_lane::<5>(b[5], d_raw);
    d_raw = load_f16_lane::<6>(b[6], d_raw);
    d_raw = load_f16_lane::<7>(b[7], d_raw);
    let d_half = vreinterpretq_f16_u16(d_raw);
    let w_scale_lo = vcvt_f32_f16(vget_low_f16(d_half));
    let w_scale_hi = vcvt_high_f32_f16(d_half);
    let act_scale = vdupq_n_f32(ab.d);

    let a_halves = ab.qs.as_chunks::<16>().0;
    let a0 = load_s8x16(&a_halves[0]);
    let a1 = load_s8x16(&a_halves[1]);

    let d0 = sdot_q8_block(b[0], a0, a1, zero_i32);
    let d1 = sdot_q8_block(b[1], a0, a1, zero_i32);
    let d2 = sdot_q8_block(b[2], a0, a1, zero_i32);
    let d3 = sdot_q8_block(b[3], a0, a1, zero_i32);
    let dots_lo = vpaddq_s32(vpaddq_s32(d0, d1), vpaddq_s32(d2, d3));

    let d4 = sdot_q8_block(b[4], a0, a1, zero_i32);
    let d5 = sdot_q8_block(b[5], a0, a1, zero_i32);
    let d6 = sdot_q8_block(b[6], a0, a1, zero_i32);
    let d7 = sdot_q8_block(b[7], a0, a1, zero_i32);
    let dots_hi = vpaddq_s32(vpaddq_s32(d4, d5), vpaddq_s32(d6, d7));

    let dot_f32_lo = vcvtq_f32_s32(dots_lo);
    let dot_f32_hi = vcvtq_f32_s32(dots_hi);

    let prod_lo = vmulq_f32(w_scale_lo, vmulq_f32(act_scale, dot_f32_lo));
    let prod_hi = vmulq_f32(w_scale_hi, vmulq_f32(act_scale, dot_f32_hi));
    (prod_lo, prod_hi)
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon,dotprod")]
fn dot_q8_0_8rows_finite_scales_inner(
    r_chunks: [&[[u8; 34]]; 8],
    next_r_chunks: Option<[&[[u8; 34]]; 8]>,
    act: &[Q8_0ActBlock],
) -> [f32; 8] {
    let blocks = act.len();
    let mut acc_lo = vdupq_n_f32(0.0_f32);
    let mut acc_hi = vdupq_n_f32(0.0_f32);
    let zero_i32 = vdupq_n_s32(0);
    let zero_u16 = vdupq_n_u16(0);

    let mut idx = 0_usize;
    while idx + 1 < blocks {
        if idx + 4 < blocks {
            for r in &r_chunks {
                prefetch_l1(r[idx + 4].as_ptr());
            }
        } else if let Some(next) = next_r_chunks {
            let next_idx = if blocks < 4 { 0 } else { idx + 4 - blocks };
            if next_idx < blocks {
                for r in &next {
                    prefetch_l1(r[next_idx].as_ptr());
                }
            }
        }
        let b_even: [&[u8; 34]; 8] = std::array::from_fn(|r| &r_chunks[r][idx]);
        let b_odd: [&[u8; 34]; 8] = std::array::from_fn(|r| &r_chunks[r][idx + 1]);
        let (p0_lo, p0_hi) =
            q8_0_8rows_finite_scales_block_prod(b_even, &act[idx], zero_u16, zero_i32);
        let (p1_lo, p1_hi) =
            q8_0_8rows_finite_scales_block_prod(b_odd, &act[idx + 1], zero_u16, zero_i32);
        acc_lo = vaddq_f32(acc_lo, p0_lo);
        acc_hi = vaddq_f32(acc_hi, p0_hi);
        acc_lo = vaddq_f32(acc_lo, p1_lo);
        acc_hi = vaddq_f32(acc_hi, p1_hi);
        idx += 2;
    }
    if idx < blocks {
        if let Some(next) = next_r_chunks
            && blocks <= 4
        {
            for r in &next {
                prefetch_l1(r[0].as_ptr());
            }
        }
        let b_last: [&[u8; 34]; 8] = std::array::from_fn(|r| &r_chunks[r][idx]);
        let (p_lo, p_hi) =
            q8_0_8rows_finite_scales_block_prod(b_last, &act[idx], zero_u16, zero_i32);
        acc_lo = vaddq_f32(acc_lo, p_lo);
        acc_hi = vaddq_f32(acc_hi, p_hi);
    }

    let mut out_lo = [0.0_f32; 4];
    let mut out_hi = [0.0_f32; 4];
    store_f32x4(acc_lo, &mut out_lo);
    store_f32x4(acc_hi, &mut out_hi);
    [
        out_lo[0], out_lo[1], out_lo[2], out_lo[3], out_hi[0], out_hi[1], out_hi[2], out_hi[3],
    ]
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
fn dot_q8_0_8rows_finite_scales_dotprod(rows: [&[u8]; 8], act: &[Q8_0ActBlock]) -> [f32; 8] {
    let r_chunks: [&[[u8; 34]]; 8] = rows.map(|r| r.as_chunks::<34>().0);
    dot_q8_0_8rows_finite_scales_inner(r_chunks, None, act)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
fn dot_q8_0_8rows_finite_scales_strips_dotprod(
    data: &[u8],
    start_row: usize,
    strip_count: usize,
    act: &[Q8_0ActBlock],
    out: &mut [f32],
) {
    let row_bytes = act.len() * Q8_0_BLOCK_BYTES;
    for strip_idx in 0..strip_count {
        let base_row = start_row + strip_idx * 8;
        let r_chunks: [&[[u8; 34]]; 8] = std::array::from_fn(|r| {
            let off = (base_row + r) * row_bytes;
            data[off..off + row_bytes].as_chunks::<34>().0
        });
        let next_r_chunks = (strip_idx + 1 < strip_count).then(|| {
            std::array::from_fn(|r| {
                let off = (base_row + 8 + r) * row_bytes;
                data[off..off + row_bytes].as_chunks::<34>().0
            })
        });
        let vals = dot_q8_0_8rows_finite_scales_inner(r_chunks, next_r_chunks, act);
        out[strip_idx * 8..(strip_idx + 1) * 8].copy_from_slice(&vals);
    }
}

/// Computes eight load-validated finite-scale `Q8_0` row dot products using 2-block unrolling
/// and direct `vld1q_lane_u16` + `vcvt_f32_f16` (`0-ULP` vs `dot_q8_0_row`).
///
/// Callers must only invoke this on `Q8_0` rows whose `f16` scales were proven finite at load time;
/// arbitrary unvalidated `f16` inputs should use [`dot_q8_0_8rows`].
pub(crate) fn dot_q8_0_8rows_finite_scales(
    caps: Arm64QuantCaps,
    rows: [&[u8]; 8],
    act: &[Q8_0ActBlock],
) -> [f32; 8] {
    if !valid_q8_rows(&rows, act.len()) {
        return [0.0; 8];
    }
    #[cfg(target_arch = "aarch64")]
    if caps.dotprod && bonsai_neon_dot::has_dotprod() {
        // SAFETY: `caps.dotprod` and `bonsai_neon_dot::has_dotprod()` verified CPU support for `neon,dotprod`.
        return unsafe { dot_q8_0_8rows_finite_scales_dotprod(rows, act) };
    }
    dot_q8_0_8rows(caps, rows, act)
}

/// Computes `strip_count` contiguous 8-row strips (`strip_count * 8` rows starting at `start_row`)
/// of load-validated finite-scale `Q8_0` dot products with cross-strip prefetching (`0-ULP` vs `dot_q8_0_row`).
pub(crate) fn dot_q8_0_8rows_finite_scales_strips(
    caps: Arm64QuantCaps,
    data: &[u8],
    start_row: usize,
    strip_count: usize,
    act: &[Q8_0ActBlock],
    out: &mut [f32],
) {
    if strip_count == 0 {
        return;
    }
    let Some(total_rows) = strip_count.checked_mul(8) else {
        return;
    };
    let Some(end_row) = start_row.checked_add(total_rows) else {
        return;
    };
    let Some(row_bytes) = act.len().checked_mul(Q8_0_BLOCK_BYTES) else {
        return;
    };
    let Some(required_bytes) = end_row.checked_mul(row_bytes) else {
        return;
    };
    if out.len() < total_rows || data.len() < required_bytes {
        return;
    }
    if act.is_empty() {
        out[..total_rows].fill(0.0);
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if caps.dotprod && bonsai_neon_dot::has_dotprod() {
        // SAFETY: `caps.dotprod` and `bonsai_neon_dot::has_dotprod()` verified CPU support for `neon,dotprod`,
        // and all row/byte/output bounds were validated above.
        unsafe {
            dot_q8_0_8rows_finite_scales_strips_dotprod(data, start_row, strip_count, act, out);
        }
        return;
    }
    for strip_idx in 0..strip_count {
        let base_row = start_row + strip_idx * 8;
        let rows: [&[u8]; 8] = std::array::from_fn(|r| {
            let off = (base_row + r) * row_bytes;
            &data[off..off + row_bytes]
        });
        let vals = dot_q8_0_8rows(caps, rows, act);
        out[strip_idx * 8..(strip_idx + 1) * 8].copy_from_slice(&vals);
    }
}

fn gemm_q8_0_8x_dispatch<const TOKENS: usize, const COL_PAIRS: usize>(
    caps: Arm64QuantCaps,
    rows: [&[u8]; 8],
    acts: [&[Q8_0ActBlock]; TOKENS],
) -> [[f32; TOKENS]; 8] {
    let blocks = acts[0].len();
    if acts.iter().any(|act| act.len() != blocks) || !valid_q8_rows(&rows, blocks) {
        return [[0.0; TOKENS]; 8];
    }
    #[cfg(target_arch = "aarch64")]
    if caps.i8mm && bonsai_neon_dot::has_i8mm() {
        // SAFETY: `caps.i8mm` and `bonsai_neon_dot::has_i8mm()` verified CPU support for `neon,i8mm`.
        return unsafe { gemm_q8_0_8x_i8mm::<TOKENS, COL_PAIRS>(rows, acts) };
    }
    let cols: [[f32; 8]; TOKENS] = std::array::from_fn(|t| dot_q8_0_8rows(caps, rows, acts[t]));
    std::array::from_fn(|r| std::array::from_fn(|t| cols[t][r]))
}

/// Computes an 8-row by 8-token `Q8_0` tile using `I8MM` (`SMMLA`) or exact `SDOT` fallback.
pub(crate) fn gemm_q8_0_8x8(
    caps: Arm64QuantCaps,
    rows: [&[u8]; 8],
    acts: [&[Q8_0ActBlock]; 8],
) -> [[f32; 8]; 8] {
    gemm_q8_0_8x_dispatch::<8, 4>(caps, rows, acts)
}

/// Computes an 8-row by 4-token `Q8_0` tile using `I8MM` (`SMMLA`) or exact `SDOT` fallback.
pub(crate) fn gemm_q8_0_8x4(
    caps: Arm64QuantCaps,
    rows: [&[u8]; 8],
    acts: [&[Q8_0ActBlock]; 4],
) -> [[f32; 4]; 8] {
    gemm_q8_0_8x_dispatch::<4, 2>(caps, rows, acts)
}

/// Computes an 8-row by 2-token `Q8_0` tile using `I8MM` (`SMMLA`) or exact `SDOT` fallback.
pub(crate) fn gemm_q8_0_8x2(
    caps: Arm64QuantCaps,
    rows: [&[u8]; 8],
    acts: [&[Q8_0ActBlock]; 2],
) -> [[f32; 2]; 8] {
    gemm_q8_0_8x_dispatch::<2, 1>(caps, rows, acts)
}

/// Pre-interleaved 8-token `Q8_0` activation panel for `I8MM` (`SMMLA`) GEMM.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default)]
pub struct Q8_0ActPanel8 {
    /// Paired token activation scales `[d0, d1, d0, d1]` for each column pair `0..4`.
    pub d_pairs: [[f32; 4]; 4],
    /// Pre-interleaved signed activation bytes `[half][col_pair][lo_hi][16]` for `SMMLA`.
    pub qs: [[[[i8; 16]; 2]; 4]; 2],
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[target_feature(enable = "neon")]
fn pack_q8_0_act_panels8_neon(
    acts: &[Q8_0ActBlock],
    blocks: usize,
    panel_count: usize,
    panels: &mut [Q8_0ActPanel8],
) {
    for p in 0..panel_count {
        let panel_dst = &mut panels[p * blocks..(p + 1) * blocks];
        let tok_base = p * 8;
        for (k, dst) in panel_dst.iter_mut().enumerate() {
            for cp in 0..4 {
                let a_even = &acts[(tok_base + cp * 2) * blocks + k];
                let a_odd = &acts[(tok_base + cp * 2 + 1) * blocks + k];
                dst.d_pairs[cp] = [a_even.d, a_odd.d, a_even.d, a_odd.d];
                let e_halves = a_even.qs.as_chunks::<16>().0;
                let o_halves = a_odd.qs.as_chunks::<16>().0;
                for half in 0..2 {
                    let e = load_s8x16(&e_halves[half]);
                    let o = load_s8x16(&o_halves[half]);
                    store_s8x16(
                        vcombine_s8(vget_low_s8(e), vget_low_s8(o)),
                        &mut dst.qs[half][cp][0],
                    );
                    store_s8x16(
                        vcombine_s8(vget_high_s8(e), vget_high_s8(o)),
                        &mut dst.qs[half][cp][1],
                    );
                }
            }
        }
    }
}

#[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
fn pack_q8_0_act_panels8_portable(
    acts: &[Q8_0ActBlock],
    blocks: usize,
    panel_count: usize,
    panels: &mut [Q8_0ActPanel8],
) {
    for p in 0..panel_count {
        let panel_dst = &mut panels[p * blocks..(p + 1) * blocks];
        let tok_base = p * 8;
        for (k, dst) in panel_dst.iter_mut().enumerate() {
            for cp in 0..4 {
                let a_even = &acts[(tok_base + cp * 2) * blocks + k];
                let a_odd = &acts[(tok_base + cp * 2 + 1) * blocks + k];
                dst.d_pairs[cp] = [a_even.d, a_odd.d, a_even.d, a_odd.d];
                for half in 0..2 {
                    let h = half * 16;
                    dst.qs[half][cp][0][0..8].copy_from_slice(&a_even.qs[h..h + 8]);
                    dst.qs[half][cp][0][8..16].copy_from_slice(&a_odd.qs[h..h + 8]);
                    dst.qs[half][cp][1][0..8].copy_from_slice(&a_even.qs[h + 8..h + 16]);
                    dst.qs[half][cp][1][8..16].copy_from_slice(&a_odd.qs[h + 8..h + 16]);
                }
            }
        }
    }
}

/// Pre-packs `panel_count * 8` tokens of `Q8_0ActBlock`s (`acts`) into `Q8_0ActPanel8`s
/// laid out as `[panel_idx * blocks + k]`.
pub(crate) fn pack_q8_0_act_panels8(
    acts: &[Q8_0ActBlock],
    blocks: usize,
    panel_count: usize,
    panels: &mut [Q8_0ActPanel8],
) {
    if blocks == 0 || panel_count == 0 {
        return;
    }
    let Some(min_acts) = panel_count
        .checked_mul(8)
        .and_then(|t| t.checked_mul(blocks))
    else {
        return;
    };
    let Some(needed_panels) = panel_count.checked_mul(blocks) else {
        return;
    };
    if acts.len() < min_acts || panels.len() < needed_panels {
        return;
    }
    let panels = &mut panels[..needed_panels];
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        // SAFETY: Baseline NEON is enabled by target configuration and slice bounds were checked above.
        unsafe { pack_q8_0_act_panels8_neon(acts, blocks, panel_count, panels) };
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        pack_q8_0_act_panels8_portable(acts, blocks, panel_count, panels);
    }
}

#[cfg(target_arch = "aarch64")]
#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct UnpackedQ8WeightBlock {
    d_w_pairs: [[f32; 4]; 4],
    qs: [[[[i8; 16]; 2]; 2]; 4],
}

#[cfg(target_arch = "aarch64")]
impl UnpackedQ8WeightBlock {
    const ZERO: Self = Self {
        d_w_pairs: [[0.0; 4]; 4],
        qs: [[[[0; 16]; 2]; 2]; 4],
    };
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,i8mm")]
fn gemm_q8_0_8x_panels_i8mm(
    rows: [&[u8]; 8],
    panels: &[Q8_0ActPanel8],
    panel_count: usize,
    blocks: usize,
    out_rows: &mut [f32],
    row_stride: usize,
) {
    const K_TILE: usize = 32;
    const T_PANEL: usize = 8;

    let r_chunks: [&[[u8; 34]]; 8] = rows.map(|r| r.as_chunks::<34>().0);
    // 32 * 320 B = 10 KiB L1D-resident weight tile + 2 KiB accumulator buffer.
    let mut w_tile = [UnpackedQ8WeightBlock::ZERO; K_TILE];
    let mut acc_buf = [[[[0.0_f32; 4]; 4]; 4]; T_PANEL];

    for p_base in (0..panel_count).step_by(T_PANEL) {
        let p_len = (panel_count - p_base).min(T_PANEL);
        for slot in &mut acc_buf[..p_len] {
            *slot = [[[0.0_f32; 4]; 4]; 4];
        }
        for k_base in (0..blocks).step_by(K_TILE) {
            let k_len = (blocks - k_base).min(K_TILE);
            for (k_local, wb) in w_tile[..k_len].iter_mut().enumerate() {
                let idx = k_base + k_local;
                if idx + 2 < blocks {
                    for r in &r_chunks {
                        prefetch_l1(r[idx + 2].as_ptr());
                    }
                }
                let blks: [&[u8; 34]; 8] = std::array::from_fn(|r| &r_chunks[r][idx]);
                let d_bits: [u16; 8] = blks.map(|b| u16::from_le_bytes([b[0], b[1]]));
                let [ws_lo_arr, ws_hi_arr] = bonsai_neon_dot::widen_f16_eight(&d_bits);
                let ws_lo = load_f32x4(&ws_lo_arr);
                let ws_hi = load_f32x4(&ws_hi_arr);
                store_f32x4(vzip1q_f32(ws_lo, ws_lo), &mut wb.d_w_pairs[0]);
                store_f32x4(vzip2q_f32(ws_lo, ws_lo), &mut wb.d_w_pairs[1]);
                store_f32x4(vzip1q_f32(ws_hi, ws_hi), &mut wb.d_w_pairs[2]);
                store_f32x4(vzip2q_f32(ws_hi, ws_hi), &mut wb.d_w_pairs[3]);

                for rp in 0..4 {
                    let halves0 = blks[rp * 2][2..34].as_chunks::<16>().0;
                    let halves1 = blks[rp * 2 + 1][2..34].as_chunks::<16>().0;
                    for half in 0..2 {
                        let w_even = vreinterpretq_s8_u8(load_u8x16(&halves0[half]));
                        let w_odd = vreinterpretq_s8_u8(load_u8x16(&halves1[half]));
                        store_s8x16(
                            vcombine_s8(vget_low_s8(w_even), vget_low_s8(w_odd)),
                            &mut wb.qs[rp][half][0],
                        );
                        store_s8x16(
                            vcombine_s8(vget_high_s8(w_even), vget_high_s8(w_odd)),
                            &mut wb.qs[rp][half][1],
                        );
                    }
                }
            }

            for (p_local, p_acc) in acc_buf[..p_len].iter_mut().enumerate() {
                let p_idx = p_base + p_local;
                let p_panels = &panels[p_idx * blocks + k_base..p_idx * blocks + k_base + k_len];
                let mut acc: [[float32x4_t; 4]; 4] =
                    std::array::from_fn(|rp| std::array::from_fn(|cp| load_f32x4(&p_acc[rp][cp])));
                for (wb, ap) in w_tile[..k_len].iter().zip(p_panels.iter()) {
                    let d_act = [
                        load_f32x4(&ap.d_pairs[0]),
                        load_f32x4(&ap.d_pairs[1]),
                        load_f32x4(&ap.d_pairs[2]),
                        load_f32x4(&ap.d_pairs[3]),
                    ];
                    for rp in 0..4 {
                        let mut c_rp = [vdupq_n_s32(0); 4];
                        for half in 0..2 {
                            let w_lo = load_s8x16(&wb.qs[rp][half][0]);
                            let w_hi = load_s8x16(&wb.qs[rp][half][1]);
                            for cp in 0..4 {
                                let a_lo = load_s8x16(&ap.qs[half][cp][0]);
                                let a_hi = load_s8x16(&ap.qs[half][cp][1]);
                                c_rp[cp] = smmla_s32(smmla_s32(c_rp[cp], w_lo, a_lo), w_hi, a_hi);
                            }
                        }
                        let d_w_v = load_f32x4(&wb.d_w_pairs[rp]);
                        for cp in 0..4 {
                            let dot_f32 = vcvtq_f32_s32(c_rp[cp]);
                            let prod = vmulq_f32(d_w_v, vmulq_f32(d_act[cp], dot_f32));
                            acc[rp][cp] = vaddq_f32(acc[rp][cp], prod);
                        }
                    }
                }
                for (rp, pair_acc) in acc.iter().enumerate() {
                    for (cp, &vec) in pair_acc.iter().enumerate() {
                        store_f32x4(vec, &mut p_acc[rp][cp]);
                    }
                }
            }
        }

        for (p_local, p_acc) in acc_buf[..p_len].iter().enumerate() {
            let t_off = (p_base + p_local) * 8;
            for (rp, pair_acc) in p_acc.iter().enumerate() {
                let r0 = rp * 2;
                let r1 = rp * 2 + 1;
                for (cp, tile) in pair_acc.iter().enumerate() {
                    let c0 = cp * 2;
                    let c1 = cp * 2 + 1;
                    out_rows[r0 * row_stride + t_off + c0] = tile[0];
                    out_rows[r0 * row_stride + t_off + c1] = tile[1];
                    out_rows[r1 * row_stride + t_off + c0] = tile[2];
                    out_rows[r1 * row_stride + t_off + c1] = tile[3];
                }
            }
        }
    }
}

/// Computes an 8-row by `(panel_count * 8)`-token `Q8_0` GEMM over pre-packed `Q8_0ActPanel8`s
/// with K-blocked L1D weight tiles (`I8MM` `SMMLA`, or exact `SDOT`/portable fallback).
pub(crate) fn gemm_q8_0_8x_panels(
    caps: Arm64QuantCaps,
    rows: [&[u8]; 8],
    panels: &[Q8_0ActPanel8],
    panel_count: usize,
    blocks: usize,
    out_rows: &mut [f32],
    row_stride: usize,
) {
    if panel_count == 0 {
        return;
    }
    let Some(tokens) = panel_count.checked_mul(8) else {
        return;
    };
    let Some(expected_panels) = panel_count.checked_mul(blocks) else {
        return;
    };
    let Some(min_out) = row_stride
        .checked_mul(7)
        .and_then(|v| v.checked_add(tokens))
    else {
        return;
    };
    if row_stride < tokens
        || out_rows.len() < min_out
        || panels.len() != expected_panels
        || !valid_q8_rows(&rows, blocks)
    {
        return;
    }
    if blocks == 0 {
        for r in 0..8 {
            out_rows[r * row_stride..r * row_stride + tokens].fill(0.0);
        }
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if caps.i8mm && bonsai_neon_dot::has_i8mm() {
        // SAFETY: `caps.i8mm` and `bonsai_neon_dot::has_i8mm()` verified CPU support for `neon,i8mm`,
        // and all slice lengths and row strides were validated above.
        unsafe {
            gemm_q8_0_8x_panels_i8mm(rows, panels, panel_count, blocks, out_rows, row_stride);
        }
        return;
    }

    let mut unpacked = vec![Q8_0ActBlock::default(); 8 * blocks];
    for p in 0..panel_count {
        let p_slice = &panels[p * blocks..(p + 1) * blocks];
        for (k, ap) in p_slice.iter().enumerate() {
            for cp in 0..4 {
                let t_even = cp * 2;
                let t_odd = cp * 2 + 1;
                {
                    let dst_even = &mut unpacked[t_even * blocks + k];
                    dst_even.d = ap.d_pairs[cp][0];
                    for half in 0..2 {
                        let h = half * 16;
                        dst_even.qs[h..h + 8].copy_from_slice(&ap.qs[half][cp][0][0..8]);
                        dst_even.qs[h + 8..h + 16].copy_from_slice(&ap.qs[half][cp][1][0..8]);
                    }
                }
                let dst_odd = &mut unpacked[t_odd * blocks + k];
                dst_odd.d = ap.d_pairs[cp][1];
                for half in 0..2 {
                    let h = half * 16;
                    dst_odd.qs[h..h + 8].copy_from_slice(&ap.qs[half][cp][0][8..16]);
                    dst_odd.qs[h + 8..h + 16].copy_from_slice(&ap.qs[half][cp][1][8..16]);
                }
            }
        }
        let t_off = p * 8;
        for t in 0..8 {
            let col = dot_q8_0_8rows(caps, rows, &unpacked[t * blocks..(t + 1) * blocks]);
            for (r, &val) in col.iter().enumerate() {
                out_rows[r * row_stride + t_off + t] = val;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn load_s32x4(values: &[i32; 4]) -> int32x4_t {
    // SAFETY: `values` is a valid 4-element `i32` array reference and `vld1q_s32` requires no alignment.
    unsafe { vld1q_s32(values.as_ptr()) }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn store_s32x4(vec: int32x4_t, out: &mut [i32; 4]) {
    // SAFETY: `out` is a valid mutable 4-element `i32` array reference and `vst1q_s32` requires no alignment.
    unsafe { vst1q_s32(out.as_mut_ptr(), vec) };
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn load_f64x2(values: &[f64; 2]) -> float64x2_t {
    // SAFETY: `values` is a valid 2-element `f64` array reference and `vld1q_f64` requires no alignment.
    unsafe { vld1q_f64(values.as_ptr()) }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn store_f64x2(vec: float64x2_t, out: &mut [f64; 2]) {
    // SAFETY: `out` is a valid mutable 2-element `f64` array reference and `vst1q_f64` requires no alignment.
    unsafe { vst1q_f64(out.as_mut_ptr(), vec) };
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn load_u16x8(bytes: &[u8; 16]) -> uint16x8_t {
    // SAFETY: `bytes` is a valid 16-byte array reference and `vld1q_u16` requires no alignment.
    unsafe { vld1q_u16(bytes.as_ptr().cast::<u16>()) }
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
fn s32x4_to_f64x2_pair(vec: int32x4_t) -> (float64x2_t, float64x2_t) {
    (
        vcvtq_f64_s64(vmovl_s32(vget_low_s32(vec))),
        vcvtq_f64_s64(vmovl_high_s32(vec)),
    )
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
fn store_f64x2_pairs_as_f32x8(acc: &[[float64x2_t; 4]; 8], out_rows: &mut [f32], row_stride: usize, t_off: usize) {
    for (r, row_acc) in acc.iter().enumerate() {
        let base = r * row_stride + t_off;
        let lo = vcombine_f32(vcvt_f32_f64(row_acc[0]), vcvt_f32_f64(row_acc[1]));
        let hi = vcombine_f32(vcvt_f32_f64(row_acc[2]), vcvt_f32_f64(row_acc[3]));
        let dst = out_rows[base..base + 8].as_chunks_mut::<4>().0;
        store_f32x4(lo, &mut dst[0]);
        store_f32x4(hi, &mut dst[1]);
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
fn widen_u8x8_to_s32x4_pair(bytes: &[u8; 8]) -> (int32x4_t, int32x4_t) {
    let v_u16 = vmovl_u8(load_u8x8(bytes));
    (
        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v_u16))),
        vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(v_u16))),
    )
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
fn accumulate_q4_k_mint_vec(m_bytes: &[u8; 64], bsums: &[i16; 16]) -> (int32x4_t, int32x4_t) {
    let mut mint_lo = vdupq_n_s32(0);
    let mut mint_hi = vdupq_n_s32(0);
    for g in 0..8 {
        let m_u8 = load_u8x8(&m_bytes[g * 8..(g + 1) * 8].as_chunks::<8>().0[0]);
        let m_u16 = vmovl_u8(m_u8);
        let b_pair_i32 = i32::from(bsums[2 * g]) + i32::from(bsums[2 * g + 1]);
        if let Ok(b_pair) = i16::try_from(b_pair_i32) {
            let m_s16 = vreinterpretq_s16_u16(m_u16);
            let b_half = vdup_n_s16(b_pair);
            let b_full = vcombine_s16(b_half, b_half);
            mint_lo = vmlal_s16(mint_lo, vget_low_s16(m_s16), b_half);
            mint_hi = vmlal_high_s16(mint_hi, m_s16, b_full);
        } else {
            let (m_lo, m_hi) = (
                vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(m_u16))),
                vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(m_u16))),
            );
            let b_vec = vdupq_n_s32(b_pair_i32);
            mint_lo = vmlaq_s32(mint_lo, m_lo, b_vec);
            mint_hi = vmlaq_s32(mint_hi, m_hi, b_vec);
        }
    }
    (mint_lo, mint_hi)
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
fn accumulate_q4_k_mint(m_bytes: &[u8; 64], bsums: &[i16; 16]) -> ([i32; 4], [i32; 4]) {
    let (mint_lo, mint_hi) = accumulate_q4_k_mint_vec(m_bytes, bsums);
    let mut out_lo = [0_i32; 4];
    let mut out_hi = [0_i32; 4];
    store_s32x4(mint_lo, &mut out_lo);
    store_s32x4(mint_hi, &mut out_hi);
    (out_lo, out_hi)
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon,dotprod")]
fn q4_k_repacked8_block_sumi_dotprod(
    block: &[u8; 1184],
    ab: &Q8_KActBlock,
    mask_0f: uint8x16_t,
    zero: int32x4_t,
) -> (int32x4_t, int32x4_t) {
    let sc_bytes = &block[32..96];
    let a_groups = ab.qs.as_chunks::<16>().0;
    let mut sumi_lo = zero;
    let mut sumi_hi = zero;
    for chunk in 0..4 {
        let a_lo0 = load_s8x16(&a_groups[chunk * 4]);
        let a_lo1 = load_s8x16(&a_groups[chunk * 4 + 1]);
        let a_hi0 = load_s8x16(&a_groups[chunk * 4 + 2]);
        let a_hi1 = load_s8x16(&a_groups[chunk * 4 + 3]);
        let qs_base = 160 + chunk * 256;
        let (sc_lo_03, sc_lo_47) = widen_u8x8_to_s32x4_pair(
            &sc_bytes[(chunk * 2) * 8..(chunk * 2 + 1) * 8]
                .as_chunks::<8>()
                .0[0],
        );
        let (sc_hi_03, sc_hi_47) = widen_u8x8_to_s32x4_pair(
            &sc_bytes[(chunk * 2 + 1) * 8..(chunk * 2 + 2) * 8]
                .as_chunks::<8>()
                .0[0],
        );

        let mut d_lo = [zero; 4];
        let mut d_hi = [zero; 4];
        for r in 0..4 {
            let row_qs = qs_base + r * 32;
            let raw0 = load_u8x16(&block[row_qs..row_qs + 16].as_chunks::<16>().0[0]);
            let raw1 = load_u8x16(&block[row_qs + 16..row_qs + 32].as_chunks::<16>().0[0]);
            let lo0 = vreinterpretq_s8_u8(vandq_u8(raw0, mask_0f));
            let lo1 = vreinterpretq_s8_u8(vandq_u8(raw1, mask_0f));
            let hi0 = vreinterpretq_s8_u8(vshrq_n_u8(raw0, 4));
            let hi1 = vreinterpretq_s8_u8(vshrq_n_u8(raw1, 4));
            d_lo[r] = vdotq_s32(vdotq_s32(zero, lo0, a_lo0), lo1, a_lo1);
            d_hi[r] = vdotq_s32(vdotq_s32(zero, hi0, a_hi0), hi1, a_hi1);
        }
        let dots_lo_03 = vpaddq_s32(vpaddq_s32(d_lo[0], d_lo[1]), vpaddq_s32(d_lo[2], d_lo[3]));
        let dots_hi_03 = vpaddq_s32(vpaddq_s32(d_hi[0], d_hi[1]), vpaddq_s32(d_hi[2], d_hi[3]));
        sumi_lo = vmlaq_s32(sumi_lo, dots_lo_03, sc_lo_03);
        sumi_lo = vmlaq_s32(sumi_lo, dots_hi_03, sc_hi_03);

        for r in 0..4 {
            let row_qs = qs_base + (r + 4) * 32;
            let raw0 = load_u8x16(&block[row_qs..row_qs + 16].as_chunks::<16>().0[0]);
            let raw1 = load_u8x16(&block[row_qs + 16..row_qs + 32].as_chunks::<16>().0[0]);
            let lo0 = vreinterpretq_s8_u8(vandq_u8(raw0, mask_0f));
            let lo1 = vreinterpretq_s8_u8(vandq_u8(raw1, mask_0f));
            let hi0 = vreinterpretq_s8_u8(vshrq_n_u8(raw0, 4));
            let hi1 = vreinterpretq_s8_u8(vshrq_n_u8(raw1, 4));
            d_lo[r] = vdotq_s32(vdotq_s32(zero, lo0, a_lo0), lo1, a_lo1);
            d_hi[r] = vdotq_s32(vdotq_s32(zero, hi0, a_hi0), hi1, a_hi1);
        }
        let dots_lo_47 = vpaddq_s32(vpaddq_s32(d_lo[0], d_lo[1]), vpaddq_s32(d_lo[2], d_lo[3]));
        let dots_hi_47 = vpaddq_s32(vpaddq_s32(d_hi[0], d_hi[1]), vpaddq_s32(d_hi[2], d_hi[3]));
        sumi_hi = vmlaq_s32(sumi_hi, dots_lo_47, sc_lo_47);
        sumi_hi = vmlaq_s32(sumi_hi, dots_hi_47, sc_hi_47);
    }
    (sumi_lo, sumi_hi)
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn prefetch_q4_k_repacked8_block(block: &[u8; 1184]) {
    let ptr = block.as_ptr();
    for off in [0_usize, 256, 512, 768, 1024] {
        prefetch_l1(ptr.wrapping_add(off));
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
fn dot_q4_k_repacked8_dotprod(strip: &[u8], act: &[Q8_KActBlock]) -> [f32; 8] {
    let blocks = strip.as_chunks::<1184>().0;
    let total_blocks = blocks.len();
    let mut acc = [0.0_f64; 8];
    let mask_0f = vdupq_n_u8(0x0f);
    let zero = vdupq_n_s32(0);

    for (blk_idx, (block, ab)) in blocks.iter().zip(act.iter()).enumerate() {
        if blk_idx + 1 < total_blocks {
            prefetch_q4_k_repacked8_block(&blocks[blk_idx + 1]);
        }
        let d_f16: [u16; 8] =
            std::array::from_fn(|r| u16::from_le_bytes([block[r * 2], block[r * 2 + 1]]));
        let dmin_f16: [u16; 8] =
            std::array::from_fn(|r| u16::from_le_bytes([block[16 + r * 2], block[16 + r * 2 + 1]]));
        let m_bytes: &[u8; 64] = &block[96..160].as_chunks::<64>().0[0];
        let (mint_lo_arr, mint_hi_arr) = accumulate_q4_k_mint(m_bytes, &ab.bsums);
        let mint = [
            mint_lo_arr[0],
            mint_lo_arr[1],
            mint_lo_arr[2],
            mint_lo_arr[3],
            mint_hi_arr[0],
            mint_hi_arr[1],
            mint_hi_arr[2],
            mint_hi_arr[3],
        ];
        let (sumi_lo, sumi_hi) = q4_k_repacked8_block_sumi_dotprod(block, ab, mask_0f, zero);

        let mut s_lo = [0_i32; 4];
        let mut s_hi = [0_i32; 4];
        store_s32x4(sumi_lo, &mut s_lo);
        store_s32x4(sumi_hi, &mut s_hi);
        let sumi = [
            s_lo[0], s_lo[1], s_lo[2], s_lo[3], s_hi[0], s_hi[1], s_hi[2], s_hi[3],
        ];

        let d_act = f64::from(ab.d);
        let d_w = bonsai_neon_dot::widen_f16_eight(&d_f16);
        let dmin_w = bonsai_neon_dot::widen_f16_eight(&dmin_f16);
        for r in 0..8 {
            let dw = f64::from(d_w[r / 4][r % 4]) * d_act;
            let dm = f64::from(dmin_w[r / 4][r % 4]) * d_act;
            #[expect(
                clippy::suboptimal_flops,
                reason = "Matches non-FMA repacked Q4_K block accumulation order."
            )]
            {
                acc[r] += dw * f64::from(sumi[r]) - dm * f64::from(mint[r]);
            }
        }
    }
    acc.map(f64_to_f32)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
fn dot_q4_k_repacked8_finite_scales_strips_dotprod(
    data: &[u8],
    start_strip: usize,
    strip_count: usize,
    act: &[Q8_KActBlock],
    out: &mut [f32],
) {
    let total_blocks = act.len();
    let strip_bytes = total_blocks * Q4_K_REPACK8_BLOCK_BYTES;
    let mask_0f = vdupq_n_u8(0x0f);
    let zero = vdupq_n_s32(0);

    for strip_idx in 0..strip_count {
        let strip_off = (start_strip + strip_idx) * strip_bytes;
        let blocks = data[strip_off..strip_off + strip_bytes]
            .as_chunks::<1184>()
            .0;
        let next_blocks = (strip_idx + 1 < strip_count).then(|| {
            data[strip_off + strip_bytes..strip_off + 2 * strip_bytes]
                .as_chunks::<1184>()
                .0
        });

        let mut acc01 = vdupq_n_f64(0.0_f64);
        let mut acc23 = vdupq_n_f64(0.0_f64);
        let mut acc45 = vdupq_n_f64(0.0_f64);
        let mut acc67 = vdupq_n_f64(0.0_f64);

        for (blk_idx, (block, ab)) in blocks.iter().zip(act.iter()).enumerate() {
            if blk_idx + 1 < total_blocks {
                prefetch_q4_k_repacked8_block(&blocks[blk_idx + 1]);
            } else if let Some(next) = next_blocks {
                prefetch_q4_k_repacked8_block(&next[0]);
            }

            let d_half = vreinterpretq_f16_u16(load_u16x8(&block[0..16].as_chunks::<16>().0[0]));
            let dmin_half =
                vreinterpretq_f16_u16(load_u16x8(&block[16..32].as_chunks::<16>().0[0]));
            let d_f32_lo = vcvt_f32_f16(vget_low_f16(d_half));
            let d_f32_hi = vcvt_high_f32_f16(d_half);
            let dmin_f32_lo = vcvt_f32_f16(vget_low_f16(dmin_half));
            let dmin_f32_hi = vcvt_high_f32_f16(dmin_half);

            let m_bytes: &[u8; 64] = &block[96..160].as_chunks::<64>().0[0];
            let (mint_lo, mint_hi) = accumulate_q4_k_mint_vec(m_bytes, &ab.bsums);
            let (sumi_lo, sumi_hi) = q4_k_repacked8_block_sumi_dotprod(block, ab, mask_0f, zero);

            let (sum_01, sum_23) = s32x4_to_f64x2_pair(sumi_lo);
            let (sum_45, sum_67) = s32x4_to_f64x2_pair(sumi_hi);
            let (mint_01, mint_23) = s32x4_to_f64x2_pair(mint_lo);
            let (mint_45, mint_67) = s32x4_to_f64x2_pair(mint_hi);

            let d_act_v = vdupq_n_f64(f64::from(ab.d));

            let dw_01 = vmulq_f64(vcvt_f64_f32(vget_low_f32(d_f32_lo)), d_act_v);
            let dm_01 = vmulq_f64(vcvt_f64_f32(vget_low_f32(dmin_f32_lo)), d_act_v);
            acc01 = vaddq_f64(
                acc01,
                vsubq_f64(vmulq_f64(dw_01, sum_01), vmulq_f64(dm_01, mint_01)),
            );

            let dw_23 = vmulq_f64(vcvt_high_f64_f32(d_f32_lo), d_act_v);
            let dm_23 = vmulq_f64(vcvt_high_f64_f32(dmin_f32_lo), d_act_v);
            acc23 = vaddq_f64(
                acc23,
                vsubq_f64(vmulq_f64(dw_23, sum_23), vmulq_f64(dm_23, mint_23)),
            );

            let dw_45 = vmulq_f64(vcvt_f64_f32(vget_low_f32(d_f32_hi)), d_act_v);
            let dm_45 = vmulq_f64(vcvt_f64_f32(vget_low_f32(dmin_f32_hi)), d_act_v);
            acc45 = vaddq_f64(
                acc45,
                vsubq_f64(vmulq_f64(dw_45, sum_45), vmulq_f64(dm_45, mint_45)),
            );

            let dw_67 = vmulq_f64(vcvt_high_f64_f32(d_f32_hi), d_act_v);
            let dm_67 = vmulq_f64(vcvt_high_f64_f32(dmin_f32_hi), d_act_v);
            acc67 = vaddq_f64(
                acc67,
                vsubq_f64(vmulq_f64(dw_67, sum_67), vmulq_f64(dm_67, mint_67)),
            );
        }

        let dst_lo = vcombine_f32(vcvt_f32_f64(acc01), vcvt_f32_f64(acc23));
        let dst_hi = vcombine_f32(vcvt_f32_f64(acc45), vcvt_f32_f64(acc67));
        let out_chunks = out[strip_idx * 8..(strip_idx + 1) * 8]
            .as_chunks_mut::<4>()
            .0;
        store_f32x4(dst_lo, &mut out_chunks[0]);
        store_f32x4(dst_hi, &mut out_chunks[1]);
    }
}

/// Computes an 8-row repacked `Q4_K` strip dot product (`0-ULP` vs `dot_q4_k_row`).
pub(crate) fn dot_q4_k_repacked8(
    caps: Arm64QuantCaps,
    strip: &[u8],
    act: &[Q8_KActBlock],
) -> [f32; 8] {
    if strip.len() != act.len().saturating_mul(Q4_K_REPACK8_BLOCK_BYTES) {
        return [0.0; 8];
    }
    #[cfg(target_arch = "aarch64")]
    if caps.dotprod && bonsai_neon_dot::has_dotprod() {
        // SAFETY: `caps.dotprod` and `bonsai_neon_dot::has_dotprod()` verified CPU support for `neon,dotprod`.
        return unsafe { dot_q4_k_repacked8_dotprod(strip, act) };
    }
    bonsai_neon_dot::kquants::dot_q4_k_repacked8(caps, strip, act)
}

/// Computes `strip_count` contiguous 8-row repacked `Q4_K` strips starting at `start_strip`
/// for load-validated finite scales with cross-strip prefetching (`0-ULP` vs `dot_q4_k_row`).
pub(crate) fn dot_q4_k_repacked8_finite_scales_strips(
    caps: Arm64QuantCaps,
    data: &[u8],
    start_strip: usize,
    strip_count: usize,
    act: &[Q8_KActBlock],
    out: &mut [f32],
) {
    if strip_count == 0 {
        return;
    }
    let Some(total_rows) = strip_count.checked_mul(8) else {
        return;
    };
    let Some(end_strip) = start_strip.checked_add(strip_count) else {
        return;
    };
    let Some(strip_bytes) = act.len().checked_mul(Q4_K_REPACK8_BLOCK_BYTES) else {
        return;
    };
    let Some(required_bytes) = end_strip.checked_mul(strip_bytes) else {
        return;
    };
    if out.len() < total_rows || data.len() < required_bytes {
        return;
    }
    if act.is_empty() {
        out[..total_rows].fill(0.0);
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if caps.dotprod && bonsai_neon_dot::has_dotprod() {
        // SAFETY: `caps.dotprod` and `bonsai_neon_dot::has_dotprod()` verified CPU support for `neon,dotprod`,
        // and all strip/byte/output bounds were validated above.
        unsafe {
            dot_q4_k_repacked8_finite_scales_strips_dotprod(
                data,
                start_strip,
                strip_count,
                act,
                out,
            );
        }
        return;
    }
    for strip_idx in 0..strip_count {
        let strip_off = (start_strip + strip_idx) * strip_bytes;
        let strip = &data[strip_off..strip_off + strip_bytes];
        let vals = dot_q4_k_repacked8(caps, strip, act);
        out[strip_idx * 8..(strip_idx + 1) * 8].copy_from_slice(&vals);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,i8mm")]
fn gemm_q4_k_repacked8_8x_i8mm<const TOKENS: usize, const COL_PAIRS: usize>(
    strip: &[u8],
    acts: [&[Q8_KActBlock]; TOKENS],
) -> [[f32; TOKENS]; 8] {
    let blocks = strip.as_chunks::<1184>().0;
    let total_blocks = blocks.len();
    let mut acc = [[0.0_f64; TOKENS]; 8];
    let mask_0f = vdupq_n_u8(0x0f);

    for (blk_idx, block) in blocks.iter().enumerate() {
        if blk_idx + 1 < total_blocks {
            let next_ptr = blocks[blk_idx + 1].as_ptr();
            for off in [0_usize, 256, 512, 768, 1024] {
                prefetch_l1(next_ptr.wrapping_add(off));
            }
        }
        let d_f16: [u16; 8] =
            std::array::from_fn(|r| u16::from_le_bytes([block[r * 2], block[r * 2 + 1]]));
        let dmin_f16: [u16; 8] =
            std::array::from_fn(|r| u16::from_le_bytes([block[16 + r * 2], block[16 + r * 2 + 1]]));
        let sc_bytes = &block[32..96];
        let m_bytes = &block[96..160];

        let mut mint_lo = [vdupq_n_s32(0); TOKENS];
        let mut mint_hi = [vdupq_n_s32(0); TOKENS];
        for g in 0..8 {
            let m_u8 = load_u8x8(&m_bytes[g * 8..(g + 1) * 8].as_chunks::<8>().0[0]);
            let m_u16 = vmovl_u8(m_u8);
            let m_s16 = vreinterpretq_s16_u16(m_u16);
            for t in 0..TOKENS {
                let ab = &acts[t][blk_idx];
                let b_pair_i32 = i32::from(ab.bsums[2 * g]) + i32::from(ab.bsums[2 * g + 1]);
                if let Ok(b_pair) = i16::try_from(b_pair_i32) {
                    let b_half = vdup_n_s16(b_pair);
                    let b_full = vcombine_s16(b_half, b_half);
                    mint_lo[t] = vmlal_s16(mint_lo[t], vget_low_s16(m_s16), b_half);
                    mint_hi[t] = vmlal_high_s16(mint_hi[t], m_s16, b_full);
                } else {
                    let m_lo = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(m_u16)));
                    let m_hi = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(m_u16)));
                    let b_vec = vdupq_n_s32(b_pair_i32);
                    mint_lo[t] = vmlaq_s32(mint_lo[t], m_lo, b_vec);
                    mint_hi[t] = vmlaq_s32(mint_hi[t], m_hi, b_vec);
                }
            }
        }
        let mut mint = [[0_i32; TOKENS]; 8];
        for t in 0..TOKENS {
            let mut m_lo = [0_i32; 4];
            let mut m_hi = [0_i32; 4];
            store_s32x4(mint_lo[t], &mut m_lo);
            store_s32x4(mint_hi[t], &mut m_hi);
            for r in 0..4 {
                mint[r][t] = m_lo[r];
                mint[r + 4][t] = m_hi[r];
            }
        }

        let a_groups: [&[[i8; 16]]; TOKENS] =
            std::array::from_fn(|t| acts[t][blk_idx].qs.as_chunks::<16>().0);
        let mut packed_a = [[[vdupq_n_s8(0); 2]; COL_PAIRS]; 16];
        for (a_sub, sub_slots) in packed_a.iter_mut().enumerate() {
            for (cp, slot) in sub_slots.iter_mut().enumerate() {
                let a_even = load_s8x16(&a_groups[cp * 2][a_sub]);
                let a_odd = load_s8x16(&a_groups[cp * 2 + 1][a_sub]);
                *slot = [
                    vcombine_s8(vget_low_s8(a_even), vget_low_s8(a_odd)),
                    vcombine_s8(vget_high_s8(a_even), vget_high_s8(a_odd)),
                ];
            }
        }

        let d_w = bonsai_neon_dot::widen_f16_eight(&d_f16);
        let dmin_w = bonsai_neon_dot::widen_f16_eight(&dmin_f16);
        let d_act: [f64; TOKENS] = std::array::from_fn(|t| f64::from(acts[t][blk_idx].d));

        for pair in 0..4 {
            let mut sum_pair = [vdupq_n_s32(0); COL_PAIRS];
            for chunk in 0..4 {
                let qs_base = 160 + chunk * 256;
                let mut lo_acc = [vdupq_n_s32(0); COL_PAIRS];
                let mut hi_acc = [vdupq_n_s32(0); COL_PAIRS];
                for sub in 0..2 {
                    let o0 = qs_base + (pair * 2) * 32 + sub * 16;
                    let o1 = qs_base + (pair * 2 + 1) * 32 + sub * 16;
                    let raw0 = load_u8x16(&block[o0..o0 + 16].as_chunks::<16>().0[0]);
                    let raw1 = load_u8x16(&block[o1..o1 + 16].as_chunks::<16>().0[0]);

                    let l0 = vreinterpretq_s8_u8(vandq_u8(raw0, mask_0f));
                    let l1 = vreinterpretq_s8_u8(vandq_u8(raw1, mask_0f));
                    let w_lo0 = vcombine_s8(vget_low_s8(l0), vget_low_s8(l1));
                    let w_lo1 = vcombine_s8(vget_high_s8(l0), vget_high_s8(l1));
                    let a_lo_sub = chunk * 4 + sub;
                    for (cp, &[a0, a1]) in packed_a[a_lo_sub].iter().enumerate() {
                        lo_acc[cp] = smmla_s32(smmla_s32(lo_acc[cp], w_lo0, a0), w_lo1, a1);
                    }

                    let h0 = vreinterpretq_s8_u8(vshrq_n_u8(raw0, 4));
                    let h1 = vreinterpretq_s8_u8(vshrq_n_u8(raw1, 4));
                    let w_hi0 = vcombine_s8(vget_low_s8(h0), vget_low_s8(h1));
                    let w_hi1 = vcombine_s8(vget_high_s8(h0), vget_high_s8(h1));
                    let a_hi_sub = chunk * 4 + 2 + sub;
                    for (cp, &[a0, a1]) in packed_a[a_hi_sub].iter().enumerate() {
                        hi_acc[cp] = smmla_s32(smmla_s32(hi_acc[cp], w_hi0, a0), w_hi1, a1);
                    }
                }
                let sc_lo = &sc_bytes[(chunk * 2) * 8..(chunk * 2 + 1) * 8];
                let sc_hi = &sc_bytes[(chunk * 2 + 1) * 8..(chunk * 2 + 2) * 8];
                let s_lo0 = i32::from(sc_lo[pair * 2]);
                let s_lo1 = i32::from(sc_lo[pair * 2 + 1]);
                let s_hi0 = i32::from(sc_hi[pair * 2]);
                let s_hi1 = i32::from(sc_hi[pair * 2 + 1]);
                let scale_lo = load_s32x4(&[s_lo0, s_lo0, s_lo1, s_lo1]);
                let scale_hi = load_s32x4(&[s_hi0, s_hi0, s_hi1, s_hi1]);
                for cp in 0..COL_PAIRS {
                    sum_pair[cp] = vmlaq_s32(sum_pair[cp], lo_acc[cp], scale_lo);
                    sum_pair[cp] = vmlaq_s32(sum_pair[cp], hi_acc[cp], scale_hi);
                }
            }

            let r0 = pair * 2;
            let r1 = pair * 2 + 1;
            let dw0 = f64::from(d_w[r0 / 4][r0 % 4]);
            let dm0 = f64::from(dmin_w[r0 / 4][r0 % 4]);
            let dw1 = f64::from(d_w[r1 / 4][r1 % 4]);
            let dm1 = f64::from(dmin_w[r1 / 4][r1 % 4]);
            for (cp, &vec) in sum_pair.iter().enumerate() {
                let mut vals = [0_i32; 4];
                store_s32x4(vec, &mut vals);
                let c0 = cp * 2;
                let c1 = cp * 2 + 1;
                let da0 = d_act[c0];
                let da1 = d_act[c1];
                #[expect(
                    clippy::suboptimal_flops,
                    reason = "Matches non-FMA repacked Q4_K block accumulation order."
                )]
                {
                    acc[r0][c0] +=
                        (dw0 * da0) * f64::from(vals[0]) - (dm0 * da0) * f64::from(mint[r0][c0]);
                    acc[r0][c1] +=
                        (dw0 * da1) * f64::from(vals[1]) - (dm0 * da1) * f64::from(mint[r0][c1]);
                    acc[r1][c0] +=
                        (dw1 * da0) * f64::from(vals[2]) - (dm1 * da0) * f64::from(mint[r1][c0]);
                    acc[r1][c1] +=
                        (dw1 * da1) * f64::from(vals[3]) - (dm1 * da1) * f64::from(mint[r1][c1]);
                }
            }
        }
    }

    acc.map(|row| row.map(f64_to_f32))
}

fn valid_repacked_q4_k_strip<const TOKENS: usize>(
    strip: &[u8],
    acts: &[&[Q8_KActBlock]; TOKENS],
) -> bool {
    let blocks = acts[0].len();
    let Some(expected) = blocks.checked_mul(Q4_K_REPACK8_BLOCK_BYTES) else {
        return false;
    };
    acts.iter().all(|act| act.len() == blocks) && strip.len() == expected
}

fn gemm_q4_k_repacked8_8x_dispatch<const TOKENS: usize, const COL_PAIRS: usize>(
    caps: Arm64QuantCaps,
    strip: &[u8],
    acts: [&[Q8_KActBlock]; TOKENS],
) -> [[f32; TOKENS]; 8] {
    if !valid_repacked_q4_k_strip(strip, &acts) {
        return [[0.0; TOKENS]; 8];
    }
    #[cfg(target_arch = "aarch64")]
    if caps.i8mm && bonsai_neon_dot::has_i8mm() {
        // SAFETY: `caps.i8mm` and `bonsai_neon_dot::has_i8mm()` verified CPU support for `neon,i8mm`.
        return unsafe { gemm_q4_k_repacked8_8x_i8mm::<TOKENS, COL_PAIRS>(strip, acts) };
    }
    let mut out = [[0.0_f32; TOKENS]; 8];
    for cp in 0..COL_PAIRS {
        let tile = gemm_q4_k_repacked8_8x2(caps, strip, acts[cp * 2], acts[cp * 2 + 1]);
        for (dst_row, src_pair) in out.iter_mut().zip(tile) {
            dst_row[cp * 2] = src_pair[0];
            dst_row[cp * 2 + 1] = src_pair[1];
        }
    }
    out
}

/// Computes an 8-row by 8-token repacked `Q4_K` strip GEMM tile using `I8MM` (`SMMLA`) or exact fallback.
pub(crate) fn gemm_q4_k_repacked8_8x8(
    caps: Arm64QuantCaps,
    strip: &[u8],
    acts: [&[Q8_KActBlock]; 8],
) -> [[f32; 8]; 8] {
    gemm_q4_k_repacked8_8x_dispatch::<8, 4>(caps, strip, acts)
}

/// Computes an 8-row by 4-token repacked `Q4_K` strip GEMM tile using `I8MM` (`SMMLA`) or exact fallback.
pub(crate) fn gemm_q4_k_repacked8_8x4(
    caps: Arm64QuantCaps,
    strip: &[u8],
    acts: [&[Q8_KActBlock]; 4],
) -> [[f32; 4]; 8] {
    gemm_q4_k_repacked8_8x_dispatch::<4, 2>(caps, strip, acts)
}

/// Pre-interleaved 8-token `Q8_K` activation panel for `I8MM` (`SMMLA`) `Q4_K` / `Q6_K` GEMM.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug)]
pub struct Q8_KActPanel8 {
    /// Per-token superblock scales `f64::from(d)` for tokens `0..8`.
    pub d: [f64; 8],
    /// Pre-summed `bsums[2 * g] + bsums[2 * g + 1]` indexed as `[group][token]`.
    pub bsum_pairs: [[i32; 8]; 8],
    /// Pre-interleaved signed activation bytes `[sub16][col_pair][lo_hi][16]` for `SMMLA`.
    pub qs: [[[[i8; 16]; 2]; 4]; 16],
}

impl Default for Q8_KActPanel8 {
    fn default() -> Self {
        Self {
            d: [0.0; 8],
            bsum_pairs: [[0; 8]; 8],
            qs: [[[[0; 16]; 2]; 4]; 16],
        }
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[target_feature(enable = "neon")]
fn pack_q8_k_act_panels8_neon(
    acts: &[Q8_KActBlock],
    blocks: usize,
    panel_count: usize,
    panels: &mut [Q8_KActPanel8],
) {
    for p in 0..panel_count {
        let panel_dst = &mut panels[p * blocks..(p + 1) * blocks];
        let tok_base = p * 8;
        for (k, dst) in panel_dst.iter_mut().enumerate() {
            for t in 0..8 {
                let ab = &acts[(tok_base + t) * blocks + k];
                dst.d[t] = f64::from(ab.d);
                for g in 0..8 {
                    dst.bsum_pairs[g][t] =
                        i32::from(ab.bsums[2 * g]) + i32::from(ab.bsums[2 * g + 1]);
                }
            }
            for cp in 0..4 {
                let a_even = &acts[(tok_base + cp * 2) * blocks + k];
                let a_odd = &acts[(tok_base + cp * 2 + 1) * blocks + k];
                let e_groups = a_even.qs.as_chunks::<16>().0;
                let o_groups = a_odd.qs.as_chunks::<16>().0;
                for a_sub in 0..16 {
                    let e = load_s8x16(&e_groups[a_sub]);
                    let o = load_s8x16(&o_groups[a_sub]);
                    store_s8x16(
                        vcombine_s8(vget_low_s8(e), vget_low_s8(o)),
                        &mut dst.qs[a_sub][cp][0],
                    );
                    store_s8x16(
                        vcombine_s8(vget_high_s8(e), vget_high_s8(o)),
                        &mut dst.qs[a_sub][cp][1],
                    );
                }
            }
        }
    }
}

#[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
fn pack_q8_k_act_panels8_portable(
    acts: &[Q8_KActBlock],
    blocks: usize,
    panel_count: usize,
    panels: &mut [Q8_KActPanel8],
) {
    for p in 0..panel_count {
        let panel_dst = &mut panels[p * blocks..(p + 1) * blocks];
        let tok_base = p * 8;
        for (k, dst) in panel_dst.iter_mut().enumerate() {
            for t in 0..8 {
                let ab = &acts[(tok_base + t) * blocks + k];
                dst.d[t] = f64::from(ab.d);
                for g in 0..8 {
                    dst.bsum_pairs[g][t] =
                        i32::from(ab.bsums[2 * g]) + i32::from(ab.bsums[2 * g + 1]);
                }
            }
            for cp in 0..4 {
                let a_even = &acts[(tok_base + cp * 2) * blocks + k];
                let a_odd = &acts[(tok_base + cp * 2 + 1) * blocks + k];
                for a_sub in 0..16 {
                    let off = a_sub * 16;
                    dst.qs[a_sub][cp][0][0..8].copy_from_slice(&a_even.qs[off..off + 8]);
                    dst.qs[a_sub][cp][0][8..16].copy_from_slice(&a_odd.qs[off..off + 8]);
                    dst.qs[a_sub][cp][1][0..8].copy_from_slice(&a_even.qs[off + 8..off + 16]);
                    dst.qs[a_sub][cp][1][8..16].copy_from_slice(&a_odd.qs[off + 8..off + 16]);
                }
            }
        }
    }
}

/// Pre-packs `panel_count * 8` tokens of `Q8_KActBlock`s (`acts`) into `Q8_KActPanel8`s
/// laid out as `[panel_idx * blocks + k]`.
pub(crate) fn pack_q8_k_act_panels8(
    acts: &[Q8_KActBlock],
    blocks: usize,
    panel_count: usize,
    panels: &mut [Q8_KActPanel8],
) {
    if blocks == 0 || panel_count == 0 {
        return;
    }
    let Some(min_acts) = panel_count
        .checked_mul(8)
        .and_then(|t| t.checked_mul(blocks))
    else {
        return;
    };
    let Some(needed_panels) = panel_count.checked_mul(blocks) else {
        return;
    };
    if acts.len() < min_acts || panels.len() < needed_panels {
        return;
    }
    let panels = &mut panels[..needed_panels];
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        // SAFETY: Baseline NEON is enabled by target configuration and slice bounds were checked above.
        unsafe { pack_q8_k_act_panels8_neon(acts, blocks, panel_count, panels) };
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        pack_q8_k_act_panels8_portable(acts, blocks, panel_count, panels);
    }
}

fn unpack_q8_k_panel8(panel: &[Q8_KActPanel8], blocks: usize, unpacked: &mut [Q8_KActBlock]) {
    for (k, ap) in panel.iter().enumerate() {
        for cp in 0..4 {
            let t_even = cp * 2;
            let t_odd = cp * 2 + 1;
            {
                let dst_even = &mut unpacked[t_even * blocks + k];
                dst_even.d = f64_to_f32(ap.d[t_even]);
                for a_sub in 0..16 {
                    let off = a_sub * 16;
                    dst_even.qs[off..off + 8].copy_from_slice(&ap.qs[a_sub][cp][0][0..8]);
                    dst_even.qs[off + 8..off + 16].copy_from_slice(&ap.qs[a_sub][cp][1][0..8]);
                }
                for g in 0..8 {
                    let pair = ap.bsum_pairs[g][t_even];
                    let half = pair / 2;
                    dst_even.bsums[2 * g] = i16::try_from(half).unwrap_or(0);
                    dst_even.bsums[2 * g + 1] = i16::try_from(pair - half).unwrap_or(0);
                }
            }
            let dst_odd = &mut unpacked[t_odd * blocks + k];
            dst_odd.d = f64_to_f32(ap.d[t_odd]);
            for a_sub in 0..16 {
                let off = a_sub * 16;
                dst_odd.qs[off..off + 8].copy_from_slice(&ap.qs[a_sub][cp][0][8..16]);
                dst_odd.qs[off + 8..off + 16].copy_from_slice(&ap.qs[a_sub][cp][1][8..16]);
            }
            for g in 0..8 {
                let pair = ap.bsum_pairs[g][t_odd];
                let half = pair / 2;
                dst_odd.bsums[2 * g] = i16::try_from(half).unwrap_or(0);
                dst_odd.bsums[2 * g + 1] = i16::try_from(pair - half).unwrap_or(0);
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct UnpackedQ4KWeightBlock {
    d_w: [f64; 8],
    dmin_w: [f64; 8],
    m_bytes: [[u8; 8]; 8],
    sc_pairs: [[[[i32; 4]; 2]; 4]; 4],
    qs: [[[[[[i8; 16]; 2]; 2]; 2]; 4]; 4],
}

#[cfg(target_arch = "aarch64")]
impl UnpackedQ4KWeightBlock {
    const ZERO: Self = Self {
        d_w: [0.0; 8],
        dmin_w: [0.0; 8],
        m_bytes: [[0; 8]; 8],
        sc_pairs: [[[[0; 4]; 2]; 4]; 4],
        qs: [[[[[[0; 16]; 2]; 2]; 2]; 4]; 4],
    };
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,i8mm")]
fn gemm_q4_k_repacked8_8x_panels_i8mm(
    strip: &[u8],
    panels: &[Q8_KActPanel8],
    panel_count: usize,
    blocks: usize,
    out_rows: &mut [f32],
    row_stride: usize,
) {
    const K_TILE: usize = 4;
    const T_PANEL: usize = 8;

    let strip_blocks = strip.as_chunks::<1184>().0;
    let mask_0f = vdupq_n_u8(0x0f);
    let mut w_tile = [UnpackedQ4KWeightBlock::ZERO; K_TILE];
    let mut acc_buf = [[[[0.0_f64; 2]; 4]; 8]; T_PANEL];

    for p_base in (0..panel_count).step_by(T_PANEL) {
        let p_len = (panel_count - p_base).min(T_PANEL);
        for slot in &mut acc_buf[..p_len] {
            *slot = [[[0.0_f64; 2]; 4]; 8];
        }
        for k_base in (0..blocks).step_by(K_TILE) {
            let k_len = (blocks - k_base).min(K_TILE);
            for (k_local, wb) in w_tile[..k_len].iter_mut().enumerate() {
                let idx = k_base + k_local;
                if idx + 1 < blocks {
                    prefetch_q4_k_repacked8_block(&strip_blocks[idx + 1]);
                }
                let block = &strip_blocks[idx];
                let d_f16: [u16; 8] =
                    std::array::from_fn(|r| u16::from_le_bytes([block[r * 2], block[r * 2 + 1]]));
                let dmin_f16: [u16; 8] = std::array::from_fn(|r| {
                    u16::from_le_bytes([block[16 + r * 2], block[16 + r * 2 + 1]])
                });
                let d_w = bonsai_neon_dot::widen_f16_eight(&d_f16);
                let dmin_w = bonsai_neon_dot::widen_f16_eight(&dmin_f16);
                for r in 0..8 {
                    wb.d_w[r] = f64::from(d_w[r / 4][r % 4]);
                    wb.dmin_w[r] = f64::from(dmin_w[r / 4][r % 4]);
                }
                let sc_bytes = &block[32..96];
                let m_bytes = &block[96..160];
                for g in 0..8 {
                    wb.m_bytes[g].copy_from_slice(&m_bytes[g * 8..(g + 1) * 8]);
                }
                for pair in 0..4 {
                    for chunk in 0..4 {
                        let qs_base = 160 + chunk * 256;
                        for sub in 0..2 {
                            let o0 = qs_base + (pair * 2) * 32 + sub * 16;
                            let o1 = qs_base + (pair * 2 + 1) * 32 + sub * 16;
                            let raw0 = load_u8x16(&block[o0..o0 + 16].as_chunks::<16>().0[0]);
                            let raw1 = load_u8x16(&block[o1..o1 + 16].as_chunks::<16>().0[0]);

                            let l0 = vreinterpretq_s8_u8(vandq_u8(raw0, mask_0f));
                            let l1 = vreinterpretq_s8_u8(vandq_u8(raw1, mask_0f));
                            store_s8x16(
                                vcombine_s8(vget_low_s8(l0), vget_low_s8(l1)),
                                &mut wb.qs[pair][chunk][sub][0][0],
                            );
                            store_s8x16(
                                vcombine_s8(vget_high_s8(l0), vget_high_s8(l1)),
                                &mut wb.qs[pair][chunk][sub][0][1],
                            );

                            let h0 = vreinterpretq_s8_u8(vshrq_n_u8(raw0, 4));
                            let h1 = vreinterpretq_s8_u8(vshrq_n_u8(raw1, 4));
                            store_s8x16(
                                vcombine_s8(vget_low_s8(h0), vget_low_s8(h1)),
                                &mut wb.qs[pair][chunk][sub][1][0],
                            );
                            store_s8x16(
                                vcombine_s8(vget_high_s8(h0), vget_high_s8(h1)),
                                &mut wb.qs[pair][chunk][sub][1][1],
                            );
                        }
                        let sc_lo = &sc_bytes[(chunk * 2) * 8..(chunk * 2 + 1) * 8];
                        let sc_hi = &sc_bytes[(chunk * 2 + 1) * 8..(chunk * 2 + 2) * 8];
                        let s_lo0 = i32::from(sc_lo[pair * 2]);
                        let s_lo1 = i32::from(sc_lo[pair * 2 + 1]);
                        let s_hi0 = i32::from(sc_hi[pair * 2]);
                        let s_hi1 = i32::from(sc_hi[pair * 2 + 1]);
                        wb.sc_pairs[pair][chunk][0] = [s_lo0, s_lo0, s_lo1, s_lo1];
                        wb.sc_pairs[pair][chunk][1] = [s_hi0, s_hi0, s_hi1, s_hi1];
                    }
                }
            }

            for (p_local, p_acc) in acc_buf[..p_len].iter_mut().enumerate() {
                let p_idx = p_base + p_local;
                let p_panels = &panels[p_idx * blocks + k_base..p_idx * blocks + k_base + k_len];
                let mut acc: [[float64x2_t; 4]; 8] = std::array::from_fn(|r| {
                    std::array::from_fn(|cp| load_f64x2(&p_acc[r][cp]))
                });

                for (wb, ap) in w_tile[..k_len].iter().zip(p_panels.iter()) {
                    let mut mint_lo = [vdupq_n_s32(0); 8];
                    let mut mint_hi = [vdupq_n_s32(0); 8];
                    for g in 0..8 {
                        let m_u8 = load_u8x8(&wb.m_bytes[g]);
                        let m_u16 = vmovl_u8(m_u8);
                        let m_s16 = vreinterpretq_s16_u16(m_u16);
                        for t in 0..8 {
                            let b_pair_i32 = ap.bsum_pairs[g][t];
                            if let Ok(b_pair) = i16::try_from(b_pair_i32) {
                                let b_half = vdup_n_s16(b_pair);
                                let b_full = vcombine_s16(b_half, b_half);
                                mint_lo[t] = vmlal_s16(mint_lo[t], vget_low_s16(m_s16), b_half);
                                mint_hi[t] = vmlal_high_s16(mint_hi[t], m_s16, b_full);
                            } else {
                                let m_lo = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(m_u16)));
                                let m_hi = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(m_u16)));
                                let b_vec = vdupq_n_s32(b_pair_i32);
                                mint_lo[t] = vmlaq_s32(mint_lo[t], m_lo, b_vec);
                                mint_hi[t] = vmlaq_s32(mint_hi[t], m_hi, b_vec);
                            }
                        }
                    }
                    let mint_pair: [[int32x4_t; 4]; 4] = [
                        std::array::from_fn(|cp| vzip1q_s32(mint_lo[cp * 2], mint_lo[cp * 2 + 1])),
                        std::array::from_fn(|cp| vzip2q_s32(mint_lo[cp * 2], mint_lo[cp * 2 + 1])),
                        std::array::from_fn(|cp| vzip1q_s32(mint_hi[cp * 2], mint_hi[cp * 2 + 1])),
                        std::array::from_fn(|cp| vzip2q_s32(mint_hi[cp * 2], mint_hi[cp * 2 + 1])),
                    ];
                    let d_act: [float64x2_t; 4] = [
                        load_f64x2(&[ap.d[0], ap.d[1]]),
                        load_f64x2(&[ap.d[2], ap.d[3]]),
                        load_f64x2(&[ap.d[4], ap.d[5]]),
                        load_f64x2(&[ap.d[6], ap.d[7]]),
                    ];

                    for pair in 0..4 {
                        let mut sum_pair = [vdupq_n_s32(0); 4];
                        for chunk in 0..4 {
                            let mut lo_acc = [vdupq_n_s32(0); 4];
                            let mut hi_acc = [vdupq_n_s32(0); 4];
                            for sub in 0..2 {
                                let w_lo0 = load_s8x16(&wb.qs[pair][chunk][sub][0][0]);
                                let w_lo1 = load_s8x16(&wb.qs[pair][chunk][sub][0][1]);
                                let a_lo_sub = chunk * 4 + sub;
                                for (cp, a_slot) in ap.qs[a_lo_sub].iter().enumerate() {
                                    let a0 = load_s8x16(&a_slot[0]);
                                    let a1 = load_s8x16(&a_slot[1]);
                                    lo_acc[cp] =
                                        smmla_s32(smmla_s32(lo_acc[cp], w_lo0, a0), w_lo1, a1);
                                }

                                let w_hi0 = load_s8x16(&wb.qs[pair][chunk][sub][1][0]);
                                let w_hi1 = load_s8x16(&wb.qs[pair][chunk][sub][1][1]);
                                let a_hi_sub = chunk * 4 + 2 + sub;
                                for (cp, a_slot) in ap.qs[a_hi_sub].iter().enumerate() {
                                    let a0 = load_s8x16(&a_slot[0]);
                                    let a1 = load_s8x16(&a_slot[1]);
                                    hi_acc[cp] =
                                        smmla_s32(smmla_s32(hi_acc[cp], w_hi0, a0), w_hi1, a1);
                                }
                            }
                            let scale_lo = load_s32x4(&wb.sc_pairs[pair][chunk][0]);
                            let scale_hi = load_s32x4(&wb.sc_pairs[pair][chunk][1]);
                            for cp in 0..4 {
                                sum_pair[cp] = vmlaq_s32(sum_pair[cp], lo_acc[cp], scale_lo);
                                sum_pair[cp] = vmlaq_s32(sum_pair[cp], hi_acc[cp], scale_hi);
                            }
                        }

                        let r0 = pair * 2;
                        let r1 = pair * 2 + 1;
                        let dw0_v = vdupq_n_f64(wb.d_w[r0]);
                        let dm0_v = vdupq_n_f64(wb.dmin_w[r0]);
                        let dw1_v = vdupq_n_f64(wb.d_w[r1]);
                        let dm1_v = vdupq_n_f64(wb.dmin_w[r1]);
                        for cp in 0..4 {
                            let (sum_r0, sum_r1) = s32x4_to_f64x2_pair(sum_pair[cp]);
                            let (mint_r0, mint_r1) = s32x4_to_f64x2_pair(mint_pair[pair][cp]);
                            let da = d_act[cp];
                            let t0 = vsubq_f64(
                                vmulq_f64(vmulq_f64(dw0_v, da), sum_r0),
                                vmulq_f64(vmulq_f64(dm0_v, da), mint_r0),
                            );
                            let t1 = vsubq_f64(
                                vmulq_f64(vmulq_f64(dw1_v, da), sum_r1),
                                vmulq_f64(vmulq_f64(dm1_v, da), mint_r1),
                            );
                            acc[r0][cp] = vaddq_f64(acc[r0][cp], t0);
                            acc[r1][cp] = vaddq_f64(acc[r1][cp], t1);
                        }
                    }
                }

                for (r, row_acc) in acc.iter().enumerate() {
                    for (cp, &vec) in row_acc.iter().enumerate() {
                        store_f64x2(vec, &mut p_acc[r][cp]);
                    }
                }
            }
        }

        for (p_local, p_acc) in acc_buf[..p_len].iter().enumerate() {
            let acc: [[float64x2_t; 4]; 8] = std::array::from_fn(|r| {
                std::array::from_fn(|cp| load_f64x2(&p_acc[r][cp]))
            });
            store_f64x2_pairs_as_f32x8(&acc, out_rows, row_stride, (p_base + p_local) * 8);
        }
    }
}

/// Computes an 8-row by `(panel_count * 8)`-token repacked `Q4_K` strip GEMM over pre-packed
/// `Q8_KActPanel8`s with K-blocked L1D weight tiles (`I8MM` `SMMLA`, or exact `SDOT`/portable fallback).
pub(crate) fn gemm_q4_k_repacked8_8x_panels(
    caps: Arm64QuantCaps,
    strip: &[u8],
    panels: &[Q8_KActPanel8],
    panel_count: usize,
    blocks: usize,
    out_rows: &mut [f32],
    row_stride: usize,
) {
    if panel_count == 0 {
        return;
    }
    let Some(tokens) = panel_count.checked_mul(8) else {
        return;
    };
    let Some(expected_panels) = panel_count.checked_mul(blocks) else {
        return;
    };
    let Some(expected_strip) = blocks.checked_mul(Q4_K_REPACK8_BLOCK_BYTES) else {
        return;
    };
    let Some(min_out) = row_stride
        .checked_mul(7)
        .and_then(|v| v.checked_add(tokens))
    else {
        return;
    };
    if row_stride < tokens
        || out_rows.len() < min_out
        || panels.len() != expected_panels
        || strip.len() != expected_strip
    {
        return;
    }
    if blocks == 0 {
        for r in 0..8 {
            out_rows[r * row_stride..r * row_stride + tokens].fill(0.0);
        }
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if caps.i8mm && bonsai_neon_dot::has_i8mm() {
        // SAFETY: `caps.i8mm` and `bonsai_neon_dot::has_i8mm()` verified CPU support for `neon,i8mm`,
        // and all slice lengths and row strides were validated above.
        unsafe {
            gemm_q4_k_repacked8_8x_panels_i8mm(
                strip,
                panels,
                panel_count,
                blocks,
                out_rows,
                row_stride,
            );
        }
        return;
    }

    let mut unpacked = vec![Q8_KActBlock::default(); 8 * blocks];
    for p in 0..panel_count {
        unpack_q8_k_panel8(&panels[p * blocks..(p + 1) * blocks], blocks, &mut unpacked);
        let t_off = p * 8;
        for t in 0..8 {
            let col = dot_q4_k_repacked8(caps, strip, &unpacked[t * blocks..(t + 1) * blocks]);
            for (r, &val) in col.iter().enumerate() {
                out_rows[r * row_stride + t_off + t] = val;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
fn unpack_q6_k_4quarters_16(
    block: &[u8; 210],
    half_idx: usize,
    sub: usize,
) -> ([int8x16_t; 4], [i32; 4]) {
    let low_off0 = half_idx * 64 + sub * 16;
    let low_off1 = low_off0 + 32;
    let high_off = 128 + half_idx * 32 + sub * 16;
    let ql0 = load_u8x16(&block[low_off0..low_off0 + 16].as_chunks::<16>().0[0]);
    let ql1 = load_u8x16(&block[low_off1..low_off1 + 16].as_chunks::<16>().0[0]);
    let qh = load_u8x16(&block[high_off..high_off + 16].as_chunks::<16>().0[0]);

    let mask_0f = vdupq_n_u8(0x0f);
    let mask_30 = vdupq_n_u8(0x30);
    let bias_32 = vdupq_n_s8(32);

    let lo0 = vandq_u8(ql0, mask_0f);
    let lo1 = vandq_u8(ql1, mask_0f);
    let lo2 = vshrq_n_u8(ql0, 4);
    let lo3 = vshrq_n_u8(ql1, 4);

    let hi0 = vandq_u8(vshlq_n_u8(qh, 4), mask_30);
    let hi1 = vandq_u8(vshlq_n_u8(qh, 2), mask_30);
    let hi2 = vandq_u8(qh, mask_30);
    let hi3 = vandq_u8(vshrq_n_u8(qh, 2), mask_30);

    let w = [
        vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(lo0, hi0)), bias_32),
        vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(lo1, hi1)), bias_32),
        vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(lo2, hi2)), bias_32),
        vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(lo3, hi3)), bias_32),
    ];
    let sc_base = 192 + half_idx * 8 + sub;
    let s = [
        i32::from(block[sc_base].cast_signed()),
        i32::from(block[sc_base + 2].cast_signed()),
        i32::from(block[sc_base + 4].cast_signed()),
        i32::from(block[sc_base + 6].cast_signed()),
    ];
    (w, s)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
fn dot_q6_k_8rows_dotprod(rows: [&[u8]; 8], act: &[Q8_KActBlock]) -> [f32; 8] {
    let r_chunks: [&[[u8; 210]]; 8] = rows.map(|r| r.as_chunks::<210>().0);
    let total_blocks = act.len();
    let mut acc = [0.0_f64; 8];
    let zero = vdupq_n_s32(0);

    for (blk_idx, ab) in act.iter().enumerate() {
        if blk_idx + 2 < total_blocks {
            for r in &r_chunks {
                let ptr = r[blk_idx + 2].as_ptr();
                prefetch_l1(ptr);
                prefetch_l1(ptr.wrapping_add(64));
                prefetch_l1(ptr.wrapping_add(128));
            }
        }
        let blocks: [&[u8; 210]; 8] = std::array::from_fn(|r| &r_chunks[r][blk_idx]);
        let a_groups = ab.qs.as_chunks::<16>().0;
        let d_act = f64::from(ab.d);
        let d_bits: [u16; 8] = blocks.map(|b| u16::from_le_bytes([b[208], b[209]]));
        let d_weights = bonsai_neon_dot::widen_f16_eight(&d_bits);

        let mut row_acc = [zero; 8];
        for half_idx in 0..2 {
            let h_base = half_idx * 8;
            for sub in 0..2 {
                let a0 = load_s8x16(&a_groups[h_base + sub]);
                let a1 = load_s8x16(&a_groups[h_base + 2 + sub]);
                let a2 = load_s8x16(&a_groups[h_base + 4 + sub]);
                let a3 = load_s8x16(&a_groups[h_base + 6 + sub]);
                for (r, block) in blocks.iter().enumerate() {
                    let (w_q, s_q) = unpack_q6_k_4quarters_16(block, half_idx, sub);
                    let d0 = vdotq_s32(zero, w_q[0], a0);
                    let d1 = vdotq_s32(zero, w_q[1], a1);
                    let d2 = vdotq_s32(zero, w_q[2], a2);
                    let d3 = vdotq_s32(zero, w_q[3], a3);
                    let q_dots = vpaddq_s32(vpaddq_s32(d0, d1), vpaddq_s32(d2, d3));
                    row_acc[r] = vmlaq_s32(row_acc[r], q_dots, load_s32x4(&s_q));
                }
            }
        }
        let sumi: [i32; 8] = std::array::from_fn(|r| vaddvq_s32(row_acc[r]));
        for r in 0..8 {
            let dw = f64::from(d_weights[r / 4][r % 4]);
            #[expect(
                clippy::suboptimal_flops,
                reason = "Matches non-FMA Q6_K block accumulation order."
            )]
            {
                acc[r] += (dw * d_act) * f64::from(sumi[r]);
            }
        }
    }
    acc.map(f64_to_f32)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,i8mm")]
fn gemm_q6_k_8x_i8mm<const TOKENS: usize, const COL_PAIRS: usize>(
    rows: [&[u8]; 8],
    acts: [&[Q8_KActBlock]; TOKENS],
) -> [[f32; TOKENS]; 8] {
    let r_chunks: [&[[u8; 210]]; 8] = rows.map(|r| r.as_chunks::<210>().0);
    let total_blocks = acts[0].len();
    let mut acc = [[0.0_f64; TOKENS]; 8];

    for blk_idx in 0..total_blocks {
        if blk_idx + 2 < total_blocks {
            for r in &r_chunks {
                let ptr = r[blk_idx + 2].as_ptr();
                prefetch_l1(ptr);
                prefetch_l1(ptr.wrapping_add(64));
                prefetch_l1(ptr.wrapping_add(128));
            }
        }
        let blocks: [&[u8; 210]; 8] = std::array::from_fn(|r| &r_chunks[r][blk_idx]);
        let d_bits: [u16; 8] = blocks.map(|b| u16::from_le_bytes([b[208], b[209]]));
        let d_weights = bonsai_neon_dot::widen_f16_eight(&d_bits);
        let a_groups: [&[[i8; 16]]; TOKENS] =
            std::array::from_fn(|t| acts[t][blk_idx].qs.as_chunks::<16>().0);

        let mut packed_a = [[[[vdupq_n_s8(0); 2]; COL_PAIRS]; 4]; 4];
        for half_idx in 0..2 {
            for sub in 0..2 {
                let hs = half_idx * 2 + sub;
                for quarter in 0..4 {
                    let a_sub = half_idx * 8 + quarter * 2 + sub;
                    for cp in 0..COL_PAIRS {
                        let a_even = load_s8x16(&a_groups[cp * 2][a_sub]);
                        let a_odd = load_s8x16(&a_groups[cp * 2 + 1][a_sub]);
                        packed_a[hs][quarter][cp] = [
                            vcombine_s8(vget_low_s8(a_even), vget_low_s8(a_odd)),
                            vcombine_s8(vget_high_s8(a_even), vget_high_s8(a_odd)),
                        ];
                    }
                }
            }
        }

        let d_act: [f64; TOKENS] = std::array::from_fn(|t| f64::from(acts[t][blk_idx].d));
        for pair in 0..4 {
            let mut pair_acc = [vdupq_n_s32(0); COL_PAIRS];
            for half_idx in 0..2 {
                for sub in 0..2 {
                    let hs = half_idx * 2 + sub;
                    let (w_even, s_even) =
                        unpack_q6_k_4quarters_16(blocks[pair * 2], half_idx, sub);
                    let (w_odd, s_odd) =
                        unpack_q6_k_4quarters_16(blocks[pair * 2 + 1], half_idx, sub);
                    for q in 0..4 {
                        let w_0 = vcombine_s8(vget_low_s8(w_even[q]), vget_low_s8(w_odd[q]));
                        let w_1 = vcombine_s8(vget_high_s8(w_even[q]), vget_high_s8(w_odd[q]));
                        let s_vec = load_s32x4(&[s_even[q], s_even[q], s_odd[q], s_odd[q]]);
                        for (cp, &[a_0, a_1]) in packed_a[hs][q].iter().enumerate() {
                            let g = smmla_s32(smmla_s32(vdupq_n_s32(0), w_0, a_0), w_1, a_1);
                            pair_acc[cp] = vmlaq_s32(pair_acc[cp], g, s_vec);
                        }
                    }
                }
            }

            let r0 = pair * 2;
            let r1 = pair * 2 + 1;
            let dw0 = f64::from(d_weights[r0 / 4][r0 % 4]);
            let dw1 = f64::from(d_weights[r1 / 4][r1 % 4]);
            for (cp, &vec) in pair_acc.iter().enumerate() {
                let mut vals = [0_i32; 4];
                store_s32x4(vec, &mut vals);
                let c0 = cp * 2;
                let c1 = cp * 2 + 1;
                let da0 = d_act[c0];
                let da1 = d_act[c1];
                #[expect(
                    clippy::suboptimal_flops,
                    reason = "Matches non-FMA Q6_K block accumulation order."
                )]
                {
                    acc[r0][c0] += (dw0 * da0) * f64::from(vals[0]);
                    acc[r0][c1] += (dw0 * da1) * f64::from(vals[1]);
                    acc[r1][c0] += (dw1 * da0) * f64::from(vals[2]);
                    acc[r1][c1] += (dw1 * da1) * f64::from(vals[3]);
                }
            }
        }
    }
    acc.map(|row| row.map(f64_to_f32))
}

fn valid_q6_k_rows(rows: &[&[u8]; 8], act_blocks: usize) -> bool {
    let Some(expected) = act_blocks.checked_mul(Q6_K_BLOCK_BYTES) else {
        return false;
    };
    rows.iter().all(|row| row.len() == expected)
}

/// Computes eight `Q6_K` row dot products while sharing activations per superblock (`0-ULP` vs `dot_q6_k_row`).
pub(crate) fn dot_q6_k_8rows(
    caps: Arm64QuantCaps,
    rows: [&[u8]; 8],
    act: &[Q8_KActBlock],
) -> [f32; 8] {
    if !valid_q6_k_rows(&rows, act.len()) {
        return [0.0; 8];
    }
    #[cfg(target_arch = "aarch64")]
    if caps.dotprod && bonsai_neon_dot::has_dotprod() {
        // SAFETY: `caps.dotprod` and `bonsai_neon_dot::has_dotprod()` verified CPU support for `neon,dotprod`.
        return unsafe { dot_q6_k_8rows_dotprod(rows, act) };
    }
    if caps.dotprod {
        let lo = dot_q6_k_4rows(caps, [rows[0], rows[1], rows[2], rows[3]], act);
        let hi = dot_q6_k_4rows(caps, [rows[4], rows[5], rows[6], rows[7]], act);
        return [lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]];
    }
    rows.map(|row| dot_q6_k_row(caps, row, act))
}

fn gemm_q6_k_8x_dispatch<const TOKENS: usize, const COL_PAIRS: usize>(
    caps: Arm64QuantCaps,
    rows: [&[u8]; 8],
    acts: [&[Q8_KActBlock]; TOKENS],
) -> [[f32; TOKENS]; 8] {
    let blocks = acts[0].len();
    if acts.iter().any(|act| act.len() != blocks) || !valid_q6_k_rows(&rows, blocks) {
        return [[0.0; TOKENS]; 8];
    }
    #[cfg(target_arch = "aarch64")]
    if caps.i8mm && bonsai_neon_dot::has_i8mm() {
        // SAFETY: `caps.i8mm` and `bonsai_neon_dot::has_i8mm()` verified CPU support for `neon,i8mm`.
        return unsafe { gemm_q6_k_8x_i8mm::<TOKENS, COL_PAIRS>(rows, acts) };
    }
    let cols: [[f32; 8]; TOKENS] = std::array::from_fn(|t| dot_q6_k_8rows(caps, rows, acts[t]));
    std::array::from_fn(|r| std::array::from_fn(|t| cols[t][r]))
}

/// Computes an 8-row by 8-token `Q6_K` tile using `I8MM` (`SMMLA`) or exact `SDOT` fallback.
pub(crate) fn gemm_q6_k_8x8(
    caps: Arm64QuantCaps,
    rows: [&[u8]; 8],
    acts: [&[Q8_KActBlock]; 8],
) -> [[f32; 8]; 8] {
    gemm_q6_k_8x_dispatch::<8, 4>(caps, rows, acts)
}

/// Computes an 8-row by 4-token `Q6_K` tile using `I8MM` (`SMMLA`) or exact `SDOT` fallback.
pub(crate) fn gemm_q6_k_8x4(
    caps: Arm64QuantCaps,
    rows: [&[u8]; 8],
    acts: [&[Q8_KActBlock]; 4],
) -> [[f32; 4]; 8] {
    gemm_q6_k_8x_dispatch::<4, 2>(caps, rows, acts)
}

/// Computes an 8-row by 2-token `Q6_K` tile using `I8MM` (`SMMLA`) or exact `SDOT`/portable fallback.
pub(crate) fn gemm_q6_k_8x2(
    caps: Arm64QuantCaps,
    rows: [&[u8]; 8],
    acts: [&[Q8_KActBlock]; 2],
) -> [[f32; 2]; 8] {
    gemm_q6_k_8x_dispatch::<2, 1>(caps, rows, acts)
}

#[cfg(target_arch = "aarch64")]
#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct UnpackedQ6KWeightBlock {
    d_w: [f64; 8],
    scales: [[[[i32; 4]; 4]; 4]; 4],
    qs: [[[[[i8; 16]; 2]; 4]; 4]; 4],
}

#[cfg(target_arch = "aarch64")]
impl UnpackedQ6KWeightBlock {
    const ZERO: Self = Self {
        d_w: [0.0; 8],
        scales: [[[[0; 4]; 4]; 4]; 4],
        qs: [[[[[0; 16]; 2]; 4]; 4]; 4],
    };
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,i8mm")]
fn gemm_q6_k_8x_panels_i8mm(
    rows: [&[u8]; 8],
    panels: &[Q8_KActPanel8],
    panel_count: usize,
    blocks: usize,
    out_rows: &mut [f32],
    row_stride: usize,
) {
    const K_TILE: usize = 4;
    const T_PANEL: usize = 8;

    let r_chunks: [&[[u8; 210]]; 8] = rows.map(|r| r.as_chunks::<210>().0);
    let mut w_tile = [UnpackedQ6KWeightBlock::ZERO; K_TILE];
    let mut acc_buf = [[[[0.0_f64; 2]; 4]; 8]; T_PANEL];

    for p_base in (0..panel_count).step_by(T_PANEL) {
        let p_len = (panel_count - p_base).min(T_PANEL);
        for slot in &mut acc_buf[..p_len] {
            *slot = [[[0.0_f64; 2]; 4]; 8];
        }
        for k_base in (0..blocks).step_by(K_TILE) {
            let k_len = (blocks - k_base).min(K_TILE);
            for (k_local, wb) in w_tile[..k_len].iter_mut().enumerate() {
                let idx = k_base + k_local;
                if idx + 2 < blocks {
                    for r in &r_chunks {
                        let ptr = r[idx + 2].as_ptr();
                        prefetch_l1(ptr);
                        prefetch_l1(ptr.wrapping_add(64));
                        prefetch_l1(ptr.wrapping_add(128));
                    }
                }
                let blks: [&[u8; 210]; 8] = std::array::from_fn(|r| &r_chunks[r][idx]);
                let d_bits: [u16; 8] = blks.map(|b| u16::from_le_bytes([b[208], b[209]]));
                let d_weights = bonsai_neon_dot::widen_f16_eight(&d_bits);
                for r in 0..8 {
                    wb.d_w[r] = f64::from(d_weights[r / 4][r % 4]);
                }
                for pair in 0..4 {
                    for half_idx in 0..2 {
                        for sub in 0..2 {
                            let hs = half_idx * 2 + sub;
                            let (w_even, s_even) =
                                unpack_q6_k_4quarters_16(blks[pair * 2], half_idx, sub);
                            let (w_odd, s_odd) =
                                unpack_q6_k_4quarters_16(blks[pair * 2 + 1], half_idx, sub);
                            for q in 0..4 {
                                store_s8x16(
                                    vcombine_s8(vget_low_s8(w_even[q]), vget_low_s8(w_odd[q])),
                                    &mut wb.qs[pair][hs][q][0],
                                );
                                store_s8x16(
                                    vcombine_s8(vget_high_s8(w_even[q]), vget_high_s8(w_odd[q])),
                                    &mut wb.qs[pair][hs][q][1],
                                );
                                wb.scales[pair][hs][q] =
                                    [s_even[q], s_even[q], s_odd[q], s_odd[q]];
                            }
                        }
                    }
                }
            }

            for (p_local, p_acc) in acc_buf[..p_len].iter_mut().enumerate() {
                let p_idx = p_base + p_local;
                let p_panels = &panels[p_idx * blocks + k_base..p_idx * blocks + k_base + k_len];
                let mut acc: [[float64x2_t; 4]; 8] = std::array::from_fn(|r| {
                    std::array::from_fn(|cp| load_f64x2(&p_acc[r][cp]))
                });

                for (wb, ap) in w_tile[..k_len].iter().zip(p_panels.iter()) {
                    let d_act: [float64x2_t; 4] = [
                        load_f64x2(&[ap.d[0], ap.d[1]]),
                        load_f64x2(&[ap.d[2], ap.d[3]]),
                        load_f64x2(&[ap.d[4], ap.d[5]]),
                        load_f64x2(&[ap.d[6], ap.d[7]]),
                    ];
                    for pair in 0..4 {
                        let mut pair_acc = [vdupq_n_s32(0); 4];
                        for half_idx in 0..2 {
                            for sub in 0..2 {
                                let hs = half_idx * 2 + sub;
                                for q in 0..4 {
                                    let a_sub = half_idx * 8 + q * 2 + sub;
                                    let w_0 = load_s8x16(&wb.qs[pair][hs][q][0]);
                                    let w_1 = load_s8x16(&wb.qs[pair][hs][q][1]);
                                    let s_vec = load_s32x4(&wb.scales[pair][hs][q]);
                                    for (cp, a_slot) in ap.qs[a_sub].iter().enumerate() {
                                        let a_0 = load_s8x16(&a_slot[0]);
                                        let a_1 = load_s8x16(&a_slot[1]);
                                        let g = smmla_s32(
                                            smmla_s32(vdupq_n_s32(0), w_0, a_0),
                                            w_1,
                                            a_1,
                                        );
                                        pair_acc[cp] = vmlaq_s32(pair_acc[cp], g, s_vec);
                                    }
                                }
                            }
                        }
                        let r0 = pair * 2;
                        let r1 = pair * 2 + 1;
                        let dw0_v = vdupq_n_f64(wb.d_w[r0]);
                        let dw1_v = vdupq_n_f64(wb.d_w[r1]);
                        for cp in 0..4 {
                            let (v_r0, v_r1) = s32x4_to_f64x2_pair(pair_acc[cp]);
                            let da = d_act[cp];
                            acc[r0][cp] = vaddq_f64(acc[r0][cp], vmulq_f64(vmulq_f64(dw0_v, da), v_r0));
                            acc[r1][cp] = vaddq_f64(acc[r1][cp], vmulq_f64(vmulq_f64(dw1_v, da), v_r1));
                        }
                    }
                }

                for (r, row_acc) in acc.iter().enumerate() {
                    for (cp, &vec) in row_acc.iter().enumerate() {
                        store_f64x2(vec, &mut p_acc[r][cp]);
                    }
                }
            }
        }

        for (p_local, p_acc) in acc_buf[..p_len].iter().enumerate() {
            let acc: [[float64x2_t; 4]; 8] = std::array::from_fn(|r| {
                std::array::from_fn(|cp| load_f64x2(&p_acc[r][cp]))
            });
            store_f64x2_pairs_as_f32x8(&acc, out_rows, row_stride, (p_base + p_local) * 8);
        }
    }
}

/// Computes an 8-row by `(panel_count * 8)`-token `Q6_K` GEMM over pre-packed `Q8_KActPanel8`s
/// with K-blocked L1D weight tiles (`I8MM` `SMMLA`, or exact `SDOT`/portable fallback).
pub(crate) fn gemm_q6_k_8x_panels(
    caps: Arm64QuantCaps,
    rows: [&[u8]; 8],
    panels: &[Q8_KActPanel8],
    panel_count: usize,
    blocks: usize,
    out_rows: &mut [f32],
    row_stride: usize,
) {
    if panel_count == 0 {
        return;
    }
    let Some(tokens) = panel_count.checked_mul(8) else {
        return;
    };
    let Some(expected_panels) = panel_count.checked_mul(blocks) else {
        return;
    };
    let Some(min_out) = row_stride
        .checked_mul(7)
        .and_then(|v| v.checked_add(tokens))
    else {
        return;
    };
    if row_stride < tokens
        || out_rows.len() < min_out
        || panels.len() != expected_panels
        || !valid_q6_k_rows(&rows, blocks)
    {
        return;
    }
    if blocks == 0 {
        for r in 0..8 {
            out_rows[r * row_stride..r * row_stride + tokens].fill(0.0);
        }
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if caps.i8mm && bonsai_neon_dot::has_i8mm() {
        // SAFETY: `caps.i8mm` and `bonsai_neon_dot::has_i8mm()` verified CPU support for `neon,i8mm`,
        // and all slice lengths and row strides were validated above.
        unsafe {
            gemm_q6_k_8x_panels_i8mm(rows, panels, panel_count, blocks, out_rows, row_stride);
        }
        return;
    }

    let mut unpacked = vec![Q8_KActBlock::default(); 8 * blocks];
    for p in 0..panel_count {
        unpack_q8_k_panel8(&panels[p * blocks..(p + 1) * blocks], blocks, &mut unpacked);
        let t_off = p * 8;
        for t in 0..8 {
            let col = dot_q6_k_8rows(caps, rows, &unpacked[t * blocks..(t + 1) * blocks]);
            for (r, &val) in col.iter().enumerate() {
                out_rows[r * row_stride + t_off + t] = val;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
fn widen_f16x8_neon(bits: &[u16; 8]) -> (float32x4_t, float32x4_t) {
    // SAFETY: `bits` is a valid 8-element `u16` array reference and `vld1q_u16` requires no alignment.
    let raw = unsafe { vld1q_u16(bits.as_ptr()) };
    if vmaxvq_u16(vandq_u16(raw, vdupq_n_u16(0x7fff))) >= 0x7c00 {
        let [lo, hi] = bonsai_neon_dot::widen_f16_eight(bits);
        return (load_f32x4(&lo), load_f32x4(&hi));
    }
    let half = vreinterpretq_f16_u16(raw);
    (vcvt_f32_f16(vget_low_f16(half)), vcvt_high_f32_f16(half))
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
fn widen_f16x4_neon(bits: &[u16; 4]) -> float32x4_t {
    // SAFETY: `bits` is a valid 4-element `u16` array reference and `vld1_u16` requires no alignment.
    let raw = unsafe { vld1_u16(bits.as_ptr()) };
    if vmaxv_u16(vand_u16(raw, vdup_n_u16(0x7fff))) >= 0x7c00 {
        return load_f32x4(&bonsai_neon_dot::widen_f16_four(bits));
    }
    vcvt_f32_f16(vreinterpret_f16_u16(raw))
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
fn dot_f16_pair_f32_neon(q0: &[f32], q1: &[f32], k_f16: &[u16]) -> (f32, f32) {
    let (k_chunks, k_tail) = k_f16.as_chunks::<4>();
    let (q0_chunks, q0_tail) = q0.as_chunks::<4>();
    let (q1_chunks, q1_tail) = q1.as_chunks::<4>();
    let (k_pairs, k_rem) = k_chunks.as_chunks::<2>();
    let (q0_pairs, q0_rem) = q0_chunks.as_chunks::<2>();
    let (q1_pairs, q1_rem) = q1_chunks.as_chunks::<2>();

    let mut acc0_0 = vdupq_n_f32(0.0_f32);
    let mut acc0_1 = vdupq_n_f32(0.0_f32);
    let mut acc1_0 = vdupq_n_f32(0.0_f32);
    let mut acc1_1 = vdupq_n_f32(0.0_f32);

    for ((kp, q0p), q1p) in k_pairs.iter().zip(q0_pairs.iter()).zip(q1_pairs.iter()) {
        let k_eight = [
            kp[0][0], kp[0][1], kp[0][2], kp[0][3], kp[1][0], kp[1][1], kp[1][2], kp[1][3],
        ];
        let (k0, k1) = widen_f16x8_neon(&k_eight);
        acc0_0 = vfmaq_f32(acc0_0, load_f32x4(&q0p[0]), k0);
        acc0_1 = vfmaq_f32(acc0_1, load_f32x4(&q0p[1]), k1);
        acc1_0 = vfmaq_f32(acc1_0, load_f32x4(&q1p[0]), k0);
        acc1_1 = vfmaq_f32(acc1_1, load_f32x4(&q1p[1]), k1);
    }
    for ((kc, q0c), q1c) in k_rem.iter().zip(q0_rem.iter()).zip(q1_rem.iter()) {
        let kv = widen_f16x4_neon(kc);
        acc0_0 = vfmaq_f32(acc0_0, load_f32x4(q0c), kv);
        acc1_0 = vfmaq_f32(acc1_0, load_f32x4(q1c), kv);
    }

    let sum0_v = vaddq_f32(acc0_0, acc0_1);
    let sum1_v = vaddq_f32(acc1_0, acc1_1);
    let pair = vpaddq_f32(sum0_v, sum1_v);
    let reduced = vpaddq_f32(pair, pair);
    let mut lanes = [0.0_f32; 4];
    store_f32x4(reduced, &mut lanes);
    let mut total0 = lanes[0];
    let mut total1 = lanes[1];
    for ((&k_bits, &q0_v), &q1_v) in k_tail.iter().zip(q0_tail.iter()).zip(q1_tail.iter()) {
        let kv = bonsai_ternary::half_to_float(k_bits);
        total0 = q0_v.mul_add(kv, total0);
        total1 = q1_v.mul_add(kv, total1);
    }
    (total0, total1)
}

#[cfg(any(test, not(all(target_arch = "aarch64", target_feature = "neon"))))]
pub(crate) fn dot_f16_pair_f32_portable(q0: &[f32], q1: &[f32], k_f16: &[u16]) -> (f32, f32) {
    if q0.len() != k_f16.len() || q1.len() != k_f16.len() {
        return (0.0, 0.0);
    }
    let (k_chunks, k_tail) = k_f16.as_chunks::<4>();
    let (q0_chunks, q0_tail) = q0.as_chunks::<4>();
    let (q1_chunks, q1_tail) = q1.as_chunks::<4>();
    let (k_pairs, k_rem) = k_chunks.as_chunks::<2>();
    let (q0_pairs, q0_rem) = q0_chunks.as_chunks::<2>();
    let (q1_pairs, q1_rem) = q1_chunks.as_chunks::<2>();

    let mut acc0_0 = [0.0_f32; 4];
    let mut acc0_1 = [0.0_f32; 4];
    let mut acc1_0 = [0.0_f32; 4];
    let mut acc1_1 = [0.0_f32; 4];

    for ((kp, q0p), q1p) in k_pairs.iter().zip(q0_pairs.iter()).zip(q1_pairs.iter()) {
        let k0 = kp[0].map(bonsai_ternary::half_to_float);
        let k1 = kp[1].map(bonsai_ternary::half_to_float);
        for lane in 0..4 {
            acc0_0[lane] = q0p[0][lane].mul_add(k0[lane], acc0_0[lane]);
            acc0_1[lane] = q0p[1][lane].mul_add(k1[lane], acc0_1[lane]);
            acc1_0[lane] = q1p[0][lane].mul_add(k0[lane], acc1_0[lane]);
            acc1_1[lane] = q1p[1][lane].mul_add(k1[lane], acc1_1[lane]);
        }
    }
    for ((kc, q0c), q1c) in k_rem.iter().zip(q0_rem.iter()).zip(q1_rem.iter()) {
        let kv = kc.map(bonsai_ternary::half_to_float);
        for lane in 0..4 {
            acc0_0[lane] = q0c[lane].mul_add(kv[lane], acc0_0[lane]);
            acc1_0[lane] = q1c[lane].mul_add(kv[lane], acc1_0[lane]);
        }
    }
    let sum0 = [
        acc0_0[0] + acc0_1[0],
        acc0_0[1] + acc0_1[1],
        acc0_0[2] + acc0_1[2],
        acc0_0[3] + acc0_1[3],
    ];
    let sum1 = [
        acc1_0[0] + acc1_1[0],
        acc1_0[1] + acc1_1[1],
        acc1_0[2] + acc1_1[2],
        acc1_0[3] + acc1_1[3],
    ];
    let mut total0 = (sum0[0] + sum0[1]) + (sum0[2] + sum0[3]);
    let mut total1 = (sum1[0] + sum1[1]) + (sum1[2] + sum1[3]);
    for ((&k_bits, &q0_v), &q1_v) in k_tail.iter().zip(q0_tail.iter()).zip(q1_tail.iter()) {
        let kv = bonsai_ternary::half_to_float(k_bits);
        total0 = q0_v.mul_add(kv, total0);
        total1 = q1_v.mul_add(kv, total1);
    }
    (total0, total1)
}

/// Computes `(dot(q0, k), dot(q1, k))` from a shared `binary16` key vector `k_f16`.
pub(crate) fn dot_f16_pair_f32(q0: &[f32], q1: &[f32], k_f16: &[u16]) -> (f32, f32) {
    if q0.len() != k_f16.len() || q1.len() != k_f16.len() {
        return (0.0, 0.0);
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        // SAFETY: Baseline NEON is enabled by target configuration and slice lengths match.
        unsafe { dot_f16_pair_f32_neon(q0, q1, k_f16) }
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        dot_f16_pair_f32_portable(q0, q1, k_f16)
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
fn axpy_f16_pair_f32_neon(w0: f32, w1: f32, v_f16: &[u16], out0: &mut [f32], out1: &mut [f32]) {
    let w0_v = vdupq_n_f32(w0);
    let w1_v = vdupq_n_f32(w1);

    let (v8, v_rem8) = v_f16.as_chunks::<8>();
    let (o0_8, o0_rem8) = out0.as_chunks_mut::<8>();
    let (o1_8, o1_rem8) = out1.as_chunks_mut::<8>();
    for ((vc, d0c), d1c) in v8.iter().zip(o0_8.iter_mut()).zip(o1_8.iter_mut()) {
        let (v_lo, v_hi) = widen_f16x8_neon(vc);
        if let [d0_lo, d0_hi] = d0c.as_chunks_mut::<4>().0 {
            let r0 = vfmaq_f32(load_f32x4(d0_lo), w0_v, v_lo);
            let r1 = vfmaq_f32(load_f32x4(d0_hi), w0_v, v_hi);
            store_f32x4(r0, d0_lo);
            store_f32x4(r1, d0_hi);
        }
        if let [d1_lo, d1_hi] = d1c.as_chunks_mut::<4>().0 {
            let r0 = vfmaq_f32(load_f32x4(d1_lo), w1_v, v_lo);
            let r1 = vfmaq_f32(load_f32x4(d1_hi), w1_v, v_hi);
            store_f32x4(r0, d1_lo);
            store_f32x4(r1, d1_hi);
        }
    }

    let (v4, v_tail) = v_rem8.as_chunks::<4>();
    let (o0_4, o0_tail) = o0_rem8.as_chunks_mut::<4>();
    let (o1_4, o1_tail) = o1_rem8.as_chunks_mut::<4>();
    if let (Some(vc), Some(d0c), Some(d1c)) = (v4.first(), o0_4.first_mut(), o1_4.first_mut()) {
        let vv = widen_f16x4_neon(vc);
        let r0 = vfmaq_f32(load_f32x4(d0c), w0_v, vv);
        let r1 = vfmaq_f32(load_f32x4(d1c), w1_v, vv);
        store_f32x4(r0, d0c);
        store_f32x4(r1, d1c);
    }

    for ((&v_bits, d0), d1) in v_tail
        .iter()
        .zip(o0_tail.iter_mut())
        .zip(o1_tail.iter_mut())
    {
        let vv = bonsai_ternary::half_to_float(v_bits);
        *d0 = w0.mul_add(vv, *d0);
        *d1 = w1.mul_add(vv, *d1);
    }
}

#[cfg(any(test, not(all(target_arch = "aarch64", target_feature = "neon"))))]
pub(crate) fn axpy_f16_pair_f32_portable(
    w0: f32,
    w1: f32,
    v_f16: &[u16],
    out0: &mut [f32],
    out1: &mut [f32],
) {
    if out0.len() != v_f16.len() || out1.len() != v_f16.len() {
        return;
    }
    for ((&v_bits, d0), d1) in v_f16.iter().zip(out0.iter_mut()).zip(out1.iter_mut()) {
        let vv = bonsai_ternary::half_to_float(v_bits);
        *d0 = w0.mul_add(vv, *d0);
        *d1 = w1.mul_add(vv, *d1);
    }
}

/// Accumulates `out0 += w0 * v` and `out1 += w1 * v` from a shared `binary16` value vector `v_f16`.
pub(crate) fn axpy_f16_pair_f32(
    w0: f32,
    w1: f32,
    v_f16: &[u16],
    out0: &mut [f32],
    out1: &mut [f32],
) {
    if out0.len() != v_f16.len() || out1.len() != v_f16.len() {
        return;
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        // SAFETY: Baseline NEON is enabled by target configuration and slice lengths match.
        unsafe { axpy_f16_pair_f32_neon(w0, w1, v_f16, out0, out1) }
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        axpy_f16_pair_f32_portable(w0, w1, v_f16, out0, out1);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
fn gemm_f32_nonfma_4x16_neon(rows: [&[f32]; 4], act_panel: &[[f32; 16]], acc: &mut [[f32; 16]; 4]) {
    let k_len = act_panel.len();
    let mut a_acc: [[float32x4_t; 4]; 4] = std::array::from_fn(|r| {
        let quads = acc[r].as_chunks::<4>().0;
        [
            load_f32x4(&quads[0]),
            load_f32x4(&quads[1]),
            load_f32x4(&quads[2]),
            load_f32x4(&quads[3]),
        ]
    });

    #[inline]
    #[target_feature(enable = "neon")]
    fn step_lane<const LANE: i32>(
        a_acc: &mut [[float32x4_t; 4]; 4],
        w: [float32x4_t; 4],
        panel_k: &[f32; 16],
    ) {
        let a_quads = panel_k.as_chunks::<4>().0;
        let a0 = load_f32x4(&a_quads[0]);
        let a1 = load_f32x4(&a_quads[1]);
        let a2 = load_f32x4(&a_quads[2]);
        let a3 = load_f32x4(&a_quads[3]);
        for (acc_row, &wr) in a_acc.iter_mut().zip(w.iter()) {
            acc_row[0] = vaddq_f32(acc_row[0], vmulq_laneq_f32::<LANE>(a0, wr));
            acc_row[1] = vaddq_f32(acc_row[1], vmulq_laneq_f32::<LANE>(a1, wr));
            acc_row[2] = vaddq_f32(acc_row[2], vmulq_laneq_f32::<LANE>(a2, wr));
            acc_row[3] = vaddq_f32(acc_row[3], vmulq_laneq_f32::<LANE>(a3, wr));
        }
    }

    let (w0_chunks, _) = rows[0].as_chunks::<4>();
    let (w1_chunks, _) = rows[1].as_chunks::<4>();
    let (w2_chunks, _) = rows[2].as_chunks::<4>();
    let (w3_chunks, _) = rows[3].as_chunks::<4>();
    let chunk_count = w0_chunks.len();
    for c_idx in 0..chunk_count {
        let k = c_idx * 4;
        let w = [
            load_f32x4(&w0_chunks[c_idx]),
            load_f32x4(&w1_chunks[c_idx]),
            load_f32x4(&w2_chunks[c_idx]),
            load_f32x4(&w3_chunks[c_idx]),
        ];
        step_lane::<0>(&mut a_acc, w, &act_panel[k]);
        step_lane::<1>(&mut a_acc, w, &act_panel[k + 1]);
        step_lane::<2>(&mut a_acc, w, &act_panel[k + 2]);
        step_lane::<3>(&mut a_acc, w, &act_panel[k + 3]);
    }
    for (k, panel_k) in act_panel.iter().enumerate().skip(chunk_count * 4) {
        let a_quads = panel_k.as_chunks::<4>().0;
        let a0 = load_f32x4(&a_quads[0]);
        let a1 = load_f32x4(&a_quads[1]);
        let a2 = load_f32x4(&a_quads[2]);
        let a3 = load_f32x4(&a_quads[3]);
        for (r, acc_row) in a_acc.iter_mut().enumerate() {
            let wv = vdupq_n_f32(rows[r][k]);
            acc_row[0] = vaddq_f32(acc_row[0], vmulq_f32(a0, wv));
            acc_row[1] = vaddq_f32(acc_row[1], vmulq_f32(a1, wv));
            acc_row[2] = vaddq_f32(acc_row[2], vmulq_f32(a2, wv));
            acc_row[3] = vaddq_f32(acc_row[3], vmulq_f32(a3, wv));
        }
    }
    let _ = k_len;

    for (dst_row, src_row) in acc.iter_mut().zip(a_acc.iter()) {
        for (dst_quad, &src_quad) in dst_row.as_chunks_mut::<4>().0.iter_mut().zip(src_row) {
            store_f32x4(src_quad, dst_quad);
        }
    }
}

/// Exact portable non-FMA 4x16 `f32` GEMM accumulation matching scalar `acc[r][c] += w * a`.
#[cfg(any(test, not(all(target_arch = "aarch64", target_feature = "neon"))))]
pub(crate) fn gemm_f32_nonfma_4x16_portable(
    rows: [&[f32]; 4],
    act_panel: &[[f32; 16]],
    acc: &mut [[f32; 16]; 4],
) {
    let k_len = act_panel.len();
    if rows.iter().any(|r| r.len() != k_len) {
        return;
    }
    for (k, panel_k) in act_panel.iter().enumerate() {
        for r in 0..4 {
            let w = rows[r][k];
            for (dst, &a) in acc[r].iter_mut().zip(panel_k.iter()) {
                *dst += w * a;
            }
        }
    }
}

/// Accumulates a 4-row by 16-column strictly non-FMA `f32` GEMM tile
/// (`acc[r][c] += sum_k rows[r][k] * act_panel[k][c]`) in exact left-to-right `k` order.
pub(crate) fn gemm_f32_nonfma_4x16(
    rows: [&[f32]; 4],
    act_panel: &[[f32; 16]],
    acc: &mut [[f32; 16]; 4],
) {
    let k_len = act_panel.len();
    if rows.iter().any(|r| r.len() != k_len) {
        return;
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        // SAFETY: Baseline NEON is enabled by target configuration and slice lengths match `k_len`.
        unsafe { gemm_f32_nonfma_4x16_neon(rows, act_panel, acc) };
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        gemm_f32_nonfma_4x16_portable(rows, act_panel, acc);
    }
}
