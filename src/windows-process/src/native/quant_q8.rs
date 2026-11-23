//! Audited ARM64 `Q8_0` 8-row `SDOT` decode and `8x8`/`8x4`/`8x2` `I8MM` (`SMMLA`) GEMM kernels.
//!
//! Every public-to-crate entry point validates slice bounds and runtime CPU capability flags
//! before executing optional ARM64 instructions, preserving exact `0-ULP` bit parity with
//! [`bonsai_neon_dot::kquants::dot_q8_0_row`].

use bonsai_neon_dot::kquants::{
    Arm64QuantCaps, Q8_0_BLOCK_BYTES, Q8_0ActBlock, dot_q8_0_4rows, dot_q8_0_row,
};

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::{
    float32x4_t, int8x16_t, int32x4_t, uint8x16_t, vaddq_f32, vcombine_s8, vcvtq_f32_s32,
    vdotq_s32, vdupq_n_f32, vdupq_n_s32, vget_high_s8, vget_low_s8, vld1q_f32, vld1q_s8, vld1q_u8,
    vmulq_f32, vpaddq_s32, vreinterpretq_s8_u8, vst1q_f32, vzip1q_f32, vzip2q_f32,
};

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

        let d_act_pairs: [float32x4_t; COL_PAIRS] = std::array::from_fn(|cp| {
            let d0 = acts[cp * 2][idx].d;
            let d1 = acts[cp * 2 + 1][idx].d;
            load_f32x4(&[d0, d1, d0, d1])
        });

        let mut c = [[vdupq_n_s32(0); COL_PAIRS]; 4];
        let blks = [b0, b1, b2, b3, b4, b5, b6, b7];
        for half in 0..2 {
            let w_vecs: [int8x16_t; 8] = blks.map(|blk| {
                let halves = blk[2..34].as_chunks::<16>().0;
                vreinterpretq_s8_u8(load_u8x16(&halves[half]))
            });
            let w_lo = [
                vcombine_s8(vget_low_s8(w_vecs[0]), vget_low_s8(w_vecs[1])),
                vcombine_s8(vget_low_s8(w_vecs[2]), vget_low_s8(w_vecs[3])),
                vcombine_s8(vget_low_s8(w_vecs[4]), vget_low_s8(w_vecs[5])),
                vcombine_s8(vget_low_s8(w_vecs[6]), vget_low_s8(w_vecs[7])),
            ];
            let w_hi = [
                vcombine_s8(vget_high_s8(w_vecs[0]), vget_high_s8(w_vecs[1])),
                vcombine_s8(vget_high_s8(w_vecs[2]), vget_high_s8(w_vecs[3])),
                vcombine_s8(vget_high_s8(w_vecs[4]), vget_high_s8(w_vecs[5])),
                vcombine_s8(vget_high_s8(w_vecs[6]), vget_high_s8(w_vecs[7])),
            ];
            for cp in 0..COL_PAIRS {
                let a_even = load_s8x16(&acts[cp * 2][idx].qs.as_chunks::<16>().0[half]);
                let a_odd = load_s8x16(&acts[cp * 2 + 1][idx].qs.as_chunks::<16>().0[half]);
                let a_lo = vcombine_s8(vget_low_s8(a_even), vget_low_s8(a_odd));
                let a_hi = vcombine_s8(vget_high_s8(a_even), vget_high_s8(a_odd));
                for rp in 0..4 {
                    c[rp][cp] = smmla_s32(smmla_s32(c[rp][cp], w_lo[rp], a_lo), w_hi[rp], a_hi);
                }
            }
        }

        for rp in 0..4 {
            for cp in 0..COL_PAIRS {
                let dot_f32 = vcvtq_f32_s32(c[rp][cp]);
                let prod = vmulq_f32(d_w_pairs[rp], vmulq_f32(d_act_pairs[cp], dot_f32));
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
