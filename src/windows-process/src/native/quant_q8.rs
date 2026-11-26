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
    float32x4_t, int8x16_t, int32x4_t, uint8x16_t, vaddq_f32, vaddq_s32, vandq_u8, vcombine_s8,
    vcvtq_f32_s32, vdotq_s32, vdupq_n_f32, vdupq_n_s8, vdupq_n_s32, vdupq_n_u8, vget_high_s8,
    vget_low_s8, vld1q_f32, vld1q_s8, vld1q_s32, vld1q_u8, vmulq_f32, vmulq_s32, vorrq_u8,
    vpaddq_s32, vreinterpretq_s8_u8, vshlq_n_u8, vshrq_n_u8, vst1q_f32, vst1q_s32, vsubq_s8,
    vzip1q_f32, vzip2q_f32,
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
            prefetch_l1(blocks[blk_idx + 1].as_ptr());
        }
        let d_f16: [u16; 8] =
            std::array::from_fn(|r| u16::from_le_bytes([block[r * 2], block[r * 2 + 1]]));
        let dmin_f16: [u16; 8] =
            std::array::from_fn(|r| u16::from_le_bytes([block[16 + r * 2], block[16 + r * 2 + 1]]));
        let sc_bytes = &block[32..96];
        let m_bytes = &block[96..160];

        let mut mint = [[0_i32; TOKENS]; 8];
        for g in 0..8 {
            let m_row = &m_bytes[g * 8..(g + 1) * 8];
            let bsums: [i32; TOKENS] = std::array::from_fn(|t| {
                let ab = &acts[t][blk_idx];
                i32::from(ab.bsums[2 * g]) + i32::from(ab.bsums[2 * g + 1])
            });
            for (r, mint_row) in mint.iter_mut().enumerate() {
                let m_val = i32::from(m_row[r]);
                for (slot, &bs) in mint_row.iter_mut().zip(&bsums) {
                    *slot += m_val * bs;
                }
            }
        }

        let a_groups: [&[[i8; 16]]; TOKENS] =
            std::array::from_fn(|t| acts[t][blk_idx].qs.as_chunks::<16>().0);

        let mut sum_pairs = [[[0_i32; 4]; COL_PAIRS]; 4];
        for chunk in 0..4 {
            let qs_base = 160 + chunk * 256;
            for is_hi in [false, true] {
                let g_idx = chunk * 2 + usize::from(is_hi);
                let mut grp_acc = [[vdupq_n_s32(0); COL_PAIRS]; 4];
                for sub in 0..2 {
                    let a_sub = chunk * 4 + usize::from(is_hi) * 2 + sub;
                    let a_0: [int8x16_t; COL_PAIRS] = std::array::from_fn(|cp| {
                        let a_even = load_s8x16(&a_groups[cp * 2][a_sub]);
                        let a_odd = load_s8x16(&a_groups[cp * 2 + 1][a_sub]);
                        vcombine_s8(vget_low_s8(a_even), vget_low_s8(a_odd))
                    });
                    let a_1: [int8x16_t; COL_PAIRS] = std::array::from_fn(|cp| {
                        let a_even = load_s8x16(&a_groups[cp * 2][a_sub]);
                        let a_odd = load_s8x16(&a_groups[cp * 2 + 1][a_sub]);
                        vcombine_s8(vget_high_s8(a_even), vget_high_s8(a_odd))
                    });

                    for (pair, pair_acc) in grp_acc.iter_mut().enumerate() {
                        let o0 = qs_base + (pair * 2) * 32 + sub * 16;
                        let o1 = qs_base + (pair * 2 + 1) * 32 + sub * 16;
                        let raw0 = load_u8x16(&block[o0..o0 + 16].as_chunks::<16>().0[0]);
                        let raw1 = load_u8x16(&block[o1..o1 + 16].as_chunks::<16>().0[0]);
                        let (q0, q1) = if is_hi {
                            (
                                vreinterpretq_s8_u8(vshrq_n_u8(raw0, 4)),
                                vreinterpretq_s8_u8(vshrq_n_u8(raw1, 4)),
                            )
                        } else {
                            (
                                vreinterpretq_s8_u8(vandq_u8(raw0, mask_0f)),
                                vreinterpretq_s8_u8(vandq_u8(raw1, mask_0f)),
                            )
                        };
                        let w_0 = vcombine_s8(vget_low_s8(q0), vget_low_s8(q1));
                        let w_1 = vcombine_s8(vget_high_s8(q0), vget_high_s8(q1));
                        for cp in 0..COL_PAIRS {
                            pair_acc[cp] =
                                smmla_s32(smmla_s32(pair_acc[cp], w_0, a_0[cp]), w_1, a_1[cp]);
                        }
                    }
                }
                let sc_grp = &sc_bytes[g_idx * 8..(g_idx + 1) * 8];
                for pair in 0..4 {
                    let s0 = i32::from(sc_grp[pair * 2]);
                    let s1 = i32::from(sc_grp[pair * 2 + 1]);
                    let scale_v = load_s32x4(&[s0, s0, s1, s1]);
                    for cp in 0..COL_PAIRS {
                        let scaled = vmulq_s32(grp_acc[pair][cp], scale_v);
                        let mut vals = [0_i32; 4];
                        store_s32x4(scaled, &mut vals);
                        for (dst, src) in sum_pairs[pair][cp].iter_mut().zip(vals) {
                            *dst += src;
                        }
                    }
                }
            }
        }

        let d_w = bonsai_neon_dot::widen_f16_eight(&d_f16);
        let dmin_w = bonsai_neon_dot::widen_f16_eight(&dmin_f16);
        let d_act: [f64; TOKENS] = std::array::from_fn(|t| f64::from(acts[t][blk_idx].d));
        for (pair, pair_sums) in sum_pairs.iter().enumerate() {
            let r0 = pair * 2;
            let r1 = pair * 2 + 1;
            let dw0 = f64::from(d_w[r0 / 4][r0 % 4]);
            let dm0 = f64::from(dmin_w[r0 / 4][r0 % 4]);
            let dw1 = f64::from(d_w[r1 / 4][r1 % 4]);
            let dm1 = f64::from(dmin_w[r1 / 4][r1 % 4]);
            for (cp, vals) in pair_sums.iter().enumerate() {
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

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
fn unpack_q6_k_16(
    block: &[u8; 210],
    half_idx: usize,
    quarter: usize,
    sub: usize,
) -> (int8x16_t, i32) {
    let low_off = half_idx * 64 + (quarter % 2) * 32 + sub * 16;
    let high_off = 128 + half_idx * 32 + sub * 16;
    let ql = load_u8x16(&block[low_off..low_off + 16].as_chunks::<16>().0[0]);
    let low = if quarter < 2 {
        vandq_u8(ql, vdupq_n_u8(0x0f))
    } else {
        vshrq_n_u8(ql, 4)
    };
    let qh = load_u8x16(&block[high_off..high_off + 16].as_chunks::<16>().0[0]);
    let mask_03 = vdupq_n_u8(0x03);
    let hi = match quarter {
        0 => vandq_u8(qh, mask_03),
        1 => vandq_u8(vshrq_n_u8(qh, 2), mask_03),
        2 => vandq_u8(vshrq_n_u8(qh, 4), mask_03),
        _ => vandq_u8(vshrq_n_u8(qh, 6), mask_03),
    };
    let w = vsubq_s8(
        vreinterpretq_s8_u8(vorrq_u8(low, vshlq_n_u8(hi, 4))),
        vdupq_n_s8(32),
    );
    let s = i32::from(block[192 + half_idx * 8 + quarter * 2 + sub].cast_signed());
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
                prefetch_l1(r[blk_idx + 2].as_ptr());
            }
        }
        let blocks: [&[u8; 210]; 8] = std::array::from_fn(|r| &r_chunks[r][blk_idx]);
        let a_groups = ab.qs.as_chunks::<16>().0;
        let d_act = f64::from(ab.d);
        let d_bits: [u16; 8] = blocks.map(|b| u16::from_le_bytes([b[208], b[209]]));
        let d_weights = bonsai_neon_dot::widen_f16_eight(&d_bits);

        let mut sumi_lo = vdupq_n_s32(0);
        let mut sumi_hi = vdupq_n_s32(0);
        for half_idx in 0..2 {
            for quarter in 0..4 {
                let g_base = half_idx * 8 + quarter * 2;
                for sub in 0..2 {
                    let a = load_s8x16(&a_groups[g_base + sub]);
                    let (w0, s0) = unpack_q6_k_16(blocks[0], half_idx, quarter, sub);
                    let (w1, s1) = unpack_q6_k_16(blocks[1], half_idx, quarter, sub);
                    let (w2, s2) = unpack_q6_k_16(blocks[2], half_idx, quarter, sub);
                    let (w3, s3) = unpack_q6_k_16(blocks[3], half_idx, quarter, sub);
                    let d0 = vdotq_s32(zero, w0, a);
                    let d1 = vdotq_s32(zero, w1, a);
                    let d2 = vdotq_s32(zero, w2, a);
                    let d3 = vdotq_s32(zero, w3, a);
                    let dots_lo = vpaddq_s32(vpaddq_s32(d0, d1), vpaddq_s32(d2, d3));
                    sumi_lo = vaddq_s32(sumi_lo, vmulq_s32(dots_lo, load_s32x4(&[s0, s1, s2, s3])));

                    let (w4, s4) = unpack_q6_k_16(blocks[4], half_idx, quarter, sub);
                    let (w5, s5) = unpack_q6_k_16(blocks[5], half_idx, quarter, sub);
                    let (w6, s6) = unpack_q6_k_16(blocks[6], half_idx, quarter, sub);
                    let (w7, s7) = unpack_q6_k_16(blocks[7], half_idx, quarter, sub);
                    let d4 = vdotq_s32(zero, w4, a);
                    let d5 = vdotq_s32(zero, w5, a);
                    let d6 = vdotq_s32(zero, w6, a);
                    let d7 = vdotq_s32(zero, w7, a);
                    let dots_hi = vpaddq_s32(vpaddq_s32(d4, d5), vpaddq_s32(d6, d7));
                    sumi_hi = vaddq_s32(sumi_hi, vmulq_s32(dots_hi, load_s32x4(&[s4, s5, s6, s7])));
                }
            }
        }
        let mut s_lo = [0_i32; 4];
        let mut s_hi = [0_i32; 4];
        store_s32x4(sumi_lo, &mut s_lo);
        store_s32x4(sumi_hi, &mut s_hi);
        let sumi = [
            s_lo[0], s_lo[1], s_lo[2], s_lo[3], s_hi[0], s_hi[1], s_hi[2], s_hi[3],
        ];
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
                prefetch_l1(r[blk_idx + 2].as_ptr());
            }
        }
        let blocks: [&[u8; 210]; 8] = std::array::from_fn(|r| &r_chunks[r][blk_idx]);
        let d_bits: [u16; 8] = blocks.map(|b| u16::from_le_bytes([b[208], b[209]]));
        let d_weights = bonsai_neon_dot::widen_f16_eight(&d_bits);
        let a_groups: [&[[i8; 16]]; TOKENS] =
            std::array::from_fn(|t| acts[t][blk_idx].qs.as_chunks::<16>().0);

        let mut sum_pairs = [[vdupq_n_s32(0); COL_PAIRS]; 4];
        for half_idx in 0..2 {
            for quarter in 0..4 {
                let g_base = half_idx * 8 + quarter * 2;
                for sub in 0..2 {
                    let a_sub = g_base + sub;
                    let a_0: [int8x16_t; COL_PAIRS] = std::array::from_fn(|cp| {
                        let a_even = load_s8x16(&a_groups[cp * 2][a_sub]);
                        let a_odd = load_s8x16(&a_groups[cp * 2 + 1][a_sub]);
                        vcombine_s8(vget_low_s8(a_even), vget_low_s8(a_odd))
                    });
                    let a_1: [int8x16_t; COL_PAIRS] = std::array::from_fn(|cp| {
                        let a_even = load_s8x16(&a_groups[cp * 2][a_sub]);
                        let a_odd = load_s8x16(&a_groups[cp * 2 + 1][a_sub]);
                        vcombine_s8(vget_high_s8(a_even), vget_high_s8(a_odd))
                    });
                    for (pair, pair_acc) in sum_pairs.iter_mut().enumerate() {
                        let (q_even, s_even) =
                            unpack_q6_k_16(blocks[pair * 2], half_idx, quarter, sub);
                        let (q_odd, s_odd) =
                            unpack_q6_k_16(blocks[pair * 2 + 1], half_idx, quarter, sub);
                        let w_0 = vcombine_s8(vget_low_s8(q_even), vget_low_s8(q_odd));
                        let w_1 = vcombine_s8(vget_high_s8(q_even), vget_high_s8(q_odd));
                        let s_vec = load_s32x4(&[s_even, s_even, s_odd, s_odd]);
                        for cp in 0..COL_PAIRS {
                            let g =
                                smmla_s32(smmla_s32(vdupq_n_s32(0), w_0, a_0[cp]), w_1, a_1[cp]);
                            pair_acc[cp] = vaddq_s32(pair_acc[cp], vmulq_s32(g, s_vec));
                        }
                    }
                }
            }
        }

        let d_act: [f64; TOKENS] = std::array::from_fn(|t| f64::from(acts[t][blk_idx].d));
        for (pair, pair_vecs) in sum_pairs.iter().enumerate() {
            let r0 = pair * 2;
            let r1 = pair * 2 + 1;
            let dw0 = f64::from(d_weights[r0 / 4][r0 % 4]);
            let dw1 = f64::from(d_weights[r1 / 4][r1 % 4]);
            for (cp, &vec) in pair_vecs.iter().enumerate() {
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

/// Computes an 8-row by 2-token `Q6_K` tile using `I8MM` (`SMMLA`) or exact `SDOT` fallback.
pub(crate) fn gemm_q6_k_8x2(
    caps: Arm64QuantCaps,
    rows: [&[u8]; 8],
    acts: [&[Q8_KActBlock]; 2],
) -> [[f32; 2]; 8] {
    gemm_q6_k_8x_dispatch::<2, 1>(caps, rows, acts)
}
