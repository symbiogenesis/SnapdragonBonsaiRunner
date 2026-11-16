//! Audited ARM64 NEON (`FCVTL`/`FCVTN`), `dotprod` (`vdotq_s32`) and `i8mm`
//! (`smmla`) primitives behind checked slice and fixed-array boundaries.

#![expect(
    clippy::redundant_pub_crate,
    reason = "Keep ARM64 SIMD helpers private to the crate boundary."
)]

use core::arch::aarch64::{
    float32x4_t, int8x16_t, int16x8_t, int32x4_t, uint8x16_t, uint16x4_t, vabsq_f32, vaddq_s32,
    vaddvq_s16, vaddvq_s32, vand_u16, vandq_u8, vcombine_s8, vcombine_s16, vcvt_f16_f32,
    vcvtnq_s32_f32, vdotq_s32, vdup_n_u16, vdupq_n_f32, vdupq_n_s8, vdupq_n_s32, vdupq_n_u8,
    vget_high_s8, vget_low_s8, vld1q_f32, vld1q_s8, vld1q_s32, vld1q_u8, vmaxq_f32, vmaxq_s8,
    vmaxv_u16, vmaxvq_f32, vmulq_f32, vmulq_s32, vorrq_u8, vpaddlq_s8, vqmovn_s16, vqmovn_s32,
    vreinterpret_u16_f16, vreinterpretq_s8_u8, vshlq_n_u8, vshrq_n_u8, vst1_u16, vst1q_s8,
    vst1q_s32, vsubq_s8,
};

use crate::quant_kernels::{
    Q4_K_REPACK8_BLOCK_BYTES, Q8_0ActBlock, Q8_KActBlock, f16_bits_to_f32, f64_to_f32,
    unpack_q4_k_scale_min,
};

#[inline]
#[target_feature(enable = "neon")]
fn load_f32x4(values: &[f32; 4]) -> float32x4_t {
    // SAFETY: `values` is a live 4-element `f32` array; unaligned loads are valid.
    unsafe { vld1q_f32(values.as_ptr()) }
}

#[inline]
#[target_feature(enable = "neon")]
fn store_u16x4(vec: uint16x4_t, out: &mut [u16; 4]) {
    // SAFETY: `out` is a unique mutable 4-element `u16` array.
    unsafe { vst1_u16(out.as_mut_ptr(), vec) };
}

#[inline]
#[target_feature(enable = "neon")]
fn load_u8x16(bytes: &[u8; 16]) -> uint8x16_t {
    // SAFETY: `bytes` is a live 16-byte array; `vld1q_u8` accepts unaligned pointers.
    unsafe { vld1q_u8(bytes.as_ptr()) }
}

#[inline]
#[target_feature(enable = "neon")]
fn load_s8x16(bytes: &[i8; 16]) -> int8x16_t {
    // SAFETY: `bytes` is a live 16-byte `i8` array.
    unsafe { vld1q_s8(bytes.as_ptr()) }
}

#[inline]
#[target_feature(enable = "neon")]
fn store_s8x16(vec: int8x16_t, out: &mut [i8; 16]) {
    // SAFETY: `out` is a unique mutable 16-byte `i8` array.
    unsafe { vst1q_s8(out.as_mut_ptr(), vec) };
}

#[inline]
#[target_feature(enable = "neon")]
fn store_s32x4(vec: int32x4_t, out: &mut [i32; 4]) {
    // SAFETY: `out` is a unique mutable 4-element `i32` array.
    unsafe { vst1q_s32(out.as_mut_ptr(), vec) };
}

#[inline]
#[target_feature(enable = "neon")]
fn load_s32x4(values: &[i32; 4]) -> int32x4_t {
    // SAFETY: `values` is a live 4-element `i32` array.
    unsafe { vld1q_s32(values.as_ptr()) }
}

#[cfg(target_feature = "neon")]
#[inline]
pub(crate) fn narrow_f32_slice_to_f16(src: &[f32], dst: &mut [u16]) -> bool {
    // SAFETY: Compiled only for targets with baseline ARM64 NEON; all slice extents checked inside.
    unsafe { narrow_f32_slice_to_f16_neon(src, dst) }
}

#[cfg(target_feature = "neon")]
#[target_feature(enable = "neon")]
fn narrow_f32_slice_to_f16_neon(src: &[f32], dst: &mut [u16]) -> bool {
    if src.len() != dst.len() {
        return false;
    }
    let (src_chunks, src_tail) = src.as_chunks::<4>();
    let (dst_chunks, dst_tail) = dst.as_chunks_mut::<4>();
    for (s, d) in src_chunks.iter().zip(dst_chunks.iter_mut()) {
        if !s.iter().all(|v| v.is_finite()) {
            return false;
        }
        let v = load_f32x4(s);
        let bits = vreinterpret_u16_f16(vcvt_f16_f32(v));
        if vmaxv_u16(vand_u16(bits, vdup_n_u16(0x7fff))) >= 0x7c00 {
            return false;
        }
        store_u16x4(bits, d);
    }
    for (&v, d) in src_tail.iter().zip(dst_tail.iter_mut()) {
        let Some(bits) = crate::quant_kernels::scalar_narrow_f32_to_f16(v) else {
            return false;
        };
        *d = bits;
    }
    true
}

#[cfg(target_feature = "neon")]
pub(crate) fn quantize_row_q8_0(input: &[f32], out: &mut [Q8_0ActBlock]) {
    // SAFETY: Compiled only for targets with baseline ARM64 NEON.
    unsafe { quantize_row_q8_0_neon(input, out) }
}

#[cfg(target_feature = "neon")]
#[target_feature(enable = "neon")]
fn quantize_row_q8_0_neon(input: &[f32], out: &mut [Q8_0ActBlock]) {
    let (chunks, _) = input.as_chunks::<32>();
    for (src, block) in chunks.iter().zip(out.iter_mut()) {
        let quads = src.as_chunks::<4>().0;
        let mut max_v = vdupq_n_f32(0.0);
        for q in quads {
            max_v = vmaxq_f32(max_v, vabsq_f32(load_f32x4(q)));
        }
        let amax = vmaxvq_f32(max_v);
        if amax == 0.0 || !amax.is_finite() {
            block.d = 0.0;
            block.qs = [0; 32];
            continue;
        }
        let d = amax / 127.0_f32;
        let id_v = vdupq_n_f32(127.0_f32 / amax);
        let min_q = vdupq_n_s8(-127);
        for half in 0..2 {
            let base = half * 4;
            let q0 = vcvtnq_s32_f32(vmulq_f32(load_f32x4(&quads[base]), id_v));
            let q1 = vcvtnq_s32_f32(vmulq_f32(load_f32x4(&quads[base + 1]), id_v));
            let q2 = vcvtnq_s32_f32(vmulq_f32(load_f32x4(&quads[base + 2]), id_v));
            let q3 = vcvtnq_s32_f32(vmulq_f32(load_f32x4(&quads[base + 3]), id_v));
            let i16_01: int16x8_t = vcombine_s16(vqmovn_s32(q0), vqmovn_s32(q1));
            let i16_23: int16x8_t = vcombine_s16(vqmovn_s32(q2), vqmovn_s32(q3));
            let packed = vmaxq_s8(vcombine_s8(vqmovn_s16(i16_01), vqmovn_s16(i16_23)), min_q);
            let out_halves = block.qs.as_chunks_mut::<16>().0;
            store_s8x16(packed, &mut out_halves[half]);
        }
        block.d = d;
    }
}

#[cfg(target_feature = "neon")]
pub(crate) fn quantize_row_q8_k(input: &[f32], out: &mut [Q8_KActBlock]) {
    // SAFETY: Compiled only for targets with baseline ARM64 NEON.
    unsafe { quantize_row_q8_k_neon(input, out) }
}

#[cfg(target_feature = "neon")]
#[target_feature(enable = "neon")]
fn quantize_row_q8_k_neon(input: &[f32], out: &mut [Q8_KActBlock]) {
    let (chunks, _) = input.as_chunks::<256>();
    for (src, block) in chunks.iter().zip(out.iter_mut()) {
        let quads = src.as_chunks::<4>().0;
        let mut max_v0 = vdupq_n_f32(0.0);
        let mut max_v1 = vdupq_n_f32(0.0);
        let (quad_pairs, _) = quads.as_chunks::<2>();
        for pair in quad_pairs {
            max_v0 = vmaxq_f32(max_v0, vabsq_f32(load_f32x4(&pair[0])));
            max_v1 = vmaxq_f32(max_v1, vabsq_f32(load_f32x4(&pair[1])));
        }
        let amax = vmaxvq_f32(vmaxq_f32(max_v0, max_v1));
        if amax == 0.0 || !amax.is_finite() {
            block.d = 0.0;
            block.qs = [0; 256];
            block.bsums = [0; 16];
            continue;
        }
        let d = amax / 127.0_f32;
        let id_v = vdupq_n_f32(127.0_f32 / amax);
        let min_q = vdupq_n_s8(-127);
        let qs_groups = block.qs.as_chunks_mut::<16>().0;
        for (g, (q_out, bsum_slot)) in qs_groups.iter_mut().zip(block.bsums.iter_mut()).enumerate()
        {
            let base = g * 4;
            let q0 = vcvtnq_s32_f32(vmulq_f32(load_f32x4(&quads[base]), id_v));
            let q1 = vcvtnq_s32_f32(vmulq_f32(load_f32x4(&quads[base + 1]), id_v));
            let q2 = vcvtnq_s32_f32(vmulq_f32(load_f32x4(&quads[base + 2]), id_v));
            let q3 = vcvtnq_s32_f32(vmulq_f32(load_f32x4(&quads[base + 3]), id_v));
            let i16_01 = vcombine_s16(vqmovn_s32(q0), vqmovn_s32(q1));
            let i16_23 = vcombine_s16(vqmovn_s32(q2), vqmovn_s32(q3));
            let packed = vmaxq_s8(vcombine_s8(vqmovn_s16(i16_01), vqmovn_s16(i16_23)), min_q);
            store_s8x16(packed, q_out);
            *bsum_slot = vaddvq_s16(vpaddlq_s8(packed));
        }
        block.d = d;
    }
}

#[inline]
#[target_feature(enable = "neon,i8mm")]
fn smmla_s32(mut acc: int32x4_t, w: int8x16_t, a: int8x16_t) -> int32x4_t {
    // SAFETY: Caller verified `i8mm`; `smmla` is a pure register-to-register instruction.
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

#[inline]
#[target_feature(enable = "neon,dotprod")]
fn dot32_s8_vec(w0: int8x16_t, w1: int8x16_t, a0: int8x16_t, a1: int8x16_t) -> int32x4_t {
    let acc0 = vdotq_s32(vdupq_n_s32(0), w0, a0);
    vdotq_s32(acc0, w1, a1)
}

#[target_feature(enable = "neon,dotprod")]
pub(crate) fn dot_q4_k_row_dotprod(w_row: &[u8], act: &[Q8_KActBlock]) -> f32 {
    let mut acc = 0.0_f64;
    let mask_0f = vdupq_n_u8(0x0f);
    for (block, ab) in w_row.as_chunks::<144>().0.iter().zip(act.iter()) {
        let a_groups = ab.qs.as_chunks::<16>().0;
        let mut mint = 0_i32;
        let mut sumi = 0_i32;
        let d_act = f64::from(ab.d);
        let d_w = f64::from(f16_bits_to_f32([block[0], block[1]]));
        let dmin_w = f64::from(f16_bits_to_f32([block[2], block[3]]));
        let mut sc = [0_i32; 8];
        for g in 0..8 {
            let (s, m) = unpack_q4_k_scale_min(g, &block[4..16]);
            sc[g] = i32::from(s);
            let b_pair = i32::from(ab.bsums[2 * g]) + i32::from(ab.bsums[2 * g + 1]);
            mint += i32::from(m) * b_pair;
        }
        for chunk in 0..4 {
            let qs_off = 16 + chunk * 32;
            let raw0 = load_u8x16(&block[qs_off..qs_off + 16].as_chunks::<16>().0[0]);
            let raw1 = load_u8x16(&block[qs_off + 16..qs_off + 32].as_chunks::<16>().0[0]);
            let lo0 = vreinterpretq_s8_u8(vandq_u8(raw0, mask_0f));
            let lo1 = vreinterpretq_s8_u8(vandq_u8(raw1, mask_0f));
            let hi0 = vreinterpretq_s8_u8(vshrq_n_u8(raw0, 4));
            let hi1 = vreinterpretq_s8_u8(vshrq_n_u8(raw1, 4));
            let d_lo = vaddvq_s32(dot32_s8_vec(
                lo0,
                lo1,
                load_s8x16(&a_groups[chunk * 4]),
                load_s8x16(&a_groups[chunk * 4 + 1]),
            ));
            let d_hi = vaddvq_s32(dot32_s8_vec(
                hi0,
                hi1,
                load_s8x16(&a_groups[chunk * 4 + 2]),
                load_s8x16(&a_groups[chunk * 4 + 3]),
            ));
            sumi += d_lo * sc[chunk * 2] + d_hi * sc[chunk * 2 + 1];
        }
        acc += (d_w * d_act) * f64::from(sumi) - (dmin_w * d_act) * f64::from(mint);
    }
    f64_to_f32(acc)
}

#[target_feature(enable = "neon,dotprod")]
pub(crate) fn dot_q4_k_4rows_dotprod(rows: [&[u8]; 4], act: &[Q8_KActBlock]) -> [f32; 4] {
    let mut acc = [0.0_f64; 4];
    let r0 = rows[0].as_chunks::<144>().0;
    let r1 = rows[1].as_chunks::<144>().0;
    let r2 = rows[2].as_chunks::<144>().0;
    let r3 = rows[3].as_chunks::<144>().0;
    let mask_0f = vdupq_n_u8(0x0f);
    for (idx, ab) in act.iter().enumerate() {
        let blocks = [&r0[idx], &r1[idx], &r2[idx], &r3[idx]];
        let a_groups = ab.qs.as_chunks::<16>().0;
        let bsum_pairs: [i32; 8] =
            std::array::from_fn(|g| i32::from(ab.bsums[2 * g]) + i32::from(ab.bsums[2 * g + 1]));
        let d_act = f64::from(ab.d);
        let d_weights = blocks.map(|b| f64::from(f16_bits_to_f32([b[0], b[1]])));
        let dmin_weights = blocks.map(|b| f64::from(f16_bits_to_f32([b[2], b[3]])));
        let scales = blocks.map(|b| {
            let mut sc = [0_i32; 8];
            let mut m = [0_i32; 8];
            for g in 0..8 {
                let (s, mv) = unpack_q4_k_scale_min(g, &b[4..16]);
                sc[g] = i32::from(s);
                m[g] = i32::from(mv);
            }
            (sc, m)
        });
        let mut mint = [0_i32; 4];
        for r in 0..4 {
            let mut m_sum = 0_i32;
            for g in 0..8 {
                m_sum += scales[r].1[g] * bsum_pairs[g];
            }
            mint[r] = m_sum;
        }
        let mut sumi = [0_i32; 4];
        for chunk in 0..4 {
            let a_lo0 = load_s8x16(&a_groups[chunk * 4]);
            let a_lo1 = load_s8x16(&a_groups[chunk * 4 + 1]);
            let a_hi0 = load_s8x16(&a_groups[chunk * 4 + 2]);
            let a_hi1 = load_s8x16(&a_groups[chunk * 4 + 3]);
            let qs_off = 16 + chunk * 32;
            for r in 0..4 {
                let raw0 = load_u8x16(&blocks[r][qs_off..qs_off + 16].as_chunks::<16>().0[0]);
                let raw1 = load_u8x16(&blocks[r][qs_off + 16..qs_off + 32].as_chunks::<16>().0[0]);
                let lo0 = vreinterpretq_s8_u8(vandq_u8(raw0, mask_0f));
                let lo1 = vreinterpretq_s8_u8(vandq_u8(raw1, mask_0f));
                let hi0 = vreinterpretq_s8_u8(vshrq_n_u8(raw0, 4));
                let hi1 = vreinterpretq_s8_u8(vshrq_n_u8(raw1, 4));
                let dot_lo = vaddvq_s32(dot32_s8_vec(lo0, lo1, a_lo0, a_lo1));
                let dot_hi = vaddvq_s32(dot32_s8_vec(hi0, hi1, a_hi0, a_hi1));
                sumi[r] += dot_lo * scales[r].0[chunk * 2] + dot_hi * scales[r].0[chunk * 2 + 1];
            }
        }
        for r in 0..4 {
            acc[r] += (d_weights[r] * d_act) * f64::from(sumi[r])
                - (dmin_weights[r] * d_act) * f64::from(mint[r]);
        }
    }
    acc.map(f64_to_f32)
}

#[target_feature(enable = "neon,i8mm")]
pub(crate) fn gemm_q4_k_4x2_i8mm(
    rows: [&[u8]; 4],
    act0: &[Q8_KActBlock],
    act1: &[Q8_KActBlock],
) -> [[f32; 2]; 4] {
    let mut acc = [[0.0_f64; 2]; 4];
    let r0 = rows[0].as_chunks::<144>().0;
    let r1 = rows[1].as_chunks::<144>().0;
    let r2 = rows[2].as_chunks::<144>().0;
    let r3 = rows[3].as_chunks::<144>().0;
    let mask_0f = vdupq_n_u8(0x0f);
    for (idx, (ab0, ab1)) in act0.iter().zip(act1.iter()).enumerate() {
        let blocks = [&r0[idx], &r1[idx], &r2[idx], &r3[idx]];
        let a0_g = ab0.qs.as_chunks::<16>().0;
        let a1_g = ab1.qs.as_chunks::<16>().0;
        let bsum0: [i32; 8] =
            std::array::from_fn(|g| i32::from(ab0.bsums[2 * g]) + i32::from(ab0.bsums[2 * g + 1]));
        let bsum1: [i32; 8] =
            std::array::from_fn(|g| i32::from(ab1.bsums[2 * g]) + i32::from(ab1.bsums[2 * g + 1]));
        let d_act0 = f64::from(ab0.d);
        let d_act1 = f64::from(ab1.d);
        let d_weights = blocks.map(|b| f64::from(f16_bits_to_f32([b[0], b[1]])));
        let dmin_weights = blocks.map(|b| f64::from(f16_bits_to_f32([b[2], b[3]])));
        let scales = blocks.map(|b| {
            let mut sc = [0_i32; 8];
            let mut m = [0_i32; 8];
            for g in 0..8 {
                let (s, mv) = unpack_q4_k_scale_min(g, &b[4..16]);
                sc[g] = i32::from(s);
                m[g] = i32::from(mv);
            }
            (sc, m)
        });
        let mut mint = [[0_i32; 2]; 4];
        for r in 0..4 {
            let mut m0 = 0_i32;
            let mut m1 = 0_i32;
            for g in 0..8 {
                m0 += scales[r].1[g] * bsum0[g];
                m1 += scales[r].1[g] * bsum1[g];
            }
            mint[r] = [m0, m1];
        }
        let mut sum01 = vdupq_n_s32(0);
        let mut sum23 = vdupq_n_s32(0);
        for chunk in 0..4 {
            let qs_off = 16 + chunk * 32;
            let mut lo01 = vdupq_n_s32(0);
            let mut lo23 = vdupq_n_s32(0);
            let mut hi01 = vdupq_n_s32(0);
            let mut hi23 = vdupq_n_s32(0);
            for sub in 0..2 {
                let a0_lo = load_s8x16(&a0_g[chunk * 4 + sub]);
                let a1_lo = load_s8x16(&a1_g[chunk * 4 + sub]);
                let a_lo_0 = vcombine_s8(vget_low_s8(a0_lo), vget_low_s8(a1_lo));
                let a_lo_1 = vcombine_s8(vget_high_s8(a0_lo), vget_high_s8(a1_lo));

                let a0_hi = load_s8x16(&a0_g[chunk * 4 + 2 + sub]);
                let a1_hi = load_s8x16(&a1_g[chunk * 4 + 2 + sub]);
                let a_hi_0 = vcombine_s8(vget_low_s8(a0_hi), vget_low_s8(a1_hi));
                let a_hi_1 = vcombine_s8(vget_high_s8(a0_hi), vget_high_s8(a1_hi));

                let off = qs_off + sub * 16;
                let r0_raw = load_u8x16(&blocks[0][off..off + 16].as_chunks::<16>().0[0]);
                let r1_raw = load_u8x16(&blocks[1][off..off + 16].as_chunks::<16>().0[0]);
                let r2_raw = load_u8x16(&blocks[2][off..off + 16].as_chunks::<16>().0[0]);
                let r3_raw = load_u8x16(&blocks[3][off..off + 16].as_chunks::<16>().0[0]);

                let l0 = vreinterpretq_s8_u8(vandq_u8(r0_raw, mask_0f));
                let l1 = vreinterpretq_s8_u8(vandq_u8(r1_raw, mask_0f));
                let l2 = vreinterpretq_s8_u8(vandq_u8(r2_raw, mask_0f));
                let l3 = vreinterpretq_s8_u8(vandq_u8(r3_raw, mask_0f));
                lo01 = smmla_s32(lo01, vcombine_s8(vget_low_s8(l0), vget_low_s8(l1)), a_lo_0);
                lo01 = smmla_s32(
                    lo01,
                    vcombine_s8(vget_high_s8(l0), vget_high_s8(l1)),
                    a_lo_1,
                );
                lo23 = smmla_s32(lo23, vcombine_s8(vget_low_s8(l2), vget_low_s8(l3)), a_lo_0);
                lo23 = smmla_s32(
                    lo23,
                    vcombine_s8(vget_high_s8(l2), vget_high_s8(l3)),
                    a_lo_1,
                );

                let h0 = vreinterpretq_s8_u8(vshrq_n_u8(r0_raw, 4));
                let h1 = vreinterpretq_s8_u8(vshrq_n_u8(r1_raw, 4));
                let h2 = vreinterpretq_s8_u8(vshrq_n_u8(r2_raw, 4));
                let h3 = vreinterpretq_s8_u8(vshrq_n_u8(r3_raw, 4));
                hi01 = smmla_s32(hi01, vcombine_s8(vget_low_s8(h0), vget_low_s8(h1)), a_hi_0);
                hi01 = smmla_s32(
                    hi01,
                    vcombine_s8(vget_high_s8(h0), vget_high_s8(h1)),
                    a_hi_1,
                );
                hi23 = smmla_s32(hi23, vcombine_s8(vget_low_s8(h2), vget_low_s8(h3)), a_hi_0);
                hi23 = smmla_s32(
                    hi23,
                    vcombine_s8(vget_high_s8(h2), vget_high_s8(h3)),
                    a_hi_1,
                );
            }
            let g_lo = chunk * 2;
            let g_hi = chunk * 2 + 1;
            let sc_lo01 = load_s32x4(&[
                scales[0].0[g_lo],
                scales[0].0[g_lo],
                scales[1].0[g_lo],
                scales[1].0[g_lo],
            ]);
            let sc_hi01 = load_s32x4(&[
                scales[0].0[g_hi],
                scales[0].0[g_hi],
                scales[1].0[g_hi],
                scales[1].0[g_hi],
            ]);
            let sc_lo23 = load_s32x4(&[
                scales[2].0[g_lo],
                scales[2].0[g_lo],
                scales[3].0[g_lo],
                scales[3].0[g_lo],
            ]);
            let sc_hi23 = load_s32x4(&[
                scales[2].0[g_hi],
                scales[2].0[g_hi],
                scales[3].0[g_hi],
                scales[3].0[g_hi],
            ]);
            sum01 = vaddq_s32(sum01, vmulq_s32(lo01, sc_lo01));
            sum01 = vaddq_s32(sum01, vmulq_s32(hi01, sc_hi01));
            sum23 = vaddq_s32(sum23, vmulq_s32(lo23, sc_lo23));
            sum23 = vaddq_s32(sum23, vmulq_s32(hi23, sc_hi23));
        }
        let mut f01 = [0_i32; 4];
        let mut f23 = [0_i32; 4];
        store_s32x4(sum01, &mut f01);
        store_s32x4(sum23, &mut f23);
        let sums = [
            [f01[0], f01[1]],
            [f01[2], f01[3]],
            [f23[0], f23[1]],
            [f23[2], f23[3]],
        ];
        for r in 0..4 {
            acc[r][0] += (d_weights[r] * d_act0) * f64::from(sums[r][0])
                - (dmin_weights[r] * d_act0) * f64::from(mint[r][0]);
            acc[r][1] += (d_weights[r] * d_act1) * f64::from(sums[r][1])
                - (dmin_weights[r] * d_act1) * f64::from(mint[r][1]);
        }
    }
    acc.map(|pair| pair.map(f64_to_f32))
}

#[target_feature(enable = "neon,dotprod")]
pub(crate) fn dot_q4_k_repacked8_dotprod(strip: &[u8], act: &[Q8_KActBlock]) -> [f32; 8] {
    let mut acc = [0.0_f64; 8];
    let mask_0f = vdupq_n_u8(0x0f);
    for (block, ab) in strip
        .as_chunks::<Q4_K_REPACK8_BLOCK_BYTES>()
        .0
        .iter()
        .zip(act.iter())
    {
        let a_groups = ab.qs.as_chunks::<16>().0;
        let bsum_pairs: [i32; 8] =
            std::array::from_fn(|g| i32::from(ab.bsums[2 * g]) + i32::from(ab.bsums[2 * g + 1]));
        let d_act = f64::from(ab.d);
        let d_f16: [u16; 8] =
            std::array::from_fn(|r| u16::from_le_bytes([block[r * 2], block[r * 2 + 1]]));
        let dmin_f16: [u16; 8] =
            std::array::from_fn(|r| u16::from_le_bytes([block[16 + r * 2], block[16 + r * 2 + 1]]));
        let sc_bytes = &block[32..96];
        let m_bytes = &block[96..160];
        let mut mint = [0_i32; 8];
        for g in 0..8 {
            let b_pair = bsum_pairs[g];
            let m_row = &m_bytes[g * 8..(g + 1) * 8];
            for r in 0..8 {
                mint[r] += i32::from(m_row[r]) * b_pair;
            }
        }
        let mut sumi = [0_i32; 8];
        for chunk in 0..4 {
            let a_lo0 = load_s8x16(&a_groups[chunk * 4]);
            let a_lo1 = load_s8x16(&a_groups[chunk * 4 + 1]);
            let a_hi0 = load_s8x16(&a_groups[chunk * 4 + 2]);
            let a_hi1 = load_s8x16(&a_groups[chunk * 4 + 3]);
            let qs_base = 160 + chunk * 256;
            let sc_lo = &sc_bytes[(chunk * 2) * 8..(chunk * 2 + 1) * 8];
            let sc_hi = &sc_bytes[(chunk * 2 + 1) * 8..(chunk * 2 + 2) * 8];
            for r in 0..8 {
                let row_qs = qs_base + r * 32;
                let raw0 = load_u8x16(&block[row_qs..row_qs + 16].as_chunks::<16>().0[0]);
                let raw1 = load_u8x16(&block[row_qs + 16..row_qs + 32].as_chunks::<16>().0[0]);
                let lo0 = vreinterpretq_s8_u8(vandq_u8(raw0, mask_0f));
                let lo1 = vreinterpretq_s8_u8(vandq_u8(raw1, mask_0f));
                let hi0 = vreinterpretq_s8_u8(vshrq_n_u8(raw0, 4));
                let hi1 = vreinterpretq_s8_u8(vshrq_n_u8(raw1, 4));
                let d_lo = vaddvq_s32(dot32_s8_vec(lo0, lo1, a_lo0, a_lo1));
                let d_hi = vaddvq_s32(dot32_s8_vec(hi0, hi1, a_hi0, a_hi1));
                sumi[r] += d_lo * i32::from(sc_lo[r]) + d_hi * i32::from(sc_hi[r]);
            }
        }
        let d_w = bonsai_neon_dot::widen_f16_eight(&d_f16);
        let dmin_w = bonsai_neon_dot::widen_f16_eight(&dmin_f16);
        for r in 0..8 {
            let dw = f64::from(d_w[r / 4][r % 4]) * d_act;
            let dm = f64::from(dmin_w[r / 4][r % 4]) * d_act;
            acc[r] += dw * f64::from(sumi[r]) - dm * f64::from(mint[r]);
        }
    }
    acc.map(f64_to_f32)
}

#[target_feature(enable = "neon,i8mm")]
pub(crate) fn gemm_q4_k_repacked8_8x2_i8mm(
    strip: &[u8],
    act0: &[Q8_KActBlock],
    act1: &[Q8_KActBlock],
) -> [[f32; 2]; 8] {
    let mut acc = [[0.0_f64; 2]; 8];
    let mask_0f = vdupq_n_u8(0x0f);
    for (block, (ab0, ab1)) in strip
        .as_chunks::<Q4_K_REPACK8_BLOCK_BYTES>()
        .0
        .iter()
        .zip(act0.iter().zip(act1.iter()))
    {
        let a0_g = ab0.qs.as_chunks::<16>().0;
        let a1_g = ab1.qs.as_chunks::<16>().0;
        let bsum0: [i32; 8] =
            std::array::from_fn(|g| i32::from(ab0.bsums[2 * g]) + i32::from(ab0.bsums[2 * g + 1]));
        let bsum1: [i32; 8] =
            std::array::from_fn(|g| i32::from(ab1.bsums[2 * g]) + i32::from(ab1.bsums[2 * g + 1]));
        let d_act0 = f64::from(ab0.d);
        let d_act1 = f64::from(ab1.d);
        let d_f16: [u16; 8] =
            std::array::from_fn(|r| u16::from_le_bytes([block[r * 2], block[r * 2 + 1]]));
        let dmin_f16: [u16; 8] =
            std::array::from_fn(|r| u16::from_le_bytes([block[16 + r * 2], block[16 + r * 2 + 1]]));
        let sc_bytes = &block[32..96];
        let m_bytes = &block[96..160];
        let mut mint = [[0_i32; 2]; 8];
        for g in 0..8 {
            let b0 = bsum0[g];
            let b1 = bsum1[g];
            let m_row = &m_bytes[g * 8..(g + 1) * 8];
            for r in 0..8 {
                let m = i32::from(m_row[r]);
                mint[r][0] += m * b0;
                mint[r][1] += m * b1;
            }
        }
        let mut sum_pairs = [vdupq_n_s32(0); 4];
        for chunk in 0..4 {
            let mut lo_acc = [vdupq_n_s32(0); 4];
            let mut hi_acc = [vdupq_n_s32(0); 4];
            let qs_base = 160 + chunk * 256;
            for sub in 0..2 {
                let a0_lo = load_s8x16(&a0_g[chunk * 4 + sub]);
                let a1_lo = load_s8x16(&a1_g[chunk * 4 + sub]);
                let a_lo_0 = vcombine_s8(vget_low_s8(a0_lo), vget_low_s8(a1_lo));
                let a_lo_1 = vcombine_s8(vget_high_s8(a0_lo), vget_high_s8(a1_lo));

                let a0_hi = load_s8x16(&a0_g[chunk * 4 + 2 + sub]);
                let a1_hi = load_s8x16(&a1_g[chunk * 4 + 2 + sub]);
                let a_hi_0 = vcombine_s8(vget_low_s8(a0_hi), vget_low_s8(a1_hi));
                let a_hi_1 = vcombine_s8(vget_high_s8(a0_hi), vget_high_s8(a1_hi));

                for pair in 0..4 {
                    let o0 = qs_base + (pair * 2) * 32 + sub * 16;
                    let o1 = qs_base + (pair * 2 + 1) * 32 + sub * 16;
                    let raw0 = load_u8x16(&block[o0..o0 + 16].as_chunks::<16>().0[0]);
                    let raw1 = load_u8x16(&block[o1..o1 + 16].as_chunks::<16>().0[0]);
                    let l0 = vreinterpretq_s8_u8(vandq_u8(raw0, mask_0f));
                    let l1 = vreinterpretq_s8_u8(vandq_u8(raw1, mask_0f));
                    lo_acc[pair] = smmla_s32(
                        lo_acc[pair],
                        vcombine_s8(vget_low_s8(l0), vget_low_s8(l1)),
                        a_lo_0,
                    );
                    lo_acc[pair] = smmla_s32(
                        lo_acc[pair],
                        vcombine_s8(vget_high_s8(l0), vget_high_s8(l1)),
                        a_lo_1,
                    );

                    let h0 = vreinterpretq_s8_u8(vshrq_n_u8(raw0, 4));
                    let h1 = vreinterpretq_s8_u8(vshrq_n_u8(raw1, 4));
                    hi_acc[pair] = smmla_s32(
                        hi_acc[pair],
                        vcombine_s8(vget_low_s8(h0), vget_low_s8(h1)),
                        a_hi_0,
                    );
                    hi_acc[pair] = smmla_s32(
                        hi_acc[pair],
                        vcombine_s8(vget_high_s8(h0), vget_high_s8(h1)),
                        a_hi_1,
                    );
                }
            }
            let sc_lo = &sc_bytes[(chunk * 2) * 8..(chunk * 2 + 1) * 8];
            let sc_hi = &sc_bytes[(chunk * 2 + 1) * 8..(chunk * 2 + 2) * 8];
            for pair in 0..4 {
                let s0 = i32::from(sc_lo[pair * 2]);
                let s1 = i32::from(sc_lo[pair * 2 + 1]);
                let h0 = i32::from(sc_hi[pair * 2]);
                let h1 = i32::from(sc_hi[pair * 2 + 1]);
                let sv_lo = load_s32x4(&[s0, s0, s1, s1]);
                let sv_hi = load_s32x4(&[h0, h0, h1, h1]);
                sum_pairs[pair] = vaddq_s32(sum_pairs[pair], vmulq_s32(lo_acc[pair], sv_lo));
                sum_pairs[pair] = vaddq_s32(sum_pairs[pair], vmulq_s32(hi_acc[pair], sv_hi));
            }
        }
        let d_w = bonsai_neon_dot::widen_f16_eight(&d_f16);
        let dmin_w = bonsai_neon_dot::widen_f16_eight(&dmin_f16);
        for pair in 0..4 {
            let mut vals = [0_i32; 4];
            store_s32x4(sum_pairs[pair], &mut vals);
            let r0 = pair * 2;
            let r1 = pair * 2 + 1;
            let dw0 = f64::from(d_w[r0 / 4][r0 % 4]);
            let dm0 = f64::from(dmin_w[r0 / 4][r0 % 4]);
            let dw1 = f64::from(d_w[r1 / 4][r1 % 4]);
            let dm1 = f64::from(dmin_w[r1 / 4][r1 % 4]);
            acc[r0][0] +=
                (dw0 * d_act0) * f64::from(vals[0]) - (dm0 * d_act0) * f64::from(mint[r0][0]);
            acc[r0][1] +=
                (dw0 * d_act1) * f64::from(vals[1]) - (dm0 * d_act1) * f64::from(mint[r0][1]);
            acc[r1][0] +=
                (dw1 * d_act0) * f64::from(vals[2]) - (dm1 * d_act0) * f64::from(mint[r1][0]);
            acc[r1][1] +=
                (dw1 * d_act1) * f64::from(vals[3]) - (dm1 * d_act1) * f64::from(mint[r1][1]);
        }
    }
    acc.map(|pair| pair.map(f64_to_f32))
}

#[inline]
#[target_feature(enable = "neon")]
fn unpack_q6_k_32(
    block: &[u8; 210],
    half_idx: usize,
    quarter: usize,
) -> (int8x16_t, int8x16_t, i32, i32) {
    let ql_off = half_idx * 64 + (quarter % 2) * 32;
    let qh_off = 128 + half_idx * 32;
    let ql0 = load_u8x16(&block[ql_off..ql_off + 16].as_chunks::<16>().0[0]);
    let ql1 = load_u8x16(&block[ql_off + 16..ql_off + 32].as_chunks::<16>().0[0]);
    let low0 = if quarter < 2 {
        vandq_u8(ql0, vdupq_n_u8(0x0f))
    } else {
        vshrq_n_u8(ql0, 4)
    };
    let low1 = if quarter < 2 {
        vandq_u8(ql1, vdupq_n_u8(0x0f))
    } else {
        vshrq_n_u8(ql1, 4)
    };
    let qh0 = load_u8x16(&block[qh_off..qh_off + 16].as_chunks::<16>().0[0]);
    let qh1 = load_u8x16(&block[qh_off + 16..qh_off + 32].as_chunks::<16>().0[0]);
    let mask_03 = vdupq_n_u8(0x03);
    let hi0 = match quarter {
        0 => vandq_u8(qh0, mask_03),
        1 => vandq_u8(vshrq_n_u8(qh0, 2), mask_03),
        2 => vandq_u8(vshrq_n_u8(qh0, 4), mask_03),
        _ => vandq_u8(vshrq_n_u8(qh0, 6), mask_03),
    };
    let hi1 = match quarter {
        0 => vandq_u8(qh1, mask_03),
        1 => vandq_u8(vshrq_n_u8(qh1, 2), mask_03),
        2 => vandq_u8(vshrq_n_u8(qh1, 4), mask_03),
        _ => vandq_u8(vshrq_n_u8(qh1, 6), mask_03),
    };
    let bias = vdupq_n_s8(32);
    let w0 = vsubq_s8(
        vreinterpretq_s8_u8(vorrq_u8(low0, vshlq_n_u8(hi0, 4))),
        bias,
    );
    let w1 = vsubq_s8(
        vreinterpretq_s8_u8(vorrq_u8(low1, vshlq_n_u8(hi1, 4))),
        bias,
    );
    let s_base = 192 + half_idx * 8 + quarter * 2;
    let s0 = i32::from(block[s_base].cast_signed());
    let s1 = i32::from(block[s_base + 1].cast_signed());
    (w0, w1, s0, s1)
}

#[target_feature(enable = "neon,dotprod")]
pub(crate) fn dot_q6_k_row_dotprod(w_row: &[u8], act: &[Q8_KActBlock]) -> f32 {
    let mut acc = 0.0_f64;
    let zero = vdupq_n_s32(0);
    for (block, ab) in w_row.as_chunks::<210>().0.iter().zip(act.iter()) {
        let a_groups = ab.qs.as_chunks::<16>().0;
        let d_act = f64::from(ab.d);
        let d_w = f64::from(f16_bits_to_f32([block[208], block[209]]));
        let mut sumi = 0_i32;
        for half_idx in 0..2 {
            for quarter in 0..4 {
                let g_base = half_idx * 8 + quarter * 2;
                let a0 = load_s8x16(&a_groups[g_base]);
                let a1 = load_s8x16(&a_groups[g_base + 1]);
                let (w0, w1, s0, s1) = unpack_q6_k_32(block, half_idx, quarter);
                let d0 = vaddvq_s32(vdotq_s32(zero, w0, a0));
                let d1 = vaddvq_s32(vdotq_s32(zero, w1, a1));
                sumi += d0 * s0 + d1 * s1;
            }
        }
        acc += (d_w * d_act) * f64::from(sumi);
    }
    f64_to_f32(acc)
}

#[target_feature(enable = "neon,dotprod")]
pub(crate) fn dot_q6_k_4rows_dotprod(rows: [&[u8]; 4], act: &[Q8_KActBlock]) -> [f32; 4] {
    let mut acc = [0.0_f64; 4];
    let r0 = rows[0].as_chunks::<210>().0;
    let r1 = rows[1].as_chunks::<210>().0;
    let r2 = rows[2].as_chunks::<210>().0;
    let r3 = rows[3].as_chunks::<210>().0;
    let zero = vdupq_n_s32(0);
    for (idx, ab) in act.iter().enumerate() {
        let blocks = [&r0[idx], &r1[idx], &r2[idx], &r3[idx]];
        let a_groups = ab.qs.as_chunks::<16>().0;
        let d_act = f64::from(ab.d);
        let d_weights = blocks.map(|b| f64::from(f16_bits_to_f32([b[208], b[209]])));
        let mut sumi = [0_i32; 4];
        for half_idx in 0..2 {
            for quarter in 0..4 {
                let g_base = half_idx * 8 + quarter * 2;
                let a0 = load_s8x16(&a_groups[g_base]);
                let a1 = load_s8x16(&a_groups[g_base + 1]);
                for r in 0..4 {
                    let (w0, w1, s0, s1) = unpack_q6_k_32(blocks[r], half_idx, quarter);
                    let d0 = vaddvq_s32(vdotq_s32(zero, w0, a0));
                    let d1 = vaddvq_s32(vdotq_s32(zero, w1, a1));
                    sumi[r] += d0 * s0 + d1 * s1;
                }
            }
        }
        for r in 0..4 {
            acc[r] += (d_weights[r] * d_act) * f64::from(sumi[r]);
        }
    }
    acc.map(f64_to_f32)
}

#[target_feature(enable = "neon,i8mm")]
pub(crate) fn gemm_q6_k_4x2_i8mm(
    rows: [&[u8]; 4],
    act0: &[Q8_KActBlock],
    act1: &[Q8_KActBlock],
) -> [[f32; 2]; 4] {
    let mut acc = [[0.0_f64; 2]; 4];
    let r0 = rows[0].as_chunks::<210>().0;
    let r1 = rows[1].as_chunks::<210>().0;
    let r2 = rows[2].as_chunks::<210>().0;
    let r3 = rows[3].as_chunks::<210>().0;
    for (idx, (ab0, ab1)) in act0.iter().zip(act1.iter()).enumerate() {
        let blocks = [&r0[idx], &r1[idx], &r2[idx], &r3[idx]];
        let a0_g = ab0.qs.as_chunks::<16>().0;
        let a1_g = ab1.qs.as_chunks::<16>().0;
        let d_act0 = f64::from(ab0.d);
        let d_act1 = f64::from(ab1.d);
        let d_weights = blocks.map(|b| f64::from(f16_bits_to_f32([b[208], b[209]])));
        let mut sum01 = vdupq_n_s32(0);
        let mut sum23 = vdupq_n_s32(0);
        for half_idx in 0..2 {
            for quarter in 0..4 {
                let g_base = half_idx * 8 + quarter * 2;
                let a0_lo = load_s8x16(&a0_g[g_base]);
                let a1_lo = load_s8x16(&a1_g[g_base]);
                let a_g0_lo = vcombine_s8(vget_low_s8(a0_lo), vget_low_s8(a1_lo));
                let a_g0_hi = vcombine_s8(vget_high_s8(a0_lo), vget_high_s8(a1_lo));

                let a0_hi = load_s8x16(&a0_g[g_base + 1]);
                let a1_hi = load_s8x16(&a1_g[g_base + 1]);
                let a_g1_lo = vcombine_s8(vget_low_s8(a0_hi), vget_low_s8(a1_hi));
                let a_g1_hi = vcombine_s8(vget_high_s8(a0_hi), vget_high_s8(a1_hi));

                let (w0_0, w0_1, s0_0, s0_1) = unpack_q6_k_32(blocks[0], half_idx, quarter);
                let (w1_0, w1_1, s1_0, s1_1) = unpack_q6_k_32(blocks[1], half_idx, quarter);
                let (w2_0, w2_1, s2_0, s2_1) = unpack_q6_k_32(blocks[2], half_idx, quarter);
                let (w3_0, w3_1, s3_0, s3_1) = unpack_q6_k_32(blocks[3], half_idx, quarter);

                let g0_01 = smmla_s32(
                    smmla_s32(
                        vdupq_n_s32(0),
                        vcombine_s8(vget_low_s8(w0_0), vget_low_s8(w1_0)),
                        a_g0_lo,
                    ),
                    vcombine_s8(vget_high_s8(w0_0), vget_high_s8(w1_0)),
                    a_g0_hi,
                );
                let g1_01 = smmla_s32(
                    smmla_s32(
                        vdupq_n_s32(0),
                        vcombine_s8(vget_low_s8(w0_1), vget_low_s8(w1_1)),
                        a_g1_lo,
                    ),
                    vcombine_s8(vget_high_s8(w0_1), vget_high_s8(w1_1)),
                    a_g1_hi,
                );
                sum01 = vaddq_s32(
                    sum01,
                    vmulq_s32(g0_01, load_s32x4(&[s0_0, s0_0, s1_0, s1_0])),
                );
                sum01 = vaddq_s32(
                    sum01,
                    vmulq_s32(g1_01, load_s32x4(&[s0_1, s0_1, s1_1, s1_1])),
                );

                let g0_23 = smmla_s32(
                    smmla_s32(
                        vdupq_n_s32(0),
                        vcombine_s8(vget_low_s8(w2_0), vget_low_s8(w3_0)),
                        a_g0_lo,
                    ),
                    vcombine_s8(vget_high_s8(w2_0), vget_high_s8(w3_0)),
                    a_g0_hi,
                );
                let g1_23 = smmla_s32(
                    smmla_s32(
                        vdupq_n_s32(0),
                        vcombine_s8(vget_low_s8(w2_1), vget_low_s8(w3_1)),
                        a_g1_lo,
                    ),
                    vcombine_s8(vget_high_s8(w2_1), vget_high_s8(w3_1)),
                    a_g1_hi,
                );
                sum23 = vaddq_s32(
                    sum23,
                    vmulq_s32(g0_23, load_s32x4(&[s2_0, s2_0, s3_0, s3_0])),
                );
                sum23 = vaddq_s32(
                    sum23,
                    vmulq_s32(g1_23, load_s32x4(&[s2_1, s2_1, s3_1, s3_1])),
                );
            }
        }
        let mut f01 = [0_i32; 4];
        let mut f23 = [0_i32; 4];
        store_s32x4(sum01, &mut f01);
        store_s32x4(sum23, &mut f23);
        let sums = [
            [f01[0], f01[1]],
            [f01[2], f01[3]],
            [f23[0], f23[1]],
            [f23[2], f23[3]],
        ];
        for r in 0..4 {
            acc[r][0] += (d_weights[r] * d_act0) * f64::from(sums[r][0]);
            acc[r][1] += (d_weights[r] * d_act1) * f64::from(sums[r][1]);
        }
    }
    acc.map(|pair| pair.map(f64_to_f32))
}
