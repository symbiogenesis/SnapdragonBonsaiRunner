//! Vectorized Chebyshev+Padé `erf`-based GELU (replacing iterative `f64` incomplete-gamma).

use fearless_simd::{Level, dispatch, f32x4, prelude::*};
use std::f32::consts::FRAC_1_SQRT_2;

const CHEB_COEFFS: [f32; 10] = [
    0.170_872_77,
    -0.822_152_23,
    1.488_515_87,
    -1.135_203_98,
    0.278_868_07,
    -0.186_288_06,
    0.096_784_18,
    0.374_091_96,
    1.000_023_68,
    -1.265_512_23,
];

#[allow(
    clippy::suboptimal_flops,
    reason = "Match scalar and SIMD non-FMA GELU rounding identically across targets."
)]
#[inline]
pub(super) fn gelu(value: f32) -> f32 {
    let xc = value.abs().min(7.5_f32);
    let z = xc * FRAC_1_SQRT_2;
    let t = 1.0_f32 / ((0.5_f32 * z) + 1.0_f32);
    let mut p = CHEB_COEFFS[0];
    for &c in &CHEB_COEFFS[1..] {
        p = (p * t) + c;
    }
    let w = (-(z * z)) + p;
    let u = w * 0.031_25_f32;
    let u2 = u * u;
    let pe = (((0.000_595_238_1_f32 * u2) + 0.107_142_86_f32) * u2) + 1.0_f32;
    let uq = u * ((0.011_904_762_f32 * u2) + 0.5_f32);
    let mut r = (pe + uq) / (pe - uq);
    for _ in 0..5 {
        r = r * r;
    }
    let erfc = t * r;
    let scale = if value < 0.0_f32 {
        0.5_f32 * erfc
    } else {
        1.0_f32 - (0.5_f32 * erfc)
    };
    value * scale
}

#[expect(
    clippy::inline_always,
    reason = "fearless_simd dispatch! requires inline(always) to inherit target_feature attributes."
)]
#[allow(
    clippy::suboptimal_flops,
    reason = "Match scalar and SIMD non-FMA GELU rounding identically across targets."
)]
#[inline(always)]
fn gelu_slice_simd<S: Simd>(simd: S, values: &mut [f32]) {
    let clamp_v = f32x4::splat(simd, 7.5_f32);
    let inv_sqrt2 = f32x4::splat(simd, FRAC_1_SQRT_2);
    let half = f32x4::splat(simd, 0.5_f32);
    let one = f32x4::splat(simd, 1.0_f32);
    let zero = f32x4::splat(simd, 0.0_f32);
    let inv32 = f32x4::splat(simd, 0.031_25_f32);
    let p2 = f32x4::splat(simd, 0.000_595_238_1_f32);
    let p1 = f32x4::splat(simd, 0.107_142_86_f32);
    let q1 = f32x4::splat(simd, 0.011_904_762_f32);

    let (chunks, tail) = values.as_chunks_mut::<4>();
    for chunk in chunks {
        let x = f32x4::from_slice(simd, chunk);
        let xc = x.abs().min(clamp_v);
        let z = xc * inv_sqrt2;
        let t = one / ((half * z) + one);
        let mut p = f32x4::splat(simd, CHEB_COEFFS[0]);
        for &c in &CHEB_COEFFS[1..] {
            p = (p * t) + f32x4::splat(simd, c);
        }
        let w = zero - (z * z) + p;
        let u = w * inv32;
        let u2 = u * u;
        let pe = (((p2 * u2) + p1) * u2) + one;
        let uq = u * ((q1 * u2) + half);
        let mut r = (pe + uq) / (pe - uq);
        for _ in 0..5 {
            r = r * r;
        }
        let erfc = t * r;
        let neg_scale = half * erfc;
        let pos_scale = one - neg_scale;
        let scale = x.simd_lt(zero).select(neg_scale, pos_scale);
        (x * scale).store_slice(chunk);
    }
    for value in tail {
        *value = gelu(*value);
    }
}

pub(super) fn gelu_slice_in_place(values: &mut [f32]) {
    let level: Level = crate::packed::simd_level();
    dispatch!(level, simd => gelu_slice_simd(simd, values));
}
