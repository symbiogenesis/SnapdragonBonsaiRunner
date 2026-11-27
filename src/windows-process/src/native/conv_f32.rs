//! Audited ARM64 NEON `vfmaq_laneq_f32` `16oc x 4f` (`1..=4f`) `f32` outer-product
//! convolution/GEMM micro-kernels with exact sequential-FMA portable fallback.

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::{
    float32x4_t, vdupq_n_f32, vfmaq_laneq_f32, vfmaq_n_f32, vld1q_f32, vst1q_f32,
};

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
fn zero_f32x4() -> float32x4_t {
    // SAFETY: `vdupq_n_f32` is a pure register broadcast on baseline ARM64 NEON.
    unsafe { vdupq_n_f32(0.0_f32) }
}

#[cfg(target_arch = "aarch64")]
#[expect(
    clippy::inline_always,
    reason = "Const lane index must inline into the outer-product FMA loop so vfmaq_laneq_f32 compiles to a single instruction."
)]
#[inline(always)]
fn fma_lane_f32<const LANE: i32>(acc: float32x4_t, w: float32x4_t, x: float32x4_t) -> float32x4_t {
    // SAFETY: `vfmaq_laneq_f32` is a pure baseline ARM64 NEON register instruction with compile-time lane `0..=3`.
    unsafe { vfmaq_laneq_f32::<LANE>(acc, w, x) }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn fma_n_f32(acc: float32x4_t, w: float32x4_t, x: f32) -> float32x4_t {
    // SAFETY: `vfmaq_n_f32` is a pure baseline ARM64 NEON register instruction.
    unsafe { vfmaq_n_f32(acc, w, x) }
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

#[cfg(any(not(target_arch = "aarch64"), test))]
pub(crate) fn gemm_f32_fma_16oc_portable<const F: usize>(
    w_ic_16: &[f32],
    x_rows: [&[f32]; F],
    in_c: usize,
    acc: &mut [[f32; 16]; F],
) {
    let Some(expected_w) = in_c.checked_mul(16) else {
        return;
    };
    if in_c == 0 || w_ic_16.len() < expected_w || x_rows.iter().any(|x| x.len() < in_c) {
        return;
    }
    let (w_steps, _) = w_ic_16[..expected_w].as_chunks::<16>();
    for (ic, w_16) in w_steps.iter().enumerate() {
        for f in 0..F {
            let x_val = x_rows[f][ic];
            for (oc, &w_val) in w_16.iter().enumerate() {
                acc[f][oc] = x_val.mul_add(w_val, acc[f][oc]);
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub(crate) fn gemm_f32_fma_16oc<const F: usize>(
    w_ic_16: &[f32],
    x_rows: [&[f32]; F],
    in_c: usize,
    acc: &mut [[f32; 16]; F],
) {
    let Some(expected_w) = in_c.checked_mul(16) else {
        return;
    };
    if in_c == 0 || w_ic_16.len() < expected_w || x_rows.iter().any(|x| x.len() < in_c) {
        return;
    }

    let mut a: [[float32x4_t; 4]; F] = core::array::from_fn(|f| {
        let (quads, _) = acc[f].as_chunks::<4>();
        [
            load_f32x4(&quads[0]),
            load_f32x4(&quads[1]),
            load_f32x4(&quads[2]),
            load_f32x4(&quads[3]),
        ]
    });

    let w_used = &w_ic_16[..expected_w];
    let (w_quad_ics, w_tail_ics) = w_used.as_chunks::<64>();
    let quad_len = w_quad_ics.len();
    let x_quad_ics: [&[[f32; 4]]; F] =
        core::array::from_fn(|f| x_rows[f][..in_c].as_chunks::<4>().0);

    for (idx, w_64) in w_quad_ics.iter().enumerate() {
        if idx + 2 < quad_len {
            let next_w = &w_quad_ics[idx + 2];
            prefetch_l1(next_w[0..].as_ptr().cast::<u8>());
            prefetch_l1(next_w[16..].as_ptr().cast::<u8>());
            prefetch_l1(next_w[32..].as_ptr().cast::<u8>());
            prefetch_l1(next_w[48..].as_ptr().cast::<u8>());
        }
        let xv: [float32x4_t; F] = core::array::from_fn(|f| load_f32x4(&x_quad_ics[f][idx]));
        let (w_steps, _) = w_64.as_chunks::<16>();

        {
            let (wq, _) = w_steps[0].as_chunks::<4>();
            let w0 = load_f32x4(&wq[0]);
            let w1 = load_f32x4(&wq[1]);
            let w2 = load_f32x4(&wq[2]);
            let w3 = load_f32x4(&wq[3]);
            for f in 0..F {
                a[f][0] = fma_lane_f32::<0>(a[f][0], w0, xv[f]);
                a[f][1] = fma_lane_f32::<0>(a[f][1], w1, xv[f]);
                a[f][2] = fma_lane_f32::<0>(a[f][2], w2, xv[f]);
                a[f][3] = fma_lane_f32::<0>(a[f][3], w3, xv[f]);
            }
        }
        {
            let (wq, _) = w_steps[1].as_chunks::<4>();
            let w0 = load_f32x4(&wq[0]);
            let w1 = load_f32x4(&wq[1]);
            let w2 = load_f32x4(&wq[2]);
            let w3 = load_f32x4(&wq[3]);
            for f in 0..F {
                a[f][0] = fma_lane_f32::<1>(a[f][0], w0, xv[f]);
                a[f][1] = fma_lane_f32::<1>(a[f][1], w1, xv[f]);
                a[f][2] = fma_lane_f32::<1>(a[f][2], w2, xv[f]);
                a[f][3] = fma_lane_f32::<1>(a[f][3], w3, xv[f]);
            }
        }
        {
            let (wq, _) = w_steps[2].as_chunks::<4>();
            let w0 = load_f32x4(&wq[0]);
            let w1 = load_f32x4(&wq[1]);
            let w2 = load_f32x4(&wq[2]);
            let w3 = load_f32x4(&wq[3]);
            for f in 0..F {
                a[f][0] = fma_lane_f32::<2>(a[f][0], w0, xv[f]);
                a[f][1] = fma_lane_f32::<2>(a[f][1], w1, xv[f]);
                a[f][2] = fma_lane_f32::<2>(a[f][2], w2, xv[f]);
                a[f][3] = fma_lane_f32::<2>(a[f][3], w3, xv[f]);
            }
        }
        {
            let (wq, _) = w_steps[3].as_chunks::<4>();
            let w0 = load_f32x4(&wq[0]);
            let w1 = load_f32x4(&wq[1]);
            let w2 = load_f32x4(&wq[2]);
            let w3 = load_f32x4(&wq[3]);
            for f in 0..F {
                a[f][0] = fma_lane_f32::<3>(a[f][0], w0, xv[f]);
                a[f][1] = fma_lane_f32::<3>(a[f][1], w1, xv[f]);
                a[f][2] = fma_lane_f32::<3>(a[f][2], w2, xv[f]);
                a[f][3] = fma_lane_f32::<3>(a[f][3], w3, xv[f]);
            }
        }
    }

    let tail_base = quad_len * 4;
    let (w_tail_steps, _) = w_tail_ics.as_chunks::<16>();
    for (t_idx, w_16) in w_tail_steps.iter().enumerate() {
        let (wq, _) = w_16.as_chunks::<4>();
        let w0 = load_f32x4(&wq[0]);
        let w1 = load_f32x4(&wq[1]);
        let w2 = load_f32x4(&wq[2]);
        let w3 = load_f32x4(&wq[3]);
        for f in 0..F {
            let x_val = x_rows[f][tail_base + t_idx];
            a[f][0] = fma_n_f32(a[f][0], w0, x_val);
            a[f][1] = fma_n_f32(a[f][1], w1, x_val);
            a[f][2] = fma_n_f32(a[f][2], w2, x_val);
            a[f][3] = fma_n_f32(a[f][3], w3, x_val);
        }
    }

    let _keep_zero = zero_f32x4();
    for f in 0..F {
        let (quads, _) = acc[f].as_chunks_mut::<4>();
        store_f32x4(a[f][0], &mut quads[0]);
        store_f32x4(a[f][1], &mut quads[1]);
        store_f32x4(a[f][2], &mut quads[2]);
        store_f32x4(a[f][3], &mut quads[3]);
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub(crate) fn gemm_f32_fma_16oc<const F: usize>(
    w_ic_16: &[f32],
    x_rows: [&[f32]; F],
    in_c: usize,
    acc: &mut [[f32; 16]; F],
) {
    gemm_f32_fma_16oc_portable(w_ic_16, x_rows, in_c, acc);
}
