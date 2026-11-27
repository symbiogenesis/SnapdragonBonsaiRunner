//! Audited ARM64 NEON `vfmaq_laneq_f32` `16oc x 4f` (`1..=4f`), `8oc x 8f`, and L1-tiled
//! multi-frame `f32` outer-product convolution/GEMM micro-kernels with exact sequential-FMA
//! portable fallbacks.

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

#[cfg(any(not(target_arch = "aarch64"), test))]
pub(crate) fn gemm_f32_fma_8oc_8f_portable(
    w_ic_8: &[f32],
    x_rows: [&[f32]; 8],
    in_c: usize,
    acc: &mut [[f32; 8]; 8],
) {
    let Some(expected_w) = in_c.checked_mul(8) else {
        return;
    };
    if in_c == 0 || w_ic_8.len() < expected_w || x_rows.iter().any(|x| x.len() < in_c) {
        return;
    }
    let (w_steps, _) = w_ic_8[..expected_w].as_chunks::<8>();
    for (ic, w_8) in w_steps.iter().enumerate() {
        for f in 0..8 {
            let x_val = x_rows[f][ic];
            for (oc, &w_val) in w_8.iter().enumerate() {
                acc[f][oc] = x_val.mul_add(w_val, acc[f][oc]);
            }
        }
    }
}

#[cfg(any(not(target_arch = "aarch64"), test))]
pub(crate) fn gemm_f32_fma_16oc_frames_portable(
    w_ic_16: &[f32],
    x_rows: &[&[f32]],
    in_c: usize,
    acc_tile: &mut [[f32; 16]],
) {
    let frames = x_rows.len();
    let Some(expected_w) = in_c.checked_mul(16) else {
        return;
    };
    if frames == 0
        || acc_tile.len() < frames
        || in_c == 0
        || w_ic_16.len() < expected_w
        || x_rows.iter().any(|x| x.len() < in_c)
    {
        return;
    }
    let (w_steps, _) = w_ic_16[..expected_w].as_chunks::<16>();
    for (ic, w_16) in w_steps.iter().enumerate() {
        for (x_row, acc_row) in x_rows.iter().zip(acc_tile[..frames].iter_mut()) {
            let x_val = x_row[ic];
            for (oc, &w_val) in w_16.iter().enumerate() {
                acc_row[oc] = x_val.mul_add(w_val, acc_row[oc]);
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[expect(
    clippy::inline_always,
    reason = "Const lane index must inline into the outer-product FMA loop so vfmaq_laneq_f32 compiles to a single instruction."
)]
#[inline(always)]
fn step_8oc_8f_lane<const LANE: i32>(
    a: &mut [[float32x4_t; 2]; 8],
    w0: float32x4_t,
    w1: float32x4_t,
    xv: &[float32x4_t; 8],
) {
    for f in 0..8 {
        a[f][0] = fma_lane_f32::<LANE>(a[f][0], w0, xv[f]);
        a[f][1] = fma_lane_f32::<LANE>(a[f][1], w1, xv[f]);
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn gemm_f32_fma_8oc_8f_strided<
    const W_STRIDE: usize,
    const W_QUAD: usize,
    const W_Q_OFF: usize,
    const ACC_STRIDE: usize,
    const ACC_Q_OFF: usize,
>(
    w_ic: &[f32],
    x_rows: [&[f32]; 8],
    in_c: usize,
    acc: &mut [[f32; ACC_STRIDE]; 8],
) {
    const {
        assert!(W_STRIDE == 8 || W_STRIDE == 16);
        assert!(W_QUAD == W_STRIDE * 4);
        assert!(W_Q_OFF + 2 <= W_STRIDE / 4);
        assert!(ACC_Q_OFF + 2 <= ACC_STRIDE / 4);
    }
    let Some(expected_w) = in_c.checked_mul(W_STRIDE) else {
        return;
    };
    if in_c == 0 || w_ic.len() < expected_w || x_rows.iter().any(|x| x.len() < in_c) {
        return;
    }

    let mut a: [[float32x4_t; 2]; 8] = core::array::from_fn(|f| {
        let (quads, _) = acc[f].as_chunks::<4>();
        [
            load_f32x4(&quads[ACC_Q_OFF]),
            load_f32x4(&quads[ACC_Q_OFF + 1]),
        ]
    });

    let w_used = &w_ic[..expected_w];
    let (w_quad_ics, w_tail_ics) = w_used.as_chunks::<W_QUAD>();
    let quad_len = w_quad_ics.len();
    let x_quad_ics: [&[[f32; 4]]; 8] =
        core::array::from_fn(|f| x_rows[f][..in_c].as_chunks::<4>().0);

    for (idx, w_quad) in w_quad_ics.iter().enumerate() {
        if W_Q_OFF == 0 && idx + 2 < quad_len {
            let next_w = &w_quad_ics[idx + 2];
            prefetch_l1(next_w[0..].as_ptr().cast::<u8>());
            prefetch_l1(next_w[16..].as_ptr().cast::<u8>());
            if W_STRIDE == 16 {
                prefetch_l1(next_w[32..].as_ptr().cast::<u8>());
                prefetch_l1(next_w[48..].as_ptr().cast::<u8>());
            }
        }
        let xv: [float32x4_t; 8] = core::array::from_fn(|f| load_f32x4(&x_quad_ics[f][idx]));
        let (w_steps, _) = w_quad.as_chunks::<W_STRIDE>();

        {
            let (wq, _) = w_steps[0].as_chunks::<4>();
            step_8oc_8f_lane::<0>(
                &mut a,
                load_f32x4(&wq[W_Q_OFF]),
                load_f32x4(&wq[W_Q_OFF + 1]),
                &xv,
            );
        }
        {
            let (wq, _) = w_steps[1].as_chunks::<4>();
            step_8oc_8f_lane::<1>(
                &mut a,
                load_f32x4(&wq[W_Q_OFF]),
                load_f32x4(&wq[W_Q_OFF + 1]),
                &xv,
            );
        }
        {
            let (wq, _) = w_steps[2].as_chunks::<4>();
            step_8oc_8f_lane::<2>(
                &mut a,
                load_f32x4(&wq[W_Q_OFF]),
                load_f32x4(&wq[W_Q_OFF + 1]),
                &xv,
            );
        }
        {
            let (wq, _) = w_steps[3].as_chunks::<4>();
            step_8oc_8f_lane::<3>(
                &mut a,
                load_f32x4(&wq[W_Q_OFF]),
                load_f32x4(&wq[W_Q_OFF + 1]),
                &xv,
            );
        }
    }

    let tail_base = quad_len * 4;
    let (w_tail_steps, _) = w_tail_ics.as_chunks::<W_STRIDE>();
    for (t_idx, w_step) in w_tail_steps.iter().enumerate() {
        let (wq, _) = w_step.as_chunks::<4>();
        let w0 = load_f32x4(&wq[W_Q_OFF]);
        let w1 = load_f32x4(&wq[W_Q_OFF + 1]);
        for f in 0..8 {
            let x_val = x_rows[f][tail_base + t_idx];
            a[f][0] = fma_n_f32(a[f][0], w0, x_val);
            a[f][1] = fma_n_f32(a[f][1], w1, x_val);
        }
    }

    for f in 0..8 {
        let (quads, _) = acc[f].as_chunks_mut::<4>();
        store_f32x4(a[f][0], &mut quads[ACC_Q_OFF]);
        store_f32x4(a[f][1], &mut quads[ACC_Q_OFF + 1]);
    }
}

#[cfg(target_arch = "aarch64")]
pub(crate) fn gemm_f32_fma_8oc_8f(
    w_ic_8: &[f32],
    x_rows: [&[f32]; 8],
    in_c: usize,
    acc: &mut [[f32; 8]; 8],
) {
    gemm_f32_fma_8oc_8f_strided::<8, 32, 0, 8, 0>(w_ic_8, x_rows, in_c, acc);
}

#[cfg(not(target_arch = "aarch64"))]
pub(crate) fn gemm_f32_fma_8oc_8f(
    w_ic_8: &[f32],
    x_rows: [&[f32]; 8],
    in_c: usize,
    acc: &mut [[f32; 8]; 8],
) {
    gemm_f32_fma_8oc_8f_portable(w_ic_8, x_rows, in_c, acc);
}

#[cfg(target_arch = "aarch64")]
pub(crate) fn gemm_f32_fma_16oc_frames(
    w_ic_16: &[f32],
    x_rows: &[&[f32]],
    in_c: usize,
    acc_tile: &mut [[f32; 16]],
) {
    const IC_TILE: usize = 128;
    let frames = x_rows.len();
    let Some(expected_w) = in_c.checked_mul(16) else {
        return;
    };
    if frames == 0
        || acc_tile.len() < frames
        || in_c == 0
        || w_ic_16.len() < expected_w
        || x_rows.iter().any(|x| x.len() < in_c)
    {
        return;
    }

    let w_used = &w_ic_16[..expected_w];
    let acc_used = &mut acc_tile[..frames];

    for ic_base in (0..in_c).step_by(IC_TILE) {
        let ic_end = (ic_base + IC_TILE).min(in_c);
        let ic_len = ic_end - ic_base;
        let w_sub = &w_used[ic_base * 16..ic_end * 16];

        let (acc_chunks8, acc_rem8) = acc_used.as_chunks_mut::<8>();
        for (chunk_idx, acc8) in acc_chunks8.iter_mut().enumerate() {
            let f = chunk_idx * 8;
            let xr: [&[f32]; 8] = core::array::from_fn(|df| &x_rows[f + df][ic_base..ic_end]);
            gemm_f32_fma_8oc_8f_strided::<16, 64, 0, 16, 0>(w_sub, xr, ic_len, acc8);
            gemm_f32_fma_8oc_8f_strided::<16, 64, 2, 16, 2>(w_sub, xr, ic_len, acc8);
        }

        let f_rem8 = acc_chunks8.len() * 8;
        let (acc_chunks4, acc_rem4) = acc_rem8.as_chunks_mut::<4>();
        if let Some(acc4) = acc_chunks4.first_mut() {
            let xr: [&[f32]; 4] = core::array::from_fn(|df| &x_rows[f_rem8 + df][ic_base..ic_end]);
            gemm_f32_fma_16oc::<4>(w_sub, xr, ic_len, acc4);
        }

        let f_rem4 = f_rem8 + acc_chunks4.len() * 4;
        match acc_rem4.len() {
            3 => {
                if let Some(acc3) = acc_rem4.first_chunk_mut::<3>() {
                    let xr: [&[f32]; 3] =
                        core::array::from_fn(|df| &x_rows[f_rem4 + df][ic_base..ic_end]);
                    gemm_f32_fma_16oc::<3>(w_sub, xr, ic_len, acc3);
                }
            }
            2 => {
                if let Some(acc2) = acc_rem4.first_chunk_mut::<2>() {
                    let xr: [&[f32]; 2] =
                        core::array::from_fn(|df| &x_rows[f_rem4 + df][ic_base..ic_end]);
                    gemm_f32_fma_16oc::<2>(w_sub, xr, ic_len, acc2);
                }
            }
            1 => {
                if let Some(acc1) = acc_rem4.first_chunk_mut::<1>() {
                    let xr = [&x_rows[f_rem4][ic_base..ic_end]];
                    gemm_f32_fma_16oc::<1>(w_sub, xr, ic_len, acc1);
                }
            }
            _ => {}
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub(crate) fn gemm_f32_fma_16oc_frames(
    w_ic_16: &[f32],
    x_rows: &[&[f32]],
    in_c: usize,
    acc_tile: &mut [[f32; 16]],
) {
    gemm_f32_fma_16oc_frames_portable(w_ic_16, x_rows, in_c, acc_tile);
}

#[cfg(any(not(target_arch = "aarch64"), test))]
pub(crate) fn gemm_f32_fma_16oc_strided_portable(
    w_ic_16: &[f32],
    x_base: &[f32],
    x_frame_stride: usize,
    in_c: usize,
    frames: usize,
    acc_tile: &mut [[f32; 16]],
) {
    let Some(expected_w) = in_c.checked_mul(16) else {
        return;
    };
    let Some(last_off) = frames
        .checked_sub(1)
        .and_then(|last| last.checked_mul(x_frame_stride))
    else {
        return;
    };
    let Some(needed_x) = last_off.checked_add(in_c) else {
        return;
    };
    if in_c == 0 || acc_tile.len() < frames || w_ic_16.len() < expected_w || x_base.len() < needed_x
    {
        return;
    }
    let (w_steps, _) = w_ic_16[..expected_w].as_chunks::<16>();
    for (f, acc_row) in acc_tile[..frames].iter_mut().enumerate() {
        let x_off = f * x_frame_stride;
        let x_row = &x_base[x_off..x_off + in_c];
        for (ic, w_16) in w_steps.iter().enumerate() {
            let x_val = x_row[ic];
            for (oc, &w_val) in w_16.iter().enumerate() {
                acc_row[oc] = x_val.mul_add(w_val, acc_row[oc]);
            }
        }
    }
}

#[cfg(any(not(target_arch = "aarch64"), test))]
#[expect(
    clippy::too_many_arguments,
    reason = "Strided 1D convolution micro-kernel mirrors public safe entry point signature."
)]
pub(crate) fn conv1d_f32_fma_16oc_strided_portable(
    w_blk: &[f32],
    ext: &[f32],
    in_c: usize,
    kernel: usize,
    stride: usize,
    dilation: usize,
    frames: usize,
    acc_tile: &mut [[f32; 16]],
) {
    let Some(k_in_c) = kernel.checked_mul(in_c) else {
        return;
    };
    let Some(expected_w) = k_in_c.checked_mul(16) else {
        return;
    };
    let Some(max_f) = frames.checked_sub(1).and_then(|f| f.checked_mul(stride)) else {
        return;
    };
    let Some(max_k) = kernel.checked_sub(1).and_then(|k| k.checked_mul(dilation)) else {
        return;
    };
    let Some(needed_ext) = max_f
        .checked_add(max_k)
        .and_then(|p| p.checked_add(1))
        .and_then(|p| p.checked_mul(in_c))
    else {
        return;
    };
    if in_c == 0 || acc_tile.len() < frames || w_blk.len() < expected_w || ext.len() < needed_ext {
        return;
    }
    let w_step_stride = in_c * 16;
    for (f, acc_row) in acc_tile[..frames].iter_mut().enumerate() {
        let f_off = f * stride;
        for k in 0..kernel {
            let w_k = &w_blk[k * w_step_stride..(k + 1) * w_step_stride];
            let x_off = (f_off + k * dilation) * in_c;
            let x_row = &ext[x_off..x_off + in_c];
            let (w_steps, _) = w_k.as_chunks::<16>();
            for (ic, w_16) in w_steps.iter().enumerate() {
                let x_val = x_row[ic];
                for (oc, &w_val) in w_16.iter().enumerate() {
                    acc_row[oc] = x_val.mul_add(w_val, acc_row[oc]);
                }
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[expect(
    clippy::inline_always,
    reason = "Const lane index and frame count must inline so 16oc x F accumulators stay in NEON registers."
)]
#[inline(always)]
fn step_16oc_lane<const F: usize, const LANE: i32>(
    a: &mut [[float32x4_t; 4]; F],
    w_step: &[f32; 16],
    xv: &[float32x4_t; F],
) {
    let (wq, _) = w_step.as_chunks::<4>();
    let w0 = load_f32x4(&wq[0]);
    let w1 = load_f32x4(&wq[1]);
    for f in 0..F {
        a[f][0] = fma_lane_f32::<LANE>(a[f][0], w0, xv[f]);
        a[f][1] = fma_lane_f32::<LANE>(a[f][1], w1, xv[f]);
    }
    let w2 = load_f32x4(&wq[2]);
    let w3 = load_f32x4(&wq[3]);
    for f in 0..F {
        a[f][2] = fma_lane_f32::<LANE>(a[f][2], w2, xv[f]);
        a[f][3] = fma_lane_f32::<LANE>(a[f][3], w3, xv[f]);
    }
}

#[cfg(target_arch = "aarch64")]
#[expect(
    clippy::inline_always,
    reason = "Single-pass outer-product FMA accumulation must inline into caller to keep accumulators in NEON registers."
)]
#[inline(always)]
fn accum_16oc_strided<const F: usize>(
    a: &mut [[float32x4_t; 4]; F],
    w_ic_16: &[f32],
    x_base: &[f32],
    x_frame_stride: usize,
    in_c: usize,
) {
    let expected_w = in_c * 16;
    let w_used = &w_ic_16[..expected_w];
    let (w_quad_ics, w_tail_ics) = w_used.as_chunks::<64>();
    let quad_len = w_quad_ics.len();
    let x_quad_ics: [&[[f32; 4]]; F] = core::array::from_fn(|f| {
        let start = f * x_frame_stride;
        x_base[start..start + in_c].as_chunks::<4>().0
    });

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
        step_16oc_lane::<F, 0>(a, &w_steps[0], &xv);
        step_16oc_lane::<F, 1>(a, &w_steps[1], &xv);
        step_16oc_lane::<F, 2>(a, &w_steps[2], &xv);
        step_16oc_lane::<F, 3>(a, &w_steps[3], &xv);
    }

    let tail_base = quad_len * 4;
    let (w_tail_steps, _) = w_tail_ics.as_chunks::<16>();
    for (t_idx, w_16) in w_tail_steps.iter().enumerate() {
        let (wq, _) = w_16.as_chunks::<4>();
        let w0 = load_f32x4(&wq[0]);
        let w1 = load_f32x4(&wq[1]);
        for f in 0..F {
            let x_val = x_base[f * x_frame_stride + tail_base + t_idx];
            a[f][0] = fma_n_f32(a[f][0], w0, x_val);
            a[f][1] = fma_n_f32(a[f][1], w1, x_val);
        }
        let w2 = load_f32x4(&wq[2]);
        let w3 = load_f32x4(&wq[3]);
        for f in 0..F {
            let x_val = x_base[f * x_frame_stride + tail_base + t_idx];
            a[f][2] = fma_n_f32(a[f][2], w2, x_val);
            a[f][3] = fma_n_f32(a[f][3], w3, x_val);
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn load_acc_16oc<const F: usize>(acc: &[[f32; 16]; F]) -> [[float32x4_t; 4]; F] {
    core::array::from_fn(|f| {
        let (quads, _) = acc[f].as_chunks::<4>();
        [
            load_f32x4(&quads[0]),
            load_f32x4(&quads[1]),
            load_f32x4(&quads[2]),
            load_f32x4(&quads[3]),
        ]
    })
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn store_acc_16oc<const F: usize>(a: &[[float32x4_t; 4]; F], acc: &mut [[f32; 16]; F]) {
    for f in 0..F {
        let (quads, _) = acc[f].as_chunks_mut::<4>();
        store_f32x4(a[f][0], &mut quads[0]);
        store_f32x4(a[f][1], &mut quads[1]);
        store_f32x4(a[f][2], &mut quads[2]);
        store_f32x4(a[f][3], &mut quads[3]);
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn gemm_f32_fma_16oc_strided_chunk<const F: usize>(
    w_used: &[f32],
    x_base: &[f32],
    x_frame_stride: usize,
    in_c: usize,
    acc: &mut [[f32; 16]; F],
) {
    let mut a = load_acc_16oc(acc);
    accum_16oc_strided::<F>(&mut a, w_used, x_base, x_frame_stride, in_c);
    store_acc_16oc(&a, acc);
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn conv1d_f32_fma_16oc_strided_chunk<const F: usize>(
    w_used: &[f32],
    ext: &[f32],
    in_c: usize,
    kernel: usize,
    stride: usize,
    dilation: usize,
    acc: &mut [[f32; 16]; F],
) {
    let mut a = load_acc_16oc(acc);
    let w_k_stride = in_c * 16;
    let x_frame_stride = stride * in_c;
    for k in 0..kernel {
        let w_k = &w_used[k * w_k_stride..(k + 1) * w_k_stride];
        let ext_k = &ext[k * dilation * in_c..];
        accum_16oc_strided::<F>(&mut a, w_k, ext_k, x_frame_stride, in_c);
    }
    store_acc_16oc(&a, acc);
}

#[cfg(target_arch = "aarch64")]
pub(crate) fn gemm_f32_fma_16oc_strided(
    w_ic_16: &[f32],
    x_base: &[f32],
    x_frame_stride: usize,
    in_c: usize,
    frames: usize,
    acc_tile: &mut [[f32; 16]],
) {
    let Some(expected_w) = in_c.checked_mul(16) else {
        return;
    };
    let Some(last_off) = frames
        .checked_sub(1)
        .and_then(|last| last.checked_mul(x_frame_stride))
    else {
        return;
    };
    let Some(needed_x) = last_off.checked_add(in_c) else {
        return;
    };
    if in_c == 0 || acc_tile.len() < frames || w_ic_16.len() < expected_w || x_base.len() < needed_x
    {
        return;
    }

    let w_used = &w_ic_16[..expected_w];
    let acc_used = &mut acc_tile[..frames];
    let (acc_chunks6, acc_rem6) = acc_used.as_chunks_mut::<6>();
    for (chunk_idx, acc6) in acc_chunks6.iter_mut().enumerate() {
        let f_off = chunk_idx * 6 * x_frame_stride;
        gemm_f32_fma_16oc_strided_chunk::<6>(w_used, &x_base[f_off..], x_frame_stride, in_c, acc6);
    }

    let f_rem6 = acc_chunks6.len() * 6;
    let (acc_chunks4, acc_rem4) = acc_rem6.as_chunks_mut::<4>();
    if let Some(acc4) = acc_chunks4.first_mut() {
        gemm_f32_fma_16oc_strided_chunk::<4>(
            w_used,
            &x_base[f_rem6 * x_frame_stride..],
            x_frame_stride,
            in_c,
            acc4,
        );
    }

    let f_rem4 = f_rem6 + acc_chunks4.len() * 4;
    match acc_rem4.len() {
        3 => {
            if let Some(acc3) = acc_rem4.first_chunk_mut::<3>() {
                gemm_f32_fma_16oc_strided_chunk::<3>(
                    w_used,
                    &x_base[f_rem4 * x_frame_stride..],
                    x_frame_stride,
                    in_c,
                    acc3,
                );
            }
        }
        2 => {
            if let Some(acc2) = acc_rem4.first_chunk_mut::<2>() {
                gemm_f32_fma_16oc_strided_chunk::<2>(
                    w_used,
                    &x_base[f_rem4 * x_frame_stride..],
                    x_frame_stride,
                    in_c,
                    acc2,
                );
            }
        }
        1 => {
            if let Some(acc1) = acc_rem4.first_chunk_mut::<1>() {
                gemm_f32_fma_16oc_strided_chunk::<1>(
                    w_used,
                    &x_base[f_rem4 * x_frame_stride..],
                    x_frame_stride,
                    in_c,
                    acc1,
                );
            }
        }
        _ => {}
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub(crate) fn gemm_f32_fma_16oc_strided(
    w_ic_16: &[f32],
    x_base: &[f32],
    x_frame_stride: usize,
    in_c: usize,
    frames: usize,
    acc_tile: &mut [[f32; 16]],
) {
    gemm_f32_fma_16oc_strided_portable(w_ic_16, x_base, x_frame_stride, in_c, frames, acc_tile);
}

#[cfg(target_arch = "aarch64")]
#[expect(
    clippy::too_many_arguments,
    reason = "Strided 1D convolution micro-kernel mirrors public safe entry point signature."
)]
pub(crate) fn conv1d_f32_fma_16oc_strided(
    w_blk: &[f32],
    ext: &[f32],
    in_c: usize,
    kernel: usize,
    stride: usize,
    dilation: usize,
    frames: usize,
    acc_tile: &mut [[f32; 16]],
) {
    let Some(k_in_c) = kernel.checked_mul(in_c) else {
        return;
    };
    let Some(expected_w) = k_in_c.checked_mul(16) else {
        return;
    };
    let Some(max_f) = frames.checked_sub(1).and_then(|f| f.checked_mul(stride)) else {
        return;
    };
    let Some(max_k) = kernel.checked_sub(1).and_then(|k| k.checked_mul(dilation)) else {
        return;
    };
    let Some(needed_ext) = max_f
        .checked_add(max_k)
        .and_then(|p| p.checked_add(1))
        .and_then(|p| p.checked_mul(in_c))
    else {
        return;
    };
    if in_c == 0 || acc_tile.len() < frames || w_blk.len() < expected_w || ext.len() < needed_ext {
        return;
    }

    let w_used = &w_blk[..expected_w];
    let acc_used = &mut acc_tile[..frames];
    let frame_step = stride * in_c;
    let (acc_chunks6, acc_rem6) = acc_used.as_chunks_mut::<6>();
    for (chunk_idx, acc6) in acc_chunks6.iter_mut().enumerate() {
        let ext_off = chunk_idx * 6 * frame_step;
        conv1d_f32_fma_16oc_strided_chunk::<6>(
            w_used,
            &ext[ext_off..],
            in_c,
            kernel,
            stride,
            dilation,
            acc6,
        );
    }

    let f_rem6 = acc_chunks6.len() * 6;
    let (acc_chunks4, acc_rem4) = acc_rem6.as_chunks_mut::<4>();
    if let Some(acc4) = acc_chunks4.first_mut() {
        conv1d_f32_fma_16oc_strided_chunk::<4>(
            w_used,
            &ext[f_rem6 * frame_step..],
            in_c,
            kernel,
            stride,
            dilation,
            acc4,
        );
    }

    let f_rem4 = f_rem6 + acc_chunks4.len() * 4;
    match acc_rem4.len() {
        3 => {
            if let Some(acc3) = acc_rem4.first_chunk_mut::<3>() {
                conv1d_f32_fma_16oc_strided_chunk::<3>(
                    w_used,
                    &ext[f_rem4 * frame_step..],
                    in_c,
                    kernel,
                    stride,
                    dilation,
                    acc3,
                );
            }
        }
        2 => {
            if let Some(acc2) = acc_rem4.first_chunk_mut::<2>() {
                conv1d_f32_fma_16oc_strided_chunk::<2>(
                    w_used,
                    &ext[f_rem4 * frame_step..],
                    in_c,
                    kernel,
                    stride,
                    dilation,
                    acc2,
                );
            }
        }
        1 => {
            if let Some(acc1) = acc_rem4.first_chunk_mut::<1>() {
                conv1d_f32_fma_16oc_strided_chunk::<1>(
                    w_used,
                    &ext[f_rem4 * frame_step..],
                    in_c,
                    kernel,
                    stride,
                    dilation,
                    acc1,
                );
            }
        }
        _ => {}
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[expect(
    clippy::too_many_arguments,
    reason = "Strided 1D convolution micro-kernel mirrors public safe entry point signature."
)]
pub(crate) fn conv1d_f32_fma_16oc_strided(
    w_blk: &[f32],
    ext: &[f32],
    in_c: usize,
    kernel: usize,
    stride: usize,
    dilation: usize,
    frames: usize,
    acc_tile: &mut [[f32; 16]],
) {
    conv1d_f32_fma_16oc_strided_portable(
        w_blk, ext, in_c, kernel, stride, dilation, frames, acc_tile,
    );
}
