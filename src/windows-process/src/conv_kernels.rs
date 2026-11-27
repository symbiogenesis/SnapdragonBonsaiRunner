//! Safe public boundary for ARM64 NEON `16oc x 1..=4f`, `8oc x 8f`, and L1-tiled
//! multi-frame outer-product `f32` convolution/GEMM kernels with exact `0-ULP` portable
//! scalar FMA fallbacks.

/// Accumulates `acc[f][oc] += sum_{ic=0..in_c} x_rows[f][ic] * w_ic_16[ic * 16 + oc]`
/// for `F` frames (`1..=4`) and 16 contiguous output channels in sequential `ic` FMA order.
///
/// No-ops if `in_c == 0` or any slice is shorter than required.
pub fn gemm_f32_fma_16oc<const F: usize>(
    w_ic_16: &[f32],
    x_rows: [&[f32]; F],
    in_c: usize,
    acc: &mut [[f32; 16]; F],
) {
    crate::native::conv_f32::gemm_f32_fma_16oc(w_ic_16, x_rows, in_c, acc);
}

/// Accumulates a 4-frame by 16-output-channel outer-product tile in sequential `ic` FMA order.
///
/// No-ops if `in_c == 0` or any slice is shorter than required.
pub fn gemm_f32_fma_16oc_4f(
    w_ic_16: &[f32],
    x_rows: [&[f32]; 4],
    in_c: usize,
    acc: &mut [[f32; 16]; 4],
) {
    crate::native::conv_f32::gemm_f32_fma_16oc::<4>(w_ic_16, x_rows, in_c, acc);
}

/// Accumulates a 3-frame by 16-output-channel outer-product tile in sequential `ic` FMA order.
///
/// No-ops if `in_c == 0` or any slice is shorter than required.
pub fn gemm_f32_fma_16oc_3f(
    w_ic_16: &[f32],
    x_rows: [&[f32]; 3],
    in_c: usize,
    acc: &mut [[f32; 16]; 3],
) {
    crate::native::conv_f32::gemm_f32_fma_16oc::<3>(w_ic_16, x_rows, in_c, acc);
}

/// Accumulates a 2-frame by 16-output-channel outer-product tile in sequential `ic` FMA order.
///
/// No-ops if `in_c == 0` or any slice is shorter than required.
pub fn gemm_f32_fma_16oc_2f(
    w_ic_16: &[f32],
    x_rows: [&[f32]; 2],
    in_c: usize,
    acc: &mut [[f32; 16]; 2],
) {
    crate::native::conv_f32::gemm_f32_fma_16oc::<2>(w_ic_16, x_rows, in_c, acc);
}

/// Accumulates a 1-frame by 16-output-channel outer-product tile in sequential `ic` FMA order.
///
/// No-ops if `in_c == 0` or any slice is shorter than required.
pub fn gemm_f32_fma_16oc_1f(
    w_ic_16: &[f32],
    x_rows: [&[f32]; 1],
    in_c: usize,
    acc: &mut [[f32; 16]; 1],
) {
    crate::native::conv_f32::gemm_f32_fma_16oc::<1>(w_ic_16, x_rows, in_c, acc);
}

/// Accumulates an 8-frame by 8-output-channel outer-product tile in sequential `ic` FMA order.
///
/// No-ops if `in_c == 0` or any slice is shorter than required.
pub fn gemm_f32_fma_8oc_8f(
    w_ic_8: &[f32],
    x_rows: [&[f32]; 8],
    in_c: usize,
    acc: &mut [[f32; 8]; 8],
) {
    crate::native::conv_f32::gemm_f32_fma_8oc_8f(w_ic_8, x_rows, in_c, acc);
}

/// Streams a packed `[in_c, 16]` weight tile across an arbitrary-length tile of frames
/// (`acc_tile[..x_rows.len()]`) using L1D-resident K-blocking and `8oc x 8f` / `16oc x 1..=4f`
/// micro-kernels in exact sequential `ic` FMA order.
///
/// No-ops if `in_c == 0`, `x_rows` is empty, or any slice is shorter than required.
pub fn gemm_f32_fma_16oc_frames(
    w_ic_16: &[f32],
    x_rows: &[&[f32]],
    in_c: usize,
    acc_tile: &mut [[f32; 16]],
) {
    crate::native::conv_f32::gemm_f32_fma_16oc_frames(w_ic_16, x_rows, in_c, acc_tile);
}

/// Streams a packed `[in_c, 16]` weight tile across `frames` rows starting at `x_base`,
/// spaced by `x_frame_stride` elements per frame, into `acc_tile[..frames]` using single-pass
/// `16oc x 6f` and `16oc x {4,3,2,1}f` NEON FMA micro-kernels in exact sequential `ic` order.
///
/// No-ops if `frames == 0`, `in_c == 0`, or any slice is shorter than required.
pub fn gemm_f32_fma_16oc_strided(
    w_ic_16: &[f32],
    x_base: &[f32],
    x_frame_stride: usize,
    in_c: usize,
    frames: usize,
    acc_tile: &mut [[f32; 16]],
) {
    crate::native::conv_f32::gemm_f32_fma_16oc_strided(
        w_ic_16,
        x_base,
        x_frame_stride,
        in_c,
        frames,
        acc_tile,
    );
}

/// Accumulates a strided/dilated 1D convolution tile (`[kernel, in_c, 16]` packed weights) over
/// `frames` output frames starting at `ext`, keeping accumulators in NEON registers across
/// `k in 0..kernel` and `ic in 0..in_c` in exact sequential `(k, ic)` FMA order.
///
/// No-ops if `frames == 0`, `in_c == 0`, `kernel == 0`, or any slice is shorter than required.
#[expect(
    clippy::too_many_arguments,
    reason = "Explicit convolution geometry parameters avoid heap wrappers on the hot path."
)]
pub fn conv1d_f32_fma_16oc_strided(
    w_blk: &[f32],
    ext: &[f32],
    in_c: usize,
    kernel: usize,
    stride: usize,
    dilation: usize,
    frames: usize,
    acc_tile: &mut [[f32; 16]],
) {
    crate::native::conv_f32::conv1d_f32_fma_16oc_strided(
        w_blk, ext, in_c, kernel, stride, dilation, frames, acc_tile,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native::conv_f32::{
        conv1d_f32_fma_16oc_strided_portable, gemm_f32_fma_8oc_8f_portable,
        gemm_f32_fma_16oc_frames_portable, gemm_f32_fma_16oc_portable,
        gemm_f32_fma_16oc_strided_portable,
    };

    fn make_edge_f32(len: usize, seed: usize) -> Vec<f32> {
        const EDGE: [u32; 8] = [
            0x0000_0000, // +0.0
            0x8000_0000, // -0.0
            0x3eaa_aaab, // 1/3
            0xbeaa_aaab, // -1/3
            0x3f80_0001, // 1 + eps
            0xbf80_0003, // -(1 + 3*eps)
            0x3b80_0000, // 2^-8
            0xc049_0fdb, // -pi
        ];
        (0..len)
            .map(|i| {
                if (seed + i) % 7 == 0 {
                    f32::from_bits(EDGE[(seed * 5 + i * 3) % EDGE.len()])
                } else {
                    let v = i16::try_from((seed * 19 + i * 31) % 251).unwrap_or(0) - 125;
                    f32::from(v) * 0.03125
                }
            })
            .collect()
    }

    fn verify_f<const F: usize>(in_c: usize, seed: usize) {
        let w = make_edge_f32(in_c * 16, seed + 1);
        let xs: [Vec<f32>; F] = std::array::from_fn(|f| make_edge_f32(in_c, seed + 10 + f));
        let x_refs: [&[f32]; F] = std::array::from_fn(|f| xs[f].as_slice());
        let init: [[f32; 16]; F] = std::array::from_fn(|f| {
            let v = make_edge_f32(16, seed + 100 + f);
            std::array::from_fn(|oc| v[oc])
        });
        let mut acc_nat = init;
        let mut acc_prt = init;
        gemm_f32_fma_16oc::<F>(&w, x_refs, in_c, &mut acc_nat);
        gemm_f32_fma_16oc_portable::<F>(&w, x_refs, in_c, &mut acc_prt);
        for f in 0..F {
            assert_eq!(
                acc_nat[f].map(f32::to_bits),
                acc_prt[f].map(f32::to_bits),
                "0-ULP parity mismatch F={F} in_c={in_c} f={f}"
            );
        }
    }

    #[test]
    fn gemm_f32_fma_16oc_matches_portable_scalar_fma_bit_exact() {
        for (idx, in_c) in [1_usize, 2, 3, 4, 5, 7, 8, 15, 16, 17, 64, 129, 256]
            .into_iter()
            .enumerate()
        {
            verify_f::<1>(in_c, idx * 7 + 1);
            verify_f::<2>(in_c, idx * 7 + 2);
            verify_f::<3>(in_c, idx * 7 + 3);
            verify_f::<4>(in_c, idx * 7 + 4);
        }

        // Explicitly exercise named 4f/3f/2f/1f wrappers and invalid/zero bounds no-ops.
        let w = make_edge_f32(19 * 16, 42);
        let x0 = make_edge_f32(19, 1);
        let x1 = make_edge_f32(19, 2);
        let x2 = make_edge_f32(19, 3);
        let x3 = make_edge_f32(19, 4);

        let mut a4_nat = [[0.25_f32; 16]; 4];
        let mut a4_prt = a4_nat;
        gemm_f32_fma_16oc_4f(&w, [&x0, &x1, &x2, &x3], 19, &mut a4_nat);
        gemm_f32_fma_16oc_portable(&w, [&x0, &x1, &x2, &x3], 19, &mut a4_prt);
        assert_eq!(
            a4_nat.map(|r| r.map(f32::to_bits)),
            a4_prt.map(|r| r.map(f32::to_bits))
        );

        let mut a3_nat = [[0.25_f32; 16]; 3];
        let mut a3_prt = a3_nat;
        gemm_f32_fma_16oc_3f(&w, [&x0, &x1, &x2], 19, &mut a3_nat);
        gemm_f32_fma_16oc_portable(&w, [&x0, &x1, &x2], 19, &mut a3_prt);
        assert_eq!(
            a3_nat.map(|r| r.map(f32::to_bits)),
            a3_prt.map(|r| r.map(f32::to_bits))
        );

        let mut a2_nat = [[0.25_f32; 16]; 2];
        let mut a2_prt = a2_nat;
        gemm_f32_fma_16oc_2f(&w, [&x0, &x1], 19, &mut a2_nat);
        gemm_f32_fma_16oc_portable(&w, [&x0, &x1], 19, &mut a2_prt);
        assert_eq!(
            a2_nat.map(|r| r.map(f32::to_bits)),
            a2_prt.map(|r| r.map(f32::to_bits))
        );

        let mut a1_nat = [[0.25_f32; 16]; 1];
        let mut a1_prt = a1_nat;
        gemm_f32_fma_16oc_1f(&w, [&x0], 19, &mut a1_nat);
        gemm_f32_fma_16oc_portable(&w, [&x0], 19, &mut a1_prt);
        assert_eq!(
            a1_nat.map(|r| r.map(f32::to_bits)),
            a1_prt.map(|r| r.map(f32::to_bits))
        );

        // Zero / short slice bounds no-op safely without panicking.
        let mut guard = [[1.5_f32; 16]; 4];
        gemm_f32_fma_16oc_4f(&w, [&x0, &x1, &x2, &x3], 0, &mut guard);
        gemm_f32_fma_16oc_4f(&w[..w.len() - 1], [&x0, &x1, &x2, &x3], 19, &mut guard);
        gemm_f32_fma_16oc_4f(&w, [&x0, &x1[..18], &x2, &x3], 19, &mut guard);
        assert_eq!(
            guard.map(|r| r.map(f32::to_bits)),
            [[1.5_f32.to_bits(); 16]; 4]
        );
    }

    #[test]
    fn gemm_f32_fma_8oc_8f_matches_portable_scalar_fma_bit_exact() {
        for (idx, in_c) in [1_usize, 2, 3, 4, 5, 7, 15, 16, 17, 128, 129]
            .into_iter()
            .enumerate()
        {
            let seed = idx * 13 + 3;
            let w = make_edge_f32(in_c * 8, seed + 1);
            let xs: [Vec<f32>; 8] = std::array::from_fn(|f| make_edge_f32(in_c, seed + 10 + f));
            let x_refs: [&[f32]; 8] = std::array::from_fn(|f| xs[f].as_slice());
            let init: [[f32; 8]; 8] = std::array::from_fn(|f| {
                let v = make_edge_f32(8, seed + 100 + f);
                std::array::from_fn(|oc| v[oc])
            });
            let mut acc_nat = init;
            let mut acc_prt = init;
            gemm_f32_fma_8oc_8f(&w, x_refs, in_c, &mut acc_nat);
            gemm_f32_fma_8oc_8f_portable(&w, x_refs, in_c, &mut acc_prt);
            for f in 0..8 {
                assert_eq!(
                    acc_nat[f].map(f32::to_bits),
                    acc_prt[f].map(f32::to_bits),
                    "8oc_8f 0-ULP parity mismatch in_c={in_c} f={f}"
                );
            }
        }

        // Zero / short slice bounds no-op safely.
        let w = make_edge_f32(19 * 8, 77);
        let xs: [Vec<f32>; 8] = std::array::from_fn(|f| make_edge_f32(19, 10 + f));
        let mut guard = [[-0.0_f32; 8]; 8];
        gemm_f32_fma_8oc_8f(&w, std::array::from_fn(|f| xs[f].as_slice()), 0, &mut guard);
        gemm_f32_fma_8oc_8f(
            &w[..w.len() - 1],
            std::array::from_fn(|f| xs[f].as_slice()),
            19,
            &mut guard,
        );
        gemm_f32_fma_8oc_8f(
            &w,
            [
                &xs[0],
                &xs[1],
                &xs[2],
                &xs[3],
                &xs[4],
                &xs[5],
                &xs[6],
                &xs[7][..18],
            ],
            19,
            &mut guard,
        );
        assert_eq!(
            guard.map(|r| r.map(f32::to_bits)),
            [[(-0.0_f32).to_bits(); 8]; 8]
        );
    }

    #[test]
    fn gemm_f32_fma_16oc_frames_matches_portable_scalar_fma_bit_exact() {
        const IN_CS: [usize; 11] = [1, 2, 3, 4, 5, 7, 15, 16, 17, 128, 129];
        const FRAMES: [usize; 12] = [1, 2, 3, 4, 5, 7, 8, 9, 16, 31, 32, 33];

        for (c_idx, &in_c) in IN_CS.iter().enumerate() {
            for (f_idx, &frames) in FRAMES.iter().enumerate() {
                let seed = c_idx * 41 + f_idx * 17 + 5;
                let w = make_edge_f32(in_c * 16, seed + 1);
                let xs: Vec<Vec<f32>> = (0..frames)
                    .map(|f| make_edge_f32(in_c, seed + 20 + f))
                    .collect();
                let x_refs: Vec<&[f32]> = xs.iter().map(Vec::as_slice).collect();
                let init: Vec<[f32; 16]> = (0..frames)
                    .map(|f| {
                        let v = make_edge_f32(16, seed + 200 + f);
                        std::array::from_fn(|oc| v[oc])
                    })
                    .collect();
                let mut acc_nat = init.clone();
                let mut acc_prt = init;
                gemm_f32_fma_16oc_frames(&w, &x_refs, in_c, &mut acc_nat);
                gemm_f32_fma_16oc_frames_portable(&w, &x_refs, in_c, &mut acc_prt);
                for f in 0..frames {
                    assert_eq!(
                        acc_nat[f].map(f32::to_bits),
                        acc_prt[f].map(f32::to_bits),
                        "16oc_frames 0-ULP parity mismatch in_c={in_c} frames={frames} f={f}"
                    );
                }
            }
        }

        // Zero / short / empty bounds no-op safely.
        let w = make_edge_f32(17 * 16, 99);
        let x0 = make_edge_f32(17, 1);
        let x1 = make_edge_f32(17, 2);
        let mut guard = [[2.5_f32; 16]; 2];
        gemm_f32_fma_16oc_frames(&w, &[], 17, &mut guard);
        gemm_f32_fma_16oc_frames(&w, &[&x0, &x1], 0, &mut guard);
        gemm_f32_fma_16oc_frames(&w[..w.len() - 1], &[&x0, &x1], 17, &mut guard);
        gemm_f32_fma_16oc_frames(&w, &[&x0, &x1[..16]], 17, &mut guard);
        gemm_f32_fma_16oc_frames(&w, &[&x0, &x1], 17, &mut guard[..1]);
        assert_eq!(
            guard.map(|r| r.map(f32::to_bits)),
            [[2.5_f32.to_bits(); 16]; 2]
        );
    }

    #[test]
    fn gemm_f32_fma_16oc_strided_matches_portable_scalar_fma_bit_exact() {
        const IN_CS: [usize; 9] = [1, 3, 4, 7, 16, 128, 129, 256, 512];

        for (c_idx, &in_c) in IN_CS.iter().enumerate() {
            for frames in 1_usize..=67 {
                // Exercise packed (`stride == in_c`) and padded (`stride > in_c`) layouts.
                let x_frame_stride = if (frames + c_idx) % 2 == 0 {
                    in_c
                } else {
                    in_c + 5
                };
                let seed = c_idx * 101 + frames * 17 + 9;
                let w = make_edge_f32(in_c * 16, seed + 1);
                let x_len = (frames - 1) * x_frame_stride + in_c;
                let x_base = make_edge_f32(x_len, seed + 23);
                let init: Vec<[f32; 16]> = (0..frames)
                    .map(|f| {
                        let v = make_edge_f32(16, seed + 300 + f);
                        std::array::from_fn(|oc| v[oc])
                    })
                    .collect();
                let mut acc_nat = init.clone();
                let mut acc_prt = init;
                gemm_f32_fma_16oc_strided(&w, &x_base, x_frame_stride, in_c, frames, &mut acc_nat);
                gemm_f32_fma_16oc_strided_portable(
                    &w,
                    &x_base,
                    x_frame_stride,
                    in_c,
                    frames,
                    &mut acc_prt,
                );
                for f in 0..frames {
                    assert_eq!(
                        acc_nat[f].map(f32::to_bits),
                        acc_prt[f].map(f32::to_bits),
                        "16oc_strided 0-ULP mismatch in_c={in_c} stride={x_frame_stride} frames={frames} f={f}"
                    );
                }
            }
        }

        // Zero / short / overflow bounds no-op safely.
        let w = make_edge_f32(17 * 16, 55);
        let x = make_edge_f32(3 * 20, 77);
        let mut guard = [[3.25_f32; 16]; 3];
        gemm_f32_fma_16oc_strided(&w, &x, 20, 17, 0, &mut guard);
        gemm_f32_fma_16oc_strided(&w, &x, 20, 0, 3, &mut guard);
        gemm_f32_fma_16oc_strided(&w[..w.len() - 1], &x, 20, 17, 3, &mut guard);
        gemm_f32_fma_16oc_strided(&w, &x[..2 * 20 + 16], 20, 17, 3, &mut guard);
        gemm_f32_fma_16oc_strided(&w, &x, 20, 17, 3, &mut guard[..2]);
        gemm_f32_fma_16oc_strided(&w, &x, usize::MAX, 17, 2, &mut guard);
        assert_eq!(
            guard.map(|r| r.map(f32::to_bits)),
            [[3.25_f32.to_bits(); 16]; 3]
        );
    }

    #[test]
    fn conv1d_f32_fma_16oc_strided_matches_portable_scalar_fma_bit_exact() {
        const IN_CS: [usize; 9] = [1, 3, 4, 7, 16, 128, 129, 256, 512];
        const KERNELS: [usize; 3] = [1, 3, 7];
        const STRIDES: [usize; 2] = [1, 2];
        const DILATIONS: [usize; 3] = [1, 3, 9];

        // Full frames = 1..=67 coverage across representative channel/kernel/stride/dilation configs.
        for frames in 1_usize..=67 {
            let in_c = IN_CS[frames % IN_CS.len()];
            let kernel = KERNELS[(frames / 3) % KERNELS.len()];
            let stride = STRIDES[frames % STRIDES.len()];
            let dilation = DILATIONS[(frames / 2) % DILATIONS.len()];
            let seed = frames * 37 + in_c * 13 + kernel * 7;
            let w = make_edge_f32(kernel * in_c * 16, seed + 1);
            let ext_frames = (frames - 1) * stride + (kernel - 1) * dilation + 1;
            let ext = make_edge_f32(ext_frames * in_c, seed + 19);
            let init: Vec<[f32; 16]> = (0..frames)
                .map(|f| {
                    let v = make_edge_f32(16, seed + 400 + f);
                    std::array::from_fn(|oc| v[oc])
                })
                .collect();
            let mut acc_nat = init.clone();
            let mut acc_prt = init;
            conv1d_f32_fma_16oc_strided(
                &w,
                &ext,
                in_c,
                kernel,
                stride,
                dilation,
                frames,
                &mut acc_nat,
            );
            conv1d_f32_fma_16oc_strided_portable(
                &w,
                &ext,
                in_c,
                kernel,
                stride,
                dilation,
                frames,
                &mut acc_prt,
            );
            for f in 0..frames {
                assert_eq!(
                    acc_nat[f].map(f32::to_bits),
                    acc_prt[f].map(f32::to_bits),
                    "conv1d_strided sweep 0-ULP mismatch frames={frames} in_c={in_c} k={kernel} s={stride} d={dilation} f={f}"
                );
            }
        }

        // Cross-product across all in_c, kernel, stride, dilation with 6f + tail frames (1, 5, 6, 7, 13, 48).
        for (c_idx, &in_c) in IN_CS.iter().enumerate() {
            for &kernel in &KERNELS {
                for &stride in &STRIDES {
                    for &dilation in &DILATIONS {
                        let frames =
                            [1_usize, 5, 6, 7, 13, 48][(c_idx + kernel + stride + dilation) % 6];
                        let seed = c_idx * 53 + kernel * 11 + stride * 3 + dilation;
                        let w = make_edge_f32(kernel * in_c * 16, seed + 2);
                        let ext_frames = (frames - 1) * stride + (kernel - 1) * dilation + 1;
                        let ext = make_edge_f32(ext_frames * in_c, seed + 29);
                        let init: Vec<[f32; 16]> = (0..frames)
                            .map(|f| {
                                let v = make_edge_f32(16, seed + 500 + f);
                                std::array::from_fn(|oc| v[oc])
                            })
                            .collect();
                        let mut acc_nat = init.clone();
                        let mut acc_prt = init;
                        conv1d_f32_fma_16oc_strided(
                            &w,
                            &ext,
                            in_c,
                            kernel,
                            stride,
                            dilation,
                            frames,
                            &mut acc_nat,
                        );
                        conv1d_f32_fma_16oc_strided_portable(
                            &w,
                            &ext,
                            in_c,
                            kernel,
                            stride,
                            dilation,
                            frames,
                            &mut acc_prt,
                        );
                        for f in 0..frames {
                            assert_eq!(
                                acc_nat[f].map(f32::to_bits),
                                acc_prt[f].map(f32::to_bits),
                                "conv1d_strided matrix 0-ULP mismatch in_c={in_c} k={kernel} s={stride} d={dilation} frames={frames} f={f}"
                            );
                        }
                    }
                }
            }
        }

        // Zero / short / overflow bounds no-op safely.
        let w = make_edge_f32(3 * 17 * 16, 88);
        let ext = make_edge_f32(15 * 17, 99);
        let mut guard = [[-1.75_f32; 16]; 4];
        conv1d_f32_fma_16oc_strided(&w, &ext, 17, 3, 2, 3, 0, &mut guard);
        conv1d_f32_fma_16oc_strided(&w, &ext, 0, 3, 2, 3, 4, &mut guard);
        conv1d_f32_fma_16oc_strided(&w, &ext, 17, 0, 2, 3, 4, &mut guard);
        conv1d_f32_fma_16oc_strided(&w[..w.len() - 1], &ext, 17, 3, 2, 3, 4, &mut guard);
        // Needed ext frames = (4 - 1)*2 + (3 - 1)*3 + 1 = 13 -> 13 * 17 = 221.
        conv1d_f32_fma_16oc_strided(&w, &ext[..220], 17, 3, 2, 3, 4, &mut guard);
        conv1d_f32_fma_16oc_strided(&w, &ext, 17, 3, 2, 3, 4, &mut guard[..3]);
        conv1d_f32_fma_16oc_strided(&w, &ext, 17, 3, usize::MAX, 3, 4, &mut guard);
        assert_eq!(
            guard.map(|r| r.map(f32::to_bits)),
            [[(-1.75_f32).to_bits(); 16]; 4]
        );
    }
}
