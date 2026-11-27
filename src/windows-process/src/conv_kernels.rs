//! Safe public boundary for ARM64 NEON `16oc x 1..=4f` outer-product `f32`
//! convolution/GEMM kernels with exact `0-ULP` portable scalar FMA fallback.

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native::conv_f32::gemm_f32_fma_16oc_portable;

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
}
