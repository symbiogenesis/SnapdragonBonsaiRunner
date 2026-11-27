//! Safe public boundary for ARM64 `SDOT` / `I8MM` / `FCVTL` / `FCVTN` quantized
//! kernels with runtime CPU feature detection and exact portable fallbacks.

pub use bonsai_neon_dot::kquants::{
    Arm64QuantCaps, Q4_K_BLOCK_BYTES, Q4_K_REPACK8_BLOCK_BYTES, Q6_K_BLOCK_BYTES, Q8_0_BLOCK_BYTES,
    Q8_0_BLOCK_ELEMENTS, Q8_0ActBlock, Q8_KActBlock, QK_K, dot_q4_k_4rows, dot_q4_k_row,
    dot_q6_k_4rows, dot_q6_k_row, dot_q8_0_4rows, dot_q8_0_row, gemm_q4_k_4x2,
    gemm_q4_k_repacked8_8x2, gemm_q6_k_4x2, gemm_q8_0_4x2, quantize_row_q8_0, quantize_row_q8_k,
    repack_q4_k_8rows,
};
pub use bonsai_neon_dot::{
    narrow_f32_slice_checked as narrow_f32_slice_to_f16, widen_f16_eight, widen_f16_four,
    widen_f16_slice,
};
pub use bonsai_ternary::{checked_float_to_half, float_to_half, unpack_q4_k_scale_min};

/// Computes an 8-row repacked `Q4_K` strip dot product (`0-ULP` vs `dot_q4_k_row`).
#[must_use]
pub fn dot_q4_k_repacked8(caps: Arm64QuantCaps, strip: &[u8], act: &[Q8_KActBlock]) -> [f32; 8] {
    crate::native::quant_q8::dot_q4_k_repacked8(caps, strip, act)
}

/// Computes eight `Q8_0` row dot products while sharing activations per block (`0-ULP` vs `dot_q8_0_row`).
#[must_use]
pub fn dot_q8_0_8rows(caps: Arm64QuantCaps, rows: [&[u8]; 8], act: &[Q8_0ActBlock]) -> [f32; 8] {
    crate::native::quant_q8::dot_q8_0_8rows(caps, rows, act)
}

/// Computes an 8-row by 8-token `Q8_0` tile using `I8MM` (`SMMLA`) or exact `SDOT`/portable fallback.
#[must_use]
pub fn gemm_q8_0_8x8(
    caps: Arm64QuantCaps,
    rows: [&[u8]; 8],
    acts: [&[Q8_0ActBlock]; 8],
) -> [[f32; 8]; 8] {
    crate::native::quant_q8::gemm_q8_0_8x8(caps, rows, acts)
}

/// Computes an 8-row by 4-token `Q8_0` tile using `I8MM` (`SMMLA`) or exact `SDOT`/portable fallback.
#[must_use]
pub fn gemm_q8_0_8x4(
    caps: Arm64QuantCaps,
    rows: [&[u8]; 8],
    acts: [&[Q8_0ActBlock]; 4],
) -> [[f32; 4]; 8] {
    crate::native::quant_q8::gemm_q8_0_8x4(caps, rows, acts)
}

/// Computes an 8-row by 2-token `Q8_0` tile using `I8MM` (`SMMLA`) or exact `SDOT`/portable fallback.
#[must_use]
pub fn gemm_q8_0_8x2(
    caps: Arm64QuantCaps,
    rows: [&[u8]; 8],
    acts: [&[Q8_0ActBlock]; 2],
) -> [[f32; 2]; 8] {
    crate::native::quant_q8::gemm_q8_0_8x2(caps, rows, acts)
}

/// Computes an 8-row by 8-token repacked `Q4_K` strip GEMM tile using `I8MM` (`SMMLA`) or exact fallback.
#[must_use]
pub fn gemm_q4_k_repacked8_8x8(
    caps: Arm64QuantCaps,
    strip: &[u8],
    acts: [&[Q8_KActBlock]; 8],
) -> [[f32; 8]; 8] {
    crate::native::quant_q8::gemm_q4_k_repacked8_8x8(caps, strip, acts)
}

/// Computes an 8-row by 4-token repacked `Q4_K` strip GEMM tile using `I8MM` (`SMMLA`) or exact fallback.
#[must_use]
pub fn gemm_q4_k_repacked8_8x4(
    caps: Arm64QuantCaps,
    strip: &[u8],
    acts: [&[Q8_KActBlock]; 4],
) -> [[f32; 4]; 8] {
    crate::native::quant_q8::gemm_q4_k_repacked8_8x4(caps, strip, acts)
}

/// Computes eight `Q6_K` row dot products while sharing activations per superblock (`0-ULP` vs `dot_q6_k_row`).
#[must_use]
pub fn dot_q6_k_8rows(caps: Arm64QuantCaps, rows: [&[u8]; 8], act: &[Q8_KActBlock]) -> [f32; 8] {
    crate::native::quant_q8::dot_q6_k_8rows(caps, rows, act)
}

/// Computes an 8-row by 8-token `Q6_K` tile using `I8MM` (`SMMLA`) or exact `SDOT`/portable fallback.
#[must_use]
pub fn gemm_q6_k_8x8(
    caps: Arm64QuantCaps,
    rows: [&[u8]; 8],
    acts: [&[Q8_KActBlock]; 8],
) -> [[f32; 8]; 8] {
    crate::native::quant_q8::gemm_q6_k_8x8(caps, rows, acts)
}

/// Computes an 8-row by 4-token `Q6_K` tile using `I8MM` (`SMMLA`) or exact `SDOT`/portable fallback.
#[must_use]
pub fn gemm_q6_k_8x4(
    caps: Arm64QuantCaps,
    rows: [&[u8]; 8],
    acts: [&[Q8_KActBlock]; 4],
) -> [[f32; 4]; 8] {
    crate::native::quant_q8::gemm_q6_k_8x4(caps, rows, acts)
}

/// Computes an 8-row by 2-token `Q6_K` tile using `I8MM` (`SMMLA`) or exact `SDOT`/portable fallback.
#[must_use]
pub fn gemm_q6_k_8x2(
    caps: Arm64QuantCaps,
    rows: [&[u8]; 8],
    acts: [&[Q8_KActBlock]; 2],
) -> [[f32; 2]; 8] {
    crate::native::quant_q8::gemm_q6_k_8x2(caps, rows, acts)
}

/// Computes `(dot(q0, k), dot(q1, k))` from a shared `binary16` key vector `k_f16`
/// using baseline ARM64 `FCVTL`/`FCVTL2` widening and the exact 2-accumulator FMA +
/// pair-lane reduction order of `speech-tts/src/numerics.rs`.
///
/// Returns `(0.0, 0.0)` when slice lengths do not match.
#[must_use]
pub fn dot_f16_pair_f32(q0: &[f32], q1: &[f32], k_f16: &[u16]) -> (f32, f32) {
    crate::native::quant_q8::dot_f16_pair_f32(q0, q1, k_f16)
}

/// Accumulates `out0 += w0 * v` and `out1 += w1 * v` from a shared `binary16` value vector `v_f16`
/// using baseline ARM64 `FCVTL`/`FCVTL2` widening and exact per-lane/tail `fma` rounding.
///
/// No-ops when slice lengths do not match.
pub fn axpy_f16_pair_f32(w0: f32, w1: f32, v_f16: &[u16], out0: &mut [f32], out1: &mut [f32]) {
    crate::native::quant_q8::axpy_f16_pair_f32(w0, w1, v_f16, out0, out1);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f16_bits_to_f32(bytes: [u8; 2]) -> f32 {
        bonsai_ternary::half_to_float(u16::from_le_bytes(bytes))
    }

    fn f32_to_f16_bits_ref(value: f32) -> u16 {
        checked_float_to_half(value).unwrap_or(0)
    }

    fn make_q8_0_row(blocks: usize, seed: usize) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(blocks * Q8_0_BLOCK_BYTES);
        for b in 0..blocks {
            let scale = 0.015_f32 * f32::from(u8::try_from((seed + b) % 13 + 1).unwrap_or(1));
            bytes.extend_from_slice(&f32_to_f16_bits_ref(scale).to_le_bytes());
            for k in 0..32 {
                let q = i8::try_from(
                    isize::try_from((seed * 17 + b * 7 + k * 5) % 199).unwrap_or(0) - 99,
                )
                .unwrap_or(0);
                bytes.push(q.cast_unsigned());
            }
        }
        bytes
    }

    fn dequant_q8_0(row: &[u8]) -> Vec<f64> {
        let mut out = Vec::new();
        for blk in row.as_chunks::<34>().0 {
            let d = f64::from(f16_bits_to_f32([blk[0], blk[1]]));
            for &q in &blk[2..] {
                out.push(d * f64::from(q.cast_signed()));
            }
        }
        out
    }

    fn make_q4_k_row(blocks: usize, seed: usize) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(blocks * Q4_K_BLOCK_BYTES);
        for b in 0..blocks {
            let d = 0.012_f32 * f32::from(u8::try_from((seed + b) % 9 + 1).unwrap_or(1));
            let dmin = 0.004_f32 * f32::from(u8::try_from((seed + b * 3) % 7 + 1).unwrap_or(1));
            bytes.extend_from_slice(&f32_to_f16_bits_ref(d).to_le_bytes());
            bytes.extend_from_slice(&f32_to_f16_bits_ref(dmin).to_le_bytes());
            for i in 0..12 {
                bytes.push(u8::try_from((seed * 13 + b * 29 + i * 41) % 251).unwrap_or(0));
            }
            for i in 0..128 {
                bytes.push(u8::try_from((seed * 31 + b * 19 + i * 37) % 256).unwrap_or(0));
            }
        }
        bytes
    }

    fn dequant_q4_k(row: &[u8]) -> Vec<f64> {
        let mut out = vec![0.0_f64; (row.len() / Q4_K_BLOCK_BYTES) * QK_K];
        for (bi, blk) in row.as_chunks::<144>().0.iter().enumerate() {
            let d = f64::from(f16_bits_to_f32([blk[0], blk[1]]));
            let dmin = f64::from(f16_bits_to_f32([blk[2], blk[3]]));
            for g in 0..8 {
                let (sc, m) = unpack_q4_k_scale_min(g, &blk[4..16]);
                let qs_base = 16 + (g / 2) * 32;
                for k in 0..32 {
                    let byte = blk[qs_base + k];
                    let q = if g % 2 == 0 { byte & 0x0f } else { byte >> 4 };
                    out[bi * 256 + g * 32 + k] =
                        d * f64::from(sc) * f64::from(q) - dmin * f64::from(m);
                }
            }
        }
        out
    }

    fn make_q6_k_row(blocks: usize, seed: usize) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(blocks * Q6_K_BLOCK_BYTES);
        for b in 0..blocks {
            for i in 0..128 {
                bytes.push(u8::try_from((seed * 23 + b * 11 + i * 43) % 256).unwrap_or(0));
            }
            for i in 0..64 {
                bytes.push(u8::try_from((seed * 19 + b * 7 + i * 53) % 256).unwrap_or(0));
            }
            for i in 0..16 {
                let s = i8::try_from(
                    isize::try_from((seed * 5 + b * 3 + i * 11) % 63).unwrap_or(0) - 31,
                )
                .unwrap_or(1);
                bytes.push(s.cast_unsigned());
            }
            let d = 0.008_f32 * f32::from(u8::try_from((seed + b) % 11 + 1).unwrap_or(1));
            bytes.extend_from_slice(&f32_to_f16_bits_ref(d).to_le_bytes());
        }
        bytes
    }

    fn dequant_q6_k(row: &[u8]) -> Vec<f64> {
        let mut out = vec![0.0_f64; (row.len() / Q6_K_BLOCK_BYTES) * QK_K];
        for (bi, blk) in row.as_chunks::<210>().0.iter().enumerate() {
            let d = f64::from(f16_bits_to_f32([blk[208], blk[209]]));
            for h in 0..2 {
                for q in 0..4 {
                    for i in 0..32 {
                        let ql = blk[h * 64 + (q % 2) * 32 + i];
                        let low = if q < 2 { ql & 0x0f } else { ql >> 4 };
                        let high = (blk[128 + h * 32 + i] >> (q * 2)) & 0x03;
                        let quant = i32::from(low | (high << 4)) - 32;
                        let sc = i32::from(blk[192 + h * 8 + q * 2 + i / 16].cast_signed());
                        out[bi * 256 + h * 128 + q * 32 + i] = d * f64::from(sc) * f64::from(quant);
                    }
                }
            }
        }
        out
    }

    fn dot_f64(w: &[f64], x: &[f32]) -> f64 {
        w.iter()
            .zip(x.iter())
            .map(|(&wv, &xv)| wv * f64::from(xv))
            .sum()
    }

    fn dot_f64_f64(w: &[f64], x: &[f64]) -> f64 {
        w.iter().zip(x.iter()).map(|(&wv, &xv)| wv * xv).sum()
    }

    fn dequant_q8_0_act(act: &[Q8_0ActBlock]) -> Vec<f64> {
        let mut out = Vec::with_capacity(act.len() * Q8_0_BLOCK_ELEMENTS);
        for blk in act {
            let d = f64::from(blk.d);
            for &q in &blk.qs {
                out.push(d * f64::from(q));
            }
        }
        out
    }

    fn dequant_q8_k_act(act: &[Q8_KActBlock]) -> Vec<f64> {
        let mut out = Vec::with_capacity(act.len() * QK_K);
        for blk in act {
            let d = f64::from(blk.d);
            for &q in &blk.qs {
                out.push(d * f64::from(q));
            }
        }
        out
    }

    #[test]
    fn half_widening_and_narrowing_match_ieee_binary16_exact_edge_cases() {
        for bits in 0..=u16::MAX {
            let four =
                std::array::from_fn(|i| bits.wrapping_add(u16::from(i.to_le_bytes()[0]) * 17));
            let expected4 = four.map(bonsai_ternary::half_to_float);
            assert_eq!(
                widen_f16_four(&four).map(f32::to_bits),
                expected4.map(f32::to_bits)
            );
            let eight =
                std::array::from_fn(|i| bits.wrapping_add(u16::from(i.to_le_bytes()[0]) * 8191));
            let actual8 = widen_f16_eight(&eight).map(|g| g.map(f32::to_bits));
            let expected8: [[u32; 4]; 2] = std::array::from_fn(|group| {
                std::array::from_fn(|lane| {
                    bonsai_ternary::half_to_float(eight[group * 4 + lane]).to_bits()
                })
            });
            assert_eq!(actual8, expected8);
            let mag = bits & 0x7fff;
            if mag < 0x7c00 {
                let widened = widen_f16_four(&[bits, bits ^ 0x8000, 0x0000, 0x8000]);
                let mut repacked = [0_u16; 4];
                assert!(narrow_f32_slice_to_f16(&widened, &mut repacked));
                assert_eq!(repacked[0], bits);
                assert_eq!(repacked[1], bits ^ 0x8000);
                assert_eq!(repacked[2], 0x0000);
                assert_eq!(repacked[3], 0x8000);
            }
        }
        let mut bad = [0_u16; 4];
        assert!(!narrow_f32_slice_to_f16(
            &[1.0, 65_520.0, 0.0, 0.0],
            &mut bad
        ));
        assert!(!narrow_f32_slice_to_f16(
            &[1.0, f32::NAN, 0.0, 0.0],
            &mut bad
        ));
        assert!(!narrow_f32_slice_to_f16(
            &[1.0, -65_520.0, 0.0, 0.0],
            &mut bad
        ));
        assert!(narrow_f32_slice_to_f16(
            &[65_504.0, -65_504.0, -0.0, 1.0 + 3.0 / 2048.0],
            &mut bad
        ));
        assert_eq!(bad[2], 0x8000);

        let mut mismatch = [0_u16; 1];
        assert!(!narrow_f32_slice_to_f16(&[1.0, 2.0], &mut mismatch));
        assert_eq!(
            checked_float_to_half(f32::from_bits(0x477f_efff)),
            Some(0x7bff)
        );
        assert_eq!(
            checked_float_to_half(f32::from_bits(0xc77f_efff)),
            Some(0xfbff)
        );
        for len in 0..=23_usize {
            let src: Vec<f32> = (0..len)
                .map(|i| {
                    bonsai_ternary::half_to_float(
                        u16::try_from((i * 199 + 1) & 0x7bff).unwrap_or(0),
                    )
                })
                .collect();
            let mut narrowed = vec![0_u16; len];
            assert!(narrow_f32_slice_to_f16(&src, &mut narrowed));
            assert_eq!(
                narrowed,
                src.iter()
                    .copied()
                    .filter_map(checked_float_to_half)
                    .collect::<Vec<_>>()
            );
            let mut widened = vec![0.0_f32; len];
            widen_f16_slice(&narrowed, &mut widened);
            assert_eq!(
                widened
                    .iter()
                    .copied()
                    .map(f32::to_bits)
                    .collect::<Vec<_>>(),
                src.iter().copied().map(f32::to_bits).collect::<Vec<_>>()
            );
            for bad_val in [65_520.0_f32, -65_520.0, f32::INFINITY, f32::NAN] {
                for bad_idx in 0..len {
                    let mut with_bad = src.clone();
                    with_bad[bad_idx] = bad_val;
                    assert_eq!(checked_float_to_half(bad_val), None);
                    assert!(!narrow_f32_slice_to_f16(&with_bad, &mut narrowed));
                }
            }
            if len > 0 {
                let mut edge = src;
                edge[len - 1] = f32::from_bits(0x477f_efff);
                assert!(narrow_f32_slice_to_f16(&edge, &mut narrowed));
                assert_eq!(narrowed[len - 1], 0x7bff);
            }
        }
        let snan = [
            0x7c01_u16, 0xfc01, 0x7e00, 0x3c00, 0x8000, 0x0001, 0x7bff, 0xfbff, 0x7c01,
        ];
        let mut snan_out = [0.0_f32; 9];
        widen_f16_slice(&snan, &mut snan_out);
        assert_eq!(
            snan_out.map(f32::to_bits),
            snan.map(|bits| bonsai_ternary::half_to_float(bits).to_bits())
        );
    }

    #[test]
    fn q8_0_q4_k_q6_k_and_repacked_kernels_match_portable_and_fp64_reference() {
        let caps = Arm64QuantCaps::detect();
        let port = Arm64QuantCaps::portable();

        let in_dim = 512_usize;
        let x0: Vec<f32> = (0..in_dim)
            .map(|i| f32::from(i16::try_from((i * 11) % 61).unwrap_or(0) - 30) * 0.05)
            .collect();
        let x1: Vec<f32> = (0..in_dim)
            .map(|i| f32::from(i16::try_from((i * 7 + 19) % 53).unwrap_or(0) - 26) * 0.06)
            .collect();

        // Q8_0
        let mut a0_q8 = vec![Q8_0ActBlock::default(); in_dim / 32];
        let mut a1_q8 = vec![Q8_0ActBlock::default(); in_dim / 32];
        quantize_row_q8_0(&x0, &mut a0_q8);
        quantize_row_q8_0(&x1, &mut a1_q8);
        let x0_q8_f64 = dequant_q8_0_act(&a0_q8);
        let x1_q8_f64 = dequant_q8_0_act(&a1_q8);
        let q8_rows: Vec<Vec<u8>> = (0..8).map(|r| make_q8_0_row(in_dim / 32, r + 1)).collect();
        let q8_refs = [
            &q8_rows[0][..],
            &q8_rows[1][..],
            &q8_rows[2][..],
            &q8_rows[3][..],
        ];
        let d_nat = dot_q8_0_4rows(caps, q8_refs, &a0_q8);
        let d_prt = dot_q8_0_4rows(port, q8_refs, &a0_q8);
        let g_nat = gemm_q8_0_4x2(caps, q8_refs, &a0_q8, &a1_q8);
        let g_prt = gemm_q8_0_4x2(port, q8_refs, &a0_q8, &a1_q8);
        assert_eq!(d_nat.map(f32::to_bits), d_prt.map(f32::to_bits));
        assert_eq!(
            dot_q8_0_row(caps, q8_refs[0], &a0_q8).to_bits(),
            d_prt[0].to_bits()
        );
        for r in 0..4 {
            assert_eq!(g_nat[r].map(f32::to_bits), g_prt[r].map(f32::to_bits));
            let w_f64 = dequant_q8_0(q8_refs[r]);
            assert!((f64::from(g_nat[r][0]) - dot_f64_f64(&w_f64, &x0_q8_f64)).abs() < 1e-2);
            assert!((f64::from(g_nat[r][1]) - dot_f64_f64(&w_f64, &x1_q8_f64)).abs() < 1e-2);
            assert!((f64::from(g_nat[r][0]) - dot_f64(&w_f64, &x0)).abs() < 2.5);
            assert!((f64::from(g_nat[r][1]) - dot_f64(&w_f64, &x1)).abs() < 2.5);
        }

        // Q4_K + Q4_K repacked x8
        let mut a0_qk = vec![Q8_KActBlock::default(); in_dim / 256];
        let mut a1_qk = vec![Q8_KActBlock::default(); in_dim / 256];
        quantize_row_q8_k(&x0, &mut a0_qk);
        quantize_row_q8_k(&x1, &mut a1_qk);
        let x0_qk_f64 = dequant_q8_k_act(&a0_qk);
        let x1_qk_f64 = dequant_q8_k_act(&a1_qk);
        let q4_rows: Vec<Vec<u8>> = (0..8).map(|r| make_q4_k_row(in_dim / 256, r + 1)).collect();
        let q4_refs = [
            &q4_rows[0][..],
            &q4_rows[1][..],
            &q4_rows[2][..],
            &q4_rows[3][..],
        ];
        assert_eq!(
            dot_q4_k_4rows(caps, q4_refs, &a0_qk).map(f32::to_bits),
            dot_q4_k_4rows(port, q4_refs, &a0_qk).map(f32::to_bits)
        );
        assert_eq!(
            dot_q4_k_row(caps, q4_refs[0], &a0_qk).to_bits(),
            dot_q4_k_row(port, q4_refs[0], &a0_qk).to_bits()
        );
        let q4_g_nat = gemm_q4_k_4x2(caps, q4_refs, &a0_qk, &a1_qk);
        let q4_g_prt = gemm_q4_k_4x2(port, q4_refs, &a0_qk, &a1_qk);
        for r in 0..4 {
            assert_eq!(q4_g_nat[r].map(f32::to_bits), q4_g_prt[r].map(f32::to_bits));
            let w_f64 = dequant_q4_k(q4_refs[r]);
            assert!((f64::from(q4_g_nat[r][0]) - dot_f64_f64(&w_f64, &x0_qk_f64)).abs() < 1e-3);
            assert!((f64::from(q4_g_nat[r][1]) - dot_f64_f64(&w_f64, &x1_qk_f64)).abs() < 1e-3);
            assert!((f64::from(q4_g_nat[r][0]) - dot_f64(&w_f64, &x0)).abs() < 2.5);
            assert!((f64::from(q4_g_nat[r][1]) - dot_f64(&w_f64, &x1)).abs() < 2.5);
        }

        let flat_q4: Vec<u8> = q4_rows.iter().flat_map(|r| r.iter().copied()).collect();
        let repacked = repack_q4_k_8rows(&flat_q4, in_dim, 8).unwrap_or_default();
        assert_eq!(repacked.len(), (in_dim / 256) * Q4_K_REPACK8_BLOCK_BYTES);
        let r8_dot_nat = dot_q4_k_repacked8(caps, &repacked, &a0_qk);
        let r8_dot_prt = dot_q4_k_repacked8(port, &repacked, &a0_qk);
        assert_eq!(r8_dot_nat.map(f32::to_bits), r8_dot_prt.map(f32::to_bits));
        for r in 0..4 {
            assert_eq!(
                r8_dot_nat[r].to_bits(),
                dot_q4_k_row(port, q4_refs[r], &a0_qk).to_bits()
            );
        }
        let r8_gemm_nat = gemm_q4_k_repacked8_8x2(caps, &repacked, &a0_qk, &a1_qk);
        let r8_gemm_prt = gemm_q4_k_repacked8_8x2(port, &repacked, &a0_qk, &a1_qk);
        for r in 0..8 {
            assert_eq!(
                r8_gemm_nat[r].map(f32::to_bits),
                r8_gemm_prt[r].map(f32::to_bits)
            );
        }

        // Q6_K
        let q6_rows: Vec<Vec<u8>> = (0..4).map(|r| make_q6_k_row(in_dim / 256, r + 1)).collect();
        let q6_refs = [
            &q6_rows[0][..],
            &q6_rows[1][..],
            &q6_rows[2][..],
            &q6_rows[3][..],
        ];
        assert_eq!(
            dot_q6_k_4rows(caps, q6_refs, &a0_qk).map(f32::to_bits),
            dot_q6_k_4rows(port, q6_refs, &a0_qk).map(f32::to_bits)
        );
        assert_eq!(
            dot_q6_k_row(caps, q6_refs[0], &a0_qk).to_bits(),
            dot_q6_k_row(port, q6_refs[0], &a0_qk).to_bits()
        );
        let q6_g_nat = gemm_q6_k_4x2(caps, q6_refs, &a0_qk, &a1_qk);
        let q6_g_prt = gemm_q6_k_4x2(port, q6_refs, &a0_qk, &a1_qk);
        for r in 0..4 {
            assert_eq!(q6_g_nat[r].map(f32::to_bits), q6_g_prt[r].map(f32::to_bits));
            let w_f64 = dequant_q6_k(q6_refs[r]);
            assert!((f64::from(q6_g_nat[r][0]) - dot_f64_f64(&w_f64, &x0_qk_f64)).abs() < 1e-3);
            assert!((f64::from(q6_g_nat[r][1]) - dot_f64_f64(&w_f64, &x1_qk_f64)).abs() < 1e-3);
            assert!((f64::from(q6_g_nat[r][0]) - dot_f64(&w_f64, &x0)).abs() < 6.0);
            assert!((f64::from(q6_g_nat[r][1]) - dot_f64(&w_f64, &x1)).abs() < 6.0);
        }
    }

    fn make_edge_q8_0_row(blocks: usize, seed: usize) -> Vec<u8> {
        // Mix positive, negative, subnormal, +0.0, -0.0, and FMA-sensitive scales plus extreme i8 values.
        const SCALE_BITS: [u16; 10] = [
            0x3C01, // 1 + 2^-10 (FMA-sensitive)
            0xBC03, // -(1 + 3*2^-10)
            0x0001, // smallest positive subnormal f16
            0x8001, // smallest negative subnormal f16
            0x0000, // +0.0
            0x8000, // -0.0
            0x3555, // ~0.33325 (FMA-sensitive repeating binary)
            0xB557, // ~-0.33374
            0x4213, // ~3.037
            0xC215, // ~-3.041
        ];
        const EXTREME_I8: [i8; 8] = [i8::MIN, i8::MIN + 1, -63, -1, 0, 1, 64, i8::MAX];
        let mut bytes = Vec::with_capacity(blocks * Q8_0_BLOCK_BYTES);
        for b in 0..blocks {
            let scale = SCALE_BITS[(seed * 5 + b * 3) % SCALE_BITS.len()];
            bytes.extend_from_slice(&scale.to_le_bytes());
            for k in 0..32 {
                let q = if (seed + b + k) % 7 == 0 {
                    EXTREME_I8[(seed + b * 11 + k) % EXTREME_I8.len()]
                } else {
                    i8::try_from(
                        isize::try_from((seed * 29 + b * 17 + k * 43) % 256).unwrap_or(0) - 128,
                    )
                    .unwrap_or(0)
                };
                bytes.push(q.cast_unsigned());
            }
        }
        bytes
    }

    fn make_edge_q8_0_act(blocks: usize, token: usize) -> Vec<Q8_0ActBlock> {
        const EXTREME_I8: [i8; 8] = [i8::MIN, -127, -1, 0, 1, 63, 126, i8::MAX];
        (0..blocks)
            .map(|b| {
                let d = match (token + b) % 7 {
                    0 => -0.0_f32,
                    1 => 0.0_f32,
                    2 => -0.333_333_34_f32,
                    3 => 1.0_f32 / 7.0_f32,
                    4 => -2.125_f32,
                    5 => 1.0e-4_f32,
                    _ => 0.777_777_8_f32,
                };
                let mut qs = [0_i8; 32];
                for (k, slot) in qs.iter_mut().enumerate() {
                    *slot = if (token + b + k) % 5 == 0 {
                        EXTREME_I8[(token * 3 + b * 7 + k) % EXTREME_I8.len()]
                    } else {
                        i8::try_from(
                            isize::try_from((token * 37 + b * 19 + k * 23) % 256).unwrap_or(0)
                                - 128,
                        )
                        .unwrap_or(0)
                    };
                }
                Q8_0ActBlock { d, qs }
            })
            .collect()
    }

    #[test]
    fn q8_0_8row_and_8x_gemm_match_row_and_portable_bit_exact() {
        let caps = Arm64QuantCaps::detect();
        let port = Arm64QuantCaps::portable();

        for blocks in [0_usize, 1, 2, 3, 5, 17] {
            let rows_vec: Vec<Vec<u8>> = (0..8)
                .map(|r| {
                    if r % 2 == 0 {
                        make_edge_q8_0_row(blocks, r + 1)
                    } else {
                        make_q8_0_row(blocks, r + 9)
                    }
                })
                .collect();
            let row_refs: [&[u8]; 8] = std::array::from_fn(|r| rows_vec[r].as_slice());

            let acts_vec: [Vec<Q8_0ActBlock>; 8] =
                std::array::from_fn(|t| make_edge_q8_0_act(blocks, t));
            let act8_refs: [&[Q8_0ActBlock]; 8] = std::array::from_fn(|t| acts_vec[t].as_slice());
            let act4_refs: [&[Q8_0ActBlock]; 4] = std::array::from_fn(|t| acts_vec[t].as_slice());
            let act2_refs: [&[Q8_0ActBlock]; 2] = std::array::from_fn(|t| acts_vec[t].as_slice());

            let expected: [[u32; 8]; 8] = std::array::from_fn(|r| {
                std::array::from_fn(|t| dot_q8_0_row(port, row_refs[r], act8_refs[t]).to_bits())
            });
            let expected_caps: [[u32; 8]; 8] = std::array::from_fn(|r| {
                std::array::from_fn(|t| dot_q8_0_row(caps, row_refs[r], act8_refs[t]).to_bits())
            });
            assert_eq!(expected, expected_caps);

            for t in 0..8 {
                let d_nat = dot_q8_0_8rows(caps, row_refs, act8_refs[t]);
                let d_prt = dot_q8_0_8rows(port, row_refs, act8_refs[t]);
                assert_eq!(d_nat.map(f32::to_bits), d_prt.map(f32::to_bits));
                for r in 0..8 {
                    assert_eq!(d_nat[r].to_bits(), expected[r][t]);
                }
            }

            let g8_nat = gemm_q8_0_8x8(caps, row_refs, act8_refs);
            let g8_prt = gemm_q8_0_8x8(port, row_refs, act8_refs);
            let g4_nat = gemm_q8_0_8x4(caps, row_refs, act4_refs);
            let g4_prt = gemm_q8_0_8x4(port, row_refs, act4_refs);
            let g2_nat = gemm_q8_0_8x2(caps, row_refs, act2_refs);
            let g2_prt = gemm_q8_0_8x2(port, row_refs, act2_refs);

            for r in 0..8 {
                assert_eq!(g8_nat[r].map(f32::to_bits), g8_prt[r].map(f32::to_bits));
                assert_eq!(g8_nat[r].map(f32::to_bits), expected[r]);
                assert_eq!(g4_nat[r].map(f32::to_bits), g4_prt[r].map(f32::to_bits));
                assert_eq!(g4_nat[r].map(f32::to_bits), expected[r][..4]);
                assert_eq!(g2_nat[r].map(f32::to_bits), g2_prt[r].map(f32::to_bits));
                assert_eq!(g2_nat[r].map(f32::to_bits), expected[r][..2]);
            }
        }
    }

    #[test]
    fn q8_0_8row_and_8x_gemm_invalid_lengths_return_zeros() {
        let caps = Arm64QuantCaps::detect();
        let valid_row = make_q8_0_row(2, 3);
        let short_row = &valid_row[..valid_row.len() - 1];
        let act = make_edge_q8_0_act(2, 0);
        let short_act = &act[..1];

        let mut bad_rows = [valid_row.as_slice(); 8];
        bad_rows[7] = short_row;
        assert_eq!(
            dot_q8_0_8rows(caps, bad_rows, &act).map(f32::to_bits),
            [0_u32; 8]
        );
        assert_eq!(
            gemm_q8_0_8x2(caps, bad_rows, [&act, &act]).map(|r| r.map(f32::to_bits)),
            [[0_u32; 2]; 8]
        );
        let good_rows = [valid_row.as_slice(); 8];
        assert_eq!(
            gemm_q8_0_8x4(caps, good_rows, [&act, short_act, &act, &act])
                .map(|r| r.map(f32::to_bits)),
            [[0_u32; 4]; 8]
        );
    }

    fn make_edge_q8_k_act(blocks: usize, token: usize) -> Vec<Q8_KActBlock> {
        const EXTREME_I8: [i8; 8] = [i8::MIN + 1, -99, -1, 0, 1, 63, 126, i8::MAX];
        (0..blocks)
            .map(|b| {
                let d = match (token + b) % 6 {
                    0 => -0.0_f32,
                    1 => 0.0_f32,
                    2 => -0.333_333_34_f32,
                    3 => 1.0_f32 / 7.0_f32,
                    4 => -1.625_f32,
                    _ => 0.618_033_9_f32,
                };
                let mut qs = [0_i8; 256];
                let mut bsums = [0_i16; 16];
                for (g, bsum) in bsums.iter_mut().enumerate() {
                    let mut sum = 0_i16;
                    for k in 0..16 {
                        let idx = g * 16 + k;
                        let q = if (token + b + idx) % 5 == 0 {
                            EXTREME_I8[(token * 3 + b * 7 + idx) % EXTREME_I8.len()]
                        } else {
                            i8::try_from(
                                isize::try_from((token * 37 + b * 19 + idx * 23) % 255)
                                    .unwrap_or(0)
                                    - 127,
                            )
                            .unwrap_or(0)
                        };
                        qs[idx] = q;
                        sum += i16::from(q);
                    }
                    *bsum = sum;
                }
                Q8_KActBlock { d, qs, bsums }
            })
            .collect()
    }

    #[test]
    fn q4_k_repacked8_8x8_and_8x4_match_row_and_portable_bit_exact() {
        let caps = Arm64QuantCaps::detect();
        let port = Arm64QuantCaps::portable();

        for blocks in [0_usize, 1, 2, 3, 5] {
            let in_dim = blocks * QK_K;
            let q4_rows: Vec<Vec<u8>> = (0..8).map(|r| make_q4_k_row(blocks, r + 1)).collect();
            let flat_q4: Vec<u8> = q4_rows.iter().flat_map(|r| r.iter().copied()).collect();
            let repacked = if blocks == 0 {
                Vec::new()
            } else {
                repack_q4_k_8rows(&flat_q4, in_dim, 8).unwrap_or_default()
            };
            assert_eq!(repacked.len(), blocks * Q4_K_REPACK8_BLOCK_BYTES);

            let acts_vec: [Vec<Q8_KActBlock>; 8] =
                std::array::from_fn(|t| make_edge_q8_k_act(blocks, t));
            let act8_refs: [&[Q8_KActBlock]; 8] = std::array::from_fn(|t| acts_vec[t].as_slice());
            let act4_refs: [&[Q8_KActBlock]; 4] = std::array::from_fn(|t| acts_vec[t].as_slice());

            let expected: [[u32; 8]; 8] = std::array::from_fn(|r| {
                std::array::from_fn(|t| dot_q4_k_row(port, &q4_rows[r], act8_refs[t]).to_bits())
            });

            let g8_nat = gemm_q4_k_repacked8_8x8(caps, &repacked, act8_refs);
            let g8_prt = gemm_q4_k_repacked8_8x8(port, &repacked, act8_refs);
            let g4_nat = gemm_q4_k_repacked8_8x4(caps, &repacked, act4_refs);
            let g4_prt = gemm_q4_k_repacked8_8x4(port, &repacked, act4_refs);

            for r in 0..8 {
                assert_eq!(g8_nat[r].map(f32::to_bits), g8_prt[r].map(f32::to_bits));
                assert_eq!(g8_nat[r].map(f32::to_bits), expected[r]);
                assert_eq!(g4_nat[r].map(f32::to_bits), g4_prt[r].map(f32::to_bits));
                assert_eq!(g4_nat[r].map(f32::to_bits), expected[r][..4]);
            }
        }

        let valid_strip = vec![0_u8; 2 * Q4_K_REPACK8_BLOCK_BYTES];
        let act = make_edge_q8_k_act(2, 0);
        let short_act = &act[..1];
        assert_eq!(
            gemm_q4_k_repacked8_8x4(caps, &valid_strip[..valid_strip.len() - 1], [&act; 4])
                .map(|r| r.map(f32::to_bits)),
            [[0_u32; 4]; 8]
        );
        assert_eq!(
            gemm_q4_k_repacked8_8x8(
                caps,
                &valid_strip,
                [&act, short_act, &act, &act, &act, &act, &act, &act]
            )
            .map(|r| r.map(f32::to_bits)),
            [[0_u32; 8]; 8]
        );
    }

    #[test]
    fn q6_k_8row_and_8x_gemm_match_row_and_portable_bit_exact() {
        let caps = Arm64QuantCaps::detect();
        let port = Arm64QuantCaps::portable();

        for blocks in [0_usize, 1, 2, 3, 5] {
            let q6_rows: Vec<Vec<u8>> = (0..8).map(|r| make_q6_k_row(blocks, r + 3)).collect();
            let row_refs: [&[u8]; 8] = std::array::from_fn(|r| q6_rows[r].as_slice());

            let acts_vec: [Vec<Q8_KActBlock>; 8] =
                std::array::from_fn(|t| make_edge_q8_k_act(blocks, t));
            let act8_refs: [&[Q8_KActBlock]; 8] = std::array::from_fn(|t| acts_vec[t].as_slice());
            let act4_refs: [&[Q8_KActBlock]; 4] = std::array::from_fn(|t| acts_vec[t].as_slice());
            let act2_refs: [&[Q8_KActBlock]; 2] = std::array::from_fn(|t| acts_vec[t].as_slice());

            let expected: [[u32; 8]; 8] = std::array::from_fn(|r| {
                std::array::from_fn(|t| dot_q6_k_row(port, row_refs[r], act8_refs[t]).to_bits())
            });
            let expected_caps: [[u32; 8]; 8] = std::array::from_fn(|r| {
                std::array::from_fn(|t| dot_q6_k_row(caps, row_refs[r], act8_refs[t]).to_bits())
            });
            assert_eq!(expected, expected_caps);

            for t in 0..8 {
                let d_nat = dot_q6_k_8rows(caps, row_refs, act8_refs[t]);
                let d_prt = dot_q6_k_8rows(port, row_refs, act8_refs[t]);
                assert_eq!(d_nat.map(f32::to_bits), d_prt.map(f32::to_bits));
                for r in 0..8 {
                    assert_eq!(d_nat[r].to_bits(), expected[r][t]);
                }
            }

            let g8_nat = gemm_q6_k_8x8(caps, row_refs, act8_refs);
            let g8_prt = gemm_q6_k_8x8(port, row_refs, act8_refs);
            let g4_nat = gemm_q6_k_8x4(caps, row_refs, act4_refs);
            let g4_prt = gemm_q6_k_8x4(port, row_refs, act4_refs);
            let g2_nat = gemm_q6_k_8x2(caps, row_refs, act2_refs);
            let g2_prt = gemm_q6_k_8x2(port, row_refs, act2_refs);

            for r in 0..8 {
                assert_eq!(g8_nat[r].map(f32::to_bits), g8_prt[r].map(f32::to_bits));
                assert_eq!(g8_nat[r].map(f32::to_bits), expected[r]);
                assert_eq!(g4_nat[r].map(f32::to_bits), g4_prt[r].map(f32::to_bits));
                assert_eq!(g4_nat[r].map(f32::to_bits), expected[r][..4]);
                assert_eq!(g2_nat[r].map(f32::to_bits), g2_prt[r].map(f32::to_bits));
                assert_eq!(g2_nat[r].map(f32::to_bits), expected[r][..2]);
            }
        }

        let valid_row = make_q6_k_row(2, 5);
        let short_row = &valid_row[..valid_row.len() - 1];
        let act = make_edge_q8_k_act(2, 0);
        let short_act = &act[..1];
        let mut bad_rows = [valid_row.as_slice(); 8];
        bad_rows[6] = short_row;
        assert_eq!(
            dot_q6_k_8rows(caps, bad_rows, &act).map(f32::to_bits),
            [0_u32; 8]
        );
        assert_eq!(
            gemm_q6_k_8x4(caps, bad_rows, [&act; 4]).map(|r| r.map(f32::to_bits)),
            [[0_u32; 4]; 8]
        );
        let good_rows = [valid_row.as_slice(); 8];
        assert_eq!(
            gemm_q6_k_8x8(
                caps,
                good_rows,
                [&act, &act, short_act, &act, &act, &act, &act, &act]
            )
            .map(|r| r.map(f32::to_bits)),
            [[0_u32; 8]; 8]
        );
    }

    #[test]
    fn dot_q4_k_repacked8_shadow_matches_row_and_portable_edge_cases_bit_exact() {
        let caps = Arm64QuantCaps::detect();
        let port = Arm64QuantCaps::portable();

        for blocks in [0_usize, 1, 2, 3, 5] {
            let in_dim = blocks * QK_K;
            let q4_rows: Vec<Vec<u8>> = (0..8).map(|r| make_q4_k_row(blocks, r + 5)).collect();
            let flat_q4: Vec<u8> = q4_rows.iter().flat_map(|r| r.iter().copied()).collect();
            let repacked = if blocks == 0 {
                Vec::new()
            } else {
                repack_q4_k_8rows(&flat_q4, in_dim, 8).unwrap_or_default()
            };
            assert_eq!(repacked.len(), blocks * Q4_K_REPACK8_BLOCK_BYTES);

            for t in 0..4 {
                let act = make_edge_q8_k_act(blocks, t);
                let nat = dot_q4_k_repacked8(caps, &repacked, &act);
                let prt = dot_q4_k_repacked8(port, &repacked, &act);
                assert_eq!(nat.map(f32::to_bits), prt.map(f32::to_bits));
                for r in 0..8 {
                    assert_eq!(
                        nat[r].to_bits(),
                        dot_q4_k_row(port, &q4_rows[r], &act).to_bits()
                    );
                }
            }
        }

        let valid_strip = vec![0_u8; 2 * Q4_K_REPACK8_BLOCK_BYTES];
        let act = make_edge_q8_k_act(2, 1);
        assert_eq!(
            dot_q4_k_repacked8(caps, &valid_strip[..valid_strip.len() - 1], &act).map(f32::to_bits),
            [0_u32; 8]
        );
    }

    #[test]
    fn f16_kv_pair_dot_and_axpy_match_portable_and_numerics_order_bit_exact() {
        let fma_dot_cases: &[(&[f32], &[f32], &[u16], f32)] = &[
            (&[], &[], &[], 0.0),
            (
                &[
                    16_777_216.0,
                    0.0,
                    0.0,
                    0.0,
                    -16_777_216.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.25,
                ],
                &[
                    16_777_216.0,
                    0.0,
                    0.0,
                    0.0,
                    -16_777_216.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.25,
                ],
                &[0x3c00; 13],
                0.25,
            ),
            (
                &[16_777_216.0, 1.0, -16_777_216.0, 1.0],
                &[16_777_216.0, 1.0, -16_777_216.0, 1.0],
                &[0x3c00; 4],
                1.0,
            ),
            (
                &[1.0, 0.0, 0.0, 0.0, f32::from_bits(0x3f80_0001)],
                &[1.0, 0.0, 0.0, 0.0, f32::from_bits(0x3f80_0001)],
                &[0xbc00, 0x0000, 0x0000, 0x0000, 0x3bff],
                f32::from_bits(0xb9ff_f002),
            ),
        ];
        for &(q0, q1, k_f16, expected) in fma_dot_cases {
            let nat = dot_f16_pair_f32(q0, q1, k_f16);
            let prt = crate::native::quant_q8::dot_f16_pair_f32_portable(q0, q1, k_f16);
            assert_eq!(nat.0.to_bits(), prt.0.to_bits());
            assert_eq!(nat.1.to_bits(), prt.1.to_bits());
            assert_eq!(nat.0.to_bits(), expected.to_bits());
        }

        const EDGE_F16: [u16; 12] = [
            0x0000, // +0.0
            0x8000, // -0.0
            0x0001, // min positive subnormal
            0x8001, // min negative subnormal
            0x3c00, // 1.0
            0xbc00, // -1.0
            0x3bff, // 1 - 2^-11
            0x3c01, // 1 + 2^-10
            0x7bff, // max finite
            0xfbff, // min finite
            0x3555, // ~0.33325
            0xb557, // ~-0.33374
        ];

        for len in 0..=35_usize {
            let k_f16: Vec<u16> = (0..len)
                .map(|i| EDGE_F16[(i * 5 + len) % EDGE_F16.len()])
                .collect();
            let q0: Vec<f32> = (0..len)
                .map(|i| f32::from(i16::try_from((i * 13 + 3) % 41).unwrap_or(0) - 20) * 0.125)
                .collect();
            let q1: Vec<f32> = (0..len)
                .map(|i| f32::from(i16::try_from((i * 7 + 11) % 37).unwrap_or(0) - 18) * 0.1875)
                .collect();

            let nat_dot = dot_f16_pair_f32(&q0, &q1, &k_f16);
            let prt_dot = crate::native::quant_q8::dot_f16_pair_f32_portable(&q0, &q1, &k_f16);
            assert_eq!(nat_dot.0.to_bits(), prt_dot.0.to_bits(), "len={len} dot0");
            assert_eq!(nat_dot.1.to_bits(), prt_dot.1.to_bits(), "len={len} dot1");

            let w0 = f32::from_bits(0x3f80_0001);
            let w1 = -0.375_f32;
            let mut out0_nat: Vec<f32> = (0..len)
                .map(|i| -1.0 + f32::from(u8::try_from(i).unwrap_or(0)) * 0.0625)
                .collect();
            let mut out1_nat: Vec<f32> = (0..len)
                .map(|i| 0.5 - f32::from(u8::try_from(i).unwrap_or(0)) * 0.03125)
                .collect();
            let mut out0_prt = out0_nat.clone();
            let mut out1_prt = out1_nat.clone();

            axpy_f16_pair_f32(w0, w1, &k_f16, &mut out0_nat, &mut out1_nat);
            crate::native::quant_q8::axpy_f16_pair_f32_portable(
                w0,
                w1,
                &k_f16,
                &mut out0_prt,
                &mut out1_prt,
            );
            assert_eq!(
                out0_nat
                    .iter()
                    .copied()
                    .map(f32::to_bits)
                    .collect::<Vec<_>>(),
                out0_prt
                    .iter()
                    .copied()
                    .map(f32::to_bits)
                    .collect::<Vec<_>>(),
                "len={len} axpy0"
            );
            assert_eq!(
                out1_nat
                    .iter()
                    .copied()
                    .map(f32::to_bits)
                    .collect::<Vec<_>>(),
                out1_prt
                    .iter()
                    .copied()
                    .map(f32::to_bits)
                    .collect::<Vec<_>>(),
                "len={len} axpy1"
            );
        }

        // Exceptional / NaN inputs preserve exact portable NaN payload and bit parity.
        let special_k = [
            0x7c01_u16, 0xfc01, 0x7e00, 0x3c00, 0x7c00, 0xfc00, 0x0001, 0x8000, 0x7c01,
        ];
        let sq0 = [1.0_f32; 9];
        let sq1 = [2.0_f32; 9];
        let nat_s = dot_f16_pair_f32(&sq0, &sq1, &special_k);
        let prt_s = crate::native::quant_q8::dot_f16_pair_f32_portable(&sq0, &sq1, &special_k);
        assert_eq!(nat_s.0.is_nan(), prt_s.0.is_nan());
        assert_eq!(nat_s.1.is_nan(), prt_s.1.is_nan());

        let mut ao0 = [0.0_f32; 9];
        let mut ao1 = [0.0_f32; 9];
        let mut po0 = [0.0_f32; 9];
        let mut po1 = [0.0_f32; 9];
        axpy_f16_pair_f32(1.0, 2.0, &special_k, &mut ao0, &mut ao1);
        crate::native::quant_q8::axpy_f16_pair_f32_portable(
            1.0, 2.0, &special_k, &mut po0, &mut po1,
        );
        assert_eq!(ao0.map(f32::to_bits), po0.map(f32::to_bits));
        assert_eq!(ao1.map(f32::to_bits), po1.map(f32::to_bits));

        // Length mismatch returns zeros / no-ops.
        assert_eq!(
            dot_f16_pair_f32(&[1.0, 2.0], &[1.0], &[0x3c00, 0x3c00]),
            (0.0, 0.0)
        );
        let mut untouched0 = [7.0_f32; 2];
        let mut untouched1 = [9.0_f32; 1];
        axpy_f16_pair_f32(
            1.0,
            1.0,
            &[0x3c00, 0x3c00],
            &mut untouched0,
            &mut untouched1,
        );
        assert_eq!(untouched0.map(f32::to_bits), [7.0_f32.to_bits(); 2]);
        assert_eq!(untouched1.map(f32::to_bits), [9.0_f32.to_bits(); 1]);
    }
}
