//! Safe public boundary for ARM64 `SDOT` / `I8MM` / `FCVTL` / `FCVTN` quantized
//! kernels with runtime CPU feature detection and exact portable fallbacks.

pub use bonsai_neon_dot::kquants::{
    Arm64QuantCaps, Q4_K_BLOCK_BYTES, Q4_K_REPACK8_BLOCK_BYTES, Q6_K_BLOCK_BYTES, Q8_0_BLOCK_BYTES,
    Q8_0_BLOCK_ELEMENTS, Q8_0ActBlock, Q8_KActBlock, QK_K, dot_q4_k_4rows, dot_q4_k_repacked8,
    dot_q4_k_row, dot_q6_k_4rows, dot_q6_k_row, dot_q8_0_4rows, dot_q8_0_row, gemm_q4_k_4x2,
    gemm_q4_k_repacked8_8x2, gemm_q6_k_4x2, gemm_q8_0_4x2, quantize_row_q8_0, quantize_row_q8_k,
    repack_q4_k_8rows,
};
pub use bonsai_neon_dot::{
    narrow_f32_slice_checked as narrow_f32_slice_to_f16, widen_f16_eight, widen_f16_four,
    widen_f16_slice,
};
pub use bonsai_ternary::{checked_float_to_half, float_to_half, unpack_q4_k_scale_min};

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
}
