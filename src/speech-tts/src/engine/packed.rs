//! Direct packed projection kernels. Keep dequantization rounding and each
//! block's two-accumulator dot order identical to the original decoded path.
#![expect(
    clippy::inline_always,
    reason = "SIMD helpers inherit the dispatch site's target features through inlining."
)]
#![allow(
    clippy::suboptimal_flops,
    reason = "Separate dequantization and accumulation rounding must match the original packed projection."
)]

use fearless_simd::{f32x4, prelude::*, u8x16};

use crate::weights::{half, unpack_q4_k_scale_min};

#[inline(always)]
fn unsigned_lanes<S: Simd>(simd: S, bytes: u8x16<S>) -> [f32x4<S>; 4] {
    let (lo, hi) = simd.widen_u8x16(bytes);
    let (a, b) = simd.widen_u16x8(lo);
    let (c, d) = simd.widen_u16x8(hi);
    [a, b, c, d].map(|value| simd.cvt_f32_u32x4(value))
}

#[inline(always)]
fn signed_lanes<S: Simd>(simd: S, bytes: u8x16<S>) -> [f32x4<S>; 4] {
    let (lo, hi) = simd.widen_i8x16(bytes.bitcast());
    let (a, b) = simd.widen_i16x8(lo);
    let (c, d) = simd.widen_i16x8(hi);
    [a, b, c, d].map(|value| simd.cvt_f32_i32x4(value))
}

struct Dot<S: Simd> {
    even: f32x4<S>,
    odd: f32x4<S>,
}

impl<S: Simd> Dot<S> {
    #[inline(always)]
    fn new(simd: S) -> Self {
        Self {
            even: f32x4::splat(simd, 0.0),
            odd: f32x4::splat(simd, 0.0),
        }
    }

    #[inline(always)]
    fn load_inputs(simd: S, input: &[f32]) -> [f32x4<S>; 4] {
        [
            f32x4::from_slice(simd, &input[0..4]),
            f32x4::from_slice(simd, &input[4..8]),
            f32x4::from_slice(simd, &input[8..12]),
            f32x4::from_slice(simd, &input[12..16]),
        ]
    }

    #[inline(always)]
    fn push_loaded(&mut self, weights: [f32x4<S>; 4], inputs: [f32x4<S>; 4]) {
        self.even = weights[0].mul_add(inputs[0], self.even);
        self.odd = weights[1].mul_add(inputs[1], self.odd);
        self.even = weights[2].mul_add(inputs[2], self.even);
        self.odd = weights[3].mul_add(inputs[3], self.odd);
    }

    #[inline(always)]
    fn finish(self) -> f32 {
        let lanes: [f32; 4] = (self.even + self.odd).into();
        (lanes[0] + lanes[1]) + (lanes[2] + lanes[3])
    }
}

#[inline(always)]
fn q8_rows<S: Simd, const N: usize>(simd: S, sources: [&[u8]; N], input: &[f32]) -> [f32; N] {
    let scales = sources.map(|source| f32x4::splat(simd, half(source)));
    let mut dots = std::array::from_fn(|_| Dot::new(simd));
    for part in 0..2 {
        let offset = part * 16;
        let inputs = Dot::load_inputs(simd, &input[offset..offset + 16]);
        for row_index in 0..N {
            let quants = u8x16::from_slice(simd, &sources[row_index][2 + offset..2 + offset + 16]);
            let weights = signed_lanes(simd, quants).map(|quant| scales[row_index] * quant);
            dots[row_index].push_loaded(weights, inputs);
        }
    }
    dots.map(Dot::finish)
}

#[inline(always)]
fn q4_k_rows<S: Simd, const N: usize>(simd: S, sources: [&[u8]; N], input: &[f32]) -> [f32; N] {
    let row_scales_min = sources.map(|source| (half(source), half(&source[2..]), &source[4..16]));
    let mut dots = std::array::from_fn(|_| Dot::new(simd));
    for group in 0..8 {
        let params = row_scales_min.map(|(scale, minimum, scales)| {
            let (s, m) = unpack_q4_k_scale_min(group, scales);
            (
                f32x4::splat(simd, scale * f32::from(s)),
                f32x4::splat(simd, minimum * f32::from(m)),
            )
        });
        for part in 0..2 {
            let input_start = group * 32 + part * 16;
            let inputs = Dot::load_inputs(simd, &input[input_start..input_start + 16]);
            let start = 16 + (group / 2) * 32 + part * 16;
            for row_index in 0..N {
                let packed = u8x16::from_slice(simd, &sources[row_index][start..start + 16]);
                let quants = if group % 2 == 0 {
                    packed & u8x16::splat(simd, 15)
                } else {
                    packed >> 4
                };
                let (multiplier, offset) = params[row_index];
                // Separate multiply/subtract preserves decode_block's rounding.
                let weights = unsigned_lanes(simd, quants).map(|quant| multiplier * quant - offset);
                dots[row_index].push_loaded(weights, inputs);
            }
        }
    }
    dots.map(Dot::finish)
}

#[inline(always)]
fn q6_k_rows<S: Simd, const N: usize>(simd: S, sources: [&[u8]; N], input: &[f32]) -> [f32; N] {
    let scales = sources.map(|source| half(&source[208..]));
    let mut dots = std::array::from_fn(|_| Dot::new(simd));
    for half_index in 0..2 {
        for quarter in 0..4 {
            for part in 0..2 {
                let input_start = half_index * 128 + quarter * 32 + part * 16;
                let inputs = Dot::load_inputs(simd, &input[input_start..input_start + 16]);
                let low_start = half_index * 64 + (quarter % 2) * 32 + part * 16;
                let high_start = 128 + half_index * 32 + part * 16;
                let shift = [0, 2, 4, 6][quarter];
                let scale_index = 192 + half_index * 8 + quarter * 2 + part;
                for row_index in 0..N {
                    let source = sources[row_index];
                    let packed = u8x16::from_slice(simd, &source[low_start..low_start + 16]);
                    let low = if quarter < 2 {
                        packed & u8x16::splat(simd, 15)
                    } else {
                        packed >> 4
                    };
                    let high = u8x16::from_slice(simd, &source[high_start..high_start + 16]);
                    let high = (high >> shift) & u8x16::splat(simd, 3);
                    let quants = low | (high << 4);
                    let subscale = i8::from_le_bytes([source[scale_index]]);
                    let multiplier = f32x4::splat(simd, scales[row_index] * f32::from(subscale));
                    let weights = unsigned_lanes(simd, quants)
                        .map(|quant| multiplier * (quant - f32x4::splat(simd, 32.0)));
                    dots[row_index].push_loaded(weights, inputs);
                }
            }
        }
    }
    dots.map(Dot::finish)
}

#[inline(always)]
pub(super) fn transpose_4x4<S: Simd>(simd: S, rows: [f32x4<S>; 4]) -> [f32x4<S>; 4] {
    let t0 = simd.zip_low_f32x4(rows[0], rows[1]);
    let t1 = simd.zip_high_f32x4(rows[0], rows[1]);
    let t2 = simd.zip_low_f32x4(rows[2], rows[3]);
    let t3 = simd.zip_high_f32x4(rows[2], rows[3]);
    let c0: f32x4<S> = simd.zip_low_u64x2(t0.bitcast(), t2.bitcast()).bitcast();
    let c1: f32x4<S> = simd.zip_high_u64x2(t0.bitcast(), t2.bitcast()).bitcast();
    let c2: f32x4<S> = simd.zip_low_u64x2(t1.bitcast(), t3.bitcast()).bitcast();
    let c3: f32x4<S> = simd.zip_high_u64x2(t1.bitcast(), t3.bitcast()).bitcast();
    [c0, c1, c2, c3]
}

#[inline(always)]
fn read_f32_le(bytes: &[u8]) -> f32 {
    f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

#[inline(always)]
fn load_f32x4_le<S: Simd>(simd: S, source: &[u8]) -> f32x4<S> {
    f32x4::from_slice(
        simd,
        &[
            read_f32_le(&source[0..4]),
            read_f32_le(&source[4..8]),
            read_f32_le(&source[8..12]),
            read_f32_le(&source[12..16]),
        ],
    )
}

#[inline(always)]
fn load_f16x4<S: Simd>(simd: S, source: &[u8]) -> f32x4<S> {
    let bits = [
        u16::from_le_bytes([source[0], source[1]]),
        u16::from_le_bytes([source[2], source[3]]),
        u16::from_le_bytes([source[4], source[5]]),
        u16::from_le_bytes([source[6], source[7]]),
    ];
    f32x4::from_slice(simd, &bonsai_neon_dot::widen_f16_four(&bits))
}

// Interleave four ordinary floating-point rows, retaining one output per SIMD
// lane. Each lane still adds its products from left to right without FMA, so
// this removes the scalar cross-row gather bottleneck without changing its bits.
#[inline(always)]
pub(super) fn rows<S: Simd>(
    simd: S,
    kind: u32,
    source: &[u8],
    input: &[f32],
    destination: &mut [f32],
) {
    let Ok((block, size)) = crate::weights::packed_layout(kind) else {
        destination.fill(f32::NAN);
        return;
    };
    let row_bytes = input.len() / block * size;
    let mut processed = 0;
    if matches!(kind, 0 | 1) {
        let col_chunks = input.len() / 4;
        let tail_start = col_chunks * 4;
        for output in destination.as_chunks_mut::<4>().0 {
            let start = processed * row_bytes;
            let r0 = &source[start..start + row_bytes];
            let r1 = &source[start + row_bytes..start + 2 * row_bytes];
            let r2 = &source[start + 2 * row_bytes..start + 3 * row_bytes];
            let r3 = &source[start + 3 * row_bytes..start + 4 * row_bytes];
            let mut sum = f32x4::from_slice(simd, output);
            for chunk_idx in 0..col_chunks {
                let col_base = chunk_idx * 4;
                let b_base = col_base * size;
                let row_vecs = if kind == 0 {
                    [
                        load_f32x4_le(simd, &r0[b_base..b_base + 16]),
                        load_f32x4_le(simd, &r1[b_base..b_base + 16]),
                        load_f32x4_le(simd, &r2[b_base..b_base + 16]),
                        load_f32x4_le(simd, &r3[b_base..b_base + 16]),
                    ]
                } else {
                    [
                        load_f16x4(simd, &r0[b_base..b_base + 8]),
                        load_f16x4(simd, &r1[b_base..b_base + 8]),
                        load_f16x4(simd, &r2[b_base..b_base + 8]),
                        load_f16x4(simd, &r3[b_base..b_base + 8]),
                    ]
                };
                let cols = transpose_4x4(simd, row_vecs);
                let in4 = &input[col_base..col_base + 4];
                for k in 0..4 {
                    let products = cols[k] * f32x4::splat(simd, in4[k]);
                    sum += products;
                }
            }
            for (tail_idx, &value) in input[tail_start..].iter().enumerate() {
                let b_off = (tail_start + tail_idx) * size;
                let weights = if kind == 0 {
                    f32x4::from_slice(
                        simd,
                        &[
                            read_f32_le(&r0[b_off..b_off + 4]),
                            read_f32_le(&r1[b_off..b_off + 4]),
                            read_f32_le(&r2[b_off..b_off + 4]),
                            read_f32_le(&r3[b_off..b_off + 4]),
                        ],
                    )
                } else {
                    f32x4::from_slice(
                        simd,
                        &[
                            half(&r0[b_off..]),
                            half(&r1[b_off..]),
                            half(&r2[b_off..]),
                            half(&r3[b_off..]),
                        ],
                    )
                };
                let products = weights * f32x4::splat(simd, value);
                sum += products;
            }
            sum.store_slice(output);
            processed += 4;
        }
    } else if matches!(kind, 8 | 12 | 14) {
        let blocks = input.len() / block;
        for output in destination.as_chunks_mut::<4>().0 {
            let row0 = processed * row_bytes;
            for block_index in 0..blocks {
                let source_offset = block_index * size;
                let input_offset = block_index * block;
                let row_slices = [
                    &source[row0 + source_offset..row0 + source_offset + size],
                    &source
                        [row0 + row_bytes + source_offset..row0 + row_bytes + source_offset + size],
                    &source[row0 + 2 * row_bytes + source_offset
                        ..row0 + 2 * row_bytes + source_offset + size],
                    &source[row0 + 3 * row_bytes + source_offset
                        ..row0 + 3 * row_bytes + source_offset + size],
                ];
                let values = &input[input_offset..input_offset + block];
                let block_sums = match kind {
                    8 => q8_rows(simd, row_slices, values),
                    12 => q4_k_rows(simd, row_slices, values),
                    _ => q6_k_rows(simd, row_slices, values),
                };
                for (dst, add) in output.iter_mut().zip(block_sums) {
                    *dst += add;
                }
            }
            processed += 4;
        }
    }
    for (source, output) in source[processed * row_bytes..]
        .chunks_exact(row_bytes)
        .zip(&mut destination[processed..])
    {
        *output = row(simd, kind, source, input, *output);
    }
}

#[inline(always)]
fn quant_row_sum<const S: usize, const B: usize>(
    source: &[u8],
    input: &[f32],
    mut sum: f32,
    mut dot: impl FnMut(&[u8; S], &[f32; B]) -> f32,
) -> f32 {
    for (bytes, values) in source
        .as_chunks::<S>()
        .0
        .iter()
        .zip(input.as_chunks::<B>().0)
    {
        sum += dot(bytes, values);
    }
    sum
}

// Validated callers supply complete rows of a supported type. F32/F16 preserve
// their original separately rounded multiplication and left-to-right summation.
#[inline(always)]
pub(super) fn row<S: Simd>(simd: S, kind: u32, source: &[u8], input: &[f32], mut sum: f32) -> f32 {
    match kind {
        0 | 1 => {
            let size = if kind == 0 { 4 } else { 2 };
            for (sources, inputs) in source.chunks_exact(size * 4).zip(input.as_chunks::<4>().0) {
                let values = f32x4::from_fn(simd, |index| {
                    let offset = index * size;
                    if kind == 0 {
                        read_f32_le(&sources[offset..offset + 4])
                    } else {
                        half(&sources[offset..])
                    }
                });
                let products: [f32; 4] = (values * f32x4::from_slice(simd, inputs)).into();
                for product in products {
                    sum += product;
                }
            }
            let tail = input.len() / 4 * 4;
            for (bytes, value) in source[tail * size..].chunks_exact(size).zip(&input[tail..]) {
                let weight = if kind == 0 {
                    read_f32_le(bytes)
                } else {
                    half(bytes)
                };
                sum += weight * value;
            }
            sum
        }
        8 => quant_row_sum::<34, 32>(source, input, sum, |b, v| q8_rows(simd, [b], v)[0]),
        12 => quant_row_sum::<144, 256>(source, input, sum, |b, v| q4_k_rows(simd, [b], v)[0]),
        14 => quant_row_sum::<210, 256>(source, input, sum, |b, v| q6_k_rows(simd, [b], v)[0]),
        _ => f32::NAN,
    }
}

#[cfg(test)]
mod tests {
    use std::io;

    use fearless_simd::{Level, dispatch};

    use crate::{
        engine::{LinearProjection, WeightQuantization, f32_dot_simd},
        weights::{decode_block, packed_layout},
    };

    #[test]
    fn packed_half_precision_preserves_f32_operands_and_accumulation() -> io::Result<()> {
        // Literal answers distinguish F32 activations/accumulators from F16
        // narrowing. Five rows exercise the four-row path and its scalar tail;
        // the last case also catches an accumulator narrowed between columns.
        let cases = [
            (vec![65_536.0_f32], 65_536.0_f32),
            (vec![1.000_488_3], 1.000_488_3),
            (
                vec![f32::from_bits(0x3300_0000)],
                f32::from_bits(0x3300_0000),
            ),
            (vec![8.0; 8193], 65_544.0),
            (vec![16_777_216.0, 1.0, -16_777_216.0, 1.0], 1.0),
        ];
        for (input, expected) in cases {
            let projection = LinearProjection::new(
                "packed-half-precision",
                input.len(),
                5,
                WeightQuantization::Packed {
                    ggml_type: 1,
                    bytes: [0x00, 0x3c].repeat(input.len() * 5),
                },
                None,
            )?;
            for workers in [1, 2] {
                let actual = projection.forward_vec_parallel(&input, None, workers)?;
                assert_eq!(actual.len(), 5);
                for value in actual {
                    assert_eq!(value.to_bits(), expected.to_bits());
                }
            }
        }
        Ok(())
    }

    #[test]
    fn packed_float_row_tiles_preserve_independent_sequential_rounding() -> io::Result<()> {
        let half_rows = [
            [0x3c00_u16, 0x3c00, 0x3c00, 0x3c00, 1, 0x3555, 0xb955],
            [0xbc00, 0x8000, 0x3c00, 0x3c01, 0x0400, 0x8001, 0],
            [0x3555, 0xb955, 0x3c01, 0xbc01, 0x8000, 1, 0x0400],
        ];
        for kind in [0, 1] {
            for row_count in [1, 3, 4, 5, 8, 19] {
                let mut bytes = Vec::new();
                let mut rows = Vec::new();
                for index in 0..row_count {
                    let mut weights = Vec::new();
                    for bits in half_rows[index % half_rows.len()] {
                        let value = crate::weights::half(&bits.to_le_bytes());
                        weights.push(value);
                        if kind == 0 {
                            bytes.extend_from_slice(&value.to_le_bytes());
                        } else {
                            bytes.extend_from_slice(&bits.to_le_bytes());
                        }
                    }
                    rows.push(weights);
                }
                let bias: Vec<f32> = (0..row_count)
                    .map(|index| [-0.0, 0.0, 0.75][index % 3])
                    .collect();
                let projection = LinearProjection::new(
                    "packed-independent-rows",
                    7,
                    row_count,
                    WeightQuantization::Packed {
                        ggml_type: kind,
                        bytes,
                    },
                    Some(bias.clone()),
                )?;
                for input in [
                    [16_777_216.0, 1.0, -16_777_216.0, 1.0, -1.0, 0.0, -0.0],
                    [-0.0; 7],
                    [0.0; 7],
                    [1.000_976_6, -1.25, 0.7, -0.3, 0.125, -0.0625, 0.03125],
                ] {
                    let expected: Vec<u32> = rows
                        .iter()
                        .zip(&bias)
                        .map(|(weights, bias)| {
                            weights
                                .iter()
                                .zip(input)
                                .fold(*bias, |sum, (weight, value)| sum + weight * value)
                                .to_bits()
                        })
                        .collect();
                    for workers in [1, 3, 18] {
                        let actual = projection.forward_vec_parallel(&input, None, workers)?;
                        assert_eq!(
                            actual
                                .iter()
                                .map(|value| value.to_bits())
                                .collect::<Vec<_>>(),
                            expected,
                            "GGUF type {kind}, rows {row_count}, workers {workers}",
                        );
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn packed_direct_kernels_match_decoded_block_rounding_and_workers() -> io::Result<()> {
        for kind in [0, 1, 8, 12, 14] {
            let (block, size) = packed_layout(kind)?;
            let width = if block == 1 { 19 } else { block * 3 };
            let rows = 7;
            let mut bytes: Vec<u8> = (0..width / block * size * rows)
                .map(|index| u8::try_from((index * 73 + 31) % 256).unwrap_or(0))
                .collect();
            for (index, source) in bytes.chunks_exact_mut(size).enumerate() {
                let scale = [0x3555_u16, 0xb955, 1, 0x8000][index % 4];
                match kind {
                    0 => source.copy_from_slice(
                        &(f32::from(i16::try_from(index % 97).unwrap_or(0)) * 0.09375 - 4.0)
                            .to_le_bytes(),
                    ),
                    1 | 8 => source[..2].copy_from_slice(&scale.to_le_bytes()),
                    12 => {
                        source[..2].copy_from_slice(&scale.to_le_bytes());
                        source[2..4].copy_from_slice(&0x2c01_u16.to_le_bytes());
                    }
                    14 => source[208..].copy_from_slice(&scale.to_le_bytes()),
                    _ => return Err(io::Error::other("Unsupported fixture type.")),
                }
            }
            let input: Vec<f32> = (0..width)
                .map(|index| (f32::from(i16::try_from(index % 37).unwrap_or(0)) - 18.0) * 0.071)
                .collect();
            let bias = vec![-0.75; rows];
            let projection = LinearProjection::new(
                "direct-packed-fixture",
                width,
                rows,
                WeightQuantization::Packed {
                    ggml_type: kind,
                    bytes: bytes.clone(),
                },
                Some(bias.clone()),
            )?;
            let level = Level::new();
            let mut expected = bias;
            let mut decoded = [0.0; 256];
            for (source, output) in bytes.chunks_exact(width / block * size).zip(&mut expected) {
                for (source, values) in source.chunks_exact(size).zip(input.chunks_exact(block)) {
                    decode_block(kind, source, &mut decoded[..block]);
                    *output += if block == 1 {
                        decoded[0] * values[0]
                    } else {
                        dispatch!(level, simd => f32_dot_simd(simd, &decoded[..block], values))
                    };
                }
            }
            let mut baseline_bits = None;
            for workers in [1, 3, 18] {
                let actual = projection.forward_vec_parallel(&input, None, workers)?;
                let actual_bits: Vec<u32> = actual.iter().map(|value| value.to_bits()).collect();
                if matches!(kind, 0 | 1) || cfg!(not(windows)) {
                    assert_eq!(
                        actual_bits,
                        expected
                            .iter()
                            .map(|value| value.to_bits())
                            .collect::<Vec<_>>(),
                        "GGUF type {kind}, workers {workers}",
                    );
                } else {
                    if let Some(prev) = &baseline_bits {
                        assert_eq!(
                            &actual_bits, prev,
                            "worker invariance GGUF type {kind}, workers {workers}"
                        );
                    } else {
                        baseline_bits = Some(actual_bits);
                    }
                    for (act, exp) in actual.iter().zip(&expected) {
                        let tol = 0.08 * exp.abs().max(1.0);
                        assert!(
                            (*act - *exp).abs() <= tol,
                            "GGUF type {kind}, workers {workers}: actual={act} expected={exp}",
                        );
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn packed_direct_float_and_half_keep_signed_zero_subnormals_and_sequential_sums()
    -> io::Result<()> {
        let half_bits = [0x3c00_u16, 0x8000, 1, 0xbc00, 0x0400, 0x3555, 0xb955];
        for kind in [0, 1] {
            let mut bytes = Vec::new();
            let mut weights = Vec::new();
            for bits in half_bits {
                let value = crate::weights::half(&bits.to_le_bytes());
                weights.push(value);
                if kind == 0 {
                    bytes.extend_from_slice(&value.to_le_bytes());
                } else {
                    bytes.extend_from_slice(&bits.to_le_bytes());
                }
            }
            for input in [
                [16_777_216.0, 1.0, -16_777_216.0, 1.0, -1.0, 0.0, -0.0],
                [-0.0; 7],
                [0.0; 7],
                [1.0, -1.0, 0.5, -0.25, 0.125, -0.0625, 0.03125],
            ] {
                for bias in [-0.0, 0.0, 0.75] {
                    let projection = LinearProjection::new(
                        "packed-float-rounding-fixture",
                        weights.len(),
                        1,
                        WeightQuantization::Packed {
                            ggml_type: kind,
                            bytes: bytes.clone(),
                        },
                        Some(vec![bias]),
                    )?;
                    let expected = weights
                        .iter()
                        .zip(input)
                        .fold(bias, |sum, (weight, value)| sum + weight * value);
                    assert_eq!(
                        projection.forward_vec(&input, None)?[0].to_bits(),
                        expected.to_bits(),
                        "GGUF type {kind}",
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn packed_four_row_quantized_tiles_match_per_row_bits_across_tails_and_bias() -> io::Result<()>
    {
        let level = Level::new();
        let initial_pool = [-0.0_f32, 0.0, -0.75, 1.25, 0.5, -0.25, 2.5, -1.5, 0.125];
        for kind in [8, 12, 14] {
            let (block, size) = packed_layout(kind)?;
            let width = block * 3;
            let row_bytes = width / block * size;
            let input: Vec<f32> = (0..width)
                .map(|index| (f32::from(i16::try_from(index % 41).unwrap_or(0)) - 20.0) * 0.0625)
                .collect();
            for row_count in 1..=9 {
                let mut bytes: Vec<u8> = (0..row_bytes * row_count)
                    .map(|index| {
                        u8::try_from((index * 67 + 19 + row_count * 11) % 256).unwrap_or(0)
                    })
                    .collect();
                for (index, source) in bytes.chunks_exact_mut(size).enumerate() {
                    let scale = [0x3555_u16, 0xb955, 0x3c00, 0x8000][index % 4];
                    match kind {
                        8 => source[..2].copy_from_slice(&scale.to_le_bytes()),
                        12 => {
                            source[..2].copy_from_slice(&scale.to_le_bytes());
                            source[2..4].copy_from_slice(&0x2c01_u16.to_le_bytes());
                        }
                        14 => source[208..].copy_from_slice(&scale.to_le_bytes()),
                        _ => return Err(io::Error::other("Unsupported quantized test type.")),
                    }
                }
                let mut actual = initial_pool[..row_count].to_vec();
                dispatch!(level, simd => super::rows(simd, kind, &bytes, &input, &mut actual));
                let mut expected = initial_pool[..row_count].to_vec();
                for (row_slice, slot) in bytes.chunks_exact(row_bytes).zip(&mut expected) {
                    *slot =
                        dispatch!(level, simd => super::row(simd, kind, row_slice, &input, *slot));
                }
                assert_eq!(
                    actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "GGUF type {kind}, rows {row_count}",
                );
            }
        }
        Ok(())
    }

    #[test]
    fn packed_four_row_float_and_half_tiles_match_per_row_bits_across_widths() -> io::Result<()> {
        let level = Level::new();
        let initial_pool = [-0.0_f32, 0.0, -0.75, 1.25, 0.5, -0.25, 2.5, -1.5, 0.125];
        let half_bits = [0x3555_u16, 0xb955, 0x3c00, 0x8000, 0x3c01, 0xbc00, 0x0400];
        for kind in [0_u32, 1] {
            let size = if kind == 0 { 4 } else { 2 };
            for width in [1_usize, 4, 7, 16, 35] {
                let row_bytes = width * size;
                let input: Vec<f32> = (0..width)
                    .map(|idx| {
                        (f32::from(i16::try_from((idx * 5) % 37).unwrap_or(0)) - 18.0) * 0.0625
                    })
                    .collect();
                for row_count in [1_usize, 4, 5, 8, 9] {
                    let mut bytes = Vec::with_capacity(row_count * row_bytes);
                    for row_idx in 0..row_count {
                        for col_idx in 0..width {
                            let bits = half_bits[(row_idx * 3 + col_idx * 5) % half_bits.len()];
                            if kind == 0 {
                                let val = crate::weights::half(&bits.to_le_bytes());
                                bytes.extend_from_slice(&val.to_le_bytes());
                            } else {
                                bytes.extend_from_slice(&bits.to_le_bytes());
                            }
                        }
                    }
                    let mut actual = initial_pool[..row_count].to_vec();
                    dispatch!(level, simd => super::rows(simd, kind, &bytes, &input, &mut actual));
                    let mut expected = initial_pool[..row_count].to_vec();
                    for (row_slice, slot) in bytes.chunks_exact(row_bytes).zip(&mut expected) {
                        *slot = dispatch!(level, simd => super::row(simd, kind, row_slice, &input, *slot));
                    }
                    assert_eq!(
                        actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        "GGUF type {kind}, width {width}, rows {row_count}",
                    );
                }
            }
        }
        Ok(())
    }
}
