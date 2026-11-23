//! Hand-calculated convolution, normalization, attention, sinusoid and tail controls.

use super::{
    Block, Config, Encoder, Linear, Norm, Projection, Weights, attention, convolve,
    convolve_with_threads, numerics, position_add,
};
use std::io;

fn zero(input: usize, output: usize) -> io::Result<Linear> {
    Ok(Linear {
        weights: Projection::f32(input, output, vec![0.0; input * output])?,
        bias: vec![0.0; output],
    })
}
fn norm(width: usize) -> Norm {
    Norm {
        weight: vec![1.0; width],
        bias: vec![0.0; width],
    }
}

pub(super) fn fixture() -> io::Result<Encoder> {
    let c = Config {
        mel_bins: 1,
        channels: 1,
        hidden: 4,
        heads: 1,
        intermediate: 4,
        output: 2,
        window: 5,
        max_positions: 2,
    };
    let block = Block {
        attention_norm: norm(4),
        ffn_norm: norm(4),
        query: zero(4, 4)?,
        key: zero(4, 4)?,
        value: zero(4, 4)?,
        attention_out: zero(4, 4)?,
        ffn_in: zero(4, 4)?,
        ffn_out: zero(4, 4)?,
    };
    let mut output_out = zero(4, 2)?;
    output_out.bias = vec![2.0, -3.0];
    Encoder::new(
        c,
        Weights {
            convolutions: [zero(9, 1)?, zero(9, 1)?, zero(9, 1)?],
            convolution_out: Projection::f32(1, 4, vec![0.0; 4])?,
            blocks: vec![block],
            output_norm: norm(4),
            output_in: zero(4, 4)?,
            output_out,
        },
    )
}

#[test]
fn rust_asr_encoder_erf_gelu_known_answers_and_signed_zero() {
    for (input, expected) in [
        (-3.0, -0.004_049_694),
        (-1.0, -0.158_655_26),
        (0.0, 0.0),
        (1.0, 0.841_344_7),
        (3.0, 2.995_950_2),
        (20.0, 20.0),
    ] {
        assert!(
            (numerics::gelu(input) - expected).abs() < 0.000_000_3,
            "{input}"
        );
    }
    assert_eq!(numerics::gelu(-0.0).to_bits(), (-0.0_f32).to_bits());
}

#[test]
fn rust_asr_encoder_stride_padding_kernel_order_and_finite_output() -> io::Result<()> {
    // A 3x3 all-one kernel over [1..9], stride two and pad one,
    // sums four corners: 12,16,24,28. GELU rounds these back exactly.
    let kernel = Linear {
        weights: Projection::f32(9, 1, vec![1.0; 9])?,
        bias: vec![0.0],
    };
    assert_eq!(
        convolve(
            &kernel,
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            3,
            3,
            1,
            &mut || false
        )?,
        [12.0, 16.0, 24.0, 28.0]
    );
    let kernel = Linear {
        weights: Projection::f32(9, 1, vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])?,
        bias: vec![0.0],
    };
    let output = convolve(
        &kernel,
        &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0],
        3,
        3,
        1,
        &mut || false,
    )?;
    assert!((output[0] - 0.841_344_7).abs() < 0.000_000_1);
    assert!((output[3] - 2.995_950_2).abs() < 0.000_000_3);
    assert_eq!(&output[1..3], &[0.0; 2]);
    Ok(())
}

#[test]
fn rust_asr_encoder_convolution_half_operands_and_every_stage_narrowing() -> io::Result<()> {
    let Encoder {
        config,
        mut weights,
        ..
    } = fixture()?;
    let mut kernel = vec![0.0; 9];
    // Binary16 ties: 16 + 2^-7 rounds down to even 16, whereas
    // 1 + 3*2^-11 rounds up to even 1 + 2^-9. The exact product
    // 16*(1 + 2^-9) = 16.03125 is large enough that GELU rounds to x.
    kernel[4] = 16.0 + 1.0 / 128.0;
    weights.convolutions[0].weights = Projection::f32(9, 1, kernel)?;
    let encoder = Encoder::new(config, weights)?;
    assert_eq!(
        convolve(
            &encoder.weights.convolutions[0],
            &[1.0 + 3.0 / 2048.0],
            1,
            1,
            1,
            &mut || false,
        )?,
        [16.03125]
    );

    let mut kernel = vec![0.0; 9];
    kernel[4] = 1.0;
    let first = Linear {
        weights: Projection::f32(9, 1, kernel.clone())?,
        bias: vec![0.0],
    };
    kernel[4] = 16.0;
    let second = Linear {
        weights: Projection::f32(9, 1, kernel)?,
        bias: vec![0.0],
    };
    let first_output = convolve(&first, &[1.0], 1, 1, 1, &mut || false)?;
    assert!((first_output[0] - 0.841_344_7).abs() < 0.000_000_1);
    // GELU(1) narrows independently to half 0x3abb = 0.84130859375
    // before the next stage. Its exact product with 16 is 13.4609375.
    assert_eq!(
        convolve(&second, &first_output, 1, 1, 1, &mut || false)?,
        [13.4609375]
    );
    Ok(())
}

#[test]
fn rust_asr_encoder_convolution_half_boundaries_and_finite_overflow() -> io::Result<()> {
    for (input, expected) in [
        (-0.0_f32, -0.0_f32),
        (1.0 / 33_554_432.0, 0.0),
        (3.0 / 33_554_432.0, 1.0 / 8_388_608.0),
        (-3.0 / 33_554_432.0, -1.0 / 8_388_608.0),
        (65_504.0, 65_504.0),
        (-65_504.0, -65_504.0),
    ] {
        assert_eq!(
            super::binary16_operand(input)?.to_bits(),
            expected.to_bits()
        );
    }
    for input in [65_520.0, -65_520.0, f32::INFINITY, f32::NAN] {
        assert!(super::binary16_operand(input).is_err());
    }
    let Encoder {
        config,
        mut weights,
        ..
    } = fixture()?;
    let mut kernel = vec![0.0; 9];
    kernel[4] = 65_520.0;
    weights.convolutions[1].weights = Projection::f32(9, 1, kernel)?;
    assert!(Encoder::new(config, weights).is_err());
    let fresh = fixture()?;
    assert!(
        convolve(
            &fresh.weights.convolutions[0],
            &[65_520.0],
            1,
            1,
            1,
            &mut || false,
        )
        .is_err()
    );
    assert_eq!(
        convolve(&fresh.weights.convolutions[0], &[1.0], 1, 1, 1, &mut || {
            false
        },)?,
        [0.0]
    );
    Ok(())
}

#[test]
fn rust_asr_encoder_layer_norm_bias_and_bidirectional_attention() -> io::Result<()> {
    let norm = Norm {
        weight: vec![2.0, 3.0],
        bias: vec![4.0, 5.0],
    };
    let output = norm.apply(&[1.0, 3.0])?;
    assert!((output[0] - 2.000_01).abs() < 0.000_001);
    assert!((output[1] - 7.999_985).abs() < 0.000_001);
    let keys = vec![vec![0.0; 4]; 2];
    let values = vec![vec![2.0, 4.0, 6.0, 8.0], vec![10.0, 12.0, 14.0, 16.0]];
    assert_eq!(
        attention(&[0.0; 4], &keys, &values, 2, &mut || false)?,
        [6.0, 8.0, 10.0, 12.0]
    );
    Ok(())
}

#[test]
fn rust_asr_encoder_sinusoid_chunk_positions() -> io::Result<()> {
    let mut zero = [0.0; 4];
    position_add(&mut zero, 0)?;
    assert_eq!(zero, [0.0, 0.0, 1.0, 1.0]);
    let mut one = [0.0; 4];
    position_add(&mut one, 1)?;
    for (actual, expected) in one.iter().zip([0.841_470_96, 0.000_1, 0.540_302_3, 1.0]) {
        assert!((*actual - expected).abs() <= 0.000_000_1);
    }
    Ok(())
}

#[test]
fn rust_asr_encoder_full_graph_independent_chunk_tail_rows() -> io::Result<()> {
    let encoder = fixture()?;
    // 10 mel rows per chunk -> 2 encoder rows, then one real tail row.
    // ceil(11/8)=2 would lose the independently padded chunk's third row.
    let output = encoder.encode_mel(&[0.0; 11], 11, &mut || false)?;
    assert_eq!(output.rows(), 3);
    assert_eq!(output.width(), 2);
    assert_eq!(output.values(), &[2.0, -3.0, 2.0, -3.0, 2.0, -3.0]);
    assert_eq!(encoder.encode_mel(&[0.0; 10], 10, &mut || false)?.rows(), 2);
    Ok(())
}

#[test]
fn rust_asr_encoder_failure_cancellation_and_replay() -> io::Result<()> {
    let encoder = fixture()?;
    assert!(encoder.encode_mel(&[f32::NAN], 1, &mut || false).is_err());
    assert!(encoder.encode_mel(&[], 0, &mut || false).is_err());
    assert!(encoder.encode_mel(&[0.0; 11], 10, &mut || false).is_err());
    let mut calls = 0;
    let result = encoder.encode_mel(&[0.0; 11], 11, &mut || {
        calls += 1;
        calls > 30
    });
    assert_eq!(
        result.err().map(|error| error.kind()),
        Some(io::ErrorKind::Interrupted)
    );
    assert_eq!(
        encoder.encode_mel(&[0.0], 1, &mut || false)?.values(),
        &[2.0, -3.0]
    );
    Ok(())
}

#[test]
fn rust_asr_encoder_batched_parallel_matches_scalar_bits() -> io::Result<()> {
    fn f32_linear(input: usize, output: usize, seed: usize) -> io::Result<Linear> {
        let mut weights = Vec::with_capacity(input * output);
        let denom = f32::from(u16::try_from(input).unwrap_or(1));
        for row in 0..output {
            for col in 0..input {
                let v = ((row * 17 + col * 23 + seed) % 15).cast_signed() - 7;
                weights.push(f32::from(i8::try_from(v).unwrap_or(0)) / (4.0 * denom));
            }
        }
        let bias = (0..output)
            .map(|i| {
                let v = ((i * 11 + seed) % 9).cast_signed() - 4;
                f32::from(i8::try_from(v).unwrap_or(0)) * 0.03125
            })
            .collect();
        Ok(Linear {
            weights: Projection::f32(input, output, weights)?,
            bias,
        })
    }

    fn q8_linear(input: usize, output: usize, seed: usize) -> io::Result<Linear> {
        let blocks_per_row = input / 32;
        let mut bytes = Vec::with_capacity(output * blocks_per_row * 34);
        let scales = [0x3800_u16, 0x3400, 0x3C00, 0xB600];
        for row in 0..output {
            for blk in 0..blocks_per_row {
                bytes.extend_from_slice(&scales[(row + blk + seed) % scales.len()].to_le_bytes());
                for elem in 0..32_u8 {
                    let v = ((row * 13 + blk * 29 + usize::from(elem) * 7 + seed) % 19)
                        .cast_signed()
                        - 9;
                    bytes.push(i8::try_from(v).unwrap_or(0).cast_unsigned());
                }
            }
        }
        let bias = (0..output)
            .map(|i| {
                let v = ((i * 7 + seed) % 11).cast_signed() - 5;
                f32::from(i8::try_from(v).unwrap_or(0)) * 0.02
            })
            .collect();
        Ok(Linear {
            weights: Projection::q8(input, output, bytes)?,
            bias,
        })
    }

    let c = Config {
        mel_bins: 8,
        channels: 4,
        hidden: 32,
        heads: 4,
        intermediate: 64,
        output: 16,
        window: 5,
        max_positions: 2,
    };
    let build = |threads: usize| -> io::Result<Encoder> {
        let mut blocks = Vec::with_capacity(2);
        for layer in 0..2 {
            let norm_at = |offset: usize| Norm {
                weight: (0..c.hidden)
                    .map(|i| 0.85 + f32::from(u8::try_from((i + offset) % 7).unwrap_or(0)) * 0.04)
                    .collect(),
                bias: (0..c.hidden)
                    .map(|i| {
                        f32::from(i8::try_from(((i + offset) % 5).cast_signed() - 2).unwrap_or(0))
                            * 0.02
                    })
                    .collect(),
            };
            blocks.push(Block {
                attention_norm: norm_at(1 + layer),
                ffn_norm: norm_at(3 + layer),
                query: q8_linear(c.hidden, c.hidden, 10 + layer)?,
                key: f32_linear(c.hidden, c.hidden, 20 + layer)?,
                value: q8_linear(c.hidden, c.hidden, 30 + layer)?,
                attention_out: f32_linear(c.hidden, c.hidden, 40 + layer)?,
                ffn_in: q8_linear(c.hidden, c.intermediate, 50 + layer)?,
                ffn_out: q8_linear(c.intermediate, c.hidden, 60 + layer)?,
            });
        }
        Encoder::new(
            c,
            Weights {
                convolutions: [
                    f32_linear(9, c.channels, 1)?,
                    f32_linear(9 * c.channels, c.channels, 2)?,
                    f32_linear(9 * c.channels, c.channels, 3)?,
                ],
                convolution_out: Projection::f32(
                    c.channels,
                    c.hidden,
                    (0..c.channels * c.hidden)
                        .map(|i| {
                            f32::from(i8::try_from((i % 7).cast_signed() - 3).unwrap_or(0)) * 0.05
                        })
                        .collect(),
                )?,
                blocks,
                output_norm: norm(c.hidden),
                output_in: q8_linear(c.hidden, c.hidden, 70)?,
                output_out: f32_linear(c.hidden, c.output, 80)?,
            },
        )?
        .with_threads(threads)
    };

    let frames = 19;
    let mut mel = Vec::with_capacity(c.mel_bins * frames);
    for idx in 0..(c.mel_bins * frames) {
        let v = ((idx * 13 + 5) % 21).cast_signed() - 10;
        mel.push(f32::from(i8::try_from(v).unwrap_or(0)) * 0.05);
    }

    let serial = build(1)?;
    let expected = serial.encode_mel(&mel, frames, &mut || false)?;
    for workers in [2, 4] {
        let parallel = build(workers)?;
        let actual = parallel.encode_mel(&mel, frames, &mut || false)?;
        assert_eq!(actual.rows(), expected.rows());
        assert_eq!(actual.width(), expected.width());
        for (idx, (got, want)) in actual.values().iter().zip(expected.values()).enumerate() {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "encoder bit mismatch for workers={workers}, element={idx}"
            );
        }
    }
    assert_eq!(
        serial.with_threads(0).err().map(|error| error.kind()),
        Some(io::ErrorKind::InvalidInput)
    );
    Ok(())
}

#[test]
fn rust_asr_encoder_parallel_conv_matches_sequential_bits() -> io::Result<()> {
    let width = 17_usize;
    let height = 13_usize;
    let channels = 4_usize;
    let out_channels = 9_usize;
    let patch_width = 9 * channels;
    let weight_vals: Vec<f32> = (0..patch_width * out_channels)
        .map(|i| (f32::from(u16::try_from(i % 29).unwrap_or(0)) - 14.0) * 0.03125)
        .collect();
    let bias: Vec<f32> = (0..out_channels)
        .map(|i| (f32::from(u8::try_from(i % 7).unwrap_or(0)) - 3.0) * 0.0625)
        .collect();
    let mut layer = Linear {
        weights: Projection::f32(patch_width, out_channels, weight_vals)?,
        bias,
    };
    layer.weights.round_convolution_weights()?;
    let input: Vec<f32> = (0..channels * height * width)
        .map(|i| (f32::from(u16::try_from(i % 37).unwrap_or(0)) - 18.0) * 0.05)
        .collect();
    let expected = convolve(&layer, &input, width, height, channels, &mut || false)?;
    for workers in [2_usize, 4, 8] {
        let actual = convolve_with_threads(
            &layer,
            &input,
            width,
            height,
            channels,
            workers,
            &mut || false,
        )?;
        assert_eq!(
            actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "conv bit mismatch for workers={workers}"
        );
    }

    let mut overflow_input = input;
    let mid = overflow_input.len() / 2;
    overflow_input[mid] = 65_520.0;
    assert!(
        convolve_with_threads(
            &layer,
            &overflow_input,
            width,
            height,
            channels,
            4,
            &mut || { false }
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn rust_asr_encoder_simd_gelu_and_windowed_attention_parity() -> io::Result<()> {
    let inputs = [
        -12.0_f32, -3.0, -1.0, -0.0, 0.0, 0.5, 1.0, 3.0, 7.5, 16.03125, 20.0,
    ];
    let mut vec_vals = inputs;
    numerics::gelu_slice_in_place(&mut vec_vals);
    for (got, x) in vec_vals.iter().zip(inputs) {
        assert_eq!(got.to_bits(), numerics::gelu(x).to_bits(), "x={x}");
    }
    assert_eq!(vec_vals[3].to_bits(), (-0.0_f32).to_bits());

    // Multi-window cu_seqlens test: window=1 -> mel_per_chunk=2, rows_per_chunk=1,
    // so window_rows=8.Perturbing mel frames after window 0 (frame >= 16) must leave
    // window 0 encoder rows (0..8) bit-identical across workers.
    let mut enc = fixture()?.with_threads(1)?;
    enc.config.window = 1;
    enc.config.max_positions = 16;
    enc.sinusoids = super::build_sinusoids(16, enc.config.hidden)?;
    let frames = 20_usize;
    let mel_a: Vec<f32> = (0..enc.config.mel_bins * frames)
        .map(|i| f32::from(i8::try_from((i % 13).cast_signed() - 6).unwrap_or(0)) * 0.05)
        .collect();
    let mut mel_b = mel_a.clone();
    for f in 16..frames {
        for b in 0..enc.config.mel_bins {
            mel_b[b * frames + f] += 0.35;
        }
    }
    let out_a = enc.encode_mel(&mel_a, frames, &mut || false)?;
    let out_b = enc.encode_mel(&mel_b, frames, &mut || false)?;
    assert_eq!(out_a.rows(), 10);
    let win0_elems = 8 * out_a.width();
    assert_eq!(
        out_a.values()[..win0_elems]
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        out_b.values()[..win0_elems]
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );
    let enc4 = enc.with_threads(4)?;
    let out_par = enc4.encode_mel(&mel_a, frames, &mut || false)?;
    assert_eq!(
        out_par
            .values()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        out_a
            .values()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );
    Ok(())
}

#[cfg(all(windows, feature = "rust-qualification"))]
#[test]
fn rust_asr_encoder_repacks_aligned_q4_k_projections() -> io::Result<()> {
    fn make_q4_k_linear(input: usize, output: usize, seed: usize) -> io::Result<Linear> {
        let blocks_per_row = input / 256;
        let mut bytes = vec![0_u8; output * blocks_per_row * 144];
        for (blk_idx, block) in bytes.chunks_exact_mut(144).enumerate() {
            block[0..2].copy_from_slice(&0x3000_u16.to_le_bytes());
            block[2..4].copy_from_slice(&0x2800_u16.to_le_bytes());
            for (i, byte) in block[4..16].iter_mut().enumerate() {
                *byte = u8::try_from((blk_idx + seed + i) % 15 + 1).unwrap_or(1);
            }
            for (i, byte) in block[16..].iter_mut().enumerate() {
                *byte = u8::try_from((blk_idx * 7 + seed + i * 3) % 256).unwrap_or(0);
            }
        }
        Ok(Linear {
            weights: Projection::q4_k(input, output, bytes)?,
            bias: vec![0.05_f32; output],
        })
    }

    let c = Config {
        mel_bins: 8,
        channels: 256,
        hidden: 256,
        heads: 4,
        intermediate: 256,
        output: 256,
        window: 5,
        max_positions: 2,
    };
    let conv_out = make_q4_k_linear(c.channels, c.hidden, 1)?;
    let block = Block {
        attention_norm: norm(c.hidden),
        ffn_norm: norm(c.hidden),
        query: make_q4_k_linear(c.hidden, c.hidden, 2)?,
        key: make_q4_k_linear(c.hidden, c.hidden, 3)?,
        value: make_q4_k_linear(c.hidden, c.hidden, 4)?,
        attention_out: make_q4_k_linear(c.hidden, c.hidden, 5)?,
        ffn_in: make_q4_k_linear(c.hidden, c.intermediate, 6)?,
        ffn_out: make_q4_k_linear(c.intermediate, c.hidden, 7)?,
    };
    let enc = Encoder::new(
        c,
        Weights {
            convolutions: [
                zero(9, c.channels)?,
                zero(9 * c.channels, c.channels)?,
                zero(9 * c.channels, c.channels)?,
            ],
            convolution_out: conv_out.weights,
            blocks: vec![block],
            output_norm: norm(c.hidden),
            output_in: make_q4_k_linear(c.hidden, c.hidden, 8)?,
            output_out: make_q4_k_linear(c.hidden, c.output, 9)?,
        },
    )?
    .with_threads(2)?;

    let expected_repack = bonsai_windows_process::quant_kernels::Arm64QuantCaps::detect().dotprod;
    assert_eq!(
        enc.weights.convolution_out.is_repacked_q4_k(),
        expected_repack
    );
    let b = &enc.weights.blocks[0];
    for proj in [
        &b.query.weights,
        &b.key.weights,
        &b.value.weights,
        &b.attention_out.weights,
        &b.ffn_in.weights,
        &b.ffn_out.weights,
        &enc.weights.output_in.weights,
        &enc.weights.output_out.weights,
    ] {
        assert_eq!(proj.is_repacked_q4_k(), expected_repack);
    }

    let out = enc.encode_mel(&vec![0.0_f32; c.mel_bins * 10], 10, &mut || false)?;
    assert_eq!(out.rows(), 2);
    assert_eq!(out.width(), c.output);
    Ok(())
}
