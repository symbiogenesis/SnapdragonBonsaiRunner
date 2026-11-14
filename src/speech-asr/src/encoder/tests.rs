//! Hand-calculated convolution, normalization, attention, sinusoid and tail controls.

use super::{
    Block, Config, Encoder, Linear, Norm, Projection, Weights, attention, convolve, numerics,
    position_add,
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
