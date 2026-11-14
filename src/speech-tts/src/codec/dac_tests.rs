//! Independent finite-impulse answers for the complete four-block DAC topology.

use std::io;

use super::{
    CausalConv1d, CausalTransConv1d, CodecStreamState, DacBlock, DacDecoder, DacResUnit, PadMode,
    SnakeBeta,
};

fn identity_snake() -> SnakeBeta {
    // A zero alpha exponential is the finite-log underflow limit. Its sine
    // correction is exactly zero, allowing a closed-form linear graph oracle.
    SnakeBeta {
        a: vec![0.0],
        inv_b: vec![1.0],
    }
}

fn current_convolution(gain: f32, dilation: usize) -> CausalConv1d {
    CausalConv1d {
        out_c: 1,
        in_c: 1,
        kernel: 7,
        dilation,
        stride: 1,
        pad_mode: PadMode::Constant,
        weight: vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gain],
        bias: None,
    }
}

fn residual(dilation: usize) -> DacResUnit {
    DacResUnit {
        act1: identity_snake(),
        conv1: current_convolution(1.0, dilation),
        act2: identity_snake(),
        conv2: CausalConv1d {
            kernel: 1,
            weight: vec![1.0],
            ..current_convolution(1.0, 1)
        },
    }
}

fn decoder() -> DacDecoder {
    DacDecoder {
        conv_pre: current_convolution(1.0 / 16.0, 1),
        blocks: [8, 5, 4, 3]
            .into_iter()
            .map(|stride| DacBlock {
                snake1: identity_snake(),
                trans_conv: CausalTransConv1d {
                    in_c: 1,
                    out_c: 1,
                    kernel: 2 * stride,
                    stride,
                    weight: vec![1.0 / 8.0; 2 * stride],
                    bias: None,
                },
                // All three residual branches execute and double the signal,
                // cancelling this block's 1/8 transposed-convolution scale.
                res_units: [residual(1), residual(3), residual(9)],
            })
            .collect(),
        snake_post: identity_snake(),
        conv_post: current_convolution(1.0, 1),
    }
}

fn initial_state() -> CodecStreamState {
    CodecStreamState {
        dac_pre: vec![0.0; 6],
        dac_carry: [8, 5, 4, 3]
            .into_iter()
            .map(|stride| vec![0.0; stride])
            .collect(),
        dac_ru: (0..4)
            .map(|_| [vec![0.0; 6], vec![0.0; 18], vec![0.0; 54]])
            .collect(),
        dac_post: vec![0.0; 6],
        ..CodecStreamState::default()
    }
}

fn rectangle_count(index: usize) -> u8 {
    // With z denoting one output-sample delay, the complete multirate filter is
    // (1 + ... + z^479) (1 + z^480) (1 + z^60) (1 + z^12) (1 + z^3).
    // This follows from factoring each length-2s box into a length-s box and
    // (1 + z^s), then telescoping the four rate changes. Count sixteen shifted
    // rectangles directly: no convolution, residual, carry or decoder routine
    // participates in the expected values.
    let mut count = 0;
    for subset in 0_u8..16 {
        let offset = [480, 60, 12, 3]
            .into_iter()
            .enumerate()
            .filter(|&(bit, _)| subset & (1 << bit) != 0)
            .map(|(_, offset)| offset)
            .sum::<usize>();
        if index >= offset && index - offset < 480 {
            count += 1;
        }
    }
    count
}

fn expected_sample(index: usize) -> f32 {
    [(0, 0.5), (480, -0.75), (960, 0.75)]
        .into_iter()
        .filter(|&(offset, _)| index >= offset)
        .map(|(offset, sample)| sample * f32::from(rectangle_count(index - offset)) / 16.0)
        .sum()
}

const INPUT: [f32; 11] = [0.5, -0.75, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

#[test]
fn dac_reference_complete_four_block_impulse_has_literal_pcm_and_flushed_state() -> io::Result<()> {
    let decoder = decoder();
    assert_eq!(decoder.sample_rate(), 24_000);
    for chunks in [&[11][..], &[1; 11][..], &[2, 3, 6][..]] {
        let mut state = initial_state();
        let mut offset = 0;
        let mut output = Vec::new();
        for &frames in chunks {
            output.extend(decoder.forward_stream(
                &INPUT[offset..offset + frames],
                frames,
                &mut state,
            )?);
            offset += frames;
        }
        assert_eq!(output.len(), 11 * 480);
        for (index, &actual) in output.iter().enumerate() {
            assert_eq!(actual, expected_sample(index), "sample {index}");
        }
        // Eight final silent input frames drain every causal history and every
        // overlap, including the final dilation-9 residual's 54-sample history.
        assert_eq!(state, initial_state());
    }
    Ok(())
}

#[test]
fn dac_reference_final_nonlinearity_and_finite_saturation_follow_closed_form() -> io::Result<()> {
    let mut decoder = decoder();
    decoder.snake_post = SnakeBeta {
        a: vec![1.0],
        inv_b: vec![0.25],
    };
    decoder.conv_post.weight[6] = 32.0;
    let output = decoder.forward_stream(&INPUT, INPUT.len(), &mut initial_state())?;
    let mut positive_clips = 0;
    let mut negative_clips = 0;
    for (index, &actual) in output.iter().enumerate() {
        let linear = f64::from(expected_sample(index));
        let sine = linear.sin();
        let projected = 32.0 * (linear + sine * sine / 4.0);
        let expected = projected.clamp(-1.0, 1.0);
        assert!(
            (f64::from(actual) - expected).abs() <= 2e-7,
            "sample {index}"
        );
        positive_clips += usize::from(projected >= 1.0);
        negative_clips += usize::from(projected <= -1.0);
    }
    assert!(positive_clips > 0);
    assert!(negative_clips > 0);
    Ok(())
}

#[test]
fn dac_reference_nonfinite_final_projection_is_rejected_before_clipping() {
    for gain in [f32::MAX, -f32::MAX, f32::NAN] {
        let decoder = DacDecoder {
            conv_pre: current_convolution(1.0, 1),
            blocks: Vec::new(),
            snake_post: identity_snake(),
            conv_post: current_convolution(gain, 1),
        };
        let mut state = CodecStreamState {
            dac_pre: vec![0.0; 6],
            dac_post: vec![0.0; 6],
            ..CodecStreamState::default()
        };
        assert!(
            decoder
                .forward_stream(&[2.0], 1, &mut state)
                .is_err_and(|error| error.kind() == io::ErrorKind::InvalidData)
        );
    }
}
