//! Closed-form codec-transformer answers, independent of serving helpers.
//!
//! These synthetic weights supply mathematical controls, not captured native
//! execution or pinned-weight codec, PCM, voice or listening qualification.

use std::io;

use super::{
    KvState, Transformer, TransformerLayer, TransformerState, append_attention, normalize,
};
use crate::engine::{LinearProjection, WeightQuantization};

fn projection(weight: [f32; 4], bias: [f32; 2]) -> io::Result<LinearProjection> {
    LinearProjection::new(
        "codec-mathematical.weight",
        2,
        2,
        WeightQuantization::F32(weight.to_vec()),
        Some(bias.to_vec()),
    )
}

fn close(actual: &[f32], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert!(
            (f64::from(actual) - expected).abs() <= tolerance,
            "actual {actual:?}, expected {expected:?}"
        );
    }
}

fn graph(layer: TransformerLayer) -> Transformer {
    Transformer {
        threads: 1,
        hidden: 2,
        heads: 1,
        kv_heads: 1,
        head_dim: 2,
        window: None,
        theta: 10_000.0,
        eps: 3.0,
        input: None,
        output: None,
        final_norm: None,
        layers: vec![layer],
    }
}

#[test]
fn codec_reference_mimi_layer_has_closed_form_normalization_attention_and_gelu_answers()
-> io::Result<()> {
    let graph = graph(TransformerLayer {
        norm: vec![2.0, 4.0],
        norm_bias: Some(vec![0.5, -0.25]),
        q: projection([0.0; 4], [0.0; 2])?,
        k: projection([0.0; 4], [0.0; 2])?,
        v: projection([1.0, 0.0, 0.0, 1.0], [0.0; 2])?,
        output: projection([1.0, 0.0, 0.0, 1.0], [0.0; 2])?,
        attn_scale: vec![2.0, -0.5],
        ffn_norm: vec![1.0; 2],
        ffn_bias: Some(vec![0.0; 2]),
        gate: None,
        up: projection([0.0; 4], [1.0, -1.0])?,
        down: projection([1.0, 0.0, 0.0, 1.0], [0.0; 2])?,
        ffn_scale: vec![0.5, -2.0],
    });
    // The first two rows have mean zero/variance one; the third has variance
    // nine. Epsilon three gives scales 1/2 and 1/sqrt(12). Uniform attention
    // therefore averages these independently derived projected values.
    let root = 3.0_f64.sqrt();
    let expected_cache = [1.5, -2.25, -0.5, 1.75, root + 0.5, -2.0 * root - 0.25];
    // Phi(1) is the standard-normal CDF's known value, independently tabulated.
    let phi_one = 0.841_344_746_068_542_9_f64;
    let ffn = [0.5 * phi_one, 2.0 * (1.0 - phi_one)];
    let mut state = TransformerState::default();
    for (index, input) in [[1.0, -1.0], [-1.0, 1.0], [3.0, -3.0]]
        .into_iter()
        .enumerate()
    {
        let count = f64::from(u16::try_from(index + 1).map_err(io::Error::other)?);
        let (sum_x, sum_y) = match index {
            0 => (1.5, -2.25),
            1 => (1.0, -0.5),
            _ => (root + 1.5, -2.0 * root - 0.75),
        };
        close(
            &graph.step(&input, &mut state)?,
            &[
                f64::from(input[0]) + 2.0 * sum_x / count + ffn[0],
                f64::from(input[1]) - 0.5 * sum_y / count + ffn[1],
            ],
            2e-6,
        );
        assert_eq!(state.position, index + 1);
        assert_eq!(state.layers.len(), 1);
        assert!(state.layers[0].keys.iter().all(|&value| value == 0.0));
        close(
            &state.layers[0].values,
            &expected_cache[..2 * (index + 1)],
            3e-7,
        );
    }
    let previous = state.clone();
    assert!(graph.step(&[f32::MAX, -f32::MAX], &mut state).is_err());
    assert_eq!(state.position, previous.position);
    assert_eq!(state.layers.len(), previous.layers.len());
    for (actual, expected) in state.layers.iter().zip(&previous.layers) {
        for (actual, expected) in [
            (&actual.keys, &expected.keys),
            (&actual.values, &expected.values),
        ] {
            assert!(
                actual
                    .iter()
                    .zip(expected)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
            assert_eq!(actual.len(), expected.len());
        }
    }
    // A rejected reduction leaves the old three rows intact for fresh input.
    close(
        &graph.step(&[0.0, 0.0], &mut state)?,
        &[
            0.5 * (root + 2.0) + ffn[0],
            0.125 * (2.0 * root + 1.0) + ffn[1],
        ],
        2e-6,
    );
    assert_eq!(state.position, 4);
    Ok(())
}

#[test]
fn codec_reference_decoder_layer_has_closed_form_rope_swiglu_and_final_projection_answers()
-> io::Result<()> {
    let log_three = 3.0_f32.ln();
    let mut graph = graph(TransformerLayer {
        norm: vec![1.0; 2],
        norm_bias: None,
        q: projection([0.0; 4], [1.0, 2.0])?,
        k: projection([0.0; 4], [3.0, 4.0])?,
        v: projection([0.0; 4], [5.0, 6.0])?,
        output: projection([0.5, 0.0, 0.0, -0.25], [0.25, -0.5])?,
        attn_scale: vec![2.0, -0.5],
        ffn_norm: vec![1.0; 2],
        ffn_bias: None,
        gate: Some(projection([0.0; 4], [-log_three, log_three])?),
        up: projection([0.0; 4], [2.0, -2.0])?,
        down: projection([1.0, 0.0, 0.0, 1.0], [0.5, 1.0])?,
        ffn_scale: vec![2.0, -1.0],
    });
    graph.window = Some(2);
    graph.input = Some(projection([2.0, 0.0, 0.0, 0.5], [1.0, -1.0])?);
    graph.final_norm = Some(vec![2.0, -1.0]);
    graph.output = Some(projection([0.5, 0.0, 0.0, 2.0], [-0.25, 0.75])?);
    // sigmoid(-ln 3)=1/4 and sigmoid(ln 3)=3/4 give exact symbolic
    // SwiGLU terms. Constant V makes attention's output [5,6] at all steps;
    // the separate nonuniform-attention control covers score normalization.
    let log_three = 3.0_f64.ln();
    let mut state = TransformerState::default();
    for (index, input) in [[0.25, -0.5], [-1.0, 2.0], [0.0, 1.0], [2.0, -3.0]]
        .into_iter()
        .enumerate()
    {
        let hidden = [
            2.0 * f64::from(input[0]) + 7.5 - log_three,
            0.5 * f64::from(input[1]) - 1.0 + 1.5 * log_three,
        ];
        let denominator = ((hidden[0].powi(2) + hidden[1].powi(2)) / 2.0 + 3.0).sqrt();
        close(
            &graph.step(&input, &mut state)?,
            &[
                hidden[0] / denominator - 0.25,
                -2.0 * hidden[1] / denominator + 0.75,
            ],
            2e-6,
        );
        assert_eq!(state.position, index + 1);
        let first = (index + 1).saturating_sub(2);
        let mut expected_keys = Vec::new();
        for position in first..=index {
            let angle = f64::from(u16::try_from(position).map_err(io::Error::other)?);
            expected_keys.extend([
                3.0 * angle.cos() - 4.0 * angle.sin(),
                3.0 * angle.sin() + 4.0 * angle.cos(),
            ]);
        }
        close(&state.layers[0].ordered_keys(), &expected_keys, 8e-7);
        assert_eq!(
            state.layers[0].ordered_values(),
            [5.0, 6.0].repeat((index + 1).min(2))
        );
    }
    Ok(())
}

#[test]
fn codec_reference_grouped_attention_has_literal_nonuniform_probability_and_window_answers()
-> io::Result<()> {
    let log_two = 2.0_f32.ln();
    let query = [log_two, 2.0 * log_two, -log_two, -2.0 * log_two];
    for window in [None, Some(2)] {
        let mut state = KvState {
            keys: vec![0.0, 0.0, 1.0, -1.0],
            values: vec![1.0, -1.0, 4.0, 8.0],
            head: 0,
        };
        let output = append_attention(
            &query,
            &[2.0, -2.0],
            &[7.0, 11.0],
            4,
            2,
            1,
            window,
            &mut state,
        )?;
        // Per group, q=ln2 gives weights 1:2:4 and q=2ln2 gives
        // 1:4:16. The causal two-row window removes only the first term.
        let expected = if window.is_some() {
            [6.0, 32.0 / 5.0, 10.0, 52.0 / 5.0]
        } else {
            [37.0 / 7.0, 129.0 / 21.0, 59.0 / 7.0, 207.0 / 21.0]
        };
        close(&output, &expected, 2e-6);
        // The enclosing full-position transaction owns physical eviction.
        assert_eq!(state.keys, [0.0, 0.0, 1.0, -1.0, 2.0, -2.0]);
        assert_eq!(state.values, [1.0, -1.0, 4.0, 8.0, 7.0, 11.0]);
    }
    Ok(())
}

#[test]
fn codec_reference_layer_norm_rejects_finite_reduction_overflow_and_invalid_parameters() {
    for input in [[f32::MAX, -f32::MAX], [f32::MAX; 2]] {
        assert!(normalize(&input, &[1.0; 2], Some(&[0.25, -0.5]), 1.0).is_err());
    }
    for epsilon in [0.0, -1.0, f32::INFINITY, f32::NAN] {
        assert!(normalize(&[1.0, -1.0], &[1.0; 2], Some(&[0.0; 2]), epsilon).is_err());
    }
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(normalize(&[invalid, 0.0], &[1.0; 2], Some(&[0.0; 2]), 1.0).is_err());
        assert!(normalize(&[1.0, -1.0], &[invalid, 1.0], Some(&[0.0; 2]), 1.0).is_err());
        assert!(normalize(&[1.0, -1.0], &[1.0; 2], Some(&[invalid, 0.0]), 1.0).is_err());
    }
    // The normalized deviation is >1 for this four-channel row, so the finite
    // scale overflows even though mean, variance and epsilon remain finite.
    assert!(
        normalize(
            &[3.0, -1.0, -1.0, -1.0],
            &[f32::MAX; 4],
            Some(&[0.0; 4]),
            0.25
        )
        .is_err()
    );
}
