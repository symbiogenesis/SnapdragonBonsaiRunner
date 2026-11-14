use super::*;

fn inputs(projection: Q8Matmul) -> (Vec<u8>, Vec<f32>) {
    let mut weights = vec![0; projection.weight_bytes()];
    for (index, block) in weights.as_chunks_mut::<34>().0.iter_mut().enumerate() {
        // Four independently specified signed scales: 1, 1/2, -1, 2.
        let scale = [0x3c00_u16, 0x3800, 0xbc00, 0x4000][index % 4];
        block[..2].copy_from_slice(&scale.to_le_bytes());
        block[2] = 0x80; // -128 must never become an unsigned dot operand.
        block[3] = 0x7f;
    }
    let mut activation = vec![0.0; projection.columns() * projection.batch()];
    for (index, vector) in activation
        .chunks_exact_mut(projection.columns())
        .enumerate()
    {
        for block in vector.as_chunks_mut::<32>().0 {
            let sign = if index % 2 == 0 { 1.0 } else { -1.0 };
            block[0] = 127.0 * sign;
            block[1] = -127.0 * sign;
        }
    }
    (weights, activation)
}

#[test]
fn q8_oracle_independent_signed_dots_scale_order_and_all_vectors() -> Result<(), Error> {
    let projection = Q8Matmul::new(128, 64, 4, 200_000)?;
    let (mut weights, activation) = inputs(projection);
    // Last row's last block additionally observes a signed tail code.
    let last = weights.len() - 1;
    weights[last] = 0xff;
    let mut activation = activation;
    for vector in activation.as_chunks_mut::<128>().0 {
        vector[127] = 3.0;
    }
    let oracle = Q8ExactOracle::new(projection, &weights, &activation)?;
    // dot=-128*127 + 127*(-127)=-32385, scaled by 1+.5-1+2=2.5.
    let mut expected = vec![0.0; 256];
    for (index, vector) in expected.as_chunks_mut::<64>().0.iter_mut().enumerate() {
        vector.fill(if index % 2 == 0 { -80_962.5 } else { 80_962.5 });
        vector[63] -= 6.0; // final code -1 * activation 3 * scale 2.
    }
    oracle.review(&expected)?;
    expected[255] += 1.0;
    assert!(oracle.review(&expected).is_err());
    expected[255] = f32::NAN;
    assert!(oracle.review(&expected).is_err());
    assert!(oracle.review(&expected[..255]).is_err());
    Ok(())
}

#[test]
fn q8_oracle_rejects_general_quantization_and_unsupported_half_scales() -> Result<(), Error> {
    let projection = Q8Matmul::new(128, 32, 2, 100_000)?;
    let (weights, activation) = inputs(projection);
    for invalid in [f32::NAN, f32::INFINITY, -128.0, 127.25, 0.5] {
        let mut changed = activation.clone();
        changed[255] = invalid;
        assert!(Q8ExactOracle::new(projection, &weights, &changed).is_err());
    }
    let mut no_anchor = activation.clone();
    no_anchor[224..256].fill(1.0);
    assert!(Q8ExactOracle::new(projection, &weights, &no_anchor).is_err());
    for invalid in [0x7c00_u16, 0x7e00, 0x0001, 0x1800, 0x6000, 0x3c01] {
        let mut changed = weights.clone();
        let end = changed.len();
        changed[end - 34..end - 32].copy_from_slice(&invalid.to_le_bytes());
        assert!(Q8ExactOracle::new(projection, &changed, &activation).is_err());
    }
    assert!(Q8ExactOracle::new(projection, &weights[..weights.len() - 1], &activation).is_err());
    assert!(Q8ExactOracle::new(projection, &weights, &activation[..255]).is_err());
    Ok(())
}

#[test]
fn q8_oracle_zero_sign_and_rounded_prefixes_cannot_grant_exact_acceptance() -> Result<(), Error> {
    let projection = Q8Matmul::new(128, 32, 1, 100_000)?;
    let (mut weights, mut activation) = inputs(projection);
    for block in weights.as_chunks_mut::<34>().0 {
        block[..2].copy_from_slice(&0x8000_u16.to_le_bytes());
    }
    let zero = Q8ExactOracle::new(projection, &weights, &activation)?;
    zero.review(&[0.0; 32])?;
    assert!(zero.review(&[-0.0; 32]).is_err());
    // A large exact first block is followed by one 2^-8 unit.
    // Even if later cancellation could recover an exact final sum, admission
    // must reject the first inexact prefix of the DSP's ordered additions.
    for block in weights.as_chunks_mut::<34>().0 {
        block.fill(0);
    }
    weights[..2].copy_from_slice(&0x5c00_u16.to_le_bytes()); // scale 256
    weights[2] = 0x80;
    weights[3] = 0x80;
    weights[4] = 0x80;
    weights[5] = 0x80;
    activation[..32].fill(0.0);
    activation[..4].fill(-127.0);
    // 4*128*127*256 = 16,646,144, exact but ulp=1.
    weights[34..36].copy_from_slice(&0x1c00_u16.to_le_bytes()); // scale 2^-8
    weights[36] = 1;
    activation[32] = 1.0;
    activation[33] = 127.0; // retain unit quantization without a weight here.
    assert!(Q8ExactOracle::new(projection, &weights, &activation).is_err());
    Ok(())
}

struct ReviewedTransport<'a> {
    weights: &'a [u8],
    activation: &'a [f32],
    value: f32,
    calls: u32,
}

impl Q8Submitter for ReviewedTransport<'_> {
    fn submit(
        &mut self,
        weights: &[u8],
        activation: &[f32],
        output: &mut [f32],
        oracle: &mut dyn FnMut(&[f32]) -> Result<(), Error>,
    ) -> Result<(), Error> {
        assert!(std::ptr::eq(weights, self.weights));
        assert!(std::ptr::eq(activation, self.activation));
        self.calls += 1;
        let candidate = vec![self.value; output.len()];
        oracle(&candidate)?;
        output.copy_from_slice(&candidate);
        Ok(())
    }
}

#[test]
fn q8_oracle_submits_original_inputs_and_refuses_corrupt_result_before_publication()
-> Result<(), Error> {
    let projection = Q8Matmul::new(128, 32, 1, 100_000)?;
    let (weights, activation) = inputs(projection);
    let oracle = Q8ExactOracle::new(projection, &weights, &activation)?;
    let mut transport = ReviewedTransport {
        weights: &weights,
        activation: &activation,
        value: -80_962.5,
        calls: 0,
    };
    let mut output = [77.0; 32];
    oracle.submit(&mut transport, &mut output)?;
    assert_eq!(output, [-80_962.5; 32]);
    transport.value = 80_962.5;
    assert!(oracle.submit(&mut transport, &mut output).is_err());
    assert_eq!(output, [-80_962.5; 32]);
    assert_eq!(transport.calls, 2);
    Ok(())
}

#[test]
fn q8_reference_fractional_signed_dots_and_late_vector_review() -> Result<(), Error> {
    for batch in [1, 2, 4] {
        let projection = Q8Matmul::new(128, 64, batch, 200_000)?;
        let mut weights = vec![0; projection.weight_bytes()];
        for (index, block) in weights.as_chunks_mut::<34>().0.iter_mut().enumerate() {
            // 1.5, -0.5, smallest positive half subnormal, negative zero.
            let scale = [0x3e00_u16, 0xb800, 0x0001, 0x8000][index % 4];
            block[..2].copy_from_slice(&scale.to_le_bytes());
            block[2..6].copy_from_slice(&[0x80, 0x7f, 0xff, 2]);
        }
        let mut activation = vec![0.0; projection.columns() * batch];
        for (index, vector) in activation.as_chunks_mut::<128>().0.iter_mut().enumerate() {
            let sign = if index % 2 == 0 { 1.0 } else { -1.0 };
            for block in vector.as_chunks_mut::<32>().0 {
                block[..4].copy_from_slice(&[0.125 * sign, -0.25 * sign, 1.5 * sign, -2.0 * sign]);
            }
        }
        // Signed dot=-16-31.75-1.5-4=-53.25. Scale sum=1+2^-24.
        // Returning -53.25 has absolute error 53.25/2^24 < 0.000004.
        let oracle = Q8ReferenceOracle::new(projection, &weights, &activation, 0.000_004, 0.0)?;
        let mut output = vec![-53.25; 64 * batch];
        for vector in output.as_chunks_mut::<64>().0.iter_mut().skip(1).step_by(2) {
            vector.fill(53.25);
        }
        oracle.review(&output)?;
        let end = output.len() - 1;
        output[end] = 0.0;
        assert!(oracle.review(&output).is_err());
        output[end] = f32::NAN;
        assert!(oracle.review(&output).is_err());
        assert!(oracle.review(&output[..end]).is_err());
    }
    Ok(())
}

#[test]
fn q8_reference_finite_half_scales_preserve_subnormals_and_signed_extremes() -> Result<(), Error> {
    let projection = Q8Matmul::new(128, 32, 1, 100_000)?;
    let mut weights = vec![0; projection.weight_bytes()];
    let mut activation = vec![0.0; 128];
    // First row dot=-128*0.5 + 127*0.5 + 2*0.75=1, exactly.
    activation[..3].copy_from_slice(&[0.5, 0.5, 0.75]);
    let half_and_expected = [
        (0x0000_u16, 0x0000_0000_u32),
        (0x8000, 0x8000_0000),
        (0x0001, 0x3380_0000),
        (0x8001, 0xb380_0000),
        (0x03ff, 0x387f_c000),
        (0x0400, 0x3880_0000),
        (0x3555, 0x3eaa_a000),
        (0xb555, 0xbeaa_a000),
        (0x7bff, 0x477f_e000),
        (0xfbff, 0xc77f_e000),
    ];
    let mut output = vec![0.0; 32];
    for (row, value) in output.iter_mut().enumerate() {
        let (half, expected) = half_and_expected[row % half_and_expected.len()];
        let start = row * 4 * 34;
        weights[start..start + 2].copy_from_slice(&half.to_le_bytes());
        weights[start + 2..start + 5].copy_from_slice(&[0x80, 0x7f, 2]);
        *value = f32::from_bits(expected);
    }
    let oracle = Q8ReferenceOracle::new(projection, &weights, &activation, 0.0, 0.0)?;
    oracle.review(&output)?;
    // Numeric error review deliberately accepts either zero sign.
    output[0] = -0.0;
    output[1] = 0.0;
    oracle.review(&output)?;
    output[31] = f32::INFINITY;
    assert!(oracle.review(&output).is_err());
    Ok(())
}

#[test]
fn q8_reference_predeclared_tolerance_boundaries_are_inclusive_and_immutable() -> Result<(), Error>
{
    let projection = Q8Matmul::new(128, 32, 1, 100_000)?;
    let mut weights = vec![0; projection.weight_bytes()];
    let mut activation = vec![0.0; 128];
    activation[0] = 0.5;
    for row in weights.as_chunks_mut::<136>().0 {
        row[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
        row[2] = 4;
    }
    let reference = Q8ReferenceOracle::new(projection, &weights, &activation, 0.25, 0.125)?;
    // Reference=2; allowed error=.25+.125*2=.5, inclusive on either side.
    reference.review(&[1.5; 32])?;
    reference.review(&[2.5; 32])?;
    assert!(
        reference
            .review(&[f32::from_bits(2.5_f32.to_bits() + 1); 32])
            .is_err()
    );
    assert!(
        reference
            .review(&[f32::from_bits(1.5_f32.to_bits() - 1); 32])
            .is_err()
    );
    reference.review(&[2.0; 32])?;
    for (absolute, relative) in [
        (-1.0, 0.0),
        (f64::NAN, 0.0),
        (f64::INFINITY, 0.0),
        (f64::MAX, 0.0),
        (0.0, -1.0),
        (0.0, f64::NAN),
        (0.0, f64::INFINITY),
        (0.0, 1.00001),
    ] {
        assert!(
            Q8ReferenceOracle::new(projection, &weights, &activation, absolute, relative).is_err()
        );
    }
    Ok(())
}

#[test]
fn q8_reference_rejects_nonfinite_extents_and_unrepresentable_projection() -> Result<(), Error> {
    let projection = Q8Matmul::new(128, 32, 1, 100_000)?;
    let mut weights = vec![0; projection.weight_bytes()];
    let mut activation = vec![0.5; 128];
    assert!(
        Q8ReferenceOracle::new(
            projection,
            &weights[..weights.len() - 1],
            &activation,
            0.0,
            0.0
        )
        .is_err()
    );
    assert!(Q8ReferenceOracle::new(projection, &weights, &activation[..127], 0.0, 0.0).is_err());
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        activation[127] = invalid;
        assert!(Q8ReferenceOracle::new(projection, &weights, &activation, 0.0, 0.0).is_err());
    }
    activation[127] = 0.5;
    let end = weights.len();
    for half in [0x7c00_u16, 0xfc00, 0x7e00] {
        weights[end - 34..end - 32].copy_from_slice(&half.to_le_bytes());
        assert!(Q8ReferenceOracle::new(projection, &weights, &activation, 0.0, 0.0).is_err());
    }
    weights[end - 34..end - 32].copy_from_slice(&0x3c00_u16.to_le_bytes());
    weights[end - 1] = 2;
    activation[127] = f32::MAX;
    assert!(Q8ReferenceOracle::new(projection, &weights, &activation, 0.0, 0.0).is_err());
    Ok(())
}

#[test]
fn q8_reference_submits_original_fractional_inputs_and_rejects_before_publication()
-> Result<(), Error> {
    let projection = Q8Matmul::new(128, 32, 1, 100_000)?;
    let mut weights = vec![0; projection.weight_bytes()];
    for row in weights.as_chunks_mut::<136>().0 {
        row[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
        row[2] = 3;
    }
    let mut activation = vec![0.0; 128];
    activation[0] = 0.125;
    let reference = Q8ReferenceOracle::new(projection, &weights, &activation, 0.0, 0.0)?;
    let mut transport = ReviewedTransport {
        weights: &weights,
        activation: &activation,
        value: 0.375,
        calls: 0,
    };
    let mut output = [77.0; 32];
    reference.submit(&mut transport, &mut output)?;
    assert_eq!(output, [0.375; 32]);
    transport.value = 0.5;
    assert!(reference.submit(&mut transport, &mut output).is_err());
    assert_eq!(output, [0.375; 32]);
    assert_eq!(transport.calls, 2);
    Ok(())
}

#[test]
fn q8_control_refusal_during_validation_and_arithmetic_returns_no_oracle() -> Result<(), Error> {
    let projection = Q8Matmul::new(128, 32, 1, 100_000)?;
    let (weights, activation) = inputs(projection);
    // Stop before intake, during activation/weight validation, during arithmetic,
    // and after the final block. No attempt may return a partial expected array.
    let mut total = 0;
    Q8ExactOracle::new_with_control(projection, &weights, &activation, || {
        total += 1;
        Ok(())
    })?;
    for refusal in [1, 3, 20, total - 1, total] {
        let mut calls = 0;
        let mut control = || {
            calls += 1;
            if calls == refusal {
                Err(Error::Contract)
            } else {
                Ok(())
            }
        };
        assert!(
            Q8ExactOracle::new_with_control(projection, &weights, &activation, &mut control)
                .is_err()
        );
        assert_eq!(calls, refusal);
        calls = 0;
        assert!(
            Q8ReferenceOracle::new_with_control(
                projection,
                &weights,
                &activation,
                0.0,
                0.0,
                || {
                    calls += 1;
                    if calls == refusal {
                        Err(Error::Contract)
                    } else {
                        Ok(())
                    }
                },
            )
            .is_err()
        );
        assert_eq!(calls, refusal);
    }
    Ok(())
}

#[test]
fn q8_control_expired_original_deadline_and_cancellation_skip_transport() -> Result<(), Error> {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Instant;
    let projection = Q8Matmul::new(128, 32, 1, 100_000)?;
    let (weights, activation) = inputs(projection);
    let exact = Q8ExactOracle::new(projection, &weights, &activation)?;
    let reference = Q8ReferenceOracle::new(projection, &weights, &activation, 0.0, 0.0)?;
    let mut transport = ReviewedTransport {
        weights: &weights,
        activation: &activation,
        value: -80_962.5,
        calls: 0,
    };
    let mut output = [77.0; 32];
    let expired = Instant::now();
    let cancelled = AtomicBool::new(false);
    let mut check = || {
        if cancelled.load(Ordering::Relaxed) || Instant::now() >= expired {
            Err(Error::Contract)
        } else {
            Ok(())
        }
    };
    assert!(
        exact
            .submit_with_control(&mut transport, &mut output, &mut check)
            .is_err()
    );
    cancelled.store(true, Ordering::Relaxed);
    assert!(
        reference
            .submit_with_control(&mut transport, &mut output, || {
                if cancelled.load(Ordering::Relaxed) {
                    Err(Error::Contract)
                } else {
                    Ok(())
                }
            })
            .is_err()
    );
    assert_eq!(transport.calls, 0);
    assert_eq!(output, [77.0; 32]);
    Ok(())
}

#[test]
fn q8_control_late_refusal_prevents_exact_and_tolerance_publication() -> Result<(), Error> {
    let projection = Q8Matmul::new(128, 32, 1, 100_000)?;
    let (weights, activation) = inputs(projection);
    let exact = Q8ExactOracle::new(projection, &weights, &activation)?;
    let reference = Q8ReferenceOracle::new(projection, &weights, &activation, 0.0, 0.0)?;
    let mut transport = ReviewedTransport {
        weights: &weights,
        activation: &activation,
        value: -80_962.5,
        calls: 0,
    };
    let mut output = [77.0; 32];
    let mut total = 0;
    exact.submit_with_control(&mut transport, &mut output, || {
        total += 1;
        Ok(())
    })?;
    assert_eq!(output, [-80_962.5; 32]);
    for refusal in [3, total - 1, total] {
        let mut calls = 0;
        output.fill(77.0);
        assert!(
            exact
                .submit_with_control(&mut transport, &mut output, || {
                    calls += 1;
                    if calls == refusal {
                        Err(Error::Contract)
                    } else {
                        Ok(())
                    }
                })
                .is_err()
        );
        assert_eq!(calls, refusal);
        assert_eq!(output, [77.0; 32]);
        calls = 0;
        assert!(
            reference
                .submit_with_control(&mut transport, &mut output, || {
                    calls += 1;
                    if calls == refusal {
                        Err(Error::Contract)
                    } else {
                        Ok(())
                    }
                })
                .is_err()
        );
        assert_eq!(calls, refusal);
        assert_eq!(output, [77.0; 32]);
    }
    Ok(())
}
