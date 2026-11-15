//! Independent original-weight Q8 projection controls.
//!
//! `Q8ReferenceOracle` reviews ordinary finite activations against an explicitly
//! ordered scalar F64 projection with caller-declared error limits. It measures
//! projection error, without emulating HVX activation quantization or QF16.
//!
//! `Q8HvxQuantizationOracle` additionally models the pinned `0x2008` binary16
//! activation scale and reciprocal stage in IEEE-754 binary16 RNE with ordered
//! `F64`-to-`F32` block projection under caller-declared tolerances, without
//! emulating HVX `QF16`/`QF32`, the `hvx_vec_inverse_f16` LUT polynomial,
//! `hvx_vec_i16_from_hf_rnd_sat`, or per-block `F32` accumulation.
//!
//! The pinned `quantize_block_f32_q8_0_tiled` converts each 32-value group to
//! half, multiplies its absolute maximum by half 0x2008, and uses the half
//! inverse before rounding to signed codes. Integer inputs with magnitude at
//! most 127 and a magnitude-127 anchor in every group have exactly unit scale
//! and unchanged codes. This control rejects all other activation inputs.
//! The signed integer dots below read original row-major weights, independently
//! of the native tiled layout. Half power-of-two weight scales and an exact
//! representation check on every running sum avoid emulating approximate HVX
//! arithmetic or substituting a Rust TTS F32/FMA oracle for the DSP algorithm.

use crate::{Error, Q8Matmul, Q8Submitter};

/// Independent scalar projection-error review for ordinary F32 activations.
///
/// Original row-major signed Q8 codes and finite binary16 scales are evaluated
/// in F64, with separate multiply/add and the original block/lane order. This
/// reference does not emulate DSP half activation quantization, its inverse,
/// QF16 arithmetic or kernel reduction order. Passing means only that every
/// returned value met the declared error limits for this scalar reference.
/// It grants no native identity, device, exact-parity or speech acceptance.
pub struct Q8ReferenceOracle<'a> {
    weights: &'a [u8],
    activation: &'a [f32],
    expected: Box<[f64]>,
    absolute_tolerance: f64,
    relative_tolerance: f64,
}

impl<'a> Q8ReferenceOracle<'a> {
    /// Evaluate every original row/vector with predeclared numerical tolerances.
    ///
    /// Each 32-code signed dot is accumulated in lane order in F64, multiplied
    /// by its exactly widened half scale, then added in original block order.
    /// All finite half scales, including subnormals and signed zero, and all
    /// finite F32 inputs are supported. A candidate must satisfy
    /// `abs(actual-reference) <= absolute + relative * abs(reference)`.
    /// Absolute tolerance is finite in `0..=f32::MAX`; relative tolerance is
    /// finite in `0..=1`. Neither has a default. The caller must independently
    /// review these limits before observing results; construction cannot
    /// authenticate that review. Unlike the exact oracle, equal zero values
    /// with different signs meet this numerical error rule.
    ///
    /// # Errors
    /// Rejects invalid tolerances, extents, nonfinite inputs/scales, allocation
    /// failure and a reference outside the finite F32 range before submission.
    pub fn new(
        projection: Q8Matmul,
        weights: &'a [u8],
        activation: &'a [f32],
        absolute_tolerance: f64,
        relative_tolerance: f64,
    ) -> Result<Self, Error> {
        Self::new_with_control(
            projection,
            weights,
            activation,
            absolute_tolerance,
            relative_tolerance,
            || Ok(()),
        )
    }

    /// Evaluate under the caller's original cancellation/deadline policy.
    ///
    /// `check` must retain the inherited absolute work deadline, including its
    /// cleanup reserve, rather than start a new allowance. It runs before input
    /// scanning/allocation, at most 32 arithmetic lanes apart and before return.
    /// This variant is required for bounded device work; [`Self::new`] supplies
    /// no cancellation or time bound. Refusal returns no partially built oracle.
    /// Synchronous allocation still requires external process supervision.
    ///
    /// # Errors
    /// Propagates control refusal and every input/arithmetic error of [`Self::new`].
    #[allow(
        clippy::suboptimal_flops,
        reason = "The independent scalar reference requires separately rounded multiplication and addition"
    )]
    pub fn new_with_control(
        projection: Q8Matmul,
        weights: &'a [u8],
        activation: &'a [f32],
        absolute_tolerance: f64,
        relative_tolerance: f64,
        mut check: impl FnMut() -> Result<(), Error>,
    ) -> Result<Self, Error> {
        check()?;
        if !absolute_tolerance.is_finite()
            || !(0.0..=f64::from(f32::MAX)).contains(&absolute_tolerance)
            || !relative_tolerance.is_finite()
            || !(0.0..=1.0).contains(&relative_tolerance)
            || weights.len() != projection.weight_bytes()
            || activation.len() != projection.columns() * projection.batch()
        {
            return Err(Error::Contract);
        }
        for block in activation.as_chunks::<32>().0 {
            check()?;
            if block.iter().any(|value| !value.is_finite()) {
                return Err(Error::Contract);
            }
        }
        for block in weights.as_chunks::<34>().0 {
            check()?;
            reference_scale(u16::from_le_bytes([block[0], block[1]]))?;
        }
        check()?;
        let mut expected = Vec::new();
        expected
            .try_reserve_exact(projection.rows() * projection.batch())
            .map_err(|_| Error::Contract)?;
        let row_bytes = projection.columns() / 32 * 34;
        for vector in activation.chunks_exact(projection.columns()) {
            for row in weights.chunks_exact(row_bytes) {
                let mut value = 0.0;
                for (weight, input) in row
                    .as_chunks::<34>()
                    .0
                    .iter()
                    .zip(vector.as_chunks::<32>().0)
                {
                    check()?;
                    let mut dot = 0.0;
                    for (&code, &input) in weight[2..].iter().zip(input) {
                        dot += f64::from(i8::from_ne_bytes([code])) * f64::from(input);
                    }
                    value += dot * reference_scale(u16::from_le_bytes([weight[0], weight[1]]))?;
                    if !value.is_finite() {
                        return Err(Error::Contract);
                    }
                }
                if value.abs() > f64::from(f32::MAX) {
                    return Err(Error::Contract);
                }
                expected.push(value);
            }
        }
        check()?;
        Ok(Self {
            weights,
            activation,
            expected: expected.into_boxed_slice(),
            absolute_tolerance,
            relative_tolerance,
        })
    }

    /// Require every finite returned value to satisfy the original error rule.
    ///
    /// The full output is checked, including later vectors and row tails.
    /// Tolerances cannot be changed after construction or a failed review.
    ///
    /// # Errors
    /// Rejects changed extents, nonfinite values or any exceeded error bound.
    pub fn review(&self, output: &[f32]) -> Result<(), Error> {
        self.review_with_control(output, || Ok(()))
    }

    /// Review every result under the original caller-owned work allowance.
    ///
    /// `check` follows the same contract as [`Self::new_with_control`]. Refusal
    /// cannot become numerical success, including after the last output value.
    ///
    /// # Errors
    /// Propagates control refusal and the numerical errors of [`Self::review`].
    #[allow(
        clippy::suboptimal_flops,
        reason = "The declared tolerance formula preserves separate multiply/add rounding"
    )]
    pub fn review_with_control(
        &self,
        output: &[f32],
        mut check: impl FnMut() -> Result<(), Error>,
    ) -> Result<(), Error> {
        check()?;
        if output.len() != self.expected.len() {
            return Err(Error::Contract);
        }
        for (&actual, &expected) in output.iter().zip(&self.expected) {
            check()?;
            if !actual.is_finite()
                || (f64::from(actual) - expected).abs()
                    > self.absolute_tolerance + self.relative_tolerance * expected.abs()
            {
                return Err(Error::Contract);
            }
        }
        check()
    }

    /// Submit unchanged borrowed inputs with mandatory scalar error review.
    ///
    /// The transport owns terminal errors, deadlines, cancellation and cleanup.
    /// No candidate output may be published before the complete review passes.
    ///
    /// # Errors
    /// Propagates transport failures or rejected numerical error bounds.
    pub fn submit(&self, transport: &mut dyn Q8Submitter, output: &mut [f32]) -> Result<(), Error> {
        self.submit_with_control(transport, output, || Ok(()))
    }

    /// Submit with the same cancellation/deadline policy used for preparation.
    ///
    /// Control refusal before submission makes no transport call. During review,
    /// refusal reaches the transport before publication and ends native ownership
    /// under [`Q8Submitter`]'s terminal-failure contract. The transport still owns
    /// its independent native-call deadline and uncertain-buffer retention.
    ///
    /// # Errors
    /// Propagates control, transport and numerical review failures.
    pub fn submit_with_control(
        &self,
        transport: &mut dyn Q8Submitter,
        output: &mut [f32],
        mut check: impl FnMut() -> Result<(), Error>,
    ) -> Result<(), Error> {
        check()?;
        transport.submit(self.weights, self.activation, output, &mut |actual| {
            self.review_with_control(actual, &mut check)
        })
    }
}

fn reference_scale(bits: u16) -> Result<f64, Error> {
    let exponent = (bits >> 10) & 31;
    let fraction = bits & 1023;
    if exponent == 31 {
        return Err(Error::Contract);
    }
    let (significand, power) = if exponent == 0 {
        (fraction, -24)
    } else {
        (1024 + fraction, i32::from(exponent) - 25)
    };
    let magnitude = f64::from(significand) * 2.0_f64.powi(power);
    Ok(if bits & 0x8000 == 0 {
        magnitude
    } else {
        -magnitude
    })
}

/// Independent numeric review of unchanged inputs in an exact diagnostic subset.
///
/// This is suitable for a separately admitted Q8 transport arithmetic control,
/// not arbitrary model activations, a native capture or speech qualification.
/// Construction opens no driver and authenticates no model/package identity.
pub struct Q8ExactOracle<'a> {
    weights: &'a [u8],
    activation: &'a [f32],
    expected: Box<[f32]>,
}

impl<'a> Q8ExactOracle<'a> {
    /// Independently evaluate every signed dot from original GGUF Q8 blocks.
    ///
    /// Each 32-value activation group must contain only integers in -127..127
    /// and have absolute maximum 127. Every weight scale must be signed zero or
    /// a signed power of two from 2^-8 through 2^8. Every prefix of the ordered
    /// 32-code block sum must be exactly representable in F32. These constraints
    /// make the selected DSP half quantization and separate multiply/add exact;
    /// they deliberately exclude general dynamic quantization and rounded sums.
    /// Original borrowed input bytes remain unchanged through submission.
    ///
    /// # Errors
    /// Rejects unsupported inputs, malformed extents, allocation failure or a
    /// running sum requiring rounding before any native submission is possible.
    pub fn new(
        projection: Q8Matmul,
        weights: &'a [u8],
        activation: &'a [f32],
    ) -> Result<Self, Error> {
        Self::new_with_control(projection, weights, activation, || Ok(()))
    }

    /// Evaluate exact dots under the original cancellation/deadline policy.
    ///
    /// `check` follows [`Q8ReferenceOracle::new_with_control`]'s inherited-clock
    /// contract and runs at most 32 arithmetic lanes apart. Refusal returns no
    /// partial oracle. [`Self::new`] supplies no cancellation or time bound.
    ///
    /// # Errors
    /// Propagates control refusal and every input/arithmetic error of [`Self::new`].
    pub fn new_with_control(
        projection: Q8Matmul,
        weights: &'a [u8],
        activation: &'a [f32],
        mut check: impl FnMut() -> Result<(), Error>,
    ) -> Result<Self, Error> {
        check()?;
        if weights.len() != projection.weight_bytes()
            || activation.len() != projection.columns() * projection.batch()
        {
            return Err(Error::Contract);
        }
        for block in activation.as_chunks::<32>().0 {
            check()?;
            if !block.iter().all(|&value| {
                value.is_finite() && (-127.0..=127.0).contains(&value) && value.fract() == 0.0
            }) || !block.iter().any(|value| value.abs() == 127.0)
            {
                return Err(Error::Contract);
            }
        }
        for block in weights.as_chunks::<34>().0 {
            check()?;
            scale(u16::from_le_bytes([block[0], block[1]]))?;
        }
        check()?;
        let count = projection.rows() * projection.batch();
        let mut expected = Vec::new();
        expected
            .try_reserve_exact(count)
            .map_err(|_| Error::Contract)?;
        let blocks = projection.columns() / 32;
        for vector in activation.chunks_exact(projection.columns()) {
            for row in weights.chunks_exact(blocks * 34) {
                // Units are 2^-8. Even K8192, i8 extremes and the maximum
                // admitted scale keep the absolute sum below 2^44, within i64.
                let mut units = 0_i64;
                let mut value = 0.0;
                for (weight, input) in row
                    .as_chunks::<34>()
                    .0
                    .iter()
                    .zip(vector.as_chunks::<32>().0)
                {
                    check()?;
                    let mut dot = 0_i32;
                    for (&code, &input) in weight[2..].iter().zip(input) {
                        dot += i32::from(i8::from_ne_bytes([code])) * exact_input(input);
                    }
                    let multiplier = scale(u16::from_le_bytes([weight[0], weight[1]]))?;
                    units += i64::from(dot) * multiplier;
                    value = exact_sum(units)?;
                }
                expected.push(value);
            }
        }
        check()?;
        Ok(Self {
            weights,
            activation,
            expected: expected.into_boxed_slice(),
        })
    }

    /// Require every returned bit, including zero sign, to match the exact dots.
    ///
    /// # Errors
    /// Rejects changed extents or any mismatching/nonfinite returned value.
    pub fn review(&self, output: &[f32]) -> Result<(), Error> {
        self.review_with_control(output, || Ok(()))
    }

    /// Review every exact result under the original caller-owned work allowance.
    ///
    /// `check` follows [`Self::new_with_control`]'s contract, including a final
    /// check after every output bit has matched.
    ///
    /// # Errors
    /// Propagates control refusal and the numerical errors of [`Self::review`].
    pub fn review_with_control(
        &self,
        output: &[f32],
        mut check: impl FnMut() -> Result<(), Error>,
    ) -> Result<(), Error> {
        check()?;
        if output.len() != self.expected.len() {
            return Err(Error::Contract);
        }
        for (actual, expected) in output.iter().zip(&self.expected) {
            check()?;
            if actual.to_bits() != expected.to_bits() {
                return Err(Error::Contract);
            }
        }
        check()
    }

    /// Submit the same immutable inputs with mandatory independent numeric review.
    ///
    /// This supplies no native authority, retry, fallback, deadline or teardown
    /// policy; the consuming transport retains all those original obligations.
    ///
    /// # Errors
    /// Propagates transport, cancellation, bound or independent numeric failure.
    pub fn submit(&self, transport: &mut dyn Q8Submitter, output: &mut [f32]) -> Result<(), Error> {
        self.submit_with_control(transport, output, || Ok(()))
    }

    /// Submit unchanged inputs with bounded exact review before publication.
    ///
    /// `check` follows [`Self::new_with_control`]'s contract. Refusal before the
    /// call skips transport; refusal inside review is a terminal native error.
    /// Native deadlines and uncertain-buffer ownership remain the transport's.
    ///
    /// # Errors
    /// Propagates control, transport and exact numerical review failures.
    pub fn submit_with_control(
        &self,
        transport: &mut dyn Q8Submitter,
        output: &mut [f32],
        mut check: impl FnMut() -> Result<(), Error>,
    ) -> Result<(), Error> {
        check()?;
        transport.submit(self.weights, self.activation, output, &mut |actual| {
            self.review_with_control(actual, &mut check)
        })
    }
}

fn scale(bits: u16) -> Result<i64, Error> {
    let magnitude = bits & 0x7fff;
    if magnitude == 0 {
        return Ok(0);
    }
    let exponent = magnitude >> 10;
    if magnitude & 0x3ff != 0 || !(7..=23).contains(&exponent) {
        return Err(Error::Contract);
    }
    let units = 1_i64 << (exponent - 7);
    Ok(if bits & 0x8000 == 0 { units } else { -units })
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "Caller already admitted finite integral activation values in -127..127"
)]
fn exact_input(value: f32) -> i32 {
    value as i32
}

#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "Every bounded i64-to-F32 conversion is checked by an exact round trip before admission"
)]
fn exact_sum(units: i64) -> Result<f32, Error> {
    let rounded = units as f32;
    if rounded as i64 != units {
        return Err(Error::Contract);
    }
    // Exact normal power-of-two scaling; zero remains positive as in the
    // DSP accumulator initialized to positive zero before its first addition.
    Ok(rounded * (1.0 / 256.0))
}

const F16_ONE_OVER_127_BITS: u16 = 0x2008;

/// Independent HVX `0x2008` half-quantization and Q8 projection oracle.
///
/// Models the `0x2008` (`129/16384`) binary16 activation scale and reciprocal
/// stage of the pinned `quantize_block_f32_q8_0_tiled` kernel on each 32-value
/// F32 group: IEEE-754 binary16 RNE narrowing, `amax * 0x2008` binary16 RNE
/// scale `d` (zero only when narrowed `amax == 0`), IEEE-754 binary16 RNE
/// reciprocal `1/d` and lane products, and RNE clamped signed `i8` codes
/// (`-127..=127`). Outputs are compared against ordered `F64` block
/// accumulation (`w_q * a_q * (d_w * d_a)`) narrowed to `F32` within fixed
/// caller-reviewed finite tolerances. This does not emulate HVX `QF16`/`QF32`,
/// the `hvx_vec_inverse_f16` LUT polynomial, `hvx_vec_i16_from_hf_rnd_sat`, or
/// per-block `F32` reduction, and grants no device or speech acceptance.
pub struct Q8HvxQuantizationOracle<'a> {
    projection: Q8Matmul,
    weights: &'a [u8],
    activation: &'a [f32],
    absolute_tolerance: f64,
    relative_tolerance: f64,
    expected: Box<[f32]>,
    quantized_scales: Box<[u16]>,
    quantized_codes: Box<[i8]>,
}

impl<'a> Q8HvxQuantizationOracle<'a> {
    /// Validate inputs, quantize activations through the pinned `0x2008`
    /// binary16 RNE model, and prepare expected projections without a budget.
    ///
    /// # Errors
    /// Rejects non-finite/overflowing inputs, invalid tolerances, nonzero-`amax`
    /// scale underflow or subreciprocal overflow (`d < 0x0101`), allocation
    /// failure, unrepresentable F32 expected values, and mismatched extents.
    pub fn new(
        projection: Q8Matmul,
        weights: &'a [u8],
        activation: &'a [f32],
        absolute_tolerance: f64,
        relative_tolerance: f64,
    ) -> Result<Self, Error> {
        Self::new_with_control(
            projection,
            weights,
            activation,
            absolute_tolerance,
            relative_tolerance,
            || Ok(()),
        )
    }

    /// Validate inputs and prepare expected HVX-quantized projections while
    /// polling the caller-owned deadline and cancellation control callback.
    ///
    /// `check` follows [`Q8ReferenceOracle::new_with_control`]'s inherited-clock
    /// contract, polling before intake/allocation, once per 32-lane block and
    /// before return. Refusal returns no partially built oracle.
    ///
    /// # Errors
    /// Rejects control cancellation/expiry, malformed inputs, F16 overflow,
    /// nonzero-`amax` scale underflow or subreciprocal overflow (`d < 0x0101`),
    /// allocation failure, and unrepresentable F32 expected projections before
    /// native submission.
    #[allow(
        clippy::suboptimal_flops,
        reason = "The independent quantized reference preserves separate block scale product and addition rounding"
    )]
    pub fn new_with_control(
        projection: Q8Matmul,
        weights: &'a [u8],
        activation: &'a [f32],
        absolute_tolerance: f64,
        relative_tolerance: f64,
        mut check: impl FnMut() -> Result<(), Error>,
    ) -> Result<Self, Error> {
        check()?;
        if weights.len() != projection.weight_bytes()
            || activation.len() != projection.columns() * projection.batch()
            || !absolute_tolerance.is_finite()
            || !(0.0..=f64::from(f32::MAX)).contains(&absolute_tolerance)
            || !relative_tolerance.is_finite()
            || !(0.0..=1.0).contains(&relative_tolerance)
        {
            return Err(Error::Contract);
        }
        for block in activation.as_chunks::<32>().0 {
            check()?;
            if block.iter().any(|value| !value.is_finite()) {
                return Err(Error::Contract);
            }
        }
        for block in weights.as_chunks::<34>().0 {
            check()?;
            reference_scale(u16::from_le_bytes([block[0], block[1]]))?;
        }
        check()?;
        let blocks = projection.columns() / 32;
        let row_bytes = blocks * 34;
        let total_act_blocks = projection.batch() * blocks;
        let mut quantized_scales = Vec::new();
        quantized_scales
            .try_reserve_exact(total_act_blocks)
            .map_err(|_| Error::Contract)?;
        let mut quantized_codes = Vec::new();
        quantized_codes
            .try_reserve_exact(activation.len())
            .map_err(|_| Error::Contract)?;
        let mut expected = Vec::new();
        expected
            .try_reserve_exact(projection.rows() * projection.batch())
            .map_err(|_| Error::Contract)?;
        let inv_127 = reference_scale(F16_ONE_OVER_127_BITS)?;
        for group in activation.as_chunks::<32>().0 {
            check()?;
            let mut half_lanes = [0_u16; 32];
            let mut amax = 0.0_f64;
            for (index, &value) in group.iter().enumerate() {
                let h = f32_to_f16_bits(value)?;
                half_lanes[index] = h;
                amax = amax.max(reference_scale(h & 0x7fff)?);
            }
            let (d_bits, id_bits) = if amax == 0.0 {
                (0_u16, 0_u16)
            } else {
                let d_bits = f64_to_f16_bits(amax * inv_127)?;
                let d_scale = reference_scale(d_bits)?;
                if d_scale == 0.0 {
                    return Err(Error::Contract);
                }
                let id_bits = f64_to_f16_bits(1.0 / d_scale)?;
                (d_bits, id_bits)
            };
            quantized_scales.push(d_bits);
            for h in half_lanes {
                if id_bits == 0 {
                    quantized_codes.push(0);
                } else {
                    let scaled_bits =
                        f64_to_f16_bits(reference_scale(h)? * reference_scale(id_bits)?)?;
                    quantized_codes.push(quantized_code(reference_scale(scaled_bits)?));
                }
            }
        }
        for vector in 0..projection.batch() {
            let act_scale_base = vector * blocks;
            let act_code_base = vector * projection.columns();
            for row in 0..projection.rows() {
                let row_weights = &weights[row * row_bytes..(row + 1) * row_bytes];
                let mut total = 0.0_f64;
                for block in 0..blocks {
                    check()?;
                    let weight_block = &row_weights[block * 34..(block + 1) * 34];
                    let weight_scale =
                        reference_scale(u16::from_le_bytes([weight_block[0], weight_block[1]]))?;
                    let act_scale = reference_scale(quantized_scales[act_scale_base + block])?;
                    let act_codes = &quantized_codes
                        [act_code_base + block * 32..act_code_base + (block + 1) * 32];
                    let mut dot = 0_i32;
                    for (lane, &weight_byte) in weight_block[2..34].iter().enumerate() {
                        let w = i32::from(i8::from_ne_bytes([weight_byte]));
                        let a = i32::from(act_codes[lane]);
                        dot += w * a;
                    }
                    total += f64::from(dot) * (weight_scale * act_scale);
                    if !total.is_finite() {
                        return Err(Error::Contract);
                    }
                }
                expected.push(narrow_expected(total)?);
            }
        }
        check()?;
        Ok(Self {
            projection,
            weights,
            activation,
            absolute_tolerance,
            relative_tolerance,
            expected: expected.into_boxed_slice(),
            quantized_scales: quantized_scales.into_boxed_slice(),
            quantized_codes: quantized_codes.into_boxed_slice(),
        })
    }

    /// Exact selected matrix and immutable solver mapping.
    #[must_use]
    pub const fn projection(&self) -> Q8Matmul {
        self.projection
    }

    /// Prepared `0x2008`-scaled binary16 block scale bits, one `u16` per 32-lane block.
    #[must_use]
    pub const fn quantized_scales(&self) -> &[u16] {
        &self.quantized_scales
    }

    /// Prepared RNE clamped signed `i8` activation codes after half reciprocal scaling.
    #[must_use]
    pub const fn quantized_codes(&self) -> &[i8] {
        &self.quantized_codes
    }

    /// Expected F32 projection values from the quantized activation model.
    #[must_use]
    pub const fn expected(&self) -> &[f32] {
        &self.expected
    }

    /// Review complete returned F32 projection outputs against fixed tolerances.
    ///
    /// # Errors
    /// Rejects shape mismatches, non-finite outputs and tolerance violations.
    pub fn review(&self, output: &[f32]) -> Result<(), Error> {
        self.review_with_control(output, || Ok(()))
    }

    /// Review complete returned F32 projection outputs while polling the control callback.
    ///
    /// # Errors
    /// Rejects cancellation/expiry, malformed extents, non-finite outputs and tolerance violations.
    #[allow(
        clippy::suboptimal_flops,
        reason = "The declared tolerance formula preserves separate multiply/add rounding"
    )]
    pub fn review_with_control(
        &self,
        output: &[f32],
        mut check: impl FnMut() -> Result<(), Error>,
    ) -> Result<(), Error> {
        check()?;
        if output.len() != self.expected.len() {
            return Err(Error::Contract);
        }
        for (&actual, &expected) in output.iter().zip(&self.expected) {
            check()?;
            if !actual.is_finite() {
                return Err(Error::Contract);
            }
            let delta = (f64::from(actual) - f64::from(expected)).abs();
            let limit =
                self.absolute_tolerance + self.relative_tolerance * f64::from(expected).abs();
            if delta > limit {
                return Err(Error::Contract);
            }
        }
        check()
    }

    /// Submit original borrowed inputs and publish outputs only after review passes.
    ///
    /// # Errors
    /// Rejects transport failure or failed output review before publication.
    pub fn submit(&self, transport: &mut dyn Q8Submitter, output: &mut [f32]) -> Result<(), Error> {
        self.submit_with_control(transport, output, || Ok(()))
    }

    /// Submit original borrowed inputs with bounded control polling through review.
    ///
    /// # Errors
    /// Rejects early cancellation/expiry, transport refusal or failed review before publication.
    pub fn submit_with_control(
        &self,
        transport: &mut dyn Q8Submitter,
        output: &mut [f32],
        mut check: impl FnMut() -> Result<(), Error>,
    ) -> Result<(), Error> {
        check()?;
        let mut oracle = |candidate: &[f32]| self.review_with_control(candidate, &mut check);
        transport.submit(self.weights, self.activation, output, &mut oracle)
    }
}

#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Finite binary16 scaled lane is rounded ties-to-even and clamped to -127..=127"
)]
fn quantized_code(scaled: f64) -> i8 {
    scaled.round_ties_even().clamp(-127.0, 127.0) as i8
}

#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Finite F64 reference within +/-f32::MAX narrows to IEEE-754 F32 RNE"
)]
fn narrow_expected(total: f64) -> Result<f32, Error> {
    if !total.is_finite() || total.abs() > f64::from(f32::MAX) {
        return Err(Error::Contract);
    }
    Ok(total as f32)
}

fn f32_to_f16_bits(value: f32) -> Result<u16, Error> {
    if !value.is_finite() {
        return Err(Error::Contract);
    }
    let bits = value.to_bits();
    let sign = if (bits & 0x8000_0000) != 0 {
        0x8000_u16
    } else {
        0
    };
    let exp = i32::try_from((bits >> 23) & 0xff).map_err(|_| Error::Contract)?;
    let frac = bits & 0x7f_ffff;
    if exp == 0 {
        return Ok(sign);
    }
    let mut unbiased = exp - 127;
    if unbiased >= -14 {
        let mut mant = frac >> 13;
        let rem = frac & 0x1fff;
        if rem > 0x1000 || (rem == 0x1000 && (mant & 1) == 1) {
            mant += 1;
            if mant == 1024 {
                mant = 0;
                unbiased += 1;
            }
        }
        if unbiased > 15 {
            return Err(Error::Contract);
        }
        let exp_bits = u16::try_from(unbiased + 15).map_err(|_| Error::Contract)?;
        let mant_bits = u16::try_from(mant).map_err(|_| Error::Contract)?;
        return Ok(sign | (exp_bits << 10) | mant_bits);
    }
    if unbiased < -25 {
        return Ok(sign);
    }
    let sig = 0x80_0000 | frac;
    let shift = u32::try_from(-14 - unbiased + 13).map_err(|_| Error::Contract)?;
    let half = 1_u32 << (shift - 1);
    let mask = (1_u32 << shift) - 1;
    let mut mant = sig >> shift;
    let rem = sig & mask;
    if rem > half || (rem == half && (mant & 1) == 1) {
        mant += 1;
    }
    if mant == 1024 {
        return Ok(sign | 0x0400);
    }
    let mant_bits = u16::try_from(mant).map_err(|_| Error::Contract)?;
    Ok(sign | mant_bits)
}

fn f64_to_f16_bits(value: f64) -> Result<u16, Error> {
    if !value.is_finite() {
        return Err(Error::Contract);
    }
    // f64 inputs here exceed binary16 precision only beyond 11 bits; converting
    // directly using IEEE-754 RNE on the 53-bit significand avoids double rounding.
    let bits = value.to_bits();
    let sign = if (bits & 0x8000_0000_0000_0000) != 0 {
        0x8000_u16
    } else {
        0
    };
    let exp = i32::try_from((bits >> 52) & 0x7ff).map_err(|_| Error::Contract)?;
    let frac = bits & 0xf_ffff_ffff_ffff;
    if exp == 0 {
        return Ok(sign);
    }
    let mut unbiased = exp - 1023;
    if unbiased >= -14 {
        let mut mant = u32::try_from(frac >> 42).map_err(|_| Error::Contract)?;
        let rem = frac & 0x3ff_ffff_ffff;
        let half = 0x200_0000_0000_u64;
        if rem > half || (rem == half && (mant & 1) == 1) {
            mant += 1;
            if mant == 1024 {
                mant = 0;
                unbiased += 1;
            }
        }
        if unbiased > 15 {
            return Err(Error::Contract);
        }
        let exp_bits = u16::try_from(unbiased + 15).map_err(|_| Error::Contract)?;
        let mant_bits = u16::try_from(mant).map_err(|_| Error::Contract)?;
        return Ok(sign | (exp_bits << 10) | mant_bits);
    }
    if unbiased < -25 {
        return Ok(sign);
    }
    let sig = 0x10_0000_0000_0000_u64 | frac;
    let shift = u32::try_from(-14 - unbiased + 42).map_err(|_| Error::Contract)?;
    let half = 1_u64 << (shift - 1);
    let mask = (1_u64 << shift) - 1;
    let mut mant = u32::try_from(sig >> shift).map_err(|_| Error::Contract)?;
    let rem = sig & mask;
    if rem > half || (rem == half && (mant & 1) == 1) {
        mant += 1;
    }
    if mant == 1024 {
        return Ok(sign | 0x0400);
    }
    let mant_bits = u16::try_from(mant).map_err(|_| Error::Contract)?;
    Ok(sign | mant_bits)
}

#[cfg(test)]
#[path = "q8_oracle_tests.rs"]
mod tests;
