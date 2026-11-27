//! Safe public boundary for ARM64 `SDOT` / `I8MM` / `FCVTL` / `FCVTN` quantized
//! kernels with runtime CPU feature detection and exact portable fallbacks.

pub use crate::native::quant_q8::{
    Q8_0ActPanel8, Q8_KActPanel8, REPACKED_Q8_0_8_BLOCK_BYTES, RowMajorBand,
};
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

/// Computes `strip_count` contiguous 8-row repacked `Q4_K` strips starting at `start_strip`
/// for load-validated finite scales with cross-strip prefetching (`0-ULP` vs `dot_q4_k_row`).
pub fn dot_q4_k_repacked8_finite_scales_strips(
    caps: Arm64QuantCaps,
    data: &[u8],
    start_strip: usize,
    strip_count: usize,
    act: &[Q8_KActBlock],
    out: &mut [f32],
) {
    crate::native::quant_q8::dot_q4_k_repacked8_finite_scales_strips(
        caps,
        data,
        start_strip,
        strip_count,
        act,
        out,
    );
}

/// Computes eight `Q8_0` row dot products while sharing activations per block (`0-ULP` vs `dot_q8_0_row`).
#[must_use]
pub fn dot_q8_0_8rows(caps: Arm64QuantCaps, rows: [&[u8]; 8], act: &[Q8_0ActBlock]) -> [f32; 8] {
    crate::native::quant_q8::dot_q8_0_8rows(caps, rows, act)
}

/// Computes eight load-validated finite-scale `Q8_0` row dot products using 2-block unrolling
/// and direct `vld1q_lane_u16` + `vcvt_f32_f16` (`0-ULP` vs `dot_q8_0_row`).
#[must_use]
pub fn dot_q8_0_8rows_finite_scales(
    caps: Arm64QuantCaps,
    rows: [&[u8]; 8],
    act: &[Q8_0ActBlock],
) -> [f32; 8] {
    crate::native::quant_q8::dot_q8_0_8rows_finite_scales(caps, rows, act)
}

/// Computes `strip_count` contiguous 8-row strips (`strip_count * 8` rows starting at `start_row`)
/// of load-validated finite-scale `Q8_0` dot products with cross-strip prefetching (`0-ULP` vs `dot_q8_0_row`).
pub fn dot_q8_0_8rows_finite_scales_strips(
    caps: Arm64QuantCaps,
    data: &[u8],
    start_row: usize,
    strip_count: usize,
    act: &[Q8_0ActBlock],
    out: &mut [f32],
) {
    crate::native::quant_q8::dot_q8_0_8rows_finite_scales_strips(
        caps,
        data,
        start_row,
        strip_count,
        act,
        out,
    );
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

/// Pre-packs `panel_count * 8` tokens of `Q8_0ActBlock`s (`acts`) into `Q8_0ActPanel8`s
/// laid out as `[panel_idx * blocks + k]`.
pub fn pack_q8_0_act_panels8(
    acts: &[Q8_0ActBlock],
    blocks: usize,
    panel_count: usize,
    panels: &mut [Q8_0ActPanel8],
) {
    crate::native::quant_q8::pack_q8_0_act_panels8(acts, blocks, panel_count, panels);
}

/// Computes an 8-row by `(panel_count * 8)`-token `Q8_0` GEMM over pre-packed `Q8_0ActPanel8`s
/// with K-blocked L1D weight tiles (`I8MM` `SMMLA`, or exact `SDOT`/portable fallback).
pub fn gemm_q8_0_8x_panels(
    caps: Arm64QuantCaps,
    rows: [&[u8]; 8],
    panels: &[Q8_0ActPanel8],
    panel_count: usize,
    blocks: usize,
    out_rows: &mut [f32],
    row_stride: usize,
) {
    crate::native::quant_q8::gemm_q8_0_8x_panels(
        caps,
        rows,
        panels,
        panel_count,
        blocks,
        out_rows,
        row_stride,
    );
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

/// Pre-packs `panel_count * 8` tokens of `Q8_KActBlock`s (`acts`) into `Q8_KActPanel8`s
/// laid out as `[panel_idx * blocks + k]`.
pub fn pack_q8_k_act_panels8(
    acts: &[Q8_KActBlock],
    blocks: usize,
    panel_count: usize,
    panels: &mut [Q8_KActPanel8],
) {
    crate::native::quant_q8::pack_q8_k_act_panels8(acts, blocks, panel_count, panels);
}

/// Computes an 8-row by `(panel_count * 8)`-token repacked `Q4_K` strip GEMM over pre-packed
/// `Q8_KActPanel8`s with K-blocked L1D weight tiles (`I8MM` `SMMLA`, or exact `SDOT`/portable fallback).
pub fn gemm_q4_k_repacked8_8x_panels(
    caps: Arm64QuantCaps,
    strip: &[u8],
    panels: &[Q8_KActPanel8],
    panel_count: usize,
    blocks: usize,
    out_rows: &mut [f32],
    row_stride: usize,
) {
    crate::native::quant_q8::gemm_q4_k_repacked8_8x_panels(
        caps,
        strip,
        panels,
        panel_count,
        blocks,
        out_rows,
        row_stride,
    );
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

/// Computes an 8-row by `(panel_count * 8)`-token `Q6_K` GEMM over pre-packed `Q8_KActPanel8`s
/// with K-blocked L1D weight tiles (`I8MM` `SMMLA`, or exact `SDOT`/portable fallback).
pub fn gemm_q6_k_8x_panels(
    caps: Arm64QuantCaps,
    rows: [&[u8]; 8],
    panels: &[Q8_KActPanel8],
    panel_count: usize,
    blocks: usize,
    out_rows: &mut [f32],
    row_stride: usize,
) {
    crate::native::quant_q8::gemm_q6_k_8x_panels(
        caps,
        rows,
        panels,
        panel_count,
        blocks,
        out_rows,
        row_stride,
    );
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

/// Accumulates a 4-row by 16-column strictly non-FMA `f32` GEMM tile
/// (`acc[r][c] += sum_k rows[r][k] * act_panel[k][c]`) in exact left-to-right `k` order
/// using ARM64 NEON `vmulq_laneq_f32` + `vaddq_f32` (`0-ULP` vs scalar).
pub fn gemm_f32_nonfma_4x16(rows: [&[f32]; 4], act_panel: &[[f32; 16]], acc: &mut [[f32; 16]; 4]) {
    crate::native::quant_q8::gemm_f32_nonfma_4x16(rows, act_panel, acc);
}

/// Partitions a `[batch, output]` row-major output slice into disjoint mutable column bands.
///
/// Returns `None` if `out.len() != batch * output` or `band_cols.iter().sum() != output`.
#[must_use]
pub fn split_row_major_bands<'a>(
    out: &'a mut [f32],
    batch: usize,
    output: usize,
    band_cols: &[usize],
) -> Option<Vec<RowMajorBand<'a>>> {
    crate::native::quant_q8::split_row_major_bands(out, batch, output, band_cols)
}

/// Repacks an 8-row-aligned `Q8_0` weight matrix (`out_dim % 8 == 0`, `in_dim % 32 == 0`)
/// into contiguous 8-row strip blocks (`REPACKED_Q8_0_8_BLOCK_BYTES = 272` bytes each).
///
/// Returns `None` if dimensions are unaligned, `src.len()` mismatches, or any `binary16`
/// block scale is non-finite.
#[must_use]
pub fn repack_q8_0_8rows(src: &[u8], in_dim: usize, out_dim: usize) -> Option<Vec<u8>> {
    crate::native::quant_q8::repack_q8_0_8rows(src, in_dim, out_dim)
}

/// Computes an 8-row repacked finite-scale `Q8_0` strip dot product (`0-ULP` vs `dot_q8_0_row`).
#[must_use]
pub fn dot_repacked_q8_0_8_finite_scales(
    caps: Arm64QuantCaps,
    strip: &[u8],
    act: &[Q8_0ActBlock],
) -> [f32; 8] {
    crate::native::quant_q8::dot_repacked_q8_0_8_finite_scales(caps, strip, act)
}

/// Computes `strip_count` contiguous 8-row repacked finite-scale `Q8_0` strips starting at
/// `start_strip` with cross-strip linear prefetching (`0-ULP` vs `dot_q8_0_row`).
pub fn dot_repacked_q8_0_8_finite_scales_strips(
    caps: Arm64QuantCaps,
    data: &[u8],
    start_strip: usize,
    strip_count: usize,
    act: &[Q8_0ActBlock],
    out: &mut [f32],
) {
    crate::native::quant_q8::dot_repacked_q8_0_8_finite_scales_strips(
        caps,
        data,
        start_strip,
        strip_count,
        act,
        out,
    );
}

/// Computes an 8-row by 8-token repacked `Q8_0` strip tile (`I8MM` `SMMLA` or exact fallback).
#[must_use]
pub fn gemm_repacked_q8_0_8_8x8(
    caps: Arm64QuantCaps,
    strip: &[u8],
    acts: [&[Q8_0ActBlock]; 8],
) -> [[f32; 8]; 8] {
    crate::native::quant_q8::gemm_repacked_q8_0_8_8x8(caps, strip, acts)
}

/// Computes an 8-row by 4-token repacked `Q8_0` strip tile (`I8MM` `SMMLA` or exact fallback).
#[must_use]
pub fn gemm_repacked_q8_0_8_8x4(
    caps: Arm64QuantCaps,
    strip: &[u8],
    acts: [&[Q8_0ActBlock]; 4],
) -> [[f32; 4]; 8] {
    crate::native::quant_q8::gemm_repacked_q8_0_8_8x4(caps, strip, acts)
}

/// Computes an 8-row by 2-token repacked `Q8_0` strip tile (`I8MM` `SMMLA` or exact fallback).
#[must_use]
pub fn gemm_repacked_q8_0_8_8x2(
    caps: Arm64QuantCaps,
    strip: &[u8],
    acts: [&[Q8_0ActBlock]; 2],
) -> [[f32; 2]; 8] {
    crate::native::quant_q8::gemm_repacked_q8_0_8_8x2(caps, strip, acts)
}

/// Computes an 8-row by `(panel_count * 8)`-token finite-scale `Q8_0` GEMM over pre-packed
/// `Q8_0ActPanel8`s (`0-ULP` vs `dot_q8_0_row`).
pub fn gemm_q8_0_8x_panels_finite_scales(
    caps: Arm64QuantCaps,
    rows: [&[u8]; 8],
    panels: &[Q8_0ActPanel8],
    panel_count: usize,
    blocks: usize,
    out_rows: &mut [f32],
    row_stride: usize,
) {
    crate::native::quant_q8::gemm_q8_0_8x_panels_finite_scales(
        caps,
        rows,
        panels,
        panel_count,
        blocks,
        out_rows,
        row_stride,
    );
}

/// Computes an 8-row by `(panel_count * 8)`-token `Q8_0` panel GEMM, scattering directly
/// into `[token * out_stride .. token * out_stride + 8]` with optional bias addition.
///
/// Returns `true` when every written output element is finite.
pub fn gemm_q8_0_8x_panels_scatter_biased<const FINITE_SCALES: bool>(
    caps: Arm64QuantCaps,
    rows: [&[u8]; 8],
    panels: &[Q8_0ActPanel8],
    panel_count: usize,
    blocks: usize,
    out_tokens: &mut [f32],
    out_stride: usize,
    bias: Option<[f32; 8]>,
) -> bool {
    crate::native::quant_q8::gemm_q8_0_8x_panels_scatter_biased::<FINITE_SCALES>(
        caps,
        rows,
        panels,
        panel_count,
        blocks,
        out_tokens,
        out_stride,
        bias,
    )
}

/// Computes an 8-row by `(panel_count * 8)`-token `Q8_0` panel GEMM into a disjoint
/// `RowMajorBand` slice at `local_col` with optional bias addition.
///
/// Returns `true` when every written output element is finite.
pub fn gemm_q8_0_8x_panels_into_band<const FINITE_SCALES: bool>(
    caps: Arm64QuantCaps,
    rows: [&[u8]; 8],
    panels: &[Q8_0ActPanel8],
    panel_count: usize,
    blocks: usize,
    band: &mut RowMajorBand<'_>,
    local_col: usize,
    bias: Option<[f32; 8]>,
) -> bool {
    crate::native::quant_q8::gemm_q8_0_8x_panels_into_band::<FINITE_SCALES>(
        caps,
        rows,
        panels,
        panel_count,
        blocks,
        band,
        local_col,
        bias,
    )
}

/// Computes an 8-row by `(panel_count * 8)`-token repacked finite-scale `Q8_0` GEMM
/// over pre-packed `Q8_0ActPanel8`s (`0-ULP` vs `dot_q8_0_row`).
pub fn gemm_repacked_q8_0_8_8x_panels_finite_scales(
    caps: Arm64QuantCaps,
    strip: &[u8],
    panels: &[Q8_0ActPanel8],
    panel_count: usize,
    blocks: usize,
    out_rows: &mut [f32],
    row_stride: usize,
) {
    crate::native::quant_q8::gemm_repacked_q8_0_8_8x_panels_finite_scales(
        caps,
        strip,
        panels,
        panel_count,
        blocks,
        out_rows,
        row_stride,
    );
}

/// Computes an 8-row by `(panel_count * 8)`-token repacked finite-scale `Q8_0` GEMM,
/// scattering directly into `[token * out_stride .. token * out_stride + 8]` with optional bias.
///
/// Returns `true` when every written output element is finite.
pub fn gemm_repacked_q8_0_8_8x_panels_finite_scales_scatter_biased(
    caps: Arm64QuantCaps,
    strip: &[u8],
    panels: &[Q8_0ActPanel8],
    panel_count: usize,
    blocks: usize,
    out_tokens: &mut [f32],
    out_stride: usize,
    bias: Option<[f32; 8]>,
) -> bool {
    crate::native::quant_q8::gemm_repacked_q8_0_8_8x_panels_finite_scales_scatter_biased(
        caps,
        strip,
        panels,
        panel_count,
        blocks,
        out_tokens,
        out_stride,
        bias,
    )
}

/// Computes an 8-row by `(panel_count * 8)`-token repacked finite-scale `Q8_0` GEMM
/// into a disjoint `RowMajorBand` slice at `local_col` with optional bias.
///
/// Returns `true` when every written output element is finite.
pub fn gemm_repacked_q8_0_8_8x_panels_finite_scales_into_band(
    caps: Arm64QuantCaps,
    strip: &[u8],
    panels: &[Q8_0ActPanel8],
    panel_count: usize,
    blocks: usize,
    band: &mut RowMajorBand<'_>,
    local_col: usize,
    bias: Option<[f32; 8]>,
) -> bool {
    crate::native::quant_q8::gemm_repacked_q8_0_8_8x_panels_finite_scales_into_band(
        caps,
        strip,
        panels,
        panel_count,
        blocks,
        band,
        local_col,
        bias,
    )
}

/// Computes an 8-row by `(panel_count * 8)`-token repacked finite-scale `Q4_K` strip GEMM
/// over pre-packed `Q8_KActPanel8`s (`0-ULP` vs `dot_q4_k_row`).
pub fn gemm_q4_k_repacked8_8x_panels_finite_scales(
    caps: Arm64QuantCaps,
    strip: &[u8],
    panels: &[Q8_KActPanel8],
    panel_count: usize,
    blocks: usize,
    out_rows: &mut [f32],
    row_stride: usize,
) {
    crate::native::quant_q8::gemm_q4_k_repacked8_8x_panels_finite_scales(
        caps,
        strip,
        panels,
        panel_count,
        blocks,
        out_rows,
        row_stride,
    );
}

/// Computes an 8-row by `(panel_count * 8)`-token repacked `Q4_K` strip GEMM, scattering directly
/// into `[token * out_stride .. token * out_stride + 8]` with optional bias addition.
///
/// Returns `true` when every written output element is finite.
pub fn gemm_q4_k_repacked8_8x_panels_scatter_biased<const FINITE_SCALES: bool>(
    caps: Arm64QuantCaps,
    strip: &[u8],
    panels: &[Q8_KActPanel8],
    panel_count: usize,
    blocks: usize,
    out_tokens: &mut [f32],
    out_stride: usize,
    bias: Option<[f32; 8]>,
) -> bool {
    crate::native::quant_q8::gemm_q4_k_repacked8_8x_panels_scatter_biased::<FINITE_SCALES>(
        caps,
        strip,
        panels,
        panel_count,
        blocks,
        out_tokens,
        out_stride,
        bias,
    )
}

/// Computes an 8-row by `(panel_count * 8)`-token repacked `Q4_K` strip GEMM into a disjoint
/// `RowMajorBand` slice at `local_col` with optional bias addition.
///
/// Returns `true` when every written output element is finite.
pub fn gemm_q4_k_repacked8_8x_panels_into_band<const FINITE_SCALES: bool>(
    caps: Arm64QuantCaps,
    strip: &[u8],
    panels: &[Q8_KActPanel8],
    panel_count: usize,
    blocks: usize,
    band: &mut RowMajorBand<'_>,
    local_col: usize,
    bias: Option<[f32; 8]>,
) -> bool {
    crate::native::quant_q8::gemm_q4_k_repacked8_8x_panels_into_band::<FINITE_SCALES>(
        caps,
        strip,
        panels,
        panel_count,
        blocks,
        band,
        local_col,
        bias,
    )
}

/// Computes an 8-row by `(panel_count * 8)`-token finite-scale `Q6_K` GEMM over pre-packed
/// `Q8_KActPanel8`s (`0-ULP` vs `dot_q6_k_row`).
pub fn gemm_q6_k_8x_panels_finite_scales(
    caps: Arm64QuantCaps,
    rows: [&[u8]; 8],
    panels: &[Q8_KActPanel8],
    panel_count: usize,
    blocks: usize,
    out_rows: &mut [f32],
    row_stride: usize,
) {
    crate::native::quant_q8::gemm_q6_k_8x_panels_finite_scales(
        caps,
        rows,
        panels,
        panel_count,
        blocks,
        out_rows,
        row_stride,
    );
}

/// Computes an 8-row by `(panel_count * 8)`-token `Q6_K` panel GEMM, scattering directly
/// into `[token * out_stride .. token * out_stride + 8]` with optional bias addition.
///
/// Returns `true` when every written output element is finite.
pub fn gemm_q6_k_8x_panels_scatter_biased<const FINITE_SCALES: bool>(
    caps: Arm64QuantCaps,
    rows: [&[u8]; 8],
    panels: &[Q8_KActPanel8],
    panel_count: usize,
    blocks: usize,
    out_tokens: &mut [f32],
    out_stride: usize,
    bias: Option<[f32; 8]>,
) -> bool {
    crate::native::quant_q8::gemm_q6_k_8x_panels_scatter_biased::<FINITE_SCALES>(
        caps,
        rows,
        panels,
        panel_count,
        blocks,
        out_tokens,
        out_stride,
        bias,
    )
}

/// Computes an 8-row by `(panel_count * 8)`-token `Q6_K` panel GEMM into a disjoint
/// `RowMajorBand` slice at `local_col` with optional bias addition.
///
/// Returns `true` when every written output element is finite.
pub fn gemm_q6_k_8x_panels_into_band<const FINITE_SCALES: bool>(
    caps: Arm64QuantCaps,
    rows: [&[u8]; 8],
    panels: &[Q8_KActPanel8],
    panel_count: usize,
    blocks: usize,
    band: &mut RowMajorBand<'_>,
    local_col: usize,
    bias: Option<[f32; 8]>,
) -> bool {
    crate::native::quant_q8::gemm_q6_k_8x_panels_into_band::<FINITE_SCALES>(
        caps,
        rows,
        panels,
        panel_count,
        blocks,
        band,
        local_col,
        bias,
    )
}

/// Computes `(dot(q0, k), dot(q1, k))` from a shared load-validated finite `binary16` key
/// vector `k_f16`, skipping per-iteration non-finite lane guards (`0-ULP` vs `dot_f16_pair_f32`).
#[must_use]
pub fn dot_f16_pair_f32_finite(q0: &[f32], q1: &[f32], k_f16: &[u16]) -> (f32, f32) {
    crate::native::quant_q8::dot_f16_pair_f32_finite(q0, q1, k_f16)
}

/// Accumulates `out0 += w0 * v` and `out1 += w1 * v` from a shared load-validated finite `binary16`
/// value vector `v_f16`, skipping per-iteration non-finite lane guards (`0-ULP` vs `axpy_f16_pair_f32`).
pub fn axpy_f16_pair_f32_finite(
    w0: f32,
    w1: f32,
    v_f16: &[u16],
    out0: &mut [f32],
    out1: &mut [f32],
) {
    crate::native::quant_q8::axpy_f16_pair_f32_finite(w0, w1, v_f16, out0, out1);
}

/// Computes a 4-query by 4-key scaled dot-product tile `scores[qi][ki] = dot(q[qi], k[ki]) / divisor`
/// using ARM64 NEON FMA (`0-ULP` vs scalar FMA).
#[must_use]
pub fn dot_4x4_f32_fma(q: [&[f32]; 4], k: [&[f32]; 4], divisor: f32) -> [[f32; 4]; 4] {
    crate::native::quant_q8::dot_4x4_f32_fma(q, k, divisor)
}

/// Accumulates a 4-group by 4-value AXPY tile `out[g] += sum_t weights[g][t] * v[t]`
/// using ARM64 NEON FMA (`0-ULP` vs scalar FMA).
pub fn axpy_4x4_f32_fma(out: &mut [&mut [f32]; 4], v: [&[f32]; 4], weights: [[f32; 4]; 4]) {
    crate::native::quant_q8::axpy_4x4_f32_fma(out, v, weights);
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
                let df_nat = dot_q8_0_8rows_finite_scales(caps, row_refs, act8_refs[t]);
                let df_prt = dot_q8_0_8rows_finite_scales(port, row_refs, act8_refs[t]);
                assert_eq!(df_nat.map(f32::to_bits), expected.map(|row| row[t]));
                assert_eq!(df_prt.map(f32::to_bits), expected.map(|row| row[t]));
            }

            if blocks > 0 {
                let row_bytes = blocks * Q8_0_BLOCK_BYTES;
                let mut flat_16 = Vec::with_capacity(16 * row_bytes);
                for s in 0..2 {
                    for r in 0..8 {
                        if (r + s) % 2 == 0 {
                            flat_16.extend_from_slice(&make_edge_q8_0_row(blocks, r + s + 1));
                        } else {
                            flat_16.extend_from_slice(&make_q8_0_row(blocks, r + s + 9));
                        }
                    }
                }
                let mut strip_nat = [f32::NAN; 16];
                let mut strip_prt = [f32::NAN; 16];
                dot_q8_0_8rows_finite_scales_strips(
                    caps,
                    &flat_16,
                    0,
                    2,
                    act8_refs[0],
                    &mut strip_nat,
                );
                dot_q8_0_8rows_finite_scales_strips(
                    port,
                    &flat_16,
                    0,
                    2,
                    act8_refs[0],
                    &mut strip_prt,
                );
                for r in 0..16 {
                    let row = &flat_16[r * row_bytes..(r + 1) * row_bytes];
                    let want = dot_q8_0_row(port, row, act8_refs[0]).to_bits();
                    assert_eq!(strip_nat[r].to_bits(), want);
                    assert_eq!(strip_prt[r].to_bits(), want);
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

    #[test]
    fn q8_0_act_panels8_and_k_blocked_gemm_match_row_bit_exact() {
        let caps = Arm64QuantCaps::detect();
        let port = Arm64QuantCaps::portable();

        // Exercise K_TILE=32 and T_PANEL=8 boundaries across blocks and panel_count.
        for blocks in [0_usize, 1, 2, 31, 32, 33, 65] {
            let rows_vec: Vec<Vec<u8>> = (0..8)
                .map(|r| {
                    if r % 2 == 0 {
                        make_edge_q8_0_row(blocks, r + 3)
                    } else {
                        make_q8_0_row(blocks, r + 11)
                    }
                })
                .collect();
            let row_refs: [&[u8]; 8] = std::array::from_fn(|r| rows_vec[r].as_slice());

            for panel_count in [0_usize, 1, 2, 8, 9] {
                let tokens = panel_count * 8;
                let mut flat_acts = Vec::with_capacity(tokens * blocks);
                for t in 0..tokens {
                    flat_acts.extend(make_edge_q8_0_act(blocks, t));
                }
                let mut panels = vec![Q8_0ActPanel8::default(); panel_count * blocks];
                pack_q8_0_act_panels8(&flat_acts, blocks, panel_count, &mut panels);

                let stride = tokens + 3;
                let mut out_nat = vec![f32::NAN; 8 * stride];
                let mut out_prt = vec![f32::NAN; 8 * stride];
                gemm_q8_0_8x_panels(
                    caps,
                    row_refs,
                    &panels,
                    panel_count,
                    blocks,
                    &mut out_nat,
                    stride,
                );
                gemm_q8_0_8x_panels(
                    port,
                    row_refs,
                    &panels,
                    panel_count,
                    blocks,
                    &mut out_prt,
                    stride,
                );

                for r in 0..8 {
                    for t in 0..tokens {
                        let expected = dot_q8_0_row(
                            port,
                            row_refs[r],
                            &flat_acts[t * blocks..(t + 1) * blocks],
                        )
                        .to_bits();
                        let idx = r * stride + t;
                        assert_eq!(
                            out_nat[idx].to_bits(),
                            expected,
                            "nat mismatch blocks={blocks} panels={panel_count} r={r} t={t}"
                        );
                        assert_eq!(
                            out_prt[idx].to_bits(),
                            expected,
                            "prt mismatch blocks={blocks} panels={panel_count} r={r} t={t}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn gemm_f32_nonfma_4x16_matches_scalar_and_fma_sensitive_edges_bit_exact() {
        for k_len in [0_usize, 1, 3, 4, 5, 9, 17, 256, 259] {
            let rows_vec: [Vec<f32>; 4] = std::array::from_fn(|r| {
                (0..k_len)
                    .map(|k| match (r + k) % 9 {
                        0 => -0.0_f32,
                        1 => f32::from_bits(0x3f80_0001),
                        2 => 16_777_216.0_f32,
                        3 => -16_777_216.0_f32,
                        _ => {
                            (f32::from(u16::try_from((r * 13 + k * 7) % 41).unwrap_or(0)) - 20.0)
                                * 0.03125
                        }
                    })
                    .collect()
            });
            let row_refs: [&[f32]; 4] = std::array::from_fn(|r| rows_vec[r].as_slice());
            let panel: Vec<[f32; 16]> = (0..k_len)
                .map(|k| {
                    std::array::from_fn(|c| match (k + c) % 7 {
                        0 => -0.0_f32,
                        1 => 1.0_f32,
                        2 => f32::from_bits(0x3bff_ffff),
                        _ => {
                            (f32::from(u16::try_from((k * 17 + c * 5) % 53).unwrap_or(0)) - 26.0)
                                * 0.0625
                        }
                    })
                })
                .collect();

            let mut acc_nat = [[0.0_f32; 16]; 4];
            let mut acc_prt = [[0.0_f32; 16]; 4];
            gemm_f32_nonfma_4x16(row_refs, &panel, &mut acc_nat);
            crate::native::quant_q8::gemm_f32_nonfma_4x16_portable(row_refs, &panel, &mut acc_prt);

            for r in 0..4 {
                assert_eq!(
                    acc_nat[r].map(f32::to_bits),
                    acc_prt[r].map(f32::to_bits),
                    "k_len={k_len} r={r}"
                );
            }
        }
    }

    fn make_edge_q4_k_row(blocks: usize, seed: usize) -> Vec<u8> {
        const SCALE_BITS: [u16; 10] = [
            0x3C01, // 1 + 2^-10
            0xBC03, // -(1 + 3*2^-10)
            0x0001, // smallest positive subnormal f16
            0x8001, // smallest negative subnormal f16
            0x0000, // +0.0
            0x8000, // -0.0
            0x3555, // ~0.33325
            0xB557, // ~-0.33374
            0x4213, // ~3.037
            0xC215, // ~-3.041
        ];
        let mut bytes = Vec::with_capacity(blocks * Q4_K_BLOCK_BYTES);
        for b in 0..blocks {
            let d_bits = SCALE_BITS[(seed * 3 + b * 5) % SCALE_BITS.len()];
            let dmin_bits = SCALE_BITS[(seed * 7 + b * 2 + 1) % SCALE_BITS.len()];
            bytes.extend_from_slice(&d_bits.to_le_bytes());
            bytes.extend_from_slice(&dmin_bits.to_le_bytes());
            for i in 0..12 {
                bytes.push(u8::try_from((seed * 19 + b * 31 + i * 47) % 251).unwrap_or(0));
            }
            for i in 0..128 {
                bytes.push(u8::try_from((seed * 29 + b * 17 + i * 41) % 256).unwrap_or(0));
            }
        }
        bytes
    }

    fn make_edge_q6_k_row(blocks: usize, seed: usize) -> Vec<u8> {
        const SCALE_BITS: [u16; 8] = [
            0x3C01, 0xBC03, 0x0001, 0x8001, 0x0000, 0x8000, 0x3555, 0xB557,
        ];
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
                    isize::try_from((seed * 13 + b * 5 + i * 17) % 255).unwrap_or(0) - 127,
                )
                .unwrap_or(1);
                bytes.push(s.cast_unsigned());
            }
            let d_bits = SCALE_BITS[(seed * 5 + b * 3) % SCALE_BITS.len()];
            bytes.extend_from_slice(&d_bits.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn q4_k_repacked8_finite_scales_strips_matches_row_across_caps_bit_exact() {
        let det = Arm64QuantCaps::detect();
        let modes = [
            ("portable", Arm64QuantCaps::portable()),
            (
                "dotprod",
                Arm64QuantCaps {
                    dotprod: det.dotprod,
                    i8mm: false,
                },
            ),
            ("i8mm", det),
        ];

        for blocks in [0_usize, 1, 2, 3, 5] {
            let in_dim = blocks * QK_K;
            let q4_rows: Vec<Vec<u8>> = (0..24)
                .map(|r| {
                    if r % 2 == 0 {
                        make_edge_q4_k_row(blocks, r + 1)
                    } else {
                        make_q4_k_row(blocks, r + 7)
                    }
                })
                .collect();
            let flat_q4: Vec<u8> = q4_rows.iter().flat_map(|r| r.iter().copied()).collect();
            let repacked = if blocks == 0 {
                Vec::new()
            } else {
                repack_q4_k_8rows(&flat_q4, in_dim, 24).unwrap_or_default()
            };
            assert_eq!(repacked.len(), 3 * blocks * Q4_K_REPACK8_BLOCK_BYTES);

            let act = make_edge_q8_k_act(blocks, 2);
            for &(mode_name, caps) in &modes {
                let mut out = vec![f32::NAN; 16];
                dot_q4_k_repacked8_finite_scales_strips(caps, &repacked, 1, 2, &act, &mut out);
                for r in 0..16 {
                    let expected = dot_q4_k_row(Arm64QuantCaps::portable(), &q4_rows[8 + r], &act);
                    assert_eq!(
                        out[r].to_bits(),
                        expected.to_bits(),
                        "mode={mode_name} blocks={blocks} r={r}"
                    );
                }
            }
        }

        // Edge cases: strip_count == 0 or mismatched buffer lengths leave out untouched.
        let valid = vec![0_u8; 2 * Q4_K_REPACK8_BLOCK_BYTES];
        let act = make_edge_q8_k_act(2, 0);
        let mut untouched = [7.0_f32; 8];
        dot_q4_k_repacked8_finite_scales_strips(det, &valid, 0, 0, &act, &mut untouched);
        assert_eq!(untouched.map(f32::to_bits), [7.0_f32.to_bits(); 8]);
        dot_q4_k_repacked8_finite_scales_strips(
            det,
            &valid[..valid.len() - 1],
            0,
            1,
            &act,
            &mut untouched,
        );
        assert_eq!(untouched.map(f32::to_bits), [7.0_f32.to_bits(); 8]);
    }

    #[test]
    fn q8_k_act_panels8_and_k_blocked_gemm_match_row_across_caps_bit_exact() {
        let det = Arm64QuantCaps::detect();
        let port = Arm64QuantCaps::portable();
        let modes = [
            ("portable", port),
            (
                "dotprod",
                Arm64QuantCaps {
                    dotprod: det.dotprod,
                    i8mm: false,
                },
            ),
            ("i8mm", det),
        ];

        // Exercise K_TILE=4 and T_PANEL=8 boundaries across blocks and panel_count.
        for blocks in [0_usize, 1, 2, 3, 4, 5, 9] {
            let in_dim = blocks * QK_K;
            let q4_rows: Vec<Vec<u8>> = (0..8)
                .map(|r| {
                    if r % 2 == 0 {
                        make_edge_q4_k_row(blocks, r + 3)
                    } else {
                        make_q4_k_row(blocks, r + 11)
                    }
                })
                .collect();
            let flat_q4: Vec<u8> = q4_rows.iter().flat_map(|r| r.iter().copied()).collect();
            let repacked = if blocks == 0 {
                Vec::new()
            } else {
                repack_q4_k_8rows(&flat_q4, in_dim, 8).unwrap_or_default()
            };

            let q6_rows: Vec<Vec<u8>> = (0..8)
                .map(|r| {
                    if r % 2 == 0 {
                        make_edge_q6_k_row(blocks, r + 5)
                    } else {
                        make_q6_k_row(blocks, r + 13)
                    }
                })
                .collect();
            let q6_refs: [&[u8]; 8] = std::array::from_fn(|r| q6_rows[r].as_slice());

            for panel_count in [0_usize, 1, 2, 8, 9] {
                let tokens = panel_count * 8;
                let mut flat_acts = Vec::with_capacity(tokens * blocks);
                for t in 0..tokens {
                    flat_acts.extend(make_edge_q8_k_act(blocks, t));
                }
                let mut panels = vec![Q8_KActPanel8::default(); panel_count * blocks];
                pack_q8_k_act_panels8(&flat_acts, blocks, panel_count, &mut panels);

                let stride = tokens + 3;
                for &(mode_name, caps) in &modes {
                    let mut out_q4 = vec![f32::NAN; 8 * stride];
                    let mut out_q6 = vec![f32::NAN; 8 * stride];
                    gemm_q4_k_repacked8_8x_panels(
                        caps,
                        &repacked,
                        &panels,
                        panel_count,
                        blocks,
                        &mut out_q4,
                        stride,
                    );
                    gemm_q6_k_8x_panels(
                        caps,
                        q6_refs,
                        &panels,
                        panel_count,
                        blocks,
                        &mut out_q6,
                        stride,
                    );

                    for r in 0..8 {
                        for t in 0..tokens {
                            let act_tok = &flat_acts[t * blocks..(t + 1) * blocks];
                            let want_q4 = dot_q4_k_row(port, &q4_rows[r], act_tok).to_bits();
                            let want_q6 = dot_q6_k_row(port, q6_refs[r], act_tok).to_bits();
                            let idx = r * stride + t;
                            assert_eq!(
                                out_q4[idx].to_bits(),
                                want_q4,
                                "q4_k mismatch mode={mode_name} blocks={blocks} panels={panel_count} r={r} t={t}"
                            );
                            assert_eq!(
                                out_q6[idx].to_bits(),
                                want_q6,
                                "q6_k mismatch mode={mode_name} blocks={blocks} panels={panel_count} r={r} t={t}"
                            );
                        }
                    }
                }
            }
        }
    }

    fn make_q8_0_rows(out_dim: usize, blocks: usize, seed: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(out_dim * blocks * Q8_0_BLOCK_BYTES);
        for r in 0..out_dim {
            out.extend_from_slice(&make_q8_0_row(blocks, seed + r * 3));
        }
        out
    }

    fn make_q8_0_act(blocks: usize, seed: usize) -> Vec<Q8_0ActBlock> {
        let mut x = Vec::with_capacity(blocks * Q8_0_BLOCK_ELEMENTS);
        for i in 0..(blocks * Q8_0_BLOCK_ELEMENTS) {
            let v = f32::from(
                i16::try_from(isize::try_from((seed * 19 + i * 29) % 211).unwrap_or(0) - 105)
                    .unwrap_or(0),
            ) * 0.031_25_f32;
            x.push(v);
        }
        let mut out = vec![Q8_0ActBlock::default(); blocks];
        quantize_row_q8_0(&x, &mut out);
        out
    }

    #[test]
    fn repacked_q8_0_8_validates_scales_and_matches_dot_q8_0_row_bit_exact() {
        let port = Arm64QuantCaps::portable();
        let dot = Arm64QuantCaps {
            dotprod: true,
            i8mm: false,
        };
        let i8mm = Arm64QuantCaps {
            dotprod: true,
            i8mm: true,
        };
        let modes = [("portable", port), ("dotprod", dot), ("i8mm", i8mm)];

        // Non-finite scales (NaN, +Inf, -Inf) must be rejected by repack_q8_0_8rows.
        for bad_bits in [0x7e00_u16, 0x7c00_u16, 0xfc00_u16] {
            let mut src = make_q8_0_rows(8, 2, 11);
            let bad = bad_bits.to_le_bytes();
            let row_bytes = 2 * Q8_0_BLOCK_BYTES;
            src[5 * row_bytes + Q8_0_BLOCK_BYTES..5 * row_bytes + Q8_0_BLOCK_BYTES + 2]
                .copy_from_slice(&bad);
            assert!(repack_q8_0_8rows(&src, 64, 8).is_none());
        }

        // Edge scales (0.0, -0.0, subnormals, max finite) are finite and accepted.
        let edge_scales = [
            0x0000_u16, 0x8000_u16, 0x0001_u16, 0x8001_u16, 0x03ff_u16, 0x7bff_u16, 0xfbff_u16,
            0x3c00_u16,
        ];
        for blocks in [1_usize, 31, 32, 33, 64, 65] {
            let in_dim = blocks * Q8_0_BLOCK_ELEMENTS;
            let out_dim = 32_usize; // 4 strips
            let mut src = make_q8_0_rows(out_dim, blocks, 17);
            let row_bytes = blocks * Q8_0_BLOCK_BYTES;
            for r in 0..8 {
                let d_bytes = edge_scales[r].to_le_bytes();
                src[r * row_bytes..r * row_bytes + 2].copy_from_slice(&d_bytes);
            }
            let Some(repacked) = repack_q8_0_8rows(&src, in_dim, out_dim) else {
                assert!(false, "finite edge scales must repack cleanly");
                return;
            };
            assert_eq!(
                repacked.len(),
                (out_dim / 8) * blocks * REPACKED_Q8_0_8_BLOCK_BYTES
            );

            let act = make_q8_0_act(blocks, 77);
            let strip_bytes = blocks * REPACKED_Q8_0_8_BLOCK_BYTES;
            for (mode_name, caps) in modes {
                let first_strip =
                    dot_repacked_q8_0_8_finite_scales(caps, &repacked[..strip_bytes], &act);
                let mut all_out = vec![0.0_f32; out_dim];
                dot_repacked_q8_0_8_finite_scales_strips(caps, &repacked, 0, 4, &act, &mut all_out);
                for r in 0..8 {
                    let row = &src[r * row_bytes..(r + 1) * row_bytes];
                    let want = dot_q8_0_row(port, row, &act).to_bits();
                    assert_eq!(
                        first_strip[r].to_bits(),
                        want,
                        "repacked dot mismatch mode={mode_name} blocks={blocks} r={r}"
                    );
                }
                for r in 0..out_dim {
                    let row = &src[r * row_bytes..(r + 1) * row_bytes];
                    let want = dot_q8_0_row(port, row, &act).to_bits();
                    assert_eq!(
                        all_out[r].to_bits(),
                        want,
                        "repacked strips mismatch mode={mode_name} blocks={blocks} r={r}"
                    );
                }
            }

            let acts: Vec<Vec<Q8_0ActBlock>> =
                (0..8).map(|t| make_q8_0_act(blocks, 100 + t)).collect();
            let act8: [&[Q8_0ActBlock]; 8] = [
                &acts[0], &acts[1], &acts[2], &acts[3], &acts[4], &acts[5], &acts[6], &acts[7],
            ];
            let act4: [&[Q8_0ActBlock]; 4] = [&acts[0], &acts[1], &acts[2], &acts[3]];
            let act2: [&[Q8_0ActBlock]; 2] = [&acts[0], &acts[1]];
            for (mode_name, caps) in modes {
                let got8 = gemm_repacked_q8_0_8_8x8(caps, &repacked[..strip_bytes], act8);
                let got4 = gemm_repacked_q8_0_8_8x4(caps, &repacked[..strip_bytes], act4);
                let got2 = gemm_repacked_q8_0_8_8x2(caps, &repacked[..strip_bytes], act2);
                for r in 0..8 {
                    let row = &src[r * row_bytes..(r + 1) * row_bytes];
                    for t in 0..8 {
                        let want = dot_q8_0_row(port, row, act8[t]).to_bits();
                        assert_eq!(got8[r][t].to_bits(), want, "8x8 {mode_name} b={blocks}");
                        if t < 4 {
                            assert_eq!(got4[r][t].to_bits(), want, "8x4 {mode_name} b={blocks}");
                        }
                        if t < 2 {
                            assert_eq!(got2[r][t].to_bits(), want, "8x2 {mode_name} b={blocks}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn panel_gemms_finite_scatter_and_bands_match_scalar_bit_exact() {
        let port = Arm64QuantCaps::portable();
        let dot = Arm64QuantCaps {
            dotprod: true,
            i8mm: false,
        };
        let i8mm = Arm64QuantCaps {
            dotprod: true,
            i8mm: true,
        };
        let modes = [("portable", port), ("dotprod", dot), ("i8mm", i8mm)];
        let bias = [0.25_f32, -1.5, 0.125, 3.0, -0.75, 2.5, 0.0, -4.25];

        for blocks in [1_usize, 31, 32, 33, 64, 65] {
            // 1. Q8_0 and RepackedQ8_0_8
            let q8_raw = make_q8_0_rows(8, blocks, 23);
            let q8_row_bytes = blocks * Q8_0_BLOCK_BYTES;
            let q8_refs: [&[u8]; 8] = [
                &q8_raw[0..q8_row_bytes],
                &q8_raw[q8_row_bytes..2 * q8_row_bytes],
                &q8_raw[2 * q8_row_bytes..3 * q8_row_bytes],
                &q8_raw[3 * q8_row_bytes..4 * q8_row_bytes],
                &q8_raw[4 * q8_row_bytes..5 * q8_row_bytes],
                &q8_raw[5 * q8_row_bytes..6 * q8_row_bytes],
                &q8_raw[6 * q8_row_bytes..7 * q8_row_bytes],
                &q8_raw[7 * q8_row_bytes..8 * q8_row_bytes],
            ];
            let Some(q8_repacked) = repack_q8_0_8rows(&q8_raw, blocks * Q8_0_BLOCK_ELEMENTS, 8)
            else {
                assert!(false, "repack_q8_0_8rows should succeed");
                return;
            };

            // 2. Q4_K repacked and Q6_K
            let q4_rows: Vec<Vec<u8>> = (0..8).map(|r| make_q4_k_row(blocks, 500 + r)).collect();
            let q4_refs: [&[u8]; 8] = [
                &q4_rows[0],
                &q4_rows[1],
                &q4_rows[2],
                &q4_rows[3],
                &q4_rows[4],
                &q4_rows[5],
                &q4_rows[6],
                &q4_rows[7],
            ];
            let flat_q4: Vec<u8> = q4_rows.iter().flat_map(|r| r.iter().copied()).collect();
            let Some(q4_repacked) = repack_q4_k_8rows(&flat_q4, blocks * QK_K, 8) else {
                assert!(false, "repack_q4_k_8rows should succeed");
                return;
            };
            let q6_rows: Vec<Vec<u8>> = (0..8).map(|r| make_q6_k_row(blocks, 900 + r)).collect();
            let q6_refs: [&[u8]; 8] = [
                &q6_rows[0],
                &q6_rows[1],
                &q6_rows[2],
                &q6_rows[3],
                &q6_rows[4],
                &q6_rows[5],
                &q6_rows[6],
                &q6_rows[7],
            ];

            for panel_count in [1_usize, 7, 8, 9, 40, 42, 50] {
                let tokens = panel_count * 8;
                let mut flat_q8 = Vec::with_capacity(tokens * blocks);
                let mut flat_qk = Vec::with_capacity(tokens * blocks);
                for t in 0..tokens {
                    flat_q8.extend_from_slice(&make_q8_0_act(blocks, 1_000 + t));
                    flat_qk.extend(make_edge_q8_k_act(blocks, 2_000 + t));
                }
                let mut panels_q8 = vec![Q8_0ActPanel8::default(); panel_count * blocks];
                pack_q8_0_act_panels8(&flat_q8, blocks, panel_count, &mut panels_q8);
                let mut panels_qk = vec![Q8_KActPanel8::default(); panel_count * blocks];
                pack_q8_k_act_panels8(&flat_qk, blocks, panel_count, &mut panels_qk);

                // Precompute expected 0-ULP scalar outputs across all 8 rows and tokens.
                let mut want_q8 = vec![0_u32; 8 * tokens];
                let mut want_q4 = vec![0_u32; 8 * tokens];
                let mut want_q6 = vec![0_u32; 8 * tokens];
                for r in 0..8 {
                    for t in 0..tokens {
                        let a8 = &flat_q8[t * blocks..(t + 1) * blocks];
                        let ak = &flat_qk[t * blocks..(t + 1) * blocks];
                        want_q8[r * tokens + t] = dot_q8_0_row(port, q8_refs[r], a8).to_bits();
                        want_q4[r * tokens + t] = dot_q4_k_row(port, q4_refs[r], ak).to_bits();
                        want_q6[r * tokens + t] = dot_q6_k_row(port, q6_refs[r], ak).to_bits();
                    }
                }

                let row_stride = tokens + 5;
                let scatter_stride = 16_usize;
                for (mode_name, caps) in modes {
                    let mut rows_q8 = vec![0.0_f32; 8 * row_stride];
                    let mut rows_rep8 = vec![0.0_f32; 8 * row_stride];
                    let mut rows_q4 = vec![0.0_f32; 8 * row_stride];
                    let mut rows_q6 = vec![0.0_f32; 8 * row_stride];

                    gemm_q8_0_8x_panels_finite_scales(
                        caps,
                        q8_refs,
                        &panels_q8,
                        panel_count,
                        blocks,
                        &mut rows_q8,
                        row_stride,
                    );
                    gemm_repacked_q8_0_8_8x_panels_finite_scales(
                        caps,
                        &q8_repacked,
                        &panels_q8,
                        panel_count,
                        blocks,
                        &mut rows_rep8,
                        row_stride,
                    );
                    gemm_q4_k_repacked8_8x_panels_finite_scales(
                        caps,
                        &q4_repacked,
                        &panels_qk,
                        panel_count,
                        blocks,
                        &mut rows_q4,
                        row_stride,
                    );
                    gemm_q6_k_8x_panels_finite_scales(
                        caps,
                        q6_refs,
                        &panels_qk,
                        panel_count,
                        blocks,
                        &mut rows_q6,
                        row_stride,
                    );

                    let mut scat_q8 = vec![0.0_f32; tokens * scatter_stride];
                    let mut scat_rep8 = vec![0.0_f32; tokens * scatter_stride];
                    let mut scat_q4 = vec![0.0_f32; tokens * scatter_stride];
                    let mut scat_q6 = vec![0.0_f32; tokens * scatter_stride];

                    assert!(gemm_q8_0_8x_panels_scatter_biased::<true>(
                        caps,
                        q8_refs,
                        &panels_q8,
                        panel_count,
                        blocks,
                        &mut scat_q8[4..],
                        scatter_stride,
                        Some(bias),
                    ));
                    assert!(gemm_repacked_q8_0_8_8x_panels_finite_scales_scatter_biased(
                        caps,
                        &q8_repacked,
                        &panels_q8,
                        panel_count,
                        blocks,
                        &mut scat_rep8[4..],
                        scatter_stride,
                        Some(bias),
                    ));
                    assert!(gemm_q4_k_repacked8_8x_panels_scatter_biased::<true>(
                        caps,
                        &q4_repacked,
                        &panels_qk,
                        panel_count,
                        blocks,
                        &mut scat_q4[4..],
                        scatter_stride,
                        Some(bias),
                    ));
                    assert!(gemm_q6_k_8x_panels_scatter_biased::<true>(
                        caps,
                        q6_refs,
                        &panels_qk,
                        panel_count,
                        blocks,
                        &mut scat_q6[4..],
                        scatter_stride,
                        Some(bias),
                    ));

                    // Multi-band RowMajorBand epilogue test (split 16 cols into [8, 8], write second band).
                    let mut band_buf = vec![0.0_f32; tokens * scatter_stride];
                    if let Some(mut bands) =
                        split_row_major_bands(&mut band_buf, tokens, scatter_stride, &[8, 8])
                    {
                        assert!(gemm_q8_0_8x_panels_into_band::<false>(
                            caps,
                            q8_refs,
                            &panels_q8,
                            panel_count,
                            blocks,
                            &mut bands[0],
                            0,
                            Some(bias),
                        ));
                        assert!(gemm_q4_k_repacked8_8x_panels_into_band::<false>(
                            caps,
                            &q4_repacked,
                            &panels_qk,
                            panel_count,
                            blocks,
                            &mut bands[1],
                            0,
                            None,
                        ));
                    } else {
                        assert!(false, "split_row_major_bands must succeed");
                    }

                    for r in 0..8 {
                        for t in 0..tokens {
                            let ridx = r * row_stride + t;
                            let w8 = want_q8[r * tokens + t];
                            let w4 = want_q4[r * tokens + t];
                            let w6 = want_q6[r * tokens + t];
                            assert_eq!(
                                rows_q8[ridx].to_bits(),
                                w8,
                                "q8 finite {mode_name} b={blocks} p={panel_count} r={r} t={t}"
                            );
                            assert_eq!(
                                rows_rep8[ridx].to_bits(),
                                w8,
                                "rep8 finite {mode_name} b={blocks} p={panel_count} r={r} t={t}"
                            );
                            assert_eq!(
                                rows_q4[ridx].to_bits(),
                                w4,
                                "q4 finite {mode_name} b={blocks} p={panel_count} r={r} t={t}"
                            );
                            assert_eq!(
                                rows_q6[ridx].to_bits(),
                                w6,
                                "q6 finite {mode_name} b={blocks} p={panel_count} r={r} t={t}"
                            );

                            let sidx = t * scatter_stride + 4 + r;
                            let b8 = (f32::from_bits(w8) + bias[r]).to_bits();
                            let b4 = (f32::from_bits(w4) + bias[r]).to_bits();
                            let b6 = (f32::from_bits(w6) + bias[r]).to_bits();
                            assert_eq!(scat_q8[sidx].to_bits(), b8, "scatter q8 {mode_name}");
                            assert_eq!(scat_rep8[sidx].to_bits(), b8, "scatter rep8 {mode_name}");
                            assert_eq!(scat_q4[sidx].to_bits(), b4, "scatter q4 {mode_name}");
                            assert_eq!(scat_q6[sidx].to_bits(), b6, "scatter q6 {mode_name}");

                            assert_eq!(
                                band_buf[t * scatter_stride + r].to_bits(),
                                b8,
                                "band0 q8 {mode_name}"
                            );
                            assert_eq!(
                                band_buf[t * scatter_stride + 8 + r].to_bits(),
                                w4,
                                "band1 q4 {mode_name}"
                            );
                        }
                    }

                    // Also test repacked Q8 band writer and Q6_K band writer + non-finite bias rejection.
                    let mut band_buf2 = vec![0.0_f32; tokens * scatter_stride];
                    if let Some(mut bands2) =
                        split_row_major_bands(&mut band_buf2, tokens, scatter_stride, &[8, 8])
                    {
                        assert!(gemm_repacked_q8_0_8_8x_panels_finite_scales_into_band(
                            caps,
                            &q8_repacked,
                            &panels_q8,
                            panel_count,
                            blocks,
                            &mut bands2[0],
                            0,
                            None,
                        ));
                        let mut bad_bias = bias;
                        bad_bias[3] = f32::NAN;
                        assert!(!gemm_q6_k_8x_panels_into_band::<true>(
                            caps,
                            q6_refs,
                            &panels_qk,
                            panel_count,
                            blocks,
                            &mut bands2[1],
                            0,
                            Some(bad_bias),
                        ));
                    }
                    for t in 0..tokens {
                        for r in 0..8 {
                            assert_eq!(
                                band_buf2[t * scatter_stride + r].to_bits(),
                                want_q8[r * tokens + t]
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn f16_finite_pair_and_4x4_fma_attention_helpers_match_scalar_bit_exact() {
        let f16_edge = [
            0x0000_u16, 0x8000, 0x0001, 0x8001, 0x3c00, 0xbc00, 0x3555, 0x7bff, 0xfbff,
        ];
        for len in [0_usize, 1, 7, 8, 15, 16, 64, 128, 129] {
            let mut q0 = Vec::with_capacity(len);
            let mut q1 = Vec::with_capacity(len);
            let mut kv_f16 = Vec::with_capacity(len);
            for i in 0..len {
                let idx_f = f32::from(u16::try_from(i).unwrap_or(0));
                q0.push(((idx_f * 0.13).sin()) * 2.0);
                q1.push(((idx_f * 0.29).cos()) * -1.75);
                let val = ((idx_f * 0.41).sin()) * 1.5;
                let bits = if i < f16_edge.len() {
                    f16_edge[i]
                } else {
                    float_to_half(val)
                };
                kv_f16.push(bits);
            }

            let got_dot = dot_f16_pair_f32_finite(&q0, &q1, &kv_f16);
            let want_dot = dot_f16_pair_f32(&q0, &q1, &kv_f16);
            assert_eq!(got_dot.0.to_bits(), want_dot.0.to_bits(), "len={len} d0");
            assert_eq!(got_dot.1.to_bits(), want_dot.1.to_bits(), "len={len} d1");

            let mut out0_got = vec![0.5_f32; len];
            let mut out1_got = vec![-0.25_f32; len];
            let mut out0_want = out0_got.clone();
            let mut out1_want = out1_got.clone();
            axpy_f16_pair_f32_finite(0.375, -0.625, &kv_f16, &mut out0_got, &mut out1_got);
            axpy_f16_pair_f32(0.375, -0.625, &kv_f16, &mut out0_want, &mut out1_want);
            for i in 0..len {
                assert_eq!(
                    out0_got[i].to_bits(),
                    out0_want[i].to_bits(),
                    "len={len} axpy0 i={i}"
                );
                assert_eq!(
                    out1_got[i].to_bits(),
                    out1_want[i].to_bits(),
                    "len={len} axpy1 i={i}"
                );
            }

            // 4x4 FMA dot & axpy vs portable scalar FMA reference.
            let mut q_vecs: [Vec<f32>; 4] = std::array::from_fn(|_| Vec::with_capacity(len));
            let mut k_vecs: [Vec<f32>; 4] = std::array::from_fn(|_| Vec::with_capacity(len));
            let mut v_vecs: [Vec<f32>; 4] = std::array::from_fn(|_| Vec::with_capacity(len));
            for i in 0..len {
                let idx_f = f32::from(u16::try_from(i).unwrap_or(0));
                for g in 0..4 {
                    let gf = f32::from(u8::try_from(g + 1).unwrap_or(1));
                    q_vecs[g].push((idx_f * 0.17 * gf).sin() * 1.25);
                    k_vecs[g].push((idx_f * 0.23 * gf).cos() * 0.875);
                    v_vecs[g].push((idx_f * 0.31 + gf).sin() * 2.5);
                }
            }
            let q_refs: [&[f32]; 4] = [&q_vecs[0], &q_vecs[1], &q_vecs[2], &q_vecs[3]];
            let k_refs: [&[f32]; 4] = [&k_vecs[0], &k_vecs[1], &k_vecs[2], &k_vecs[3]];
            let divisor = 11.313_708_f32;
            let got_4x4 = dot_4x4_f32_fma(q_refs, k_refs, divisor);
            let want_4x4 =
                crate::native::quant_q8::dot_4x4_f32_fma_portable(q_refs, k_refs, divisor);
            for qi in 0..4 {
                for ki in 0..4 {
                    assert_eq!(
                        got_4x4[qi][ki].to_bits(),
                        want_4x4[qi][ki].to_bits(),
                        "dot_4x4 len={len} qi={qi} ki={ki}"
                    );
                }
            }

            let v_refs: [&[f32]; 4] = [&v_vecs[0], &v_vecs[1], &v_vecs[2], &v_vecs[3]];
            let weights = [
                [0.125_f32, 0.25, 0.5, 0.125],
                [0.4_f32, 0.3, 0.2, 0.1],
                [0.0_f32, 0.5, 0.25, 0.25],
                [0.75_f32, 0.125, 0.0625, 0.0625],
            ];
            let mut out_got: [Vec<f32>; 4] = std::array::from_fn(|g| {
                vec![f32::from(u8::try_from(g).unwrap_or(0)) * 0.1 - 0.2; len]
            });
            let mut out_want = out_got.clone();
            {
                let [o0, o1, o2, o3] = &mut out_got;
                let mut refs = [
                    o0.as_mut_slice(),
                    o1.as_mut_slice(),
                    o2.as_mut_slice(),
                    o3.as_mut_slice(),
                ];
                axpy_4x4_f32_fma(&mut refs, v_refs, weights);
            }
            {
                let [o0, o1, o2, o3] = &mut out_want;
                let mut refs = [
                    o0.as_mut_slice(),
                    o1.as_mut_slice(),
                    o2.as_mut_slice(),
                    o3.as_mut_slice(),
                ];
                crate::native::quant_q8::axpy_4x4_f32_fma_portable(&mut refs, v_refs, weights);
            }
            for g in 0..4 {
                for i in 0..len {
                    assert_eq!(
                        out_got[g][i].to_bits(),
                        out_want[g][i].to_bits(),
                        "axpy_4x4 len={len} g={g} i={i}"
                    );
                }
            }
        }
    }
}
