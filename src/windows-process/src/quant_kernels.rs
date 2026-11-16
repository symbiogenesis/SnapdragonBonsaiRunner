//! Safe public boundary for ARM64 `SDOT` / `I8MM` / `FCVTL` / `FCVTN` quantized
//! kernels with runtime CPU feature detection and exact portable fallbacks.

/// Block size (in elements) of GGML `Q8_0` blocks.
pub const Q8_0_BLOCK_ELEMENTS: usize = 32;
/// Byte size of one GGML `Q8_0` block (`f16` scale + 32 `i8` quants).
pub const Q8_0_BLOCK_BYTES: usize = 34;
/// Superblock size (in elements) of GGML K-quants (`Q4_K`, `Q6_K`, `Q8_K`).
pub const QK_K: usize = 256;
/// Byte size of one GGML `Q4_K` superblock.
pub const Q4_K_BLOCK_BYTES: usize = 144;
/// Byte size of one GGML `Q6_K` superblock.
pub const Q6_K_BLOCK_BYTES: usize = 210;
/// Byte size of one repacked 8-row `Q4_K` superblock strip (`16+16+64+64+1024`).
pub const Q4_K_REPACK8_BLOCK_BYTES: usize = 1184;

/// Detected ARM64 optional quantized dot-product and matrix-multiply capabilities.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Arm64QuantCaps {
    /// ARMv8.2-A `dotprod` (`SDOT` / `vdotq_s32`) availability verified by the OS.
    pub dotprod: bool,
    /// ARMv8.2-A `i8mm` (`SMMLA`) availability verified by the OS.
    pub i8mm: bool,
}

impl Arm64QuantCaps {
    /// Returns runtime OS-verified ARM64 `dotprod` / `i8mm` capabilities (cached process-wide).
    #[must_use]
    pub fn detect() -> Self {
        Self {
            dotprod: bonsai_neon_dot::has_dotprod(),
            i8mm: bonsai_neon_dot::has_i8mm(),
        }
    }

    /// Returns a capability set with optional `dotprod` and `i8mm` disabled for fallback controls.
    #[must_use]
    pub const fn portable() -> Self {
        Self {
            dotprod: false,
            i8mm: false,
        }
    }
}

/// On-the-fly symmetric `Q8_0` quantized activation block (`32` values).
#[derive(Clone, Copy, Debug, Default)]
pub struct Q8_0ActBlock {
    /// Block scale factor (`max(|x|) / 127`).
    pub d: f32,
    /// Quantized signed activation lanes `[-127, 127]`.
    pub qs: [i8; 32],
}

/// On-the-fly symmetric `Q8_K` quantized activation superblock (`256` values + `16` sub-sums).
#[allow(
    non_camel_case_types,
    reason = "Match GGML Q8_0 / Q8_K activation block naming."
)]
#[derive(Clone, Copy, Debug)]
pub struct Q8_KActBlock {
    /// Superblock scale factor (`max(|x|) / 127`).
    pub d: f32,
    /// Quantized signed activation lanes `[-127, 127]`.
    pub qs: [i8; 256],
    /// Exact integer sums for each 16-element sub-group `qs[g*16..(g+1)*16]`.
    pub bsums: [i16; 16],
}

impl Default for Q8_KActBlock {
    fn default() -> Self {
        Self {
            d: 0.0,
            qs: [0; 256],
            bsums: [0; 16],
        }
    }
}

#[inline]
#[allow(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    reason = "Quantization scale multiplication converts bounded i32 dot sums back to f32."
)]
pub(crate) fn i32_to_f32(value: i32) -> f32 {
    value as f32
}

#[inline]
#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Superblock f64 scale product narrows once per block to f32."
)]
pub(crate) fn f64_to_f32(value: f64) -> f32 {
    value as f32
}

#[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
#[inline]
#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Scalar fallback clamps scaled float input to [-127, 127] before rounding to i8."
)]
pub(crate) fn quantize_i8_scalar(value: f32) -> i8 {
    value.round_ties_even().clamp(-127.0, 127.0) as i8
}

#[inline]
pub(crate) fn f16_bits_to_f32(bytes: [u8; 2]) -> f32 {
    bonsai_ternary::half_to_float(u16::from_le_bytes(bytes))
}

#[inline]
pub(crate) fn scalar_narrow_f32_to_f16(value: f32) -> Option<u16> {
    if !value.is_finite() {
        return None;
    }
    let bits = value.to_bits();
    let sign = u16::try_from((bits >> 16) & 0x8000).ok()?;
    let exp = (bits >> 23) & 0xff;
    let mant = bits & 0x007f_ffff;
    if exp >= 143 || (exp == 142 && mant >= 0x007f_f000) {
        return None;
    }
    if exp >= 113 {
        let mut h_exp = exp - 112;
        let mut h_mant = mant >> 13;
        let rem = mant & 0x1fff;
        if rem > 0x1000 || (rem == 0x1000 && (h_mant & 1) != 0) {
            h_mant += 1;
            if h_mant == 0x0400 {
                h_mant = 0;
                h_exp += 1;
            }
        }
        let h_bits = u16::try_from((h_exp << 10) | h_mant).ok()?;
        return Some(sign | h_bits);
    }
    if exp < 102 {
        return Some(sign);
    }
    let sig = 0x0080_0000 | mant;
    let shift = 126 - exp;
    let mut h_mant = sig >> shift;
    let half = 1_u32 << (shift - 1);
    let rem = sig & ((1_u32 << shift) - 1);
    if rem > half || (rem == half && (h_mant & 1) != 0) {
        h_mant += 1;
    }
    let h_bits = u16::try_from(h_mant).ok()?;
    Some(sign | h_bits)
}

#[inline]
pub(crate) fn unpack_q4_k_scale_min(group: usize, scales: &[u8]) -> (u8, u8) {
    if group < 4 {
        (scales[group] & 63, scales[group + 4] & 63)
    } else {
        (
            (scales[group + 4] & 15) | ((scales[group - 4] >> 6) << 4),
            (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4),
        )
    }
}

/// Widens four IEEE binary16 bit patterns using baseline ARM64 `FCVTL` for finite inputs.
#[must_use]
pub fn widen_f16_four(bits: &[u16; 4]) -> [f32; 4] {
    bonsai_neon_dot::widen_f16_four(bits)
}

/// Widens eight IEEE binary16 bit patterns using baseline ARM64 `FCVTL`/`FCVTL2` for finite inputs.
#[must_use]
pub fn widen_f16_eight(bits: &[u16; 8]) -> [[f32; 4]; 2] {
    bonsai_neon_dot::widen_f16_eight(bits)
}

/// Narrows finite `f32` values into IEEE binary16 bits, rejecting nonfinite or overflowing elements.
#[must_use]
pub fn narrow_f32_slice_to_f16(src: &[f32], dst: &mut [u16]) -> bool {
    if src.len() != dst.len() {
        return false;
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        crate::native::narrow_f32_slice_to_f16(src, dst)
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        for (&v, slot) in src.iter().zip(dst.iter_mut()) {
            let Some(bits) = scalar_narrow_f32_to_f16(v) else {
                return false;
            };
            *slot = bits;
        }
        true
    }
}

/// Quantizes `input` (`out.len() * 32` floats) into symmetric `Q8_0ActBlock`s once per matmul.
pub fn quantize_row_q8_0(input: &[f32], out: &mut [Q8_0ActBlock]) {
    if input.len() != out.len() * Q8_0_BLOCK_ELEMENTS {
        return;
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        crate::native::quantize_row_q8_0(input, out);
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        for (chunk, block) in input.as_chunks::<32>().0.iter().zip(out.iter_mut()) {
            let amax = chunk.iter().copied().fold(0.0_f32, |m, v| m.max(v.abs()));
            if amax == 0.0 || !amax.is_finite() {
                block.d = 0.0;
                block.qs = [0; 32];
                continue;
            }
            let d = amax / 127.0_f32;
            let inv_d = 127.0_f32 / amax;
            for (src, dst) in chunk.iter().zip(block.qs.iter_mut()) {
                *dst = quantize_i8_scalar(src * inv_d);
            }
            block.d = d;
        }
    }
}

/// Quantizes `input` (`out.len() * 256` floats) into `Q8_KActBlock`s with exact 16-lane `bsums`.
pub fn quantize_row_q8_k(input: &[f32], out: &mut [Q8_KActBlock]) {
    if input.len() != out.len() * QK_K {
        return;
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        crate::native::quantize_row_q8_k(input, out);
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        for (chunk, block) in input.as_chunks::<256>().0.iter().zip(out.iter_mut()) {
            let amax = chunk.iter().copied().fold(0.0_f32, |m, v| m.max(v.abs()));
            if amax == 0.0 || !amax.is_finite() {
                block.d = 0.0;
                block.qs = [0; 256];
                block.bsums = [0; 16];
                continue;
            }
            let d = amax / 127.0_f32;
            let inv_d = 127.0_f32 / amax;
            for ((src16, dst16), bsum) in chunk
                .as_chunks::<16>()
                .0
                .iter()
                .zip(block.qs.as_chunks_mut::<16>().0.iter_mut())
                .zip(block.bsums.iter_mut())
            {
                let mut sum = 0_i16;
                for (&x, q) in src16.iter().zip(dst16.iter_mut()) {
                    let qi = quantize_i8_scalar(x * inv_d);
                    *q = qi;
                    sum += i16::from(qi);
                }
                *bsum = sum;
            }
            block.d = d;
        }
    }
}

fn portable_dot_q8_0_row(w_row: &[u8], act: &[Q8_0ActBlock]) -> f32 {
    let mut sum = 0.0_f32;
    for (wb, ab) in w_row.as_chunks::<34>().0.iter().zip(act.iter()) {
        let w_d = f16_bits_to_f32([wb[0], wb[1]]);
        let mut int_sum = 0_i32;
        for (&w_q, &a_q) in wb[2..].iter().zip(ab.qs.iter()) {
            int_sum += i32::from(w_q.cast_signed()) * i32::from(a_q);
        }
        sum += w_d * (ab.d * i32_to_f32(int_sum));
    }
    sum
}

/// Computes `dot(w_row, act)` for one packed GGML `Q8_0` weight row against `Q8_0ActBlock`s.
#[must_use]
pub fn dot_q8_0_row(caps: Arm64QuantCaps, w_row: &[u8], act: &[Q8_0ActBlock]) -> f32 {
    if w_row.len() != act.len() * Q8_0_BLOCK_BYTES {
        return 0.0;
    }
    if caps.dotprod {
        let mut sum = 0.0_f32;
        for (wb, ab) in w_row.as_chunks::<34>().0.iter().zip(act.iter()) {
            let w_d = f16_bits_to_f32([wb[0], wb[1]]);
            let w_quants = &wb[2..].as_chunks::<32>().0[0];
            let int_sum = bonsai_neon_dot::dot_q8_0_block(w_quants, &ab.qs);
            sum += w_d * (ab.d * i32_to_f32(int_sum));
        }
        return sum;
    }
    portable_dot_q8_0_row(w_row, act)
}

/// Computes four `Q8_0 × Q8_0` row dot products while loading activations once per block.
#[must_use]
pub fn dot_q8_0_4rows(caps: Arm64QuantCaps, rows: [&[u8]; 4], act: &[Q8_0ActBlock]) -> [f32; 4] {
    let expected = act.len() * Q8_0_BLOCK_BYTES;
    if rows.iter().any(|row| row.len() != expected) {
        return [0.0; 4];
    }
    if caps.dotprod {
        let mut acc = [0.0_f32; 4];
        let r0 = rows[0].as_chunks::<34>().0;
        let r1 = rows[1].as_chunks::<34>().0;
        let r2 = rows[2].as_chunks::<34>().0;
        let r3 = rows[3].as_chunks::<34>().0;
        for (idx, ab) in act.iter().enumerate() {
            let blocks = [&r0[idx], &r1[idx], &r2[idx], &r3[idx]];
            let d_bits: [u16; 4] = blocks.map(|b| u16::from_le_bytes([b[0], b[1]]));
            let d_vals = bonsai_neon_dot::widen_f16_four(&d_bits);
            for row_idx in 0..4 {
                let w_quants = &blocks[row_idx][2..].as_chunks::<32>().0[0];
                let int_sum = bonsai_neon_dot::dot_q8_0_block(w_quants, &ab.qs);
                acc[row_idx] += d_vals[row_idx] * (ab.d * i32_to_f32(int_sum));
            }
        }
        return acc;
    }
    rows.map(|row| portable_dot_q8_0_row(row, act))
}

/// Computes a 4-row × 2-token `Q8_0 × Q8_0` tile using `I8MM` (`SMMLA`) or `SDOT` fallback.
#[must_use]
pub fn gemm_q8_0_4x2(
    caps: Arm64QuantCaps,
    rows: [&[u8]; 4],
    act0: &[Q8_0ActBlock],
    act1: &[Q8_0ActBlock],
) -> [[f32; 2]; 4] {
    let expected = act0.len() * Q8_0_BLOCK_BYTES;
    if act0.len() != act1.len() || rows.iter().any(|row| row.len() != expected) {
        return [[0.0; 2]; 4];
    }
    if caps.i8mm {
        let mut acc = [[0.0_f32; 2]; 4];
        let r0 = rows[0].as_chunks::<34>().0;
        let r1 = rows[1].as_chunks::<34>().0;
        let r2 = rows[2].as_chunks::<34>().0;
        let r3 = rows[3].as_chunks::<34>().0;
        for (idx, (ab0, ab1)) in act0.iter().zip(act1.iter()).enumerate() {
            let blocks = [&r0[idx], &r1[idx], &r2[idx], &r3[idx]];
            let w_quants = blocks.map(|b| &b[2..].as_chunks::<32>().0[0]);
            let tile = bonsai_neon_dot::smmla_q8_0_4x2(w_quants, [&ab0.qs, &ab1.qs]);
            let d_bits: [u16; 4] = blocks.map(|b| u16::from_le_bytes([b[0], b[1]]));
            let d_vals = bonsai_neon_dot::widen_f16_four(&d_bits);
            for row_idx in 0..4 {
                acc[row_idx][0] += d_vals[row_idx] * (ab0.d * i32_to_f32(tile[row_idx][0]));
                acc[row_idx][1] += d_vals[row_idx] * (ab1.d * i32_to_f32(tile[row_idx][1]));
            }
        }
        return acc;
    }
    let col0 = dot_q8_0_4rows(caps, rows, act0);
    let col1 = dot_q8_0_4rows(caps, rows, act1);
    std::array::from_fn(|r| [col0[r], col1[r]])
}

fn portable_dot_q4_k_row(w_row: &[u8], act: &[Q8_KActBlock]) -> f32 {
    let mut acc = 0.0_f64;
    for (block, ab) in w_row.as_chunks::<144>().0.iter().zip(act.iter()) {
        let d_w = f64::from(f16_bits_to_f32([block[0], block[1]]));
        let dmin_w = f64::from(f16_bits_to_f32([block[2], block[3]]));
        let d_act = f64::from(ab.d);
        let mut sumi = 0_i32;
        let mut mint = 0_i32;
        for group in 0..8 {
            let (sc, m) = unpack_q4_k_scale_min(group, &block[4..16]);
            let b_pair = i32::from(ab.bsums[2 * group]) + i32::from(ab.bsums[2 * group + 1]);
            mint += i32::from(m) * b_pair;
            let qs_base = 16 + (group / 2) * 32;
            let a_base = group * 32;
            let mut g_dot = 0_i32;
            for k in 0..32 {
                let b = block[qs_base + k];
                let q = if group % 2 == 0 { b & 0x0f } else { b >> 4 };
                g_dot += i32::from(q) * i32::from(ab.qs[a_base + k]);
            }
            sumi += g_dot * i32::from(sc);
        }
        acc += (d_w * d_act) * f64::from(sumi) - (dmin_w * d_act) * f64::from(mint);
    }
    f64_to_f32(acc)
}

/// Computes `dot(w_row, act)` for one packed GGML `Q4_K` weight row against `Q8_KActBlock`s.
#[must_use]
pub fn dot_q4_k_row(caps: Arm64QuantCaps, w_row: &[u8], act: &[Q8_KActBlock]) -> f32 {
    if w_row.len() != act.len() * Q4_K_BLOCK_BYTES {
        return 0.0;
    }
    #[cfg(target_arch = "aarch64")]
    if caps.dotprod
        && let Some(sum) = crate::native::quant_dot_q4_k_row(w_row, act)
    {
        return sum;
    }
    #[cfg(not(target_arch = "aarch64"))]
    let _ = caps;
    portable_dot_q4_k_row(w_row, act)
}

/// Computes four `Q4_K × Q8_K` row dot products while sharing activations per superblock.
#[must_use]
pub fn dot_q4_k_4rows(caps: Arm64QuantCaps, rows: [&[u8]; 4], act: &[Q8_KActBlock]) -> [f32; 4] {
    let expected = act.len() * Q4_K_BLOCK_BYTES;
    if rows.iter().any(|row| row.len() != expected) {
        return [0.0; 4];
    }
    #[cfg(target_arch = "aarch64")]
    if caps.dotprod
        && let Some(sums) = crate::native::quant_dot_q4_k_4rows(rows, act)
    {
        return sums;
    }
    #[cfg(not(target_arch = "aarch64"))]
    let _ = caps;
    rows.map(|row| portable_dot_q4_k_row(row, act))
}

/// Computes a 4-row × 2-token `Q4_K × Q8_K` tile using `I8MM` (`SMMLA`) or `SDOT` fallback.
#[must_use]
pub fn gemm_q4_k_4x2(
    caps: Arm64QuantCaps,
    rows: [&[u8]; 4],
    act0: &[Q8_KActBlock],
    act1: &[Q8_KActBlock],
) -> [[f32; 2]; 4] {
    let expected = act0.len() * Q4_K_BLOCK_BYTES;
    if act0.len() != act1.len() || rows.iter().any(|row| row.len() != expected) {
        return [[0.0; 2]; 4];
    }
    #[cfg(target_arch = "aarch64")]
    if caps.i8mm
        && let Some(tile) = crate::native::quant_gemm_q4_k_4x2(rows, act0, act1)
    {
        return tile;
    }
    let col0 = dot_q4_k_4rows(caps, rows, act0);
    let col1 = dot_q4_k_4rows(caps, rows, act1);
    std::array::from_fn(|r| [col0[r], col1[r]])
}

/// Repacks row-major GGML `Q4_K` matrix bytes into 8-row sequential strips (`1184` B per strip block).
///
/// Returns `None` if dimensions are not divisible by `(256, 8)` or any block scale is nonfinite.
#[must_use]
pub fn repack_q4_k_8rows(src: &[u8], in_dim: usize, out_dim: usize) -> Option<Vec<u8>> {
    if in_dim == 0 || out_dim == 0 || !in_dim.is_multiple_of(QK_K) || !out_dim.is_multiple_of(8) {
        return None;
    }
    let blocks_per_row = in_dim / QK_K;
    let src_row_bytes = blocks_per_row.checked_mul(Q4_K_BLOCK_BYTES)?;
    if src.len() != out_dim.checked_mul(src_row_bytes)? {
        return None;
    }
    let strips = out_dim / 8;
    let out_bytes = strips
        .checked_mul(blocks_per_row)?
        .checked_mul(Q4_K_REPACK8_BLOCK_BYTES)?;
    let mut dst = Vec::new();
    dst.try_reserve_exact(out_bytes).ok()?;
    dst.resize(out_bytes, 0);

    for strip in 0..strips {
        let r_base = strip * 8;
        for blk in 0..blocks_per_row {
            let out_off = (strip * blocks_per_row + blk) * Q4_K_REPACK8_BLOCK_BYTES;
            let out_blk = &mut dst[out_off..out_off + Q4_K_REPACK8_BLOCK_BYTES];
            for r in 0..8 {
                let src_off = (r_base + r) * src_row_bytes + blk * Q4_K_BLOCK_BYTES;
                let sb = &src[src_off..src_off + Q4_K_BLOCK_BYTES];
                let d_val = f16_bits_to_f32([sb[0], sb[1]]);
                let dm_val = f16_bits_to_f32([sb[2], sb[3]]);
                if !d_val.is_finite() || !dm_val.is_finite() {
                    return None;
                }
                out_blk[r * 2..r * 2 + 2].copy_from_slice(&sb[0..2]);
                out_blk[16 + r * 2..16 + r * 2 + 2].copy_from_slice(&sb[2..4]);
                for g in 0..8 {
                    let (sc, m) = unpack_q4_k_scale_min(g, &sb[4..16]);
                    out_blk[32 + g * 8 + r] = sc;
                    out_blk[96 + g * 8 + r] = m;
                }
                for chunk in 0..4 {
                    let s_off = 16 + chunk * 32;
                    let d_off = 160 + chunk * 256 + r * 32;
                    out_blk[d_off..d_off + 32].copy_from_slice(&sb[s_off..s_off + 32]);
                }
            }
        }
    }
    Some(dst)
}

fn portable_dot_q4_k_repacked8(strip: &[u8], act: &[Q8_KActBlock]) -> [f32; 8] {
    let mut acc = [0.0_f64; 8];
    for (block, ab) in strip
        .as_chunks::<Q4_K_REPACK8_BLOCK_BYTES>()
        .0
        .iter()
        .zip(act.iter())
    {
        let d_act = f64::from(ab.d);
        let d_weights: [f64; 8] =
            std::array::from_fn(|r| f64::from(f16_bits_to_f32([block[r * 2], block[r * 2 + 1]])));
        let dmin_weights: [f64; 8] = std::array::from_fn(|r| {
            f64::from(f16_bits_to_f32([block[16 + r * 2], block[16 + r * 2 + 1]]))
        });
        let mut mint = [0_i32; 8];
        let mut sumi = [0_i32; 8];
        for g in 0..8 {
            let b_pair = i32::from(ab.bsums[2 * g]) + i32::from(ab.bsums[2 * g + 1]);
            for r in 0..8 {
                mint[r] += i32::from(block[96 + g * 8 + r]) * b_pair;
            }
        }
        for chunk in 0..4 {
            let qs_base = 160 + chunk * 256;
            for r in 0..8 {
                let row_qs = &block[qs_base + r * 32..qs_base + (r + 1) * 32];
                let mut dot_lo = 0_i32;
                let mut dot_hi = 0_i32;
                let a_lo = &ab.qs[(chunk * 2) * 32..(chunk * 2 + 1) * 32];
                let a_hi = &ab.qs[(chunk * 2 + 1) * 32..(chunk * 2 + 2) * 32];
                for k in 0..32 {
                    dot_lo += i32::from(row_qs[k] & 0x0f) * i32::from(a_lo[k]);
                    dot_hi += i32::from(row_qs[k] >> 4) * i32::from(a_hi[k]);
                }
                sumi[r] += dot_lo * i32::from(block[32 + (chunk * 2) * 8 + r])
                    + dot_hi * i32::from(block[32 + (chunk * 2 + 1) * 8 + r]);
            }
        }
        for r in 0..8 {
            acc[r] += (d_weights[r] * d_act) * f64::from(sumi[r])
                - (dmin_weights[r] * d_act) * f64::from(mint[r]);
        }
    }
    acc.map(f64_to_f32)
}

/// Computes an 8-row repacked `Q4_K × Q8_K` strip dot product.
#[must_use]
pub fn dot_q4_k_repacked8(caps: Arm64QuantCaps, strip: &[u8], act: &[Q8_KActBlock]) -> [f32; 8] {
    if strip.len() != act.len() * Q4_K_REPACK8_BLOCK_BYTES {
        return [0.0; 8];
    }
    #[cfg(target_arch = "aarch64")]
    if caps.dotprod
        && let Some(sums) = crate::native::quant_dot_q4_k_repacked8(strip, act)
    {
        return sums;
    }
    #[cfg(not(target_arch = "aarch64"))]
    let _ = caps;
    portable_dot_q4_k_repacked8(strip, act)
}

/// Computes an 8-row × 2-token repacked `Q4_K × Q8_K` strip GEMM tile.
#[must_use]
pub fn gemm_q4_k_repacked8_8x2(
    caps: Arm64QuantCaps,
    strip: &[u8],
    act0: &[Q8_KActBlock],
    act1: &[Q8_KActBlock],
) -> [[f32; 2]; 8] {
    if act0.len() != act1.len() || strip.len() != act0.len() * Q4_K_REPACK8_BLOCK_BYTES {
        return [[0.0; 2]; 8];
    }
    #[cfg(target_arch = "aarch64")]
    if caps.i8mm
        && let Some(tile) = crate::native::quant_gemm_q4_k_repacked8_8x2(strip, act0, act1)
    {
        return tile;
    }
    let col0 = dot_q4_k_repacked8(caps, strip, act0);
    let col1 = dot_q4_k_repacked8(caps, strip, act1);
    std::array::from_fn(|r| [col0[r], col1[r]])
}

fn portable_dot_q6_k_row(w_row: &[u8], act: &[Q8_KActBlock]) -> f32 {
    let mut acc = 0.0_f64;
    for (block, ab) in w_row.as_chunks::<210>().0.iter().zip(act.iter()) {
        let d_w = f64::from(f16_bits_to_f32([block[208], block[209]]));
        let d_act = f64::from(ab.d);
        let mut sumi = 0_i32;
        for half_idx in 0..2 {
            for quarter in 0..4 {
                let s0 = i32::from(block[192 + half_idx * 8 + quarter * 2].cast_signed());
                let s1 = i32::from(block[192 + half_idx * 8 + quarter * 2 + 1].cast_signed());
                let mut d0 = 0_i32;
                let mut d1 = 0_i32;
                let ql_off = half_idx * 64 + (quarter % 2) * 32;
                let qh_off = 128 + half_idx * 32;
                let a_off = half_idx * 128 + quarter * 32;
                for i in 0..16 {
                    let ql = block[ql_off + i];
                    let low = if quarter < 2 { ql & 0x0f } else { ql >> 4 };
                    let high = (block[qh_off + i] >> (quarter * 2)) & 0x03;
                    let q = i32::from(low | (high << 4)) - 32;
                    d0 += q * i32::from(ab.qs[a_off + i]);
                }
                for i in 0..16 {
                    let ql = block[ql_off + 16 + i];
                    let low = if quarter < 2 { ql & 0x0f } else { ql >> 4 };
                    let high = (block[qh_off + 16 + i] >> (quarter * 2)) & 0x03;
                    let q = i32::from(low | (high << 4)) - 32;
                    d1 += q * i32::from(ab.qs[a_off + 16 + i]);
                }
                sumi += d0 * s0 + d1 * s1;
            }
        }
        acc += (d_w * d_act) * f64::from(sumi);
    }
    f64_to_f32(acc)
}

/// Computes `dot(w_row, act)` for one packed GGML `Q6_K` weight row against `Q8_KActBlock`s.
#[must_use]
pub fn dot_q6_k_row(caps: Arm64QuantCaps, w_row: &[u8], act: &[Q8_KActBlock]) -> f32 {
    if w_row.len() != act.len() * Q6_K_BLOCK_BYTES {
        return 0.0;
    }
    #[cfg(target_arch = "aarch64")]
    if caps.dotprod
        && let Some(sum) = crate::native::quant_dot_q6_k_row(w_row, act)
    {
        return sum;
    }
    #[cfg(not(target_arch = "aarch64"))]
    let _ = caps;
    portable_dot_q6_k_row(w_row, act)
}

/// Computes four `Q6_K × Q8_K` row dot products while sharing activations per superblock.
#[must_use]
pub fn dot_q6_k_4rows(caps: Arm64QuantCaps, rows: [&[u8]; 4], act: &[Q8_KActBlock]) -> [f32; 4] {
    let expected = act.len() * Q6_K_BLOCK_BYTES;
    if rows.iter().any(|row| row.len() != expected) {
        return [0.0; 4];
    }
    #[cfg(target_arch = "aarch64")]
    if caps.dotprod
        && let Some(sums) = crate::native::quant_dot_q6_k_4rows(rows, act)
    {
        return sums;
    }
    #[cfg(not(target_arch = "aarch64"))]
    let _ = caps;
    rows.map(|row| portable_dot_q6_k_row(row, act))
}

/// Computes a 4-row × 2-token `Q6_K × Q8_K` tile using `I8MM` (`SMMLA`) or `SDOT` fallback.
#[must_use]
pub fn gemm_q6_k_4x2(
    caps: Arm64QuantCaps,
    rows: [&[u8]; 4],
    act0: &[Q8_KActBlock],
    act1: &[Q8_KActBlock],
) -> [[f32; 2]; 4] {
    let expected = act0.len() * Q6_K_BLOCK_BYTES;
    if act0.len() != act1.len() || rows.iter().any(|row| row.len() != expected) {
        return [[0.0; 2]; 4];
    }
    #[cfg(target_arch = "aarch64")]
    if caps.i8mm
        && let Some(tile) = crate::native::quant_gemm_q6_k_4x2(rows, act0, act1)
    {
        return tile;
    }
    let col0 = dot_q6_k_4rows(caps, rows, act0);
    let col1 = dot_q6_k_4rows(caps, rows, act1);
    std::array::from_fn(|r| [col0[r], col1[r]])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_to_f16_bits_ref(value: f32) -> u16 {
        scalar_narrow_f32_to_f16(value).unwrap_or(0)
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
