//! 8×8 inverse DCT for DNxHD block decoding.
//!
//! Identical algorithm to the ProRes IDCT: separable 2-D IDCT-II
//! implemented with Q15 integer arithmetic. The DC/AC coefficients
//! from DNxHD dequantization are in the same range as ProRes.

/// Q15 cosine constants. `COS_Q15[k]` = `round(cos(k·π/16) · 32768)`.
pub(super) const COS_Q15: [i32; 9] = [
    32768, // cos(0π/16) = 1.000000
    32138, // cos(1π/16) = 0.9807852804
    30274, // cos(2π/16) = 0.9238795325
    27246, // cos(3π/16) = 0.8314696123
    23170, // cos(4π/16) = 0.7071067812 (1/√2)
    18205, // cos(5π/16) = 0.5555702330
    12540, // cos(6π/16) = 0.3826834324
    6393,  // cos(7π/16) = 0.1950903220
    0,     // cos(8π/16) = 0
];

/// `cos(angle·π/16)` in Q15 with 32-step periodicity.
pub(super) fn cos_q15_periodic(angle: usize) -> i32 {
    let angle = angle % 32;
    match angle {
        0..=8 => COS_Q15[angle],
        9..=15 => -COS_Q15[16 - angle],
        16..=23 => -COS_Q15[angle - 16],
        _ => COS_Q15[32 - angle],
    }
}

/// 1-D 8-point integer IDCT with Q15 cosines.
fn idct_1d(input: &[i32; 8]) -> [i32; 8] {
    let mut out = [0i32; 8];
    for (n, slot) in out.iter_mut().enumerate() {
        let mut acc: i64 = i64::from(input[0]) * i64::from(COS_Q15[4]);
        for k in 1..8 {
            let phase_index = ((2 * n + 1) * k) % 32;
            let cos_val = cos_q15_periodic(phase_index);
            acc += i64::from(input[k]) * i64::from(cos_val);
        }
        *slot = ((acc + (1 << 14)) >> 15) as i32;
    }
    out
}

/// 2-D 8×8 inverse DCT. Input and output are 64 values in raster order.
#[must_use]
pub fn idct_8x8(coeffs: &[i32; 64]) -> [i32; 64] {
    // Row pass.
    let mut intermediate = [0i32; 64];
    for row in 0..8 {
        let row_in: [i32; 8] = std::array::from_fn(|c| coeffs[row * 8 + c]);
        let row_out = idct_1d(&row_in);
        for (c, &v) in row_out.iter().enumerate() {
            intermediate[row * 8 + c] = v;
        }
    }
    // Column pass.
    let mut output = [0i32; 64];
    for col in 0..8 {
        let col_in: [i32; 8] = std::array::from_fn(|r| intermediate[r * 8 + col]);
        let col_out = idct_1d(&col_in);
        for (r, &v) in col_out.iter().enumerate() {
            output[r * 8 + col] = v;
        }
    }
    output
}

/// Finalize an IDCT output sample to 8-bit unsigned.
///
/// `dc_offset` is added before clamping (e.g. `128` to centre Y/Cb/Cr in
/// 8-bit video range). The shift folds the remaining Q15 scale from `idct_1d`.
#[must_use]
pub fn finalize_8bit(val: i32, dc_offset: i32) -> u8 {
    let centered = (val + dc_offset + 128).clamp(0, 255);
    centered as u8
}

/// Finalize an IDCT output sample to 10-bit unsigned (returned as u16).
///
/// `dc_offset` is added before clamping (e.g. `512` to centre in 10-bit range).
#[must_use]
pub fn finalize_10bit(val: i32, dc_offset: i32) -> u16 {
    let centered = (val + dc_offset + 512).clamp(0, 1023);
    centered as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idct_dc_only_is_uniform() {
        // A block with only DC non-zero should produce a uniform block.
        let mut coeffs = [0i32; 64];
        // 128 * 8 → after both IDCT passes the spatial value is ~128.
        coeffs[0] = 128 * 8;
        let out = idct_8x8(&coeffs);
        let first = out[0];
        for (i, &v) in out.iter().enumerate() {
            assert!(
                (v - first).abs() <= 2,
                "DC-only IDCT sample[{i}]={v} vs first={first}"
            );
        }
    }

    #[test]
    fn idct_zero_input_is_zero() {
        let out = idct_8x8(&[0i32; 64]);
        assert!(out.iter().all(|&v| v == 0));
    }

    #[test]
    fn finalize_8bit_clamps() {
        assert_eq!(finalize_8bit(i32::MAX / 4, 0), 255);
        assert_eq!(finalize_8bit(i32::MIN / 4, 0), 0);
    }

    #[test]
    fn finalize_10bit_clamps() {
        assert_eq!(finalize_10bit(i32::MAX / 4, 0), 1023);
        assert_eq!(finalize_10bit(i32::MIN / 4, 0), 0);
    }

    #[test]
    fn cos_q15_periodic_quarter_periods() {
        assert_eq!(cos_q15_periodic(0), 32768); // cos(0) = 1
        assert_eq!(cos_q15_periodic(8), 0); // cos(π/2) = 0
        assert_eq!(cos_q15_periodic(16), -32768); // cos(π) = -1
        assert_eq!(cos_q15_periodic(24), 0); // cos(3π/2) = 0
                                             // Periodicity.
        assert_eq!(cos_q15_periodic(32), cos_q15_periodic(0));
    }
}
