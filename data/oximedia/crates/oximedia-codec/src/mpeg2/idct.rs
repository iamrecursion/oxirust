//! 8×8 inverse DCT for MPEG-2 block reconstruction (ISO/IEC 13818-2 Annex A).
//!
//! Implements the separable two-dimensional inverse DCT
//!
//! ```text
//! f(x,y) = (2/N) · Σ_u Σ_v C(u)·C(v)·F(u,v)·cos[(2x+1)uπ/2N]·cos[(2y+1)vπ/2N]
//! ```
//!
//! with `N = 8`, `C(0) = 1/√2`, `C(k) = 1` for `k > 0`. The implementation uses
//! Q15 fixed-point cosine constants and i64 accumulators (no `unsafe`), and is
//! IEEE-1180 tolerant: intermediate results retain extra fractional bits and
//! only the final stage rounds to integer.
//!
//! For a DC-only block (`F(0,0) = D`, all AC zero) the kernel yields a flat
//! spatial block of value `D / 8`, matching ISO/IEC 13818-2 Annex A.

/// Q15 cosine constants. `COS_Q15[k] = round(cos(k·π/16) · 32768)`.
const COS_Q15: [i64; 9] = [
    32768, // cos(0π/16) = 1.0
    32138, // cos(1π/16)
    30274, // cos(2π/16)
    27246, // cos(3π/16)
    23170, // cos(4π/16) = 1/√2
    18205, // cos(5π/16)
    12540, // cos(6π/16)
    6393,  // cos(7π/16)
    0,     // cos(8π/16) = 0
];

/// `cos(angle·π/16)` in Q15 with 32-step periodicity.
fn cos_q15_periodic(angle: usize) -> i64 {
    let angle = angle % 32;
    match angle {
        0..=8 => COS_Q15[angle],
        9..=15 => -COS_Q15[16 - angle],
        16..=23 => -COS_Q15[angle - 16],
        _ => COS_Q15[32 - angle],
    }
}

// Pass-1 rounds away 11 of the 15 fractional bits, keeping 4 guard bits.
const PASS1_SHIFT: u32 = 11;
// Pass-2 removes the remaining scale so a DC-only block reconstructs to D/8.
//
// Per dimension the kernel multiplies the DC term by C(0) = COS_Q15[4]
// (= 1/√2 in Q15). Over two passes the DC gain is COS_Q15[4]² ≈ 2^29.0036,
// of which pass 1 already removed 2^11. To land on D/8 (an extra 2^3) the
// pass-2 shift must remove 2^21 (since 29 − 11 + 3 ≈ 21).
const PASS2_SHIFT: u32 = 21;

/// One-dimensional 8-point inverse DCT in Q15.
///
/// The DC term uses `C(0) = cos(4π/16) = 1/√2`; all other terms use the
/// periodic cosine table directly (`C(k) = 1`). Output is `(Σ … + round) >>
/// shift`.
fn idct_1d(input: &[i64; 8], shift: u32) -> [i64; 8] {
    let mut out = [0i64; 8];
    let round = if shift == 0 { 0 } else { 1i64 << (shift - 1) };
    for (n, slot) in out.iter_mut().enumerate() {
        // DC term: input[0] · C(0) where C(0) = 1/√2 (= COS_Q15[4]).
        let mut acc: i64 = input[0] * COS_Q15[4];
        for (k, &coeff) in input.iter().enumerate().skip(1) {
            let phase = ((2 * n + 1) * k) % 32;
            acc += coeff * cos_q15_periodic(phase);
        }
        *slot = (acc + round) >> shift;
    }
    out
}

/// Two-dimensional 8×8 inverse DCT.
///
/// Input and output are 64 values in raster order (`row * 8 + col`). Output
/// values are signed integers (the spatial-domain residual / sample value
/// before the +128 level shift and clipping applied by [`clip_to_u8`]).
#[must_use]
pub fn idct_8x8(coeffs: &[i32; 64]) -> [i32; 64] {
    // Row pass (keeps 4 guard bits).
    let mut intermediate = [0i64; 64];
    for row in 0..8 {
        let row_in: [i64; 8] = std::array::from_fn(|c| i64::from(coeffs[row * 8 + c]));
        let row_out = idct_1d(&row_in, PASS1_SHIFT);
        for (c, &v) in row_out.iter().enumerate() {
            intermediate[row * 8 + c] = v;
        }
    }
    // Column pass (final normalisation).
    let mut output = [0i32; 64];
    for col in 0..8 {
        let col_in: [i64; 8] = std::array::from_fn(|r| intermediate[r * 8 + col]);
        let col_out = idct_1d(&col_in, PASS2_SHIFT);
        for (r, &v) in col_out.iter().enumerate() {
            output[r * 8 + col] = v as i32;
        }
    }
    output
}

/// Clip an IDCT residual sample to the 8-bit unsigned range `[0, 255]` after
/// adding the `+128` level shift mandated by the intra DC reconstruction.
#[must_use]
pub fn clip_to_u8(val: i32) -> u8 {
    val.clamp(0, 255) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dc_only_is_flat_and_dc_over_8() {
        // F(0,0) = 1024 → spatial value 1024/8 = 128, uniform.
        let mut coeffs = [0i32; 64];
        coeffs[0] = 1024;
        let out = idct_8x8(&coeffs);
        for (i, &v) in out.iter().enumerate() {
            assert!(
                (v - 128).abs() <= 1,
                "DC-only sample[{i}] = {v}, expected ~128"
            );
        }
    }

    #[test]
    fn dc_only_various_values() {
        for dc in [8i32, 64, 256, 512, 800, 2040] {
            let mut coeffs = [0i32; 64];
            coeffs[0] = dc;
            let out = idct_8x8(&coeffs);
            let expected = dc / 8;
            for &v in &out {
                assert!(
                    (v - expected).abs() <= 1,
                    "DC {dc}: sample {v}, expected ~{expected}"
                );
            }
        }
    }

    #[test]
    fn zero_input_is_zero() {
        let out = idct_8x8(&[0i32; 64]);
        assert!(out.iter().all(|&v| v == 0));
    }

    #[test]
    fn negative_dc_is_flat() {
        let mut coeffs = [0i32; 64];
        coeffs[0] = -512;
        let out = idct_8x8(&coeffs);
        for &v in &out {
            assert!((v - (-64)).abs() <= 1, "neg DC sample {v}, expected ~-64");
        }
    }

    #[test]
    fn cos_q15_periodic_quarter_periods() {
        assert_eq!(cos_q15_periodic(0), 32768);
        assert_eq!(cos_q15_periodic(8), 0);
        assert_eq!(cos_q15_periodic(16), -32768);
        assert_eq!(cos_q15_periodic(24), 0);
        assert_eq!(cos_q15_periodic(32), cos_q15_periodic(0));
    }

    #[test]
    fn clip_to_u8_clamps() {
        assert_eq!(clip_to_u8(-5), 0);
        assert_eq!(clip_to_u8(300), 255);
        assert_eq!(clip_to_u8(200), 200);
    }

    #[test]
    fn single_ac_coefficient_has_zero_mean() {
        // A pure AC basis function should integrate (sum) to ~0 over the block.
        let mut coeffs = [0i32; 64];
        coeffs[1] = 1024; // horizontal frequency 1
        let out = idct_8x8(&coeffs);
        let sum: i64 = out.iter().map(|&v| i64::from(v)).sum();
        assert!(sum.abs() <= 16, "AC basis sum {sum} should be near zero");
    }
}
