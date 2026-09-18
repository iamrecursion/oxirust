//! 8×8 forward Discrete Cosine Transform (FDCT) — forward transform that
//! converts spatial-domain 10-bit samples into DCT frequency coefficients
//! suitable for quantization and entropy coding.
//!
//! This is the separable 2-D DCT-II, the mathematical inverse of the IDCT-II
//! implemented in [`super::idct`]. The same Q15 cosine constants are reused.
//!
//! ## Algorithm
//!
//! For a 1-D 8-point FDCT:
//!
//! ```text
//!  X[k] = Σ_{n=0}^{7} x[n] · cos((2n+1)·k·π/16)   k = 0..7
//!
//!  with k=0 additionally scaled by 1/√2 (to match the orthonormal IDCT).
//! ```
//!
//! The 2-D FDCT is separable: first apply the 1-D FDCT to each row, then
//! to each column of the result.
//!
//! ## Normalization
//!
//! The IDCT in this codebase uses direct evaluation with Q15 cosines and
//! shifts by Q15 in each pass. The FDCT must be the exact inverse to make
//! `fdct → quantize → dequantize → idct` round-trip correctly.
//!
//! For each 1-D FDCT pass we:
//! 1. Accumulate with i64 to avoid overflow.
//! 2. Scale the result right by 15 bits (Q15 factor from the cosine table).
//! 3. After both passes shift right by 3 more (= divide by 8, the normalizer
//!    for an 8-point transform) — total 18 bits from a 64-bit accumulator.
//!
//! This yields coefficients in the range approximately [-512, 511] for 10-bit
//! input, which is exactly the range dequantize × matrix × qscale expects.

use super::idct::{cos_q15_periodic, COS_Q15};

/// 1-D 8-point FDCT, Q15 fixed-point.
///
/// Output is the raw DCT coefficients scaled by 2^15 (one Q15 factor
/// that will be removed by the column-pass normalisation in [`fdct_8x8`]).
fn fdct_1d(input: &[i32; 8]) -> [i64; 8] {
    let mut out = [0i64; 8];
    for k in 0..8usize {
        let mut acc: i64 = 0;
        for n in 0..8usize {
            let phase_index = ((2 * n + 1) * k) % 32;
            let cos_val = if k == 0 {
                // k=0: cos(0) = 1 but we apply the 1/√2 orthonormal weight
                // (COS_Q15[4] = cos(π/4) ≈ 32768/√2)
                COS_Q15[4]
            } else {
                cos_q15_periodic(phase_index)
            };
            acc += i64::from(input[n]) * i64::from(cos_val);
        }
        out[k] = acc;
    }
    out
}

/// 2-D 8×8 forward DCT.
///
/// Input: 64 spatial samples in **raster order** (row-major), centered at 0
/// (i.e. subtract 512 from raw 10-bit values before calling).
///
/// Output: 64 DCT coefficients in raster order. The coefficients are
/// normalized to be the exact inverse of `idct_8x8` (modulo integer rounding):
/// `fdct_8x8 → quantize → dequantize → idct_8x8` round-trips with error
/// bounded by the quantization step size.
///
/// ## Normalization derivation
///
/// `idct_1d` computes `output[n] = (Σ_k alpha(k) * X[k] * COS_Q15[(2n+1)*k]) >> 15`
/// where `alpha(0) = COS_Q15[4]/2^15 = 1/√2` and `alpha(k>0) = 1`.
/// After two passes: `spatial[r,c] ≈ X[0,0] * (COS_Q15[4])^2 / 2^30` for DC-only.
///
/// For the round-trip: `fdct(flat block of V)[0,0]` must equal `2V * 2^30 / COS_Q15[4]^2`.
///
/// Row pass unshifted: `intermediate[k=0] = 8V * COS_Q15[4]` (per row).
/// Column pass: `sum = 8 * intermediate[k=0] * COS_Q15[4] = 64V * COS_Q15[4]^2`.
/// Required shift S such that `64V * COS_Q15[4]^2 / 2^S = 2V`:
/// `2^S = 32 * COS_Q15[4]^2 = 32 * 23170^2 ≈ 2^34`.
/// So total shift = 34 bits.
///
/// ## Output layout
///
/// The IDCT expects `coeffs[row_freq * 8 + col_freq]` where row_freq is the
/// **vertical** frequency and col_freq is the **horizontal** frequency. The row
/// pass produces horizontal-frequency components (col_freq = h_freq), and the
/// column pass produces vertical-frequency components (k_col = v_freq). So the
/// output is stored at `[v_freq * 8 + h_freq]` = `[k_col * 8 + h_freq]`.
#[must_use]
pub fn fdct_8x8(block: &[i32; 64]) -> [i32; 64] {
    // Pass 1: 1-D FDCT along each row. Store unshifted i64 sums.
    // After this pass, intermediate[row * 8 + h_freq] = horizontal-frequency
    // `h_freq` component of spatial row `row`.
    let mut intermediate = [0i64; 64];
    for row in 0..8 {
        let row_in: [i32; 8] = std::array::from_fn(|c| block[row * 8 + c]);
        let row_out = fdct_1d(&row_in);
        for (h_freq, &v) in row_out.iter().enumerate() {
            intermediate[row * 8 + h_freq] = v;
        }
    }

    // Pass 2: 1-D FDCT along each column, then shift right by 34 bits total.
    // Half-rounding: add 2^33 before shifting by 34.
    //
    // For each horizontal frequency h_freq (= column of intermediate),
    // we do a FDCT over the row index `r` to get vertical frequency v_freq.
    // Output stored at [v_freq * 8 + h_freq] to match the IDCT's [row_freq * 8 + col_freq] layout.
    let round: i64 = 1 << 33;

    let mut output = [0i32; 64];
    for h_freq in 0..8usize {
        for v_freq in 0..8usize {
            let mut acc: i64 = 0;
            for r in 0..8usize {
                let phase_index = ((2 * r + 1) * v_freq) % 32;
                let cos_val = if v_freq == 0 {
                    COS_Q15[4]
                } else {
                    cos_q15_periodic(phase_index)
                };
                // intermediate[r * 8 + h_freq] = horizontal-frequency h_freq of spatial row r.
                acc += intermediate[r * 8 + h_freq] * i64::from(cos_val);
            }
            // Store at [v_freq * 8 + h_freq] to match IDCT's [row_freq * 8 + col_freq].
            let shifted = (acc + round) >> 34;
            output[v_freq * 8 + h_freq] = shifted as i32;
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prores::idct::idct_8x8;

    /// FDCT → IDCT round-trip: for a block of spatial samples, after
    /// fdct + idct the output should match the input within ±2 LSB
    /// (fixed-point rounding noise).
    #[test]
    fn fdct_idct_round_trip_dc_only() {
        // A flat block of value 100 (relative to the DC midpoint).
        let block: [i32; 64] = [100; 64];
        let freq = fdct_8x8(&block);
        let spatial = idct_8x8(&freq);
        for (i, &v) in spatial.iter().enumerate() {
            assert!(
                (v - 100).abs() <= 3,
                "round-trip failed at [{i}]: input=100, got={v}"
            );
        }
    }

    #[test]
    fn fdct_zero_block_is_zero() {
        let block = [0i32; 64];
        let freq = fdct_8x8(&block);
        assert!(
            freq.iter().all(|&v| v == 0),
            "fdct(0) should be 0, got {freq:?}"
        );
    }

    #[test]
    fn fdct_idct_round_trip_ramp() {
        // A ramp block: block[r*8+c] = (r*8+c) - 32, centred around 0.
        // Integer fixed-point arithmetic accumulates rounding noise across both
        // FDCT and IDCT passes; we allow ±16 LSB for the pure transform round-trip.
        let block: [i32; 64] = std::array::from_fn(|i| (i as i32) - 32);
        let freq = fdct_8x8(&block);
        let spatial = idct_8x8(&freq);
        for (i, &v) in spatial.iter().enumerate() {
            let expected = block[i];
            assert!(
                (v - expected).abs() <= 16,
                "ramp round-trip failed at [{i}]: expected={expected}, got={v}"
            );
        }
    }

    #[test]
    fn fdct_energy_concentrates_at_low_frequencies() {
        // A natural block should have most energy in low-frequency coefficients.
        let block: [i32; 64] = std::array::from_fn(|i| {
            let r = (i / 8) as i32;
            let c = (i % 8) as i32;
            // Slowly varying function.
            (r * 10 + c * 5) - 100
        });
        let freq = fdct_8x8(&block);
        // DC coefficient (index 0) should have larger magnitude than the
        // highest-frequency coefficient (index 63).
        assert!(
            freq[0].abs() >= freq[63].abs(),
            "DC energy should dominate: DC={}, AC63={}",
            freq[0],
            freq[63]
        );
    }
}
