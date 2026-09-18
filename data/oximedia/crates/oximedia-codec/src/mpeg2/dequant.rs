//! Intra inverse quantisation for MPEG-2 (ISO/IEC 13818-2 §7.4).
//!
//! For **intra** blocks the quantised coefficients `QF[v][u]` are reconstructed
//! to `F[v][u]` as follows:
//!
//! - **DC** (`u == v == 0`): `F[0][0] = intra_dc_mult · QF[0][0]`, where
//!   `intra_dc_mult = [8, 4, 2, 1][intra_dc_precision]` corresponding to
//!   8/9/10/11-bit DC precision. The DC value `QF[0][0]` is the already
//!   DPCM-reconstructed differential predictor output.
//! - **AC**: `F[v][u] = (2·QF[v][u]·W[v][u]·quantiser_scale) / 32`, where `W`
//!   is the intra quantiser matrix (default or downloaded).
//!
//! After the per-coefficient reconstruction the values are **saturated** to
//! `[-2048, 2047]` (§7.4.3) and **mismatch control** (§7.4.4) is applied: the
//! sum of all reconstructed coefficients is computed; if it is even, the LSB of
//! `F[7][7]` is toggled.
//!
//! All arithmetic is integer; `quantiser_scale` is the value derived from the
//! `quantiser_scale_code` via the linear or non-linear mapping (`q_scale_type`).

/// Default intra quantiser matrix (ISO/IEC 13818-2 Table, §6.3.11 / Figure 7-1),
/// in **zig-zag** order is *not* used here — this is the raster-order default
/// intra matrix (the one signalled when `load_intra_quantiser_matrix == 0`).
pub const DEFAULT_INTRA_MATRIX: [u8; 64] = [
    8, 16, 19, 22, 26, 27, 29, 34, //
    16, 16, 22, 24, 27, 29, 34, 37, //
    19, 22, 26, 27, 29, 34, 34, 38, //
    22, 22, 26, 27, 29, 34, 37, 40, //
    22, 26, 27, 29, 32, 35, 40, 48, //
    26, 27, 29, 32, 35, 40, 48, 58, //
    26, 27, 29, 34, 38, 46, 56, 69, //
    27, 29, 35, 38, 46, 56, 69, 83,
];

/// Default non-intra quantiser matrix (all 16s). Provided for completeness;
/// the intra-only decoder does not use it for reconstruction but the sequence
/// header may carry a downloaded copy.
pub const DEFAULT_NON_INTRA_MATRIX: [u8; 64] = [16u8; 64];

/// Lower saturation bound for reconstructed DCT coefficients (§7.4.3).
pub const F_MIN: i32 = -2048;
/// Upper saturation bound for reconstructed DCT coefficients (§7.4.3).
pub const F_MAX: i32 = 2047;

/// Compute `intra_dc_mult` from `intra_dc_precision` (0..=3 → 8/9/10/11-bit DC).
#[must_use]
pub fn intra_dc_mult(intra_dc_precision: u8) -> i32 {
    match intra_dc_precision & 0x03 {
        0 => 8,
        1 => 4,
        2 => 2,
        _ => 1,
    }
}

/// Non-linear / linear `quantiser_scale` mapping (ISO/IEC 13818-2 Table 7-6).
///
/// `quantiser_scale_code` is in `1..=31`. If `q_scale_type == false` the scale
/// is `2 · code` (linear); otherwise it follows the non-linear table.
#[must_use]
pub fn quantiser_scale(quantiser_scale_code: u8, q_scale_type: bool) -> i32 {
    let code = quantiser_scale_code as usize;
    if !q_scale_type {
        // Linear: scale = 2 * code.
        2 * code as i32
    } else {
        // Non-linear mapping, ISO/IEC 13818-2 Table 7-6 (index 1..=31).
        const NONLINEAR: [i32; 32] = [
            0, // index 0 unused
            1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 22, 24, 28, 32, 36, 40, 44, 48, 52, 56,
            64, 72, 80, 88, 96, 104, 112,
        ];
        NONLINEAR[code.min(31)]
    }
}

/// Reconstruct one intra 8×8 block from quantised coefficients.
///
/// - `quantised` holds `QF[v][u]` in **raster order**; `quantised[0]` is the
///   already-DPCM-reconstructed DC value.
/// - `intra_matrix` is the active intra quantiser matrix (`W`), raster order.
/// - `intra_dc_precision` selects the DC multiplier.
/// - `q_scale` is the reconstructed `quantiser_scale` integer.
///
/// Returns the reconstructed `F[v][u]` in raster order, saturated and with
/// mismatch control applied.
#[must_use]
pub fn dequantize_intra(
    quantised: &[i32; 64],
    intra_matrix: &[u8; 64],
    intra_dc_precision: u8,
    q_scale: i32,
) -> [i32; 64] {
    let mut f = [0i32; 64];

    // DC term: F[0][0] = intra_dc_mult * QF[0][0].
    f[0] = intra_dc_mult(intra_dc_precision)
        .saturating_mul(quantised[0])
        .clamp(F_MIN, F_MAX);

    // AC terms: F = (2 * QF * W * quantiser_scale) / 32, then saturate.
    for i in 1..64 {
        let qf = quantised[i];
        if qf == 0 {
            f[i] = 0;
            continue;
        }
        let w = i64::from(intra_matrix[i]);
        let recon = (2 * i64::from(qf) * w * i64::from(q_scale)) / 32;
        f[i] = recon.clamp(i64::from(F_MIN), i64::from(F_MAX)) as i32;
    }

    // Mismatch control (§7.4.4): toggle LSB of F[63] if the coefficient sum is
    // even.
    let sum: i64 = f.iter().map(|&v| i64::from(v)).sum();
    if sum & 1 == 0 {
        // Toggle the LSB of the last coefficient (raster position 63).
        if f[63] & 1 != 0 {
            f[63] -= 1;
        } else {
            f[63] += 1;
        }
        // Re-saturate in case the toggle pushed it past the bound.
        f[63] = f[63].clamp(F_MIN, F_MAX);
    }

    f
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intra_dc_mult_table() {
        assert_eq!(intra_dc_mult(0), 8);
        assert_eq!(intra_dc_mult(1), 4);
        assert_eq!(intra_dc_mult(2), 2);
        assert_eq!(intra_dc_mult(3), 1);
    }

    #[test]
    fn dc_scaling_per_precision() {
        // QF[0] = 100 → F[0] = intra_dc_mult * 100, but saturated to 2047.
        for (prec, mult) in [(0u8, 8i32), (1, 4), (2, 2), (3, 1)] {
            let mut q = [0i32; 64];
            q[0] = 10;
            let f = dequantize_intra(&q, &DEFAULT_INTRA_MATRIX, prec, 2);
            assert_eq!(f[0], 10 * mult, "precision {prec}");
        }
    }

    #[test]
    fn dc_saturates_high() {
        let mut q = [0i32; 64];
        q[0] = 1000; // 1000 * 8 = 8000 > 2047 → saturates.
        let f = dequantize_intra(&q, &DEFAULT_INTRA_MATRIX, 0, 2);
        assert_eq!(f[0], F_MAX);
    }

    #[test]
    fn quantiser_scale_linear() {
        assert_eq!(quantiser_scale(1, false), 2);
        assert_eq!(quantiser_scale(8, false), 16);
        assert_eq!(quantiser_scale(31, false), 62);
    }

    #[test]
    fn quantiser_scale_nonlinear() {
        assert_eq!(quantiser_scale(1, true), 1);
        assert_eq!(quantiser_scale(9, true), 10);
        assert_eq!(quantiser_scale(31, true), 112);
    }

    #[test]
    fn ac_reconstruction_formula() {
        // QF[1] = 3, W[1] = 16 (default), q_scale = 8.
        // F = (2 * 3 * 16 * 8) / 32 = 768 / 32 ... wait: 2*3*16*8 = 768; /32 = 24.
        let mut q = [0i32; 64];
        q[1] = 3;
        let mut m = [16u8; 64];
        m[1] = 16;
        let f = dequantize_intra(&q, &m, 0, 8);
        assert_eq!(f[1], (2 * 3 * 16 * 8) / 32);
    }

    #[test]
    fn ac_saturates() {
        let mut q = [0i32; 64];
        q[1] = 1000;
        let mut m = [255u8; 64];
        m[1] = 255;
        let f = dequantize_intra(&q, &m, 0, 31 * 2);
        assert_eq!(f[1], F_MAX);
    }

    #[test]
    fn mismatch_control_makes_sum_odd() {
        // Construct a block whose raw F sum is even, verify the toggle makes it
        // odd.
        let mut q = [0i32; 64];
        q[0] = 2; // F[0] = 16 (even).
                  // No AC → sum = 16 (even) → mismatch toggles F[63] from 0 to 1.
        let f = dequantize_intra(&q, &DEFAULT_INTRA_MATRIX, 0, 2);
        let sum: i64 = f.iter().map(|&v| i64::from(v)).sum();
        assert_eq!(sum & 1, 1, "sum must be odd after mismatch control");
        assert_eq!(f[63], 1);
    }

    #[test]
    fn mismatch_control_leaves_odd_sum_alone() {
        // F[0] = intra_dc_mult * 1 = 8 with q[0]=1 → even. Add an AC that makes
        // total odd before toggle.
        let mut q = [0i32; 64];
        q[0] = 1; // F[0] = 8
        q[1] = 1; // F[1] = (2*1*16*2)/32 = 2 → sum so far 10 (even).
                  // Hmm still even; pick values that yield odd. Use q[2] to add 1.
        let mut m = DEFAULT_INTRA_MATRIX;
        m[2] = 8; // F[2] = (2*1*8*2)/32 = 1 → odd contribution.
        q[2] = 1;
        let f = dequantize_intra(&q, &m, 0, 2);
        let sum: i64 = f.iter().map(|&v| i64::from(v)).sum();
        // After mismatch control the final sum is always odd.
        assert_eq!(sum & 1, 1);
    }
}
