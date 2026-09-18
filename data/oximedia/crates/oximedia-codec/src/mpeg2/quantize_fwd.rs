//! Forward intra quantisation for MPEG-2 (ISO/IEC 13818-2 §7.4, inverse path).
//!
//! This is the encoder-side inverse of [`super::dequant::dequantize_intra`].
//! Given reconstructed DCT coefficients `F[v][u]` (raster order, as produced by
//! [`super::fdct::fdct_8x8`]) it computes the quantised coefficients `QF[v][u]`
//! that the decoder's inverse quantiser maps back to `≈ F`:
//!
//! - **DC** (`u == v == 0`): the decoder computes `F[0] = intra_dc_mult · QF[0]`
//!   with `intra_dc_mult = [8, 4, 2, 1][intra_dc_precision]`. The forward step
//!   is therefore `QF[0] = round(F[0] / intra_dc_mult)`.
//! - **AC**: the decoder computes `F = (2·QF·W·q_scale) / 32 = QF·W·q_scale / 16`.
//!   The forward step is `QF = round(16·F / (W·q_scale))`, clamped to the legal
//!   signed intra coefficient range.
//!
//! The reconstructed `F` after dequantisation differs from the input `F` only
//! by the quantiser step (`W·q_scale/16`), so a round-trip is bounded by one
//! quant step plus the IDCT/FDCT fixed-point noise.

use super::dequant::intra_dc_mult;

/// Smallest legal signed quantised AC coefficient the entropy coder emits.
///
/// MPEG-2 codes a level in 12-bit two's complement via the escape sequence
/// (`-2048..=2047`); `-2048` is the forbidden value (ISO/IEC 13818-2 §7.2.2.3)
/// so the usable signed range is `[-2047, 2047]`. A magnitude of `0` means the
/// coefficient is simply not coded (skipped in the run/level scan).
pub const QF_AC_MIN: i32 = -2047;
/// Largest legal signed quantised AC coefficient.
pub const QF_AC_MAX: i32 = 2047;
/// Smallest legal signed quantised DC differential predictor output.
pub const QF_DC_MIN: i32 = -2047;
/// Largest legal signed quantised DC value.
pub const QF_DC_MAX: i32 = 2047;

/// Round-to-nearest signed integer division (ties away from zero).
///
/// `den` must be non-zero; callers guarantee this (quant matrix entries are
/// `>= 1` and `q_scale >= 1`).
#[must_use]
fn round_div(num: i64, den: i64) -> i64 {
    if den == 0 {
        return 0;
    }
    if (num >= 0) == (den > 0) {
        (num + den.abs() / 2) / den
    } else {
        (num - den.abs() / 2) / den
    }
}

/// Forward-quantise the DC coefficient `f0` for the given `intra_dc_precision`.
///
/// Returns `QF[0] = round(f0 / intra_dc_mult)`, clamped to the legal DC range.
#[must_use]
pub fn quantize_dc(f0: i32, intra_dc_precision: u8) -> i32 {
    let mult = i64::from(intra_dc_mult(intra_dc_precision));
    let qf = round_div(i64::from(f0), mult);
    (qf as i32).clamp(QF_DC_MIN, QF_DC_MAX)
}

/// Forward-quantise one AC coefficient.
///
/// `f` is the reconstructed coefficient, `w` the active intra quant-matrix
/// entry (`>= 1`), `q_scale` the reconstructed quantiser scale (`>= 1`).
/// Returns `QF = round(16·f / (w·q_scale))`, clamped to `[QF_AC_MIN, QF_AC_MAX]`.
#[must_use]
pub fn quantize_ac(f: i32, w: u8, q_scale: i32) -> i32 {
    if f == 0 {
        return 0;
    }
    let den = i64::from(w) * i64::from(q_scale);
    let qf = round_div(16 * i64::from(f), den);
    (qf as i32).clamp(QF_AC_MIN, QF_AC_MAX)
}

/// Forward-quantise a full intra 8×8 block.
///
/// - `coeffs` holds `F[v][u]` in raster order (`coeffs[0]` is the DC term from
///   the forward DCT).
/// - `intra_matrix` is the active intra quant matrix `W` (raster order).
/// - `intra_dc_precision` selects the DC divisor.
/// - `q_scale` is the reconstructed `quantiser_scale` integer.
///
/// Returns `QF[v][u]` in raster order; `out[0]` is the **absolute** quantised
/// DC (the DPCM differential is formed later by the entropy stage).
#[must_use]
pub fn quantize_intra(
    coeffs: &[i32; 64],
    intra_matrix: &[u8; 64],
    intra_dc_precision: u8,
    q_scale: i32,
) -> [i32; 64] {
    let mut qf = [0i32; 64];
    qf[0] = quantize_dc(coeffs[0], intra_dc_precision);
    for i in 1..64 {
        qf[i] = quantize_ac(coeffs[i], intra_matrix[i], q_scale);
    }
    qf
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mpeg2::dequant::{dequantize_intra, DEFAULT_INTRA_MATRIX};

    #[test]
    fn round_div_rounds_to_nearest() {
        assert_eq!(round_div(10, 3), 3); // 3.33 → 3
        assert_eq!(round_div(11, 3), 4); // 3.67 → 4
        assert_eq!(round_div(-10, 3), -3);
        assert_eq!(round_div(-11, 3), -4);
        assert_eq!(round_div(5, 2), 3); // tie away from zero
        assert_eq!(round_div(-5, 2), -3);
    }

    #[test]
    fn dc_forward_inverse_round_trip() {
        // For each precision, QF then dequant should reproduce F0 within ±mult/2.
        for prec in 0u8..=3 {
            let mult = intra_dc_mult(prec);
            for f0 in [0i32, 8, 100, 1024, -16, -512] {
                let qf0 = quantize_dc(f0, prec);
                let recon = qf0 * mult;
                assert!(
                    (recon - f0).abs() <= mult,
                    "prec {prec}, f0 {f0}: recon {recon}"
                );
            }
        }
    }

    #[test]
    fn ac_forward_inverse_round_trip() {
        // QF = round(16 F / (W q)), dequant F' = QF W q / 16 ≈ F within one step.
        let w = 16u8;
        let q = 8i32;
        let step = (i32::from(w) * q) / 16; // = 8
        for f in [0i32, 8, 24, 100, -40, -160, 500] {
            let qf = quantize_ac(f, w, q);
            let recon = (2 * qf * i32::from(w) * q) / 32;
            assert!(
                (recon - f).abs() <= step,
                "f {f}: qf {qf}, recon {recon}, step {step}"
            );
        }
    }

    #[test]
    fn ac_zero_stays_zero() {
        assert_eq!(quantize_ac(0, 16, 8), 0);
    }

    #[test]
    fn ac_clamps_to_legal_range() {
        // A huge coefficient with tiny denominator saturates.
        let qf = quantize_ac(1_000_000, 1, 2);
        assert_eq!(qf, QF_AC_MAX);
        let qf = quantize_ac(-1_000_000, 1, 2);
        assert_eq!(qf, QF_AC_MIN);
    }

    #[test]
    fn quantize_intra_matches_dequant_dc() {
        // Build coefficients, quantise, dequantise, and confirm DC matches.
        let mut coeffs = [0i32; 64];
        coeffs[0] = 1024; // → QF0 = 128 at prec 0 (mult 8)
        let qf = quantize_intra(&coeffs, &DEFAULT_INTRA_MATRIX, 0, 4);
        assert_eq!(qf[0], 128);
        let recon = dequantize_intra(&qf, &DEFAULT_INTRA_MATRIX, 0, 4);
        assert_eq!(recon[0], 1024);
    }

    #[test]
    fn quantize_intra_full_block_round_trip() {
        // A smooth block: quantise then dequantise, errors bounded by step.
        let coeffs: [i32; 64] = std::array::from_fn(|i| {
            if i == 0 {
                1024
            } else {
                ((i as i32 % 7) - 3) * 20
            }
        });
        let q_scale = 8;
        let qf = quantize_intra(&coeffs, &DEFAULT_INTRA_MATRIX, 0, q_scale);
        let recon = dequantize_intra(&qf, &DEFAULT_INTRA_MATRIX, 0, q_scale);
        // DC exact at prec 0 because 1024 is a multiple of 8.
        assert_eq!(recon[0], 1024);
        for i in 1..64 {
            let step = (i32::from(DEFAULT_INTRA_MATRIX[i]) * q_scale) / 16 + 1;
            // Mismatch control can toggle F[63] by 1, so allow one extra LSB there.
            let tol = if i == 63 { step + 1 } else { step };
            assert!(
                (recon[i] - coeffs[i]).abs() <= tol.max(1),
                "coeff[{i}] {} recon {} step {step}",
                coeffs[i],
                recon[i]
            );
        }
    }
}
