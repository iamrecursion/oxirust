//! DNG dual-illuminant camera color profile.
//!
//! Digital Negative (DNG) cameras often ship two color matrices:
//! - `ColorMatrix1`: calibrated under Standard Illuminant A (~2850 K).
//! - `ColorMatrix2`: calibrated under D50 (~5000 K).
//!
//! This module provides interpolation between the two matrices by correlated
//! color temperature (CCT) using the Robertson reciprocal-CCT weight.

use crate::Matrix3x3;
use serde::{Deserialize, Serialize};

/// Dual-illuminant calibration data for DNG color processing.
///
/// Holds the two color matrices shipped in a DNG tag set and computes
/// per-CCT interpolated matrices using the Robertson weight formula.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DualIlluminantCalibration {
    /// Color matrix calibrated under D50 illuminant (~5000 K).
    pub matrix_d50: Matrix3x3,
    /// Color matrix calibrated under Standard Illuminant A (~2850 K).
    pub matrix_illuminant_a: Matrix3x3,
}

impl DualIlluminantCalibration {
    /// Create a new dual-illuminant calibration.
    ///
    /// # Arguments
    ///
    /// * `matrix_d50` – `ColorMatrix2` (D50, 5000 K).
    /// * `matrix_illuminant_a` – `ColorMatrix1` (Illuminant A, 2850 K).
    #[must_use]
    pub fn new(matrix_d50: Matrix3x3, matrix_illuminant_a: Matrix3x3) -> Self {
        Self {
            matrix_d50,
            matrix_illuminant_a,
        }
    }

    /// Interpolate the color matrix for a given correlated color temperature.
    ///
    /// Uses the Robertson reciprocal-CCT weight:
    ///
    /// ```text
    /// w = (1/cct − 1/5000) / (1/2850 − 1/5000)
    /// w = clamp(w, 0.0, 1.0)
    /// result = w × matrix_illuminant_a + (1−w) × matrix_d50
    /// ```
    ///
    /// At 2850 K → w = 1.0 → returns `matrix_illuminant_a`.
    /// At 5000 K → w = 0.0 → returns `matrix_d50`.
    ///
    /// # Arguments
    ///
    /// * `cct_kelvin` – Correlated color temperature in Kelvin (positive).
    #[must_use]
    pub fn interpolate_matrix_by_cct(&self, cct_kelvin: f64) -> Matrix3x3 {
        // Guard against invalid CCT.
        let cct = cct_kelvin.max(1.0);

        // Reciprocal-CCT Robertson weight.
        let inv_cct = 1.0 / cct;
        let inv_d50 = 1.0 / 5000.0;
        let inv_a = 1.0 / 2850.0;

        let w = ((inv_cct - inv_d50) / (inv_a - inv_d50)).clamp(0.0, 1.0);
        let w_inv = 1.0 - w;

        let mut result = [[0.0_f64; 3]; 3];
        for row in 0..3 {
            for col in 0..3 {
                result[row][col] =
                    w * self.matrix_illuminant_a[row][col] + w_inv * self.matrix_d50[row][col];
            }
        }
        result
    }
}

/// A DNG color profile backed by dual-illuminant calibration data.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DngColorProfile {
    /// Profile description / camera model string.
    pub description: String,
    /// Optional dual-illuminant calibration (requires `with_dual_illuminant`).
    pub dual_illuminant: Option<DualIlluminantCalibration>,
    /// Single forward matrix used when no dual-illuminant data is set.
    pub forward_matrix: Matrix3x3,
}

impl DngColorProfile {
    /// Create a DNG color profile with a single forward matrix.
    #[must_use]
    pub fn new(description: String, forward_matrix: Matrix3x3) -> Self {
        Self {
            description,
            dual_illuminant: None,
            forward_matrix,
        }
    }

    /// Create a DNG color profile with dual-illuminant calibration matrices.
    ///
    /// # Arguments
    ///
    /// * `description` – Profile description string.
    /// * `d50` – Color matrix for D50 illuminant (`ColorMatrix2`).
    /// * `illum_a` – Color matrix for Illuminant A (`ColorMatrix1`).
    #[must_use]
    pub fn with_dual_illuminant(description: String, d50: Matrix3x3, illum_a: Matrix3x3) -> Self {
        Self {
            description,
            forward_matrix: d50,
            dual_illuminant: Some(DualIlluminantCalibration::new(d50, illum_a)),
        }
    }

    /// Return the color matrix best suited for the given CCT.
    ///
    /// If dual-illuminant data is present the interpolated matrix is returned;
    /// otherwise `forward_matrix` is returned unchanged.
    ///
    /// # Arguments
    ///
    /// * `cct_kelvin` – Correlated color temperature.
    #[must_use]
    pub fn matrix_for_cct(&self, cct_kelvin: f64) -> Matrix3x3 {
        match &self.dual_illuminant {
            Some(di) => di.interpolate_matrix_by_cct(cct_kelvin),
            None => self.forward_matrix,
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference matrices (identity used for simplicity so assertions are exact).
    fn d50_matrix() -> Matrix3x3 {
        // "D50 matrix": identity for easy verification.
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    }

    fn illum_a_matrix() -> Matrix3x3 {
        // "Illuminant A matrix": all-twos so blending is visible.
        [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]
    }

    /// At exactly D50 (5000 K) the weight w=0 → result should equal `matrix_d50`.
    #[test]
    fn test_dual_illuminant_d50_limit() {
        let di = DualIlluminantCalibration::new(d50_matrix(), illum_a_matrix());
        let mat = di.interpolate_matrix_by_cct(5000.0);

        for row in 0..3 {
            for col in 0..3 {
                let expected = d50_matrix()[row][col];
                assert!(
                    (mat[row][col] - expected).abs() < 1e-10,
                    "at 5000 K matrix[{row}][{col}] should be {expected}, got {}",
                    mat[row][col]
                );
            }
        }
    }

    /// At exactly Illuminant A (2850 K) the weight w=1 → result should equal
    /// `matrix_illuminant_a`.
    #[test]
    fn test_dual_illuminant_illum_a_limit() {
        let di = DualIlluminantCalibration::new(d50_matrix(), illum_a_matrix());
        let mat = di.interpolate_matrix_by_cct(2850.0);

        for row in 0..3 {
            for col in 0..3 {
                let expected = illum_a_matrix()[row][col];
                assert!(
                    (mat[row][col] - expected).abs() < 1e-10,
                    "at 2850 K matrix[{row}][{col}] should be {expected}, got {}",
                    mat[row][col]
                );
            }
        }
    }

    /// At the midpoint CCT the result should be the average of the two matrices.
    ///
    /// The Robertson weight at the reciprocal midpoint `cct_mid` where
    /// `1/cct_mid = (1/2850 + 1/5000) / 2` gives w = 0.5.
    #[test]
    fn test_dual_illuminant_midpoint() {
        let inv_mid = (1.0_f64 / 2850.0 + 1.0_f64 / 5000.0) / 2.0;
        let cct_mid = 1.0 / inv_mid;

        let di = DualIlluminantCalibration::new(d50_matrix(), illum_a_matrix());
        let mat = di.interpolate_matrix_by_cct(cct_mid);

        // At w=0.5: result = 0.5 * illum_a + 0.5 * d50.
        // d50[0][0]=1.0, illum_a[0][0]=2.0 → expected = 1.5
        assert!(
            (mat[0][0] - 1.5).abs() < 1e-10,
            "midpoint [0][0] should be 1.5, got {}",
            mat[0][0]
        );
        // Off-diagonal elements are 0 in both → should remain 0.
        assert!(
            mat[0][1].abs() < 1e-10,
            "off-diagonal [0][1] should be 0.0, got {}",
            mat[0][1]
        );
    }

    /// `DngColorProfile::with_dual_illuminant` wires through correctly.
    #[test]
    fn test_dng_color_profile_with_dual_illuminant() {
        let profile = DngColorProfile::with_dual_illuminant(
            "Test".to_string(),
            d50_matrix(),
            illum_a_matrix(),
        );

        assert!(profile.dual_illuminant.is_some());

        // At D50 limit, matrix_for_cct should equal d50_matrix.
        let mat = profile.matrix_for_cct(5000.0);
        assert!((mat[0][0] - 1.0).abs() < 1e-10);

        // At Illuminant A limit, matrix_for_cct should equal illum_a_matrix.
        let mat_a = profile.matrix_for_cct(2850.0);
        assert!((mat_a[0][0] - 2.0).abs() < 1e-10);
    }

    /// When CCT is above 5000 K, w is clamped to 0 → returns `matrix_d50`.
    #[test]
    fn test_dual_illuminant_cct_above_d50_clamped() {
        let di = DualIlluminantCalibration::new(d50_matrix(), illum_a_matrix());
        let mat = di.interpolate_matrix_by_cct(10_000.0);

        // w is clamped to 0 → pure d50_matrix.
        assert!((mat[0][0] - 1.0).abs() < 1e-10);
    }

    /// When CCT is below 2850 K, w is clamped to 1 → returns `matrix_illuminant_a`.
    #[test]
    fn test_dual_illuminant_cct_below_a_clamped() {
        let di = DualIlluminantCalibration::new(d50_matrix(), illum_a_matrix());
        let mat = di.interpolate_matrix_by_cct(1000.0);

        // w is clamped to 1 → pure illum_a_matrix.
        assert!((mat[0][0] - 2.0).abs() < 1e-10);
    }
}
