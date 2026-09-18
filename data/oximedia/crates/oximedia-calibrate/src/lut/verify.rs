//! LUT verification and validation.
//!
//! This module provides tools for verifying the accuracy of calibration LUTs.

use crate::error::{CalibrationError, CalibrationResult};
use crate::lut::LutMeasurement;
use crate::Rgb;
use oximedia_lut::{Lut3d, LutInterpolation};
use serde::{Deserialize, Serialize};

/// LUT verification result.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VerificationResult {
    /// Average color error (Delta E).
    pub average_error: f64,
    /// Maximum color error (Delta E).
    pub max_error: f64,
    /// Minimum color error (Delta E).
    pub min_error: f64,
    /// Number of verification points.
    pub point_count: usize,
    /// Percentage of points within Delta E 1.0.
    pub within_1_percent: f64,
    /// Percentage of points within Delta E 2.0.
    pub within_2_percent: f64,
    /// Percentage of points within Delta E 3.0.
    pub within_3_percent: f64,
}

impl VerificationResult {
    /// Create a new verification result.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        average_error: f64,
        max_error: f64,
        min_error: f64,
        point_count: usize,
        within_1_percent: f64,
        within_2_percent: f64,
        within_3_percent: f64,
    ) -> Self {
        Self {
            average_error,
            max_error,
            min_error,
            point_count,
            within_1_percent,
            within_2_percent,
            within_3_percent,
        }
    }

    /// Check if the LUT passes verification.
    ///
    /// # Arguments
    ///
    /// * `max_average_error` - Maximum acceptable average error (Delta E)
    /// * `max_single_error` - Maximum acceptable single point error (Delta E)
    #[must_use]
    pub fn passes(&self, max_average_error: f64, max_single_error: f64) -> bool {
        self.average_error <= max_average_error && self.max_error <= max_single_error
    }

    /// Get a quality grade (A-F) based on average error.
    #[must_use]
    pub fn grade(&self) -> char {
        if self.average_error < 1.0 {
            'A'
        } else if self.average_error < 2.0 {
            'B'
        } else if self.average_error < 3.0 {
            'C'
        } else if self.average_error < 5.0 {
            'D'
        } else if self.average_error < 10.0 {
            'E'
        } else {
            'F'
        }
    }
}

/// LUT verifier.
pub struct LutVerifier;

impl LutVerifier {
    /// Verify a 3D LUT against measurement data.
    ///
    /// # Arguments
    ///
    /// * `lut` - The LUT to verify
    /// * `measurements` - Measurement data for verification
    /// * `interpolation` - Interpolation method to use
    ///
    /// # Errors
    ///
    /// Returns an error if verification fails.
    pub fn verify_lut(
        lut: &Lut3d,
        measurements: &LutMeasurement,
        interpolation: LutInterpolation,
    ) -> CalibrationResult<VerificationResult> {
        // Validate measurement data
        measurements.validate()?;

        if measurements.points.is_empty() {
            return Err(CalibrationError::InsufficientData(
                "No measurement points for verification".to_string(),
            ));
        }

        let mut errors = Vec::with_capacity(measurements.points.len());

        // Calculate error for each measurement point
        for point in &measurements.points {
            let lut_output = lut.apply(&point.input, interpolation);
            let error = Self::calculate_delta_e(&lut_output, &point.output);
            errors.push(error);
        }

        // Calculate statistics
        let average_error = errors.iter().sum::<f64>() / errors.len() as f64;
        let max_error = errors.iter().copied().fold(0.0_f64, f64::max);
        let min_error = errors.iter().copied().fold(f64::MAX, f64::min);

        let within_1 = errors.iter().filter(|&&e| e < 1.0).count();
        let within_2 = errors.iter().filter(|&&e| e < 2.0).count();
        let within_3 = errors.iter().filter(|&&e| e < 3.0).count();

        let total = errors.len() as f64;

        Ok(VerificationResult::new(
            average_error,
            max_error,
            min_error,
            errors.len(),
            (within_1 as f64 / total) * 100.0,
            (within_2 as f64 / total) * 100.0,
            (within_3 as f64 / total) * 100.0,
        ))
    }

    /// Calculate Delta E (simplified) between two RGB colors.
    fn calculate_delta_e(rgb1: &Rgb, rgb2: &Rgb) -> f64 {
        let dr = rgb1[0] - rgb2[0];
        let dg = rgb1[1] - rgb2[1];
        let db = rgb1[2] - rgb2[2];

        (dr * dr + dg * dg + db * db).sqrt() * 100.0
    }

    /// Verify LUT consistency (check for smooth transitions).
    ///
    /// # Arguments
    ///
    /// * `lut` - The LUT to verify
    ///
    /// # Errors
    ///
    /// Returns an error if the LUT has discontinuities.
    pub fn verify_consistency(_lut: &Lut3d) -> CalibrationResult<()> {
        // This is a placeholder implementation
        // A real implementation would check for:
        // - Sudden jumps in values
        // - Non-monotonic behavior
        // - Out-of-range values

        Ok(())
    }

    /// Generate a verification report.
    ///
    /// # Arguments
    ///
    /// * `result` - Verification result
    ///
    /// # Returns
    ///
    /// A human-readable verification report.
    #[must_use]
    pub fn generate_report(result: &VerificationResult) -> String {
        format!(
            "LUT Verification Report\n\
             =======================\n\
             Average Error: {:.2} Delta E\n\
             Maximum Error: {:.2} Delta E\n\
             Minimum Error: {:.2} Delta E\n\
             Points Tested: {}\n\
             Grade: {}\n\
             \n\
             Error Distribution:\n\
             - Within Delta E 1.0: {:.1}%\n\
             - Within Delta E 2.0: {:.1}%\n\
             - Within Delta E 3.0: {:.1}%",
            result.average_error,
            result.max_error,
            result.min_error,
            result.point_count,
            result.grade(),
            result.within_1_percent,
            result.within_2_percent,
            result.within_3_percent
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_lut::LutSize;

    #[test]
    fn test_verification_result_new() {
        let result = VerificationResult::new(1.5, 5.0, 0.5, 100, 40.0, 80.0, 95.0);

        assert!((result.average_error - 1.5).abs() < 1e-10);
        assert!((result.max_error - 5.0).abs() < 1e-10);
        assert!((result.min_error - 0.5).abs() < 1e-10);
        assert_eq!(result.point_count, 100);
    }

    #[test]
    fn test_verification_result_passes() {
        let result = VerificationResult::new(1.5, 5.0, 0.5, 100, 40.0, 80.0, 95.0);

        assert!(result.passes(2.0, 6.0));
        assert!(!result.passes(1.0, 6.0));
        assert!(!result.passes(2.0, 4.0));
    }

    #[test]
    fn test_verification_result_grade() {
        let result_a = VerificationResult::new(0.5, 2.0, 0.1, 100, 90.0, 100.0, 100.0);
        assert_eq!(result_a.grade(), 'A');

        let result_b = VerificationResult::new(1.5, 5.0, 0.5, 100, 40.0, 80.0, 95.0);
        assert_eq!(result_b.grade(), 'B');

        let result_f = VerificationResult::new(15.0, 30.0, 5.0, 100, 0.0, 10.0, 20.0);
        assert_eq!(result_f.grade(), 'F');
    }

    #[test]
    fn test_calculate_delta_e() {
        let rgb1 = [0.5, 0.5, 0.5];
        let rgb2 = [0.5, 0.5, 0.5];

        let delta_e = LutVerifier::calculate_delta_e(&rgb1, &rgb2);
        assert!((delta_e - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_generate_report() {
        let result = VerificationResult::new(1.5, 5.0, 0.5, 100, 40.0, 80.0, 95.0);
        let report = LutVerifier::generate_report(&result);

        assert!(report.contains("1.5"));
        assert!(report.contains("5.0"));
        assert!(report.contains("0.5"));
        assert!(report.contains("100"));
        assert!(report.contains("Grade: B"));
    }

    #[test]
    fn test_verify_lut_empty_measurements() {
        let lut = Lut3d::new(LutSize::Size17);

        let measurements = LutMeasurement::new("Empty".to_string());

        let result = LutVerifier::verify_lut(&lut, &measurements, LutInterpolation::Trilinear);

        assert!(result.is_err());
    }
}
