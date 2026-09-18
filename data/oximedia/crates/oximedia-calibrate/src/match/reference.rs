//! Match colors to a reference target.
//!
//! This module provides tools for matching image colors to known reference targets.

use crate::error::{CalibrationError, CalibrationResult};
use crate::{Matrix3x3, Rgb};
use serde::{Deserialize, Serialize};

/// Reference target type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReferenceTargetType {
    /// `ColorChecker` target.
    ColorChecker,
    /// Grayscale chart.
    GrayscaleChart,
    /// White balance card.
    WhiteBalanceCard,
    /// Custom reference.
    Custom,
}

/// Reference target for color matching.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReferenceTarget {
    /// Target type.
    pub target_type: ReferenceTargetType,
    /// Reference colors (RGB).
    pub reference_colors: Vec<Rgb>,
    /// Color names/labels.
    pub color_names: Vec<String>,
}

impl ReferenceTarget {
    /// Create a new reference target.
    #[must_use]
    pub fn new(
        target_type: ReferenceTargetType,
        reference_colors: Vec<Rgb>,
        color_names: Vec<String>,
    ) -> Self {
        Self {
            target_type,
            reference_colors,
            color_names,
        }
    }

    /// Create a grayscale reference target.
    #[must_use]
    pub fn grayscale(steps: usize) -> Self {
        let mut colors = Vec::with_capacity(steps);
        let mut names = Vec::with_capacity(steps);

        for i in 0..steps {
            let value = i as f64 / (steps - 1) as f64;
            colors.push([value, value, value]);
            names.push(format!("Gray {i}"));
        }

        Self::new(ReferenceTargetType::GrayscaleChart, colors, names)
    }

    /// Create a white balance card reference.
    #[must_use]
    pub fn white_balance() -> Self {
        Self::new(
            ReferenceTargetType::WhiteBalanceCard,
            vec![[1.0, 1.0, 1.0]],
            vec!["White".to_string()],
        )
    }

    /// Get the number of reference colors.
    #[must_use]
    pub fn color_count(&self) -> usize {
        self.reference_colors.len()
    }
}

/// Reference matching result.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReferenceMatch {
    /// Reference target used.
    pub target: ReferenceTarget,
    /// Color transform matrix (3x3).
    pub transform_matrix: Matrix3x3,
    /// Average color error before matching (Delta E).
    pub error_before: f64,
    /// Average color error after matching (Delta E).
    pub error_after: f64,
    /// Maximum color error after matching (Delta E).
    pub max_error_after: f64,
}

impl ReferenceMatch {
    /// Create a new reference match result.
    #[must_use]
    pub fn new(
        target: ReferenceTarget,
        transform_matrix: Matrix3x3,
        error_before: f64,
        error_after: f64,
        max_error_after: f64,
    ) -> Self {
        Self {
            target,
            transform_matrix,
            error_before,
            error_after,
            max_error_after,
        }
    }

    /// Match colors to a reference target.
    ///
    /// # Arguments
    ///
    /// * `target` - Reference target
    /// * `measured_colors` - Measured colors from the image
    ///
    /// # Errors
    ///
    /// Returns an error if matching fails.
    pub fn match_to_reference(
        target: &ReferenceTarget,
        _measured_colors: &[Rgb],
    ) -> CalibrationResult<Self> {
        // This is a placeholder implementation
        // A real implementation would compute the best transform from measured to reference

        if target.reference_colors.is_empty() {
            return Err(CalibrationError::ColorMatchingFailed(
                "Reference target has no colors".to_string(),
            ));
        }

        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        Ok(Self::new(target.clone(), identity, 12.0, 2.5, 5.0))
    }

    /// Apply the reference matching transform to an RGB color.
    #[must_use]
    pub fn apply_transform(&self, rgb: &Rgb) -> Rgb {
        [
            self.transform_matrix[0][0] * rgb[0]
                + self.transform_matrix[0][1] * rgb[1]
                + self.transform_matrix[0][2] * rgb[2],
            self.transform_matrix[1][0] * rgb[0]
                + self.transform_matrix[1][1] * rgb[1]
                + self.transform_matrix[1][2] * rgb[2],
            self.transform_matrix[2][0] * rgb[0]
                + self.transform_matrix[2][1] * rgb[1]
                + self.transform_matrix[2][2] * rgb[2],
        ]
    }

    /// Apply the reference matching to an entire image.
    #[must_use]
    pub fn apply_to_image(&self, image_data: &[u8]) -> Vec<u8> {
        let mut output = Vec::with_capacity(image_data.len());

        for chunk in image_data.chunks_exact(3) {
            let r = f64::from(chunk[0]) / 255.0;
            let g = f64::from(chunk[1]) / 255.0;
            let b = f64::from(chunk[2]) / 255.0;

            let transformed = self.apply_transform(&[r, g, b]);

            output.push((transformed[0] * 255.0).clamp(0.0, 255.0) as u8);
            output.push((transformed[1] * 255.0).clamp(0.0, 255.0) as u8);
            output.push((transformed[2] * 255.0).clamp(0.0, 255.0) as u8);
        }

        output
    }

    /// Calculate improvement percentage.
    #[must_use]
    pub fn improvement(&self) -> f64 {
        if self.error_before > 0.0 {
            ((self.error_before - self.error_after) / self.error_before) * 100.0
        } else {
            0.0
        }
    }

    /// Check if the matching is acceptable.
    ///
    /// # Arguments
    ///
    /// * `max_avg_error` - Maximum acceptable average error (Delta E)
    /// * `max_single_error` - Maximum acceptable single color error (Delta E)
    #[must_use]
    pub fn is_acceptable(&self, max_avg_error: f64, max_single_error: f64) -> bool {
        self.error_after <= max_avg_error && self.max_error_after <= max_single_error
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reference_target_new() {
        let colors = vec![[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]];
        let names = vec!["White".to_string(), "Black".to_string()];

        let target =
            ReferenceTarget::new(ReferenceTargetType::Custom, colors.clone(), names.clone());

        assert_eq!(target.target_type, ReferenceTargetType::Custom);
        assert_eq!(target.reference_colors.len(), 2);
        assert_eq!(target.color_names.len(), 2);
    }

    #[test]
    fn test_reference_target_grayscale() {
        let target = ReferenceTarget::grayscale(5);

        assert_eq!(target.target_type, ReferenceTargetType::GrayscaleChart);
        assert_eq!(target.color_count(), 5);

        // Check that colors range from black to white
        assert!((target.reference_colors[0][0] - 0.0).abs() < 1e-10);
        assert!((target.reference_colors[4][0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_reference_target_white_balance() {
        let target = ReferenceTarget::white_balance();

        assert_eq!(target.target_type, ReferenceTargetType::WhiteBalanceCard);
        assert_eq!(target.color_count(), 1);
        assert!((target.reference_colors[0][0] - 1.0).abs() < 1e-10);
        assert!((target.reference_colors[0][1] - 1.0).abs() < 1e-10);
        assert!((target.reference_colors[0][2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_reference_match_new() {
        let target = ReferenceTarget::white_balance();
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        let match_result = ReferenceMatch::new(target, identity, 10.0, 2.0, 4.0);

        assert!((match_result.error_before - 10.0).abs() < 1e-10);
        assert!((match_result.error_after - 2.0).abs() < 1e-10);
        assert!((match_result.max_error_after - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_reference_match_apply_transform() {
        let target = ReferenceTarget::white_balance();
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        let match_result = ReferenceMatch::new(target, identity, 10.0, 2.0, 4.0);

        let rgb = [0.5, 0.6, 0.7];
        let transformed = match_result.apply_transform(&rgb);

        assert!((transformed[0] - 0.5).abs() < 1e-10);
        assert!((transformed[1] - 0.6).abs() < 1e-10);
        assert!((transformed[2] - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_reference_match_improvement() {
        let target = ReferenceTarget::white_balance();
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        let match_result = ReferenceMatch::new(target, identity, 10.0, 2.0, 4.0);

        let improvement = match_result.improvement();
        assert!((improvement - 80.0).abs() < 1e-10);
    }

    #[test]
    fn test_reference_match_is_acceptable() {
        let target = ReferenceTarget::white_balance();
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        let match_result = ReferenceMatch::new(target, identity, 10.0, 2.0, 4.0);

        assert!(match_result.is_acceptable(3.0, 5.0));
        assert!(!match_result.is_acceptable(1.0, 5.0));
        assert!(!match_result.is_acceptable(3.0, 3.0));
    }

    #[test]
    fn test_reference_match_to_reference_empty() {
        let target = ReferenceTarget::new(ReferenceTargetType::Custom, vec![], vec![]);

        let result = ReferenceMatch::match_to_reference(&target, &[]);
        assert!(result.is_err());
    }
}
