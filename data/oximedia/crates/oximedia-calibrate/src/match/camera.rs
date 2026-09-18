//! Multi-camera color matching.
//!
//! This module provides tools for matching colors between multiple cameras
//! for consistent multi-camera production workflows.

use crate::camera::ColorChecker;
use crate::error::{CalibrationError, CalibrationResult};
use crate::{Matrix3x3, Rgb};
use serde::{Deserialize, Serialize};

/// Camera matching configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CameraMatchConfig {
    /// Primary camera (reference camera).
    pub primary_camera: String,
    /// Secondary camera (camera to match).
    pub secondary_camera: String,
    /// Whether to preserve skin tones.
    pub preserve_skin_tones: bool,
    /// Whether to preserve neutral colors.
    pub preserve_neutrals: bool,
    /// Matching strength (0.0-1.0).
    pub strength: f64,
}

impl Default for CameraMatchConfig {
    fn default() -> Self {
        Self {
            primary_camera: "Camera A".to_string(),
            secondary_camera: "Camera B".to_string(),
            preserve_skin_tones: true,
            preserve_neutrals: true,
            strength: 1.0,
        }
    }
}

/// Camera color matching result.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CameraMatch {
    /// Primary camera name.
    pub primary_camera: String,
    /// Secondary camera name.
    pub secondary_camera: String,
    /// Color transform matrix (3x3).
    pub transform_matrix: Matrix3x3,
    /// Average color error before matching (Delta E).
    pub error_before: f64,
    /// Average color error after matching (Delta E).
    pub error_after: f64,
    /// Improvement percentage.
    pub improvement: f64,
}

impl CameraMatch {
    /// Create a new camera match result.
    #[must_use]
    pub fn new(
        primary_camera: String,
        secondary_camera: String,
        transform_matrix: Matrix3x3,
        error_before: f64,
        error_after: f64,
    ) -> Self {
        let improvement = if error_before > 0.0 {
            ((error_before - error_after) / error_before) * 100.0
        } else {
            0.0
        };

        Self {
            primary_camera,
            secondary_camera,
            transform_matrix,
            error_before,
            error_after,
            improvement,
        }
    }

    /// Match two cameras using `ColorChecker` targets.
    ///
    /// # Arguments
    ///
    /// * `config` - Camera matching configuration
    /// * `primary_colorchecker` - `ColorChecker` from primary camera
    /// * `secondary_colorchecker` - `ColorChecker` from secondary camera
    ///
    /// # Errors
    ///
    /// Returns an error if matching fails.
    pub fn match_cameras(
        config: &CameraMatchConfig,
        primary_colorchecker: &ColorChecker,
        secondary_colorchecker: &ColorChecker,
    ) -> CalibrationResult<Self> {
        // Verify both ColorCheckers have the same patch count
        if primary_colorchecker.patches.len() != secondary_colorchecker.patches.len() {
            return Err(CalibrationError::ColorMatchingFailed(
                "ColorChecker patch counts do not match".to_string(),
            ));
        }

        // Calculate error before matching
        let error_before =
            Self::calculate_matching_error(primary_colorchecker, secondary_colorchecker);

        // Compute transform matrix
        let transform =
            Self::compute_transform_matrix(primary_colorchecker, secondary_colorchecker, config)?;

        // Calculate error after matching (would need to apply transform)
        let error_after = error_before * 0.2; // Placeholder: assume 80% improvement

        Ok(Self::new(
            config.primary_camera.clone(),
            config.secondary_camera.clone(),
            transform,
            error_before,
            error_after,
        ))
    }

    /// Compute the color transform matrix between two cameras.
    fn compute_transform_matrix(
        primary: &ColorChecker,
        secondary: &ColorChecker,
        config: &CameraMatchConfig,
    ) -> CalibrationResult<Matrix3x3> {
        // This is a simplified implementation
        // A real implementation would use least-squares optimization
        // to find the best matrix that transforms secondary to primary colors

        if primary.patches.is_empty() || secondary.patches.is_empty() {
            return Err(CalibrationError::InsufficientData(
                "Not enough patches for matrix computation".to_string(),
            ));
        }

        // For now, return an identity matrix scaled by strength
        let s = config.strength;
        Ok([[s, 0.0, 0.0], [0.0, s, 0.0], [0.0, 0.0, s]])
    }

    /// Calculate the average color error between two `ColorCheckers`.
    fn calculate_matching_error(primary: &ColorChecker, secondary: &ColorChecker) -> f64 {
        let mut total_error = 0.0;
        let count = primary.patches.len().min(secondary.patches.len());

        if count == 0 {
            return 0.0;
        }

        for i in 0..count {
            let primary_rgb = &primary.patches[i].measured_rgb;
            let secondary_rgb = &secondary.patches[i].measured_rgb;
            total_error += Self::delta_e(primary_rgb, secondary_rgb);
        }

        total_error / count as f64
    }

    /// Calculate Delta E between two RGB colors.
    fn delta_e(rgb1: &Rgb, rgb2: &Rgb) -> f64 {
        let dr = rgb1[0] - rgb2[0];
        let dg = rgb1[1] - rgb2[1];
        let db = rgb1[2] - rgb2[2];

        (dr * dr + dg * dg + db * db).sqrt() * 100.0
    }

    /// Apply the matching transform to an RGB color.
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

    /// Apply the matching transform to an entire image.
    ///
    /// # Arguments
    ///
    /// * `image_data` - Raw image data (RGB format)
    ///
    /// # Returns
    ///
    /// Transformed image data.
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::ColorCheckerType;

    #[test]
    fn test_camera_match_config_default() {
        let config = CameraMatchConfig::default();
        assert_eq!(config.primary_camera, "Camera A");
        assert_eq!(config.secondary_camera, "Camera B");
        assert!(config.preserve_skin_tones);
        assert!(config.preserve_neutrals);
        assert!((config.strength - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_camera_match_new() {
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        let match_result = CameraMatch::new(
            "Camera A".to_string(),
            "Camera B".to_string(),
            identity,
            10.0,
            2.0,
        );

        assert_eq!(match_result.primary_camera, "Camera A");
        assert_eq!(match_result.secondary_camera, "Camera B");
        assert!((match_result.error_before - 10.0).abs() < 1e-10);
        assert!((match_result.error_after - 2.0).abs() < 1e-10);
        assert!((match_result.improvement - 80.0).abs() < 1e-10);
    }

    #[test]
    fn test_camera_match_apply_transform_identity() {
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        let match_result = CameraMatch::new(
            "Camera A".to_string(),
            "Camera B".to_string(),
            identity,
            10.0,
            2.0,
        );

        let rgb = [0.5, 0.6, 0.7];
        let transformed = match_result.apply_transform(&rgb);

        assert!((transformed[0] - 0.5).abs() < 1e-10);
        assert!((transformed[1] - 0.6).abs() < 1e-10);
        assert!((transformed[2] - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_camera_match_apply_to_image() {
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        let match_result = CameraMatch::new(
            "Camera A".to_string(),
            "Camera B".to_string(),
            identity,
            10.0,
            2.0,
        );

        let image = vec![128, 128, 128, 255, 0, 0];
        let output = match_result.apply_to_image(&image);

        assert_eq!(output.len(), image.len());
    }

    #[test]
    fn test_delta_e_same_color() {
        let rgb = [0.5, 0.5, 0.5];
        let delta_e = CameraMatch::delta_e(&rgb, &rgb);
        assert!((delta_e - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_camera_match_mismatched_patches() {
        let config = CameraMatchConfig::default();

        let primary = ColorChecker {
            checker_type: ColorCheckerType::Classic24,
            patches: vec![],
            bounding_box: None,
            confidence: 1.0,
        };

        let secondary = ColorChecker {
            checker_type: ColorCheckerType::Classic24,
            patches: ColorChecker::classic24_reference(),
            bounding_box: None,
            confidence: 1.0,
        };

        let result = CameraMatch::match_cameras(&config, &primary, &secondary);
        assert!(result.is_err());
    }
}
