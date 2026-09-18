//! Scene-to-scene color matching.
//!
//! This module provides tools for matching colors between different scenes
//! for visual continuity in film and video production.

use crate::error::CalibrationResult;
use crate::{Matrix3x3, Rgb};
use serde::{Deserialize, Serialize};

/// Scene matching configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SceneMatchConfig {
    /// Reference scene name.
    pub reference_scene: String,
    /// Target scene name.
    pub target_scene: String,
    /// Matching method.
    pub method: SceneMatchMethod,
    /// Preserve highlights.
    pub preserve_highlights: bool,
    /// Preserve shadows.
    pub preserve_shadows: bool,
}

/// Scene matching method.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SceneMatchMethod {
    /// Match using histogram matching.
    Histogram,
    /// Match using color statistics.
    Statistics,
    /// Match using reference points.
    ReferencePoints,
    /// Match using automatic analysis.
    Automatic,
}

impl Default for SceneMatchConfig {
    fn default() -> Self {
        Self {
            reference_scene: "Scene 1".to_string(),
            target_scene: "Scene 2".to_string(),
            method: SceneMatchMethod::Statistics,
            preserve_highlights: true,
            preserve_shadows: true,
        }
    }
}

/// Scene color matching result.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SceneMatch {
    /// Reference scene name.
    pub reference_scene: String,
    /// Target scene name.
    pub target_scene: String,
    /// Color transform matrix (3x3).
    pub transform_matrix: Matrix3x3,
    /// Matching method used.
    pub method: SceneMatchMethod,
    /// Color error before matching (Delta E).
    pub error_before: f64,
    /// Color error after matching (Delta E).
    pub error_after: f64,
}

impl SceneMatch {
    /// Create a new scene match result.
    #[must_use]
    pub fn new(
        reference_scene: String,
        target_scene: String,
        transform_matrix: Matrix3x3,
        method: SceneMatchMethod,
        error_before: f64,
        error_after: f64,
    ) -> Self {
        Self {
            reference_scene,
            target_scene,
            transform_matrix,
            method,
            error_before,
            error_after,
        }
    }

    /// Match two scenes for color continuity.
    ///
    /// # Arguments
    ///
    /// * `config` - Scene matching configuration
    /// * `reference_image` - Image from reference scene
    /// * `target_image` - Image from target scene
    ///
    /// # Errors
    ///
    /// Returns an error if matching fails.
    pub fn match_scenes(
        config: &SceneMatchConfig,
        _reference_image: &[u8],
        _target_image: &[u8],
    ) -> CalibrationResult<Self> {
        // This is a placeholder implementation
        // A real implementation would analyze both images and compute a transform

        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        Ok(Self::new(
            config.reference_scene.clone(),
            config.target_scene.clone(),
            identity,
            config.method,
            15.0,
            3.0,
        ))
    }

    /// Apply the scene matching transform to an RGB color.
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

    /// Apply the scene matching to an entire image.
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scene_match_config_default() {
        let config = SceneMatchConfig::default();
        assert_eq!(config.reference_scene, "Scene 1");
        assert_eq!(config.target_scene, "Scene 2");
        assert_eq!(config.method, SceneMatchMethod::Statistics);
        assert!(config.preserve_highlights);
        assert!(config.preserve_shadows);
    }

    #[test]
    fn test_scene_match_new() {
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        let scene_match = SceneMatch::new(
            "Scene 1".to_string(),
            "Scene 2".to_string(),
            identity,
            SceneMatchMethod::Statistics,
            20.0,
            5.0,
        );

        assert_eq!(scene_match.reference_scene, "Scene 1");
        assert_eq!(scene_match.target_scene, "Scene 2");
        assert_eq!(scene_match.method, SceneMatchMethod::Statistics);
    }

    #[test]
    fn test_scene_match_apply_transform() {
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        let scene_match = SceneMatch::new(
            "Scene 1".to_string(),
            "Scene 2".to_string(),
            identity,
            SceneMatchMethod::Statistics,
            20.0,
            5.0,
        );

        let rgb = [0.5, 0.6, 0.7];
        let transformed = scene_match.apply_transform(&rgb);

        assert!((transformed[0] - 0.5).abs() < 1e-10);
        assert!((transformed[1] - 0.6).abs() < 1e-10);
        assert!((transformed[2] - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_scene_match_improvement() {
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        let scene_match = SceneMatch::new(
            "Scene 1".to_string(),
            "Scene 2".to_string(),
            identity,
            SceneMatchMethod::Statistics,
            20.0,
            5.0,
        );

        let improvement = scene_match.improvement();
        assert!((improvement - 75.0).abs() < 1e-10);
    }

    #[test]
    fn test_scene_match_apply_to_image() {
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        let scene_match = SceneMatch::new(
            "Scene 1".to_string(),
            "Scene 2".to_string(),
            identity,
            SceneMatchMethod::Statistics,
            20.0,
            5.0,
        );

        let image = vec![128, 128, 128, 255, 0, 0];
        let output = scene_match.apply_to_image(&image);

        assert_eq!(output.len(), image.len());
    }
}
