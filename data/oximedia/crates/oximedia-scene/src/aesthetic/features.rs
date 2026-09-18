//! Aesthetic feature extraction.

use crate::error::{SceneError, SceneResult};
use serde::{Deserialize, Serialize};

/// Aesthetic features.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AestheticFeatures {
    /// Color features.
    pub color_features: Vec<f32>,
    /// Texture features.
    pub texture_features: Vec<f32>,
    /// Composition features.
    pub composition_features: Vec<f32>,
}

/// Feature extractor for aesthetics.
pub struct FeatureExtractor;

impl FeatureExtractor {
    /// Create a new feature extractor.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Extract aesthetic features.
    ///
    /// # Errors
    ///
    /// Returns error if extraction fails.
    pub fn extract(
        &self,
        rgb_data: &[u8],
        width: usize,
        height: usize,
    ) -> SceneResult<AestheticFeatures> {
        if rgb_data.len() != width * height * 3 {
            return Err(SceneError::InvalidDimensions(
                "RGB data size mismatch".to_string(),
            ));
        }

        let color_features = self.extract_color_features(rgb_data);
        let texture_features = self.extract_texture_features(rgb_data, width, height);
        let composition_features = self.extract_composition_features(rgb_data, width, height);

        Ok(AestheticFeatures {
            color_features,
            texture_features,
            composition_features,
        })
    }

    fn extract_color_features(&self, rgb_data: &[u8]) -> Vec<f32> {
        let mut features = Vec::new();

        // Color moments (mean, std)
        let mut means = [0.0f32; 3];
        let mut stds = [0.0f32; 3];

        for i in (0..rgb_data.len()).step_by(3) {
            for c in 0..3 {
                means[c] += rgb_data[i + c] as f32;
            }
        }

        let count = rgb_data.len() / 3;
        for c in 0..3 {
            means[c] /= count as f32;
        }

        for i in (0..rgb_data.len()).step_by(3) {
            for c in 0..3 {
                let diff = rgb_data[i + c] as f32 - means[c];
                stds[c] += diff * diff;
            }
        }

        for c in 0..3 {
            stds[c] = (stds[c] / count as f32).sqrt();
        }

        features.extend_from_slice(&means);
        features.extend_from_slice(&stds);

        features
    }

    fn extract_texture_features(&self, rgb_data: &[u8], width: usize, height: usize) -> Vec<f32> {
        let mut features = Vec::new();

        // Simple texture energy
        let mut energy = 0.0;
        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let idx = (y * width + x) * 3;
                for c in 0..3 {
                    let diff = (rgb_data[idx + c] as i32 - rgb_data[idx + 3 + c] as i32)
                        .unsigned_abs() as f32;
                    energy += diff * diff;
                }
            }
        }

        features.push((energy / ((width - 2) * (height - 2)) as f32 / 255.0 / 255.0).min(1.0));

        features
    }

    fn extract_composition_features(
        &self,
        _rgb_data: &[u8],
        _width: usize,
        _height: usize,
    ) -> Vec<f32> {
        // Placeholder for composition features
        vec![0.5; 4]
    }
}

impl Default for FeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}
