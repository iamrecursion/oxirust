//! Technical quality classification (sharp/blurry, noisy, etc.).

use crate::error::{SceneError, SceneResult};
use serde::{Deserialize, Serialize};

/// Quality classification result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Overall quality score (0.0-1.0).
    pub overall_score: f32,
    /// Sharpness score (0.0-1.0).
    pub sharpness: f32,
    /// Noise level (0.0-1.0, higher is noisier).
    pub noise_level: f32,
    /// Contrast score (0.0-1.0).
    pub contrast: f32,
    /// Exposure quality (0.0-1.0).
    pub exposure: f32,
    /// Color balance (0.0-1.0).
    pub color_balance: f32,
    /// Compression artifacts (0.0-1.0, higher is worse).
    pub compression_artifacts: f32,
    /// Interlacing artifacts (0.0-1.0).
    pub interlacing: f32,
    /// Quality classification.
    pub classification: QualityClass,
}

/// Quality classification categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityClass {
    /// Excellent quality.
    Excellent,
    /// Good quality.
    Good,
    /// Fair quality.
    Fair,
    /// Poor quality.
    Poor,
    /// Very poor quality.
    VeryPoor,
}

impl QualityClass {
    /// Get human-readable name.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Excellent => "Excellent",
            Self::Good => "Good",
            Self::Fair => "Fair",
            Self::Poor => "Poor",
            Self::VeryPoor => "Very Poor",
        }
    }

    /// Create from overall score.
    #[must_use]
    pub fn from_score(score: f32) -> Self {
        if score >= 0.9 {
            Self::Excellent
        } else if score >= 0.75 {
            Self::Good
        } else if score >= 0.6 {
            Self::Fair
        } else if score >= 0.4 {
            Self::Poor
        } else {
            Self::VeryPoor
        }
    }
}

/// Configuration for quality analysis.
#[derive(Debug, Clone)]
pub struct QualityConfig {
    /// Enable sharpness analysis.
    pub analyze_sharpness: bool,
    /// Enable noise analysis.
    pub analyze_noise: bool,
    /// Enable contrast analysis.
    pub analyze_contrast: bool,
    /// Enable exposure analysis.
    pub analyze_exposure: bool,
    /// Enable color analysis.
    pub analyze_color: bool,
    /// Enable artifact detection.
    pub detect_artifacts: bool,
}

impl Default for QualityConfig {
    fn default() -> Self {
        Self {
            analyze_sharpness: true,
            analyze_noise: true,
            analyze_contrast: true,
            analyze_exposure: true,
            analyze_color: true,
            detect_artifacts: true,
        }
    }
}

/// Quality classifier for technical quality assessment.
pub struct QualityClassifier {
    config: QualityConfig,
}

impl QualityClassifier {
    /// Create a new quality classifier.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: QualityConfig::default(),
        }
    }

    /// Create with custom configuration.
    #[must_use]
    pub fn with_config(config: QualityConfig) -> Self {
        Self { config }
    }

    /// Analyze quality of an RGB image.
    ///
    /// # Arguments
    ///
    /// * `rgb_data` - RGB image data (height x width x 3)
    /// * `width` - Image width
    /// * `height` - Image height
    ///
    /// # Errors
    ///
    /// Returns error if analysis fails or invalid dimensions.
    pub fn analyze(
        &self,
        rgb_data: &[u8],
        width: usize,
        height: usize,
    ) -> SceneResult<QualityMetrics> {
        if rgb_data.len() != width * height * 3 {
            return Err(SceneError::InvalidDimensions(format!(
                "Expected {} bytes, got {}",
                width * height * 3,
                rgb_data.len()
            )));
        }

        let sharpness = if self.config.analyze_sharpness {
            self.measure_sharpness(rgb_data, width, height)
        } else {
            0.5
        };

        let noise_level = if self.config.analyze_noise {
            self.measure_noise(rgb_data, width, height)
        } else {
            0.0
        };

        let contrast = if self.config.analyze_contrast {
            self.measure_contrast(rgb_data, width, height)
        } else {
            0.5
        };

        let exposure = if self.config.analyze_exposure {
            self.measure_exposure(rgb_data, width, height)
        } else {
            0.5
        };

        let color_balance = if self.config.analyze_color {
            self.measure_color_balance(rgb_data, width, height)
        } else {
            0.5
        };

        let (compression_artifacts, interlacing) = if self.config.detect_artifacts {
            (
                self.detect_compression_artifacts(rgb_data, width, height),
                self.detect_interlacing(rgb_data, width, height),
            )
        } else {
            (0.0, 0.0)
        };

        // Calculate overall score
        let overall_score = self.calculate_overall_score(
            sharpness,
            noise_level,
            contrast,
            exposure,
            color_balance,
            compression_artifacts,
            interlacing,
        );

        let classification = QualityClass::from_score(overall_score);

        Ok(QualityMetrics {
            overall_score,
            sharpness,
            noise_level,
            contrast,
            exposure,
            color_balance,
            compression_artifacts,
            interlacing,
            classification,
        })
    }

    /// Measure sharpness using Laplacian variance.
    fn measure_sharpness(&self, rgb_data: &[u8], width: usize, height: usize) -> f32 {
        // Convert to grayscale and apply Laplacian
        let mut laplacian_sum = 0.0;
        let mut count = 0;

        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let idx = (y * width + x) * 3;

                // Convert to grayscale
                let center = 0.299 * rgb_data[idx] as f32
                    + 0.587 * rgb_data[idx + 1] as f32
                    + 0.114 * rgb_data[idx + 2] as f32;

                // Get neighbors
                let neighbors = [
                    ((y - 1) * width + x) * 3, // top
                    ((y + 1) * width + x) * 3, // bottom
                    (y * width + (x - 1)) * 3, // left
                    (y * width + (x + 1)) * 3, // right
                ];

                let mut neighbor_sum = 0.0;
                for &n_idx in &neighbors {
                    neighbor_sum += 0.299 * rgb_data[n_idx] as f32
                        + 0.587 * rgb_data[n_idx + 1] as f32
                        + 0.114 * rgb_data[n_idx + 2] as f32;
                }

                let laplacian = (4.0 * center - neighbor_sum).abs();
                laplacian_sum += laplacian;
                count += 1;
            }
        }

        let variance = laplacian_sum / count as f32;
        // Normalize to 0-1 range (empirical max ~100)
        (variance / 100.0).clamp(0.0, 1.0)
    }

    /// Measure noise level using local variance.
    fn measure_noise(&self, rgb_data: &[u8], width: usize, height: usize) -> f32 {
        let block_size = 8;
        let mut noise_sum = 0.0;
        let mut block_count = 0;

        for y in (0..height - block_size).step_by(block_size) {
            for x in (0..width - block_size).step_by(block_size) {
                // Calculate variance in this block
                let mut sum = 0.0;
                let mut sum_sq = 0.0;
                let mut count = 0;

                for dy in 0..block_size {
                    for dx in 0..block_size {
                        let idx = ((y + dy) * width + (x + dx)) * 3;
                        let gray = 0.299 * rgb_data[idx] as f32
                            + 0.587 * rgb_data[idx + 1] as f32
                            + 0.114 * rgb_data[idx + 2] as f32;
                        sum += gray;
                        sum_sq += gray * gray;
                        count += 1;
                    }
                }

                let mean = sum / count as f32;
                let variance = (sum_sq / count as f32) - (mean * mean);
                noise_sum += variance;
                block_count += 1;
            }
        }

        let avg_noise = noise_sum / block_count as f32;
        // Normalize to 0-1 range (empirical max ~500 for noisy images)
        (avg_noise / 500.0).clamp(0.0, 1.0)
    }

    /// Measure contrast using histogram.
    fn measure_contrast(&self, rgb_data: &[u8], _width: usize, _height: usize) -> f32 {
        let mut histogram = vec![0u32; 256];

        for i in (0..rgb_data.len()).step_by(3) {
            let gray = (0.299 * rgb_data[i] as f32
                + 0.587 * rgb_data[i + 1] as f32
                + 0.114 * rgb_data[i + 2] as f32) as usize;
            histogram[gray.min(255)] += 1;
        }

        // Calculate dynamic range
        let mut min_val = 0;
        let mut max_val = 255;
        let threshold = (rgb_data.len() / 3 / 100) as u32; // 1% threshold

        // Find min (first value with >1% pixels)
        for (i, &count) in histogram.iter().enumerate() {
            if count > threshold {
                min_val = i;
                break;
            }
        }

        // Find max (last value with >1% pixels)
        for (i, &count) in histogram.iter().enumerate().rev() {
            if count > threshold {
                max_val = i;
                break;
            }
        }

        let dynamic_range = (max_val - min_val) as f32 / 255.0;
        dynamic_range.clamp(0.0, 1.0)
    }

    /// Measure exposure quality.
    fn measure_exposure(&self, rgb_data: &[u8], _width: usize, _height: usize) -> f32 {
        let mut brightness_sum = 0.0;
        let mut overexposed = 0;
        let mut underexposed = 0;
        let pixel_count = rgb_data.len() / 3;

        for i in (0..rgb_data.len()).step_by(3) {
            let gray = 0.299 * rgb_data[i] as f32
                + 0.587 * rgb_data[i + 1] as f32
                + 0.114 * rgb_data[i + 2] as f32;
            brightness_sum += gray;

            if gray > 250.0 {
                overexposed += 1;
            } else if gray < 5.0 {
                underexposed += 1;
            }
        }

        let avg_brightness = brightness_sum / pixel_count as f32;
        let overexposed_ratio = overexposed as f32 / pixel_count as f32;
        let underexposed_ratio = underexposed as f32 / pixel_count as f32;

        // Good exposure: brightness around 127, low clipping
        let brightness_score = 1.0 - ((avg_brightness - 127.0).abs() / 127.0);
        let clipping_penalty = (overexposed_ratio + underexposed_ratio) * 2.0;

        (brightness_score - clipping_penalty).clamp(0.0, 1.0)
    }

    /// Measure color balance.
    fn measure_color_balance(&self, rgb_data: &[u8], _width: usize, _height: usize) -> f32 {
        let mut r_sum = 0u64;
        let mut g_sum = 0u64;
        let mut b_sum = 0u64;
        let pixel_count = rgb_data.len() / 3;

        for i in (0..rgb_data.len()).step_by(3) {
            r_sum += u64::from(rgb_data[i]);
            g_sum += u64::from(rgb_data[i + 1]);
            b_sum += u64::from(rgb_data[i + 2]);
        }

        let r_avg = r_sum as f32 / pixel_count as f32;
        let g_avg = g_sum as f32 / pixel_count as f32;
        let b_avg = b_sum as f32 / pixel_count as f32;

        // Good balance: R, G, B averages close to each other
        let max_avg = r_avg.max(g_avg).max(b_avg);
        let min_avg = r_avg.min(g_avg).min(b_avg);

        if max_avg == 0.0 {
            return 0.5;
        }

        let balance = 1.0 - ((max_avg - min_avg) / max_avg);
        balance.clamp(0.0, 1.0)
    }

    /// Detect compression artifacts (blocking).
    fn detect_compression_artifacts(&self, rgb_data: &[u8], width: usize, height: usize) -> f32 {
        let block_size = 8;
        let mut edge_discontinuity = 0.0;
        let mut count = 0;

        // Check for discontinuities at 8x8 block boundaries
        for y in (block_size..height).step_by(block_size) {
            for x in 0..width - 1 {
                let idx_above = ((y - 1) * width + x) * 3;
                let idx_below = (y * width + x) * 3;

                for c in 0..3 {
                    let diff = (rgb_data[idx_below + c] as i32 - rgb_data[idx_above + c] as i32)
                        .unsigned_abs() as f32;
                    edge_discontinuity += diff;
                }
                count += 3;
            }
        }

        for x in (block_size..width).step_by(block_size) {
            for y in 0..height - 1 {
                let idx_left = (y * width + (x - 1)) * 3;
                let idx_right = (y * width + x) * 3;

                for c in 0..3 {
                    let diff = (rgb_data[idx_right + c] as i32 - rgb_data[idx_left + c] as i32)
                        .unsigned_abs() as f32;
                    edge_discontinuity += diff;
                }
                count += 3;
            }
        }

        if count == 0 {
            return 0.0;
        }

        let avg_discontinuity = edge_discontinuity / count as f32;
        (avg_discontinuity / 30.0).clamp(0.0, 1.0)
    }

    /// Detect interlacing artifacts.
    fn detect_interlacing(&self, rgb_data: &[u8], width: usize, height: usize) -> f32 {
        let mut field_diff = 0.0;
        let mut count = 0;

        // Compare even and odd lines
        for y in (2..height - 2).step_by(2) {
            for x in 0..width {
                let idx_even = (y * width + x) * 3;
                let idx_odd_above = ((y - 1) * width + x) * 3;
                let idx_odd_below = ((y + 1) * width + x) * 3;

                for c in 0..3 {
                    let even_val = rgb_data[idx_even + c] as i32;
                    let odd_avg = (rgb_data[idx_odd_above + c] as i32
                        + rgb_data[idx_odd_below + c] as i32)
                        / 2;
                    let diff = (even_val - odd_avg).unsigned_abs() as f32;
                    field_diff += diff;
                }
                count += 3;
            }
        }

        if count == 0 {
            return 0.0;
        }

        let avg_diff = field_diff / count as f32;
        (avg_diff / 50.0).clamp(0.0, 1.0)
    }

    /// Calculate overall quality score from components.
    #[allow(clippy::too_many_arguments)]
    fn calculate_overall_score(
        &self,
        sharpness: f32,
        noise_level: f32,
        contrast: f32,
        exposure: f32,
        color_balance: f32,
        compression_artifacts: f32,
        interlacing: f32,
    ) -> f32 {
        let mut score = 0.0;
        let mut weight_sum = 0.0;

        if self.config.analyze_sharpness {
            score += sharpness * 0.25;
            weight_sum += 0.25;
        }
        if self.config.analyze_noise {
            score += (1.0 - noise_level) * 0.20;
            weight_sum += 0.20;
        }
        if self.config.analyze_contrast {
            score += contrast * 0.20;
            weight_sum += 0.20;
        }
        if self.config.analyze_exposure {
            score += exposure * 0.20;
            weight_sum += 0.20;
        }
        if self.config.analyze_color {
            score += color_balance * 0.10;
            weight_sum += 0.10;
        }
        if self.config.detect_artifacts {
            score += (1.0 - compression_artifacts) * 0.025;
            score += (1.0 - interlacing) * 0.025;
            weight_sum += 0.05;
        }

        if weight_sum > 0.0 {
            (score / weight_sum).clamp(0.0, 1.0)
        } else {
            0.5
        }
    }
}

impl Default for QualityClassifier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_class_from_score() {
        assert_eq!(QualityClass::from_score(0.95), QualityClass::Excellent);
        assert_eq!(QualityClass::from_score(0.8), QualityClass::Good);
        assert_eq!(QualityClass::from_score(0.65), QualityClass::Fair);
        assert_eq!(QualityClass::from_score(0.5), QualityClass::Poor);
        assert_eq!(QualityClass::from_score(0.2), QualityClass::VeryPoor);
    }

    #[test]
    fn test_quality_classifier() {
        let classifier = QualityClassifier::new();
        let width = 100;
        let height = 100;

        // Create a reasonably good quality image
        let mut rgb_data = vec![128u8; width * height * 3];
        for i in (0..rgb_data.len()).step_by(3) {
            rgb_data[i] = ((i / 3) % 256) as u8;
            rgb_data[i + 1] = ((i / 3) % 128) as u8;
            rgb_data[i + 2] = ((i / 3) % 64) as u8;
        }

        let result = classifier.analyze(&rgb_data, width, height);
        assert!(result.is_ok());

        let metrics = result.expect("should succeed in test");
        assert!(metrics.overall_score >= 0.0 && metrics.overall_score <= 1.0);
        assert!(metrics.sharpness >= 0.0 && metrics.sharpness <= 1.0);
        assert!(metrics.noise_level >= 0.0 && metrics.noise_level <= 1.0);
    }

    #[test]
    fn test_quality_class_name() {
        assert_eq!(QualityClass::Excellent.name(), "Excellent");
        assert_eq!(QualityClass::Poor.name(), "Poor");
    }
}
