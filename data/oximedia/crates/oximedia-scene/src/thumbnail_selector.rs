//! Thumbnail selector module for choosing the most visually representative frame per scene.
//!
//! Uses a combination of sharpness, color diversity, brightness, and composition
//! metrics to score frames and select the best candidate for scene thumbnails.

use crate::error::{SceneError, SceneResult};
use serde::{Deserialize, Serialize};

/// Score components for a candidate thumbnail frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThumbnailScore {
    /// Overall quality score (0.0-1.0).
    pub overall: f32,
    /// Sharpness/focus score (0.0-1.0).
    pub sharpness: f32,
    /// Color diversity (0.0-1.0).
    pub color_diversity: f32,
    /// Brightness appropriateness (0.0-1.0, penalizes too dark or too bright).
    pub brightness: f32,
    /// Contrast score (0.0-1.0).
    pub contrast: f32,
    /// Center-weighted interest (0.0-1.0).
    pub center_interest: f32,
    /// Frame index.
    pub frame_index: usize,
}

/// Configuration for thumbnail selection.
#[derive(Debug, Clone)]
pub struct ThumbnailSelectorConfig {
    /// Weight for sharpness in overall score.
    pub sharpness_weight: f32,
    /// Weight for color diversity.
    pub color_weight: f32,
    /// Weight for brightness.
    pub brightness_weight: f32,
    /// Weight for contrast.
    pub contrast_weight: f32,
    /// Weight for center interest.
    pub center_weight: f32,
    /// Target brightness (frames near this are preferred).
    pub target_brightness: f32,
}

impl Default for ThumbnailSelectorConfig {
    fn default() -> Self {
        Self {
            sharpness_weight: 0.3,
            color_weight: 0.2,
            brightness_weight: 0.15,
            contrast_weight: 0.15,
            center_weight: 0.2,
            target_brightness: 0.45,
        }
    }
}

/// Thumbnail selector that scores frames and picks the best one.
pub struct ThumbnailSelector {
    config: ThumbnailSelectorConfig,
}

impl ThumbnailSelector {
    /// Create a new thumbnail selector.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: ThumbnailSelectorConfig::default(),
        }
    }

    /// Create with custom configuration.
    #[must_use]
    pub fn with_config(config: ThumbnailSelectorConfig) -> Self {
        Self { config }
    }

    /// Score a single frame for thumbnail suitability.
    ///
    /// # Errors
    ///
    /// Returns error if dimensions are invalid.
    pub fn score_frame(
        &self,
        rgb_data: &[u8],
        width: usize,
        height: usize,
        frame_index: usize,
    ) -> SceneResult<ThumbnailScore> {
        if rgb_data.len() != width * height * 3 {
            return Err(SceneError::InvalidDimensions(
                "RGB data size mismatch".to_string(),
            ));
        }

        let sharpness = self.compute_sharpness(rgb_data, width, height);
        let color_diversity = self.compute_color_diversity(rgb_data);
        let brightness = self.compute_brightness_score(rgb_data);
        let contrast = self.compute_contrast(rgb_data);
        let center_interest = self.compute_center_interest(rgb_data, width, height);

        let overall = sharpness * self.config.sharpness_weight
            + color_diversity * self.config.color_weight
            + brightness * self.config.brightness_weight
            + contrast * self.config.contrast_weight
            + center_interest * self.config.center_weight;

        Ok(ThumbnailScore {
            overall: overall.clamp(0.0, 1.0),
            sharpness,
            color_diversity,
            brightness,
            contrast,
            center_interest,
            frame_index,
        })
    }

    /// Select the best thumbnail from a sequence of frames.
    ///
    /// Each frame is provided as (rgb_data, width, height, frame_index).
    ///
    /// # Errors
    ///
    /// Returns error if no frames are provided or scoring fails.
    pub fn select_best(
        &self,
        frames: &[(&[u8], usize, usize, usize)],
    ) -> SceneResult<ThumbnailScore> {
        if frames.is_empty() {
            return Err(SceneError::InsufficientData(
                "No frames to select from".to_string(),
            ));
        }

        let mut best: Option<ThumbnailScore> = None;

        for &(rgb_data, width, height, frame_index) in frames {
            let score = self.score_frame(rgb_data, width, height, frame_index)?;

            let is_better = best.as_ref().map_or(true, |b| score.overall > b.overall);

            if is_better {
                best = Some(score);
            }
        }

        best.ok_or_else(|| SceneError::InsufficientData("No frames scored".to_string()))
    }

    /// Compute sharpness using Laplacian variance.
    fn compute_sharpness(&self, rgb_data: &[u8], width: usize, height: usize) -> f32 {
        if width < 3 || height < 3 {
            return 0.0;
        }

        let mut laplacian_sum = 0.0_f64;
        let mut count = 0_u64;

        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let idx = (y * width + x) * 3;
                // Luminance of center and neighbors
                let lum = |i: usize| -> f32 {
                    0.299 * rgb_data[i] as f32
                        + 0.587 * rgb_data[i + 1] as f32
                        + 0.114 * rgb_data[i + 2] as f32
                };

                let center = lum(idx);
                let left = lum((y * width + x - 1) * 3);
                let right = lum((y * width + x + 1) * 3);
                let top = lum(((y - 1) * width + x) * 3);
                let bottom = lum(((y + 1) * width + x) * 3);

                let laplacian = (4.0 * center - left - right - top - bottom).abs();
                laplacian_sum += laplacian as f64;
                count += 1;
            }
        }

        if count == 0 {
            return 0.0;
        }

        let avg = laplacian_sum / count as f64;
        // Normalize: typical sharp images have avg ~10-50
        (avg as f32 / 30.0).clamp(0.0, 1.0)
    }

    /// Compute color diversity using a simplified histogram approach.
    fn compute_color_diversity(&self, rgb_data: &[u8]) -> f32 {
        // Quantize to 4x4x4 = 64 bins
        let mut bins = [0u32; 64];

        for chunk in rgb_data.chunks_exact(3) {
            let r_bin = (chunk[0] >> 6) as usize;
            let g_bin = (chunk[1] >> 6) as usize;
            let b_bin = (chunk[2] >> 6) as usize;
            let idx = r_bin * 16 + g_bin * 4 + b_bin;
            bins[idx] += 1;
        }

        // Count non-empty bins
        let occupied = bins.iter().filter(|&&b| b > 0).count();
        (occupied as f32 / 64.0).clamp(0.0, 1.0)
    }

    /// Compute brightness score (penalize extremes).
    fn compute_brightness_score(&self, rgb_data: &[u8]) -> f32 {
        let mut total_lum = 0.0_f64;
        let pixel_count = rgb_data.len() / 3;

        for chunk in rgb_data.chunks_exact(3) {
            total_lum +=
                (0.299 * chunk[0] as f64 + 0.587 * chunk[1] as f64 + 0.114 * chunk[2] as f64)
                    / 255.0;
        }

        if pixel_count == 0 {
            return 0.0;
        }

        let avg_brightness = total_lum / pixel_count as f64;
        // Score: 1.0 at target, falling off towards 0 and 1
        let distance = (avg_brightness as f32 - self.config.target_brightness).abs();
        (1.0 - distance * 2.0).clamp(0.0, 1.0)
    }

    /// Compute contrast using min-max of luminance.
    fn compute_contrast(&self, rgb_data: &[u8]) -> f32 {
        let mut min_lum = f32::MAX;
        let mut max_lum = f32::MIN;

        for chunk in rgb_data.chunks_exact(3) {
            let lum = 0.299 * chunk[0] as f32 + 0.587 * chunk[1] as f32 + 0.114 * chunk[2] as f32;
            min_lum = min_lum.min(lum);
            max_lum = max_lum.max(lum);
        }

        if max_lum <= min_lum {
            return 0.0;
        }

        ((max_lum - min_lum) / 255.0).clamp(0.0, 1.0)
    }

    /// Compute center-weighted interest (edges near center are more interesting).
    fn compute_center_interest(&self, rgb_data: &[u8], width: usize, height: usize) -> f32 {
        if width < 3 || height < 3 {
            return 0.0;
        }

        let cx = width as f32 / 2.0;
        let cy = height as f32 / 2.0;
        let max_dist = ((cx * cx + cy * cy) as f32).sqrt();

        let mut weighted_edge_sum = 0.0_f64;
        let mut weight_sum = 0.0_f64;

        // Sample every 4th pixel for speed
        let step = 4;
        for y in (1..height - 1).step_by(step) {
            for x in (1..width - 1).step_by(step) {
                let idx = (y * width + x) * 3;
                let idx_next = (y * width + x + 1) * 3;
                if idx_next + 2 < rgb_data.len() {
                    let edge = (rgb_data[idx] as i32 - rgb_data[idx_next] as i32).abs()
                        + (rgb_data[idx + 1] as i32 - rgb_data[idx_next + 1] as i32).abs()
                        + (rgb_data[idx + 2] as i32 - rgb_data[idx_next + 2] as i32).abs();

                    let dist = ((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)).sqrt();
                    let weight = 1.0 - (dist / max_dist).min(1.0);

                    weighted_edge_sum += edge as f64 * weight as f64;
                    weight_sum += weight as f64;
                }
            }
        }

        if weight_sum > 0.0 {
            ((weighted_edge_sum / weight_sum / 255.0) as f32).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
}

impl Default for ThumbnailSelector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_score_uniform_frame() {
        let selector = ThumbnailSelector::new();
        let width = 100;
        let height = 100;
        let rgb_data = vec![128u8; width * height * 3];

        let score = selector.score_frame(&rgb_data, width, height, 0);
        assert!(score.is_ok());
        let s = score.expect("should succeed");
        assert!(s.overall >= 0.0 && s.overall <= 1.0);
        assert_eq!(s.frame_index, 0);
        // Uniform frame should have low sharpness
        assert!(s.sharpness < 0.5);
    }

    #[test]
    fn test_score_varied_frame() {
        let selector = ThumbnailSelector::new();
        let width = 100;
        let height = 100;
        let mut rgb_data = vec![0u8; width * height * 3];
        // Create a gradient
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * 3;
                rgb_data[idx] = (x * 255 / width) as u8;
                rgb_data[idx + 1] = (y * 255 / height) as u8;
                rgb_data[idx + 2] = 128;
            }
        }

        let score = selector.score_frame(&rgb_data, width, height, 5);
        assert!(score.is_ok());
        let s = score.expect("should succeed");
        assert!(s.color_diversity > 0.0);
        assert!(s.contrast > 0.0);
        assert_eq!(s.frame_index, 5);
    }

    #[test]
    fn test_select_best_single() {
        let selector = ThumbnailSelector::new();
        let width = 50;
        let height = 50;
        let frame = vec![128u8; width * height * 3];
        let frames = vec![(&frame[..], width, height, 0)];

        let result = selector.select_best(&frames);
        assert!(result.is_ok());
        assert_eq!(result.expect("should succeed").frame_index, 0);
    }

    #[test]
    fn test_select_best_multiple() {
        let selector = ThumbnailSelector::new();
        let width = 50;
        let height = 50;

        // Dark frame
        let dark = vec![10u8; width * height * 3];
        // Bright frame
        let bright = vec![250u8; width * height * 3];
        // Medium frame (should be preferred)
        let medium = vec![128u8; width * height * 3];

        let frames = vec![
            (&dark[..], width, height, 0),
            (&bright[..], width, height, 1),
            (&medium[..], width, height, 2),
        ];

        let result = selector.select_best(&frames);
        assert!(result.is_ok());
    }

    #[test]
    fn test_select_best_empty() {
        let selector = ThumbnailSelector::new();
        let result = selector.select_best(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_dimensions() {
        let selector = ThumbnailSelector::new();
        let result = selector.score_frame(&[0u8; 10], 100, 100, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_sharpness_range() {
        let selector = ThumbnailSelector::new();
        let width = 50;
        let height = 50;
        let rgb_data = vec![128u8; width * height * 3];
        let sharpness = selector.compute_sharpness(&rgb_data, width, height);
        assert!(sharpness >= 0.0 && sharpness <= 1.0);
    }

    #[test]
    fn test_color_diversity_uniform() {
        let selector = ThumbnailSelector::new();
        let rgb_data = vec![128u8; 100 * 3];
        let diversity = selector.compute_color_diversity(&rgb_data);
        // Single color should have very low diversity
        assert!(diversity < 0.1);
    }

    #[test]
    fn test_color_diversity_varied() {
        let selector = ThumbnailSelector::new();
        let mut rgb_data = Vec::new();
        // Generate many distinct colors across the spectrum
        for r in 0..8 {
            for g in 0..8 {
                for b in 0..4 {
                    rgb_data.push(r * 32);
                    rgb_data.push(g * 32);
                    rgb_data.push(b * 64);
                }
            }
        }
        let diversity = selector.compute_color_diversity(&rgb_data);
        assert!(
            diversity > 0.1,
            "expected diversity > 0.1 but got {diversity}"
        );
    }

    #[test]
    fn test_custom_config() {
        let config = ThumbnailSelectorConfig {
            sharpness_weight: 1.0,
            color_weight: 0.0,
            brightness_weight: 0.0,
            contrast_weight: 0.0,
            center_weight: 0.0,
            target_brightness: 0.5,
        };
        let selector = ThumbnailSelector::with_config(config);
        let width = 50;
        let height = 50;
        let rgb_data = vec![128u8; width * height * 3];
        let score = selector.score_frame(&rgb_data, width, height, 0);
        assert!(score.is_ok());
    }
}
