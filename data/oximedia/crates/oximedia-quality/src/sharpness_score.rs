#![allow(dead_code)]
//! Edge-based sharpness and clarity scoring for video frames.
//!
//! Provides multiple approaches to measuring image sharpness including
//! Laplacian variance, gradient magnitude, and Tenengrad methods.

/// Result of a sharpness analysis.
#[derive(Debug, Clone)]
pub struct SharpnessResult {
    /// Laplacian variance score (higher = sharper).
    pub laplacian_variance: f64,
    /// Gradient magnitude mean.
    pub gradient_mean: f64,
    /// Tenengrad score (sum of squared gradients above threshold).
    pub tenengrad: f64,
    /// Normalized sharpness score (0.0 = very blurry, 1.0 = very sharp).
    pub normalized_score: f64,
    /// Sharpness rating category.
    pub rating: SharpnessRating,
}

/// Qualitative sharpness rating.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SharpnessRating {
    /// Image is very blurry, likely out of focus.
    VeryBlurry,
    /// Image is somewhat blurry.
    Blurry,
    /// Image has acceptable sharpness.
    Acceptable,
    /// Image is sharp.
    Sharp,
    /// Image is very sharp with strong edges.
    VerySharp,
}

/// Configuration for sharpness analysis.
#[derive(Debug, Clone)]
pub struct SharpnessConfig {
    /// Threshold for Tenengrad gradient magnitude.
    pub tenengrad_threshold: f64,
    /// Normalization ceiling for Laplacian variance (values above are clamped to 1.0).
    pub laplacian_max: f64,
    /// Weight for Laplacian component in combined score.
    pub weight_laplacian: f64,
    /// Weight for gradient component in combined score.
    pub weight_gradient: f64,
    /// Weight for Tenengrad component in combined score.
    pub weight_tenengrad: f64,
}

impl Default for SharpnessConfig {
    fn default() -> Self {
        Self {
            tenengrad_threshold: 10.0,
            laplacian_max: 500.0,
            weight_laplacian: 0.4,
            weight_gradient: 0.3,
            weight_tenengrad: 0.3,
        }
    }
}

impl SharpnessConfig {
    /// Creates a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the Tenengrad threshold.
    #[must_use]
    pub fn with_tenengrad_threshold(mut self, t: f64) -> Self {
        self.tenengrad_threshold = t;
        self
    }

    /// Sets the Laplacian max normalization ceiling.
    #[must_use]
    pub fn with_laplacian_max(mut self, m: f64) -> Self {
        self.laplacian_max = m;
        self
    }
}

/// Sharpness analyzer for 8-bit grayscale images.
#[derive(Debug, Clone)]
pub struct SharpnessAnalyzer {
    /// Configuration.
    config: SharpnessConfig,
}

impl SharpnessAnalyzer {
    /// Creates a new analyzer with the given configuration.
    #[must_use]
    pub fn new(config: SharpnessConfig) -> Self {
        Self { config }
    }

    /// Creates a new analyzer with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self {
            config: SharpnessConfig::default(),
        }
    }

    /// Analyzes the sharpness of a grayscale image.
    ///
    /// `data` is the raw pixel data in row-major order.
    /// `width` and `height` are the image dimensions.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn analyze(&self, data: &[u8], width: usize, height: usize) -> SharpnessResult {
        let lap_var = self.laplacian_variance(data, width, height);
        let grad_mean = self.gradient_magnitude_mean(data, width, height);
        let tenengrad = self.tenengrad(data, width, height);

        let norm_lap = (lap_var / self.config.laplacian_max).min(1.0);
        let norm_grad = (grad_mean / 50.0).min(1.0); // 50 is a typical "sharp" gradient mean
        let norm_ten = (tenengrad / (self.config.laplacian_max * 2.0)).min(1.0);

        let combined = self.config.weight_laplacian * norm_lap
            + self.config.weight_gradient * norm_grad
            + self.config.weight_tenengrad * norm_ten;

        let normalized_score = combined.clamp(0.0, 1.0);

        let rating = if normalized_score < 0.15 {
            SharpnessRating::VeryBlurry
        } else if normalized_score < 0.35 {
            SharpnessRating::Blurry
        } else if normalized_score < 0.55 {
            SharpnessRating::Acceptable
        } else if normalized_score < 0.75 {
            SharpnessRating::Sharp
        } else {
            SharpnessRating::VerySharp
        };

        SharpnessResult {
            laplacian_variance: lap_var,
            gradient_mean: grad_mean,
            tenengrad,
            normalized_score,
            rating,
        }
    }

    /// Computes the Laplacian variance (focus measure).
    /// Uses the 3x3 Laplacian kernel: [0 1 0; 1 -4 1; 0 1 0].
    #[allow(clippy::cast_precision_loss)]
    fn laplacian_variance(&self, data: &[u8], width: usize, height: usize) -> f64 {
        if width < 3 || height < 3 {
            return 0.0;
        }

        let mut sum = 0.0f64;
        let mut sum_sq = 0.0f64;
        let mut count = 0u64;

        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let center = data[y * width + x] as f64;
                let up = data[(y - 1) * width + x] as f64;
                let down = data[(y + 1) * width + x] as f64;
                let left = data[y * width + (x - 1)] as f64;
                let right = data[y * width + (x + 1)] as f64;

                let lap = up + down + left + right - 4.0 * center;
                sum += lap;
                sum_sq += lap * lap;
                count += 1;
            }
        }

        if count == 0 {
            return 0.0;
        }
        let mean = sum / count as f64;
        sum_sq / count as f64 - mean * mean
    }

    /// Computes the mean gradient magnitude using Sobel-like operators.
    #[allow(clippy::cast_precision_loss)]
    fn gradient_magnitude_mean(&self, data: &[u8], width: usize, height: usize) -> f64 {
        if width < 3 || height < 3 {
            return 0.0;
        }

        let mut sum = 0.0f64;
        let mut count = 0u64;

        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let gx = data[y * width + (x + 1)] as f64 - data[y * width + (x - 1)] as f64;
                let gy = data[(y + 1) * width + x] as f64 - data[(y - 1) * width + x] as f64;
                sum += (gx * gx + gy * gy).sqrt();
                count += 1;
            }
        }

        if count == 0 {
            return 0.0;
        }
        sum / count as f64
    }

    /// Computes the Tenengrad focus measure (sum of squared gradients above threshold).
    #[allow(clippy::cast_precision_loss)]
    fn tenengrad(&self, data: &[u8], width: usize, height: usize) -> f64 {
        if width < 3 || height < 3 {
            return 0.0;
        }

        let threshold_sq = self.config.tenengrad_threshold * self.config.tenengrad_threshold;
        let mut sum = 0.0f64;
        let mut count = 0u64;

        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let gx = data[y * width + (x + 1)] as f64 - data[y * width + (x - 1)] as f64;
                let gy = data[(y + 1) * width + x] as f64 - data[(y - 1) * width + x] as f64;
                let mag_sq = gx * gx + gy * gy;
                if mag_sq > threshold_sq {
                    sum += mag_sq;
                }
                count += 1;
            }
        }

        if count == 0 {
            return 0.0;
        }
        sum / count as f64
    }
}

/// Convenience function: analyze sharpness with default configuration.
#[must_use]
pub fn analyze_sharpness(data: &[u8], width: usize, height: usize) -> SharpnessResult {
    SharpnessAnalyzer::with_defaults().analyze(data, width, height)
}

/// Compares sharpness between two frames. Returns positive if frame A is sharper.
#[must_use]
pub fn compare_sharpness(
    data_a: &[u8],
    width_a: usize,
    height_a: usize,
    data_b: &[u8],
    width_b: usize,
    height_b: usize,
) -> f64 {
    let sa = analyze_sharpness(data_a, width_a, height_a);
    let sb = analyze_sharpness(data_b, width_b, height_b);
    sa.normalized_score - sb.normalized_score
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_image(width: usize, height: usize, value: u8) -> Vec<u8> {
        vec![value; width * height]
    }

    fn edge_image(width: usize, height: usize) -> Vec<u8> {
        let mut data = vec![0u8; width * height];
        for y in 0..height {
            for x in 0..width {
                if x >= width / 2 {
                    data[y * width + x] = 255;
                }
            }
        }
        data
    }

    fn gradient_image(width: usize, height: usize) -> Vec<u8> {
        let mut data = vec![0u8; width * height];
        for y in 0..height {
            for x in 0..width {
                #[allow(clippy::cast_precision_loss)]
                let val = (x as f64 / width as f64 * 255.0) as u8;
                data[y * width + x] = val;
            }
        }
        data
    }

    #[test]
    fn test_flat_image_not_sharp() {
        let data = flat_image(64, 64, 128);
        let result = analyze_sharpness(&data, 64, 64);
        assert!(result.laplacian_variance < 1.0);
        assert_eq!(result.rating, SharpnessRating::VeryBlurry);
    }

    #[test]
    fn test_edge_image_sharp() {
        let data = edge_image(64, 64);
        let result = analyze_sharpness(&data, 64, 64);
        assert!(result.laplacian_variance > 0.0);
        assert!(result.gradient_mean > 0.0);
    }

    #[test]
    fn test_gradient_image() {
        let data = gradient_image(64, 64);
        let result = analyze_sharpness(&data, 64, 64);
        assert!(result.gradient_mean > 0.0);
    }

    #[test]
    fn test_tiny_image() {
        let data = flat_image(2, 2, 128);
        let result = analyze_sharpness(&data, 2, 2);
        assert!((result.laplacian_variance).abs() < 1e-10);
    }

    #[test]
    fn test_config_default() {
        let config = SharpnessConfig::default();
        assert!((config.tenengrad_threshold - 10.0).abs() < 1e-10);
        assert!((config.laplacian_max - 500.0).abs() < 1e-10);
    }

    #[test]
    fn test_config_builder() {
        let config = SharpnessConfig::new()
            .with_tenengrad_threshold(20.0)
            .with_laplacian_max(1000.0);
        assert!((config.tenengrad_threshold - 20.0).abs() < 1e-10);
        assert!((config.laplacian_max - 1000.0).abs() < 1e-10);
    }

    #[test]
    fn test_analyzer_with_config() {
        let config = SharpnessConfig::new().with_laplacian_max(100.0);
        let analyzer = SharpnessAnalyzer::new(config);
        let data = edge_image(32, 32);
        let result = analyzer.analyze(&data, 32, 32);
        assert!(result.normalized_score >= 0.0);
        assert!(result.normalized_score <= 1.0);
    }

    #[test]
    fn test_normalized_score_clamped() {
        let data = edge_image(64, 64);
        let result = analyze_sharpness(&data, 64, 64);
        assert!(result.normalized_score >= 0.0);
        assert!(result.normalized_score <= 1.0);
    }

    #[test]
    fn test_compare_sharpness_flat_vs_edge() {
        let flat = flat_image(32, 32, 128);
        let edge = edge_image(32, 32);
        // Edge should be sharper than flat
        let diff = compare_sharpness(&edge, 32, 32, &flat, 32, 32);
        assert!(diff > 0.0);
    }

    #[test]
    fn test_compare_sharpness_same() {
        let data = gradient_image(32, 32);
        let diff = compare_sharpness(&data, 32, 32, &data, 32, 32);
        assert!((diff).abs() < 1e-10);
    }

    #[test]
    fn test_rating_categories() {
        // Very blurry: flat image
        let flat = flat_image(32, 32, 128);
        let r = analyze_sharpness(&flat, 32, 32);
        assert_eq!(r.rating, SharpnessRating::VeryBlurry);
    }

    #[test]
    fn test_tenengrad_positive_for_edges() {
        let data = edge_image(64, 64);
        let result = analyze_sharpness(&data, 64, 64);
        assert!(result.tenengrad > 0.0);
    }
}
