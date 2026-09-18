#![allow(dead_code)]
//! Crowd density estimation for video frames.
//!
//! This module provides tools for estimating how densely populated a scene is
//! by analyzing pixel statistics, edge density, and texture complexity as
//! proxies for head-count. No external ML dependencies are required.

/// Density classification of a crowd scene.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DensityLevel {
    /// No crowd detected (0-2 people equivalent).
    Empty,
    /// Sparse crowd (a handful of individuals).
    Sparse,
    /// Moderate crowd density.
    Moderate,
    /// Dense crowd, individual separation is difficult.
    Dense,
    /// Extremely packed crowd (concert, stadium, protest).
    Packed,
}

impl DensityLevel {
    /// Return a human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Empty => "empty",
            Self::Sparse => "sparse",
            Self::Moderate => "moderate",
            Self::Dense => "dense",
            Self::Packed => "packed",
        }
    }

    /// Convert a numeric score (0.0 to 1.0) into a density level.
    pub fn from_score(score: f64) -> Self {
        match score {
            s if s < 0.1 => Self::Empty,
            s if s < 0.3 => Self::Sparse,
            s if s < 0.55 => Self::Moderate,
            s if s < 0.8 => Self::Dense,
            _ => Self::Packed,
        }
    }
}

/// Result of a crowd density estimation.
#[derive(Debug, Clone, PartialEq)]
pub struct DensityEstimate {
    /// Overall density score between 0.0 (empty) and 1.0 (packed).
    pub score: f64,
    /// Classified density level.
    pub level: DensityLevel,
    /// Edge density contribution (0.0-1.0).
    pub edge_density: f64,
    /// Texture complexity contribution (0.0-1.0).
    pub texture_complexity: f64,
    /// Variance of the luminance channel.
    pub luminance_variance: f64,
}

/// Configuration for crowd density estimation.
#[derive(Debug, Clone)]
pub struct DensityEstimatorConfig {
    /// Weight for edge density in the combined score.
    pub edge_weight: f64,
    /// Weight for texture complexity in the combined score.
    pub texture_weight: f64,
    /// Weight for luminance variance in the combined score.
    pub variance_weight: f64,
    /// Sobel threshold for edge detection (0-255 scale).
    pub edge_threshold: f64,
}

impl Default for DensityEstimatorConfig {
    fn default() -> Self {
        Self {
            edge_weight: 0.4,
            texture_weight: 0.35,
            variance_weight: 0.25,
            edge_threshold: 30.0,
        }
    }
}

/// Crowd density estimator.
#[derive(Debug)]
pub struct DensityEstimator {
    /// Configuration.
    config: DensityEstimatorConfig,
}

impl DensityEstimator {
    /// Create a new estimator with default configuration.
    pub fn new() -> Self {
        Self {
            config: DensityEstimatorConfig::default(),
        }
    }

    /// Create a new estimator with custom configuration.
    pub fn with_config(config: DensityEstimatorConfig) -> Self {
        Self { config }
    }

    /// Estimate crowd density from a grayscale frame.
    ///
    /// `pixels` contains luminance values 0-255, laid out row-major.
    #[allow(clippy::cast_precision_loss)]
    pub fn estimate(&self, pixels: &[u8], width: usize, height: usize) -> DensityEstimate {
        if width == 0 || height == 0 || pixels.len() < width * height {
            return DensityEstimate {
                score: 0.0,
                level: DensityLevel::Empty,
                edge_density: 0.0,
                texture_complexity: 0.0,
                luminance_variance: 0.0,
            };
        }

        let edge_density = self.compute_edge_density(pixels, width, height);
        let texture_complexity = self.compute_texture_complexity(pixels, width, height);
        let luminance_variance = compute_variance(pixels);

        // Normalize variance to 0-1 range (max theoretical variance for 0-255 is ~16256)
        let norm_variance = (luminance_variance / 4000.0).min(1.0);

        let score = (self.config.edge_weight * edge_density
            + self.config.texture_weight * texture_complexity
            + self.config.variance_weight * norm_variance)
            .clamp(0.0, 1.0);

        let level = DensityLevel::from_score(score);

        DensityEstimate {
            score,
            level,
            edge_density,
            texture_complexity,
            luminance_variance,
        }
    }

    /// Compute edge density using a simplified Sobel operator.
    #[allow(clippy::cast_precision_loss)]
    fn compute_edge_density(&self, pixels: &[u8], width: usize, height: usize) -> f64 {
        if width < 3 || height < 3 {
            return 0.0;
        }
        let mut edge_count = 0_u64;
        let total = ((width - 2) * (height - 2)) as u64;
        if total == 0 {
            return 0.0;
        }

        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let gx = -(pixels[(y - 1) * width + (x - 1)] as i32)
                    + (pixels[(y - 1) * width + (x + 1)] as i32)
                    - 2 * (pixels[y * width + (x - 1)] as i32)
                    + 2 * (pixels[y * width + (x + 1)] as i32)
                    - (pixels[(y + 1) * width + (x - 1)] as i32)
                    + (pixels[(y + 1) * width + (x + 1)] as i32);

                let gy = -(pixels[(y - 1) * width + (x - 1)] as i32)
                    - 2 * (pixels[(y - 1) * width + x] as i32)
                    - (pixels[(y - 1) * width + (x + 1)] as i32)
                    + (pixels[(y + 1) * width + (x - 1)] as i32)
                    + 2 * (pixels[(y + 1) * width + x] as i32)
                    + (pixels[(y + 1) * width + (x + 1)] as i32);

                let magnitude = ((gx * gx + gy * gy) as f64).sqrt();
                if magnitude > self.config.edge_threshold {
                    edge_count += 1;
                }
            }
        }

        edge_count as f64 / total as f64
    }

    /// Compute texture complexity using local standard deviation.
    #[allow(clippy::cast_precision_loss)]
    fn compute_texture_complexity(&self, pixels: &[u8], width: usize, height: usize) -> f64 {
        let block_size = 8;
        if width < block_size || height < block_size {
            return 0.0;
        }

        let blocks_x = width / block_size;
        let blocks_y = height / block_size;
        if blocks_x == 0 || blocks_y == 0 {
            return 0.0;
        }

        let mut total_std_dev = 0.0_f64;
        let block_count = blocks_x * blocks_y;

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let mut sum = 0.0_f64;
                let mut sum_sq = 0.0_f64;
                let n = (block_size * block_size) as f64;

                for dy in 0..block_size {
                    for dx in 0..block_size {
                        let val = f64::from(
                            pixels[(by * block_size + dy) * width + bx * block_size + dx],
                        );
                        sum += val;
                        sum_sq += val * val;
                    }
                }

                let mean = sum / n;
                let variance = (sum_sq / n) - mean * mean;
                total_std_dev += variance.max(0.0).sqrt();
            }
        }

        let avg_std = total_std_dev / block_count as f64;
        // Normalize: max std for 0-255 is ~127.5
        (avg_std / 60.0).min(1.0)
    }
}

impl Default for DensityEstimator {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute variance of a byte slice.
#[allow(clippy::cast_precision_loss)]
fn compute_variance(data: &[u8]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let n = data.len() as f64;
    let sum: f64 = data.iter().map(|&v| f64::from(v)).sum();
    let sum_sq: f64 = data.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
    let mean = sum / n;
    (sum_sq / n) - mean * mean
}

/// Track crowd density over time across multiple frames.
#[derive(Debug)]
pub struct DensityTracker {
    /// History of density scores.
    history: Vec<f64>,
    /// Maximum number of entries to keep.
    max_history: usize,
}

impl DensityTracker {
    /// Create a new density tracker with the given history limit.
    pub fn new(max_history: usize) -> Self {
        Self {
            history: Vec::new(),
            max_history: max_history.max(1),
        }
    }

    /// Record a new density measurement.
    pub fn record(&mut self, score: f64) {
        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push(score);
    }

    /// Return the current running average density.
    #[allow(clippy::cast_precision_loss)]
    pub fn average(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.history.iter().sum();
        sum / self.history.len() as f64
    }

    /// Return the trend: positive means density is increasing.
    #[allow(clippy::cast_precision_loss)]
    pub fn trend(&self) -> f64 {
        if self.history.len() < 2 {
            return 0.0;
        }
        let half = self.history.len() / 2;
        let first_half: f64 = self.history[..half].iter().sum::<f64>() / half as f64;
        let second_half: f64 =
            self.history[half..].iter().sum::<f64>() / (self.history.len() - half) as f64;
        second_half - first_half
    }

    /// Return the number of recorded measurements.
    pub fn count(&self) -> usize {
        self.history.len()
    }

    /// Clear all recorded history.
    pub fn clear(&mut self) {
        self.history.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_density_level_from_score() {
        assert_eq!(DensityLevel::from_score(0.0), DensityLevel::Empty);
        assert_eq!(DensityLevel::from_score(0.05), DensityLevel::Empty);
        assert_eq!(DensityLevel::from_score(0.15), DensityLevel::Sparse);
        assert_eq!(DensityLevel::from_score(0.4), DensityLevel::Moderate);
        assert_eq!(DensityLevel::from_score(0.7), DensityLevel::Dense);
        assert_eq!(DensityLevel::from_score(0.9), DensityLevel::Packed);
    }

    #[test]
    fn test_density_level_label() {
        assert_eq!(DensityLevel::Empty.label(), "empty");
        assert_eq!(DensityLevel::Packed.label(), "packed");
    }

    #[test]
    fn test_estimate_empty_frame() {
        let estimator = DensityEstimator::new();
        let result = estimator.estimate(&[], 0, 0);
        assert_eq!(result.level, DensityLevel::Empty);
        assert!((result.score - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_estimate_uniform_frame() {
        let estimator = DensityEstimator::new();
        // Uniform gray frame - no edges, no texture
        let pixels = vec![128_u8; 64 * 64];
        let result = estimator.estimate(&pixels, 64, 64);
        assert!(result.score < 0.1);
        assert_eq!(result.level, DensityLevel::Empty);
    }

    #[test]
    fn test_estimate_noisy_frame() {
        let estimator = DensityEstimator::new();
        // 2px-wide vertical stripes — dense edge boundaries at every other
        // pair of columns, giving > 50 % Sobel edge pixels.
        let mut pixels = vec![0_u8; 64 * 64];
        for y in 0..64 {
            for x in 0..64 {
                pixels[y * 64 + x] = if (x / 2) % 2 == 0 { 255 } else { 0 };
            }
        }
        let result = estimator.estimate(&pixels, 64, 64);
        assert!(result.edge_density > 0.3);
        assert!(result.score > 0.2);
    }

    #[test]
    fn test_compute_variance_uniform() {
        let data = vec![100_u8; 50];
        assert!(compute_variance(&data) < 1e-10);
    }

    #[test]
    fn test_compute_variance_binary() {
        // Half 0, half 255
        let mut data = vec![0_u8; 50];
        data.extend(vec![255_u8; 50]);
        let v = compute_variance(&data);
        // Should be substantial
        assert!(v > 1000.0);
    }

    #[test]
    fn test_tracker_record_and_average() {
        let mut tracker = DensityTracker::new(10);
        tracker.record(0.5);
        tracker.record(0.7);
        assert!((tracker.average() - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_tracker_max_history() {
        let mut tracker = DensityTracker::new(3);
        tracker.record(0.1);
        tracker.record(0.2);
        tracker.record(0.3);
        tracker.record(0.9);
        assert_eq!(tracker.count(), 3);
        // Oldest (0.1) should be evicted
        assert!((tracker.average() - (0.2 + 0.3 + 0.9) / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_tracker_trend_increasing() {
        let mut tracker = DensityTracker::new(100);
        for i in 0..10 {
            tracker.record(i as f64 * 0.1);
        }
        assert!(tracker.trend() > 0.0);
    }

    #[test]
    fn test_tracker_trend_decreasing() {
        let mut tracker = DensityTracker::new(100);
        for i in (0..10).rev() {
            tracker.record(i as f64 * 0.1);
        }
        assert!(tracker.trend() < 0.0);
    }

    #[test]
    fn test_tracker_clear() {
        let mut tracker = DensityTracker::new(10);
        tracker.record(0.5);
        tracker.clear();
        assert_eq!(tracker.count(), 0);
        assert!((tracker.average() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_estimator_config_custom() {
        let config = DensityEstimatorConfig {
            edge_weight: 1.0,
            texture_weight: 0.0,
            variance_weight: 0.0,
            edge_threshold: 10.0,
        };
        let estimator = DensityEstimator::with_config(config);
        // Vertical stripes (4px wide) produce strong horizontal Sobel edges
        let mut pixels = vec![0_u8; 32 * 32];
        for y in 0..32 {
            for x in 0..32 {
                pixels[y * 32 + x] = if (x / 4) % 2 == 0 { 200 } else { 50 };
            }
        }
        let result = estimator.estimate(&pixels, 32, 32);
        assert!(result.score > 0.3);
    }
}
