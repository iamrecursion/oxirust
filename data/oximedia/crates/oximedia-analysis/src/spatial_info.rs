#![allow(dead_code)]

//! Spatial information (SI) and perceptual complexity metrics.
//!
//! This module implements ITU-T P.910 spatial information (SI) computation
//! and related spatial complexity metrics. SI measures the amount of spatial
//! detail in a video frame using the Sobel edge filter, and is commonly used
//! in objective video quality evaluation and test-sequence selection.

/// Default Sobel kernel for horizontal edges.
const SOBEL_H: [[i32; 3]; 3] = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]];

/// Default Sobel kernel for vertical edges.
const SOBEL_V: [[i32; 3]; 3] = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]];

/// Result of spatial information computation for a single frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpatialInfoResult {
    /// Spatial information (SI) — std-dev of Sobel-filtered frame.
    pub si: f64,
    /// Mean magnitude of the Sobel-filtered frame.
    pub mean_gradient: f64,
    /// Maximum gradient magnitude in the frame.
    pub max_gradient: f64,
    /// Frame index.
    pub frame_index: u64,
}

/// Aggregate spatial information across multiple frames.
#[derive(Debug, Clone, PartialEq)]
pub struct SpatialInfoSummary {
    /// Number of frames analyzed.
    pub frame_count: usize,
    /// Mean SI across all frames.
    pub mean_si: f64,
    /// Maximum SI across all frames.
    pub max_si: f64,
    /// Minimum SI across all frames.
    pub min_si: f64,
    /// Standard deviation of SI across frames.
    pub std_dev_si: f64,
    /// Classification of spatial complexity.
    pub complexity: SpatialComplexity,
}

/// Classification of spatial complexity level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpatialComplexity {
    /// Very simple content (solid colors, simple graphics).
    VeryLow,
    /// Low complexity (talking heads, simple backgrounds).
    Low,
    /// Medium complexity (typical broadcast content).
    Medium,
    /// High complexity (detailed textures, busy scenes).
    High,
    /// Very high complexity (dense natural scenes, high detail).
    VeryHigh,
}

impl SpatialComplexity {
    /// Classify from SI value (ITU-T P.910 scale, typical 8-bit luma).
    pub fn from_si(si: f64) -> Self {
        if si < 20.0 {
            Self::VeryLow
        } else if si < 50.0 {
            Self::Low
        } else if si < 90.0 {
            Self::Medium
        } else if si < 130.0 {
            Self::High
        } else {
            Self::VeryHigh
        }
    }

    /// Returns a human-readable description.
    pub fn description(self) -> &'static str {
        match self {
            Self::VeryLow => "very low (simple graphics)",
            Self::Low => "low (talking heads)",
            Self::Medium => "medium (typical broadcast)",
            Self::High => "high (detailed textures)",
            Self::VeryHigh => "very high (dense natural scenes)",
        }
    }
}

/// Analyzer for spatial information metrics.
#[derive(Debug)]
pub struct SpatialInfoAnalyzer {
    /// Per-frame SI results.
    results: Vec<SpatialInfoResult>,
}

impl SpatialInfoAnalyzer {
    /// Create a new spatial information analyzer.
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
        }
    }

    /// Analyze a single luma (Y) frame and store the result.
    ///
    /// `luma` is the Y-plane pixel data, `width` and `height` are dimensions.
    #[allow(clippy::cast_precision_loss)]
    pub fn analyze_frame(
        &mut self,
        luma: &[u8],
        width: usize,
        height: usize,
        frame_index: u64,
    ) -> SpatialInfoResult {
        let gradients = compute_sobel_magnitude(luma, width, height);
        let n = gradients.len();

        if n == 0 {
            let result = SpatialInfoResult {
                si: 0.0,
                mean_gradient: 0.0,
                max_gradient: 0.0,
                frame_index,
            };
            self.results.push(result);
            return result;
        }

        let sum: f64 = gradients.iter().sum();
        let mean = sum / n as f64;
        let max = gradients.iter().copied().fold(0.0_f64, f64::max);

        let variance = if n > 1 {
            gradients.iter().map(|g| (g - mean).powi(2)).sum::<f64>() / (n - 1) as f64
        } else {
            0.0
        };
        let si = variance.sqrt();

        let result = SpatialInfoResult {
            si,
            mean_gradient: mean,
            max_gradient: max,
            frame_index,
        };
        self.results.push(result);
        result
    }

    /// Get the summary of all analyzed frames.
    #[allow(clippy::cast_precision_loss)]
    pub fn summarize(&self) -> SpatialInfoSummary {
        if self.results.is_empty() {
            return SpatialInfoSummary {
                frame_count: 0,
                mean_si: 0.0,
                max_si: 0.0,
                min_si: 0.0,
                std_dev_si: 0.0,
                complexity: SpatialComplexity::VeryLow,
            };
        }

        let n = self.results.len();
        let si_values: Vec<f64> = self.results.iter().map(|r| r.si).collect();

        let sum: f64 = si_values.iter().sum();
        let mean = sum / n as f64;
        let max = si_values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let min = si_values.iter().copied().fold(f64::INFINITY, f64::min);

        let variance = if n > 1 {
            si_values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64
        } else {
            0.0
        };
        let std_dev = variance.sqrt();

        SpatialInfoSummary {
            frame_count: n,
            mean_si: mean,
            max_si: max,
            min_si: min,
            std_dev_si: std_dev,
            complexity: SpatialComplexity::from_si(mean),
        }
    }

    /// Return the number of frames analyzed so far.
    pub fn frame_count(&self) -> usize {
        self.results.len()
    }

    /// Return all per-frame results.
    pub fn results(&self) -> &[SpatialInfoResult] {
        &self.results
    }

    /// Clear all results.
    pub fn clear(&mut self) {
        self.results.clear();
    }
}

impl Default for SpatialInfoAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute Sobel gradient magnitude for each pixel (excluding 1-pixel border).
#[allow(clippy::cast_precision_loss)]
fn compute_sobel_magnitude(luma: &[u8], width: usize, height: usize) -> Vec<f64> {
    if width < 3 || height < 3 || luma.len() < width * height {
        return Vec::new();
    }

    let mut magnitudes = Vec::with_capacity((width - 2) * (height - 2));

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let mut gx: i32 = 0;
            let mut gy: i32 = 0;

            for ky in 0..3 {
                for kx in 0..3 {
                    let py = y + ky - 1;
                    let px = x + kx - 1;
                    let pixel = luma[py * width + px] as i32;
                    gx += pixel * SOBEL_H[ky][kx];
                    gy += pixel * SOBEL_V[ky][kx];
                }
            }

            let mag = ((gx * gx + gy * gy) as f64).sqrt();
            magnitudes.push(mag);
        }
    }

    magnitudes
}

/// Compute spatial information for a single frame without accumulation.
#[allow(clippy::cast_precision_loss)]
pub fn compute_si(luma: &[u8], width: usize, height: usize) -> f64 {
    let gradients = compute_sobel_magnitude(luma, width, height);
    if gradients.is_empty() {
        return 0.0;
    }
    let n = gradients.len();
    let mean: f64 = gradients.iter().sum::<f64>() / n as f64;
    let variance = if n > 1 {
        gradients.iter().map(|g| (g - mean).powi(2)).sum::<f64>() / (n - 1) as f64
    } else {
        0.0
    };
    variance.sqrt()
}

/// Compute the mean gradient magnitude for a frame.
#[allow(clippy::cast_precision_loss)]
pub fn compute_mean_gradient(luma: &[u8], width: usize, height: usize) -> f64 {
    let gradients = compute_sobel_magnitude(luma, width, height);
    if gradients.is_empty() {
        return 0.0;
    }
    gradients.iter().sum::<f64>() / gradients.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_frame(width: usize, height: usize, value: u8) -> Vec<u8> {
        vec![value; width * height]
    }

    /// Gradient frame with variable step sizes across rows to produce non-constant
    /// Sobel magnitudes (required for SI > 0).
    fn gradient_frame(width: usize, height: usize) -> Vec<u8> {
        let mut data = vec![0u8; width * height];
        for y in 0..height {
            for x in 0..width {
                // Vary both x and y contributions so rows have different slopes.
                let val = ((x * 4 + y * 2) % 256) as u8;
                data[y * width + x] = val;
            }
        }
        data
    }

    #[test]
    fn test_flat_frame_si_near_zero() {
        let frame = flat_frame(64, 64, 128);
        let si = compute_si(&frame, 64, 64);
        assert!(si < 0.01, "Flat frame should have near-zero SI, got {si}");
    }

    #[test]
    fn test_gradient_frame_has_positive_si() {
        let frame = gradient_frame(64, 64);
        let si = compute_si(&frame, 64, 64);
        assert!(si > 0.0, "Gradient frame should have positive SI");
    }

    #[test]
    fn test_mean_gradient_flat() {
        let frame = flat_frame(32, 32, 100);
        let mg = compute_mean_gradient(&frame, 32, 32);
        assert!(mg < 0.01, "Flat frame should have near-zero mean gradient");
    }

    #[test]
    fn test_mean_gradient_gradient_frame() {
        let frame = gradient_frame(32, 32);
        let mg = compute_mean_gradient(&frame, 32, 32);
        assert!(
            mg > 0.0,
            "Gradient frame should have positive mean gradient"
        );
    }

    #[test]
    fn test_analyzer_empty() {
        let analyzer = SpatialInfoAnalyzer::new();
        let summary = analyzer.summarize();
        assert_eq!(summary.frame_count, 0);
        assert!((summary.mean_si).abs() < f64::EPSILON);
    }

    #[test]
    fn test_analyzer_single_frame() {
        let mut analyzer = SpatialInfoAnalyzer::new();
        let frame = gradient_frame(32, 32);
        let result = analyzer.analyze_frame(&frame, 32, 32, 0);
        assert!(result.si > 0.0);
        assert_eq!(analyzer.frame_count(), 1);
    }

    #[test]
    fn test_analyzer_multiple_frames() {
        let mut analyzer = SpatialInfoAnalyzer::new();
        for i in 0..5 {
            let frame = flat_frame(16, 16, (i * 30) as u8);
            analyzer.analyze_frame(&frame, 16, 16, i);
        }
        let summary = analyzer.summarize();
        assert_eq!(summary.frame_count, 5);
    }

    #[test]
    fn test_complexity_classification() {
        assert_eq!(SpatialComplexity::from_si(10.0), SpatialComplexity::VeryLow);
        assert_eq!(SpatialComplexity::from_si(35.0), SpatialComplexity::Low);
        assert_eq!(SpatialComplexity::from_si(70.0), SpatialComplexity::Medium);
        assert_eq!(SpatialComplexity::from_si(110.0), SpatialComplexity::High);
        assert_eq!(
            SpatialComplexity::from_si(150.0),
            SpatialComplexity::VeryHigh
        );
    }

    #[test]
    fn test_complexity_description() {
        assert!(!SpatialComplexity::VeryLow.description().is_empty());
        assert!(!SpatialComplexity::VeryHigh.description().is_empty());
    }

    #[test]
    fn test_too_small_frame() {
        let frame = vec![128u8; 4]; // 2x2 — too small for Sobel
        let si = compute_si(&frame, 2, 2);
        assert!((si).abs() < f64::EPSILON);
    }

    #[test]
    fn test_analyzer_clear() {
        let mut analyzer = SpatialInfoAnalyzer::new();
        let frame = flat_frame(16, 16, 128);
        analyzer.analyze_frame(&frame, 16, 16, 0);
        assert_eq!(analyzer.frame_count(), 1);
        analyzer.clear();
        assert_eq!(analyzer.frame_count(), 0);
    }

    #[test]
    fn test_summary_max_si_correct() {
        let mut analyzer = SpatialInfoAnalyzer::new();
        let flat = flat_frame(32, 32, 128);
        let grad = gradient_frame(32, 32);
        analyzer.analyze_frame(&flat, 32, 32, 0);
        analyzer.analyze_frame(&grad, 32, 32, 1);
        let summary = analyzer.summarize();
        assert_eq!(summary.frame_count, 2);
        // The gradient frame should have higher SI than the flat one
        assert!(summary.max_si > summary.min_si);
    }
}
