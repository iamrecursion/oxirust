//! Video complexity analysis module.
//!
//! Provides tools for measuring temporal, spatial, colour and motion complexity
//! of video content, which are useful for predicting encoding difficulty and
//! selecting appropriate encoding parameters.

use serde::{Deserialize, Serialize};

/// Comprehensive complexity metrics for a video segment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityMetrics {
    /// Temporal complexity – standard deviation of inter-frame differences
    pub temporal_complexity: f64,
    /// Spatial complexity – edge density measure
    pub spatial_complexity: f64,
    /// Colour complexity – variance of colour distribution
    pub color_complexity: f64,
    /// Aggregate motion energy derived from motion vectors
    pub motion_energy: f64,
    /// Scene change rate (changes per second or per 100 frames)
    pub scene_change_rate: f64,
}

impl ComplexityMetrics {
    /// Create a zeroed `ComplexityMetrics`.
    #[must_use]
    pub fn zero() -> Self {
        Self {
            temporal_complexity: 0.0,
            spatial_complexity: 0.0,
            color_complexity: 0.0,
            motion_energy: 0.0,
            scene_change_rate: 0.0,
        }
    }

    /// Weighted average of all complexity dimensions.
    ///
    /// Weights: temporal 30 %, spatial 25 %, colour 15 %, motion 20 %, scene 10 %.
    #[must_use]
    pub fn overall_complexity(&self) -> f64 {
        let score = 0.30 * self.temporal_complexity
            + 0.25 * self.spatial_complexity
            + 0.15 * self.color_complexity
            + 0.20 * self.motion_energy
            + 0.10 * self.scene_change_rate;
        score.clamp(0.0, 1.0)
    }

    /// Classify encoding difficulty based on `overall_complexity`.
    #[must_use]
    pub fn encoding_difficulty(&self) -> EncodingDifficulty {
        match self.overall_complexity() {
            s if s < 0.25 => EncodingDifficulty::Simple,
            s if s < 0.50 => EncodingDifficulty::Moderate,
            s if s < 0.75 => EncodingDifficulty::Complex,
            _ => EncodingDifficulty::VeryComplex,
        }
    }
}

/// Classification of how difficult a video segment is to encode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EncodingDifficulty {
    /// Low complexity – simple scenes, talking heads, slides
    Simple,
    /// Medium complexity – mixed content
    Moderate,
    /// High complexity – action, fast motion, detailed textures
    Complex,
    /// Very high complexity – sports, particle effects, film grain
    VeryComplex,
}

/// Compute temporal complexity as the standard deviation of frame differences.
///
/// `frame_diffs` is a slice of per-frame difference values (e.g. mean absolute
/// difference of luma planes between consecutive frames), each in [0, 255].
/// The returned value is normalised to [0, 1] by dividing by 255.
#[must_use]
pub fn temporal_complexity(frame_diffs: &[f64]) -> f64 {
    if frame_diffs.len() < 2 {
        return 0.0;
    }
    let n = frame_diffs.len() as f64;
    let mean = frame_diffs.iter().sum::<f64>() / n;
    let variance = frame_diffs.iter().map(|&d| (d - mean).powi(2)).sum::<f64>() / n;
    let std_dev = variance.sqrt();
    // Normalise: max possible std dev over [0,255] is ~127.5
    (std_dev / 127.5).clamp(0.0, 1.0)
}

/// Compute spatial complexity as a simple edge-density metric.
///
/// Applies a 3×3 horizontal Sobel-like operator over the luma plane and
/// returns the fraction of pixels whose gradient magnitude exceeds a fixed
/// threshold (16).  Output is in [0, 1].
#[allow(clippy::cast_lossless)]
#[must_use]
pub fn spatial_complexity(luma: &[u8], width: usize, height: usize) -> f64 {
    if luma.len() < width * height || width < 3 || height < 3 {
        return 0.0;
    }

    let threshold: i32 = 16;
    let mut edge_count: usize = 0;
    let total_inner = (width - 2) * (height - 2);

    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            // Horizontal Sobel
            let tl = luma[(y - 1) * width + (x - 1)] as i32;
            let tr = luma[(y - 1) * width + (x + 1)] as i32;
            let ml = luma[y * width + (x - 1)] as i32;
            let mr = luma[y * width + (x + 1)] as i32;
            let bl = luma[(y + 1) * width + (x - 1)] as i32;
            let br = luma[(y + 1) * width + (x + 1)] as i32;

            let gx = -tl + tr - 2 * ml + 2 * mr - bl + br;
            let gy = -tl - 2 * luma[(y - 1) * width + x] as i32 - tr
                + bl
                + 2 * luma[(y + 1) * width + x] as i32
                + br;
            let mag = ((gx * gx + gy * gy) as f64).sqrt() as i32;

            if mag > threshold {
                edge_count += 1;
            }
        }
    }

    if total_inner == 0 {
        0.0
    } else {
        (edge_count as f64 / total_inner as f64).clamp(0.0, 1.0)
    }
}

/// Compute a motion energy score from a collection of motion vectors.
///
/// Returns the root-mean-square magnitude of all motion vectors, normalised
/// to [0, 1] assuming a maximum meaningful magnitude of 128 pixels.
#[must_use]
pub fn motion_energy_score(motion_vectors: &[(f32, f32)]) -> f64 {
    if motion_vectors.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = motion_vectors
        .iter()
        .map(|(dx, dy)| f64::from(*dx).powi(2) + f64::from(*dy).powi(2))
        .sum();
    let rms = (sum_sq / motion_vectors.len() as f64).sqrt();
    // Normalise: assume max expected magnitude of 128 pixels
    (rms / 128.0).clamp(0.0, 1.0)
}

/// Sliding-window complexity analyser that tracks a rolling history.
#[derive(Debug, Clone)]
pub struct ComplexityAnalyzer {
    /// Number of frames to retain in the rolling window
    pub window_size: usize,
    /// Rolling history of complexity metrics
    pub history: Vec<ComplexityMetrics>,
}

impl ComplexityAnalyzer {
    /// Create a new `ComplexityAnalyzer` with the given window size.
    #[must_use]
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size: window_size.max(1),
            history: Vec::new(),
        }
    }

    /// Add a new `ComplexityMetrics` sample and return the current windowed average.
    ///
    /// Old samples beyond `window_size` are dropped automatically.
    pub fn update(&mut self, metrics: ComplexityMetrics) -> f64 {
        self.history.push(metrics);
        if self.history.len() > self.window_size {
            self.history.remove(0);
        }
        self.windowed_average()
    }

    /// Peak `overall_complexity` seen across all retained history entries.
    #[must_use]
    pub fn peak_complexity(&self) -> f64 {
        self.history
            .iter()
            .map(ComplexityMetrics::overall_complexity)
            .fold(0.0_f64, f64::max)
    }

    /// Mean `overall_complexity` over the current window.
    #[must_use]
    pub fn windowed_average(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        let sum: f64 = self
            .history
            .iter()
            .map(ComplexityMetrics::overall_complexity)
            .sum();
        sum / self.history.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metrics(tc: f64, sc: f64, cc: f64, me: f64, scr: f64) -> ComplexityMetrics {
        ComplexityMetrics {
            temporal_complexity: tc,
            spatial_complexity: sc,
            color_complexity: cc,
            motion_energy: me,
            scene_change_rate: scr,
        }
    }

    #[test]
    fn test_overall_complexity_zero() {
        let m = ComplexityMetrics::zero();
        assert_eq!(m.overall_complexity(), 0.0);
    }

    #[test]
    fn test_overall_complexity_max() {
        let m = metrics(1.0, 1.0, 1.0, 1.0, 1.0);
        assert!((m.overall_complexity() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_encoding_difficulty_simple() {
        let m = metrics(0.0, 0.1, 0.0, 0.0, 0.0);
        assert_eq!(m.encoding_difficulty(), EncodingDifficulty::Simple);
    }

    #[test]
    fn test_encoding_difficulty_very_complex() {
        let m = metrics(1.0, 1.0, 1.0, 1.0, 1.0);
        assert_eq!(m.encoding_difficulty(), EncodingDifficulty::VeryComplex);
    }

    #[test]
    fn test_temporal_complexity_empty() {
        assert_eq!(temporal_complexity(&[]), 0.0);
    }

    #[test]
    fn test_temporal_complexity_single() {
        assert_eq!(temporal_complexity(&[50.0]), 0.0);
    }

    #[test]
    fn test_temporal_complexity_constant() {
        // All the same value → std dev = 0
        let diffs = vec![10.0, 10.0, 10.0, 10.0];
        assert_eq!(temporal_complexity(&diffs), 0.0);
    }

    #[test]
    fn test_temporal_complexity_varied() {
        let diffs = vec![0.0, 127.5, 0.0, 127.5];
        let tc = temporal_complexity(&diffs);
        assert!(tc > 0.0 && tc <= 1.0);
    }

    #[test]
    fn test_spatial_complexity_too_small() {
        // 2×2 image is too small for the Sobel kernel
        let luma = vec![128u8; 4];
        assert_eq!(spatial_complexity(&luma, 2, 2), 0.0);
    }

    #[test]
    fn test_spatial_complexity_uniform() {
        // Uniform grey image → no edges
        let luma = vec![128u8; 5 * 5];
        assert_eq!(spatial_complexity(&luma, 5, 5), 0.0);
    }

    #[test]
    fn test_spatial_complexity_checkerboard() {
        // Vertical edge pattern: left half black, right half white → strong Sobel gradients
        let mut luma = vec![0u8; 10 * 10];
        for y in 0..10usize {
            for x in 5..10usize {
                luma[y * 10 + x] = 255;
            }
        }
        let sc = spatial_complexity(&luma, 10, 10);
        assert!(sc > 0.0);
    }

    #[test]
    fn test_motion_energy_score_empty() {
        assert_eq!(motion_energy_score(&[]), 0.0);
    }

    #[test]
    fn test_motion_energy_score_zero_vectors() {
        let mvs = vec![(0.0f32, 0.0f32); 10];
        assert_eq!(motion_energy_score(&mvs), 0.0);
    }

    #[test]
    fn test_motion_energy_score_clamped() {
        // Very large motion vectors should clamp to 1.0
        let mvs = vec![(1000.0f32, 1000.0f32); 5];
        assert_eq!(motion_energy_score(&mvs), 1.0);
    }

    #[test]
    fn test_complexity_analyzer_window() {
        let mut analyzer = ComplexityAnalyzer::new(3);
        for _ in 0..5 {
            analyzer.update(metrics(0.5, 0.5, 0.5, 0.5, 0.5));
        }
        // Window should contain only 3 entries
        assert_eq!(analyzer.history.len(), 3);
    }

    #[test]
    fn test_complexity_analyzer_peak() {
        let mut analyzer = ComplexityAnalyzer::new(10);
        analyzer.update(metrics(0.1, 0.1, 0.1, 0.1, 0.1));
        analyzer.update(metrics(1.0, 1.0, 1.0, 1.0, 1.0));
        analyzer.update(metrics(0.2, 0.2, 0.2, 0.2, 0.2));
        assert!(analyzer.peak_complexity() > 0.9);
    }

    #[test]
    fn test_complexity_analyzer_empty_peak() {
        let analyzer = ComplexityAnalyzer::new(5);
        assert_eq!(analyzer.peak_complexity(), 0.0);
    }
}
