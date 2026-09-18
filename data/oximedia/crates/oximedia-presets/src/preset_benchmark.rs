#![allow(dead_code)]
//! Preset performance benchmarking and comparison scoring.
//!
//! Provides tools for evaluating presets against quality metrics,
//! encoding speed estimates, file size predictions, and overall
//! scoring for informed preset selection.

use std::collections::HashMap;

/// Quality metric identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum QualityMetric {
    /// Peak Signal-to-Noise Ratio.
    Psnr,
    /// Structural Similarity Index.
    Ssim,
    /// Video Multimethod Assessment Fusion.
    Vmaf,
    /// Bits per pixel.
    BitsPerPixel,
    /// Custom named metric.
    Custom(String),
}

impl QualityMetric {
    /// Return a human-readable label for this metric.
    #[must_use]
    pub fn label(&self) -> &str {
        match self {
            QualityMetric::Psnr => "PSNR",
            QualityMetric::Ssim => "SSIM",
            QualityMetric::Vmaf => "VMAF",
            QualityMetric::BitsPerPixel => "Bits/Pixel",
            QualityMetric::Custom(name) => name.as_str(),
        }
    }
}

/// A single benchmark measurement for a preset.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Preset identifier that was benchmarked.
    pub preset_id: String,
    /// Quality scores keyed by metric.
    pub quality_scores: HashMap<QualityMetric, f64>,
    /// Estimated encoding speed as a multiplier of realtime (e.g. 2.0 = 2x realtime).
    pub encode_speed_factor: f64,
    /// Estimated output file size in bytes per second of content.
    pub bytes_per_second: u64,
    /// Wall clock time for the benchmark in milliseconds.
    pub wall_time_ms: u64,
    /// Additional notes.
    pub notes: String,
}

impl BenchmarkResult {
    /// Create a new benchmark result for a preset.
    #[must_use]
    pub fn new(preset_id: &str) -> Self {
        Self {
            preset_id: preset_id.to_string(),
            quality_scores: HashMap::new(),
            encode_speed_factor: 0.0,
            bytes_per_second: 0,
            wall_time_ms: 0,
            notes: String::new(),
        }
    }

    /// Record a quality score.
    #[must_use]
    pub fn with_quality(mut self, metric: QualityMetric, value: f64) -> Self {
        self.quality_scores.insert(metric, value);
        self
    }

    /// Set encoding speed factor.
    #[must_use]
    pub fn with_speed(mut self, factor: f64) -> Self {
        self.encode_speed_factor = factor;
        self
    }

    /// Set output bytes per second.
    #[must_use]
    pub fn with_size(mut self, bytes_per_sec: u64) -> Self {
        self.bytes_per_second = bytes_per_sec;
        self
    }

    /// Set wall time.
    #[must_use]
    pub fn with_wall_time(mut self, ms: u64) -> Self {
        self.wall_time_ms = ms;
        self
    }

    /// Get a quality score for a specific metric.
    #[must_use]
    pub fn get_quality(&self, metric: &QualityMetric) -> Option<f64> {
        self.quality_scores.get(metric).copied()
    }

    /// Compute an overall quality score (average of all quality metrics,
    /// normalized to 0-100 scale assuming VMAF-like scaling).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn overall_quality(&self) -> f64 {
        if self.quality_scores.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.quality_scores.values().sum();
        sum / self.quality_scores.len() as f64
    }
}

/// Weights for computing a weighted composite score.
#[derive(Debug, Clone)]
pub struct ScoringWeights {
    /// Weight for quality (0.0-1.0).
    pub quality_weight: f64,
    /// Weight for encoding speed (0.0-1.0).
    pub speed_weight: f64,
    /// Weight for file size efficiency (0.0-1.0).
    pub size_weight: f64,
}

impl ScoringWeights {
    /// Create new scoring weights. Weights are normalized internally.
    #[must_use]
    pub fn new(quality: f64, speed: f64, size: f64) -> Self {
        Self {
            quality_weight: quality,
            speed_weight: speed,
            size_weight: size,
        }
    }

    /// Balanced weights (equal priority).
    #[must_use]
    pub fn balanced() -> Self {
        Self::new(1.0, 1.0, 1.0)
    }

    /// Quality-focused weights.
    #[must_use]
    pub fn quality_focused() -> Self {
        Self::new(3.0, 1.0, 1.0)
    }

    /// Speed-focused weights.
    #[must_use]
    pub fn speed_focused() -> Self {
        Self::new(1.0, 3.0, 1.0)
    }

    /// Size-focused weights.
    #[must_use]
    pub fn size_focused() -> Self {
        Self::new(1.0, 1.0, 3.0)
    }

    /// Get the total weight sum for normalization.
    #[must_use]
    fn total(&self) -> f64 {
        self.quality_weight + self.speed_weight + self.size_weight
    }

    /// Normalized quality weight.
    #[must_use]
    pub fn norm_quality(&self) -> f64 {
        let total = self.total();
        if total == 0.0 {
            return 0.0;
        }
        self.quality_weight / total
    }

    /// Normalized speed weight.
    #[must_use]
    pub fn norm_speed(&self) -> f64 {
        let total = self.total();
        if total == 0.0 {
            return 0.0;
        }
        self.speed_weight / total
    }

    /// Normalized size weight.
    #[must_use]
    pub fn norm_size(&self) -> f64 {
        let total = self.total();
        if total == 0.0 {
            return 0.0;
        }
        self.size_weight / total
    }
}

impl Default for ScoringWeights {
    fn default() -> Self {
        Self::balanced()
    }
}

/// Computed composite score for a benchmark result.
#[derive(Debug, Clone)]
pub struct CompositeScore {
    /// The preset identifier.
    pub preset_id: String,
    /// Quality component (0-100).
    pub quality_score: f64,
    /// Speed component (0-100).
    pub speed_score: f64,
    /// Size efficiency component (0-100).
    pub size_score: f64,
    /// Weighted composite (0-100).
    pub composite: f64,
}

/// Score calculator that converts benchmark results into comparable scores.
pub struct ScoreCalculator {
    /// Maximum expected encode speed factor for normalization.
    max_speed: f64,
    /// Maximum expected bytes per second for normalization.
    max_bytes_per_sec: u64,
}

impl ScoreCalculator {
    /// Create a new score calculator with normalization parameters.
    #[must_use]
    pub fn new(max_speed: f64, max_bytes_per_sec: u64) -> Self {
        Self {
            max_speed,
            max_bytes_per_sec,
        }
    }

    /// Create a calculator with sensible defaults.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(10.0, 10_000_000)
    }

    /// Compute a composite score for a benchmark result.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn compute(&self, result: &BenchmarkResult, weights: &ScoringWeights) -> CompositeScore {
        let quality_score = result.overall_quality().clamp(0.0, 100.0);

        let speed_score = if self.max_speed > 0.0 {
            (result.encode_speed_factor / self.max_speed * 100.0).clamp(0.0, 100.0)
        } else {
            0.0
        };

        let size_score = if self.max_bytes_per_sec > 0 {
            let ratio = 1.0 - (result.bytes_per_second as f64 / self.max_bytes_per_sec as f64);
            (ratio * 100.0).clamp(0.0, 100.0)
        } else {
            0.0
        };

        let composite = quality_score * weights.norm_quality()
            + speed_score * weights.norm_speed()
            + size_score * weights.norm_size();

        CompositeScore {
            preset_id: result.preset_id.clone(),
            quality_score,
            speed_score,
            size_score,
            composite,
        }
    }

    /// Rank multiple benchmark results by composite score (descending).
    #[must_use]
    pub fn rank(
        &self,
        results: &[BenchmarkResult],
        weights: &ScoringWeights,
    ) -> Vec<CompositeScore> {
        let mut scores: Vec<CompositeScore> =
            results.iter().map(|r| self.compute(r, weights)).collect();
        scores.sort_by(|a, b| {
            b.composite
                .partial_cmp(&a.composite)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scores
    }
}

impl Default for ScoreCalculator {
    fn default() -> Self {
        Self::with_defaults()
    }
}

/// Comparison summary between two benchmark results.
#[derive(Debug, Clone)]
pub struct BenchmarkComparison {
    /// First preset identifier.
    pub preset_a: String,
    /// Second preset identifier.
    pub preset_b: String,
    /// Quality difference (A - B).
    pub quality_diff: f64,
    /// Speed difference (A - B) as factor.
    pub speed_diff: f64,
    /// Size difference (A - B) in bytes per second.
    pub size_diff: i64,
    /// Which preset is recommended (true = A, false = B).
    pub recommend_a: bool,
}

/// Compare two benchmark results.
#[must_use]
#[allow(clippy::cast_possible_wrap)]
pub fn compare_benchmarks(a: &BenchmarkResult, b: &BenchmarkResult) -> BenchmarkComparison {
    let quality_diff = a.overall_quality() - b.overall_quality();
    let speed_diff = a.encode_speed_factor - b.encode_speed_factor;
    let size_diff = a.bytes_per_second as i64 - b.bytes_per_second as i64;

    // Simple heuristic: prefer higher quality, then faster speed
    let recommend_a = if (quality_diff).abs() > 1.0 {
        quality_diff > 0.0
    } else {
        speed_diff >= 0.0
    };

    BenchmarkComparison {
        preset_a: a.preset_id.clone(),
        preset_b: b.preset_id.clone(),
        quality_diff,
        speed_diff,
        size_diff,
        recommend_a,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_result(id: &str, vmaf: f64, speed: f64, size: u64) -> BenchmarkResult {
        BenchmarkResult::new(id)
            .with_quality(QualityMetric::Vmaf, vmaf)
            .with_speed(speed)
            .with_size(size)
            .with_wall_time(1000)
    }

    #[test]
    fn test_quality_metric_label() {
        assert_eq!(QualityMetric::Psnr.label(), "PSNR");
        assert_eq!(QualityMetric::Vmaf.label(), "VMAF");
        assert_eq!(QualityMetric::Custom("MyMetric".into()).label(), "MyMetric");
    }

    #[test]
    fn test_benchmark_result_builder() {
        let result = sample_result("test", 95.0, 2.5, 500_000);
        assert_eq!(result.preset_id, "test");
        assert_eq!(result.get_quality(&QualityMetric::Vmaf), Some(95.0));
        assert_eq!(result.encode_speed_factor, 2.5);
        assert_eq!(result.bytes_per_second, 500_000);
    }

    #[test]
    fn test_overall_quality() {
        let result = BenchmarkResult::new("test")
            .with_quality(QualityMetric::Vmaf, 90.0)
            .with_quality(QualityMetric::Ssim, 80.0);
        let avg = result.overall_quality();
        assert!((avg - 85.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_overall_quality_empty() {
        let result = BenchmarkResult::new("empty");
        assert!((result.overall_quality() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_scoring_weights_balanced() {
        let w = ScoringWeights::balanced();
        let nq = w.norm_quality();
        let ns = w.norm_speed();
        let nz = w.norm_size();
        assert!((nq - 1.0 / 3.0).abs() < 1e-9);
        assert!((ns - 1.0 / 3.0).abs() < 1e-9);
        assert!((nz - 1.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_scoring_weights_quality_focused() {
        let w = ScoringWeights::quality_focused();
        assert!(w.norm_quality() > w.norm_speed());
        assert!(w.norm_quality() > w.norm_size());
    }

    #[test]
    fn test_composite_score() {
        let calc = ScoreCalculator::with_defaults();
        let result = sample_result("p1", 90.0, 5.0, 2_000_000);
        let score = calc.compute(&result, &ScoringWeights::balanced());
        assert!(score.composite > 0.0);
        assert!(score.composite <= 100.0);
        assert_eq!(score.preset_id, "p1");
    }

    #[test]
    fn test_rank_multiple() {
        let calc = ScoreCalculator::with_defaults();
        let results = vec![
            sample_result("low", 50.0, 8.0, 1_000_000),
            sample_result("high", 95.0, 2.0, 5_000_000),
            sample_result("mid", 75.0, 5.0, 3_000_000),
        ];
        let ranked = calc.rank(&results, &ScoringWeights::quality_focused());
        assert_eq!(ranked.len(), 3);
        // With quality focus, the "high" quality preset should rank first
        assert_eq!(ranked[0].preset_id, "high");
    }

    #[test]
    fn test_compare_benchmarks_quality_wins() {
        let a = sample_result("a", 95.0, 2.0, 3_000_000);
        let b = sample_result("b", 70.0, 4.0, 2_000_000);
        let cmp = compare_benchmarks(&a, &b);
        assert!(cmp.recommend_a);
        assert!(cmp.quality_diff > 0.0);
    }

    #[test]
    fn test_compare_benchmarks_speed_tiebreak() {
        let a = sample_result("a", 90.0, 5.0, 3_000_000);
        let b = sample_result("b", 90.0, 2.0, 3_000_000);
        let cmp = compare_benchmarks(&a, &b);
        // Quality is very close, so speed should break tie
        assert!(cmp.recommend_a);
    }

    #[test]
    fn test_score_calculator_zero_max() {
        let calc = ScoreCalculator::new(0.0, 0);
        let result = sample_result("test", 80.0, 3.0, 100);
        let score = calc.compute(&result, &ScoringWeights::balanced());
        // Speed and size should be 0 when max is 0
        assert!((score.speed_score - 0.0).abs() < f64::EPSILON);
        assert!((score.size_score - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_scoring_weights_zero_total() {
        let w = ScoringWeights::new(0.0, 0.0, 0.0);
        assert!((w.norm_quality() - 0.0).abs() < f64::EPSILON);
        assert!((w.norm_speed() - 0.0).abs() < f64::EPSILON);
        assert!((w.norm_size() - 0.0).abs() < f64::EPSILON);
    }
}
