#![allow(dead_code)]
//! Shot rhythm and editing pace analysis.
//!
//! Analyzes the rhythm of editing by examining shot durations, transitions,
//! and temporal patterns. Computes metrics like average shot length (ASL),
//! cutting rate, rhythm regularity, acceleration/deceleration, and
//! identifies rhythmic patterns such as montage sequences.

/// Rhythm metric describing the pace of a sequence.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RhythmMetrics {
    /// Average shot length in frames.
    pub avg_shot_length: f64,
    /// Median shot length in frames.
    pub median_shot_length: f64,
    /// Standard deviation of shot lengths.
    pub std_deviation: f64,
    /// Coefficient of variation (std_dev / mean).
    pub coefficient_of_variation: f64,
    /// Minimum shot length.
    pub min_length: u64,
    /// Maximum shot length.
    pub max_length: u64,
    /// Cutting rate (cuts per unit, e.g. per 1000 frames).
    pub cutting_rate: f64,
    /// Regularity score (0.0 = very irregular, 1.0 = perfectly regular).
    pub regularity: f64,
}

/// Describes the trend (acceleration or deceleration) over a segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaceTrend {
    /// Shots are getting shorter (faster cutting).
    Accelerating,
    /// Shots are getting longer (slower cutting).
    Decelerating,
    /// Shot lengths are roughly constant.
    Steady,
}

/// A segment of the timeline with a detected pace trend.
#[derive(Debug, Clone)]
pub struct PaceSegment {
    /// Start index (shot index) of this segment.
    pub start_idx: usize,
    /// End index (exclusive) of this segment.
    pub end_idx: usize,
    /// Detected trend.
    pub trend: PaceTrend,
    /// Average shot duration in this segment (frames).
    pub avg_duration: f64,
    /// Slope of duration change (negative = accelerating).
    pub slope: f64,
}

/// A detected rhythmic pattern (e.g. regular alternation, montage).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RhythmPattern {
    /// Roughly equal-length shots.
    RegularCadence,
    /// Alternating long and short shots.
    Alternating,
    /// Progressively faster cuts.
    BuildUp,
    /// Progressively slower cuts (cool-down).
    CoolDown,
    /// Very rapid cutting (montage style).
    Montage,
    /// Long takes with few cuts.
    Contemplative,
    /// No clear pattern.
    Freeform,
}

/// Result of full rhythm analysis.
#[derive(Debug, Clone)]
pub struct RhythmAnalysisResult {
    /// Global rhythm metrics.
    pub metrics: RhythmMetrics,
    /// Pace segments detected.
    pub segments: Vec<PaceSegment>,
    /// Dominant pattern.
    pub dominant_pattern: RhythmPattern,
    /// Total duration in frames.
    pub total_duration: u64,
    /// Total shot count.
    pub shot_count: usize,
}

/// Shot rhythm analyzer.
#[derive(Debug, Clone)]
pub struct ShotRhythmAnalyzer {
    /// Window size for computing local pace trends.
    trend_window: usize,
    /// Threshold slope for classifying acceleration vs. steady.
    slope_threshold: f64,
    /// Montage threshold: shots shorter than this (frames) are montage-like.
    montage_threshold: u64,
    /// Contemplative threshold: average shot length above this is contemplative.
    contemplative_threshold: u64,
}

impl ShotRhythmAnalyzer {
    /// Create a new rhythm analyzer with default parameters.
    pub fn new() -> Self {
        Self {
            trend_window: 5,
            slope_threshold: 2.0,
            montage_threshold: 15,
            contemplative_threshold: 300,
        }
    }

    /// Set the trend window size.
    pub fn with_trend_window(mut self, window: usize) -> Self {
        self.trend_window = window;
        self
    }

    /// Set the slope threshold.
    pub fn with_slope_threshold(mut self, threshold: f64) -> Self {
        self.slope_threshold = threshold;
        self
    }

    /// Set the montage threshold.
    pub fn with_montage_threshold(mut self, threshold: u64) -> Self {
        self.montage_threshold = threshold;
        self
    }

    /// Analyze the rhythm of a sequence given shot durations (in frames).
    #[allow(clippy::cast_precision_loss)]
    pub fn analyze(&self, durations: &[u64]) -> RhythmAnalysisResult {
        let shot_count = durations.len();
        let total_duration: u64 = durations.iter().sum();

        if durations.is_empty() {
            return RhythmAnalysisResult {
                metrics: RhythmMetrics {
                    avg_shot_length: 0.0,
                    median_shot_length: 0.0,
                    std_deviation: 0.0,
                    coefficient_of_variation: 0.0,
                    min_length: 0,
                    max_length: 0,
                    cutting_rate: 0.0,
                    regularity: 0.0,
                },
                segments: Vec::new(),
                dominant_pattern: RhythmPattern::Freeform,
                total_duration,
                shot_count,
            };
        }

        let metrics = self.compute_metrics(durations, total_duration);
        let segments = self.detect_segments(durations);
        let dominant_pattern = self.classify_pattern(&metrics, durations);

        RhythmAnalysisResult {
            metrics,
            segments,
            dominant_pattern,
            total_duration,
            shot_count,
        }
    }

    /// Compute global rhythm metrics from durations.
    #[allow(clippy::cast_precision_loss)]
    fn compute_metrics(&self, durations: &[u64], total_duration: u64) -> RhythmMetrics {
        let n = durations.len() as f64;
        let sum: u64 = durations.iter().sum();
        let avg = sum as f64 / n;

        let mut sorted = durations.to_vec();
        sorted.sort_unstable();
        let median = if sorted.len() % 2 == 0 {
            let mid = sorted.len() / 2;
            (sorted[mid - 1] as f64 + sorted[mid] as f64) / 2.0
        } else {
            sorted[sorted.len() / 2] as f64
        };

        let variance: f64 = durations
            .iter()
            .map(|&d| (d as f64 - avg).powi(2))
            .sum::<f64>()
            / n;
        let std_dev = variance.sqrt();
        let cv = if avg > f64::EPSILON {
            std_dev / avg
        } else {
            0.0
        };

        let min_length = sorted.first().copied().unwrap_or(0);
        let max_length = sorted.last().copied().unwrap_or(0);

        let cutting_rate = if total_duration > 0 {
            (durations.len() as f64 / total_duration as f64) * 1000.0
        } else {
            0.0
        };

        // Regularity: inverse of coefficient of variation, clamped to [0, 1]
        let regularity = if cv > f64::EPSILON {
            (1.0 / (1.0 + cv)).min(1.0)
        } else {
            1.0
        };

        RhythmMetrics {
            avg_shot_length: avg,
            median_shot_length: median,
            std_deviation: std_dev,
            coefficient_of_variation: cv,
            min_length,
            max_length,
            cutting_rate,
            regularity,
        }
    }

    /// Detect pace segments using a sliding window.
    #[allow(clippy::cast_precision_loss)]
    fn detect_segments(&self, durations: &[u64]) -> Vec<PaceSegment> {
        if durations.len() < 2 {
            return Vec::new();
        }

        let window = self.trend_window.max(2).min(durations.len());
        let mut segments = Vec::new();
        let mut seg_start = 0_usize;
        let mut prev_trend = self.compute_trend(&durations[..window.min(durations.len())]);

        let step = 1;
        let mut i = step;
        while i + window <= durations.len() {
            let current_trend = self.compute_trend(&durations[i..i + window]);
            if current_trend.0 != prev_trend.0 {
                // End previous segment
                let avg_dur: f64 = durations[seg_start..i]
                    .iter()
                    .map(|&d| d as f64)
                    .sum::<f64>()
                    / (i - seg_start) as f64;
                segments.push(PaceSegment {
                    start_idx: seg_start,
                    end_idx: i,
                    trend: prev_trend.0,
                    avg_duration: avg_dur,
                    slope: prev_trend.1,
                });
                seg_start = i;
                prev_trend = current_trend;
            }
            i += step;
        }

        // Final segment
        let avg_dur: f64 = durations[seg_start..]
            .iter()
            .map(|&d| d as f64)
            .sum::<f64>()
            / (durations.len() - seg_start) as f64;
        segments.push(PaceSegment {
            start_idx: seg_start,
            end_idx: durations.len(),
            trend: prev_trend.0,
            avg_duration: avg_dur,
            slope: prev_trend.1,
        });

        segments
    }

    /// Compute the trend (slope) of a window of durations via linear regression.
    #[allow(clippy::cast_precision_loss)]
    fn compute_trend(&self, window: &[u64]) -> (PaceTrend, f64) {
        let n = window.len() as f64;
        if n < 2.0 {
            return (PaceTrend::Steady, 0.0);
        }

        let mut sum_x = 0.0_f64;
        let mut sum_y = 0.0_f64;
        let mut sum_xy = 0.0_f64;
        let mut sum_x2 = 0.0_f64;

        for (i, &d) in window.iter().enumerate() {
            let x = i as f64;
            let y = d as f64;
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_x2 += x * x;
        }

        let denom = n * sum_x2 - sum_x * sum_x;
        let slope = if denom.abs() > f64::EPSILON {
            (n * sum_xy - sum_x * sum_y) / denom
        } else {
            0.0
        };

        let trend = if slope < -self.slope_threshold {
            PaceTrend::Accelerating
        } else if slope > self.slope_threshold {
            PaceTrend::Decelerating
        } else {
            PaceTrend::Steady
        };

        (trend, slope)
    }

    /// Classify the dominant rhythm pattern.
    #[allow(clippy::cast_precision_loss)]
    fn classify_pattern(&self, metrics: &RhythmMetrics, durations: &[u64]) -> RhythmPattern {
        // Montage: very short shots
        if metrics.avg_shot_length < self.montage_threshold as f64 {
            return RhythmPattern::Montage;
        }

        // Contemplative: very long shots
        if metrics.avg_shot_length > self.contemplative_threshold as f64 {
            return RhythmPattern::Contemplative;
        }

        // Regular cadence: low coefficient of variation
        if metrics.coefficient_of_variation < 0.2 {
            return RhythmPattern::RegularCadence;
        }

        // Check for alternating pattern
        if self.is_alternating(durations) {
            return RhythmPattern::Alternating;
        }

        // Check for build-up or cool-down
        if durations.len() >= 3 {
            let (trend, _) = self.compute_trend(durations);
            match trend {
                PaceTrend::Accelerating => return RhythmPattern::BuildUp,
                PaceTrend::Decelerating => return RhythmPattern::CoolDown,
                PaceTrend::Steady => {}
            }
        }

        RhythmPattern::Freeform
    }

    /// Check if durations alternate between long and short.
    #[allow(clippy::cast_precision_loss)]
    fn is_alternating(&self, durations: &[u64]) -> bool {
        if durations.len() < 4 {
            return false;
        }

        let avg = durations.iter().sum::<u64>() as f64 / durations.len() as f64;
        let mut alternation_count = 0_usize;

        for i in 1..durations.len() {
            let prev_above = durations[i - 1] as f64 > avg;
            let curr_above = durations[i] as f64 > avg;
            if prev_above != curr_above {
                alternation_count += 1;
            }
        }

        let ratio = alternation_count as f64 / (durations.len() - 1) as f64;
        ratio > 0.7
    }

    /// Compute the instantaneous cutting rate over a sliding window.
    #[allow(clippy::cast_precision_loss)]
    pub fn local_cutting_rates(&self, durations: &[u64], window: usize) -> Vec<f64> {
        if durations.is_empty() || window == 0 {
            return Vec::new();
        }

        let w = window.min(durations.len());
        let mut rates = Vec::new();

        for i in 0..=durations.len().saturating_sub(w) {
            let slice = &durations[i..i + w];
            let total_dur: u64 = slice.iter().sum();
            let rate = if total_dur > 0 {
                (slice.len() as f64 / total_dur as f64) * 1000.0
            } else {
                0.0
            };
            rates.push(rate);
        }

        rates
    }
}

impl Default for ShotRhythmAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_analysis() {
        let analyzer = ShotRhythmAnalyzer::new();
        let result = analyzer.analyze(&[]);
        assert_eq!(result.shot_count, 0);
        assert!((result.metrics.avg_shot_length).abs() < f64::EPSILON);
        assert_eq!(result.dominant_pattern, RhythmPattern::Freeform);
    }

    #[test]
    fn test_single_shot() {
        let analyzer = ShotRhythmAnalyzer::new();
        let result = analyzer.analyze(&[60]);
        assert_eq!(result.shot_count, 1);
        assert!((result.metrics.avg_shot_length - 60.0).abs() < f64::EPSILON);
        assert_eq!(result.total_duration, 60);
    }

    #[test]
    fn test_regular_cadence() {
        let analyzer = ShotRhythmAnalyzer::new();
        let durations = vec![30, 30, 30, 30, 30, 30, 30, 30];
        let result = analyzer.analyze(&durations);
        assert!((result.metrics.avg_shot_length - 30.0).abs() < f64::EPSILON);
        assert!((result.metrics.std_deviation).abs() < f64::EPSILON);
        assert_eq!(result.dominant_pattern, RhythmPattern::RegularCadence);
    }

    #[test]
    fn test_montage_pattern() {
        let analyzer = ShotRhythmAnalyzer::new().with_montage_threshold(20);
        let durations = vec![5, 8, 6, 10, 7, 5, 9, 6];
        let result = analyzer.analyze(&durations);
        assert_eq!(result.dominant_pattern, RhythmPattern::Montage);
    }

    #[test]
    fn test_contemplative_pattern() {
        let analyzer = ShotRhythmAnalyzer::new();
        let durations = vec![400, 500, 350, 600, 450];
        let result = analyzer.analyze(&durations);
        assert_eq!(result.dominant_pattern, RhythmPattern::Contemplative);
    }

    #[test]
    fn test_metrics_min_max() {
        let analyzer = ShotRhythmAnalyzer::new();
        let durations = vec![10, 50, 20, 80, 30];
        let result = analyzer.analyze(&durations);
        assert_eq!(result.metrics.min_length, 10);
        assert_eq!(result.metrics.max_length, 80);
    }

    #[test]
    fn test_median_even() {
        let analyzer = ShotRhythmAnalyzer::new();
        let durations = vec![10, 20, 30, 40];
        let result = analyzer.analyze(&durations);
        assert!((result.metrics.median_shot_length - 25.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_median_odd() {
        let analyzer = ShotRhythmAnalyzer::new();
        let durations = vec![10, 20, 30];
        let result = analyzer.analyze(&durations);
        assert!((result.metrics.median_shot_length - 20.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cutting_rate() {
        let analyzer = ShotRhythmAnalyzer::new();
        let durations = vec![100, 100, 100, 100, 100]; // 5 shots, total 500 frames
        let result = analyzer.analyze(&durations);
        // cutting_rate = (5/500)*1000 = 10.0
        assert!((result.metrics.cutting_rate - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_local_cutting_rates() {
        let analyzer = ShotRhythmAnalyzer::new();
        let durations = vec![50, 50, 50, 50];
        let rates = analyzer.local_cutting_rates(&durations, 2);
        // Each window: 2 shots, 100 frames => rate = 20.0
        assert!(!rates.is_empty());
        for rate in &rates {
            assert!((*rate - 20.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_local_cutting_rates_empty() {
        let analyzer = ShotRhythmAnalyzer::new();
        let rates = analyzer.local_cutting_rates(&[], 5);
        assert!(rates.is_empty());
    }

    #[test]
    fn test_pace_segments_steady() {
        let analyzer = ShotRhythmAnalyzer::new().with_trend_window(3);
        let durations = vec![30, 30, 30, 30, 30, 30];
        let result = analyzer.analyze(&durations);
        // All shots same length => all segments should be Steady
        for seg in &result.segments {
            assert_eq!(seg.trend, PaceTrend::Steady);
        }
    }

    #[test]
    fn test_regularity_score() {
        let analyzer = ShotRhythmAnalyzer::new();
        // Perfectly regular
        let result = analyzer.analyze(&[30, 30, 30, 30]);
        assert!((result.metrics.regularity - 1.0).abs() < f64::EPSILON);

        // Irregular
        let result2 = analyzer.analyze(&[5, 200, 10, 300, 3]);
        assert!(result2.metrics.regularity < 0.5);
    }

    #[test]
    fn test_default_analyzer() {
        let analyzer = ShotRhythmAnalyzer::default();
        let result = analyzer.analyze(&[30, 60, 30, 60]);
        assert_eq!(result.shot_count, 4);
    }
}
