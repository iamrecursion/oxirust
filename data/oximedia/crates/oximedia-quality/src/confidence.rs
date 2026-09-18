//! Quality metric confidence intervals based on frame count.
//!
//! When assessing video quality, the reliability of the aggregated metric
//! depends on how many frames have been sampled. This module computes
//! confidence intervals (CI) for quality metrics using the Student-t
//! distribution approximation.
//!
//! # Method
//!
//! For each metric, given `n` samples with mean `x_bar` and sample standard
//! deviation `s`, the CI at confidence level `1 - alpha` is:
//!
//! ```text
//! x_bar +/- t_{alpha/2, n-1} * s / sqrt(n)
//! ```
//!
//! For large `n` (>= 30), the normal approximation `z_{alpha/2}` is used.
//! For small `n`, tabulated critical values of the t-distribution are used.
//!
//! # Example
//!
//! ```
//! use oximedia_quality::confidence::{ConfidenceCalculator, ConfidenceLevel};
//!
//! let calculator = ConfidenceCalculator::new(ConfidenceLevel::Ninety5);
//! let scores = vec![35.0, 37.0, 36.5, 38.0, 34.5];
//! let ci = calculator.compute(&scores);
//! assert!(ci.is_some());
//! let ci = ci.unwrap();
//! assert!(ci.lower < ci.mean);
//! assert!(ci.mean < ci.upper);
//! ```

use serde::{Deserialize, Serialize};

/// Confidence level for interval estimation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConfidenceLevel {
    /// 90% confidence interval (alpha = 0.10).
    Ninety,
    /// 95% confidence interval (alpha = 0.05).
    Ninety5,
    /// 99% confidence interval (alpha = 0.01).
    Ninety9,
}

impl ConfidenceLevel {
    /// Returns the z-value for the normal approximation (large samples).
    #[must_use]
    pub const fn z_value(&self) -> f64 {
        match self {
            Self::Ninety => 1.645,
            Self::Ninety5 => 1.960,
            Self::Ninety9 => 2.576,
        }
    }

    /// Returns the alpha (two-tailed) value.
    #[must_use]
    pub const fn alpha(&self) -> f64 {
        match self {
            Self::Ninety => 0.10,
            Self::Ninety5 => 0.05,
            Self::Ninety9 => 0.01,
        }
    }

    /// Returns the descriptive label.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Ninety => "90%",
            Self::Ninety5 => "95%",
            Self::Ninety9 => "99%",
        }
    }
}

/// A confidence interval for a quality metric.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceInterval {
    /// Sample mean.
    pub mean: f64,
    /// Lower bound of the confidence interval.
    pub lower: f64,
    /// Upper bound of the confidence interval.
    pub upper: f64,
    /// Half-width of the interval (margin of error).
    pub margin_of_error: f64,
    /// Sample standard deviation.
    pub sample_stddev: f64,
    /// Number of samples.
    pub sample_count: usize,
    /// Confidence level used.
    pub confidence_level: ConfidenceLevel,
    /// The critical value used (t or z).
    pub critical_value: f64,
    /// Standard error of the mean.
    pub standard_error: f64,
}

impl ConfidenceInterval {
    /// Returns the width of the confidence interval.
    #[must_use]
    pub fn width(&self) -> f64 {
        self.upper - self.lower
    }

    /// Returns true if the interval contains the given value.
    #[must_use]
    pub fn contains(&self, value: f64) -> bool {
        value >= self.lower && value <= self.upper
    }

    /// Returns the relative margin of error as a fraction of the mean.
    ///
    /// Returns `None` if the mean is zero.
    #[must_use]
    pub fn relative_margin(&self) -> Option<f64> {
        if self.mean.abs() < 1e-15 {
            None
        } else {
            Some(self.margin_of_error / self.mean.abs())
        }
    }

    /// Returns a human-readable summary.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "{} CI: {:.4} [{:.4}, {:.4}] (n={}, MoE={:.4})",
            self.confidence_level.label(),
            self.mean,
            self.lower,
            self.upper,
            self.sample_count,
            self.margin_of_error
        )
    }
}

/// Calculator for quality metric confidence intervals.
#[derive(Debug, Clone)]
pub struct ConfidenceCalculator {
    /// Confidence level.
    level: ConfidenceLevel,
}

impl ConfidenceCalculator {
    /// Creates a new calculator with the given confidence level.
    #[must_use]
    pub fn new(level: ConfidenceLevel) -> Self {
        Self { level }
    }

    /// Computes the confidence interval for the given sample values.
    ///
    /// Returns `None` if fewer than 2 samples are provided (CI requires
    /// at least 2 values to compute sample standard deviation).
    #[must_use]
    pub fn compute(&self, values: &[f64]) -> Option<ConfidenceInterval> {
        if values.len() < 2 {
            return None;
        }

        let n = values.len();
        let n_f = n as f64;
        let mean = values.iter().sum::<f64>() / n_f;

        // Sample standard deviation (Bessel's correction: n-1)
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n_f - 1.0);
        let sample_stddev = variance.sqrt();

        let standard_error = sample_stddev / n_f.sqrt();

        // Choose critical value: t-distribution for small n, z for large n
        let critical_value = if n >= 30 {
            self.level.z_value()
        } else {
            t_critical(n - 1, self.level)
        };

        let margin_of_error = critical_value * standard_error;
        let lower = mean - margin_of_error;
        let upper = mean + margin_of_error;

        Some(ConfidenceInterval {
            mean,
            lower,
            upper,
            margin_of_error,
            sample_stddev,
            sample_count: n,
            confidence_level: self.level,
            critical_value,
            standard_error,
        })
    }

    /// Estimates the minimum number of samples needed for a given margin
    /// of error, assuming the provided pilot standard deviation.
    ///
    /// Uses the formula: n = (z * sigma / margin)^2
    ///
    /// Returns at least 2.
    #[must_use]
    pub fn min_samples_for_margin(&self, pilot_stddev: f64, target_margin: f64) -> usize {
        if target_margin <= 0.0 || pilot_stddev <= 0.0 {
            return 2;
        }
        let z = self.level.z_value();
        let n_f = (z * pilot_stddev / target_margin).powi(2);
        (n_f.ceil() as usize).max(2)
    }

    /// Computes confidence intervals for multiple named metrics at once.
    ///
    /// Returns a map of metric name to confidence interval. Metrics with
    /// fewer than 2 samples are omitted.
    #[must_use]
    pub fn compute_multi(
        &self,
        metrics: &std::collections::HashMap<String, Vec<f64>>,
    ) -> std::collections::HashMap<String, ConfidenceInterval> {
        metrics
            .iter()
            .filter_map(|(name, values)| self.compute(values).map(|ci| (name.clone(), ci)))
            .collect()
    }
}

/// Tabulated critical values of the Student-t distribution.
///
/// For degrees of freedom `df` and the given confidence level, returns the
/// two-tailed critical value. Uses a lookup table for df 1..29 and falls
/// back to the z-value for df >= 30.
fn t_critical(df: usize, level: ConfidenceLevel) -> f64 {
    // Tabulated t-values for df=1..29 at 90%, 95%, 99% confidence
    // Source: standard statistical tables
    #[rustfmt::skip]
    const T90: [f64; 29] = [
        6.314, 2.920, 2.353, 2.132, 2.015, 1.943, 1.895, 1.860, 1.833, 1.812,
        1.796, 1.782, 1.771, 1.761, 1.753, 1.746, 1.740, 1.734, 1.729, 1.725,
        1.721, 1.717, 1.714, 1.711, 1.708, 1.706, 1.703, 1.701, 1.699,
    ];
    #[rustfmt::skip]
    const T95: [f64; 29] = [
        12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
        2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
        2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045,
    ];
    #[rustfmt::skip]
    const T99: [f64; 29] = [
        63.657, 9.925, 5.841, 4.604, 4.032, 3.707, 3.499, 3.355, 3.250, 3.169,
        3.106, 3.055, 3.012, 2.977, 2.947, 2.921, 2.898, 2.878, 2.861, 2.845,
        2.831, 2.819, 2.807, 2.797, 2.787, 2.779, 2.771, 2.763, 2.756,
    ];

    if df == 0 {
        return level.z_value();
    }
    if df > 29 {
        return level.z_value();
    }

    let idx = df - 1;
    match level {
        ConfidenceLevel::Ninety => T90[idx],
        ConfidenceLevel::Ninety5 => T95[idx],
        ConfidenceLevel::Ninety9 => T99[idx],
    }
}

/// Convenience function: compute a 95% confidence interval from a slice.
///
/// Returns `None` if fewer than 2 values.
#[must_use]
pub fn confidence_interval_95(values: &[f64]) -> Option<ConfidenceInterval> {
    ConfidenceCalculator::new(ConfidenceLevel::Ninety5).compute(values)
}

/// Determines a qualitative reliability rating based on the relative margin
/// of error.
///
/// | Rating       | Relative MoE  |
/// |-------------|---------------|
/// | Excellent   | < 1%          |
/// | Good        | < 5%          |
/// | Fair        | < 10%         |
/// | Poor        | < 20%         |
/// | Insufficient| >= 20%        |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReliabilityRating {
    /// Relative margin < 1%
    Excellent,
    /// Relative margin < 5%
    Good,
    /// Relative margin < 10%
    Fair,
    /// Relative margin < 20%
    Poor,
    /// Relative margin >= 20% or insufficient data
    Insufficient,
}

impl ReliabilityRating {
    /// Determines the rating from a confidence interval.
    #[must_use]
    pub fn from_ci(ci: &ConfidenceInterval) -> Self {
        match ci.relative_margin() {
            Some(rm) if rm < 0.01 => Self::Excellent,
            Some(rm) if rm < 0.05 => Self::Good,
            Some(rm) if rm < 0.10 => Self::Fair,
            Some(rm) if rm < 0.20 => Self::Poor,
            _ => Self::Insufficient,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_ci_basic() {
        let calc = ConfidenceCalculator::new(ConfidenceLevel::Ninety5);
        let values = vec![35.0, 37.0, 36.5, 38.0, 34.5];
        let ci = calc.compute(&values).expect("should compute CI");
        assert!(ci.lower < ci.mean);
        assert!(ci.mean < ci.upper);
        assert!(ci.margin_of_error > 0.0);
    }

    #[test]
    fn test_ci_single_value_returns_none() {
        let calc = ConfidenceCalculator::new(ConfidenceLevel::Ninety5);
        assert!(calc.compute(&[42.0]).is_none());
    }

    #[test]
    fn test_ci_empty_returns_none() {
        let calc = ConfidenceCalculator::new(ConfidenceLevel::Ninety5);
        assert!(calc.compute(&[]).is_none());
    }

    #[test]
    fn test_ci_two_values() {
        let calc = ConfidenceCalculator::new(ConfidenceLevel::Ninety5);
        let ci = calc.compute(&[10.0, 20.0]).expect("should compute CI");
        assert!((ci.mean - 15.0).abs() < 1e-6);
        assert!(ci.lower < 15.0);
        assert!(ci.upper > 15.0);
        // df=1, t_0.025 = 12.706 → very wide interval
        assert!(ci.width() > 50.0);
    }

    #[test]
    fn test_ci_identical_values_zero_width() {
        let calc = ConfidenceCalculator::new(ConfidenceLevel::Ninety5);
        let values = vec![42.0; 10];
        let ci = calc.compute(&values).expect("should compute CI");
        assert!((ci.mean - 42.0).abs() < 1e-10);
        assert!(ci.width() < 1e-10);
        assert!(ci.margin_of_error < 1e-10);
    }

    #[test]
    fn test_ci_contains() {
        let calc = ConfidenceCalculator::new(ConfidenceLevel::Ninety5);
        let values = vec![10.0, 12.0, 11.0, 13.0, 9.0];
        let ci = calc.compute(&values).expect("should compute CI");
        assert!(ci.contains(ci.mean));
        assert!(!ci.contains(ci.lower - 100.0));
    }

    #[test]
    fn test_ci_width() {
        let calc = ConfidenceCalculator::new(ConfidenceLevel::Ninety5);
        let values: Vec<f64> = (0..50).map(|i| 30.0 + (i as f64) * 0.1).collect();
        let ci = calc.compute(&values).expect("should compute CI");
        let expected_width = 2.0 * ci.margin_of_error;
        assert!((ci.width() - expected_width).abs() < 1e-10);
    }

    #[test]
    fn test_ci_narrows_with_more_samples() {
        let calc = ConfidenceCalculator::new(ConfidenceLevel::Ninety5);

        let small: Vec<f64> = (0..5).map(|i| 30.0 + i as f64).collect();
        let large: Vec<f64> = (0..100).map(|i| 30.0 + (i as f64) * 5.0 / 100.0).collect();

        let ci_small = calc.compute(&small).expect("should compute");
        let ci_large = calc.compute(&large).expect("should compute");

        // More samples → smaller margin of error (given similar stddev)
        assert!(ci_large.standard_error < ci_small.standard_error);
    }

    #[test]
    fn test_ci_90_narrower_than_99() {
        let values: Vec<f64> = (0..20).map(|i| 35.0 + i as f64 * 0.5).collect();
        let ci90 = ConfidenceCalculator::new(ConfidenceLevel::Ninety)
            .compute(&values)
            .expect("should compute");
        let ci99 = ConfidenceCalculator::new(ConfidenceLevel::Ninety9)
            .compute(&values)
            .expect("should compute");
        assert!(ci90.margin_of_error < ci99.margin_of_error);
    }

    #[test]
    fn test_ci_large_sample_uses_z() {
        let calc = ConfidenceCalculator::new(ConfidenceLevel::Ninety5);
        let values: Vec<f64> = (0..50).map(|i| 40.0 + i as f64 * 0.1).collect();
        let ci = calc.compute(&values).expect("should compute");
        // For n >= 30, critical value should be z = 1.96
        assert!((ci.critical_value - 1.96).abs() < 1e-3);
    }

    #[test]
    fn test_ci_small_sample_uses_t() {
        let calc = ConfidenceCalculator::new(ConfidenceLevel::Ninety5);
        let values: Vec<f64> = (0..5).map(|i| 40.0 + i as f64).collect();
        let ci = calc.compute(&values).expect("should compute");
        // df=4, t_0.025 = 2.776
        assert!((ci.critical_value - 2.776).abs() < 1e-3);
    }

    #[test]
    fn test_t_critical_table() {
        // Spot-check a few known values
        assert!((t_critical(1, ConfidenceLevel::Ninety5) - 12.706).abs() < 1e-3);
        assert!((t_critical(10, ConfidenceLevel::Ninety5) - 2.228).abs() < 1e-3);
        assert!((t_critical(29, ConfidenceLevel::Ninety5) - 2.045).abs() < 1e-3);
    }

    #[test]
    fn test_t_critical_large_df_falls_back_to_z() {
        let val = t_critical(100, ConfidenceLevel::Ninety5);
        assert!((val - 1.96).abs() < 1e-3);
    }

    #[test]
    fn test_min_samples_for_margin() {
        let calc = ConfidenceCalculator::new(ConfidenceLevel::Ninety5);
        let n = calc.min_samples_for_margin(5.0, 1.0);
        // (1.96 * 5 / 1)^2 = 96.04 → 97
        assert!(n >= 97);
        assert!(n <= 100);
    }

    #[test]
    fn test_min_samples_edge_cases() {
        let calc = ConfidenceCalculator::new(ConfidenceLevel::Ninety5);
        assert_eq!(calc.min_samples_for_margin(0.0, 1.0), 2);
        assert_eq!(calc.min_samples_for_margin(5.0, 0.0), 2);
        assert_eq!(calc.min_samples_for_margin(-1.0, 1.0), 2);
    }

    #[test]
    fn test_compute_multi() {
        let calc = ConfidenceCalculator::new(ConfidenceLevel::Ninety5);
        let mut metrics = HashMap::new();
        metrics.insert("psnr".to_string(), vec![35.0, 36.0, 37.0, 34.0, 38.0]);
        metrics.insert("ssim".to_string(), vec![0.95, 0.96, 0.94, 0.97]);
        metrics.insert("single".to_string(), vec![42.0]); // Should be omitted
        let results = calc.compute_multi(&metrics);
        assert!(results.contains_key("psnr"));
        assert!(results.contains_key("ssim"));
        assert!(!results.contains_key("single"));
    }

    #[test]
    fn test_convenience_function() {
        let values = vec![30.0, 35.0, 33.0, 37.0, 32.0];
        let ci = confidence_interval_95(&values).expect("should compute");
        assert_eq!(ci.confidence_level, ConfidenceLevel::Ninety5);
    }

    #[test]
    fn test_relative_margin() {
        let calc = ConfidenceCalculator::new(ConfidenceLevel::Ninety5);
        let values: Vec<f64> = (0..100).map(|i| 50.0 + i as f64 * 0.01).collect();
        let ci = calc.compute(&values).expect("should compute");
        let rm = ci.relative_margin().expect("mean is non-zero");
        assert!(rm > 0.0);
        assert!(rm < 1.0);
    }

    #[test]
    fn test_relative_margin_zero_mean() {
        let calc = ConfidenceCalculator::new(ConfidenceLevel::Ninety5);
        let values = vec![-1.0, 1.0, -1.0, 1.0];
        let ci = calc.compute(&values).expect("should compute");
        if ci.mean.abs() < 1e-10 {
            assert!(ci.relative_margin().is_none());
        }
    }

    #[test]
    fn test_reliability_rating_excellent() {
        let ci = ConfidenceInterval {
            mean: 100.0,
            lower: 99.5,
            upper: 100.5,
            margin_of_error: 0.5,
            sample_stddev: 1.0,
            sample_count: 1000,
            confidence_level: ConfidenceLevel::Ninety5,
            critical_value: 1.96,
            standard_error: 0.255,
        };
        assert_eq!(
            ReliabilityRating::from_ci(&ci),
            ReliabilityRating::Excellent
        );
    }

    #[test]
    fn test_reliability_rating_poor() {
        let ci = ConfidenceInterval {
            mean: 10.0,
            lower: 8.5,
            upper: 11.5,
            margin_of_error: 1.5,
            sample_stddev: 5.0,
            sample_count: 5,
            confidence_level: ConfidenceLevel::Ninety5,
            critical_value: 2.776,
            standard_error: 0.54,
        };
        assert_eq!(ReliabilityRating::from_ci(&ci), ReliabilityRating::Poor);
    }

    #[test]
    fn test_ci_summary_format() {
        let calc = ConfidenceCalculator::new(ConfidenceLevel::Ninety5);
        let values = vec![35.0, 36.0, 37.0, 34.0, 38.0];
        let ci = calc.compute(&values).expect("should compute");
        let summary = ci.summary();
        assert!(summary.contains("95%"));
        assert!(summary.contains("n=5"));
    }

    #[test]
    fn test_confidence_level_labels() {
        assert_eq!(ConfidenceLevel::Ninety.label(), "90%");
        assert_eq!(ConfidenceLevel::Ninety5.label(), "95%");
        assert_eq!(ConfidenceLevel::Ninety9.label(), "99%");
    }

    #[test]
    fn test_confidence_level_alpha() {
        assert!((ConfidenceLevel::Ninety.alpha() - 0.10).abs() < 1e-10);
        assert!((ConfidenceLevel::Ninety5.alpha() - 0.05).abs() < 1e-10);
        assert!((ConfidenceLevel::Ninety9.alpha() - 0.01).abs() < 1e-10);
    }
}
