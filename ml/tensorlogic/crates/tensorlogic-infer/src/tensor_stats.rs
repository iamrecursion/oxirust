//! Tensor statistics and anomaly detection for inference monitoring.
//!
//! Provides per-tensor statistics (mean, std, percentiles), IQR-based outlier
//! detection, and activation history tracking for debugging training pipelines.

use std::collections::HashMap;
use thiserror::Error;

/// Error types for statistics operations.
#[derive(Debug, Error)]
pub enum StatsError {
    /// The input data slice was empty.
    #[error("Empty data slice")]
    EmptyData,
    /// The requested percentile was out of range [0, 1].
    #[error("Invalid percentile: {0}")]
    InvalidPercentile(f64),
}

/// Summary statistics for a tensor.
#[derive(Debug, Clone)]
pub struct TensorStats {
    /// Arithmetic mean of finite values.
    pub mean: f64,
    /// Population standard deviation of finite values.
    pub std: f64,
    /// Minimum finite value.
    pub min: f64,
    /// Maximum finite value.
    pub max: f64,
    /// 25th percentile.
    pub p25: f64,
    /// 50th percentile (median).
    pub p50: f64,
    /// 75th percentile.
    pub p75: f64,
    /// Number of NaN values.
    pub nan_count: usize,
    /// Number of Inf values (positive or negative).
    pub inf_count: usize,
    /// Total number of elements.
    pub element_count: usize,
}

impl TensorStats {
    /// Compute statistics from a slice of f64 values.
    ///
    /// NaN and Inf values are counted but excluded from statistical calculations.
    pub fn compute(data: &[f64]) -> Result<Self, StatsError> {
        if data.is_empty() {
            return Err(StatsError::EmptyData);
        }
        let nan_count = data.iter().filter(|v| v.is_nan()).count();
        let inf_count = data.iter().filter(|v| v.is_infinite()).count();

        // Filter to finite values for stats
        let mut finite: Vec<f64> = data.iter().copied().filter(|v| v.is_finite()).collect();
        if finite.is_empty() {
            return Ok(TensorStats {
                mean: f64::NAN,
                std: f64::NAN,
                min: f64::NAN,
                max: f64::NAN,
                p25: f64::NAN,
                p50: f64::NAN,
                p75: f64::NAN,
                nan_count,
                inf_count,
                element_count: data.len(),
            });
        }
        finite.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = finite.len() as f64;
        let mean = finite.iter().sum::<f64>() / n;
        let variance = finite.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        let std = variance.sqrt();
        let min = finite[0];
        let max = finite[finite.len() - 1];

        // Percentiles using linear interpolation
        let p25 = percentile(&finite, 0.25);
        let p50 = percentile(&finite, 0.50);
        let p75 = percentile(&finite, 0.75);

        Ok(TensorStats {
            mean,
            std,
            min,
            max,
            p25,
            p50,
            p75,
            nan_count,
            inf_count,
            element_count: data.len(),
        })
    }

    /// Whether any NaN or Inf values are present.
    pub fn has_anomalies(&self) -> bool {
        self.nan_count > 0 || self.inf_count > 0
    }

    /// Interquartile range (p75 - p25).
    pub fn iqr(&self) -> f64 {
        self.p75 - self.p25
    }

    /// Range (max - min).
    pub fn range(&self) -> f64 {
        self.max - self.min
    }

    /// Coefficient of variation (std / |mean|).
    pub fn cv(&self) -> f64 {
        if self.mean.abs() < 1e-15 {
            f64::INFINITY
        } else {
            self.std / self.mean.abs()
        }
    }
}

/// Compute a percentile from sorted data using linear interpolation.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let idx = p * (sorted.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    let frac = idx - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// Kind of anomaly detected.
#[derive(Debug, Clone, PartialEq)]
pub enum AnomalyKind {
    /// Not-a-Number value.
    NaN,
    /// Infinite value.
    Inf,
    /// Statistical outlier (z-score exceeds threshold).
    Outlier {
        /// The z-score of the outlier.
        z_score: f64,
    },
    /// All values are identical (could indicate dead neuron).
    Constant,
}

/// Report of anomalies found in a tensor.
#[derive(Debug, Clone)]
pub struct AnomalyReport {
    /// List of (element_index, anomaly_kind) pairs.
    pub anomalies: Vec<(usize, AnomalyKind)>,
    /// Total anomaly count.
    pub anomaly_count: usize,
    /// True if no anomalies found.
    pub is_clean: bool,
}

/// Configurable anomaly detector.
pub struct AnomalyDetector {
    /// IQR multiplier for outlier detection (default 1.5).
    pub iqr_multiplier: f64,
    /// Z-score threshold for outlier detection (default 3.0).
    pub z_score_threshold: f64,
    /// Whether to flag constant-valued tensors.
    pub check_constant: bool,
}

impl Default for AnomalyDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl AnomalyDetector {
    /// Create a new anomaly detector with default settings.
    pub fn new() -> Self {
        AnomalyDetector {
            iqr_multiplier: 1.5,
            z_score_threshold: 3.0,
            check_constant: true,
        }
    }

    /// Set the IQR multiplier for outlier detection.
    pub fn with_iqr_multiplier(mut self, m: f64) -> Self {
        self.iqr_multiplier = m;
        self
    }

    /// Set the z-score threshold for outlier detection.
    pub fn with_z_score_threshold(mut self, t: f64) -> Self {
        self.z_score_threshold = t;
        self
    }

    /// Set whether to check for constant-valued tensors.
    pub fn with_check_constant(mut self, c: bool) -> Self {
        self.check_constant = c;
        self
    }

    /// Detect anomalies in data.
    pub fn detect(&self, data: &[f64]) -> AnomalyReport {
        let mut anomalies = Vec::new();

        // Check NaN and Inf
        for (i, &v) in data.iter().enumerate() {
            if v.is_nan() {
                anomalies.push((i, AnomalyKind::NaN));
            } else if v.is_infinite() {
                anomalies.push((i, AnomalyKind::Inf));
            }
        }

        // Compute stats for outlier detection (finite values only)
        let finite: Vec<f64> = data.iter().copied().filter(|v| v.is_finite()).collect();
        if finite.len() >= 2 {
            let mean = finite.iter().sum::<f64>() / finite.len() as f64;
            let std = (finite.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                / finite.len() as f64)
                .sqrt();

            if std > 1e-15 {
                for (i, &v) in data.iter().enumerate() {
                    if v.is_finite() {
                        let z = ((v - mean) / std).abs();
                        if z > self.z_score_threshold {
                            anomalies.push((i, AnomalyKind::Outlier { z_score: z }));
                        }
                    }
                }
            }

            // Check constant
            if self.check_constant && std < 1e-15 {
                anomalies.push((0, AnomalyKind::Constant));
            }
        } else if self.check_constant && finite.len() == 1 && data.len() > 1 {
            // All same value or only one finite value among many
            anomalies.push((0, AnomalyKind::Constant));
        }

        let count = anomalies.len();
        AnomalyReport {
            anomalies,
            anomaly_count: count,
            is_clean: count == 0,
        }
    }
}

/// Track statistics for named tensors across training steps.
pub struct ActivationStatistics {
    history: HashMap<String, Vec<TensorStats>>,
    max_history: usize,
}

impl ActivationStatistics {
    /// Create a new activation statistics tracker with the given history limit.
    pub fn new(max_history: usize) -> Self {
        ActivationStatistics {
            history: HashMap::new(),
            max_history: max_history.max(1),
        }
    }

    /// Record statistics for a named tensor.
    pub fn record(&mut self, name: &str, data: &[f64]) -> Result<(), StatsError> {
        let stats = TensorStats::compute(data)?;
        let entry = self.history.entry(name.to_string()).or_default();
        entry.push(stats);
        if entry.len() > self.max_history {
            entry.remove(0);
        }
        Ok(())
    }

    /// Get the most recent stats for a named tensor.
    pub fn latest(&self, name: &str) -> Option<&TensorStats> {
        self.history.get(name).and_then(|v| v.last())
    }

    /// Get the trend of means over history for a named tensor.
    pub fn trend_mean(&self, name: &str) -> Option<Vec<f64>> {
        self.history
            .get(name)
            .map(|v| v.iter().map(|s| s.mean).collect())
    }

    /// Get the trend of stds over history.
    pub fn trend_std(&self, name: &str) -> Option<Vec<f64>> {
        self.history
            .get(name)
            .map(|v| v.iter().map(|s| s.std).collect())
    }

    /// Iterator over all tracked tensor names.
    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.history.keys()
    }

    /// Number of tracked tensors.
    pub fn tracked_count(&self) -> usize {
        self.history.len()
    }

    /// Clear all history.
    pub fn clear(&mut self) {
        self.history.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_stats_basic() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = TensorStats::compute(&data).expect("compute failed");
        assert!(approx_eq(stats.mean, 3.0, EPSILON));
        // Population std = sqrt(2.0)
        assert!(approx_eq(stats.std, 2.0_f64.sqrt(), 1e-6));
        assert!(approx_eq(stats.min, 1.0, EPSILON));
        assert!(approx_eq(stats.max, 5.0, EPSILON));
    }

    #[test]
    fn test_stats_percentiles() {
        let data: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let stats = TensorStats::compute(&data).expect("compute failed");
        assert!(approx_eq(stats.p25, 25.75, 1e-6));
        assert!(approx_eq(stats.p50, 50.5, 1e-6));
        assert!(approx_eq(stats.p75, 75.25, 1e-6));
    }

    #[test]
    fn test_stats_single_element() {
        let data = [42.0];
        let stats = TensorStats::compute(&data).expect("compute failed");
        assert!(approx_eq(stats.mean, 42.0, EPSILON));
        assert!(approx_eq(stats.std, 0.0, EPSILON));
        assert!(approx_eq(stats.min, 42.0, EPSILON));
        assert!(approx_eq(stats.max, 42.0, EPSILON));
    }

    #[test]
    fn test_stats_all_same() {
        let data = [5.0, 5.0, 5.0, 5.0];
        let stats = TensorStats::compute(&data).expect("compute failed");
        assert!(approx_eq(stats.std, 0.0, EPSILON));
        assert!(approx_eq(stats.iqr(), 0.0, EPSILON));
    }

    #[test]
    fn test_stats_nan_count() {
        let data = [1.0, f64::NAN, 3.0];
        let stats = TensorStats::compute(&data).expect("compute failed");
        assert_eq!(stats.nan_count, 1);
        assert!(approx_eq(stats.mean, 2.0, EPSILON));
    }

    #[test]
    fn test_stats_inf_count() {
        let data = [1.0, f64::INFINITY, 3.0];
        let stats = TensorStats::compute(&data).expect("compute failed");
        assert_eq!(stats.inf_count, 1);
    }

    #[test]
    fn test_stats_has_anomalies() {
        let data = [1.0, f64::NAN, 3.0];
        let stats = TensorStats::compute(&data).expect("compute failed");
        assert!(stats.has_anomalies());
    }

    #[test]
    fn test_stats_empty_err() {
        let data: &[f64] = &[];
        let result = TensorStats::compute(data);
        assert!(result.is_err());
        assert!(matches!(result, Err(StatsError::EmptyData)));
    }

    #[test]
    fn test_stats_iqr() {
        let data: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let stats = TensorStats::compute(&data).expect("compute failed");
        let expected_iqr = stats.p75 - stats.p25;
        assert!(approx_eq(stats.iqr(), expected_iqr, EPSILON));
    }

    #[test]
    fn test_stats_cv() {
        let data = [2.0, 4.0, 6.0, 8.0, 10.0];
        let stats = TensorStats::compute(&data).expect("compute failed");
        let expected_cv = stats.std / stats.mean.abs();
        assert!(approx_eq(stats.cv(), expected_cv, EPSILON));
    }

    #[test]
    fn test_anomaly_clean() {
        let detector = AnomalyDetector::new();
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let report = detector.detect(&data);
        assert!(report.is_clean);
    }

    #[test]
    fn test_anomaly_nan() {
        let detector = AnomalyDetector::new();
        let data = [f64::NAN];
        let report = detector.detect(&data);
        assert!(!report.is_clean);
        assert!(report
            .anomalies
            .iter()
            .any(|(_, k)| matches!(k, AnomalyKind::NaN)));
    }

    #[test]
    fn test_anomaly_inf() {
        let detector = AnomalyDetector::new();
        let data = [f64::INFINITY];
        let report = detector.detect(&data);
        assert!(!report.is_clean);
        assert!(report
            .anomalies
            .iter()
            .any(|(_, k)| matches!(k, AnomalyKind::Inf)));
    }

    #[test]
    fn test_anomaly_outlier_zscore() {
        let detector = AnomalyDetector::new().with_z_score_threshold(1.5);
        let data = [0.0, 0.0, 0.0, 0.0, 100.0];
        let report = detector.detect(&data);
        assert!(!report.is_clean);
        assert!(report
            .anomalies
            .iter()
            .any(|(_, k)| matches!(k, AnomalyKind::Outlier { .. })));
    }

    #[test]
    fn test_anomaly_constant() {
        let detector = AnomalyDetector::new();
        let data = [7.0, 7.0, 7.0, 7.0];
        let report = detector.detect(&data);
        assert!(!report.is_clean);
        assert!(report
            .anomalies
            .iter()
            .any(|(_, k)| matches!(k, AnomalyKind::Constant)));
    }

    #[test]
    fn test_anomaly_no_constant_when_disabled() {
        let detector = AnomalyDetector::new().with_check_constant(false);
        let data = [7.0, 7.0, 7.0, 7.0];
        let report = detector.detect(&data);
        assert!(report.is_clean);
    }

    #[test]
    fn test_activation_record_and_latest() {
        let mut tracker = ActivationStatistics::new(10);
        tracker
            .record("layer1", &[1.0, 2.0, 3.0])
            .expect("record failed");
        tracker
            .record("layer1", &[4.0, 5.0, 6.0])
            .expect("record failed");
        tracker
            .record("layer1", &[7.0, 8.0, 9.0])
            .expect("record failed");
        let latest = tracker.latest("layer1").expect("no latest");
        assert!(approx_eq(latest.mean, 8.0, EPSILON));
    }

    #[test]
    fn test_activation_trend_mean() {
        let mut tracker = ActivationStatistics::new(10);
        tracker
            .record("layer1", &[1.0, 2.0, 3.0])
            .expect("record failed");
        tracker
            .record("layer1", &[4.0, 5.0, 6.0])
            .expect("record failed");
        tracker
            .record("layer1", &[7.0, 8.0, 9.0])
            .expect("record failed");
        let trend = tracker.trend_mean("layer1").expect("no trend");
        assert_eq!(trend.len(), 3);
        assert!(approx_eq(trend[0], 2.0, EPSILON));
        assert!(approx_eq(trend[1], 5.0, EPSILON));
        assert!(approx_eq(trend[2], 8.0, EPSILON));
    }

    #[test]
    fn test_activation_max_history_cap() {
        let mut tracker = ActivationStatistics::new(2);
        for i in 0..5 {
            let data = [i as f64];
            tracker.record("layer1", &data).expect("record failed");
        }
        let trend = tracker.trend_mean("layer1").expect("no trend");
        assert_eq!(trend.len(), 2);
        // Should have last two: 3.0 and 4.0
        assert!(approx_eq(trend[0], 3.0, EPSILON));
        assert!(approx_eq(trend[1], 4.0, EPSILON));
    }

    #[test]
    fn test_activation_clear() {
        let mut tracker = ActivationStatistics::new(10);
        tracker
            .record("layer1", &[1.0, 2.0])
            .expect("record failed");
        tracker
            .record("layer2", &[3.0, 4.0])
            .expect("record failed");
        assert_eq!(tracker.tracked_count(), 2);
        tracker.clear();
        assert_eq!(tracker.tracked_count(), 0);
        assert!(tracker.latest("layer1").is_none());
    }
}
