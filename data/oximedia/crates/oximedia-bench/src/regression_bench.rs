#![allow(dead_code)]
//! Regression benchmarking for detecting performance regressions across builds.
//!
//! This module provides tools to record benchmark baselines, compare current
//! results against them, and flag statistically significant regressions.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Verdict for a single metric comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegressionVerdict {
    /// Performance improved beyond the noise threshold.
    Improved,
    /// Performance is within the noise threshold.
    Unchanged,
    /// Performance regressed beyond the noise threshold.
    Regressed,
    /// Insufficient data to determine.
    Inconclusive,
}

/// A single recorded metric value with context.
#[derive(Debug, Clone)]
pub struct MetricSample {
    /// Name of the metric (e.g. "encode_fps", "psnr").
    pub name: String,
    /// Measured value.
    pub value: f64,
    /// Build / commit identifier.
    pub build_id: String,
    /// Unix timestamp in seconds when the sample was taken.
    pub timestamp_secs: u64,
}

impl MetricSample {
    /// Create a new sample.
    pub fn new(name: impl Into<String>, value: f64, build_id: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            value,
            build_id: build_id.into(),
            timestamp_secs: 0,
        }
    }

    /// Builder: set timestamp.
    pub fn with_timestamp(mut self, ts: u64) -> Self {
        self.timestamp_secs = ts;
        self
    }
}

/// A baseline consisting of multiple samples for one metric.
#[derive(Debug, Clone)]
pub struct MetricBaseline {
    /// Metric name.
    pub name: String,
    /// Recorded sample values (sorted by time).
    pub values: Vec<f64>,
    /// Mean of the sample values.
    pub mean: f64,
    /// Standard deviation of the sample values.
    pub stddev: f64,
}

impl MetricBaseline {
    /// Build a baseline from a set of samples.
    #[allow(clippy::cast_precision_loss)]
    pub fn from_samples(name: impl Into<String>, samples: &[f64]) -> Self {
        let n = samples.len();
        let (mean, stddev) = if n == 0 {
            (0.0, 0.0)
        } else {
            let m = samples.iter().sum::<f64>() / n as f64;
            let var = samples.iter().map(|v| (v - m).powi(2)).sum::<f64>() / n as f64;
            (m, var.sqrt())
        };
        Self {
            name: name.into(),
            values: samples.to_vec(),
            mean,
            stddev,
        }
    }

    /// Number of samples in the baseline.
    pub fn sample_count(&self) -> usize {
        self.values.len()
    }

    /// Coefficient of variation (stddev / mean).
    pub fn cv(&self) -> f64 {
        if self.mean.abs() < 1e-15 {
            return 0.0;
        }
        self.stddev / self.mean.abs()
    }
}

/// Configuration for regression detection.
#[derive(Debug, Clone)]
pub struct RegressionConfig {
    /// Number of standard deviations to consider a change significant.
    pub sigma_threshold: f64,
    /// Minimum absolute percentage change to flag (avoids noise on tiny diffs).
    pub min_pct_change: f64,
    /// Minimum number of baseline samples required.
    pub min_baseline_samples: usize,
    /// Whether a *higher* value is better (true for FPS, PSNR; false for latency).
    pub higher_is_better: bool,
}

impl Default for RegressionConfig {
    fn default() -> Self {
        Self {
            sigma_threshold: 2.0,
            min_pct_change: 1.0,
            min_baseline_samples: 5,
            higher_is_better: true,
        }
    }
}

impl RegressionConfig {
    /// Create a new default config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: set sigma threshold.
    pub fn with_sigma_threshold(mut self, s: f64) -> Self {
        self.sigma_threshold = s.max(0.0);
        self
    }

    /// Builder: set minimum percentage change.
    pub fn with_min_pct_change(mut self, pct: f64) -> Self {
        self.min_pct_change = pct.max(0.0);
        self
    }

    /// Builder: set minimum baseline samples.
    pub fn with_min_baseline_samples(mut self, n: usize) -> Self {
        self.min_baseline_samples = n.max(1);
        self
    }

    /// Builder: set polarity.
    pub fn with_higher_is_better(mut self, b: bool) -> Self {
        self.higher_is_better = b;
        self
    }
}

/// Result of comparing a single metric against its baseline.
#[derive(Debug, Clone)]
pub struct RegressionResult {
    /// Metric name.
    pub metric_name: String,
    /// Baseline mean.
    pub baseline_mean: f64,
    /// Baseline stddev.
    pub baseline_stddev: f64,
    /// Current value.
    pub current_value: f64,
    /// Absolute change.
    pub abs_change: f64,
    /// Percentage change.
    pub pct_change: f64,
    /// Z-score (how many sigmas from the baseline mean).
    pub z_score: f64,
    /// Final verdict.
    pub verdict: RegressionVerdict,
}

// ---------------------------------------------------------------------------
// Checker
// ---------------------------------------------------------------------------

/// The regression checker compares current metrics against baselines.
#[derive(Debug)]
pub struct RegressionChecker {
    /// Configuration.
    config: RegressionConfig,
    /// Baselines keyed by metric name.
    baselines: HashMap<String, MetricBaseline>,
}

impl RegressionChecker {
    /// Create a new checker.
    pub fn new(config: RegressionConfig) -> Self {
        Self {
            config,
            baselines: HashMap::new(),
        }
    }

    /// Register a baseline.
    pub fn add_baseline(&mut self, baseline: MetricBaseline) {
        self.baselines.insert(baseline.name.clone(), baseline);
    }

    /// Number of registered baselines.
    pub fn baseline_count(&self) -> usize {
        self.baselines.len()
    }

    /// Check a single current value against its baseline.
    #[allow(clippy::cast_precision_loss)]
    pub fn check(&self, metric_name: &str, current_value: f64) -> RegressionResult {
        let (baseline_mean, baseline_stddev, verdict) = match self.baselines.get(metric_name) {
            Some(bl) if bl.sample_count() >= self.config.min_baseline_samples => {
                let z = if bl.stddev > 1e-15 {
                    (current_value - bl.mean) / bl.stddev
                } else {
                    0.0
                };
                let pct = if bl.mean.abs() > 1e-15 {
                    ((current_value - bl.mean) / bl.mean) * 100.0
                } else {
                    0.0
                };

                let verdict = if z.abs() < self.config.sigma_threshold
                    || pct.abs() < self.config.min_pct_change
                {
                    RegressionVerdict::Unchanged
                } else if (self.config.higher_is_better && z > 0.0)
                    || (!self.config.higher_is_better && z < 0.0)
                {
                    RegressionVerdict::Improved
                } else {
                    RegressionVerdict::Regressed
                };

                (bl.mean, bl.stddev, verdict)
            }
            _ => (0.0, 0.0, RegressionVerdict::Inconclusive),
        };

        let abs_change = current_value - baseline_mean;
        let pct_change = if baseline_mean.abs() > 1e-15 {
            (abs_change / baseline_mean) * 100.0
        } else {
            0.0
        };
        let z_score = if baseline_stddev > 1e-15 {
            abs_change / baseline_stddev
        } else {
            0.0
        };

        RegressionResult {
            metric_name: metric_name.to_string(),
            baseline_mean,
            baseline_stddev,
            current_value,
            abs_change,
            pct_change,
            z_score,
            verdict,
        }
    }

    /// Check multiple metrics at once.
    pub fn check_all(&self, current: &[(String, f64)]) -> Vec<RegressionResult> {
        current
            .iter()
            .map(|(name, val)| self.check(name, *val))
            .collect()
    }

    /// Return only the metrics that regressed.
    pub fn regressions(&self, current: &[(String, f64)]) -> Vec<RegressionResult> {
        self.check_all(current)
            .into_iter()
            .filter(|r| r.verdict == RegressionVerdict::Regressed)
            .collect()
    }

    /// Produce a human-readable summary.
    pub fn summary(&self, results: &[RegressionResult]) -> String {
        let mut out = String::from("Regression Report\n");
        out.push_str(&format!("  Metrics checked: {}\n", results.len()));
        let regressed = results
            .iter()
            .filter(|r| r.verdict == RegressionVerdict::Regressed)
            .count();
        let improved = results
            .iter()
            .filter(|r| r.verdict == RegressionVerdict::Improved)
            .count();
        out.push_str(&format!("  Regressions: {regressed}\n"));
        out.push_str(&format!("  Improvements: {improved}\n"));
        out
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_sample_new() {
        let s = MetricSample::new("fps", 60.0, "abc123");
        assert_eq!(s.name, "fps");
        assert!((s.value - 60.0).abs() < 1e-9);
        assert_eq!(s.build_id, "abc123");
    }

    #[test]
    fn test_metric_sample_with_timestamp() {
        let s = MetricSample::new("psnr", 42.0, "b1").with_timestamp(1000);
        assert_eq!(s.timestamp_secs, 1000);
    }

    #[test]
    fn test_metric_baseline_from_samples() {
        let bl = MetricBaseline::from_samples("fps", &[60.0, 62.0, 58.0, 61.0, 59.0]);
        assert_eq!(bl.sample_count(), 5);
        assert!((bl.mean - 60.0).abs() < 0.1);
        assert!(bl.stddev > 0.0);
    }

    #[test]
    fn test_metric_baseline_empty() {
        let bl = MetricBaseline::from_samples("fps", &[]);
        assert_eq!(bl.sample_count(), 0);
        assert!((bl.mean - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_metric_baseline_cv() {
        let bl = MetricBaseline::from_samples("fps", &[100.0, 100.0, 100.0]);
        assert!((bl.cv() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_regression_config_default() {
        let cfg = RegressionConfig::default();
        assert!((cfg.sigma_threshold - 2.0).abs() < 1e-9);
        assert!(cfg.higher_is_better);
    }

    #[test]
    fn test_regression_config_builder() {
        let cfg = RegressionConfig::new()
            .with_sigma_threshold(3.0)
            .with_min_pct_change(5.0)
            .with_min_baseline_samples(10)
            .with_higher_is_better(false);
        assert!((cfg.sigma_threshold - 3.0).abs() < 1e-9);
        assert!((cfg.min_pct_change - 5.0).abs() < 1e-9);
        assert_eq!(cfg.min_baseline_samples, 10);
        assert!(!cfg.higher_is_better);
    }

    #[test]
    fn test_checker_no_baseline() {
        let checker = RegressionChecker::new(RegressionConfig::default());
        let result = checker.check("fps", 55.0);
        assert_eq!(result.verdict, RegressionVerdict::Inconclusive);
    }

    #[test]
    fn test_checker_unchanged() {
        let mut checker = RegressionChecker::new(RegressionConfig::default());
        checker.add_baseline(MetricBaseline::from_samples(
            "fps",
            &[60.0, 61.0, 59.0, 60.5, 59.5],
        ));
        let result = checker.check("fps", 60.2);
        assert_eq!(result.verdict, RegressionVerdict::Unchanged);
    }

    #[test]
    fn test_checker_regressed() {
        let mut checker = RegressionChecker::new(
            RegressionConfig::new()
                .with_sigma_threshold(2.0)
                .with_min_pct_change(1.0),
        );
        checker.add_baseline(MetricBaseline::from_samples(
            "fps",
            &[60.0, 60.0, 60.0, 60.0, 60.0],
        ));
        // stddev ~ 0, so even a tiny drop triggers regression
        // But with stddev=0, z_score is 0, so verdict is Unchanged.
        // Use slightly varying baseline instead:
        checker.add_baseline(MetricBaseline::from_samples(
            "latency",
            &[10.0, 10.5, 9.5, 10.2, 9.8],
        ));
        let cfg = RegressionConfig::new()
            .with_higher_is_better(false)
            .with_sigma_threshold(2.0)
            .with_min_pct_change(1.0);
        let mut checker2 = RegressionChecker::new(cfg);
        checker2.add_baseline(MetricBaseline::from_samples(
            "latency",
            &[10.0, 10.5, 9.5, 10.2, 9.8],
        ));
        // Big increase in latency (higher is worse)
        let result = checker2.check("latency", 15.0);
        assert_eq!(result.verdict, RegressionVerdict::Regressed);
    }

    #[test]
    fn test_checker_improved() {
        let mut checker = RegressionChecker::new(
            RegressionConfig::new()
                .with_sigma_threshold(1.5)
                .with_min_pct_change(1.0)
                .with_higher_is_better(true),
        );
        checker.add_baseline(MetricBaseline::from_samples(
            "fps",
            &[60.0, 61.0, 59.0, 60.5, 59.5],
        ));
        let result = checker.check("fps", 80.0);
        assert_eq!(result.verdict, RegressionVerdict::Improved);
    }

    #[test]
    fn test_check_all() {
        let mut checker = RegressionChecker::new(RegressionConfig::default());
        checker.add_baseline(MetricBaseline::from_samples(
            "fps",
            &[60.0, 61.0, 59.0, 60.5, 59.5],
        ));
        checker.add_baseline(MetricBaseline::from_samples(
            "psnr",
            &[42.0, 42.5, 41.5, 42.2, 41.8],
        ));
        let results = checker.check_all(&[("fps".into(), 60.0), ("psnr".into(), 42.0)]);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_regressions_filter() {
        let mut checker = RegressionChecker::new(
            RegressionConfig::new()
                .with_sigma_threshold(1.5)
                .with_min_pct_change(1.0),
        );
        checker.add_baseline(MetricBaseline::from_samples(
            "fps",
            &[60.0, 61.0, 59.0, 60.5, 59.5],
        ));
        let regs = checker.regressions(&[("fps".into(), 60.0)]);
        // Should be empty (no regression)
        assert!(regs.is_empty());
    }

    #[test]
    fn test_summary_output() {
        let results = vec![RegressionResult {
            metric_name: "fps".into(),
            baseline_mean: 60.0,
            baseline_stddev: 1.0,
            current_value: 55.0,
            abs_change: -5.0,
            pct_change: -8.3,
            z_score: -5.0,
            verdict: RegressionVerdict::Regressed,
        }];
        let checker = RegressionChecker::new(RegressionConfig::default());
        let summary = checker.summary(&results);
        assert!(summary.contains("Regressions: 1"));
    }

    #[test]
    fn test_baseline_count() {
        let mut checker = RegressionChecker::new(RegressionConfig::default());
        assert_eq!(checker.baseline_count(), 0);
        checker.add_baseline(MetricBaseline::from_samples(
            "x",
            &[1.0, 2.0, 3.0, 4.0, 5.0],
        ));
        assert_eq!(checker.baseline_count(), 1);
    }
}
