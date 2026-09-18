//! Benchmark comparison and regression detection.
//!
//! Provides utilities for comparing a current benchmark run against a baseline,
//! computing speedup/slowdown, and detecting regressions or improvements.

#![allow(dead_code)]

/// The performance delta between a baseline and a current benchmark run.
#[derive(Debug, Clone, PartialEq)]
pub struct BenchDelta {
    /// Baseline measurement in nanoseconds.
    pub baseline_ns: f64,
    /// Current measurement in nanoseconds.
    pub current_ns: f64,
}

impl BenchDelta {
    /// Create a new `BenchDelta`.
    #[must_use]
    pub fn new(baseline_ns: f64, current_ns: f64) -> Self {
        Self {
            baseline_ns,
            current_ns,
        }
    }

    /// Speedup factor: `baseline / current`.
    ///
    /// A value > 1.0 means the current run is faster; < 1.0 means it is slower.
    /// Returns `f64::INFINITY` when `current_ns` is zero.
    #[must_use]
    pub fn speedup(&self) -> f64 {
        if self.current_ns == 0.0 {
            return f64::INFINITY;
        }
        self.baseline_ns / self.current_ns
    }

    /// Percentage change relative to the baseline.
    ///
    /// Positive values indicate the current run is **slower** (regression).
    /// Returns `f64::INFINITY` when `baseline_ns` is zero.
    #[must_use]
    pub fn pct_change(&self) -> f64 {
        if self.baseline_ns == 0.0 {
            return f64::INFINITY;
        }
        (self.current_ns - self.baseline_ns) / self.baseline_ns * 100.0
    }

    /// Returns `true` when `pct_change()` exceeds `threshold_pct` (current is slower).
    #[must_use]
    pub fn is_regression(&self, threshold_pct: f64) -> bool {
        self.pct_change() > threshold_pct
    }

    /// Returns `true` when `pct_change()` is below `-threshold_pct` (current is faster).
    #[must_use]
    pub fn is_improvement(&self, threshold_pct: f64) -> bool {
        self.pct_change() < -threshold_pct
    }
}

/// The result of comparing a single named benchmark against its baseline.
#[derive(Debug, Clone)]
pub struct PerfComparisonResult {
    /// Benchmark name.
    pub name: String,
    /// Performance delta.
    pub delta: BenchDelta,
    /// Whether the change is statistically significant (caller-determined).
    pub significant: bool,
}

impl PerfComparisonResult {
    /// Returns a human-readable verdict: `"regression"`, `"improvement"`, or `"unchanged"`.
    #[must_use]
    pub fn verdict(&self) -> &str {
        if !self.significant {
            return "unchanged";
        }
        let pct = self.delta.pct_change();
        if pct > 0.0 {
            "regression"
        } else if pct < 0.0 {
            "improvement"
        } else {
            "unchanged"
        }
    }
}

/// Compares benchmark results against a baseline and identifies regressions.
#[derive(Debug, Clone)]
pub struct BenchComparator {
    /// Percentage change threshold above which a result is a regression.
    pub regression_threshold_pct: f64,
    /// Percentage change threshold below which a result is an improvement.
    pub improvement_threshold_pct: f64,
}

impl Default for BenchComparator {
    fn default() -> Self {
        Self {
            regression_threshold_pct: 5.0,
            improvement_threshold_pct: 5.0,
        }
    }
}

impl BenchComparator {
    /// Create a comparator with custom thresholds.
    #[must_use]
    pub fn new(regression_threshold_pct: f64, improvement_threshold_pct: f64) -> Self {
        Self {
            regression_threshold_pct,
            improvement_threshold_pct,
        }
    }

    /// Compare a named benchmark's `current` against `baseline` (both in ns).
    #[must_use]
    pub fn compare(&self, name: &str, baseline: f64, current: f64) -> PerfComparisonResult {
        let delta = BenchDelta::new(baseline, current);
        let significant = delta.is_regression(self.regression_threshold_pct)
            || delta.is_improvement(self.improvement_threshold_pct);

        PerfComparisonResult {
            name: name.to_string(),
            delta,
            significant,
        }
    }

    /// Returns `true` when any result in `results` is a regression.
    #[must_use]
    pub fn has_regression(&self, results: &[PerfComparisonResult]) -> bool {
        results
            .iter()
            .any(|r| r.delta.is_regression(self.regression_threshold_pct))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ----- BenchDelta tests -----

    #[test]
    fn test_bench_delta_speedup_faster() {
        let d = BenchDelta::new(200.0, 100.0);
        assert!((d.speedup() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_bench_delta_speedup_slower() {
        let d = BenchDelta::new(100.0, 200.0);
        assert!((d.speedup() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_bench_delta_speedup_zero_current() {
        let d = BenchDelta::new(100.0, 0.0);
        assert_eq!(d.speedup(), f64::INFINITY);
    }

    #[test]
    fn test_bench_delta_pct_change_regression() {
        let d = BenchDelta::new(100.0, 120.0);
        assert!((d.pct_change() - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_bench_delta_pct_change_improvement() {
        let d = BenchDelta::new(100.0, 80.0);
        assert!((d.pct_change() - (-20.0)).abs() < 1e-10);
    }

    #[test]
    fn test_bench_delta_pct_change_no_change() {
        let d = BenchDelta::new(100.0, 100.0);
        assert_eq!(d.pct_change(), 0.0);
    }

    #[test]
    fn test_bench_delta_is_regression_above_threshold() {
        let d = BenchDelta::new(100.0, 115.0);
        assert!(d.is_regression(10.0)); // 15% > 10%
    }

    #[test]
    fn test_bench_delta_is_regression_below_threshold() {
        let d = BenchDelta::new(100.0, 103.0);
        assert!(!d.is_regression(10.0)); // 3% < 10%
    }

    #[test]
    fn test_bench_delta_is_improvement() {
        let d = BenchDelta::new(100.0, 80.0);
        assert!(d.is_improvement(10.0)); // -20% is an improvement
    }

    #[test]
    fn test_bench_delta_not_improvement_small_change() {
        let d = BenchDelta::new(100.0, 97.0);
        assert!(!d.is_improvement(10.0)); // -3% < threshold
    }

    // ----- PerfComparisonResult tests -----

    #[test]
    fn test_verdict_regression() {
        let result = PerfComparisonResult {
            name: "foo".to_string(),
            delta: BenchDelta::new(100.0, 120.0),
            significant: true,
        };
        assert_eq!(result.verdict(), "regression");
    }

    #[test]
    fn test_verdict_improvement() {
        let result = PerfComparisonResult {
            name: "foo".to_string(),
            delta: BenchDelta::new(100.0, 80.0),
            significant: true,
        };
        assert_eq!(result.verdict(), "improvement");
    }

    #[test]
    fn test_verdict_unchanged() {
        let result = PerfComparisonResult {
            name: "foo".to_string(),
            delta: BenchDelta::new(100.0, 102.0),
            significant: false,
        };
        assert_eq!(result.verdict(), "unchanged");
    }

    // ----- BenchComparator tests -----

    #[test]
    fn test_comparator_default() {
        let cmp = BenchComparator::default();
        assert!((cmp.regression_threshold_pct - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_comparator_compare_regression() {
        let cmp = BenchComparator::default();
        let result = cmp.compare("bench_x", 100.0, 120.0);
        assert_eq!(result.verdict(), "regression");
    }

    #[test]
    fn test_comparator_has_regression_true() {
        let cmp = BenchComparator::default();
        let results = vec![cmp.compare("a", 100.0, 150.0)];
        assert!(cmp.has_regression(&results));
    }

    #[test]
    fn test_comparator_has_regression_false() {
        let cmp = BenchComparator::default();
        let results = vec![cmp.compare("a", 100.0, 101.0)];
        assert!(!cmp.has_regression(&results));
    }
}
