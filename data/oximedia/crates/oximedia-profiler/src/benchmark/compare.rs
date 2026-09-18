//! Benchmark comparison.

use super::runner::BenchmarkResult;
use serde::{Deserialize, Serialize};

/// Comparison result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonResult {
    /// Name of the comparison.
    pub name: String,

    /// Baseline result.
    pub baseline: BenchmarkResult,

    /// New result.
    pub current: BenchmarkResult,

    /// Speedup factor (>1.0 means faster, <1.0 means slower).
    pub speedup: f64,

    /// Percentage change.
    pub change_percent: f64,

    /// Whether this is an improvement.
    pub is_improvement: bool,

    /// Whether this is a regression.
    pub is_regression: bool,
}

/// Benchmark comparison.
#[derive(Debug)]
pub struct BenchmarkComparison {
    regression_threshold: f64,
}

impl BenchmarkComparison {
    /// Create a new benchmark comparison.
    pub fn new(regression_threshold: f64) -> Self {
        Self {
            regression_threshold,
        }
    }

    /// Compare two benchmark results.
    pub fn compare(&self, baseline: BenchmarkResult, current: BenchmarkResult) -> ComparisonResult {
        let baseline_secs = baseline.mean.as_secs_f64();
        let current_secs = current.mean.as_secs_f64();

        let speedup = if current_secs > 0.0 {
            baseline_secs / current_secs
        } else {
            1.0
        };

        let change_percent = (speedup - 1.0) * 100.0;
        let is_improvement = speedup > 1.0 + self.regression_threshold / 100.0;
        let is_regression = speedup < 1.0 - self.regression_threshold / 100.0;

        ComparisonResult {
            name: baseline.name.clone(),
            baseline,
            current,
            speedup,
            change_percent,
            is_improvement,
            is_regression,
        }
    }

    /// Generate a comparison report.
    pub fn report(&self, comparison: &ComparisonResult) -> String {
        let mut report = String::new();

        report.push_str(&format!("Benchmark: {}\n", comparison.name));
        report.push_str(&format!("Baseline: {:?}\n", comparison.baseline.mean));
        report.push_str(&format!("Current:  {:?}\n", comparison.current.mean));
        report.push_str(&format!("Speedup:  {:.2}x\n", comparison.speedup));
        report.push_str(&format!("Change:   {:+.2}%\n", comparison.change_percent));

        if comparison.is_improvement {
            report.push_str("Result:   IMPROVEMENT\n");
        } else if comparison.is_regression {
            report.push_str("Result:   REGRESSION\n");
        } else {
            report.push_str("Result:   NO SIGNIFICANT CHANGE\n");
        }

        report
    }
}

impl Default for BenchmarkComparison {
    fn default() -> Self {
        Self::new(5.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn make_result(name: &str, mean_ms: u64) -> BenchmarkResult {
        BenchmarkResult {
            name: name.to_string(),
            iterations: 100,
            mean: Duration::from_millis(mean_ms),
            median: Duration::from_millis(mean_ms),
            std_dev: Duration::ZERO,
            min: Duration::from_millis(mean_ms),
            max: Duration::from_millis(mean_ms),
            throughput: 1000.0 / mean_ms as f64,
        }
    }

    #[test]
    fn test_benchmark_comparison() {
        let comparison = BenchmarkComparison::new(5.0);
        let baseline = make_result("test", 100);
        let current = make_result("test", 50);

        let result = comparison.compare(baseline, current);
        assert_eq!(result.speedup, 2.0);
        assert!(result.is_improvement);
        assert!(!result.is_regression);
    }

    #[test]
    fn test_regression_detection() {
        let comparison = BenchmarkComparison::new(5.0);
        let baseline = make_result("test", 50);
        let current = make_result("test", 100);

        let result = comparison.compare(baseline, current);
        assert_eq!(result.speedup, 0.5);
        assert!(!result.is_improvement);
        assert!(result.is_regression);
    }

    #[test]
    fn test_no_change() {
        let comparison = BenchmarkComparison::new(5.0);
        let baseline = make_result("test", 100);
        let current = make_result("test", 102);

        let result = comparison.compare(baseline, current);
        assert!(!result.is_improvement);
        assert!(!result.is_regression);
    }
}
