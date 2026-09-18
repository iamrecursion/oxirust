//! Profile snapshot comparison and regression detection.
//!
//! Compares two `ProfileSnapshot`s — a baseline and a current measurement —
//! and emits a `RegressionReport` listing regressions (>10 % slowdown by
//! default) and improvements.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

// ---------------------------------------------------------------------------
// NodeStats
// ---------------------------------------------------------------------------

/// Aggregated timing statistics for a single named profiling node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeStats {
    /// Minimum observed duration.
    pub min_ms: f64,
    /// Maximum observed duration.
    pub max_ms: f64,
    /// Arithmetic mean duration.
    pub mean_ms: f64,
    /// 50th-percentile (median) duration.
    pub p50_ms: f64,
    /// 95th-percentile duration.
    pub p95_ms: f64,
    /// 99th-percentile duration.
    pub p99_ms: f64,
    /// Number of observations.
    pub count: u64,
}

impl NodeStats {
    /// Builds a `NodeStats` from a slice of raw millisecond observations.
    ///
    /// Returns `None` if `samples` is empty.
    #[must_use]
    pub fn from_samples(samples: &[f64]) -> Option<Self> {
        if samples.is_empty() {
            return None;
        }

        let mut sorted = samples.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = sorted.len();
        let min_ms = sorted[0];
        let max_ms = sorted[n - 1];
        let mean_ms = sorted.iter().sum::<f64>() / n as f64;
        let p50_ms = percentile(&sorted, 0.50);
        let p95_ms = percentile(&sorted, 0.95);
        let p99_ms = percentile(&sorted, 0.99);

        Some(Self {
            min_ms,
            max_ms,
            mean_ms,
            p50_ms,
            p95_ms,
            p99_ms,
            count: n as u64,
        })
    }

    /// Builds `NodeStats` from a slice of `Duration` values.
    #[must_use]
    pub fn from_durations(durations: &[Duration]) -> Option<Self> {
        if durations.is_empty() {
            return None;
        }
        let ms: Vec<f64> = durations
            .iter()
            .map(|d| d.as_secs_f64() * 1_000.0)
            .collect();
        Self::from_samples(&ms)
    }
}

/// Linear-interpolation percentile for a pre-sorted slice.
fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.len() == 1 {
        return sorted[0];
    }
    let idx = q * (sorted.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    let frac = idx - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

// ---------------------------------------------------------------------------
// ProfileSnapshot
// ---------------------------------------------------------------------------

/// A snapshot of per-node timing statistics at a specific point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileSnapshot {
    /// Wall-clock time when the snapshot was taken.
    pub timestamp: SystemTime,
    /// Optional human-readable label for this snapshot.
    pub label: Option<String>,
    /// Per-node timing statistics, keyed by node name.
    pub node_timings: HashMap<String, NodeStats>,
}

impl ProfileSnapshot {
    /// Creates an empty snapshot with the current wall-clock time.
    #[must_use]
    pub fn new() -> Self {
        Self {
            timestamp: SystemTime::now(),
            label: None,
            node_timings: HashMap::new(),
        }
    }

    /// Creates a snapshot with a human-readable label.
    #[must_use]
    pub fn with_label(label: impl Into<String>) -> Self {
        Self {
            label: Some(label.into()),
            ..Self::new()
        }
    }

    /// Inserts statistics for `node_name` from raw millisecond samples.
    ///
    /// Returns `false` and does nothing if `samples` is empty.
    pub fn record_samples(&mut self, node_name: impl Into<String>, samples: &[f64]) -> bool {
        match NodeStats::from_samples(samples) {
            Some(stats) => {
                self.node_timings.insert(node_name.into(), stats);
                true
            }
            None => false,
        }
    }

    /// Inserts statistics for `node_name` from `Duration` observations.
    pub fn record_durations(
        &mut self,
        node_name: impl Into<String>,
        durations: &[Duration],
    ) -> bool {
        match NodeStats::from_durations(durations) {
            Some(stats) => {
                self.node_timings.insert(node_name.into(), stats);
                true
            }
            None => false,
        }
    }
}

impl Default for ProfileSnapshot {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// RegressionReport items
// ---------------------------------------------------------------------------

/// A single performance regression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Regression {
    /// Name of the profiling node that regressed.
    pub node_name: String,
    /// Baseline p95 latency in milliseconds.
    pub baseline_p95_ms: f64,
    /// Current p95 latency in milliseconds.
    pub current_p95_ms: f64,
    /// Percentage change relative to the baseline (positive = slower).
    pub pct_change: f64,
    /// Absolute difference (current − baseline) in milliseconds.
    pub delta_ms: f64,
}

/// A single performance improvement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Improvement {
    /// Name of the profiling node that improved.
    pub node_name: String,
    /// Baseline p95 latency in milliseconds.
    pub baseline_p95_ms: f64,
    /// Current p95 latency in milliseconds.
    pub current_p95_ms: f64,
    /// Percentage change (negative = faster).
    pub pct_change: f64,
    /// Absolute difference (current − baseline) in milliseconds.
    pub delta_ms: f64,
}

// ---------------------------------------------------------------------------
// RegressionReport
// ---------------------------------------------------------------------------

/// The result of comparing two `ProfileSnapshot`s.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionReport {
    /// Nodes that became slower by at least the configured threshold.
    pub regressions: Vec<Regression>,
    /// Nodes that became faster by at least the configured threshold.
    pub improvements: Vec<Improvement>,
    /// Nodes present in both snapshots with negligible change.
    pub stable_count: usize,
    /// Nodes only in the baseline snapshot (removed in current).
    pub removed_nodes: Vec<String>,
    /// Nodes only in the current snapshot (new since baseline).
    pub new_nodes: Vec<String>,
    /// Human-readable executive summary.
    pub summary: String,
}

impl RegressionReport {
    /// Returns `true` if any regressions were found.
    #[must_use]
    pub fn has_regressions(&self) -> bool {
        !self.regressions.is_empty()
    }

    /// Returns the most severe regression, if any.
    #[must_use]
    pub fn worst_regression(&self) -> Option<&Regression> {
        self.regressions.iter().max_by(|a, b| {
            a.pct_change
                .partial_cmp(&b.pct_change)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Returns the most significant improvement, if any.
    #[must_use]
    pub fn best_improvement(&self) -> Option<&Improvement> {
        self.improvements.iter().min_by(|a, b| {
            a.pct_change
                .partial_cmp(&b.pct_change)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }
}

// ---------------------------------------------------------------------------
// ProfileComparator
// ---------------------------------------------------------------------------

/// Compares two `ProfileSnapshot`s and produces a `RegressionReport`.
#[derive(Debug, Clone)]
pub struct ProfileComparator {
    /// Minimum percentage increase in p95 to flag as a regression (default 10.0).
    pub regression_threshold_pct: f64,
    /// Minimum percentage decrease in p95 to flag as an improvement (default 5.0).
    pub improvement_threshold_pct: f64,
}

impl ProfileComparator {
    /// Creates a `ProfileComparator` with the given thresholds.
    #[must_use]
    pub fn new(regression_threshold_pct: f64, improvement_threshold_pct: f64) -> Self {
        Self {
            regression_threshold_pct,
            improvement_threshold_pct,
        }
    }

    /// Compares `baseline` against `current` and returns a `RegressionReport`.
    #[must_use]
    pub fn compare(
        &self,
        baseline: &ProfileSnapshot,
        current: &ProfileSnapshot,
    ) -> RegressionReport {
        let mut regressions = Vec::new();
        let mut improvements = Vec::new();
        let mut stable_count = 0usize;
        let mut removed_nodes = Vec::new();
        let mut new_nodes = Vec::new();

        // --- Nodes present in baseline ---
        for (name, base_stats) in &baseline.node_timings {
            match current.node_timings.get(name) {
                None => removed_nodes.push(name.clone()),
                Some(cur_stats) => {
                    let delta = cur_stats.p95_ms - base_stats.p95_ms;
                    let pct = if base_stats.p95_ms > f64::EPSILON {
                        (delta / base_stats.p95_ms) * 100.0
                    } else {
                        0.0
                    };

                    if pct >= self.regression_threshold_pct {
                        regressions.push(Regression {
                            node_name: name.clone(),
                            baseline_p95_ms: base_stats.p95_ms,
                            current_p95_ms: cur_stats.p95_ms,
                            pct_change: pct,
                            delta_ms: delta,
                        });
                    } else if pct <= -self.improvement_threshold_pct {
                        improvements.push(Improvement {
                            node_name: name.clone(),
                            baseline_p95_ms: base_stats.p95_ms,
                            current_p95_ms: cur_stats.p95_ms,
                            pct_change: pct,
                            delta_ms: delta,
                        });
                    } else {
                        stable_count += 1;
                    }
                }
            }
        }

        // --- Nodes only in current ---
        for name in current.node_timings.keys() {
            if !baseline.node_timings.contains_key(name) {
                new_nodes.push(name.clone());
            }
        }

        // Sort for deterministic output
        regressions.sort_by(|a, b| {
            b.pct_change
                .partial_cmp(&a.pct_change)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        improvements.sort_by(|a, b| {
            a.pct_change
                .partial_cmp(&b.pct_change)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        removed_nodes.sort();
        new_nodes.sort();

        let summary = build_summary(
            &regressions,
            &improvements,
            stable_count,
            &removed_nodes,
            &new_nodes,
            baseline,
            current,
        );

        RegressionReport {
            regressions,
            improvements,
            stable_count,
            removed_nodes,
            new_nodes,
            summary,
        }
    }
}

impl Default for ProfileComparator {
    fn default() -> Self {
        Self::new(10.0, 5.0)
    }
}

fn build_summary(
    regressions: &[Regression],
    improvements: &[Improvement],
    stable_count: usize,
    removed_nodes: &[String],
    new_nodes: &[String],
    baseline: &ProfileSnapshot,
    current: &ProfileSnapshot,
) -> String {
    let baseline_label = baseline.label.as_deref().unwrap_or("baseline");
    let current_label = current.label.as_deref().unwrap_or("current");

    let mut s = format!(
        "Profile comparison: '{}' vs '{}'\n",
        baseline_label, current_label
    );
    s.push_str(&format!(
        "  Regressions: {}  Improvements: {}  Stable: {}  New: {}  Removed: {}\n",
        regressions.len(),
        improvements.len(),
        stable_count,
        new_nodes.len(),
        removed_nodes.len(),
    ));

    if let Some(worst) = regressions.first() {
        s.push_str(&format!(
            "  Worst regression: '{}' +{:.1}% (p95 {:.2} ms → {:.2} ms)\n",
            worst.node_name, worst.pct_change, worst.baseline_p95_ms, worst.current_p95_ms
        ));
    }

    if let Some(best) = improvements.first() {
        s.push_str(&format!(
            "  Best improvement:  '{}' {:.1}% (p95 {:.2} ms → {:.2} ms)\n",
            best.node_name, best.pct_change, best.baseline_p95_ms, best.current_p95_ms
        ));
    }

    s
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a snapshot with a single node sampled at a fixed latency.
    fn snapshot_fixed(node: &str, ms: f64, label: &str) -> ProfileSnapshot {
        let mut snap = ProfileSnapshot::with_label(label);
        let samples: Vec<f64> = vec![ms; 20];
        snap.record_samples(node, &samples);
        snap
    }

    // Helper: build a snapshot with a node sampled at varying latency.
    fn snapshot_variable(node: &str, samples: &[f64], label: &str) -> ProfileSnapshot {
        let mut snap = ProfileSnapshot::with_label(label);
        snap.record_samples(node, samples);
        snap
    }

    #[test]
    fn test_no_regression_when_within_threshold() {
        let baseline = snapshot_fixed("encode", 100.0, "v1");
        let current = snapshot_fixed("encode", 105.0, "v2"); // +5 % < 10 % threshold
        let cmp = ProfileComparator::default();
        let report = cmp.compare(&baseline, &current);
        assert!(!report.has_regressions(), "unexpected regression");
        assert_eq!(report.stable_count, 1);
    }

    #[test]
    fn test_regression_detected_above_threshold() {
        let baseline = snapshot_fixed("encode", 100.0, "v1");
        let current = snapshot_fixed("encode", 120.0, "v2"); // +20 %
        let cmp = ProfileComparator::default();
        let report = cmp.compare(&baseline, &current);
        assert!(report.has_regressions());
        assert_eq!(report.regressions[0].node_name, "encode");
        assert!((report.regressions[0].pct_change - 20.0).abs() < 0.5);
    }

    #[test]
    fn test_improvement_detected() {
        let baseline = snapshot_fixed("decode", 100.0, "v1");
        let current = snapshot_fixed("decode", 80.0, "v2"); // -20 %
        let cmp = ProfileComparator::default();
        let report = cmp.compare(&baseline, &current);
        assert_eq!(report.improvements.len(), 1);
        assert!(report.improvements[0].pct_change < -5.0);
    }

    #[test]
    fn test_removed_node_detected() {
        let baseline = snapshot_fixed("old_step", 50.0, "v1");
        let current = ProfileSnapshot::with_label("v2"); // empty
        let cmp = ProfileComparator::default();
        let report = cmp.compare(&baseline, &current);
        assert!(report.removed_nodes.contains(&"old_step".to_owned()));
    }

    #[test]
    fn test_new_node_detected() {
        let baseline = ProfileSnapshot::with_label("v1");
        let current = snapshot_fixed("new_step", 30.0, "v2");
        let cmp = ProfileComparator::default();
        let report = cmp.compare(&baseline, &current);
        assert!(report.new_nodes.contains(&"new_step".to_owned()));
    }

    #[test]
    fn test_multiple_nodes_mixed() {
        let mut baseline = ProfileSnapshot::with_label("v1");
        baseline.record_samples("fast", &vec![10.0; 20]);
        baseline.record_samples("slow", &vec![100.0; 20]);
        baseline.record_samples("stable", &vec![50.0; 20]);

        let mut current = ProfileSnapshot::with_label("v2");
        current.record_samples("fast", &vec![7.0; 20]); // improved
        current.record_samples("slow", &vec![150.0; 20]); // regression +50 %
        current.record_samples("stable", &vec![52.0; 20]); // stable

        let cmp = ProfileComparator::default();
        let report = cmp.compare(&baseline, &current);

        assert_eq!(report.regressions.len(), 1);
        assert_eq!(report.regressions[0].node_name, "slow");
        assert_eq!(report.improvements.len(), 1);
        assert_eq!(report.improvements[0].node_name, "fast");
        assert_eq!(report.stable_count, 1);
    }

    #[test]
    fn test_worst_regression_accessor() {
        let mut baseline = ProfileSnapshot::with_label("base");
        baseline.record_samples("a", &vec![100.0; 20]);
        baseline.record_samples("b", &vec![100.0; 20]);

        let mut current = ProfileSnapshot::with_label("cur");
        current.record_samples("a", &vec![120.0; 20]); // +20 %
        current.record_samples("b", &vec![200.0; 20]); // +100 %

        let report = ProfileComparator::default().compare(&baseline, &current);
        let worst = report.worst_regression().expect("should have worst");
        assert_eq!(worst.node_name, "b");
    }

    #[test]
    fn test_best_improvement_accessor() {
        let mut baseline = ProfileSnapshot::with_label("base");
        baseline.record_samples("a", &vec![100.0; 20]);
        baseline.record_samples("b", &vec![100.0; 20]);

        let mut current = ProfileSnapshot::with_label("cur");
        current.record_samples("a", &vec![90.0; 20]); // -10 %
        current.record_samples("b", &vec![50.0; 20]); // -50 %

        let report = ProfileComparator::default().compare(&baseline, &current);
        let best = report.best_improvement().expect("should have best");
        assert_eq!(best.node_name, "b");
    }

    #[test]
    fn test_node_stats_from_samples_percentiles() {
        let samples: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let stats = NodeStats::from_samples(&samples).expect("non-empty");
        // p95 of 1..=100 should be ≈ 95
        assert!(
            (stats.p95_ms - 95.05).abs() < 1.0,
            "p95 was {}",
            stats.p95_ms
        );
        assert_eq!(stats.min_ms as u64, 1);
        assert_eq!(stats.max_ms as u64, 100);
        assert_eq!(stats.count, 100);
    }

    #[test]
    fn test_node_stats_empty_returns_none() {
        assert!(NodeStats::from_samples(&[]).is_none());
        assert!(NodeStats::from_durations(&[]).is_none());
    }

    #[test]
    fn test_custom_regression_threshold() {
        let baseline = snapshot_fixed("step", 100.0, "base");
        let current = snapshot_fixed("step", 107.0, "cur"); // +7 %
                                                            // default threshold = 10 % → stable
        let default_cmp = ProfileComparator::default();
        assert!(!default_cmp.compare(&baseline, &current).has_regressions());
        // strict threshold = 5 % → regression
        let strict_cmp = ProfileComparator::new(5.0, 2.0);
        assert!(strict_cmp.compare(&baseline, &current).has_regressions());
    }

    #[test]
    fn test_summary_string_non_empty() {
        let baseline = snapshot_fixed("op", 100.0, "baseline");
        let current = snapshot_fixed("op", 130.0, "current");
        let report = ProfileComparator::default().compare(&baseline, &current);
        assert!(!report.summary.is_empty());
        assert!(report.summary.contains("Worst regression"));
    }

    #[test]
    fn test_record_durations_helper() {
        let mut snap = ProfileSnapshot::new();
        let durations: Vec<Duration> = (1..=10).map(|i| Duration::from_millis(i * 10)).collect();
        assert!(snap.record_durations("step", &durations));
        let stats = &snap.node_timings["step"];
        assert_eq!(stats.count, 10);
        assert!((stats.min_ms - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_zero_baseline_p95_does_not_panic() {
        let baseline = snapshot_fixed("zero", 0.0, "base");
        let current = snapshot_fixed("zero", 0.0, "cur");
        let report = ProfileComparator::default().compare(&baseline, &current);
        // Should complete without panic; stable or no-change
        assert!(!report.has_regressions());
    }
}
