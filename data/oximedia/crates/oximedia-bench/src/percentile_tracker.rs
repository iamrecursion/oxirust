#![allow(dead_code)]
//! Percentile-based performance tracking for benchmarks.
//!
//! Tracks iteration timings and computes standard percentiles (p50, p90, p95,
//! p99, etc.) using a sorted-insertion approach. This is useful for detecting
//! tail latencies and jitter in encoding / decoding benchmarks.

use std::fmt;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Which percentile to query.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Percentile {
    /// 50th percentile (median).
    P50,
    /// 75th percentile.
    P75,
    /// 90th percentile.
    P90,
    /// 95th percentile.
    P95,
    /// 99th percentile.
    P99,
    /// 99.9th percentile.
    P999,
    /// Custom percentile (0..100).
    Custom(f64),
}

impl Percentile {
    /// Return the numeric value (0..100).
    pub fn value(&self) -> f64 {
        match self {
            Self::P50 => 50.0,
            Self::P75 => 75.0,
            Self::P90 => 90.0,
            Self::P95 => 95.0,
            Self::P99 => 99.0,
            Self::P999 => 99.9,
            Self::Custom(v) => *v,
        }
    }
}

impl fmt::Display for Percentile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::P50 => write!(f, "p50"),
            Self::P75 => write!(f, "p75"),
            Self::P90 => write!(f, "p90"),
            Self::P95 => write!(f, "p95"),
            Self::P99 => write!(f, "p99"),
            Self::P999 => write!(f, "p99.9"),
            Self::Custom(v) => write!(f, "p{v}"),
        }
    }
}

/// Summary of all standard percentiles.
#[derive(Debug, Clone)]
pub struct PercentileSummary {
    /// Number of data points.
    pub count: usize,
    /// Minimum value.
    pub min: f64,
    /// Maximum value.
    pub max: f64,
    /// Mean value.
    pub mean: f64,
    /// 50th percentile.
    pub p50: f64,
    /// 75th percentile.
    pub p75: f64,
    /// 90th percentile.
    pub p90: f64,
    /// 95th percentile.
    pub p95: f64,
    /// 99th percentile.
    pub p99: f64,
    /// 99.9th percentile.
    pub p999: f64,
}

impl fmt::Display for PercentileSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "count={} min={:.1} mean={:.1} p50={:.1} p90={:.1} p99={:.1} max={:.1}",
            self.count, self.min, self.mean, self.p50, self.p90, self.p99, self.max
        )
    }
}

// ---------------------------------------------------------------------------
// Tracker
// ---------------------------------------------------------------------------

/// Collects values and computes percentiles.
#[derive(Debug, Clone)]
pub struct PercentileTracker {
    /// Stored values (kept sorted).
    values: Vec<f64>,
    /// Running sum for mean calculation.
    sum: f64,
}

impl Default for PercentileTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl PercentileTracker {
    /// Create an empty tracker.
    pub fn new() -> Self {
        Self {
            values: Vec::new(),
            sum: 0.0,
        }
    }

    /// Create a tracker pre-allocated for `capacity` values.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            values: Vec::with_capacity(capacity),
            sum: 0.0,
        }
    }

    /// Record a new value.
    pub fn record(&mut self, value: f64) {
        let pos = self.values.partition_point(|&v| v < value);
        self.values.insert(pos, value);
        self.sum += value;
    }

    /// Record many values at once.
    pub fn record_many(&mut self, values: &[f64]) {
        for &v in values {
            self.record(v);
        }
    }

    /// Number of recorded values.
    pub fn count(&self) -> usize {
        self.values.len()
    }

    /// Whether the tracker is empty.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Reset the tracker.
    pub fn reset(&mut self) {
        self.values.clear();
        self.sum = 0.0;
    }

    /// Mean of all recorded values.
    #[allow(clippy::cast_precision_loss)]
    pub fn mean(&self) -> f64 {
        if self.values.is_empty() {
            return 0.0;
        }
        self.sum / self.values.len() as f64
    }

    /// Minimum value.
    pub fn min(&self) -> f64 {
        self.values.first().copied().unwrap_or(0.0)
    }

    /// Maximum value.
    pub fn max(&self) -> f64 {
        self.values.last().copied().unwrap_or(0.0)
    }

    /// Query a specific percentile.
    #[allow(clippy::cast_precision_loss)]
    pub fn percentile(&self, p: Percentile) -> f64 {
        if self.values.is_empty() {
            return 0.0;
        }
        let pct = p.value().clamp(0.0, 100.0) / 100.0;
        let idx_f = pct * (self.values.len() as f64 - 1.0);
        let lo = idx_f.floor() as usize;
        let hi = idx_f.ceil().min((self.values.len() - 1) as f64) as usize;
        let frac = idx_f - lo as f64;
        self.values[lo] * (1.0 - frac) + self.values[hi] * frac
    }

    /// Compute a full percentile summary.
    pub fn summary(&self) -> PercentileSummary {
        PercentileSummary {
            count: self.count(),
            min: self.min(),
            max: self.max(),
            mean: self.mean(),
            p50: self.percentile(Percentile::P50),
            p75: self.percentile(Percentile::P75),
            p90: self.percentile(Percentile::P90),
            p95: self.percentile(Percentile::P95),
            p99: self.percentile(Percentile::P99),
            p999: self.percentile(Percentile::P999),
        }
    }

    /// Return the sorted values.
    pub fn sorted_values(&self) -> &[f64] {
        &self.values
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_tracker() {
        let t = PercentileTracker::new();
        assert!(t.is_empty());
        assert_eq!(t.count(), 0);
        assert!((t.mean() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_single_value() {
        let mut t = PercentileTracker::new();
        t.record(42.0);
        assert_eq!(t.count(), 1);
        assert!((t.mean() - 42.0).abs() < f64::EPSILON);
        assert!((t.min() - 42.0).abs() < f64::EPSILON);
        assert!((t.max() - 42.0).abs() < f64::EPSILON);
        assert!((t.percentile(Percentile::P50) - 42.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_sorted_insertion() {
        let mut t = PercentileTracker::new();
        t.record(30.0);
        t.record(10.0);
        t.record(20.0);
        assert_eq!(t.sorted_values(), &[10.0, 20.0, 30.0]);
    }

    #[test]
    fn test_percentile_median() {
        let mut t = PercentileTracker::new();
        for v in [1.0, 2.0, 3.0, 4.0, 5.0] {
            t.record(v);
        }
        let p50 = t.percentile(Percentile::P50);
        assert!((p50 - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_percentile_p99() {
        let mut t = PercentileTracker::new();
        for i in 1..=100 {
            t.record(i as f64);
        }
        let p99 = t.percentile(Percentile::P99);
        // p99 of 1..100 should be close to 99.01
        assert!(p99 >= 98.0);
        assert!(p99 <= 100.0);
    }

    #[test]
    fn test_record_many() {
        let mut t = PercentileTracker::new();
        t.record_many(&[5.0, 1.0, 3.0, 2.0, 4.0]);
        assert_eq!(t.count(), 5);
        assert_eq!(t.sorted_values(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_mean() {
        let mut t = PercentileTracker::new();
        t.record_many(&[10.0, 20.0, 30.0]);
        assert!((t.mean() - 20.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_summary() {
        let mut t = PercentileTracker::new();
        for i in 1..=100 {
            t.record(i as f64);
        }
        let s = t.summary();
        assert_eq!(s.count, 100);
        assert!((s.min - 1.0).abs() < f64::EPSILON);
        assert!((s.max - 100.0).abs() < f64::EPSILON);
        assert!((s.mean - 50.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_reset() {
        let mut t = PercentileTracker::new();
        t.record(1.0);
        t.record(2.0);
        t.reset();
        assert!(t.is_empty());
        assert_eq!(t.count(), 0);
    }

    #[test]
    fn test_percentile_display() {
        assert_eq!(format!("{}", Percentile::P50), "p50");
        assert_eq!(format!("{}", Percentile::P999), "p99.9");
        assert_eq!(format!("{}", Percentile::Custom(99.5)), "p99.5");
    }

    #[test]
    fn test_percentile_value() {
        assert!((Percentile::P50.value() - 50.0).abs() < f64::EPSILON);
        assert!((Percentile::P99.value() - 99.0).abs() < f64::EPSILON);
        assert!((Percentile::Custom(42.0).value() - 42.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_summary_display() {
        let mut t = PercentileTracker::new();
        t.record_many(&[1.0, 2.0, 3.0]);
        let s = t.summary();
        let display = format!("{s}");
        assert!(display.contains("count=3"));
    }

    #[test]
    fn test_with_capacity() {
        let t = PercentileTracker::with_capacity(100);
        assert!(t.is_empty());
    }
}
