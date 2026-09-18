#![allow(dead_code)]
//! Loudness measurement history and rolling statistics.
//!
//! [`LoudnessHistory`] accumulates time-stamped integrated-loudness readings
//! and provides helpers such as [`rolling_average`](LoudnessHistory::rolling_average),
//! peak/minimum queries, and compliance-window checks.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

// ─────────────────────────────────────────────────────────────────────────────
// LoudnessMeasurement
// ─────────────────────────────────────────────────────────────────────────────

/// A single integrated-loudness reading captured at a point in time.
#[derive(Debug, Clone)]
pub struct LoudnessMeasurement {
    /// Integrated loudness in LUFS.
    pub lufs: f64,
    /// Loudness range (LRA) in LU at the time of measurement.
    pub lra_lu: f64,
    /// True-peak level in dBTP.
    pub true_peak_dbtp: f64,
    /// Wall-clock time at which this measurement was taken.
    pub captured_at: Instant,
}

impl LoudnessMeasurement {
    /// Create a measurement captured right now.
    pub fn new(lufs: f64, lra_lu: f64, true_peak_dbtp: f64) -> Self {
        Self {
            lufs,
            lra_lu,
            true_peak_dbtp,
            captured_at: Instant::now(),
        }
    }

    /// Age of this measurement.
    pub fn age(&self) -> Duration {
        self.captured_at.elapsed()
    }

    /// Returns `true` when integrated loudness is within `±tolerance_lu` of
    /// `target_lufs`.
    pub fn is_compliant(&self, target_lufs: f64, tolerance_lu: f64) -> bool {
        (self.lufs - target_lufs).abs() <= tolerance_lu
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LoudnessHistory
// ─────────────────────────────────────────────────────────────────────────────

/// Accumulates [`LoudnessMeasurement`] records and provides rolling statistics.
///
/// When `max_capacity` is set the oldest entries are evicted once the limit is
/// reached, keeping memory consumption bounded.
#[derive(Debug)]
pub struct LoudnessHistory {
    entries: VecDeque<LoudnessMeasurement>,
    /// Maximum number of entries to retain.  `0` means unlimited.
    max_capacity: usize,
}

impl LoudnessHistory {
    /// Create an unlimited-capacity history.
    pub fn new() -> Self {
        Self {
            entries: VecDeque::new(),
            max_capacity: 0,
        }
    }

    /// Create a history that retains at most `max_capacity` entries.
    pub fn with_capacity(max_capacity: usize) -> Self {
        Self {
            entries: VecDeque::with_capacity(max_capacity),
            max_capacity,
        }
    }

    /// Append a measurement.  Evicts the oldest entry when capacity is
    /// exceeded.
    pub fn push(&mut self, measurement: LoudnessMeasurement) {
        if self.max_capacity > 0 && self.entries.len() >= self.max_capacity {
            self.entries.pop_front();
        }
        self.entries.push_back(measurement);
    }

    /// Convenience wrapper that constructs and appends a measurement.
    pub fn record(&mut self, lufs: f64, lra_lu: f64, true_peak_dbtp: f64) {
        self.push(LoudnessMeasurement::new(lufs, lra_lu, true_peak_dbtp));
    }

    /// Number of stored measurements.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` when no measurements have been recorded.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Arithmetic mean of integrated loudness across all stored measurements.
    ///
    /// Returns `None` if the history is empty.
    pub fn average_lufs(&self) -> Option<f64> {
        if self.entries.is_empty() {
            return None;
        }
        let sum: f64 = self.entries.iter().map(|m| m.lufs).sum();
        #[allow(clippy::cast_precision_loss)]
        Some(sum / self.entries.len() as f64)
    }

    /// Rolling average of integrated loudness over the most recent `window`
    /// duration.
    ///
    /// Returns `None` if no measurements fall within the window.
    pub fn rolling_average(&self, window: Duration) -> Option<f64> {
        let now = Instant::now();
        let cutoff = now.checked_sub(window).unwrap_or(now);
        let recent: Vec<f64> = self
            .entries
            .iter()
            .filter(|m| m.captured_at >= cutoff)
            .map(|m| m.lufs)
            .collect();
        if recent.is_empty() {
            return None;
        }
        #[allow(clippy::cast_precision_loss)]
        Some(recent.iter().sum::<f64>() / recent.len() as f64)
    }

    /// Maximum integrated loudness recorded.
    pub fn peak_lufs(&self) -> Option<f64> {
        self.entries.iter().map(|m| m.lufs).reduce(f64::max)
    }

    /// Minimum integrated loudness recorded.
    pub fn min_lufs(&self) -> Option<f64> {
        self.entries.iter().map(|m| m.lufs).reduce(f64::min)
    }

    /// Maximum true-peak level recorded.
    pub fn peak_true_peak_dbtp(&self) -> Option<f64> {
        self.entries
            .iter()
            .map(|m| m.true_peak_dbtp)
            .reduce(f64::max)
    }

    /// Average loudness range across all measurements.
    pub fn average_lra(&self) -> Option<f64> {
        if self.entries.is_empty() {
            return None;
        }
        let sum: f64 = self.entries.iter().map(|m| m.lra_lu).sum();
        #[allow(clippy::cast_precision_loss)]
        Some(sum / self.entries.len() as f64)
    }

    /// Fraction (0.0–1.0) of measurements that are compliant with the given
    /// target and tolerance.
    pub fn compliance_rate(&self, target_lufs: f64, tolerance_lu: f64) -> f64 {
        if self.entries.is_empty() {
            return 0.0;
        }
        let compliant = self
            .entries
            .iter()
            .filter(|m| m.is_compliant(target_lufs, tolerance_lu))
            .count();
        #[allow(clippy::cast_precision_loss)]
        (compliant as f64 / self.entries.len() as f64)
    }

    /// Remove all entries older than `max_age`.
    pub fn evict_older_than(&mut self, max_age: Duration) {
        let now = Instant::now();
        self.entries.retain(|m| {
            now.checked_sub(m.captured_at.elapsed())
                .map_or(true, |_| m.age() <= max_age)
        });
    }

    /// Clear all recorded measurements.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Iterator over all stored measurements (oldest first).
    pub fn iter(&self) -> impl Iterator<Item = &LoudnessMeasurement> {
        self.entries.iter()
    }
}

impl Default for LoudnessHistory {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_measurement(lufs: f64) -> LoudnessMeasurement {
        LoudnessMeasurement::new(lufs, 8.0, -1.5)
    }

    #[test]
    fn test_history_starts_empty() {
        let h = LoudnessHistory::new();
        assert!(h.is_empty());
        assert_eq!(h.len(), 0);
    }

    #[test]
    fn test_push_increases_len() {
        let mut h = LoudnessHistory::new();
        h.push(sample_measurement(-23.0));
        h.push(sample_measurement(-22.0));
        assert_eq!(h.len(), 2);
    }

    #[test]
    fn test_record_convenience() {
        let mut h = LoudnessHistory::new();
        h.record(-23.0, 8.0, -1.0);
        assert_eq!(h.len(), 1);
    }

    #[test]
    fn test_average_lufs_single() {
        let mut h = LoudnessHistory::new();
        h.record(-20.0, 5.0, -1.0);
        assert!((h.average_lufs().expect("should succeed in test") - (-20.0)).abs() < 1e-9);
    }

    #[test]
    fn test_average_lufs_multiple() {
        let mut h = LoudnessHistory::new();
        h.record(-20.0, 5.0, -1.0);
        h.record(-24.0, 5.0, -1.0);
        let avg = h.average_lufs().expect("should succeed in test");
        assert!((avg - (-22.0)).abs() < 1e-9);
    }

    #[test]
    fn test_average_lufs_empty_is_none() {
        let h = LoudnessHistory::new();
        assert!(h.average_lufs().is_none());
    }

    #[test]
    fn test_peak_and_min_lufs() {
        let mut h = LoudnessHistory::new();
        h.record(-18.0, 5.0, -1.0);
        h.record(-25.0, 5.0, -1.0);
        h.record(-22.0, 5.0, -1.0);
        assert!((h.peak_lufs().expect("should succeed in test") - (-18.0)).abs() < 1e-9);
        assert!((h.min_lufs().expect("should succeed in test") - (-25.0)).abs() < 1e-9);
    }

    #[test]
    fn test_peak_true_peak() {
        let mut h = LoudnessHistory::new();
        h.record(-23.0, 8.0, -2.0);
        h.record(-23.0, 8.0, -0.5);
        assert!((h.peak_true_peak_dbtp().expect("should succeed in test") - (-0.5)).abs() < 1e-9);
    }

    #[test]
    fn test_average_lra() {
        let mut h = LoudnessHistory::new();
        h.record(-23.0, 6.0, -1.0);
        h.record(-23.0, 10.0, -1.0);
        let avg = h.average_lra().expect("should succeed in test");
        assert!((avg - 8.0).abs() < 1e-9);
    }

    #[test]
    fn test_compliance_rate_all_compliant() {
        let mut h = LoudnessHistory::new();
        h.record(-23.0, 8.0, -1.0);
        h.record(-22.5, 8.0, -1.0);
        assert!((h.compliance_rate(-23.0, 1.0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_compliance_rate_none_compliant() {
        let mut h = LoudnessHistory::new();
        h.record(-18.0, 8.0, -1.0); // 5 LU above -23 target
        assert!((h.compliance_rate(-23.0, 1.0) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_compliance_rate_empty_is_zero() {
        let h = LoudnessHistory::new();
        assert_eq!(h.compliance_rate(-23.0, 1.0), 0.0);
    }

    #[test]
    fn test_with_capacity_evicts_oldest() {
        let mut h = LoudnessHistory::with_capacity(3);
        h.record(-20.0, 5.0, -1.0);
        h.record(-21.0, 5.0, -1.0);
        h.record(-22.0, 5.0, -1.0);
        h.record(-23.0, 5.0, -1.0); // should evict -20.0
        assert_eq!(h.len(), 3);
        // The minimum should now be -23 (oldest -20 was evicted).
        assert!((h.min_lufs().expect("should succeed in test") - (-23.0)).abs() < 1e-9);
    }

    #[test]
    fn test_rolling_average_includes_all_recent() {
        let mut h = LoudnessHistory::new();
        h.record(-23.0, 8.0, -1.0);
        h.record(-21.0, 8.0, -1.0);
        // 1-hour window should include both just-recorded entries.
        let avg = h
            .rolling_average(Duration::from_secs(3600))
            .expect("should succeed in test");
        assert!((avg - (-22.0)).abs() < 1e-9);
    }

    #[test]
    fn test_rolling_average_no_entries_in_window_is_none() {
        let h = LoudnessHistory::new();
        let result = h.rolling_average(Duration::from_secs(60));
        assert!(result.is_none());
    }

    #[test]
    fn test_clear_empties_history() {
        let mut h = LoudnessHistory::new();
        h.record(-23.0, 8.0, -1.0);
        h.clear();
        assert!(h.is_empty());
    }

    #[test]
    fn test_measurement_is_compliant() {
        let m = LoudnessMeasurement::new(-23.0, 8.0, -1.0);
        assert!(m.is_compliant(-23.0, 1.0));
        assert!(!m.is_compliant(-20.0, 1.0));
    }

    #[test]
    fn test_iter_returns_all_entries() {
        let mut h = LoudnessHistory::new();
        h.record(-20.0, 5.0, -1.0);
        h.record(-21.0, 5.0, -1.0);
        assert_eq!(h.iter().count(), 2);
    }
}
