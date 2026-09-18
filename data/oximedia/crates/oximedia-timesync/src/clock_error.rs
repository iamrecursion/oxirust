//! Clock error tracking and classification for time synchronization.
//!
//! This module provides types for representing, classifying, and tracking
//! clock errors over time, enabling adaptive synchronization strategies.

#![allow(dead_code)]

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Classification of clock error types observed during synchronization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ClockErrorType {
    /// Offset error: the clock is ahead or behind the reference.
    Offset,
    /// Frequency error: the clock runs fast or slow.
    FrequencyDrift,
    /// Jitter: short-term variation in offset.
    Jitter,
    /// Wander: long-term variation in frequency offset.
    Wander,
    /// Step: a sudden discontinuity in the clock value.
    Step,
    /// Holdover degradation: error accumulated while running without reference.
    Holdover,
    /// Phase noise from the oscillator.
    PhaseNoise,
}

impl ClockErrorType {
    /// Returns a human-readable label for the error type.
    pub fn label(self) -> &'static str {
        match self {
            Self::Offset => "offset",
            Self::FrequencyDrift => "frequency_drift",
            Self::Jitter => "jitter",
            Self::Wander => "wander",
            Self::Step => "step",
            Self::Holdover => "holdover",
            Self::PhaseNoise => "phase_noise",
        }
    }

    /// Returns whether this error type is generally correctable via slewing.
    pub fn is_slew_correctable(self) -> bool {
        matches!(
            self,
            Self::Offset | Self::FrequencyDrift | Self::Jitter | Self::Wander
        )
    }
}

/// A single clock error measurement.
#[derive(Debug, Clone)]
pub struct ClockError {
    /// The type of error.
    pub error_type: ClockErrorType,
    /// Error magnitude in nanoseconds (signed: positive = ahead, negative = behind).
    pub magnitude_ns: i64,
    /// When this error was measured.
    pub timestamp: Instant,
    /// Optional source identifier (e.g., "PTP", "NTP").
    pub source: Option<String>,
}

impl ClockError {
    /// Create a new clock error measurement.
    pub fn new(error_type: ClockErrorType, magnitude_ns: i64) -> Self {
        Self {
            error_type,
            magnitude_ns,
            timestamp: Instant::now(),
            source: None,
        }
    }

    /// Attach a source label to this error.
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Returns the absolute magnitude in nanoseconds.
    pub fn abs_magnitude_ns(&self) -> u64 {
        self.magnitude_ns.unsigned_abs()
    }

    /// Returns true if the error exceeds the given threshold (nanoseconds).
    pub fn exceeds_threshold(&self, threshold_ns: u64) -> bool {
        self.abs_magnitude_ns() > threshold_ns
    }

    /// Returns the age of this measurement.
    pub fn age(&self) -> Duration {
        self.timestamp.elapsed()
    }
}

/// Statistics summary for a window of clock errors.
#[derive(Debug, Clone, Default)]
pub struct ClockErrorStats {
    /// Number of samples.
    pub count: usize,
    /// Mean error in nanoseconds.
    pub mean_ns: f64,
    /// Root mean square error in nanoseconds.
    pub rms_ns: f64,
    /// Maximum absolute error observed in nanoseconds.
    pub max_abs_ns: u64,
    /// Standard deviation in nanoseconds.
    pub std_dev_ns: f64,
}

/// Tracks clock errors over a sliding window and provides statistics.
#[derive(Debug)]
pub struct ClockErrorTracker {
    /// Sliding window of recent errors.
    window: VecDeque<ClockError>,
    /// Maximum number of samples to retain.
    capacity: usize,
    /// Threshold above which an error is flagged as excessive (nanoseconds).
    excessive_threshold_ns: u64,
    /// Count of how many samples exceeded the threshold.
    excessive_count: usize,
}

impl ClockErrorTracker {
    /// Create a new tracker with the given window capacity.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Maximum number of error samples to retain in the window.
    /// * `excessive_threshold_ns` - Errors above this value (in ns) are counted as excessive.
    pub fn new(capacity: usize, excessive_threshold_ns: u64) -> Self {
        Self {
            window: VecDeque::with_capacity(capacity),
            capacity,
            excessive_threshold_ns,
            excessive_count: 0,
        }
    }

    /// Record a new clock error measurement.
    pub fn record(&mut self, error: ClockError) {
        if error.exceeds_threshold(self.excessive_threshold_ns) {
            self.excessive_count += 1;
        }
        if self.window.len() == self.capacity {
            if let Some(evicted) = self.window.pop_front() {
                if evicted.exceeds_threshold(self.excessive_threshold_ns) {
                    self.excessive_count = self.excessive_count.saturating_sub(1);
                }
            }
        }
        self.window.push_back(error);
    }

    /// Record a simple offset error (convenience method).
    pub fn record_offset(&mut self, magnitude_ns: i64) {
        self.record(ClockError::new(ClockErrorType::Offset, magnitude_ns));
    }

    /// Returns the maximum absolute error seen in the current window (nanoseconds).
    pub fn max_error(&self) -> u64 {
        self.window
            .iter()
            .map(ClockError::abs_magnitude_ns)
            .max()
            .unwrap_or(0)
    }

    /// Returns the most recent error magnitude, or `None` if the window is empty.
    pub fn latest_error(&self) -> Option<i64> {
        self.window.back().map(|e| e.magnitude_ns)
    }

    /// Returns the number of samples in the current window.
    pub fn len(&self) -> usize {
        self.window.len()
    }

    /// Returns true if no samples have been recorded yet.
    pub fn is_empty(&self) -> bool {
        self.window.is_empty()
    }

    /// Returns the number of samples that exceeded the excessive threshold.
    pub fn excessive_count(&self) -> usize {
        self.excessive_count
    }

    /// Returns whether the clock is currently considered stable (no excessive errors in window).
    pub fn is_stable(&self) -> bool {
        self.excessive_count == 0 && !self.window.is_empty()
    }

    /// Compute and return statistics for all errors currently in the window.
    pub fn stats(&self) -> ClockErrorStats {
        if self.window.is_empty() {
            return ClockErrorStats::default();
        }

        let count = self.window.len();
        #[allow(clippy::cast_precision_loss)]
        let n = count as f64;

        let sum: i64 = self.window.iter().map(|e| e.magnitude_ns).sum();
        #[allow(clippy::cast_precision_loss)]
        let mean_ns = sum as f64 / n;

        #[allow(clippy::cast_precision_loss)]
        let sum_sq: f64 = self
            .window
            .iter()
            .map(|e| {
                let v = e.magnitude_ns as f64;
                v * v
            })
            .sum();
        let rms_ns = (sum_sq / n).sqrt();

        #[allow(clippy::cast_precision_loss)]
        let variance: f64 = self
            .window
            .iter()
            .map(|e| {
                let diff = e.magnitude_ns as f64 - mean_ns;
                diff * diff
            })
            .sum::<f64>()
            / n;

        let max_abs_ns = self.max_error();

        ClockErrorStats {
            count,
            mean_ns,
            rms_ns,
            max_abs_ns,
            std_dev_ns: variance.sqrt(),
        }
    }

    /// Drain all entries older than `max_age`.
    pub fn prune_old(&mut self, max_age: Duration) {
        while let Some(front) = self.window.front() {
            if front.age() > max_age {
                if let Some(evicted) = self.window.pop_front() {
                    if evicted.exceeds_threshold(self.excessive_threshold_ns) {
                        self.excessive_count = self.excessive_count.saturating_sub(1);
                    }
                }
            } else {
                break;
            }
        }
    }

    /// Reset the tracker, clearing all samples.
    pub fn reset(&mut self) {
        self.window.clear();
        self.excessive_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_type_label() {
        assert_eq!(ClockErrorType::Offset.label(), "offset");
        assert_eq!(ClockErrorType::FrequencyDrift.label(), "frequency_drift");
        assert_eq!(ClockErrorType::Jitter.label(), "jitter");
        assert_eq!(ClockErrorType::Wander.label(), "wander");
        assert_eq!(ClockErrorType::Step.label(), "step");
        assert_eq!(ClockErrorType::Holdover.label(), "holdover");
        assert_eq!(ClockErrorType::PhaseNoise.label(), "phase_noise");
    }

    #[test]
    fn test_error_type_slew_correctable() {
        assert!(ClockErrorType::Offset.is_slew_correctable());
        assert!(ClockErrorType::FrequencyDrift.is_slew_correctable());
        assert!(ClockErrorType::Jitter.is_slew_correctable());
        assert!(!ClockErrorType::Step.is_slew_correctable());
        assert!(!ClockErrorType::Holdover.is_slew_correctable());
    }

    #[test]
    fn test_clock_error_new() {
        let err = ClockError::new(ClockErrorType::Offset, -500);
        assert_eq!(err.error_type, ClockErrorType::Offset);
        assert_eq!(err.magnitude_ns, -500);
        assert_eq!(err.abs_magnitude_ns(), 500);
        assert!(err.source.is_none());
    }

    #[test]
    fn test_clock_error_with_source() {
        let err = ClockError::new(ClockErrorType::Jitter, 100).with_source("PTP");
        assert_eq!(err.source.as_deref(), Some("PTP"));
    }

    #[test]
    fn test_clock_error_exceeds_threshold() {
        let err = ClockError::new(ClockErrorType::Offset, 1001);
        assert!(err.exceeds_threshold(1000));
        assert!(!err.exceeds_threshold(2000));
    }

    #[test]
    fn test_tracker_empty() {
        let tracker = ClockErrorTracker::new(10, 1000);
        assert!(tracker.is_empty());
        assert_eq!(tracker.max_error(), 0);
        assert!(tracker.latest_error().is_none());
    }

    #[test]
    fn test_tracker_record_and_max_error() {
        let mut tracker = ClockErrorTracker::new(10, 10_000);
        tracker.record_offset(200);
        tracker.record_offset(-500);
        tracker.record_offset(300);
        assert_eq!(tracker.max_error(), 500);
        assert_eq!(tracker.len(), 3);
    }

    #[test]
    fn test_tracker_capacity_eviction() {
        let mut tracker = ClockErrorTracker::new(3, 10_000);
        tracker.record_offset(100);
        tracker.record_offset(200);
        tracker.record_offset(300);
        tracker.record_offset(400);
        // Oldest (100) should be evicted
        assert_eq!(tracker.len(), 3);
        assert_eq!(tracker.max_error(), 400);
    }

    #[test]
    fn test_tracker_excessive_count() {
        let mut tracker = ClockErrorTracker::new(10, 1000);
        tracker.record_offset(500); // below threshold
        tracker.record_offset(2000); // above threshold
        tracker.record_offset(-1500); // above threshold
        assert_eq!(tracker.excessive_count(), 2);
        assert!(!tracker.is_stable());
    }

    #[test]
    fn test_tracker_is_stable() {
        let mut tracker = ClockErrorTracker::new(10, 10_000);
        tracker.record_offset(100);
        tracker.record_offset(-50);
        assert!(tracker.is_stable());
    }

    #[test]
    fn test_tracker_stats_basic() {
        let mut tracker = ClockErrorTracker::new(10, 10_000);
        tracker.record_offset(100);
        tracker.record_offset(-100);
        let stats = tracker.stats();
        assert_eq!(stats.count, 2);
        assert!((stats.mean_ns - 0.0).abs() < 1e-9);
        assert!(stats.rms_ns > 0.0);
        assert_eq!(stats.max_abs_ns, 100);
    }

    #[test]
    fn test_tracker_stats_empty() {
        let tracker = ClockErrorTracker::new(10, 10_000);
        let stats = tracker.stats();
        assert_eq!(stats.count, 0);
        assert_eq!(stats.max_abs_ns, 0);
    }

    #[test]
    fn test_tracker_reset() {
        let mut tracker = ClockErrorTracker::new(10, 1000);
        tracker.record_offset(5000);
        assert_eq!(tracker.excessive_count(), 1);
        tracker.reset();
        assert!(tracker.is_empty());
        assert_eq!(tracker.excessive_count(), 0);
    }

    #[test]
    fn test_tracker_latest_error() {
        let mut tracker = ClockErrorTracker::new(10, 10_000);
        tracker.record_offset(111);
        tracker.record_offset(222);
        assert_eq!(tracker.latest_error(), Some(222));
    }

    #[test]
    fn test_excessive_eviction_decrements_count() {
        let mut tracker = ClockErrorTracker::new(2, 1000);
        tracker.record_offset(2000); // excessive, enters window
        tracker.record_offset(3000); // excessive, enters window
        assert_eq!(tracker.excessive_count(), 2);
        tracker.record_offset(50); // causes eviction of first excessive sample
        assert_eq!(tracker.excessive_count(), 1);
    }
}
