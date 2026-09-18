#![allow(dead_code)]
//! Holdover quality estimation for clock synchronization.
//!
//! When a reference clock signal is lost, the local oscillator must free-run
//! ("holdover"). This module estimates and predicts how quickly the local clock
//! will drift during holdover, based on historical drift observations.

use std::collections::VecDeque;

/// Quality grade of holdover capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum HoldoverGrade {
    /// Excellent: drift < 100 ns/s, suitable for broadcast.
    Excellent,
    /// Good: drift < 1 us/s, suitable for professional AV.
    Good,
    /// Fair: drift < 10 us/s, acceptable for most applications.
    Fair,
    /// Poor: drift >= 10 us/s, may cause sync issues.
    Poor,
}

impl HoldoverGrade {
    /// Returns a human-readable label.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Excellent => "Excellent",
            Self::Good => "Good",
            Self::Fair => "Fair",
            Self::Poor => "Poor",
        }
    }

    /// Returns the maximum drift rate in nanoseconds per second for this grade.
    #[must_use]
    pub const fn max_drift_ns_per_sec(&self) -> u64 {
        match self {
            Self::Excellent => 100,
            Self::Good => 1_000,
            Self::Fair => 10_000,
            Self::Poor => u64::MAX,
        }
    }
}

/// A single drift observation sample.
#[derive(Debug, Clone, Copy)]
pub struct DriftSample {
    /// Elapsed time in seconds since reference was established.
    pub elapsed_secs: f64,
    /// Measured drift rate in nanoseconds per second.
    pub drift_ns_per_sec: f64,
}

/// Estimates holdover quality based on historical drift observations.
#[derive(Debug, Clone)]
pub struct HoldoverEstimator {
    /// Recent drift samples.
    samples: VecDeque<DriftSample>,
    /// Maximum number of samples to retain.
    max_samples: usize,
    /// Exponential moving average of drift rate (ns/s).
    ema_drift: f64,
    /// Smoothing factor for EMA (0..1).
    alpha: f64,
    /// Whether holdover mode is currently active.
    in_holdover: bool,
    /// Elapsed seconds since holdover started.
    holdover_elapsed: f64,
}

impl HoldoverEstimator {
    /// Creates a new holdover estimator.
    #[must_use]
    pub fn new(max_samples: usize) -> Self {
        Self {
            samples: VecDeque::with_capacity(max_samples),
            max_samples,
            ema_drift: 0.0,
            alpha: 0.1,
            in_holdover: false,
            holdover_elapsed: 0.0,
        }
    }

    /// Creates a new estimator with a custom smoothing factor.
    #[must_use]
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha.clamp(0.01, 0.99);
        self
    }

    /// Records a new drift observation.
    #[allow(clippy::cast_precision_loss)]
    pub fn add_sample(&mut self, elapsed_secs: f64, drift_ns_per_sec: f64) {
        let sample = DriftSample {
            elapsed_secs,
            drift_ns_per_sec,
        };
        if self.samples.len() >= self.max_samples {
            self.samples.pop_front();
        }
        self.samples.push_back(sample);

        // Update EMA
        if self.samples.len() == 1 {
            self.ema_drift = drift_ns_per_sec.abs();
        } else {
            self.ema_drift =
                self.alpha * drift_ns_per_sec.abs() + (1.0 - self.alpha) * self.ema_drift;
        }
    }

    /// Returns the number of samples currently stored.
    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Returns the current exponential moving average of drift rate.
    #[must_use]
    pub fn ema_drift_rate(&self) -> f64 {
        self.ema_drift
    }

    /// Returns the mean drift rate across all samples.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn mean_drift_rate(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.samples.iter().map(|s| s.drift_ns_per_sec.abs()).sum();
        sum / self.samples.len() as f64
    }

    /// Returns the maximum observed drift rate.
    #[must_use]
    pub fn max_drift_rate(&self) -> f64 {
        self.samples
            .iter()
            .map(|s| s.drift_ns_per_sec.abs())
            .fold(0.0_f64, f64::max)
    }

    /// Returns the standard deviation of drift rates.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn drift_stddev(&self) -> f64 {
        if self.samples.len() < 2 {
            return 0.0;
        }
        let mean = self.mean_drift_rate();
        let variance: f64 = self
            .samples
            .iter()
            .map(|s| {
                let diff = s.drift_ns_per_sec.abs() - mean;
                diff * diff
            })
            .sum::<f64>()
            / (self.samples.len() - 1) as f64;
        variance.sqrt()
    }

    /// Estimates the holdover grade based on observed drift.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn estimate_grade(&self) -> HoldoverGrade {
        let drift = self.ema_drift;
        let drift_abs = drift.abs() as u64;
        if drift_abs < HoldoverGrade::Excellent.max_drift_ns_per_sec() {
            HoldoverGrade::Excellent
        } else if drift_abs < HoldoverGrade::Good.max_drift_ns_per_sec() {
            HoldoverGrade::Good
        } else if drift_abs < HoldoverGrade::Fair.max_drift_ns_per_sec() {
            HoldoverGrade::Fair
        } else {
            HoldoverGrade::Poor
        }
    }

    /// Predicts the accumulated drift (ns) after `seconds` of holdover.
    #[must_use]
    pub fn predict_drift_ns(&self, seconds: f64) -> f64 {
        self.ema_drift * seconds
    }

    /// Predicts how many seconds until the accumulated drift exceeds `threshold_ns`.
    #[must_use]
    pub fn time_to_threshold(&self, threshold_ns: f64) -> Option<f64> {
        if self.ema_drift <= 0.0 {
            return None;
        }
        Some(threshold_ns / self.ema_drift)
    }

    /// Enters holdover mode.
    pub fn enter_holdover(&mut self) {
        self.in_holdover = true;
        self.holdover_elapsed = 0.0;
    }

    /// Exits holdover mode.
    pub fn exit_holdover(&mut self) {
        self.in_holdover = false;
    }

    /// Advances holdover time by the given number of seconds.
    pub fn advance_holdover(&mut self, secs: f64) {
        if self.in_holdover {
            self.holdover_elapsed += secs;
        }
    }

    /// Returns `true` if currently in holdover mode.
    #[must_use]
    pub fn is_in_holdover(&self) -> bool {
        self.in_holdover
    }

    /// Returns the elapsed holdover time in seconds.
    #[must_use]
    pub fn holdover_elapsed_secs(&self) -> f64 {
        self.holdover_elapsed
    }

    /// Returns the estimated accumulated drift since holdover began.
    #[must_use]
    pub fn estimated_holdover_drift_ns(&self) -> f64 {
        self.predict_drift_ns(self.holdover_elapsed)
    }

    /// Clears all samples and resets state.
    pub fn reset(&mut self) {
        self.samples.clear();
        self.ema_drift = 0.0;
        self.in_holdover = false;
        self.holdover_elapsed = 0.0;
    }
}

impl Default for HoldoverEstimator {
    fn default() -> Self {
        Self::new(256)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_holdover_grade_labels() {
        assert_eq!(HoldoverGrade::Excellent.label(), "Excellent");
        assert_eq!(HoldoverGrade::Poor.label(), "Poor");
    }

    #[test]
    fn test_holdover_grade_ordering() {
        assert!(HoldoverGrade::Excellent < HoldoverGrade::Good);
        assert!(HoldoverGrade::Good < HoldoverGrade::Fair);
        assert!(HoldoverGrade::Fair < HoldoverGrade::Poor);
    }

    #[test]
    fn test_new_estimator() {
        let est = HoldoverEstimator::new(100);
        assert_eq!(est.sample_count(), 0);
        assert!(!est.is_in_holdover());
        assert!((est.ema_drift_rate() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_add_samples() {
        let mut est = HoldoverEstimator::new(10);
        est.add_sample(1.0, 50.0);
        assert_eq!(est.sample_count(), 1);
        est.add_sample(2.0, 60.0);
        assert_eq!(est.sample_count(), 2);
    }

    #[test]
    fn test_max_samples_eviction() {
        let mut est = HoldoverEstimator::new(3);
        est.add_sample(1.0, 10.0);
        est.add_sample(2.0, 20.0);
        est.add_sample(3.0, 30.0);
        est.add_sample(4.0, 40.0);
        assert_eq!(est.sample_count(), 3);
    }

    #[test]
    fn test_mean_drift_rate() {
        let mut est = HoldoverEstimator::new(10);
        est.add_sample(1.0, 100.0);
        est.add_sample(2.0, 200.0);
        est.add_sample(3.0, 300.0);
        let mean = est.mean_drift_rate();
        assert!((mean - 200.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_max_drift_rate() {
        let mut est = HoldoverEstimator::new(10);
        est.add_sample(1.0, 50.0);
        est.add_sample(2.0, 150.0);
        est.add_sample(3.0, 75.0);
        assert!((est.max_drift_rate() - 150.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_drift_stddev() {
        let mut est = HoldoverEstimator::new(10);
        est.add_sample(1.0, 100.0);
        est.add_sample(2.0, 100.0);
        est.add_sample(3.0, 100.0);
        assert!(est.drift_stddev() < f64::EPSILON);
    }

    #[test]
    fn test_estimate_grade_excellent() {
        let mut est = HoldoverEstimator::new(10);
        est.add_sample(1.0, 50.0); // 50 ns/s drift
        assert_eq!(est.estimate_grade(), HoldoverGrade::Excellent);
    }

    #[test]
    fn test_predict_drift() {
        let mut est = HoldoverEstimator::new(10);
        est.add_sample(1.0, 100.0);
        // EMA should be 100 after first sample
        let predicted = est.predict_drift_ns(10.0);
        assert!((predicted - 1000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_time_to_threshold() {
        let mut est = HoldoverEstimator::new(10);
        est.add_sample(1.0, 200.0);
        let t = est
            .time_to_threshold(1000.0)
            .expect("should succeed in test");
        assert!((t - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_holdover_lifecycle() {
        let mut est = HoldoverEstimator::new(10);
        est.add_sample(1.0, 100.0);
        assert!(!est.is_in_holdover());

        est.enter_holdover();
        assert!(est.is_in_holdover());
        assert!((est.holdover_elapsed_secs() - 0.0).abs() < f64::EPSILON);

        est.advance_holdover(5.0);
        assert!((est.holdover_elapsed_secs() - 5.0).abs() < f64::EPSILON);
        assert!((est.estimated_holdover_drift_ns() - 500.0).abs() < f64::EPSILON);

        est.exit_holdover();
        assert!(!est.is_in_holdover());
    }

    #[test]
    fn test_reset() {
        let mut est = HoldoverEstimator::new(10);
        est.add_sample(1.0, 100.0);
        est.enter_holdover();
        est.advance_holdover(10.0);
        est.reset();
        assert_eq!(est.sample_count(), 0);
        assert!(!est.is_in_holdover());
        assert!((est.ema_drift_rate() - 0.0).abs() < f64::EPSILON);
    }
}
