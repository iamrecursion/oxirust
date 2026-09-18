#![allow(dead_code)]
//! Network bandwidth estimation for RTP/live streams.
//!
//! Provides a simple sliding-window bandwidth estimator with trend detection
//! and critical-alert thresholds.

/// A single bandwidth measurement taken at a point in time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RtpBandwidthSample {
    /// Bytes transferred during the observation window.
    pub bytes: u64,
    /// Duration of the observation window in milliseconds.
    pub duration_ms: u64,
}

impl RtpBandwidthSample {
    /// Creates a new sample.
    ///
    /// # Panics
    /// Does not panic; `duration_ms == 0` yields `0 kbps`.
    #[must_use]
    pub const fn new(bytes: u64, duration_ms: u64) -> Self {
        Self { bytes, duration_ms }
    }

    /// Converts the sample to kilobits per second.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn kbps(&self) -> f64 {
        if self.duration_ms == 0 {
            return 0.0;
        }
        (self.bytes as f64 * 8.0) / (self.duration_ms as f64)
    }
}

/// The direction of the bandwidth trend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BandwidthTrend {
    /// Bandwidth is broadly stable (< 10 % change between halves).
    Stable,
    /// Bandwidth is increasing.
    Rising,
    /// Bandwidth is decreasing.
    Falling,
}

/// A sliding-window bandwidth estimator.
///
/// Keeps up to `capacity` samples and exposes a weighted average.
#[derive(Debug)]
pub struct RtpBandwidthEstimator {
    samples: Vec<RtpBandwidthSample>,
    capacity: usize,
}

impl RtpBandwidthEstimator {
    /// Creates a new estimator with the given window capacity.
    ///
    /// # Panics
    /// Panics if `capacity` is 0.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "capacity must be > 0");
        Self {
            samples: Vec::with_capacity(capacity),
            capacity,
        }
    }

    /// Adds a new bandwidth sample, evicting the oldest when the window is full.
    pub fn add_sample(&mut self, sample: RtpBandwidthSample) {
        if self.samples.len() == self.capacity {
            self.samples.remove(0);
        }
        self.samples.push(sample);
    }

    /// Returns the number of samples currently held.
    #[must_use]
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Returns `true` when no samples have been recorded yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Estimates current bandwidth in kbps as a simple arithmetic mean of the window.
    #[must_use]
    pub fn estimate_kbps(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let total: f64 = self.samples.iter().map(RtpBandwidthSample::kbps).sum();
        total / self.samples.len() as f64
    }

    /// Detects the trend by comparing the mean of the first half vs the second half
    /// of the current window.
    #[must_use]
    pub fn trend(&self) -> BandwidthTrend {
        if self.samples.len() < 2 {
            return BandwidthTrend::Stable;
        }
        let mid = self.samples.len() / 2;
        let first_half = &self.samples[..mid];
        let second_half = &self.samples[mid..];

        let avg = |slice: &[RtpBandwidthSample]| -> f64 {
            slice.iter().map(RtpBandwidthSample::kbps).sum::<f64>() / slice.len() as f64
        };

        let old_avg = avg(first_half);
        let new_avg = avg(second_half);

        if old_avg == 0.0 {
            return BandwidthTrend::Stable;
        }

        let ratio = (new_avg - old_avg) / old_avg;
        if ratio > 0.10 {
            BandwidthTrend::Rising
        } else if ratio < -0.10 {
            BandwidthTrend::Falling
        } else {
            BandwidthTrend::Stable
        }
    }
}

/// An alert raised when bandwidth crosses a critical threshold.
#[derive(Debug, Clone, PartialEq)]
pub struct BandwidthAlert {
    /// Current estimated bandwidth in kbps.
    pub current_kbps: f64,
    /// The threshold that was breached (kbps).
    pub threshold_kbps: f64,
    /// Human-readable description of the alert condition.
    pub message: String,
}

impl BandwidthAlert {
    /// Creates a new `BandwidthAlert`.
    #[must_use]
    pub fn new(current_kbps: f64, threshold_kbps: f64, message: impl Into<String>) -> Self {
        Self {
            current_kbps,
            threshold_kbps,
            message: message.into(),
        }
    }

    /// Returns `true` if the current bandwidth is below the critical threshold.
    #[must_use]
    pub fn is_critical(&self) -> bool {
        self.current_kbps < self.threshold_kbps
    }
}

// ─── Unit tests ───────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    // 1. kbps – typical values
    #[test]
    fn test_sample_kbps_basic() {
        // 1000 bytes in 1000 ms = 8 kbps
        let s = RtpBandwidthSample::new(1000, 1000);
        assert!((s.kbps() - 8.0).abs() < 1e-9);
    }

    // 2. kbps – zero duration
    #[test]
    fn test_sample_kbps_zero_duration() {
        let s = RtpBandwidthSample::new(500, 0);
        assert_eq!(s.kbps(), 0.0);
    }

    // 3. kbps – zero bytes
    #[test]
    fn test_sample_kbps_zero_bytes() {
        let s = RtpBandwidthSample::new(0, 1000);
        assert_eq!(s.kbps(), 0.0);
    }

    // 4. estimator starts empty
    #[test]
    fn test_estimator_starts_empty() {
        let est = RtpBandwidthEstimator::new(5);
        assert!(est.is_empty());
        assert_eq!(est.estimate_kbps(), 0.0);
    }

    // 5. add_sample increases len
    #[test]
    fn test_estimator_add_sample() {
        let mut est = RtpBandwidthEstimator::new(5);
        est.add_sample(RtpBandwidthSample::new(1000, 1000));
        assert_eq!(est.len(), 1);
    }

    // 6. eviction when capacity exceeded
    #[test]
    fn test_estimator_eviction() {
        let mut est = RtpBandwidthEstimator::new(3);
        for i in 0u64..5 {
            est.add_sample(RtpBandwidthSample::new(i * 100, 1000));
        }
        assert_eq!(est.len(), 3);
    }

    // 7. estimate_kbps arithmetic mean
    #[test]
    fn test_estimate_kbps_mean() {
        let mut est = RtpBandwidthEstimator::new(2);
        est.add_sample(RtpBandwidthSample::new(1000, 1000)); // 8 kbps
        est.add_sample(RtpBandwidthSample::new(2000, 1000)); // 16 kbps
        assert!((est.estimate_kbps() - 12.0).abs() < 1e-6);
    }

    // 8. trend – stable
    #[test]
    fn test_trend_stable() {
        let mut est = RtpBandwidthEstimator::new(4);
        for _ in 0..4 {
            est.add_sample(RtpBandwidthSample::new(1000, 1000));
        }
        assert_eq!(est.trend(), BandwidthTrend::Stable);
    }

    // 9. trend – rising
    #[test]
    fn test_trend_rising() {
        let mut est = RtpBandwidthEstimator::new(4);
        // first half: 1000 bytes, second half: 3000 bytes (+200%)
        est.add_sample(RtpBandwidthSample::new(1000, 1000));
        est.add_sample(RtpBandwidthSample::new(1000, 1000));
        est.add_sample(RtpBandwidthSample::new(3000, 1000));
        est.add_sample(RtpBandwidthSample::new(3000, 1000));
        assert_eq!(est.trend(), BandwidthTrend::Rising);
    }

    // 10. trend – falling
    #[test]
    fn test_trend_falling() {
        let mut est = RtpBandwidthEstimator::new(4);
        est.add_sample(RtpBandwidthSample::new(3000, 1000));
        est.add_sample(RtpBandwidthSample::new(3000, 1000));
        est.add_sample(RtpBandwidthSample::new(1000, 1000));
        est.add_sample(RtpBandwidthSample::new(1000, 1000));
        assert_eq!(est.trend(), BandwidthTrend::Falling);
    }

    // 11. BandwidthAlert – is_critical true
    #[test]
    fn test_alert_is_critical() {
        let alert = BandwidthAlert::new(50.0, 100.0, "low bandwidth");
        assert!(alert.is_critical());
    }

    // 12. BandwidthAlert – is_critical false
    #[test]
    fn test_alert_not_critical() {
        let alert = BandwidthAlert::new(200.0, 100.0, "ok");
        assert!(!alert.is_critical());
    }

    // 13. trend – single sample
    #[test]
    fn test_trend_single_sample() {
        let mut est = RtpBandwidthEstimator::new(5);
        est.add_sample(RtpBandwidthSample::new(1000, 1000));
        assert_eq!(est.trend(), BandwidthTrend::Stable);
    }
}
