#![allow(dead_code)]
//! Bandwidth estimation and throttling for proxy file transfers.
//!
//! Provides tools to measure available bandwidth, estimate transfer times,
//! and apply rate limiting to proxy file transfers to avoid saturating
//! network links during production workflows.

use std::collections::VecDeque;
use std::time::Duration;

/// Units for bandwidth measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BandwidthUnit {
    /// Bits per second.
    Bps,
    /// Kilobits per second.
    Kbps,
    /// Megabits per second.
    Mbps,
    /// Gigabits per second.
    Gbps,
}

impl BandwidthUnit {
    /// Convert a value in this unit to bits per second.
    #[allow(clippy::cast_precision_loss)]
    pub fn to_bps(&self, value: f64) -> f64 {
        match self {
            Self::Bps => value,
            Self::Kbps => value * 1_000.0,
            Self::Mbps => value * 1_000_000.0,
            Self::Gbps => value * 1_000_000_000.0,
        }
    }

    /// Convert bits per second to this unit.
    #[allow(clippy::cast_precision_loss)]
    pub fn from_bps(&self, bps: f64) -> f64 {
        match self {
            Self::Bps => bps,
            Self::Kbps => bps / 1_000.0,
            Self::Mbps => bps / 1_000_000.0,
            Self::Gbps => bps / 1_000_000_000.0,
        }
    }
}

/// A single bandwidth measurement sample.
#[derive(Debug, Clone, Copy)]
pub struct BandwidthSample {
    /// Bytes transferred in this sample.
    pub bytes: u64,
    /// Duration of the sample.
    pub duration_ms: u64,
}

impl BandwidthSample {
    /// Create a new sample.
    pub fn new(bytes: u64, duration_ms: u64) -> Self {
        Self { bytes, duration_ms }
    }

    /// Compute bits per second for this sample.
    #[allow(clippy::cast_precision_loss)]
    pub fn bps(&self) -> f64 {
        if self.duration_ms == 0 {
            return 0.0;
        }
        (self.bytes as f64 * 8.0 * 1000.0) / self.duration_ms as f64
    }
}

/// Rolling bandwidth estimator based on recent samples.
pub struct BandwidthEstimator {
    /// Recent samples.
    samples: VecDeque<BandwidthSample>,
    /// Maximum number of samples to keep.
    max_samples: usize,
}

impl BandwidthEstimator {
    /// Create a new estimator with a given window size.
    pub fn new(max_samples: usize) -> Self {
        Self {
            samples: VecDeque::new(),
            max_samples: max_samples.max(1),
        }
    }

    /// Add a new sample.
    pub fn add_sample(&mut self, sample: BandwidthSample) {
        if self.samples.len() >= self.max_samples {
            self.samples.pop_front();
        }
        self.samples.push_back(sample);
    }

    /// Number of samples currently stored.
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Average bandwidth in bits per second.
    #[allow(clippy::cast_precision_loss)]
    pub fn average_bps(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let total_bytes: u64 = self.samples.iter().map(|s| s.bytes).sum();
        let total_ms: u64 = self.samples.iter().map(|s| s.duration_ms).sum();
        if total_ms == 0 {
            return 0.0;
        }
        (total_bytes as f64 * 8.0 * 1000.0) / total_ms as f64
    }

    /// Estimated time to transfer the given number of bytes.
    #[allow(clippy::cast_precision_loss)]
    pub fn estimate_transfer_time(&self, bytes: u64) -> Duration {
        let bps = self.average_bps();
        if bps <= 0.0 {
            return Duration::from_secs(u64::MAX);
        }
        let bits = bytes as f64 * 8.0;
        let seconds = bits / bps;
        Duration::from_secs_f64(seconds)
    }

    /// Peak bandwidth observed across all samples.
    pub fn peak_bps(&self) -> f64 {
        self.samples.iter().map(|s| s.bps()).fold(0.0_f64, f64::max)
    }

    /// Minimum bandwidth observed.
    pub fn min_bps(&self) -> f64 {
        self.samples
            .iter()
            .map(|s| s.bps())
            .fold(f64::MAX, f64::min)
    }

    /// Clear all samples.
    pub fn clear(&mut self) {
        self.samples.clear();
    }
}

/// Rate limiter for proxy transfers using a token bucket algorithm.
pub struct RateLimiter {
    /// Maximum rate in bytes per second.
    max_bytes_per_sec: u64,
    /// Token bucket capacity in bytes.
    bucket_capacity: u64,
    /// Current tokens available.
    tokens: u64,
    /// Last refill timestamp in milliseconds.
    last_refill_ms: u64,
}

impl RateLimiter {
    /// Create a new rate limiter.
    pub fn new(max_bytes_per_sec: u64) -> Self {
        let capacity = max_bytes_per_sec; // 1 second of burst
        Self {
            max_bytes_per_sec,
            bucket_capacity: capacity,
            tokens: capacity,
            last_refill_ms: 0,
        }
    }

    /// Set the bucket capacity (burst size).
    pub fn with_burst_size(mut self, bytes: u64) -> Self {
        self.bucket_capacity = bytes;
        self.tokens = self.tokens.min(bytes);
        self
    }

    /// Refill tokens based on elapsed time.
    #[allow(clippy::cast_precision_loss)]
    pub fn refill(&mut self, current_time_ms: u64) {
        if current_time_ms <= self.last_refill_ms {
            return;
        }
        let elapsed_ms = current_time_ms - self.last_refill_ms;
        let new_tokens = (self.max_bytes_per_sec as f64 * elapsed_ms as f64 / 1000.0) as u64;
        self.tokens = (self.tokens + new_tokens).min(self.bucket_capacity);
        self.last_refill_ms = current_time_ms;
    }

    /// Try to consume tokens for a transfer of the given size.
    /// Returns `true` if the transfer is allowed.
    pub fn try_consume(&mut self, bytes: u64) -> bool {
        if bytes <= self.tokens {
            self.tokens -= bytes;
            true
        } else {
            false
        }
    }

    /// Current available tokens.
    pub fn available_tokens(&self) -> u64 {
        self.tokens
    }

    /// Maximum rate in bytes per second.
    pub fn max_rate(&self) -> u64 {
        self.max_bytes_per_sec
    }
}

/// Transfer time estimate for a specific file.
#[derive(Debug, Clone)]
pub struct TransferEstimate {
    /// File size in bytes.
    pub file_size_bytes: u64,
    /// Estimated bandwidth in bits per second.
    pub estimated_bps: f64,
    /// Estimated transfer duration.
    pub estimated_duration: Duration,
    /// Confidence level from 0.0 to 1.0.
    pub confidence: f64,
}

impl TransferEstimate {
    /// Create a new transfer estimate.
    #[allow(clippy::cast_precision_loss)]
    pub fn new(file_size_bytes: u64, estimated_bps: f64, sample_count: usize) -> Self {
        let bits = file_size_bytes as f64 * 8.0;
        let seconds = if estimated_bps > 0.0 {
            bits / estimated_bps
        } else {
            f64::MAX
        };
        let confidence = (sample_count as f64 / 20.0).min(1.0);
        Self {
            file_size_bytes,
            estimated_bps,
            estimated_duration: Duration::from_secs_f64(seconds.min(u64::MAX as f64)),
            confidence,
        }
    }

    /// Whether the estimate has high confidence (>= 0.8).
    pub fn is_high_confidence(&self) -> bool {
        self.confidence >= 0.8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bandwidth_unit_to_bps() {
        assert!((BandwidthUnit::Kbps.to_bps(1.0) - 1000.0).abs() < 1e-9);
        assert!((BandwidthUnit::Mbps.to_bps(1.0) - 1_000_000.0).abs() < 1e-9);
        assert!((BandwidthUnit::Gbps.to_bps(1.0) - 1_000_000_000.0).abs() < 1e-9);
    }

    #[test]
    fn test_bandwidth_unit_from_bps() {
        assert!((BandwidthUnit::Mbps.from_bps(1_000_000.0) - 1.0).abs() < 1e-9);
        assert!((BandwidthUnit::Kbps.from_bps(1_000.0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_bandwidth_sample_bps() {
        let sample = BandwidthSample::new(1000, 1000);
        // 1000 bytes in 1000ms = 8000 bps
        assert!((sample.bps() - 8000.0).abs() < 1e-9);
    }

    #[test]
    fn test_bandwidth_sample_zero_duration() {
        let sample = BandwidthSample::new(1000, 0);
        assert_eq!(sample.bps(), 0.0);
    }

    #[test]
    fn test_estimator_empty() {
        let est = BandwidthEstimator::new(10);
        assert_eq!(est.average_bps(), 0.0);
        assert_eq!(est.sample_count(), 0);
    }

    #[test]
    fn test_estimator_single_sample() {
        let mut est = BandwidthEstimator::new(10);
        est.add_sample(BandwidthSample::new(1_000_000, 1000));
        // 1MB in 1s = 8Mbps
        assert!((est.average_bps() - 8_000_000.0).abs() < 1e-3);
    }

    #[test]
    fn test_estimator_window_eviction() {
        let mut est = BandwidthEstimator::new(3);
        for i in 0..5 {
            est.add_sample(BandwidthSample::new(100 * (i + 1), 1000));
        }
        assert_eq!(est.sample_count(), 3);
    }

    #[test]
    fn test_estimate_transfer_time() {
        let mut est = BandwidthEstimator::new(10);
        // 1MB/s = 8Mbps
        est.add_sample(BandwidthSample::new(1_000_000, 1000));
        let time = est.estimate_transfer_time(10_000_000);
        // 10MB at 1MB/s = 10 seconds
        assert!((time.as_secs_f64() - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_peak_and_min_bps() {
        let mut est = BandwidthEstimator::new(10);
        est.add_sample(BandwidthSample::new(100, 1000)); // 800 bps
        est.add_sample(BandwidthSample::new(1000, 1000)); // 8000 bps
        est.add_sample(BandwidthSample::new(500, 1000)); // 4000 bps
        assert!((est.peak_bps() - 8000.0).abs() < 1e-9);
        assert!((est.min_bps() - 800.0).abs() < 1e-9);
    }

    #[test]
    fn test_rate_limiter_consume() {
        let mut limiter = RateLimiter::new(1_000_000);
        assert!(limiter.try_consume(500_000));
        assert_eq!(limiter.available_tokens(), 500_000);
        assert!(limiter.try_consume(500_000));
        assert!(!limiter.try_consume(1));
    }

    #[test]
    fn test_rate_limiter_refill() {
        let mut limiter = RateLimiter::new(1000);
        limiter.try_consume(1000);
        assert_eq!(limiter.available_tokens(), 0);
        limiter.refill(500); // 0.5s -> 500 bytes
        assert_eq!(limiter.available_tokens(), 500);
    }

    #[test]
    fn test_rate_limiter_burst() {
        let limiter = RateLimiter::new(1000).with_burst_size(500);
        assert_eq!(limiter.available_tokens(), 500);
    }

    #[test]
    fn test_transfer_estimate_confidence() {
        let est = TransferEstimate::new(1_000_000, 8_000_000.0, 5);
        assert!(!est.is_high_confidence());
        let est2 = TransferEstimate::new(1_000_000, 8_000_000.0, 20);
        assert!(est2.is_high_confidence());
    }

    #[test]
    fn test_estimator_clear() {
        let mut est = BandwidthEstimator::new(10);
        est.add_sample(BandwidthSample::new(100, 100));
        est.clear();
        assert_eq!(est.sample_count(), 0);
        assert_eq!(est.average_bps(), 0.0);
    }
}
