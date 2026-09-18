#![allow(dead_code)]
//! Bandwidth estimation utilities for adaptive streaming.
//!
//! This module provides tools for estimating required bandwidth, analyzing
//! segment delivery performance, computing buffer levels, and recommending
//! quality switches for adaptive bitrate streaming.

use std::collections::VecDeque;
use std::time::Duration;

/// A single bandwidth sample measurement.
#[derive(Debug, Clone, Copy)]
pub struct BandwidthSample {
    /// Measured bandwidth in bits per second.
    pub bandwidth_bps: u64,
    /// Timestamp when this sample was taken.
    pub timestamp_ms: u64,
    /// Size of the segment downloaded (bytes).
    pub segment_bytes: u64,
    /// Time it took to download (milliseconds).
    pub download_time_ms: u64,
}

impl BandwidthSample {
    /// Create a new bandwidth sample from download metrics.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn from_download(segment_bytes: u64, download_time_ms: u64, timestamp_ms: u64) -> Self {
        let bandwidth_bps = (segment_bytes * 8 * 1000)
            .checked_div(download_time_ms)
            .unwrap_or(0);
        Self {
            bandwidth_bps,
            timestamp_ms,
            segment_bytes,
            download_time_ms,
        }
    }
}

/// Bandwidth estimator using a sliding window of samples.
pub struct BandwidthEstimator {
    /// Sliding window of recent bandwidth samples.
    samples: VecDeque<BandwidthSample>,
    /// Maximum number of samples to keep.
    max_samples: usize,
    /// Maximum age of samples in milliseconds.
    max_age_ms: u64,
    /// Safety factor (0.0 to 1.0) applied to estimates.
    safety_factor: f64,
}

impl BandwidthEstimator {
    /// Create a new bandwidth estimator.
    #[must_use]
    pub fn new(max_samples: usize, max_age_ms: u64) -> Self {
        Self {
            samples: VecDeque::new(),
            max_samples: max_samples.max(1),
            max_age_ms,
            safety_factor: 0.85,
        }
    }

    /// Create a new estimator with a custom safety factor.
    #[must_use]
    pub fn with_safety_factor(mut self, factor: f64) -> Self {
        self.safety_factor = factor.clamp(0.1, 1.0);
        self
    }

    /// Add a bandwidth sample.
    pub fn add_sample(&mut self, sample: BandwidthSample) {
        self.samples.push_back(sample);

        // Trim oldest if over max
        while self.samples.len() > self.max_samples {
            self.samples.pop_front();
        }

        // Trim expired samples
        let cutoff = sample.timestamp_ms.saturating_sub(self.max_age_ms);
        while let Some(front) = self.samples.front() {
            if front.timestamp_ms < cutoff {
                self.samples.pop_front();
            } else {
                break;
            }
        }
    }

    /// Get the number of samples currently stored.
    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Compute the estimated bandwidth using a weighted average.
    ///
    /// More recent samples are weighted more heavily.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn estimate(&self) -> u64 {
        if self.samples.is_empty() {
            return 0;
        }

        let _n = self.samples.len();
        let mut weighted_sum = 0.0_f64;
        let mut weight_total = 0.0_f64;

        for (i, sample) in self.samples.iter().enumerate() {
            // Linear weighting: newest sample gets weight n, oldest gets weight 1
            let weight = (i + 1) as f64;
            weighted_sum += sample.bandwidth_bps as f64 * weight;
            weight_total += weight;
        }

        let raw_estimate = weighted_sum / weight_total;
        (raw_estimate * self.safety_factor) as u64
    }

    /// Compute the harmonic mean of bandwidth samples.
    ///
    /// Harmonic mean is better for averaging rates.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn harmonic_mean(&self) -> u64 {
        if self.samples.is_empty() {
            return 0;
        }

        let _n = self.samples.len();
        let sum_reciprocals: f64 = self
            .samples
            .iter()
            .filter(|s| s.bandwidth_bps > 0)
            .map(|s| 1.0 / s.bandwidth_bps as f64)
            .sum();

        if sum_reciprocals <= 0.0 {
            return 0;
        }

        let non_zero_count = self.samples.iter().filter(|s| s.bandwidth_bps > 0).count();
        if non_zero_count == 0 {
            return 0;
        }

        let hm = non_zero_count as f64 / sum_reciprocals;
        (hm * self.safety_factor) as u64
    }

    /// Get the minimum bandwidth observed in the current window.
    #[must_use]
    pub fn minimum(&self) -> u64 {
        self.samples
            .iter()
            .map(|s| s.bandwidth_bps)
            .min()
            .unwrap_or(0)
    }

    /// Get the maximum bandwidth observed in the current window.
    #[must_use]
    pub fn maximum(&self) -> u64 {
        self.samples
            .iter()
            .map(|s| s.bandwidth_bps)
            .max()
            .unwrap_or(0)
    }

    /// Clear all samples.
    pub fn clear(&mut self) {
        self.samples.clear();
    }
}

impl Default for BandwidthEstimator {
    fn default() -> Self {
        Self::new(20, 30_000)
    }
}

/// A variant in the bitrate ladder for quality selection.
#[derive(Debug, Clone)]
pub struct QualityVariant {
    /// Variant identifier / index.
    pub index: usize,
    /// Required bandwidth in bits per second.
    pub bitrate_bps: u64,
    /// Resolution width.
    pub width: u32,
    /// Resolution height.
    pub height: u32,
}

impl QualityVariant {
    /// Create a new quality variant.
    #[must_use]
    pub fn new(index: usize, bitrate_bps: u64, width: u32, height: u32) -> Self {
        Self {
            index,
            bitrate_bps,
            width,
            height,
        }
    }
}

/// Quality selector that recommends variants based on bandwidth estimates.
pub struct QualitySelector {
    /// Available variants, sorted by bitrate ascending.
    variants: Vec<QualityVariant>,
    /// Headroom factor: estimated bandwidth must exceed variant bitrate by this factor.
    headroom_factor: f64,
}

impl QualitySelector {
    /// Create a new quality selector.
    #[must_use]
    pub fn new(mut variants: Vec<QualityVariant>, headroom_factor: f64) -> Self {
        variants.sort_by_key(|v| v.bitrate_bps);
        Self {
            variants,
            headroom_factor: headroom_factor.max(1.0),
        }
    }

    /// Select the best quality variant for the given estimated bandwidth.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn select(&self, estimated_bps: u64) -> Option<&QualityVariant> {
        let threshold = estimated_bps as f64 / self.headroom_factor;

        self.variants
            .iter()
            .rev()
            .find(|v| (v.bitrate_bps as f64) <= threshold)
    }

    /// Get the number of available variants.
    #[must_use]
    pub fn variant_count(&self) -> usize {
        self.variants.len()
    }

    /// Get the lowest quality variant.
    #[must_use]
    pub fn lowest(&self) -> Option<&QualityVariant> {
        self.variants.first()
    }

    /// Get the highest quality variant.
    #[must_use]
    pub fn highest(&self) -> Option<&QualityVariant> {
        self.variants.last()
    }
}

/// Buffer level tracker for adaptive streaming.
pub struct BufferTracker {
    /// Current buffer level in milliseconds.
    buffer_ms: u64,
    /// Minimum buffer threshold in milliseconds (below which we switch down).
    min_threshold_ms: u64,
    /// Maximum buffer threshold in milliseconds (above which we switch up).
    max_threshold_ms: u64,
}

impl BufferTracker {
    /// Create a new buffer tracker.
    #[must_use]
    pub fn new(min_threshold_ms: u64, max_threshold_ms: u64) -> Self {
        Self {
            buffer_ms: 0,
            min_threshold_ms,
            max_threshold_ms,
        }
    }

    /// Add content to the buffer.
    pub fn add(&mut self, duration: Duration) {
        self.buffer_ms += duration.as_millis() as u64;
    }

    /// Consume content from the buffer (playback).
    pub fn consume(&mut self, duration: Duration) {
        let ms = duration.as_millis() as u64;
        self.buffer_ms = self.buffer_ms.saturating_sub(ms);
    }

    /// Get the current buffer level in milliseconds.
    #[must_use]
    pub fn level_ms(&self) -> u64 {
        self.buffer_ms
    }

    /// Check if the buffer is below the minimum threshold.
    #[must_use]
    pub fn is_low(&self) -> bool {
        self.buffer_ms < self.min_threshold_ms
    }

    /// Check if the buffer is above the maximum threshold.
    #[must_use]
    pub fn is_high(&self) -> bool {
        self.buffer_ms > self.max_threshold_ms
    }

    /// Check if the buffer is in the safe zone (between thresholds).
    #[must_use]
    pub fn is_stable(&self) -> bool {
        self.buffer_ms >= self.min_threshold_ms && self.buffer_ms <= self.max_threshold_ms
    }

    /// Reset the buffer level.
    pub fn reset(&mut self) {
        self.buffer_ms = 0;
    }
}

impl Default for BufferTracker {
    fn default() -> Self {
        Self::new(5_000, 30_000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bandwidth_sample_from_download() {
        // 1 MB downloaded in 1 second = 8 Mbps
        let sample = BandwidthSample::from_download(1_000_000, 1000, 0);
        assert_eq!(sample.bandwidth_bps, 8_000_000);
    }

    #[test]
    fn test_bandwidth_sample_zero_time() {
        let sample = BandwidthSample::from_download(1000, 0, 0);
        assert_eq!(sample.bandwidth_bps, 0);
    }

    #[test]
    fn test_estimator_empty() {
        let est = BandwidthEstimator::default();
        assert_eq!(est.estimate(), 0);
        assert_eq!(est.sample_count(), 0);
    }

    #[test]
    fn test_estimator_single_sample() {
        let mut est = BandwidthEstimator::new(10, 30_000).with_safety_factor(1.0);
        est.add_sample(BandwidthSample::from_download(1_000_000, 1000, 0));
        assert_eq!(est.estimate(), 8_000_000);
    }

    #[test]
    fn test_estimator_weighted_average() {
        let mut est = BandwidthEstimator::new(10, 60_000).with_safety_factor(1.0);
        est.add_sample(BandwidthSample::from_download(1_000_000, 1000, 0));
        est.add_sample(BandwidthSample::from_download(2_000_000, 1000, 1000));
        // Weighted: (8M*1 + 16M*2) / 3 = 40M/3 ~= 13333333
        let estimate = est.estimate();
        assert!(estimate > 13_000_000);
        assert!(estimate < 14_000_000);
    }

    #[test]
    fn test_estimator_safety_factor() {
        let mut est = BandwidthEstimator::new(10, 30_000).with_safety_factor(0.5);
        est.add_sample(BandwidthSample::from_download(1_000_000, 1000, 0));
        assert_eq!(est.estimate(), 4_000_000);
    }

    #[test]
    fn test_estimator_harmonic_mean() {
        let mut est = BandwidthEstimator::new(10, 60_000).with_safety_factor(1.0);
        est.add_sample(BandwidthSample::from_download(1_000_000, 1000, 0)); // 8 Mbps
        est.add_sample(BandwidthSample::from_download(1_000_000, 1000, 1000)); // 8 Mbps
        assert_eq!(est.harmonic_mean(), 8_000_000);
    }

    #[test]
    fn test_estimator_min_max() {
        let mut est = BandwidthEstimator::new(10, 60_000);
        est.add_sample(BandwidthSample::from_download(500_000, 1000, 0));
        est.add_sample(BandwidthSample::from_download(2_000_000, 1000, 1000));
        assert_eq!(est.minimum(), 4_000_000);
        assert_eq!(est.maximum(), 16_000_000);
    }

    #[test]
    fn test_estimator_max_samples_trim() {
        let mut est = BandwidthEstimator::new(3, 60_000);
        for i in 0..5 {
            est.add_sample(BandwidthSample::from_download(1000, 1, i));
        }
        assert_eq!(est.sample_count(), 3);
    }

    #[test]
    fn test_estimator_clear() {
        let mut est = BandwidthEstimator::default();
        est.add_sample(BandwidthSample::from_download(1000, 1, 0));
        est.clear();
        assert_eq!(est.sample_count(), 0);
    }

    #[test]
    fn test_quality_selector_select_best() {
        let variants = vec![
            QualityVariant::new(0, 500_000, 426, 240),
            QualityVariant::new(1, 1_500_000, 854, 480),
            QualityVariant::new(2, 3_000_000, 1280, 720),
            QualityVariant::new(3, 5_000_000, 1920, 1080),
        ];
        let selector = QualitySelector::new(variants, 1.2);

        // With 4 Mbps, can afford 720p (3M * 1.2 = 3.6M < 4M)
        let selected = selector.select(4_000_000).expect("should succeed in test");
        assert_eq!(selected.index, 2);
    }

    #[test]
    fn test_quality_selector_too_low() {
        let variants = vec![QualityVariant::new(0, 1_000_000, 426, 240)];
        let selector = QualitySelector::new(variants, 1.5);
        let selected = selector.select(500_000);
        assert!(selected.is_none());
    }

    #[test]
    fn test_quality_selector_lowest_highest() {
        let variants = vec![
            QualityVariant::new(0, 500_000, 426, 240),
            QualityVariant::new(1, 5_000_000, 1920, 1080),
        ];
        let selector = QualitySelector::new(variants, 1.0);
        assert_eq!(
            selector
                .lowest()
                .expect("should succeed in test")
                .bitrate_bps,
            500_000
        );
        assert_eq!(
            selector
                .highest()
                .expect("should succeed in test")
                .bitrate_bps,
            5_000_000
        );
        assert_eq!(selector.variant_count(), 2);
    }

    #[test]
    fn test_buffer_tracker_basic() {
        let mut tracker = BufferTracker::new(5_000, 30_000);
        assert!(tracker.is_low());

        tracker.add(Duration::from_secs(10));
        assert_eq!(tracker.level_ms(), 10_000);
        assert!(tracker.is_stable());
    }

    #[test]
    fn test_buffer_tracker_consume() {
        let mut tracker = BufferTracker::new(5_000, 30_000);
        tracker.add(Duration::from_secs(20));
        tracker.consume(Duration::from_secs(16));
        assert_eq!(tracker.level_ms(), 4_000);
        assert!(tracker.is_low());
    }

    #[test]
    fn test_buffer_tracker_high() {
        let mut tracker = BufferTracker::new(5_000, 30_000);
        tracker.add(Duration::from_secs(35));
        assert!(tracker.is_high());
        assert!(!tracker.is_stable());
    }

    #[test]
    fn test_buffer_tracker_reset() {
        let mut tracker = BufferTracker::default();
        tracker.add(Duration::from_secs(15));
        tracker.reset();
        assert_eq!(tracker.level_ms(), 0);
    }

    #[test]
    fn test_buffer_tracker_underflow_protection() {
        let mut tracker = BufferTracker::new(5_000, 30_000);
        tracker.add(Duration::from_secs(1));
        tracker.consume(Duration::from_secs(5));
        assert_eq!(tracker.level_ms(), 0);
    }
}
