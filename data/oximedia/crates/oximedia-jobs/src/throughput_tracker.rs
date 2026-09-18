#![allow(dead_code)]
//! Real-time throughput tracking and reporting for the job queue.
//!
//! Tracks jobs completed per time window, computes moving averages,
//! and provides peak/trough detection and trend analysis.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// A single sample in the throughput timeline.
#[derive(Debug, Clone, Copy)]
pub struct ThroughputSample {
    /// Timestamp when this sample was recorded.
    pub timestamp: Instant,
    /// Number of jobs completed in this sample window.
    pub count: u64,
    /// Total bytes processed in this window (optional metric).
    pub bytes_processed: u64,
}

/// Configuration for the throughput tracker.
#[derive(Debug, Clone)]
pub struct ThroughputConfig {
    /// Duration of each sample bucket.
    pub bucket_duration: Duration,
    /// Maximum number of buckets to retain.
    pub max_buckets: usize,
    /// Window size for the moving average (in buckets).
    pub moving_avg_window: usize,
}

impl Default for ThroughputConfig {
    fn default() -> Self {
        Self {
            bucket_duration: Duration::from_secs(60),
            max_buckets: 1440, // 24 hours at 1-min buckets
            moving_avg_window: 5,
        }
    }
}

/// Throughput trend direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThroughputTrend {
    /// Throughput is increasing.
    Increasing,
    /// Throughput is stable.
    Stable,
    /// Throughput is decreasing.
    Decreasing,
    /// Not enough data to determine.
    Insufficient,
}

impl std::fmt::Display for ThroughputTrend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Increasing => write!(f, "Increasing"),
            Self::Stable => write!(f, "Stable"),
            Self::Decreasing => write!(f, "Decreasing"),
            Self::Insufficient => write!(f, "Insufficient"),
        }
    }
}

/// Summary statistics for a given time range.
#[derive(Debug, Clone)]
pub struct ThroughputSummary {
    /// Total jobs completed.
    pub total_jobs: u64,
    /// Total bytes processed.
    pub total_bytes: u64,
    /// Average jobs per bucket.
    pub avg_jobs_per_bucket: f64,
    /// Peak jobs in a single bucket.
    pub peak_jobs: u64,
    /// Minimum jobs in a single bucket.
    pub min_jobs: u64,
    /// Number of samples.
    pub sample_count: usize,
    /// Current trend.
    pub trend: ThroughputTrend,
}

/// Real-time throughput tracker.
#[derive(Debug)]
pub struct ThroughputTracker {
    /// Configuration.
    config: ThroughputConfig,
    /// Recorded samples.
    samples: VecDeque<ThroughputSample>,
    /// Accumulator for the current (open) bucket.
    current_count: u64,
    /// Bytes accumulator for the current bucket.
    current_bytes: u64,
    /// When the current bucket started.
    bucket_start: Instant,
    /// Lifetime total jobs tracked.
    lifetime_total: u64,
}

impl ThroughputTracker {
    /// Create a new throughput tracker.
    pub fn new(config: ThroughputConfig) -> Self {
        Self {
            config,
            samples: VecDeque::new(),
            current_count: 0,
            current_bytes: 0,
            bucket_start: Instant::now(),
            lifetime_total: 0,
        }
    }

    /// Record a completed job.
    pub fn record_completion(&mut self, bytes: u64) {
        self.maybe_rotate_bucket();
        self.current_count += 1;
        self.current_bytes += bytes;
        self.lifetime_total += 1;
    }

    /// Record multiple completed jobs at once.
    pub fn record_batch(&mut self, count: u64, bytes: u64) {
        self.maybe_rotate_bucket();
        self.current_count += count;
        self.current_bytes += bytes;
        self.lifetime_total += count;
    }

    /// Check if the current bucket should be rotated and do so if needed.
    fn maybe_rotate_bucket(&mut self) {
        let elapsed = self.bucket_start.elapsed();
        if elapsed >= self.config.bucket_duration {
            let sample = ThroughputSample {
                timestamp: self.bucket_start,
                count: self.current_count,
                bytes_processed: self.current_bytes,
            };
            self.samples.push_back(sample);
            if self.samples.len() > self.config.max_buckets {
                self.samples.pop_front();
            }
            self.current_count = 0;
            self.current_bytes = 0;
            self.bucket_start = Instant::now();
        }
    }

    /// Force-close the current bucket and start a new one.
    pub fn flush_bucket(&mut self) {
        let sample = ThroughputSample {
            timestamp: self.bucket_start,
            count: self.current_count,
            bytes_processed: self.current_bytes,
        };
        self.samples.push_back(sample);
        if self.samples.len() > self.config.max_buckets {
            self.samples.pop_front();
        }
        self.current_count = 0;
        self.current_bytes = 0;
        self.bucket_start = Instant::now();
    }

    /// Get the number of recorded samples (completed buckets).
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Get lifetime total jobs.
    pub fn lifetime_total(&self) -> u64 {
        self.lifetime_total
    }

    /// Compute a moving average over the most recent N buckets.
    #[allow(clippy::cast_precision_loss)]
    pub fn moving_average(&self) -> f64 {
        let window = self.config.moving_avg_window.min(self.samples.len());
        if window == 0 {
            return 0.0;
        }
        let sum: u64 = self
            .samples
            .iter()
            .rev()
            .take(window)
            .map(|s| s.count)
            .sum();
        sum as f64 / window as f64
    }

    /// Compute peak throughput across all samples.
    pub fn peak_throughput(&self) -> u64 {
        self.samples.iter().map(|s| s.count).max().unwrap_or(0)
    }

    /// Compute minimum throughput across all samples.
    pub fn min_throughput(&self) -> u64 {
        self.samples.iter().map(|s| s.count).min().unwrap_or(0)
    }

    /// Determine the current trend by comparing the last two moving-average windows.
    #[allow(clippy::cast_precision_loss)]
    pub fn trend(&self) -> ThroughputTrend {
        let window = self.config.moving_avg_window;
        if self.samples.len() < window * 2 {
            return ThroughputTrend::Insufficient;
        }
        let recent_sum: u64 = self
            .samples
            .iter()
            .rev()
            .take(window)
            .map(|s| s.count)
            .sum();
        let previous_sum: u64 = self
            .samples
            .iter()
            .rev()
            .skip(window)
            .take(window)
            .map(|s| s.count)
            .sum();
        let recent_avg = recent_sum as f64 / window as f64;
        let previous_avg = previous_sum as f64 / window as f64;
        let ratio = if previous_avg > 0.0 {
            recent_avg / previous_avg
        } else if recent_avg > 0.0 {
            return ThroughputTrend::Increasing;
        } else {
            return ThroughputTrend::Stable;
        };
        if ratio > 1.1 {
            ThroughputTrend::Increasing
        } else if ratio < 0.9 {
            ThroughputTrend::Decreasing
        } else {
            ThroughputTrend::Stable
        }
    }

    /// Generate a summary of all tracked data.
    #[allow(clippy::cast_precision_loss)]
    pub fn summary(&self) -> ThroughputSummary {
        let total_jobs: u64 = self.samples.iter().map(|s| s.count).sum();
        let total_bytes: u64 = self.samples.iter().map(|s| s.bytes_processed).sum();
        let count = self.samples.len();
        let avg = if count > 0 {
            total_jobs as f64 / count as f64
        } else {
            0.0
        };
        ThroughputSummary {
            total_jobs,
            total_bytes,
            avg_jobs_per_bucket: avg,
            peak_jobs: self.peak_throughput(),
            min_jobs: self.min_throughput(),
            sample_count: count,
            trend: self.trend(),
        }
    }

    /// Get the last N samples.
    pub fn recent_samples(&self, n: usize) -> Vec<&ThroughputSample> {
        self.samples.iter().rev().take(n).collect()
    }

    /// Compute bytes-per-second over the most recent window.
    #[allow(clippy::cast_precision_loss)]
    pub fn bytes_per_second(&self) -> f64 {
        let window = self.config.moving_avg_window.min(self.samples.len());
        if window == 0 {
            return 0.0;
        }
        let total_bytes: u64 = self
            .samples
            .iter()
            .rev()
            .take(window)
            .map(|s| s.bytes_processed)
            .sum();
        let window_duration = self.config.bucket_duration.as_secs_f64() * window as f64;
        if window_duration > 0.0 {
            total_bytes as f64 / window_duration
        } else {
            0.0
        }
    }

    /// Clear all samples.
    pub fn clear(&mut self) {
        self.samples.clear();
        self.current_count = 0;
        self.current_bytes = 0;
        self.bucket_start = Instant::now();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tracker_with_samples(counts: &[u64]) -> ThroughputTracker {
        let config = ThroughputConfig {
            bucket_duration: Duration::from_secs(60),
            max_buckets: 100,
            moving_avg_window: 3,
        };
        let mut tracker = ThroughputTracker::new(config);
        for &c in counts {
            for _ in 0..c {
                tracker.current_count += 1;
                tracker.lifetime_total += 1;
            }
            tracker.flush_bucket();
        }
        tracker
    }

    #[test]
    fn test_default_config() {
        let config = ThroughputConfig::default();
        assert_eq!(config.bucket_duration, Duration::from_secs(60));
        assert_eq!(config.max_buckets, 1440);
        assert_eq!(config.moving_avg_window, 5);
    }

    #[test]
    fn test_record_completion() {
        let config = ThroughputConfig::default();
        let mut tracker = ThroughputTracker::new(config);
        tracker.record_completion(1024);
        assert_eq!(tracker.lifetime_total(), 1);
    }

    #[test]
    fn test_record_batch() {
        let config = ThroughputConfig::default();
        let mut tracker = ThroughputTracker::new(config);
        tracker.record_batch(10, 5000);
        assert_eq!(tracker.lifetime_total(), 10);
    }

    #[test]
    fn test_flush_bucket() {
        let config = ThroughputConfig::default();
        let mut tracker = ThroughputTracker::new(config);
        tracker.record_completion(100);
        tracker.record_completion(200);
        tracker.flush_bucket();
        assert_eq!(tracker.sample_count(), 1);
    }

    #[test]
    fn test_moving_average() {
        let tracker = make_tracker_with_samples(&[10, 20, 30]);
        let avg = tracker.moving_average();
        assert!((avg - 20.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_moving_average_empty() {
        let config = ThroughputConfig::default();
        let tracker = ThroughputTracker::new(config);
        assert!((tracker.moving_average() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_peak_throughput() {
        let tracker = make_tracker_with_samples(&[5, 50, 25]);
        assert_eq!(tracker.peak_throughput(), 50);
    }

    #[test]
    fn test_min_throughput() {
        let tracker = make_tracker_with_samples(&[5, 50, 25]);
        assert_eq!(tracker.min_throughput(), 5);
    }

    #[test]
    fn test_trend_insufficient() {
        let tracker = make_tracker_with_samples(&[10, 20]);
        assert_eq!(tracker.trend(), ThroughputTrend::Insufficient);
    }

    #[test]
    fn test_trend_increasing() {
        // window=3, need 6 samples: first 3 low, last 3 high
        let tracker = make_tracker_with_samples(&[1, 1, 1, 10, 10, 10]);
        assert_eq!(tracker.trend(), ThroughputTrend::Increasing);
    }

    #[test]
    fn test_trend_decreasing() {
        let tracker = make_tracker_with_samples(&[10, 10, 10, 1, 1, 1]);
        assert_eq!(tracker.trend(), ThroughputTrend::Decreasing);
    }

    #[test]
    fn test_trend_stable() {
        let tracker = make_tracker_with_samples(&[10, 10, 10, 10, 10, 10]);
        assert_eq!(tracker.trend(), ThroughputTrend::Stable);
    }

    #[test]
    fn test_summary() {
        let tracker = make_tracker_with_samples(&[10, 20, 30]);
        let summary = tracker.summary();
        assert_eq!(summary.total_jobs, 60);
        assert_eq!(summary.peak_jobs, 30);
        assert_eq!(summary.min_jobs, 10);
        assert_eq!(summary.sample_count, 3);
    }

    #[test]
    fn test_recent_samples() {
        let tracker = make_tracker_with_samples(&[5, 10, 15, 20]);
        let recent = tracker.recent_samples(2);
        assert_eq!(recent.len(), 2);
        assert_eq!(recent[0].count, 20);
        assert_eq!(recent[1].count, 15);
    }

    #[test]
    fn test_clear() {
        let mut tracker = make_tracker_with_samples(&[5, 10]);
        tracker.clear();
        assert_eq!(tracker.sample_count(), 0);
    }

    #[test]
    fn test_throughput_trend_display() {
        assert_eq!(ThroughputTrend::Increasing.to_string(), "Increasing");
        assert_eq!(ThroughputTrend::Stable.to_string(), "Stable");
        assert_eq!(ThroughputTrend::Decreasing.to_string(), "Decreasing");
        assert_eq!(ThroughputTrend::Insufficient.to_string(), "Insufficient");
    }

    #[test]
    fn test_max_buckets_eviction() {
        let config = ThroughputConfig {
            bucket_duration: Duration::from_secs(1),
            max_buckets: 3,
            moving_avg_window: 2,
        };
        let mut tracker = ThroughputTracker::new(config);
        for i in 0..5 {
            tracker.current_count = i;
            tracker.flush_bucket();
        }
        assert_eq!(tracker.sample_count(), 3);
    }
}
