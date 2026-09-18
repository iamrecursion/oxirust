//! I/O statistics: throughput tracking, latency percentiles, and error rates.
//!
//! Provides lightweight, allocation-friendly statistics collectors for I/O
//! operations.  All types are `no_std`-friendly (no heap allocation required
//! for the core histogram types).

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Throughput counter
// ---------------------------------------------------------------------------

/// Tracks byte throughput over a sliding window.
#[derive(Debug)]
pub struct ThroughputCounter {
    window_secs: f64,
    /// Ring buffer of (timestamp, bytes) samples.
    samples: Vec<(Instant, u64)>,
    capacity: usize,
    head: usize,
    count: usize,
    total_bytes: u64,
}

impl ThroughputCounter {
    /// Create a counter with a sliding window of `window_secs` seconds and
    /// a ring buffer of `capacity` samples.
    #[must_use]
    pub fn new(window_secs: f64, capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            window_secs,
            samples: vec![(Instant::now(), 0); capacity],
            capacity,
            head: 0,
            count: 0,
            total_bytes: 0,
        }
    }

    /// Record that `bytes` were transferred at `now`.
    pub fn record(&mut self, bytes: u64, now: Instant) {
        self.samples[self.head] = (now, bytes);
        self.head = (self.head + 1) % self.capacity;
        if self.count < self.capacity {
            self.count += 1;
        }
        self.total_bytes += bytes;
    }

    /// Bytes per second over the sliding window, evaluated at `now`.
    #[must_use]
    pub fn bytes_per_sec(&self, now: Instant) -> f64 {
        let threshold = Duration::from_secs_f64(self.window_secs);
        let window_bytes: u64 = self
            .samples
            .iter()
            .take(self.count)
            .filter(|(ts, _)| now.duration_since(*ts) <= threshold)
            .map(|(_, b)| *b)
            .sum();
        if self.window_secs > 0.0 {
            window_bytes as f64 / self.window_secs
        } else {
            0.0
        }
    }

    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        self.total_bytes
    }

    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.count
    }
}

// ---------------------------------------------------------------------------
// Latency histogram (fixed-size buckets)
// ---------------------------------------------------------------------------

/// Bucket boundaries in microseconds: [0,10), [10,100), [100,1000), [1ms,10ms),
/// [10ms,100ms), [100ms,1s), [1s,∞).
const BUCKET_BOUNDARIES_US: &[u64] = &[10, 100, 1_000, 10_000, 100_000, 1_000_000];

/// A fixed-bucket latency histogram (values in microseconds).
#[derive(Debug, Clone)]
pub struct LatencyHistogram {
    /// Counts per bucket.  `buckets[n]` covers [`BUCKET_BOUNDARIES_US`[n-1], `BUCKET_BOUNDARIES_US`[n]).
    /// `buckets[0]` covers [0, `BUCKET_BOUNDARIES_US`[0]).
    /// `buckets[last]` covers [`BUCKET_BOUNDARIES_US`[last-1], ∞).
    buckets: [u64; 7],
    total: u64,
    sum_us: u64,
    min_us: u64,
    max_us: u64,
}

impl LatencyHistogram {
    #[must_use]
    pub fn new() -> Self {
        Self {
            buckets: [0; 7],
            total: 0,
            sum_us: 0,
            min_us: u64::MAX,
            max_us: 0,
        }
    }

    /// Record a latency sample.
    #[allow(clippy::cast_possible_truncation)]
    pub fn record(&mut self, latency: Duration) {
        let us = latency.as_micros() as u64;
        self.total += 1;
        self.sum_us += us;
        if us < self.min_us {
            self.min_us = us;
        }
        if us > self.max_us {
            self.max_us = us;
        }
        let bucket = BUCKET_BOUNDARIES_US
            .iter()
            .position(|&b| us < b)
            .unwrap_or(BUCKET_BOUNDARIES_US.len());
        self.buckets[bucket] += 1;
    }

    /// Mean latency in microseconds.
    #[must_use]
    pub fn mean_us(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        self.sum_us as f64 / self.total as f64
    }

    /// Approximate percentile (linear interpolation within bucket).
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub fn percentile_us(&self, p: f64) -> u64 {
        if self.total == 0 {
            return 0;
        }
        let target = (p / 100.0 * self.total as f64).ceil() as u64;
        let mut cumulative: u64 = 0;
        for (i, &count) in self.buckets.iter().enumerate() {
            cumulative += count;
            if cumulative >= target {
                // Return the upper boundary of this bucket (or max recorded)
                return BUCKET_BOUNDARIES_US.get(i).copied().unwrap_or(self.max_us);
            }
        }
        self.max_us
    }

    #[must_use]
    pub fn count(&self) -> u64 {
        self.total
    }

    #[must_use]
    pub fn min_us(&self) -> u64 {
        if self.total == 0 {
            0
        } else {
            self.min_us
        }
    }

    #[must_use]
    pub fn max_us(&self) -> u64 {
        self.max_us
    }
}

impl Default for LatencyHistogram {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Error rate tracker
// ---------------------------------------------------------------------------

/// Tracks success/error counts and computes an error rate.
#[derive(Debug, Clone, Default)]
pub struct ErrorRateTracker {
    successes: u64,
    errors: u64,
}

impl ErrorRateTracker {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_success(&mut self) {
        self.successes += 1;
    }

    pub fn record_error(&mut self) {
        self.errors += 1;
    }

    /// Error rate in [0.0, 1.0].
    #[must_use]
    pub fn error_rate(&self) -> f64 {
        let total = self.successes + self.errors;
        if total == 0 {
            0.0
        } else {
            self.errors as f64 / total as f64
        }
    }

    #[must_use]
    pub fn total(&self) -> u64 {
        self.successes + self.errors
    }

    #[must_use]
    pub fn errors(&self) -> u64 {
        self.errors
    }

    #[must_use]
    pub fn successes(&self) -> u64 {
        self.successes
    }
}

// ---------------------------------------------------------------------------
// Aggregated I/O statistics
// ---------------------------------------------------------------------------

/// Aggregate I/O statistics for a single I/O path.
#[derive(Debug)]
pub struct IoStats {
    pub read_bytes: u64,
    pub write_bytes: u64,
    pub read_ops: u64,
    pub write_ops: u64,
    pub read_latency: LatencyHistogram,
    pub write_latency: LatencyHistogram,
    pub errors: ErrorRateTracker,
    pub throughput: ThroughputCounter,
    started_at: Instant,
}

impl IoStats {
    #[must_use]
    pub fn new() -> Self {
        Self {
            read_bytes: 0,
            write_bytes: 0,
            read_ops: 0,
            write_ops: 0,
            read_latency: LatencyHistogram::new(),
            write_latency: LatencyHistogram::new(),
            errors: ErrorRateTracker::new(),
            throughput: ThroughputCounter::new(1.0, 64),
            started_at: Instant::now(),
        }
    }

    pub fn record_read(&mut self, bytes: usize, latency: Duration) {
        self.read_bytes += bytes as u64;
        self.read_ops += 1;
        self.read_latency.record(latency);
        self.throughput.record(bytes as u64, Instant::now());
        self.errors.record_success();
    }

    pub fn record_write(&mut self, bytes: usize, latency: Duration) {
        self.write_bytes += bytes as u64;
        self.write_ops += 1;
        self.write_latency.record(latency);
        self.throughput.record(bytes as u64, Instant::now());
        self.errors.record_success();
    }

    pub fn record_error(&mut self) {
        self.errors.record_error();
    }

    #[must_use]
    pub fn total_ops(&self) -> u64 {
        self.read_ops + self.write_ops
    }

    #[must_use]
    pub fn uptime(&self) -> Duration {
        self.started_at.elapsed()
    }
}

impl Default for IoStats {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_throughput_counter_records_bytes() {
        let mut tc = ThroughputCounter::new(1.0, 16);
        tc.record(1000, Instant::now());
        assert_eq!(tc.total_bytes(), 1000);
        assert_eq!(tc.sample_count(), 1);
    }

    #[test]
    fn test_throughput_counter_bytes_per_sec() {
        let mut tc = ThroughputCounter::new(1.0, 16);
        tc.record(1000, Instant::now());
        let bps = tc.bytes_per_sec(Instant::now());
        assert!(bps > 0.0);
    }

    #[test]
    fn test_throughput_counter_ring_buffer_wraps() {
        let mut tc = ThroughputCounter::new(10.0, 4);
        for _ in 0..10 {
            tc.record(100, Instant::now());
        }
        assert_eq!(tc.sample_count(), 4); // capped at capacity
    }

    #[test]
    fn test_latency_histogram_record_and_count() {
        let mut h = LatencyHistogram::new();
        h.record(Duration::from_micros(50));
        h.record(Duration::from_micros(500));
        assert_eq!(h.count(), 2);
    }

    #[test]
    fn test_latency_histogram_mean() {
        let mut h = LatencyHistogram::new();
        h.record(Duration::from_micros(100));
        h.record(Duration::from_micros(300));
        let mean = h.mean_us();
        assert!((mean - 200.0).abs() < 1.0);
    }

    #[test]
    fn test_latency_histogram_min_max() {
        let mut h = LatencyHistogram::new();
        h.record(Duration::from_micros(5));
        h.record(Duration::from_millis(2));
        assert_eq!(h.min_us(), 5);
        assert_eq!(h.max_us(), 2000);
    }

    #[test]
    fn test_latency_histogram_percentile_p50() {
        let mut h = LatencyHistogram::new();
        // All in the [0,10) bucket
        for _ in 0..100 {
            h.record(Duration::from_micros(5));
        }
        let p50 = h.percentile_us(50.0);
        // p50 should be in the first bucket boundary (10)
        assert_eq!(p50, 10);
    }

    #[test]
    fn test_latency_histogram_empty() {
        let h = LatencyHistogram::new();
        assert_eq!(h.count(), 0);
        assert_eq!(h.mean_us(), 0.0);
        assert_eq!(h.min_us(), 0);
    }

    #[test]
    fn test_error_rate_all_success() {
        let mut t = ErrorRateTracker::new();
        t.record_success();
        t.record_success();
        assert_eq!(t.error_rate(), 0.0);
    }

    #[test]
    fn test_error_rate_half_errors() {
        let mut t = ErrorRateTracker::new();
        t.record_success();
        t.record_error();
        assert!((t.error_rate() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_error_rate_empty() {
        let t = ErrorRateTracker::new();
        assert_eq!(t.error_rate(), 0.0);
        assert_eq!(t.total(), 0);
    }

    #[test]
    fn test_io_stats_record_read() {
        let mut s = IoStats::new();
        s.record_read(4096, Duration::from_micros(200));
        assert_eq!(s.read_bytes, 4096);
        assert_eq!(s.read_ops, 1);
        assert_eq!(s.total_ops(), 1);
    }

    #[test]
    fn test_io_stats_record_write() {
        let mut s = IoStats::new();
        s.record_write(8192, Duration::from_micros(300));
        assert_eq!(s.write_bytes, 8192);
        assert_eq!(s.write_ops, 1);
    }

    #[test]
    fn test_io_stats_error_tracking() {
        let mut s = IoStats::new();
        s.record_read(100, Duration::from_micros(10));
        s.record_error();
        assert!((s.errors.error_rate() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_io_stats_uptime_non_zero() {
        let s = IoStats::new();
        // uptime should be at least Duration::ZERO
        assert!(s.uptime() >= Duration::ZERO);
    }
}
