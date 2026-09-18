#![allow(dead_code)]

//! Synchronisation statistics collection and reporting.
//!
//! Aggregates statistics about time-sync performance, including jitter,
//! offset distributions, lock durations, and sync quality metrics.

use std::collections::VecDeque;
use std::fmt;
use std::time::Duration;

/// Quality level of a synchronisation session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SyncQuality {
    /// No synchronisation achieved.
    NoSync,
    /// Poor quality — frequent unlocks.
    Poor,
    /// Fair quality — occasional glitches.
    Fair,
    /// Good quality — stable lock.
    Good,
    /// Excellent quality — sub-microsecond stability.
    Excellent,
}

impl fmt::Display for SyncQuality {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoSync => write!(f, "NoSync"),
            Self::Poor => write!(f, "Poor"),
            Self::Fair => write!(f, "Fair"),
            Self::Good => write!(f, "Good"),
            Self::Excellent => write!(f, "Excellent"),
        }
    }
}

/// A single offset measurement used for statistics.
#[derive(Debug, Clone, Copy)]
pub struct OffsetMeasurement {
    /// Offset from reference in nanoseconds (signed).
    pub offset_ns: i64,
    /// Round-trip delay in nanoseconds.
    pub delay_ns: u64,
}

impl OffsetMeasurement {
    /// Create a new measurement.
    pub fn new(offset_ns: i64, delay_ns: u64) -> Self {
        Self {
            offset_ns,
            delay_ns,
        }
    }
}

/// Running statistics accumulator for a scalar value.
#[derive(Debug, Clone)]
pub struct RunningStats {
    /// Number of observations.
    count: u64,
    /// Running mean.
    mean: f64,
    /// Running M2 for variance (Welford's algorithm).
    m2: f64,
    /// Minimum observed value.
    min: f64,
    /// Maximum observed value.
    max: f64,
}

impl RunningStats {
    /// Create a new empty accumulator.
    pub fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
            min: f64::MAX,
            max: f64::MIN,
        }
    }

    /// Add a value (Welford's online algorithm).
    pub fn push(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        #[allow(clippy::cast_precision_loss)]
        let n = self.count as f64;
        self.mean += delta / n;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
        if value < self.min {
            self.min = value;
        }
        if value > self.max {
            self.max = value;
        }
    }

    /// Number of observations.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Mean value.
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Population variance.
    #[allow(clippy::cast_precision_loss)]
    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        self.m2 / self.count as f64
    }

    /// Population standard deviation.
    pub fn stddev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Sample variance (Bessel correction).
    #[allow(clippy::cast_precision_loss)]
    pub fn sample_variance(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        self.m2 / (self.count - 1) as f64
    }

    /// Sample standard deviation.
    pub fn sample_stddev(&self) -> f64 {
        self.sample_variance().sqrt()
    }

    /// Minimum value observed.
    pub fn min(&self) -> Option<f64> {
        if self.count == 0 {
            None
        } else {
            Some(self.min)
        }
    }

    /// Maximum value observed.
    pub fn max(&self) -> Option<f64> {
        if self.count == 0 {
            None
        } else {
            Some(self.max)
        }
    }

    /// Range (max - min).
    pub fn range(&self) -> Option<f64> {
        if self.count == 0 {
            None
        } else {
            Some(self.max - self.min)
        }
    }
}

impl Default for RunningStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Aggregated synchronisation statistics.
#[derive(Debug, Clone)]
pub struct SyncStatsCollector {
    /// Offset statistics in nanoseconds.
    pub offset_stats: RunningStats,
    /// Delay statistics in nanoseconds.
    pub delay_stats: RunningStats,
    /// Recent measurements (bounded window).
    recent: VecDeque<OffsetMeasurement>,
    /// Maximum recent-window size.
    window_size: usize,
    /// Count of lock events.
    pub lock_count: u64,
    /// Count of unlock events.
    pub unlock_count: u64,
    /// Total locked duration.
    pub locked_duration: Duration,
    /// Total unlocked duration.
    pub unlocked_duration: Duration,
}

impl SyncStatsCollector {
    /// Create a new collector with the given window size.
    pub fn new(window_size: usize) -> Self {
        Self {
            offset_stats: RunningStats::new(),
            delay_stats: RunningStats::new(),
            recent: VecDeque::with_capacity(window_size),
            window_size,
            lock_count: 0,
            unlock_count: 0,
            locked_duration: Duration::ZERO,
            unlocked_duration: Duration::ZERO,
        }
    }

    /// Record an offset measurement.
    #[allow(clippy::cast_precision_loss)]
    pub fn record(&mut self, measurement: OffsetMeasurement) {
        self.offset_stats.push(measurement.offset_ns as f64);
        self.delay_stats.push(measurement.delay_ns as f64);
        if self.recent.len() >= self.window_size {
            self.recent.pop_front();
        }
        self.recent.push_back(measurement);
    }

    /// Record a lock event.
    pub fn record_lock(&mut self) {
        self.lock_count += 1;
    }

    /// Record an unlock event.
    pub fn record_unlock(&mut self) {
        self.unlock_count += 1;
    }

    /// Add to the locked duration counter.
    pub fn add_locked_duration(&mut self, d: Duration) {
        self.locked_duration += d;
    }

    /// Add to the unlocked duration counter.
    pub fn add_unlocked_duration(&mut self, d: Duration) {
        self.unlocked_duration += d;
    }

    /// Total measurement count.
    pub fn measurement_count(&self) -> u64 {
        self.offset_stats.count()
    }

    /// Number of recent samples currently stored.
    pub fn recent_count(&self) -> usize {
        self.recent.len()
    }

    /// Mean jitter (standard deviation of recent offsets).
    #[allow(clippy::cast_precision_loss)]
    pub fn jitter_ns(&self) -> f64 {
        if self.recent.len() < 2 {
            return 0.0;
        }
        let n = self.recent.len() as f64;
        let mean: f64 = self.recent.iter().map(|m| m.offset_ns as f64).sum::<f64>() / n;
        let var: f64 = self
            .recent
            .iter()
            .map(|m| {
                let d = m.offset_ns as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / (n - 1.0);
        var.sqrt()
    }

    /// Lock availability as a ratio in [0.0, 1.0].
    pub fn availability(&self) -> f64 {
        let total = self.locked_duration + self.unlocked_duration;
        if total.is_zero() {
            return 0.0;
        }
        self.locked_duration.as_secs_f64() / total.as_secs_f64()
    }

    /// Evaluate overall sync quality from collected data.
    pub fn quality(&self) -> SyncQuality {
        if self.measurement_count() == 0 {
            return SyncQuality::NoSync;
        }
        let jitter = self.jitter_ns();
        let avail = self.availability();
        if avail < 0.5 {
            SyncQuality::Poor
        } else if jitter > 100_000.0 {
            SyncQuality::Poor
        } else if jitter > 10_000.0 || avail < 0.9 {
            SyncQuality::Fair
        } else if jitter > 1_000.0 || avail < 0.99 {
            SyncQuality::Good
        } else {
            SyncQuality::Excellent
        }
    }

    /// Generate a summary report as a formatted string.
    pub fn summary(&self) -> String {
        format!(
            "Measurements: {}, Jitter: {:.1} ns, Quality: {}, Availability: {:.1}%",
            self.measurement_count(),
            self.jitter_ns(),
            self.quality(),
            self.availability() * 100.0,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_running_stats_empty() {
        let s = RunningStats::new();
        assert_eq!(s.count(), 0);
        assert!((s.mean() - 0.0).abs() < f64::EPSILON);
        assert!(s.min().is_none());
        assert!(s.max().is_none());
    }

    #[test]
    fn test_running_stats_single() {
        let mut s = RunningStats::new();
        s.push(42.0);
        assert_eq!(s.count(), 1);
        assert!((s.mean() - 42.0).abs() < f64::EPSILON);
        assert!((s.variance() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_running_stats_mean() {
        let mut s = RunningStats::new();
        s.push(10.0);
        s.push(20.0);
        s.push(30.0);
        assert!((s.mean() - 20.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_running_stats_stddev() {
        let mut s = RunningStats::new();
        for v in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
            s.push(v);
        }
        // Population stddev = 2.0
        assert!((s.stddev() - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_running_stats_min_max_range() {
        let mut s = RunningStats::new();
        s.push(5.0);
        s.push(15.0);
        s.push(10.0);
        assert!((s.min().expect("should succeed in test") - 5.0).abs() < f64::EPSILON);
        assert!((s.max().expect("should succeed in test") - 15.0).abs() < f64::EPSILON);
        assert!((s.range().expect("should succeed in test") - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_collector_record() {
        let mut c = SyncStatsCollector::new(100);
        c.record(OffsetMeasurement::new(100, 50));
        c.record(OffsetMeasurement::new(200, 60));
        assert_eq!(c.measurement_count(), 2);
        assert_eq!(c.recent_count(), 2);
    }

    #[test]
    fn test_collector_window() {
        let mut c = SyncStatsCollector::new(3);
        for i in 0..5 {
            c.record(OffsetMeasurement::new(i * 10, 10));
        }
        assert_eq!(c.recent_count(), 3);
    }

    #[test]
    fn test_collector_jitter_constant() {
        let mut c = SyncStatsCollector::new(100);
        for _ in 0..10 {
            c.record(OffsetMeasurement::new(100, 50));
        }
        assert!((c.jitter_ns() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_collector_jitter_nonzero() {
        let mut c = SyncStatsCollector::new(100);
        c.record(OffsetMeasurement::new(100, 50));
        c.record(OffsetMeasurement::new(200, 50));
        assert!(c.jitter_ns() > 0.0);
    }

    #[test]
    fn test_availability_zero() {
        let c = SyncStatsCollector::new(100);
        assert!((c.availability() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_availability_full() {
        let mut c = SyncStatsCollector::new(100);
        c.add_locked_duration(Duration::from_secs(100));
        assert!((c.availability() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_availability_half() {
        let mut c = SyncStatsCollector::new(100);
        c.add_locked_duration(Duration::from_secs(50));
        c.add_unlocked_duration(Duration::from_secs(50));
        assert!((c.availability() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_quality_no_sync() {
        let c = SyncStatsCollector::new(100);
        assert_eq!(c.quality(), SyncQuality::NoSync);
    }

    #[test]
    fn test_quality_excellent() {
        let mut c = SyncStatsCollector::new(100);
        for _ in 0..20 {
            c.record(OffsetMeasurement::new(100, 50));
        }
        c.add_locked_duration(Duration::from_secs(1000));
        assert_eq!(c.quality(), SyncQuality::Excellent);
    }

    #[test]
    fn test_lock_unlock_count() {
        let mut c = SyncStatsCollector::new(100);
        c.record_lock();
        c.record_lock();
        c.record_unlock();
        assert_eq!(c.lock_count, 2);
        assert_eq!(c.unlock_count, 1);
    }

    #[test]
    fn test_summary_string() {
        let mut c = SyncStatsCollector::new(100);
        c.record(OffsetMeasurement::new(100, 50));
        let s = c.summary();
        assert!(s.contains("Measurements: 1"));
    }

    #[test]
    fn test_sync_quality_display() {
        assert_eq!(SyncQuality::Excellent.to_string(), "Excellent");
        assert_eq!(SyncQuality::NoSync.to_string(), "NoSync");
    }

    #[test]
    fn test_offset_measurement_creation() {
        let m = OffsetMeasurement::new(-500, 1000);
        assert_eq!(m.offset_ns, -500);
        assert_eq!(m.delay_ns, 1000);
    }

    #[test]
    fn test_running_stats_sample_variance() {
        let mut s = RunningStats::new();
        s.push(10.0);
        s.push(20.0);
        // sample variance = (10-15)^2 + (20-15)^2 / (2-1) = 50
        assert!((s.sample_variance() - 50.0).abs() < 0.01);
        assert!((s.sample_stddev() - 50.0_f64.sqrt()).abs() < 0.01);
    }
}
