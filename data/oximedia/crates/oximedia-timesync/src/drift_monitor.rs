#![allow(dead_code)]

//! Drift monitoring for time synchronisation.
//!
//! Continuously tracks clock drift between a local clock and reference,
//! maintaining a history window for trend analysis and alerting.

use std::collections::VecDeque;
use std::fmt;
use std::time::Duration;

/// Severity of a drift event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DriftSeverity {
    /// Within normal operating range.
    Normal,
    /// Slightly elevated, warrants monitoring.
    Elevated,
    /// Approaching threshold, attention needed.
    Warning,
    /// Exceeds acceptable limits.
    Alarm,
}

impl fmt::Display for DriftSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Normal => write!(f, "Normal"),
            Self::Elevated => write!(f, "Elevated"),
            Self::Warning => write!(f, "Warning"),
            Self::Alarm => write!(f, "Alarm"),
        }
    }
}

/// A single drift measurement sample.
#[derive(Debug, Clone, Copy)]
pub struct DriftSample {
    /// Measured offset in nanoseconds (signed: positive = local ahead).
    pub offset_ns: i64,
    /// Timestamp of the measurement in nanoseconds since epoch.
    pub timestamp_ns: u64,
}

impl DriftSample {
    /// Create a new drift sample.
    pub fn new(offset_ns: i64, timestamp_ns: u64) -> Self {
        Self {
            offset_ns,
            timestamp_ns,
        }
    }

    /// Absolute offset magnitude in nanoseconds.
    pub fn abs_offset_ns(&self) -> u64 {
        self.offset_ns.unsigned_abs()
    }
}

/// Configuration for the drift monitor.
#[derive(Debug, Clone)]
pub struct DriftMonitorConfig {
    /// Maximum number of samples to retain.
    pub window_size: usize,
    /// Normal threshold in nanoseconds.
    pub normal_threshold_ns: u64,
    /// Elevated threshold in nanoseconds.
    pub elevated_threshold_ns: u64,
    /// Warning threshold in nanoseconds.
    pub warning_threshold_ns: u64,
    /// Alarm threshold in nanoseconds.
    pub alarm_threshold_ns: u64,
    /// Minimum samples before drift rate is considered valid.
    pub min_samples_for_rate: usize,
}

impl Default for DriftMonitorConfig {
    fn default() -> Self {
        Self {
            window_size: 1000,
            normal_threshold_ns: 1_000,    // 1 us
            elevated_threshold_ns: 10_000, // 10 us
            warning_threshold_ns: 100_000, // 100 us
            alarm_threshold_ns: 1_000_000, // 1 ms
            min_samples_for_rate: 10,
        }
    }
}

/// Monitor that tracks drift over time and evaluates severity.
#[derive(Debug, Clone)]
pub struct DriftMonitor {
    /// Configuration.
    config: DriftMonitorConfig,
    /// Samples in chronological order.
    samples: VecDeque<DriftSample>,
    /// Peak absolute offset observed.
    peak_offset_ns: u64,
    /// Count of alarm events.
    alarm_count: u64,
}

impl DriftMonitor {
    /// Create a new drift monitor with default configuration.
    pub fn new() -> Self {
        Self::with_config(DriftMonitorConfig::default())
    }

    /// Create a drift monitor with a given configuration.
    pub fn with_config(config: DriftMonitorConfig) -> Self {
        Self {
            samples: VecDeque::with_capacity(config.window_size),
            config,
            peak_offset_ns: 0,
            alarm_count: 0,
        }
    }

    /// Push a new sample into the monitor.
    pub fn push(&mut self, sample: DriftSample) {
        if self.samples.len() >= self.config.window_size {
            self.samples.pop_front();
        }
        let abs = sample.abs_offset_ns();
        if abs > self.peak_offset_ns {
            self.peak_offset_ns = abs;
        }
        if abs >= self.config.alarm_threshold_ns {
            self.alarm_count += 1;
        }
        self.samples.push_back(sample);
    }

    /// Number of samples currently stored.
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Peak absolute offset ever observed in the current window.
    pub fn peak_offset_ns(&self) -> u64 {
        self.peak_offset_ns
    }

    /// Total alarm events.
    pub fn alarm_count(&self) -> u64 {
        self.alarm_count
    }

    /// Get the latest sample.
    pub fn latest(&self) -> Option<&DriftSample> {
        self.samples.back()
    }

    /// Mean offset in nanoseconds over the current window.
    #[allow(clippy::cast_precision_loss)]
    pub fn mean_offset_ns(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let sum: i64 = self.samples.iter().map(|s| s.offset_ns).sum();
        sum as f64 / self.samples.len() as f64
    }

    /// Standard deviation of offset in nanoseconds.
    #[allow(clippy::cast_precision_loss)]
    pub fn stddev_offset_ns(&self) -> f64 {
        if self.samples.len() < 2 {
            return 0.0;
        }
        let mean = self.mean_offset_ns();
        let n = self.samples.len() as f64;
        let var: f64 = self
            .samples
            .iter()
            .map(|s| {
                let d = s.offset_ns as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / (n - 1.0);
        var.sqrt()
    }

    /// Minimum offset in the window.
    pub fn min_offset_ns(&self) -> Option<i64> {
        self.samples.iter().map(|s| s.offset_ns).min()
    }

    /// Maximum offset in the window.
    pub fn max_offset_ns(&self) -> Option<i64> {
        self.samples.iter().map(|s| s.offset_ns).max()
    }

    /// Estimated drift rate in nanoseconds per second (ppb equivalent).
    /// Uses a simple linear regression over the sample window.
    #[allow(clippy::cast_precision_loss)]
    pub fn drift_rate_ns_per_sec(&self) -> Option<f64> {
        if self.samples.len() < self.config.min_samples_for_rate {
            return None;
        }
        let n = self.samples.len() as f64;
        let t0 = self.samples.front()?.timestamp_ns;
        let sum_x: f64 = self
            .samples
            .iter()
            .map(|s| (s.timestamp_ns - t0) as f64)
            .sum();
        let sum_y: f64 = self.samples.iter().map(|s| s.offset_ns as f64).sum();
        let sum_xy: f64 = self
            .samples
            .iter()
            .map(|s| (s.timestamp_ns - t0) as f64 * s.offset_ns as f64)
            .sum();
        let sum_xx: f64 = self
            .samples
            .iter()
            .map(|s| {
                let x = (s.timestamp_ns - t0) as f64;
                x * x
            })
            .sum();
        let denom = n * sum_xx - sum_x * sum_x;
        if denom.abs() < f64::EPSILON {
            return None;
        }
        let slope_per_ns = (n * sum_xy - sum_x * sum_y) / denom;
        // Convert from ns/ns to ns/s
        Some(slope_per_ns * 1_000_000_000.0)
    }

    /// Evaluate severity of the latest sample.
    pub fn severity(&self) -> DriftSeverity {
        let abs = match self.latest() {
            Some(s) => s.abs_offset_ns(),
            None => return DriftSeverity::Normal,
        };
        if abs >= self.config.alarm_threshold_ns {
            DriftSeverity::Alarm
        } else if abs >= self.config.warning_threshold_ns {
            DriftSeverity::Warning
        } else if abs >= self.config.elevated_threshold_ns {
            DriftSeverity::Elevated
        } else {
            DriftSeverity::Normal
        }
    }

    /// Clear all samples and reset counters.
    pub fn reset(&mut self) {
        self.samples.clear();
        self.peak_offset_ns = 0;
        self.alarm_count = 0;
    }

    /// Estimated time until the drift exceeds the alarm threshold.
    /// Returns None if drift rate is zero or trending downward.
    pub fn time_to_alarm(&self) -> Option<Duration> {
        let rate = self.drift_rate_ns_per_sec()?;
        if rate.abs() < f64::EPSILON {
            return None;
        }
        let current = self.latest()?.abs_offset_ns();
        if current >= self.config.alarm_threshold_ns {
            return Some(Duration::ZERO);
        }
        let remaining_ns = self.config.alarm_threshold_ns - current;
        #[allow(clippy::cast_precision_loss)]
        let secs = remaining_ns as f64 / rate.abs();
        if secs <= 0.0 || !secs.is_finite() {
            return None;
        }
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        Some(Duration::from_secs(secs as u64))
    }
}

impl Default for DriftMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_monitor(window: usize) -> DriftMonitor {
        DriftMonitor::with_config(DriftMonitorConfig {
            window_size: window,
            ..DriftMonitorConfig::default()
        })
    }

    #[test]
    fn test_empty_monitor() {
        let m = DriftMonitor::new();
        assert_eq!(m.sample_count(), 0);
        assert!(m.latest().is_none());
        assert!((m.mean_offset_ns() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_push_sample() {
        let mut m = make_monitor(10);
        m.push(DriftSample::new(100, 1_000));
        assert_eq!(m.sample_count(), 1);
        assert_eq!(m.latest().expect("should succeed in test").offset_ns, 100);
    }

    #[test]
    fn test_window_eviction() {
        let mut m = make_monitor(3);
        m.push(DriftSample::new(10, 1000));
        m.push(DriftSample::new(20, 2000));
        m.push(DriftSample::new(30, 3000));
        m.push(DriftSample::new(40, 4000));
        assert_eq!(m.sample_count(), 3);
        // Oldest (10) should have been evicted
        assert_eq!(m.min_offset_ns(), Some(20));
    }

    #[test]
    fn test_mean_offset() {
        let mut m = make_monitor(10);
        m.push(DriftSample::new(100, 1000));
        m.push(DriftSample::new(200, 2000));
        m.push(DriftSample::new(300, 3000));
        assert!((m.mean_offset_ns() - 200.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_stddev_offset() {
        let mut m = make_monitor(10);
        m.push(DriftSample::new(10, 1000));
        m.push(DriftSample::new(10, 2000));
        m.push(DriftSample::new(10, 3000));
        assert!((m.stddev_offset_ns() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_min_max_offset() {
        let mut m = make_monitor(10);
        m.push(DriftSample::new(-50, 1000));
        m.push(DriftSample::new(100, 2000));
        m.push(DriftSample::new(25, 3000));
        assert_eq!(m.min_offset_ns(), Some(-50));
        assert_eq!(m.max_offset_ns(), Some(100));
    }

    #[test]
    fn test_peak_offset() {
        let mut m = make_monitor(10);
        m.push(DriftSample::new(-200, 1000));
        m.push(DriftSample::new(100, 2000));
        assert_eq!(m.peak_offset_ns(), 200);
    }

    #[test]
    fn test_severity_normal() {
        let mut m = DriftMonitor::new();
        m.push(DriftSample::new(500, 1000)); // 500 ns < 1000 ns normal threshold
        assert_eq!(m.severity(), DriftSeverity::Normal);
    }

    #[test]
    fn test_severity_alarm() {
        let mut m = DriftMonitor::new();
        m.push(DriftSample::new(2_000_000, 1000)); // 2 ms > 1 ms alarm
        assert_eq!(m.severity(), DriftSeverity::Alarm);
        assert_eq!(m.alarm_count(), 1);
    }

    #[test]
    fn test_severity_warning() {
        let mut m = DriftMonitor::new();
        m.push(DriftSample::new(500_000, 1000)); // 500 us
        assert_eq!(m.severity(), DriftSeverity::Warning);
    }

    #[test]
    fn test_drift_rate_insufficient_samples() {
        let mut m = make_monitor(100);
        m.push(DriftSample::new(100, 1000));
        assert!(m.drift_rate_ns_per_sec().is_none());
    }

    #[test]
    fn test_drift_rate_linear() {
        let mut m = DriftMonitor::with_config(DriftMonitorConfig {
            window_size: 100,
            min_samples_for_rate: 3,
            ..DriftMonitorConfig::default()
        });
        // Simulate a linear drift: 1 ns per second offset growth
        // offset = t_ns / 1e9 ns  → slope = 1 ns/ns * 1e9 = 1 ns/s
        for i in 0..20 {
            let t_ns = i as u64 * 1_000_000_000; // i seconds in ns
            let offset_ns = i as i64; // 1 ns per second
            m.push(DriftSample::new(offset_ns, t_ns));
        }
        let rate = m.drift_rate_ns_per_sec().expect("should succeed in test");
        assert!((rate - 1.0).abs() < 0.1, "rate was {rate}");
    }

    #[test]
    fn test_reset() {
        let mut m = make_monitor(10);
        m.push(DriftSample::new(100, 1000));
        m.push(DriftSample::new(2_000_000, 2000));
        assert!(m.alarm_count() > 0);
        m.reset();
        assert_eq!(m.sample_count(), 0);
        assert_eq!(m.peak_offset_ns(), 0);
        assert_eq!(m.alarm_count(), 0);
    }

    #[test]
    fn test_severity_display() {
        assert_eq!(DriftSeverity::Normal.to_string(), "Normal");
        assert_eq!(DriftSeverity::Alarm.to_string(), "Alarm");
    }

    #[test]
    fn test_drift_sample_abs() {
        let s = DriftSample::new(-500, 0);
        assert_eq!(s.abs_offset_ns(), 500);
        let s2 = DriftSample::new(300, 0);
        assert_eq!(s2.abs_offset_ns(), 300);
    }

    #[test]
    fn test_severity_elevated() {
        let mut m = DriftMonitor::new();
        m.push(DriftSample::new(50_000, 1000)); // 50 us
        assert_eq!(m.severity(), DriftSeverity::Elevated);
    }

    #[test]
    fn test_default_config() {
        let cfg = DriftMonitorConfig::default();
        assert_eq!(cfg.window_size, 1000);
        assert_eq!(cfg.alarm_threshold_ns, 1_000_000);
    }
}
