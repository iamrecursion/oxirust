//! Time synchronisation quality metrics.
//!
//! Provides types for collecting and evaluating PTP/NTP synchronisation
//! measurements, computing statistics such as average offset, max jitter, and
//! percentile offsets, and producing a [`SyncHealthReport`].

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

// ---------------------------------------------------------------------------
// SyncQuality
// ---------------------------------------------------------------------------

/// Qualitative assessment of synchronisation quality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SyncQuality {
    /// |offset| < 100 ns — broadcast-grade PTP.
    Excellent,
    /// |offset| < 1 µs.
    Good,
    /// |offset| < 10 µs.
    Fair,
    /// |offset| < 1 ms.
    Poor,
    /// |offset| ≥ 1 ms.
    Unacceptable,
}

impl SyncQuality {
    /// Maximum absolute offset (nanoseconds) that still qualifies for this
    /// quality tier.
    #[must_use]
    pub fn max_offset_ns(&self) -> i64 {
        match self {
            SyncQuality::Excellent => 100,
            SyncQuality::Good => 1_000,
            SyncQuality::Fair => 10_000,
            SyncQuality::Poor => 1_000_000,
            SyncQuality::Unacceptable => i64::MAX,
        }
    }

    /// Classify an absolute offset in nanoseconds into a [`SyncQuality`] tier.
    #[must_use]
    pub fn from_offset_ns(offset_ns: i64) -> Self {
        let abs = offset_ns.unsigned_abs() as i64;
        if abs < 100 {
            SyncQuality::Excellent
        } else if abs < 1_000 {
            SyncQuality::Good
        } else if abs < 10_000 {
            SyncQuality::Fair
        } else if abs < 1_000_000 {
            SyncQuality::Poor
        } else {
            SyncQuality::Unacceptable
        }
    }
}

// ---------------------------------------------------------------------------
// SyncMeasurement
// ---------------------------------------------------------------------------

/// A single time-synchronisation sample.
#[derive(Debug, Clone, Copy)]
pub struct SyncMeasurement {
    /// Measured clock offset from the master, in nanoseconds.
    pub offset_ns: i64,
    /// Round-trip network delay, in nanoseconds.
    pub delay_ns: u64,
    /// Jitter (variation in delay), in nanoseconds.
    pub jitter_ns: u64,
    /// Wall-clock time when this measurement was taken, in milliseconds since
    /// the Unix epoch.
    pub timestamp_ms: u64,
}

impl SyncMeasurement {
    /// Create a new measurement.
    #[must_use]
    pub fn new(offset_ns: i64, delay_ns: u64, jitter_ns: u64, timestamp_ms: u64) -> Self {
        Self {
            offset_ns,
            delay_ns,
            jitter_ns,
            timestamp_ms,
        }
    }

    /// Quality tier of this measurement based on its offset.
    #[must_use]
    pub fn quality(&self) -> SyncQuality {
        SyncQuality::from_offset_ns(self.offset_ns)
    }
}

// ---------------------------------------------------------------------------
// SyncMetricsWindow
// ---------------------------------------------------------------------------

/// A sliding window of recent synchronisation measurements.
#[derive(Debug, Clone)]
pub struct SyncMetricsWindow {
    /// Stored measurements (oldest first).
    measurements: Vec<SyncMeasurement>,
    /// Maximum number of measurements to retain.
    max_size: usize,
}

impl SyncMetricsWindow {
    /// Create a new window with the given capacity.
    #[must_use]
    pub fn new(max_size: usize) -> Self {
        Self {
            measurements: Vec::with_capacity(max_size),
            max_size,
        }
    }

    /// Add a measurement to the window, evicting the oldest if full.
    pub fn add(&mut self, m: SyncMeasurement) {
        if self.measurements.len() >= self.max_size {
            self.measurements.remove(0);
        }
        self.measurements.push(m);
    }

    /// Number of measurements currently stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.measurements.len()
    }

    /// Returns `true` when no measurements have been added.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.measurements.is_empty()
    }

    /// Mean clock offset (nanoseconds) across all stored measurements.
    /// Returns `0.0` when the window is empty.
    #[must_use]
    pub fn avg_offset_ns(&self) -> f64 {
        if self.measurements.is_empty() {
            return 0.0;
        }
        let sum: i64 = self.measurements.iter().map(|m| m.offset_ns).sum();
        sum as f64 / self.measurements.len() as f64
    }

    /// Maximum jitter across all stored measurements.
    #[must_use]
    pub fn max_jitter_ns(&self) -> u64 {
        self.measurements
            .iter()
            .map(|m| m.jitter_ns)
            .max()
            .unwrap_or(0)
    }

    /// 99th-percentile absolute offset (nanoseconds).
    ///
    /// Uses the nearest-rank method. Returns `0` when the window is empty.
    #[must_use]
    pub fn p99_offset_ns(&self) -> i64 {
        if self.measurements.is_empty() {
            return 0;
        }
        let mut abs_offsets: Vec<i64> = self
            .measurements
            .iter()
            .map(|m| m.offset_ns.abs())
            .collect();
        abs_offsets.sort_unstable();
        let idx = ((abs_offsets.len() * 99).saturating_sub(1)) / 100;
        abs_offsets[idx.min(abs_offsets.len() - 1)]
    }

    /// The most recent measurement, or `None` if the window is empty.
    #[must_use]
    pub fn latest(&self) -> Option<&SyncMeasurement> {
        self.measurements.last()
    }
}

// ---------------------------------------------------------------------------
// SyncHealthReport
// ---------------------------------------------------------------------------

/// A summary report of synchronisation health over a measurement window.
#[derive(Debug, Clone)]
pub struct SyncHealthReport {
    /// Duration of the measurement window in seconds.
    pub window_seconds: u64,
    /// Mean clock offset (ns) across the window.
    pub avg_offset_ns: f64,
    /// 99th-percentile absolute offset (ns).
    pub p99_offset_ns: i64,
    /// Quality tier of the most recent measurement.
    pub current_quality: SyncQuality,
    /// `true` when `p99_offset_ns` is within the given spec.
    pub within_spec: bool,
}

impl SyncHealthReport {
    /// Generate a health report from a `window`.
    ///
    /// `spec_ns` is the maximum allowable absolute offset (nanoseconds) for
    /// the deployment to be considered within specification.
    #[must_use]
    pub fn generate(window: &SyncMetricsWindow, spec_ns: i64) -> Self {
        let avg_offset_ns = window.avg_offset_ns();
        let p99_offset_ns = window.p99_offset_ns();
        let current_quality = window
            .latest()
            .map(|m| m.quality())
            .unwrap_or(SyncQuality::Unacceptable);
        let within_spec = p99_offset_ns.abs() <= spec_ns;

        // Rough window duration: timestamps of first and last measurement.
        let window_seconds = if window.measurements.len() >= 2 {
            let first = window.measurements[0].timestamp_ms;
            let last = window.measurements[window.measurements.len() - 1].timestamp_ms;
            (last.saturating_sub(first)) / 1_000
        } else {
            0
        };

        Self {
            window_seconds,
            avg_offset_ns,
            p99_offset_ns,
            current_quality,
            within_spec,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_m(offset_ns: i64, jitter_ns: u64, ts_ms: u64) -> SyncMeasurement {
        SyncMeasurement::new(offset_ns, 500_000, jitter_ns, ts_ms)
    }

    // --- SyncQuality ---

    #[test]
    fn test_quality_excellent_boundary() {
        assert_eq!(SyncQuality::from_offset_ns(0), SyncQuality::Excellent);
        assert_eq!(SyncQuality::from_offset_ns(99), SyncQuality::Excellent);
        assert_eq!(SyncQuality::from_offset_ns(-99), SyncQuality::Excellent);
    }

    #[test]
    fn test_quality_good() {
        assert_eq!(SyncQuality::from_offset_ns(500), SyncQuality::Good);
        assert_eq!(SyncQuality::from_offset_ns(-500), SyncQuality::Good);
    }

    #[test]
    fn test_quality_fair() {
        assert_eq!(SyncQuality::from_offset_ns(5_000), SyncQuality::Fair);
    }

    #[test]
    fn test_quality_poor() {
        assert_eq!(SyncQuality::from_offset_ns(100_000), SyncQuality::Poor);
    }

    #[test]
    fn test_quality_unacceptable() {
        assert_eq!(
            SyncQuality::from_offset_ns(2_000_000),
            SyncQuality::Unacceptable
        );
    }

    #[test]
    fn test_max_offset_ns_excellent() {
        assert_eq!(SyncQuality::Excellent.max_offset_ns(), 100);
    }

    #[test]
    fn test_max_offset_ns_unacceptable() {
        assert_eq!(SyncQuality::Unacceptable.max_offset_ns(), i64::MAX);
    }

    // --- SyncMeasurement ---

    #[test]
    fn test_measurement_quality() {
        let m = make_m(50, 10, 0);
        assert_eq!(m.quality(), SyncQuality::Excellent);
    }

    #[test]
    fn test_measurement_quality_poor() {
        let m = make_m(500_000, 0, 0);
        assert_eq!(m.quality(), SyncQuality::Poor);
    }

    // --- SyncMetricsWindow ---

    #[test]
    fn test_window_empty() {
        let w = SyncMetricsWindow::new(10);
        assert!(w.is_empty());
        assert_eq!(w.avg_offset_ns(), 0.0);
        assert_eq!(w.max_jitter_ns(), 0);
        assert_eq!(w.p99_offset_ns(), 0);
    }

    #[test]
    fn test_window_avg_offset() {
        let mut w = SyncMetricsWindow::new(10);
        w.add(make_m(100, 0, 0));
        w.add(make_m(200, 0, 1000));
        w.add(make_m(300, 0, 2000));
        let avg = w.avg_offset_ns();
        assert!((avg - 200.0).abs() < 1.0);
    }

    #[test]
    fn test_window_max_jitter() {
        let mut w = SyncMetricsWindow::new(10);
        w.add(make_m(0, 10, 0));
        w.add(make_m(0, 50, 1000));
        w.add(make_m(0, 30, 2000));
        assert_eq!(w.max_jitter_ns(), 50);
    }

    #[test]
    fn test_window_eviction() {
        let mut w = SyncMetricsWindow::new(3);
        for i in 0..5u64 {
            w.add(make_m(i as i64, 0, i * 1000));
        }
        assert_eq!(w.len(), 3);
    }

    #[test]
    fn test_window_p99_single_element() {
        let mut w = SyncMetricsWindow::new(10);
        w.add(make_m(1234, 0, 0));
        assert_eq!(w.p99_offset_ns(), 1234);
    }

    #[test]
    fn test_window_p99_multiple_elements() {
        let mut w = SyncMetricsWindow::new(100);
        for i in 1_i64..=100 {
            w.add(make_m(i * 10, 0, i as u64 * 1000));
        }
        // p99 should be close to the 99th value: 990
        let p99 = w.p99_offset_ns();
        assert!(p99 >= 980 && p99 <= 1000);
    }

    #[test]
    fn test_window_latest() {
        let mut w = SyncMetricsWindow::new(5);
        w.add(make_m(100, 0, 0));
        w.add(make_m(200, 0, 1000));
        assert_eq!(w.latest().expect("should succeed in test").offset_ns, 200);
    }

    // --- SyncHealthReport ---

    #[test]
    fn test_report_within_spec() {
        let mut w = SyncMetricsWindow::new(100);
        for i in 0..50_i64 {
            w.add(make_m(i, 5, i as u64 * 1000));
        }
        let report = SyncHealthReport::generate(&w, 1_000);
        assert!(report.within_spec);
        assert_eq!(report.current_quality, SyncQuality::Excellent);
    }

    #[test]
    fn test_report_outside_spec() {
        let mut w = SyncMetricsWindow::new(10);
        w.add(make_m(2_000_000, 0, 0));
        w.add(make_m(2_000_000, 0, 5000));
        let report = SyncHealthReport::generate(&w, 1_000);
        assert!(!report.within_spec);
        assert_eq!(report.current_quality, SyncQuality::Unacceptable);
    }

    #[test]
    fn test_report_window_duration() {
        let mut w = SyncMetricsWindow::new(10);
        w.add(make_m(0, 0, 0));
        w.add(make_m(0, 0, 10_000)); // 10 s later
        let report = SyncHealthReport::generate(&w, 1_000_000);
        assert_eq!(report.window_seconds, 10);
    }
}
