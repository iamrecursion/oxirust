//! NDI stream statistics collection for `oximedia-ndi`.
//!
//! Provides per-stream counters and a rolling-window throughput tracker that
//! can be queried for averages and totals.

#![allow(dead_code)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::cast_precision_loss)]

// ---------------------------------------------------------------------------
// StreamCounters
// ---------------------------------------------------------------------------

/// Cumulative counters for a single NDI stream direction (send or receive).
#[derive(Debug, Clone, Default)]
pub struct StreamCounters {
    /// Total frames processed (sent or received).
    pub frames: u64,
    /// Total bytes transferred.
    pub bytes: u64,
    /// Frames that were dropped.
    pub dropped: u64,
    /// Frames that arrived / were sent late.
    pub late: u64,
    /// Number of connection resets since the counter was created.
    pub resets: u64,
}

impl StreamCounters {
    /// Create zeroed counters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a successfully transferred frame of `byte_count` bytes.
    pub fn record_frame(&mut self, byte_count: u64) {
        self.frames += 1;
        self.bytes += byte_count;
    }

    /// Record a dropped frame.
    pub fn record_drop(&mut self) {
        self.dropped += 1;
    }

    /// Record a late frame (also counted towards `frames`).
    pub fn record_late(&mut self, byte_count: u64) {
        self.frames += 1;
        self.bytes += byte_count;
        self.late += 1;
    }

    /// Record a connection reset event.
    pub fn record_reset(&mut self) {
        self.resets += 1;
    }

    /// Fraction of frames that were dropped [0.0, 1.0].
    ///
    /// Returns `0.0` when no frames have been seen.
    pub fn drop_rate(&self) -> f64 {
        let total = self.frames + self.dropped;
        if total == 0 {
            return 0.0;
        }
        self.dropped as f64 / total as f64
    }

    /// Fraction of received/sent frames that were late [0.0, 1.0].
    pub fn late_rate(&self) -> f64 {
        if self.frames == 0 {
            return 0.0;
        }
        self.late as f64 / self.frames as f64
    }
}

// ---------------------------------------------------------------------------
// ThroughputSample
// ---------------------------------------------------------------------------

/// A single throughput measurement in a rolling window.
#[derive(Debug, Clone, Copy)]
pub struct ThroughputSample {
    /// Timestamp in milliseconds when this measurement was taken.
    pub timestamp_ms: u64,
    /// Bytes transferred in this measurement interval.
    pub bytes: u64,
    /// Duration of the measurement interval in milliseconds.
    pub interval_ms: u64,
}

impl ThroughputSample {
    /// Create a new sample.
    pub fn new(timestamp_ms: u64, bytes: u64, interval_ms: u64) -> Self {
        Self {
            timestamp_ms,
            bytes,
            interval_ms,
        }
    }

    /// Throughput in kilobits per second for this sample.
    pub fn kbps(&self) -> f64 {
        if self.interval_ms == 0 {
            return 0.0;
        }
        (self.bytes as f64 * 8.0) / (self.interval_ms as f64)
    }
}

// ---------------------------------------------------------------------------
// ThroughputTracker
// ---------------------------------------------------------------------------

/// Rolling-window throughput tracker that keeps up to `window_size` samples.
#[derive(Debug, Clone)]
pub struct ThroughputTracker {
    samples: Vec<ThroughputSample>,
    window_size: usize,
}

impl ThroughputTracker {
    /// Create a new tracker with `window_size` sample capacity.
    pub fn new(window_size: usize) -> Self {
        Self {
            samples: Vec::new(),
            window_size: window_size.max(1),
        }
    }

    /// Record a new throughput sample.
    pub fn record(&mut self, sample: ThroughputSample) {
        if self.samples.len() >= self.window_size {
            self.samples.remove(0);
        }
        self.samples.push(sample);
    }

    /// Average throughput across all samples in the window, in kbps.
    pub fn avg_kbps(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.samples.iter().map(|s| s.kbps()).sum();
        sum / self.samples.len() as f64
    }

    /// Peak throughput observed in the current window, in kbps.
    pub fn peak_kbps(&self) -> f64 {
        self.samples
            .iter()
            .map(|s| s.kbps())
            .fold(0.0_f64, f64::max)
    }

    /// Total bytes across all samples in the window.
    pub fn total_bytes(&self) -> u64 {
        self.samples.iter().map(|s| s.bytes).sum()
    }

    /// Number of samples currently in the window.
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_counters_default_zeros() {
        let c = StreamCounters::new();
        assert_eq!(c.frames, 0);
        assert_eq!(c.bytes, 0);
        assert_eq!(c.dropped, 0);
    }

    #[test]
    fn test_counters_record_frame() {
        let mut c = StreamCounters::new();
        c.record_frame(1000);
        assert_eq!(c.frames, 1);
        assert_eq!(c.bytes, 1000);
    }

    #[test]
    fn test_counters_record_drop() {
        let mut c = StreamCounters::new();
        c.record_frame(100);
        c.record_drop();
        assert_eq!(c.dropped, 1);
    }

    #[test]
    fn test_counters_drop_rate() {
        let mut c = StreamCounters::new();
        c.record_frame(100); // 1 good
        c.record_drop(); // 1 dropped → 1/(1+1) = 0.5
        assert!((c.drop_rate() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_counters_drop_rate_zero_when_no_frames() {
        let c = StreamCounters::new();
        assert_eq!(c.drop_rate(), 0.0);
    }

    #[test]
    fn test_counters_record_late() {
        let mut c = StreamCounters::new();
        c.record_late(200);
        assert_eq!(c.frames, 1);
        assert_eq!(c.late, 1);
    }

    #[test]
    fn test_counters_late_rate() {
        let mut c = StreamCounters::new();
        c.record_frame(100);
        c.record_late(100);
        assert!((c.late_rate() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_counters_record_reset() {
        let mut c = StreamCounters::new();
        c.record_reset();
        c.record_reset();
        assert_eq!(c.resets, 2);
    }

    #[test]
    fn test_sample_kbps() {
        // 1000 bytes in 1000 ms = 8 kbps
        let s = ThroughputSample::new(0, 1000, 1000);
        assert!((s.kbps() - 8.0).abs() < 1e-9);
    }

    #[test]
    fn test_sample_kbps_zero_interval() {
        let s = ThroughputSample::new(0, 1000, 0);
        assert_eq!(s.kbps(), 0.0);
    }

    #[test]
    fn test_tracker_avg_kbps_empty() {
        let t = ThroughputTracker::new(5);
        assert_eq!(t.avg_kbps(), 0.0);
    }

    #[test]
    fn test_tracker_avg_kbps() {
        let mut t = ThroughputTracker::new(5);
        t.record(ThroughputSample::new(0, 1000, 1000)); // 8 kbps
        t.record(ThroughputSample::new(1000, 2000, 1000)); // 16 kbps
        assert!((t.avg_kbps() - 12.0).abs() < 1e-6);
    }

    #[test]
    fn test_tracker_peak_kbps() {
        let mut t = ThroughputTracker::new(5);
        t.record(ThroughputSample::new(0, 1000, 1000)); // 8 kbps
        t.record(ThroughputSample::new(1000, 3000, 1000)); // 24 kbps
        assert!((t.peak_kbps() - 24.0).abs() < 1e-6);
    }

    #[test]
    fn test_tracker_window_eviction() {
        let mut t = ThroughputTracker::new(2);
        t.record(ThroughputSample::new(0, 100, 1000));
        t.record(ThroughputSample::new(1000, 200, 1000));
        t.record(ThroughputSample::new(2000, 300, 1000)); // evicts first
        assert_eq!(t.sample_count(), 2);
    }

    #[test]
    fn test_tracker_total_bytes() {
        let mut t = ThroughputTracker::new(5);
        t.record(ThroughputSample::new(0, 500, 1000));
        t.record(ThroughputSample::new(1000, 300, 1000));
        assert_eq!(t.total_bytes(), 800);
    }
}
