//! Throughput and bandwidth profiling.
//!
//! Measure bytes-per-second and frames-per-second throughput over a sliding
//! time window and detect bottlenecks across pipeline stages.

/// A single throughput measurement.
#[derive(Debug, Clone)]
pub struct ThroughputSample {
    /// Millisecond timestamp when this sample was taken.
    pub timestamp_ms: u64,
    /// Number of bytes transferred since the previous sample.
    pub bytes: u64,
    /// Number of frames processed since the previous sample.
    pub frames: u32,
}

impl ThroughputSample {
    /// Create a new throughput sample.
    pub fn new(timestamp_ms: u64, bytes: u64, frames: u32) -> Self {
        Self {
            timestamp_ms,
            bytes,
            frames,
        }
    }

    /// Return the bandwidth in Mbps given that `duration_ms` elapsed.
    ///
    /// Returns `0.0` if `duration_ms` is zero.
    pub fn bandwidth_mbps(&self, duration_ms: u64) -> f64 {
        if duration_ms == 0 {
            return 0.0;
        }
        #[allow(clippy::cast_precision_loss)]
        let bits = (self.bytes * 8) as f64;
        #[allow(clippy::cast_precision_loss)]
        let duration_s = duration_ms as f64 / 1000.0;
        bits / duration_s / 1_000_000.0
    }

    /// Return the frame rate given that `duration_ms` elapsed.
    ///
    /// Returns `0.0` if `duration_ms` is zero.
    pub fn fps(&self, duration_ms: u64) -> f64 {
        if duration_ms == 0 {
            return 0.0;
        }
        #[allow(clippy::cast_precision_loss)]
        let frames = self.frames as f64;
        #[allow(clippy::cast_precision_loss)]
        let duration_s = duration_ms as f64 / 1000.0;
        frames / duration_s
    }
}

/// Accumulates throughput samples and provides windowed statistics.
#[derive(Debug)]
pub struct ThroughputProfiler {
    samples: Vec<ThroughputSample>,
    /// Width of the sliding window in milliseconds.
    pub window_ms: u64,
}

impl ThroughputProfiler {
    /// Create a new profiler with the given sliding-window width.
    pub fn new(window_ms: u64) -> Self {
        Self {
            samples: vec![],
            window_ms,
        }
    }

    /// Record a new sample at time `now_ms`.
    pub fn record(&mut self, bytes: u64, frames: u32, now_ms: u64) {
        self.samples
            .push(ThroughputSample::new(now_ms, bytes, frames));
    }

    /// Return all samples that fall within the current sliding window.
    pub fn samples_in_window(&self, now_ms: u64) -> Vec<&ThroughputSample> {
        let cutoff = now_ms.saturating_sub(self.window_ms);
        self.samples
            .iter()
            .filter(|s| s.timestamp_ms >= cutoff)
            .collect()
    }

    /// Return the current bandwidth in Mbps computed over the sliding window.
    pub fn current_bandwidth_mbps(&self, now_ms: u64) -> f64 {
        let window = self.samples_in_window(now_ms);
        if window.is_empty() || self.window_ms == 0 {
            return 0.0;
        }
        #[allow(clippy::cast_precision_loss)]
        let total_bytes: u64 = window.iter().map(|s| s.bytes).sum();
        ThroughputSample::new(0, total_bytes, 0).bandwidth_mbps(self.window_ms)
    }

    /// Return the current frame rate computed over the sliding window.
    pub fn current_fps(&self, now_ms: u64) -> f64 {
        let window = self.samples_in_window(now_ms);
        if window.is_empty() || self.window_ms == 0 {
            return 0.0;
        }
        #[allow(clippy::cast_precision_loss)]
        let total_frames: u32 = window
            .iter()
            .map(|s| s.frames)
            .fold(0u32, |acc, f| acc.saturating_add(f));
        ThroughputSample::new(0, 0, total_frames).fps(self.window_ms)
    }

    /// Return the peak bandwidth (Mbps) ever observed across all individual samples.
    ///
    /// Each sample is assessed over a 1-second equivalent window (1000 ms) to
    /// produce a per-sample rate.
    pub fn peak_bandwidth_mbps(&self) -> f64 {
        self.samples
            .iter()
            .map(|s| s.bandwidth_mbps(1000))
            .fold(0.0f64, f64::max)
    }

    /// Return the total number of samples recorded.
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }
}

/// Utilisation analysis for a single pipeline stage.
#[derive(Debug, Clone)]
pub struct BottleneckAnalysis {
    /// Name of the pipeline stage.
    pub stage: String,
    /// Utilisation as a fraction in `[0.0, 1.0]`.
    pub utilization: f32,
    /// Whether this stage is considered the bottleneck.
    pub is_bottleneck: bool,
}

impl BottleneckAnalysis {
    /// Create a new analysis record.
    pub fn new(stage: &str, utilization: f32) -> Self {
        Self {
            stage: stage.to_string(),
            utilization,
            is_bottleneck: false,
        }
    }

    /// Return a reference to the stage that is the bottleneck in `analyses`, if any.
    ///
    /// The bottleneck is the stage with the highest utilisation.
    /// Returns `None` if `analyses` is empty.
    pub fn find_bottleneck(analyses: &[BottleneckAnalysis]) -> Option<&BottleneckAnalysis> {
        analyses.iter().max_by(|a, b| {
            a.utilization
                .partial_cmp(&b.utilization)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_bandwidth_mbps_basic() {
        // 1 MB in 1000 ms => 8 Mbps
        let s = ThroughputSample::new(0, 1_000_000, 0);
        let mbps = s.bandwidth_mbps(1000);
        assert!((mbps - 8.0).abs() < 0.001);
    }

    #[test]
    fn test_sample_bandwidth_mbps_zero_duration() {
        let s = ThroughputSample::new(0, 1000, 0);
        assert_eq!(s.bandwidth_mbps(0), 0.0);
    }

    #[test]
    fn test_sample_fps_basic() {
        // 25 frames in 1000 ms => 25 fps
        let s = ThroughputSample::new(0, 0, 25);
        let fps = s.fps(1000);
        assert!((fps - 25.0).abs() < 0.001);
    }

    #[test]
    fn test_sample_fps_zero_duration() {
        let s = ThroughputSample::new(0, 0, 30);
        assert_eq!(s.fps(0), 0.0);
    }

    #[test]
    fn test_profiler_record_and_count() {
        let mut p = ThroughputProfiler::new(1000);
        p.record(1000, 25, 0);
        p.record(2000, 25, 500);
        assert_eq!(p.sample_count(), 2);
    }

    #[test]
    fn test_samples_in_window_filters_old() {
        let mut p = ThroughputProfiler::new(500);
        p.record(1000, 25, 0); // age 1000 from now=1000 => outside window
        p.record(1000, 25, 600); // age  400 from now=1000 => inside window
        p.record(1000, 25, 900); // age  100 from now=1000 => inside window
        let w = p.samples_in_window(1000);
        assert_eq!(w.len(), 2);
    }

    #[test]
    fn test_current_bandwidth_mbps_no_samples() {
        let p = ThroughputProfiler::new(1000);
        assert_eq!(p.current_bandwidth_mbps(1000), 0.0);
    }

    #[test]
    fn test_current_bandwidth_mbps_with_samples() {
        let mut p = ThroughputProfiler::new(1000);
        // 1 MB in the window => 8 Mbps over 1 s window
        p.record(1_000_000, 0, 500);
        let mbps = p.current_bandwidth_mbps(1000);
        assert!((mbps - 8.0).abs() < 0.001);
    }

    #[test]
    fn test_current_fps_with_samples() {
        let mut p = ThroughputProfiler::new(1000);
        p.record(0, 25, 500);
        let fps = p.current_fps(1000);
        assert!((fps - 25.0).abs() < 0.001);
    }

    #[test]
    fn test_peak_bandwidth_mbps_selects_max() {
        let mut p = ThroughputProfiler::new(5000);
        p.record(500_000, 0, 0); // 4 Mbps
        p.record(2_000_000, 0, 0); // 16 Mbps
        p.record(1_000_000, 0, 0); // 8 Mbps
        let peak = p.peak_bandwidth_mbps();
        assert!((peak - 16.0).abs() < 0.001);
    }

    #[test]
    fn test_peak_bandwidth_mbps_no_samples() {
        let p = ThroughputProfiler::new(1000);
        assert_eq!(p.peak_bandwidth_mbps(), 0.0);
    }

    #[test]
    fn test_bottleneck_find_bottleneck_single() {
        let analyses = vec![BottleneckAnalysis::new("encode", 0.9)];
        let b = BottleneckAnalysis::find_bottleneck(&analyses).expect("should succeed in test");
        assert_eq!(b.stage, "encode");
    }

    #[test]
    fn test_bottleneck_find_bottleneck_multiple() {
        let analyses = vec![
            BottleneckAnalysis::new("ingest", 0.5),
            BottleneckAnalysis::new("decode", 0.95),
            BottleneckAnalysis::new("encode", 0.7),
        ];
        let b = BottleneckAnalysis::find_bottleneck(&analyses).expect("should succeed in test");
        assert_eq!(b.stage, "decode");
    }

    #[test]
    fn test_bottleneck_find_bottleneck_empty() {
        let analyses: Vec<BottleneckAnalysis> = vec![];
        assert!(BottleneckAnalysis::find_bottleneck(&analyses).is_none());
    }
}
