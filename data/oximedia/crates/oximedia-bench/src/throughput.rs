//! Throughput benchmarking for media processing pipelines.
//!
//! This module provides tools for measuring and comparing throughput across
//! different processing operations, including frames-per-second, bitrate,
//! and pixel throughput.

/// Unit for expressing throughput measurements.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub enum ThroughputUnit {
    /// Frames per second.
    FramesPerSec,
    /// Megabits per second.
    MbitsPerSec,
    /// Gigabytes per second.
    GbytesPerSec,
    /// Megapixels per second.
    Mpixels,
}

impl ThroughputUnit {
    /// Format a value in this unit as a human-readable string.
    #[must_use]
    pub fn format(&self, value: f64) -> String {
        match self {
            Self::FramesPerSec => format!("{value:.2} fps"),
            Self::MbitsPerSec => format!("{value:.2} Mbps"),
            Self::GbytesPerSec => format!("{value:.3} GB/s"),
            Self::Mpixels => format!("{value:.2} Mpix/s"),
        }
    }

    /// Short unit label.
    #[must_use]
    pub fn label(&self) -> &str {
        match self {
            Self::FramesPerSec => "fps",
            Self::MbitsPerSec => "Mbps",
            Self::GbytesPerSec => "GB/s",
            Self::Mpixels => "Mpix/s",
        }
    }
}

/// A single throughput measurement.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ThroughputMeasurement {
    /// Unit of the measurement.
    pub unit: ThroughputUnit,
    /// Computed throughput value.
    pub value: f64,
    /// Wall-clock duration of the measurement window in milliseconds.
    pub duration_ms: u64,
    /// Number of work items (frames, bytes, pixels, …) processed.
    pub work_items: u64,
}

impl ThroughputMeasurement {
    /// Compute throughput from timing data.
    ///
    /// `work` is the number of items processed; `elapsed_ms` is the wall-clock
    /// duration of the measurement window.
    #[must_use]
    pub fn from_timing(unit: ThroughputUnit, work: u64, elapsed_ms: u64) -> Self {
        let value = if elapsed_ms == 0 {
            0.0
        } else {
            work as f64 / (elapsed_ms as f64 / 1000.0)
        };
        Self {
            unit,
            value,
            duration_ms: elapsed_ms,
            work_items: work,
        }
    }
}

/// A named collection of throughput measurements for statistical summary.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ThroughputBenchmark {
    /// Name / label for this benchmark.
    pub name: String,
    /// Individual measurements collected during the benchmark.
    pub measurements: Vec<ThroughputMeasurement>,
}

impl ThroughputBenchmark {
    /// Create a new (empty) benchmark.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            measurements: Vec::new(),
        }
    }

    /// Add a measurement.
    pub fn push(&mut self, m: ThroughputMeasurement) {
        self.measurements.push(m);
    }

    /// Arithmetic mean of the measured throughput values.
    ///
    /// Returns `0.0` if there are no measurements.
    #[must_use]
    pub fn mean(&self) -> f64 {
        if self.measurements.is_empty() {
            return 0.0;
        }
        self.measurements.iter().map(|m| m.value).sum::<f64>() / self.measurements.len() as f64
    }

    /// 95th percentile of the measured throughput values.
    #[must_use]
    pub fn p95(&self) -> f64 {
        percentile_of(&self.values_sorted(), 95.0)
    }

    /// 99th percentile of the measured throughput values.
    #[must_use]
    pub fn p99(&self) -> f64 {
        percentile_of(&self.values_sorted(), 99.0)
    }

    fn values_sorted(&self) -> Vec<f64> {
        let mut v: Vec<f64> = self.measurements.iter().map(|m| m.value).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        v
    }
}

/// Compute a percentile from a pre-sorted slice.
fn percentile_of(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let raw = p / 100.0 * sorted.len() as f64;
    let idx = if raw.fract() < f64::EPSILON {
        (raw as usize).saturating_sub(1)
    } else {
        raw.ceil() as usize - 1
    }
    .min(sorted.len() - 1);
    sorted[idx]
}

/// Comparison utilities for throughput benchmarks.
pub struct ThroughputComparison;

impl ThroughputComparison {
    /// Geometric-mean speedup of `candidate` relative to `baseline`.
    ///
    /// A value > 1.0 indicates the candidate is faster.
    /// Returns `1.0` when either benchmark is empty or the baseline mean is zero.
    #[must_use]
    pub fn speedup(baseline: &ThroughputBenchmark, candidate: &ThroughputBenchmark) -> f64 {
        let b = baseline.mean();
        let c = candidate.mean();
        if b.abs() < f64::EPSILON {
            return 1.0;
        }
        c / b
    }
}

/// Simulated sustained throughput test.
///
/// Models a processing loop running for a fixed wall-clock duration, counting
/// how many frames complete within that window given a target FPS and a fixed
/// per-frame processing cost.
#[derive(Debug, Clone, Default)]
pub struct SustainedThroughputTest;

impl SustainedThroughputTest {
    /// Run a simulated sustained throughput test.
    ///
    /// - `duration_ms`: how long the simulated window runs.
    /// - `target_fps`: the ideal frame rate.
    /// - `frame_ms`: actual processing time per frame in milliseconds.
    ///
    /// Returns a `ThroughputMeasurement` reflecting the achieved frame rate.
    #[must_use]
    pub fn simulate(
        &self,
        duration_ms: u64,
        target_fps: f32,
        frame_ms: f32,
    ) -> ThroughputMeasurement {
        if frame_ms <= 0.0 || duration_ms == 0 {
            return ThroughputMeasurement::from_timing(
                ThroughputUnit::FramesPerSec,
                0,
                duration_ms,
            );
        }

        // Frames completed = floor(duration / frame_ms), bounded by target.
        let ideal_interval_ms = 1000.0 / target_fps;
        let effective_interval_ms = frame_ms.max(ideal_interval_ms);
        let frames_completed = (duration_ms as f32 / effective_interval_ms).floor() as u64;

        ThroughputMeasurement::from_timing(
            ThroughputUnit::FramesPerSec,
            frames_completed,
            duration_ms,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unit_format_fps() {
        assert_eq!(ThroughputUnit::FramesPerSec.format(60.0), "60.00 fps");
    }

    #[test]
    fn test_unit_format_mbps() {
        assert_eq!(ThroughputUnit::MbitsPerSec.format(1000.0), "1000.00 Mbps");
    }

    #[test]
    fn test_unit_format_gbytes() {
        assert_eq!(ThroughputUnit::GbytesPerSec.format(1.5), "1.500 GB/s");
    }

    #[test]
    fn test_unit_format_mpixels() {
        assert_eq!(ThroughputUnit::Mpixels.format(248.83), "248.83 Mpix/s");
    }

    #[test]
    fn test_from_timing_basic() {
        let m = ThroughputMeasurement::from_timing(ThroughputUnit::FramesPerSec, 600, 10_000);
        assert!((m.value - 60.0).abs() < 1e-9);
    }

    #[test]
    fn test_from_timing_zero_elapsed() {
        let m = ThroughputMeasurement::from_timing(ThroughputUnit::FramesPerSec, 100, 0);
        assert_eq!(m.value, 0.0);
    }

    #[test]
    fn test_benchmark_mean() {
        let mut bench = ThroughputBenchmark::new("test");
        for fps in [30.0, 60.0, 90.0] {
            bench.push(ThroughputMeasurement::from_timing(
                ThroughputUnit::FramesPerSec,
                (fps * 10.0) as u64,
                10_000,
            ));
        }
        assert!((bench.mean() - 60.0).abs() < 1e-9);
    }

    #[test]
    fn test_benchmark_mean_empty() {
        let bench = ThroughputBenchmark::new("empty");
        assert_eq!(bench.mean(), 0.0);
    }

    #[test]
    fn test_benchmark_p95() {
        let mut bench = ThroughputBenchmark::new("p95");
        for i in 1u64..=100 {
            bench.push(ThroughputMeasurement::from_timing(
                ThroughputUnit::FramesPerSec,
                i * 10,
                10_000,
            ));
        }
        let p95 = bench.p95();
        assert!(p95 > bench.mean()); // p95 should be above the mean
    }

    #[test]
    fn test_benchmark_p99() {
        let mut bench = ThroughputBenchmark::new("p99");
        for i in 1u64..=100 {
            bench.push(ThroughputMeasurement::from_timing(
                ThroughputUnit::FramesPerSec,
                i * 10,
                10_000,
            ));
        }
        assert!(bench.p99() >= bench.p95());
    }

    #[test]
    fn test_speedup_faster_candidate() {
        let mut baseline = ThroughputBenchmark::new("baseline");
        baseline.push(ThroughputMeasurement::from_timing(
            ThroughputUnit::FramesPerSec,
            300,
            10_000,
        ));
        let mut candidate = ThroughputBenchmark::new("candidate");
        candidate.push(ThroughputMeasurement::from_timing(
            ThroughputUnit::FramesPerSec,
            600,
            10_000,
        ));
        let speedup = ThroughputComparison::speedup(&baseline, &candidate);
        assert!((speedup - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_speedup_zero_baseline() {
        let baseline = ThroughputBenchmark::new("zero");
        let candidate = ThroughputBenchmark::new("cand");
        assert_eq!(ThroughputComparison::speedup(&baseline, &candidate), 1.0);
    }

    #[test]
    fn test_sustained_simulate_basic() {
        let test = SustainedThroughputTest;
        // 30 fps target, 33 ms per frame, 1 second window => ~30 frames
        let m = test.simulate(1_000, 30.0, 33.3);
        assert!(m.work_items > 0 && m.work_items <= 31);
    }

    #[test]
    fn test_sustained_simulate_zero_frame_ms() {
        let test = SustainedThroughputTest;
        let m = test.simulate(1_000, 60.0, 0.0);
        assert_eq!(m.work_items, 0);
    }

    #[test]
    fn test_unit_label() {
        assert_eq!(ThroughputUnit::FramesPerSec.label(), "fps");
        assert_eq!(ThroughputUnit::MbitsPerSec.label(), "Mbps");
    }
}
