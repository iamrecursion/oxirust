#![allow(dead_code)]
//! Memory-bandwidth measurement utilities for `oximedia-accel`.
//!
//! Provides synthetic benchmark helpers that measure achievable memory
//! throughput for sequential reads, writes, and copy operations.  Results
//! are used by the scheduler and profile selector to make better dispatch
//! decisions.

use std::time::Instant;

/// A single bandwidth measurement.
#[derive(Debug, Clone)]
pub struct BandwidthResult {
    /// Label describing what was measured (e.g. `"seq_read"`).
    pub label: String,
    /// Number of bytes transferred.
    pub bytes: u64,
    /// Elapsed time in seconds.
    pub elapsed_secs: f64,
    /// Computed throughput in MB/s.
    pub throughput_mbs: f64,
}

impl BandwidthResult {
    /// Create a result from a label, bytes transferred, and elapsed seconds.
    pub fn new(label: impl Into<String>, bytes: u64, elapsed_secs: f64) -> Self {
        let throughput_mbs = if elapsed_secs > 0.0 {
            (bytes as f64) / (elapsed_secs * 1_048_576.0)
        } else {
            0.0
        };
        Self {
            label: label.into(),
            bytes,
            elapsed_secs,
            throughput_mbs,
        }
    }

    /// Returns `true` if the measured throughput meets the given minimum (MB/s).
    #[must_use]
    pub fn meets_minimum(&self, min_mbs: f64) -> bool {
        self.throughput_mbs >= min_mbs
    }
}

/// Configuration for a bandwidth benchmark run.
#[derive(Debug, Clone)]
pub struct BandwidthTest {
    /// Number of bytes to use per test buffer.
    pub buffer_size: usize,
    /// Number of iterations to average over.
    pub iterations: u32,
    /// Whether to include a copy (read+write) test.
    pub include_copy: bool,
}

impl Default for BandwidthTest {
    fn default() -> Self {
        Self {
            buffer_size: 64 * 1024 * 1024, // 64 MiB
            iterations: 3,
            include_copy: true,
        }
    }
}

impl BandwidthTest {
    /// Create a lightweight test suitable for unit tests.
    #[must_use]
    pub fn lightweight() -> Self {
        Self {
            buffer_size: 1024 * 1024, // 1 MiB
            iterations: 1,
            include_copy: true,
        }
    }
}

/// Aggregated bandwidth profiling results.
#[derive(Debug)]
pub struct BandwidthProfiler {
    test: BandwidthTest,
    results: Vec<BandwidthResult>,
}

impl BandwidthProfiler {
    /// Create a profiler with the given test configuration.
    #[must_use]
    pub fn new(test: BandwidthTest) -> Self {
        Self {
            test,
            results: Vec::new(),
        }
    }

    /// Create a profiler with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(BandwidthTest::default())
    }

    /// Run the benchmark and populate internal results.
    ///
    /// This performs in-process CPU memory bandwidth measurements.  GPU
    /// bandwidth measurements require a separate compute pass and are not
    /// included here.
    pub fn benchmark(&mut self) {
        self.results.clear();

        let size = self.test.buffer_size;
        let iters = self.test.iterations;

        // Sequential read: sum all bytes in a buffer.
        let read_result = self.bench_sequential_read(size, iters);
        self.results.push(read_result);

        // Sequential write: fill a buffer.
        let write_result = self.bench_sequential_write(size, iters);
        self.results.push(write_result);

        // Copy: memcpy-equivalent.
        if self.test.include_copy {
            let copy_result = self.bench_copy(size, iters);
            self.results.push(copy_result);
        }
    }

    /// Return all benchmark results collected so far.
    #[must_use]
    pub fn results(&self) -> &[BandwidthResult] {
        &self.results
    }

    /// Return the peak throughput (MB/s) across all results.
    pub fn peak_throughput_mbs(&self) -> f64 {
        self.results
            .iter()
            .map(|r| r.throughput_mbs)
            .fold(0.0_f64, f64::max)
    }

    /// Return the average throughput (MB/s) across all results.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn avg_throughput_mbs(&self) -> f64 {
        if self.results.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.results.iter().map(|r| r.throughput_mbs).sum();
        sum / self.results.len() as f64
    }

    /// Find the result for a specific label.
    #[must_use]
    pub fn result_for(&self, label: &str) -> Option<&BandwidthResult> {
        self.results.iter().find(|r| r.label == label)
    }

    // ── private helpers ──────────────────────────────────────────────────────

    #[allow(clippy::cast_precision_loss)]
    fn bench_sequential_read(&self, size: usize, iters: u32) -> BandwidthResult {
        let buf: Vec<u8> = (0..size).map(|i| (i & 0xFF) as u8).collect();
        let total_bytes = size as u64 * u64::from(iters);

        let start = Instant::now();
        let mut checksum: u64 = 0;
        for _ in 0..iters {
            checksum = checksum.wrapping_add(buf.iter().map(|&b| u64::from(b)).sum::<u64>());
        }
        let elapsed = start.elapsed().as_secs_f64();

        // Use checksum to prevent the loop from being optimised away.
        let _ = checksum;
        BandwidthResult::new("seq_read", total_bytes, elapsed)
    }

    #[allow(clippy::cast_precision_loss)]
    fn bench_sequential_write(&self, size: usize, iters: u32) -> BandwidthResult {
        let mut buf = vec![0u8; size];
        let total_bytes = size as u64 * u64::from(iters);

        let start = Instant::now();
        for i in 0..iters {
            let val = (i & 0xFF) as u8;
            buf.fill(val);
        }
        let elapsed = start.elapsed().as_secs_f64();

        let _ = buf[0]; // prevent optimisation
        BandwidthResult::new("seq_write", total_bytes, elapsed)
    }

    #[allow(clippy::cast_precision_loss)]
    fn bench_copy(&self, size: usize, iters: u32) -> BandwidthResult {
        let src: Vec<u8> = (0..size).map(|i| (i & 0xFF) as u8).collect();
        let mut dst = vec![0u8; size];
        let total_bytes = size as u64 * u64::from(iters) * 2; // read + write

        let start = Instant::now();
        for _ in 0..iters {
            dst.copy_from_slice(&src);
        }
        let elapsed = start.elapsed().as_secs_f64();

        let _ = dst[0];
        BandwidthResult::new("copy", total_bytes, elapsed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_profiler() -> BandwidthProfiler {
        let mut p = BandwidthProfiler::new(BandwidthTest::lightweight());
        p.benchmark();
        p
    }

    #[test]
    fn test_benchmark_produces_results() {
        let p = run_profiler();
        assert!(!p.results().is_empty());
    }

    #[test]
    fn test_benchmark_has_seq_read() {
        let p = run_profiler();
        assert!(p.result_for("seq_read").is_some());
    }

    #[test]
    fn test_benchmark_has_seq_write() {
        let p = run_profiler();
        assert!(p.result_for("seq_write").is_some());
    }

    #[test]
    fn test_benchmark_has_copy() {
        let p = run_profiler();
        assert!(p.result_for("copy").is_some());
    }

    #[test]
    fn test_throughput_positive() {
        let p = run_profiler();
        for r in p.results() {
            assert!(
                r.throughput_mbs > 0.0,
                "Expected positive throughput for {}",
                r.label
            );
        }
    }

    #[test]
    fn test_peak_throughput_ge_avg() {
        let p = run_profiler();
        assert!(p.peak_throughput_mbs() >= p.avg_throughput_mbs());
    }

    #[test]
    fn test_no_copy_when_disabled() {
        let test = BandwidthTest {
            buffer_size: 1024 * 1024,
            iterations: 1,
            include_copy: false,
        };
        let mut p = BandwidthProfiler::new(test);
        p.benchmark();
        assert!(p.result_for("copy").is_none());
    }

    #[test]
    fn test_avg_throughput_empty() {
        let p = BandwidthProfiler::new(BandwidthTest::default());
        assert_eq!(p.avg_throughput_mbs(), 0.0);
    }

    #[test]
    fn test_bandwidth_result_new_computes_throughput() {
        // 1 MiB in 1 second = 1 MB/s.
        let r = BandwidthResult::new("test", 1_048_576, 1.0);
        assert!((r.throughput_mbs - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_bandwidth_result_meets_minimum_true() {
        let r = BandwidthResult::new("test", 1_048_576 * 1000, 1.0); // ~1000 MB/s
        assert!(r.meets_minimum(500.0));
    }

    #[test]
    fn test_bandwidth_result_meets_minimum_false() {
        let r = BandwidthResult::new("test", 1_048_576, 1.0); // 1 MB/s
        assert!(!r.meets_minimum(100.0));
    }

    #[test]
    fn test_bandwidth_result_zero_elapsed() {
        let r = BandwidthResult::new("test", 1000, 0.0);
        assert_eq!(r.throughput_mbs, 0.0);
    }

    #[test]
    fn test_results_cleared_on_re_benchmark() {
        let mut p = BandwidthProfiler::new(BandwidthTest::lightweight());
        p.benchmark();
        let first_count = p.results().len();
        p.benchmark();
        assert_eq!(p.results().len(), first_count); // not doubled
    }

    #[test]
    fn test_with_defaults_creates_profiler() {
        let p = BandwidthProfiler::with_defaults();
        assert_eq!(p.results().len(), 0); // no benchmark run yet
    }

    #[test]
    fn test_lightweight_test_config() {
        let t = BandwidthTest::lightweight();
        assert_eq!(t.buffer_size, 1024 * 1024);
        assert_eq!(t.iterations, 1);
        assert!(t.include_copy);
    }
}
