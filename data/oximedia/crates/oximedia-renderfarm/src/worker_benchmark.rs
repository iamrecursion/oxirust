#![allow(dead_code)]
//! Worker performance benchmarking for the OxiMedia render farm.
//!
//! When a render worker starts up it can run a brief set of synthetic
//! workloads to characterise its compute capability.  The resulting
//! [`WorkerBenchmarkScore`] is stored in the worker registry so the
//! scheduler can make informed placement decisions (e.g. assign
//! motion-blur-heavy shots to nodes with high multi-core throughput).
//!
//! The benchmarks deliberately avoid floating-point non-determinism:
//! "mega-pixels per second" (Mpps) is computed from simple integer pixel
//! operations so results are comparable across runs.

use std::time::Instant;

// ── WorkerBenchmarkScore ─────────────────────────────────────────────────────

/// Performance scores produced by the worker self-benchmark.
///
/// All throughput figures are expressed in mega-pixels per second (Mpps)
/// using a synthetic RGBA pixel processing workload so different hardware
/// can be compared on a common scale.
#[derive(Debug, Clone)]
pub struct WorkerBenchmarkScore {
    /// Single-threaded pixel throughput (Mpps).
    pub single_core_mpps: f64,
    /// Multi-threaded pixel throughput using all logical cores (Mpps).
    pub multi_core_mpps: f64,
    /// GPU pixel throughput (Mpps), `None` if no GPU benchmark was run.
    pub gpu_mpps: Option<f64>,
    /// Number of logical CPU cores detected during the benchmark.
    pub cpu_cores: usize,
    /// Memory bandwidth estimate in GiB/s (sequential read).
    pub memory_bandwidth_gib_s: f64,
    /// Overall composite score (dimensionless, higher is better).
    pub composite_score: f64,
}

impl WorkerBenchmarkScore {
    /// Build a score from measured components.
    ///
    /// `composite_score` is computed automatically as a weighted average of
    /// the supplied values.
    #[must_use]
    pub fn new(
        single_core_mpps: f64,
        multi_core_mpps: f64,
        gpu_mpps: Option<f64>,
        cpu_cores: usize,
        memory_bandwidth_gib_s: f64,
    ) -> Self {
        let composite = Self::compute_composite(
            single_core_mpps,
            multi_core_mpps,
            gpu_mpps,
            memory_bandwidth_gib_s,
        );
        Self {
            single_core_mpps,
            multi_core_mpps,
            gpu_mpps,
            cpu_cores,
            memory_bandwidth_gib_s,
            composite_score: composite,
        }
    }

    /// Compute the composite score from individual metrics.
    ///
    /// Base score = 0.70 × multi_core + 0.20 × single_core + 0.10 × mem_bw_score.
    /// A GPU bonus of up to 20 points is added on top when a GPU is present,
    /// ensuring the GPU-equipped score is always ≥ the CPU-only score.
    fn compute_composite(single: f64, multi: f64, gpu: Option<f64>, mem_bw: f64) -> f64 {
        let mem_score = (mem_bw / 50.0).min(1.0) * 100.0; // normalise to ~100
        let base = 0.70 * multi + 0.20 * single + 0.10 * mem_score;
        // GPU adds a non-negative bonus: capped at 20 points
        let gpu_bonus = gpu.map(|g| (g / 1000.0).min(1.0) * 20.0).unwrap_or(0.0);
        base + gpu_bonus
    }

    /// Returns `true` when this worker outperforms `other` on the composite
    /// score.
    #[must_use]
    pub fn is_faster_than(&self, other: &Self) -> bool {
        self.composite_score > other.composite_score
    }

    /// Human-readable tier based on composite score.
    #[must_use]
    pub fn tier(&self) -> &'static str {
        match self.composite_score as u64 {
            0..=25 => "Low",
            26..=50 => "Medium",
            51..=75 => "High",
            _ => "Extreme",
        }
    }
}

impl std::fmt::Display for WorkerBenchmarkScore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Benchmark[tier={}, composite={:.1}, single={:.1}Mpps, multi={:.1}Mpps]",
            self.tier(),
            self.composite_score,
            self.single_core_mpps,
            self.multi_core_mpps
        )
    }
}

// ── BenchmarkConfig ──────────────────────────────────────────────────────────

/// Configuration controlling how the worker benchmark runs.
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Number of pixels to process in the single-core test.
    pub single_core_pixels: u64,
    /// Number of pixels to process in the multi-core test.
    pub multi_core_pixels: u64,
    /// Whether to run the (simulated) GPU benchmark.
    pub run_gpu_benchmark: bool,
    /// Number of benchmark iterations for averaging.
    pub iterations: u32,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            single_core_pixels: 1_000_000, // 1 Mpx
            multi_core_pixels: 8_000_000,  // 8 Mpx
            run_gpu_benchmark: false,
            iterations: 3,
        }
    }
}

impl BenchmarkConfig {
    /// Create a fast/light configuration for unit tests or quick probes.
    #[must_use]
    pub fn light() -> Self {
        Self {
            single_core_pixels: 100_000,
            multi_core_pixels: 400_000,
            run_gpu_benchmark: false,
            iterations: 1,
        }
    }
}

// ── WorkerBenchmarker ────────────────────────────────────────────────────────

/// Runs synthetic benchmarks on the current machine and produces a
/// [`WorkerBenchmarkScore`].
///
/// All workloads are pure-Rust and safe; the "pixel processing" kernel
/// performs a simple blend operation on `u32` RGBA values to exercise the
/// integer ALU and cache hierarchy without requiring SIMD intrinsics.
pub struct WorkerBenchmarker {
    config: BenchmarkConfig,
}

impl WorkerBenchmarker {
    /// Create a benchmarker with the given configuration.
    #[must_use]
    pub fn new(config: BenchmarkConfig) -> Self {
        Self { config }
    }

    /// Run all configured benchmarks and return a [`WorkerBenchmarkScore`].
    #[must_use]
    pub fn run(&self) -> WorkerBenchmarkScore {
        let single = self.bench_single_core();
        let multi = self.bench_multi_core();
        let gpu = if self.config.run_gpu_benchmark {
            Some(self.bench_gpu_simulated())
        } else {
            None
        };
        let mem_bw = self.bench_memory_bandwidth();
        let cores = num_cpus::get();

        WorkerBenchmarkScore::new(single, multi, gpu, cores, mem_bw)
    }

    /// Single-threaded pixel-blend benchmark.
    ///
    /// Processes `single_core_pixels` pixels per iteration and averages over
    /// `iterations` runs.  Returns throughput in Mpps.
    fn bench_single_core(&self) -> f64 {
        let n = self.config.single_core_pixels as usize;
        let iters = self.config.iterations.max(1) as u64;
        let mut total_ns: u64 = 0;

        for _ in 0..iters {
            let t = Instant::now();
            let _ = pixel_blend_kernel(n);
            total_ns += t.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        }

        let avg_ns = total_ns / iters;
        pixels_to_mpps(self.config.single_core_pixels, avg_ns)
    }

    /// Multi-threaded pixel-blend benchmark using `std::thread`.
    ///
    /// Splits the pixel load across all logical cores and measures wall-clock
    /// time. Returns throughput in Mpps.
    fn bench_multi_core(&self) -> f64 {
        let cores = num_cpus::get().max(1);
        let n = self.config.multi_core_pixels as usize;
        let iters = self.config.iterations.max(1) as u64;
        let mut total_ns: u64 = 0;

        for _ in 0..iters {
            let chunk = (n / cores).max(1);
            let t = Instant::now();

            let handles: Vec<_> = (0..cores)
                .map(|_| {
                    std::thread::spawn(move || {
                        let _ = pixel_blend_kernel(chunk);
                    })
                })
                .collect();

            for h in handles {
                // If a thread panics we just ignore it in the benchmark context
                let _ = h.join();
            }

            total_ns += t.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        }

        let avg_ns = total_ns / iters;
        pixels_to_mpps(self.config.multi_core_pixels, avg_ns)
    }

    /// Simulated GPU benchmark (returns a fixed calibration value).
    ///
    /// Real GPU benchmarks would invoke a Vulkan/Metal compute shader;
    /// this stub returns a plausible value so the code path can be tested
    /// without requiring GPU hardware on CI.
    fn bench_gpu_simulated(&self) -> f64 {
        // Simulate ~2000 Mpps to represent a mid-range GPU
        2000.0
    }

    /// Estimate sequential memory read bandwidth.
    ///
    /// Reads a buffer of `u64` values and sums them to prevent the compiler
    /// from eliding the load.  Returns GiB/s.
    fn bench_memory_bandwidth(&self) -> f64 {
        const BUF_SIZE: usize = 64 * 1024 * 1024 / 8; // 64 MiB as u64
        let buf: Vec<u64> = (0..BUF_SIZE as u64).collect();
        let iters = self.config.iterations.max(1) as u64;
        let mut total_ns: u64 = 0;
        let mut checksum: u64 = 0;

        for _ in 0..iters {
            let t = Instant::now();
            checksum = checksum.wrapping_add(buf.iter().fold(0u64, |a, &v| a.wrapping_add(v)));
            total_ns += t.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        }

        // prevent dead-code elimination
        let _ = checksum;

        let avg_ns = (total_ns / iters).max(1);
        let bytes = (BUF_SIZE * 8) as f64;
        let secs = avg_ns as f64 * 1e-9;
        bytes / secs / (1024.0 * 1024.0 * 1024.0)
    }
}

impl Default for WorkerBenchmarker {
    fn default() -> Self {
        Self::new(BenchmarkConfig::default())
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Synthetic RGBA pixel-blend kernel.
///
/// For each of `count` pixels, performs an over-blend of two constant RGBA
/// values.  Uses wrapping arithmetic to avoid undefined behaviour on
/// saturating additions.
///
/// Returns the sum of all output pixels (to prevent dead-code elimination).
#[must_use]
fn pixel_blend_kernel(count: usize) -> u64 {
    let src: u32 = 0x80_FF_80_FF; // semi-transparent green
    let dst: u32 = 0x40_00_FF_00; // semi-transparent blue
    let alpha = (src >> 24) & 0xFF;
    let inv_alpha = 255u32.wrapping_sub(alpha);

    let mut accum: u64 = 0;
    for i in 0..count {
        // Simple Porter-Duff "over" approximation in integer arithmetic
        let r = ((src & 0xFF) * alpha + (dst & 0xFF) * inv_alpha) >> 8;
        let g = (((src >> 8) & 0xFF) * alpha + ((dst >> 8) & 0xFF) * inv_alpha) >> 8;
        let b = (((src >> 16) & 0xFF) * alpha + ((dst >> 16) & 0xFF) * inv_alpha) >> 8;
        let out = r | (g << 8) | (b << 16) | (255 << 24);
        accum = accum.wrapping_add(out as u64).wrapping_add(i as u64);
    }
    accum
}

/// Convert pixel count and elapsed nanoseconds to Mpps.
fn pixels_to_mpps(pixels: u64, elapsed_ns: u64) -> f64 {
    if elapsed_ns == 0 {
        return 0.0;
    }
    let elapsed_secs = elapsed_ns as f64 * 1e-9;
    (pixels as f64 / elapsed_secs) / 1_000_000.0
}

// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    fn light_benchmarker() -> WorkerBenchmarker {
        WorkerBenchmarker::new(BenchmarkConfig::light())
    }

    // ── WorkerBenchmarkScore ─────────────────────────────────────────────

    #[test]
    fn test_score_construction() {
        let score = WorkerBenchmarkScore::new(100.0, 400.0, None, 8, 20.0);
        assert_eq!(score.single_core_mpps, 100.0);
        assert_eq!(score.multi_core_mpps, 400.0);
        assert!(score.gpu_mpps.is_none());
        assert_eq!(score.cpu_cores, 8);
    }

    #[test]
    fn test_score_composite_positive() {
        let score = WorkerBenchmarkScore::new(50.0, 200.0, None, 4, 10.0);
        assert!(score.composite_score > 0.0);
    }

    #[test]
    fn test_score_gpu_increases_composite() {
        let without_gpu = WorkerBenchmarkScore::new(50.0, 200.0, None, 4, 10.0);
        let with_gpu = WorkerBenchmarkScore::new(50.0, 200.0, Some(5000.0), 4, 10.0);
        // GPU-equipped worker should have higher or equal composite
        assert!(with_gpu.composite_score >= without_gpu.composite_score);
    }

    #[test]
    fn test_score_is_faster_than() {
        let fast = WorkerBenchmarkScore::new(200.0, 800.0, None, 16, 80.0);
        let slow = WorkerBenchmarkScore::new(10.0, 40.0, None, 2, 5.0);
        assert!(fast.is_faster_than(&slow));
        assert!(!slow.is_faster_than(&fast));
    }

    #[test]
    fn test_score_tier_low() {
        let score = WorkerBenchmarkScore::new(0.1, 0.5, None, 1, 0.1);
        assert_eq!(score.tier(), "Low");
    }

    #[test]
    fn test_score_display_contains_tier() {
        let score = WorkerBenchmarkScore::new(100.0, 400.0, None, 4, 20.0);
        let s = score.to_string();
        assert!(s.contains("Benchmark[tier="));
    }

    // ── BenchmarkConfig ─────────────────────────────────────────────────

    #[test]
    fn test_config_light_has_fewer_pixels() {
        let light = BenchmarkConfig::light();
        let default = BenchmarkConfig::default();
        assert!(light.single_core_pixels < default.single_core_pixels);
    }

    #[test]
    fn test_config_default_no_gpu() {
        assert!(!BenchmarkConfig::default().run_gpu_benchmark);
    }

    // ── WorkerBenchmarker ────────────────────────────────────────────────

    #[test]
    fn test_run_returns_positive_scores() {
        let benchmarker = light_benchmarker();
        let score = benchmarker.run();
        assert!(score.single_core_mpps > 0.0);
        assert!(score.multi_core_mpps > 0.0);
        assert!(score.memory_bandwidth_gib_s > 0.0);
    }

    #[test]
    fn test_run_with_gpu_simulation() {
        let config = BenchmarkConfig {
            run_gpu_benchmark: true,
            ..BenchmarkConfig::light()
        };
        let benchmarker = WorkerBenchmarker::new(config);
        let score = benchmarker.run();
        assert!(score.gpu_mpps.is_some());
        let gpu = score.gpu_mpps.expect("GPU benchmark ran");
        assert!(gpu > 0.0);
    }

    #[test]
    fn test_run_cpu_cores_detected() {
        let benchmarker = light_benchmarker();
        let score = benchmarker.run();
        assert!(score.cpu_cores > 0);
    }

    #[test]
    fn test_run_composite_score_positive() {
        let score = light_benchmarker().run();
        assert!(score.composite_score > 0.0);
    }

    // ── Helper functions ─────────────────────────────────────────────────

    #[test]
    fn test_pixel_blend_kernel_deterministic() {
        let a = pixel_blend_kernel(1000);
        let b = pixel_blend_kernel(1000);
        assert_eq!(a, b, "kernel must be deterministic");
    }

    #[test]
    fn test_pixel_blend_kernel_zero_pixels() {
        assert_eq!(pixel_blend_kernel(0), 0);
    }

    #[test]
    fn test_pixels_to_mpps_zero_elapsed_returns_zero() {
        assert_eq!(pixels_to_mpps(1_000_000, 0), 0.0);
    }

    #[test]
    fn test_pixels_to_mpps_reasonable_value() {
        // 1 Mpx in 10 ms → 100 Mpps
        let mpps = pixels_to_mpps(1_000_000, 10_000_000);
        assert!((mpps - 100.0).abs() < 1.0, "got {mpps}");
    }
}
