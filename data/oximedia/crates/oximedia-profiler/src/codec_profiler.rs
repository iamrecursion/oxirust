//! Codec performance benchmarking and comparison.
//!
//! Provides simulated codec profiling, Pareto-front analysis, and a
//! structured comparison report between a baseline and a candidate codec.

#![allow(dead_code)]

// ── CodecBenchmark ────────────────────────────────────────────────────────────

/// A single codec benchmark result.
#[derive(Debug, Clone)]
pub struct CodecBenchmark {
    /// Short name of the codec (e.g. "h264", "av1", "vp9").
    pub codec_name: String,
    /// Resolution (width, height) used for the benchmark.
    pub resolution: (u32, u32),
    /// Source frame-rate in frames per second.
    pub fps: f32,
    /// Target bit rate in kbps.
    pub bitrate_kbps: u32,
    /// Achieved encode throughput in fps.
    pub encode_fps: f32,
    /// Achieved decode throughput in fps.
    pub decode_fps: f32,
    /// Averaged PSNR in dB.
    pub psnr_db: f32,
    /// CPU utilisation during encoding (0–100 %).
    pub cpu_usage_pct: f32,
    /// Peak memory consumption in MiB.
    pub memory_mb: u32,
}

// ── CodecBenchmarkSuite ───────────────────────────────────────────────────────

/// A collection of codec benchmarks from a single benchmark run.
#[derive(Debug, Clone, Default)]
pub struct CodecBenchmarkSuite {
    /// All benchmark entries.
    pub benchmarks: Vec<CodecBenchmark>,
    /// Wall-clock duration of the entire run in seconds.
    pub run_duration_secs: f64,
}

impl CodecBenchmarkSuite {
    /// Create an empty suite.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a benchmark result to the suite.
    pub fn add(&mut self, b: CodecBenchmark) {
        self.benchmarks.push(b);
    }

    /// Return the benchmark with the highest PSNR.
    #[must_use]
    pub fn best_by_psnr(&self) -> Option<&CodecBenchmark> {
        self.benchmarks.iter().max_by(|a, b| {
            a.psnr_db
                .partial_cmp(&b.psnr_db)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Return the benchmark with the highest encode fps.
    #[must_use]
    pub fn best_by_speed(&self) -> Option<&CodecBenchmark> {
        self.benchmarks.iter().max_by(|a, b| {
            a.encode_fps
                .partial_cmp(&b.encode_fps)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Return only Pareto-optimal benchmarks (quality vs. speed).
    ///
    /// Benchmark A dominates B if `A.psnr_db >= B.psnr_db` **and**
    /// `A.encode_fps >= B.encode_fps` (with at least one strict inequality).
    #[must_use]
    pub fn pareto_optimal(&self) -> Vec<&CodecBenchmark> {
        let mut result = Vec::new();
        for a in &self.benchmarks {
            let dominated = self.benchmarks.iter().any(|b| {
                // b dominates a
                b.psnr_db >= a.psnr_db
                    && b.encode_fps >= a.encode_fps
                    && (b.psnr_db > a.psnr_db || b.encode_fps > a.encode_fps)
            });
            if !dominated {
                result.push(a);
            }
        }
        result
    }
}

// ── CodecProfiler ─────────────────────────────────────────────────────────────

/// Simulated codec profiler (no real encoding performed).
pub struct CodecProfiler;

impl CodecProfiler {
    /// Simulate profiling a codec and return a `CodecBenchmark`.
    ///
    /// Results are deterministically derived from the parameters so that
    /// tests are reproducible.
    #[must_use]
    pub fn profile(codec: &str, resolution: (u32, u32), frames: u32) -> CodecBenchmark {
        let pixels = resolution.0 as f32 * resolution.1 as f32;
        // Higher pixel count → slower encode
        let encode_fps = (1_920.0 * 1_080.0 / pixels * 30.0).max(1.0);
        let decode_fps = encode_fps * 4.0;
        // Cheap heuristic: h264 < vp9 < av1 in PSNR
        let psnr_base = match codec {
            c if c.contains("av1") => 42.0,
            c if c.contains("vp9") => 40.0,
            c if c.contains("h265") || c.contains("hevc") => 41.0,
            _ => 38.0, // h264 and others
        };
        let psnr_db = psnr_base + (frames as f32).ln() * 0.5;
        let bitrate_kbps = ((pixels / 1000.0) * 5.0) as u32;
        let cpu_usage_pct = (100.0 * pixels / (1_920.0 * 1_080.0)).min(100.0);
        let memory_mb = (pixels / 1_000_000.0 * 64.0) as u32 + 128;

        CodecBenchmark {
            codec_name: codec.to_string(),
            resolution,
            fps: 30.0,
            bitrate_kbps,
            encode_fps,
            decode_fps,
            psnr_db,
            cpu_usage_pct,
            memory_mb,
        }
    }
}

// ── ComparisonReport ──────────────────────────────────────────────────────────

/// Verdict from a codec comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// Candidate offers better quality.
    BetterQuality,
    /// Candidate offers better encoding speed.
    BetterSpeed,
    /// Candidate offers better bitrate efficiency.
    BetterEfficiency,
    /// Candidate is worse in all compared dimensions.
    Worse,
    /// Candidate is within noise of baseline.
    Similar,
}

/// Structured comparison between a baseline and a candidate codec.
#[derive(Debug, Clone)]
pub struct ComparisonReport {
    /// PSNR difference (candidate − baseline) in dB.
    pub psnr_delta: f32,
    /// Speed ratio (candidate / baseline encode fps).
    pub speed_ratio: f32,
    /// Bitrate efficiency ratio (baseline_kbps / candidate_kbps; >1 = candidate uses less bits).
    pub bitrate_efficiency: f32,
    /// Overall verdict.
    pub verdict: Verdict,
}

/// Epsilon for "similar" determination.
const SIMILAR_PSNR: f32 = 0.5;
const SIMILAR_SPEED: f32 = 0.05; // 5 %

/// Comparison utility.
pub struct BenchmarkComparison;

impl BenchmarkComparison {
    /// Compare a `candidate` codec benchmark against a `baseline`.
    #[must_use]
    pub fn compare(baseline: &CodecBenchmark, candidate: &CodecBenchmark) -> ComparisonReport {
        let psnr_delta = candidate.psnr_db - baseline.psnr_db;
        let speed_ratio = if baseline.encode_fps > 0.0 {
            candidate.encode_fps / baseline.encode_fps
        } else {
            1.0
        };
        let bitrate_efficiency = if candidate.bitrate_kbps > 0 {
            baseline.bitrate_kbps as f32 / candidate.bitrate_kbps as f32
        } else {
            1.0
        };

        let verdict =
            if psnr_delta.abs() < SIMILAR_PSNR && (speed_ratio - 1.0).abs() < SIMILAR_SPEED {
                Verdict::Similar
            } else if psnr_delta > SIMILAR_PSNR && speed_ratio >= 1.0 - SIMILAR_SPEED {
                Verdict::BetterQuality
            } else if speed_ratio > 1.0 + SIMILAR_SPEED && psnr_delta >= -SIMILAR_PSNR {
                Verdict::BetterSpeed
            } else if bitrate_efficiency > 1.05 {
                Verdict::BetterEfficiency
            } else {
                Verdict::Worse
            };

        ComparisonReport {
            psnr_delta,
            speed_ratio,
            bitrate_efficiency,
            verdict,
        }
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_benchmark(codec: &str, psnr: f32, encode_fps: f32, bitrate: u32) -> CodecBenchmark {
        CodecBenchmark {
            codec_name: codec.to_string(),
            resolution: (1920, 1080),
            fps: 30.0,
            bitrate_kbps: bitrate,
            encode_fps,
            decode_fps: encode_fps * 4.0,
            psnr_db: psnr,
            cpu_usage_pct: 80.0,
            memory_mb: 256,
        }
    }

    #[test]
    fn test_best_by_psnr() {
        let mut suite = CodecBenchmarkSuite::new();
        suite.add(make_benchmark("h264", 38.0, 60.0, 5000));
        suite.add(make_benchmark("av1", 42.0, 10.0, 3000));
        suite.add(make_benchmark("vp9", 40.0, 30.0, 4000));
        let best = suite.best_by_psnr().expect("should succeed in test");
        assert_eq!(best.codec_name, "av1");
    }

    #[test]
    fn test_best_by_speed() {
        let mut suite = CodecBenchmarkSuite::new();
        suite.add(make_benchmark("h264", 38.0, 60.0, 5000));
        suite.add(make_benchmark("av1", 42.0, 10.0, 3000));
        let best = suite.best_by_speed().expect("should succeed in test");
        assert_eq!(best.codec_name, "h264");
    }

    #[test]
    fn test_pareto_optimal_single() {
        let mut suite = CodecBenchmarkSuite::new();
        suite.add(make_benchmark("h264", 38.0, 60.0, 5000));
        let pareto = suite.pareto_optimal();
        assert_eq!(pareto.len(), 1);
    }

    #[test]
    fn test_pareto_optimal_dominated() {
        let mut suite = CodecBenchmarkSuite::new();
        // av1 dominates h264 (higher PSNR AND higher speed)
        suite.add(make_benchmark("h264", 38.0, 30.0, 5000));
        suite.add(make_benchmark("av1", 42.0, 60.0, 3000)); // dominates h264
        let pareto = suite.pareto_optimal();
        assert_eq!(pareto.len(), 1);
        assert_eq!(pareto[0].codec_name, "av1");
    }

    #[test]
    fn test_pareto_optimal_tradeoff() {
        let mut suite = CodecBenchmarkSuite::new();
        // h264: fast but lower quality; av1: slow but higher quality
        suite.add(make_benchmark("h264", 38.0, 60.0, 5000));
        suite.add(make_benchmark("av1", 44.0, 5.0, 3000));
        let pareto = suite.pareto_optimal();
        // Neither dominates the other → both on Pareto front
        assert_eq!(pareto.len(), 2);
    }

    #[test]
    fn test_empty_suite() {
        let suite = CodecBenchmarkSuite::new();
        assert!(suite.best_by_psnr().is_none());
        assert!(suite.best_by_speed().is_none());
        assert!(suite.pareto_optimal().is_empty());
    }

    #[test]
    fn test_codec_profiler_h264() {
        let b = CodecProfiler::profile("h264", (1920, 1080), 300);
        assert_eq!(b.codec_name, "h264");
        assert!(b.psnr_db > 0.0);
        assert!(b.encode_fps > 0.0);
    }

    #[test]
    fn test_codec_profiler_av1_higher_psnr() {
        let h264 = CodecProfiler::profile("h264", (1920, 1080), 300);
        let av1 = CodecProfiler::profile("av1", (1920, 1080), 300);
        assert!(av1.psnr_db > h264.psnr_db, "av1 PSNR should exceed h264");
    }

    #[test]
    fn test_comparison_better_quality() {
        let baseline = make_benchmark("h264", 38.0, 30.0, 5000);
        let candidate = make_benchmark("av1", 43.0, 30.0, 3000);
        let report = BenchmarkComparison::compare(&baseline, &candidate);
        assert_eq!(report.verdict, Verdict::BetterQuality);
        assert!(report.psnr_delta > 0.0);
    }

    #[test]
    fn test_comparison_better_speed() {
        let baseline = make_benchmark("av1", 42.0, 10.0, 3000);
        let candidate = make_benchmark("h264", 42.0, 60.0, 5000);
        let report = BenchmarkComparison::compare(&baseline, &candidate);
        assert_eq!(report.verdict, Verdict::BetterSpeed);
        assert!(report.speed_ratio > 1.0);
    }

    #[test]
    fn test_comparison_similar() {
        let baseline = make_benchmark("h264", 38.0, 30.0, 5000);
        let candidate = make_benchmark("h264_fast", 38.2, 30.5, 5100);
        let report = BenchmarkComparison::compare(&baseline, &candidate);
        assert_eq!(report.verdict, Verdict::Similar);
    }

    #[test]
    fn test_comparison_worse() {
        let baseline = make_benchmark("av1", 44.0, 60.0, 3000);
        let candidate = make_benchmark("h264", 35.0, 20.0, 6000);
        let report = BenchmarkComparison::compare(&baseline, &candidate);
        assert_eq!(report.verdict, Verdict::Worse);
    }
}
