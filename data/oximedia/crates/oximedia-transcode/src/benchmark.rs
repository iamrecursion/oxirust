//! TranscodeBenchmark — compare speed and quality metrics across codec configurations.
//!
//! `TranscodeBenchmark` measures and stores encoding metrics (time, bitrate,
//! quality estimations) for multiple `BenchmarkCandidate` configurations, then
//! produces ranked comparison reports.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::time::{Duration, Instant};

use crate::{Result, TranscodeError};

// ─── Quality metrics ──────────────────────────────────────────────────────────

/// Quality and performance metrics from a single encode run.
#[derive(Debug, Clone)]
pub struct EncodeMetrics {
    /// Candidate name.
    pub name: String,
    /// Wall-clock encoding time.
    pub encode_time: Duration,
    /// Encoded file size in bytes.
    pub file_size_bytes: u64,
    /// Average bitrate in kbps (computed from size and duration).
    pub bitrate_kbps: f64,
    /// Content duration in seconds.
    pub duration_secs: f64,
    /// Encoding speed factor (content_duration / encode_time).
    pub speed_factor: f64,
    /// Peak signal-to-noise ratio (dB) — `None` if not measured.
    pub psnr_db: Option<f64>,
    /// Structural similarity (0.0–1.0) — `None` if not measured.
    pub ssim: Option<f64>,
    /// Video Multi-Method Assessment Fusion score — `None` if not measured.
    pub vmaf: Option<f64>,
}

impl EncodeMetrics {
    /// Creates a new `EncodeMetrics` from timing / size data alone.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        encode_time: Duration,
        file_size_bytes: u64,
        duration_secs: f64,
    ) -> Self {
        let encode_secs = encode_time.as_secs_f64();
        let speed_factor = if encode_secs > 0.0 {
            duration_secs / encode_secs
        } else {
            f64::INFINITY
        };
        let bitrate_kbps = if duration_secs > 0.0 {
            file_size_bytes as f64 * 8.0 / 1_000.0 / duration_secs
        } else {
            0.0
        };
        Self {
            name: name.into(),
            encode_time,
            file_size_bytes,
            bitrate_kbps,
            duration_secs,
            speed_factor,
            psnr_db: None,
            ssim: None,
            vmaf: None,
        }
    }

    /// Attaches PSNR.
    #[must_use]
    pub fn with_psnr(mut self, psnr: f64) -> Self {
        self.psnr_db = Some(psnr);
        self
    }

    /// Attaches SSIM.
    #[must_use]
    pub fn with_ssim(mut self, ssim: f64) -> Self {
        self.ssim = Some(ssim);
        self
    }

    /// Attaches VMAF.
    #[must_use]
    pub fn with_vmaf(mut self, vmaf: f64) -> Self {
        self.vmaf = Some(vmaf);
        self
    }

    /// Returns a "BD-Rate proxy" — bits-per-pixel-per-frame.
    ///
    /// Lower is more efficient.
    #[must_use]
    pub fn bits_per_pixel_per_frame(
        &self,
        width: u32,
        height: u32,
        fps_num: u32,
        fps_den: u32,
    ) -> f64 {
        let pixels_per_frame = u64::from(width) * u64::from(height);
        let total_frames = if fps_den > 0 && fps_num > 0 {
            self.duration_secs * f64::from(fps_num) / f64::from(fps_den)
        } else {
            1.0
        };
        if pixels_per_frame == 0 || total_frames <= 0.0 {
            return f64::INFINITY;
        }
        let total_bits = self.file_size_bytes as f64 * 8.0;
        total_bits / (pixels_per_frame as f64 * total_frames)
    }
}

// ─── BenchmarkCandidate ───────────────────────────────────────────────────────

/// A codec configuration to be benchmarked.
#[derive(Debug, Clone)]
pub struct BenchmarkCandidate {
    /// Human-readable name.
    pub name: String,
    /// Codec name (e.g. `"av1"`, `"vp9"`, `"h264"`).
    pub codec: String,
    /// Encoder preset / speed (e.g. `"medium"`, `"5"`, `"slow"`).
    pub preset: String,
    /// CRF value.
    pub crf: u8,
    /// Additional codec-specific parameters (key → value).
    pub extra_params: Vec<(String, String)>,
}

impl BenchmarkCandidate {
    /// Creates a new candidate.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        codec: impl Into<String>,
        preset: impl Into<String>,
        crf: u8,
    ) -> Self {
        Self {
            name: name.into(),
            codec: codec.into(),
            preset: preset.into(),
            crf,
            extra_params: Vec::new(),
        }
    }

    /// Adds an extra codec parameter.
    #[must_use]
    pub fn param(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra_params.push((key.into(), value.into()));
        self
    }
}

// ─── BenchmarkResult ──────────────────────────────────────────────────────────

/// Outcome of a single benchmark run.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// The candidate that was benchmarked.
    pub candidate: BenchmarkCandidate,
    /// Measured metrics.
    pub metrics: EncodeMetrics,
}

impl BenchmarkResult {
    /// Returns a composite quality score combining PSNR, SSIM, and speed.
    ///
    /// The score is on an arbitrary scale; higher = better.
    /// Weights: PSNR 40 %, SSIM 40 %, speed 20 %.
    #[must_use]
    pub fn composite_score(&self) -> f64 {
        let psnr_component = self
            .metrics
            .psnr_db
            .map(|p| (p - 30.0).max(0.0) / 20.0)
            .unwrap_or(0.5);

        let ssim_component = self.metrics.ssim.unwrap_or(0.9);

        // Normalise speed: 1× = 0.5, 2× = 0.75, 4× = 1.0 (capped)
        let speed_component = (self.metrics.speed_factor / 4.0).min(1.0);

        0.4 * psnr_component + 0.4 * ssim_component + 0.2 * speed_component
    }
}

// ─── TranscodeBenchmark ───────────────────────────────────────────────────────

/// Utility for benchmarking and comparing multiple codec configurations.
///
/// `TranscodeBenchmark` accumulates `BenchmarkResult`s and provides sorting /
/// reporting helpers.  The actual encoding is driven by the caller; this type
/// manages the result lifecycle and report generation.
pub struct TranscodeBenchmark {
    /// All recorded results.
    results: Vec<BenchmarkResult>,
    /// Content duration used as the reference.
    content_duration_secs: f64,
}

impl TranscodeBenchmark {
    /// Creates a new benchmark for content with the given duration.
    #[must_use]
    pub fn new(content_duration_secs: f64) -> Self {
        Self {
            results: Vec::new(),
            content_duration_secs,
        }
    }

    /// Starts timing for a candidate.  Returns a [`BenchmarkTimer`] that records
    /// the elapsed time when dropped or when [`BenchmarkTimer::finish`] is called.
    #[must_use]
    pub fn start_timing(&self) -> BenchmarkTimer {
        BenchmarkTimer {
            start: Instant::now(),
        }
    }

    /// Records a result from an already-completed encode.
    pub fn record_result(&mut self, result: BenchmarkResult) {
        self.results.push(result);
    }

    /// Convenience: build a `BenchmarkResult` from timing and file-size data and
    /// record it.
    pub fn record(
        &mut self,
        candidate: BenchmarkCandidate,
        elapsed: Duration,
        file_size_bytes: u64,
        psnr: Option<f64>,
        ssim: Option<f64>,
    ) {
        let mut metrics = EncodeMetrics::new(
            &candidate.name,
            elapsed,
            file_size_bytes,
            self.content_duration_secs,
        );
        if let Some(p) = psnr {
            metrics = metrics.with_psnr(p);
        }
        if let Some(s) = ssim {
            metrics = metrics.with_ssim(s);
        }
        self.results.push(BenchmarkResult { candidate, metrics });
    }

    /// Returns the number of recorded results.
    #[must_use]
    pub fn result_count(&self) -> usize {
        self.results.len()
    }

    /// Returns results sorted by encoding speed (fastest first).
    #[must_use]
    pub fn by_speed(&self) -> Vec<&BenchmarkResult> {
        let mut sorted: Vec<&BenchmarkResult> = self.results.iter().collect();
        sorted.sort_by(|a, b| {
            b.metrics
                .speed_factor
                .partial_cmp(&a.metrics.speed_factor)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted
    }

    /// Returns results sorted by file size (smallest first).
    #[must_use]
    pub fn by_file_size(&self) -> Vec<&BenchmarkResult> {
        let mut sorted: Vec<&BenchmarkResult> = self.results.iter().collect();
        sorted.sort_by_key(|r| r.metrics.file_size_bytes);
        sorted
    }

    /// Returns results sorted by PSNR (highest first).
    ///
    /// Results without PSNR data are sorted to the end.
    #[must_use]
    pub fn by_psnr(&self) -> Vec<&BenchmarkResult> {
        let mut sorted: Vec<&BenchmarkResult> = self.results.iter().collect();
        sorted.sort_by(|a, b| {
            let pa = a.metrics.psnr_db.unwrap_or(f64::NEG_INFINITY);
            let pb = b.metrics.psnr_db.unwrap_or(f64::NEG_INFINITY);
            pb.partial_cmp(&pa).unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted
    }

    /// Returns results sorted by composite score (best first).
    #[must_use]
    pub fn by_composite_score(&self) -> Vec<&BenchmarkResult> {
        let mut sorted: Vec<&BenchmarkResult> = self.results.iter().collect();
        sorted.sort_by(|a, b| {
            b.composite_score()
                .partial_cmp(&a.composite_score())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted
    }

    /// Returns the result with the best composite score, if any.
    #[must_use]
    pub fn best(&self) -> Option<&BenchmarkResult> {
        self.by_composite_score().into_iter().next()
    }

    /// Generates a human-readable Markdown report table.
    ///
    /// # Errors
    ///
    /// Returns an error if there are no results to report.
    pub fn report(&self) -> Result<String> {
        if self.results.is_empty() {
            return Err(TranscodeError::PipelineError(
                "No benchmark results to report".into(),
            ));
        }

        let mut out = String::new();
        out.push_str("| Name | Codec | CRF | Speed | Size (MB) | Bitrate (kbps) | PSNR (dB) | SSIM | Score |\n");
        out.push_str("|------|-------|-----|-------|-----------|----------------|-----------|------|-------|\n");

        for result in self.by_composite_score() {
            let m = &result.metrics;
            let c = &result.candidate;
            let size_mb = m.file_size_bytes as f64 / (1024.0 * 1024.0);
            let psnr = m
                .psnr_db
                .map(|p| format!("{p:.2}"))
                .unwrap_or_else(|| "-".to_string());
            let ssim = m
                .ssim
                .map(|s| format!("{s:.4}"))
                .unwrap_or_else(|| "-".to_string());
            out.push_str(&format!(
                "| {} | {} | {} | {:.2}x | {:.2} | {:.0} | {} | {} | {:.3} |\n",
                c.name,
                c.codec,
                c.crf,
                m.speed_factor,
                size_mb,
                m.bitrate_kbps,
                psnr,
                ssim,
                result.composite_score(),
            ));
        }

        Ok(out)
    }

    /// Returns all raw results.
    #[must_use]
    pub fn results(&self) -> &[BenchmarkResult] {
        &self.results
    }
}

// ─── BenchmarkTimer ───────────────────────────────────────────────────────────

/// A simple wall-clock timer for benchmarking.
pub struct BenchmarkTimer {
    start: Instant,
}

impl BenchmarkTimer {
    /// Returns the elapsed time since the timer was created.
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    /// Consumes the timer and returns the elapsed duration.
    #[must_use]
    pub fn finish(self) -> Duration {
        self.start.elapsed()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_result(name: &str, codec: &str, crf: u8, secs: f64, size: u64) -> BenchmarkResult {
        let candidate = BenchmarkCandidate::new(name, codec, "medium", crf);
        let metrics = EncodeMetrics::new(name, Duration::from_secs_f64(secs), size, 60.0);
        BenchmarkResult { candidate, metrics }
    }

    #[test]
    fn test_encode_metrics_speed_factor() {
        let m = EncodeMetrics::new("test", Duration::from_secs(10), 10_000_000, 60.0);
        assert!((m.speed_factor - 6.0).abs() < 1e-9);
    }

    #[test]
    fn test_encode_metrics_bitrate() {
        // 10 MB / 60 s * 8 / 1000 = 13333 kbps
        let m = EncodeMetrics::new("test", Duration::from_secs(1), 10_000_000, 60.0);
        assert!((m.bitrate_kbps - 10_000_000.0 * 8.0 / 1_000.0 / 60.0).abs() < 1.0);
    }

    #[test]
    fn test_encode_metrics_zero_duration() {
        let m = EncodeMetrics::new("test", Duration::from_secs(1), 1024, 0.0);
        assert_eq!(m.bitrate_kbps, 0.0);
    }

    #[test]
    fn test_benchmark_record_and_count() {
        let mut bench = TranscodeBenchmark::new(60.0);
        let cand = BenchmarkCandidate::new("AV1 CRF28", "av1", "5", 28);
        bench.record(
            cand,
            Duration::from_secs(20),
            5_000_000,
            Some(42.5),
            Some(0.97),
        );
        assert_eq!(bench.result_count(), 1);
    }

    #[test]
    fn test_benchmark_by_speed() {
        let mut bench = TranscodeBenchmark::new(60.0);
        bench.record_result(make_result("slow", "h264", 23, 60.0, 10_000_000));
        bench.record_result(make_result("fast", "h264", 23, 10.0, 12_000_000));

        let sorted = bench.by_speed();
        assert_eq!(sorted[0].candidate.name, "fast");
    }

    #[test]
    fn test_benchmark_by_file_size() {
        let mut bench = TranscodeBenchmark::new(60.0);
        bench.record_result(make_result("big", "h264", 18, 20.0, 50_000_000));
        bench.record_result(make_result("small", "av1", 30, 60.0, 5_000_000));

        let sorted = bench.by_file_size();
        assert_eq!(sorted[0].candidate.name, "small");
    }

    #[test]
    fn test_benchmark_by_psnr() {
        let mut bench = TranscodeBenchmark::new(60.0);
        let cand_a = BenchmarkCandidate::new("A", "h264", "medium", 23);
        let cand_b = BenchmarkCandidate::new("B", "av1", "5", 30);
        let m_a = EncodeMetrics::new("A", Duration::from_secs(10), 5_000_000, 60.0).with_psnr(42.0);
        let m_b = EncodeMetrics::new("B", Duration::from_secs(30), 4_000_000, 60.0).with_psnr(44.0);
        bench.record_result(BenchmarkResult {
            candidate: cand_a,
            metrics: m_a,
        });
        bench.record_result(BenchmarkResult {
            candidate: cand_b,
            metrics: m_b,
        });

        let sorted = bench.by_psnr();
        assert_eq!(sorted[0].candidate.name, "B");
    }

    #[test]
    fn test_benchmark_best() {
        let mut bench = TranscodeBenchmark::new(60.0);
        bench.record_result(make_result("a", "h264", 23, 20.0, 5_000_000));
        bench.record_result(make_result("b", "av1", 30, 90.0, 3_000_000));
        assert!(bench.best().is_some());
    }

    #[test]
    fn test_benchmark_report() {
        let mut bench = TranscodeBenchmark::new(60.0);
        let cand = BenchmarkCandidate::new("VP9 medium", "vp9", "medium", 31);
        let metrics = EncodeMetrics::new("VP9 medium", Duration::from_secs(15), 8_000_000, 60.0)
            .with_psnr(41.0)
            .with_ssim(0.96);
        bench.record_result(BenchmarkResult {
            candidate: cand,
            metrics,
        });

        let report = bench.report().expect("report ok");
        assert!(report.contains("VP9 medium"));
        assert!(report.contains("41.00"));
        assert!(report.contains("0.9600"));
    }

    #[test]
    fn test_benchmark_report_empty_error() {
        let bench = TranscodeBenchmark::new(60.0);
        assert!(bench.report().is_err());
    }

    #[test]
    fn test_benchmark_timer() {
        let bench = TranscodeBenchmark::new(60.0);
        let timer = bench.start_timing();
        let elapsed = timer.finish();
        // Should be very short in test environment
        assert!(elapsed.as_secs() < 10);
    }

    #[test]
    fn test_composite_score_range() {
        let mut bench = TranscodeBenchmark::new(60.0);
        let cand = BenchmarkCandidate::new("X", "h264", "fast", 23);
        let metrics = EncodeMetrics::new("X", Duration::from_secs(5), 4_000_000, 60.0)
            .with_psnr(40.0)
            .with_ssim(0.95);
        let result = BenchmarkResult {
            candidate: cand,
            metrics,
        };
        let score = result.composite_score();
        assert!(
            score >= 0.0 && score <= 1.0,
            "score {score} out of range [0,1]"
        );
        bench.record_result(result);
        assert_eq!(bench.result_count(), 1);
    }

    #[test]
    fn test_bits_per_pixel_per_frame() {
        let m = EncodeMetrics::new("t", Duration::from_secs(10), 9_000_000, 30.0);
        let bppf = m.bits_per_pixel_per_frame(1920, 1080, 30, 1);
        // 9_000_000 * 8 = 72_000_000 bits / (1920*1080*30*30) ≈ 72M / (1866240000)
        assert!(bppf > 0.0 && bppf < 1.0);
    }
}
