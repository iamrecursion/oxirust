//! Bitrate estimation from QP/CRF, resolution, and frame-rate parameters.
//!
//! This module also integrates [`RunningStats`](crate::running_stats::RunningStats)
//! and [`BitrateRunningAnalyzer`]
//! so that callers can accumulate per-frame bitrate statistics incrementally
//! (Welford online algorithm) without storing all past values.

use crate::running_stats::BitrateRunningAnalyzer;

/// Estimates output bitrate (bits per second) from encode parameters.
///
/// The model is a simplified empirical formula:
/// `bitrate ≈ base_bpp * pixels_per_frame * frame_rate * quality_factor`
///
/// where `quality_factor` decreases as QP/CRF increases.
///
/// An embedded [`BitrateRunningAnalyzer`] accumulates per-frame bitrate
/// statistics incrementally via Welford's online algorithm so that callers
/// can query running mean/variance without storing historical data.
///
/// # Example
///
/// ```
/// use oximedia_transcode::bitrate_estimator::BitrateEstimator;
///
/// let mut est = BitrateEstimator::new();
/// let bps = est.estimate_from_crf(23, 1920, 1080, 30.0);
/// assert!(bps > 0);
///
/// // Feed observed per-frame bit counts into the running analyzer
/// est.record_frame_bits(50_000);
/// est.record_frame_bits(45_000);
/// let summary = est.running_summary();
/// assert!(summary.mean_bps > 0.0);
/// ```
#[derive(Debug)]
pub struct BitrateEstimator {
    /// Base bits-per-pixel at CRF/QP = 0.
    base_bpp: f64,
    /// Exponential decay coefficient for quality degradation with QP.
    decay: f64,
    /// Online running statistics for per-frame bitrate (Welford algorithm).
    analyzer: BitrateRunningAnalyzer,
}

impl Default for BitrateEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl BitrateEstimator {
    /// Creates a `BitrateEstimator` with default empirical parameters.
    ///
    /// The embedded running analyzer is initialised for 30 fps with a 60-frame
    /// rolling window; call [`with_params_and_fps`](Self::with_params_and_fps)
    /// to customise these.
    #[must_use]
    pub fn new() -> Self {
        Self {
            base_bpp: 0.10, // 0.10 bits per pixel at lossless
            decay: 0.065,   // tuned to match real-world H.264/VP9 behaviour
            analyzer: BitrateRunningAnalyzer::new(30.0, 60),
        }
    }

    /// Creates a `BitrateEstimator` with custom parameters.
    ///
    /// * `base_bpp` – bits per pixel at QP = 0.
    /// * `decay` – exponential decay constant; higher = steeper quality/bitrate curve.
    #[must_use]
    pub fn with_params(base_bpp: f64, decay: f64) -> Self {
        Self {
            base_bpp,
            decay,
            analyzer: BitrateRunningAnalyzer::new(30.0, 60),
        }
    }

    /// Creates a `BitrateEstimator` with custom model parameters and analyzer config.
    ///
    /// * `fps`           – Frame rate for the running analyzer (bits/frame → bits/s).
    /// * `window_frames` – Rolling-window size for recent-peak detection.
    #[must_use]
    pub fn with_params_and_fps(base_bpp: f64, decay: f64, fps: f64, window_frames: usize) -> Self {
        Self {
            base_bpp,
            decay,
            analyzer: BitrateRunningAnalyzer::new(fps, window_frames),
        }
    }

    /// Records the observed bit count for one encoded frame into the running analyzer.
    ///
    /// Use this to track actual per-frame bitrate statistics incrementally
    /// (Welford algorithm) without buffering all historical frame data.
    pub fn record_frame_bits(&mut self, bits_per_frame: u64) {
        self.analyzer.push_frame(bits_per_frame);
    }

    /// Returns a snapshot of current running bitrate statistics.
    ///
    /// The summary reflects all frames recorded via [`record_frame_bits`](Self::record_frame_bits).
    #[must_use]
    pub fn running_summary(&self) -> crate::running_stats::BitrateSummary {
        self.analyzer.summary()
    }

    /// Resets the running bitrate statistics to their initial state.
    pub fn reset_running_stats(&mut self) {
        self.analyzer.reset();
    }

    /// Estimates output bitrate in bits/s from a CRF value (0–51 for H.264/H.265).
    ///
    /// * `crf`        – Constant Rate Factor (0 = lossless, 51 = worst).
    /// * `width`      – Frame width in pixels.
    /// * `height`     – Frame height in pixels.
    /// * `frame_rate` – Frames per second.
    #[must_use]
    pub fn estimate_from_crf(&self, crf: u8, width: u32, height: u32, frame_rate: f64) -> u64 {
        self.estimate_from_qp(f64::from(crf), width, height, frame_rate)
    }

    /// Estimates output bitrate in bits/s from a floating-point QP value.
    ///
    /// * `qp`         – Quantization parameter.
    /// * `width`      – Frame width in pixels.
    /// * `height`     – Frame height in pixels.
    /// * `frame_rate` – Frames per second.
    #[must_use]
    pub fn estimate_from_qp(&self, qp: f64, width: u32, height: u32, frame_rate: f64) -> u64 {
        if frame_rate <= 0.0 || width == 0 || height == 0 {
            return 0;
        }
        let pixels = f64::from(width) * f64::from(height);
        let quality_factor = (-self.decay * qp).exp();
        let bps = self.base_bpp * pixels * frame_rate * quality_factor;
        bps.round() as u64
    }

    /// Estimates bitrate from a target VMAF score (0–100).
    ///
    /// Linearly maps VMAF → effective QP, then delegates to `estimate_from_qp`.
    /// VMAF 100 ≈ QP 0 (lossless), VMAF 0 ≈ QP 51 (worst).
    #[must_use]
    pub fn estimate_from_vmaf(&self, vmaf: f64, width: u32, height: u32, frame_rate: f64) -> u64 {
        let vmaf_clamped = vmaf.clamp(0.0, 100.0);
        let qp = 51.0 * (1.0 - vmaf_clamped / 100.0);
        self.estimate_from_qp(qp, width, height, frame_rate)
    }

    /// Infers the CRF value that would be required to hit a target bitrate.
    ///
    /// Returns `None` when the target cannot be reached with valid QP values (0–63).
    #[must_use]
    pub fn crf_for_target_bitrate(
        &self,
        target_bps: u64,
        width: u32,
        height: u32,
        frame_rate: f64,
    ) -> Option<u8> {
        if frame_rate <= 0.0 || width == 0 || height == 0 || target_bps == 0 {
            return None;
        }
        let pixels = f64::from(width) * f64::from(height);
        // bps = base_bpp * pixels * fps * e^(-decay * qp)
        // qp = -ln(bps / (base_bpp * pixels * fps)) / decay
        let denominator = self.base_bpp * pixels * frame_rate;
        if denominator <= 0.0 {
            return None;
        }
        let qp = -(target_bps as f64 / denominator).ln() / self.decay;
        if !(0.0..=63.0).contains(&qp) {
            return None;
        }
        Some(qp.round() as u8)
    }

    /// Returns an estimated size in bytes for encoding `duration_secs` of video.
    #[must_use]
    pub fn estimate_file_size(
        &self,
        crf: u8,
        width: u32,
        height: u32,
        frame_rate: f64,
        duration_secs: f64,
    ) -> u64 {
        let bps = self.estimate_from_crf(crf, width, height, frame_rate);
        ((bps as f64 * duration_secs) / 8.0).round() as u64
    }
}

/// A helper that bundles video parameters for convenience.
#[derive(Debug, Clone, Copy)]
pub struct VideoParams {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Frames per second.
    pub frame_rate: f64,
    /// CRF value (0–51 for most codecs).
    pub crf: u8,
}

impl VideoParams {
    /// Creates new video params.
    #[must_use]
    pub fn new(width: u32, height: u32, frame_rate: f64, crf: u8) -> Self {
        Self {
            width,
            height,
            frame_rate,
            crf,
        }
    }

    /// Estimates bitrate using a `BitrateEstimator`.
    #[must_use]
    pub fn estimate_bitrate(&self, estimator: &BitrateEstimator) -> u64 {
        estimator.estimate_from_crf(self.crf, self.width, self.height, self.frame_rate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_from_crf_positive() {
        let est = BitrateEstimator::new();
        let bps = est.estimate_from_crf(23, 1920, 1080, 30.0);
        assert!(bps > 0, "Expected positive bitrate, got {bps}");
    }

    #[test]
    fn test_lower_crf_higher_bitrate() {
        let est = BitrateEstimator::new();
        let high_quality = est.estimate_from_crf(18, 1920, 1080, 30.0);
        let low_quality = est.estimate_from_crf(28, 1920, 1080, 30.0);
        assert!(
            high_quality > low_quality,
            "CRF 18 should yield more bits than CRF 28"
        );
    }

    #[test]
    fn test_higher_resolution_higher_bitrate() {
        let est = BitrateEstimator::new();
        let fhd = est.estimate_from_crf(23, 1920, 1080, 30.0);
        let uhd = est.estimate_from_crf(23, 3840, 2160, 30.0);
        assert!(uhd > fhd, "4K should require more bits than 1080p");
    }

    #[test]
    fn test_higher_fps_higher_bitrate() {
        let est = BitrateEstimator::new();
        let fps30 = est.estimate_from_crf(23, 1920, 1080, 30.0);
        let fps60 = est.estimate_from_crf(23, 1920, 1080, 60.0);
        assert!(fps60 > fps30, "60 fps should require more bits than 30 fps");
        assert!(
            (fps60 as f64 / fps30 as f64 - 2.0).abs() < 0.01,
            "Should scale linearly with fps"
        );
    }

    #[test]
    fn test_zero_dimensions_returns_zero() {
        let est = BitrateEstimator::new();
        assert_eq!(est.estimate_from_crf(23, 0, 1080, 30.0), 0);
        assert_eq!(est.estimate_from_crf(23, 1920, 0, 30.0), 0);
        assert_eq!(est.estimate_from_crf(23, 1920, 1080, 0.0), 0);
    }

    #[test]
    fn test_vmaf_estimate_high_quality() {
        let est = BitrateEstimator::new();
        let high = est.estimate_from_vmaf(95.0, 1920, 1080, 30.0);
        let low = est.estimate_from_vmaf(50.0, 1920, 1080, 30.0);
        assert!(high > low, "VMAF 95 should need more bits than VMAF 50");
    }

    #[test]
    fn test_crf_for_target_bitrate_roundtrip() {
        let est = BitrateEstimator::new();
        let target_crf: u8 = 23;
        let bps = est.estimate_from_crf(target_crf, 1920, 1080, 30.0);
        if let Some(inferred_crf) = est.crf_for_target_bitrate(bps, 1920, 1080, 30.0) {
            // Allow ±1 due to rounding.
            assert!(
                (inferred_crf as i16 - target_crf as i16).abs() <= 1,
                "Expected CRF ~{target_crf}, got {inferred_crf}"
            );
        }
    }

    #[test]
    fn test_estimate_file_size() {
        let est = BitrateEstimator::new();
        let bytes = est.estimate_file_size(23, 1920, 1080, 30.0, 60.0); // 60 s clip
        assert!(bytes > 0);
        // File size in bytes = bps * duration / 8
        let bps = est.estimate_from_crf(23, 1920, 1080, 30.0);
        let expected = (bps as f64 * 60.0 / 8.0).round() as u64;
        assert_eq!(bytes, expected);
    }

    #[test]
    fn test_video_params_estimate_bitrate() {
        let params = VideoParams::new(1920, 1080, 30.0, 23);
        let est = BitrateEstimator::new();
        let bps = params.estimate_bitrate(&est);
        assert_eq!(bps, est.estimate_from_crf(23, 1920, 1080, 30.0));
    }

    #[test]
    fn test_custom_params() {
        let est = BitrateEstimator::with_params(0.2, 0.05);
        let bps = est.estimate_from_crf(20, 1280, 720, 25.0);
        assert!(bps > 0);
    }

    // ── Running-statistics integration tests (T1) ─────────────────────────────

    /// Verify that `RunningStats` (Welford) produces the same mean and sample
    /// variance as a classic two-pass batch computation over the same data.
    #[test]
    fn test_running_stats_matches_batch_computation() {
        use crate::running_stats::RunningStats;

        // Representative per-frame bit counts (arbitrary realistic values)
        let samples = [
            10_000.0_f64,
            12_500.0,
            8_700.0,
            15_000.0,
            9_800.0,
            11_300.0,
            13_600.0,
            7_900.0,
            14_200.0,
            10_500.0,
        ];

        // Online (Welford) accumulator
        let mut stats = RunningStats::new();
        for &s in &samples {
            stats.push(s);
        }

        // Batch computation (two-pass)
        let n = samples.len() as f64;
        let batch_mean = samples.iter().sum::<f64>() / n;
        let batch_var = samples
            .iter()
            .map(|&x| (x - batch_mean).powi(2))
            .sum::<f64>()
            / (n - 1.0); // sample variance (n-1)

        let tol = 1e-6;
        assert!(
            (stats.mean() - batch_mean).abs() < tol,
            "mean mismatch: welford={}, batch={}",
            stats.mean(),
            batch_mean
        );
        assert!(
            (stats.variance() - batch_var).abs() < tol,
            "variance mismatch: welford={}, batch={}",
            stats.variance(),
            batch_var
        );
    }

    /// Verify that incremental updates to `RunningStats` give the same final
    /// state as pushing all samples in one go.
    #[test]
    fn test_running_stats_incremental_update() {
        use crate::running_stats::RunningStats;

        let all_samples = [1.0_f64, 4.0, 9.0, 16.0, 25.0, 36.0];

        // Single-batch accumulator
        let mut batch = RunningStats::new();
        for &s in &all_samples {
            batch.push(s);
        }

        // Incremental: push half, check intermediate state, then push the rest
        let mut incremental = RunningStats::new();
        let (first_half, second_half) = all_samples.split_at(3);
        for &s in first_half {
            incremental.push(s);
        }
        // Intermediate mean should match the first three samples
        let expected_mid_mean = first_half.iter().sum::<f64>() / first_half.len() as f64;
        assert!(
            (incremental.mean() - expected_mid_mean).abs() < 1e-10,
            "mid-point mean mismatch: got {}, expected {}",
            incremental.mean(),
            expected_mid_mean
        );

        for &s in second_half {
            incremental.push(s);
        }

        // Final state must match the single-pass accumulator
        let tol = 1e-10;
        assert_eq!(incremental.count(), batch.count());
        assert!(
            (incremental.mean() - batch.mean()).abs() < tol,
            "final mean mismatch"
        );
        assert!(
            (incremental.variance() - batch.variance()).abs() < tol,
            "final variance mismatch"
        );
        assert_eq!(incremental.min(), batch.min());
        assert_eq!(incremental.max(), batch.max());
    }
}
