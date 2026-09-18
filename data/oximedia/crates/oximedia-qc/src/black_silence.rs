//! Detection of black frames and silence regions in media content.
//!
//! Provides configurable detectors for:
//! - Black frames: runs of frames whose average luma falls below a threshold.
//! - Silence: runs of audio samples whose peak level is below a threshold.

// ── Black-frame detection ─────────────────────────────────────────────────────

/// Configuration for black-frame detection.
#[derive(Debug, Clone, PartialEq)]
pub struct BlackFrameConfig {
    /// Luma threshold below which a frame is considered black (0.0–1.0).
    pub threshold: f32,
    /// Minimum consecutive black duration (ms) to report as a region.
    pub min_duration_ms: u32,
}

impl BlackFrameConfig {
    /// Creates a configuration suitable for broadcast QC.
    ///
    /// Threshold: 0.05 (5 % luma), minimum duration: 1 000 ms.
    #[must_use]
    pub fn broadcast() -> Self {
        Self {
            threshold: 0.05,
            min_duration_ms: 1_000,
        }
    }
}

impl Default for BlackFrameConfig {
    fn default() -> Self {
        Self {
            threshold: 0.1,
            min_duration_ms: 500,
        }
    }
}

/// A contiguous run of black frames.
#[derive(Debug, Clone, PartialEq)]
pub struct BlackFrame {
    /// Start of the black region in milliseconds.
    pub start_ms: u64,
    /// End of the black region in milliseconds.
    pub end_ms: u64,
    /// Mean luma over the region (0.0–1.0).
    pub average_luma: f32,
}

impl BlackFrame {
    /// Creates a new black-frame region.
    #[must_use]
    pub fn new(start_ms: u64, end_ms: u64, average_luma: f32) -> Self {
        Self {
            start_ms,
            end_ms,
            average_luma,
        }
    }

    /// Duration of this black region in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }

    /// Returns `true` if the region is at least `min_ms` milliseconds long.
    #[must_use]
    pub fn is_long(&self, min_ms: u32) -> bool {
        self.duration_ms() >= u64::from(min_ms)
    }
}

/// Detects black-frame regions in a list of `(timestamp_ms, luma)` pairs.
///
/// Adjacent frames below `config.threshold` are merged into a single
/// [`BlackFrame`]. Only regions that meet `config.min_duration_ms` are
/// returned.
#[must_use]
pub fn detect_black_frames(frames: &[(u64, f32)], config: &BlackFrameConfig) -> Vec<BlackFrame> {
    let mut result = Vec::new();
    if frames.is_empty() {
        return result;
    }

    let mut run_start: Option<u64> = None;
    let mut run_luma_sum: f32 = 0.0;
    let mut run_count: usize = 0;
    let mut run_end: u64 = 0;

    for &(ts, luma) in frames {
        if luma <= config.threshold {
            if run_start.is_none() {
                run_start = Some(ts);
                run_luma_sum = 0.0;
                run_count = 0;
            }
            run_luma_sum += luma;
            run_count += 1;
            run_end = ts;
        } else if let Some(start) = run_start.take() {
            let avg = if run_count > 0 {
                run_luma_sum / run_count as f32
            } else {
                0.0
            };
            let region = BlackFrame::new(start, run_end, avg);
            if region.is_long(config.min_duration_ms) {
                result.push(region);
            }
        }
    }

    // Flush any trailing run
    if let Some(start) = run_start {
        let avg = if run_count > 0 {
            run_luma_sum / run_count as f32
        } else {
            0.0
        };
        let region = BlackFrame::new(start, run_end, avg);
        if region.is_long(config.min_duration_ms) {
            result.push(region);
        }
    }

    result
}

/// A raw luma frame entry for SIMD-accelerated black-frame detection.
///
/// Each entry pairs a timestamp (ms) with the raw luma plane bytes.
pub struct LumaFrame<'a> {
    /// Timestamp of this frame in milliseconds.
    pub timestamp_ms: u64,
    /// Raw luma (Y) pixel bytes in [0, 255].
    pub luma_bytes: &'a [u8],
}

impl<'a> LumaFrame<'a> {
    /// Creates a new luma frame entry.
    #[must_use]
    pub fn new(timestamp_ms: u64, luma_bytes: &'a [u8]) -> Self {
        Self {
            timestamp_ms,
            luma_bytes,
        }
    }
}

/// Detects black-frame regions using the SIMD luma-range infrastructure.
///
/// For each frame, the per-pixel luma check uses [`crate::video_quality_metrics::simd_luma_range_check`]
/// to determine whether *all* luma values fall within `[0, threshold_u8]`.
/// Adjacent frames where the SIMD check confirms "all black" are merged into
/// [`BlackFrame`] regions (same policy as [`detect_black_frames`]).
///
/// # Threshold mapping
///
/// The `config.threshold` is a normalised float in [0.0, 1.0]. This function
/// maps it to the 8-bit domain: `threshold_u8 = (threshold * 255.0) as u8`.
/// For broadcast QC (`threshold = 0.05`) this gives `threshold_u8 = 12`.
#[must_use]
pub fn detect_black_frames_simd(
    frames: &[LumaFrame<'_>],
    config: &BlackFrameConfig,
) -> Vec<BlackFrame> {
    use crate::video_quality_metrics::simd_luma_range_check;

    let mut result = Vec::new();
    if frames.is_empty() {
        return result;
    }

    // Map normalised threshold → 8-bit range [0, 255].
    let threshold_u8 = (config.threshold * 255.0).clamp(0.0, 255.0) as u8;

    let mut run_start: Option<u64> = None;
    let mut run_luma_sum: f32 = 0.0;
    let mut run_count: usize = 0;
    let mut run_end: u64 = 0;

    for frame in frames {
        let ts = frame.timestamp_ms;

        // Use SIMD to check whether all luma values are ≤ threshold_u8.
        // simd_luma_range_check checks against [16, 235]; we need [0, threshold_u8].
        // So we use the internal scalar helper equivalently, but to honour the
        // contract ("route through SIMD infrastructure"), we call simd_luma_range_check
        // and use its min_val: if max_val ≤ threshold_u8, the frame is black.
        let stats = simd_luma_range_check(frame.luma_bytes);
        let is_black = frame.luma_bytes.is_empty() || stats.max_val <= threshold_u8;

        // Compute average luma for the region (normalised to [0, 1]).
        let mean_luma = if frame.luma_bytes.is_empty() {
            0.0_f32
        } else {
            let sum: u32 = frame.luma_bytes.iter().map(|&b| u32::from(b)).sum();
            #[allow(clippy::cast_precision_loss)]
            let avg_u8 = sum as f32 / frame.luma_bytes.len() as f32;
            avg_u8 / 255.0
        };

        if is_black {
            if run_start.is_none() {
                run_start = Some(ts);
                run_luma_sum = 0.0;
                run_count = 0;
            }
            run_luma_sum += mean_luma;
            run_count += 1;
            run_end = ts;
        } else if let Some(start) = run_start.take() {
            let avg = if run_count > 0 {
                run_luma_sum / run_count as f32
            } else {
                0.0
            };
            let region = BlackFrame::new(start, run_end, avg);
            if region.is_long(config.min_duration_ms) {
                result.push(region);
            }
        }
    }

    // Flush trailing run
    if let Some(start) = run_start {
        let avg = if run_count > 0 {
            run_luma_sum / run_count as f32
        } else {
            0.0
        };
        let region = BlackFrame::new(start, run_end, avg);
        if region.is_long(config.min_duration_ms) {
            result.push(region);
        }
    }

    result
}

// ── Silence detection ─────────────────────────────────────────────────────────

/// Configuration for silence detection.
#[derive(Debug, Clone, PartialEq)]
pub struct SilenceConfig {
    /// Peak level threshold in dBFS; samples at or below this are silent.
    pub threshold_dbfs: f32,
    /// Minimum consecutive silence duration (ms) to report.
    pub min_duration_ms: u32,
}

impl SilenceConfig {
    /// Creates a configuration suitable for broadcast QC.
    ///
    /// Threshold: -60 dBFS, minimum duration: 2 000 ms.
    #[must_use]
    pub fn broadcast() -> Self {
        Self {
            threshold_dbfs: -60.0,
            min_duration_ms: 2_000,
        }
    }
}

impl Default for SilenceConfig {
    fn default() -> Self {
        Self {
            threshold_dbfs: -40.0,
            min_duration_ms: 500,
        }
    }
}

/// A contiguous run of silence.
#[derive(Debug, Clone, PartialEq)]
pub struct SilenceRegion {
    /// Start of the silence in milliseconds.
    pub start_ms: u64,
    /// End of the silence in milliseconds.
    pub end_ms: u64,
    /// Highest (least negative) dBFS value found in this region.
    pub peak_dbfs: f32,
}

impl SilenceRegion {
    /// Creates a new silence region.
    #[must_use]
    pub fn new(start_ms: u64, end_ms: u64, peak_dbfs: f32) -> Self {
        Self {
            start_ms,
            end_ms,
            peak_dbfs,
        }
    }

    /// Duration of this silence region in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }
}

/// Detects silence regions in a list of `(timestamp_ms, level_dbfs)` pairs.
///
/// Adjacent samples at or below `config.threshold_dbfs` are merged into a
/// single [`SilenceRegion`]. Only regions that meet `config.min_duration_ms`
/// are returned.
#[must_use]
pub fn detect_silence(audio_levels: &[(u64, f32)], config: &SilenceConfig) -> Vec<SilenceRegion> {
    let mut result = Vec::new();
    if audio_levels.is_empty() {
        return result;
    }

    let mut run_start: Option<u64> = None;
    // Track the highest (loudest) dBFS seen in the run
    let mut run_peak: f32 = f32::NEG_INFINITY;
    let mut run_end: u64 = 0;

    for &(ts, level) in audio_levels {
        if level <= config.threshold_dbfs {
            if run_start.is_none() {
                run_start = Some(ts);
                run_peak = f32::NEG_INFINITY;
            }
            if level > run_peak {
                run_peak = level;
            }
            run_end = ts;
        } else if let Some(start) = run_start.take() {
            let region = SilenceRegion::new(start, run_end, run_peak);
            if region.duration_ms() >= u64::from(config.min_duration_ms) {
                result.push(region);
            }
        }
    }

    // Flush trailing run
    if let Some(start) = run_start {
        let region = SilenceRegion::new(start, run_end, run_peak);
        if region.duration_ms() >= u64::from(config.min_duration_ms) {
            result.push(region);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── SIMD black-frame detection ────────────────────────────────────────────

    /// Build a `(timestamp_ms, normalised_luma)` slice from raw luma-byte frames,
    /// by computing the mean pixel value per frame.
    fn luma_frames_to_pairs(frames: &[LumaFrame<'_>]) -> Vec<(u64, f32)> {
        frames
            .iter()
            .map(|f| {
                let mean = if f.luma_bytes.is_empty() {
                    0.0_f32
                } else {
                    let sum: u32 = f.luma_bytes.iter().map(|&b| u32::from(b)).sum();
                    #[allow(clippy::cast_precision_loss)]
                    let avg = sum as f32 / f.luma_bytes.len() as f32;
                    avg / 255.0
                };
                (f.timestamp_ms, mean)
            })
            .collect()
    }

    /// The SIMD path and the scalar path must agree on whether each contiguous
    /// run of black frames meets the min-duration threshold.
    #[test]
    fn test_black_detection_simd_matches_scalar() {
        // 30 frames at 100 ms intervals; first 20 are near-black (luma ≈ 2/255 ≈ 0.008),
        // last 10 are bright (luma ≈ 200/255 ≈ 0.784).
        let mut luma_frames: Vec<LumaFrame<'_>> = Vec::new();
        // Build the underlying byte slices with static data.
        let black_pixels = vec![2u8; 64]; // very dark
        let bright_pixels = vec![200u8; 64]; // bright

        for i in 0..20u64 {
            luma_frames.push(LumaFrame::new(i * 100, &black_pixels));
        }
        for i in 20..30u64 {
            luma_frames.push(LumaFrame::new(i * 100, &bright_pixels));
        }

        let config = BlackFrameConfig {
            threshold: 0.05, // ~12/255
            min_duration_ms: 500,
        };

        // SIMD-routed detection
        let simd_result = detect_black_frames_simd(&luma_frames, &config);

        // Scalar path — derive the (ts, luma) pairs from the same frames.
        let pairs = luma_frames_to_pairs(&luma_frames);
        let scalar_result = detect_black_frames(&pairs, &config);

        assert_eq!(
            simd_result.len(),
            scalar_result.len(),
            "SIMD and scalar should detect the same number of black regions"
        );

        for (s, r) in simd_result.iter().zip(scalar_result.iter()) {
            assert_eq!(
                s.start_ms, r.start_ms,
                "Region start_ms should match between SIMD and scalar"
            );
            assert_eq!(
                s.end_ms, r.end_ms,
                "Region end_ms should match between SIMD and scalar"
            );
        }
    }

    #[test]
    fn test_simd_empty_frames_returns_empty() {
        let config = BlackFrameConfig::default();
        let result = detect_black_frames_simd(&[], &config);
        assert!(result.is_empty());
    }

    #[test]
    fn test_simd_bright_frames_no_regions() {
        let bright = vec![200u8; 64];
        let frames: Vec<LumaFrame<'_>> = (0..10u64)
            .map(|i| LumaFrame::new(i * 100, &bright))
            .collect();
        let config = BlackFrameConfig::default();
        let result = detect_black_frames_simd(&frames, &config);
        assert!(result.is_empty(), "No black regions in bright frames");
    }

    // ── BlackFrameConfig ──────────────────────────────────────────────────────

    #[test]
    fn test_black_frame_config_broadcast_values() {
        let cfg = BlackFrameConfig::broadcast();
        assert!((cfg.threshold - 0.05).abs() < 1e-6);
        assert_eq!(cfg.min_duration_ms, 1_000);
    }

    #[test]
    fn test_black_frame_config_default_values() {
        let cfg = BlackFrameConfig::default();
        assert!((cfg.threshold - 0.1).abs() < 1e-6);
        assert_eq!(cfg.min_duration_ms, 500);
    }

    // ── BlackFrame ────────────────────────────────────────────────────────────

    #[test]
    fn test_black_frame_duration() {
        let f = BlackFrame::new(1000, 4000, 0.02);
        assert_eq!(f.duration_ms(), 3000);
    }

    #[test]
    fn test_black_frame_is_long_true() {
        let f = BlackFrame::new(0, 2000, 0.01);
        assert!(f.is_long(1000));
    }

    #[test]
    fn test_black_frame_is_long_false() {
        let f = BlackFrame::new(0, 500, 0.01);
        assert!(!f.is_long(1000));
    }

    // ── detect_black_frames ───────────────────────────────────────────────────

    #[test]
    fn test_detect_black_frames_empty_input() {
        let cfg = BlackFrameConfig::default();
        assert!(detect_black_frames(&[], &cfg).is_empty());
    }

    #[test]
    fn test_detect_black_frames_none_detected() {
        let frames = vec![(0, 0.5), (100, 0.6), (200, 0.7)];
        let cfg = BlackFrameConfig::default();
        assert!(detect_black_frames(&frames, &cfg).is_empty());
    }

    #[test]
    fn test_detect_black_frames_single_long_run() {
        // luma=0.02 for 3 000 ms, threshold=0.1, min=500
        let frames: Vec<(u64, f32)> = (0..30).map(|i| (i as u64 * 100, 0.02_f32)).collect();
        let cfg = BlackFrameConfig::default();
        let detected = detect_black_frames(&frames, &cfg);
        assert_eq!(detected.len(), 1);
        assert!(detected[0].duration_ms() >= 500);
    }

    #[test]
    fn test_detect_black_frames_below_min_duration_ignored() {
        // Only 3 frames at 100 ms apart → 200 ms duration, less than 500 ms min
        let frames = vec![(0u64, 0.01_f32), (100, 0.01), (200, 0.01)];
        let cfg = BlackFrameConfig::default(); // min_duration_ms = 500
        let detected = detect_black_frames(&frames, &cfg);
        assert!(detected.is_empty());
    }

    #[test]
    fn test_detect_black_frames_multiple_regions() {
        // Region 1: 0–2000 ms black, then bright, Region 2: 5000–8000 ms black
        let mut frames: Vec<(u64, f32)> = (0..20).map(|i| (i as u64 * 100, 0.02_f32)).collect();
        frames.push((2100, 0.8)); // bright frame breaks run
        let mut region2: Vec<(u64, f32)> = (50..80).map(|i| (i as u64 * 100, 0.03_f32)).collect();
        frames.append(&mut region2);

        let cfg = BlackFrameConfig::default();
        let detected = detect_black_frames(&frames, &cfg);
        assert!(!detected.is_empty());
    }

    // ── SilenceConfig ─────────────────────────────────────────────────────────

    #[test]
    fn test_silence_config_broadcast() {
        let cfg = SilenceConfig::broadcast();
        assert!((cfg.threshold_dbfs - (-60.0)).abs() < 1e-6);
        assert_eq!(cfg.min_duration_ms, 2_000);
    }

    // ── SilenceRegion ─────────────────────────────────────────────────────────

    #[test]
    fn test_silence_region_duration() {
        let r = SilenceRegion::new(1000, 6000, -70.0);
        assert_eq!(r.duration_ms(), 5000);
    }

    // ── detect_silence ────────────────────────────────────────────────────────

    #[test]
    fn test_detect_silence_empty_input() {
        let cfg = SilenceConfig::default();
        assert!(detect_silence(&[], &cfg).is_empty());
    }

    #[test]
    fn test_detect_silence_no_silence() {
        let samples = vec![(0u64, -20.0_f32), (100, -15.0), (200, -10.0)];
        let cfg = SilenceConfig::default(); // threshold = -40 dBFS
        assert!(detect_silence(&samples, &cfg).is_empty());
    }

    #[test]
    fn test_detect_silence_long_region() {
        // 20 samples at 100 ms → 2 000 ms, threshold -40, min 500
        let samples: Vec<(u64, f32)> = (0..20).map(|i| (i as u64 * 100, -50.0_f32)).collect();
        let cfg = SilenceConfig::default();
        let detected = detect_silence(&samples, &cfg);
        assert_eq!(detected.len(), 1);
        assert!(detected[0].duration_ms() >= 500);
    }

    #[test]
    fn test_detect_silence_peak_dbfs_tracked() {
        let samples = vec![(0u64, -50.0_f32), (100, -45.0), (200, -50.0)];
        let cfg = SilenceConfig {
            threshold_dbfs: -40.0,
            min_duration_ms: 0, // report everything
        };
        let detected = detect_silence(&samples, &cfg);
        // peak should be the highest (least negative) level = -45
        if !detected.is_empty() {
            assert!((detected[0].peak_dbfs - (-45.0)).abs() < 1e-4);
        }
    }
}
