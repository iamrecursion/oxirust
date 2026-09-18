//! Video temporal splice detection.
//!
//! Detects temporal splices across frame sequences by analysing sudden changes
//! in histogram distributions, noise profiles, and brightness statistics
//! between adjacent frames. A genuine video has smoothly varying per-frame
//! statistics; a splice point introduces abrupt discontinuities.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

// ---------------------------------------------------------------------------
// Frame statistics
// ---------------------------------------------------------------------------

/// Per-frame statistics used for temporal consistency analysis.
#[derive(Debug, Clone)]
pub struct FrameStatistics {
    /// Frame index in the sequence (0-based).
    pub frame_index: u64,
    /// Mean brightness (luma) of the frame, in [0, 1].
    pub mean_brightness: f32,
    /// Standard deviation of brightness across the frame.
    pub brightness_std: f32,
    /// Noise estimate (MAD-based) for the frame.
    pub noise_estimate: f32,
    /// Normalised 16-bin luma histogram (sums to 1.0).
    pub histogram: [f32; 16],
}

impl FrameStatistics {
    /// Compute frame statistics from a luma buffer.
    ///
    /// `luma` values are expected in [0, 1], row-major, `width * height` elements.
    #[must_use]
    pub fn from_luma(frame_index: u64, luma: &[f32]) -> Self {
        if luma.is_empty() {
            return Self {
                frame_index,
                mean_brightness: 0.0,
                brightness_std: 0.0,
                noise_estimate: 0.0,
                histogram: [0.0; 16],
            };
        }

        let n = luma.len() as f32;
        let mean: f32 = luma.iter().sum::<f32>() / n;
        let variance: f32 = luma.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n;
        let std_dev = variance.sqrt();

        // 16-bin histogram
        let mut bins = [0_u32; 16];
        for &v in luma {
            let bin = ((v.clamp(0.0, 0.9999)) * 16.0) as usize;
            let bin = bin.min(15);
            bins[bin] += 1;
        }
        let mut histogram = [0.0_f32; 16];
        for (i, &count) in bins.iter().enumerate() {
            histogram[i] = count as f32 / n;
        }

        // Noise estimate via MAD on consecutive pixel differences
        let noise_estimate = estimate_frame_noise(luma);

        Self {
            frame_index,
            mean_brightness: mean,
            brightness_std: std_dev,
            noise_estimate,
            histogram,
        }
    }

    /// Chi-squared distance between this frame's histogram and another's.
    #[must_use]
    pub fn histogram_chi2(&self, other: &Self) -> f32 {
        let mut chi2 = 0.0_f32;
        for i in 0..16 {
            let a = self.histogram[i];
            let b = other.histogram[i];
            let sum = a + b;
            if sum > 1e-10 {
                let diff = a - b;
                chi2 += diff * diff / sum;
            }
        }
        chi2 * 0.5
    }
}

// ---------------------------------------------------------------------------
// Temporal splice detection
// ---------------------------------------------------------------------------

/// A detected temporal splice point between two consecutive frames.
#[derive(Debug, Clone)]
pub struct TemporalSplicePoint {
    /// Frame index of the frame *before* the splice.
    pub frame_before: u64,
    /// Frame index of the frame *after* the splice.
    pub frame_after: u64,
    /// Histogram chi-squared distance between the two frames.
    pub histogram_distance: f32,
    /// Absolute change in noise estimate.
    pub noise_change: f32,
    /// Absolute change in mean brightness.
    pub brightness_change: f32,
    /// Combined splice score (0.0..1.0).
    pub score: f32,
    /// Human-readable reason for flagging this splice.
    pub reason: String,
}

/// Configuration for temporal splice detection.
#[derive(Debug, Clone)]
pub struct TemporalSpliceConfig {
    /// Minimum histogram chi-squared distance to flag a splice (default 0.15).
    pub histogram_threshold: f32,
    /// Minimum noise change ratio to flag a splice (default 0.3).
    pub noise_threshold: f32,
    /// Minimum brightness change to flag a splice (default 0.15).
    pub brightness_threshold: f32,
    /// Number of standard deviations above the running mean to flag (default 3.0).
    pub adaptive_sigma: f32,
}

impl Default for TemporalSpliceConfig {
    fn default() -> Self {
        Self {
            histogram_threshold: 0.15,
            noise_threshold: 0.3,
            brightness_threshold: 0.15,
            adaptive_sigma: 3.0,
        }
    }
}

/// Result of temporal splice analysis on a frame sequence.
#[derive(Debug, Clone)]
pub struct TemporalSpliceResult {
    /// Detected splice points.
    pub splice_points: Vec<TemporalSplicePoint>,
    /// Number of frames analysed.
    pub frames_analyzed: usize,
    /// Whether any splice was detected.
    pub detected: bool,
    /// Overall confidence (0.0..1.0).
    pub confidence: f32,
    /// Textual findings.
    pub findings: Vec<String>,
}

/// Analyse a sequence of frame statistics for temporal splice points.
///
/// This function compares adjacent frames for sudden changes in histogram
/// distribution, noise profile, and brightness that would indicate content
/// from different sources has been spliced together.
#[must_use]
pub fn detect_temporal_splices(
    frames: &[FrameStatistics],
    config: &TemporalSpliceConfig,
) -> TemporalSpliceResult {
    if frames.len() < 2 {
        return TemporalSpliceResult {
            splice_points: Vec::new(),
            frames_analyzed: frames.len(),
            detected: false,
            confidence: 0.0,
            findings: vec!["Not enough frames for temporal analysis".to_string()],
        };
    }

    // Compute pairwise distances
    let num_pairs = frames.len() - 1;
    let mut hist_dists = Vec::with_capacity(num_pairs);
    let mut noise_changes = Vec::with_capacity(num_pairs);
    let mut brightness_changes = Vec::with_capacity(num_pairs);

    for i in 0..num_pairs {
        let chi2 = frames[i].histogram_chi2(&frames[i + 1]);
        let noise_delta = (frames[i + 1].noise_estimate - frames[i].noise_estimate).abs();
        let bright_delta = (frames[i + 1].mean_brightness - frames[i].mean_brightness).abs();
        hist_dists.push(chi2);
        noise_changes.push(noise_delta);
        brightness_changes.push(bright_delta);
    }

    // Compute running statistics for adaptive thresholding
    let hist_mean: f32 = hist_dists.iter().sum::<f32>() / num_pairs as f32;
    let hist_var: f32 = hist_dists
        .iter()
        .map(|&v| (v - hist_mean) * (v - hist_mean))
        .sum::<f32>()
        / num_pairs as f32;
    let hist_std = hist_var.sqrt();

    let noise_mean: f32 = noise_changes.iter().sum::<f32>() / num_pairs as f32;
    let noise_var: f32 = noise_changes
        .iter()
        .map(|&v| (v - noise_mean) * (v - noise_mean))
        .sum::<f32>()
        / num_pairs as f32;
    let noise_std = noise_var.sqrt();

    let bright_mean: f32 = brightness_changes.iter().sum::<f32>() / num_pairs as f32;
    let bright_var: f32 = brightness_changes
        .iter()
        .map(|&v| (v - bright_mean) * (v - bright_mean))
        .sum::<f32>()
        / num_pairs as f32;
    let bright_std = bright_var.sqrt();

    let mut splice_points = Vec::new();

    for i in 0..num_pairs {
        let hist_d = hist_dists[i];
        let noise_d = noise_changes[i];
        let bright_d = brightness_changes[i];

        // Check fixed thresholds
        let hist_flag = hist_d > config.histogram_threshold;
        let noise_flag = noise_d > config.noise_threshold;
        let bright_flag = bright_d > config.brightness_threshold;

        // Check adaptive thresholds (z-score based)
        let hist_z = if hist_std > 1e-9 {
            (hist_d - hist_mean) / hist_std
        } else {
            0.0
        };
        let noise_z = if noise_std > 1e-9 {
            (noise_d - noise_mean) / noise_std
        } else {
            0.0
        };
        let bright_z = if bright_std > 1e-9 {
            (bright_d - bright_mean) / bright_std
        } else {
            0.0
        };

        let adaptive_flag = hist_z > config.adaptive_sigma
            || noise_z > config.adaptive_sigma
            || bright_z > config.adaptive_sigma;

        // Need at least two out of three fixed flags, or the adaptive flag
        let flag_count = hist_flag as u32 + noise_flag as u32 + bright_flag as u32;
        let is_splice = flag_count >= 2 || (flag_count >= 1 && adaptive_flag);

        if is_splice {
            // Combine into a score
            let score = (hist_d / config.histogram_threshold.max(0.01) * 0.4
                + noise_d / config.noise_threshold.max(0.01) * 0.3
                + bright_d / config.brightness_threshold.max(0.01) * 0.3)
                .clamp(0.0, 1.0);

            let mut reasons = Vec::new();
            if hist_flag {
                reasons.push(format!("histogram chi2={hist_d:.4}"));
            }
            if noise_flag {
                reasons.push(format!("noise change={noise_d:.4}"));
            }
            if bright_flag {
                reasons.push(format!("brightness change={bright_d:.4}"));
            }
            if adaptive_flag {
                reasons.push("adaptive z-score exceeded".to_string());
            }

            splice_points.push(TemporalSplicePoint {
                frame_before: frames[i].frame_index,
                frame_after: frames[i + 1].frame_index,
                histogram_distance: hist_d,
                noise_change: noise_d,
                brightness_change: bright_d,
                score,
                reason: reasons.join("; "),
            });
        }
    }

    let detected = !splice_points.is_empty();
    let confidence = if detected {
        let max_score = splice_points
            .iter()
            .map(|s| s.score)
            .fold(0.0_f32, f32::max);
        max_score
    } else {
        0.0
    };

    let mut findings = Vec::new();
    findings.push(format!(
        "Analysed {} frame pairs for temporal splices",
        num_pairs
    ));
    if detected {
        findings.push(format!(
            "Detected {} temporal splice point(s)",
            splice_points.len()
        ));
    } else {
        findings.push("No temporal splices detected".to_string());
    }

    TemporalSpliceResult {
        splice_points,
        frames_analyzed: frames.len(),
        detected,
        confidence,
        findings,
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Estimate per-frame noise using MAD of consecutive pixel differences.
fn estimate_frame_noise(luma: &[f32]) -> f32 {
    if luma.len() < 2 {
        return 0.0;
    }

    let mut diffs: Vec<f32> = luma.windows(2).map(|w| (w[1] - w[0]).abs()).collect();

    diffs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = diffs.len() / 2;
    let median = if diffs.len() % 2 == 0 && mid > 0 {
        (diffs[mid - 1] + diffs[mid]) / 2.0
    } else {
        diffs[mid]
    };

    median * 1.4826 // MAD scaling for Gaussian consistency
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_uniform_frame(index: u64, brightness: f32) -> FrameStatistics {
        let luma = vec![brightness; 64 * 64];
        FrameStatistics::from_luma(index, &luma)
    }

    fn make_varied_frame(index: u64, base: f32, amplitude: f32) -> FrameStatistics {
        let luma: Vec<f32> = (0..64 * 64)
            .map(|i| (base + amplitude * (i as f32 * 0.1).sin()).clamp(0.0, 1.0))
            .collect();
        FrameStatistics::from_luma(index, &luma)
    }

    // ── FrameStatistics ─────────────────────────────────────────────────────

    #[test]
    fn test_frame_statistics_from_empty() {
        let stats = FrameStatistics::from_luma(0, &[]);
        assert_eq!(stats.frame_index, 0);
        assert!((stats.mean_brightness).abs() < 1e-6);
        assert!((stats.noise_estimate).abs() < 1e-6);
    }

    #[test]
    fn test_frame_statistics_uniform() {
        let stats = make_uniform_frame(5, 0.5);
        assert_eq!(stats.frame_index, 5);
        assert!((stats.mean_brightness - 0.5).abs() < 1e-4);
        assert!(stats.brightness_std < 1e-4);
        assert!(stats.noise_estimate < 1e-4);
    }

    #[test]
    fn test_frame_statistics_histogram_sums_to_one() {
        let stats = make_varied_frame(0, 0.5, 0.3);
        let sum: f32 = stats.histogram.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "Histogram should sum to ~1.0");
    }

    #[test]
    fn test_frame_statistics_varied_has_nonzero_std() {
        let stats = make_varied_frame(0, 0.5, 0.3);
        assert!(stats.brightness_std > 0.01);
    }

    #[test]
    fn test_frame_statistics_varied_has_nonzero_noise() {
        let stats = make_varied_frame(0, 0.5, 0.3);
        assert!(stats.noise_estimate > 0.0);
    }

    // ── histogram_chi2 ──────────────────────────────────────────────────────

    #[test]
    fn test_histogram_chi2_identical() {
        let a = make_uniform_frame(0, 0.5);
        let b = make_uniform_frame(1, 0.5);
        let chi2 = a.histogram_chi2(&b);
        assert!(
            chi2 < 1e-6,
            "Identical frames should have zero chi2 distance"
        );
    }

    #[test]
    fn test_histogram_chi2_different() {
        let a = make_uniform_frame(0, 0.1);
        let b = make_uniform_frame(1, 0.9);
        let chi2 = a.histogram_chi2(&b);
        assert!(
            chi2 > 0.5,
            "Very different frames should have large chi2 distance"
        );
    }

    #[test]
    fn test_histogram_chi2_symmetric() {
        let a = make_varied_frame(0, 0.3, 0.2);
        let b = make_varied_frame(1, 0.7, 0.1);
        let d1 = a.histogram_chi2(&b);
        let d2 = b.histogram_chi2(&a);
        assert!((d1 - d2).abs() < 1e-6, "Chi2 should be symmetric");
    }

    // ── detect_temporal_splices ──────────────────────────────────────────────

    #[test]
    fn test_detect_temporal_splices_not_enough_frames() {
        let config = TemporalSpliceConfig::default();
        let result = detect_temporal_splices(&[], &config);
        assert!(!result.detected);
        assert_eq!(result.frames_analyzed, 0);
    }

    #[test]
    fn test_detect_temporal_splices_single_frame() {
        let config = TemporalSpliceConfig::default();
        let frames = vec![make_uniform_frame(0, 0.5)];
        let result = detect_temporal_splices(&frames, &config);
        assert!(!result.detected);
    }

    #[test]
    fn test_detect_temporal_splices_consistent_sequence() {
        let config = TemporalSpliceConfig::default();
        // Smoothly varying sequence
        let frames: Vec<FrameStatistics> = (0..10)
            .map(|i| make_uniform_frame(i, 0.5 + i as f32 * 0.01))
            .collect();
        let result = detect_temporal_splices(&frames, &config);
        assert!(
            !result.detected,
            "Smoothly varying sequence should not be flagged"
        );
        assert_eq!(result.frames_analyzed, 10);
    }

    #[test]
    fn test_detect_temporal_splices_abrupt_change() {
        let config = TemporalSpliceConfig {
            histogram_threshold: 0.10,
            noise_threshold: 0.01,
            brightness_threshold: 0.10,
            adaptive_sigma: 2.0,
        };
        // Sequence with abrupt splice at frame 5
        let mut frames: Vec<FrameStatistics> = (0..5).map(|i| make_uniform_frame(i, 0.2)).collect();
        // Splice: sudden jump to bright noisy content
        for i in 5..10 {
            frames.push(make_varied_frame(i, 0.8, 0.15));
        }
        let result = detect_temporal_splices(&frames, &config);
        assert!(result.detected, "Abrupt change should be detected");
        assert!(!result.splice_points.is_empty());
        // The splice should be detected around frame 4-5
        let splice = &result.splice_points[0];
        assert_eq!(splice.frame_before, 4);
        assert_eq!(splice.frame_after, 5);
        assert!(splice.score > 0.0);
    }

    #[test]
    fn test_detect_temporal_splices_multiple_splices() {
        let config = TemporalSpliceConfig {
            histogram_threshold: 0.10,
            noise_threshold: 0.01,
            brightness_threshold: 0.10,
            adaptive_sigma: 2.0,
        };
        // Dark → bright → dark
        let mut frames = Vec::new();
        for i in 0..4 {
            frames.push(make_uniform_frame(i, 0.1));
        }
        for i in 4..8 {
            frames.push(make_varied_frame(i, 0.85, 0.1));
        }
        for i in 8..12 {
            frames.push(make_uniform_frame(i, 0.15));
        }
        let result = detect_temporal_splices(&frames, &config);
        assert!(
            result.splice_points.len() >= 2,
            "Should detect at least two splice points"
        );
    }

    #[test]
    fn test_temporal_splice_result_confidence_bounded() {
        let config = TemporalSpliceConfig::default();
        let mut frames: Vec<FrameStatistics> = (0..3).map(|i| make_uniform_frame(i, 0.2)).collect();
        frames.push(make_varied_frame(3, 0.9, 0.4));
        let result = detect_temporal_splices(&frames, &config);
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
    }

    #[test]
    fn test_temporal_splice_config_defaults() {
        let config = TemporalSpliceConfig::default();
        assert!((config.histogram_threshold - 0.15).abs() < 1e-6);
        assert!((config.noise_threshold - 0.3).abs() < 1e-6);
        assert!((config.brightness_threshold - 0.15).abs() < 1e-6);
        assert!((config.adaptive_sigma - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_temporal_splice_findings_non_empty() {
        let config = TemporalSpliceConfig::default();
        let frames = vec![make_uniform_frame(0, 0.5), make_uniform_frame(1, 0.5)];
        let result = detect_temporal_splices(&frames, &config);
        assert!(!result.findings.is_empty());
    }

    // ── estimate_frame_noise ────────────────────────────────────────────────

    #[test]
    fn test_estimate_frame_noise_uniform() {
        let luma = vec![0.5_f32; 100];
        let noise = estimate_frame_noise(&luma);
        assert!(noise < 1e-5, "Uniform frame should have near-zero noise");
    }

    #[test]
    fn test_estimate_frame_noise_noisy() {
        let luma: Vec<f32> = (0..100)
            .map(|i| (i as f32 * 0.5).sin() * 0.4 + 0.5)
            .collect();
        let noise = estimate_frame_noise(&luma);
        assert!(
            noise > 0.0,
            "Noisy frame should have positive noise estimate"
        );
    }

    #[test]
    fn test_estimate_frame_noise_empty() {
        assert!(estimate_frame_noise(&[]).abs() < 1e-10);
    }

    #[test]
    fn test_estimate_frame_noise_single() {
        assert!(estimate_frame_noise(&[0.5]).abs() < 1e-10);
    }
}
