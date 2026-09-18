#![allow(dead_code)]
//! Signal statistics computation for video frames.
//!
//! This module computes per-frame and per-region signal statistics including
//! minimum, maximum, mean, standard deviation, peak-to-peak range, and
//! percentile values for luma and chroma channels. Useful for QC, exposure
//! evaluation, and automated grading decisions.

/// Statistics computed for a single channel or region.
#[derive(Debug, Clone, Copy)]
pub struct ChannelStats {
    /// Minimum sample value (0 - 255 for 8-bit).
    pub min: f64,
    /// Maximum sample value.
    pub max: f64,
    /// Arithmetic mean.
    pub mean: f64,
    /// Standard deviation.
    pub std_dev: f64,
    /// Median (50th percentile).
    pub median: f64,
    /// Peak-to-peak range (max - min).
    pub range: f64,
    /// Number of samples analyzed.
    pub count: u64,
}

impl Default for ChannelStats {
    fn default() -> Self {
        Self {
            min: f64::MAX,
            max: f64::MIN,
            mean: 0.0,
            std_dev: 0.0,
            median: 0.0,
            range: 0.0,
            count: 0,
        }
    }
}

/// Full-frame signal statistics across R, G, B, and computed luma.
#[derive(Debug, Clone)]
pub struct FrameSignalStats {
    /// Red channel statistics.
    pub red: ChannelStats,
    /// Green channel statistics.
    pub green: ChannelStats,
    /// Blue channel statistics.
    pub blue: ChannelStats,
    /// Luma (Y) statistics derived from Rec.709 weighting.
    pub luma: ChannelStats,
    /// Total number of pixels.
    pub total_pixels: u64,
    /// Frame width.
    pub width: u32,
    /// Frame height.
    pub height: u32,
}

/// Compute statistics for a single-channel sample buffer.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn compute_channel_stats(samples: &[f64]) -> ChannelStats {
    if samples.is_empty() {
        return ChannelStats::default();
    }

    let count = samples.len() as u64;
    let mut min = f64::MAX;
    let mut max = f64::MIN;
    let mut sum = 0.0_f64;
    let mut sum_sq = 0.0_f64;

    for &v in samples {
        min = min.min(v);
        max = max.max(v);
        sum += v;
        sum_sq += v * v;
    }

    let n = count as f64;
    let mean = sum / n;
    let variance = (sum_sq / n) - (mean * mean);
    let std_dev = if variance > 0.0 { variance.sqrt() } else { 0.0 };

    // Compute median via sorting a copy
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = if sorted.len() % 2 == 0 {
        let mid = sorted.len() / 2;
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[sorted.len() / 2]
    };

    ChannelStats {
        min,
        max,
        mean,
        std_dev,
        median,
        range: max - min,
        count,
    }
}

/// Compute a specific percentile (0.0 - 100.0) from a sorted slice.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
#[must_use]
pub fn percentile(sorted: &[f64], pct: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let pct = pct.clamp(0.0, 100.0);
    let idx = pct / 100.0 * (sorted.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil().min((sorted.len() - 1) as f64) as usize;
    let frac = idx - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// Compute full-frame signal statistics from an RGB24 buffer.
///
/// * `frame` — RGB24 pixel data (3 bytes per pixel)
/// * `width` — frame width in pixels
/// * `height` — frame height in pixels
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn compute_frame_stats(frame: &[u8], width: u32, height: u32) -> FrameSignalStats {
    let num_pixels = (width as u64) * (height as u64);
    let expected = num_pixels as usize * 3;

    let actual_pixels = if frame.len() >= expected {
        num_pixels as usize
    } else {
        frame.len() / 3
    };

    let mut r_samples = Vec::with_capacity(actual_pixels);
    let mut g_samples = Vec::with_capacity(actual_pixels);
    let mut b_samples = Vec::with_capacity(actual_pixels);
    let mut y_samples = Vec::with_capacity(actual_pixels);

    for i in 0..actual_pixels {
        let r = f64::from(frame[i * 3]);
        let g = f64::from(frame[i * 3 + 1]);
        let b = f64::from(frame[i * 3 + 2]);
        // Rec.709 luma
        let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        r_samples.push(r);
        g_samples.push(g);
        b_samples.push(b);
        y_samples.push(y);
    }

    FrameSignalStats {
        red: compute_channel_stats(&r_samples),
        green: compute_channel_stats(&g_samples),
        blue: compute_channel_stats(&b_samples),
        luma: compute_channel_stats(&y_samples),
        total_pixels: actual_pixels as u64,
        width,
        height,
    }
}

/// Compute the signal-to-noise ratio estimate from a flat region.
///
/// Given a region where the signal should be constant, SNR is estimated as
/// mean / std_dev (linear), returned in dB.
#[must_use]
pub fn estimate_snr_db(stats: &ChannelStats) -> f64 {
    if stats.std_dev <= 0.0 || stats.mean <= 0.0 {
        return f64::INFINITY;
    }
    20.0 * (stats.mean / stats.std_dev).log10()
}

/// Compute dynamic range from min/max in dB-like scale.
///
/// Assumes values are in 0-255 range (8-bit).
#[must_use]
pub fn dynamic_range_db(stats: &ChannelStats) -> f64 {
    if stats.min <= 0.0 || stats.max <= 0.0 {
        return 0.0;
    }
    20.0 * (stats.max / stats.min).log10()
}

/// Determine whether a frame is predominantly dark (average luma below threshold).
#[must_use]
pub fn is_dark_frame(luma_stats: &ChannelStats, threshold: f64) -> bool {
    luma_stats.mean < threshold
}

/// Determine whether a frame is predominantly bright (average luma above threshold).
#[must_use]
pub fn is_bright_frame(luma_stats: &ChannelStats, threshold: f64) -> bool {
    luma_stats.mean > threshold
}

/// Determine whether a frame has low contrast (small luma range).
#[must_use]
pub fn is_low_contrast(luma_stats: &ChannelStats, min_range: f64) -> bool {
    luma_stats.range < min_range
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_channel_stats_basic() {
        let samples = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let stats = compute_channel_stats(&samples);
        assert!((stats.min - 10.0).abs() < f64::EPSILON);
        assert!((stats.max - 50.0).abs() < f64::EPSILON);
        assert!((stats.mean - 30.0).abs() < f64::EPSILON);
        assert!((stats.median - 30.0).abs() < f64::EPSILON);
        assert!((stats.range - 40.0).abs() < f64::EPSILON);
        assert_eq!(stats.count, 5);
    }

    #[test]
    fn test_compute_channel_stats_single() {
        let samples = vec![42.0];
        let stats = compute_channel_stats(&samples);
        assert!((stats.min - 42.0).abs() < f64::EPSILON);
        assert!((stats.max - 42.0).abs() < f64::EPSILON);
        assert!((stats.std_dev).abs() < f64::EPSILON);
    }

    #[test]
    fn test_compute_channel_stats_empty() {
        let samples: Vec<f64> = vec![];
        let stats = compute_channel_stats(&samples);
        assert_eq!(stats.count, 0);
    }

    #[test]
    fn test_compute_channel_stats_even_median() {
        let samples = vec![1.0, 2.0, 3.0, 4.0];
        let stats = compute_channel_stats(&samples);
        assert!((stats.median - 2.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_percentile_basic() {
        let sorted = vec![0.0, 25.0, 50.0, 75.0, 100.0];
        assert!((percentile(&sorted, 0.0)).abs() < f64::EPSILON);
        assert!((percentile(&sorted, 50.0) - 50.0).abs() < f64::EPSILON);
        assert!((percentile(&sorted, 100.0) - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_percentile_interpolation() {
        let sorted = vec![0.0, 100.0];
        assert!((percentile(&sorted, 50.0) - 50.0).abs() < f64::EPSILON);
        assert!((percentile(&sorted, 25.0) - 25.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_percentile_empty() {
        let sorted: Vec<f64> = vec![];
        assert!((percentile(&sorted, 50.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_compute_frame_stats_black() {
        let frame = vec![0u8; 4 * 4 * 3]; // 4x4 black
        let stats = compute_frame_stats(&frame, 4, 4);
        assert_eq!(stats.total_pixels, 16);
        assert!((stats.luma.mean).abs() < f64::EPSILON);
        assert!((stats.red.mean).abs() < f64::EPSILON);
    }

    #[test]
    fn test_compute_frame_stats_white() {
        let frame = vec![255u8; 2 * 2 * 3]; // 2x2 white
        let stats = compute_frame_stats(&frame, 2, 2);
        assert!((stats.red.mean - 255.0).abs() < f64::EPSILON);
        assert!((stats.green.mean - 255.0).abs() < f64::EPSILON);
        assert!((stats.blue.mean - 255.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_estimate_snr_db_flat() {
        let samples = vec![100.0; 100];
        let stats = compute_channel_stats(&samples);
        let snr = estimate_snr_db(&stats);
        assert!(snr.is_infinite());
    }

    #[test]
    fn test_dynamic_range_db() {
        let stats = ChannelStats {
            min: 1.0,
            max: 100.0,
            mean: 50.0,
            std_dev: 10.0,
            median: 50.0,
            range: 99.0,
            count: 100,
        };
        let dr = dynamic_range_db(&stats);
        assert!((dr - 40.0).abs() < 0.01);
    }

    #[test]
    fn test_is_dark_frame() {
        let stats = ChannelStats {
            mean: 15.0,
            ..ChannelStats::default()
        };
        assert!(is_dark_frame(&stats, 20.0));
        assert!(!is_dark_frame(&stats, 10.0));
    }

    #[test]
    fn test_is_bright_frame() {
        let stats = ChannelStats {
            mean: 230.0,
            ..ChannelStats::default()
        };
        assert!(is_bright_frame(&stats, 200.0));
        assert!(!is_bright_frame(&stats, 240.0));
    }

    #[test]
    fn test_is_low_contrast() {
        let stats = ChannelStats {
            range: 10.0,
            ..ChannelStats::default()
        };
        assert!(is_low_contrast(&stats, 20.0));
        assert!(!is_low_contrast(&stats, 5.0));
    }
}
