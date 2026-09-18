#![allow(dead_code)]
//! Histogram statistics for video frames.
//!
//! Provides:
//! * [`HistogramChannel`] – channel selector with display label.
//! * [`HistogramBin`]     – a single histogram bucket with occupancy percentage.
//! * [`ImageHistogram`]   – full histogram with per-channel access and clipping metrics.
//! * [`HistogramStats`]   – derived statistics including Shannon entropy.

// ---------------------------------------------------------------------------
// HistogramChannel
// ---------------------------------------------------------------------------

/// Selects which channel to analyse.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HistogramChannel {
    /// Red channel (byte index 0 of RGB24).
    Red,
    /// Green channel (byte index 1 of RGB24).
    Green,
    /// Blue channel (byte index 2 of RGB24).
    Blue,
    /// Perceptual luma (Y′ from BT.709 coefficients).
    Luma,
}

impl HistogramChannel {
    /// Display label for the channel.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Red => "R",
            Self::Green => "G",
            Self::Blue => "B",
            Self::Luma => "Y",
        }
    }

    /// `true` when this channel maps directly to a single byte index.
    #[must_use]
    pub const fn is_direct(self) -> bool {
        !matches!(self, Self::Luma)
    }
}

// ---------------------------------------------------------------------------
// HistogramBin
// ---------------------------------------------------------------------------

/// A single bucket in an 8-bit histogram (256 bins, values 0–255).
#[derive(Debug, Clone, Copy)]
pub struct HistogramBin {
    /// Bin index `[0, 255]`.
    pub index: u8,
    /// Number of pixels in this bin.
    pub count: u64,
    /// Total pixels in the histogram (for percentage calculation).
    pub total: u64,
}

impl HistogramBin {
    /// Create a bin.
    #[must_use]
    pub const fn new(index: u8, count: u64, total: u64) -> Self {
        Self {
            index,
            count,
            total,
        }
    }

    /// Fraction of total pixels in this bin `[0.0, 1.0]`.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn occupancy_pct(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        self.count as f64 / self.total as f64
    }

    /// Normalised bin value `[0.0, 1.0]` corresponding to `index / 255`.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn normalised_value(&self) -> f64 {
        self.index as f64 / 255.0
    }
}

// ---------------------------------------------------------------------------
// ImageHistogram
// ---------------------------------------------------------------------------

const BINS: usize = 256;

/// 8-bit histogram for all channels of an RGB24 frame.
#[derive(Debug, Clone)]
pub struct ImageHistogram {
    /// Counts for R channel (bins 0–255).
    red: [u64; BINS],
    /// Counts for G channel.
    green: [u64; BINS],
    /// Counts for B channel.
    blue: [u64; BINS],
    /// Counts for luma channel.
    luma: [u64; BINS],
    /// Total number of pixels.
    pixel_count: u64,
}

impl ImageHistogram {
    /// Build a histogram from an RGB24 frame.
    ///
    /// `frame` must have `width * height * 3` bytes.
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    #[must_use]
    pub fn from_rgb24(frame: &[u8], width: usize, height: usize) -> Self {
        let mut red = [0u64; BINS];
        let mut green = [0u64; BINS];
        let mut blue = [0u64; BINS];
        let mut luma = [0u64; BINS];

        let total = width * height;
        let actual = (frame.len() / 3).min(total);

        for i in 0..actual {
            let r = frame[i * 3];
            let g = frame[i * 3 + 1];
            let b = frame[i * 3 + 2];
            red[r as usize] += 1;
            green[g as usize] += 1;
            blue[b as usize] += 1;
            // BT.709 luma, scaled to 0–255
            let y = 0.2126 * r as f64 + 0.7152 * g as f64 + 0.0722 * b as f64;
            let y_idx = (y.round() as usize).min(255);
            luma[y_idx] += 1;
        }

        Self {
            red,
            green,
            blue,
            luma,
            pixel_count: actual as u64,
        }
    }

    /// Returns the histogram bins for the given channel as an array of [`HistogramBin`].
    #[must_use]
    pub fn for_channel(&self, channel: HistogramChannel) -> Vec<HistogramBin> {
        let counts = self.raw_counts(channel);
        (0u8..=255)
            .map(|i| HistogramBin::new(i, counts[i as usize], self.pixel_count))
            .collect()
    }

    /// Raw counts array for a channel.
    #[must_use]
    pub fn raw_counts(&self, channel: HistogramChannel) -> &[u64; BINS] {
        match channel {
            HistogramChannel::Red => &self.red,
            HistogramChannel::Green => &self.green,
            HistogramChannel::Blue => &self.blue,
            HistogramChannel::Luma => &self.luma,
        }
    }

    /// Fraction of pixels clipping to pure white (bin 255) for `channel`.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn clipping_pct(&self, channel: HistogramChannel) -> f64 {
        if self.pixel_count == 0 {
            return 0.0;
        }
        let counts = self.raw_counts(channel);
        counts[255] as f64 / self.pixel_count as f64
    }

    /// Fraction of pixels crushed to pure black (bin 0) for `channel`.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn crushing_pct(&self, channel: HistogramChannel) -> f64 {
        if self.pixel_count == 0 {
            return 0.0;
        }
        let counts = self.raw_counts(channel);
        counts[0] as f64 / self.pixel_count as f64
    }

    /// Total pixels analysed.
    #[must_use]
    pub fn pixel_count(&self) -> u64 {
        self.pixel_count
    }
}

// ---------------------------------------------------------------------------
// HistogramStats
// ---------------------------------------------------------------------------

/// Derived statistical measures for a single histogram channel.
#[derive(Debug, Clone)]
pub struct HistogramStats {
    /// Mean pixel value `[0.0, 255.0]`.
    pub mean: f64,
    /// Standard deviation.
    pub std_dev: f64,
    /// Shannon entropy in bits.
    pub entropy: f64,
    /// Median bin index.
    pub median: u8,
    /// Mode bin index (most frequent value).
    pub mode: u8,
    /// Bin index at the 1st percentile.
    pub p01: u8,
    /// Bin index at the 99th percentile.
    pub p99: u8,
}

impl HistogramStats {
    /// Compute stats from raw bin counts.
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    #[must_use]
    pub fn from_counts(counts: &[u64; BINS]) -> Self {
        let total: u64 = counts.iter().sum();
        if total == 0 {
            return Self {
                mean: 0.0,
                std_dev: 0.0,
                entropy: 0.0,
                median: 0,
                mode: 0,
                p01: 0,
                p99: 0,
            };
        }

        // Mean
        let mean: f64 = counts
            .iter()
            .enumerate()
            .map(|(i, &c)| i as f64 * c as f64)
            .sum::<f64>()
            / total as f64;

        // Std dev
        let variance: f64 = counts
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                let diff = i as f64 - mean;
                diff * diff * c as f64
            })
            .sum::<f64>()
            / total as f64;
        let std_dev = variance.sqrt();

        // Shannon entropy
        let entropy: f64 = counts
            .iter()
            .filter(|&&c| c > 0)
            .map(|&c| {
                let p = c as f64 / total as f64;
                -p * p.log2()
            })
            .sum();

        // Mode
        let mode = counts
            .iter()
            .enumerate()
            .max_by_key(|(_, &c)| c)
            .map(|(i, _)| i as u8)
            .unwrap_or(0);

        // Percentile helper (CDF walk)
        let percentile = |pct: f64| -> u8 {
            let target = (total as f64 * pct).ceil() as u64;
            let mut cum = 0u64;
            for (i, &c) in counts.iter().enumerate() {
                cum += c;
                if cum >= target {
                    return i as u8;
                }
            }
            255
        };

        let median = percentile(0.50);
        let p01 = percentile(0.01);
        let p99 = percentile(0.99);

        Self {
            mean,
            std_dev,
            entropy,
            median,
            mode,
            p01,
            p99,
        }
    }

    /// Shannon entropy in bits.
    #[must_use]
    pub fn entropy(&self) -> f64 {
        self.entropy
    }
}

// ---------------------------------------------------------------------------
// Percentile markers overlay for histogram display
// ---------------------------------------------------------------------------

/// Percentile marker positions for histogram overlay rendering.
///
/// Contains the bin index (0-255) for each of the key percentile markers.
#[derive(Debug, Clone, Copy)]
pub struct PercentileMarkers {
    /// 1st percentile bin index.
    pub p01: u8,
    /// 5th percentile bin index.
    pub p05: u8,
    /// 50th percentile (median) bin index.
    pub p50: u8,
    /// 95th percentile bin index.
    pub p95: u8,
    /// 99th percentile bin index.
    pub p99: u8,
}

impl PercentileMarkers {
    /// Computes percentile markers from raw bin counts.
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    #[must_use]
    pub fn from_counts(counts: &[u64; BINS]) -> Self {
        let total: u64 = counts.iter().sum();
        if total == 0 {
            return Self {
                p01: 0,
                p05: 0,
                p50: 0,
                p95: 0,
                p99: 0,
            };
        }

        let percentile = |pct: f64| -> u8 {
            let target = (total as f64 * pct).ceil() as u64;
            let mut cum = 0u64;
            for (i, &c) in counts.iter().enumerate() {
                cum += c;
                if cum >= target {
                    return i as u8;
                }
            }
            255
        };

        Self {
            p01: percentile(0.01),
            p05: percentile(0.05),
            p50: percentile(0.50),
            p95: percentile(0.95),
            p99: percentile(0.99),
        }
    }

    /// Computes percentile markers from an `ImageHistogram` for a given channel.
    #[must_use]
    pub fn from_histogram(hist: &ImageHistogram, channel: HistogramChannel) -> Self {
        Self::from_counts(hist.raw_counts(channel))
    }

    /// Returns all marker positions as `(label, bin_index)` pairs.
    #[must_use]
    pub fn as_labeled_pairs(&self) -> Vec<(&'static str, u8)> {
        vec![
            ("P01", self.p01),
            ("P05", self.p05),
            ("P50", self.p50),
            ("P95", self.p95),
            ("P99", self.p99),
        ]
    }

    /// Returns the inter-percentile range (P99 - P01), i.e., the effective
    /// dynamic range of the data excluding extreme outliers.
    #[must_use]
    pub fn inter_percentile_range(&self) -> u8 {
        self.p99.saturating_sub(self.p01)
    }

    /// Returns the inter-quartile-like range (P95 - P05).
    #[must_use]
    pub fn robust_range(&self) -> u8 {
        self.p95.saturating_sub(self.p05)
    }
}

/// Histogram overlay data with percentile markers for all channels.
#[derive(Debug, Clone)]
pub struct HistogramOverlay {
    /// Percentile markers for the red channel.
    pub red: PercentileMarkers,
    /// Percentile markers for the green channel.
    pub green: PercentileMarkers,
    /// Percentile markers for the blue channel.
    pub blue: PercentileMarkers,
    /// Percentile markers for the luma channel.
    pub luma: PercentileMarkers,
}

impl HistogramOverlay {
    /// Computes percentile markers for all channels from an `ImageHistogram`.
    #[must_use]
    pub fn from_histogram(hist: &ImageHistogram) -> Self {
        Self {
            red: PercentileMarkers::from_histogram(hist, HistogramChannel::Red),
            green: PercentileMarkers::from_histogram(hist, HistogramChannel::Green),
            blue: PercentileMarkers::from_histogram(hist, HistogramChannel::Blue),
            luma: PercentileMarkers::from_histogram(hist, HistogramChannel::Luma),
        }
    }

    /// Builds the overlay directly from an RGB24 frame.
    #[must_use]
    pub fn from_rgb24(frame: &[u8], width: usize, height: usize) -> Self {
        let hist = ImageHistogram::from_rgb24(frame, width, height);
        Self::from_histogram(&hist)
    }

    /// Returns the luma dynamic range (P99 - P01) in code values.
    #[must_use]
    pub fn luma_dynamic_range(&self) -> u8 {
        self.luma.inter_percentile_range()
    }

    /// Returns true if any channel has P99 at 255 (potential highlight clipping).
    #[must_use]
    pub fn has_highlight_clipping(&self) -> bool {
        self.red.p99 == 255 || self.green.p99 == 255 || self.blue.p99 == 255
    }

    /// Returns true if any channel has P01 at 0 (potential shadow crushing).
    #[must_use]
    pub fn has_shadow_crushing(&self) -> bool {
        self.red.p01 == 0 || self.green.p01 == 0 || self.blue.p01 == 0
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_label() {
        assert_eq!(HistogramChannel::Red.label(), "R");
        assert_eq!(HistogramChannel::Green.label(), "G");
        assert_eq!(HistogramChannel::Blue.label(), "B");
        assert_eq!(HistogramChannel::Luma.label(), "Y");
    }

    #[test]
    fn test_channel_is_direct() {
        assert!(HistogramChannel::Red.is_direct());
        assert!(!HistogramChannel::Luma.is_direct());
    }

    #[test]
    fn test_histogram_bin_occupancy_pct() {
        let bin = HistogramBin::new(128, 50, 100);
        assert!((bin.occupancy_pct() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_histogram_bin_occupancy_zero_total() {
        let bin = HistogramBin::new(0, 0, 0);
        assert_eq!(bin.occupancy_pct(), 0.0);
    }

    #[test]
    fn test_histogram_bin_normalised_value() {
        let bin = HistogramBin::new(255, 1, 1);
        assert!((bin.normalised_value() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_image_histogram_from_black_frame() {
        let frame = vec![0u8; 4 * 4 * 3];
        let hist = ImageHistogram::from_rgb24(&frame, 4, 4);
        assert_eq!(hist.pixel_count(), 16);
        assert_eq!(hist.raw_counts(HistogramChannel::Red)[0], 16);
        assert_eq!(hist.raw_counts(HistogramChannel::Red)[1], 0);
    }

    #[test]
    fn test_image_histogram_from_white_frame() {
        let frame = vec![255u8; 2 * 2 * 3];
        let hist = ImageHistogram::from_rgb24(&frame, 2, 2);
        assert_eq!(hist.raw_counts(HistogramChannel::Red)[255], 4);
    }

    #[test]
    fn test_image_histogram_clipping_pct_white() {
        let frame = vec![255u8; 4 * 3];
        let hist = ImageHistogram::from_rgb24(&frame, 4, 1);
        assert!((hist.clipping_pct(HistogramChannel::Red) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_image_histogram_clipping_pct_black() {
        let frame = vec![0u8; 4 * 3];
        let hist = ImageHistogram::from_rgb24(&frame, 4, 1);
        assert_eq!(hist.clipping_pct(HistogramChannel::Red), 0.0);
    }

    #[test]
    fn test_image_histogram_for_channel_length() {
        let frame = vec![128u8; 8 * 3];
        let hist = ImageHistogram::from_rgb24(&frame, 8, 1);
        let bins = hist.for_channel(HistogramChannel::Green);
        assert_eq!(bins.len(), 256);
    }

    #[test]
    fn test_image_histogram_crushing_pct() {
        let frame = vec![0u8; 8 * 3];
        let hist = ImageHistogram::from_rgb24(&frame, 8, 1);
        assert!((hist.crushing_pct(HistogramChannel::Blue) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_uniform_entropy_max() {
        // A perfectly uniform distribution has the highest entropy for 256 bins.
        let counts = [1u64; BINS];
        let stats = HistogramStats::from_counts(&counts);
        // Entropy should be close to log2(256) = 8 bits
        assert!(stats.entropy() > 7.9, "entropy was {}", stats.entropy());
    }

    #[test]
    fn test_stats_single_bin_entropy_zero() {
        let mut counts = [0u64; BINS];
        counts[128] = 100;
        let stats = HistogramStats::from_counts(&counts);
        assert!(stats.entropy() < 1e-9);
    }

    #[test]
    fn test_stats_mean_midpoint() {
        let mut counts = [0u64; BINS];
        counts[100] = 50;
        counts[156] = 50;
        let stats = HistogramStats::from_counts(&counts);
        assert!((stats.mean - 128.0).abs() < 1.0);
    }

    #[test]
    fn test_stats_empty_counts() {
        let counts = [0u64; BINS];
        let stats = HistogramStats::from_counts(&counts);
        assert_eq!(stats.mean, 0.0);
        assert_eq!(stats.entropy(), 0.0);
    }

    // ── PercentileMarkers tests ──────────────────────────────────────

    #[test]
    fn test_percentile_markers_uniform() {
        let counts = [1u64; BINS];
        let m = PercentileMarkers::from_counts(&counts);
        // P01 ≈ bin 2 (ceil(256*0.01)=3), P99 ≈ bin 252
        assert!(m.p01 < 10);
        assert!(m.p99 > 245);
        assert!((m.p50 as i16 - 127).abs() < 3);
    }

    #[test]
    fn test_percentile_markers_single_bin() {
        let mut counts = [0u64; BINS];
        counts[100] = 1000;
        let m = PercentileMarkers::from_counts(&counts);
        assert_eq!(m.p01, 100);
        assert_eq!(m.p05, 100);
        assert_eq!(m.p50, 100);
        assert_eq!(m.p95, 100);
        assert_eq!(m.p99, 100);
    }

    #[test]
    fn test_percentile_markers_empty() {
        let counts = [0u64; BINS];
        let m = PercentileMarkers::from_counts(&counts);
        assert_eq!(m.p01, 0);
        assert_eq!(m.p50, 0);
        assert_eq!(m.p99, 0);
    }

    #[test]
    fn test_percentile_markers_two_bins() {
        let mut counts = [0u64; BINS];
        counts[0] = 50;
        counts[255] = 50;
        let m = PercentileMarkers::from_counts(&counts);
        assert_eq!(m.p01, 0);
        assert_eq!(m.p50, 0); // 50% threshold lands on first bin
        assert_eq!(m.p99, 255);
    }

    #[test]
    fn test_inter_percentile_range() {
        let mut counts = [0u64; BINS];
        counts[10] = 500;
        counts[200] = 500;
        let m = PercentileMarkers::from_counts(&counts);
        assert!(m.inter_percentile_range() > 100);
    }

    #[test]
    fn test_robust_range() {
        let mut counts = [0u64; BINS];
        counts[20] = 100;
        counts[180] = 100;
        let m = PercentileMarkers::from_counts(&counts);
        assert!(m.robust_range() > 100);
    }

    #[test]
    fn test_as_labeled_pairs() {
        let mut counts = [0u64; BINS];
        counts[128] = 100;
        let m = PercentileMarkers::from_counts(&counts);
        let pairs = m.as_labeled_pairs();
        assert_eq!(pairs.len(), 5);
        assert_eq!(pairs[0].0, "P01");
        assert_eq!(pairs[4].0, "P99");
    }

    // ── HistogramOverlay tests ───────────────────────────────────────

    #[test]
    fn test_histogram_overlay_from_rgb24() {
        let frame = vec![128u8; 100 * 3];
        let overlay = HistogramOverlay::from_rgb24(&frame, 100, 1);
        // All channels should have P01..P99 at 128
        assert_eq!(overlay.red.p50, 128);
        assert_eq!(overlay.green.p50, 128);
        assert_eq!(overlay.blue.p50, 128);
    }

    #[test]
    fn test_histogram_overlay_from_histogram() {
        let frame = vec![128u8; 100 * 3];
        let hist = ImageHistogram::from_rgb24(&frame, 100, 1);
        let overlay = HistogramOverlay::from_histogram(&hist);
        assert_eq!(overlay.luma.p50, 128);
    }

    #[test]
    fn test_luma_dynamic_range() {
        // All same value => range 0
        let frame = vec![100u8; 100 * 3];
        let overlay = HistogramOverlay::from_rgb24(&frame, 100, 1);
        assert_eq!(overlay.luma_dynamic_range(), 0);
    }

    #[test]
    fn test_has_highlight_clipping_true() {
        let frame = vec![255u8; 100 * 3];
        let overlay = HistogramOverlay::from_rgb24(&frame, 100, 1);
        assert!(overlay.has_highlight_clipping());
    }

    #[test]
    fn test_has_highlight_clipping_false() {
        let frame = vec![128u8; 100 * 3];
        let overlay = HistogramOverlay::from_rgb24(&frame, 100, 1);
        assert!(!overlay.has_highlight_clipping());
    }

    #[test]
    fn test_has_shadow_crushing_true() {
        let frame = vec![0u8; 100 * 3];
        let overlay = HistogramOverlay::from_rgb24(&frame, 100, 1);
        assert!(overlay.has_shadow_crushing());
    }

    #[test]
    fn test_has_shadow_crushing_false() {
        let frame = vec![128u8; 100 * 3];
        let overlay = HistogramOverlay::from_rgb24(&frame, 100, 1);
        assert!(!overlay.has_shadow_crushing());
    }

    #[test]
    fn test_gradient_percentile_markers() {
        // 256 pixels with values 0..255
        let mut frame = Vec::with_capacity(256 * 3);
        for i in 0u8..=255 {
            frame.push(i);
            frame.push(i);
            frame.push(i);
        }
        let overlay = HistogramOverlay::from_rgb24(&frame, 256, 1);
        // P05 ~= 12, P95 ~= 242
        assert!(overlay.luma.p05 < 20);
        assert!(overlay.luma.p95 > 230);
        assert!(overlay.luma_dynamic_range() > 200);
    }
}
