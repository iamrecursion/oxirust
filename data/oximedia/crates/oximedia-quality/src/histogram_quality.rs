#![allow(dead_code)]
//! Histogram-based video quality analysis.
//!
//! Provides tools for analyzing frame quality through histogram statistics,
//! including dynamic range, contrast, clipping detection, and exposure analysis.

/// Number of bins in a standard 8-bit histogram.
const NUM_BINS: usize = 256;

/// Histogram computed from a single channel of pixel data.
#[derive(Debug, Clone)]
pub struct ChannelHistogram {
    /// Bin counts (index 0..255 maps to pixel value 0..255).
    pub bins: [u64; NUM_BINS],
    /// Total number of pixels.
    pub total_pixels: u64,
}

impl ChannelHistogram {
    /// Creates a histogram from raw 8-bit pixel data.
    #[must_use]
    pub fn from_data(data: &[u8]) -> Self {
        let mut bins = [0u64; NUM_BINS];
        for &v in data {
            bins[v as usize] += 1;
        }
        Self {
            bins,
            total_pixels: data.len() as u64,
        }
    }

    /// Creates an empty histogram.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            bins: [0u64; NUM_BINS],
            total_pixels: 0,
        }
    }

    /// Returns the mean pixel value.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn mean(&self) -> f64 {
        if self.total_pixels == 0 {
            return 0.0;
        }
        let sum: u64 = self
            .bins
            .iter()
            .enumerate()
            .map(|(i, &c)| i as u64 * c)
            .sum();
        sum as f64 / self.total_pixels as f64
    }

    /// Returns the standard deviation of pixel values.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn std_dev(&self) -> f64 {
        if self.total_pixels == 0 {
            return 0.0;
        }
        let mean = self.mean();
        let variance: f64 = self
            .bins
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                let diff = i as f64 - mean;
                diff * diff * c as f64
            })
            .sum::<f64>()
            / self.total_pixels as f64;
        variance.sqrt()
    }

    /// Returns the median pixel value.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn median(&self) -> f64 {
        if self.total_pixels == 0 {
            return 0.0;
        }
        let half = self.total_pixels / 2;
        let mut cumulative = 0u64;
        for (i, &c) in self.bins.iter().enumerate() {
            cumulative += c;
            if cumulative >= half {
                return i as f64;
            }
        }
        255.0
    }

    /// Returns the specified percentile (0..100).
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn percentile(&self, p: f64) -> f64 {
        if self.total_pixels == 0 {
            return 0.0;
        }
        let target = (p / 100.0 * self.total_pixels as f64) as u64;
        let target = target.max(1);
        let mut cumulative = 0u64;
        for (i, &c) in self.bins.iter().enumerate() {
            cumulative += c;
            if cumulative >= target {
                return i as f64;
            }
        }
        255.0
    }

    /// Returns the minimum non-zero bin index.
    #[must_use]
    pub fn min_value(&self) -> u8 {
        for (i, &c) in self.bins.iter().enumerate() {
            if c > 0 {
                return i as u8;
            }
        }
        0
    }

    /// Returns the maximum non-zero bin index.
    #[must_use]
    pub fn max_value(&self) -> u8 {
        for i in (0..NUM_BINS).rev() {
            if self.bins[i] > 0 {
                return i as u8;
            }
        }
        0
    }
}

/// Dynamic range analysis result.
#[derive(Debug, Clone)]
pub struct DynamicRangeResult {
    /// The minimum non-zero pixel value.
    pub min_value: u8,
    /// The maximum non-zero pixel value.
    pub max_value: u8,
    /// Range span (max - min).
    pub range: u8,
    /// Range as a fraction of the full 0-255 range (0.0 to 1.0).
    pub range_ratio: f64,
    /// 1st percentile value (shadow clip point).
    pub percentile_1: f64,
    /// 99th percentile value (highlight clip point).
    pub percentile_99: f64,
}

/// Computes the dynamic range of a channel.
#[must_use]
pub fn analyze_dynamic_range(hist: &ChannelHistogram) -> DynamicRangeResult {
    let min_val = hist.min_value();
    let max_val = hist.max_value();
    let range = max_val.saturating_sub(min_val);
    #[allow(clippy::cast_precision_loss)]
    let range_ratio = range as f64 / 255.0;
    DynamicRangeResult {
        min_value: min_val,
        max_value: max_val,
        range,
        range_ratio,
        percentile_1: hist.percentile(1.0),
        percentile_99: hist.percentile(99.0),
    }
}

/// Clipping analysis result.
#[derive(Debug, Clone)]
pub struct ClippingResult {
    /// Fraction of pixels clipped at the black level (value 0).
    pub black_clip_ratio: f64,
    /// Fraction of pixels clipped at the white level (value 255).
    pub white_clip_ratio: f64,
    /// Total clipping ratio.
    pub total_clip_ratio: f64,
    /// Whether clipping is considered significant (> threshold).
    pub is_significant: bool,
}

/// Analyzes clipping in a histogram.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn analyze_clipping(hist: &ChannelHistogram, threshold: f64) -> ClippingResult {
    let total = hist.total_pixels.max(1) as f64;
    let black_clip = hist.bins[0] as f64 / total;
    let white_clip = hist.bins[255] as f64 / total;
    let total_clip = black_clip + white_clip;
    ClippingResult {
        black_clip_ratio: black_clip,
        white_clip_ratio: white_clip,
        total_clip_ratio: total_clip,
        is_significant: total_clip > threshold,
    }
}

/// Contrast score based on histogram spread.
#[derive(Debug, Clone)]
pub struct ContrastScore {
    /// RMS contrast (standard deviation of pixel values / 255).
    pub rms_contrast: f64,
    /// Michelson contrast: (max - min) / (max + min).
    pub michelson_contrast: f64,
    /// Overall contrast quality score (0.0 to 1.0).
    pub quality_score: f64,
}

/// Computes a contrast quality score from histogram statistics.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn compute_contrast(hist: &ChannelHistogram) -> ContrastScore {
    let std = hist.std_dev();
    let rms_contrast = std / 255.0;

    let min_val = hist.min_value() as f64;
    let max_val = hist.max_value() as f64;
    let michelson = if (max_val + min_val) > 0.0 {
        (max_val - min_val) / (max_val + min_val)
    } else {
        0.0
    };

    // Quality score: prefer std_dev around 50-70 (well-distributed histogram)
    // Penalize very low or very high std_dev
    let ideal_std = 60.0;
    let quality = 1.0 - ((std - ideal_std).abs() / ideal_std).min(1.0);

    ContrastScore {
        rms_contrast,
        michelson_contrast: michelson,
        quality_score: quality,
    }
}

/// Exposure analysis result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExposureRating {
    /// Frame is significantly underexposed.
    Underexposed,
    /// Frame is slightly dark.
    SlightlyDark,
    /// Frame has normal exposure.
    Normal,
    /// Frame is slightly bright.
    SlightlyBright,
    /// Frame is significantly overexposed.
    Overexposed,
}

/// Analyzes exposure based on histogram mean.
#[must_use]
pub fn analyze_exposure(hist: &ChannelHistogram) -> ExposureRating {
    let mean = hist.mean();
    if mean < 40.0 {
        ExposureRating::Underexposed
    } else if mean < 80.0 {
        ExposureRating::SlightlyDark
    } else if mean < 176.0 {
        ExposureRating::Normal
    } else if mean < 216.0 {
        ExposureRating::SlightlyBright
    } else {
        ExposureRating::Overexposed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform_data() -> Vec<u8> {
        // Every value from 0 to 255 appears exactly once
        (0..=255u8).collect()
    }

    fn dark_data() -> Vec<u8> {
        vec![0u8; 200]
    }

    fn bright_data() -> Vec<u8> {
        vec![255u8; 200]
    }

    fn mid_data() -> Vec<u8> {
        vec![128u8; 200]
    }

    #[test]
    fn test_histogram_from_data() {
        let data = uniform_data();
        let hist = ChannelHistogram::from_data(&data);
        assert_eq!(hist.total_pixels, 256);
        assert_eq!(hist.bins[0], 1);
        assert_eq!(hist.bins[255], 1);
    }

    #[test]
    fn test_histogram_empty() {
        let hist = ChannelHistogram::empty();
        assert_eq!(hist.total_pixels, 0);
        assert!((hist.mean()).abs() < 1e-10);
        assert!((hist.std_dev()).abs() < 1e-10);
    }

    #[test]
    fn test_mean_uniform() {
        let hist = ChannelHistogram::from_data(&uniform_data());
        let mean = hist.mean();
        assert!((mean - 127.5).abs() < 0.01);
    }

    #[test]
    fn test_mean_constant() {
        let hist = ChannelHistogram::from_data(&mid_data());
        let mean = hist.mean();
        assert!((mean - 128.0).abs() < 0.01);
    }

    #[test]
    fn test_std_dev_constant() {
        let hist = ChannelHistogram::from_data(&mid_data());
        assert!((hist.std_dev()).abs() < 0.01);
    }

    #[test]
    fn test_std_dev_uniform() {
        let hist = ChannelHistogram::from_data(&uniform_data());
        // std dev of uniform 0..255 is ~73.9
        assert!(hist.std_dev() > 70.0 && hist.std_dev() < 78.0);
    }

    #[test]
    fn test_median() {
        let hist = ChannelHistogram::from_data(&uniform_data());
        let med = hist.median();
        assert!((med - 128.0).abs() < 2.0);
    }

    #[test]
    fn test_percentile() {
        let hist = ChannelHistogram::from_data(&uniform_data());
        let p50 = hist.percentile(50.0);
        assert!((p50 - 128.0).abs() < 2.0);
    }

    #[test]
    fn test_min_max_value() {
        let hist = ChannelHistogram::from_data(&uniform_data());
        assert_eq!(hist.min_value(), 0);
        assert_eq!(hist.max_value(), 255);
    }

    #[test]
    fn test_dynamic_range() {
        let hist = ChannelHistogram::from_data(&uniform_data());
        let dr = analyze_dynamic_range(&hist);
        assert_eq!(dr.range, 255);
        assert!((dr.range_ratio - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_clipping_dark() {
        let hist = ChannelHistogram::from_data(&dark_data());
        let clip = analyze_clipping(&hist, 0.01);
        assert!((clip.black_clip_ratio - 1.0).abs() < 0.01);
        assert!(clip.is_significant);
    }

    #[test]
    fn test_clipping_normal() {
        let hist = ChannelHistogram::from_data(&mid_data());
        let clip = analyze_clipping(&hist, 0.01);
        assert!((clip.black_clip_ratio).abs() < 0.01);
        assert!((clip.white_clip_ratio).abs() < 0.01);
        assert!(!clip.is_significant);
    }

    #[test]
    fn test_contrast_constant() {
        let hist = ChannelHistogram::from_data(&mid_data());
        let c = compute_contrast(&hist);
        assert!(c.rms_contrast < 0.01);
    }

    #[test]
    fn test_exposure_dark() {
        let hist = ChannelHistogram::from_data(&dark_data());
        assert_eq!(analyze_exposure(&hist), ExposureRating::Underexposed);
    }

    #[test]
    fn test_exposure_bright() {
        let hist = ChannelHistogram::from_data(&bright_data());
        assert_eq!(analyze_exposure(&hist), ExposureRating::Overexposed);
    }

    #[test]
    fn test_exposure_normal() {
        let hist = ChannelHistogram::from_data(&mid_data());
        assert_eq!(analyze_exposure(&hist), ExposureRating::Normal);
    }
}
