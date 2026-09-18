#![allow(dead_code)]
//! Histogram computation, equalization, and matching for image data.
//!
//! This module provides tools for analyzing and manipulating the tonal distribution
//! of image pixel data through histogram operations commonly used in professional
//! image processing workflows.

use std::fmt;

/// Number of bins in a standard 8-bit histogram.
const HIST_BINS: usize = 256;

/// A histogram representing the distribution of pixel intensities.
#[derive(Clone)]
pub struct Histogram {
    /// Bin counts for each intensity level (0..255).
    pub bins: [u64; HIST_BINS],
    /// Total number of samples accumulated.
    pub total_samples: u64,
}

impl fmt::Debug for Histogram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Histogram")
            .field("total_samples", &self.total_samples)
            .field(
                "non_zero_bins",
                &self.bins.iter().filter(|&&b| b > 0).count(),
            )
            .finish()
    }
}

impl Histogram {
    /// Creates a new empty histogram with all bins set to zero.
    pub fn new() -> Self {
        Self {
            bins: [0u64; HIST_BINS],
            total_samples: 0,
        }
    }

    /// Accumulates a single pixel value into the histogram.
    pub fn accumulate(&mut self, value: u8) {
        self.bins[value as usize] += 1;
        self.total_samples += 1;
    }

    /// Accumulates a slice of pixel values into the histogram.
    pub fn accumulate_slice(&mut self, values: &[u8]) {
        for &v in values {
            self.bins[v as usize] += 1;
        }
        self.total_samples += values.len() as u64;
    }

    /// Returns the bin with the highest count (mode).
    #[allow(clippy::cast_precision_loss)]
    pub fn mode(&self) -> u8 {
        let mut max_idx = 0usize;
        let mut max_val = 0u64;
        for (i, &count) in self.bins.iter().enumerate() {
            if count > max_val {
                max_val = count;
                max_idx = i;
            }
        }
        max_idx as u8
    }

    /// Returns the mean intensity value.
    #[allow(clippy::cast_precision_loss)]
    pub fn mean(&self) -> f64 {
        if self.total_samples == 0 {
            return 0.0;
        }
        let sum: f64 = self
            .bins
            .iter()
            .enumerate()
            .map(|(i, &count)| i as f64 * count as f64)
            .sum();
        sum / self.total_samples as f64
    }

    /// Returns the standard deviation of pixel intensities.
    #[allow(clippy::cast_precision_loss)]
    pub fn std_dev(&self) -> f64 {
        if self.total_samples == 0 {
            return 0.0;
        }
        let mean = self.mean();
        let variance: f64 = self
            .bins
            .iter()
            .enumerate()
            .map(|(i, &count)| {
                let diff = i as f64 - mean;
                diff * diff * count as f64
            })
            .sum::<f64>()
            / self.total_samples as f64;
        variance.sqrt()
    }

    /// Returns the median intensity value.
    #[allow(clippy::cast_precision_loss)]
    pub fn median(&self) -> u8 {
        if self.total_samples == 0 {
            return 0;
        }
        let half = self.total_samples / 2;
        let mut cumulative = 0u64;
        for (i, &count) in self.bins.iter().enumerate() {
            cumulative += count;
            if cumulative > half {
                return i as u8;
            }
        }
        255
    }

    /// Returns the normalized cumulative distribution function (CDF).
    #[allow(clippy::cast_precision_loss)]
    pub fn cdf(&self) -> [f64; HIST_BINS] {
        let mut cdf = [0.0f64; HIST_BINS];
        if self.total_samples == 0 {
            return cdf;
        }
        let mut cumulative = 0u64;
        for (i, &count) in self.bins.iter().enumerate() {
            cumulative += count;
            cdf[i] = cumulative as f64 / self.total_samples as f64;
        }
        cdf
    }

    /// Returns the percentile value (0.0..=1.0).
    #[allow(clippy::cast_precision_loss)]
    pub fn percentile(&self, p: f64) -> u8 {
        let p = p.clamp(0.0, 1.0);
        if self.total_samples == 0 {
            return 0;
        }
        let threshold = (p * self.total_samples as f64) as u64;
        let mut cumulative = 0u64;
        for (i, &count) in self.bins.iter().enumerate() {
            cumulative += count;
            if cumulative >= threshold {
                return i as u8;
            }
        }
        255
    }

    /// Returns the dynamic range as (min_nonzero_bin, max_nonzero_bin).
    pub fn dynamic_range(&self) -> (u8, u8) {
        let min = self.bins.iter().position(|&c| c > 0).unwrap_or(0);
        let max = self.bins.iter().rposition(|&c| c > 0).unwrap_or(0);
        (min as u8, max as u8)
    }
}

impl Default for Histogram {
    fn default() -> Self {
        Self::new()
    }
}

/// Builds a histogram equalization lookup table from a source histogram.
///
/// This maps pixel intensities to produce a more uniform histogram distribution,
/// improving contrast in images with narrow tonal ranges.
#[allow(clippy::cast_precision_loss)]
pub fn equalization_lut(hist: &Histogram) -> [u8; HIST_BINS] {
    let mut lut = [0u8; HIST_BINS];
    if hist.total_samples == 0 {
        for (i, entry) in lut.iter_mut().enumerate() {
            *entry = i as u8;
        }
        return lut;
    }
    let cdf = hist.cdf();
    // Find the minimum non-zero CDF value
    let cdf_min = cdf.iter().copied().find(|&v| v > 0.0).unwrap_or(0.0);
    let scale = 255.0 / (1.0 - cdf_min).max(1e-10);
    for (i, entry) in lut.iter_mut().enumerate() {
        let mapped = ((cdf[i] - cdf_min) * scale).round().clamp(0.0, 255.0);
        *entry = mapped as u8;
    }
    lut
}

/// Applies a lookup table to transform pixel values.
pub fn apply_lut(pixels: &mut [u8], lut: &[u8; HIST_BINS]) {
    for px in pixels.iter_mut() {
        *px = lut[*px as usize];
    }
}

/// Builds a histogram matching/specification lookup table.
///
/// This creates a mapping that transforms the `source` histogram distribution
/// to approximate the `target` histogram distribution.
#[allow(clippy::cast_precision_loss)]
pub fn matching_lut(source: &Histogram, target: &Histogram) -> [u8; HIST_BINS] {
    let mut lut = [0u8; HIST_BINS];
    let src_cdf = source.cdf();
    let tgt_cdf = target.cdf();

    for (i, entry) in lut.iter_mut().enumerate() {
        let src_val = src_cdf[i];
        // Find closest CDF value in target
        let mut best_j = 0usize;
        let mut best_diff = f64::MAX;
        for (j, &tgt_val) in tgt_cdf.iter().enumerate() {
            let diff = (src_val - tgt_val).abs();
            if diff < best_diff {
                best_diff = diff;
                best_j = j;
            }
        }
        *entry = best_j as u8;
    }
    lut
}

/// Computes a histogram from floating-point pixel data in [0.0, 1.0].
///
/// Values are quantized to 256 bins. Values outside [0.0, 1.0] are clamped.
#[allow(clippy::cast_precision_loss)]
pub fn histogram_from_f32(pixels: &[f32]) -> Histogram {
    let mut hist = Histogram::new();
    for &v in pixels {
        let clamped = v.clamp(0.0, 1.0);
        let bin = (clamped * 255.0).round() as u8;
        hist.accumulate(bin);
    }
    hist
}

/// Stretches the histogram to fill the full [0, 255] range (contrast stretching).
///
/// Maps the current `[min, max]` range linearly to `[0, 255]`.
#[allow(clippy::cast_precision_loss)]
pub fn contrast_stretch_lut(hist: &Histogram) -> [u8; HIST_BINS] {
    let mut lut = [0u8; HIST_BINS];
    let (lo, hi) = hist.dynamic_range();
    if lo >= hi {
        for (i, entry) in lut.iter_mut().enumerate() {
            *entry = i as u8;
        }
        return lut;
    }
    let range = f64::from(hi) - f64::from(lo);
    for (i, entry) in lut.iter_mut().enumerate() {
        let val = if (i as u8) <= lo {
            0.0
        } else if (i as u8) >= hi {
            255.0
        } else {
            (f64::from(i as u8 - lo) / range * 255.0).round()
        };
        *entry = val.clamp(0.0, 255.0) as u8;
    }
    lut
}

/// Multi-channel histogram for RGB images.
#[derive(Clone, Debug)]
pub struct RgbHistogram {
    /// Red channel histogram.
    pub red: Histogram,
    /// Green channel histogram.
    pub green: Histogram,
    /// Blue channel histogram.
    pub blue: Histogram,
}

impl RgbHistogram {
    /// Creates a new empty RGB histogram.
    pub fn new() -> Self {
        Self {
            red: Histogram::new(),
            green: Histogram::new(),
            blue: Histogram::new(),
        }
    }

    /// Accumulates an RGB triplet.
    pub fn accumulate_rgb(&mut self, r: u8, g: u8, b: u8) {
        self.red.accumulate(r);
        self.green.accumulate(g);
        self.blue.accumulate(b);
    }

    /// Builds the histogram from interleaved RGB data (3 bytes per pixel).
    pub fn from_rgb_data(data: &[u8]) -> Self {
        let mut hist = Self::new();
        for chunk in data.chunks_exact(3) {
            hist.accumulate_rgb(chunk[0], chunk[1], chunk[2]);
        }
        hist
    }

    /// Returns per-channel equalization LUTs.
    pub fn equalization_luts(&self) -> ([u8; HIST_BINS], [u8; HIST_BINS], [u8; HIST_BINS]) {
        (
            equalization_lut(&self.red),
            equalization_lut(&self.green),
            equalization_lut(&self.blue),
        )
    }
}

impl Default for RgbHistogram {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_histogram_new_is_empty() {
        let h = Histogram::new();
        assert_eq!(h.total_samples, 0);
        assert!(h.bins.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_histogram_accumulate() {
        let mut h = Histogram::new();
        h.accumulate(100);
        h.accumulate(100);
        h.accumulate(200);
        assert_eq!(h.total_samples, 3);
        assert_eq!(h.bins[100], 2);
        assert_eq!(h.bins[200], 1);
    }

    #[test]
    fn test_histogram_accumulate_slice() {
        let mut h = Histogram::new();
        h.accumulate_slice(&[10, 20, 10, 30]);
        assert_eq!(h.total_samples, 4);
        assert_eq!(h.bins[10], 2);
        assert_eq!(h.bins[20], 1);
        assert_eq!(h.bins[30], 1);
    }

    #[test]
    fn test_histogram_mode() {
        let mut h = Histogram::new();
        h.accumulate_slice(&[50, 50, 50, 100, 100, 200]);
        assert_eq!(h.mode(), 50);
    }

    #[test]
    fn test_histogram_mean() {
        let mut h = Histogram::new();
        h.accumulate_slice(&[0, 100, 200]);
        let mean = h.mean();
        assert!((mean - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_histogram_std_dev() {
        let mut h = Histogram::new();
        // All same value => std_dev = 0
        h.accumulate_slice(&[128, 128, 128]);
        assert!((h.std_dev()).abs() < 0.01);
    }

    #[test]
    fn test_histogram_median() {
        let mut h = Histogram::new();
        h.accumulate_slice(&[10, 20, 30, 40, 50]);
        assert_eq!(h.median(), 30);
    }

    #[test]
    fn test_histogram_cdf() {
        let mut h = Histogram::new();
        h.accumulate_slice(&[0, 0, 128, 255]);
        let cdf = h.cdf();
        assert!((cdf[0] - 0.5).abs() < 0.01);
        assert!((cdf[128] - 0.75).abs() < 0.01);
        assert!((cdf[255] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_histogram_percentile() {
        let mut h = Histogram::new();
        for i in 0..=255u8 {
            h.accumulate(i);
        }
        let p50 = h.percentile(0.5);
        // Should be around 127-128
        assert!(p50 >= 126 && p50 <= 129);
    }

    #[test]
    fn test_histogram_dynamic_range() {
        let mut h = Histogram::new();
        h.accumulate_slice(&[50, 60, 70, 200]);
        let (lo, hi) = h.dynamic_range();
        assert_eq!(lo, 50);
        assert_eq!(hi, 200);
    }

    #[test]
    fn test_equalization_lut() {
        let mut h = Histogram::new();
        h.accumulate_slice(&[100, 100, 100, 200, 200]);
        let lut = equalization_lut(&h);
        // LUT should map values; key property: monotonically non-decreasing
        for i in 1..HIST_BINS {
            assert!(lut[i] >= lut[i - 1]);
        }
    }

    #[test]
    fn test_apply_lut() {
        let lut: [u8; 256] = std::array::from_fn(|i| 255 - i as u8);
        let mut pixels = vec![0u8, 128, 255];
        apply_lut(&mut pixels, &lut);
        assert_eq!(pixels, vec![255, 127, 0]);
    }

    #[test]
    fn test_matching_lut_identity() {
        let mut h = Histogram::new();
        for i in 0..=255u8 {
            h.accumulate(i);
        }
        // Matching identical histograms should produce near-identity LUT
        let lut = matching_lut(&h, &h);
        for (i, &val) in lut.iter().enumerate() {
            let diff = (val as i32 - i as i32).unsigned_abs();
            assert!(diff <= 1, "bin {i}: expected ~{i}, got {val}");
        }
    }

    #[test]
    fn test_histogram_from_f32() {
        let pixels = vec![0.0f32, 0.5, 1.0];
        let h = histogram_from_f32(&pixels);
        assert_eq!(h.total_samples, 3);
        assert_eq!(h.bins[0], 1);
        assert_eq!(h.bins[128], 1);
        assert_eq!(h.bins[255], 1);
    }

    #[test]
    fn test_contrast_stretch_lut() {
        let mut h = Histogram::new();
        h.accumulate_slice(&[50, 100, 150]);
        let lut = contrast_stretch_lut(&h);
        assert_eq!(lut[50], 0);
        assert_eq!(lut[150], 255);
        // Mid-point should be near 128
        assert!(lut[100] > 100 && lut[100] < 160);
    }

    #[test]
    fn test_rgb_histogram_from_data() {
        let data = vec![
            255, 0, 0, // red pixel
            0, 255, 0, // green pixel
            0, 0, 255, // blue pixel
        ];
        let h = RgbHistogram::from_rgb_data(&data);
        assert_eq!(h.red.bins[255], 1);
        assert_eq!(h.green.bins[255], 1);
        assert_eq!(h.blue.bins[255], 1);
        assert_eq!(h.red.bins[0], 2);
    }

    #[test]
    fn test_histogram_empty_stats() {
        let h = Histogram::new();
        assert_eq!(h.mean(), 0.0);
        assert_eq!(h.std_dev(), 0.0);
        assert_eq!(h.median(), 0);
        assert_eq!(h.mode(), 0);
    }
}
