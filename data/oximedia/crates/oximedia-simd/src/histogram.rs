//! Histogram computation for pixel data analysis.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

/// A 1D histogram over a range of values.
#[derive(Debug, Clone)]
pub struct Histogram {
    /// Bin counts
    pub bins: Vec<u32>,
    /// Minimum value of the range (inclusive)
    pub min_val: f32,
    /// Maximum value of the range (inclusive)
    pub max_val: f32,
}

impl Histogram {
    /// Create a new histogram with `num_bins` bins covering `[min, max]`.
    ///
    /// Panics if `num_bins` is zero.
    #[must_use]
    pub fn new(num_bins: usize, min: f32, max: f32) -> Self {
        assert!(num_bins > 0, "num_bins must be > 0");
        Self {
            bins: vec![0u32; num_bins],
            min_val: min,
            max_val: max,
        }
    }

    /// Add a single value to the histogram.
    ///
    /// Values outside the `[min_val, max_val]` range are clamped to the
    /// nearest edge bin.
    pub fn add(&mut self, value: f32) {
        let bin = self.get_bin(value);
        self.bins[bin] = self.bins[bin].saturating_add(1);
    }

    /// Compute the bin index for a given value.
    ///
    /// Values are clamped to fall within `[0, num_bins - 1]`.
    #[must_use]
    pub fn get_bin(&self, value: f32) -> usize {
        let n = self.bins.len();
        if n == 1 {
            return 0;
        }
        let range = self.max_val - self.min_val;
        if range < f32::EPSILON {
            return 0;
        }
        let t = (value - self.min_val) / range;
        let idx = (t * n as f32).floor() as i64;
        idx.clamp(0, n as i64 - 1) as usize
    }

    /// Return the total number of samples added.
    #[must_use]
    pub fn total_count(&self) -> u32 {
        self.bins.iter().sum()
    }

    /// Return the index of the bin with the highest count.
    ///
    /// Returns 0 for an empty histogram.
    #[must_use]
    pub fn peak_bin(&self) -> usize {
        self.bins
            .iter()
            .enumerate()
            .max_by_key(|&(_, &count)| count)
            .map_or(0, |(i, _)| i)
    }

    /// Return the cumulative histogram (running sum of bin counts).
    #[must_use]
    pub fn cumulative(&self) -> Vec<u32> {
        let mut cum = Vec::with_capacity(self.bins.len());
        let mut running = 0u32;
        for &count in &self.bins {
            running = running.saturating_add(count);
            cum.push(running);
        }
        cum
    }
}

/// Per-channel RGB histogram with 256 bins per channel over [0, 255].
#[derive(Debug, Clone)]
pub struct RgbHistogram {
    /// Red channel histogram
    pub r: Histogram,
    /// Green channel histogram
    pub g: Histogram,
    /// Blue channel histogram
    pub b: Histogram,
}

impl RgbHistogram {
    /// Create a new `RgbHistogram` with 256 bins per channel, range [0.0, 255.0].
    #[must_use]
    pub fn new() -> Self {
        Self {
            r: Histogram::new(256, 0.0, 255.0),
            g: Histogram::new(256, 0.0, 255.0),
            b: Histogram::new(256, 0.0, 255.0),
        }
    }

    /// Add a single pixel's RGB values to the histogram.
    pub fn add_pixel(&mut self, r: u8, g: u8, b: u8) {
        self.r.add(f32::from(r));
        self.g.add(f32::from(g));
        self.b.add(f32::from(b));
    }

    /// Return the total number of pixels added (uses R channel count).
    #[must_use]
    pub fn total_pixels(&self) -> u32 {
        self.r.total_count()
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
    fn test_histogram_new() {
        let h = Histogram::new(10, 0.0, 1.0);
        assert_eq!(h.bins.len(), 10);
        assert!((h.min_val - 0.0).abs() < 1e-6);
        assert!((h.max_val - 1.0).abs() < 1e-6);
        assert!(h.bins.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_histogram_add_basic() {
        let mut h = Histogram::new(10, 0.0, 10.0);
        h.add(0.5); // bin 0
        h.add(5.0); // bin 5
        h.add(9.9); // bin 9
        assert_eq!(h.total_count(), 3);
    }

    #[test]
    fn test_histogram_get_bin_min() {
        let h = Histogram::new(10, 0.0, 10.0);
        assert_eq!(h.get_bin(0.0), 0);
    }

    #[test]
    fn test_histogram_get_bin_max() {
        let h = Histogram::new(10, 0.0, 10.0);
        assert_eq!(h.get_bin(10.0), 9);
    }

    #[test]
    fn test_histogram_get_bin_mid() {
        let h = Histogram::new(10, 0.0, 10.0);
        assert_eq!(h.get_bin(5.0), 5);
    }

    #[test]
    fn test_histogram_get_bin_clamp_below() {
        let h = Histogram::new(10, 0.0, 10.0);
        assert_eq!(h.get_bin(-5.0), 0);
    }

    #[test]
    fn test_histogram_get_bin_clamp_above() {
        let h = Histogram::new(10, 0.0, 10.0);
        assert_eq!(h.get_bin(20.0), 9);
    }

    #[test]
    fn test_histogram_total_count_empty() {
        let h = Histogram::new(8, 0.0, 1.0);
        assert_eq!(h.total_count(), 0);
    }

    #[test]
    fn test_histogram_total_count_after_adds() {
        let mut h = Histogram::new(4, 0.0, 4.0);
        for i in 0..10 {
            h.add(i as f32 % 4.0);
        }
        assert_eq!(h.total_count(), 10);
    }

    #[test]
    fn test_histogram_peak_bin() {
        let mut h = Histogram::new(4, 0.0, 4.0);
        h.add(1.5); // bin 1
        h.add(1.5); // bin 1
        h.add(1.5); // bin 1
        h.add(3.0); // bin 3
        assert_eq!(h.peak_bin(), 1);
    }

    #[test]
    fn test_histogram_cumulative() {
        let mut h = Histogram::new(4, 0.0, 4.0);
        h.add(0.5); // bin 0
        h.add(1.5); // bin 1
        h.add(1.5); // bin 1
        h.add(3.5); // bin 3
        let cum = h.cumulative();
        assert_eq!(cum, vec![1, 3, 3, 4]);
    }

    #[test]
    fn test_histogram_cumulative_empty() {
        let h = Histogram::new(3, 0.0, 1.0);
        let cum = h.cumulative();
        assert_eq!(cum, vec![0, 0, 0]);
    }

    #[test]
    fn test_rgb_histogram_new() {
        let h = RgbHistogram::new();
        assert_eq!(h.r.bins.len(), 256);
        assert_eq!(h.g.bins.len(), 256);
        assert_eq!(h.b.bins.len(), 256);
    }

    #[test]
    fn test_rgb_histogram_add_pixel() {
        let mut h = RgbHistogram::new();
        h.add_pixel(255, 128, 0);
        assert_eq!(h.total_pixels(), 1);
    }

    #[test]
    fn test_rgb_histogram_total_pixels() {
        let mut h = RgbHistogram::new();
        for i in 0u8..10 {
            h.add_pixel(i * 25, i * 10, 0);
        }
        assert_eq!(h.total_pixels(), 10);
    }

    #[test]
    fn test_rgb_histogram_default() {
        let h = RgbHistogram::default();
        assert_eq!(h.total_pixels(), 0);
    }
}
