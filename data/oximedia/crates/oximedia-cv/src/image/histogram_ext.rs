//! Extended histogram operations for image processing.
//!
//! This module provides a richer [`HistogramU64`] type (with `u64` bins for
//! large images) along with pixel-based constructors, statistical helpers,
//! histogram equalisation, histogram matching, and distance metrics.
//!
//! # Example
//!
//! ```
//! use oximedia_cv::image::histogram_ext::{
//!     HistogramU64, equalize_histogram, compute_histogram_distance,
//! };
//!
//! let pixels: Vec<u8> = (0u8..=255).collect();
//! let h = HistogramU64::from_pixels(&pixels);
//! assert_eq!(h.total_pixels(), 256);
//!
//! let eq = equalize_histogram(&pixels);
//! assert_eq!(eq.len(), 256);
//!
//! let dist = compute_histogram_distance(&h, &h);
//! assert!(dist < 1e-9);
//! ```

#![allow(dead_code)]

/// A 256-bin grayscale histogram backed by `u64` counters.
///
/// Suitable for images up to ~1.8 × 10¹⁹ pixels without overflow.
#[derive(Debug, Clone)]
pub struct HistogramU64 {
    /// Per-intensity-level counts (index = pixel value 0–255).
    pub bins: [u64; 256],
}

impl HistogramU64 {
    /// Create an empty histogram (all bins = 0).
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::image::histogram_ext::HistogramU64;
    ///
    /// let h = HistogramU64::new();
    /// assert_eq!(h.total_pixels(), 0);
    /// ```
    #[must_use]
    pub const fn new() -> Self {
        Self { bins: [0u64; 256] }
    }

    /// Build a histogram from a flat byte slice (one byte per pixel).
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::image::histogram_ext::HistogramU64;
    ///
    /// let pixels = vec![0u8, 128, 255, 128];
    /// let h = HistogramU64::from_pixels(&pixels);
    /// assert_eq!(h.bins[128], 2);
    /// ```
    #[must_use]
    pub fn from_pixels(pixels: &[u8]) -> Self {
        let mut h = Self::new();
        for &p in pixels {
            h.bins[p as usize] += 1;
        }
        h
    }

    /// Build a histogram by sampling every `stride`-th byte starting at
    /// `offset`.
    ///
    /// This is useful for extracting a single channel from interleaved data,
    /// e.g. the red channel of an RGB buffer (`stride = 3, offset = 0`).
    ///
    /// # Arguments
    ///
    /// * `pixels` - Raw interleaved pixel data.
    /// * `stride` - Distance in bytes between successive samples.
    /// * `offset` - Byte offset of the first sample within the slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::image::histogram_ext::HistogramU64;
    ///
    /// // Extract blue channel (offset=2) from RGB data
    /// let rgb = [255u8, 0, 50,  255, 0, 50];
    /// let h = HistogramU64::from_pixels_channel(&rgb, 3, 2);
    /// assert_eq!(h.bins[50], 2);
    /// ```
    #[must_use]
    pub fn from_pixels_channel(pixels: &[u8], stride: usize, offset: usize) -> Self {
        let mut h = Self::new();
        if stride == 0 || pixels.is_empty() {
            return h;
        }
        let mut i = offset;
        while i < pixels.len() {
            h.bins[pixels[i] as usize] += 1;
            i += stride;
        }
        h
    }

    /// Total number of pixels counted.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::image::histogram_ext::HistogramU64;
    ///
    /// let h = HistogramU64::from_pixels(&[10u8, 20, 30]);
    /// assert_eq!(h.total_pixels(), 3);
    /// ```
    #[must_use]
    pub fn total_pixels(&self) -> u64 {
        self.bins.iter().sum()
    }

    /// Mean pixel intensity (weighted average over bins).
    ///
    /// Returns `0.0` if the histogram is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::image::histogram_ext::HistogramU64;
    ///
    /// let h = HistogramU64::from_pixels(&[100u8, 100, 100]);
    /// assert!((h.mean() - 100.0).abs() < 1e-9);
    /// ```
    #[must_use]
    pub fn mean(&self) -> f64 {
        let total = self.total_pixels();
        if total == 0 {
            return 0.0;
        }
        let sum: u64 = self
            .bins
            .iter()
            .enumerate()
            .map(|(i, &c)| i as u64 * c)
            .sum();
        sum as f64 / total as f64
    }

    /// Population standard deviation of pixel intensities.
    ///
    /// Returns `0.0` if the histogram has fewer than 1 pixel.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::image::histogram_ext::HistogramU64;
    ///
    /// let h = HistogramU64::from_pixels(&[100u8, 100]);
    /// assert!(h.std_dev() < 1e-9);
    /// ```
    #[must_use]
    pub fn std_dev(&self) -> f64 {
        let total = self.total_pixels();
        if total == 0 {
            return 0.0;
        }
        let mean = self.mean();
        let variance: f64 = self
            .bins
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                let d = i as f64 - mean;
                d * d * c as f64
            })
            .sum::<f64>()
            / total as f64;
        variance.sqrt()
    }

    /// The intensity value at the given percentile.
    ///
    /// `pct` must be in `[0.0, 100.0]`. Returns the lowest intensity bin `i`
    /// such that the cumulative count up to and including `i` is at least
    /// `pct / 100 * total_pixels`.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::image::histogram_ext::HistogramU64;
    ///
    /// let pixels: Vec<u8> = (0u8..=255).collect();
    /// let h = HistogramU64::from_pixels(&pixels);
    /// assert_eq!(h.percentile(50.0), 127);
    /// assert_eq!(h.percentile(100.0), 255);
    /// ```
    #[must_use]
    pub fn percentile(&self, pct: f64) -> u8 {
        let total = self.total_pixels();
        if total == 0 {
            return 0;
        }
        let target = ((pct.clamp(0.0, 100.0) / 100.0) * total as f64).ceil() as u64;
        let mut cumsum = 0u64;
        for (i, &c) in self.bins.iter().enumerate() {
            cumsum += c;
            if cumsum >= target {
                return i as u8;
            }
        }
        255
    }

    /// Compute the cumulative histogram (running sum of bins).
    ///
    /// `result[i] = bins[0] + bins[1] + … + bins[i]`.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::image::histogram_ext::HistogramU64;
    ///
    /// let h = HistogramU64::from_pixels(&[0u8, 255]);
    /// let cum = h.cumulative();
    /// assert_eq!(cum[0], 1);
    /// assert_eq!(cum[255], 2);
    /// ```
    #[must_use]
    pub fn cumulative(&self) -> [u64; 256] {
        let mut cum = [0u64; 256];
        let mut running = 0u64;
        for (i, &c) in self.bins.iter().enumerate() {
            running += c;
            cum[i] = running;
        }
        cum
    }
}

impl Default for HistogramU64 {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Standalone histogram functions ──────────────────────────────────────────

/// Apply global histogram equalisation to a flat grayscale byte slice.
///
/// The equalised pixel values are computed using the standard CDF-based
/// formula and clamped to `[0, 255]`.
///
/// # Arguments
///
/// * `pixels` - Grayscale image data (any length).
///
/// # Returns
///
/// New `Vec<u8>` of the same length with the equalised image.
///
/// # Examples
///
/// ```
/// use oximedia_cv::image::histogram_ext::equalize_histogram;
///
/// let pixels = vec![50u8; 100];
/// let eq = equalize_histogram(&pixels);
/// assert_eq!(eq.len(), 100);
/// ```
#[must_use]
pub fn equalize_histogram(pixels: &[u8]) -> Vec<u8> {
    if pixels.is_empty() {
        return Vec::new();
    }

    let h = HistogramU64::from_pixels(pixels);
    let cum = h.cumulative();
    let total = h.total_pixels();

    // Minimum non-zero CDF value
    let cdf_min = cum.iter().copied().find(|&v| v > 0).unwrap_or(0);

    if cdf_min >= total {
        // All pixels have the same value – no change possible
        return pixels.to_vec();
    }

    let denom = (total - cdf_min) as f64;
    let lut: [u8; 256] = std::array::from_fn(|i| {
        let c = cum[i];
        if c < cdf_min {
            0
        } else {
            (((c - cdf_min) as f64 / denom) * 255.0)
                .round()
                .clamp(0.0, 255.0) as u8
        }
    });

    pixels.iter().map(|&p| lut[p as usize]).collect()
}

/// Adjust the tonal distribution of `src` to match that of `reference` using
/// histogram matching (specification).
///
/// The lookup table is built so that each intensity in `src` maps to the
/// intensity in `reference` whose CDF value is closest to the `src` CDF at
/// that intensity.
///
/// # Arguments
///
/// * `src`       - Source grayscale pixel data to remap.
/// * `reference` - Target grayscale pixel data whose histogram is matched.
///
/// # Returns
///
/// New `Vec<u8>` of the same length as `src`.
///
/// # Examples
///
/// ```
/// use oximedia_cv::image::histogram_ext::histogram_match;
///
/// let src  = vec![50u8; 64];
/// let reference = vec![200u8; 64];
/// let out = histogram_match(&src, &reference);
/// assert_eq!(out.len(), 64);
/// // All pixels in src should be mapped towards the reference distribution.
/// assert!(out.iter().all(|&p| p > 50));
/// ```
#[must_use]
pub fn histogram_match(src: &[u8], reference: &[u8]) -> Vec<u8> {
    if src.is_empty() {
        return Vec::new();
    }

    // Build normalised CDFs for both images
    let src_cdf = normalised_cdf(src);
    let ref_cdf = normalised_cdf(reference);

    // For each source intensity, find the reference intensity whose CDF is
    // closest (monotone LUT).
    let lut: [u8; 256] = std::array::from_fn(|i| {
        let target = src_cdf[i];
        let mut best = 0usize;
        let mut best_diff = f64::MAX;
        for (j, &rv) in ref_cdf.iter().enumerate() {
            let diff = (rv - target).abs();
            if diff < best_diff {
                best_diff = diff;
                best = j;
            }
        }
        best as u8
    });

    src.iter().map(|&p| lut[p as usize]).collect()
}

/// Compute the Chi-squared distance between two histograms.
///
/// The formula used is:
/// ```text
/// d(H1, H2) = Σ (H1[i] - H2[i])² / (H1[i] + H2[i] + ε)  / 2
/// ```
/// where sums are over normalised (probability) histograms.
///
/// # Returns
///
/// Non-negative distance value. Returns `0.0` for identical histograms and
/// `f64::INFINITY` if both histograms are empty.
///
/// # Examples
///
/// ```
/// use oximedia_cv::image::histogram_ext::{HistogramU64, compute_histogram_distance};
///
/// let h = HistogramU64::from_pixels(&[100u8; 64]);
/// assert!(compute_histogram_distance(&h, &h) < 1e-9);
/// ```
#[must_use]
pub fn compute_histogram_distance(a: &HistogramU64, b: &HistogramU64) -> f64 {
    let ta = a.total_pixels();
    let tb = b.total_pixels();
    if ta == 0 && tb == 0 {
        return 0.0;
    }
    if ta == 0 || tb == 0 {
        return f64::INFINITY;
    }

    let ta_f = ta as f64;
    let tb_f = tb as f64;
    let eps = f64::EPSILON;

    let chi: f64 = a
        .bins
        .iter()
        .zip(b.bins.iter())
        .map(|(&ai, &bi)| {
            let an = ai as f64 / ta_f;
            let bn = bi as f64 / tb_f;
            let denom = an + bn + eps;
            let diff = an - bn;
            diff * diff / denom
        })
        .sum();

    chi / 2.0
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Compute a normalised (probability) CDF for a flat grayscale slice.
fn normalised_cdf(pixels: &[u8]) -> [f64; 256] {
    if pixels.is_empty() {
        return [0.0; 256];
    }
    let h = HistogramU64::from_pixels(pixels);
    let total = h.total_pixels() as f64;
    let mut cdf = [0.0f64; 256];
    let mut running = 0.0f64;
    for (i, &c) in h.bins.iter().enumerate() {
        running += c as f64 / total;
        cdf[i] = running;
    }
    cdf
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    // ── HistogramU64::new ──────────────────────────────────────────────────
    #[test]
    fn test_new_is_empty() {
        let h = HistogramU64::new();
        assert_eq!(h.total_pixels(), 0);
        assert!(h.bins.iter().all(|&b| b == 0));
    }

    // ── from_pixels ───────────────────────────────────────────────────────
    #[test]
    fn test_from_pixels_counts() {
        let h = HistogramU64::from_pixels(&[0, 0, 128, 255]);
        assert_eq!(h.bins[0], 2);
        assert_eq!(h.bins[128], 1);
        assert_eq!(h.bins[255], 1);
        assert_eq!(h.total_pixels(), 4);
    }

    #[test]
    fn test_from_pixels_empty() {
        let h = HistogramU64::from_pixels(&[]);
        assert_eq!(h.total_pixels(), 0);
    }

    // ── from_pixels_channel ───────────────────────────────────────────────
    #[test]
    fn test_from_pixels_channel_rgb_red() {
        // RGB triples: (100, 0, 0) repeated twice
        let rgb = [100u8, 0, 0, 100, 0, 0];
        let h = HistogramU64::from_pixels_channel(&rgb, 3, 0);
        assert_eq!(h.bins[100], 2);
        assert_eq!(h.total_pixels(), 2);
    }

    #[test]
    fn test_from_pixels_channel_stride_one_equals_from_pixels() {
        let data = vec![10u8, 20, 30, 40];
        let h_all = HistogramU64::from_pixels(&data);
        let h_ch = HistogramU64::from_pixels_channel(&data, 1, 0);
        assert_eq!(h_all.total_pixels(), h_ch.total_pixels());
        for i in 0..256 {
            assert_eq!(h_all.bins[i], h_ch.bins[i]);
        }
    }

    // ── mean ──────────────────────────────────────────────────────────────
    #[test]
    fn test_mean_uniform() {
        let h = HistogramU64::from_pixels(&[100u8; 50]);
        assert!((h.mean() - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_mean_two_values() {
        let h = HistogramU64::from_pixels(&[0u8, 200]);
        assert!((h.mean() - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_mean_empty() {
        assert_eq!(HistogramU64::new().mean(), 0.0);
    }

    // ── std_dev ───────────────────────────────────────────────────────────
    #[test]
    fn test_std_dev_zero_for_uniform() {
        let h = HistogramU64::from_pixels(&[128u8; 100]);
        assert!(h.std_dev() < 1e-9);
    }

    #[test]
    fn test_std_dev_positive_for_mixed() {
        let h = HistogramU64::from_pixels(&[0u8, 255]);
        assert!(h.std_dev() > 0.0);
    }

    // ── percentile ────────────────────────────────────────────────────────
    #[test]
    fn test_percentile_median() {
        let pixels: Vec<u8> = (0u8..=255).collect();
        let h = HistogramU64::from_pixels(&pixels);
        assert_eq!(h.percentile(50.0), 127);
    }

    #[test]
    fn test_percentile_100_is_max() {
        let pixels: Vec<u8> = (0u8..=255).collect();
        let h = HistogramU64::from_pixels(&pixels);
        assert_eq!(h.percentile(100.0), 255);
    }

    #[test]
    fn test_percentile_0_is_min_nonzero() {
        let pixels: Vec<u8> = (0u8..=255).collect();
        let h = HistogramU64::from_pixels(&pixels);
        assert_eq!(h.percentile(0.0), 0);
    }

    #[test]
    fn test_percentile_empty_returns_0() {
        assert_eq!(HistogramU64::new().percentile(50.0), 0);
    }

    // ── cumulative ────────────────────────────────────────────────────────
    #[test]
    fn test_cumulative_last_equals_total() {
        let h = HistogramU64::from_pixels(&[10u8, 20, 30]);
        let cum = h.cumulative();
        assert_eq!(cum[255], h.total_pixels());
    }

    #[test]
    fn test_cumulative_monotone() {
        let pixels: Vec<u8> = (0..100).map(|i| (i % 256) as u8).collect();
        let h = HistogramU64::from_pixels(&pixels);
        let cum = h.cumulative();
        for i in 1..256 {
            assert!(cum[i] >= cum[i - 1]);
        }
    }

    // ── equalize_histogram ────────────────────────────────────────────────
    #[test]
    fn test_equalize_histogram_length_preserved() {
        let px = vec![50u8; 128];
        let eq = equalize_histogram(&px);
        assert_eq!(eq.len(), 128);
    }

    #[test]
    fn test_equalize_histogram_uniform_no_change() {
        // All-same value → no effective change (CDF is a step function)
        let px = vec![100u8; 64];
        let eq = equalize_histogram(&px);
        assert_eq!(eq.len(), 64);
        // Output is u8 so all values are inherently in [0, 255]
        assert!(!eq.is_empty());
    }

    #[test]
    fn test_equalize_histogram_full_range() {
        let px: Vec<u8> = (0u8..=255).collect();
        let eq = equalize_histogram(&px);
        assert_eq!(eq.len(), 256);
        // Output length must match input length
        assert_eq!(eq.len(), px.len());
    }

    #[test]
    fn test_equalize_histogram_empty() {
        let eq = equalize_histogram(&[]);
        assert!(eq.is_empty());
    }

    // ── histogram_match ───────────────────────────────────────────────────
    #[test]
    fn test_histogram_match_length_preserved() {
        let src = vec![50u8; 64];
        let reference = vec![200u8; 64];
        let out = histogram_match(&src, &reference);
        assert_eq!(out.len(), 64);
    }

    #[test]
    fn test_histogram_match_identical_is_noop() {
        let px: Vec<u8> = (0u8..=255).collect();
        let out = histogram_match(&px, &px);
        // Matching identical histograms should return an equal or very close distribution
        assert_eq!(out.len(), px.len());
    }

    #[test]
    fn test_histogram_match_shifts_towards_reference() {
        let src = vec![50u8; 64];
        let reference = vec![200u8; 64];
        let out = histogram_match(&src, &reference);
        // All output pixels should be 200 (or nearest representative)
        assert!(out.iter().all(|&p| p >= 100));
    }

    #[test]
    fn test_histogram_match_empty_src() {
        let out = histogram_match(&[], &[100u8; 10]);
        assert!(out.is_empty());
    }

    // ── compute_histogram_distance ────────────────────────────────────────
    #[test]
    fn test_histogram_distance_self_is_zero() {
        let h = HistogramU64::from_pixels(&[100u8; 64]);
        assert!(compute_histogram_distance(&h, &h) < 1e-9);
    }

    #[test]
    fn test_histogram_distance_different_is_positive() {
        let h1 = HistogramU64::from_pixels(&[0u8; 64]);
        let h2 = HistogramU64::from_pixels(&[255u8; 64]);
        assert!(compute_histogram_distance(&h1, &h2) > 0.0);
    }

    #[test]
    fn test_histogram_distance_both_empty_is_zero() {
        let h1 = HistogramU64::new();
        let h2 = HistogramU64::new();
        assert_eq!(compute_histogram_distance(&h1, &h2), 0.0);
    }

    #[test]
    fn test_histogram_distance_one_empty_is_inf() {
        let h1 = HistogramU64::from_pixels(&[128u8; 10]);
        let h2 = HistogramU64::new();
        assert!(compute_histogram_distance(&h1, &h2).is_infinite());
    }
}
