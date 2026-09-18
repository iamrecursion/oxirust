#![allow(dead_code)]
//! Histogram back-projection for object localization in images.
//!
//! This module implements histogram back-projection, a technique used to
//! find regions of an image that match a reference histogram model. It is
//! commonly used for object tracking and skin-color detection.
//!
//! # Algorithms
//!
//! - **Ratio back-projection**: Computes the ratio of model histogram to
//!   scene histogram for each pixel.
//! - **Bayesian back-projection**: Uses Bayes' theorem to compute the
//!   probability that a pixel belongs to the target object.
//! - **Mean-shift iteration**: Iteratively finds the mode of the
//!   back-projection map to localize the object center.

use std::collections::HashMap;

/// Number of bins used for default histogram quantization.
const DEFAULT_BINS: usize = 64;

/// A 1D histogram with a configurable number of bins.
#[derive(Debug, Clone)]
pub struct Histogram {
    /// Bin counts for the histogram.
    pub bins: Vec<f64>,
    /// Number of bins in the histogram.
    pub num_bins: usize,
    /// Minimum value of the histogram range.
    pub range_min: f64,
    /// Maximum value of the histogram range.
    pub range_max: f64,
}

impl Histogram {
    /// Creates a new empty histogram with the given number of bins.
    pub fn new(num_bins: usize, range_min: f64, range_max: f64) -> Self {
        Self {
            bins: vec![0.0; num_bins],
            num_bins,
            range_min,
            range_max,
        }
    }

    /// Creates a histogram with default parameters (64 bins, 0..256).
    pub fn default_256() -> Self {
        Self::new(DEFAULT_BINS, 0.0, 256.0)
    }

    /// Returns the bin index for a given value.
    #[allow(clippy::cast_precision_loss)]
    pub fn bin_index(&self, value: f64) -> Option<usize> {
        if value < self.range_min || value >= self.range_max {
            return None;
        }
        let span = self.range_max - self.range_min;
        if span <= 0.0 {
            return None;
        }
        let idx = ((value - self.range_min) / span * self.num_bins as f64) as usize;
        Some(idx.min(self.num_bins - 1))
    }

    /// Adds a value to the histogram.
    pub fn add(&mut self, value: f64) {
        if let Some(idx) = self.bin_index(value) {
            self.bins[idx] += 1.0;
        }
    }

    /// Normalizes the histogram so all bins sum to 1.0.
    pub fn normalize(&mut self) {
        let total: f64 = self.bins.iter().sum();
        if total > 0.0 {
            for bin in &mut self.bins {
                *bin /= total;
            }
        }
    }

    /// Returns the total count of all bins.
    pub fn total(&self) -> f64 {
        self.bins.iter().sum()
    }

    /// Returns the mean value of the distribution.
    #[allow(clippy::cast_precision_loss)]
    pub fn mean(&self) -> f64 {
        let total = self.total();
        if total == 0.0 {
            return 0.0;
        }
        let span = self.range_max - self.range_min;
        let bin_width = span / self.num_bins as f64;
        let mut sum = 0.0;
        for (i, &count) in self.bins.iter().enumerate() {
            let center = self.range_min + (i as f64 + 0.5) * bin_width;
            sum += center * count;
        }
        sum / total
    }
}

/// A 2D histogram for two-channel data (e.g., Hue-Saturation).
#[derive(Debug, Clone)]
pub struct Histogram2D {
    /// Flattened bin data in row-major order.
    pub bins: Vec<f64>,
    /// Number of bins along the first axis.
    pub bins_x: usize,
    /// Number of bins along the second axis.
    pub bins_y: usize,
    /// Range for the first axis.
    pub range_x: (f64, f64),
    /// Range for the second axis.
    pub range_y: (f64, f64),
}

impl Histogram2D {
    /// Creates a new 2D histogram with specified bins and ranges.
    pub fn new(bins_x: usize, bins_y: usize, range_x: (f64, f64), range_y: (f64, f64)) -> Self {
        Self {
            bins: vec![0.0; bins_x * bins_y],
            bins_x,
            bins_y,
            range_x,
            range_y,
        }
    }

    /// Adds a sample at coordinates (x, y).
    #[allow(clippy::cast_precision_loss)]
    pub fn add(&mut self, x: f64, y: f64) {
        if let Some((ix, iy)) = self.bin_indices(x, y) {
            self.bins[iy * self.bins_x + ix] += 1.0;
        }
    }

    /// Returns the bin indices for given (x, y) values.
    #[allow(clippy::cast_precision_loss)]
    pub fn bin_indices(&self, x: f64, y: f64) -> Option<(usize, usize)> {
        let span_x = self.range_x.1 - self.range_x.0;
        let span_y = self.range_y.1 - self.range_y.0;
        if span_x <= 0.0 || span_y <= 0.0 {
            return None;
        }
        if x < self.range_x.0 || x >= self.range_x.1 {
            return None;
        }
        if y < self.range_y.0 || y >= self.range_y.1 {
            return None;
        }
        let ix = ((x - self.range_x.0) / span_x * self.bins_x as f64) as usize;
        let iy = ((y - self.range_y.0) / span_y * self.bins_y as f64) as usize;
        Some((ix.min(self.bins_x - 1), iy.min(self.bins_y - 1)))
    }

    /// Normalizes the histogram so all bins sum to 1.0.
    pub fn normalize(&mut self) {
        let total: f64 = self.bins.iter().sum();
        if total > 0.0 {
            for bin in &mut self.bins {
                *bin /= total;
            }
        }
    }

    /// Gets the value at the given bin indices.
    pub fn get(&self, ix: usize, iy: usize) -> f64 {
        if ix < self.bins_x && iy < self.bins_y {
            self.bins[iy * self.bins_x + ix]
        } else {
            0.0
        }
    }
}

/// Configuration for back-projection computation.
#[derive(Debug, Clone)]
pub struct BackProjectConfig {
    /// Number of bins for the histogram.
    pub num_bins: usize,
    /// Whether to normalize the output to [0, 1].
    pub normalize_output: bool,
    /// Threshold for clipping low probabilities.
    pub clip_threshold: f64,
    /// Smoothing kernel size (0 = no smoothing).
    pub smooth_kernel: usize,
}

impl Default for BackProjectConfig {
    fn default() -> Self {
        Self {
            num_bins: DEFAULT_BINS,
            normalize_output: true,
            clip_threshold: 0.0,
            smooth_kernel: 0,
        }
    }
}

/// Computes the ratio back-projection of a model histogram against a scene histogram.
///
/// For each bin, the ratio is `model[i] / scene[i]`. The resulting map
/// gives higher values where the scene matches the model.
pub fn ratio_backproject(model: &Histogram, scene: &Histogram) -> Vec<f64> {
    assert_eq!(
        model.num_bins, scene.num_bins,
        "Histograms must have the same number of bins"
    );
    let mut ratio = vec![0.0; model.num_bins];
    for i in 0..model.num_bins {
        ratio[i] = if scene.bins[i] > 0.0 {
            model.bins[i] / scene.bins[i]
        } else {
            0.0
        };
    }
    ratio
}

/// Applies back-projection to a 1D data array using the ratio map.
///
/// Each value in `data` is looked up in the histogram to find its bin,
/// and the corresponding ratio value is written to the output.
#[allow(clippy::cast_precision_loss)]
pub fn apply_backproject(data: &[f64], ratio: &[f64], range_min: f64, range_max: f64) -> Vec<f64> {
    let num_bins = ratio.len();
    let span = range_max - range_min;
    data.iter()
        .map(|&v| {
            if v < range_min || v >= range_max || span <= 0.0 {
                0.0
            } else {
                let idx = ((v - range_min) / span * num_bins as f64) as usize;
                let idx = idx.min(num_bins - 1);
                ratio[idx]
            }
        })
        .collect()
}

/// Computes Bayesian back-projection probability.
///
/// Uses Bayes' theorem: `P(object | pixel) = P(pixel | object) * P(object) / P(pixel)`.
#[allow(clippy::cast_precision_loss)]
pub fn bayesian_backproject(model: &Histogram, scene: &Histogram, prior: f64) -> Vec<f64> {
    assert_eq!(model.num_bins, scene.num_bins);
    let mut result = vec![0.0; model.num_bins];
    for i in 0..model.num_bins {
        let p_pixel_given_object = model.bins[i];
        let p_pixel = scene.bins[i];
        result[i] = if p_pixel > 0.0 {
            (p_pixel_given_object * prior / p_pixel).min(1.0)
        } else {
            0.0
        };
    }
    result
}

/// Represents a 2D point for mean-shift tracking.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point2D {
    /// X coordinate.
    pub x: f64,
    /// Y coordinate.
    pub y: f64,
}

impl Point2D {
    /// Creates a new 2D point.
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Computes the Euclidean distance to another point.
    pub fn distance(&self, other: &Point2D) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
}

/// Mean-shift iteration on a 2D probability map.
///
/// Given a probability map (rows x cols) and an initial center, this
/// iterates mean-shift to find the mode of the distribution.
#[allow(clippy::cast_precision_loss)]
pub fn mean_shift_iterate(
    prob_map: &[f64],
    width: usize,
    height: usize,
    initial: Point2D,
    window_radius: f64,
    max_iterations: usize,
    convergence_threshold: f64,
) -> Point2D {
    let mut center = initial;

    for _iter in 0..max_iterations {
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_w = 0.0;

        for row in 0..height {
            for col in 0..width {
                let px = col as f64 + 0.5;
                let py = row as f64 + 0.5;
                let dx = px - center.x;
                let dy = py - center.y;
                let dist_sq = dx * dx + dy * dy;
                let r_sq = window_radius * window_radius;
                if dist_sq <= r_sq {
                    let weight = prob_map[row * width + col];
                    sum_x += px * weight;
                    sum_y += py * weight;
                    sum_w += weight;
                }
            }
        }

        if sum_w <= 0.0 {
            break;
        }

        let new_center = Point2D::new(sum_x / sum_w, sum_y / sum_w);
        let shift = center.distance(&new_center);
        center = new_center;

        if shift < convergence_threshold {
            break;
        }
    }

    center
}

/// Histogram intersection metric (Swain-Ballard).
///
/// Returns a value between 0.0 (no match) and 1.0 (perfect match)
/// when both histograms are normalized.
pub fn histogram_intersection(a: &Histogram, b: &Histogram) -> f64 {
    assert_eq!(a.num_bins, b.num_bins);
    let mut intersection = 0.0;
    for i in 0..a.num_bins {
        intersection += a.bins[i].min(b.bins[i]);
    }
    intersection
}

/// Bhattacharyya distance between two normalized histograms.
///
/// Returns a value between 0.0 (identical) and 1.0 (completely different).
pub fn bhattacharyya_distance(a: &Histogram, b: &Histogram) -> f64 {
    assert_eq!(a.num_bins, b.num_bins);
    let mut bc = 0.0;
    for i in 0..a.num_bins {
        bc += (a.bins[i] * b.bins[i]).sqrt();
    }
    let dist_sq = 1.0 - bc;
    if dist_sq < 0.0 {
        0.0
    } else {
        dist_sq.sqrt()
    }
}

/// Chi-squared distance between two histograms.
///
/// `chi2 = sum((a[i] - b[i])^2 / (a[i] + b[i]))` where non-zero denominators.
pub fn chi_squared_distance(a: &Histogram, b: &Histogram) -> f64 {
    assert_eq!(a.num_bins, b.num_bins);
    let mut chi2 = 0.0;
    for i in 0..a.num_bins {
        let denom = a.bins[i] + b.bins[i];
        if denom > 0.0 {
            let diff = a.bins[i] - b.bins[i];
            chi2 += (diff * diff) / denom;
        }
    }
    chi2
}

/// Correlation coefficient between two histograms.
///
/// Returns a value between -1.0 and 1.0 where 1.0 means perfect correlation.
pub fn histogram_correlation(a: &Histogram, b: &Histogram) -> f64 {
    assert_eq!(a.num_bins, b.num_bins);
    let n = a.num_bins;
    let mean_a: f64 = a.bins.iter().sum::<f64>() / n as f64;
    let mean_b: f64 = b.bins.iter().sum::<f64>() / n as f64;

    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;
    for i in 0..n {
        let da = a.bins[i] - mean_a;
        let db = b.bins[i] - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }

    let denom = (var_a * var_b).sqrt();
    if denom > 0.0 {
        cov / denom
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_histogram_creation() {
        let h = Histogram::new(32, 0.0, 256.0);
        assert_eq!(h.num_bins, 32);
        assert_eq!(h.bins.len(), 32);
        assert_eq!(h.total(), 0.0);
    }

    #[test]
    fn test_histogram_add_and_total() {
        let mut h = Histogram::new(8, 0.0, 8.0);
        h.add(0.5);
        h.add(1.5);
        h.add(7.5);
        assert!((h.total() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_histogram_bin_index() {
        let h = Histogram::new(4, 0.0, 4.0);
        assert_eq!(h.bin_index(0.0), Some(0));
        assert_eq!(h.bin_index(1.0), Some(1));
        assert_eq!(h.bin_index(3.9), Some(3));
        assert_eq!(h.bin_index(4.0), None);
        assert_eq!(h.bin_index(-0.1), None);
    }

    #[test]
    fn test_histogram_normalize() {
        let mut h = Histogram::new(4, 0.0, 4.0);
        h.add(0.5);
        h.add(0.5);
        h.add(2.5);
        h.add(3.5);
        h.normalize();
        assert!((h.total() - 1.0).abs() < 1e-10);
        assert!((h.bins[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_histogram_mean() {
        let mut h = Histogram::new(4, 0.0, 4.0);
        // All values in bin 0 (center = 0.5)
        h.add(0.1);
        h.add(0.2);
        let mean = h.mean();
        assert!((mean - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_histogram_2d_creation() {
        let h = Histogram2D::new(8, 8, (0.0, 180.0), (0.0, 256.0));
        assert_eq!(h.bins.len(), 64);
        assert_eq!(h.bins_x, 8);
        assert_eq!(h.bins_y, 8);
    }

    #[test]
    fn test_histogram_2d_add() {
        let mut h = Histogram2D::new(4, 4, (0.0, 4.0), (0.0, 4.0));
        h.add(0.5, 0.5);
        h.add(2.5, 2.5);
        assert!((h.get(0, 0) - 1.0).abs() < 1e-10);
        assert!((h.get(2, 2) - 1.0).abs() < 1e-10);
        assert!((h.get(1, 1) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_ratio_backproject() {
        let mut model = Histogram::new(4, 0.0, 4.0);
        model.bins = vec![0.5, 0.3, 0.1, 0.1];
        let mut scene = Histogram::new(4, 0.0, 4.0);
        scene.bins = vec![0.25, 0.25, 0.25, 0.25];
        let ratio = ratio_backproject(&model, &scene);
        assert!((ratio[0] - 2.0).abs() < 1e-10);
        assert!((ratio[1] - 1.2).abs() < 1e-10);
    }

    #[test]
    fn test_apply_backproject() {
        let ratio = vec![1.0, 2.0, 0.5, 0.25];
        let data = vec![0.5, 1.5, 2.5, 3.5];
        let result = apply_backproject(&data, &ratio, 0.0, 4.0);
        assert!((result[0] - 1.0).abs() < 1e-10);
        assert!((result[1] - 2.0).abs() < 1e-10);
        assert!((result[2] - 0.5).abs() < 1e-10);
        assert!((result[3] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_bayesian_backproject() {
        let mut model = Histogram::new(4, 0.0, 4.0);
        model.bins = vec![0.4, 0.3, 0.2, 0.1];
        let mut scene = Histogram::new(4, 0.0, 4.0);
        scene.bins = vec![0.25, 0.25, 0.25, 0.25];
        let result = bayesian_backproject(&model, &scene, 0.5);
        assert!((result[0] - 0.8).abs() < 1e-10);
        assert!((result[1] - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_point2d_distance() {
        let a = Point2D::new(0.0, 0.0);
        let b = Point2D::new(3.0, 4.0);
        assert!((a.distance(&b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_mean_shift_converges() {
        // 3x3 map with a peak at center
        let prob_map = vec![0.0, 0.1, 0.0, 0.1, 1.0, 0.1, 0.0, 0.1, 0.0];
        let initial = Point2D::new(0.5, 0.5);
        let result = mean_shift_iterate(&prob_map, 3, 3, initial, 3.0, 100, 0.001);
        // Should converge near center (1.5, 1.5)
        assert!((result.x - 1.5).abs() < 0.5);
        assert!((result.y - 1.5).abs() < 0.5);
    }

    #[test]
    fn test_histogram_intersection() {
        let mut a = Histogram::new(4, 0.0, 4.0);
        a.bins = vec![0.25, 0.25, 0.25, 0.25];
        let mut b = Histogram::new(4, 0.0, 4.0);
        b.bins = vec![0.25, 0.25, 0.25, 0.25];
        assert!((histogram_intersection(&a, &b) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_bhattacharyya_distance_identical() {
        let mut a = Histogram::new(4, 0.0, 4.0);
        a.bins = vec![0.25, 0.25, 0.25, 0.25];
        let dist = bhattacharyya_distance(&a, &a);
        assert!(dist < 1e-10);
    }

    #[test]
    fn test_chi_squared_distance() {
        let mut a = Histogram::new(4, 0.0, 4.0);
        a.bins = vec![1.0, 0.0, 0.0, 0.0];
        let mut b = Histogram::new(4, 0.0, 4.0);
        b.bins = vec![0.0, 1.0, 0.0, 0.0];
        let chi2 = chi_squared_distance(&a, &b);
        assert!((chi2 - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_histogram_correlation_perfect() {
        let mut a = Histogram::new(4, 0.0, 4.0);
        a.bins = vec![1.0, 2.0, 3.0, 4.0];
        let corr = histogram_correlation(&a, &a);
        assert!((corr - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_default_config() {
        let cfg = BackProjectConfig::default();
        assert_eq!(cfg.num_bins, DEFAULT_BINS);
        assert!(cfg.normalize_output);
        assert!((cfg.clip_threshold - 0.0).abs() < 1e-10);
    }
}
