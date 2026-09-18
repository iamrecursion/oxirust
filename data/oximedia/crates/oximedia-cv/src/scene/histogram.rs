//! Histogram-based scene detection.
//!
//! This module provides scene detection using histogram comparison methods.
//! It supports both RGB and HSV color spaces and various comparison metrics.
//!
//! [`ColorHistogram::compute_rgb`] uses rayon parallel reduction for large
//! frames: each thread accumulates its own bin array and they are merged at
//! the end.  The result is bit-exact with the serial path.

use rayon::prelude::*;

use crate::error::{CvError, CvResult};
use crate::image::Histogram;
use oximedia_codec::VideoFrame;
use oximedia_core::PixelFormat;

use super::{ChangeType, SceneChange, SceneConfig, SceneMetadata};

/// Configuration for histogram-based detection.
#[derive(Debug, Clone)]
pub struct HistogramConfig {
    /// Number of bins per channel (default: 256 for grayscale, 64 for RGB).
    pub bins: usize,
    /// Use color histogram (true) or grayscale (false).
    pub use_color: bool,
    /// Comparison metric to use.
    pub metric: HistogramMetric,
    /// Weight for each color channel (R, G, B).
    pub channel_weights: [f64; 3],
}

impl Default for HistogramConfig {
    fn default() -> Self {
        Self {
            bins: 64,
            use_color: true,
            metric: HistogramMetric::ChiSquared,
            channel_weights: [0.299, 0.587, 0.114], // Rec. 601 luma coefficients
        }
    }
}

/// Histogram comparison metric.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HistogramMetric {
    /// Chi-squared distance.
    ChiSquared,
    /// Histogram intersection.
    Intersection,
    /// Bhattacharyya distance.
    Bhattacharyya,
    /// Correlation.
    Correlation,
}

/// Color histogram for RGB frames.
#[derive(Debug, Clone)]
pub struct ColorHistogram {
    /// Red channel histogram.
    pub r: Vec<u32>,
    /// Green channel histogram.
    pub g: Vec<u32>,
    /// Blue channel histogram.
    pub b: Vec<u32>,
    /// Number of bins per channel.
    pub bins: usize,
}

impl ColorHistogram {
    /// Create a new color histogram with the specified number of bins.
    #[must_use]
    pub fn new(bins: usize) -> Self {
        Self {
            r: vec![0; bins],
            g: vec![0; bins],
            b: vec![0; bins],
            bins,
        }
    }

    /// Compute histogram from RGB frame data.
    ///
    /// For large frames the pixel array is split across rayon threads, each
    /// of which builds its own `[u64; bins * 3]` accumulator; the per-thread
    /// results are reduced by element-wise addition.  The final counts are
    /// truncated back to `u32`, matching the serial result exactly.
    pub fn compute_rgb(data: &[u8], width: u32, height: u32, bins: usize) -> CvResult<Self> {
        if width == 0 || height == 0 {
            return Err(CvError::invalid_dimensions(width, height));
        }

        let expected_size = (width * height * 3) as usize;
        if data.len() < expected_size {
            return Err(CvError::insufficient_data(expected_size, data.len()));
        }

        let bin_scale = bins as f64 / 256.0;
        let bins3 = bins * 3;

        // Parallel reduce: each rayon thread accumulates into its own flat
        // `[u64; bins*3]` array (layout: r[0..bins] | g[bins..2*bins] | b[2*bins..3*bins]).
        let flat: Vec<u64> = data[..expected_size]
            .par_chunks(3)
            .fold(
                || vec![0u64; bins3],
                |mut acc, chunk| {
                    let r_bin = ((chunk[0] as f64 * bin_scale) as usize).min(bins - 1);
                    let g_bin = ((chunk[1] as f64 * bin_scale) as usize).min(bins - 1);
                    let b_bin = ((chunk[2] as f64 * bin_scale) as usize).min(bins - 1);
                    acc[r_bin] += 1;
                    acc[bins + g_bin] += 1;
                    acc[2 * bins + b_bin] += 1;
                    acc
                },
            )
            .reduce(
                || vec![0u64; bins3],
                |mut a, b| {
                    for (av, bv) in a.iter_mut().zip(b.iter()) {
                        *av += bv;
                    }
                    a
                },
            );

        let mut hist = Self::new(bins);
        for i in 0..bins {
            hist.r[i] = flat[i] as u32;
            hist.g[i] = flat[bins + i] as u32;
            hist.b[i] = flat[2 * bins + i] as u32;
        }

        Ok(hist)
    }

    /// Serial (single-threaded) reference path — used in tests to verify
    /// bit-exactness of the parallel implementation.
    #[cfg(test)]
    pub(crate) fn compute_rgb_serial(
        data: &[u8],
        width: u32,
        height: u32,
        bins: usize,
    ) -> CvResult<Self> {
        if width == 0 || height == 0 {
            return Err(CvError::invalid_dimensions(width, height));
        }
        let expected_size = (width * height * 3) as usize;
        if data.len() < expected_size {
            return Err(CvError::insufficient_data(expected_size, data.len()));
        }
        let mut hist = Self::new(bins);
        let bin_scale = bins as f64 / 256.0;
        for chunk in data[..expected_size].chunks_exact(3) {
            let r_bin = ((chunk[0] as f64 * bin_scale) as usize).min(bins - 1);
            let g_bin = ((chunk[1] as f64 * bin_scale) as usize).min(bins - 1);
            let b_bin = ((chunk[2] as f64 * bin_scale) as usize).min(bins - 1);
            hist.r[r_bin] += 1;
            hist.g[g_bin] += 1;
            hist.b[b_bin] += 1;
        }
        Ok(hist)
    }

    /// Normalize the histogram to [0, 1].
    #[must_use]
    pub fn normalized(&self) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let total_r: u32 = self.r.iter().sum();
        let total_g: u32 = self.g.iter().sum();
        let total_b: u32 = self.b.iter().sum();

        let norm_r: Vec<f64> = if total_r > 0 {
            self.r.iter().map(|&v| v as f64 / total_r as f64).collect()
        } else {
            vec![0.0; self.bins]
        };

        let norm_g: Vec<f64> = if total_g > 0 {
            self.g.iter().map(|&v| v as f64 / total_g as f64).collect()
        } else {
            vec![0.0; self.bins]
        };

        let norm_b: Vec<f64> = if total_b > 0 {
            self.b.iter().map(|&v| v as f64 / total_b as f64).collect()
        } else {
            vec![0.0; self.bins]
        };

        (norm_r, norm_g, norm_b)
    }

    /// Compare with another color histogram using the specified metric.
    #[must_use]
    pub fn compare(&self, other: &Self, metric: HistogramMetric, weights: &[f64; 3]) -> f64 {
        let (n1_r, n1_g, n1_b) = self.normalized();
        let (n2_r, n2_g, n2_b) = other.normalized();

        let dist_r = compare_histogram_vectors(&n1_r, &n2_r, metric);
        let dist_g = compare_histogram_vectors(&n1_g, &n2_g, metric);
        let dist_b = compare_histogram_vectors(&n1_b, &n2_b, metric);

        // Weighted average
        dist_r * weights[0] + dist_g * weights[1] + dist_b * weights[2]
    }
}

/// HSV histogram for color-based detection.
#[derive(Debug, Clone)]
pub struct HsvHistogram {
    /// Hue channel histogram.
    pub h: Vec<u32>,
    /// Saturation channel histogram.
    pub s: Vec<u32>,
    /// Value channel histogram.
    pub v: Vec<u32>,
    /// Number of bins per channel.
    pub bins: usize,
}

impl HsvHistogram {
    /// Create a new HSV histogram.
    #[must_use]
    pub fn new(bins: usize) -> Self {
        Self {
            h: vec![0; bins],
            s: vec![0; bins],
            v: vec![0; bins],
            bins,
        }
    }

    /// Compute histogram from RGB frame data (converts to HSV).
    pub fn compute_from_rgb(data: &[u8], width: u32, height: u32, bins: usize) -> CvResult<Self> {
        if width == 0 || height == 0 {
            return Err(CvError::invalid_dimensions(width, height));
        }

        let expected_size = (width * height * 3) as usize;
        if data.len() < expected_size {
            return Err(CvError::insufficient_data(expected_size, data.len()));
        }

        let mut hist = Self::new(bins);
        let bin_scale = bins as f64 / 256.0;

        for chunk in data.chunks_exact(3) {
            let (h, s, v) = rgb_to_hsv(chunk[0], chunk[1], chunk[2]);

            let h_bin = ((h * bin_scale) as usize).min(bins - 1);
            let s_bin = ((s * bin_scale) as usize).min(bins - 1);
            let v_bin = ((v * bin_scale) as usize).min(bins - 1);

            hist.h[h_bin] += 1;
            hist.s[s_bin] += 1;
            hist.v[v_bin] += 1;
        }

        Ok(hist)
    }

    /// Normalize the histogram.
    #[must_use]
    pub fn normalized(&self) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let total_h: u32 = self.h.iter().sum();
        let total_s: u32 = self.s.iter().sum();
        let total_v: u32 = self.v.iter().sum();

        let norm_h: Vec<f64> = if total_h > 0 {
            self.h.iter().map(|&v| v as f64 / total_h as f64).collect()
        } else {
            vec![0.0; self.bins]
        };

        let norm_s: Vec<f64> = if total_s > 0 {
            self.s.iter().map(|&v| v as f64 / total_s as f64).collect()
        } else {
            vec![0.0; self.bins]
        };

        let norm_v: Vec<f64> = if total_v > 0 {
            self.v.iter().map(|&v| v as f64 / total_v as f64).collect()
        } else {
            vec![0.0; self.bins]
        };

        (norm_h, norm_s, norm_v)
    }

    /// Compare with another HSV histogram.
    #[must_use]
    pub fn compare(&self, other: &Self, metric: HistogramMetric) -> f64 {
        let (n1_h, n1_s, n1_v) = self.normalized();
        let (n2_h, n2_s, n2_v) = other.normalized();

        // HSV comparison: weight hue more heavily for color differences
        let dist_h = compare_histogram_vectors(&n1_h, &n2_h, metric);
        let dist_s = compare_histogram_vectors(&n1_s, &n2_s, metric);
        let dist_v = compare_histogram_vectors(&n1_v, &n2_v, metric);

        dist_h * 0.5 + dist_s * 0.3 + dist_v * 0.2
    }
}

/// Convert RGB to HSV.
fn rgb_to_hsv(r: u8, g: u8, b: u8) -> (f64, f64, f64) {
    let r = r as f64 / 255.0;
    let g = g as f64 / 255.0;
    let b = b as f64 / 255.0;

    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;

    let h = if delta < f64::EPSILON {
        0.0
    } else if (max - r).abs() < f64::EPSILON {
        60.0 * (((g - b) / delta) % 6.0)
    } else if (max - g).abs() < f64::EPSILON {
        60.0 * (((b - r) / delta) + 2.0)
    } else {
        60.0 * (((r - g) / delta) + 4.0)
    };

    let h = if h < 0.0 { h + 360.0 } else { h };
    let h = (h / 360.0 * 255.0).clamp(0.0, 255.0); // Scale to 0-255

    let s = if max < f64::EPSILON {
        0.0
    } else {
        (delta / max) * 255.0
    };

    let v = max * 255.0;

    (h, s, v)
}

/// Compare two histogram vectors using the specified metric.
fn compare_histogram_vectors(h1: &[f64], h2: &[f64], metric: HistogramMetric) -> f64 {
    match metric {
        HistogramMetric::ChiSquared => chi_squared_distance(h1, h2),
        HistogramMetric::Intersection => 1.0 - histogram_intersection(h1, h2),
        HistogramMetric::Bhattacharyya => bhattacharyya_distance(h1, h2),
        HistogramMetric::Correlation => 1.0 - histogram_correlation(h1, h2),
    }
}

/// Compute chi-squared distance between histograms.
fn chi_squared_distance(h1: &[f64], h2: &[f64]) -> f64 {
    let mut chi_sq = 0.0;

    for (v1, v2) in h1.iter().zip(h2.iter()) {
        let sum = v1 + v2;
        if sum > f64::EPSILON {
            let diff = v1 - v2;
            chi_sq += diff * diff / sum;
        }
    }

    // Normalize to [0, 1]
    (chi_sq / 2.0).min(1.0)
}

/// Compute histogram intersection (similarity measure).
fn histogram_intersection(h1: &[f64], h2: &[f64]) -> f64 {
    h1.iter().zip(h2.iter()).map(|(v1, v2)| v1.min(*v2)).sum()
}

/// Compute Bhattacharyya distance.
fn bhattacharyya_distance(h1: &[f64], h2: &[f64]) -> f64 {
    let bc: f64 = h1
        .iter()
        .zip(h2.iter())
        .map(|(v1, v2)| (v1 * v2).sqrt())
        .sum();

    // Bhattacharyya coefficient to distance
    if bc > 0.0 {
        (-bc.ln()).sqrt().min(1.0)
    } else {
        1.0
    }
}

/// Compute histogram correlation.
fn histogram_correlation(h1: &[f64], h2: &[f64]) -> f64 {
    let mean1: f64 = h1.iter().sum::<f64>() / h1.len() as f64;
    let mean2: f64 = h2.iter().sum::<f64>() / h2.len() as f64;

    let mut numerator = 0.0;
    let mut denom1 = 0.0;
    let mut denom2 = 0.0;

    for (v1, v2) in h1.iter().zip(h2.iter()) {
        let d1 = v1 - mean1;
        let d2 = v2 - mean2;
        numerator += d1 * d2;
        denom1 += d1 * d1;
        denom2 += d2 * d2;
    }

    let denom = (denom1 * denom2).sqrt();
    if denom > f64::EPSILON {
        (numerator / denom).clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

/// Extract frame data as RGB.
fn extract_frame_rgb(frame: &VideoFrame) -> CvResult<Vec<u8>> {
    match frame.format {
        PixelFormat::Rgb24 => {
            if frame.planes.is_empty() {
                return Err(CvError::insufficient_data(1, 0));
            }
            Ok(frame.planes[0].data.clone())
        }
        PixelFormat::Yuv420p | PixelFormat::Yuv422p | PixelFormat::Yuv444p => {
            // Convert YUV to RGB
            convert_yuv_to_rgb(frame)
        }
        _ => Err(CvError::unsupported_format(format!("{:?}", frame.format))),
    }
}

/// Convert YUV frame to RGB.
fn convert_yuv_to_rgb(frame: &VideoFrame) -> CvResult<Vec<u8>> {
    if frame.planes.len() < 3 {
        return Err(CvError::insufficient_data(3, frame.planes.len()));
    }

    let width = frame.width as usize;
    let height = frame.height as usize;
    let mut rgb = vec![0u8; width * height * 3];

    let y_plane = &frame.planes[0].data;
    let u_plane = &frame.planes[1].data;
    let v_plane = &frame.planes[2].data;

    let (h_ratio, v_ratio) = frame.format.chroma_subsampling();

    for y in 0..height {
        for x in 0..width {
            let y_idx = y * width + x;
            let uv_x = x / h_ratio as usize;
            let uv_y = y / v_ratio as usize;
            let uv_width = width.div_ceil(h_ratio as usize);
            let uv_idx = uv_y * uv_width + uv_x;

            if y_idx >= y_plane.len() || uv_idx >= u_plane.len() || uv_idx >= v_plane.len() {
                continue;
            }

            let y_val = y_plane[y_idx] as i32;
            let u_val = u_plane[uv_idx] as i32 - 128;
            let v_val = v_plane[uv_idx] as i32 - 128;

            let r = (y_val + ((v_val * 91_881) >> 16)).clamp(0, 255) as u8;
            let g = (y_val - ((u_val * 22_553 + v_val * 46_801) >> 16)).clamp(0, 255) as u8;
            let b = (y_val + ((u_val * 116_129) >> 16)).clamp(0, 255) as u8;

            let rgb_idx = y_idx * 3;
            rgb[rgb_idx] = r;
            rgb[rgb_idx + 1] = g;
            rgb[rgb_idx + 2] = b;
        }
    }

    Ok(rgb)
}

/// Compute frame similarity using histogram comparison.
pub fn compute_frame_similarity(frame1: &VideoFrame, frame2: &VideoFrame) -> CvResult<f64> {
    let config = HistogramConfig::default();
    compute_frame_similarity_with_config(frame1, frame2, &config)
}

/// Compute frame similarity using histogram comparison with custom config.
pub fn compute_frame_similarity_with_config(
    frame1: &VideoFrame,
    frame2: &VideoFrame,
    config: &HistogramConfig,
) -> CvResult<f64> {
    if frame1.width != frame2.width || frame1.height != frame2.height {
        return Err(CvError::invalid_parameter(
            "frames",
            "dimensions must match",
        ));
    }

    let data1 = extract_frame_rgb(frame1)?;
    let data2 = extract_frame_rgb(frame2)?;

    let hist1 = ColorHistogram::compute_rgb(&data1, frame1.width, frame1.height, config.bins)?;
    let hist2 = ColorHistogram::compute_rgb(&data2, frame2.width, frame2.height, config.bins)?;

    let distance = hist1.compare(&hist2, config.metric, &config.channel_weights);

    // Convert distance to similarity
    Ok(1.0 - distance)
}

/// Compute frame similarity using HSV histogram.
pub fn compute_frame_similarity_hsv(frame1: &VideoFrame, frame2: &VideoFrame) -> CvResult<f64> {
    if frame1.width != frame2.width || frame1.height != frame2.height {
        return Err(CvError::invalid_parameter(
            "frames",
            "dimensions must match",
        ));
    }

    let config = HistogramConfig::default();
    let data1 = extract_frame_rgb(frame1)?;
    let data2 = extract_frame_rgb(frame2)?;

    let hist1 = HsvHistogram::compute_from_rgb(&data1, frame1.width, frame1.height, config.bins)?;
    let hist2 = HsvHistogram::compute_from_rgb(&data2, frame2.width, frame2.height, config.bins)?;

    let distance = hist1.compare(&hist2, config.metric);

    Ok(1.0 - distance)
}

/// Detect histogram-based scene changes.
pub fn detect_histogram_changes(
    frames: &[VideoFrame],
    config: &SceneConfig,
) -> CvResult<Vec<SceneChange>> {
    let mut changes = Vec::new();

    for i in 1..frames.len() {
        let similarity = compute_frame_similarity_with_config(
            &frames[i - 1],
            &frames[i],
            &config.histogram_config,
        )?;
        let diff = 1.0 - similarity;

        if diff > config.threshold {
            changes.push(SceneChange {
                frame_number: i,
                timestamp: frames[i].timestamp,
                confidence: diff,
                change_type: ChangeType::Cut,
                metadata: SceneMetadata {
                    histogram_diff: Some(diff),
                    ..Default::default()
                },
            });
        }
    }

    Ok(changes)
}

/// Detect histogram-based scene changes using HSV.
pub fn detect_histogram_hsv_changes(
    frames: &[VideoFrame],
    config: &SceneConfig,
) -> CvResult<Vec<SceneChange>> {
    let mut changes = Vec::new();

    for i in 1..frames.len() {
        let similarity = compute_frame_similarity_hsv(&frames[i - 1], &frames[i])?;
        let diff = 1.0 - similarity;

        if diff > config.threshold {
            changes.push(SceneChange {
                frame_number: i,
                timestamp: frames[i].timestamp,
                confidence: diff,
                change_type: ChangeType::Cut,
                metadata: SceneMetadata {
                    histogram_diff: Some(diff),
                    ..Default::default()
                },
            });
        }
    }

    Ok(changes)
}

/// Compute average brightness of a frame.
pub fn compute_average_brightness(frame: &VideoFrame) -> CvResult<f64> {
    match frame.format {
        PixelFormat::Yuv420p | PixelFormat::Yuv422p | PixelFormat::Yuv444p => {
            if frame.planes.is_empty() {
                return Err(CvError::insufficient_data(1, 0));
            }

            let y_plane = &frame.planes[0].data;
            let sum: u64 = y_plane.iter().map(|&v| v as u64).sum();
            let avg = sum as f64 / y_plane.len() as f64;

            Ok(avg)
        }
        PixelFormat::Rgb24 => {
            let data = extract_frame_rgb(frame)?;
            let mut sum = 0u64;

            for chunk in data.chunks_exact(3) {
                // Use Rec. 601 luma coefficients
                let luma = (chunk[0] as f64 * 0.299
                    + chunk[1] as f64 * 0.587
                    + chunk[2] as f64 * 0.114) as u64;
                sum += luma;
            }

            let pixels = (frame.width * frame.height) as f64;
            Ok(sum as f64 / pixels)
        }
        _ => Err(CvError::unsupported_format(format!("{:?}", frame.format))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Solid red 100×100 image: all R-channel pixels = 255, G = 0, B = 0.
    ///
    /// With 256 bins:
    /// - R bin 255 must hold exactly `pixels` counts; all other R bins are zero.
    /// - G bin 0 must hold exactly `pixels` counts (G=0 maps to bin 0); all other G bins are zero.
    /// - B bin 0 must hold exactly `pixels` counts (B=0 maps to bin 0); all other B bins are zero.
    #[test]
    fn test_compute_rgb_solid_red_analytic() {
        let w = 100u32;
        let h = 100u32;
        let pixels = (w * h) as usize;
        // Solid red: [255, 0, 0] repeated
        let data: Vec<u8> = (0..pixels).flat_map(|_| [255u8, 0u8, 0u8]).collect();
        let hist =
            ColorHistogram::compute_rgb(&data, w, h, 256).expect("compute_rgb should succeed");

        // R: all pixels map to bin 255
        assert_eq!(
            hist.r[255], pixels as u32,
            "r[255] should equal pixel count"
        );
        for (i, &v) in hist.r.iter().enumerate() {
            if i != 255 {
                assert_eq!(v, 0, "r[{i}] should be zero");
            }
        }
        // G: all pixels map to bin 0 (value 0 → bin 0)
        assert_eq!(hist.g[0], pixels as u32, "g[0] should equal pixel count");
        for (i, &v) in hist.g.iter().enumerate() {
            if i != 0 {
                assert_eq!(v, 0, "g[{i}] should be zero");
            }
        }
        // B: all pixels map to bin 0 (value 0 → bin 0)
        assert_eq!(hist.b[0], pixels as u32, "b[0] should equal pixel count");
        for (i, &v) in hist.b.iter().enumerate() {
            if i != 0 {
                assert_eq!(v, 0, "b[{i}] should be zero");
            }
        }
    }

    /// Parallel and serial paths must be bit-exact on a 1920×1080 pseudo-random frame.
    #[test]
    fn test_compute_rgb_parallel_matches_serial_1920x1080() {
        let w = 1920u32;
        let h = 1080u32;
        let pixels = (w * h) as usize;
        // Deterministic pseudo-random data via a simple LCG
        let mut v = 12345u32;
        let data: Vec<u8> = (0..pixels * 3)
            .map(|_| {
                v = v.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (v >> 24) as u8
            })
            .collect();

        let parallel = ColorHistogram::compute_rgb(&data, w, h, 64).expect("parallel compute_rgb");
        let serial =
            ColorHistogram::compute_rgb_serial(&data, w, h, 64).expect("serial compute_rgb");

        assert_eq!(parallel.r, serial.r, "R histogram mismatch");
        assert_eq!(parallel.g, serial.g, "G histogram mismatch");
        assert_eq!(parallel.b, serial.b, "B histogram mismatch");
    }

    /// Verify total bin counts equal pixel count (conservation).
    #[test]
    fn test_compute_rgb_bin_conservation() {
        let w = 64u32;
        let h = 64u32;
        let pixels = (w * h) as usize;
        let data: Vec<u8> = (0..pixels * 3).map(|i| (i % 256) as u8).collect();
        let hist =
            ColorHistogram::compute_rgb(&data, w, h, 64).expect("compute_rgb should succeed");
        let total_r: u32 = hist.r.iter().sum();
        let total_g: u32 = hist.g.iter().sum();
        let total_b: u32 = hist.b.iter().sum();
        assert_eq!(total_r, pixels as u32);
        assert_eq!(total_g, pixels as u32);
        assert_eq!(total_b, pixels as u32);
    }
}
