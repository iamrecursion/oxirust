//! Utility functions for media analysis.
//!
//! This module provides helper functions and utilities used across
//! different analysis modules.

use crate::{AnalysisError, AnalysisResult};

/// Frame statistics calculator.
pub struct FrameStats {
    /// Minimum pixel value
    pub min: u8,
    /// Maximum pixel value
    pub max: u8,
    /// Average pixel value
    pub average: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Histogram
    pub histogram: [usize; 256],
}

impl FrameStats {
    /// Compute statistics for a frame.
    pub fn compute(frame: &[u8]) -> Self {
        let mut histogram = [0usize; 256];
        let mut min = 255u8;
        let mut max = 0u8;
        let mut sum = 0u64;

        for &pixel in frame {
            histogram[pixel as usize] += 1;
            min = min.min(pixel);
            max = max.max(pixel);
            sum += u64::from(pixel);
        }

        let average = sum as f64 / frame.len() as f64;

        // Compute standard deviation
        let variance: f64 = frame
            .iter()
            .map(|&p| {
                let diff = f64::from(p) - average;
                diff * diff
            })
            .sum::<f64>()
            / frame.len() as f64;

        let std_dev = variance.sqrt();

        Self {
            min,
            max,
            average,
            std_dev,
            histogram,
        }
    }

    /// Compute entropy of the frame.
    pub fn entropy(&self) -> f64 {
        let total = self.histogram.iter().sum::<usize>() as f64;
        if total == 0.0 {
            return 0.0;
        }

        self.histogram
            .iter()
            .filter(|&&count| count > 0)
            .map(|&count| {
                let p = count as f64 / total;
                -p * p.log2()
            })
            .sum()
    }

    /// Get percentile value.
    pub fn percentile(&self, p: f64) -> u8 {
        let target = (self.histogram.iter().sum::<usize>() as f64 * p) as usize;
        let mut cumulative = 0;

        for (value, &count) in self.histogram.iter().enumerate() {
            cumulative += count;
            if cumulative >= target {
                return value as u8;
            }
        }

        255
    }
}

/// Downsample a frame to a smaller size.
pub fn downsample_frame(
    frame: &[u8],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
) -> Vec<u8> {
    let mut downsampled = vec![0u8; dst_width * dst_height];

    let scale_x = src_width as f64 / dst_width as f64;
    let scale_y = src_height as f64 / dst_height as f64;

    for dst_y in 0..dst_height {
        for dst_x in 0..dst_width {
            let src_x = (dst_x as f64 * scale_x) as usize;
            let src_y = (dst_y as f64 * scale_y) as usize;

            let src_idx = src_y.min(src_height - 1) * src_width + src_x.min(src_width - 1);
            downsampled[dst_y * dst_width + dst_x] = frame[src_idx];
        }
    }

    downsampled
}

/// Compute PSNR (Peak Signal-to-Noise Ratio) between two frames.
pub fn compute_psnr(original: &[u8], modified: &[u8]) -> f64 {
    if original.len() != modified.len() {
        return 0.0;
    }

    let mse: f64 = original
        .iter()
        .zip(modified.iter())
        .map(|(&a, &b)| {
            let diff = i32::from(a) - i32::from(b);
            f64::from(diff * diff)
        })
        .sum::<f64>()
        / original.len() as f64;

    if mse < f64::EPSILON {
        return 100.0; // Perfect match
    }

    20.0 * (255.0f64).log10() - 10.0 * mse.log10()
}

/// Compute SSIM (Structural Similarity Index) for a window.
pub fn compute_ssim_window(
    original: &[u8],
    modified: &[u8],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    window_size: usize,
) -> f64 {
    const C1: f64 = 6.5025; // (0.01 * 255)^2
    const C2: f64 = 58.5225; // (0.03 * 255)^2

    let mut orig_sum = 0.0;
    let mut mod_sum = 0.0;
    let mut orig_sq_sum = 0.0;
    let mut mod_sq_sum = 0.0;
    let mut orig_mod_sum = 0.0;
    let mut count = 0;

    for dy in 0..window_size {
        for dx in 0..window_size {
            let px = x + dx;
            let py = y + dy;

            if px >= width || py >= height {
                continue;
            }

            let idx = py * width + px;
            let orig_val = f64::from(original[idx]);
            let mod_val = f64::from(modified[idx]);

            orig_sum += orig_val;
            mod_sum += mod_val;
            orig_sq_sum += orig_val * orig_val;
            mod_sq_sum += mod_val * mod_val;
            orig_mod_sum += orig_val * mod_val;
            count += 1;
        }
    }

    if count == 0 {
        return 0.0;
    }

    let count_f = f64::from(count);
    let mean_orig = orig_sum / count_f;
    let mean_mod = mod_sum / count_f;
    let var_orig = orig_sq_sum / count_f - mean_orig * mean_orig;
    let var_mod = mod_sq_sum / count_f - mean_mod * mean_mod;
    let covar = orig_mod_sum / count_f - mean_orig * mean_mod;

    let numerator = (2.0 * mean_orig * mean_mod + C1) * (2.0 * covar + C2);
    let denominator =
        (mean_orig * mean_orig + mean_mod * mean_mod + C1) * (var_orig + var_mod + C2);

    numerator / denominator
}

/// Apply a simple blur filter to a frame.
pub fn apply_blur(frame: &[u8], width: usize, height: usize, radius: usize) -> Vec<u8> {
    let mut blurred = vec![0u8; frame.len()];

    for y in 0..height {
        for x in 0..width {
            let mut sum = 0u32;
            let mut count = 0u32;

            for dy in -(radius as i32)..=(radius as i32) {
                for dx in -(radius as i32)..=(radius as i32) {
                    let px = (x as i32 + dx).max(0).min(width as i32 - 1) as usize;
                    let py = (y as i32 + dy).max(0).min(height as i32 - 1) as usize;

                    sum += u32::from(frame[py * width + px]);
                    count += 1;
                }
            }

            blurred[y * width + x] = (sum / count) as u8;
        }
    }

    blurred
}

/// Detect edges using Sobel operator.
pub fn detect_edges(frame: &[u8], width: usize, height: usize, threshold: u8) -> Vec<bool> {
    let mut edges = vec![false; frame.len()];

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            // Sobel X
            let gx = (i32::from(frame[(y - 1) * width + (x + 1)])
                + 2 * i32::from(frame[y * width + (x + 1)])
                + i32::from(frame[(y + 1) * width + (x + 1)]))
                - (i32::from(frame[(y - 1) * width + (x - 1)])
                    + 2 * i32::from(frame[y * width + (x - 1)])
                    + i32::from(frame[(y + 1) * width + (x - 1)]));

            // Sobel Y
            let gy = (i32::from(frame[(y + 1) * width + (x - 1)])
                + 2 * i32::from(frame[(y + 1) * width + x])
                + i32::from(frame[(y + 1) * width + (x + 1)]))
                - (i32::from(frame[(y - 1) * width + (x - 1)])
                    + 2 * i32::from(frame[(y - 1) * width + x])
                    + i32::from(frame[(y - 1) * width + (x + 1)]));

            let magnitude = f64::from(gx * gx + gy * gy).sqrt() as u8;
            edges[y * width + x] = magnitude > threshold;
        }
    }

    edges
}

/// Compute optical flow using Lucas-Kanade method (simplified).
pub fn compute_optical_flow(
    prev_frame: &[u8],
    curr_frame: &[u8],
    width: usize,
    height: usize,
    window_size: usize,
) -> Vec<(f64, f64)> {
    let mut flow = vec![(0.0, 0.0); width * height];

    for y in window_size..height - window_size {
        for x in window_size..width - window_size {
            let (fx, fy, _ft) = compute_gradients(prev_frame, curr_frame, width, x, y);

            if fx.abs() < 0.01 && fy.abs() < 0.01 {
                continue;
            }

            // Lucas-Kanade: solve Av = b
            let mut a11 = 0.0;
            let mut a12 = 0.0;
            let mut a22 = 0.0;
            let mut b1 = 0.0;
            let mut b2 = 0.0;

            for dy in -(window_size as i32)..=(window_size as i32) {
                for dx in -(window_size as i32)..=(window_size as i32) {
                    let px = (x as i32 + dx) as usize;
                    let py = (y as i32 + dy) as usize;

                    if px >= width || py >= height {
                        continue;
                    }

                    let (ix, iy, it) = compute_gradients(prev_frame, curr_frame, width, px, py);

                    a11 += ix * ix;
                    a12 += ix * iy;
                    a22 += iy * iy;
                    b1 -= ix * it;
                    b2 -= iy * it;
                }
            }

            let det = a11 * a22 - a12 * a12;
            if det.abs() > 0.0001 {
                let vx = (a22 * b1 - a12 * b2) / det;
                let vy = (a11 * b2 - a12 * b1) / det;
                flow[y * width + x] = (vx, vy);
            }
        }
    }

    flow
}

/// Compute image gradients.
fn compute_gradients(
    prev: &[u8],
    curr: &[u8],
    width: usize,
    x: usize,
    y: usize,
) -> (f64, f64, f64) {
    if x == 0 || y == 0 {
        return (0.0, 0.0, 0.0);
    }

    let idx = y * width + x;

    // Spatial gradients (using current frame)
    let fx = (f64::from(curr[idx + 1]) - f64::from(curr[idx - 1])) / 2.0;
    let fy = (f64::from(curr[idx + width]) - f64::from(curr[idx - width])) / 2.0;

    // Temporal gradient
    let ft = f64::from(curr[idx]) - f64::from(prev[idx]);

    (fx, fy, ft)
}

/// Convert RGB to YUV.
pub fn rgb_to_yuv(r: u8, g: u8, b: u8) -> (u8, u8, u8) {
    let r = f64::from(r);
    let g = f64::from(g);
    let b = f64::from(b);

    let y = 0.299 * r + 0.587 * g + 0.114 * b;
    let u = -0.147 * r - 0.289 * g + 0.436 * b + 128.0;
    let v = 0.615 * r - 0.515 * g - 0.100 * b + 128.0;

    (
        y.clamp(0.0, 255.0) as u8,
        u.clamp(0.0, 255.0) as u8,
        v.clamp(0.0, 255.0) as u8,
    )
}

/// Convert YUV to RGB.
pub fn yuv_to_rgb(y: u8, u: u8, v: u8) -> (u8, u8, u8) {
    let y = f64::from(y);
    let u = f64::from(u) - 128.0;
    let v = f64::from(v) - 128.0;

    let r = y + 1.140 * v;
    let g = y - 0.394 * u - 0.581 * v;
    let b = y + 2.032 * u;

    (
        r.clamp(0.0, 255.0) as u8,
        g.clamp(0.0, 255.0) as u8,
        b.clamp(0.0, 255.0) as u8,
    )
}

/// Normalize a frame to [0, 255] range.
pub fn normalize_frame(frame: &[u8]) -> Vec<u8> {
    let min = *frame.iter().min().unwrap_or(&0);
    let max = *frame.iter().max().unwrap_or(&255);

    if min == max {
        return vec![128; frame.len()];
    }

    let range = f64::from(max - min);

    frame
        .iter()
        .map(|&p| (f64::from(p - min) * 255.0 / range) as u8)
        .collect()
}

/// Apply histogram equalization to a frame.
pub fn histogram_equalization(frame: &[u8]) -> Vec<u8> {
    let mut histogram = [0usize; 256];

    // Compute histogram
    for &pixel in frame {
        histogram[pixel as usize] += 1;
    }

    // Compute CDF
    let mut cdf = [0usize; 256];
    cdf[0] = histogram[0];
    for i in 1..256 {
        cdf[i] = cdf[i - 1] + histogram[i];
    }

    let cdf_min = *cdf.iter().find(|&&x| x > 0).unwrap_or(&0);
    let total = frame.len();

    // Apply equalization
    frame
        .iter()
        .map(|&p| ((cdf[p as usize] - cdf_min) as f64 * 255.0 / (total - cdf_min) as f64) as u8)
        .collect()
}

/// Compute cross-correlation between two signals.
pub fn cross_correlation(signal1: &[f64], signal2: &[f64]) -> Vec<f64> {
    let len1 = signal1.len();
    let len2 = signal2.len();
    let max_lag = len1.min(len2);

    let mut correlation = vec![0.0; 2 * max_lag - 1];

    for lag in -(max_lag as i32 - 1)..max_lag as i32 {
        let mut sum = 0.0;
        let mut count = 0;

        for i in 0..len1 {
            let j = i as i32 + lag;
            if j >= 0 && (j as usize) < len2 {
                sum += signal1[i] * signal2[j as usize];
                count += 1;
            }
        }

        let idx = (lag + max_lag as i32 - 1) as usize;
        correlation[idx] = if count > 0 {
            sum / f64::from(count)
        } else {
            0.0
        };
    }

    correlation
}

/// Find peaks in a signal.
pub fn find_peaks(signal: &[f64], threshold: f64) -> Vec<usize> {
    let mut peaks = Vec::new();

    for i in 1..signal.len() - 1 {
        if signal[i] > threshold && signal[i] > signal[i - 1] && signal[i] > signal[i + 1] {
            peaks.push(i);
        }
    }

    peaks
}

/// Compute autocorrelation of a signal.
pub fn autocorrelation(signal: &[f64], max_lag: usize) -> Vec<f64> {
    let mut acf = vec![0.0; max_lag];

    let mean = signal.iter().sum::<f64>() / signal.len() as f64;

    for lag in 0..max_lag {
        let mut sum = 0.0;
        let mut count = 0;

        for i in 0..signal.len() - lag {
            sum += (signal[i] - mean) * (signal[i + lag] - mean);
            count += 1;
        }

        acf[lag] = if count > 0 {
            sum / f64::from(count)
        } else {
            0.0
        };
    }

    // Normalize by variance
    if acf[0] > 0.0 {
        let variance = acf[0];
        for val in &mut acf {
            *val /= variance;
        }
    }

    acf
}

/// Compute moving average of a signal.
pub fn moving_average(signal: &[f64], window_size: usize) -> Vec<f64> {
    if signal.len() < window_size {
        return signal.to_vec();
    }

    let mut averaged = Vec::with_capacity(signal.len());

    for i in 0..signal.len() {
        let start = i.saturating_sub((window_size - 1) / 2);
        let end = (i + (window_size - 1) / 2 + 1).min(signal.len());

        let sum: f64 = signal[start..end].iter().sum();
        let avg = sum / (end - start) as f64;
        averaged.push(avg);
    }

    averaged
}

/// Compute median filter of a signal.
pub fn median_filter(signal: &[f64], window_size: usize) -> Vec<f64> {
    let mut filtered = Vec::with_capacity(signal.len());

    for i in 0..signal.len() {
        let start = i.saturating_sub(window_size / 2);
        let end = (i + (window_size - 1) / 2 + 1).min(signal.len());

        let mut window: Vec<f64> = signal[start..end].to_vec();
        window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let median = if window.len() % 2 == 0 {
            (window[window.len() / 2 - 1] + window[window.len() / 2]) / 2.0
        } else {
            window[window.len() / 2]
        };

        filtered.push(median);
    }

    filtered
}

/// Validate frame dimensions.
pub fn validate_dimensions(
    y_plane: &[u8],
    u_plane: &[u8],
    v_plane: &[u8],
    width: usize,
    height: usize,
) -> AnalysisResult<()> {
    if y_plane.len() != width * height {
        return Err(AnalysisError::InvalidInput(format!(
            "Y plane size mismatch: expected {}, got {}",
            width * height,
            y_plane.len()
        )));
    }

    let uv_width = width.div_ceil(2);
    let uv_height = height.div_ceil(2);

    if u_plane.len() != uv_width * uv_height {
        return Err(AnalysisError::InvalidInput(format!(
            "U plane size mismatch: expected {}, got {}",
            uv_width * uv_height,
            u_plane.len()
        )));
    }

    if v_plane.len() != uv_width * uv_height {
        return Err(AnalysisError::InvalidInput(format!(
            "V plane size mismatch: expected {}, got {}",
            uv_width * uv_height,
            v_plane.len()
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_stats() {
        let frame = vec![100u8, 150, 200, 50, 25];
        let stats = FrameStats::compute(&frame);

        assert_eq!(stats.min, 25);
        assert_eq!(stats.max, 200);
        assert!((stats.average - 105.0).abs() < 0.1);
        assert!(stats.std_dev > 0.0);
    }

    #[test]
    fn test_entropy() {
        // Uniform distribution should have high entropy
        let mut frame = Vec::new();
        for i in 0..256 {
            frame.push(i as u8);
        }
        let stats = FrameStats::compute(&frame);
        let entropy = stats.entropy();
        assert!(entropy > 7.0); // Should be close to 8.0

        // Single value should have zero entropy
        let uniform_frame = vec![128u8; 1000];
        let uniform_stats = FrameStats::compute(&uniform_frame);
        let uniform_entropy = uniform_stats.entropy();
        assert!(uniform_entropy < 0.01);
    }

    #[test]
    fn test_percentile() {
        let frame: Vec<u8> = (0..256).map(|i| i as u8).collect();
        let stats = FrameStats::compute(&frame);

        assert_eq!(stats.percentile(0.5), 127); // Median
        assert!(stats.percentile(0.95) > 240);
    }

    #[test]
    fn test_downsample() {
        let frame = vec![255u8; 1920 * 1080];
        let downsampled = downsample_frame(&frame, 1920, 1080, 640, 480);

        assert_eq!(downsampled.len(), 640 * 480);
        assert!(downsampled.iter().all(|&p| p == 255));
    }

    #[test]
    fn test_psnr() {
        let frame1 = vec![100u8; 1000];
        let frame2 = vec![100u8; 1000];

        let psnr = compute_psnr(&frame1, &frame2);
        assert!(psnr > 90.0); // Should be very high for identical frames
    }

    #[test]
    fn test_blur() {
        let mut frame = vec![0u8; 100 * 100];

        // Create a bright spot
        frame[50 * 100 + 50] = 255;

        let blurred = apply_blur(&frame, 100, 100, 2);

        // Bright spot should be spread
        assert!(blurred[50 * 100 + 50] < 255);
        assert!(blurred[50 * 100 + 51] > 0);
    }

    #[test]
    fn test_edge_detection() {
        let mut frame = vec![0u8; 100 * 100];

        // Create a vertical edge
        for y in 0..100 {
            for x in 50..100 {
                frame[y * 100 + x] = 255;
            }
        }

        let edges = detect_edges(&frame, 100, 100, 30);
        let edge_count = edges.iter().filter(|&&e| e).count();

        assert!(edge_count > 50); // Should detect edge pixels
    }

    #[test]
    fn test_rgb_yuv_conversion() {
        let (y, u, v) = rgb_to_yuv(255, 0, 0); // Red
        let (r, g, b) = yuv_to_rgb(y, u, v);

        // Should be close to original red
        assert!((r as i32 - 255).abs() < 50);
        assert!((g as i32).abs() < 25);
        assert!((b as i32).abs() < 10);
    }

    #[test]
    fn test_normalize() {
        let frame = vec![50u8, 100, 150, 200];
        let normalized = normalize_frame(&frame);

        assert_eq!(normalized[0], 0);
        assert_eq!(normalized[3], 255);
    }

    #[test]
    fn test_histogram_equalization() {
        let frame = vec![50u8; 100];
        let equalized = histogram_equalization(&frame);

        // Should spread values
        assert!(!equalized.iter().all(|&p| p == 50));
    }

    #[test]
    fn test_cross_correlation() {
        let signal1 = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let signal2 = vec![1.0, 2.0, 3.0, 2.0, 1.0];

        let corr = cross_correlation(&signal1, &signal2);
        // Should have peak at zero lag
        assert!(!corr.is_empty());
    }

    #[test]
    fn test_find_peaks() {
        let signal = vec![0.1, 0.5, 0.2, 0.8, 0.3, 0.6, 0.1];
        let peaks = find_peaks(&signal, 0.4);

        assert!(peaks.contains(&1)); // 0.5
        assert!(peaks.contains(&3)); // 0.8
        assert!(peaks.contains(&5)); // 0.6
    }

    #[test]
    fn test_autocorrelation() {
        let signal = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let acf = autocorrelation(&signal, 3);

        assert_eq!(acf.len(), 3);
        assert!((acf[0] - 1.0).abs() < 0.01); // Should be 1.0 at lag 0
    }

    #[test]
    fn test_moving_average() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let averaged = moving_average(&signal, 3);

        assert_eq!(averaged.len(), 5);
        assert!((averaged[2] - 3.0).abs() < 0.1);
    }

    #[test]
    fn test_median_filter() {
        let signal = vec![1.0, 10.0, 2.0, 3.0, 4.0]; // 10.0 is outlier
        let filtered = median_filter(&signal, 3);

        // Outlier should be removed
        assert!(filtered[1] < 6.0);
    }

    #[test]
    fn test_validate_dimensions() {
        let y = vec![0u8; 1920 * 1080];
        let u = vec![0u8; 960 * 540];
        let v = vec![0u8; 960 * 540];

        assert!(validate_dimensions(&y, &u, &v, 1920, 1080).is_ok());

        let bad_y = vec![0u8; 1000];
        assert!(validate_dimensions(&bad_y, &u, &v, 1920, 1080).is_err());
    }
}
