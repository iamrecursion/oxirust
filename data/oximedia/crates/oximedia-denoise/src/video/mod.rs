//! Video denoising algorithms.
//!
//! Provides spatial and temporal denoising filters for video frames represented
//! as flat `f32` slices (row-major, single channel / luma).

use std::collections::VecDeque;

// ============================================================
// Spatial Denoiser – Gaussian
// ============================================================

/// Spatial denoiser using Gaussian blur.
pub struct SpatialDenoiser;

impl SpatialDenoiser {
    /// Create a new `SpatialDenoiser`.
    pub fn new() -> Self {
        Self
    }

    /// Apply a 5×5 Gaussian kernel parameterized by `sigma`.
    ///
    /// `frame` is a row-major, single-channel image of size `width × height`.
    pub fn gaussian_denoise(
        &self,
        frame: &[f32],
        width: usize,
        height: usize,
        sigma: f32,
    ) -> Vec<f32> {
        if frame.is_empty() || width == 0 || height == 0 {
            return frame.to_vec();
        }

        let kernel = gaussian_kernel_5x5(sigma);
        apply_5x5_kernel(frame, width, height, &kernel)
    }
}

impl Default for SpatialDenoiser {
    fn default() -> Self {
        Self::new()
    }
}

/// Build a 5×5 Gaussian kernel (25 elements, row-major) for a given `sigma`.
fn gaussian_kernel_5x5(sigma: f32) -> [f32; 25] {
    let sigma_sq = sigma * sigma;
    let mut kernel = [0.0f32; 25];
    let mut sum = 0.0f32;

    for dy in -2_i32..=2 {
        for dx in -2_i32..=2 {
            let idx = ((dy + 2) * 5 + (dx + 2)) as usize;
            let v = (-(dx * dx + dy * dy) as f32 / (2.0 * sigma_sq)).exp();
            kernel[idx] = v;
            sum += v;
        }
    }
    // Normalize
    for k in &mut kernel {
        *k /= sum;
    }
    kernel
}

/// Apply a 5×5 separable-style kernel to a frame with clamp-to-edge padding.
fn apply_5x5_kernel(frame: &[f32], width: usize, height: usize, kernel: &[f32; 25]) -> Vec<f32> {
    let mut output = vec![0.0f32; width * height];

    for y in 0..height {
        for x in 0..width {
            let mut acc = 0.0f32;
            for ky in 0..5_usize {
                for kx in 0..5_usize {
                    let sy = (y as i64 + ky as i64 - 2).clamp(0, height as i64 - 1) as usize;
                    let sx = (x as i64 + kx as i64 - 2).clamp(0, width as i64 - 1) as usize;
                    acc += frame[sy * width + sx] * kernel[ky * 5 + kx];
                }
            }
            output[y * width + x] = acc;
        }
    }
    output
}

// ============================================================
// Bilateral Filter
// ============================================================

/// Bilateral filter for edge-preserving denoising.
pub struct BilateralFilter;

impl BilateralFilter {
    /// Create a new `BilateralFilter`.
    pub fn new() -> Self {
        Self
    }

    /// Apply a bilateral filter.
    ///
    /// `sigma_space` controls the spatial Gaussian width (in pixels).
    /// `sigma_range` controls the intensity Gaussian width.
    /// Uses a 5×5 window.
    pub fn filter(
        &self,
        frame: &[f32],
        width: usize,
        height: usize,
        sigma_space: f32,
        sigma_range: f32,
    ) -> Vec<f32> {
        if frame.is_empty() || width == 0 || height == 0 {
            return frame.to_vec();
        }

        let ss2 = 2.0 * sigma_space * sigma_space;
        let sr2 = 2.0 * sigma_range * sigma_range;
        let mut output = vec![0.0f32; width * height];

        for y in 0..height {
            for x in 0..width {
                let center = frame[y * width + x];
                let mut acc = 0.0f32;
                let mut weight_sum = 0.0f32;

                for dy in -2_i32..=2 {
                    for dx in -2_i32..=2 {
                        let ny = (y as i64 + dy as i64).clamp(0, height as i64 - 1) as usize;
                        let nx = (x as i64 + dx as i64).clamp(0, width as i64 - 1) as usize;
                        let neighbor = frame[ny * width + nx];

                        let spatial_dist_sq = (dx * dx + dy * dy) as f32;
                        let range_dist_sq = (neighbor - center) * (neighbor - center);

                        let w = (-spatial_dist_sq / ss2 - range_dist_sq / sr2).exp();
                        acc += neighbor * w;
                        weight_sum += w;
                    }
                }

                output[y * width + x] = if weight_sum > 0.0 {
                    acc / weight_sum
                } else {
                    center
                };
            }
        }
        output
    }
}

impl Default for BilateralFilter {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// Median Filter
// ============================================================

/// Median filter for impulse noise removal.
pub struct MedianFilter;

impl MedianFilter {
    /// Create a new `MedianFilter`.
    pub fn new() -> Self {
        Self
    }

    /// Apply a 3×3 median filter.
    pub fn filter_3x3(&self, frame: &[f32], width: usize, height: usize) -> Vec<f32> {
        apply_median_filter(frame, width, height, 1)
    }

    /// Apply a 5×5 median filter.
    pub fn filter_5x5(&self, frame: &[f32], width: usize, height: usize) -> Vec<f32> {
        apply_median_filter(frame, width, height, 2)
    }
}

impl Default for MedianFilter {
    fn default() -> Self {
        Self::new()
    }
}

/// Apply a median filter with a square window of radius `r` (window = (2r+1)²).
fn apply_median_filter(frame: &[f32], width: usize, height: usize, r: usize) -> Vec<f32> {
    if frame.is_empty() || width == 0 || height == 0 {
        return frame.to_vec();
    }
    let mut output = vec![0.0f32; width * height];
    let r = r as i64;

    for y in 0..height {
        for x in 0..width {
            let mut window: Vec<f32> = Vec::with_capacity(((2 * r + 1) * (2 * r + 1)) as usize);
            for dy in -r..=r {
                for dx in -r..=r {
                    let ny = (y as i64 + dy).clamp(0, height as i64 - 1) as usize;
                    let nx = (x as i64 + dx).clamp(0, width as i64 - 1) as usize;
                    window.push(frame[ny * width + nx]);
                }
            }
            window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            output[y * width + x] = window[window.len() / 2];
        }
    }
    output
}

// ============================================================
// Temporal Averager
// ============================================================

/// Temporal frame averager using a ring buffer of `n` frames.
pub struct TemporalAverager {
    buffer: VecDeque<Vec<f32>>,
    capacity: usize,
}

impl TemporalAverager {
    /// Create a new `TemporalAverager` with the given ring buffer size.
    pub fn new(n: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(n.max(1)),
            capacity: n.max(1),
        }
    }

    /// Add a frame to the ring buffer.
    pub fn add_frame(&mut self, frame: Vec<f32>) {
        if self.buffer.len() == self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(frame);
    }

    /// Compute the pixel-wise average of all buffered frames.
    ///
    /// Returns an empty vector if the buffer is empty.
    pub fn get_averaged(&self) -> Vec<f32> {
        if self.buffer.is_empty() {
            return Vec::new();
        }

        let len = self.buffer[0].len();
        let count = self.buffer.len() as f32;
        let mut averaged = vec![0.0f32; len];

        for frame in &self.buffer {
            let frame_len = frame.len().min(len);
            for (a, &f) in averaged[..frame_len]
                .iter_mut()
                .zip(frame[..frame_len].iter())
            {
                *a += f;
            }
        }

        for a in &mut averaged {
            *a /= count;
        }
        averaged
    }

    /// Number of frames currently in the buffer.
    pub fn frame_count(&self) -> usize {
        self.buffer.len()
    }
}

// ============================================================
// Noise Estimator
// ============================================================

/// Frame-level noise sigma estimator.
pub struct NoiseEstimator;

impl NoiseEstimator {
    /// Create a new `NoiseEstimator`.
    pub fn new() -> Self {
        Self
    }

    /// Estimate noise sigma using the MAD (Median Absolute Deviation) estimator
    /// applied to the high-pass filtered frame.
    ///
    /// Uses a simple horizontal first-difference as the high-pass filter:
    /// d\[x\] = frame\[x+1\] - frame\[x\].
    ///
    /// Sigma = MAD / 0.6745
    pub fn estimate_sigma(frame: &[f32]) -> f32 {
        if frame.len() < 2 {
            return 0.0;
        }

        // Compute horizontal differences (first-order high-pass)
        let mut diffs: Vec<f32> = frame.windows(2).map(|w| (w[1] - w[0]).abs()).collect();

        // MAD
        diffs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = diffs[diffs.len() / 2];

        // Sigma estimate for Gaussian noise
        median / 0.6745
    }
}

impl Default for NoiseEstimator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_checkerboard(width: usize, height: usize) -> Vec<f32> {
        (0..width * height)
            .map(|i| {
                let x = i % width;
                let y = i / width;
                if (x + y) % 2 == 0 {
                    1.0
                } else {
                    0.0
                }
            })
            .collect()
    }

    #[test]
    fn test_gaussian_denoise_output_size() {
        let frame = vec![0.5f32; 64 * 64];
        let out = SpatialDenoiser::new().gaussian_denoise(&frame, 64, 64, 1.0);
        assert_eq!(out.len(), 64 * 64);
    }

    #[test]
    fn test_gaussian_denoise_constant_frame() {
        // Gaussian blur of a constant image should be the same constant
        let frame = vec![0.7f32; 32 * 32];
        let out = SpatialDenoiser::new().gaussian_denoise(&frame, 32, 32, 1.5);
        for &v in &out {
            assert!(
                (v - 0.7).abs() < 1e-4,
                "Constant frame should stay constant: {v}"
            );
        }
    }

    #[test]
    fn test_gaussian_denoise_smooths_checkerboard() {
        let frame = make_checkerboard(16, 16);
        let out = SpatialDenoiser::new().gaussian_denoise(&frame, 16, 16, 1.5);
        // Smoothed checkerboard should be less extreme than 0 and 1
        let max = out.iter().cloned().fold(0.0_f32, f32::max);
        let min = out.iter().cloned().fold(1.0_f32, f32::min);
        assert!(max < 1.0, "Smoothed max should be < 1.0: {max}");
        assert!(min > 0.0, "Smoothed min should be > 0.0: {min}");
    }

    #[test]
    fn test_bilateral_output_size() {
        let frame = vec![0.5f32; 32 * 32];
        let out = BilateralFilter::new().filter(&frame, 32, 32, 3.0, 0.1);
        assert_eq!(out.len(), 32 * 32);
    }

    #[test]
    fn test_bilateral_constant_frame() {
        let frame = vec![0.4f32; 16 * 16];
        let out = BilateralFilter::new().filter(&frame, 16, 16, 2.0, 0.5);
        for &v in &out {
            assert!((v - 0.4).abs() < 1e-4, "Constant frame: got {v}");
        }
    }

    #[test]
    fn test_median_3x3_output_size() {
        let frame = vec![0.5f32; 32 * 32];
        let out = MedianFilter::new().filter_3x3(&frame, 32, 32);
        assert_eq!(out.len(), 32 * 32);
    }

    #[test]
    fn test_median_5x5_output_size() {
        let frame = vec![0.5f32; 32 * 32];
        let out = MedianFilter::new().filter_5x5(&frame, 32, 32);
        assert_eq!(out.len(), 32 * 32);
    }

    #[test]
    fn test_median_removes_impulse() {
        let mut frame = vec![0.5f32; 16 * 16];
        // Single impulse in the middle
        frame[8 * 16 + 8] = 1.0;
        let out = MedianFilter::new().filter_3x3(&frame, 16, 16);
        // The impulse should be suppressed
        assert!(
            out[8 * 16 + 8] < 0.9,
            "Median should suppress impulse: got {}",
            out[8 * 16 + 8]
        );
    }

    #[test]
    fn test_temporal_averager_empty() {
        let ta = TemporalAverager::new(5);
        assert!(ta.get_averaged().is_empty());
    }

    #[test]
    fn test_temporal_averager_single_frame() {
        let mut ta = TemporalAverager::new(5);
        let frame = vec![0.3f32, 0.6f32, 0.9f32];
        ta.add_frame(frame.clone());
        let avg = ta.get_averaged();
        assert_eq!(avg.len(), 3);
        for (a, &f) in avg.iter().zip(frame.iter()) {
            assert!((a - f).abs() < 1e-6);
        }
    }

    #[test]
    fn test_temporal_averager_average() {
        let mut ta = TemporalAverager::new(3);
        ta.add_frame(vec![0.0f32, 1.0]);
        ta.add_frame(vec![1.0f32, 0.0]);
        let avg = ta.get_averaged();
        assert!((avg[0] - 0.5).abs() < 1e-6);
        assert!((avg[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_temporal_averager_ring_buffer() {
        let mut ta = TemporalAverager::new(2);
        ta.add_frame(vec![0.0f32]);
        ta.add_frame(vec![1.0f32]);
        ta.add_frame(vec![2.0f32]); // Evicts first frame
        assert_eq!(ta.frame_count(), 2);
        let avg = ta.get_averaged();
        // Average of 1.0 and 2.0 = 1.5
        assert!((avg[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_noise_estimator_constant() {
        // Constant signal has no differences, so sigma = 0
        let frame = vec![0.5f32; 100];
        let sigma = NoiseEstimator::estimate_sigma(&frame);
        assert!((sigma - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_noise_estimator_white_noise() {
        // White noise should give positive sigma estimate
        let frame: Vec<f32> = (0..1000)
            .map(|i| ((i * 1234567) % 1000) as f32 / 1000.0 - 0.5)
            .collect();
        let sigma = NoiseEstimator::estimate_sigma(&frame);
        assert!(
            sigma > 0.0,
            "White noise should have positive sigma: {sigma}"
        );
    }
}
