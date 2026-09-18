//! SIMD-friendly image filtering operations.
//!
//! Provides box-blur (separable horizontal + vertical passes), Gaussian
//! weight generation, 1-D kernel convolution, and 3×3 convolution kernel
//! application. All inner loops are written to encourage auto-vectorisation.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(dead_code)]

use std::f32::consts::PI;

// ── Box blur (separable) ──────────────────────────────────────────────────────

/// Horizontal box-blur pass.
///
/// Operates on a single-channel (grayscale) buffer stored in row-major order.
/// Each pixel is replaced by the average of itself and `radius` neighbours on
/// each side, clamped to the row boundary.
///
/// `src` and `dst` must both have length `width * height`.
pub fn box_blur_horizontal(src: &[u8], dst: &mut [u8], width: usize, height: usize, radius: usize) {
    if width == 0 || height == 0 {
        return;
    }
    let diameter = 2 * radius + 1;
    for row in 0..height {
        let row_start = row * width;
        let mut sum: u32 = 0;
        // Seed with the leftmost pixel repeated for the left half of the window.
        for _ in 0..=radius {
            sum += u32::from(src[row_start]);
        }
        for col in 0..radius {
            let right_idx = col + radius;
            if right_idx < width {
                sum += u32::from(src[row_start + right_idx]);
            } else {
                sum += u32::from(src[row_start + width - 1]);
            }
        }
        for col in 0..width {
            dst[row_start + col] = (sum / diameter as u32) as u8;

            // Slide the window: remove leftmost, add next rightmost.
            let left_idx = col.saturating_sub(radius);
            sum = sum.saturating_sub(u32::from(src[row_start + left_idx]));

            let right_idx = col + radius + 1;
            if right_idx < width {
                sum += u32::from(src[row_start + right_idx]);
            } else {
                sum += u32::from(src[row_start + width - 1]);
            }
        }
    }
}

/// Vertical box-blur pass.
///
/// Operates on a single-channel (grayscale) buffer stored in row-major order.
/// Each pixel is replaced by the average of itself and `radius` neighbours
/// above and below, clamped to the column boundary.
///
/// `src` and `dst` must both have length `width * height`.
pub fn box_blur_vertical(src: &[u8], dst: &mut [u8], width: usize, height: usize, radius: usize) {
    if width == 0 || height == 0 {
        return;
    }
    let diameter = 2 * radius + 1;
    for col in 0..width {
        let mut sum: u32 = 0;
        for _ in 0..=radius {
            sum += u32::from(src[col]); // top row repeated
        }
        for row in 0..radius {
            let below = if row + radius < height {
                row + radius
            } else {
                height - 1
            };
            sum += u32::from(src[below * width + col]);
        }
        for row in 0..height {
            dst[row * width + col] = (sum / diameter as u32) as u8;

            let top = row.saturating_sub(radius);
            sum = sum.saturating_sub(u32::from(src[top * width + col]));

            let bottom = row + radius + 1;
            if bottom < height {
                sum += u32::from(src[bottom * width + col]);
            } else {
                sum += u32::from(src[(height - 1) * width + col]);
            }
        }
    }
}

// ── Gaussian weights ─────────────────────────────────────────────────────────

/// Compute normalised 1-D Gaussian kernel weights.
///
/// Returns a vector of `2 * radius + 1` weights that sum to `1.0`, using
/// the Gaussian formula `exp(-x^2 / (2 σ²)) / (σ √(2π))`.
///
/// If `sigma` is zero or negative, a degenerate kernel with a single `1.0`
/// weight (radius 0) is returned.
#[must_use]
pub fn gaussian_weights(sigma: f32, radius: usize) -> Vec<f32> {
    if sigma <= 0.0 {
        return vec![1.0];
    }
    let size = 2 * radius + 1;
    let r = radius as i32;
    let two_sigma_sq = 2.0 * sigma * sigma;
    let norm = 1.0 / ((2.0 * PI).sqrt() * sigma);

    let mut weights: Vec<f32> = (0..size)
        .map(|i| {
            let x = (i as i32 - r) as f32;
            norm * (-x * x / two_sigma_sq).exp()
        })
        .collect();

    // Normalise so weights sum to exactly 1.0.
    let total: f32 = weights.iter().sum();
    if total > 1e-8 {
        for w in &mut weights {
            *w /= total;
        }
    }
    weights
}

// ── 1-D filter convolution ────────────────────────────────────────────────────

/// Convolve a 1-D signal `src` with a `kernel` and return the result.
///
/// Boundary pixels are handled by clamping the sample index to the valid
/// range (replicating the edge value). The output length equals `src.len()`.
#[must_use]
pub fn apply_1d_filter(src: &[f32], kernel: &[f32]) -> Vec<f32> {
    if src.is_empty() || kernel.is_empty() {
        return src.to_vec();
    }
    let half = (kernel.len() / 2) as i64;
    let n = src.len() as i64;
    (0..src.len())
        .map(|i| {
            let i = i as i64;
            kernel.iter().enumerate().fold(0.0f32, |acc, (k, &w)| {
                let j = (i + k as i64 - half).clamp(0, n - 1) as usize;
                acc + src[j] * w
            })
        })
        .collect()
}

// ── 3×3 convolution kernel ────────────────────────────────────────────────────

/// Return a standard 3×3 sharpening kernel (Laplacian-based).
#[must_use]
pub const fn sharpen_kernel() -> [f32; 9] {
    [0.0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0]
}

/// Apply a 3×3 convolution kernel to a single-channel grayscale image.
///
/// `src` and `dst` must both have length `width * height`. Border pixels are
/// handled by clamping coordinates to the valid range.
pub fn apply_3x3_kernel(src: &[u8], dst: &mut [u8], width: usize, height: usize, kernel: [f32; 9]) {
    if width == 0 || height == 0 {
        return;
    }
    let w = width as i64;
    let h = height as i64;
    for row in 0..height {
        for col in 0..width {
            let mut acc = 0.0f32;
            for ky in 0..3i64 {
                for kx in 0..3i64 {
                    let sy = (row as i64 + ky - 1).clamp(0, h - 1) as usize;
                    let sx = (col as i64 + kx - 1).clamp(0, w - 1) as usize;
                    acc += f32::from(src[sy * width + sx]) * kernel[(ky * 3 + kx) as usize];
                }
            }
            dst[row * width + col] = acc.round().clamp(0.0, 255.0) as u8;
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_box_blur_horizontal_uniform() {
        // A uniform row should be unchanged after blurring.
        let src = vec![128u8; 10];
        let mut dst = vec![0u8; 10];
        box_blur_horizontal(&src, &mut dst, 10, 1, 2);
        for &v in &dst {
            assert!((i32::from(v) - 128).abs() <= 1);
        }
    }

    #[test]
    fn test_box_blur_horizontal_preserves_length() {
        let src = vec![100u8; 20];
        let mut dst = vec![0u8; 20];
        box_blur_horizontal(&src, &mut dst, 10, 2, 1);
        assert_eq!(dst.len(), 20);
    }

    #[test]
    fn test_box_blur_vertical_uniform() {
        let src = vec![200u8; 25];
        let mut dst = vec![0u8; 25];
        box_blur_vertical(&src, &mut dst, 5, 5, 1);
        for &v in &dst {
            assert!((i32::from(v) - 200).abs() <= 1);
        }
    }

    #[test]
    fn test_box_blur_empty_no_panic() {
        let src: Vec<u8> = vec![];
        let mut dst: Vec<u8> = vec![];
        box_blur_horizontal(&src, &mut dst, 0, 0, 1);
        box_blur_vertical(&src, &mut dst, 0, 0, 1);
    }

    #[test]
    fn test_gaussian_weights_sum_to_one() {
        let w = gaussian_weights(1.0, 3);
        let total: f32 = w.iter().sum();
        assert!((total - 1.0).abs() < 1e-5, "sum = {total}");
    }

    #[test]
    fn test_gaussian_weights_symmetric() {
        let w = gaussian_weights(2.0, 4);
        let n = w.len();
        for i in 0..n / 2 {
            assert!((w[i] - w[n - 1 - i]).abs() < 1e-6, "asymmetric at {i}");
        }
    }

    #[test]
    fn test_gaussian_weights_zero_sigma_returns_one() {
        let w = gaussian_weights(0.0, 3);
        assert_eq!(w.len(), 1);
        assert!((w[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_gaussian_weights_center_is_largest() {
        let w = gaussian_weights(1.5, 3);
        let center = w.len() / 2;
        for (i, &v) in w.iter().enumerate() {
            assert!(v <= w[center] + 1e-6, "value at {i} exceeds center");
        }
    }

    #[test]
    fn test_apply_1d_filter_identity_kernel() {
        let src = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let kernel = vec![0.0, 1.0, 0.0]; // delta
        let out = apply_1d_filter(&src, &kernel);
        for (a, b) in src.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn test_apply_1d_filter_mean_kernel() {
        let src = vec![0.0f32, 0.0, 9.0, 0.0, 0.0];
        let kernel = vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
        let out = apply_1d_filter(&src, &kernel);
        // The peak should be spread across the centre three output pixels.
        assert!((out[2] - 3.0).abs() < 0.5, "center={}", out[2]);
    }

    #[test]
    fn test_apply_1d_filter_empty_src() {
        let src: Vec<f32> = vec![];
        let kernel = vec![0.5, 0.5];
        let out = apply_1d_filter(&src, &kernel);
        assert!(out.is_empty());
    }

    #[test]
    fn test_sharpen_kernel_sum() {
        let k = sharpen_kernel();
        let sum: f32 = k.iter().sum();
        // The sharpening kernel sums to 1 (identity preserving).
        assert!((sum - 1.0).abs() < 1e-5, "kernel sum = {sum}");
    }

    #[test]
    fn test_apply_3x3_kernel_uniform_identity() {
        // On a uniform image the identity kernel should be a no-op.
        let identity: [f32; 9] = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let src = vec![128u8; 9];
        let mut dst = vec![0u8; 9];
        apply_3x3_kernel(&src, &mut dst, 3, 3, identity);
        for &v in &dst {
            assert_eq!(v, 128);
        }
    }

    #[test]
    fn test_apply_3x3_kernel_empty_no_panic() {
        let src: Vec<u8> = vec![];
        let mut dst: Vec<u8> = vec![];
        apply_3x3_kernel(&src, &mut dst, 0, 0, sharpen_kernel());
    }
}
