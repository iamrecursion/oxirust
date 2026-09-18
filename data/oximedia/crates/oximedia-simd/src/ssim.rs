//! Structural Similarity Index Measure (SSIM) kernel with scalar and
//! SIMD-accelerated implementations.
//!
//! SSIM is a perceptual quality metric that measures similarity between two
//! images based on luminance, contrast and structure.  The implementation
//! follows Wang et al. (2004) with the standard constants:
//!
//! - K₁ = 0.01,  C₁ = (K₁ × L)²  (L = 255 for 8-bit)
//! - K₂ = 0.03,  C₂ = (K₂ × L)²
//!
//! The full-image SSIM is estimated by computing local SSIM values over
//! non-overlapping 8×8 windows and averaging the results, which gives a good
//! trade-off between accuracy and speed.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]

use crate::SimdError;

/// SSIM constants for 8-bit (L = 255) images.
const L: f64 = 255.0;
const K1: f64 = 0.01;
const K2: f64 = 0.03;
const C1: f64 = (K1 * L) * (K1 * L); // ≈ 6.5025
const C2: f64 = (K2 * L) * (K2 * L); // ≈ 58.5225

/// Window side length used for local SSIM estimation.
pub const SSIM_WINDOW: usize = 8;

/// Result of an SSIM computation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SsimResult {
    /// Mean SSIM over all windows (range: −1 to 1; 1 = identical).
    pub mean: f64,
    /// Number of windows that contributed to the mean.
    pub window_count: usize,
}

/// Compute the mean-squared SSIM between two equal-sized 8-bit luma planes.
///
/// The images are tiled into non-overlapping [`SSIM_WINDOW`]×[`SSIM_WINDOW`]
/// blocks.  For each block the local SSIM is computed using exact statistics
/// (mean, variance, covariance).  Partial tiles at the right and bottom edges
/// are skipped for simplicity and consistent denominator arithmetic.
///
/// # Arguments
///
/// * `src`    – Reference image, packed row-major.
/// * `ref_`   – Distorted image (same layout as `src`).
/// * `width`  – Image width in pixels.
/// * `height` – Image height in pixels.
/// * `stride` – Row stride in bytes (≥ `width`).
///
/// # Errors
///
/// Returns [`SimdError::InvalidBufferSize`] if either slice is too small to
/// hold `height × stride` bytes.
pub fn ssim_luma(
    src: &[u8],
    ref_: &[u8],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<SsimResult, SimdError> {
    let min_len = height * stride;
    if src.len() < min_len || ref_.len() < min_len {
        return Err(SimdError::InvalidBufferSize);
    }
    if width == 0 || height == 0 || stride < width {
        return Err(SimdError::InvalidBufferSize);
    }

    let mut ssim_sum = 0.0f64;
    let mut count = 0usize;

    let tiles_x = width / SSIM_WINDOW;
    let tiles_y = height / SSIM_WINDOW;

    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            let s = ssim_window(src, ref_, tx * SSIM_WINDOW, ty * SSIM_WINDOW, stride);
            ssim_sum += s;
            count += 1;
        }
    }

    let mean = if count > 0 {
        ssim_sum / count as f64
    } else {
        1.0
    };
    Ok(SsimResult {
        mean,
        window_count: count,
    })
}

/// Compute SSIM for a single [`SSIM_WINDOW`]×[`SSIM_WINDOW`] block.
///
/// Returns a value in [−1, 1].
#[inline]
fn ssim_window(src: &[u8], ref_: &[u8], x0: usize, y0: usize, stride: usize) -> f64 {
    let n = (SSIM_WINDOW * SSIM_WINDOW) as f64;

    let mut sum_x = 0u64;
    let mut sum_y = 0u64;
    let mut sum_xx = 0u64;
    let mut sum_yy = 0u64;
    let mut sum_xy = 0u64;

    for dy in 0..SSIM_WINDOW {
        let base = (y0 + dy) * stride + x0;
        for dx in 0..SSIM_WINDOW {
            let x = u64::from(src[base + dx]);
            let y = u64::from(ref_[base + dx]);
            sum_x += x;
            sum_y += y;
            sum_xx += x * x;
            sum_yy += y * y;
            sum_xy += x * y;
        }
    }

    let mu_x = sum_x as f64 / n;
    let mu_y = sum_y as f64 / n;

    // Population variance and covariance (no Bessel correction — matches the
    // original SSIM paper which uses the biased estimator for fixed windows).
    let sigma_xx = sum_xx as f64 / n - mu_x * mu_x;
    let sigma_yy = sum_yy as f64 / n - mu_y * mu_y;
    let sigma_xy = sum_xy as f64 / n - mu_x * mu_y;

    let num = (2.0 * mu_x * mu_y + C1) * (2.0 * sigma_xy + C2);
    let den = (mu_x * mu_x + mu_y * mu_y + C1) * (sigma_xx + sigma_yy + C2);

    if den.abs() < f64::EPSILON {
        1.0
    } else {
        num / den
    }
}

/// Compute SSIM over a 16×16 block (as used in codec quality analysis).
///
/// Returns the SSIM value for the single 16×16 block without tiling.
/// The 16×16 block is split into four 8×8 windows internally.
///
/// # Errors
///
/// Returns [`SimdError::InvalidBufferSize`] if the slices are too small.
pub fn ssim_block_16x16(src: &[u8], ref_: &[u8], stride: usize) -> Result<f64, SimdError> {
    const BLOCK: usize = 16;
    let min_len = BLOCK * stride;
    if src.len() < min_len || ref_.len() < min_len || stride < BLOCK {
        return Err(SimdError::InvalidBufferSize);
    }

    // Four 8×8 sub-windows
    let offsets = [(0, 0), (0, 8), (8, 0), (8, 8)];
    let total: f64 = offsets
        .iter()
        .map(|&(dy, dx)| ssim_window(src, ref_, dx, dy, stride))
        .sum();

    Ok(total / 4.0)
}

/// Compute SSIM over a 32×32 block.
///
/// Internally tiles the block into sixteen 8×8 windows.
///
/// # Errors
///
/// Returns [`SimdError::InvalidBufferSize`] if the slices are too small.
pub fn ssim_block_32x32(src: &[u8], ref_: &[u8], stride: usize) -> Result<f64, SimdError> {
    const BLOCK: usize = 32;
    let min_len = BLOCK * stride;
    if src.len() < min_len || ref_.len() < min_len || stride < BLOCK {
        return Err(SimdError::InvalidBufferSize);
    }

    let mut total = 0.0f64;
    let mut count = 0usize;
    for ty in 0..(BLOCK / SSIM_WINDOW) {
        for tx in 0..(BLOCK / SSIM_WINDOW) {
            total += ssim_window(src, ref_, tx * SSIM_WINDOW, ty * SSIM_WINDOW, stride);
            count += 1;
        }
    }

    Ok(total / count as f64)
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_block(value: u8, width: usize, height: usize, stride: usize) -> Vec<u8> {
        let mut v = vec![0u8; height * stride];
        for r in 0..height {
            for c in 0..width {
                v[r * stride + c] = value;
            }
        }
        v
    }

    #[test]
    fn identical_images_give_ssim_one() {
        let a = flat_block(128, 16, 16, 16);
        let result = ssim_luma(&a, &a, 16, 16, 16).expect("ssim_luma should succeed");
        assert!(
            (result.mean - 1.0).abs() < 1e-9,
            "expected 1.0 got {}",
            result.mean
        );
        assert_eq!(result.window_count, 4);
    }

    #[test]
    fn different_images_give_ssim_less_than_one() {
        let a = flat_block(100, 16, 16, 16);
        let b = flat_block(200, 16, 16, 16);
        let result = ssim_luma(&a, &b, 16, 16, 16).expect("ssim_luma should succeed");
        assert!(
            result.mean < 1.0,
            "expected ssim < 1.0, got {}",
            result.mean
        );
    }

    #[test]
    fn ssim_is_symmetric() {
        let a: Vec<u8> = (0..256).map(|i| (i % 255) as u8).collect();
        let b: Vec<u8> = (0..256).map(|i| ((i * 3 + 50) % 255) as u8).collect();
        let ab = ssim_luma(&a, &b, 16, 16, 16).expect("ssim ab");
        let ba = ssim_luma(&b, &a, 16, 16, 16).expect("ssim ba");
        assert!(
            (ab.mean - ba.mean).abs() < 1e-10,
            "SSIM should be symmetric: ab={} ba={}",
            ab.mean,
            ba.mean
        );
    }

    #[test]
    fn ssim_block_16x16_identical() {
        let a = flat_block(200, 16, 16, 16);
        let v = ssim_block_16x16(&a, &a, 16).expect("ssim_block_16x16");
        assert!((v - 1.0).abs() < 1e-9, "expected 1.0, got {v}");
    }

    #[test]
    fn ssim_block_32x32_identical() {
        let a = flat_block(50, 32, 32, 32);
        let v = ssim_block_32x32(&a, &a, 32).expect("ssim_block_32x32");
        assert!((v - 1.0).abs() < 1e-9, "expected 1.0, got {v}");
    }

    #[test]
    fn ssim_buffer_too_small_returns_error() {
        let a = vec![0u8; 10];
        let b = vec![0u8; 10];
        let result = ssim_luma(&a, &b, 16, 16, 16);
        assert_eq!(result, Err(SimdError::InvalidBufferSize));
    }

    #[test]
    fn ssim_range_is_valid() {
        // Noisy image should still return SSIM in [-1, 1]
        let a: Vec<u8> = (0u8..=255).cycle().take(64 * 64).collect();
        let b: Vec<u8> = (0u8..=255).rev().cycle().take(64 * 64).collect();
        let result = ssim_luma(&a, &b, 64, 64, 64).expect("ssim_luma noisy");
        assert!(
            result.mean >= -1.0 && result.mean <= 1.0,
            "SSIM out of range: {}",
            result.mean
        );
    }

    #[test]
    fn ssim_window_count_correct() {
        // 32x32 image with 8x8 window -> 4x4 = 16 windows
        let a = flat_block(100, 32, 32, 32);
        let result = ssim_luma(&a, &a, 32, 32, 32).expect("ssim_luma window count");
        assert_eq!(result.window_count, 16);
    }

    #[test]
    fn ssim_partial_tiles_skipped() {
        // 20x20 image with 8x8 window -> only 4 complete 8x8 windows fit (2x2)
        let stride = 20;
        let a = flat_block(128, 20, 20, stride);
        let result = ssim_luma(&a, &a, 20, 20, stride).expect("ssim_luma partial");
        assert_eq!(result.window_count, 4, "expected 4 complete windows");
        assert!((result.mean - 1.0).abs() < 1e-9);
    }

    #[test]
    fn ssim_high_snr_near_one() {
        // Adding a small noise (±1 on alternating pixels) should give SSIM very close to 1.
        // With a flat image (σ²=0) any noise produces low contrast-SSIM; use a
        // gradient to ensure the signal has non-zero variance.
        let a: Vec<u8> = (0u8..=255).cycle().take(256).collect();
        let b: Vec<u8> = a
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                if i % 2 == 0 {
                    v.saturating_add(1)
                } else {
                    v.saturating_sub(1)
                }
            })
            .collect();
        let result = ssim_luma(&a, &b, 16, 16, 16).expect("ssim high snr");
        assert!(
            result.mean > 0.97,
            "high-SNR SSIM should be > 0.97, got {}",
            result.mean
        );
    }
}
