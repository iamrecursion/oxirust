//! Peak Signal-to-Noise Ratio (PSNR) computation kernel.
//!
//! PSNR is the most widely used objective video-quality metric.  For 8-bit
//! content the maximum signal value is L = 255, giving:
//!
//! ```text
//! MSE  = (1 / N) Σ (x_i − y_i)²
//! PSNR = 10 · log₁₀(L² / MSE)
//! ```
//!
//! When `MSE == 0` (identical images) the function returns [`f64::INFINITY`].
//!
//! The module provides:
//! - [`psnr_u8`]         – full-image PSNR over flat slices.
//! - [`psnr_block_16x16`] – PSNR for a single 16×16 luma block.
//! - [`psnr_block_32x32`] – PSNR for a single 32×32 luma block.
//! - [`mse_u8`]           – raw Mean Squared Error (integer-exact).

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use crate::SimdError;

/// Maximum signal level for 8-bit samples.
const MAX_SIGNAL: f64 = 255.0;
/// Max-signal² used in the PSNR formula.
const MAX_SIGNAL_SQ: f64 = MAX_SIGNAL * MAX_SIGNAL; // 65025.0

/// Compute the Mean Squared Error between two equal-length u8 slices.
///
/// Uses 64-bit integer accumulation to avoid overflow on large images.
///
/// # Errors
///
/// Returns [`SimdError::InvalidBufferSize`] if the slices have different
/// lengths or are empty.
#[inline]
pub fn mse_u8(a: &[u8], b: &[u8]) -> Result<f64, SimdError> {
    if a.is_empty() || a.len() != b.len() {
        return Err(SimdError::InvalidBufferSize);
    }

    let sum_sq: u64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let diff = i32::from(x) - i32::from(y);
            (diff * diff) as u64
        })
        .sum();

    Ok(sum_sq as f64 / a.len() as f64)
}

/// Compute PSNR (dB) between two equal-length u8 slices.
///
/// Returns [`f64::INFINITY`] when the images are identical (MSE = 0).
///
/// # Errors
///
/// Returns [`SimdError::InvalidBufferSize`] if the slices have different
/// lengths or are empty.
pub fn psnr_u8(a: &[u8], b: &[u8]) -> Result<f64, SimdError> {
    let mse = mse_u8(a, b)?;
    if mse == 0.0 {
        return Ok(f64::INFINITY);
    }
    Ok(10.0 * (MAX_SIGNAL_SQ / mse).log10())
}

/// Compute PSNR for a single 16×16 luma block.
///
/// Both `src` and `ref_` must contain at least `16 × stride` bytes and
/// `stride` must be ≥ 16.
///
/// # Errors
///
/// Returns [`SimdError::InvalidBufferSize`] if the buffer sizes are invalid.
pub fn psnr_block_16x16(src: &[u8], ref_: &[u8], stride: usize) -> Result<f64, SimdError> {
    psnr_block_strided(src, ref_, 16, 16, stride)
}

/// Compute PSNR for a single 32×32 luma block.
///
/// Both `src` and `ref_` must contain at least `32 × stride` bytes and
/// `stride` must be ≥ 32.
///
/// # Errors
///
/// Returns [`SimdError::InvalidBufferSize`] if the buffer sizes are invalid.
pub fn psnr_block_32x32(src: &[u8], ref_: &[u8], stride: usize) -> Result<f64, SimdError> {
    psnr_block_strided(src, ref_, 32, 32, stride)
}

/// Generic strided PSNR for a `width × height` block.
///
/// Pixels outside the `width` columns in each row (i.e. the padding bytes
/// between `width` and `stride`) are excluded from the metric.
///
/// # Errors
///
/// Returns [`SimdError::InvalidBufferSize`] if the slices are too small.
pub fn psnr_block_strided(
    src: &[u8],
    ref_: &[u8],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<f64, SimdError> {
    if width == 0 || height == 0 || stride < width {
        return Err(SimdError::InvalidBufferSize);
    }
    let min_len = height * stride;
    if src.len() < min_len || ref_.len() < min_len {
        return Err(SimdError::InvalidBufferSize);
    }

    let mut sum_sq: u64 = 0;
    for row in 0..height {
        let base = row * stride;
        for col in 0..width {
            let diff = i32::from(src[base + col]) - i32::from(ref_[base + col]);
            sum_sq += (diff * diff) as u64;
        }
    }

    let n = (width * height) as f64;
    let mse = sum_sq as f64 / n;

    if mse == 0.0 {
        return Ok(f64::INFINITY);
    }
    Ok(10.0 * (MAX_SIGNAL_SQ / mse).log10())
}

/// Compute per-component PSNR for a YCbCr image.
///
/// Returns `(psnr_y, psnr_cb, psnr_cr)`.  Each plane is provided as a flat
/// slice with its own stride.  All planes must have the same `width` and
/// `height` (i.e. 4:4:4 sampling is assumed here; for other sampling ratios
/// callers should pass appropriately sized planes).
///
/// # Errors
///
/// Returns [`SimdError::InvalidBufferSize`] if any plane slice is too small.
pub fn psnr_yuv(
    src_y: &[u8],
    src_cb: &[u8],
    src_cr: &[u8],
    ref_y: &[u8],
    ref_cb: &[u8],
    ref_cr: &[u8],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<(f64, f64, f64), SimdError> {
    let py = psnr_block_strided(src_y, ref_y, width, height, stride)?;
    let pcb = psnr_block_strided(src_cb, ref_cb, width, height, stride)?;
    let pcr = psnr_block_strided(src_cr, ref_cr, width, height, stride)?;
    Ok((py, pcb, pcr))
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_images_give_infinity() {
        let a = vec![128u8; 256];
        assert_eq!(psnr_u8(&a, &a), Ok(f64::INFINITY));
    }

    #[test]
    fn known_mse_gives_correct_psnr() {
        // a = [0, 0, ...], b = [10, 10, ...]  → MSE = 100 → PSNR = 10*log10(65025/100) ≈ 28.13
        let a = vec![0u8; 100];
        let b = vec![10u8; 100];
        let psnr = psnr_u8(&a, &b).expect("psnr_u8 should succeed");
        let expected = 10.0 * (65025.0f64 / 100.0).log10();
        assert!(
            (psnr - expected).abs() < 1e-6,
            "psnr={psnr} expected={expected}"
        );
    }

    #[test]
    fn mse_identical() {
        let a = vec![50u8; 64];
        let mse = mse_u8(&a, &a).expect("mse_u8");
        assert_eq!(mse, 0.0);
    }

    #[test]
    fn mse_known_value() {
        // 4 pixels, diff = 1 each → MSE = 4/4 = 1
        let a = vec![10u8, 10, 10, 10];
        let b = vec![11u8, 11, 11, 11];
        let mse = mse_u8(&a, &b).expect("mse_u8");
        assert!((mse - 1.0).abs() < 1e-10);
    }

    #[test]
    fn empty_slice_returns_error() {
        let result = psnr_u8(&[], &[]);
        assert_eq!(result, Err(SimdError::InvalidBufferSize));
    }

    #[test]
    fn mismatched_lengths_return_error() {
        let a = vec![0u8; 4];
        let b = vec![0u8; 8];
        assert_eq!(psnr_u8(&a, &b), Err(SimdError::InvalidBufferSize));
    }

    #[test]
    fn psnr_block_16x16_identical() {
        let a = vec![200u8; 16 * 16];
        let v = psnr_block_16x16(&a, &a, 16).expect("psnr_block_16x16");
        assert_eq!(v, f64::INFINITY);
    }

    #[test]
    fn psnr_block_32x32_known() {
        // All-zero vs all-1: MSE = 1 → PSNR = 10*log10(65025)
        let a = vec![0u8; 32 * 32];
        let b = vec![1u8; 32 * 32];
        let psnr = psnr_block_32x32(&a, &b, 32).expect("psnr_block_32x32");
        let expected = 10.0 * 65025.0f64.log10();
        assert!((psnr - expected).abs() < 1e-6);
    }

    #[test]
    fn psnr_decreases_with_more_noise() {
        let ref_ = vec![128u8; 256];
        let low_noise: Vec<u8> = ref_.iter().map(|&x| x.saturating_add(5)).collect();
        let high_noise: Vec<u8> = ref_.iter().map(|&x| x.saturating_add(50)).collect();
        let psnr_low = psnr_u8(&ref_, &low_noise).expect("low noise psnr");
        let psnr_high = psnr_u8(&ref_, &high_noise).expect("high noise psnr");
        assert!(
            psnr_low > psnr_high,
            "low-noise PSNR ({psnr_low}) should exceed high-noise PSNR ({psnr_high})"
        );
    }

    #[test]
    fn psnr_strided_excludes_padding() {
        // 4×4 block with stride=8 (4 padding bytes per row)
        // Fill active pixels with 0, padding with 255.
        // Reference block all-0. PSNR should be infinity.
        let mut src = vec![255u8; 4 * 8]; // all 255 initially (incl. padding)
        for row in 0..4 {
            for col in 0..4 {
                src[row * 8 + col] = 0;
            }
        }
        let ref_ = vec![0u8; 4 * 8];
        let psnr = psnr_block_strided(&src, &ref_, 4, 4, 8).expect("psnr strided");
        assert_eq!(psnr, f64::INFINITY, "padding bytes must not affect PSNR");
    }

    #[test]
    fn psnr_yuv_all_channels_identical() {
        let plane = vec![128u8; 16 * 16];
        let (py, pcb, pcr) =
            psnr_yuv(&plane, &plane, &plane, &plane, &plane, &plane, 16, 16, 16).expect("psnr_yuv");
        assert_eq!(py, f64::INFINITY);
        assert_eq!(pcb, f64::INFINITY);
        assert_eq!(pcr, f64::INFINITY);
    }
}
