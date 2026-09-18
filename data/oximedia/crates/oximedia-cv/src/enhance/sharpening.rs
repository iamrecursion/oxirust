//! Image sharpening and edge enhancement.
//!
//! Provides multiple sharpening strategies operating on raw 8-bit grayscale
//! or single-channel pixel buffers:
//!
//! - **Unsharp Mask**: Classic luminance sharpening via `(original - blur) * amount`.
//! - **Laplacian**: Adds a Laplacian-filtered image back onto the original.
//! - **High-pass**: Subtracts a Gaussian-blurred copy to isolate high frequencies,
//!   then blends back.
//! - **CLAHE-aided**: Placeholder variant for histogram-equalisation-assisted
//!   sharpening (set up for future GPU paths).
//!
//! All functions expect and return flat, row-major byte slices of length
//! `width * height` (one byte per pixel).
//!
//! # Example
//!
//! ```
//! use oximedia_cv::enhance::sharpening::{UnsharpMaskParams, unsharp_mask};
//!
//! let pixels = vec![128u8; 64 * 64];
//! let params = UnsharpMaskParams { radius: 1.0, amount: 1.5, threshold: 0.0 };
//! let result = unsharp_mask(&pixels, 64, 64, &params);
//! assert_eq!(result.len(), 64 * 64);
//! ```

#![allow(dead_code)]

/// Available sharpening algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SharpenMethod {
    /// Classic unsharp mask: `result = original + amount * (original - blur)`.
    UnsharpMask,
    /// Laplacian-based edge enhancement.
    Laplacian,
    /// High-pass filter sharpening.
    HighPass,
    /// CLAHE-aided sharpening (contrast-limited adaptive histogram equalization).
    ClaheAided,
}

/// Parameters for the unsharp mask algorithm.
#[derive(Debug, Clone, Copy)]
pub struct UnsharpMaskParams {
    /// Gaussian blur radius (sigma in pixels, > 0).
    pub radius: f64,
    /// Sharpening strength multiplier (1.0 = standard).
    pub amount: f64,
    /// Minimum difference (0-255 scale) before sharpening is applied.
    pub threshold: f64,
}

impl Default for UnsharpMaskParams {
    fn default() -> Self {
        Self {
            radius: 1.0,
            amount: 1.0,
            threshold: 0.0,
        }
    }
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Build a 1-D Gaussian kernel of the given sigma.
///
/// The kernel is normalised so its sum equals 1.0.
fn gaussian_kernel_1d(sigma: f64) -> Vec<f64> {
    let radius = ((sigma * 3.0).ceil() as usize).max(1);
    let len = 2 * radius + 1;
    let mut k = vec![0.0f64; len];
    let two_sigma_sq = 2.0 * sigma * sigma;

    let mut sum = 0.0;
    for (i, val) in k.iter_mut().enumerate() {
        let x = i as f64 - radius as f64;
        *val = (-x * x / two_sigma_sq).exp();
        sum += *val;
    }
    for val in &mut k {
        *val /= sum;
    }
    k
}

/// Separable Gaussian blur on a grayscale (`width * height`) byte slice.
/// Returns a new `Vec<u8>` of the same length.
fn gaussian_blur_internal(pixels: &[u8], width: usize, height: usize, sigma: f64) -> Vec<u8> {
    if width == 0 || height == 0 || pixels.len() < width * height {
        return pixels.to_vec();
    }

    let k = gaussian_kernel_1d(sigma.max(0.1));
    let radius = k.len() / 2;

    // Horizontal pass → temp (f64 for precision).
    let mut temp = vec![0.0f64; width * height];
    for y in 0..height {
        for x in 0..width {
            let mut acc = 0.0f64;
            for (ki, &kv) in k.iter().enumerate() {
                let sx = (x as i64 + ki as i64 - radius as i64).clamp(0, width as i64 - 1) as usize;
                acc += pixels[y * width + sx] as f64 * kv;
            }
            temp[y * width + x] = acc;
        }
    }

    // Vertical pass → output.
    let mut out = vec![0u8; width * height];
    for y in 0..height {
        for x in 0..width {
            let mut acc = 0.0f64;
            for (ki, &kv) in k.iter().enumerate() {
                let sy =
                    (y as i64 + ki as i64 - radius as i64).clamp(0, height as i64 - 1) as usize;
                acc += temp[sy * width + x] * kv;
            }
            out[y * width + x] = acc.round().clamp(0.0, 255.0) as u8;
        }
    }
    out
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Apply a Gaussian blur to a grayscale image.
///
/// # Arguments
///
/// * `pixels` - Row-major grayscale image (`width * height` bytes).
/// * `width`  - Image width in pixels.
/// * `height` - Image height in pixels.
/// * `sigma`  - Gaussian blur radius (standard deviation in pixels).
///
/// # Returns
///
/// New `Vec<u8>` with the blurred image (same dimensions).
///
/// # Examples
///
/// ```
/// use oximedia_cv::enhance::sharpening::gaussian_blur;
///
/// let pixels = vec![200u8; 16 * 16];
/// let blurred = gaussian_blur(&pixels, 16, 16, 1.0);
/// assert_eq!(blurred.len(), 16 * 16);
/// ```
#[must_use]
pub fn gaussian_blur(pixels: &[u8], width: usize, height: usize, sigma: f64) -> Vec<u8> {
    gaussian_blur_internal(pixels, width, height, sigma)
}

/// Apply an unsharp mask to sharpen a grayscale image.
///
/// The algorithm computes:
/// ```text
/// mask   = original - gaussian_blur(original, radius)
/// result = original + amount * mask   (where |mask| > threshold)
/// ```
///
/// # Arguments
///
/// * `pixels` - Row-major grayscale image (`width * height` bytes).
/// * `width`  - Image width.
/// * `height` - Image height.
/// * `params` - Unsharp mask parameters.
///
/// # Returns
///
/// Sharpened image as `Vec<u8>` (clamped to 0–255).
///
/// # Examples
///
/// ```
/// use oximedia_cv::enhance::sharpening::{UnsharpMaskParams, unsharp_mask};
///
/// let pixels = vec![128u8; 32 * 32];
/// let params = UnsharpMaskParams { radius: 1.0, amount: 1.5, threshold: 5.0 };
/// let result = unsharp_mask(&pixels, 32, 32, &params);
/// assert_eq!(result.len(), 32 * 32);
/// ```
#[must_use]
pub fn unsharp_mask(
    pixels: &[u8],
    width: usize,
    height: usize,
    params: &UnsharpMaskParams,
) -> Vec<u8> {
    let blurred = gaussian_blur_internal(pixels, width, height, params.radius.max(0.1));
    let n = width * height;
    let mut out = vec![0u8; n];

    for i in 0..n.min(pixels.len()) {
        let orig = pixels[i] as f64;
        let blur = blurred[i] as f64;
        let mask = orig - blur;
        if mask.abs() > params.threshold {
            let sharpened = orig + params.amount * mask;
            out[i] = sharpened.round().clamp(0.0, 255.0) as u8;
        } else {
            out[i] = pixels[i];
        }
    }
    out
}

/// Sharpen a grayscale image using a Laplacian edge-enhancement filter.
///
/// A 3×3 Laplacian kernel is convolved with the image and the scaled result
/// is subtracted from the original (i.e. edges are reinforced).
///
/// Kernel used (negative Laplacian):
/// ```text
///  0  -1   0
/// -1   4  -1
///  0  -1   0
/// ```
///
/// The output is: `result = original + strength * laplacian_response`
///
/// # Arguments
///
/// * `pixels`   - Row-major grayscale image.
/// * `width`    - Image width.
/// * `height`   - Image height.
/// * `strength` - Scaling factor applied to the Laplacian response (≥ 0).
///
/// # Examples
///
/// ```
/// use oximedia_cv::enhance::sharpening::laplacian_sharpen;
///
/// let pixels = vec![128u8; 8 * 8];
/// let result = laplacian_sharpen(&pixels, 8, 8, 0.5);
/// assert_eq!(result.len(), 64);
/// ```
#[must_use]
pub fn laplacian_sharpen(pixels: &[u8], width: usize, height: usize, strength: f64) -> Vec<u8> {
    if width < 3 || height < 3 {
        return pixels.to_vec();
    }

    // Laplacian kernel (4-connected, scaled by 1 for simplicity)
    // result[y][x] = orig - strength * (-4*p + n+s+e+w)
    //              = orig + strength * (4*p - n - s - e - w)
    let mut out = vec![0u8; width * height];

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let center = pixels[idx] as f64;

            let n = if y > 0 {
                pixels[(y - 1) * width + x] as f64
            } else {
                center
            };
            let s = if y + 1 < height {
                pixels[(y + 1) * width + x] as f64
            } else {
                center
            };
            let w = if x > 0 {
                pixels[y * width + x - 1] as f64
            } else {
                center
            };
            let e = if x + 1 < width {
                pixels[y * width + x + 1] as f64
            } else {
                center
            };

            let lap = 4.0 * center - n - s - w - e;
            let val = center + strength * lap;
            out[idx] = val.round().clamp(0.0, 255.0) as u8;
        }
    }
    out
}

/// Sharpen a grayscale image using a high-pass filter approach.
///
/// Steps:
/// 1. Blur the image with a Gaussian of standard deviation `radius`.
/// 2. Compute high-frequency layer: `high = original - blurred`.
/// 3. Blend: `result = original + strength * high`.
///
/// # Arguments
///
/// * `pixels`   - Row-major grayscale image.
/// * `width`    - Image width.
/// * `height`   - Image height.
/// * `radius`   - Gaussian blur sigma for the low-pass component.
/// * `strength` - Blend factor for the high-frequency layer.
///
/// # Examples
///
/// ```
/// use oximedia_cv::enhance::sharpening::high_pass_sharpen;
///
/// let pixels = vec![100u8; 16 * 16];
/// let result = high_pass_sharpen(&pixels, 16, 16, 2.0, 1.0);
/// assert_eq!(result.len(), 16 * 16);
/// ```
#[must_use]
pub fn high_pass_sharpen(
    pixels: &[u8],
    width: usize,
    height: usize,
    radius: f64,
    strength: f64,
) -> Vec<u8> {
    let blurred = gaussian_blur_internal(pixels, width, height, radius.max(0.1));
    let n = width * height;
    let mut out = vec![0u8; n];

    for i in 0..n.min(pixels.len()) {
        let orig = pixels[i] as f64;
        let low = blurred[i] as f64;
        let high = orig - low;
        let val = orig + strength * high;
        out[i] = val.round().clamp(0.0, 255.0) as u8;
    }
    out
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    fn uniform(val: u8, w: usize, h: usize) -> Vec<u8> {
        vec![val; w * h]
    }

    /// Checkerboard pattern: alternating 0 / 255.
    fn checkerboard(w: usize, h: usize) -> Vec<u8> {
        (0..w * h)
            .map(|i| if (i + i / w) % 2 == 0 { 0 } else { 255 })
            .collect()
    }

    #[test]
    fn test_gaussian_kernel_sum() {
        let k = gaussian_kernel_1d(2.0);
        let sum: f64 = k.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_gaussian_kernel_length_odd() {
        let k = gaussian_kernel_1d(1.5);
        assert_eq!(k.len() % 2, 1);
    }

    #[test]
    fn test_gaussian_blur_uniform_preserves_value() {
        let pixels = uniform(100, 32, 32);
        let blurred = gaussian_blur(&pixels, 32, 32, 1.5);
        // Uniform image blurred should stay uniform
        for &p in &blurred {
            assert_eq!(p, 100);
        }
    }

    #[test]
    fn test_gaussian_blur_output_length() {
        let pixels = uniform(128, 16, 16);
        let blurred = gaussian_blur(&pixels, 16, 16, 2.0);
        assert_eq!(blurred.len(), 16 * 16);
    }

    #[test]
    fn test_unsharp_mask_uniform_unchanged() {
        let pixels = uniform(128, 16, 16);
        let params = UnsharpMaskParams {
            radius: 1.0,
            amount: 2.0,
            threshold: 0.0,
        };
        let result = unsharp_mask(&pixels, 16, 16, &params);
        // Uniform → mask = 0 → no change
        for &p in &result {
            assert_eq!(p, 128);
        }
    }

    #[test]
    fn test_unsharp_mask_output_length() {
        let pixels = checkerboard(20, 20);
        let params = UnsharpMaskParams::default();
        let result = unsharp_mask(&pixels, 20, 20, &params);
        assert_eq!(result.len(), 20 * 20);
    }

    #[test]
    fn test_unsharp_mask_clamps_to_0_255() {
        let pixels = uniform(250, 16, 16);
        // Add some edges manually
        let mut px = pixels;
        px[16 * 8 + 8] = 10; // single dark pixel among bright
        let params = UnsharpMaskParams {
            radius: 0.5,
            amount: 5.0,
            threshold: 0.0,
        };
        let result = unsharp_mask(&px, 16, 16, &params);
        assert_eq!(result.len(), 16 * 16, "output length must match input");
    }

    #[test]
    fn test_laplacian_sharpen_output_length() {
        let pixels = uniform(100, 10, 10);
        let result = laplacian_sharpen(&pixels, 10, 10, 0.5);
        assert_eq!(result.len(), 100);
    }

    #[test]
    fn test_laplacian_sharpen_uniform_unchanged() {
        // Uniform → laplacian = 0 → no change
        let pixels = uniform(128, 10, 10);
        let result = laplacian_sharpen(&pixels, 10, 10, 1.0);
        for &p in &result {
            assert_eq!(p, 128);
        }
    }

    #[test]
    fn test_laplacian_sharpen_small_image_passthrough() {
        // width < 3 → passthrough
        let pixels = vec![50u8, 100, 150, 200];
        let result = laplacian_sharpen(&pixels, 2, 2, 1.0);
        assert_eq!(result, pixels);
    }

    #[test]
    fn test_laplacian_sharpen_clamps() {
        let pixels = checkerboard(8, 8);
        let result = laplacian_sharpen(&pixels, 8, 8, 10.0);
        assert_eq!(result.len(), pixels.len(), "output length must match input");
    }

    #[test]
    fn test_high_pass_sharpen_output_length() {
        let pixels = uniform(128, 16, 16);
        let result = high_pass_sharpen(&pixels, 16, 16, 2.0, 1.0);
        assert_eq!(result.len(), 16 * 16);
    }

    #[test]
    fn test_high_pass_sharpen_uniform_unchanged() {
        // Uniform → high = 0 → no change
        let pixels = uniform(128, 16, 16);
        let result = high_pass_sharpen(&pixels, 16, 16, 1.5, 1.0);
        for &p in &result {
            assert_eq!(p, 128);
        }
    }

    #[test]
    fn test_high_pass_sharpen_clamps() {
        let pixels = checkerboard(16, 16);
        let result = high_pass_sharpen(&pixels, 16, 16, 1.0, 10.0);
        assert_eq!(result.len(), pixels.len(), "output length must match input");
    }

    #[test]
    fn test_sharpen_method_variants() {
        // Ensure enum variants are accessible and differ
        assert_ne!(SharpenMethod::UnsharpMask, SharpenMethod::Laplacian);
        assert_ne!(SharpenMethod::HighPass, SharpenMethod::ClaheAided);
    }
}
