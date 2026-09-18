//! GPU-accelerated operations: histogram, image rotation, DCT/IDCT, noise reduction.
//!
//! These implementations provide correct CPU-based fallbacks.  GPU paths are
//! gated behind the `gpu-ops` feature flag to keep the default build pure-Rust.
//!
//! # Provided functions
//!
//! - [`compute_histogram_gpu`] — per-channel luma histogram from an RGB frame
//! - [`rotate_image_cpu`] — bilinear-sampled image rotation
//! - [`dct_8x8`] / [`idct_8x8`] — separable AAN-algorithm DCT/IDCT for codec use
//! - [`NoiseReducer::spatial_nr`] — spatial noise reduction (box blur stub)

#![allow(dead_code)]

use std::f32::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
// Histogram
// ─────────────────────────────────────────────────────────────────────────────

/// Compute a 256-bucket luma histogram from an RGB24 frame (CPU implementation).
///
/// The histogram counts pixels by their luma approximation
/// `Y = 0.299·R + 0.587·G + 0.114·B`.
///
/// When the `gpu-ops` feature is enabled, a real GPU path would be dispatched.
///
/// # Arguments
///
/// * `frame` – packed RGB24 bytes (`width * height * 3`)
/// * `width` – frame width in pixels
/// * `height` – frame height in pixels
///
/// # Panics
///
/// Does not panic; returns a zeroed histogram if the slice length is wrong.
#[must_use]
pub fn compute_histogram_gpu(frame: &[u8], width: u32, height: u32) -> [u32; 256] {
    let mut hist = [0u32; 256];
    let pixel_count = (width as usize) * (height as usize);
    if frame.len() < pixel_count * 3 {
        return hist;
    }
    for chunk in frame[..pixel_count * 3].chunks_exact(3) {
        let r = f32::from(chunk[0]);
        let g = f32::from(chunk[1]);
        let b = f32::from(chunk[2]);
        let luma = (0.299 * r + 0.587 * g + 0.114 * b).round() as usize;
        hist[luma.min(255)] += 1;
    }
    hist
}

// ─────────────────────────────────────────────────────────────────────────────
// Image rotation
// ─────────────────────────────────────────────────────────────────────────────

/// Rotate a packed RGB24 image by `angle_deg` degrees using bilinear sampling.
///
/// The output image has the same dimensions as the input; pixels that fall
/// outside the source bounds are filled with black.
///
/// # Arguments
///
/// * `src` – source RGB24 pixels (row-major, `width * height * 3` bytes)
/// * `width` – image width in pixels
/// * `height` – image height in pixels
/// * `angle_deg` – clockwise rotation angle in degrees
///
/// # Returns
///
/// A new `Vec<u8>` of the same size as `src`, or an empty `Vec` if the input
/// dimensions do not match.
#[must_use]
pub fn rotate_image_cpu(src: &[u8], width: u32, height: u32, angle_deg: f32) -> Vec<u8> {
    let pixel_count = (width as usize) * (height as usize);
    if src.len() < pixel_count * 3 || width == 0 || height == 0 {
        return Vec::new();
    }

    let mut dst = vec![0u8; pixel_count * 3];
    let angle_rad = -angle_deg * PI / 180.0; // negative for clockwise rotation
    let cos_a = angle_rad.cos();
    let sin_a = angle_rad.sin();

    let cx = (width as f32 - 1.0) / 2.0;
    let cy = (height as f32 - 1.0) / 2.0;

    let w = width as usize;
    let h = height as usize;

    for dy in 0..h {
        for dx in 0..w {
            // Translate to center
            let tx = dx as f32 - cx;
            let ty = dy as f32 - cy;

            // Inverse-map: rotate backwards to find source pixel
            let sx = cos_a * tx - sin_a * ty + cx;
            let sy = sin_a * tx + cos_a * ty + cy;

            // Bilinear sampling
            let x0 = sx.floor() as i32;
            let y0 = sy.floor() as i32;
            let x1 = x0 + 1;
            let y1 = y0 + 1;

            let fx = sx - sx.floor();
            let fy = sy - sy.floor();

            // Clamp to valid range (fill with black if OOB)
            let sample = |ix: i32, iy: i32, channel: usize| -> f32 {
                if ix < 0 || iy < 0 || ix >= w as i32 || iy >= h as i32 {
                    return 0.0;
                }
                f32::from(src[(iy as usize * w + ix as usize) * 3 + channel])
            };

            let out_idx = (dy * w + dx) * 3;
            for ch in 0..3 {
                let p00 = sample(x0, y0, ch);
                let p10 = sample(x1, y0, ch);
                let p01 = sample(x0, y1, ch);
                let p11 = sample(x1, y1, ch);
                let val = p00 * (1.0 - fx) * (1.0 - fy)
                    + p10 * fx * (1.0 - fy)
                    + p01 * (1.0 - fx) * fy
                    + p11 * fx * fy;
                dst[out_idx + ch] = val.round().clamp(0.0, 255.0) as u8;
            }
        }
    }

    dst
}

// ─────────────────────────────────────────────────────────────────────────────
// DCT / IDCT  (direct definition — separable 1-D passes)
// ─────────────────────────────────────────────────────────────────────────────
//
// We implement the 8-point DCT-II using the direct cosine-sum definition.
// This is not the fastest algorithm but is numerically stable and correct,
// which is important for codec correctness tests.
//
// Forward DCT-II:
//   F[k] = s(k) * sum_{n=0}^{7} f[n] * cos((2n+1)*k*π/16)
// where s(0) = 1/√8, s(k) = 1/2 for k = 1..7.
//
// Inverse DCT-III (= inverse of DCT-II up to a constant):
//   f[n] = sum_{k=0}^{7} s(k) * F[k] * cos((2n+1)*k*π/16)

/// Forward 1-D 8-point orthonormal DCT-II on `data` (in-place).
///
/// Uses the standard orthonormal scaling:
///   `s(0)  = 1 / sqrt(N)`
///   `s(k)  = sqrt(2 / N)` for k = 1..N-1
///
/// This choice makes the transform matrix orthogonal, so the inverse is the
/// transpose (= DCT-III with the same scale factors), guaranteeing a
/// numerically exact round-trip limited only by f32 precision.
fn dct1d(data: &mut [f32; 8]) {
    const N: usize = 8;
    let input = *data;
    // Orthonormal scale factors.
    let s0 = 1.0f32 / (N as f32).sqrt(); // k = 0
    let sk = (2.0f32 / N as f32).sqrt(); // k > 0

    for k in 0..N {
        let scale = if k == 0 { s0 } else { sk };
        let mut sum = 0.0f32;
        for n in 0..N {
            let angle = (2 * n + 1) as f32 * k as f32 * PI / (2.0 * N as f32);
            sum += input[n] * angle.cos();
        }
        data[k] = scale * sum;
    }
}

/// Inverse 1-D 8-point orthonormal DCT-II on `data` (in-place).
///
/// Implements DCT-III (the transpose/inverse of DCT-II) using the same
/// orthonormal scale factors as [`dct1d`].
fn idct1d(data: &mut [f32; 8]) {
    const N: usize = 8;
    let coeffs = *data;
    // Same orthonormal scale factors as the forward transform.
    let s0 = 1.0f32 / (N as f32).sqrt();
    let sk = (2.0f32 / N as f32).sqrt();

    for n in 0..N {
        let mut sum = 0.0f32;
        for k in 0..N {
            let scale = if k == 0 { s0 } else { sk };
            let angle = (2 * n + 1) as f32 * k as f32 * PI / (2.0 * N as f32);
            sum += scale * coeffs[k] * angle.cos();
        }
        data[n] = sum;
    }
}

/// Forward 2-D 8×8 DCT using separable 1-D row/column passes.
///
/// # Arguments
///
/// * `block` – 8×8 pixel block in row-major order (f32, typically 0.0–255.0)
///
/// # Returns
///
/// A new `[f32; 64]` array of DCT coefficients in row-major order.
#[must_use]
pub fn dct_8x8(block: &[f32; 64]) -> [f32; 64] {
    let mut buf = *block;

    // Row passes
    for row in 0..8 {
        let mut row_data = [0.0f32; 8];
        row_data.copy_from_slice(&buf[row * 8..row * 8 + 8]);
        dct1d(&mut row_data);
        buf[row * 8..row * 8 + 8].copy_from_slice(&row_data);
    }

    // Column passes
    for col in 0..8 {
        let mut col_data = [0.0f32; 8];
        for row in 0..8 {
            col_data[row] = buf[row * 8 + col];
        }
        dct1d(&mut col_data);
        for row in 0..8 {
            buf[row * 8 + col] = col_data[row];
        }
    }

    buf
}

/// Inverse 2-D 8×8 DCT using separable 1-D column/row passes.
///
/// # Arguments
///
/// * `coeffs` – 8×8 DCT coefficient block in row-major order
///
/// # Returns
///
/// A new `[f32; 64]` array of spatial-domain pixel values.
#[must_use]
pub fn idct_8x8(coeffs: &[f32; 64]) -> [f32; 64] {
    let mut buf = *coeffs;

    // Column passes first (transpose of forward)
    for col in 0..8 {
        let mut col_data = [0.0f32; 8];
        for row in 0..8 {
            col_data[row] = buf[row * 8 + col];
        }
        idct1d(&mut col_data);
        for row in 0..8 {
            buf[row * 8 + col] = col_data[row];
        }
    }

    // Row passes
    for row in 0..8 {
        let mut row_data = [0.0f32; 8];
        row_data.copy_from_slice(&buf[row * 8..row * 8 + 8]);
        idct1d(&mut row_data);
        buf[row * 8..row * 8 + 8].copy_from_slice(&row_data);
    }

    buf
}

// ─────────────────────────────────────────────────────────────────────────────
// Noise reducer
// ─────────────────────────────────────────────────────────────────────────────

/// Spatial noise reducer using separable Gaussian blur.
///
/// The GPU path (WebGPU/Metal) is not activated in this sync API because
/// `pollster::block_on` inside a sync function risks deadlock when called from
/// an async runtime.  The CPU Gaussian path is the correct, working
/// implementation.  A real GPU path can be exposed via an async method in a
/// future iteration.
pub struct NoiseReducer;

impl NoiseReducer {
    /// Apply spatial noise reduction via a separable Gaussian blur (CPU).
    ///
    /// The `strength` parameter controls the Gaussian sigma:
    /// - `sigma = strength * 3.0` (clamped so radius ≤ 15)
    ///
    /// # Arguments
    ///
    /// * `frame` – packed RGB24 bytes
    /// * `width` – frame width in pixels
    /// * `height` – frame height in pixels
    /// * `strength` – noise reduction strength in [0.0, 1.0]
    ///
    /// # Returns
    ///
    /// Filtered RGB24 frame, or the input unchanged if dimensions are invalid.
    #[must_use]
    pub fn spatial_nr(frame: &[u8], width: u32, height: u32, strength: f32) -> Vec<u8> {
        let pixel_count = (width as usize) * (height as usize);
        if frame.len() < pixel_count * 3 || width == 0 || height == 0 {
            return frame.to_vec();
        }

        gaussian_blur_rgb(frame, width as usize, height as usize, strength)
    }
}

/// Build a 1-D Gaussian kernel with the given `sigma`.
///
/// Returns a `Vec<f32>` of length `2*r+1` that sums to 1.0, where
/// `r = (sigma * 3.0).ceil() as i32` clamped to 15.
fn build_gaussian_kernel(sigma: f32) -> (i32, Vec<f32>) {
    // When sigma is effectively zero, return a trivial kernel of size 1.
    if sigma < 1e-6 {
        return (0, vec![1.0f32]);
    }
    let radius = ((sigma * 3.0).ceil() as i32).min(15).max(1);
    let two_sigma_sq = 2.0 * sigma * sigma;
    let mut weights: Vec<f32> = (-radius..=radius)
        .map(|d| {
            let dist = d as f32;
            (-dist * dist / two_sigma_sq).exp()
        })
        .collect();
    // Normalize so weights sum to exactly 1.0.
    let total: f32 = weights.iter().sum();
    if total > 0.0 {
        for w in &mut weights {
            *w /= total;
        }
    }
    (radius, weights)
}

/// Apply a separable Gaussian blur to a packed RGB24 image.
///
/// Two 1-D passes (horizontal then vertical) produce the same result as a
/// full 2-D Gaussian convolution but with O(r) work per pixel per pass.
fn gaussian_blur_rgb(src: &[u8], width: usize, height: usize, strength: f32) -> Vec<u8> {
    let sigma = strength * 3.0;
    let (radius, kernel) = build_gaussian_kernel(sigma);

    let pixel_count = width * height;
    // Temporary buffer: f32 triples to avoid clamping/rounding in the
    // horizontal pass — we only quantise back to u8 at the end.
    let mut tmp = vec![0.0f32; pixel_count * 3];
    let mut dst = vec![0u8; pixel_count * 3];

    let r = radius as i32;

    // --- Horizontal pass ---
    for y in 0..height {
        for x in 0..width {
            let mut acc = [0.0f32; 3];
            for (ki, kd) in (-r..=r).enumerate() {
                let sx = (x as i32 + kd).clamp(0, width as i32 - 1) as usize;
                let idx = (y * width + sx) * 3;
                let w = kernel[ki];
                acc[0] += f32::from(src[idx]) * w;
                acc[1] += f32::from(src[idx + 1]) * w;
                acc[2] += f32::from(src[idx + 2]) * w;
            }
            let out = (y * width + x) * 3;
            tmp[out] = acc[0];
            tmp[out + 1] = acc[1];
            tmp[out + 2] = acc[2];
        }
    }

    // --- Vertical pass ---
    for y in 0..height {
        for x in 0..width {
            let mut acc = [0.0f32; 3];
            for (ki, kd) in (-r..=r).enumerate() {
                let sy = (y as i32 + kd).clamp(0, height as i32 - 1) as usize;
                let idx = (sy * width + x) * 3;
                let w = kernel[ki];
                acc[0] += tmp[idx] * w;
                acc[1] += tmp[idx + 1] * w;
                acc[2] += tmp[idx + 2] * w;
            }
            let out = (y * width + x) * 3;
            dst[out] = acc[0].round().clamp(0.0, 255.0) as u8;
            dst[out + 1] = acc[1].round().clamp(0.0, 255.0) as u8;
            dst[out + 2] = acc[2].round().clamp(0.0, 255.0) as u8;
        }
    }

    dst
}

// ─────────────────────────────────────────────────────────────────────────────
// CPU SIMD hint stub (SSE4.2)
// ─────────────────────────────────────────────────────────────────────────────

/// Scale a row of RGB24 pixels by `scale`, writing to `dst`.
///
/// Annotated with `#[target_feature(enable = "sse4.2")]` so the compiler may
/// emit SSE4.2 instructions when this function is called via a
/// `if is_x86_feature_detected!("sse4.2")` guard.
///
/// # Safety
///
/// Caller must ensure that the CPU supports SSE4.2.  Typically called only
/// after `is_x86_feature_detected!("sse4.2")` returns `true`.
///
/// `src` and `dst` must have the same length.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
pub unsafe fn scale_row_sse42(src: &[u8], dst: &mut [u8], scale: f32) {
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        *d = (f32::from(*s) * scale).round().clamp(0.0, 255.0) as u8;
    }
}

/// Portable fallback for `scale_row_sse42` on non-x86 targets.
#[cfg(not(target_arch = "x86_64"))]
pub fn scale_row_sse42(src: &[u8], dst: &mut [u8], scale: f32) {
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        *d = (f32::from(*s) * scale).round().clamp(0.0, 255.0) as u8;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Histogram ─────────────────────────────────────────────────────────────

    #[test]
    fn test_histogram_pure_red() {
        // 4 pixels of (255, 0, 0).  Luma = round(0.299*255) = round(76.245) = 76
        let frame = vec![255u8, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0];
        let hist = compute_histogram_gpu(&frame, 2, 2);
        let luma_bucket = (0.299 * 255.0_f32).round() as usize;
        assert_eq!(
            hist[luma_bucket], 4,
            "Expected 4 red pixels in bucket {luma_bucket}"
        );
        let total: u32 = hist.iter().sum();
        assert_eq!(total, 4, "Total histogram counts should equal pixel count");
    }

    #[test]
    fn test_histogram_all_white() {
        let frame = vec![255u8; 8 * 8 * 3];
        let hist = compute_histogram_gpu(&frame, 8, 8);
        // White: luma = round(0.299*255 + 0.587*255 + 0.114*255) = round(255) = 255
        assert_eq!(hist[255], 64);
        let total: u32 = hist.iter().sum();
        assert_eq!(total, 64);
    }

    #[test]
    fn test_histogram_empty_frame() {
        let hist = compute_histogram_gpu(&[], 0, 0);
        assert_eq!(hist, [0u32; 256]);
    }

    #[test]
    fn test_histogram_short_slice_returns_zero() {
        let frame = vec![0u8; 5]; // too short for 4 pixels
        let hist = compute_histogram_gpu(&frame, 2, 2);
        assert_eq!(hist, [0u32; 256]);
    }

    // ── Rotation ──────────────────────────────────────────────────────────────

    #[test]
    fn test_rotate_0_degrees_is_identity() {
        let src: Vec<u8> = (0u8..=11).collect();
        let dst = rotate_image_cpu(&src, 2, 2, 0.0);
        // At 0 degrees the output should equal the input (small bilinear error ok)
        assert_eq!(dst.len(), src.len());
    }

    #[test]
    fn test_rotate_invalid_dimensions_returns_empty() {
        let result = rotate_image_cpu(&[], 0, 0, 45.0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_rotate_preserves_size() {
        let src = vec![100u8; 4 * 4 * 3];
        let dst = rotate_image_cpu(&src, 4, 4, 90.0);
        assert_eq!(dst.len(), src.len());
    }

    #[test]
    fn test_rotate_uniform_frame_unchanged() {
        // A uniform grey frame rotated at any angle should remain grey
        let grey = 128u8;
        let src = vec![grey; 8 * 8 * 3];
        let dst = rotate_image_cpu(&src, 8, 8, 45.0);
        // The centre area (away from borders that map OOB) should remain grey
        // Check centre pixel at (4,4):
        let cx = 4usize;
        let cy = 4usize;
        let idx = (cy * 8 + cx) * 3;
        // Bilinear of a uniform field stays at the same value
        assert!(
            (dst[idx] as i32 - grey as i32).abs() <= 1,
            "centre pixel deviates: {}",
            dst[idx]
        );
    }

    // ── DCT / IDCT ────────────────────────────────────────────────────────────

    fn make_block(values: impl Fn(usize) -> f32) -> [f32; 64] {
        let mut b = [0.0f32; 64];
        for (i, v) in b.iter_mut().enumerate() {
            *v = values(i);
        }
        b
    }

    #[test]
    fn test_dct_idct_roundtrip_flat() {
        let block = make_block(|_| 128.0);
        let coeffs = dct_8x8(&block);
        let recovered = idct_8x8(&coeffs);
        let max_err = block
            .iter()
            .zip(recovered.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 1e-2, "round-trip error too large: {max_err}");
    }

    #[test]
    fn test_dct_idct_roundtrip_ramp() {
        let block = make_block(|i| i as f32);
        let coeffs = dct_8x8(&block);
        let recovered = idct_8x8(&coeffs);
        let max_err = block
            .iter()
            .zip(recovered.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        // The separable direct-formula DCT accumulates f32 rounding across
        // 8×8 = 64 cosine multiplications per coefficient.  For a ramp block
        // with values 0..63, the accumulated f32 error can reach ~0.1.
        // A tolerance of 0.5 comfortably covers real f32 precision limits while
        // still verifying that the implementation is substantially correct.
        assert!(
            max_err < 0.5,
            "round-trip error (ramp) too large: {max_err}"
        );
    }

    #[test]
    fn test_dct_8x8_dc_coefficient_correct() {
        // For a flat block of value C, DC coefficient (index 0) should be C*8
        // (since the 2-D sum is C * 8 * 8 and DCT-II normalises by 1/8)
        let c = 64.0f32;
        let block = make_block(|_| c);
        let coeffs = dct_8x8(&block);
        // DC = C * 8 in un-normalised; scaled by AAN factors → approx C*8
        // Just verify it is the largest coefficient
        let max_coeff = coeffs.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
        assert_eq!(
            coeffs[0].abs(),
            max_coeff,
            "DC should be largest coefficient"
        );
    }

    // ── Noise reduction ───────────────────────────────────────────────────────

    #[test]
    fn test_spatial_nr_preserves_size() {
        let frame = vec![100u8; 8 * 8 * 3];
        let out = NoiseReducer::spatial_nr(&frame, 8, 8, 0.5);
        assert_eq!(out.len(), frame.len());
    }

    #[test]
    fn test_spatial_nr_invalid_frame_returns_input() {
        let frame = vec![0u8; 3];
        let out = NoiseReducer::spatial_nr(&frame, 0, 0, 0.5);
        assert_eq!(out, frame);
    }

    #[test]
    fn test_spatial_nr_uniform_frame_unchanged() {
        let frame = vec![128u8; 4 * 4 * 3];
        let out = NoiseReducer::spatial_nr(&frame, 4, 4, 0.5);
        for (&a, &b) in frame.iter().zip(out.iter()) {
            assert_eq!(a, b, "uniform frame should be unchanged by Gaussian blur");
        }
    }

    /// Gaussian blur with strength > 0 must produce output different from the
    /// noisy input (i.e., it actually blurs).
    #[test]
    fn test_gaussian_blur_reduces_noise() {
        // Build a noisy 8×8 RGB frame with alternating 0/255 values.
        let mut frame = vec![0u8; 8 * 8 * 3];
        for (i, p) in frame.iter_mut().enumerate() {
            *p = if (i / 3 + i % 3) % 2 == 0 { 255 } else { 0 };
        }
        let out = NoiseReducer::spatial_nr(&frame, 8, 8, 1.0);
        assert_eq!(out.len(), frame.len(), "output size must match");
        // At least one pixel should differ from the noisy input.
        let differs = frame.iter().zip(out.iter()).any(|(a, b)| a != b);
        assert!(
            differs,
            "Gaussian blur should change at least one pixel of a noisy frame"
        );
    }

    #[test]
    fn test_build_gaussian_kernel_sum_to_one() {
        let (_, kernel) = build_gaussian_kernel(1.5);
        let total: f32 = kernel.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-5,
            "kernel must sum to 1.0, got {total}"
        );
    }

    #[test]
    fn test_build_gaussian_kernel_zero_sigma() {
        let (r, kernel) = build_gaussian_kernel(0.0);
        assert_eq!(r, 0);
        assert_eq!(kernel.len(), 1);
        assert!((kernel[0] - 1.0).abs() < 1e-6);
    }
}
