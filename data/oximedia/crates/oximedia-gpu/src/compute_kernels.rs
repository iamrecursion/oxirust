//! Pure-Rust SIMD-optimised compute kernels for image processing.
//!
//! All algorithms are implemented as scalar loops written in a style that
//! LLVM / rustc can auto-vectorise (chunk-unrolled, no data-dependent
//! branches inside hot loops).  No external C/Fortran dependencies are used.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_lossless)]

/// Configuration for a [`ComputeKernel`].
#[derive(Debug, Clone)]
pub struct KernelConfig {
    /// Whether to enable SIMD-style optimised code paths.
    pub use_simd: bool,
    /// Number of threads to use in future parallel extensions.
    pub thread_count: usize,
    /// Processing chunk size (default 256).  Must be a power of two ≥ 8.
    pub chunk_size: usize,
}

impl Default for KernelConfig {
    fn default() -> Self {
        Self {
            use_simd: true,
            thread_count: 1,
            chunk_size: 256,
        }
    }
}

/// Collection of CPU compute kernels for image processing.
///
/// Every method is a pure function — `&self` is used only to read
/// [`KernelConfig`]; no mutable state is kept.
pub struct ComputeKernel {
    config: KernelConfig,
}

impl ComputeKernel {
    /// Create a new `ComputeKernel` with the given configuration.
    #[must_use]
    pub fn new(config: KernelConfig) -> Self {
        Self { config }
    }

    /// Create a new `ComputeKernel` with default configuration.
    #[must_use]
    pub fn default_config() -> Self {
        Self::new(KernelConfig::default())
    }

    /// Return a reference to the current configuration.
    #[must_use]
    pub fn config(&self) -> &KernelConfig {
        &self.config
    }

    // -----------------------------------------------------------------------
    // RGBA → YUV 420 (BT.601)
    // -----------------------------------------------------------------------

    /// Convert packed RGBA to planar YUV 420.
    ///
    /// Layout of the returned buffer:
    /// - Y plane  : `width * height` bytes
    /// - Cb plane : `(width/2) * (height/2)` bytes
    /// - Cr plane : `(width/2) * (height/2)` bytes
    ///
    /// `rgba` must have length `width * height * 4`.
    /// Returns `None` if the input length is unexpected.
    pub fn rgba_to_yuv420(&self, rgba: &[u8], width: u32, height: u32) -> Option<Vec<u8>> {
        let w = width as usize;
        let h = height as usize;
        if rgba.len() != w * h * 4 {
            return None;
        }

        let y_size = w * h;
        let uv_w = (w + 1) / 2;
        let uv_h = (h + 1) / 2;
        let uv_size = uv_w * uv_h;

        let mut out = vec![0u8; y_size + 2 * uv_size];
        let (y_plane, uv_rest) = out.split_at_mut(y_size);
        let (cb_plane, cr_plane) = uv_rest.split_at_mut(uv_size);

        // Unrolled Y-plane pass: process 8 pixels at a time where possible.
        let chunks = w * h;
        let chunk8 = chunks / 8;
        let remainder = chunks % 8;

        // Helper closure — avoids redundant indexing arithmetic in the loop.
        let sample_y = |idx: usize| -> u8 {
            let base = idx * 4;
            let r = rgba[base] as f32;
            let g = rgba[base + 1] as f32;
            let b = rgba[base + 2] as f32;
            let y = 0.299_f32 * r + 0.587_f32 * g + 0.114_f32 * b;
            y.round().clamp(0.0, 255.0) as u8
        };

        for i in 0..chunk8 {
            let base = i * 8;
            y_plane[base] = sample_y(base);
            y_plane[base + 1] = sample_y(base + 1);
            y_plane[base + 2] = sample_y(base + 2);
            y_plane[base + 3] = sample_y(base + 3);
            y_plane[base + 4] = sample_y(base + 4);
            y_plane[base + 5] = sample_y(base + 5);
            y_plane[base + 6] = sample_y(base + 6);
            y_plane[base + 7] = sample_y(base + 7);
        }
        let rem_start = chunk8 * 8;
        for i in 0..remainder {
            y_plane[rem_start + i] = sample_y(rem_start + i);
        }

        // Cb / Cr — 2×2 average subsampling.
        for block_y in 0..uv_h {
            for block_x in 0..uv_w {
                let mut sum_cb = 0.0_f32;
                let mut sum_cr = 0.0_f32;
                let mut count = 0_u32;

                for dy in 0..2_usize {
                    let sy = block_y * 2 + dy;
                    if sy >= h {
                        continue;
                    }
                    for dx in 0..2_usize {
                        let sx = block_x * 2 + dx;
                        if sx >= w {
                            continue;
                        }
                        let base = (sy * w + sx) * 4;
                        let r = rgba[base] as f32;
                        let g = rgba[base + 1] as f32;
                        let b = rgba[base + 2] as f32;
                        sum_cb += -0.168_736_f32 * r - 0.331_264_f32 * g + 0.5_f32 * b + 128.0;
                        sum_cr += 0.5_f32 * r - 0.418_688_f32 * g - 0.081_312_f32 * b + 128.0;
                        count += 1;
                    }
                }

                let uv_idx = block_y * uv_w + block_x;
                if count > 0 {
                    cb_plane[uv_idx] = (sum_cb / count as f32).round().clamp(0.0, 255.0) as u8;
                    cr_plane[uv_idx] = (sum_cr / count as f32).round().clamp(0.0, 255.0) as u8;
                }
            }
        }

        Some(out)
    }

    // -----------------------------------------------------------------------
    // YUV 420 → RGBA
    // -----------------------------------------------------------------------

    /// Convert planar YUV 420 to packed RGBA.
    ///
    /// Expects the same memory layout as produced by `rgba_to_yuv420`.
    /// Returns `None` if the input length is unexpected.
    pub fn yuv420_to_rgba(&self, yuv: &[u8], width: u32, height: u32) -> Option<Vec<u8>> {
        let w = width as usize;
        let h = height as usize;
        let y_size = w * h;
        let uv_w = (w + 1) / 2;
        let uv_h = (h + 1) / 2;
        let uv_size = uv_w * uv_h;
        let expected = y_size + 2 * uv_size;

        if yuv.len() != expected {
            return None;
        }

        let y_plane = &yuv[..y_size];
        let cb_plane = &yuv[y_size..y_size + uv_size];
        let cr_plane = &yuv[y_size + uv_size..];

        let mut rgba = vec![0u8; w * h * 4];

        // Process 4 pixels at a time for auto-vectorisation.
        let total_pixels = w * h;
        let chunk4 = total_pixels / 4;
        let rem4 = total_pixels % 4;

        let convert_pixel = |pix_idx: usize, out: &mut [u8]| {
            let py = pix_idx / w;
            let px = pix_idx % w;
            let uv_x = px / 2;
            let uv_y = py / 2;
            let uv_idx = uv_y * uv_w + uv_x;

            let yv = y_plane[pix_idx] as f32;
            let cb = cb_plane[uv_idx] as f32 - 128.0;
            let cr = cr_plane[uv_idx] as f32 - 128.0;

            let r = (yv + 1.402_f32 * cr).round().clamp(0.0, 255.0) as u8;
            let g = (yv - 0.344_136_f32 * cb - 0.714_136_f32 * cr)
                .round()
                .clamp(0.0, 255.0) as u8;
            let b = (yv + 1.772_f32 * cb).round().clamp(0.0, 255.0) as u8;

            let base = pix_idx * 4;
            out[base] = r;
            out[base + 1] = g;
            out[base + 2] = b;
            out[base + 3] = 255;
        };

        for i in 0..chunk4 {
            let base = i * 4;
            convert_pixel(base, &mut rgba);
            convert_pixel(base + 1, &mut rgba);
            convert_pixel(base + 2, &mut rgba);
            convert_pixel(base + 3, &mut rgba);
        }
        let rem_start = chunk4 * 4;
        for i in 0..rem4 {
            convert_pixel(rem_start + i, &mut rgba);
        }

        Some(rgba)
    }

    // -----------------------------------------------------------------------
    // Gaussian blur (separable)
    // -----------------------------------------------------------------------

    /// Apply a separable Gaussian blur.
    ///
    /// Kernel radius is `ceil(3 * sigma)`.  Each pixel channel is treated as
    /// an independent `f32` sample (grayscale or multi-channel flattened).
    /// Returns `None` if the input length doesn't match `width * height`.
    pub fn gaussian_blur(
        &self,
        pixels: &[f32],
        width: u32,
        height: u32,
        sigma: f32,
    ) -> Option<Vec<f32>> {
        let w = width as usize;
        let h = height as usize;
        if pixels.len() != w * h {
            return None;
        }

        if sigma <= 0.0 {
            return Some(pixels.to_vec());
        }

        let radius = (3.0 * sigma).ceil() as usize;
        let kernel = build_gaussian_kernel_1d(radius, sigma);

        // --- Horizontal pass ---
        let mut tmp = vec![0.0_f32; w * h];
        for row in 0..h {
            let row_start = row * w;
            for col in 0..w {
                let mut acc = 0.0_f32;
                let mut weight_sum = 0.0_f32;
                for ki in 0..kernel.len() {
                    let koff = ki as isize - radius as isize;
                    let src_col = col as isize + koff;
                    if src_col >= 0 && src_col < w as isize {
                        let k = kernel[ki];
                        acc += pixels[row_start + src_col as usize] * k;
                        weight_sum += k;
                    }
                }
                tmp[row_start + col] = if weight_sum > 0.0 {
                    acc / weight_sum
                } else {
                    0.0
                };
            }
        }

        // --- Vertical pass ---
        let mut out = vec![0.0_f32; w * h];
        for col in 0..w {
            for row in 0..h {
                let mut acc = 0.0_f32;
                let mut weight_sum = 0.0_f32;
                for ki in 0..kernel.len() {
                    let koff = ki as isize - radius as isize;
                    let src_row = row as isize + koff;
                    if src_row >= 0 && src_row < h as isize {
                        let k = kernel[ki];
                        acc += tmp[src_row as usize * w + col] * k;
                        weight_sum += k;
                    }
                }
                out[row * w + col] = if weight_sum > 0.0 {
                    acc / weight_sum
                } else {
                    0.0
                };
            }
        }

        Some(out)
    }

    // -----------------------------------------------------------------------
    // Sobel edge detection
    // -----------------------------------------------------------------------

    /// Compute Sobel gradient magnitude for a grayscale image.
    ///
    /// Input `gray` must have length `width * height`.
    /// Returns `None` if length mismatch.  Border pixels are set to 0.
    pub fn sobel_edges(&self, gray: &[f32], width: u32, height: u32) -> Option<Vec<f32>> {
        let w = width as usize;
        let h = height as usize;
        if gray.len() != w * h {
            return None;
        }

        let mut out = vec![0.0_f32; w * h];

        // Kernels:
        // Gx = [[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]]
        // Gy = [[-1, -2, -1], [0, 0, 0], [+1, +2, +1]]
        for row in 1..h.saturating_sub(1) {
            let row_base = row * w;
            for col in 1..w.saturating_sub(1) {
                let tl = gray[(row - 1) * w + (col - 1)];
                let tc = gray[(row - 1) * w + col];
                let tr = gray[(row - 1) * w + (col + 1)];
                let ml = gray[row * w + (col - 1)];
                let mr = gray[row * w + (col + 1)];
                let bl = gray[(row + 1) * w + (col - 1)];
                let bc = gray[(row + 1) * w + col];
                let br = gray[(row + 1) * w + (col + 1)];

                let gx = -tl + tr - 2.0 * ml + 2.0 * mr - bl + br;
                let gy = -tl - 2.0 * tc - tr + bl + 2.0 * bc + br;

                out[row_base + col] = (gx * gx + gy * gy).sqrt();
            }
        }

        Some(out)
    }

    // -----------------------------------------------------------------------
    // Histogram equalization
    // -----------------------------------------------------------------------

    /// Apply histogram equalization to an 8-bit grayscale image.
    ///
    /// Returns `None` if `gray.len() != width * height`.
    pub fn histogram_equalization(&self, gray: &[u8], width: u32, height: u32) -> Option<Vec<u8>> {
        let n = width as usize * height as usize;
        if gray.len() != n {
            return None;
        }

        // Build histogram.
        let mut hist = [0u64; 256];
        for &px in gray {
            hist[px as usize] += 1;
        }

        // CDF.
        let mut cdf = [0u64; 256];
        cdf[0] = hist[0];
        for i in 1..256 {
            cdf[i] = cdf[i - 1] + hist[i];
        }

        let cdf_min = cdf.iter().copied().find(|&v| v > 0).unwrap_or(0);
        let total = n as u64;

        // Lookup table.
        let lut: Vec<u8> = (0..256)
            .map(|i| {
                if total == cdf_min {
                    i as u8
                } else {
                    let v = (cdf[i] - cdf_min) as f64 * 255.0 / (total - cdf_min) as f64;
                    v.round().clamp(0.0, 255.0) as u8
                }
            })
            .collect();

        Some(gray.iter().map(|&px| lut[px as usize]).collect())
    }

    // -----------------------------------------------------------------------
    // Otsu thresholding
    // -----------------------------------------------------------------------

    /// Compute Otsu's optimal threshold and produce a binary image.
    ///
    /// Returns `(threshold, binary_image)` or `None` on size mismatch.
    /// Binary output: 0 for below-threshold, 255 for at/above.
    pub fn threshold_otsu(&self, gray: &[u8], width: u32, height: u32) -> Option<(u8, Vec<u8>)> {
        let n = width as usize * height as usize;
        if gray.len() != n {
            return None;
        }

        let mut hist = [0u64; 256];
        for &px in gray {
            hist[px as usize] += 1;
        }

        let total = n as f64;
        let mut sum_total = 0.0_f64;
        for i in 0..256_usize {
            sum_total += i as f64 * hist[i] as f64;
        }

        let mut sum_b = 0.0_f64;
        let mut w_b = 0.0_f64;
        let mut max_var = 0.0_f64;
        let mut threshold = 0u8;

        for i in 0..256_usize {
            w_b += hist[i] as f64;
            if w_b == 0.0 {
                continue;
            }
            let w_f = total - w_b;
            if w_f == 0.0 {
                break;
            }

            sum_b += i as f64 * hist[i] as f64;
            let m_b = sum_b / w_b;
            let m_f = (sum_total - sum_b) / w_f;
            let diff = m_b - m_f;
            let between_var = w_b * w_f * diff * diff;

            if between_var > max_var {
                max_var = between_var;
                threshold = i as u8;
            }
        }

        let binary: Vec<u8> = gray
            .iter()
            .map(|&px| if px > threshold { 255 } else { 0 })
            .collect();

        Some((threshold, binary))
    }

    // -----------------------------------------------------------------------
    // Alpha compositing (Porter-Duff "over")
    // -----------------------------------------------------------------------

    /// Composite `fg` over `bg` using the Porter-Duff "over" operator.
    ///
    /// Both buffers must be RGBA, length `width * height * 4`.
    /// Returns `None` on size mismatch.
    pub fn alpha_composite(
        &self,
        fg: &[u8],
        bg: &[u8],
        width: u32,
        height: u32,
    ) -> Option<Vec<u8>> {
        let n = width as usize * height as usize;
        let expected = n * 4;
        if fg.len() != expected || bg.len() != expected {
            return None;
        }

        let mut out = vec![0u8; expected];
        let chunk_size = self.config.chunk_size.max(8) / 4 * 4; // keep multiple of 4

        let chunks = n / (chunk_size / 4);
        let rem = n % (chunk_size / 4);

        let composite_pixel = |i: usize, out: &mut [u8]| {
            let base = i * 4;
            let fa = fg[base + 3] as f32 / 255.0;
            let ba = bg[base + 3] as f32 / 255.0;
            let out_a = fa + ba * (1.0 - fa);
            if out_a <= 0.0 {
                return;
            }
            let inv_out = 1.0 / out_a;
            out[base] = ((fg[base] as f32 * fa + bg[base] as f32 * ba * (1.0 - fa)) * inv_out)
                .round()
                .clamp(0.0, 255.0) as u8;
            out[base + 1] = ((fg[base + 1] as f32 * fa + bg[base + 1] as f32 * ba * (1.0 - fa))
                * inv_out)
                .round()
                .clamp(0.0, 255.0) as u8;
            out[base + 2] = ((fg[base + 2] as f32 * fa + bg[base + 2] as f32 * ba * (1.0 - fa))
                * inv_out)
                .round()
                .clamp(0.0, 255.0) as u8;
            out[base + 3] = (out_a * 255.0).round().clamp(0.0, 255.0) as u8;
        };

        let pixels_per_chunk = chunk_size / 4;
        for c in 0..chunks {
            let start = c * pixels_per_chunk;
            for p in 0..pixels_per_chunk {
                composite_pixel(start + p, &mut out);
            }
        }
        let rem_start = chunks * pixels_per_chunk;
        for p in 0..rem {
            composite_pixel(rem_start + p, &mut out);
        }

        Some(out)
    }

    // -----------------------------------------------------------------------
    // Bilinear image scaling
    // -----------------------------------------------------------------------

    /// Scale an RGBA image using bilinear interpolation.
    ///
    /// Input `pixels` must be packed RGBA with length `src_w * src_h * 4`.
    /// Returns `None` on size mismatch or zero dimensions.
    pub fn scale_image(
        &self,
        pixels: &[u8],
        src_w: u32,
        src_h: u32,
        dst_w: u32,
        dst_h: u32,
    ) -> Option<Vec<u8>> {
        let sw = src_w as usize;
        let sh = src_h as usize;
        let dw = dst_w as usize;
        let dh = dst_h as usize;

        if sw == 0 || sh == 0 || dw == 0 || dh == 0 {
            return None;
        }
        if pixels.len() != sw * sh * 4 {
            return None;
        }

        let mut out = vec![0u8; dw * dh * 4];

        let x_scale = sw as f32 / dw as f32;
        let y_scale = sh as f32 / dh as f32;

        for dy in 0..dh {
            // Continuous source y coordinate (centre-of-pixel mapping).
            let src_y = (dy as f32 + 0.5) * y_scale - 0.5;
            let y0 = (src_y.floor() as isize).clamp(0, sh as isize - 1) as usize;
            let y1 = (y0 + 1).min(sh - 1);
            let ty = (src_y - src_y.floor()).max(0.0).min(1.0);

            for dx in 0..dw {
                let src_x = (dx as f32 + 0.5) * x_scale - 0.5;
                let x0 = (src_x.floor() as isize).clamp(0, sw as isize - 1) as usize;
                let x1 = (x0 + 1).min(sw - 1);
                let tx = (src_x - src_x.floor()).max(0.0).min(1.0);

                let i00 = (y0 * sw + x0) * 4;
                let i10 = (y0 * sw + x1) * 4;
                let i01 = (y1 * sw + x0) * 4;
                let i11 = (y1 * sw + x1) * 4;

                let dst_base = (dy * dw + dx) * 4;

                // Unrolled over 4 channels.
                out[dst_base] =
                    bilinear_u8(pixels[i00], pixels[i10], pixels[i01], pixels[i11], tx, ty);
                out[dst_base + 1] = bilinear_u8(
                    pixels[i00 + 1],
                    pixels[i10 + 1],
                    pixels[i01 + 1],
                    pixels[i11 + 1],
                    tx,
                    ty,
                );
                out[dst_base + 2] = bilinear_u8(
                    pixels[i00 + 2],
                    pixels[i10 + 2],
                    pixels[i01 + 2],
                    pixels[i11 + 2],
                    tx,
                    ty,
                );
                out[dst_base + 3] = bilinear_u8(
                    pixels[i00 + 3],
                    pixels[i10 + 3],
                    pixels[i01 + 3],
                    pixels[i11 + 3],
                    tx,
                    ty,
                );
            }
        }

        Some(out)
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Build a 1-D Gaussian kernel of radius `r` (length = 2r+1).
fn build_gaussian_kernel_1d(radius: usize, sigma: f32) -> Vec<f32> {
    let len = 2 * radius + 1;
    let mut k = Vec::with_capacity(len);
    let two_sigma_sq = 2.0 * sigma * sigma;
    let mut sum = 0.0_f32;
    for i in 0..len {
        let x = (i as isize - radius as isize) as f32;
        let v = (-x * x / two_sigma_sq).exp();
        k.push(v);
        sum += v;
    }
    if sum > 0.0 {
        for v in &mut k {
            *v /= sum;
        }
    }
    k
}

/// Bilinear interpolation for a single `u8` channel.
#[inline(always)]
fn bilinear_u8(c00: u8, c10: u8, c01: u8, c11: u8, tx: f32, ty: f32) -> u8 {
    let v00 = c00 as f32;
    let v10 = c10 as f32;
    let v01 = c01 as f32;
    let v11 = c11 as f32;
    let top = v00 + (v10 - v00) * tx;
    let bottom = v01 + (v11 - v01) * tx;
    (top + (bottom - top) * ty).round().clamp(0.0, 255.0) as u8
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_kernel() -> ComputeKernel {
        ComputeKernel::default_config()
    }

    // --- rgba_to_yuv420 ---

    #[test]
    fn test_rgba_to_yuv420_size() {
        let kernel = make_kernel();
        // 4×4 image: each pixel = [100, 149, 237, 255]
        let rgba: Vec<u8> = (0..16).flat_map(|_| [100u8, 149, 237, 255]).collect();
        let yuv = kernel
            .rgba_to_yuv420(&rgba, 4, 4)
            .expect("conversion failed");
        assert_eq!(yuv.len(), 4 * 4 + 2 * 2 * 2); // Y + 2*(2*2)
    }

    #[test]
    fn test_rgba_to_yuv420_invalid_size() {
        let kernel = make_kernel();
        let rgba = vec![0u8; 10]; // wrong size
        assert!(kernel.rgba_to_yuv420(&rgba, 4, 4).is_none());
    }

    #[test]
    fn test_rgba_to_yuv420_white_pixel() {
        let kernel = make_kernel();
        // 2×2 white pixels
        let rgba: Vec<u8> = (0..4).flat_map(|_| [255u8, 255, 255, 255]).collect();
        let yuv = kernel
            .rgba_to_yuv420(&rgba, 2, 2)
            .expect("conversion failed");
        // Y for white ≈ 255
        assert!(yuv[0] > 230, "Y for white should be ≈ 255, got {}", yuv[0]);
    }

    #[test]
    fn test_rgba_to_yuv420_black_pixel() {
        let kernel = make_kernel();
        // 2×2 black pixels
        let rgba: Vec<u8> = (0..4).flat_map(|_| [0u8, 0, 0, 255]).collect();
        let yuv = kernel
            .rgba_to_yuv420(&rgba, 2, 2)
            .expect("conversion failed");
        assert_eq!(yuv[0], 0, "Y for black should be 0");
    }

    // --- yuv420_to_rgba ---

    #[test]
    fn test_yuv420_roundtrip() {
        let kernel = make_kernel();
        // Build a simple 4×4 RGBA image (mid-grey).
        let rgba_in: Vec<u8> = (0..16).flat_map(|_| [128u8, 128, 128, 255]).collect();
        let yuv = kernel.rgba_to_yuv420(&rgba_in, 4, 4).expect("to_yuv");
        let rgba_out = kernel.yuv420_to_rgba(&yuv, 4, 4).expect("to_rgba");
        // Round-trip: each channel should be within ±4 due to quantisation.
        for i in (0..rgba_out.len()).step_by(4) {
            let diff = (rgba_in[i] as i32 - rgba_out[i] as i32).unsigned_abs();
            assert!(diff <= 4, "channel diff too large: {diff}");
        }
    }

    #[test]
    fn test_yuv420_to_rgba_invalid_size() {
        let kernel = make_kernel();
        let bad = vec![0u8; 5];
        assert!(kernel.yuv420_to_rgba(&bad, 4, 4).is_none());
    }

    // --- gaussian_blur ---

    #[test]
    fn test_gaussian_blur_flat_image() {
        let kernel = make_kernel();
        let pixels = vec![1.0_f32; 8 * 8];
        let blurred = kernel.gaussian_blur(&pixels, 8, 8, 1.0).expect("blur");
        // Blurring a constant image should leave it unchanged.
        for &v in &blurred {
            assert!((v - 1.0).abs() < 1e-4, "expected ~1.0 got {v}");
        }
    }

    #[test]
    fn test_gaussian_blur_zero_sigma() {
        let kernel = make_kernel();
        let pixels = vec![0.5_f32; 4 * 4];
        let out = kernel.gaussian_blur(&pixels, 4, 4, 0.0).expect("blur");
        // sigma=0 → identity
        for &v in &out {
            assert!((v - 0.5).abs() < 1e-5);
        }
    }

    #[test]
    fn test_gaussian_blur_invalid_size() {
        let kernel = make_kernel();
        let pixels = vec![0.0_f32; 3];
        assert!(kernel.gaussian_blur(&pixels, 4, 4, 1.0).is_none());
    }

    // --- sobel_edges ---

    #[test]
    fn test_sobel_flat_image_is_zero() {
        let kernel = make_kernel();
        let gray = vec![0.5_f32; 8 * 8];
        let edges = kernel.sobel_edges(&gray, 8, 8).expect("sobel");
        // Interior pixels of a flat image → gradient = 0.
        for row in 1..7_usize {
            for col in 1..7_usize {
                let v = edges[row * 8 + col];
                assert!(v.abs() < 1e-5, "expected 0 at ({row},{col}), got {v}");
            }
        }
    }

    #[test]
    fn test_sobel_vertical_edge() {
        let kernel = make_kernel();
        // Left half = 0, right half = 1 → strong vertical edge in the middle.
        let mut gray = vec![0.0_f32; 8 * 8];
        for row in 0..8_usize {
            for col in 4..8_usize {
                gray[row * 8 + col] = 1.0;
            }
        }
        let edges = kernel.sobel_edges(&gray, 8, 8).expect("sobel");
        // The column just at the boundary (col=3 or col=4, row interior) should
        // have a non-zero gradient.
        let mid_val = edges[3 * 8 + 3];
        assert!(mid_val > 0.1, "expected edge at boundary, got {mid_val}");
    }

    // --- histogram_equalization ---

    #[test]
    fn test_histogram_equalization_constant() {
        let kernel = make_kernel();
        let gray = vec![100u8; 4 * 4];
        let out = kernel.histogram_equalization(&gray, 4, 4).expect("eq");
        // With constant input, all output values should be the same.
        let first = out[0];
        for &v in &out {
            assert_eq!(v, first);
        }
    }

    // --- threshold_otsu ---

    #[test]
    fn test_threshold_otsu_bimodal() {
        let kernel = make_kernel();
        // Two classes: 50 pixels at value 30 (dark), 50 pixels at value 200 (bright).
        // Otsu's threshold will be 30 (the dark class value) because the maximum
        // between-class variance occurs at the boundary between the two modes.
        // With `px > threshold` classification: 30 → 0 (bg), 200 → 255 (fg).
        let mut gray = vec![30u8; 50];
        gray.extend_from_slice(&[200u8; 50]);
        let (thresh, binary) = kernel.threshold_otsu(&gray, 10, 10).expect("otsu");
        // Threshold should be at the dark-class value (30).
        assert!(
            thresh < 200,
            "threshold {thresh} must be less than bright class value 200"
        );
        // Dark pixels should map to background (0), bright to foreground (255).
        assert_eq!(binary[0], 0, "dark pixel (value 30) should be background");
        assert_eq!(
            binary[50], 255,
            "bright pixel (value 200) should be foreground"
        );
    }

    // --- alpha_composite ---

    #[test]
    fn test_alpha_composite_opaque_fg() {
        let kernel = make_kernel();
        // Fully opaque red fg over any bg → output = red.
        let fg: Vec<u8> = (0..4).flat_map(|_| [255u8, 0, 0, 255]).collect();
        let bg: Vec<u8> = (0..4).flat_map(|_| [0u8, 0, 255, 255]).collect();
        let out = kernel.alpha_composite(&fg, &bg, 2, 2).expect("composite");
        assert_eq!(&out[0..4], &[255u8, 0, 0, 255]);
    }

    #[test]
    fn test_alpha_composite_transparent_fg() {
        let kernel = make_kernel();
        // Fully transparent fg → output = bg.
        let fg: Vec<u8> = (0..4).flat_map(|_| [255u8, 0, 0, 0u8]).collect();
        let bg: Vec<u8> = (0..4).flat_map(|_| [0u8, 0, 255, 255]).collect();
        let out = kernel.alpha_composite(&fg, &bg, 2, 2).expect("composite");
        assert_eq!(&out[0..4], &[0u8, 0, 255, 255]);
    }

    #[test]
    fn test_alpha_composite_size_mismatch() {
        let kernel = make_kernel();
        let fg = vec![0u8; 8];
        let bg = vec![0u8; 16];
        assert!(kernel.alpha_composite(&fg, &bg, 2, 2).is_none());
    }

    // --- scale_image ---

    #[test]
    fn test_scale_image_identity() {
        let kernel = make_kernel();
        let pixels: Vec<u8> = (0..16)
            .flat_map(|i: u8| [i * 4, i * 4, i * 4, 255])
            .collect();
        let out = kernel.scale_image(&pixels, 4, 4, 4, 4).expect("scale");
        assert_eq!(out, pixels);
    }

    #[test]
    fn test_scale_image_upscale_size() {
        let kernel = make_kernel();
        let pixels = vec![128u8; 4 * 4 * 4]; // 4×4 grey
        let out = kernel.scale_image(&pixels, 4, 4, 8, 8).expect("scale");
        assert_eq!(out.len(), 8 * 8 * 4);
    }

    #[test]
    fn test_scale_image_zero_dimension() {
        let kernel = make_kernel();
        let pixels = vec![0u8; 4 * 4 * 4];
        assert!(kernel.scale_image(&pixels, 4, 4, 0, 8).is_none());
    }
}
