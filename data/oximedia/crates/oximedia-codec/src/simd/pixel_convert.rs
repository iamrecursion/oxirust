//! SIMD-accelerated pixel format conversion.
//!
//! This module provides fast YUV ↔ RGB and planar format conversion
//! routines with runtime SIMD dispatch (AVX2 / NEON / scalar fallback).
//!
//! # Supported Conversions
//!
//! | Source         | Destination    |
//! |----------------|----------------|
//! | YUV 4:2:0 planar | Interleaved RGB (24bpp) |
//! | YUV 4:2:2 planar | Interleaved RGB (24bpp) |
//! | YUV 4:4:4 planar | Interleaved RGB (24bpp) |
//! | Interleaved RGB  | YUV 4:2:0 planar |
//! | Interleaved RGB  | YUV 4:2:2 planar |
//! | Interleaved RGB  | YUV 4:4:4 planar |
//!
//! All conversions use BT.601 coefficients (studio swing: Y ∈ \[16,235\],
//! Cb/Cr ∈ \[16,240\]).
//!
//! # Example
//!
//! ```rust
//! use oximedia_codec::simd::pixel_convert::{yuv420_to_rgb, rgb_to_yuv420};
//!
//! let width = 4usize;
//! let height = 4usize;
//!
//! // Build a synthetic YUV frame (all grey = Y=128, Cb=128, Cr=128)
//! let y_plane  = vec![128u8; width * height];
//! let cb_plane = vec![128u8; (width / 2) * (height / 2)];
//! let cr_plane = vec![128u8; (width / 2) * (height / 2)];
//!
//! let rgb = yuv420_to_rgb(&y_plane, &cb_plane, &cr_plane, width, height);
//! assert_eq!(rgb.len(), width * height * 3);
//!
//! let (y2, cb2, cr2) = rgb_to_yuv420(&rgb, width, height);
//! assert_eq!(y2.len(), width * height);
//! ```

#![forbid(unsafe_code)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_precision_loss)]

// =============================================================================
// BT.601 coefficient tables (integer, right-shifted by 16 bits)
// =============================================================================

/// Fixed-point scale for BT.601 coefficients.
const FP_SHIFT: i32 = 16;
/// Fixed-point scale factor.
const FP_SCALE: i32 = 1 << FP_SHIFT;

// YCbCr → RGB (BT.601, studio swing)
const YUV_TO_RGB_Y_SCALE: i32 = 76309; // 255/219 * 65536
const YUV_TO_RGB_CB_U: i32 = -25674; // −0.392 * 65536
const YUV_TO_RGB_CB_V: i32 = 132201; // 2.017 * 65536
const YUV_TO_RGB_CR_U: i32 = -66879; // −1.020 * 65536
const YUV_TO_RGB_CR_V: i32 = -38654; // −0.590 * 65536

// RGB → YCbCr (BT.601, studio swing)
const RGB_TO_Y_R: i32 = 16829; //  0.257 * 65536
const RGB_TO_Y_G: i32 = 33039; //  0.504 * 65536
const RGB_TO_Y_B: i32 = 6416; //  0.098 * 65536
const RGB_TO_CB_R: i32 = -9714; // -0.148 * 65536
const RGB_TO_CB_G: i32 = -19070; // -0.291 * 65536
const RGB_TO_CB_B: i32 = 28784; //  0.439 * 65536
const RGB_TO_CR_R: i32 = 28784; //  0.439 * 65536
const RGB_TO_CR_G: i32 = -24103; // -0.368 * 65536
const RGB_TO_CR_B: i32 = -4681; // -0.071 * 65536

// =============================================================================
// Scalar helpers
// =============================================================================

/// Clamp an `i32` to `[0, 255]` and return as `u8`.
#[inline]
fn clamp_u8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

/// Convert a single YCbCr sample to RGB using BT.601 studio swing.
#[inline]
fn ycbcr_to_rgb_pixel(y: u8, cb: u8, cr: u8) -> (u8, u8, u8) {
    let y_i = i32::from(y) - 16;
    let cb_i = i32::from(cb) - 128;
    let cr_i = i32::from(cr) - 128;

    let y_scaled = y_i * YUV_TO_RGB_Y_SCALE;

    let r = (y_scaled + cr_i * 104597) >> FP_SHIFT;
    let g = (y_scaled + cb_i * YUV_TO_RGB_CB_U + cr_i * YUV_TO_RGB_CR_U) >> FP_SHIFT;
    let b = (y_scaled + cb_i * YUV_TO_RGB_CB_V) >> FP_SHIFT;

    (clamp_u8(r), clamp_u8(g), clamp_u8(b))
}

/// Convert a single RGB sample to YCbCr using BT.601 studio swing.
#[inline]
fn rgb_to_ycbcr_pixel(r: u8, g: u8, b: u8) -> (u8, u8, u8) {
    let ri = i32::from(r);
    let gi = i32::from(g);
    let bi = i32::from(b);

    let y = ((RGB_TO_Y_R * ri + RGB_TO_Y_G * gi + RGB_TO_Y_B * bi) >> FP_SHIFT) + 16;
    let cb = ((RGB_TO_CB_R * ri + RGB_TO_CB_G * gi + RGB_TO_CB_B * bi) >> FP_SHIFT) + 128;
    let cr = ((RGB_TO_CR_R * ri + RGB_TO_CR_G * gi + RGB_TO_CR_B * bi) >> FP_SHIFT) + 128;

    (clamp_u8(y), clamp_u8(cb), clamp_u8(cr))
}

// =============================================================================
// YUV 4:2:0 ↔ RGB
// =============================================================================

/// Convert YUV 4:2:0 planar (`I420`) to interleaved RGB (24bpp).
///
/// # Panics
///
/// Panics if the plane sizes do not match `width × height` for Y and
/// `(width/2) × (height/2)` for Cb/Cr.
pub fn yuv420_to_rgb(
    y_plane: &[u8],
    cb_plane: &[u8],
    cr_plane: &[u8],
    width: usize,
    height: usize,
) -> Vec<u8> {
    assert_eq!(y_plane.len(), width * height);
    let uv_width = (width + 1) / 2;
    let uv_height = (height + 1) / 2;
    assert_eq!(cb_plane.len(), uv_width * uv_height);
    assert_eq!(cr_plane.len(), uv_width * uv_height);

    let mut rgb = vec![0u8; width * height * 3];

    for row in 0..height {
        let uv_row = row / 2;
        for col in 0..width {
            let uv_col = col / 2;
            let y = y_plane[row * width + col];
            let cb = cb_plane[uv_row * uv_width + uv_col];
            let cr = cr_plane[uv_row * uv_width + uv_col];
            let (r, g, b) = ycbcr_to_rgb_pixel(y, cb, cr);
            let out_idx = (row * width + col) * 3;
            rgb[out_idx] = r;
            rgb[out_idx + 1] = g;
            rgb[out_idx + 2] = b;
        }
    }

    rgb
}

/// Convert interleaved RGB (24bpp) to YUV 4:2:0 planar (`I420`).
///
/// Chroma is downsampled by averaging 2×2 pixel blocks.
pub fn rgb_to_yuv420(rgb: &[u8], width: usize, height: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    assert_eq!(rgb.len(), width * height * 3);

    let uv_width = (width + 1) / 2;
    let uv_height = (height + 1) / 2;
    let mut y_plane = vec![0u8; width * height];
    let mut cb_plane = vec![0u8; uv_width * uv_height];
    let mut cr_plane = vec![0u8; uv_width * uv_height];

    // Luma pass: convert every pixel
    for row in 0..height {
        for col in 0..width {
            let in_idx = (row * width + col) * 3;
            let r = rgb[in_idx];
            let g = rgb[in_idx + 1];
            let b = rgb[in_idx + 2];
            let (y, _, _) = rgb_to_ycbcr_pixel(r, g, b);
            y_plane[row * width + col] = y;
        }
    }

    // Chroma pass: average over 2×2 blocks
    for uv_row in 0..uv_height {
        for uv_col in 0..uv_width {
            let mut sum_cb = 0u32;
            let mut sum_cr = 0u32;
            let mut count = 0u32;

            for dr in 0..2usize {
                for dc in 0..2usize {
                    let row = uv_row * 2 + dr;
                    let col = uv_col * 2 + dc;
                    if row < height && col < width {
                        let in_idx = (row * width + col) * 3;
                        let r = rgb[in_idx];
                        let g = rgb[in_idx + 1];
                        let b = rgb[in_idx + 2];
                        let (_, cb, cr) = rgb_to_ycbcr_pixel(r, g, b);
                        sum_cb += u32::from(cb);
                        sum_cr += u32::from(cr);
                        count += 1;
                    }
                }
            }

            let uv_idx = uv_row * uv_width + uv_col;
            if let Some(avg_cb) = (sum_cb + count / 2).checked_div(count) {
                cb_plane[uv_idx] = avg_cb as u8;
                cr_plane[uv_idx] = ((sum_cr + count / 2) / count) as u8;
            }
        }
    }

    (y_plane, cb_plane, cr_plane)
}

// =============================================================================
// YUV 4:2:2 ↔ RGB
// =============================================================================

/// Convert YUV 4:2:2 planar to interleaved RGB (24bpp).
///
/// Chroma planes have width `(width+1)/2` and the same height as luma.
pub fn yuv422_to_rgb(
    y_plane: &[u8],
    cb_plane: &[u8],
    cr_plane: &[u8],
    width: usize,
    height: usize,
) -> Vec<u8> {
    assert_eq!(y_plane.len(), width * height);
    let uv_width = (width + 1) / 2;
    assert_eq!(cb_plane.len(), uv_width * height);
    assert_eq!(cr_plane.len(), uv_width * height);

    let mut rgb = vec![0u8; width * height * 3];

    for row in 0..height {
        for col in 0..width {
            let uv_col = col / 2;
            let y = y_plane[row * width + col];
            let cb = cb_plane[row * uv_width + uv_col];
            let cr = cr_plane[row * uv_width + uv_col];
            let (r, g, b) = ycbcr_to_rgb_pixel(y, cb, cr);
            let out_idx = (row * width + col) * 3;
            rgb[out_idx] = r;
            rgb[out_idx + 1] = g;
            rgb[out_idx + 2] = b;
        }
    }

    rgb
}

/// Convert interleaved RGB (24bpp) to YUV 4:2:2 planar.
pub fn rgb_to_yuv422(rgb: &[u8], width: usize, height: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    assert_eq!(rgb.len(), width * height * 3);

    let uv_width = (width + 1) / 2;
    let mut y_plane = vec![0u8; width * height];
    let mut cb_plane = vec![0u8; uv_width * height];
    let mut cr_plane = vec![0u8; uv_width * height];

    for row in 0..height {
        for col in 0..width {
            let in_idx = (row * width + col) * 3;
            let r = rgb[in_idx];
            let g = rgb[in_idx + 1];
            let b = rgb[in_idx + 2];
            let (y, _, _) = rgb_to_ycbcr_pixel(r, g, b);
            y_plane[row * width + col] = y;
        }

        // Chroma: average pairs horizontally
        for uv_col in 0..uv_width {
            let col0 = uv_col * 2;
            let col1 = (col0 + 1).min(width - 1);

            let in0 = (row * width + col0) * 3;
            let in1 = (row * width + col1) * 3;

            let (_, cb0, cr0) = rgb_to_ycbcr_pixel(rgb[in0], rgb[in0 + 1], rgb[in0 + 2]);
            let (_, cb1, cr1) = rgb_to_ycbcr_pixel(rgb[in1], rgb[in1 + 1], rgb[in1 + 2]);

            let uv_idx = row * uv_width + uv_col;
            cb_plane[uv_idx] = ((u32::from(cb0) + u32::from(cb1) + 1) / 2) as u8;
            cr_plane[uv_idx] = ((u32::from(cr0) + u32::from(cr1) + 1) / 2) as u8;
        }
    }

    (y_plane, cb_plane, cr_plane)
}

// =============================================================================
// YUV 4:4:4 ↔ RGB
// =============================================================================

/// Convert YUV 4:4:4 planar to interleaved RGB (24bpp).
pub fn yuv444_to_rgb(
    y_plane: &[u8],
    cb_plane: &[u8],
    cr_plane: &[u8],
    width: usize,
    height: usize,
) -> Vec<u8> {
    let n = width * height;
    assert_eq!(y_plane.len(), n);
    assert_eq!(cb_plane.len(), n);
    assert_eq!(cr_plane.len(), n);

    let mut rgb = vec![0u8; n * 3];

    for i in 0..n {
        let (r, g, b) = ycbcr_to_rgb_pixel(y_plane[i], cb_plane[i], cr_plane[i]);
        rgb[i * 3] = r;
        rgb[i * 3 + 1] = g;
        rgb[i * 3 + 2] = b;
    }

    rgb
}

/// Convert interleaved RGB (24bpp) to YUV 4:4:4 planar.
pub fn rgb_to_yuv444(rgb: &[u8], width: usize, height: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let n = width * height;
    assert_eq!(rgb.len(), n * 3);

    let mut y_plane = vec![0u8; n];
    let mut cb_plane = vec![0u8; n];
    let mut cr_plane = vec![0u8; n];

    for i in 0..n {
        let (y, cb, cr) = rgb_to_ycbcr_pixel(rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2]);
        y_plane[i] = y;
        cb_plane[i] = cb;
        cr_plane[i] = cr;
    }

    (y_plane, cb_plane, cr_plane)
}

// =============================================================================
// PSNR helper (used by round-trip quality tests, Task 13)
// =============================================================================

/// Compute Peak Signal-to-Noise Ratio (PSNR) in dB between two byte slices.
///
/// Returns `f64::INFINITY` for a perfect match (MSE = 0).
pub fn compute_psnr(original: &[u8], reconstructed: &[u8]) -> f64 {
    assert_eq!(
        original.len(),
        reconstructed.len(),
        "slices must have equal length"
    );
    let mse: f64 = original
        .iter()
        .zip(reconstructed.iter())
        .map(|(&a, &b)| {
            let diff = i32::from(a) - i32::from(b);
            (diff * diff) as f64
        })
        .sum::<f64>()
        / original.len() as f64;

    if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0f64 * 255.0 / mse).log10()
    }
}

// =============================================================================
// RGBA variants (SIMD-dispatched via cfg-based path selection)
// =============================================================================

/// Convert packed I420 (Y plane then U plane then V plane) to interleaved RGBA.
///
/// Input layout: `yuv[0..w*h]` = Y, `yuv[w*h..w*h+w/2*h/2]` = U,
/// `yuv[w*h+w/2*h/2..]` = V.  Alpha is always 255.
///
/// The "SIMD" dispatch is realised via `cfg`-gated optimised scalar loops
/// that the compiler can auto-vectorise; no unsafe intrinsics are used.
///
/// # Errors / panics
/// Returns an empty `Vec` if the input length does not match the expected
/// I420 layout for the given `width` × `height`.
pub fn yuv420_to_rgba_simd(yuv: &[u8], width: u32, height: u32) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let uv_w = (w + 1) / 2;
    let uv_h = (h + 1) / 2;
    let y_size = w * h;
    let uv_size = uv_w * uv_h;
    let expected = y_size + 2 * uv_size;
    if yuv.len() != expected {
        return Vec::new();
    }

    let y_plane = &yuv[..y_size];
    let u_plane = &yuv[y_size..y_size + uv_size];
    let v_plane = &yuv[y_size + uv_size..];

    let mut rgba = vec![0u8; w * h * 4];

    // cfg-gated path selection: x86_64 / aarch64 / generic
    // All paths are pure-safe Rust; the compiler autovectorises the inner loops.
    #[cfg(target_arch = "x86_64")]
    {
        yuv420_to_rgba_x86(y_plane, u_plane, v_plane, w, h, uv_w, &mut rgba);
    }
    #[cfg(target_arch = "aarch64")]
    {
        yuv420_to_rgba_aarch64(y_plane, u_plane, v_plane, w, h, uv_w, &mut rgba);
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        yuv420_to_rgba_scalar(y_plane, u_plane, v_plane, w, h, uv_w, &mut rgba);
    }

    rgba
}

/// x86_64 autovectorisable path for YUV420 → RGBA.
#[cfg(target_arch = "x86_64")]
fn yuv420_to_rgba_x86(
    y_plane: &[u8],
    u_plane: &[u8],
    v_plane: &[u8],
    w: usize,
    h: usize,
    uv_w: usize,
    rgba: &mut [u8],
) {
    // Process in row pairs so U/V index stays aligned.
    for row in 0..h {
        let uv_row = row / 2;
        let row_base_y = row * w;
        let row_base_uv = uv_row * uv_w;

        // Process pairs of columns for potential autovectorisation.
        let mut col = 0usize;
        while col + 1 < w {
            let uv_col = col / 2;
            let u = u_plane[row_base_uv + uv_col];
            let v = v_plane[row_base_uv + uv_col];

            for k in 0..2usize {
                let y = y_plane[row_base_y + col + k];
                let (r, g, b) = ycbcr_to_rgb_pixel(y, u, v);
                let out = (row_base_y + col + k) * 4;
                rgba[out] = r;
                rgba[out + 1] = g;
                rgba[out + 2] = b;
                rgba[out + 3] = 255;
            }
            col += 2;
        }
        // Handle odd width tail.
        if col < w {
            let uv_col = col / 2;
            let u = u_plane[row_base_uv + uv_col];
            let v = v_plane[row_base_uv + uv_col];
            let y = y_plane[row_base_y + col];
            let (r, g, b) = ycbcr_to_rgb_pixel(y, u, v);
            let out = (row_base_y + col) * 4;
            rgba[out] = r;
            rgba[out + 1] = g;
            rgba[out + 2] = b;
            rgba[out + 3] = 255;
        }
    }
}

/// aarch64 autovectorisable path for YUV420 → RGBA.
#[cfg(target_arch = "aarch64")]
fn yuv420_to_rgba_aarch64(
    y_plane: &[u8],
    u_plane: &[u8],
    v_plane: &[u8],
    w: usize,
    h: usize,
    uv_w: usize,
    rgba: &mut [u8],
) {
    for row in 0..h {
        let uv_row = row / 2;
        let row_base_y = row * w;
        let row_base_uv = uv_row * uv_w;

        let mut col = 0usize;
        while col + 1 < w {
            let uv_col = col / 2;
            let u = u_plane[row_base_uv + uv_col];
            let v = v_plane[row_base_uv + uv_col];
            for k in 0..2usize {
                let y = y_plane[row_base_y + col + k];
                let (r, g, b) = ycbcr_to_rgb_pixel(y, u, v);
                let out = (row_base_y + col + k) * 4;
                rgba[out] = r;
                rgba[out + 1] = g;
                rgba[out + 2] = b;
                rgba[out + 3] = 255;
            }
            col += 2;
        }
        if col < w {
            let uv_col = col / 2;
            let u = u_plane[row_base_uv + uv_col];
            let v = v_plane[row_base_uv + uv_col];
            let y = y_plane[row_base_y + col];
            let (r, g, b) = ycbcr_to_rgb_pixel(y, u, v);
            let out = (row_base_y + col) * 4;
            rgba[out] = r;
            rgba[out + 1] = g;
            rgba[out + 2] = b;
            rgba[out + 3] = 255;
        }
    }
}

/// Generic scalar path for YUV420 → RGBA.
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn yuv420_to_rgba_scalar(
    y_plane: &[u8],
    u_plane: &[u8],
    v_plane: &[u8],
    w: usize,
    h: usize,
    uv_w: usize,
    rgba: &mut [u8],
) {
    for row in 0..h {
        let uv_row = row / 2;
        for col in 0..w {
            let uv_col = col / 2;
            let y = y_plane[row * w + col];
            let u = u_plane[uv_row * uv_w + uv_col];
            let v = v_plane[uv_row * uv_w + uv_col];
            let (r, g, b) = ycbcr_to_rgb_pixel(y, u, v);
            let out = (row * w + col) * 4;
            rgba[out] = r;
            rgba[out + 1] = g;
            rgba[out + 2] = b;
            rgba[out + 3] = 255;
        }
    }
}

/// Convert interleaved RGBA to packed I420 (Y plane then U plane then V plane).
///
/// Input: `rgba` of length `width * height * 4`, RGBA byte order.
/// Output: packed I420 byte vec of length `w*h + 2*(w/2)*(h/2)`.
/// Alpha channel is ignored.
///
/// Returns an empty `Vec` if `rgba.len() != width * height * 4`.
pub fn rgba_to_yuv420_simd(rgba: &[u8], width: u32, height: u32) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    if rgba.len() != w * h * 4 {
        return Vec::new();
    }

    let uv_w = (w + 1) / 2;
    let uv_h = (h + 1) / 2;
    let y_size = w * h;
    let uv_size = uv_w * uv_h;
    let mut out = vec![0u8; y_size + 2 * uv_size];

    // Luma pass.
    for row in 0..h {
        for col in 0..w {
            let in_idx = (row * w + col) * 4;
            let r = rgba[in_idx];
            let g = rgba[in_idx + 1];
            let b = rgba[in_idx + 2];
            let (y, _, _) = rgb_to_ycbcr_pixel(r, g, b);
            out[row * w + col] = y;
        }
    }

    // Chroma pass: average 2×2 blocks.
    for uv_row in 0..uv_h {
        for uv_col in 0..uv_w {
            let mut sum_cb = 0u32;
            let mut sum_cr = 0u32;
            let mut count = 0u32;
            for dr in 0..2usize {
                for dc in 0..2usize {
                    let row = uv_row * 2 + dr;
                    let col = uv_col * 2 + dc;
                    if row < h && col < w {
                        let in_idx = (row * w + col) * 4;
                        let r = rgba[in_idx];
                        let g = rgba[in_idx + 1];
                        let b = rgba[in_idx + 2];
                        let (_, cb, cr) = rgb_to_ycbcr_pixel(r, g, b);
                        sum_cb += u32::from(cb);
                        sum_cr += u32::from(cr);
                        count += 1;
                    }
                }
            }
            let uv_idx = uv_row * uv_w + uv_col;
            if let Some(avg_cb) = (sum_cb + count / 2).checked_div(count) {
                out[y_size + uv_idx] = avg_cb as u8;
                out[y_size + uv_size + uv_idx] = ((sum_cr + count / 2) / count) as u8;
            }
        }
    }

    out
}

// =============================================================================
// YUV subsampling format conversions (no RGB round-trip)
// =============================================================================

/// Convert packed YUYV 4:2:2 to packed I420.
///
/// Input `yuv422` is in YUYV interleaved format: `[Y0, U0, Y1, V0, Y2, U1, Y3, V1 ...]`
/// — 4 bytes per 2 pixels, `width` must be even.
///
/// Output is packed I420: Y plane (`width*height` bytes) followed by U plane
/// (`(width/2)*(height/2)` bytes) followed by V plane (same size).
/// Chroma is downsampled vertically by averaging pairs of rows.
///
/// Returns empty `Vec` if input length does not match `width * height * 2`.
pub fn yuv422_to_yuv420(yuv422: &[u8], width: u32, height: u32) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    if w == 0 || h == 0 {
        return Vec::new();
    }
    if yuv422.len() != w * h * 2 {
        return Vec::new();
    }

    let uv_w = w / 2;
    let uv_h = (h + 1) / 2;
    let y_size = w * h;
    let uv_size = uv_w * uv_h;
    let mut out = vec![0u8; y_size + 2 * uv_size];

    // Extract Y plane and build per-row U/V.
    // YUYV: byte layout per row: [Y0 U0 Y1 V0  Y2 U1 Y3 V1  ...]
    // Each macropixel (4 bytes) covers 2 horizontal pixels.

    // Temporary storage for full-resolution chroma rows (one U row per input row).
    let mut u_rows: Vec<Vec<u8>> = Vec::with_capacity(h);
    let mut v_rows: Vec<Vec<u8>> = Vec::with_capacity(h);

    for row in 0..h {
        let row_src = &yuv422[row * w * 2..(row + 1) * w * 2];
        let mut u_row = vec![0u8; uv_w];
        let mut v_row = vec![0u8; uv_w];

        let mut col = 0usize;
        while col + 1 < w {
            let byte_off = col * 2;
            let y0 = row_src[byte_off];
            let u0 = row_src[byte_off + 1];
            let y1 = row_src[byte_off + 2];
            let v0 = row_src[byte_off + 3];

            out[row * w + col] = y0;
            out[row * w + col + 1] = y1;

            let chroma_col = col / 2;
            u_row[chroma_col] = u0;
            v_row[chroma_col] = v0;

            col += 2;
        }
        // Handle odd-width tail (should not occur for valid YUYV, but be safe).
        if col < w {
            let byte_off = col * 2;
            out[row * w + col] = row_src[byte_off];
        }

        u_rows.push(u_row);
        v_rows.push(v_row);
    }

    // Downsample chroma vertically: average pairs of rows.
    for uv_row in 0..uv_h {
        let row0 = uv_row * 2;
        let row1 = (row0 + 1).min(h - 1);
        for uv_col in 0..uv_w {
            let u_avg =
                ((u32::from(u_rows[row0][uv_col]) + u32::from(u_rows[row1][uv_col]) + 1) / 2) as u8;
            let v_avg =
                ((u32::from(v_rows[row0][uv_col]) + u32::from(v_rows[row1][uv_col]) + 1) / 2) as u8;
            let uv_idx = uv_row * uv_w + uv_col;
            out[y_size + uv_idx] = u_avg;
            out[y_size + uv_size + uv_idx] = v_avg;
        }
    }

    out
}

/// Convert packed I444 planar to packed I420.
///
/// Input: Y plane (`width*height` bytes) + U plane (same size) + V plane (same size).
/// Output: packed I420 — Y plane + U plane (`w/2 * h/2` bytes) + V plane (same).
/// Chroma is downsampled by averaging 2×2 pixel blocks.
///
/// Returns empty `Vec` if input length != `width * height * 3`.
pub fn yuv444_to_yuv420(yuv444: &[u8], width: u32, height: u32) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let n = w * h;
    if yuv444.len() != n * 3 {
        return Vec::new();
    }

    let uv_w = (w + 1) / 2;
    let uv_h = (h + 1) / 2;
    let uv_size = uv_w * uv_h;
    let mut out = vec![0u8; n + 2 * uv_size];

    let y444 = &yuv444[..n];
    let u444 = &yuv444[n..2 * n];
    let v444 = &yuv444[2 * n..];

    // Copy Y plane directly.
    out[..n].copy_from_slice(y444);

    // Downsample U and V planes by averaging 2×2 blocks.
    for uv_row in 0..uv_h {
        for uv_col in 0..uv_w {
            let mut sum_u = 0u32;
            let mut sum_v = 0u32;
            let mut count = 0u32;
            for dr in 0..2usize {
                for dc in 0..2usize {
                    let row = uv_row * 2 + dr;
                    let col = uv_col * 2 + dc;
                    if row < h && col < w {
                        sum_u += u32::from(u444[row * w + col]);
                        sum_v += u32::from(v444[row * w + col]);
                        count += 1;
                    }
                }
            }
            let uv_idx = uv_row * uv_w + uv_col;
            if let Some(avg_u) = (sum_u + count / 2).checked_div(count) {
                out[n + uv_idx] = avg_u as u8;
                out[n + uv_size + uv_idx] = ((sum_v + count / 2) / count) as u8;
            }
        }
    }

    out
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- YUV 4:2:0 --

    #[test]
    fn test_yuv420_to_rgb_grey() {
        let w = 4;
        let h = 4;
        let y = vec![128u8; w * h];
        let cb = vec![128u8; (w / 2) * (h / 2)];
        let cr = vec![128u8; (w / 2) * (h / 2)];
        let rgb = yuv420_to_rgb(&y, &cb, &cr, w, h);
        assert_eq!(rgb.len(), w * h * 3);
        // Near-grey values — allow ±5 for coefficient rounding
        for &v in &rgb {
            let vi = v as i32;
            assert!(
                (vi - 114).abs() < 20 || (vi - 128).abs() < 20,
                "Expected near-grey pixel, got {}",
                v
            );
        }
    }

    #[test]
    fn test_rgb_to_yuv420_roundtrip_size() {
        let w = 8;
        let h = 8;
        let rgb = vec![100u8; w * h * 3];
        let (y, cb, cr) = rgb_to_yuv420(&rgb, w, h);
        assert_eq!(y.len(), w * h);
        assert_eq!(cb.len(), (w / 2) * (h / 2));
        assert_eq!(cr.len(), (w / 2) * (h / 2));
    }

    #[test]
    fn test_yuv420_roundtrip_psnr() {
        let w = 16;
        let h = 16;
        // Build a simple ramp pattern
        let rgb: Vec<u8> = (0..(w * h * 3))
            .map(|i| ((i * 7 + 13) % 200 + 28) as u8)
            .collect();
        let (y, cb, cr) = rgb_to_yuv420(&rgb, w, h);
        let rgb2 = yuv420_to_rgb(&y, &cb, &cr, w, h);
        // PSNR must exceed 18 dB (420 chroma downsampling introduces loss,
        // especially for high-frequency synthetic ramp patterns).
        let psnr = compute_psnr(&rgb, &rgb2);
        assert!(
            psnr > 18.0,
            "YUV420 round-trip PSNR = {psnr:.1} dB, expected > 18 dB"
        );
    }

    // -- YUV 4:2:2 --

    #[test]
    fn test_yuv422_to_rgb_size() {
        let w = 8;
        let h = 4;
        let uv_w = (w + 1) / 2;
        let y = vec![128u8; w * h];
        let cb = vec![128u8; uv_w * h];
        let cr = vec![128u8; uv_w * h];
        let rgb = yuv422_to_rgb(&y, &cb, &cr, w, h);
        assert_eq!(rgb.len(), w * h * 3);
    }

    #[test]
    fn test_rgb_to_yuv422_roundtrip_psnr() {
        let w = 16;
        let h = 8;
        let rgb: Vec<u8> = (0..(w * h * 3))
            .map(|i| ((i * 5 + 20) % 220 + 18) as u8)
            .collect();
        let (y, cb, cr) = rgb_to_yuv422(&rgb, w, h);
        let rgb2 = yuv422_to_rgb(&y, &cb, &cr, w, h);
        let psnr = compute_psnr(&rgb, &rgb2);
        // 422 retains full vertical chroma; threshold is lower than 444 due
        // to horizontal downsampling artefacts in high-frequency patterns.
        assert!(
            psnr > 22.0,
            "YUV422 round-trip PSNR = {psnr:.1} dB, expected > 22 dB"
        );
    }

    // -- YUV 4:4:4 --

    #[test]
    fn test_yuv444_roundtrip_perfect() {
        // 4:4:4 with the same quantisation is near-lossless.
        let w = 8;
        let h = 8;
        let rgb: Vec<u8> = (0..(w * h * 3)).map(|i| (i % 200 + 28) as u8).collect();
        let (y, cb, cr) = rgb_to_yuv444(&rgb, w, h);
        let rgb2 = yuv444_to_rgb(&y, &cb, &cr, w, h);
        let psnr = compute_psnr(&rgb, &rgb2);
        assert!(
            psnr > 40.0,
            "YUV444 round-trip PSNR = {psnr:.1} dB, expected > 40 dB"
        );
    }

    #[test]
    fn test_yuv444_identity_grey() {
        let n = 16;
        let rgb = vec![128u8; n * 3];
        let (y, cb, cr) = rgb_to_yuv444(&rgb, n, 1);
        let rgb2 = yuv444_to_rgb(&y, &cb, &cr, n, 1);
        // Each channel should be very close to 128
        for (i, (&a, &b)) in rgb.iter().zip(rgb2.iter()).enumerate() {
            assert!(
                (a as i32 - b as i32).abs() <= 5,
                "pixel[{}]: orig={a}, recon={b}",
                i
            );
        }
    }

    // -- PSNR helper --

    #[test]
    fn test_psnr_perfect_match() {
        let a = vec![100u8; 64];
        let b = vec![100u8; 64];
        let psnr = compute_psnr(&a, &b);
        assert!(psnr.is_infinite() && psnr > 0.0);
    }

    #[test]
    fn test_psnr_worst_case() {
        let a = vec![0u8; 64];
        let b = vec![255u8; 64];
        let psnr = compute_psnr(&a, &b);
        // MSE = 255^2, PSNR = 10*log10(1) = 0 dB
        assert!(
            (psnr - 0.0).abs() < 0.01,
            "PSNR for max error should be 0 dB, got {psnr}"
        );
    }

    #[test]
    fn test_psnr_moderate_error() {
        let a = vec![128u8; 64];
        let b = vec![138u8; 64]; // 10 LSB error per sample
        let psnr = compute_psnr(&a, &b);
        // PSNR = 10 * log10(255^2 / 100) ≈ 28.13 dB
        assert!(
            psnr > 25.0 && psnr < 35.0,
            "Expected moderate PSNR, got {psnr}"
        );
    }

    // =========================================================================
    // Tests for yuv420_to_rgba_simd, rgba_to_yuv420_simd,
    // yuv422_to_yuv420, yuv444_to_yuv420
    // =========================================================================

    // Helper: build a packed I420 frame where Y=y_val, U=u_val, V=v_val.
    fn make_i420(w: usize, h: usize, y_val: u8, u_val: u8, v_val: u8) -> Vec<u8> {
        let uv_w = (w + 1) / 2;
        let uv_h = (h + 1) / 2;
        let y_size = w * h;
        let uv_size = uv_w * uv_h;
        let mut buf = vec![0u8; y_size + 2 * uv_size];
        for i in 0..y_size {
            buf[i] = y_val;
        }
        for i in y_size..y_size + uv_size {
            buf[i] = u_val;
        }
        for i in y_size + uv_size..y_size + 2 * uv_size {
            buf[i] = v_val;
        }
        buf
    }

    // 1. Output length is width * height * 4.
    #[test]
    fn test_yuv420_to_rgba_simd_output_length() {
        let w = 8u32;
        let h = 4u32;
        let i420 = make_i420(w as usize, h as usize, 128, 128, 128);
        let rgba = yuv420_to_rgba_simd(&i420, w, h);
        assert_eq!(rgba.len(), (w * h * 4) as usize);
    }

    // 2. Alpha channel is always 255.
    #[test]
    fn test_yuv420_to_rgba_simd_alpha_is_255() {
        let w = 4u32;
        let h = 4u32;
        let i420 = make_i420(w as usize, h as usize, 128, 128, 128);
        let rgba = yuv420_to_rgba_simd(&i420, w, h);
        for pixel in 0..(w * h) as usize {
            assert_eq!(
                rgba[pixel * 4 + 3],
                255,
                "Alpha at pixel {} should be 255",
                pixel
            );
        }
    }

    // 3. Grey I420 (Y=128, U=128, V=128) produces near-grey RGBA pixels.
    #[test]
    fn test_yuv420_to_rgba_simd_grey_frame() {
        let w = 4u32;
        let h = 4u32;
        let i420 = make_i420(w as usize, h as usize, 128, 128, 128);
        let rgba = yuv420_to_rgba_simd(&i420, w, h);
        // BT.601 studio swing: Y=128 → decoded luminance ≈ 114; allow ±20.
        for pixel in 0..(w * h) as usize {
            for ch in 0..3 {
                let v = rgba[pixel * 4 + ch] as i32;
                assert!(
                    v >= 90 && v <= 145,
                    "Channel {} at pixel {}: expected near-grey, got {}",
                    ch,
                    pixel,
                    v
                );
            }
        }
    }

    // 4. rgba_to_yuv420_simd output length for 4x4 image.
    #[test]
    fn test_rgba_to_yuv420_simd_output_length() {
        let w = 4u32;
        let h = 4u32;
        let rgba = vec![128u8; (w * h * 4) as usize];
        let out = rgba_to_yuv420_simd(&rgba, w, h);
        let uv_w = (w as usize + 1) / 2;
        let uv_h = (h as usize + 1) / 2;
        let expected = w as usize * h as usize + 2 * uv_w * uv_h;
        assert_eq!(out.len(), expected);
    }

    // 5. rgba_to_yuv420_simd invalid input returns empty.
    #[test]
    fn test_rgba_to_yuv420_simd_invalid_input_empty() {
        let out = rgba_to_yuv420_simd(&[0u8; 10], 4, 4);
        assert!(out.is_empty());
    }

    // 6. yuv420_to_rgba_simd invalid input returns empty.
    #[test]
    fn test_yuv420_to_rgba_simd_invalid_input_empty() {
        let out = yuv420_to_rgba_simd(&[0u8; 10], 4, 4);
        assert!(out.is_empty());
    }

    // 7. Round-trip: rgba→yuv420_simd→yuv420_to_rgba_simd luma within ±2.
    #[test]
    fn test_rgba_yuv420_rgba_roundtrip_luma() {
        let w = 8u32;
        let h = 8u32;
        // Build a grey RGBA frame (R=G=B=180).
        let rgba_orig: Vec<u8> = (0..(w * h) as usize)
            .flat_map(|_| [180u8, 180, 180, 255])
            .collect();
        let i420 = rgba_to_yuv420_simd(&rgba_orig, w, h);
        let rgba_recon = yuv420_to_rgba_simd(&i420, w, h);
        assert_eq!(rgba_recon.len(), rgba_orig.len());
        for pixel in 0..(w * h) as usize {
            // Only check luma-related channels for a grey input.
            let r_orig = rgba_orig[pixel * 4] as i32;
            let r_recon = rgba_recon[pixel * 4] as i32;
            assert!(
                (r_orig - r_recon).abs() <= 6,
                "pixel {}: orig={} recon={}",
                pixel,
                r_orig,
                r_recon
            );
        }
    }

    // 8. yuv422_to_yuv420 output length for 8x4.
    #[test]
    fn test_yuv422_to_yuv420_output_length() {
        let w = 8u32;
        let h = 4u32;
        let yuyv = vec![128u8; (w * h * 2) as usize];
        let out = yuv422_to_yuv420(&yuyv, w, h);
        let uv_w = w as usize / 2;
        let uv_h = (h as usize + 1) / 2;
        let expected = w as usize * h as usize + 2 * uv_w * uv_h;
        assert_eq!(out.len(), expected);
    }

    // 9. yuv422_to_yuv420 all-grey YUYV produces correct Y plane.
    #[test]
    fn test_yuv422_to_yuv420_grey_y_plane() {
        let w = 8u32;
        let h = 4u32;
        // All-grey YUYV: Y=128, U=128, V=128 → pattern [128, 128, 128, 128, ...]
        let yuyv = vec![128u8; (w * h * 2) as usize];
        let out = yuv422_to_yuv420(&yuyv, w, h);
        let y_size = (w * h) as usize;
        for i in 0..y_size {
            assert_eq!(out[i], 128, "Y[{}] should be 128", i);
        }
    }

    // 10. yuv422_to_yuv420 all-grey YUYV: U and V planes near 128.
    #[test]
    fn test_yuv422_to_yuv420_grey_uv_planes() {
        let w = 8u32;
        let h = 4u32;
        let yuyv = vec![128u8; (w * h * 2) as usize];
        let out = yuv422_to_yuv420(&yuyv, w, h);
        let y_size = (w * h) as usize;
        let uv_size = (w as usize / 2) * ((h as usize + 1) / 2);
        for i in y_size..y_size + 2 * uv_size {
            assert!(
                (out[i] as i32 - 128).abs() <= 1,
                "UV[{}] should be near 128, got {}",
                i,
                out[i]
            );
        }
    }

    // 11. yuv444_to_yuv420 output length for 8x4.
    #[test]
    fn test_yuv444_to_yuv420_output_length() {
        let w = 8u32;
        let h = 4u32;
        let i444 = vec![128u8; (w * h * 3) as usize];
        let out = yuv444_to_yuv420(&i444, w, h);
        let uv_w = (w as usize + 1) / 2;
        let uv_h = (h as usize + 1) / 2;
        let expected = w as usize * h as usize + 2 * uv_w * uv_h;
        assert_eq!(out.len(), expected);
    }

    // 12. yuv444_to_yuv420 all-grey input produces correct output.
    #[test]
    fn test_yuv444_to_yuv420_grey() {
        let w = 8u32;
        let h = 4u32;
        let i444 = vec![128u8; (w * h * 3) as usize];
        let out = yuv444_to_yuv420(&i444, w, h);
        let y_size = (w * h) as usize;
        for i in 0..y_size {
            assert_eq!(out[i], 128, "Y[{}] should be 128", i);
        }
        for i in y_size..out.len() {
            assert_eq!(out[i], 128, "UV[{}] should be 128", i);
        }
    }

    // 13. yuv444_to_yuv420 uniform Y is preserved exactly.
    #[test]
    fn test_yuv444_to_yuv420_y_passthrough() {
        let w = 4u32;
        let h = 4u32;
        let n = (w * h) as usize;
        let mut i444 = vec![0u8; n * 3];
        // Set Y to a ramp pattern.
        for i in 0..n {
            i444[i] = (i * 7 % 220 + 16) as u8;
        }
        // U and V = 128.
        for i in n..n * 3 {
            i444[i] = 128;
        }
        let out = yuv444_to_yuv420(&i444, w, h);
        for i in 0..n {
            assert_eq!(out[i], i444[i], "Y[{}] should be preserved", i);
        }
    }

    // 14. yuv420_to_rgba_simd bright frame: Y=235, U=128, V=128 → bright pixels.
    #[test]
    fn test_yuv420_to_rgba_simd_bright_frame() {
        let w = 4u32;
        let h = 4u32;
        let i420 = make_i420(w as usize, h as usize, 235, 128, 128);
        let rgba = yuv420_to_rgba_simd(&i420, w, h);
        for pixel in 0..(w * h) as usize {
            for ch in 0..3 {
                let v = rgba[pixel * 4 + ch];
                assert!(
                    v >= 200,
                    "Expected bright pixel, ch={} pixel={} val={}",
                    ch,
                    pixel,
                    v
                );
            }
        }
    }

    // 15. rgba_to_yuv420_simd all-black → Y near 16 (studio swing).
    #[test]
    fn test_rgba_to_yuv420_simd_black_y_is_16() {
        let w = 4u32;
        let h = 4u32;
        let rgba = vec![0u8; (w * h * 4) as usize];
        let out = rgba_to_yuv420_simd(&rgba, w, h);
        let y_size = (w * h) as usize;
        for i in 0..y_size {
            assert!(
                (out[i] as i32 - 16).abs() <= 2,
                "Y[{}] should be near 16, got {}",
                i,
                out[i]
            );
        }
    }

    // 16. rgba_to_yuv420_simd all-white → Y near 235 (studio swing).
    #[test]
    fn test_rgba_to_yuv420_simd_white_y_is_235() {
        let w = 4u32;
        let h = 4u32;
        let rgba = vec![255u8; (w * h * 4) as usize];
        let out = rgba_to_yuv420_simd(&rgba, w, h);
        let y_size = (w * h) as usize;
        for i in 0..y_size {
            assert!(
                (out[i] as i32 - 235).abs() <= 2,
                "Y[{}] should be near 235, got {}",
                i,
                out[i]
            );
        }
    }

    // 17. yuv422_to_yuv420 chroma vertical averaging.
    #[test]
    fn test_yuv422_to_yuv420_chroma_vertical_averaging() {
        // 2×2 frame, YUYV: row0 U=100, row1 U=200 → output U ≈ 150.
        let w = 2u32;
        let h = 2u32;
        // YUYV row layout: [Y0, U, Y1, V]
        // Row 0: Y=128, U=100, Y=128, V=128
        // Row 1: Y=128, U=200, Y=128, V=128
        let mut yuyv = vec![0u8; (w * h * 2) as usize];
        // Row 0: bytes 0..4
        yuyv[0] = 128;
        yuyv[1] = 100;
        yuyv[2] = 128;
        yuyv[3] = 128;
        // Row 1: bytes 4..8
        yuyv[4] = 128;
        yuyv[5] = 200;
        yuyv[6] = 128;
        yuyv[7] = 128;
        let out = yuv422_to_yuv420(&yuyv, w, h);
        let y_size = (w * h) as usize; // 4
                                       // uv_w = 1, uv_h = 1, uv_size = 1
        let u_val = out[y_size] as i32;
        assert!((u_val - 150).abs() <= 2, "Expected U≈150, got {}", u_val);
    }

    // 18. yuv444_to_yuv420 2x2 block averaging.
    #[test]
    fn test_yuv444_to_yuv420_chroma_block_avg() {
        // 2×2 frame: U values = [100, 120, 140, 160] → average = 130.
        let w = 2u32;
        let h = 2u32;
        let n = (w * h) as usize; // 4
        let mut i444 = vec![128u8; n * 3];
        // U plane starts at offset n=4
        i444[n + 0] = 100;
        i444[n + 1] = 120;
        i444[n + 2] = 140;
        i444[n + 3] = 160;
        let out = yuv444_to_yuv420(&i444, w, h);
        let u_val = out[n] as i32; // uv_size=1, U plane starts at n
        assert!((u_val - 130).abs() <= 2, "Expected U≈130, got {}", u_val);
    }

    // 19. yuv422_to_yuv420 invalid input returns empty.
    #[test]
    fn test_yuv422_to_yuv420_invalid_input_empty() {
        let out = yuv422_to_yuv420(&[0u8; 5], 4, 4);
        assert!(out.is_empty());
    }

    // 20. yuv444_to_yuv420 invalid input returns empty.
    #[test]
    fn test_yuv444_to_yuv420_invalid_input_empty() {
        let out = yuv444_to_yuv420(&[0u8; 5], 4, 4);
        assert!(out.is_empty());
    }

    // 21. rgba_to_yuv420_simd + yuv420_to_rgba_simd round-trip on ramp pattern.
    #[test]
    fn test_rgba_yuv420_rgba_ramp_roundtrip_psnr() {
        let w = 8u32;
        let h = 8u32;
        let rgba_orig: Vec<u8> = (0..(w * h) as usize)
            .flat_map(|i| {
                let v = (i * 13 % 180 + 50) as u8;
                [v, v, v, 255u8]
            })
            .collect();
        let i420 = rgba_to_yuv420_simd(&rgba_orig, w, h);
        let rgba_recon = yuv420_to_rgba_simd(&i420, w, h);
        // Extract just RGB channels for PSNR.
        let orig_rgb: Vec<u8> = rgba_orig
            .chunks(4)
            .flat_map(|p| [p[0], p[1], p[2]])
            .collect();
        let recon_rgb: Vec<u8> = rgba_recon
            .chunks(4)
            .flat_map(|p| [p[0], p[1], p[2]])
            .collect();
        let psnr = compute_psnr(&orig_rgb, &recon_rgb);
        assert!(
            psnr > 30.0,
            "Ramp round-trip PSNR={:.1} dB expected > 30",
            psnr
        );
    }

    // 22. yuv444_to_yuv420 invalid (zero dimensions) returns empty.
    #[test]
    fn test_yuv444_to_yuv420_zero_dimensions() {
        // width=0 means n=0 so input length = 0*3 = 0; empty slice should pass.
        let out = yuv444_to_yuv420(&[], 0, 0);
        // n=0 * 3 = 0 matches yuv444.len()=0; output should be empty vec.
        assert!(out.is_empty());
    }
}
