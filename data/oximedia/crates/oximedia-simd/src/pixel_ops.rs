//! SIMD-optimized pixel / image operations.
//!
//! Provides scalar implementations with SIMD-friendly memory layouts.
//! All pixel buffers are assumed to be packed RGBA (4 bytes per pixel) unless
//! the function documentation states otherwise.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(dead_code)]

/// A single RGBA pixel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Pixel {
    /// Red channel.
    pub r: u8,
    /// Green channel.
    pub g: u8,
    /// Blue channel.
    pub b: u8,
    /// Alpha channel (0 = fully transparent, 255 = fully opaque).
    pub a: u8,
}

impl Pixel {
    /// Create a new [`Pixel`].
    #[must_use]
    pub const fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    /// Convert to a packed `[r, g, b, a]` array.
    #[must_use]
    pub const fn to_array(self) -> [u8; 4] {
        [self.r, self.g, self.b, self.a]
    }
}

/// Porter-Duff "over" alpha compositing.
///
/// Both `src` and `dst` must be RGBA-interleaved (4 bytes per pixel).  The
/// slices must have the same length; any trailing bytes that don't form a
/// complete pixel are ignored.
#[must_use]
pub fn blend_over(src: &[u8], dst: &[u8]) -> Vec<u8> {
    let pixels = src.len().min(dst.len()) / 4;
    let mut out = vec![0u8; pixels * 4];
    for i in 0..pixels {
        let base = i * 4;
        let sa = u32::from(src[base + 3]);
        let inv_sa = 255 - sa;

        out[base] = ((u32::from(src[base]) * sa + u32::from(dst[base]) * inv_sa) / 255) as u8;
        out[base + 1] =
            ((u32::from(src[base + 1]) * sa + u32::from(dst[base + 1]) * inv_sa) / 255) as u8;
        out[base + 2] =
            ((u32::from(src[base + 2]) * sa + u32::from(dst[base + 2]) * inv_sa) / 255) as u8;
        out[base + 3] = (sa + u32::from(dst[base + 3]) * inv_sa / 255) as u8;
    }
    out
}

/// Convert packed RGB (3 bytes/pixel) to YUV (3 bytes/pixel) using BT.709.
///
/// Output layout: [Y, U, V, Y, U, V, …] with U/V biased to 128.
#[must_use]
pub fn convert_rgb_to_yuv(rgb: &[u8]) -> Vec<u8> {
    let pixels = rgb.len() / 3;
    let mut out = vec![0u8; pixels * 3];
    for i in 0..pixels {
        let r = f32::from(rgb[i * 3]);
        let g = f32::from(rgb[i * 3 + 1]);
        let b = f32::from(rgb[i * 3 + 2]);

        let y = (0.2126 * r + 0.7152 * g + 0.0722 * b).round();
        let u = (-0.1146 * r - 0.3854 * g + 0.5 * b + 128.0).round();
        let v = (0.5 * r - 0.4542 * g - 0.0458 * b + 128.0).round();

        out[i * 3] = y.clamp(0.0, 255.0) as u8;
        out[i * 3 + 1] = u.clamp(0.0, 255.0) as u8;
        out[i * 3 + 2] = v.clamp(0.0, 255.0) as u8;
    }
    out
}

/// Convert packed YUV (3 bytes/pixel, U/V biased to 128) to RGB using BT.709.
#[must_use]
pub fn convert_yuv_to_rgb(yuv: &[u8]) -> Vec<u8> {
    let pixels = yuv.len() / 3;
    let mut out = vec![0u8; pixels * 3];
    for i in 0..pixels {
        let y = f32::from(yuv[i * 3]);
        let u = f32::from(yuv[i * 3 + 1]) - 128.0;
        let v = f32::from(yuv[i * 3 + 2]) - 128.0;

        let r = (y + 1.5748 * v).round();
        let g = (y - 0.1873 * u - 0.4681 * v).round();
        let b = (y + 1.8556 * u).round();

        out[i * 3] = r.clamp(0.0, 255.0) as u8;
        out[i * 3 + 1] = g.clamp(0.0, 255.0) as u8;
        out[i * 3 + 2] = b.clamp(0.0, 255.0) as u8;
    }
    out
}

/// Apply a 1-D LUT (look-up table) to every byte in `pixels`.
#[must_use]
pub fn apply_lut(pixels: &[u8], lut: &[u8; 256]) -> Vec<u8> {
    pixels.iter().map(|&p| lut[p as usize]).collect()
}

/// Apply a binary threshold to every byte in `pixels`.
///
/// Values ≥ `thresh` become 255; values below become 0.
#[must_use]
pub fn threshold(pixels: &[u8], thresh: u8) -> Vec<u8> {
    pixels
        .iter()
        .map(|&p| if p >= thresh { 255 } else { 0 })
        .collect()
}

/// Flip an image horizontally (mirror left↔right).
///
/// `pixels` must be a packed row-major buffer of `width * height` bytes (e.g.
/// grayscale).  Use a stride of `width * bytes_per_pixel` and pass a
/// byte-expanded buffer for multi-channel images.
#[must_use]
pub fn horizontal_flip(pixels: &[u8], width: usize, height: usize) -> Vec<u8> {
    let mut out = vec![0u8; pixels.len()];
    for row in 0..height {
        for col in 0..width {
            let src_idx = row * width + col;
            let dst_idx = row * width + (width - 1 - col);
            if src_idx < pixels.len() && dst_idx < out.len() {
                out[dst_idx] = pixels[src_idx];
            }
        }
    }
    out
}

/// Flip an image vertically (mirror top↔bottom).
#[must_use]
pub fn vertical_flip(pixels: &[u8], width: usize, height: usize) -> Vec<u8> {
    let mut out = vec![0u8; pixels.len()];
    for row in 0..height {
        let src_start = row * width;
        let dst_start = (height - 1 - row) * width;
        let len = width.min(pixels.len().saturating_sub(src_start));
        if dst_start + len <= out.len() {
            out[dst_start..dst_start + len].copy_from_slice(&pixels[src_start..src_start + len]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pixel_new_and_array() {
        let p = Pixel::new(10, 20, 30, 255);
        assert_eq!(p.to_array(), [10, 20, 30, 255]);
    }

    #[test]
    fn test_blend_over_fully_opaque_src() {
        // Opaque red over blue -> red
        let src = vec![255u8, 0, 0, 255];
        let dst = vec![0u8, 0, 255, 255];
        let out = blend_over(&src, &dst);
        assert_eq!(out[0], 255); // R
        assert_eq!(out[2], 0); // B
    }

    #[test]
    fn test_blend_over_fully_transparent_src() {
        // Transparent src leaves dst unchanged
        let src = vec![255u8, 0, 0, 0];
        let dst = vec![0u8, 0, 255, 255];
        let out = blend_over(&src, &dst);
        assert_eq!(out[2], 255); // B preserved
    }

    #[test]
    fn test_blend_over_empty() {
        let out = blend_over(&[], &[]);
        assert!(out.is_empty());
    }

    #[test]
    fn test_apply_lut_identity() {
        let mut lut = [0u8; 256];
        for (i, v) in lut.iter_mut().enumerate() {
            *v = i as u8;
        }
        let pixels = vec![0u8, 127, 255];
        assert_eq!(apply_lut(&pixels, &lut), pixels);
    }

    #[test]
    fn test_apply_lut_invert() {
        let mut lut = [0u8; 256];
        for (i, v) in lut.iter_mut().enumerate() {
            *v = 255 - i as u8;
        }
        let pixels = vec![0u8, 255];
        let out = apply_lut(&pixels, &lut);
        assert_eq!(out, vec![255, 0]);
    }

    #[test]
    fn test_threshold_above() {
        let pixels = vec![100u8, 200, 50, 128];
        let out = threshold(&pixels, 128);
        assert_eq!(out, vec![0, 255, 0, 255]);
    }

    #[test]
    fn test_horizontal_flip_2x2() {
        // 2-wide, 2-tall: [1,2,3,4] -> [2,1,4,3]
        let pixels = vec![1u8, 2, 3, 4];
        let out = horizontal_flip(&pixels, 2, 2);
        assert_eq!(out, vec![2, 1, 4, 3]);
    }

    #[test]
    fn test_vertical_flip_2x2() {
        // [1,2,3,4] -> [3,4,1,2]
        let pixels = vec![1u8, 2, 3, 4];
        let out = vertical_flip(&pixels, 2, 2);
        assert_eq!(out, vec![3, 4, 1, 2]);
    }

    #[test]
    fn test_horizontal_flip_single_row() {
        let pixels = vec![1u8, 2, 3];
        let out = horizontal_flip(&pixels, 3, 1);
        assert_eq!(out, vec![3, 2, 1]);
    }

    #[test]
    fn test_vertical_flip_single_col() {
        let pixels = vec![10u8, 20, 30];
        let out = vertical_flip(&pixels, 1, 3);
        assert_eq!(out, vec![30, 20, 10]);
    }

    #[test]
    fn test_rgb_yuv_roundtrip_white() {
        // White (255,255,255) should roundtrip close to white
        let rgb = vec![255u8, 255, 255];
        let yuv = convert_rgb_to_yuv(&rgb);
        let rgb2 = convert_yuv_to_rgb(&yuv);
        for (a, b) in rgb.iter().zip(rgb2.iter()) {
            let diff = (i32::from(*a) - i32::from(*b)).abs();
            assert!(diff <= 3, "diff too large: {diff}");
        }
    }

    #[test]
    fn test_rgb_yuv_roundtrip_black() {
        let rgb = vec![0u8, 0, 0];
        let yuv = convert_rgb_to_yuv(&rgb);
        let rgb2 = convert_yuv_to_rgb(&yuv);
        for (a, b) in rgb.iter().zip(rgb2.iter()) {
            let diff = (i32::from(*a) - i32::from(*b)).abs();
            assert!(diff <= 3, "diff too large: {diff}");
        }
    }
}
