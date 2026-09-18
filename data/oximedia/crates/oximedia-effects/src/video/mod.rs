//! Video effects for `OxiMedia`.
//!
//! This module provides professional-quality video effects for pixel-level processing.
//! All effects operate on raw pixel data (`&[u8]` or `&mut [u8]`) in RGBA or RGB formats.
//!
//! # Effects
//!
//! - [`LensFlare`] - Configurable lens flare with halo and streaks
//! - [`ChromaticAberration`] - RGB channel separation effect
//! - [`MotionBlur`] - Directional convolution-based motion blur
//! - [`Vignette`] - Radial darkening with configurable falloff
//! - [`GrainEffect`] - Film grain simulation
//! - [`ColorGrade`] - 3-way color grading (lift/gamma/gain)

pub mod blend;
pub mod chromakey;
pub mod chromatic_aberration;
pub mod color_grade;
pub mod grain;
pub mod lens_flare;
pub mod motion_blur;
pub mod vignette;

pub use blend::{blend_frames, blend_pixel, BlendMode};
pub use chromakey::{apply_chroma_key, detect_key_color, ChromaKeyParams};
pub use chromatic_aberration::{ChromaticAberration, ChromaticAberrationConfig};
pub use color_grade::{ColorGrade, ColorGradeConfig, LiftGammaGain};
pub use grain::{GrainConfig, GrainEffect};
pub use lens_flare::{LensFlare, LensFlareColor, LensFlareConfig};
pub use motion_blur::{MotionBlur, MotionBlurConfig};
pub use vignette::{Vignette, VignetteConfig};

use crate::EffectError;

/// Pixel format for video effects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    /// 3 bytes per pixel: R, G, B
    Rgb,
    /// 4 bytes per pixel: R, G, B, A
    Rgba,
}

impl PixelFormat {
    /// Returns number of bytes per pixel.
    #[must_use]
    pub const fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Rgb => 3,
            Self::Rgba => 4,
        }
    }
}

/// Shared result type for video effects.
pub type VideoResult<T> = Result<T, EffectError>;

/// Validate that buffer size matches dimensions and format.
pub(crate) fn validate_buffer(
    data: &[u8],
    width: usize,
    height: usize,
    format: PixelFormat,
) -> VideoResult<()> {
    let expected = width * height * format.bytes_per_pixel();
    if data.len() != expected {
        return Err(EffectError::BufferSizeMismatch {
            expected,
            actual: data.len(),
        });
    }
    Ok(())
}

/// Clamp a value to [0, 255] and convert to u8.
#[inline]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub(crate) fn clamp_u8(val: f32) -> u8 {
    val.clamp(0.0, 255.0) as u8
}

/// Bilinear sample from pixel buffer, returning [R, G, B] (or [R, G, B, A]).
/// Out-of-bounds pixels use edge clamping.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss
)]
pub(crate) fn sample_bilinear(
    data: &[u8],
    width: usize,
    height: usize,
    bpp: usize,
    x: f32,
    y: f32,
) -> [f32; 4] {
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let fx = x - x0 as f32;
    let fy = y - y0 as f32;

    let p00 = get_pixel_clamped(data, width, height, bpp, x0, y0);
    let p10 = get_pixel_clamped(data, width, height, bpp, x1, y0);
    let p01 = get_pixel_clamped(data, width, height, bpp, x0, y1);
    let p11 = get_pixel_clamped(data, width, height, bpp, x1, y1);

    let mut result = [0.0f32; 4];
    for c in 0..4 {
        let top = p00[c] * (1.0 - fx) + p10[c] * fx;
        let bot = p01[c] * (1.0 - fx) + p11[c] * fx;
        result[c] = top * (1.0 - fy) + bot * fy;
    }
    result
}

/// Get pixel with edge clamping.
#[allow(
    clippy::many_single_char_names,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]
pub(crate) fn get_pixel_clamped(
    data: &[u8],
    width: usize,
    height: usize,
    bpp: usize,
    x: i32,
    y: i32,
) -> [f32; 4] {
    let cx = x.clamp(0, width as i32 - 1) as usize;
    let cy = y.clamp(0, height as i32 - 1) as usize;
    let idx = (cy * width + cx) * bpp;
    let r = f32::from(data[idx]);
    let g = f32::from(data[idx + 1]);
    let b = f32::from(data[idx + 2]);
    let a = if bpp >= 4 {
        f32::from(data[idx + 3])
    } else {
        255.0
    };
    [r, g, b, a]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pixel_format_bpp() {
        assert_eq!(PixelFormat::Rgb.bytes_per_pixel(), 3);
        assert_eq!(PixelFormat::Rgba.bytes_per_pixel(), 4);
    }

    #[test]
    fn test_validate_buffer_ok() {
        let buf = vec![0u8; 4 * 4 * 3];
        assert!(validate_buffer(&buf, 4, 4, PixelFormat::Rgb).is_ok());
    }

    #[test]
    fn test_validate_buffer_err() {
        let buf = vec![0u8; 10];
        assert!(validate_buffer(&buf, 4, 4, PixelFormat::Rgb).is_err());
    }

    #[test]
    fn test_clamp_u8() {
        assert_eq!(clamp_u8(-5.0), 0);
        assert_eq!(clamp_u8(300.0), 255);
        assert_eq!(clamp_u8(128.5), 128);
    }
}
