//! SIMD-optimized blending operations
//!
//! Provides alpha compositing, multiply/screen blends, and vectorized
//! pixel blend loops using portable SIMD-friendly scalar code.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

/// Blend mode selector
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendMode {
    /// Normal alpha-over compositing
    AlphaOver,
    /// Multiply blend: dst * src
    Multiply,
    /// Screen blend: 1 - (1-dst)*(1-src)
    Screen,
    /// Overlay blend
    Overlay,
    /// Additive blend (clamped)
    Add,
    /// Difference blend
    Difference,
    /// Darken: min(src, dst)
    Darken,
    /// Lighten: max(src, dst)
    Lighten,
}

/// Pixel with premultiplied alpha (RGBA, u8 each)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Pixel {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Pixel {
    /// Construct a new pixel
    #[must_use]
    pub const fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    /// Convert to f32 components in [0.0, 1.0]
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn to_f32(self) -> [f32; 4] {
        [
            f32::from(self.r) / 255.0,
            f32::from(self.g) / 255.0,
            f32::from(self.b) / 255.0,
            f32::from(self.a) / 255.0,
        ]
    }

    /// Construct from f32 components (clamped to [0.0, 1.0])
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn from_f32(c: [f32; 4]) -> Self {
        let clamp = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        Self::new(clamp(c[0]), clamp(c[1]), clamp(c[2]), clamp(c[3]))
    }
}

/// Alpha-over compositing: src over dst
///
/// Formula: `out = src + dst * (1 - src.a)`
#[must_use]
pub fn alpha_over(src: Pixel, dst: Pixel) -> Pixel {
    let s = src.to_f32();
    let d = dst.to_f32();
    let inv_a = 1.0 - s[3];
    Pixel::from_f32([
        s[0] * s[3] + d[0] * d[3] * inv_a,
        s[1] * s[3] + d[1] * d[3] * inv_a,
        s[2] * s[3] + d[2] * d[3] * inv_a,
        s[3] + d[3] * inv_a,
    ])
}

/// Multiply blend (RGB channels only, alpha passed through from src)
#[must_use]
pub fn multiply_blend(src: Pixel, dst: Pixel) -> Pixel {
    Pixel::new(
        ((u16::from(src.r) * u16::from(dst.r)) / 255) as u8,
        ((u16::from(src.g) * u16::from(dst.g)) / 255) as u8,
        ((u16::from(src.b) * u16::from(dst.b)) / 255) as u8,
        src.a,
    )
}

/// Screen blend
///
/// Formula: `1 - (1-src) * (1-dst)` per channel
#[must_use]
pub fn screen_blend(src: Pixel, dst: Pixel) -> Pixel {
    let blend_channel = |s: u8, d: u8| -> u8 {
        let s = u16::from(s);
        let d = u16::from(d);
        (255 - ((255 - s) * (255 - d) / 255)) as u8
    };
    Pixel::new(
        blend_channel(src.r, dst.r),
        blend_channel(src.g, dst.g),
        blend_channel(src.b, dst.b),
        src.a,
    )
}

/// Overlay blend
#[must_use]
pub fn overlay_blend(src: Pixel, dst: Pixel) -> Pixel {
    let blend_channel = |s: u8, d: u8| -> u8 {
        if d < 128 {
            ((2 * u16::from(s) * u16::from(d)) / 255) as u8
        } else {
            let inv_s = 255u16 - u16::from(s);
            let inv_d = 255u16 - u16::from(d);
            (255 - (2 * inv_s * inv_d / 255)) as u8
        }
    };
    Pixel::new(
        blend_channel(src.r, dst.r),
        blend_channel(src.g, dst.g),
        blend_channel(src.b, dst.b),
        src.a,
    )
}

/// Additive blend (clamped to 255)
#[must_use]
pub fn add_blend(src: Pixel, dst: Pixel) -> Pixel {
    Pixel::new(
        src.r.saturating_add(dst.r),
        src.g.saturating_add(dst.g),
        src.b.saturating_add(dst.b),
        src.a,
    )
}

/// Difference blend
#[must_use]
pub fn difference_blend(src: Pixel, dst: Pixel) -> Pixel {
    Pixel::new(
        src.r.abs_diff(dst.r),
        src.g.abs_diff(dst.g),
        src.b.abs_diff(dst.b),
        src.a,
    )
}

/// Darken blend (min per channel)
#[must_use]
pub fn darken_blend(src: Pixel, dst: Pixel) -> Pixel {
    Pixel::new(src.r.min(dst.r), src.g.min(dst.g), src.b.min(dst.b), src.a)
}

/// Lighten blend (max per channel)
#[must_use]
pub fn lighten_blend(src: Pixel, dst: Pixel) -> Pixel {
    Pixel::new(src.r.max(dst.r), src.g.max(dst.g), src.b.max(dst.b), src.a)
}

/// Dispatch blend mode
#[must_use]
pub fn blend(src: Pixel, dst: Pixel, mode: BlendMode) -> Pixel {
    match mode {
        BlendMode::AlphaOver => alpha_over(src, dst),
        BlendMode::Multiply => multiply_blend(src, dst),
        BlendMode::Screen => screen_blend(src, dst),
        BlendMode::Overlay => overlay_blend(src, dst),
        BlendMode::Add => add_blend(src, dst),
        BlendMode::Difference => difference_blend(src, dst),
        BlendMode::Darken => darken_blend(src, dst),
        BlendMode::Lighten => lighten_blend(src, dst),
    }
}

/// Blend two RGBA buffers element-wise with a given mode.
///
/// `src`, `dst`, and `out` must all have the same length (multiple of 4).
///
/// # Errors
/// Returns an error string if buffer lengths don't match or are not a multiple of 4.
pub fn blend_buffers(
    src: &[u8],
    dst: &[u8],
    out: &mut [u8],
    mode: BlendMode,
) -> Result<(), String> {
    if src.len() != dst.len() || src.len() != out.len() {
        return Err("Buffer length mismatch".to_string());
    }
    if !src.len().is_multiple_of(4) {
        return Err("Buffer length must be a multiple of 4".to_string());
    }
    for i in (0..src.len()).step_by(4) {
        let s = Pixel::new(src[i], src[i + 1], src[i + 2], src[i + 3]);
        let d = Pixel::new(dst[i], dst[i + 1], dst[i + 2], dst[i + 3]);
        let o = blend(s, d, mode);
        out[i] = o.r;
        out[i + 1] = o.g;
        out[i + 2] = o.b;
        out[i + 3] = o.a;
    }
    Ok(())
}

/// Apply a constant alpha multiplier to an RGBA buffer in-place.
pub fn apply_alpha_gain(buf: &mut [u8], alpha: f32) {
    let gain = alpha.clamp(0.0, 1.0);
    for chunk in buf.chunks_mut(4) {
        if chunk.len() == 4 {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let new_a = (f32::from(chunk[3]) * gain + 0.5) as u8;
            chunk[3] = new_a;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pixel_roundtrip() {
        let p = Pixel::new(128, 64, 32, 255);
        let f = p.to_f32();
        let p2 = Pixel::from_f32(f);
        assert!((i32::from(p2.r) - i32::from(p.r)).abs() <= 1);
        assert!((i32::from(p2.g) - i32::from(p.g)).abs() <= 1);
        assert!((i32::from(p2.b) - i32::from(p.b)).abs() <= 1);
    }

    #[test]
    fn test_alpha_over_full_opacity() {
        let src = Pixel::new(200, 100, 50, 255);
        let dst = Pixel::new(10, 20, 30, 255);
        let out = alpha_over(src, dst);
        // src is fully opaque so output should equal src RGB
        assert!((i32::from(out.r) - 200).abs() <= 1);
        assert!((i32::from(out.g) - 100).abs() <= 1);
        assert!((i32::from(out.b) - 50).abs() <= 1);
    }

    #[test]
    fn test_alpha_over_transparent_src() {
        let src = Pixel::new(200, 100, 50, 0);
        let dst = Pixel::new(10, 20, 30, 255);
        let out = alpha_over(src, dst);
        // fully transparent src -> output equals dst
        assert!((i32::from(out.r) - 10).abs() <= 1);
        assert!((i32::from(out.g) - 20).abs() <= 1);
        assert!((i32::from(out.b) - 30).abs() <= 1);
    }

    #[test]
    fn test_multiply_blend() {
        let src = Pixel::new(255, 128, 0, 255);
        let dst = Pixel::new(255, 255, 255, 255);
        let out = multiply_blend(src, dst);
        assert_eq!(out.r, 255);
        assert!((i32::from(out.g) - 128).abs() <= 1);
        assert_eq!(out.b, 0);
    }

    #[test]
    fn test_multiply_black() {
        let src = Pixel::new(0, 0, 0, 255);
        let dst = Pixel::new(200, 150, 100, 255);
        let out = multiply_blend(src, dst);
        assert_eq!(out.r, 0);
        assert_eq!(out.g, 0);
        assert_eq!(out.b, 0);
    }

    #[test]
    fn test_screen_blend_with_black() {
        // Screening with black should return the other color
        let src = Pixel::new(0, 0, 0, 255);
        let dst = Pixel::new(100, 150, 200, 255);
        let out = screen_blend(src, dst);
        assert_eq!(out.r, 100);
        assert_eq!(out.g, 150);
        assert_eq!(out.b, 200);
    }

    #[test]
    fn test_screen_blend_with_white() {
        // Screening with white gives white
        let src = Pixel::new(255, 255, 255, 255);
        let dst = Pixel::new(100, 150, 200, 255);
        let out = screen_blend(src, dst);
        assert_eq!(out.r, 255);
        assert_eq!(out.g, 255);
        assert_eq!(out.b, 255);
    }

    #[test]
    fn test_add_blend_clamp() {
        let src = Pixel::new(200, 200, 200, 255);
        let dst = Pixel::new(100, 100, 100, 255);
        let out = add_blend(src, dst);
        assert_eq!(out.r, 255);
        assert_eq!(out.g, 255);
        assert_eq!(out.b, 255);
    }

    #[test]
    fn test_difference_blend() {
        let src = Pixel::new(200, 100, 50, 255);
        let dst = Pixel::new(100, 200, 50, 255);
        let out = difference_blend(src, dst);
        assert_eq!(out.r, 100);
        assert_eq!(out.g, 100);
        assert_eq!(out.b, 0);
    }

    #[test]
    fn test_darken_lighten() {
        let src = Pixel::new(100, 200, 150, 255);
        let dst = Pixel::new(150, 100, 200, 255);
        let dark = darken_blend(src, dst);
        let light = lighten_blend(src, dst);
        assert_eq!(dark.r, 100);
        assert_eq!(dark.g, 100);
        assert_eq!(dark.b, 150);
        assert_eq!(light.r, 150);
        assert_eq!(light.g, 200);
        assert_eq!(light.b, 200);
    }

    #[test]
    fn test_blend_buffers_multiply() {
        let src = vec![255u8, 128, 0, 255, 100, 200, 50, 128];
        let dst = vec![255u8, 255, 255, 255, 255, 255, 255, 255];
        let mut out = vec![0u8; 8];
        blend_buffers(&src, &dst, &mut out, BlendMode::Multiply).expect("should succeed in test");
        assert_eq!(out[0], 255);
        assert!((i32::from(out[1]) - 128).abs() <= 1);
        assert_eq!(out[2], 0);
    }

    #[test]
    fn test_blend_buffers_length_mismatch() {
        let src = vec![0u8; 4];
        let dst = vec![0u8; 8];
        let mut out = vec![0u8; 4];
        assert!(blend_buffers(&src, &dst, &mut out, BlendMode::Add).is_err());
    }

    #[test]
    fn test_blend_buffers_not_multiple_of_4() {
        let src = vec![0u8; 5];
        let dst = vec![0u8; 5];
        let mut out = vec![0u8; 5];
        assert!(blend_buffers(&src, &dst, &mut out, BlendMode::Add).is_err());
    }

    #[test]
    fn test_apply_alpha_gain() {
        let mut buf = vec![255u8, 255, 255, 200];
        apply_alpha_gain(&mut buf, 0.5);
        assert!((i32::from(buf[3]) - 100).abs() <= 1);
    }

    #[test]
    fn test_blend_dispatch_all_modes() {
        let src = Pixel::new(128, 128, 128, 200);
        let dst = Pixel::new(64, 64, 64, 255);
        let modes = [
            BlendMode::AlphaOver,
            BlendMode::Multiply,
            BlendMode::Screen,
            BlendMode::Overlay,
            BlendMode::Add,
            BlendMode::Difference,
            BlendMode::Darken,
            BlendMode::Lighten,
        ];
        for mode in modes {
            let _ = blend(src, dst, mode); // Just check no panic
        }
    }
}
