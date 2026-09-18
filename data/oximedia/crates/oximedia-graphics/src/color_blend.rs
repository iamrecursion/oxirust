#![allow(dead_code)]
//! Color blending modes for broadcast graphics compositing.
//!
//! Implements standard Photoshop-style blend modes for RGBA pixel compositing,
//! including normal, multiply, screen, overlay, soft light, hard light, and more.

/// An RGBA color with 8-bit channels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rgba8 {
    /// Red channel (0-255).
    pub r: u8,
    /// Green channel (0-255).
    pub g: u8,
    /// Blue channel (0-255).
    pub b: u8,
    /// Alpha channel (0-255).
    pub a: u8,
}

impl Rgba8 {
    /// Create a new color.
    pub fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    /// Create an opaque color.
    pub fn opaque(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b, a: 255 }
    }

    /// Pure black.
    pub fn black() -> Self {
        Self::opaque(0, 0, 0)
    }

    /// Pure white.
    pub fn white() -> Self {
        Self::opaque(255, 255, 255)
    }

    /// Fully transparent.
    pub fn transparent() -> Self {
        Self::new(0, 0, 0, 0)
    }

    /// Convert to normalized f64 components.
    #[allow(clippy::cast_precision_loss)]
    pub fn to_f64(self) -> (f64, f64, f64, f64) {
        (
            self.r as f64 / 255.0,
            self.g as f64 / 255.0,
            self.b as f64 / 255.0,
            self.a as f64 / 255.0,
        )
    }

    /// Create from normalized f64 components.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub fn from_f64(r: f64, g: f64, b: f64, a: f64) -> Self {
        Self {
            r: (r.clamp(0.0, 1.0) * 255.0).round() as u8,
            g: (g.clamp(0.0, 1.0) * 255.0).round() as u8,
            b: (b.clamp(0.0, 1.0) * 255.0).round() as u8,
            a: (a.clamp(0.0, 1.0) * 255.0).round() as u8,
        }
    }

    /// Compute the luminance (ITU-R BT.709).
    #[allow(clippy::cast_precision_loss)]
    pub fn luminance(&self) -> f64 {
        0.2126 * (self.r as f64 / 255.0)
            + 0.7152 * (self.g as f64 / 255.0)
            + 0.0722 * (self.b as f64 / 255.0)
    }
}

/// Blending modes for compositing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlendMode {
    /// Normal alpha blending.
    Normal,
    /// Multiply: result = base * blend.
    Multiply,
    /// Screen: result = 1 - (1-base) * (1-blend).
    Screen,
    /// Overlay: multiply if base < 0.5, screen otherwise.
    Overlay,
    /// Darken: min(base, blend).
    Darken,
    /// Lighten: max(base, blend).
    Lighten,
    /// Color dodge: base / (1 - blend).
    ColorDodge,
    /// Color burn: 1 - (1-base)/blend.
    ColorBurn,
    /// Hard light: multiply if blend < 0.5, screen otherwise.
    HardLight,
    /// Soft light: gentle overlay.
    SoftLight,
    /// Difference: |base - blend|.
    Difference,
    /// Exclusion: base + blend - 2*base*blend.
    Exclusion,
    /// Additive: base + blend (clamped).
    Add,
}

impl BlendMode {
    /// Apply this blend mode to a single channel (0.0 to 1.0).
    fn blend_channel(self, base: f64, blend: f64) -> f64 {
        match self {
            Self::Normal => blend,
            Self::Multiply => base * blend,
            Self::Screen => 1.0 - (1.0 - base) * (1.0 - blend),
            Self::Overlay => {
                if base < 0.5 {
                    2.0 * base * blend
                } else {
                    1.0 - 2.0 * (1.0 - base) * (1.0 - blend)
                }
            }
            Self::Darken => base.min(blend),
            Self::Lighten => base.max(blend),
            Self::ColorDodge => {
                if blend >= 1.0 {
                    1.0
                } else {
                    (base / (1.0 - blend)).min(1.0)
                }
            }
            Self::ColorBurn => {
                if blend <= 0.0 {
                    0.0
                } else {
                    (1.0 - (1.0 - base) / blend).max(0.0)
                }
            }
            Self::HardLight => {
                if blend < 0.5 {
                    2.0 * base * blend
                } else {
                    1.0 - 2.0 * (1.0 - base) * (1.0 - blend)
                }
            }
            Self::SoftLight => {
                if blend <= 0.5 {
                    base - (1.0 - 2.0 * blend) * base * (1.0 - base)
                } else {
                    let d = if base <= 0.25 {
                        ((16.0 * base - 12.0) * base + 4.0) * base
                    } else {
                        base.sqrt()
                    };
                    base + (2.0 * blend - 1.0) * (d - base)
                }
            }
            Self::Difference => (base - blend).abs(),
            Self::Exclusion => base + blend - 2.0 * base * blend,
            Self::Add => (base + blend).min(1.0),
        }
    }
}

/// Blend two RGBA colors with a given mode and opacity.
pub fn blend_rgba(base: Rgba8, blend_color: Rgba8, mode: BlendMode, opacity: f64) -> Rgba8 {
    let (br, bg, bb, ba) = base.to_f64();
    let (sr, sg, sb, sa) = blend_color.to_f64();

    let eff_alpha = sa * opacity.clamp(0.0, 1.0);

    if eff_alpha <= 0.0 {
        return base;
    }

    // Apply blend mode per channel
    let cr = mode.blend_channel(br, sr);
    let cg = mode.blend_channel(bg, sg);
    let cb = mode.blend_channel(bb, sb);

    // Composite with alpha
    let out_a = eff_alpha + ba * (1.0 - eff_alpha);
    if out_a <= 0.0 {
        return Rgba8::transparent();
    }

    let out_r = (cr * eff_alpha + br * ba * (1.0 - eff_alpha)) / out_a;
    let out_g = (cg * eff_alpha + bg * ba * (1.0 - eff_alpha)) / out_a;
    let out_b = (cb * eff_alpha + bb * ba * (1.0 - eff_alpha)) / out_a;

    Rgba8::from_f64(out_r, out_g, out_b, out_a)
}

/// Blend a buffer of pixels in-place with a uniform overlay color.
pub fn blend_buffer(buffer: &mut [Rgba8], overlay: Rgba8, mode: BlendMode, opacity: f64) {
    for px in buffer.iter_mut() {
        *px = blend_rgba(*px, overlay, mode, opacity);
    }
}

/// Pre-multiply alpha for an RGBA color.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
pub fn premultiply_alpha(c: Rgba8) -> Rgba8 {
    let a = c.a as f64 / 255.0;
    Rgba8 {
        r: (c.r as f64 * a).round() as u8,
        g: (c.g as f64 * a).round() as u8,
        b: (c.b as f64 * a).round() as u8,
        a: c.a,
    }
}

/// Un-premultiply alpha for an RGBA color.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
pub fn unpremultiply_alpha(c: Rgba8) -> Rgba8 {
    if c.a == 0 {
        return Rgba8::transparent();
    }
    let a = c.a as f64 / 255.0;
    Rgba8 {
        r: (c.r as f64 / a).round().min(255.0) as u8,
        g: (c.g as f64 / a).round().min(255.0) as u8,
        b: (c.b as f64 / a).round().min(255.0) as u8,
        a: c.a,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rgba8_new() {
        let c = Rgba8::new(100, 150, 200, 255);
        assert_eq!(c.r, 100);
        assert_eq!(c.g, 150);
        assert_eq!(c.b, 200);
        assert_eq!(c.a, 255);
    }

    #[test]
    fn test_rgba8_black_white() {
        assert_eq!(Rgba8::black(), Rgba8::opaque(0, 0, 0));
        assert_eq!(Rgba8::white(), Rgba8::opaque(255, 255, 255));
    }

    #[test]
    fn test_to_f64_and_back() {
        let c = Rgba8::opaque(128, 64, 192);
        let (r, g, b, a) = c.to_f64();
        let c2 = Rgba8::from_f64(r, g, b, a);
        assert_eq!(c, c2);
    }

    #[test]
    fn test_luminance() {
        let white = Rgba8::white();
        assert!((white.luminance() - 1.0).abs() < 0.01);
        let black = Rgba8::black();
        assert!(black.luminance().abs() < 0.01);
    }

    #[test]
    fn test_normal_blend_opaque() {
        let base = Rgba8::opaque(100, 100, 100);
        let overlay = Rgba8::opaque(200, 200, 200);
        let result = blend_rgba(base, overlay, BlendMode::Normal, 1.0);
        assert_eq!(result.r, 200);
        assert_eq!(result.g, 200);
        assert_eq!(result.b, 200);
    }

    #[test]
    fn test_multiply_blend() {
        let base = Rgba8::opaque(255, 255, 255);
        let overlay = Rgba8::opaque(128, 128, 128);
        let result = blend_rgba(base, overlay, BlendMode::Multiply, 1.0);
        // 1.0 * 0.502 ~ 128
        assert!((result.r as i16 - 128).abs() <= 1);
    }

    #[test]
    fn test_screen_blend() {
        let base = Rgba8::opaque(0, 0, 0);
        let overlay = Rgba8::opaque(128, 128, 128);
        let result = blend_rgba(base, overlay, BlendMode::Screen, 1.0);
        // Screen with black base returns the overlay
        assert!((result.r as i16 - 128).abs() <= 1);
    }

    #[test]
    fn test_overlay_blend() {
        let base = Rgba8::opaque(64, 64, 64);
        let overlay = Rgba8::opaque(200, 200, 200);
        let result = blend_rgba(base, overlay, BlendMode::Overlay, 1.0);
        // base < 0.5 so multiply-like
        assert!(result.r < 128);
    }

    #[test]
    fn test_darken_lighten() {
        let base = Rgba8::opaque(100, 200, 50);
        let overlay = Rgba8::opaque(150, 100, 100);
        let dark = blend_rgba(base, overlay, BlendMode::Darken, 1.0);
        let light = blend_rgba(base, overlay, BlendMode::Lighten, 1.0);
        assert_eq!(dark.r, 100);
        assert_eq!(dark.g, 100);
        assert_eq!(light.r, 150);
        assert_eq!(light.g, 200);
    }

    #[test]
    fn test_difference_blend() {
        let base = Rgba8::opaque(200, 100, 50);
        let overlay = Rgba8::opaque(100, 200, 50);
        let result = blend_rgba(base, overlay, BlendMode::Difference, 1.0);
        assert!((result.r as i16 - 100).abs() <= 1);
        assert!((result.g as i16 - 100).abs() <= 1);
        assert_eq!(result.b, 0);
    }

    #[test]
    fn test_add_blend() {
        let base = Rgba8::opaque(200, 200, 200);
        let overlay = Rgba8::opaque(100, 100, 100);
        let result = blend_rgba(base, overlay, BlendMode::Add, 1.0);
        // Clamped to 255
        assert_eq!(result.r, 255);
    }

    #[test]
    fn test_blend_with_zero_opacity() {
        let base = Rgba8::opaque(100, 100, 100);
        let overlay = Rgba8::opaque(200, 200, 200);
        let result = blend_rgba(base, overlay, BlendMode::Normal, 0.0);
        assert_eq!(result, base);
    }

    #[test]
    fn test_premultiply_unpremultiply() {
        let c = Rgba8::new(200, 100, 50, 128);
        let pre = premultiply_alpha(c);
        let un = unpremultiply_alpha(pre);
        // Should round-trip within +-1
        assert!((un.r as i16 - c.r as i16).abs() <= 1);
        assert!((un.g as i16 - c.g as i16).abs() <= 1);
        assert!((un.b as i16 - c.b as i16).abs() <= 1);
    }

    #[test]
    fn test_blend_buffer() {
        let mut buf = vec![Rgba8::opaque(100, 100, 100); 4];
        blend_buffer(
            &mut buf,
            Rgba8::opaque(200, 200, 200),
            BlendMode::Normal,
            1.0,
        );
        for px in &buf {
            assert_eq!(px.r, 200);
        }
    }

    #[test]
    fn test_color_dodge_blend() {
        let base = Rgba8::opaque(128, 128, 128);
        let overlay = Rgba8::opaque(128, 128, 128);
        let result = blend_rgba(base, overlay, BlendMode::ColorDodge, 1.0);
        // base / (1 - blend) = 0.502 / (1 - 0.502) ~ 1.0 => 255
        assert_eq!(result.r, 255);
    }

    #[test]
    fn test_exclusion_blend() {
        let base = Rgba8::opaque(128, 128, 128);
        let overlay = Rgba8::opaque(128, 128, 128);
        let result = blend_rgba(base, overlay, BlendMode::Exclusion, 1.0);
        // base + blend - 2*base*blend ~ 0.502 + 0.502 - 2*0.252 ~ 0.5
        assert!((result.r as i16 - 128).abs() <= 2);
    }

    #[test]
    fn test_unpremultiply_transparent() {
        let c = Rgba8::transparent();
        let un = unpremultiply_alpha(c);
        assert_eq!(un, Rgba8::transparent());
    }
}
