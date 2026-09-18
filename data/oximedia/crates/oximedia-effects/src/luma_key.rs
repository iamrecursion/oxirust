//! Luminance (luma) keying for video effects.
//!
//! Provides threshold-based luma keying to generate alpha mattes from
//! brightness values, with soft-edge falloff and optional inversion.

#![allow(dead_code)]
#![allow(missing_docs)]

/// Parameters controlling a luma key operation.
#[derive(Debug, Clone)]
pub struct LumaKeyParams {
    /// Lower boundary of the key range (pixels below this are keyed out).
    pub low_threshold: f32,
    /// Upper boundary of the key range (pixels above this are keyed out).
    pub high_threshold: f32,
    /// Width of the soft-edge feather zone (0 = hard edge).
    pub feather: f32,
    /// If `true`, invert the resulting matte (key the bright areas instead).
    pub invert: bool,
}

impl LumaKeyParams {
    /// Create a new luma key configuration.
    #[must_use]
    pub fn new(low_threshold: f32, high_threshold: f32, feather: f32, invert: bool) -> Self {
        Self {
            low_threshold: low_threshold.clamp(0.0, 1.0),
            high_threshold: high_threshold.clamp(0.0, 1.0),
            feather: feather.max(0.0),
            invert,
        }
    }

    /// Preset that keys out dark areas (keeps bright areas).
    #[must_use]
    pub fn bright_key() -> Self {
        Self::new(0.5, 1.0, 0.05, false)
    }

    /// Preset that keys out bright areas (keeps dark areas).
    #[must_use]
    pub fn dark_key() -> Self {
        Self::new(0.0, 0.5, 0.05, true)
    }
}

/// Compute the alpha value for a single luma sample given the key parameters.
///
/// Returns a value in `[0.0, 1.0]` where `1.0` is fully opaque (kept)
/// and `0.0` is fully transparent (keyed out).
#[must_use]
pub fn key_pixel_luma(luma: f32, params: &LumaKeyParams) -> f32 {
    let luma = luma.clamp(0.0, 1.0);
    let lo = params.low_threshold;
    let hi = params.high_threshold;
    let f = params.feather.max(1e-6);

    // Ramp up from lo-f to lo, fully opaque from lo to hi, ramp down hi to hi+f.
    let alpha = if luma < lo - f {
        0.0_f32
    } else if luma < lo {
        (luma - (lo - f)) / f
    } else if luma <= hi {
        1.0_f32
    } else if luma < hi + f {
        1.0_f32 - (luma - hi) / f
    } else {
        0.0_f32
    };

    let alpha = alpha.clamp(0.0, 1.0);
    if params.invert {
        1.0 - alpha
    } else {
        alpha
    }
}

/// Apply luma keying to an RGBA pixel slice (4 bytes per pixel).
///
/// The ITU-R BT.709 luma weights are used to derive luma from R, G, B.
/// The computed alpha replaces the existing alpha channel.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn apply_luma_key(pixels_rgba: &mut [u8], params: &LumaKeyParams) {
    for chunk in pixels_rgba.chunks_exact_mut(4) {
        let r = f32::from(chunk[0]) / 255.0;
        let g = f32::from(chunk[1]) / 255.0;
        let b = f32::from(chunk[2]) / 255.0;
        // BT.709 luma coefficients
        let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        let alpha = key_pixel_luma(luma, params);
        chunk[3] = (alpha * 255.0).round() as u8;
    }
}

/// Combine two alpha mattes using a simple multiply operation.
///
/// Useful for refining a luma matte with a second mask.
#[must_use]
pub fn combine_alpha_multiply(a: f32, b: f32) -> f32 {
    (a * b).clamp(0.0, 1.0)
}

/// Combine two alpha mattes using a screen operation (prevents double-darkening).
#[must_use]
pub fn combine_alpha_screen(a: f32, b: f32) -> f32 {
    (1.0 - (1.0 - a) * (1.0 - b)).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_key_pixel_luma_inside_range() {
        let params = LumaKeyParams::new(0.2, 0.8, 0.0, false);
        assert!((key_pixel_luma(0.5, &params) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_key_pixel_luma_below_range() {
        let params = LumaKeyParams::new(0.2, 0.8, 0.0, false);
        assert!((key_pixel_luma(0.1, &params)).abs() < 1e-5);
    }

    #[test]
    fn test_key_pixel_luma_above_range() {
        let params = LumaKeyParams::new(0.2, 0.8, 0.0, false);
        assert!((key_pixel_luma(0.9, &params)).abs() < 1e-5);
    }

    #[test]
    fn test_key_pixel_luma_inverted() {
        let params = LumaKeyParams::new(0.2, 0.8, 0.0, true);
        // inside range → originally 1.0, inverted → 0.0
        assert!((key_pixel_luma(0.5, &params)).abs() < 1e-5);
        // outside range → originally 0.0, inverted → 1.0
        assert!((key_pixel_luma(0.1, &params) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_key_pixel_luma_feather_lower() {
        // feather of 0.1: at luma = low_threshold, alpha should be 1.0
        let params = LumaKeyParams::new(0.3, 0.7, 0.1, false);
        let alpha = key_pixel_luma(0.3, &params);
        assert!((alpha - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_key_pixel_luma_feather_midpoint() {
        // midpoint of lower feather zone: 0.5 alpha
        let params = LumaKeyParams::new(0.3, 0.7, 0.1, false);
        let alpha = key_pixel_luma(0.25, &params); // halfway between 0.2 and 0.3
        assert!((alpha - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_apply_luma_key_modifies_alpha() {
        let mut pixels = vec![
            200u8, 200, 200, 255, // bright pixel
            10u8, 10, 10, 255, // dark pixel
        ];
        let params = LumaKeyParams::new(0.5, 1.0, 0.0, false);
        apply_luma_key(&mut pixels, &params);
        // bright pixel → keyed (alpha > 0)
        assert!(pixels[3] > 0);
        // dark pixel → keyed out (alpha == 0)
        assert_eq!(pixels[7], 0);
    }

    #[test]
    fn test_apply_luma_key_empty_slice() {
        let mut pixels: Vec<u8> = vec![];
        let params = LumaKeyParams::new(0.2, 0.8, 0.0, false);
        apply_luma_key(&mut pixels, &params);
        // should not panic
    }

    #[test]
    fn test_combine_alpha_multiply() {
        assert!((combine_alpha_multiply(0.5, 0.5) - 0.25).abs() < 1e-5);
        assert_eq!(combine_alpha_multiply(1.0, 1.0), 1.0);
        assert_eq!(combine_alpha_multiply(0.0, 1.0), 0.0);
    }

    #[test]
    fn test_combine_alpha_screen() {
        // screen(0.5, 0.5) = 1 - 0.5*0.5 = 0.75
        assert!((combine_alpha_screen(0.5, 0.5) - 0.75).abs() < 1e-5);
        assert!((combine_alpha_screen(1.0, 1.0) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_bright_key_preset() {
        let p = LumaKeyParams::bright_key();
        assert_eq!(p.invert, false);
        assert!(p.low_threshold >= 0.4);
    }

    #[test]
    fn test_dark_key_preset() {
        let p = LumaKeyParams::dark_key();
        assert_eq!(p.invert, true);
    }

    #[test]
    fn test_luma_key_clamp_out_of_range_input() {
        let params = LumaKeyParams::new(0.2, 0.8, 0.0, false);
        // Should not panic or return out-of-range
        let a = key_pixel_luma(-0.5, &params);
        let b = key_pixel_luma(1.5, &params);
        assert!(a >= 0.0 && a <= 1.0);
        assert!(b >= 0.0 && b <= 1.0);
    }
}
