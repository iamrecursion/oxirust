//! Color adjustment operations: brightness/contrast, HSL, color grading, gamma.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

/// Brightness and contrast adjustment parameters.
#[derive(Clone, Copy, Debug)]
pub struct BrightnessContrast {
    /// Brightness offset in [-255, 255].
    pub brightness: f32,
    /// Contrast multiplier around mid-grey (0.5).  1.0 = no change.
    pub contrast: f32,
}

impl BrightnessContrast {
    /// Create a new `BrightnessContrast`.
    #[must_use]
    pub const fn new(brightness: f32, contrast: f32) -> Self {
        Self {
            brightness,
            contrast,
        }
    }

    /// Apply brightness/contrast to a single u8 pixel value.
    ///
    /// Formula: out = (pixel/255 - 0.5) * contrast + 0.5 + brightness/255
    /// Result is clamped to [0, 255].
    #[must_use]
    pub fn apply(self, pixel: u8) -> u8 {
        let normalized = pixel as f32 / 255.0;
        let adjusted = (normalized - 0.5) * self.contrast + 0.5 + self.brightness / 255.0;
        (adjusted * 255.0).clamp(0.0, 255.0).round() as u8
    }
}

impl Default for BrightnessContrast {
    fn default() -> Self {
        Self {
            brightness: 0.0,
            contrast: 1.0,
        }
    }
}

/// Hue, saturation, and lightness adjustment parameters.
#[derive(Clone, Copy, Debug)]
pub struct HueSaturationLightness {
    /// Hue shift in degrees [-180, 180].
    pub hue_shift: f32,
    /// Saturation multiplier. 1.0 = no change, 0.0 = greyscale.
    pub saturation: f32,
    /// Lightness offset in [-1.0, 1.0].
    pub lightness: f32,
}

impl HueSaturationLightness {
    /// Create a new `HueSaturationLightness`.
    #[must_use]
    pub const fn new(hue_shift: f32, saturation: f32, lightness: f32) -> Self {
        Self {
            hue_shift,
            saturation,
            lightness,
        }
    }

    /// Shift a hue value (in degrees [0, 360]) by `self.hue_shift`, wrapping into [0, 360).
    #[must_use]
    pub fn shift_hue_pixel(&self, h: f32) -> f32 {
        let shifted = h + self.hue_shift;
        shifted.rem_euclid(360.0)
    }
}

impl Default for HueSaturationLightness {
    fn default() -> Self {
        Self {
            hue_shift: 0.0,
            saturation: 1.0,
            lightness: 0.0,
        }
    }
}

/// Three-zone color grade: separate control over shadows, midtones, highlights.
#[derive(Clone, Copy, Debug)]
pub struct ColorGradeParams {
    /// Shadows lift/lower (additive offset for dark pixels).
    pub shadows: f32,
    /// Midtones pivot gain.
    pub midtones: f32,
    /// Highlights roll-off (additive offset for bright pixels).
    pub highlights: f32,
}

impl ColorGradeParams {
    /// Create a new `ColorGradeParams`.
    #[must_use]
    pub const fn new(shadows: f32, midtones: f32, highlights: f32) -> Self {
        Self {
            shadows,
            midtones,
            highlights,
        }
    }

    /// Apply zone-based grading to a normalised luminance value in [0, 1].
    ///
    /// The three zones blend smoothly across the luminance range and the result
    /// is clamped to [0, 1].
    #[must_use]
    pub fn apply_luminance(self, luma: f32) -> f32 {
        // Shadow weight: strong at luma=0, zero at luma=1
        let shadow_weight = (1.0 - luma).powi(2);
        // Highlight weight: strong at luma=1, zero at luma=0
        let highlight_weight = luma.powi(2);
        // Midtone weight: bell centred at luma=0.5
        let midtone_weight = 1.0 - shadow_weight - highlight_weight;

        let result = luma
            + self.shadows * shadow_weight
            + self.midtones * midtone_weight
            + self.highlights * highlight_weight;

        result.clamp(0.0, 1.0)
    }
}

impl Default for ColorGradeParams {
    fn default() -> Self {
        Self {
            shadows: 0.0,
            midtones: 0.0,
            highlights: 0.0,
        }
    }
}

/// Apply `BrightnessContrast` to every pixel in a byte slice.
#[must_use]
pub fn apply_brightness_contrast(pixels: &[u8], bc: &BrightnessContrast) -> Vec<u8> {
    pixels.iter().map(|&p| bc.apply(p)).collect()
}

/// Apply gamma correction to a byte slice.
///
/// Each pixel `p` is mapped via `(p/255)^gamma * 255`.
/// `gamma` must be positive; values < 1 brighten, > 1 darken.
#[must_use]
pub fn adjust_gamma(pixels: &[u8], gamma: f32) -> Vec<u8> {
    assert!(gamma > 0.0, "gamma must be positive");
    pixels
        .iter()
        .map(|&p| {
            let normalized = p as f32 / 255.0;
            (normalized.powf(gamma) * 255.0).clamp(0.0, 255.0).round() as u8
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------- BrightnessContrast tests ----------

    #[test]
    fn test_bc_default_identity() {
        let bc = BrightnessContrast::default();
        for p in [0u8, 128, 255] {
            assert_eq!(bc.apply(p), p, "default BC should be identity for {p}");
        }
    }

    #[test]
    fn test_bc_brightness_increase() {
        let bc = BrightnessContrast::new(50.0, 1.0);
        assert!(bc.apply(100) > 100);
    }

    #[test]
    fn test_bc_brightness_decrease() {
        let bc = BrightnessContrast::new(-50.0, 1.0);
        assert!(bc.apply(100) < 100);
    }

    #[test]
    fn test_bc_contrast_increase() {
        let bc = BrightnessContrast::new(0.0, 2.0);
        // Mid-grey (128) should stay near 128
        let mid = bc.apply(128);
        assert!((mid as i16 - 128).abs() <= 2);
        // Dark pixel should become darker
        assert!(bc.apply(50) < 50);
        // Bright pixel should become brighter
        assert!(bc.apply(200) > 200);
    }

    #[test]
    fn test_bc_clamp_upper() {
        let bc = BrightnessContrast::new(255.0, 1.0);
        assert_eq!(bc.apply(200), 255);
    }

    #[test]
    fn test_bc_clamp_lower() {
        let bc = BrightnessContrast::new(-255.0, 1.0);
        assert_eq!(bc.apply(10), 0);
    }

    // ---------- HueSaturationLightness tests ----------

    #[test]
    fn test_hsl_shift_hue_no_wrap() {
        let hsl = HueSaturationLightness::new(30.0, 1.0, 0.0);
        let result = hsl.shift_hue_pixel(100.0);
        assert!((result - 130.0).abs() < 1e-4);
    }

    #[test]
    fn test_hsl_shift_hue_wrap_positive() {
        let hsl = HueSaturationLightness::new(40.0, 1.0, 0.0);
        let result = hsl.shift_hue_pixel(340.0);
        assert!((result - 20.0).abs() < 1e-4);
    }

    #[test]
    fn test_hsl_shift_hue_wrap_negative() {
        let hsl = HueSaturationLightness::new(-30.0, 1.0, 0.0);
        let result = hsl.shift_hue_pixel(10.0);
        assert!((result - 340.0).abs() < 1e-4);
    }

    #[test]
    fn test_hsl_zero_shift() {
        let hsl = HueSaturationLightness::default();
        assert!((hsl.shift_hue_pixel(200.0) - 200.0).abs() < 1e-4);
    }

    // ---------- ColorGradeParams tests ----------

    #[test]
    fn test_grade_default_identity() {
        let cg = ColorGradeParams::default();
        for luma in [0.0f32, 0.25, 0.5, 0.75, 1.0] {
            let out = cg.apply_luminance(luma);
            assert!(
                (out - luma).abs() < 1e-5,
                "default grade should be identity at {luma}"
            );
        }
    }

    #[test]
    fn test_grade_shadows_lift() {
        let cg = ColorGradeParams::new(0.1, 0.0, 0.0);
        // Dark pixels should be lifted
        assert!(cg.apply_luminance(0.0) > 0.0);
        // Bright pixels largely unaffected
        let bright = cg.apply_luminance(1.0);
        assert!((bright - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_grade_highlights_push() {
        let cg = ColorGradeParams::new(0.0, 0.0, 0.05);
        assert!(cg.apply_luminance(1.0) >= 1.0); // clamped
        assert!(cg.apply_luminance(0.8) > 0.8);
        // Dark pixels unaffected
        let dark = cg.apply_luminance(0.0);
        assert!((dark - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_grade_clamped_range() {
        let cg = ColorGradeParams::new(1.0, 1.0, 1.0);
        for luma in [0.0f32, 0.5, 1.0] {
            let out = cg.apply_luminance(luma);
            assert!(out >= 0.0 && out <= 1.0, "out of range: {out}");
        }
    }

    // ---------- apply_brightness_contrast tests ----------

    #[test]
    fn test_apply_brightness_contrast_slice() {
        let pixels = vec![100u8, 150, 200];
        let bc = BrightnessContrast::new(0.0, 1.0);
        let out = apply_brightness_contrast(&pixels, &bc);
        assert_eq!(out, pixels);
    }

    // ---------- adjust_gamma tests ----------

    #[test]
    fn test_gamma_1_identity() {
        let pixels: Vec<u8> = (0u8..=255).collect();
        let out = adjust_gamma(&pixels, 1.0);
        for (&orig, &adj) in pixels.iter().zip(out.iter()) {
            assert!((orig as i16 - adj as i16).abs() <= 1);
        }
    }

    #[test]
    fn test_gamma_brightens() {
        // gamma < 1 brightens the image
        let pixels = vec![100u8, 150];
        let out = adjust_gamma(&pixels, 0.5);
        assert!(out[0] > pixels[0]);
        assert!(out[1] > pixels[1]);
    }

    #[test]
    fn test_gamma_darkens() {
        // gamma > 1 darkens the image
        let pixels = vec![100u8, 150];
        let out = adjust_gamma(&pixels, 2.0);
        assert!(out[0] < pixels[0]);
        assert!(out[1] < pixels[1]);
    }

    #[test]
    fn test_gamma_endpoints_preserved() {
        let pixels = vec![0u8, 255];
        let out = adjust_gamma(&pixels, 2.2);
        assert_eq!(out[0], 0);
        assert_eq!(out[1], 255);
    }
}
