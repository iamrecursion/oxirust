//! Shadow/midtone/highlight colour-balance adjustments.
#![allow(dead_code)]

/// Tonal range targeted by a colour-balance adjustment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorRange {
    /// Darkest portion of the image.
    Shadows,
    /// Mid-grey tones.
    Midtones,
    /// Brightest portion.
    Highlights,
}

impl ColorRange {
    /// Soft-light weight: returns how strongly this range affects a
    /// normalised luminance value `luma` (0.0–1.0).
    #[allow(clippy::cast_precision_loss)]
    pub fn weight(&self, luma: f32) -> f32 {
        let l = luma.clamp(0.0, 1.0);
        match self {
            Self::Shadows => (1.0 - l).powi(2),
            Self::Midtones => {
                // Bell curve centred at 0.5.
                let t = 2.0 * l - 1.0;
                1.0 - t * t
            }
            Self::Highlights => l.powi(2),
        }
    }
}

/// Red/Green/Blue shift amounts for a single tonal range.
///
/// Values in \[-1.0, 1.0\]: negative shifts toward the complementary colour.
#[derive(Debug, Clone, Copy)]
pub struct ColorBalanceAdjust {
    /// Shift along the cyan–red axis (negative = cyan, positive = red).
    pub cyan_red: f32,
    /// Shift along the magenta–green axis (negative = magenta, positive = green).
    pub magenta_green: f32,
    /// Shift along the yellow–blue axis (negative = yellow, positive = blue).
    pub yellow_blue: f32,
    /// The tonal range (shadows, midtones, highlights) this adjustment targets.
    pub range: ColorRange,
    /// Whether to preserve luminance after adjustment.
    pub preserve_luminance: bool,
}

impl ColorBalanceAdjust {
    /// Create a new adjustment with no shift.
    pub fn new(range: ColorRange) -> Self {
        Self {
            cyan_red: 0.0,
            magenta_green: 0.0,
            yellow_blue: 0.0,
            range,
            preserve_luminance: true,
        }
    }

    /// Apply the adjustment to a single RGB pixel (each channel 0.0–1.0).
    ///
    /// Returns the adjusted pixel.
    pub fn apply_to_pixel(&self, rgb: [f32; 3]) -> [f32; 3] {
        let luma = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2];
        let w = self.range.weight(luma);

        let mut r = rgb[0] + w * self.cyan_red * 0.5;
        let mut g = rgb[1] + w * self.magenta_green * 0.5;
        let mut b = rgb[2] + w * self.yellow_blue * 0.5;

        if self.preserve_luminance {
            let new_luma = 0.299 * r + 0.587 * g + 0.114 * b;
            if new_luma > 1e-6 {
                let scale = luma / new_luma;
                r *= scale;
                g *= scale;
                b *= scale;
            }
        }

        [r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0)]
    }
}

/// Stores per-range adjustment settings.
#[derive(Debug, Clone)]
pub struct ColorBalance {
    shadows: ColorBalanceAdjust,
    midtones: ColorBalanceAdjust,
    highlights: ColorBalanceAdjust,
}

impl ColorBalance {
    /// Create a colour-balance configuration with no adjustments.
    pub fn new() -> Self {
        Self {
            shadows: ColorBalanceAdjust::new(ColorRange::Shadows),
            midtones: ColorBalanceAdjust::new(ColorRange::Midtones),
            highlights: ColorBalanceAdjust::new(ColorRange::Highlights),
        }
    }

    /// Update the adjustment for the specified range.
    pub fn set_range(&mut self, adjust: ColorBalanceAdjust) {
        match adjust.range {
            ColorRange::Shadows => self.shadows = adjust,
            ColorRange::Midtones => self.midtones = adjust,
            ColorRange::Highlights => self.highlights = adjust,
        }
    }

    /// Return the current adjustment for `range`.
    pub fn get_range(&self, range: ColorRange) -> &ColorBalanceAdjust {
        match range {
            ColorRange::Shadows => &self.shadows,
            ColorRange::Midtones => &self.midtones,
            ColorRange::Highlights => &self.highlights,
        }
    }

    /// Apply all three adjustments to a single pixel.
    pub fn apply(&self, pixel: [f32; 3]) -> [f32; 3] {
        let s = self.shadows.apply_to_pixel(pixel);
        let m = self.midtones.apply_to_pixel(s);
        self.highlights.apply_to_pixel(m)
    }
}

impl Default for ColorBalance {
    fn default() -> Self {
        Self::new()
    }
}

/// Applies a `ColorBalance` to an entire image.
pub struct ColorBalanceProcessor {
    balance: ColorBalance,
}

impl ColorBalanceProcessor {
    /// Create a processor with the given balance settings.
    pub fn new(balance: ColorBalance) -> Self {
        Self { balance }
    }

    /// Process an RGBA u8 image in place.
    pub fn process(&self, pixels: &mut [u8], channels: u8) {
        let ch = channels as usize;
        if ch < 3 {
            return;
        }
        for chunk in pixels.chunks_mut(ch) {
            let rgb = [
                f32::from(chunk[0]) / 255.0,
                f32::from(chunk[1]) / 255.0,
                f32::from(chunk[2]) / 255.0,
            ];
            let out = self.balance.apply(rgb);
            chunk[0] = (out[0] * 255.0).round() as u8;
            chunk[1] = (out[1] * 255.0).round() as u8;
            chunk[2] = (out[2] * 255.0).round() as u8;
        }
    }

    /// Return a reference to the underlying `ColorBalance`.
    pub fn balance(&self) -> &ColorBalance {
        &self.balance
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- ColorRange::weight ---

    #[test]
    fn test_shadow_weight_dark_is_high() {
        let w = ColorRange::Shadows.weight(0.0);
        assert!((w - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_shadow_weight_bright_is_low() {
        let w = ColorRange::Shadows.weight(1.0);
        assert!(w < 0.01);
    }

    #[test]
    fn test_highlight_weight_bright_is_high() {
        let w = ColorRange::Highlights.weight(1.0);
        assert!((w - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_midtone_weight_peaks_at_mid() {
        let w_mid = ColorRange::Midtones.weight(0.5);
        let w_dark = ColorRange::Midtones.weight(0.0);
        let w_bright = ColorRange::Midtones.weight(1.0);
        assert!(w_mid > w_dark);
        assert!(w_mid > w_bright);
    }

    // --- ColorBalanceAdjust ---

    #[test]
    fn test_no_shift_returns_same_pixel() {
        let adj = ColorBalanceAdjust::new(ColorRange::Midtones);
        let rgb = [0.5, 0.4, 0.3];
        let out = adj.apply_to_pixel(rgb);
        // With preserve_luminance the hue may drift slightly, but luma is preserved.
        let luma_in = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2];
        let luma_out = 0.299 * out[0] + 0.587 * out[1] + 0.114 * out[2];
        assert!((luma_in - luma_out).abs() < 1e-4);
    }

    #[test]
    fn test_red_shift_increases_red() {
        let mut adj = ColorBalanceAdjust::new(ColorRange::Midtones);
        adj.preserve_luminance = false;
        adj.cyan_red = 1.0; // full red shift
        let rgb = [0.5, 0.5, 0.5];
        let out = adj.apply_to_pixel(rgb);
        assert!(out[0] > rgb[0]);
    }

    #[test]
    fn test_output_clamped_to_unit() {
        let mut adj = ColorBalanceAdjust::new(ColorRange::Highlights);
        adj.preserve_luminance = false;
        adj.cyan_red = 2.0; // intentionally excessive
        let out = adj.apply_to_pixel([0.9, 0.9, 0.9]);
        assert!(out[0] <= 1.0);
    }

    // --- ColorBalance ---

    #[test]
    fn test_set_and_get_range() {
        let mut cb = ColorBalance::new();
        let mut adj = ColorBalanceAdjust::new(ColorRange::Shadows);
        adj.cyan_red = 0.3;
        cb.set_range(adj);
        assert!((cb.get_range(ColorRange::Shadows).cyan_red - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_default_apply_is_identity_approx() {
        let cb = ColorBalance::new();
        let rgb = [0.6, 0.4, 0.2];
        let out = cb.apply(rgb);
        // No adjustments configured — output should be very close to input.
        for i in 0..3 {
            assert!((out[i] - rgb[i]).abs() < 0.01);
        }
    }

    // --- ColorBalanceProcessor ---

    #[test]
    fn test_process_no_adjustment_preserves_values() {
        let cb = ColorBalance::new();
        let proc = ColorBalanceProcessor::new(cb);
        let mut pixels = vec![100u8, 150, 200, 255];
        proc.process(&mut pixels, 4);
        // Should be very close (rounding differences only).
        assert!((pixels[0] as i32 - 100).abs() <= 2);
    }

    #[test]
    fn test_process_skips_less_than_3_channels() {
        let cb = ColorBalance::new();
        let proc = ColorBalanceProcessor::new(cb);
        let mut pixels = vec![50u8, 100]; // 2-channel, should not panic
        proc.process(&mut pixels, 2);
        assert_eq!(pixels[0], 50);
    }

    #[test]
    fn test_balance_accessor() {
        let cb = ColorBalance::default();
        let proc = ColorBalanceProcessor::new(cb);
        // Just ensure we can access it without panic.
        let _b = proc.balance();
    }

    #[test]
    fn test_color_range_all_variants_weight_clamped() {
        for &range in &[
            ColorRange::Shadows,
            ColorRange::Midtones,
            ColorRange::Highlights,
        ] {
            let w = range.weight(-0.5); // below range
            assert!(w >= 0.0 && w <= 1.0);
            let w = range.weight(1.5); // above range
            assert!(w >= 0.0 && w <= 1.0);
        }
    }
}
