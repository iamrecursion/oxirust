//! Tilt-shift miniature effect.
//!
//! Simulates a tilt-shift lens by applying a gradient blur mask: the centre
//! band stays sharp while the top and bottom regions receive progressively
//! stronger blur, creating the illusion of a miniature model.

use crate::{EffectParams, Frame, VfxError, VfxResult, VideoEffect};
use serde::{Deserialize, Serialize};

/// Configuration for the tilt-shift miniature effect.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TiltShiftConfig {
    /// Vertical centre of the sharp band (normalised 0.0-1.0, 0.5 = middle).
    pub focus_position: f32,
    /// Width of the fully sharp band (normalised 0.0-1.0).
    pub focus_width: f32,
    /// Transition zone width from sharp to full blur (normalised).
    pub transition_width: f32,
    /// Maximum blur radius in pixels.
    pub max_blur_radius: u32,
    /// Saturation boost in the focus region (1.0 = no change, >1 = more vivid).
    pub saturation_boost: f32,
    /// Angle of the focus band in degrees (0 = horizontal).
    pub angle_degrees: f32,
}

impl Default for TiltShiftConfig {
    fn default() -> Self {
        Self {
            focus_position: 0.5,
            focus_width: 0.15,
            transition_width: 0.15,
            max_blur_radius: 8,
            saturation_boost: 1.1,
            angle_degrees: 0.0,
        }
    }
}

impl TiltShiftConfig {
    /// Compute the blur amount (0.0-1.0) for a given normalised y position.
    ///
    /// Accounts for the focus band position, width, and transition zone.
    #[must_use]
    pub fn blur_amount_at(&self, ny: f32) -> f32 {
        let half_focus = self.focus_width / 2.0;
        let dist = (ny - self.focus_position).abs();

        if dist <= half_focus {
            0.0
        } else if self.transition_width <= 0.0 {
            1.0
        } else {
            let t = ((dist - half_focus) / self.transition_width).min(1.0);
            // Smooth hermite transition
            t * t * (3.0 - 2.0 * t)
        }
    }

    /// Effective blur radius in pixels at a given normalised y.
    #[must_use]
    pub fn blur_radius_at(&self, ny: f32) -> u32 {
        let amount = self.blur_amount_at(ny);
        (amount * self.max_blur_radius as f32).round() as u32
    }
}

/// Tilt-shift miniature video effect.
///
/// Applies a gradient-masked depth-of-field blur to simulate a tilt-shift
/// lens, creating a miniature/diorama look.
pub struct TiltShiftEffect {
    config: TiltShiftConfig,
}

impl TiltShiftEffect {
    /// Create a new tilt-shift effect with the given configuration.
    #[must_use]
    pub fn new(config: TiltShiftConfig) -> Self {
        Self { config }
    }

    /// Create with default settings.
    #[must_use]
    pub fn default_miniature() -> Self {
        Self::new(TiltShiftConfig::default())
    }

    /// Set focus position.
    #[must_use]
    pub fn with_focus_position(mut self, pos: f32) -> Self {
        self.config.focus_position = pos.clamp(0.0, 1.0);
        self
    }

    /// Set focus width.
    #[must_use]
    pub fn with_focus_width(mut self, w: f32) -> Self {
        self.config.focus_width = w.clamp(0.0, 1.0);
        self
    }

    /// Set max blur radius.
    #[must_use]
    pub fn with_max_blur(mut self, radius: u32) -> Self {
        self.config.max_blur_radius = radius.min(32);
        self
    }

    /// Apply a simple box blur to a single pixel by averaging neighbours.
    ///
    /// Returns the blurred RGBA value.
    fn box_blur_pixel(input: &Frame, x: u32, y: u32, radius: u32) -> [u8; 4] {
        if radius == 0 {
            return input.get_pixel(x, y).unwrap_or([0, 0, 0, 0]);
        }

        let r = radius as i64;
        let mut sum_r = 0u32;
        let mut sum_g = 0u32;
        let mut sum_b = 0u32;
        let mut sum_a = 0u32;
        let mut count = 0u32;

        for dy in -r..=r {
            for dx in -r..=r {
                let sx = (x as i64 + dx).clamp(0, input.width as i64 - 1) as u32;
                let sy = (y as i64 + dy).clamp(0, input.height as i64 - 1) as u32;
                if let Some(p) = input.get_pixel(sx, sy) {
                    sum_r += p[0] as u32;
                    sum_g += p[1] as u32;
                    sum_b += p[2] as u32;
                    sum_a += p[3] as u32;
                    count += 1;
                }
            }
        }

        if count == 0 {
            return [0, 0, 0, 0];
        }

        [
            (sum_r / count) as u8,
            (sum_g / count) as u8,
            (sum_b / count) as u8,
            (sum_a / count) as u8,
        ]
    }

    /// Apply saturation boost to a pixel.
    fn boost_saturation(pixel: [u8; 4], boost: f32) -> [u8; 4] {
        if (boost - 1.0).abs() < 0.001 {
            return pixel;
        }
        let luma = 0.299 * pixel[0] as f32 + 0.587 * pixel[1] as f32 + 0.114 * pixel[2] as f32;
        let r = (luma + (pixel[0] as f32 - luma) * boost).clamp(0.0, 255.0) as u8;
        let g = (luma + (pixel[1] as f32 - luma) * boost).clamp(0.0, 255.0) as u8;
        let b = (luma + (pixel[2] as f32 - luma) * boost).clamp(0.0, 255.0) as u8;
        [r, g, b, pixel[3]]
    }
}

impl VideoEffect for TiltShiftEffect {
    fn name(&self) -> &str {
        "TiltShift"
    }

    fn description(&self) -> &'static str {
        "Tilt-shift miniature effect with gradient blur mask"
    }

    fn apply(
        &mut self,
        input: &Frame,
        output: &mut Frame,
        _params: &EffectParams,
    ) -> VfxResult<()> {
        if input.width != output.width || input.height != output.height {
            return Err(VfxError::InvalidDimensions {
                width: output.width,
                height: output.height,
            });
        }
        if input.height == 0 {
            return Ok(());
        }

        let h = input.height as f32;

        for y in 0..input.height {
            let ny = y as f32 / h;
            let blur_radius = self.config.blur_radius_at(ny);
            let blur_amount = self.config.blur_amount_at(ny);

            for x in 0..input.width {
                let original = input.get_pixel(x, y).unwrap_or([0, 0, 0, 0]);

                if blur_radius == 0 {
                    // In focus band: apply saturation boost
                    let boosted = Self::boost_saturation(original, self.config.saturation_boost);
                    output.set_pixel(x, y, boosted);
                } else {
                    // Blur zone: blend between sharp and blurred
                    let blurred = Self::box_blur_pixel(input, x, y, blur_radius);
                    let t = blur_amount;
                    let inv_t = 1.0 - t;
                    let pixel = [
                        (original[0] as f32 * inv_t + blurred[0] as f32 * t) as u8,
                        (original[1] as f32 * inv_t + blurred[1] as f32 * t) as u8,
                        (original[2] as f32 * inv_t + blurred[2] as f32 * t) as u8,
                        original[3],
                    ];
                    output.set_pixel(x, y, pixel);
                }
            }
        }

        Ok(())
    }

    fn supports_gpu(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_frame(w: u32, h: u32) -> Frame {
        let mut f = Frame::new(w, h).expect("test frame");
        f.clear([128, 128, 128, 255]);
        f
    }

    #[test]
    fn test_tilt_shift_config_default() {
        let cfg = TiltShiftConfig::default();
        assert!((cfg.focus_position - 0.5).abs() < 0.01);
        assert!(cfg.focus_width > 0.0);
    }

    #[test]
    fn test_blur_amount_at_focus_center() {
        let cfg = TiltShiftConfig::default();
        assert_eq!(cfg.blur_amount_at(0.5), 0.0);
    }

    #[test]
    fn test_blur_amount_at_edges() {
        let cfg = TiltShiftConfig::default();
        let top = cfg.blur_amount_at(0.0);
        let bottom = cfg.blur_amount_at(1.0);
        assert!(top > 0.0, "top should be blurred");
        assert!(bottom > 0.0, "bottom should be blurred");
    }

    #[test]
    fn test_blur_radius_at_focus_center_zero() {
        let cfg = TiltShiftConfig::default();
        assert_eq!(cfg.blur_radius_at(0.5), 0);
    }

    #[test]
    fn test_blur_radius_at_extreme() {
        let cfg = TiltShiftConfig {
            focus_position: 0.5,
            focus_width: 0.1,
            transition_width: 0.1,
            max_blur_radius: 10,
            ..Default::default()
        };
        let radius = cfg.blur_radius_at(0.0);
        assert!(radius > 0, "edge should have blur");
    }

    #[test]
    fn test_tilt_shift_effect_basic() {
        let mut effect = TiltShiftEffect::default_miniature().with_max_blur(2);
        let input = test_frame(32, 32);
        let mut output = Frame::new(32, 32).expect("test frame");
        let params = EffectParams::new();
        effect.apply(&input, &mut output, &params).expect("apply");

        // Centre should be boosted / unchanged, edges blurred
        let center = output.get_pixel(16, 16).expect("center");
        assert!(center[3] == 255);
    }

    #[test]
    fn test_tilt_shift_dimension_mismatch() {
        let mut effect = TiltShiftEffect::default_miniature();
        let input = Frame::new(100, 100).expect("frame");
        let mut output = Frame::new(50, 50).expect("frame");
        let params = EffectParams::new();
        assert!(effect.apply(&input, &mut output, &params).is_err());
    }

    #[test]
    fn test_tilt_shift_name() {
        let effect = TiltShiftEffect::default_miniature();
        assert_eq!(effect.name(), "TiltShift");
        assert!(!effect.description().is_empty());
    }

    #[test]
    fn test_saturation_boost_identity() {
        let pixel = [128, 64, 32, 255];
        let result = TiltShiftEffect::boost_saturation(pixel, 1.0);
        assert_eq!(result, pixel);
    }

    #[test]
    fn test_saturation_boost_increase() {
        let pixel = [200, 100, 50, 255];
        let result = TiltShiftEffect::boost_saturation(pixel, 1.5);
        // More saturated: red should increase, green/blue should decrease relative to luma
        let luma = 0.299 * 200.0 + 0.587 * 100.0 + 0.114 * 50.0;
        assert!(result[0] as f32 > luma, "red should be above luma");
    }

    #[test]
    fn test_blur_amount_transition_smooth() {
        let cfg = TiltShiftConfig {
            focus_position: 0.5,
            focus_width: 0.2,
            transition_width: 0.2,
            ..Default::default()
        };
        // Just inside focus band
        assert_eq!(cfg.blur_amount_at(0.5), 0.0);
        // Edge of focus
        assert_eq!(cfg.blur_amount_at(0.4), 0.0);
        // In transition
        let mid_trans = cfg.blur_amount_at(0.3);
        assert!(mid_trans > 0.0 && mid_trans < 1.0);
    }

    #[test]
    fn test_tilt_shift_with_builders() {
        let effect = TiltShiftEffect::default_miniature()
            .with_focus_position(0.3)
            .with_focus_width(0.2)
            .with_max_blur(16);
        assert!((effect.config.focus_position - 0.3).abs() < 0.01);
        assert!((effect.config.focus_width - 0.2).abs() < 0.01);
        assert_eq!(effect.config.max_blur_radius, 16);
    }

    #[test]
    fn test_zero_transition_width() {
        let cfg = TiltShiftConfig {
            focus_position: 0.5,
            focus_width: 0.2,
            transition_width: 0.0,
            ..Default::default()
        };
        // Outside focus band with zero transition => instant full blur
        assert_eq!(cfg.blur_amount_at(0.0), 1.0);
        // Inside focus
        assert_eq!(cfg.blur_amount_at(0.5), 0.0);
    }
}
