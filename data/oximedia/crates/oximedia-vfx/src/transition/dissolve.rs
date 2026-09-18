//! Cross-dissolve transition with custom curves.

use crate::{EffectParams, Frame, TransitionEffect, VfxResult};

/// Cross-dissolve transition.
///
/// Smoothly blends from one frame to another using alpha blending.
/// Supports custom easing curves for non-linear dissolves.
pub struct Dissolve {
    /// Power curve for non-linear dissolve (1.0 = linear).
    power: f32,
    /// Apply dip to black.
    dip_to_black: bool,
}

impl Dissolve {
    /// Create a new dissolve transition.
    #[must_use]
    pub fn new() -> Self {
        Self {
            power: 1.0,
            dip_to_black: false,
        }
    }

    /// Set power curve (>1.0 = ease in, <1.0 = ease out).
    #[must_use]
    pub fn with_power(mut self, power: f32) -> Self {
        self.power = power.max(0.1);
        self
    }

    /// Enable dip to black in the middle.
    #[must_use]
    pub const fn with_dip_to_black(mut self, enable: bool) -> Self {
        self.dip_to_black = enable;
        self
    }

    fn blend_pixel(from: [u8; 4], to: [u8; 4], t: f32) -> [u8; 4] {
        let t = t.clamp(0.0, 1.0);
        let inv_t = 1.0 - t;

        [
            (f32::from(from[0]) * inv_t + f32::from(to[0]) * t) as u8,
            (f32::from(from[1]) * inv_t + f32::from(to[1]) * t) as u8,
            (f32::from(from[2]) * inv_t + f32::from(to[2]) * t) as u8,
            (f32::from(from[3]) * inv_t + f32::from(to[3]) * t) as u8,
        ]
    }
}

impl Default for Dissolve {
    fn default() -> Self {
        Self::new()
    }
}

impl TransitionEffect for Dissolve {
    fn name(&self) -> &'static str {
        "Dissolve"
    }

    fn description(&self) -> &'static str {
        "Cross-dissolve between two frames with custom curves"
    }

    fn apply(
        &mut self,
        from: &Frame,
        to: &Frame,
        output: &mut Frame,
        params: &EffectParams,
    ) -> VfxResult<()> {
        let mut progress = params.progress;

        // Apply power curve
        if (self.power - 1.0).abs() > 0.001 {
            progress = progress.powf(self.power);
        }

        // Apply dip to black
        let (from_opacity, to_opacity) = if self.dip_to_black {
            if progress < 0.5 {
                (1.0 - progress * 2.0, 0.0)
            } else {
                (0.0, (progress - 0.5) * 2.0)
            }
        } else {
            (1.0 - progress, progress)
        };

        for y in 0..output.height {
            for x in 0..output.width {
                let from_pixel = from.get_pixel(x, y).unwrap_or([0, 0, 0, 0]);
                let to_pixel = to.get_pixel(x, y).unwrap_or([0, 0, 0, 0]);

                let pixel = if self.dip_to_black {
                    // Blend through black
                    let from_dark = [
                        (f32::from(from_pixel[0]) * from_opacity) as u8,
                        (f32::from(from_pixel[1]) * from_opacity) as u8,
                        (f32::from(from_pixel[2]) * from_opacity) as u8,
                        from_pixel[3],
                    ];
                    let to_dark = [
                        (f32::from(to_pixel[0]) * to_opacity) as u8,
                        (f32::from(to_pixel[1]) * to_opacity) as u8,
                        (f32::from(to_pixel[2]) * to_opacity) as u8,
                        to_pixel[3],
                    ];
                    Self::blend_pixel(from_dark, to_dark, 0.5)
                } else {
                    Self::blend_pixel(from_pixel, to_pixel, progress)
                };

                output.set_pixel(x, y, pixel);
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

    #[test]
    fn test_dissolve_basic() {
        let mut dissolve = Dissolve::new();
        let from = Frame::new(100, 100).expect("should succeed in test");
        let to = Frame::new(100, 100).expect("should succeed in test");
        let mut output = Frame::new(100, 100).expect("should succeed in test");

        let params = EffectParams::new().with_progress(0.5);
        dissolve
            .apply(&from, &to, &mut output, &params)
            .expect("should succeed in test");
    }

    #[test]
    fn test_dissolve_power_curve() {
        let dissolve = Dissolve::new().with_power(2.0);
        assert_eq!(dissolve.power, 2.0);
    }

    #[test]
    fn test_dissolve_dip_to_black() {
        let dissolve = Dissolve::new().with_dip_to_black(true);
        assert!(dissolve.dip_to_black);
    }
}
