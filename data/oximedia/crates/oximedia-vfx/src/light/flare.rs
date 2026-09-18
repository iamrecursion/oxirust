//! Lens flare effect.

use crate::{Color, EffectParams, Frame, VfxResult, VideoEffect};
use serde::{Deserialize, Serialize};

/// Lens flare type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FlareType {
    /// Anamorphic flare.
    Anamorphic,
    /// Spherical flare.
    Spherical,
    /// Star burst.
    StarBurst,
}

/// Lens flare effect.
pub struct LensFlare {
    flare_type: FlareType,
    position_x: f32,
    position_y: f32,
    intensity: f32,
    color: Color,
}

impl LensFlare {
    /// Create a new lens flare effect.
    #[must_use]
    pub const fn new(flare_type: FlareType) -> Self {
        Self {
            flare_type,
            position_x: 0.5,
            position_y: 0.5,
            intensity: 1.0,
            color: Color::white(),
        }
    }

    /// Set flare position (0.0 - 1.0).
    #[must_use]
    pub fn with_position(mut self, x: f32, y: f32) -> Self {
        self.position_x = x.clamp(0.0, 1.0);
        self.position_y = y.clamp(0.0, 1.0);
        self
    }

    /// Set flare intensity.
    #[must_use]
    pub fn with_intensity(mut self, intensity: f32) -> Self {
        self.intensity = intensity.max(0.0);
        self
    }

    /// Set flare color.
    #[must_use]
    pub const fn with_color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }
}

impl VideoEffect for LensFlare {
    fn name(&self) -> &'static str {
        "Lens Flare"
    }

    fn description(&self) -> &'static str {
        "Realistic lens flare effect"
    }

    fn apply(
        &mut self,
        input: &Frame,
        output: &mut Frame,
        _params: &EffectParams,
    ) -> VfxResult<()> {
        let center_x = self.position_x * output.width as f32;
        let center_y = self.position_y * output.height as f32;

        for y in 0..output.height {
            for x in 0..output.width {
                let pixel = input.get_pixel(x, y).unwrap_or([0, 0, 0, 0]);
                let dx = x as f32 - center_x;
                let dy = y as f32 - center_y;
                let dist = (dx * dx + dy * dy).sqrt();

                let flare_intensity = match self.flare_type {
                    FlareType::Anamorphic => {
                        let horizontal = (-dx * dx / 10000.0).exp();
                        let vertical = (-dy * dy / 100.0).exp();
                        horizontal * vertical * self.intensity
                    }
                    FlareType::Spherical => {
                        let falloff = (-dist * dist / 10000.0).exp();
                        falloff * self.intensity
                    }
                    FlareType::StarBurst => {
                        let angle = dy.atan2(dx);
                        let rays = 6.0;
                        let ray_pattern = ((angle * rays).cos().abs() * 0.5 + 0.5).powf(10.0);
                        let falloff = (-dist * dist / 10000.0).exp();
                        ray_pattern * falloff * self.intensity
                    }
                };

                let flare_r = (f32::from(self.color.r) * flare_intensity) as u8;
                let flare_g = (f32::from(self.color.g) * flare_intensity) as u8;
                let flare_b = (f32::from(self.color.b) * flare_intensity) as u8;

                let result = [
                    pixel[0].saturating_add(flare_r),
                    pixel[1].saturating_add(flare_g),
                    pixel[2].saturating_add(flare_b),
                    pixel[3],
                ];

                output.set_pixel(x, y, result);
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
    fn test_lens_flare() {
        let mut flare = LensFlare::new(FlareType::Spherical)
            .with_position(0.5, 0.5)
            .with_intensity(1.0);

        let input = Frame::new(100, 100).expect("should succeed in test");
        let mut output = Frame::new(100, 100).expect("should succeed in test");
        let params = EffectParams::new();
        flare
            .apply(&input, &mut output, &params)
            .expect("should succeed in test");
    }
}
