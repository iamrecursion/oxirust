//! Light rays/god rays effect.

use crate::{Color, EffectParams, Frame, VfxResult, VideoEffect};
use serde::{Deserialize, Serialize};

/// Ray pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RayPattern {
    /// Radial rays from center.
    Radial,
    /// Parallel rays.
    Parallel,
    /// Volumetric god rays.
    Volumetric,
}

/// Light rays effect.
pub struct LightRays {
    pattern: RayPattern,
    source_x: f32,
    source_y: f32,
    intensity: f32,
    color: Color,
    num_rays: u32,
}

impl LightRays {
    /// Create a new light rays effect.
    #[must_use]
    pub const fn new(pattern: RayPattern) -> Self {
        Self {
            pattern,
            source_x: 0.5,
            source_y: 0.5,
            intensity: 0.5,
            color: Color::white(),
            num_rays: 12,
        }
    }

    /// Set light source position (0.0 - 1.0).
    #[must_use]
    pub fn with_source(mut self, x: f32, y: f32) -> Self {
        self.source_x = x.clamp(0.0, 1.0);
        self.source_y = y.clamp(0.0, 1.0);
        self
    }

    /// Set ray intensity.
    #[must_use]
    pub fn with_intensity(mut self, intensity: f32) -> Self {
        self.intensity = intensity.clamp(0.0, 1.0);
        self
    }

    /// Set number of rays.
    #[must_use]
    pub fn with_num_rays(mut self, num: u32) -> Self {
        self.num_rays = num.max(1);
        self
    }
}

impl VideoEffect for LightRays {
    fn name(&self) -> &'static str {
        "Light Rays"
    }

    fn description(&self) -> &'static str {
        "God rays and volumetric light"
    }

    fn apply(&mut self, input: &Frame, output: &mut Frame, params: &EffectParams) -> VfxResult<()> {
        let source_x = self.source_x * output.width as f32;
        let source_y = self.source_y * output.height as f32;

        for y in 0..output.height {
            for x in 0..output.width {
                let pixel = input.get_pixel(x, y).unwrap_or([0, 0, 0, 0]);

                let dx = x as f32 - source_x;
                let dy = y as f32 - source_y;
                let angle = dy.atan2(dx);
                let dist = (dx * dx + dy * dy).sqrt();

                let ray_value = match self.pattern {
                    RayPattern::Radial => {
                        let ray_angle = angle * self.num_rays as f32 / std::f32::consts::TAU;
                        let in_ray = (ray_angle.fract() - 0.5).abs() < 0.1;
                        if in_ray {
                            (1.0 - dist / (output.width.max(output.height) as f32)) * self.intensity
                        } else {
                            0.0
                        }
                    }
                    RayPattern::Parallel => {
                        let parallel_pos = x as f32 / output.width as f32;
                        let _ray_num = (parallel_pos * self.num_rays as f32).floor();
                        let in_ray = (parallel_pos * self.num_rays as f32).fract() < 0.2;
                        if in_ray {
                            self.intensity
                        } else {
                            0.0
                        }
                    }
                    RayPattern::Volumetric => {
                        let ray_angle = angle * self.num_rays as f32 / std::f32::consts::TAU;
                        let in_ray = (ray_angle.fract() - 0.5).abs() < 0.15;
                        if in_ray {
                            let falloff = (-(dist / 100.0).powi(2)).exp();
                            falloff * self.intensity * (params.time as f32 * 0.5).sin().abs()
                        } else {
                            0.0
                        }
                    }
                };

                let ray_r = (f32::from(self.color.r) * ray_value) as u8;
                let ray_g = (f32::from(self.color.g) * ray_value) as u8;
                let ray_b = (f32::from(self.color.b) * ray_value) as u8;

                let result = [
                    pixel[0].saturating_add(ray_r),
                    pixel[1].saturating_add(ray_g),
                    pixel[2].saturating_add(ray_b),
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
    fn test_light_rays() {
        let mut rays = LightRays::new(RayPattern::Radial)
            .with_source(0.5, 0.5)
            .with_num_rays(12);

        let input = Frame::new(100, 100).expect("should succeed in test");
        let mut output = Frame::new(100, 100).expect("should succeed in test");
        let params = EffectParams::new();
        rays.apply(&input, &mut output, &params)
            .expect("should succeed in test");
    }
}
