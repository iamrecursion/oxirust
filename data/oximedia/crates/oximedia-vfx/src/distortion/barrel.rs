//! Barrel and pincushion distortion.

use crate::{EffectParams, Frame, VfxResult, VideoEffect};
use serde::{Deserialize, Serialize};

/// Distortion type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistortionType {
    /// Barrel distortion (bulge outward).
    Barrel,
    /// Pincushion distortion (pinch inward).
    Pincushion,
}

/// Barrel/pincushion distortion effect.
pub struct BarrelDistortion {
    distortion_type: DistortionType,
    strength: f32,
}

impl BarrelDistortion {
    /// Create a new barrel distortion effect.
    #[must_use]
    pub const fn new(distortion_type: DistortionType) -> Self {
        Self {
            distortion_type,
            strength: 0.5,
        }
    }

    /// Set distortion strength (-1.0 to 1.0).
    #[must_use]
    pub fn with_strength(mut self, strength: f32) -> Self {
        self.strength = strength.clamp(-1.0, 1.0);
        self
    }
}

impl VideoEffect for BarrelDistortion {
    fn name(&self) -> &'static str {
        "Barrel Distortion"
    }

    fn description(&self) -> &'static str {
        "Barrel or pincushion distortion"
    }

    fn apply(
        &mut self,
        input: &Frame,
        output: &mut Frame,
        _params: &EffectParams,
    ) -> VfxResult<()> {
        let cx = output.width as f32 / 2.0;
        let cy = output.height as f32 / 2.0;
        let max_radius = (cx * cx + cy * cy).sqrt();

        let k = match self.distortion_type {
            DistortionType::Barrel => self.strength,
            DistortionType::Pincushion => -self.strength,
        };

        for y in 0..output.height {
            for x in 0..output.width {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let r = (dx * dx + dy * dy).sqrt() / max_radius;

                let factor = 1.0 + k * r * r;
                let src_x = cx + dx * factor;
                let src_y = cy + dy * factor;

                let pixel = if src_x >= 0.0
                    && src_x < input.width as f32
                    && src_y >= 0.0
                    && src_y < input.height as f32
                {
                    input
                        .get_pixel(src_x as u32, src_y as u32)
                        .unwrap_or([0, 0, 0, 0])
                } else {
                    [0, 0, 0, 0]
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
    fn test_barrel_distortion() {
        let mut barrel = BarrelDistortion::new(DistortionType::Barrel).with_strength(0.3);
        let input = Frame::new(100, 100).expect("should succeed in test");
        let mut output = Frame::new(100, 100).expect("should succeed in test");
        let params = EffectParams::new();
        barrel
            .apply(&input, &mut output, &params)
            .expect("should succeed in test");
    }
}
