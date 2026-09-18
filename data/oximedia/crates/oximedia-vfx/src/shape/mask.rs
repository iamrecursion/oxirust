//! Shape-based masking.

use crate::{EffectParams, Frame, VfxResult, VideoEffect};
use serde::{Deserialize, Serialize};

use super::draw::Shape;

/// Mask mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MaskMode {
    /// Show inside mask.
    ShowInside,
    /// Show outside mask.
    ShowOutside,
    /// Feathered edge.
    Feathered,
}

/// Shape mask effect.
pub struct ShapeMask {
    shape: Shape,
    mode: MaskMode,
    feather: f32,
}

impl ShapeMask {
    /// Create a new shape mask.
    #[must_use]
    pub const fn new(shape: Shape, mode: MaskMode) -> Self {
        Self {
            shape,
            mode,
            feather: 0.0,
        }
    }

    /// Set feather amount.
    #[must_use]
    pub fn with_feather(mut self, feather: f32) -> Self {
        self.feather = feather.max(0.0);
        self
    }

    fn point_in_shape(&self, x: f32, y: f32) -> bool {
        let cx = self.shape.bounds.x + self.shape.bounds.width / 2.0;
        let cy = self.shape.bounds.y + self.shape.bounds.height / 2.0;
        let radius = self.shape.bounds.width / 2.0;

        let dx = x - cx;
        let dy = y - cy;
        let dist = (dx * dx + dy * dy).sqrt();

        dist <= radius
    }
}

impl VideoEffect for ShapeMask {
    fn name(&self) -> &'static str {
        "Shape Mask"
    }

    fn description(&self) -> &'static str {
        "Mask video with shapes"
    }

    fn apply(
        &mut self,
        input: &Frame,
        output: &mut Frame,
        _params: &EffectParams,
    ) -> VfxResult<()> {
        for y in 0..output.height {
            for x in 0..output.width {
                let pixel = input.get_pixel(x, y).unwrap_or([0, 0, 0, 0]);
                let inside = self.point_in_shape(x as f32, y as f32);

                let show = match self.mode {
                    MaskMode::ShowInside => inside,
                    MaskMode::ShowOutside => !inside,
                    MaskMode::Feathered => inside, // Simplified
                };

                if show {
                    output.set_pixel(x, y, pixel);
                } else {
                    output.set_pixel(x, y, [0, 0, 0, 0]);
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

    #[test]
    fn test_shape_mask() {
        let shape = Shape::circle(50.0, 50.0, 30.0);
        let mut mask = ShapeMask::new(shape, MaskMode::ShowInside);

        let input = Frame::new(100, 100).expect("should succeed in test");
        let mut output = Frame::new(100, 100).expect("should succeed in test");
        let params = EffectParams::new();
        mask.apply(&input, &mut output, &params)
            .expect("should succeed in test");
    }
}
