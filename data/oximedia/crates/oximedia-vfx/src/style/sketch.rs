//! Sketch/pencil effect.

use crate::{EffectParams, Frame, VfxResult, VideoEffect};
use serde::{Deserialize, Serialize};

/// Sketch style.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SketchStyle {
    /// Pencil sketch.
    Pencil,
    /// Pen and ink.
    PenInk,
    /// Charcoal.
    Charcoal,
}

/// Sketch effect.
pub struct Sketch {
    style: SketchStyle,
    intensity: f32,
}

impl Sketch {
    /// Create a new sketch effect.
    #[must_use]
    pub const fn new(style: SketchStyle) -> Self {
        Self {
            style,
            intensity: 1.0,
        }
    }

    /// Set sketch intensity.
    #[must_use]
    pub fn with_intensity(mut self, intensity: f32) -> Self {
        self.intensity = intensity.clamp(0.0, 1.0);
        self
    }

    fn sobel_edge(&self, input: &Frame, x: u32, y: u32) -> u8 {
        let gx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]];

        let gy = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]];

        let mut sum_x = 0_i32;
        let mut sum_y = 0_i32;

        for dy in -1..=1 {
            for dx in -1..=1 {
                let nx = (x as i32 + dx).max(0).min(input.width as i32 - 1) as u32;
                let ny = (y as i32 + dy).max(0).min(input.height as i32 - 1) as u32;

                if let Some(pixel) = input.get_pixel(nx, ny) {
                    let gray =
                        (u32::from(pixel[0]) + u32::from(pixel[1]) + u32::from(pixel[2])) / 3;
                    let gx_val = gx[(dy + 1) as usize][(dx + 1) as usize];
                    let gy_val = gy[(dy + 1) as usize][(dx + 1) as usize];

                    sum_x += gx_val * gray as i32;
                    sum_y += gy_val * gray as i32;
                }
            }
        }

        let magnitude = ((sum_x * sum_x + sum_y * sum_y) as f32).sqrt();
        (255.0 - (magnitude * self.intensity).min(255.0)) as u8
    }
}

impl VideoEffect for Sketch {
    fn name(&self) -> &'static str {
        "Sketch"
    }

    fn description(&self) -> &'static str {
        "Sketch/pencil effect"
    }

    fn apply(
        &mut self,
        input: &Frame,
        output: &mut Frame,
        _params: &EffectParams,
    ) -> VfxResult<()> {
        for y in 0..output.height {
            for x in 0..output.width {
                let edge = self.sobel_edge(input, x, y);
                output.set_pixel(x, y, [edge, edge, edge, 255]);
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
    fn test_sketch() {
        let mut sketch = Sketch::new(SketchStyle::Pencil).with_intensity(0.8);
        let input = Frame::new(100, 100).expect("should succeed in test");
        let mut output = Frame::new(100, 100).expect("should succeed in test");
        let params = EffectParams::new();
        sketch
            .apply(&input, &mut output, &params)
            .expect("should succeed in test");
    }
}
