//! Halftone/comic book effect.

use crate::{EffectParams, Frame, VfxResult, VideoEffect};
use serde::{Deserialize, Serialize};

/// Halftone pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HalftonePattern {
    /// Circular dots.
    Dots,
    /// Lines.
    Lines,
    /// Crosshatch.
    Crosshatch,
}

/// Halftone effect.
pub struct Halftone {
    pattern: HalftonePattern,
    dot_size: u32,
    angle: f32,
}

impl Halftone {
    /// Create a new halftone effect.
    #[must_use]
    pub const fn new(pattern: HalftonePattern) -> Self {
        Self {
            pattern,
            dot_size: 4,
            angle: 45.0,
        }
    }

    /// Set dot/line size.
    #[must_use]
    pub fn with_dot_size(mut self, size: u32) -> Self {
        self.dot_size = size.max(1);
        self
    }

    /// Set pattern angle in degrees.
    #[must_use]
    pub const fn with_angle(mut self, angle: f32) -> Self {
        self.angle = angle;
        self
    }
}

impl VideoEffect for Halftone {
    fn name(&self) -> &'static str {
        "Halftone"
    }

    fn description(&self) -> &'static str {
        "Halftone/comic book effect"
    }

    fn apply(
        &mut self,
        input: &Frame,
        output: &mut Frame,
        _params: &EffectParams,
    ) -> VfxResult<()> {
        let angle_rad = self.angle.to_radians();

        for y in 0..output.height {
            for x in 0..output.width {
                let pixel = input.get_pixel(x, y).unwrap_or([0, 0, 0, 0]);
                let gray = (u32::from(pixel[0]) + u32::from(pixel[1]) + u32::from(pixel[2])) / 3;

                // Rotate coordinates
                let fx = x as f32;
                let fy = y as f32;
                let rx = fx * angle_rad.cos() - fy * angle_rad.sin();
                let ry = fx * angle_rad.sin() + fy * angle_rad.cos();

                let grid_x = (rx / self.dot_size as f32).floor() as i32;
                let grid_y = (ry / self.dot_size as f32).floor() as i32;

                let in_grid_x = rx - (grid_x as f32 * self.dot_size as f32);
                let in_grid_y = ry - (grid_y as f32 * self.dot_size as f32);

                let threshold = gray as f32 / 255.0;

                let visible = match self.pattern {
                    HalftonePattern::Dots => {
                        let cx = self.dot_size as f32 / 2.0;
                        let cy = self.dot_size as f32 / 2.0;
                        let dx = in_grid_x - cx;
                        let dy = in_grid_y - cy;
                        let dist = (dx * dx + dy * dy).sqrt();
                        let max_radius = cx * threshold;
                        dist < max_radius
                    }
                    HalftonePattern::Lines => in_grid_y < self.dot_size as f32 * threshold,
                    HalftonePattern::Crosshatch => {
                        in_grid_x < self.dot_size as f32 * threshold
                            || in_grid_y < self.dot_size as f32 * threshold
                    }
                };

                let result = if visible { 0 } else { 255 };
                output.set_pixel(x, y, [result, result, result, 255]);
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
    fn test_halftone() {
        let mut halftone = Halftone::new(HalftonePattern::Dots)
            .with_dot_size(4)
            .with_angle(45.0);

        let input = Frame::new(100, 100).expect("should succeed in test");
        let mut output = Frame::new(100, 100).expect("should succeed in test");
        let params = EffectParams::new();
        halftone
            .apply(&input, &mut output, &params)
            .expect("should succeed in test");
    }
}
