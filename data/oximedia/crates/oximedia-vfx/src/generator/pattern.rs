//! Test pattern generator.

use crate::{Color, EffectParams, Frame, VfxResult, VideoEffect};
use serde::{Deserialize, Serialize};

/// Test pattern type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PatternType {
    /// Checkerboard pattern.
    Checkerboard,
    /// Grid pattern.
    Grid,
    /// Zone plate (frequency test).
    ZonePlate,
    /// Crosshatch pattern.
    Crosshatch,
    /// Ramp horizontal.
    RampHorizontal,
    /// Ramp vertical.
    RampVertical,
    /// Circle pattern.
    Circles,
}

/// Test pattern generator.
pub struct Pattern {
    pattern_type: PatternType,
    size: u32,
    color1: Color,
    color2: Color,
}

impl Pattern {
    /// Create a new pattern generator.
    #[must_use]
    pub const fn new(pattern_type: PatternType) -> Self {
        Self {
            pattern_type,
            size: 32,
            color1: Color::white(),
            color2: Color::black(),
        }
    }

    /// Set pattern size/scale.
    #[must_use]
    pub fn with_size(mut self, size: u32) -> Self {
        self.size = size.max(1);
        self
    }

    /// Set primary color.
    #[must_use]
    pub const fn with_color1(mut self, color: Color) -> Self {
        self.color1 = color;
        self
    }

    /// Set secondary color.
    #[must_use]
    pub const fn with_color2(mut self, color: Color) -> Self {
        self.color2 = color;
        self
    }
}

impl VideoEffect for Pattern {
    fn name(&self) -> &'static str {
        "Test Pattern"
    }

    fn description(&self) -> &'static str {
        "Generate various test patterns"
    }

    fn apply(
        &mut self,
        _input: &Frame,
        output: &mut Frame,
        _params: &EffectParams,
    ) -> VfxResult<()> {
        for y in 0..output.height {
            for x in 0..output.width {
                let color = match self.pattern_type {
                    PatternType::Checkerboard => {
                        let gx = x / self.size;
                        let gy = y / self.size;
                        if (gx + gy) % 2 == 0 {
                            self.color1
                        } else {
                            self.color2
                        }
                    }
                    PatternType::Grid => {
                        if x % self.size == 0 || y % self.size == 0 {
                            self.color1
                        } else {
                            self.color2
                        }
                    }
                    PatternType::ZonePlate => {
                        let cx = output.width as f32 / 2.0;
                        let cy = output.height as f32 / 2.0;
                        let dx = x as f32 - cx;
                        let dy = y as f32 - cy;
                        let dist = (dx * dx + dy * dy).sqrt();
                        let freq = self.size as f32;
                        let value = ((dist * freq / 100.0).sin() * 0.5 + 0.5) * 255.0;
                        Color::rgb(value as u8, value as u8, value as u8)
                    }
                    PatternType::Crosshatch => {
                        let line_width = self.size / 4;
                        if (x % self.size < line_width) || (y % self.size < line_width) {
                            self.color1
                        } else {
                            self.color2
                        }
                    }
                    PatternType::RampHorizontal => {
                        let value = (x as f32 / output.width as f32 * 255.0) as u8;
                        Color::rgb(value, value, value)
                    }
                    PatternType::RampVertical => {
                        let value = (y as f32 / output.height as f32 * 255.0) as u8;
                        Color::rgb(value, value, value)
                    }
                    PatternType::Circles => {
                        let cx = output.width as f32 / 2.0;
                        let cy = output.height as f32 / 2.0;
                        let dx = x as f32 - cx;
                        let dy = y as f32 - cy;
                        let dist = (dx * dx + dy * dy).sqrt();
                        if (dist as u32 / self.size) % 2 == 0 {
                            self.color1
                        } else {
                            self.color2
                        }
                    }
                };

                output.set_pixel(x, y, color.to_rgba());
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
    fn test_pattern_types() {
        let patterns = [
            PatternType::Checkerboard,
            PatternType::Grid,
            PatternType::ZonePlate,
        ];

        for pattern_type in patterns {
            let mut pattern = Pattern::new(pattern_type);
            let input = Frame::new(100, 100).expect("should succeed in test");
            let mut output = Frame::new(100, 100).expect("should succeed in test");
            let params = EffectParams::new();
            pattern
                .apply(&input, &mut output, &params)
                .expect("should succeed in test");
        }
    }

    #[test]
    fn test_pattern_customization() {
        let pattern = Pattern::new(PatternType::Checkerboard)
            .with_size(16)
            .with_color1(Color::rgb(255, 0, 0))
            .with_color2(Color::rgb(0, 0, 255));

        assert_eq!(pattern.size, 16);
    }
}
