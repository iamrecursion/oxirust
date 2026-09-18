//! Cartoon/cel-shading effect.

use crate::{Color, EffectParams, Frame, VfxResult, VideoEffect};
use serde::{Deserialize, Serialize};

/// Cartoon style.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CartoonStyle {
    /// Basic cel-shading.
    CelShading,
    /// Comic book style.
    ComicBook,
    /// Posterize.
    Posterize,
}

/// Cartoon effect.
pub struct Cartoon {
    style: CartoonStyle,
    levels: u8,
    edge_threshold: f32,
    edge_color: Color,
}

impl Cartoon {
    /// Create a new cartoon effect.
    #[must_use]
    pub const fn new(style: CartoonStyle) -> Self {
        Self {
            style,
            levels: 4,
            edge_threshold: 0.3,
            edge_color: Color::black(),
        }
    }

    /// Set number of color levels.
    #[must_use]
    pub fn with_levels(mut self, levels: u8) -> Self {
        self.levels = levels.max(2);
        self
    }

    /// Set edge detection threshold.
    #[must_use]
    pub fn with_edge_threshold(mut self, threshold: f32) -> Self {
        self.edge_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    fn quantize(&self, value: u8) -> u8 {
        let step = 255 / self.levels;
        (value / step) * step
    }

    fn detect_edge(&self, input: &Frame, x: u32, y: u32) -> bool {
        let pixel = input.get_pixel(x, y).unwrap_or([0, 0, 0, 0]);
        let neighbors = [
            input.get_pixel(x.saturating_sub(1), y),
            input.get_pixel(x.saturating_add(1).min(input.width - 1), y),
            input.get_pixel(x, y.saturating_sub(1)),
            input.get_pixel(x, y.saturating_add(1).min(input.height - 1)),
        ];

        for neighbor in neighbors.iter().flatten() {
            let diff = ((i32::from(pixel[0]) - i32::from(neighbor[0])).abs()
                + (i32::from(pixel[1]) - i32::from(neighbor[1])).abs()
                + (i32::from(pixel[2]) - i32::from(neighbor[2])).abs())
                as f32
                / (255.0 * 3.0);

            if diff > self.edge_threshold {
                return true;
            }
        }

        false
    }
}

impl VideoEffect for Cartoon {
    fn name(&self) -> &'static str {
        "Cartoon"
    }

    fn description(&self) -> &'static str {
        "Cartoon/cel-shading effect"
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

                let is_edge = self.detect_edge(input, x, y);

                let result = if is_edge {
                    self.edge_color.to_rgba()
                } else {
                    [
                        self.quantize(pixel[0]),
                        self.quantize(pixel[1]),
                        self.quantize(pixel[2]),
                        pixel[3],
                    ]
                };

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
    fn test_cartoon() {
        let mut cartoon = Cartoon::new(CartoonStyle::CelShading).with_levels(4);
        let input = Frame::new(100, 100).expect("should succeed in test");
        let mut output = Frame::new(100, 100).expect("should succeed in test");
        let params = EffectParams::new();
        cartoon
            .apply(&input, &mut output, &params)
            .expect("should succeed in test");
    }
}
