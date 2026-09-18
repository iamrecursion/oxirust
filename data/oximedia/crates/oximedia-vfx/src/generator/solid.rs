//! Solid color generator.

use crate::{Color, EffectParams, Frame, VfxResult, VideoEffect};

/// Solid color generator.
pub struct Solid {
    color: Color,
}

impl Solid {
    /// Create a new solid color generator.
    #[must_use]
    pub const fn new(color: Color) -> Self {
        Self { color }
    }

    /// Set color.
    #[must_use]
    pub const fn with_color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }
}

impl Default for Solid {
    fn default() -> Self {
        Self::new(Color::black())
    }
}

impl VideoEffect for Solid {
    fn name(&self) -> &'static str {
        "Solid Color"
    }

    fn description(&self) -> &'static str {
        "Generate solid color frames"
    }

    fn apply(
        &mut self,
        _input: &Frame,
        output: &mut Frame,
        _params: &EffectParams,
    ) -> VfxResult<()> {
        output.clear(self.color.to_rgba());
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
    fn test_solid_color() {
        let mut solid = Solid::new(Color::rgb(255, 0, 0));
        let input = Frame::new(100, 100).expect("should succeed in test");
        let mut output = Frame::new(100, 100).expect("should succeed in test");
        let params = EffectParams::new();
        solid
            .apply(&input, &mut output, &params)
            .expect("should succeed in test");

        let pixel = output.get_pixel(50, 50).expect("should succeed in test");
        assert_eq!(pixel, [255, 0, 0, 255]);
    }
}
