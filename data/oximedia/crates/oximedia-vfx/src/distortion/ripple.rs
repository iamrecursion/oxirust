//! Ripple effect.

use crate::{EffectParams, Frame, VfxResult, VideoEffect};
use serde::{Deserialize, Serialize};

/// Ripple pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RipplePattern {
    /// Circular ripples from center.
    Circular,
    /// Radial ripples.
    Radial,
    /// Pond ripples (multiple centers).
    Pond,
}

/// Ripple effect.
pub struct Ripple {
    pattern: RipplePattern,
    amplitude: f32,
    frequency: f32,
    phase: f32,
    center_x: f32,
    center_y: f32,
}

impl Ripple {
    /// Create a new ripple effect.
    #[must_use]
    pub const fn new(pattern: RipplePattern) -> Self {
        Self {
            pattern,
            amplitude: 10.0,
            frequency: 0.05,
            phase: 0.0,
            center_x: 0.5,
            center_y: 0.5,
        }
    }

    /// Set ripple amplitude.
    #[must_use]
    pub const fn with_amplitude(mut self, amplitude: f32) -> Self {
        self.amplitude = amplitude;
        self
    }

    /// Set ripple frequency.
    #[must_use]
    pub const fn with_frequency(mut self, frequency: f32) -> Self {
        self.frequency = frequency;
        self
    }

    /// Set center point (0.0 - 1.0).
    #[must_use]
    pub fn with_center(mut self, x: f32, y: f32) -> Self {
        self.center_x = x.clamp(0.0, 1.0);
        self.center_y = y.clamp(0.0, 1.0);
        self
    }
}

impl VideoEffect for Ripple {
    fn name(&self) -> &'static str {
        "Ripple"
    }

    fn description(&self) -> &'static str {
        "Water ripple effect"
    }

    fn apply(&mut self, input: &Frame, output: &mut Frame, params: &EffectParams) -> VfxResult<()> {
        let cx = self.center_x * output.width as f32;
        let cy = self.center_y * output.height as f32;
        let animated_phase = self.phase + params.time as f32;

        for y in 0..output.height {
            for x in 0..output.width {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let dist = (dx * dx + dy * dy).sqrt();

                let ripple = (dist * self.frequency - animated_phase).sin() * self.amplitude;
                let angle = dy.atan2(dx);

                let offset_x = angle.cos() * ripple;
                let offset_y = angle.sin() * ripple;

                let src_x = x as f32 + offset_x;
                let src_y = y as f32 + offset_y;

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
    fn test_ripple() {
        let mut ripple = Ripple::new(RipplePattern::Circular)
            .with_amplitude(5.0)
            .with_frequency(0.1);

        let input = Frame::new(100, 100).expect("should succeed in test");
        let mut output = Frame::new(100, 100).expect("should succeed in test");
        let params = EffectParams::new();
        ripple
            .apply(&input, &mut output, &params)
            .expect("should succeed in test");
    }
}
