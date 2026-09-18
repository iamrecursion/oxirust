//! Spill suppression algorithms.

use crate::{EffectParams, Frame, VfxResult, VideoEffect};
use serde::{Deserialize, Serialize};

/// Spill suppression method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpillMethod {
    /// Simple color suppression.
    Simple,
    /// Limit color method.
    Limit,
    /// Advanced color correction.
    Advanced,
}

/// Spill suppression effect.
pub struct SpillSuppress {
    method: SpillMethod,
    strength: f32,
    target_channel: usize, // 0=R, 1=G, 2=B
}

impl SpillSuppress {
    /// Create a new spill suppress effect.
    #[must_use]
    pub const fn new(method: SpillMethod, target_channel: usize) -> Self {
        Self {
            method,
            strength: 1.0,
            target_channel,
        }
    }

    /// Set suppression strength (0.0 - 1.0).
    #[must_use]
    pub fn with_strength(mut self, strength: f32) -> Self {
        self.strength = strength.clamp(0.0, 1.0);
        self
    }
}

impl VideoEffect for SpillSuppress {
    fn name(&self) -> &'static str {
        "Spill Suppress"
    }

    fn description(&self) -> &'static str {
        "Remove color spill from keyed edges"
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
                let r = f32::from(pixel[0]);
                let g = f32::from(pixel[1]);
                let b = f32::from(pixel[2]);

                let new_pixel = match self.method {
                    SpillMethod::Simple => {
                        let values = [r, g, b];
                        let spill = values[self.target_channel];
                        let other_max = values
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| *i != self.target_channel)
                            .map(|(_, v)| v)
                            .fold(0.0_f32, |a, b| a.max(*b));

                        let excess = (spill - other_max).max(0.0);
                        let suppressed = spill - excess * self.strength;

                        let mut result = pixel;
                        result[self.target_channel] = suppressed as u8;
                        result
                    }
                    SpillMethod::Limit => {
                        let values = [r, g, b];
                        let other_avg = values
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| *i != self.target_channel)
                            .map(|(_, v)| v)
                            .sum::<f32>()
                            / 2.0;

                        let mut result = pixel;
                        result[self.target_channel] = (values[self.target_channel]
                            .min(other_avg * (1.0 + self.strength)))
                            as u8;
                        result
                    }
                    SpillMethod::Advanced => {
                        // More sophisticated color correction
                        let values = [r, g, b];
                        let target = values[self.target_channel];
                        let other_max = values
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| *i != self.target_channel)
                            .map(|(_, v)| v)
                            .fold(0.0_f32, |a, b| a.max(*b));

                        let ratio = if target > 0.0 {
                            other_max / target
                        } else {
                            1.0
                        };

                        let corrected = if ratio < 1.0 {
                            target * (ratio + (1.0 - ratio) * (1.0 - self.strength))
                        } else {
                            target
                        };

                        let mut result = pixel;
                        result[self.target_channel] = corrected as u8;
                        result
                    }
                };

                output.set_pixel(x, y, new_pixel);
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
    fn test_spill_suppress() {
        let mut suppress = SpillSuppress::new(SpillMethod::Simple, 1); // Green channel
        let mut input = Frame::new(100, 100).expect("should succeed in test");
        let mut output = Frame::new(100, 100).expect("should succeed in test");

        // Fill with greenish spill
        for y in 0..100 {
            for x in 0..100 {
                input.set_pixel(x, y, [100, 150, 100, 255]);
            }
        }

        let params = EffectParams::new();
        suppress
            .apply(&input, &mut output, &params)
            .expect("should succeed in test");
    }
}
