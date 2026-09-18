//! Wave distortion effect.

use crate::{EffectParams, Frame, VfxResult, VideoEffect};
use serde::{Deserialize, Serialize};

/// Wave direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WaveDirection {
    /// Horizontal waves.
    Horizontal,
    /// Vertical waves.
    Vertical,
    /// Both directions.
    Both,
}

/// Wave distortion effect.
pub struct Wave {
    direction: WaveDirection,
    amplitude: f32,
    frequency: f32,
    phase: f32,
}

impl Wave {
    /// Create a new wave effect.
    #[must_use]
    pub const fn new(direction: WaveDirection) -> Self {
        Self {
            direction,
            amplitude: 10.0,
            frequency: 0.1,
            phase: 0.0,
        }
    }

    /// Set wave amplitude.
    #[must_use]
    pub const fn with_amplitude(mut self, amplitude: f32) -> Self {
        self.amplitude = amplitude;
        self
    }

    /// Set wave frequency.
    #[must_use]
    pub const fn with_frequency(mut self, frequency: f32) -> Self {
        self.frequency = frequency;
        self
    }

    /// Set wave phase.
    #[must_use]
    pub const fn with_phase(mut self, phase: f32) -> Self {
        self.phase = phase;
        self
    }
}

impl VideoEffect for Wave {
    fn name(&self) -> &'static str {
        "Wave"
    }

    fn description(&self) -> &'static str {
        "Wave distortion effect"
    }

    fn apply(&mut self, input: &Frame, output: &mut Frame, params: &EffectParams) -> VfxResult<()> {
        let animated_phase = self.phase + params.time as f32;

        for y in 0..output.height {
            for x in 0..output.width {
                let (offset_x, offset_y) = match self.direction {
                    WaveDirection::Horizontal => {
                        let wave =
                            (y as f32 * self.frequency + animated_phase).sin() * self.amplitude;
                        (wave, 0.0)
                    }
                    WaveDirection::Vertical => {
                        let wave =
                            (x as f32 * self.frequency + animated_phase).sin() * self.amplitude;
                        (0.0, wave)
                    }
                    WaveDirection::Both => {
                        let wave_x =
                            (y as f32 * self.frequency + animated_phase).sin() * self.amplitude;
                        let wave_y =
                            (x as f32 * self.frequency + animated_phase).sin() * self.amplitude;
                        (wave_x, wave_y)
                    }
                };

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
    fn test_wave() {
        let mut wave = Wave::new(WaveDirection::Horizontal)
            .with_amplitude(5.0)
            .with_frequency(0.05);

        let input = Frame::new(100, 100).expect("should succeed in test");
        let mut output = Frame::new(100, 100).expect("should succeed in test");
        let params = EffectParams::new();
        wave.apply(&input, &mut output, &params)
            .expect("should succeed in test");
    }
}
