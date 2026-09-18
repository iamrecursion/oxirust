//! LED signal processing and optimization
//!
//! Provides signal processing for LED wall output including gamma correction,
//! dithering, and temporal optimization.

use crate::{Result, VirtualProductionError};
use serde::{Deserialize, Serialize};

/// LED processor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LedProcessorConfig {
    /// Enable gamma correction
    pub gamma_correction: bool,
    /// Gamma value
    pub gamma: f32,
    /// Enable dithering
    pub dithering: bool,
    /// Enable temporal optimization
    pub temporal_optimization: bool,
    /// Bit depth (8, 10, 12, 16)
    pub bit_depth: u8,
}

impl Default for LedProcessorConfig {
    fn default() -> Self {
        Self {
            gamma_correction: true,
            gamma: 2.2,
            dithering: true,
            temporal_optimization: true,
            bit_depth: 10,
        }
    }
}

/// Gamma lookup table
#[derive(Debug, Clone)]
pub struct GammaLut {
    lut: Vec<u16>,
    #[allow(dead_code)]
    gamma: f32,
    #[allow(dead_code)]
    bit_depth: u8,
}

impl GammaLut {
    /// Create new gamma LUT
    #[must_use]
    pub fn new(gamma: f32, bit_depth: u8) -> Self {
        let max_val = (1 << bit_depth) - 1;
        let lut_size = 256;

        let lut: Vec<u16> = (0..lut_size)
            .map(|i| {
                let normalized = i as f32 / (lut_size - 1) as f32;
                let corrected = normalized.powf(gamma);
                (corrected * max_val as f32) as u16
            })
            .collect();

        Self {
            lut,
            gamma,
            bit_depth,
        }
    }

    /// Apply gamma correction to 8-bit value
    #[must_use]
    pub fn apply(&self, value: u8) -> u16 {
        self.lut[value as usize]
    }

    /// Apply gamma correction to RGB pixel
    #[must_use]
    pub fn apply_rgb(&self, rgb: [u8; 3]) -> [u16; 3] {
        [self.apply(rgb[0]), self.apply(rgb[1]), self.apply(rgb[2])]
    }
}

/// Temporal dithering for LED walls
#[derive(Debug, Clone)]
pub struct TemporalDither {
    frame_count: u64,
    pattern: Vec<Vec<i8>>,
}

impl TemporalDither {
    /// Create new temporal dithering
    #[must_use]
    pub fn new() -> Self {
        // 4x4 Bayer matrix for temporal dithering
        let pattern = vec![
            vec![0, 8, 2, 10],
            vec![12, 4, 14, 6],
            vec![3, 11, 1, 9],
            vec![15, 7, 13, 5],
        ];

        Self {
            frame_count: 0,
            pattern,
        }
    }

    /// Apply temporal dithering
    pub fn apply(&mut self, value: u16, x: usize, y: usize) -> u16 {
        let threshold = self.pattern[y % 4][x % 4];
        let temporal_offset = (self.frame_count % 4) as i8;

        let dither_value = (threshold + temporal_offset * 4) % 16;

        value.saturating_add(dither_value as u16 / 16)
    }

    /// Next frame
    pub fn next_frame(&mut self) {
        self.frame_count += 1;
    }
}

impl Default for TemporalDither {
    fn default() -> Self {
        Self::new()
    }
}

/// LED signal processor
pub struct LedProcessor {
    config: LedProcessorConfig,
    gamma_lut: Option<GammaLut>,
    dither: Option<TemporalDither>,
}

impl LedProcessor {
    /// Create new LED processor
    pub fn new(config: LedProcessorConfig) -> Result<Self> {
        let gamma_lut = if config.gamma_correction {
            Some(GammaLut::new(config.gamma, config.bit_depth))
        } else {
            None
        };

        let dither = if config.dithering {
            Some(TemporalDither::new())
        } else {
            None
        };

        Ok(Self {
            config,
            gamma_lut,
            dither,
        })
    }

    /// Process frame for LED output
    pub fn process(&mut self, input: &[u8], width: usize, height: usize) -> Result<Vec<u16>> {
        if input.len() != width * height * 3 {
            return Err(VirtualProductionError::LedWall(
                "Invalid input size".to_string(),
            ));
        }

        let mut output = Vec::with_capacity(width * height * 3);

        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * 3;
                let rgb = [input[idx], input[idx + 1], input[idx + 2]];

                // Apply gamma correction
                let mut corrected = if let Some(lut) = &self.gamma_lut {
                    lut.apply_rgb(rgb)
                } else {
                    [u16::from(rgb[0]), u16::from(rgb[1]), u16::from(rgb[2])]
                };

                // Apply dithering
                if let Some(dither) = &mut self.dither {
                    corrected[0] = dither.apply(corrected[0], x, y);
                    corrected[1] = dither.apply(corrected[1], x, y);
                    corrected[2] = dither.apply(corrected[2], x, y);
                }

                output.push(corrected[0]);
                output.push(corrected[1]);
                output.push(corrected[2]);
            }
        }

        // Advance dither frame
        if let Some(dither) = &mut self.dither {
            dither.next_frame();
        }

        Ok(output)
    }

    /// Get configuration
    #[must_use]
    pub fn config(&self) -> &LedProcessorConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gamma_lut() {
        let lut = GammaLut::new(2.2, 10);
        let result = lut.apply(128);
        assert!(result > 0);
        assert!(result < 1024);
    }

    #[test]
    fn test_gamma_lut_rgb() {
        let lut = GammaLut::new(2.2, 10);
        let result = lut.apply_rgb([255, 128, 0]);
        assert_eq!(result[0], 1023);
        assert!(result[1] > 0);
    }

    #[test]
    fn test_temporal_dither() {
        let mut dither = TemporalDither::new();
        let result = dither.apply(512, 0, 0);
        assert!(result >= 512);
    }

    #[test]
    fn test_led_processor() {
        let config = LedProcessorConfig::default();
        let mut processor = LedProcessor::new(config).expect("should succeed in test");

        let input = vec![128u8; 10 * 10 * 3];
        let output = processor
            .process(&input, 10, 10)
            .expect("should succeed in test");

        assert_eq!(output.len(), 10 * 10 * 3);
    }

    #[test]
    fn test_led_processor_no_gamma() {
        let config = LedProcessorConfig {
            gamma_correction: false,
            gamma: 2.2,
            dithering: false,
            temporal_optimization: false,
            bit_depth: 10,
        };

        let mut processor = LedProcessor::new(config).expect("should succeed in test");
        let input = vec![128u8; 10 * 10 * 3];
        let output = processor
            .process(&input, 10, 10)
            .expect("should succeed in test");

        assert_eq!(output.len(), 10 * 10 * 3);
    }
}
