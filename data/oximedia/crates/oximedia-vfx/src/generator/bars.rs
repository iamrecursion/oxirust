//! SMPTE and EBU color bars generator.

use crate::{Color, EffectParams, Frame, VfxResult, VideoEffect};
use serde::{Deserialize, Serialize};

/// Color bars type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BarsType {
    /// SMPTE color bars (standard).
    Smpte,
    /// EBU color bars.
    Ebu,
    /// SMPTE HD bars.
    SmpteHd,
}

/// Color bars generator.
///
/// Generates SMPTE or EBU color bars for testing and calibration.
pub struct ColorBars {
    bars_type: BarsType,
}

impl ColorBars {
    /// Create a new color bars generator.
    #[must_use]
    pub const fn new(bars_type: BarsType) -> Self {
        Self { bars_type }
    }

    fn get_smpte_colors() -> Vec<Color> {
        vec![
            Color::rgb(192, 192, 192), // White (75%)
            Color::rgb(192, 192, 0),   // Yellow
            Color::rgb(0, 192, 192),   // Cyan
            Color::rgb(0, 192, 0),     // Green
            Color::rgb(192, 0, 192),   // Magenta
            Color::rgb(192, 0, 0),     // Red
            Color::rgb(0, 0, 192),     // Blue
        ]
    }

    fn get_ebu_colors() -> Vec<Color> {
        vec![
            Color::rgb(255, 255, 255), // White (100%)
            Color::rgb(255, 255, 0),   // Yellow
            Color::rgb(0, 255, 255),   // Cyan
            Color::rgb(0, 255, 0),     // Green
            Color::rgb(255, 0, 255),   // Magenta
            Color::rgb(255, 0, 0),     // Red
            Color::rgb(0, 0, 255),     // Blue
            Color::rgb(0, 0, 0),       // Black
        ]
    }

    fn render_smpte(&self, output: &mut Frame) {
        let colors = Self::get_smpte_colors();
        let width = output.width;
        let height = output.height;

        // Top section (2/3 height) - main bars
        let top_height = (height * 2) / 3;
        let bar_width = width / 7;

        for y in 0..top_height {
            for x in 0..width {
                let bar_idx = ((x / bar_width) as usize).min(6);
                output.set_pixel(x, y, colors[bar_idx].to_rgba());
            }
        }

        // Middle section (1/12 height) - reverse blue bars
        let mid_start = top_height;
        let mid_height = height / 12;
        let mid_colors = vec![
            Color::rgb(0, 0, 192),     // Blue
            Color::rgb(0, 0, 0),       // Black
            Color::rgb(192, 0, 192),   // Magenta
            Color::rgb(0, 0, 0),       // Black
            Color::rgb(0, 192, 192),   // Cyan
            Color::rgb(0, 0, 0),       // Black
            Color::rgb(192, 192, 192), // White
        ];

        for y in mid_start..mid_start + mid_height {
            for x in 0..width {
                let bar_idx = ((x / bar_width) as usize).min(6);
                output.set_pixel(x, y, mid_colors[bar_idx].to_rgba());
            }
        }

        // Bottom section (1/4 height) - PLUGE and other test signals
        let bottom_start = mid_start + mid_height;

        for y in bottom_start..height {
            for x in 0..width {
                let section = x * 4 / width;
                let color = match section {
                    0 => Color::rgb(0, 33, 76),     // -I
                    1 => Color::rgb(255, 255, 255), // White
                    2 => Color::rgb(50, 0, 128),    // +Q
                    _ => Color::rgb(0, 0, 0),       // Black
                };
                output.set_pixel(x, y, color.to_rgba());
            }
        }
    }

    fn render_ebu(&self, output: &mut Frame) {
        let colors = Self::get_ebu_colors();
        let width = output.width;
        let height = output.height;
        let bar_width = width / 8;

        for y in 0..height {
            for x in 0..width {
                let bar_idx = ((x / bar_width) as usize).min(7);
                output.set_pixel(x, y, colors[bar_idx].to_rgba());
            }
        }
    }
}

impl VideoEffect for ColorBars {
    fn name(&self) -> &'static str {
        "Color Bars"
    }

    fn description(&self) -> &'static str {
        "SMPTE/EBU color bars generator"
    }

    fn apply(
        &mut self,
        _input: &Frame,
        output: &mut Frame,
        _params: &EffectParams,
    ) -> VfxResult<()> {
        match self.bars_type {
            BarsType::Smpte | BarsType::SmpteHd => self.render_smpte(output),
            BarsType::Ebu => self.render_ebu(output),
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
    fn test_smpte_bars() {
        let mut bars = ColorBars::new(BarsType::Smpte);
        let input = Frame::new(640, 480).expect("should succeed in test");
        let mut output = Frame::new(640, 480).expect("should succeed in test");
        let params = EffectParams::new();
        bars.apply(&input, &mut output, &params)
            .expect("should succeed in test");

        // Check that output has colors
        let pixel = output.get_pixel(100, 100).expect("should succeed in test");
        assert!(pixel[0] > 0 || pixel[1] > 0 || pixel[2] > 0);
    }

    #[test]
    fn test_ebu_bars() {
        let mut bars = ColorBars::new(BarsType::Ebu);
        let input = Frame::new(640, 480).expect("should succeed in test");
        let mut output = Frame::new(640, 480).expect("should succeed in test");
        let params = EffectParams::new();
        bars.apply(&input, &mut output, &params)
            .expect("should succeed in test");
    }
}
