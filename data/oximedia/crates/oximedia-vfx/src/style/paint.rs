//! Oil painting effect.

use crate::{EffectParams, Frame, VfxResult, VideoEffect};
use serde::{Deserialize, Serialize};

/// Paint style.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PaintStyle {
    /// Oil painting.
    Oil,
    /// Watercolor.
    Watercolor,
    /// Acrylic.
    Acrylic,
}

/// Oil paint effect.
pub struct OilPaint {
    style: PaintStyle,
    radius: u32,
    levels: u8,
}

impl OilPaint {
    /// Create a new oil paint effect.
    #[must_use]
    pub const fn new(style: PaintStyle) -> Self {
        Self {
            style,
            radius: 3,
            levels: 20,
        }
    }

    /// Set brush radius.
    #[must_use]
    pub fn with_radius(mut self, radius: u32) -> Self {
        self.radius = radius.max(1);
        self
    }

    /// Set intensity levels.
    #[must_use]
    pub fn with_levels(mut self, levels: u8) -> Self {
        self.levels = levels.max(1);
        self
    }
}

impl VideoEffect for OilPaint {
    fn name(&self) -> &'static str {
        "Oil Paint"
    }

    fn description(&self) -> &'static str {
        "Oil painting effect"
    }

    fn apply(
        &mut self,
        input: &Frame,
        output: &mut Frame,
        _params: &EffectParams,
    ) -> VfxResult<()> {
        let radius = self.radius as i32;

        for y in 0..output.height {
            for x in 0..output.width {
                let mut intensity_count = vec![0u32; self.levels as usize];
                let mut avg_r = vec![0u32; self.levels as usize];
                let mut avg_g = vec![0u32; self.levels as usize];
                let mut avg_b = vec![0u32; self.levels as usize];

                for dy in -radius..=radius {
                    for dx in -radius..=radius {
                        let nx = (x as i32 + dx).max(0).min(input.width as i32 - 1) as u32;
                        let ny = (y as i32 + dy).max(0).min(input.height as i32 - 1) as u32;

                        if let Some(pixel) = input.get_pixel(nx, ny) {
                            let intensity =
                                ((u32::from(pixel[0]) + u32::from(pixel[1]) + u32::from(pixel[2]))
                                    / 3
                                    * u32::from(self.levels)
                                    / 256) as usize;
                            let intensity = intensity.min((self.levels - 1) as usize);

                            intensity_count[intensity] += 1;
                            avg_r[intensity] += u32::from(pixel[0]);
                            avg_g[intensity] += u32::from(pixel[1]);
                            avg_b[intensity] += u32::from(pixel[2]);
                        }
                    }
                }

                // Find most common intensity
                let max_idx = intensity_count
                    .iter()
                    .enumerate()
                    .max_by_key(|(_, &count)| count)
                    .map_or(0, |(idx, _)| idx);

                let count = intensity_count[max_idx].max(1);
                let result = [
                    (avg_r[max_idx] / count) as u8,
                    (avg_g[max_idx] / count) as u8,
                    (avg_b[max_idx] / count) as u8,
                    255,
                ];

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
    fn test_oil_paint() {
        let mut paint = OilPaint::new(PaintStyle::Oil)
            .with_radius(3)
            .with_levels(20);

        let input = Frame::new(100, 100).expect("should succeed in test");
        let mut output = Frame::new(100, 100).expect("should succeed in test");
        let params = EffectParams::new();
        paint
            .apply(&input, &mut output, &params)
            .expect("should succeed in test");
    }
}
