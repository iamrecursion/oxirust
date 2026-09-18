//! Mosaic/pixelate effect.

use crate::{EffectParams, Frame, VfxResult, VideoEffect};
use serde::{Deserialize, Serialize};

/// Mosaic mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MosaicMode {
    /// Square blocks.
    Square,
    /// Hexagonal tiles.
    Hexagonal,
    /// Circular dots.
    Circular,
}

/// Mosaic effect.
pub struct Mosaic {
    mode: MosaicMode,
    block_size: u32,
}

impl Mosaic {
    /// Create a new mosaic effect.
    #[must_use]
    pub const fn new(mode: MosaicMode) -> Self {
        Self {
            mode,
            block_size: 8,
        }
    }

    /// Set block/tile size.
    #[must_use]
    pub fn with_block_size(mut self, size: u32) -> Self {
        self.block_size = size.max(1);
        self
    }
}

impl VideoEffect for Mosaic {
    fn name(&self) -> &'static str {
        "Mosaic"
    }

    fn description(&self) -> &'static str {
        "Mosaic/pixelate effect"
    }

    fn apply(
        &mut self,
        input: &Frame,
        output: &mut Frame,
        _params: &EffectParams,
    ) -> VfxResult<()> {
        for y in 0..output.height {
            for x in 0..output.width {
                let block_x = (x / self.block_size) * self.block_size;
                let block_y = (y / self.block_size) * self.block_size;

                // Sample center of block
                let sample_x = block_x + self.block_size / 2;
                let sample_y = block_y + self.block_size / 2;

                let pixel = input
                    .get_pixel(
                        sample_x.min(input.width - 1),
                        sample_y.min(input.height - 1),
                    )
                    .unwrap_or([0, 0, 0, 0]);

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
    fn test_mosaic() {
        let mut mosaic = Mosaic::new(MosaicMode::Square).with_block_size(16);
        let input = Frame::new(100, 100).expect("should succeed in test");
        let mut output = Frame::new(100, 100).expect("should succeed in test");
        let params = EffectParams::new();
        mosaic
            .apply(&input, &mut output, &params)
            .expect("should succeed in test");
    }
}
