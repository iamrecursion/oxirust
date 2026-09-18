//! Real-time LUT application

use crate::Result;

/// LUT processor
pub struct LutProcessor;

impl LutProcessor {
    /// Create new LUT processor
    pub fn new() -> Result<Self> {
        Ok(Self)
    }

    /// Apply LUT to frame
    pub fn apply(&mut self, frame: &[u8], _width: usize, _height: usize) -> Result<Vec<u8>> {
        Ok(frame.to_vec())
    }
}
