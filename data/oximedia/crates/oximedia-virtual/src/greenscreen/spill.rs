//! Spill suppression for green screen

use crate::Result;

/// Spill suppressor
pub struct SpillSuppressor {
    #[allow(dead_code)]
    strength: f32,
}

impl SpillSuppressor {
    /// Create new spill suppressor
    #[must_use]
    pub fn new(strength: f32) -> Self {
        Self { strength }
    }

    /// Suppress spill in frame
    pub fn suppress(&mut self, frame: &[u8], _width: usize, _height: usize) -> Result<Vec<u8>> {
        Ok(frame.to_vec())
    }
}
