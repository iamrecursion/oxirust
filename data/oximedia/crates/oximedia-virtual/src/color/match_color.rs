//! Camera to LED color matching

use crate::Result;

/// Color matcher
pub struct ColorMatcher;

impl ColorMatcher {
    /// Create new color matcher
    pub fn new() -> Result<Self> {
        Ok(Self)
    }

    /// Process frame
    pub fn process(&mut self, frame: &[u8], _width: usize, _height: usize) -> Result<Vec<u8>> {
        Ok(frame.to_vec())
    }
}
