//! Multi-output preview system

use crate::Result;

/// Preview output
pub struct PreviewOutput {
    outputs: Vec<String>,
}

impl PreviewOutput {
    /// Create new preview output
    #[must_use]
    pub fn new() -> Self {
        Self {
            outputs: Vec::new(),
        }
    }

    /// Add output
    pub fn add_output(&mut self, name: String) {
        self.outputs.push(name);
    }

    /// Send frame to outputs
    pub fn send(&mut self, _frame: &[u8], _width: usize, _height: usize) -> Result<()> {
        Ok(())
    }
}

impl Default for PreviewOutput {
    fn default() -> Self {
        Self::new()
    }
}
