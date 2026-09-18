//! Difference visualization.

/// Difference calculator.
pub struct DifferenceCalculator;

impl DifferenceCalculator {
    /// Create a new difference calculator.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Calculate difference map.
    #[must_use]
    pub fn calculate(&self, _reference: &[u8], _signal: &[u8]) -> Vec<u8> {
        Vec::new()
    }
}

impl Default for DifferenceCalculator {
    fn default() -> Self {
        Self::new()
    }
}
