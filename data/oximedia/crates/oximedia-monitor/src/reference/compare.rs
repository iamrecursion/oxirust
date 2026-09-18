//! Signal comparison utilities.

use super::ReferenceDiff;

/// Signal comparator.
pub struct SignalComparator;

impl SignalComparator {
    /// Create a new signal comparator.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Compare two signals.
    #[must_use]
    pub fn compare(&self, _reference: &[u8], _signal: &[u8]) -> ReferenceDiff {
        ReferenceDiff::default()
    }
}

impl Default for SignalComparator {
    fn default() -> Self {
        Self::new()
    }
}
