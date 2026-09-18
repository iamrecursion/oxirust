//! Two-shot detection.

use crate::types::Shot;

/// Two-shot detector.
pub struct TwoShotDetector;

impl TwoShotDetector {
    /// Create a new two-shot detector.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Detect if a shot is a two-shot.
    #[must_use]
    pub fn is_two_shot(&self, shot: &Shot) -> bool {
        matches!(shot.coverage, crate::types::CoverageType::TwoShot)
    }
}

impl Default for TwoShotDetector {
    fn default() -> Self {
        Self::new()
    }
}
