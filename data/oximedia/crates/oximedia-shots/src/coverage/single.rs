//! Single shot detection.

use crate::types::Shot;

/// Single shot detector.
pub struct SingleDetector;

impl SingleDetector {
    /// Create a new single shot detector.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Detect if a shot is a single (one person).
    #[must_use]
    pub fn is_single(&self, shot: &Shot) -> bool {
        matches!(
            shot.shot_type,
            crate::types::ShotType::CloseUp
                | crate::types::ShotType::MediumCloseUp
                | crate::types::ShotType::MediumShot
        )
    }
}

impl Default for SingleDetector {
    fn default() -> Self {
        Self::new()
    }
}
