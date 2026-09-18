//! Master shot detection.

use crate::types::Shot;

/// Master shot detector.
pub struct MasterDetector;

impl MasterDetector {
    /// Create a new master shot detector.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Detect if a shot is a master shot.
    #[must_use]
    pub fn is_master(&self, shot: &Shot) -> bool {
        // Master shots are typically wide shots at the beginning of scenes
        matches!(
            shot.shot_type,
            crate::types::ShotType::LongShot | crate::types::ShotType::ExtremeLongShot
        )
    }
}

impl Default for MasterDetector {
    fn default() -> Self {
        Self::new()
    }
}
