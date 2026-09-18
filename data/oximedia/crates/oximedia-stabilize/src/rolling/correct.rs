//! Rolling shutter correction.

use crate::error::{StabilizeError, StabilizeResult};
use crate::transform::calculate::StabilizationTransform;
use crate::Frame;

/// Rolling shutter corrector.
#[derive(Debug)]
pub struct RollingShutterCorrector {
    scan_line_time: f64,
}

impl RollingShutterCorrector {
    /// Create a new rolling shutter corrector.
    #[must_use]
    pub fn new() -> Self {
        Self {
            scan_line_time: 1.0 / 1000.0,
        }
    }

    /// Correct rolling shutter in transforms.
    ///
    /// # Errors
    ///
    /// Returns an error if transforms or frames are empty.
    pub fn correct_transforms(
        &self,
        transforms: &[StabilizationTransform],
        _frames: &[Frame],
    ) -> StabilizeResult<Vec<StabilizationTransform>> {
        if transforms.is_empty() {
            return Err(StabilizeError::EmptyFrameSequence);
        }

        // For now, return transforms as-is
        // Full implementation would analyze per-scanline motion
        Ok(transforms.to_vec())
    }
}

impl Default for RollingShutterCorrector {
    fn default() -> Self {
        Self::new()
    }
}
