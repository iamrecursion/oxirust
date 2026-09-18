//! Horizon leveling.

use crate::error::{StabilizeError, StabilizeResult};
use crate::horizon::detect::{HorizonDetector, HorizonLine};
use crate::transform::calculate::StabilizationTransform;
use crate::Frame;

/// Horizon leveler.
#[derive(Debug)]
pub struct HorizonLeveler {
    detector: HorizonDetector,
    correction_strength: f64,
}

impl HorizonLeveler {
    /// Create a new horizon leveler.
    #[must_use]
    pub fn new() -> Self {
        Self {
            detector: HorizonDetector::new(),
            correction_strength: 1.0,
        }
    }

    /// Apply horizon leveling to transforms.
    ///
    /// # Errors
    ///
    /// Returns an error if transforms or frames are empty.
    pub fn level(
        &self,
        transforms: &[StabilizationTransform],
        frames: &[Frame],
    ) -> StabilizeResult<Vec<StabilizationTransform>> {
        if transforms.is_empty() || frames.is_empty() {
            return Err(StabilizeError::EmptyFrameSequence);
        }

        let leveled: Vec<_> = transforms
            .iter()
            .zip(frames.iter())
            .map(|(transform, frame)| {
                if let Some(horizon) = self.detector.detect(frame) {
                    self.apply_horizon_correction(transform, &horizon)
                } else {
                    *transform
                }
            })
            .collect();

        Ok(leveled)
    }

    /// Apply horizon correction to a transform.
    fn apply_horizon_correction(
        &self,
        transform: &StabilizationTransform,
        horizon: &HorizonLine,
    ) -> StabilizationTransform {
        let mut corrected = *transform;
        corrected.angle -= horizon.angle * self.correction_strength * horizon.confidence;
        corrected
    }
}

impl Default for HorizonLeveler {
    fn default() -> Self {
        Self::new()
    }
}
