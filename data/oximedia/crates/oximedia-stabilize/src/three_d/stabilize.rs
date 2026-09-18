//! 3D-based stabilization.

use crate::error::{StabilizeError, StabilizeResult};
use crate::motion::tracker::FeatureTrack;
use crate::transform::calculate::StabilizationTransform;

/// 3D stabilizer.
#[derive(Debug)]
pub struct ThreeDStabilizer {
    enable_full_3d: bool,
}

impl ThreeDStabilizer {
    /// Create a new 3D stabilizer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            enable_full_3d: true,
        }
    }

    /// Apply full 3D camera-motion stabilization to transforms.
    ///
    /// # Honesty note
    ///
    /// This is **not implemented**. A real 3D camera-motion solve requires
    /// structure-from-motion: triangulating 3D points from `tracks`,
    /// recovering per-frame camera pose, smoothing the resulting 3D camera
    /// path, and re-projecting it back to 2D correction transforms. An
    /// earlier revision of this function silently returned the input
    /// `transforms` unchanged, which let `StabilizationMode::ThreeD`
    /// masquerade as "Full 3D camera motion" (see the `lib.rs` doc comment)
    /// while actually just degrading to whatever the base affine-approximation
    /// estimator already computed (see `motion::estimate::MotionEstimator`,
    /// which explicitly comments "For 3D mode, start with affine
    /// approximation"). Returning `Ok` with a passthrough result is
    /// fabricated success: the caller cannot distinguish it from a real 3D
    /// solve. This now fails loudly instead.
    ///
    /// # Errors
    ///
    /// Returns [`StabilizeError::EmptyFrameSequence`] if `transforms` is
    /// empty, or [`StabilizeError::ThreeDStabilizationError`]
    /// unconditionally otherwise, since no real 3D solve is implemented yet.
    // TODO(0.2.x): real 3D camera-motion solve (structure-from-motion):
    // triangulate 3D points from `tracks`, recover per-frame camera pose,
    // smooth the 3D camera path, and re-project to correction transforms.
    pub fn stabilize_3d(
        &self,
        transforms: &[StabilizationTransform],
        _tracks: &[FeatureTrack],
    ) -> StabilizeResult<Vec<StabilizationTransform>> {
        if transforms.is_empty() {
            return Err(StabilizeError::EmptyFrameSequence);
        }

        let _ = self.enable_full_3d;
        Err(StabilizeError::ThreeDStabilizationError(
            "full 3D camera-motion stabilization (structure-from-motion) is not yet \
             implemented; refusing to silently pass base-mode transforms through as if \
             they were 3D-corrected"
                .to_string(),
        ))
    }
}

impl Default for ThreeDStabilizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stabilize_3d_is_honest_err_not_passthrough() {
        let stabilizer = ThreeDStabilizer::new();
        let transforms = vec![StabilizationTransform::identity(0)];

        let result = stabilizer.stabilize_3d(&transforms, &[]);

        assert!(
            result.is_err(),
            "stabilize_3d must not silently pass transforms through as if they were \
             really 3D-corrected"
        );
        assert!(matches!(
            result.unwrap_err(),
            StabilizeError::ThreeDStabilizationError(_)
        ));
    }

    #[test]
    fn test_stabilize_3d_empty_transforms_still_errors_empty_sequence() {
        let stabilizer = ThreeDStabilizer::new();

        let result = stabilizer.stabilize_3d(&[], &[]);

        assert!(matches!(
            result.unwrap_err(),
            StabilizeError::EmptyFrameSequence
        ));
    }
}
