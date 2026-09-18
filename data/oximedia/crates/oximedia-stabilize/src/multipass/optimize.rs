//! Global optimization across entire video.

use crate::transform::calculate::StabilizationTransform;

/// Global optimizer for video-wide consistency.
pub struct GlobalOptimizer;

impl GlobalOptimizer {
    /// Create a new global optimizer.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Optimize transforms globally.
    #[must_use]
    pub fn optimize(&self, transforms: &[StabilizationTransform]) -> Vec<StabilizationTransform> {
        transforms.to_vec()
    }
}

impl Default for GlobalOptimizer {
    fn default() -> Self {
        Self::new()
    }
}
