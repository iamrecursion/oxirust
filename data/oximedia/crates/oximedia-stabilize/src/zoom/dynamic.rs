//! Dynamic zoom effects (Ken Burns).

use crate::transform::calculate::StabilizationTransform;

/// Dynamic zoom effect generator.
pub struct DynamicZoom {
    zoom_speed: f64,
}

impl DynamicZoom {
    /// Create a new dynamic zoom generator.
    #[must_use]
    pub fn new(zoom_speed: f64) -> Self {
        Self { zoom_speed }
    }

    /// Apply dynamic zoom to transforms.
    #[must_use]
    pub fn apply(&self, transforms: &[StabilizationTransform]) -> Vec<StabilizationTransform> {
        transforms
            .iter()
            .enumerate()
            .map(|(i, t)| {
                let mut zoomed = *t;
                let zoom = 1.0 + (i as f64 * self.zoom_speed);
                zoomed.scale *= zoom;
                zoomed
            })
            .collect()
    }
}
