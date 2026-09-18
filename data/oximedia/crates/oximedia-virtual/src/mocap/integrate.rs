//! Motion capture system integration

use super::MocapMarker;

/// Motion capture integrator
pub struct MocapIntegrator {
    markers: Vec<MocapMarker>,
}

impl MocapIntegrator {
    /// Create new integrator
    #[must_use]
    pub fn new() -> Self {
        Self {
            markers: Vec::new(),
        }
    }

    /// Update markers
    pub fn update(&mut self, markers: Vec<MocapMarker>) {
        self.markers = markers;
    }
}

impl Default for MocapIntegrator {
    fn default() -> Self {
        Self::new()
    }
}
