//! Automated camera switching

use super::{CameraId, MultiCameraState};
use crate::Result;

/// Camera switcher
pub struct CameraSwitcher {
    #[allow(dead_code)]
    switch_threshold: f32,
}

impl CameraSwitcher {
    /// Create new camera switcher
    #[must_use]
    pub fn new(switch_threshold: f32) -> Self {
        Self { switch_threshold }
    }

    /// Determine best camera
    pub fn select_camera(&self, state: &MultiCameraState) -> Result<CameraId> {
        // For now, just return active camera
        Ok(state.active_camera)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_camera_switcher() {
        let switcher = CameraSwitcher::new(0.5);
        let state = MultiCameraState::new();
        let result = switcher.select_camera(&state);
        assert!(result.is_ok());
    }
}
