//! Multi-camera coordination
//!
//! Provides multi-camera tracking and coordination with automated switching.

pub mod manager;
pub mod switch;

use crate::tracking::CameraPose;
use serde::{Deserialize, Serialize};

/// Camera identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CameraId(pub usize);

/// Multi-camera state
#[derive(Debug, Clone)]
pub struct MultiCameraState {
    /// Active camera
    pub active_camera: CameraId,
    /// All camera poses
    pub poses: Vec<(CameraId, CameraPose)>,
}

impl MultiCameraState {
    /// Create new multi-camera state
    #[must_use]
    pub fn new() -> Self {
        Self {
            active_camera: CameraId(0),
            poses: Vec::new(),
        }
    }

    /// Get active camera pose
    #[must_use]
    pub fn active_pose(&self) -> Option<&CameraPose> {
        self.poses
            .iter()
            .find(|(id, _)| *id == self.active_camera)
            .map(|(_, pose)| pose)
    }
}

impl Default for MultiCameraState {
    fn default() -> Self {
        Self::new()
    }
}
