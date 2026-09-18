//! Tracking data export for Unreal Engine

use super::UnrealTransform;
use serde::{Deserialize, Serialize};

/// Tracking data packet
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackingData {
    /// Transforms
    pub transforms: Vec<UnrealTransform>,
    /// Frame number
    pub frame: u64,
}

impl TrackingData {
    /// Create new tracking data
    #[must_use]
    pub fn new(frame: u64) -> Self {
        Self {
            transforms: Vec::new(),
            frame,
        }
    }

    /// Add transform
    pub fn add_transform(&mut self, transform: UnrealTransform) {
        self.transforms.push(transform);
    }
}
