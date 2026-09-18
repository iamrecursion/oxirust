//! Unreal Engine metadata export

use super::UnrealTransform;
use serde::{Deserialize, Serialize};

/// Unreal metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnrealMetadata {
    /// Camera transform
    pub camera: UnrealTransform,
    /// Frame number
    pub frame: u64,
    /// Timestamp
    pub timestamp: f64,
}

impl UnrealMetadata {
    /// Create new metadata
    #[must_use]
    pub fn new(frame: u64, timestamp: f64) -> Self {
        Self {
            camera: UnrealTransform::default(),
            frame,
            timestamp,
        }
    }

    /// Export to JSON
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}
