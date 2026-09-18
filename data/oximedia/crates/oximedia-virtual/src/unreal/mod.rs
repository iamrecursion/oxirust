//! Unreal Engine integration
//!
//! Provides metadata export for Unreal Engine virtual production.

pub mod camera;
pub mod metadata;
pub mod tracking;

use serde::{Deserialize, Serialize};

/// Unreal coordinate system (left-handed, Z-up)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct UnrealTransform {
    /// Position in centimeters
    pub position: [f64; 3],
    /// Rotation in degrees
    pub rotation: [f64; 3],
    /// Scale
    pub scale: [f64; 3],
}

impl UnrealTransform {
    /// Create new Unreal transform
    #[must_use]
    pub fn new() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0],
            scale: [1.0, 1.0, 1.0],
        }
    }
}

impl Default for UnrealTransform {
    fn default() -> Self {
        Self::new()
    }
}
