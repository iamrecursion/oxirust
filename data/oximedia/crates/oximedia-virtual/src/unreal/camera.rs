//! Camera data export for Unreal Engine

use super::UnrealTransform;
use crate::tracking::CameraPose;

/// Convert camera pose to Unreal transform
#[must_use]
pub fn pose_to_unreal(pose: &CameraPose) -> UnrealTransform {
    // Convert from meters to centimeters and coordinate system
    UnrealTransform {
        position: [
            pose.position.x * 100.0,
            pose.position.y * 100.0,
            pose.position.z * 100.0,
        ],
        rotation: [0.0, 0.0, 0.0], // Would convert quaternion to Euler
        scale: [1.0, 1.0, 1.0],
    }
}
