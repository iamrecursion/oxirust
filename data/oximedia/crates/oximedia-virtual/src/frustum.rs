//! Camera frustum geometry for virtual set rendering.
//!
//! Provides a pure-Rust frustum implementation (independent of nalgebra)
//! for use in virtual production pipelines. Uses the `TrackedPose` from the
//! local `camera_tracking` module.

#![allow(dead_code)]

use crate::camera_tracking::TrackedPose;

// ---------------------------------------------------------------------------
// Frustum
// ---------------------------------------------------------------------------

/// A symmetric perspective frustum defined by horizontal FOV and aspect ratio.
#[derive(Debug, Clone, PartialEq)]
pub struct Frustum {
    /// Horizontal field-of-view in degrees.
    pub fov_h_deg: f64,
    /// Vertical field-of-view in degrees (derived from horizontal FOV and aspect ratio).
    pub fov_v_deg: f64,
    /// Near clip plane distance in metres.
    pub near: f64,
    /// Far clip plane distance in metres.
    pub far: f64,
}

impl Frustum {
    /// Create a frustum from horizontal FOV and aspect ratio (width / height).
    ///
    /// Vertical FOV is derived automatically.
    #[must_use]
    pub fn new(fov_h_deg: f64, aspect_ratio: f64, near: f64, far: f64) -> Self {
        let fov_v = Self::fov_v_from_fov_h(fov_h_deg, aspect_ratio);
        Self {
            fov_h_deg,
            fov_v_deg: fov_v,
            near,
            far,
        }
    }

    /// Compute vertical FOV from horizontal FOV and aspect ratio (width / height).
    ///
    /// Uses `fov_v = 2 * atan(tan(fov_h / 2) / aspect)`.
    #[must_use]
    pub fn fov_v_from_fov_h(fov_h: f64, aspect: f64) -> f64 {
        if aspect <= 0.0 {
            return 0.0;
        }
        let half_h = (fov_h.to_radians() / 2.0).tan();
        let half_v = half_h / aspect;
        half_v.atan().to_degrees() * 2.0
    }

    /// Returns `true` if the point `(x, y, z)` in camera space lies inside the frustum.
    ///
    /// Assumes the camera looks along the positive Z axis.
    #[must_use]
    pub fn contains_point(&self, x: f64, y: f64, z: f64) -> bool {
        if z < self.near || z > self.far {
            return false;
        }
        let half_h = (self.fov_h_deg.to_radians() / 2.0).tan() * z;
        let half_v = (self.fov_v_deg.to_radians() / 2.0).tan() * z;
        x.abs() <= half_h && y.abs() <= half_v
    }

    /// Approximate solid angle of the frustum in steradians.
    ///
    /// Formula: `4 * atan(sin(fov_h/2) * sin(fov_v/2) / cos(fov_h/2 + fov_v/2 - π/2 + π/2))`.
    /// For small angles, this simplifies to `fov_h_rad * fov_v_rad`.
    #[must_use]
    pub fn solid_angle(&self) -> f64 {
        let h_rad = self.fov_h_deg.to_radians();
        let v_rad = self.fov_v_deg.to_radians();
        // Spherical rectangle solid angle approximation.
        4.0 * ((h_rad / 2.0).sin() * (v_rad / 2.0).sin()).asin()
    }
}

// ---------------------------------------------------------------------------
// View frustum (frustum + pose)
// ---------------------------------------------------------------------------

/// A frustum anchored at a tracked camera pose in world space.
#[derive(Debug, Clone)]
pub struct ViewFrustum {
    /// The frustum geometry.
    pub frustum: Frustum,
    /// The current camera pose.
    pub pose: TrackedPose,
}

impl ViewFrustum {
    /// Create a new view frustum with identity pose.
    #[must_use]
    pub fn new(frustum: Frustum) -> Self {
        Self {
            frustum,
            pose: TrackedPose::new(0.0, 0.0, 0.0),
        }
    }

    /// Update the camera pose.
    pub fn update_pose(&mut self, pose: TrackedPose) {
        self.pose = pose;
    }

    /// Transform a world-space point into camera space.
    ///
    /// This is a simplified implementation that only applies translation
    /// (no rotation matrix). Rotation support can be layered on top when
    /// a full matrix library is available.
    #[must_use]
    pub fn world_to_camera(&self, wx: f64, wy: f64, wz: f64) -> (f64, f64, f64) {
        (wx - self.pose.x, wy - self.pose.y, wz - self.pose.z)
    }
}

// ---------------------------------------------------------------------------
// Screen projection
// ---------------------------------------------------------------------------

/// Project a camera-space point onto a screen of given dimensions.
///
/// Returns `None` if the point is behind the camera (z ≤ 0).
///
/// # Arguments
/// * `x`, `y`, `z` – point in camera space (z is depth; must be > 0)
/// * `width`, `height` – screen dimensions in pixels
/// * `fov_h` – horizontal field-of-view in degrees
#[must_use]
pub fn project_to_screen(
    x: f64,
    y: f64,
    z: f64,
    width: u32,
    height: u32,
    fov_h: f64,
) -> Option<(f32, f32)> {
    if z <= 0.0 {
        return None;
    }
    let aspect = f64::from(width) / f64::from(height.max(1));
    let tan_half_h = (fov_h.to_radians() / 2.0).tan();
    let ndc_x = x / (z * tan_half_h);
    let ndc_y = -y / (z * tan_half_h / aspect); // flip Y for screen space

    // Map NDC [-1,1] to pixel coordinates
    let px = ((ndc_x + 1.0) / 2.0 * f64::from(width)) as f32;
    let py = ((ndc_y + 1.0) / 2.0 * f64::from(height)) as f32;
    Some((px, py))
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frustum_new_fov_v_derived() {
        let f = Frustum::new(90.0, 1.0, 0.1, 1000.0);
        // For aspect=1, fov_v should equal fov_h
        assert!((f.fov_v_deg - 90.0).abs() < 1e-6);
    }

    #[test]
    fn test_frustum_fov_v_from_fov_h_16x9() {
        // 16:9 aspect → fov_v < fov_h
        let fov_v = Frustum::fov_v_from_fov_h(90.0, 16.0 / 9.0);
        assert!(fov_v < 90.0);
        assert!(fov_v > 0.0);
    }

    #[test]
    fn test_frustum_fov_v_zero_aspect() {
        let fov_v = Frustum::fov_v_from_fov_h(90.0, 0.0);
        assert_eq!(fov_v, 0.0);
    }

    #[test]
    fn test_frustum_contains_point_center() {
        let f = Frustum::new(90.0, 1.0, 0.1, 1000.0);
        // A point directly ahead at z=10 should be inside.
        assert!(f.contains_point(0.0, 0.0, 10.0));
    }

    #[test]
    fn test_frustum_contains_point_behind() {
        let f = Frustum::new(90.0, 1.0, 0.1, 1000.0);
        assert!(!f.contains_point(0.0, 0.0, -1.0));
    }

    #[test]
    fn test_frustum_contains_point_beyond_far() {
        let f = Frustum::new(90.0, 1.0, 0.1, 100.0);
        assert!(!f.contains_point(0.0, 0.0, 200.0));
    }

    #[test]
    fn test_frustum_contains_point_outside_edge() {
        let f = Frustum::new(90.0, 1.0, 0.1, 1000.0);
        // At z=10 with fov_h=90°, half-width = 10; point at x=15 is outside.
        assert!(!f.contains_point(15.0, 0.0, 10.0));
    }

    #[test]
    fn test_frustum_solid_angle_positive() {
        let f = Frustum::new(60.0, 16.0 / 9.0, 0.1, 1000.0);
        assert!(f.solid_angle() > 0.0);
    }

    #[test]
    fn test_view_frustum_new_identity_pose() {
        let f = Frustum::new(90.0, 1.0, 0.1, 1000.0);
        let vf = ViewFrustum::new(f);
        assert_eq!(vf.pose.x, 0.0);
    }

    #[test]
    fn test_view_frustum_update_pose() {
        let f = Frustum::new(90.0, 1.0, 0.1, 1000.0);
        let mut vf = ViewFrustum::new(f);
        vf.update_pose(TrackedPose::new(5.0, 2.0, 0.0));
        assert_eq!(vf.pose.x, 5.0);
    }

    #[test]
    fn test_view_frustum_world_to_camera() {
        let f = Frustum::new(90.0, 1.0, 0.1, 1000.0);
        let mut vf = ViewFrustum::new(f);
        vf.update_pose(TrackedPose::new(1.0, 2.0, 3.0));
        let (cx, cy, cz) = vf.world_to_camera(4.0, 6.0, 8.0);
        assert!((cx - 3.0).abs() < 1e-10);
        assert!((cy - 4.0).abs() < 1e-10);
        assert!((cz - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_project_to_screen_center() {
        // A point at camera origin projected straight ahead
        let result = project_to_screen(0.0, 0.0, 10.0, 1920, 1080, 90.0);
        assert!(result.is_some());
        let (px, py) = result.expect("should succeed in test");
        // Should land roughly at the screen center (960, 540)
        assert!((px - 960.0).abs() < 1.0);
        assert!((py - 540.0).abs() < 1.0);
    }

    #[test]
    fn test_project_to_screen_behind_camera() {
        let result = project_to_screen(0.0, 0.0, -5.0, 1920, 1080, 90.0);
        assert!(result.is_none());
    }
}
