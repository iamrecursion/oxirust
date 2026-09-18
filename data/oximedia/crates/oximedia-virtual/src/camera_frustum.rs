//! Camera frustum types for virtual production scene culling and projection.
//!
//! Provides a frustum definition derived from physical camera parameters
//! (focal length / sensor size), six-plane frustum culling, and simple
//! viewport mapping.  This module is a self-contained, pure-Rust implementation
//! that avoids any external linear-algebra dependency.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// CameraFrustum
// ---------------------------------------------------------------------------

/// A perspective frustum derived from camera intrinsics.
#[derive(Debug, Clone, PartialEq)]
pub struct CameraFrustum {
    /// Horizontal field of view in degrees.
    pub fov_h_deg: f32,
    /// Vertical field of view in degrees.
    pub fov_v_deg: f32,
    /// Near clip plane distance in metres.
    pub near_m: f32,
    /// Far clip plane distance in metres.
    pub far_m: f32,
}

impl CameraFrustum {
    /// Create a frustum directly from FOV values.
    #[must_use]
    pub fn new(fov_h_deg: f32, fov_v_deg: f32, near_m: f32, far_m: f32) -> Self {
        Self {
            fov_h_deg,
            fov_v_deg,
            near_m,
            far_m,
        }
    }

    /// Derive a frustum from physical camera / lens parameters.
    ///
    /// # Arguments
    /// * `focal_mm`    – Effective focal length in millimetres.
    /// * `sensor_w_mm` – Sensor width in millimetres.
    /// * `sensor_h_mm` – Sensor height in millimetres.
    #[must_use]
    pub fn from_focal_length(focal_mm: f32, sensor_w_mm: f32, sensor_h_mm: f32) -> Self {
        let fov_h = 2.0 * ((sensor_w_mm / (2.0 * focal_mm)).atan()) * (180.0 / PI);
        let fov_v = 2.0 * ((sensor_h_mm / (2.0 * focal_mm)).atan()) * (180.0 / PI);
        Self {
            fov_h_deg: fov_h,
            fov_v_deg: fov_v,
            near_m: 0.1,
            far_m: 1000.0,
        }
    }

    /// Aspect ratio (width / height) implied by the FOV values.
    #[must_use]
    pub fn aspect_ratio(&self) -> f32 {
        let tan_h = (self.fov_h_deg.to_radians() / 2.0).tan();
        let tan_v = (self.fov_v_deg.to_radians() / 2.0).tan();
        if tan_v == 0.0 {
            return 1.0;
        }
        tan_h / tan_v
    }
}

// ---------------------------------------------------------------------------
// FrustumPlane
// ---------------------------------------------------------------------------

/// A half-space defined by a plane equation: `dot(normal, point) + distance >= 0` → in front.
#[derive(Debug, Clone, PartialEq)]
pub struct FrustumPlane {
    /// Outward-facing unit normal of the plane.
    pub normal: [f32; 3],
    /// Signed distance from the origin along the normal.
    pub distance: f32,
}

impl FrustumPlane {
    /// Create a new frustum plane.
    #[must_use]
    pub fn new(normal: [f32; 3], distance: f32) -> Self {
        Self { normal, distance }
    }

    /// Signed distance of a point from the plane.
    ///
    /// Positive values indicate the point is on the side the normal points toward.
    #[must_use]
    pub fn point_distance(&self, p: [f32; 3]) -> f32 {
        self.normal[0] * p[0] + self.normal[1] * p[1] + self.normal[2] * p[2] + self.distance
    }

    /// Returns `true` if the point is on the front (positive) side of the plane.
    #[must_use]
    pub fn is_point_in_front(&self, p: [f32; 3]) -> bool {
        self.point_distance(p) >= 0.0
    }
}

// ---------------------------------------------------------------------------
// FrustumCuller
// ---------------------------------------------------------------------------

/// Six-plane frustum culler.
///
/// All tests assume the camera looks along the **positive Z axis** in camera
/// space, with Y pointing up and X pointing right.
#[derive(Debug, Clone)]
pub struct FrustumCuller {
    /// The six frustum planes (left, right, bottom, top, near, far).
    pub planes: Vec<FrustumPlane>,
}

impl FrustumCuller {
    /// Build the six frustum planes from a [`CameraFrustum`].
    ///
    /// All planes have their normals pointing **inward** (toward the frustum
    /// interior) so that a positive `point_distance` means inside.
    #[must_use]
    pub fn from_frustum(frustum: &CameraFrustum) -> Self {
        let half_h = (frustum.fov_h_deg.to_radians() / 2.0).tan();
        let half_v = (frustum.fov_v_deg.to_radians() / 2.0).tan();

        // We build planes in camera space where camera looks along +Z.
        // For each side plane the normal is derived from the edge direction.
        let norm = |x: f32, y: f32, z: f32| -> [f32; 3] {
            let len = (x * x + y * y + z * z).sqrt();
            [x / len, y / len, z / len]
        };

        let planes = vec![
            // Left plane:  normal points right (+X direction, rotated inward)
            FrustumPlane::new(norm(1.0, 0.0, half_h), 0.0),
            // Right plane: normal points left
            FrustumPlane::new(norm(-1.0, 0.0, half_h), 0.0),
            // Bottom plane: normal points up
            FrustumPlane::new(norm(0.0, 1.0, half_v), 0.0),
            // Top plane: normal points down
            FrustumPlane::new(norm(0.0, -1.0, half_v), 0.0),
            // Near plane: normal points forward (+Z)
            FrustumPlane::new([0.0, 0.0, 1.0], -frustum.near_m),
            // Far plane: normal points backward (-Z)
            FrustumPlane::new([0.0, 0.0, -1.0], frustum.far_m),
        ];

        Self { planes }
    }

    /// Returns `true` if the point is inside all six frustum planes.
    #[must_use]
    pub fn point_inside(&self, p: [f32; 3]) -> bool {
        self.planes.iter().all(|plane| plane.is_point_in_front(p))
    }

    /// Returns `true` if a sphere (centre + radius) intersects or is inside the frustum.
    ///
    /// A sphere is considered outside only when it is entirely on the wrong side
    /// of at least one plane (signed distance < −radius).
    #[must_use]
    pub fn sphere_inside(&self, center: [f32; 3], radius: f32) -> bool {
        self.planes
            .iter()
            .all(|plane| plane.point_distance(center) >= -radius)
    }
}

// ---------------------------------------------------------------------------
// ViewportMapping
// ---------------------------------------------------------------------------

/// Maps 3D camera-space points onto 2D pixel coordinates.
#[derive(Debug, Clone)]
pub struct ViewportMapping {
    /// The camera frustum.
    pub frustum: CameraFrustum,
    /// Output image width in pixels.
    pub width_px: u32,
    /// Output image height in pixels.
    pub height_px: u32,
}

impl ViewportMapping {
    /// Create a new viewport mapping.
    #[must_use]
    pub fn new(frustum: CameraFrustum, width_px: u32, height_px: u32) -> Self {
        Self {
            frustum,
            width_px,
            height_px,
        }
    }

    /// Project a camera-space world point onto the viewport.
    ///
    /// Returns `None` if the point is behind or exactly at the camera origin
    /// (z ≤ 0), or outside the frustum's near/far range.
    ///
    /// The returned coordinates are clamped to `[0, width_px)` × `[0, height_px)`.
    #[must_use]
    pub fn project_point(&self, world: [f32; 3]) -> Option<(u32, u32)> {
        let z = world[2];
        if z <= 0.0 || z < self.frustum.near_m || z > self.frustum.far_m {
            return None;
        }
        let tan_h = (self.frustum.fov_h_deg.to_radians() / 2.0).tan();
        let tan_v = (self.frustum.fov_v_deg.to_radians() / 2.0).tan();

        // Normalised device coordinates in [-1, 1].
        let ndc_x = world[0] / (z * tan_h);
        let ndc_y = -world[1] / (z * tan_v); // flip Y: screen Y grows downward

        if !(-1.0..=1.0).contains(&ndc_x) || !(-1.0..=1.0).contains(&ndc_y) {
            return None;
        }

        let px = ((ndc_x + 1.0) / 2.0 * self.width_px as f32) as u32;
        let py = ((ndc_y + 1.0) / 2.0 * self.height_px as f32) as u32;
        Some((
            px.min(self.width_px.saturating_sub(1)),
            py.min(self.height_px.saturating_sub(1)),
        ))
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn basic_frustum() -> CameraFrustum {
        CameraFrustum::new(90.0, 60.0, 0.1, 100.0)
    }

    #[test]
    fn test_camera_frustum_new() {
        let f = basic_frustum();
        assert_eq!(f.fov_h_deg, 90.0);
        assert_eq!(f.fov_v_deg, 60.0);
        assert_eq!(f.near_m, 0.1);
        assert_eq!(f.far_m, 100.0);
    }

    #[test]
    fn test_from_focal_length_produces_positive_fov() {
        // Full-frame 50 mm lens: sensor 36×24 mm
        let f = CameraFrustum::from_focal_length(50.0, 36.0, 24.0);
        assert!(f.fov_h_deg > 0.0);
        assert!(f.fov_v_deg > 0.0);
        assert!(f.fov_h_deg > f.fov_v_deg); // landscape sensor
    }

    #[test]
    fn test_from_focal_length_wide_angle() {
        // Wide-angle 14 mm lens: larger FOV
        let wide = CameraFrustum::from_focal_length(14.0, 36.0, 24.0);
        let normal = CameraFrustum::from_focal_length(50.0, 36.0, 24.0);
        assert!(wide.fov_h_deg > normal.fov_h_deg);
    }

    #[test]
    fn test_frustum_plane_point_distance_on_plane() {
        let plane = FrustumPlane::new([1.0, 0.0, 0.0], 0.0);
        assert!((plane.point_distance([0.0, 0.0, 0.0])).abs() < 1e-6);
    }

    #[test]
    fn test_frustum_plane_is_point_in_front() {
        let plane = FrustumPlane::new([1.0, 0.0, 0.0], 0.0);
        assert!(plane.is_point_in_front([1.0, 0.0, 0.0]));
        assert!(!plane.is_point_in_front([-1.0, 0.0, 0.0]));
    }

    #[test]
    fn test_frustum_culler_has_six_planes() {
        let f = basic_frustum();
        let culler = FrustumCuller::from_frustum(&f);
        assert_eq!(culler.planes.len(), 6);
    }

    #[test]
    fn test_frustum_culler_center_point_inside() {
        let f = basic_frustum();
        let culler = FrustumCuller::from_frustum(&f);
        // A point directly ahead, inside near/far range.
        assert!(culler.point_inside([0.0, 0.0, 5.0]));
    }

    #[test]
    fn test_frustum_culler_behind_camera_outside() {
        let f = basic_frustum();
        let culler = FrustumCuller::from_frustum(&f);
        assert!(!culler.point_inside([0.0, 0.0, -1.0]));
    }

    #[test]
    fn test_frustum_culler_beyond_far_outside() {
        let f = basic_frustum();
        let culler = FrustumCuller::from_frustum(&f);
        assert!(!culler.point_inside([0.0, 0.0, 200.0]));
    }

    #[test]
    fn test_frustum_culler_sphere_inside_when_center_inside() {
        let f = basic_frustum();
        let culler = FrustumCuller::from_frustum(&f);
        assert!(culler.sphere_inside([0.0, 0.0, 5.0], 0.5));
    }

    #[test]
    fn test_frustum_culler_sphere_partially_overlapping() {
        let f = basic_frustum();
        let culler = FrustumCuller::from_frustum(&f);
        // A large sphere centred just behind the camera still straddles the near plane.
        assert!(culler.sphere_inside([0.0, 0.0, -0.05_f32], 5.0));
    }

    #[test]
    fn test_viewport_mapping_center_point() {
        let f = CameraFrustum::new(90.0, 90.0, 0.1, 100.0);
        let vm = ViewportMapping::new(f, 1920, 1080);
        // A point directly ahead should map to roughly the screen centre.
        let result = vm.project_point([0.0, 0.0, 10.0]);
        assert!(result.is_some());
        let (px, py) = result.expect("should succeed in test");
        assert!((px as i32 - 960).abs() <= 1);
        assert!((py as i32 - 540).abs() <= 1);
    }

    #[test]
    fn test_viewport_mapping_behind_camera_none() {
        let f = CameraFrustum::new(90.0, 90.0, 0.1, 100.0);
        let vm = ViewportMapping::new(f, 1920, 1080);
        assert!(vm.project_point([0.0, 0.0, -1.0]).is_none());
    }

    #[test]
    fn test_viewport_mapping_beyond_far_none() {
        let f = CameraFrustum::new(90.0, 90.0, 0.1, 100.0);
        let vm = ViewportMapping::new(f, 1920, 1080);
        assert!(vm.project_point([0.0, 0.0, 200.0]).is_none());
    }
}
