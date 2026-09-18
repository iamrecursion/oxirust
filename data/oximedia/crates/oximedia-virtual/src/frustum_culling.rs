//! View frustum culling for virtual production scene management.
//!
//! Provides a complete frustum built from camera pan/tilt/roll Euler angles,
//! AABB and sphere culling, and a bulk visibility filter suitable for real-time
//! scene management in LED volume productions.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use crate::camera_tracker::CameraTransform;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Degrees to radians conversion.
#[inline]
fn deg2rad(deg: f32) -> f32 {
    deg * std::f32::consts::PI / 180.0
}

/// Normalise a 3-vector, returning the zero vector if the input is degenerate.
#[inline]
fn normalize(v: (f32, f32, f32)) -> (f32, f32, f32) {
    let len = (v.0 * v.0 + v.1 * v.1 + v.2 * v.2).sqrt();
    if len < 1e-10 {
        return (0.0, 0.0, 0.0);
    }
    (v.0 / len, v.1 / len, v.2 / len)
}

/// Dot product of two 3-vectors.
#[inline]
fn dot(a: (f32, f32, f32), b: (f32, f32, f32)) -> f32 {
    a.0 * b.0 + a.1 * b.1 + a.2 * b.2
}

/// Cross product of two 3-vectors.
#[inline]
fn cross(a: (f32, f32, f32), b: (f32, f32, f32)) -> (f32, f32, f32) {
    (
        a.1 * b.2 - a.2 * b.1,
        a.2 * b.0 - a.0 * b.2,
        a.0 * b.1 - a.1 * b.0,
    )
}

// ---------------------------------------------------------------------------
// CullResult
// ---------------------------------------------------------------------------

/// Result of a frustum culling test.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CullResult {
    /// The object is fully inside the frustum.
    Inside,
    /// The object is fully outside the frustum.
    Outside,
    /// The object straddles one or more frustum planes.
    Intersecting,
}

// ---------------------------------------------------------------------------
// BoundingBox
// ---------------------------------------------------------------------------

/// Axis-aligned bounding box (AABB).
#[derive(Debug, Clone, PartialEq)]
pub struct BoundingBox {
    /// Minimum corner (x, y, z) in metres.
    pub min: (f32, f32, f32),
    /// Maximum corner (x, y, z) in metres.
    pub max: (f32, f32, f32),
}

impl BoundingBox {
    /// Construct a bounding box from its minimum and maximum corners.
    #[must_use]
    pub fn new(min: (f32, f32, f32), max: (f32, f32, f32)) -> Self {
        Self { min, max }
    }

    /// Compute the bounding sphere of this AABB as `(centre, radius)`.
    #[must_use]
    pub fn sphere_bounds(&self) -> ((f32, f32, f32), f32) {
        let cx = (self.min.0 + self.max.0) * 0.5;
        let cy = (self.min.1 + self.max.1) * 0.5;
        let cz = (self.min.2 + self.max.2) * 0.5;
        let dx = self.max.0 - self.min.0;
        let dy = self.max.1 - self.min.1;
        let dz = self.max.2 - self.min.2;
        let radius = (dx * dx + dy * dy + dz * dz).sqrt() * 0.5;
        ((cx, cy, cz), radius)
    }
}

// ---------------------------------------------------------------------------
// ViewFrustum
// ---------------------------------------------------------------------------

/// A perspective view frustum derived from a camera transform and lens params.
#[derive(Debug, Clone)]
pub struct ViewFrustum {
    /// Horizontal field-of-view in degrees.
    pub fov_horizontal_deg: f32,
    /// Vertical field-of-view in degrees.
    pub fov_vertical_deg: f32,
    /// Near clip plane in metres.
    pub near_plane_m: f32,
    /// Far clip plane in metres.
    pub far_plane_m: f32,
    /// Camera world-space position in metres.
    pub camera_pos: (f32, f32, f32),
    /// Camera forward direction (normalised).
    pub camera_dir: (f32, f32, f32),
    /// Camera up vector (normalised).
    pub camera_up: (f32, f32, f32),
}

impl ViewFrustum {
    /// Build a [`ViewFrustum`] from a [`CameraTransform`] and lens parameters.
    ///
    /// The `CameraTransform` positions and orientations are specified in the
    /// following convention:
    /// - `pan_deg` rotates around world Y (yaw)
    /// - `tilt_deg` rotates around world X (pitch)
    /// - `roll_deg` rotates around the Z (forward) axis
    /// - Positions (`x_mm`, `y_mm`, `z_mm`) are converted from mm to metres
    #[must_use]
    pub fn from_transform(
        transform: &CameraTransform,
        fov_h: f32,
        fov_v: f32,
        near: f32,
        far: f32,
    ) -> Self {
        // Convert pan/tilt/roll (in degrees) to a direction vector using
        // ZYX Euler convention (pan=yaw around Y, tilt=pitch around X).
        let pan = deg2rad(transform.pan_deg);
        let tilt = deg2rad(transform.tilt_deg);
        let roll = deg2rad(transform.roll_deg);

        // Forward direction: start with (0, 0, 1), apply tilt, then pan.
        let dir_x = tilt.cos() * pan.sin();
        let dir_y = -tilt.sin();
        let dir_z = tilt.cos() * pan.cos();
        let camera_dir = normalize((dir_x, dir_y, dir_z));

        // Compute up vector with roll applied.
        // Base up = (0, 1, 0) rotated by roll around the forward axis.
        // camera_right = cross(base_up, camera_dir): for forward=(0,0,1), up=(0,1,0)
        // this gives (1,0,0) — the correct +X right direction.
        let base_up = (0.0_f32, 1.0_f32, 0.0_f32);
        let right_pre = normalize(cross(base_up, camera_dir));
        // If camera dir is parallel to world up, fallback to world Z as up.
        let right =
            if right_pre.0.abs() < 1e-5 && right_pre.1.abs() < 1e-5 && right_pre.2.abs() < 1e-5 {
                normalize(cross((0.0_f32, 0.0_f32, 1.0_f32), camera_dir))
            } else {
                right_pre
            };
        // up_no_roll = cross(camera_dir, right): for dir=(0,0,1), right=(1,0,0)
        // gives (0,1,0) — the correct +Y up direction.
        let up_no_roll = normalize(cross(camera_dir, right));
        // Apply roll: rotate up vector around forward axis.
        let cos_roll = roll.cos();
        let sin_roll = roll.sin();
        let camera_up = (
            cos_roll * up_no_roll.0 + sin_roll * right.0,
            cos_roll * up_no_roll.1 + sin_roll * right.1,
            cos_roll * up_no_roll.2 + sin_roll * right.2,
        );

        Self {
            fov_horizontal_deg: fov_h,
            fov_vertical_deg: fov_v,
            near_plane_m: near,
            far_plane_m: far,
            camera_pos: (
                transform.x_mm / 1000.0,
                transform.y_mm / 1000.0,
                transform.z_mm / 1000.0,
            ),
            camera_dir,
            camera_up: normalize(camera_up),
        }
    }

    /// Compute the six frustum planes as `(nx, ny, nz, d)` tuples.
    ///
    /// Plane equation: `n · p + d = 0`.  A point is on the positive (inside)
    /// side when `n · p + d > 0`.
    ///
    /// Planes are returned in order: [near, far, left, right, bottom, top].
    #[must_use]
    pub fn planes(&self) -> [(f32, f32, f32, f32); 6] {
        let pos = self.camera_pos;
        let dir = normalize(self.camera_dir);
        let up = normalize(self.camera_up);
        // Camera-right = cross(up, dir) gives a consistent right-hand coordinate
        // system where: right = +X for forward=+Z, up=+Y.
        let right = normalize(cross(up, dir));

        let half_h = deg2rad(self.fov_horizontal_deg * 0.5).tan();
        let half_v = deg2rad(self.fov_vertical_deg * 0.5).tan();

        // Build a plane from an inward normal and any point on that plane.
        // Convention: n·p + d = 0 defines the plane; points on the
        // inside satisfy n·p + d > 0.
        let make_plane =
            |normal: (f32, f32, f32), point: (f32, f32, f32)| -> (f32, f32, f32, f32) {
                let n = normalize(normal);
                let d = -(n.0 * point.0 + n.1 * point.1 + n.2 * point.2);
                (n.0, n.1, n.2, d)
            };

        // Near plane: inward normal = +dir, positioned at camera + dir * near.
        let near_pt = (
            pos.0 + dir.0 * self.near_plane_m,
            pos.1 + dir.1 * self.near_plane_m,
            pos.2 + dir.2 * self.near_plane_m,
        );
        let near_plane = make_plane(dir, near_pt);

        // Far plane: inward normal = -dir, positioned at camera + dir * far.
        let far_pt = (
            pos.0 + dir.0 * self.far_plane_m,
            pos.1 + dir.1 * self.far_plane_m,
            pos.2 + dir.2 * self.far_plane_m,
        );
        let far_plane = make_plane((-dir.0, -dir.1, -dir.2), far_pt);

        // Left frustum plane.
        // Edge ray along the left border of the frustum: dir - right * tan(fov_h/2).
        // Inward normal = cross(up, left_edge) — points toward +right, into the volume.
        let left_edge = normalize((
            dir.0 - right.0 * half_h,
            dir.1 - right.1 * half_h,
            dir.2 - right.2 * half_h,
        ));
        let left_normal = normalize(cross(up, left_edge));
        let left_plane = make_plane(left_normal, pos);

        // Right frustum plane.
        // Edge ray: dir + right * tan(fov_h/2).
        // Inward normal = cross(right_edge, up) — points toward -right, into the volume.
        let right_edge = normalize((
            dir.0 + right.0 * half_h,
            dir.1 + right.1 * half_h,
            dir.2 + right.2 * half_h,
        ));
        let right_normal = normalize(cross(right_edge, up));
        let right_plane = make_plane(right_normal, pos);

        // Bottom frustum plane.
        // Edge ray: dir - up * tan(fov_v/2).
        // Inward normal = cross(bottom_edge, right) — points toward +up, into the volume.
        let bottom_edge = normalize((
            dir.0 - up.0 * half_v,
            dir.1 - up.1 * half_v,
            dir.2 - up.2 * half_v,
        ));
        let bottom_normal = normalize(cross(bottom_edge, right));
        let bottom_plane = make_plane(bottom_normal, pos);

        // Top frustum plane.
        // Edge ray: dir + up * tan(fov_v/2).
        // Inward normal = cross(right, top_edge) — points toward -up, into the volume.
        let top_edge = normalize((
            dir.0 + up.0 * half_v,
            dir.1 + up.1 * half_v,
            dir.2 + up.2 * half_v,
        ));
        let top_normal = normalize(cross(right, top_edge));
        let top_plane = make_plane(top_normal, pos);

        [
            near_plane,
            far_plane,
            left_plane,
            right_plane,
            bottom_plane,
            top_plane,
        ]
    }

    /// Test a sphere (centre + radius) against the view frustum.
    #[must_use]
    pub fn cull_sphere(&self, center: (f32, f32, f32), radius: f32) -> CullResult {
        let planes = self.planes();
        let mut intersecting = false;

        for (nx, ny, nz, d) in &planes {
            let dist = nx * center.0 + ny * center.1 + nz * center.2 + d;
            if dist < -radius {
                return CullResult::Outside;
            }
            if dist < radius {
                intersecting = true;
            }
        }

        if intersecting {
            CullResult::Intersecting
        } else {
            CullResult::Inside
        }
    }

    /// Test an AABB against the view frustum.
    ///
    /// Uses the p-vertex / n-vertex optimisation per plane.
    #[must_use]
    pub fn cull_box(&self, bb: &BoundingBox) -> CullResult {
        let planes = self.planes();
        let mut result = CullResult::Inside;

        for (nx, ny, nz, d) in &planes {
            // p-vertex: the corner most in the direction of the plane normal.
            let px = if *nx >= 0.0 { bb.max.0 } else { bb.min.0 };
            let py = if *ny >= 0.0 { bb.max.1 } else { bb.min.1 };
            let pz = if *nz >= 0.0 { bb.max.2 } else { bb.min.2 };

            // n-vertex: opposite corner.
            let qx = if *nx >= 0.0 { bb.min.0 } else { bb.max.0 };
            let qy = if *ny >= 0.0 { bb.min.1 } else { bb.max.1 };
            let qz = if *nz >= 0.0 { bb.min.2 } else { bb.max.2 };

            let p_dist = nx * px + ny * py + nz * pz + d;
            let n_dist = nx * qx + ny * qy + nz * qz + d;

            if p_dist < 0.0 {
                // Even the most favourable corner is outside → fully outside.
                return CullResult::Outside;
            }
            if n_dist < 0.0 {
                result = CullResult::Intersecting;
            }
        }

        result
    }

    /// Filter a slice of `(name, BoundingBox)` pairs and return the names of
    /// objects that are either `Inside` or `Intersecting` the frustum.
    #[must_use]
    pub fn visible_objects<'a>(&self, objects: &'a [(String, BoundingBox)]) -> Vec<&'a str> {
        objects
            .iter()
            .filter_map(|(name, bb)| {
                let result = self.cull_box(bb);
                if result != CullResult::Outside {
                    Some(name.as_str())
                } else {
                    None
                }
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_frustum() -> ViewFrustum {
        let t = CameraTransform::identity();
        ViewFrustum::from_transform(&t, 90.0, 60.0, 0.1, 1000.0)
    }

    #[test]
    fn test_bounding_box_new() {
        let bb = BoundingBox::new((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0));
        assert_eq!(bb.min, (-1.0, -1.0, -1.0));
        assert_eq!(bb.max, (1.0, 1.0, 1.0));
    }

    #[test]
    fn test_bounding_box_sphere_bounds_unit_cube() {
        let bb = BoundingBox::new((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0));
        let ((cx, cy, cz), radius) = bb.sphere_bounds();
        assert!((cx).abs() < 1e-5);
        assert!((cy).abs() < 1e-5);
        assert!((cz).abs() < 1e-5);
        // diagonal/2 = sqrt(3) * 2 / 2 ≈ 1.732
        assert!((radius - 3_f32.sqrt()).abs() < 1e-4);
    }

    #[test]
    fn test_view_frustum_from_transform_identity_dir() {
        let f = identity_frustum();
        // Identity camera looks along +Z, so camera_dir.z should be ~1.
        assert!(f.camera_dir.2 > 0.8, "camera should look roughly along +Z");
    }

    #[test]
    fn test_view_frustum_from_transform_position_conversion() {
        let mut t = CameraTransform::identity();
        t.x_mm = 1000.0; // 1 m
        t.y_mm = 2000.0; // 2 m
        let f = ViewFrustum::from_transform(&t, 90.0, 60.0, 0.1, 1000.0);
        assert!((f.camera_pos.0 - 1.0).abs() < 1e-4);
        assert!((f.camera_pos.1 - 2.0).abs() < 1e-4);
    }

    #[test]
    fn test_view_frustum_planes_returns_six() {
        let f = identity_frustum();
        let planes = f.planes();
        assert_eq!(planes.len(), 6);
    }

    #[test]
    fn test_cull_sphere_inside() {
        let f = identity_frustum();
        // Sphere dead-centre ahead of camera (camera at origin, looking +Z).
        let result = f.cull_sphere((0.0, 0.0, 50.0), 1.0);
        assert_ne!(
            result,
            CullResult::Outside,
            "sphere at (0,0,50) should not be outside"
        );
    }

    #[test]
    fn test_cull_sphere_behind_camera() {
        let f = identity_frustum();
        // Far behind the camera on -Z.
        let result = f.cull_sphere((0.0, 0.0, -100.0), 0.5);
        assert_eq!(result, CullResult::Outside);
    }

    #[test]
    fn test_cull_sphere_beyond_far() {
        let f = identity_frustum();
        // Beyond the 1000 m far plane.
        let result = f.cull_sphere((0.0, 0.0, 2000.0), 0.5);
        assert_eq!(result, CullResult::Outside);
    }

    #[test]
    fn test_cull_box_inside() {
        let f = identity_frustum();
        let bb = BoundingBox::new((-0.5, -0.5, 5.0), (0.5, 0.5, 10.0));
        let result = f.cull_box(&bb);
        assert_ne!(
            result,
            CullResult::Outside,
            "box in front should not be outside"
        );
    }

    #[test]
    fn test_cull_box_behind() {
        let f = identity_frustum();
        let bb = BoundingBox::new((-1.0, -1.0, -200.0), (1.0, 1.0, -100.0));
        let result = f.cull_box(&bb);
        assert_eq!(result, CullResult::Outside);
    }

    #[test]
    fn test_visible_objects_filters_correctly() {
        let f = identity_frustum();
        let objects = vec![
            (
                "visible".to_owned(),
                BoundingBox::new((-1.0, -1.0, 5.0), (1.0, 1.0, 10.0)),
            ),
            (
                "behind".to_owned(),
                BoundingBox::new((-1.0, -1.0, -50.0), (1.0, 1.0, -10.0)),
            ),
        ];
        let visible = f.visible_objects(&objects);
        assert!(visible.contains(&"visible"), "visible object should appear");
        assert!(
            !visible.contains(&"behind"),
            "behind object should be culled"
        );
    }

    #[test]
    fn test_visible_objects_empty_scene() {
        let f = identity_frustum();
        let objects: Vec<(String, BoundingBox)> = vec![];
        let visible = f.visible_objects(&objects);
        assert!(visible.is_empty());
    }
}
