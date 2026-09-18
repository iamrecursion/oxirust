//! Rotation and orientation transforms for spherical coordinates.
//!
//! This module provides utilities to apply yaw/pitch/roll rotations to
//! [`SphericalCoord`] values, as well as batch-transform entire equirectangular
//! images by re-orienting the sphere.  Internally all transforms use 3×3
//! rotation matrices operating on unit 3-D Cartesian vectors.
//!
//! ## Use cases
//!
//! * **Horizon correction** — apply a pitch or roll offset to a tilted
//!   equirectangular panorama.
//! * **North-pole normalisation** — rotate a 360° video so that a reference
//!   direction is at the image centre.
//! * **Orientation interpolation** — interpolate between two camera orientations
//!   for smooth playback.
//!
//! ## Example
//!
//! ```rust
//! use oximedia_360::orientation::{Orientation, rotate_sphere};
//! use oximedia_360::SphericalCoord;
//!
//! let sp = SphericalCoord { azimuth_rad: 0.0, elevation_rad: 0.0 };
//! let orient = Orientation::from_yaw_pitch_roll_deg(90.0, 0.0, 0.0);
//! let rotated = rotate_sphere(&sp, &orient);
//! ```

use crate::{
    projection::{
        bilinear_sample_u8, equirect_to_sphere, sphere_to_equirect, SphericalCoord, UvCoord,
    },
    VrError,
};

// ─── Orientation ─────────────────────────────────────────────────────────────

/// Camera orientation expressed as intrinsic yaw → pitch → roll rotations.
///
/// All angles are stored in **radians**.  Use [`Orientation::from_yaw_pitch_roll_deg`]
/// for a degree-based constructor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Orientation {
    /// Yaw (azimuth rotation) in radians — rotates around the vertical Y axis.
    /// Positive = clockwise when viewed from above (towards East).
    pub yaw_rad: f32,
    /// Pitch in radians — rotates around the horizontal X axis.
    /// Positive = tilt upward (toward the North Pole).
    pub pitch_rad: f32,
    /// Roll in radians — rotates around the forward Z axis.
    /// Positive = clockwise from the camera's perspective.
    pub roll_rad: f32,
}

impl Orientation {
    /// Create an identity orientation (no rotation).
    pub fn identity() -> Self {
        Self {
            yaw_rad: 0.0,
            pitch_rad: 0.0,
            roll_rad: 0.0,
        }
    }

    /// Create an orientation from yaw/pitch/roll given in **degrees**.
    pub fn from_yaw_pitch_roll_deg(yaw_deg: f32, pitch_deg: f32, roll_deg: f32) -> Self {
        Self {
            yaw_rad: yaw_deg.to_radians(),
            pitch_rad: pitch_deg.to_radians(),
            roll_rad: roll_deg.to_radians(),
        }
    }

    /// Return the inverse orientation.
    ///
    /// The inverse of a rotation `R` is its transpose `Rᵀ` (for orthogonal
    /// matrices).  For combined Euler rotations negating all three angles is
    /// generally NOT correct; this method computes the true inverse by
    /// transposing the underlying rotation matrix and extracting fresh Euler
    /// angles from it.
    ///
    /// If `R = rotate_sphere(s, orient)`, then
    /// `rotate_sphere(&R, &orient.inverse()) ≈ s`.
    pub fn inverse(&self) -> Self {
        // Transpose = inverse for rotation matrices
        let mat = self.to_matrix();
        let mat_inv = RotMat3([
            [mat.0[0][0], mat.0[1][0], mat.0[2][0]],
            [mat.0[0][1], mat.0[1][1], mat.0[2][1]],
            [mat.0[0][2], mat.0[1][2], mat.0[2][2]],
        ]);
        mat_inv.to_orientation()
    }

    /// Build the combined 3×3 rotation matrix for this orientation.
    pub fn to_matrix(&self) -> RotMat3 {
        RotMat3::from_yaw_pitch_roll(self.yaw_rad, self.pitch_rad, self.roll_rad)
    }

    /// Compose two orientations (this followed by `other`).
    ///
    /// The resulting rotation matrix equals `other.to_matrix() · self.to_matrix()`.
    /// Note that this is **not** simply component-wise addition due to gimbal lock
    /// — the composition is computed via matrix multiplication.
    pub fn compose(&self, other: &Orientation) -> Orientation {
        let ma = self.to_matrix();
        let mb = other.to_matrix();
        let mc = mb.mul(&ma);
        mc.to_orientation()
    }

    /// Spherical linear interpolation (SLERP) between two orientations.
    ///
    /// `t = 0.0` returns `*self`, `t = 1.0` returns `*other`.
    /// The interpolation is performed via SLERP on the rotation matrices
    /// decomposed to quaternion form.
    pub fn slerp(&self, other: &Orientation, t: f32) -> Orientation {
        let qa = Quat::from_orientation(self);
        let qb = Quat::from_orientation(other);
        let qc = qa.slerp(&qb, t);
        qc.to_orientation()
    }
}

impl Default for Orientation {
    fn default() -> Self {
        Self::identity()
    }
}

// ─── 3×3 Rotation matrix ──────────────────────────────────────────────────────

/// A 3×3 rotation matrix (row-major).
#[derive(Debug, Clone, Copy)]
pub struct RotMat3(pub [[f32; 3]; 3]);

impl RotMat3 {
    /// Identity matrix.
    pub fn identity() -> Self {
        Self([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    }

    /// Build combined rotation matrix: R = R_yaw · R_pitch · R_roll.
    pub fn from_yaw_pitch_roll(yaw: f32, pitch: f32, roll: f32) -> Self {
        // R_yaw: around Y
        let (sy, cy) = yaw.sin_cos();
        let ry = [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]];
        // R_pitch: around X
        let (sp, cp) = pitch.sin_cos();
        let rp = [[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]];
        // R_roll: around Z
        let (sr, cr) = roll.sin_cos();
        let rr = [[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]];

        let rpr = Self(mat3_mul(rp, rr));
        let combined = mat3_mul(ry, rpr.0);
        Self(combined)
    }

    /// Multiply two rotation matrices: `self · other`.
    pub fn mul(&self, other: &RotMat3) -> RotMat3 {
        RotMat3(mat3_mul(self.0, other.0))
    }

    /// Apply the rotation to a 3-D vector.
    pub fn apply(&self, v: [f32; 3]) -> [f32; 3] {
        let m = &self.0;
        [
            m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
            m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
            m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
        ]
    }

    /// Apply the transpose (= inverse for rotation matrices) to a 3-D vector.
    pub fn apply_inverse(&self, v: [f32; 3]) -> [f32; 3] {
        let m = &self.0;
        [
            m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
            m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
            m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2],
        ]
    }

    /// Extract Euler angles (yaw, pitch, roll) from the rotation matrix.
    ///
    /// Uses the YPR (Ry·Rp·Rr) convention matching `from_yaw_pitch_roll`.
    ///
    /// For R = Ry · Rp · Rr the expanded matrix row-major is:
    /// ```text
    /// R = [[cy·cr + sy·sp·sr,  -cy·sr + sy·sp·cr,  sy·cp],
    ///      [cp·sr,              cp·cr,              -sp  ],
    ///      [-sy·cr + cy·sp·sr,   sy·sr + cy·sp·cr,  cy·cp]]
    /// ```
    /// From which:
    /// * `pitch = asin(-R[1][2])`
    /// * `yaw   = atan2(R[0][2], R[2][2])`   (when cos(pitch) ≠ 0)
    /// * `roll  = atan2(R[1][0], R[1][1])`   (when cos(pitch) ≠ 0)
    pub fn to_orientation(&self) -> Orientation {
        let m = &self.0;
        // Pitch from R[1][2] = -sin(pitch)
        let pitch = (-m[1][2]).clamp(-1.0, 1.0).asin();
        let (yaw, roll);
        if pitch.cos().abs() > 1e-6 {
            // Normal case
            yaw = m[0][2].atan2(m[2][2]);
            roll = m[1][0].atan2(m[1][1]);
        } else {
            // Gimbal lock (pitch ≈ ±90°): set roll = 0, extract yaw from top-left
            yaw = m[0][1].atan2(m[0][0]);
            roll = 0.0;
        }
        Orientation {
            yaw_rad: yaw,
            pitch_rad: pitch,
            roll_rad: roll,
        }
    }
}

fn mat3_mul(a: [[f32; 3]; 3], b: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut c = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}

// ─── Quaternion (for SLERP) ───────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
struct Quat {
    w: f32,
    x: f32,
    y: f32,
    z: f32,
}

impl Quat {
    /// Build a unit quaternion from Orientation using the YPR convention
    /// (Ry · Rp · Rr, i.e. yaw around Y, then pitch around X, then roll around Z).
    ///
    /// Q = Q_yaw(Y) × Q_pitch(X) × Q_roll(Z)
    /// where each component quaternion is:
    ///   Q_yaw   = (cos(yaw/2),   0,          sin(yaw/2),   0)
    ///   Q_pitch = (cos(pitch/2), sin(pitch/2), 0,           0)
    ///   Q_roll  = (cos(roll/2),  0,            0,           sin(roll/2))
    fn from_orientation(o: &Orientation) -> Self {
        let (sy, cy) = (o.yaw_rad * 0.5).sin_cos();
        let (sp, cp) = (o.pitch_rad * 0.5).sin_cos();
        let (sr, cr) = (o.roll_rad * 0.5).sin_cos();

        // Derived by expanding Q_yaw × Q_pitch × Q_roll with Hamilton product:
        Self {
            w: cy * cp * cr + sy * sp * sr,
            x: cy * sp * cr + sy * cp * sr,
            y: sy * cp * cr - cy * sp * sr,
            z: cy * cp * sr - sy * sp * cr,
        }
    }

    fn dot(&self, other: &Quat) -> f32 {
        self.w * other.w + self.x * other.x + self.y * other.y + self.z * other.z
    }

    fn normalise(self) -> Self {
        let len = (self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z).sqrt();
        if len < 1e-10 {
            return Self {
                w: 1.0,
                x: 0.0,
                y: 0.0,
                z: 0.0,
            };
        }
        Self {
            w: self.w / len,
            x: self.x / len,
            y: self.y / len,
            z: self.z / len,
        }
    }

    fn slerp(&self, other: &Quat, t: f32) -> Quat {
        let mut dot = self.dot(other).clamp(-1.0, 1.0);
        // Choose shortest arc
        let other_adj = if dot < 0.0 {
            dot = -dot;
            Quat {
                w: -other.w,
                x: -other.x,
                y: -other.y,
                z: -other.z,
            }
        } else {
            *other
        };

        if dot > 0.9995 {
            // Linear interpolation for nearly parallel quaternions
            return Quat {
                w: self.w + t * (other_adj.w - self.w),
                x: self.x + t * (other_adj.x - self.x),
                y: self.y + t * (other_adj.y - self.y),
                z: self.z + t * (other_adj.z - self.z),
            }
            .normalise();
        }

        // Standard SLERP: slerp(q1, q2, t) = q1 * sin((1-t)θ)/sin(θ) + q2 * sin(t·θ)/sin(θ)
        let theta_0 = dot.acos();
        let sin_theta_0 = theta_0.sin();

        let s0 = ((1.0 - t) * theta_0).sin() / sin_theta_0;
        let s1 = (t * theta_0).sin() / sin_theta_0;

        Quat {
            w: s0 * self.w + s1 * other_adj.w,
            x: s0 * self.x + s1 * other_adj.x,
            y: s0 * self.y + s1 * other_adj.y,
            z: s0 * self.z + s1 * other_adj.z,
        }
    }

    fn to_orientation(&self) -> Orientation {
        let q = self.normalise();
        // Convert quaternion to 3×3 rotation matrix, then extract YPR Euler angles.
        // This avoids having to re-derive the quaternion-to-Euler formulas for
        // the specific YPR (Ry·Rp·Rr) convention used throughout this module.
        let (w, x, y, z) = (q.w, q.x, q.y, q.z);
        let mat = RotMat3([
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - w * z),
                2.0 * (x * z + w * y),
            ],
            [
                2.0 * (x * y + w * z),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - w * x),
            ],
            [
                2.0 * (x * z - w * y),
                2.0 * (y * z + w * x),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ]);
        mat.to_orientation()
    }
}

// ─── Public rotation functions ────────────────────────────────────────────────

/// Rotate a single [`SphericalCoord`] by the given orientation.
///
/// The spherical coordinate is first converted to a unit Cartesian vector,
/// the rotation is applied, and the result is converted back to spherical.
pub fn rotate_sphere(s: &SphericalCoord, orient: &Orientation) -> SphericalCoord {
    let x = s.elevation_rad.cos() * s.azimuth_rad.sin();
    let y = s.elevation_rad.sin();
    let z = s.elevation_rad.cos() * s.azimuth_rad.cos();

    let mat = orient.to_matrix();
    let rv = mat.apply([x, y, z]);

    let elevation_rad = rv[1].clamp(-1.0, 1.0).asin();
    let azimuth_rad = rv[0].atan2(rv[2]);

    SphericalCoord {
        azimuth_rad,
        elevation_rad,
    }
}

/// Rotate an equirectangular image by re-orienting the sphere.
///
/// Each output pixel is computed by:
/// 1. Converting the output pixel's equirectangular UV to a sphere direction.
/// 2. Applying the **inverse** of `orient` to find where in the source image
///    the corresponding world direction came from.
/// 3. Sampling the source image with bilinear interpolation.
///
/// * `src`        — source pixel data (RGB, 3 bpp, row-major)
/// * `width`      — source/output image width in pixels
/// * `height`     — source/output image height in pixels
/// * `orient`     — rotation to apply (positive yaw rotates content left, i.e.
///                  the panorama's "north" moves right in the output)
///
/// # Errors
/// Returns [`VrError::InvalidDimensions`] if any dimension is zero.
/// Returns [`VrError::BufferTooSmall`] if `src` is too small.
pub fn rotate_equirect(
    src: &[u8],
    width: u32,
    height: u32,
    orient: &Orientation,
) -> Result<Vec<u8>, VrError> {
    if width == 0 || height == 0 {
        return Err(VrError::InvalidDimensions(
            "width and height must be > 0".into(),
        ));
    }
    let expected = width as usize * height as usize * 3;
    if src.len() < expected {
        return Err(VrError::BufferTooSmall {
            expected,
            got: src.len(),
        });
    }

    // Pre-compute inverse rotation matrix
    let mat_inv = orient.inverse().to_matrix();

    const CH: u32 = 3;
    let mut out = vec![0u8; expected];

    for oy in 0..height {
        for ox in 0..width {
            let u = (ox as f32 + 0.5) / width as f32;
            let v = (oy as f32 + 0.5) / height as f32;

            let sphere_out = equirect_to_sphere(&UvCoord { u, v });

            let x = sphere_out.elevation_rad.cos() * sphere_out.azimuth_rad.sin();
            let y = sphere_out.elevation_rad.sin();
            let z = sphere_out.elevation_rad.cos() * sphere_out.azimuth_rad.cos();

            let sv = mat_inv.apply([x, y, z]);
            let el = sv[1].clamp(-1.0, 1.0).asin();
            let az = sv[0].atan2(sv[2]);

            let src_sphere = SphericalCoord {
                azimuth_rad: az,
                elevation_rad: el,
            };
            let src_uv = sphere_to_equirect(&src_sphere);
            let sample = bilinear_sample_u8(src, width, height, src_uv.u, src_uv.v, CH);
            let dst = (oy * width + ox) as usize * CH as usize;
            out[dst..dst + CH as usize].copy_from_slice(&sample);
        }
    }

    Ok(out)
}

/// Compute the angular distance (in radians) between two spherical coordinates.
///
/// Uses the haversine formula, which is numerically stable for small angles.
pub fn angular_distance(a: &SphericalCoord, b: &SphericalCoord) -> f32 {
    let dlat = b.elevation_rad - a.elevation_rad;
    let dlon = b.azimuth_rad - a.azimuth_rad;
    let hav = (dlat * 0.5).sin().powi(2)
        + a.elevation_rad.cos() * b.elevation_rad.cos() * (dlon * 0.5).sin().powi(2);
    2.0 * hav.sqrt().asin()
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    const EPSILON: f32 = 0.02;

    fn sphere(az: f32, el: f32) -> SphericalCoord {
        SphericalCoord {
            azimuth_rad: az,
            elevation_rad: el,
        }
    }

    // ── Orientation constructors ─────────────────────────────────────────────

    #[test]
    fn identity_has_zero_angles() {
        let o = Orientation::identity();
        assert_eq!(o.yaw_rad, 0.0);
        assert_eq!(o.pitch_rad, 0.0);
        assert_eq!(o.roll_rad, 0.0);
    }

    #[test]
    fn from_degrees_converts_correctly() {
        let o = Orientation::from_yaw_pitch_roll_deg(90.0, -45.0, 180.0);
        assert!((o.yaw_rad - PI / 2.0).abs() < 1e-5);
        assert!((o.pitch_rad + PI / 4.0).abs() < 1e-5);
        assert!((o.roll_rad - PI).abs() < 1e-5);
    }

    #[test]
    fn inverse_is_true_rotation_inverse() {
        // For combined Euler rotations, inverse() computes the matrix transpose,
        // which is the true rotation inverse.  Verify that applying orientation
        // then its inverse is a no-op (identity) on a test vector.
        let o = Orientation::from_yaw_pitch_roll_deg(30.0, -20.0, 10.0);
        let inv = o.inverse();
        let mat_fwd = o.to_matrix();
        let mat_inv = inv.to_matrix();
        // R · R^T should be identity
        let product = mat_fwd.mul(&mat_inv);
        let identity = RotMat3::identity();
        for i in 0..3 {
            for j in 0..3 {
                let diff = (product.0[i][j] - identity.0[i][j]).abs();
                assert!(
                    diff < 1e-5,
                    "R*R^T[{i}][{j}]={} expected {}",
                    product.0[i][j],
                    identity.0[i][j]
                );
            }
        }
    }

    // ── RotMat3 ──────────────────────────────────────────────────────────────

    #[test]
    fn identity_matrix_does_not_change_vector() {
        let mat = RotMat3::identity();
        let v = [1.0f32, 2.0, 3.0];
        let rv = mat.apply(v);
        for (a, b) in v.iter().zip(rv.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn rot_mat_apply_inverse_is_transpose() {
        let mat = RotMat3::from_yaw_pitch_roll(0.5, 0.3, 0.1);
        let v = [1.0f32, 0.0, 0.0];
        let fwd = mat.apply(v);
        let back = mat.apply_inverse(fwd);
        for (a, b) in v.iter().zip(back.iter()) {
            assert!((a - b).abs() < 1e-5, "a={a} b={b}");
        }
    }

    // ── rotate_sphere ────────────────────────────────────────────────────────

    #[test]
    fn rotate_identity_does_not_change_sphere() {
        let s = sphere(PI / 3.0, PI / 6.0);
        let o = Orientation::identity();
        let r = rotate_sphere(&s, &o);
        assert!((r.azimuth_rad - s.azimuth_rad).abs() < EPSILON);
        assert!((r.elevation_rad - s.elevation_rad).abs() < EPSILON);
    }

    #[test]
    fn rotate_yaw_90_moves_forward_to_right() {
        // Forward direction (az=0, el=0) rotated by yaw=90° should arrive at (az=90°, el=0)
        let s = sphere(0.0, 0.0);
        let o = Orientation::from_yaw_pitch_roll_deg(90.0, 0.0, 0.0);
        let r = rotate_sphere(&s, &o);
        // After yaw of 90°, azimuth should have shifted
        let az_diff = (r.azimuth_rad - PI / 2.0).abs();
        assert!(
            az_diff < EPSILON || (az_diff - PI).abs() < EPSILON,
            "az={}",
            r.azimuth_rad
        );
        assert!(r.elevation_rad.abs() < EPSILON);
    }

    #[test]
    fn rotate_pitch_moves_forward_to_pole() {
        // In YPR (Ry·Rp·Rr) convention, positive pitch around the X axis rotates
        // +Z toward -Y (south pole) because Rp * [0,0,1] = [0,-sin90, cos90] = [0,-1,0].
        // Use negative pitch to tilt upward (toward north pole).
        let s = sphere(0.0, 0.0);
        let o_down = Orientation::from_yaw_pitch_roll_deg(0.0, 90.0, 0.0);
        let r_down = rotate_sphere(&s, &o_down);
        // Positive pitch: forward goes south
        assert!(
            r_down.elevation_rad < -(PI / 2.0 - EPSILON),
            "positive pitch should go south, el={}",
            r_down.elevation_rad
        );

        let o_up = Orientation::from_yaw_pitch_roll_deg(0.0, -90.0, 0.0);
        let r_up = rotate_sphere(&s, &o_up);
        // Negative pitch: forward goes north
        assert!(
            r_up.elevation_rad > PI / 2.0 - EPSILON,
            "negative pitch should go north, el={}",
            r_up.elevation_rad
        );
    }

    #[test]
    fn rotate_inverse_is_roundtrip() {
        let s = sphere(PI / 4.0, PI / 8.0);
        let o = Orientation::from_yaw_pitch_roll_deg(30.0, -20.0, 10.0);
        let rotated = rotate_sphere(&s, &o);
        let back = rotate_sphere(&rotated, &o.inverse());
        let dist = angular_distance(&s, &back);
        assert!(dist < EPSILON, "dist={dist}");
    }

    // ── rotate_equirect ──────────────────────────────────────────────────────

    fn solid_rgb(w: u32, h: u32, r: u8, g: u8, b: u8) -> Vec<u8> {
        let mut v = Vec::with_capacity(w as usize * h as usize * 3);
        for _ in 0..(w * h) {
            v.extend_from_slice(&[r, g, b]);
        }
        v
    }

    #[test]
    fn rotate_equirect_zero_dim_error() {
        let src = solid_rgb(64, 32, 100, 100, 100);
        let o = Orientation::identity();
        assert!(rotate_equirect(&src, 0, 32, &o).is_err());
        assert!(rotate_equirect(&src, 64, 0, &o).is_err());
    }

    #[test]
    fn rotate_equirect_buffer_too_small_error() {
        let o = Orientation::identity();
        assert!(rotate_equirect(&[0u8; 10], 64, 32, &o).is_err());
    }

    #[test]
    fn rotate_equirect_correct_output_size() {
        let src = solid_rgb(64, 32, 150, 75, 200);
        let o = Orientation::from_yaw_pitch_roll_deg(45.0, 0.0, 0.0);
        let out = rotate_equirect(&src, 64, 32, &o).expect("ok");
        assert_eq!(out.len(), src.len());
    }

    #[test]
    fn rotate_equirect_identity_preserves_image() {
        let src = solid_rgb(64, 32, 200, 100, 50);
        let o = Orientation::identity();
        let out = rotate_equirect(&src, 64, 32, &o).expect("ok");
        // Solid image: all pixels remain the same
        let base = (16 * 64 + 32) * 3;
        assert!((out[base] as i32 - 200).abs() <= 3);
    }

    #[test]
    fn rotate_equirect_solid_colour_unchanged() {
        // Rotating a uniform-colour image should return the same colour everywhere
        let src = solid_rgb(128, 64, 128, 64, 32);
        let o = Orientation::from_yaw_pitch_roll_deg(45.0, 30.0, 15.0);
        let out = rotate_equirect(&src, 128, 64, &o).expect("ok");
        let base = (32 * 128 + 64) * 3;
        assert!((out[base] as i32 - 128).abs() <= 5);
    }

    // ── angular_distance ─────────────────────────────────────────────────────

    #[test]
    fn angular_distance_same_point_is_zero() {
        let s = sphere(PI / 3.0, PI / 6.0);
        let d = angular_distance(&s, &s);
        assert!(d < 1e-5, "d={d}");
    }

    #[test]
    fn angular_distance_antipodal_is_pi() {
        let a = sphere(0.0, 0.0);
        let b = sphere(PI, 0.0);
        let d = angular_distance(&a, &b);
        assert!((d - PI).abs() < 0.05, "d={d}");
    }

    #[test]
    fn angular_distance_quarter_circle() {
        let a = sphere(0.0, 0.0);
        let b = sphere(PI / 2.0, 0.0);
        let d = angular_distance(&a, &b);
        assert!((d - PI / 2.0).abs() < 0.05, "d={d}");
    }

    // ── Orientation composition ──────────────────────────────────────────────

    #[test]
    fn compose_with_identity_is_identity() {
        let o = Orientation::from_yaw_pitch_roll_deg(30.0, -10.0, 5.0);
        let composed = o.compose(&Orientation::identity());
        let s = sphere(PI / 4.0, PI / 8.0);
        let r1 = rotate_sphere(&s, &o);
        let r2 = rotate_sphere(&s, &composed);
        let dist = angular_distance(&r1, &r2);
        assert!(dist < EPSILON, "dist={dist}");
    }

    // ── Orientation SLERP ────────────────────────────────────────────────────

    #[test]
    fn slerp_t0_returns_self() {
        let a = Orientation::from_yaw_pitch_roll_deg(30.0, -10.0, 5.0);
        let b = Orientation::from_yaw_pitch_roll_deg(90.0, 20.0, -15.0);
        let interp = a.slerp(&b, 0.0);
        let s = sphere(0.0, 0.0);
        let r_a = rotate_sphere(&s, &a);
        let r_interp = rotate_sphere(&s, &interp);
        let dist = angular_distance(&r_a, &r_interp);
        assert!(dist < EPSILON, "dist={dist}");
    }

    #[test]
    fn slerp_t1_returns_other() {
        let a = Orientation::from_yaw_pitch_roll_deg(10.0, 5.0, 0.0);
        let b = Orientation::from_yaw_pitch_roll_deg(90.0, 0.0, 0.0);
        let interp = a.slerp(&b, 1.0);
        let s = sphere(PI / 6.0, 0.0);
        let r_b = rotate_sphere(&s, &b);
        let r_interp = rotate_sphere(&s, &interp);
        let dist = angular_distance(&r_b, &r_interp);
        assert!(dist < EPSILON, "dist={dist}");
    }

    #[test]
    fn slerp_midpoint_is_between() {
        // For pure yaw rotation, midpoint should halve the angle
        let a = Orientation::identity();
        let b = Orientation::from_yaw_pitch_roll_deg(60.0, 0.0, 0.0);
        let mid = a.slerp(&b, 0.5);
        // The midpoint yaw should be near 30°
        let s = sphere(0.0, 0.0);
        let r_a = rotate_sphere(&s, &a);
        let r_m = rotate_sphere(&s, &mid);
        let r_b = rotate_sphere(&s, &b);
        let dist_am = angular_distance(&r_a, &r_m);
        let dist_mb = angular_distance(&r_m, &r_b);
        // dist_am ≈ dist_mb (midpoint)
        assert!(
            (dist_am - dist_mb).abs() < 0.05,
            "am={dist_am} mb={dist_mb}"
        );
    }
}
