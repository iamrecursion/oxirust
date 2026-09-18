//! Geometric camera calibration models.
//!
//! Implements the pinhole camera model, radial and tangential distortion,
//! a combined camera model with (un)distortion, and homography transforms.

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// PinholeModel
// ---------------------------------------------------------------------------

/// Pinhole camera intrinsic parameters.
///
/// - `fx`, `fy`: focal lengths in pixel units
/// - `cx`, `cy`: principal point (image centre) in pixel units
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PinholeModel {
    /// Horizontal focal length (px).
    pub fx: f64,
    /// Vertical focal length (px).
    pub fy: f64,
    /// Principal-point x coordinate (px).
    pub cx: f64,
    /// Principal-point y coordinate (px).
    pub cy: f64,
}

impl PinholeModel {
    /// Project a 3-D point `(X, Y, Z)` into image coordinates `(u, v)`.
    ///
    /// The camera is assumed to be at the world origin looking down +Z.
    /// Returns `(NAN, NAN)` if `Z` is zero.
    #[must_use]
    pub fn project(&self, point_3d: (f64, f64, f64)) -> (f64, f64) {
        let (x, y, z) = point_3d;
        if z.abs() < f64::EPSILON {
            return (f64::NAN, f64::NAN);
        }
        let u = self.fx * (x / z) + self.cx;
        let v = self.fy * (y / z) + self.cy;
        (u, v)
    }
}

// ---------------------------------------------------------------------------
// RadialDistortion
// ---------------------------------------------------------------------------

/// Radial (barrel / pincushion) distortion coefficients.
///
/// Models the radial distortion factor:
/// `distortion_factor = 1 + k1*r² + k2*r⁴ + k3*r⁶`
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RadialDistortion {
    /// Second-order radial coefficient.
    pub k1: f64,
    /// Fourth-order radial coefficient.
    pub k2: f64,
    /// Sixth-order radial coefficient.
    pub k3: f64,
}

impl RadialDistortion {
    /// Compute the radial distortion factor for squared radius `r2`.
    #[must_use]
    pub fn apply(&self, r2: f64) -> f64 {
        1.0 + self.k1 * r2 + self.k2 * r2 * r2 + self.k3 * r2 * r2 * r2
    }
}

// ---------------------------------------------------------------------------
// TangentialDistortion
// ---------------------------------------------------------------------------

/// Tangential (decentering) distortion coefficients.
///
/// Models lens de-centering using the Brown–Conrady model.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TangentialDistortion {
    /// First tangential coefficient.
    pub p1: f64,
    /// Second tangential coefficient.
    pub p2: f64,
}

impl TangentialDistortion {
    /// Compute the tangential displacement `(dx, dy)` at normalised image
    /// coordinates `(x, y)`.
    #[must_use]
    pub fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        let r2 = x * x + y * y;
        let dx = 2.0 * self.p1 * x * y + self.p2 * (r2 + 2.0 * x * x);
        let dy = self.p1 * (r2 + 2.0 * y * y) + 2.0 * self.p2 * x * y;
        (dx, dy)
    }
}

// ---------------------------------------------------------------------------
// CameraModel
// ---------------------------------------------------------------------------

/// Complete camera model combining pinhole projection, radial and tangential
/// distortion.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CameraModel {
    /// Intrinsic pinhole parameters.
    pub pinhole: PinholeModel,
    /// Radial distortion coefficients.
    pub radial: RadialDistortion,
    /// Tangential distortion coefficients.
    pub tangential: TangentialDistortion,
}

impl CameraModel {
    /// Apply distortion to a normalised image point `(x, y)` and return the
    /// distorted normalised point.
    #[must_use]
    pub fn distort_point(&self, x: f64, y: f64) -> (f64, f64) {
        let r2 = x * x + y * y;
        let radial_factor = self.radial.apply(r2);
        let (tdx, tdy) = self.tangential.apply(x, y);
        let xd = x * radial_factor + tdx;
        let yd = y * radial_factor + tdy;
        (xd, yd)
    }

    /// Iteratively undistort a normalised image point `(xd, yd)`.
    ///
    /// Uses 10 Gauss–Newton iterations to invert the distortion model.
    #[must_use]
    pub fn undistort_point(&self, xd: f64, yd: f64) -> (f64, f64) {
        let mut x = xd;
        let mut y = yd;
        for _ in 0..10 {
            let r2 = x * x + y * y;
            let radial_factor = self.radial.apply(r2);
            let (tdx, tdy) = self.tangential.apply(x, y);
            x = (xd - tdx) / radial_factor;
            y = (yd - tdy) / radial_factor;
        }
        (x, y)
    }
}

// ---------------------------------------------------------------------------
// HomographyMatrix
// ---------------------------------------------------------------------------

/// 3×3 homography (projective transform) matrix in row-major order.
///
/// Transforms planar points between two views using homogeneous coordinates.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct HomographyMatrix {
    /// Row-major 3×3 matrix.
    pub m: [[f64; 3]; 3],
}

impl HomographyMatrix {
    /// Identity homography – no transformation.
    #[must_use]
    pub const fn identity() -> Self {
        Self {
            m: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }
    }

    /// Transform a 2-D point `(x, y)` by this homography.
    ///
    /// Returns `(NAN, NAN)` when the homogeneous weight `w` is zero.
    #[must_use]
    pub fn transform_point(&self, x: f64, y: f64) -> (f64, f64) {
        let h = &self.m;
        let xp = h[0][0] * x + h[0][1] * y + h[0][2];
        let yp = h[1][0] * x + h[1][1] * y + h[1][2];
        let wp = h[2][0] * x + h[2][1] * y + h[2][2];
        if wp.abs() < f64::EPSILON {
            return (f64::NAN, f64::NAN);
        }
        (xp / wp, yp / wp)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_pinhole() -> PinholeModel {
        PinholeModel {
            fx: 800.0,
            fy: 800.0,
            cx: 320.0,
            cy: 240.0,
        }
    }

    fn zero_distortion() -> (RadialDistortion, TangentialDistortion) {
        (
            RadialDistortion {
                k1: 0.0,
                k2: 0.0,
                k3: 0.0,
            },
            TangentialDistortion { p1: 0.0, p2: 0.0 },
        )
    }

    // ── PinholeModel ─────────────────────────────────────────────────────

    #[test]
    fn test_pinhole_project_centre() {
        let cam = default_pinhole();
        // Point on the optical axis projects to the principal point
        let (u, v) = cam.project((0.0, 0.0, 1.0));
        assert!((u - 320.0).abs() < 1e-9, "u={u}");
        assert!((v - 240.0).abs() < 1e-9, "v={v}");
    }

    #[test]
    fn test_pinhole_project_offset_point() {
        let cam = default_pinhole();
        // (1, 0, 1) → u = fx * 1 + cx = 800 + 320 = 1120
        let (u, v) = cam.project((1.0, 0.0, 1.0));
        assert!((u - 1120.0).abs() < 1e-9, "u={u}");
        assert!((v - 240.0).abs() < 1e-9, "v={v}");
    }

    #[test]
    fn test_pinhole_project_zero_z_returns_nan() {
        let cam = default_pinhole();
        let (u, v) = cam.project((1.0, 1.0, 0.0));
        assert!(u.is_nan() && v.is_nan());
    }

    #[test]
    fn test_pinhole_project_behind_camera() {
        let cam = default_pinhole();
        // Negative Z should still project (behind camera)
        let (u, _v) = cam.project((0.0, 0.0, -1.0));
        // u = fx * (0 / -1) + cx = cx
        assert!((u - 320.0).abs() < 1e-9, "u={u}");
    }

    // ── RadialDistortion ─────────────────────────────────────────────────

    #[test]
    fn test_radial_distortion_zero_coefficients() {
        let r = RadialDistortion {
            k1: 0.0,
            k2: 0.0,
            k3: 0.0,
        };
        assert!((r.apply(1.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_radial_distortion_applies_k1() {
        let r = RadialDistortion {
            k1: 0.1,
            k2: 0.0,
            k3: 0.0,
        };
        // factor = 1 + 0.1 * 4.0 = 1.4 at r2 = 4
        assert!((r.apply(4.0) - 1.4).abs() < 1e-9);
    }

    #[test]
    fn test_radial_distortion_all_coefficients() {
        let r = RadialDistortion {
            k1: 0.1,
            k2: 0.01,
            k3: 0.001,
        };
        // factor = 1 + 0.1*1 + 0.01*1 + 0.001*1 = 1.111 at r2 = 1
        assert!((r.apply(1.0) - 1.111).abs() < 1e-9);
    }

    // ── TangentialDistortion ──────────────────────────────────────────────

    #[test]
    fn test_tangential_distortion_zero_at_origin() {
        let t = TangentialDistortion { p1: 0.5, p2: 0.3 };
        let (dx, dy) = t.apply(0.0, 0.0);
        assert!(dx.abs() < 1e-12 && dy.abs() < 1e-12);
    }

    #[test]
    fn test_tangential_distortion_zero_coefficients() {
        let t = TangentialDistortion { p1: 0.0, p2: 0.0 };
        let (dx, dy) = t.apply(1.0, 1.0);
        assert!(dx.abs() < 1e-12 && dy.abs() < 1e-12);
    }

    // ── CameraModel ───────────────────────────────────────────────────────

    #[test]
    fn test_camera_model_distort_undistort_round_trip() {
        let (radial, tangential) = zero_distortion();
        let cam = CameraModel {
            pinhole: default_pinhole(),
            radial,
            tangential,
        };
        // With zero distortion, distort then undistort should recover original
        let (xd, yd) = cam.distort_point(0.3, 0.2);
        let (xu, yu) = cam.undistort_point(xd, yd);
        assert!((xu - 0.3).abs() < 1e-9, "xu={xu}");
        assert!((yu - 0.2).abs() < 1e-9, "yu={yu}");
    }

    #[test]
    fn test_camera_model_distort_with_radial() {
        let radial = RadialDistortion {
            k1: -0.1,
            k2: 0.0,
            k3: 0.0,
        };
        let tangential = TangentialDistortion { p1: 0.0, p2: 0.0 };
        let cam = CameraModel {
            pinhole: default_pinhole(),
            radial,
            tangential,
        };
        let (xd, yd) = cam.distort_point(1.0, 0.0);
        // r2 = 1, factor = 1 - 0.1 = 0.9 → xd = 0.9
        assert!((xd - 0.9).abs() < 1e-9, "xd={xd}");
        assert!((yd).abs() < 1e-12, "yd={yd}");
    }

    // ── HomographyMatrix ──────────────────────────────────────────────────

    #[test]
    fn test_homography_identity_transform() {
        let h = HomographyMatrix::identity();
        let (xp, yp) = h.transform_point(3.0, 5.0);
        assert!((xp - 3.0).abs() < 1e-9, "xp={xp}");
        assert!((yp - 5.0).abs() < 1e-9, "yp={yp}");
    }

    #[test]
    fn test_homography_translation() {
        let mut h = HomographyMatrix::identity();
        h.m[0][2] = 10.0; // tx
        h.m[1][2] = -5.0; // ty
        let (xp, yp) = h.transform_point(0.0, 0.0);
        assert!((xp - 10.0).abs() < 1e-9, "xp={xp}");
        assert!((yp - (-5.0)).abs() < 1e-9, "yp={yp}");
    }

    #[test]
    fn test_homography_zero_w_returns_nan() {
        let mut h = HomographyMatrix::identity();
        // Make the last row zero so w = 0
        h.m[2] = [0.0, 0.0, 0.0];
        let (xp, yp) = h.transform_point(1.0, 1.0);
        assert!(xp.is_nan() && yp.is_nan());
    }
}
