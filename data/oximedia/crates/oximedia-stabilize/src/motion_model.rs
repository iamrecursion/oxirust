//! Standalone motion model utilities: affine, homography, and translation-only
//! representations with conversion, composition, and inversion helpers.
//!
//! This module provides lightweight, self-contained motion model types that do
//! not depend on the full `motion` sub-tree, making them easy to use in unit
//! tests and small utilities.

#![allow(dead_code)]

use std::f64::consts::PI;

/// A 2-D point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point2 {
    /// X coordinate.
    pub x: f64,
    /// Y coordinate.
    pub y: f64,
}

impl Point2 {
    /// Creates a new [`Point2`].
    #[must_use]
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Euclidean distance to another point.
    #[must_use]
    pub fn distance_to(self, other: Self) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }
}

/// Pure 2-D translation (pan/tilt model).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TranslationModel {
    /// Horizontal displacement in pixels.
    pub dx: f64,
    /// Vertical displacement in pixels.
    pub dy: f64,
}

impl TranslationModel {
    /// Creates a new [`TranslationModel`].
    #[must_use]
    pub const fn new(dx: f64, dy: f64) -> Self {
        Self { dx, dy }
    }

    /// Returns the identity (no movement).
    #[must_use]
    pub const fn identity() -> Self {
        Self { dx: 0.0, dy: 0.0 }
    }

    /// Applies the translation to a point.
    #[must_use]
    pub fn apply(&self, p: Point2) -> Point2 {
        Point2::new(p.x + self.dx, p.y + self.dy)
    }

    /// Returns the inverse translation.
    #[must_use]
    pub const fn inverse(&self) -> Self {
        Self {
            dx: -self.dx,
            dy: -self.dy,
        }
    }

    /// Composes `self` then `other`.
    #[must_use]
    pub fn compose(&self, other: &Self) -> Self {
        Self {
            dx: self.dx + other.dx,
            dy: self.dy + other.dy,
        }
    }

    /// Returns the magnitude of the translation vector.
    #[must_use]
    pub fn magnitude(&self) -> f64 {
        (self.dx.powi(2) + self.dy.powi(2)).sqrt()
    }
}

/// Affine motion model: translation + rotation + uniform scale.
///
/// The 2×3 affine matrix is stored in row-major order as:
/// ```text
/// [ a00  a01  tx ]
/// [ a10  a11  ty ]
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AffineModel {
    /// Affine matrix coefficients `[a00, a01, tx, a10, a11, ty]`.
    pub matrix: [f64; 6],
}

impl AffineModel {
    /// Creates an [`AffineModel`] from raw matrix coefficients.
    #[must_use]
    pub const fn from_matrix(matrix: [f64; 6]) -> Self {
        Self { matrix }
    }

    /// Creates an identity (no-op) affine model.
    #[must_use]
    pub const fn identity() -> Self {
        Self {
            matrix: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        }
    }

    /// Creates an affine model from translation, rotation, and scale.
    #[must_use]
    pub fn from_trs(tx: f64, ty: f64, rotation_rad: f64, scale: f64) -> Self {
        let cos_r = rotation_rad.cos() * scale;
        let sin_r = rotation_rad.sin() * scale;
        Self {
            matrix: [cos_r, -sin_r, tx, sin_r, cos_r, ty],
        }
    }

    /// Applies the affine transform to a point.
    #[must_use]
    pub fn apply(&self, p: Point2) -> Point2 {
        let [a00, a01, tx, a10, a11, ty] = self.matrix;
        Point2::new(a00 * p.x + a01 * p.y + tx, a10 * p.x + a11 * p.y + ty)
    }

    /// Returns the translation component.
    #[must_use]
    pub fn translation(&self) -> TranslationModel {
        TranslationModel::new(self.matrix[2], self.matrix[5])
    }

    /// Extracts the rotation angle in radians (approximate, assumes small shear).
    #[must_use]
    pub fn rotation_rad(&self) -> f64 {
        self.matrix[3].atan2(self.matrix[0])
    }

    /// Extracts the scale factor (geometric mean of the two diagonal elements).
    #[must_use]
    pub fn scale(&self) -> f64 {
        let sx = (self.matrix[0].powi(2) + self.matrix[3].powi(2)).sqrt();
        let sy = (self.matrix[1].powi(2) + self.matrix[4].powi(2)).sqrt();
        (sx * sy).sqrt()
    }

    /// Composes `self` with `other` (applies `self` first, then `other`).
    #[must_use]
    pub fn compose(&self, other: &Self) -> Self {
        let [a, b, tx1, c, d, ty1] = self.matrix;
        let [e, f, tx2, g, h, ty2] = other.matrix;
        Self {
            matrix: [
                a * e + b * g,
                a * f + b * h,
                a * tx2 + b * ty2 + tx1,
                c * e + d * g,
                c * f + d * h,
                c * tx2 + d * ty2 + ty1,
            ],
        }
    }

    /// Attempts to invert the affine model.
    ///
    /// Returns `None` if the matrix is singular (determinant ≈ 0).
    #[must_use]
    pub fn inverse(&self) -> Option<Self> {
        let [a, b, tx, c, d, ty] = self.matrix;
        let det = a * d - b * c;
        if det.abs() < 1e-10 {
            return None;
        }
        let inv_det = 1.0 / det;
        Some(Self {
            matrix: [
                d * inv_det,
                -b * inv_det,
                (b * ty - d * tx) * inv_det,
                -c * inv_det,
                a * inv_det,
                (c * tx - a * ty) * inv_det,
            ],
        })
    }
}

/// 3×3 homography matrix (perspective model) stored in row-major order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HomographyModel {
    /// Row-major 3×3 matrix.
    pub h: [f64; 9],
}

impl HomographyModel {
    /// Creates a [`HomographyModel`] from a raw 9-element array.
    #[must_use]
    pub const fn from_matrix(h: [f64; 9]) -> Self {
        Self { h }
    }

    /// Returns the identity homography.
    #[must_use]
    pub const fn identity() -> Self {
        Self {
            h: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        }
    }

    /// Applies the homography to a 2-D point using homogeneous coordinates.
    ///
    /// Returns `None` if the w-component is too close to zero.
    #[must_use]
    pub fn apply(&self, p: Point2) -> Option<Point2> {
        let [h0, h1, h2, h3, h4, h5, h6, h7, h8] = self.h;
        let w = h6 * p.x + h7 * p.y + h8;
        if w.abs() < 1e-10 {
            return None;
        }
        let xp = (h0 * p.x + h1 * p.y + h2) / w;
        let yp = (h3 * p.x + h4 * p.y + h5) / w;
        Some(Point2::new(xp, yp))
    }

    /// Converts this homography to an affine model by discarding the
    /// perspective rows (h6, h7 set to 0, h8 to 1).
    #[must_use]
    pub fn to_affine(&self) -> AffineModel {
        let [a, b, c, d, e, f, _, _, _] = self.h;
        AffineModel::from_matrix([a, b, c, d, e, f])
    }

    /// Returns `true` if the last row is close to `[0, 0, 1]` (affine case).
    #[must_use]
    pub fn is_affine(&self) -> bool {
        self.h[6].abs() < 1e-8 && self.h[7].abs() < 1e-8 && (self.h[8] - 1.0).abs() < 1e-8
    }
}

/// Converts a rotation angle in degrees to radians.
#[must_use]
pub fn degrees_to_radians(deg: f64) -> f64 {
    deg * PI / 180.0
}

/// Converts a rotation angle in radians to degrees.
#[must_use]
pub fn radians_to_degrees(rad: f64) -> f64 {
    rad * 180.0 / PI
}

/// Fits a [`TranslationModel`] from two lists of corresponding points using
/// the mean displacement.
///
/// Returns `None` if the input is empty or mismatched.
#[must_use]
pub fn fit_translation(src: &[Point2], dst: &[Point2]) -> Option<TranslationModel> {
    if src.is_empty() || src.len() != dst.len() {
        return None;
    }
    let n = src.len() as f64;
    let dx: f64 = src.iter().zip(dst).map(|(s, d)| d.x - s.x).sum::<f64>() / n;
    let dy: f64 = src.iter().zip(dst).map(|(s, d)| d.y - s.y).sum::<f64>() / n;
    Some(TranslationModel::new(dx, dy))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_distance() {
        let p1 = Point2::new(0.0, 0.0);
        let p2 = Point2::new(3.0, 4.0);
        assert!((p1.distance_to(p2) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_translation_apply() {
        let t = TranslationModel::new(10.0, -5.0);
        let p = Point2::new(0.0, 0.0);
        let q = t.apply(p);
        assert!((q.x - 10.0).abs() < 1e-10);
        assert!((q.y - (-5.0)).abs() < 1e-10);
    }

    #[test]
    fn test_translation_inverse() {
        let t = TranslationModel::new(3.0, 7.0);
        let inv = t.inverse();
        let p = Point2::new(5.0, 5.0);
        let moved = t.apply(p);
        let back = inv.apply(moved);
        assert!((back.x - p.x).abs() < 1e-10);
        assert!((back.y - p.y).abs() < 1e-10);
    }

    #[test]
    fn test_translation_compose() {
        let a = TranslationModel::new(1.0, 2.0);
        let b = TranslationModel::new(3.0, 4.0);
        let c = a.compose(&b);
        assert!((c.dx - 4.0).abs() < 1e-10);
        assert!((c.dy - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_translation_magnitude() {
        let t = TranslationModel::new(3.0, 4.0);
        assert!((t.magnitude() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_affine_identity_apply() {
        let a = AffineModel::identity();
        let p = Point2::new(7.0, -3.0);
        let q = a.apply(p);
        assert!((q.x - p.x).abs() < 1e-10);
        assert!((q.y - p.y).abs() < 1e-10);
    }

    #[test]
    fn test_affine_from_trs_rotation_only() {
        let angle = PI / 2.0;
        let a = AffineModel::from_trs(0.0, 0.0, angle, 1.0);
        let p = Point2::new(1.0, 0.0);
        let q = a.apply(p);
        // 90° rotation of (1, 0) → approximately (0, 1)
        assert!(q.x.abs() < 1e-10, "x = {}", q.x);
        assert!((q.y - 1.0).abs() < 1e-10, "y = {}", q.y);
    }

    #[test]
    fn test_affine_inverse_roundtrip() {
        let a = AffineModel::from_trs(5.0, -3.0, 0.2, 1.1);
        let inv = a.inverse().expect("should be invertible");
        let p = Point2::new(10.0, 20.0);
        let q = a.apply(p);
        let back = inv.apply(q);
        assert!(
            (back.x - p.x).abs() < 1e-8,
            "x diff = {}",
            (back.x - p.x).abs()
        );
        assert!(
            (back.y - p.y).abs() < 1e-8,
            "y diff = {}",
            (back.y - p.y).abs()
        );
    }

    #[test]
    fn test_affine_compose_with_identity() {
        let a = AffineModel::from_trs(4.0, 2.0, 0.5, 1.0);
        let id = AffineModel::identity();
        let c = a.compose(&id);
        let p = Point2::new(1.0, 1.0);
        let q1 = a.apply(p);
        let q2 = c.apply(p);
        assert!((q1.x - q2.x).abs() < 1e-10);
        assert!((q1.y - q2.y).abs() < 1e-10);
    }

    #[test]
    fn test_affine_rotation_extraction() {
        let angle = 0.3;
        let a = AffineModel::from_trs(0.0, 0.0, angle, 1.0);
        let extracted = a.rotation_rad();
        assert!(
            (extracted - angle).abs() < 1e-8,
            "angle diff = {}",
            (extracted - angle).abs()
        );
    }

    #[test]
    fn test_homography_identity_apply() {
        let h = HomographyModel::identity();
        let p = Point2::new(5.0, 8.0);
        let q = h.apply(p).expect("should succeed in test");
        assert!((q.x - p.x).abs() < 1e-10);
        assert!((q.y - p.y).abs() < 1e-10);
    }

    #[test]
    fn test_homography_is_affine() {
        let h = HomographyModel::identity();
        assert!(h.is_affine());
    }

    #[test]
    fn test_homography_to_affine_identity() {
        let h = HomographyModel::identity();
        let a = h.to_affine();
        let id = AffineModel::identity();
        for (v, expected) in a.matrix.iter().zip(id.matrix.iter()) {
            assert!((v - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_fit_translation_empty_is_none() {
        assert!(fit_translation(&[], &[]).is_none());
    }

    #[test]
    fn test_fit_translation_mismatch_is_none() {
        let a = vec![Point2::new(0.0, 0.0)];
        let b = vec![Point2::new(1.0, 1.0), Point2::new(2.0, 2.0)];
        assert!(fit_translation(&a, &b).is_none());
    }

    #[test]
    fn test_fit_translation_exact() {
        let src = vec![Point2::new(0.0, 0.0), Point2::new(1.0, 1.0)];
        let dst = vec![Point2::new(5.0, -3.0), Point2::new(6.0, -2.0)];
        let t = fit_translation(&src, &dst).expect("should succeed in test");
        assert!((t.dx - 5.0).abs() < 1e-10);
        assert!((t.dy - (-3.0)).abs() < 1e-10);
    }

    #[test]
    fn test_degrees_radians_roundtrip() {
        let deg = 45.0;
        let rad = degrees_to_radians(deg);
        let back = radians_to_degrees(rad);
        assert!((back - deg).abs() < 1e-10);
    }
}
