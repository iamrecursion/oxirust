//! Affine transformation for image alignment.
//!
//! Provides 2D affine transformation matrices for translating, rotating, and
//! scaling images during the alignment process.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// A 3×3 affine transformation matrix stored in row-major order.
///
/// The third row is always `[0, 0, 1]` for 2D homogeneous coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AffineMatrix {
    /// Row-major 3×3 matrix data.
    pub data: [[f32; 3]; 3],
}

impl AffineMatrix {
    /// Return the identity matrix.
    #[must_use]
    pub fn identity() -> Self {
        Self {
            data: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }
    }

    /// Return a translation matrix for `(tx, ty)`.
    #[must_use]
    pub fn translation(tx: f32, ty: f32) -> Self {
        Self {
            data: [[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]],
        }
    }

    /// Return a counter-clockwise rotation matrix for the given angle (radians).
    #[must_use]
    pub fn rotation(angle_rad: f32) -> Self {
        let c = angle_rad.cos();
        let s = angle_rad.sin();
        Self {
            data: [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        }
    }

    /// Return a scaling matrix with independent x and y scale factors.
    #[must_use]
    pub fn scale(sx: f32, sy: f32) -> Self {
        Self {
            data: [[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]],
        }
    }

    /// Multiply this matrix by `rhs` and return the result.
    ///
    /// The product represents first applying `rhs`, then `self`.
    #[must_use]
    pub fn multiply(&self, rhs: &AffineMatrix) -> AffineMatrix {
        let a = &self.data;
        let b = &rhs.data;
        let mut out = [[0.0_f32; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    out[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        AffineMatrix { data: out }
    }

    /// Apply this matrix to the homogeneous 2D point `(x, y, 1)` and return
    /// the transformed `(x', y')`.
    #[must_use]
    pub fn transform_point(&self, x: f32, y: f32) -> (f32, f32) {
        let d = &self.data;
        let xp = d[0][0] * x + d[0][1] * y + d[0][2];
        let yp = d[1][0] * x + d[1][1] * y + d[1][2];
        (xp, yp)
    }

    /// Return `true` if this matrix is (approximately) the identity.
    ///
    /// Uses an element-wise tolerance of `1e-5`.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        let id = AffineMatrix::identity();
        for i in 0..3 {
            for j in 0..3 {
                if (self.data[i][j] - id.data[i][j]).abs() > 1e-5 {
                    return false;
                }
            }
        }
        true
    }
}

impl Default for AffineMatrix {
    fn default() -> Self {
        Self::identity()
    }
}

/// A wrapper around [`AffineMatrix`] that supports composition and inversion.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AffineTransform {
    /// The underlying 3×3 matrix.
    pub matrix: AffineMatrix,
}

impl AffineTransform {
    /// Wrap an existing matrix in an `AffineTransform`.
    #[must_use]
    pub fn new(matrix: AffineMatrix) -> Self {
        Self { matrix }
    }

    /// Compose this transform with `other`, returning the combined transform.
    ///
    /// The result first applies `other`, then `self`.
    #[must_use]
    pub fn compose(&self, other: &AffineTransform) -> AffineTransform {
        AffineTransform {
            matrix: self.matrix.multiply(&other.matrix),
        }
    }

    /// Return the inverse translation transform (negates tx and ty, keeps
    /// rotation/scale as-is).
    ///
    /// This is a simplified inversion suitable only for pure translation matrices.
    #[must_use]
    pub fn inverse_translation(&self) -> AffineTransform {
        let tx = -self.matrix.data[0][2];
        let ty = -self.matrix.data[1][2];
        AffineTransform {
            matrix: AffineMatrix::translation(tx, ty),
        }
    }
}

impl Default for AffineTransform {
    fn default() -> Self {
        Self::new(AffineMatrix::identity())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn test_identity_is_identity() {
        let m = AffineMatrix::identity();
        assert!(m.is_identity());
    }

    #[test]
    fn test_translation_matrix_values() {
        let m = AffineMatrix::translation(3.0, -5.0);
        assert_eq!(m.data[0][2], 3.0);
        assert_eq!(m.data[1][2], -5.0);
        assert_eq!(m.data[2][2], 1.0);
        assert!(!m.is_identity());
    }

    #[test]
    fn test_transform_point_identity() {
        let m = AffineMatrix::identity();
        let (xp, yp) = m.transform_point(7.0, -3.0);
        assert!((xp - 7.0).abs() < 1e-5);
        assert!((yp + 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_transform_point_translation() {
        let m = AffineMatrix::translation(10.0, 20.0);
        let (xp, yp) = m.transform_point(1.0, 2.0);
        assert!((xp - 11.0).abs() < 1e-5);
        assert!((yp - 22.0).abs() < 1e-5);
    }

    #[test]
    fn test_rotation_90_degrees() {
        let m = AffineMatrix::rotation(PI / 2.0);
        let (xp, yp) = m.transform_point(1.0, 0.0);
        // Rotating (1,0) by 90° CCW gives (0,1)
        assert!((xp - 0.0).abs() < 1e-5);
        assert!((yp - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_rotation_zero_is_identity() {
        let m = AffineMatrix::rotation(0.0);
        assert!(m.is_identity());
    }

    #[test]
    fn test_scale_matrix() {
        let m = AffineMatrix::scale(2.0, 3.0);
        let (xp, yp) = m.transform_point(4.0, 5.0);
        assert!((xp - 8.0).abs() < 1e-5);
        assert!((yp - 15.0).abs() < 1e-5);
    }

    #[test]
    fn test_scale_identity_is_identity() {
        let m = AffineMatrix::scale(1.0, 1.0);
        assert!(m.is_identity());
    }

    #[test]
    fn test_multiply_identity_with_translation() {
        let id = AffineMatrix::identity();
        let t = AffineMatrix::translation(5.0, -2.0);
        let result = id.multiply(&t);
        assert!((result.data[0][2] - 5.0).abs() < 1e-5);
        assert!((result.data[1][2] + 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_multiply_two_translations() {
        let t1 = AffineMatrix::translation(3.0, 4.0);
        let t2 = AffineMatrix::translation(1.0, -1.0);
        let result = t1.multiply(&t2);
        assert!((result.data[0][2] - 4.0).abs() < 1e-5);
        assert!((result.data[1][2] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_affine_transform_compose() {
        let t1 = AffineTransform::new(AffineMatrix::translation(2.0, 0.0));
        let t2 = AffineTransform::new(AffineMatrix::translation(0.0, 3.0));
        let composed = t1.compose(&t2);
        let (xp, yp) = composed.matrix.transform_point(0.0, 0.0);
        assert!((xp - 2.0).abs() < 1e-5);
        assert!((yp - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_inverse_translation() {
        let t = AffineTransform::new(AffineMatrix::translation(7.0, -4.0));
        let inv = t.inverse_translation();
        assert!((inv.matrix.data[0][2] + 7.0).abs() < 1e-5);
        assert!((inv.matrix.data[1][2] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_translation_then_inverse_is_identity() {
        let t = AffineTransform::new(AffineMatrix::translation(5.0, 3.0));
        let inv = t.inverse_translation();
        let composed = t.compose(&inv);
        // Composing pure translation with its inverse should give identity translation
        let (xp, yp) = composed.matrix.transform_point(1.0, 1.0);
        assert!((xp - 1.0).abs() < 1e-4);
        assert!((yp - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_default_affine_matrix_is_identity() {
        let m = AffineMatrix::default();
        assert!(m.is_identity());
    }

    #[test]
    fn test_default_affine_transform_is_identity() {
        let t = AffineTransform::default();
        assert!(t.matrix.is_identity());
    }
}
