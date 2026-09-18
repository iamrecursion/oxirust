//! SIMD-friendly 4×4 matrix operations.
//!
//! Provides a row-major `Mat4x4` (stored as `[f32; 16]`) together with
//! common affine and projection helpers. All inner loops are manually
//! unrolled so that auto-vectorisers can emit SIMD instructions.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(dead_code)]

#[cfg(test)]
use std::f32::consts::PI;

/// A 4×4 single-precision floating-point matrix stored in row-major order.
///
/// Element `(row, col)` lives at index `row * 4 + col`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat4x4 {
    /// Flat row-major storage.
    pub data: [f32; 16],
}

impl Mat4x4 {
    /// Create a matrix from a flat row-major array.
    #[inline]
    #[must_use]
    pub const fn from_array(data: [f32; 16]) -> Self {
        Self { data }
    }

    /// Access element at `(row, col)`.
    #[inline]
    #[must_use]
    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.data[row * 4 + col]
    }

    /// Set element at `(row, col)`.
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: f32) {
        self.data[row * 4 + col] = value;
    }
}

impl Default for Mat4x4 {
    fn default() -> Self {
        mat4_identity()
    }
}

// ── Constructors ─────────────────────────────────────────────────────────────

/// Return a 4×4 identity matrix.
#[must_use]
pub fn mat4_identity() -> Mat4x4 {
    Mat4x4::from_array([
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ])
}

/// Return a uniform-scale matrix `diag(sx, sy, sz, 1)`.
#[must_use]
pub fn mat4_scale(sx: f32, sy: f32, sz: f32) -> Mat4x4 {
    Mat4x4::from_array([
        sx, 0.0, 0.0, 0.0, 0.0, sy, 0.0, 0.0, 0.0, 0.0, sz, 0.0, 0.0, 0.0, 0.0, 1.0,
    ])
}

/// Return a translation matrix for vector `(tx, ty, tz)`.
#[must_use]
pub fn mat4_translation(tx: f32, ty: f32, tz: f32) -> Mat4x4 {
    Mat4x4::from_array([
        1.0, 0.0, 0.0, tx, 0.0, 1.0, 0.0, ty, 0.0, 0.0, 1.0, tz, 0.0, 0.0, 0.0, 1.0,
    ])
}

/// Return a rotation matrix around the Z-axis for `angle` radians.
#[must_use]
pub fn mat4_rotation_z(angle: f32) -> Mat4x4 {
    let c = angle.cos();
    let s = angle.sin();
    Mat4x4::from_array([
        c, -s, 0.0, 0.0, s, c, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ])
}

// ── Operations ───────────────────────────────────────────────────────────────

/// Multiply two 4×4 matrices: return `a * b`.
///
/// The inner loop is fully unrolled (16 dot-products of length 4) to give
/// the auto-vectoriser the best possible view of the computation.
#[must_use]
pub fn mat4_multiply(a: &Mat4x4, b: &Mat4x4) -> Mat4x4 {
    let a = &a.data;
    let b = &b.data;
    let mut c = [0.0f32; 16];

    // Row 0
    c[0] = a[0] * b[0] + a[1] * b[4] + a[2] * b[8] + a[3] * b[12];
    c[1] = a[0] * b[1] + a[1] * b[5] + a[2] * b[9] + a[3] * b[13];
    c[2] = a[0] * b[2] + a[1] * b[6] + a[2] * b[10] + a[3] * b[14];
    c[3] = a[0] * b[3] + a[1] * b[7] + a[2] * b[11] + a[3] * b[15];
    // Row 1
    c[4] = a[4] * b[0] + a[5] * b[4] + a[6] * b[8] + a[7] * b[12];
    c[5] = a[4] * b[1] + a[5] * b[5] + a[6] * b[9] + a[7] * b[13];
    c[6] = a[4] * b[2] + a[5] * b[6] + a[6] * b[10] + a[7] * b[14];
    c[7] = a[4] * b[3] + a[5] * b[7] + a[6] * b[11] + a[7] * b[15];
    // Row 2
    c[8] = a[8] * b[0] + a[9] * b[4] + a[10] * b[8] + a[11] * b[12];
    c[9] = a[8] * b[1] + a[9] * b[5] + a[10] * b[9] + a[11] * b[13];
    c[10] = a[8] * b[2] + a[9] * b[6] + a[10] * b[10] + a[11] * b[14];
    c[11] = a[8] * b[3] + a[9] * b[7] + a[10] * b[11] + a[11] * b[15];
    // Row 3
    c[12] = a[12] * b[0] + a[13] * b[4] + a[14] * b[8] + a[15] * b[12];
    c[13] = a[12] * b[1] + a[13] * b[5] + a[14] * b[9] + a[15] * b[13];
    c[14] = a[12] * b[2] + a[13] * b[6] + a[14] * b[10] + a[15] * b[14];
    c[15] = a[12] * b[3] + a[13] * b[7] + a[14] * b[11] + a[15] * b[15];

    Mat4x4::from_array(c)
}

/// Transpose a 4×4 matrix.
#[must_use]
pub fn mat4_transpose(m: &Mat4x4) -> Mat4x4 {
    let d = &m.data;
    Mat4x4::from_array([
        d[0], d[4], d[8], d[12], d[1], d[5], d[9], d[13], d[2], d[6], d[10], d[14], d[3], d[7],
        d[11], d[15],
    ])
}

/// Multiply a 4×4 matrix by a column vector `[x, y, z, w]` and return the
/// result as `[x', y', z', w']`.
#[must_use]
pub fn mat4_vec_multiply(m: &Mat4x4, v: [f32; 4]) -> [f32; 4] {
    let d = &m.data;
    [
        d[0] * v[0] + d[1] * v[1] + d[2] * v[2] + d[3] * v[3],
        d[4] * v[0] + d[5] * v[1] + d[6] * v[2] + d[7] * v[3],
        d[8] * v[0] + d[9] * v[1] + d[10] * v[2] + d[11] * v[3],
        d[12] * v[0] + d[13] * v[1] + d[14] * v[2] + d[15] * v[3],
    ]
}

/// Compute the determinant of a 4×4 matrix via cofactor expansion along
/// the first row.
#[must_use]
pub fn mat4_determinant(m: &Mat4x4) -> f32 {
    let d = &m.data;

    // 3×3 sub-determinants (cofactors of row 0)
    let c00 = det3(d[5], d[6], d[7], d[9], d[10], d[11], d[13], d[14], d[15]);
    let c01 = det3(d[4], d[6], d[7], d[8], d[10], d[11], d[12], d[14], d[15]);
    let c02 = det3(d[4], d[5], d[7], d[8], d[9], d[11], d[12], d[13], d[15]);
    let c03 = det3(d[4], d[5], d[6], d[8], d[9], d[10], d[12], d[13], d[14]);

    d[0] * c00 - d[1] * c01 + d[2] * c02 - d[3] * c03
}

/// Helper: determinant of a 3×3 matrix given its 9 elements row-by-row.
#[inline]
fn det3(a: f32, b: f32, c: f32, d: f32, e: f32, f: f32, g: f32, h: f32, i: f32) -> f32 {
    a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
}

// ── Additional helpers ───────────────────────────────────────────────────────

/// Return a rotation matrix around the X-axis for `angle` radians.
#[must_use]
pub fn mat4_rotation_x(angle: f32) -> Mat4x4 {
    let c = angle.cos();
    let s = angle.sin();
    Mat4x4::from_array([
        1.0, 0.0, 0.0, 0.0, 0.0, c, -s, 0.0, 0.0, s, c, 0.0, 0.0, 0.0, 0.0, 1.0,
    ])
}

/// Return a rotation matrix around the Y-axis for `angle` radians.
#[must_use]
pub fn mat4_rotation_y(angle: f32) -> Mat4x4 {
    let c = angle.cos();
    let s = angle.sin();
    Mat4x4::from_array([
        c, 0.0, s, 0.0, 0.0, 1.0, 0.0, 0.0, -s, 0.0, c, 0.0, 0.0, 0.0, 0.0, 1.0,
    ])
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    fn mat_approx_eq(a: &Mat4x4, b: &Mat4x4) -> bool {
        a.data
            .iter()
            .zip(b.data.iter())
            .all(|(x, y)| approx_eq(*x, *y))
    }

    #[test]
    fn test_identity_diagonal() {
        let id = mat4_identity();
        for i in 0..4 {
            assert!(approx_eq(id.get(i, i), 1.0), "diagonal {i} should be 1");
            for j in 0..4 {
                if i != j {
                    assert!(
                        approx_eq(id.get(i, j), 0.0),
                        "off-diag ({i},{j}) should be 0"
                    );
                }
            }
        }
    }

    #[test]
    fn test_multiply_identity() {
        let id = mat4_identity();
        let a = mat4_scale(2.0, 3.0, 4.0);
        let result = mat4_multiply(&a, &id);
        assert!(mat_approx_eq(&result, &a));
    }

    #[test]
    fn test_multiply_identity_left() {
        let id = mat4_identity();
        let a = mat4_translation(1.0, 2.0, 3.0);
        let result = mat4_multiply(&id, &a);
        assert!(mat_approx_eq(&result, &a));
    }

    #[test]
    fn test_scale_matrix() {
        let s = mat4_scale(2.0, 3.0, 4.0);
        assert!(approx_eq(s.get(0, 0), 2.0));
        assert!(approx_eq(s.get(1, 1), 3.0));
        assert!(approx_eq(s.get(2, 2), 4.0));
        assert!(approx_eq(s.get(3, 3), 1.0));
    }

    #[test]
    fn test_translation_applies_to_point() {
        let t = mat4_translation(5.0, -3.0, 7.0);
        let p = [1.0f32, 0.0, 0.0, 1.0]; // point (1,0,0)
        let r = mat4_vec_multiply(&t, p);
        assert!(approx_eq(r[0], 6.0));
        assert!(approx_eq(r[1], -3.0));
        assert!(approx_eq(r[2], 7.0));
        assert!(approx_eq(r[3], 1.0));
    }

    #[test]
    fn test_rotation_z_90_degrees() {
        let rot = mat4_rotation_z(PI / 2.0);
        let v = [1.0f32, 0.0, 0.0, 0.0];
        let r = mat4_vec_multiply(&rot, v);
        // Rotating (1,0,0) by 90° around Z gives (0,1,0)
        assert!(approx_eq(r[0], 0.0));
        assert!(approx_eq(r[1], 1.0));
        assert!(approx_eq(r[2], 0.0));
    }

    #[test]
    fn test_rotation_z_identity_at_zero() {
        let rot = mat4_rotation_z(0.0);
        let id = mat4_identity();
        assert!(mat_approx_eq(&rot, &id));
    }

    #[test]
    fn test_transpose_identity() {
        let id = mat4_identity();
        let t = mat4_transpose(&id);
        assert!(mat_approx_eq(&id, &t));
    }

    #[test]
    fn test_transpose_involution() {
        let m = mat4_scale(1.0, 2.0, 3.0);
        let tt = mat4_transpose(&mat4_transpose(&m));
        assert!(mat_approx_eq(&m, &tt));
    }

    #[test]
    fn test_transpose_swaps_elements() {
        let t = mat4_translation(1.0, 2.0, 3.0);
        let tr = mat4_transpose(&t);
        // Original: row 0 col 3 = 1.0; transposed: row 3 col 0 = 1.0
        assert!(approx_eq(tr.get(3, 0), 1.0));
        assert!(approx_eq(tr.get(3, 1), 2.0));
        assert!(approx_eq(tr.get(3, 2), 3.0));
    }

    #[test]
    fn test_determinant_identity() {
        let id = mat4_identity();
        assert!(approx_eq(mat4_determinant(&id), 1.0));
    }

    #[test]
    fn test_determinant_scale() {
        // det(diag(2,3,4,1)) = 2*3*4*1 = 24
        let s = mat4_scale(2.0, 3.0, 4.0);
        assert!(approx_eq(mat4_determinant(&s), 24.0));
    }

    #[test]
    fn test_determinant_singular_zero() {
        let mut m = mat4_identity();
        // Zero out the first row → singular
        m.set(0, 0, 0.0);
        m.set(0, 1, 0.0);
        m.set(0, 2, 0.0);
        m.set(0, 3, 0.0);
        assert!(approx_eq(mat4_determinant(&m), 0.0));
    }

    #[test]
    fn test_rotation_x_90_degrees() {
        let rot = mat4_rotation_x(PI / 2.0);
        let v = [0.0f32, 1.0, 0.0, 0.0];
        let r = mat4_vec_multiply(&rot, v);
        // Rotating (0,1,0) by 90° around X gives (0,0,1)
        assert!(approx_eq(r[1], 0.0));
        assert!(approx_eq(r[2], 1.0));
    }

    #[test]
    fn test_rotation_y_90_degrees() {
        let rot = mat4_rotation_y(PI / 2.0);
        let v = [0.0f32, 0.0, 1.0, 0.0];
        let r = mat4_vec_multiply(&rot, v);
        // Rotating (0,0,1) by 90° around Y gives (1,0,0)
        assert!(approx_eq(r[0], 1.0));
        assert!(approx_eq(r[2], 0.0));
    }

    #[test]
    fn test_get_set_roundtrip() {
        let mut m = mat4_identity();
        m.set(2, 3, 42.0);
        assert!(approx_eq(m.get(2, 3), 42.0));
    }

    #[test]
    fn test_default_is_identity() {
        let d = Mat4x4::default();
        let id = mat4_identity();
        assert!(mat_approx_eq(&d, &id));
    }
}
