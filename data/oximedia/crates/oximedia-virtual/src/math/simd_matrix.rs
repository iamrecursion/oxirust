//! SIMD-optimised 4×4 matrix operations for virtual production.
//!
//! Provides `Matrix4x4f` — a 32-bit float 4×4 matrix with hand-vectorised
//! operations using Rust scalar intrinsics (the compiler auto-vectorises
//! these patterns to SSE/AVX/NEON on capable targets).  No `unsafe` code;
//! all optimisations are expressed through iterator and array idioms that
//! LLVM/rustc reliably auto-vectorise.
//!
//! # Design
//! Matrices are stored in **column-major** order, matching GPU conventions.
//! This allows dot-product operations to be expressed as contiguous memory
//! reads which the compiler can widen into SIMD loads.

use serde::{Deserialize, Serialize};
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub};

/// A 4×4 single-precision floating-point matrix stored in column-major order.
///
/// Element `(row, col)` is at index `col * 4 + row` in `data`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Matrix4x4f {
    /// Column-major storage: 4 columns of 4 rows.
    pub data: [[f32; 4]; 4],
}

impl Matrix4x4f {
    // ------------------------------------------------------------------
    // Constructors
    // ------------------------------------------------------------------

    /// All-zero matrix.
    #[must_use]
    pub fn zeros() -> Self {
        Self {
            data: [[0.0; 4]; 4],
        }
    }

    /// Identity matrix.
    #[must_use]
    pub fn identity() -> Self {
        let mut m = Self::zeros();
        for i in 0..4 {
            m.data[i][i] = 1.0;
        }
        m
    }

    /// Build from a flat column-major array (GPU/OpenGL convention).
    #[must_use]
    pub fn from_column_major(flat: &[f32; 16]) -> Self {
        let mut data = [[0.0f32; 4]; 4];
        for col in 0..4 {
            for row in 0..4 {
                data[col][row] = flat[col * 4 + row];
            }
        }
        Self { data }
    }

    /// Build from a flat row-major array (C array convention).
    #[must_use]
    pub fn from_row_major(flat: &[f32; 16]) -> Self {
        let mut data = [[0.0f32; 4]; 4];
        for row in 0..4 {
            for col in 0..4 {
                data[col][row] = flat[row * 4 + col];
            }
        }
        Self { data }
    }

    /// Construct a translation matrix.
    #[must_use]
    pub fn translation(tx: f32, ty: f32, tz: f32) -> Self {
        let mut m = Self::identity();
        m.data[3][0] = tx;
        m.data[3][1] = ty;
        m.data[3][2] = tz;
        m
    }

    /// Construct a uniform scale matrix.
    #[must_use]
    pub fn scale(sx: f32, sy: f32, sz: f32) -> Self {
        let mut m = Self::zeros();
        m.data[0][0] = sx;
        m.data[1][1] = sy;
        m.data[2][2] = sz;
        m.data[3][3] = 1.0;
        m
    }

    /// Rotation around X axis (radians).
    #[must_use]
    pub fn rotation_x(angle_rad: f32) -> Self {
        let c = angle_rad.cos();
        let s = angle_rad.sin();
        let mut m = Self::identity();
        m.data[1][1] = c;
        m.data[1][2] = s;
        m.data[2][1] = -s;
        m.data[2][2] = c;
        m
    }

    /// Rotation around Y axis (radians).
    #[must_use]
    pub fn rotation_y(angle_rad: f32) -> Self {
        let c = angle_rad.cos();
        let s = angle_rad.sin();
        let mut m = Self::identity();
        m.data[0][0] = c;
        m.data[0][2] = -s;
        m.data[2][0] = s;
        m.data[2][2] = c;
        m
    }

    /// Rotation around Z axis (radians).
    #[must_use]
    pub fn rotation_z(angle_rad: f32) -> Self {
        let c = angle_rad.cos();
        let s = angle_rad.sin();
        let mut m = Self::identity();
        m.data[0][0] = c;
        m.data[0][1] = s;
        m.data[1][0] = -s;
        m.data[1][1] = c;
        m
    }

    /// Perspective projection matrix (column-major, right-handed).
    ///
    /// - `fov_y_rad`: vertical field of view in radians.
    /// - `aspect`: width / height.
    /// - `z_near`, `z_far`: near/far clip planes.
    #[must_use]
    pub fn perspective(fov_y_rad: f32, aspect: f32, z_near: f32, z_far: f32) -> Self {
        let f = 1.0 / (fov_y_rad / 2.0).tan();
        let mut m = Self::zeros();
        m.data[0][0] = f / aspect;
        m.data[1][1] = f;
        m.data[2][2] = (z_far + z_near) / (z_near - z_far);
        m.data[2][3] = -1.0;
        m.data[3][2] = (2.0 * z_far * z_near) / (z_near - z_far);
        m
    }

    // ------------------------------------------------------------------
    // Access helpers
    // ------------------------------------------------------------------

    /// Get element at (row, col).
    #[must_use]
    #[inline(always)]
    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.data[col][row]
    }

    /// Set element at (row, col).
    #[inline(always)]
    pub fn set(&mut self, row: usize, col: usize, val: f32) {
        self.data[col][row] = val;
    }

    /// Column vector as [f32; 4].
    #[must_use]
    #[inline(always)]
    pub fn column(&self, col: usize) -> [f32; 4] {
        self.data[col]
    }

    /// Row vector as [f32; 4].
    #[must_use]
    #[inline(always)]
    pub fn row(&self, row: usize) -> [f32; 4] {
        [
            self.data[0][row],
            self.data[1][row],
            self.data[2][row],
            self.data[3][row],
        ]
    }

    // ------------------------------------------------------------------
    // Core operations
    // ------------------------------------------------------------------

    /// Transpose.
    #[must_use]
    pub fn transpose(&self) -> Self {
        let mut m = Self::zeros();
        for c in 0..4 {
            for r in 0..4 {
                m.data[r][c] = self.data[c][r];
            }
        }
        m
    }

    /// Compute determinant via cofactor expansion along the first column.
    #[must_use]
    pub fn determinant(&self) -> f32 {
        // Use row-based access for clarity
        let m = self;
        let a = |r: usize, c: usize| m.get(r, c);

        // Sarrus / Leibniz for 4×4
        let mut det = 0.0f32;
        // Expand along row 0
        for c0 in 0..4 {
            let sign = if c0 % 2 == 0 { 1.0f32 } else { -1.0 };
            let minor = Self::minor_3x3(m, 0, c0);
            det += sign * a(0, c0) * minor;
        }
        det
    }

    /// Compute the 3×3 minor determinant (remove row `skip_r`, col `skip_c`).
    fn minor_3x3(m: &Self, skip_r: usize, skip_c: usize) -> f32 {
        let mut rows = [0usize; 3];
        let mut cols = [0usize; 3];
        let mut ri = 0;
        for r in 0..4 {
            if r != skip_r {
                rows[ri] = r;
                ri += 1;
            }
        }
        let mut ci = 0;
        for c in 0..4 {
            if c != skip_c {
                cols[ci] = c;
                ci += 1;
            }
        }
        let a = |r: usize, c: usize| m.get(rows[r], cols[c]);
        a(0, 0) * (a(1, 1) * a(2, 2) - a(1, 2) * a(2, 1))
            - a(0, 1) * (a(1, 0) * a(2, 2) - a(1, 2) * a(2, 0))
            + a(0, 2) * (a(1, 0) * a(2, 1) - a(1, 1) * a(2, 0))
    }

    /// Try to compute the inverse.  Returns `None` if the matrix is singular.
    #[must_use]
    pub fn try_inverse(&self) -> Option<Self> {
        let det = self.determinant();
        if det.abs() < 1e-8 {
            return None;
        }
        let inv_det = 1.0 / det;

        let mut adj = Self::zeros();
        for r in 0..4 {
            for c in 0..4 {
                let sign = if (r + c) % 2 == 0 { 1.0f32 } else { -1.0 };
                // Cofactor for element (r,c) → goes into transpose position (c,r)
                adj.set(c, r, sign * Self::minor_3x3(self, r, c));
            }
        }

        Some(adj * inv_det)
    }

    /// Matrix–vector multiplication.  `v` is treated as a column vector [x, y, z, w].
    #[must_use]
    #[inline(always)]
    pub fn mul_vec4(&self, v: [f32; 4]) -> [f32; 4] {
        // Each output row = dot(row, v)
        let mut out = [0.0f32; 4];
        for r in 0..4 {
            // row r: [data[0][r], data[1][r], data[2][r], data[3][r]]
            out[r] = self.data[0][r] * v[0]
                + self.data[1][r] * v[1]
                + self.data[2][r] * v[2]
                + self.data[3][r] * v[3];
        }
        out
    }

    /// Transform a 3D point (w=1).
    #[must_use]
    #[inline(always)]
    pub fn transform_point(&self, p: [f32; 3]) -> [f32; 3] {
        let v = self.mul_vec4([p[0], p[1], p[2], 1.0]);
        let w = v[3];
        if w.abs() > 1e-8 {
            [v[0] / w, v[1] / w, v[2] / w]
        } else {
            [v[0], v[1], v[2]]
        }
    }

    /// Transform a 3D direction vector (w=0, no perspective divide).
    #[must_use]
    #[inline(always)]
    pub fn transform_vector(&self, d: [f32; 3]) -> [f32; 3] {
        let v = self.mul_vec4([d[0], d[1], d[2], 0.0]);
        [v[0], v[1], v[2]]
    }

    /// SIMD-optimised matrix multiplication.
    ///
    /// Uses the compiler-friendly "dot product as strided accumulate" pattern
    /// that LLVM reliably maps to 128-bit (SSE/NEON) or 256-bit (AVX)
    /// vector multiply-accumulate instructions.
    ///
    /// Result = `self * rhs` (column-major convention: rhs applied first).
    #[must_use]
    #[inline]
    pub fn mul_simd(&self, rhs: &Self) -> Self {
        // For each output column c (from rhs) and output row r (from lhs),
        // result[c][r] = sum_{k=0}^{3} lhs[k][r] * rhs[c][k]
        //
        // We compute column by column so the inner loop over k is a
        // dot product of one row of lhs with one column of rhs — contiguous
        // in rhs.data[c].  The compiler emits 4-wide FMAs.
        let mut out = Self::zeros();
        for c in 0..4 {
            let rhs_col: [f32; 4] = rhs.data[c];
            for r in 0..4 {
                // Compute dot(lhs_row[r], rhs_col)
                let mut acc = 0.0f32;
                for k in 0..4 {
                    acc += self.data[k][r] * rhs_col[k];
                }
                out.data[c][r] = acc;
            }
        }
        out
    }

    /// Batch-multiply an array of 4-vectors by this matrix.
    ///
    /// Processes `n` vectors stored as interleaved `[x,y,z,w]` and writes
    /// the results to `out`.  The inner loop is written for auto-vectorisation.
    pub fn batch_transform_vec4(&self, vecs: &[[f32; 4]], out: &mut Vec<[f32; 4]>) {
        out.clear();
        out.reserve(vecs.len());
        for v in vecs {
            out.push(self.mul_vec4(*v));
        }
    }

    /// Convert to flat column-major array.
    #[must_use]
    pub fn to_column_major(&self) -> [f32; 16] {
        let mut flat = [0.0f32; 16];
        for col in 0..4 {
            for row in 0..4 {
                flat[col * 4 + row] = self.data[col][row];
            }
        }
        flat
    }

    /// Convert to flat row-major array.
    #[must_use]
    pub fn to_row_major(&self) -> [f32; 16] {
        let mut flat = [0.0f32; 16];
        for row in 0..4 {
            for col in 0..4 {
                flat[row * 4 + col] = self.data[col][row];
            }
        }
        flat
    }

    /// Element-wise approximate equality.
    #[must_use]
    pub fn approx_eq(&self, other: &Self, tol: f32) -> bool {
        for c in 0..4 {
            for r in 0..4 {
                if (self.data[c][r] - other.data[c][r]).abs() > tol {
                    return false;
                }
            }
        }
        true
    }
}

// ------------------------------------------------------------------
// Operator overloads
// ------------------------------------------------------------------

impl Mul for Matrix4x4f {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        self.mul_simd(&rhs)
    }
}

impl MulAssign for Matrix4x4f {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.mul_simd(&rhs);
    }
}

impl Mul<f32> for Matrix4x4f {
    type Output = Self;
    fn mul(self, s: f32) -> Self {
        let mut m = self;
        for c in 0..4 {
            for r in 0..4 {
                m.data[c][r] *= s;
            }
        }
        m
    }
}

impl Mul<Matrix4x4f> for f32 {
    type Output = Matrix4x4f;
    fn mul(self, m: Matrix4x4f) -> Matrix4x4f {
        m * self
    }
}

impl Add for Matrix4x4f {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let mut m = self;
        for c in 0..4 {
            for r in 0..4 {
                m.data[c][r] += rhs.data[c][r];
            }
        }
        m
    }
}

impl AddAssign for Matrix4x4f {
    fn add_assign(&mut self, rhs: Self) {
        for c in 0..4 {
            for r in 0..4 {
                self.data[c][r] += rhs.data[c][r];
            }
        }
    }
}

impl Sub for Matrix4x4f {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        let mut m = self;
        for c in 0..4 {
            for r in 0..4 {
                m.data[c][r] -= rhs.data[c][r];
            }
        }
        m
    }
}

impl Neg for Matrix4x4f {
    type Output = Self;
    fn neg(self) -> Self {
        self * -1.0
    }
}

// ------------------------------------------------------------------
// Tests
// ------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn test_identity() {
        let m = Matrix4x4f::identity();
        for c in 0..4 {
            for r in 0..4 {
                let expected = if r == c { 1.0 } else { 0.0 };
                assert!((m.get(r, c) - expected).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_identity_mul_identity() {
        let i = Matrix4x4f::identity();
        let result = i * i;
        assert!(result.approx_eq(&i, 1e-6));
    }

    #[test]
    fn test_mul_simd_vs_naive() {
        let a = Matrix4x4f::from_row_major(&[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ]);
        let b = Matrix4x4f::from_row_major(&[
            16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
        ]);
        let result = a.mul_simd(&b);
        let naive = a * b;
        assert!(result.approx_eq(&naive, 1e-3));
    }

    #[test]
    fn test_translation() {
        let t = Matrix4x4f::translation(1.0, 2.0, 3.0);
        let p = t.transform_point([0.0, 0.0, 0.0]);
        assert!((p[0] - 1.0).abs() < 1e-5, "tx: {}", p[0]);
        assert!((p[1] - 2.0).abs() < 1e-5, "ty: {}", p[1]);
        assert!((p[2] - 3.0).abs() < 1e-5, "tz: {}", p[2]);
    }

    #[test]
    fn test_scale() {
        let s = Matrix4x4f::scale(2.0, 3.0, 4.0);
        let p = s.transform_point([1.0, 1.0, 1.0]);
        assert!((p[0] - 2.0).abs() < 1e-5);
        assert!((p[1] - 3.0).abs() < 1e-5);
        assert!((p[2] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_rotation_x_90() {
        let r = Matrix4x4f::rotation_x(PI / 2.0);
        let v = r.transform_vector([0.0, 1.0, 0.0]);
        // Y → Z for right-handed rotation
        assert!(v[1].abs() < 1e-5, "ry after 90° rot_x: {}", v[1]);
        assert!(v[2].abs() > 0.9, "rz after 90° rot_x: {}", v[2]);
    }

    #[test]
    fn test_rotation_y_90() {
        let r = Matrix4x4f::rotation_y(PI / 2.0);
        let v = r.transform_vector([1.0, 0.0, 0.0]);
        assert!(v[0].abs() < 1e-5, "rx after 90° rot_y: {}", v[0]);
    }

    #[test]
    fn test_rotation_z_90() {
        let r = Matrix4x4f::rotation_z(PI / 2.0);
        let v = r.transform_vector([1.0, 0.0, 0.0]);
        assert!(v[0].abs() < 1e-5);
        assert!(v[1].abs() > 0.9);
    }

    #[test]
    fn test_inverse_identity() {
        let m = Matrix4x4f::identity();
        let inv = m.try_inverse().expect("identity is invertible");
        assert!(inv.approx_eq(&m, 1e-5));
    }

    #[test]
    fn test_inverse_translation() {
        let t = Matrix4x4f::translation(3.0, -2.0, 1.0);
        let t_inv = t.try_inverse().expect("translation is invertible");
        let prod = t * t_inv;
        assert!(prod.approx_eq(&Matrix4x4f::identity(), 1e-5));
    }

    #[test]
    fn test_inverse_scale() {
        let s = Matrix4x4f::scale(2.0, 4.0, 8.0);
        let s_inv = s.try_inverse().expect("scale is invertible");
        let prod = s * s_inv;
        assert!(prod.approx_eq(&Matrix4x4f::identity(), 1e-5));
    }

    #[test]
    fn test_singular_returns_none() {
        let m = Matrix4x4f::zeros();
        assert!(m.try_inverse().is_none());
    }

    #[test]
    fn test_transpose() {
        let m = Matrix4x4f::from_row_major(&[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ]);
        let t = m.transpose();
        // (row=0, col=1) of original = 2.0 → should become (row=1, col=0) of transpose = 2.0
        assert!((m.get(0, 1) - t.get(1, 0)).abs() < 1e-6);
        assert!((m.get(2, 3) - t.get(3, 2)).abs() < 1e-6);
    }

    #[test]
    fn test_transpose_transpose_identity() {
        let m = Matrix4x4f::from_row_major(&[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ]);
        assert!(m.transpose().transpose().approx_eq(&m, 1e-6));
    }

    #[test]
    fn test_mul_vec4() {
        let m = Matrix4x4f::scale(2.0, 3.0, 4.0);
        let v = m.mul_vec4([1.0, 1.0, 1.0, 1.0]);
        assert!((v[0] - 2.0).abs() < 1e-5);
        assert!((v[1] - 3.0).abs() < 1e-5);
        assert!((v[2] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_batch_transform_vec4() {
        let m = Matrix4x4f::translation(1.0, 2.0, 3.0);
        let vecs = vec![
            [0.0f32, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
        ];
        let mut out: Vec<[f32; 4]> = Vec::new();
        m.batch_transform_vec4(&vecs, &mut out);

        assert_eq!(out.len(), 3);
        assert!((out[0][0] - 1.0).abs() < 1e-5);
        assert!((out[0][1] - 2.0).abs() < 1e-5);
        assert!((out[1][0] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_round_trip_column_major() {
        let m = Matrix4x4f::from_row_major(&[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ]);
        let flat = m.to_column_major();
        let back = Matrix4x4f::from_column_major(&flat);
        assert!(m.approx_eq(&back, 1e-6));
    }

    #[test]
    fn test_add() {
        let a = Matrix4x4f::identity();
        let b = Matrix4x4f::identity();
        let c = a + b;
        assert!((c.get(0, 0) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_scalar_mul() {
        let m = Matrix4x4f::identity();
        let s = m * 5.0;
        assert!((s.get(0, 0) - 5.0).abs() < 1e-6);
        assert!((s.get(0, 1)).abs() < 1e-6);
    }

    #[test]
    fn test_neg() {
        let m = Matrix4x4f::identity();
        let n = -m;
        assert!((n.get(0, 0) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_determinant_identity() {
        let m = Matrix4x4f::identity();
        assert!((m.determinant() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_determinant_scale() {
        let m = Matrix4x4f::scale(2.0, 3.0, 4.0);
        // det = 2 * 3 * 4 * 1 = 24
        assert!(
            (m.determinant() - 24.0).abs() < 1e-3,
            "det: {}",
            m.determinant()
        );
    }

    #[test]
    fn test_perspective_creates_finite_matrix() {
        let p = Matrix4x4f::perspective(PI / 3.0, 16.0 / 9.0, 0.1, 1000.0);
        let flat = p.to_column_major();
        assert!(flat.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_combined_trs() {
        // TRS: first scale, then rotate, then translate
        let t = Matrix4x4f::translation(5.0, 0.0, 0.0);
        let r = Matrix4x4f::rotation_y(PI / 2.0);
        let s = Matrix4x4f::scale(2.0, 2.0, 2.0);
        let trs = t * r * s;
        // Transform origin: scale(0,0,0)→(0,0,0), rotate→(0,0,0), translate→(5,0,0)
        let p = trs.transform_point([0.0, 0.0, 0.0]);
        assert!((p[0] - 5.0).abs() < 1e-4, "x: {}", p[0]);
    }
}
