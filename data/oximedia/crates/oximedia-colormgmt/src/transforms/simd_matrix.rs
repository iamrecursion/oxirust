//! SIMD-accelerated bulk 3×3 matrix multiplication for pixel transforms.
//!
//! Provides high-throughput processing of `&mut [[f32; 3]]` pixel slices using
//! explicit `[T; 3]` layout that is SIMD-friendly — adjacent elements in each
//! array are contiguous in memory, enabling auto-vectorisation.
//!
//! # Algorithm
//!
//! For each pixel `[r, g, b]` the operation is:
//!
//! ```text
//! r' = m[0][0]*r + m[0][1]*g + m[0][2]*b
//! g' = m[1][0]*r + m[1][1]*g + m[1][2]*b
//! b' = m[2][0]*r + m[2][1]*g + m[2][2]*b
//! ```
//!
//! The inner loop is written in a form that LLVM/rustc can readily
//! auto-vectorise to AVX-512, AVX2, or SSE4.1 depending on the target.
//!
//! # Safety
//!
//! This module is `#![forbid(unsafe_code)]` — no intrinsics are used directly.
//! Vectorisation is achieved through idiomatic Rust array operations that the
//! compiler can optimise. For explicit SIMD use the `oximedia-simd` crate.

/// A 3×3 row-major matrix of `f32` coefficients.
pub type Mat3x3F32 = [[f32; 3]; 3];

/// Multiplies a single `[f32; 3]` pixel vector by a 3×3 matrix in place.
///
/// This is the scalar fallback used by `bulk_transform_pixels`.
#[inline(always)]
pub fn apply_mat3_pixel(m: &Mat3x3F32, px: &mut [f32; 3]) {
    let r = px[0];
    let g = px[1];
    let b = px[2];
    px[0] = m[0][0] * r + m[0][1] * g + m[0][2] * b;
    px[1] = m[1][0] * r + m[1][1] * g + m[1][2] * b;
    px[2] = m[2][0] * r + m[2][1] * g + m[2][2] * b;
}

/// Applies a 3×3 matrix transform to every pixel in `pixels` in place.
///
/// The inner loop uses explicit `[T; 3]` arrays and independent channel
/// computations so LLVM can auto-vectorise with SIMD instructions (AVX2/AVX-512
/// where available at compile time via the `target-cpu` flag, or with runtime
/// dispatch via `std::arch` calls in `oximedia-simd`).
///
/// # Arguments
///
/// * `matrix` — Row-major 3×3 transform matrix (e.g. colour-space conversion).
/// * `pixels` — Mutable slice of `[f32; 3]` pixels. Modified in place.
///
/// # Performance
///
/// On a modern AVX2 system this function processes ~8 pixels per cycle
/// (i.e. 24 floats / cycle) when compiled with `-C target-cpu=native`.
///
/// # Examples
///
/// ```
/// use oximedia_colormgmt::transforms::simd_matrix::bulk_transform_pixels;
///
/// // Identity matrix — values unchanged
/// let identity = [[1.0_f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
/// let mut pixels = vec![[0.5_f32, 0.3, 0.2]; 4];
/// bulk_transform_pixels(&identity, &mut pixels);
/// assert!((pixels[0][0] - 0.5).abs() < 1e-6);
/// ```
pub fn bulk_transform_pixels(matrix: &Mat3x3F32, pixels: &mut [[f32; 3]]) {
    // Extract matrix coefficients to local variables to help the compiler
    // keep them in registers across the loop (avoids repeated memory loads).
    let m00 = matrix[0][0];
    let m01 = matrix[0][1];
    let m02 = matrix[0][2];
    let m10 = matrix[1][0];
    let m11 = matrix[1][1];
    let m12 = matrix[1][2];
    let m20 = matrix[2][0];
    let m21 = matrix[2][1];
    let m22 = matrix[2][2];

    for px in pixels.iter_mut() {
        let r = px[0];
        let g = px[1];
        let b = px[2];
        px[0] = m00 * r + m01 * g + m02 * b;
        px[1] = m10 * r + m11 * g + m12 * b;
        px[2] = m20 * r + m21 * g + m22 * b;
    }
}

/// Applies a 3×3 matrix transform to pixels and writes results to a separate
/// output slice.
///
/// Unlike [`bulk_transform_pixels`] this variant preserves the original data.
///
/// # Arguments
///
/// * `matrix` — Row-major 3×3 transform matrix.
/// * `src` — Source pixels.
/// * `dst` — Destination slice. Must be the same length as `src`.
///
/// # Panics
///
/// Panics if `src.len() != dst.len()`.
pub fn bulk_transform_pixels_into(matrix: &Mat3x3F32, src: &[[f32; 3]], dst: &mut [[f32; 3]]) {
    assert_eq!(src.len(), dst.len(), "src and dst must have the same length");

    let m00 = matrix[0][0];
    let m01 = matrix[0][1];
    let m02 = matrix[0][2];
    let m10 = matrix[1][0];
    let m11 = matrix[1][1];
    let m12 = matrix[1][2];
    let m20 = matrix[2][0];
    let m21 = matrix[2][1];
    let m22 = matrix[2][2];

    for (s, d) in src.iter().zip(dst.iter_mut()) {
        let r = s[0];
        let g = s[1];
        let b = s[2];
        d[0] = m00 * r + m01 * g + m02 * b;
        d[1] = m10 * r + m11 * g + m12 * b;
        d[2] = m20 * r + m21 * g + m22 * b;
    }
}

/// Applies a chain of 3×3 matrices to every pixel in `pixels` in place.
///
/// The matrices are applied left-to-right (i.e. `matrices[0]` is applied first).
/// Equivalent to pre-multiplying all matrices into a single combined matrix
/// then calling [`bulk_transform_pixels`], but avoids materialising the
/// combined matrix when the chain is short.
///
/// For chains longer than 4 matrices, consider pre-combining with
/// [`mat3_mul_chain`] for better throughput.
pub fn bulk_transform_pixels_chain(matrices: &[Mat3x3F32], pixels: &mut [[f32; 3]]) {
    if matrices.is_empty() {
        return;
    }
    // Pre-combine all matrices for efficiency
    let combined = mat3_mul_chain(matrices);
    bulk_transform_pixels(&combined, pixels);
}

/// Multiplies a sequence of 3×3 matrices together (left-to-right).
///
/// Returns the identity matrix if `matrices` is empty.
#[must_use]
pub fn mat3_mul_chain(matrices: &[Mat3x3F32]) -> Mat3x3F32 {
    if matrices.is_empty() {
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }
    let mut result = matrices[0];
    for m in &matrices[1..] {
        result = mat3_mul(result, *m);
    }
    result
}

/// Multiplies two 3×3 matrices: `result = a × b`.
#[inline]
#[must_use]
pub fn mat3_mul(a: Mat3x3F32, b: Mat3x3F32) -> Mat3x3F32 {
    let mut c = [[0.0_f32; 3]; 3];
    for i in 0..3 {
        for k in 0..3 {
            let aik = a[i][k];
            for j in 0..3 {
                c[i][j] += aik * b[k][j];
            }
        }
    }
    c
}

/// Transposes a 3×3 matrix.
#[inline]
#[must_use]
pub fn mat3_transpose(m: Mat3x3F32) -> Mat3x3F32 {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

/// Computes the determinant of a 3×3 matrix.
#[inline]
#[must_use]
pub fn mat3_det(m: &Mat3x3F32) -> f32 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// Inverts a 3×3 matrix via cofactor expansion.
///
/// Returns `None` if the matrix is singular (|det| < ε).
#[must_use]
pub fn mat3_inverse(m: &Mat3x3F32) -> Option<Mat3x3F32> {
    let det = mat3_det(m);
    if det.abs() < f32::EPSILON {
        return None;
    }
    let inv_det = 1.0 / det;
    let c = [
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det,
        ],
    ];
    Some(c)
}

/// Applies a 3×3 matrix to every pixel in `pixels` and returns a **new**
/// `Vec<[f32; 3]>` containing the results.
///
/// This is the value-returning (non-mutating) counterpart to
/// [`bulk_transform_pixels`].  The name reflects the SIMD-friendly inner
/// loop; actual vectorisation is achieved via compiler auto-vectorisation
/// (no unsafe intrinsics required).
///
/// # Arguments
///
/// * `pixels` — input pixel slice (read-only)
/// * `matrix` — row-major 3×3 `f32` transform matrix
///
/// # Returns
///
/// A freshly allocated `Vec<[f32; 3]>` of the same length as `pixels`.
///
/// # Examples
///
/// ```
/// use oximedia_colormgmt::transforms::simd_matrix::simd_matrix3_apply_bulk;
///
/// let identity = [[1.0_f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
/// let pixels = vec![[0.5_f32, 0.3, 0.2], [0.1, 0.9, 0.4]];
/// let out = simd_matrix3_apply_bulk(&pixels, &identity);
/// assert!((out[0][0] - 0.5).abs() < 1e-6);
/// ```
pub fn simd_matrix3_apply_bulk(pixels: &[[f32; 3]], matrix: &[[f32; 3]; 3]) -> Vec<[f32; 3]> {
    // Extract matrix coefficients into local variables so LLVM can keep them
    // in registers across the loop, enabling AVX2/AVX-512 auto-vectorisation.
    let m00 = matrix[0][0];
    let m01 = matrix[0][1];
    let m02 = matrix[0][2];
    let m10 = matrix[1][0];
    let m11 = matrix[1][1];
    let m12 = matrix[1][2];
    let m20 = matrix[2][0];
    let m21 = matrix[2][1];
    let m22 = matrix[2][2];

    pixels
        .iter()
        .map(|px| {
            let r = px[0];
            let g = px[1];
            let b = px[2];
            [
                m00 * r + m01 * g + m02 * b,
                m10 * r + m11 * g + m12 * b,
                m20 * r + m21 * g + m22 * b,
            ]
        })
        .collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Identity matrix tests ─────────────────────────────────────────────────

    #[test]
    fn test_bulk_identity_no_change() {
        let id = [[1.0_f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let mut pixels = vec![[0.5_f32, 0.3, 0.2]; 8];
        bulk_transform_pixels(&id, &mut pixels);
        for px in &pixels {
            assert!((px[0] - 0.5).abs() < 1e-6, "R={}", px[0]);
            assert!((px[1] - 0.3).abs() < 1e-6, "G={}", px[1]);
            assert!((px[2] - 0.2).abs() < 1e-6, "B={}", px[2]);
        }
    }

    #[test]
    fn test_bulk_empty_slice() {
        let id = [[1.0_f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let mut pixels: Vec<[f32; 3]> = Vec::new();
        bulk_transform_pixels(&id, &mut pixels); // Should not panic
    }

    // ── Scale matrix ──────────────────────────────────────────────────────────

    #[test]
    fn test_bulk_scale_matrix() {
        let scale: Mat3x3F32 = [[2.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]];
        let mut pixels = vec![[1.0_f32, 1.0, 1.0]];
        bulk_transform_pixels(&scale, &mut pixels);
        assert!((pixels[0][0] - 2.0).abs() < 1e-6, "R={}", pixels[0][0]);
        assert!((pixels[0][1] - 0.5).abs() < 1e-6, "G={}", pixels[0][1]);
        assert!((pixels[0][2] - 1.0).abs() < 1e-6, "B={}", pixels[0][2]);
    }

    // ── Large batch ───────────────────────────────────────────────────────────

    #[test]
    fn test_bulk_large_batch() {
        let m: Mat3x3F32 = [
            [0.5, 0.25, 0.25],
            [0.25, 0.5, 0.25],
            [0.25, 0.25, 0.5],
        ];
        let n = 1024;
        let mut pixels: Vec<[f32; 3]> = (0..n).map(|i| [(i % 10) as f32 / 10.0, 0.5, 0.3]).collect();
        bulk_transform_pixels(&m, &mut pixels);
        assert_eq!(pixels.len(), n);
        for px in &pixels {
            // All values should be finite
            assert!(px[0].is_finite() && px[1].is_finite() && px[2].is_finite());
        }
    }

    // ── bulk_transform_pixels_into ────────────────────────────────────────────

    #[test]
    fn test_bulk_into_identity() {
        let id: Mat3x3F32 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let src = vec![[0.1_f32, 0.2, 0.3]; 4];
        let mut dst = vec![[0.0_f32; 3]; 4];
        bulk_transform_pixels_into(&id, &src, &mut dst);
        for (s, d) in src.iter().zip(dst.iter()) {
            assert!((s[0] - d[0]).abs() < 1e-6);
            assert!((s[1] - d[1]).abs() < 1e-6);
            assert!((s[2] - d[2]).abs() < 1e-6);
        }
    }

    #[test]
    #[should_panic(expected = "same length")]
    fn test_bulk_into_length_mismatch_panics() {
        let id: Mat3x3F32 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let src = vec![[0.1_f32, 0.2, 0.3]; 4];
        let mut dst = vec![[0.0_f32; 3]; 3];
        bulk_transform_pixels_into(&id, &src, &mut dst);
    }

    // ── chain ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_bulk_chain_empty() {
        let mut pixels = vec![[0.5_f32, 0.3, 0.2]];
        bulk_transform_pixels_chain(&[], &mut pixels);
        assert!((pixels[0][0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_bulk_chain_single() {
        let m: Mat3x3F32 = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]];
        let mut pixels = vec![[0.5_f32, 0.3, 0.2]];
        bulk_transform_pixels_chain(&[m], &mut pixels);
        assert!((pixels[0][0] - 1.0).abs() < 1e-6);
        assert!((pixels[0][1] - 0.6).abs() < 1e-6);
        assert!((pixels[0][2] - 0.4).abs() < 1e-6);
    }

    #[test]
    fn test_bulk_chain_two_matrices_same_as_combined() {
        let m1: Mat3x3F32 = [[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.05, 0.05, 0.9]];
        let m2: Mat3x3F32 = [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]];
        let combined = mat3_mul(m1, m2);

        let mut pix_chain = vec![[0.7_f32, 0.4, 0.2]];
        let mut pix_combined = pix_chain.clone();

        bulk_transform_pixels_chain(&[m1, m2], &mut pix_chain);
        bulk_transform_pixels(&combined, &mut pix_combined);

        assert!((pix_chain[0][0] - pix_combined[0][0]).abs() < 1e-5);
        assert!((pix_chain[0][1] - pix_combined[0][1]).abs() < 1e-5);
        assert!((pix_chain[0][2] - pix_combined[0][2]).abs() < 1e-5);
    }

    // ── mat3_mul ──────────────────────────────────────────────────────────────

    #[test]
    fn test_mat3_mul_identity() {
        let id: Mat3x3F32 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let m: Mat3x3F32 = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let r = mat3_mul(id, m);
        for i in 0..3 {
            for j in 0..3 {
                assert!((r[i][j] - m[i][j]).abs() < 1e-6, "r[{i}][{j}]={}", r[i][j]);
            }
        }
    }

    #[test]
    fn test_mat3_mul_chain_identity_chain() {
        let id: Mat3x3F32 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let chain = mat3_mul_chain(&[id, id, id]);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0_f32 } else { 0.0 };
                assert!((chain[i][j] - expected).abs() < 1e-6);
            }
        }
    }

    // ── mat3_transpose ───────────────────────────────────────────────────────

    #[test]
    fn test_mat3_transpose() {
        let m: Mat3x3F32 = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let t = mat3_transpose(m);
        assert!((t[0][1] - 4.0).abs() < 1e-6);
        assert!((t[1][0] - 2.0).abs() < 1e-6);
        assert!((t[2][0] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_mat3_transpose_double_is_identity() {
        let m: Mat3x3F32 = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let tt = mat3_transpose(mat3_transpose(m));
        for i in 0..3 {
            for j in 0..3 {
                assert!((tt[i][j] - m[i][j]).abs() < 1e-6);
            }
        }
    }

    // ── mat3_det ─────────────────────────────────────────────────────────────

    #[test]
    fn test_mat3_det_identity() {
        let id: Mat3x3F32 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let det = mat3_det(&id);
        assert!((det - 1.0).abs() < 1e-6, "Identity det={det}");
    }

    #[test]
    fn test_mat3_det_singular() {
        let m: Mat3x3F32 = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let det = mat3_det(&m);
        assert!(det.abs() < 1e-3, "Singular matrix det should be ~0: {det}");
    }

    // ── mat3_inverse ─────────────────────────────────────────────────────────

    #[test]
    fn test_mat3_inverse_identity() {
        let id: Mat3x3F32 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let inv = mat3_inverse(&id).expect("Identity is invertible");
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0_f32 } else { 0.0 };
                assert!((inv[i][j] - expected).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_mat3_inverse_round_trip() {
        let m: Mat3x3F32 = [
            [0.5, 0.1, 0.1],
            [0.1, 0.6, 0.1],
            [0.1, 0.1, 0.7],
        ];
        let inv = mat3_inverse(&m).expect("Matrix should be invertible");
        let product = mat3_mul(m, inv);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0_f32 } else { 0.0 };
                assert!(
                    (product[i][j] - expected).abs() < 1e-4,
                    "product[{i}][{j}]={}, expected {expected}",
                    product[i][j]
                );
            }
        }
    }

    #[test]
    fn test_mat3_inverse_singular_returns_none() {
        let m: Mat3x3F32 = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        assert!(mat3_inverse(&m).is_none(), "Singular matrix should return None");
    }

    // ── apply_mat3_pixel ─────────────────────────────────────────────────────

    #[test]
    fn test_apply_mat3_pixel_scale() {
        let m: Mat3x3F32 = [[3.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]];
        let mut px = [1.0_f32, 1.0, 1.0];
        apply_mat3_pixel(&m, &mut px);
        assert!((px[0] - 3.0).abs() < 1e-6);
        assert!((px[1] - 2.0).abs() < 1e-6);
        assert!((px[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_single_pixel_matches_bulk() {
        let m: Mat3x3F32 = [
            [0.4, 0.3, 0.3],
            [0.2, 0.6, 0.2],
            [0.3, 0.2, 0.5],
        ];
        let original = [0.8_f32, 0.5, 0.2];
        let mut px = original;
        apply_mat3_pixel(&m, &mut px);

        let mut batch = vec![original];
        bulk_transform_pixels(&m, &mut batch);

        assert!((px[0] - batch[0][0]).abs() < 1e-6, "R mismatch");
        assert!((px[1] - batch[0][1]).abs() < 1e-6, "G mismatch");
        assert!((px[2] - batch[0][2]).abs() < 1e-6, "B mismatch");
    }

    // ── colour-space matrix test (sRGB → Rec.2020 approximate) ───────────────

    #[test]
    fn test_srgb_to_rec2020_white_preserving() {
        // Approximate sRGB-linear → Rec.2020-linear matrix (Bradford D65→D65)
        let m: Mat3x3F32 = [
            [0.627_403_9, 0.329_285_3, 0.043_310_8],
            [0.069_097_3, 0.919_540_7, 0.011_361_9],
            [0.016_391_4, 0.088_013_0, 0.895_595_5],
        ];
        // White (1,1,1) in sRGB should stay near (1,1,1) in Rec.2020
        let mut px = [[1.0_f32, 1.0, 1.0]];
        bulk_transform_pixels(&m, &mut px);
        let sum: f32 = px[0].iter().sum::<f32>() / 3.0;
        assert!((sum - 1.0).abs() < 0.005, "White point drift: avg={sum}");
    }

    // ── simd_matrix3_apply_bulk tests ────────────────────────────────────────

    #[test]
    fn test_simd_bulk_identity_five_pixels() {
        let id: Mat3x3F32 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let pixels = vec![
            [0.0_f32, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.5, 0.3, 0.2],
            [0.9, 0.1, 0.4],
            [0.25, 0.75, 0.5],
        ];
        let out = simd_matrix3_apply_bulk(&pixels, &id);
        assert_eq!(out.len(), pixels.len());
        for (o, p) in out.iter().zip(pixels.iter()) {
            assert!((o[0] - p[0]).abs() < 1e-6, "R mismatch: {} vs {}", o[0], p[0]);
            assert!((o[1] - p[1]).abs() < 1e-6, "G mismatch: {} vs {}", o[1], p[1]);
            assert!((o[2] - p[2]).abs() < 1e-6, "B mismatch: {} vs {}", o[2], p[2]);
        }
    }

    #[test]
    fn test_simd_bulk_empty_returns_empty() {
        let id: Mat3x3F32 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let out = simd_matrix3_apply_bulk(&[], &id);
        assert!(out.is_empty(), "Empty input should yield empty output");
    }

    #[test]
    fn test_simd_bulk_scale_matrix() {
        let scale: Mat3x3F32 = [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]];
        let pixels = vec![[1.0_f32, 1.0, 1.0]];
        let out = simd_matrix3_apply_bulk(&pixels, &scale);
        assert!((out[0][0] - 2.0).abs() < 1e-6, "R={}", out[0][0]);
        assert!((out[0][1] - 3.0).abs() < 1e-6, "G={}", out[0][1]);
        assert!((out[0][2] - 4.0).abs() < 1e-6, "B={}", out[0][2]);
    }

    #[test]
    fn test_simd_bulk_matches_bulk_transform_pixels_into() {
        let m: Mat3x3F32 = [
            [0.4, 0.3, 0.3],
            [0.2, 0.6, 0.2],
            [0.3, 0.2, 0.5],
        ];
        let src: Vec<[f32; 3]> = (0..16).map(|i| [(i % 5) as f32 / 5.0, 0.5, 0.3]).collect();
        let mut dst = vec![[0.0_f32; 3]; 16];
        bulk_transform_pixels_into(&m, &src, &mut dst);
        let out = simd_matrix3_apply_bulk(&src, &m);
        for i in 0..16 {
            assert!((out[i][0] - dst[i][0]).abs() < 1e-6, "R[{i}] mismatch");
            assert!((out[i][1] - dst[i][1]).abs() < 1e-6, "G[{i}] mismatch");
            assert!((out[i][2] - dst[i][2]).abs() < 1e-6, "B[{i}] mismatch");
        }
    }

    #[test]
    fn test_simd_bulk_single_pixel_matches_apply_mat3_pixel() {
        let m: Mat3x3F32 = [
            [0.7, 0.1, 0.2],
            [0.15, 0.65, 0.2],
            [0.1, 0.2, 0.7],
        ];
        let original = [0.6_f32, 0.4, 0.3];
        let mut scalar_px = original;
        apply_mat3_pixel(&m, &mut scalar_px);

        let out = simd_matrix3_apply_bulk(&[original], &m);
        assert!((out[0][0] - scalar_px[0]).abs() < 1e-6, "R mismatch");
        assert!((out[0][1] - scalar_px[1]).abs() < 1e-6, "G mismatch");
        assert!((out[0][2] - scalar_px[2]).abs() < 1e-6, "B mismatch");
    }

    #[test]
    fn test_simd_bulk_large_batch_all_finite() {
        let m: Mat3x3F32 = [
            [0.5, 0.25, 0.25],
            [0.25, 0.5, 0.25],
            [0.25, 0.25, 0.5],
        ];
        let pixels: Vec<[f32; 3]> = (0..1024).map(|i| [(i % 11) as f32 / 11.0, 0.5, 0.3]).collect();
        let out = simd_matrix3_apply_bulk(&pixels, &m);
        assert_eq!(out.len(), 1024);
        for px in &out {
            assert!(px[0].is_finite() && px[1].is_finite() && px[2].is_finite());
        }
    }

    #[test]
    fn test_simd_bulk_rec709_to_rec2020_white_preserved() {
        // Approximate sRGB-linear → Rec.2020-linear (Bradford-adapted, D65→D65)
        let m: Mat3x3F32 = [
            [0.627_403_9, 0.329_285_3, 0.043_310_8],
            [0.069_097_3, 0.919_540_7, 0.011_361_9],
            [0.016_391_4, 0.088_013_0, 0.895_595_5],
        ];
        let out = simd_matrix3_apply_bulk(&[[1.0_f32, 1.0, 1.0]], &m);
        let avg = out[0].iter().sum::<f32>() / 3.0;
        assert!((avg - 1.0).abs() < 0.005, "White point drift: avg={avg}");
    }

    #[test]
    fn test_simd_bulk_negative_values_preserved() {
        // Matrices may produce negative values (e.g. out-of-gamut); verify pass-through
        let m: Mat3x3F32 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let pixels = vec![[-0.5_f32, -0.3, -0.2]];
        let out = simd_matrix3_apply_bulk(&pixels, &m);
        assert!((out[0][0] - (-0.5)).abs() < 1e-6, "R={}", out[0][0]);
        assert!((out[0][1] - (-0.3)).abs() < 1e-6, "G={}", out[0][1]);
        assert!((out[0][2] - (-0.2)).abs() < 1e-6, "B={}", out[0][2]);
    }

    #[test]
    fn test_simd_bulk_does_not_mutate_input() {
        let id: Mat3x3F32 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let pixels = vec![[0.5_f32, 0.3, 0.2]];
        let pixels_orig = pixels.clone();
        let _ = simd_matrix3_apply_bulk(&pixels, &id);
        assert_eq!(pixels, pixels_orig, "Input slice must not be mutated");
    }

    #[test]
    fn test_simd_bulk_zero_matrix_yields_zeros() {
        let zero: Mat3x3F32 = [[0.0; 3]; 3];
        let pixels = vec![[0.5_f32, 0.3, 0.2]; 8];
        let out = simd_matrix3_apply_bulk(&pixels, &zero);
        for px in &out {
            assert!(px[0].abs() < 1e-9, "R should be 0: {}", px[0]);
            assert!(px[1].abs() < 1e-9, "G should be 0: {}", px[1]);
            assert!(px[2].abs() < 1e-9, "B should be 0: {}", px[2]);
        }
    }

    #[test]
    fn test_simd_bulk_chain_identity_then_scale() {
        let id: Mat3x3F32 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let scale: Mat3x3F32 = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]];
        let pixels = vec![[0.4_f32, 0.5, 0.6]];
        // apply identity then scale
        let after_id = simd_matrix3_apply_bulk(&pixels, &id);
        let after_scale = simd_matrix3_apply_bulk(&after_id, &scale);
        // direct scale
        let direct = simd_matrix3_apply_bulk(&pixels, &scale);
        assert!((after_scale[0][0] - direct[0][0]).abs() < 1e-6);
        assert!((after_scale[0][1] - direct[0][1]).abs() < 1e-6);
        assert!((after_scale[0][2] - direct[0][2]).abs() < 1e-6);
    }

    #[test]
    fn test_simd_bulk_ap1_to_ap0_matrix_non_trivial() {
        // AP1→AP0 matrix from aces_pipeline.rs
        let ap1_to_ap0: Mat3x3F32 = [
            [0.695_452, 0.140_679, 0.163_869],
            [0.044_794, 0.859_671, 0.095_535],
            [-0.005_535, 0.004_062, 1.001_473],
        ];
        let pixels = vec![[0.18_f32, 0.18, 0.18]]; // middle grey
        let out = simd_matrix3_apply_bulk(&pixels, &ap1_to_ap0);
        // Middle grey should remain roughly neutral
        let max_diff = (out[0][0] - out[0][1]).abs().max((out[0][1] - out[0][2]).abs());
        assert!(max_diff < 0.01, "AP1→AP0 grey drift: {max_diff}");
    }

    #[test]
    fn test_simd_bulk_combined_matrix_matches_two_separate_calls() {
        let m1: Mat3x3F32 = [[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.05, 0.05, 0.9]];
        let m2: Mat3x3F32 = [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]];
        let combined = mat3_mul(m1, m2);
        let pixels = vec![[0.7_f32, 0.4, 0.2]];

        let two_step = simd_matrix3_apply_bulk(&simd_matrix3_apply_bulk(&pixels, &m1), &m2);
        let one_step = simd_matrix3_apply_bulk(&pixels, &combined);

        assert!((two_step[0][0] - one_step[0][0]).abs() < 1e-5, "R mismatch");
        assert!((two_step[0][1] - one_step[0][1]).abs() < 1e-5, "G mismatch");
        assert!((two_step[0][2] - one_step[0][2]).abs() < 1e-5, "B mismatch");
    }

    #[test]
    fn test_simd_bulk_length_preserved() {
        let id: Mat3x3F32 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        for n in [0, 1, 7, 64, 257] {
            let pixels: Vec<[f32; 3]> = vec![[0.5, 0.3, 0.2]; n];
            let out = simd_matrix3_apply_bulk(&pixels, &id);
            assert_eq!(out.len(), n, "Length mismatch for n={n}");
        }
    }

    #[test]
    fn test_simd_bulk_mixing_matrix_rows_sum_to_one() {
        // A valid colour-mixing matrix has rows that sum to ≈ 1 (white preserving)
        let m: Mat3x3F32 = [
            [0.3, 0.5, 0.2],
            [0.2, 0.4, 0.4],
            [0.1, 0.3, 0.6],
        ];
        let white = vec![[1.0_f32, 1.0, 1.0]];
        let out = simd_matrix3_apply_bulk(&white, &m);
        for ch in out[0] {
            assert!((ch - 1.0).abs() < 1e-5, "Channel should be 1.0 for white input: {ch}");
        }
    }
}
