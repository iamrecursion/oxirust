//! SIMD-optimized mathematical operations on f32 slices.
//!
//! Provides scalar implementations with SIMD-friendly access patterns for
//! dot products, matrix multiplication, prefix sums, and other common
//! linear-algebra primitives used in media processing pipelines.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(dead_code)]

/// Compute the dot product of two slices.
///
/// If the slices differ in length only the shorter prefix is used.
#[must_use]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Element-wise addition of two slices.
///
/// Output length equals `min(a.len(), b.len())`.
#[must_use]
pub fn vector_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

/// Multiply every element of `v` by `scalar`.
#[must_use]
pub fn vector_scale(v: &[f32], scalar: f32) -> Vec<f32> {
    v.iter().map(|x| x * scalar).collect()
}

/// Dense matrix multiplication: C = A × B.
///
/// - `a`: row-major matrix of shape `(rows_a, cols_a)`
/// - `b`: row-major matrix of shape `(cols_a, cols_b)`
/// - Returns a row-major matrix of shape `(rows_a, cols_b)`
///
/// # Panics
///
/// Panics in debug mode if `a.len() != rows_a * cols_a` or
/// `b.len() != cols_a * cols_b`.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn matrix_multiply(
    a: &[f32],
    b: &[f32],
    rows_a: usize,
    cols_a: usize,
    cols_b: usize,
) -> Vec<f32> {
    debug_assert_eq!(a.len(), rows_a * cols_a);
    debug_assert_eq!(b.len(), cols_a * cols_b);

    let mut out = vec![0.0_f32; rows_a * cols_b];
    for i in 0..rows_a {
        for k in 0..cols_a {
            let a_ik = a[i * cols_a + k];
            for j in 0..cols_b {
                out[i * cols_b + j] += a_ik * b[k * cols_b + j];
            }
        }
    }
    out
}

/// Compute the inclusive prefix (cumulative) sum of `data`.
#[must_use]
pub fn prefix_sum(data: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(data.len());
    let mut acc = 0.0_f32;
    for &v in data {
        acc += v;
        out.push(acc);
    }
    out
}

/// Clamp every element of `data` to [`min`, `max`] in-place.
pub fn clamp_slice(data: &mut [f32], min: f32, max: f32) {
    for v in data.iter_mut() {
        *v = v.clamp(min, max);
    }
}

/// Element-wise absolute difference: |a\[i\] - b\[i\]|.
///
/// Output length equals `min(a.len(), b.len())`.
#[must_use]
pub fn abs_diff(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).collect()
}

/// Compute the sum of squares of all elements in `data`.
#[must_use]
pub fn sum_squares(data: &[f32]) -> f32 {
    data.iter().map(|x| x * x).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    #[test]
    fn test_dot_product_basic() {
        let a = vec![1.0_f32, 2.0, 3.0];
        let b = vec![4.0_f32, 5.0, 6.0];
        assert!(approx_eq(dot_product(&a, &b), 32.0)); // 4+10+18
    }

    #[test]
    fn test_dot_product_empty() {
        assert!(approx_eq(dot_product(&[], &[]), 0.0));
    }

    #[test]
    fn test_dot_product_orthogonal() {
        let a = vec![1.0_f32, 0.0];
        let b = vec![0.0_f32, 1.0];
        assert!(approx_eq(dot_product(&a, &b), 0.0));
    }

    #[test]
    fn test_vector_add() {
        let a = vec![1.0_f32, 2.0, 3.0];
        let b = vec![4.0_f32, 5.0, 6.0];
        let c = vector_add(&a, &b);
        assert_eq!(c, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_vector_scale() {
        let v = vec![1.0_f32, -2.0, 3.0];
        let out = vector_scale(&v, 2.0);
        assert_eq!(out, vec![2.0, -4.0, 6.0]);
    }

    #[test]
    fn test_matrix_multiply_identity() {
        // I * I = I  (2x2)
        let i = vec![1.0_f32, 0.0, 0.0, 1.0];
        let out = matrix_multiply(&i, &i, 2, 2, 2);
        assert!(approx_eq(out[0], 1.0));
        assert!(approx_eq(out[1], 0.0));
        assert!(approx_eq(out[2], 0.0));
        assert!(approx_eq(out[3], 1.0));
    }

    #[test]
    fn test_matrix_multiply_basic() {
        // [1,2] * [[5],[6]] = [17]
        let a = vec![1.0_f32, 2.0];
        let b = vec![5.0_f32, 6.0];
        let out = matrix_multiply(&a, &b, 1, 2, 1);
        assert!(approx_eq(out[0], 17.0));
    }

    #[test]
    fn test_prefix_sum() {
        let data = vec![1.0_f32, 2.0, 3.0, 4.0];
        let out = prefix_sum(&data);
        assert_eq!(out, vec![1.0, 3.0, 6.0, 10.0]);
    }

    #[test]
    fn test_prefix_sum_empty() {
        let out = prefix_sum(&[]);
        assert!(out.is_empty());
    }

    #[test]
    fn test_clamp_slice() {
        let mut data = vec![-5.0_f32, 0.5, 2.0, -0.1, 1.0];
        clamp_slice(&mut data, 0.0, 1.0);
        assert_eq!(data, vec![0.0, 0.5, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_abs_diff() {
        let a = vec![3.0_f32, 1.0, 5.0];
        let b = vec![1.0_f32, 4.0, 5.0];
        let out = abs_diff(&a, &b);
        assert_eq!(out, vec![2.0, 3.0, 0.0]);
    }

    #[test]
    fn test_sum_squares() {
        let data = vec![1.0_f32, 2.0, 3.0];
        assert!(approx_eq(sum_squares(&data), 14.0)); // 1+4+9
    }

    #[test]
    fn test_sum_squares_empty() {
        assert!(approx_eq(sum_squares(&[]), 0.0));
    }
}
