//! SIMD vector norm integration tests
//!
//! This module tests the SIMD-accelerated vector norm implementations
//! in scirs2-linalg, ensuring correctness and performance.

use scirs2_core::ndarray::{array, Array1};
use scirs2_linalg::vector_norm_simd;

// ============================================================================
// L1 Norm Tests (Manhattan norm)
// ============================================================================

#[test]
fn test_simd_norm_l1_f64_basic() {
    let x = array![3.0f64, -4.0, 5.0];

    let norm = vector_norm_simd(&x.view(), 1).expect("Test: operation failed");

    // L1 norm: |3| + |-4| + |5| = 3 + 4 + 5 = 12
    let expected = 12.0f64;
    assert!(
        (norm - expected).abs() < 1e-10,
        "Got {}, expected {}",
        norm,
        expected
    );
}

#[test]
fn test_simd_norm_l1_f32_basic() {
    let x = array![3.0f32, -4.0, 5.0];

    let norm = vector_norm_simd(&x.view(), 1).expect("Test: operation failed");

    let expected = 12.0f32;
    assert!(
        (norm - expected).abs() < 1e-6,
        "Got {}, expected {}",
        norm,
        expected
    );
}

#[test]
fn test_simd_norm_l1_f64_large() {
    // Test with large array to ensure SIMD path is used
    let x: Array1<f64> = Array1::from_vec(
        (0..1000)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect(),
    );

    let norm = vector_norm_simd(&x.view(), 1).expect("Test: operation failed");

    // 1000 elements with absolute value 1
    let expected = 1000.0f64;
    assert!(
        (norm - expected).abs() < 1e-10,
        "Got {}, expected {}",
        norm,
        expected
    );
}

#[test]
fn test_simd_norm_l1_f32_large() {
    let x: Array1<f32> = Array1::from_vec(
        (0..1000)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect(),
    );

    let norm = vector_norm_simd(&x.view(), 1).expect("Test: operation failed");

    let expected = 1000.0f32;
    assert!(
        (norm - expected).abs() < 1e-4,
        "Got {}, expected {}",
        norm,
        expected
    );
}

// ============================================================================
// L2 Norm Tests (Euclidean norm)
// ============================================================================

#[test]
fn test_simd_norm_l2_f64_basic() {
    let x = array![3.0f64, 4.0];

    let norm = vector_norm_simd(&x.view(), 2).expect("Test: operation failed");

    // L2 norm: sqrt(3^2 + 4^2) = sqrt(9 + 16) = sqrt(25) = 5
    let expected = 5.0f64;
    assert!(
        (norm - expected).abs() < 1e-10,
        "Got {}, expected {}",
        norm,
        expected
    );
}

#[test]
fn test_simd_norm_l2_f32_basic() {
    let x = array![3.0f32, 4.0];

    let norm = vector_norm_simd(&x.view(), 2).expect("Test: operation failed");

    let expected = 5.0f32;
    assert!(
        (norm - expected).abs() < 1e-6,
        "Got {}, expected {}",
        norm,
        expected
    );
}

#[test]
fn test_simd_norm_l2_f64_large() {
    // Test with large array to ensure SIMD path is used
    let x: Array1<f64> = Array1::from_vec((0..1000).map(|_| 1.0).collect());

    let norm = vector_norm_simd(&x.view(), 2).expect("Test: operation failed");

    // L2 norm of 1000 ones: sqrt(1000)
    let expected = 1000.0f64.sqrt();
    assert!(
        (norm - expected).abs() < 1e-10,
        "Got {}, expected {}",
        norm,
        expected
    );
}

#[test]
fn test_simd_norm_l2_f32_large() {
    let x: Array1<f32> = Array1::from_vec((0..1000).map(|_| 1.0).collect());

    let norm = vector_norm_simd(&x.view(), 2).expect("Test: operation failed");

    let expected = 1000.0f32.sqrt();
    assert!(
        (norm - expected).abs() < 1e-4,
        "Got {}, expected {}",
        norm,
        expected
    );
}

// ============================================================================
// L∞ Norm Tests (Chebyshev norm / maximum absolute value)
// ============================================================================

#[test]
fn test_simd_norm_linf_f64_basic() {
    let x = array![3.0f64, -7.0, 5.0, 2.0];

    let norm = vector_norm_simd(&x.view(), usize::MAX).expect("Test: operation failed");

    // L∞ norm: max(|3|, |-7|, |5|, |2|) = max(3, 7, 5, 2) = 7
    let expected = 7.0f64;
    assert!(
        (norm - expected).abs() < 1e-10,
        "Got {}, expected {}",
        norm,
        expected
    );
}

#[test]
fn test_simd_norm_linf_f32_basic() {
    let x = array![3.0f32, -7.0, 5.0, 2.0];

    let norm = vector_norm_simd(&x.view(), usize::MAX).expect("Test: operation failed");

    let expected = 7.0f32;
    assert!(
        (norm - expected).abs() < 1e-6,
        "Got {}, expected {}",
        norm,
        expected
    );
}

#[test]
fn test_simd_norm_linf_f64_large() {
    // Test with large array to ensure SIMD path is used
    let x: Array1<f64> = Array1::from_vec(
        (0..1000)
            .map(|i| if i == 500 { 100.0 } else { 1.0 })
            .collect(),
    );

    let norm = vector_norm_simd(&x.view(), usize::MAX).expect("Test: operation failed");

    // Maximum value is 100.0 at index 500
    let expected = 100.0f64;
    assert!(
        (norm - expected).abs() < 1e-10,
        "Got {}, expected {}",
        norm,
        expected
    );
}

#[test]
fn test_simd_norm_linf_f32_large() {
    let x: Array1<f32> = Array1::from_vec(
        (0..1000)
            .map(|i| if i == 500 { 100.0 } else { 1.0 })
            .collect(),
    );

    let norm = vector_norm_simd(&x.view(), usize::MAX).expect("Test: operation failed");

    let expected = 100.0f32;
    assert!(
        (norm - expected).abs() < 1e-4,
        "Got {}, expected {}",
        norm,
        expected
    );
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_simd_norm_l1_negative_values() {
    let x = array![-1.0f64, -2.0, -3.0, -4.0];

    let norm = vector_norm_simd(&x.view(), 1).expect("Test: operation failed");

    // L1 norm: |-1| + |-2| + |-3| + |-4| = 1 + 2 + 3 + 4 = 10
    let expected = 10.0f64;
    assert!((norm - expected).abs() < 1e-10);
}

#[test]
fn test_simd_norm_l2_negative_values() {
    let x = array![-3.0f64, 4.0];

    let norm = vector_norm_simd(&x.view(), 2).expect("Test: operation failed");

    // L2 norm: sqrt((-3)^2 + 4^2) = sqrt(9 + 16) = 5
    let expected = 5.0f64;
    assert!((norm - expected).abs() < 1e-10);
}

#[test]
fn test_simd_norm_linf_negative_maximum() {
    let x = array![1.0f64, -10.0, 3.0, -5.0];

    let norm = vector_norm_simd(&x.view(), usize::MAX).expect("Test: operation failed");

    // L∞ norm: max(|1|, |-10|, |3|, |-5|) = max(1, 10, 3, 5) = 10
    let expected = 10.0f64;
    assert!((norm - expected).abs() < 1e-10);
}

#[test]
fn test_simd_norm_l1_zero_vector() {
    let x = array![0.0f64, 0.0, 0.0];

    let norm = vector_norm_simd(&x.view(), 1).expect("Test: operation failed");

    assert!(norm.abs() < 1e-10, "L1 norm of zero vector should be 0");
}

#[test]
fn test_simd_norm_l2_zero_vector() {
    let x = array![0.0f64, 0.0, 0.0];

    let norm = vector_norm_simd(&x.view(), 2).expect("Test: operation failed");

    assert!(norm.abs() < 1e-10, "L2 norm of zero vector should be 0");
}

#[test]
fn test_simd_norm_linf_zero_vector() {
    let x = array![0.0f64, 0.0, 0.0];

    let norm = vector_norm_simd(&x.view(), usize::MAX).expect("Test: operation failed");

    assert!(norm.abs() < 1e-10, "L∞ norm of zero vector should be 0");
}

#[test]
fn test_simd_norm_l1_single_element() {
    let x = array![5.0f64];

    let norm = vector_norm_simd(&x.view(), 1).expect("Test: operation failed");

    let expected = 5.0f64;
    assert!((norm - expected).abs() < 1e-10);
}

#[test]
fn test_simd_norm_l2_single_element() {
    let x = array![5.0f64];

    let norm = vector_norm_simd(&x.view(), 2).expect("Test: operation failed");

    let expected = 5.0f64;
    assert!((norm - expected).abs() < 1e-10);
}

#[test]
fn test_simd_norm_linf_single_element() {
    let x = array![5.0f64];

    let norm = vector_norm_simd(&x.view(), usize::MAX).expect("Test: operation failed");

    let expected = 5.0f64;
    assert!((norm - expected).abs() < 1e-10);
}

// ============================================================================
// Very Large Array Tests (10K+ elements)
// ============================================================================

#[test]
fn test_simd_norm_l1_very_large() {
    // 10K elements
    let x: Array1<f64> = Array1::from_vec((0..10000).map(|i| (i as f64) * 0.1).collect());

    let norm = vector_norm_simd(&x.view(), 1).expect("Test: operation failed");

    // L1 norm: sum of |i * 0.1| for i in 0..10000 = 0.1 * sum(0..10000) = 0.1 * 49995000 = 4999500
    let expected = 0.1 * (9999.0 * 10000.0 / 2.0);
    assert!(
        (norm - expected).abs() / expected < 1e-10,
        "Got {}, expected {}",
        norm,
        expected
    );
}

#[test]
fn test_simd_norm_l2_very_large() {
    // 10K elements with value 1.0
    let x: Array1<f64> = Array1::from_vec((0..10000).map(|_| 1.0).collect());

    let norm = vector_norm_simd(&x.view(), 2).expect("Test: operation failed");

    // L2 norm: sqrt(10000)
    let expected = 100.0f64;
    assert!(
        (norm - expected).abs() < 1e-10,
        "Got {}, expected {}",
        norm,
        expected
    );
}

#[test]
fn test_simd_norm_linf_very_large() {
    // 10K elements with one large value
    let x: Array1<f64> = Array1::from_vec(
        (0..10000)
            .map(|i| if i == 5000 { 999.0 } else { 1.0 })
            .collect(),
    );

    let norm = vector_norm_simd(&x.view(), usize::MAX).expect("Test: operation failed");

    let expected = 999.0f64;
    assert!(
        (norm - expected).abs() < 1e-10,
        "Got {}, expected {}",
        norm,
        expected
    );
}

// ============================================================================
// Norm Comparison Tests
// ============================================================================

#[test]
fn test_simd_norm_relationship() {
    // For the same vector, L∞ <= L2 <= L1 (in general)
    let x = array![1.0f64, 1.0, 1.0, 1.0];

    let norm_l1 = vector_norm_simd(&x.view(), 1).expect("Test: operation failed");
    let norm_l2 = vector_norm_simd(&x.view(), 2).expect("Test: operation failed");
    let norm_linf = vector_norm_simd(&x.view(), usize::MAX).expect("Test: operation failed");

    // For vector of all ones:
    // L∞ = 1, L2 = sqrt(4) = 2, L1 = 4
    assert_eq!(norm_linf, 1.0);
    assert_eq!(norm_l2, 2.0);
    assert_eq!(norm_l1, 4.0);

    assert!(norm_linf <= norm_l2);
    assert!(norm_l2 <= norm_l1);
}

// ============================================================================
// Frobenius Norm Tests (Matrix)
// ============================================================================

#[test]
fn test_simd_frobenius_norm_f64_basic() {
    use scirs2_core::ndarray::array;
    use scirs2_linalg::matrix_norm_simd;

    let a = array![[1.0f64, 2.0], [3.0, 4.0]];
    let norm = matrix_norm_simd(&a.view(), "fro", None).expect("Test: operation failed");

    // Frobenius norm: sqrt(1² + 2² + 3² + 4²) = sqrt(30) ≈ 5.477225575051661
    let expected = 5.477225575051661f64;
    assert!(
        (norm - expected).abs() < 1e-10,
        "Got {}, expected {}",
        norm,
        expected
    );
}

#[test]
fn test_simd_frobenius_norm_f32_basic() {
    use scirs2_core::ndarray::array;
    use scirs2_linalg::matrix_norm_simd;

    let a = array![[1.0f32, 2.0], [3.0, 4.0]];
    let norm = matrix_norm_simd(&a.view(), "fro", None).expect("Test: operation failed");

    let expected = 5.477_226_f32;
    assert!(
        (norm - expected).abs() < 1e-5,
        "Got {}, expected {}",
        norm,
        expected
    );
}

#[test]
fn test_simd_frobenius_norm_large_matrix() {
    use scirs2_core::ndarray::Array2;
    use scirs2_linalg::matrix_norm_simd;

    // Test with large matrix to ensure SIMD path is used (>1000 elements)
    let a = Array2::<f64>::ones((40, 40)); // 1600 elements
    let norm = matrix_norm_simd(&a.view(), "fro", None).expect("Test: operation failed");

    // Frobenius norm: sqrt(1600 * 1²) = sqrt(1600) = 40.0
    let expected = 40.0f64;
    assert!(
        (norm - expected).abs() < 1e-10,
        "Got {}, expected {}",
        norm,
        expected
    );
}

#[test]
fn test_simd_frobenius_norm_zeros() {
    use scirs2_core::ndarray::Array2;
    use scirs2_linalg::matrix_norm_simd;

    let a = Array2::<f64>::zeros((5, 5));
    let norm = matrix_norm_simd(&a.view(), "fro", None).expect("Test: operation failed");

    assert!(
        norm.abs() < 1e-15,
        "Frobenius norm of zero matrix should be 0, got {}",
        norm
    );
}

#[test]
fn test_simd_frobenius_norm_identity() {
    use scirs2_core::ndarray::Array2;
    use scirs2_linalg::matrix_norm_simd;

    let a = Array2::<f64>::eye(5);
    let norm = matrix_norm_simd(&a.view(), "fro", None).expect("Test: operation failed");

    // Frobenius norm of 5x5 identity: sqrt(5 * 1²) = sqrt(5) ≈ 2.23606797749979
    let expected = 5.0f64.sqrt();
    assert!(
        (norm - expected).abs() < 1e-10,
        "Got {}, expected {}",
        norm,
        expected
    );
}

#[test]
fn test_simd_frobenius_norm_rectangular() {
    use scirs2_core::ndarray::Array2;
    use scirs2_linalg::matrix_norm_simd;

    let a = Array2::<f64>::ones((10, 20)); // 200 elements
    let norm = matrix_norm_simd(&a.view(), "fro", None).expect("Test: operation failed");

    // Frobenius norm: sqrt(200 * 1²) = sqrt(200) = 10*sqrt(2) ≈ 14.142135623730951
    let expected = 200.0f64.sqrt();
    assert!(
        (norm - expected).abs() < 1e-10,
        "Got {}, expected {}",
        norm,
        expected
    );
}

#[test]
fn test_simd_frobenius_vs_scalar() {
    use scirs2_core::ndarray::Array2;
    use scirs2_linalg::{matrix_norm, matrix_norm_simd};

    // Compare SIMD and scalar implementations
    let a = Array2::<f64>::from_shape_fn((30, 30), |(i, j)| (i * j) as f64 * 0.1);

    let norm_simd = matrix_norm_simd(&a.view(), "fro", None).expect("Test: operation failed");
    let norm_scalar = matrix_norm(&a.view(), "fro", None).expect("Test: operation failed");

    assert!(
        (norm_simd - norm_scalar).abs() < 1e-10,
        "SIMD and scalar implementations should match: SIMD={}, scalar={}",
        norm_simd,
        norm_scalar
    );
}
