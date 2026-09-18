//! Integration tests for SIMD-accelerated matrix multiplication
//!
//! These tests verify the blocked GEMM implementation works correctly
//! and integrates properly with the rest of the codebase.

use scirs2_core::ndarray::Array2;
use scirs2_core::simd_ops::functions::SimdUnifiedOps;
use scirs2_core::simd_ops::matmul::{simd_dot_product_f32, simd_matrix_multiply_f32};

#[test]
fn test_simd_gemm_integration() {
    // Test that simd_gemm uses the optimized path for large matrices
    let n = 128;

    let a = Array2::<f32>::from_elem((n, n), 1.0);
    let b = Array2::<f32>::from_elem((n, n), 2.0);
    let mut c = Array2::<f32>::zeros((n, n));

    // Call through the SimdUnifiedOps trait
    f32::simd_gemm(1.0, &a.view(), &b.view(), 0.0, &mut c);

    // Each element should be 2*n (sum of 1*2 for n iterations)
    let expected = 2.0 * n as f32;
    for val in c.iter() {
        assert!(
            (*val - expected).abs() < 1e-2,
            "Expected {}, got {}",
            expected,
            val
        );
    }
}

#[test]
fn test_simd_dot_product_integration() {
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let b = vec![6.0, 7.0, 8.0, 9.0, 10.0];

    let result = simd_dot_product_f32(&a, &b);

    // 1*6 + 2*7 + 3*8 + 4*9 + 5*10 = 6 + 14 + 24 + 36 + 50 = 130
    assert!((result - 130.0).abs() < 1e-5);
}

#[test]
fn test_matrix_multiply_correctness() {
    // Test correctness for various matrix sizes
    let sizes = vec![4, 8, 16, 32, 64, 128];

    for n in sizes {
        let a = vec![1.0f32; n * n];
        let b = vec![2.0f32; n * n];
        let mut c = vec![0.0f32; n * n];

        simd_matrix_multiply_f32(n, n, n, 1.0, &a, &b, 0.0, &mut c);

        let expected = 2.0 * n as f32;
        for val in &c {
            assert!(
                (*val - expected).abs() < 1e-2,
                "Size {}: Expected {}, got {}",
                n,
                expected,
                val
            );
        }
    }
}

#[test]
fn test_matrix_multiply_identity() {
    let n = 64;

    // A = sequential values
    let a: Vec<f32> = (0..n * n).map(|i| i as f32).collect();

    // B = identity matrix
    let mut b = vec![0.0f32; n * n];
    for i in 0..n {
        b[i * n + i] = 1.0;
    }

    let mut c = vec![0.0f32; n * n];

    simd_matrix_multiply_f32(n, n, n, 1.0, &a, &b, 0.0, &mut c);

    // C should equal A (since B is identity)
    for i in 0..n * n {
        assert!(
            (c[i] - a[i]).abs() < 1e-4,
            "Mismatch at index {}: expected {}, got {}",
            i,
            a[i],
            c[i]
        );
    }
}

#[test]
fn test_matrix_multiply_transpose() {
    // Test A * A^T
    let m = 32;
    let k = 32;

    let a: Vec<f32> = (0..m * k).map(|i| (i % 10) as f32).collect();

    // Compute A^T
    let mut a_t = vec![0.0f32; k * m];
    for i in 0..m {
        for j in 0..k {
            a_t[j * m + i] = a[i * k + j];
        }
    }

    let mut c = vec![0.0f32; m * m];

    simd_matrix_multiply_f32(m, k, m, 1.0, &a, &a_t, 0.0, &mut c);

    // Verify C is symmetric (since C = A * A^T)
    for i in 0..m {
        for j in i..m {
            assert!(
                (c[i * m + j] - c[j * m + i]).abs() < 1e-4,
                "Matrix is not symmetric at ({}, {})",
                i,
                j
            );
        }
    }
}

#[test]
fn test_rectangular_matrices() {
    // Test C[3x5] = A[3x4] * B[4x5]
    let m = 3;
    let k = 4;
    let n = 5;

    let a: Vec<f32> = (1..=12).map(|i| i as f32).collect();
    let b: Vec<f32> = (1..=20).map(|i| i as f32).collect();
    let mut c = vec![0.0f32; m * n];

    simd_matrix_multiply_f32(m, k, n, 1.0, &a, &b, 0.0, &mut c);

    // Manually verify first element: C[0,0] = sum(A[0,k] * B[k,0])
    // = 1*1 + 2*6 + 3*11 + 4*16 = 1 + 12 + 33 + 64 = 110
    assert!(
        (c[0] - 110.0).abs() < 1e-4,
        "C[0,0] expected 110.0, got {}",
        c[0]
    );
}

#[test]
fn test_alpha_beta_scaling() {
    let m = 4;
    let k = 4;
    let n = 4;

    let a = vec![2.0f32; m * k];
    let b = vec![3.0f32; k * n];
    let mut c = vec![1.0f32; m * n];

    // C = 2.0 * C + 0.5 * A * B
    // A*B gives k*2*3 = 4*6 = 24 per element
    // 2.0 * 1.0 + 0.5 * 24 = 2.0 + 12.0 = 14.0
    simd_matrix_multiply_f32(m, k, n, 0.5, &a, &b, 2.0, &mut c);

    for val in &c {
        assert!((*val - 14.0).abs() < 1e-4, "Expected 14.0, got {}", val);
    }
}

#[test]
fn test_zero_beta() {
    let n = 32;

    let a = vec![1.0f32; n * n];
    let b = vec![1.0f32; n * n];
    let mut c = vec![999.0f32; n * n]; // Initialize with non-zero values

    // beta=0 should zero out C before computing
    simd_matrix_multiply_f32(n, n, n, 1.0, &a, &b, 0.0, &mut c);

    let expected = n as f32;
    for val in &c {
        assert!(
            (*val - expected).abs() < 1e-3,
            "Expected {}, got {}",
            expected,
            val
        );
    }
}

#[test]
fn test_performance_benchmark() {
    // Simple performance check: large matrix should complete in reasonable time
    let n = 256;

    let a = vec![1.0f32; n * n];
    let b = vec![1.0f32; n * n];
    let mut c = vec![0.0f32; n * n];

    let start = std::time::Instant::now();
    simd_matrix_multiply_f32(n, n, n, 1.0, &a, &b, 0.0, &mut c);
    let elapsed = start.elapsed();

    // Should complete in under 500ms (generous for unoptimized test builds)
    assert!(
        elapsed.as_millis() < 500,
        "Matrix multiply took too long: {:?}",
        elapsed
    );

    // Verify correctness
    let expected = n as f32;
    for val in &c {
        assert!(
            (*val - expected).abs() < 1e-2,
            "Expected {}, got {}",
            expected,
            val
        );
    }
}
