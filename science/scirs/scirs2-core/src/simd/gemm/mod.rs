//! High-performance blocked GEMM (General Matrix Multiply) implementation
//!
//! This module provides cache-optimized matrix multiplication using a 3-level
//! blocking strategy inspired by GotoBLAS and BLIS. It achieves 10-50x speedup
//! over naive implementations for matrices >= 256x256.
//!
//! # Features
//!
//! - **3-level cache blocking**: Optimized for L1, L2, and L3 cache hierarchies
//! - **Platform-specific micro-kernels**: AVX-512, AVX2, SSE, NEON optimizations
//! - **Memory packing**: Cache-friendly data layouts for peak performance
//! - **Adaptive thresholds**: Falls back to simple loops for small matrices
//! - **FMA instructions**: Single-cycle multiply-add on supported platforms
//!
//! # Performance
//!
//! Expected speedups for square matrices (compared to naive triple loop):
//!
//! | Size  | Speedup | Notes                              |
//! |-------|---------|-------------------------------------|
//! | 64    | 2-3x    | Small, benefits from SIMD only     |
//! | 128   | 5-8x    | Starts benefiting from blocking    |
//! | 256   | 10-15x  | Full blocking benefits visible     |
//! | 512   | 15-25x  | Optimal cache utilization          |
//! | 1024+ | 20-30x  | Peak performance, all optimizations|
//!
//! # Example
//!
//! ```rust
//! use scirs2_core::simd::gemm::{MatMulConfig, blocked_gemm_f32};
//!
//! let m = 256;
//! let k = 256;
//! let n = 256;
//!
//! let a = vec![1.0f32; m * k];
//! let b = vec![1.0f32; k * n];
//! let mut c = vec![0.0f32; m * n];
//!
//! let config = MatMulConfig::for_f32();
//!
//! unsafe {
//!     blocked_gemm_f32(
//!         m, k, n,
//!         1.0,           // alpha
//!         a.as_ptr(), k, // A matrix, lda
//!         b.as_ptr(), n, // B matrix, ldb
//!         0.0,           // beta
//!         c.as_mut_ptr(), n, // C matrix, ldc
//!         &config,
//!     );
//! }
//! ```
//!
//! # Algorithm Structure
//!
//! The implementation uses 5 nested loops with cache-aware blocking:
//!
//! ```text
//! for jc = 0:NC:N              // L3: columns of B
//!     for pc = 0:KC:K          // L2: inner dimension
//!         for ic = 0:MC:M      // L3: rows of A
//!             pack B[KC x NR]
//!             for jr = 0:NR:NC  // Micro-panel width
//!                 for ir = 0:MR:MC  // Micro-panel height
//!                     micro_kernel: C[MR x NR] += A[MR x KC] * B[KC x NR]
//! ```
//!
//! # References
//!
//! - Goto, K., & Van De Geijn, R. (2008). "Anatomy of high-performance matrix multiplication"
//! - Van Zee, F. G., & Van De Geijn, R. A. (2015). "BLIS: A framework for rapidly instantiating BLAS functionality"

pub mod blocked;
pub mod config;
pub mod micro_kernels;
pub mod packing;

// Re-export public API
pub use blocked::{blocked_gemm_f32, blocked_gemm_f64, should_use_blocked};
pub use config::MatMulConfig;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemm_config_f32() {
        let config = MatMulConfig::for_f32();

        // Verify config is reasonable
        assert!(config.mc > 0);
        assert!(config.kc > 0);
        assert!(config.nc > 0);
        assert!(config.mr > 0);
        assert!(config.nr > 0);

        // Verify micro-kernel dimensions are multiples of 4 (SIMD-friendly)
        assert_eq!(config.mr % 4, 0);
        assert_eq!(config.nr % 4, 0);
    }

    #[test]
    fn test_gemm_config_f64() {
        let config = MatMulConfig::for_f64();

        // Verify config is reasonable
        assert!(config.mc > 0);
        assert!(config.kc > 0);
        assert!(config.nc > 0);
        assert!(config.mr > 0);
        assert!(config.nr > 0);

        // f64 should have smaller blocking sizes than f32
        let config_f32 = MatMulConfig::for_f32();
        assert!(config.mc <= config_f32.mc);
        assert!(config.kc <= config_f32.kc);
    }

    #[test]
    fn test_identity_multiply() {
        // Test C = A * I where I is identity
        let n = 64;
        let a: Vec<f32> = (0..n * n).map(|i| i as f32).collect();

        // Identity matrix
        let mut b = vec![0.0f32; n * n];
        for i in 0..n {
            b[i * n + i] = 1.0;
        }

        let mut c = vec![0.0f32; n * n];
        let config = MatMulConfig::for_f32();

        unsafe {
            blocked_gemm_f32(
                n,
                n,
                n,
                1.0,
                a.as_ptr(),
                n,
                b.as_ptr(),
                n,
                0.0,
                c.as_mut_ptr(),
                n,
                &config,
            );
        }

        // C should equal A
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
    fn test_rectangular_multiply() {
        // Test non-square matrices: C[3x5] = A[3x4] * B[4x5]
        let m = 3;
        let k = 4;
        let n = 5;

        let a: Vec<f32> = (1..=12).map(|i| i as f32).collect();
        let b: Vec<f32> = (1..=20).map(|i| i as f32).collect();
        let mut c = vec![0.0f32; m * n];

        let config = MatMulConfig::for_f32();

        unsafe {
            blocked_gemm_f32(
                m,
                k,
                n,
                1.0,
                a.as_ptr(),
                k,
                b.as_ptr(),
                n,
                0.0,
                c.as_mut_ptr(),
                n,
                &config,
            );
        }

        // Manually computed expected result for first row
        // C[0,0] = 1*1 + 2*6 + 3*11 + 4*16 = 1+12+33+64 = 110
        assert!(
            (c[0] - 110.0).abs() < 1e-4,
            "C[0,0] expected 110.0, got {}",
            c[0]
        );

        // C[0,1] = 1*2 + 2*7 + 3*12 + 4*17 = 2+14+36+68 = 120
        assert!(
            (c[1] - 120.0).abs() < 1e-4,
            "C[0,1] expected 120.0, got {}",
            c[1]
        );
    }

    #[test]
    fn test_gemm_with_strided_access() {
        // Test with non-contiguous matrices (stride > actual width)
        let m = 4;
        let k = 4;
        let n = 4;

        let lda = 8; // A has extra columns
        let ldb = 8; // B has extra columns
        let ldc = 8; // C has extra columns

        let mut a = vec![0.0f32; m * lda];
        let mut b = vec![0.0f32; k * ldb];
        let mut c = vec![0.0f32; m * ldc];

        // Fill A with identity in first 4x4 block
        for i in 0..m {
            a[i * lda + i] = 1.0;
        }

        // Fill B with identity in first 4x4 block
        for i in 0..k {
            b[i * ldb + i] = 1.0;
        }

        let config = MatMulConfig::for_f32();

        unsafe {
            blocked_gemm_f32(
                m,
                k,
                n,
                1.0,
                a.as_ptr(),
                lda,
                b.as_ptr(),
                ldb,
                0.0,
                c.as_mut_ptr(),
                ldc,
                &config,
            );
        }

        // Result should be identity in first 4x4 block
        for i in 0..m {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                let actual = c[i * ldc + j];
                assert!(
                    (actual - expected).abs() < 1e-5,
                    "Mismatch at ({},{}): expected {}, got {}",
                    i,
                    j,
                    expected,
                    actual
                );
            }
        }
    }

    #[test]
    fn test_large_matrix_correctness() {
        // Test a larger matrix to ensure blocked algorithm is correct
        let n = 200;

        // A = all ones
        let a = vec![1.0f32; n * n];

        // B = all twos
        let b = vec![2.0f32; n * n];

        let mut c = vec![0.0f32; n * n];
        let config = MatMulConfig::for_f32();

        unsafe {
            blocked_gemm_f32(
                n,
                n,
                n,
                1.0,
                a.as_ptr(),
                n,
                b.as_ptr(),
                n,
                0.0,
                c.as_mut_ptr(),
                n,
                &config,
            );
        }

        // Each element should be 2*n (sum of 1*2 for n iterations)
        let expected = 2.0 * n as f32;
        for i in 0..n * n {
            assert!(
                (c[i] - expected).abs() < 1e-2,
                "Mismatch at index {}: expected {}, got {}",
                i,
                expected,
                c[i]
            );
        }
    }

    // ==================== f64 GEMM Tests ====================

    #[test]
    fn test_identity_multiply_f64() {
        // Test C = A * I where I is identity
        let n = 64;
        let a: Vec<f64> = (0..n * n).map(|i| i as f64).collect();

        // Identity matrix
        let mut b = vec![0.0f64; n * n];
        for i in 0..n {
            b[i * n + i] = 1.0;
        }

        let mut c = vec![0.0f64; n * n];
        let config = MatMulConfig::for_f64();

        unsafe {
            blocked_gemm_f64(
                n,
                n,
                n,
                1.0,
                a.as_ptr(),
                n,
                b.as_ptr(),
                n,
                0.0,
                c.as_mut_ptr(),
                n,
                &config,
            );
        }

        // C should equal A
        for i in 0..n * n {
            assert!(
                (c[i] - a[i]).abs() < 1e-10,
                "Mismatch at index {}: expected {}, got {}",
                i,
                a[i],
                c[i]
            );
        }
    }

    #[test]
    fn test_rectangular_multiply_f64() {
        // Test non-square matrices: C[3x5] = A[3x4] * B[4x5]
        let m = 3;
        let k = 4;
        let n = 5;

        let a: Vec<f64> = (1..=12).map(|i| i as f64).collect();
        let b: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let mut c = vec![0.0f64; m * n];

        let config = MatMulConfig::for_f64();

        unsafe {
            blocked_gemm_f64(
                m,
                k,
                n,
                1.0,
                a.as_ptr(),
                k,
                b.as_ptr(),
                n,
                0.0,
                c.as_mut_ptr(),
                n,
                &config,
            );
        }

        // Manually computed expected result for first row
        // C[0,0] = 1*1 + 2*6 + 3*11 + 4*16 = 1+12+33+64 = 110
        assert!(
            (c[0] - 110.0).abs() < 1e-10,
            "C[0,0] expected 110.0, got {}",
            c[0]
        );

        // C[0,1] = 1*2 + 2*7 + 3*12 + 4*17 = 2+14+36+68 = 120
        assert!(
            (c[1] - 120.0).abs() < 1e-10,
            "C[0,1] expected 120.0, got {}",
            c[1]
        );
    }

    #[test]
    fn test_gemm_with_alpha_beta_f64() {
        let m = 2;
        let k = 2;
        let n = 2;

        let a: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let b: Vec<f64> = vec![5.0, 6.0, 7.0, 8.0];
        let mut c = vec![1.0, 1.0, 1.0, 1.0];

        let config = MatMulConfig::for_f64();

        unsafe {
            blocked_gemm_f64(
                m,
                k,
                n,
                2.0, // alpha
                a.as_ptr(),
                k,
                b.as_ptr(),
                n,
                3.0, // beta
                c.as_mut_ptr(),
                n,
                &config,
            );
        }

        // C = beta*C + alpha*A*B
        // A*B = [[19, 22], [43, 50]]
        // C = 3*[1,1,1,1] + 2*[[19,22],[43,50]]
        // C = [3,3,3,3] + [38,44,86,100] = [41,47,89,103]

        let expected = [41.0, 47.0, 89.0, 103.0];
        for i in 0..4 {
            assert!(
                (c[i] - expected[i]).abs() < 1e-10,
                "Mismatch at index {}: expected {}, got {}",
                i,
                expected[i],
                c[i]
            );
        }
    }

    #[test]
    fn test_large_matrix_correctness_f64() {
        // Test a larger matrix to ensure blocked algorithm is correct
        let n = 128;

        // A = all ones
        let a = vec![1.0f64; n * n];

        // B = all twos
        let b = vec![2.0f64; n * n];

        let mut c = vec![0.0f64; n * n];
        let config = MatMulConfig::for_f64();

        unsafe {
            blocked_gemm_f64(
                n,
                n,
                n,
                1.0,
                a.as_ptr(),
                n,
                b.as_ptr(),
                n,
                0.0,
                c.as_mut_ptr(),
                n,
                &config,
            );
        }

        // Each element should be 2*n (sum of 1*2 for n iterations)
        let expected = 2.0 * n as f64;
        for i in 0..n * n {
            assert!(
                (c[i] - expected).abs() < 1e-8,
                "Mismatch at index {}: expected {}, got {}",
                i,
                expected,
                c[i]
            );
        }
    }

    #[test]
    fn test_gemm_with_strided_access_f64() {
        // Test with non-contiguous matrices (stride > actual width)
        let m = 4;
        let k = 4;
        let n = 4;

        let lda = 8; // A has extra columns
        let ldb = 8; // B has extra columns
        let ldc = 8; // C has extra columns

        let mut a = vec![0.0f64; m * lda];
        let mut b = vec![0.0f64; k * ldb];
        let mut c = vec![0.0f64; m * ldc];

        // Fill A with identity in first 4x4 block
        for i in 0..m {
            a[i * lda + i] = 1.0;
        }

        // Fill B with identity in first 4x4 block
        for i in 0..k {
            b[i * ldb + i] = 1.0;
        }

        let config = MatMulConfig::for_f64();

        unsafe {
            blocked_gemm_f64(
                m,
                k,
                n,
                1.0,
                a.as_ptr(),
                lda,
                b.as_ptr(),
                ldb,
                0.0,
                c.as_mut_ptr(),
                ldc,
                &config,
            );
        }

        // Result should be identity in first 4x4 block
        for i in 0..m {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                let actual = c[i * ldc + j];
                assert!(
                    (actual - expected).abs() < 1e-10,
                    "Mismatch at ({},{}): expected {}, got {}",
                    i,
                    j,
                    expected,
                    actual
                );
            }
        }
    }
}
