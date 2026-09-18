//! 3-level cache-blocked GEMM algorithm
//!
//! This module implements the core blocked GEMM algorithm with 3-level cache
//! hierarchy optimization (L1, L2, L3). The algorithm is based on the GotoBLAS/BLIS
//! design with aggressive optimizations for modern CPUs.
//!
//! # Algorithm Structure
//!
//! ```text
//! for jc = 0:NC:N              // L3 cache: B[KC x NC]
//!     for pc = 0:KC:K          // L2 cache: A[MC x KC], B[KC x NC]
//!         for ic = 0:MC:M      // L3 cache: A[MC x KC]
//!             for jr = 0:NR:NC  // Pack B[KC x NR]
//!                 for ir = 0:MR:MC  // Micro-kernel: C[MR x NR] += A[MR x KC] * B[KC x NR]
//! ```
//!
//! # Performance Characteristics
//!
//! - Expected speedup: 10-30x over naive implementation for 256x256+ matrices
//! - Optimized for matrices >= 64x64
//! - Falls back to simple loops for small matrices

use super::config::MatMulConfig;
use super::micro_kernels::{micro_kernel_f32, micro_kernel_f64};
use super::packing::{pack_b_f32, pack_b_f32_fast, pack_b_f64_fast};

/// Blocked GEMM implementation for f32: C = beta*C + alpha*A*B
///
/// # Arguments
///
/// * `m` - Number of rows of A and C
/// * `k` - Number of columns of A and rows of B
/// * `n` - Number of columns of B and C
/// * `alpha` - Scalar multiplier for A*B
/// * `a` - Pointer to A matrix (m x k, row-major)
/// * `lda` - Leading dimension of A (typically k)
/// * `b` - Pointer to B matrix (k x n, row-major)
/// * `ldb` - Leading dimension of B (typically n)
/// * `beta` - Scalar multiplier for C
/// * `c` - Pointer to C matrix (m x n, row-major)
/// * `ldc` - Leading dimension of C (typically n)
/// * `config` - GEMM configuration parameters
///
/// # Safety
///
/// Caller must ensure:
/// - `a` points to valid memory with at least `m * lda` elements
/// - `b` points to valid memory with at least `k * ldb` elements
/// - `c` points to valid memory with at least `m * ldc` elements
/// - `lda >= k`, `ldb >= n`, `ldc >= n`
pub unsafe fn blocked_gemm_f32(
    m: usize,
    k: usize,
    n: usize,
    alpha: f32,
    a: *const f32,
    lda: usize,
    b: *const f32,
    ldb: usize,
    beta: f32,
    c: *mut f32,
    ldc: usize,
    config: &MatMulConfig,
) {
    // For very small matrices, use simple implementation
    if m < 32 || n < 32 || k < 32 {
        gemm_small_f32(m, k, n, alpha, a, lda, b, ldb, beta, c, ldc);
        return;
    }

    let mc = config.mc;
    let kc = config.kc;
    let nc = config.nc;
    let mr = config.mr;
    let nr = config.nr;

    // Allocate packing buffer for B
    // Size: KC x NR (one micro-panel of B)
    let b_buffer_size = kc * nr;
    let mut b_packed = vec![0.0f32; b_buffer_size];
    let b_packed_ptr = b_packed.as_mut_ptr();

    // Loop over N dimension (columns of B and C) with NC blocking
    let mut jc = 0;
    while jc < n {
        let nc_curr = (n - jc).min(nc);

        // Loop over K dimension (inner dimension) with KC blocking
        let mut pc = 0;
        while pc < k {
            let kc_curr = (k - pc).min(kc);

            // Loop over M dimension (rows of A and C) with MC blocking
            let mut ic = 0;
            while ic < m {
                let mc_curr = (m - ic).min(mc);

                // Loop over NR panels (micro-panel of B)
                let mut jr = 0;
                while jr < nc_curr {
                    let nr_curr = (nc_curr - jr).min(nr);

                    // Pack B[KC x NR] for this micro-panel
                    let b_panel = b.add(pc * ldb + jc + jr);
                    pack_b_f32_fast(kc_curr, nr_curr, b_panel, ldb, b_packed_ptr, config);

                    // Loop over MR panels (micro-panel of A)
                    let mut ir = 0;
                    while ir < mc_curr {
                        let mr_curr = (mc_curr - ir).min(mr);

                        // Get pointers to current micro-panels
                        let a_panel = a.add((ic + ir) * lda + pc);
                        let c_block = c.add((ic + ir) * ldc + jc + jr);

                        // Compute micro-kernel: C[MR x NR] += A[MR x KC] * B[KC x NR]
                        // Only apply beta on first iteration (pc == 0)
                        let beta_curr = if pc == 0 { beta } else { 1.0 };

                        if mr_curr == mr && nr_curr == nr && kc_curr == kc {
                            // Fast path: full micro-kernel
                            micro_kernel_f32(
                                kc_curr,
                                alpha,
                                a_panel,
                                b_packed_ptr,
                                beta_curr,
                                c_block,
                                ldc,
                                config,
                            );
                        } else {
                            // Edge case: partial micro-kernel
                            micro_kernel_f32_edge(
                                mr_curr,
                                nr_curr,
                                kc_curr,
                                alpha,
                                a_panel,
                                lda,
                                b_packed_ptr,
                                beta_curr,
                                c_block,
                                ldc,
                            );
                        }

                        ir += mr;
                    }

                    jr += nr;
                }

                ic += mc;
            }

            pc += kc;
        }

        jc += nc;
    }
}

/// Blocked GEMM implementation for f64: C = beta*C + alpha*A*B
///
/// # Arguments
///
/// * `m` - Number of rows of A and C
/// * `k` - Number of columns of A and rows of B
/// * `n` - Number of columns of B and C
/// * `alpha` - Scalar multiplier for A*B
/// * `a` - Pointer to A matrix (m x k, row-major)
/// * `lda` - Leading dimension of A (typically k)
/// * `b` - Pointer to B matrix (k x n, row-major)
/// * `ldb` - Leading dimension of B (typically n)
/// * `beta` - Scalar multiplier for C
/// * `c` - Pointer to C matrix (m x n, row-major)
/// * `ldc` - Leading dimension of C (typically n)
/// * `config` - GEMM configuration parameters
///
/// # Safety
///
/// Caller must ensure:
/// - `a` points to valid memory with at least `m * lda` elements
/// - `b` points to valid memory with at least `k * ldb` elements
/// - `c` points to valid memory with at least `m * ldc` elements
/// - `lda >= k`, `ldb >= n`, `ldc >= n`
pub unsafe fn blocked_gemm_f64(
    m: usize,
    k: usize,
    n: usize,
    alpha: f64,
    a: *const f64,
    lda: usize,
    b: *const f64,
    ldb: usize,
    beta: f64,
    c: *mut f64,
    ldc: usize,
    config: &MatMulConfig,
) {
    // For very small matrices, use simple implementation
    if m < 32 || n < 32 || k < 32 {
        gemm_small_f64(m, k, n, alpha, a, lda, b, ldb, beta, c, ldc);
        return;
    }

    let mc = config.mc;
    let kc = config.kc;
    let nc = config.nc;
    let mr = config.mr;
    let nr = config.nr;

    // Allocate packing buffer for B
    // Size: KC x NR (one micro-panel of B)
    let b_buffer_size = kc * nr;
    let mut b_packed = vec![0.0f64; b_buffer_size];
    let b_packed_ptr = b_packed.as_mut_ptr();

    // Loop over N dimension (columns of B and C) with NC blocking
    let mut jc = 0;
    while jc < n {
        let nc_curr = (n - jc).min(nc);

        // Loop over K dimension (inner dimension) with KC blocking
        let mut pc = 0;
        while pc < k {
            let kc_curr = (k - pc).min(kc);

            // Loop over M dimension (rows of A and C) with MC blocking
            let mut ic = 0;
            while ic < m {
                let mc_curr = (m - ic).min(mc);

                // Loop over NR panels (micro-panel of B)
                let mut jr = 0;
                while jr < nc_curr {
                    let nr_curr = (nc_curr - jr).min(nr);

                    // Pack B[KC x NR] for this micro-panel
                    let b_panel = b.add(pc * ldb + jc + jr);
                    pack_b_f64_fast(kc_curr, nr_curr, b_panel, ldb, b_packed_ptr, config);

                    // Loop over MR panels (micro-panel of A)
                    let mut ir = 0;
                    while ir < mc_curr {
                        let mr_curr = (mc_curr - ir).min(mr);

                        // Get pointers to current micro-panels
                        let a_panel = a.add((ic + ir) * lda + pc);
                        let c_block = c.add((ic + ir) * ldc + jc + jr);

                        // Compute micro-kernel: C[MR x NR] += A[MR x KC] * B[KC x NR]
                        // Only apply beta on first iteration (pc == 0)
                        let beta_curr = if pc == 0 { beta } else { 1.0 };

                        if mr_curr == mr && nr_curr == nr && kc_curr == kc {
                            // Fast path: full micro-kernel
                            micro_kernel_f64(
                                kc_curr,
                                alpha,
                                a_panel,
                                b_packed_ptr,
                                beta_curr,
                                c_block,
                                ldc,
                                config,
                            );
                        } else {
                            // Edge case: partial micro-kernel
                            micro_kernel_f64_edge(
                                mr_curr,
                                nr_curr,
                                kc_curr,
                                alpha,
                                a_panel,
                                lda,
                                b_packed_ptr,
                                beta_curr,
                                c_block,
                                ldc,
                            );
                        }

                        ir += mr;
                    }

                    jr += nr;
                }

                ic += mc;
            }

            pc += kc;
        }

        jc += nc;
    }
}

/// Edge case micro-kernel for f64 partial blocks
///
/// Handles cases where mr_curr < MR or nr_curr < NR
#[inline]
unsafe fn micro_kernel_f64_edge(
    mr_curr: usize,
    nr_curr: usize,
    kc_curr: usize,
    alpha: f64,
    a: *const f64,
    lda: usize,
    b_packed: *const f64,
    beta: f64,
    c: *mut f64,
    ldc: usize,
) {
    // Simple scalar implementation for edge cases
    for i in 0..mr_curr {
        for j in 0..nr_curr {
            let mut sum = 0.0f64;

            for p in 0..kc_curr {
                let a_val = *a.add(i * lda + p);
                let b_val = *b_packed.add(p * nr_curr + j);
                sum += a_val * b_val;
            }

            let c_ptr = c.add(i * ldc + j);
            if beta == 0.0 {
                *c_ptr = alpha * sum;
            } else {
                *c_ptr = beta * (*c_ptr) + alpha * sum;
            }
        }
    }
}

/// Simple GEMM for small f64 matrices (fallback)
///
/// Uses straightforward triple-loop implementation optimized for small sizes.
#[inline]
unsafe fn gemm_small_f64(
    m: usize,
    k: usize,
    n: usize,
    alpha: f64,
    a: *const f64,
    lda: usize,
    b: *const f64,
    ldb: usize,
    beta: f64,
    c: *mut f64,
    ldc: usize,
) {
    // Scale C by beta first
    if beta == 0.0 {
        for i in 0..m {
            for j in 0..n {
                *c.add(i * ldc + j) = 0.0;
            }
        }
    } else if beta != 1.0 {
        for i in 0..m {
            for j in 0..n {
                let c_ptr = c.add(i * ldc + j);
                *c_ptr *= beta;
            }
        }
    }

    // Simple triple loop: C += alpha * A * B
    // Use ikj order for better cache locality
    for i in 0..m {
        for p in 0..k {
            let a_val = alpha * (*a.add(i * lda + p));

            // Vectorize the inner loop when possible
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                    use std::arch::x86_64::*;

                    let a_broadcast = _mm256_set1_pd(a_val);
                    let mut j = 0;

                    while j + 4 <= n {
                        let b_vec = _mm256_loadu_pd(b.add(p * ldb + j));
                        let c_ptr = c.add(i * ldc + j);
                        let c_vec = _mm256_loadu_pd(c_ptr);
                        let result = _mm256_fmadd_pd(a_broadcast, b_vec, c_vec);
                        _mm256_storeu_pd(c_ptr, result);
                        j += 4;
                    }

                    // Handle remaining elements
                    while j < n {
                        let c_ptr = c.add(i * ldc + j);
                        *c_ptr += a_val * (*b.add(p * ldb + j));
                        j += 1;
                    }

                    continue;
                }
            }

            #[cfg(target_arch = "aarch64")]
            {
                if std::arch::is_aarch64_feature_detected!("neon") {
                    use std::arch::aarch64::*;

                    let a_broadcast = vdupq_n_f64(a_val);
                    let mut j = 0;

                    while j + 2 <= n {
                        let b_vec = vld1q_f64(b.add(p * ldb + j));
                        let c_ptr = c.add(i * ldc + j);
                        let c_vec = vld1q_f64(c_ptr);
                        let result = vfmaq_f64(c_vec, a_broadcast, b_vec);
                        vst1q_f64(c_ptr, result);
                        j += 2;
                    }

                    // Handle remaining elements
                    while j < n {
                        let c_ptr = c.add(i * ldc + j);
                        *c_ptr += a_val * (*b.add(p * ldb + j));
                        j += 1;
                    }

                    continue;
                }
            }

            // Scalar fallback
            for j in 0..n {
                let c_ptr = c.add(i * ldc + j);
                *c_ptr += a_val * (*b.add(p * ldb + j));
            }
        }
    }
}

/// Edge case micro-kernel for partial blocks
///
/// Handles cases where mr_curr < MR or nr_curr < NR
#[inline]
unsafe fn micro_kernel_f32_edge(
    mr_curr: usize,
    nr_curr: usize,
    kc_curr: usize,
    alpha: f32,
    a: *const f32,
    lda: usize,
    b_packed: *const f32,
    beta: f32,
    c: *mut f32,
    ldc: usize,
) {
    // Simple scalar implementation for edge cases
    for i in 0..mr_curr {
        for j in 0..nr_curr {
            let mut sum = 0.0f32;

            for p in 0..kc_curr {
                let a_val = *a.add(i * lda + p);
                let b_val = *b_packed.add(p * nr_curr + j);
                sum += a_val * b_val;
            }

            let c_ptr = c.add(i * ldc + j);
            if beta == 0.0 {
                *c_ptr = alpha * sum;
            } else {
                *c_ptr = beta * (*c_ptr) + alpha * sum;
            }
        }
    }
}

/// Simple GEMM for small matrices (fallback)
///
/// Uses straightforward triple-loop implementation optimized for small sizes.
#[inline]
unsafe fn gemm_small_f32(
    m: usize,
    k: usize,
    n: usize,
    alpha: f32,
    a: *const f32,
    lda: usize,
    b: *const f32,
    ldb: usize,
    beta: f32,
    c: *mut f32,
    ldc: usize,
) {
    // Scale C by beta first
    if beta == 0.0 {
        for i in 0..m {
            for j in 0..n {
                *c.add(i * ldc + j) = 0.0;
            }
        }
    } else if beta != 1.0 {
        for i in 0..m {
            for j in 0..n {
                let c_ptr = c.add(i * ldc + j);
                *c_ptr *= beta;
            }
        }
    }

    // Simple triple loop: C += alpha * A * B
    // Use ikj order for better cache locality
    for i in 0..m {
        for p in 0..k {
            let a_val = alpha * (*a.add(i * lda + p));

            // Vectorize the inner loop when possible
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2") {
                    use std::arch::x86_64::*;

                    let a_broadcast = _mm256_set1_ps(a_val);
                    let mut j = 0;

                    while j + 8 <= n {
                        let b_vec = _mm256_loadu_ps(b.add(p * ldb + j));
                        let c_ptr = c.add(i * ldc + j);
                        let c_vec = _mm256_loadu_ps(c_ptr);
                        let result = _mm256_fmadd_ps(a_broadcast, b_vec, c_vec);
                        _mm256_storeu_ps(c_ptr, result);
                        j += 8;
                    }

                    // Handle remaining elements
                    while j < n {
                        let c_ptr = c.add(i * ldc + j);
                        *c_ptr += a_val * (*b.add(p * ldb + j));
                        j += 1;
                    }

                    continue;
                }
            }

            // Scalar fallback
            for j in 0..n {
                let c_ptr = c.add(i * ldc + j);
                *c_ptr += a_val * (*b.add(p * ldb + j));
            }
        }
    }
}

/// Threshold below which we use simple implementation instead of blocked
const GEMM_BLOCK_THRESHOLD: usize = 64;

/// Check if matrices are large enough to benefit from blocking
#[inline]
pub fn should_use_blocked(m: usize, n: usize, k: usize) -> bool {
    m >= GEMM_BLOCK_THRESHOLD && n >= GEMM_BLOCK_THRESHOLD && k >= GEMM_BLOCK_THRESHOLD
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_small_gemm() {
        // Test 4x4 matrix multiplication
        let m = 4;
        let k = 4;
        let n = 4;

        let a: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];

        let b: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];

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

        // B is identity, so C should equal A
        for i in 0..16 {
            assert!(
                (c[i] - a[i]).abs() < 1e-5,
                "Mismatch at index {}: expected {}, got {}",
                i,
                a[i],
                c[i]
            );
        }
    }

    #[test]
    fn test_gemm_with_alpha_beta() {
        let m = 2;
        let k = 2;
        let n = 2;

        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let b: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];
        let mut c = vec![1.0, 1.0, 1.0, 1.0];

        let config = MatMulConfig::for_f32();

        unsafe {
            blocked_gemm_f32(
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
                (c[i] - expected[i]).abs() < 1e-4,
                "Mismatch at index {}: expected {}, got {}",
                i,
                expected[i],
                c[i]
            );
        }
    }

    #[test]
    fn test_large_gemm() {
        // Test larger matrix to trigger blocked path
        let m = 128;
        let k = 128;
        let n = 128;

        let a = vec![1.0f32; m * k];
        let b = vec![1.0f32; k * n];
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

        // Each element should be k (sum of 1.0 * 1.0 for k iterations)
        for i in 0..m {
            for j in 0..n {
                let val = c[i * n + j];
                assert!(
                    (val - k as f32).abs() < 1e-3,
                    "Mismatch at ({}, {}): expected {}, got {}",
                    i,
                    j,
                    k,
                    val
                );
            }
        }
    }

    #[test]
    fn test_should_use_blocked() {
        assert!(!should_use_blocked(32, 32, 32)); // Too small
        assert!(!should_use_blocked(64, 64, 32)); // K too small
        assert!(should_use_blocked(64, 64, 64)); // All dimensions large enough
        assert!(should_use_blocked(256, 256, 256)); // Large matrices
    }
}
