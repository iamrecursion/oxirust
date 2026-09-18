//! Platform-specific micro-kernels for GEMM operations
//!
//! This module provides highly optimized micro-kernels that form the computational core
//! of the blocked GEMM algorithm. Each kernel computes C[MR x NR] += A[MR x KC] * B[KC x NR]
//! using SIMD instructions optimized for the target architecture.
//!
//! # Optimization Techniques
//!
//! - FMA instructions for single-cycle multiply-add
//! - Register blocking to maximize register utilization
//! - Loop unrolling to reduce overhead
//! - Prefetching to hide memory latency
//! - Multiple accumulators to avoid dependency chains

use super::config::MatMulConfig;

/// Micro-kernel for f32 matrix multiplication: C[MR x NR] += A[MR x KC] * B_packed[KC x NR]
///
/// # Arguments
///
/// * `kc` - Inner dimension (number of elements to accumulate)
/// * `alpha` - Scalar multiplier for A*B
/// * `a` - Pointer to A panel (MR x KC, row-major)
/// * `b_packed` - Pointer to packed B panel (KC x NR, packed format)
/// * `beta` - Scalar multiplier for existing C values
/// * `c` - Pointer to C block (MR x NR, row-major)
/// * `rs_c` - Row stride of C (typically NC for C[M x N])
/// * `config` - GEMM configuration
///
/// # Safety
///
/// Caller must ensure:
/// - `a` points to valid memory of at least `MR * kc` elements
/// - `b_packed` points to valid memory of at least `kc * NR` elements
/// - `c` points to valid memory with proper strides
#[inline(always)]
pub unsafe fn micro_kernel_f32(
    kc: usize,
    alpha: f32,
    a: *const f32,
    b_packed: *const f32,
    beta: f32,
    c: *mut f32,
    rs_c: usize,
    config: &MatMulConfig,
) {
    let mr = config.mr;
    let nr = config.nr;

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && nr == 16 && mr == 8 {
            avx512_micro_kernel_f32_8x16(kc, alpha, a, b_packed, beta, c, rs_c);
        } else if is_x86_feature_detected!("avx2") && nr == 8 && mr == 8 {
            avx2_micro_kernel_f32_8x8(kc, alpha, a, b_packed, beta, c, rs_c);
        } else if is_x86_feature_detected!("sse") && nr == 4 && mr == 4 {
            sse_micro_kernel_f32_4x4(kc, alpha, a, b_packed, beta, c, rs_c);
        } else {
            scalar_micro_kernel_f32(kc, alpha, a, b_packed, beta, c, rs_c, mr, nr);
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") && nr == 4 && mr == 8 {
            neon_micro_kernel_f32_8x4(kc, alpha, a, b_packed, beta, c, rs_c);
        } else {
            scalar_micro_kernel_f32(kc, alpha, a, b_packed, beta, c, rs_c, mr, nr);
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        scalar_micro_kernel_f32(kc, alpha, a, b_packed, beta, c, rs_c, mr, nr);
    }
}

// ==================== x86_64 AVX-512 Micro-kernel (8x16) ====================

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx512f")]
#[allow(clippy::identity_op)] // 0 * kc is intentional for clarity (row_index * kc pattern)
unsafe fn avx512_micro_kernel_f32_8x16(
    kc: usize,
    alpha: f32,
    a: *const f32,
    b_packed: *const f32,
    beta: f32,
    c: *mut f32,
    rs_c: usize,
) {
    use std::arch::x86_64::*;

    // 8x16 = 8 rows x 16 cols (one AVX-512 vector per column)
    // We need 8 AVX-512 registers to accumulate the 8x16 result
    let mut c0 = _mm512_setzero_ps();
    let mut c1 = _mm512_setzero_ps();
    let mut c2 = _mm512_setzero_ps();
    let mut c3 = _mm512_setzero_ps();
    let mut c4 = _mm512_setzero_ps();
    let mut c5 = _mm512_setzero_ps();
    let mut c6 = _mm512_setzero_ps();
    let mut c7 = _mm512_setzero_ps();

    // Main accumulation loop: C += A * B
    for p in 0..kc {
        // Load one column of B (16 elements = 1 AVX-512 vector)
        let b_vec = _mm512_loadu_ps(b_packed.add(p * 16));

        // Broadcast each element of A column and FMA with B
        let a0 = _mm512_set1_ps(*a.add(p));
        let a1 = _mm512_set1_ps(*a.add(1 * kc + p));
        let a2 = _mm512_set1_ps(*a.add(2 * kc + p));
        let a3 = _mm512_set1_ps(*a.add(3 * kc + p));
        let a4 = _mm512_set1_ps(*a.add(4 * kc + p));
        let a5 = _mm512_set1_ps(*a.add(5 * kc + p));
        let a6 = _mm512_set1_ps(*a.add(6 * kc + p));
        let a7 = _mm512_set1_ps(*a.add(7 * kc + p));

        c0 = _mm512_fmadd_ps(a0, b_vec, c0);
        c1 = _mm512_fmadd_ps(a1, b_vec, c1);
        c2 = _mm512_fmadd_ps(a2, b_vec, c2);
        c3 = _mm512_fmadd_ps(a3, b_vec, c3);
        c4 = _mm512_fmadd_ps(a4, b_vec, c4);
        c5 = _mm512_fmadd_ps(a5, b_vec, c5);
        c6 = _mm512_fmadd_ps(a6, b_vec, c6);
        c7 = _mm512_fmadd_ps(a7, b_vec, c7);
    }

    // Scale by alpha
    let alpha_vec = _mm512_set1_ps(alpha);
    c0 = _mm512_mul_ps(c0, alpha_vec);
    c1 = _mm512_mul_ps(c1, alpha_vec);
    c2 = _mm512_mul_ps(c2, alpha_vec);
    c3 = _mm512_mul_ps(c3, alpha_vec);
    c4 = _mm512_mul_ps(c4, alpha_vec);
    c5 = _mm512_mul_ps(c5, alpha_vec);
    c6 = _mm512_mul_ps(c6, alpha_vec);
    c7 = _mm512_mul_ps(c7, alpha_vec);

    // Store back to C with beta scaling
    let beta_vec = _mm512_set1_ps(beta);

    for i in 0..8 {
        let c_ptr = c.add(i * rs_c);
        let c_row = match i {
            0 => c0,
            1 => c1,
            2 => c2,
            3 => c3,
            4 => c4,
            5 => c5,
            6 => c6,
            _ => c7,
        };

        if beta == 0.0 {
            _mm512_storeu_ps(c_ptr, c_row);
        } else {
            let c_old = _mm512_loadu_ps(c_ptr);
            let c_new = _mm512_fmadd_ps(c_old, beta_vec, c_row);
            _mm512_storeu_ps(c_ptr, c_new);
        }
    }
}

// ==================== x86_64 AVX2 Micro-kernel (8x8) ====================

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
#[allow(clippy::identity_op)] // 0 * kc is intentional for clarity
unsafe fn avx2_micro_kernel_f32_8x8(
    kc: usize,
    alpha: f32,
    a: *const f32,
    b_packed: *const f32,
    beta: f32,
    c: *mut f32,
    rs_c: usize,
) {
    use std::arch::x86_64::*;

    // 8x8 = 8 rows x 8 cols (one AVX2 vector per column)
    // We need 8 AVX2 registers to accumulate the 8x8 result
    let mut c0 = _mm256_setzero_ps();
    let mut c1 = _mm256_setzero_ps();
    let mut c2 = _mm256_setzero_ps();
    let mut c3 = _mm256_setzero_ps();
    let mut c4 = _mm256_setzero_ps();
    let mut c5 = _mm256_setzero_ps();
    let mut c6 = _mm256_setzero_ps();
    let mut c7 = _mm256_setzero_ps();

    // Prefetch distance
    const PREFETCH_DISTANCE: usize = 128;

    // Main accumulation loop with prefetching
    for p in 0..kc {
        // Prefetch ahead
        if p + PREFETCH_DISTANCE < kc {
            _mm_prefetch(
                b_packed.add((p + PREFETCH_DISTANCE) * 8) as *const i8,
                _MM_HINT_T0,
            );
        }

        // Load one column of B (8 elements = 1 AVX2 vector)
        let b_vec = _mm256_loadu_ps(b_packed.add(p * 8));

        // Broadcast each element of A column and FMA with B
        let a0 = _mm256_set1_ps(*a.add(p));
        let a1 = _mm256_set1_ps(*a.add(1 * kc + p));
        let a2 = _mm256_set1_ps(*a.add(2 * kc + p));
        let a3 = _mm256_set1_ps(*a.add(3 * kc + p));
        let a4 = _mm256_set1_ps(*a.add(4 * kc + p));
        let a5 = _mm256_set1_ps(*a.add(5 * kc + p));
        let a6 = _mm256_set1_ps(*a.add(6 * kc + p));
        let a7 = _mm256_set1_ps(*a.add(7 * kc + p));

        c0 = _mm256_fmadd_ps(a0, b_vec, c0);
        c1 = _mm256_fmadd_ps(a1, b_vec, c1);
        c2 = _mm256_fmadd_ps(a2, b_vec, c2);
        c3 = _mm256_fmadd_ps(a3, b_vec, c3);
        c4 = _mm256_fmadd_ps(a4, b_vec, c4);
        c5 = _mm256_fmadd_ps(a5, b_vec, c5);
        c6 = _mm256_fmadd_ps(a6, b_vec, c6);
        c7 = _mm256_fmadd_ps(a7, b_vec, c7);
    }

    // Scale by alpha
    let alpha_vec = _mm256_set1_ps(alpha);
    c0 = _mm256_mul_ps(c0, alpha_vec);
    c1 = _mm256_mul_ps(c1, alpha_vec);
    c2 = _mm256_mul_ps(c2, alpha_vec);
    c3 = _mm256_mul_ps(c3, alpha_vec);
    c4 = _mm256_mul_ps(c4, alpha_vec);
    c5 = _mm256_mul_ps(c5, alpha_vec);
    c6 = _mm256_mul_ps(c6, alpha_vec);
    c7 = _mm256_mul_ps(c7, alpha_vec);

    // Store back to C with beta scaling
    let beta_vec = _mm256_set1_ps(beta);

    for i in 0..8 {
        let c_ptr = c.add(i * rs_c);
        let c_row = match i {
            0 => c0,
            1 => c1,
            2 => c2,
            3 => c3,
            4 => c4,
            5 => c5,
            6 => c6,
            _ => c7,
        };

        if beta == 0.0 {
            _mm256_storeu_ps(c_ptr, c_row);
        } else {
            let c_old = _mm256_loadu_ps(c_ptr);
            let c_new = _mm256_fmadd_ps(c_old, beta_vec, c_row);
            _mm256_storeu_ps(c_ptr, c_new);
        }
    }
}

// ==================== x86_64 SSE Micro-kernel (4x4) ====================

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "sse")]
#[allow(clippy::identity_op)] // 0 * kc is intentional for clarity
unsafe fn sse_micro_kernel_f32_4x4(
    kc: usize,
    alpha: f32,
    a: *const f32,
    b_packed: *const f32,
    beta: f32,
    c: *mut f32,
    rs_c: usize,
) {
    use std::arch::x86_64::*;

    // 4x4 = 4 rows x 4 cols (one SSE vector per column)
    let mut c0 = _mm_setzero_ps();
    let mut c1 = _mm_setzero_ps();
    let mut c2 = _mm_setzero_ps();
    let mut c3 = _mm_setzero_ps();

    // Main accumulation loop
    for p in 0..kc {
        let b_vec = _mm_loadu_ps(b_packed.add(p * 4));

        let a0 = _mm_set1_ps(*a.add(p));
        let a1 = _mm_set1_ps(*a.add(1 * kc + p));
        let a2 = _mm_set1_ps(*a.add(2 * kc + p));
        let a3 = _mm_set1_ps(*a.add(3 * kc + p));

        // Use mul+add since FMA might not be available
        c0 = _mm_add_ps(c0, _mm_mul_ps(a0, b_vec));
        c1 = _mm_add_ps(c1, _mm_mul_ps(a1, b_vec));
        c2 = _mm_add_ps(c2, _mm_mul_ps(a2, b_vec));
        c3 = _mm_add_ps(c3, _mm_mul_ps(a3, b_vec));
    }

    // Scale by alpha
    let alpha_vec = _mm_set1_ps(alpha);
    c0 = _mm_mul_ps(c0, alpha_vec);
    c1 = _mm_mul_ps(c1, alpha_vec);
    c2 = _mm_mul_ps(c2, alpha_vec);
    c3 = _mm_mul_ps(c3, alpha_vec);

    // Store back to C with beta scaling
    let beta_vec = _mm_set1_ps(beta);

    for i in 0..4 {
        let c_ptr = c.add(i * rs_c);
        let c_row = match i {
            0 => c0,
            1 => c1,
            2 => c2,
            _ => c3,
        };

        if beta == 0.0 {
            _mm_storeu_ps(c_ptr, c_row);
        } else {
            let c_old = _mm_loadu_ps(c_ptr);
            let c_new = _mm_add_ps(_mm_mul_ps(c_old, beta_vec), c_row);
            _mm_storeu_ps(c_ptr, c_new);
        }
    }
}

// ==================== ARM NEON Micro-kernel (8x4) ====================

#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(clippy::identity_op)] // 0 * kc is intentional for clarity
unsafe fn neon_micro_kernel_f32_8x4(
    kc: usize,
    alpha: f32,
    a: *const f32,
    b_packed: *const f32,
    beta: f32,
    c: *mut f32,
    rs_c: usize,
) {
    use std::arch::aarch64::*;

    // 8x4 = 8 rows x 4 cols (one NEON vector per column)
    let mut c0 = vdupq_n_f32(0.0);
    let mut c1 = vdupq_n_f32(0.0);
    let mut c2 = vdupq_n_f32(0.0);
    let mut c3 = vdupq_n_f32(0.0);
    let mut c4 = vdupq_n_f32(0.0);
    let mut c5 = vdupq_n_f32(0.0);
    let mut c6 = vdupq_n_f32(0.0);
    let mut c7 = vdupq_n_f32(0.0);

    // Main accumulation loop
    #[allow(clippy::identity_op, clippy::erasing_op)] // 0 * kc is intentional for clarity
    for p in 0..kc {
        let b_vec = vld1q_f32(b_packed.add(p * 4));

        let a0 = vdupq_n_f32(*a.add(0 * kc + p));
        let a1 = vdupq_n_f32(*a.add(1 * kc + p));
        let a2 = vdupq_n_f32(*a.add(2 * kc + p));
        let a3 = vdupq_n_f32(*a.add(3 * kc + p));
        let a4 = vdupq_n_f32(*a.add(4 * kc + p));
        let a5 = vdupq_n_f32(*a.add(5 * kc + p));
        let a6 = vdupq_n_f32(*a.add(6 * kc + p));
        let a7 = vdupq_n_f32(*a.add(7 * kc + p));

        // FMA on ARM: c = c + a * b
        c0 = vfmaq_f32(c0, a0, b_vec);
        c1 = vfmaq_f32(c1, a1, b_vec);
        c2 = vfmaq_f32(c2, a2, b_vec);
        c3 = vfmaq_f32(c3, a3, b_vec);
        c4 = vfmaq_f32(c4, a4, b_vec);
        c5 = vfmaq_f32(c5, a5, b_vec);
        c6 = vfmaq_f32(c6, a6, b_vec);
        c7 = vfmaq_f32(c7, a7, b_vec);
    }

    // Scale by alpha
    let alpha_vec = vdupq_n_f32(alpha);
    c0 = vmulq_f32(c0, alpha_vec);
    c1 = vmulq_f32(c1, alpha_vec);
    c2 = vmulq_f32(c2, alpha_vec);
    c3 = vmulq_f32(c3, alpha_vec);
    c4 = vmulq_f32(c4, alpha_vec);
    c5 = vmulq_f32(c5, alpha_vec);
    c6 = vmulq_f32(c6, alpha_vec);
    c7 = vmulq_f32(c7, alpha_vec);

    // Store back to C with beta scaling
    let beta_vec = vdupq_n_f32(beta);

    for i in 0..8 {
        let c_ptr = c.add(i * rs_c);
        let c_row = match i {
            0 => c0,
            1 => c1,
            2 => c2,
            3 => c3,
            4 => c4,
            5 => c5,
            6 => c6,
            _ => c7,
        };

        if beta == 0.0 {
            vst1q_f32(c_ptr, c_row);
        } else {
            let c_old = vld1q_f32(c_ptr);
            let c_new = vfmaq_f32(c_row, c_old, beta_vec);
            vst1q_f32(c_ptr, c_new);
        }
    }
}

// ==================== Scalar Fallback Micro-kernel ====================

#[inline(always)]
unsafe fn scalar_micro_kernel_f32(
    kc: usize,
    alpha: f32,
    a: *const f32,
    b_packed: *const f32,
    beta: f32,
    c: *mut f32,
    rs_c: usize,
    mr: usize,
    nr: usize,
) {
    // Compute C[mr x nr] += alpha * A[mr x kc] * B[kc x nr]
    for i in 0..mr {
        for j in 0..nr {
            let mut sum = 0.0f32;

            // Dot product of row i of A with column j of B
            for p in 0..kc {
                let a_val = *a.add(i * kc + p);
                let b_val = *b_packed.add(p * nr + j);
                sum += a_val * b_val;
            }

            let c_ptr = c.add(i * rs_c + j);
            if beta == 0.0 {
                *c_ptr = alpha * sum;
            } else {
                *c_ptr = beta * (*c_ptr) + alpha * sum;
            }
        }
    }
}

// ==================== f64 Micro-kernels ====================

/// Micro-kernel for f64 matrix multiplication: C[MR x NR] += A[MR x KC] * B_packed[KC x NR]
///
/// # Arguments
///
/// * `kc` - Inner dimension (number of elements to accumulate)
/// * `alpha` - Scalar multiplier for A*B
/// * `a` - Pointer to A panel (MR x KC, row-major)
/// * `b_packed` - Pointer to packed B panel (KC x NR, packed format)
/// * `beta` - Scalar multiplier for existing C values
/// * `c` - Pointer to C block (MR x NR, row-major)
/// * `rs_c` - Row stride of C (typically NC for C[M x N])
/// * `config` - GEMM configuration
///
/// # Safety
///
/// Caller must ensure:
/// - `a` points to valid memory of at least `MR * kc` elements
/// - `b_packed` points to valid memory of at least `kc * NR` elements
/// - `c` points to valid memory with proper strides
#[inline(always)]
pub unsafe fn micro_kernel_f64(
    kc: usize,
    alpha: f64,
    a: *const f64,
    b_packed: *const f64,
    beta: f64,
    c: *mut f64,
    rs_c: usize,
    config: &MatMulConfig,
) {
    let mr = config.mr;
    let nr = config.nr;

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && nr == 8 && mr == 8 {
            avx512_micro_kernel_f64_8x8(kc, alpha, a, b_packed, beta, c, rs_c);
        } else if is_x86_feature_detected!("avx2") && nr == 4 && mr == 8 {
            avx2_micro_kernel_f64_8x4(kc, alpha, a, b_packed, beta, c, rs_c);
        } else if is_x86_feature_detected!("sse2") && nr == 2 && mr == 4 {
            sse2_micro_kernel_f64_4x2(kc, alpha, a, b_packed, beta, c, rs_c);
        } else {
            scalar_micro_kernel_f64(kc, alpha, a, b_packed, beta, c, rs_c, mr, nr);
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") && nr == 2 && mr == 8 {
            neon_micro_kernel_f64_8x2(kc, alpha, a, b_packed, beta, c, rs_c);
        } else {
            scalar_micro_kernel_f64(kc, alpha, a, b_packed, beta, c, rs_c, mr, nr);
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        scalar_micro_kernel_f64(kc, alpha, a, b_packed, beta, c, rs_c, mr, nr);
    }
}

// ==================== x86_64 AVX-512 Micro-kernel for f64 (8x8) ====================

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx512f")]
#[allow(clippy::identity_op)]
unsafe fn avx512_micro_kernel_f64_8x8(
    kc: usize,
    alpha: f64,
    a: *const f64,
    b_packed: *const f64,
    beta: f64,
    c: *mut f64,
    rs_c: usize,
) {
    use std::arch::x86_64::*;

    // 8x8 = 8 rows x 8 cols (one AVX-512 vector of 8 f64s per row)
    let mut c0 = _mm512_setzero_pd();
    let mut c1 = _mm512_setzero_pd();
    let mut c2 = _mm512_setzero_pd();
    let mut c3 = _mm512_setzero_pd();
    let mut c4 = _mm512_setzero_pd();
    let mut c5 = _mm512_setzero_pd();
    let mut c6 = _mm512_setzero_pd();
    let mut c7 = _mm512_setzero_pd();

    // Main accumulation loop
    for p in 0..kc {
        // Load one column of B (8 f64 elements = 1 AVX-512 vector)
        let b_vec = _mm512_loadu_pd(b_packed.add(p * 8));

        // Broadcast each element of A column and FMA with B
        let a0 = _mm512_set1_pd(*a.add(p));
        let a1 = _mm512_set1_pd(*a.add(1 * kc + p));
        let a2 = _mm512_set1_pd(*a.add(2 * kc + p));
        let a3 = _mm512_set1_pd(*a.add(3 * kc + p));
        let a4 = _mm512_set1_pd(*a.add(4 * kc + p));
        let a5 = _mm512_set1_pd(*a.add(5 * kc + p));
        let a6 = _mm512_set1_pd(*a.add(6 * kc + p));
        let a7 = _mm512_set1_pd(*a.add(7 * kc + p));

        c0 = _mm512_fmadd_pd(a0, b_vec, c0);
        c1 = _mm512_fmadd_pd(a1, b_vec, c1);
        c2 = _mm512_fmadd_pd(a2, b_vec, c2);
        c3 = _mm512_fmadd_pd(a3, b_vec, c3);
        c4 = _mm512_fmadd_pd(a4, b_vec, c4);
        c5 = _mm512_fmadd_pd(a5, b_vec, c5);
        c6 = _mm512_fmadd_pd(a6, b_vec, c6);
        c7 = _mm512_fmadd_pd(a7, b_vec, c7);
    }

    // Scale by alpha
    let alpha_vec = _mm512_set1_pd(alpha);
    c0 = _mm512_mul_pd(c0, alpha_vec);
    c1 = _mm512_mul_pd(c1, alpha_vec);
    c2 = _mm512_mul_pd(c2, alpha_vec);
    c3 = _mm512_mul_pd(c3, alpha_vec);
    c4 = _mm512_mul_pd(c4, alpha_vec);
    c5 = _mm512_mul_pd(c5, alpha_vec);
    c6 = _mm512_mul_pd(c6, alpha_vec);
    c7 = _mm512_mul_pd(c7, alpha_vec);

    // Store back to C with beta scaling
    let beta_vec = _mm512_set1_pd(beta);

    for i in 0..8 {
        let c_ptr = c.add(i * rs_c);
        let c_row = match i {
            0 => c0,
            1 => c1,
            2 => c2,
            3 => c3,
            4 => c4,
            5 => c5,
            6 => c6,
            _ => c7,
        };

        if beta == 0.0 {
            _mm512_storeu_pd(c_ptr, c_row);
        } else {
            let c_old = _mm512_loadu_pd(c_ptr);
            let c_new = _mm512_fmadd_pd(c_old, beta_vec, c_row);
            _mm512_storeu_pd(c_ptr, c_new);
        }
    }
}

// ==================== x86_64 AVX2 Micro-kernel for f64 (8x4) ====================

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
#[allow(clippy::identity_op)]
unsafe fn avx2_micro_kernel_f64_8x4(
    kc: usize,
    alpha: f64,
    a: *const f64,
    b_packed: *const f64,
    beta: f64,
    c: *mut f64,
    rs_c: usize,
) {
    use std::arch::x86_64::*;

    // 8x4 = 8 rows x 4 cols (one AVX2 vector of 4 f64s per row)
    let mut c0 = _mm256_setzero_pd();
    let mut c1 = _mm256_setzero_pd();
    let mut c2 = _mm256_setzero_pd();
    let mut c3 = _mm256_setzero_pd();
    let mut c4 = _mm256_setzero_pd();
    let mut c5 = _mm256_setzero_pd();
    let mut c6 = _mm256_setzero_pd();
    let mut c7 = _mm256_setzero_pd();

    // Prefetch distance
    const PREFETCH_DISTANCE: usize = 64;

    // Main accumulation loop with prefetching
    for p in 0..kc {
        // Prefetch ahead
        if p + PREFETCH_DISTANCE < kc {
            _mm_prefetch(
                b_packed.add((p + PREFETCH_DISTANCE) * 4) as *const i8,
                _MM_HINT_T0,
            );
        }

        // Load one column of B (4 f64 elements = 1 AVX2 vector)
        let b_vec = _mm256_loadu_pd(b_packed.add(p * 4));

        // Broadcast each element of A column and FMA with B
        let a0 = _mm256_set1_pd(*a.add(p));
        let a1 = _mm256_set1_pd(*a.add(1 * kc + p));
        let a2 = _mm256_set1_pd(*a.add(2 * kc + p));
        let a3 = _mm256_set1_pd(*a.add(3 * kc + p));
        let a4 = _mm256_set1_pd(*a.add(4 * kc + p));
        let a5 = _mm256_set1_pd(*a.add(5 * kc + p));
        let a6 = _mm256_set1_pd(*a.add(6 * kc + p));
        let a7 = _mm256_set1_pd(*a.add(7 * kc + p));

        c0 = _mm256_fmadd_pd(a0, b_vec, c0);
        c1 = _mm256_fmadd_pd(a1, b_vec, c1);
        c2 = _mm256_fmadd_pd(a2, b_vec, c2);
        c3 = _mm256_fmadd_pd(a3, b_vec, c3);
        c4 = _mm256_fmadd_pd(a4, b_vec, c4);
        c5 = _mm256_fmadd_pd(a5, b_vec, c5);
        c6 = _mm256_fmadd_pd(a6, b_vec, c6);
        c7 = _mm256_fmadd_pd(a7, b_vec, c7);
    }

    // Scale by alpha
    let alpha_vec = _mm256_set1_pd(alpha);
    c0 = _mm256_mul_pd(c0, alpha_vec);
    c1 = _mm256_mul_pd(c1, alpha_vec);
    c2 = _mm256_mul_pd(c2, alpha_vec);
    c3 = _mm256_mul_pd(c3, alpha_vec);
    c4 = _mm256_mul_pd(c4, alpha_vec);
    c5 = _mm256_mul_pd(c5, alpha_vec);
    c6 = _mm256_mul_pd(c6, alpha_vec);
    c7 = _mm256_mul_pd(c7, alpha_vec);

    // Store back to C with beta scaling
    let beta_vec = _mm256_set1_pd(beta);

    for i in 0..8 {
        let c_ptr = c.add(i * rs_c);
        let c_row = match i {
            0 => c0,
            1 => c1,
            2 => c2,
            3 => c3,
            4 => c4,
            5 => c5,
            6 => c6,
            _ => c7,
        };

        if beta == 0.0 {
            _mm256_storeu_pd(c_ptr, c_row);
        } else {
            let c_old = _mm256_loadu_pd(c_ptr);
            let c_new = _mm256_fmadd_pd(c_old, beta_vec, c_row);
            _mm256_storeu_pd(c_ptr, c_new);
        }
    }
}

// ==================== x86_64 SSE2 Micro-kernel for f64 (4x2) ====================

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "sse2")]
#[allow(clippy::identity_op)]
unsafe fn sse2_micro_kernel_f64_4x2(
    kc: usize,
    alpha: f64,
    a: *const f64,
    b_packed: *const f64,
    beta: f64,
    c: *mut f64,
    rs_c: usize,
) {
    use std::arch::x86_64::*;

    // 4x2 = 4 rows x 2 cols (one SSE2 vector of 2 f64s per row)
    let mut c0 = _mm_setzero_pd();
    let mut c1 = _mm_setzero_pd();
    let mut c2 = _mm_setzero_pd();
    let mut c3 = _mm_setzero_pd();

    // Main accumulation loop
    for p in 0..kc {
        let b_vec = _mm_loadu_pd(b_packed.add(p * 2));

        let a0 = _mm_set1_pd(*a.add(p));
        let a1 = _mm_set1_pd(*a.add(1 * kc + p));
        let a2 = _mm_set1_pd(*a.add(2 * kc + p));
        let a3 = _mm_set1_pd(*a.add(3 * kc + p));

        // Use mul+add since FMA might not be available
        c0 = _mm_add_pd(c0, _mm_mul_pd(a0, b_vec));
        c1 = _mm_add_pd(c1, _mm_mul_pd(a1, b_vec));
        c2 = _mm_add_pd(c2, _mm_mul_pd(a2, b_vec));
        c3 = _mm_add_pd(c3, _mm_mul_pd(a3, b_vec));
    }

    // Scale by alpha
    let alpha_vec = _mm_set1_pd(alpha);
    c0 = _mm_mul_pd(c0, alpha_vec);
    c1 = _mm_mul_pd(c1, alpha_vec);
    c2 = _mm_mul_pd(c2, alpha_vec);
    c3 = _mm_mul_pd(c3, alpha_vec);

    // Store back to C with beta scaling
    let beta_vec = _mm_set1_pd(beta);

    for i in 0..4 {
        let c_ptr = c.add(i * rs_c);
        let c_row = match i {
            0 => c0,
            1 => c1,
            2 => c2,
            _ => c3,
        };

        if beta == 0.0 {
            _mm_storeu_pd(c_ptr, c_row);
        } else {
            let c_old = _mm_loadu_pd(c_ptr);
            let c_new = _mm_add_pd(_mm_mul_pd(c_old, beta_vec), c_row);
            _mm_storeu_pd(c_ptr, c_new);
        }
    }
}

// ==================== ARM NEON Micro-kernel for f64 (8x2) ====================

#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(clippy::identity_op)]
unsafe fn neon_micro_kernel_f64_8x2(
    kc: usize,
    alpha: f64,
    a: *const f64,
    b_packed: *const f64,
    beta: f64,
    c: *mut f64,
    rs_c: usize,
) {
    use std::arch::aarch64::*;

    // 8x2 = 8 rows x 2 cols (one NEON vector of 2 f64s per row)
    let mut c0 = vdupq_n_f64(0.0);
    let mut c1 = vdupq_n_f64(0.0);
    let mut c2 = vdupq_n_f64(0.0);
    let mut c3 = vdupq_n_f64(0.0);
    let mut c4 = vdupq_n_f64(0.0);
    let mut c5 = vdupq_n_f64(0.0);
    let mut c6 = vdupq_n_f64(0.0);
    let mut c7 = vdupq_n_f64(0.0);

    // Main accumulation loop
    #[allow(clippy::identity_op, clippy::erasing_op)]
    for p in 0..kc {
        let b_vec = vld1q_f64(b_packed.add(p * 2));

        let a0 = vdupq_n_f64(*a.add(0 * kc + p));
        let a1 = vdupq_n_f64(*a.add(1 * kc + p));
        let a2 = vdupq_n_f64(*a.add(2 * kc + p));
        let a3 = vdupq_n_f64(*a.add(3 * kc + p));
        let a4 = vdupq_n_f64(*a.add(4 * kc + p));
        let a5 = vdupq_n_f64(*a.add(5 * kc + p));
        let a6 = vdupq_n_f64(*a.add(6 * kc + p));
        let a7 = vdupq_n_f64(*a.add(7 * kc + p));

        // FMA on ARM: c = c + a * b
        c0 = vfmaq_f64(c0, a0, b_vec);
        c1 = vfmaq_f64(c1, a1, b_vec);
        c2 = vfmaq_f64(c2, a2, b_vec);
        c3 = vfmaq_f64(c3, a3, b_vec);
        c4 = vfmaq_f64(c4, a4, b_vec);
        c5 = vfmaq_f64(c5, a5, b_vec);
        c6 = vfmaq_f64(c6, a6, b_vec);
        c7 = vfmaq_f64(c7, a7, b_vec);
    }

    // Scale by alpha
    let alpha_vec = vdupq_n_f64(alpha);
    c0 = vmulq_f64(c0, alpha_vec);
    c1 = vmulq_f64(c1, alpha_vec);
    c2 = vmulq_f64(c2, alpha_vec);
    c3 = vmulq_f64(c3, alpha_vec);
    c4 = vmulq_f64(c4, alpha_vec);
    c5 = vmulq_f64(c5, alpha_vec);
    c6 = vmulq_f64(c6, alpha_vec);
    c7 = vmulq_f64(c7, alpha_vec);

    // Store back to C with beta scaling
    let beta_vec = vdupq_n_f64(beta);

    for i in 0..8 {
        let c_ptr = c.add(i * rs_c);
        let c_row = match i {
            0 => c0,
            1 => c1,
            2 => c2,
            3 => c3,
            4 => c4,
            5 => c5,
            6 => c6,
            _ => c7,
        };

        if beta == 0.0 {
            vst1q_f64(c_ptr, c_row);
        } else {
            let c_old = vld1q_f64(c_ptr);
            let c_new = vfmaq_f64(c_row, c_old, beta_vec);
            vst1q_f64(c_ptr, c_new);
        }
    }
}

// ==================== Scalar Fallback Micro-kernel for f64 ====================

#[inline(always)]
unsafe fn scalar_micro_kernel_f64(
    kc: usize,
    alpha: f64,
    a: *const f64,
    b_packed: *const f64,
    beta: f64,
    c: *mut f64,
    rs_c: usize,
    mr: usize,
    nr: usize,
) {
    // Compute C[mr x nr] += alpha * A[mr x kc] * B[kc x nr]
    for i in 0..mr {
        for j in 0..nr {
            let mut sum = 0.0f64;

            // Dot product of row i of A with column j of B
            for p in 0..kc {
                let a_val = *a.add(i * kc + p);
                let b_val = *b_packed.add(p * nr + j);
                sum += a_val * b_val;
            }

            let c_ptr = c.add(i * rs_c + j);
            if beta == 0.0 {
                *c_ptr = alpha * sum;
            } else {
                *c_ptr = beta * (*c_ptr) + alpha * sum;
            }
        }
    }
}
