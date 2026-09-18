//! NEON-optimized matrix operations for mobile platforms
//!
//! Optimized GEMM (General Matrix Multiply) and GEMV (General Matrix-Vector)
//! operations using ARM NEON intrinsics, with special focus on mobile battery
//! efficiency and cache optimization.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// NEON-optimized matrix-vector multiplication (GEMV) for f32
///
/// Computes y = alpha * A * x + beta * y
/// where A is m x n matrix in row-major format
///
/// # Safety
///
/// Caller must ensure the CPU supports NEON instructions (aarch64).
/// The slices `a` (length >= m*n), `x` (length >= n), and `y` (length >= m)
/// must be valid for reading/writing respectively.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_gemv_f32_impl(
    m: usize,
    n: usize,
    alpha: f32,
    a: &[f32],
    x: &[f32],
    beta: f32,
    y: &mut [f32],
) {
    let valpha = vdupq_n_f32(alpha);
    let vbeta = vdupq_n_f32(beta);

    for i in 0..m {
        let row_offset = i * n;
        let mut sum = vdupq_n_f32(0.0);
        let mut j = 0;

        // NEON vectorized loop - process 4 elements at a time
        while j + 4 <= n {
            let va = vld1q_f32(a.as_ptr().add(row_offset + j));
            let vx = vld1q_f32(x.as_ptr().add(j));
            sum = vfmaq_f32(sum, va, vx);
            j += 4;
        }

        // Horizontal reduction
        let mut dot = vaddvq_f32(sum);

        // Scalar remainder
        while j < n {
            dot += a[row_offset + j] * x[j];
            j += 1;
        }

        // y[i] = alpha * dot + beta * y[i]
        y[i] = alpha * dot + beta * y[i];
    }
}

pub fn neon_gemv_f32(
    m: usize,
    n: usize,
    alpha: f32,
    a: &[f32],
    x: &[f32],
    beta: f32,
    y: &mut [f32],
) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { neon_gemv_f32_impl(m, n, alpha, a, x, beta, y) }
        } else {
            fallback_gemv_f32(m, n, alpha, a, x, beta, y)
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        fallback_gemv_f32(m, n, alpha, a, x, beta, y)
    }
}

/// NEON-optimized matrix-vector multiplication (GEMV) for f64
///
/// # Safety
///
/// Caller must ensure the CPU supports NEON instructions (aarch64).
/// The slices `a` (length >= m*n), `x` (length >= n), and `y` (length >= m)
/// must be valid for reading/writing respectively.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_gemv_f64_impl(
    m: usize,
    n: usize,
    alpha: f64,
    a: &[f64],
    x: &[f64],
    beta: f64,
    y: &mut [f64],
) {
    for i in 0..m {
        let row_offset = i * n;
        let mut sum = vdupq_n_f64(0.0);
        let mut j = 0;

        // NEON vectorized loop - process 2 elements at a time
        while j + 2 <= n {
            let va = vld1q_f64(a.as_ptr().add(row_offset + j));
            let vx = vld1q_f64(x.as_ptr().add(j));
            sum = vfmaq_f64(sum, va, vx);
            j += 2;
        }

        // Horizontal reduction
        let mut dot = vaddvq_f64(sum);

        // Scalar remainder
        while j < n {
            dot += a[row_offset + j] * x[j];
            j += 1;
        }

        y[i] = alpha * dot + beta * y[i];
    }
}

pub fn neon_gemv_f64(
    m: usize,
    n: usize,
    alpha: f64,
    a: &[f64],
    x: &[f64],
    beta: f64,
    y: &mut [f64],
) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { neon_gemv_f64_impl(m, n, alpha, a, x, beta, y) }
        } else {
            fallback_gemv_f64(m, n, alpha, a, x, beta, y)
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        fallback_gemv_f64(m, n, alpha, a, x, beta, y)
    }
}

/// NEON-optimized matrix-matrix multiplication (GEMM) for f32
///
/// Computes C = alpha * A * B + beta * C
/// where A is m x k, B is k x n, C is m x n (all row-major)
///
/// Uses micro-kernel approach optimized for mobile cache hierarchies
///
/// # Safety
///
/// Caller must ensure the CPU supports NEON instructions (aarch64).
/// The slices `a` (length >= m*k), `b` (length >= k*n), and `c` (length >= m*n)
/// must be valid for reading/writing respectively.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_gemm_f32_impl(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
) {
    // Mobile-optimized block sizes (conservative for battery efficiency)
    const MC: usize = 64; // Rows of A to process at once
    const NC: usize = 256; // Columns of B to process at once
    const KC: usize = 128; // Common dimension blocking

    for jc in (0..n).step_by(NC) {
        let nc = (jc + NC).min(n) - jc;

        for pc in (0..k).step_by(KC) {
            let kc = (pc + KC).min(k) - pc;

            for ic in (0..m).step_by(MC) {
                let mc = (ic + MC).min(m) - ic;

                // Micro-kernel: optimized inner loops
                for i in 0..mc {
                    for j in 0..nc {
                        let mut sum = vdupq_n_f32(0.0);
                        let mut p = 0;

                        // Vectorized inner product
                        while p + 4 <= kc {
                            let va = vld1q_f32(a.as_ptr().add((ic + i) * k + pc + p));
                            let vb = vld1q_f32(b.as_ptr().add((pc + p) * n + jc + j));
                            sum = vfmaq_f32(sum, va, vb);
                            p += 4;
                        }

                        let mut dot = vaddvq_f32(sum);

                        // Scalar remainder
                        while p < kc {
                            dot += a[(ic + i) * k + pc + p] * b[(pc + p) * n + jc + j];
                            p += 1;
                        }

                        let c_idx = (ic + i) * n + jc + j;
                        if pc == 0 {
                            c[c_idx] = alpha * dot + beta * c[c_idx];
                        } else {
                            c[c_idx] += alpha * dot;
                        }
                    }
                }
            }
        }
    }
}

pub fn neon_gemm_f32(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { neon_gemm_f32_impl(m, n, k, alpha, a, b, beta, c) }
        } else {
            fallback_gemm_f32(m, n, k, alpha, a, b, beta, c)
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        fallback_gemm_f32(m, n, k, alpha, a, b, beta, c)
    }
}

/// NEON-optimized matrix-matrix multiplication (GEMM) for f64
///
/// # Safety
///
/// Caller must ensure the CPU supports NEON instructions (aarch64).
/// The slices `a` (length >= m*k), `b` (length >= k*n), and `c` (length >= m*n)
/// must be valid for reading/writing respectively.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_gemm_f64_impl(
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    a: &[f64],
    b: &[f64],
    beta: f64,
    c: &mut [f64],
) {
    const MC: usize = 32;
    const NC: usize = 128;
    const KC: usize = 64;

    for jc in (0..n).step_by(NC) {
        let nc = (jc + NC).min(n) - jc;

        for pc in (0..k).step_by(KC) {
            let kc = (pc + KC).min(k) - pc;

            for ic in (0..m).step_by(MC) {
                let mc = (ic + MC).min(m) - ic;

                for i in 0..mc {
                    for j in 0..nc {
                        let mut sum = vdupq_n_f64(0.0);
                        let mut p = 0;

                        while p + 2 <= kc {
                            let va = vld1q_f64(a.as_ptr().add((ic + i) * k + pc + p));
                            let vb = vld1q_f64(b.as_ptr().add((pc + p) * n + jc + j));
                            sum = vfmaq_f64(sum, va, vb);
                            p += 2;
                        }

                        let mut dot = vaddvq_f64(sum);

                        while p < kc {
                            dot += a[(ic + i) * k + pc + p] * b[(pc + p) * n + jc + j];
                            p += 1;
                        }

                        let c_idx = (ic + i) * n + jc + j;
                        if pc == 0 {
                            c[c_idx] = alpha * dot + beta * c[c_idx];
                        } else {
                            c[c_idx] += alpha * dot;
                        }
                    }
                }
            }
        }
    }
}

pub fn neon_gemm_f64(
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    a: &[f64],
    b: &[f64],
    beta: f64,
    c: &mut [f64],
) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { neon_gemm_f64_impl(m, n, k, alpha, a, b, beta, c) }
        } else {
            fallback_gemm_f64(m, n, k, alpha, a, b, beta, c)
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        fallback_gemm_f64(m, n, k, alpha, a, b, beta, c)
    }
}

// Fallback implementations
fn fallback_gemv_f32(
    m: usize,
    n: usize,
    alpha: f32,
    a: &[f32],
    x: &[f32],
    beta: f32,
    y: &mut [f32],
) {
    for i in 0..m {
        let mut sum = 0.0;
        for j in 0..n {
            sum += a[i * n + j] * x[j];
        }
        y[i] = alpha * sum + beta * y[i];
    }
}

fn fallback_gemv_f64(
    m: usize,
    n: usize,
    alpha: f64,
    a: &[f64],
    x: &[f64],
    beta: f64,
    y: &mut [f64],
) {
    for i in 0..m {
        let mut sum = 0.0;
        for j in 0..n {
            sum += a[i * n + j] * x[j];
        }
        y[i] = alpha * sum + beta * y[i];
    }
}

fn fallback_gemm_f32(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = alpha * sum + beta * c[i * n + j];
        }
    }
}

fn fallback_gemm_f64(
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    a: &[f64],
    b: &[f64],
    beta: f64,
    c: &mut [f64],
) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = alpha * sum + beta * c[i * n + j];
        }
    }
}
