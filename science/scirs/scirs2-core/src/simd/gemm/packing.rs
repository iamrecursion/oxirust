//! Memory packing functions for optimized data layout in GEMM
//!
//! This module provides functions to pack matrix data into cache-friendly formats.
//! Packing reorganizes data to improve:
//! - Spatial locality (consecutive memory access)
//! - Temporal locality (reuse data in cache)
//! - SIMD vectorization efficiency
//!
//! # Packing Strategy
//!
//! For B matrix packing:
//! - Pack KC x NR panels in column-major order
//! - Align to SIMD vector boundaries
//! - Optimize for streaming through micro-kernel

use super::config::MatMulConfig;

/// Pack a panel of B matrix for efficient micro-kernel access
///
/// Packs B[KC x NR] from the original matrix into a contiguous buffer
/// optimized for SIMD access patterns in the micro-kernel.
///
/// # Layout
///
/// Original B (row-major): B\[k\]\[j\] at b.add(k * ldb + j)
/// Packed B: B\[p\]\[j\] at buffer.add(p * NR + j) for p in 0..KC, j in 0..NR
///
/// This layout ensures:
/// - Each column of NR elements is contiguous (one SIMD load)
/// - Sequential access through k dimension
/// - No stride issues in micro-kernel
///
/// # Arguments
///
/// * `kc` - Number of rows to pack (inner dimension)
/// * `nr` - Number of columns to pack (micro-kernel width)
/// * `b` - Pointer to source B matrix
/// * `ldb` - Leading dimension of B (row stride)
/// * `buffer` - Destination buffer for packed data
///
/// # Safety
///
/// Caller must ensure:
/// - `b` points to valid memory with at least `kc * ldb` elements
/// - `buffer` has capacity for at least `kc * nr` elements
#[inline]
pub unsafe fn pack_b_f32(kc: usize, nr: usize, b: *const f32, ldb: usize, buffer: *mut f32) {
    // Pack B[kc x nr] into buffer in column-major micro-panel format
    for p in 0..kc {
        for j in 0..nr {
            let b_val = *b.add(p * ldb + j);
            *buffer.add(p * nr + j) = b_val;
        }
    }
}

/// Pack a panel of B matrix for f64 operations
///
/// Same as [`pack_b_f32`] but for double-precision data.
///
/// # Safety
///
/// Same safety requirements as [`pack_b_f32`]
#[inline]
pub unsafe fn pack_b_f64(kc: usize, nr: usize, b: *const f64, ldb: usize, buffer: *mut f64) {
    for p in 0..kc {
        for j in 0..nr {
            let b_val = *b.add(p * ldb + j);
            *buffer.add(p * nr + j) = b_val;
        }
    }
}

/// Pack a panel of A matrix for efficient micro-kernel access
///
/// Packs A[MR x KC] from the original matrix into a contiguous buffer.
/// While the micro-kernel can work with A in original layout, packing
/// can improve performance for very large problems.
///
/// # Layout
///
/// Original A (row-major): A\[i\]\[k\] at a.add(i * lda + k)
/// Packed A: A\[i\]\[k\] at buffer.add(i * KC + k) for i in 0..MR, k in 0..KC
///
/// # Arguments
///
/// * `mr` - Number of rows to pack (micro-kernel height)
/// * `kc` - Number of columns to pack (inner dimension)
/// * `a` - Pointer to source A matrix
/// * `lda` - Leading dimension of A (row stride)
/// * `buffer` - Destination buffer for packed data
///
/// # Safety
///
/// Caller must ensure:
/// - `a` points to valid memory with at least `mr * lda` elements
/// - `buffer` has capacity for at least `mr * kc` elements
#[inline]
pub unsafe fn pack_a_f32(mr: usize, kc: usize, a: *const f32, lda: usize, buffer: *mut f32) {
    // Pack A[mr x kc] into buffer in row-major micro-panel format
    for i in 0..mr {
        for k in 0..kc {
            let a_val = *a.add(i * lda + k);
            *buffer.add(i * kc + k) = a_val;
        }
    }
}

/// Pack a panel of A matrix for f64 operations
///
/// Same as [`pack_a_f32`] but for double-precision data.
///
/// # Safety
///
/// Same safety requirements as [`pack_a_f32`]
#[inline]
pub unsafe fn pack_a_f64(mr: usize, kc: usize, a: *const f64, lda: usize, buffer: *mut f64) {
    for i in 0..mr {
        for k in 0..kc {
            let a_val = *a.add(i * lda + k);
            *buffer.add(i * kc + k) = a_val;
        }
    }
}

/// Optimized pack for B matrix with vectorized copy (f32)
///
/// Uses SIMD instructions to accelerate the packing operation when possible.
/// Falls back to scalar implementation on unsupported architectures.
///
/// # Safety
///
/// Same safety requirements as [`pack_b_f32`]
#[inline]
pub unsafe fn pack_b_f32_fast(
    kc: usize,
    nr: usize,
    b: *const f32,
    ldb: usize,
    buffer: *mut f32,
    _config: &MatMulConfig,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && nr >= 8 {
            pack_b_f32_avx2(kc, nr, b, ldb, buffer);
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") && nr >= 4 {
            pack_b_f32_neon(kc, nr, b, ldb, buffer);
            return;
        }
    }

    // Fallback to scalar implementation
    pack_b_f32(kc, nr, b, ldb, buffer);
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn pack_b_f32_avx2(kc: usize, nr: usize, b: *const f32, ldb: usize, buffer: *mut f32) {
    use std::arch::x86_64::*;

    for p in 0..kc {
        let src = b.add(p * ldb);
        let dst = buffer.add(p * nr);

        let mut j = 0;

        // Process 8 elements at a time with AVX2
        while j + 8 <= nr {
            let values = _mm256_loadu_ps(src.add(j));
            _mm256_storeu_ps(dst.add(j), values);
            j += 8;
        }

        // Handle remaining elements
        while j < nr {
            *dst.add(j) = *src.add(j);
            j += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn pack_b_f32_neon(kc: usize, nr: usize, b: *const f32, ldb: usize, buffer: *mut f32) {
    use std::arch::aarch64::*;

    for p in 0..kc {
        let src = b.add(p * ldb);
        let dst = buffer.add(p * nr);

        let mut j = 0;

        // Process 4 elements at a time with NEON
        while j + 4 <= nr {
            let values = vld1q_f32(src.add(j));
            vst1q_f32(dst.add(j), values);
            j += 4;
        }

        // Handle remaining elements
        while j < nr {
            *dst.add(j) = *src.add(j);
            j += 1;
        }
    }
}

/// Optimized pack for B matrix with vectorized copy (f64)
///
/// Uses SIMD instructions to accelerate the packing operation when possible.
/// Falls back to scalar implementation on unsupported architectures.
///
/// # Safety
///
/// Same safety requirements as [`pack_b_f64`]
#[inline]
pub unsafe fn pack_b_f64_fast(
    kc: usize,
    nr: usize,
    b: *const f64,
    ldb: usize,
    buffer: *mut f64,
    _config: &MatMulConfig,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && nr >= 4 {
            pack_b_f64_avx2(kc, nr, b, ldb, buffer);
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") && nr >= 2 {
            pack_b_f64_neon(kc, nr, b, ldb, buffer);
            return;
        }
    }

    // Fallback to scalar implementation
    pack_b_f64(kc, nr, b, ldb, buffer);
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn pack_b_f64_avx2(kc: usize, nr: usize, b: *const f64, ldb: usize, buffer: *mut f64) {
    use std::arch::x86_64::*;

    for p in 0..kc {
        let src = b.add(p * ldb);
        let dst = buffer.add(p * nr);

        let mut j = 0;

        // Process 4 elements at a time with AVX2
        while j + 4 <= nr {
            let values = _mm256_loadu_pd(src.add(j));
            _mm256_storeu_pd(dst.add(j), values);
            j += 4;
        }

        // Handle remaining elements
        while j < nr {
            *dst.add(j) = *src.add(j);
            j += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn pack_b_f64_neon(kc: usize, nr: usize, b: *const f64, ldb: usize, buffer: *mut f64) {
    use std::arch::aarch64::*;

    for p in 0..kc {
        let src = b.add(p * ldb);
        let dst = buffer.add(p * nr);

        let mut j = 0;

        // Process 2 elements at a time with NEON
        while j + 2 <= nr {
            let values = vld1q_f64(src.add(j));
            vst1q_f64(dst.add(j), values);
            j += 2;
        }

        // Handle remaining elements
        while j < nr {
            *dst.add(j) = *src.add(j);
            j += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_b_f32_small() {
        let kc = 4;
        let nr = 4;
        let ldb = 8;

        // Create test matrix B (4x8, we'll pack 4x4 of it)
        let b: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let mut packed = vec![0.0f32; kc * nr];

        unsafe {
            pack_b_f32(kc, nr, b.as_ptr(), ldb, packed.as_mut_ptr());
        }

        // Verify packing: packed[p*nr + j] should equal b[p*ldb + j]
        for p in 0..kc {
            for j in 0..nr {
                let expected = b[p * ldb + j];
                let actual = packed[p * nr + j];
                assert_eq!(
                    actual, expected,
                    "Mismatch at p={}, j={}: expected {}, got {}",
                    p, j, expected, actual
                );
            }
        }
    }

    #[test]
    fn test_pack_a_f32_small() {
        let mr = 4;
        let kc = 4;
        let lda = 8;

        // Create test matrix A (4x8, we'll pack 4x4 of it)
        let a: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let mut packed = vec![0.0f32; mr * kc];

        unsafe {
            pack_a_f32(mr, kc, a.as_ptr(), lda, packed.as_mut_ptr());
        }

        // Verify packing: packed[i*kc + k] should equal a[i*lda + k]
        for i in 0..mr {
            for k in 0..kc {
                let expected = a[i * lda + k];
                let actual = packed[i * kc + k];
                assert_eq!(
                    actual, expected,
                    "Mismatch at i={}, k={}: expected {}, got {}",
                    i, k, expected, actual
                );
            }
        }
    }

    #[test]
    fn test_pack_b_f64() {
        let kc = 2;
        let nr = 2;
        let ldb = 4;

        let b: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut packed = vec![0.0f64; kc * nr];

        unsafe {
            pack_b_f64(kc, nr, b.as_ptr(), ldb, packed.as_mut_ptr());
        }

        // Check packing is correct
        assert_eq!(packed[0], 1.0); // b[0][0]
        assert_eq!(packed[1], 2.0); // b[0][1]
        assert_eq!(packed[2], 5.0); // b[1][0]
        assert_eq!(packed[3], 6.0); // b[1][1]
    }
}
