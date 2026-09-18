//! AVX2 SIMD implementation for x86_64.
//!
//! This module provides optimized implementations of SIMD operations
//! using AVX2 instructions, available on Intel Haswell (2013) and later,
//! and AMD Excavator (2015) and later processors.

#![allow(unsafe_code)]

use crate::simd::traits::{SimdOps, SimdOpsExt};
use crate::simd::types::{I16x16, I16x8, I32x4, I32x8, U8x16, U8x32};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// AVX2 SIMD implementation.
#[derive(Clone, Copy, Debug)]
pub struct Avx2Simd;

impl Avx2Simd {
    /// Create a new AVX2 SIMD instance.
    ///
    /// # Safety
    ///
    /// The caller must ensure that AVX2 is available on the current CPU.
    /// Use `is_available()` to check before calling SIMD operations.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Check if AVX2 is available at runtime.
    #[inline]
    #[must_use]
    pub fn is_available() -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            is_x86_feature_detected!("avx2")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }
}

impl SimdOps for Avx2Simd {
    #[inline]
    fn name(&self) -> &'static str {
        "avx2"
    }

    #[inline]
    fn is_available(&self) -> bool {
        Self::is_available()
    }

    // ========================================================================
    // Vector Arithmetic
    // ========================================================================

    #[inline]
    #[cfg(target_arch = "x86_64")]
    fn add_i16x8(&self, a: I16x8, b: I16x8) -> I16x8 {
        // SAFETY: AVX2 is checked at runtime before calling this
        unsafe {
            let a_vec = _mm_loadu_si128(a.as_ptr().cast());
            let b_vec = _mm_loadu_si128(b.as_ptr().cast());
            let result = _mm_add_epi16(a_vec, b_vec);
            let mut out = I16x8::zero();
            _mm_storeu_si128(out.as_mut_ptr().cast(), result);
            out
        }
    }

    #[inline]
    #[cfg(not(target_arch = "x86_64"))]
    fn add_i16x8(&self, a: I16x8, b: I16x8) -> I16x8 {
        let mut result = I16x8::zero();
        for i in 0..8 {
            result[i] = a[i].wrapping_add(b[i]);
        }
        result
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    fn sub_i16x8(&self, a: I16x8, b: I16x8) -> I16x8 {
        // SAFETY: AVX2 is checked at runtime
        unsafe {
            let a_vec = _mm_loadu_si128(a.as_ptr().cast());
            let b_vec = _mm_loadu_si128(b.as_ptr().cast());
            let result = _mm_sub_epi16(a_vec, b_vec);
            let mut out = I16x8::zero();
            _mm_storeu_si128(out.as_mut_ptr().cast(), result);
            out
        }
    }

    #[inline]
    #[cfg(not(target_arch = "x86_64"))]
    fn sub_i16x8(&self, a: I16x8, b: I16x8) -> I16x8 {
        let mut result = I16x8::zero();
        for i in 0..8 {
            result[i] = a[i].wrapping_sub(b[i]);
        }
        result
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    fn mul_i16x8(&self, a: I16x8, b: I16x8) -> I16x8 {
        // SAFETY: AVX2 is checked at runtime
        unsafe {
            let a_vec = _mm_loadu_si128(a.as_ptr().cast());
            let b_vec = _mm_loadu_si128(b.as_ptr().cast());
            let result = _mm_mullo_epi16(a_vec, b_vec);
            let mut out = I16x8::zero();
            _mm_storeu_si128(out.as_mut_ptr().cast(), result);
            out
        }
    }

    #[inline]
    #[cfg(not(target_arch = "x86_64"))]
    fn mul_i16x8(&self, a: I16x8, b: I16x8) -> I16x8 {
        let mut result = I16x8::zero();
        for i in 0..8 {
            result[i] = a[i].wrapping_mul(b[i]);
        }
        result
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    fn add_i32x4(&self, a: I32x4, b: I32x4) -> I32x4 {
        // SAFETY: AVX2 is checked at runtime
        unsafe {
            let a_vec = _mm_loadu_si128(a.as_ptr().cast());
            let b_vec = _mm_loadu_si128(b.as_ptr().cast());
            let result = _mm_add_epi32(a_vec, b_vec);
            let mut out = I32x4::zero();
            _mm_storeu_si128(out.as_mut_ptr().cast(), result);
            out
        }
    }

    #[inline]
    #[cfg(not(target_arch = "x86_64"))]
    fn add_i32x4(&self, a: I32x4, b: I32x4) -> I32x4 {
        let mut result = I32x4::zero();
        for i in 0..4 {
            result[i] = a[i].wrapping_add(b[i]);
        }
        result
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    fn sub_i32x4(&self, a: I32x4, b: I32x4) -> I32x4 {
        // SAFETY: AVX2 is checked at runtime
        unsafe {
            let a_vec = _mm_loadu_si128(a.as_ptr().cast());
            let b_vec = _mm_loadu_si128(b.as_ptr().cast());
            let result = _mm_sub_epi32(a_vec, b_vec);
            let mut out = I32x4::zero();
            _mm_storeu_si128(out.as_mut_ptr().cast(), result);
            out
        }
    }

    #[inline]
    #[cfg(not(target_arch = "x86_64"))]
    fn sub_i32x4(&self, a: I32x4, b: I32x4) -> I32x4 {
        let mut result = I32x4::zero();
        for i in 0..4 {
            result[i] = a[i].wrapping_sub(b[i]);
        }
        result
    }

    // ========================================================================
    // Min/Max/Clamp
    // ========================================================================

    #[inline]
    #[cfg(target_arch = "x86_64")]
    fn min_i16x8(&self, a: I16x8, b: I16x8) -> I16x8 {
        // SAFETY: AVX2 is checked at runtime
        unsafe {
            let a_vec = _mm_loadu_si128(a.as_ptr().cast());
            let b_vec = _mm_loadu_si128(b.as_ptr().cast());
            let result = _mm_min_epi16(a_vec, b_vec);
            let mut out = I16x8::zero();
            _mm_storeu_si128(out.as_mut_ptr().cast(), result);
            out
        }
    }

    #[inline]
    #[cfg(not(target_arch = "x86_64"))]
    fn min_i16x8(&self, a: I16x8, b: I16x8) -> I16x8 {
        let mut result = I16x8::zero();
        for i in 0..8 {
            result[i] = a[i].min(b[i]);
        }
        result
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    fn max_i16x8(&self, a: I16x8, b: I16x8) -> I16x8 {
        // SAFETY: AVX2 is checked at runtime
        unsafe {
            let a_vec = _mm_loadu_si128(a.as_ptr().cast());
            let b_vec = _mm_loadu_si128(b.as_ptr().cast());
            let result = _mm_max_epi16(a_vec, b_vec);
            let mut out = I16x8::zero();
            _mm_storeu_si128(out.as_mut_ptr().cast(), result);
            out
        }
    }

    #[inline]
    #[cfg(not(target_arch = "x86_64"))]
    fn max_i16x8(&self, a: I16x8, b: I16x8) -> I16x8 {
        let mut result = I16x8::zero();
        for i in 0..8 {
            result[i] = a[i].max(b[i]);
        }
        result
    }

    #[inline]
    fn clamp_i16x8(&self, v: I16x8, min: i16, max: i16) -> I16x8 {
        let min_vec = I16x8::splat(min);
        let max_vec = I16x8::splat(max);
        let clamped_min = self.max_i16x8(v, min_vec);
        self.min_i16x8(clamped_min, max_vec)
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    fn min_u8x16(&self, a: U8x16, b: U8x16) -> U8x16 {
        // SAFETY: AVX2 is checked at runtime
        unsafe {
            let a_vec = _mm_loadu_si128(a.as_ptr().cast());
            let b_vec = _mm_loadu_si128(b.as_ptr().cast());
            let result = _mm_min_epu8(a_vec, b_vec);
            let mut out = U8x16::zero();
            _mm_storeu_si128(out.as_mut_ptr().cast(), result);
            out
        }
    }

    #[inline]
    #[cfg(not(target_arch = "x86_64"))]
    fn min_u8x16(&self, a: U8x16, b: U8x16) -> U8x16 {
        let mut result = U8x16::zero();
        for i in 0..16 {
            result[i] = a[i].min(b[i]);
        }
        result
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    fn max_u8x16(&self, a: U8x16, b: U8x16) -> U8x16 {
        // SAFETY: AVX2 is checked at runtime
        unsafe {
            let a_vec = _mm_loadu_si128(a.as_ptr().cast());
            let b_vec = _mm_loadu_si128(b.as_ptr().cast());
            let result = _mm_max_epu8(a_vec, b_vec);
            let mut out = U8x16::zero();
            _mm_storeu_si128(out.as_mut_ptr().cast(), result);
            out
        }
    }

    #[inline]
    #[cfg(not(target_arch = "x86_64"))]
    fn max_u8x16(&self, a: U8x16, b: U8x16) -> U8x16 {
        let mut result = U8x16::zero();
        for i in 0..16 {
            result[i] = a[i].max(b[i]);
        }
        result
    }

    #[inline]
    fn clamp_u8x16(&self, v: U8x16, min: u8, max: u8) -> U8x16 {
        let min_vec = U8x16::splat(min);
        let max_vec = U8x16::splat(max);
        let clamped_min = self.max_u8x16(v, min_vec);
        self.min_u8x16(clamped_min, max_vec)
    }

    // ========================================================================
    // Horizontal Operations
    // ========================================================================

    #[inline]
    #[cfg(target_arch = "x86_64")]
    fn horizontal_sum_i16x8(&self, v: I16x8) -> i32 {
        // SAFETY: AVX2 is checked at runtime
        unsafe {
            let vec = _mm_loadu_si128(v.as_ptr().cast());
            // Horizontal add to get pairs
            let sum1 = _mm_hadd_epi16(vec, vec);
            let sum2 = _mm_hadd_epi16(sum1, sum1);
            let sum3 = _mm_hadd_epi16(sum2, sum2);
            _mm_extract_epi16(sum3, 0) as i16 as i32
        }
    }

    #[inline]
    #[cfg(not(target_arch = "x86_64"))]
    fn horizontal_sum_i16x8(&self, v: I16x8) -> i32 {
        v.iter().map(|&x| i32::from(x)).sum()
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    fn horizontal_sum_i32x4(&self, v: I32x4) -> i32 {
        // SAFETY: AVX2 is checked at runtime
        unsafe {
            let vec = _mm_loadu_si128(v.as_ptr().cast());
            let sum1 = _mm_hadd_epi32(vec, vec);
            let sum2 = _mm_hadd_epi32(sum1, sum1);
            _mm_extract_epi32(sum2, 0)
        }
    }

    #[inline]
    #[cfg(not(target_arch = "x86_64"))]
    fn horizontal_sum_i32x4(&self, v: I32x4) -> i32 {
        v.iter().sum()
    }

    // ========================================================================
    // SAD (Sum of Absolute Differences)
    // ========================================================================

    #[inline]
    #[cfg(target_arch = "x86_64")]
    fn sad_u8x16(&self, a: U8x16, b: U8x16) -> u32 {
        // SAFETY: AVX2 is checked at runtime
        unsafe {
            let a_vec = _mm_loadu_si128(a.as_ptr().cast());
            let b_vec = _mm_loadu_si128(b.as_ptr().cast());
            let sad = _mm_sad_epu8(a_vec, b_vec);
            let low = _mm_extract_epi64(sad, 0) as u32;
            let high = _mm_extract_epi64(sad, 1) as u32;
            low + high
        }
    }

    #[inline]
    #[cfg(not(target_arch = "x86_64"))]
    fn sad_u8x16(&self, a: U8x16, b: U8x16) -> u32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| u32::from(x.abs_diff(y)))
            .sum()
    }

    #[inline]
    fn sad_8(&self, a: &[u8], b: &[u8]) -> u32 {
        assert!(a.len() >= 8 && b.len() >= 8);
        a[..8]
            .iter()
            .zip(b[..8].iter())
            .map(|(&x, &y)| u32::from(x.abs_diff(y)))
            .sum()
    }

    #[inline]
    fn sad_16(&self, a: &[u8], b: &[u8]) -> u32 {
        assert!(a.len() >= 16 && b.len() >= 16);
        let mut a_vec = U8x16::zero();
        let mut b_vec = U8x16::zero();
        a_vec.copy_from_slice(&a[..16]);
        b_vec.copy_from_slice(&b[..16]);
        self.sad_u8x16(a_vec, b_vec)
    }

    // ========================================================================
    // Widening/Narrowing
    // ========================================================================

    #[inline]
    #[cfg(target_arch = "x86_64")]
    fn widen_low_u8_to_i16(&self, v: U8x16) -> I16x8 {
        // SAFETY: AVX2 is checked at runtime
        unsafe {
            let vec = _mm_loadu_si128(v.as_ptr().cast());
            let zero = _mm_setzero_si128();
            let result = _mm_unpacklo_epi8(vec, zero);
            let mut out = I16x8::zero();
            _mm_storeu_si128(out.as_mut_ptr().cast(), result);
            out
        }
    }

    #[inline]
    #[cfg(not(target_arch = "x86_64"))]
    fn widen_low_u8_to_i16(&self, v: U8x16) -> I16x8 {
        let mut result = I16x8::zero();
        for i in 0..8 {
            result[i] = i16::from(v[i]);
        }
        result
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    fn widen_high_u8_to_i16(&self, v: U8x16) -> I16x8 {
        // SAFETY: AVX2 is checked at runtime
        unsafe {
            let vec = _mm_loadu_si128(v.as_ptr().cast());
            let zero = _mm_setzero_si128();
            let result = _mm_unpackhi_epi8(vec, zero);
            let mut out = I16x8::zero();
            _mm_storeu_si128(out.as_mut_ptr().cast(), result);
            out
        }
    }

    #[inline]
    #[cfg(not(target_arch = "x86_64"))]
    fn widen_high_u8_to_i16(&self, v: U8x16) -> I16x8 {
        let mut result = I16x8::zero();
        for i in 0..8 {
            result[i] = i16::from(v[i + 8]);
        }
        result
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    fn narrow_i32x4_to_i16x8(&self, low: I32x4, high: I32x4) -> I16x8 {
        // SAFETY: AVX2 is checked at runtime
        unsafe {
            let low_vec = _mm_loadu_si128(low.as_ptr().cast());
            let high_vec = _mm_loadu_si128(high.as_ptr().cast());
            let result = _mm_packs_epi32(low_vec, high_vec);
            let mut out = I16x8::zero();
            _mm_storeu_si128(out.as_mut_ptr().cast(), result);
            out
        }
    }

    #[inline]
    #[cfg(not(target_arch = "x86_64"))]
    fn narrow_i32x4_to_i16x8(&self, low: I32x4, high: I32x4) -> I16x8 {
        let mut result = I16x8::zero();
        for i in 0..4 {
            result[i] = low[i].clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
            result[i + 4] = high[i].clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
        }
        result
    }

    // ========================================================================
    // Multiply-Add
    // ========================================================================

    #[inline]
    fn madd_i16x8(&self, a: I16x8, b: I16x8, c: I16x8) -> I16x8 {
        let prod = self.mul_i16x8(a, b);
        self.add_i16x8(prod, c)
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    fn pmaddwd(&self, a: I16x8, b: I16x8) -> I32x4 {
        // SAFETY: AVX2 is checked at runtime
        unsafe {
            let a_vec = _mm_loadu_si128(a.as_ptr().cast());
            let b_vec = _mm_loadu_si128(b.as_ptr().cast());
            let result = _mm_madd_epi16(a_vec, b_vec);
            let mut out = I32x4::zero();
            _mm_storeu_si128(out.as_mut_ptr().cast(), result);
            out
        }
    }

    #[inline]
    #[cfg(not(target_arch = "x86_64"))]
    fn pmaddwd(&self, a: I16x8, b: I16x8) -> I32x4 {
        let mut result = I32x4::zero();
        for i in 0..4 {
            result[i] = i32::from(a[i * 2]) * i32::from(b[i * 2])
                + i32::from(a[i * 2 + 1]) * i32::from(b[i * 2 + 1]);
        }
        result
    }

    // ========================================================================
    // Shift Operations
    // ========================================================================

    #[inline]
    #[cfg(target_arch = "x86_64")]
    fn shr_i16x8(&self, v: I16x8, shift: u32) -> I16x8 {
        // SAFETY: AVX2 is checked at runtime
        unsafe {
            let vec = _mm_loadu_si128(v.as_ptr().cast());
            let shift_vec = _mm_cvtsi32_si128(shift as i32);
            let result = _mm_sra_epi16(vec, shift_vec);
            let mut out = I16x8::zero();
            _mm_storeu_si128(out.as_mut_ptr().cast(), result);
            out
        }
    }

    #[inline]
    #[cfg(not(target_arch = "x86_64"))]
    fn shr_i16x8(&self, v: I16x8, shift: u32) -> I16x8 {
        let mut result = I16x8::zero();
        for i in 0..8 {
            result[i] = v[i] >> shift;
        }
        result
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    fn shl_i16x8(&self, v: I16x8, shift: u32) -> I16x8 {
        // SAFETY: AVX2 is checked at runtime
        unsafe {
            let vec = _mm_loadu_si128(v.as_ptr().cast());
            let shift_vec = _mm_cvtsi32_si128(shift as i32);
            let result = _mm_sll_epi16(vec, shift_vec);
            let mut out = I16x8::zero();
            _mm_storeu_si128(out.as_mut_ptr().cast(), result);
            out
        }
    }

    #[inline]
    #[cfg(not(target_arch = "x86_64"))]
    fn shl_i16x8(&self, v: I16x8, shift: u32) -> I16x8 {
        let mut result = I16x8::zero();
        for i in 0..8 {
            result[i] = v[i] << shift;
        }
        result
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    fn shr_i32x4(&self, v: I32x4, shift: u32) -> I32x4 {
        // SAFETY: AVX2 is checked at runtime
        unsafe {
            let vec = _mm_loadu_si128(v.as_ptr().cast());
            let shift_vec = _mm_cvtsi32_si128(shift as i32);
            let result = _mm_sra_epi32(vec, shift_vec);
            let mut out = I32x4::zero();
            _mm_storeu_si128(out.as_mut_ptr().cast(), result);
            out
        }
    }

    #[inline]
    #[cfg(not(target_arch = "x86_64"))]
    fn shr_i32x4(&self, v: I32x4, shift: u32) -> I32x4 {
        let mut result = I32x4::zero();
        for i in 0..4 {
            result[i] = v[i] >> shift;
        }
        result
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    fn shl_i32x4(&self, v: I32x4, shift: u32) -> I32x4 {
        // SAFETY: AVX2 is checked at runtime
        unsafe {
            let vec = _mm_loadu_si128(v.as_ptr().cast());
            let shift_vec = _mm_cvtsi32_si128(shift as i32);
            let result = _mm_sll_epi32(vec, shift_vec);
            let mut out = I32x4::zero();
            _mm_storeu_si128(out.as_mut_ptr().cast(), result);
            out
        }
    }

    #[inline]
    #[cfg(not(target_arch = "x86_64"))]
    fn shl_i32x4(&self, v: I32x4, shift: u32) -> I32x4 {
        let mut result = I32x4::zero();
        for i in 0..4 {
            result[i] = v[i] << shift;
        }
        result
    }

    // ========================================================================
    // Averaging
    // ========================================================================

    #[inline]
    #[cfg(target_arch = "x86_64")]
    fn avg_u8x16(&self, a: U8x16, b: U8x16) -> U8x16 {
        // SAFETY: AVX2 is checked at runtime
        unsafe {
            let a_vec = _mm_loadu_si128(a.as_ptr().cast());
            let b_vec = _mm_loadu_si128(b.as_ptr().cast());
            let result = _mm_avg_epu8(a_vec, b_vec);
            let mut out = U8x16::zero();
            _mm_storeu_si128(out.as_mut_ptr().cast(), result);
            out
        }
    }

    #[inline]
    #[cfg(not(target_arch = "x86_64"))]
    fn avg_u8x16(&self, a: U8x16, b: U8x16) -> U8x16 {
        let mut result = U8x16::zero();
        for i in 0..16 {
            result[i] = ((u16::from(a[i]) + u16::from(b[i]) + 1) / 2) as u8;
        }
        result
    }
}

impl SimdOpsExt for Avx2Simd {
    #[inline]
    fn load4_u8_to_i16x8(&self, src: &[u8]) -> I16x8 {
        assert!(src.len() >= 4);
        let mut result = I16x8::zero();
        for i in 0..4 {
            result[i] = i16::from(src[i]);
        }
        result
    }

    #[inline]
    fn load8_u8_to_i16x8(&self, src: &[u8]) -> I16x8 {
        assert!(src.len() >= 8);
        let mut result = I16x8::zero();
        for i in 0..8 {
            result[i] = i16::from(src[i]);
        }
        result
    }

    #[inline]
    fn store4_i16x8_as_u8(&self, v: I16x8, dst: &mut [u8]) {
        assert!(dst.len() >= 4);
        for i in 0..4 {
            dst[i] = v[i].clamp(0, 255) as u8;
        }
    }

    #[inline]
    fn store8_i16x8_as_u8(&self, v: I16x8, dst: &mut [u8]) {
        assert!(dst.len() >= 8);
        for i in 0..8 {
            dst[i] = v[i].clamp(0, 255) as u8;
        }
    }

    #[inline]
    fn transpose_4x4_i16(&self, rows: &[I16x8; 4]) -> [I16x8; 4] {
        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: AVX2 is checked at runtime
            unsafe {
                // Load 4 rows
                let r0 = _mm_loadl_epi64(rows[0].as_ptr().cast());
                let r1 = _mm_loadl_epi64(rows[1].as_ptr().cast());
                let r2 = _mm_loadl_epi64(rows[2].as_ptr().cast());
                let r3 = _mm_loadl_epi64(rows[3].as_ptr().cast());

                // Interleave pairs
                let t0 = _mm_unpacklo_epi16(r0, r1);
                let t1 = _mm_unpacklo_epi16(r2, r3);

                // Final interleave
                let o0 = _mm_unpacklo_epi32(t0, t1);
                let o1 = _mm_unpackhi_epi32(t0, t1);
                let o2 = _mm_unpacklo_epi32(_mm_unpackhi_epi16(r0, r1), _mm_unpackhi_epi16(r2, r3));
                let o3 = _mm_unpackhi_epi32(_mm_unpackhi_epi16(r0, r1), _mm_unpackhi_epi16(r2, r3));

                let mut out = [I16x8::zero(); 4];
                _mm_storeu_si128(out[0].as_mut_ptr().cast(), o0);
                _mm_storeu_si128(out[1].as_mut_ptr().cast(), o1);
                _mm_storeu_si128(out[2].as_mut_ptr().cast(), o2);
                _mm_storeu_si128(out[3].as_mut_ptr().cast(), o3);
                out
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let mut out = [I16x8::zero(); 4];
            for i in 0..4 {
                for j in 0..4 {
                    out[i][j] = rows[j][i];
                }
            }
            out
        }
    }

    #[inline]
    fn transpose_8x8_i16(&self, rows: &[I16x8; 8]) -> [I16x8; 8] {
        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: AVX2 is checked at runtime
            unsafe {
                // Load all 8 rows
                let r0 = _mm_loadu_si128(rows[0].as_ptr().cast());
                let r1 = _mm_loadu_si128(rows[1].as_ptr().cast());
                let r2 = _mm_loadu_si128(rows[2].as_ptr().cast());
                let r3 = _mm_loadu_si128(rows[3].as_ptr().cast());
                let r4 = _mm_loadu_si128(rows[4].as_ptr().cast());
                let r5 = _mm_loadu_si128(rows[5].as_ptr().cast());
                let r6 = _mm_loadu_si128(rows[6].as_ptr().cast());
                let r7 = _mm_loadu_si128(rows[7].as_ptr().cast());

                // First level of interleaving
                let t0 = _mm_unpacklo_epi16(r0, r1);
                let t1 = _mm_unpackhi_epi16(r0, r1);
                let t2 = _mm_unpacklo_epi16(r2, r3);
                let t3 = _mm_unpackhi_epi16(r2, r3);
                let t4 = _mm_unpacklo_epi16(r4, r5);
                let t5 = _mm_unpackhi_epi16(r4, r5);
                let t6 = _mm_unpacklo_epi16(r6, r7);
                let t7 = _mm_unpackhi_epi16(r6, r7);

                // Second level
                let u0 = _mm_unpacklo_epi32(t0, t2);
                let u1 = _mm_unpackhi_epi32(t0, t2);
                let u2 = _mm_unpacklo_epi32(t1, t3);
                let u3 = _mm_unpackhi_epi32(t1, t3);
                let u4 = _mm_unpacklo_epi32(t4, t6);
                let u5 = _mm_unpackhi_epi32(t4, t6);
                let u6 = _mm_unpacklo_epi32(t5, t7);
                let u7 = _mm_unpackhi_epi32(t5, t7);

                // Third level
                let o0 = _mm_unpacklo_epi64(u0, u4);
                let o1 = _mm_unpackhi_epi64(u0, u4);
                let o2 = _mm_unpacklo_epi64(u1, u5);
                let o3 = _mm_unpackhi_epi64(u1, u5);
                let o4 = _mm_unpacklo_epi64(u2, u6);
                let o5 = _mm_unpackhi_epi64(u2, u6);
                let o6 = _mm_unpacklo_epi64(u3, u7);
                let o7 = _mm_unpackhi_epi64(u3, u7);

                let mut out = [I16x8::zero(); 8];
                _mm_storeu_si128(out[0].as_mut_ptr().cast(), o0);
                _mm_storeu_si128(out[1].as_mut_ptr().cast(), o1);
                _mm_storeu_si128(out[2].as_mut_ptr().cast(), o2);
                _mm_storeu_si128(out[3].as_mut_ptr().cast(), o3);
                _mm_storeu_si128(out[4].as_mut_ptr().cast(), o4);
                _mm_storeu_si128(out[5].as_mut_ptr().cast(), o5);
                _mm_storeu_si128(out[6].as_mut_ptr().cast(), o6);
                _mm_storeu_si128(out[7].as_mut_ptr().cast(), o7);
                out
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let mut out = [I16x8::zero(); 8];
            for i in 0..8 {
                for j in 0..8 {
                    out[i][j] = rows[j][i];
                }
            }
            out
        }
    }

    #[inline]
    fn butterfly_i16x8(&self, a: I16x8, b: I16x8) -> (I16x8, I16x8) {
        let sum = self.add_i16x8(a, b);
        let diff = self.sub_i16x8(a, b);
        (sum, diff)
    }

    #[inline]
    fn butterfly_i32x4(&self, a: I32x4, b: I32x4) -> (I32x4, I32x4) {
        let sum = self.add_i32x4(a, b);
        let diff = self.sub_i32x4(a, b);
        (sum, diff)
    }
}
