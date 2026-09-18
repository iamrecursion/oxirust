//! ARM NEON SIMD implementation.
//!
//! This module provides optimized implementations using ARM NEON instructions,
//! available on all ARMv7-A with NEON (2008+) and all ARMv8-A/AArch64 (2011+) processors.

#![allow(unsafe_code)]
#![allow(
    clippy::transmute_undefined_repr,
    clippy::missing_transmute_annotations
)]

use crate::simd::traits::{SimdOps, SimdOpsExt};
use crate::simd::types::{I16x8, I32x4, U8x16};

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// ARM NEON SIMD implementation.
#[derive(Clone, Copy, Debug)]
pub struct NeonSimd;

impl NeonSimd {
    /// Create a new NEON SIMD instance.
    ///
    /// # Safety
    ///
    /// On AArch64, NEON is always available. On ARMv7, the caller must
    /// ensure NEON is available before calling SIMD operations.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Check if NEON is available at runtime.
    #[inline]
    #[must_use]
    pub fn is_available() -> bool {
        #[cfg(target_arch = "aarch64")]
        {
            // On AArch64, NEON is always available
            true
        }
        #[cfg(all(target_arch = "arm", target_feature = "neon"))]
        {
            true
        }
        #[cfg(not(any(
            target_arch = "aarch64",
            all(target_arch = "arm", target_feature = "neon")
        )))]
        {
            false
        }
    }
}

impl SimdOps for NeonSimd {
    #[inline]
    fn name(&self) -> &'static str {
        "neon"
    }

    #[inline]
    fn is_available(&self) -> bool {
        Self::is_available()
    }

    // ========================================================================
    // Vector Arithmetic
    // ========================================================================

    #[inline]
    #[cfg(target_arch = "aarch64")]
    fn add_i16x8(&self, a: I16x8, b: I16x8) -> I16x8 {
        // SAFETY: NEON is always available on AArch64
        unsafe {
            let a_vec = vld1q_s16(a.as_ptr());
            let b_vec = vld1q_s16(b.as_ptr());
            let result = vaddq_s16(a_vec, b_vec);
            let mut out = I16x8::zero();
            vst1q_s16(out.as_mut_ptr(), result);
            out
        }
    }

    #[inline]
    #[cfg(not(target_arch = "aarch64"))]
    fn add_i16x8(&self, a: I16x8, b: I16x8) -> I16x8 {
        let mut result = I16x8::zero();
        for i in 0..8 {
            result[i] = a[i].wrapping_add(b[i]);
        }
        result
    }

    #[inline]
    #[cfg(target_arch = "aarch64")]
    fn sub_i16x8(&self, a: I16x8, b: I16x8) -> I16x8 {
        unsafe {
            let a_vec = vld1q_s16(a.as_ptr());
            let b_vec = vld1q_s16(b.as_ptr());
            let result = vsubq_s16(a_vec, b_vec);
            let mut out = I16x8::zero();
            vst1q_s16(out.as_mut_ptr(), result);
            out
        }
    }

    #[inline]
    #[cfg(not(target_arch = "aarch64"))]
    fn sub_i16x8(&self, a: I16x8, b: I16x8) -> I16x8 {
        let mut result = I16x8::zero();
        for i in 0..8 {
            result[i] = a[i].wrapping_sub(b[i]);
        }
        result
    }

    #[inline]
    #[cfg(target_arch = "aarch64")]
    fn mul_i16x8(&self, a: I16x8, b: I16x8) -> I16x8 {
        unsafe {
            let a_vec = vld1q_s16(a.as_ptr());
            let b_vec = vld1q_s16(b.as_ptr());
            let result = vmulq_s16(a_vec, b_vec);
            let mut out = I16x8::zero();
            vst1q_s16(out.as_mut_ptr(), result);
            out
        }
    }

    #[inline]
    #[cfg(not(target_arch = "aarch64"))]
    fn mul_i16x8(&self, a: I16x8, b: I16x8) -> I16x8 {
        let mut result = I16x8::zero();
        for i in 0..8 {
            result[i] = a[i].wrapping_mul(b[i]);
        }
        result
    }

    #[inline]
    #[cfg(target_arch = "aarch64")]
    fn add_i32x4(&self, a: I32x4, b: I32x4) -> I32x4 {
        unsafe {
            let a_vec = vld1q_s32(a.as_ptr());
            let b_vec = vld1q_s32(b.as_ptr());
            let result = vaddq_s32(a_vec, b_vec);
            let mut out = I32x4::zero();
            vst1q_s32(out.as_mut_ptr(), result);
            out
        }
    }

    #[inline]
    #[cfg(not(target_arch = "aarch64"))]
    fn add_i32x4(&self, a: I32x4, b: I32x4) -> I32x4 {
        let mut result = I32x4::zero();
        for i in 0..4 {
            result[i] = a[i].wrapping_add(b[i]);
        }
        result
    }

    #[inline]
    #[cfg(target_arch = "aarch64")]
    fn sub_i32x4(&self, a: I32x4, b: I32x4) -> I32x4 {
        unsafe {
            let a_vec = vld1q_s32(a.as_ptr());
            let b_vec = vld1q_s32(b.as_ptr());
            let result = vsubq_s32(a_vec, b_vec);
            let mut out = I32x4::zero();
            vst1q_s32(out.as_mut_ptr(), result);
            out
        }
    }

    #[inline]
    #[cfg(not(target_arch = "aarch64"))]
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
    #[cfg(target_arch = "aarch64")]
    fn min_i16x8(&self, a: I16x8, b: I16x8) -> I16x8 {
        unsafe {
            let a_vec = vld1q_s16(a.as_ptr());
            let b_vec = vld1q_s16(b.as_ptr());
            let result = vminq_s16(a_vec, b_vec);
            let mut out = I16x8::zero();
            vst1q_s16(out.as_mut_ptr(), result);
            out
        }
    }

    #[inline]
    #[cfg(not(target_arch = "aarch64"))]
    fn min_i16x8(&self, a: I16x8, b: I16x8) -> I16x8 {
        let mut result = I16x8::zero();
        for i in 0..8 {
            result[i] = a[i].min(b[i]);
        }
        result
    }

    #[inline]
    #[cfg(target_arch = "aarch64")]
    fn max_i16x8(&self, a: I16x8, b: I16x8) -> I16x8 {
        unsafe {
            let a_vec = vld1q_s16(a.as_ptr());
            let b_vec = vld1q_s16(b.as_ptr());
            let result = vmaxq_s16(a_vec, b_vec);
            let mut out = I16x8::zero();
            vst1q_s16(out.as_mut_ptr(), result);
            out
        }
    }

    #[inline]
    #[cfg(not(target_arch = "aarch64"))]
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
    #[cfg(target_arch = "aarch64")]
    fn min_u8x16(&self, a: U8x16, b: U8x16) -> U8x16 {
        unsafe {
            let a_vec = vld1q_u8(a.as_ptr());
            let b_vec = vld1q_u8(b.as_ptr());
            let result = vminq_u8(a_vec, b_vec);
            let mut out = U8x16::zero();
            vst1q_u8(out.as_mut_ptr(), result);
            out
        }
    }

    #[inline]
    #[cfg(not(target_arch = "aarch64"))]
    fn min_u8x16(&self, a: U8x16, b: U8x16) -> U8x16 {
        let mut result = U8x16::zero();
        for i in 0..16 {
            result[i] = a[i].min(b[i]);
        }
        result
    }

    #[inline]
    #[cfg(target_arch = "aarch64")]
    fn max_u8x16(&self, a: U8x16, b: U8x16) -> U8x16 {
        unsafe {
            let a_vec = vld1q_u8(a.as_ptr());
            let b_vec = vld1q_u8(b.as_ptr());
            let result = vmaxq_u8(a_vec, b_vec);
            let mut out = U8x16::zero();
            vst1q_u8(out.as_mut_ptr(), result);
            out
        }
    }

    #[inline]
    #[cfg(not(target_arch = "aarch64"))]
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
    #[cfg(target_arch = "aarch64")]
    fn horizontal_sum_i16x8(&self, v: I16x8) -> i32 {
        unsafe {
            let vec = vld1q_s16(v.as_ptr());
            // Add pairs: 8 elements -> 4 elements
            let pair_sum = vpaddlq_s16(vec);
            // Add pairs again: 4 elements -> 2 elements
            let quad_sum = vpaddlq_s32(pair_sum);
            // Final horizontal add
            let arr: [i64; 2] = std::mem::transmute(quad_sum);
            (arr[0] + arr[1]) as i32
        }
    }

    #[inline]
    #[cfg(not(target_arch = "aarch64"))]
    fn horizontal_sum_i16x8(&self, v: I16x8) -> i32 {
        v.iter().map(|&x| i32::from(x)).sum()
    }

    #[inline]
    #[cfg(target_arch = "aarch64")]
    fn horizontal_sum_i32x4(&self, v: I32x4) -> i32 {
        unsafe {
            let vec = vld1q_s32(v.as_ptr());
            let pair_sum = vpaddlq_s32(vec);
            let arr: [i64; 2] = std::mem::transmute(pair_sum);
            (arr[0] + arr[1]) as i32
        }
    }

    #[inline]
    #[cfg(not(target_arch = "aarch64"))]
    fn horizontal_sum_i32x4(&self, v: I32x4) -> i32 {
        v.iter().sum()
    }

    // ========================================================================
    // SAD (Sum of Absolute Differences)
    // ========================================================================

    #[inline]
    #[cfg(target_arch = "aarch64")]
    fn sad_u8x16(&self, a: U8x16, b: U8x16) -> u32 {
        unsafe {
            let a_vec = vld1q_u8(a.as_ptr());
            let b_vec = vld1q_u8(b.as_ptr());

            // Compute absolute difference
            let diff = vabdq_u8(a_vec, b_vec);

            // Sum all elements by repeated pairwise addition
            let sum16 = vpaddlq_u8(diff); // 16xu8 -> 8xu16
            let sum32 = vpaddlq_u16(sum16); // 8xu16 -> 4xu32
            let sum64 = vpaddlq_u32(sum32); // 4xu32 -> 2xu64

            let arr: [u64; 2] = std::mem::transmute(sum64);
            (arr[0] + arr[1]) as u32
        }
    }

    #[inline]
    #[cfg(not(target_arch = "aarch64"))]
    fn sad_u8x16(&self, a: U8x16, b: U8x16) -> u32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y): (&u8, &u8)| u32::from(x.abs_diff(y)))
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
    #[cfg(target_arch = "aarch64")]
    fn widen_low_u8_to_i16(&self, v: U8x16) -> I16x8 {
        unsafe {
            let vec = vld1q_u8(v.as_ptr());
            let low = vget_low_u8(vec);
            let widened = vmovl_u8(low);
            let mut out = I16x8::zero();
            vst1q_s16(out.as_mut_ptr(), std::mem::transmute(widened));
            out
        }
    }

    #[inline]
    #[cfg(not(target_arch = "aarch64"))]
    fn widen_low_u8_to_i16(&self, v: U8x16) -> I16x8 {
        let mut result = I16x8::zero();
        for i in 0..8 {
            result[i] = i16::from(v[i]);
        }
        result
    }

    #[inline]
    #[cfg(target_arch = "aarch64")]
    fn widen_high_u8_to_i16(&self, v: U8x16) -> I16x8 {
        unsafe {
            let vec = vld1q_u8(v.as_ptr());
            let high = vget_high_u8(vec);
            let widened = vmovl_u8(high);
            let mut out = I16x8::zero();
            vst1q_s16(out.as_mut_ptr(), std::mem::transmute(widened));
            out
        }
    }

    #[inline]
    #[cfg(not(target_arch = "aarch64"))]
    fn widen_high_u8_to_i16(&self, v: U8x16) -> I16x8 {
        let mut result = I16x8::zero();
        for i in 0..8 {
            result[i] = i16::from(v[i + 8]);
        }
        result
    }

    #[inline]
    #[cfg(target_arch = "aarch64")]
    fn narrow_i32x4_to_i16x8(&self, low: I32x4, high: I32x4) -> I16x8 {
        unsafe {
            let low_vec = vld1q_s32(low.as_ptr());
            let high_vec = vld1q_s32(high.as_ptr());
            let narrow_low = vqmovn_s32(low_vec);
            let narrow_high = vqmovn_s32(high_vec);
            let result = vcombine_s16(narrow_low, narrow_high);
            let mut out = I16x8::zero();
            vst1q_s16(out.as_mut_ptr(), result);
            out
        }
    }

    #[inline]
    #[cfg(not(target_arch = "aarch64"))]
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
    #[cfg(target_arch = "aarch64")]
    fn madd_i16x8(&self, a: I16x8, b: I16x8, c: I16x8) -> I16x8 {
        unsafe {
            let a_vec = vld1q_s16(a.as_ptr());
            let b_vec = vld1q_s16(b.as_ptr());
            let c_vec = vld1q_s16(c.as_ptr());
            let result = vmlaq_s16(c_vec, a_vec, b_vec);
            let mut out = I16x8::zero();
            vst1q_s16(out.as_mut_ptr(), result);
            out
        }
    }

    #[inline]
    #[cfg(not(target_arch = "aarch64"))]
    fn madd_i16x8(&self, a: I16x8, b: I16x8, c: I16x8) -> I16x8 {
        let mut result = I16x8::zero();
        for i in 0..8 {
            result[i] = a[i].wrapping_mul(b[i]).wrapping_add(c[i]);
        }
        result
    }

    #[inline]
    fn pmaddwd(&self, a: I16x8, b: I16x8) -> I32x4 {
        // NEON doesn't have a direct pmaddwd equivalent
        // Emulate: multiply pairs and add adjacent results
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
    #[cfg(target_arch = "aarch64")]
    fn shr_i16x8(&self, v: I16x8, shift: u32) -> I16x8 {
        unsafe {
            let vec = vld1q_s16(v.as_ptr());
            let shift_vec = vdupq_n_s16(-(shift as i16));
            let result = vshlq_s16(vec, shift_vec);
            let mut out = I16x8::zero();
            vst1q_s16(out.as_mut_ptr(), result);
            out
        }
    }

    #[inline]
    #[cfg(not(target_arch = "aarch64"))]
    fn shr_i16x8(&self, v: I16x8, shift: u32) -> I16x8 {
        let mut result = I16x8::zero();
        for i in 0..8 {
            result[i] = v[i] >> shift;
        }
        result
    }

    #[inline]
    #[cfg(target_arch = "aarch64")]
    fn shl_i16x8(&self, v: I16x8, shift: u32) -> I16x8 {
        unsafe {
            let vec = vld1q_s16(v.as_ptr());
            let shift_vec = vdupq_n_s16(shift as i16);
            let result = vshlq_s16(vec, shift_vec);
            let mut out = I16x8::zero();
            vst1q_s16(out.as_mut_ptr(), result);
            out
        }
    }

    #[inline]
    #[cfg(not(target_arch = "aarch64"))]
    fn shl_i16x8(&self, v: I16x8, shift: u32) -> I16x8 {
        let mut result = I16x8::zero();
        for i in 0..8 {
            result[i] = v[i] << shift;
        }
        result
    }

    #[inline]
    #[cfg(target_arch = "aarch64")]
    fn shr_i32x4(&self, v: I32x4, shift: u32) -> I32x4 {
        unsafe {
            let vec = vld1q_s32(v.as_ptr());
            let shift_vec = vdupq_n_s32(-(shift as i32));
            let result = vshlq_s32(vec, shift_vec);
            let mut out = I32x4::zero();
            vst1q_s32(out.as_mut_ptr(), result);
            out
        }
    }

    #[inline]
    #[cfg(not(target_arch = "aarch64"))]
    fn shr_i32x4(&self, v: I32x4, shift: u32) -> I32x4 {
        let mut result = I32x4::zero();
        for i in 0..4 {
            result[i] = v[i] >> shift;
        }
        result
    }

    #[inline]
    #[cfg(target_arch = "aarch64")]
    fn shl_i32x4(&self, v: I32x4, shift: u32) -> I32x4 {
        unsafe {
            let vec = vld1q_s32(v.as_ptr());
            let shift_vec = vdupq_n_s32(shift as i32);
            let result = vshlq_s32(vec, shift_vec);
            let mut out = I32x4::zero();
            vst1q_s32(out.as_mut_ptr(), result);
            out
        }
    }

    #[inline]
    #[cfg(not(target_arch = "aarch64"))]
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
    #[cfg(target_arch = "aarch64")]
    fn avg_u8x16(&self, a: U8x16, b: U8x16) -> U8x16 {
        unsafe {
            let a_vec = vld1q_u8(a.as_ptr());
            let b_vec = vld1q_u8(b.as_ptr());
            let result = vrhaddq_u8(a_vec, b_vec); // Rounding halving add
            let mut out = U8x16::zero();
            vst1q_u8(out.as_mut_ptr(), result);
            out
        }
    }

    #[inline]
    #[cfg(not(target_arch = "aarch64"))]
    fn avg_u8x16(&self, a: U8x16, b: U8x16) -> U8x16 {
        let mut result = U8x16::zero();
        for i in 0..16 {
            result[i] = ((u16::from(a[i]) + u16::from(b[i]) + 1) / 2) as u8;
        }
        result
    }
}

impl SimdOpsExt for NeonSimd {
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
        #[cfg(target_arch = "aarch64")]
        {
            unsafe {
                // Load 4x4 matrix (only first 4 elements of each row)
                let r0 = vld1_s16(rows[0].as_ptr());
                let r1 = vld1_s16(rows[1].as_ptr());
                let r2 = vld1_s16(rows[2].as_ptr());
                let r3 = vld1_s16(rows[3].as_ptr());

                // Transpose using interleaving
                let t0 = vtrn_s16(r0, r1);
                let t1 = vtrn_s16(r2, r3);

                let t2 = vtrn_s32(std::mem::transmute(t0.0), std::mem::transmute(t1.0));
                let t3 = vtrn_s32(std::mem::transmute(t0.1), std::mem::transmute(t1.1));

                let mut out = [I16x8::zero(); 4];
                vst1_s16(out[0].as_mut_ptr(), std::mem::transmute(t2.0));
                vst1_s16(out[1].as_mut_ptr(), std::mem::transmute(t2.1));
                vst1_s16(out[2].as_mut_ptr(), std::mem::transmute(t3.0));
                vst1_s16(out[3].as_mut_ptr(), std::mem::transmute(t3.1));
                out
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
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
        #[cfg(target_arch = "aarch64")]
        {
            unsafe {
                // Load all 8 rows
                let r0 = vld1q_s16(rows[0].as_ptr());
                let r1 = vld1q_s16(rows[1].as_ptr());
                let r2 = vld1q_s16(rows[2].as_ptr());
                let r3 = vld1q_s16(rows[3].as_ptr());
                let r4 = vld1q_s16(rows[4].as_ptr());
                let r5 = vld1q_s16(rows[5].as_ptr());
                let r6 = vld1q_s16(rows[6].as_ptr());
                let r7 = vld1q_s16(rows[7].as_ptr());

                // First level of interleaving (16-bit)
                let t0 = vtrnq_s16(r0, r1);
                let t1 = vtrnq_s16(r2, r3);
                let t2 = vtrnq_s16(r4, r5);
                let t3 = vtrnq_s16(r6, r7);

                // Second level (32-bit)
                let u0 = vtrnq_s32(std::mem::transmute(t0.0), std::mem::transmute(t1.0));
                let u1 = vtrnq_s32(std::mem::transmute(t0.1), std::mem::transmute(t1.1));
                let u2 = vtrnq_s32(std::mem::transmute(t2.0), std::mem::transmute(t3.0));
                let u3 = vtrnq_s32(std::mem::transmute(t2.1), std::mem::transmute(t3.1));

                // Third level (64-bit) - using vtrn is limited, use manual construction
                let o0 = vcombine_s16(
                    vget_low_s16(std::mem::transmute(u0.0)),
                    vget_low_s16(std::mem::transmute(u2.0)),
                );
                let o1 = vcombine_s16(
                    vget_low_s16(std::mem::transmute(u0.1)),
                    vget_low_s16(std::mem::transmute(u2.1)),
                );
                let o2 = vcombine_s16(
                    vget_low_s16(std::mem::transmute(u1.0)),
                    vget_low_s16(std::mem::transmute(u3.0)),
                );
                let o3 = vcombine_s16(
                    vget_low_s16(std::mem::transmute(u1.1)),
                    vget_low_s16(std::mem::transmute(u3.1)),
                );
                let o4 = vcombine_s16(
                    vget_high_s16(std::mem::transmute(u0.0)),
                    vget_high_s16(std::mem::transmute(u2.0)),
                );
                let o5 = vcombine_s16(
                    vget_high_s16(std::mem::transmute(u0.1)),
                    vget_high_s16(std::mem::transmute(u2.1)),
                );
                let o6 = vcombine_s16(
                    vget_high_s16(std::mem::transmute(u1.0)),
                    vget_high_s16(std::mem::transmute(u3.0)),
                );
                let o7 = vcombine_s16(
                    vget_high_s16(std::mem::transmute(u1.1)),
                    vget_high_s16(std::mem::transmute(u3.1)),
                );

                let mut out = [I16x8::zero(); 8];
                vst1q_s16(out[0].as_mut_ptr(), o0);
                vst1q_s16(out[1].as_mut_ptr(), o1);
                vst1q_s16(out[2].as_mut_ptr(), o2);
                vst1q_s16(out[3].as_mut_ptr(), o3);
                vst1q_s16(out[4].as_mut_ptr(), o4);
                vst1q_s16(out[5].as_mut_ptr(), o5);
                vst1q_s16(out[6].as_mut_ptr(), o6);
                vst1q_s16(out[7].as_mut_ptr(), o7);
                out
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
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
