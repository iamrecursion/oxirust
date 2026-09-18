//! SIMD operation traits for video codec implementations.
//!
//! This module defines the core traits for SIMD operations. All implementations
//! (scalar fallback, SSE, AVX, NEON) implement these traits, allowing codec
//! code to be written generically.
//!
//! # Example
//!
//! ```ignore
//! use oximedia_codec::simd::{SimdOps, get_simd_impl};
//!
//! let simd = get_simd_impl();
//! let sum = simd.horizontal_sum_i16(&[1, 2, 3, 4, 5, 6, 7, 8]);
//! ```

#![forbid(unsafe_code)]

use super::types::{I16x8, I32x4, U8x16};

/// Core SIMD operations trait.
///
/// This trait defines the fundamental SIMD operations needed for video codec
/// implementations. All operations are designed to map efficiently to
/// hardware SIMD instructions.
pub trait SimdOps: Send + Sync {
    /// Get the name of this SIMD implementation.
    fn name(&self) -> &'static str;

    /// Check if this implementation is available on the current CPU.
    fn is_available(&self) -> bool;

    // ========================================================================
    // Vector Arithmetic
    // ========================================================================

    /// Element-wise addition of two i16x8 vectors.
    fn add_i16x8(&self, a: I16x8, b: I16x8) -> I16x8;

    /// Element-wise subtraction of two i16x8 vectors.
    fn sub_i16x8(&self, a: I16x8, b: I16x8) -> I16x8;

    /// Element-wise multiplication of two i16x8 vectors.
    fn mul_i16x8(&self, a: I16x8, b: I16x8) -> I16x8;

    /// Element-wise addition of two i32x4 vectors.
    fn add_i32x4(&self, a: I32x4, b: I32x4) -> I32x4;

    /// Element-wise subtraction of two i32x4 vectors.
    fn sub_i32x4(&self, a: I32x4, b: I32x4) -> I32x4;

    // ========================================================================
    // Min/Max/Clamp
    // ========================================================================

    /// Element-wise minimum of two i16x8 vectors.
    fn min_i16x8(&self, a: I16x8, b: I16x8) -> I16x8;

    /// Element-wise maximum of two i16x8 vectors.
    fn max_i16x8(&self, a: I16x8, b: I16x8) -> I16x8;

    /// Element-wise clamp of i16x8 vector.
    fn clamp_i16x8(&self, v: I16x8, min: i16, max: i16) -> I16x8;

    /// Element-wise minimum of two u8x16 vectors.
    fn min_u8x16(&self, a: U8x16, b: U8x16) -> U8x16;

    /// Element-wise maximum of two u8x16 vectors.
    fn max_u8x16(&self, a: U8x16, b: U8x16) -> U8x16;

    /// Element-wise clamp of u8x16 vector.
    fn clamp_u8x16(&self, v: U8x16, min: u8, max: u8) -> U8x16;

    // ========================================================================
    // Horizontal Operations
    // ========================================================================

    /// Horizontal sum of all elements in an i16x8 vector.
    fn horizontal_sum_i16x8(&self, v: I16x8) -> i32;

    /// Horizontal sum of all elements in an i32x4 vector.
    fn horizontal_sum_i32x4(&self, v: I32x4) -> i32;

    // ========================================================================
    // SAD (Sum of Absolute Differences)
    // ========================================================================

    /// Sum of absolute differences between two u8x16 vectors.
    ///
    /// Computes: sum(|a\[i\] - b\[i\]|) for all i
    fn sad_u8x16(&self, a: U8x16, b: U8x16) -> u32;

    /// Sum of absolute differences for 8 bytes.
    fn sad_8(&self, a: &[u8], b: &[u8]) -> u32;

    /// Sum of absolute differences for 16 bytes.
    fn sad_16(&self, a: &[u8], b: &[u8]) -> u32;

    // ========================================================================
    // Widening/Narrowing
    // ========================================================================

    /// Widen u8x16 low half to i16x8.
    fn widen_low_u8_to_i16(&self, v: U8x16) -> I16x8;

    /// Widen u8x16 high half to i16x8.
    fn widen_high_u8_to_i16(&self, v: U8x16) -> I16x8;

    /// Narrow two i32x4 to i16x8 with saturation.
    fn narrow_i32x4_to_i16x8(&self, low: I32x4, high: I32x4) -> I16x8;

    // ========================================================================
    // Multiply-Add
    // ========================================================================

    /// Multiply and add: a * b + c for i16x8.
    fn madd_i16x8(&self, a: I16x8, b: I16x8, c: I16x8) -> I16x8;

    /// Multiply pairs and add adjacent results (pmaddwd equivalent).
    ///
    /// Multiplies pairs of i16 elements and adds adjacent products:
    /// result\[0\] = a\[0\]*b\[0\] + a\[1\]*b\[1\]
    /// result\[1\] = a\[2\]*b\[2\] + a\[3\]*b\[3\]
    /// etc.
    fn pmaddwd(&self, a: I16x8, b: I16x8) -> I32x4;

    // ========================================================================
    // Shift Operations
    // ========================================================================

    /// Arithmetic right shift of i16x8 by immediate.
    fn shr_i16x8(&self, v: I16x8, shift: u32) -> I16x8;

    /// Logical left shift of i16x8 by immediate.
    fn shl_i16x8(&self, v: I16x8, shift: u32) -> I16x8;

    /// Arithmetic right shift of i32x4 by immediate.
    fn shr_i32x4(&self, v: I32x4, shift: u32) -> I32x4;

    /// Logical left shift of i32x4 by immediate.
    fn shl_i32x4(&self, v: I32x4, shift: u32) -> I32x4;

    // ========================================================================
    // Averaging
    // ========================================================================

    /// Average of two u8x16 vectors (rounding up).
    fn avg_u8x16(&self, a: U8x16, b: U8x16) -> U8x16;
}

/// Extended SIMD operations for more complex codec operations.
pub trait SimdOpsExt: SimdOps {
    // ========================================================================
    // Block Operations
    // ========================================================================

    /// Load 4 bytes from memory and zero-extend to i16x8.
    fn load4_u8_to_i16x8(&self, src: &[u8]) -> I16x8;

    /// Load 8 bytes from memory and zero-extend to i16x8.
    fn load8_u8_to_i16x8(&self, src: &[u8]) -> I16x8;

    /// Store lower 4 elements of i16x8 to memory as saturated u8.
    fn store4_i16x8_as_u8(&self, v: I16x8, dst: &mut [u8]);

    /// Store lower 8 elements of i16x8 to memory as saturated u8.
    fn store8_i16x8_as_u8(&self, v: I16x8, dst: &mut [u8]);

    // ========================================================================
    // Transpose Operations (for DCT)
    // ========================================================================

    /// Transpose 4x4 block of i16 values.
    ///
    /// Input: 4 rows stored in 4 I16x8 vectors (only lower 4 elements used)
    /// Output: Transposed 4x4 block
    fn transpose_4x4_i16(&self, rows: &[I16x8; 4]) -> [I16x8; 4];

    /// Transpose 8x8 block of i16 values.
    fn transpose_8x8_i16(&self, rows: &[I16x8; 8]) -> [I16x8; 8];

    // ========================================================================
    // Butterfly Operations (for DCT)
    // ========================================================================

    /// DCT butterfly: (a + b, a - b).
    fn butterfly_i16x8(&self, a: I16x8, b: I16x8) -> (I16x8, I16x8);

    /// DCT butterfly for i32x4.
    fn butterfly_i32x4(&self, a: I32x4, b: I32x4) -> (I32x4, I32x4);
}

/// Trait for selecting SIMD implementation at runtime.
pub trait SimdSelector {
    /// Get the best available SIMD implementation for this CPU.
    fn select(&self) -> &dyn SimdOps;

    /// Get the best available extended SIMD implementation.
    fn select_ext(&self) -> &dyn SimdOpsExt;
}

#[cfg(test)]
mod tests {
    use super::*;

    // Basic trait bound tests
    fn _assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn test_trait_bounds() {
        #[allow(dead_code)]
        #[allow(clippy::used_underscore_items)]
        fn assert_simd_ops<T: SimdOps>() {
            _assert_send_sync::<T>();
        }

        // This test just ensures the trait bounds compile correctly
        fn _check_bounds<T: SimdOps>(_t: &T) {
            assert_simd_ops::<T>();
        }
    }
}
