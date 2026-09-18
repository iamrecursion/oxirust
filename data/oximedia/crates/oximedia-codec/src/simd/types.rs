//! Common SIMD types and type aliases.
//!
//! This module defines type aliases for common SIMD vector types used in
//! video codec implementations. These types abstract over the underlying
//! SIMD implementation (scalar fallback, SSE, AVX, NEON, etc.).
//!
//! # Naming Convention
//!
//! Types follow the pattern `{element_type}x{lane_count}`:
//! - `i16x8` - 8 lanes of `i16` (128-bit)
//! - `i32x4` - 4 lanes of `i32` (128-bit)
//! - `u8x16` - 16 lanes of `u8` (128-bit)

use std::ops::{Add, Index, IndexMut, Mul, Sub};

/// 8-lane vector of 16-bit signed integers (128-bit).
///
/// Common uses:
/// - DCT coefficients
/// - Pixel differences
/// - Filter taps
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct I16x8(pub [i16; 8]);

/// 16-lane vector of 16-bit signed integers (256-bit).
///
/// Common uses:
/// - Wide DCT operations
/// - Parallel coefficient processing
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct I16x16(pub [i16; 16]);

/// 4-lane vector of 32-bit signed integers (128-bit).
///
/// Common uses:
/// - Accumulated DCT results
/// - Intermediate filter calculations
/// - SAD accumulation
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct I32x4(pub [i32; 4]);

/// 8-lane vector of 32-bit signed integers (256-bit).
///
/// Common uses:
/// - Wide accumulation
/// - 8-point parallel operations
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct I32x8(pub [i32; 8]);

/// 16-lane vector of 8-bit unsigned integers (128-bit).
///
/// Common uses:
/// - Raw pixel data
/// - SAD calculations
/// - Luma/chroma samples
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct U8x16(pub [u8; 16]);

/// 32-lane vector of 8-bit unsigned integers (256-bit).
///
/// Common uses:
/// - Wide pixel operations
/// - AVX-width processing
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct U8x32(pub [u8; 32]);

// ============================================================================
// I16x8 Implementation
// ============================================================================

impl I16x8 {
    /// Create a new vector with all lanes set to zero.
    #[inline]
    #[must_use]
    pub const fn zero() -> Self {
        Self([0; 8])
    }

    /// Create a new vector with all lanes set to the same value.
    #[inline]
    #[must_use]
    pub const fn splat(value: i16) -> Self {
        Self([value; 8])
    }

    /// Create a vector from an array.
    #[inline]
    #[must_use]
    pub const fn from_array(arr: [i16; 8]) -> Self {
        Self(arr)
    }

    /// Convert to an array.
    #[inline]
    #[must_use]
    pub const fn to_array(self) -> [i16; 8] {
        self.0
    }

    /// Get element at index.
    #[inline]
    #[must_use]
    pub fn get(&self, index: usize) -> i16 {
        self.0[index]
    }

    /// Set element at index.
    #[inline]
    pub fn set(&mut self, index: usize, value: i16) {
        self.0[index] = value;
    }

    /// Widen to I32x4 (low half).
    #[inline]
    #[must_use]
    pub fn widen_low(self) -> I32x4 {
        I32x4([
            i32::from(self.0[0]),
            i32::from(self.0[1]),
            i32::from(self.0[2]),
            i32::from(self.0[3]),
        ])
    }

    /// Widen to I32x4 (high half).
    #[inline]
    #[must_use]
    pub fn widen_high(self) -> I32x4 {
        I32x4([
            i32::from(self.0[4]),
            i32::from(self.0[5]),
            i32::from(self.0[6]),
            i32::from(self.0[7]),
        ])
    }

    /// Get a pointer to the underlying array.
    #[inline]
    #[must_use]
    pub const fn as_ptr(&self) -> *const i16 {
        self.0.as_ptr()
    }

    /// Get a mutable pointer to the underlying array.
    #[inline]
    #[must_use]
    pub fn as_mut_ptr(&mut self) -> *mut i16 {
        self.0.as_mut_ptr()
    }

    /// Get an iterator over the elements.
    #[inline]
    pub fn iter(&self) -> std::slice::Iter<'_, i16> {
        self.0.iter()
    }

    /// Copy elements from a slice.
    #[inline]
    pub fn copy_from_slice(&mut self, src: &[i16]) {
        self.0.copy_from_slice(src);
    }
}

impl Add for I16x8 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self([
            self.0[0].wrapping_add(rhs.0[0]),
            self.0[1].wrapping_add(rhs.0[1]),
            self.0[2].wrapping_add(rhs.0[2]),
            self.0[3].wrapping_add(rhs.0[3]),
            self.0[4].wrapping_add(rhs.0[4]),
            self.0[5].wrapping_add(rhs.0[5]),
            self.0[6].wrapping_add(rhs.0[6]),
            self.0[7].wrapping_add(rhs.0[7]),
        ])
    }
}

impl Sub for I16x8 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self([
            self.0[0].wrapping_sub(rhs.0[0]),
            self.0[1].wrapping_sub(rhs.0[1]),
            self.0[2].wrapping_sub(rhs.0[2]),
            self.0[3].wrapping_sub(rhs.0[3]),
            self.0[4].wrapping_sub(rhs.0[4]),
            self.0[5].wrapping_sub(rhs.0[5]),
            self.0[6].wrapping_sub(rhs.0[6]),
            self.0[7].wrapping_sub(rhs.0[7]),
        ])
    }
}

impl Mul for I16x8 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self([
            self.0[0].wrapping_mul(rhs.0[0]),
            self.0[1].wrapping_mul(rhs.0[1]),
            self.0[2].wrapping_mul(rhs.0[2]),
            self.0[3].wrapping_mul(rhs.0[3]),
            self.0[4].wrapping_mul(rhs.0[4]),
            self.0[5].wrapping_mul(rhs.0[5]),
            self.0[6].wrapping_mul(rhs.0[6]),
            self.0[7].wrapping_mul(rhs.0[7]),
        ])
    }
}

impl Index<usize> for I16x8 {
    type Output = i16;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl IndexMut<usize> for I16x8 {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

// ============================================================================
// I16x16 Implementation
// ============================================================================

impl I16x16 {
    /// Create a new vector with all lanes set to zero.
    #[inline]
    #[must_use]
    pub const fn zero() -> Self {
        Self([0; 16])
    }

    /// Create a new vector with all lanes set to the same value.
    #[inline]
    #[must_use]
    pub const fn splat(value: i16) -> Self {
        Self([value; 16])
    }

    /// Create a vector from an array.
    #[inline]
    #[must_use]
    pub const fn from_array(arr: [i16; 16]) -> Self {
        Self(arr)
    }

    /// Convert to an array.
    #[inline]
    #[must_use]
    pub const fn to_array(self) -> [i16; 16] {
        self.0
    }
}

// ============================================================================
// I32x4 Implementation
// ============================================================================

impl I32x4 {
    /// Create a new vector with all lanes set to zero.
    #[inline]
    #[must_use]
    pub const fn zero() -> Self {
        Self([0; 4])
    }

    /// Create a new vector with all lanes set to the same value.
    #[inline]
    #[must_use]
    pub const fn splat(value: i32) -> Self {
        Self([value; 4])
    }

    /// Create a vector from an array.
    #[inline]
    #[must_use]
    pub const fn from_array(arr: [i32; 4]) -> Self {
        Self(arr)
    }

    /// Convert to an array.
    #[inline]
    #[must_use]
    pub const fn to_array(self) -> [i32; 4] {
        self.0
    }

    /// Horizontal sum of all elements.
    #[inline]
    #[must_use]
    pub fn horizontal_sum(self) -> i32 {
        self.0[0]
            .wrapping_add(self.0[1])
            .wrapping_add(self.0[2])
            .wrapping_add(self.0[3])
    }

    /// Narrow to I16x8 with another I32x4 (saturating).
    #[inline]
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn narrow_sat(self, high: Self) -> I16x8 {
        let saturate = |v: i32| -> i16 { v.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16 };
        I16x8([
            saturate(self.0[0]),
            saturate(self.0[1]),
            saturate(self.0[2]),
            saturate(self.0[3]),
            saturate(high.0[0]),
            saturate(high.0[1]),
            saturate(high.0[2]),
            saturate(high.0[3]),
        ])
    }

    /// Get a pointer to the underlying array.
    #[inline]
    #[must_use]
    pub const fn as_ptr(&self) -> *const i32 {
        self.0.as_ptr()
    }

    /// Get a mutable pointer to the underlying array.
    #[inline]
    #[must_use]
    pub fn as_mut_ptr(&mut self) -> *mut i32 {
        self.0.as_mut_ptr()
    }

    /// Get an iterator over the elements.
    #[inline]
    pub fn iter(&self) -> std::slice::Iter<'_, i32> {
        self.0.iter()
    }
}

impl Add for I32x4 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self([
            self.0[0].wrapping_add(rhs.0[0]),
            self.0[1].wrapping_add(rhs.0[1]),
            self.0[2].wrapping_add(rhs.0[2]),
            self.0[3].wrapping_add(rhs.0[3]),
        ])
    }
}

impl Sub for I32x4 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self([
            self.0[0].wrapping_sub(rhs.0[0]),
            self.0[1].wrapping_sub(rhs.0[1]),
            self.0[2].wrapping_sub(rhs.0[2]),
            self.0[3].wrapping_sub(rhs.0[3]),
        ])
    }
}

impl Index<usize> for I32x4 {
    type Output = i32;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl IndexMut<usize> for I32x4 {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

// ============================================================================
// I32x8 Implementation
// ============================================================================

impl I32x8 {
    /// Create a new vector with all lanes set to zero.
    #[inline]
    #[must_use]
    pub const fn zero() -> Self {
        Self([0; 8])
    }

    /// Create a new vector with all lanes set to the same value.
    #[inline]
    #[must_use]
    pub const fn splat(value: i32) -> Self {
        Self([value; 8])
    }

    /// Create a vector from an array.
    #[inline]
    #[must_use]
    pub const fn from_array(arr: [i32; 8]) -> Self {
        Self(arr)
    }

    /// Convert to an array.
    #[inline]
    #[must_use]
    pub const fn to_array(self) -> [i32; 8] {
        self.0
    }

    /// Horizontal sum of all elements.
    #[inline]
    #[must_use]
    pub fn horizontal_sum(self) -> i32 {
        self.0.iter().fold(0i32, |acc, &x| acc.wrapping_add(x))
    }
}

// ============================================================================
// U8x16 Implementation
// ============================================================================

impl U8x16 {
    /// Create a new vector with all lanes set to zero.
    #[inline]
    #[must_use]
    pub const fn zero() -> Self {
        Self([0; 16])
    }

    /// Create a new vector with all lanes set to the same value.
    #[inline]
    #[must_use]
    pub const fn splat(value: u8) -> Self {
        Self([value; 16])
    }

    /// Create a vector from an array.
    #[inline]
    #[must_use]
    pub const fn from_array(arr: [u8; 16]) -> Self {
        Self(arr)
    }

    /// Convert to an array.
    #[inline]
    #[must_use]
    pub const fn to_array(self) -> [u8; 16] {
        self.0
    }

    /// Get element at index.
    #[inline]
    #[must_use]
    pub fn get(&self, index: usize) -> u8 {
        self.0[index]
    }

    /// Set element at index.
    #[inline]
    pub fn set(&mut self, index: usize, value: u8) {
        self.0[index] = value;
    }

    /// Widen low 8 bytes to I16x8.
    #[inline]
    #[must_use]
    pub fn widen_low_i16(self) -> I16x8 {
        I16x8([
            i16::from(self.0[0]),
            i16::from(self.0[1]),
            i16::from(self.0[2]),
            i16::from(self.0[3]),
            i16::from(self.0[4]),
            i16::from(self.0[5]),
            i16::from(self.0[6]),
            i16::from(self.0[7]),
        ])
    }

    /// Widen high 8 bytes to I16x8.
    #[inline]
    #[must_use]
    pub fn widen_high_i16(self) -> I16x8 {
        I16x8([
            i16::from(self.0[8]),
            i16::from(self.0[9]),
            i16::from(self.0[10]),
            i16::from(self.0[11]),
            i16::from(self.0[12]),
            i16::from(self.0[13]),
            i16::from(self.0[14]),
            i16::from(self.0[15]),
        ])
    }

    /// Get a pointer to the underlying array.
    #[inline]
    #[must_use]
    pub const fn as_ptr(&self) -> *const u8 {
        self.0.as_ptr()
    }

    /// Get a mutable pointer to the underlying array.
    #[inline]
    #[must_use]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.0.as_mut_ptr()
    }

    /// Get an iterator over the elements.
    #[inline]
    pub fn iter(&self) -> std::slice::Iter<'_, u8> {
        self.0.iter()
    }

    /// Copy elements from a slice.
    #[inline]
    pub fn copy_from_slice(&mut self, src: &[u8]) {
        self.0.copy_from_slice(src);
    }
}

impl Index<usize> for U8x16 {
    type Output = u8;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl IndexMut<usize> for U8x16 {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

// ============================================================================
// U8x32 Implementation
// ============================================================================

impl U8x32 {
    /// Create a new vector with all lanes set to zero.
    #[inline]
    #[must_use]
    pub const fn zero() -> Self {
        Self([0; 32])
    }

    /// Create a new vector with all lanes set to the same value.
    #[inline]
    #[must_use]
    pub const fn splat(value: u8) -> Self {
        Self([value; 32])
    }

    /// Create a vector from an array.
    #[inline]
    #[must_use]
    pub const fn from_array(arr: [u8; 32]) -> Self {
        Self(arr)
    }

    /// Convert to an array.
    #[inline]
    #[must_use]
    pub const fn to_array(self) -> [u8; 32] {
        self.0
    }

    /// Split into two U8x16 vectors.
    #[inline]
    #[must_use]
    pub fn split(self) -> (U8x16, U8x16) {
        let mut low = [0u8; 16];
        let mut high = [0u8; 16];
        low.copy_from_slice(&self.0[0..16]);
        high.copy_from_slice(&self.0[16..32]);
        (U8x16(low), U8x16(high))
    }

    /// Get a pointer to the underlying array.
    #[inline]
    #[must_use]
    pub const fn as_ptr(&self) -> *const u8 {
        self.0.as_ptr()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_i16x8_basic() {
        let a = I16x8::splat(10);
        let b = I16x8::splat(5);
        let sum = a + b;
        assert_eq!(sum.0, [15; 8]);

        let diff = a - b;
        assert_eq!(diff.0, [5; 8]);
    }

    #[test]
    fn test_i16x8_widen() {
        let v = I16x8::from_array([1, 2, 3, 4, 5, 6, 7, 8]);
        let low = v.widen_low();
        let high = v.widen_high();
        assert_eq!(low.0, [1, 2, 3, 4]);
        assert_eq!(high.0, [5, 6, 7, 8]);
    }

    #[test]
    fn test_i32x4_horizontal_sum() {
        let v = I32x4::from_array([1, 2, 3, 4]);
        assert_eq!(v.horizontal_sum(), 10);
    }

    #[test]
    fn test_i32x4_narrow_sat() {
        let low = I32x4::from_array([100, -100, 32767, -32768]);
        let high = I32x4::from_array([40000, -40000, 0, 1]);
        let result = low.narrow_sat(high);
        assert_eq!(result.0, [100, -100, 32767, -32768, 32767, -32768, 0, 1]);
    }

    #[test]
    fn test_u8x16_widen() {
        let v = U8x16::from_array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
        let low = v.widen_low_i16();
        let high = v.widen_high_i16();
        assert_eq!(low.0, [0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(high.0, [8, 9, 10, 11, 12, 13, 14, 15]);
    }

    #[test]
    fn test_u8x32_split() {
        let mut arr = [0u8; 32];
        for (i, elem) in arr.iter_mut().enumerate() {
            *elem = i as u8;
        }
        let v = U8x32::from_array(arr);
        let (low, high) = v.split();
        assert_eq!(low.0[0], 0);
        assert_eq!(low.0[15], 15);
        assert_eq!(high.0[0], 16);
        assert_eq!(high.0[15], 31);
    }
}
