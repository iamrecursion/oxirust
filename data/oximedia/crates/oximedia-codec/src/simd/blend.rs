//! Blending operations for video codec implementations.
//!
//! This module provides blending primitives used in:
//! - Motion compensation (bilinear interpolation)
//! - Frame blending
//! - Alpha compositing
//!
//! All operations are designed to map efficiently to SIMD instructions.

#![forbid(unsafe_code)]

use super::scalar::ScalarFallback;
use super::traits::{SimdOps, SimdOpsExt};
use super::types::{I16x8, U8x16};

/// Blending operations using SIMD.
pub struct BlendOps<S: SimdOps> {
    simd: S,
}

impl<S: SimdOps + Default> Default for BlendOps<S> {
    fn default() -> Self {
        Self::new(S::default())
    }
}

impl<S: SimdOps> BlendOps<S> {
    /// Create a new blending operations instance.
    #[inline]
    #[must_use]
    pub const fn new(simd: S) -> Self {
        Self { simd }
    }

    /// Get the underlying SIMD implementation.
    #[inline]
    #[must_use]
    pub const fn simd(&self) -> &S {
        &self.simd
    }

    /// Linear interpolation between two values.
    ///
    /// Returns: a + (b - a) * weight / 256
    ///
    /// Weight is in range [0, 256] where:
    /// - 0 = 100% a
    /// - 256 = 100% b
    /// - 128 = 50% each
    #[inline]
    #[allow(clippy::cast_sign_loss)]
    pub fn lerp_u8(&self, a: u8, b: u8, weight: u8) -> u8 {
        let a32 = i32::from(a);
        let b32 = i32::from(b);
        let w32 = i32::from(weight);
        let result = a32 + ((b32 - a32) * w32 + 128) / 256;
        // Safe: clamping to [0, 255] ensures the value fits in u8
        result.clamp(0, 255) as u8
    }

    /// Linear interpolation for i16x8 vectors.
    ///
    /// Returns: a + (b - a) * weight / 256
    #[inline]
    pub fn lerp_i16x8(&self, a: I16x8, b: I16x8, weight: i16) -> I16x8 {
        let diff = self.simd.sub_i16x8(b, a);
        let weight_vec = I16x8::splat(weight);
        let scaled = self.simd.mul_i16x8(diff, weight_vec);
        let shifted = self.simd.shr_i16x8(scaled, 8);
        self.simd.add_i16x8(a, shifted)
    }

    /// Weighted average of two u8x16 vectors.
    ///
    /// Returns: (a * (256 - weight) + b * weight + 128) / 256
    #[inline]
    #[allow(clippy::needless_range_loop, clippy::cast_possible_truncation)]
    pub fn weighted_avg_u8x16(&self, a: U8x16, b: U8x16, weight: u8) -> U8x16 {
        let mut result = [0u8; 16];
        let w = u16::from(weight);
        let inv_w = 256 - w;

        for i in 0..16 {
            // Result is always in [0, 255] due to the weighted average
            let val = (u16::from(a.0[i]) * inv_w + u16::from(b.0[i]) * w + 128) / 256;
            result[i] = val as u8;
        }

        U8x16(result)
    }

    /// Bilinear blend for motion compensation.
    ///
    /// Blends 4 samples using horizontal and vertical weights.
    /// Used for sub-pixel motion estimation.
    ///
    /// Layout:
    /// ```text
    /// tl --- tr
    /// |      |
    /// bl --- br
    /// ```
    ///
    /// Returns: blend of all 4 based on (hweight, vweight)
    #[inline]
    #[allow(dead_code)]
    pub fn bilinear_blend_u8(
        &self,
        tl: u8,
        tr: u8,
        bl: u8,
        br: u8,
        hweight: u8,
        vweight: u8,
    ) -> u8 {
        // Horizontal interpolation for top and bottom
        let top = self.lerp_u8(tl, tr, hweight);
        let bottom = self.lerp_u8(bl, br, hweight);

        // Vertical interpolation
        self.lerp_u8(top, bottom, vweight)
    }

    /// Bilinear blend for a row of 8 pixels.
    ///
    /// Takes 4 input rows and blends them bilinearly.
    #[inline]
    #[allow(dead_code, clippy::too_many_arguments)]
    pub fn bilinear_blend_row_8(
        &self,
        tl: &[u8],
        tr: &[u8],
        bl: &[u8],
        br: &[u8],
        hweight: u8,
        vweight: u8,
        dst: &mut [u8],
    ) {
        let len = 8
            .min(tl.len())
            .min(tr.len())
            .min(bl.len())
            .min(br.len())
            .min(dst.len());
        for i in 0..len {
            dst[i] = self.bilinear_blend_u8(tl[i], tr[i], bl[i], br[i], hweight, vweight);
        }
    }
}

impl<S: SimdOps + SimdOpsExt> BlendOps<S> {
    /// Bilinear blend using SIMD for a row of 8 pixels.
    #[allow(dead_code, clippy::similar_names, clippy::too_many_arguments)]
    pub fn bilinear_blend_row_8_simd(
        &self,
        tl: &[u8],
        tr: &[u8],
        bl: &[u8],
        br: &[u8],
        hweight: u8,
        vweight: u8,
        dst: &mut [u8],
    ) {
        // Load as i16 for computation
        let tl_v = self.simd.load8_u8_to_i16x8(tl);
        let tr_v = self.simd.load8_u8_to_i16x8(tr);
        let bl_v = self.simd.load8_u8_to_i16x8(bl);
        let br_v = self.simd.load8_u8_to_i16x8(br);

        // Horizontal blend
        let top = self.lerp_i16x8(tl_v, tr_v, i16::from(hweight));
        let bottom = self.lerp_i16x8(bl_v, br_v, i16::from(hweight));

        // Vertical blend
        let result = self.lerp_i16x8(top, bottom, i16::from(vweight));

        // Store result
        self.simd.store8_i16x8_as_u8(result, dst);
    }
}

/// Create a blending operations instance with scalar fallback.
#[inline]
#[must_use]
pub fn blend_ops() -> BlendOps<ScalarFallback> {
    BlendOps::new(ScalarFallback::new())
}

/// Half-pixel interpolation filter taps (6-tap filter).
///
/// Used for sub-pixel motion compensation in H.264/AV1.
#[allow(dead_code)]
pub const HALF_PEL_FILTER: [i16; 6] = [1, -5, 20, 20, -5, 1];

/// Quarter-pixel interpolation filter taps.
#[allow(dead_code)]
pub const QUARTER_PEL_FILTER: [i16; 6] = [1, -5, 52, 20, -5, 1];

/// Apply 6-tap horizontal filter for half-pixel interpolation.
#[allow(dead_code, clippy::cast_sign_loss)]
pub fn apply_half_pel_h(src: &[u8], dst: &mut [u8], width: usize) {
    if width < 6 || src.len() < width + 5 {
        return;
    }

    for x in 0..width {
        let mut sum: i32 = 0;
        for (k, &tap) in HALF_PEL_FILTER.iter().enumerate() {
            sum += i32::from(src[x + k]) * i32::from(tap);
        }
        // Round and clip - safe because we clamp to [0, 255]
        let result = (sum + 16) >> 5;
        dst[x] = result.clamp(0, 255) as u8;
    }
}

/// Apply 6-tap vertical filter for half-pixel interpolation.
#[allow(dead_code, clippy::cast_sign_loss)]
pub fn apply_half_pel_v(src: &[&[u8]], dst: &mut [u8], width: usize) {
    if src.len() < 6 {
        return;
    }

    for x in 0..width.min(dst.len()) {
        let mut sum: i32 = 0;
        for (k, &tap) in HALF_PEL_FILTER.iter().enumerate() {
            if x < src[k].len() {
                sum += i32::from(src[k][x]) * i32::from(tap);
            }
        }
        // Round and clip - safe because we clamp to [0, 255]
        let result = (sum + 16) >> 5;
        dst[x] = result.clamp(0, 255) as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lerp_u8() {
        let blend = blend_ops();

        // 0 weight = 100% a
        assert_eq!(blend.lerp_u8(100, 200, 0), 100);

        // 128 weight = 50% each (approximately)
        let mid = blend.lerp_u8(0, 200, 128);
        assert!(mid >= 99 && mid <= 101); // Allow rounding

        // Near full weight
        let high = blend.lerp_u8(0, 200, 255);
        assert!(high >= 198 && high <= 200);
    }

    #[test]
    fn test_weighted_avg_u8x16() {
        let blend = blend_ops();

        let a = U8x16::splat(100);
        let b = U8x16::splat(200);

        // 50% blend
        let result = blend.weighted_avg_u8x16(a, b, 128);
        for &v in &result.0 {
            assert!(v >= 149 && v <= 151);
        }

        // 0% = all a
        let result_a = blend.weighted_avg_u8x16(a, b, 0);
        assert_eq!(result_a.0, [100; 16]);

        // 100% = all b (weight = 256, but we use 255 max)
        let result_b = blend.weighted_avg_u8x16(a, b, 255);
        for &v in &result_b.0 {
            assert!(v >= 199 && v <= 200);
        }
    }

    #[test]
    fn test_bilinear_blend() {
        let blend = blend_ops();

        // All same values should return same value
        let result = blend.bilinear_blend_u8(100, 100, 100, 100, 128, 128);
        assert_eq!(result, 100);

        // Corner cases
        let tl_only = blend.bilinear_blend_u8(100, 0, 0, 0, 0, 0);
        assert_eq!(tl_only, 100);

        let tr_only = blend.bilinear_blend_u8(0, 100, 0, 0, 255, 0);
        assert!(tr_only >= 99);

        let bl_only = blend.bilinear_blend_u8(0, 0, 100, 0, 0, 255);
        assert!(bl_only >= 99);
    }

    #[test]
    fn test_lerp_i16x8() {
        let blend = blend_ops();

        let a = I16x8::from_array([0, 10, 20, 30, 40, 50, 60, 70]);
        let b = I16x8::from_array([100, 110, 120, 130, 140, 150, 160, 170]);

        // 50% blend (weight = 128)
        let result = blend.lerp_i16x8(a, b, 128);
        // Each should be approximately (a + b) / 2
        assert!(result.0[0] >= 49 && result.0[0] <= 51);
    }

    #[test]
    fn test_bilinear_row() {
        let blend = blend_ops();

        let tl = [100u8; 8];
        let tr = [100u8; 8];
        let bl = [100u8; 8];
        let br = [100u8; 8];
        let mut dst = [0u8; 8];

        blend.bilinear_blend_row_8(&tl, &tr, &bl, &br, 128, 128, &mut dst);

        for &v in &dst {
            assert_eq!(v, 100);
        }
    }

    #[test]
    fn test_half_pel_filter() {
        // Sum of filter taps should be 32 (for normalization by >>5)
        let sum: i16 = HALF_PEL_FILTER.iter().sum();
        assert_eq!(sum, 32);
    }

    #[test]
    fn test_apply_half_pel_h() {
        // Create a simple test pattern
        let src = [128u8; 16];
        let mut dst = [0u8; 10];

        apply_half_pel_h(&src, &mut dst, 10);

        // Constant input should produce constant output
        for &v in &dst {
            assert!(v >= 127 && v <= 129);
        }
    }
}
