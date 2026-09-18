//! Filter operations for video codec implementations.
//!
//! This module provides filtering primitives used in:
//! - Scaling (horizontal and vertical resampling)
//! - Loop filtering (deblocking)
//! - In-loop restoration filters
//!
//! All operations are designed to map efficiently to SIMD instructions.

#![forbid(unsafe_code)]
// Allow truncation and sign loss casts for filter operations (values are clamped)
#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
// Allow loop indexing for filter operations
#![allow(clippy::needless_range_loop)]

use super::scalar::ScalarFallback;
use super::traits::{SimdOps, SimdOpsExt};
use super::types::I16x8;

/// Filter operations using SIMD.
pub struct FilterOps<S: SimdOps> {
    simd: S,
}

impl<S: SimdOps + Default> Default for FilterOps<S> {
    fn default() -> Self {
        Self::new(S::default())
    }
}

impl<S: SimdOps> FilterOps<S> {
    /// Create a new filter operations instance.
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

    /// Apply a horizontal 2-tap filter (bilinear).
    ///
    /// Used for simple 2x scaling or half-pixel interpolation.
    #[allow(dead_code)]
    pub fn filter_h_2tap(&self, src: &[u8], dst: &mut [u8], width: usize) {
        if src.len() < width + 1 || dst.len() < width {
            return;
        }

        for x in 0..width {
            // Simple average of two adjacent pixels
            let a = u16::from(src[x]);
            let b = u16::from(src[x + 1]);
            dst[x] = ((a + b + 1) >> 1) as u8;
        }
    }

    /// Apply a horizontal 4-tap filter.
    ///
    /// Common filter coefficients for sub-pixel interpolation.
    #[allow(dead_code)]
    pub fn filter_h_4tap(&self, src: &[u8], dst: &mut [u8], coeffs: &[i16; 4], width: usize) {
        if src.len() < width + 3 || dst.len() < width {
            return;
        }

        for x in 0..width {
            let mut sum = 0i32;
            for k in 0..4 {
                sum += i32::from(src[x + k]) * i32::from(coeffs[k]);
            }
            // Round and clip
            let result = (sum + 64) >> 7;
            dst[x] = result.clamp(0, 255) as u8;
        }
    }

    /// Apply a horizontal 6-tap filter.
    #[allow(dead_code)]
    pub fn filter_h_6tap(&self, src: &[u8], dst: &mut [u8], coeffs: &[i16; 6], width: usize) {
        if src.len() < width + 5 || dst.len() < width {
            return;
        }

        for x in 0..width {
            let mut sum = 0i32;
            for k in 0..6 {
                sum += i32::from(src[x + k]) * i32::from(coeffs[k]);
            }
            let result = (sum + 64) >> 7;
            dst[x] = result.clamp(0, 255) as u8;
        }
    }

    /// Apply a horizontal 8-tap filter.
    ///
    /// Used in AV1 for high-quality scaling.
    #[allow(dead_code)]
    pub fn filter_h_8tap(&self, src: &[u8], dst: &mut [u8], coeffs: &[i16; 8], width: usize) {
        if src.len() < width + 7 || dst.len() < width {
            return;
        }

        for x in 0..width {
            let mut sum = 0i32;
            for k in 0..8 {
                sum += i32::from(src[x + k]) * i32::from(coeffs[k]);
            }
            let result = (sum + 64) >> 7;
            dst[x] = result.clamp(0, 255) as u8;
        }
    }

    /// Apply a vertical filter to a column of pixels.
    ///
    /// Takes pointers to multiple rows and produces one output pixel.
    #[allow(dead_code)]
    pub fn filter_v_8tap(&self, rows: &[&[u8]; 8], col: usize, coeffs: &[i16; 8]) -> u8 {
        let mut sum = 0i32;
        for k in 0..8 {
            if col < rows[k].len() {
                sum += i32::from(rows[k][col]) * i32::from(coeffs[k]);
            }
        }
        let result = (sum + 64) >> 7;
        result.clamp(0, 255) as u8
    }

    /// Apply vertical filter to a row of pixels.
    #[allow(dead_code)]
    pub fn filter_v_row_8tap(
        &self,
        rows: &[&[u8]; 8],
        dst: &mut [u8],
        coeffs: &[i16; 8],
        width: usize,
    ) {
        let width = width.min(dst.len());
        for x in 0..width {
            dst[x] = self.filter_v_8tap(rows, x, coeffs);
        }
    }
}

impl<S: SimdOps + SimdOpsExt> FilterOps<S> {
    /// SIMD-accelerated horizontal 8-tap filter.
    #[allow(dead_code)]
    pub fn filter_h_8tap_simd(&self, src: &[u8], dst: &mut [u8], coeffs: &[i16; 8], width: usize) {
        if src.len() < width + 7 || dst.len() < width {
            return;
        }

        let coeff_vec = I16x8::from_array(*coeffs);
        let mut x = 0;

        // Process 8 pixels at a time
        while x + 8 <= width {
            let mut results = [0i16; 8];

            for i in 0..8 {
                let src_slice = &src[x + i..];
                let samples = self.simd.load8_u8_to_i16x8(src_slice);
                let prod = self.simd.pmaddwd(samples, coeff_vec);
                let sum = self.simd.horizontal_sum_i32x4(prod);
                results[i] = ((sum + 64) >> 7).clamp(0, 255) as i16;
            }

            let result_vec = I16x8::from_array(results);
            self.simd.store8_i16x8_as_u8(result_vec, &mut dst[x..]);
            x += 8;
        }

        // Handle remaining pixels
        while x < width {
            let mut sum = 0i32;
            for k in 0..8 {
                sum += i32::from(src[x + k]) * i32::from(coeffs[k]);
            }
            dst[x] = ((sum + 64) >> 7).clamp(0, 255) as u8;
            x += 1;
        }
    }
}

// ============================================================================
// Loop Filter Operations
// ============================================================================

/// Deblocking filter strength parameters.
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
pub struct LoopFilterParams {
    /// Filter level (0-63).
    pub level: u8,
    /// Sharp threshold.
    pub sharpness: u8,
    /// Block edge strength (0-3).
    pub edge_strength: u8,
}

impl Default for LoopFilterParams {
    fn default() -> Self {
        Self {
            level: 32,
            sharpness: 0,
            edge_strength: 0,
        }
    }
}

/// Calculate loop filter thresholds from parameters.
#[allow(dead_code)]
#[must_use]
pub fn calculate_thresholds(params: &LoopFilterParams) -> (u8, u8, u8) {
    let level = params.level;
    let sharpness = params.sharpness;

    // E (edge) threshold
    let e = if level == 0 {
        0
    } else {
        (u16::from(level) * 2 + 1).min(255) as u8
    };

    // I (interior) threshold
    let i = if sharpness == 0 {
        level
    } else if sharpness <= 4 {
        level.saturating_sub(sharpness * 2)
    } else {
        level.saturating_sub(8)
    };

    // Hev (high edge variance) threshold
    let hev = if level <= 15 {
        0
    } else if level <= 40 {
        1
    } else {
        2
    };

    (e, i, hev)
}

/// Simple 4-tap deblocking filter.
///
/// Applies filtering to reduce blocking artifacts at block boundaries.
#[allow(dead_code)]
pub fn loop_filter_4(
    p1: &mut u8,
    p0: &mut u8,
    q0: &mut u8,
    q1: &mut u8,
    e_threshold: u8,
    i_threshold: u8,
) {
    // Check if filtering should be applied
    let p1_val = i16::from(*p1);
    let p0_val = i16::from(*p0);
    let q0_val = i16::from(*q0);
    let q1_val = i16::from(*q1);

    let edge = (p0_val - q0_val).abs();
    if edge > i16::from(e_threshold) {
        return;
    }

    let interior = (p1_val - p0_val).abs().max((q1_val - q0_val).abs());
    if interior > i16::from(i_threshold) {
        return;
    }

    // Apply simple filter
    let delta = ((q0_val - p0_val) * 4 + (p1_val - q1_val) + 4) >> 3;
    let delta = delta.clamp(-128, 127);

    *p0 = (p0_val + delta).clamp(0, 255) as u8;
    *q0 = (q0_val - delta).clamp(0, 255) as u8;
}

/// Strong 8-tap deblocking filter.
///
/// Applied at strong edges with flat regions.
#[allow(dead_code, clippy::too_many_arguments)]
pub fn loop_filter_8(
    p3: &mut u8,
    p2: &mut u8,
    p1: &mut u8,
    p0: &mut u8,
    q0: &mut u8,
    q1: &mut u8,
    q2: &mut u8,
    q3: &mut u8,
    threshold: u8,
) {
    let p = [*p3, *p2, *p1, *p0];
    let q = [*q0, *q1, *q2, *q3];

    // Check flatness
    let is_flat = (0..4).all(|i| {
        let diff_p = (i16::from(p[i]) - i16::from(p[3])).abs();
        let diff_q = (i16::from(q[i]) - i16::from(q[0])).abs();
        diff_p <= i16::from(threshold) && diff_q <= i16::from(threshold)
    });

    if !is_flat {
        // Fall back to simple filter
        loop_filter_4(p1, p0, q0, q1, threshold, threshold);
        return;
    }

    // Strong filtering: average all 8 pixels
    let sum: i32 = p.iter().chain(q.iter()).map(|&v| i32::from(v)).sum();
    let avg = ((sum + 4) >> 3).clamp(0, 255) as u8;

    // Blend toward average
    *p0 = blend_to_avg(*p0, avg);
    *q0 = blend_to_avg(*q0, avg);
    *p1 = blend_to_avg(*p1, avg);
    *q1 = blend_to_avg(*q1, avg);
    *p2 = blend_to_avg(*p2, avg);
    *q2 = blend_to_avg(*q2, avg);
    *p3 = blend_to_avg(*p3, avg);
    *q3 = blend_to_avg(*q3, avg);
}

/// Blend a value toward an average.
#[inline]
#[allow(clippy::cast_possible_truncation)]
fn blend_to_avg(val: u8, avg: u8) -> u8 {
    // 50% blend - result is always in range [0, 255] since both inputs are u8
    ((u16::from(val) + u16::from(avg) + 1) >> 1) as u8
}

// ============================================================================
// Standard Filter Coefficients
// ============================================================================

/// Bilinear interpolation coefficients (2-tap).
#[allow(dead_code)]
pub const BILINEAR_COEFFS: [[i16; 2]; 8] = [
    [128, 0],  // 0/8 = 0
    [112, 16], // 1/8
    [96, 32],  // 2/8
    [80, 48],  // 3/8
    [64, 64],  // 4/8 = 0.5
    [48, 80],  // 5/8
    [32, 96],  // 6/8
    [16, 112], // 7/8
];

/// 6-tap sub-pixel interpolation coefficients.
#[allow(dead_code)]
pub const SUBPEL_6TAP_COEFFS: [[i16; 6]; 8] = [
    [0, 0, 128, 0, 0, 0],     // 0/8
    [1, -5, 126, 8, -2, 0],   // 1/8
    [1, -11, 114, 28, -7, 3], // 2/8
    [2, -14, 98, 48, -12, 6], // 3/8
    [2, -16, 78, 78, -16, 2], // 4/8 (symmetric)
    [6, -12, 48, 98, -14, 2], // 5/8
    [3, -7, 28, 114, -11, 1], // 6/8
    [0, -2, 8, 126, -5, 1],   // 7/8
];

/// 8-tap high-quality interpolation coefficients (AV1 regular filter).
#[allow(dead_code)]
pub const SUBPEL_8TAP_REGULAR: [[i16; 8]; 16] = [
    [0, 0, 0, 128, 0, 0, 0, 0],
    [0, 2, -6, 126, 8, -2, 0, 0],
    [0, 2, -10, 122, 18, -4, 0, 0],
    [0, 2, -12, 116, 28, -8, 2, 0],
    [0, 2, -14, 110, 38, -10, 2, 0],
    [0, 2, -14, 102, 48, -12, 2, 0],
    [0, 2, -16, 94, 58, -12, 2, 0],
    [0, 2, -14, 84, 66, -12, 2, 0],
    [0, 2, -14, 76, 76, -14, 2, 0], // symmetric
    [0, 2, -12, 66, 84, -14, 2, 0],
    [0, 2, -12, 58, 94, -16, 2, 0],
    [0, 2, -12, 48, 102, -14, 2, 0],
    [0, 2, -10, 38, 110, -14, 2, 0],
    [0, 2, -8, 28, 116, -12, 2, 0],
    [0, 0, -4, 18, 122, -10, 2, 0],
    [0, 0, -2, 8, 126, -6, 2, 0],
];

/// Create a filter operations instance with scalar fallback.
#[inline]
#[must_use]
pub fn filter_ops() -> FilterOps<ScalarFallback> {
    FilterOps::new(ScalarFallback::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_h_2tap() {
        let ops = filter_ops();

        let src = [100u8, 200, 100, 200, 100, 200, 100, 200];
        let mut dst = [0u8; 7];

        ops.filter_h_2tap(&src, &mut dst, 7);

        // Each output should be average of adjacent pixels
        for (i, &v) in dst.iter().enumerate() {
            let expected = ((u16::from(src[i]) + u16::from(src[i + 1]) + 1) >> 1) as u8;
            assert_eq!(v, expected);
        }
    }

    #[test]
    fn test_filter_h_4tap() {
        let ops = filter_ops();

        // Simple averaging filter
        let coeffs = [32i16, 32, 32, 32];
        let src = [100u8; 16];
        let mut dst = [0u8; 12];

        ops.filter_h_4tap(&src, &mut dst, &coeffs, 12);

        // Constant input should produce constant output
        for &v in &dst {
            assert!(v >= 99 && v <= 101);
        }
    }

    #[test]
    fn test_filter_h_8tap() {
        let ops = filter_ops();

        // Use identity-like filter (all weight on center)
        let coeffs = [0i16, 0, 0, 128, 0, 0, 0, 0];
        let src = [50u8, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160];
        let mut dst = [0u8; 4];

        ops.filter_h_8tap(&src, &mut dst, &coeffs, 4);

        // Output should match input offset by 3 (filter center)
        assert_eq!(dst[0], 80);
        assert_eq!(dst[1], 90);
        assert_eq!(dst[2], 100);
        assert_eq!(dst[3], 110);
    }

    #[test]
    fn test_loop_filter_4() {
        let mut p1 = 100u8;
        let mut p0 = 110u8;
        let mut q0 = 150u8;
        let mut q1 = 160u8;

        loop_filter_4(&mut p1, &mut p0, &mut q0, &mut q1, 50, 30);

        // Filter should reduce the p0-q0 difference
        let diff_after = (i16::from(p0) - i16::from(q0)).abs();
        assert!(diff_after < 40);
    }

    #[test]
    fn test_loop_filter_4_no_filter() {
        let mut p1 = 100u8;
        let mut p0 = 110u8;
        let mut q0 = 150u8;
        let mut q1 = 160u8;

        // Very low threshold should prevent filtering
        loop_filter_4(&mut p1, &mut p0, &mut q0, &mut q1, 5, 5);

        // Values should be unchanged
        assert_eq!(p0, 110);
        assert_eq!(q0, 150);
    }

    #[test]
    fn test_calculate_thresholds() {
        let params = LoopFilterParams {
            level: 32,
            sharpness: 0,
            edge_strength: 0,
        };

        let (e, i, hev) = calculate_thresholds(&params);

        assert!(e > 0);
        assert_eq!(i, 32); // Same as level when sharpness is 0
        assert_eq!(hev, 1); // Level 32 is in middle range
    }

    #[test]
    fn test_calculate_thresholds_zero_level() {
        let params = LoopFilterParams {
            level: 0,
            sharpness: 0,
            edge_strength: 0,
        };

        let (e, i, hev) = calculate_thresholds(&params);

        assert_eq!(e, 0);
        assert_eq!(i, 0);
        assert_eq!(hev, 0);
    }

    #[test]
    fn test_bilinear_coeffs_sum() {
        // Each pair should sum to 128
        for coeffs in BILINEAR_COEFFS {
            assert_eq!(coeffs[0] + coeffs[1], 128);
        }
    }

    #[test]
    fn test_subpel_coeffs_sum() {
        // 6-tap coefficients should sum to 128
        for coeffs in SUBPEL_6TAP_COEFFS {
            let sum: i16 = coeffs.iter().sum();
            assert_eq!(sum, 128, "Sum mismatch: {}", sum);
        }

        // 8-tap coefficients should sum to 128
        for coeffs in SUBPEL_8TAP_REGULAR {
            let sum: i16 = coeffs.iter().sum();
            assert_eq!(sum, 128, "Sum mismatch: {}", sum);
        }
    }

    #[test]
    fn test_loop_filter_8_flat() {
        // Create a flat region that should be smoothed
        let mut p3 = 100u8;
        let mut p2 = 101u8;
        let mut p1 = 102u8;
        let mut p0 = 103u8;
        let mut q0 = 104u8;
        let mut q1 = 105u8;
        let mut q2 = 106u8;
        let mut q3 = 107u8;

        loop_filter_8(
            &mut p3, &mut p2, &mut p1, &mut p0, &mut q0, &mut q1, &mut q2, &mut q3, 10,
        );

        // After filtering, values should be closer to average
        let avg = (100 + 101 + 102 + 103 + 104 + 105 + 106 + 107) / 8;
        assert!((i16::from(p0) - avg as i16).abs() < 5);
    }

    #[test]
    fn test_filter_v_8tap() {
        let ops = filter_ops();

        // Create 8 rows of constant value
        let row = [128u8; 16];
        let rows: [&[u8]; 8] = [&row, &row, &row, &row, &row, &row, &row, &row];

        // Identity filter centered on position 3
        let coeffs = [0i16, 0, 0, 128, 0, 0, 0, 0];

        let result = ops.filter_v_8tap(&rows, 0, &coeffs);
        assert_eq!(result, 128);
    }
}
