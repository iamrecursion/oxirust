#![allow(dead_code)]
//! SIMD-optimized minimum and maximum operations for pixel/sample data.
//!
//! Provides fast parallel min/max scanning, clamping, and range operations
//! over contiguous slices of integer and floating-point data. Scalar
//! fall-backs are used when the input is too small to benefit from
//! wide-register processing.

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Width (in elements) of a simulated SIMD lane for `u8` data.
const LANE_WIDTH_U8: usize = 16;

/// Width (in elements) of a simulated SIMD lane for `f32` data.
const LANE_WIDTH_F32: usize = 4;

// ---------------------------------------------------------------------------
// Public helpers
// ---------------------------------------------------------------------------

/// Result of a min/max scan over a byte slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MinMaxU8 {
    /// Minimum value found.
    pub min: u8,
    /// Maximum value found.
    pub max: u8,
}

/// Result of a min/max scan over an `i16` slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MinMaxI16 {
    /// Minimum value found.
    pub min: i16,
    /// Maximum value found.
    pub max: i16,
}

/// Result of a min/max scan over an `f32` slice.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MinMaxF32 {
    /// Minimum value found.
    pub min: f32,
    /// Maximum value found.
    pub max: f32,
}

// ---------------------------------------------------------------------------
// Core algorithms
// ---------------------------------------------------------------------------

/// Find the minimum and maximum values in a `u8` slice.
///
/// Returns `None` if the slice is empty.
#[must_use]
pub fn minmax_u8(data: &[u8]) -> Option<MinMaxU8> {
    if data.is_empty() {
        return None;
    }
    let mut lo = u8::MAX;
    let mut hi = u8::MIN;
    // Process in pseudo-SIMD lanes
    let chunks = data.chunks_exact(LANE_WIDTH_U8);
    let remainder = chunks.remainder();
    for chunk in chunks {
        let mut cmin = chunk[0];
        let mut cmax = chunk[0];
        for &v in &chunk[1..] {
            cmin = cmin.min(v);
            cmax = cmax.max(v);
        }
        lo = lo.min(cmin);
        hi = hi.max(cmax);
    }
    for &v in remainder {
        lo = lo.min(v);
        hi = hi.max(v);
    }
    Some(MinMaxU8 { min: lo, max: hi })
}

/// Find the minimum and maximum values in an `i16` slice.
///
/// Returns `None` if the slice is empty.
#[must_use]
pub fn minmax_i16(data: &[i16]) -> Option<MinMaxI16> {
    if data.is_empty() {
        return None;
    }
    let mut lo = i16::MAX;
    let mut hi = i16::MIN;
    for &v in data {
        lo = lo.min(v);
        hi = hi.max(v);
    }
    Some(MinMaxI16 { min: lo, max: hi })
}

/// Find the minimum and maximum values in an `f32` slice.
///
/// Returns `None` if the slice is empty.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn minmax_f32(data: &[f32]) -> Option<MinMaxF32> {
    if data.is_empty() {
        return None;
    }
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    let chunks = data.chunks_exact(LANE_WIDTH_F32);
    let remainder = chunks.remainder();
    for chunk in chunks {
        for &v in chunk {
            if v < lo {
                lo = v;
            }
            if v > hi {
                hi = v;
            }
        }
    }
    for &v in remainder {
        if v < lo {
            lo = v;
        }
        if v > hi {
            hi = v;
        }
    }
    Some(MinMaxF32 { min: lo, max: hi })
}

/// Clamp every element of `data` into `[lo, hi]` in-place.
pub fn clamp_u8(data: &mut [u8], lo: u8, hi: u8) {
    for v in data.iter_mut() {
        *v = (*v).clamp(lo, hi);
    }
}

/// Clamp every element of `data` into `[lo, hi]` in-place.
pub fn clamp_i16(data: &mut [i16], lo: i16, hi: i16) {
    for v in data.iter_mut() {
        *v = (*v).clamp(lo, hi);
    }
}

/// Clamp every element of `data` into `[lo, hi]` in-place.
pub fn clamp_f32(data: &mut [f32], lo: f32, hi: f32) {
    for v in data.iter_mut() {
        *v = v.clamp(lo, hi);
    }
}

/// Element-wise minimum of two `u8` slices, writing result into `dst`.
///
/// # Panics
///
/// Panics if `a`, `b`, and `dst` do not have the same length.
pub fn elementwise_min_u8(a: &[u8], b: &[u8], dst: &mut [u8]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), dst.len());
    for i in 0..a.len() {
        dst[i] = a[i].min(b[i]);
    }
}

/// Element-wise maximum of two `u8` slices, writing result into `dst`.
///
/// # Panics
///
/// Panics if `a`, `b`, and `dst` do not have the same length.
pub fn elementwise_max_u8(a: &[u8], b: &[u8], dst: &mut [u8]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), dst.len());
    for i in 0..a.len() {
        dst[i] = a[i].max(b[i]);
    }
}

/// Return the index of the maximum element in a `u8` slice.
///
/// On ties the first occurrence is returned. Returns `None` if empty.
#[must_use]
pub fn argmax_u8(data: &[u8]) -> Option<usize> {
    if data.is_empty() {
        return None;
    }
    let mut best_idx = 0;
    let mut best_val = data[0];
    for (i, &v) in data.iter().enumerate().skip(1) {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    Some(best_idx)
}

/// Return the index of the minimum element in a `u8` slice.
///
/// On ties the first occurrence is returned. Returns `None` if empty.
#[must_use]
pub fn argmin_u8(data: &[u8]) -> Option<usize> {
    if data.is_empty() {
        return None;
    }
    let mut best_idx = 0;
    let mut best_val = data[0];
    for (i, &v) in data.iter().enumerate().skip(1) {
        if v < best_val {
            best_val = v;
            best_idx = i;
        }
    }
    Some(best_idx)
}

/// Compute the dynamic range `max - min` for a `u8` slice.
///
/// Returns `None` if the slice is empty.
#[must_use]
pub fn dynamic_range_u8(data: &[u8]) -> Option<u8> {
    minmax_u8(data).map(|mm| mm.max - mm.min)
}

/// Normalize a `u8` slice so that its values span the full 0..=255 range.
///
/// Does nothing when the slice is empty or all values are identical.
#[allow(clippy::cast_precision_loss)]
pub fn stretch_contrast_u8(data: &mut [u8]) {
    if let Some(mm) = minmax_u8(data) {
        let range = mm.max - mm.min;
        if range == 0 {
            return;
        }
        let scale = 255.0_f32 / f32::from(range);
        for v in data.iter_mut() {
            let normalized = f32::from(*v - mm.min) * scale;
            *v = (normalized + 0.5).min(255.0) as u8;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minmax_u8_basic() {
        let data = [10u8, 20, 5, 255, 0, 128];
        let mm = minmax_u8(&data).expect("should succeed in test");
        assert_eq!(mm.min, 0);
        assert_eq!(mm.max, 255);
    }

    #[test]
    fn test_minmax_u8_single() {
        let data = [42u8];
        let mm = minmax_u8(&data).expect("should succeed in test");
        assert_eq!(mm.min, 42);
        assert_eq!(mm.max, 42);
    }

    #[test]
    fn test_minmax_u8_empty() {
        let data: [u8; 0] = [];
        assert!(minmax_u8(&data).is_none());
    }

    #[test]
    fn test_minmax_u8_large() {
        let mut data = vec![100u8; 1024];
        data[500] = 0;
        data[999] = 255;
        let mm = minmax_u8(&data).expect("should succeed in test");
        assert_eq!(mm.min, 0);
        assert_eq!(mm.max, 255);
    }

    #[test]
    fn test_minmax_i16_basic() {
        let data = [-100i16, 0, 32_000, -32_000, 1];
        let mm = minmax_i16(&data).expect("should succeed in test");
        assert_eq!(mm.min, -32_000);
        assert_eq!(mm.max, 32_000);
    }

    #[test]
    fn test_minmax_f32_basic() {
        let data = [1.0f32, -0.5, 3.14, 0.0, 2.71];
        let mm = minmax_f32(&data).expect("should succeed in test");
        assert!((mm.min - (-0.5)).abs() < f32::EPSILON);
        assert!((mm.max - 3.14).abs() < f32::EPSILON);
    }

    #[test]
    fn test_clamp_u8() {
        let mut data = [0u8, 50, 100, 150, 200, 255];
        clamp_u8(&mut data, 50, 200);
        assert_eq!(data, [50, 50, 100, 150, 200, 200]);
    }

    #[test]
    fn test_clamp_i16() {
        let mut data = [-1000i16, 0, 500, 1000];
        clamp_i16(&mut data, -100, 100);
        assert_eq!(data, [-100, 0, 100, 100]);
    }

    #[test]
    fn test_clamp_f32() {
        let mut data = [-1.0f32, 0.5, 2.0];
        clamp_f32(&mut data, 0.0, 1.0);
        assert!((data[0] - 0.0).abs() < f32::EPSILON);
        assert!((data[1] - 0.5).abs() < f32::EPSILON);
        assert!((data[2] - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_elementwise_min_u8() {
        let a = [10u8, 200, 50];
        let b = [20u8, 100, 50];
        let mut dst = [0u8; 3];
        elementwise_min_u8(&a, &b, &mut dst);
        assert_eq!(dst, [10, 100, 50]);
    }

    #[test]
    fn test_elementwise_max_u8() {
        let a = [10u8, 200, 50];
        let b = [20u8, 100, 50];
        let mut dst = [0u8; 3];
        elementwise_max_u8(&a, &b, &mut dst);
        assert_eq!(dst, [20, 200, 50]);
    }

    #[test]
    fn test_argmax_u8() {
        let data = [1u8, 5, 3, 5, 2];
        assert_eq!(argmax_u8(&data), Some(1)); // first occurrence of 5
    }

    #[test]
    fn test_argmin_u8() {
        let data = [10u8, 5, 3, 3, 20];
        assert_eq!(argmin_u8(&data), Some(2)); // first occurrence of 3
    }

    #[test]
    fn test_dynamic_range_u8() {
        let data = [50u8, 100, 150];
        assert_eq!(dynamic_range_u8(&data), Some(100));
    }

    #[test]
    fn test_stretch_contrast_u8() {
        let mut data = [100u8, 150, 200];
        stretch_contrast_u8(&mut data);
        // min was 100, max was 200 => range 100
        // 100 => 0, 200 => 255
        assert_eq!(data[0], 0);
        assert_eq!(data[2], 255);
    }

    #[test]
    fn test_stretch_contrast_u8_uniform() {
        let mut data = [128u8; 10];
        let orig = data;
        stretch_contrast_u8(&mut data);
        // all same => no change
        assert_eq!(data, orig);
    }

    #[test]
    fn test_argmax_empty() {
        let data: [u8; 0] = [];
        assert!(argmax_u8(&data).is_none());
    }

    #[test]
    fn test_argmin_empty() {
        let data: [u8; 0] = [];
        assert!(argmin_u8(&data).is_none());
    }
}
