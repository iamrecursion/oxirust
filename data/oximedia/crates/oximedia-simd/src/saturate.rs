#![allow(dead_code)]
//! Saturating arithmetic primitives for media pixel pipelines.
//!
//! Provides lane-parallel saturating add, subtract, multiply, and average
//! operations for `u8`, `i16`, and `u16` buffers. All operations clamp to
//! the representable range of the target type rather than wrapping around.
//! This is exactly the behaviour required for pixel blending, gain
//! adjustment, and DC-offset correction in video codecs.

// ---------------------------------------------------------------------------
// Scalar saturating ops — u8
// ---------------------------------------------------------------------------

/// Saturating add: `dst[i] = (a[i] + b[i]).clamp(0, 255)`.
///
/// # Panics
///
/// Panics if `a`, `b`, and `dst` do not have the same length.
pub fn saturating_add_u8(a: &[u8], b: &[u8], dst: &mut [u8]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), dst.len());
    for i in 0..a.len() {
        dst[i] = a[i].saturating_add(b[i]);
    }
}

/// Saturating subtract: `dst[i] = (a[i] - b[i]).clamp(0, 255)`.
///
/// # Panics
///
/// Panics if `a`, `b`, and `dst` do not have the same length.
pub fn saturating_sub_u8(a: &[u8], b: &[u8], dst: &mut [u8]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), dst.len());
    for i in 0..a.len() {
        dst[i] = a[i].saturating_sub(b[i]);
    }
}

/// Saturating average: `dst[i] = (a[i] + b[i] + 1) / 2`.
///
/// # Panics
///
/// Panics if `a`, `b`, and `dst` do not have the same length.
pub fn saturating_avg_u8(a: &[u8], b: &[u8], dst: &mut [u8]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), dst.len());
    for i in 0..a.len() {
        let sum = u16::from(a[i]) + u16::from(b[i]) + 1;
        dst[i] = (sum >> 1) as u8;
    }
}

// ---------------------------------------------------------------------------
// Scalar saturating ops — i16
// ---------------------------------------------------------------------------

/// Saturating add for `i16` lanes.
///
/// # Panics
///
/// Panics if `a`, `b`, and `dst` do not have the same length.
pub fn saturating_add_i16(a: &[i16], b: &[i16], dst: &mut [i16]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), dst.len());
    for i in 0..a.len() {
        dst[i] = a[i].saturating_add(b[i]);
    }
}

/// Saturating subtract for `i16` lanes.
///
/// # Panics
///
/// Panics if `a`, `b`, and `dst` do not have the same length.
pub fn saturating_sub_i16(a: &[i16], b: &[i16], dst: &mut [i16]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), dst.len());
    for i in 0..a.len() {
        dst[i] = a[i].saturating_sub(b[i]);
    }
}

// ---------------------------------------------------------------------------
// Scalar saturating ops — u16 (10-bit / 12-bit paths)
// ---------------------------------------------------------------------------

/// Saturating add for `u16` lanes.
///
/// # Panics
///
/// Panics if `a`, `b`, and `dst` do not have the same length.
pub fn saturating_add_u16(a: &[u16], b: &[u16], dst: &mut [u16]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), dst.len());
    for i in 0..a.len() {
        dst[i] = a[i].saturating_add(b[i]);
    }
}

/// Saturating subtract for `u16` lanes.
///
/// # Panics
///
/// Panics if `a`, `b`, and `dst` do not have the same length.
pub fn saturating_sub_u16(a: &[u16], b: &[u16], dst: &mut [u16]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), dst.len());
    for i in 0..a.len() {
        dst[i] = a[i].saturating_sub(b[i]);
    }
}

// ---------------------------------------------------------------------------
// Gain / multiply with saturation
// ---------------------------------------------------------------------------

/// Multiply every `u8` element by a fixed-point factor (Q8 — 256 == 1.0)
/// and saturate to `[0, 255]`.
///
/// # Panics
///
/// Panics if `src` and `dst` do not have the same length.
#[allow(clippy::cast_possible_truncation)]
pub fn saturating_mul_u8_q8(src: &[u8], factor: u16, dst: &mut [u8]) {
    assert_eq!(src.len(), dst.len());
    for i in 0..src.len() {
        let product = u32::from(src[i]) * u32::from(factor);
        let result = (product + 128) >> 8; // round
        dst[i] = result.min(255) as u8;
    }
}

/// Weighted blend of two `u8` slices: `dst[i] = (a[i]*wa + b[i]*wb + 128) >> 8`
/// where `wa` and `wb` are Q8 weights.
///
/// # Panics
///
/// Panics if `a`, `b`, and `dst` do not have the same length.
#[allow(clippy::cast_possible_truncation)]
pub fn saturating_blend_u8(a: &[u8], b: &[u8], wa: u16, wb: u16, dst: &mut [u8]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), dst.len());
    for i in 0..a.len() {
        let val = u32::from(a[i]) * u32::from(wa) + u32::from(b[i]) * u32::from(wb) + 128;
        dst[i] = (val >> 8).min(255) as u8;
    }
}

/// Absolute difference: `dst[i] = |a[i] - b[i]|`.
///
/// # Panics
///
/// Panics if `a`, `b`, and `dst` do not have the same length.
pub fn abs_diff_u8(a: &[u8], b: &[u8], dst: &mut [u8]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), dst.len());
    for i in 0..a.len() {
        dst[i] = a[i].abs_diff(b[i]);
    }
}

/// Sum of absolute differences across an entire slice pair.
///
/// # Panics
///
/// Panics if `a` and `b` do not have the same length.
#[must_use]
pub fn sad_u8(a: &[u8], b: &[u8]) -> u64 {
    assert_eq!(a.len(), b.len());
    let mut acc = 0u64;
    for i in 0..a.len() {
        acc += u64::from(a[i].abs_diff(b[i]));
    }
    acc
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_saturating_add_u8_basic() {
        let a = [200u8, 100, 0, 255];
        let b = [100u8, 100, 0, 1];
        let mut dst = [0u8; 4];
        saturating_add_u8(&a, &b, &mut dst);
        assert_eq!(dst, [255, 200, 0, 255]);
    }

    #[test]
    fn test_saturating_sub_u8_basic() {
        let a = [200u8, 50, 0, 255];
        let b = [100u8, 100, 10, 0];
        let mut dst = [0u8; 4];
        saturating_sub_u8(&a, &b, &mut dst);
        assert_eq!(dst, [100, 0, 0, 255]);
    }

    #[test]
    fn test_saturating_avg_u8() {
        let a = [0u8, 100, 200, 254];
        let b = [0u8, 100, 200, 255];
        let mut dst = [0u8; 4];
        saturating_avg_u8(&a, &b, &mut dst);
        assert_eq!(dst[0], 0);
        assert_eq!(dst[1], 100);
        assert_eq!(dst[2], 200);
        assert_eq!(dst[3], 255); // (254+255+1)/2 = 255
    }

    #[test]
    fn test_saturating_add_i16() {
        let a = [30_000i16, -30_000, 0];
        let b = [10_000i16, -10_000, 0];
        let mut dst = [0i16; 3];
        saturating_add_i16(&a, &b, &mut dst);
        assert_eq!(dst, [32_767, -32_768, 0]);
    }

    #[test]
    fn test_saturating_sub_i16() {
        let a = [-30_000i16, 30_000];
        let b = [10_000i16, -10_000];
        let mut dst = [0i16; 2];
        saturating_sub_i16(&a, &b, &mut dst);
        assert_eq!(dst, [-32_768, 32_767]);
    }

    #[test]
    fn test_saturating_add_u16() {
        let a = [60_000u16, 0];
        let b = [10_000u16, 0];
        let mut dst = [0u16; 2];
        saturating_add_u16(&a, &b, &mut dst);
        assert_eq!(dst, [65_535, 0]);
    }

    #[test]
    fn test_saturating_sub_u16() {
        let a = [100u16, 0];
        let b = [200u16, 0];
        let mut dst = [0u16; 2];
        saturating_sub_u16(&a, &b, &mut dst);
        assert_eq!(dst, [0, 0]);
    }

    #[test]
    fn test_saturating_mul_u8_q8_identity() {
        let src = [0u8, 128, 255];
        let mut dst = [0u8; 3];
        saturating_mul_u8_q8(&src, 256, &mut dst); // factor 1.0
        assert_eq!(dst, [0, 128, 255]);
    }

    #[test]
    fn test_saturating_mul_u8_q8_double() {
        let src = [50u8, 100, 200];
        let mut dst = [0u8; 3];
        saturating_mul_u8_q8(&src, 512, &mut dst); // factor 2.0
        assert_eq!(dst, [100, 200, 255]); // 200*2 = 400 => saturated 255
    }

    #[test]
    fn test_saturating_blend_u8() {
        // 50/50 blend
        let a = [0u8, 200, 100];
        let b = [100u8, 0, 100];
        let mut dst = [0u8; 3];
        saturating_blend_u8(&a, &b, 128, 128, &mut dst);
        // (0*128 + 100*128 + 128) >> 8 = 12928 >> 8 = 50
        assert_eq!(dst[0], 50);
    }

    #[test]
    fn test_abs_diff_u8() {
        let a = [100u8, 200, 50];
        let b = [150u8, 100, 50];
        let mut dst = [0u8; 3];
        abs_diff_u8(&a, &b, &mut dst);
        assert_eq!(dst, [50, 100, 0]);
    }

    #[test]
    fn test_sad_u8() {
        let a = [10u8, 20, 30, 40];
        let b = [15u8, 25, 20, 40];
        assert_eq!(sad_u8(&a, &b), (5 + 5 + 10));
    }

    #[test]
    fn test_saturating_add_u8_zeros() {
        let a = [0u8; 8];
        let b = [0u8; 8];
        let mut dst = [0u8; 8];
        saturating_add_u8(&a, &b, &mut dst);
        assert_eq!(dst, [0u8; 8]);
    }

    #[test]
    fn test_sad_u8_identical() {
        let a = [128u8; 64];
        assert_eq!(sad_u8(&a, &a), 0);
    }
}
