//! Fixed-point arithmetic helpers for SIMD-friendly computations.
//!
//! Provides 16-bit and 8-bit fixed-point types used to avoid floating-point
//! operations in inner loops where integer SIMD lanes are preferred.

#![allow(dead_code)]

/// A 16-bit fixed-point number with 8 fractional bits (Q8.8 format).
///
/// The value is stored as `raw / 256.0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Fixed16 {
    raw: i16,
}

impl Fixed16 {
    /// The fixed-point scale factor (2^8 = 256).
    const SCALE: i32 = 256;

    /// Create a `Fixed16` from a 32-bit float.
    ///
    /// Values outside the representable range are saturated.
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn from_f32(v: f32) -> Self {
        let raw = (v * Self::SCALE as f32).round().clamp(-32768.0, 32767.0) as i16;
        Self { raw }
    }

    /// Convert back to a 32-bit float.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn to_f32(self) -> f32 {
        f32::from(self.raw) / Self::SCALE as f32
    }

    /// Multiply two `Fixed16` values, returning a saturated result.
    #[allow(clippy::cast_possible_truncation)]
    #[must_use]
    pub fn mul(self, other: Self) -> Self {
        let result = (i32::from(self.raw) * i32::from(other.raw)) / Self::SCALE;
        let clamped = result.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
        Self { raw: clamped }
    }

    /// Add two `Fixed16` values, returning a saturating result.
    #[must_use]
    pub fn add(self, other: Self) -> Self {
        Self {
            raw: self.raw.saturating_add(other.raw),
        }
    }

    /// Return the raw integer representation.
    #[must_use]
    pub fn raw(self) -> i16 {
        self.raw
    }
}

/// An 8-bit fixed-point number (stored as a plain `u8` with an implicit
/// fractional denominator of 255).
///
/// Useful for weights and blend coefficients.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FixedPoint8 {
    raw: u8,
}

impl FixedPoint8 {
    /// Create a `FixedPoint8` directly from a byte value (0 = 0.0, 255 = 1.0).
    #[must_use]
    pub fn from_byte(b: u8) -> Self {
        Self { raw: b }
    }

    /// Linearly interpolate between two `u8` values using this weight.
    ///
    /// `weight = 0` returns `a`; `weight = 255` returns `b`.
    #[allow(clippy::cast_possible_truncation)]
    #[must_use]
    pub fn interpolate(self, a: u8, b: u8) -> u8 {
        let w = u16::from(self.raw);
        let result = (u16::from(a) * (255 - w) + u16::from(b) * w + 127) / 255;
        result as u8
    }

    /// Return the raw byte.
    #[must_use]
    pub fn raw(self) -> u8 {
        self.raw
    }
}

/// Collection of fixed-point math utilities.
pub struct FixedMath;

impl FixedMath {
    /// Linear interpolation between two `Fixed16` values.
    ///
    /// `t` should be in [0.0, 1.0].
    #[must_use]
    pub fn lerp_fixed(a: Fixed16, b: Fixed16, t: Fixed16) -> Fixed16 {
        // result = a + t * (b - a)
        let diff = b.add(Fixed16 {
            raw: a.raw.saturating_neg(),
        });
        let scaled = t.mul(diff);
        a.add(scaled)
    }

    /// Clamp a `Fixed16` value to the range [min, max].
    #[must_use]
    pub fn clamp_fixed(v: Fixed16, min: Fixed16, max: Fixed16) -> Fixed16 {
        if v.raw < min.raw {
            min
        } else if v.raw > max.raw {
            max
        } else {
            v
        }
    }

    /// Scale a `u8` pixel value by a `FixedPoint8` weight.
    #[allow(clippy::cast_possible_truncation)]
    #[must_use]
    pub fn scale_pixel(pixel: u8, weight: FixedPoint8) -> u8 {
        ((u16::from(pixel) * u16::from(weight.raw) + 127) / 255) as u8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed16_from_to_f32_identity() {
        let v = Fixed16::from_f32(1.5);
        let back = v.to_f32();
        assert!((back - 1.5).abs() < 0.01, "got {back}");
    }

    #[test]
    fn test_fixed16_from_f32_negative() {
        let v = Fixed16::from_f32(-0.5);
        assert!(v.to_f32() < 0.0);
    }

    #[test]
    fn test_fixed16_add() {
        let a = Fixed16::from_f32(0.25);
        let b = Fixed16::from_f32(0.75);
        let c = a.add(b);
        assert!((c.to_f32() - 1.0).abs() < 0.01, "got {}", c.to_f32());
    }

    #[test]
    fn test_fixed16_mul() {
        let a = Fixed16::from_f32(2.0);
        let b = Fixed16::from_f32(3.0);
        let c = a.mul(b);
        assert!((c.to_f32() - 6.0).abs() < 0.05, "got {}", c.to_f32());
    }

    #[test]
    fn test_fixed16_saturating_add() {
        let a = Fixed16 { raw: i16::MAX };
        let b = Fixed16::from_f32(1.0);
        let _ = a.add(b); // must not panic
    }

    #[test]
    fn test_fixed_point8_interpolate_zero_weight() {
        let w = FixedPoint8::from_byte(0);
        assert_eq!(w.interpolate(10, 200), 10);
    }

    #[test]
    fn test_fixed_point8_interpolate_full_weight() {
        let w = FixedPoint8::from_byte(255);
        assert_eq!(w.interpolate(10, 200), 200);
    }

    #[test]
    fn test_fixed_point8_interpolate_midpoint() {
        let w = FixedPoint8::from_byte(128);
        let result = w.interpolate(0, 200);
        assert!((i16::from(result) - 100).abs() <= 2, "got {result}");
    }

    #[test]
    fn test_fixed_math_lerp_endpoints() {
        let a = Fixed16::from_f32(0.0);
        let b = Fixed16::from_f32(10.0);
        let t0 = Fixed16::from_f32(0.0);
        let t1 = Fixed16::from_f32(1.0);
        let at_zero = FixedMath::lerp_fixed(a, b, t0);
        let at_one = FixedMath::lerp_fixed(a, b, t1);
        assert!(at_zero.to_f32().abs() < 0.1, "at_zero={}", at_zero.to_f32());
        assert!(
            (at_one.to_f32() - 10.0).abs() < 0.1,
            "at_one={}",
            at_one.to_f32()
        );
    }

    #[test]
    fn test_fixed_math_clamp_fixed() {
        let v = Fixed16::from_f32(5.0);
        let lo = Fixed16::from_f32(0.0);
        let hi = Fixed16::from_f32(3.0);
        let c = FixedMath::clamp_fixed(v, lo, hi);
        assert!((c.to_f32() - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_fixed_math_clamp_fixed_below() {
        let v = Fixed16::from_f32(-1.0);
        let lo = Fixed16::from_f32(0.0);
        let hi = Fixed16::from_f32(1.0);
        let c = FixedMath::clamp_fixed(v, lo, hi);
        assert!(c.to_f32() >= 0.0);
    }

    #[test]
    fn test_fixed_math_scale_pixel() {
        let w = FixedPoint8::from_byte(128);
        let result = FixedMath::scale_pixel(200, w);
        assert!((i16::from(result) - 100).abs() <= 2, "got {result}");
    }

    #[test]
    fn test_fixed16_raw() {
        let v = Fixed16::from_f32(1.0);
        assert_eq!(v.raw(), 256); // Q8.8: 1.0 * 256 = 256
    }
}
