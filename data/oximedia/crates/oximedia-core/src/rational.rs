//! Rational number arithmetic for frame rates and time bases.
//!
//! This module provides two rational number types:
//!
//! - [`Rational`] — lightweight `i32`-based type for frame-rate representation
//! - [`RationalTime`] — high-precision `i64`-based type for exact media timestamp
//!   arithmetic (GCD reduction, `from_fps`, `rescale`, `lcm`)
//!
//! For the workspace-level `i64` rational used in codec negotiation and
//! container metadata, see [`crate::types::Rational`].

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// A rational number with `i32` numerator and denominator.
///
/// Intended for frame-rate and time-base representation where `i32` range
/// is sufficient (e.g. 30000/1001, 24000/1001).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Rational {
    /// Numerator.
    pub num: i32,
    /// Denominator. Must not be zero.
    pub den: i32,
}

/// Computes the greatest common divisor of two non-negative integers.
fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

impl Rational {
    /// Creates a new `Rational`.
    ///
    /// # Panics
    ///
    /// Panics if `d` is zero.
    #[must_use]
    pub fn new(n: i32, d: i32) -> Self {
        assert!(d != 0, "Rational denominator must not be zero");
        Self { num: n, den: d }
    }

    /// Creates a `Rational` from a floating-point value using continued-fraction
    /// approximation with a maximum denominator of 1001.
    #[must_use]
    pub fn from_f64(f: f64) -> Self {
        const MAX_DEN: i32 = 1001;
        if f == 0.0 {
            return Self::new(0, 1);
        }
        let negative = f < 0.0;
        let f = f.abs();
        let mut best_num = 0i32;
        let mut best_den = 1i32;
        let mut best_err = f64::MAX;
        for den in 1..=MAX_DEN {
            #[allow(clippy::cast_possible_truncation)]
            let num = (f * f64::from(den)).round() as i32;
            let err = (f - f64::from(num) / f64::from(den)).abs();
            if err < best_err {
                best_err = err;
                best_num = num;
                best_den = den;
            }
            if err < 1e-9 {
                break;
            }
        }
        Self::new(if negative { -best_num } else { best_num }, best_den)
    }

    /// Converts this rational to a 64-bit float.
    #[must_use]
    pub fn to_f64(self) -> f64 {
        f64::from(self.num) / f64::from(self.den)
    }

    /// Returns a new `Rational` reduced to lowest terms.
    #[must_use]
    #[allow(clippy::cast_possible_wrap)]
    pub fn reduce(self) -> Self {
        let g = gcd(self.num.unsigned_abs(), self.den.unsigned_abs());
        if g == 0 {
            return self;
        }
        let g = g as i32;
        let sign = if self.den < 0 { -1 } else { 1 };
        Self {
            num: sign * self.num / g,
            den: sign * self.den / g,
        }
    }

    /// Adds two rationals and returns the (unreduced) result.
    #[must_use]
    pub fn add(&self, other: &Self) -> Self {
        Self {
            num: self.num * other.den + other.num * self.den,
            den: self.den * other.den,
        }
        .reduce()
    }

    /// Multiplies two rationals and returns the reduced result.
    #[must_use]
    pub fn multiply(&self, other: &Self) -> Self {
        Self {
            num: self.num * other.num,
            den: self.den * other.den,
        }
        .reduce()
    }

    /// Returns `true` if this rational equals zero.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.num == 0
    }

    /// Returns `true` if this rational equals one (after reduction).
    #[must_use]
    pub fn is_one(&self) -> bool {
        let r = self.reduce();
        r.num == r.den
    }

    // --- Common frame rate constructors ---

    /// 24 fps (cinema).
    #[must_use]
    pub fn fps_24() -> Self {
        Self::new(24, 1)
    }

    /// 25 fps (PAL).
    #[must_use]
    pub fn fps_25() -> Self {
        Self::new(25, 1)
    }

    /// 30 fps.
    #[must_use]
    pub fn fps_30() -> Self {
        Self::new(30, 1)
    }

    /// ~29.97 fps (NTSC, 30000/1001).
    #[must_use]
    pub fn fps_2997() -> Self {
        Self::new(30000, 1001)
    }

    /// 60 fps.
    #[must_use]
    pub fn fps_60() -> Self {
        Self::new(60, 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_basic() {
        let r = Rational::new(3, 4);
        assert_eq!(r.num, 3);
        assert_eq!(r.den, 4);
    }

    #[test]
    #[should_panic(expected = "denominator must not be zero")]
    fn test_new_zero_den_panics() {
        let _ = Rational::new(1, 0);
    }

    #[test]
    fn test_to_f64_half() {
        let r = Rational::new(1, 2);
        assert!((r.to_f64() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_reduce_fraction() {
        let r = Rational::new(6, 9).reduce();
        assert_eq!(r.num, 2);
        assert_eq!(r.den, 3);
    }

    #[test]
    fn test_reduce_already_reduced() {
        let r = Rational::new(3, 7).reduce();
        assert_eq!(r.num, 3);
        assert_eq!(r.den, 7);
    }

    #[test]
    fn test_add_fractions() {
        let a = Rational::new(1, 3);
        let b = Rational::new(1, 6);
        let sum = a.add(&b);
        assert_eq!(sum, Rational::new(1, 2));
    }

    #[test]
    fn test_multiply_fractions() {
        let a = Rational::new(2, 3);
        let b = Rational::new(3, 4);
        let prod = a.multiply(&b);
        assert_eq!(prod, Rational::new(1, 2));
    }

    #[test]
    fn test_is_zero_true() {
        assert!(Rational::new(0, 5).is_zero());
    }

    #[test]
    fn test_is_zero_false() {
        assert!(!Rational::new(1, 5).is_zero());
    }

    #[test]
    fn test_is_one_true() {
        assert!(Rational::new(4, 4).is_one());
    }

    #[test]
    fn test_is_one_false() {
        assert!(!Rational::new(1, 2).is_one());
    }

    #[test]
    fn test_fps_24() {
        let r = Rational::fps_24();
        assert_eq!(r.to_f64(), 24.0);
    }

    #[test]
    fn test_fps_25() {
        let r = Rational::fps_25();
        assert_eq!(r.to_f64(), 25.0);
    }

    #[test]
    fn test_fps_30() {
        let r = Rational::fps_30();
        assert_eq!(r.to_f64(), 30.0);
    }

    #[test]
    fn test_fps_2997() {
        let r = Rational::fps_2997();
        assert_eq!(r.num, 30000);
        assert_eq!(r.den, 1001);
        assert!((r.to_f64() - 29.970_029_97).abs() < 1e-6);
    }

    #[test]
    fn test_fps_60() {
        let r = Rational::fps_60();
        assert_eq!(r.to_f64(), 60.0);
    }

    #[test]
    fn test_from_f64_half() {
        let r = Rational::from_f64(0.5);
        assert_eq!(r.num, 1);
        assert_eq!(r.den, 2);
    }

    #[test]
    fn test_from_f64_zero() {
        let r = Rational::from_f64(0.0);
        assert!(r.is_zero());
    }
}

// ---------------------------------------------------------------------------
// i64-precision rational for exact media timestamp arithmetic
// ---------------------------------------------------------------------------

use std::cmp::Ordering;
use std::ops::{Add, Div, Mul, Sub};

/// Computes the greatest common divisor of two `i64` values using the
/// Euclidean algorithm.  Both inputs are treated by absolute value.
#[must_use]
pub fn gcd_i64(mut a: i64, mut b: i64) -> i64 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Computes the least common multiple of two `i64` values.
/// Returns 0 if either input is 0.
#[must_use]
pub fn lcm_i64(a: i64, b: i64) -> i64 {
    if a == 0 || b == 0 {
        return 0;
    }
    (a / gcd_i64(a, b)) * b
}

/// A rational number with `i64` numerator and denominator, designed for
/// exact media timestamp arithmetic.
///
/// Invariants maintained after construction:
/// - `den` is always strictly positive
/// - the fraction is always in lowest terms (GCD-reduced)
///
/// # Examples
///
/// ```
/// use oximedia_core::rational::RationalTime;
///
/// let r = RationalTime::new(30_000, 1001);
/// assert!((r.to_f64() - 29.97).abs() < 0.01);
///
/// let ntsc = RationalTime::from_fps(29.97);
/// assert_eq!(ntsc.num, 30_000);
/// assert_eq!(ntsc.den, 1001);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RationalTime {
    /// Numerator (sign-bearing).
    pub num: i64,
    /// Denominator (always positive, never zero).
    pub den: i64,
}

impl RationalTime {
    /// Creates a new `RationalTime` normalising sign and reducing by GCD.
    ///
    /// # Panics
    ///
    /// Panics if `den` is zero.
    #[must_use]
    pub fn new(num: i64, den: i64) -> Self {
        assert!(den != 0, "RationalTime denominator must not be zero");
        let sign = if den < 0 { -1i64 } else { 1i64 };
        let g = gcd_i64(num.abs(), den.abs());
        let g = if g == 0 { 1 } else { g };
        Self {
            num: sign * num / g,
            den: sign * den / g,
        }
    }

    /// Returns the rational `0/1`.
    #[must_use]
    pub const fn zero() -> Self {
        Self { num: 0, den: 1 }
    }

    /// Returns the rational `1/1`.
    #[must_use]
    pub const fn one() -> Self {
        Self { num: 1, den: 1 }
    }

    /// Converts a floating-point frame-rate to a `RationalTime`.
    ///
    /// Well-known drop-frame rates are mapped exactly:
    /// - 23.976 → 24000/1001
    /// - 29.97  → 30000/1001
    /// - 59.94  → 60000/1001
    ///
    /// All other values are approximated as `(fps * 1000) as i64 / 1000` and
    /// then GCD-reduced.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn from_fps(fps: f64) -> Self {
        // Exact mappings for common NTSC drop-frame rates.
        const EPSILON: f64 = 1e-4;
        if (fps - 23.976).abs() < EPSILON {
            return Self::new(24_000, 1001);
        }
        if (fps - 29.97).abs() < EPSILON {
            return Self::new(30_000, 1001);
        }
        if (fps - 59.94).abs() < EPSILON {
            return Self::new(60_000, 1001);
        }
        // General case: multiply by 1000 to capture 3 decimal places of
        // precision, store as num/1000 and reduce.
        let scaled = (fps * 1000.0).round() as i64;
        Self::new(scaled, 1000)
    }

    /// Converts this rational to a 64-bit float.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn to_f64(self) -> f64 {
        self.num as f64 / self.den as f64
    }

    /// Rescales a timestamp `self` to a different denominator.
    ///
    /// Given that `self` represents a tick count expressed as
    /// `self.num / self.den`, this converts the tick count to the
    /// equivalent count in `dst_den` ticks-per-second using rounding:
    ///
    /// ```text
    /// result = (self.num * dst_den + self.den / 2) / self.den
    /// ```
    ///
    /// This is the standard FFmpeg `av_rescale_rnd` rounding-half-up formula.
    #[must_use]
    pub fn rescale(self, dst_den: i64) -> i64 {
        let half = self.den / 2;
        (self.num * dst_den + half) / self.den
    }
}

impl Add for RationalTime {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        // a/b + c/d = (a*d + c*b) / (b*d)
        let l = lcm_i64(self.den, rhs.den);
        let num = self.num * (l / self.den) + rhs.num * (l / rhs.den);
        Self::new(num, l)
    }
}

impl Sub for RationalTime {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        let l = lcm_i64(self.den, rhs.den);
        let num = self.num * (l / self.den) - rhs.num * (l / rhs.den);
        Self::new(num, l)
    }
}

impl Mul for RationalTime {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self::new(self.num * rhs.num, self.den * rhs.den)
    }
}

impl Div for RationalTime {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        assert!(rhs.num != 0, "RationalTime: cannot divide by zero");
        Self::new(self.num * rhs.den, self.den * rhs.num)
    }
}

impl PartialOrd for RationalTime {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RationalTime {
    fn cmp(&self, other: &Self) -> Ordering {
        // a/b cmp c/d  →  a*d cmp c*b  (both dens positive after normalisation)
        (self.num * other.den).cmp(&(other.num * self.den))
    }
}

#[cfg(test)]
mod tests_rational_time {
    use super::*;

    #[test]
    fn test_new_reduces() {
        let r = RationalTime::new(6, 9);
        assert_eq!(r.num, 2);
        assert_eq!(r.den, 3);
    }

    #[test]
    fn test_new_normalises_sign() {
        let r = RationalTime::new(3, -4);
        assert_eq!(r.num, -3);
        assert_eq!(r.den, 4);
    }

    #[test]
    fn test_zero() {
        let r = RationalTime::zero();
        assert_eq!(r.num, 0);
        assert_eq!(r.den, 1);
    }

    #[test]
    fn test_one() {
        let r = RationalTime::one();
        assert_eq!(r.num, 1);
        assert_eq!(r.den, 1);
    }

    #[test]
    fn test_from_fps_ntsc_2997() {
        let r = RationalTime::from_fps(29.97);
        assert_eq!(r.num, 30_000);
        assert_eq!(r.den, 1001);
    }

    #[test]
    fn test_from_fps_ntsc_23976() {
        let r = RationalTime::from_fps(23.976);
        assert_eq!(r.num, 24_000);
        assert_eq!(r.den, 1001);
    }

    #[test]
    fn test_from_fps_ntsc_5994() {
        let r = RationalTime::from_fps(59.94);
        assert_eq!(r.num, 60_000);
        assert_eq!(r.den, 1001);
    }

    #[test]
    fn test_from_fps_integer() {
        let r = RationalTime::from_fps(25.0);
        assert_eq!(r.num, 25);
        assert_eq!(r.den, 1);
    }

    #[test]
    fn test_to_f64() {
        let r = RationalTime::new(1, 2);
        assert!((r.to_f64() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_rescale_90k_to_1k() {
        // 90000 ticks at 1/90000 time base → convert to 1/1000 time base
        // expected: (90000 * 1000 + 45000) / 90000 = 1000
        let r = RationalTime::new(90_000, 90_000);
        assert_eq!(r.rescale(1000), 1000);
    }

    #[test]
    fn test_add() {
        let a = RationalTime::new(1, 3);
        let b = RationalTime::new(1, 6);
        assert_eq!(a + b, RationalTime::new(1, 2));
    }

    #[test]
    fn test_sub() {
        let a = RationalTime::new(3, 4);
        let b = RationalTime::new(1, 4);
        assert_eq!(a - b, RationalTime::new(1, 2));
    }

    #[test]
    fn test_mul() {
        let a = RationalTime::new(2, 3);
        let b = RationalTime::new(3, 4);
        assert_eq!(a * b, RationalTime::new(1, 2));
    }

    #[test]
    fn test_div() {
        let a = RationalTime::new(2, 3);
        let b = RationalTime::new(4, 5);
        assert_eq!(a / b, RationalTime::new(5, 6));
    }

    #[test]
    fn test_ord() {
        let a = RationalTime::new(1, 3);
        let b = RationalTime::new(1, 2);
        assert!(a < b);
        assert!(b > a);
    }

    #[test]
    fn test_gcd_i64() {
        assert_eq!(gcd_i64(12, 8), 4);
        assert_eq!(gcd_i64(7, 5), 1);
        assert_eq!(gcd_i64(0, 5), 5);
        assert_eq!(gcd_i64(-6, 9), 3);
    }

    #[test]
    fn test_lcm_i64() {
        assert_eq!(lcm_i64(4, 6), 12);
        assert_eq!(lcm_i64(0, 5), 0);
        assert_eq!(lcm_i64(7, 1), 7);
    }
}
