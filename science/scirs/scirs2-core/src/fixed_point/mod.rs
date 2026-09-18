//! Fixed-Point Arithmetic for Embedded Systems
//!
//! This module provides fixed-point number types and operations for embedded systems
//! without hardware floating-point units (FPU). Fixed-point arithmetic offers:
//!
//! - **Deterministic Performance**: Constant-time operations
//! - **No FPU Required**: Works on ARM Cortex-M0/M3, AVR, etc.
//! - **Predictable Precision**: Known precision at compile-time
//! - **Efficient**: Uses integer operations
//!
//! # Fixed-Point Format
//!
//! Fixed-point numbers use Q notation: Qm.n format where:
//! - m = integer bits
//! - n = fractional bits
//!
//! Common formats:
//! - **Q15.16**: 16-bit integer, 16-bit fraction (i32 storage)
//! - **Q7.24**: 8-bit integer, 24-bit fraction (i32 storage)
//! - **Q31.32**: 32-bit integer, 32-bit fraction (i64 storage)
//!
//! # Example: Basic Fixed-Point Math
//!
//! ```rust
//! use scirs2_core::fixed_point::*;
//!
//! // Q15.16 format (most common for general use)
//! let a = Fixed32::<16>::from_float(3.14159);
//! let b = Fixed32::<16>::from_float(2.0);
//!
//! let sum = a + b;
//! let product = a * b;
//!
//! let result: f32 = sum.to_float();
//! assert!((result - 5.14159).abs() < 0.001);
//! ```
//!
//! # Example: Signal Processing
//!
//! ```rust
//! use scirs2_core::fixed_point::*;
//!
//! // FIR filter with fixed-point coefficients
//! let coeffs = [
//!     Fixed32::<16>::from_float(0.1),
//!     Fixed32::<16>::from_float(0.2),
//!     Fixed32::<16>::from_float(0.4),
//!     Fixed32::<16>::from_float(0.2),
//!     Fixed32::<16>::from_float(0.1),
//! ];
//!
//! // Apply filter
//! let mut output = Fixed32::<16>::ZERO;
//! for coeff in &coeffs {
//!     output = output + *coeff;
//! }
//! ```

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "fixed-point")]
use fixed::{types::*, FixedI32, FixedI64};

use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Fixed-point number with 32-bit storage (Q format)
///
/// Generic over the number of fractional bits N.
/// Common configurations:
/// - N=16: Q15.16 format (range: ±32768, precision: 0.0000153)
/// - N=24: Q7.24 format (range: ±128, precision: 0.00000006)
///
/// # Example
///
/// ```rust
/// use scirs2_core::fixed_point::Fixed32;
///
/// // Q15.16 format
/// let x = Fixed32::<16>::from_float(3.14159);
/// let y = Fixed32::<16>::from_int(2);
/// let z = x * y;
///
/// assert!((z.to_float() - 6.28318).abs() < 0.001);
/// ```
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Fixed32<const FRAC: u32> {
    raw: i32,
}

impl<const FRAC: u32> Fixed32<FRAC> {
    /// Zero constant
    pub const ZERO: Self = Self { raw: 0 };

    /// One constant
    pub const ONE: Self = Self { raw: 1 << FRAC };

    /// Minimum value
    pub const MIN: Self = Self { raw: i32::MIN };

    /// Maximum value
    pub const MAX: Self = Self { raw: i32::MAX };

    /// Create from raw integer representation
    #[inline]
    pub const fn from_raw(raw: i32) -> Self {
        Self { raw }
    }

    /// Get the raw integer representation
    #[inline]
    pub const fn to_raw(self) -> i32 {
        self.raw
    }

    /// Create from integer (shifts left by FRAC bits)
    #[inline]
    pub const fn from_int(value: i32) -> Self {
        Self { raw: value << FRAC }
    }

    /// Convert to integer (truncates fractional part)
    #[inline]
    pub const fn to_int(self) -> i32 {
        self.raw >> FRAC
    }

    /// Create from float (compile-time if possible)
    #[inline]
    pub fn from_float(value: f32) -> Self {
        let scale = (1u64 << FRAC) as f32;
        Self {
            raw: (value * scale) as i32,
        }
    }

    /// Convert to float
    #[inline]
    pub fn to_float(self) -> f32 {
        let scale = (1u64 << FRAC) as f32;
        self.raw as f32 / scale
    }

    /// Absolute value
    #[inline]
    pub const fn abs(self) -> Self {
        Self {
            raw: self.raw.abs(),
        }
    }

    /// Square root (using integer approximation)
    ///
    /// For a fixed-point number in Q format with FRAC fractional bits,
    /// we compute sqrt by shifting to 64-bit intermediate to maintain precision.
    /// Given raw value r representing r / 2^FRAC, we want sqrt(r / 2^FRAC) = sqrt(r * 2^FRAC) / 2^FRAC.
    /// So we compute isqrt(r << FRAC) which gives the raw result directly.
    pub fn sqrt(self) -> Self {
        if self.raw <= 0 {
            return Self::ZERO;
        }

        // Shift to 64-bit to preserve precision: isqrt(raw << FRAC)
        let scaled = (self.raw as u64) << FRAC;

        // Newton-Raphson iteration for integer sqrt of scaled value
        // Initial guess using bit length
        let bits = 64 - scaled.leading_zeros();
        let mut result: u64 = 1u64 << ((bits + 1) / 2);

        // Newton iterations (sufficient for 64-bit convergence)
        for _ in 0..32 {
            if result == 0 {
                break;
            }
            let next = (result + scaled / result) >> 1;
            if next >= result {
                break; // Converged
            }
            result = next;
        }

        Self { raw: result as i32 }
    }

    /// Saturating addition
    #[inline]
    pub const fn saturating_add(self, rhs: Self) -> Self {
        Self {
            raw: self.raw.saturating_add(rhs.raw),
        }
    }

    /// Saturating subtraction
    #[inline]
    pub const fn saturating_sub(self, rhs: Self) -> Self {
        Self {
            raw: self.raw.saturating_sub(rhs.raw),
        }
    }

    /// Saturating multiplication
    #[inline]
    pub fn saturating_mul(self, rhs: Self) -> Self {
        let result = (self.raw as i64 * rhs.raw as i64) >> FRAC;
        Self {
            raw: result.clamp(i32::MIN as i64, i32::MAX as i64) as i32,
        }
    }
}

// Arithmetic operations
impl<const FRAC: u32> Add for Fixed32<FRAC> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            raw: self.raw + rhs.raw,
        }
    }
}

impl<const FRAC: u32> AddAssign for Fixed32<FRAC> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.raw += rhs.raw;
    }
}

impl<const FRAC: u32> Sub for Fixed32<FRAC> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            raw: self.raw - rhs.raw,
        }
    }
}

impl<const FRAC: u32> SubAssign for Fixed32<FRAC> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.raw -= rhs.raw;
    }
}

impl<const FRAC: u32> Mul for Fixed32<FRAC> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        let result = (self.raw as i64 * rhs.raw as i64) >> FRAC;
        Self { raw: result as i32 }
    }
}

impl<const FRAC: u32> MulAssign for Fixed32<FRAC> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<const FRAC: u32> Div for Fixed32<FRAC> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        let result = ((self.raw as i64) << FRAC) / rhs.raw as i64;
        Self { raw: result as i32 }
    }
}

impl<const FRAC: u32> DivAssign for Fixed32<FRAC> {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl<const FRAC: u32> Neg for Fixed32<FRAC> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self { raw: -self.raw }
    }
}

impl<const FRAC: u32> Default for Fixed32<FRAC> {
    fn default() -> Self {
        Self::ZERO
    }
}

impl<const FRAC: u32> fmt::Display for Fixed32<FRAC> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_float())
    }
}

/// Fixed-point number with 64-bit storage (Q format)
///
/// For higher precision or larger range requirements.
///
/// # Example
///
/// ```rust
/// use scirs2_core::fixed_point::Fixed64;
///
/// // Q31.32 format (high precision)
/// let x = Fixed64::<32>::from_double(3.14159265358979);
/// let y = Fixed64::<32>::from_double(2.0);
/// let z = x * y;
///
/// assert!((z.to_double() - 6.28318530717958).abs() < 1e-8);
/// ```
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Fixed64<const FRAC: u32> {
    raw: i64,
}

impl<const FRAC: u32> Fixed64<FRAC> {
    /// Zero constant
    pub const ZERO: Self = Self { raw: 0 };

    /// One constant
    pub const ONE: Self = Self { raw: 1 << FRAC };

    /// Create from raw integer representation
    #[inline]
    pub const fn from_raw(raw: i64) -> Self {
        Self { raw }
    }

    /// Get the raw integer representation
    #[inline]
    pub const fn to_raw(self) -> i64 {
        self.raw
    }

    /// Create from integer
    #[inline]
    pub const fn from_int(value: i64) -> Self {
        Self { raw: value << FRAC }
    }

    /// Convert to integer
    #[inline]
    pub const fn to_int(self) -> i64 {
        self.raw >> FRAC
    }

    /// Create from float
    #[inline]
    pub fn from_float(value: f32) -> Self {
        let scale = (1u64 << FRAC) as f32;
        Self {
            raw: (value * scale) as i64,
        }
    }

    /// Create from double
    #[inline]
    pub fn from_double(value: f64) -> Self {
        let scale = (1u64 << FRAC) as f64;
        Self {
            raw: (value * scale) as i64,
        }
    }

    /// Convert to float
    #[inline]
    pub fn to_float(self) -> f32 {
        let scale = (1u64 << FRAC) as f32;
        self.raw as f32 / scale
    }

    /// Convert to double
    #[inline]
    pub fn to_double(self) -> f64 {
        let scale = (1u64 << FRAC) as f64;
        self.raw as f64 / scale
    }
}

// Arithmetic operations for Fixed64
impl<const FRAC: u32> Add for Fixed64<FRAC> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            raw: self.raw + rhs.raw,
        }
    }
}

impl<const FRAC: u32> Sub for Fixed64<FRAC> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            raw: self.raw - rhs.raw,
        }
    }
}

impl<const FRAC: u32> Mul for Fixed64<FRAC> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        let result = (self.raw as i128 * rhs.raw as i128) >> FRAC;
        Self { raw: result as i64 }
    }
}

impl<const FRAC: u32> Div for Fixed64<FRAC> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        let result = ((self.raw as i128) << FRAC) / rhs.raw as i128;
        Self { raw: result as i64 }
    }
}

impl<const FRAC: u32> Default for Fixed64<FRAC> {
    fn default() -> Self {
        Self::ZERO
    }
}

impl<const FRAC: u32> fmt::Display for Fixed64<FRAC> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_double())
    }
}

/// Type aliases for common fixed-point formats
pub mod types {
    use super::*;

    /// Q15.16 format - most common for general use
    /// Range: ±32768, Precision: ~0.000015
    pub type Q16 = Fixed32<16>;

    /// Q7.24 format - higher precision, smaller range
    /// Range: ±128, Precision: ~0.00000006
    pub type Q24 = Fixed32<24>;

    /// Q23.8 format - lower precision, larger range
    /// Range: ±8388608, Precision: ~0.004
    pub type Q8 = Fixed32<8>;

    /// Q31.32 format - high precision with 64-bit storage
    /// Range: ±2147483648, Precision: ~2.3e-10
    pub type Q32 = Fixed64<32>;
}

/// Mathematical functions for fixed-point numbers
pub mod math {
    use super::*;

    /// Sine approximation using Taylor series (for embedded systems)
    ///
    /// Accurate for angles in radians within ±2π
    pub fn sin<const FRAC: u32>(x: Fixed32<FRAC>) -> Fixed32<FRAC> {
        // Reduce angle to [-π, π]
        let pi = Fixed32::<FRAC>::from_float(core::f32::consts::PI);
        let two_pi = Fixed32::<FRAC>::from_float(2.0 * core::f32::consts::PI);

        let mut angle = x;
        while angle > pi {
            angle -= two_pi;
        }
        while angle < -pi {
            angle += two_pi;
        }

        // Taylor series: sin(x) ≈ x - x³/3! + x⁵/5! - x⁷/7!
        let x2 = angle * angle;
        let x3 = x2 * angle;
        let x5 = x3 * x2;
        let x7 = x5 * x2;

        let term1 = angle;
        let term2 = x3 / Fixed32::<FRAC>::from_int(6); // x³/3!
        let term3 = x5 / Fixed32::<FRAC>::from_int(120); // x⁵/5!
        let term4 = x7 / Fixed32::<FRAC>::from_int(5040); // x⁷/7!

        term1 - term2 + term3 - term4
    }

    /// Cosine approximation
    pub fn cos<const FRAC: u32>(x: Fixed32<FRAC>) -> Fixed32<FRAC> {
        let half_pi = Fixed32::<FRAC>::from_float(core::f32::consts::FRAC_PI_2);
        sin(x + half_pi)
    }

    /// Exponential approximation (e^x)
    pub fn exp<const FRAC: u32>(x: Fixed32<FRAC>) -> Fixed32<FRAC> {
        // Series: e^x = 1 + x + x²/2! + x³/3! + x⁴/4! + ...
        let mut result = Fixed32::<FRAC>::ONE;
        let mut term = Fixed32::<FRAC>::ONE;

        for n in 1..10 {
            term = term * x / Fixed32::<FRAC>::from_int(n);
            result += term;

            // Early exit if term becomes negligible
            if term.abs().to_raw().abs() < 4 {
                break;
            }
        }

        result
    }

    /// Natural logarithm approximation
    pub fn ln<const FRAC: u32>(x: Fixed32<FRAC>) -> Fixed32<FRAC> {
        if x.to_raw() <= 0 {
            return Fixed32::<FRAC>::MIN;
        }

        // Use series expansion around x=1
        // ln(x) = 2 * ((x-1)/(x+1) + (1/3)*((x-1)/(x+1))³ + ...)
        let one = Fixed32::<FRAC>::ONE;
        let numerator = x - one;
        let denominator = x + one;
        let y = numerator / denominator;
        let y2 = y * y;

        let term1 = y;
        let term2 = (y * y2) / Fixed32::<FRAC>::from_int(3);
        let term3 = (y * y2 * y2) / Fixed32::<FRAC>::from_int(5);

        Fixed32::<FRAC>::from_int(2) * (term1 + term2 + term3)
    }
}

/// Signal processing utilities with fixed-point
pub mod signal {
    use super::*;

    /// FIR filter with fixed-point coefficients
    ///
    /// # Example
    ///
    /// ```rust
    /// use scirs2_core::fixed_point::{Fixed32, signal::FirFilter};
    ///
    /// let coeffs = [Fixed32::<16>::from_float(0.2); 5];
    /// let mut filter = FirFilter::new(&coeffs);
    ///
    /// let output = filter.process(Fixed32::<16>::from_float(1.0));
    /// ```
    pub struct FirFilter<const FRAC: u32, const N: usize> {
        coeffs: [Fixed32<FRAC>; N],
        buffer: [Fixed32<FRAC>; N],
        index: usize,
    }

    impl<const FRAC: u32, const N: usize> FirFilter<FRAC, N> {
        /// Create a new FIR filter
        pub fn new(coeffs: &[Fixed32<FRAC>; N]) -> Self {
            Self {
                coeffs: *coeffs,
                buffer: [Fixed32::<FRAC>::ZERO; N],
                index: 0,
            }
        }

        /// Process a single sample
        pub fn process(&mut self, input: Fixed32<FRAC>) -> Fixed32<FRAC> {
            // Store input in circular buffer
            self.buffer[self.index] = input;
            self.index = (self.index + 1) % N;

            // Convolution
            let mut output = Fixed32::<FRAC>::ZERO;
            for i in 0..N {
                let buf_idx = (self.index + N - 1 - i) % N;
                output += self.coeffs[i] * self.buffer[buf_idx];
            }

            output
        }

        /// Reset the filter state
        pub fn reset(&mut self) {
            self.buffer = [Fixed32::<FRAC>::ZERO; N];
            self.index = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed32_basic() {
        let a = Fixed32::<16>::from_float(core::f32::consts::PI);
        let b = Fixed32::<16>::from_float(2.0);

        let sum = a + b;
        assert!((sum.to_float() - (core::f32::consts::PI + 2.0)).abs() < 0.01);

        let product = a * b;
        assert!((product.to_float() - core::f32::consts::TAU).abs() < 0.01);
    }

    #[test]
    fn test_fixed32_sqrt() {
        let x = Fixed32::<16>::from_float(4.0);
        let sqrt_x = x.sqrt();
        assert!((sqrt_x.to_float() - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_fixed64_precision() {
        let a = Fixed64::<32>::from_double(std::f64::consts::PI);
        let b = Fixed64::<32>::from_double(2.0);

        let product = a * b;
        assert!((product.to_double() - std::f64::consts::TAU).abs() < 1e-9);
    }

    #[test]
    fn test_math_functions() {
        use math::*;

        let x = Fixed32::<16>::from_float(0.0);
        let sin_x = sin(x);
        assert!((sin_x.to_float() - 0.0).abs() < 0.01);

        let half_pi = Fixed32::<16>::from_float(core::f32::consts::FRAC_PI_2);
        let sin_half_pi = sin(half_pi);
        assert!((sin_half_pi.to_float() - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_fir_filter() {
        use signal::*;

        let coeffs = [Fixed32::<16>::from_float(0.2); 5];
        let mut filter = FirFilter::new(&coeffs);

        let input = Fixed32::<16>::from_float(1.0);
        let output = filter.process(input);

        assert!(output.to_float() > 0.0);
    }
}
