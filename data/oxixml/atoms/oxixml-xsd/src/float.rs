//! The XML Schema `float` datatype.

use crate::double::{format_finite_float, is_xsd_binary_float_lexical};
use crate::{Boolean, Double, Integer};
use std::cmp::Ordering;
use std::fmt;
use std::num::ParseFloatError;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::str::FromStr;

/// [XML Schema `float` datatype](https://www.w3.org/TR/xmlschema11-2/#float)
///
/// Uses internally a [`f32`].
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[repr(transparent)]
pub struct Float {
    value: f32,
}

impl Float {
    /// Positive infinity (`INF`).
    pub const INFINITY: Self = Self {
        value: f32::INFINITY,
    };
    /// The largest finite value.
    pub const MAX: Self = Self { value: f32::MAX };
    /// The smallest (most negative) finite value.
    pub const MIN: Self = Self { value: f32::MIN };
    /// Not a Number (`NaN`).
    pub const NAN: Self = Self { value: f32::NAN };
    /// Negative infinity (`-INF`).
    pub const NEG_INFINITY: Self = Self {
        value: f32::NEG_INFINITY,
    };

    /// Reads the value from its IEEE-754 big-endian encoding.
    #[inline]
    #[must_use]
    pub fn from_be_bytes(bytes: [u8; 4]) -> Self {
        Self {
            value: f32::from_be_bytes(bytes),
        }
    }

    /// Returns the IEEE-754 big-endian encoding of the value.
    #[inline]
    #[must_use]
    pub fn to_be_bytes(self) -> [u8; 4] {
        self.value.to_be_bytes()
    }

    /// [fn:abs](https://www.w3.org/TR/xpath-functions-31/#func-abs)
    #[inline]
    #[must_use]
    pub fn abs(self) -> Self {
        self.value.abs().into()
    }

    /// [fn:ceiling](https://www.w3.org/TR/xpath-functions-31/#func-ceiling)
    #[inline]
    #[must_use]
    pub fn ceil(self) -> Self {
        self.value.ceil().into()
    }

    /// [fn:floor](https://www.w3.org/TR/xpath-functions-31/#func-floor)
    #[inline]
    #[must_use]
    pub fn floor(self) -> Self {
        self.value.floor().into()
    }

    /// [fn:round](https://www.w3.org/TR/xpath-functions-31/#func-round)
    #[inline]
    #[must_use]
    pub fn round(self) -> Self {
        self.value.round().into()
    }

    /// Returns `true` if this value is `NaN`.
    #[inline]
    #[must_use]
    pub fn is_nan(self) -> bool {
        self.value.is_nan()
    }

    /// Returns `true` if this value is neither infinite nor `NaN`.
    #[inline]
    #[must_use]
    pub fn is_finite(self) -> bool {
        self.value.is_finite()
    }

    /// Checks if the two values are [identical](https://www.w3.org/TR/xmlschema11-2/#identity).
    ///
    /// Unlike [`PartialEq`], `NaN` is identical to itself and `0.0` is not
    /// identical to `-0.0`.
    #[inline]
    #[must_use]
    pub fn is_identical_with(self, other: Self) -> bool {
        self.value.to_bits() == other.value.to_bits()
    }
}

impl From<Float> for f32 {
    #[inline]
    fn from(value: Float) -> Self {
        value.value
    }
}

impl From<Float> for f64 {
    #[inline]
    fn from(value: Float) -> Self {
        value.value.into()
    }
}

impl From<f32> for Float {
    #[inline]
    fn from(value: f32) -> Self {
        Self { value }
    }
}

impl From<i8> for Float {
    #[inline]
    fn from(value: i8) -> Self {
        Self {
            value: value.into(),
        }
    }
}

impl From<i16> for Float {
    #[inline]
    fn from(value: i16) -> Self {
        Self {
            value: value.into(),
        }
    }
}

impl From<u8> for Float {
    #[inline]
    fn from(value: u8) -> Self {
        Self {
            value: value.into(),
        }
    }
}

impl From<u16> for Float {
    #[inline]
    fn from(value: u16) -> Self {
        Self {
            value: value.into(),
        }
    }
}

impl From<Boolean> for Float {
    #[inline]
    fn from(value: Boolean) -> Self {
        f32::from(bool::from(value)).into()
    }
}

impl From<Integer> for Float {
    #[inline]
    fn from(value: Integer) -> Self {
        (i64::from(value) as f32).into()
    }
}

impl From<Double> for Float {
    #[inline]
    fn from(value: Double) -> Self {
        Self {
            value: f64::from(value) as f32,
        }
    }
}

impl FromStr for Float {
    type Err = ParseFloatError;

    /// Parses the XSD `float` lexical space (`INF`, `+INF`, `-INF`, `NaN` and the usual
    /// mantissa/exponent forms).
    ///
    /// Rust's own [`f32::from_str`] is *more* permissive than XSD: it accepts `Infinity`,
    /// `inf`, `INFINITY`, `nan`, `naN`, … in any casing. Those lexical forms are **not** in
    /// the `float` lexical space ([XSD 1.1 Part 2
    /// §3.3.4](https://www.w3.org/TR/xmlschema11-2/#float)), so they are rejected here.
    #[inline]
    fn from_str(input: &str) -> Result<Self, Self::Err> {
        // Anything outside the XSD lexical space is funnelled through a string that no float
        // parser accepts, so the rejection surfaces as the very same `ParseFloatError`.
        let lexical = if is_xsd_binary_float_lexical(input) {
            input
        } else {
            ""
        };
        Ok(f32::from_str(lexical)?.into())
    }
}

impl fmt::Display for Float {
    /// Formats the XSD canonical representation, e.g. `1.0E0`, `INF` or `NaN`.
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.value == f32::INFINITY {
            f.write_str("INF")
        } else if self.value == f32::NEG_INFINITY {
            f.write_str("-INF")
        } else if self.value.is_nan() {
            f.write_str("NaN")
        } else {
            format_finite_float(self.value, f)
        }
    }
}

impl PartialOrd for Float {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

impl Neg for Float {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        (-self.value).into()
    }
}

impl Add for Float {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        (self.value + rhs.value).into()
    }
}

impl Sub for Float {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        (self.value - rhs.value).into()
    }
}

impl Mul for Float {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        (self.value * rhs.value).into()
    }
}

impl Div for Float {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self {
        (self.value / rhs.value).into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eq() {
        assert_eq!(Float::from(0.), Float::from(0.));
        assert_ne!(Float::NAN, Float::NAN);
        assert_eq!(Float::from(-0.), Float::from(0.));
    }

    #[test]
    fn cmp() {
        assert_eq!(
            Float::from(0.).partial_cmp(&Float::from(0.)),
            Some(Ordering::Equal)
        );
        assert_eq!(
            Float::INFINITY.partial_cmp(&Float::MAX),
            Some(Ordering::Greater)
        );
        assert_eq!(
            Float::NEG_INFINITY.partial_cmp(&Float::MIN),
            Some(Ordering::Less)
        );
        assert_eq!(Float::NAN.partial_cmp(&Float::from(0.)), None);
        assert_eq!(Float::NAN.partial_cmp(&Float::NAN), None);
        assert_eq!(
            Float::from(0.).partial_cmp(&Float::from(-0.)),
            Some(Ordering::Equal)
        );
    }

    #[test]
    fn is_identical_with() {
        assert!(Float::from(0.).is_identical_with(Float::from(0.)));
        assert!(Float::NAN.is_identical_with(Float::NAN));
        assert!(!Float::from(-0.).is_identical_with(Float::from(0.)));
    }

    #[test]
    fn from_str() -> Result<(), ParseFloatError> {
        assert_eq!(Float::from_str("NaN")?.to_string(), "NaN");
        assert_eq!(Float::from_str("INF")?.to_string(), "INF");
        assert_eq!(Float::from_str("+INF")?.to_string(), "INF");
        assert_eq!(Float::from_str("-INF")?.to_string(), "-INF");
        assert_eq!(Float::from_str("0.0E0")?.to_string(), "0.0E0");
        assert_eq!(Float::from_str("-0.0E0")?.to_string(), "-0.0E0");
        assert_eq!(Float::from_str("0.1")?.to_string(), "1.0E-1");
        assert_eq!(Float::from_str("0.1e1")?.to_string(), "1.0E0");
        assert_eq!(Float::from_str("-0.1e1")?.to_string(), "-1.0E0");
        assert_eq!(Float::from_str("1.e1")?.to_string(), "1.0E1");
        assert_eq!(Float::from_str("-1.e1")?.to_string(), "-1.0E1");
        assert_eq!(Float::from_str("12")?.to_string(), "1.2E1");
        assert_eq!(Float::from_str("0.1")?.to_string(), "1.0E-1");
        assert_eq!(Float::from_str("0.01")?.to_string(), "1.0E-2");
        assert_eq!(Float::from_str("1")?.to_string(), "1.0E0");
        assert_eq!(Float::from_str("-1")?.to_string(), "-1.0E0");
        assert_eq!(Float::from_str("1.")?.to_string(), "1.0E0");
        assert_eq!(Float::from_str("-1.")?.to_string(), "-1.0E0");
        assert_eq!(Float::from_str("+1e20")?.to_string(), "1.0E20");
        assert_eq!(Float::from_str(&f32::MIN.to_string())?, Float::MIN);
        assert_eq!(Float::from_str(&f32::MAX.to_string())?, Float::MAX);
        Ok(())
    }

    #[test]
    fn format() {
        assert_eq!(Float::from(f32::INFINITY).to_string(), "INF");
        assert_eq!(Float::from(f32::NEG_INFINITY).to_string(), "-INF");
        assert_eq!(Float::from(f32::NAN).to_string(), "NaN");
        assert_eq!(Float::from(f32::MIN).to_string(), "-3.4028235E38");
        assert_eq!(Float::from(f32::MAX).to_string(), "3.4028235E38");
        assert_eq!(Float::from(f32::EPSILON).to_string(), "1.1920929E-7");
    }
    /// Regression test for the `Float` lexical space
    /// ([XSD 1.1 Part 2](https://www.w3.org/TR/xmlschema11-2/#float)):
    /// `numericalSpecialRep ::= 'INF' | '+INF' | '-INF' | 'NaN'` is case-sensitive and
    /// exhaustive. Rust's own [`f32::from_str`] additionally accepts `Infinity`, `inf`,
    /// `INFINITY`, `nan`, `naN`, … and delegating to it straight made every one of those a
    /// valid `xs:float`.
    #[test]
    fn rejects_non_xsd_special_values() -> Result<(), ParseFloatError> {
        for lexical in [
            "Infinity",
            "-Infinity",
            "+Infinity",
            "infinity",
            "INFINITY",
            "inf",
            "Inf",
            "-inf",
            "nan",
            "naN",
            "NAN",
            "Nan",
        ] {
            assert!(
                Float::from_str(lexical).is_err(),
                "{lexical} is outside the xs:float lexical space"
            );
        }

        // Nothing else that XSD allows may be lost.
        assert_eq!(Float::from_str("INF")?.to_string(), "INF");
        assert_eq!(Float::from_str("+INF")?.to_string(), "INF");
        assert_eq!(Float::from_str("-INF")?.to_string(), "-INF");
        assert_eq!(Float::from_str("NaN")?.to_string(), "NaN");
        assert_eq!(Float::from_str("1")?.to_string(), "1.0E0");
        assert_eq!(Float::from_str("-1.5e2")?.to_string(), "-1.5E2");
        assert_eq!(Float::from_str(".5")?.to_string(), "5.0E-1");
        assert_eq!(Float::from_str("1.")?.to_string(), "1.0E0");

        // Neither may plainly malformed numerals become acceptable.
        for lexical in [
            "", " 1", "1 ", "1e", "1e+", "e1", ".", "+.", "1.2.3", "0x1", "1_0",
        ] {
            assert!(
                Float::from_str(lexical).is_err(),
                "{lexical:?} is not a xs:float numeral"
            );
        }
        Ok(())
    }
}
