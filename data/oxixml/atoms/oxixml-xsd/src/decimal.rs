//! The XML Schema `decimal` datatype.

use crate::{Boolean, Double, Float, Integer, TooLargeForIntegerError};
use std::cmp::Ordering;
use std::error::Error;
use std::fmt;
use std::fmt::Write;
use std::str::FromStr;

const DECIMAL_PART_DIGITS: u32 = 18;
const DECIMAL_PART_POW: i128 = 1_000_000_000_000_000_000;
const DECIMAL_PART_POW_MINUS_ONE: i128 = 100_000_000_000_000_000;

/// [XML Schema `decimal` datatype](https://www.w3.org/TR/xmlschema11-2/#decimal)
///
/// It stores the decimal in a fixed-point encoding allowing nearly 18 digits
/// before and 18 digits after the `.`.
///
/// It stores the value in a [`i128`] integer after multiplying it by 10¹⁸.
#[derive(Eq, PartialEq, Ord, PartialOrd, Debug, Clone, Copy, Hash, Default)]
#[repr(Rust, packed(8))]
pub struct Decimal {
    value: i128, // value * 10^18
}

impl Decimal {
    /// The largest representable `decimal` value.
    pub const MAX: Self = Self { value: i128::MAX };
    /// The smallest representable `decimal` value.
    pub const MIN: Self = Self { value: i128::MIN };
    #[cfg(test)]
    const STEP: Self = Self { value: 1 };

    /// Constructs the decimal `i / 10^n`.
    ///
    /// Returns an error if `n` is greater than 18 or if the value does not fit
    /// the internal representation.
    #[inline]
    pub const fn new(i: i128, n: u32) -> Result<Self, TooLargeForDecimalError> {
        let Some(shift) = DECIMAL_PART_DIGITS.checked_sub(n) else {
            return Err(TooLargeForDecimalError);
        };
        let Some(value) = i.checked_mul(10_i128.pow(shift)) else {
            return Err(TooLargeForDecimalError);
        };
        Ok(Self { value })
    }

    pub(crate) const fn new_from_i128_unchecked(value: i128) -> Self {
        Self {
            value: value * DECIMAL_PART_POW,
        }
    }

    /// Reads the value from its big-endian in-memory encoding.
    #[inline]
    #[must_use]
    pub fn from_be_bytes(bytes: [u8; 16]) -> Self {
        Self {
            value: i128::from_be_bytes(bytes),
        }
    }

    /// Returns the big-endian in-memory encoding of the value.
    #[inline]
    #[must_use]
    pub fn to_be_bytes(self) -> [u8; 16] {
        self.value.to_be_bytes()
    }

    /// [op:numeric-add](https://www.w3.org/TR/xpath-functions-31/#func-numeric-add)
    ///
    /// Returns `None` in case of overflow ([FOAR0002](https://www.w3.org/TR/xpath-functions-31/#ERRFOAR0002)).
    #[inline]
    #[must_use]
    pub fn checked_add(self, rhs: impl Into<Self>) -> Option<Self> {
        Some(Self {
            value: self.value.checked_add(rhs.into().value)?,
        })
    }

    /// [op:numeric-subtract](https://www.w3.org/TR/xpath-functions-31/#func-numeric-subtract)
    ///
    /// Returns `None` in case of overflow ([FOAR0002](https://www.w3.org/TR/xpath-functions-31/#ERRFOAR0002)).
    #[inline]
    #[must_use]
    pub fn checked_sub(self, rhs: impl Into<Self>) -> Option<Self> {
        Some(Self {
            value: self.value.checked_sub(rhs.into().value)?,
        })
    }

    /// [op:numeric-multiply](https://www.w3.org/TR/xpath-functions-31/#func-numeric-multiply)
    ///
    /// Returns `None` in case of overflow ([FOAR0002](https://www.w3.org/TR/xpath-functions-31/#ERRFOAR0002)).
    #[inline]
    #[must_use]
    pub fn checked_mul(self, rhs: impl Into<Self>) -> Option<Self> {
        // Idea: we shift right as much as possible to keep as much precision as possible,
        // do the multiplication and then the required left shift.
        //
        // Zero is handled first: it has no trailing zeros to shift out, so the shift budget
        // below would come up `DECIMAL_PART_DIGITS` short and report an overflow for a product
        // that is simply `0` — which used to make `1.5 * 0` and even `0 * 0` raise `FOAR0002`.
        let rhs = rhs.into();
        if self.value == 0 || rhs.value == 0 {
            return Some(Self { value: 0 });
        }
        let mut left = self.value;
        let mut shift_left = 0_u32;
        while left % 10 == 0 {
            left /= 10;
            shift_left += 1;
        }

        let mut right = rhs.value;
        let mut shift_right = 0_u32;
        while right % 10 == 0 {
            right /= 10;
            shift_right += 1;
        }

        // We do multiplication + shift.
        let shift = (shift_left + shift_right).checked_sub(DECIMAL_PART_DIGITS)?;
        Some(Self {
            value: left
                .checked_mul(right)?
                .checked_mul(10_i128.checked_pow(shift)?)?,
        })
    }

    /// [op:numeric-divide](https://www.w3.org/TR/xpath-functions-31/#func-numeric-divide)
    ///
    /// Returns `None` in case of division by 0 ([FOAR0001](https://www.w3.org/TR/xpath-functions-31/#ERRFOAR0001)) or overflow ([FOAR0002](https://www.w3.org/TR/xpath-functions-31/#ERRFOAR0002)).
    #[inline]
    #[must_use]
    pub fn checked_div(self, rhs: impl Into<Self>) -> Option<Self> {
        // Idea: we shift the dividend left as much as possible to keep as much precision as possible
        // and we shift the divisor right as much as possible, then do the division and the required shift.
        //
        // A zero dividend is answered directly, for the same reason `checked_mul` answers a zero
        // operand directly: it cannot be shifted, so the shift budget below would come up short
        // and report an overflow for the perfectly ordinary quotient `0`. Division *by* zero
        // still fails, as `FOAR0001` requires.
        let rhs = rhs.into();
        if self.value == 0 {
            return (rhs.value != 0).then_some(Self { value: 0 });
        }
        let mut left = self.value;
        let mut shift_left = 0_u32;
        while let Some(r) = left.checked_mul(10) {
            left = r;
            shift_left += 1;
        }
        let mut right = rhs.value;
        let mut shift_right = 0_u32;
        if right != 0 {
            while right % 10 == 0 {
                right /= 10;
                shift_right += 1;
            }
        }

        // We do division + shift.
        let shift = (shift_left + shift_right).checked_sub(DECIMAL_PART_DIGITS)?;
        if right == 0 {
            return None;
        }

        // Both divisions below discard a remainder, and simply dropping them
        // truncates the quotient towards zero: `830993497117024304 div
        // -999999999999999999` would answer -0.830993497117024304 where the
        // exact quotient is -0.830993497117024304830993..., whose eighteenth
        // fractional digit rounds to ...305. QT3 `op/numeric-divide`
        // `op-numeric-divideintg2args-2` asserts exactly that value (and
        // likewise `dividedec2args-2`, `dividedec2args-4`,
        // `dividenpi2args-4`), so the two remainders are kept and the single
        // rounding step below applies round-half-to-even, the rule
        // `fn:round-half-to-even` — which those cases wrap the division in —
        // would apply at the same eighteen digits.
        //
        // The magnitudes are computed in `u128` because the exact quotient of
        // `Decimal::MIN` by one is `Decimal::MIN`, whose magnitude is one past
        // `i128::MAX`.
        let negative = (left < 0) != (right < 0);
        let dividend = left.unsigned_abs();
        let divisor = right.unsigned_abs();
        let scale = 10_u128.checked_pow(shift)?;
        let (quotient, remainder) = (dividend / divisor, dividend % divisor);
        let (mut magnitude, dropped) = (quotient / scale, quotient % scale);

        // How the discarded tail `(dropped + remainder / divisor) / scale`
        // compares with one half. Writing `kept` for `dropped` and `rest` for
        // `scale - dropped`, that is `kept + 2 * remainder / divisor` against
        // `rest`, which stays inside `u128` — `2 * dropped` and `2 * remainder`
        // would not, since either can be as large as `u128::MAX / 2` and more.
        let kept = dropped;
        let rest = scale - dropped;
        let tail = if kept > rest {
            Ordering::Greater
        } else if kept + 1 < rest {
            // `2 * remainder / divisor` is under 2, so even at its largest the
            // tail cannot reach the half.
            Ordering::Less
        } else if kept == rest {
            // Exactly half of `scale` was dropped: a tie, unless the inner
            // division left something over.
            if remainder == 0 {
                Ordering::Equal
            } else {
                Ordering::Greater
            }
        } else {
            // `kept + 1 == rest` needs an odd `scale`, so `shift` is 0 and the
            // whole tail is the inner division's own remainder.
            remainder.cmp(&(divisor - remainder))
        };
        let round_away = match tail {
            Ordering::Greater => true,
            Ordering::Less => false,
            // Half-to-even: a tie goes to the neighbour with an even last
            // digit.
            Ordering::Equal => magnitude % 2 == 1,
        };
        if round_away {
            magnitude = magnitude.checked_add(1)?;
        }

        Some(Self {
            value: if negative {
                0_i128.checked_sub_unsigned(magnitude)?
            } else {
                i128::try_from(magnitude).ok()?
            },
        })
    }

    /// [op:numeric-mod](https://www.w3.org/TR/xpath-functions-31/#func-numeric-mod)
    ///
    /// Returns `None` in case of division by 0 ([FOAR0001](https://www.w3.org/TR/xpath-functions-31/#ERRFOAR0001)) or overflow ([FOAR0002](https://www.w3.org/TR/xpath-functions-31/#ERRFOAR0002)).
    #[inline]
    #[must_use]
    pub fn checked_rem(self, rhs: impl Into<Self>) -> Option<Self> {
        Some(Self {
            value: self.value.checked_rem(rhs.into().value)?,
        })
    }

    /// Euclidean remainder.
    ///
    /// Returns `None` in case of division by 0 ([FOAR0001](https://www.w3.org/TR/xpath-functions-31/#ERRFOAR0001)) or overflow ([FOAR0002](https://www.w3.org/TR/xpath-functions-31/#ERRFOAR0002)).
    #[inline]
    #[must_use]
    pub fn checked_rem_euclid(self, rhs: impl Into<Self>) -> Option<Self> {
        Some(Self {
            value: self.value.checked_rem_euclid(rhs.into().value)?,
        })
    }

    /// [op:numeric-unary-minus](https://www.w3.org/TR/xpath-functions-31/#func-numeric-unary-minus)
    ///
    /// Returns `None` in case of overflow ([FOAR0002](https://www.w3.org/TR/xpath-functions-31/#ERRFOAR0002)).
    #[inline]
    #[must_use]
    pub fn checked_neg(self) -> Option<Self> {
        Some(Self {
            value: self.value.checked_neg()?,
        })
    }

    /// [fn:abs](https://www.w3.org/TR/xpath-functions-31/#func-abs)
    ///
    /// Returns `None` in case of overflow ([FOAR0002](https://www.w3.org/TR/xpath-functions-31/#ERRFOAR0002)).
    #[inline]
    #[must_use]
    pub fn checked_abs(self) -> Option<Self> {
        Some(Self {
            value: self.value.checked_abs()?,
        })
    }

    /// [fn:round](https://www.w3.org/TR/xpath-functions-31/#func-round)
    ///
    /// Returns `None` in case of overflow ([FOAR0002](https://www.w3.org/TR/xpath-functions-31/#ERRFOAR0002)).
    #[inline]
    #[must_use]
    pub fn checked_round(self) -> Option<Self> {
        let value = self.value / DECIMAL_PART_POW_MINUS_ONE;
        Some(Self {
            value: if value >= 0 {
                value / 10 + i128::from(value % 10 >= 5)
            } else {
                value / 10 - i128::from(-value % 10 > 5)
            }
            .checked_mul(DECIMAL_PART_POW)?,
        })
    }

    /// [fn:ceiling](https://www.w3.org/TR/xpath-functions-31/#func-ceiling)
    ///
    /// Returns `None` in case of overflow ([FOAR0002](https://www.w3.org/TR/xpath-functions-31/#ERRFOAR0002)).
    #[inline]
    #[must_use]
    pub fn checked_ceil(self) -> Option<Self> {
        Some(Self {
            value: if self.value > 0 && self.value % DECIMAL_PART_POW != 0 {
                self.value / DECIMAL_PART_POW + 1
            } else {
                self.value / DECIMAL_PART_POW
            }
            .checked_mul(DECIMAL_PART_POW)?,
        })
    }

    /// [fn:floor](https://www.w3.org/TR/xpath-functions-31/#func-floor)
    ///
    /// Returns `None` in case of overflow ([FOAR0002](https://www.w3.org/TR/xpath-functions-31/#ERRFOAR0002)).
    #[inline]
    #[must_use]
    pub fn checked_floor(self) -> Option<Self> {
        Some(Self {
            value: if self.value >= 0 || self.value % DECIMAL_PART_POW == 0 {
                self.value / DECIMAL_PART_POW
            } else {
                self.value / DECIMAL_PART_POW - 1
            }
            .checked_mul(DECIMAL_PART_POW)?,
        })
    }

    /// Returns `true` if the value is strictly less than zero.
    #[inline]
    #[must_use]
    pub const fn is_negative(self) -> bool {
        self.value < 0
    }

    /// Returns `true` if the value is strictly greater than zero.
    #[inline]
    #[must_use]
    pub const fn is_positive(self) -> bool {
        self.value > 0
    }

    /// Checks if the two values are [identical](https://www.w3.org/TR/xmlschema11-2/#identity).
    ///
    /// For `decimal` this is the same as [`PartialEq`].
    #[inline]
    #[must_use]
    pub fn is_identical_with(self, other: Self) -> bool {
        self == other
    }

    #[inline]
    #[must_use]
    pub(crate) const fn as_i128(self) -> i128 {
        self.value / DECIMAL_PART_POW
    }
}

impl From<bool> for Decimal {
    #[inline]
    fn from(value: bool) -> Self {
        Self {
            value: i128::from(value) * DECIMAL_PART_POW,
        }
    }
}

impl From<i8> for Decimal {
    #[inline]
    fn from(value: i8) -> Self {
        Self {
            value: i128::from(value) * DECIMAL_PART_POW,
        }
    }
}

impl From<i16> for Decimal {
    #[inline]
    fn from(value: i16) -> Self {
        Self {
            value: i128::from(value) * DECIMAL_PART_POW,
        }
    }
}

impl From<i32> for Decimal {
    #[inline]
    fn from(value: i32) -> Self {
        Self {
            value: i128::from(value) * DECIMAL_PART_POW,
        }
    }
}

impl From<i64> for Decimal {
    #[inline]
    fn from(value: i64) -> Self {
        Self {
            value: i128::from(value) * DECIMAL_PART_POW,
        }
    }
}

impl From<u8> for Decimal {
    #[inline]
    fn from(value: u8) -> Self {
        Self {
            value: i128::from(value) * DECIMAL_PART_POW,
        }
    }
}

impl From<u16> for Decimal {
    #[inline]
    fn from(value: u16) -> Self {
        Self {
            value: i128::from(value) * DECIMAL_PART_POW,
        }
    }
}

impl From<u32> for Decimal {
    #[inline]
    fn from(value: u32) -> Self {
        Self {
            value: i128::from(value) * DECIMAL_PART_POW,
        }
    }
}

impl From<u64> for Decimal {
    #[inline]
    fn from(value: u64) -> Self {
        Self {
            value: i128::from(value) * DECIMAL_PART_POW,
        }
    }
}

impl From<Integer> for Decimal {
    #[inline]
    fn from(value: Integer) -> Self {
        i64::from(value).into()
    }
}

impl TryFrom<i128> for Decimal {
    type Error = TooLargeForDecimalError;

    #[inline]
    fn try_from(value: i128) -> Result<Self, Self::Error> {
        Ok(Self {
            value: value
                .checked_mul(DECIMAL_PART_POW)
                .ok_or(TooLargeForDecimalError)?,
        })
    }
}

impl TryFrom<u128> for Decimal {
    type Error = TooLargeForDecimalError;

    #[inline]
    fn try_from(value: u128) -> Result<Self, Self::Error> {
        Ok(Self {
            value: i128::try_from(value)
                .map_err(|_| TooLargeForDecimalError)?
                .checked_mul(DECIMAL_PART_POW)
                .ok_or(TooLargeForDecimalError)?,
        })
    }
}

impl From<Boolean> for Decimal {
    #[inline]
    fn from(value: Boolean) -> Self {
        bool::from(value).into()
    }
}

impl TryFrom<Float> for Decimal {
    type Error = TooLargeForDecimalError;

    #[inline]
    fn try_from(value: Float) -> Result<Self, Self::Error> {
        Double::from(value).try_into()
    }
}

impl TryFrom<Double> for Decimal {
    type Error = TooLargeForDecimalError;

    #[inline]
    fn try_from(value: Double) -> Result<Self, Self::Error> {
        let shifted = f64::from(value) * (DECIMAL_PART_POW as f64);
        if (i128::MIN as f64) <= shifted && shifted <= (i128::MAX as f64) {
            Ok(Self {
                value: shifted as i128,
            })
        } else {
            Err(TooLargeForDecimalError)
        }
    }
}

impl From<Decimal> for Float {
    #[inline]
    fn from(value: Decimal) -> Self {
        Double::from(value).into()
    }
}

impl From<Decimal> for Double {
    #[inline]
    fn from(value: Decimal) -> Self {
        let mut value = value.value;
        let mut shift = DECIMAL_PART_POW;

        // Hack to improve precision.
        if value != 0 {
            while shift != 1 && value % 10 == 0 {
                value /= 10;
                shift /= 10;
            }
        }

        ((value as f64) / (shift as f64)).into()
    }
}

impl TryFrom<Decimal> for Integer {
    type Error = TooLargeForIntegerError;

    #[inline]
    fn try_from(value: Decimal) -> Result<Self, Self::Error> {
        Ok(i64::try_from(
            value
                .value
                .checked_div(DECIMAL_PART_POW)
                .ok_or(TooLargeForIntegerError)?,
        )
        .map_err(|_| TooLargeForIntegerError)?
        .into())
    }
}

impl FromStr for Decimal {
    type Err = ParseDecimalError;

    /// Parses the XSD `decimal` lexical mapping:
    /// `(\+|-)?([0-9]+(\.[0-9]*)?|\.[0-9]+)`.
    fn from_str(input: &str) -> Result<Self, Self::Err> {
        let input = input.as_bytes();
        if input.is_empty() {
            return Err(PARSE_UNEXPECTED_END);
        }

        let (sign, mut input) = match input.first() {
            Some(b'+') => (1_i128, &input[1..]),
            Some(b'-') => (-1_i128, &input[1..]),
            _ => (1, input),
        };

        let mut value = 0_i128;
        let with_before_dot = input.first().is_some_and(u8::is_ascii_digit);
        while let Some(c) = input.first() {
            if c.is_ascii_digit() {
                value = value
                    .checked_mul(10)
                    .ok_or(PARSE_OVERFLOW)?
                    .checked_add(sign * i128::from(*c - b'0'))
                    .ok_or(PARSE_OVERFLOW)?;
                input = &input[1..];
            } else {
                break;
            }
        }

        let mut exp = DECIMAL_PART_POW;
        if let Some(c) = input.first() {
            if *c != b'.' {
                return Err(PARSE_UNEXPECTED_CHAR);
            }
            input = &input[1..];
            if input.is_empty() && !with_before_dot {
                // We only have a dot.
                return Err(PARSE_UNEXPECTED_END);
            }
            while input.last() == Some(&b'0') {
                // Hack to avoid underflows.
                input = &input[..input.len() - 1];
            }
            while let Some(c) = input.first() {
                if c.is_ascii_digit() {
                    exp /= 10;
                    value = value
                        .checked_mul(10)
                        .ok_or(PARSE_OVERFLOW)?
                        .checked_add(sign * i128::from(*c - b'0'))
                        .ok_or(PARSE_OVERFLOW)?;
                    input = &input[1..];
                } else {
                    return Err(PARSE_UNEXPECTED_CHAR);
                }
            }
            if exp == 0 {
                // Underflow.
                return Err(PARSE_UNDERFLOW);
            }
        } else if !with_before_dot {
            // It's empty.
            return Err(PARSE_UNEXPECTED_END);
        }

        Ok(Self {
            value: value.checked_mul(exp).ok_or(PARSE_OVERFLOW)?,
        })
    }
}

impl fmt::Display for Decimal {
    /// Formats the decimal following its canonical representation.
    ///
    /// Integral values print without a decimal point (e.g. `"1"` rather than
    /// `"1.0"`); non-integral values have at least one digit before and after
    /// the point and no trailing fractional zeros (e.g. `"0.1"`).
    ///
    /// An explicit [`width`](fmt::Formatter::width) zero-pads the integer part (reserving
    /// one character for the sign, if negative), like the standard numeric `Display` impls,
    /// for any width -- including zero and values far beyond what this type can ever need.
    /// An explicit [`precision`](fmt::Formatter::precision) rounds the fractional part to
    /// that many digits (half rounds away from zero, propagating a carry into the integer
    /// part if needed) instead of truncating, and omits the point entirely if that leaves
    /// nothing to print after it (e.g. `{:.0}` on `1.99` prints `"2"`, not `"2."`).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let decimal_part_digits = usize::try_from(DECIMAL_PART_DIGITS).map_err(|_| fmt::Error)?;

        if self.value == 0 {
            return if let Some(width) = f.width() {
                for _ in 0..width {
                    f.write_char('0')?;
                }
                Ok(())
            } else {
                f.write_char('0')
            };
        }

        if self.value.is_negative() {
            f.write_char('-')?;
        }

        // `digits[0]` is the least significant digit (the 10⁻¹⁸ place); `digits[decimal_part_digits]`
        // is the ones place; each following index is one power of ten more significant. Each
        // slot holds the digit's numeric value (0..=9), not an ASCII byte.
        let mut digits = [0_u8; 40];
        let mut value = self.value;
        let mut len = 0;
        while value != 0 {
            digits[len] = (value % 10).unsigned_abs() as u8;
            value /= 10;
            len += 1;
        }

        let mut last_non_zero = len - 1;
        let mut first_non_zero = digits.iter().position(|&d| d != 0).unwrap_or(digits.len());

        // Round to `precision` fractional digits -- when the value has more significant
        // fractional digits than that -- instead of truncating.
        if let Some(precision) = f.precision() {
            if precision < decimal_part_digits {
                let keep_from = decimal_part_digits - precision;
                if keep_from > first_non_zero {
                    if digits[keep_from - 1] >= 5 {
                        let mut i = keep_from;
                        loop {
                            let Some(digit) = digits.get_mut(i) else {
                                // Unreachable for any in-range `Decimal`: its magnitude never
                                // needs more than 39 decimal digits, one short of `digits.len()`.
                                return Err(fmt::Error);
                            };
                            if *digit == 9 {
                                *digit = 0;
                                i += 1;
                            } else {
                                *digit += 1;
                                break;
                            }
                        }
                        if i > last_non_zero {
                            last_non_zero = i;
                        }
                    }
                    // The dropped digits are no longer part of the printed value, whether or
                    // not rounding them up carried into the digits that are kept.
                    first_non_zero = keep_from;
                }
            }
        }

        if last_non_zero >= decimal_part_digits {
            let natural_len = last_non_zero - decimal_part_digits + 1;
            if let Some(mut width) = f.width() {
                if self.value.is_negative() {
                    width = width.saturating_sub(1);
                }
                for _ in natural_len..width {
                    f.write_char('0')?;
                }
            }
            for c in digits[decimal_part_digits..=last_non_zero].iter().rev() {
                f.write_char(char::from(b'0' + *c))?;
            }
        } else {
            f.write_char('0')?;
        }
        if decimal_part_digits > first_non_zero {
            f.write_char('.')?;
            for c in digits[first_non_zero..decimal_part_digits].iter().rev() {
                f.write_char(char::from(b'0' + *c))?;
            }
        }

        Ok(())
    }
}

/// An error raised while parsing a [`Decimal`].
#[derive(Debug, Clone)]
pub struct ParseDecimalError {
    kind: DecimalParseErrorKind,
}

#[derive(Debug, Clone)]
enum DecimalParseErrorKind {
    Overflow,
    Underflow,
    UnexpectedChar,
    UnexpectedEnd,
}

impl fmt::Display for ParseDecimalError {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self.kind {
            DecimalParseErrorKind::Overflow => "Value overflow",
            DecimalParseErrorKind::Underflow => "Value underflow",
            DecimalParseErrorKind::UnexpectedChar => "Unexpected character",
            DecimalParseErrorKind::UnexpectedEnd => "Unexpected end of string",
        })
    }
}

impl Error for ParseDecimalError {}

const PARSE_OVERFLOW: ParseDecimalError = ParseDecimalError {
    kind: DecimalParseErrorKind::Overflow,
};
const PARSE_UNDERFLOW: ParseDecimalError = ParseDecimalError {
    kind: DecimalParseErrorKind::Underflow,
};
const PARSE_UNEXPECTED_CHAR: ParseDecimalError = ParseDecimalError {
    kind: DecimalParseErrorKind::UnexpectedChar,
};
const PARSE_UNEXPECTED_END: ParseDecimalError = ParseDecimalError {
    kind: DecimalParseErrorKind::UnexpectedEnd,
};

impl From<TooLargeForDecimalError> for ParseDecimalError {
    #[inline]
    fn from(_: TooLargeForDecimalError) -> Self {
        Self {
            kind: DecimalParseErrorKind::Overflow,
        }
    }
}

/// The input is too large to fit into a [`Decimal`].
///
/// Matches XPath [`FOCA0001` error](https://www.w3.org/TR/xpath-functions-31/#ERRFOCA0001).
#[derive(Debug, Clone, Copy)]
pub struct TooLargeForDecimalError;

impl fmt::Display for TooLargeForDecimalError {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Value too large for xsd:decimal internal representation")
    }
}

impl Error for TooLargeForDecimalError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new() -> Result<(), ParseDecimalError> {
        assert_eq!(Decimal::new(1, 0)?.to_string(), "1");
        assert_eq!(Decimal::new(1, 1)?.to_string(), "0.1");
        assert_eq!(Decimal::new(10, 0)?.to_string(), "10");
        assert_eq!(Decimal::new(10, 1)?.to_string(), "1");
        assert_eq!(Decimal::new(10, 2)?.to_string(), "0.1");
        Ok(())
    }

    #[test]
    fn from_str() -> Result<(), ParseDecimalError> {
        assert!(Decimal::from_str("").is_err());
        assert!(Decimal::from_str("+").is_err());
        assert!(Decimal::from_str("-").is_err());
        assert!(Decimal::from_str(".").is_err());
        assert!(Decimal::from_str("+.").is_err());
        assert!(Decimal::from_str("-.").is_err());
        assert!(Decimal::from_str("a").is_err());
        assert!(Decimal::from_str(".a").is_err());
        assert_eq!(Decimal::from_str("210")?.to_string(), "210");
        assert_eq!(Decimal::from_str("1000")?.to_string(), "1000");
        assert_eq!(Decimal::from_str("-1.23")?.to_string(), "-1.23");
        assert_eq!(
            Decimal::from_str("12678967.543233")?.to_string(),
            "12678967.543233"
        );
        assert_eq!(Decimal::from_str("+100000.00")?.to_string(), "100000");
        assert_eq!(Decimal::from_str("0.1220")?.to_string(), "0.122");
        assert_eq!(Decimal::from_str(".12200")?.to_string(), "0.122");
        assert_eq!(Decimal::from_str("1.")?.to_string(), "1");
        assert_eq!(Decimal::from_str("1.0")?.to_string(), "1");
        assert_eq!(Decimal::from_str("01.0")?.to_string(), "1");
        assert_eq!(Decimal::from_str("0")?.to_string(), "0");
        assert_eq!(Decimal::from_str("-0")?.to_string(), "0");
        assert_eq!(Decimal::from_str(&Decimal::MAX.to_string())?, Decimal::MAX);
        assert_eq!(Decimal::from_str(&Decimal::MIN.to_string())?, Decimal::MIN);
        assert!(Decimal::from_str("0.0000000000000000001").is_err());
        assert!(Decimal::from_str("1000000000000000000000").is_err());
        assert_eq!(
            Decimal::from_str("0.100000000000000000000000000")?,
            Decimal::from_str("0.1")?
        );
        Ok(())
    }

    #[test]
    fn format() {
        assert_eq!(format!("{}", Decimal::from(0)), "0");
        assert_eq!(format!("{}", Decimal::from(1)), "1");
        assert_eq!(format!("{}", Decimal::from(10)), "10");
        assert_eq!(format!("{}", Decimal::from(100)), "100");
        assert_eq!(format!("{}", Decimal::from(-1)), "-1");
        assert_eq!(format!("{}", Decimal::from(-10)), "-10");

        assert_eq!(format!("{:02}", Decimal::from(0)), "00");
        assert_eq!(format!("{:02}", Decimal::from(1)), "01");
        assert_eq!(format!("{:02}", Decimal::from(10)), "10");
        assert_eq!(format!("{:02}", Decimal::from(100)), "100");
        assert_eq!(format!("{:02}", Decimal::from(-1)), "-1");
        assert_eq!(format!("{:02}", Decimal::from(-10)), "-10");
    }

    /// A width of `0` on a negative value used to underflow the `usize` subtraction that
    /// reserves a character for the sign, and any width large enough to need more integer
    /// digits than the internal digit buffer holds used to slice past its end. Neither may
    /// panic: both are ordinary column-alignment widths a caller might reasonably pick
    /// without knowing this type's internal buffer size.
    #[test]
    fn format_width_does_not_panic() -> Result<(), ParseDecimalError> {
        assert_eq!(format!("{:1$}", Decimal::from_str("-5")?, 0), "-5");
        assert_eq!(format!("{:1$}", Decimal::from(-5), 1), "-5");
        assert_eq!(
            format!("{:1$}", Decimal::from(1), 23),
            format!("{}1", "0".repeat(22))
        );
        assert_eq!(
            format!("{:1$}", Decimal::from(-1), 23),
            format!("-{}1", "0".repeat(21))
        );
        // Even a width far beyond anything this type could ever need must not panic.
        assert_eq!(
            format!("{:1$}", Decimal::from(1), 40),
            format!("{}1", "0".repeat(39))
        );
        Ok(())
    }

    /// An explicit precision rounds the fractional part instead of truncating it, and never
    /// leaves a bare trailing `.` with nothing after it.
    #[test]
    fn format_precision_rounds_instead_of_truncating() -> Result<(), ParseDecimalError> {
        assert_eq!(format!("{:.1}", Decimal::from_str("1.99")?), "2.0");
        assert_eq!(format!("{:.0}", Decimal::from_str("1.99")?), "2");
        assert_eq!(format!("{:.0}", Decimal::from_str("0.4")?), "0");
        assert_eq!(format!("{:.0}", Decimal::from_str("0.9")?), "1");
        // The carry can ripple all the way from the fractional part into the integer part.
        assert_eq!(format!("{:.1}", Decimal::from_str("9.99")?), "10.0");
        assert_eq!(format!("{:.0}", Decimal::from_str("99.5")?), "100");
        // A precision that keeps every significant digit changes nothing.
        assert_eq!(format!("{:.2}", Decimal::from_str("1.99")?), "1.99");
        assert_eq!(format!("{:.5}", Decimal::from_str("1.99")?), "1.99");
        // Negative values round the same way (half away from zero).
        assert_eq!(format!("{:.1}", Decimal::from_str("-1.99")?), "-2.0");
        assert_eq!(format!("{:.0}", Decimal::from_str("-0.4")?), "-0");
        Ok(())
    }

    #[test]
    fn add() {
        assert!(Decimal::MIN.checked_add(Decimal::STEP).is_some());
        assert!(Decimal::MAX.checked_add(Decimal::STEP).is_none());
        assert_eq!(
            Decimal::MAX.checked_add(Decimal::MIN),
            Decimal::STEP.checked_neg()
        );
    }

    #[test]
    fn sub() {
        assert!(Decimal::MIN.checked_sub(Decimal::STEP).is_none());
        assert!(Decimal::MAX.checked_sub(Decimal::STEP).is_some());
    }

    #[test]
    fn mul() -> Result<(), ParseDecimalError> {
        assert_eq!(Decimal::from(1).checked_mul(-1), Some(Decimal::from(-1)));
        assert_eq!(
            Decimal::from(1000).checked_mul(1000),
            Some(Decimal::from(1_000_000))
        );
        assert_eq!(
            Decimal::from_str("0.1")?.checked_mul(Decimal::from_str("0.01")?),
            Some(Decimal::from_str("0.001")?)
        );
        assert_eq!(Decimal::from(0).checked_mul(1), Some(Decimal::from(0)));
        assert_eq!(Decimal::from(1).checked_mul(0), Some(Decimal::from(0)));
        // A fractional operand times zero is zero, not an overflow: zero has no trailing
        // zeros of its own to pay for the shift the algorithm below performs.
        assert_eq!(
            Decimal::from_str("1.5")?.checked_mul(0),
            Some(Decimal::from(0))
        );
        assert_eq!(
            Decimal::from(0).checked_mul(Decimal::from_str("1.5")?),
            Some(Decimal::from(0))
        );
        assert_eq!(Decimal::from(0).checked_mul(0), Some(Decimal::from(0)));
        assert_eq!(Decimal::MAX.checked_mul(1), Some(Decimal::MAX));
        assert_eq!(Decimal::MIN.checked_mul(1), Some(Decimal::MIN));
        assert_eq!(
            Decimal::from(1).checked_mul(Decimal::MAX),
            Some(Decimal::MAX)
        );
        assert_eq!(
            Decimal::from(1).checked_mul(Decimal::MIN),
            Some(Decimal::MIN)
        );
        assert_eq!(
            Decimal::MAX.checked_mul(-1),
            Decimal::MIN.checked_add(Decimal::STEP)
        );
        assert_eq!(Decimal::MIN.checked_mul(-1), None);
        assert_eq!(
            Decimal::MIN
                .checked_add(Decimal::STEP)
                .and_then(|value| value.checked_mul(-1)),
            Some(Decimal::MAX)
        );
        Ok(())
    }

    #[test]
    fn div() -> Result<(), ParseDecimalError> {
        assert_eq!(Decimal::from(1).checked_div(1), Some(Decimal::from(1)));
        assert_eq!(Decimal::from(100).checked_div(10), Some(Decimal::from(10)));
        assert_eq!(
            Decimal::from(10).checked_div(100),
            Some(Decimal::from_str("0.1")?)
        );
        assert_eq!(Decimal::from(1).checked_div(0), None);
        assert_eq!(Decimal::from(0).checked_div(1), Some(Decimal::from(0)));
        // Zero divided by a fractional divisor is zero, for the same reason.
        assert_eq!(
            Decimal::from(0).checked_div(Decimal::from_str("1.5")?),
            Some(Decimal::from(0))
        );
        assert_eq!(Decimal::from(0).checked_div(0), None);
        assert_eq!(Decimal::MAX.checked_div(1), Some(Decimal::MAX));
        assert_eq!(Decimal::MIN.checked_div(1), Some(Decimal::MIN));
        assert_eq!(
            Decimal::MAX.checked_div(-1),
            Decimal::MIN.checked_add(Decimal::STEP)
        );
        assert_eq!(Decimal::MIN.checked_div(-1), None);
        assert_eq!(
            Decimal::MIN
                .checked_add(Decimal::STEP)
                .and_then(|value| value.checked_div(-1)),
            Some(Decimal::MAX)
        );
        Ok(())
    }

    /// The four QT3 `op/numeric-divide` cases whose quotient does not
    /// terminate inside eighteen fractional digits. Each wraps the division in
    /// `fn:round-half-to-even(…, 18)` and asserts the rounded value, so the
    /// last digit is the whole point: truncating towards zero answers
    /// `-0.830993497117024304` for the first, one unit below what the catalog
    /// accepts.
    #[test]
    fn div_rounds_half_to_even_at_the_last_representable_digit() -> Result<(), ParseDecimalError> {
        // op-numeric-divideintg2args-2: exact quotient
        // -0.830993497117024304830993..., tail above the half.
        assert_eq!(
            Decimal::from_str("830993497117024304")?
                .checked_div(Decimal::from_str("-999999999999999999")?),
            Some(Decimal::from_str("-0.830993497117024305")?)
        );
        // op-numeric-dividedec2args-2: exact quotient
        // -0.617375191608514839617375..., which carries into the last digit.
        assert_eq!(
            Decimal::from_str("617375191608514839")?
                .checked_div(Decimal::from_str("-999999999999999999")?),
            Some(Decimal::from_str("-0.61737519160851484")?)
        );
        // op-numeric-dividedec2args-4.
        assert_eq!(
            Decimal::from_str("-999999999999999999")?
                .checked_div(Decimal::from_str("617375191608514839")?),
            Some(Decimal::from_str("-1.619760582531006901")?)
        );
        // op-numeric-dividenpi2args-4.
        assert_eq!(
            Decimal::from_str("-999999999999999999")?
                .checked_div(Decimal::from_str("-475688437271870490")?),
            Some(Decimal::from_str("2.102216328265447024")?)
        );
        Ok(())
    }

    /// The tie itself, which none of the QT3 cases above lands on: a quotient
    /// exactly half a unit in the last place must go to the even neighbour,
    /// not away from zero. `1 div 2·10¹⁸` is 5·10⁻¹⁹, half of the smallest
    /// representable step, and rounds down to zero because zero is even;
    /// `3 div 2·10¹⁸` is 1.5·10⁻¹⁸ and rounds *up* to 2·10⁻¹⁸ for the same
    /// reason. Half-away-from-zero would answer 1·10⁻¹⁸ for the first.
    #[test]
    fn div_breaks_an_exact_tie_towards_the_even_digit() -> Result<(), ParseDecimalError> {
        let two_e18 = Decimal::from_str("2000000000000000000")?;
        assert_eq!(
            Decimal::from(1).checked_div(two_e18),
            Some(Decimal::from(0))
        );
        assert_eq!(
            Decimal::from(3).checked_div(two_e18),
            Some(Decimal::from_str("0.000000000000000002")?)
        );
        // Negative operands round on the magnitude, symmetrically.
        assert_eq!(
            Decimal::from(-1).checked_div(two_e18),
            Some(Decimal::from(0))
        );
        assert_eq!(
            Decimal::from(-3).checked_div(two_e18),
            Some(Decimal::from_str("-0.000000000000000002")?)
        );
        Ok(())
    }

    /// Quotients that terminate well inside eighteen digits are unaffected by
    /// the rounding step, and so are the exact ones at the extremes of the
    /// representation.
    #[test]
    fn div_leaves_exact_quotients_alone() -> Result<(), ParseDecimalError> {
        assert_eq!(
            Decimal::from(1).checked_div(2),
            Some(Decimal::from_str("0.5")?)
        );
        assert_eq!(
            Decimal::from(1).checked_div(8),
            Some(Decimal::from_str("0.125")?)
        );
        assert_eq!(
            Decimal::from(-7).checked_div(2),
            Some(Decimal::from_str("-3.5")?)
        );
        // 1/3 = 0.333... : the tail is below the half, so the last digit stays
        // a 3 rather than becoming a 4.
        assert_eq!(
            Decimal::from(1).checked_div(3),
            Some(Decimal::from_str("0.333333333333333333")?)
        );
        // 2/3 = 0.666... : the tail is above the half and the last digit
        // carries.
        assert_eq!(
            Decimal::from(2).checked_div(3),
            Some(Decimal::from_str("0.666666666666666667")?)
        );
        Ok(())
    }

    #[test]
    fn rem() -> Result<(), ParseDecimalError> {
        assert_eq!(Decimal::from(10).checked_rem(3), Some(Decimal::from(1)));
        assert_eq!(Decimal::from(6).checked_rem(-2), Some(Decimal::from(0)));
        assert_eq!(
            Decimal::from_str("4.5")?.checked_rem(Decimal::from_str("1.2")?),
            Some(Decimal::from_str("0.9")?)
        );
        assert_eq!(Decimal::from(1).checked_rem(0), None);
        assert_eq!(
            Decimal::MAX.checked_rem(1),
            Some(Decimal::from_str("0.687303715884105727")?)
        );
        assert_eq!(
            Decimal::MIN.checked_rem(1),
            Some(Decimal::from_str("-0.687303715884105728")?)
        );
        assert_eq!(
            Decimal::MAX.checked_rem(Decimal::STEP),
            Some(Decimal::default())
        );
        assert_eq!(
            Decimal::MIN.checked_rem(Decimal::STEP),
            Some(Decimal::default())
        );
        assert_eq!(
            Decimal::MAX.checked_rem(Decimal::MAX),
            Some(Decimal::default())
        );
        assert_eq!(
            Decimal::MIN.checked_rem(Decimal::MIN),
            Some(Decimal::default())
        );
        Ok(())
    }

    #[test]
    fn round() -> Result<(), ParseDecimalError> {
        assert_eq!(Decimal::from(10).checked_round(), Some(Decimal::from(10)));
        assert_eq!(Decimal::from(-10).checked_round(), Some(Decimal::from(-10)));
        assert_eq!(
            Decimal::from(i64::MIN).checked_round(),
            Some(Decimal::from(i64::MIN))
        );
        assert_eq!(
            Decimal::from(i64::MAX).checked_round(),
            Some(Decimal::from(i64::MAX))
        );
        assert_eq!(
            Decimal::from_str("2.5")?.checked_round(),
            Some(Decimal::from(3))
        );
        assert_eq!(
            Decimal::from_str("2.4999")?.checked_round(),
            Some(Decimal::from(2))
        );
        assert_eq!(
            Decimal::from_str("-2.5")?.checked_round(),
            Some(Decimal::from(-2))
        );
        assert_eq!(Decimal::MAX.checked_round(), None);
        assert_eq!(
            Decimal::MAX
                .checked_sub(Decimal::from_str("0.5")?)
                .and_then(Decimal::checked_round),
            Some(Decimal::from_str("170141183460469231731")?)
        );
        assert_eq!(Decimal::MIN.checked_round(), None);
        assert_eq!(
            Decimal::MIN
                .checked_add(Decimal::from_str("0.5")?)
                .and_then(Decimal::checked_round),
            Some(Decimal::from_str("-170141183460469231731")?)
        );
        Ok(())
    }

    #[test]
    fn ceil() -> Result<(), ParseDecimalError> {
        assert_eq!(Decimal::from(10).checked_ceil(), Some(Decimal::from(10)));
        assert_eq!(Decimal::from(-10).checked_ceil(), Some(Decimal::from(-10)));
        assert_eq!(
            Decimal::from_str("10.5")?.checked_ceil(),
            Some(Decimal::from(11))
        );
        assert_eq!(
            Decimal::from_str("-10.5")?.checked_ceil(),
            Some(Decimal::from(-10))
        );
        assert_eq!(
            Decimal::from(i64::MIN).checked_ceil(),
            Some(Decimal::from(i64::MIN))
        );
        assert_eq!(
            Decimal::from(i64::MAX).checked_ceil(),
            Some(Decimal::from(i64::MAX))
        );
        assert_eq!(Decimal::MAX.checked_ceil(), None);
        assert_eq!(
            Decimal::MAX
                .checked_sub(Decimal::from(1))
                .and_then(Decimal::checked_ceil),
            Some(Decimal::from_str("170141183460469231731")?)
        );
        assert_eq!(
            Decimal::MIN.checked_ceil(),
            Some(Decimal::from_str("-170141183460469231731")?)
        );
        Ok(())
    }

    #[test]
    fn floor() -> Result<(), ParseDecimalError> {
        assert_eq!(Decimal::from(10).checked_floor(), Some(Decimal::from(10)));
        assert_eq!(Decimal::from(-10).checked_floor(), Some(Decimal::from(-10)));
        assert_eq!(
            Decimal::from_str("10.5")?.checked_floor(),
            Some(Decimal::from(10))
        );
        assert_eq!(
            Decimal::from_str("-10.5")?.checked_floor(),
            Some(Decimal::from(-11))
        );
        assert_eq!(
            Decimal::from(i64::MIN).checked_floor(),
            Some(Decimal::from(i64::MIN))
        );
        assert_eq!(
            Decimal::from(i64::MAX).checked_floor(),
            Some(Decimal::from(i64::MAX))
        );
        assert_eq!(
            Decimal::MAX.checked_floor(),
            Some(Decimal::from_str("170141183460469231731")?)
        );
        assert_eq!(Decimal::MIN.checked_floor(), None);
        assert_eq!(
            Decimal::MIN
                .checked_add(Decimal::from_str("1")?)
                .and_then(Decimal::checked_floor),
            Some(Decimal::from_str("-170141183460469231731")?)
        );
        Ok(())
    }

    #[test]
    fn to_be_bytes() -> Result<(), ParseDecimalError> {
        assert_eq!(
            Decimal::from_be_bytes(Decimal::MIN.to_be_bytes()),
            Decimal::MIN
        );
        assert_eq!(
            Decimal::from_be_bytes(Decimal::MAX.to_be_bytes()),
            Decimal::MAX
        );
        assert_eq!(
            Decimal::from_be_bytes(Decimal::from(i64::MIN).to_be_bytes()),
            Decimal::from(i64::MIN)
        );
        assert_eq!(
            Decimal::from_be_bytes(Decimal::from(i64::MAX).to_be_bytes()),
            Decimal::from(i64::MAX)
        );
        assert_eq!(
            Decimal::from_be_bytes(Decimal::from(0).to_be_bytes()),
            Decimal::from(0)
        );
        assert_eq!(
            Decimal::from_be_bytes(Decimal::from_str("0.01")?.to_be_bytes()),
            Decimal::from_str("0.01")?
        );
        Ok(())
    }

    #[test]
    fn from_bool() {
        assert_eq!(Decimal::from(false), Decimal::from(0_u8));
        assert_eq!(Decimal::from(true), Decimal::from(1_u8));
    }

    #[test]
    fn from_float() -> Result<(), ParseDecimalError> {
        assert_eq!(
            Decimal::try_from(Float::from(0.)).ok(),
            Some(Decimal::from(0))
        );
        assert_eq!(
            Decimal::try_from(Float::from(-0.)).ok(),
            Some(Decimal::from(0))
        );
        assert_eq!(
            Decimal::try_from(Float::from(-123.5)).ok(),
            Some(Decimal::from_str("-123.5")?)
        );
        assert!(Decimal::try_from(Float::from(f32::NAN)).is_err());
        assert!(Decimal::try_from(Float::from(f32::INFINITY)).is_err());
        assert!(Decimal::try_from(Float::from(f32::NEG_INFINITY)).is_err());
        assert!(Decimal::try_from(Float::from(f32::MIN)).is_err());
        assert!(Decimal::try_from(Float::from(f32::MAX)).is_err());
        assert!(
            Decimal::try_from(Float::from(1_672_507_300_000.))?
                .checked_sub(Decimal::from(1_672_507_293_696_i64))
                .and_then(Decimal::checked_abs)
                .expect("no overflow")
                < Decimal::from(1)
        );
        Ok(())
    }

    #[test]
    fn from_double() -> Result<(), ParseDecimalError> {
        assert_eq!(
            Decimal::try_from(Double::from(0.)).ok(),
            Some(Decimal::from(0))
        );
        assert_eq!(
            Decimal::try_from(Double::from(-0.)).ok(),
            Some(Decimal::from(0))
        );
        assert_eq!(
            Decimal::try_from(Double::from(-123.1)).ok(),
            Some(Decimal::from_str("-123.1")?)
        );
        assert!(
            Decimal::try_from(Double::from(1_672_507_302_466.))?
                .checked_sub(Decimal::from(1_672_507_302_466_i64))
                .and_then(Decimal::checked_abs)
                .expect("no overflow")
                < Decimal::from(1)
        );
        assert!(Decimal::try_from(Double::from(f64::NAN)).is_err());
        assert!(Decimal::try_from(Double::from(f64::INFINITY)).is_err());
        assert!(Decimal::try_from(Double::from(f64::NEG_INFINITY)).is_err());
        assert!(Decimal::try_from(Double::from(f64::MIN)).is_err());
        assert!(Decimal::try_from(Double::from(f64::MAX)).is_err());
        Ok(())
    }

    #[test]
    fn to_float() -> Result<(), ParseDecimalError> {
        assert_eq!(Float::from(Decimal::from(0)), Float::from(0.));
        assert_eq!(Float::from(Decimal::from(1)), Float::from(1.));
        assert_eq!(Float::from(Decimal::from(10)), Float::from(10.));
        assert_eq!(Float::from(Decimal::from_str("0.1")?), Float::from(0.1));
        assert!((Float::from(Decimal::MAX) - Float::from(1.701_412e20)).abs() < Float::from(1.));
        assert!((Float::from(Decimal::MIN) - Float::from(-1.701_412e20)).abs() < Float::from(1.));
        Ok(())
    }

    #[test]
    fn to_double() -> Result<(), ParseDecimalError> {
        assert_eq!(Double::from(Decimal::from(0)), Double::from(0.));
        assert_eq!(Double::from(Decimal::from(1)), Double::from(1.));
        assert_eq!(Double::from(Decimal::from(10)), Double::from(10.));
        assert!(
            Double::from(Decimal::from_str("0.1")?) - Double::from(0.1)
                < Double::from(f64::from(f32::EPSILON))
        );
        assert!(
            (Double::from(Decimal::MAX) - Double::from(1.701_411_834_604_692_4e20)).abs()
                < Double::from(1.)
        );
        assert!(
            (Double::from(Decimal::MIN) - Double::from(-1.701_411_834_604_692_4e20)).abs()
                < Double::from(1.)
        );
        Ok(())
    }

    #[test]
    fn minimally_conformant() -> Result<(), ParseDecimalError> {
        // All minimally conforming processors must support decimal values whose absolute value can be
        // expressed as i / 10^k, where i and k are nonnegative integers such that i < 10^16 and k ≤ 16.
        assert_eq!(
            Decimal::from_str("1234567890123456")?.to_string(),
            "1234567890123456"
        );
        assert_eq!(
            Decimal::from_str("-1234567890123456")?.to_string(),
            "-1234567890123456"
        );
        assert_eq!(
            Decimal::from_str("0.1234567890123456")?.to_string(),
            "0.1234567890123456"
        );
        assert_eq!(
            Decimal::from_str("-0.1234567890123456")?.to_string(),
            "-0.1234567890123456"
        );
        Ok(())
    }
}
