//! Arbitrary-precision `xs:integer` and `xs:decimal` for the values that overflow the
//! fixed-width [`oxixml_xsd::Integer`] (an `i64`) and [`oxixml_xsd::Decimal`] (an `i128` scaled
//! by `10^18`).
//!
//! XML Schema requires `xs:integer` to be unbounded and `xs:decimal` to have arbitrary
//! precision; the fixed-width mirrors this crate builds on cannot express either past their
//! range. [`BigInteger`] and [`BigDecimal`] close that gap *additively*: they carry the exact
//! value for the (rare) results and literals that do not fit, while every in-range value keeps
//! its fixed-width payload and its exact former behaviour.
//!
//! Both types reuse [`oxixml_support::bigint::BigUint`] for the underlying magnitude arithmetic,
//! so no new dependency is introduced. That primitive refuses to operate on magnitudes wider than
//! [`BigUint::MAX_BITS`] (16384 bits, about 4932 decimal digits); the checked operations here
//! surface that ceiling as [`None`] rather than a panic, so a caller can map it to the numeric
//! overflow error its specification prescribes.
//!
//! # Canonical form
//!
//! A [`BigInteger`] keeps its magnitude normalized (no value carries a redundant sign on zero).
//! A [`BigDecimal`] keeps the *smallest* scale that represents its value — trailing fractional
//! zeros are removed at construction — so that two equal decimals share one representation and
//! hash alike, and so that a decimal whose value is integral hashes like the equal integer.

use core::cmp::Ordering;
use oxixml_support::bigint::BigUint;
use oxixml_xsd::{Decimal, Integer};
use std::fmt;

/// An arbitrary-precision signed integer: the value space of `xs:integer`, without the `i64`
/// ceiling of [`oxixml_xsd::Integer`].
///
/// The magnitude is a [`BigUint`]; the sign is carried beside it, with the invariant that zero is
/// never negative. Values that fit an `i64` are *not* meant to be held here — the query layer
/// demotes them to the fixed-width payload — so equality and hashing stay consistent across the
/// two representations.
#[derive(Clone, Debug)]
pub struct BigInteger {
    /// `true` only for a strictly negative value; a zero magnitude is always non-negative.
    negative: bool,
    /// The absolute value.
    magnitude: BigUint,
}

impl BigInteger {
    /// The value zero.
    #[must_use]
    pub fn zero() -> Self {
        Self {
            negative: false,
            magnitude: BigUint::zero(),
        }
    }

    /// Builds from a sign flag and a magnitude, restoring the "zero is not negative" invariant.
    fn from_parts(negative: bool, magnitude: BigUint) -> Self {
        if magnitude.is_zero() {
            Self {
                negative: false,
                magnitude,
            }
        } else {
            Self {
                negative,
                magnitude,
            }
        }
    }

    /// Builds from an [`i64`].
    #[must_use]
    pub fn from_i64(value: i64) -> Self {
        Self::from_parts(value < 0, BigUint::from_u64(value.unsigned_abs()))
    }

    /// Builds from an [`i128`].
    #[must_use]
    pub fn from_i128(value: i128) -> Self {
        Self::from_parts(value < 0, BigUint::from_u128(value.unsigned_abs()))
    }

    /// Whether this value is zero.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.magnitude.is_zero()
    }

    /// Whether this value is strictly negative.
    #[must_use]
    pub fn is_negative(&self) -> bool {
        self.negative
    }

    /// The sign as one of `-1`, `0` or `1`.
    fn signum(&self) -> i32 {
        if self.magnitude.is_zero() {
            0
        } else if self.negative {
            -1
        } else {
            1
        }
    }

    /// The magnitude, for the comparison and hashing helpers in the sibling modules.
    pub(crate) fn magnitude(&self) -> &BigUint {
        &self.magnitude
    }

    /// The value as an [`i64`], or [`None`] if it does not fit.
    #[must_use]
    pub fn to_i64(&self) -> Option<i64> {
        let magnitude = biguint_to_u128(&self.magnitude)?;
        if self.negative {
            if magnitude <= (i64::MAX as u128) + 1 {
                Some((magnitude as i128).wrapping_neg() as i64)
            } else {
                None
            }
        } else {
            i64::try_from(magnitude).ok()
        }
    }

    /// The value as an [`i128`], or [`None`] if it does not fit.
    #[must_use]
    pub fn to_i128(&self) -> Option<i128> {
        let magnitude = biguint_to_u128(&self.magnitude)?;
        if self.negative {
            if magnitude <= (i128::MAX as u128) + 1 {
                Some((magnitude as i128).wrapping_neg())
            } else {
                None
            }
        } else {
            i128::try_from(magnitude).ok()
        }
    }

    /// The value as an [`f64`], rounded to the nearest representable double and saturating to an
    /// infinity when it is out of range. This is the conversion XPath's numeric promotion of an
    /// `xs:integer` to `xs:double` performs.
    ///
    /// It goes through the decimal string so that it is *correctly rounded* and agrees, bit for
    /// bit, with the double a numerically equal [`BigDecimal`] produces — a property the map-key
    /// hashing depends on.
    #[must_use]
    pub fn to_f64(&self) -> f64 {
        self.to_string().parse::<f64>().unwrap_or(0.0)
    }

    /// Negates the value.
    #[must_use]
    pub fn neg(&self) -> Self {
        Self::from_parts(!self.negative, self.magnitude.clone())
    }

    /// The absolute value.
    #[must_use]
    pub fn abs(&self) -> Self {
        Self::from_parts(false, self.magnitude.clone())
    }

    /// Whether the value is even.
    fn is_even(&self) -> bool {
        !self.magnitude.bit(0)
    }

    /// The number of significant bits of the magnitude; see [`BigUint::MAX_BITS`].
    fn bit_len(&self) -> usize {
        self.magnitude.bit_len()
    }

    /// Checked addition, [`None`] if the result would be wider than [`BigUint::MAX_BITS`].
    #[must_use]
    pub fn checked_add(&self, other: &BigInteger) -> Option<BigInteger> {
        let result = if self.negative == other.negative {
            Self::from_parts(self.negative, self.magnitude.checked_add(&other.magnitude)?)
        } else {
            match self.magnitude.cmp(&other.magnitude) {
                Ordering::Equal => Self::zero(),
                Ordering::Greater => {
                    Self::from_parts(self.negative, self.magnitude.checked_sub(&other.magnitude)?)
                }
                Ordering::Less => Self::from_parts(
                    other.negative,
                    other.magnitude.checked_sub(&self.magnitude)?,
                ),
            }
        };
        result.within_bounds()
    }

    /// Checked subtraction, [`None`] if the result would be wider than [`BigUint::MAX_BITS`].
    #[must_use]
    pub fn checked_sub(&self, other: &BigInteger) -> Option<BigInteger> {
        self.checked_add(&other.neg())
    }

    /// Checked multiplication, [`None`] if the result would be wider than [`BigUint::MAX_BITS`].
    #[must_use]
    pub fn checked_mul(&self, other: &BigInteger) -> Option<BigInteger> {
        let magnitude = self.magnitude.checked_mul(&other.magnitude)?;
        Self::from_parts(self.negative != other.negative, magnitude).within_bounds()
    }

    /// Truncating division and remainder: `(quotient, remainder)` with the quotient rounded
    /// towards zero and the remainder taking the sign of the dividend, so that
    /// `self == quotient * divisor + remainder`. [`None`] on a zero divisor or an operand wider
    /// than [`BigUint::MAX_BITS`].
    #[must_use]
    pub fn div_rem(&self, divisor: &BigInteger) -> Option<(BigInteger, BigInteger)> {
        let (quotient, remainder) = self.magnitude.div_rem(&divisor.magnitude)?;
        Some((
            Self::from_parts(self.negative != divisor.negative, quotient),
            Self::from_parts(self.negative, remainder),
        ))
    }

    /// Returns `self` if its magnitude is within [`BigUint::MAX_BITS`], else [`None`].
    fn within_bounds(self) -> Option<Self> {
        if self.bit_len() > BigUint::MAX_BITS {
            None
        } else {
            Some(self)
        }
    }
}

impl From<Integer> for BigInteger {
    fn from(value: Integer) -> Self {
        Self::from_i64(i64::from(value))
    }
}

impl PartialEq for BigInteger {
    fn eq(&self, other: &Self) -> bool {
        self.negative == other.negative && self.magnitude == other.magnitude
    }
}

impl Eq for BigInteger {}

impl Ord for BigInteger {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.signum().cmp(&other.signum()) {
            Ordering::Equal => {
                let magnitudes = self.magnitude.cmp(&other.magnitude);
                if self.negative {
                    magnitudes.reverse()
                } else {
                    magnitudes
                }
            }
            unequal => unequal,
        }
    }
}

impl PartialOrd for BigInteger {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Display for BigInteger {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.negative {
            f.write_str("-")?;
        }
        f.write_str(&biguint_to_decimal_string(&self.magnitude))
    }
}

/// Parses a lexical `xs:integer` — an optional sign then one or more ASCII digits — into a
/// [`BigInteger`]. Returns [`None`] for any other shape, or for a magnitude wider than
/// [`BigUint::MAX_BITS`]. Leading and trailing whitespace must already have been trimmed.
#[must_use]
pub fn parse_big_integer(lexical: &str) -> Option<BigInteger> {
    let (negative, digits) = split_sign(lexical)?;
    let magnitude = parse_big_uint(digits)?;
    let value = BigInteger::from_parts(negative, magnitude);
    value.within_bounds()
}

// --- BigUint bridging helpers ---------------------------------------------------------------

/// The largest power of ten that fits a `u64`, used to convert to and from decimal in 18-digit
/// chunks so a formatting pass is linear in the number of limbs rather than the number of digits.
const TEN_POW_18: u64 = 1_000_000_000_000_000_000;

/// A [`BigUint`] of one billion billion, the base of the 18-digit chunking above.
fn chunk_base() -> BigUint {
    BigUint::from_u64(TEN_POW_18)
}

/// The value of `magnitude` as a [`u128`], or [`None`] if it needs more than 128 bits.
fn biguint_to_u128(magnitude: &BigUint) -> Option<u128> {
    if magnitude.bit_len() > 128 {
        return None;
    }
    let bytes = magnitude.to_be_bytes();
    let mut value = 0u128;
    for byte in bytes {
        value = (value << 8) | u128::from(byte);
    }
    Some(value)
}

/// Formats a [`BigUint`] in base ten (`"0"` for zero, no leading zeros otherwise).
fn biguint_to_decimal_string(magnitude: &BigUint) -> String {
    if magnitude.is_zero() {
        return "0".to_owned();
    }
    let base = chunk_base();
    let mut chunks = Vec::new();
    let mut remaining = magnitude.clone();
    while !remaining.is_zero() {
        match remaining.div_rem(&base) {
            Some((quotient, remainder)) => {
                chunks.push(biguint_to_u128(&remainder).unwrap_or(0) as u64);
                remaining = quotient;
            }
            // `div_rem` only fails for a zero divisor or an operand past `MAX_BITS`; the base is
            // non-zero and every value that reaches this module is within bounds.
            None => break,
        }
    }
    let mut out = String::new();
    if let Some((first, rest)) = chunks.split_last() {
        out.push_str(&first.to_string());
        for chunk in rest.iter().rev() {
            out.push_str(&format!("{chunk:018}"));
        }
    }
    out
}

/// Parses a run of ASCII digits into a [`BigUint`], or [`None`] on any non-digit or on a value
/// wider than [`BigUint::MAX_BITS`]. An empty string is rejected.
fn parse_big_uint(digits: &str) -> Option<BigUint> {
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    let base = chunk_base();
    let mut value = BigUint::zero();
    let bytes = digits.as_bytes();
    // The most significant chunk is shorter than 18 digits whenever the length is not a multiple
    // of 18; consume it first, then whole 18-digit chunks.
    let head = bytes.len() % 18;
    let mut index = 0;
    if head > 0 {
        value = BigUint::from_u64(parse_digit_chunk(&bytes[..head])?);
        index = head;
    }
    while index < bytes.len() {
        let chunk = parse_digit_chunk(&bytes[index..index + 18])?;
        value = value
            .checked_mul(&base)?
            .checked_add(&BigUint::from_u64(chunk))?;
        if value.bit_len() > BigUint::MAX_BITS {
            return None;
        }
        index += 18;
    }
    Some(value)
}

/// Parses up to eighteen ASCII digits into the `u64` they spell.
fn parse_digit_chunk(bytes: &[u8]) -> Option<u64> {
    let mut value = 0u64;
    for &byte in bytes {
        let digit = (byte as char).to_digit(10)?;
        value = value * 10 + u64::from(digit);
    }
    Some(value)
}

/// Splits an optional leading `+`/`-` sign from the digits, returning `(negative, digits)`.
fn split_sign(lexical: &str) -> Option<(bool, &str)> {
    match lexical.as_bytes().first() {
        Some(b'-') => Some((true, &lexical[1..])),
        Some(b'+') => Some((false, &lexical[1..])),
        Some(_) => Some((false, lexical)),
        None => None,
    }
}

/// Raises `base` to the `exponent`, or [`None`] if the result would exceed [`BigUint::MAX_BITS`].
fn pow_big_uint(base: &BigUint, mut exponent: u32) -> Option<BigUint> {
    let mut result = BigUint::one();
    let mut factor = base.clone();
    while exponent > 0 {
        if exponent & 1 == 1 {
            result = result.checked_mul(&factor)?;
            if result.bit_len() > BigUint::MAX_BITS {
                return None;
            }
        }
        exponent >>= 1;
        if exponent > 0 {
            factor = factor.checked_mul(&factor)?;
            if factor.bit_len() > BigUint::MAX_BITS {
                return None;
            }
        }
    }
    Some(result)
}

/// Ten raised to `exponent`, as a non-negative [`BigInteger`].
fn pow_ten(exponent: u32) -> Option<BigInteger> {
    Some(BigInteger::from_parts(
        false,
        pow_big_uint(&BigUint::from_u64(10), exponent)?,
    ))
}

/// An arbitrary-precision decimal: the value space of `xs:decimal`, without the 18-fractional-
/// digit / `i128` ceiling of [`oxixml_xsd::Decimal`].
///
/// The value is `unscaled × 10^(-scale)`. The representation is kept canonical: the scale is the
/// smallest that spells the value, so trailing fractional zeros never survive construction, an
/// integral value has scale `0`, and two equal decimals are bit-for-bit the same.
#[derive(Clone, Debug)]
pub struct BigDecimal {
    /// The significand, sign included.
    unscaled: BigInteger,
    /// The number of fractional digits; the value is `unscaled × 10^(-scale)`.
    scale: u32,
}

impl BigDecimal {
    /// Builds from an unscaled significand and a scale, then canonicalizes (drops trailing
    /// fractional zeros).
    fn from_parts(unscaled: BigInteger, scale: u32) -> Self {
        let mut value = Self { unscaled, scale };
        value.normalize();
        value
    }

    /// Removes trailing factors of ten from the significand, lowering the scale to match, so the
    /// representation is canonical.
    fn normalize(&mut self) {
        if self.unscaled.is_zero() {
            self.scale = 0;
            return;
        }
        let ten = BigInteger::from_i64(10);
        while self.scale > 0 {
            let Some((quotient, remainder)) = self.unscaled.div_rem(&ten) else {
                break;
            };
            if !remainder.is_zero() {
                break;
            }
            self.unscaled = quotient;
            self.scale -= 1;
        }
    }

    /// A decimal whose value is the integer `value`.
    #[must_use]
    pub fn from_big_integer(value: BigInteger) -> Self {
        Self::from_parts(value, 0)
    }

    /// The exact value of a fixed-width [`oxixml_xsd::Decimal`], which stores its significand
    /// scaled by `10^18`.
    #[must_use]
    pub fn from_fixed(value: Decimal) -> Self {
        let raw = i128::from_be_bytes(value.to_be_bytes());
        Self::from_parts(BigInteger::from_i128(raw), 18)
    }

    /// The exact value of a finite [`f64`] as a decimal, or [`None`] for an infinity or `NaN`.
    ///
    /// A double is `mantissa × 2^exponent`. With a non-negative exponent that is an integer; with
    /// a negative one it is `mantissa × 5^|exponent| × 10^(-|exponent|)`, since `2^-e = 5^e ×
    /// 10^-e`. Either way the result is exact, which is what the map-key relation needs to tell
    /// `xs:double 2^64` apart from an `xs:integer` that merely rounds to it.
    #[must_use]
    pub fn from_f64_exact(value: f64) -> Option<Self> {
        if !value.is_finite() {
            return None;
        }
        if value == 0.0 {
            return Some(Self::from_parts(BigInteger::zero(), 0));
        }
        let bits = value.to_bits();
        let negative = bits >> 63 == 1;
        let raw_exponent = ((bits >> 52) & 0x7ff) as i64;
        let raw_mantissa = bits & 0x000f_ffff_ffff_ffff;
        let (mantissa, exponent) = if raw_exponent == 0 {
            (raw_mantissa, -1074)
        } else {
            (raw_mantissa | 0x0010_0000_0000_0000, raw_exponent - 1075)
        };
        let significand = BigInteger::from_parts(negative, BigUint::from_u64(mantissa));
        if exponent >= 0 {
            let factor = BigInteger::from_parts(false, pow_two(exponent as u32)?);
            Some(Self::from_parts(significand.checked_mul(&factor)?, 0))
        } else {
            let magnitude = (-exponent) as u32;
            let factor = BigInteger::from_parts(false, pow_five(magnitude)?);
            Some(Self::from_parts(
                significand.checked_mul(&factor)?,
                magnitude,
            ))
        }
    }

    /// This value as a fixed-width [`oxixml_xsd::Decimal`], or [`None`] if it does not fit (its
    /// scale exceeds 18, or its significand overflows the `i128`).
    #[must_use]
    pub fn to_fixed(&self) -> Option<Decimal> {
        if self.scale > 18 {
            return None;
        }
        let unscaled = self.unscaled.to_i128()?;
        Decimal::new(unscaled, self.scale).ok()
    }

    /// Whether this value is zero.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.unscaled.is_zero()
    }

    /// Whether this value is strictly negative.
    #[must_use]
    pub fn is_negative(&self) -> bool {
        self.unscaled.is_negative()
    }

    /// The significand, for the comparison and hashing helpers in the sibling modules.
    pub(crate) fn unscaled(&self) -> &BigInteger {
        &self.unscaled
    }

    /// The scale (number of fractional digits), for the map-key hashing helper.
    pub(crate) fn scale(&self) -> u32 {
        self.scale
    }

    /// The value as an [`f64`], correctly rounded (through the decimal string, so it agrees with
    /// [`BigInteger::to_f64`] on integral values).
    #[must_use]
    pub fn to_f64(&self) -> f64 {
        self.to_string().parse::<f64>().unwrap_or(0.0)
    }

    /// Negates the value.
    #[must_use]
    pub fn neg(&self) -> Self {
        Self {
            unscaled: self.unscaled.neg(),
            scale: self.scale,
        }
    }
}

/// Two raised to `exponent`, as a non-negative [`BigUint`].
fn pow_two(exponent: u32) -> Option<BigUint> {
    pow_big_uint(&BigUint::from_u64(2), exponent)
}

/// Five raised to `exponent`, as a non-negative [`BigUint`].
fn pow_five(exponent: u32) -> Option<BigUint> {
    pow_big_uint(&BigUint::from_u64(5), exponent)
}

impl std::hash::Hash for BigInteger {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.negative.hash(state);
        self.magnitude.hash(state);
    }
}

impl BigDecimal {
    /// The significand this value would have at `target` fractional digits (`target` must be at
    /// least the current scale); `None` only if the widened significand exceeds
    /// [`BigUint::MAX_BITS`].
    fn rescaled(&self, target: u32) -> Option<BigInteger> {
        let extra = target.checked_sub(self.scale)?;
        if extra == 0 {
            return Some(self.unscaled.clone());
        }
        self.unscaled.checked_mul(&pow_ten(extra)?)
    }

    /// Returns `self` if its significand is within [`BigUint::MAX_BITS`], else [`None`].
    fn within_bounds(self) -> Option<Self> {
        self.unscaled.clone().within_bounds().map(|_| self)
    }

    /// Checked addition, canonicalized; [`None`] past [`BigUint::MAX_BITS`].
    #[must_use]
    pub fn checked_add(&self, other: &BigDecimal) -> Option<BigDecimal> {
        let scale = self.scale.max(other.scale);
        let sum = self.rescaled(scale)?.checked_add(&other.rescaled(scale)?)?;
        Self::from_parts(sum, scale).within_bounds()
    }

    /// Checked subtraction, canonicalized; [`None`] past [`BigUint::MAX_BITS`].
    #[must_use]
    pub fn checked_sub(&self, other: &BigDecimal) -> Option<BigDecimal> {
        self.checked_add(&other.neg())
    }

    /// Checked multiplication, canonicalized; [`None`] past [`BigUint::MAX_BITS`].
    #[must_use]
    pub fn checked_mul(&self, other: &BigDecimal) -> Option<BigDecimal> {
        let scale = self.scale.checked_add(other.scale)?;
        let product = self.unscaled.checked_mul(&other.unscaled)?;
        Self::from_parts(product, scale).within_bounds()
    }

    /// The number of fractional digits a division result is computed to: enough to keep the
    /// operands' own precision, and never fewer than the fixed-width decimal's eighteen.
    fn division_scale(&self, other: &BigDecimal) -> u32 {
        self.scale.max(other.scale).max(18)
    }

    /// Checked division, truncating toward zero at enough fractional digits to keep the operands'
    /// own precision (and never fewer than the fixed-width decimal's eighteen), matching that
    /// type's truncating division. [`None`] on a zero divisor or past [`BigUint::MAX_BITS`].
    #[must_use]
    pub fn checked_div(&self, other: &BigDecimal) -> Option<BigDecimal> {
        if other.is_zero() {
            return None;
        }
        let result_scale = self.division_scale(other);
        // value × 10^result_scale = (ua/ub) × 10^(sb - sa + result_scale); compute that quotient
        // as an integer, truncating, then read it back at `result_scale`.
        let shift = i64::from(result_scale) + i64::from(other.scale) - i64::from(self.scale);
        let (numerator, denominator) = if shift >= 0 {
            (
                self.unscaled
                    .checked_mul(&pow_ten(u32::try_from(shift).ok()?)?)?,
                other.unscaled.clone(),
            )
        } else {
            (
                self.unscaled.clone(),
                other
                    .unscaled
                    .checked_mul(&pow_ten(u32::try_from(-shift).ok()?)?)?,
            )
        };
        let (quotient, _) = numerator.div_rem(&denominator)?;
        Self::from_parts(quotient, result_scale).within_bounds()
    }

    /// The integer quotient `truncate(self / other)`, as XPath's `idiv` needs it; [`None`] on a
    /// zero divisor or past [`BigUint::MAX_BITS`].
    #[must_use]
    pub fn checked_integer_divide(&self, other: &BigDecimal) -> Option<BigInteger> {
        if other.is_zero() {
            return None;
        }
        let scale = self.scale.max(other.scale);
        let (quotient, _) = self.rescaled(scale)?.div_rem(&other.rescaled(scale)?)?;
        Some(quotient)
    }

    /// Checked remainder, the decimal `self - truncate(self / other) × other` XPath's `mod` uses;
    /// [`None`] on a zero divisor or past [`BigUint::MAX_BITS`].
    #[must_use]
    pub fn checked_rem(&self, other: &BigDecimal) -> Option<BigDecimal> {
        let quotient = BigDecimal::from_big_integer(self.checked_integer_divide(other)?);
        self.checked_sub(&quotient.checked_mul(other)?)
    }

    /// The integer part of the value, truncated toward zero — the `xs:decimal` → `xs:integer`
    /// cast. [`None`] only past [`BigUint::MAX_BITS`].
    #[must_use]
    pub fn truncate(&self) -> Option<BigInteger> {
        if self.scale == 0 {
            return Some(self.unscaled.clone());
        }
        let divisor = pow_ten(self.scale)?;
        Some(self.unscaled.div_rem(&divisor)?.0)
    }

    /// The absolute value.
    #[must_use]
    pub fn abs(&self) -> Self {
        Self {
            unscaled: self.unscaled.abs(),
            scale: self.scale,
        }
    }

    /// The largest integer that is not greater than the value (`fn:floor`), as a decimal.
    #[must_use]
    pub fn floor(&self) -> Option<Self> {
        let truncated = self.truncate()?;
        let adjusted = if self.is_negative() && self.scale > 0 {
            truncated.checked_sub(&BigInteger::from_i64(1))?
        } else {
            truncated
        };
        Some(Self::from_big_integer(adjusted))
    }

    /// The smallest integer that is not less than the value (`fn:ceiling`), as a decimal.
    #[must_use]
    pub fn ceiling(&self) -> Option<Self> {
        let truncated = self.truncate()?;
        let adjusted = if !self.is_negative() && !self.is_zero() && self.scale > 0 {
            truncated.checked_add(&BigInteger::from_i64(1))?
        } else {
            truncated
        };
        Some(Self::from_big_integer(adjusted))
    }

    /// Rounds to `precision` fractional digits. `half_to_even` selects the tie rule: the even
    /// neighbour when true (`fn:round-half-to-even`), otherwise towards positive infinity
    /// (`fn:round`). A negative `precision` rounds to a multiple of `10^(-precision)`.
    #[must_use]
    pub fn round(&self, precision: i32, half_to_even: bool) -> Option<Self> {
        // Nothing to do when the value has no digit at or beyond the rounding position.
        if precision >= 0 && self.scale <= precision.unsigned_abs() {
            return Some(self.clone());
        }
        // Shift the digit being rounded into the units place, round the significand to an integer,
        // then shift back.
        let shifted_scale = u32::try_from(i64::from(self.scale) - i64::from(precision)).ok()?;
        let rounded = round_significand(&self.unscaled, shifted_scale, half_to_even)?;
        if precision >= 0 {
            Some(Self::from_parts(rounded, precision.unsigned_abs()))
        } else {
            let factor = pow_ten(precision.unsigned_abs())?;
            Some(Self::from_big_integer(rounded.checked_mul(&factor)?))
        }
    }

    /// Aligns the magnitudes of two values to a common scale and compares them; the sign is dealt
    /// with by the caller.
    fn cmp_magnitude(&self, other: &BigDecimal) -> Ordering {
        let scale = self.scale.max(other.scale);
        let left = self
            .unscaled
            .magnitude()
            .checked_mul(&ten_pow_or_zero(scale - self.scale));
        let right = other
            .unscaled
            .magnitude()
            .checked_mul(&ten_pow_or_zero(scale - other.scale));
        match (left, right) {
            (Some(left), Some(right)) => left.cmp(&right),
            // The multiplications above cannot fail (`checked_mul` on `BigUint` never does), so
            // this arm is unreachable; treat an impossible pair as equal rather than panicking.
            _ => Ordering::Equal,
        }
    }
}

/// Ten raised to `exponent` as a [`BigUint`], falling back to one on the (unreachable past
/// `MAX_BITS`) overflow so the comparison helper stays total.
fn ten_pow_or_zero(exponent: u32) -> BigUint {
    pow_big_uint(&BigUint::from_u64(10), exponent).unwrap_or_else(BigUint::one)
}

/// Rounds `unscaled × 10^(-scale)` to the nearest integer, resolving a tie by the even neighbour
/// when `half_to_even`, otherwise towards positive infinity. Truncation is towards zero, so the
/// magnitude is only ever rounded *away* from zero.
fn round_significand(unscaled: &BigInteger, scale: u32, half_to_even: bool) -> Option<BigInteger> {
    if scale == 0 {
        return Some(unscaled.clone());
    }
    let divisor = pow_ten(scale)?;
    let (quotient, remainder) = unscaled.div_rem(&divisor)?;
    if remainder.is_zero() {
        return Some(quotient);
    }
    let twice_remainder = remainder.abs().checked_mul(&BigInteger::from_i64(2))?;
    let away = if unscaled.is_negative() {
        BigInteger::from_i64(-1)
    } else {
        BigInteger::from_i64(1)
    };
    match twice_remainder.cmp(&divisor) {
        Ordering::Less => Some(quotient),
        Ordering::Greater => quotient.checked_add(&away),
        Ordering::Equal => {
            let round_up = if half_to_even {
                !quotient.is_even()
            } else {
                // `fn:round` breaks a tie towards positive infinity, which raises the magnitude of
                // a positive value and lowers (keeps) that of a negative one.
                !unscaled.is_negative()
            };
            if round_up {
                quotient.checked_add(&away)
            } else {
                Some(quotient)
            }
        }
    }
}

impl PartialEq for BigDecimal {
    fn eq(&self, other: &Self) -> bool {
        // The representation is canonical, so numerically equal values are structurally equal.
        self.scale == other.scale && self.unscaled == other.unscaled
    }
}

impl Eq for BigDecimal {}

impl std::hash::Hash for BigDecimal {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.unscaled.hash(state);
        self.scale.hash(state);
    }
}

impl Ord for BigDecimal {
    fn cmp(&self, other: &Self) -> Ordering {
        let signs = decimal_signum(self).cmp(&decimal_signum(other));
        if signs != Ordering::Equal {
            return signs;
        }
        let magnitudes = self.cmp_magnitude(other);
        if self.is_negative() {
            magnitudes.reverse()
        } else {
            magnitudes
        }
    }
}

impl PartialOrd for BigDecimal {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// The sign of a decimal as `-1`, `0` or `1`, for ordering.
fn decimal_signum(value: &BigDecimal) -> i32 {
    if value.is_zero() {
        0
    } else if value.is_negative() {
        -1
    } else {
        1
    }
}

impl fmt::Display for BigDecimal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.unscaled.is_negative() {
            f.write_str("-")?;
        }
        let digits = biguint_to_decimal_string(self.unscaled.magnitude());
        if self.scale == 0 {
            return f.write_str(&digits);
        }
        let scale = self.scale as usize;
        if digits.len() > scale {
            let point = digits.len() - scale;
            write!(f, "{}.{}", &digits[..point], &digits[point..])
        } else {
            write!(f, "0.{}{}", "0".repeat(scale - digits.len()), digits)
        }
    }
}

/// Parses a lexical `xs:decimal` — an optional sign, then digits with at most one decimal point
/// and at least one digit overall — into a [`BigDecimal`]. Returns [`None`] for any other shape,
/// or for a value wider than [`BigUint::MAX_BITS`]. Whitespace must already have been trimmed.
#[must_use]
pub fn parse_big_decimal(lexical: &str) -> Option<BigDecimal> {
    let (negative, rest) = split_sign(lexical)?;
    let (integer_part, fraction_part) = match rest.split_once('.') {
        Some((integer, fraction)) => (integer, fraction),
        None => (rest, ""),
    };
    if integer_part.is_empty() && fraction_part.is_empty() {
        return None;
    }
    let mut digits = String::with_capacity(integer_part.len() + fraction_part.len());
    digits.push_str(integer_part);
    digits.push_str(fraction_part);
    let magnitude = parse_big_uint(&digits)?;
    let scale = u32::try_from(fraction_part.len()).ok()?;
    let value = BigDecimal::from_parts(BigInteger::from_parts(negative, magnitude), scale);
    value.within_bounds()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integer_string_round_trips_a_hundred_digits() {
        let text = "1".to_owned() + &"0".repeat(99);
        let value = parse_big_integer(&text).expect("parses");
        assert_eq!(value.to_string(), text);
        let negative = format!("-{text}");
        assert_eq!(
            parse_big_integer(&negative).expect("parses").to_string(),
            negative
        );
    }

    #[test]
    fn integer_factorial_beyond_i64_is_exact() {
        // 25! = 15511210043330985984000000, well past i64::MAX.
        let mut product = BigInteger::from_i64(1);
        for n in 1..=25 {
            product = product
                .checked_mul(&BigInteger::from_i64(n))
                .expect("in bounds");
        }
        assert_eq!(product.to_string(), "15511210043330985984000000");
        assert_eq!(product.to_i64(), None, "does not fit i64");
    }

    #[test]
    fn integer_add_sub_cross_zero() {
        let a = parse_big_integer("10000000000000000000000").expect("valid");
        let b = parse_big_integer("-9999999999999999999999").expect("valid");
        assert_eq!(a.checked_add(&b).expect("valid").to_string(), "1");
        assert_eq!(a.checked_sub(&a).expect("valid").to_string(), "0");
        assert!(!a.checked_sub(&a).expect("valid").is_negative());
    }

    #[test]
    fn integer_div_rem_truncates_toward_zero() {
        let a = BigInteger::from_i64(-7);
        let b = BigInteger::from_i64(2);
        let (q, r) = a.div_rem(&b).expect("valid");
        assert_eq!(q.to_string(), "-3");
        assert_eq!(r.to_string(), "-1");
    }

    #[test]
    fn integer_ordering() {
        let big = parse_big_integer("100000000000000000000").expect("valid");
        let neg = parse_big_integer("-100000000000000000000").expect("valid");
        assert!(neg < BigInteger::from_i64(0));
        assert!(BigInteger::from_i64(0) < big);
        assert!(neg < big);
    }

    #[test]
    fn integer_demotes_at_the_i64_boundary() {
        assert_eq!(BigInteger::from_i64(i64::MAX).to_i64(), Some(i64::MAX));
        assert_eq!(BigInteger::from_i64(i64::MIN).to_i64(), Some(i64::MIN));
        let over = BigInteger::from_i64(i64::MAX)
            .checked_add(&BigInteger::from_i64(1))
            .expect("valid");
        assert_eq!(over.to_i64(), None);
        assert_eq!(over.to_string(), "9223372036854775808");
    }

    #[test]
    fn decimal_string_round_trips_high_precision() {
        for text in [
            "123456789012345678901234567890.1234567890123456789",
            "-0.0000000000000000000000000000001",
            "79228162514264337593543950336",
        ] {
            assert_eq!(parse_big_decimal(text).expect("parses").to_string(), text);
        }
    }

    #[test]
    fn decimal_normalizes_trailing_zeros() {
        assert_eq!(parse_big_decimal("1.50").expect("valid").to_string(), "1.5");
        assert_eq!(parse_big_decimal("10.0").expect("valid").to_string(), "10");
        assert_eq!(parse_big_decimal("-0.000").expect("valid").to_string(), "0");
        // An integral decimal and the equal integer share one representation.
        let one_e30 = "1".to_owned() + &"0".repeat(30);
        assert_eq!(
            parse_big_decimal(&format!("{one_e30}.0")).expect("valid"),
            BigDecimal::from_big_integer(parse_big_integer(&one_e30).expect("valid"))
        );
    }

    #[test]
    fn decimal_sum_needs_more_than_eighteen_fractional_digits() {
        let a = parse_big_decimal("0.0000000000000000001").expect("valid");
        let sum = a.checked_add(&a).expect("valid");
        assert_eq!(sum.to_string(), "0.0000000000000000002");
    }

    #[test]
    fn decimal_division_is_exact_when_it_can_be() {
        let a = parse_big_decimal("10000000000000000000000000000.0").expect("valid");
        let tenth = parse_big_decimal("0.1").expect("valid");
        assert_eq!(
            a.checked_div(&tenth).expect("valid").to_string(),
            "100000000000000000000000000000"
        );
    }

    #[test]
    fn decimal_fixed_round_trip_and_demote() {
        let value = Decimal::from(12345_i64);
        let big = BigDecimal::from_fixed(value);
        assert_eq!(big.to_fixed(), Some(value));
        // A value with more than 18 fractional digits cannot demote.
        let precise = parse_big_decimal("0.0000000000000000001").expect("valid");
        assert_eq!(precise.to_fixed(), None);
    }

    #[test]
    fn decimal_abs_floor_ceiling_and_round() {
        let d = |text: &str| parse_big_decimal(text).expect("valid");
        // These operate past the fixed width, so they exercise the arbitrary-precision path.
        let big = d("123456789012345678901234567890.75");
        assert_eq!(big.abs().to_string(), "123456789012345678901234567890.75");
        assert_eq!(
            big.neg().abs().to_string(),
            "123456789012345678901234567890.75"
        );
        assert_eq!(
            big.floor().expect("in bounds").to_string(),
            "123456789012345678901234567890"
        );
        assert_eq!(
            big.ceiling().expect("in bounds").to_string(),
            "123456789012345678901234567891"
        );
        // Negative floor/ceiling round the right way.
        let neg = d("-123456789012345678901234567890.75");
        assert_eq!(
            neg.floor().expect("in bounds").to_string(),
            "-123456789012345678901234567891"
        );
        assert_eq!(
            neg.ceiling().expect("in bounds").to_string(),
            "-123456789012345678901234567890"
        );
        // fn:round breaks a tie towards positive infinity; round-half-to-even to the even side.
        let tie = d("100000000000000000000.5");
        assert_eq!(
            tie.round(0, false).expect("in bounds").to_string(),
            "100000000000000000001"
        );
        assert_eq!(
            tie.round(0, true).expect("in bounds").to_string(),
            "100000000000000000000"
        );
        assert_eq!(
            d("100000000000000000001.5")
                .round(0, true)
                .expect("in bounds")
                .to_string(),
            "100000000000000000002"
        );
    }

    #[test]
    fn f64_converts_to_its_exact_decimal_value() {
        // 0.5 is exactly a half; 0.1 is the long dyadic value the double actually holds.
        assert_eq!(
            BigDecimal::from_f64_exact(0.5).expect("valid").to_string(),
            "0.5"
        );
        assert_eq!(
            BigDecimal::from_f64_exact(0.1).expect("valid").to_string(),
            "0.1000000000000000055511151231257827021181583404541015625"
        );
        // 2^64 is an exact integer as a double.
        assert_eq!(
            BigDecimal::from_f64_exact(18446744073709551616.0)
                .expect("valid")
                .to_string(),
            "18446744073709551616"
        );
        assert_eq!(BigDecimal::from_f64_exact(f64::INFINITY), None);
        assert_eq!(BigDecimal::from_f64_exact(f64::NAN), None);
    }

    #[test]
    fn arithmetic_past_max_bits_returns_none_rather_than_panicking() {
        // 2^16383 sits at exactly BigUint::MAX_BITS (16384 bits); any op that widens it crosses
        // the ceiling and must report `None`, never panic or wrap.
        let two = BigInteger::from_i64(2);
        let mut at_ceiling = BigInteger::from_i64(1);
        for _ in 0..16383 {
            at_ceiling = at_ceiling
                .checked_mul(&two)
                .expect("2^16383 is within MAX_BITS");
        }
        assert_eq!(at_ceiling.checked_add(&at_ceiling), None);
        assert_eq!(at_ceiling.checked_mul(&two), None);
        assert_eq!(at_ceiling.checked_mul(&at_ceiling), None);
    }

    #[test]
    fn big_integer_and_big_decimal_agree_on_the_same_double() {
        // A power of two large enough to escape i64 but exact as a double.
        let integer = parse_big_integer("18446744073709551616").expect("valid");
        assert_eq!(integer.to_f64(), 18446744073709551616.0_f64);
        assert_eq!(
            integer.to_f64().to_bits(),
            BigDecimal::from_big_integer(integer).to_f64().to_bits()
        );
    }
}
