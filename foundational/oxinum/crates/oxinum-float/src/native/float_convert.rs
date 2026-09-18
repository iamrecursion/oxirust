//! Conversions to and from primitive numeric types.
//!
//! Provides:
//!
//! - [`BigFloat::from_i64`] — exact-then-rounded integer encoding.
//! - [`BigFloat::from_f64`] — exact decomposition of an IEEE-754 `f64` value.
//! - [`BigFloat::to_f64`] — IEEE-754-correct, round-to-nearest-even encoding.
//! - [`BigFloat::from_bigint`] — convert a [`BigInt`] to `BigFloat` at specified precision.
//! - [`BigFloat::from_biguint`] — convert a [`BigUint`] to `BigFloat` at specified precision.

use oxinum_core::{OxiNumError, OxiNumResult, Sign};
use oxinum_int::native::{BigInt, BigUint};

use super::float::{BigFloat, FloatClass, RoundDirection, RoundingMode};

/// Bit budget for an *exact* `BigFloat` → integer / rational conversion.
///
/// A `BigFloat` carries a free `i64` exponent — [`BigFloat::from_parts`]
/// imposes no EMAX and [`BigFloat::from_hex_float`] parses the `p` exponent
/// straight out of an untrusted string — so `2^(2^40)` is an ordinary,
/// perfectly valid value. Its *exact* integer form, however, needs ~137 GB.
/// Without a budget the conversion requests that allocation and the process
/// dies on an allocation failure, which is an abort rather than a catchable
/// panic.
///
/// `2^27` bits is a 16 MiB integer: orders of magnitude beyond any legitimate
/// exact conversion, yet small enough that a 15-character input string cannot
/// amplify into a memory-exhaustion attack. The infallible conversions panic
/// above this budget (documented under each `# Panics` section); the `try_*`
/// variants return
/// [`OxiNumError::Overflow`](oxinum_core::OxiNumError::Overflow) instead.
pub const MAX_EXACT_CONVERSION_BITS: u64 = 1 << 27;

/// Build the shared over-budget error for exact conversions.
pub(crate) fn exact_conversion_overflow(what: &str, bits: u64) -> OxiNumError {
    OxiNumError::Overflow(
        format!(
            "{what}: exact result needs {bits} bits, above the \
             MAX_EXACT_CONVERSION_BITS budget of {MAX_EXACT_CONVERSION_BITS}"
        )
        .into(),
    )
}

// ---------------------------------------------------------------------------
// BigFloat → BigInt conversions
// ---------------------------------------------------------------------------

impl BigFloat {
    /// Returns `true` if this value has a non-zero fractional part
    /// (i.e., is not an exact integer).
    ///
    /// Zero and values with `exponent >= 0` are always exact integers.
    fn has_fractional_part(&self) -> bool {
        if self.is_zero() || self.exponent >= 0 {
            return false;
        }
        // `unsigned_abs`, not `-self.exponent`: the exponent can legitimately
        // reach `i64::MIN` because `round_to_precision_in_place` lowers it with
        // `saturating_sub`, and negating `i64::MIN` overflows.
        let shift = self.exponent.unsigned_abs();
        // fractional bits live in the low `shift` bits of the mantissa.
        // Any of those bits being set means there is a fractional part.
        // Equivalent: trailing_zeros < shift.
        self.mantissa.trailing_zeros() < shift
    }

    /// Returns `true` if the fractional part is exactly 1/2
    /// (the half-bit is set and all lower bits are zero).
    fn half_exactly(&self) -> bool {
        if self.is_zero() || self.exponent >= 0 {
            return false;
        }
        // `unsigned_abs`, not `-self.exponent`: the exponent can legitimately
        // reach `i64::MIN` because `round_to_precision_in_place` lowers it with
        // `saturating_sub`, and negating `i64::MIN` overflows.
        let shift = self.exponent.unsigned_abs();
        // The half-bit is at position (shift - 1).
        // Exactly 1/2: bit (shift-1) == 1 AND all bits below it are 0.
        // i.e. trailing_zeros == shift - 1.
        self.mantissa.test_bit(shift - 1) && self.mantissa.trailing_zeros() == shift - 1
    }

    /// Returns `true` if the fractional part is strictly greater than 1/2.
    fn more_than_half(&self) -> bool {
        if self.is_zero() || self.exponent >= 0 {
            return false;
        }
        // `unsigned_abs`, not `-self.exponent`: the exponent can legitimately
        // reach `i64::MIN` because `round_to_precision_in_place` lowers it with
        // `saturating_sub`, and negating `i64::MIN` overflows.
        let shift = self.exponent.unsigned_abs();
        // The half-bit is at position (shift - 1).
        // More than 1/2: bit (shift-1) == 1 AND at least one lower bit is 1.
        // i.e. trailing_zeros < shift - 1.
        self.mantissa.test_bit(shift - 1) && self.mantissa.trailing_zeros() < shift - 1
    }

    /// Convert to [`BigInt`] by truncating toward zero (round-toward-zero).
    ///
    /// Returns the integer part of `self`, discarding any fractional bits.
    ///
    /// Non-finite values (NaN, ±Inf) have no integer representation.
    /// This method returns `BigInt::zero()` as a documented lossy fallback
    /// for those inputs; callers that may encounter non-finite values should
    /// use [`BigFloat::is_finite`] to guard before calling.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxinum_float::native::BigFloat;
    /// use oxinum_int::native::BigInt;
    ///
    /// let x = BigFloat::from_f64(3.7, 64).expect("3.7");
    /// assert_eq!(x.to_bigint_trunc(), BigInt::from(3i64));
    ///
    /// let y = BigFloat::from_f64(-3.7, 64).expect("-3.7");
    /// assert_eq!(y.to_bigint_trunc(), BigInt::from(-3i64));
    ///
    /// // Non-finite values return zero (lossy).
    /// assert_eq!(BigFloat::nan(53).to_bigint_trunc(), BigInt::zero());
    /// assert_eq!(BigFloat::infinity(53).to_bigint_trunc(), BigInt::zero());
    /// ```
    ///
    /// # Panics
    ///
    /// Panics when the exact integer would exceed
    /// [`MAX_EXACT_CONVERSION_BITS`] — e.g. `2^(2^40)`, which is a perfectly
    /// valid `BigFloat` whose integer form needs ~137 GB. Use
    /// [`BigFloat::try_to_bigint_trunc`] for a non-panicking alternative.
    pub fn to_bigint_trunc(&self) -> BigInt {
        match self.try_to_bigint_trunc() {
            Ok(v) => v,
            Err(e) => panic!("BigFloat::to_bigint_trunc: {e}"),
        }
    }

    /// Non-panicking [`BigFloat::to_bigint_trunc`].
    ///
    /// Returns [`OxiNumError::Overflow`] when the exact integer would exceed
    /// [`MAX_EXACT_CONVERSION_BITS`] instead of requesting an allocation large
    /// enough to abort the process.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxinum_core::Sign;
    /// use oxinum_float::native::{BigFloat, RoundingMode};
    /// use oxinum_int::native::{BigInt, BigUint};
    ///
    /// let x = BigFloat::from_f64(3.7, 64).expect("3.7");
    /// assert_eq!(x.try_to_bigint_trunc().expect("in budget"), BigInt::from(3i64));
    ///
    /// // 2^(2^40) is a valid BigFloat, but its exact integer form is ~137 GB.
    /// let huge = BigFloat::from_parts(
    ///     Sign::Positive,
    ///     BigUint::one(),
    ///     1i64 << 40,
    ///     53,
    ///     RoundingMode::HalfEven,
    /// );
    /// assert!(huge.try_to_bigint_trunc().is_err());
    /// ```
    pub fn try_to_bigint_trunc(&self) -> OxiNumResult<BigInt> {
        // Non-finite BigFloat has no meaningful integer representation.
        // Return zero as a documented lossy fallback for NaN and ±Inf.
        if !self.is_finite() {
            return Ok(BigInt::zero());
        }
        if self.is_zero() {
            return Ok(BigInt::zero());
        }
        if self.exponent >= 0 {
            // No fractional bits: value is mantissa * 2^exponent (integer).
            let shift = self.exponent as u64;
            let bits = self.mantissa.bit_length().saturating_add(shift);
            if bits > MAX_EXACT_CONVERSION_BITS {
                return Err(exact_conversion_overflow("BigFloat -> BigInt", bits));
            }
            let mag = self.mantissa.shl_bits(shift);
            return Ok(BigInt::from_parts(self.sign, mag));
        }
        // Shift right to drop the fractional part. Always shrinks, so there is
        // no budget to enforce here.
        // `unsigned_abs`, not `-self.exponent`: the exponent can legitimately
        // reach `i64::MIN` because `round_to_precision_in_place` lowers it with
        // `saturating_sub`, and negating `i64::MIN` overflows.
        let shift = self.exponent.unsigned_abs();
        let mag = self.mantissa.shr_bits(shift);
        if mag.is_zero() {
            Ok(BigInt::zero())
        } else {
            Ok(BigInt::from_parts(self.sign, mag))
        }
    }

    /// Convert to [`BigInt`] by rounding toward negative infinity (floor).
    ///
    /// For negative values with a non-zero fractional part, the result is
    /// one less than the truncation (i.e. more negative).
    ///
    /// # Examples
    ///
    /// ```
    /// use oxinum_float::native::BigFloat;
    /// use oxinum_int::native::BigInt;
    ///
    /// let x = BigFloat::from_f64(3.7, 64).expect("3.7");
    /// assert_eq!(x.to_bigint_floor(), BigInt::from(3i64));
    ///
    /// let y = BigFloat::from_f64(-3.7, 64).expect("-3.7");
    /// assert_eq!(y.to_bigint_floor(), BigInt::from(-4i64));
    /// ```
    ///
    /// # Panics
    ///
    /// Same budget as [`BigFloat::to_bigint_trunc`]. Use
    /// [`BigFloat::try_to_bigint_floor`] for a non-panicking alternative.
    pub fn to_bigint_floor(&self) -> BigInt {
        match self.try_to_bigint_floor() {
            Ok(v) => v,
            Err(e) => panic!("BigFloat::to_bigint_floor: {e}"),
        }
    }

    /// Non-panicking [`BigFloat::to_bigint_floor`].
    pub fn try_to_bigint_floor(&self) -> OxiNumResult<BigInt> {
        let t = self.try_to_bigint_trunc()?;
        if self.sign == Sign::Negative && self.has_fractional_part() {
            Ok(&t - &BigInt::one())
        } else {
            Ok(t)
        }
    }

    /// Convert to [`BigInt`] by rounding toward positive infinity (ceiling).
    ///
    /// For positive values with a non-zero fractional part, the result is
    /// one more than the truncation.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxinum_float::native::BigFloat;
    /// use oxinum_int::native::BigInt;
    ///
    /// let x = BigFloat::from_f64(3.7, 64).expect("3.7");
    /// assert_eq!(x.to_bigint_ceil(), BigInt::from(4i64));
    ///
    /// let y = BigFloat::from_f64(-3.7, 64).expect("-3.7");
    /// assert_eq!(y.to_bigint_ceil(), BigInt::from(-3i64));
    /// ```
    ///
    /// # Panics
    ///
    /// Same budget as [`BigFloat::to_bigint_trunc`]. Use
    /// [`BigFloat::try_to_bigint_ceil`] for a non-panicking alternative.
    pub fn to_bigint_ceil(&self) -> BigInt {
        match self.try_to_bigint_ceil() {
            Ok(v) => v,
            Err(e) => panic!("BigFloat::to_bigint_ceil: {e}"),
        }
    }

    /// Non-panicking [`BigFloat::to_bigint_ceil`].
    pub fn try_to_bigint_ceil(&self) -> OxiNumResult<BigInt> {
        let t = self.try_to_bigint_trunc()?;
        if self.sign == Sign::Positive && self.has_fractional_part() {
            Ok(&t + &BigInt::one())
        } else {
            Ok(t)
        }
    }

    /// Convert to [`BigInt`] by rounding half-away-from-zero.
    ///
    /// - Fractional part < 1/2: truncate toward zero.
    /// - Fractional part == 1/2: round away from zero (ties-away).
    /// - Fractional part > 1/2: round away from zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxinum_float::native::BigFloat;
    /// use oxinum_int::native::BigInt;
    ///
    /// let x = BigFloat::from_f64(3.7, 64).expect("3.7");
    /// assert_eq!(x.to_bigint_round(), BigInt::from(4i64));
    ///
    /// let y = BigFloat::from_f64(-3.7, 64).expect("-3.7");
    /// assert_eq!(y.to_bigint_round(), BigInt::from(-4i64));
    /// ```
    ///
    /// # Panics
    ///
    /// Same budget as [`BigFloat::to_bigint_trunc`]. Use
    /// [`BigFloat::try_to_bigint_round`] for a non-panicking alternative.
    pub fn to_bigint_round(&self) -> BigInt {
        match self.try_to_bigint_round() {
            Ok(v) => v,
            Err(e) => panic!("BigFloat::to_bigint_round: {e}"),
        }
    }

    /// Non-panicking [`BigFloat::to_bigint_round`].
    pub fn try_to_bigint_round(&self) -> OxiNumResult<BigInt> {
        if !self.has_fractional_part() {
            return self.try_to_bigint_trunc();
        }
        // If fractional part >= 1/2 (half or more), round away from zero.
        if self.half_exactly() || self.more_than_half() {
            let t = self.try_to_bigint_trunc()?;
            if self.sign == Sign::Positive {
                Ok(&t + &BigInt::one())
            } else {
                Ok(&t - &BigInt::one())
            }
        } else {
            // Fractional part < 1/2: truncate toward zero.
            self.try_to_bigint_trunc()
        }
    }
}

impl BigFloat {
    /// Encode the integer `n` as a `BigFloat` at `prec` bits of precision.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxinum_float::native::{BigFloat, RoundingMode};
    /// let a = BigFloat::from_i64(-42, 16, RoundingMode::HalfEven);
    /// assert_eq!(a.to_f64(), -42.0);
    /// ```
    pub fn from_i64(n: i64, prec: u32, mode: RoundingMode) -> Self {
        assert!(prec > 0, "BigFloat precision must be > 0");
        if n == 0 {
            return Self::zero(prec);
        }
        let (sign, mag_u64) = if n < 0 {
            // Avoid `-i64::MIN` overflow: take the two's-complement magnitude
            // via wrapping_neg, which is exact in unsigned space.
            (Sign::Negative, (n as i128).unsigned_abs() as u64)
        } else {
            (Sign::Positive, n as u64)
        };
        let mantissa = BigUint::from_u64(mag_u64);
        Self::from_parts(sign, mantissa, 0, prec, mode)
    }

    /// Decompose an IEEE-754 `f64` value into a `BigFloat` at `prec` bits.
    ///
    /// At `prec >= 53` the decomposition is exact (no rounding occurs). The
    /// rounding mode used for any narrowing is [`RoundingMode::HalfEven`].
    ///
    /// # Errors
    ///
    /// Returns [`OxiNumError::Parse`] if `x` is `NaN` or infinite — native
    /// `BigFloat` does not yet model those special values.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxinum_float::native::BigFloat;
    /// let half = BigFloat::from_f64(0.5, 1).expect("0.5 fits in 1 bit");
    /// assert_eq!(half.to_f64(), 0.5);
    /// ```
    pub fn from_f64(x: f64, prec: u32) -> OxiNumResult<Self> {
        assert!(prec > 0, "BigFloat precision must be > 0");
        if x.is_nan() {
            return Err(OxiNumError::Parse("cannot encode NaN as BigFloat".into()));
        }
        if x.is_infinite() {
            return Err(OxiNumError::Parse(
                "cannot encode infinity as BigFloat".into(),
            ));
        }
        if x == 0.0 {
            // Both +0.0 and -0.0 land at canonical zero.
            return Ok(Self::zero(prec));
        }
        // Decode the IEEE-754 layout.
        let bits = x.to_bits();
        let sign_bit = (bits >> 63) & 1;
        let biased_exp = ((bits >> 52) & 0x7FF) as i64;
        let fraction = bits & ((1u64 << 52) - 1);
        let sign = if sign_bit == 1 {
            Sign::Negative
        } else {
            Sign::Positive
        };
        let (mantissa_u64, unbiased_exp) = if biased_exp == 0 {
            // Subnormal: no implicit leading bit, exponent fixed at -1074.
            (fraction, -1074_i64)
        } else {
            // Normal: implicit leading bit at position 52, exponent unbiased.
            (fraction | (1u64 << 52), biased_exp - 1023 - 52)
        };
        let mantissa = BigUint::from_u64(mantissa_u64);
        // `try_from_parts`: an `f64` exponent spans only `[-1074, 1024]`, so
        // the range check can never fire — taking the fallible path anyway
        // keeps every externally-fed constructor on one code path and would
        // surface, rather than saturate, any future change to the decoding.
        Self::try_from_parts(sign, mantissa, unbiased_exp, prec, RoundingMode::HalfEven)
    }

    /// Round to the nearest [`f64`] (ties-to-even), with IEEE-754 gradual
    /// underflow. Values whose magnitudes exceed the `f64` overflow threshold
    /// saturate to ±∞; values below half of `2^-1074` round to ±0.
    ///
    /// # Single rounding
    ///
    /// The conversion rounds **once**, straight from the exact stored value
    /// `mantissa * 2^exponent` onto the `f64` grid — including the subnormal
    /// grid, whose spacing is a fixed `2^-1074` rather than the 53
    /// significant bits a normal carries. Rounding to 53 bits first and then
    /// onto the subnormal grid would be a double rounding, and double
    /// rounding is not merely imprecise: for a quotient landing in
    /// `[2^-1023, 2^-1022)` it disagrees with the correctly-rounded result
    /// about a quarter of the time, because the halfway points of the 52-bit
    /// subnormal grid are exactly the points of the 53-bit normal grid.
    ///
    /// The value stored in a `BigFloat` may itself be the rounded result of a
    /// previous operation, and re-rounding it here would double-round for the
    /// same reason. That residue is removed by consulting the direction of
    /// that previous rounding whenever this conversion lands on an exact tie:
    /// the recorded direction says which side of the tie the un-rounded value
    /// lay on, which is precisely the information an exact tie is missing. It
    /// is the same device as MPFR's ternary value plus `mpfr_subnormalize`,
    /// and it is what makes `a.div_ref(&b).to_f64()` agree with the `f64`
    /// quotient bit for bit all the way down to `2^-1074`.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxinum_float::native::{BigFloat, RoundingMode};
    /// let one = BigFloat::from_i64(1, 53, RoundingMode::HalfEven);
    /// assert_eq!(one.to_f64(), 1.0);
    ///
    /// // Gradual underflow: 3 * 2^-1076 = 1.5 * 2^-1075 is below the smallest
    /// // subnormal but above half of it, so it rounds up rather than to zero.
    /// use oxinum_core::Sign;
    /// use oxinum_int::native::BigUint;
    /// let tiny = BigFloat::from_parts(
    ///     Sign::Positive,
    ///     BigUint::from_u64(3),
    ///     -1076,
    ///     53,
    ///     RoundingMode::HalfEven,
    /// );
    /// assert_eq!(tiny.to_f64(), 5e-324);
    /// ```
    pub fn to_f64(&self) -> f64 {
        // Non-finite values must be handled before the is_zero() check, since
        // NaN and Inf have mantissa=0 and would otherwise fall through to the
        // subnormal path, producing incorrect finite results.
        match self.class {
            FloatClass::Nan => return f64::NAN,
            FloatClass::Infinite => return signed_inf(self.sign == Sign::Negative),
            FloatClass::Finite => {}
        }
        if self.is_zero() {
            // The canonical zero is unsigned; `from_f64` maps both ±0.0 onto
            // it, so `+0.0` is the only faithful image back.
            return 0.0;
        }
        let negative = self.sign == Sign::Negative;
        let bits = self.mantissa.bit_length();
        // Position of the leading set bit: `2^top_bit_exp <= |x| < 2^(top_bit_exp+1)`.
        // Evaluated in `i128`: the raw `exponent` can legitimately sit near
        // `i64::MIN` (`round_to_precision_in_place` lowers it with
        // `saturating_sub`), so the addition would overflow in `i64`.
        let top_bit_exp = i128::from(self.exponent) + bits as i128 - 1;
        // At or above 2^1024 nothing finite is left. The borderline case —
        // `top_bit_exp == 1023` with a mantissa that rounds up to 2^1024 — is
        // caught by the carry check further down, so the IEEE overflow
        // threshold (2^1024 - 2^970) is honoured exactly rather than
        // approximated by this cheap pre-filter.
        if top_bit_exp > 1023 {
            return signed_inf(negative);
        }
        // Strictly below half the smallest subnormal there is nothing to round
        // to. The general path below reaches the same answer; this is only a
        // shortcut past a pointlessly wide shift.
        if top_bit_exp < -1076 {
            return signed_zero(negative);
        }
        // `q` is the binary exponent of the target grid. An `f64` is `n * 2^q`
        // with `n` a 53-bit integer for normals; subnormals share the single
        // grid `q == -1074` with `n < 2^52`. Clamping `q` at `-1074` is
        // precisely what makes the underflow *gradual* — without the clamp the
        // grid would keep refining below `f64::MIN_POSITIVE` and the value
        // would be encoded at a resolution the format does not have.
        let q = if top_bit_exp - 52 > -1074 {
            top_bit_exp - 52
        } else {
            -1074
        };
        // Rounding the magnitude onto that grid is rounding `mantissa /
        // 2^shift` to an integer, ties to even.
        let shift = q - i128::from(self.exponent);
        let mut n: u64 = if shift <= 0 {
            // The grid is finer than the stored value: exact, no rounding.
            // `-shift` is at most `53 - bits`, so the widened mantissa still
            // fits inside 53 bits.
            let widen = u64::try_from(-shift).unwrap_or(u64::MAX);
            biguint_to_u64_saturating(&self.mantissa.shl_bits(widen))
        } else {
            // `shift` is bounded by `971 + 2^63`, so the narrowing always
            // succeeds; `u64::MAX` is a total fallback, and `shr_bits` /
            // `test_bit` both answer with 0 / false past the mantissa's width.
            let drop = u64::try_from(shift).unwrap_or(u64::MAX);
            let truncated = biguint_to_u64_saturating(&self.mantissa.shr_bits(drop));
            let guard = self.mantissa.test_bit(drop - 1);
            let sticky = drop >= 2 && self.mantissa.trailing_zeros() < drop - 1;
            // Round half to even — except that `guard && !sticky` is only an
            // exact tie against the *stored* value. If the stored value is
            // itself a rounded approximation, the direction it was rounded in
            // says which side of this tie the un-rounded value lay on, and
            // that beats the ties-to-even convention: it is the true answer
            // rather than a coin flip. Without it, a correctly-rounded
            // division whose quotient lands in `[2^-1023, 2^-1022)` disagrees
            // with the correctly-rounded `f64` quotient roughly a quarter of
            // the time, because the halfway points of that subnormal grid are
            // exactly the points of the 53-bit normal grid.
            let round_up = guard
                && if sticky {
                    true
                } else {
                    match self.round_dir {
                        // Stored magnitude sits above the exact one, so the
                        // exact value is below this tie: round down.
                        RoundDirection::Up => false,
                        // Symmetrically, below the stored value means above
                        // the tie.
                        RoundDirection::Down => true,
                        // A genuine tie: ties-to-even.
                        RoundDirection::Unknown => truncated & 1 == 1,
                    }
                };
            // `saturating_add`: `truncated` is provably below `2^53` here, so
            // the carry cannot overflow — but a total conversion should not
            // acquire a debug-only panic on a branch it can never take.
            truncated.saturating_add(u64::from(round_up))
        };
        if n == 0 {
            return signed_zero(negative);
        }
        // A round-up can carry out of the 53-bit window (`0b1…1` → `0b10…0`).
        // Re-normalizing costs one grid step; overflow past `f64::MAX` is then
        // caught by the exponent check below, which is what implements the
        // IEEE threshold of `2^1024 - 2^970` rather than `f64::MAX` itself.
        let mut q = q;
        if n == 1u64 << 53 {
            n >>= 1;
            q += 1;
        }
        let sign_bit = u64::from(negative);
        if n >= 1u64 << 52 {
            // Normal: `n` spans 53 bits and its top bit is the implicit one.
            let unbiased = q + 52;
            if unbiased > 1023 {
                return signed_inf(negative);
            }
            let biased = (unbiased + 1023) as u64;
            let fraction = n & ((1u64 << 52) - 1);
            f64::from_bits((sign_bit << 63) | (biased << 52) | fraction)
        } else {
            // Subnormal: the grid clamp forced `q == -1074`, which makes `n`
            // the stored 52-bit fraction itself with a zero biased exponent.
            debug_assert_eq!(q, -1074, "sub-2^52 significand outside the subnormal grid");
            f64::from_bits((sign_bit << 63) | n)
        }
    }
}

/// `BigUint` → `u64`, saturating instead of panicking.
///
/// Every call site in [`BigFloat::to_f64`] has already bounded the value below
/// `2^53`, so the saturation is unreachable; it exists so the conversion stays
/// total without an `expect`.
fn biguint_to_u64_saturating(n: &BigUint) -> u64 {
    n.to_u64().unwrap_or(u64::MAX)
}

/// `±0.0`, selected by a "is negative" flag.
#[inline]
fn signed_zero(negative: bool) -> f64 {
    if negative {
        -0.0
    } else {
        0.0
    }
}

/// `±∞`, selected by a "is negative" flag.
#[inline]
fn signed_inf(negative: bool) -> f64 {
    if negative {
        f64::NEG_INFINITY
    } else {
        f64::INFINITY
    }
}

impl BigFloat {
    /// Convert a [`BigInt`] (signed arbitrary-precision integer) to `BigFloat`
    /// at `prec` bits of precision using the given rounding mode.
    ///
    /// The conversion is exact when `prec >= n.magnitude().bit_length()`;
    /// otherwise the result is rounded according to `mode`.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxinum_float::native::{BigFloat, RoundingMode};
    /// use oxinum_int::native::BigInt;
    ///
    /// let n = BigInt::from(-42i64);
    /// let f = BigFloat::from_bigint(&n, 64, RoundingMode::HalfEven);
    /// assert_eq!(f.to_f64(), -42.0);
    /// ```
    pub fn from_bigint(n: &BigInt, prec: u32, mode: RoundingMode) -> Self {
        assert!(prec > 0, "BigFloat precision must be > 0");
        let sign = n.sign();
        let mag = n.magnitude().clone();
        let f = Self::from_biguint(&mag, prec, mode);
        if sign == Sign::Negative && !f.is_zero() {
            f.neg()
        } else {
            f
        }
    }

    /// Convert a [`BigUint`] (non-negative arbitrary-precision integer) to
    /// `BigFloat` at `prec` bits of precision using the given rounding mode.
    ///
    /// The conversion is exact when `prec >= n.bit_length()`; otherwise the
    /// result is rounded according to `mode`.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxinum_float::native::{BigFloat, RoundingMode};
    /// use oxinum_int::native::BigUint;
    ///
    /// let n = BigUint::from_u64(1024);
    /// let f = BigFloat::from_biguint(&n, 64, RoundingMode::HalfEven);
    /// assert_eq!(f.to_f64(), 1024.0);
    /// ```
    pub fn from_biguint(n: &BigUint, prec: u32, mode: RoundingMode) -> Self {
        assert!(prec > 0, "BigFloat precision must be > 0");
        if n.is_zero() {
            return Self::zero(prec);
        }
        // `from_parts` calls `canonicalize_normalize` then `round_to_precision_in_place`,
        // which handles all the bit-length vs prec casework for us.
        Self::from_parts(Sign::Positive, n.clone(), 0, prec, mode)
    }
}

impl From<i64> for BigFloat {
    /// Encodes `n` at 64 bits of precision with banker's rounding.
    ///
    /// Use [`BigFloat::from_i64`] for explicit precision control.
    fn from(n: i64) -> Self {
        Self::from_i64(n, 64, RoundingMode::HalfEven)
    }
}

impl TryFrom<f64> for BigFloat {
    type Error = OxiNumError;
    fn try_from(x: f64) -> Result<Self, Self::Error> {
        // 53 bits captures every finite double exactly.
        Self::from_f64(x, 53)
    }
}
