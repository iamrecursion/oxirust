//! Addition, subtraction, and negation for `BigFloat`.
//!
//! The strategy used here is exact-shift alignment + sign-aware mantissa
//! arithmetic. Because [`BigUint`] is arbitrary-precision, we can always
//! shift the higher-exponent mantissa *left* (no precision loss) so that
//! both operands share the smaller exponent. After the integer add or sub,
//! the result is rounded to the larger of the two operand precisions.

use core::ops::{Add, AddAssign, Neg, Sub, SubAssign};

use oxinum_core::Sign;
use oxinum_int::native::BigUint;

use super::float::{BigFloat, RoundingMode};
use super::nonfinite::{nonfinite_propagate, BinOp};

impl BigFloat {
    /// Return `self + other`, rounding the result to
    /// `max(self.precision, other.precision)` using banker's rounding.
    pub fn add_ref(&self, other: &BigFloat) -> BigFloat {
        self.add_ref_with_mode(other, RoundingMode::HalfEven)
    }

    /// Return `self + other`, rounding the result to
    /// `max(self.precision, other.precision)` using the chosen rounding mode.
    pub fn add_ref_with_mode(&self, other: &BigFloat, mode: RoundingMode) -> BigFloat {
        // Non-finite fast-path: propagate NaN/Inf per IEEE 754.
        if let Some(result) = nonfinite_propagate(self, other, BinOp::Add) {
            return result;
        }
        let target_prec = self.precision.max(other.precision);
        if self.is_zero() {
            return other.clone().round_to_precision(target_prec, mode);
        }
        if other.is_zero() {
            return self.clone().round_to_precision(target_prec, mode);
        }

        // Step 1: Align to the smaller exponent by shifting the higher-exp
        // mantissa left by exp_diff (bounded — see `align_to_common_exp`).
        let (common_exp, lhs_mag, rhs_mag) = align_to_common_exp(self, other, target_prec);

        // Step 2: Sign-aware add/sub of magnitudes.
        let (out_sign, out_mag) = match (self.sign, other.sign) {
            (Sign::Positive, Sign::Positive) | (Sign::Negative, Sign::Negative) => {
                (self.sign, &lhs_mag + &rhs_mag)
            }
            _ => {
                // Magnitudes will be subtracted: we compute |lhs_mag - rhs_mag|
                // and pick the sign from the larger operand. By construction
                // lhs_mag corresponds to `self` and rhs_mag to `other`.
                if lhs_mag >= rhs_mag {
                    // |self| >= |other| -> sign of self.
                    let diff = lhs_mag.checked_sub(&rhs_mag).expect(
                        "add/sub invariant: lhs_mag >= rhs_mag in this branch ensures \
                         non-negative subtraction",
                    );
                    (self.sign, diff)
                } else {
                    let diff = rhs_mag.checked_sub(&lhs_mag).expect(
                        "add/sub invariant: lhs_mag < rhs_mag in this branch ensures rhs_mag > \
                         lhs_mag",
                    );
                    (other.sign, diff)
                }
            }
        };

        // Step 3: Land back at the canonical form at target precision.
        // `from_parts` (saturating), not `try_from_parts`: `add_ref_with_mode`
        // and the `Add`/`Sub` operators built on it are total functions with no
        // `Result` to carry an error. A sum can leave the exponent range only
        // when an operand already sits within one bit of `EMAX`, in which case
        // the documented saturation of `BigFloat::from_parts` applies. Callers
        // that must detect the condition can compare `ilogb()` against `EMAX`.
        BigFloat::from_parts(out_sign, out_mag, common_exp, target_prec, mode)
    }

    /// Return `self - other` at `max(p_self, p_other)` precision, banker's
    /// rounding.
    pub fn sub_ref(&self, other: &BigFloat) -> BigFloat {
        self.sub_ref_with_mode(other, RoundingMode::HalfEven)
    }

    /// Return `self - other` at `max(p_self, p_other)` precision with the
    /// given rounding mode.
    pub fn sub_ref_with_mode(&self, other: &BigFloat, mode: RoundingMode) -> BigFloat {
        self.add_ref_with_mode(&other.neg(), mode)
    }
}

/// Guard bits kept below the retained precision. When the two operands are so
/// far apart that the smaller one lands entirely below this guard region it can
/// no longer affect the result except through the round/sticky decision, so we
/// fold it into a single sticky bit instead of materializing the full shift.
const ALIGN_GUARD_BITS: u32 = 2;

/// Align two non-zero `BigFloat`s so they share a common exponent. Returns
/// `(common_exp, lhs_mantissa, rhs_mantissa)` such that
/// `lhs.value == lhs_mantissa * 2^common_exp` and likewise for `rhs` — either
/// exactly, or (for an extreme exponent gap) with the smaller operand collapsed
/// to a sticky bit that reproduces the correctly-rounded result.
///
/// A naive alignment shifts the higher-exponent mantissa left by the full
/// exponent difference, so the allocation grows linearly with that gap. For a
/// large but perfectly valid exponent gap that is an unbounded allocation that
/// aborts the process. Because the sum is rounded to `target_prec` bits, once
/// the smaller operand sits more than `target_prec + guard` bits below the
/// larger one it cannot influence any retained bit — only the sticky bit — so
/// we cap the shift and record its entire contribution as a single sticky `1`.
fn align_to_common_exp(a: &BigFloat, b: &BigFloat, target_prec: u32) -> (i64, BigUint, BigUint) {
    if a.exponent == b.exponent {
        return (a.exponent, a.mantissa.clone(), b.mantissa.clone());
    }
    // Identify the higher-exponent ("big") and lower-exponent ("small") operand.
    let (big, small, big_is_a) = if a.exponent > b.exponent {
        (a, b, true)
    } else {
        (b, a, false)
    };
    // Exponent gap; computed in i128 so it cannot overflow i64 subtraction.
    let gap: u64 = (big.exponent as i128 - small.exponent as i128) as u64;
    // Beyond `cap` bits of shift the small operand cannot reach the guard
    // region below the retained mantissa. `cap` is bounded by the operand
    // precisions, so the shifted mantissa allocation is always bounded.
    let cap = big.mantissa.bit_length() + u64::from(target_prec) + u64::from(ALIGN_GUARD_BITS);
    // Use exact alignment while the small operand can still touch (or overlap)
    // the guard region; `threshold` guarantees `small < 1` unit at `common_exp`
    // in the capped branch below.
    let threshold = cap + small.mantissa.bit_length();
    let (common_exp, big_mag, small_mag) = if gap <= threshold {
        // Exact alignment — the shift is bounded by `threshold`.
        (
            small.exponent,
            big.mantissa.shl_bits(gap),
            small.mantissa.clone(),
        )
    } else {
        // Capped alignment: shift the big operand up by only `cap` bits and
        // represent the far-below small operand as one sticky bit. This keeps
        // add/sub correctly rounded:
        //   - equal signs:   big_mag + 1 sets the sticky bit above big's zero
        //                     tail, so the sum rounds back to `big`;
        //   - opposite signs: |big_mag - 1| borrows through big's zero tail,
        //                     leaving an all-ones sticky tail — the near-tie
        //                     and power-of-two predecessor cases then round
        //                     correctly in `from_parts`.
        (
            big.exponent.saturating_sub(cap as i64),
            big.mantissa.shl_bits(cap),
            BigUint::one(),
        )
    };
    if big_is_a {
        (common_exp, big_mag, small_mag)
    } else {
        (common_exp, small_mag, big_mag)
    }
}

// ---------------------------------------------------------------------------
// Operator impls — owned and borrowed
// ---------------------------------------------------------------------------

impl Add<&BigFloat> for &BigFloat {
    type Output = BigFloat;
    #[inline]
    fn add(self, rhs: &BigFloat) -> BigFloat {
        self.add_ref(rhs)
    }
}

impl Add<BigFloat> for BigFloat {
    type Output = BigFloat;
    #[inline]
    fn add(self, rhs: BigFloat) -> BigFloat {
        self.add_ref(&rhs)
    }
}

impl Add<&BigFloat> for BigFloat {
    type Output = BigFloat;
    #[inline]
    fn add(self, rhs: &BigFloat) -> BigFloat {
        self.add_ref(rhs)
    }
}

impl Add<BigFloat> for &BigFloat {
    type Output = BigFloat;
    #[inline]
    fn add(self, rhs: BigFloat) -> BigFloat {
        self.add_ref(&rhs)
    }
}

impl Sub<&BigFloat> for &BigFloat {
    type Output = BigFloat;
    #[inline]
    fn sub(self, rhs: &BigFloat) -> BigFloat {
        self.sub_ref(rhs)
    }
}

impl Sub<BigFloat> for BigFloat {
    type Output = BigFloat;
    #[inline]
    fn sub(self, rhs: BigFloat) -> BigFloat {
        self.sub_ref(&rhs)
    }
}

impl Sub<&BigFloat> for BigFloat {
    type Output = BigFloat;
    #[inline]
    fn sub(self, rhs: &BigFloat) -> BigFloat {
        self.sub_ref(rhs)
    }
}

impl Sub<BigFloat> for &BigFloat {
    type Output = BigFloat;
    #[inline]
    fn sub(self, rhs: BigFloat) -> BigFloat {
        self.sub_ref(&rhs)
    }
}

impl AddAssign<&BigFloat> for BigFloat {
    #[inline]
    fn add_assign(&mut self, rhs: &BigFloat) {
        *self = self.add_ref(rhs);
    }
}

impl AddAssign<BigFloat> for BigFloat {
    #[inline]
    fn add_assign(&mut self, rhs: BigFloat) {
        *self = self.add_ref(&rhs);
    }
}

impl SubAssign<&BigFloat> for BigFloat {
    #[inline]
    fn sub_assign(&mut self, rhs: &BigFloat) {
        *self = self.sub_ref(rhs);
    }
}

impl SubAssign<BigFloat> for BigFloat {
    #[inline]
    fn sub_assign(&mut self, rhs: BigFloat) {
        *self = self.sub_ref(&rhs);
    }
}

impl Neg for BigFloat {
    type Output = BigFloat;
    #[inline]
    fn neg(self) -> BigFloat {
        BigFloat::neg(&self)
    }
}

impl Neg for &BigFloat {
    type Output = BigFloat;
    #[inline]
    fn neg(self) -> BigFloat {
        BigFloat::neg(self)
    }
}
