//! Division for `BigFloat`.
//!
//! Strategy: rigorous integer division on scaled mantissas. Conceptually,
//!
//! ```text
//! a / b = (m_a * 2^e_a) / (m_b * 2^e_b)
//!       = (m_a / m_b) * 2^(e_a - e_b)
//! ```
//!
//! `m_a / m_b` is generally not an integer, so to get `target_prec` correct
//! bits in the result mantissa we left-shift `m_a` by `shift = target_prec +
//! guard` bits before integer-dividing — that promotes `guard` extra bits of
//! quotient resolution which the post-division rounding pass then consumes.
//!
//! Result exponent is `e_a - e_b - shift`. We use saturating arithmetic to
//! mirror the convention established in `float_add.rs` /
//! `round_to_precision_in_place`.
//!
//! This is the "simple reference division" path called out in the N4b plan
//! as the rigorous fallback for the Newton-Raphson reciprocal approach.
//! Trade-off: one big integer multiply (the shift) plus one big integer
//! division. For the precisions OxiNum targets (a few hundred to a few
//! thousand bits) this is comfortably fast, and crucially it is exact — no
//! seed-convergence questions to debug at extreme exponents.

use core::ops::{Div, DivAssign, Rem, RemAssign};

use oxinum_core::{OxiNumError, OxiNumResult, Sign};
use oxinum_int::native::{divrem, BigUint};

use super::float::{BigFloat, RoundingMode};
use super::nonfinite::{nonfinite_binop, nonfinite_propagate, BinOp};

/// Extra guard bits used during the quotient computation so the final round
/// has well-defined round/sticky semantics. 8 bits ensures the round-bit and
/// at least 7 sticky bits are present.
const DIV_GUARD_BITS: u32 = 8;

impl BigFloat {
    /// Return `self / other` at `max(p_self, p_other)` precision using
    /// banker's rounding.
    ///
    /// # Errors
    ///
    /// Returns [`OxiNumError::DivByZero`] if `other` is the canonical zero.
    pub fn div_ref(&self, other: &BigFloat) -> OxiNumResult<BigFloat> {
        self.div_ref_with_mode(other, RoundingMode::HalfEven)
    }

    /// Return `self / other` at `max(p_self, p_other)` precision with the
    /// given rounding mode.
    ///
    /// # Errors
    ///
    /// Returns [`OxiNumError::DivByZero`] if `other` is the canonical zero.
    pub fn div_ref_with_mode(
        &self,
        other: &BigFloat,
        mode: RoundingMode,
    ) -> OxiNumResult<BigFloat> {
        // Non-finite fast-path: propagate NaN/Inf inputs per IEEE 754.
        // Does NOT generate Inf from finite/0 — that stays as Err(DivByZero).
        if let Some(result) = nonfinite_propagate(self, other, BinOp::Div) {
            return Ok(result);
        }
        if other.is_zero() {
            return Err(OxiNumError::DivByZero);
        }
        let target_prec = self.precision.max(other.precision);
        if self.is_zero() {
            // 0 / nonzero -> canonical zero at the target precision.
            return Ok(BigFloat::zero(target_prec));
        }
        // Result sign: equal signs -> Positive, opposite signs -> Negative.
        let out_sign = if self.sign == other.sign {
            Sign::Positive
        } else {
            Sign::Negative
        };
        // Shift m_a left by `target_prec + guard` bits (plus however many bits
        // m_b is wider than m_a), then floor-divide by m_b. The quotient
        // carries enough bits for from_parts to round back to `target_prec`
        // correctly.
        //
        // WHY THE `bits_b - bits_a` TERM. `floor((m_a << s) / m_b)` has
        // `bits_a + s - bits_b` or one more bits. Sizing `s` against the
        // *target* alone is only sound while the operands are equally wide:
        // with `p_b` much larger than `p_a` — e.g. `1` at precision 8 divided
        // by `3` at precision 53 — the quotient lands *below* `target_prec`,
        // `round_to_precision_in_place` takes its zero-pad branch instead of
        // rounding, and the result carries only `bits_a + s - bits_b` correct
        // bits with the forced sticky bit (below) frozen into the value rather
        // than consumed by a rounding pass. Widening the shift by
        // `bits_b - bits_a` restores `quotient_bits >= target_prec + guard`
        // for every operand pair, which is also the precondition the
        // round/sticky split below relies on.
        //
        // CORRECTLY-ROUNDED DIVISION: we must also carry the division remainder
        // into the sticky bit. Without this, if the low guard bits of the
        // quotient happen to be zero but the remainder is nonzero, the
        // round_drop_low_bits pass would compute sticky=false and round wrong
        // (off by 1 ULP). Setting bit 0 of the quotient when remainder≠0
        // is safe because DIV_GUARD_BITS >= 2, so bit 0 is strictly below the
        // round bit and only affects the sticky bits.
        let bits_a = self.mantissa.bit_length();
        let bits_b = other.mantissa.bit_length();
        let shift_bits = (u64::from(target_prec) + u64::from(DIV_GUARD_BITS))
            .saturating_add(bits_b.saturating_sub(bits_a));
        let scaled = self.mantissa.shl_bits(shift_bits);
        let (mut quotient, remainder) = divrem(&scaled, &other.mantissa);
        if !remainder.is_zero() {
            // Ensure the sticky bit is set so that round_drop_low_bits can
            // correctly round toward nearest-even when the true quotient is
            // not an exact multiple of 2^(−shift_bits).
            quotient.set_bit(0);
        }
        // Exponent of the bare integer quotient.
        // value = quotient * 2^(e_a - e_b - shift_bits)
        let exp_after_div = self
            .exponent
            .saturating_sub(other.exponent)
            .saturating_sub(shift_bits as i64);
        // Land at canonical form (normalize + round to target_prec).
        // `from_parts` (saturating), not `try_from_parts`: the `Div` operators
        // route through `div_ieee`, which expects `div_ref` to fail only with
        // `DivByZero`, and this function's documented `# Errors` contract is
        // exactly that one variant. A quotient whose magnitude leaves
        // `[EMIN, EMAX]` — only reachable when the operands straddle nearly the
        // whole range, e.g. `2^EMAX / 2^EMIN` — therefore saturates as
        // documented on `BigFloat::from_parts` instead of introducing a new
        // error variant into a stable signature.
        Ok(BigFloat::from_parts(
            out_sign,
            quotient,
            exp_after_div,
            target_prec,
            mode,
        ))
    }
}

// ---------------------------------------------------------------------------
// Operator impls — owned and borrowed.
//
// The trait `Div` returns `Self::Output` (no Result). The operators follow
// IEEE 754: finite/0 → ±Inf, 0/0 → NaN, Inf/Inf → NaN. Callers that need a
// `Result` (and the DivByZero error on finite/0) should call `div_ref` /
// `div_ref_with_mode` directly.
// ---------------------------------------------------------------------------

fn div_ieee(a: &BigFloat, b: &BigFloat) -> BigFloat {
    // Operator path: use nonfinite_binop which generates Inf from finite/0.
    if let Some(result) = nonfinite_binop(a, b, BinOp::Div) {
        return result;
    }
    // Both finite, b is non-zero: call div_ref (normal path).
    a.div_ref(b)
        .expect("div_ieee: finite non-zero divisor guaranteed by nonfinite_binop guard")
}

impl Div<&BigFloat> for &BigFloat {
    type Output = BigFloat;
    #[inline]
    fn div(self, rhs: &BigFloat) -> BigFloat {
        div_ieee(self, rhs)
    }
}

impl Div<BigFloat> for BigFloat {
    type Output = BigFloat;
    #[inline]
    fn div(self, rhs: BigFloat) -> BigFloat {
        div_ieee(&self, &rhs)
    }
}

impl Div<&BigFloat> for BigFloat {
    type Output = BigFloat;
    #[inline]
    fn div(self, rhs: &BigFloat) -> BigFloat {
        div_ieee(&self, rhs)
    }
}

impl Div<BigFloat> for &BigFloat {
    type Output = BigFloat;
    #[inline]
    fn div(self, rhs: BigFloat) -> BigFloat {
        div_ieee(self, &rhs)
    }
}

impl DivAssign<&BigFloat> for BigFloat {
    #[inline]
    fn div_assign(&mut self, rhs: &BigFloat) {
        *self = div_ieee(self, rhs);
    }
}

impl DivAssign<BigFloat> for BigFloat {
    #[inline]
    fn div_assign(&mut self, rhs: BigFloat) {
        *self = div_ieee(self, &rhs);
    }
}

// ---------------------------------------------------------------------------
// Rem (floating-point remainder: a - trunc(a/b)*b)
//
// This is the IEEE-style truncated remainder (not floor-based).  Follows
// IEEE 754: Inf%y=NaN, x%0=NaN, finite%Inf=finite (returns lhs unchanged),
// NaN propagates.
// ---------------------------------------------------------------------------

/// Compute `2^exp mod modulus` by square-and-multiply.
///
/// Every intermediate is reduced modulo `modulus`, so the peak allocation is
/// twice `modulus`'s width regardless of how large `exp` is. This is what lets
/// [`rem_core`] handle an arbitrary exponent gap without materializing the
/// `exp`-bit shift that a naive alignment would need.
///
/// `modulus` must be non-zero (the callers check `b.is_zero()` first).
fn pow2_mod(exp: u128, modulus: &BigUint) -> BigUint {
    if modulus.is_one() {
        // Everything is 0 mod 1.
        return BigUint::zero();
    }
    let two = BigUint::from_u64(2);
    let mut base = {
        let (_q, r) = divrem(&two, modulus);
        r
    };
    let mut result = BigUint::one();
    let mut remaining = exp;
    while remaining > 0 {
        if remaining & 1 == 1 {
            let (_q, r) = divrem(&(&result * &base), modulus);
            result = r;
        }
        remaining >>= 1;
        if remaining > 0 {
            let (_q, r) = divrem(&(&base * &base), modulus);
            base = r;
        }
    }
    result
}

fn rem_core(a: &BigFloat, b: &BigFloat) -> BigFloat {
    // IEEE 754 non-finite table for remainder.
    if a.is_nan() || b.is_nan() || a.is_infinite() || b.is_zero() {
        return BigFloat::nan(a.precision.max(b.precision));
    }
    // finite % Inf = the finite value (IEEE rule).
    if b.is_infinite() {
        return a.clone();
    }
    let prec = a.precision.max(b.precision);
    // Both finite, b nonzero.
    if a.is_zero() {
        return BigFloat::zero(prec);
    }
    // Truncated IEEE remainder: r = a - trunc(a/b) * b, with |r| < |b| and
    // sign(r) == sign(a). The integer quotient must be computed at FULL
    // precision: rounding a/b to `prec` bits first (as an earlier version did)
    // corrupts the integer part whenever it needs more than `prec` bits, which
    // leaves a remainder outside [0, |b|). Working directly on the mantissas
    // keeps the remainder exact.
    //
    // With a = m_a * 2^e_a and b = m_b * 2^e_b, let d = e_a - e_b:
    //   d >= 0 : numerator = m_a << d,  denom = m_b,          rem_exp = e_b
    //   d <  0 : numerator = m_a,       denom = m_b << (-d),  rem_exp = e_a
    // Then floor(numerator / denom) == trunc(|a/b|), and the division remainder
    // r satisfies  |a| - trunc(|a/b|) * |b| == r * 2^min(e_a, e_b), with
    // 0 <= r < denom, so the returned value lies in [0, |b|). The `i128`
    // subtraction avoids overflow when the exponents straddle the i64 range.
    //
    // UNBOUNDED-ALLOCATION GUARD. Materializing `m_a << d` costs memory linear
    // in `d`, and `d` is driven by the operands' exponents, which are free i64
    // values (`BigFloat::from_parts` imposes no EMAX, and `from_hex_float`
    // parses an arbitrary `p` exponent straight out of an untrusted string).
    // A 2^40 exponent gap would request a ~137 GB intermediate and abort the
    // process. The remainder itself is always bounded (`|r| < |b|`), so the
    // huge intermediate is entirely avoidable:
    //
    //   d >= 0 : r = (m_a << d) mod m_b = ((m_a mod m_b) · (2^d mod m_b)) mod m_b
    //            — `pow2_mod` evaluates `2^d mod m_b` by square-and-multiply,
    //            so no intermediate ever exceeds `m_b`'s width.
    //   d <  0 : once `-d >= bit_length(m_a)` the denominator `m_b << (-d)` has
    //            strictly more bits than the numerator `m_a`, hence the quotient
    //            is 0 and the remainder is `m_a` itself — no shift needed.
    //
    // The direct shift is kept for the common case (it is one division instead
    // of ~64 modular squarings) and is bounded by the operand mantissa widths.
    let d: i128 = a.exponent as i128 - b.exponent as i128;
    let direct_shift_cap: i128 =
        a.mantissa.bit_length() as i128 + b.mantissa.bit_length() as i128 + 64;
    let r = if d >= 0 {
        if d <= direct_shift_cap {
            let numerator = a.mantissa.shl_bits(d as u64);
            let (_quotient, r) = divrem(&numerator, &b.mantissa);
            r
        } else {
            // Bounded modular route: every intermediate stays below `m_b`.
            let reduced_a = {
                let (_q, r) = divrem(&a.mantissa, &b.mantissa);
                r
            };
            let pow = pow2_mod(d as u128, &b.mantissa);
            let (_q, r) = divrem(&(&reduced_a * &pow), &b.mantissa);
            r
        }
    } else {
        let shift = (-d) as u128;
        if shift >= a.mantissa.bit_length() as u128 {
            // denom > numerator ⇒ trunc(a/b) == 0 ⇒ remainder == |a|.
            a.mantissa.clone()
        } else {
            let denom = b.mantissa.shl_bits(shift as u64);
            let (_quotient, r) = divrem(&a.mantissa, &denom);
            r
        }
    };
    let rem_exp = if d >= 0 { b.exponent } else { a.exponent };
    if r.is_zero() {
        return BigFloat::zero(prec);
    }
    // r * 2^rem_exp is exact and strictly less than |b|; carry the dividend's
    // sign per the IEEE truncated-remainder rule.
    // `from_parts` is used unconditionally here — `%` is a total operator — and
    // its saturation is unreachable: `|r * 2^rem_exp| < |b|` and
    // `rem_exp >= min(e_a, e_b)`, so the remainder's magnitude is bracketed by
    // the operands', both of which are already inside `[EMIN, EMAX]`.
    BigFloat::from_parts(a.sign, r, rem_exp, prec, RoundingMode::ToZero)
}

impl Rem<&BigFloat> for &BigFloat {
    type Output = BigFloat;
    #[inline]
    fn rem(self, rhs: &BigFloat) -> BigFloat {
        rem_core(self, rhs)
    }
}

impl Rem<BigFloat> for BigFloat {
    type Output = BigFloat;
    #[inline]
    fn rem(self, rhs: BigFloat) -> BigFloat {
        rem_core(&self, &rhs)
    }
}

impl Rem<&BigFloat> for BigFloat {
    type Output = BigFloat;
    #[inline]
    fn rem(self, rhs: &BigFloat) -> BigFloat {
        rem_core(&self, rhs)
    }
}

impl Rem<BigFloat> for &BigFloat {
    type Output = BigFloat;
    #[inline]
    fn rem(self, rhs: BigFloat) -> BigFloat {
        rem_core(self, &rhs)
    }
}

impl RemAssign<&BigFloat> for BigFloat {
    #[inline]
    fn rem_assign(&mut self, rhs: &BigFloat) {
        *self = rem_core(self, rhs);
    }
}

impl RemAssign<BigFloat> for BigFloat {
    #[inline]
    fn rem_assign(&mut self, rhs: BigFloat) {
        *self = rem_core(self, &rhs);
    }
}
