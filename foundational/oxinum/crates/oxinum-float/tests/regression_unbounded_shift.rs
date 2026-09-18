//! Regression tests for exponent-driven unbounded allocations.
//!
//! `BigFloat` used to store a free `i64` exponent, and `from_hex_float` parses
//! the `p` exponent straight out of an untrusted string. Several operations
//! materialized a shift proportional to that exponent, so a short input such as
//! `"0x1p2000000000"` requested a multi-gigabyte intermediate and aborted the
//! process (an abort, not a catchable panic).
//!
//! The exponent range is now enforced at construction
//! (`BigFloat::EMIN..=BigFloat::EMAX`, `±2^40`; `try_from_parts` rejects,
//! `from_parts` saturates), which bounds the *inputs*. The per-operation
//! guards exercised here remain the bound on what a single operation may
//! *materialize*: `2^40` bits is still 137 GB, far above any budget.
//!
//! Covered here:
//!
//! 1. `%` / `rem` shifted the dividend mantissa left by the full exponent gap
//!    even though the remainder is always bounded by `|b|`.
//! 2. `bs_transcendental::split_arg` lifted the argument of `exp`/`sin`/`cos`
//!    to an exact rational with a `2^|exponent|`-bit denominator.
//! 3. `to_bigint_*` / `float_to_rational` had no bit budget and negated the
//!    exponent with `-e`, which overflows for `e == i64::MIN`.

use oxinum_core::{OxiNumError, Sign};
use oxinum_float::native::{BigFloat, RoundingMode};
use oxinum_int::native::{BigInt, BigUint};

const MODE: RoundingMode = RoundingMode::HalfEven;

/// `2^n` at `prec` bits, for an arbitrarily large `n`.
fn pow2(exponent: i64, prec: u32) -> BigFloat {
    BigFloat::from_parts(Sign::Positive, BigUint::one(), exponent, prec, MODE)
}

// ---------------------------------------------------------------------------
// 1. rem / %
// ---------------------------------------------------------------------------

/// `2^(2^40) mod 3`. The naive alignment shifts the dividend mantissa left by
/// `2^40` bits (~137 GB) before dividing; the bounded route reduces modulo the
/// divisor mantissa instead. `2^k mod 3` is `1` for even `k`, so the answer is
/// exactly `1`.
#[test]
fn rem_huge_positive_exponent_gap_is_bounded_and_exact() {
    let a = pow2(1i64 << 40, 53);
    let b = BigFloat::from_f64(3.0, 53).expect("3.0");

    let r = &a % &b;

    assert!(r >= BigFloat::zero(53), "remainder must be >= 0");
    assert!(r < b, "remainder must be < |b|");
    assert_eq!(
        r.to_f64(),
        1.0,
        "2^(2^40) mod 3 must equal 1 (even exponent)"
    );
}

/// Odd-exponent sibling: `2^k mod 3 == 2` for odd `k`.
///
/// `2^40 - 1` rather than `2^40 + 1`: the latter is one bit above
/// `BigFloat::EMAX`, so the saturating `from_parts` would clamp it back down to
/// the (even) `2^40` and the test would silently stop testing an odd exponent.
#[test]
fn rem_huge_odd_exponent_gap_is_bounded_and_exact() {
    let a = pow2((1i64 << 40) - 1, 53);
    assert_eq!(a.ilogb(), Some((1i64 << 40) - 1), "must stay in range");
    let b = BigFloat::from_f64(3.0, 53).expect("3.0");

    let r = &a % &b;

    assert_eq!(
        r.to_f64(),
        2.0,
        "2^(2^40 - 1) mod 3 must equal 2 (odd exponent)"
    );
}

/// The modular route must agree bit-for-bit with the direct shift on the same
/// inputs, so sweep a range of exponents across the direct/modular threshold.
#[test]
fn rem_modular_route_matches_direct_route() {
    let b = BigFloat::from_f64(3.0, 53).expect("3.0");
    for k in 0i64..400 {
        let a = pow2(k, 53);
        let r = &a % &b;
        let expected = if k % 2 == 0 { 1.0 } else { 2.0 };
        assert_eq!(r.to_f64(), expected, "2^{k} mod 3");
    }
}

/// Huge *negative* gap: the divisor dwarfs the dividend, so the quotient is 0
/// and the remainder is the dividend itself. The naive route shifted the
/// divisor mantissa left by the full gap first.
#[test]
fn rem_huge_negative_exponent_gap_is_bounded() {
    let a = BigFloat::from_f64(1.0, 53).expect("1.0");
    let b = pow2(1i64 << 40, 53);

    let r = &a % &b;

    assert_eq!(r.to_f64(), 1.0, "1 mod 2^(2^40) == 1");
}

/// Exponents at the extremes of `i64` must not overflow the gap arithmetic.
#[test]
fn rem_extreme_exponents_do_not_panic() {
    let hi = pow2(i64::MAX, 53);
    let lo = pow2(i64::MIN, 53);
    let three = BigFloat::from_f64(3.0, 53).expect("3.0");

    let _ = &hi % &three;
    let _ = &lo % &three;
    let _ = &hi % &lo;
    let _ = &lo % &hi;
    let _ = &three % &lo;
}

// ---------------------------------------------------------------------------
// 2. exp / sin / cos via binary splitting
// ---------------------------------------------------------------------------

/// `exp(2^-1_000_000_000)` at a precision above `BS_THRESHOLD_BITS` used to
/// request a 10^9-bit denominator inside `split_arg`. The value is far below
/// the working-precision ULP, so the result is exactly `1`.
#[test]
fn exp_of_negligible_argument_is_bounded() {
    let tiny = pow2(-1_000_000_000, 1024);
    let y = tiny.exp(1024, MODE).expect("exp of a tiny positive value");
    assert_eq!(y.to_f64(), 1.0);

    let tiny_neg = BigFloat::from_parts(Sign::Negative, BigUint::one(), -1_000_000_000, 1024, MODE);
    let y_neg = tiny_neg
        .exp(1024, MODE)
        .expect("exp of a tiny negative value");
    assert_eq!(y_neg.to_f64(), 1.0);
}

/// Same guard for the trigonometric binary-splitting path.
#[test]
fn sin_cos_of_negligible_argument_are_bounded() {
    let tiny = pow2(-1_000_000_000, 1024);

    let s = tiny.sin(1024, MODE).expect("sin of a tiny value");
    let c = tiny.cos(1024, MODE).expect("cos of a tiny value");

    assert_eq!(s.to_f64(), 0.0, "sin(2^-1e9) underflows f64 to +0");
    assert_eq!(c.to_f64(), 1.0, "cos(2^-1e9) rounds to 1");
    assert!(s > BigFloat::zero(1024), "sin(x) keeps the sign of x");
    assert!(c <= BigFloat::from_f64(1.0, 1024).expect("1.0"));
}

/// The negligible-argument shortcut must not perturb ordinary arguments: the
/// binary-splitting path is still taken and still agrees with the low-precision
/// Taylor path.
#[test]
fn transcendentals_at_normal_arguments_are_unchanged() {
    let one = BigFloat::from_f64(1.0, 1024).expect("1.0");
    let e_hi = one.exp(1024, MODE).expect("exp(1) at 1024 bits");
    let e_lo = BigFloat::from_f64(1.0, 64)
        .expect("1.0")
        .exp(64, MODE)
        .expect("exp(1) at 64 bits");
    assert!((e_hi.to_f64() - core::f64::consts::E).abs() < 1e-12);
    assert!((e_lo.to_f64() - core::f64::consts::E).abs() < 1e-12);

    let half = BigFloat::from_f64(0.5, 1024).expect("0.5");
    let s = half.sin(1024, MODE).expect("sin(0.5)");
    let c = half.cos(1024, MODE).expect("cos(0.5)");
    assert!((s.to_f64() - 0.5f64.sin()).abs() < 1e-12);
    assert!((c.to_f64() - 0.5f64.cos()).abs() < 1e-12);
}

/// The negligible-argument shortcut must stay rounding-mode faithful: the
/// binary-splitting path it replaces was, and a shortcut that collapsed to a
/// mode-insensitive `1` would be silently wrong under directed rounding.
///
/// For tiny `y > 0`: `exp(y) > 1` so `ToInf` must give the *successor* of 1
/// while `ToZero` gives 1; `cos(y) < 1` so `ToZero` must give the
/// *predecessor* of 1 while `ToInf` gives 1.
#[test]
fn negligible_argument_shortcut_respects_rounding_mode() {
    let one_hex = "0x1p0";

    let up = pow2(-1_000_000_000, 1024);
    let exp_up = up
        .exp(1024, RoundingMode::ToInf)
        .expect("exp under ToInf")
        .to_hex_string();
    let exp_down = up
        .exp(1024, RoundingMode::ToZero)
        .expect("exp under ToZero")
        .to_hex_string();
    assert_eq!(exp_down, one_hex, "exp(tiny) truncates to exactly 1");
    assert_ne!(
        exp_up, exp_down,
        "exp(tiny) > 1, so ToInf must round up to the successor of 1"
    );
    assert!(
        exp_up.starts_with("0x1.") && exp_up.ends_with("2p0"),
        "expected the successor of 1, got {exp_up}"
    );

    let cos_up = up
        .cos(1024, RoundingMode::ToInf)
        .expect("cos under ToInf")
        .to_hex_string();
    let cos_down = up
        .cos(1024, RoundingMode::ToZero)
        .expect("cos under ToZero")
        .to_hex_string();
    assert_eq!(cos_up, one_hex, "cos(tiny) rounds up to exactly 1");
    assert_ne!(
        cos_up, cos_down,
        "cos(tiny) < 1, so ToZero must round down to the predecessor of 1"
    );
    assert!(
        cos_down.ends_with("ep-1"),
        "expected the predecessor of 1, got {cos_down}"
    );

    // sin(tiny) < tiny, so directed-down rounding lands one ULP lower.
    let sin_near = up.sin(1024, RoundingMode::ToInf).expect("sin under ToInf");
    let sin_down = up
        .sin(1024, RoundingMode::ToZero)
        .expect("sin under ToZero");
    assert_ne!(
        sin_near.exponent(),
        sin_down.exponent(),
        "sin(tiny) must keep its magnitude ordering across directed modes"
    );
}

/// Exponents saturating at `i64::MIN` used to overflow `-e` inside `split_arg`.
#[test]
fn transcendentals_at_extreme_exponents_do_not_panic() {
    let lo = pow2(i64::MIN, 1024);
    assert_eq!(lo.exp(1024, MODE).expect("exp").to_f64(), 1.0);
    assert_eq!(lo.sin(1024, MODE).expect("sin").to_f64(), 0.0);
    assert_eq!(lo.cos(1024, MODE).expect("cos").to_f64(), 1.0);
}

// ---------------------------------------------------------------------------
// 3. Untrusted-string entry point
// ---------------------------------------------------------------------------

/// Exponents at the very edge of `i64` are far outside `[EMIN, EMAX]`
/// (`+-2^40`, see `BigFloat::EMAX`/`BigFloat::EMIN`), so `from_hex_float` —
/// the crate's one untrusted-string entry point — must reject them outright
/// with `OxiNumError::Overflow` rather than silently saturating into a
/// different, much smaller, number. This is defense in depth alongside the
/// bounded-path fixes exercised below: the hostile string never even
/// produces a `BigFloat` to feed into `%`/`sin`/`cos`/`exp`.
#[test]
fn hex_float_beyond_emax_emin_is_rejected_at_parse_time() {
    for s in ["0x1p-9223372036854775807", "0x1p9223372036854775807"] {
        let err = BigFloat::from_hex_float(s, 53).expect_err(&format!(
            "{s} is far outside [EMIN, EMAX] and must be rejected"
        ));
        assert!(
            matches!(err, OxiNumError::Overflow(_)),
            "expected Overflow for {s}, got {err:?}"
        );
    }
}

/// The whole chain from a short hostile-but-in-range string through the
/// fixed operations. `+-2_000_000_000` sits comfortably inside
/// `[EMIN, EMAX] = [-2^40, 2^40]` (~+-1.1e12), so these parse successfully and
/// must flow through every bounded operation without panicking.
#[test]
fn hex_float_extreme_in_range_exponents_flow_through_bounded_paths() {
    for s in ["0x1p2000000000", "0x1p-2000000000"] {
        let x = BigFloat::from_hex_float(s, 53).unwrap_or_else(|e| panic!("parse {s}: {e}"));
        let three = BigFloat::from_f64(3.0, 53).expect("3.0");
        let _ = &x % &three;
        let _ = &three % &x;
        let _ = &x + &three;
        let _ = x.sin(1024, MODE);
        let _ = x.cos(1024, MODE);
        let _ = x.exp(1024, MODE);
    }
}

/// The saturating constructor still admits the extremes of the range, so the
/// same operations must stay bounded on values built that way — `from_parts`
/// clamps `i64::MAX` / `i64::MIN` to `ilogb == EMAX` / `EMIN`, which is still a
/// `2^41`-bit exponent gap between the two.
#[test]
fn saturated_extreme_values_flow_through_bounded_paths() {
    let hi = pow2(i64::MAX, 53);
    let lo = pow2(i64::MIN, 53);
    assert_eq!(hi.ilogb(), Some(BigFloat::EMAX));
    assert_eq!(lo.ilogb(), Some(BigFloat::EMIN));

    for x in [&hi, &lo] {
        let three = BigFloat::from_f64(3.0, 53).expect("3.0");
        let _ = x % &three;
        let _ = &three % x;
        let _ = x + &three;
        let _ = x.sin(1024, MODE);
        let _ = x.cos(1024, MODE);
        let _ = x.exp(1024, MODE);
    }
    let _ = &hi % &lo;
    let _ = &lo % &hi;
}

// ---------------------------------------------------------------------------
// 4. Exact BigFloat -> BigInt conversions
// ---------------------------------------------------------------------------

/// `to_bigint_*` on a value whose exact integer form needs ~137 GB must report
/// the budget rather than request the allocation.
#[test]
fn to_bigint_over_budget_is_reported_not_allocated() {
    let huge = pow2(1i64 << 40, 53);

    let err = huge
        .try_to_bigint_trunc()
        .expect_err("2^(2^40) is far over the conversion budget");
    assert!(
        matches!(err, OxiNumError::Overflow(_)),
        "expected Overflow, got {err:?}"
    );

    assert!(huge.try_to_bigint_floor().is_err());
    assert!(huge.try_to_bigint_ceil().is_err());
    assert!(huge.try_to_bigint_round().is_err());
}

/// Values inside the budget keep their exact previous behaviour, including the
/// non-finite lossy fallback.
#[test]
fn to_bigint_within_budget_is_unchanged() {
    let x = BigFloat::from_f64(3.7, 64).expect("3.7");
    assert_eq!(x.to_bigint_trunc(), BigInt::from(3i64));
    assert_eq!(x.to_bigint_floor(), BigInt::from(3i64));
    assert_eq!(x.to_bigint_ceil(), BigInt::from(4i64));
    assert_eq!(x.to_bigint_round(), BigInt::from(4i64));

    let y = BigFloat::from_f64(-3.7, 64).expect("-3.7");
    assert_eq!(y.to_bigint_trunc(), BigInt::from(-3i64));
    assert_eq!(y.to_bigint_floor(), BigInt::from(-4i64));
    assert_eq!(y.to_bigint_ceil(), BigInt::from(-3i64));
    assert_eq!(y.to_bigint_round(), BigInt::from(-4i64));

    assert_eq!(
        BigFloat::nan(53).try_to_bigint_trunc().expect("nan"),
        BigInt::zero()
    );
    assert_eq!(
        BigFloat::infinity(53).try_to_bigint_trunc().expect("inf"),
        BigInt::zero()
    );

    // A large-but-affordable exponent still converts exactly.
    let ok = pow2(1000, 53);
    assert_eq!(
        ok.try_to_bigint_trunc().expect("1000 bits is in budget"),
        BigInt::from_parts(Sign::Positive, BigUint::one().shl_bits(1000))
    );
}

// ---------------------------------------------------------------------------
// 5. Decimal string formatting
// ---------------------------------------------------------------------------

/// `to_scientific_string` / `to_engineering_string` build exact big-integer
/// scaffolding ~`|exponent|` bits wide and then *multiply* two such operands.
/// Beyond the budget they must fall back to the exact hex rendering instead of
/// hanging / exhausting memory.
#[test]
fn decimal_formatting_over_budget_falls_back_to_hex() {
    let huge = pow2(1i64 << 40, 53);
    assert_eq!(huge.to_scientific_string(6), huge.to_hex_string());
    assert_eq!(huge.to_engineering_string(6), huge.to_hex_string());

    let tiny = pow2(-(1i64 << 40), 53);
    assert_eq!(tiny.to_scientific_string(6), tiny.to_hex_string());

    let lo = pow2(i64::MIN, 53);
    assert_eq!(lo.to_scientific_string(6), lo.to_hex_string());

    // The fallback round-trips exactly, so nothing is lost.
    let back = BigFloat::from_hex_float(&huge.to_scientific_string(6), 53).expect("round-trip");
    assert_eq!(back, huge);
}

/// In-budget values keep their decimal rendering unchanged.
#[test]
fn decimal_formatting_within_budget_is_unchanged() {
    let x = BigFloat::from_i64(12345, 64, MODE);
    assert_eq!(x.to_scientific_string(5), "1.2345e4");
    assert_eq!(x.to_engineering_string(5), "12.345e3");

    let big_but_ok = pow2(100_000, 64);
    assert!(
        big_but_ok.to_scientific_string(6).contains('e'),
        "2^100000 is well inside the budget and must still render in decimal"
    );
}

/// `exponent == i64::MIN` used to overflow the `-self.exponent` negation in
/// `to_bigint_trunc` / `has_fractional_part` (a debug-build panic).
#[test]
fn to_bigint_at_min_exponent_does_not_panic() {
    let lo = pow2(i64::MIN, 53);
    assert_eq!(lo.to_bigint_trunc(), BigInt::zero());
    assert_eq!(lo.to_bigint_floor(), BigInt::zero());
    assert_eq!(lo.to_bigint_ceil(), BigInt::from(1i64));
    assert_eq!(lo.to_bigint_round(), BigInt::zero());
}
