//! Regression tests for the exact `BigFloat` → `BigRational` conversion.
//!
//! `float_to_rational` used to shift by the full (free `i64`) exponent, so a
//! valid value such as `2^(2^40)` requested a ~137 GB numerator and killed the
//! process on allocation failure. Exponents saturating at `i64::MIN`
//! additionally overflowed the `-exponent` negation.

use oxinum_core::{OxiNumError, Sign};
use oxinum_float::native::{BigFloat, RoundingMode};
use oxinum_int::native::{BigInt, BigUint};
use oxinum_rational::native::{float_to_rational, try_float_to_rational, BigRational};

const MODE: RoundingMode = RoundingMode::HalfEven;

fn pow2(exponent: i64, prec: u32) -> BigFloat {
    BigFloat::from_parts(Sign::Positive, BigUint::one(), exponent, prec, MODE)
}

#[test]
fn huge_positive_exponent_is_reported_not_allocated() {
    let huge = pow2(1i64 << 40, 53);
    let err = try_float_to_rational(&huge).expect_err("numerator is far over budget");
    assert!(
        matches!(err, OxiNumError::Overflow(_)),
        "expected Overflow, got {err:?}"
    );
}

#[test]
fn huge_negative_exponent_is_reported_not_allocated() {
    let tiny = pow2(-(1i64 << 40), 53);
    let err = try_float_to_rational(&tiny).expect_err("denominator is far over budget");
    assert!(
        matches!(err, OxiNumError::Overflow(_)),
        "expected Overflow, got {err:?}"
    );
}

/// `exponent == i64::MIN` used to overflow `-exponent`.
#[test]
fn min_exponent_does_not_panic() {
    let lo = pow2(i64::MIN, 53);
    assert!(try_float_to_rational(&lo).is_err());
}

/// A short hostile hex-float string must not be able to drive the allocation.
///
/// Two layers now stand in the way, and both are exercised: an exponent beyond
/// `BigFloat::EMAX` never becomes a `BigFloat` at all, and one that is inside
/// the exponent range but still over the conversion bit budget is reported by
/// `try_float_to_rational`.
#[test]
fn hex_float_extreme_exponent_is_reported() {
    // Layer 1 — outside [EMIN, EMAX]: refused by the parser.
    for s in ["0x1p-9223372036854775807", "0x1p9223372036854775807"] {
        let err = BigFloat::from_hex_float(s, 53)
            .expect_err("an exponent outside [EMIN, EMAX] must be refused at parse time");
        assert!(
            matches!(err, OxiNumError::Overflow(_)),
            "expected Overflow for {s}, got {err:?}"
        );
    }

    // Layer 2 — inside the exponent range, over the conversion budget.
    for s in ["0x1p2000000000", "0x1p-2000000000"] {
        let x = BigFloat::from_hex_float(s, 53).unwrap_or_else(|e| panic!("parse {s}: {e}"));
        assert!(
            try_float_to_rational(&x).is_err(),
            "{s} should be reported as over budget"
        );
    }
}

/// In-budget conversions are bit-for-bit unchanged.
#[test]
fn within_budget_conversions_are_unchanged() {
    let f = BigFloat::from_f64(1.5, 64).expect("1.5");
    let expected =
        BigRational::from_parts(BigInt::from(3i64), BigUint::from_u64(2)).expect("3/2 is valid");
    assert_eq!(float_to_rational(&f), expected);
    assert_eq!(try_float_to_rational(&f).expect("in budget"), expected);

    let zero = BigFloat::zero(64);
    assert_eq!(float_to_rational(&zero), BigRational::zero());

    let neg = BigFloat::from_f64(-0.25, 64).expect("-0.25");
    let expected_neg =
        BigRational::from_parts(BigInt::from(-1i64), BigUint::from_u64(4)).expect("-1/4 is valid");
    assert_eq!(float_to_rational(&neg), expected_neg);

    // Large-but-affordable exponents still convert exactly.
    let ok = pow2(1000, 53);
    let r = try_float_to_rational(&ok).expect("1000 bits is in budget");
    assert_eq!(
        r,
        BigRational::from_integer(BigInt::from_parts(
            Sign::Positive,
            BigUint::one().shl_bits(1000)
        ))
    );
}
