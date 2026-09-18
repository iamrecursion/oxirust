//! `BigFloat` exponent range (`EMIN` / `EMAX`) enforced at construction.
//!
//! Before this range existed, `BigFloat::from_parts` accepted any `i64`
//! exponent, so `2^(2^62)` was an ordinary value and every operation that
//! shifted by an exponent had to defend itself individually. The range is the
//! structural half of that defence:
//!
//! * [`BigFloat::try_from_parts`] *rejects* a value whose binary magnitude
//!   (`ilogb`, the position of the leading set bit) falls outside
//!   `[EMIN, EMAX]`, returning `OxiNumError::Overflow`;
//! * [`BigFloat::from_parts`] *saturates* to the boundary so that the total,
//!   non-`Result` operators stay total;
//! * the untrusted-string entry point (`from_hex_float`) takes the rejecting
//!   path.
//!
//! The per-operation bit budgets (`MAX_EXACT_CONVERSION_BITS` and friends) are
//! the complementary half and are covered by `regression_unbounded_shift.rs` /
//! `regression_float_to_rational_budget.rs`: `EMAX` bounds what a value may
//! *be*, the budgets bound what one operation may *materialize*.

use oxinum_core::{OxiNumError, Sign};
use oxinum_float::native::{BigFloat, RoundingMode};
use oxinum_int::native::BigUint;

const MODE: RoundingMode = RoundingMode::HalfEven;

fn try_pow2(exponent: i64, prec: u32) -> Result<BigFloat, OxiNumError> {
    BigFloat::try_from_parts(Sign::Positive, BigUint::one(), exponent, prec, MODE)
}

fn pow2(exponent: i64, prec: u32) -> BigFloat {
    BigFloat::from_parts(Sign::Positive, BigUint::one(), exponent, prec, MODE)
}

// ---------------------------------------------------------------------------
// The constants themselves
// ---------------------------------------------------------------------------

#[test]
fn emax_and_emin_are_the_documented_bounds() {
    assert_eq!(BigFloat::EMAX, 1i64 << 40);
    assert_eq!(BigFloat::EMIN, -(1i64 << 40));
    // Symmetric, and the full span fits in i64 with room to spare — that is
    // what makes the exponent-gap arithmetic in add/div overflow-free.
    assert_eq!(BigFloat::EMAX, -BigFloat::EMIN);
    assert!(BigFloat::EMAX.checked_sub(BigFloat::EMIN).is_some());
}

// ---------------------------------------------------------------------------
// ilogb
// ---------------------------------------------------------------------------

#[test]
fn ilogb_is_the_leading_bit_position() {
    for (value, expected) in [
        (1.0f64, 0i64),
        (1.5, 0),
        (2.0, 1),
        (3.999, 1),
        (4.0, 2),
        (0.5, -1),
        (0.25, -2),
        (-8.0, 3),
    ] {
        let x = BigFloat::from_f64(value, 53).expect("finite f64");
        assert_eq!(x.ilogb(), Some(expected), "ilogb({value})");
    }
}

#[test]
fn ilogb_is_none_for_zero_and_non_finite() {
    assert_eq!(BigFloat::zero(53).ilogb(), None);
    assert_eq!(BigFloat::nan(53).ilogb(), None);
    assert_eq!(BigFloat::infinity(53).ilogb(), None);
    assert_eq!(BigFloat::neg_infinity(53).ilogb(), None);
}

#[test]
fn ilogb_is_invariant_under_precision_change() {
    // Padding lowers the raw exponent by exactly the bits it adds, so the
    // magnitude — and hence the value — is unchanged. This is why the range is
    // stated in terms of ilogb rather than of the raw exponent field.
    let x = BigFloat::from_f64(3.5, 8).expect("3.5");
    let wide = x.clone().with_precision(200, MODE);
    assert_eq!(x.ilogb(), wide.ilogb());
    assert_ne!(x.exponent(), wide.exponent());
    assert_eq!(wide.to_f64(), 3.5);
}

// ---------------------------------------------------------------------------
// try_from_parts: boundaries
// ---------------------------------------------------------------------------

#[test]
fn try_from_parts_accepts_both_boundaries_exactly() {
    let top = try_pow2(BigFloat::EMAX, 53).expect("2^EMAX is representable");
    assert_eq!(top.ilogb(), Some(BigFloat::EMAX));
    assert!(top.is_finite());
    assert_eq!(top.mantissa().bit_length(), 53, "still normalized");

    let bottom = try_pow2(BigFloat::EMIN, 53).expect("2^EMIN is representable");
    assert_eq!(bottom.ilogb(), Some(BigFloat::EMIN));
    assert!(bottom.is_finite());
    assert_eq!(bottom.mantissa().bit_length(), 53);
}

#[test]
fn try_from_parts_rejects_one_bit_beyond_each_boundary() {
    let over = try_pow2(BigFloat::EMAX + 1, 53).expect_err("2^(EMAX+1) is out of range");
    assert!(
        matches!(over, OxiNumError::Overflow(_)),
        "expected Overflow, got {over:?}"
    );
    assert!(
        over.to_string().contains("overflow"),
        "the message must name the direction: {over}"
    );

    let under = try_pow2(BigFloat::EMIN - 1, 53).expect_err("2^(EMIN-1) is out of range");
    assert!(
        matches!(under, OxiNumError::Overflow(_)),
        "expected Overflow, got {under:?}"
    );
    assert!(
        under.to_string().contains("underflow"),
        "the message must name the direction: {under}"
    );
}

#[test]
fn try_from_parts_measures_magnitude_not_the_raw_exponent() {
    // A wide mantissa pushes the magnitude up even at a small exponent: with
    // 65 mantissa bits, exponent EMAX - 64 already lands exactly on EMAX.
    let mantissa = BigUint::one().shl_bits(64); // 65 bits: 2^64
    let at_top = BigFloat::try_from_parts(
        Sign::Positive,
        mantissa.clone(),
        BigFloat::EMAX - 64,
        53,
        MODE,
    )
    .expect("magnitude is exactly EMAX");
    assert_eq!(at_top.ilogb(), Some(BigFloat::EMAX));

    let over = BigFloat::try_from_parts(Sign::Positive, mantissa, BigFloat::EMAX - 63, 53, MODE)
        .expect_err("one bit further is out of range");
    assert!(matches!(over, OxiNumError::Overflow(_)));

    // The same exponent with a 1-bit mantissa is comfortably inside the range,
    // proving the check is not a bare `|exponent| <= EMAX`.
    assert!(try_pow2(BigFloat::EMAX - 63, 53).is_ok());
}

#[test]
fn try_from_parts_accepts_any_zero_and_reports_zero_precision() {
    // Zero has no magnitude, so no exponent can put it out of range.
    for exponent in [0, i64::MAX, i64::MIN] {
        let z = BigFloat::try_from_parts(Sign::Positive, BigUint::zero(), exponent, 53, MODE)
            .expect("zero is always representable");
        assert!(z.is_zero());
        assert_eq!(z.exponent(), 0, "canonical zero");
    }

    // prec == 0 is reported instead of asserted (from_parts panics).
    let err = BigFloat::try_from_parts(Sign::Positive, BigUint::one(), 0, 0, MODE)
        .expect_err("precision 0 is invalid");
    assert!(
        matches!(err, OxiNumError::Precision(_)),
        "expected Precision, got {err:?}"
    );
}

#[test]
fn try_from_parts_matches_from_parts_inside_the_range() {
    for exponent in [-1000i64, -3, 0, 1, 7, 1000, 1_000_000] {
        for prec in [1u32, 8, 53, 200] {
            let checked = try_pow2(exponent, prec).expect("in range");
            let saturating = pow2(exponent, prec);
            assert_eq!(checked, saturating);
            assert_eq!(checked.exponent(), saturating.exponent());
            assert_eq!(checked.precision(), saturating.precision());
        }
    }
}

// ---------------------------------------------------------------------------
// from_parts: saturation
// ---------------------------------------------------------------------------

#[test]
fn from_parts_saturates_instead_of_wrapping_or_panicking() {
    for exponent in [BigFloat::EMAX + 1, 1i64 << 50, i64::MAX] {
        let x = pow2(exponent, 53);
        assert!(x.is_finite(), "saturation keeps the value finite");
        assert_eq!(
            x.ilogb(),
            Some(BigFloat::EMAX),
            "exponent {exponent} must clamp to EMAX"
        );
        assert_eq!(x.mantissa().bit_length(), 53, "invariants still hold");
    }

    for exponent in [BigFloat::EMIN - 1, -(1i64 << 50), i64::MIN] {
        let x = pow2(exponent, 53);
        assert!(x.is_finite());
        assert_eq!(
            x.ilogb(),
            Some(BigFloat::EMIN),
            "exponent {exponent} must clamp to EMIN"
        );
        assert_eq!(x.mantissa().bit_length(), 53);
    }
}

#[test]
fn from_parts_saturation_preserves_sign_and_mantissa_bits() {
    let mantissa = BigUint::from_u64(0b1101);
    let x = BigFloat::from_parts(Sign::Negative, mantissa, i64::MAX, 4, MODE);
    assert_eq!(x.sign(), Sign::Negative);
    assert_eq!(x.mantissa(), &BigUint::from_u64(0b1101));
    assert_eq!(x.ilogb(), Some(BigFloat::EMAX));
    assert_eq!(x.precision(), 4);
}

#[test]
fn saturated_values_still_round_trip_through_hex() {
    // Saturation lands on a value the parser accepts, so the hex round-trip
    // (which reparses the printed exponent) is not disturbed by it.
    let x = pow2(i64::MAX, 53);
    let s = x.to_hex_string();
    let back = BigFloat::from_hex_float(&s, 53).unwrap_or_else(|e| panic!("re-parse {s}: {e}"));
    assert_eq!(back, x);
    assert_eq!(back.exponent(), x.exponent());
}

// ---------------------------------------------------------------------------
// Entry points that take the rejecting path
// ---------------------------------------------------------------------------

#[test]
fn from_hex_float_enforces_the_range() {
    // In range: parses.
    let ok = BigFloat::from_hex_float("0x1p1099511627776", 53).expect("2^EMAX parses");
    assert_eq!(ok.ilogb(), Some(BigFloat::EMAX));

    // One bit further: refused rather than saturated into a different number.
    for s in [
        "0x1p1099511627777",
        "0x1p-1099511627777",
        "0x1p9223372036854775807",
        "0x1p-9223372036854775807",
    ] {
        let err = BigFloat::from_hex_float(s, 53)
            .err()
            .unwrap_or_else(|| panic!("{s} must not parse"));
        assert!(
            matches!(err, OxiNumError::Overflow(_)),
            "expected Overflow for {s}, got {err:?}"
        );
        // The bound must be named, so an operator can tell an out-of-range
        // literal from a malformed one.
        assert!(
            err.to_string().contains("EMAX") || err.to_string().contains("EMIN"),
            "message must name the bound: {err}"
        );
    }

    // A fractional significand shifts the magnitude: `0x2p...` is one bit
    // higher than `0x1p...`, and the check must see that.
    assert!(BigFloat::from_hex_float("0x2p1099511627775", 53).is_ok());
    assert!(BigFloat::from_hex_float("0x2p1099511627776", 53).is_err());
}

#[test]
fn from_f64_is_unaffected() {
    // f64 magnitudes span [-1074, 1024], nowhere near the bound.
    for v in [f64::MIN_POSITIVE, 1.0, -2.5, f64::MAX, 5e-324] {
        let x = BigFloat::from_f64(v, 64).unwrap_or_else(|e| panic!("from_f64({v}): {e}"));
        assert_eq!(x.to_f64(), v);
        let l = x.ilogb().expect("non-zero");
        assert!(l > BigFloat::EMIN && l < BigFloat::EMAX);
    }
}

// ---------------------------------------------------------------------------
// The range must not disturb ordinary arithmetic
// ---------------------------------------------------------------------------

#[test]
fn arithmetic_near_the_boundary_stays_finite_and_bounded() {
    let top = try_pow2(BigFloat::EMAX, 64).expect("2^EMAX");
    let two = BigFloat::from_f64(2.0, 64).expect("2.0");

    // Doubling 2^EMAX overflows the range: the operator is total, so it
    // saturates (documented on `from_parts`) rather than panicking.
    let doubled = &top * &two;
    assert!(doubled.is_finite());
    assert_eq!(doubled.ilogb(), Some(BigFloat::EMAX));

    // Halving stays inside and is exact.
    let halved = &top / &two;
    assert_eq!(halved.ilogb(), Some(BigFloat::EMAX - 1));

    // sqrt of the largest value is exact and well inside the range.
    let root = top.sqrt(64, MODE).expect("sqrt(2^EMAX)");
    assert_eq!(root.ilogb(), Some(BigFloat::EMAX / 2));

    // ln of the largest value: ~ EMAX * ln 2.
    let l = top.ln(64, MODE).expect("ln(2^EMAX)");
    let expected = (BigFloat::EMAX as f64) * core::f64::consts::LN_2;
    assert!(
        (l.to_f64() - expected).abs() / expected < 1e-12,
        "ln(2^EMAX) = {} vs {expected}",
        l.to_f64()
    );
}

#[test]
fn transcendentals_of_the_smallest_value_are_correct() {
    // 2^EMIN is far below the working ULP: exp -> 1, sin -> the argument
    // itself, cos -> 1. The negligible-argument shortcut must not cancel the
    // argument to zero when the tail it would subtract is itself unrepresentable.
    let tiny = try_pow2(BigFloat::EMIN, 1024).expect("2^EMIN");

    let e = tiny.exp(1024, MODE).expect("exp(2^EMIN)");
    assert_eq!(e.to_f64(), 1.0);

    let s = tiny.sin(1024, MODE).expect("sin(2^EMIN)");
    assert!(s > BigFloat::zero(1024), "sin(x) > 0 for tiny x > 0");
    assert_eq!(
        s.ilogb(),
        Some(BigFloat::EMIN),
        "sin(x) ~ x, not a cancelled zero"
    );

    let c = tiny.cos(1024, MODE).expect("cos(2^EMIN)");
    assert_eq!(c.to_f64(), 1.0);
    assert!(c <= BigFloat::from_f64(1.0, 1024).expect("1.0"));
}

#[test]
fn ordinary_values_are_bit_for_bit_unchanged() {
    // A broad sanity sweep: the range check must be invisible for everything a
    // real program computes with.
    let a = BigFloat::from_f64(1.0, 128).expect("1.0");
    let b = BigFloat::from_f64(3.0, 128).expect("3.0");
    let third = a.div_ref(&b).expect("1/3");
    assert!((third.to_f64() - 1.0 / 3.0).abs() < 1e-15);
    assert_eq!((&b * &third).to_f64(), 1.0);
    assert_eq!((&b % &two_f()).to_f64(), 1.0);
    assert_eq!(b.sqrt(128, MODE).expect("sqrt 3").to_f64(), 3.0f64.sqrt());
    assert!((a.exp(128, MODE).expect("e").to_f64() - core::f64::consts::E).abs() < 1e-15);
}

fn two_f() -> BigFloat {
    BigFloat::from_f64(2.0, 128).expect("2.0")
}

// ---------------------------------------------------------------------------
// Deserialization is the second untrusted-input boundary
// ---------------------------------------------------------------------------

#[cfg(feature = "serde")]
mod serde_boundary {
    use super::*;
    use serde_json::Value;

    /// A hand-crafted record carrying an exponent outside `[EMIN, EMAX]` must
    /// be refused. The wire format is a flat struct, so nothing stops a peer
    /// from sending `exponent: i64::MAX`; if `TryFrom<BigFloatRepr>` built the
    /// value directly it would reintroduce the unbounded exponent the range
    /// exists to prevent.
    #[test]
    fn deserializing_an_out_of_range_exponent_is_rejected() {
        let valid = BigFloat::from_f64(1.0, 53).expect("1.0");
        for exponent in [i64::MAX, i64::MIN, BigFloat::EMAX + 1] {
            let mut val: Value = serde_json::to_value(&valid).expect("serialize");
            val["exponent"] = Value::from(exponent);
            let result: Result<BigFloat, _> = serde_json::from_value(val);
            let err = result
                .err()
                .unwrap_or_else(|| panic!("exponent {exponent} must be rejected"));
            let msg = err.to_string();
            assert!(
                msg.contains("EMAX") || msg.contains("EMIN"),
                "the rejection must name the bound, got {msg}"
            );
        }
    }

    /// In-range records — including both boundaries — still round-trip
    /// bit-for-bit, so the new check cannot reject anything this crate emits.
    #[test]
    fn in_range_records_round_trip_unchanged() {
        let cases = [
            BigFloat::from_f64(1.0, 53).expect("1.0"),
            BigFloat::from_f64(-0.125, 53).expect("-0.125"),
            BigFloat::zero(53),
            BigFloat::nan(53),
            BigFloat::infinity(53),
            try_pow2(BigFloat::EMAX, 53).expect("2^EMAX"),
            try_pow2(BigFloat::EMIN, 53).expect("2^EMIN"),
        ];
        for x in cases {
            let json = serde_json::to_string(&x).expect("serialize");
            let back: BigFloat = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(back.exponent(), x.exponent(), "{json}");
            assert_eq!(back.precision(), x.precision(), "{json}");
            assert_eq!(back.mantissa(), x.mantissa(), "{json}");
            assert_eq!(back.is_nan(), x.is_nan());
            assert_eq!(back.is_infinite(), x.is_infinite());
        }
    }
}
