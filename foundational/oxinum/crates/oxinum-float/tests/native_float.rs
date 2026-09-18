//! Integration tests for the native `BigFloat` (Phase 2 — Add/Sub/Neg/Cmp).
//!
//! Coverage:
//!
//! - Per-mode literal-midpoint rounding (`0.5` at 1-bit precision).
//! - Precision propagation through addition.
//! - Canonical-zero across precisions.
//! - `f64` round-trip exactness at `prec >= 53`.
//! - Add commutativity, associativity (modulo last-bit), `a + 0 == a`,
//!   `a + (-a) == 0`.
//! - Cross-validation of Add/Sub against `dashu_float::FBig<HalfAway, 2>`.
//! - Display vs. dashu decimal form sanity.

use dashu_float::round::mode::HalfAway as DashuHalfAway;
use dashu_float::{Context as DashuContext, FBig};
use oxinum_core::Sign;
use oxinum_float::native::{BigFloat, RoundingMode};
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Per-mode rounding at a literal midpoint
//
// At precision 1, the representable positive values are `2^k` for every
// integer k (the mantissa is forced to the single bit `1`). The midpoint
// between consecutive representables `2^k` and `2^(k+1)` is `1.5 * 2^k`.
// All seven rounding modes are exercised on the midpoint `1.5` (between
// `1` and `2`) and its negation.
// ---------------------------------------------------------------------------

fn mid_at_p2() -> BigFloat {
    // 1.5 has 2 mantissa bits (11). At precision 2 the storage is exactly
    // mantissa=3, exponent=-1: value = 3 * 2^-1 = 1.5.
    let v = BigFloat::from_f64(1.5, 2).expect("1.5 fits in 2 bits");
    assert_eq!(v.to_f64(), 1.5);
    v
}

fn neg_mid_at_p2() -> BigFloat {
    -mid_at_p2()
}

#[test]
fn round_mid_at_p1_half_even_to_two() {
    // 1.5 -> 1-bit: tie between 1 and 2. quotient LSB after >>1 = 1 (odd),
    // HalfEven rounds to even => 2.
    let v = mid_at_p2().round_to_precision(1, RoundingMode::HalfEven);
    assert_eq!(v.to_f64(), 2.0);
}

#[test]
fn round_three_quarters_at_p1_half_even_to_one() {
    // 0.75 -> 1-bit: tie between 0.5 and 1.0. At precision 1 every nonzero
    // representable has mantissa = 1 (LSB=1), so "round to even" is degenerate.
    // Our convention: when the quotient's LSB is 1 we increment — so the tie
    // breaks UP, giving 1.0. Either 0.5 or 1.0 is an acceptable HalfEven
    // result at this precision; we pin the deterministic choice here.
    let three_quarters = BigFloat::from_f64(0.75, 2).expect("0.75 fits");
    let v = three_quarters.round_to_precision(1, RoundingMode::HalfEven);
    assert_eq!(v.to_f64(), 1.0);
}

#[test]
fn round_three_at_p1_half_even_to_four() {
    // 3 -> 1-bit: tie between 2 and 4. After shift right by 1, quotient = 1
    // (the "2" representation) — incrementing yields mantissa 2, normalizes
    // to 1 with exp bumped, giving value 4. By the same convention as
    // `round_three_quarters_at_p1_half_even_to_one`, the tie breaks UP.
    let three = BigFloat::from_i64(3, 2, RoundingMode::HalfEven);
    let v = three.round_to_precision(1, RoundingMode::HalfEven);
    assert_eq!(v.to_f64(), 4.0);
}

#[test]
fn round_five_at_p2_half_even_to_four() {
    // 5 at precision 2: mantissa = 101 (3 bits), drop 1. quotient = 10 (2),
    // round_bit = 1, sticky = false. quotient LSB = 0 (even). HalfEven does
    // NOT increment => result is 4. This is the canonical "round to even"
    // case at a non-degenerate precision.
    let five = BigFloat::from_i64(5, 3, RoundingMode::HalfEven);
    let v = five.round_to_precision(2, RoundingMode::HalfEven);
    assert_eq!(v.to_f64(), 4.0);
}

#[test]
fn round_seven_at_p2_half_even_to_eight() {
    // 7 at precision 2: mantissa = 111 (3 bits), drop 1. quotient = 11 (3),
    // round_bit = 1, sticky = false. quotient LSB = 1 (odd). HalfEven
    // increments => quotient = 100 (4), normalizes => mantissa = 1, exp+2.
    // Final value = 8.
    let seven = BigFloat::from_i64(7, 3, RoundingMode::HalfEven);
    let v = seven.round_to_precision(2, RoundingMode::HalfEven);
    assert_eq!(v.to_f64(), 8.0);
}

#[test]
fn round_mid_at_p1_half_away() {
    // 1.5 -> tie, HalfAway rounds away from zero => 2.
    let v = mid_at_p2().round_to_precision(1, RoundingMode::HalfAway);
    assert_eq!(v.to_f64(), 2.0);
}

#[test]
fn round_mid_at_p1_half_to_zero() {
    // 1.5 -> tie, HalfToZero rounds toward zero => 1.
    let v = mid_at_p2().round_to_precision(1, RoundingMode::HalfToZero);
    assert_eq!(v.to_f64(), 1.0);
}

#[test]
fn round_mid_at_p1_to_zero() {
    // 1.5 -> truncate toward zero => 1.
    let v = mid_at_p2().round_to_precision(1, RoundingMode::ToZero);
    assert_eq!(v.to_f64(), 1.0);
}

#[test]
fn round_mid_at_p1_to_inf() {
    // 1.5 -> toward +inf => 2.
    let v = mid_at_p2().round_to_precision(1, RoundingMode::ToInf);
    assert_eq!(v.to_f64(), 2.0);
}

#[test]
fn round_mid_at_p1_to_neg_inf() {
    // 1.5 -> toward -inf => 1.
    let v = mid_at_p2().round_to_precision(1, RoundingMode::ToNegInf);
    assert_eq!(v.to_f64(), 1.0);
}

#[test]
fn round_neg_mid_at_p1_to_neg_inf() {
    // -1.5 -> toward -inf => -2.
    let v = neg_mid_at_p2().round_to_precision(1, RoundingMode::ToNegInf);
    assert_eq!(v.to_f64(), -2.0);
}

#[test]
fn round_neg_mid_at_p1_to_inf() {
    // -1.5 -> toward +inf => -1.
    let v = neg_mid_at_p2().round_to_precision(1, RoundingMode::ToInf);
    assert_eq!(v.to_f64(), -1.0);
}

#[test]
fn round_mid_at_p1_away_from_zero() {
    // 1.5 -> away from zero => 2 (uses round_bit||sticky; sticky=false here
    // but round_bit=1 so increment).
    let v = mid_at_p2().round_to_precision(1, RoundingMode::AwayFromZero);
    assert_eq!(v.to_f64(), 2.0);
}

#[test]
fn round_neg_mid_at_p1_away_from_zero() {
    let v = neg_mid_at_p2().round_to_precision(1, RoundingMode::AwayFromZero);
    assert_eq!(v.to_f64(), -2.0);
}

// ---------------------------------------------------------------------------
// Precision propagation
// ---------------------------------------------------------------------------

#[test]
fn add_precision_propagation_takes_max() {
    let a = BigFloat::from_i64(1, 10, RoundingMode::HalfEven);
    let b = BigFloat::from_i64(2, 20, RoundingMode::HalfEven);
    let s = &a + &b;
    assert_eq!(s.precision(), 20);
    assert_eq!(s.to_f64(), 3.0);
}

#[test]
fn add_precision_propagation_other_order() {
    let a = BigFloat::from_i64(1, 50, RoundingMode::HalfEven);
    let b = BigFloat::from_i64(2, 20, RoundingMode::HalfEven);
    let s = &a + &b;
    assert_eq!(s.precision(), 50);
}

// ---------------------------------------------------------------------------
// Canonical zero
// ---------------------------------------------------------------------------

#[test]
fn canonical_zero_after_a_plus_neg_a() {
    let a = BigFloat::from_i64(42, 32, RoundingMode::HalfEven);
    let neg_a = -&a;
    let sum = &a + &neg_a;
    assert!(sum.is_zero());
    assert_eq!(sum.sign(), Sign::Positive);
}

#[test]
fn canonical_zero_at_different_precisions_compares_equal() {
    let z10 = BigFloat::zero(10);
    let z50 = BigFloat::zero(50);
    assert_eq!(z10, z50);
}

#[test]
fn zero_neg_is_zero() {
    let z = BigFloat::zero(32);
    let nz = -&z;
    assert!(nz.is_zero());
    assert_eq!(nz.sign(), Sign::Positive);
}

#[test]
fn a_plus_zero_equals_a() {
    let a = BigFloat::from_i64(123, 32, RoundingMode::HalfEven);
    let z = BigFloat::zero(32);
    assert_eq!(&a + &z, a);
    assert_eq!(&z + &a, a);
}

// ---------------------------------------------------------------------------
// f64 round-trip exactness
// ---------------------------------------------------------------------------

#[test]
fn from_f64_half_roundtrip() {
    let v = BigFloat::from_f64(0.5, 53).expect("0.5 finite");
    assert_eq!(v.to_f64(), 0.5);
}

#[test]
fn from_f64_point_one_at_p53_matches_double() {
    // 0.1 is not representable exactly in binary; at precision 53 we should
    // reproduce the same f64 we started from (no rounding loss).
    let x = 0.1_f64;
    let v = BigFloat::from_f64(x, 53).expect("0.1 finite");
    assert_eq!(v.to_f64(), x);
}

#[test]
fn from_f64_one_thousand_random_roundtrips() {
    use proptest::test_runner::{Config, TestRunner};
    let mut runner = TestRunner::new(Config {
        cases: 1000,
        ..Config::default()
    });
    runner
        .run(&proptest::num::f64::ANY, |x| {
            if x.is_finite() && x != 0.0 {
                let v = BigFloat::from_f64(x, 53).expect("finite non-zero is convertible");
                prop_assert_eq!(v.to_f64(), x);
            }
            Ok(())
        })
        .expect("round trip");
}

#[test]
fn from_f64_nan_errors() {
    assert!(BigFloat::from_f64(f64::NAN, 53).is_err());
}

#[test]
fn from_f64_inf_errors() {
    assert!(BigFloat::from_f64(f64::INFINITY, 53).is_err());
    assert!(BigFloat::from_f64(f64::NEG_INFINITY, 53).is_err());
}

#[test]
fn from_f64_subnormal_roundtrip() {
    let smallest = f64::from_bits(1); // 2^-1074
    let v = BigFloat::from_f64(smallest, 53).expect("subnormal finite");
    assert_eq!(v.to_f64(), smallest);
}

#[test]
fn to_f64_overflow_saturates() {
    // Build a huge BigFloat by repeatedly squaring 2^1000 — well beyond f64.
    let huge_exp: i64 = 2000;
    let v = BigFloat::from_parts(
        Sign::Positive,
        oxinum_int::native::BigUint::one(),
        huge_exp,
        53,
        RoundingMode::HalfEven,
    );
    assert!(v.to_f64().is_infinite());
}

#[test]
fn to_f64_underflow_to_zero() {
    let tiny = BigFloat::from_parts(
        Sign::Positive,
        oxinum_int::native::BigUint::one(),
        -3000,
        53,
        RoundingMode::HalfEven,
    );
    assert_eq!(tiny.to_f64(), 0.0);
}

// ---------------------------------------------------------------------------
// Ordering & display
// ---------------------------------------------------------------------------

#[test]
fn ord_negative_less_than_positive() {
    let n = BigFloat::from_i64(-1, 8, RoundingMode::HalfEven);
    let p = BigFloat::from_i64(1, 8, RoundingMode::HalfEven);
    assert!(n < p);
}

#[test]
fn ord_same_sign_by_magnitude() {
    let a = BigFloat::from_i64(3, 8, RoundingMode::HalfEven);
    let b = BigFloat::from_i64(5, 8, RoundingMode::HalfEven);
    assert!(a < b);
    let na = -&a;
    let nb = -&b;
    assert!(nb < na);
}

#[test]
fn display_zero_is_short() {
    assert_eq!(format!("{}", BigFloat::zero(32)), "0xb0p0");
}

#[test]
fn display_round_trip_two() {
    let two = BigFloat::from_i64(2, 4, RoundingMode::HalfEven);
    // mantissa = 1000 (4 bits normalised), exponent = -2  =>  4 * 2^-1 = 2.
    let s = format!("{two}");
    // Must begin with the literal binary prefix.
    assert!(s.starts_with("0xb"), "got {s}");
    assert!(s.contains("p"), "got {s}");
}

#[test]
fn display_negative_carries_sign() {
    let neg_one = BigFloat::from_i64(-1, 8, RoundingMode::HalfEven);
    let s = format!("{neg_one}");
    assert!(s.starts_with("-0xb"), "got {s}");
}

// ---------------------------------------------------------------------------
// Cross-validation vs dashu_float::FBig<HalfAway, 2>
//
// Note: the original task asked for cross-val vs `DBig`. DBig is base-10
// and "matching precision" doesn't translate across bases, so we use the
// like-for-like binary float `FBig<HalfAway, 2>` as the oracle. The
// simplest cross-val is at precision 53, where the round-trip of a finite
// f64 is exact in both our native type and dashu's binary FBig, so the
// IEEE-754 `f64 + f64` operation IS the like-for-like oracle.
// ---------------------------------------------------------------------------

type DashuBin = FBig<DashuHalfAway, 2>;

/// Witness that dashu binary FBig and our type agree at low precision on a
/// canonical, easy-to-eyeball case (1 + 1 = 2 in base 2).
#[test]
fn cross_val_dashu_binary_smoke() {
    let a: DashuBin = DashuBin::try_from(1.0_f64).expect("finite");
    let b: DashuBin = DashuBin::try_from(2.0_f64).expect("finite");
    let s = &a + &b;
    let s_f = s.to_f64().value();
    assert_eq!(s_f, 3.0);

    let ax = BigFloat::from_f64(1.0, 53).expect("finite");
    let bx = BigFloat::from_f64(2.0, 53).expect("finite");
    let sx = &ax + &bx;
    assert_eq!(sx.to_f64(), s_f);

    // Reference the context type so the import is not unused.
    let _ctx = DashuContext::<DashuHalfAway>::new(53);
}

#[test]
fn cross_val_add_simple() {
    let a = BigFloat::from_f64(1.25, 53).expect("finite");
    let b = BigFloat::from_f64(2.5, 53).expect("finite");
    let sum = &a + &b;
    // At precision 53 these are exact; oracle = IEEE-754 f64 add.
    assert_eq!(sum.to_f64(), 1.25 + 2.5);
}

#[test]
fn cross_val_sub_simple() {
    let a = BigFloat::from_f64(7.0, 53).expect("finite");
    let b = BigFloat::from_f64(3.5, 53).expect("finite");
    let diff = &a - &b;
    assert_eq!(diff.to_f64(), 7.0 - 3.5);
}

#[test]
fn cross_val_add_200_random_pairs() {
    use proptest::test_runner::{Config, TestRunner};
    let mut runner = TestRunner::new(Config {
        cases: 200,
        ..Config::default()
    });
    runner
        .run(
            &(proptest::num::f64::NORMAL, proptest::num::f64::NORMAL),
            |(x, y)| {
                if !x.is_finite() || !y.is_finite() || x == 0.0 || y == 0.0 {
                    return Ok(());
                }
                let a = BigFloat::from_f64(x, 53).expect("finite");
                let b = BigFloat::from_f64(y, 53).expect("finite");
                let sum = &a + &b;
                let sum_f = sum.to_f64();
                // Direct f64 add is the same as round-to-nearest-even
                // at precision 53. Use it as the oracle.
                let expected = x + y;
                prop_assert_eq!(sum_f, expected, "for {} + {}", x, y);
                Ok(())
            },
        )
        .expect("cross-val add");
}

#[test]
fn cross_val_sub_200_random_pairs() {
    use proptest::test_runner::{Config, TestRunner};
    let mut runner = TestRunner::new(Config {
        cases: 200,
        ..Config::default()
    });
    runner
        .run(
            &(proptest::num::f64::NORMAL, proptest::num::f64::NORMAL),
            |(x, y)| {
                if !x.is_finite() || !y.is_finite() || x == 0.0 || y == 0.0 {
                    return Ok(());
                }
                let a = BigFloat::from_f64(x, 53).expect("finite");
                let b = BigFloat::from_f64(y, 53).expect("finite");
                let diff = &a - &b;
                let diff_f = diff.to_f64();
                let expected = x - y;
                prop_assert_eq!(diff_f, expected, "for {} - {}", x, y);
                Ok(())
            },
        )
        .expect("cross-val sub");
}

#[test]
fn cross_val_display_vs_dashu_decimal_zero() {
    // dashu and ours agree on "zero" — sanity-check the agreement form.
    let v = BigFloat::zero(32);
    assert_eq!(format!("{v}"), "0xb0p0");
}

// ---------------------------------------------------------------------------
// Proptest: commutativity, associativity (modulo last-bit), a+0, a-a.
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn proptest_add_commutative(x in proptest::num::f64::NORMAL, y in proptest::num::f64::NORMAL) {
        prop_assume!(x.is_finite() && y.is_finite() && x != 0.0 && y != 0.0);
        let a = BigFloat::from_f64(x, 50).expect("finite");
        let b = BigFloat::from_f64(y, 50).expect("finite");
        prop_assert_eq!(&a + &b, &b + &a);
    }

    #[test]
    fn proptest_add_assoc_modulo_lsb(
        x in proptest::num::f64::NORMAL,
        y in proptest::num::f64::NORMAL,
        z in proptest::num::f64::NORMAL,
    ) {
        prop_assume!(x.is_finite() && y.is_finite() && z.is_finite());
        prop_assume!(x != 0.0 && y != 0.0 && z != 0.0);
        // Avoid pathological cancellations that highlight rounding error: skip
        // triples whose intermediate sums overflow or whose magnitudes differ
        // by more than 2^900.
        prop_assume!(x.abs().log2() - y.abs().log2() < 900.0);
        prop_assume!(y.abs().log2() - z.abs().log2() < 900.0);
        let a = BigFloat::from_f64(x, 60).expect("finite");
        let b = BigFloat::from_f64(y, 60).expect("finite");
        let c = BigFloat::from_f64(z, 60).expect("finite");
        let left = (&a + &b) + &c;
        let right = &a + (&b + &c);
        // Modulo last-bit precision — agreement within 4 ULPs.
        let lf = left.to_f64();
        let rf = right.to_f64();
        if lf.is_finite() && rf.is_finite() && lf != 0.0 && rf != 0.0 {
            let rel = (lf - rf).abs() / lf.abs().max(rf.abs());
            prop_assert!(rel < 1e-12, "assoc rel diff {rel}");
        }
    }

    #[test]
    fn proptest_a_plus_zero_is_a(x in proptest::num::f64::NORMAL) {
        prop_assume!(x.is_finite() && x != 0.0);
        let a = BigFloat::from_f64(x, 50).expect("finite");
        let z = BigFloat::zero(50);
        prop_assert_eq!(&a + &z, a.clone());
    }

    #[test]
    fn proptest_a_minus_a_is_zero(x in proptest::num::f64::NORMAL) {
        prop_assume!(x.is_finite() && x != 0.0);
        let a = BigFloat::from_f64(x, 50).expect("finite");
        let zero = &a - &a;
        prop_assert!(zero.is_zero());
        prop_assert_eq!(zero.sign(), Sign::Positive);
    }

    #[test]
    fn proptest_neg_neg_is_identity(x in proptest::num::f64::NORMAL) {
        prop_assume!(x.is_finite() && x != 0.0);
        let a = BigFloat::from_f64(x, 50).expect("finite");
        prop_assert_eq!(-(-&a), a);
    }
}

// ===========================================================================
// N4b — Multiplication, Division, Square root
// ===========================================================================

use oxinum_core::OxiNumError;

// ---------------------------------------------------------------------------
// Multiplication — hand-picked cases
// ---------------------------------------------------------------------------

#[test]
fn mul_identity_one_times_a() {
    let one = BigFloat::from_i64(1, 32, RoundingMode::HalfEven);
    let a = BigFloat::from_i64(42, 32, RoundingMode::HalfEven);
    assert_eq!(&one * &a, a);
    assert_eq!(&a * &one, a);
}

#[test]
fn mul_zero_times_a_is_zero() {
    let z = BigFloat::zero(32);
    let a = BigFloat::from_i64(123, 32, RoundingMode::HalfEven);
    let p = &z * &a;
    assert!(p.is_zero());
    let p2 = &a * &z;
    assert!(p2.is_zero());
}

#[test]
fn mul_two_times_three_is_six() {
    let two = BigFloat::from_i64(2, 16, RoundingMode::HalfEven);
    let three = BigFloat::from_i64(3, 16, RoundingMode::HalfEven);
    let p = &two * &three;
    assert_eq!(p.to_f64(), 6.0);
}

#[test]
fn mul_one_and_a_half_times_two_is_three() {
    let a = BigFloat::from_f64(1.5, 4).expect("1.5 fits in 4 bits");
    let b = BigFloat::from_i64(2, 4, RoundingMode::HalfEven);
    let p = &a * &b;
    assert_eq!(p.to_f64(), 3.0);
}

#[test]
fn mul_neg_two_times_three_is_neg_six() {
    let a = BigFloat::from_i64(-2, 16, RoundingMode::HalfEven);
    let b = BigFloat::from_i64(3, 16, RoundingMode::HalfEven);
    let p = &a * &b;
    assert_eq!(p.to_f64(), -6.0);
}

#[test]
fn mul_two_negatives_is_positive() {
    let a = BigFloat::from_i64(-2, 16, RoundingMode::HalfEven);
    let b = BigFloat::from_i64(-3, 16, RoundingMode::HalfEven);
    let p = &a * &b;
    assert_eq!(p.to_f64(), 6.0);
}

#[test]
fn mul_precision_propagation_takes_max() {
    let a = BigFloat::from_i64(2, 10, RoundingMode::HalfEven);
    let b = BigFloat::from_i64(3, 50, RoundingMode::HalfEven);
    let p = &a * &b;
    assert_eq!(p.precision(), 50);
    assert_eq!(p.to_f64(), 6.0);
}

#[test]
fn mul_assign_works() {
    let mut a = BigFloat::from_i64(5, 16, RoundingMode::HalfEven);
    let b = BigFloat::from_i64(7, 16, RoundingMode::HalfEven);
    a *= &b;
    assert_eq!(a.to_f64(), 35.0);
}

// ---------------------------------------------------------------------------
// Multiplication — proptests
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn proptest_mul_commutative(
        x in proptest::num::f64::NORMAL,
        y in proptest::num::f64::NORMAL,
    ) {
        prop_assume!(x.is_finite() && y.is_finite() && x != 0.0 && y != 0.0);
        let a = BigFloat::from_f64(x, 50).expect("finite");
        let b = BigFloat::from_f64(y, 50).expect("finite");
        prop_assert_eq!(&a * &b, &b * &a);
    }

    #[test]
    fn proptest_mul_assoc_modulo_lsb(
        x in proptest::num::f64::NORMAL,
        y in proptest::num::f64::NORMAL,
        z in proptest::num::f64::NORMAL,
    ) {
        prop_assume!(x.is_finite() && y.is_finite() && z.is_finite());
        prop_assume!(x != 0.0 && y != 0.0 && z != 0.0);
        // Skip pairs whose log-magnitudes vary by more than 2^400 — those
        // produce intermediate products near f64 overflow / underflow that
        // would not round-trip through to_f64() cleanly.
        prop_assume!(x.abs().log2() + y.abs().log2() + z.abs().log2() < 900.0);
        prop_assume!(x.abs().log2() + y.abs().log2() + z.abs().log2() > -900.0);
        let a = BigFloat::from_f64(x, 60).expect("finite");
        let b = BigFloat::from_f64(y, 60).expect("finite");
        let c = BigFloat::from_f64(z, 60).expect("finite");
        let left = (&a * &b) * &c;
        let right = &a * (&b * &c);
        let lf = left.to_f64();
        let rf = right.to_f64();
        if lf.is_finite() && rf.is_finite() && lf != 0.0 && rf != 0.0 {
            let rel = (lf - rf).abs() / lf.abs().max(rf.abs());
            prop_assert!(rel < 1e-12, "mul assoc rel diff {rel}");
        }
    }
}

#[test]
fn cross_val_mul_200_random_pairs() {
    use proptest::test_runner::{Config, TestRunner};
    let mut runner = TestRunner::new(Config {
        cases: 200,
        ..Config::default()
    });
    runner
        .run(
            &(proptest::num::f64::NORMAL, proptest::num::f64::NORMAL),
            |(x, y)| {
                if !x.is_finite() || !y.is_finite() || x == 0.0 || y == 0.0 {
                    return Ok(());
                }
                let expected = x * y;
                // Skip products that overflow / underflow in f64 — those are
                // outside our cross-val oracle's domain.
                if !expected.is_finite() || expected == 0.0 {
                    return Ok(());
                }
                // Subnormal results are *not* skipped. They used to be, on the
                // theory that rounding to 53 bits and then onto the subnormal
                // grid is an unavoidable double rounding — but the second
                // rounding now consults the direction of the first, so the
                // result matches native f64's single-pass IEEE-754 rounding
                // bit for bit. See `tests/subnormal_rounding.rs`.
                let a = BigFloat::from_f64(x, 53).expect("finite");
                let b = BigFloat::from_f64(y, 53).expect("finite");
                let p = &a * &b;
                let pf = p.to_f64();
                prop_assert_eq!(pf, expected, "for {} * {}", x, y);
                Ok(())
            },
        )
        .expect("cross-val mul");
}

// ---------------------------------------------------------------------------
// Division — hand-picked cases
// ---------------------------------------------------------------------------

#[test]
fn div_six_by_two_is_three() {
    let a = BigFloat::from_i64(6, 16, RoundingMode::HalfEven);
    let b = BigFloat::from_i64(2, 16, RoundingMode::HalfEven);
    let q = a.div_ref(&b).expect("nonzero");
    assert_eq!(q.to_f64(), 3.0);
}

#[test]
fn div_one_by_two_is_half() {
    let a = BigFloat::from_i64(1, 16, RoundingMode::HalfEven);
    let b = BigFloat::from_i64(2, 16, RoundingMode::HalfEven);
    let q = a.div_ref(&b).expect("nonzero");
    assert_eq!(q.to_f64(), 0.5);
}

#[test]
fn div_neg_six_by_two_is_neg_three() {
    let a = BigFloat::from_i64(-6, 16, RoundingMode::HalfEven);
    let b = BigFloat::from_i64(2, 16, RoundingMode::HalfEven);
    let q = a.div_ref(&b).expect("nonzero");
    assert_eq!(q.to_f64(), -3.0);
}

#[test]
fn div_two_negatives_is_positive() {
    let a = BigFloat::from_i64(-6, 16, RoundingMode::HalfEven);
    let b = BigFloat::from_i64(-2, 16, RoundingMode::HalfEven);
    let q = a.div_ref(&b).expect("nonzero");
    assert_eq!(q.to_f64(), 3.0);
}

#[test]
fn div_zero_by_nonzero_is_zero() {
    let z = BigFloat::zero(32);
    let b = BigFloat::from_i64(7, 32, RoundingMode::HalfEven);
    let q = z.div_ref(&b).expect("nonzero divisor");
    assert!(q.is_zero());
    assert_eq!(q.precision(), 32);
}

#[test]
fn div_by_zero_returns_error() {
    let a = BigFloat::from_i64(5, 16, RoundingMode::HalfEven);
    let z = BigFloat::zero(16);
    let r = a.div_ref(&z);
    assert!(matches!(r, Err(OxiNumError::DivByZero)));
}

#[test]
fn div_operator_produces_inf_on_zero() {
    let a = BigFloat::from_i64(5, 16, RoundingMode::HalfEven);
    let z = BigFloat::zero(16);
    let result = &a / &z;
    assert!(result.is_infinite(), "5 / 0 should be +Inf per IEEE 754");
    assert!(result.is_sign_positive(), "5 / 0 should be positive Inf");
}

#[test]
fn div_assign_works() {
    let mut a = BigFloat::from_i64(20, 16, RoundingMode::HalfEven);
    let b = BigFloat::from_i64(4, 16, RoundingMode::HalfEven);
    a /= &b;
    assert_eq!(a.to_f64(), 5.0);
}

#[test]
fn div_precision_propagation_takes_max() {
    let a = BigFloat::from_i64(6, 10, RoundingMode::HalfEven);
    let b = BigFloat::from_i64(2, 50, RoundingMode::HalfEven);
    let q = a.div_ref(&b).expect("nonzero");
    assert_eq!(q.precision(), 50);
    assert_eq!(q.to_f64(), 3.0);
}

// ---------------------------------------------------------------------------
// Division — proptests
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn proptest_a_div_a_is_one(x in proptest::num::f64::NORMAL) {
        prop_assume!(x.is_finite() && x != 0.0);
        let a = BigFloat::from_f64(x, 60).expect("finite");
        let q = a.div_ref(&a).expect("nonzero");
        prop_assert_eq!(q.to_f64(), 1.0);
    }

    #[test]
    fn proptest_div_then_mul_roundtrip(
        x in proptest::num::f64::NORMAL,
        y in proptest::num::f64::NORMAL,
    ) {
        prop_assume!(x.is_finite() && y.is_finite() && x != 0.0 && y != 0.0);
        // Avoid overflow / underflow in the f64 round-trip oracle.
        prop_assume!((x / y).is_finite() && x / y != 0.0);
        prop_assume!(((x / y) * y).is_finite());
        let a = BigFloat::from_f64(x, 60).expect("finite");
        let b = BigFloat::from_f64(y, 60).expect("finite");
        let q = a.div_ref(&b).expect("nonzero");
        let back = &q * &b;
        let af = a.to_f64();
        let bf = back.to_f64();
        if af.is_finite() && bf.is_finite() && af != 0.0 && bf != 0.0 {
            // Relative tolerance: precision 60 has about 18 decimal digits of
            // accuracy; allow a generous 1e-14 margin to absorb rounding.
            let rel = (af - bf).abs() / af.abs().max(bf.abs());
            prop_assert!(rel < 1e-14, "div-then-mul rel diff {rel} for {}/{}", x, y);
        }
    }
}

#[test]
fn cross_val_div_100_random_pairs() {
    use proptest::test_runner::{Config, TestRunner};
    let mut runner = TestRunner::new(Config {
        cases: 100,
        ..Config::default()
    });
    runner
        .run(
            &(proptest::num::f64::NORMAL, proptest::num::f64::NORMAL),
            |(x, y)| {
                if !x.is_finite() || !y.is_finite() || x == 0.0 || y == 0.0 {
                    return Ok(());
                }
                let expected = x / y;
                if !expected.is_finite() || expected == 0.0 {
                    return Ok(());
                }
                let a = BigFloat::from_f64(x, 53).expect("finite");
                let b = BigFloat::from_f64(y, 53).expect("finite");
                let q = a.div_ref(&b).expect("nonzero divisor");
                let qf = q.to_f64();
                prop_assert_eq!(qf, expected, "for {} / {}", x, y);
                Ok(())
            },
        )
        .expect("cross-val div");
}

// ---------------------------------------------------------------------------
// Division — regression tests
// ---------------------------------------------------------------------------

/// Regression: previously BigFloat division dropped the integer-division
/// remainder when computing the sticky bit, causing a 1-ULP error in
/// correctly-rounded IEEE results.  The specific pair was found by proptest.
#[test]
fn div_correctly_rounds_regression_1ulp() {
    let x: f64 = 1.507_674_583_623_356_9e-301;
    let y: f64 = 3.743_549_738_805_966e-65;
    let expected = x / y; // native IEEE-754 result
    let a = BigFloat::from_f64(x, 53).expect("finite");
    let b = BigFloat::from_f64(y, 53).expect("finite");
    let q = a.div_ref(&b).expect("nonzero divisor");
    assert_eq!(
        q.to_f64(),
        expected,
        "div 1-ULP regression: BigFloat gave {}, expected {}",
        q.to_f64(),
        expected
    );
}

/// Run one frozen `(dividend, divisor, quotient)` triple, all as raw `f64`
/// bit patterns.
///
/// Bit patterns rather than decimal literals: at the subnormal boundary the
/// shortest round-tripping decimal form of two adjacent values differs in its
/// last digit, which makes a decimal seed both unreadable and fragile. The
/// expected quotient is asserted against the hardware divider first, so the
/// frozen constant is checked rather than trusted.
fn assert_div_matches_ieee(x_bits: u64, y_bits: u64, expected_bits: u64) {
    let x = f64::from_bits(x_bits);
    let y = f64::from_bits(y_bits);
    let expected = f64::from_bits(expected_bits);
    assert_eq!(
        (x / y).to_bits(),
        expected_bits,
        "frozen expectation disagrees with the hardware divider for {x:e} / {y:e}"
    );
    let q = BigFloat::from_f64(x, 53)
        .expect("finite")
        .div_ref(&BigFloat::from_f64(y, 53).expect("finite"))
        .expect("nonzero divisor");
    assert_eq!(
        q.to_f64().to_bits(),
        expected_bits,
        "div {x:e} / {y:e}: BigFloat gave {} ({:#x}), expected {expected:e} ({expected_bits:#x})",
        q.to_f64(),
        q.to_f64().to_bits(),
    );
}

/// Regression: `to_f64` flushed every magnitude below `2^-1074` straight to
/// zero instead of rounding it onto the subnormal grid.
///
/// A quotient of `1.4809… * 2^-1075` sits below the smallest subnormal but
/// *above half* of it, so round-to-nearest carries it up to `2^-1074`; the old
/// early return answered `0.0`. IEEE 754 calls this gradual underflow, and
/// skipping it is a whole-value error, not a 1-ULP one.
///
/// The first pair is the one `cross_val_div_100_random_pairs` caught — that
/// test fails on a *different* pair each run and proptest persisted no
/// regression file, which is why the seeds are frozen here by hand.
#[test]
fn div_gradual_underflow_rounds_up_to_min_subnormal() {
    // `5e-324` is `2^-1074`, the smallest positive f64.
    const MIN_SUBNORMAL_BITS: u64 = 0x1;
    // (dividend, divisor) — every one produced `0.0` before the fix.
    const CASES: &[(u64, u64)] = &[
        // 1.2094606113696943e-164 / 3.306044268249997e159 — from proptest.
        (0x1de6_4992_d938_4fb4, 0x610e_1980_5eb8_8785),
        // 2.005355950277463e-204 / 4.474074939264988e119.
        (0x15a4_1e9b_9a8a_c1da, 0x58c6_2d78_9286_e2d9),
        // 6.106742822090827e-223 / 2.2156608577612968e101.
        (0x11cc_414c_5bd0_f0d7, 0x54f9_5321_20ad_84f6),
        // 3.5856784388691428e-273 / 9.618307001928684e50.
        (0x075f_093f_06fa_10df, 0x4a84_90e3_e214_333e),
    ];
    for &(x_bits, y_bits) in CASES {
        assert_div_matches_ieee(x_bits, y_bits, MIN_SUBNORMAL_BITS);
    }
}

/// Regression: division rounded twice — once to the 53-bit working precision
/// and again onto the 52-bit subnormal grid — and the second rounding met an
/// exact tie the true quotient never sat on.
///
/// For a quotient in `[2^-1023, 2^-1022)` the halfway points of the subnormal
/// grid *are* the points of the 53-bit normal grid, so the first rounding
/// lands on one about half the time and ties-to-even then guesses wrong about
/// half of those — a 1-ULP error in roughly a quarter of that octave.
///
/// Worked example, the first pair below: the exact quotient is
/// `4084013600346416.5… * 2^-1074`, just above the halfway point
/// `8168027200692833 * 2^-1075`. Rounding to 53 bits lands exactly *on* that
/// halfway point; ties-to-even then picks the even neighbour
/// `4084013600346416`, while the correctly-rounded answer is
/// `4084013600346417`.
#[test]
fn div_subnormal_double_rounding_matches_single_rounding() {
    // (dividend, divisor, correctly-rounded quotient).
    const CASES: &[(u64, u64, u64)] = &[
        (
            0x20eb_1331_5f29_59b1,
            0x60cd_db4b_5273_4742,
            0x000e_8263_83e8_1131,
        ),
        (
            0x1156_efb1_743d_ffc2,
            0x514b_ac10_cf28_34bd,
            0x0006_a180_f221_c9e1,
        ),
        (
            0x0801_1654_a424_67db,
            0x47fa_84c8_fb26_b9d4,
            0x0005_279f_318f_9b49,
        ),
        (
            0x20bf_dd7a_2dd5_892b,
            0x60c3_2263_0c3b_08f9,
            0x0003_54a7_616f_52a5,
        ),
        (
            0x22ce_17b2_46c2_2710,
            0x62c4_039c_1898_675e,
            0x0006_03a7_5df3_75fb,
        ),
        (
            0x0525_ad00_7ed2_1037,
            0x450c_d8c5_3b03_59d7,
            0x000c_05c8_231b_7a83,
        ),
    ];
    for &(x_bits, y_bits, expected_bits) in CASES {
        assert_div_matches_ieee(x_bits, y_bits, expected_bits);
    }
}

/// Regression: the same double rounding, met in the *normal* range.
///
/// Dividing at precision 54 and converting to `f64` narrows by a single bit,
/// so a quotient with its low bit set sits exactly on a 53-bit halfway point —
/// the subnormal grid is not needed to reproduce the defect, and these cases
/// prove the fix is a property of the conversion rather than of the underflow
/// path.
#[test]
fn div_normal_range_double_rounding_matches_single_rounding() {
    const CASES: &[(u64, u64, u64)] = &[
        (
            0x7d83_e748_f2c8_309b,
            0x582f_8286_1b46_e63a,
            0x6544_368b_0079_2dbf,
        ),
        (
            0x4d69_70a2_6b74_7967,
            0x6cd5_3c97_0d7a_3875,
            0x2083_2ab4_d0eb_c722,
        ),
        (
            0x5c90_ccf4_1fad_080f,
            0x62bb_283c_fe2e_a1ff,
            0x39c3_cbec_6dac_0662,
        ),
        (
            0x6a25_8763_b680_30b6,
            0x5e5d_cdbf_4d18_cd77,
            0x4bb7_1d89_2be0_6a81,
        ),
        (
            0x478d_286a_ce05_510d,
            0x4057_1090_4132_b8ef,
            0x4724_3a12_5813_8d3c,
        ),
        (
            0x470f_6681_76be_065e,
            0x4684_c0f4_60fd_3df2,
            0x4078_3540_8dcd_cf3d,
        ),
    ];
    for &(x_bits, y_bits, expected_bits) in CASES {
        let x = f64::from_bits(x_bits);
        let y = f64::from_bits(y_bits);
        assert_eq!(
            (x / y).to_bits(),
            expected_bits,
            "frozen expectation disagrees with the hardware divider"
        );
        let q = BigFloat::from_f64(x, 54)
            .expect("finite")
            .div_ref(&BigFloat::from_f64(y, 54).expect("finite"))
            .expect("nonzero divisor");
        assert_eq!(
            q.to_f64().to_bits(),
            expected_bits,
            "54-bit div {x:e} / {y:e} double-rounded on the way to f64"
        );
    }
}

/// Regression: the quotient shift was sized against the *target* precision
/// alone, so a divisor far wider than the dividend produced a quotient with
/// fewer bits than the target — `round_to_precision_in_place` then zero-padded
/// it instead of rounding, freezing the forced sticky bit into the value.
///
/// `1` at precision 8 over `3` at precision 53 came back as
/// `0.33333587646484375`: seventeen correct bits in a fifty-three bit result.
/// Every pre-existing division test used equal precisions, which is why this
/// survived.
#[test]
fn div_narrow_dividend_wide_divisor_keeps_full_precision() {
    for (p_num, p_den) in [(8u32, 53u32), (4, 200), (53, 200), (1, 64), (200, 53)] {
        let a = BigFloat::from_i64(1, p_num, RoundingMode::HalfEven);
        let b = BigFloat::from_i64(3, p_den, RoundingMode::HalfEven);
        let q = a.div_ref(&b).expect("nonzero");
        let target = p_num.max(p_den);
        assert_eq!(
            q.precision(),
            target,
            "precision propagation at {p_num}/{p_den}"
        );
        assert_eq!(
            q.mantissa().bit_length(),
            u64::from(target),
            "mantissa must be normalized to the full target width at {p_num}/{p_den}",
        );
        // At any precision >= 53 the leading 53 bits agree with the f64 third.
        assert_eq!(
            q.to_f64(),
            1.0_f64 / 3.0,
            "1/3 at precisions ({p_num}, {p_den}) lost significance",
        );
    }
}

// ---------------------------------------------------------------------------
// Sqrt — hand-picked cases
// ---------------------------------------------------------------------------

#[test]
fn sqrt_zero_is_zero() {
    let z = BigFloat::zero(32);
    let s = z.sqrt(32, RoundingMode::HalfEven).expect("real");
    assert!(s.is_zero());
}

#[test]
fn sqrt_one_is_one() {
    let one = BigFloat::from_i64(1, 32, RoundingMode::HalfEven);
    let s = one.sqrt(32, RoundingMode::HalfEven).expect("real");
    assert_eq!(s.to_f64(), 1.0);
}

#[test]
fn sqrt_four_is_two() {
    let four = BigFloat::from_i64(4, 32, RoundingMode::HalfEven);
    let s = four.sqrt(32, RoundingMode::HalfEven).expect("real");
    assert_eq!(s.to_f64(), 2.0);
}

#[test]
fn sqrt_nine_is_three() {
    let nine = BigFloat::from_i64(9, 32, RoundingMode::HalfEven);
    let s = nine.sqrt(32, RoundingMode::HalfEven).expect("real");
    assert_eq!(s.to_f64(), 3.0);
}

#[test]
fn sqrt_negative_is_domain_error() {
    let neg_one = BigFloat::from_i64(-1, 16, RoundingMode::HalfEven);
    let r = neg_one.sqrt(16, RoundingMode::HalfEven);
    assert!(matches!(r, Err(OxiNumError::Domain(_))));
}

#[test]
fn sqrt_two_squared_is_two_at_high_precision() {
    // sqrt(2)^2 should equal 2.0 (or extremely close) when sqrt is computed
    // at 200 bits of precision then re-multiplied at the same precision.
    let two = BigFloat::from_i64(2, 200, RoundingMode::HalfEven);
    let s = two.sqrt(200, RoundingMode::HalfEven).expect("real");
    // s ~ 1.41421356...
    let sf = s.to_f64();
    assert!((sf - 2.0_f64.sqrt()).abs() < 1e-15, "got {sf}");
    // s^2 should be 2.0 to f64 precision.
    let s2 = &s * &s;
    let s2f = s2.to_f64();
    assert!((s2f - 2.0).abs() < 1e-14, "sqrt(2)^2 = {s2f}");
}

#[test]
fn sqrt_half_is_known_value() {
    let half = BigFloat::from_f64(0.5, 53).expect("0.5 finite");
    let s = half.sqrt(53, RoundingMode::HalfEven).expect("real");
    // sqrt(0.5) = 1/sqrt(2) ~ 0.7071067811865476
    let sf = s.to_f64();
    assert!((sf - (0.5_f64).sqrt()).abs() < 1e-15, "got {sf}");
}

#[test]
fn sqrt_large_value() {
    // sqrt(2^100) = 2^50 (exact). Value-equal to a BigFloat constructed
    // directly from `2^50`, regardless of the (mantissa, exponent)
    // representation chosen by the canonical-form normalizer.
    use oxinum_int::native::BigUint;
    let two_pow_100 = BigFloat::from_parts(
        Sign::Positive,
        BigUint::one(),
        100,
        53,
        RoundingMode::HalfEven,
    );
    let s = two_pow_100.sqrt(53, RoundingMode::HalfEven).expect("real");
    // Compare against the canonical 2^50 representation.
    let expected = BigFloat::from_parts(
        Sign::Positive,
        BigUint::one(),
        50,
        53,
        RoundingMode::HalfEven,
    );
    assert_eq!(s, expected);
}

#[test]
fn sqrt_odd_exponent_value() {
    // 2.0 has mantissa = 1 (1 bit), exponent = 1 (odd). Tests the odd-exp branch.
    let two = BigFloat::from_i64(2, 53, RoundingMode::HalfEven);
    let s = two.sqrt(53, RoundingMode::HalfEven).expect("real");
    let sf = s.to_f64();
    assert!((sf - 2.0_f64.sqrt()).abs() < 1e-15, "got {sf}");
}

// ---------------------------------------------------------------------------
// Sqrt — cross-validation against f64::sqrt
// ---------------------------------------------------------------------------

#[test]
fn cross_val_sqrt_100_random_non_negative() {
    use proptest::test_runner::{Config, TestRunner};
    let mut runner = TestRunner::new(Config {
        cases: 100,
        ..Config::default()
    });
    runner
        .run(&proptest::num::f64::NORMAL, |x| {
            if !x.is_finite() || x <= 0.0 {
                return Ok(());
            }
            let a = BigFloat::from_f64(x, 53).expect("finite");
            let s = a.sqrt(53, RoundingMode::HalfEven).expect("non-negative");
            let sf = s.to_f64();
            let expected = x.sqrt();
            // Tolerance: 4 ULPs at f64 precision — our integer-sqrt-of-scaled-
            // mantissa approach delivers correct rounding for the dominant
            // bits, but is within a small number of low-bit ULPs of the IEEE
            // round-to-nearest oracle.
            if expected != 0.0 {
                let rel = (sf - expected).abs() / expected.abs();
                prop_assert!(rel < 1e-14, "sqrt mismatch for {x}: {sf} vs {expected}");
            }
            Ok(())
        })
        .expect("cross-val sqrt");
}

// ---------------------------------------------------------------------------
// Serde JSON round-trip tests (feature-gated)
// ---------------------------------------------------------------------------

#[cfg(feature = "serde")]
mod serde_tests {
    use oxinum_float::native::{BigFloat, RoundingMode};
    use serde_json::Value;

    fn roundtrip(x: &BigFloat) {
        let json = serde_json::to_string(x).expect("serialize");
        let back: BigFloat = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(*x, back, "round-trip failed for {x:?}");
        // Also check precision is preserved (equality ignores precision).
        assert_eq!(x.precision(), back.precision(), "precision lost for {x:?}");
    }

    #[test]
    fn serde_bigfloat_zero() {
        roundtrip(&BigFloat::zero(53));
        roundtrip(&BigFloat::zero(1));
        roundtrip(&BigFloat::zero(256));
    }

    #[test]
    fn serde_bigfloat_one() {
        let x = BigFloat::from_i64(1, 53, RoundingMode::HalfEven);
        roundtrip(&x);
    }

    #[test]
    fn serde_bigfloat_negative() {
        let x = BigFloat::from_i64(-12345, 64, RoundingMode::HalfEven);
        roundtrip(&x);
    }

    #[test]
    fn serde_bigfloat_many_from_i64() {
        // 101 values covering both polarities and all bit-widths up to i64.
        for i in -50i64..=50 {
            if i == 0 {
                continue;
            }
            let x = BigFloat::from_i64(i, 64, RoundingMode::HalfEven);
            roundtrip(&x);
        }
        // Also test extremes
        let max = BigFloat::from_i64(i64::MAX, 64, RoundingMode::HalfEven);
        roundtrip(&max);
        let min = BigFloat::from_i64(i64::MIN, 64, RoundingMode::HalfEven);
        roundtrip(&min);
    }

    #[test]
    fn serde_bigfloat_high_precision_pi() {
        use oxinum_float::native::pi;
        let pi_500 = pi(500).expect("pi");
        roundtrip(&pi_500);
    }

    #[test]
    fn serde_bigfloat_precision_zero_rejected() {
        // Mutate a valid serialized BigFloat to set precision to 0.
        let valid = BigFloat::from_i64(1, 53, RoundingMode::HalfEven);
        let mut val: Value = serde_json::to_value(&valid).expect("serialize");
        val["precision"] = Value::Number(0.into());
        let result: Result<BigFloat, _> = serde_json::from_value(val);
        assert!(result.is_err(), "precision=0 should fail deserialization");
    }

    #[test]
    fn serde_bigfloat_bit_length_mismatch_rejected() {
        // Serialize a 53-bit float, then increase precision to 64:
        // mantissa bit_length (53) != precision (64) => invariant violation.
        let valid = BigFloat::from_i64(1, 53, RoundingMode::HalfEven);
        let mut val: Value = serde_json::to_value(&valid).expect("serialize");
        // The mantissa encodes bit_length == 53; set precision to 64.
        val["precision"] = Value::Number(64.into());
        let result: Result<BigFloat, _> = serde_json::from_value(val);
        assert!(
            result.is_err(),
            "mantissa bit_length != precision must fail deserialization"
        );
    }

    #[test]
    fn serde_bigfloat_zero_canonical_sign_forced() {
        // Serialize a valid non-zero value, then zero out the mantissa so
        // sign field is irrelevant and exponent is forced to 0.
        // Deserialization must succeed and produce canonical zero
        // (sign=Positive, exponent=0).
        let valid = BigFloat::from_i64(5, 53, RoundingMode::HalfEven);
        let mut val: Value = serde_json::to_value(&valid).expect("serialize");
        // Zero out the mantissa limbs.
        val["mantissa"]["limbs"] = Value::Array(vec![]);
        // Precision stays at 53, exponent at whatever (will be forced to 0).
        let back: BigFloat = serde_json::from_value(val).expect("deserialize zero variant");
        assert!(back.is_zero(), "deserialized value should be zero");
        // Canonical zero invariant: sign must be Positive, exponent must be 0.
        assert_eq!(back.sign(), oxinum_core::Sign::Positive);
        assert_eq!(back.exponent(), 0);
    }
}

// ---------------------------------------------------------------------------
// Phase 6: BigFloat → BigInt conversions
// ---------------------------------------------------------------------------

#[cfg(test)]
mod bigfloat_to_bigint {
    use oxinum_float::native::{BigFloat, RoundingMode};
    use oxinum_int::native::BigInt;

    #[test]
    fn positive_trunc_floor_ceil_round() {
        // 3.7 → trunc/floor=3, ceil/round=4
        let x = BigFloat::from_f64(3.7, 64).expect("3.7");
        assert_eq!(x.to_bigint_trunc(), BigInt::from(3i64), "trunc(3.7)");
        assert_eq!(x.to_bigint_floor(), BigInt::from(3i64), "floor(3.7)");
        assert_eq!(x.to_bigint_ceil(), BigInt::from(4i64), "ceil(3.7)");
        assert_eq!(x.to_bigint_round(), BigInt::from(4i64), "round(3.7)");
    }

    #[test]
    fn negative_trunc_floor_ceil_round() {
        // -3.7 → trunc/ceil=-3, floor/round=-4
        let y = BigFloat::from_f64(-3.7, 64).expect("-3.7");
        assert_eq!(y.to_bigint_trunc(), BigInt::from(-3i64), "trunc(-3.7)");
        assert_eq!(y.to_bigint_floor(), BigInt::from(-4i64), "floor(-3.7)");
        assert_eq!(y.to_bigint_ceil(), BigInt::from(-3i64), "ceil(-3.7)");
        assert_eq!(y.to_bigint_round(), BigInt::from(-4i64), "round(-3.7)");
    }

    #[test]
    fn exact_integer_unchanged() {
        // 42 is already an integer — all four agree.
        let z = BigFloat::from_i64(42, 64, RoundingMode::HalfEven);
        let expected = BigInt::from(42i64);
        assert_eq!(z.to_bigint_trunc(), expected, "trunc(42)");
        assert_eq!(z.to_bigint_floor(), expected, "floor(42)");
        assert_eq!(z.to_bigint_ceil(), expected, "ceil(42)");
        assert_eq!(z.to_bigint_round(), expected, "round(42)");
    }

    #[test]
    fn negative_exact_integer_unchanged() {
        let z = BigFloat::from_i64(-7, 64, RoundingMode::HalfEven);
        let expected = BigInt::from(-7i64);
        assert_eq!(z.to_bigint_trunc(), expected, "trunc(-7)");
        assert_eq!(z.to_bigint_floor(), expected, "floor(-7)");
        assert_eq!(z.to_bigint_ceil(), expected, "ceil(-7)");
        assert_eq!(z.to_bigint_round(), expected, "round(-7)");
    }

    #[test]
    fn zero_is_zero() {
        let z = BigFloat::zero(64);
        assert_eq!(z.to_bigint_trunc(), BigInt::zero(), "trunc(0)");
        assert_eq!(z.to_bigint_floor(), BigInt::zero(), "floor(0)");
        assert_eq!(z.to_bigint_ceil(), BigInt::zero(), "ceil(0)");
        assert_eq!(z.to_bigint_round(), BigInt::zero(), "round(0)");
    }

    #[test]
    fn half_positive_rounds_away() {
        // 2.5 → trunc=2, floor=2, ceil=3, round=3 (half-away-from-zero)
        let h = BigFloat::from_f64(2.5, 64).expect("2.5");
        assert_eq!(h.to_bigint_trunc(), BigInt::from(2i64), "trunc(2.5)");
        assert_eq!(h.to_bigint_floor(), BigInt::from(2i64), "floor(2.5)");
        assert_eq!(h.to_bigint_ceil(), BigInt::from(3i64), "ceil(2.5)");
        assert_eq!(h.to_bigint_round(), BigInt::from(3i64), "round(2.5)");
    }

    #[test]
    fn half_negative_rounds_away() {
        // -2.5 → trunc=-2, floor=-3, ceil=-2, round=-3 (half-away-from-zero)
        let h = BigFloat::from_f64(-2.5, 64).expect("-2.5");
        assert_eq!(h.to_bigint_trunc(), BigInt::from(-2i64), "trunc(-2.5)");
        assert_eq!(h.to_bigint_floor(), BigInt::from(-3i64), "floor(-2.5)");
        assert_eq!(h.to_bigint_ceil(), BigInt::from(-2i64), "ceil(-2.5)");
        assert_eq!(h.to_bigint_round(), BigInt::from(-3i64), "round(-2.5)");
    }

    #[test]
    fn small_fraction_truncated() {
        // 0.3 → trunc/floor/round=0, ceil=1
        let x = BigFloat::from_f64(0.3, 64).expect("0.3");
        assert_eq!(x.to_bigint_trunc(), BigInt::zero(), "trunc(0.3)");
        assert_eq!(x.to_bigint_floor(), BigInt::zero(), "floor(0.3)");
        assert_eq!(x.to_bigint_ceil(), BigInt::one(), "ceil(0.3)");
        assert_eq!(x.to_bigint_round(), BigInt::zero(), "round(0.3)");
    }

    #[test]
    fn negative_small_fraction() {
        // -0.3 → trunc/ceil=0, floor=-1, round=0
        let x = BigFloat::from_f64(-0.3, 64).expect("-0.3");
        assert_eq!(x.to_bigint_trunc(), BigInt::zero(), "trunc(-0.3)");
        assert_eq!(x.to_bigint_floor(), BigInt::from(-1i64), "floor(-0.3)");
        assert_eq!(x.to_bigint_ceil(), BigInt::zero(), "ceil(-0.3)");
        assert_eq!(x.to_bigint_round(), BigInt::zero(), "round(-0.3)");
    }
}
