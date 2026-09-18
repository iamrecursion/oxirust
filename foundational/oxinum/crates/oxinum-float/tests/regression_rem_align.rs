//! Regression tests for two confirmed `BigFloat` defects:
//!
//! 1. `%` / `rem` rounded the quotient to `prec` significant bits before
//!    truncating, so whenever the integer part of `|a/b|` needed more than
//!    `prec` bits the remainder fell outside `[0, |b|)`.
//! 2. add/sub aligned operands by shifting the higher-exponent mantissa left
//!    by the full exponent gap, so a large (but perfectly valid) exponent gap
//!    triggered an unbounded allocation that aborted the process.

use oxinum_core::Sign;
use oxinum_float::native::{BigFloat, RoundingMode};
use oxinum_int::native::BigUint;

/// The integer part of `|a/b|` needs ~99 bits while the working precision is
/// only 8 bits. The remainder must still be exact and lie in `[0, |b|)`.
/// `2^100 mod 3 == 1` (since `2^100 = 4^50 ≡ 1 (mod 3)`).
#[test]
fn rem_quotient_integer_part_exceeds_precision() {
    // a = 2^100 exactly, stored at only 8 bits of precision.
    let a = BigFloat::from_parts(
        Sign::Positive,
        BigUint::one(),
        100,
        8,
        RoundingMode::HalfEven,
    );
    let b = BigFloat::from_f64(3.0, 8).expect("3 fits in 8 bits");

    let r = &a % &b;

    // Remainder must lie in [0, |b|).
    assert!(r >= BigFloat::zero(8), "remainder must be >= 0");
    assert!(r < b, "remainder must be < |b|, got {}", r.to_f64());
    // And it must be exactly 1.
    assert_eq!(r.to_f64(), 1.0, "2^100 mod 3 must equal 1");
}

/// A non-power-of-two dividend with the same oversized-quotient shape, to guard
/// against the exponent-branch arithmetic. `(2^100 + 2) mod 3`: `2^100 ≡ 1`,
/// `2 ≡ 2`, so the sum `≡ 0 (mod 3)` and the remainder is 0.
#[test]
fn rem_large_quotient_exact_zero_remainder() {
    // a = 2^100 + 2. Mantissa "100...010" scaled so the value is exact at
    // 99 bits of precision (the span from bit 100 down to bit 1).
    let mut mantissa = BigUint::one().shl_bits(99); // bit 99 set -> 2^99
    mantissa.set_bit(0); // -> 2^99 + 1 (100-bit, odd mantissa)
                         // value = mantissa * 2^1 = (2^99 + 1) * 2 = 2^100 + 2, exact at 128 bits.
    let a = BigFloat::from_parts(Sign::Positive, mantissa, 1, 128, RoundingMode::HalfEven);
    let b = BigFloat::from_f64(3.0, 8).expect("3 fits in 8 bits");

    let r = &a % &b;

    assert!(r >= BigFloat::zero(8), "remainder must be >= 0");
    assert!(r < b, "remainder must be < |b|");
    assert_eq!(r.to_f64(), 0.0, "(2^100 + 2) mod 3 must equal 0");
}

/// Adding operands whose exponents differ by ~10^15 bits must complete in
/// bounded memory (the naive full-gap shift would try to materialize ~125 TB
/// of mantissa and abort) and return the correctly-rounded sum.
#[test]
fn add_huge_exponent_gap_is_bounded_and_correctly_rounded() {
    // Big operand: 1.0 at 53 bits.
    let big = BigFloat::from_f64(1.0, 53).expect("1.0");
    // Tiny operand: 2^(-1_000_000_000_000_000) — a valid BigFloat with an
    // enormous negative exponent, far below the last retained bit of `big`.
    let tiny = BigFloat::from_parts(
        Sign::Positive,
        BigUint::one(),
        -1_000_000_000_000_000,
        53,
        RoundingMode::HalfEven,
    );

    // 1.0 + tiny rounds to 1.0.
    let s1 = &big + &tiny;
    assert_eq!(s1.to_f64(), 1.0);

    // Reverse operand order exercises the opposite alignment branch.
    let s2 = &tiny + &big;
    assert_eq!(s2.to_f64(), 1.0);

    // Subtraction with the same huge gap: 1.0 - tiny rounds to 1.0.
    let d = &big - &tiny;
    assert_eq!(d.to_f64(), 1.0);
}
