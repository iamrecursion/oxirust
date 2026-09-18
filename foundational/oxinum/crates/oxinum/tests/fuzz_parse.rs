//! Proptest-based fuzz harness for [`oxinum::parse`], the crate's universal
//! numeric-string parser (auto-detects integer / rational / float format).
//!
//! Two dimensions, mirroring the harnesses already in `oxinum-int` and
//! `oxinum-float`:
//!
//! 1. **Garbage input safety** — feed arbitrary strings (and arbitrary raw
//!    bytes, lossily converted to `String`) to [`oxinum::parse`]. The
//!    contract is: it must return `Ok(_)` or `Err(_)`, **never panic, never
//!    hang**.
//! 2. **Valid input round-trip** — generate well-formed integer, rational,
//!    and float strings from random numeric values, parse them, and check
//!    that the parser both (a) picks the expected [`ParsedNumber`] variant
//!    and (b) recovers the original value.
//!
//! Run with:
//!   `cargo nextest run -p oxinum --all-features`

use oxinum::{parse, ParsedNumber};
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Dimension A — garbage input safety (must not panic or hang)
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(proptest::test_runner::Config::with_cases(1024))]

    /// Arbitrary Unicode strings must never panic `parse`.
    #[test]
    fn parse_arbitrary_string_no_panic(s in any::<String>()) {
        let _ = parse(&s);
    }

    /// Arbitrary raw bytes (lossily decoded to `String`, so invalid UTF-8
    /// sequences are exercised too via the replacement character) must
    /// never panic `parse`.
    #[test]
    fn parse_arbitrary_bytes_no_panic(bytes in proptest::collection::vec(any::<u8>(), 0..256)) {
        let s = String::from_utf8_lossy(&bytes);
        let _ = parse(&s);
    }

    /// Strings built only from characters that plausibly *look* numeric
    /// (digits, sign, separators, exponent markers) are the adversarial
    /// sweet spot for parser bugs — still must never panic.
    #[test]
    fn parse_numeric_looking_garbage_no_panic(
        s in "[-+0-9./eE ]{0,64}"
    ) {
        let _ = parse(&s);
    }

    /// Whitespace-only and empty-ish strings must never panic and must
    /// always be rejected as empty input.
    #[test]
    fn parse_whitespace_only_errors_no_panic(s in "[ \t\n\r]{0,32}") {
        let result = parse(&s);
        prop_assert!(result.is_err());
    }
}

// ---------------------------------------------------------------------------
// Dimension B — valid input round-trip
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(proptest::test_runner::Config::with_cases(256))]

    /// Any `i128`-formatted decimal integer string round-trips through
    /// `parse` as `ParsedNumber::Integer` with the exact original value.
    #[test]
    fn parse_integer_roundtrip(n in any::<i128>()) {
        let s = n.to_string();
        match parse(&s) {
            Ok(ParsedNumber::Integer(v)) => {
                prop_assert_eq!(v, oxinum::Int::from(n));
            }
            other => prop_assert!(false, "expected Integer for {s:?}, got {other:?}"),
        }
    }

    /// `"<numerator>/<denominator>"` built from a nonzero `i64` numerator
    /// and a nonzero `u64` denominator round-trips as `ParsedNumber::Rational`.
    #[test]
    fn parse_rational_roundtrip(
        num in any::<i64>(),
        den in 1u64..=u64::MAX
    ) {
        let s = format!("{num}/{den}");
        match parse(&s) {
            Ok(ParsedNumber::Rational(v)) => {
                let expected = oxinum::Rational::from(num) / oxinum::Rational::from(den);
                prop_assert_eq!(v, expected);
            }
            other => prop_assert!(false, "expected Rational for {s:?}, got {other:?}"),
        }
    }

    /// Any finite `f64` formatted with a decimal point round-trips as
    /// `ParsedNumber::Float`, recovering the same `f64` value to within
    /// 2 ULPs.
    ///
    /// `DBig` parses the decimal string exactly (as an exact base-10
    /// rational) and `{n:?}` is Rust's shortest round-trippable decimal for
    /// `n`, but the decimal-to-binary base conversion inside
    /// `DBig::to_f64` is a separate rounding step from the one Rust's own
    /// float formatter/parser pair uses internally, so an occasional
    /// last-bit (1 ULP) difference is expected and tolerated here — the
    /// property under test is "recovers essentially the same value", not
    /// bit-for-bit identity with `str::parse::<f64>`.
    #[test]
    fn parse_float_roundtrip(n in proptest::num::f64::NORMAL) {
        // `{:?}` on f64 always includes a decimal point (Rust's Debug for
        // floats appends `.0` for integral values), so this always takes
        // the float-detection branch inside `parse`.
        let s = format!("{n:?}");
        match parse(&s) {
            Ok(ParsedNumber::Float(v)) => {
                let got = v.to_f64().value();
                let ulp_diff = (got.to_bits() as i128 - n.to_bits() as i128).unsigned_abs();
                prop_assert!(
                    ulp_diff <= 2,
                    "float roundtrip for {s:?}: got {got:e}, expected {n:e} ({ulp_diff} ULPs off)"
                );
            }
            other => prop_assert!(false, "expected Float for {s:?}, got {other:?}"),
        }
    }

    /// Detection-rule precedence: a string containing `/` is always routed
    /// to the rational parser, even if it also contains `.` or `e`/`E`.
    #[test]
    fn parse_slash_always_wins_over_float_markers(
        num in any::<i32>(),
        den in 1u32..=u32::MAX,
    ) {
        // Not a realistic float, but exercises the detection-order branch:
        // contains both '/' and '.', '/' must win.
        let s = format!("{num}.0/{den}");
        // This is not necessarily parseable (the numerator has a stray
        // '.0'), so only assert it never panics and, if it succeeds, that
        // it picked the Rational branch (never Float or Integer).
        if let Ok(result) = parse(&s) {
            prop_assert!(matches!(result, ParsedNumber::Rational(_)));
        }
    }
}

// ---------------------------------------------------------------------------
// Explicit deterministic edge cases
// ---------------------------------------------------------------------------

#[test]
fn parse_edge_cases_never_panic() {
    // Empty / whitespace-only input.
    assert!(parse("").is_err());
    assert!(parse("   ").is_err());
    assert!(parse("\t\n").is_err());

    // Degenerate rational forms.
    assert!(parse("/").is_err());
    assert!(parse("1/0").is_err());
    assert!(parse("1/0/2").is_err());
    assert!(parse("/5").is_err());
    assert!(parse("5/").is_err());

    // Degenerate float forms.
    assert!(parse(".").is_err());
    assert!(parse("e").is_err());
    assert!(parse("1e").is_err());
    assert!(parse("1.2.3").is_err());
    assert!(parse("--1.5").is_err());

    // Degenerate integer forms.
    assert!(parse("+").is_err());
    assert!(parse("-").is_err());
    assert!(parse("abc").is_err());
    assert!(parse("１２３").is_err()); // fullwidth digits, not ASCII

    // Very long garbage input must not panic or hang.
    let long_garbage: String = "x1.e/-".repeat(500);
    let _ = parse(&long_garbage);

    // Very long valid integer must parse successfully.
    let long_int = "9".repeat(1000);
    assert!(matches!(parse(&long_int), Ok(ParsedNumber::Integer(_))));

    // Leading/trailing whitespace is trimmed.
    assert!(matches!(parse("  42  "), Ok(ParsedNumber::Integer(_))));

    // Null bytes and other control characters must not panic.
    assert!(parse("\0").is_err());
    assert!(parse("1\x002").is_err());
}

/// Regression test: `"1/0"` (and other explicit-zero-denominator forms with
/// a non-zero numerator) must be rejected by `parse` with a typed error,
/// never accepted as a malformed `ParsedNumber::Rational` that would later
/// panic on arithmetic such as `.to_f64()`. (The underlying
/// `dashu_ratio::RBig::from_str` does not itself validate this — it silently
/// builds an invalid `numerator/0` value — so `parse` must guard it
/// explicitly.)
#[test]
fn parse_rejects_explicit_zero_denominator() {
    assert!(parse("1/0").is_err());
    assert!(parse("-1/0").is_err());
    assert!(parse("12345/00000").is_err());
    // "0/0" is the one exception: `dashu_ratio` special-cases a zero
    // numerator during reduction and canonicalizes it to `0/1` before this
    // guard ever runs, so it is accepted as (arguably debatable, but safe
    // and non-panicking) zero rather than rejected.
    assert!(matches!(parse("0/0"), Ok(ParsedNumber::Rational(_))));
}
