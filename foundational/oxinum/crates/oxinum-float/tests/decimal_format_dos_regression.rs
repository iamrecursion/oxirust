//! Regression tests for the `pow10`/`decimal_magnitude` performance-DoS in
//! `to_scientific_string` / `to_engineering_string`
//! (`crates/oxinum-float/src/native/format_ext.rs`).
//!
//! # The defect
//!
//! `BigFloat::from_hex_float` accepts an untrusted, attacker-controlled
//! C99-`%a`-style string with an essentially free decimal exponent (up to
//! `BigFloat::EMAX`/`EMIN`, `±2^40`). `decimal_magnitude` (the shared
//! engine behind both decimal renderers) needs an exact power of ten roughly
//! proportional to that exponent, and the old `pow10` built it by
//! repeatedly multiplying a growing accumulator by a fixed small (`10^19`)
//! chunk — `n / 19` sequential `k`-limb-by-1-limb multiplies. That pattern
//! is `O(n^2)` no matter how fast the underlying big-integer multiply is
//! (Karatsuba/Toom-3 both need *balanced* operands to pay off, and a 1-limb
//! operand never qualifies), so a single hex-float literal with a
//! multi-million binary exponent could burn tens of seconds of CPU in one
//! formatting call — confirmed here by a real libFuzzer timeout artifact at
//! `fuzz/artifacts/hex_float_parse/timeout-0d4e887fd73ddc408ca9c29bb425d632eb35dcc3`
//! (`timeout 120` did not observe it finish).
//!
//! # The fix
//!
//! 1. `pow10` now does binary (square-and-multiply) exponentiation, so
//!    every multiplication is balanced and Karatsuba/Toom-3 actually
//!    engage.
//! 2. `decimal_magnitude` caches that power of ten (`Pow10Cache`) across the
//!    handful of nearby exponents it evaluates (the correction loop's
//!    candidates and Step 3's exponent are all within `O(sig_digits)` of
//!    each other), instead of recomputing a fresh `pow10` — even a fast one
//!    — at the same huge magnitude several times.
//! 3. A tighter, dedicated bit budget (`MAX_DECIMAL_FORMAT_BITS`, `2^19`
//!    bits) gates the decimal-formatting path specifically: even with (1)
//!    and (2), the remaining division of two `~n`-bit operands is real,
//!    super-linear work once `n` leaves the small regime, so a value whose
//!    exact decimal expansion would need more than `2^19` bits of exact
//!    scaffolding renders via the already-fast, lossless
//!    [`BigFloat::to_hex_string`] instead. This is a *fallback to a
//!    different exact rendering*, not a rejection: `from_hex_float` still
//!    accepts the value, and `to_hex_string`'s output round-trips through
//!    `from_hex_float` bit-for-bit. The bound is calibrated well above
//!    (~5x) the largest exponent this crate's test suite (or any realistic
//!    finite-precision float) ever renders in decimal, so no previously
//!    "fast" input is affected — only inputs that were already multi-second
//!    or worse.
//!
//! # What this file covers
//!
//! 1. The exact fuzz-artifact input (and its negative-exponent mirror, which
//!    the artifact alone does not exercise) now completes in well under a
//!    second, both for `to_scientific_string` and `to_engineering_string`.
//! 2. The decimal/hex boundary introduced by `MAX_DECIMAL_FORMAT_BITS` is
//!    pinned exactly (one bit below the threshold stays decimal, one bit at
//!    the threshold's edge falls back to hex).
//! 3. Byte-identical decimal output for a spread of ordinary values (zero,
//!    a subnormal-magnitude value, a plain negative exponent, exact
//!    non-negative powers of ten) that stay far inside the new budget and
//!    so must render exactly as before.

use oxinum_core::Sign;
use oxinum_float::native::{BigFloat, RoundingMode};
use oxinum_int::native::BigUint;
use std::time::{Duration, Instant};

const MODE: RoundingMode = RoundingMode::HalfEven;

/// Generous wall-clock ceiling for the "must be fast now" assertions below.
/// The old implementation ran past a 120s hard kill on the same input; a
/// correctly bounded implementation finishes in micro- to low-millisecond
/// time in isolation. 10 seconds leaves enormous headroom above that —
/// enough to absorb scheduling jitter when the full workspace suite runs
/// with `cargo nextest`'s parallel test threads contending for CPU — while
/// still failing loudly, and well inside nextest's default 120s slow-test
/// termination window, if the quadratic behavior (or something equally bad)
/// ever comes back.
const MUST_BE_FAST: Duration = Duration::from_secs(10);

/// `2^exponent` at `prec` bits, mirroring the helper used throughout
/// `regression_unbounded_shift.rs`.
fn pow2(exponent: i64, prec: u32) -> BigFloat {
    BigFloat::from_parts(Sign::Positive, BigUint::one(), exponent, prec, MODE)
}

// ---------------------------------------------------------------------------
// 1. The fuzz artifact itself, and its negative-exponent mirror.
// ---------------------------------------------------------------------------

/// Exact reproduction of
/// `fuzz/artifacts/hex_float_parse/timeout-0d4e887fd73ddc408ca9c29bb425d632eb35dcc3`:
/// byte 0 (`0x01`) drives `prec = 1 + 1 = 2` in the fuzz harness; the
/// remaining bytes, decoded as UTF-8, are this hex-float string.
const ARTIFACT_HEX_FLOAT: &str = "0X.05555555A5083E0P7905784";
const ARTIFACT_PREC: u32 = 2;

#[test]
fn fuzz_artifact_timeout_input_is_now_fast() {
    let parsed = BigFloat::from_hex_float(ARTIFACT_HEX_FLOAT, ARTIFACT_PREC)
        .unwrap_or_else(|e| panic!("{ARTIFACT_HEX_FLOAT:?} must parse: {e}"));
    // Sanity: this really is the multi-million-bit exponent that drove the
    // timeout, not some unrelated small value.
    assert!(
        parsed.exponent() > 1_000_000,
        "expected a multi-million binary exponent, got {}",
        parsed.exponent()
    );

    let t0 = Instant::now();
    let sci = parsed.to_scientific_string(10);
    assert!(
        t0.elapsed() < MUST_BE_FAST,
        "to_scientific_string(10) on the timeout artifact took {:?}, expected < {MUST_BE_FAST:?}",
        t0.elapsed()
    );

    let t0 = Instant::now();
    let eng = parsed.to_engineering_string(10);
    assert!(
        t0.elapsed() < MUST_BE_FAST,
        "to_engineering_string(10) on the timeout artifact took {:?}, expected < {MUST_BE_FAST:?}",
        t0.elapsed()
    );

    // This exponent is far beyond MAX_DECIMAL_FORMAT_BITS, so both renderers
    // fall back to the lossless hex form -- and that fallback must still be
    // exact and round-trip.
    let hex = parsed.to_hex_string();
    assert_eq!(
        sci, hex,
        "over-budget to_scientific_string must equal to_hex_string"
    );
    assert_eq!(
        eng, hex,
        "over-budget to_engineering_string must equal to_hex_string"
    );
    let back = BigFloat::from_hex_float(&hex, ARTIFACT_PREC).expect("hex fallback round-trips");
    assert_eq!(back, parsed);
}

/// The artifact only exercises a huge *positive* exponent. In the
/// comparator inside `decimal_magnitude`, a huge *negative* exponent swaps
/// which side of the cross-multiplied comparison carries the expensive
/// `pow10` call (`m * 10^|cand|` instead of `2^|e| * 10^cand`), so it is a
/// structurally distinct path that must be checked independently.
#[test]
fn fuzz_artifact_negative_exponent_mirror_is_also_fast() {
    let s = "0x1p-7905784";
    let parsed = BigFloat::from_hex_float(s, ARTIFACT_PREC)
        .unwrap_or_else(|e| panic!("{s:?} must parse: {e}"));
    assert!(parsed.exponent() < -1_000_000);

    let t0 = Instant::now();
    let sci = parsed.to_scientific_string(10);
    assert!(
        t0.elapsed() < MUST_BE_FAST,
        "to_scientific_string(10) on the negative-exponent mirror took {:?}",
        t0.elapsed()
    );
    let t0 = Instant::now();
    let eng = parsed.to_engineering_string(10);
    assert!(
        t0.elapsed() < MUST_BE_FAST,
        "to_engineering_string(10) on the negative-exponent mirror took {:?}",
        t0.elapsed()
    );

    let hex = parsed.to_hex_string();
    assert_eq!(sci, hex);
    assert_eq!(eng, hex);
    let back = BigFloat::from_hex_float(&hex, ARTIFACT_PREC).expect("hex fallback round-trips");
    assert_eq!(back, parsed);
}

/// A sweep of exponents between the crate's existing "obviously fine" floor
/// (1e5 bits, pinned decimal by `regression_unbounded_shift.rs`) and the
/// astronomically-out-of-budget ceiling must all resolve quickly, whichever
/// side of the decimal/hex boundary they land on. This is the general
/// version of the two targeted tests above: no exponent in `[EMIN, EMAX]`
/// may cost more than a fraction of a second.
#[test]
fn decimal_formatting_is_fast_across_the_full_exponent_range() {
    for exponent in [
        1,
        1_000,
        100_000,
        400_000,
        524_287, // one bit below the budget: still decimal
        524_288, // exactly at the budget: falls back to hex
        1_000_000,
        1 << 30,
        1 << 40, // BigFloat::EMAX
    ] {
        for x in [pow2(exponent, 64), pow2(-exponent, 64)] {
            let t0 = Instant::now();
            let _ = x.to_scientific_string(10);
            assert!(
                t0.elapsed() < MUST_BE_FAST,
                "to_scientific_string(10) at exponent {exponent} took {:?}",
                t0.elapsed()
            );
            let t0 = Instant::now();
            let _ = x.to_engineering_string(10);
            assert!(
                t0.elapsed() < MUST_BE_FAST,
                "to_engineering_string(10) at exponent {exponent} took {:?}",
                t0.elapsed()
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 2. The decimal/hex boundary introduced by MAX_DECIMAL_FORMAT_BITS is
//    pinned exactly, so a future change to the constant is a visible,
//    deliberate decision rather than a silent drift.
// ---------------------------------------------------------------------------

#[test]
fn decimal_hex_boundary_is_pinned_at_2_pow_19_bits() {
    // decimal_magnitude_cost_bits == mantissa.bit_length() + |exponent|.
    // `pow2` builds a 1-bit mantissa, so cost_bits == 1 + exponent here.
    let just_inside = pow2((1i64 << 19) - 1, 64); // cost_bits == 2^19: still decimal.
    let just_outside = pow2(1i64 << 19, 64); // cost_bits == 2^19 + 1: hex fallback.

    let inside_sci = just_inside.to_scientific_string(6);
    assert!(
        !inside_sci.starts_with("0x"),
        "one bit below the budget must still render in decimal, got {inside_sci:?}"
    );
    let outside_sci = just_outside.to_scientific_string(6);
    assert_eq!(
        outside_sci,
        just_outside.to_hex_string(),
        "one bit at/over the budget must fall back to hex"
    );

    let inside_eng = just_inside.to_engineering_string(6);
    assert!(!inside_eng.starts_with("0x"));
    let outside_eng = just_outside.to_engineering_string(6);
    assert_eq!(outside_eng, just_outside.to_hex_string());
}

// ---------------------------------------------------------------------------
// 3. Byte-identical decimal output for ordinary values well inside the
//    budget -- the algorithmic fix (binary exponentiation + caching) must
//    not perturb a single character of previously correct output.
// ---------------------------------------------------------------------------

#[test]
fn zero_is_unchanged() {
    let z = BigFloat::zero(53);
    assert_eq!(z.to_scientific_string(1), "0e0");
    assert_eq!(z.to_scientific_string(4), "0.000e0");
    assert_eq!(z.to_engineering_string(1), "0e0");
    assert_eq!(z.to_engineering_string(4), "0.000e0");
}

/// The smallest positive `f64` subnormal, `2^-1074`, carried through
/// exactly. At 53-bit precision `BigFloat` normalizes the mantissa to
/// exactly 53 significant bits (`2^52`, so `mantissa * 2^exponent ==
/// 2^52 * 2^-1126 == 2^-1074`) -- the *value* is exact either way. Expected
/// digits cross-checked against Python's `decimal.Decimal(2) ** -1074` at 60
/// digits of precision:
/// `4.94065645841246544176568792868221372365059802614324764425586e-324`.
#[test]
fn subnormal_magnitude_value_is_unchanged() {
    let tiny = BigFloat::from_f64(f64::from_bits(1), 53).expect("smallest positive f64 subnormal");
    assert_eq!(tiny.exponent(), -1126);
    assert_eq!(
        tiny.to_f64(),
        f64::from_bits(1),
        "value itself must still be 2^-1074"
    );
    assert_eq!(tiny.to_scientific_string(10), "4.940656458e-324");
    assert_eq!(tiny.to_engineering_string(10), "4.940656458e-324");

    let neg_tiny = BigFloat::from_f64(-f64::from_bits(1), 53).expect("negative smallest subnormal");
    assert_eq!(neg_tiny.to_scientific_string(10), "-4.940656458e-324");
}

/// A plain negative-exponent value, `2^-20`. Expected digits cross-checked
/// against `decimal.Decimal(2) ** -20 == 9.5367431640625E-7`.
#[test]
fn negative_exponent_value_is_unchanged() {
    let x = pow2(-20, 64);
    assert_eq!(x.to_scientific_string(10), "9.536743164e-7");
    assert_eq!(x.to_engineering_string(10), "953.6743164e-9");
}

/// Exact (non-negative) powers of ten -- integers, so exactly representable
/// in binary and exactly representable in decimal, giving an unambiguous
/// byte-exact oracle.
#[test]
fn exact_powers_of_ten_are_unchanged() {
    let ten = BigFloat::from_i64(10, 64, MODE);
    assert_eq!(ten.to_scientific_string(1), "1e1");
    assert_eq!(ten.to_scientific_string(5), "1.0000e1");
    assert_eq!(ten.to_engineering_string(5), "10.000e0");

    let million = BigFloat::from_i64(1_000_000, 64, MODE);
    assert_eq!(million.to_scientific_string(1), "1e6");
    assert_eq!(million.to_scientific_string(7), "1.000000e6");
    assert_eq!(million.to_engineering_string(7), "1.000000e6");

    let one = BigFloat::from_i64(1, 64, MODE);
    assert_eq!(one.to_scientific_string(1), "1e0");
    assert_eq!(one.to_engineering_string(1), "1e0");
}

/// `BigFloat::EMAX`/`EMIN` extremes: already known to fall back to hex
/// (`regression_unbounded_shift.rs`), but that file only exercises
/// `to_scientific_string` for the `EMIN`-magnitude value. Round out the
/// coverage for `to_engineering_string` too, and confirm both renderers
/// agree and round-trip.
#[test]
fn extreme_emax_emin_values_are_unchanged() {
    let hi = pow2(BigFloat::EMAX, 53);
    let lo = pow2(BigFloat::EMIN, 53);

    for x in [&hi, &lo] {
        let sci = x.to_scientific_string(6);
        let eng = x.to_engineering_string(6);
        let hex = x.to_hex_string();
        assert_eq!(sci, hex);
        assert_eq!(eng, hex);
        let back = BigFloat::from_hex_float(&hex, 53).expect("round-trips");
        assert_eq!(&back, x);
    }
}

/// Values already covered by `native_format.rs` (small, ordinary
/// magnitudes) must remain byte-identical -- a direct spot check alongside
/// that file rather than a duplicate of it.
#[test]
fn small_ordinary_values_are_unchanged() {
    let x = BigFloat::from_i64(12345, 64, MODE);
    assert_eq!(x.to_scientific_string(5), "1.2345e4");
    assert_eq!(x.to_engineering_string(5), "12.345e3");

    let y = BigFloat::from_f64(0.0009765625, 64).expect("0.0009765625 == 2^-10");
    assert_eq!(y.to_scientific_string(7), "9.765625e-4");
}
