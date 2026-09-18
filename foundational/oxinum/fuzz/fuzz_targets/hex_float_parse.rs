//! Fuzz target: `BigFloat` hex-float parsing / rendering
//! (`crates/oxinum-float/src/native/float.rs`,
//! `crates/oxinum-float/src/native/format_ext.rs`).
//!
//! `BigFloat::from_hex_float` parses an untrusted, attacker-controlled
//! C99-`%a`-style string. The contract under fuzzing:
//!
//! 1. **No panics** — malformed input must be rejected with `Err`, never
//!    panic or hang, for any byte sequence (including non-UTF-8, decoded
//!    lossily exactly as `crates/oxinum/tests/fuzz_parse.rs` does for the
//!    top-level parser).
//! 2. **Exact round-trip** — for any successfully parsed *finite* value,
//!    `to_hex_string()` is documented as the exact inverse of
//!    `from_hex_float`, so re-parsing the rendered string must reproduce the
//!    original value bit-for-bit (`total_cmp` equal, which also checks
//!    precision/sign — stricter than numeric `==`).
//! 3. **Decimal renderers never panic** — `to_scientific_string` /
//!    `to_engineering_string` must not panic even for the extreme exponents
//!    this parser can produce (regression coverage for the
//!    `MAX_EXACT_CONVERSION_BITS` hex-string fallback).
//!
//! Run with:
//!   cargo fuzz run hex_float_parse

#![no_main]

use libfuzzer_sys::fuzz_target;
use oxinum_float::native::{BigFloat, RoundingMode};

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }
    // First byte derives the working precision and rounding mode. Kept
    // `prec >= 1`: `BigFloat`'s precision invariant (`prec > 0`) is a
    // documented caller precondition, not part of the untrusted-string
    // surface this target is exercising, so we must never violate it
    // ourselves. The remaining bytes are the candidate hex-float string.
    let prec = 1 + u32::from(data[0]);
    let mode = match data[0] % 7 {
        0 => RoundingMode::HalfEven,
        1 => RoundingMode::HalfAway,
        2 => RoundingMode::HalfToZero,
        3 => RoundingMode::ToZero,
        4 => RoundingMode::ToInf,
        5 => RoundingMode::ToNegInf,
        _ => RoundingMode::AwayFromZero,
    };
    let s = String::from_utf8_lossy(&data[1..]);

    let parsed = match BigFloat::from_hex_float(&s, prec) {
        Ok(v) => v,
        Err(_) => return,
    };

    if parsed.is_finite() {
        let rendered = parsed.to_hex_string();
        let reparsed = BigFloat::from_hex_float(&rendered, prec).unwrap_or_else(|e| {
            panic!("re-parsing our own to_hex_string output {rendered:?} failed: {e}")
        });
        assert_eq!(
            parsed.total_cmp(&reparsed),
            std::cmp::Ordering::Equal,
            "hex-float round-trip mismatch: {s:?} (prec {prec}) -> {rendered:?}"
        );
    }

    // Regression coverage: these must never panic, including for the
    // extreme exponents `from_hex_float` can produce (e.g. `0x1p9999999999`)
    // that drove the MAX_EXACT_CONVERSION_BITS hex-string fallback.
    let _ = parsed.to_scientific_string(10);
    let _ = parsed.to_engineering_string(10);

    // `mode` is otherwise unused by the parse/render surface under test;
    // fold it into a trivial rounding op so it still affects the corpus the
    // fuzzer explores (via `add_ref_with_mode`'s internal rounding) without
    // asserting anything about the result.
    let _ = parsed.add_ref_with_mode(&BigFloat::zero(prec), mode);
});
