//! Fuzz target: `BigUint`/`BigInt` radix parsing (`crates/oxinum-int/src/native/radix.rs`).
//!
//! `from_str_radix` / `from_radix` accept untrusted, attacker-controlled
//! strings in any radix 2..=36. The contract under fuzzing:
//!
//! 1. **No panics** — malformed input must be rejected with `Err`, never
//!    panic or hang, for any byte sequence (including non-UTF-8, decoded
//!    lossily exactly as `crates/oxinum/tests/fuzz_parse.rs` does for the
//!    top-level parser).
//! 2. **Round-trip** — any successfully parsed value, re-encoded via
//!    `to_radix` and re-parsed, must reproduce the exact same value. A
//!    mismatch here means either direction of the radix conversion is wrong.
//!
//! Coverage-guided fuzzing complements the bounded proptest sampling already
//! in `crates/oxinum-int/src/native/radix.rs`'s own test module by exploring
//! digit-string shapes a fixed proptest strategy would not generate.
//!
//! Run with:
//!   cargo fuzz run radix_roundtrip

#![no_main]

use libfuzzer_sys::fuzz_target;
use oxinum_core::{FromRadix, ToRadix};
use oxinum_int::native::{BigInt, BigUint};

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }
    // First byte selects the radix (2..=36, the only range either parser
    // accepts); the remaining bytes are the candidate digit string.
    let radix = 2 + (u32::from(data[0]) % 35); // 2..=36
    let s = String::from_utf8_lossy(&data[1..]);

    // --- BigUint::from_str_radix (inherent method) ---
    if let Ok(n) = BigUint::from_str_radix(&s, radix) {
        let encoded = n.to_radix(radix).unwrap_or_else(|e| {
            panic!("BigUint::to_radix({radix}) failed for a value from_str_radix produced: {e}")
        });
        let n2 = BigUint::from_str_radix(&encoded, radix).unwrap_or_else(|e| {
            panic!("re-parsing our own canonical radix-{radix} output {encoded:?} failed: {e}")
        });
        assert_eq!(
            n, n2,
            "BigUint radix round-trip mismatch for radix {radix}, input {s:?}"
        );
    }

    // --- BigInt::from_radix (FromRadix/ToRadix traits; exercises the
    // optional leading '-' sign-stripping path in radix.rs too) ---
    if let Ok(n) = BigInt::from_radix(&s, radix) {
        let encoded = n.to_radix(radix).unwrap_or_else(|e| {
            panic!("BigInt::to_radix({radix}) failed for a value from_radix produced: {e}")
        });
        let n2 = BigInt::from_radix(&encoded, radix).unwrap_or_else(|e| {
            panic!(
                "re-parsing our own canonical signed radix-{radix} output {encoded:?} failed: {e}"
            )
        });
        assert_eq!(
            n, n2,
            "BigInt radix round-trip mismatch for radix {radix}, input {s:?}"
        );
    }
});
