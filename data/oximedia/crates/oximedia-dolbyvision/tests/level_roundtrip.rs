//! Integration tests pinning the public Level round-trip conformance verifiers.
//!
//! Each test drives one of the seven public `verify_*_roundtrip` functions
//! exposed by [`oximedia_dolbyvision::conformance`]. Together they ensure that
//! the bitstream writer and parser agree on the binary representation of every
//! fully-implemented Dolby Vision metadata level.
//!
//! ## Coverage map
//!
//! | Level | Status                                                            |
//! |-------|-------------------------------------------------------------------|
//! | L1    | Public verifier — exercised here.                                 |
//! | L2    | Internal `#[cfg(test)]` test in `conformance.rs` (no public fn).   |
//! | L3    | Reserved — no encoder/decoder.                                    |
//! | L4    | Internal `#[cfg(test)]` test in `conformance.rs` (no public fn).   |
//! | L5    | Public verifier — exercised here.                                 |
//! | L6    | Public verifier — exercised here.                                 |
//! | L7    | Internal `#[cfg(test)]` test in `conformance.rs` (no public fn).   |
//! | L8    | Public verifier — exercised here.                                 |
//! | L9    | Public verifier — exercised here.                                 |
//! | L10   | Reserved — no encoder/decoder.                                    |
//! | L11   | Public verifier — exercised here.                                 |
//!
//! L2/L4/L7 round-trip behaviour is verified by `conformance.rs`'s internal
//! `#[cfg(test)]` module (those levels have no public verifier). L3/L10 are
//! reserved in the implementation and have no encoder, so no round-trip exists.

use oximedia_dolbyvision::conformance;

#[test]
fn level1_roundtrip() {
    conformance::verify_level1_roundtrip().expect("L1");
}

#[test]
fn level5_roundtrip() {
    conformance::verify_level5_roundtrip().expect("L5");
}

#[test]
fn level6_roundtrip() {
    conformance::verify_level6_roundtrip().expect("L6");
}

#[test]
fn level8_roundtrip() {
    conformance::verify_level8_roundtrip().expect("L8");
}

#[test]
fn level9_roundtrip() {
    conformance::verify_level9_roundtrip().expect("L9");
}

#[test]
fn level11_roundtrip() {
    conformance::verify_level11_roundtrip().expect("L11");
}

#[test]
fn profile8_minimal_roundtrip() {
    conformance::verify_profile8_minimal_roundtrip().expect("Profile 8 minimal");
}

/// Drive every public Level verifier within a single test, in level order.
///
/// This guards the combined invariant that the seven publicly-implemented
/// levels each survive a write-then-parse round-trip. L2/L4/L7 are covered by
/// the internal `#[cfg(test)]` module inside `conformance.rs` (they have no
/// public verifier), and L3/L10 are reserved with no encoder.
#[test]
fn all_public_levels_roundtrip_in_sequence() {
    conformance::verify_profile8_minimal_roundtrip().expect("Profile 8 minimal");
    conformance::verify_level1_roundtrip().expect("L1");
    conformance::verify_level5_roundtrip().expect("L5");
    conformance::verify_level6_roundtrip().expect("L6");
    conformance::verify_level8_roundtrip().expect("L8");
    conformance::verify_level9_roundtrip().expect("L9");
    conformance::verify_level11_roundtrip().expect("L11");
}
