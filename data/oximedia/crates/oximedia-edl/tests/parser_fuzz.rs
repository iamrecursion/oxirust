//! Robustness and silent-skip-contract tests for the *main* `parse_edl`
//! entry point (the nom-based parser in `oximedia_edl::parser`).
//!
//! `parse_edl` SILENTLY SKIPS event lines it cannot parse — it returns
//! `Ok(edl)` with fewer events. It returns `Err(EdlError::Parse{..})` only when
//! a *successfully parsed* event fails `EdlEvent::validate()` (e.g. an inverted
//! source range, or a wipe/dissolve whose required fields are absent).
//!
//! The fuzz test uses a small, deterministically-seeded xorshift PRNG (no
//! external crate, no proptest) to feed ~1000 random ASCII lines through the
//! parser and assert it never panics or hangs.

use oximedia_edl::{parse_edl, EdlError};

/// A truncated event line (only the first timecode column present). The nom
/// parser cannot complete it, so it is silently skipped -> zero events.
const EDL_TRUNCATED: &str = "001  A001  V  C  01:00:00:00\n";

/// An event line whose first timecode column is garbage ("BADTC"). The nom
/// timecode parser rejects it, the line fails to parse, and is silently
/// skipped -> zero events.
const EDL_BAD_TC: &str = "001  AX  V  C  BADTC 00:00:01:00 00:00:00:00 00:00:01:00\n";

/// Pure garbage including control bytes. No line resembles an event ->
/// zero events, no panic.
const EDL_GARBAGE: &str = "%%% not an edl @@@\n\x01\x02 random\nHELLO WORLD\n";

/// A syntactically valid event whose source_in (01:00:05:00) is greater than
/// its source_out (01:00:00:00). The event parses, but `validate()` rejects the
/// inverted source range, surfacing as `EdlError::Parse`.
const EDL_INVERTED_TC: &str = "001  AX  V  C  01:00:05:00 01:00:00:00 01:00:00:00 01:00:05:00\n";

/// A wipe event fed through the MAIN parser. The nom parser never populates
/// `wipe_pattern` (the numeric "001" is consumed as the transition duration),
/// so `validate()` rejects it for the missing wipe pattern. See the comment in
/// `test_wipe_without_pattern_is_rejected` for the design decision.
const EDL_WIPE_KEY: &str = "TITLE: Wipe And Key\n001  A001     V     W    001 01:00:00:00 01:00:02:00 01:00:00:00 01:00:02:00\n002  BL       V     K        01:00:02:00 01:00:04:00 01:00:02:00 01:00:04:00\n";

/// Test 6 — silent-skip contract: malformed event lines yield `Ok` with an
/// empty event list and never panic.
#[test]
fn test_parse_edl_silently_skips_malformed() {
    for (name, input) in [
        ("EDL_TRUNCATED", EDL_TRUNCATED),
        ("EDL_BAD_TC", EDL_BAD_TC),
        ("EDL_GARBAGE", EDL_GARBAGE),
    ] {
        let edl = parse_edl(input).unwrap_or_else(|e| {
            panic!("{name} should parse to Ok under the silent-skip contract, got Err: {e}")
        });
        assert!(
            edl.events.is_empty(),
            "{name} should yield zero events; got {}",
            edl.events.len()
        );
    }
}

/// Test 7 — a parsed-but-invalid event (inverted source range) surfaces as
/// `Err(EdlError::Parse{..})`.
#[test]
fn test_parse_edl_inverted_timecode_errors() {
    let err = parse_edl(EDL_INVERTED_TC).expect_err("inverted source range must error");
    assert!(
        matches!(err, EdlError::Parse { .. }),
        "validation failures surface as EdlError::Parse; got {err:?}"
    );
}

/// Test 8 — wipe-without-pattern is rejected by the main parser.
///
/// DESIGN DECISION / FINDING: no production bug exists. `EdlEvent::validate()`
/// (src/event.rs) already enforces `wipe_pattern.is_some()` for `EditType::Wipe`
/// (and `transition_duration.is_some()` for both Wipe and Dissolve). The nom
/// `parse_edl` path consumes the numeric "001" as the transition duration and
/// never sets `wipe_pattern`, so the parsed wipe event fails validation and the
/// error is propagated as `EdlError::Parse`. We therefore assert `Err` directly
/// — `validate()` did not need fixing.
///
/// (Contrast with `test_cmx_wipe_and_key`, which feeds the *same* sample to the
/// dedicated `cmx3600::parse_cmx`, which performs no validation and keeps both
/// events.)
#[test]
fn test_wipe_without_pattern_is_rejected() {
    let result = parse_edl(EDL_WIPE_KEY);
    let err = result
        .expect_err("a wipe event with no wipe_pattern must fail validation in the main parser");
    assert!(
        matches!(err, EdlError::Parse { .. }),
        "missing wipe pattern surfaces as EdlError::Parse; got {err:?}"
    );
}

/// A tiny deterministic xorshift64* PRNG. Seeded with a fixed constant so the
/// fuzz corpus is fully reproducible run-to-run. Avoids any external crate.
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    const fn new(seed: u64) -> Self {
        // xorshift64* requires a non-zero seed.
        Self {
            state: if seed == 0 {
                0x9E37_79B9_7F4A_7C15
            } else {
                seed
            },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform-ish integer in `[0, n)` for small `n`.
    fn below(&mut self, n: u32) -> u32 {
        (self.next_u64() % u64::from(n)) as u32
    }
}

/// Test 9 — fuzz: ~1000 random ASCII lines must yield `Ok` or `Err` but never
/// panic or hang. The PRNG is deterministically seeded for reproducibility.
#[test]
fn test_parse_edl_fuzz_never_panics() {
    let mut rng = XorShift64::new(0xDEAD_BEEF_CAFE_F00D);

    // A pool of printable-ASCII bytes plus a few control/structural characters
    // that commonly appear in (or break) EDL lines.
    let alphabet: &[u8] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ :;./*\t\x01-";

    for _ in 0..1000 {
        let line_len = rng.below(40); // 0..=39 chars per line
        let mut line = String::with_capacity(line_len as usize);
        for _ in 0..line_len {
            let idx = rng.below(alphabet.len() as u32) as usize;
            line.push(alphabet[idx] as char);
        }
        line.push('\n');

        // The contract is simply: this returns without panicking or hanging.
        // Both Ok and Err are acceptable outcomes for random input.
        match parse_edl(&line) {
            Ok(_) | Err(_) => {}
        }
    }

    // Also fuzz multi-line blobs to exercise event-accumulation paths.
    for _ in 0..200 {
        let n_lines = 1 + rng.below(6);
        let mut blob = String::new();
        for _ in 0..n_lines {
            let line_len = rng.below(50);
            for _ in 0..line_len {
                let idx = rng.below(alphabet.len() as u32) as usize;
                blob.push(alphabet[idx] as char);
            }
            blob.push('\n');
        }
        match parse_edl(&blob) {
            Ok(_) | Err(_) => {}
        }
    }
}
