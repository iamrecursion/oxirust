//! CMX 3600 conformance tests pinning the *real* behaviour of the
//! `oximedia_edl::cmx3600` module against hand-authored EDL strings.
//!
//! IMPORTANT — verified parser limitation: `cmx3600::parse_cmx` splits each
//! event line on whitespace and treats columns 4..=7 as the four timecodes. It
//! does **not** model the optional numeric transition-duration column that real
//! CMX 3600 dissolve/wipe lines carry (e.g. `... D  025  <4 timecodes>`). When a
//! duration column is present it lands in column 4, fails `parse_cmx_timecode`,
//! and `parse_cmx` returns `Err`. The classification/round-trip tests below
//! therefore use the duration-less form that `parse_cmx` actually accepts, and
//! `test_cmx_rejects_duration_column` pins the rejection of the duration form.

use oximedia_edl::cmx3600::{parse_cmx, parse_cmx_timecode, serialize_cmx};

/// Two-event EDL in the form `parse_cmx` accepts: a straight cut followed by a
/// dissolve written WITHOUT a separate numeric duration column (the transition
/// token sits in column 3, four timecodes follow in columns 4..=7).
const EDL_CUT_DISSOLVE: &str = "TITLE: Two Event Test\nFCM: NON-DROP FRAME\n\n001  A001     V     C        01:00:00:00 01:00:05:00 01:00:00:00 01:00:05:00\n002  A002     V     D        01:00:05:00 01:00:10:00 01:00:05:00 01:00:10:00\n";

/// Two-event EDL: a wipe (duration-less form) followed by a key against black.
const EDL_WIPE_KEY: &str = "TITLE: Wipe And Key\n001  A001     V     W        01:00:00:00 01:00:02:00 01:00:00:00 01:00:02:00\n002  BL       V     K        01:00:02:00 01:00:04:00 01:00:02:00 01:00:04:00\n";

/// The directive's original samples that DO carry a numeric duration column.
/// `parse_cmx` rejects these (see `test_cmx_rejects_duration_column`).
const EDL_CUT_DISSOLVE_WITH_DURATION: &str = "TITLE: Two Event Test\nFCM: NON-DROP FRAME\n\n001  A001     V     C        01:00:00:00 01:00:05:00 01:00:00:00 01:00:05:00\n002  A002     V     D    025 01:00:05:00 01:00:10:00 01:00:05:00 01:00:10:00\n";
const EDL_WIPE_KEY_WITH_DURATION: &str = "TITLE: Wipe And Key\n001  A001     V     W    001 01:00:00:00 01:00:02:00 01:00:00:00 01:00:02:00\n002  BL       V     K        01:00:02:00 01:00:04:00 01:00:02:00 01:00:04:00\n";

/// Test 1 — cut/dissolve classification, title, and reel extraction.
#[test]
fn test_cmx_cut_then_dissolve() {
    let edl = parse_cmx(EDL_CUT_DISSOLVE).expect("EDL_CUT_DISSOLVE should parse");

    assert_eq!(edl.event_count(), 2, "two valid event lines expected");
    assert!(edl.events[0].is_cut(), "event 0 should be a straight cut");
    assert!(
        !edl.events[0].is_dissolve(),
        "a cut must not classify as a dissolve"
    );
    assert!(
        edl.events[1].is_dissolve(),
        "event 1 (transition 'D') should be a dissolve"
    );
    assert!(
        !edl.events[1].is_cut(),
        "a dissolve must not classify as a cut"
    );

    assert_eq!(
        edl.title.as_deref(),
        Some("Two Event Test"),
        "title from the TITLE: header"
    );

    // reels_used is deduplicated and sorted.
    assert_eq!(edl.reels_used(), vec!["A001", "A002"]);
}

/// Test 2 — wipe/key events: the dedicated CMX parser does NOT validate, so
/// both events survive, and a wipe classifies as neither cut nor dissolve.
#[test]
fn test_cmx_wipe_and_key() {
    let edl = parse_cmx(EDL_WIPE_KEY).expect("EDL_WIPE_KEY should parse via parse_cmx");

    assert_eq!(edl.event_count(), 2, "wipe + key = two events");
    assert!(
        edl.events[0].transition.starts_with('W'),
        "event 0 transition column should start with 'W', got {:?}",
        edl.events[0].transition
    );
    assert_eq!(
        edl.events[1].transition, "K",
        "event 1 should be a key transition"
    );

    // A wipe is neither a cut nor a dissolve.
    assert!(!edl.events[0].is_cut(), "wipe is not a cut");
    assert!(!edl.events[0].is_dissolve(), "wipe is not a dissolve");
}

/// Test 3 — `parse_cmx` -> `serialize_cmx` -> `parse_cmx` round-trip preserves
/// the event count and every load-bearing per-event field.
#[test]
fn test_cmx_serialize_roundtrip() {
    let original = parse_cmx(EDL_CUT_DISSOLVE).expect("original should parse");
    let serialized = serialize_cmx(&original);
    let reparsed = parse_cmx(&serialized).expect("serialized text should re-parse");

    assert_eq!(
        original.event_count(),
        reparsed.event_count(),
        "round-trip must preserve event count; serialized:\n{serialized}"
    );

    for (a, b) in original.events.iter().zip(reparsed.events.iter()) {
        assert_eq!(
            a.event_num, b.event_num,
            "event_num must survive round-trip"
        );
        assert_eq!(a.reel, b.reel, "reel must survive round-trip");
        assert_eq!(
            a.transition, b.transition,
            "transition must survive round-trip"
        );
        assert_eq!(
            a.source_in, b.source_in,
            "source_in must survive round-trip"
        );
        assert_eq!(
            a.source_out, b.source_out,
            "source_out must survive round-trip"
        );
    }
}

/// Real-behaviour finding — `parse_cmx` rejects event lines that carry a
/// numeric transition-duration column, because that column displaces the first
/// timecode into column 4 where it fails the timecode validity check. This pins
/// the documented limitation so a future parser upgrade (that learns to skip
/// the duration column) will trip this test and force the assertions to be
/// updated deliberately.
#[test]
fn test_cmx_rejects_duration_column() {
    let dissolve = parse_cmx(EDL_CUT_DISSOLVE_WITH_DURATION);
    assert!(
        dissolve.is_err(),
        "a dissolve line with a numeric duration column ('D 025 ...') is rejected; got {dissolve:?}"
    );
    let err = dissolve.expect_err("must be Err");
    assert!(
        err.contains("025"),
        "the error should name the offending duration token '025'; got {err:?}"
    );

    let wipe = parse_cmx(EDL_WIPE_KEY_WITH_DURATION);
    assert!(
        wipe.is_err(),
        "a wipe line with a numeric duration column ('W 001 ...') is rejected; got {wipe:?}"
    );
}

/// Test 4 — `parse_cmx_timecode` accepts well-formed timecodes, including the
/// drop-frame `;` separator on the last field and the maximum valid values.
#[test]
fn test_parse_cmx_timecode_valid() {
    assert_eq!(
        parse_cmx_timecode("01:02:03:04"),
        Some((1, 2, 3, 4)),
        "standard HH:MM:SS:FF"
    );
    assert_eq!(
        parse_cmx_timecode("01:00:00;00"),
        Some((1, 0, 0, 0)),
        "drop-frame ';' separator on the frames field"
    );
    assert_eq!(
        parse_cmx_timecode("23:59:59:29"),
        Some((23, 59, 59, 29)),
        "maximum valid 30fps NDF values"
    );
}

/// Test 5 — `parse_cmx_timecode` rejects out-of-range and malformed input by
/// returning `None` (never panicking), per the documented contract:
/// `len < 11`, `!= 4` parts, `min > 59`, or `sec > 59`.
#[test]
fn test_parse_cmx_timecode_out_of_range() {
    assert_eq!(parse_cmx_timecode("01:60:00:00"), None, "minutes > 59");
    assert_eq!(parse_cmx_timecode("01:00:60:00"), None, "seconds > 59");
    assert_eq!(parse_cmx_timecode("01:02"), None, "too short / wrong arity");
    assert_eq!(parse_cmx_timecode(""), None, "empty string");
}
