//! Behavior-pinning integration tests for SRT parsing, overlap detection, and
//! multi-track subtitle merging.
//!
//! All inputs are in-memory literals (no file I/O, no clocks). These tests
//! document the *current* observed behavior of the crate so that regressions
//! are caught; where the behavior is lenient (e.g. a missing blank line between
//! SRT cues), the test pins the present behavior and the rationale is recorded
//! in a comment rather than asserting an idealized outcome.

use oximedia_subtitle::error::SubtitleError;
use oximedia_subtitle::overlap_detect::{
    DetectableCue, OverlapDetector, OverlapReport, OverlapType,
};
use oximedia_subtitle::parser::srt::{parse, parse_srt};
use oximedia_subtitle::subtitle_merge::{MergeStrategy, SubtitleEntry, SubtitleMerger};

// ============================================================================
// SRT parser edge cases (BOM, CRLF, lone CR, missing blank line)
// ============================================================================

/// 1. A UTF-8 BOM prefix on the byte stream must be stripped, and the single
///    cue parsed with correct timing and text.
#[test]
fn test_srt_utf8_bom_is_stripped() {
    let mut bytes = vec![0xEF, 0xBB, 0xBF];
    bytes.extend_from_slice(b"1\n00:00:01,000 --> 00:00:04,000\nHello\n\n");

    let subs = parse(&bytes).expect("BOM-prefixed SRT should parse");
    assert_eq!(subs.len(), 1, "exactly one cue expected after BOM");
    assert_eq!(subs[0].start_time, 1000);
    assert_eq!(subs[0].end_time, 4000);
    assert_eq!(subs[0].text, "Hello");
}

/// 2. A CRLF-terminated body must parse identically to the LF-terminated body,
///    field-by-field, because `parse_srt` normalizes `\r\n` to `\n`.
#[test]
fn test_srt_crlf_equals_lf_field_by_field() {
    let lf = "1\n00:00:01,000 --> 00:00:04,000\nFirst line\n\n\
              2\n00:00:05,000 --> 00:00:08,000\nSecond line\n\n";
    let crlf = "1\r\n00:00:01,000 --> 00:00:04,000\r\nFirst line\r\n\r\n\
                2\r\n00:00:05,000 --> 00:00:08,000\r\nSecond line\r\n\r\n";

    let lf_subs = parse_srt(lf).expect("LF body should parse");
    let crlf_subs = parse_srt(crlf).expect("CRLF body should parse");

    assert_eq!(lf_subs.len(), 2);
    assert_eq!(crlf_subs.len(), lf_subs.len(), "cue count must match");
    for (a, b) in lf_subs.iter().zip(crlf_subs.iter()) {
        assert_eq!(a.start_time, b.start_time, "start_time must match");
        assert_eq!(a.end_time, b.end_time, "end_time must match");
        assert_eq!(a.text, b.text, "text must match");
    }
}

/// 3. A body using lone `\r` (old-Mac line endings) is NOT normalized by
///    `parse_srt` (which only handles `\r\n`). nom's `line_ending` therefore
///    fails to match, so parsing returns `Err(ParseError)` without panicking.
#[test]
fn test_srt_lone_cr_yields_parse_error() {
    let body = "1\r00:00:01,000 --> 00:00:04,000\rHello\r\r";

    let result = parse_srt(body);
    assert!(result.is_err(), "lone-CR body should not parse");
    match result {
        Err(err) => assert!(
            matches!(err, SubtitleError::ParseError(_)),
            "expected ParseError variant, got {err:?}"
        ),
        Ok(subs) => panic!("expected Err for lone-CR body, got {} cues", subs.len()),
    }
}

/// 4. Missing blank line between two cues. CURRENT (lenient) behavior: the
///    `subtitle_text` reader keeps consuming non-empty lines until a blank
///    line or EOF, so the second cue's number/timestamp/text are absorbed into
///    the FIRST cue's `text`, yielding exactly ONE `Subtitle` spanning the
///    first cue's timing.
///
///    For broadcast QC this is arguably wrong (a hard cue boundary is silently
///    lost), but the test PINS the present behavior so any future tightening of
///    the parser is a deliberate, reviewed change rather than an accident.
#[test]
fn test_srt_missing_blank_line_is_lenient_single_cue() {
    let body = "1\n00:00:01,000 --> 00:00:02,000\nFirst\n\
                2\n00:00:03,000 --> 00:00:04,000\nSecond\n\n";

    let subs = parse_srt(body).expect("lenient parser should still return Ok");
    assert_eq!(
        subs.len(),
        1,
        "missing blank line currently collapses into a single cue"
    );
    assert_eq!(subs[0].start_time, 1000, "first cue start retained");
    assert_eq!(subs[0].end_time, 2000, "first cue end retained");
    assert!(
        subs[0].text.contains("Second"),
        "second cue text absorbed: {:?}",
        subs[0].text
    );
    assert!(
        subs[0].text.contains("00:00:03,000"),
        "second cue timestamp line absorbed as text: {:?}",
        subs[0].text
    );
}

// ============================================================================
// Overlap detection
// ============================================================================

/// 5. Two cues sharing a partial window: idx0=[0,2000], idx1=[1000,3000].
#[test]
fn test_overlap_partial_pair() {
    let cues = vec![
        DetectableCue::new(0, 0, 2000, "a"),
        DetectableCue::new(1, 1000, 3000, "b"),
    ];
    let overlaps = OverlapDetector::new().find_overlaps(&cues);

    assert_eq!(overlaps.len(), 1, "exactly one overlapping pair");
    let o = &overlaps[0];
    assert_eq!(o.cue_a, 0);
    assert_eq!(o.cue_b, 1);
    assert_eq!(o.overlap_type, OverlapType::Partial);
    assert_eq!(o.overlap_start_ms, 1000);
    assert_eq!(o.overlap_end_ms, 2000);
    assert_eq!(o.duration_ms(), 1000);
}

/// 6. Three cues forming a chain of two partial overlaps:
///    0=[0,2000], 1=[1000,3000], 2=[2500,4000]. Cue 1 participates in both.
#[test]
fn test_overlap_chain_report_counts() {
    let cues = vec![
        DetectableCue::new(0, 0, 2000, "a"),
        DetectableCue::new(1, 1000, 3000, "b"),
        DetectableCue::new(2, 2500, 4000, "c"),
    ];
    let overlaps = OverlapDetector::new().find_overlaps(&cues);
    assert_eq!(overlaps.len(), 2, "two overlapping pairs expected");

    let report = OverlapReport::from_overlaps(overlaps);
    assert_eq!(report.total_overlaps, 2);
    assert_eq!(report.partial_overlap_count, 2);
    assert_eq!(report.full_overlap_count, 0);
    assert_eq!(
        report.overlaps_for_cue(1).len(),
        2,
        "cue 1 sits in both overlaps"
    );

    // The detected pairs must be exactly {(0,1), (1,2)}.
    let mut pairs: Vec<(usize, usize)> =
        report.overlaps.iter().map(|o| (o.cue_a, o.cue_b)).collect();
    pairs.sort_unstable();
    assert_eq!(pairs, vec![(0, 1), (1, 2)]);
}

/// 7. Full containment: idx0=[0,5000] entirely contains idx1=[1000,3000].
#[test]
fn test_overlap_full_containment() {
    let cues = vec![
        DetectableCue::new(0, 0, 5000, "outer"),
        DetectableCue::new(1, 1000, 3000, "inner"),
    ];
    let overlaps = OverlapDetector::new().find_overlaps(&cues);

    assert_eq!(overlaps.len(), 1);
    let o = &overlaps[0];
    assert!(o.is_full(), "containment must be flagged full");
    assert_eq!(o.overlap_type, OverlapType::Full);
    assert_eq!(o.overlap_start_ms, 1000);
    assert_eq!(o.overlap_end_ms, 3000);
}

// ============================================================================
// Multi-track merge strategies
// ============================================================================

/// Build the canonical conflicting two-track scenario used by the merge tests:
/// track A = [0,2000] "a"; track B = [1000,3000] "b". The ranges overlap.
fn conflicting_tracks() -> (Vec<SubtitleEntry>, Vec<SubtitleEntry>) {
    (
        vec![SubtitleEntry::new(0, 0, 2000, "a")],
        vec![SubtitleEntry::new(0, 1000, 3000, "b")],
    )
}

/// 8. PreferFirst keeps the earlier-added track's entry on conflict.
#[test]
fn test_merge_prefer_first() {
    let (a, b) = conflicting_tracks();
    let mut merger = SubtitleMerger::new(MergeStrategy::PreferFirst);
    merger.add_track(a);
    merger.add_track(b);
    let result = merger.merge();

    assert_eq!(result.entry_count(), 1);
    assert_eq!(result.entries[0].text, "a");
    assert_eq!(result.conflict_count(), 1);
}

/// 9. PreferLast replaces the conflicting entry with the later-added one.
#[test]
fn test_merge_prefer_last() {
    let (a, b) = conflicting_tracks();
    let mut merger = SubtitleMerger::new(MergeStrategy::PreferLast);
    merger.add_track(a);
    merger.add_track(b);
    let result = merger.merge();

    assert_eq!(result.entry_count(), 1);
    assert_eq!(result.entries[0].text, "b");
    assert_eq!(result.conflict_count(), 1);
}

/// 10. KeepAll retains both entries, flags the conflicting one, and the output
///     is sorted by start time (so "a" at 0ms precedes "b" at 1000ms).
#[test]
fn test_merge_keep_all_sorted_and_flagged() {
    let (a, b) = conflicting_tracks();
    let mut merger = SubtitleMerger::new(MergeStrategy::KeepAll);
    merger.add_track(a);
    merger.add_track(b);
    let result = merger.merge();

    assert_eq!(result.entry_count(), 2);
    assert_eq!(result.conflict_count(), 1);
    assert_eq!(result.conflicted_entries().len(), 1);
    // Sorted by start_ms.
    assert_eq!(result.entries[0].text, "a");
    assert_eq!(result.entries[1].text, "b");
}

/// 11. DropOnConflict drops the incoming (later) entry, leaving "a".
#[test]
fn test_merge_drop_on_conflict() {
    let (a, b) = conflicting_tracks();
    let mut merger = SubtitleMerger::new(MergeStrategy::DropOnConflict);
    merger.add_track(a);
    merger.add_track(b);
    let result = merger.merge();

    assert_eq!(result.entry_count(), 1);
    assert_eq!(result.entries[0].text, "a");
    assert_eq!(result.conflict_count(), 1);
}

/// 12. Disjoint tracks (no time overlap) produce two entries and zero conflicts
///     under every strategy.
#[test]
fn test_merge_disjoint_no_conflict_all_strategies() {
    for strategy in [
        MergeStrategy::PreferFirst,
        MergeStrategy::PreferLast,
        MergeStrategy::KeepAll,
        MergeStrategy::DropOnConflict,
    ] {
        let mut merger = SubtitleMerger::new(strategy);
        merger.add_track(vec![SubtitleEntry::new(0, 0, 1000, "a")]);
        merger.add_track(vec![SubtitleEntry::new(0, 2000, 3000, "b")]);
        let result = merger.merge();

        assert_eq!(
            result.entry_count(),
            2,
            "{} should keep both disjoint entries",
            strategy.name()
        );
        assert_eq!(
            result.conflict_count(),
            0,
            "{} should report no conflicts for disjoint tracks",
            strategy.name()
        );
    }
}
