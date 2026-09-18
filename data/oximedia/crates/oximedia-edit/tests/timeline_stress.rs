//! Stress tests: verify that the timeline stays correct with many clips and that
//! `get_clips_at` queries remain functional as clip count grows.
//!
//! Clip counts are deliberately modest so tests run fast in debug mode.

use oximedia_core::Rational;
use oximedia_edit::{Clip, ClipType, Timeline, TrackType};

/// Build a timeline with `n` sequential video clips of `duration_ms` each.
fn build_sequential_timeline(n: usize, duration_ms: i64) -> Timeline {
    let mut tl = Timeline::new(Rational::new(1, 1000), Rational::new(30, 1));
    let vt = tl.add_track(TrackType::Video);
    for i in 0..n {
        let start = i as i64 * duration_ms;
        let clip = Clip::new(i as u64 + 1, ClipType::Video, start, duration_ms);
        tl.add_clip(vt, clip).expect("add_clip must not fail");
    }
    tl
}

// ── Basic structural tests ────────────────────────────────────────────────────

#[test]
fn test_stress_100_clips_correct_count() {
    let tl = build_sequential_timeline(100, 1000);
    assert_eq!(tl.clip_count(), 100);
}

#[test]
fn test_stress_200_clips_correct_count() {
    let tl = build_sequential_timeline(200, 500);
    assert_eq!(tl.clip_count(), 200);
}

#[test]
fn test_stress_clips_do_not_overlap() {
    let tl = build_sequential_timeline(50, 1000);
    // For any two distinct clips, they must not overlap.
    let all_clips = tl.tracks[0].clips.as_slice();
    for (i, a) in all_clips.iter().enumerate() {
        for b in all_clips.iter().skip(i + 1) {
            let overlap =
                a.timeline_start < b.timeline_end() && b.timeline_start < a.timeline_end();
            assert!(
                !overlap,
                "clips {} and {} must not overlap: [{},{}), [{},{})",
                a.id,
                b.id,
                a.timeline_start,
                a.timeline_end(),
                b.timeline_start,
                b.timeline_end()
            );
        }
    }
}

// ── get_clips_at correctness ──────────────────────────────────────────────────

#[test]
fn test_get_clips_at_returns_correct_clip_at_start() {
    let tl = build_sequential_timeline(10, 1000);
    // Clip 1 starts at 0 and ends at 1000; querying position 0 must return it.
    let result = tl.get_clips_at(0);
    assert_eq!(result.len(), 1, "exactly one clip at position 0");
    // id assigned by add_clip from next_clip_id starting at 1.
    assert_eq!(result[0].1.id, 1);
}

#[test]
fn test_get_clips_at_returns_correct_clip_in_middle() {
    let tl = build_sequential_timeline(10, 1000);
    // Clip 5 spans [4000, 5000). Query at 4500.
    let result = tl.get_clips_at(4500);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].1.id, 5);
}

#[test]
fn test_get_clips_at_returns_correct_clip_at_last() {
    let tl = build_sequential_timeline(10, 1000);
    // Last clip (id=10) spans [9000, 10000). Query at 9999.
    let result = tl.get_clips_at(9999);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].1.id, 10);
}

#[test]
fn test_get_clips_at_empty_at_boundary() {
    let tl = build_sequential_timeline(5, 1000);
    // Position 5000 is exactly at the end of the last clip (exclusive end), so nothing.
    let result = tl.get_clips_at(5000);
    assert!(
        result.is_empty(),
        "position at exclusive end must return nothing"
    );
}

#[test]
fn test_get_clips_at_empty_before_timeline() {
    let tl = build_sequential_timeline(5, 1000);
    let result = tl.get_clips_at(-1);
    assert!(result.is_empty(), "negative position must return nothing");
}

// ── Multi-track stress test ────────────────────────────────────────────────────

#[test]
fn test_stress_multi_track_clip_counts() {
    let mut tl = Timeline::new(Rational::new(1, 1000), Rational::new(30, 1));
    let vt = tl.add_track(TrackType::Video);
    let at = tl.add_track(TrackType::Audio);

    for i in 0..50u64 {
        let start = i as i64 * 1000;
        tl.add_clip(vt, Clip::new(i + 1, ClipType::Video, start, 1000))
            .expect("add video clip");
        tl.add_clip(at, Clip::new(i + 51, ClipType::Audio, start, 1000))
            .expect("add audio clip");
    }

    assert_eq!(
        tl.clip_count(),
        100,
        "50 video + 50 audio clips = 100 total"
    );
}

#[test]
fn test_stress_get_clips_at_multi_track() {
    let mut tl = Timeline::new(Rational::new(1, 1000), Rational::new(30, 1));
    let vt = tl.add_track(TrackType::Video);
    let at = tl.add_track(TrackType::Audio);

    for i in 0..10u64 {
        let start = i as i64 * 1000;
        tl.add_clip(vt, Clip::new(i + 1, ClipType::Video, start, 1000))
            .expect("add video clip");
        tl.add_clip(at, Clip::new(i + 11, ClipType::Audio, start, 1000))
            .expect("add audio clip");
    }

    // At position 5500 there should be one video clip and one audio clip.
    let result = tl.get_clips_at(5500);
    assert_eq!(result.len(), 2, "must find one clip per track at 5500");
}

// ── Query timing sanity ───────────────────────────────────────────────────────

#[test]
fn test_stress_repeated_queries_are_consistent() {
    let tl = build_sequential_timeline(100, 500);
    // Query the same position 1000 times and verify consistency.
    let first = tl.get_clips_at(24999);
    for _ in 0..1000 {
        let result = tl.get_clips_at(24999);
        assert_eq!(
            result.len(),
            first.len(),
            "repeated queries must return consistent results"
        );
        if !result.is_empty() && !first.is_empty() {
            assert_eq!(result[0].1.id, first[0].1.id);
        }
    }
}

#[test]
fn test_stress_all_positions_covered() {
    // Verify that for every clip start position, get_clips_at finds exactly one clip.
    let n = 30usize;
    let dur = 1000i64;
    let tl = build_sequential_timeline(n, dur);
    for i in 0..n {
        let pos = i as i64 * dur;
        let result = tl.get_clips_at(pos);
        assert_eq!(
            result.len(),
            1,
            "position {pos} (clip {}) must return exactly one clip",
            i + 1
        );
    }
}
