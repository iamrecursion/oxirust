//! Atomic rollback tests: if any clip in a linked batch would overlap, the
//! entire batch must be rolled back.

use oximedia_core::Rational;
use oximedia_edit::clip::{Clip, ClipType};
use oximedia_edit::error::EditError;
use oximedia_edit::group::LinkType;
use oximedia_edit::timeline::{Timeline, TrackType};

fn base() -> Timeline {
    Timeline::new(Rational::new(1, 1000), Rational::new(30, 1))
}

#[test]
fn test_move_rollback_on_overlap() {
    // Layout:
    //   Track 0 (Video): A [0..50], B [60..100]
    //   Track 1 (Audio): C [200..250] — linked to A
    //
    // Attempt: move A to t=55 → A[55..105] overlaps B[60..100] → error.
    // Expected: A stays at t=0, C stays at t=200.
    let mut tl = base();

    let vt = tl.add_track(TrackType::Video);
    let at = tl.add_track(TrackType::Audio);

    let clip_a = Clip::new(0, ClipType::Video, 0, 50);
    let clip_b = Clip::new(0, ClipType::Video, 60, 40);
    let clip_c = Clip::new(0, ClipType::Audio, 200, 50);

    let id_a = tl.add_clip(vt, clip_a).expect("add A");
    let _id_b = tl.add_clip(vt, clip_b).expect("add B");
    let id_c = tl.add_clip(at, clip_c).expect("add C");

    // Link A ↔ C.
    tl.links.add_link(id_a, id_c, LinkType::VideoAudio);

    // Attempt: move A to t=55 (A[55..105] overlaps B[60..100]).
    let result = tl.move_clip(id_a, 55);
    assert!(
        matches!(result, Err(EditError::ClipOverlap(..))),
        "expected ClipOverlap error, got: {:?}",
        result
    );

    // Rollback: A and C must be at their original positions.
    let a_start = tl.get_clip(id_a).expect("A exists").timeline_start;
    let c_start = tl.get_clip(id_c).expect("C exists").timeline_start;

    assert_eq!(a_start, 0, "A must be rolled back to t=0");
    assert_eq!(c_start, 200, "C must be rolled back to t=200");
}

#[test]
fn test_successful_batch_move_applies_all() {
    // Layout:
    //   Track 0 (Video): A [0..50]
    //   Track 1 (Audio): B [0..50] — linked to A
    //   No other clips → move to t=300 cannot overlap anything.
    let mut tl = base();

    let vt = tl.add_track(TrackType::Video);
    let at = tl.add_track(TrackType::Audio);

    let clip_a = Clip::new(0, ClipType::Video, 0, 50);
    let clip_b = Clip::new(0, ClipType::Audio, 0, 50);

    let id_a = tl.add_clip(vt, clip_a).expect("add A");
    let id_b = tl.add_clip(at, clip_b).expect("add B");
    tl.links.add_link(id_a, id_b, LinkType::VideoAudio);

    tl.move_clip(id_a, 300).expect("batch move should succeed");

    assert_eq!(tl.get_clip(id_a).unwrap().timeline_start, 300);
    assert_eq!(tl.get_clip(id_b).unwrap().timeline_start, 300);
}

#[test]
fn test_zero_delta_is_noop() {
    // Moving a clip to its current position is a no-op (delta = 0).
    let mut tl = base();
    let vt = tl.add_track(TrackType::Video);
    let clip = Clip::new(0, ClipType::Video, 100, 50);
    let id = tl.add_clip(vt, clip).expect("add");

    tl.move_clip(id, 100).expect("noop move");
    assert_eq!(tl.get_clip(id).unwrap().timeline_start, 100);
}
