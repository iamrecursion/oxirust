//! Basic magnetic snap tests: snap to clip edges and disabled-by-default behaviour.
//!
//! Layout used in most tests:
//!   Track 0 (Video): anchor [0..100]
//!   Track 0 (Video): mover  [500..550]  ← moved to ~108, snaps to 100 (edge)
//!
//! At t=108 the mover is [108..158] — no overlap with anchor [0..100].
//! After snap to t=100 the mover is [100..150] — still no overlap.

use oximedia_core::Rational;
use oximedia_edit::clip::{Clip, ClipType};
use oximedia_edit::magnetic_snap::{MagneticSnapConfig, MagneticSnapEngine};
use oximedia_edit::timeline::{Timeline, TrackType};

fn base_timebase() -> Rational {
    Rational::new(1, 1000)
}

fn base_fps() -> Rational {
    Rational::new(30, 1)
}

/// Build a timeline with two clips on the same video track.
///   anchor: [0..100]   (will be a snap target)
///   mover:  [500..550] (will be moved)
fn timeline_two_clips() -> (Timeline, u64, u64) {
    let mut tl = Timeline::new(base_timebase(), base_fps());
    let vt = tl.add_track(TrackType::Video);
    let anchor = Clip::new(0, ClipType::Video, 0, 100);
    let mover = Clip::new(0, ClipType::Video, 500, 50);
    let anchor_id = tl.add_clip(vt, anchor).expect("add anchor");
    let mover_id = tl.add_clip(vt, mover).expect("add mover");
    (tl, anchor_id, mover_id)
}

#[test]
fn test_snap_to_clip_edge() {
    let (mut tl, _anchor_id, mover_id) = timeline_two_clips();

    // Enable magnetic snap with threshold=10.
    // Do NOT exclude the anchor — it IS the target we want to snap to.
    let cfg = MagneticSnapConfig {
        enabled: true,
        threshold: 10,
        snap_to_clips: true,
        snap_to_playhead: false,
        snap_to_markers: false,
        snap_to_grid: false,
        snap_to_in_out: false,
        ..Default::default()
    };
    tl.snap_engine = Some(MagneticSnapEngine::new(cfg));

    // Move mover to t=108 → within threshold of anchor end (t=100); snaps to 100.
    // [108..158] has no overlap with [0..100], and after snap [100..150] is still clean.
    tl.move_clip(mover_id, 108)
        .expect("move_clip should succeed");

    let pos = tl.get_clip(mover_id).expect("clip exists").timeline_start;
    assert_eq!(pos, 100, "clip should have snapped to anchor end at t=100");
}

#[test]
fn test_snap_disabled_by_default() {
    let (mut tl, _anchor_id, mover_id) = timeline_two_clips();

    // No snap engine — Timeline::new() leaves snap_engine = None.
    // Move to t=108 → should stay at 108, no snapping.
    tl.move_clip(mover_id, 108)
        .expect("move_clip should succeed");

    let pos = tl.get_clip(mover_id).expect("clip exists").timeline_start;
    assert_eq!(pos, 108, "no snap engine: clip stays exactly where placed");
}
