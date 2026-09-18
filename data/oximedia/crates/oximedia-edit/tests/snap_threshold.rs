//! Snap threshold enforcement tests.

use oximedia_core::Rational;
use oximedia_edit::clip::{Clip, ClipType};
use oximedia_edit::magnetic_snap::{MagneticSnapConfig, MagneticSnapEngine};
use oximedia_edit::timeline::{Timeline, TrackType};

fn make_timeline_with_snap(threshold: i64) -> (Timeline, u64) {
    let mut tl = Timeline::new(Rational::new(1, 1000), Rational::new(30, 1));

    // Clip at 0..200 — snap target edge at t=200.
    let vt = tl.add_track(TrackType::Video);
    let anchor = Clip::new(0, ClipType::Video, 0, 200);
    let _anchor_id = tl.add_clip(vt, anchor).expect("add anchor");

    // Moving clip at t=500.
    let mover = Clip::new(0, ClipType::Video, 500, 50);
    let mover_id = tl.add_clip(vt, mover).expect("add mover");

    let cfg = MagneticSnapConfig {
        enabled: true,
        threshold,
        snap_to_clips: true,
        snap_to_playhead: false,
        snap_to_markers: false,
        snap_to_grid: false,
        snap_to_in_out: false,
        ..Default::default()
    };
    tl.snap_engine = Some(MagneticSnapEngine::new(cfg));

    (tl, mover_id)
}

#[test]
fn test_snap_within_threshold_snaps() {
    // threshold = 5 → at distance 3 should snap.
    let (mut tl, mover_id) = make_timeline_with_snap(5);

    // t=203 is 3 units from the anchor edge at 200.
    tl.move_clip(mover_id, 203).expect("move should succeed");

    let pos = tl.get_clip(mover_id).expect("exists").timeline_start;
    assert_eq!(pos, 200, "should snap to 200 (distance 3 < threshold 5)");
}

#[test]
fn test_snap_outside_threshold_no_snap() {
    // threshold = 5 → at distance 7 should NOT snap.
    let (mut tl, mover_id) = make_timeline_with_snap(5);

    // t=207 is 7 units from anchor edge at 200.
    tl.move_clip(mover_id, 207).expect("move should succeed");

    let pos = tl.get_clip(mover_id).expect("exists").timeline_start;
    assert_eq!(pos, 207, "should NOT snap: distance 7 > threshold 5");
}
