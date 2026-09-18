//! Tests for motion effects (`oximedia_edl::motion`), the timeline-analysis
//! helpers (`oximedia_edl::edl_timeline`), and the EDL->timeline bridge
//! (`oximedia_edl::to_timeline`).

use oximedia_edl::edl_timeline::{TimelineAnalyzer, TimelineEvent};
use oximedia_edl::event::{EditType, EdlEvent, TrackType};
use oximedia_edl::motion::MotionEffect;
use oximedia_edl::timecode::{EdlFrameRate, EdlTimecode};
use oximedia_edl::to_timeline::convert_edl_to_timeline;
use oximedia_edl::{Edl, EdlFormat};

/// Test 10 — motion-effect constructors, predicates, effective speed, and the
/// `validate()` rejection rules.
#[test]
fn test_motion_effect_constructors_and_validation() {
    // reverse(): speed -1.0, reverse flag set, effective speed -1.0, valid.
    let rev = MotionEffect::reverse();
    assert!(
        (rev.speed - (-1.0)).abs() < f64::EPSILON,
        "reverse speed -1.0"
    );
    assert!(rev.is_reverse(), "reverse() sets the reverse flag");
    assert!(
        (rev.effective_speed() - (-1.0)).abs() < f64::EPSILON,
        "effective speed of reverse is -1.0"
    );
    rev.validate()
        .expect("a valid reverse effect should pass validation");

    // Negative speed WITHOUT the reverse flag is invalid.
    let bad = MotionEffect {
        speed: -2.0,
        reverse: false,
        freeze_frames: None,
        interpolation: oximedia_edl::motion::InterpolationMethod::Blend,
    };
    assert!(
        bad.validate().is_err(),
        "negative speed without reverse must fail validation"
    );

    // freeze(30): frozen, 30 frames recorded, speed 0.0, valid.
    let frz = MotionEffect::freeze(30);
    assert!(frz.is_freeze(), "freeze() yields a freeze effect");
    assert_eq!(frz.freeze_frames, Some(30), "freeze frame count is 30");
    assert!(frz.speed.abs() < f64::EPSILON, "freeze speed is 0.0");
    frz.validate()
        .expect("a valid freeze effect should pass validation");

    // A freeze mutated to a non-zero speed is contradictory -> invalid.
    let mut bad_freeze = MotionEffect::freeze(30);
    bad_freeze.speed = 1.0;
    assert!(
        bad_freeze.validate().is_err(),
        "freeze with non-zero speed must fail validation"
    );
}

/// Test 11 — the M2 comment round-trip used for motion effects in EDL exports.
#[test]
fn test_motion_m2_comment_roundtrip() {
    let parsed =
        MotionEffect::from_m2_comment("M2 A001 0.50 100").expect("valid M2 comment should parse");
    assert!(
        (parsed.speed - 0.5).abs() < f64::EPSILON,
        "M2 speed field parsed as 0.5; got {}",
        parsed.speed
    );

    let emitted = MotionEffect::new(0.5).to_m2_comment("A001", 100);
    assert_eq!(
        emitted, "M2 A001 0.50 100",
        "to_m2_comment formats speed to two decimals"
    );

    // Too few fields -> Err (no reel/speed/frame triplet after "M2").
    assert!(
        MotionEffect::from_m2_comment("M2 A001").is_err(),
        "an incomplete M2 comment must fail to parse"
    );
}

/// Test 12a — EDL->timeline bridge: build a two-cut EDL at 25 fps and verify
/// the converted timeline has exactly two clips whose record-in/out frames
/// match the source events.
#[test]
fn test_convert_edl_to_timeline_clip_frames() {
    let fps = EdlFrameRate::Fps25;
    let tc = |h: u8, m: u8, s: u8, f: u8| -> EdlTimecode {
        EdlTimecode::new(h, m, s, f, fps).expect("valid timecode")
    };

    let mut edl = Edl::cmx3600();
    edl.set_frame_rate(fps);

    // Event 1: record 01:00:00:00 -> 01:00:05:00
    let r1_in = tc(1, 0, 0, 0);
    let r1_out = tc(1, 0, 5, 0);
    let ev1 = EdlEvent::new(
        1,
        "A001".to_string(),
        TrackType::Video,
        EditType::Cut,
        r1_in,
        r1_out,
        r1_in,
        r1_out,
    );
    edl.add_event(ev1).expect("event 1 should be valid");

    // Event 2: record 01:00:05:00 -> 01:00:10:00
    let r2_in = tc(1, 0, 5, 0);
    let r2_out = tc(1, 0, 10, 0);
    let ev2 = EdlEvent::new(
        2,
        "A002".to_string(),
        TrackType::Video,
        EditType::Cut,
        r2_in,
        r2_out,
        r2_in,
        r2_out,
    );
    edl.add_event(ev2).expect("event 2 should be valid");

    let timeline = convert_edl_to_timeline(&edl).expect("conversion should succeed");

    assert_eq!(
        timeline.total_clips(),
        2,
        "two cuts should yield two timeline clips"
    );

    // Both video cuts land on the default "V1" track.
    let track = timeline
        .get_track("V1")
        .expect("video clips should be on the V1 track");

    let clip1 = track
        .clips
        .iter()
        .find(|c| c.id == 1)
        .expect("clip for event 1 should exist");
    assert_eq!(
        clip1.timeline_in_frames,
        r1_in.to_frames(),
        "clip 1 record-in frames"
    );
    assert_eq!(
        clip1.timeline_out_frames,
        r1_out.to_frames(),
        "clip 1 record-out frames"
    );

    let clip2 = track
        .clips
        .iter()
        .find(|c| c.id == 2)
        .expect("clip for event 2 should exist");
    assert_eq!(
        clip2.timeline_in_frames,
        r2_in.to_frames(),
        "clip 2 record-in frames"
    );
    assert_eq!(
        clip2.timeline_out_frames,
        r2_out.to_frames(),
        "clip 2 record-out frames"
    );

    // Sanity: the EDL really is CMX 3600.
    assert_eq!(edl.format, EdlFormat::Cmx3600);
}

/// Test 12b — `TimelineAnalyzer` overlap and coverage accounting on two
/// overlapping events: (1, 0, 100) and (2, 80, 180).
#[test]
fn test_timeline_analyzer_overlap_and_coverage() {
    let mut analyzer = TimelineAnalyzer::new();
    analyzer.add_event(TimelineEvent::new(1, 0, 100));
    analyzer.add_event(TimelineEvent::new(2, 80, 180));

    let overlaps = analyzer.find_overlaps();
    assert_eq!(overlaps.len(), 1, "exactly one overlap between the events");

    // The merged coverage spans [0, 180) = 180 frames despite the [80, 100)
    // overlap being counted once.
    assert_eq!(
        analyzer.total_coverage_frames(),
        180,
        "overlapping ranges merge to a single 180-frame span"
    );
}
