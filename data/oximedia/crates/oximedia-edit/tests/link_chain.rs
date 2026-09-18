//! Tests for BFS cycle detection: A↔B mutual link must not infinite-loop.

use oximedia_core::Rational;
use oximedia_edit::clip::{Clip, ClipType};
use oximedia_edit::group::LinkType;
use oximedia_edit::timeline::{Timeline, TrackType};

#[test]
fn test_link_chain_no_infinite_loop() {
    let mut tl = Timeline::new(Rational::new(1, 1000), Rational::new(30, 1));

    let vt = tl.add_track(TrackType::Video);
    let at = tl.add_track(TrackType::Audio);

    let vid = Clip::new(0, ClipType::Video, 0, 100);
    let aud = Clip::new(0, ClipType::Audio, 0, 100);

    let vid_id = tl.add_clip(vt, vid).expect("add video");
    let aud_id = tl.add_clip(at, aud).expect("add audio");

    // Create a cycle: A↔B (both directions).
    tl.links.add_link(vid_id, aud_id, LinkType::VideoAudio);
    // Adding the reciprocal — BFS must terminate regardless.
    tl.links.add_link(aud_id, vid_id, LinkType::Synchronized);

    // Move should complete without hanging.
    tl.move_clip(vid_id, 200)
        .expect("move should not infinite-loop");

    let vid_start = tl.get_clip(vid_id).expect("exists").timeline_start;
    let aud_start = tl.get_clip(aud_id).expect("exists").timeline_start;

    assert_eq!(vid_start, 200, "video should be at t=200");
    assert_eq!(aud_start, 200, "audio should follow to t=200");
}

#[test]
fn test_link_chain_three_clips() {
    // A linked to B, B linked to C — all should move together.
    let mut tl = Timeline::new(Rational::new(1, 1000), Rational::new(30, 1));

    let vt = tl.add_track(TrackType::Video);
    let at1 = tl.add_track(TrackType::Audio);
    let at2 = tl.add_track(TrackType::Audio);

    let c1 = Clip::new(0, ClipType::Video, 0, 100);
    let c2 = Clip::new(0, ClipType::Audio, 0, 100);
    let c3 = Clip::new(0, ClipType::Audio, 0, 100);

    let id1 = tl.add_clip(vt, c1).expect("add c1");
    let id2 = tl.add_clip(at1, c2).expect("add c2");
    let id3 = tl.add_clip(at2, c3).expect("add c3");

    tl.links.add_link(id1, id2, LinkType::Synchronized);
    tl.links.add_link(id2, id3, LinkType::Synchronized);

    tl.move_clip(id1, 300).expect("move chain");

    assert_eq!(tl.get_clip(id1).unwrap().timeline_start, 300);
    assert_eq!(tl.get_clip(id2).unwrap().timeline_start, 300);
    assert_eq!(tl.get_clip(id3).unwrap().timeline_start, 300);
}
