//! Tests for V/A link enforcement: linked clips move together, and deleting one
//! clip does not cascade-delete the linked clip.

use oximedia_core::Rational;
use oximedia_edit::clip::{Clip, ClipType};
use oximedia_edit::timeline::{Timeline, TrackType};

fn make_linked_timeline() -> (Timeline, u64, u64) {
    let mut tl = Timeline::new(Rational::new(1, 1000), Rational::new(30, 1));

    let vt = tl.add_track(TrackType::Video);
    let at = tl.add_track(TrackType::Audio);

    // Both clips start at t=0, duration 200.
    let video_clip = Clip::new(0, ClipType::Video, 0, 200);
    let audio_clip = Clip::new(0, ClipType::Audio, 0, 200);

    let vid_id = tl.add_clip(vt, video_clip).expect("add video clip");
    let aud_id = tl.add_clip(at, audio_clip).expect("add audio clip");

    // Link them as a V/A pair.
    tl.links.link_video_audio(vid_id, aud_id);

    (tl, vid_id, aud_id)
}

#[test]
fn test_linked_clips_move_together() {
    let (mut tl, vid_id, aud_id) = make_linked_timeline();

    // Move the video clip to t=50; audio should also shift to t=50.
    tl.move_clip(vid_id, 50).expect("move should succeed");

    let vid_start = tl.get_clip(vid_id).expect("video exists").timeline_start;
    let aud_start = tl.get_clip(aud_id).expect("audio exists").timeline_start;

    assert_eq!(vid_start, 50, "video clip should be at t=50");
    assert_eq!(
        aud_start, 50,
        "audio clip should move to t=50 (link cascade)"
    );
}

#[test]
fn test_linked_clips_preserve_relative_offset() {
    let mut tl = Timeline::new(Rational::new(1, 1000), Rational::new(30, 1));

    let vt = tl.add_track(TrackType::Video);
    let at = tl.add_track(TrackType::Audio);

    // Video at t=0, audio at t=10 (10-unit offset).
    let video_clip = Clip::new(0, ClipType::Video, 0, 200);
    let audio_clip = Clip::new(0, ClipType::Audio, 10, 200);

    let vid_id = tl.add_clip(vt, video_clip).expect("add video");
    let aud_id = tl.add_clip(at, audio_clip).expect("add audio");
    tl.links.link_video_audio(vid_id, aud_id);

    // Move video clip by +100.
    tl.move_clip(vid_id, 100).expect("move should succeed");

    let vid_start = tl.get_clip(vid_id).expect("video exists").timeline_start;
    let aud_start = tl.get_clip(aud_id).expect("audio exists").timeline_start;

    assert_eq!(vid_start, 100, "video at 100");
    // Original delta was 10, same delta applied to audio: 10 + 100 = 110.
    assert_eq!(
        aud_start, 110,
        "audio preserves +10 relative offset → t=110"
    );
}

#[test]
fn test_delete_clip_preserves_linked_clip() {
    let (mut tl, vid_id, aud_id) = make_linked_timeline();

    // Delete the video clip.
    tl.remove_clip(vid_id).expect("remove video clip");

    // Audio clip must still exist.
    assert!(
        tl.get_clip(aud_id).is_some(),
        "audio clip should survive after video clip deletion"
    );

    // The link entry for the deleted clip should be gone.
    assert!(
        tl.links.get_clip_links(vid_id).is_empty(),
        "link entries for deleted clip should be removed"
    );
}
