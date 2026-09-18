//! Tests for TrackTypeMismatch enforcement in add_clip.

use oximedia_core::Rational;
use oximedia_edit::clip::{Clip, ClipType};
use oximedia_edit::error::EditError;
use oximedia_edit::timeline::{Timeline, TrackType};

fn make_tl() -> Timeline {
    Timeline::new(Rational::new(1, 1000), Rational::new(30, 1))
}

#[test]
fn test_add_audio_clip_to_video_track_fails() {
    let mut tl = make_tl();
    let vt = tl.add_track(TrackType::Video);
    let audio_clip = Clip::new(0, ClipType::Audio, 0, 100);

    let result = tl.add_clip(vt, audio_clip);
    assert!(
        matches!(
            result,
            Err(EditError::TrackTypeMismatch {
                expected: ClipType::Video,
                got: ClipType::Audio
            })
        ),
        "expected TrackTypeMismatch, got: {:?}",
        result
    );
}

#[test]
fn test_add_video_clip_to_audio_track_fails() {
    let mut tl = make_tl();
    let at = tl.add_track(TrackType::Audio);
    let video_clip = Clip::new(0, ClipType::Video, 0, 100);

    let result = tl.add_clip(at, video_clip);
    assert!(
        matches!(
            result,
            Err(EditError::TrackTypeMismatch {
                expected: ClipType::Audio,
                got: ClipType::Video
            })
        ),
        "expected TrackTypeMismatch, got: {:?}",
        result
    );
}

#[test]
fn test_add_subtitle_clip_to_video_track_fails() {
    let mut tl = make_tl();
    let vt = tl.add_track(TrackType::Video);
    let sub_clip = Clip::new(0, ClipType::Subtitle, 0, 100);

    let result = tl.add_clip(vt, sub_clip);
    assert!(
        matches!(
            result,
            Err(EditError::TrackTypeMismatch {
                expected: ClipType::Video,
                got: ClipType::Subtitle
            })
        ),
        "expected TrackTypeMismatch, got: {:?}",
        result
    );
}

#[test]
fn test_correct_clip_types_accepted() {
    let mut tl = make_tl();
    let vt = tl.add_track(TrackType::Video);
    let at = tl.add_track(TrackType::Audio);
    let st = tl.add_track(TrackType::Subtitle);

    tl.add_clip(vt, Clip::new(0, ClipType::Video, 0, 100))
        .expect("video clip on video track is valid");
    tl.add_clip(at, Clip::new(0, ClipType::Audio, 0, 100))
        .expect("audio clip on audio track is valid");
    tl.add_clip(st, Clip::new(0, ClipType::Subtitle, 0, 100))
        .expect("subtitle clip on subtitle track is valid");
}
