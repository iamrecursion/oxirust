//! Integration tests for unsupported / missing source handling.
//!
//! When a clip has no source file, or points to an unsupported format, the
//! render pipeline must gracefully produce a blank (TestPattern or silence)
//! output rather than panicking.

use std::sync::Arc;

use oximedia_core::Rational;
use oximedia_edit::render_source::RenderSource;
use oximedia_edit::{Clip, ClipType, RenderConfig, Timeline, TimelineRenderer, TrackType};

// ─── No source file (None) → TestPattern ─────────────────────────────────────

#[test]
fn test_no_source_path_video_is_not_blank() {
    // clip.source = None → TestPattern → SMPTE bars.
    let src = RenderSource::TestPattern;
    let frame = src.sample_video(0, 16, 8);
    assert_eq!(frame.len(), 16 * 8 * 4);
    let all_zero = frame.iter().all(|&b| b == 0);
    assert!(!all_zero, "TestPattern must not be fully black");
}

// ─── Unsupported extension ────────────────────────────────────────────────────

#[test]
fn test_unsupported_extension_path_resolves() {
    let tmp = std::env::temp_dir().join("oximedia_pipeline_unsupported_test.xyz");
    std::fs::write(&tmp, b"not media data").ok();

    let result = RenderSource::from_path(&tmp);
    assert!(
        result.is_ok(),
        "from_path must not error on unknown extension"
    );

    let src = result.expect("RenderSource");
    let video = src.sample_video(0, 8, 4);
    let audio = src.sample_audio(0, 48, 2, 48_000);

    assert_eq!(video.len(), 8 * 4 * 4, "video buffer has correct size");
    assert!(video.iter().all(|&b| b == 0), "unsupported video is black");
    assert!(
        audio.iter().all(|&s| s == 0.0),
        "unsupported audio is silence"
    );

    std::fs::remove_file(&tmp).ok();
}

#[test]
fn test_missing_file_falls_back_gracefully() {
    let nonexistent = std::env::temp_dir().join("nonexistent_oximedia_clip.png");
    // Make sure it does not exist.
    std::fs::remove_file(&nonexistent).ok();

    let result = RenderSource::from_path(&nonexistent);
    // File read failure → EditError, not a panic.
    assert!(
        result.is_err(),
        "missing file must return an Err (not panic)"
    );
}

// ─── Full pipeline: clip with unsupported source ──────────────────────────────

#[tokio::test]
async fn test_render_with_unsupported_clip_source_does_not_panic() {
    let tmp = std::env::temp_dir().join("oximedia_pipeline_render_unsupported.xyz");
    std::fs::write(&tmp, b"not media data").ok();

    let timebase = Rational::new(1, 1000);
    let frame_rate = Rational::new(30, 1);
    let mut tl = Timeline::new(timebase, frame_rate);
    let track = tl.add_track(TrackType::Video);

    let mut clip = Clip::new(1, ClipType::Video, 0, 2000);
    clip.source = Some(tmp.clone());
    tl.add_clip(track, clip).expect("add clip");

    let tl = Arc::new(tl);
    let config = RenderConfig {
        width: 8,
        height: 4,
        render_video: true,
        render_audio: false,
        ..Default::default()
    };
    let mut renderer = TimelineRenderer::new(tl, config);

    // Must not panic — unsupported source → black frame.
    let frame = renderer
        .render_frame_at(500)
        .await
        .expect("render must succeed");
    // The clip is present, so we expect a video frame.
    assert!(
        frame.has_video(),
        "must have video even for unsupported source"
    );

    std::fs::remove_file(&tmp).ok();
}
