//! Integration tests for the video rendering pipeline:
//! `TimelineRenderer::render_frame_at` with video clips.

use std::sync::Arc;

use oximedia_core::Rational;
use oximedia_edit::{Clip, ClipType, RenderConfig, Timeline, TimelineRenderer, TrackType};

/// Build a minimal timeline with one video track and one TestPattern clip.
fn make_timeline_with_video_clip() -> Arc<Timeline> {
    let timebase = Rational::new(1, 1000);
    let frame_rate = Rational::new(30, 1);
    let mut tl = Timeline::new(timebase, frame_rate);
    let track = tl.add_track(TrackType::Video);

    // Clip from t=0 to t=2000 ms (no source file → TestPattern).
    let clip = Clip::new(1, ClipType::Video, 0, 2000);
    tl.add_clip(track, clip).expect("add_clip should succeed");
    Arc::new(tl)
}

#[tokio::test]
async fn test_render_video_frame_has_video() {
    let tl = make_timeline_with_video_clip();
    let config = RenderConfig {
        width: 16,
        height: 8,
        render_video: true,
        render_audio: false,
        ..Default::default()
    };
    let mut renderer = TimelineRenderer::new(tl, config);
    let frame = renderer
        .render_frame_at(500)
        .await
        .expect("render at 500ms");
    assert!(frame.has_video(), "frame at 500ms must contain video");
}

#[tokio::test]
async fn test_render_video_frame_no_video_outside_clip() {
    let tl = make_timeline_with_video_clip();
    let config = RenderConfig {
        width: 16,
        height: 8,
        render_video: true,
        render_audio: false,
        ..Default::default()
    };
    let mut renderer = TimelineRenderer::new(tl, config);
    // Position 5000 ms is beyond the 2000 ms clip.
    let frame = renderer
        .render_frame_at(5000)
        .await
        .expect("render at 5000ms");
    assert!(!frame.has_video(), "no video expected outside all clips");
}

#[tokio::test]
async fn test_render_video_frame_dimensions_match_config() {
    let tl = make_timeline_with_video_clip();
    let config = RenderConfig {
        width: 32,
        height: 16,
        render_video: true,
        render_audio: false,
        ..Default::default()
    };
    let mut renderer = TimelineRenderer::new(tl, config);
    let frame = renderer.render_frame_at(1000).await.expect("render");
    let vf = frame.video.expect("must have video frame");
    assert_eq!(vf.width, 32, "output width");
    assert_eq!(vf.height, 16, "output height");
}

#[tokio::test]
async fn test_render_video_disabled_skips_rendering() {
    let tl = make_timeline_with_video_clip();
    let config = RenderConfig {
        render_video: false,
        render_audio: false,
        ..Default::default()
    };
    let mut renderer = TimelineRenderer::new(tl, config);
    let frame = renderer.render_frame_at(500).await.expect("render");
    assert!(
        !frame.has_video(),
        "render_video=false must not produce video"
    );
}

#[tokio::test]
async fn test_render_video_frame_cached_on_second_call() {
    let tl = make_timeline_with_video_clip();
    let config = RenderConfig {
        width: 8,
        height: 4,
        render_video: true,
        render_audio: false,
        ..Default::default()
    };
    let mut renderer = TimelineRenderer::new(tl, config);
    let f1 = renderer.render_frame_at(100).await.expect("first render");
    let f2 = renderer
        .render_frame_at(100)
        .await
        .expect("second render (cached)");
    // Both frames should represent the same position.
    assert_eq!(f1.position, f2.position);
}
