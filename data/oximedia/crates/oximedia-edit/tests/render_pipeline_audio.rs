//! Integration tests for the audio rendering pipeline:
//! `TimelineRenderer::render_frame_at` with audio clips.

use std::sync::Arc;

use oximedia_core::Rational;
use oximedia_edit::{Clip, ClipType, RenderConfig, Timeline, TimelineRenderer, TrackType};

/// Build a timeline with one audio track and a TestPattern audio clip.
fn make_timeline_with_audio_clip() -> Arc<Timeline> {
    let timebase = Rational::new(1, 1000);
    let frame_rate = Rational::new(30, 1);
    let mut tl = Timeline::new(timebase, frame_rate);
    let track = tl.add_track(TrackType::Audio);

    let clip = Clip::new(1, ClipType::Audio, 0, 2000);
    tl.add_clip(track, clip).expect("add audio clip");
    Arc::new(tl)
}

#[tokio::test]
async fn test_render_audio_frame_has_audio() {
    let tl = make_timeline_with_audio_clip();
    let config = RenderConfig {
        render_video: false,
        render_audio: true,
        ..Default::default()
    };
    let mut renderer = TimelineRenderer::new(tl, config);
    let frame = renderer.render_frame_at(500).await.expect("render");
    assert!(frame.has_audio(), "frame must contain audio");
}

#[tokio::test]
async fn test_render_audio_frame_no_audio_outside_clip() {
    let tl = make_timeline_with_audio_clip();
    let config = RenderConfig {
        render_video: false,
        render_audio: true,
        ..Default::default()
    };
    let mut renderer = TimelineRenderer::new(tl, config);
    // Position 9999 ms is beyond the 2000 ms clip.
    let frame = renderer.render_frame_at(9999).await.expect("render");
    assert!(!frame.has_audio(), "no audio expected past clip end");
}

#[tokio::test]
async fn test_render_audio_disabled_skips_audio() {
    let tl = make_timeline_with_audio_clip();
    let config = RenderConfig {
        render_video: false,
        render_audio: false,
        ..Default::default()
    };
    let mut renderer = TimelineRenderer::new(tl, config);
    let frame = renderer.render_frame_at(500).await.expect("render");
    assert!(
        !frame.has_audio(),
        "render_audio=false must not produce audio"
    );
}

#[tokio::test]
async fn test_render_audio_frame_sample_rate_matches_config() {
    let tl = make_timeline_with_audio_clip();
    let config = RenderConfig {
        render_video: false,
        render_audio: true,
        sample_rate: 44_100,
        ..Default::default()
    };
    let mut renderer = TimelineRenderer::new(tl, config);
    let frame = renderer.render_frame_at(200).await.expect("render");
    let af = frame.audio.expect("must have audio");
    assert_eq!(af.sample_rate, 44_100, "sample_rate must match config");
}

#[tokio::test]
async fn test_render_audio_muted_clip_produces_silence() {
    let timebase = Rational::new(1, 1000);
    let frame_rate = Rational::new(30, 1);
    let mut tl = Timeline::new(timebase, frame_rate);
    let track = tl.add_track(TrackType::Audio);

    let mut clip = Clip::new(1, ClipType::Audio, 0, 2000);
    clip.muted = true;
    tl.add_clip(track, clip).expect("add muted clip");

    let tl = Arc::new(tl);
    let config = RenderConfig {
        render_video: false,
        render_audio: true,
        ..Default::default()
    };
    let mut renderer = TimelineRenderer::new(tl, config);
    let frame = renderer.render_frame_at(500).await.expect("render");
    // Muted clip → no audio output (returns None because only clip is muted).
    // The implementation skips muted clips, so with only one muted clip
    // the mix stays zero → the frame may or may not have audio, but if it
    // does, all samples must be zero.
    if let Some(af) = frame.audio {
        use oximedia_audio::AudioBuffer;
        if let AudioBuffer::Interleaved(bytes) = &af.samples {
            let all_zero = bytes
                .chunks_exact(4)
                .all(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]).abs() < 1e-9);
            assert!(all_zero, "muted clip must produce silence");
        }
    }
}
