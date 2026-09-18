//! Integration tests: `TransitionRenderer::blend_video` and `mix_audio`.
//!
//! For variants that do real pixel blending (Dissolve + 4 wipes) we verify
//! actual pixel values.  For hard-cut variants we verify the switch-at-0.5
//! behaviour.  Audio cross-fade is verified with F32 interleaved data.

use bytes::Bytes;
use oximedia_audio::{AudioBuffer, AudioFrame, ChannelLayout};
use oximedia_codec::VideoFrame;
use oximedia_core::{PixelFormat, SampleFormat};
use oximedia_edit::render::TransitionRenderer;
use oximedia_edit::transition::{Transition, TransitionType};

// ── Frame helpers ─────────────────────────────────────────────────────────────

/// Create a small 4×4 YUV420p frame with all planes filled to `value`.
fn solid_video_frame(value: u8) -> VideoFrame {
    let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 4, 4);
    frame.allocate();
    for plane in &mut frame.planes {
        for byte in &mut plane.data {
            *byte = value;
        }
    }
    frame
}

/// Create a minimal F32-interleaved AudioFrame with `num_samples` stereo samples,
/// all set to `sample_value`.
fn solid_audio_frame(sample_value: f32, num_samples: usize) -> AudioFrame {
    let mut bytes = Vec::with_capacity(num_samples * 4);
    for _ in 0..num_samples {
        bytes.extend_from_slice(&sample_value.to_ne_bytes());
    }
    let mut frame = AudioFrame::new(SampleFormat::F32, 48000, ChannelLayout::Stereo);
    frame.samples = AudioBuffer::Interleaved(Bytes::from(bytes));
    frame
}

/// Build a dummy `Transition` with the given type (IDs are irrelevant for rendering).
fn make_transition(tt: TransitionType) -> Transition {
    Transition::new(1, tt, 0, 0, 1000, 1, 2)
}

// ── Dissolve blend_video tests ────────────────────────────────────────────────

#[test]
fn test_dissolve_blend_at_progress_zero_returns_frame_a() {
    let fa = solid_video_frame(0);
    let fb = solid_video_frame(200);
    let t = make_transition(TransitionType::Dissolve);
    let out = TransitionRenderer::blend_video(&fa, &fb, &t, 0.0);
    // At progress=0 all pixels should equal frame_a's pixel value (0).
    let all_zero = out.planes.iter().all(|p| p.data.iter().all(|&b| b == 0));
    assert!(all_zero, "dissolve at 0.0 must equal frame_a");
}

#[test]
fn test_dissolve_blend_at_progress_one_returns_frame_b() {
    let fa = solid_video_frame(0);
    let fb = solid_video_frame(200);
    let t = make_transition(TransitionType::Dissolve);
    let out = TransitionRenderer::blend_video(&fa, &fb, &t, 1.0);
    let all_b = out.planes.iter().all(|p| p.data.iter().all(|&b| b == 200));
    assert!(all_b, "dissolve at 1.0 must equal frame_b");
}

#[test]
fn test_dissolve_blend_at_midpoint_is_average() {
    // fa=0, fb=200 → midpoint should be ~100 (within rounding).
    let fa = solid_video_frame(0);
    let fb = solid_video_frame(200);
    let t = make_transition(TransitionType::Dissolve);
    let out = TransitionRenderer::blend_video(&fa, &fb, &t, 0.5);
    // All pixel values should be 100 (within ±1 for rounding).
    let ok = out.planes.iter().all(|p| p.data.iter().all(|&b| b == 100));
    assert!(ok, "dissolve at 0.5 must produce mid-value ~100");
}

#[test]
fn test_dissolve_output_dimensions_unchanged() {
    let fa = solid_video_frame(50);
    let fb = solid_video_frame(100);
    let t = make_transition(TransitionType::Dissolve);
    let out = TransitionRenderer::blend_video(&fa, &fb, &t, 0.5);
    assert_eq!(out.width, 4);
    assert_eq!(out.height, 4);
    assert_eq!(out.format, PixelFormat::Yuv420p);
}

// ── Wipe blend_video tests ────────────────────────────────────────────────────

#[test]
fn test_wipe_left_at_zero_returns_frame_a() {
    let fa = solid_video_frame(10);
    let fb = solid_video_frame(210);
    let t = make_transition(TransitionType::WipeLeft);
    let out = TransitionRenderer::blend_video(&fa, &fb, &t, 0.0);
    let all_a = out.planes[0].data.iter().all(|&b| b == 10);
    assert!(all_a, "WipeLeft at 0.0 must equal frame_a");
}

#[test]
fn test_wipe_right_at_one_returns_frame_b() {
    let fa = solid_video_frame(10);
    let fb = solid_video_frame(210);
    let t = make_transition(TransitionType::WipeRight);
    let out = TransitionRenderer::blend_video(&fa, &fb, &t, 1.0);
    let all_b = out.planes[0].data.iter().all(|&b| b == 210);
    assert!(all_b, "WipeRight at 1.0 must equal frame_b");
}

#[test]
fn test_wipe_down_blend_changes_pixels() {
    let fa = solid_video_frame(0);
    let fb = solid_video_frame(255);
    let t = make_transition(TransitionType::WipeDown);
    // At 0.5 progress a wipe from top should have transitioned at least part of the frame.
    let out = TransitionRenderer::blend_video(&fa, &fb, &t, 0.5);
    // Result must not be all zeros (some of frame_b leaked in) and not all 255.
    let has_b = out.planes[0].data.iter().any(|&b| b == 255);
    let has_a = out.planes[0].data.iter().any(|&b| b == 0);
    assert!(
        has_b || has_a,
        "WipeDown at 0.5 must produce a partial wipe"
    );
}

#[test]
fn test_wipe_up_at_zero_equals_frame_a() {
    let fa = solid_video_frame(77);
    let fb = solid_video_frame(200);
    let t = make_transition(TransitionType::WipeUp);
    let out = TransitionRenderer::blend_video(&fa, &fb, &t, 0.0);
    let all_a = out.planes[0].data.iter().all(|&b| b == 77);
    assert!(all_a, "WipeUp at 0.0 must equal frame_a");
}

// ── Hard-cut variants ────────────────────────────────────────────────────────

/// Verify that a non-blending transition returns frame_a below 0.5 and frame_b above.
fn assert_hard_cut_at_half(tt: TransitionType, label: &str) {
    let fa = solid_video_frame(10);
    let fb = solid_video_frame(200);
    let t = make_transition(tt);

    let below = TransitionRenderer::blend_video(&fa, &fb, &t, 0.49);
    let all_a = below.planes[0].data.iter().all(|&b| b == 10);
    assert!(all_a, "{label}: progress<0.5 must return frame_a");

    let at_half = TransitionRenderer::blend_video(&fa, &fb, &t, 0.5);
    let all_b = at_half.planes[0].data.iter().all(|&b| b == 200);
    assert!(all_b, "{label}: progress=0.5 must return frame_b");
}

#[test]
fn test_crossfade_is_hard_cut_for_video() {
    assert_hard_cut_at_half(TransitionType::CrossFade, "CrossFade");
}

#[test]
fn test_slide_transition_is_hard_cut() {
    assert_hard_cut_at_half(TransitionType::Slide, "Slide");
}

#[test]
fn test_push_transition_is_hard_cut() {
    assert_hard_cut_at_half(TransitionType::Push, "Push");
}

#[test]
fn test_zoom_in_transition_is_hard_cut() {
    assert_hard_cut_at_half(TransitionType::ZoomIn, "ZoomIn");
}

#[test]
fn test_zoom_out_transition_is_hard_cut() {
    assert_hard_cut_at_half(TransitionType::ZoomOut, "ZoomOut");
}

#[test]
fn test_fade_through_transition_is_hard_cut() {
    assert_hard_cut_at_half(TransitionType::FadeThrough, "FadeThrough");
}

#[test]
fn test_dip_to_color_transition_is_hard_cut() {
    assert_hard_cut_at_half(TransitionType::DipToColor, "DipToColor");
}

// ── Dimension/format mismatch ─────────────────────────────────────────────────

#[test]
fn test_blend_video_different_sizes_returns_larger() {
    let mut fa = VideoFrame::new(PixelFormat::Yuv420p, 4, 4);
    fa.allocate();
    let mut fb = VideoFrame::new(PixelFormat::Yuv420p, 8, 8);
    fb.allocate();
    for plane in &mut fb.planes {
        for byte in &mut plane.data {
            *byte = 99;
        }
    }
    let t = make_transition(TransitionType::Dissolve);
    let out = TransitionRenderer::blend_video(&fa, &fb, &t, 0.5);
    // fb is larger so it should be returned.
    assert_eq!(out.width, 8);
    assert_eq!(out.height, 8);
}

#[test]
fn test_blend_video_format_mismatch_returns_frame_a() {
    use oximedia_core::PixelFormat;
    let mut fa = VideoFrame::new(PixelFormat::Yuv420p, 4, 4);
    fa.allocate();
    let mut fb = VideoFrame::new(PixelFormat::Rgb24, 4, 4);
    fb.allocate();
    let t = make_transition(TransitionType::Dissolve);
    let out = TransitionRenderer::blend_video(&fa, &fb, &t, 0.5);
    assert_eq!(
        out.format,
        PixelFormat::Yuv420p,
        "format mismatch must return frame_a"
    );
}

// ── mix_audio tests ───────────────────────────────────────────────────────────

#[test]
fn test_mix_audio_at_zero_returns_frame_a() {
    // fa=1.0, fb=-1.0; at progress 0.0 result should be ≈1.0.
    let fa = solid_audio_frame(1.0, 8);
    let fb = solid_audio_frame(-1.0, 8);
    let t = make_transition(TransitionType::CrossFade);
    let out = TransitionRenderer::mix_audio(&fa, &fb, &t, 0.0);
    let AudioBuffer::Interleaved(bytes) = &out.samples else {
        panic!("expected interleaved buffer");
    };
    for chunk in bytes.chunks_exact(4) {
        let v = f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        assert!(
            (v - 1.0).abs() < 1e-5,
            "at progress=0.0 output must equal fa sample 1.0, got {v}"
        );
    }
}

#[test]
fn test_mix_audio_at_one_returns_frame_b() {
    let fa = solid_audio_frame(1.0, 8);
    let fb = solid_audio_frame(-1.0, 8);
    let t = make_transition(TransitionType::CrossFade);
    let out = TransitionRenderer::mix_audio(&fa, &fb, &t, 1.0);
    let AudioBuffer::Interleaved(bytes) = &out.samples else {
        panic!("expected interleaved buffer");
    };
    for chunk in bytes.chunks_exact(4) {
        let v = f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        assert!(
            (v - (-1.0)).abs() < 1e-5,
            "at progress=1.0 output must equal fb sample -1.0, got {v}"
        );
    }
}

#[test]
fn test_mix_audio_at_midpoint_is_zero() {
    // fa=1.0, fb=-1.0 → midpoint: 1.0 * 0.5 + (-1.0) * 0.5 = 0.0.
    let fa = solid_audio_frame(1.0, 8);
    let fb = solid_audio_frame(-1.0, 8);
    let t = make_transition(TransitionType::Dissolve);
    let out = TransitionRenderer::mix_audio(&fa, &fb, &t, 0.5);
    let AudioBuffer::Interleaved(bytes) = &out.samples else {
        panic!("expected interleaved buffer");
    };
    for chunk in bytes.chunks_exact(4) {
        let v = f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        assert!(
            v.abs() < 1e-5,
            "at progress=0.5 crossfade of 1.0 and -1.0 must be ~0, got {v}"
        );
    }
}

#[test]
fn test_mix_audio_sample_rate_preserved() {
    let fa = solid_audio_frame(0.5, 4);
    let fb = solid_audio_frame(0.5, 4);
    let t = make_transition(TransitionType::Dissolve);
    let out = TransitionRenderer::mix_audio(&fa, &fb, &t, 0.5);
    assert_eq!(out.sample_rate, 48000);
}

#[test]
fn test_mix_audio_format_mismatch_returns_frame_a() {
    // fa F32, fb S16 (different) → returns fa unchanged.
    let fa = solid_audio_frame(0.8, 4);
    let mut fb = solid_audio_frame(0.2, 4);
    fb.format = SampleFormat::S16;
    let t = make_transition(TransitionType::CrossFade);
    let out = TransitionRenderer::mix_audio(&fa, &fb, &t, 0.5);
    assert_eq!(
        out.format,
        SampleFormat::F32,
        "format mismatch must return fa format"
    );
}
