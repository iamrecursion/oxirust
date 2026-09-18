//! Integration tests for video transition blending via `TransitionRenderer`.

use oximedia_codec::VideoFrame;
use oximedia_core::PixelFormat;
use oximedia_edit::render::TransitionRenderer;
use oximedia_edit::transition::{Transition, TransitionType};

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn make_video_frame(w: u32, h: u32, value: u8) -> VideoFrame {
    let mut f = VideoFrame::new(PixelFormat::Yuv420p, w, h);
    f.allocate();
    for plane in &mut f.planes {
        for b in &mut plane.data {
            *b = value;
        }
    }
    f
}

fn make_transition(tt: TransitionType) -> Transition {
    Transition::new(0, tt, 0, 0, 1000, 0, 1)
}

// ─── Dissolve ────────────────────────────────────────────────────────────────

#[test]
fn test_dissolve_at_zero_returns_frame_a() {
    let fa = make_video_frame(8, 4, 10);
    let fb = make_video_frame(8, 4, 200);
    let t = make_transition(TransitionType::Dissolve);

    let out = TransitionRenderer::blend_video(&fa, &fb, &t, 0.0);

    for plane in &out.planes {
        for &b in &plane.data {
            assert_eq!(b, 10, "dissolve at 0 must equal frame_a");
        }
    }
}

#[test]
fn test_dissolve_at_one_returns_frame_b() {
    let fa = make_video_frame(8, 4, 10);
    let fb = make_video_frame(8, 4, 200);
    let t = make_transition(TransitionType::Dissolve);

    let out = TransitionRenderer::blend_video(&fa, &fb, &t, 1.0);

    for plane in &out.planes {
        for &b in &plane.data {
            assert_eq!(b, 200, "dissolve at 1 must equal frame_b");
        }
    }
}

#[test]
fn test_dissolve_at_midpoint_is_between() {
    let fa = make_video_frame(8, 4, 100);
    let fb = make_video_frame(8, 4, 200);
    let t = make_transition(TransitionType::Dissolve);

    let out = TransitionRenderer::blend_video(&fa, &fb, &t, 0.5);

    for plane in &out.planes {
        for &b in &plane.data {
            let diff = (i32::from(b) - 150).abs();
            assert!(diff <= 1, "dissolve at 0.5 expected ~150, got {b}");
        }
    }
}

// ─── WipeLeft ────────────────────────────────────────────────────────────────

#[test]
fn test_wipe_left_at_zero_is_frame_a() {
    let fa = make_video_frame(8, 4, 50);
    let fb = make_video_frame(8, 4, 200);
    let t = make_transition(TransitionType::WipeLeft);

    let out = TransitionRenderer::blend_video(&fa, &fb, &t, 0.0);

    // With progress=0.0 the boundary is at 0 columns → all frame_a.
    for plane in &out.planes {
        for &b in &plane.data {
            assert_eq!(b, 50, "wipe at 0 must equal frame_a");
        }
    }
}

// ─── Dimension / format mismatch ─────────────────────────────────────────────

#[test]
fn test_blend_video_dimension_mismatch_no_panic() {
    let fa = make_video_frame(8, 4, 100);
    let fb = make_video_frame(16, 8, 200);
    let t = make_transition(TransitionType::Dissolve);

    // Must not panic; returns the larger frame.
    let out = TransitionRenderer::blend_video(&fa, &fb, &t, 0.5);
    assert_eq!(out.width, 16);
    assert_eq!(out.height, 8);
}

#[test]
fn test_blend_video_format_mismatch_returns_frame_a() {
    let fa = make_video_frame(8, 4, 100);
    let mut fb = make_video_frame(8, 4, 200);
    fb.format = PixelFormat::Rgb24; // intentional mismatch
    let t = make_transition(TransitionType::Dissolve);

    let out = TransitionRenderer::blend_video(&fa, &fb, &t, 0.5);
    // Must return frame_a (same format as fa).
    assert_eq!(out.format, PixelFormat::Yuv420p);
}

// ─── CrossFade (audio only) falls back to mid-point cut ──────────────────────

#[test]
fn test_crossfade_type_at_half_returns_frame_b() {
    let fa = make_video_frame(8, 4, 50);
    let fb = make_video_frame(8, 4, 200);
    let t = make_transition(TransitionType::CrossFade);

    // CrossFade is audio-only; for video it behaves as a cut at 0.5.
    let out = TransitionRenderer::blend_video(&fa, &fb, &t, 0.6);
    for plane in &out.planes {
        for &b in &plane.data {
            assert_eq!(b, 200, "CrossFade at >0.5 must return frame_b");
        }
    }
}
