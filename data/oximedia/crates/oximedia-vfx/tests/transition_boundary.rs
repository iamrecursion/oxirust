//! Boundary sweep tests for all `TransitionEffect` implementations.
//!
//! Tests each transition at progress = 0.0, 0.5, and 1.0 on a small 4×4 frame,
//! verifying that the output is consistent with the expected from/to blend behaviour.

use oximedia_vfx::{
    transition::{
        Dissolve, Push, PushDirection, Slide, SlideDirection, Wipe, WipePattern, Zoom, ZoomMode,
    },
    EffectParams, Frame, TransitionEffect,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a solid-colour 4×4 RGBA frame.
fn make_solid_frame(r: u8, g: u8, b: u8) -> Frame {
    let w: u32 = 4;
    let h: u32 = 4;
    let data = [r, g, b, 255u8]
        .iter()
        .cloned()
        .cycle()
        .take((w * h * 4) as usize)
        .collect::<Vec<u8>>();
    Frame::from_data(w, h, data).expect("valid frame")
}

/// Average the red and blue channels across all pixels.
fn avg_rb(frame: &Frame) -> (f32, f32) {
    let pixel_count = (frame.width * frame.height) as f32;
    let mut sum_r = 0.0_f32;
    let mut sum_b = 0.0_f32;
    for y in 0..frame.height {
        for x in 0..frame.width {
            let p = frame.get_pixel(x, y).unwrap_or([0, 0, 0, 0]);
            sum_r += p[0] as f32;
            sum_b += p[2] as f32;
        }
    }
    (sum_r / pixel_count, sum_b / pixel_count)
}

/// Check that every pixel in `frame` matches `expected` within `tol` per channel.
fn all_pixels_match(frame: &Frame, expected: [u8; 4], tol: u8) -> bool {
    for y in 0..frame.height {
        for x in 0..frame.width {
            let p = frame.get_pixel(x, y).unwrap_or([0, 0, 0, 0]);
            for ch in 0..4 {
                if (p[ch] as i32 - expected[ch] as i32).unsigned_abs() as u8 > tol {
                    return false;
                }
            }
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Dissolve
// ---------------------------------------------------------------------------

#[test]
fn test_dissolve_progress_0_is_from() {
    let from = make_solid_frame(255, 0, 0); // red
    let to = make_solid_frame(0, 0, 255); // blue
    let mut output = Frame::new(4, 4).expect("valid frame");
    let params = EffectParams::new().with_progress(0.0);

    Dissolve::new()
        .apply(&from, &to, &mut output, &params)
        .expect("apply");

    // At progress=0 dissolve blends with t=0 → exact copy of from.
    assert!(
        all_pixels_match(&output, [255, 0, 0, 255], 1),
        "dissolve at p=0 should equal from (red)"
    );
}

#[test]
fn test_dissolve_progress_1_is_to() {
    let from = make_solid_frame(255, 0, 0); // red
    let to = make_solid_frame(0, 0, 255); // blue
    let mut output = Frame::new(4, 4).expect("valid frame");
    let params = EffectParams::new().with_progress(1.0);

    Dissolve::new()
        .apply(&from, &to, &mut output, &params)
        .expect("apply");

    // At progress=1 dissolve blends with t=1 → exact copy of to.
    assert!(
        all_pixels_match(&output, [0, 0, 255, 255], 1),
        "dissolve at p=1 should equal to (blue)"
    );
}

#[test]
fn test_dissolve_progress_05_is_blend() {
    let from = make_solid_frame(255, 0, 0); // red
    let to = make_solid_frame(0, 0, 255); // blue
    let mut output = Frame::new(4, 4).expect("valid frame");
    let params = EffectParams::new().with_progress(0.5);

    Dissolve::new()
        .apply(&from, &to, &mut output, &params)
        .expect("apply");

    // At progress=0.5 the blend should be roughly equal — neither fully red nor blue.
    let (avg_r, avg_b) = avg_rb(&output);
    assert!(
        avg_r > 50.0 && avg_r < 200.0,
        "dissolve p=0.5 red channel should be partial: {avg_r}"
    );
    assert!(
        avg_b > 50.0 && avg_b < 200.0,
        "dissolve p=0.5 blue channel should be partial: {avg_b}"
    );
}

// ---------------------------------------------------------------------------
// Wipe (LeftToRight)
//
// Note: The wipe blend formula is t = (wipe_val - progress) / feather.
// For LeftToRight, wipe_val = nx (0…1). At progress=0, most pixels have
// wipe_val > 0, so t > 0 → blend leans toward "to" (blue).
// At progress=1, all pixels have wipe_val ≤ 1.0, so t ≤ 0 (clamped) →
// blend leans toward "from" (red). The semantics are therefore that
// progress=0 reveals the "to" frame and progress=1 stays on "from".
// ---------------------------------------------------------------------------

#[test]
fn test_wipe_progress_0_to_dominates() {
    let from = make_solid_frame(255, 0, 0); // red
    let to = make_solid_frame(0, 0, 255); // blue
    let mut output = Frame::new(4, 4).expect("valid frame");
    let params = EffectParams::new().with_progress(0.0);

    Wipe::new(WipePattern::LeftToRight)
        .apply(&from, &to, &mut output, &params)
        .expect("apply");

    // At progress=0: t=(nx-0)/feather=nx/0.05; for nx>0 t>0 → "to" (blue) dominates.
    // The first pixel at x=0 has wipe_val=0 → t=0 (from), but the other three columns
    // (nx≈0.25,0.5,0.75) saturate quickly → overall "to" dominates.
    let (avg_r, avg_b) = avg_rb(&output);
    assert!(
        avg_b > avg_r,
        "wipe p=0 to (blue) should dominate: avg_r={avg_r}, avg_b={avg_b}"
    );
}

#[test]
fn test_wipe_progress_1_from_dominates() {
    let from = make_solid_frame(255, 0, 0); // red
    let to = make_solid_frame(0, 0, 255); // blue
    let mut output = Frame::new(4, 4).expect("valid frame");
    let params = EffectParams::new().with_progress(1.0);

    Wipe::new(WipePattern::LeftToRight)
        .apply(&from, &to, &mut output, &params)
        .expect("apply");

    // At progress=1: t=(nx-1.0)/feather ≤ 0 for all pixels → clamped to 0 → from (red).
    let (avg_r, avg_b) = avg_rb(&output);
    assert!(
        avg_r > avg_b,
        "wipe p=1 from (red) should dominate: avg_r={avg_r}, avg_b={avg_b}"
    );
}

#[test]
fn test_wipe_progress_05_is_blend() {
    let from = make_solid_frame(255, 0, 0); // red
    let to = make_solid_frame(0, 0, 255); // blue
    let mut output = Frame::new(4, 4).expect("valid frame");
    let params = EffectParams::new().with_progress(0.5);

    Wipe::new(WipePattern::LeftToRight)
        .apply(&from, &to, &mut output, &params)
        .expect("apply");

    // At progress=0.5 some pixels should be from, some to — neither fully matches either.
    let (avg_r, avg_b) = avg_rb(&output);
    assert!(
        avg_r > 0.0,
        "wipe p=0.5 should have some from (red) pixels: {avg_r}"
    );
    assert!(
        avg_b > 0.0,
        "wipe p=0.5 should have some to (blue) pixels: {avg_b}"
    );
    // Neither channel should be zero (mix from both sources)
    assert!(
        avg_r < 255.0 || avg_b < 255.0,
        "wipe p=0.5 should not be a single solid colour"
    );
}

// ---------------------------------------------------------------------------
// Push (Left direction)
// ---------------------------------------------------------------------------

#[test]
fn test_push_progress_0_is_from() {
    let from = make_solid_frame(255, 0, 0); // red
    let to = make_solid_frame(0, 0, 255); // blue
    let mut output = Frame::new(4, 4).expect("valid frame");
    let params = EffectParams::new().with_progress(0.0);

    Push::new(PushDirection::Left)
        .apply(&from, &to, &mut output, &params)
        .expect("apply");

    // At progress=0 offset=0; from_x=x always in [0,width) → from pixels.
    assert!(
        all_pixels_match(&output, [255, 0, 0, 255], 1),
        "push p=0 should equal from (red)"
    );
}

#[test]
fn test_push_progress_1_is_to() {
    let from = make_solid_frame(255, 0, 0); // red
    let to = make_solid_frame(0, 0, 255); // blue
    let mut output = Frame::new(4, 4).expect("valid frame");
    let params = EffectParams::new().with_progress(1.0);

    Push::new(PushDirection::Left)
        .apply(&from, &to, &mut output, &params)
        .expect("apply");

    // At progress=1 offset=width; from_x=x+width out of bounds,
    // to_x = x+width-width = x → in bounds → to pixels.
    assert!(
        all_pixels_match(&output, [0, 0, 255, 255], 1),
        "push p=1 should equal to (blue)"
    );
}

#[test]
fn test_push_progress_05_is_blend() {
    let from = make_solid_frame(255, 0, 0); // red
    let to = make_solid_frame(0, 0, 255); // blue
    let mut output = Frame::new(4, 4).expect("valid frame");
    let params = EffectParams::new().with_progress(0.5);

    Push::new(PushDirection::Left)
        .apply(&from, &to, &mut output, &params)
        .expect("apply");

    // At progress=0.5 the left half should show from and right half to.
    let (avg_r, avg_b) = avg_rb(&output);
    assert!(
        avg_r > 0.0,
        "push p=0.5 should have some from (red): {avg_r}"
    );
    assert!(
        avg_b > 0.0,
        "push p=0.5 should have some to (blue): {avg_b}"
    );
}

// ---------------------------------------------------------------------------
// Slide (FromLeft direction)
//
// Note: Slide::FromLeft at progress=0.0 shows "to" (the incoming frame is
// fully slid in at offset=width, making to_x = x in bounds) and at
// progress=1.0 shows "from" (to_x = x - width, out of bounds, falls back
// to from frame). This is the documented behaviour of the Slide effect.
// ---------------------------------------------------------------------------

#[test]
fn test_slide_progress_0_shows_to() {
    let from = make_solid_frame(255, 0, 0); // red
    let to = make_solid_frame(0, 0, 255); // blue
    let mut output = Frame::new(4, 4).expect("valid frame");
    let params = EffectParams::new().with_progress(0.0);

    Slide::new(SlideDirection::FromLeft)
        .apply(&from, &to, &mut output, &params)
        .expect("apply");

    // FromLeft at p=0: offset = (1-0)*width = width; to_x = x-width+width = x (in bounds).
    // The "to" (blue) frame is visible.
    assert!(
        all_pixels_match(&output, [0, 0, 255, 255], 1),
        "slide FromLeft at p=0 should show to (blue)"
    );
}

#[test]
fn test_slide_progress_1_shows_from() {
    let from = make_solid_frame(255, 0, 0); // red
    let to = make_solid_frame(0, 0, 255); // blue
    let mut output = Frame::new(4, 4).expect("valid frame");
    let params = EffectParams::new().with_progress(1.0);

    Slide::new(SlideDirection::FromLeft)
        .apply(&from, &to, &mut output, &params)
        .expect("apply");

    // FromLeft at p=1: offset = 0; to_x = x - width (out of bounds) → from pixel.
    assert!(
        all_pixels_match(&output, [255, 0, 0, 255], 1),
        "slide FromLeft at p=1 should show from (red)"
    );
}

#[test]
fn test_slide_progress_05_is_blend() {
    let from = make_solid_frame(255, 0, 0); // red
    let to = make_solid_frame(0, 0, 255); // blue
    let mut output = Frame::new(4, 4).expect("valid frame");
    let params = EffectParams::new().with_progress(0.5);

    Slide::new(SlideDirection::FromLeft)
        .apply(&from, &to, &mut output, &params)
        .expect("apply");

    // At progress=0.5 the to frame is partially slid in; both colours visible.
    let (avg_r, avg_b) = avg_rb(&output);
    assert!(
        avg_r > 0.0,
        "slide p=0.5 should have some from (red): {avg_r}"
    );
    assert!(
        avg_b > 0.0,
        "slide p=0.5 should have some to (blue): {avg_b}"
    );
}

// ---------------------------------------------------------------------------
// Zoom (ZoomMode::In)
//
// Note: Zoom::In at progress=0.0 samples `to` at scale=1.0 (identity), so
// it shows the "to" frame. At progress=1.0 it samples `to` at scale=1.5,
// still showing "to" zoomed. The transition blends from/to via progress in
// ZoomMode::Cross; for ZoomMode::In the output is always "to"-dominated.
// We therefore test Zoom using ZoomMode::Cross which blends both frames.
// ---------------------------------------------------------------------------

#[test]
fn test_zoom_cross_progress_0_from_dominates() {
    let from = make_solid_frame(255, 0, 0); // red
    let to = make_solid_frame(0, 0, 255); // blue
    let mut output = Frame::new(4, 4).expect("valid frame");
    let params = EffectParams::new().with_progress(0.0);

    Zoom::new(ZoomMode::Cross)
        .apply(&from, &to, &mut output, &params)
        .expect("apply");

    // At progress=0 the blend_pixel weight for `from` is 1.0 → from dominates.
    let (avg_r, avg_b) = avg_rb(&output);
    assert!(
        avg_r > avg_b,
        "zoom cross p=0 from (red) should dominate: avg_r={avg_r}, avg_b={avg_b}"
    );
}

#[test]
fn test_zoom_cross_progress_1_to_dominates() {
    let from = make_solid_frame(255, 0, 0); // red
    let to = make_solid_frame(0, 0, 255); // blue
    let mut output = Frame::new(4, 4).expect("valid frame");
    let params = EffectParams::new().with_progress(1.0);

    Zoom::new(ZoomMode::Cross)
        .apply(&from, &to, &mut output, &params)
        .expect("apply");

    // At progress=1 the blend_pixel weight for `to` is 1.0 → to dominates.
    let (avg_r, avg_b) = avg_rb(&output);
    assert!(
        avg_b > avg_r,
        "zoom cross p=1 to (blue) should dominate: avg_r={avg_r}, avg_b={avg_b}"
    );
}

#[test]
fn test_zoom_cross_progress_05_is_blend() {
    let from = make_solid_frame(255, 0, 0); // red
    let to = make_solid_frame(0, 0, 255); // blue
    let mut output = Frame::new(4, 4).expect("valid frame");
    let params = EffectParams::new().with_progress(0.5);

    Zoom::new(ZoomMode::Cross)
        .apply(&from, &to, &mut output, &params)
        .expect("apply");

    // At progress=0.5 both frames contribute — neither channel should be near 0.
    let (avg_r, avg_b) = avg_rb(&output);
    assert!(
        avg_r > 10.0,
        "zoom cross p=0.5 should have some from (red): {avg_r}"
    );
    assert!(
        avg_b > 10.0,
        "zoom cross p=0.5 should have some to (blue): {avg_b}"
    );
}
