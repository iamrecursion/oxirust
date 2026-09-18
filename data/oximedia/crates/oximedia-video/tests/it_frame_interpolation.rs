//! Integration tests for frame_interpolation (L38).
//!
//! Tests verify that `interpolate_pair` with `alpha = 0.5` produces a frame
//! that is the midpoint between two given frames (for Blend mode with known
//! linear motion — 0→100 in all channels, midpoint should be ~50).

use oximedia_video::frame_interpolation::{FrameInterpolationMethod, FrameInterpolator};

/// RGBA frame of uniform colour.
fn solid_rgba(r: u8, g: u8, b: u8, a: u8, width: u32, height: u32) -> Vec<u8> {
    let pixel_count = (width * height) as usize;
    let mut buf = Vec::with_capacity(pixel_count * 4);
    for _ in 0..pixel_count {
        buf.push(r);
        buf.push(g);
        buf.push(b);
        buf.push(a);
    }
    buf
}

#[test]
fn test_interpolate_pair_midpoint_linear_motion() {
    let width = 16u32;
    let height = 16u32;

    // Frame A: all pixels (0, 0, 0, 255)
    let frame_a = solid_rgba(0, 0, 0, 255, width, height);
    // Frame B: all pixels (100, 100, 100, 255)
    let frame_b = solid_rgba(100, 100, 100, 255, width, height);

    let interp = FrameInterpolator::new(24.0, 48.0, FrameInterpolationMethod::Blend);
    let mid = interp.interpolate_pair(&frame_a, &frame_b, width, height, 0.5);

    assert_eq!(
        mid.len(),
        (width * height * 4) as usize,
        "output frame byte-length mismatch"
    );

    // Each RGB channel should be approximately 50 (within ±2 for integer rounding).
    for chunk in mid.chunks(4) {
        let (r, g, b) = (chunk[0], chunk[1], chunk[2]);
        assert!((45..=55).contains(&r), "R channel {r} not near 50");
        assert!((45..=55).contains(&g), "G channel {g} not near 50");
        assert!((45..=55).contains(&b), "B channel {b} not near 50");
    }
}

#[test]
fn test_interpolate_pair_alpha_zero_equals_frame_a() {
    let width = 8u32;
    let height = 8u32;
    let frame_a = solid_rgba(80, 80, 80, 255, width, height);
    let frame_b = solid_rgba(200, 200, 200, 255, width, height);

    let interp = FrameInterpolator::new(24.0, 48.0, FrameInterpolationMethod::Blend);
    let result = interp.interpolate_pair(&frame_a, &frame_b, width, height, 0.0);

    // alpha=0 → should equal frame_a
    assert_eq!(result.len(), frame_a.len());
    for (i, (&out, &expected)) in result.iter().zip(frame_a.iter()).enumerate() {
        assert!(
            out.abs_diff(expected) <= 2,
            "pixel {i}: got {out}, expected {expected} at alpha=0.0"
        );
    }
}

#[test]
fn test_interpolate_pair_alpha_one_equals_frame_b() {
    let width = 8u32;
    let height = 8u32;
    let frame_a = solid_rgba(80, 80, 80, 255, width, height);
    let frame_b = solid_rgba(200, 200, 200, 255, width, height);

    let interp = FrameInterpolator::new(24.0, 48.0, FrameInterpolationMethod::Blend);
    let result = interp.interpolate_pair(&frame_a, &frame_b, width, height, 1.0);

    // alpha=1 → should equal frame_b
    assert_eq!(result.len(), frame_b.len());
    for (i, (&out, &expected)) in result.iter().zip(frame_b.iter()).enumerate() {
        assert!(
            out.abs_diff(expected) <= 2,
            "pixel {i}: got {out}, expected {expected} at alpha=1.0"
        );
    }
}

#[test]
fn test_output_frame_count_double_rate() {
    let interp = FrameInterpolator::new(24.0, 48.0, FrameInterpolationMethod::Blend);
    // 2× frame rate → should produce approximately 2× the frames.
    let count = interp.output_frame_count(10);
    // Exact ratio depends on implementation, but must be ≥ 10 for 2× upscaling.
    assert!(
        count >= 10,
        "expected >= 10 output frames for 10 input frames at 2x rate, got {count}"
    );
}

#[test]
fn test_process_produces_nonempty_output() {
    let width = 8u32;
    let height = 8u32;
    let frame_a = solid_rgba(50, 50, 50, 255, width, height);
    let frame_b = solid_rgba(150, 150, 150, 255, width, height);
    let frames = vec![frame_a, frame_b];

    let interp = FrameInterpolator::new(24.0, 48.0, FrameInterpolationMethod::Blend);
    let result = interp.process(&frames, width, height);

    assert!(
        !result.frames.is_empty(),
        "process should produce at least one output frame"
    );
    // Every output frame must have the correct byte size.
    for (i, frame) in result.frames.iter().enumerate() {
        assert_eq!(
            frame.len(),
            (width * height * 4) as usize,
            "output frame {i} has wrong size"
        );
    }
}
