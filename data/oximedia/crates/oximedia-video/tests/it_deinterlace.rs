//! Integration tests for deinterlace (L37).
//!
//! Tests verify:
//! 1. Bob round-trip: input interlaced luma, output has 2 frames of correct dimensions.
//! 2. All output pixels remain in a plausible range.
//! 3. Bob output frame dimensions match the input frame dimensions.

use oximedia_video::deinterlace::{DeinterlaceMethod, Deinterlacer, FieldOrder};

/// Build a synthetic interlaced luma frame where even rows = 200 and odd rows = 50.
fn make_interlaced_luma(width: u32, height: u32) -> Vec<u8> {
    let mut buf = vec![0u8; (width * height) as usize];
    for row in 0..height as usize {
        let val = if row % 2 == 0 { 200u8 } else { 50u8 };
        for col in 0..width as usize {
            buf[row * width as usize + col] = val;
        }
    }
    buf
}

#[test]
fn test_deinterlace_roundtrip_psnr() {
    let width = 64u32;
    let height = 64u32;
    let interlaced = make_interlaced_luma(width, height);

    let d = Deinterlacer::new(DeinterlaceMethod::Bob, FieldOrder::TopFieldFirst);
    let output = d.deinterlace_frame(None, &interlaced, None, width, height);

    // Bob must return 2 frames.
    assert_eq!(output.len(), 2, "Bob should return 2 frames");

    // Each frame must have the same number of pixels as the input.
    for (i, frame) in output.iter().enumerate() {
        assert_eq!(
            frame.len(),
            (width * height) as usize,
            "frame {i} length mismatch"
        );
        // Pixels must be within a plausible interpolation range between the two field values.
        for &px in frame {
            assert!(
                px >= 40 && px <= 210,
                "out-of-range pixel {px} in frame {i}"
            );
        }
    }
}

#[test]
fn test_deinterlace_bob_output_dimensions_unchanged() {
    let width = 32u32;
    let height = 32u32;
    let frame = vec![128u8; (width * height) as usize];

    let d = Deinterlacer::new(DeinterlaceMethod::Bob, FieldOrder::TopFieldFirst);
    let out = d.deinterlace_frame(None, &frame, None, width, height);

    assert_eq!(out.len(), 2, "Bob must return exactly 2 frames");
    assert_eq!(
        out[0].len(),
        (width * height) as usize,
        "field 0 length wrong"
    );
    assert_eq!(
        out[1].len(),
        (width * height) as usize,
        "field 1 length wrong"
    );
}

#[test]
fn test_deinterlace_weave_returns_single_frame() {
    let width = 64u32;
    let height = 32u32;
    let frame = make_interlaced_luma(width, height);

    let d = Deinterlacer::new(DeinterlaceMethod::Weave, FieldOrder::TopFieldFirst);
    let out = d.deinterlace_frame(None, &frame, None, width, height);

    // Weave (non-Bob) returns exactly 1 frame.
    assert_eq!(out.len(), 1, "Weave should return 1 frame");
    assert_eq!(out[0].len(), (width * height) as usize);
}

#[test]
fn test_deinterlace_blend_uniform_frame_is_unchanged() {
    let width = 16u32;
    let height = 16u32;
    // A completely uniform frame should survive Blend deinterlacing unchanged.
    let frame = vec![128u8; (width * height) as usize];

    let d = Deinterlacer::new(DeinterlaceMethod::Blend, FieldOrder::TopFieldFirst);
    let out = d.deinterlace_frame(None, &frame, None, width, height);

    assert_eq!(out.len(), 1);
    // All pixels should remain 128 because blending 128 and 128 = 128.
    for &px in &out[0] {
        assert_eq!(px, 128, "blend of uniform field changed pixel value");
    }
}

#[test]
fn test_deinterlace_process_sequence_length() {
    let width = 16u32;
    let height = 16u32;
    let frame = vec![100u8; (width * height) as usize];
    // Feed a sequence of 5 identical frames through Weave.
    let frames: Vec<Vec<u8>> = (0..5).map(|_| frame.clone()).collect();

    let d = Deinterlacer::new(DeinterlaceMethod::Weave, FieldOrder::TopFieldFirst);
    let out = d.process_sequence(&frames, width, height);

    // process_sequence should produce ≥ number-of-input-frames outputs (1:1 for non-Bob).
    assert!(
        out.len() >= frames.len(),
        "expected at least {} output frames, got {}",
        frames.len(),
        out.len()
    );
}
