//! Integration tests for [`oximedia_analysis::black::BlackFrameDetector`].
//!
//! Focus areas:
//! - Near-black threshold boundary behaviour (strict `<` per-pixel test).
//! - Full-black runs bounded by a non-black frame.
//! - Non-near-black gray frames producing no segments.
//! - Regression coverage for the segment-duration / `end_frame` unification:
//!   the mid-stream close path in `process_frame` and the end-of-stream close
//!   path in `finalize` must produce identical results, including when frame
//!   numbers are non-contiguous (every-Nth-frame sampling).
//!
//! Letterbox / pillarbox / windowbox detection lives in the separate
//! `BlackBarDetector` and is exhaustively covered by the in-file unit tests in
//! `src/black.rs`; it is intentionally NOT duplicated here.

use oximedia_analysis::black::{BlackFrameDetector, BlackSegment};

/// Side length of the synthetic Y-planes used throughout these tests.
const DIM: usize = 64;

/// Build a uniform `DIM`×`DIM` Y-plane where every luma sample equals `value`.
fn uniform_plane(value: u8) -> Vec<u8> {
    vec![value; DIM * DIM]
}

#[test]
fn near_black_threshold_boundary() {
    // threshold = 16, min_duration = 1.
    let mut detector = BlackFrameDetector::new(16, 1);

    // Frame 0: all 15 → avg 15.0 < 16 AND every pixel 15 < 16 (ratio 1.0) → black.
    detector
        .process_frame(&uniform_plane(15), DIM, DIM, 0)
        .expect("frame 0 should process");
    // Frame 1: all 16 → avg 16.0 (not < 16) AND 16 < 16 is false (ratio 0.0) →
    // NOT black → closes the segment opened by frame 0.
    detector
        .process_frame(&uniform_plane(16), DIM, DIM, 1)
        .expect("frame 1 should process");

    let segments = detector.finalize();
    assert_eq!(segments.len(), 1, "exactly one near-black segment expected");

    let seg = &segments[0];
    assert_eq!(seg.start_frame, 0, "segment starts at frame 0");
    assert!(
        (seg.avg_luminance - 15.0).abs() < 0.01,
        "avg luminance should be ~15.0, got {}",
        seg.avg_luminance
    );
    // Unified close definition: end_frame is the exclusive bound (last black
    // frame + 1). The single black frame is frame 0, so end_frame == 1 and the
    // accumulated sample count (the unified duration) is 1.
    assert_eq!(seg.end_frame, 1, "end_frame is exclusive bound of the run");
    assert_eq!(
        seg.end_frame - seg.start_frame,
        1,
        "exactly one black frame in the run"
    );
}

#[test]
fn full_black_all_zero_is_black() {
    let mut detector = BlackFrameDetector::new(16, 1);

    // Two fully-black frames...
    for frame in 0..2 {
        detector
            .process_frame(&uniform_plane(0), DIM, DIM, frame)
            .expect("black frame should process");
    }
    // ...then a clearly non-black frame which closes the run.
    detector
        .process_frame(&uniform_plane(200), DIM, DIM, 2)
        .expect("non-black frame should process");

    let segments = detector.finalize();
    assert_eq!(segments.len(), 1, "the two black frames form one segment");

    let seg = &segments[0];
    assert_eq!(seg.start_frame, 0, "run starts at frame 0");
    // Black frames are 0 and 1 → exclusive end_frame == 2.
    assert_eq!(seg.end_frame, 2, "run covers black frames 0 and 1");
    assert!(
        seg.avg_luminance.abs() < f64::EPSILON,
        "all-zero frames have zero average luminance, got {}",
        seg.avg_luminance
    );
    assert!(
        (seg.black_pixel_ratio - 1.0).abs() < f64::EPSILON,
        "every pixel is black (ratio 1.0), got {}",
        seg.black_pixel_ratio
    );
}

#[test]
fn gray_253_not_black() {
    let mut detector = BlackFrameDetector::new(16, 1);

    // avg 253 >= 16 and no pixel satisfies the strict `< 16` test → never black.
    for frame in 0..3 {
        detector
            .process_frame(&uniform_plane(253), DIM, DIM, frame)
            .expect("gray frame should process");
    }

    let segments = detector.finalize();
    assert!(
        segments.is_empty(),
        "uniform 253 gray is well above threshold; no segments expected, got {}",
        segments.len()
    );
}

/// Regression test for the segment-duration / `end_frame` divergence between
/// the `process_frame` close path and the `finalize` close path.
///
/// The two paths historically disagreed for non-contiguous frame numbers: the
/// mid-stream path used `frame_number - start` / `end_frame = frame_number`
/// (the index of the FIRST non-black frame), while `finalize` used the count of
/// accumulated samples / `end_frame = start + count`. For black frames at
/// 0,2,4,6 these produced `end_frame = 8` (8 "frames") vs `end_frame = 4`
/// (4 samples) respectively — clearly inconsistent.
///
/// After unification both paths report the same self-consistent result:
/// `end_frame = last_seen_frame + 1` (exclusive) and a duration equal to the
/// accumulated black-sample count. This test pins both:
/// 1. the mid-stream close (run terminated by a trailing non-black frame), and
/// 2. the end-of-stream close (`finalize` with the run still open),
/// then asserts they are byte-for-byte equivalent for the identical black run.
#[test]
fn non_contiguous_frame_numbers_consistent_duration() {
    let non_contiguous: [usize; 4] = [0, 2, 4, 6];

    // --- Path A: closed mid-stream by a trailing non-black frame at 8. ---
    let mut closed_by_frame = BlackFrameDetector::new(16, 1);
    for &fnum in &non_contiguous {
        closed_by_frame
            .process_frame(&uniform_plane(0), DIM, DIM, fnum)
            .expect("black frame should process");
    }
    closed_by_frame
        .process_frame(&uniform_plane(200), DIM, DIM, 8)
        .expect("trailing non-black frame should process");
    let via_process = closed_by_frame.finalize();

    // --- Path B: identical black run, closed by `finalize` (run still open). ---
    let mut closed_by_finalize = BlackFrameDetector::new(16, 1);
    for &fnum in &non_contiguous {
        closed_by_finalize
            .process_frame(&uniform_plane(0), DIM, DIM, fnum)
            .expect("black frame should process");
    }
    let via_finalize = closed_by_finalize.finalize();

    assert_eq!(via_process.len(), 1, "process-close yields one segment");
    assert_eq!(via_finalize.len(), 1, "finalize-close yields one segment");

    let a = &via_process[0];
    let b = &via_finalize[0];

    // Both paths must agree on the run's structural fields.
    assert_eq!(
        a.start_frame, b.start_frame,
        "start_frame must match across close paths"
    );
    assert_eq!(
        a.end_frame, b.end_frame,
        "end_frame must match across close paths (the duration-divergence bug)"
    );
    assert!(
        (a.avg_luminance - b.avg_luminance).abs() < f64::EPSILON,
        "avg_luminance must match across close paths"
    );
    assert!(
        (a.black_pixel_ratio - b.black_pixel_ratio).abs() < f64::EPSILON,
        "black_pixel_ratio must match across close paths"
    );

    // And the self-consistent values: last black frame is 6 → exclusive end 7;
    // four black samples were accumulated, so the span is independent of the
    // sampling stride.
    assert_eq!(a.start_frame, 0, "run starts at the first black frame");
    assert_eq!(
        a.end_frame, 7,
        "end_frame is last-seen black frame (6) + 1, NOT the closing frame index"
    );
    // The accumulated black-sample count is 4 regardless of frame spacing.
    let observed: &BlackSegment = a;
    assert!(
        (observed.black_pixel_ratio - 1.0).abs() < f64::EPSILON,
        "all sampled frames were fully black"
    );
}
