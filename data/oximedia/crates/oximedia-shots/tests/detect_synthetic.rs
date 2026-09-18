//! Synthetic ground-truth tests for shot/transition detection.
//!
//! These tests pin the *real* behaviour of [`oximedia_shots::ShotDetector`]
//! and the low-level [`oximedia_shots::detect`] primitives against frame
//! sequences whose cut/dissolve structure is known by construction.
//!
//! Synthetic corpus:
//! - **cut sequence**: gray(50) for frames 0–9, white(255) for frames 10–24,
//!   gray(100) for frames 25–39 — two hard cuts, at frame 10 and frame 25.
//!   Every frame is spatially uniform, so the Sobel edge term contributes
//!   nothing and the histogram (chi-squared) term alone drives detection.
//! - **dissolve ramp**: a 12-frame linear luminance ramp 40→200.

use oximedia_shots::detect::{CutDetector, DissolveDetector};
use oximedia_shots::frame_buffer::FrameBuffer;
use oximedia_shots::ShotDetector;

/// Frame side length used across the synthetic corpus.
const SIDE: usize = 64;

/// Build the canonical "two hard cuts" sequence.
///
/// Returns 40 frames: `[50; 0..10] ++ [255; 10..25] ++ [100; 25..40]`.
/// Ground-truth shot starts are frames {0, 10, 25}.
fn cut_sequence() -> Vec<FrameBuffer> {
    let mut frames = Vec::with_capacity(40);
    for _ in 0..10 {
        frames.push(FrameBuffer::from_elem(SIDE, SIDE, 3, 50));
    }
    for _ in 10..25 {
        frames.push(FrameBuffer::from_elem(SIDE, SIDE, 3, 255));
    }
    for _ in 25..40 {
        frames.push(FrameBuffer::from_elem(SIDE, SIDE, 3, 100));
    }
    frames
}

/// Build a 12-frame linear dissolve ramp from value 40 up to 200.
///
/// `value(k) = 40 + (200 - 40) * k / 10` for `k` in `0..12`.
fn dissolve_ramp() -> Vec<FrameBuffer> {
    let mut frames = Vec::with_capacity(12);
    for k in 0..12u32 {
        let v = (40 + (200 - 40) * k / 10) as u8;
        frames.push(FrameBuffer::from_elem(SIDE, SIDE, 3, v));
    }
    frames
}

/// Test 1 — the canonical cut sequence must yield exactly 3 shots whose
/// starts line up with the ground-truth boundaries {0, 10, 25} (±1 frame,
/// since the edge term can also flag the frame immediately after a cut).
#[test]
fn cut_sequence_yields_three_shots_at_expected_starts() {
    let detector = ShotDetector::default();
    let frames = cut_sequence();

    let shots = detector
        .detect_shots(&frames)
        .expect("detect_shots on a valid uniform sequence must succeed");

    assert_eq!(
        shots.len(),
        3,
        "expected exactly 3 shots from a two-cut sequence, got {}",
        shots.len()
    );

    // A boundary at pair index i means a shot STARTS at frame i; the shot's
    // start PTS is recorded in frames at a 1/30 timebase.
    let expected_starts = [0i64, 10, 25];
    for (shot, &want) in shots.iter().zip(expected_starts.iter()) {
        let got = shot.start.pts;
        assert!(
            (got - want).abs() <= 1,
            "shot start {got} not within +/-1 of expected {want}"
        );
    }
}

/// Test 2 — a run of identical frames contains no cut, so the whole run is a
/// single shot.
#[test]
fn identical_frames_collapse_to_single_shot() {
    let detector = ShotDetector::default();
    let frames: Vec<FrameBuffer> = (0..20)
        .map(|_| FrameBuffer::from_elem(SIDE, SIDE, 3, 120))
        .collect();

    let shots = detector
        .detect_shots(&frames)
        .expect("detect_shots on identical frames must succeed");

    assert_eq!(
        shots.len(),
        1,
        "20 identical gray frames must produce exactly one shot, got {}",
        shots.len()
    );
}

/// Test 3 — the low-level [`CutDetector`] flags a gray→white pair as a cut
/// with a meaningful score, and a gray→gray pair as a non-cut with a tiny
/// score.
#[test]
fn cut_detector_flags_gray_to_white_and_not_gray_to_gray() {
    let detector = CutDetector::with_params(0.3, 0.4, 5);

    let gray = FrameBuffer::from_elem(SIDE, SIDE, 3, 50);
    let white = FrameBuffer::from_elem(SIDE, SIDE, 3, 255);

    let (is_cut, score) = detector
        .detect_cut(&gray, &white)
        .expect("detect_cut gray->white must succeed");
    assert!(is_cut, "gray->white must be detected as a cut");
    assert!(
        score > 0.3,
        "gray->white cut score should exceed 0.3, got {score}"
    );

    let gray2 = FrameBuffer::from_elem(SIDE, SIDE, 3, 50);
    let (is_cut2, score2) = detector
        .detect_cut(&gray, &gray2)
        .expect("detect_cut gray->gray must succeed");
    assert!(!is_cut2, "identical gray frames must not be a cut");
    assert!(
        score2 < 0.1,
        "gray->gray score should be near zero, got {score2}"
    );
}

/// Test 4 — relative scoring: the smooth 12-frame dissolve ramp's peak
/// dissolve score should exceed the dissolve score harvested from an abrupt
/// gray→white→gray triplet (padded to the detector's minimum window so the
/// detector actually runs its windowed analysis on both inputs).
#[test]
fn dissolve_ramp_scores_higher_than_abrupt_triplet() {
    let detector = DissolveDetector::new();

    let ramp = dissolve_ramp();
    let (_ramp_is, ramp_score, _ramp_pos) = detector
        .detect_dissolve(&ramp)
        .expect("detect_dissolve on ramp must succeed");

    // Abrupt triplet gray->white->gray, padded with steady frames on both
    // ends so the sequence reaches the >=10-frame minimum the detector needs.
    let mut abrupt = Vec::with_capacity(12);
    for _ in 0..5 {
        abrupt.push(FrameBuffer::from_elem(SIDE, SIDE, 3, 40));
    }
    abrupt.push(FrameBuffer::from_elem(SIDE, SIDE, 3, 40)); // gray
    abrupt.push(FrameBuffer::from_elem(SIDE, SIDE, 3, 255)); // white
    abrupt.push(FrameBuffer::from_elem(SIDE, SIDE, 3, 40)); // gray
    for _ in 0..4 {
        abrupt.push(FrameBuffer::from_elem(SIDE, SIDE, 3, 40));
    }
    let (_ab_is, abrupt_score, _ab_pos) = detector
        .detect_dissolve(&abrupt)
        .expect("detect_dissolve on abrupt triplet must succeed");

    assert!(
        ramp_score > abrupt_score,
        "smooth dissolve ramp score {ramp_score} should exceed abrupt triplet score {abrupt_score}"
    );
}

/// Test 5 — the dissolve detector returns the explicit `(false, 0.0, 0)`
/// sentinel when handed fewer frames than its minimum window.
#[test]
fn dissolve_detector_below_minimum_returns_sentinel() {
    let detector = DissolveDetector::new();
    let short: Vec<FrameBuffer> = (0..5)
        .map(|k| FrameBuffer::from_elem(SIDE, SIDE, 3, (40 + k * 20) as u8))
        .collect();

    let (is_dissolve, score, pos) = detector
        .detect_dissolve(&short)
        .expect("detect_dissolve on a short sequence must still return Ok");

    assert!(
        !is_dissolve,
        "fewer than min_duration frames cannot dissolve"
    );
    assert!(
        (score - 0.0).abs() < f32::EPSILON,
        "short-sequence dissolve score must be exactly 0.0, got {score}"
    );
    assert_eq!(pos, 0, "short-sequence dissolve position must be 0");
}
