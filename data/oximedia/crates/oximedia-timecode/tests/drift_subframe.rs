//! Sub-frame (`f64`) drift-channel conformance tests for `oximedia-timecode`.
//!
//! The integer `DriftSample::{observed,expected}_frames` (u64) path cannot
//! resolve a small clock offset over a short window: a +10 ppm offset at 30 fps
//! accumulates only `0.0003 frames/sec`, so over a single hour the total drift
//! is `0.0003 * 3600 ≈ 1.08` frames — comparable to the ±0.5-frame quantization
//! of a `u64` count. A linear regression over such integer data is dominated by
//! rounding artifacts (it reads ≈13.8 ppm, not 10 ppm). Wave 28's
//! `tests/drift_synthetic.rs` worked around this by sampling over a 24-hour
//! window to accumulate enough whole-frame signal.
//!
//! The additive exact-`f64` channel (`DriftSample::new_subframe`) retains the
//! true sub-frame counts, so the same regression resolves ~10.0 ppm cleanly
//! over a *one-hour* window — closing the 24-hour workaround. These tests pin:
//!   * the exact-float path recovering ~10 ppm over 1 h (3601 samples),
//!   * the integer path over the same window being quantization-dominated
//!     (~13.8 ppm, demonstrably *not* 10 ppm), and
//!   * the `new_subframe` constructor's rounded-integer / exact-`f64` contract
//!     and its byte-identical fallback for `new`-built (exact = `None`) samples.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use oximedia_timecode::tc_drift::{DriftConfig, DriftDetector, DriftSample};
use oximedia_timecode::FrameRate;

/// Parts-per-million target offset for the drift scenarios.
const TARGET_PPM: f64 = 10.0;

/// Exact-`f64` channel: feeding `new_subframe` with the true (un-rounded)
/// sub-frame counts lets a 1-hour window resolve a +10 ppm offset at 30 fps to
/// within 0.1 ppm — something the integer path cannot do over the same window.
#[test]
fn detects_known_plus_10ppm_drift_30fps_1h_subframe() {
    let mut det = DriftDetector::new(DriftConfig::new(FrameRate::Fps30).with_tolerance(0));

    let fast_rate = 30.0_f64 * (1.0 + TARGET_PPM * 1e-6);
    for t in 0..=3600u64 {
        let expected_frames_exact = 30.0_f64 * t as f64;
        let observed_frames_exact = fast_rate * t as f64;
        det.add_sample(DriftSample::new_subframe(
            t as f64,
            observed_frames_exact,
            expected_frames_exact,
        ));
    }

    let a = det.analyze().expect(">=min_samples");
    assert!(
        (a.drift_ppm - TARGET_PPM).abs() < 0.1,
        "exact-float drift_ppm = {} (expected ~{TARGET_PPM})",
        a.drift_ppm
    );
    assert_eq!(a.sample_count, 3601);
}

/// Integer path control: the same 1-hour scenario built with the rounded `u64`
/// constructor `new` is quantization-dominated — it reads ~13.8 ppm and is
/// demonstrably *not* the true 10 ppm. This is the regression the exact-float
/// channel exists to defeat.
#[test]
fn integer_path_over_1h_is_quantization_dominated() {
    let mut det = DriftDetector::new(DriftConfig::new(FrameRate::Fps30).with_tolerance(0));

    let fast_rate = 30.0_f64 * (1.0 + TARGET_PPM * 1e-6);
    for t in 0..=3600u64 {
        let observed_frames = (fast_rate * t as f64).round() as u64;
        det.add_sample(DriftSample::new(t as f64, observed_frames, 30 * t));
    }

    let a = det.analyze().expect(">=min_samples");
    assert!(
        (a.drift_ppm - 13.8087).abs() < 0.05,
        "integer-path drift_ppm = {} (expected the quantization-dominated ~13.8087)",
        a.drift_ppm
    );
    assert!(
        (a.drift_ppm - TARGET_PPM).abs() > 1.0,
        "integer-path drift_ppm = {} should be far from the true {TARGET_PPM} ppm",
        a.drift_ppm
    );
}

/// `new_subframe` rounds the integer accessors to nearest while retaining the
/// exact `f64` values; `drift_frames()` reflects the (zero) rounded difference,
/// `drift_frames_exact()` reflects the true sub-frame difference.
#[test]
fn new_subframe_sets_rounded_integer_fields() {
    let s = DriftSample::new_subframe(1.0, 30.0003, 30.0);
    assert_eq!(s.observed_frames, 30);
    assert_eq!(s.expected_frames, 30);
    assert_eq!(s.observed_frames_exact, Some(30.0003));
    assert_eq!(s.expected_frames_exact, Some(30.0));
    assert_eq!(s.drift_frames(), 0);
    assert!(
        (s.drift_frames_exact() - 0.0003).abs() < 1e-9,
        "drift_frames_exact = {}",
        s.drift_frames_exact()
    );
}

/// `new` leaves the exact channel as `None`, so `drift_frames_exact()` falls
/// back to the integer drift bit-for-bit — guaranteeing existing integer-path
/// behavior is unchanged.
#[test]
fn new_keeps_integer_path_unchanged() {
    let s = DriftSample::new(2.0, 61, 60);
    assert_eq!(s.observed_frames_exact, None);
    assert_eq!(s.expected_frames_exact, None);
    assert_eq!(s.drift_frames(), 1);
    assert!(
        (s.drift_frames_exact() - 1.0).abs() < 1e-12,
        "drift_frames_exact = {}",
        s.drift_frames_exact()
    );
}
