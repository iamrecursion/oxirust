//! Synthetic drift-detection conformance tests for `oximedia-timecode`.
//!
//! These tests feed the stateful `DriftDetector` known synthetic drift profiles
//! and assert that the linear-regression analysis recovers the expected
//! parts-per-million (PPM) clock offset and the correct correction strategy.
//!
//! ## Sampling note (why a long window, not 1 h)
//!
//! `DriftSample` carries `observed_frames`/`expected_frames` as `u64`, so every
//! sampled drift value is an integer number of frames. A +10 ppm offset at
//! 30 fps accumulates only `0.0003 frames/sec`. Over a single hour that is a
//! total drift of just `0.0003 * 3600 ≈ 1.08` frames — smaller than the
//! ±0.5-frame quantization of a `u64` count. A regression over such data is
//! dominated by rounding artifacts (it reads ≈13.8 ppm, not 10 ppm), so a
//! 1-hour / 3601-sample window cannot physically resolve a 10 ppm offset with
//! integer frame counts.
//!
//! To assert a *meaningful* +10 ppm recovery we therefore sample once per second
//! over a 24-hour window (86 401 samples). At 24 h the accumulated drift is
//! ~26 frames, which the regression resolves to 10.0072 ppm — comfortably inside
//! the ±0.5 ppm tolerance. The detector's regression math itself is exact; this
//! is purely about giving it enough whole-frame signal to fit.

use oximedia_timecode::tc_drift::{CorrectionStrategy, DriftConfig, DriftDetector, DriftSample};
use oximedia_timecode::FrameRate;

/// Seconds in a 24-hour synthetic observation window.
const WINDOW_SECS: u64 = 24 * 3600;

/// Parts-per-million target offset for the drift scenario.
const TARGET_PPM: f64 = 10.0;

/// Feeds a known +10 ppm clock offset at 30 fps and asserts the detector
/// recovers `drift_ppm ≈ 10.0`.
///
/// The reference clock advances at exactly 30 fps (`expected_frames = 30 * t`),
/// while the observed clock runs `(1 + 10e-6)` faster
/// (`observed_frames = round(30 * 1.00001 * t)`). Sampling once per second over
/// 24 h yields 86 401 samples with ~26 frames of accumulated drift.
#[test]
fn detects_known_plus_10ppm_drift_30fps_24h() {
    let mut det = DriftDetector::new(DriftConfig::new(FrameRate::Fps30).with_tolerance(0));

    let fast_rate = 30.0_f64 * (1.0 + TARGET_PPM * 1e-6);
    for t in 0..=WINDOW_SECS {
        let expected_frames = 30 * t;
        let observed_frames = (fast_rate * t as f64).round() as u64;
        det.add_sample(DriftSample::new(t as f64, observed_frames, expected_frames));
    }

    let a = det.analyze().expect(">=min_samples");
    assert!(
        (a.drift_ppm - TARGET_PPM).abs() < 0.5,
        "drift_ppm = {} (expected ~{TARGET_PPM})",
        a.drift_ppm
    );
    assert_eq!(a.sample_count, (WINDOW_SECS + 1) as usize);
    // Accumulated drift (~26 frames) far exceeds the zero-frame tolerance.
    assert!(
        !a.within_tolerance,
        "max_drift_frames = {} should exceed tolerance 0",
        a.max_drift_frames
    );
}

/// With observed == expected at every sample, the analysis must report
/// effectively zero drift, be within tolerance, and recommend no correction.
#[test]
fn zero_drift_none_strategy() {
    let mut det = DriftDetector::new(DriftConfig::new(FrameRate::Fps30));
    for t in 0..=100u64 {
        let frames = 30 * t;
        det.add_sample(DriftSample::new(t as f64, frames, frames));
    }

    let a = det.analyze().expect(">=min_samples");
    assert!(a.drift_ppm.abs() < 1.0, "drift_ppm = {}", a.drift_ppm);
    assert!(a.within_tolerance);
    assert_eq!(a.recommended_strategy, CorrectionStrategy::None);
}

/// Below the configured `min_samples` (default 3) the detector must refuse to
/// analyze and return `None`.
#[test]
fn below_min_samples_returns_none() {
    let mut det = DriftDetector::new(DriftConfig::new(FrameRate::Fps30));
    det.add_sample(DriftSample::new(0.0, 0, 0));
    det.add_sample(DriftSample::new(1.0, 30, 30));
    assert!(det.analyze().is_none());
}
