//! End-to-end forensic splice-detection tests.
//!
//! These exercise the full `EditDetector::detect` and `AuthenticityVerifier::verify`
//! pipelines (energy + spectral-centroid + phase-discontinuity fusion), not just the
//! lower-level `PhaseDiscontinuityDetector` (which is covered in-crate).
//!
//! A 300 Hz → 3000 Hz splice is chosen because the spectral centroid jumps by far
//! more than the detector's fixed 500 Hz threshold, so the splice is robustly
//! detectable. If even this is missed, it points to a real bug — not an over-tight
//! expectation.

use oximedia_audio_analysis::forensics::authenticity::AuthenticityVerifier;
use oximedia_audio_analysis::forensics::edit::EditDetector;
use oximedia_audio_analysis::AnalysisConfig;

const SR: f32 = 44_100.0;
const TWO_PI: f32 = 2.0 * std::f32::consts::PI;

/// Generate `n` samples of a sine at `freq` Hz, amplitude `amp`, starting at phase 0.
fn sine(freq: f32, amp: f32, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (TWO_PI * freq * i as f32 / SR).sin() * amp)
        .collect()
}

/// Hard-concatenate a 300 Hz half and a 3000 Hz half to form a spliced signal.
/// The splice is at sample `n_each` (≈ `n_each / SR` seconds).
fn spliced_300_to_3000(n_each: usize) -> Vec<f32> {
    let mut s = sine(300.0, 0.5, n_each);
    s.extend(sine(3000.0, 0.5, n_each));
    s
}

#[test]
fn spliced_audio_detected_near_boundary() {
    let detector = EditDetector::new(AnalysisConfig::default());

    // 0.5 s of 300 Hz then 0.5 s of 3000 Hz → splice at sample 22050 (t ≈ 0.5 s).
    let n_each = 22_050_usize;
    let samples = spliced_300_to_3000(n_each);

    let result = detector
        .detect(&samples, SR)
        .expect("edit detection should succeed");

    assert!(
        result.num_edits >= 1,
        "expected at least one detected edit at the 300→3000 Hz splice, got {} (times: {:?})",
        result.num_edits,
        result.edit_times
    );

    let splice_t = n_each as f32 / SR; // ≈ 0.5 s
    let near = result
        .edit_times
        .iter()
        .any(|&t| (t - splice_t).abs() <= 0.05);
    assert!(
        near,
        "expected a detected edit within ±0.05 s of {splice_t:.3} s, got times: {:?}",
        result.edit_times
    );
}

#[test]
fn continuous_sine_no_splice() {
    let detector = EditDetector::new(AnalysisConfig::default());

    // A single, phase-continuous 440 Hz tone has no discontinuities anywhere.
    let samples = sine(440.0, 0.5, 44_100);

    let result = detector
        .detect(&samples, SR)
        .expect("edit detection should succeed");

    // A clean continuous sine MUST produce zero edits. A non-zero count here is a
    // false positive and would indicate a real bug in the energy / spectral /
    // phase fusion — investigate rather than loosen this bound.
    assert_eq!(
        result.num_edits, 0,
        "continuous sine should yield 0 edits (false-positive bug if >0), got {} at times {:?}",
        result.num_edits, result.edit_times
    );
}

#[test]
fn authenticity_verifier_scores_spliced_lower_than_clean() {
    let verifier = AuthenticityVerifier::new(AnalysisConfig::default());

    // Clean: 1 s of a single 440 Hz tone.
    let clean = sine(440.0, 0.5, 44_100);
    // Spliced: 0.5 s of 300 Hz then 0.5 s of 3000 Hz (total 1 s).
    let spliced = spliced_300_to_3000(22_050);

    let clean_r = verifier
        .verify(&clean, SR)
        .expect("clean verification should succeed");
    let splice_r = verifier
        .verify(&spliced, SR)
        .expect("spliced verification should succeed");

    assert!(
        splice_r.detected_edits >= 1,
        "verifier should detect ≥1 edit in the spliced signal, got {}",
        splice_r.detected_edits
    );
    assert!(
        splice_r.authenticity_score <= clean_r.authenticity_score,
        "spliced authenticity_score ({}) should be <= clean authenticity_score ({})",
        splice_r.authenticity_score,
        clean_r.authenticity_score
    );
}
