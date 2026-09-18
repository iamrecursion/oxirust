//! ABR-ladder ↔ encode-ladder-validator integration tests.
//!
//! These tests bridge the two sibling abstractions:
//!
//! * `abr::AbrLadder` — the generation side (HLS / DASH rung presets).
//! * `encode_ladder_validator::{EncodeLadder, LadderValidator}` — the
//!   validation side (RFC 8216 / ISO-DASH / CMAF compliance).
//!
//! Each `AbrRung` is mapped into a `LadderRung`, then validated against the
//! relevant `LadderSpec`.  This pins that the shipped HLS/DASH presets are
//! actually spec-conformant when run through the validator.

use oximedia_transcode::encode_ladder_validator::{
    EncodeLadder, LadderRung, LadderSpec, LadderValidator,
};
use oximedia_transcode::{AbrLadder, AbrRung};

/// Maps an `AbrRung` (generation side) into a `LadderRung` (validation side).
///
/// `frame_rate` is a `(num, den)` rational on the ABR side; convert to `f64`
/// fps.  Audio and segment metadata are carried over so segment-duration and
/// audio checks see realistic values.
fn rung_to_ladder_rung(rung: &AbrRung, segment_secs: f64) -> LadderRung {
    let (num, den) = rung.frame_rate;
    let fps = f64::from(num) / f64::from(den);
    LadderRung::new(
        rung.width,
        rung.height,
        rung.video_bitrate,
        fps,
        &rung.codec,
    )
    .with_audio(rung.audio_bitrate)
    .with_segment_duration(segment_secs)
}

/// Builds an `EncodeLadder` from an `AbrLadder`, sorted highest-quality-first
/// as the validator expects.
fn encode_ladder_from(ladder: &AbrLadder, segment_secs: f64) -> EncodeLadder {
    let rungs: Vec<LadderRung> = ladder
        .rungs
        .iter()
        .map(|r| rung_to_ladder_rung(r, segment_secs))
        .collect();
    let mut el = EncodeLadder::new(rungs);
    el.sort_descending();
    el
}

// ── Test 6: HLS standard ladder validates clean against LadderSpec::Hls ────────

/// The shipped `AbrLadder::hls_standard()` (5 rungs, H.264) — once mapped and
/// sorted descending — must pass `LadderSpec::Hls` validation with zero errors.
#[test]
fn test_hls_standard_ladder_passes_hls_spec() {
    let abr = AbrLadder::hls_standard();
    assert_eq!(abr.rung_count(), 5, "HLS standard ladder has 5 rungs");

    // HLS segment durations live in [2.0, 10.0]; pick a compliant 4.0 s.
    let ladder = encode_ladder_from(&abr, 4.0);
    let report = LadderValidator::new(LadderSpec::Hls).validate(&ladder);

    assert!(
        report.is_ok(),
        "HLS standard ladder must pass HLS spec; errors: {:?}",
        report.errors()
    );
    assert_eq!(report.error_count(), 0, "expected zero HLS spec errors");
}

// ── Test 7: aggressive ladder validates against DASH and contains a UHD rung ───

/// `AbrLadder::aggressive()` (8 rungs incl. 2160p) must pass `LadderSpec::Dash`
/// and contain a true UHD rung (`height >= 2160 && video_bitrate >= 15 Mbps`).
#[test]
fn test_aggressive_ladder_passes_dash_and_has_uhd_rung() {
    let abr = AbrLadder::aggressive();
    assert_eq!(abr.rung_count(), 8, "aggressive ladder has 8 rungs");

    // The aggressive preset must contain a real UHD rung.
    let has_uhd = abr
        .rungs
        .iter()
        .any(|r| r.height >= 2160 && r.video_bitrate >= 15_000_000);
    assert!(
        has_uhd,
        "aggressive ladder must contain a UHD rung (≥2160p, ≥15 Mbps)"
    );

    // DASH segment durations live in [1.0, 10.0]; pick a compliant 2.0 s.
    let ladder = encode_ladder_from(&abr, 2.0);
    let report = LadderValidator::new(LadderSpec::Dash).validate(&ladder);

    assert!(
        report.is_ok(),
        "aggressive ladder must pass DASH spec; errors: {:?}",
        report.errors()
    );
}

// ── Test 8: an ascending-order ladder is rejected by the validator ─────────────

/// The validator expects rungs highest-quality-first (descending bitrate).
///
/// Here we feed an *ascending-bitrate* ladder where each taller-resolution
/// rung sits at a lower index than a shorter rung yet carries a smaller
/// bitrate.  The validator walks adjacent `(upper, lower)` pairs and, finding
/// `upper.pixels() > lower.pixels()` together with `upper.bitrate < lower.bitrate`,
/// raises a bitrate-crossover error.  So `is_ok()` must be false.
#[test]
fn test_ascending_order_ladder_is_rejected() {
    // Ascending by bitrate (1.0M → 2.5M → 4.0M) but descending by resolution,
    // which is exactly the mis-ordering the validator must reject:
    //   i=1: upper=1080p(2.07M px, 1.0M bps) vs lower=720p(0.92M px, 2.5M bps)
    //        → larger resolution but lower bitrate ⇒ crossover.
    //   i=2: upper=720p(0.92M px, 2.5M bps)  vs lower=480p(0.41M px, 4.0M bps)
    //        → crossover again.
    let ladder = EncodeLadder::new(vec![
        LadderRung::new(1920, 1080, 1_000_000, 30.0, "h264"),
        LadderRung::new(1280, 720, 2_500_000, 30.0, "h264"),
        LadderRung::new(854, 480, 4_000_000, 30.0, "h264"),
    ]);

    let report = LadderValidator::new(LadderSpec::Hls).validate(&ladder);
    assert!(
        !report.is_ok(),
        "ascending-bitrate / descending-resolution ladder must be rejected; \
         findings: {:?}",
        report.findings
    );
    assert!(
        report.error_count() >= 1,
        "expected at least one crossover error, got {}",
        report.error_count()
    );
    // Confirm the rejection is specifically a crossover, not something else.
    let crossover = report
        .errors()
        .iter()
        .any(|f| f.message.contains("crossover"));
    assert!(
        crossover,
        "rejection must be a bitrate crossover; errors: {:?}",
        report
            .errors()
            .iter()
            .map(|f| &f.message)
            .collect::<Vec<_>>()
    );
}
