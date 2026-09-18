//! Smoke tests verifying newly wired metering orphan modules compile and expose
//! at least one public item from each module.

use oximedia_metering::{
    bs2051_weights::{compute_integrated_loudness_bs2051, Bs2051ChannelGroup, Bs2051Weights},
    bs2132::{compute_bs2132_loudness, Bs2132Config},
    clip_counter::{ClipCounter, ClipThreshold},
    leq::{LeqMeter, LeqWeighting, TimeWeighting},
    ms_ssim::ms_ssim,
    rms_envelope::{RmsEnvelopeConfig, RmsEnvelopeFollower},
    silence_detect::{SilenceDetectConfig, SilenceDetector},
    true_peak_meter::TruePeakMeter,
    vmaf_estimate::estimate_vmaf,
    vmaf_features::VmafExtractor,
};

#[test]
fn test_bs2051_weights_channel_groups() {
    let weights = Bs2051Weights {
        channel_groups: vec![Bs2051ChannelGroup::MidLayer, Bs2051ChannelGroup::MidLayer],
    };
    assert_eq!(weights.channel_groups.len(), 2);
}

#[test]
fn test_bs2051_integrated_loudness_silence() {
    let layout = Bs2051Weights {
        channel_groups: vec![Bs2051ChannelGroup::MidLayer, Bs2051ChannelGroup::MidLayer],
    };
    let silence = vec![vec![0.0f32; 100], vec![0.0f32; 100]];
    let loudness = compute_integrated_loudness_bs2051(&silence, &layout);
    // Silence → minus-infinity or a very small finite value — either is acceptable.
    assert!(loudness.is_finite() || loudness.is_infinite());
}

#[test]
fn test_bs2132_config_and_loudness() {
    // compute_bs2132_loudness takes (&[f64], &Bs2132Config) and returns MeteringResult
    let config = Bs2132Config::new(48000.0, 2, 0.0);
    // Empty input should not panic — result is Ok or Err, both acceptable.
    let result = compute_bs2132_loudness(&[], &config);
    // Just verify no panic occurred.
    drop(result);
}

#[test]
fn test_clip_counter_no_clips_on_silence() {
    let mut counter = ClipCounter::new(2, 48000.0, ClipThreshold::DigitalFullScale, false);
    let silence: Vec<f64> = vec![0.0; 1024];
    counter.process_interleaved(&silence);
    assert_eq!(counter.total_clip_events(), 0);
}

#[test]
fn test_clip_counter_detects_full_scale() {
    let mut counter = ClipCounter::new(1, 48000.0, ClipThreshold::DigitalFullScale, false);
    // Alternating +1 / -1 samples will clip.
    let samples: Vec<f64> = (0..100)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    counter.process_interleaved(&samples);
    assert!(counter.total_clip_events() > 0);
}

#[test]
fn test_leq_meter_new_and_process() {
    // LeqMeter::new returns MeteringResult<Self>
    let mut meter = LeqMeter::new(48000.0, 1, LeqWeighting::Flat, TimeWeighting::Fast)
        .expect("LeqMeter::new should succeed");
    let silence: Vec<f64> = vec![0.0; 1024];
    meter.process_interleaved(&silence);
    let leq = meter.leq_max();
    assert!(leq.is_finite() || leq.is_infinite());
}

#[test]
fn test_ms_ssim_identical_frames() {
    // Identical solid-grey frames should yield SSIM ≈ 1.0.
    let frame = vec![128u8; 64 * 64];
    let score = ms_ssim(&frame, &frame, 64, 64);
    assert!(
        (score - 1.0).abs() < 0.05,
        "MS-SSIM of identical frames should be ~1.0, got {score}"
    );
}

#[test]
fn test_rms_envelope_follower_silence() {
    let config = RmsEnvelopeConfig::new(48000.0, 1);
    let mut follower = RmsEnvelopeFollower::new(config);
    let silence: Vec<f64> = vec![0.0; 512];
    follower.process_interleaved(&silence);
    // Silent input → envelope is zero.
    assert!((follower.rms_linear(0) - 0.0).abs() < 1e-9);
}

#[test]
fn test_silence_detector_detects_silence() {
    // Use a very short min_duration so silence is detected quickly.
    let config = SilenceDetectConfig::new(48000.0, 1)
        .with_threshold(-60.0)
        .with_min_duration(0.001); // 1 ms minimum silence
    let mut detector = SilenceDetector::new(config);
    // 200 ms of silence followed by a loud sample to trigger event emission.
    let mut samples: Vec<f64> = vec![0.0; 9600];
    samples.push(1.0); // non-silent sample ends the silence region and fires the event
    detector.process_interleaved(&samples);
    let events = detector.events();
    assert!(
        !events.is_empty(),
        "Should detect at least one silence event after silence region ends"
    );
}

#[test]
fn test_true_peak_meter_silence() {
    let meter = TruePeakMeter::new();
    let silence: Vec<f32> = vec![0.0; 512];
    let peak = meter.measure(&silence);
    assert!(
        (peak - 0.0).abs() < 1e-9,
        "True peak of silence should be 0.0, got {peak}"
    );
}

#[test]
fn test_estimate_vmaf_identical_frames() {
    // Identical frames → VMAF should be at or near 100.
    let frame = vec![128u8; 64 * 64];
    let score = estimate_vmaf(&frame, &frame, 64, 64);
    assert!(
        score >= 95.0,
        "VMAF of identical frames should be ≥95, got {score}"
    );
}

#[test]
fn test_vmaf_extractor_new() {
    // VmafExtractor requires at least 128×128 for 4-scale analysis.
    let extractor =
        VmafExtractor::new(128, 128).expect("VmafExtractor::new(128,128) should succeed");
    let _ = extractor;
}
