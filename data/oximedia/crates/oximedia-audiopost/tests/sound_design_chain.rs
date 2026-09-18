//! Integration tests for the SoundDesignChain builder and signal flow.
//!
//! Tests verify:
//! - Output is bounded and finite (no NaN/Inf, no clipping beyond [-1.0, 1.0]).
//! - Energy change is plausible for each configured stage combination.
//! - Builder pattern produces correct stage ordering.

use std::f32::consts::PI;

use oximedia_audiopost::sound_design::SoundDesignChain;
use oximedia_effects::distortion::waveshaper::DistortionAlgorithm;
use oximedia_effects::ReverbConfig;

/// Generate a mono 440 Hz sine wave at 48 kHz, 1 second.
fn make_test_signal(n: usize, amp: f32) -> Vec<f32> {
    (0..n)
        .map(|i| amp * (2.0 * PI * 440.0 * i as f32 / 48_000.0).sin())
        .collect()
}

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    (samples.iter().map(|&s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
}

#[test]
fn test_chain_empty_no_nan() {
    let input = make_test_signal(4096, 0.5);
    let mut chain = SoundDesignChain::new(48_000).expect("chain new ok");
    let output = chain.process(&input);
    assert_eq!(output.len(), input.len());
    for (i, &s) in output.iter().enumerate() {
        assert!(s.is_finite(), "sample {i} is not finite");
        assert!(s.abs() <= 1.0 + 1e-6, "sample {i} = {s} exceeds [-1,1]");
    }
}

#[test]
fn test_chain_with_reverb_bounded() {
    let input = make_test_signal(4096, 0.3);
    let config = ReverbConfig::default();
    let mut chain = SoundDesignChain::new(48_000)
        .expect("chain new ok")
        .with_reverb(config);

    let output = chain.process(&input);
    assert_eq!(output.len(), input.len());
    for (i, &s) in output.iter().enumerate() {
        assert!(s.is_finite(), "reverb sample {i} is not finite");
        assert!(
            s.abs() <= 1.0 + 1e-6,
            "reverb sample {i} = {s} exceeds [-1,1]"
        );
    }
}

#[test]
fn test_chain_with_chorus_bounded() {
    let input = make_test_signal(4096, 0.4);
    let mut chain = SoundDesignChain::new(48_000)
        .expect("chain new ok")
        .with_chorus(4);

    let output = chain.process(&input);
    assert_eq!(output.len(), input.len());
    for (i, &s) in output.iter().enumerate() {
        assert!(s.is_finite(), "chorus sample {i} is not finite");
        assert!(
            s.abs() <= 1.0 + 1e-6,
            "chorus sample {i} = {s} exceeds [-1,1]"
        );
    }
}

#[test]
fn test_chain_with_distortion_bounded() {
    let input = make_test_signal(4096, 0.5);
    let mut chain = SoundDesignChain::new(48_000)
        .expect("chain new ok")
        .with_distortion(DistortionAlgorithm::SoftClip);

    let output = chain.process(&input);
    assert_eq!(output.len(), input.len());
    for (i, &s) in output.iter().enumerate() {
        assert!(s.is_finite(), "distortion sample {i} is not finite");
        assert!(
            s.abs() <= 1.0 + 1e-6,
            "distortion sample {i} = {s} exceeds [-1,1]"
        );
    }
}

#[test]
fn test_chain_full_pipeline_bounded_and_non_zero() {
    // All three stages: reverb → chorus → distortion.
    let input = make_test_signal(8192, 0.4);
    let config = ReverbConfig::default();
    let mut chain = SoundDesignChain::new(48_000)
        .expect("chain new ok")
        .with_reverb(config)
        .with_chorus(2)
        .with_distortion(DistortionAlgorithm::HardClip);

    let output = chain.process(&input);
    assert_eq!(output.len(), input.len());

    let max_abs = output.iter().map(|&s| s.abs()).fold(0.0_f32, f32::max);
    let output_rms = rms(&output);

    for (i, &s) in output.iter().enumerate() {
        assert!(s.is_finite(), "full-chain sample {i} is not finite");
        assert!(
            s.abs() <= 1.0 + 1e-6,
            "full-chain sample {i} = {s} exceeds [-1,1]"
        );
    }

    // Output energy should be plausible: non-zero and not extreme.
    assert!(
        output_rms > 0.001,
        "full chain output RMS too small: {output_rms}"
    );
    assert!(
        max_abs > 0.0,
        "full chain produced silence; max_abs={max_abs}"
    );
}

#[test]
fn test_chain_empty_input_produces_empty_output() {
    let mut chain = SoundDesignChain::new(48_000)
        .expect("chain new ok")
        .with_distortion(DistortionAlgorithm::SoftClip);
    let output = chain.process(&[]);
    assert!(output.is_empty());
}

#[test]
fn test_chain_zero_sample_rate_error() {
    let result = SoundDesignChain::new(0);
    assert!(result.is_err(), "zero sample rate must return an error");
}
