//! Integration tests for audio fingerprint matching robustness.
//!
//! Verifies that:
//! - A clean 440 Hz sine and a lightly noisy version have high similarity.
//! - A 440 Hz sine and white noise have low similarity.

use oximedia_dedup::audio::{compute_fingerprint, AudioData};
use std::f32::consts::PI;

/// Generate a pure 440 Hz sine wave with the given duration and sample rate.
fn sine_440hz(sample_rate: u32, duration_secs: f64) -> Vec<f32> {
    let n_samples = (sample_rate as f64 * duration_secs) as usize;
    (0..n_samples)
        .map(|i| (2.0 * PI * 440.0 * i as f32 / sample_rate as f32).sin())
        .collect()
}

/// Add Gaussian-like noise with amplitude `scale` to a sample buffer.
/// Uses a simple LCG PRNG to remain `no_std` compatible.
fn add_noise(samples: &[f32], scale: f32, seed: u64) -> Vec<f32> {
    let mut state = seed;
    samples
        .iter()
        .map(|&s| {
            // LCG: next pseudo-random in [0, 1)
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let rand_f32 = (state >> 33) as f32 / (u32::MAX as f32); // [0, 1)
            let noise = (rand_f32 - 0.5) * 2.0 * scale; // [-scale, scale]
            (s + noise).clamp(-1.0, 1.0)
        })
        .collect()
}

/// Generate band-limited white noise (LCG-based) in [-1, 1].
fn white_noise(n_samples: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..n_samples)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let rand_f32 = (state >> 33) as f32 / (u32::MAX as f32);
            (rand_f32 - 0.5) * 2.0 // [-1, 1]
        })
        .collect()
}

#[test]
fn test_similar_audio_high_similarity() {
    let sample_rate = 44_100u32;
    let duration = 3.0f64; // seconds

    let clean = sine_440hz(sample_rate, duration);
    // Light noise: 5% amplitude relative to the full signal range
    let noisy = add_noise(&clean, 0.05, 0xABCD_1234_5678_EF00);

    let audio_clean = AudioData {
        sample_rate,
        channels: 1,
        samples: clean,
    };
    let audio_noisy = AudioData {
        sample_rate,
        channels: 1,
        samples: noisy,
    };

    let fp_clean = compute_fingerprint(&audio_clean);
    let fp_noisy = compute_fingerprint(&audio_noisy);

    let similarity = fp_clean.similarity(&fp_noisy);
    assert!(
        similarity >= 0.75,
        "440 Hz sine vs lightly noisy sine: similarity {similarity:.4} < 0.75. \
         Fingerprint should be robust to small amounts of noise."
    );
}

#[test]
fn test_different_audio_low_similarity() {
    let sample_rate = 44_100u32;
    let duration = 3.0f64;
    let n_samples = (sample_rate as f64 * duration) as usize;

    let sine = sine_440hz(sample_rate, duration);
    let noise = white_noise(n_samples, 0xDEAD_BEEF_CAFE_1234);

    let audio_sine = AudioData {
        sample_rate,
        channels: 1,
        samples: sine,
    };
    let audio_noise = AudioData {
        sample_rate,
        channels: 1,
        samples: noise,
    };

    let fp_sine = compute_fingerprint(&audio_sine);
    let fp_noise = compute_fingerprint(&audio_noise);

    let similarity = fp_sine.similarity(&fp_noise);
    assert!(
        similarity <= 0.70,
        "440 Hz sine vs white noise: similarity {similarity:.4} >= 0.70. \
         Fingerprint should distinguish tonal signals from noise."
    );
}

#[test]
fn test_identical_audio_perfect_similarity() {
    let sample_rate = 22_050u32;
    let samples = sine_440hz(sample_rate, 2.0);

    let audio = AudioData {
        sample_rate,
        channels: 1,
        samples: samples.clone(),
    };
    let audio_copy = AudioData {
        sample_rate,
        channels: 1,
        samples,
    };

    let fp1 = compute_fingerprint(&audio);
    let fp2 = compute_fingerprint(&audio_copy);

    let similarity = fp1.similarity(&fp2);
    assert!(
        (similarity - 1.0).abs() < f64::EPSILON * 100.0,
        "identical audio streams must produce identical fingerprints (similarity={similarity:.6})"
    );
}
