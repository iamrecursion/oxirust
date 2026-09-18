//! Integration tests for the free-function metering helpers.

use oximedia_audiopost::metering::{analyze_spectrum, measure_lufs, measure_true_peak};
use std::f32::consts::PI;

fn sine_wave(freq_hz: f32, amplitude: f32, num_samples: usize, sample_rate: u32) -> Vec<f32> {
    let sr = sample_rate as f32;
    (0..num_samples)
        .map(|i| amplitude * (2.0 * PI * freq_hz * i as f32 / sr).sin())
        .collect()
}

// ── analyze_spectrum ──────────────────────────────────────────────────────────

#[test]
fn test_analyze_spectrum_basic() {
    let sample_rate = 48_000_u32;
    let samples = sine_wave(1000.0, 0.5, 4096, sample_rate);
    let result = analyze_spectrum(&samples, sample_rate).expect("analyze_spectrum should succeed");

    assert_eq!(result.frequencies.len(), 513); // FFT_SIZE/2 + 1 = 513
    assert_eq!(result.magnitudes.len(), 513);
    assert_eq!(result.power.len(), 513);

    // All values should be finite and non-negative.
    assert!(result.magnitudes.iter().all(|&m| m.is_finite() && m >= 0.0));
    assert!(result.power.iter().all(|&p| p.is_finite() && p >= 0.0));
}

#[test]
fn test_analyze_spectrum_peak_near_1khz() {
    let sample_rate = 48_000_u32;
    let samples = sine_wave(1000.0, 1.0, 4096, sample_rate);
    let result = analyze_spectrum(&samples, sample_rate).expect("analyze_spectrum should succeed");

    // Find peak frequency bin.
    let (peak_idx, _) = result
        .magnitudes
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .expect("magnitudes must be non-empty");

    let peak_freq = result.frequencies[peak_idx];
    // Allow ±50 Hz tolerance given the 1024-pt window.
    assert!(
        (peak_freq - 1000.0).abs() < 50.0,
        "peak frequency = {peak_freq:.1} Hz, expected ~1000 Hz"
    );
}

#[test]
fn test_analyze_spectrum_empty_error() {
    assert!(analyze_spectrum(&[], 48000).is_err());
}

#[test]
fn test_analyze_spectrum_zero_sample_rate_error() {
    let samples = vec![0.5_f32; 1024];
    assert!(analyze_spectrum(&samples, 0).is_err());
}

// ── measure_lufs ─────────────────────────────────────────────────────────────

#[test]
fn test_measure_lufs_returns_finite() {
    let sample_rate = 48_000_u32;
    let samples = sine_wave(1000.0, 0.5, sample_rate as usize, sample_rate);
    let lufs = measure_lufs(&samples, sample_rate).expect("measure_lufs should succeed");
    assert!(lufs.is_finite(), "LUFS must be finite");
    assert!(
        lufs < 0.0,
        "LUFS must be negative for a sub-full-scale signal"
    );
}

#[test]
fn test_measure_lufs_empty_error() {
    assert!(measure_lufs(&[], 48000).is_err());
}

#[test]
fn test_measure_lufs_zero_sample_rate_error() {
    let samples = vec![0.5_f32; 1024];
    assert!(measure_lufs(&samples, 0).is_err());
}

// ── measure_true_peak ─────────────────────────────────────────────────────────

#[test]
fn test_measure_true_peak_basic() {
    let sample_rate = 48_000_u32;
    let samples = sine_wave(1000.0, 0.9, 4096, sample_rate);
    let result =
        measure_true_peak(&samples, sample_rate).expect("measure_true_peak should succeed");

    assert!(result.true_peak_dbtp.is_finite());
    assert!(result.linear_peak > 0.0);
    assert!(result.linear_peak <= 1.0);
}

#[test]
fn test_measure_true_peak_empty_error() {
    assert!(measure_true_peak(&[], 48000).is_err());
}

#[test]
fn test_measure_true_peak_zero_sample_rate_error() {
    let samples = vec![0.5_f32; 1024];
    assert!(measure_true_peak(&samples, 0).is_err());
}

#[test]
fn test_measure_true_peak_full_scale() {
    // 0 dBFS signal: true peak should be close to 0 dBTP.
    let sample_rate = 48_000_u32;
    let samples = sine_wave(1000.0, 1.0, 4096, sample_rate);
    let result =
        measure_true_peak(&samples, sample_rate).expect("measure_true_peak should succeed");
    // True-peak can be slightly above 0 dBTP due to inter-sample peaks —
    // but linear peak of a 1.0-amplitude sine is 1.0.
    assert!(
        result.linear_peak > 0.99,
        "linear_peak = {:.4}, expected > 0.99",
        result.linear_peak
    );
}
