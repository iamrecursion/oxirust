//! Integration tests for measure_true_peak() in the metering module.
//!
//! Scenario: 0 dBFS sine wave at 1 kHz with a 0.5-sample phase offset.
//! True-peak measurement (via ITU-R BS.1770-4 4× upsampling) must report
//! a true peak within ±1.0 dBTP of 0 dBTP for a full-scale sine.

use std::f32::consts::PI;

use oximedia_audiopost::metering::{analyze_spectrum, measure_lufs, measure_true_peak};

/// Generate a 0 dBFS mono sine at `freq_hz` with a half-sample phase offset.
fn make_fullscale_sine_half_offset(n: usize, freq_hz: f32, sample_rate: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            // Half-sample phase offset to exercise true-peak detection between samples.
            (2.0 * PI * freq_hz * (i as f32 + 0.5) / sample_rate as f32).sin()
        })
        .collect()
}

#[test]
fn test_true_peak_fullscale_sine_near_zero_dbtp() {
    let sample_rate = 48_000_u32;
    let n = sample_rate as usize; // 1 second
    let signal = make_fullscale_sine_half_offset(n, 1_000.0, sample_rate);

    let result = measure_true_peak(&signal, sample_rate).expect("measure_true_peak should succeed");

    eprintln!(
        "DIAG true_peak: linear_peak={:.6}, true_peak_dbtp={:.6} dBTP",
        result.linear_peak, result.true_peak_dbtp
    );

    // The linear peak of a full-scale sine with half-sample offset should be ≤ 1.0
    // (exact 1.0 is not guaranteed by sample positions, but true-peak via upsampling
    // will find the inter-sample peak, which for a 1 kHz / 48 kHz sine is very close to 1.0).
    assert!(
        result.linear_peak >= 0.9 && result.linear_peak <= 1.0,
        "linear_peak={} not in [0.9, 1.0]",
        result.linear_peak
    );

    // True peak in dBTP must be within [-1.0, +1.0] dBTP relative to 0 dBFS.
    assert!(
        result.true_peak_dbtp >= -1.0 && result.true_peak_dbtp <= 1.0,
        "true_peak_dbtp={:.3} dBTP not within [-1.0, +1.0] dBTP",
        result.true_peak_dbtp
    );
}

#[test]
fn test_true_peak_silence_is_minus_infinity_or_very_low() {
    let sample_rate = 48_000_u32;
    let silent = vec![0.0_f32; 4_096];
    let result = measure_true_peak(&silent, sample_rate).expect("silent signal ok");

    // Silent signal should yield a very low or -inf true peak.
    assert!(
        result.true_peak_dbtp < -60.0,
        "silent signal true peak should be < -60 dBTP, got {:.2}",
        result.true_peak_dbtp
    );
    assert_eq!(result.linear_peak, 0.0);
}

#[test]
fn test_true_peak_result_is_finite() {
    // Any non-empty non-silent signal must produce finite metering values.
    let sample_rate = 44_100_u32;
    let n = 4_096_usize;
    let signal: Vec<f32> = (0..n)
        .map(|i| (2.0 * PI * 440.0 * i as f32 / sample_rate as f32).sin() * 0.5)
        .collect();

    let result = measure_true_peak(&signal, sample_rate).expect("measure_true_peak ok");
    assert!(
        result.true_peak_dbtp.is_finite(),
        "true_peak_dbtp must be finite"
    );
    assert!(result.linear_peak.is_finite(), "linear_peak must be finite");
    assert!(
        result.linear_peak >= 0.0,
        "linear_peak must be non-negative"
    );
}

#[test]
fn test_true_peak_empty_input_error() {
    let result = measure_true_peak(&[], 48_000);
    assert!(result.is_err(), "empty input must return an error");
}

#[test]
fn test_true_peak_zero_sample_rate_error() {
    let signal = vec![0.5_f32; 100];
    let result = measure_true_peak(&signal, 0);
    assert!(result.is_err(), "zero sample rate must return an error");
}

#[test]
fn test_analyze_spectrum_returns_positive_freq_bins() {
    let sample_rate = 48_000_u32;
    // 1024-sample FFT: should return 1024/2+1 = 513 bins.
    let signal: Vec<f32> = (0..2048)
        .map(|i| (2.0 * PI * 1_000.0 * i as f32 / sample_rate as f32).sin() * 0.5)
        .collect();
    let analysis = analyze_spectrum(&signal, sample_rate).expect("analyze_spectrum ok");
    assert_eq!(analysis.frequencies.len(), 513);
    assert_eq!(analysis.magnitudes.len(), 513);
    assert_eq!(analysis.power.len(), 513);
    // DC frequency must be 0 Hz; Nyquist must be sample_rate/2.
    assert!((analysis.frequencies[0] - 0.0).abs() < 1.0);
    assert!((analysis.frequencies[512] - 24_000.0).abs() < 100.0);
}

#[test]
fn test_measure_lufs_returns_finite() {
    let sample_rate = 48_000_u32;
    let n = sample_rate as usize; // 1 second
    let signal: Vec<f32> = (0..n)
        .map(|i| (2.0 * PI * 440.0 * i as f32 / sample_rate as f32).sin() * 0.3)
        .collect();
    let lufs = measure_lufs(&signal, sample_rate).expect("measure_lufs ok");
    // Result is either a finite LUFS or -infinity (gated-out).
    assert!(lufs.is_finite() || lufs == f64::NEG_INFINITY || lufs < 0.0);
}
