//! Integration tests for the declick() restoration function.
//!
//! Scenario: 1 kHz sine wave with 5 impulsive clicks injected.
//! After declicking, the Pearson correlation with the original sine must exceed 0.99.

use std::f32::consts::PI;

use oximedia_audiopost::restoration::{declick, DeclickConfig};

/// Generate a mono 1 kHz sine wave at `sample_rate` of length `n`.
fn make_sine(n: usize, freq_hz: f32, amp: f32, sample_rate: u32) -> Vec<f32> {
    (0..n)
        .map(|i| amp * (2.0 * PI * freq_hz * i as f32 / sample_rate as f32).sin())
        .collect()
}

/// Pearson correlation coefficient between two equal-length slices.
fn pearson(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let n = a.len() as f64;
    let mean_a = a.iter().map(|&x| x as f64).sum::<f64>() / n;
    let mean_b = b.iter().map(|&x| x as f64).sum::<f64>() / n;
    let num: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(xx, yy)| ((*xx) as f64 - mean_a) * ((*yy) as f64 - mean_b))
        .sum();
    let den_a: f64 = a
        .iter()
        .map(|&x| (x as f64 - mean_a).powi(2))
        .sum::<f64>()
        .sqrt();
    let den_b: f64 = b
        .iter()
        .map(|&x| (x as f64 - mean_b).powi(2))
        .sum::<f64>()
        .sqrt();
    if den_a * den_b < 1e-30 {
        return 1.0;
    }
    num / (den_a * den_b)
}

#[test]
fn test_declick_correlation_above_threshold() {
    let sample_rate = 48_000_u32;
    let n = 4_096_usize;
    let original = make_sine(n, 1_000.0, 0.5, sample_rate);

    // Inject 5 clicks at positions spread across the signal.
    let mut noisy = original.clone();
    let click_positions = [200usize, 700, 1500, 2500, 3600];
    for &pos in &click_positions {
        noisy[pos] = 2.0;
    }

    let config = DeclickConfig::default();
    let repaired = declick(&noisy, &config).expect("declick should succeed");

    assert_eq!(repaired.len(), n);

    let corr = pearson(&repaired, &original);
    assert!(
        corr > 0.99,
        "correlation after declick should be >0.99 but got {corr:.4}"
    );
}

#[test]
fn test_declick_preserves_clean_signal() {
    let sample_rate = 44_100_u32;
    let n = 2_048_usize;
    let clean = make_sine(n, 440.0, 0.3, sample_rate);

    let config = DeclickConfig::default();
    let result = declick(&clean, &config).expect("declick on clean signal");

    let corr = pearson(&result, &clean);
    assert!(
        corr > 0.999,
        "declicking a clean signal must preserve it (corr={corr:.5})"
    );
}

#[test]
fn test_declick_output_bounded_and_finite() {
    let sample_rate = 48_000_u32;
    let n = 1_024_usize;
    let mut signal = make_sine(n, 1_000.0, 0.5, sample_rate);
    signal[100] = 5.0;
    signal[500] = -5.0;

    let config = DeclickConfig::default();
    let result = declick(&signal, &config).expect("declick should succeed");

    for (i, &s) in result.iter().enumerate() {
        assert!(s.is_finite(), "sample {i} is not finite: {s}");
        assert!(s.abs() < 6.0, "sample {i} has amplitude {s} > 6.0");
    }
}

#[test]
fn test_declick_empty_input_error() {
    let config = DeclickConfig::default();
    assert!(declick(&[], &config).is_err());
}

#[test]
fn test_declick_single_sample_passthrough() {
    let config = DeclickConfig::default();
    let result = declick(&[0.5_f32], &config).expect("single sample ok");
    assert_eq!(result.len(), 1);
    assert!((result[0] - 0.5_f32).abs() < 1e-6_f32);
}
