//! Integration tests for spectral_subtract() (Boll 1979 spectral subtraction + Wiener).
//!
//! Scenario: noise-only lead-in section (1024 samples) lets the estimator
//! calibrate the noise PSD.  The remaining samples contain a 1 kHz tone mixed
//! with the same noise.  After spectral subtraction the tone-region SNR must
//! improve by at least 3 dB relative to the input SNR.

use std::f32::consts::PI;

use oximedia_audiopost::restoration::{spectral_subtract, SpectralSubtractionConfig};

/// Deterministic pseudo-white noise via a hash-LCG with amplitude `amp`.
fn make_pseudo_noise(n: usize, amp: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = (i as f32 * 12.989_8 + 78.233).sin() * 43_758.545_3;
            (x - x.floor() - 0.5) * 2.0 * amp
        })
        .collect()
}

/// Compute signal-to-noise ratio in dB: 10 * log10(signal_power / noise_power).
/// Returns `None` when noise power is zero.
fn snr_db(signal: &[f32], noise: &[f32]) -> Option<f32> {
    assert_eq!(signal.len(), noise.len());
    let n = signal.len() as f32;
    let sig_power = signal.iter().map(|&s| s * s).sum::<f32>() / n;
    let nse_power = noise.iter().map(|&s| s * s).sum::<f32>() / n;
    if nse_power < 1e-30 {
        return None;
    }
    Some(10.0 * (sig_power / nse_power).log10())
}

#[test]
fn test_spectral_subtract_snr_improves() {
    let sample_rate = 48_000_u32;
    let tone_region_len = 4_096_usize;
    let lead_in_len = 1_024_usize; // noise-only; lets the estimator see pure noise
    let n = lead_in_len + tone_region_len;

    let noise = make_pseudo_noise(n, 0.3);
    let tone_freq = 1_000.0_f32;
    let tone_amp = 0.5_f32;

    // Build the mixed signal: noise only for lead-in, then tone+noise.
    let mixed: Vec<f32> = (0..n)
        .map(|i| {
            let tone = if i >= lead_in_len {
                tone_amp
                    * (2.0 * PI * tone_freq * (i - lead_in_len) as f32 / sample_rate as f32).sin()
            } else {
                0.0
            };
            tone + noise[i]
        })
        .collect();

    // Build the pure tone reference for the tone region only.
    let tone_ref: Vec<f32> = (0..tone_region_len)
        .map(|i| tone_amp * (2.0 * PI * tone_freq * i as f32 / sample_rate as f32).sin())
        .collect();

    let config = SpectralSubtractionConfig::default();
    let result = spectral_subtract(&mixed, &config).expect("spectral_subtract should succeed");

    assert_eq!(result.len(), n, "output length must equal input length");

    // All output samples must be finite.
    for (i, &s) in result.iter().enumerate() {
        assert!(s.is_finite(), "sample {i} is not finite: {s}");
    }

    // Evaluate SNR only on the tone region of the output.
    let result_tone_region = &result[lead_in_len..];
    assert_eq!(result_tone_region.len(), tone_region_len);

    // Input SNR in the tone region.
    let noise_tone_region = &noise[lead_in_len..];
    let input_snr = snr_db(&tone_ref, noise_tone_region).expect("noise power must be non-zero");

    // Residual (error) after spectral subtraction.
    let residual: Vec<f32> = result_tone_region
        .iter()
        .zip(tone_ref.iter())
        .map(|(&r, &t)| r - t)
        .collect();
    let output_snr = snr_db(&tone_ref, &residual).unwrap_or(-100.0);

    let snr_improvement = output_snr - input_snr;
    eprintln!(
        "DIAG spectral_subtract: input_snr={input_snr:.2} dB, output_snr={output_snr:.2} dB, improvement={snr_improvement:.2} dB"
    );
    assert!(
        snr_improvement >= 3.0,
        "SNR must improve by ≥3 dB after spectral subtraction; \
         input_snr={input_snr:.2} dB, output_snr={output_snr:.2} dB, \
         improvement={snr_improvement:.2} dB"
    );
}

#[test]
fn test_spectral_subtract_output_finite_no_nan() {
    // Minimal smoke test: any non-empty input must produce all-finite output.
    let input: Vec<f32> = (0..2048)
        .map(|i| {
            (2.0 * PI * 440.0 * i as f32 / 44_100.0).sin() * 0.4 + (i as f32 * 0.0013).sin() * 0.2
        })
        .collect();
    let config = SpectralSubtractionConfig::default();
    let result = spectral_subtract(&input, &config).expect("spectral_subtract smoke test");
    assert_eq!(result.len(), input.len());
    for (i, &s) in result.iter().enumerate() {
        assert!(
            s.is_finite(),
            "sample {i} is NaN/Inf after spectral_subtract"
        );
    }
}

#[test]
fn test_spectral_subtract_empty_input_error() {
    let config = SpectralSubtractionConfig::default();
    assert!(spectral_subtract(&[], &config).is_err());
}

#[test]
fn test_spectral_subtract_silent_input() {
    // A silent input should produce a silent (or near-silent) output.
    let silent = vec![0.0_f32; 4096];
    let config = SpectralSubtractionConfig::default();
    let result = spectral_subtract(&silent, &config).expect("silent input ok");
    assert_eq!(result.len(), silent.len());
    let max_abs = result.iter().map(|&s| s.abs()).fold(0.0_f32, f32::max);
    assert!(
        max_abs < 1e-6,
        "silent input must produce near-zero output; max_abs={max_abs}"
    );
}
