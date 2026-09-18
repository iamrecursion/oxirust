//! Signal-to-noise ratio (SNR) computation.

use crate::compute_rms;

/// Compute signal-to-noise ratio.
///
/// # Arguments
/// * `signal` - Clean signal samples
/// * `noise` - Noise samples
///
/// # Returns
/// SNR in linear scale (signal power / noise power)
#[must_use]
pub fn signal_to_noise_ratio(signal: &[f32], noise: &[f32]) -> f32 {
    if signal.is_empty() || noise.is_empty() {
        return 0.0;
    }

    let signal_power = compute_rms(signal).powi(2);
    let noise_power = compute_rms(noise).powi(2);

    if noise_power > 0.0 {
        signal_power / noise_power
    } else {
        f32::INFINITY
    }
}

/// Compute SNR in decibels.
#[must_use]
pub fn compute_snr_db(signal: &[f32], noise: &[f32]) -> f32 {
    let snr = signal_to_noise_ratio(signal, noise);

    if snr > 0.0 && snr.is_finite() {
        10.0 * snr.log10()
    } else if snr.is_infinite() {
        100.0 // Cap at 100 dB
    } else {
        -100.0 // Floor at -100 dB
    }
}

/// Estimate SNR from a single signal by separating signal and noise.
///
/// Uses a simple approach: assumes noise is the low-amplitude portions.
#[must_use]
pub fn estimate_snr(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }

    // Sort by amplitude to separate signal from noise
    let mut sorted: Vec<f32> = samples.iter().map(|&x| x.abs()).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Bottom 25% is noise
    let noise_cutoff = sorted.len() / 4;
    let noise_samples = &sorted[..noise_cutoff];

    // Top 75% is signal + noise
    let signal_samples = &sorted[noise_cutoff..];

    signal_to_noise_ratio(signal_samples, noise_samples)
}

/// Estimate SNR in decibels.
#[must_use]
pub fn estimate_snr_db(samples: &[f32]) -> f32 {
    let snr = estimate_snr(samples);

    if snr > 0.0 && snr.is_finite() {
        10.0 * snr.log10()
    } else {
        -100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snr_computation() {
        // Clean signal
        let signal = vec![1.0; 100];
        // Low noise
        let noise = vec![0.1; 100];

        let snr = signal_to_noise_ratio(&signal, &noise);
        assert!(snr > 10.0); // Should have high SNR

        let snr_db = compute_snr_db(&signal, &noise);
        assert!(snr_db > 10.0);
    }

    #[test]
    fn test_snr_estimation() {
        // Signal with some noise
        let mut samples = vec![0.01; 100]; // Noise floor
        samples.extend(vec![1.0; 100]); // Signal

        let snr_db = estimate_snr_db(&samples);
        assert!(snr_db > 0.0);
    }

    // ── Analytical accuracy test ───────────────────────────────────────────────

    /// Generate a 1000 Hz sinusoidal signal at unit amplitude and scaled white
    /// noise at exactly -20 dB relative to the signal RMS.  Then verify that
    /// `compute_snr_db` returns ≈ 20 dB (±2 dB).
    ///
    /// The noise amplitude scaling factor: if noise RMS = 1 and signal RMS ≈ 0.707
    /// (sine), then to achieve SNR = 20 dB (power ratio = 100) we need
    /// noise_rms = signal_rms / sqrt(100) = signal_rms / 10.
    ///
    /// A deterministic pseudo-noise sequence is used instead of a real RNG so
    /// the test is reproducible without adding test dependencies.
    #[test]
    fn test_snr_known_ratio() {
        let sample_rate = 44100.0_f32;
        let signal_freq = 1000.0_f32;
        // 4096 samples gives a clean integer number of cycles near 1000 Hz.
        let n = 4096_usize;

        // Build the signal.
        let signal: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * signal_freq * i as f32 / sample_rate).sin())
            .collect();

        // Compute signal RMS.
        let signal_rms = {
            let sum_sq: f32 = signal.iter().map(|&x| x * x).sum();
            (sum_sq / n as f32).sqrt()
        };

        // Desired SNR = 20 dB → noise_rms = signal_rms / 10.
        let desired_snr_db = 20.0_f32;
        let noise_amplitude = signal_rms / 10.0_f32;

        // Deterministic pseudo-noise via a simple LCG (Lehmer/MINSTD).
        let noise: Vec<f32> = {
            let mut state: u32 = 0xDEAD_BEEF;
            (0..n)
                .map(|_| {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    // Map to [-1, 1] and scale to the desired noise amplitude.
                    let normalised = (state as f32 / u32::MAX as f32) * 2.0 - 1.0;
                    normalised * noise_amplitude * std::f32::consts::SQRT_2
                })
                .collect()
        };

        let measured_db = compute_snr_db(&signal, &noise);
        let tolerance = 2.0_f32; // ±2 dB

        assert!(
            (measured_db - desired_snr_db).abs() < tolerance,
            "SNR should be ≈ {desired_snr_db:.1} dB ± {tolerance:.1} dB, measured {measured_db:.2} dB"
        );
    }
}
