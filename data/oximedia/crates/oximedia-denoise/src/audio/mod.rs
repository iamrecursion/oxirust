//! Audio denoising algorithms.
//!
//! Provides spectral subtraction, Wiener filtering, hiss removal and click
//! repair for audio signals.

use std::f32::consts::PI;

// ============================================================
// Noise Profile
// ============================================================

/// Estimated noise characteristics for a signal.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct NoiseProfile {
    /// Per-bin noise power spectrum (linear power, length = DFT size / 2 + 1).
    pub spectrum: Vec<f32>,
    /// Estimated signal-to-noise ratio in dB.
    pub estimated_snr_db: f32,
}

impl NoiseProfile {
    /// Estimate noise profile from a silence segment.
    ///
    /// The `samples` slice should contain audio that is known to be
    /// noise/silence, e.g. the first few hundred milliseconds of a recording.
    pub fn estimate_from_silence(samples: &[f32]) -> Self {
        if samples.is_empty() {
            return Self {
                spectrum: Vec::new(),
                estimated_snr_db: 0.0,
            };
        }

        // Use a simple 256-point DFT to estimate noise spectrum
        let n = 256_usize.min(samples.len());
        let window = &samples[..n];

        // Compute power spectrum via direct DFT
        let spectrum = real_dft_power_spectrum(window, n);

        // Estimate overall RMS power of noise
        let noise_power: f32 = spectrum.iter().sum::<f32>() / spectrum.len() as f32;
        // Signal power from total RMS
        let signal_power: f32 = samples.iter().map(|&s| s * s).sum::<f32>() / samples.len() as f32;

        let estimated_snr_db = if noise_power > 0.0 && signal_power > noise_power {
            10.0 * (signal_power / noise_power).log10()
        } else {
            0.0
        };

        Self {
            spectrum,
            estimated_snr_db,
        }
    }
}

// ============================================================
// Spectral Subtraction
// ============================================================

/// Spectral subtraction denoiser.
///
/// Estimates per-frequency noise power from a `NoiseProfile` and subtracts
/// it from the signal spectrum, then reconstructs via inverse DFT.
pub struct SpectralSubtraction;

impl SpectralSubtraction {
    /// Create a new `SpectralSubtraction` denoiser.
    pub fn new() -> Self {
        Self
    }

    /// Denoise audio using spectral subtraction.
    ///
    /// Processes the signal in non-overlapping blocks using a direct DFT.
    pub fn denoise(&self, samples: &[f32], profile: &NoiseProfile, _sample_rate: u32) -> Vec<f32> {
        if samples.is_empty() || profile.spectrum.is_empty() {
            return samples.to_vec();
        }

        let block_size = (profile.spectrum.len().saturating_sub(1)) * 2;
        let block_size = if block_size < 2 { 256 } else { block_size };
        let mut output = vec![0.0f32; samples.len()];

        let mut pos = 0;
        while pos < samples.len() {
            let end = (pos + block_size).min(samples.len());
            let block = &samples[pos..end];

            // Pad to block_size if needed
            let mut padded = vec![0.0f32; block_size];
            padded[..block.len()].copy_from_slice(block);

            // Forward DFT: get real and imaginary parts
            let (re, im) = real_dft(&padded, block_size);

            // Spectral subtraction: reduce magnitude where noise exceeds signal
            let mut re_out = vec![0.0f32; re.len()];
            let mut im_out = vec![0.0f32; im.len()];
            for k in 0..re.len() {
                let power = re[k] * re[k] + im[k] * im[k];
                let noise_power = if k < profile.spectrum.len() {
                    profile.spectrum[k]
                } else {
                    *profile.spectrum.last().unwrap_or(&0.0)
                };

                // Subtract noise power, keep phase, enforce non-negative
                let clean_power = (power - noise_power).max(power * 0.01);
                let scale = if power > 0.0 {
                    (clean_power / power).sqrt()
                } else {
                    0.0
                };
                re_out[k] = re[k] * scale;
                im_out[k] = im[k] * scale;
            }

            // Inverse DFT
            let reconstructed = real_idft(&re_out, &im_out, block_size);
            let copy_len = block.len();
            output[pos..pos + copy_len].copy_from_slice(&reconstructed[..copy_len]);
            pos += block_size;
        }

        output
    }
}

impl Default for SpectralSubtraction {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// Wiener Filter
// ============================================================

/// Wiener filter gain computation.
pub struct WienerFilter;

impl WienerFilter {
    /// Compute Wiener filter gain: signal / (signal + noise).
    ///
    /// All powers are in linear (not dB) scale.
    pub fn compute_gain(signal_power: f32, noise_power: f32) -> f32 {
        let total = signal_power + noise_power;
        if total <= 0.0 {
            return 0.0;
        }
        signal_power / total
    }
}

/// Wiener denoiser.
pub struct WienerDenoiser;

impl WienerDenoiser {
    /// Create a new `WienerDenoiser`.
    pub fn new() -> Self {
        Self
    }

    /// Denoise using a Wiener filter with a fixed noise floor estimate.
    ///
    /// `noise_floor_db` is the estimated noise level in dBFS (e.g., -60.0).
    pub fn denoise(&self, samples: &[f32], noise_floor_db: f32) -> Vec<f32> {
        if samples.is_empty() {
            return Vec::new();
        }

        let noise_power_linear = 10.0_f32.powf(noise_floor_db / 10.0);
        let block_size = 256_usize;
        let mut output = vec![0.0f32; samples.len()];

        let mut pos = 0;
        while pos < samples.len() {
            let end = (pos + block_size).min(samples.len());
            let block = &samples[pos..end];
            let mut padded = vec![0.0f32; block_size];
            padded[..block.len()].copy_from_slice(block);

            let (re, im) = real_dft(&padded, block_size);

            let mut re_out = vec![0.0f32; re.len()];
            let mut im_out = vec![0.0f32; im.len()];
            for k in 0..re.len() {
                let signal_power = re[k] * re[k] + im[k] * im[k];
                let gain = WienerFilter::compute_gain(signal_power, noise_power_linear);
                re_out[k] = re[k] * gain;
                im_out[k] = im[k] * gain;
            }

            let reconstructed = real_idft(&re_out, &im_out, block_size);
            let copy_len = block.len();
            output[pos..pos + copy_len].copy_from_slice(&reconstructed[..copy_len]);
            pos += block_size;
        }

        output
    }
}

impl Default for WienerDenoiser {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// Hiss Remover
// ============================================================

/// High-frequency hiss detector and remover.
pub struct HissRemover;

impl HissRemover {
    /// Create a new `HissRemover`.
    pub fn new() -> Self {
        Self
    }

    /// Detect hiss as the ratio of high-frequency energy to total energy.
    ///
    /// Returns a value in [0, 1]; values near 1 indicate strong hiss.
    pub fn detect_hiss(&self, samples: &[f32], sample_rate: u32) -> f32 {
        if samples.is_empty() || sample_rate == 0 {
            return 0.0;
        }

        let n = 512_usize.min(samples.len());
        let block = &samples[..n];
        let (re, im) = real_dft(block, n);

        let total_bins = re.len();
        // High-frequency bins: above nyquist/4 (i.e., above sr/8)
        let hf_start = total_bins / 4;

        let total_energy: f32 = re
            .iter()
            .zip(im.iter())
            .map(|(r, i)| r * r + i * i)
            .sum::<f32>();

        let hf_energy: f32 = re[hf_start..]
            .iter()
            .zip(im[hf_start..].iter())
            .map(|(r, i)| r * r + i * i)
            .sum::<f32>();

        if total_energy <= 0.0 {
            0.0
        } else {
            hf_energy / total_energy
        }
    }
}

impl Default for HissRemover {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// Click Remover
// ============================================================

/// Click detector and repairer.
pub struct ClickRemover;

impl ClickRemover {
    /// Create a new `ClickRemover`.
    pub fn new() -> Self {
        Self
    }

    /// Detect click positions as statistical outliers.
    ///
    /// Returns indices of samples that deviate more than `threshold_sigma`
    /// standard deviations from the local mean.
    pub fn detect_clicks(&self, samples: &[f32], threshold_sigma: f32) -> Vec<usize> {
        if samples.len() < 3 {
            return Vec::new();
        }

        // Compute global mean and standard deviation
        let mean: f32 = samples.iter().sum::<f32>() / samples.len() as f32;
        let variance: f32 = samples
            .iter()
            .map(|&s| (s - mean) * (s - mean))
            .sum::<f32>()
            / samples.len() as f32;
        let std_dev = variance.sqrt();

        if std_dev <= 0.0 {
            return Vec::new();
        }

        samples
            .iter()
            .enumerate()
            .filter(|&(_, &s)| ((s - mean) / std_dev).abs() > threshold_sigma)
            .map(|(i, _)| i)
            .collect()
    }

    /// Repair clicks using linear interpolation from neighboring samples.
    ///
    /// `window` is the number of samples on each side to use for interpolation context.
    pub fn repair_clicks(samples: &mut Vec<f32>, click_indices: &[usize], window: usize) {
        if click_indices.is_empty() || samples.is_empty() {
            return;
        }

        let len = samples.len();
        for &idx in click_indices {
            if idx >= len {
                continue;
            }

            // Find nearest clean samples outside the window
            let left = idx.saturating_sub(window);
            let right = (idx + window + 1).min(len - 1);

            let left_val = samples[left];
            let right_val = samples[right];

            // Linear interpolation across the click region
            let span = (right - left) as f32;
            if span > 0.0 {
                let t = (idx - left) as f32 / span;
                samples[idx] = left_val + t * (right_val - left_val);
            }
        }
    }
}

impl Default for ClickRemover {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// DFT Helpers (no external FFT libs)
// ============================================================

/// Compute the forward real DFT of `samples`, returning (Re, Im) for bins 0..=N/2.
fn real_dft(samples: &[f32], n: usize) -> (Vec<f32>, Vec<f32>) {
    let actual_n = samples.len().min(n);
    let bins = n / 2 + 1;
    let mut re = vec![0.0f32; bins];
    let mut im = vec![0.0f32; bins];

    for k in 0..bins {
        let mut re_sum = 0.0_f32;
        let mut im_sum = 0.0_f32;
        for t in 0..actual_n {
            let angle = 2.0 * PI * (k as f32) * (t as f32) / (n as f32);
            re_sum += samples[t] * angle.cos();
            im_sum -= samples[t] * angle.sin();
        }
        re[k] = re_sum / n as f32;
        im[k] = im_sum / n as f32;
    }
    (re, im)
}

/// Compute per-bin power spectrum via real DFT.
fn real_dft_power_spectrum(samples: &[f32], n: usize) -> Vec<f32> {
    let (re, im) = real_dft(samples, n);
    re.iter()
        .zip(im.iter())
        .map(|(r, i)| r * r + i * i)
        .collect()
}

/// Compute the inverse real DFT from (Re, Im) bins of length `bins = n/2+1`.
fn real_idft(re: &[f32], im: &[f32], n: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; n];
    let bins = re.len();

    for t in 0..n {
        let mut sum = 0.0_f32;
        // DC bin (k=0)
        sum += re[0];
        // Middle bins
        for k in 1..bins.saturating_sub(1) {
            let angle = 2.0 * PI * (k as f32) * (t as f32) / (n as f32);
            sum += 2.0 * (re[k] * angle.cos() - im[k] * angle.sin());
        }
        // Nyquist bin if n is even
        if bins > 1 {
            let k = bins - 1;
            let angle = 2.0 * PI * (k as f32) * (t as f32) / (n as f32);
            sum += re[k] * angle.cos() - im[k] * angle.sin();
        }
        output[t] = sum;
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noise_profile_from_silence() {
        let samples = vec![0.0f32; 256];
        let profile = NoiseProfile::estimate_from_silence(&samples);
        // Silence should yield near-zero spectrum
        for &p in &profile.spectrum {
            assert!(p.abs() < 1e-10, "Silence spectrum should be near zero");
        }
    }

    #[test]
    fn test_noise_profile_empty() {
        let profile = NoiseProfile::estimate_from_silence(&[]);
        assert!(profile.spectrum.is_empty());
    }

    #[test]
    fn test_spectral_subtraction_silence() {
        let samples = vec![0.0f32; 512];
        let profile = NoiseProfile::estimate_from_silence(&samples);
        let denoised = SpectralSubtraction::new().denoise(&samples, &profile, 48000);
        assert_eq!(denoised.len(), samples.len());
        for &s in &denoised {
            assert!(s.is_finite(), "Output should be finite");
        }
    }

    #[test]
    fn test_spectral_subtraction_sine() {
        let n = 512;
        let samples: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 48000.0).sin() * 0.5)
            .collect();
        let noise_samples = vec![0.0f32; 256];
        let profile = NoiseProfile::estimate_from_silence(&noise_samples);
        let denoised = SpectralSubtraction::new().denoise(&samples, &profile, 48000);
        assert_eq!(denoised.len(), n);
    }

    #[test]
    fn test_wiener_filter_gain() {
        // With equal signal and noise, gain = 0.5
        let g = WienerFilter::compute_gain(1.0, 1.0);
        assert!((g - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_wiener_filter_gain_no_noise() {
        let g = WienerFilter::compute_gain(1.0, 0.0);
        assert!((g - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_wiener_filter_gain_all_noise() {
        let g = WienerFilter::compute_gain(0.0, 1.0);
        assert!((g - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_wiener_denoiser() {
        let samples: Vec<f32> = (0..256).map(|i| (i as f32 * 0.1).sin() * 0.5).collect();
        let denoised = WienerDenoiser::new().denoise(&samples, -60.0);
        assert_eq!(denoised.len(), samples.len());
        for &s in &denoised {
            assert!(s.is_finite());
        }
    }

    #[test]
    fn test_wiener_denoiser_empty() {
        let denoised = WienerDenoiser::new().denoise(&[], -60.0);
        assert!(denoised.is_empty());
    }

    #[test]
    fn test_hiss_remover_silence() {
        let samples = vec![0.0f32; 512];
        let hiss = HissRemover::new().detect_hiss(&samples, 48000);
        // Silence has no frequency content; ratio is 0 or NaN, but not > 0.5
        assert!(hiss == 0.0 || hiss.is_nan());
    }

    #[test]
    fn test_hiss_remover_high_freq() {
        let sr = 48000u32;
        // Pure high-frequency tone should have high hiss ratio
        let samples: Vec<f32> = (0..512)
            .map(|i| (2.0 * PI * 20000.0 * i as f32 / sr as f32).sin())
            .collect();
        let hiss = HissRemover::new().detect_hiss(&samples, sr);
        assert!(
            hiss > 0.1,
            "High frequency signal should have elevated hiss ratio: {hiss}"
        );
    }

    #[test]
    fn test_click_remover_detect_no_clicks() {
        let samples: Vec<f32> = (0..100).map(|i| (i as f32 * 0.01).sin()).collect();
        let clicks = ClickRemover::new().detect_clicks(&samples, 3.0);
        // Smooth sine should have very few or no clicks
        assert!(clicks.len() < 5);
    }

    #[test]
    fn test_click_remover_detect_impulse() {
        let mut samples = vec![0.1f32; 200];
        samples[100] = 10.0; // Obvious click
        let clicks = ClickRemover::new().detect_clicks(&samples, 3.0);
        assert!(
            clicks.contains(&100),
            "Should detect click at index 100; got {clicks:?}"
        );
    }

    #[test]
    fn test_click_remover_repair() {
        let mut samples = vec![0.5f32; 200];
        samples[100] = 10.0; // Click
        let click_indices = vec![100];
        ClickRemover::repair_clicks(&mut samples, &click_indices, 2);
        // After repair, sample 100 should be much closer to 0.5
        assert!(
            (samples[100] - 0.5).abs() < 1.0,
            "Click should be repaired: got {}",
            samples[100]
        );
    }

    #[test]
    fn test_dft_roundtrip() {
        let n = 16;
        let original: Vec<f32> = (0..n).map(|i| (i as f32 * 0.5).sin()).collect();
        let (re, im) = real_dft(&original, n);
        let recovered = real_idft(&re, &im, n);
        for (o, r) in original.iter().zip(recovered.iter()) {
            assert!(
                (o - r).abs() < 0.02,
                "DFT roundtrip mismatch: original={o}, recovered={r}"
            );
        }
    }
}
