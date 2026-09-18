//! Noise reduction for speech clarity.
//!
//! Provides three complementary noise-reduction algorithms:
//! - **Noise gate**: suppress signal below an amplitude threshold.
//! - **Spectral subtraction**: subtract an estimated noise spectrum in the
//!   frequency domain.
//! - **Wiener filter**: optimal linear filter that minimises the mean-square
//!   error between the clean and noisy signals.

use crate::error::AccessResult;
use oximedia_audio::frame::AudioBuffer;

// ---------------------------------------------------------------------------
// Noise gate
// ---------------------------------------------------------------------------

/// A simple noise gate that suppresses samples below a threshold.
///
/// When the instantaneous amplitude drops below `threshold`, the gate
/// attenuates the signal by `attenuation_db` dB.  A short look-ahead and
/// release time prevent audible clicks.
#[derive(Debug, Clone)]
pub struct NoiseGate {
    /// Amplitude threshold (linear, 0.0–1.0).
    threshold: f32,
    /// Attenuation applied in the closed state (linear gain, 0.0–1.0).
    floor_gain: f32,
    /// Attack coefficient (exponential smoothing, 0–1).
    attack_coeff: f32,
    /// Release coefficient (exponential smoothing, 0–1).
    release_coeff: f32,
}

impl NoiseGate {
    /// Create a noise gate with an explicit threshold and floor level.
    ///
    /// * `threshold` – level below which the gate closes (0.0–1.0 linear).
    /// * `floor_db` – attenuation when gate is closed (e.g. `-60.0` dB).
    /// * `attack_ms` – attack time in milliseconds.
    /// * `release_ms` – release time in milliseconds.
    /// * `sample_rate` – audio sample rate in Hz.
    #[must_use]
    pub fn new(
        threshold: f32,
        floor_db: f32,
        attack_ms: f32,
        release_ms: f32,
        sample_rate: u32,
    ) -> Self {
        let sr = sample_rate as f32;
        let attack_samples = (attack_ms * sr / 1000.0).max(1.0);
        let release_samples = (release_ms * sr / 1000.0).max(1.0);
        Self {
            threshold: threshold.clamp(0.0, 1.0),
            floor_gain: 10.0_f32.powf(floor_db.min(0.0) / 20.0),
            attack_coeff: 1.0 - (-1.0_f32 / attack_samples).exp(),
            release_coeff: 1.0 - (-1.0_f32 / release_samples).exp(),
        }
    }

    /// Apply the gate to a buffer of f32 samples in-place.
    ///
    /// Returns the number of samples that were gated (attenuated).
    #[must_use]
    #[allow(dead_code)]
    pub fn apply(&self, samples: &mut [f32]) -> usize {
        let mut gated_count = 0usize;
        let mut envelope = 0.0_f32;
        let mut gain = 1.0_f32;

        for sample in samples.iter_mut() {
            let abs = sample.abs();
            // Envelope follower
            let coeff = if abs > envelope {
                self.attack_coeff
            } else {
                self.release_coeff
            };
            envelope = envelope + coeff * (abs - envelope);

            // Compute target gain
            let target_gain = if envelope < self.threshold {
                self.floor_gain
            } else {
                1.0
            };

            // Smooth gain changes
            let gain_coeff = if target_gain < gain {
                self.attack_coeff
            } else {
                self.release_coeff
            };
            gain = gain + gain_coeff * (target_gain - gain);

            if gain < 1.0 - f32::EPSILON {
                gated_count += 1;
            }
            *sample *= gain;
        }
        gated_count
    }

    /// Threshold accessor.
    #[must_use]
    pub const fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Floor gain accessor (linear).
    #[must_use]
    pub const fn floor_gain(&self) -> f32 {
        self.floor_gain
    }
}

// ---------------------------------------------------------------------------
// Spectral subtraction
// ---------------------------------------------------------------------------

/// Spectral subtraction noise reducer.
///
/// Estimates the noise spectrum from a silent reference frame and subtracts
/// it from subsequent frames in the frequency domain.  A half-wave
/// rectification (musical noise floor) prevents spectral holes.
#[derive(Debug, Clone)]
pub struct SpectralSubtractor {
    /// Over-subtraction factor (α ≥ 1.0).  Higher values remove more noise
    /// but may introduce musical noise.
    alpha: f32,
    /// Spectral floor β (0.0–1.0).  Prevents negative power estimates.
    beta: f32,
    /// FFT frame size (number of samples).
    frame_size: usize,
}

impl SpectralSubtractor {
    /// Create a spectral subtractor.
    ///
    /// * `alpha` – over-subtraction factor (1.0–2.0 is typical).
    /// * `beta` – spectral floor (0.001–0.02 is typical).
    /// * `frame_size` – FFT frame size (power of two, e.g. 512).
    #[must_use]
    pub fn new(alpha: f32, beta: f32, frame_size: usize) -> Self {
        Self {
            alpha: alpha.max(1.0),
            beta: beta.clamp(0.0, 1.0),
            frame_size,
        }
    }

    /// Estimate noise power spectrum from a noise-only frame.
    ///
    /// `noise_samples` should be a short segment of background noise
    /// (at least `frame_size` samples long).
    ///
    /// Returns a vector of per-bin noise power values (length `frame_size / 2 + 1`).
    #[must_use]
    #[allow(dead_code)]
    pub fn estimate_noise_spectrum(&self, noise_samples: &[f32]) -> Vec<f32> {
        let n = self.frame_size;
        let num_bins = n / 2 + 1;
        if noise_samples.is_empty() {
            return vec![0.0; num_bins];
        }

        // Simple DFT-based power estimation (no windowing for brevity)
        // In production, an FFT library (e.g. rustfft) would be used.
        let len = noise_samples.len().min(n) as f32;
        let mut noise_psd = vec![0.0_f32; num_bins];

        for bin in 0..num_bins {
            let freq_idx = bin as f32;
            let mut re = 0.0_f32;
            let mut im = 0.0_f32;
            for (k, &s) in noise_samples.iter().take(n).enumerate() {
                let angle = -2.0 * std::f32::consts::PI * freq_idx * k as f32 / len;
                re += s * angle.cos();
                im += s * angle.sin();
            }
            noise_psd[bin] = (re * re + im * im) / (len * len);
        }
        noise_psd
    }

    /// Apply spectral subtraction to a frame of samples.
    ///
    /// * `frame` – input samples (length `frame_size`).
    /// * `noise_psd` – noise power spectrum from [`Self::estimate_noise_spectrum`].
    ///
    /// Returns the processed frame with reduced noise.
    #[must_use]
    #[allow(dead_code)]
    pub fn process_frame(&self, frame: &[f32], noise_psd: &[f32]) -> Vec<f32> {
        let n = self.frame_size.min(frame.len());
        let num_bins = n / 2 + 1;

        // Compute DFT
        let mut re = vec![0.0_f32; num_bins];
        let mut im = vec![0.0_f32; num_bins];
        for bin in 0..num_bins {
            let freq_idx = bin as f32;
            for (k, &s) in frame.iter().take(n).enumerate() {
                let angle = -2.0 * std::f32::consts::PI * freq_idx * k as f32 / n as f32;
                re[bin] += s * angle.cos();
                im[bin] += s * angle.sin();
            }
        }

        // Spectral subtraction
        let mut gain = vec![1.0_f32; num_bins];
        for bin in 0..num_bins {
            let signal_psd = (re[bin] * re[bin] + im[bin] * im[bin]) / (n * n) as f32;
            let noise = if bin < noise_psd.len() {
                noise_psd[bin]
            } else {
                0.0
            };
            let subtracted = signal_psd - self.alpha * noise;
            let floored = subtracted.max(self.beta * signal_psd);
            gain[bin] = if signal_psd > f32::EPSILON {
                (floored / signal_psd).sqrt().clamp(0.0, 1.0)
            } else {
                0.0
            };
        }

        // Apply gain and inverse DFT
        let mut output = vec![0.0_f32; n];
        for k in 0..n {
            let mut val = 0.0_f32;
            for bin in 0..num_bins {
                let freq_idx = bin as f32;
                let angle = 2.0 * std::f32::consts::PI * freq_idx * k as f32 / n as f32;
                val += gain[bin] * (re[bin] * angle.cos() - im[bin] * angle.sin());
            }
            output[k] = val / n as f32;
        }
        output
    }

    /// Alpha (over-subtraction) accessor.
    #[must_use]
    pub const fn alpha(&self) -> f32 {
        self.alpha
    }

    /// Beta (spectral floor) accessor.
    #[must_use]
    pub const fn beta(&self) -> f32 {
        self.beta
    }
}

// ---------------------------------------------------------------------------
// Wiener filter
// ---------------------------------------------------------------------------

/// Wiener filter for noise reduction.
///
/// The Wiener filter minimises the mean-square error between the estimated
/// clean signal and the true clean signal.  In the frequency domain, the
/// optimal filter gain for each bin is:
///
/// ```text
/// H(ω) = SNR(ω) / (1 + SNR(ω))
/// ```
///
/// where `SNR(ω) = P_signal(ω) / P_noise(ω)`.
#[derive(Debug, Clone)]
pub struct WienerFilter {
    /// Assumed minimum SNR floor to prevent division by zero (dB).
    min_snr_db: f32,
    /// Smoothing factor for noise PSD estimate update (0–1, lower = slower).
    noise_smooth: f32,
    /// FFT frame size.
    frame_size: usize,
}

impl WienerFilter {
    /// Create a Wiener filter.
    ///
    /// * `min_snr_db` – minimum SNR floor in dB (e.g. `-10.0`).
    /// * `noise_smooth` – PSD smoothing coefficient (e.g. `0.98`).
    /// * `frame_size` – FFT frame size.
    #[must_use]
    pub fn new(min_snr_db: f32, noise_smooth: f32, frame_size: usize) -> Self {
        Self {
            min_snr_db,
            noise_smooth: noise_smooth.clamp(0.0, 1.0),
            frame_size,
        }
    }

    /// Compute the per-bin Wiener gain given signal and noise PSDs.
    ///
    /// Returns a vector of linear gains (length = `signal_psd.len()`).
    #[must_use]
    #[allow(dead_code)]
    pub fn compute_gains(&self, signal_psd: &[f32], noise_psd: &[f32]) -> Vec<f32> {
        let min_snr_linear = 10.0_f32.powf(self.min_snr_db / 10.0);
        signal_psd
            .iter()
            .zip(noise_psd.iter())
            .map(|(&sig, &noise)| {
                if noise <= f32::EPSILON {
                    return 1.0;
                }
                let snr = (sig / noise).max(min_snr_linear);
                (snr / (1.0 + snr)).clamp(0.0, 1.0)
            })
            .collect()
    }

    /// Update the smoothed noise PSD estimate using first-order IIR smoothing.
    ///
    /// `current_psd` – current frame PSD estimate.
    /// `prev_noise_psd` – previous smoothed noise estimate (mutated in-place).
    #[allow(dead_code)]
    pub fn update_noise_psd(&self, current_psd: &[f32], prev_noise_psd: &mut Vec<f32>) {
        if prev_noise_psd.len() != current_psd.len() {
            *prev_noise_psd = current_psd.to_vec();
            return;
        }
        for (prev, &cur) in prev_noise_psd.iter_mut().zip(current_psd.iter()) {
            *prev = self.noise_smooth * *prev + (1.0 - self.noise_smooth) * cur;
        }
    }

    /// Apply the Wiener filter to a frame of samples.
    ///
    /// * `frame` – input samples.
    /// * `noise_psd` – noise power spectral density estimate.
    ///
    /// Returns the filtered output frame.
    #[must_use]
    #[allow(dead_code)]
    pub fn filter_frame(&self, frame: &[f32], noise_psd: &[f32]) -> Vec<f32> {
        let n = self.frame_size.min(frame.len());
        let num_bins = n / 2 + 1;

        // DFT
        let mut re = vec![0.0_f32; num_bins];
        let mut im = vec![0.0_f32; num_bins];
        for bin in 0..num_bins {
            let freq_idx = bin as f32;
            for (k, &s) in frame.iter().take(n).enumerate() {
                let angle = -2.0 * std::f32::consts::PI * freq_idx * k as f32 / n as f32;
                re[bin] += s * angle.cos();
                im[bin] += s * angle.sin();
            }
        }

        // Signal PSD
        let signal_psd: Vec<f32> = (0..num_bins)
            .map(|b| (re[b] * re[b] + im[b] * im[b]) / (n * n) as f32)
            .collect();

        // Wiener gains
        let gains = self.compute_gains(&signal_psd, noise_psd);

        // Apply gains and inverse DFT
        let mut output = vec![0.0_f32; n];
        for k in 0..n {
            let mut val = 0.0_f32;
            for bin in 0..num_bins {
                let freq_idx = bin as f32;
                let angle = 2.0 * std::f32::consts::PI * freq_idx * k as f32 / n as f32;
                val += gains[bin] * (re[bin] * angle.cos() - im[bin] * angle.sin());
            }
            output[k] = val / n as f32;
        }
        output
    }

    /// `min_snr_db` accessor.
    #[must_use]
    pub const fn min_snr_db(&self) -> f32 {
        self.min_snr_db
    }

    /// Noise smoothing coefficient accessor.
    #[must_use]
    pub const fn noise_smooth(&self) -> f32 {
        self.noise_smooth
    }
}

// ---------------------------------------------------------------------------
// Combined NoiseReducer (existing API + new algorithms)
// ---------------------------------------------------------------------------

/// Reduces background noise to improve speech clarity.
pub struct NoiseReducer {
    reduction_level: f32,
    /// Embedded noise gate for pre-processing.
    gate: NoiseGate,
    /// Spectral subtractor for frequency-domain noise removal.
    subtractor: SpectralSubtractor,
    /// Wiener filter for optimal linear filtering.
    wiener: WienerFilter,
}

impl NoiseReducer {
    /// Create a new noise reducer.
    #[must_use]
    pub fn new(reduction_level: f32) -> Self {
        let level = reduction_level.clamp(0.0, 1.0);
        // Map reduction level to algorithm parameters
        let gate_threshold = 0.1 * (1.0 - level * 0.5); // lower threshold for stronger reduction
        let alpha = 1.0 + level; // over-subtraction proportional to reduction level
        Self {
            reduction_level: level,
            gate: NoiseGate::new(gate_threshold, -60.0, 5.0, 50.0, 48000),
            subtractor: SpectralSubtractor::new(alpha, 0.002, 512),
            wiener: WienerFilter::new(-10.0, 0.98, 512),
        }
    }

    /// Reduce noise in audio.
    pub fn reduce(&self, audio: &AudioBuffer) -> AccessResult<AudioBuffer> {
        // In production, this would:
        // 1. Estimate noise floor
        // 2. Apply spectral subtraction
        // 3. Use Wiener filtering
        // 4. Preserve speech components

        Ok(audio.clone())
    }

    /// Reduce noise while preserving speech.
    pub fn reduce_preserve_speech(&self, audio: &AudioBuffer) -> AccessResult<AudioBuffer> {
        // More aggressive preservation of speech frequencies
        Ok(audio.clone())
    }

    /// Get reduction level.
    #[must_use]
    pub const fn reduction_level(&self) -> f32 {
        self.reduction_level
    }

    /// Access the embedded noise gate.
    #[must_use]
    pub const fn gate(&self) -> &NoiseGate {
        &self.gate
    }

    /// Access the embedded spectral subtractor.
    #[must_use]
    pub const fn spectral_subtractor(&self) -> &SpectralSubtractor {
        &self.subtractor
    }

    /// Access the embedded Wiener filter.
    #[must_use]
    pub const fn wiener_filter(&self) -> &WienerFilter {
        &self.wiener
    }
}

impl Default for NoiseReducer {
    fn default() -> Self {
        Self::new(0.5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;

    // ---- NoiseReducer --------------------------------------------------

    #[test]
    fn test_reducer_creation() {
        let reducer = NoiseReducer::new(0.8);
        assert!((reducer.reduction_level() - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn test_reduce() {
        let reducer = NoiseReducer::default();
        let audio = AudioBuffer::Interleaved(Bytes::from(vec![0u8; 48000 * 4]));
        let result = reducer.reduce(&audio);
        assert!(result.is_ok());
    }

    // ---- NoiseGate -----------------------------------------------------

    #[test]
    fn test_noise_gate_creation() {
        let gate = NoiseGate::new(0.05, -60.0, 5.0, 50.0, 48000);
        assert!((gate.threshold() - 0.05).abs() < f32::EPSILON);
        // floor_gain for -60 dB ≈ 0.001
        assert!(gate.floor_gain() < 0.01);
    }

    #[test]
    fn test_noise_gate_attenuates_below_threshold() {
        let gate = NoiseGate::new(0.5, -60.0, 1.0, 1.0, 48000);
        // A small signal well below threshold should be attenuated
        let mut samples = vec![0.01_f32; 1000];
        let gated = gate.apply(&mut samples);
        assert!(gated > 0, "Expected some samples to be gated");
        // All output samples should be small
        for s in &samples {
            assert!(s.abs() < 0.5, "Expected attenuated output, got {s}");
        }
    }

    #[test]
    fn test_noise_gate_passes_loud_signal() {
        let gate = NoiseGate::new(0.05, -60.0, 1.0, 10.0, 48000);
        // A full-scale signal should pass essentially unchanged
        let original = vec![1.0_f32; 200];
        let mut samples = original.clone();
        let _gated = gate.apply(&mut samples);
        let max_err = original
            .iter()
            .zip(samples.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_err < 0.3,
            "Expected small error for loud signal, got {max_err}"
        );
    }

    #[test]
    fn test_noise_gate_empty_input() {
        let gate = NoiseGate::new(0.1, -40.0, 5.0, 50.0, 48000);
        let mut samples: Vec<f32> = vec![];
        let gated = gate.apply(&mut samples);
        assert_eq!(gated, 0);
    }

    // ---- SpectralSubtractor --------------------------------------------

    #[test]
    fn test_spectral_subtractor_creation() {
        let sub = SpectralSubtractor::new(1.5, 0.01, 512);
        assert!((sub.alpha() - 1.5).abs() < f32::EPSILON);
        assert!((sub.beta() - 0.01).abs() < f32::EPSILON);
    }

    #[test]
    fn test_spectral_subtraction_silent_noise_estimate() {
        let sub = SpectralSubtractor::new(1.5, 0.01, 64);
        // Empty noise → zero PSD
        let psd = sub.estimate_noise_spectrum(&[]);
        assert!(!psd.is_empty());
        for v in &psd {
            assert!((v - 0.0).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_spectral_subtraction_process_frame() {
        let sub = SpectralSubtractor::new(1.2, 0.005, 32);
        let noise_psd = vec![0.0001_f32; 17]; // 32/2+1 bins
        let frame: Vec<f32> = (0..32_u32)
            .map(|i| (2.0 * std::f32::consts::PI * 4.0 * i as f32 / 32.0).sin())
            .collect();
        let output = sub.process_frame(&frame, &noise_psd);
        assert_eq!(output.len(), 32);
        // Output should be finite
        assert!(output.iter().all(|v| v.is_finite()));
    }

    // ---- WienerFilter --------------------------------------------------

    #[test]
    fn test_wiener_filter_creation() {
        let wf = WienerFilter::new(-10.0, 0.98, 512);
        assert!((wf.min_snr_db() - (-10.0)).abs() < f32::EPSILON);
        assert!((wf.noise_smooth() - 0.98).abs() < f32::EPSILON);
    }

    #[test]
    fn test_wiener_gains_high_snr() {
        let wf = WienerFilter::new(-10.0, 0.98, 64);
        // High signal, low noise → gains approach 1.0
        let signal_psd = vec![1.0_f32; 10];
        let noise_psd = vec![0.001_f32; 10];
        let gains = wf.compute_gains(&signal_psd, &noise_psd);
        for g in &gains {
            assert!(*g > 0.9, "Expected high gain, got {g}");
        }
    }

    #[test]
    fn test_wiener_gains_low_snr() {
        let wf = WienerFilter::new(-10.0, 0.98, 64);
        // Low signal, high noise → gains are low
        let min_snr_linear = 10.0_f32.powf(-10.0 / 10.0);
        let expected_max_gain = min_snr_linear / (1.0 + min_snr_linear) + 0.05;
        let signal_psd = vec![0.001_f32; 10];
        let noise_psd = vec![1.0_f32; 10];
        let gains = wf.compute_gains(&signal_psd, &noise_psd);
        for g in &gains {
            assert!(*g < expected_max_gain, "Expected low gain, got {g}");
        }
    }

    #[test]
    fn test_wiener_update_noise_psd() {
        let wf = WienerFilter::new(-10.0, 0.5, 64);
        let current = vec![1.0_f32; 4];
        let mut prev = vec![0.0_f32; 4];
        wf.update_noise_psd(&current, &mut prev);
        // With smooth=0.5: prev = 0.5*0 + 0.5*1 = 0.5
        for v in &prev {
            assert!((v - 0.5).abs() < 1e-5, "Expected 0.5, got {v}");
        }
    }

    #[test]
    fn test_wiener_filter_frame_finite() {
        let wf = WienerFilter::new(-10.0, 0.98, 32);
        let frame: Vec<f32> = (0..32_u32)
            .map(|i| (2.0 * std::f32::consts::PI * 2.0 * i as f32 / 32.0).sin())
            .collect();
        let noise_psd = vec![0.0001_f32; 17];
        let output = wf.filter_frame(&frame, &noise_psd);
        assert_eq!(output.len(), 32);
        assert!(output.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_wiener_gains_zero_noise() {
        let wf = WienerFilter::new(-10.0, 0.98, 64);
        let signal_psd = vec![0.5_f32; 5];
        let noise_psd = vec![0.0_f32; 5];
        let gains = wf.compute_gains(&signal_psd, &noise_psd);
        // Zero noise → gain should be 1.0
        for g in &gains {
            assert!((g - 1.0).abs() < f32::EPSILON, "Expected 1.0, got {g}");
        }
    }
}
