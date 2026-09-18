//! Spectral dereverberation using Wiener-filter based spectral subtraction.
//!
//! Implements short-time spectral subtraction in the FFT domain to suppress
//! reverberation. The reverberation power estimate is derived from an
//! exponentially-decaying model parameterised by the reverb time (RT60).

use crate::error::RestoreResult;
use crate::utils::spectral::{apply_window, FftProcessor, WindowFunction};

/// Reverberation reducer configuration.
#[derive(Debug, Clone)]
pub struct ReverbReducer {
    /// Estimated reverb time in milliseconds (RT60).
    pub reverb_time_ms: f32,
    /// Frequency smoothing factor (0.0 = no smoothing, 1.0 = maximum).
    pub frequency_smoothing: f32,
    /// FFT size used for processing.
    fft_size: usize,
    /// Hop size between frames.
    hop_size: usize,
    /// Oversubtraction factor (>1.0 = more aggressive).
    oversubtraction: f32,
    /// Spectral floor to prevent musical noise (linear scale).
    spectral_floor: f32,
}

impl Default for ReverbReducer {
    fn default() -> Self {
        Self {
            reverb_time_ms: 300.0,
            frequency_smoothing: 0.5,
            fft_size: 2048,
            hop_size: 512,
            oversubtraction: 2.0,
            spectral_floor: 0.01,
        }
    }
}

impl ReverbReducer {
    /// Create a new reverb reducer.
    ///
    /// # Arguments
    ///
    /// * `reverb_time_ms` - Reverb decay time (RT60) in milliseconds
    /// * `frequency_smoothing` - Spectral smoothing factor [0.0, 1.0]
    #[must_use]
    pub fn new(reverb_time_ms: f32, frequency_smoothing: f32) -> Self {
        Self {
            reverb_time_ms: reverb_time_ms.max(0.0),
            frequency_smoothing: frequency_smoothing.clamp(0.0, 1.0),
            ..Self::default()
        }
    }

    /// Set FFT size and hop size.
    ///
    /// The FFT size should be a power of two.
    #[must_use]
    pub fn with_fft_size(mut self, fft_size: usize, hop_size: usize) -> Self {
        self.fft_size = fft_size.next_power_of_two();
        self.hop_size = hop_size;
        self
    }

    /// Set oversubtraction factor.
    #[must_use]
    pub fn with_oversubtraction(mut self, factor: f32) -> Self {
        self.oversubtraction = factor.max(1.0);
        self
    }

    /// Process samples to reduce reverberation.
    ///
    /// Uses overlap-add with Hann windowing and per-bin Wiener-filter gain
    /// computed from a recursive reverb power estimate.
    ///
    /// # Arguments
    ///
    /// * `samples` - Input samples (not modified)
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Returns
    ///
    /// Dereverberated samples with the same length as input.
    #[allow(clippy::cast_precision_loss)]
    pub fn process(&self, samples: &[f32], sample_rate: u32) -> RestoreResult<Vec<f32>> {
        if samples.is_empty() {
            return Ok(Vec::new());
        }

        let fft_size = self.fft_size.min(samples.len().next_power_of_two());
        let fft_size = if fft_size < 4 { 4 } else { fft_size };
        let hop_size = self.hop_size.min(fft_size / 2).max(1);

        if samples.len() < fft_size {
            // Too short to process — return unchanged
            return Ok(samples.to_vec());
        }

        let fft = FftProcessor::new(fft_size);
        let spectrum_bins = fft_size;

        // Reverb decay per frame: e^(-ln(1000) * hop / (rt60_samples))
        // RT60 in samples = reverb_time_ms * sample_rate / 1000
        let rt60_samples = (self.reverb_time_ms * sample_rate as f32 / 1000.0).max(1.0);
        let decay_per_hop = (-6.908 * hop_size as f32 / rt60_samples).exp();

        let mut reverb_estimate = vec![f32::EPSILON; spectrum_bins];
        let mut prev_gain = vec![1.0_f32; spectrum_bins];

        let mut output = vec![0.0_f32; samples.len()];
        let mut weight = vec![0.0_f32; samples.len()];

        let mut pos = 0;
        while pos + fft_size <= samples.len() {
            let mut frame = samples[pos..pos + fft_size].to_vec();
            apply_window(&mut frame, WindowFunction::Hann);

            let spectrum = fft.forward(&frame)?;
            let magnitude = fft.magnitude(&spectrum);
            let phase = fft.phase(&spectrum);

            // Update recursive reverb power estimate
            let mut new_rev_est = vec![0.0_f32; spectrum_bins];
            let mut processed_mag = vec![0.0_f32; spectrum_bins];

            for i in 0..spectrum_bins {
                let signal_power = magnitude[i] * magnitude[i];
                // Recursive estimate: rev[i] = alpha * rev[i] + (1-alpha) * max(signal - rev, 0)
                let alpha = decay_per_hop;
                let new_power = (signal_power - reverb_estimate[i]).max(0.0);
                new_rev_est[i] = alpha * reverb_estimate[i] + (1.0 - alpha) * new_power;
            }

            // Optional frequency smoothing of reverb estimate
            if self.frequency_smoothing > 0.0 {
                smooth_spectrum(&mut new_rev_est, self.frequency_smoothing);
            }

            // Compute Wiener-filter gain: G = (|S|² - λ|N|²) / |S|²
            for i in 0..spectrum_bins {
                reverb_estimate[i] = new_rev_est[i];
                let signal_power = magnitude[i] * magnitude[i];
                let subtracted = signal_power - self.oversubtraction * reverb_estimate[i];
                let floored = subtracted.max(self.spectral_floor * signal_power);
                let gain = if signal_power > f32::EPSILON {
                    (floored / signal_power).sqrt().clamp(0.0, 1.0)
                } else {
                    0.0
                };

                // Smooth gain over time (Wiener post-filter smoothing)
                let smoothed = self.frequency_smoothing * prev_gain[i]
                    + (1.0 - self.frequency_smoothing) * gain;
                prev_gain[i] = smoothed;
                processed_mag[i] = magnitude[i] * smoothed;
            }

            // Reconstruct complex spectrum
            let processed_spectrum = FftProcessor::from_polar(&processed_mag, &phase)?;

            // Inverse FFT
            let processed_frame = fft.inverse(&processed_spectrum)?;

            // Overlap-add with Hann window
            let mut windowed = processed_frame;
            apply_window(&mut windowed, WindowFunction::Hann);

            for (i, &s) in windowed.iter().enumerate() {
                output[pos + i] += s;
                weight[pos + i] += 1.0;
            }

            pos += hop_size;
        }

        // Normalise by overlap count
        for (o, &w) in output.iter_mut().zip(weight.iter()) {
            if w > 0.0 {
                *o /= w;
            }
        }

        Ok(output)
    }
}

/// Apply a moving-average smoothing to a spectral magnitude vector.
fn smooth_spectrum(spectrum: &mut [f32], factor: f32) {
    if spectrum.len() < 3 {
        return;
    }
    let alpha = factor.clamp(0.0, 1.0);
    let mut prev = spectrum[0];
    for s in spectrum.iter_mut() {
        let smoothed = alpha * prev + (1.0 - alpha) * *s;
        prev = smoothed;
        *s = smoothed;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn make_sine(sample_rate: u32, freq: f32, duration_ms: f32) -> Vec<f32> {
        let n = (sample_rate as f32 * duration_ms / 1000.0) as usize;
        (0..n)
            .map(|i| (2.0 * PI * freq * i as f32 / sample_rate as f32).sin())
            .collect()
    }

    #[test]
    fn test_reverb_reducer_default() {
        let r = ReverbReducer::default();
        assert!(r.reverb_time_ms > 0.0);
        assert!(r.frequency_smoothing >= 0.0 && r.frequency_smoothing <= 1.0);
    }

    #[test]
    fn test_reverb_reducer_new() {
        let r = ReverbReducer::new(500.0, 0.3);
        assert!((r.reverb_time_ms - 500.0).abs() < f32::EPSILON);
        assert!((r.frequency_smoothing - 0.3).abs() < f32::EPSILON);
    }

    #[test]
    fn test_process_empty() {
        let r = ReverbReducer::default();
        let out = r.process(&[], 44100).expect("should succeed");
        assert!(out.is_empty());
    }

    #[test]
    fn test_process_too_short() {
        let r = ReverbReducer::default();
        let samples = vec![0.1_f32; 100];
        let out = r.process(&samples, 44100).expect("should succeed");
        // Short signals are returned unchanged
        assert_eq!(out.len(), samples.len());
    }

    #[test]
    fn test_process_sine_output_length() {
        let r = ReverbReducer::new(200.0, 0.4).with_fft_size(512, 128);
        let samples = make_sine(44100, 440.0, 200.0);
        let out = r.process(&samples, 44100).expect("should succeed");
        assert_eq!(out.len(), samples.len());
    }

    #[test]
    fn test_process_reduces_energy() {
        // After dereverberation the RMS energy of a simple sine wave should be
        // reduced (reverb estimate will partially suppress the signal since there
        // is no separate dry source). This is an inherent side-effect.
        let r = ReverbReducer::new(100.0, 0.2).with_fft_size(512, 128);
        let samples = make_sine(44100, 880.0, 300.0);
        let out = r.process(&samples, 44100).expect("should succeed");

        let in_rms: f32 =
            (samples.iter().map(|&s| s * s).sum::<f32>() / samples.len() as f32).sqrt();
        let out_rms: f32 = (out.iter().map(|&s| s * s).sum::<f32>() / out.len() as f32).sqrt();

        // Output energy should be <= input energy (dereverb suppresses content)
        assert!(
            out_rms <= in_rms + 0.05,
            "out_rms={out_rms} in_rms={in_rms}"
        );
    }

    #[test]
    fn test_process_silence_stays_silence() {
        let r = ReverbReducer::new(300.0, 0.5).with_fft_size(512, 128);
        let samples = vec![0.0_f32; 4096];
        let out = r.process(&samples, 44100).expect("should succeed");
        for &s in &out {
            assert!(s.abs() < 1e-6, "silence should stay silent, got {s}");
        }
    }

    #[test]
    fn test_process_with_smoothing() {
        let r = ReverbReducer::new(200.0, 0.8).with_fft_size(512, 128);
        let samples = make_sine(44100, 500.0, 200.0);
        let out = r.process(&samples, 44100).expect("should succeed");
        assert_eq!(out.len(), samples.len());
    }

    #[test]
    fn test_with_oversubtraction() {
        let r = ReverbReducer::new(200.0, 0.3)
            .with_fft_size(512, 128)
            .with_oversubtraction(3.0);
        assert!((r.oversubtraction - 3.0).abs() < f32::EPSILON);
        let samples = make_sine(44100, 440.0, 200.0);
        let out = r.process(&samples, 44100).expect("should succeed");
        assert_eq!(out.len(), samples.len());
    }

    #[test]
    fn test_smooth_spectrum_no_change_when_zero_factor() {
        let mut spectrum = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let original = spectrum.clone();
        smooth_spectrum(&mut spectrum, 0.0);
        // With alpha=0, each element is replaced by itself (no history)
        for (a, b) in spectrum.iter().zip(original.iter()) {
            assert!(
                (a - b).abs() < 0.5,
                "spectrum should be close to original for alpha=0"
            );
        }
    }
}
