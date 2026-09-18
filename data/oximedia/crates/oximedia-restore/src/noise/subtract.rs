//! Spectral subtraction for noise reduction.

use crate::error::RestoreResult;
use crate::noise::profile::NoiseProfile;
use crate::utils::spectral::{window_coefficients, FftProcessor, WindowFunction};

/// Spectral subtraction configuration.
#[derive(Debug, Clone)]
pub struct SpectralSubtractionConfig {
    /// Oversubtraction factor (1.0 = basic subtraction, >1.0 = more aggressive).
    pub oversubtraction_factor: f32,
    /// Spectral floor in dB to prevent musical noise.
    pub spectral_floor_db: f32,
    /// Smoothing factor for spectral subtraction (0.0 to 1.0).
    pub smoothing: f32,
}

impl Default for SpectralSubtractionConfig {
    fn default() -> Self {
        Self {
            oversubtraction_factor: 1.5,
            spectral_floor_db: -40.0,
            smoothing: 0.8,
        }
    }
}

/// Number of frames used when estimating the noise floor from initial silence.
const INITIAL_SILENCE_FRAMES: usize = 8;

/// Spectral subtraction processor.
#[derive(Debug)]
pub struct SpectralSubtraction {
    config: SpectralSubtractionConfig,
    noise_profile: NoiseProfile,
    fft_size: usize,
    hop_size: usize,
    prev_gain: Vec<f32>,
}

impl SpectralSubtraction {
    /// Create a new spectral subtraction processor.
    ///
    /// # Arguments
    ///
    /// * `noise_profile` - Noise profile to subtract
    /// * `hop_size` - Hop size between frames
    /// * `config` - Configuration
    #[must_use]
    pub fn new(
        noise_profile: NoiseProfile,
        hop_size: usize,
        config: SpectralSubtractionConfig,
    ) -> Self {
        let spectrum_size = noise_profile.fft_size / 2 + 1;
        Self {
            config,
            fft_size: noise_profile.fft_size,
            noise_profile,
            hop_size,
            prev_gain: vec![1.0; spectrum_size],
        }
    }

    /// Construct a `SpectralSubtraction` by **estimating the noise floor automatically**
    /// from the initial silence at the start of `samples`.
    ///
    /// The estimator:
    /// 1. Scans the first `INITIAL_SILENCE_FRAMES` × `fft_size` samples.
    /// 2. Computes the per-bin average magnitude spectrum across those frames.
    /// 3. Applies a gentle over-estimation factor (`1.05×`) so that the subtracted
    ///    profile is slightly conservative, avoiding musical noise artefacts when the
    ///    true signal-to-noise ratio is high.
    ///
    /// If the initial segment is too short for even one FFT frame, this falls back
    /// to an empty (flat-zero) profile.
    ///
    /// # Arguments
    ///
    /// * `samples` - Full audio buffer; the initial silence must be at the front
    /// * `fft_size` - FFT frame size (power of two recommended)
    /// * `hop_size` - Hop size between frames used during processing
    /// * `config` - Spectral subtraction configuration
    pub fn new_from_initial_silence(
        samples: &[f32],
        fft_size: usize,
        hop_size: usize,
        config: SpectralSubtractionConfig,
    ) -> crate::error::RestoreResult<Self> {
        let spectrum_size = fft_size / 2 + 1;

        // How many samples to use for the initial-silence noise estimate
        let silence_region_len = (INITIAL_SILENCE_FRAMES * fft_size).min(samples.len());

        let noise_profile = if silence_region_len >= fft_size {
            let silence_samples = &samples[..silence_region_len];
            let silence_hop = fft_size / 2;
            let mut profile = NoiseProfile::learn(silence_samples, fft_size, silence_hop)?;

            // Apply a conservative over-estimation factor to reduce musical noise
            const OVER_EST: f32 = 1.05;
            for m in &mut profile.magnitude {
                *m *= OVER_EST;
            }
            for p in &mut profile.power {
                *p *= OVER_EST * OVER_EST;
            }
            profile
        } else {
            // Not enough data — empty flat profile (no subtraction)
            NoiseProfile::new(fft_size)
        };

        Ok(Self {
            config,
            fft_size: noise_profile.fft_size,
            noise_profile,
            hop_size,
            prev_gain: vec![1.0; spectrum_size],
        })
    }

    /// Process samples to remove noise.
    ///
    /// # Arguments
    ///
    /// * `samples` - Input samples
    ///
    /// # Returns
    ///
    /// Noise-reduced samples.
    pub fn process(&mut self, samples: &[f32]) -> RestoreResult<Vec<f32>> {
        if samples.len() < self.fft_size {
            return Ok(samples.to_vec());
        }

        let fft = FftProcessor::new(self.fft_size);
        // Same Hann window for analysis and synthesis → weighted overlap-add
        // (WOLA).  Normalise by the accumulated `Σ w[i]²` so that a
        // unity-gain (empty/zero noise profile) configuration reconstructs the
        // input exactly, instead of leaving the position-dependent gain error
        // a raw frame-count divisor would.
        let window = window_coefficients(self.fft_size, WindowFunction::Hann);
        let mut output = vec![0.0; samples.len()];
        let mut window_sum = vec![0.0f32; samples.len()];

        let spectral_floor = db_to_linear(self.config.spectral_floor_db);

        let mut pos = 0;
        while pos + self.fft_size <= samples.len() {
            // Extract and window frame
            let mut frame = samples[pos..pos + self.fft_size].to_vec();
            for (s, &w) in frame.iter_mut().zip(window.iter()) {
                *s *= w;
            }

            // Forward FFT
            let spectrum = fft.forward(&frame)?;
            let magnitude = fft.magnitude(&spectrum);
            let phase = fft.phase(&spectrum);

            // Spectral subtraction.
            //
            // The input is real, so the full N-point spectrum is Hermitian:
            // `X[N-k] == conj(X[k])`.  The noise profile only spans the unique
            // half `[0, N/2]` (length `N/2 + 1`).  We compute the suppression
            // gain on that half and **mirror it onto the conjugate bins**
            // (`N-k`) so the processed spectrum stays Hermitian — otherwise the
            // upper half is left untouched (or zeroed) and the inverse FFT's
            // real part collapses to half amplitude.
            let half = self.fft_size / 2;
            let mut processed_mag = magnitude.clone();
            for i in 0..=half {
                let mag = magnitude[i];
                let noise_mag = self.noise_profile.magnitude.get(i).copied().unwrap_or(0.0);

                // Subtract noise with oversubtraction factor
                let subtracted = mag - self.config.oversubtraction_factor * noise_mag;

                // Apply spectral floor
                let floored = subtracted.max(spectral_floor * mag);

                // Compute gain
                let gain = if mag > f32::EPSILON {
                    floored / mag
                } else {
                    0.0
                };

                // Smooth gain over time to reduce musical noise
                let smoothed_gain = self.config.smoothing * self.prev_gain[i]
                    + (1.0 - self.config.smoothing) * gain;
                self.prev_gain[i] = smoothed_gain;

                processed_mag[i] = mag * smoothed_gain;
                // Mirror onto the conjugate-symmetric bin to preserve Hermitian
                // symmetry (skip DC and Nyquist, which are self-conjugate).
                if i > 0 && i < half {
                    let mirror = self.fft_size - i;
                    processed_mag[mirror] = magnitude[mirror] * smoothed_gain;
                }
            }

            // Reconstruct complex spectrum (Hermitian-preserving)
            let processed_spectrum = FftProcessor::from_polar(&processed_mag, &phase)?;

            // Inverse FFT
            let processed_frame = fft.inverse(&processed_spectrum)?;

            // Apply synthesis window and weighted overlap-add
            for (i, (&sample, &w)) in processed_frame.iter().zip(window.iter()).enumerate() {
                output[pos + i] += sample * w;
                window_sum[pos + i] += w * w;
            }

            pos += self.hop_size;
        }

        // Normalize by the window-overlap-squared sum (WOLA).
        for (i, &wsum) in window_sum.iter().enumerate() {
            if wsum > f32::EPSILON {
                output[i] /= wsum;
            }
        }

        Ok(output)
    }

    /// Reset processor state.
    pub fn reset(&mut self) {
        self.prev_gain.fill(1.0);
    }
}

/// Adaptive spectral subtraction with VAD.
#[derive(Debug)]
pub struct AdaptiveSpectralSubtraction {
    config: SpectralSubtractionConfig,
    noise_profile: NoiseProfile,
    fft_size: usize,
    hop_size: usize,
    prev_gain: Vec<f32>,
    vad_threshold: f32,
}

impl AdaptiveSpectralSubtraction {
    /// Create a new adaptive spectral subtraction processor.
    #[must_use]
    pub fn new(
        noise_profile: NoiseProfile,
        hop_size: usize,
        config: SpectralSubtractionConfig,
        vad_threshold: f32,
    ) -> Self {
        let spectrum_size = noise_profile.fft_size / 2 + 1;
        Self {
            config,
            fft_size: noise_profile.fft_size,
            noise_profile,
            hop_size,
            prev_gain: vec![1.0; spectrum_size],
            vad_threshold,
        }
    }

    /// Process samples with adaptive noise profile update.
    pub fn process(&mut self, samples: &[f32]) -> RestoreResult<Vec<f32>> {
        if samples.len() < self.fft_size {
            return Ok(samples.to_vec());
        }

        let fft = FftProcessor::new(self.fft_size);
        // WOLA reconstruction: see `SpectralSubtraction::process` for the
        // rationale behind normalising by `Σ w[i]²` rather than the frame count.
        let window = window_coefficients(self.fft_size, WindowFunction::Hann);
        let mut output = vec![0.0; samples.len()];
        let mut window_sum = vec![0.0f32; samples.len()];

        let spectral_floor = db_to_linear(self.config.spectral_floor_db);

        let mut pos = 0;
        while pos + self.fft_size <= samples.len() {
            let mut frame = samples[pos..pos + self.fft_size].to_vec();
            for (s, &w) in frame.iter_mut().zip(window.iter()) {
                *s *= w;
            }

            let spectrum = fft.forward(&frame)?;
            let magnitude = fft.magnitude(&spectrum);
            let phase = fft.phase(&spectrum);

            // Simple VAD: check frame energy
            let energy: f32 = magnitude.iter().map(|&m| m * m).sum();
            let is_speech = energy > self.vad_threshold;

            // Update noise profile during non-speech (unique half only)
            let half = self.fft_size / 2;
            if !is_speech {
                let alpha = 0.05; // Update rate
                for i in 0..=half {
                    if i < self.noise_profile.magnitude.len() {
                        self.noise_profile.magnitude[i] =
                            alpha * magnitude[i] + (1.0 - alpha) * self.noise_profile.magnitude[i];
                    }
                }
            }

            // Spectral subtraction with Hermitian-preserving conjugate mirroring
            // (see `SpectralSubtraction::process` for the rationale).
            let mut processed_mag = magnitude.clone();
            for i in 0..=half {
                let mag = magnitude[i];
                let noise_mag = self.noise_profile.magnitude.get(i).copied().unwrap_or(0.0);
                let subtracted = mag - self.config.oversubtraction_factor * noise_mag;
                let floored = subtracted.max(spectral_floor * mag);

                let gain = if mag > f32::EPSILON {
                    floored / mag
                } else {
                    0.0
                };

                let smoothed_gain = self.config.smoothing * self.prev_gain[i]
                    + (1.0 - self.config.smoothing) * gain;
                self.prev_gain[i] = smoothed_gain;

                processed_mag[i] = mag * smoothed_gain;
                if i > 0 && i < half {
                    let mirror = self.fft_size - i;
                    processed_mag[mirror] = magnitude[mirror] * smoothed_gain;
                }
            }

            let processed_spectrum = FftProcessor::from_polar(&processed_mag, &phase)?;
            let processed_frame = fft.inverse(&processed_spectrum)?;

            for (i, (&sample, &w)) in processed_frame.iter().zip(window.iter()).enumerate() {
                output[pos + i] += sample * w;
                window_sum[pos + i] += w * w;
            }

            pos += self.hop_size;
        }

        for (i, &wsum) in window_sum.iter().enumerate() {
            if wsum > f32::EPSILON {
                output[i] /= wsum;
            }
        }

        Ok(output)
    }
}

/// Convert dB to linear scale.
#[must_use]
fn db_to_linear(db: f32) -> f32 {
    10.0_f32.powf(db / 20.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectral_subtraction() {
        use rand::RngExt;
        let mut rng = rand::rng();

        // Create noise profile
        let noise_samples: Vec<f32> = (0..8192).map(|_| rng.random_range(-0.1..0.1)).collect();
        let profile =
            NoiseProfile::learn(&noise_samples, 2048, 1024).expect("should succeed in test");

        // Create noisy signal
        let mut signal: Vec<f32> = (0..8192)
            .map(|i| {
                use std::f32::consts::PI;
                (2.0 * PI * 440.0 * i as f32 / 44100.0).sin()
            })
            .collect();

        for i in 0..signal.len() {
            signal[i] += rng.random_range(-0.1..0.1);
        }

        let mut processor =
            SpectralSubtraction::new(profile, 1024, SpectralSubtractionConfig::default());
        let output = processor.process(&signal).expect("should succeed in test");

        assert_eq!(output.len(), signal.len());
    }

    #[test]
    fn test_adaptive_spectral_subtraction() {
        use rand::RngExt;
        let mut rng = rand::rng();

        let noise_samples: Vec<f32> = (0..8192).map(|_| rng.random_range(-0.1..0.1)).collect();
        let profile =
            NoiseProfile::learn(&noise_samples, 2048, 1024).expect("should succeed in test");

        let signal: Vec<f32> = (0..8192).map(|_| rng.random_range(-0.2..0.2)).collect();

        let mut processor = AdaptiveSpectralSubtraction::new(
            profile,
            1024,
            SpectralSubtractionConfig::default(),
            0.1,
        );
        let output = processor.process(&signal).expect("should succeed in test");

        assert_eq!(output.len(), signal.len());
    }

    #[test]
    fn test_db_to_linear() {
        assert!((db_to_linear(0.0) - 1.0).abs() < 1e-5);
        assert!((db_to_linear(-20.0) - 0.1).abs() < 1e-3);
    }

    #[test]
    fn test_config_default() {
        let config = SpectralSubtractionConfig::default();
        assert!(config.oversubtraction_factor > 1.0);
        assert!(config.spectral_floor_db < 0.0);
    }

    #[test]
    fn test_spectral_subtraction_from_initial_silence() {
        use rand::RngExt;
        let mut rng = rand::rng();

        // Build a signal: first 8 × 2048 = 16384 samples are near-silence noise,
        // the rest is a 440 Hz tone buried in lighter noise.
        let fft_size = 2048_usize;
        let silence_len = 8 * fft_size; // = 16384

        let mut signal = Vec::with_capacity(silence_len + 8192);
        for _ in 0..silence_len {
            signal.push(rng.random_range(-0.05_f32..0.05));
        }
        for i in 0..8192_usize {
            use std::f32::consts::PI;
            let t = i as f32 / 44100.0;
            signal.push((2.0 * PI * 440.0 * t).sin() + rng.random_range(-0.05..0.05));
        }

        let mut processor = SpectralSubtraction::new_from_initial_silence(
            &signal,
            fft_size,
            fft_size / 2,
            SpectralSubtractionConfig::default(),
        )
        .expect("should succeed");

        let output = processor.process(&signal).expect("process should succeed");
        assert_eq!(output.len(), signal.len());
    }

    #[test]
    fn test_spectral_subtraction_from_initial_silence_too_short() {
        // Buffer shorter than one FFT frame — should create an empty profile gracefully
        let short_samples = vec![0.01_f32; 512];
        let mut processor = SpectralSubtraction::new_from_initial_silence(
            &short_samples,
            2048,
            1024,
            SpectralSubtractionConfig::default(),
        )
        .expect("should succeed even with short input");
        let output = processor
            .process(&short_samples)
            .expect("process should succeed");
        assert_eq!(output.len(), short_samples.len());
    }
}
