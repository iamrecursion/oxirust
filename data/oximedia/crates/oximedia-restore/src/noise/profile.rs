//! Noise profile learning and management.

use crate::error::{RestoreError, RestoreResult};
use crate::utils::spectral::{apply_window, FftProcessor, WindowFunction};

/// Noise profile.
#[derive(Debug, Clone)]
pub struct NoiseProfile {
    /// FFT size used for profile.
    pub fft_size: usize,
    /// Average magnitude spectrum of noise.
    pub magnitude: Vec<f32>,
    /// Power spectrum of noise.
    pub power: Vec<f32>,
    /// Number of frames used to build profile.
    pub frame_count: usize,
}

impl NoiseProfile {
    /// Create a new empty noise profile.
    #[must_use]
    pub fn new(fft_size: usize) -> Self {
        let spectrum_size = fft_size / 2 + 1;
        Self {
            fft_size,
            magnitude: vec![0.0; spectrum_size],
            power: vec![0.0; spectrum_size],
            frame_count: 0,
        }
    }

    /// Learn noise profile from samples.
    ///
    /// # Arguments
    ///
    /// * `samples` - Noise samples
    /// * `sample_rate` - Sample rate in Hz
    /// * `hop_size` - Hop size between frames
    ///
    /// # Returns
    ///
    /// Learned noise profile.
    pub fn learn(samples: &[f32], fft_size: usize, hop_size: usize) -> RestoreResult<Self> {
        if samples.len() < fft_size {
            return Err(RestoreError::NotEnoughData {
                needed: fft_size,
                have: samples.len(),
            });
        }

        let mut profile = Self::new(fft_size);
        let fft = FftProcessor::new(fft_size);

        // Process frames
        let mut pos = 0;
        while pos + fft_size <= samples.len() {
            let mut frame = samples[pos..pos + fft_size].to_vec();
            apply_window(&mut frame, WindowFunction::Hann);

            let spectrum = fft.forward(&frame)?;
            let magnitude = fft.magnitude(&spectrum);
            let power = fft.power(&spectrum);

            // Accumulate
            for (i, (&mag, &pow)) in magnitude.iter().zip(power.iter()).enumerate() {
                if i < profile.magnitude.len() {
                    profile.magnitude[i] += mag;
                    profile.power[i] += pow;
                }
            }

            profile.frame_count += 1;
            pos += hop_size;
        }

        if profile.frame_count == 0 {
            return Err(RestoreError::InvalidData("No frames processed".to_string()));
        }

        // Average
        #[allow(clippy::cast_precision_loss)]
        let scale = 1.0 / profile.frame_count as f32;
        for i in 0..profile.magnitude.len() {
            profile.magnitude[i] *= scale;
            profile.power[i] *= scale;
        }

        Ok(profile)
    }

    /// Update profile with new samples.
    pub fn update(&mut self, samples: &[f32], hop_size: usize) -> RestoreResult<()> {
        if samples.len() < self.fft_size {
            return Err(RestoreError::NotEnoughData {
                needed: self.fft_size,
                have: samples.len(),
            });
        }

        let fft = FftProcessor::new(self.fft_size);

        let mut pos = 0;
        while pos + self.fft_size <= samples.len() {
            let mut frame = samples[pos..pos + self.fft_size].to_vec();
            apply_window(&mut frame, WindowFunction::Hann);

            let spectrum = fft.forward(&frame)?;
            let magnitude = fft.magnitude(&spectrum);
            let power = fft.power(&spectrum);

            // Update with exponential averaging
            let alpha = 0.1; // Smoothing factor
            for (i, (&mag, &pow)) in magnitude.iter().zip(power.iter()).enumerate() {
                if i < self.magnitude.len() {
                    self.magnitude[i] = alpha * mag + (1.0 - alpha) * self.magnitude[i];
                    self.power[i] = alpha * pow + (1.0 - alpha) * self.power[i];
                }
            }

            self.frame_count += 1;
            pos += hop_size;
        }

        Ok(())
    }

    /// Estimate signal-to-noise ratio for a frame.
    ///
    /// # Arguments
    ///
    /// * `magnitude` - Frame magnitude spectrum
    ///
    /// # Returns
    ///
    /// SNR estimate in dB.
    #[must_use]
    pub fn estimate_snr(&self, magnitude: &[f32]) -> f32 {
        if magnitude.len() != self.magnitude.len() {
            return 0.0;
        }

        let mut signal_power = 0.0;
        let mut noise_power = 0.0;

        for (i, &mag) in magnitude.iter().enumerate() {
            let signal = mag.max(0.0);
            let noise = self.magnitude[i].max(1e-10);

            signal_power += signal * signal;
            noise_power += noise * noise;
        }

        if noise_power > f32::EPSILON {
            10.0 * (signal_power / noise_power).log10()
        } else {
            100.0
        }
    }
}

/// Detect silent regions in audio for noise profiling.
///
/// # Arguments
///
/// * `samples` - Input samples
/// * `threshold` - Energy threshold for silence detection
/// * `min_duration` - Minimum silent region duration in samples
///
/// # Returns
///
/// List of (start, end) indices of silent regions.
#[must_use]
pub fn detect_silent_regions(
    samples: &[f32],
    threshold: f32,
    min_duration: usize,
) -> Vec<(usize, usize)> {
    let mut regions = Vec::new();
    let mut in_silence = false;
    let mut silence_start = 0;

    for (i, &sample) in samples.iter().enumerate() {
        let is_silent = sample.abs() < threshold;

        if !in_silence && is_silent {
            // Start of silence
            in_silence = true;
            silence_start = i;
        } else if in_silence && !is_silent {
            // End of silence
            let duration = i - silence_start;
            if duration >= min_duration {
                regions.push((silence_start, i));
            }
            in_silence = false;
        }
    }

    // Handle silence at end
    if in_silence {
        let duration = samples.len() - silence_start;
        if duration >= min_duration {
            regions.push((silence_start, samples.len()));
        }
    }

    regions
}

/// Auto-learn noise profile from silent regions.
///
/// # Arguments
///
/// * `samples` - Input samples
/// * `fft_size` - FFT size
/// * `silence_threshold` - Threshold for silence detection
///
/// # Returns
///
/// Learned noise profile if silent regions found.
pub fn auto_learn_noise_profile(
    samples: &[f32],
    fft_size: usize,
    silence_threshold: f32,
) -> RestoreResult<Option<NoiseProfile>> {
    let min_duration = fft_size * 4; // Need at least a few frames
    let regions = detect_silent_regions(samples, silence_threshold, min_duration);

    if regions.is_empty() {
        return Ok(None);
    }

    // Use the longest silent region
    let longest_region = regions
        .iter()
        .max_by_key(|(start, end)| end - start)
        .copied();

    if let Some((start, end)) = longest_region {
        let noise_samples = &samples[start..end];
        let hop_size = fft_size / 2;
        let profile = NoiseProfile::learn(noise_samples, fft_size, hop_size)?;
        Ok(Some(profile))
    } else {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noise_profile_learn() {
        // Create noise signal
        use rand::RngExt;
        let mut rng = rand::rng();
        let samples: Vec<f32> = (0..8192).map(|_| rng.random_range(-0.1..0.1)).collect();

        let profile = NoiseProfile::learn(&samples, 2048, 1024).expect("should succeed in test");

        assert_eq!(profile.fft_size, 2048);
        assert!(!profile.magnitude.is_empty());
        assert!(profile.frame_count > 0);
    }

    #[test]
    fn test_noise_profile_update() {
        let mut profile = NoiseProfile::new(2048);

        use rand::RngExt;
        let mut rng = rand::rng();
        let samples: Vec<f32> = (0..4096).map(|_| rng.random_range(-0.1..0.1)).collect();

        profile
            .update(&samples, 1024)
            .expect("should succeed in test");
        assert!(profile.frame_count > 0);
    }

    #[test]
    fn test_detect_silent_regions() {
        let mut samples = vec![0.0; 1000];
        // Add some signal in the middle
        for i in 400..600 {
            samples[i] = 0.5;
        }

        let regions = detect_silent_regions(&samples, 0.1, 100);
        assert!(!regions.is_empty());
    }

    #[test]
    fn test_auto_learn_noise_profile() {
        let mut samples = vec![0.0; 20000];
        // Add quiet noise to first part
        use rand::RngExt;
        let mut rng = rand::rng();
        for i in 0..10000 {
            samples[i] = rng.random_range(-0.05..0.05);
        }
        // Add signal to second part
        for i in 10000..20000 {
            samples[i] = 0.5;
        }

        // Use higher threshold since noise can spike to 0.05
        let profile =
            auto_learn_noise_profile(&samples, 2048, 0.15).expect("should succeed in test");
        assert!(profile.is_some());
    }

    #[test]
    fn test_estimate_snr() {
        let profile = NoiseProfile::new(1025);
        let magnitude = vec![1.0; 1025];

        let snr = profile.estimate_snr(&magnitude);
        assert!(snr.is_finite());
    }
}
