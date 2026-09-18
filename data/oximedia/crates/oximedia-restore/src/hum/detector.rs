//! Hum frequency detection.

use crate::error::RestoreResult;
use crate::utils::spectral::{apply_window, find_peaks, FftProcessor, WindowFunction};

/// Detected hum frequencies.
#[derive(Debug, Clone)]
pub struct HumFrequencies {
    /// Fundamental frequency (50 or 60 Hz).
    pub fundamental: f32,
    /// Detected harmonics.
    pub harmonics: Vec<f32>,
    /// Confidence score (0.0 to 1.0).
    pub confidence: f32,
}

/// Hum detector configuration.
#[derive(Debug, Clone)]
pub struct HumDetectorConfig {
    /// Expected fundamental frequencies to search (e.g., [50.0, 60.0]).
    pub expected_fundamentals: Vec<f32>,
    /// Maximum number of harmonics to detect.
    pub max_harmonics: usize,
    /// Frequency tolerance in Hz.
    pub frequency_tolerance: f32,
    /// Minimum peak magnitude.
    pub min_peak_magnitude: f32,
}

impl Default for HumDetectorConfig {
    fn default() -> Self {
        Self {
            expected_fundamentals: vec![50.0, 60.0],
            max_harmonics: 10,
            frequency_tolerance: 2.0,
            min_peak_magnitude: 0.001,
        }
    }
}

/// Hum detector.
#[derive(Debug)]
pub struct HumDetector {
    config: HumDetectorConfig,
    fft_size: usize,
}

impl HumDetector {
    /// Create a new hum detector.
    ///
    /// # Arguments
    ///
    /// * `config` - Detector configuration
    /// * `fft_size` - FFT size for analysis
    #[must_use]
    pub fn new(config: HumDetectorConfig, fft_size: usize) -> Self {
        Self { config, fft_size }
    }

    /// Detect hum frequencies in samples.
    ///
    /// # Arguments
    ///
    /// * `samples` - Input samples
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Returns
    ///
    /// Detected hum frequencies.
    pub fn detect(
        &self,
        samples: &[f32],
        sample_rate: u32,
    ) -> RestoreResult<Option<HumFrequencies>> {
        if samples.len() < self.fft_size {
            return Ok(None);
        }

        // Take first fft_size samples and apply window
        let mut windowed = samples[..self.fft_size].to_vec();
        apply_window(&mut windowed, WindowFunction::BlackmanHarris);

        // Perform FFT
        let fft = FftProcessor::new(self.fft_size);
        let spectrum = fft.forward(&windowed)?;
        let magnitude = fft.magnitude(&spectrum);

        // Find peaks in spectrum
        let peaks = find_peaks(&magnitude, self.config.min_peak_magnitude, 5);

        // Convert bin indices to frequencies
        #[allow(clippy::cast_precision_loss)]
        let bin_to_freq =
            |bin: usize| -> f32 { bin as f32 * sample_rate as f32 / self.fft_size as f32 };

        // Try each expected fundamental
        let mut best_match: Option<HumFrequencies> = None;
        let mut best_score = 0.0;

        for &fundamental in &self.config.expected_fundamentals {
            let (harmonics, score) = self.find_harmonics(&peaks, bin_to_freq, fundamental);

            if score > best_score {
                best_score = score;
                best_match = Some(HumFrequencies {
                    fundamental,
                    harmonics,
                    confidence: score,
                });
            }
        }

        Ok(best_match)
    }

    /// Find harmonics for a given fundamental frequency.
    fn find_harmonics(
        &self,
        peaks: &[usize],
        bin_to_freq: impl Fn(usize) -> f32,
        fundamental: f32,
    ) -> (Vec<f32>, f32) {
        let mut harmonics = Vec::new();
        let mut matched_count = 0;

        for n in 1..=self.config.max_harmonics {
            #[allow(clippy::cast_precision_loss)]
            let expected_freq = fundamental * n as f32;

            // Find peak closest to expected harmonic frequency
            let closest_peak = peaks
                .iter()
                .map(|&peak_bin| {
                    let freq = bin_to_freq(peak_bin);
                    (peak_bin, (freq - expected_freq).abs())
                })
                .min_by(|(_, dist_a), (_, dist_b)| {
                    dist_a
                        .partial_cmp(dist_b)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

            if let Some((peak_bin, distance)) = closest_peak {
                if distance < self.config.frequency_tolerance {
                    let freq = bin_to_freq(peak_bin);
                    harmonics.push(freq);
                    matched_count += 1;
                }
            }
        }

        // Compute confidence score based on matched harmonics
        #[allow(clippy::cast_precision_loss)]
        let score = if !harmonics.is_empty() {
            matched_count as f32 / self.config.max_harmonics.min(5) as f32
        } else {
            0.0
        };

        (harmonics, score)
    }
}

/// Detect hum using autocorrelation.
///
/// # Arguments
///
/// * `samples` - Input samples
/// * `sample_rate` - Sample rate in Hz
/// * `min_freq` - Minimum frequency to detect (e.g., 45 Hz)
/// * `max_freq` - Maximum frequency to detect (e.g., 65 Hz)
///
/// # Returns
///
/// Detected fundamental frequency if found.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn detect_hum_autocorrelation(
    samples: &[f32],
    sample_rate: u32,
    min_freq: f32,
    max_freq: f32,
) -> Option<f32> {
    if samples.len() < 2 {
        return None;
    }

    let min_period = (sample_rate as f32 / max_freq) as usize;
    let max_period = (sample_rate as f32 / min_freq) as usize;

    if max_period >= samples.len() {
        return None;
    }

    // Compute autocorrelation
    let mut max_corr = 0.0;
    let mut best_period = min_period;

    for period in min_period..=max_period.min(samples.len() - 1) {
        let mut corr = 0.0;
        let mut energy = 0.0;

        for i in 0..samples.len() - period {
            corr += samples[i] * samples[i + period];
            energy += samples[i] * samples[i];
        }

        if energy > f32::EPSILON {
            corr /= energy;

            if corr > max_corr {
                max_corr = corr;
                best_period = period;
            }
        }
    }

    if max_corr > 0.5 {
        Some(sample_rate as f32 / best_period as f32)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hum_detector() {
        use std::f32::consts::PI;

        // Create signal with 50 Hz hum and harmonics
        let sample_rate = 44100;
        let duration = 1.0;
        let n_samples = (sample_rate as f32 * duration) as usize;

        let mut samples = vec![0.0; n_samples];
        for i in 0..n_samples {
            let t = i as f32 / sample_rate as f32;
            // 50 Hz fundamental + 100 Hz and 150 Hz harmonics
            samples[i] = (2.0 * PI * 50.0 * t).sin()
                + 0.5 * (2.0 * PI * 100.0 * t).sin()
                + 0.3 * (2.0 * PI * 150.0 * t).sin();
        }

        let detector = HumDetector::new(HumDetectorConfig::default(), 8192);
        let result = detector
            .detect(&samples, sample_rate)
            .expect("should succeed in test");

        assert!(result.is_some());
        if let Some(hum) = result {
            assert!((hum.fundamental - 50.0).abs() < 2.0);
            assert!(!hum.harmonics.is_empty());
        }
    }

    #[test]
    fn test_detect_hum_autocorrelation() {
        use std::f32::consts::PI;

        let sample_rate = 44100;
        let mut samples = vec![0.0; sample_rate as usize];

        for i in 0..samples.len() {
            let t = i as f32 / sample_rate as f32;
            samples[i] = (2.0 * PI * 50.0 * t).sin();
        }

        let freq = detect_hum_autocorrelation(&samples, sample_rate, 45.0, 55.0);
        assert!(freq.is_some());
        if let Some(f) = freq {
            assert!((f - 50.0).abs() < 5.0);
        }
    }

    #[test]
    fn test_config_default() {
        let config = HumDetectorConfig::default();
        assert!(config.expected_fundamentals.contains(&50.0));
        assert!(config.expected_fundamentals.contains(&60.0));
    }
}
