//! Psychoacoustic masking model.
//!
//! This module implements psychoacoustic masking threshold calculation
//! to ensure watermark imperceptibility based on human hearing properties.

use oxifft::Complex;
use std::f32::consts::PI;

/// Psychoacoustic model for masking threshold calculation.
pub struct PsychoacousticModel {
    sample_rate: u32,
    frame_size: usize,
    bark_bands: Vec<BarkBand>,
}

/// Bark scale frequency band.
struct BarkBand {
    index: usize,
    center_freq: f32,
    lower_freq: f32,
    upper_freq: f32,
}

impl PsychoacousticModel {
    /// Create a new psychoacoustic model.
    #[must_use]
    pub fn new(sample_rate: u32, frame_size: usize) -> Self {
        let bark_bands = Self::init_bark_bands(sample_rate, frame_size);

        Self {
            sample_rate,
            frame_size,
            bark_bands,
        }
    }

    /// Calculate masking threshold for audio frame.
    #[must_use]
    pub fn calculate_masking_threshold(&self, samples: &[f32]) -> Vec<f32> {
        // Apply Hann window
        let windowed = self.apply_hann_window(samples);

        // Compute magnitude spectrum
        let spectrum = self.compute_spectrum(&windowed);

        // Group into bark bands
        let bark_energies = self.compute_bark_energies(&spectrum);

        // Calculate masking threshold per bark band
        let masking = self.calculate_masking(&bark_energies);

        // Map back to frequency bins
        self.map_to_frequency_bins(&masking)
    }

    /// Apply Hann window to samples.
    fn apply_hann_window(&self, samples: &[f32]) -> Vec<f32> {
        samples
            .iter()
            .enumerate()
            .map(|(i, &s)| {
                #[allow(clippy::cast_precision_loss)]
                let window = 0.5 * (1.0 - (2.0 * PI * i as f32 / self.frame_size as f32).cos());
                s * window
            })
            .collect()
    }

    /// Compute magnitude spectrum using FFT.
    fn compute_spectrum(&self, samples: &[f32]) -> Vec<f32> {
        let buffer: Vec<Complex<f32>> = samples.iter().map(|&s| Complex::new(s, 0.0)).collect();

        let fft_result = oxifft::fft(&buffer);

        // Return magnitude spectrum (first half due to symmetry)
        fft_result[..self.frame_size / 2]
            .iter()
            .map(|c| c.norm())
            .collect()
    }

    /// Compute energy in each bark band.
    fn compute_bark_energies(&self, spectrum: &[f32]) -> Vec<f32> {
        let mut energies = vec![0.0f32; self.bark_bands.len()];

        for (i, &mag) in spectrum.iter().enumerate() {
            #[allow(clippy::cast_precision_loss)]
            let freq = i as f32 * self.sample_rate as f32 / self.frame_size as f32;

            // Find bark band for this frequency
            if let Some(band_idx) = self.find_bark_band(freq) {
                energies[band_idx] += mag * mag;
            }
        }

        // Convert to power in dB
        energies
            .iter()
            .map(|&e| if e > 1e-10 { 10.0 * e.log10() } else { -100.0 })
            .collect()
    }

    /// Calculate masking threshold using spreading function.
    fn calculate_masking(&self, bark_energies: &[f32]) -> Vec<f32> {
        let mut masking = vec![0.0f32; bark_energies.len()];

        for (i, &energy) in bark_energies.iter().enumerate() {
            // Apply spreading function to neighboring bands
            for (j, mask) in masking.iter_mut().enumerate() {
                #[allow(clippy::cast_precision_loss)]
                let bark_dist = (i as f32 - j as f32).abs();
                let spread = Self::spreading_function(bark_dist);
                *mask = mask.max(energy + spread);
            }
        }

        // Add absolute threshold of hearing
        for (i, mask) in masking.iter_mut().enumerate() {
            let freq = self.bark_bands[i].center_freq;
            let abs_threshold = Self::absolute_threshold(freq);
            *mask = mask.max(abs_threshold);
        }

        masking
    }

    /// Spreading function for masking.
    fn spreading_function(bark_distance: f32) -> f32 {
        if bark_distance < 1.0 {
            -25.0 * bark_distance
        } else {
            -10.0 * bark_distance - 15.0
        }
    }

    /// Absolute threshold of hearing (in dB SPL).
    fn absolute_threshold(freq_hz: f32) -> f32 {
        let f_khz = freq_hz / 1000.0;
        3.64 * f_khz.powf(-0.8) - 6.5 * (-0.6 * (f_khz - 3.3).powi(2)).exp() + 1e-3 * f_khz.powi(4)
            - 90.0 // Normalize to reasonable range
    }

    /// Map bark band masking to frequency bins.
    fn map_to_frequency_bins(&self, bark_masking: &[f32]) -> Vec<f32> {
        let mut threshold = vec![0.0f32; self.frame_size / 2];

        for (i, thresh) in threshold.iter_mut().enumerate() {
            #[allow(clippy::cast_precision_loss)]
            let freq = i as f32 * self.sample_rate as f32 / self.frame_size as f32;

            if let Some(band_idx) = self.find_bark_band(freq) {
                *thresh = bark_masking[band_idx];
            }
        }

        threshold
    }

    /// Find bark band index for given frequency.
    fn find_bark_band(&self, freq: f32) -> Option<usize> {
        self.bark_bands
            .iter()
            .find(|band| freq >= band.lower_freq && freq < band.upper_freq)
            .map(|band| band.index)
    }

    /// Initialize bark scale frequency bands.
    fn init_bark_bands(sample_rate: u32, _frame_size: usize) -> Vec<BarkBand> {
        let mut bands = Vec::new();
        #[allow(clippy::cast_precision_loss)]
        let nyquist = sample_rate as f32 / 2.0;

        // Critical band approximation (simplified)
        let critical_freqs = [
            20.0, 100.0, 200.0, 300.0, 400.0, 510.0, 630.0, 770.0, 920.0, 1080.0, 1270.0, 1480.0,
            1720.0, 2000.0, 2320.0, 2700.0, 3150.0, 3700.0, 4400.0, 5300.0, 6400.0, 7700.0, 9500.0,
            12000.0, 15500.0,
        ];

        for (i, window) in critical_freqs.windows(2).enumerate() {
            if window[0] >= nyquist {
                break;
            }

            let lower = window[0];
            let upper = window[1].min(nyquist);
            let center = (lower + upper) / 2.0;

            bands.push(BarkBand {
                index: i,
                center_freq: center,
                lower_freq: lower,
                upper_freq: upper,
            });
        }

        bands
    }
}

/// Calculate Signal-to-Mask Ratio (SMR) for watermark strength.
#[must_use]
pub fn calculate_smr(signal_db: f32, mask_db: f32) -> f32 {
    signal_db - mask_db
}

/// Convert linear amplitude to dB.
#[must_use]
pub fn amplitude_to_db(amplitude: f32) -> f32 {
    if amplitude > 1e-10 {
        20.0 * amplitude.log10()
    } else {
        -100.0
    }
}

/// Convert dB to linear amplitude.
#[must_use]
pub fn db_to_amplitude(db: f32) -> f32 {
    10.0f32.powf(db / 20.0)
}

/// Frequency to Bark scale conversion.
#[must_use]
pub fn freq_to_bark(freq: f32) -> f32 {
    13.0 * (0.00076 * freq).atan() + 3.5 * ((freq / 7500.0).powi(2)).atan()
}

/// Bark to frequency conversion.
#[must_use]
pub fn bark_to_freq(bark: f32) -> f32 {
    // Approximation
    1960.0 * (bark + 0.53) / (26.28 - bark)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_psychoacoustic_model() {
        let model = PsychoacousticModel::new(44100, 1024);
        let samples: Vec<f32> = (0..1024)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                (2.0 * PI * 1000.0 * i as f32 / 44100.0).sin()
            })
            .collect();

        let threshold = model.calculate_masking_threshold(&samples);
        assert_eq!(threshold.len(), 512);
        assert!(threshold.iter().any(|&t| t > -100.0));
    }

    #[test]
    fn test_db_conversion() {
        let amp = 0.5f32;
        let db = amplitude_to_db(amp);
        let amp2 = db_to_amplitude(db);
        assert!((amp - amp2).abs() < 1e-6);
    }

    #[test]
    fn test_bark_conversion() {
        let freq = 1000.0f32;
        let bark = freq_to_bark(freq);
        assert!(bark > 0.0 && bark < 25.0);
    }

    #[test]
    fn test_spreading_function() {
        let spread_0 = PsychoacousticModel::spreading_function(0.0);
        let spread_1 = PsychoacousticModel::spreading_function(1.0);
        let spread_2 = PsychoacousticModel::spreading_function(2.0);

        assert!(spread_0 > spread_1);
        assert!(spread_1 > spread_2);
    }
}
