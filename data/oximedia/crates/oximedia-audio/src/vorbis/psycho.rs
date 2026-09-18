//! Psychoacoustic model for Vorbis encoding.
//!
//! The psychoacoustic model analyzes the audio signal to determine
//! masking thresholds and allocate bits efficiently based on
//! perceptual importance.

#![forbid(unsafe_code)]

/// Psychoacoustic model for Vorbis.
#[derive(Debug, Clone)]
pub struct PsychoModel {
    /// Sample rate.
    sample_rate: u32,
    /// Block size.
    block_size: usize,
    /// Bark scale mapping.
    bark_scale: Vec<f32>,
    /// Masking thresholds.
    masking_thresholds: Vec<f32>,
}

impl PsychoModel {
    /// Create new psychoacoustic model.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Audio sample rate in Hz
    /// * `block_size` - MDCT block size
    #[must_use]
    pub fn new(sample_rate: u32, block_size: usize) -> Self {
        let bark_scale = Self::compute_bark_scale(sample_rate, block_size);
        let masking_thresholds = vec![0.0; block_size / 2];

        Self {
            sample_rate,
            block_size,
            bark_scale,
            masking_thresholds,
        }
    }

    /// Compute Bark scale for frequency bins.
    ///
    /// The Bark scale is a psychoacoustic frequency scale where each Bark
    /// represents a critical band of human hearing.
    #[allow(clippy::cast_precision_loss)]
    fn compute_bark_scale(sample_rate: u32, block_size: usize) -> Vec<f32> {
        let n = block_size / 2;
        let mut bark = Vec::with_capacity(n);

        for i in 0..n {
            let freq = i as f32 * sample_rate as f32 / block_size as f32;
            let bark_value = Self::freq_to_bark(freq);
            bark.push(bark_value);
        }

        bark
    }

    /// Convert frequency (Hz) to Bark scale.
    ///
    /// Uses Traunmüller's formula: Bark = 26.81 * f / (1960 + f) - 0.53
    #[allow(clippy::cast_precision_loss)]
    fn freq_to_bark(freq: f32) -> f32 {
        26.81 * freq / (1960.0 + freq) - 0.53
    }

    /// Convert Bark scale to frequency (Hz).
    #[allow(dead_code)]
    fn bark_to_freq(bark: f32) -> f32 {
        1960.0 * (bark + 0.53) / (26.28 - bark)
    }

    /// Compute spreading function for masking.
    ///
    /// The spreading function models how energy in one frequency bin
    /// masks nearby frequencies.
    fn spreading_function(delta_bark: f32) -> f32 {
        let abs_delta = delta_bark.abs();
        if abs_delta < 1.0 {
            // Close frequencies: strong masking
            -6.025 - 0.275 * delta_bark
        } else if abs_delta < 3.0 {
            // Medium distance: moderate masking
            -17.0 - 0.4 * abs_delta + 11.0 * (1.0 - abs_delta / 3.0)
        } else {
            // Far frequencies: weak masking
            -100.0
        }
    }

    /// Analyze MDCT coefficients and compute masking thresholds.
    ///
    /// # Arguments
    ///
    /// * `coeffs` - MDCT coefficients
    ///
    /// # Returns
    ///
    /// Masking threshold for each frequency bin (in dB).
    #[allow(clippy::cast_precision_loss)]
    pub fn analyze(&mut self, coeffs: &[f32]) -> &[f32] {
        let n = coeffs.len().min(self.block_size / 2);

        // Compute power spectrum (in dB)
        let mut power_db = vec![0.0; n];
        for (i, &coeff) in coeffs.iter().enumerate().take(n) {
            let power = coeff * coeff;
            power_db[i] = if power > 1e-10 {
                10.0 * power.log10()
            } else {
                -100.0
            };
        }

        // Compute masking thresholds using spreading function
        for i in 0..n {
            let mut threshold: f32 = -100.0; // Start with very low threshold

            for j in 0..n {
                let delta_bark = self.bark_scale[i] - self.bark_scale[j];
                let spreading = Self::spreading_function(delta_bark);
                let masked_level = power_db[j] + spreading;
                threshold = threshold.max(masked_level);
            }

            // Add absolute threshold of hearing (ATH)
            let freq = i as f32 * self.sample_rate as f32 / self.block_size as f32;
            let ath = Self::absolute_threshold(freq);
            threshold = threshold.max(ath);

            self.masking_thresholds[i] = threshold;
        }

        &self.masking_thresholds
    }

    /// Compute absolute threshold of hearing.
    ///
    /// The ATH represents the quietest sound that can be heard at each frequency.
    /// Uses simplified ATH curve.
    #[allow(clippy::cast_precision_loss)]
    fn absolute_threshold(freq: f32) -> f32 {
        if freq < 1.0 {
            return -10.0;
        }

        let f_khz = freq / 1000.0;
        3.64 * f_khz.powf(-0.8) - 6.5 * (-0.6 * (f_khz - 3.3).powi(2)).exp() + 1e-3 * f_khz.powi(4)
    }

    /// Compute signal-to-mask ratio (SMR).
    ///
    /// # Arguments
    ///
    /// * `coeffs` - MDCT coefficients
    ///
    /// # Returns
    ///
    /// SMR for each frequency bin (in dB).
    #[allow(clippy::cast_precision_loss)]
    pub fn compute_smr(&self, coeffs: &[f32]) -> Vec<f32> {
        let n = coeffs.len().min(self.masking_thresholds.len());
        let mut smr = Vec::with_capacity(n);

        for i in 0..n {
            let power = coeffs[i] * coeffs[i];
            let signal_db = if power > 1e-10 {
                10.0 * power.log10()
            } else {
                -100.0
            };

            let ratio = signal_db - self.masking_thresholds[i];
            smr.push(ratio);
        }

        smr
    }

    /// Detect transients in the audio signal.
    ///
    /// Transients are rapid changes in signal energy that require
    /// shorter block sizes for accurate encoding.
    ///
    /// # Arguments
    ///
    /// * `samples` - Time-domain audio samples
    ///
    /// # Returns
    ///
    /// `true` if a transient is detected.
    #[allow(clippy::cast_precision_loss)]
    pub fn detect_transient(&self, samples: &[f32]) -> bool {
        if samples.len() < 128 {
            return false;
        }

        // Divide into segments and compute energy
        let segment_size = 64;
        let num_segments = samples.len() / segment_size;

        let mut energies = Vec::with_capacity(num_segments);
        for i in 0..num_segments {
            let start = i * segment_size;
            let end = (start + segment_size).min(samples.len());
            let energy: f32 = samples[start..end].iter().map(|x| x * x).sum();
            energies.push(energy);
        }

        // Check for rapid energy increase
        for i in 1..energies.len() {
            if energies[i] > energies[i - 1] * 4.0 {
                return true;
            }
        }

        false
    }

    /// Compute tonality measure.
    ///
    /// Tonality indicates whether the signal is tonal (like a pure tone)
    /// or noisy. Tonal signals can be quantized more coarsely.
    ///
    /// # Arguments
    ///
    /// * `coeffs` - MDCT coefficients
    ///
    /// # Returns
    ///
    /// Tonality measure for each bin (0.0 = noise, 1.0 = pure tone).
    #[allow(clippy::cast_precision_loss)]
    pub fn compute_tonality(&self, coeffs: &[f32]) -> Vec<f32> {
        let n = coeffs.len().min(self.block_size / 2);
        let mut tonality = Vec::with_capacity(n);

        for i in 0..n {
            // Simplified tonality: compare magnitude to neighbors
            let mag = coeffs[i].abs();
            let prev_mag = if i > 0 { coeffs[i - 1].abs() } else { 0.0 };
            let next_mag = if i + 1 < n { coeffs[i + 1].abs() } else { 0.0 };

            let neighbor_avg = (prev_mag + next_mag) / 2.0;
            let t = if neighbor_avg > 1e-10 {
                (mag / neighbor_avg).min(1.0)
            } else {
                0.0
            };

            tonality.push(t);
        }

        tonality
    }

    /// Get sample rate.
    #[must_use]
    pub const fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Get block size.
    #[must_use]
    pub const fn block_size(&self) -> usize {
        self.block_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_psycho_model_creation() {
        let model = PsychoModel::new(44100, 2048);
        assert_eq!(model.sample_rate(), 44100);
        assert_eq!(model.block_size(), 2048);
    }

    #[test]
    fn test_freq_to_bark() {
        let bark_100 = PsychoModel::freq_to_bark(100.0);
        let bark_1000 = PsychoModel::freq_to_bark(1000.0);
        let bark_10000 = PsychoModel::freq_to_bark(10000.0);

        // Higher frequencies should have higher Bark values
        assert!(bark_1000 > bark_100);
        assert!(bark_10000 > bark_1000);
    }

    #[test]
    fn test_spreading_function() {
        let s0 = PsychoModel::spreading_function(0.0);
        let s1 = PsychoModel::spreading_function(1.0);
        let s3 = PsychoModel::spreading_function(3.0);

        // Masking should decrease with distance
        assert!(s0 > s1);
        assert!(s1 > s3);
    }

    #[test]
    fn test_absolute_threshold() {
        let ath_100 = PsychoModel::absolute_threshold(100.0);
        let ath_1000 = PsychoModel::absolute_threshold(1000.0);
        let ath_10000 = PsychoModel::absolute_threshold(10000.0);

        // All thresholds should be finite
        assert!(ath_100.is_finite());
        assert!(ath_1000.is_finite());
        assert!(ath_10000.is_finite());
    }

    #[test]
    fn test_analyze() {
        let mut model = PsychoModel::new(44100, 2048);
        let coeffs = vec![1.0; 1024];

        let thresholds = model.analyze(&coeffs);
        assert_eq!(thresholds.len(), 1024);

        // All thresholds should be finite
        assert!(thresholds.iter().all(|&t| t.is_finite()));
    }

    #[test]
    fn test_compute_smr() {
        let mut model = PsychoModel::new(44100, 2048);
        let coeffs = vec![1.0; 1024];

        model.analyze(&coeffs);
        let smr = model.compute_smr(&coeffs);

        assert_eq!(smr.len(), 1024);
        assert!(smr.iter().all(|&s| s.is_finite()));
    }

    #[test]
    fn test_detect_transient() {
        let model = PsychoModel::new(44100, 2048);

        // Constant signal - no transient
        let constant = vec![1.0; 256];
        assert!(!model.detect_transient(&constant));

        // Signal with sudden increase - transient
        let mut transient = vec![0.1; 256];
        for i in 128..256 {
            transient[i] = 2.0;
        }
        assert!(model.detect_transient(&transient));
    }

    #[test]
    fn test_compute_tonality() {
        let model = PsychoModel::new(44100, 2048);

        // Create a tonal signal (peak at one frequency)
        let mut coeffs = vec![0.1; 512];
        coeffs[100] = 10.0; // Strong peak

        let tonality = model.compute_tonality(&coeffs);
        assert_eq!(tonality.len(), 512);

        // Peak should have high tonality
        assert!(tonality[100] > 0.5);
    }
}
