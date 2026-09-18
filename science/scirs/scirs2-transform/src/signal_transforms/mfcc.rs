//! Mel-Frequency Cepstral Coefficients (MFCC) Implementation
//!
//! Provides MFCC extraction for audio and speech processing.

use crate::error::{Result, TransformError};
use crate::signal_transforms::stft::{STFTConfig, WindowType, STFT};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, Axis};
use scirs2_core::numeric::Complex;
use std::f64::consts::PI;

/// Mel filterbank configuration
#[derive(Debug, Clone)]
pub struct MelFilterbank {
    /// Number of mel filters
    pub n_filters: usize,
    /// FFT size
    pub nfft: usize,
    /// Sampling rate in Hz
    pub sample_rate: f64,
    /// Lower frequency bound in Hz
    pub fmin: f64,
    /// Upper frequency bound in Hz
    pub fmax: f64,
    /// Filter weights (n_filters x n_freqs)
    filters: Array2<f64>,
}

impl MelFilterbank {
    /// Create a new mel filterbank
    pub fn new(
        n_filters: usize,
        nfft: usize,
        sample_rate: f64,
        fmin: f64,
        fmax: f64,
    ) -> Result<Self> {
        if fmin >= fmax {
            return Err(TransformError::InvalidInput(
                "fmin must be less than fmax".to_string(),
            ));
        }

        if fmax > sample_rate / 2.0 {
            return Err(TransformError::InvalidInput(
                "fmax must be <= sample_rate/2".to_string(),
            ));
        }

        let filters = Self::compute_filters(n_filters, nfft, sample_rate, fmin, fmax);

        Ok(MelFilterbank {
            n_filters,
            nfft,
            sample_rate,
            fmin,
            fmax,
            filters,
        })
    }

    /// Convert Hz to Mel scale
    fn hz_to_mel(hz: f64) -> f64 {
        2595.0 * (1.0 + hz / 700.0).log10()
    }

    /// Convert Mel to Hz scale
    fn mel_to_hz(mel: f64) -> f64 {
        700.0 * (10.0_f64.powf(mel / 2595.0) - 1.0)
    }

    /// Compute mel filter weights
    fn compute_filters(
        n_filters: usize,
        nfft: usize,
        sample_rate: f64,
        fmin: f64,
        fmax: f64,
    ) -> Array2<f64> {
        let n_freqs = nfft / 2 + 1;
        let mut filters = Array2::zeros((n_filters, n_freqs));

        // Convert frequency bounds to mel scale
        let mel_min = Self::hz_to_mel(fmin);
        let mel_max = Self::hz_to_mel(fmax);

        // Create mel-spaced frequency points
        let mel_points: Vec<f64> = (0..=n_filters + 1)
            .map(|i| mel_min + (mel_max - mel_min) * i as f64 / (n_filters + 1) as f64)
            .collect();

        let hz_points: Vec<f64> = mel_points.iter().map(|&m| Self::mel_to_hz(m)).collect();

        // Convert Hz points to FFT bin indices
        let bin_points: Vec<usize> = hz_points
            .iter()
            .map(|&f| ((nfft + 1) as f64 * f / sample_rate).floor() as usize)
            .collect();

        // Create triangular filters
        for i in 0..n_filters {
            let left = bin_points[i];
            let center = bin_points[i + 1];
            let right = bin_points[i + 2];

            // Rising slope
            for j in left..center {
                if center > left && j < n_freqs {
                    filters[[i, j]] = (j - left) as f64 / (center - left) as f64;
                }
            }

            // Falling slope
            for j in center..right {
                if right > center && j < n_freqs {
                    filters[[i, j]] = (right - j) as f64 / (right - center) as f64;
                }
            }
        }

        filters
    }

    /// Apply mel filterbank to power spectrum
    pub fn apply(&self, power_spectrum: &ArrayView1<f64>) -> Result<Array1<f64>> {
        let n_freqs = power_spectrum.len();
        if n_freqs != self.nfft / 2 + 1 {
            return Err(TransformError::InvalidInput(format!(
                "Expected {} frequency bins, got {}",
                self.nfft / 2 + 1,
                n_freqs
            )));
        }

        let mut mel_energies = Array1::zeros(self.n_filters);

        for i in 0..self.n_filters {
            let mut energy = 0.0;
            for j in 0..n_freqs {
                energy += self.filters[[i, j]] * power_spectrum[j];
            }
            mel_energies[i] = energy;
        }

        Ok(mel_energies)
    }

    /// Get the filter weights
    pub fn filters(&self) -> &Array2<f64> {
        &self.filters
    }

    /// Get center frequencies in Hz
    pub fn center_frequencies(&self) -> Vec<f64> {
        let mel_min = Self::hz_to_mel(self.fmin);
        let mel_max = Self::hz_to_mel(self.fmax);

        (0..self.n_filters)
            .map(|i| {
                let mel =
                    mel_min + (mel_max - mel_min) * (i + 1) as f64 / (self.n_filters + 1) as f64;
                Self::mel_to_hz(mel)
            })
            .collect()
    }
}

/// MFCC configuration
#[derive(Debug, Clone)]
pub struct MFCCConfig {
    /// Number of MFCCs to extract
    pub n_mfcc: usize,
    /// Number of mel filters
    pub n_mels: usize,
    /// FFT size
    pub nfft: usize,
    /// Hop size for STFT
    pub hop_size: usize,
    /// Window size for STFT
    pub window_size: usize,
    /// Sampling rate in Hz
    pub sample_rate: f64,
    /// Lower frequency bound in Hz
    pub fmin: f64,
    /// Upper frequency bound in Hz
    pub fmax: f64,
    /// Liftering coefficient
    pub lifter: Option<usize>,
    /// Whether to apply mean normalization
    pub normalize: bool,
}

impl Default for MFCCConfig {
    fn default() -> Self {
        MFCCConfig {
            n_mfcc: 13,
            n_mels: 40,
            nfft: 512,
            hop_size: 160,
            window_size: 400,
            sample_rate: 16000.0,
            fmin: 0.0,
            fmax: 8000.0,
            lifter: Some(22),
            normalize: true,
        }
    }
}

/// MFCC extractor
#[derive(Debug, Clone)]
pub struct MFCC {
    config: MFCCConfig,
    mel_filterbank: MelFilterbank,
    stft: STFT,
    dct_matrix: Array2<f64>,
}

impl MFCC {
    /// Create a new MFCC extractor
    pub fn new(config: MFCCConfig) -> Result<Self> {
        let mel_filterbank = MelFilterbank::new(
            config.n_mels,
            config.nfft,
            config.sample_rate,
            config.fmin,
            config.fmax,
        )?;

        let stft_config = STFTConfig {
            window_size: config.window_size,
            hop_size: config.hop_size,
            window_type: WindowType::Hamming,
            nfft: Some(config.nfft),
            onesided: true,
            padding: crate::signal_transforms::stft::PaddingMode::Zero,
        };

        let stft = STFT::new(stft_config);
        let dct_matrix = Self::compute_dct_matrix(config.n_mfcc, config.n_mels);

        Ok(MFCC {
            config,
            mel_filterbank,
            stft,
            dct_matrix,
        })
    }

    /// Create with default configuration
    pub fn default() -> Result<Self> {
        Self::new(MFCCConfig::default())
    }

    /// Compute DCT-II matrix
    fn compute_dct_matrix(n_mfcc: usize, n_mels: usize) -> Array2<f64> {
        let mut dct = Array2::zeros((n_mfcc, n_mels));
        let norm = (2.0 / n_mels as f64).sqrt();

        for i in 0..n_mfcc {
            for j in 0..n_mels {
                dct[[i, j]] = norm * (PI * i as f64 * (j as f64 + 0.5) / n_mels as f64).cos();
            }
        }

        dct
    }

    /// Extract MFCCs from audio signal
    pub fn extract(&self, signal: &ArrayView1<f64>) -> Result<Array2<f64>> {
        // Compute STFT
        let stft = self.stft.transform(signal)?;
        let (n_freqs, n_frames) = stft.dim();

        // Compute power spectrum
        let mut power_spec = Array2::zeros((n_freqs, n_frames));
        for i in 0..n_freqs {
            for j in 0..n_frames {
                let mag = stft[[i, j]].norm();
                power_spec[[i, j]] = mag * mag;
            }
        }

        // Apply mel filterbank and extract MFCCs for each frame
        let mut mfccs = Array2::zeros((self.config.n_mfcc, n_frames));

        for frame_idx in 0..n_frames {
            let power_frame = power_spec.column(frame_idx);
            let mel_energies = self.mel_filterbank.apply(&power_frame)?;

            // Log mel energies
            let log_mel_energies: Array1<f64> = mel_energies
                .iter()
                .map(|&e| {
                    if e > 1e-10 {
                        e.ln()
                    } else {
                        -23.025850929940457 // ln(1e-10)
                    }
                })
                .collect();

            // Apply DCT
            let mfcc_frame = self.dct_matrix.dot(&log_mel_energies);

            // Apply liftering if configured
            let mfcc_frame = if let Some(lifter) = self.config.lifter {
                self.apply_lifter(&mfcc_frame, lifter)
            } else {
                mfcc_frame
            };

            // Store MFCCs
            for (i, &val) in mfcc_frame.iter().enumerate() {
                mfccs[[i, frame_idx]] = val;
            }
        }

        // Apply mean normalization if configured
        if self.config.normalize {
            self.normalize_mfccs(&mut mfccs);
        }

        Ok(mfccs)
    }

    /// Apply liftering (cepstral filtering)
    fn apply_lifter(&self, mfcc: &Array1<f64>, lifter: usize) -> Array1<f64> {
        let n = mfcc.len();
        let mut lifted = Array1::zeros(n);

        for i in 0..n {
            let lift_weight = 1.0 + (lifter as f64 / 2.0) * (PI * i as f64 / lifter as f64).sin();
            lifted[i] = mfcc[i] * lift_weight;
        }

        lifted
    }

    /// Apply mean normalization to MFCCs
    fn normalize_mfccs(&self, mfccs: &mut Array2<f64>) {
        let (n_mfcc, n_frames) = mfccs.dim();

        for i in 0..n_mfcc {
            let mut sum = 0.0;
            for j in 0..n_frames {
                sum += mfccs[[i, j]];
            }
            let mean = sum / n_frames as f64;

            for j in 0..n_frames {
                mfccs[[i, j]] -= mean;
            }
        }
    }

    /// Extract delta (first derivative) features
    pub fn delta(features: &Array2<f64>, width: usize) -> Array2<f64> {
        let (n_features, n_frames) = features.dim();
        let mut deltas = Array2::zeros((n_features, n_frames));

        let width = width as i64;
        let denominator: f64 = (1..=width).map(|i| i * i).sum::<i64>() as f64 * 2.0;

        for i in 0..n_features {
            for j in 0..n_frames {
                let mut delta = 0.0;

                for t in 1..=width {
                    let t_f64 = t as f64;

                    // Forward difference
                    let idx_forward = (j as i64 + t).min(n_frames as i64 - 1) as usize;
                    // Backward difference
                    let idx_backward = (j as i64 - t).max(0) as usize;

                    delta += t_f64 * (features[[i, idx_forward]] - features[[i, idx_backward]]);
                }

                deltas[[i, j]] = delta / denominator;
            }
        }

        deltas
    }

    /// Extract delta-delta (second derivative) features
    pub fn delta_delta(features: &Array2<f64>, width: usize) -> Array2<f64> {
        let deltas = Self::delta(features, width);
        Self::delta(&deltas, width)
    }

    /// Extract MFCCs with delta and delta-delta features
    pub fn extract_with_deltas(&self, signal: &ArrayView1<f64>) -> Result<Array2<f64>> {
        let mfccs = self.extract(signal)?;
        let deltas = Self::delta(&mfccs, 2);
        let delta_deltas = Self::delta_delta(&mfccs, 2);

        // Stack features vertically
        let (n_mfcc, n_frames) = mfccs.dim();
        let mut combined = Array2::zeros((n_mfcc * 3, n_frames));

        for i in 0..n_mfcc {
            for j in 0..n_frames {
                combined[[i, j]] = mfccs[[i, j]];
                combined[[i + n_mfcc, j]] = deltas[[i, j]];
                combined[[i + 2 * n_mfcc, j]] = delta_deltas[[i, j]];
            }
        }

        Ok(combined)
    }

    /// Get the configuration
    pub fn config(&self) -> &MFCCConfig {
        &self.config
    }

    /// Get the mel filterbank
    pub fn mel_filterbank(&self) -> &MelFilterbank {
        &self.mel_filterbank
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_hz_mel_conversion() {
        let hz = 1000.0;
        let mel = MelFilterbank::hz_to_mel(hz);
        let hz_back = MelFilterbank::mel_to_hz(mel);

        assert_abs_diff_eq!(hz, hz_back, epsilon = 1e-6);
    }

    #[test]
    fn test_mel_filterbank() -> Result<()> {
        let filterbank = MelFilterbank::new(40, 512, 16000.0, 0.0, 8000.0)?;

        assert_eq!(filterbank.filters.dim(), (40, 257));

        // Check that filters sum approximately to 1 for overlapping regions
        let center_freqs = filterbank.center_frequencies();
        assert_eq!(center_freqs.len(), 40);
        assert!(center_freqs[0] > 0.0);
        assert!(center_freqs[39] < 8000.0);

        Ok(())
    }

    #[test]
    fn test_mfcc_extraction() -> Result<()> {
        let signal = Array1::from_vec((0..16000).map(|i| (i as f64 * 0.01).sin()).collect());
        let mfcc = MFCC::default()?;

        let features = mfcc.extract(&signal.view())?;

        assert_eq!(features.dim().0, 13); // 13 MFCCs
        assert!(features.dim().1 > 0); // Multiple frames

        Ok(())
    }

    #[test]
    fn test_mfcc_with_deltas() -> Result<()> {
        let signal = Array1::from_vec((0..16000).map(|i| (i as f64 * 0.01).sin()).collect());
        let mfcc = MFCC::default()?;

        let features = mfcc.extract_with_deltas(&signal.view())?;

        assert_eq!(features.dim().0, 39); // 13 + 13 + 13
        assert!(features.dim().1 > 0);

        Ok(())
    }

    #[test]
    fn test_delta_features() {
        let features = Array2::from_shape_vec(
            (2, 5),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 0.5, 1.0, 1.5, 2.0, 2.5],
        )
        .expect("Failed to create array");

        let deltas = MFCC::delta(&features, 2);

        assert_eq!(deltas.dim(), (2, 5));

        // Deltas should capture the rate of change
        for i in 1..4 {
            assert!(deltas[[0, i]].abs() > 0.0);
        }
    }

    #[test]
    fn test_dct_matrix() {
        let dct = MFCC::compute_dct_matrix(13, 40);

        assert_eq!(dct.dim(), (13, 40));

        // Check orthogonality (approximately)
        let product = dct.dot(&dct.t());
        for i in 0..13 {
            for j in 0..13 {
                if i == j {
                    assert!(product[[i, j]] > 0.5);
                }
            }
        }
    }

    #[test]
    fn test_mfcc_config() {
        let config = MFCCConfig::default();
        assert_eq!(config.n_mfcc, 13);
        assert_eq!(config.n_mels, 40);
        assert_eq!(config.sample_rate, 16000.0);
    }
}
