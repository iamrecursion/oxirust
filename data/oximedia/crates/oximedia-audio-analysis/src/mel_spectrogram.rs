//! Mel spectrogram computation for ML feature extraction.
//!
//! Implements triangular mel filterbanks over a short-time FFT magnitude
//! spectrum, producing the 2-D mel spectrogram (time × mel bins) suitable for
//! feeding into neural-network audio models.

use crate::AnalysisError;
use rayon::prelude::*;

/// Convert a frequency in Hz to the mel scale.
///
/// Uses the formula: mel = 2595 × log₁₀(1 + hz / 700)
#[inline]
fn hz_to_mel(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Convert a mel-scale value back to Hz.
#[inline]
fn mel_to_hz(mel: f64) -> f64 {
    700.0 * (10.0_f64.powf(mel / 2595.0) - 1.0)
}

/// Build a triangular mel filterbank.
///
/// Returns a matrix of shape `[n_mels][n_bins]` where `n_bins = n_fft / 2 + 1`.
/// Each row contains the weights for one mel filter, following the standard
/// triangular overlap scheme (HTK / librosa convention).
///
/// # Arguments
/// * `n_mels` - Number of mel filter bands.
/// * `n_fft` - FFT size (must be even and > 0).
/// * `sample_rate` - Audio sample rate in Hz.
/// * `f_min` - Lowest frequency included (Hz), default 0.0.
/// * `f_max` - Highest frequency included (Hz), defaults to `sample_rate / 2`.
///
/// # Errors
/// Returns [`AnalysisError::InvalidConfig`] if parameters are invalid.
pub fn build_mel_filterbank(
    n_mels: usize,
    n_fft: usize,
    sample_rate: u32,
    f_min: f64,
    f_max: f64,
) -> crate::Result<Vec<Vec<f32>>> {
    if n_mels == 0 {
        return Err(AnalysisError::InvalidConfig(
            "n_mels must be > 0".to_string(),
        ));
    }
    if n_fft < 2 {
        return Err(AnalysisError::InvalidConfig(
            "n_fft must be >= 2".to_string(),
        ));
    }
    if sample_rate == 0 {
        return Err(AnalysisError::InvalidConfig(
            "sample_rate must be > 0".to_string(),
        ));
    }
    let nyquist = f64::from(sample_rate) / 2.0;
    if f_min < 0.0 || f_min >= f_max {
        return Err(AnalysisError::InvalidConfig(format!(
            "f_min ({f_min}) must be >= 0 and < f_max ({f_max})"
        )));
    }
    if f_max > nyquist {
        return Err(AnalysisError::InvalidConfig(format!(
            "f_max ({f_max}) exceeds Nyquist ({nyquist})"
        )));
    }

    let n_bins = n_fft / 2 + 1;
    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(f_max);

    // n_mels + 2 equally-spaced points in mel space
    let mel_points: Vec<f64> = (0..=(n_mels + 1))
        .map(|i| mel_min + (mel_max - mel_min) * i as f64 / (n_mels + 1) as f64)
        .collect();

    // Convert mel points to FFT bin indices
    let bin_points: Vec<f64> = mel_points
        .iter()
        .map(|&m| {
            let hz = mel_to_hz(m);
            hz * n_fft as f64 / f64::from(sample_rate)
        })
        .collect();

    let mut filterbank = vec![vec![0.0_f32; n_bins]; n_mels];

    for m in 0..n_mels {
        let f_left = bin_points[m];
        let f_center = bin_points[m + 1];
        let f_right = bin_points[m + 2];

        for k in 0..n_bins {
            let k_f = k as f64;
            let weight = if k_f >= f_left && k_f <= f_center {
                let denom = f_center - f_left;
                if denom > 1e-12 {
                    (k_f - f_left) / denom
                } else {
                    0.0
                }
            } else if k_f > f_center && k_f <= f_right {
                let denom = f_right - f_center;
                if denom > 1e-12 {
                    (f_right - k_f) / denom
                } else {
                    0.0
                }
            } else {
                0.0
            };
            filterbank[m][k] = weight as f32;
        }
    }

    Ok(filterbank)
}

/// Apply a mel filterbank to a magnitude spectrum.
///
/// # Arguments
/// * `magnitude` - Magnitude spectrum of length `n_fft / 2 + 1`.
/// * `filterbank` - Filterbank matrix from [`build_mel_filterbank`].
///
/// # Returns
/// Vec of length `n_mels` with mel-band energies.
#[must_use]
pub fn apply_mel_filterbank(magnitude: &[f32], filterbank: &[Vec<f32>]) -> Vec<f32> {
    filterbank
        .iter()
        .map(|filter| {
            filter
                .iter()
                .zip(magnitude.iter())
                .map(|(&w, &m)| w * m)
                .sum()
        })
        .collect()
}

/// Configuration for mel spectrogram computation.
#[derive(Debug, Clone)]
pub struct MelSpectrogramConfig {
    /// Number of mel filter bands.
    pub n_mels: usize,
    /// FFT size in samples (must be a power of two for efficiency).
    pub n_fft: usize,
    /// Hop length in samples between successive frames.
    pub hop_length: usize,
    /// Lowest mel filter frequency in Hz.
    pub f_min: f64,
    /// Highest mel filter frequency in Hz.  `None` defaults to `sample_rate / 2`.
    pub f_max: Option<f64>,
    /// Whether to convert output to decibels (log scale).
    pub log_scale: bool,
    /// Small epsilon added before log to avoid -inf.
    pub log_epsilon: f32,
}

impl Default for MelSpectrogramConfig {
    fn default() -> Self {
        Self {
            n_mels: 128,
            n_fft: 2048,
            hop_length: 512,
            f_min: 0.0,
            f_max: None,
            log_scale: false,
            log_epsilon: 1e-10,
        }
    }
}

/// Compute the mel spectrogram of an audio signal.
///
/// Applies a Hann-windowed STFT and then projects each magnitude frame through
/// a triangular mel filterbank.  The result is a 2-D matrix indexed as
/// `spectrogram[frame][mel_bin]`.
///
/// # Arguments
/// * `samples` - Mono audio samples as `f32`.
/// * `sample_rate` - Audio sample rate in Hz.
/// * `config` - Mel spectrogram configuration.
///
/// # Returns
/// `Vec<Vec<f32>>` of shape `[num_frames][n_mels]`.
///
/// # Errors
/// Returns [`AnalysisError`] if parameters are invalid or samples are too short.
pub fn compute_mel_spectrogram(
    samples: &[f32],
    sample_rate: u32,
    config: &MelSpectrogramConfig,
) -> crate::Result<Vec<Vec<f32>>> {
    if sample_rate == 0 {
        return Err(AnalysisError::InvalidSampleRate(0.0));
    }
    if config.n_fft < 2 {
        return Err(AnalysisError::InvalidConfig(
            "n_fft must be >= 2".to_string(),
        ));
    }
    if config.hop_length == 0 {
        return Err(AnalysisError::InvalidConfig(
            "hop_length must be > 0".to_string(),
        ));
    }
    if samples.len() < config.n_fft {
        return Err(AnalysisError::InsufficientSamples {
            needed: config.n_fft,
            got: samples.len(),
        });
    }

    let f_max = config.f_max.unwrap_or_else(|| f64::from(sample_rate) / 2.0);

    let filterbank = build_mel_filterbank(
        config.n_mels,
        config.n_fft,
        sample_rate,
        config.f_min,
        f_max,
    )?;

    // Pre-compute Hann window
    let window: Vec<f32> = (0..config.n_fft)
        .map(|i| {
            let x = std::f64::consts::PI * i as f64 / (config.n_fft - 1) as f64;
            (0.5 * (1.0 - x.cos())) as f32
        })
        .collect();

    let n_bins = config.n_fft / 2 + 1;
    let num_frames = (samples.len() - config.n_fft) / config.hop_length + 1;

    // Compute the set of valid frame start positions up-front so we can
    // drive a parallel iterator without holding mutable state.
    let frame_starts: Vec<usize> = (0..num_frames)
        .map(|fi| fi * config.hop_length)
        .take_while(|&start| start + config.n_fft <= samples.len())
        .collect();

    // Each frame is independent: parallelize with rayon.
    let log_scale = config.log_scale;
    let log_eps = config.log_epsilon;

    let spectrogram: Vec<Vec<f32>> = frame_starts
        .par_iter()
        .map(|&start| {
            let end = start + config.n_fft;

            // Apply Hann window and build complex input.
            let windowed: Vec<oxifft::Complex<f64>> = samples[start..end]
                .iter()
                .zip(window.iter())
                .map(|(&s, &w)| oxifft::Complex::new(f64::from(s * w), 0.0))
                .collect();

            // FFT.
            let spectrum = oxifft::fft(&windowed);

            // Magnitude spectrum (positive frequencies only).
            let magnitude: Vec<f32> = spectrum[..n_bins]
                .iter()
                .map(|c| (c.re * c.re + c.im * c.im).sqrt() as f32)
                .collect();

            // Apply mel filterbank.
            let mut mel_frame = apply_mel_filterbank(&magnitude, &filterbank);

            // Optional log scaling.
            if log_scale {
                for v in &mut mel_frame {
                    *v = (*v + log_eps).ln();
                }
            }

            mel_frame
        })
        .collect();

    Ok(spectrogram)
}

/// Convenience function using default configuration.
///
/// Computes a mel spectrogram with 128 mel bands, 2048-point FFT, and 512-sample
/// hop length.  Output is a linear-scale (not log) energy matrix.
///
/// # Arguments
/// * `samples` - Mono audio samples.
/// * `sample_rate` - Sample rate in Hz.
/// * `n_mels` - Number of mel filter bands.
/// * `n_fft` - FFT window size.
/// * `hop_length` - Hop size between frames.
///
/// # Returns
/// `Vec<Vec<f32>>` indexed as `[frame][mel_bin]`.
///
/// # Errors
/// Returns [`AnalysisError`] on invalid parameters or insufficient input.
pub fn mel_spectrogram(
    samples: &[f32],
    sample_rate: u32,
    n_mels: usize,
    n_fft: usize,
    hop_length: usize,
) -> crate::Result<Vec<Vec<f32>>> {
    let config = MelSpectrogramConfig {
        n_mels,
        n_fft,
        hop_length,
        f_min: 0.0,
        f_max: None,
        log_scale: false,
        log_epsilon: 1e-10,
    };
    compute_mel_spectrogram(samples, sample_rate, &config)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine_wave(freq: f64, sample_rate: u32, duration_secs: f64) -> Vec<f32> {
        let num = (f64::from(sample_rate) * duration_secs) as usize;
        (0..num)
            .map(|i| {
                (2.0 * std::f64::consts::PI * freq * i as f64 / f64::from(sample_rate)).sin() as f32
            })
            .collect()
    }

    #[test]
    fn test_hz_mel_roundtrip() {
        for hz in [100.0, 440.0, 1000.0, 4000.0, 8000.0] {
            let mel = hz_to_mel(hz);
            let back = mel_to_hz(mel);
            assert!(
                (hz - back).abs() < 1e-6,
                "roundtrip failed for {hz} Hz: got {back}"
            );
        }
    }

    #[test]
    fn test_filterbank_shape() {
        let fb =
            build_mel_filterbank(40, 1024, 22050, 0.0, 11025.0).expect("filterbank should build");
        assert_eq!(fb.len(), 40, "should have 40 mel filters");
        for row in &fb {
            assert_eq!(row.len(), 513, "each filter should span n_fft/2+1 bins");
        }
    }

    #[test]
    fn test_filterbank_non_negative() {
        let fb =
            build_mel_filterbank(20, 512, 16000, 0.0, 8000.0).expect("filterbank should build");
        for (m, row) in fb.iter().enumerate() {
            for (k, &v) in row.iter().enumerate() {
                assert!(v >= 0.0, "filter {m} bin {k} negative: {v}");
            }
        }
    }

    #[test]
    fn test_filterbank_invalid_params() {
        assert!(build_mel_filterbank(0, 1024, 44100, 0.0, 22050.0).is_err());
        assert!(build_mel_filterbank(40, 0, 44100, 0.0, 22050.0).is_err());
        assert!(build_mel_filterbank(40, 1024, 0, 0.0, 22050.0).is_err());
        assert!(build_mel_filterbank(40, 1024, 44100, 1000.0, 500.0).is_err());
        assert!(build_mel_filterbank(40, 1024, 44100, 0.0, 30000.0).is_err());
    }

    #[test]
    fn test_mel_spectrogram_shape() {
        let samples = sine_wave(440.0, 22050, 1.0);
        let n_fft = 1024;
        let hop = 256;
        let n_mels = 64;
        let spec = mel_spectrogram(&samples, 22050, n_mels, n_fft, hop)
            .expect("mel spectrogram should succeed");

        let expected_frames = (samples.len() - n_fft) / hop + 1;
        assert_eq!(spec.len(), expected_frames);
        for frame in &spec {
            assert_eq!(frame.len(), n_mels);
        }
    }

    #[test]
    fn test_mel_spectrogram_values_non_negative() {
        let samples = sine_wave(1000.0, 16000, 0.5);
        let spec = mel_spectrogram(&samples, 16000, 40, 512, 128).expect("should succeed");
        for (fi, frame) in spec.iter().enumerate() {
            for (mi, &v) in frame.iter().enumerate() {
                assert!(v >= 0.0, "frame {fi} mel {mi} has negative value {v}");
            }
        }
    }

    #[test]
    fn test_mel_spectrogram_log_scale() {
        let samples = sine_wave(440.0, 22050, 0.5);
        let config = MelSpectrogramConfig {
            n_mels: 32,
            n_fft: 512,
            hop_length: 256,
            f_min: 0.0,
            f_max: None,
            log_scale: true,
            log_epsilon: 1e-10,
        };
        let spec = compute_mel_spectrogram(&samples, 22050, &config).expect("should succeed");
        assert!(!spec.is_empty());
        // All values should be finite (no -inf from unguarded log)
        for frame in &spec {
            for &v in frame {
                assert!(v.is_finite(), "log-scale value should be finite: {v}");
            }
        }
    }

    #[test]
    fn test_mel_spectrogram_too_short() {
        let short = vec![0.0_f32; 100];
        let result = mel_spectrogram(&short, 22050, 40, 1024, 512);
        assert!(result.is_err());
    }

    #[test]
    fn test_mel_spectrogram_zero_hop_error() {
        let samples = sine_wave(440.0, 22050, 0.1);
        let config = MelSpectrogramConfig {
            hop_length: 0,
            ..MelSpectrogramConfig::default()
        };
        let result = compute_mel_spectrogram(&samples, 22050, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_higher_freq_more_energy_in_higher_mels() {
        // A high-frequency sine should concentrate energy in upper mel bins
        let low = sine_wave(200.0, 22050, 0.5);
        let high = sine_wave(8000.0, 22050, 0.5);
        let n_mels = 64;
        let spec_low = mel_spectrogram(&low, 22050, n_mels, 1024, 256).expect("low should succeed");
        let spec_high =
            mel_spectrogram(&high, 22050, n_mels, 1024, 256).expect("high should succeed");

        // Average over frames
        let mean_low: Vec<f32> = (0..n_mels)
            .map(|m| spec_low.iter().map(|f| f[m]).sum::<f32>() / spec_low.len() as f32)
            .collect();
        let mean_high: Vec<f32> = (0..n_mels)
            .map(|m| spec_high.iter().map(|f| f[m]).sum::<f32>() / spec_high.len() as f32)
            .collect();

        // Sum of lower half vs upper half
        let mid = n_mels / 2;
        let low_lower: f32 = mean_low[..mid].iter().sum();
        let high_upper: f32 = mean_high[mid..].iter().sum();
        let low_upper: f32 = mean_low[mid..].iter().sum();
        let high_lower: f32 = mean_high[..mid].iter().sum();

        assert!(
            low_lower > low_upper,
            "Low-freq signal should have more energy in lower mels"
        );
        assert!(
            high_upper > high_lower,
            "High-freq signal should have more energy in upper mels"
        );
    }

    #[test]
    fn test_apply_mel_filterbank_shape() {
        let fb = build_mel_filterbank(20, 512, 16000, 0.0, 8000.0).expect("filterbank");
        let mag = vec![1.0_f32; 257];
        let mel = apply_mel_filterbank(&mag, &fb);
        assert_eq!(mel.len(), 20);
    }

    #[test]
    fn test_config_default() {
        let cfg = MelSpectrogramConfig::default();
        assert_eq!(cfg.n_mels, 128);
        assert_eq!(cfg.n_fft, 2048);
        assert_eq!(cfg.hop_length, 512);
        assert!(!cfg.log_scale);
    }
}
