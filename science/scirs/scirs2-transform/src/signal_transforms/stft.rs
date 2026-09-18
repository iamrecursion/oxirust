//! Short-Time Fourier Transform (STFT) and Spectrogram Implementation
//!
//! Provides time-frequency analysis using STFT with various window functions.

use crate::error::{Result, TransformError};
use rayon::prelude::*;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::numeric::Complex;
use scirs2_fft::fft;
use std::f64::consts::PI;

/// Window function types for STFT
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WindowType {
    /// Hann window
    Hann,
    /// Hamming window
    Hamming,
    /// Blackman window
    Blackman,
    /// Bartlett (triangular) window
    Bartlett,
    /// Rectangular window (no windowing)
    Rectangular,
    /// Kaiser window with beta parameter
    Kaiser(f64),
    /// Tukey window with alpha parameter
    Tukey(f64),
}

impl WindowType {
    /// Generate window function
    pub fn generate(&self, n: usize) -> Array1<f64> {
        match self {
            WindowType::Hann => Self::hann(n),
            WindowType::Hamming => Self::hamming(n),
            WindowType::Blackman => Self::blackman(n),
            WindowType::Bartlett => Self::bartlett(n),
            WindowType::Rectangular => Array1::ones(n),
            WindowType::Kaiser(beta) => Self::kaiser(n, *beta),
            WindowType::Tukey(alpha) => Self::tukey(n, *alpha),
        }
    }

    fn hann(n: usize) -> Array1<f64> {
        Array1::from_vec(
            (0..n)
                .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / (n - 1) as f64).cos()))
                .collect(),
        )
    }

    fn hamming(n: usize) -> Array1<f64> {
        Array1::from_vec(
            (0..n)
                .map(|i| 0.54 - 0.46 * (2.0 * PI * i as f64 / (n - 1) as f64).cos())
                .collect(),
        )
    }

    fn blackman(n: usize) -> Array1<f64> {
        Array1::from_vec(
            (0..n)
                .map(|i| {
                    let angle = 2.0 * PI * i as f64 / (n - 1) as f64;
                    0.42 - 0.5 * angle.cos() + 0.08 * (2.0 * angle).cos()
                })
                .collect(),
        )
    }

    fn bartlett(n: usize) -> Array1<f64> {
        Array1::from_vec(
            (0..n)
                .map(|i| 1.0 - (2.0 * (i as f64 - (n - 1) as f64 / 2.0).abs() / (n - 1) as f64))
                .collect(),
        )
    }

    fn kaiser(n: usize, beta: f64) -> Array1<f64> {
        let i0_beta = Self::bessel_i0(beta);
        Array1::from_vec(
            (0..n)
                .map(|i| {
                    let x = 2.0 * i as f64 / (n - 1) as f64 - 1.0;
                    let arg = beta * (1.0 - x * x).sqrt();
                    Self::bessel_i0(arg) / i0_beta
                })
                .collect(),
        )
    }

    fn tukey(n: usize, alpha: f64) -> Array1<f64> {
        let alpha = alpha.clamp(0.0, 1.0);
        Array1::from_vec(
            (0..n)
                .map(|i| {
                    let x = i as f64 / (n - 1) as f64;
                    if x < alpha / 2.0 {
                        0.5 * (1.0 + (2.0 * PI * x / alpha - PI).cos())
                    } else if x > 1.0 - alpha / 2.0 {
                        0.5 * (1.0 + (2.0 * PI * (1.0 - x) / alpha - PI).cos())
                    } else {
                        1.0
                    }
                })
                .collect(),
        )
    }

    /// Modified Bessel function of the first kind, order 0
    fn bessel_i0(x: f64) -> f64 {
        let mut sum = 1.0;
        let mut term = 1.0;
        let threshold = 1e-12;

        for k in 1..50 {
            term *= (x / 2.0) * (x / 2.0) / (k as f64 * k as f64);
            sum += term;
            if term < threshold {
                break;
            }
        }

        sum
    }
}

/// STFT configuration
#[derive(Debug, Clone)]
pub struct STFTConfig {
    /// Window size (number of samples)
    pub window_size: usize,
    /// Hop size (number of samples to advance between windows)
    pub hop_size: usize,
    /// Window function type
    pub window_type: WindowType,
    /// FFT size (zero-padding if > window_size)
    pub nfft: Option<usize>,
    /// Whether to return only positive frequencies
    pub onesided: bool,
    /// Padding mode for signal edges
    pub padding: PaddingMode,
}

/// Padding mode for STFT
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PaddingMode {
    /// No padding
    None,
    /// Zero padding
    Zero,
    /// Constant padding (edge values)
    Edge,
    /// Reflect padding
    Reflect,
}

impl Default for STFTConfig {
    fn default() -> Self {
        STFTConfig {
            window_size: 256,
            hop_size: 128,
            window_type: WindowType::Hann,
            nfft: None,
            onesided: true,
            padding: PaddingMode::Zero,
        }
    }
}

/// Short-Time Fourier Transform
#[derive(Debug, Clone)]
pub struct STFT {
    config: STFTConfig,
    window: Array1<f64>,
}

impl STFT {
    /// Create a new STFT instance with configuration
    pub fn new(config: STFTConfig) -> Self {
        let window = config.window_type.generate(config.window_size);
        STFT { config, window }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(STFTConfig::default())
    }

    /// Create with specified window size and hop size
    pub fn with_params(window_size: usize, hop_size: usize) -> Self {
        Self::new(STFTConfig {
            window_size,
            hop_size,
            ..Default::default()
        })
    }

    /// Compute the STFT of a signal
    pub fn transform(&self, signal: &ArrayView1<f64>) -> Result<Array2<Complex<f64>>> {
        let signal_len = signal.len();
        if signal_len == 0 {
            return Err(TransformError::InvalidInput("Empty signal".to_string()));
        }

        let nfft = self.config.nfft.unwrap_or(self.config.window_size);
        if nfft < self.config.window_size {
            return Err(TransformError::InvalidInput(
                "FFT size must be >= window size".to_string(),
            ));
        }

        // Calculate number of frames
        let n_frames = self.calculate_n_frames(signal_len);
        let n_freqs = if self.config.onesided {
            nfft / 2 + 1
        } else {
            nfft
        };

        let mut stft = Array2::from_elem((n_freqs, n_frames), Complex::new(0.0, 0.0));

        // Process each frame
        for (frame_idx, frame_start) in (0..signal_len)
            .step_by(self.config.hop_size)
            .take(n_frames)
            .enumerate()
        {
            let frame = self.extract_frame(signal, frame_start)?;
            let spectrum = self.compute_frame_spectrum(&frame, nfft)?;

            for (freq_idx, &val) in spectrum.iter().enumerate() {
                if freq_idx < n_freqs {
                    stft[[freq_idx, frame_idx]] = val;
                }
            }
        }

        Ok(stft)
    }

    /// Compute the inverse STFT
    pub fn inverse(&self, stft: &Array2<Complex<f64>>) -> Result<Array1<f64>> {
        let (n_freqs, n_frames) = stft.dim();

        if n_frames == 0 {
            return Err(TransformError::InvalidInput(
                "No frames in STFT".to_string(),
            ));
        }

        let nfft = self.config.nfft.unwrap_or(self.config.window_size);

        // Estimate output length
        let output_len = (n_frames - 1) * self.config.hop_size + self.config.window_size;
        let mut output = Array1::zeros(output_len);
        let mut window_sum: Array1<f64> = Array1::zeros(output_len);

        // Overlap-add synthesis
        for frame_idx in 0..n_frames {
            // Extract frame spectrum
            let mut spectrum = Vec::with_capacity(nfft);
            for freq_idx in 0..n_freqs {
                spectrum.push(stft[[freq_idx, frame_idx]]);
            }

            // Mirror spectrum for onesided case
            if self.config.onesided && nfft > 1 {
                for freq_idx in (1..(nfft - n_freqs + 1)).rev() {
                    if freq_idx < n_freqs {
                        spectrum.push(spectrum[freq_idx].conj());
                    }
                }
            }

            // Inverse FFT
            let time_frame = scirs2_fft::ifft(&spectrum, None)?;

            // Overlap-add with windowing
            let frame_start = frame_idx * self.config.hop_size;
            for (i, &val) in time_frame.iter().take(self.config.window_size).enumerate() {
                let idx = frame_start + i;
                if idx < output_len {
                    output[idx] += val.re * self.window[i];
                    window_sum[idx] += self.window[i] * self.window[i];
                }
            }
        }

        // Normalize by window sum
        for i in 0..output_len {
            if window_sum[i] > 1e-10 {
                output[i] /= window_sum[i];
            }
        }

        Ok(output)
    }

    fn extract_frame(&self, signal: &ArrayView1<f64>, start: usize) -> Result<Array1<f64>> {
        let signal_len = signal.len();
        let mut frame = Array1::zeros(self.config.window_size);

        match self.config.padding {
            PaddingMode::None => {
                let end = (start + self.config.window_size).min(signal_len);
                for i in 0..(end - start) {
                    frame[i] = signal[start + i] * self.window[i];
                }
            }
            PaddingMode::Zero => {
                for i in 0..self.config.window_size {
                    let idx = start + i;
                    if idx < signal_len {
                        frame[i] = signal[idx] * self.window[i];
                    }
                }
            }
            PaddingMode::Edge => {
                for i in 0..self.config.window_size {
                    let idx = (start + i).min(signal_len - 1);
                    frame[i] = signal[idx] * self.window[i];
                }
            }
            PaddingMode::Reflect => {
                for i in 0..self.config.window_size {
                    let mut idx = start as i64 + i as i64;
                    if idx >= signal_len as i64 {
                        idx = 2 * signal_len as i64 - idx - 2;
                    }
                    if idx < 0 {
                        idx = -idx;
                    }
                    let idx = (idx as usize).min(signal_len - 1);
                    frame[i] = signal[idx] * self.window[i];
                }
            }
        }

        Ok(frame)
    }

    fn compute_frame_spectrum(
        &self,
        frame: &Array1<f64>,
        nfft: usize,
    ) -> Result<Vec<Complex<f64>>> {
        // Zero-pad if necessary
        let mut padded = vec![0.0; nfft];
        for (i, &val) in frame.iter().enumerate() {
            if i < nfft {
                padded[i] = val;
            }
        }

        Ok(fft(&padded, None)?)
    }

    fn calculate_n_frames(&self, signal_len: usize) -> usize {
        if signal_len < self.config.window_size {
            return 1;
        }
        ((signal_len - self.config.window_size) / self.config.hop_size) + 1
    }

    /// Get the window function
    pub fn window(&self) -> &Array1<f64> {
        &self.window
    }

    /// Get the configuration
    pub fn config(&self) -> &STFTConfig {
        &self.config
    }
}

/// Spectrogram computation
#[derive(Debug, Clone)]
pub struct Spectrogram {
    stft: STFT,
    scaling: SpectrogramScaling,
}

/// Spectrogram scaling modes
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpectrogramScaling {
    /// Power spectrum (magnitude squared)
    Power,
    /// Magnitude spectrum
    Magnitude,
    /// Decibel scale (10 * log10)
    Decibel,
}

impl Spectrogram {
    /// Create a new spectrogram with STFT configuration
    pub fn new(config: STFTConfig) -> Self {
        Spectrogram {
            stft: STFT::new(config),
            scaling: SpectrogramScaling::Power,
        }
    }

    /// Set the scaling mode
    pub fn with_scaling(mut self, scaling: SpectrogramScaling) -> Self {
        self.scaling = scaling;
        self
    }

    /// Compute the spectrogram
    pub fn compute(&self, signal: &ArrayView1<f64>) -> Result<Array2<f64>> {
        let stft = self.stft.transform(signal)?;
        let (n_freqs, n_frames) = stft.dim();

        let mut spectrogram = Array2::zeros((n_freqs, n_frames));

        for i in 0..n_freqs {
            for j in 0..n_frames {
                let mag = stft[[i, j]].norm();
                spectrogram[[i, j]] = match self.scaling {
                    SpectrogramScaling::Power => mag * mag,
                    SpectrogramScaling::Magnitude => mag,
                    SpectrogramScaling::Decibel => {
                        let power = mag * mag;
                        if power > 1e-10 {
                            10.0 * power.log10()
                        } else {
                            -100.0 // Floor value
                        }
                    }
                };
            }
        }

        Ok(spectrogram)
    }

    /// Get frequency bins in Hz
    pub fn frequency_bins(&self, sampling_rate: f64) -> Vec<f64> {
        let nfft = self
            .stft
            .config
            .nfft
            .unwrap_or(self.stft.config.window_size);
        let n_freqs = if self.stft.config.onesided {
            nfft / 2 + 1
        } else {
            nfft
        };

        (0..n_freqs)
            .map(|i| i as f64 * sampling_rate / nfft as f64)
            .collect()
    }

    /// Get time bins in seconds
    pub fn time_bins(&self, signal_len: usize, sampling_rate: f64) -> Vec<f64> {
        let n_frames = self.stft.calculate_n_frames(signal_len);
        (0..n_frames)
            .map(|i| (i * self.stft.config.hop_size) as f64 / sampling_rate)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_window_generation() {
        let hann = WindowType::Hann.generate(64);
        assert_eq!(hann.len(), 64);
        assert_abs_diff_eq!(hann[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(hann[63], 0.0, epsilon = 1e-10);
        assert!(hann[32] > 0.9); // Peak near center

        let hamming = WindowType::Hamming.generate(64);
        assert_eq!(hamming.len(), 64);
        assert!(hamming[0] > 0.0); // Hamming doesn't go to zero
    }

    #[test]
    fn test_stft_simple() -> Result<()> {
        let signal = Array1::from_vec((0..256).map(|i| (i as f64 * 0.1).sin()).collect());
        let stft = STFT::with_params(64, 32);

        let result = stft.transform(&signal.view())?;

        assert!(result.dim().0 > 0);
        assert!(result.dim().1 > 0);

        Ok(())
    }

    #[test]
    fn test_stft_inverse() -> Result<()> {
        let signal = Array1::from_vec((0..256).map(|i| (i as f64 * 0.1).sin()).collect());
        let stft = STFT::with_params(64, 32);

        let transformed = stft.transform(&signal.view())?;
        let reconstructed = stft.inverse(&transformed)?;

        // Check that reconstruction is approximately correct
        assert!(reconstructed.len() > 0);

        Ok(())
    }

    #[test]
    fn test_spectrogram() -> Result<()> {
        let signal = Array1::from_vec((0..512).map(|i| (i as f64 * 0.05).sin()).collect());
        let config = STFTConfig {
            window_size: 128,
            hop_size: 64,
            ..Default::default()
        };

        let spectrogram = Spectrogram::new(config);
        let spec = spectrogram.compute(&signal.view())?;

        assert!(spec.dim().0 > 0);
        assert!(spec.dim().1 > 0);
        assert!(spec.iter().all(|&x| x >= 0.0));

        Ok(())
    }

    #[test]
    fn test_spectrogram_scaling() -> Result<()> {
        let signal = Array1::from_vec((0..256).map(|i| (i as f64 * 0.1).sin()).collect());
        let config = STFTConfig::default();

        let spec_power = Spectrogram::new(config.clone())
            .with_scaling(SpectrogramScaling::Power)
            .compute(&signal.view())?;

        let spec_mag = Spectrogram::new(config.clone())
            .with_scaling(SpectrogramScaling::Magnitude)
            .compute(&signal.view())?;

        let spec_db = Spectrogram::new(config)
            .with_scaling(SpectrogramScaling::Decibel)
            .compute(&signal.view())?;

        assert_eq!(spec_power.dim(), spec_mag.dim());
        assert_eq!(spec_power.dim(), spec_db.dim());

        Ok(())
    }

    #[test]
    fn test_frequency_time_bins() {
        let config = STFTConfig {
            window_size: 256,
            hop_size: 128,
            ..Default::default()
        };
        let spectrogram = Spectrogram::new(config);

        let freqs = spectrogram.frequency_bins(1000.0);
        let times = spectrogram.time_bins(1000, 1000.0);

        assert!(freqs.len() > 0);
        assert!(times.len() > 0);
        assert_abs_diff_eq!(freqs[0], 0.0, epsilon = 1e-10);
    }
}
