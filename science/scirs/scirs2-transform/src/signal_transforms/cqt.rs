//! Constant-Q Transform (CQT) and Chromagram Implementation
//!
//! Provides musically-motivated time-frequency analysis with logarithmic frequency spacing.

use crate::error::{Result, TransformError};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::numeric::Complex;
use scirs2_fft::fft;
use std::f64::consts::PI;

/// CQT configuration
#[derive(Debug, Clone)]
pub struct CQTConfig {
    /// Sampling rate in Hz
    pub sample_rate: f64,
    /// Hop size in samples
    pub hop_size: usize,
    /// Minimum frequency in Hz
    pub fmin: f64,
    /// Number of bins per octave
    pub bins_per_octave: usize,
    /// Number of octaves
    pub n_octaves: usize,
    /// Filter quality factor
    pub q_factor: f64,
    /// Window type
    pub window: WindowFunction,
}

impl Default for CQTConfig {
    fn default() -> Self {
        CQTConfig {
            sample_rate: 22050.0,
            hop_size: 512,
            fmin: 32.7, // C1
            bins_per_octave: 12,
            n_octaves: 7,
            q_factor: 1.0,
            window: WindowFunction::Hann,
        }
    }
}

/// Window functions for CQT
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WindowFunction {
    /// Hann window
    Hann,
    /// Hamming window
    Hamming,
    /// Blackman window
    Blackman,
}

impl WindowFunction {
    /// Generate window of given length
    fn generate(&self, n: usize) -> Array1<f64> {
        match self {
            WindowFunction::Hann => Array1::from_vec(
                (0..n)
                    .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / (n - 1) as f64).cos()))
                    .collect(),
            ),
            WindowFunction::Hamming => Array1::from_vec(
                (0..n)
                    .map(|i| 0.54 - 0.46 * (2.0 * PI * i as f64 / (n - 1) as f64).cos())
                    .collect(),
            ),
            WindowFunction::Blackman => Array1::from_vec(
                (0..n)
                    .map(|i| {
                        let angle = 2.0 * PI * i as f64 / (n - 1) as f64;
                        0.42 - 0.5 * angle.cos() + 0.08 * (2.0 * angle).cos()
                    })
                    .collect(),
            ),
        }
    }
}

/// Constant-Q Transform
#[derive(Debug, Clone)]
pub struct CQT {
    config: CQTConfig,
    kernel: Vec<Array1<Complex<f64>>>,
    frequencies: Vec<f64>,
}

impl CQT {
    /// Create a new CQT instance
    pub fn new(config: CQTConfig) -> Result<Self> {
        let n_bins = config.bins_per_octave * config.n_octaves;
        let mut kernel = Vec::with_capacity(n_bins);
        let mut frequencies = Vec::with_capacity(n_bins);

        // Compute frequency for each bin
        for k in 0..n_bins {
            let freq = config.fmin * 2.0_f64.powf(k as f64 / config.bins_per_octave as f64);
            frequencies.push(freq);

            // Compute kernel for this frequency
            let bin_kernel = Self::compute_kernel(
                freq,
                config.sample_rate,
                config.q_factor,
                config.bins_per_octave,
                &config.window,
            )?;
            kernel.push(bin_kernel);
        }

        Ok(CQT {
            config,
            kernel,
            frequencies,
        })
    }

    /// Create with default configuration
    pub fn default() -> Result<Self> {
        Self::new(CQTConfig::default())
    }

    /// Compute CQT kernel for a specific frequency
    fn compute_kernel(
        freq: f64,
        sample_rate: f64,
        q_factor: f64,
        bins_per_octave: usize,
        window: &WindowFunction,
    ) -> Result<Array1<Complex<f64>>> {
        // Calculate Q value
        let q = q_factor / (2.0_f64.powf(1.0 / bins_per_octave as f64) - 1.0);

        // Calculate filter length
        let filter_len = ((q * sample_rate / freq).ceil() as usize).max(1);

        // Generate window
        let window_vec = window.generate(filter_len);

        // Create complex exponential
        let mut kernel = Array1::from_elem(filter_len, Complex::new(0.0, 0.0));

        for n in 0..filter_len {
            let phase = 2.0 * PI * freq * n as f64 / sample_rate;
            let win_val = window_vec[n];
            kernel[n] = Complex::new(win_val * phase.cos(), -win_val * phase.sin());
        }

        // Normalize
        let norm: f64 = kernel.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        if norm > 1e-10 {
            for val in kernel.iter_mut() {
                *val = *val / norm;
            }
        }

        Ok(kernel)
    }

    /// Compute the CQT of a signal
    pub fn transform(&self, signal: &ArrayView1<f64>) -> Result<Array2<Complex<f64>>> {
        let signal_len = signal.len();
        if signal_len == 0 {
            return Err(TransformError::InvalidInput("Empty signal".to_string()));
        }

        let n_bins = self.kernel.len();
        let n_frames = (signal_len / self.config.hop_size).max(1);

        let mut cqt = Array2::from_elem((n_bins, n_frames), Complex::new(0.0, 0.0));

        // Process each frame
        for frame_idx in 0..n_frames {
            let frame_start = frame_idx * self.config.hop_size;

            // Apply each frequency kernel
            for (bin_idx, kernel) in self.kernel.iter().enumerate() {
                let mut response = Complex::new(0.0, 0.0);

                for (k, &kernel_val) in kernel.iter().enumerate() {
                    let signal_idx = frame_start + k;
                    if signal_idx < signal_len {
                        response = response + kernel_val * signal[signal_idx];
                    }
                }

                cqt[[bin_idx, frame_idx]] = response;
            }
        }

        Ok(cqt)
    }

    /// Compute magnitude of CQT
    pub fn magnitude(&self, signal: &ArrayView1<f64>) -> Result<Array2<f64>> {
        let cqt = self.transform(signal)?;
        let (n_bins, n_frames) = cqt.dim();

        let mut magnitude = Array2::zeros((n_bins, n_frames));
        for i in 0..n_bins {
            for j in 0..n_frames {
                magnitude[[i, j]] = cqt[[i, j]].norm();
            }
        }

        Ok(magnitude)
    }

    /// Compute power (magnitude squared) of CQT
    pub fn power(&self, signal: &ArrayView1<f64>) -> Result<Array2<f64>> {
        let cqt = self.transform(signal)?;
        let (n_bins, n_frames) = cqt.dim();

        let mut power = Array2::zeros((n_bins, n_frames));
        for i in 0..n_bins {
            for j in 0..n_frames {
                power[[i, j]] = cqt[[i, j]].norm_sqr();
            }
        }

        Ok(power)
    }

    /// Get the frequencies corresponding to each bin
    pub fn frequencies(&self) -> &[f64] {
        &self.frequencies
    }

    /// Get the configuration
    pub fn config(&self) -> &CQTConfig {
        &self.config
    }

    /// Get time bins in seconds
    pub fn time_bins(&self, signal_len: usize) -> Vec<f64> {
        let n_frames = (signal_len / self.config.hop_size).max(1);
        (0..n_frames)
            .map(|i| (i * self.config.hop_size) as f64 / self.config.sample_rate)
            .collect()
    }
}

/// Chromagram (pitch class profile)
#[derive(Debug, Clone)]
pub struct Chromagram {
    cqt: CQT,
    n_chroma: usize,
}

impl Chromagram {
    /// Create a new chromagram with CQT configuration
    pub fn new(config: CQTConfig) -> Result<Self> {
        // Ensure bins_per_octave is a multiple of 12 for clean folding
        let adjusted_config = CQTConfig {
            bins_per_octave: 12 * ((config.bins_per_octave + 11) / 12),
            ..config
        };

        let cqt = CQT::new(adjusted_config)?;

        Ok(Chromagram { cqt, n_chroma: 12 })
    }

    /// Create with default configuration
    pub fn default() -> Result<Self> {
        Self::new(CQTConfig::default())
    }

    /// Compute chromagram from signal
    pub fn compute(&self, signal: &ArrayView1<f64>) -> Result<Array2<f64>> {
        // Get CQT magnitude
        let cqt_mag = self.cqt.magnitude(signal)?;
        let (n_bins, n_frames) = cqt_mag.dim();

        // Fold into 12 chroma bins
        let mut chroma = Array2::zeros((self.n_chroma, n_frames));

        for i in 0..n_bins {
            let chroma_bin = i % self.n_chroma;
            for j in 0..n_frames {
                chroma[[chroma_bin, j]] += cqt_mag[[i, j]];
            }
        }

        // Normalize each frame
        for j in 0..n_frames {
            let mut sum = 0.0;
            for i in 0..self.n_chroma {
                sum += chroma[[i, j]];
            }
            if sum > 1e-10 {
                for i in 0..self.n_chroma {
                    chroma[[i, j]] /= sum;
                }
            }
        }

        Ok(chroma)
    }

    /// Compute energy-normalized chromagram
    pub fn compute_normalized(&self, signal: &ArrayView1<f64>) -> Result<Array2<f64>> {
        let cqt_power = self.cqt.power(signal)?;
        let (n_bins, n_frames) = cqt_power.dim();

        // Fold into 12 chroma bins
        let mut chroma = Array2::zeros((self.n_chroma, n_frames));

        for i in 0..n_bins {
            let chroma_bin = i % self.n_chroma;
            for j in 0..n_frames {
                chroma[[chroma_bin, j]] += cqt_power[[i, j]];
            }
        }

        // L2 normalize each frame
        for j in 0..n_frames {
            let mut norm: f64 = 0.0;
            for i in 0..self.n_chroma {
                norm += chroma[[i, j]] * chroma[[i, j]];
            }
            norm = norm.sqrt();

            if norm > 1e-10 {
                for i in 0..self.n_chroma {
                    chroma[[i, j]] /= norm;
                }
            }
        }

        Ok(chroma)
    }

    /// Get chroma labels (note names)
    pub fn chroma_labels() -> Vec<&'static str> {
        vec![
            "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
        ]
    }

    /// Get the underlying CQT
    pub fn cqt(&self) -> &CQT {
        &self.cqt
    }

    /// Get time bins in seconds
    pub fn time_bins(&self, signal_len: usize) -> Vec<f64> {
        self.cqt.time_bins(signal_len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_cqt_creation() -> Result<()> {
        let cqt = CQT::default()?;

        assert!(cqt.frequencies().len() > 0);
        assert_eq!(cqt.frequencies().len(), cqt.kernel.len());

        // Check that frequencies are logarithmically spaced
        let freqs = cqt.frequencies();
        for i in 1..freqs.len() {
            let ratio = freqs[i] / freqs[i - 1];
            assert!(ratio > 1.0);
        }

        Ok(())
    }

    #[test]
    fn test_cqt_transform() -> Result<()> {
        let signal = Array1::from_vec((0..22050).map(|i| (i as f64 * 0.01).sin()).collect());
        let cqt = CQT::default()?;

        let result = cqt.transform(&signal.view())?;

        assert!(result.dim().0 > 0);
        assert!(result.dim().1 > 0);

        Ok(())
    }

    #[test]
    fn test_cqt_magnitude() -> Result<()> {
        let signal = Array1::from_vec(
            (0..22050)
                .map(|i| {
                    // Simple tone at A4 (440 Hz)
                    (2.0 * PI * 440.0 * i as f64 / 22050.0).sin()
                })
                .collect(),
        );

        let config = CQTConfig {
            sample_rate: 22050.0,
            fmin: 55.0, // A1
            bins_per_octave: 12,
            n_octaves: 6,
            ..Default::default()
        };

        let cqt = CQT::new(config)?;
        let mag = cqt.magnitude(&signal.view())?;

        assert!(mag.dim().0 > 0);
        assert!(mag.dim().1 > 0);
        assert!(mag.iter().all(|&x| x >= 0.0));

        Ok(())
    }

    #[test]
    fn test_chromagram_creation() -> Result<()> {
        let chroma = Chromagram::default()?;

        assert_eq!(chroma.n_chroma, 12);

        Ok(())
    }

    #[test]
    fn test_chromagram_compute() -> Result<()> {
        let signal = Array1::from_vec((0..22050).map(|i| (i as f64 * 0.01).sin()).collect());
        let chroma = Chromagram::default()?;

        let result = chroma.compute(&signal.view())?;

        assert_eq!(result.dim().0, 12);
        assert!(result.dim().1 > 0);

        // Check normalization (each column should sum to ~1)
        for j in 0..result.dim().1 {
            let mut sum = 0.0;
            for i in 0..12 {
                sum += result[[i, j]];
            }
            if sum > 1e-10 {
                assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);
            }
        }

        Ok(())
    }

    #[test]
    fn test_chromagram_normalized() -> Result<()> {
        let signal = Array1::from_vec((0..22050).map(|i| (i as f64 * 0.01).sin()).collect());
        let chroma = Chromagram::default()?;

        let result = chroma.compute_normalized(&signal.view())?;

        assert_eq!(result.dim().0, 12);
        assert!(result.dim().1 > 0);

        // Check L2 normalization
        for j in 0..result.dim().1 {
            let mut norm = 0.0;
            for i in 0..12 {
                norm += result[[i, j]] * result[[i, j]];
            }
            if norm > 1e-10 {
                assert_abs_diff_eq!(norm.sqrt(), 1.0, epsilon = 1e-6);
            }
        }

        Ok(())
    }

    #[test]
    fn test_chroma_labels() {
        let labels = Chromagram::chroma_labels();
        assert_eq!(labels.len(), 12);
        assert_eq!(labels[0], "C");
        assert_eq!(labels[11], "B");
    }

    #[test]
    fn test_window_functions() {
        let hann = WindowFunction::Hann.generate(64);
        assert_eq!(hann.len(), 64);
        assert_abs_diff_eq!(hann[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(hann[63], 0.0, epsilon = 1e-10);

        let hamming = WindowFunction::Hamming.generate(64);
        assert_eq!(hamming.len(), 64);
        assert!(hamming[0] > 0.0);

        let blackman = WindowFunction::Blackman.generate(64);
        assert_eq!(blackman.len(), 64);
        assert_abs_diff_eq!(blackman[0], 0.0, epsilon = 1e-2);
    }

    #[test]
    fn test_cqt_time_bins() -> Result<()> {
        let cqt = CQT::default()?;
        let time_bins = cqt.time_bins(22050);

        assert!(time_bins.len() > 0);
        assert_abs_diff_eq!(time_bins[0], 0.0, epsilon = 1e-10);

        // Check that time bins are uniformly spaced
        if time_bins.len() > 1 {
            let dt = time_bins[1] - time_bins[0];
            for i in 2..time_bins.len() {
                assert_abs_diff_eq!(time_bins[i] - time_bins[i - 1], dt, epsilon = 1e-6);
            }
        }

        Ok(())
    }
}
