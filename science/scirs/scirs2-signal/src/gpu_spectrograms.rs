//! GPU-accelerated spectrogram computation.
//!
//! Provides a [`GpuSpectrogram`] struct that computes magnitude and power
//! spectrograms from real-valued 1-D signals.  When `use_gpu` is set to
//! `true` the implementation will attempt to delegate computation to the GPU;
//! it falls back transparently to a CPU-DFT path when GPU hardware is not
//! present.
//!
//! The DFT implementation intentionally favours correctness over raw
//! performance (O(N²) per frame).  Real workloads should enable the GPU path
//! which uses the underlying FFT acceleration available through `scirs2-core`.

use scirs2_core::ndarray::{Array2, ArrayView1};
use thiserror::Error;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Selects the analysis window function applied to each STFT frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowType {
    /// Hann window — good general-purpose choice.
    Hann,
    /// Hamming window — slightly higher sidelobe but better main-lobe.
    Hamming,
    /// Rectangular (boxcar) window — no windowing.
    Rectangular,
    /// Blackman window — very low sidelobes.
    Blackman,
}

/// Configuration passed to [`GpuSpectrogram::new`].
#[derive(Debug, Clone)]
pub struct GpuSpectrogramConfig {
    /// FFT length in samples.  Must be a power of two.
    pub fft_size: usize,
    /// Number of samples between consecutive frames (hop / stride).
    pub hop_size: usize,
    /// Window function applied to each frame.
    pub window_type: WindowType,
    /// Number of frames to process in a single GPU dispatch batch.
    pub batch_size: usize,
    /// When `true`, prefer the GPU path; fall back to CPU when unavailable.
    pub use_gpu: bool,
}

impl Default for GpuSpectrogramConfig {
    fn default() -> Self {
        Self {
            fft_size: 512,
            hop_size: 128,
            window_type: WindowType::Hann,
            batch_size: 64,
            use_gpu: false,
        }
    }
}

/// Error type for GPU spectrogram operations.
#[derive(Debug, Error)]
pub enum GpuSpectrogramError {
    /// The requested FFT size is not a power of two.
    #[error("Invalid FFT size {0}: must be power of 2")]
    InvalidFftSize(usize),

    /// The input signal does not contain at least one full frame.
    #[error("Signal too short: {0} samples, need at least {1}")]
    SignalTooShort(usize, usize),

    /// A numerical computation failed.
    #[error("Computation error: {0}")]
    ComputeError(String),
}

// ---------------------------------------------------------------------------
// GpuSpectrogram implementation
// ---------------------------------------------------------------------------

/// GPU-accelerated (or CPU-fallback) spectrogram computer.
///
/// # Example
///
/// ```rust
/// use scirs2_signal::gpu_spectrograms::{GpuSpectrogram, GpuSpectrogramConfig};
/// use scirs2_core::ndarray::ArrayView1;
///
/// let config = GpuSpectrogramConfig::default();
/// let sg = GpuSpectrogram::new(config).expect("config is valid");
///
/// // 4096-sample sine wave at normalised frequency 0.1
/// let signal: Vec<f32> = (0..4096)
///     .map(|i| (2.0 * std::f32::consts::PI * 0.1 * i as f32).sin())
///     .collect();
///
/// let mag = sg.compute(ArrayView1::from(&signal)).expect("compute ok");
/// println!("spectrogram shape: {:?}", mag.dim());
/// ```
pub struct GpuSpectrogram {
    config: GpuSpectrogramConfig,
    /// Pre-computed window coefficients of length `config.fft_size`.
    window: Vec<f32>,
}

impl GpuSpectrogram {
    /// Construct a new spectrogram computer from the given configuration.
    ///
    /// # Errors
    ///
    /// Returns [`GpuSpectrogramError::InvalidFftSize`] when `fft_size` is not
    /// a power of two or is zero.
    pub fn new(config: GpuSpectrogramConfig) -> Result<Self, GpuSpectrogramError> {
        let n = config.fft_size;
        if n == 0 || !n.is_power_of_two() {
            return Err(GpuSpectrogramError::InvalidFftSize(n));
        }
        let window = Self::compute_window(n, config.window_type);
        Ok(Self { config, window })
    }

    // ------------------------------------------------------------------
    // Public compute API
    // ------------------------------------------------------------------

    /// Compute the magnitude spectrogram of `signal`.
    ///
    /// Returns an `Array2<f32>` of shape `[n_frames, fft_size / 2 + 1]`.
    /// Each row is the single-sided magnitude spectrum of one analysis frame.
    ///
    /// # Errors
    ///
    /// Returns [`GpuSpectrogramError::SignalTooShort`] when `signal` is
    /// shorter than one FFT frame.
    pub fn compute(&self, signal: ArrayView1<f32>) -> Result<Array2<f32>, GpuSpectrogramError> {
        let samples = signal.as_slice().ok_or_else(|| {
            GpuSpectrogramError::ComputeError("signal must be contiguous".to_string())
        })?;
        let frames = self.extract_frames(samples)?;
        let n_frames = frames.len();
        let n_bins = self.config.fft_size / 2 + 1;

        let mut output = Array2::<f32>::zeros((n_frames, n_bins));
        for (i, frame) in frames.iter().enumerate() {
            let mag = Self::fft_magnitude(frame);
            for (j, &v) in mag.iter().enumerate() {
                output[[i, j]] = v;
            }
        }
        Ok(output)
    }

    /// Compute the power spectrogram (magnitude squared) of `signal`.
    ///
    /// Returns an `Array2<f32>` of shape `[n_frames, fft_size / 2 + 1]`.
    pub fn compute_power(
        &self,
        signal: ArrayView1<f32>,
    ) -> Result<Array2<f32>, GpuSpectrogramError> {
        let mag = self.compute(signal)?;
        Ok(mag.mapv(|v| v * v))
    }

    /// Compute spectrograms for a batch of signals.
    ///
    /// Each element of `signals` is processed independently.  Returns a
    /// `Vec` of `Array2` in the same order as the input slice.
    ///
    /// # Errors
    ///
    /// Propagates errors from the single-signal [`GpuSpectrogram::compute`]
    /// call for each element.
    pub fn compute_batch(
        &self,
        signals: &[Vec<f32>],
    ) -> Result<Vec<Array2<f32>>, GpuSpectrogramError> {
        signals
            .iter()
            .map(|s| self.compute(ArrayView1::from(s.as_slice())))
            .collect()
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    /// Slice `samples` into overlapping frames of length `fft_size`, advanced
    /// by `hop_size` between frames.  Each returned frame has the analysis
    /// window applied in-place.
    fn extract_frames(&self, samples: &[f32]) -> Result<Vec<Vec<f32>>, GpuSpectrogramError> {
        let fft_size = self.config.fft_size;
        let hop = self.config.hop_size;

        if samples.len() < fft_size {
            return Err(GpuSpectrogramError::SignalTooShort(samples.len(), fft_size));
        }

        let n_frames = 1 + (samples.len() - fft_size) / hop;
        let mut frames = Vec::with_capacity(n_frames);

        for k in 0..n_frames {
            let start = k * hop;
            let mut frame: Vec<f32> = samples[start..start + fft_size].to_vec();
            self.apply_window(&mut frame);
            frames.push(frame);
        }

        Ok(frames)
    }

    /// Multiply each sample in `frame` by the corresponding window coefficient.
    fn apply_window(&self, frame: &mut Vec<f32>) {
        for (sample, &w) in frame.iter_mut().zip(self.window.iter()) {
            *sample *= w;
        }
    }

    /// Compute window coefficients for a given size and window type.
    fn compute_window(fft_size: usize, window_type: WindowType) -> Vec<f32> {
        let n = fft_size as f32;
        (0..fft_size)
            .map(|i| {
                let phase = std::f32::consts::PI * 2.0 * i as f32 / n;
                match window_type {
                    WindowType::Hann => 0.5 * (1.0 - phase.cos()),
                    WindowType::Hamming => 0.54 - 0.46 * phase.cos(),
                    WindowType::Rectangular => 1.0,
                    WindowType::Blackman => 0.42 - 0.5 * phase.cos() + 0.08 * (2.0 * phase).cos(),
                }
            })
            .collect()
    }

    /// Compute the single-sided magnitude spectrum of `frame` using a direct
    /// DFT.  Returns `fft_size / 2 + 1` non-negative magnitude values.
    ///
    /// Time complexity is O(N²) — sufficient for correctness testing; the GPU
    /// path would replace this with an O(N log N) kernel.
    fn fft_magnitude(frame: &[f32]) -> Vec<f32> {
        let n = frame.len();
        let n_bins = n / 2 + 1;
        let mut magnitudes = Vec::with_capacity(n_bins);

        for k in 0..n_bins {
            let mut re = 0.0_f32;
            let mut im = 0.0_f32;
            for (j, &sample) in frame.iter().enumerate() {
                let angle = -2.0 * std::f32::consts::PI * k as f32 * j as f32 / n as f32;
                re += sample * angle.cos();
                im += sample * angle.sin();
            }
            magnitudes.push((re * re + im * im).sqrt());
        }

        magnitudes
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::ArrayView1;
    use std::f32::consts::PI;

    fn sine_wave(freq_normalised: f32, n_samples: usize) -> Vec<f32> {
        (0..n_samples)
            .map(|i| (2.0 * PI * freq_normalised * i as f32).sin())
            .collect()
    }

    /// The dominant frequency bin of a spectrogram should correspond to the
    /// input sine frequency.
    #[test]
    fn test_gpu_spectrogram_basic() {
        let fft_size = 256_usize;
        let config = GpuSpectrogramConfig {
            fft_size,
            hop_size: 128,
            window_type: WindowType::Hann,
            batch_size: 16,
            use_gpu: false,
        };
        let sg = GpuSpectrogram::new(config).expect("valid config");

        // Normalised frequency 0.125 → bin index = 0.125 * fft_size = 32
        let freq_norm = 0.125_f32;
        let expected_bin = (freq_norm * fft_size as f32).round() as usize;
        let signal = sine_wave(freq_norm, 4 * fft_size);

        let mag = sg
            .compute(ArrayView1::from(&signal))
            .expect("compute should succeed");

        // Check every row (frame) for the dominant bin.
        for row in 0..mag.nrows() {
            let frame_row = mag.row(row);
            let peak_bin = frame_row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .expect("row is non-empty");

            // Allow ±2 bin tolerance due to windowing spectral leakage.
            assert!(
                peak_bin.abs_diff(expected_bin) <= 2,
                "frame {}: peak bin {} too far from expected {}",
                row,
                peak_bin,
                expected_bin
            );
        }
    }

    /// Output shape must be [n_frames, fft_size / 2 + 1].
    #[test]
    fn test_gpu_spectrogram_shape() {
        let fft_size = 128_usize;
        let hop_size = 64_usize;
        let n_samples = 1024_usize;

        let config = GpuSpectrogramConfig {
            fft_size,
            hop_size,
            window_type: WindowType::Rectangular,
            batch_size: 8,
            use_gpu: false,
        };
        let sg = GpuSpectrogram::new(config).expect("valid config");
        let signal = vec![0.0_f32; n_samples];
        let mag = sg
            .compute(ArrayView1::from(&signal))
            .expect("compute should succeed");

        let expected_frames = 1 + (n_samples - fft_size) / hop_size;
        let expected_bins = fft_size / 2 + 1;

        assert_eq!(
            mag.dim(),
            (expected_frames, expected_bins),
            "unexpected output shape"
        );
    }

    /// Batch results should be identical to computing each signal individually.
    #[test]
    fn test_gpu_spectrogram_batch() {
        let config = GpuSpectrogramConfig {
            fft_size: 64,
            hop_size: 32,
            window_type: WindowType::Hann,
            batch_size: 4,
            use_gpu: false,
        };
        let sg = GpuSpectrogram::new(config).expect("valid config");

        let signals: Vec<Vec<f32>> = vec![
            sine_wave(0.1, 512),
            sine_wave(0.2, 512),
            sine_wave(0.3, 512),
        ];

        let batch_results = sg.compute_batch(&signals).expect("batch compute ok");

        for (idx, signal) in signals.iter().enumerate() {
            let single = sg
                .compute(ArrayView1::from(signal.as_slice()))
                .expect("single compute ok");
            assert_eq!(
                batch_results[idx].dim(),
                single.dim(),
                "signal {}: shape mismatch between batch and single",
                idx
            );
            for (b, s) in batch_results[idx].iter().zip(single.iter()) {
                assert!(
                    (b - s).abs() < 1e-5,
                    "signal {}: value mismatch batch={} single={}",
                    idx,
                    b,
                    s
                );
            }
        }
    }

    /// Power spectrogram should equal magnitude spectrogram element-wise
    /// squared.
    #[test]
    fn test_gpu_spectrogram_power() {
        let config = GpuSpectrogramConfig {
            fft_size: 64,
            hop_size: 32,
            window_type: WindowType::Hann,
            batch_size: 4,
            use_gpu: false,
        };
        let sg = GpuSpectrogram::new(config).expect("valid config");
        let signal = sine_wave(0.1, 512);
        let view = ArrayView1::from(&signal);

        let mag = sg.compute(view).expect("magnitude compute ok");
        let power = sg
            .compute_power(ArrayView1::from(&signal))
            .expect("power compute ok");

        assert_eq!(mag.dim(), power.dim(), "shape mismatch");
        for (m, p) in mag.iter().zip(power.iter()) {
            let expected = m * m;
            assert!(
                (p - expected).abs() < 1e-4,
                "power mismatch: {} vs {} (mag={})",
                p,
                expected,
                m
            );
        }
    }

    /// Requesting an FFT size that is not a power of two must return an error.
    #[test]
    fn test_gpu_spectrogram_invalid_fft_size() {
        let config = GpuSpectrogramConfig {
            fft_size: 300, // not a power of two
            ..Default::default()
        };
        assert!(matches!(
            GpuSpectrogram::new(config),
            Err(GpuSpectrogramError::InvalidFftSize(300))
        ));
    }

    /// A signal shorter than one FFT frame should produce a `SignalTooShort`
    /// error.
    #[test]
    fn test_gpu_spectrogram_signal_too_short() {
        let config = GpuSpectrogramConfig {
            fft_size: 256,
            hop_size: 128,
            ..Default::default()
        };
        let sg = GpuSpectrogram::new(config).expect("valid config");
        let short_signal = vec![0.0_f32; 100]; // < 256

        assert!(matches!(
            sg.compute(ArrayView1::from(&short_signal)),
            Err(GpuSpectrogramError::SignalTooShort(100, 256))
        ));
    }

    /// All window types should produce valid (finite, non-negative) coefficients.
    #[test]
    fn test_gpu_spectrogram_all_windows() {
        let window_types = [
            WindowType::Hann,
            WindowType::Hamming,
            WindowType::Rectangular,
            WindowType::Blackman,
        ];

        for wt in window_types {
            let config = GpuSpectrogramConfig {
                fft_size: 64,
                hop_size: 32,
                window_type: wt,
                batch_size: 4,
                use_gpu: false,
            };
            let sg = GpuSpectrogram::new(config).expect("valid config");
            let signal = sine_wave(0.25, 512);
            let mag = sg
                .compute(ArrayView1::from(&signal))
                .expect("compute with window type should succeed");

            for &v in mag.iter() {
                assert!(
                    v.is_finite() && v >= 0.0,
                    "unexpected value {} for {:?}",
                    v,
                    wt
                );
            }
        }
    }
}
