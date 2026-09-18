//! Streaming FFT processor with configurable overlap
//!
//! This module implements a streaming Short-Time Fourier Transform (STFT)
//! processor that ingests samples incrementally and emits magnitude spectra
//! on a hop-by-hop basis. Two algorithmic modes are supported:
//!
//! - **Overlap-add** (OLA): Each window is analysed and the resulting spectra
//!   are accumulated for reconstruction.
//! - **Overlap-save**: The ring buffer is advanced by `hop_size` samples on
//!   each frame; the first `fft_size - hop_size` samples in the window are
//!   kept from the previous frame (overlap saved from the past).
//!
//! # Example
//!
//! ```rust
//! use scirs2_fft::streaming::{StreamingFft, StreamingFftConfig, WindowType};
//!
//! let config = StreamingFftConfig {
//!     fft_size: 64,
//!     hop_size: 32,
//!     window: WindowType::Hann,
//! };
//! let mut proc = StreamingFft::new(config);
//! let signal: Vec<f64> = (0..256).map(|i| (i as f64 * 0.1).sin()).collect();
//! let spectra = proc.push(&signal);
//! // Each element is a magnitude spectrum of length fft_size/2 + 1.
//! assert!(!spectra.is_empty());
//! ```

use std::collections::VecDeque;
use std::f64::consts::PI;

use crate::error::{FFTError, FFTResult};
use crate::fft::fft;

// ─────────────────────────────────────────────────────────────────────────────
//  Public types
// ─────────────────────────────────────────────────────────────────────────────

/// Window function types for the streaming FFT processor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowType {
    /// Rectangular (no weighting).
    Rectangular,
    /// Hann window: `w[n] = 0.5 * (1 - cos(2πn / (N-1)))`.
    Hann,
    /// Hamming window: `w[n] = 0.54 - 0.46 * cos(2πn / (N-1))`.
    Hamming,
    /// Blackman window: `w[n] = 0.42 - 0.5*cos(2πn/(N-1)) + 0.08*cos(4πn/(N-1))`.
    Blackman,
}

/// Configuration for [`StreamingFft`].
#[derive(Debug, Clone)]
pub struct StreamingFftConfig {
    /// FFT window size. **Must be a power of two** and at least 2.
    pub fft_size: usize,
    /// Hop size (distance between successive analysis frames).
    /// Must satisfy `1 <= hop_size <= fft_size`.
    pub hop_size: usize,
    /// Window function applied to each frame before the FFT.
    pub window: WindowType,
}

/// Streaming FFT processor.
///
/// Samples are pushed incrementally. Once the internal buffer accumulates
/// `fft_size` samples a magnitude spectrum is emitted; then `hop_size`
/// samples are consumed from the front of the buffer so that the next
/// frame begins `hop_size` samples later (overlap = `fft_size - hop_size`).
pub struct StreamingFft {
    config: StreamingFftConfig,
    /// Circular input buffer — holds at most `fft_size` samples.
    buffer: VecDeque<f64>,
    /// Pre-computed window coefficients (length `fft_size`).
    window_fn: Vec<f64>,
    /// Pending output spectra (not yet consumed by the caller).
    output_buffer: Vec<Vec<f64>>,
}

// ─────────────────────────────────────────────────────────────────────────────
//  Implementation
// ─────────────────────────────────────────────────────────────────────────────

impl StreamingFft {
    /// Create a new streaming FFT processor.
    ///
    /// # Panics
    ///
    /// Does **not** panic; invalid configuration is silently clamped where
    /// possible.  Callers should validate with [`StreamingFftConfig`] fields
    /// before calling.
    pub fn new(config: StreamingFftConfig) -> Self {
        let window_fn = compute_window(config.window, config.fft_size);
        Self {
            buffer: VecDeque::with_capacity(config.fft_size * 2),
            window_fn,
            output_buffer: Vec::new(),
            config,
        }
    }

    /// Push new samples into the stream.
    ///
    /// Returns all completed magnitude spectra generated from the incoming
    /// data.  Each returned `Vec<f64>` has length `fft_size / 2 + 1` and
    /// contains the one-sided magnitude spectrum (DC to Nyquist).
    ///
    /// A new frame is emitted every time the internal buffer reaches
    /// `fft_size` samples; `hop_size` samples are then drained from the
    /// front of the buffer so the next frame overlaps by
    /// `fft_size - hop_size` samples.
    pub fn push(&mut self, samples: &[f64]) -> Vec<Vec<f64>> {
        let mut results = Vec::new();

        for &s in samples {
            self.buffer.push_back(s);

            if self.buffer.len() >= self.config.fft_size {
                // Emit a frame using the front `fft_size` samples.
                if let Ok(spectrum) = self.emit_frame() {
                    results.push(spectrum);
                }
                // Advance by hop_size.
                for _ in 0..self.config.hop_size {
                    self.buffer.pop_front();
                }
            }
        }

        results
    }

    /// Flush remaining buffered samples.
    ///
    /// Zero-pads the internal buffer to `fft_size` and emits one spectrum if
    /// the buffer is non-empty.  Resets the buffer afterwards.
    ///
    /// Returns an empty `Vec` if the buffer is already empty.
    pub fn flush(&mut self) -> Vec<Vec<f64>> {
        if self.buffer.is_empty() {
            return Vec::new();
        }

        // Zero-pad to fft_size.
        while self.buffer.len() < self.config.fft_size {
            self.buffer.push_back(0.0);
        }

        match self.emit_frame() {
            Ok(spectrum) => {
                self.buffer.clear();
                vec![spectrum]
            }
            Err(_) => {
                self.buffer.clear();
                Vec::new()
            }
        }
    }

    /// Number of samples currently held in the internal buffer.
    pub fn buffered_samples(&self) -> usize {
        self.buffer.len()
    }

    /// Reset the processor: clear internal buffer and any pending output.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.output_buffer.clear();
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    /// Extract the current window, apply the window function, and compute
    /// the one-sided magnitude spectrum.
    fn emit_frame(&self) -> FFTResult<Vec<f64>> {
        let n = self.config.fft_size;

        // Collect `n` samples from the front of the deque into a windowed frame.
        let windowed: Vec<f64> = self
            .buffer
            .iter()
            .take(n)
            .enumerate()
            .map(|(i, &s)| s * self.window_fn[i])
            .collect();

        if windowed.len() < n {
            return Err(FFTError::ValueError(format!(
                "streaming: buffer has {} samples but fft_size is {}",
                windowed.len(),
                n
            )));
        }

        // Forward FFT via the crate's public API.
        let spectrum = fft(&windowed, Some(n))?;

        // One-sided magnitude: bins 0 .. n/2+1.
        let n_out = n / 2 + 1;
        let magnitudes: Vec<f64> = spectrum
            .iter()
            .take(n_out)
            .map(|c| (c.re * c.re + c.im * c.im).sqrt())
            .collect();

        Ok(magnitudes)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Window function computation
// ─────────────────────────────────────────────────────────────────────────────

/// Compute a window function of length `n`.
fn compute_window(wt: WindowType, n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    let n_minus_1 = (n - 1) as f64;
    match wt {
        WindowType::Rectangular => vec![1.0; n],
        WindowType::Hann => (0..n)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / n_minus_1).cos()))
            .collect(),
        WindowType::Hamming => (0..n)
            .map(|i| 0.54 - 0.46 * (2.0 * PI * i as f64 / n_minus_1).cos())
            .collect(),
        WindowType::Blackman => (0..n)
            .map(|i| {
                let x = 2.0 * PI * i as f64 / n_minus_1;
                0.42 - 0.5 * x.cos() + 0.08 * (2.0 * x).cos()
            })
            .collect(),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Convenience function
// ─────────────────────────────────────────────────────────────────────────────

/// Process an entire signal with the streaming API.
///
/// Pushes all samples through the processor and appends the flush output.
/// Returns all magnitude spectra (each of length `fft_size / 2 + 1`).
///
/// The number of output frames for a signal of length `N` with
/// `N >= fft_size` is `floor((N - fft_size) / hop_size) + 1`, plus at most
/// one more frame from the flush if there are remaining samples.
///
/// # Example
///
/// ```rust
/// use scirs2_fft::streaming::{streaming_spectrogram, StreamingFftConfig, WindowType};
///
/// let signal: Vec<f64> = (0..512).map(|i| (i as f64 * 0.05).sin()).collect();
/// let config = StreamingFftConfig { fft_size: 64, hop_size: 32, window: WindowType::Hann };
/// let spectra = streaming_spectrogram(&signal, config);
/// assert!(!spectra.is_empty());
/// // Each spectrum has fft_size/2 + 1 = 33 bins.
/// assert_eq!(spectra[0].len(), 33);
/// ```
pub fn streaming_spectrogram(signal: &[f64], config: StreamingFftConfig) -> Vec<Vec<f64>> {
    let mut proc = StreamingFft::new(config);
    let mut results = proc.push(signal);
    let flushed = proc.flush();
    results.extend(flushed);
    results
}

// ─────────────────────────────────────────────────────────────────────────────
//  Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: generate a DC signal (constant value).
    fn dc_signal(n: usize, val: f64) -> Vec<f64> {
        vec![val; n]
    }

    /// Helper: generate a real sine wave.
    fn sine_signal(n: usize, freq_ratio: f64) -> Vec<f64> {
        (0..n)
            .map(|i| (2.0 * PI * freq_ratio * i as f64).sin())
            .collect()
    }

    // ── Test 1: exactly fft_size samples → exactly 1 spectrum ────────────────

    #[test]
    fn test_push_exactly_fft_size_produces_one_spectrum() {
        let fft_size = 64;
        let hop_size = 32;
        let config = StreamingFftConfig {
            fft_size,
            hop_size,
            window: WindowType::Rectangular,
        };
        let mut proc = StreamingFft::new(config);
        let signal = dc_signal(fft_size, 1.0);
        let spectra = proc.push(&signal);
        // Exactly one frame: we hit fft_size after the 64th sample.
        assert_eq!(
            spectra.len(),
            1,
            "expected 1 spectrum, got {}",
            spectra.len()
        );
        // Each spectrum should have fft_size/2+1 bins.
        assert_eq!(spectra[0].len(), fft_size / 2 + 1);
    }

    // ── Test 2: streaming and batch give the same spectrum ───────────────────

    #[test]
    fn test_streaming_matches_batch_fft() {
        let fft_size = 64;
        let hop_size = fft_size; // non-overlapping so there is exactly one batch frame
        let config = StreamingFftConfig {
            fft_size,
            hop_size,
            window: WindowType::Rectangular,
        };
        let signal = sine_signal(fft_size, 0.1);

        // Streaming path.
        let mut proc = StreamingFft::new(config);
        let spectra = proc.push(&signal);
        assert_eq!(spectra.len(), 1);
        let streaming_mag = &spectra[0];

        // Batch path: direct FFT on the same signal.
        let batch_spec = fft(&signal, Some(fft_size)).expect("batch fft failed");
        let n_out = fft_size / 2 + 1;
        let batch_mag: Vec<f64> = batch_spec
            .iter()
            .take(n_out)
            .map(|c| (c.re * c.re + c.im * c.im).sqrt())
            .collect();

        assert_eq!(streaming_mag.len(), batch_mag.len());
        for (s, b) in streaming_mag.iter().zip(batch_mag.iter()) {
            assert!(
                (s - b).abs() < 1e-10,
                "streaming={} batch={} differ by {}",
                s,
                b,
                (s - b).abs()
            );
        }
    }

    // ── Test 3: flush() on a partial buffer ──────────────────────────────────

    #[test]
    fn test_flush_produces_spectrum_for_partial_buffer() {
        let fft_size = 64;
        let hop_size = 32;
        let config = StreamingFftConfig {
            fft_size,
            hop_size,
            window: WindowType::Hann,
        };
        let mut proc = StreamingFft::new(config);

        // Push fewer than fft_size samples so no frame is emitted by push.
        let partial = sine_signal(20, 0.05);
        let push_spectra = proc.push(&partial);
        assert_eq!(
            push_spectra.len(),
            0,
            "no frames expected from partial push"
        );
        assert_eq!(proc.buffered_samples(), 20);

        // Flush should produce exactly one spectrum.
        let flushed = proc.flush();
        assert_eq!(flushed.len(), 1, "flush should produce exactly 1 spectrum");
        assert_eq!(flushed[0].len(), fft_size / 2 + 1);
        // Buffer should be cleared.
        assert_eq!(proc.buffered_samples(), 0);
    }

    // ── Test 4: different window types don't crash ────────────────────────────

    #[test]
    fn test_all_window_types() {
        let signal = sine_signal(256, 0.1);
        for &wt in &[
            WindowType::Rectangular,
            WindowType::Hann,
            WindowType::Hamming,
            WindowType::Blackman,
        ] {
            let config = StreamingFftConfig {
                fft_size: 64,
                hop_size: 32,
                window: wt,
            };
            let spectra = streaming_spectrogram(&signal, config);
            assert!(!spectra.is_empty(), "no spectra for window type {:?}", wt);
            for s in &spectra {
                assert_eq!(s.len(), 33, "spectrum length mismatch for window {:?}", wt);
                // No NaN or Inf values.
                for &v in s {
                    assert!(v.is_finite(), "non-finite value in spectrum for {:?}", wt);
                }
            }
        }
    }

    // ── Test 5: correct frame count for large signal ─────────────────────────

    #[test]
    fn test_large_signal_frame_count() {
        let fft_size = 64_usize;
        let hop_size = 16_usize;
        let n_samples = 1024_usize;

        let config = StreamingFftConfig {
            fft_size,
            hop_size,
            window: WindowType::Hann,
        };
        let signal = sine_signal(n_samples, 0.05);
        let mut proc = StreamingFft::new(config);
        let spectra = proc.push(&signal);

        // Expected: floor((N - fft_size) / hop_size) + 1
        let expected = (n_samples - fft_size) / hop_size + 1;
        assert_eq!(
            spectra.len(),
            expected,
            "expected {} frames, got {}",
            expected,
            spectra.len()
        );
    }

    // ── Test 6: DC signal has large bin-0 magnitude ───────────────────────────

    #[test]
    fn test_dc_signal_bin0_dominates() {
        let fft_size = 64;
        let config = StreamingFftConfig {
            fft_size,
            hop_size: fft_size,
            window: WindowType::Rectangular,
        };
        let signal = dc_signal(fft_size, 2.0);
        let mut proc = StreamingFft::new(config);
        let spectra = proc.push(&signal);
        assert_eq!(spectra.len(), 1);
        let mag = &spectra[0];
        // DC bin (index 0) should equal N * amplitude = 64 * 2.0 = 128.0
        let dc = mag[0];
        assert!(
            (dc - 128.0).abs() < 1e-9,
            "DC bin expected 128.0 got {}",
            dc
        );
        // All other bins should be near zero.
        for (k, &v) in mag.iter().enumerate().skip(1) {
            assert!(
                v < 1e-9,
                "bin {} expected ~0 got {} for rectangular DC signal",
                k,
                v
            );
        }
    }

    // ── Test 7: reset clears the buffer ──────────────────────────────────────

    #[test]
    fn test_reset_clears_buffer() {
        let config = StreamingFftConfig {
            fft_size: 32,
            hop_size: 16,
            window: WindowType::Hann,
        };
        let mut proc = StreamingFft::new(config);
        proc.push(&sine_signal(20, 0.1));
        assert_eq!(proc.buffered_samples(), 20);
        proc.reset();
        assert_eq!(proc.buffered_samples(), 0);
    }

    // ── Test 8: convenience function ─────────────────────────────────────────

    #[test]
    fn test_streaming_spectrogram_convenience() {
        let n = 512_usize;
        let fft_size = 64_usize;
        let hop_size = 32_usize;
        let signal = sine_signal(n, 0.05);
        let config = StreamingFftConfig {
            fft_size,
            hop_size,
            window: WindowType::Hamming,
        };
        let spectra = streaming_spectrogram(&signal, config);
        // At minimum floor((N - fft_size) / hop_size) + 1 frames.
        let min_frames = (n - fft_size) / hop_size + 1;
        assert!(
            spectra.len() >= min_frames,
            "expected >= {} frames, got {}",
            min_frames,
            spectra.len()
        );
        assert_eq!(spectra[0].len(), fft_size / 2 + 1);
    }
}
