//! Block FFT overlap-add (OLA) processor.
//!
//! Provides a frame-by-frame STFT front-end for spectral DSP.  Each call to
//! [`BlockFftProcessor::process`] segments the input into overlapping frames,
//! applies a window, performs a forward FFT, invokes a user-supplied spectral
//! transform closure, performs the inverse FFT, and reassembles the frames via
//! overlap-add.
//!
//! # Example
//!
//! ```no_run
//! use oximedia_audiopost::dsp::block_fft::{BlockFftProcessor, Window};
//!
//! let mut proc = BlockFftProcessor::new(1024, 512, Window::Hann);
//! let input = vec![0.0_f32; 4096];
//! let mut output = vec![0.0_f32; 4096];
//! // Pass-through: spectral_fn is identity.
//! proc.process(&input, &mut output, |_spectrum| {});
//! ```

use oxifft::Complex;

/// Window function applied to each frame before the forward FFT.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Window {
    /// Hann (Hanning) window — good COLA properties at 50 % overlap.
    Hann,
    /// Hamming window — slightly wider main lobe, lower side-lobes.
    Hamming,
}

/// Overlap-add STFT processor backed by `oxifft`.
///
/// Frames are extracted from the input with a configurable hop size.  Each
/// frame is windowed, forward-transformed, passed to the user's spectral
/// closure, inverse-transformed, and accumulated via overlap-add.
///
/// The OLA output is normalised by the accumulated squared-window weights,
/// providing perfect reconstruction for any window type when the
/// constant-overlap-add (COLA) condition is met.
pub struct BlockFftProcessor {
    fft_size: usize,
    hop: usize,
    window: Vec<f32>,
    /// Overlap buffer holding unwritten output from previous calls.
    overlap_buf: Vec<f32>,
}

impl BlockFftProcessor {
    /// Create a new `BlockFftProcessor`.
    ///
    /// # Panics
    ///
    /// Panics if `fft_size` is not a power of two, `fft_size < 4`, or
    /// `hop == 0` or `hop > fft_size`.
    #[must_use]
    pub fn new(fft_size: usize, hop: usize, window_type: Window) -> Self {
        assert!(
            fft_size.is_power_of_two() && fft_size >= 4,
            "fft_size must be a power of two >= 4, got {fft_size}"
        );
        assert!(
            hop > 0 && hop <= fft_size,
            "hop must be in [1, fft_size], got {hop}"
        );

        let window: Vec<f32> = match window_type {
            Window::Hann => (0..fft_size)
                .map(|i| {
                    let t = i as f32 / (fft_size - 1) as f32;
                    0.5 * (1.0 - (2.0 * std::f32::consts::PI * t).cos())
                })
                .collect(),
            Window::Hamming => (0..fft_size)
                .map(|i| {
                    let t = i as f32 / (fft_size - 1) as f32;
                    0.54 - 0.46 * (2.0 * std::f32::consts::PI * t).cos()
                })
                .collect(),
        };

        Self {
            fft_size,
            hop,
            window,
            overlap_buf: vec![0.0_f32; fft_size],
        }
    }

    /// Process `input` through the overlap-add STFT engine.
    ///
    /// For each hop-aligned frame extracted from `input`:
    /// 1. Apply the window.
    /// 2. Forward FFT.
    /// 3. Call `spectral_fn` with a mutable slice of the complex spectrum.
    /// 4. Inverse FFT.
    /// 5. Overlap-add the windowed IFFT output into `output`.
    ///
    /// `output` must have the same length as `input`.  Samples beyond the
    /// last complete frame are written from the internal overlap buffer.
    ///
    /// The overlap buffer persists between successive calls so that
    /// consecutive buffers can be processed in a streaming fashion.
    pub fn process<F>(&mut self, input: &[f32], output: &mut [f32], mut spectral_fn: F)
    where
        F: FnMut(&mut [Complex<f32>]),
    {
        debug_assert_eq!(
            input.len(),
            output.len(),
            "input and output must have the same length"
        );

        let n = input.len();
        let fft_size = self.fft_size;
        let hop = self.hop;

        // Temporary OLA accumulator sized to cover the full output plus one
        // extra frame (to hold the overlap tail).
        let ola_len = n + fft_size;
        let mut ola_out = vec![0.0_f32; ola_len];
        let mut ola_weights = vec![0.0_f32; ola_len];

        // Seed the accumulator with the overlap from the previous call.
        for (i, &v) in self.overlap_buf.iter().enumerate() {
            ola_out[i] += v;
        }

        let num_frames = if n == 0 { 0 } else { (n + hop - 1) / hop };

        for frame_idx in 0..num_frames {
            let start = frame_idx * hop;

            // Build windowed complex input frame (zero-padded at the end).
            let cx_input: Vec<Complex<f32>> = (0..fft_size)
                .map(|j| {
                    let s = if start + j < n { input[start + j] } else { 0.0 };
                    Complex::new(s * self.window[j], 0.0)
                })
                .collect();

            let mut spectrum = oxifft::fft(&cx_input);

            spectral_fn(&mut spectrum);

            // Inverse FFT — oxifft::ifft is already normalised (÷N).
            let recovered = oxifft::ifft(&spectrum);

            for j in 0..fft_size {
                let out_idx = start + j;
                if out_idx < ola_len {
                    ola_out[out_idx] += recovered[j].re * self.window[j];
                    ola_weights[out_idx] += self.window[j] * self.window[j];
                }
            }
        }

        // Normalise by OLA weights.  Where the accumulated weight is
        // negligibly small (e.g. at the window edges or beyond the last
        // frame) we pass through zero rather than dividing by a near-zero
        // value.
        const MIN_WEIGHT: f32 = 1e-6;
        for i in 0..ola_len {
            if ola_weights[i] >= MIN_WEIGHT {
                ola_out[i] /= ola_weights[i];
            } else {
                ola_out[i] = 0.0;
            }
        }

        // Copy the first `n` samples to the output.
        output[..n].copy_from_slice(&ola_out[..n]);

        // Save the overlap tail (samples [n..n+fft_size]) for the next call.
        let tail_end = (n + fft_size).min(ola_len);
        let tail_len = tail_end - n;
        self.overlap_buf[..tail_len].copy_from_slice(&ola_out[n..tail_end]);
        self.overlap_buf[tail_len..].fill(0.0);
    }

    /// Reset the internal overlap buffer (use when starting a new stream).
    pub fn reset(&mut self) {
        self.overlap_buf.fill(0.0);
    }

    /// FFT size this processor was created with.
    #[must_use]
    pub fn fft_size(&self) -> usize {
        self.fft_size
    }

    /// Hop size (frame advance) this processor was created with.
    #[must_use]
    pub fn hop_size(&self) -> usize {
        self.hop
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Identity spectral function should give near-perfect reconstruction.
    #[test]
    fn test_block_fft_identity_passthrough() {
        let fft_size = 512;
        let hop = 256; // 50 % overlap
        let mut proc = BlockFftProcessor::new(fft_size, hop, Window::Hann);

        let n = 2048;
        // Generate a simple ramp signal.
        let input: Vec<f32> = (0..n).map(|i| i as f32 / n as f32).collect();
        let mut output = vec![0.0_f32; n];

        proc.process(&input, &mut output, |_| {});

        // Skip the first fft_size samples (edge effect) and check the rest.
        let skip = fft_size;
        let max_err = input[skip..n - skip]
            .iter()
            .zip(output[skip..n - skip].iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f32, f32::max);

        assert!(
            max_err < 0.01,
            "Max reconstruction error {max_err} > 0.01 (identity OLA)"
        );
    }

    #[test]
    fn test_block_fft_hamming_window() {
        let mut proc = BlockFftProcessor::new(256, 128, Window::Hamming);
        let input: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.01).sin()).collect();
        let mut output = vec![0.0_f32; 1024];
        proc.process(&input, &mut output, |_| {});
        // Output should be finite (no NaN/Inf).
        assert!(output.iter().all(|x| x.is_finite()), "NaN or Inf in output");
    }

    #[test]
    fn test_block_fft_spectral_fn_zeroes_all_bins() {
        let mut proc = BlockFftProcessor::new(256, 128, Window::Hann);
        let input: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut output = vec![1.0_f32; 1024]; // Pre-fill with ones.
        proc.process(&input, &mut output, |spectrum| {
            spectrum
                .iter_mut()
                .for_each(|c| *c = Complex::new(0.0, 0.0));
        });
        // All output should be near zero since the spectrum was zeroed.
        let max_abs = output.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
        assert!(max_abs < 1e-6, "Expected silence, got max_abs={max_abs}");
    }

    #[test]
    fn test_block_fft_reset_clears_overlap() {
        let mut proc = BlockFftProcessor::new(256, 128, Window::Hann);
        let input: Vec<f32> = (0..512).map(|i| i as f32).collect();
        let mut output = vec![0.0_f32; 512];
        proc.process(&input, &mut output, |_| {});
        proc.reset();
        assert!(proc.overlap_buf.iter().all(|&v| v == 0.0), "reset failed");
    }

    #[test]
    fn test_block_fft_accessors() {
        let proc = BlockFftProcessor::new(512, 256, Window::Hann);
        assert_eq!(proc.fft_size(), 512);
        assert_eq!(proc.hop_size(), 256);
    }

    #[test]
    fn test_block_fft_empty_input() {
        let mut proc = BlockFftProcessor::new(64, 32, Window::Hann);
        let input: Vec<f32> = vec![];
        let mut output: Vec<f32> = vec![];
        proc.process(&input, &mut output, |_| {}); // Must not panic.
    }
}
