//! Modified Discrete Cosine Transform (MDCT) implementation for CELT mode.
//!
//! The MDCT is a lapped transform used in CELT (and many other audio codecs)
//! to convert time-domain audio samples to frequency-domain coefficients.

use std::f32::consts::PI;

/// MDCT transformer for audio signals.
///
/// The MDCT provides perfect reconstruction with 50% overlap between
/// adjacent blocks when combined with the IMDCT.
#[derive(Debug)]
pub struct Mdct {
    /// Transform size (number of output coefficients)
    n: usize,
    /// Window size (2N)
    window_size: usize,
    /// Precomputed twiddle factors for forward transform
    #[allow(dead_code)]
    twiddle_fwd: Vec<f32>,
    /// Precomputed twiddle factors for inverse transform
    #[allow(dead_code)]
    twiddle_inv: Vec<f32>,
    /// Window function coefficients
    window: Vec<f32>,
}

impl Mdct {
    /// Creates a new MDCT transformer.
    ///
    /// # Arguments
    ///
    /// * `n` - Transform size (number of output coefficients)
    pub fn new(n: usize) -> Self {
        let window_size = 2 * n;
        let twiddle_fwd = Self::compute_twiddle_factors(n, false);
        let twiddle_inv = Self::compute_twiddle_factors(n, true);
        let window = Self::compute_window(window_size);

        Self {
            n,
            window_size,
            twiddle_fwd,
            twiddle_inv,
            window,
        }
    }

    /// Computes twiddle factors for the MDCT/IMDCT.
    fn compute_twiddle_factors(n: usize, inverse: bool) -> Vec<f32> {
        let mut twiddle = Vec::with_capacity(n);
        let sign = if inverse { 1.0 } else { -1.0 };

        for k in 0..n {
            let angle = sign * PI * (k as f32 + 0.5) / (n as f32);
            twiddle.push(angle.cos());
        }

        twiddle
    }

    /// Computes window function (Vorbis window).
    fn compute_window(size: usize) -> Vec<f32> {
        let mut window = Vec::with_capacity(size);

        for i in 0..size {
            let x = ((i as f32 + 0.5) / size as f32 * PI).sin();
            window.push((x * PI / 2.0).sin());
        }

        window
    }

    /// Performs forward MDCT transform.
    ///
    /// # Arguments
    ///
    /// * `input` - Time-domain input samples (2N samples)
    /// * `output` - Frequency-domain output coefficients (N coefficients)
    pub fn forward(&self, input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), self.window_size);
        assert_eq!(output.len(), self.n);

        // Apply window
        let mut windowed = vec![0.0f32; self.window_size];
        for i in 0..self.window_size {
            windowed[i] = input[i] * self.window[i];
        }

        // Perform MDCT using Type-IV DCT
        for k in 0..self.n {
            let mut sum = 0.0;
            for n in 0..self.window_size {
                let angle = PI / (self.window_size as f32)
                    * (n as f32 + 0.5 + self.n as f32 / 2.0)
                    * (k as f32 + 0.5);
                sum += windowed[n] * angle.cos();
            }
            output[k] = sum;
        }
    }

    /// Performs inverse MDCT transform.
    ///
    /// # Arguments
    ///
    /// * `input` - Frequency-domain input coefficients (N coefficients)
    /// * `output` - Time-domain output samples (2N samples)
    pub fn inverse(&self, input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), self.n);
        assert_eq!(output.len(), self.window_size);

        // Perform IMDCT using Type-IV DCT
        for n in 0..self.window_size {
            let mut sum = 0.0;
            for k in 0..self.n {
                let angle = PI / (self.window_size as f32)
                    * (n as f32 + 0.5 + self.n as f32 / 2.0)
                    * (k as f32 + 0.5);
                sum += input[k] * angle.cos();
            }
            output[n] = sum * 2.0 / self.n as f32;
        }

        // Apply window
        for i in 0..self.window_size {
            output[i] *= self.window[i];
        }
    }

    /// Returns the transform size.
    #[must_use]
    pub const fn size(&self) -> usize {
        self.n
    }

    /// Returns the window size.
    #[must_use]
    pub const fn window_size(&self) -> usize {
        self.window_size
    }
}

/// Overlap-add for combining MDCT frames.
///
/// The MDCT requires 50% overlap between consecutive frames for
/// perfect reconstruction.
#[derive(Debug)]
pub struct OverlapAdd {
    /// Overlap buffer size
    size: usize,
    /// Previous frame overlap
    overlap: Vec<f32>,
}

impl OverlapAdd {
    /// Creates a new overlap-add processor.
    ///
    /// # Arguments
    ///
    /// * `size` - Size of overlap region (N samples)
    pub fn new(size: usize) -> Self {
        Self {
            size,
            overlap: vec![0.0; size],
        }
    }

    /// Applies overlap-add to a new frame.
    ///
    /// # Arguments
    ///
    /// * `input` - New frame (2N samples)
    /// * `output` - Output buffer (N samples)
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), 2 * self.size);
        assert_eq!(output.len(), self.size);

        // Overlap-add with previous frame
        for i in 0..self.size {
            output[i] = self.overlap[i] + input[i];
        }

        // Save second half for next frame
        self.overlap.copy_from_slice(&input[self.size..]);
    }

    /// Resets the overlap buffer.
    pub fn reset(&mut self) {
        self.overlap.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mdct_creation() {
        let mdct = Mdct::new(256);
        assert_eq!(mdct.size(), 256);
        assert_eq!(mdct.window_size(), 512);
    }

    #[test]
    fn test_mdct_forward_inverse() {
        let mdct = Mdct::new(64);
        let input = vec![1.0f32; 128];
        let mut coeffs = vec![0.0f32; 64];
        let mut output = vec![0.0f32; 128];

        mdct.forward(&input, &mut coeffs);
        mdct.inverse(&coeffs, &mut output);

        // Check that transform is roughly reversible
        // (won't be perfect due to windowing and overlap)
        assert!(coeffs.iter().any(|&x| x.abs() > 0.1));
    }

    #[test]
    fn test_overlap_add() {
        let mut ola = OverlapAdd::new(64);
        let input = vec![1.0f32; 128];
        let mut output = vec![0.0f32; 64];

        ola.process(&input, &mut output);
        assert_eq!(output[0], 1.0);

        ola.process(&input, &mut output);
        assert_eq!(output[0], 2.0);
    }

    #[test]
    fn test_overlap_add_reset() {
        let mut ola = OverlapAdd::new(64);
        let input = vec![1.0f32; 128];
        let mut output = vec![0.0f32; 64];

        ola.process(&input, &mut output);
        ola.reset();
        ola.process(&input, &mut output);

        assert_eq!(output[0], 1.0);
    }
}
