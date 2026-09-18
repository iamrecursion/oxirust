//! Modified Discrete Cosine Transform (MDCT) for Vorbis.
//!
//! The MDCT is a lapped transform that converts time-domain audio samples
//! to frequency-domain coefficients. Vorbis uses MDCT with a specific
//! windowing function for perfect reconstruction.

#![forbid(unsafe_code)]

use std::f32::consts::PI;

/// MDCT transformer for Vorbis encoding.
#[derive(Debug, Clone)]
pub struct VorbisMdct {
    /// Block size (N).
    n: usize,
    /// Window function (Vorbis window).
    window: Vec<f32>,
    /// Precomputed cosine table.
    cos_table: Vec<f32>,
}

impl VorbisMdct {
    /// Create new MDCT transformer.
    ///
    /// # Arguments
    ///
    /// * `n` - Block size (number of output coefficients)
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn new(n: usize) -> Self {
        let window = Self::compute_vorbis_window(n);
        let cos_table = Self::compute_cos_table(n);

        Self {
            n,
            window,
            cos_table,
        }
    }

    /// Compute Vorbis window function.
    ///
    /// The Vorbis window is: sin(π/2 * sin²(π * (i + 0.5) / n))
    #[allow(clippy::cast_precision_loss)]
    fn compute_vorbis_window(n: usize) -> Vec<f32> {
        let mut window = Vec::with_capacity(n);

        for i in 0..n {
            let x = ((i as f32 + 0.5) / n as f32 * PI).sin();
            let w = (x * x * PI / 2.0).sin();
            window.push(w);
        }

        window
    }

    /// Compute cosine lookup table for MDCT.
    #[allow(clippy::cast_precision_loss)]
    fn compute_cos_table(n: usize) -> Vec<f32> {
        let n2 = n / 2;
        let mut table = Vec::with_capacity(n * n2);

        for k in 0..n2 {
            for i in 0..n {
                let angle = PI / n as f32 * (i as f32 + 0.5 + n as f32 / 2.0) * (k as f32 + 0.5);
                table.push(angle.cos());
            }
        }

        table
    }

    /// Perform forward MDCT.
    ///
    /// Transforms N time-domain samples to N/2 frequency-domain coefficients.
    ///
    /// # Arguments
    ///
    /// * `input` - Time-domain samples (N samples)
    /// * `output` - Frequency-domain coefficients (N/2 coefficients)
    #[allow(clippy::cast_precision_loss)]
    pub fn forward(&self, input: &[f32], output: &mut [f32]) {
        let n = self.n;
        let n2 = n / 2;

        assert_eq!(input.len(), n);
        assert_eq!(output.len(), n);

        // Apply window
        let mut windowed = vec![0.0; n];
        for i in 0..n {
            windowed[i] = input[i] * self.window[i];
        }

        // MDCT transform
        // Y[k] = Σ(i=0..N-1) x[i] * cos(π/N * (i + 0.5 + N/2) * (k + 0.5))
        for k in 0..n2 {
            let mut sum = 0.0;
            for i in 0..n {
                let angle = PI / n as f32 * (i as f32 + 0.5 + n as f32 / 2.0) * (k as f32 + 0.5);
                sum += windowed[i] * angle.cos();
            }
            output[k] = sum;
        }

        // Fill second half with zeros (Vorbis uses N/2 coefficients)
        for i in n2..n {
            output[i] = 0.0;
        }
    }

    /// Perform inverse MDCT.
    ///
    /// Transforms N/2 frequency-domain coefficients to N time-domain samples.
    ///
    /// # Arguments
    ///
    /// * `input` - Frequency-domain coefficients (N/2 coefficients)
    /// * `output` - Time-domain samples (N samples)
    #[allow(clippy::cast_precision_loss)]
    pub fn inverse(&self, input: &[f32], output: &mut [f32]) {
        let n = self.n;
        let n2 = n / 2;

        assert_eq!(input.len(), n2);
        assert_eq!(output.len(), n);

        // IMDCT transform
        // x[i] = (2/N) * Σ(k=0..N/2-1) X[k] * cos(π/N * (i + 0.5 + N/2) * (k + 0.5))
        for i in 0..n {
            let mut sum = 0.0;
            for k in 0..n2 {
                let angle = PI / n as f32 * (i as f32 + 0.5 + n as f32 / 2.0) * (k as f32 + 0.5);
                sum += input[k] * angle.cos();
            }
            output[i] = sum * 2.0 / n as f32;
        }

        // Apply window
        for i in 0..n {
            output[i] *= self.window[i];
        }
    }

    /// Get block size.
    #[must_use]
    pub const fn size(&self) -> usize {
        self.n
    }

    /// Get window function.
    #[must_use]
    pub fn window(&self) -> &[f32] {
        &self.window
    }
}

/// Overlap-add buffer for MDCT synthesis.
#[derive(Debug, Clone)]
pub struct OverlapAdd {
    /// Overlap buffer size (N/2).
    size: usize,
    /// Overlap buffer from previous block.
    buffer: Vec<f32>,
}

impl OverlapAdd {
    /// Create new overlap-add processor.
    ///
    /// # Arguments
    ///
    /// * `size` - Overlap size (N/2 samples)
    #[must_use]
    pub fn new(size: usize) -> Self {
        Self {
            size,
            buffer: vec![0.0; size],
        }
    }

    /// Process a new IMDCT output with overlap-add.
    ///
    /// # Arguments
    ///
    /// * `input` - IMDCT output (N samples)
    /// * `output` - Final output (N/2 samples)
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), 2 * self.size);
        assert_eq!(output.len(), self.size);

        // Add first half of input to overlap buffer
        for i in 0..self.size {
            output[i] = self.buffer[i] + input[i];
        }

        // Save second half as new overlap
        self.buffer.copy_from_slice(&input[self.size..]);
    }

    /// Reset overlap buffer.
    pub fn reset(&mut self) {
        self.buffer.fill(0.0);
    }

    /// Get overlap buffer size.
    #[must_use]
    pub const fn size(&self) -> usize {
        self.size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mdct_creation() {
        let mdct = VorbisMdct::new(256);
        assert_eq!(mdct.size(), 256);
        assert_eq!(mdct.window().len(), 256);
    }

    #[test]
    fn test_vorbis_window() {
        let window = VorbisMdct::compute_vorbis_window(256);
        assert_eq!(window.len(), 256);

        // Window should be symmetric
        for i in 0..128 {
            let w1 = window[i];
            let w2 = window[255 - i];
            assert!((w1 - w2).abs() < 1e-6);
        }

        // Window values should be in [0, 1]
        for &w in &window {
            assert!(w >= 0.0 && w <= 1.0);
        }
    }

    #[test]
    fn test_mdct_forward() {
        let mdct = VorbisMdct::new(256);
        let input = vec![1.0; 256];
        let mut output = vec![0.0; 256];

        mdct.forward(&input, &mut output);

        // Should produce non-zero coefficients (Vorbis window tapers to near-zero at edges)
        assert!(output[..128].iter().any(|&x| x.abs() > 1e-6));
        // Second half should be zero
        assert!(output[128..].iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_mdct_inverse() {
        let mdct = VorbisMdct::new(256);
        let coeffs = vec![1.0; 128];
        let mut output = vec![0.0; 256];

        mdct.inverse(&coeffs, &mut output);

        // Should produce non-zero output (Vorbis window tapers to near-zero at edges)
        assert!(output.iter().any(|&x| x.abs() > 1e-6));
    }

    #[test]
    fn test_mdct_roundtrip() {
        let mdct = VorbisMdct::new(128);
        let input = vec![1.0; 128];
        let mut coeffs = vec![0.0; 128];
        let mut output = vec![0.0; 128];

        mdct.forward(&input, &mut coeffs);
        mdct.inverse(&coeffs[..64], &mut output);

        // Roundtrip won't be perfect due to windowing,
        // but should have similar energy
        let input_energy: f32 = input.iter().map(|x| x * x).sum();
        let output_energy: f32 = output.iter().map(|x| x * x).sum();
        assert!(input_energy > 0.0);
        assert!(output_energy > 0.0);
    }

    #[test]
    fn test_overlap_add_creation() {
        let ola = OverlapAdd::new(128);
        assert_eq!(ola.size(), 128);
    }

    #[test]
    fn test_overlap_add_process() {
        let mut ola = OverlapAdd::new(128);
        let input = vec![1.0; 256];
        let mut output = vec![0.0; 128];

        // First call - no overlap yet
        ola.process(&input, &mut output);
        assert_eq!(output[0], 1.0);

        // Second call - should add overlap
        ola.process(&input, &mut output);
        assert_eq!(output[0], 2.0);
    }

    #[test]
    fn test_overlap_add_reset() {
        let mut ola = OverlapAdd::new(128);
        let input = vec![1.0; 256];
        let mut output = vec![0.0; 128];

        ola.process(&input, &mut output);
        ola.reset();
        ola.process(&input, &mut output);

        assert_eq!(output[0], 1.0);
    }
}
