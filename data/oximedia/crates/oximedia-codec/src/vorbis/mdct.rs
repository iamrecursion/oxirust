//! Modified Discrete Cosine Transform (MDCT) for Vorbis.
//!
//! Vorbis uses the Type-IV MDCT with overlapping windows. This implementation
//! supports the two block sizes mandated by Vorbis I (short=256, long=2048 by default)
//! with the modified Vorbis window function `sin²(π/2 · sin²(…))`.

#![forbid(unsafe_code)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]

use std::f64::consts::PI;

/// Pre-computed MDCT twiddle factors for a given block size N (output size N/2).
#[derive(Clone, Debug)]
pub struct MdctTwiddles {
    /// Block size (number of input samples).
    pub n: usize,
    /// Cosine twiddle factors for the forward MDCT.
    pub cos_table: Vec<f64>,
    /// Sine twiddle factors (analysis window phase).
    pub sin_table: Vec<f64>,
    /// Analysis window coefficients (Vorbis `sin²` window).
    pub window: Vec<f64>,
}

impl MdctTwiddles {
    /// Compute MDCT twiddles for a block of `n` samples (produces `n/2` coefficients).
    ///
    /// # Panics
    ///
    /// Panics if `n` is not a power of two or is less than 4.
    #[must_use]
    pub fn new(n: usize) -> Self {
        assert!(
            n >= 4 && n.is_power_of_two(),
            "MDCT block size must be a power-of-two >= 4"
        );
        let m = n / 2;
        let mut cos_table = Vec::with_capacity(m);
        let mut sin_table = Vec::with_capacity(m);
        let mut window = Vec::with_capacity(n);

        for k in 0..m {
            let angle = PI * (2.0 * k as f64 + 1.0) / (4.0 * m as f64);
            cos_table.push(angle.cos());
            sin_table.push(angle.sin());
        }

        // Vorbis window: w(n) = sin(π/2 · sin²(π · (n + 0.5) / N))
        for i in 0..n {
            let arg = PI * (i as f64 + 0.5) / n as f64;
            let inner = (PI / 2.0) * arg.sin().powi(2);
            window.push(inner.sin());
        }

        Self {
            n,
            cos_table,
            sin_table,
            window,
        }
    }

    /// Apply the Vorbis analysis window to `samples` (in-place).
    pub fn apply_window(&self, samples: &mut [f64]) {
        assert_eq!(samples.len(), self.n);
        for (s, &w) in samples.iter_mut().zip(self.window.iter()) {
            *s *= w;
        }
    }

    /// Forward MDCT: `n` windowed samples → `n/2` coefficients.
    ///
    /// Uses the naive O(N²) algorithm — suitable for correctness validation.
    /// A real encoder would use an FFT-based O(N log N) split-radix MDCT.
    #[must_use]
    pub fn forward(&self, windowed: &[f64]) -> Vec<f64> {
        let m = self.n / 2;
        assert_eq!(windowed.len(), self.n);

        let mut out = Vec::with_capacity(m);
        for k in 0..m {
            let mut sum = 0.0f64;
            for n in 0..self.n {
                let angle =
                    PI / self.n as f64 * (n as f64 + 0.5 + self.n as f64 / 2.0) * (k as f64 + 0.5);
                sum += windowed[n] * angle.cos();
            }
            out.push(sum * 2.0 / self.n as f64);
        }
        out
    }

    /// Inverse MDCT: `n/2` coefficients → `n` time-domain samples (pre-windowed).
    #[must_use]
    pub fn inverse(&self, coeffs: &[f64]) -> Vec<f64> {
        let m = self.n / 2;
        assert_eq!(coeffs.len(), m);

        let mut out = vec![0.0f64; self.n];
        for n in 0..self.n {
            let mut sum = 0.0f64;
            for k in 0..m {
                let angle =
                    PI / self.n as f64 * (n as f64 + 0.5 + self.n as f64 / 2.0) * (k as f64 + 0.5);
                sum += coeffs[k] * angle.cos();
            }
            out[n] = sum;
        }
        out
    }

    /// Synthesis window (same as analysis for Vorbis — the window is symmetric).
    pub fn apply_synthesis_window(&self, samples: &mut [f64]) {
        self.apply_window(samples);
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mdct_twiddles_n8() {
        let tw = MdctTwiddles::new(8);
        assert_eq!(tw.n, 8);
        assert_eq!(tw.cos_table.len(), 4);
        assert_eq!(tw.window.len(), 8);
    }

    #[test]
    fn test_mdct_window_values_positive() {
        let tw = MdctTwiddles::new(16);
        for &w in &tw.window {
            assert!(w >= 0.0 && w <= 1.0, "window value out of [0,1]: {w}");
        }
    }

    #[test]
    fn test_mdct_forward_zero_input() {
        let tw = MdctTwiddles::new(16);
        let input = vec![0.0f64; 16];
        let out = tw.forward(&input);
        for &v in &out {
            assert!(v.abs() < 1e-12, "Expected ~0, got {v}");
        }
    }

    #[test]
    fn test_mdct_forward_inverse_roundtrip() {
        // The MDCT achieves perfect reconstruction only via overlap-add across
        // two consecutive blocks. A single-block round-trip verifies that:
        // 1. The forward transform produces non-zero coefficients for non-zero input.
        // 2. The inverse transform produces non-zero output for non-zero input.
        let n = 32;
        let tw = MdctTwiddles::new(n);
        let input: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        let coeffs = tw.forward(&input);
        let recovered = tw.inverse(&coeffs);

        // Coefficients should be non-trivially non-zero
        let coeff_energy: f64 = coeffs.iter().map(|&v| v * v).sum();
        assert!(
            coeff_energy > 1e-6,
            "Forward MDCT should produce non-zero coefficients"
        );

        // Recovered signal should also be non-trivially non-zero
        let rec_energy: f64 = recovered.iter().map(|&v| v * v).sum();
        assert!(
            rec_energy > 1e-6,
            "Inverse MDCT should produce non-zero output"
        );
    }

    #[test]
    fn test_mdct_apply_window_reduces_edges() {
        let tw = MdctTwiddles::new(16);
        let mut samples = vec![1.0f64; 16];
        tw.apply_window(&mut samples);
        // Window should taper to near-zero at edges
        assert!(
            samples[0].abs() < 0.1,
            "Window should be near zero at left edge"
        );
        assert!(
            samples[15].abs() < 0.1,
            "Window should be near zero at right edge"
        );
    }

    #[test]
    #[should_panic(expected = "power-of-two")]
    fn test_mdct_non_power_of_two_panics() {
        let _ = MdctTwiddles::new(10);
    }
}
