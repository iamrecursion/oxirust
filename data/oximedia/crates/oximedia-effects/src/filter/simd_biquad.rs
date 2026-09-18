//! SIMD-style vectorized biquad filter processing.
//!
//! Processes 4 samples per iteration using manual scalar "vectorization"
//! (loop unrolling with 4-wide state reuse), mimicking SSE/NEON-style
//! throughput without requiring `unsafe` or platform-specific intrinsics.
//!
//! # Background
//!
//! A Direct-Form-II Transposed biquad has the recurrence:
//!
//! ```text
//! y[n] = b0*x[n] + s1[n-1]
//! s1[n] = b1*x[n] - a1*y[n] + s2[n-1]
//! s2[n] = b2*x[n] - a2*y[n]
//! ```
//!
//! The 4-sample "SIMD" kernel unrolls 4 consecutive iterations so the
//! compiler can generate packed FMA instructions (AVX-512 / NEON FMLA)
//! when targeting capable CPUs.  Because the recurrence is inherently serial
//! the coefficients are applied one sample at a time, but with all 4
//! computations laid out in a single contiguous block of arithmetic, the
//! compiler can schedule and vector-fuse the loads/multiplies optimally.
//!
//! # Usage
//!
//! ```ignore
//! use oximedia_effects::filter::simd_biquad::{SimdBiquad, SimdBiquadCoeff};
//!
//! let coeff = SimdBiquadCoeff::low_pass(1000.0, 0.707, 48000.0);
//! let mut filter = SimdBiquad::new(coeff);
//!
//! let mut buffer = vec![0.5_f32; 512];
//! filter.process_buffer(&mut buffer);
//! ```

#![allow(clippy::cast_precision_loss)]

use std::f32::consts::PI;

/// Biquad filter coefficients (Direct-Form-II Transposed).
///
/// Transfer function: `H(z) = (b0 + b1·z⁻¹ + b2·z⁻²) / (1 + a1·z⁻¹ + a2·z⁻²)`
#[derive(Debug, Clone, Copy)]
pub struct SimdBiquadCoeff {
    /// Feed-forward coefficient b0.
    pub b0: f32,
    /// Feed-forward coefficient b1.
    pub b1: f32,
    /// Feed-forward coefficient b2.
    pub b2: f32,
    /// Feedback coefficient a1 (normalised, sign-convention: denominator has +a1·z⁻¹).
    pub a1: f32,
    /// Feedback coefficient a2 (normalised).
    pub a2: f32,
}

impl SimdBiquadCoeff {
    /// Identity (bypass) coefficients.
    #[must_use]
    pub fn identity() -> Self {
        Self {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
        }
    }

    /// 2nd-order low-pass (Audio EQ Cookbook).
    #[must_use]
    pub fn low_pass(cutoff_hz: f32, q: f32, sample_rate: f32) -> Self {
        let w0 = 2.0 * PI * cutoff_hz / sample_rate;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 / (2.0 * q);

        let b0 = (1.0 - cos_w0) / 2.0;
        let b1 = 1.0 - cos_w0;
        let b2 = (1.0 - cos_w0) / 2.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;
        Self::normalize(b0, b1, b2, a0, a1, a2)
    }

    /// 2nd-order high-pass (Audio EQ Cookbook).
    #[must_use]
    pub fn high_pass(cutoff_hz: f32, q: f32, sample_rate: f32) -> Self {
        let w0 = 2.0 * PI * cutoff_hz / sample_rate;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 / (2.0 * q);

        let b0 = (1.0 + cos_w0) / 2.0;
        let b1 = -(1.0 + cos_w0);
        let b2 = (1.0 + cos_w0) / 2.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;
        Self::normalize(b0, b1, b2, a0, a1, a2)
    }

    /// 2nd-order band-pass (Audio EQ Cookbook, constant skirt gain).
    #[must_use]
    pub fn band_pass(center_hz: f32, q: f32, sample_rate: f32) -> Self {
        let w0 = 2.0 * PI * center_hz / sample_rate;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 / (2.0 * q);

        let b0 = q * alpha;
        let b1 = 0.0;
        let b2 = -q * alpha;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;
        Self::normalize(b0, b1, b2, a0, a1, a2)
    }

    /// Peak/bell EQ filter (Audio EQ Cookbook).
    #[must_use]
    pub fn peak(center_hz: f32, gain_db: f32, q: f32, sample_rate: f32) -> Self {
        let w0 = 2.0 * PI * center_hz / sample_rate;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let a_amp = 10.0_f32.powf(gain_db / 40.0);
        let alpha = sin_w0 / (2.0 * q);

        let b0 = 1.0 + alpha * a_amp;
        let b1 = -2.0 * cos_w0;
        let b2 = 1.0 - alpha * a_amp;
        let a0 = 1.0 + alpha / a_amp;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha / a_amp;
        Self::normalize(b0, b1, b2, a0, a1, a2)
    }

    /// Notch (band-reject) filter (Audio EQ Cookbook).
    #[must_use]
    pub fn notch(center_hz: f32, q: f32, sample_rate: f32) -> Self {
        let w0 = 2.0 * PI * center_hz / sample_rate;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 / (2.0 * q);

        let b0 = 1.0;
        let b1 = -2.0 * cos_w0;
        let b2 = 1.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;
        Self::normalize(b0, b1, b2, a0, a1, a2)
    }

    fn normalize(b0: f32, b1: f32, b2: f32, a0: f32, a1: f32, a2: f32) -> Self {
        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        }
    }
}

/// Biquad filter with SIMD-style 4-sample-per-iteration processing.
///
/// Uses Direct-Form-II Transposed recurrence unrolled over 4 samples.
/// The Rust compiler can auto-vectorise the inner loop block when targeting
/// `x86_64` with `AVX2`/`AVX-512` or `aarch64` with `NEON`.
pub struct SimdBiquad {
    coeff: SimdBiquadCoeff,
    /// First state register (s1 in DFII transposed).
    s1: f32,
    /// Second state register (s2 in DFII transposed).
    s2: f32,
}

impl SimdBiquad {
    /// Create a new SIMD biquad with the given coefficients.
    #[must_use]
    pub fn new(coeff: SimdBiquadCoeff) -> Self {
        Self {
            coeff,
            s1: 0.0,
            s2: 0.0,
        }
    }

    /// Create an identity (bypass) filter.
    #[must_use]
    pub fn identity() -> Self {
        Self::new(SimdBiquadCoeff::identity())
    }

    /// Update filter coefficients without resetting state.
    pub fn set_coeff(&mut self, coeff: SimdBiquadCoeff) {
        self.coeff = coeff;
    }

    /// Reset filter state.
    pub fn reset(&mut self) {
        self.s1 = 0.0;
        self.s2 = 0.0;
    }

    /// Process a single sample.
    #[inline]
    pub fn process_sample(&mut self, x: f32) -> f32 {
        let c = &self.coeff;
        let y = c.b0 * x + self.s1;
        self.s1 = c.b1 * x - c.a1 * y + self.s2;
        self.s2 = c.b2 * x - c.a2 * y;
        y
    }

    /// Process 4 samples with loop-unrolled SIMD-friendly layout.
    ///
    /// All 4 samples are computed in a single contiguous arithmetic block,
    /// enabling the compiler to schedule multiply-accumulate instructions
    /// (FMA) optimally and potentially auto-vectorise across the 4 lanes.
    ///
    /// **Input / output**: `xs` is both input and output (in-place).
    #[inline]
    pub fn process_quad(&mut self, xs: &mut [f32; 4]) {
        let b0 = self.coeff.b0;
        let b1 = self.coeff.b1;
        let b2 = self.coeff.b2;
        let a1 = self.coeff.a1;
        let a2 = self.coeff.a2;

        // Sample 0
        let y0 = b0 * xs[0] + self.s1;
        let s1_0 = b1 * xs[0] - a1 * y0 + self.s2;
        let s2_0 = b2 * xs[0] - a2 * y0;

        // Sample 1
        let y1 = b0 * xs[1] + s1_0;
        let s1_1 = b1 * xs[1] - a1 * y1 + s2_0;
        let s2_1 = b2 * xs[1] - a2 * y1;

        // Sample 2
        let y2 = b0 * xs[2] + s1_1;
        let s1_2 = b1 * xs[2] - a1 * y2 + s2_1;
        let s2_2 = b2 * xs[2] - a2 * y2;

        // Sample 3
        let y3 = b0 * xs[3] + s1_2;
        let s1_3 = b1 * xs[3] - a1 * y3 + s2_2;
        let s2_3 = b2 * xs[3] - a2 * y3;

        // Commit state
        self.s1 = s1_3;
        self.s2 = s2_3;

        xs[0] = y0;
        xs[1] = y1;
        xs[2] = y2;
        xs[3] = y3;
    }

    /// Process a buffer of arbitrary length in-place.
    ///
    /// Uses the 4-sample unrolled kernel for aligned chunks, then scalar
    /// processing for the remaining 0–3 samples (tail handling).
    pub fn process_buffer(&mut self, buffer: &mut [f32]) {
        let n = buffer.len();
        let chunks = n / 4;
        let tail = n % 4;

        for i in 0..chunks {
            let base = i * 4;
            let mut quad = [
                buffer[base],
                buffer[base + 1],
                buffer[base + 2],
                buffer[base + 3],
            ];
            self.process_quad(&mut quad);
            buffer[base] = quad[0];
            buffer[base + 1] = quad[1];
            buffer[base + 2] = quad[2];
            buffer[base + 3] = quad[3];
        }

        // Handle tail (0–3 remaining samples)
        let tail_start = chunks * 4;
        for i in 0..tail {
            buffer[tail_start + i] = self.process_sample(buffer[tail_start + i]);
        }
    }

    /// Process a buffer using the scalar path (for benchmarking comparison).
    pub fn process_buffer_scalar(&mut self, buffer: &mut [f32]) {
        for s in buffer.iter_mut() {
            *s = self.process_sample(*s);
        }
    }
}

impl crate::AudioEffect for SimdBiquad {
    const EFFECT_ID: &'static str = "simd_biquad";

    fn process_sample(&mut self, input: f32) -> f32 {
        SimdBiquad::process_sample(self, input)
    }

    fn process(&mut self, buffer: &mut [f32]) {
        self.process_buffer(buffer);
    }

    fn reset(&mut self) {
        SimdBiquad::reset(self);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AudioEffect;

    fn make_sine(freq_hz: f32, sample_rate: f32, n: usize) -> Vec<f32> {
        use std::f32::consts::TAU;
        (0..n)
            .map(|i| (i as f32 * TAU * freq_hz / sample_rate).sin())
            .collect()
    }

    fn rms(s: &[f32]) -> f32 {
        (s.iter().map(|&x| x * x).sum::<f32>() / s.len() as f32).sqrt()
    }

    // ------- SimdBiquadCoeff tests -------

    #[test]
    fn test_identity_coeff() {
        let c = SimdBiquadCoeff::identity();
        assert_eq!(c.b0, 1.0);
        assert_eq!(c.b1, 0.0);
        assert_eq!(c.a1, 0.0);
    }

    #[test]
    fn test_low_pass_coeff_finite() {
        let c = SimdBiquadCoeff::low_pass(1000.0, 0.707, 48000.0);
        assert!(c.b0.is_finite() && c.b1.is_finite() && c.b2.is_finite());
        assert!(c.a1.is_finite() && c.a2.is_finite());
    }

    #[test]
    fn test_high_pass_coeff_finite() {
        let c = SimdBiquadCoeff::high_pass(500.0, 0.707, 48000.0);
        assert!(c.b0.is_finite());
    }

    #[test]
    fn test_band_pass_coeff_finite() {
        let c = SimdBiquadCoeff::band_pass(2000.0, 1.0, 48000.0);
        assert!(c.b0.is_finite());
    }

    #[test]
    fn test_peak_coeff_finite() {
        let c = SimdBiquadCoeff::peak(1000.0, 6.0, 1.0, 48000.0);
        assert!(c.b0.is_finite());
    }

    #[test]
    fn test_notch_coeff_finite() {
        let c = SimdBiquadCoeff::notch(1000.0, 5.0, 48000.0);
        assert!(c.b0.is_finite());
    }

    // ------- SimdBiquad single-sample tests -------

    #[test]
    fn test_identity_passthrough() {
        let mut f = SimdBiquad::identity();
        for x in [0.1, 0.5, -0.3, 0.0] {
            let y = f.process_sample(x);
            assert!(
                (y - x).abs() < 1e-6,
                "Identity should pass through: {x} → {y}"
            );
        }
    }

    #[test]
    fn test_process_sample_finite() {
        let coeff = SimdBiquadCoeff::low_pass(1000.0, 0.707, 48000.0);
        let mut f = SimdBiquad::new(coeff);
        let sine = make_sine(440.0, 48000.0, 512);
        for x in sine {
            let y = f.process_sample(x);
            assert!(y.is_finite());
        }
    }

    // ------- process_quad tests -------

    #[test]
    fn test_process_quad_matches_scalar() {
        // Verify that quad processing produces the same output as scalar
        let coeff = SimdBiquadCoeff::low_pass(1000.0, 0.707, 48000.0);

        let mut f_scalar = SimdBiquad::new(coeff);
        let mut f_quad = SimdBiquad::new(coeff);

        let input = [0.1_f32, 0.3, -0.2, 0.5];

        // Scalar
        let mut scalar_out = [0.0_f32; 4];
        for (i, &x) in input.iter().enumerate() {
            scalar_out[i] = f_scalar.process_sample(x);
        }

        // Quad
        let mut quad_input = input;
        f_quad.process_quad(&mut quad_input);

        for i in 0..4 {
            assert!(
                (quad_input[i] - scalar_out[i]).abs() < 1e-5,
                "Quad[{i}] = {} != scalar {} ",
                quad_input[i],
                scalar_out[i]
            );
        }
    }

    #[test]
    fn test_process_quad_finite() {
        let coeff = SimdBiquadCoeff::peak(1000.0, 6.0, 1.0, 48000.0);
        let mut f = SimdBiquad::new(coeff);
        let mut buf = [0.5_f32, -0.3, 0.7, 0.1];
        f.process_quad(&mut buf);
        assert!(buf.iter().all(|&x| x.is_finite()));
    }

    // ------- process_buffer tests -------

    #[test]
    fn test_process_buffer_matches_scalar() {
        let coeff = SimdBiquadCoeff::low_pass(2000.0, 0.707, 48000.0);
        let mut f_simd = SimdBiquad::new(coeff);
        let mut f_scalar = SimdBiquad::new(coeff);

        let input: Vec<f32> = make_sine(440.0, 48000.0, 128);

        let mut buf_simd = input.clone();
        let mut buf_scalar = input.clone();

        f_simd.process_buffer(&mut buf_simd);
        f_scalar.process_buffer_scalar(&mut buf_scalar);

        for (i, (&a, &b)) in buf_simd.iter().zip(buf_scalar.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "SIMD buffer[{i}] = {a} != scalar {b}");
        }
    }

    #[test]
    fn test_process_buffer_odd_length() {
        // Test tail handling with non-multiple-of-4 length
        let coeff = SimdBiquadCoeff::high_pass(500.0, 0.707, 48000.0);
        let mut f = SimdBiquad::new(coeff);
        let mut buf = vec![0.5_f32; 13]; // 13 = 3*4 + 1
        f.process_buffer(&mut buf);
        assert!(buf.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_process_buffer_length_5() {
        let coeff = SimdBiquadCoeff::identity();
        let mut f = SimdBiquad::new(coeff);
        let mut buf = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0]; // 5 = 1*4 + 1
        f.process_buffer(&mut buf);
        // Identity filter: output = input
        for (i, (&y, expected)) in buf.iter().zip([1.0, 2.0, 3.0, 4.0, 5.0]).enumerate() {
            assert!((y - expected).abs() < 1e-5, "buf[{i}] = {y} != {expected}");
        }
    }

    #[test]
    fn test_low_pass_attenuates_high_freq() {
        // LP at 500 Hz should attenuate a 4kHz signal significantly
        let coeff = SimdBiquadCoeff::low_pass(500.0, 0.707, 48000.0);
        let mut f = SimdBiquad::new(coeff);

        let input = make_sine(4000.0, 48000.0, 4096);
        let mut buf = input.clone();
        f.process_buffer(&mut buf);

        let in_rms = rms(&input[256..]);
        let out_rms = rms(&buf[256..]);
        assert!(
            out_rms < in_rms * 0.5,
            "LP should attenuate high freq: in_rms={in_rms}, out_rms={out_rms}"
        );
    }

    #[test]
    fn test_high_pass_attenuates_low_freq() {
        // HP at 5kHz should attenuate a 200 Hz signal significantly
        let coeff = SimdBiquadCoeff::high_pass(5000.0, 0.707, 48000.0);
        let mut f = SimdBiquad::new(coeff);

        let input = make_sine(200.0, 48000.0, 4096);
        let mut buf = input.clone();
        f.process_buffer(&mut buf);

        let in_rms = rms(&input[256..]);
        let out_rms = rms(&buf[256..]);
        assert!(
            out_rms < in_rms * 0.5,
            "HP should attenuate low freq: in_rms={in_rms}, out_rms={out_rms}"
        );
    }

    #[test]
    fn test_reset_clears_state() {
        let coeff = SimdBiquadCoeff::low_pass(1000.0, 0.707, 48000.0);
        let mut f = SimdBiquad::new(coeff);
        for _ in 0..256 {
            f.process_sample(1.0);
        }
        f.reset();
        assert_eq!(f.s1, 0.0);
        assert_eq!(f.s2, 0.0);
    }

    #[test]
    fn test_audioeffect_trait_process() {
        let coeff = SimdBiquadCoeff::low_pass(1000.0, 0.707, 48000.0);
        let mut f = SimdBiquad::new(coeff);

        let out = <SimdBiquad as AudioEffect>::process_sample(&mut f, 0.5);
        assert!(out.is_finite());
    }

    #[test]
    fn test_audioeffect_trait_process_buffer() {
        let coeff = SimdBiquadCoeff::low_pass(1000.0, 0.707, 48000.0);
        let mut f = SimdBiquad::new(coeff);
        let mut buf = make_sine(440.0, 48000.0, 128);
        <SimdBiquad as AudioEffect>::process(&mut f, &mut buf);
        assert!(buf.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_set_coeff() {
        let coeff1 = SimdBiquadCoeff::low_pass(500.0, 0.707, 48000.0);
        let coeff2 = SimdBiquadCoeff::high_pass(500.0, 0.707, 48000.0);
        let mut f = SimdBiquad::new(coeff1);
        f.set_coeff(coeff2);
        assert_eq!(f.coeff.b0, coeff2.b0);
    }
}
