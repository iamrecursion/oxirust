//! Biquad filter implementation (IIR) using f32 arithmetic.
//!
//! This module provides a lightweight, self-contained second-order IIR filter
//! designed for real-time audio processing.  It complements the higher-precision
//! f64 implementation found in [`crate::dsp`] by offering faster throughput at
//! the cost of slightly reduced numerical precision – appropriate for most
//! sample-processing pipelines.
//!
//! # Filter designs
//!
//! | Constructor | Type |
//! |---|---|
//! | [`BiquadCoeffs::lowpass`] | Second-order Butterworth low-pass |
//! | [`BiquadCoeffs::highpass`] | Second-order Butterworth high-pass |
//! | [`BiquadCoeffs::bandpass`] | Constant-skirt band-pass |
//! | [`BiquadCoeffs::peaking_eq`] | Peaking parametric EQ |
//!
//! # Example
//!
//! ```
//! use oximedia_audio::biquad::{BiquadCoeffs, BiquadFilter};
//!
//! let coeffs = BiquadCoeffs::lowpass(1000.0, 0.707, 48000.0);
//! let mut filter = BiquadFilter::new(coeffs);
//! let output = filter.process(0.5);
//! assert!(output.is_finite());
//! ```

#![forbid(unsafe_code)]
#![allow(clippy::cast_precision_loss)]

use std::f32::consts::PI;

/// Biquad filter coefficients in direct form I.
///
/// The filter equation is:
/// `y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]`
///
/// All `a` coefficients are stored **normalised** (divided by a0).
#[derive(Clone, Debug, PartialEq)]
#[allow(dead_code)]
pub struct BiquadCoeffs {
    /// Feed-forward coefficient 0.
    pub b0: f32,
    /// Feed-forward coefficient 1.
    pub b1: f32,
    /// Feed-forward coefficient 2.
    pub b2: f32,
    /// Feedback coefficient 1 (normalised).
    pub a1: f32,
    /// Feedback coefficient 2 (normalised).
    pub a2: f32,
}

impl BiquadCoeffs {
    /// Create a pass-through (identity) coefficient set.
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

    /// Design a second-order Butterworth **low-pass** filter.
    ///
    /// # Arguments
    ///
    /// * `freq_hz` – Cutoff frequency in Hz.
    /// * `q` – Q factor (0.707 ≈ Butterworth; higher → resonant peak).
    /// * `sample_rate` – Sample rate in Hz.
    #[must_use]
    pub fn lowpass(freq_hz: f32, q: f32, sample_rate: f32) -> Self {
        let w0 = 2.0 * PI * freq_hz / sample_rate;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 / (2.0 * q);

        let b0 = (1.0 - cos_w0) / 2.0;
        let b1 = 1.0 - cos_w0;
        let b2 = (1.0 - cos_w0) / 2.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;

        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        }
    }

    /// Design a second-order Butterworth **high-pass** filter.
    ///
    /// # Arguments
    ///
    /// * `freq_hz` – Cutoff frequency in Hz.
    /// * `q` – Q factor.
    /// * `sample_rate` – Sample rate in Hz.
    #[must_use]
    pub fn highpass(freq_hz: f32, q: f32, sample_rate: f32) -> Self {
        let w0 = 2.0 * PI * freq_hz / sample_rate;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 / (2.0 * q);

        let b0 = (1.0 + cos_w0) / 2.0;
        let b1 = -(1.0 + cos_w0);
        let b2 = (1.0 + cos_w0) / 2.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;

        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        }
    }

    /// Design a constant-skirt **band-pass** filter.
    ///
    /// The bandwidth `bw_hz` is the –3 dB bandwidth in Hz.
    ///
    /// # Arguments
    ///
    /// * `freq_hz` – Centre frequency in Hz.
    /// * `bw_hz` – Bandwidth in Hz.
    /// * `sample_rate` – Sample rate in Hz.
    #[must_use]
    pub fn bandpass(freq_hz: f32, bw_hz: f32, sample_rate: f32) -> Self {
        let w0 = 2.0 * PI * freq_hz / sample_rate;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        // Convert bandwidth in Hz to Q
        let q = freq_hz / bw_hz.max(1e-6);
        let alpha = sin_w0 / (2.0 * q);

        let b0 = alpha;
        let b1 = 0.0_f32;
        let b2 = -alpha;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;

        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        }
    }

    /// Design a **peaking EQ** filter.
    ///
    /// # Arguments
    ///
    /// * `freq_hz` – Centre frequency in Hz.
    /// * `gain_db` – Gain at centre frequency in dB (positive = boost, negative = cut).
    /// * `q` – Q factor (bandwidth control).
    /// * `sample_rate` – Sample rate in Hz.
    #[must_use]
    pub fn peaking_eq(freq_hz: f32, gain_db: f32, q: f32, sample_rate: f32) -> Self {
        let w0 = 2.0 * PI * freq_hz / sample_rate;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 / (2.0 * q);
        let a = 10.0_f32.powf(gain_db / 40.0);

        let b0 = 1.0 + alpha * a;
        let b1 = -2.0 * cos_w0;
        let b2 = 1.0 - alpha * a;
        let a0 = 1.0 + alpha / a;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha / a;

        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        }
    }
}

/// Filter state (delay elements).
///
/// This struct stores the two previous input samples and two previous output
/// samples required by a direct-form-I biquad filter.
#[derive(Clone, Debug, Default)]
#[allow(dead_code)]
pub struct BiquadState {
    /// Previous input sample (x[n-1]).
    pub x1: f32,
    /// Input sample two steps back (x[n-2]).
    pub x2: f32,
    /// Previous output sample (y[n-1]).
    pub y1: f32,
    /// Output sample two steps back (y[n-2]).
    pub y2: f32,
}

impl BiquadState {
    /// Create a new zeroed state.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Reset all delay elements to zero.
    pub fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }
}

/// A complete biquad filter: coefficients + state.
///
/// `BiquadFilter` combines [`BiquadCoeffs`] and [`BiquadState`] into a
/// single entity for convenient per-sample or block processing.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct BiquadFilter {
    /// Filter coefficients.
    pub coeffs: BiquadCoeffs,
    /// Filter state (delay-line memory).
    pub state: BiquadState,
}

impl BiquadFilter {
    /// Create a new filter with the given coefficients and zeroed state.
    #[must_use]
    pub fn new(coeffs: BiquadCoeffs) -> Self {
        Self {
            coeffs,
            state: BiquadState::new(),
        }
    }

    /// Process a single sample and return the filtered output.
    pub fn process(&mut self, sample: f32) -> f32 {
        let output = self.coeffs.b0 * sample
            + self.coeffs.b1 * self.state.x1
            + self.coeffs.b2 * self.state.x2
            - self.coeffs.a1 * self.state.y1
            - self.coeffs.a2 * self.state.y2;

        self.state.x2 = self.state.x1;
        self.state.x1 = sample;
        self.state.y2 = self.state.y1;
        self.state.y1 = output;

        output
    }

    /// Process a block of samples, returning a new `Vec<f32>` with the filtered output.
    #[must_use]
    pub fn process_block(&mut self, samples: &[f32]) -> Vec<f32> {
        samples.iter().map(|&s| self.process(s)).collect()
    }

    /// Reset the filter state (delay-line memory) to zero.
    pub fn reset(&mut self) {
        self.state.reset();
    }

    /// Replace the filter coefficients (state is preserved).
    pub fn set_coeffs(&mut self, coeffs: BiquadCoeffs) {
        self.coeffs = coeffs;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Direct Form II Transposed (DF2T) biquad
// ─────────────────────────────────────────────────────────────────────────────

/// Biquad filter state for the Direct Form II Transposed topology.
///
/// DF2T uses only two delay elements (w1, w2) and avoids the "noise gain"
/// issue that affects Direct Form I when the filter poles are near the unit
/// circle.  This makes it the preferred topology for high-Q filters and
/// filters operating at high frequencies relative to the sample rate.
///
/// ## Filter equations
///
/// ```text
/// y[n]   =  b0·x[n] + w1[n-1]
/// w1[n]  =  b1·x[n] – a1·y[n] + w2[n-1]
/// w2[n]  =  b2·x[n] – a2·y[n]
/// ```
#[derive(Clone, Debug, Default)]
#[allow(dead_code)]
pub struct BiquadStateDf2t {
    /// First delay element.
    pub w1: f32,
    /// Second delay element.
    pub w2: f32,
}

impl BiquadStateDf2t {
    /// Create a new, zeroed DF2T state.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Reset the delay elements to zero.
    pub fn reset(&mut self) {
        self.w1 = 0.0;
        self.w2 = 0.0;
    }
}

/// Biquad filter in the **Direct Form II Transposed** topology.
///
/// Compared to [`BiquadFilter`] (Direct Form I), `BiquadFilterDf2t`:
///
/// - Uses **2 delay elements** instead of 4 → lower memory footprint.
/// - Has **better numerical stability** for high-Q resonant filters.
/// - Is the preferred choice for EQ bands close to Nyquist.
///
/// The coefficient design functions from [`BiquadCoeffs`] are reused since
/// the transfer function is identical; only the state-update equations differ.
///
/// # Example
///
/// ```
/// use oximedia_audio::biquad::{BiquadCoeffs, BiquadFilterDf2t};
///
/// let coeffs = BiquadCoeffs::peaking_eq(1000.0, 6.0, 1.0, 48000.0);
/// let mut filter = BiquadFilterDf2t::new(coeffs);
/// let output = filter.process(0.5);
/// assert!(output.is_finite());
/// ```
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct BiquadFilterDf2t {
    /// Shared coefficient set.
    pub coeffs: BiquadCoeffs,
    /// DF2T state.
    pub state: BiquadStateDf2t,
}

impl BiquadFilterDf2t {
    /// Create a new DF2T filter with the given coefficients and zeroed state.
    #[must_use]
    pub fn new(coeffs: BiquadCoeffs) -> Self {
        Self {
            coeffs,
            state: BiquadStateDf2t::new(),
        }
    }

    /// Process a single sample and return the filtered output.
    ///
    /// Implements the DF2T equations:
    ///
    /// ```text
    /// y   = b0·x + w1
    /// w1' = b1·x – a1·y + w2
    /// w2' = b2·x – a2·y
    /// ```
    #[inline]
    pub fn process(&mut self, x: f32) -> f32 {
        let y = self.coeffs.b0 * x + self.state.w1;
        let new_w1 = self.coeffs.b1 * x - self.coeffs.a1 * y + self.state.w2;
        let new_w2 = self.coeffs.b2 * x - self.coeffs.a2 * y;
        self.state.w1 = new_w1;
        self.state.w2 = new_w2;
        y
    }

    /// Process a block of samples and return the filtered output as a new `Vec<f32>`.
    #[must_use]
    pub fn process_block(&mut self, samples: &[f32]) -> Vec<f32> {
        samples.iter().map(|&s| self.process(s)).collect()
    }

    /// Process a block of samples in-place.
    pub fn process_block_inplace(&mut self, samples: &mut [f32]) {
        for s in samples.iter_mut() {
            *s = self.process(*s);
        }
    }

    /// Reset the filter state to silence.
    pub fn reset(&mut self) {
        self.state.reset();
    }

    /// Replace the filter coefficients without resetting state (smooth parameter change).
    pub fn set_coeffs(&mut self, coeffs: BiquadCoeffs) {
        self.coeffs = coeffs;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 48_000.0;

    // ── coefficient sanity ────────────────────────────────────────────────────

    #[test]
    fn test_identity_coefficients_values() {
        let c = BiquadCoeffs::identity();
        assert_eq!(c.b0, 1.0);
        assert_eq!(c.b1, 0.0);
        assert_eq!(c.b2, 0.0);
        assert_eq!(c.a1, 0.0);
        assert_eq!(c.a2, 0.0);
    }

    #[test]
    fn test_identity_is_passthrough() {
        let mut f = BiquadFilter::new(BiquadCoeffs::identity());
        let sample = 0.75_f32;
        assert!((f.process(sample) - sample).abs() < 1e-7);
    }

    #[test]
    fn test_lowpass_coeffs_are_finite() {
        let c = BiquadCoeffs::lowpass(1000.0, 0.707, SR);
        assert!(c.b0.is_finite() && c.b1.is_finite() && c.b2.is_finite());
        assert!(c.a1.is_finite() && c.a2.is_finite());
    }

    #[test]
    fn test_highpass_coeffs_are_finite() {
        let c = BiquadCoeffs::highpass(1000.0, 0.707, SR);
        assert!(c.b0.is_finite() && c.b1.is_finite() && c.b2.is_finite());
        assert!(c.a1.is_finite() && c.a2.is_finite());
    }

    #[test]
    fn test_bandpass_b1_is_zero() {
        // Band-pass constant-skirt: b1 is always zero, b0 = -b2
        let c = BiquadCoeffs::bandpass(1000.0, 200.0, SR);
        assert!(c.b1.abs() < 1e-7, "b1 should be zero, got {}", c.b1);
        assert!(
            (c.b0 + c.b2).abs() < 1e-6,
            "b0 should equal -b2, b0={} b2={}",
            c.b0,
            c.b2
        );
    }

    #[test]
    fn test_peaking_eq_zero_gain_is_nearly_identity() {
        // A peaking EQ with 0 dB gain should be a unity-gain filter at DC
        let c = BiquadCoeffs::peaking_eq(1000.0, 0.0, 1.0, SR);
        let mut f = BiquadFilter::new(c);
        // Feed DC and wait for transient to settle
        let mut out = 0.0_f32;
        for _ in 0..500 {
            out = f.process(1.0);
        }
        assert!(
            (out - 1.0).abs() < 0.02,
            "0 dB peaking should be near unity at DC, got {out}"
        );
    }

    // ── state management ──────────────────────────────────────────────────────

    #[test]
    fn test_state_reset_clears_memory() {
        let mut f = BiquadFilter::new(BiquadCoeffs::lowpass(500.0, 0.707, SR));
        for _ in 0..100 {
            f.process(1.0);
        }
        f.reset();
        // After reset, feeding silence should give silence
        let out = f.process(0.0);
        assert_eq!(out, 0.0, "Output after reset+silence should be 0");
    }

    #[test]
    fn test_biquad_state_new_is_zeroed() {
        let s = BiquadState::new();
        assert_eq!(s.x1, 0.0);
        assert_eq!(s.x2, 0.0);
        assert_eq!(s.y1, 0.0);
        assert_eq!(s.y2, 0.0);
    }

    // ── frequency-domain behaviour ────────────────────────────────────────────

    #[test]
    fn test_lowpass_passes_dc() {
        let mut f = BiquadFilter::new(BiquadCoeffs::lowpass(4000.0, 0.707, SR));
        let mut out = 0.0_f32;
        for _ in 0..2000 {
            out = f.process(1.0);
        }
        assert!(out > 0.9, "Low-pass should pass DC; got {out}");
    }

    #[test]
    fn test_highpass_blocks_dc() {
        let mut f = BiquadFilter::new(BiquadCoeffs::highpass(1000.0, 0.707, SR));
        let mut out = 0.0_f32;
        for _ in 0..2000 {
            out = f.process(1.0);
        }
        assert!(out.abs() < 0.01, "High-pass should block DC; got {out}");
    }

    #[test]
    fn test_peaking_boost_amplifies_at_center() {
        // Peaking EQ +12 dB at 1 kHz: drive the filter with a 1 kHz sine at
        // 48 kHz sample-rate and confirm the output envelope is louder than input.
        let c = BiquadCoeffs::peaking_eq(1000.0, 12.0, 1.0, SR);
        let mut f = BiquadFilter::new(c);
        let freq = 1000.0_f32;
        let mut peak_in = 0.0_f32;
        let mut peak_out = 0.0_f32;
        for i in 0..4800_usize {
            let s = (2.0 * PI * freq * i as f32 / SR).sin();
            peak_in = peak_in.max(s.abs());
            let y = f.process(s);
            if i > 960 {
                // Skip transient
                peak_out = peak_out.max(y.abs());
            }
        }
        assert!(
            peak_out > peak_in * 1.5,
            "Peaking boost should increase amplitude; in={peak_in} out={peak_out}"
        );
    }

    #[test]
    fn test_peaking_cut_attenuates_at_center() {
        let c = BiquadCoeffs::peaking_eq(1000.0, -12.0, 1.0, SR);
        let mut f = BiquadFilter::new(c);
        let freq = 1000.0_f32;
        let mut peak_out = 0.0_f32;
        for i in 0..4800_usize {
            let s = (2.0 * PI * freq * i as f32 / SR).sin();
            let y = f.process(s);
            if i > 960 {
                peak_out = peak_out.max(y.abs());
            }
        }
        assert!(
            peak_out < 0.5,
            "Peaking cut should attenuate; peak_out={peak_out}"
        );
    }

    // ── block processing ──────────────────────────────────────────────────────

    #[test]
    fn test_process_block_matches_per_sample() {
        let c = BiquadCoeffs::lowpass(2000.0, 0.707, SR);
        let input: Vec<f32> = (0..128).map(|i| (i as f32 * 0.05).sin()).collect();

        let mut f1 = BiquadFilter::new(c.clone());
        let expected: Vec<f32> = input.iter().map(|&s| f1.process(s)).collect();

        let mut f2 = BiquadFilter::new(c);
        let got = f2.process_block(&input);

        for (e, g) in expected.iter().zip(got.iter()) {
            assert!((e - g).abs() < 1e-6, "Mismatch: expected {e} got {g}");
        }
    }

    #[test]
    fn test_process_block_output_length_matches_input() {
        let mut f = BiquadFilter::new(BiquadCoeffs::highpass(500.0, 0.707, SR));
        let input = vec![0.1_f32; 256];
        let output = f.process_block(&input);
        assert_eq!(output.len(), input.len());
    }

    #[test]
    fn test_process_block_all_finite() {
        let mut f = BiquadFilter::new(BiquadCoeffs::bandpass(1000.0, 200.0, SR));
        let input: Vec<f32> = (0..256).map(|i| (i as f32).sin()).collect();
        let output = f.process_block(&input);
        assert!(
            output.iter().all(|x| x.is_finite()),
            "All outputs must be finite"
        );
    }

    // ── set_coeffs ────────────────────────────────────────────────────────────

    // ── DF2T biquad tests ─────────────────────────────────────────────────────

    #[test]
    fn test_df2t_identity_passthrough() {
        let mut f = BiquadFilterDf2t::new(BiquadCoeffs::identity());
        let out = f.process(0.75);
        assert!(
            (out - 0.75).abs() < 1e-7,
            "DF2T identity should pass sample; got {out}"
        );
    }

    #[test]
    fn test_df2t_lowpass_passes_dc() {
        let mut f = BiquadFilterDf2t::new(BiquadCoeffs::lowpass(4000.0, 0.707, SR));
        let mut out = 0.0_f32;
        for _ in 0..2000 {
            out = f.process(1.0);
        }
        assert!(out > 0.9, "DF2T LP should pass DC; got {out}");
    }

    #[test]
    fn test_df2t_highpass_blocks_dc() {
        let mut f = BiquadFilterDf2t::new(BiquadCoeffs::highpass(1000.0, 0.707, SR));
        let mut out = 0.0_f32;
        for _ in 0..2000 {
            out = f.process(1.0);
        }
        assert!(out.abs() < 0.01, "DF2T HP should block DC; got {out}");
    }

    #[test]
    fn test_df2t_reset_clears_state() {
        let mut f = BiquadFilterDf2t::new(BiquadCoeffs::lowpass(500.0, 0.707, SR));
        for _ in 0..100 {
            f.process(1.0);
        }
        f.reset();
        let out = f.process(0.0);
        assert_eq!(out, 0.0, "DF2T after reset+silence should give 0");
    }

    #[test]
    fn test_df2t_matches_df1_output() {
        // DF2T and DF1 share the same transfer function; outputs should match
        let coeffs = BiquadCoeffs::lowpass(2000.0, 0.707, SR);
        let mut df1 = BiquadFilter::new(coeffs.clone());
        let mut df2t = BiquadFilterDf2t::new(coeffs);
        for i in 0..512 {
            let x = (i as f32 * 0.1).sin();
            let y1 = df1.process(x);
            let y2 = df2t.process(x);
            assert!(
                (y1 - y2).abs() < 1e-5,
                "DF1 vs DF2T mismatch at sample {i}: {y1} vs {y2}"
            );
        }
    }

    #[test]
    fn test_df2t_block_length() {
        let mut f = BiquadFilterDf2t::new(BiquadCoeffs::bandpass(1000.0, 200.0, SR));
        let input = vec![0.5_f32; 256];
        let out = f.process_block(&input);
        assert_eq!(out.len(), 256);
    }

    #[test]
    fn test_df2t_block_inplace_matches_block() {
        let coeffs = BiquadCoeffs::peaking_eq(1000.0, 6.0, 1.0, SR);
        let input: Vec<f32> = (0..128).map(|i| (i as f32 * 0.05).sin()).collect();
        let mut f1 = BiquadFilterDf2t::new(coeffs.clone());
        let expected = f1.process_block(&input);
        let mut f2 = BiquadFilterDf2t::new(coeffs);
        let mut buf = input.clone();
        f2.process_block_inplace(&mut buf);
        for (e, g) in expected.iter().zip(buf.iter()) {
            assert!((e - g).abs() < 1e-6, "inplace vs block mismatch");
        }
    }

    #[test]
    fn test_set_coeffs_updates_filter() {
        let mut f = BiquadFilter::new(BiquadCoeffs::identity());
        // The identity filter should pass DC unchanged
        let out_before = f.process(1.0);
        assert!((out_before - 1.0).abs() < 1e-7);

        // Now switch to a high-pass that blocks DC
        f.set_coeffs(BiquadCoeffs::highpass(4000.0, 0.707, SR));
        f.reset();
        let mut out = 0.0_f32;
        for _ in 0..2000 {
            out = f.process(1.0);
        }
        assert!(
            out.abs() < 0.05,
            "HP should block DC after set_coeffs; got {out}"
        );
    }
}
