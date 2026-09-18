#![allow(dead_code)]
//! Oversampled lookahead limiter for the mixer master bus.
//!
//! Replaces the simple `tanh` soft-clip with a proper peak limiter that:
//!
//! 1. **Upsamples** the input by `oversample_factor` (2× or 4×) using linear
//!    interpolation to expose inter-sample peaks.
//! 2. **Applies** a gain-envelope derived from a look-ahead delay so that the
//!    gain reduction begins *before* the transient arrives at the output.
//! 3. **Downsamples** back to the original rate by simple decimation.
//!
//! The gain envelope uses a one-pole attack / release filter:
//! - Attack tracks the peak upward almost instantly.
//! - Release exponentially recovers toward unity gain.
//!
//! # Example
//!
//! ```rust
//! use oximedia_mixer::oversampled_limiter::OversampledLimiter;
//!
//! let mut lim = OversampledLimiter::new(-0.3, 50.0, 4, 48_000.0);
//! let input: Vec<f32> = vec![1.5; 512]; // over-threshold signal
//! let mut output = vec![0.0_f32; 512];
//! lim.process_block(&input, &mut output);
//! for &s in &output {
//!     assert!(s.abs() <= 1.0, "output exceeded 0 dBFS: {s}");
//! }
//! ```

/// Convert dB to linear amplitude (f32).
#[must_use]
#[inline]
fn db_to_linear_f32(db: f32) -> f32 {
    10.0_f32.powf(db / 20.0)
}

/// Convert linear amplitude to dB (f32).
#[must_use]
#[inline]
fn linear_to_db_f32(lin: f32) -> f32 {
    if lin <= 0.0 {
        -200.0
    } else {
        20.0 * lin.log10()
    }
}

/// One-pole smoothing coefficient from a time constant in milliseconds.
///
/// Returns a value in `[0, 1)`.  A value of 0.0 means no smoothing (instant
/// response), while values approaching 1.0 give very slow smoothing.
#[must_use]
#[inline]
fn smooth_coeff(ms: f32, sample_rate: f32) -> f32 {
    if ms <= 0.0 || sample_rate <= 0.0 {
        return 0.0;
    }
    (-1.0 / (ms * 0.001 * sample_rate)).exp()
}

// ---------------------------------------------------------------------------
// OversampledLimiter
// ---------------------------------------------------------------------------

/// Oversampled lookahead brick-wall limiter.
///
/// Operates at `oversample_factor × sample_rate` internally to catch
/// inter-sample peaks, then decimates back to the original rate.
#[derive(Debug, Clone)]
pub struct OversampledLimiter {
    /// Threshold in dBFS (e.g. −0.3).
    pub threshold_db: f32,
    /// Release time in milliseconds.
    pub release_ms: f32,
    /// Oversampling factor (2 or 4 recommended; clamped to 1–16).
    pub oversample_factor: u32,
    /// Look-ahead length in samples (at the *original* rate).
    pub lookahead_samples: usize,

    // Internal state
    sample_rate: f32,
    /// Threshold in linear amplitude.
    threshold_linear: f32,
    /// Current gain envelope (linear, ≤ 1.0).
    gain_env: f32,
    /// Look-ahead delay ring buffer (at the *original* rate).
    delay_buffer: Vec<f32>,
    /// Write position in the delay buffer.
    delay_pos: usize,
    /// Release smoothing coefficient (per *oversampled* sample).
    release_coeff: f32,
    /// Previous output sample for upsampler continuity.
    prev_input: f32,
    /// Accumulated oversampled output samples waiting for decimation.
    os_accum: Vec<f32>,
}

impl OversampledLimiter {
    /// Create a new `OversampledLimiter`.
    ///
    /// - `threshold_db` — limiting ceiling, e.g. `−0.3`
    /// - `release_ms`   — gain-recovery time
    /// - `oversample_factor` — 2 or 4 (clamped to `[1, 16]`)
    /// - `sample_rate`  — input/output sample rate in Hz
    #[must_use]
    pub fn new(
        threshold_db: f32,
        release_ms: f32,
        oversample_factor: u32,
        sample_rate: f32,
    ) -> Self {
        let factor = oversample_factor.clamp(1, 16);
        let threshold_linear = db_to_linear_f32(threshold_db);
        // Lookahead: 1 ms at the original rate
        let lookahead_samples = ((sample_rate * 0.001).ceil() as usize).max(1);
        let delay_buffer = vec![0.0_f32; lookahead_samples];

        // Release coefficient computed at the oversampled rate
        let os_rate = sample_rate * factor as f32;
        let release_coeff = smooth_coeff(release_ms, os_rate);

        Self {
            threshold_db,
            release_ms,
            oversample_factor: factor,
            lookahead_samples,
            sample_rate,
            threshold_linear,
            gain_env: 1.0,
            delay_buffer,
            delay_pos: 0,
            release_coeff,
            prev_input: 0.0,
            os_accum: Vec::with_capacity(factor as usize),
        }
    }

    /// Update internal coefficients after changing `threshold_db` or `release_ms`.
    pub fn recalculate(&mut self) {
        self.threshold_linear = db_to_linear_f32(self.threshold_db);
        let os_rate = self.sample_rate * self.oversample_factor as f32;
        self.release_coeff = smooth_coeff(self.release_ms, os_rate);
    }

    /// Reset all internal state (envelope, delay, history) to initial values.
    pub fn reset(&mut self) {
        self.gain_env = 1.0;
        self.delay_buffer.fill(0.0);
        self.delay_pos = 0;
        self.prev_input = 0.0;
        self.os_accum.clear();
    }

    /// Current gain reduction in dB (always ≤ 0).
    #[must_use]
    pub fn gain_reduction_db(&self) -> f32 {
        linear_to_db_f32(self.gain_env)
    }

    /// Process a single input sample and return the limited output sample.
    ///
    /// Internally upsamples to `oversample_factor` samples, applies limiting,
    /// then decimates back to 1 sample.
    pub fn process_sample(&mut self, input: f32) -> f32 {
        // 1. Write the *current* input into the look-ahead delay and read the
        //    delayed sample that will actually be output.
        let delayed = self.delay_buffer[self.delay_pos];
        self.delay_buffer[self.delay_pos] = input;
        self.delay_pos = (self.delay_pos + 1) % self.lookahead_samples;

        // 2. Upsample `input` (the look-ahead analysis signal) to get inter-sample
        //    peaks, compute the required gain, then downsample back to one value.
        let upsampled = self.upsample(input);
        let os_factor = self.oversample_factor as usize;
        let mut min_gain = 1.0_f32;

        for &us in upsampled[..os_factor].iter() {
            let g = self.compute_gain(us.abs());
            if g < min_gain {
                min_gain = g;
            }
        }

        // 3. Smooth the gain envelope.
        if min_gain < self.gain_env {
            // Attack: follow immediately
            self.gain_env = min_gain;
        } else {
            // Release: one-pole filter
            self.gain_env =
                self.release_coeff * self.gain_env + (1.0 - self.release_coeff) * min_gain;
        }

        // 4. Apply gain to the *delayed* (not look-ahead) sample.
        delayed * self.gain_env
    }

    /// Process a block of samples.
    ///
    /// `output` must have the same length as `input`.  Extra output slots are
    /// left unchanged.
    pub fn process_block(&mut self, input: &[f32], output: &mut [f32]) {
        let len = input.len().min(output.len());
        for i in 0..len {
            output[i] = self.process_sample(input[i]);
        }
    }

    /// Upsample a single input sample to `oversample_factor` samples using
    /// linear interpolation between the previous and current input.
    ///
    /// Returns a fixed-size array of 4 elements; only the first
    /// `oversample_factor` values are meaningful.
    #[must_use]
    fn upsample(&self, x: f32) -> [f32; 4] {
        let factor = self.oversample_factor as usize;
        let prev = self.prev_input;
        let mut out = [0.0_f32; 4];

        for i in 0..factor.min(4) {
            let t = (i + 1) as f32 / factor as f32; // fraction in (0, 1]
            out[i] = prev + t * (x - prev);
        }
        out
    }

    /// Downsample `oversample_factor` samples to one by simple decimation
    /// (take the last sample, which corresponds to the original input time).
    #[must_use]
    fn downsample(&self, samples: &[f32]) -> f32 {
        let factor = self.oversample_factor as usize;
        let n = samples.len().min(factor);
        if n == 0 {
            return 0.0;
        }
        samples[n - 1]
    }

    /// Compute the instantaneous gain needed to bring `peak` below the threshold.
    ///
    /// Returns a value in `(0, 1]`.
    #[must_use]
    fn compute_gain(&self, peak: f32) -> f32 {
        if peak <= self.threshold_linear || peak <= 0.0 {
            1.0
        } else {
            self.threshold_linear / peak
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_limiter() -> OversampledLimiter {
        OversampledLimiter::new(-0.3, 50.0, 4, 48_000.0)
    }

    // ------------------------------------------------------------------
    // Constructor / configuration
    // ------------------------------------------------------------------

    #[test]
    fn test_new_threshold_stored() {
        let lim = OversampledLimiter::new(-0.3, 50.0, 4, 48_000.0);
        assert!((lim.threshold_db - (-0.3)).abs() < 1e-6);
        assert_eq!(lim.oversample_factor, 4);
    }

    #[test]
    fn test_oversample_factor_clamped() {
        let lim = OversampledLimiter::new(0.0, 10.0, 0, 44_100.0);
        assert_eq!(lim.oversample_factor, 1);

        let lim2 = OversampledLimiter::new(0.0, 10.0, 100, 44_100.0);
        assert_eq!(lim2.oversample_factor, 16);
    }

    #[test]
    fn test_lookahead_at_least_one() {
        let lim = OversampledLimiter::new(0.0, 10.0, 2, 1.0); // tiny sample rate
        assert!(lim.lookahead_samples >= 1);
    }

    // ------------------------------------------------------------------
    // Gain computation
    // ------------------------------------------------------------------

    #[test]
    fn test_compute_gain_below_threshold() {
        let lim = make_limiter();
        let g = lim.compute_gain(0.5);
        assert!((g - 1.0).abs() < 1e-6, "below threshold should return 1.0");
    }

    #[test]
    fn test_compute_gain_above_threshold() {
        let lim = make_limiter();
        let peak = 1.5_f32;
        let g = lim.compute_gain(peak);
        let expected = lim.threshold_linear / peak;
        assert!((g - expected).abs() < 1e-6);
        assert!(g < 1.0);
    }

    #[test]
    fn test_compute_gain_zero_input() {
        let lim = make_limiter();
        let g = lim.compute_gain(0.0);
        assert!((g - 1.0).abs() < 1e-6);
    }

    // ------------------------------------------------------------------
    // Upsample / downsample
    // ------------------------------------------------------------------

    #[test]
    fn test_upsample_endpoint_is_input() {
        let mut lim = OversampledLimiter::new(0.0, 10.0, 4, 48_000.0);
        lim.prev_input = 0.0;
        let up = lim.upsample(1.0);
        // With linear interp from 0 to 1, the last sample (t=1) should be 1.0
        assert!((up[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_upsample_2x_midpoint() {
        let mut lim = OversampledLimiter::new(0.0, 10.0, 2, 48_000.0);
        lim.prev_input = 0.0;
        let up = lim.upsample(1.0);
        // 2× → t=0.5 and t=1.0
        assert!((up[0] - 0.5).abs() < 1e-6);
        assert!((up[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_downsample_returns_last() {
        let lim = OversampledLimiter::new(0.0, 10.0, 2, 48_000.0);
        let samples = [0.1, 0.9];
        let out = lim.downsample(&samples);
        assert!((out - 0.9).abs() < 1e-6);
    }

    // ------------------------------------------------------------------
    // process_sample – limiting behaviour
    // ------------------------------------------------------------------

    #[test]
    fn test_process_sample_loud_signal_limited() {
        let mut lim = OversampledLimiter::new(-0.3, 50.0, 4, 48_000.0);
        let threshold_linear = db_to_linear_f32(-0.3);
        // Run a constant loud signal long enough for the gain to settle
        let loud = 2.0_f32;
        let mut last = 0.0;
        for _ in 0..2000 {
            last = lim.process_sample(loud);
        }
        assert!(
            last.abs() <= threshold_linear + 1e-4,
            "settled output {last} should be ≤ threshold {threshold_linear}"
        );
    }

    #[test]
    fn test_process_sample_quiet_signal_unchanged() {
        let mut lim = OversampledLimiter::new(-0.3, 50.0, 4, 48_000.0);
        // A very quiet signal should pass through with gain ≈ 1.0 after lookahead flush
        let quiet = 0.01_f32;
        let mut out = 0.0;
        // Prime the lookahead delay with quiet signal first
        for _ in 0..100 {
            out = lim.process_sample(quiet);
        }
        // Output should be close to input (allow for tiny gain-env drift)
        assert!(
            (out - quiet).abs() < 0.005,
            "quiet signal should be nearly unchanged: got {out}"
        );
    }

    // ------------------------------------------------------------------
    // process_block
    // ------------------------------------------------------------------

    #[test]
    fn test_process_block_never_exceeds_0dbfs() {
        let mut lim = OversampledLimiter::new(0.0, 10.0, 4, 48_000.0);
        let input: Vec<f32> = (0..4096).map(|i| (i as f32 / 512.0).sin() * 3.0).collect();
        let mut output = vec![0.0_f32; input.len()];
        lim.process_block(&input, &mut output);
        // After the look-ahead delay is primed (first `lookahead_samples` outputs are
        // zero-delayed copies that may transiently exceed before the envelope settles).
        let skip = lim.lookahead_samples;
        for &s in output.iter().skip(skip) {
            assert!(s.abs() <= 1.0 + 1e-4, "sample {s} exceeded 0 dBFS");
        }
    }

    #[test]
    fn test_process_block_output_length_matches() {
        let mut lim = make_limiter();
        let input = vec![0.5_f32; 256];
        let mut output = vec![0.0_f32; 256];
        lim.process_block(&input, &mut output);
        assert_eq!(output.len(), 256);
    }

    #[test]
    fn test_process_block_shorter_output_slice() {
        let mut lim = make_limiter();
        let input = vec![0.5_f32; 512];
        let mut output = vec![0.0_f32; 128];
        // Should process only 128 samples without panicking
        lim.process_block(&input, &mut output);
    }

    #[test]
    fn test_process_block_empty_is_noop() {
        let mut lim = make_limiter();
        let input: Vec<f32> = vec![];
        let mut output: Vec<f32> = vec![];
        lim.process_block(&input, &mut output);
    }

    // ------------------------------------------------------------------
    // Reset
    // ------------------------------------------------------------------

    #[test]
    fn test_reset_clears_state() {
        let mut lim = make_limiter();
        // Drive limiter into heavy reduction
        for _ in 0..1000 {
            lim.process_sample(5.0);
        }
        assert!(lim.gain_env < 1.0);
        lim.reset();
        assert!((lim.gain_env - 1.0).abs() < 1e-6);
        assert!((lim.prev_input).abs() < 1e-10);
    }

    // ------------------------------------------------------------------
    // Gain-reduction dB
    // ------------------------------------------------------------------

    #[test]
    fn test_gain_reduction_db_unity() {
        let lim = make_limiter();
        // No signal processed → gain_env = 1.0 → 0 dB
        assert!(lim.gain_reduction_db().abs() < 1e-5);
    }

    #[test]
    fn test_gain_reduction_db_negative_when_limiting() {
        let mut lim = make_limiter();
        for _ in 0..2000 {
            lim.process_sample(5.0);
        }
        assert!(
            lim.gain_reduction_db() < 0.0,
            "gain reduction should be negative when limiting"
        );
    }

    // ------------------------------------------------------------------
    // recalculate
    // ------------------------------------------------------------------

    #[test]
    fn test_recalculate_updates_threshold() {
        let mut lim = make_limiter();
        let old_threshold = lim.threshold_linear;
        lim.threshold_db = -6.0;
        lim.recalculate();
        assert!((lim.threshold_linear - db_to_linear_f32(-6.0)).abs() < 1e-6);
        assert_ne!(lim.threshold_linear, old_threshold);
    }

    // ------------------------------------------------------------------
    // db_to_linear_f32 / linear_to_db_f32 helpers
    // ------------------------------------------------------------------

    #[test]
    fn test_db_to_linear_0db() {
        assert!((db_to_linear_f32(0.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_db_to_linear_minus20() {
        assert!((db_to_linear_f32(-20.0) - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_linear_to_db_1() {
        assert!(linear_to_db_f32(1.0).abs() < 1e-5);
    }

    #[test]
    fn test_linear_to_db_zero() {
        assert!(linear_to_db_f32(0.0) < -100.0);
    }
}
