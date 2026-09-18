//! Look-ahead compressor for transparent dynamic range control.
//!
//! Uses a short look-ahead delay to anticipate transients and apply
//! gain reduction before they occur, eliminating pumping artifacts.

#![allow(dead_code)]

/// Look-ahead configuration for a compressor.
#[derive(Debug, Clone)]
pub struct LookAhead {
    /// Look-ahead time in milliseconds.
    pub lookahead_ms: f32,
    /// Sample rate used to convert ms to samples.
    pub sample_rate: f32,
    delay_buffer: Vec<f32>,
    write_pos: usize,
}

impl LookAhead {
    /// Create a look-ahead delay of `lookahead_ms` milliseconds.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    #[must_use]
    pub fn new(lookahead_ms: f32, sample_rate: f32) -> Self {
        let delay_samples = ((lookahead_ms / 1000.0) * sample_rate) as usize;
        let size = delay_samples.max(1);
        Self {
            lookahead_ms,
            sample_rate,
            delay_buffer: vec![0.0; size],
            write_pos: 0,
        }
    }

    /// Returns `true` if the look-ahead is non-trivial (> 0 ms).
    #[must_use]
    pub fn has_look_ahead(&self) -> bool {
        self.lookahead_ms > 0.0
    }

    /// Delay in samples.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    #[must_use]
    pub fn delay_samples(&self) -> usize {
        ((self.lookahead_ms / 1000.0) * self.sample_rate) as usize
    }

    /// Push a sample into the delay; returns the delayed output sample.
    pub fn push(&mut self, input: f32) -> f32 {
        let out = self.delay_buffer[self.write_pos];
        self.delay_buffer[self.write_pos] = input;
        self.write_pos = (self.write_pos + 1) % self.delay_buffer.len();
        out
    }

    /// Reset the delay buffer.
    pub fn reset(&mut self) {
        self.delay_buffer.fill(0.0);
        self.write_pos = 0;
    }
}

/// Parameters controlling the compressor dynamics.
#[derive(Debug, Clone)]
pub struct CompressorParams {
    /// Threshold in dB (signals above this are compressed).
    pub threshold_db: f32,
    /// Compression ratio (e.g. 4.0 = 4:1).
    pub ratio: f32,
    /// Attack time in milliseconds.
    pub attack_ms: f32,
    /// Release time in milliseconds.
    pub release_ms: f32,
    /// Knee width in dB (0 = hard knee).
    pub knee_db: f32,
    /// Make-up gain in dB.
    pub makeup_db: f32,
}

impl Default for CompressorParams {
    fn default() -> Self {
        Self {
            threshold_db: -18.0,
            ratio: 4.0,
            attack_ms: 5.0,
            release_ms: 50.0,
            knee_db: 3.0,
            makeup_db: 0.0,
        }
    }
}

/// A look-ahead compressor instance.
pub struct CompressorLook {
    /// Compressor parameters.
    pub params: CompressorParams,
    lookahead: LookAhead,
    /// Current smoothed gain reduction in dB (negative = gain reduction).
    envelope_db: f32,
    sample_rate: f32,
    attack_coeff: f32,
    release_coeff: f32,
}

impl CompressorLook {
    /// Create a compressor with look-ahead.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn new(params: CompressorParams, lookahead_ms: f32, sample_rate: f32) -> Self {
        let attack_coeff = Self::time_to_coeff(params.attack_ms, sample_rate);
        let release_coeff = Self::time_to_coeff(params.release_ms, sample_rate);
        Self {
            params,
            lookahead: LookAhead::new(lookahead_ms, sample_rate),
            envelope_db: 0.0,
            sample_rate,
            attack_coeff,
            release_coeff,
        }
    }

    /// Convert a time constant (ms) to a one-pole smoothing coefficient.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    fn time_to_coeff(time_ms: f32, sample_rate: f32) -> f32 {
        if time_ms <= 0.0 {
            return 0.0;
        }
        (-1.0 / (time_ms * 0.001 * sample_rate)).exp()
    }

    /// Convert linear amplitude to dB.
    #[must_use]
    fn lin_to_db(lin: f32) -> f32 {
        if lin <= 1e-10 {
            -200.0
        } else {
            20.0 * lin.log10()
        }
    }

    /// Compute gain reduction (in dB, ≤ 0) for a given input level.
    #[must_use]
    pub fn compute_gain_reduction(&mut self, input_sample: f32) -> f32 {
        let level_db = Self::lin_to_db(input_sample.abs());
        let threshold = self.params.threshold_db;
        let ratio = self.params.ratio;
        let knee = self.params.knee_db;

        // Compute static gain reduction
        let target_gr = if level_db < threshold - knee / 2.0 {
            0.0
        } else if level_db > threshold + knee / 2.0 {
            (threshold - level_db) * (1.0 - 1.0 / ratio)
        } else {
            let x = level_db - threshold + knee / 2.0;
            (1.0 - 1.0 / ratio) * x * x / (2.0 * knee)
        };

        // Smooth with attack/release
        let coeff = if target_gr < self.envelope_db {
            self.attack_coeff
        } else {
            self.release_coeff
        };
        self.envelope_db = coeff * self.envelope_db + (1.0 - coeff) * target_gr;
        self.envelope_db
    }

    /// Process one input sample: delay, compress, and apply makeup gain.
    #[allow(clippy::cast_precision_loss)]
    pub fn process(&mut self, input: f32) -> f32 {
        let gain_db = self.compute_gain_reduction(input);
        let delayed = self.lookahead.push(input);
        let gain_lin = 10.0_f32.powf((gain_db + self.params.makeup_db) / 20.0);
        delayed * gain_lin
    }

    /// Release the gain envelope toward 0 dB (call with 0.0 input when idle).
    #[must_use]
    pub fn release_gain(&mut self) -> f32 {
        self.envelope_db *= self.release_coeff;
        self.envelope_db
    }

    /// Current envelope value in dB.
    #[must_use]
    pub fn envelope_db(&self) -> f32 {
        self.envelope_db
    }

    /// Returns `true` if look-ahead is active.
    #[must_use]
    pub fn has_look_ahead(&self) -> bool {
        self.lookahead.has_look_ahead()
    }

    /// Reset compressor and look-ahead state.
    pub fn reset(&mut self) {
        self.envelope_db = 0.0;
        self.lookahead.reset();
    }
}

/// Aggregate statistics gathered during compression.
#[derive(Debug, Clone, Default)]
pub struct CompressorStats {
    sample_count: u64,
    gain_reduction_sum_db: f64,
    peak_gain_reduction_db: f32,
    active_samples: u64,
}

impl CompressorStats {
    /// Create an empty stats tracker.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one compressor gain-reduction sample.
    pub fn record(&mut self, gain_reduction_db: f32) {
        self.sample_count += 1;
        self.gain_reduction_sum_db += f64::from(gain_reduction_db);
        if gain_reduction_db < self.peak_gain_reduction_db {
            self.peak_gain_reduction_db = gain_reduction_db;
        }
        if gain_reduction_db < -0.1 {
            self.active_samples += 1;
        }
    }

    /// Average gain reduction in dB across all recorded samples.
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    #[must_use]
    pub fn avg_gain_reduction_db(&self) -> f32 {
        if self.sample_count == 0 {
            return 0.0;
        }
        (self.gain_reduction_sum_db / self.sample_count as f64) as f32
    }

    /// Peak (most negative) gain reduction in dB.
    #[must_use]
    pub fn peak_gain_reduction_db(&self) -> f32 {
        self.peak_gain_reduction_db
    }

    /// Fraction of samples where gain reduction was active.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn activity_ratio(&self) -> f32 {
        if self.sample_count == 0 {
            return 0.0;
        }
        self.active_samples as f32 / self.sample_count as f32
    }

    /// Total recorded samples.
    #[must_use]
    pub fn sample_count(&self) -> u64 {
        self.sample_count
    }

    /// Reset all statistics.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lookahead_has_look_ahead_true() {
        let la = LookAhead::new(5.0, 48000.0);
        assert!(la.has_look_ahead());
    }

    #[test]
    fn test_lookahead_has_look_ahead_false() {
        let la = LookAhead::new(0.0, 48000.0);
        assert!(!la.has_look_ahead());
    }

    #[test]
    fn test_lookahead_delay_samples() {
        let la = LookAhead::new(1.0, 48000.0);
        assert_eq!(la.delay_samples(), 48);
    }

    #[test]
    fn test_lookahead_push_returns_delayed() {
        let mut la = LookAhead::new(0.0, 48000.0);
        // With 0ms look-ahead buffer size is 1 — push returns previous value
        la.push(0.5); // overwrites 0.0, returns 0.0
        let out = la.push(0.9); // returns 0.5
        assert!((out - 0.5).abs() < 1e-6, "got {out}");
    }

    #[test]
    fn test_lookahead_reset() {
        let mut la = LookAhead::new(1.0, 48000.0);
        for _ in 0..10 {
            la.push(1.0);
        }
        la.reset();
        // After reset, buffer is all zeros
        let out = la.push(0.0);
        assert_eq!(out, 0.0);
    }

    #[test]
    fn test_compressor_params_default() {
        let p = CompressorParams::default();
        assert!(p.threshold_db < 0.0);
        assert!(p.ratio > 1.0);
    }

    #[test]
    fn test_compressor_look_no_gain_reduction_below_threshold() {
        let params = CompressorParams {
            threshold_db: 0.0,
            ratio: 4.0,
            ..Default::default()
        };
        let mut comp = CompressorLook::new(params, 0.0, 48000.0);
        // Signal well below 0 dBFS — should produce near-zero gain reduction
        let gr = comp.compute_gain_reduction(0.001);
        assert!(gr <= 0.0);
        assert!(gr > -1.0, "expected near 0 GR, got {gr}");
    }

    #[test]
    fn test_compressor_look_gain_reduction_above_threshold() {
        let params = CompressorParams {
            threshold_db: -20.0,
            ratio: 10.0,
            attack_ms: 0.0,
            ..Default::default()
        };
        let mut comp = CompressorLook::new(params, 0.0, 48000.0);
        // Full-scale signal: level_db = 0, threshold = -20 → large GR
        // Run several samples to let attack settle
        for _ in 0..200 {
            let _ = comp.compute_gain_reduction(1.0);
        }
        let gr = comp.envelope_db();
        assert!(gr < -5.0, "expected significant GR, got {gr}");
    }

    #[test]
    fn test_compressor_look_has_look_ahead() {
        let comp = CompressorLook::new(Default::default(), 5.0, 48000.0);
        assert!(comp.has_look_ahead());
    }

    #[test]
    fn test_compressor_look_reset() {
        let mut comp = CompressorLook::new(Default::default(), 1.0, 48000.0);
        for _ in 0..100 {
            comp.process(0.9);
        }
        comp.reset();
        assert_eq!(comp.envelope_db(), 0.0);
    }

    #[test]
    fn test_release_gain_decays_toward_zero() {
        let params = CompressorParams {
            threshold_db: -20.0,
            ratio: 10.0,
            ..Default::default()
        };
        let mut comp = CompressorLook::new(params, 0.0, 48000.0);
        for _ in 0..200 {
            let _ = comp.compute_gain_reduction(1.0);
        }
        let initial_gr = comp.envelope_db();
        // Release with silence
        for _ in 0..100 {
            let _ = comp.release_gain();
        }
        let after_gr = comp.envelope_db();
        assert!(
            after_gr > initial_gr,
            "envelope should recover toward 0: initial={initial_gr}, after={after_gr}"
        );
    }

    #[test]
    fn test_stats_record_and_avg() {
        let mut stats = CompressorStats::new();
        stats.record(-3.0);
        stats.record(-6.0);
        let avg = stats.avg_gain_reduction_db();
        assert!((avg - (-4.5)).abs() < 0.01, "got avg={avg}");
    }

    #[test]
    fn test_stats_peak_gain_reduction() {
        let mut stats = CompressorStats::new();
        stats.record(-2.0);
        stats.record(-8.0);
        stats.record(-1.0);
        assert!((stats.peak_gain_reduction_db() - (-8.0)).abs() < 0.01);
    }

    #[test]
    fn test_stats_activity_ratio() {
        let mut stats = CompressorStats::new();
        stats.record(0.0); // not active
        stats.record(-3.0); // active
        stats.record(-5.0); // active
        let ratio = stats.activity_ratio();
        assert!((ratio - (2.0 / 3.0)).abs() < 0.01, "got {ratio}");
    }

    #[test]
    fn test_stats_reset() {
        let mut stats = CompressorStats::new();
        stats.record(-6.0);
        stats.reset();
        assert_eq!(stats.sample_count(), 0);
        assert_eq!(stats.avg_gain_reduction_db(), 0.0);
    }
}
