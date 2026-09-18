#![allow(dead_code)]
//! Brick-wall and look-ahead limiter for the mixer's master bus.
//!
//! This module provides peak limiting for broadcast and mastering workflows.
//! It includes a simple brick-wall clipper, a soft-knee limiter, and a
//! look-ahead limiter that smoothly reduces gain before transients arrive
//! to avoid hard clipping artifacts.

/// Limiter operating mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LimiterMode {
    /// Hard clip — samples exceeding the ceiling are clipped instantly.
    HardClip,
    /// Soft knee — gradual gain reduction near the ceiling.
    SoftKnee,
    /// Look-ahead — uses a delay buffer to anticipate peaks.
    LookAhead,
}

/// Configuration for a limiter instance.
#[derive(Debug, Clone)]
pub struct LimiterConfig {
    /// Operating mode.
    pub mode: LimiterMode,
    /// Output ceiling in dB (e.g. -0.3 dBFS).
    pub ceiling_db: f64,
    /// Attack time in milliseconds.
    pub attack_ms: f64,
    /// Release time in milliseconds.
    pub release_ms: f64,
    /// Look-ahead time in milliseconds (only for `LookAhead` mode).
    pub lookahead_ms: f64,
    /// Soft knee width in dB (only for `SoftKnee` mode).
    pub knee_db: f64,
    /// Sample rate in Hz.
    pub sample_rate: u32,
}

impl Default for LimiterConfig {
    fn default() -> Self {
        Self {
            mode: LimiterMode::LookAhead,
            ceiling_db: -0.3,
            attack_ms: 0.5,
            release_ms: 50.0,
            lookahead_ms: 1.0,
            knee_db: 3.0,
            sample_rate: 48000,
        }
    }
}

/// Convert dB to linear amplitude.
#[must_use]
fn db_to_linear(db: f64) -> f64 {
    10.0_f64.powf(db / 20.0)
}

/// Convert linear amplitude to dB.
#[must_use]
fn linear_to_db(lin: f64) -> f64 {
    if lin <= 0.0 {
        -200.0
    } else {
        20.0 * lin.log10()
    }
}

/// Compute a one-pole smoothing coefficient from a time constant in ms.
#[allow(clippy::cast_precision_loss)]
#[must_use]
fn time_constant(ms: f64, sample_rate: u32) -> f64 {
    if ms <= 0.0 {
        return 0.0;
    }
    (-1.0 / (ms * 0.001 * f64::from(sample_rate))).exp()
}

/// Gain reduction metering data.
#[derive(Debug, Clone, Copy, Default)]
pub struct LimiterMetrics {
    /// Current gain reduction in dB (always <= 0).
    pub gain_reduction_db: f64,
    /// Peak input level in dB.
    pub peak_input_db: f64,
    /// Peak output level in dB.
    pub peak_output_db: f64,
    /// Number of samples that hit the ceiling in the last buffer.
    pub clipped_samples: u64,
}

/// Peak limiter processor.
#[derive(Debug, Clone)]
pub struct Limiter {
    /// Current configuration.
    config: LimiterConfig,
    /// Ceiling in linear amplitude.
    ceiling_linear: f64,
    /// Attack coefficient.
    attack_coeff: f64,
    /// Release coefficient.
    release_coeff: f64,
    /// Current envelope follower level.
    envelope: f64,
    /// Gain reduction in linear.
    gain_reduction: f64,
    /// Look-ahead delay buffer (interleaved).
    delay_buffer: Vec<f64>,
    /// Write position in the delay buffer.
    delay_write: usize,
    /// Delay length in frames.
    delay_frames: usize,
    /// Number of channels.
    num_channels: usize,
    /// Metrics from the last buffer.
    metrics: LimiterMetrics,
}

impl Limiter {
    /// Create a new limiter.
    #[must_use]
    pub fn new(config: LimiterConfig, num_channels: usize) -> Self {
        let ceiling_linear = db_to_linear(config.ceiling_db);
        let attack_coeff = time_constant(config.attack_ms, config.sample_rate);
        let release_coeff = time_constant(config.release_ms, config.sample_rate);

        #[allow(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss
        )]
        let delay_frames =
            (config.lookahead_ms * 0.001 * f64::from(config.sample_rate)).ceil() as usize;

        let delay_buffer = vec![0.0; delay_frames * num_channels.max(1)];

        Self {
            config,
            ceiling_linear,
            attack_coeff,
            release_coeff,
            envelope: 0.0,
            gain_reduction: 1.0,
            delay_buffer,
            delay_write: 0,
            delay_frames,
            num_channels,
            metrics: LimiterMetrics::default(),
        }
    }

    /// Create a limiter with default settings.
    #[must_use]
    pub fn with_defaults(num_channels: usize) -> Self {
        Self::new(LimiterConfig::default(), num_channels)
    }

    /// Get current metrics.
    #[must_use]
    pub fn metrics(&self) -> &LimiterMetrics {
        &self.metrics
    }

    /// Get the configuration.
    #[must_use]
    pub fn config(&self) -> &LimiterConfig {
        &self.config
    }

    /// Set a new ceiling in dB.
    pub fn set_ceiling_db(&mut self, db: f64) {
        self.config.ceiling_db = db;
        self.ceiling_linear = db_to_linear(db);
    }

    /// Set the attack time in ms.
    pub fn set_attack_ms(&mut self, ms: f64) {
        self.config.attack_ms = ms;
        self.attack_coeff = time_constant(ms, self.config.sample_rate);
    }

    /// Set the release time in ms.
    pub fn set_release_ms(&mut self, ms: f64) {
        self.config.release_ms = ms;
        self.release_coeff = time_constant(ms, self.config.sample_rate);
    }

    /// Reset internal state.
    pub fn reset(&mut self) {
        self.envelope = 0.0;
        self.gain_reduction = 1.0;
        for v in &mut self.delay_buffer {
            *v = 0.0;
        }
        self.delay_write = 0;
        self.metrics = LimiterMetrics::default();
    }

    /// Process a buffer of interleaved samples in-place.
    ///
    /// * `buffer` — interleaved audio samples (modified in-place)
    /// * `num_channels` — number of interleaved channels
    pub fn process_buffer(&mut self, buffer: &mut [f64], num_channels: usize) {
        if num_channels == 0 || buffer.is_empty() {
            return;
        }

        let num_frames = buffer.len() / num_channels;
        let mut peak_in = 0.0_f64;
        let mut peak_out = 0.0_f64;
        let mut clipped = 0_u64;

        match self.config.mode {
            LimiterMode::HardClip => {
                self.process_hard_clip(
                    buffer,
                    num_channels,
                    num_frames,
                    &mut peak_in,
                    &mut peak_out,
                    &mut clipped,
                );
            }
            LimiterMode::SoftKnee => {
                self.process_soft_knee(
                    buffer,
                    num_channels,
                    num_frames,
                    &mut peak_in,
                    &mut peak_out,
                );
            }
            LimiterMode::LookAhead => {
                self.process_lookahead(
                    buffer,
                    num_channels,
                    num_frames,
                    &mut peak_in,
                    &mut peak_out,
                );
            }
        }

        self.metrics.peak_input_db = linear_to_db(peak_in);
        self.metrics.peak_output_db = linear_to_db(peak_out);
        self.metrics.gain_reduction_db = linear_to_db(self.gain_reduction);
        self.metrics.clipped_samples = clipped;
    }

    /// Hard-clip processing.
    #[allow(clippy::too_many_arguments)]
    fn process_hard_clip(
        &self,
        buffer: &mut [f64],
        num_channels: usize,
        num_frames: usize,
        peak_in: &mut f64,
        peak_out: &mut f64,
        clipped: &mut u64,
    ) {
        let ceil = self.ceiling_linear;
        for frame in 0..num_frames {
            for ch in 0..num_channels {
                let idx = frame * num_channels + ch;
                let s = buffer[idx];
                *peak_in = peak_in.max(s.abs());
                if s > ceil {
                    buffer[idx] = ceil;
                    *clipped += 1;
                } else if s < -ceil {
                    buffer[idx] = -ceil;
                    *clipped += 1;
                }
                *peak_out = peak_out.max(buffer[idx].abs());
            }
        }
    }

    /// Soft-knee processing with envelope following.
    fn process_soft_knee(
        &mut self,
        buffer: &mut [f64],
        num_channels: usize,
        num_frames: usize,
        peak_in: &mut f64,
        peak_out: &mut f64,
    ) {
        let ceil = self.ceiling_linear;
        let knee = db_to_linear(self.config.knee_db) - 1.0;

        for frame in 0..num_frames {
            // Find peak across channels for this frame
            let mut frame_peak = 0.0_f64;
            for ch in 0..num_channels {
                let idx = frame * num_channels + ch;
                frame_peak = frame_peak.max(buffer[idx].abs());
            }
            *peak_in = peak_in.max(frame_peak);

            // Envelope following
            let coeff = if frame_peak > self.envelope {
                self.attack_coeff
            } else {
                self.release_coeff
            };
            self.envelope = coeff * self.envelope + (1.0 - coeff) * frame_peak;

            // Compute gain reduction with soft knee
            let gain = if self.envelope <= ceil {
                1.0
            } else if knee > 0.0 && self.envelope < ceil * (1.0 + knee) {
                let x = (self.envelope - ceil) / (ceil * knee);
                1.0 - x * (1.0 - ceil / self.envelope)
            } else {
                ceil / self.envelope
            };

            self.gain_reduction = gain;

            for ch in 0..num_channels {
                let idx = frame * num_channels + ch;
                buffer[idx] *= gain;
                *peak_out = peak_out.max(buffer[idx].abs());
            }
        }
    }

    /// Look-ahead processing.
    fn process_lookahead(
        &mut self,
        buffer: &mut [f64],
        num_channels: usize,
        num_frames: usize,
        peak_in: &mut f64,
        peak_out: &mut f64,
    ) {
        let ceil = self.ceiling_linear;

        for frame in 0..num_frames {
            // Find peak across channels
            let mut frame_peak = 0.0_f64;
            for ch in 0..num_channels {
                let idx = frame * num_channels + ch;
                frame_peak = frame_peak.max(buffer[idx].abs());
            }
            *peak_in = peak_in.max(frame_peak);

            // Envelope
            let coeff = if frame_peak > self.envelope {
                self.attack_coeff
            } else {
                self.release_coeff
            };
            self.envelope = coeff * self.envelope + (1.0 - coeff) * frame_peak;

            // Gain
            let gain = if self.envelope <= ceil {
                1.0
            } else {
                ceil / self.envelope
            };
            self.gain_reduction = gain;

            // Write current samples into delay buffer and read delayed samples
            if self.delay_frames > 0 && !self.delay_buffer.is_empty() {
                let read_pos = self.delay_write;
                for ch in 0..num_channels {
                    let buf_idx = frame * num_channels + ch;
                    let delay_idx = read_pos * num_channels + ch;
                    if delay_idx < self.delay_buffer.len() {
                        let delayed = self.delay_buffer[delay_idx];
                        self.delay_buffer[delay_idx] = buffer[buf_idx];
                        buffer[buf_idx] = delayed * gain;
                    }
                }
                self.delay_write = (self.delay_write + 1) % self.delay_frames;
            } else {
                for ch in 0..num_channels {
                    let idx = frame * num_channels + ch;
                    buffer[idx] *= gain;
                }
            }

            for ch in 0..num_channels {
                let idx = frame * num_channels + ch;
                *peak_out = peak_out.max(buffer[idx].abs());
            }
        }
    }
}

/// Simple brick-wall clip function for a single sample.
#[must_use]
pub fn brick_wall_clip(sample: f64, ceiling: f64) -> f64 {
    sample.clamp(-ceiling, ceiling)
}

/// Compute the gain reduction needed to bring a level below the ceiling.
///
/// Returns the linear gain multiplier (1.0 = no reduction).
#[must_use]
pub fn compute_gain_reduction(level_linear: f64, ceiling_linear: f64) -> f64 {
    if level_linear <= ceiling_linear || level_linear <= 0.0 {
        1.0
    } else {
        ceiling_linear / level_linear
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_db_to_linear_zero() {
        let l = db_to_linear(0.0);
        assert!((l - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_db_to_linear_minus6() {
        let l = db_to_linear(-6.0206);
        assert!((l - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_linear_to_db_one() {
        let db = linear_to_db(1.0);
        assert!(db.abs() < 1e-10);
    }

    #[test]
    fn test_linear_to_db_zero() {
        let db = linear_to_db(0.0);
        assert!(db < -100.0);
    }

    #[test]
    fn test_brick_wall_clip() {
        assert!((brick_wall_clip(0.5, 1.0) - 0.5).abs() < f64::EPSILON);
        assert!((brick_wall_clip(1.5, 1.0) - 1.0).abs() < f64::EPSILON);
        assert!((brick_wall_clip(-1.5, 1.0) - (-1.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_compute_gain_reduction_no_reduction() {
        let g = compute_gain_reduction(0.5, 1.0);
        assert!((g - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_compute_gain_reduction_needed() {
        let g = compute_gain_reduction(2.0, 1.0);
        assert!((g - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_limiter_config_default() {
        let cfg = LimiterConfig::default();
        assert_eq!(cfg.mode, LimiterMode::LookAhead);
        assert!((cfg.ceiling_db - (-0.3)).abs() < f64::EPSILON);
        assert_eq!(cfg.sample_rate, 48000);
    }

    #[test]
    fn test_limiter_hard_clip_mode() {
        let config = LimiterConfig {
            mode: LimiterMode::HardClip,
            ceiling_db: 0.0, // 1.0 linear
            ..Default::default()
        };
        let mut limiter = Limiter::new(config, 1);
        let mut buf = vec![0.5, 1.5, -2.0, 0.0];
        limiter.process_buffer(&mut buf, 1);
        assert!((buf[0] - 0.5).abs() < f64::EPSILON);
        assert!((buf[1] - 1.0).abs() < f64::EPSILON);
        assert!((buf[2] - (-1.0)).abs() < f64::EPSILON);
        assert!((buf[3]).abs() < f64::EPSILON);
    }

    #[test]
    fn test_limiter_soft_knee_reduces_peaks() {
        let config = LimiterConfig {
            mode: LimiterMode::SoftKnee,
            ceiling_db: -6.0,
            attack_ms: 0.0,
            release_ms: 100.0,
            ..Default::default()
        };
        let mut limiter = Limiter::new(config, 1);
        let mut buf = vec![1.0; 512];
        limiter.process_buffer(&mut buf, 1);
        // All outputs should be <= ceiling (0.5 linear for -6 dB)
        let ceil = db_to_linear(-6.0);
        for &s in &buf[1..] {
            assert!(s.abs() <= ceil + 0.05, "sample {s} exceeds ceiling {ceil}");
        }
    }

    #[test]
    fn test_limiter_reset() {
        let mut limiter = Limiter::with_defaults(2);
        limiter.envelope = 0.9;
        limiter.reset();
        assert!((limiter.envelope).abs() < f64::EPSILON);
        assert!((limiter.gain_reduction - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_limiter_set_ceiling() {
        let mut limiter = Limiter::with_defaults(1);
        limiter.set_ceiling_db(-1.0);
        assert!((limiter.config().ceiling_db - (-1.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_limiter_metrics_initial() {
        let limiter = Limiter::with_defaults(2);
        let m = limiter.metrics();
        assert!((m.gain_reduction_db).abs() < f64::EPSILON);
        assert_eq!(m.clipped_samples, 0);
    }

    #[test]
    fn test_limiter_empty_buffer() {
        let mut limiter = Limiter::with_defaults(2);
        let mut buf: Vec<f64> = vec![];
        limiter.process_buffer(&mut buf, 2);
        // Should not panic
    }

    #[test]
    fn test_time_constant_zero() {
        let c = time_constant(0.0, 48000);
        assert!((c).abs() < f64::EPSILON);
    }
}
