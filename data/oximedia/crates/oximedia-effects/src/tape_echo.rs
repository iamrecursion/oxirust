#![allow(dead_code)]
//! Tape echo simulation effect.
//!
//! This module emulates the sound of classic tape echo machines such as the
//! Roland Space Echo and Echoplex. It models tape saturation, wow and flutter
//! (pitch modulation), high-frequency roll-off from tape degradation, and
//! multiple playback head positions.

use std::f32::consts::PI;

/// Configuration for individual tape playback heads.
#[derive(Debug, Clone)]
pub struct TapeHead {
    /// Delay time in milliseconds for this head.
    pub delay_ms: f32,
    /// Feedback amount for this head (0.0 - 1.0).
    pub feedback: f32,
    /// Level of this head in the mix (0.0 - 1.0).
    pub level: f32,
    /// Pan position of this head (-1.0 left, 0.0 center, 1.0 right).
    pub pan: f32,
    /// Whether this head is enabled.
    pub enabled: bool,
}

impl Default for TapeHead {
    fn default() -> Self {
        Self {
            delay_ms: 300.0,
            feedback: 0.4,
            level: 1.0,
            pan: 0.0,
            enabled: true,
        }
    }
}

/// Configuration for the tape echo effect.
#[derive(Debug, Clone)]
pub struct TapeEchoConfig {
    /// Tape heads configuration (up to 4 heads).
    pub heads: Vec<TapeHead>,
    /// Tape saturation amount (0.0 = clean, 1.0 = heavy saturation).
    pub saturation: f32,
    /// Wow and flutter depth (0.0 = none, 1.0 = extreme).
    pub wow_flutter: f32,
    /// Wow and flutter rate in Hz.
    pub wow_rate_hz: f32,
    /// High-frequency damping (tape degradation, 0.0 = none, 1.0 = heavy).
    pub hf_damping: f32,
    /// Master wet/dry mix (0.0 = dry, 1.0 = wet only).
    pub mix: f32,
    /// Input gain (linear).
    pub input_gain: f32,
    /// Sample rate in Hz.
    pub sample_rate: f32,
}

impl Default for TapeEchoConfig {
    fn default() -> Self {
        Self {
            heads: vec![TapeHead::default()],
            saturation: 0.3,
            wow_flutter: 0.1,
            wow_rate_hz: 0.5,
            hf_damping: 0.3,
            mix: 0.5,
            input_gain: 1.0,
            sample_rate: 48000.0,
        }
    }
}

/// Internal delay buffer with fractional-sample interpolation.
#[derive(Debug)]
struct DelayBuffer {
    /// Circular buffer.
    buffer: Vec<f32>,
    /// Write position.
    write_pos: usize,
    /// Buffer length.
    length: usize,
}

impl DelayBuffer {
    /// Create a new delay buffer with the given maximum length in samples.
    fn new(max_samples: usize) -> Self {
        Self {
            buffer: vec![0.0; max_samples],
            write_pos: 0,
            length: max_samples,
        }
    }

    /// Write a sample into the buffer.
    fn write(&mut self, sample: f32) {
        self.buffer[self.write_pos] = sample;
        self.write_pos = (self.write_pos + 1) % self.length;
    }

    /// Read from the buffer at a fractional delay position using linear interpolation.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    fn read(&self, delay_samples: f32) -> f32 {
        let delay_int = delay_samples as usize;
        let frac = delay_samples - delay_int as f32;

        let idx0 = (self.write_pos + self.length - delay_int - 1) % self.length;
        let idx1 = (self.write_pos + self.length - delay_int - 2) % self.length;

        self.buffer[idx0] * (1.0 - frac) + self.buffer[idx1] * frac
    }

    /// Clear the buffer.
    fn clear(&mut self) {
        self.buffer.fill(0.0);
        self.write_pos = 0;
    }
}

/// Tape echo effect processor.
#[derive(Debug)]
pub struct TapeEcho {
    /// Configuration.
    config: TapeEchoConfig,
    /// Delay buffer.
    delay_buf: DelayBuffer,
    /// Low-pass filter state for HF damping.
    lp_state: f32,
    /// Wow/flutter LFO phase.
    wow_phase: f32,
    /// Wow/flutter phase increment per sample.
    wow_phase_inc: f32,
    /// Feedback accumulator.
    feedback_acc: f32,
}

impl TapeEcho {
    /// Create a new tape echo effect with the given configuration.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub fn new(config: TapeEchoConfig) -> Self {
        // Calculate max delay in samples (max head delay + wow modulation margin)
        let max_delay_ms = config
            .heads
            .iter()
            .map(|h| h.delay_ms)
            .fold(0.0f32, f32::max)
            + 50.0; // margin for wow
        let max_samples = ((max_delay_ms / 1000.0) * config.sample_rate) as usize + 1;

        let wow_phase_inc = config.wow_rate_hz / config.sample_rate;

        Self {
            delay_buf: DelayBuffer::new(max_samples.max(1)),
            lp_state: 0.0,
            wow_phase: 0.0,
            wow_phase_inc,
            feedback_acc: 0.0,
            config,
        }
    }

    /// Create a tape echo with default settings.
    #[must_use]
    pub fn default_effect() -> Self {
        Self::new(TapeEchoConfig::default())
    }

    /// Reset the effect state.
    pub fn reset(&mut self) {
        self.delay_buf.clear();
        self.lp_state = 0.0;
        self.wow_phase = 0.0;
        self.feedback_acc = 0.0;
    }

    /// Set the master mix level.
    pub fn set_mix(&mut self, mix: f32) {
        self.config.mix = mix.clamp(0.0, 1.0);
    }

    /// Set the saturation amount.
    pub fn set_saturation(&mut self, saturation: f32) {
        self.config.saturation = saturation.clamp(0.0, 1.0);
    }

    /// Set the wow and flutter depth.
    pub fn set_wow_flutter(&mut self, depth: f32) {
        self.config.wow_flutter = depth.clamp(0.0, 1.0);
    }

    /// Set the HF damping amount.
    pub fn set_hf_damping(&mut self, damping: f32) {
        self.config.hf_damping = damping.clamp(0.0, 1.0);
    }

    /// Apply tape saturation (soft clipping).
    fn saturate(sample: f32, amount: f32) -> f32 {
        if amount < 0.001 {
            return sample;
        }
        let drive = 1.0 + amount * 4.0;
        let driven = sample * drive;
        // Soft clipping via tanh approximation
        let x = driven.clamp(-3.0, 3.0);
        let x2 = x * x;
        let result = x * (27.0 + x2) / (27.0 + 9.0 * x2);
        result / drive.sqrt()
    }

    /// Apply one-pole low-pass filter for HF damping.
    fn apply_lp(&mut self, sample: f32) -> f32 {
        let coeff = self.config.hf_damping * 0.7;
        self.lp_state = self.lp_state + coeff * (sample - self.lp_state);
        self.lp_state
    }

    /// Get the current wow/flutter modulation in samples.
    #[allow(clippy::cast_precision_loss)]
    fn wow_modulation(&self) -> f32 {
        let max_mod_ms = self.config.wow_flutter * 5.0; // up to 5ms modulation
        let mod_samples = (max_mod_ms / 1000.0) * self.config.sample_rate;
        (2.0 * PI * self.wow_phase).sin() * mod_samples
    }

    /// Process a single mono sample.
    #[allow(clippy::cast_precision_loss)]
    pub fn process_sample(&mut self, input: f32) -> f32 {
        let input_gained = input * self.config.input_gain;

        // Write input + feedback into the delay buffer
        let write_val = Self::saturate(input_gained + self.feedback_acc, self.config.saturation);
        self.delay_buf.write(write_val);

        // Read from each head and accumulate
        let mut wet = 0.0f32;
        let mut feedback_sum = 0.0f32;
        let wow_mod = self.wow_modulation();

        for head in &self.config.heads {
            if !head.enabled {
                continue;
            }
            let delay_samples = (head.delay_ms / 1000.0) * self.config.sample_rate + wow_mod;
            let delay_samples = delay_samples.max(1.0);

            // Clamp to buffer bounds
            let max_delay = (self.delay_buf.length as f32) - 2.0;
            let clamped_delay = delay_samples.min(max_delay);

            let tap = self.delay_buf.read(clamped_delay);
            wet += tap * head.level;
            feedback_sum += tap * head.feedback;
        }

        // Apply HF damping to feedback path
        if self.config.hf_damping > 0.001 {
            feedback_sum = self.apply_lp(feedback_sum);
        }

        // Prevent feedback runaway
        self.feedback_acc = feedback_sum.clamp(-2.0, 2.0);

        // Advance wow/flutter LFO
        self.wow_phase += self.wow_phase_inc;
        if self.wow_phase >= 1.0 {
            self.wow_phase -= 1.0;
        }

        // Mix dry and wet
        input * (1.0 - self.config.mix) + wet * self.config.mix
    }

    /// Process a mono buffer in-place.
    pub fn process(&mut self, buffer: &mut [f32]) {
        for sample in buffer.iter_mut() {
            *sample = self.process_sample(*sample);
        }
    }

    /// Process stereo buffers in-place (mono echo applied to both channels).
    pub fn process_stereo(&mut self, left: &mut [f32], right: &mut [f32]) {
        let len = left.len().min(right.len());
        for i in 0..len {
            let mono = (left[i] + right[i]) * 0.5;
            let processed = self.process_sample(mono);
            // Apply per-head panning
            left[i] = left[i] * (1.0 - self.config.mix) + processed * self.config.mix;
            right[i] = right[i] * (1.0 - self.config.mix) + processed * self.config.mix;
        }
    }

    /// Get the current number of enabled heads.
    #[must_use]
    pub fn active_head_count(&self) -> usize {
        self.config.heads.iter().filter(|h| h.enabled).count()
    }

    /// Get the maximum delay time across all heads in milliseconds.
    pub fn max_delay_ms(&self) -> f32 {
        self.config
            .heads
            .iter()
            .filter(|h| h.enabled)
            .map(|h| h.delay_ms)
            .fold(0.0f32, f32::max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_tape_head() {
        let head = TapeHead::default();
        assert!((head.delay_ms - 300.0).abs() < 1e-6);
        assert!((head.feedback - 0.4).abs() < 1e-6);
        assert!(head.enabled);
    }

    #[test]
    fn test_default_config() {
        let cfg = TapeEchoConfig::default();
        assert_eq!(cfg.heads.len(), 1);
        assert!((cfg.saturation - 0.3).abs() < 1e-6);
        assert!((cfg.mix - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_tape_echo_creation() {
        let echo = TapeEcho::default_effect();
        assert!(echo.delay_buf.length > 0);
        assert!((echo.wow_phase - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_silence_passthrough() {
        let mut echo = TapeEcho::default_effect();
        let mut buffer = vec![0.0f32; 256];
        echo.process(&mut buffer);
        for &s in &buffer {
            assert!(s.abs() < 1e-6, "Echo of silence should be near-silence");
        }
    }

    #[test]
    fn test_dry_mix() {
        let mut echo = TapeEcho::new(TapeEchoConfig {
            mix: 0.0,
            ..Default::default()
        });
        let input = 0.5f32;
        let output = echo.process_sample(input);
        assert!((output - input).abs() < 1e-4);
    }

    #[test]
    fn test_saturate_clean() {
        let result = TapeEcho::saturate(0.5, 0.0);
        assert!((result - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_saturate_soft_clip() {
        let result = TapeEcho::saturate(1.0, 1.0);
        // Saturated output should be lower than drive * input
        assert!(result < 5.0);
        assert!(result > 0.0);
    }

    #[test]
    fn test_delay_buffer_write_read() {
        let mut buf = DelayBuffer::new(100);
        buf.write(1.0);
        let val = buf.read(0.0);
        assert!((val - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_delay_buffer_clear() {
        let mut buf = DelayBuffer::new(100);
        buf.write(1.0);
        buf.clear();
        let val = buf.read(0.0);
        assert!(val.abs() < 1e-9);
    }

    #[test]
    fn test_set_mix() {
        let mut echo = TapeEcho::default_effect();
        echo.set_mix(0.8);
        assert!((echo.config.mix - 0.8).abs() < 1e-6);
        echo.set_mix(2.0);
        assert!((echo.config.mix - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_set_saturation() {
        let mut echo = TapeEcho::default_effect();
        echo.set_saturation(0.7);
        assert!((echo.config.saturation - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_active_head_count() {
        let echo = TapeEcho::new(TapeEchoConfig {
            heads: vec![
                TapeHead {
                    enabled: true,
                    ..Default::default()
                },
                TapeHead {
                    enabled: false,
                    ..Default::default()
                },
                TapeHead {
                    enabled: true,
                    ..Default::default()
                },
            ],
            ..Default::default()
        });
        assert_eq!(echo.active_head_count(), 2);
    }

    #[test]
    fn test_max_delay_ms() {
        let echo = TapeEcho::new(TapeEchoConfig {
            heads: vec![
                TapeHead {
                    delay_ms: 200.0,
                    ..Default::default()
                },
                TapeHead {
                    delay_ms: 500.0,
                    ..Default::default()
                },
                TapeHead {
                    delay_ms: 100.0,
                    enabled: false,
                    ..Default::default()
                },
            ],
            ..Default::default()
        });
        assert!((echo.max_delay_ms() - 500.0).abs() < 1e-6);
    }

    #[test]
    fn test_process_stereo() {
        let mut echo = TapeEcho::default_effect();
        let mut left = vec![0.5f32; 64];
        let mut right = vec![0.5f32; 64];
        echo.process_stereo(&mut left, &mut right);
        // Should not crash, output should be finite
        for &s in left.iter().chain(right.iter()) {
            assert!(s.is_finite());
        }
    }

    #[test]
    fn test_reset() {
        let mut echo = TapeEcho::default_effect();
        let mut buffer = vec![1.0f32; 128];
        echo.process(&mut buffer);
        echo.reset();
        assert!((echo.feedback_acc - 0.0).abs() < 1e-9);
        assert!((echo.lp_state - 0.0).abs() < 1e-9);
    }
}
