//! Delay line and delay effect implementations.
//!
//! This module provides delay effects with feedback, ping-pong mode,
//! and various modulation capabilities.

#![forbid(unsafe_code)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use std::collections::VecDeque;

/// Delay mode configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DelayMode {
    /// Normal delay - same delay for all channels.
    #[default]
    Normal,
    /// Ping-pong stereo delay - alternates between left and right.
    PingPong,
    /// Slapback delay - very short delay for doubling effect.
    Slapback,
}

/// Configuration for the delay effect.
#[derive(Clone, Debug)]
pub struct DelayConfig {
    /// Delay time in milliseconds.
    pub delay_ms: f64,
    /// Feedback amount (0.0 to 1.0).
    pub feedback: f64,
    /// Dry/wet mix (0.0 = dry only, 1.0 = wet only).
    pub mix: f64,
    /// Delay mode.
    pub mode: DelayMode,
    /// High-frequency damping factor (0.0 = none, 1.0 = full).
    pub damping: f64,
}

impl Default for DelayConfig {
    fn default() -> Self {
        Self {
            delay_ms: 250.0,
            feedback: 0.5,
            mix: 0.5,
            mode: DelayMode::Normal,
            damping: 0.0,
        }
    }
}

impl DelayConfig {
    /// Create a new delay configuration.
    #[must_use]
    pub fn new(delay_ms: f64) -> Self {
        Self {
            delay_ms,
            ..Default::default()
        }
    }

    /// Set feedback amount.
    #[must_use]
    pub fn with_feedback(mut self, feedback: f64) -> Self {
        self.feedback = feedback.clamp(0.0, 0.99);
        self
    }

    /// Set dry/wet mix.
    #[must_use]
    pub fn with_mix(mut self, mix: f64) -> Self {
        self.mix = mix.clamp(0.0, 1.0);
        self
    }

    /// Set delay mode.
    #[must_use]
    pub fn with_mode(mut self, mode: DelayMode) -> Self {
        self.mode = mode;
        self
    }

    /// Enable ping-pong mode.
    #[must_use]
    pub fn ping_pong(mut self) -> Self {
        self.mode = DelayMode::PingPong;
        self
    }

    /// Set damping factor.
    #[must_use]
    pub fn with_damping(mut self, damping: f64) -> Self {
        self.damping = damping.clamp(0.0, 1.0);
        self
    }

    /// Create a slapback delay preset (short delay, low feedback).
    #[must_use]
    pub fn slapback() -> Self {
        Self {
            delay_ms: 75.0,
            feedback: 0.2,
            mix: 0.3,
            mode: DelayMode::Slapback,
            damping: 0.0,
        }
    }

    /// Create an echo preset (longer delay, moderate feedback).
    #[must_use]
    pub fn echo() -> Self {
        Self {
            delay_ms: 375.0,
            feedback: 0.5,
            mix: 0.4,
            mode: DelayMode::Normal,
            damping: 0.3,
        }
    }

    /// Create a ping-pong preset.
    #[must_use]
    pub fn ping_pong_preset() -> Self {
        Self {
            delay_ms: 250.0,
            feedback: 0.6,
            mix: 0.5,
            mode: DelayMode::PingPong,
            damping: 0.2,
        }
    }
}

/// A delay line for a single channel.
#[derive(Clone, Debug)]
pub struct DelayLine {
    /// Circular buffer.
    buffer: VecDeque<f64>,
    /// Delay in samples.
    delay_samples: usize,
    /// Low-pass filter state for damping.
    lp_state: f64,
    /// Damping coefficient.
    damping: f64,
}

impl DelayLine {
    /// Create a new delay line.
    ///
    /// # Arguments
    ///
    /// * `delay_ms` - Delay time in milliseconds
    /// * `sample_rate` - Sample rate in Hz
    /// * `damping` - Damping coefficient (0.0 to 1.0)
    #[must_use]
    pub fn new(delay_ms: f64, sample_rate: f64, damping: f64) -> Self {
        let delay_samples = ((delay_ms * 0.001 * sample_rate) as usize).max(1);
        let mut buffer = VecDeque::with_capacity(delay_samples + 1);

        for _ in 0..delay_samples {
            buffer.push_back(0.0);
        }

        Self {
            buffer,
            delay_samples,
            lp_state: 0.0,
            damping,
        }
    }

    /// Process one sample through the delay line with feedback.
    ///
    /// # Arguments
    ///
    /// * `input` - Input sample
    /// * `feedback` - Feedback amount (0.0 to 1.0)
    ///
    /// # Returns
    ///
    /// Delayed output sample
    pub fn process(&mut self, input: f64, feedback: f64) -> f64 {
        let output = self.buffer.pop_front().unwrap_or(0.0);

        let damped = if self.damping > 0.0 {
            self.lp_state = self.lp_state + self.damping * (output - self.lp_state);
            output - self.damping * (output - self.lp_state)
        } else {
            output
        };

        self.buffer.push_back(input + damped * feedback);

        damped
    }

    /// Process one sample without feedback (simple delay).
    pub fn process_simple(&mut self, input: f64) -> f64 {
        let output = self.buffer.pop_front().unwrap_or(0.0);
        self.buffer.push_back(input);
        output
    }

    /// Tap the delay line at a specific position without removing the sample.
    ///
    /// # Arguments
    ///
    /// * `tap_samples` - Number of samples to look back
    #[must_use]
    pub fn tap(&self, tap_samples: usize) -> f64 {
        if tap_samples >= self.buffer.len() {
            return 0.0;
        }
        self.buffer[self.buffer.len() - 1 - tap_samples]
    }

    /// Reset the delay line.
    pub fn reset(&mut self) {
        self.buffer.clear();
        for _ in 0..self.delay_samples {
            self.buffer.push_back(0.0);
        }
        self.lp_state = 0.0;
    }

    /// Get current delay in samples.
    #[must_use]
    pub fn delay_samples(&self) -> usize {
        self.delay_samples
    }

    /// Set damping coefficient.
    pub fn set_damping(&mut self, damping: f64) {
        self.damping = damping.clamp(0.0, 1.0);
    }
}

/// Multi-tap delay line that can read from multiple positions.
#[derive(Clone, Debug)]
pub struct MultiTapDelay {
    /// Main delay line.
    delay_line: DelayLine,
    /// Tap positions in samples.
    tap_positions: Vec<usize>,
    /// Tap gains.
    tap_gains: Vec<f64>,
}

impl MultiTapDelay {
    /// Create a new multi-tap delay.
    ///
    /// # Arguments
    ///
    /// * `max_delay_ms` - Maximum delay time in milliseconds
    /// * `sample_rate` - Sample rate in Hz
    #[must_use]
    pub fn new(max_delay_ms: f64, sample_rate: f64) -> Self {
        Self {
            delay_line: DelayLine::new(max_delay_ms, sample_rate, 0.0),
            tap_positions: Vec::new(),
            tap_gains: Vec::new(),
        }
    }

    /// Add a tap at a specific delay time.
    ///
    /// # Arguments
    ///
    /// * `delay_ms` - Delay time for this tap in milliseconds
    /// * `gain` - Gain for this tap
    /// * `sample_rate` - Sample rate in Hz
    pub fn add_tap(&mut self, delay_ms: f64, gain: f64, sample_rate: f64) {
        let tap_samples = (delay_ms * 0.001 * sample_rate) as usize;
        if tap_samples < self.delay_line.delay_samples() {
            self.tap_positions.push(tap_samples);
            self.tap_gains.push(gain);
        }
    }

    /// Process one sample through all taps.
    pub fn process(&mut self, input: f64) -> f64 {
        let mut output = 0.0;

        for (i, &tap_pos) in self.tap_positions.iter().enumerate() {
            let tap_value = self.delay_line.tap(tap_pos);
            output += tap_value * self.tap_gains[i];
        }

        self.delay_line.process_simple(input);

        output
    }

    /// Reset all delay lines.
    pub fn reset(&mut self) {
        self.delay_line.reset();
    }

    /// Clear all taps.
    pub fn clear_taps(&mut self) {
        self.tap_positions.clear();
        self.tap_gains.clear();
    }
}

/// Stereo delay processor.
pub struct StereoDelay {
    /// Configuration.
    config: DelayConfig,
    /// Left channel delay line.
    left_delay: DelayLine,
    /// Right channel delay line.
    right_delay: DelayLine,
    /// Ping-pong state.
    ping_pong_state: bool,
    /// Sample rate.
    sample_rate: f64,
}

impl StereoDelay {
    /// Create a new stereo delay.
    ///
    /// # Arguments
    ///
    /// * `config` - Delay configuration
    /// * `sample_rate` - Sample rate in Hz
    #[must_use]
    pub fn new(config: DelayConfig, sample_rate: f64) -> Self {
        Self {
            left_delay: DelayLine::new(config.delay_ms, sample_rate, config.damping),
            right_delay: DelayLine::new(config.delay_ms, sample_rate, config.damping),
            config,
            ping_pong_state: false,
            sample_rate,
        }
    }

    /// Set the delay configuration.
    pub fn set_config(&mut self, config: DelayConfig) {
        self.left_delay = DelayLine::new(config.delay_ms, self.sample_rate, config.damping);
        self.right_delay = DelayLine::new(config.delay_ms, self.sample_rate, config.damping);
        self.config = config;
    }

    /// Get the current configuration.
    #[must_use]
    pub fn config(&self) -> &DelayConfig {
        &self.config
    }

    /// Process stereo samples (interleaved).
    ///
    /// # Arguments
    ///
    /// * `samples` - Interleaved stereo input/output buffer
    /// * `num_samples` - Number of sample frames
    pub fn process_interleaved(&mut self, samples: &mut [f64], num_samples: usize) {
        for i in 0..num_samples {
            let left_idx = i * 2;
            let right_idx = i * 2 + 1;

            if left_idx >= samples.len() || right_idx >= samples.len() {
                break;
            }

            let dry_left = samples[left_idx];
            let dry_right = samples[right_idx];

            let (wet_left, wet_right) = match self.config.mode {
                DelayMode::Normal => (
                    self.left_delay.process(dry_left, self.config.feedback),
                    self.right_delay.process(dry_right, self.config.feedback),
                ),
                DelayMode::PingPong => {
                    if self.ping_pong_state {
                        (
                            self.left_delay.process(dry_right, self.config.feedback),
                            self.right_delay.process(dry_left, self.config.feedback),
                        )
                    } else {
                        (
                            self.left_delay.process(dry_left, self.config.feedback),
                            self.right_delay.process(dry_right, self.config.feedback),
                        )
                    }
                }
                DelayMode::Slapback => (
                    self.left_delay.process(dry_left, self.config.feedback),
                    self.right_delay.process(dry_right, self.config.feedback),
                ),
            };

            if matches!(self.config.mode, DelayMode::PingPong) {
                self.ping_pong_state = !self.ping_pong_state;
            }

            samples[left_idx] = dry_left * (1.0 - self.config.mix) + wet_left * self.config.mix;
            samples[right_idx] = dry_right * (1.0 - self.config.mix) + wet_right * self.config.mix;
        }
    }

    /// Process stereo samples (planar).
    ///
    /// # Arguments
    ///
    /// * `left` - Left channel input/output buffer
    /// * `right` - Right channel input/output buffer
    pub fn process_planar(&mut self, left: &mut [f64], right: &mut [f64]) {
        let num_samples = left.len().min(right.len());

        for i in 0..num_samples {
            let dry_left = left[i];
            let dry_right = right[i];

            let (wet_left, wet_right) = match self.config.mode {
                DelayMode::Normal => (
                    self.left_delay.process(dry_left, self.config.feedback),
                    self.right_delay.process(dry_right, self.config.feedback),
                ),
                DelayMode::PingPong => {
                    if self.ping_pong_state {
                        (
                            self.left_delay.process(dry_right, self.config.feedback),
                            self.right_delay.process(dry_left, self.config.feedback),
                        )
                    } else {
                        (
                            self.left_delay.process(dry_left, self.config.feedback),
                            self.right_delay.process(dry_right, self.config.feedback),
                        )
                    }
                }
                DelayMode::Slapback => (
                    self.left_delay.process(dry_left, self.config.feedback),
                    self.right_delay.process(dry_right, self.config.feedback),
                ),
            };

            if matches!(self.config.mode, DelayMode::PingPong) {
                self.ping_pong_state = !self.ping_pong_state;
            }

            left[i] = dry_left * (1.0 - self.config.mix) + wet_left * self.config.mix;
            right[i] = dry_right * (1.0 - self.config.mix) + wet_right * self.config.mix;
        }
    }

    /// Reset all delay state.
    pub fn reset(&mut self) {
        self.left_delay.reset();
        self.right_delay.reset();
        self.ping_pong_state = false;
    }
}

/// Mono delay processor.
pub struct MonoDelay {
    /// Configuration.
    config: DelayConfig,
    /// Delay line.
    delay_line: DelayLine,
}

impl MonoDelay {
    /// Create a new mono delay.
    ///
    /// # Arguments
    ///
    /// * `config` - Delay configuration
    /// * `sample_rate` - Sample rate in Hz
    #[must_use]
    pub fn new(config: DelayConfig, sample_rate: f64) -> Self {
        Self {
            delay_line: DelayLine::new(config.delay_ms, sample_rate, config.damping),
            config,
        }
    }

    /// Set the delay configuration.
    pub fn set_config(&mut self, config: DelayConfig, sample_rate: f64) {
        self.delay_line = DelayLine::new(config.delay_ms, sample_rate, config.damping);
        self.config = config;
    }

    /// Get the current configuration.
    #[must_use]
    pub fn config(&self) -> &DelayConfig {
        &self.config
    }

    /// Process mono samples.
    ///
    /// # Arguments
    ///
    /// * `samples` - Mono input/output buffer
    pub fn process(&mut self, samples: &mut [f64]) {
        for sample in samples.iter_mut() {
            let dry = *sample;
            let wet = self.delay_line.process(dry, self.config.feedback);
            *sample = dry * (1.0 - self.config.mix) + wet * self.config.mix;
        }
    }

    /// Reset delay state.
    pub fn reset(&mut self) {
        self.delay_line.reset();
    }
}
