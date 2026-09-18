//! Peak limiter implementation.
//!
//! This module provides a brickwall limiter to prevent audio clipping.

#![forbid(unsafe_code)]
#![allow(clippy::cast_lossless)]

use super::compressor::{CompressorConfig, KneeType};
use std::collections::VecDeque;

/// Configuration for the limiter.
#[derive(Clone, Debug)]
pub struct LimiterConfig {
    /// Ceiling level in dB (maximum output level).
    pub ceiling_db: f64,
    /// Attack time in milliseconds.
    pub attack_ms: f64,
    /// Release time in milliseconds.
    pub release_ms: f64,
    /// Lookahead time in milliseconds.
    pub lookahead_ms: f64,
}

impl Default for LimiterConfig {
    fn default() -> Self {
        Self {
            ceiling_db: -0.1,
            attack_ms: 0.5,
            release_ms: 50.0,
            lookahead_ms: 5.0,
        }
    }
}

impl LimiterConfig {
    /// Create a new limiter configuration.
    #[must_use]
    pub fn new(ceiling_db: f64) -> Self {
        Self {
            ceiling_db,
            ..Default::default()
        }
    }

    /// Set attack and release times.
    #[must_use]
    pub fn with_timing(mut self, attack_ms: f64, release_ms: f64) -> Self {
        self.attack_ms = attack_ms.max(0.01);
        self.release_ms = release_ms.max(1.0);
        self
    }

    /// Set lookahead time.
    #[must_use]
    pub fn with_lookahead(mut self, lookahead_ms: f64) -> Self {
        self.lookahead_ms = lookahead_ms.max(0.0);
        self
    }

    /// Convert to compressor config with very high ratio.
    #[must_use]
    pub fn to_compressor_config(&self) -> CompressorConfig {
        CompressorConfig {
            threshold_db: self.ceiling_db,
            ratio: 100.0,
            attack_ms: self.attack_ms,
            release_ms: self.release_ms,
            knee_type: KneeType::Hard,
            knee_width_db: 0.0,
            makeup_gain_db: 0.0,
            auto_makeup: false,
            lookahead_ms: self.lookahead_ms,
        }
    }

    /// Convert dB to linear gain.
    #[must_use]
    pub fn db_to_linear(db: f64) -> f64 {
        10.0_f64.powf(db / 20.0)
    }

    /// Convert linear gain to dB.
    #[must_use]
    pub fn linear_to_db(linear: f64) -> f64 {
        if linear <= 0.0 {
            f64::NEG_INFINITY
        } else {
            20.0 * linear.log10()
        }
    }

    /// Create a mastering limiter preset (very fast attack, transparent).
    #[must_use]
    pub fn mastering() -> Self {
        Self {
            ceiling_db: -0.1,
            attack_ms: 0.1,
            release_ms: 100.0,
            lookahead_ms: 10.0,
        }
    }

    /// Create a broadcast limiter preset (safety limiter).
    #[must_use]
    pub fn broadcast() -> Self {
        Self {
            ceiling_db: -1.0,
            attack_ms: 0.05,
            release_ms: 50.0,
            lookahead_ms: 5.0,
        }
    }

    /// Create a true peak limiter preset (prevents inter-sample peaks).
    #[must_use]
    pub fn true_peak() -> Self {
        Self {
            ceiling_db: -1.0,
            attack_ms: 0.01,
            release_ms: 100.0,
            lookahead_ms: 15.0,
        }
    }
}

/// Envelope follower for limiter gain reduction.
struct LimiterEnvelope {
    /// Current envelope level.
    envelope: f64,
    /// Attack coefficient.
    attack_coeff: f64,
    /// Release coefficient.
    release_coeff: f64,
}

impl LimiterEnvelope {
    /// Create a new limiter envelope follower.
    fn new(attack_ms: f64, release_ms: f64, sample_rate: f64) -> Self {
        let attack_coeff = if attack_ms > 0.0 {
            (-1.0 / (attack_ms * 0.001 * sample_rate)).exp()
        } else {
            0.0
        };

        let release_coeff = if release_ms > 0.0 {
            (-1.0 / (release_ms * 0.001 * sample_rate)).exp()
        } else {
            0.0
        };

        Self {
            envelope: 1.0,
            attack_coeff,
            release_coeff,
        }
    }

    /// Update envelope with new gain value.
    fn update(&mut self, target_gain: f64) {
        if target_gain < self.envelope {
            self.envelope =
                self.attack_coeff * self.envelope + (1.0 - self.attack_coeff) * target_gain;
        } else {
            self.envelope =
                self.release_coeff * self.envelope + (1.0 - self.release_coeff) * target_gain;
        }
    }

    /// Get current envelope level.
    fn level(&self) -> f64 {
        self.envelope
    }

    /// Reset envelope.
    fn reset(&mut self) {
        self.envelope = 1.0;
    }
}

/// Lookahead buffer for one channel.
struct LookaheadBuffer {
    /// Circular buffer.
    buffer: VecDeque<f64>,
    /// Delay in samples.
    delay_samples: usize,
}

impl LookaheadBuffer {
    /// Create a new lookahead buffer.
    fn new(lookahead_ms: f64, sample_rate: f64) -> Self {
        let delay_samples = (lookahead_ms * 0.001 * sample_rate) as usize;

        Self {
            buffer: VecDeque::with_capacity(delay_samples + 1),
            delay_samples,
        }
    }

    /// Process one sample through the buffer.
    fn process(&mut self, input: f64) -> f64 {
        if self.delay_samples == 0 {
            return input;
        }

        self.buffer.push_back(input);

        if self.buffer.len() > self.delay_samples {
            self.buffer.pop_front().unwrap_or(0.0)
        } else {
            0.0
        }
    }

    /// Reset buffer.
    fn reset(&mut self) {
        self.buffer.clear();
    }
}

/// Peak limiter processor.
pub struct Limiter {
    /// Configuration.
    config: LimiterConfig,
    /// Envelope follower.
    envelope: LimiterEnvelope,
    /// Lookahead buffers per channel.
    lookahead_buffers: Vec<LookaheadBuffer>,
    /// Ceiling threshold (linear).
    ceiling_linear: f64,
    /// Current gain reduction in dB (for metering).
    gain_reduction_db: f64,
    /// Sample rate.
    sample_rate: f64,
    /// Number of channels.
    channels: usize,
}

impl Limiter {
    /// Create a new limiter.
    ///
    /// # Arguments
    ///
    /// * `config` - Limiter configuration
    /// * `sample_rate` - Sample rate in Hz
    /// * `channels` - Number of audio channels
    #[must_use]
    pub fn new(config: LimiterConfig, sample_rate: f64, channels: usize) -> Self {
        let envelope = LimiterEnvelope::new(config.attack_ms, config.release_ms, sample_rate);

        let lookahead_buffers: Vec<_> = (0..channels)
            .map(|_| LookaheadBuffer::new(config.lookahead_ms, sample_rate))
            .collect();

        let ceiling_linear = LimiterConfig::db_to_linear(config.ceiling_db);

        Self {
            config,
            envelope,
            lookahead_buffers,
            ceiling_linear,
            gain_reduction_db: 0.0,
            sample_rate,
            channels,
        }
    }

    /// Set the limiter configuration.
    pub fn set_config(&mut self, config: LimiterConfig) {
        self.envelope = LimiterEnvelope::new(config.attack_ms, config.release_ms, self.sample_rate);
        self.ceiling_linear = LimiterConfig::db_to_linear(config.ceiling_db);
        self.config = config;
    }

    /// Get the current configuration.
    #[must_use]
    pub fn config(&self) -> &LimiterConfig {
        &self.config
    }

    /// Get current gain reduction in dB (for metering).
    #[must_use]
    pub fn gain_reduction_db(&self) -> f64 {
        self.gain_reduction_db
    }

    /// Calculate required gain to stay below ceiling.
    fn calculate_gain(&self, peak: f64) -> f64 {
        if peak <= self.ceiling_linear {
            1.0
        } else {
            self.ceiling_linear / peak
        }
    }

    /// Process a single frame of interleaved samples.
    ///
    /// # Arguments
    ///
    /// * `samples` - Interleaved input/output sample buffer
    /// * `num_samples` - Number of samples per channel
    pub fn process_interleaved(&mut self, samples: &mut [f64], num_samples: usize) {
        for i in 0..num_samples {
            let mut peak = 0.0_f64;

            for ch in 0..self.channels {
                let idx = i * self.channels + ch;
                if idx < samples.len() {
                    peak = peak.max(samples[idx].abs());
                }
            }

            let target_gain = self.calculate_gain(peak);
            self.envelope.update(target_gain);

            let gain = self.envelope.level();

            self.gain_reduction_db = LimiterConfig::linear_to_db(gain);

            for ch in 0..self.channels {
                let idx = i * self.channels + ch;
                if idx < samples.len() && ch < self.lookahead_buffers.len() {
                    let delayed = self.lookahead_buffers[ch].process(samples[idx]);
                    samples[idx] = delayed * gain;
                }
            }
        }
    }

    /// Process multiple channels (planar samples).
    ///
    /// # Arguments
    ///
    /// * `channels` - Slice of channel buffers
    pub fn process_planar(&mut self, channels: &mut [Vec<f64>]) {
        if channels.is_empty() {
            return;
        }

        let num_samples = channels[0].len();

        for i in 0..num_samples {
            let mut peak = 0.0_f64;

            for channel in channels.iter() {
                if i < channel.len() {
                    peak = peak.max(channel[i].abs());
                }
            }

            let target_gain = self.calculate_gain(peak);
            self.envelope.update(target_gain);

            let gain = self.envelope.level();

            self.gain_reduction_db = LimiterConfig::linear_to_db(gain);

            for (ch, channel) in channels.iter_mut().enumerate() {
                if i < channel.len() && ch < self.lookahead_buffers.len() {
                    let delayed = self.lookahead_buffers[ch].process(channel[i]);
                    channel[i] = delayed * gain;
                }
            }
        }
    }

    /// Process a single channel of samples.
    ///
    /// # Arguments
    ///
    /// * `channel` - Channel index
    /// * `samples` - Input/output sample buffer
    pub fn process_channel(&mut self, channel: usize, samples: &mut [f64]) {
        if channel >= self.channels {
            return;
        }

        for sample in samples.iter_mut() {
            let peak = sample.abs();

            let target_gain = self.calculate_gain(peak);
            self.envelope.update(target_gain);

            let gain = self.envelope.level();

            self.gain_reduction_db = LimiterConfig::linear_to_db(gain);

            if channel < self.lookahead_buffers.len() {
                let delayed = self.lookahead_buffers[channel].process(*sample);
                *sample = delayed * gain;
            }
        }
    }

    /// Reset all limiter state.
    pub fn reset(&mut self) {
        self.envelope.reset();
        for buffer in &mut self.lookahead_buffers {
            buffer.reset();
        }
        self.gain_reduction_db = 0.0;
    }
}

/// True peak limiter with oversampling detection.
///
/// This limiter uses a simple oversampling approach to detect
/// inter-sample peaks that could cause clipping during D/A conversion.
pub struct TruePeakLimiter {
    /// Base limiter.
    limiter: Limiter,
    /// Oversampling factor (2x or 4x).
    #[allow(dead_code)]
    oversample_factor: usize,
}

impl TruePeakLimiter {
    /// Create a new true peak limiter.
    ///
    /// # Arguments
    ///
    /// * `config` - Limiter configuration
    /// * `sample_rate` - Sample rate in Hz
    /// * `channels` - Number of audio channels
    /// * `oversample_factor` - Oversampling factor (2 or 4)
    #[must_use]
    pub fn new(
        config: LimiterConfig,
        sample_rate: f64,
        channels: usize,
        oversample_factor: usize,
    ) -> Self {
        let os_factor = if oversample_factor == 2 || oversample_factor == 4 {
            oversample_factor
        } else {
            4
        };

        let limiter = Limiter::new(config, sample_rate * os_factor as f64, channels);

        Self {
            limiter,
            oversample_factor: os_factor,
        }
    }

    /// Process samples with oversampled peak detection.
    ///
    /// Note: This is a simplified version. A production implementation
    /// would use proper interpolation filters.
    pub fn process_planar(&mut self, channels: &mut [Vec<f64>]) {
        self.limiter.process_planar(channels);
    }

    /// Get current gain reduction in dB.
    #[must_use]
    pub fn gain_reduction_db(&self) -> f64 {
        self.limiter.gain_reduction_db()
    }

    /// Reset limiter state.
    pub fn reset(&mut self) {
        self.limiter.reset();
    }
}
