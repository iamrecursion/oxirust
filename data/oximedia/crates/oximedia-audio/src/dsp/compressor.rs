//! Dynamics compressor DSP implementation.
//!
//! This module provides a full-featured dynamics compressor with
//! threshold, ratio, attack, release, knee, and makeup gain.

#![forbid(unsafe_code)]
#![allow(clippy::cast_lossless)]

use std::collections::VecDeque;

/// Knee type for compression curve.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum KneeType {
    /// Hard knee - abrupt transition at threshold.
    #[default]
    Hard,
    /// Soft knee - gradual transition around threshold.
    Soft,
}

/// Configuration for the dynamics compressor.
#[derive(Clone, Debug)]
pub struct CompressorConfig {
    /// Threshold in dB.
    pub threshold_db: f64,
    /// Compression ratio (e.g., 4.0 for 4:1).
    pub ratio: f64,
    /// Attack time in milliseconds.
    pub attack_ms: f64,
    /// Release time in milliseconds.
    pub release_ms: f64,
    /// Knee type.
    pub knee_type: KneeType,
    /// Soft knee width in dB.
    pub knee_width_db: f64,
    /// Makeup gain in dB.
    pub makeup_gain_db: f64,
    /// Auto makeup gain enabled.
    pub auto_makeup: bool,
    /// Lookahead time in milliseconds.
    pub lookahead_ms: f64,
}

impl Default for CompressorConfig {
    fn default() -> Self {
        Self {
            threshold_db: -20.0,
            ratio: 4.0,
            attack_ms: 10.0,
            release_ms: 100.0,
            knee_type: KneeType::Hard,
            knee_width_db: 6.0,
            makeup_gain_db: 0.0,
            auto_makeup: false,
            lookahead_ms: 0.0,
        }
    }
}

impl CompressorConfig {
    /// Create a new compressor configuration.
    #[must_use]
    pub fn new(threshold_db: f64, ratio: f64) -> Self {
        Self {
            threshold_db,
            ratio,
            ..Default::default()
        }
    }

    /// Set attack and release times.
    #[must_use]
    pub fn with_timing(mut self, attack_ms: f64, release_ms: f64) -> Self {
        self.attack_ms = attack_ms.max(0.1);
        self.release_ms = release_ms.max(0.1);
        self
    }

    /// Set soft knee.
    #[must_use]
    pub fn with_soft_knee(mut self, width_db: f64) -> Self {
        self.knee_type = KneeType::Soft;
        self.knee_width_db = width_db.max(0.0);
        self
    }

    /// Set hard knee.
    #[must_use]
    pub fn with_hard_knee(mut self) -> Self {
        self.knee_type = KneeType::Hard;
        self
    }

    /// Set makeup gain.
    #[must_use]
    pub fn with_makeup_gain(mut self, gain_db: f64) -> Self {
        self.makeup_gain_db = gain_db;
        self.auto_makeup = false;
        self
    }

    /// Enable auto makeup gain.
    #[must_use]
    pub fn with_auto_makeup(mut self) -> Self {
        self.auto_makeup = true;
        self
    }

    /// Set lookahead time.
    #[must_use]
    pub fn with_lookahead(mut self, lookahead_ms: f64) -> Self {
        self.lookahead_ms = lookahead_ms.max(0.0);
        self
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

    /// Calculate auto makeup gain based on threshold and ratio.
    #[must_use]
    pub fn calculate_auto_makeup(&self) -> f64 {
        if self.ratio <= 1.0 {
            return 0.0;
        }

        let gain_at_threshold = self.threshold_db - (self.threshold_db / self.ratio);
        -gain_at_threshold * 0.5
    }
}

/// Envelope follower for gain reduction.
struct EnvelopeFollower {
    /// Current envelope level.
    envelope: f64,
    /// Attack coefficient.
    attack_coeff: f64,
    /// Release coefficient.
    release_coeff: f64,
}

impl EnvelopeFollower {
    /// Create a new envelope follower.
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
            envelope: 0.0,
            attack_coeff,
            release_coeff,
        }
    }

    /// Update envelope with new input level.
    fn update(&mut self, input_level: f64) {
        if input_level > self.envelope {
            self.envelope =
                self.attack_coeff * self.envelope + (1.0 - self.attack_coeff) * input_level;
        } else {
            self.envelope =
                self.release_coeff * self.envelope + (1.0 - self.release_coeff) * input_level;
        }
    }

    /// Get current envelope level.
    fn level(&self) -> f64 {
        self.envelope
    }

    /// Reset envelope.
    fn reset(&mut self) {
        self.envelope = 0.0;
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

/// Dynamics compressor processor.
pub struct Compressor {
    /// Configuration.
    config: CompressorConfig,
    /// Envelope follower.
    envelope: EnvelopeFollower,
    /// Lookahead buffers per channel.
    lookahead_buffers: Vec<LookaheadBuffer>,
    /// Effective makeup gain (linear).
    makeup_gain: f64,
    /// Current gain reduction in dB (for metering).
    gain_reduction_db: f64,
    /// Sample rate.
    sample_rate: f64,
    /// Number of channels.
    channels: usize,
}

impl Compressor {
    /// Create a new compressor.
    ///
    /// # Arguments
    ///
    /// * `config` - Compressor configuration
    /// * `sample_rate` - Sample rate in Hz
    /// * `channels` - Number of audio channels
    #[must_use]
    pub fn new(config: CompressorConfig, sample_rate: f64, channels: usize) -> Self {
        let envelope = EnvelopeFollower::new(config.attack_ms, config.release_ms, sample_rate);

        let lookahead_buffers: Vec<_> = (0..channels)
            .map(|_| LookaheadBuffer::new(config.lookahead_ms, sample_rate))
            .collect();

        let makeup_gain = if config.auto_makeup {
            CompressorConfig::db_to_linear(config.calculate_auto_makeup())
        } else {
            CompressorConfig::db_to_linear(config.makeup_gain_db)
        };

        Self {
            config,
            envelope,
            lookahead_buffers,
            makeup_gain,
            gain_reduction_db: 0.0,
            sample_rate,
            channels,
        }
    }

    /// Set the compressor configuration.
    pub fn set_config(&mut self, config: CompressorConfig) {
        self.envelope =
            EnvelopeFollower::new(config.attack_ms, config.release_ms, self.sample_rate);

        self.makeup_gain = if config.auto_makeup {
            CompressorConfig::db_to_linear(config.calculate_auto_makeup())
        } else {
            CompressorConfig::db_to_linear(config.makeup_gain_db)
        };

        self.config = config;
    }

    /// Get the current configuration.
    #[must_use]
    pub fn config(&self) -> &CompressorConfig {
        &self.config
    }

    /// Get current gain reduction in dB (for metering).
    #[must_use]
    pub fn gain_reduction_db(&self) -> f64 {
        self.gain_reduction_db
    }

    /// Calculate gain reduction for a given input level in dB.
    fn calculate_gain_reduction(&self, input_db: f64) -> f64 {
        if input_db < self.config.threshold_db {
            return 0.0;
        }

        match self.config.knee_type {
            KneeType::Hard => {
                let excess = input_db - self.config.threshold_db;
                let compressed_excess = excess / self.config.ratio;
                -(excess - compressed_excess)
            }
            KneeType::Soft => {
                let half_knee = self.config.knee_width_db / 2.0;
                let knee_start = self.config.threshold_db - half_knee;
                let knee_end = self.config.threshold_db + half_knee;

                if input_db < knee_start {
                    0.0
                } else if input_db > knee_end {
                    let excess = input_db - self.config.threshold_db;
                    let compressed_excess = excess / self.config.ratio;
                    -(excess - compressed_excess)
                } else {
                    let x = input_db - knee_start;
                    let knee_factor = x / self.config.knee_width_db;
                    let ratio_blend = 1.0 + (self.config.ratio - 1.0) * knee_factor;
                    let excess = x;
                    let compressed = excess / ratio_blend;
                    -(excess - compressed) * knee_factor
                }
            }
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

            self.envelope.update(peak);

            let envelope_db = CompressorConfig::linear_to_db(self.envelope.level());
            self.gain_reduction_db = self.calculate_gain_reduction(envelope_db);

            let gain = CompressorConfig::db_to_linear(self.gain_reduction_db) * self.makeup_gain;

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

            self.envelope.update(peak);

            let envelope_db = CompressorConfig::linear_to_db(self.envelope.level());
            self.gain_reduction_db = self.calculate_gain_reduction(envelope_db);

            let gain = CompressorConfig::db_to_linear(self.gain_reduction_db) * self.makeup_gain;

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

            self.envelope.update(peak);

            let envelope_db = CompressorConfig::linear_to_db(self.envelope.level());
            self.gain_reduction_db = self.calculate_gain_reduction(envelope_db);

            let gain = CompressorConfig::db_to_linear(self.gain_reduction_db) * self.makeup_gain;

            if channel < self.lookahead_buffers.len() {
                let delayed = self.lookahead_buffers[channel].process(*sample);
                *sample = delayed * gain;
            }
        }
    }

    /// Reset all compressor state.
    pub fn reset(&mut self) {
        self.envelope.reset();
        for buffer in &mut self.lookahead_buffers {
            buffer.reset();
        }
        self.gain_reduction_db = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_RATE: f64 = 48000.0;

    #[test]
    fn test_compressor_config_default() {
        let config = CompressorConfig::default();
        assert_eq!(config.threshold_db, -20.0);
        assert_eq!(config.ratio, 4.0);
        assert_eq!(config.attack_ms, 10.0);
        assert_eq!(config.release_ms, 100.0);
        assert_eq!(config.knee_type, KneeType::Hard);
    }

    #[test]
    fn test_compressor_config_new() {
        let config = CompressorConfig::new(-12.0, 8.0);
        assert_eq!(config.threshold_db, -12.0);
        assert_eq!(config.ratio, 8.0);
    }

    #[test]
    fn test_compressor_config_with_timing() {
        let config = CompressorConfig::new(-20.0, 4.0).with_timing(5.0, 200.0);
        assert_eq!(config.attack_ms, 5.0);
        assert_eq!(config.release_ms, 200.0);
    }

    #[test]
    fn test_compressor_config_with_soft_knee() {
        let config = CompressorConfig::new(-20.0, 4.0).with_soft_knee(6.0);
        assert_eq!(config.knee_type, KneeType::Soft);
        assert_eq!(config.knee_width_db, 6.0);
    }

    #[test]
    fn test_compressor_config_with_hard_knee() {
        let config = CompressorConfig::new(-20.0, 4.0)
            .with_soft_knee(6.0)
            .with_hard_knee();
        assert_eq!(config.knee_type, KneeType::Hard);
    }

    #[test]
    fn test_compressor_config_with_makeup_gain() {
        let config = CompressorConfig::new(-20.0, 4.0).with_makeup_gain(6.0);
        assert_eq!(config.makeup_gain_db, 6.0);
        assert!(!config.auto_makeup);
    }

    #[test]
    fn test_compressor_config_with_auto_makeup() {
        let config = CompressorConfig::new(-20.0, 4.0).with_auto_makeup();
        assert!(config.auto_makeup);
    }

    #[test]
    fn test_db_to_linear_and_back() {
        let db = -6.0;
        let linear = CompressorConfig::db_to_linear(db);
        let back = CompressorConfig::linear_to_db(linear);
        assert!((back - db).abs() < 1e-10);
    }

    #[test]
    fn test_linear_to_db_zero() {
        let result = CompressorConfig::linear_to_db(0.0);
        assert_eq!(result, f64::NEG_INFINITY);
    }

    #[test]
    fn test_calculate_auto_makeup() {
        let config = CompressorConfig::new(-20.0, 4.0);
        let makeup = config.calculate_auto_makeup();
        // Auto makeup should be positive for threshold < 0 and ratio > 1
        assert!(makeup > 0.0);
    }

    #[test]
    fn test_compressor_new() {
        let config = CompressorConfig::default();
        let comp = Compressor::new(config, SAMPLE_RATE, 2);
        assert_eq!(comp.gain_reduction_db(), 0.0);
    }

    #[test]
    fn test_compressor_no_reduction_below_threshold() {
        // Signal below threshold should have no gain reduction
        let config = CompressorConfig::new(-20.0, 4.0).with_timing(1.0, 100.0);
        let mut comp = Compressor::new(config, SAMPLE_RATE, 1);
        // Feed a quiet signal (below -20 dB threshold)
        let mut samples = vec![0.001_f64; 100];
        comp.process_channel(0, &mut samples);
        assert!(
            comp.gain_reduction_db() > -1.0,
            "Should have minimal gain reduction below threshold"
        );
    }

    #[test]
    fn test_compressor_reduces_loud_signal() {
        // Signal well above threshold should be compressed
        let config = CompressorConfig::new(-40.0, 8.0).with_timing(1.0, 100.0);
        let mut comp = Compressor::new(config, SAMPLE_RATE, 1);
        // Feed a full-scale signal
        let mut samples = vec![1.0_f64; 2000];
        comp.process_interleaved(&mut samples, 2000);
        // After attack, gain reduction should be significant
        assert!(
            comp.gain_reduction_db() < -3.0,
            "Should have gain reduction, got {}",
            comp.gain_reduction_db()
        );
    }

    #[test]
    fn test_compressor_process_planar() {
        let config = CompressorConfig::new(-20.0, 4.0);
        let mut comp = Compressor::new(config, SAMPLE_RATE, 2);
        let mut channels = vec![vec![0.5_f64; 256]; 2];
        comp.process_planar(&mut channels);
        for ch in &channels {
            for s in ch {
                assert!(s.is_finite());
            }
        }
    }

    #[test]
    fn test_compressor_reset() {
        let config = CompressorConfig::new(-20.0, 4.0);
        let mut comp = Compressor::new(config, SAMPLE_RATE, 1);
        let mut samples = vec![1.0_f64; 1000];
        comp.process_channel(0, &mut samples);
        comp.reset();
        assert_eq!(comp.gain_reduction_db(), 0.0);
    }

    #[test]
    fn test_compressor_set_config() {
        let config = CompressorConfig::new(-20.0, 4.0);
        let mut comp = Compressor::new(config, SAMPLE_RATE, 1);
        let new_config = CompressorConfig::new(-10.0, 2.0);
        comp.set_config(new_config);
        assert_eq!(comp.config().threshold_db, -10.0);
        assert_eq!(comp.config().ratio, 2.0);
    }

    #[test]
    fn test_compressor_soft_knee() {
        let config = CompressorConfig::new(-20.0, 4.0)
            .with_soft_knee(6.0)
            .with_timing(1.0, 50.0);
        let mut comp = Compressor::new(config, SAMPLE_RATE, 1);
        // Process a loud signal - should work without panicking
        let mut samples = vec![0.8_f64; 1000];
        comp.process_channel(0, &mut samples);
        for s in &samples {
            assert!(s.is_finite());
        }
    }

    #[test]
    fn test_compressor_with_lookahead() {
        let config = CompressorConfig::new(-20.0, 4.0).with_lookahead(5.0);
        let mut comp = Compressor::new(config, SAMPLE_RATE, 1);
        let mut samples = vec![0.5_f64; 500];
        comp.process_channel(0, &mut samples);
        for s in &samples {
            assert!(s.is_finite());
        }
    }

    #[test]
    fn test_compressor_with_auto_makeup() {
        let config = CompressorConfig::new(-20.0, 4.0).with_auto_makeup();
        let mut comp = Compressor::new(config, SAMPLE_RATE, 2);
        // With auto makeup and a loud signal, the makeup gain > 1 means output can exceed input gain reduction
        // Run a mid-level signal to verify it compresses without panicking
        let mut samples = vec![0.5_f64; 500];
        comp.process_channel(0, &mut samples);
        for s in &samples {
            assert!(s.is_finite());
        }
    }
}
