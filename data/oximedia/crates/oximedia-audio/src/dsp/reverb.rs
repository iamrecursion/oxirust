//! Reverb DSP implementation.
//!
//! This module provides a Schroeder reverb algorithm with parallel comb filters
//! followed by series all-pass filters.

#![forbid(unsafe_code)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use std::collections::VecDeque;

/// Number of parallel comb filters.
const NUM_COMBS: usize = 8;

/// Number of series all-pass filters.
const NUM_ALLPASS: usize = 4;

/// Base comb filter delays (in samples at 44.1kHz).
/// These are prime numbers to avoid artifacts.
const COMB_DELAYS: [usize; NUM_COMBS] = [1557, 1617, 1491, 1422, 1277, 1356, 1188, 1116];

/// Base all-pass filter delays (in samples at 44.1kHz).
const ALLPASS_DELAYS: [usize; NUM_ALLPASS] = [225, 556, 441, 341];

/// Configuration for the reverb effect.
#[derive(Clone, Debug)]
pub struct ReverbConfig {
    /// Room size (0.0 to 1.0). Larger values create longer reverb tails.
    pub room_size: f64,
    /// Damping factor (0.0 to 1.0). Higher values absorb more high frequencies.
    pub damping: f64,
    /// Stereo width (0.0 to 1.0). Controls the width of the stereo field.
    pub width: f64,
    /// Dry/wet mix (0.0 = dry only, 1.0 = wet only).
    pub mix: f64,
    /// Pre-delay in milliseconds.
    pub pre_delay_ms: f64,
}

impl Default for ReverbConfig {
    fn default() -> Self {
        Self {
            room_size: 0.5,
            damping: 0.5,
            width: 1.0,
            mix: 0.3,
            pre_delay_ms: 0.0,
        }
    }
}

impl ReverbConfig {
    /// Create a new reverb configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set room size.
    #[must_use]
    pub fn with_room_size(mut self, room_size: f64) -> Self {
        self.room_size = room_size.clamp(0.0, 1.0);
        self
    }

    /// Set damping.
    #[must_use]
    pub fn with_damping(mut self, damping: f64) -> Self {
        self.damping = damping.clamp(0.0, 1.0);
        self
    }

    /// Set stereo width.
    #[must_use]
    pub fn with_width(mut self, width: f64) -> Self {
        self.width = width.clamp(0.0, 1.0);
        self
    }

    /// Set dry/wet mix.
    #[must_use]
    pub fn with_mix(mut self, mix: f64) -> Self {
        self.mix = mix.clamp(0.0, 1.0);
        self
    }

    /// Set pre-delay.
    #[must_use]
    pub fn with_pre_delay(mut self, pre_delay_ms: f64) -> Self {
        self.pre_delay_ms = pre_delay_ms.max(0.0);
        self
    }

    /// Create a small room preset.
    #[must_use]
    pub fn small_room() -> Self {
        Self::new()
            .with_room_size(0.3)
            .with_damping(0.5)
            .with_width(0.8)
            .with_mix(0.25)
    }

    /// Create a medium room preset.
    #[must_use]
    pub fn medium_room() -> Self {
        Self::new()
            .with_room_size(0.5)
            .with_damping(0.5)
            .with_width(1.0)
            .with_mix(0.3)
    }

    /// Create a large hall preset.
    #[must_use]
    pub fn large_hall() -> Self {
        Self::new()
            .with_room_size(0.85)
            .with_damping(0.3)
            .with_width(1.0)
            .with_mix(0.4)
            .with_pre_delay(20.0)
    }

    /// Create a plate reverb preset.
    #[must_use]
    pub fn plate() -> Self {
        Self::new()
            .with_room_size(0.6)
            .with_damping(0.7)
            .with_width(0.5)
            .with_mix(0.35)
    }
}

/// Comb filter for reverb.
struct CombFilter {
    /// Circular buffer.
    buffer: VecDeque<f64>,
    /// Buffer size.
    buffer_size: usize,
    /// Feedback gain.
    feedback: f64,
    /// Damping coefficient.
    damping: f64,
    /// Low-pass filter state for damping.
    filter_state: f64,
}

impl CombFilter {
    /// Create a new comb filter.
    fn new(delay_samples: usize) -> Self {
        let mut buffer = VecDeque::with_capacity(delay_samples + 1);
        for _ in 0..delay_samples {
            buffer.push_back(0.0);
        }

        Self {
            buffer,
            buffer_size: delay_samples,
            feedback: 0.0,
            damping: 0.0,
            filter_state: 0.0,
        }
    }

    /// Set feedback gain.
    fn set_feedback(&mut self, feedback: f64) {
        self.feedback = feedback;
    }

    /// Set damping coefficient.
    fn set_damping(&mut self, damping: f64) {
        self.damping = damping;
    }

    /// Process one sample through the comb filter.
    fn process(&mut self, input: f64) -> f64 {
        let output = self.buffer.pop_front().unwrap_or(0.0);

        let filtered = output * (1.0 - self.damping) + self.filter_state * self.damping;
        self.filter_state = filtered;

        self.buffer.push_back(input + filtered * self.feedback);

        output
    }

    /// Reset filter state.
    fn reset(&mut self) {
        self.buffer.clear();
        for _ in 0..self.buffer_size {
            self.buffer.push_back(0.0);
        }
        self.filter_state = 0.0;
    }
}

/// All-pass filter for reverb.
struct AllPassFilter {
    /// Circular buffer.
    buffer: VecDeque<f64>,
    /// Buffer size.
    buffer_size: usize,
    /// Feedback/feedforward gain.
    gain: f64,
}

impl AllPassFilter {
    /// Create a new all-pass filter.
    fn new(delay_samples: usize) -> Self {
        let mut buffer = VecDeque::with_capacity(delay_samples + 1);
        for _ in 0..delay_samples {
            buffer.push_back(0.0);
        }

        Self {
            buffer,
            buffer_size: delay_samples,
            gain: 0.5,
        }
    }

    /// Process one sample through the all-pass filter.
    fn process(&mut self, input: f64) -> f64 {
        let delayed = self.buffer.pop_front().unwrap_or(0.0);

        let output = -input + delayed;

        self.buffer.push_back(input + delayed * self.gain);

        output
    }

    /// Reset filter state.
    fn reset(&mut self) {
        self.buffer.clear();
        for _ in 0..self.buffer_size {
            self.buffer.push_back(0.0);
        }
    }
}

/// Pre-delay buffer.
struct PreDelayBuffer {
    /// Circular buffer.
    buffer: VecDeque<f64>,
    /// Delay in samples.
    delay_samples: usize,
}

impl PreDelayBuffer {
    /// Create a new pre-delay buffer.
    fn new(delay_ms: f64, sample_rate: f64) -> Self {
        let delay_samples = (delay_ms * 0.001 * sample_rate) as usize;
        Self {
            buffer: VecDeque::with_capacity(delay_samples + 1),
            delay_samples,
        }
    }

    /// Process one sample through the pre-delay.
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

/// Reverb processor for a single channel.
struct ReverbChannel {
    /// Parallel comb filters.
    combs: Vec<CombFilter>,
    /// Series all-pass filters.
    allpass: Vec<AllPassFilter>,
    /// Pre-delay buffer.
    pre_delay: PreDelayBuffer,
}

impl ReverbChannel {
    /// Create a new reverb channel.
    fn new(sample_rate: f64, pre_delay_ms: f64, stereo_spread: bool) -> Self {
        let rate_factor = sample_rate / 44100.0;

        let mut combs = Vec::new();
        for (i, &base_delay) in COMB_DELAYS.iter().enumerate() {
            let delay = (base_delay as f64 * rate_factor) as usize;
            let adjusted_delay = if stereo_spread && i % 2 == 1 {
                (delay as f64 * 1.03) as usize
            } else {
                delay
            };
            combs.push(CombFilter::new(adjusted_delay.max(1)));
        }

        let mut allpass = Vec::new();
        for (i, &base_delay) in ALLPASS_DELAYS.iter().enumerate() {
            let delay = (base_delay as f64 * rate_factor) as usize;
            let adjusted_delay = if stereo_spread && i % 2 == 1 {
                (delay as f64 * 1.03) as usize
            } else {
                delay
            };
            allpass.push(AllPassFilter::new(adjusted_delay.max(1)));
        }

        let pre_delay = PreDelayBuffer::new(pre_delay_ms, sample_rate);

        Self {
            combs,
            allpass,
            pre_delay,
        }
    }

    /// Update parameters.
    fn update_parameters(&mut self, config: &ReverbConfig) {
        let room_scale = 0.28;
        let room_offset = 0.7;
        let feedback = config.room_size * room_scale + room_offset;

        for comb in &mut self.combs {
            comb.set_feedback(feedback);
            comb.set_damping(config.damping);
        }
    }

    /// Process one sample.
    fn process(&mut self, input: f64) -> f64 {
        let pre_delayed = self.pre_delay.process(input);

        let mut comb_sum = 0.0;
        for comb in &mut self.combs {
            comb_sum += comb.process(pre_delayed);
        }

        let mut output = comb_sum;
        for ap in &mut self.allpass {
            output = ap.process(output);
        }

        output
    }

    /// Reset all filters.
    fn reset(&mut self) {
        for comb in &mut self.combs {
            comb.reset();
        }
        for ap in &mut self.allpass {
            ap.reset();
        }
        self.pre_delay.reset();
    }
}

/// Stereo reverb processor.
pub struct Reverb {
    /// Configuration.
    config: ReverbConfig,
    /// Left channel reverb.
    left_reverb: ReverbChannel,
    /// Right channel reverb.
    right_reverb: ReverbChannel,
    /// Sample rate.
    #[allow(dead_code)]
    sample_rate: f64,
}

impl Reverb {
    /// Create a new stereo reverb.
    ///
    /// # Arguments
    ///
    /// * `config` - Reverb configuration
    /// * `sample_rate` - Sample rate in Hz
    #[must_use]
    pub fn new(config: ReverbConfig, sample_rate: f64) -> Self {
        let mut reverb = Self {
            left_reverb: ReverbChannel::new(sample_rate, config.pre_delay_ms, false),
            right_reverb: ReverbChannel::new(sample_rate, config.pre_delay_ms, true),
            config: config.clone(),
            sample_rate,
        };

        reverb.left_reverb.update_parameters(&config);
        reverb.right_reverb.update_parameters(&config);

        reverb
    }

    /// Set the reverb configuration.
    pub fn set_config(&mut self, config: ReverbConfig) {
        self.left_reverb.update_parameters(&config);
        self.right_reverb.update_parameters(&config);
        self.config = config;
    }

    /// Get the current configuration.
    #[must_use]
    pub fn config(&self) -> &ReverbConfig {
        &self.config
    }

    /// Process stereo samples (interleaved).
    ///
    /// # Arguments
    ///
    /// * `samples` - Interleaved stereo input/output buffer
    /// * `num_samples` - Number of sample frames
    pub fn process_stereo_interleaved(&mut self, samples: &mut [f64], num_samples: usize) {
        for i in 0..num_samples {
            let left_idx = i * 2;
            let right_idx = i * 2 + 1;

            if left_idx >= samples.len() || right_idx >= samples.len() {
                break;
            }

            let dry_left = samples[left_idx];
            let dry_right = samples[right_idx];

            let input_mono = (dry_left + dry_right) * 0.5;

            let wet_left = self.left_reverb.process(input_mono);
            let wet_right = self.right_reverb.process(input_mono);

            let width1 = self.config.width * 0.5;
            let width2 = (1.0 - self.config.width) * 0.5;

            let wet_left_final = wet_left * width1 + wet_right * width2;
            let wet_right_final = wet_right * width1 + wet_left * width2;

            samples[left_idx] =
                dry_left * (1.0 - self.config.mix) + wet_left_final * self.config.mix;
            samples[right_idx] =
                dry_right * (1.0 - self.config.mix) + wet_right_final * self.config.mix;
        }
    }

    /// Process stereo samples (planar).
    ///
    /// # Arguments
    ///
    /// * `left` - Left channel input/output buffer
    /// * `right` - Right channel input/output buffer
    pub fn process_stereo_planar(&mut self, left: &mut [f64], right: &mut [f64]) {
        let num_samples = left.len().min(right.len());

        for i in 0..num_samples {
            let dry_left = left[i];
            let dry_right = right[i];

            let input_mono = (dry_left + dry_right) * 0.5;

            let wet_left = self.left_reverb.process(input_mono);
            let wet_right = self.right_reverb.process(input_mono);

            let width1 = self.config.width * 0.5;
            let width2 = (1.0 - self.config.width) * 0.5;

            let wet_left_final = wet_left * width1 + wet_right * width2;
            let wet_right_final = wet_right * width1 + wet_left * width2;

            left[i] = dry_left * (1.0 - self.config.mix) + wet_left_final * self.config.mix;
            right[i] = dry_right * (1.0 - self.config.mix) + wet_right_final * self.config.mix;
        }
    }

    /// Process mono samples.
    ///
    /// # Arguments
    ///
    /// * `samples` - Mono input/output buffer
    pub fn process_mono(&mut self, samples: &mut [f64]) {
        for sample in samples.iter_mut() {
            let dry = *sample;

            let wet_left = self.left_reverb.process(dry);
            let wet_right = self.right_reverb.process(dry);

            let wet = (wet_left + wet_right) * 0.5;

            *sample = dry * (1.0 - self.config.mix) + wet * self.config.mix;
        }
    }

    /// Reset all reverb state.
    pub fn reset(&mut self) {
        self.left_reverb.reset();
        self.right_reverb.reset();
    }
}

/// Mono reverb processor (single channel).
pub struct MonoReverb {
    /// Configuration.
    config: ReverbConfig,
    /// Reverb channel.
    reverb: ReverbChannel,
}

impl MonoReverb {
    /// Create a new mono reverb.
    ///
    /// # Arguments
    ///
    /// * `config` - Reverb configuration
    /// * `sample_rate` - Sample rate in Hz
    #[must_use]
    pub fn new(config: ReverbConfig, sample_rate: f64) -> Self {
        let mut reverb_channel = ReverbChannel::new(sample_rate, config.pre_delay_ms, false);
        reverb_channel.update_parameters(&config);

        Self {
            config,
            reverb: reverb_channel,
        }
    }

    /// Set the reverb configuration.
    pub fn set_config(&mut self, config: ReverbConfig) {
        self.reverb.update_parameters(&config);
        self.config = config;
    }

    /// Get the current configuration.
    #[must_use]
    pub fn config(&self) -> &ReverbConfig {
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
            let wet = self.reverb.process(dry);
            *sample = dry * (1.0 - self.config.mix) + wet * self.config.mix;
        }
    }

    /// Reset reverb state.
    pub fn reset(&mut self) {
        self.reverb.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_RATE: f64 = 48000.0;

    #[test]
    fn test_reverb_config_default() {
        let config = ReverbConfig::default();
        assert_eq!(config.room_size, 0.5);
        assert_eq!(config.damping, 0.5);
        assert_eq!(config.width, 1.0);
        assert_eq!(config.mix, 0.3);
        assert_eq!(config.pre_delay_ms, 0.0);
    }

    #[test]
    fn test_reverb_config_with_room_size() {
        let config = ReverbConfig::new().with_room_size(0.8);
        assert_eq!(config.room_size, 0.8);
    }

    #[test]
    fn test_reverb_config_room_size_clamped() {
        let config = ReverbConfig::new().with_room_size(1.5);
        assert_eq!(config.room_size, 1.0);
        let config2 = ReverbConfig::new().with_room_size(-0.5);
        assert_eq!(config2.room_size, 0.0);
    }

    #[test]
    fn test_reverb_config_with_damping() {
        let config = ReverbConfig::new().with_damping(0.7);
        assert_eq!(config.damping, 0.7);
    }

    #[test]
    fn test_reverb_config_with_width() {
        let config = ReverbConfig::new().with_width(0.5);
        assert_eq!(config.width, 0.5);
    }

    #[test]
    fn test_reverb_config_with_mix() {
        let config = ReverbConfig::new().with_mix(0.4);
        assert_eq!(config.mix, 0.4);
    }

    #[test]
    fn test_reverb_config_mix_clamped() {
        let config = ReverbConfig::new().with_mix(1.5);
        assert_eq!(config.mix, 1.0);
    }

    #[test]
    fn test_reverb_config_small_room() {
        let config = ReverbConfig::small_room();
        assert!(config.room_size < 0.5);
    }

    #[test]
    fn test_reverb_config_medium_room() {
        let config = ReverbConfig::medium_room();
        assert_eq!(config.room_size, 0.5);
    }

    #[test]
    fn test_reverb_config_large_hall() {
        let config = ReverbConfig::large_hall();
        assert!(config.room_size > 0.7);
        assert!(config.pre_delay_ms > 0.0);
    }

    #[test]
    fn test_reverb_config_plate() {
        let config = ReverbConfig::plate();
        assert!(config.room_size > 0.0 && config.room_size <= 1.0);
    }

    #[test]
    fn test_stereo_reverb_new() {
        let config = ReverbConfig::default();
        let _reverb = Reverb::new(config, SAMPLE_RATE);
        // Just ensure it creates without panicking
    }

    #[test]
    fn test_stereo_reverb_process_mono() {
        let config = ReverbConfig::default();
        let mut reverb = Reverb::new(config, SAMPLE_RATE);
        let mut samples = vec![1.0_f64; 512];
        reverb.process_mono(&mut samples);
        // All outputs should be finite and within reasonable range
        for s in &samples {
            assert!(s.is_finite(), "Sample should be finite, got {s}");
        }
    }

    #[test]
    fn test_stereo_reverb_process_stereo_interleaved() {
        let config = ReverbConfig::medium_room();
        let mut reverb = Reverb::new(config, SAMPLE_RATE);
        let mut samples = vec![0.5_f64; 256]; // 128 frames * 2 channels
        reverb.process_stereo_interleaved(&mut samples, 128);
        for s in &samples {
            assert!(s.is_finite());
        }
    }

    #[test]
    fn test_stereo_reverb_process_stereo_planar() {
        let config = ReverbConfig::large_hall();
        let mut reverb = Reverb::new(config, SAMPLE_RATE);
        let mut left = vec![1.0_f64; 256];
        let mut right = vec![1.0_f64; 256];
        reverb.process_stereo_planar(&mut left, &mut right);
        for (l, r) in left.iter().zip(right.iter()) {
            assert!(l.is_finite());
            assert!(r.is_finite());
        }
    }

    #[test]
    fn test_stereo_reverb_dry_mix_zero() {
        // With mix=0, output should equal input (dry only)
        let config = ReverbConfig::new().with_mix(0.0);
        let mut reverb = Reverb::new(config, SAMPLE_RATE);
        let mut left = vec![0.7_f64; 100];
        let mut right = vec![0.3_f64; 100];
        reverb.process_stereo_planar(&mut left, &mut right);
        for l in &left {
            assert!((l - 0.7).abs() < 1e-10);
        }
        for r in &right {
            assert!((r - 0.3).abs() < 1e-10);
        }
    }

    #[test]
    fn test_stereo_reverb_reset() {
        let config = ReverbConfig::default();
        let mut reverb = Reverb::new(config, SAMPLE_RATE);
        let mut samples = vec![0.8_f64; 512];
        reverb.process_mono(&mut samples);
        reverb.reset();
        // After reset, tail should be cleared
        let mut silence = vec![0.0_f64; 100];
        reverb.process_mono(&mut silence);
        // After reset and silence, output should be near zero
        assert!(
            silence[99].abs() < 0.01,
            "After reset, reverb tail should clear quickly"
        );
    }

    #[test]
    fn test_stereo_reverb_set_config() {
        let config = ReverbConfig::small_room();
        let mut reverb = Reverb::new(config, SAMPLE_RATE);
        let new_config = ReverbConfig::large_hall();
        reverb.set_config(new_config.clone());
        assert_eq!(reverb.config().room_size, new_config.room_size);
    }

    #[test]
    fn test_stereo_reverb_with_pre_delay() {
        let config = ReverbConfig::new().with_pre_delay(10.0).with_mix(0.5);
        let mut reverb = Reverb::new(config, SAMPLE_RATE);
        let mut samples = vec![0.5_f64; 1024];
        reverb.process_mono(&mut samples);
        for s in &samples {
            assert!(s.is_finite());
        }
    }

    #[test]
    fn test_mono_reverb_new() {
        let config = ReverbConfig::default();
        let _reverb = MonoReverb::new(config, SAMPLE_RATE);
    }

    #[test]
    fn test_mono_reverb_process() {
        let config = ReverbConfig::medium_room();
        let mut reverb = MonoReverb::new(config, SAMPLE_RATE);
        let mut samples = vec![1.0_f64; 512];
        reverb.process(&mut samples);
        for s in &samples {
            assert!(s.is_finite());
        }
    }

    #[test]
    fn test_mono_reverb_reset() {
        let config = ReverbConfig::default();
        let mut reverb = MonoReverb::new(config, SAMPLE_RATE);
        let mut samples = vec![0.9_f64; 512];
        reverb.process(&mut samples);
        reverb.reset();
        let mut silence = vec![0.0_f64; 50];
        reverb.process(&mut silence);
        assert!(silence[49].abs() < 0.01);
    }

    #[test]
    fn test_mono_reverb_set_config() {
        let config = ReverbConfig::small_room();
        let mut reverb = MonoReverb::new(config, SAMPLE_RATE);
        reverb.set_config(ReverbConfig::large_hall());
        assert!(reverb.config().room_size > 0.7);
    }
}
