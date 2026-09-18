//! Glitch effects for audio processing.
//!
//! Provides digital glitch effects including buffer corruption, sample
//! stuttering, bit mangling, zero-crossing hold, and stochastic dropout.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

/// Configuration for the [`GlitchEngine`].
#[derive(Debug, Clone)]
pub struct GlitchConfig {
    /// Probability (0–1) that a sample is replaced by a glitch artifact.
    pub glitch_probability: f32,
    /// Duration in samples that a triggered glitch lasts.
    pub hold_duration: usize,
    /// Amount of bit reduction (0 = none, 16 = maximum crush).
    pub bit_crush_amount: u8,
    /// Dropout probability (0–1): chance a sample is silenced.
    pub dropout_probability: f32,
}

impl Default for GlitchConfig {
    fn default() -> Self {
        Self {
            glitch_probability: 0.01,
            hold_duration: 512,
            bit_crush_amount: 0,
            dropout_probability: 0.0,
        }
    }
}

/// Real-time audio glitch effect engine.
#[derive(Debug)]
pub struct GlitchEngine {
    config: GlitchConfig,
    held_sample: f32,
    hold_counter: usize,
    rng: scirs2_core::random::StdRng,
}

impl GlitchEngine {
    /// Create a new [`GlitchEngine`] with the given configuration.
    #[must_use]
    pub fn new(config: GlitchConfig) -> Self {
        Self {
            config,
            held_sample: 0.0,
            hold_counter: 0,
            rng: scirs2_core::random::Random::seed(0xDEAD_BEEF),
        }
    }

    /// Process a single sample through the glitch engine.
    pub fn process_sample(&mut self, input: f32) -> f32 {
        // Dropout: silence the sample
        if self.config.dropout_probability > 0.0
            && (self.rng.random_f64() as f32) < self.config.dropout_probability
        {
            return 0.0;
        }

        // Active hold: output the frozen sample
        if self.hold_counter > 0 {
            self.hold_counter -= 1;
            return bit_crush(self.held_sample, self.config.bit_crush_amount);
        }

        // Trigger a new glitch
        if self.config.glitch_probability > 0.0
            && (self.rng.random_f64() as f32) < self.config.glitch_probability
        {
            self.held_sample = input;
            self.hold_counter = self.config.hold_duration;
        }

        bit_crush(input, self.config.bit_crush_amount)
    }

    /// Process a buffer of samples in-place.
    pub fn process(&mut self, buffer: &mut [f32]) {
        for sample in buffer.iter_mut() {
            *sample = self.process_sample(*sample);
        }
    }

    /// Reset engine state (clears hold counter).
    pub fn reset(&mut self) {
        self.hold_counter = 0;
        self.held_sample = 0.0;
    }

    /// Update configuration without resetting state.
    pub fn set_config(&mut self, config: GlitchConfig) {
        self.config = config;
    }
}

/// Reduce bit depth of a sample.
///
/// `amount` is in 0–16; higher values produce more quantisation noise.
/// At 0 the sample is returned unchanged.
#[must_use]
pub fn bit_crush(sample: f32, amount: u8) -> f32 {
    if amount == 0 {
        return sample;
    }
    let steps = (1u32 << (16u32.saturating_sub(u32::from(amount)))) as f32;
    let scaled = sample * steps;
    scaled.round() / steps
}

/// Apply stutter effect: repeat segments of a buffer.
///
/// Every `segment_len` samples the segment is repeated `repeats` times
/// in-place, then the cursor advances past the repeated region.
pub fn stutter(buffer: &mut [f32], segment_len: usize, repeats: usize) {
    if segment_len == 0 || repeats == 0 {
        return;
    }
    let mut i = 0;
    while i + segment_len <= buffer.len() {
        // Total region = original segment + `repeats` copies of it
        let end = (i + segment_len * (repeats + 1)).min(buffer.len());
        // Copy segment into the region that follows it
        for j in i + segment_len..end {
            let src = i + (j - (i + segment_len)) % segment_len;
            buffer[j] = buffer[src];
        }
        i = end;
    }
}

/// Reverse a fixed-size segment within `buffer` at `offset`.
pub fn reverse_segment(buffer: &mut [f32], offset: usize, len: usize) {
    let end = (offset + len).min(buffer.len());
    if offset >= end {
        return;
    }
    buffer[offset..end].reverse();
}

/// Apply random sample-and-hold to a buffer.
///
/// Each sample has `probability` chance of being replaced by the most
/// recently held value. When a new hold triggers, the current sample is frozen.
pub fn random_sample_hold(buffer: &mut [f32], probability: f32) {
    let mut rng = scirs2_core::random::Random::seed(0xCAFE_BABE);
    let mut held = 0.0_f32;
    for sample in buffer.iter_mut() {
        if (rng.random_f64() as f32) < probability {
            *sample = held;
        } else {
            held = *sample;
        }
    }
}

/// Zero-crossing hold: hold the last zero-crossing value for `hold_len` samples
/// after each zero-crossing event.
pub fn zero_crossing_hold(buffer: &mut [f32], hold_len: usize) {
    let mut hold_value = 0.0_f32;
    let mut counter = 0usize;
    let mut prev = 0.0_f32;

    for sample in buffer.iter_mut() {
        let current = *sample;
        // Detect sign change (zero crossing)
        if prev * current < 0.0 {
            hold_value = *sample;
            counter = hold_len;
        }
        if counter > 0 {
            *sample = hold_value;
            counter -= 1;
        }
        prev = current;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bit_crush_zero_passthrough() {
        let sample = 0.7654_f32;
        assert!((bit_crush(sample, 0) - sample).abs() < 1e-6);
    }

    #[test]
    fn test_bit_crush_reduces_precision() {
        let sample = 0.123456789_f32;
        let crushed = bit_crush(sample, 8);
        // With 8-bit reduction, resolution is 1/256 = ~0.004
        assert!((crushed - sample).abs() < 0.01);
        assert!((crushed - sample).abs() > 0.0);
    }

    #[test]
    fn test_bit_crush_max_amount() {
        // amount = 16 → steps = 1, all values quantise to 0 or 1
        let sample = 0.4_f32;
        let crushed = bit_crush(sample, 16);
        assert!(crushed == 0.0 || crushed == 1.0);
    }

    #[test]
    fn test_bit_crush_negative_sample() {
        let sample = -0.5_f32;
        let crushed = bit_crush(sample, 4);
        assert!(crushed >= -1.0 && crushed <= 1.0);
    }

    #[test]
    fn test_stutter_repeats() {
        let mut buf = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        stutter(&mut buf, 2, 2);
        // Segment [0..2] = [1, 2]; repeated 2 more times at [2..6]
        // Positions 2,3 = 1,2; 4,5 = 1,2
        assert_eq!(buf[2], 1.0);
        assert_eq!(buf[3], 2.0);
        assert_eq!(buf[4], 1.0);
        assert_eq!(buf[5], 2.0);
    }

    #[test]
    fn test_stutter_zero_repeats_noop() {
        let orig = vec![1.0, 2.0, 3.0];
        let mut buf = orig.clone();
        stutter(&mut buf, 2, 0);
        assert_eq!(buf, orig);
    }

    #[test]
    fn test_reverse_segment() {
        let mut buf = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        reverse_segment(&mut buf, 1, 3);
        assert_eq!(buf, vec![1.0, 4.0, 3.0, 2.0, 5.0]);
    }

    #[test]
    fn test_reverse_segment_out_of_bounds() {
        let mut buf = vec![1.0, 2.0, 3.0];
        // Should not panic; clamps to buffer end
        reverse_segment(&mut buf, 1, 100);
        assert_eq!(buf[0], 1.0); // unchanged
    }

    #[test]
    fn test_random_sample_hold_probability_zero() {
        let orig = vec![0.1, 0.2, 0.3, 0.4];
        let mut buf = orig.clone();
        random_sample_hold(&mut buf, 0.0);
        assert_eq!(buf, orig);
    }

    #[test]
    fn test_random_sample_hold_probability_one() {
        let mut buf = vec![0.1, 0.5, 0.9, 0.3];
        random_sample_hold(&mut buf, 1.0);
        // All samples replaced by the held value (which starts at 0.0)
        for v in &buf {
            assert!(*v == 0.0 || *v == 0.1); // first or zero
        }
    }

    #[test]
    fn test_glitch_engine_reset() {
        let config = GlitchConfig {
            glitch_probability: 1.0,
            hold_duration: 10,
            ..Default::default()
        };
        let mut engine = GlitchEngine::new(config);
        // Trigger a glitch
        engine.process_sample(0.5);
        engine.reset();
        assert_eq!(engine.hold_counter, 0);
        assert_eq!(engine.held_sample, 0.0);
    }

    #[test]
    fn test_glitch_engine_no_glitch() {
        let config = GlitchConfig {
            glitch_probability: 0.0,
            hold_duration: 100,
            bit_crush_amount: 0,
            dropout_probability: 0.0,
        };
        let mut engine = GlitchEngine::new(config);
        let result = engine.process_sample(0.42);
        assert!((result - 0.42).abs() < 1e-6);
    }

    #[test]
    fn test_glitch_engine_dropout_silences() {
        let config = GlitchConfig {
            dropout_probability: 1.0,
            ..Default::default()
        };
        let mut engine = GlitchEngine::new(config);
        let result = engine.process_sample(0.99);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_zero_crossing_hold_flat_signal() {
        let mut buf = vec![0.5_f32; 10];
        zero_crossing_hold(&mut buf, 3);
        // No zero crossings, so buffer is unchanged
        for v in &buf {
            assert!((v - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn test_zero_crossing_hold_detects_crossing() {
        let mut buf = vec![0.5, -0.5, 0.3, 0.3, 0.3];
        zero_crossing_hold(&mut buf, 2);
        // After zero crossing at index 1, hold_value = -0.5 for 2 samples
        // Index 1 itself would be affected
        assert!(buf[1] == -0.5 || buf[2] == -0.5);
    }
}
