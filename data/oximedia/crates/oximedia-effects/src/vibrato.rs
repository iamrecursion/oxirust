//! Vibrato effect — LFO-modulated pitch variation via delay modulation.
//!
//! Vibrato is a periodic variation of pitch achieved by continuously varying
//! the playback position within a short delay buffer. Unlike a flanger, vibrato
//! produces **pure pitch modulation** (100 % wet) without a fixed comb-filter
//! artifact — only the delayed path is output.
//!
//! This module provides [`VibratoEffect`] configured via [`VibratoEffectConfig`].
//! (See [`crate::modulation`] for the `StereoVibrato` variant that integrates
//! with the [`crate::AudioEffect`] trait.)
//!
//! # Example
//!
//! ```
//! use oximedia_effects::vibrato::{VibratoEffectConfig, VibratoEffect};
//!
//! let config = VibratoEffectConfig::default();
//! let mut vibrato = VibratoEffect::new(config, 44_100.0);
//!
//! let mut buffer: Vec<f32> = (0..512)
//!     .map(|i| (i as f32 * 0.1).sin() * 0.5)
//!     .collect();
//!
//! vibrato.apply_buffer(&mut buffer);
//! ```

#![allow(dead_code)]

use std::f32::consts::PI;

/// Configuration for the [`VibratoEffect`].
#[derive(Debug, Clone)]
pub struct VibratoEffectConfig {
    /// Modulation depth — maximum delay excursion in milliseconds (0.1 – 30 ms).
    pub depth_ms: f32,
    /// Modulation rate in Hz (0.1 – 20 Hz).
    pub rate_hz: f32,
    /// Wet/dry ratio (0.0 = bypass, 1.0 = full vibrato). Typically kept at 1.0.
    pub mix: f32,
    /// If `true`, add a small pre-delay so the modulation is centred in time.
    pub centred: bool,
}

impl Default for VibratoEffectConfig {
    fn default() -> Self {
        Self {
            depth_ms: 3.0,
            rate_hz: 5.0,
            mix: 1.0,
            centred: true,
        }
    }
}

impl VibratoEffectConfig {
    /// Subtle, classical string-vibrato preset.
    #[must_use]
    pub fn subtle() -> Self {
        Self {
            depth_ms: 1.0,
            rate_hz: 5.5,
            mix: 1.0,
            centred: true,
        }
    }

    /// Wide, expressive guitar-vibrato preset.
    #[must_use]
    pub fn wide() -> Self {
        Self {
            depth_ms: 8.0,
            rate_hz: 4.0,
            mix: 1.0,
            centred: false,
        }
    }

    /// Fast, shallow tremolo-adjacent effect.
    #[must_use]
    pub fn fast_shallow() -> Self {
        Self {
            depth_ms: 0.5,
            rate_hz: 10.0,
            mix: 1.0,
            centred: true,
        }
    }

    /// Return `true` if the configuration is valid.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.depth_ms > 0.0 && self.rate_hz > 0.0 && self.mix >= 0.0 && self.mix <= 1.0
    }
}

/// A vibrato effect backed by a circular delay buffer.
///
/// The write head moves at a fixed rate (1 sample per tick) while the read
/// head is modulated by a sine LFO, producing pitch variation.
pub struct VibratoEffect {
    config: VibratoEffectConfig,
    sample_rate: f32,
    /// Internal delay ring buffer.
    buffer: Vec<f32>,
    mask: usize,
    write_pos: usize,
    /// LFO phase accumulator in radians.
    lfo_phase: f32,
    lfo_inc: f32,
    /// Centre delay in samples (used when `centred` is true).
    centre_delay: f32,
    /// Maximum modulation depth in samples.
    depth_samples: f32,
}

impl VibratoEffect {
    /// Create a new vibrato effect at the given sample rate.
    #[must_use]
    pub fn new(config: VibratoEffectConfig, sample_rate: f32) -> Self {
        let depth_samples = config.depth_ms * 0.001 * sample_rate;
        let centre_delay = if config.centred { depth_samples } else { 0.0 };

        // Buffer must hold depth * 2 + some margin
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let min_cap = (depth_samples * 3.0) as usize + 4;
        let capacity = min_cap.next_power_of_two().max(4);

        let lfo_inc = 2.0 * PI * config.rate_hz / sample_rate;

        Self {
            config,
            sample_rate,
            buffer: vec![0.0; capacity],
            mask: capacity - 1,
            write_pos: 0,
            lfo_phase: 0.0,
            lfo_inc,
            centre_delay,
            depth_samples,
        }
    }

    /// Read a linearly interpolated sample from the delay buffer.
    #[allow(clippy::cast_precision_loss)]
    fn read_interpolated(&self, delay_samples: f32) -> f32 {
        let d = delay_samples.clamp(0.0, (self.buffer.len() - 1) as f32);
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let d_int = d as usize;
        let d_frac = d - d_int as f32;

        let i0 = self.write_pos.wrapping_sub(d_int).wrapping_sub(1) & self.mask;
        let i1 = self.write_pos.wrapping_sub(d_int).wrapping_sub(2) & self.mask;
        let s0 = self.buffer[i0];
        let s1 = self.buffer[i1];
        s0 + d_frac * (s1 - s0)
    }

    /// Process a single sample through the vibrato.
    #[must_use]
    pub fn apply_sample(&mut self, input: f32) -> f32 {
        // Write to buffer
        self.buffer[self.write_pos] = input;
        self.write_pos = (self.write_pos + 1) & self.mask;

        // Compute modulated read delay
        let lfo = self.lfo_phase.sin(); // [-1, +1]
        let delay = self.centre_delay + lfo * self.depth_samples;
        let delay = delay.max(0.5); // guarantee at least half-sample delay

        // Read delayed sample
        let wet = self.read_interpolated(delay);

        // Advance LFO
        self.lfo_phase += self.lfo_inc;
        if self.lfo_phase >= 2.0 * PI {
            self.lfo_phase -= 2.0 * PI;
        }

        // Mix
        input * (1.0 - self.config.mix) + wet * self.config.mix
    }

    /// Process a mono buffer of samples in-place.
    pub fn apply_buffer(&mut self, buffer: &mut [f32]) {
        for s in buffer.iter_mut() {
            *s = self.apply_sample(*s);
        }
    }

    /// Process stereo buffers in-place.
    ///
    /// Both channels share the same LFO phase.
    pub fn apply_stereo(&mut self, left: &mut [f32], right: &mut [f32]) {
        let len = left.len().min(right.len());
        for i in 0..len {
            left[i] = self.apply_sample(left[i]);
            right[i] = self.apply_sample(right[i]);
        }
    }

    /// Reset internal state (buffer, LFO phase).
    pub fn reset(&mut self) {
        for s in &mut self.buffer {
            *s = 0.0;
        }
        self.write_pos = 0;
        self.lfo_phase = 0.0;
    }

    /// Set a new modulation rate.
    pub fn set_rate_hz(&mut self, rate_hz: f32) {
        self.config.rate_hz = rate_hz.max(1e-3);
        self.lfo_inc = 2.0 * PI * self.config.rate_hz / self.sample_rate;
    }

    /// Set a new modulation depth in milliseconds.
    ///
    /// If the new depth exceeds half the buffer capacity, the buffer is resized.
    pub fn set_depth_ms(&mut self, depth_ms: f32) {
        self.config.depth_ms = depth_ms.max(0.0);
        self.depth_samples = self.config.depth_ms * 0.001 * self.sample_rate;
        if self.config.centred {
            self.centre_delay = self.depth_samples;
        }
    }

    /// Return the current LFO phase.
    #[must_use]
    pub fn lfo_phase(&self) -> f32 {
        self.lfo_phase
    }

    /// Return the configured depth in samples.
    #[must_use]
    pub fn depth_samples(&self) -> f32 {
        self.depth_samples
    }

    /// Return a reference to the current configuration.
    #[must_use]
    pub fn config(&self) -> &VibratoEffectConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vibrato() -> VibratoEffect {
        VibratoEffect::new(VibratoEffectConfig::default(), 48_000.0)
    }

    #[test]
    fn test_vibrato_config_default_valid() {
        assert!(VibratoEffectConfig::default().is_valid());
    }

    #[test]
    fn test_vibrato_config_subtle_valid() {
        assert!(VibratoEffectConfig::subtle().is_valid());
    }

    #[test]
    fn test_vibrato_config_wide_valid() {
        assert!(VibratoEffectConfig::wide().is_valid());
    }

    #[test]
    fn test_vibrato_config_fast_shallow_valid() {
        assert!(VibratoEffectConfig::fast_shallow().is_valid());
    }

    #[test]
    fn test_vibrato_config_invalid_depth() {
        let c = VibratoEffectConfig {
            depth_ms: 0.0,
            ..VibratoEffectConfig::default()
        };
        assert!(!c.is_valid());
    }

    #[test]
    fn test_vibrato_new_buffer_power_of_two() {
        let v = make_vibrato();
        assert!(v.buffer.len().is_power_of_two());
    }

    #[test]
    fn test_vibrato_apply_sample_no_panic() {
        let mut v = make_vibrato();
        for i in 0..4096 {
            let _ = v.apply_sample((i as f32 * 0.01).sin());
        }
    }

    #[test]
    fn test_vibrato_mix_zero_is_bypass() {
        let config = VibratoEffectConfig {
            mix: 0.0,
            ..VibratoEffectConfig::default()
        };
        let mut v = VibratoEffect::new(config, 48_000.0);
        let out = v.apply_sample(0.75);
        assert!((out - 0.75).abs() < 1e-5);
    }

    #[test]
    fn test_vibrato_reset_zeros_buffer() {
        let mut v = make_vibrato();
        for _ in 0..256 {
            let _ = v.apply_sample(1.0);
        }
        v.reset();
        for &s in v.buffer.iter() {
            assert_eq!(s, 0.0);
        }
        assert_eq!(v.write_pos, 0);
        assert_eq!(v.lfo_phase(), 0.0);
    }

    #[test]
    fn test_vibrato_lfo_phase_advances() {
        let mut v = make_vibrato();
        let p0 = v.lfo_phase();
        let _ = v.apply_sample(0.0);
        assert!(v.lfo_phase() > p0);
    }

    #[test]
    fn test_vibrato_set_rate_hz() {
        let mut v = make_vibrato();
        v.set_rate_hz(8.0);
        assert!((v.config().rate_hz - 8.0).abs() < 1e-5);
    }

    #[test]
    fn test_vibrato_set_depth_ms() {
        let mut v = make_vibrato();
        v.set_depth_ms(5.0);
        let expected = 5.0 * 0.001 * 48_000.0;
        assert!((v.depth_samples() - expected).abs() < 1e-2);
    }

    #[test]
    fn test_vibrato_apply_buffer_length_preserved() {
        let mut v = make_vibrato();
        let mut buf = vec![0.5_f32; 256];
        v.apply_buffer(&mut buf);
        assert_eq!(buf.len(), 256);
    }

    #[test]
    fn test_vibrato_apply_stereo_no_panic() {
        let mut v = make_vibrato();
        let mut l = vec![0.3_f32; 128];
        let mut r = vec![-0.3_f32; 128];
        v.apply_stereo(&mut l, &mut r);
        assert_eq!(l.len(), 128);
        assert_eq!(r.len(), 128);
    }

    #[test]
    fn test_vibrato_depth_samples_matches_config() {
        let config = VibratoEffectConfig {
            depth_ms: 5.0,
            ..VibratoEffectConfig::default()
        };
        let v = VibratoEffect::new(config, 48_000.0);
        let expected = 5.0 * 0.001 * 48_000.0;
        assert!((v.depth_samples() - expected).abs() < 1e-2);
    }

    #[test]
    fn test_vibrato_silence_through_large_buffer() {
        let mut v = make_vibrato();
        // Drive with silence — output should remain very close to zero
        for _ in 0..8192 {
            let out = v.apply_sample(0.0);
            assert!(out.abs() < 1e-6, "Expected silence, got {out}");
        }
    }
}
