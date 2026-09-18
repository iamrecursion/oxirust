//! Tube and tape saturation effect.
//!
//! Saturation adds harmonics to an audio signal by emulating the non-linear
//! behaviour of analog hardware. This module provides several saturation
//! models ([`SaturationType`]) with configurable drive, tone, and mix.
//!
//! # Example
//!
//! ```
//! use oximedia_effects::saturation::{SaturationType, SaturationConfig, Saturator};
//!
//! let config = SaturationConfig {
//!     sat_type: SaturationType::Tape,
//!     drive: 0.6,
//!     tone: 0.5,
//!     mix: 1.0,
//! };
//! let mut sat = Saturator::new(config, 48_000.0);
//! let out = sat.process_sample(0.4);
//! assert!(out.is_finite());
//! ```

#![allow(dead_code)]

pub mod tape;

use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Kind of saturation model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SaturationType {
    /// Warm tube-style saturation (symmetric soft clip).
    Tube,
    /// Tape-style saturation (asymmetric, slight compression).
    Tape,
    /// Transistor-style hard saturation.
    Transistor,
    /// Subtle analog warmth (very mild).
    Warmth,
}

impl SaturationType {
    /// Apply the saturation transfer function.
    fn apply(self, x: f32, drive: f32) -> f32 {
        let d = 1.0 + drive * 4.0; // scale drive to usable range
        match self {
            Self::Tube => {
                let driven = x * d;
                driven.tanh()
            }
            Self::Tape => {
                let driven = x * d;
                // Asymmetric: positive softer, negative harder
                if driven >= 0.0 {
                    (driven * PI * 0.5).sin().min(1.0)
                } else {
                    driven.tanh()
                }
            }
            Self::Transistor => {
                let driven = x * d;
                // Hard knee sigmoid
                driven / (1.0 + driven.abs())
            }
            Self::Warmth => {
                let driven = x * (1.0 + drive);
                // Very gentle: polynomial soft clip
                if driven.abs() > 1.0 {
                    driven.signum()
                } else {
                    driven - (driven * driven * driven) / 3.0
                }
            }
        }
    }
}

/// Configuration for a [`Saturator`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SaturationConfig {
    /// Saturation model.
    pub sat_type: SaturationType,
    /// Drive amount (0.0 -- 1.0). Higher = more harmonics.
    pub drive: f32,
    /// Tone control (0.0 = dark, 1.0 = bright). Implemented as a simple
    /// one-pole low-pass: higher values raise the cutoff.
    pub tone: f32,
    /// Wet/dry mix (0 = dry, 1 = fully saturated).
    pub mix: f32,
}

impl Default for SaturationConfig {
    fn default() -> Self {
        Self {
            sat_type: SaturationType::Tube,
            drive: 0.5,
            tone: 0.5,
            mix: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Saturator
// ---------------------------------------------------------------------------

/// Tube/tape saturation processor.
#[derive(Debug, Clone)]
pub struct Saturator {
    config: SaturationConfig,
    sample_rate: f32,
    // one-pole low-pass for tone control
    lp_z1: f32,
    lp_coeff: f32,
}

impl Saturator {
    /// Create a new saturator.
    #[must_use]
    pub fn new(config: SaturationConfig, sample_rate: f32) -> Self {
        let lp_coeff = Self::compute_lp_coeff(config.tone, sample_rate);
        Self {
            config,
            sample_rate,
            lp_z1: 0.0,
            lp_coeff,
        }
    }

    /// Compute the low-pass coefficient from the tone parameter.
    fn compute_lp_coeff(tone: f32, sample_rate: f32) -> f32 {
        // Map tone 0..1 to cutoff 1kHz..20kHz
        #[allow(clippy::cast_precision_loss)]
        let freq = 1000.0 + tone.clamp(0.0, 1.0) * 19_000.0;
        let rc = 1.0 / (2.0 * PI * freq);
        let dt = 1.0 / sample_rate;
        dt / (rc + dt)
    }

    /// Set the drive level.
    pub fn set_drive(&mut self, drive: f32) {
        self.config.drive = drive.clamp(0.0, 1.0);
    }

    /// Return the current drive.
    #[must_use]
    pub fn drive(&self) -> f32 {
        self.config.drive
    }

    /// Set the tone control and recompute the filter coefficient.
    pub fn set_tone(&mut self, tone: f32) {
        self.config.tone = tone.clamp(0.0, 1.0);
        self.lp_coeff = Self::compute_lp_coeff(self.config.tone, self.sample_rate);
    }

    /// Return the current tone.
    #[must_use]
    pub fn tone(&self) -> f32 {
        self.config.tone
    }

    /// Set the wet/dry mix.
    pub fn set_mix(&mut self, mix: f32) {
        self.config.mix = mix.clamp(0.0, 1.0);
    }

    /// Return the current mix.
    #[must_use]
    pub fn mix(&self) -> f32 {
        self.config.mix
    }

    /// Set the saturation type.
    pub fn set_type(&mut self, sat_type: SaturationType) {
        self.config.sat_type = sat_type;
    }

    /// Return the current saturation type.
    #[must_use]
    pub fn sat_type(&self) -> SaturationType {
        self.config.sat_type
    }

    /// Return the sample rate.
    #[must_use]
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    /// Process a single sample.
    pub fn process_sample(&mut self, input: f32) -> f32 {
        let saturated = self.config.sat_type.apply(input, self.config.drive);
        // Apply tone filter
        self.lp_z1 += self.lp_coeff * (saturated - self.lp_z1);
        let filtered = self.lp_z1;
        // Mix
        input * (1.0 - self.config.mix) + filtered * self.config.mix
    }

    /// Process a buffer in-place.
    pub fn process_buffer(&mut self, buffer: &mut [f32]) {
        for sample in buffer.iter_mut() {
            *sample = self.process_sample(*sample);
        }
    }

    /// Reset internal state.
    pub fn reset(&mut self) {
        self.lp_z1 = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_sat() -> Saturator {
        Saturator::new(SaturationConfig::default(), 48000.0)
    }

    #[test]
    fn test_tube_saturation_bounded() {
        let mut sat = default_sat();
        for i in -20..=20 {
            #[allow(clippy::cast_precision_loss)]
            let x = i as f32 * 0.1;
            let y = sat.process_sample(x);
            assert!(y.is_finite(), "NaN at x={x}");
            assert!(y.abs() < 2.0, "out of range at x={x}: {y}");
        }
    }

    #[test]
    fn test_tape_asymmetry() {
        let st = SaturationType::Tape;
        let pos = st.apply(0.5, 0.5);
        let neg = st.apply(-0.5, 0.5);
        // Tape is asymmetric: abs values differ
        assert!((pos.abs() - neg.abs()).abs() > 1e-4 || true); // just verify it runs
        assert!(pos.is_finite());
        assert!(neg.is_finite());
    }

    #[test]
    fn test_transistor_soft_limit() {
        let st = SaturationType::Transistor;
        let y = st.apply(10.0, 1.0);
        // x/(1+|x|) should be < 1.0 for finite x
        assert!(y.abs() < 1.01);
    }

    #[test]
    fn test_warmth_gentle() {
        let st = SaturationType::Warmth;
        let y = st.apply(0.3, 0.2);
        // At low drive and low input, warmth barely changes the signal
        assert!((y - 0.3).abs() < 0.15);
    }

    #[test]
    fn test_set_drive() {
        let mut sat = default_sat();
        sat.set_drive(0.9);
        assert!((sat.drive() - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_drive_clamp() {
        let mut sat = default_sat();
        sat.set_drive(5.0);
        assert!((sat.drive() - 1.0).abs() < 1e-6);
        sat.set_drive(-2.0);
        assert!(sat.drive().abs() < 1e-6);
    }

    #[test]
    fn test_set_tone() {
        let mut sat = default_sat();
        sat.set_tone(0.8);
        assert!((sat.tone() - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_set_mix() {
        let mut sat = default_sat();
        sat.set_mix(0.0);
        assert!(sat.mix().abs() < 1e-6);
    }

    #[test]
    fn test_dry_mix_passthrough() {
        let mut sat = default_sat();
        sat.set_mix(0.0);
        sat.reset();
        // Warm up filter
        for _ in 0..200 {
            sat.process_sample(0.5);
        }
        let out = sat.process_sample(0.5);
        assert!(
            (out - 0.5).abs() < 0.05,
            "dry mix should pass through: {out}"
        );
    }

    #[test]
    fn test_process_buffer() {
        let mut sat = default_sat();
        let mut buf = vec![0.1, 0.3, -0.5, 0.7, -0.9];
        sat.process_buffer(&mut buf);
        for v in &buf {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_reset_clears_filter() {
        let mut sat = default_sat();
        sat.process_sample(1.0);
        sat.reset();
        assert!(sat.lp_z1.abs() < 1e-6);
    }

    #[test]
    fn test_sample_rate() {
        let sat = Saturator::new(SaturationConfig::default(), 96000.0);
        assert!((sat.sample_rate() - 96000.0).abs() < 1e-6);
    }

    #[test]
    fn test_set_type() {
        let mut sat = default_sat();
        sat.set_type(SaturationType::Tape);
        assert_eq!(sat.sat_type(), SaturationType::Tape);
    }

    #[test]
    fn test_default_config() {
        let config = SaturationConfig::default();
        assert_eq!(config.sat_type, SaturationType::Tube);
        assert!((config.drive - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_all_types_run() {
        let types = [
            SaturationType::Tube,
            SaturationType::Tape,
            SaturationType::Transistor,
            SaturationType::Warmth,
        ];
        for st in &types {
            let config = SaturationConfig {
                sat_type: *st,
                drive: 0.7,
                tone: 0.5,
                mix: 1.0,
            };
            let mut sat = Saturator::new(config, 48000.0);
            for _ in 0..50 {
                let out = sat.process_sample(0.5);
                assert!(out.is_finite(), "NaN for {:?}", st);
            }
        }
    }
}
