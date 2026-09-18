#![allow(dead_code)]
//! Automatic panning effect for stereo audio.
//!
//! This module implements an auto-pan effect that automatically sweeps the
//! stereo position of a mono or stereo signal using an LFO (Low Frequency
//! Oscillator). Supports multiple LFO waveforms (sine, triangle, square,
//! random) and configurable depth, rate, and phase.

use std::f32::consts::PI;

/// LFO waveform shape for the auto-pan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PanWaveform {
    /// Smooth sinusoidal panning.
    #[default]
    Sine,
    /// Linear triangle wave panning.
    Triangle,
    /// Hard-switching square wave panning.
    Square,
    /// Saw-tooth (ramp) panning left to right.
    Sawtooth,
    /// Smooth S-curve panning with eased transitions.
    SCurve,
}

/// Pan law used for gain distribution between channels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PanLaw {
    /// Linear pan law: simple gain split.
    Linear,
    /// Constant-power (equal power) pan law using sine/cosine.
    #[default]
    ConstantPower,
    /// -3 dB center compensation.
    Minus3dB,
    /// -4.5 dB center compensation.
    Minus4_5dB,
    /// -6 dB center compensation (linear).
    Minus6dB,
}

/// Configuration for the auto-pan effect.
#[derive(Debug, Clone)]
pub struct AutoPanConfig {
    /// LFO rate in Hz (typical range: 0.1 - 20.0).
    pub rate_hz: f32,
    /// Depth of panning (0.0 = no pan, 1.0 = full pan).
    pub depth: f32,
    /// LFO waveform shape.
    pub waveform: PanWaveform,
    /// Pan law for gain calculation.
    pub pan_law: PanLaw,
    /// Phase offset in radians (0.0 - 2*PI).
    pub phase_offset: f32,
    /// Center position (-1.0 left, 0.0 center, 1.0 right).
    pub center: f32,
    /// Whether to invert the LFO (swap left/right).
    pub invert: bool,
    /// Sample rate in Hz.
    pub sample_rate: f32,
}

impl Default for AutoPanConfig {
    fn default() -> Self {
        Self {
            rate_hz: 1.0,
            depth: 1.0,
            waveform: PanWaveform::Sine,
            pan_law: PanLaw::ConstantPower,
            phase_offset: 0.0,
            center: 0.0,
            invert: false,
            sample_rate: 48000.0,
        }
    }
}

/// Auto-pan stereo effect processor.
#[derive(Debug)]
pub struct AutoPan {
    /// Current configuration.
    config: AutoPanConfig,
    /// Current LFO phase (0.0 - 1.0).
    phase: f32,
    /// Phase increment per sample.
    phase_inc: f32,
}

impl AutoPan {
    /// Create a new auto-pan effect with the given configuration.
    #[must_use]
    pub fn new(config: AutoPanConfig) -> Self {
        let phase_inc = config.rate_hz / config.sample_rate;
        Self {
            config,
            phase: 0.0,
            phase_inc,
        }
    }

    /// Create an auto-pan with default settings.
    #[must_use]
    pub fn default_effect() -> Self {
        Self::new(AutoPanConfig::default())
    }

    /// Reset the LFO phase to zero.
    pub fn reset(&mut self) {
        self.phase = 0.0;
    }

    /// Set the LFO rate in Hz.
    pub fn set_rate(&mut self, rate_hz: f32) {
        self.config.rate_hz = rate_hz.max(0.001);
        self.phase_inc = self.config.rate_hz / self.config.sample_rate;
    }

    /// Set the panning depth.
    pub fn set_depth(&mut self, depth: f32) {
        self.config.depth = depth.clamp(0.0, 1.0);
    }

    /// Set the waveform shape.
    pub fn set_waveform(&mut self, waveform: PanWaveform) {
        self.config.waveform = waveform;
    }

    /// Set the sample rate and recalculate phase increment.
    pub fn set_sample_rate(&mut self, sample_rate: f32) {
        self.config.sample_rate = sample_rate.max(1.0);
        self.phase_inc = self.config.rate_hz / self.config.sample_rate;
    }

    /// Get the current LFO value (-1.0 to 1.0).
    fn lfo_value(&self) -> f32 {
        let p = (self.phase + self.config.phase_offset / (2.0 * PI)) % 1.0;
        let raw = match self.config.waveform {
            PanWaveform::Sine => (2.0 * PI * p).sin(),
            PanWaveform::Triangle => {
                if p < 0.25 {
                    4.0 * p
                } else if p < 0.75 {
                    2.0 - 4.0 * p
                } else {
                    -4.0 + 4.0 * p
                }
            }
            PanWaveform::Square => {
                if p < 0.5 {
                    1.0
                } else {
                    -1.0
                }
            }
            PanWaveform::Sawtooth => 2.0 * p - 1.0,
            PanWaveform::SCurve => {
                let s = (2.0 * PI * p).sin();
                // Cubic S-curve for smoother transitions
                s * s * s
            }
        };
        if self.config.invert {
            -raw
        } else {
            raw
        }
    }

    /// Calculate left and right gains for a given pan position.
    #[allow(clippy::cast_precision_loss)]
    fn compute_gains(&self, pan_position: f32) -> (f32, f32) {
        // pan_position: -1.0 (left) to 1.0 (right)
        let p = (pan_position + 1.0) * 0.5; // normalize to 0..1

        match self.config.pan_law {
            PanLaw::Linear | PanLaw::Minus6dB => (1.0 - p, p),
            PanLaw::ConstantPower => {
                let angle = p * PI * 0.5;
                (angle.cos(), angle.sin())
            }
            PanLaw::Minus3dB => {
                let angle = p * PI * 0.5;
                let l = angle.cos();
                let r = angle.sin();
                // Slight center boost compensation
                let compensation = 1.0 / (2.0f32).sqrt();
                (l / compensation, r / compensation)
            }
            PanLaw::Minus4_5dB => {
                let l = ((1.0 - p) * PI * 0.5).sin().powf(0.75);
                let r = (p * PI * 0.5).sin().powf(0.75);
                (l, r)
            }
        }
    }

    /// Advance the LFO by one sample.
    fn advance(&mut self) {
        self.phase += self.phase_inc;
        if self.phase >= 1.0 {
            self.phase -= 1.0;
        }
    }

    /// Process a single stereo sample pair.
    pub fn process_sample(&mut self, left: f32, right: f32) -> (f32, f32) {
        let lfo = self.lfo_value();
        let pan_pos = (self.config.center + lfo * self.config.depth).clamp(-1.0, 1.0);
        let (gain_l, gain_r) = self.compute_gains(pan_pos);

        self.advance();

        let mono = (left + right) * 0.5;
        (mono * gain_l, mono * gain_r)
    }

    /// Process a buffer of interleaved stereo samples in-place.
    pub fn process_stereo(&mut self, left: &mut [f32], right: &mut [f32]) {
        let len = left.len().min(right.len());
        for i in 0..len {
            let (l, r) = self.process_sample(left[i], right[i]);
            left[i] = l;
            right[i] = r;
        }
    }

    /// Process a mono buffer into stereo output buffers.
    pub fn process_mono_to_stereo(
        &mut self,
        input: &[f32],
        left_out: &mut [f32],
        right_out: &mut [f32],
    ) {
        let len = input.len().min(left_out.len()).min(right_out.len());
        for i in 0..len {
            let (l, r) = self.process_sample(input[i], input[i]);
            left_out[i] = l;
            right_out[i] = r;
        }
    }

    /// Get the current pan position (-1.0 to 1.0).
    #[must_use]
    pub fn current_position(&self) -> f32 {
        let lfo = self.lfo_value();
        (self.config.center + lfo * self.config.depth).clamp(-1.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = AutoPanConfig::default();
        assert!((cfg.rate_hz - 1.0).abs() < 1e-6);
        assert!((cfg.depth - 1.0).abs() < 1e-6);
        assert_eq!(cfg.waveform, PanWaveform::Sine);
        assert_eq!(cfg.pan_law, PanLaw::ConstantPower);
    }

    #[test]
    fn test_waveform_default() {
        assert_eq!(PanWaveform::default(), PanWaveform::Sine);
    }

    #[test]
    fn test_pan_law_default() {
        assert_eq!(PanLaw::default(), PanLaw::ConstantPower);
    }

    #[test]
    fn test_auto_pan_creation() {
        let pan = AutoPan::default_effect();
        assert!((pan.phase - 0.0).abs() < 1e-6);
        assert!(pan.phase_inc > 0.0);
    }

    #[test]
    fn test_set_rate() {
        let mut pan = AutoPan::default_effect();
        pan.set_rate(2.0);
        let expected_inc = 2.0 / 48000.0;
        assert!((pan.phase_inc - expected_inc).abs() < 1e-9);
    }

    #[test]
    fn test_set_depth() {
        let mut pan = AutoPan::default_effect();
        pan.set_depth(0.5);
        assert!((pan.config.depth - 0.5).abs() < 1e-6);
        pan.set_depth(2.0);
        assert!((pan.config.depth - 1.0).abs() < 1e-6);
        pan.set_depth(-1.0);
        assert!((pan.config.depth - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_lfo_sine_range() {
        let mut pan = AutoPan::new(AutoPanConfig {
            rate_hz: 100.0,
            sample_rate: 1000.0,
            ..Default::default()
        });
        for _ in 0..1000 {
            let pos = pan.current_position();
            assert!(pos >= -1.0 && pos <= 1.0);
            pan.advance();
        }
    }

    #[test]
    fn test_lfo_triangle() {
        let mut pan = AutoPan::new(AutoPanConfig {
            waveform: PanWaveform::Triangle,
            rate_hz: 100.0,
            sample_rate: 1000.0,
            ..Default::default()
        });
        for _ in 0..1000 {
            let pos = pan.current_position();
            assert!(pos >= -1.0 && pos <= 1.0);
            pan.advance();
        }
    }

    #[test]
    fn test_lfo_square() {
        let pan = AutoPan::new(AutoPanConfig {
            waveform: PanWaveform::Square,
            rate_hz: 100.0,
            sample_rate: 1000.0,
            ..Default::default()
        });
        // Square wave: only extreme positions
        let lfo = pan.lfo_value();
        assert!(lfo.abs() > 0.99);
    }

    #[test]
    fn test_constant_power_gains() {
        let pan = AutoPan::new(AutoPanConfig {
            pan_law: PanLaw::ConstantPower,
            ..Default::default()
        });
        // Center position: both gains should be approximately equal
        let (l, r) = pan.compute_gains(0.0);
        assert!((l - r).abs() < 0.01);
        // Sum of squares should be approximately 1 (constant power)
        let power = l * l + r * r;
        assert!((power - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_linear_gains_extremes() {
        let pan = AutoPan::new(AutoPanConfig {
            pan_law: PanLaw::Linear,
            ..Default::default()
        });
        let (l, r) = pan.compute_gains(-1.0);
        assert!((l - 1.0).abs() < 1e-6);
        assert!(r.abs() < 1e-6);

        let (l, r) = pan.compute_gains(1.0);
        assert!(l.abs() < 1e-6);
        assert!((r - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_process_mono_to_stereo() {
        let mut pan = AutoPan::new(AutoPanConfig {
            depth: 0.0,
            ..Default::default()
        });
        let input = vec![1.0f32; 100];
        let mut left = vec![0.0f32; 100];
        let mut right = vec![0.0f32; 100];
        pan.process_mono_to_stereo(&input, &mut left, &mut right);
        // With zero depth, gains should be equal
        for i in 0..100 {
            assert!((left[i] - right[i]).abs() < 0.01);
        }
    }

    #[test]
    fn test_process_stereo_in_place() {
        let mut pan = AutoPan::default_effect();
        let mut left = vec![0.5f32; 64];
        let mut right = vec![0.5f32; 64];
        pan.process_stereo(&mut left, &mut right);
        // Output should be modified (not all identical)
        let all_same = left.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-9);
        // With LFO running, values should vary
        assert!(!all_same || left[0].abs() < 1e-9);
    }

    #[test]
    fn test_reset_phase() {
        let mut pan = AutoPan::default_effect();
        for _ in 0..1000 {
            pan.advance();
        }
        assert!(pan.phase > 0.0);
        pan.reset();
        assert!((pan.phase - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_invert() {
        let pan_normal = AutoPan::new(AutoPanConfig {
            invert: false,
            ..Default::default()
        });
        let pan_invert = AutoPan::new(AutoPanConfig {
            invert: true,
            ..Default::default()
        });
        let v1 = pan_normal.lfo_value();
        let v2 = pan_invert.lfo_value();
        assert!((v1 + v2).abs() < 1e-6);
    }
}
