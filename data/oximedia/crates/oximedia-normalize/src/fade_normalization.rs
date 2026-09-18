#![allow(dead_code)]
//! Fade-in and fade-out normalization curves for smooth level transitions.
//!
//! This module provides various fade curve shapes (linear, logarithmic,
//! exponential, S-curve, equal power) and a fade processor that can apply
//! fade-in and fade-out envelopes to audio while respecting loudness targets.

use std::f64::consts::PI;

/// Available fade curve shapes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FadeCurve {
    /// Linear fade (constant rate of change).
    Linear,
    /// Logarithmic fade (fast start, slow finish for fade-in).
    Logarithmic,
    /// Exponential fade (slow start, fast finish for fade-in).
    Exponential,
    /// S-curve (smooth acceleration and deceleration).
    SCurve,
    /// Equal-power / cosine fade (preserves perceived loudness).
    EqualPower,
    /// Sine curve.
    Sine,
}

impl FadeCurve {
    /// Compute the gain value for a position along the curve.
    ///
    /// `t` ranges from 0.0 (start of fade) to 1.0 (end of fade).
    /// Returns a gain multiplier from 0.0 to 1.0.
    pub fn evaluate(&self, t: f64) -> f64 {
        let t = t.clamp(0.0, 1.0);
        match self {
            Self::Linear => t,
            Self::Logarithmic => {
                // log curve: fast attack
                if t <= 0.0 {
                    0.0
                } else {
                    (1.0 + 9.0 * t).log10()
                }
            }
            Self::Exponential => {
                // exponential: slow attack
                t * t
            }
            Self::SCurve => {
                // Hermite / smoothstep
                3.0 * t * t - 2.0 * t * t * t
            }
            Self::EqualPower => {
                // Cosine-based equal power
                (t * PI / 2.0).sin()
            }
            Self::Sine => {
                // Simple sine quarter-wave
                (t * PI / 2.0).sin()
            }
        }
    }

    /// Compute the inverse gain (for fade-out: 1.0 at start, 0.0 at end).
    pub fn evaluate_out(&self, t: f64) -> f64 {
        self.evaluate(1.0 - t)
    }
}

/// Direction of a fade operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FadeDirection {
    /// Fade in (silence to full level).
    In,
    /// Fade out (full level to silence).
    Out,
}

/// Configuration for a fade operation.
#[derive(Debug, Clone)]
pub struct FadeConfig {
    /// Duration of the fade in samples.
    pub duration_samples: usize,
    /// Curve shape.
    pub curve: FadeCurve,
    /// Direction.
    pub direction: FadeDirection,
    /// Starting gain (usually 0.0 for fade-in, 1.0 for fade-out).
    pub start_gain: f64,
    /// Ending gain (usually 1.0 for fade-in, 0.0 for fade-out).
    pub end_gain: f64,
}

impl FadeConfig {
    /// Create a fade-in configuration.
    pub fn fade_in(duration_samples: usize, curve: FadeCurve) -> Self {
        Self {
            duration_samples,
            curve,
            direction: FadeDirection::In,
            start_gain: 0.0,
            end_gain: 1.0,
        }
    }

    /// Create a fade-out configuration.
    pub fn fade_out(duration_samples: usize, curve: FadeCurve) -> Self {
        Self {
            duration_samples,
            curve,
            direction: FadeDirection::Out,
            start_gain: 1.0,
            end_gain: 0.0,
        }
    }

    /// Create a crossfade pair (fade-out + fade-in).
    pub fn crossfade(duration_samples: usize, curve: FadeCurve) -> (Self, Self) {
        (
            Self::fade_out(duration_samples, curve),
            Self::fade_in(duration_samples, curve),
        )
    }

    /// Get the gain at a specific sample position within the fade.
    pub fn gain_at_sample(&self, sample_index: usize) -> f64 {
        if self.duration_samples == 0 {
            return self.end_gain;
        }

        let t = sample_index as f64 / self.duration_samples as f64;
        let t = t.clamp(0.0, 1.0);

        let curve_val = self.curve.evaluate(t);

        self.start_gain + (self.end_gain - self.start_gain) * curve_val
    }
}

/// Generate a complete fade envelope as a vector of gain values.
pub fn generate_envelope(config: &FadeConfig) -> Vec<f64> {
    (0..config.duration_samples)
        .map(|i| config.gain_at_sample(i))
        .collect()
}

/// Apply a fade envelope to f32 audio samples in-place.
pub fn apply_fade_f32(samples: &mut [f32], config: &FadeConfig, offset: usize) {
    let end = (offset + config.duration_samples).min(samples.len());
    for i in offset..end {
        let gain = config.gain_at_sample(i - offset);
        samples[i] = (f64::from(samples[i]) * gain) as f32;
    }
}

/// Apply a fade envelope to f64 audio samples in-place.
pub fn apply_fade_f64(samples: &mut [f64], config: &FadeConfig, offset: usize) {
    let end = (offset + config.duration_samples).min(samples.len());
    for i in offset..end {
        let gain = config.gain_at_sample(i - offset);
        samples[i] *= gain;
    }
}

/// Fade processor that handles fade-in and fade-out for a stream of audio.
#[derive(Debug)]
pub struct FadeProcessor {
    /// Fade-in configuration (if any).
    fade_in: Option<FadeConfig>,
    /// Fade-out configuration (if any).
    fade_out: Option<FadeConfig>,
    /// Total number of samples in the stream.
    total_samples: usize,
    /// Current position in samples.
    position: usize,
}

impl FadeProcessor {
    /// Create a new fade processor.
    pub fn new(total_samples: usize) -> Self {
        Self {
            fade_in: None,
            fade_out: None,
            total_samples,
            position: 0,
        }
    }

    /// Set the fade-in configuration.
    pub fn set_fade_in(&mut self, duration_samples: usize, curve: FadeCurve) {
        self.fade_in = Some(FadeConfig::fade_in(duration_samples, curve));
    }

    /// Set the fade-out configuration.
    pub fn set_fade_out(&mut self, duration_samples: usize, curve: FadeCurve) {
        self.fade_out = Some(FadeConfig::fade_out(duration_samples, curve));
    }

    /// Get the current position in samples.
    pub fn position(&self) -> usize {
        self.position
    }

    /// Get total samples.
    pub fn total_samples(&self) -> usize {
        self.total_samples
    }

    /// Compute the gain at the current global position.
    pub fn gain_at_position(&self, pos: usize) -> f64 {
        let mut gain = 1.0;

        // Apply fade-in gain
        if let Some(ref fi) = self.fade_in {
            if pos < fi.duration_samples {
                gain *= fi.gain_at_sample(pos);
            }
        }

        // Apply fade-out gain
        if let Some(ref fo) = self.fade_out {
            if self.total_samples > 0
                && pos >= self.total_samples.saturating_sub(fo.duration_samples)
            {
                let fade_out_start = self.total_samples.saturating_sub(fo.duration_samples);
                let local_pos = pos - fade_out_start;
                gain *= fo.gain_at_sample(local_pos);
            }
        }

        gain
    }

    /// Process a block of f32 samples.
    pub fn process_f32(&mut self, samples: &mut [f32]) {
        for (i, sample) in samples.iter_mut().enumerate() {
            let global_pos = self.position + i;
            let gain = self.gain_at_position(global_pos);
            *sample = (f64::from(*sample) * gain) as f32;
        }
        self.position += samples.len();
    }

    /// Process a block of f64 samples.
    pub fn process_f64(&mut self, samples: &mut [f64]) {
        for (i, sample) in samples.iter_mut().enumerate() {
            let global_pos = self.position + i;
            let gain = self.gain_at_position(global_pos);
            *sample *= gain;
        }
        self.position += samples.len();
    }

    /// Reset the processor to the beginning.
    pub fn reset(&mut self) {
        self.position = 0;
    }
}

/// Compute the RMS level of a slice of f32 samples.
fn rms_f32(samples: &[f32]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: f64 = samples.iter().map(|&s| f64::from(s) * f64::from(s)).sum();
    (sum / samples.len() as f64).sqrt()
}

/// Compute the peak level of a slice of f32 samples.
fn peak_f32(samples: &[f32]) -> f32 {
    samples.iter().fold(0.0_f32, |acc, &s| acc.max(s.abs()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_fade_endpoints() {
        let curve = FadeCurve::Linear;
        assert!((curve.evaluate(0.0)).abs() < f64::EPSILON);
        assert!((curve.evaluate(1.0) - 1.0).abs() < f64::EPSILON);
        assert!((curve.evaluate(0.5) - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_scurve_endpoints() {
        let curve = FadeCurve::SCurve;
        assert!((curve.evaluate(0.0)).abs() < f64::EPSILON);
        assert!((curve.evaluate(1.0) - 1.0).abs() < f64::EPSILON);
        // S-curve at 0.5 should be exactly 0.5
        assert!((curve.evaluate(0.5) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_equal_power_endpoints() {
        let curve = FadeCurve::EqualPower;
        assert!((curve.evaluate(0.0)).abs() < f64::EPSILON);
        assert!((curve.evaluate(1.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_logarithmic_monotonic() {
        let curve = FadeCurve::Logarithmic;
        let mut prev = 0.0;
        for i in 0..=100 {
            let t = i as f64 / 100.0;
            let val = curve.evaluate(t);
            assert!(
                val >= prev - 1e-12,
                "Logarithmic curve is not monotonic at t={t}"
            );
            prev = val;
        }
    }

    #[test]
    fn test_exponential_monotonic() {
        let curve = FadeCurve::Exponential;
        let mut prev = 0.0;
        for i in 0..=100 {
            let t = i as f64 / 100.0;
            let val = curve.evaluate(t);
            assert!(
                val >= prev - 1e-12,
                "Exponential curve is not monotonic at t={t}"
            );
            prev = val;
        }
    }

    #[test]
    fn test_evaluate_out() {
        let curve = FadeCurve::Linear;
        assert!((curve.evaluate_out(0.0) - 1.0).abs() < f64::EPSILON);
        assert!((curve.evaluate_out(1.0)).abs() < f64::EPSILON);
        assert!((curve.evaluate_out(0.5) - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_fade_config_fade_in() {
        let config = FadeConfig::fade_in(1000, FadeCurve::Linear);
        assert_eq!(config.direction, FadeDirection::In);
        assert!((config.start_gain).abs() < f64::EPSILON);
        assert!((config.end_gain - 1.0).abs() < f64::EPSILON);
        // First sample should be ~0
        assert!(config.gain_at_sample(0) < 0.01);
        // Last sample should be ~1
        assert!(config.gain_at_sample(999) > 0.99);
    }

    #[test]
    fn test_fade_config_fade_out() {
        let config = FadeConfig::fade_out(1000, FadeCurve::Linear);
        assert_eq!(config.direction, FadeDirection::Out);
        // First sample should be ~1
        assert!(config.gain_at_sample(0) > 0.99);
        // Last sample should be ~0
        assert!(config.gain_at_sample(999) < 0.01);
    }

    #[test]
    fn test_crossfade_pair() {
        let (fo, fi) = FadeConfig::crossfade(100, FadeCurve::EqualPower);
        assert_eq!(fo.direction, FadeDirection::Out);
        assert_eq!(fi.direction, FadeDirection::In);
        assert_eq!(fo.duration_samples, 100);
        assert_eq!(fi.duration_samples, 100);
    }

    #[test]
    fn test_generate_envelope_length() {
        let config = FadeConfig::fade_in(256, FadeCurve::Linear);
        let env = generate_envelope(&config);
        assert_eq!(env.len(), 256);
    }

    #[test]
    fn test_apply_fade_f32() {
        let mut samples = vec![1.0_f32; 100];
        let config = FadeConfig::fade_in(100, FadeCurve::Linear);
        apply_fade_f32(&mut samples, &config, 0);
        // First sample should be near 0
        assert!(samples[0].abs() < 0.02);
        // Last sample should be near 1
        assert!((samples[99] - 1.0).abs() < 0.02);
    }

    #[test]
    fn test_apply_fade_f64() {
        let mut samples = vec![1.0_f64; 100];
        let config = FadeConfig::fade_out(100, FadeCurve::Linear);
        apply_fade_f64(&mut samples, &config, 0);
        // First sample should be near 1
        assert!((samples[0] - 1.0).abs() < 0.02);
        // Last sample should be near 0
        assert!(samples[99].abs() < 0.02);
    }

    #[test]
    fn test_fade_processor_creation() {
        let proc = FadeProcessor::new(48000);
        assert_eq!(proc.position(), 0);
        assert_eq!(proc.total_samples(), 48000);
    }

    #[test]
    fn test_fade_processor_with_fades() {
        let mut proc = FadeProcessor::new(10000);
        proc.set_fade_in(1000, FadeCurve::Linear);
        proc.set_fade_out(1000, FadeCurve::Linear);

        // Position 0 (fade-in start): gain should be ~0
        assert!(proc.gain_at_position(0) < 0.01);
        // Position 500 (mid fade-in)
        assert!(proc.gain_at_position(500) > 0.3);
        assert!(proc.gain_at_position(500) < 0.7);
        // Position 5000 (middle, no fade): gain should be 1.0
        assert!((proc.gain_at_position(5000) - 1.0).abs() < f64::EPSILON);
        // Position 9999 (fade-out end): gain should be ~0
        assert!(proc.gain_at_position(9999) < 0.01);
    }

    #[test]
    fn test_fade_processor_process_f32() {
        let mut proc = FadeProcessor::new(200);
        proc.set_fade_in(100, FadeCurve::Linear);

        let mut samples = vec![1.0_f32; 200];
        proc.process_f32(&mut samples);

        assert!(samples[0].abs() < 0.02);
        assert!((samples[150] - 1.0).abs() < 0.02);
        assert_eq!(proc.position(), 200);
    }

    #[test]
    fn test_fade_processor_reset() {
        let mut proc = FadeProcessor::new(1000);
        let mut buf = vec![1.0_f32; 100];
        proc.process_f32(&mut buf);
        assert_eq!(proc.position(), 100);
        proc.reset();
        assert_eq!(proc.position(), 0);
    }

    #[test]
    fn test_rms_f32_silence() {
        let samples = vec![0.0_f32; 100];
        assert!(rms_f32(&samples).abs() < f64::EPSILON);
    }

    #[test]
    fn test_peak_f32() {
        let samples = vec![0.1_f32, -0.5, 0.3, -0.9, 0.2];
        assert!((peak_f32(&samples) - 0.9).abs() < f32::EPSILON);
    }
}
