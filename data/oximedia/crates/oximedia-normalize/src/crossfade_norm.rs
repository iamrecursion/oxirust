#![allow(dead_code)]
//! Crossfade-aware normalization between audio segments.
//!
//! When normalizing audio that contains crossfades between segments of
//! different loudness, naive gain application can produce audible artifacts.
//! This module provides crossfade-aware normalization that smoothly
//! interpolates gain values across transition regions.

use std::collections::VecDeque;

/// Shape of the crossfade curve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrossfadeShape {
    /// Linear crossfade.
    Linear,
    /// Equal-power (sinusoidal) crossfade.
    EqualPower,
    /// S-curve (smooth) crossfade.
    SCurve,
    /// Logarithmic crossfade.
    Logarithmic,
    /// Exponential crossfade.
    Exponential,
}

/// A segment with measured loudness for crossfade normalization.
#[derive(Debug, Clone)]
pub struct NormSegment {
    /// Start sample of the segment.
    pub start_sample: usize,
    /// End sample (exclusive) of the segment.
    pub end_sample: usize,
    /// Measured integrated loudness in LUFS.
    pub loudness_lufs: f64,
    /// Target loudness in LUFS.
    pub target_lufs: f64,
    /// Computed gain in linear scale.
    pub gain_linear: f64,
}

impl NormSegment {
    /// Create a new normalization segment.
    pub fn new(start: usize, end: usize, loudness_lufs: f64, target_lufs: f64) -> Self {
        let gain_db = target_lufs - loudness_lufs;
        let gain_linear = 10.0_f64.powf(gain_db / 20.0);
        Self {
            start_sample: start,
            end_sample: end,
            loudness_lufs,
            target_lufs,
            gain_linear,
        }
    }

    /// Get the gain in decibels.
    pub fn gain_db(&self) -> f64 {
        self.target_lufs - self.loudness_lufs
    }

    /// Get the segment duration in samples.
    pub fn duration_samples(&self) -> usize {
        self.end_sample.saturating_sub(self.start_sample)
    }
}

/// Configuration for crossfade normalization.
#[derive(Debug, Clone)]
pub struct CrossfadeNormConfig {
    /// Crossfade duration in samples.
    pub crossfade_samples: usize,
    /// Crossfade curve shape.
    pub shape: CrossfadeShape,
    /// Target loudness in LUFS.
    pub target_lufs: f64,
    /// Maximum gain change rate in dB/second (to prevent clicks).
    pub max_gain_rate_db_per_sec: f64,
    /// Sample rate in Hz.
    pub sample_rate: f64,
    /// Number of audio channels.
    pub channels: usize,
}

impl CrossfadeNormConfig {
    /// Create a new crossfade normalization configuration.
    pub fn new(sample_rate: f64, channels: usize) -> Self {
        Self {
            crossfade_samples: (sample_rate * 0.01) as usize, // 10ms default
            shape: CrossfadeShape::EqualPower,
            target_lufs: -23.0,
            max_gain_rate_db_per_sec: 40.0,
            sample_rate,
            channels,
        }
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.sample_rate < 8000.0 || self.sample_rate > 192_000.0 {
            return Err(format!("Invalid sample rate: {}", self.sample_rate));
        }
        if self.channels == 0 || self.channels > 16 {
            return Err(format!("Invalid channel count: {}", self.channels));
        }
        if self.crossfade_samples == 0 {
            return Err("Crossfade samples must be > 0".to_string());
        }
        if self.max_gain_rate_db_per_sec <= 0.0 {
            return Err("Max gain rate must be positive".to_string());
        }
        Ok(())
    }
}

/// Compute a crossfade coefficient for position `t` in range [0.0, 1.0].
pub fn crossfade_coefficient(t: f64, shape: CrossfadeShape) -> f64 {
    let t = t.clamp(0.0, 1.0);
    match shape {
        CrossfadeShape::Linear => t,
        CrossfadeShape::EqualPower => (t * std::f64::consts::FRAC_PI_2).sin(),
        CrossfadeShape::SCurve => {
            // Hermite interpolation: 3t^2 - 2t^3
            t * t * (3.0 - 2.0 * t)
        }
        CrossfadeShape::Logarithmic => {
            if t <= 0.0 {
                0.0
            } else {
                (1.0 + (t * 9.0 + 1.0).ln() / 10.0_f64.ln()).min(1.0)
            }
        }
        CrossfadeShape::Exponential => (10.0_f64.powf(t) - 1.0) / 9.0,
    }
}

/// Crossfade normalization processor.
///
/// Manages gain transitions between segments, ensuring smooth crossfades
/// that avoid artifacts at segment boundaries.
#[derive(Debug)]
pub struct CrossfadeNormalizer {
    /// Configuration.
    config: CrossfadeNormConfig,
    /// Segments to process.
    segments: Vec<NormSegment>,
    /// Gain schedule: per-sample gain values for the crossfade region.
    gain_schedule: VecDeque<f64>,
    /// Current sample position.
    current_sample: usize,
    /// Current gain value.
    current_gain: f64,
    /// Previous segment gain (for interpolation).
    prev_gain: f64,
}

impl CrossfadeNormalizer {
    /// Create a new crossfade normalizer.
    pub fn new(config: CrossfadeNormConfig) -> Self {
        Self {
            config,
            segments: Vec::new(),
            gain_schedule: VecDeque::new(),
            current_sample: 0,
            current_gain: 1.0,
            prev_gain: 1.0,
        }
    }

    /// Add a segment for normalization.
    pub fn add_segment(&mut self, segment: NormSegment) {
        self.segments.push(segment);
    }

    /// Get the number of segments.
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Build the gain schedule for all segments.
    pub fn build_schedule(&mut self) {
        self.gain_schedule.clear();
        if self.segments.is_empty() {
            return;
        }

        // Sort segments by start sample
        self.segments.sort_by_key(|s| s.start_sample);

        let total_samples = self.segments.last().map(|s| s.end_sample).unwrap_or(0);

        // Build per-sample gain values
        let cf_len = self.config.crossfade_samples;
        let mut prev_gain = self.segments[0].gain_linear;

        for seg_idx in 0..self.segments.len() {
            let seg = &self.segments[seg_idx];
            let seg_gain = seg.gain_linear;

            if seg_idx == 0 {
                // First segment: no crossfade at start
                let body_len = if seg.duration_samples() > cf_len {
                    seg.duration_samples() - cf_len
                } else {
                    seg.duration_samples()
                };
                for _ in 0..body_len {
                    self.gain_schedule.push_back(seg_gain);
                }
            }

            if seg_idx > 0 {
                // Crossfade from previous segment
                let actual_cf = cf_len.min(seg.duration_samples());
                for i in 0..actual_cf {
                    let t = if actual_cf > 1 {
                        i as f64 / (actual_cf - 1) as f64
                    } else {
                        1.0
                    };
                    let coeff = crossfade_coefficient(t, self.config.shape);
                    let gain = prev_gain * (1.0 - coeff) + seg_gain * coeff;
                    self.gain_schedule.push_back(gain);
                }

                // Remaining body samples
                let body_len = seg.duration_samples().saturating_sub(actual_cf);
                let next_cf = if seg_idx + 1 < self.segments.len() {
                    cf_len.min(body_len)
                } else {
                    0
                };
                for _ in 0..body_len.saturating_sub(next_cf) {
                    self.gain_schedule.push_back(seg_gain);
                }
            }

            prev_gain = seg_gain;
        }

        let _ = total_samples; // used for sizing reference
    }

    /// Get the gain value for the current sample position.
    pub fn get_gain(&self, sample_idx: usize) -> f64 {
        self.gain_schedule
            .get(sample_idx)
            .copied()
            .unwrap_or(self.current_gain)
    }

    /// Apply crossfade normalization to a buffer of samples.
    pub fn process(&mut self, samples: &mut [f32]) {
        for sample in samples.iter_mut() {
            let gain = self
                .gain_schedule
                .get(self.current_sample)
                .copied()
                .unwrap_or(self.current_gain);
            *sample *= gain as f32;
            self.current_sample += 1;
        }
    }

    /// Reset the processor.
    pub fn reset(&mut self) {
        self.segments.clear();
        self.gain_schedule.clear();
        self.current_sample = 0;
        self.current_gain = 1.0;
        self.prev_gain = 1.0;
    }

    /// Get the current sample position.
    pub fn position(&self) -> usize {
        self.current_sample
    }

    /// Get the total length of the gain schedule.
    pub fn schedule_len(&self) -> usize {
        self.gain_schedule.len()
    }
}

/// Interpolate gain between two values over a given number of samples.
pub fn interpolate_gain(start_gain_db: f64, end_gain_db: f64, num_samples: usize) -> Vec<f64> {
    if num_samples == 0 {
        return Vec::new();
    }
    if num_samples == 1 {
        return vec![10.0_f64.powf(end_gain_db / 20.0)];
    }
    (0..num_samples)
        .map(|i| {
            let t = i as f64 / (num_samples - 1) as f64;
            let db = start_gain_db + t * (end_gain_db - start_gain_db);
            10.0_f64.powf(db / 20.0)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crossfade_shape_linear() {
        assert!((crossfade_coefficient(0.0, CrossfadeShape::Linear) - 0.0).abs() < 1e-10);
        assert!((crossfade_coefficient(0.5, CrossfadeShape::Linear) - 0.5).abs() < 1e-10);
        assert!((crossfade_coefficient(1.0, CrossfadeShape::Linear) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_crossfade_shape_equal_power() {
        let c = crossfade_coefficient(0.5, CrossfadeShape::EqualPower);
        // sin(pi/4) = sqrt(2)/2 ~ 0.7071
        assert!((c - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-10);
    }

    #[test]
    fn test_crossfade_shape_scurve() {
        assert!((crossfade_coefficient(0.0, CrossfadeShape::SCurve) - 0.0).abs() < 1e-10);
        assert!((crossfade_coefficient(0.5, CrossfadeShape::SCurve) - 0.5).abs() < 1e-10);
        assert!((crossfade_coefficient(1.0, CrossfadeShape::SCurve) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_crossfade_shape_exponential() {
        let c0 = crossfade_coefficient(0.0, CrossfadeShape::Exponential);
        let c1 = crossfade_coefficient(1.0, CrossfadeShape::Exponential);
        assert!((c0 - 0.0).abs() < 1e-10);
        assert!((c1 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_crossfade_clamping() {
        let c_neg = crossfade_coefficient(-0.5, CrossfadeShape::Linear);
        let c_over = crossfade_coefficient(1.5, CrossfadeShape::Linear);
        assert!((c_neg - 0.0).abs() < 1e-10);
        assert!((c_over - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_norm_segment_creation() {
        let seg = NormSegment::new(0, 48000, -20.0, -23.0);
        assert_eq!(seg.start_sample, 0);
        assert_eq!(seg.end_sample, 48000);
        assert!((seg.gain_db() - (-3.0)).abs() < 1e-10);
        assert!(seg.gain_linear < 1.0); // Gain should reduce
    }

    #[test]
    fn test_norm_segment_duration() {
        let seg = NormSegment::new(1000, 5000, -23.0, -23.0);
        assert_eq!(seg.duration_samples(), 4000);
    }

    #[test]
    fn test_config_validation() {
        let config = CrossfadeNormConfig::new(48000.0, 2);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validation_bad_rate() {
        let mut config = CrossfadeNormConfig::new(48000.0, 2);
        config.sample_rate = 0.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_normalizer_add_segments() {
        let config = CrossfadeNormConfig::new(48000.0, 1);
        let mut norm = CrossfadeNormalizer::new(config);
        norm.add_segment(NormSegment::new(0, 48000, -20.0, -23.0));
        norm.add_segment(NormSegment::new(48000, 96000, -30.0, -23.0));
        assert_eq!(norm.segment_count(), 2);
    }

    #[test]
    fn test_normalizer_build_schedule() {
        let config = CrossfadeNormConfig::new(48000.0, 1);
        let mut norm = CrossfadeNormalizer::new(config);
        norm.add_segment(NormSegment::new(0, 1000, -20.0, -23.0));
        norm.add_segment(NormSegment::new(1000, 2000, -30.0, -23.0));
        norm.build_schedule();
        assert!(norm.schedule_len() > 0);
    }

    #[test]
    fn test_normalizer_process() {
        let config = CrossfadeNormConfig::new(48000.0, 1);
        let mut norm = CrossfadeNormalizer::new(config);
        norm.add_segment(NormSegment::new(0, 100, -23.0, -23.0));
        norm.build_schedule();
        let mut buf = vec![1.0f32; 50];
        norm.process(&mut buf);
        assert_eq!(norm.position(), 50);
    }

    #[test]
    fn test_normalizer_reset() {
        let config = CrossfadeNormConfig::new(48000.0, 1);
        let mut norm = CrossfadeNormalizer::new(config);
        norm.add_segment(NormSegment::new(0, 100, -20.0, -23.0));
        norm.build_schedule();
        norm.reset();
        assert_eq!(norm.segment_count(), 0);
        assert_eq!(norm.schedule_len(), 0);
        assert_eq!(norm.position(), 0);
    }

    #[test]
    fn test_interpolate_gain() {
        let gains = interpolate_gain(0.0, 0.0, 10);
        assert_eq!(gains.len(), 10);
        // 0 dB => gain = 1.0
        for g in &gains {
            assert!((g - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_interpolate_gain_empty() {
        let gains = interpolate_gain(0.0, 6.0, 0);
        assert!(gains.is_empty());
    }

    #[test]
    fn test_interpolate_gain_single() {
        let gains = interpolate_gain(0.0, 6.0, 1);
        assert_eq!(gains.len(), 1);
        // Should be end gain: 10^(6/20) ~ 1.995
        assert!((gains[0] - 10.0_f64.powf(6.0 / 20.0)).abs() < 1e-6);
    }
}
