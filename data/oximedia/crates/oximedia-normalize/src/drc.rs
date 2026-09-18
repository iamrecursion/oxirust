//! Dynamic range compressor for broadcast.
//!
//! Implements a professional broadcast-quality compressor with adjustable
//! threshold, ratio, attack, release, and knee.

use crate::{NormalizeError, NormalizeResult};

/// Dynamic range compressor configuration.
#[derive(Clone, Debug)]
pub struct DrcConfig {
    /// Sample rate in Hz.
    pub sample_rate: f64,

    /// Number of channels.
    pub channels: usize,

    /// Threshold in dB.
    pub threshold_db: f64,

    /// Compression ratio (e.g., 3.0 for 3:1).
    pub ratio: f64,

    /// Attack time in milliseconds.
    pub attack_ms: f64,

    /// Release time in milliseconds.
    pub release_ms: f64,

    /// Knee width in dB (0 = hard knee).
    pub knee_db: f64,

    /// Makeup gain in dB.
    pub makeup_gain_db: f64,
}

impl DrcConfig {
    /// Create a new DRC configuration with broadcast defaults.
    pub fn new(sample_rate: f64, channels: usize) -> Self {
        Self {
            sample_rate,
            channels,
            threshold_db: -20.0,
            ratio: 3.0,
            attack_ms: 5.0,
            release_ms: 100.0,
            knee_db: 6.0,
            makeup_gain_db: 0.0,
        }
    }

    /// Create a gentle DRC configuration.
    pub fn gentle(sample_rate: f64, channels: usize) -> Self {
        Self {
            sample_rate,
            channels,
            threshold_db: -15.0,
            ratio: 2.0,
            attack_ms: 10.0,
            release_ms: 200.0,
            knee_db: 10.0,
            makeup_gain_db: 0.0,
        }
    }

    /// Create an aggressive DRC configuration.
    pub fn aggressive(sample_rate: f64, channels: usize) -> Self {
        Self {
            sample_rate,
            channels,
            threshold_db: -25.0,
            ratio: 6.0,
            attack_ms: 1.0,
            release_ms: 50.0,
            knee_db: 3.0,
            makeup_gain_db: 3.0,
        }
    }

    /// Validate the configuration.
    pub fn validate(&self) -> NormalizeResult<()> {
        if self.sample_rate < 8000.0 || self.sample_rate > 192_000.0 {
            return Err(NormalizeError::InvalidConfig(
                "Sample rate out of range".to_string(),
            ));
        }

        if self.channels == 0 || self.channels > 16 {
            return Err(NormalizeError::InvalidConfig(
                "Channel count out of range".to_string(),
            ));
        }

        if self.ratio < 1.0 || self.ratio > 100.0 {
            return Err(NormalizeError::InvalidConfig(format!(
                "Ratio {} is out of valid range (1.0-100.0)",
                self.ratio
            )));
        }

        if self.knee_db < 0.0 {
            return Err(NormalizeError::InvalidConfig(
                "Knee width must be >= 0".to_string(),
            ));
        }

        Ok(())
    }
}

/// Dynamic range compressor.
///
/// Reduces the dynamic range of audio by attenuating signals above a threshold.
pub struct DynamicRangeCompressor {
    config: DrcConfig,
    threshold_linear: f64,
    attack_coeff: f64,
    release_coeff: f64,
    makeup_gain_linear: f64,
    envelope: Vec<f64>,
}

impl DynamicRangeCompressor {
    /// Create a new dynamic range compressor.
    pub fn new(config: DrcConfig) -> NormalizeResult<Self> {
        config.validate()?;

        let threshold_linear = db_to_linear(config.threshold_db);

        // Calculate attack/release coefficients
        let attack_samples = (config.attack_ms / 1000.0) * config.sample_rate;
        let attack_coeff = (-1.0 / attack_samples).exp();

        let release_samples = (config.release_ms / 1000.0) * config.sample_rate;
        let release_coeff = (-1.0 / release_samples).exp();

        let makeup_gain_linear = db_to_linear(config.makeup_gain_db);

        let envelope = vec![0.0; config.channels];

        Ok(Self {
            config,
            threshold_linear,
            attack_coeff,
            release_coeff,
            makeup_gain_linear,
            envelope,
        })
    }

    /// Process f32 audio in-place.
    pub fn process_f32_inplace(&mut self, samples: &mut [f32]) -> NormalizeResult<()> {
        // Convert to f64, process, convert back
        let mut f64_samples: Vec<f64> = samples.iter().map(|&s| f64::from(s)).collect();
        self.process_f64_inplace(&mut f64_samples)?;
        for (i, &sample) in f64_samples.iter().enumerate() {
            samples[i] = sample as f32;
        }
        Ok(())
    }

    /// Process f64 audio in-place.
    pub fn process_f64_inplace(&mut self, samples: &mut [f64]) -> NormalizeResult<()> {
        if samples.len() % self.config.channels != 0 {
            return Err(NormalizeError::ProcessingError(
                "Sample count must be multiple of channel count".to_string(),
            ));
        }

        let frame_count = samples.len() / self.config.channels;

        for frame_idx in 0..frame_count {
            let frame_start = frame_idx * self.config.channels;
            let _frame_end = frame_start + self.config.channels;

            for ch in 0..self.config.channels {
                let sample_idx = frame_start + ch;
                let sample = samples[sample_idx];
                let sample_abs = sample.abs();

                // Update envelope follower
                let target = sample_abs;
                if target > self.envelope[ch] {
                    // Attack
                    self.envelope[ch] =
                        self.attack_coeff * self.envelope[ch] + (1.0 - self.attack_coeff) * target;
                } else {
                    // Release
                    self.envelope[ch] = self.release_coeff * self.envelope[ch]
                        + (1.0 - self.release_coeff) * target;
                }

                // Calculate gain reduction
                let gain = self.compute_gain(self.envelope[ch]);

                // Apply compression and makeup gain
                samples[sample_idx] = sample * gain * self.makeup_gain_linear;
            }
        }

        Ok(())
    }

    /// Compute compression gain for a given input level.
    fn compute_gain(&self, input_linear: f64) -> f64 {
        if input_linear <= 0.0 {
            return 1.0;
        }

        let input_db = linear_to_db(input_linear);
        let threshold_db = self.config.threshold_db;
        let knee_half = self.config.knee_db / 2.0;

        let gain_db = if self.config.knee_db > 0.0 {
            // Soft knee compression
            if input_db < threshold_db - knee_half {
                // Below knee - no compression
                0.0
            } else if input_db < threshold_db + knee_half {
                // In knee region - smooth transition
                let delta = input_db - (threshold_db - knee_half);
                let knee_ratio = delta / self.config.knee_db;
                let excess = delta;
                -(excess * knee_ratio * (1.0 - 1.0 / self.config.ratio))
            } else {
                // Above knee - full compression
                let excess = input_db - threshold_db;
                -(excess * (1.0 - 1.0 / self.config.ratio))
            }
        } else {
            // Hard knee compression
            if input_db <= threshold_db {
                0.0
            } else {
                let excess = input_db - threshold_db;
                -(excess * (1.0 - 1.0 / self.config.ratio))
            }
        };

        db_to_linear(gain_db)
    }

    /// Reset the compressor state.
    pub fn reset(&mut self) {
        self.envelope.fill(0.0);
    }

    /// Get the current envelope values for each channel.
    pub fn envelope(&self) -> &[f64] {
        &self.envelope
    }

    /// Get the compressor configuration.
    pub fn config(&self) -> &DrcConfig {
        &self.config
    }

    /// Calculate current gain reduction in dB for a channel.
    pub fn gain_reduction_db(&self, channel: usize) -> f64 {
        if channel >= self.config.channels {
            return 0.0;
        }

        let gain = self.compute_gain(self.envelope[channel]);
        linear_to_db(gain)
    }
}

/// Convert dB to linear gain.
#[inline]
fn db_to_linear(db: f64) -> f64 {
    10.0_f64.powf(db / 20.0)
}

/// Convert linear gain to dB.
#[inline]
fn linear_to_db(linear: f64) -> f64 {
    20.0 * linear.log10()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_drc_config_validation() {
        let config = DrcConfig::new(48000.0, 2);
        assert!(config.validate().is_ok());

        let bad_config = DrcConfig {
            ratio: 0.5, // Invalid ratio
            ..config
        };
        assert!(bad_config.validate().is_err());
    }

    #[test]
    fn test_drc_creation() {
        let config = DrcConfig::new(48000.0, 2);
        let drc = DynamicRangeCompressor::new(config);
        assert!(drc.is_ok());
    }

    #[test]
    fn test_drc_presets() {
        let gentle = DrcConfig::gentle(48000.0, 2);
        assert_eq!(gentle.ratio, 2.0);

        let aggressive = DrcConfig::aggressive(48000.0, 2);
        assert_eq!(aggressive.ratio, 6.0);
    }

    #[test]
    fn test_gain_computation() {
        let config = DrcConfig::new(48000.0, 2);
        let drc = DynamicRangeCompressor::new(config).expect("should succeed in test");

        // Test that signal below threshold has no gain reduction
        let low_level = db_to_linear(-30.0);
        let gain = drc.compute_gain(low_level);
        assert!((gain - 1.0).abs() < 0.1);
    }
}
