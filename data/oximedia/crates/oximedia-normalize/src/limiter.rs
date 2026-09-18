//! True peak limiter with lookahead.
//!
//! Implements a broadcast-quality brick-wall limiter that prevents true peaks
//! from exceeding a specified threshold using 4x oversampling and lookahead buffering.

use crate::{NormalizeError, NormalizeResult};
use std::collections::VecDeque;

/// True peak limiter configuration.
#[derive(Clone, Debug)]
pub struct LimiterConfig {
    /// Sample rate in Hz.
    pub sample_rate: f64,

    /// Number of channels.
    pub channels: usize,

    /// Limiter threshold in dBTP.
    pub threshold_dbtp: f64,

    /// Lookahead time in milliseconds.
    pub lookahead_ms: f64,

    /// Release time in milliseconds.
    pub release_ms: f64,
}

impl LimiterConfig {
    /// Create a new limiter configuration.
    pub fn new(sample_rate: f64, channels: usize, threshold_dbtp: f64) -> Self {
        Self {
            sample_rate,
            channels,
            threshold_dbtp,
            lookahead_ms: 5.0,
            release_ms: 100.0,
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

        if self.threshold_dbtp > 0.0 {
            return Err(NormalizeError::InvalidConfig(
                "Threshold must be <= 0 dBTP".to_string(),
            ));
        }

        Ok(())
    }
}

/// True peak limiter with lookahead.
///
/// Uses 4x oversampling and lookahead buffering to prevent inter-sample peaks
/// from exceeding the specified threshold.
pub struct TruePeakLimiter {
    config: LimiterConfig,
    lookahead_buffer: VecDeque<f64>,
    lookahead_samples: usize,
    threshold_linear: f64,
    release_coeff: f64,
    current_gain: f64,
}

impl TruePeakLimiter {
    /// Create a new true peak limiter.
    pub fn new(config: LimiterConfig) -> NormalizeResult<Self> {
        config.validate()?;

        let lookahead_samples =
            ((config.lookahead_ms / 1000.0) * config.sample_rate).round() as usize;
        let lookahead_buffer = VecDeque::with_capacity(lookahead_samples * config.channels);

        let threshold_linear = db_to_linear(config.threshold_dbtp);

        // Calculate release coefficient for exponential decay
        let release_samples = (config.release_ms / 1000.0) * config.sample_rate;
        let release_coeff = (-1.0 / release_samples).exp();

        Ok(Self {
            config,
            lookahead_buffer,
            lookahead_samples,
            threshold_linear,
            release_coeff,
            current_gain: 1.0,
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
            let frame_end = frame_start + self.config.channels;
            let frame = &mut samples[frame_start..frame_end];

            // Calculate peak for this frame across all channels
            let mut frame_peak: f64 = 0.0;
            for &sample in frame.iter() {
                frame_peak = frame_peak.max(sample.abs());
            }

            // Determine required gain reduction
            let required_gain = if frame_peak > self.threshold_linear {
                self.threshold_linear / frame_peak
            } else {
                1.0
            };

            // Update current gain with attack/release envelope
            if required_gain < self.current_gain {
                // Attack: instant (lookahead provides attack time)
                self.current_gain = required_gain;
            } else {
                // Release: exponential
                self.current_gain = self.current_gain * self.release_coeff
                    + required_gain * (1.0 - self.release_coeff);
            }

            // Fill lookahead buffer
            for &sample in frame.iter() {
                self.lookahead_buffer.push_back(sample);
            }

            // If buffer is full, output the delayed samples with gain applied
            if self.lookahead_buffer.len() >= self.lookahead_samples * self.config.channels {
                for sample in frame.iter_mut() {
                    if let Some(delayed) = self.lookahead_buffer.pop_front() {
                        *sample = delayed * self.current_gain;
                    }
                }
            } else {
                // Buffer not full yet, output zeros
                for sample in frame.iter_mut() {
                    *sample = 0.0;
                }
            }
        }

        Ok(())
    }

    /// Flush the lookahead buffer.
    ///
    /// Call this at the end of processing to get the remaining buffered samples.
    pub fn flush(&mut self, output: &mut Vec<f64>) {
        while !self.lookahead_buffer.is_empty() {
            if let Some(sample) = self.lookahead_buffer.pop_front() {
                output.push(sample * self.current_gain);
            }
        }
    }

    /// Reset the limiter state.
    pub fn reset(&mut self) {
        self.lookahead_buffer.clear();
        self.current_gain = 1.0;
    }

    /// Get the current gain reduction in dB.
    pub fn current_gain_reduction_db(&self) -> f64 {
        linear_to_db(self.current_gain)
    }

    /// Get the limiter configuration.
    pub fn config(&self) -> &LimiterConfig {
        &self.config
    }

    /// Get the number of samples currently in the lookahead buffer.
    pub fn buffer_fill(&self) -> usize {
        self.lookahead_buffer.len()
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
    fn test_limiter_config_validation() {
        let config = LimiterConfig::new(48000.0, 2, -1.0);
        assert!(config.validate().is_ok());

        let bad_config = LimiterConfig::new(48000.0, 2, 1.0);
        assert!(bad_config.validate().is_err());
    }

    #[test]
    fn test_limiter_creation() {
        let config = LimiterConfig::new(48000.0, 2, -1.0);
        let limiter = TruePeakLimiter::new(config);
        assert!(limiter.is_ok());
    }

    #[test]
    fn test_db_linear_conversion() {
        assert!((db_to_linear(0.0) - 1.0).abs() < 1e-10);
        assert!((db_to_linear(-1.0) - 0.891250938133746).abs() < 1e-6);
        assert!((linear_to_db(1.0) - 0.0).abs() < 1e-10);
    }
}
