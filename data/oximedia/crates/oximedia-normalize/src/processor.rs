//! Loudness normalization processor.
//!
//! This module implements the core normalization processing, applying gain
//! adjustments with optional limiting and dynamic range compression.

use crate::{
    analyzer::db_to_linear, DrcConfig, DynamicRangeCompressor, LimiterConfig, NormalizeError,
    NormalizeResult, TruePeakLimiter,
};

/// Normalization processor configuration.
#[derive(Clone, Debug)]
#[allow(clippy::struct_excessive_bools)]
pub struct ProcessorConfig {
    /// Sample rate in Hz.
    pub sample_rate: f64,

    /// Number of channels.
    pub channels: usize,

    /// Enable true peak limiter.
    pub enable_limiter: bool,

    /// Enable dynamic range compression.
    pub enable_drc: bool,

    /// Lookahead time in milliseconds.
    pub lookahead_ms: f64,
}

impl ProcessorConfig {
    /// Create a new processor configuration.
    pub fn new(sample_rate: f64, channels: usize) -> Self {
        Self {
            sample_rate,
            channels,
            enable_limiter: true,
            enable_drc: false,
            lookahead_ms: 5.0,
        }
    }

    /// Create a minimal configuration (gain only, no processing).
    pub fn minimal(sample_rate: f64, channels: usize) -> Self {
        Self {
            sample_rate,
            channels,
            enable_limiter: false,
            enable_drc: false,
            lookahead_ms: 0.0,
        }
    }

    /// Validate the configuration.
    pub fn validate(&self) -> NormalizeResult<()> {
        if self.sample_rate < 8000.0 || self.sample_rate > 192_000.0 {
            return Err(NormalizeError::InvalidConfig(format!(
                "Sample rate {} Hz is out of valid range",
                self.sample_rate
            )));
        }

        if self.channels == 0 || self.channels > 16 {
            return Err(NormalizeError::InvalidConfig(format!(
                "Channel count {} is out of valid range",
                self.channels
            )));
        }

        Ok(())
    }
}

/// Normalization processor.
///
/// Applies gain adjustments with optional limiting and dynamic range compression.
pub struct NormalizationProcessor {
    config: ProcessorConfig,
    limiter: Option<TruePeakLimiter>,
    drc: Option<DynamicRangeCompressor>,
}

impl NormalizationProcessor {
    /// Create a new normalization processor.
    pub fn new(config: ProcessorConfig) -> NormalizeResult<Self> {
        config.validate()?;

        let limiter = if config.enable_limiter {
            let limiter_config = LimiterConfig {
                sample_rate: config.sample_rate,
                channels: config.channels,
                threshold_dbtp: -1.0,
                lookahead_ms: config.lookahead_ms,
                release_ms: 100.0,
            };
            Some(TruePeakLimiter::new(limiter_config)?)
        } else {
            None
        };

        let drc = if config.enable_drc {
            let drc_config = DrcConfig {
                sample_rate: config.sample_rate,
                channels: config.channels,
                threshold_db: -20.0,
                ratio: 3.0,
                attack_ms: 5.0,
                release_ms: 100.0,
                knee_db: 6.0,
                makeup_gain_db: 0.0,
            };
            Some(DynamicRangeCompressor::new(drc_config)?)
        } else {
            None
        };

        Ok(Self {
            config,
            limiter,
            drc,
        })
    }

    /// Process f32 audio with gain adjustment.
    pub fn process_f32(
        &mut self,
        input: &[f32],
        output: &mut [f32],
        gain_db: f64,
    ) -> NormalizeResult<()> {
        if input.len() != output.len() {
            return Err(NormalizeError::ProcessingError(
                "Input and output buffers must have same length".to_string(),
            ));
        }

        let gain = db_to_linear(gain_db) as f32;

        // Apply gain
        for (i, &sample) in input.iter().enumerate() {
            output[i] = sample * gain;
        }

        // Apply DRC if enabled
        if let Some(ref mut drc) = self.drc {
            drc.process_f32_inplace(output)?;
        }

        // Apply limiting if enabled
        if let Some(ref mut limiter) = self.limiter {
            limiter.process_f32_inplace(output)?;
        }

        Ok(())
    }

    /// Process f64 audio with gain adjustment.
    pub fn process_f64(
        &mut self,
        input: &[f64],
        output: &mut [f64],
        gain_db: f64,
    ) -> NormalizeResult<()> {
        if input.len() != output.len() {
            return Err(NormalizeError::ProcessingError(
                "Input and output buffers must have same length".to_string(),
            ));
        }

        let gain = db_to_linear(gain_db);

        // Apply gain
        for (i, &sample) in input.iter().enumerate() {
            output[i] = sample * gain;
        }

        // Apply DRC if enabled
        if let Some(ref mut drc) = self.drc {
            drc.process_f64_inplace(output)?;
        }

        // Apply limiting if enabled
        if let Some(ref mut limiter) = self.limiter {
            limiter.process_f64_inplace(output)?;
        }

        Ok(())
    }

    /// Process f32 audio in-place with gain adjustment.
    pub fn process_f32_inplace(
        &mut self,
        samples: &mut [f32],
        gain_db: f64,
    ) -> NormalizeResult<()> {
        let gain = db_to_linear(gain_db) as f32;

        // Apply gain
        for sample in samples.iter_mut() {
            *sample *= gain;
        }

        // Apply DRC if enabled
        if let Some(ref mut drc) = self.drc {
            drc.process_f32_inplace(samples)?;
        }

        // Apply limiting if enabled
        if let Some(ref mut limiter) = self.limiter {
            limiter.process_f32_inplace(samples)?;
        }

        Ok(())
    }

    /// Process f64 audio in-place with gain adjustment.
    pub fn process_f64_inplace(
        &mut self,
        samples: &mut [f64],
        gain_db: f64,
    ) -> NormalizeResult<()> {
        let gain = db_to_linear(gain_db);

        // Apply gain
        for sample in samples.iter_mut() {
            *sample *= gain;
        }

        // Apply DRC if enabled
        if let Some(ref mut drc) = self.drc {
            drc.process_f64_inplace(samples)?;
        }

        // Apply limiting if enabled
        if let Some(ref mut limiter) = self.limiter {
            limiter.process_f64_inplace(samples)?;
        }

        Ok(())
    }

    /// Reset the processor state.
    pub fn reset(&mut self) {
        if let Some(ref mut limiter) = self.limiter {
            limiter.reset();
        }
        if let Some(ref mut drc) = self.drc {
            drc.reset();
        }
    }

    /// Get the processor configuration.
    pub fn config(&self) -> &ProcessorConfig {
        &self.config
    }

    /// Check if limiter is enabled.
    pub fn has_limiter(&self) -> bool {
        self.limiter.is_some()
    }

    /// Check if DRC is enabled.
    pub fn has_drc(&self) -> bool {
        self.drc.is_some()
    }

    /// Get mutable reference to limiter if present.
    pub fn limiter_mut(&mut self) -> Option<&mut TruePeakLimiter> {
        self.limiter.as_mut()
    }

    /// Get mutable reference to DRC if present.
    pub fn drc_mut(&mut self) -> Option<&mut DynamicRangeCompressor> {
        self.drc.as_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_processor_config_validation() {
        let config = ProcessorConfig::new(48000.0, 2);
        assert!(config.validate().is_ok());

        let bad_config = ProcessorConfig {
            sample_rate: 1000.0,
            ..config
        };
        assert!(bad_config.validate().is_err());
    }

    #[test]
    fn test_processor_creation() {
        let config = ProcessorConfig::new(48000.0, 2);
        let processor = NormalizationProcessor::new(config);
        assert!(processor.is_ok());
    }

    #[test]
    fn test_gain_application() {
        let config = ProcessorConfig::minimal(48000.0, 2);
        let mut processor = NormalizationProcessor::new(config).expect("should succeed in test");

        let input = vec![0.5f32; 100];
        let mut output = vec![0.0f32; 100];

        // Apply +6 dB gain (~1.995x)
        processor
            .process_f32(&input, &mut output, 6.0)
            .expect("should succeed in test");

        // Check that gain was applied correctly (0.5 * 1.995 ≈ 0.998)
        let expected = 0.5f32 * 1.9952623149688797f32;
        assert!((output[0] - expected).abs() < 1e-5);
    }

    #[test]
    fn test_inplace_processing() {
        let config = ProcessorConfig::minimal(48000.0, 2);
        let mut processor = NormalizationProcessor::new(config).expect("should succeed in test");

        let mut samples = vec![0.5f32; 100];

        // Apply +6 dB gain (~1.995x) in-place
        processor
            .process_f32_inplace(&mut samples, 6.0)
            .expect("should succeed in test");

        // Check that gain was applied correctly (0.5 * 1.995 ≈ 0.998)
        let expected = 0.5f32 * 1.9952623149688797f32;
        assert!((samples[0] - expected).abs() < 1e-5);
    }
}
