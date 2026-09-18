//! Real-time loudness normalization.
//!
//! Implements low-latency normalization suitable for live streaming and broadcast.

use crate::{
    analyzer::db_to_linear, LimiterConfig, NormalizeError, NormalizeResult, TruePeakLimiter,
};
use oximedia_metering::{LoudnessMeter, MeterConfig, Standard};
use std::collections::VecDeque;

/// Real-time normalizer configuration.
#[derive(Clone, Debug)]
pub struct RealtimeConfig {
    /// Target loudness standard.
    pub standard: Standard,

    /// Sample rate in Hz.
    pub sample_rate: f64,

    /// Number of channels.
    pub channels: usize,

    /// Buffer size in samples (per channel).
    pub buffer_size: usize,

    /// Lookahead time in milliseconds.
    pub lookahead_ms: f64,

    /// Gain smoothing time in seconds.
    pub smoothing_time_s: f64,

    /// Enable true peak limiting.
    pub enable_limiter: bool,
}

impl RealtimeConfig {
    /// Create a new real-time configuration.
    pub fn new(standard: Standard, sample_rate: f64, channels: usize) -> Self {
        Self {
            standard,
            sample_rate,
            channels,
            buffer_size: 1024,
            lookahead_ms: 10.0,
            smoothing_time_s: 1.0,
            enable_limiter: true,
        }
    }

    /// Create a low-latency configuration.
    pub fn low_latency(standard: Standard, sample_rate: f64, channels: usize) -> Self {
        Self {
            standard,
            sample_rate,
            channels,
            buffer_size: 256,
            lookahead_ms: 3.0,
            smoothing_time_s: 0.3,
            enable_limiter: true,
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

        if self.buffer_size == 0 || self.buffer_size > 16384 {
            return Err(NormalizeError::InvalidConfig(
                "Buffer size out of range (1-16384)".to_string(),
            ));
        }

        Ok(())
    }

    /// Get the lookahead buffer size in samples.
    pub fn lookahead_samples(&self) -> usize {
        ((self.lookahead_ms / 1000.0) * self.sample_rate).round() as usize
    }
}

/// Real-time normalizer.
///
/// Provides low-latency normalization with lookahead buffering and smooth gain transitions.
/// Buffer recycling via `process_into` avoids heap allocation on the hot path after warm-up.
pub struct RealtimeNormalizer {
    config: RealtimeConfig,
    meter: LoudnessMeter,
    limiter: Option<TruePeakLimiter>,
    lookahead_buffer: VecDeque<f64>,
    current_gain: f64,
    target_gain: f64,
    smoothing_coeff: f64,
    samples_processed: usize,
    /// Scratch buffer for allocation-free `process_into` recycling.
    scratch_out: Vec<f32>,
}

impl RealtimeNormalizer {
    /// Create a new real-time normalizer.
    pub fn new(config: RealtimeConfig) -> NormalizeResult<Self> {
        config.validate()?;

        let meter_config = MeterConfig::new(config.standard, config.sample_rate, config.channels);
        let meter = LoudnessMeter::new(meter_config)?;

        let limiter = if config.enable_limiter {
            let limiter_config = LimiterConfig {
                sample_rate: config.sample_rate,
                channels: config.channels,
                threshold_dbtp: config.standard.max_true_peak_dbtp(),
                lookahead_ms: config.lookahead_ms,
                release_ms: 100.0,
            };
            Some(TruePeakLimiter::new(limiter_config)?)
        } else {
            None
        };

        let lookahead_size = config.lookahead_samples() * config.channels;
        let lookahead_buffer = VecDeque::with_capacity(lookahead_size);

        // Calculate smoothing coefficient for exponential averaging
        let smoothing_samples = config.smoothing_time_s * config.sample_rate;
        let smoothing_coeff = (-1.0 / smoothing_samples).exp();

        Ok(Self {
            config,
            meter,
            limiter,
            lookahead_buffer,
            current_gain: 1.0,
            target_gain: 1.0,
            smoothing_coeff,
            samples_processed: 0,
            scratch_out: Vec::new(),
        })
    }

    /// Process a chunk of audio.
    pub fn process_chunk(&mut self, input: &[f32], output: &mut [f32]) -> NormalizeResult<()> {
        if input.len() != output.len() {
            return Err(NormalizeError::ProcessingError(
                "Input and output buffers must have same length".to_string(),
            ));
        }

        if input.len() % self.config.channels != 0 {
            return Err(NormalizeError::ProcessingError(
                "Buffer size must be multiple of channel count".to_string(),
            ));
        }

        // Convert to f64 for processing
        let f64_input: Vec<f64> = input.iter().map(|&s| f64::from(s)).collect();

        // Update loudness measurement
        self.meter.process_f64(&f64_input);

        // Update target gain every buffer
        self.update_target_gain();

        // Fill lookahead buffer
        for &sample in &f64_input {
            self.lookahead_buffer.push_back(sample);
        }

        // Process samples with smoothed gain
        let frame_count = input.len() / self.config.channels;
        for frame_idx in 0..frame_count {
            // Smooth the gain transition
            self.current_gain = self.current_gain * self.smoothing_coeff
                + self.target_gain * (1.0 - self.smoothing_coeff);

            let frame_start = frame_idx * self.config.channels;

            // If lookahead buffer is full, output delayed samples with gain
            if self.lookahead_buffer.len() >= self.config.lookahead_samples() * self.config.channels
            {
                for ch in 0..self.config.channels {
                    if let Some(delayed) = self.lookahead_buffer.pop_front() {
                        output[frame_start + ch] = (delayed * self.current_gain) as f32;
                    }
                }
            } else {
                // Buffer not full yet, output silence
                for ch in 0..self.config.channels {
                    output[frame_start + ch] = 0.0;
                }
            }
        }

        // Apply limiting if enabled
        if let Some(ref mut limiter) = self.limiter {
            limiter.process_f32_inplace(output)?;
        }

        self.samples_processed += frame_count;

        Ok(())
    }

    /// Process a chunk of audio into a caller-supplied `Vec<f32>`, recycling its allocation.
    ///
    /// Clears `out` and writes exactly `input.len()` samples.  After the first call with a given
    /// `input.len()`, subsequent calls with the same length will **not** grow `out.capacity()`,
    /// eliminating heap allocation on the hot path.
    ///
    /// Returns an error under the same conditions as [`Self::process_chunk`].
    pub fn process_into(&mut self, input: &[f32], out: &mut Vec<f32>) -> NormalizeResult<()> {
        out.clear();
        // Ensure capacity without re-allocating on subsequent calls.
        if out.capacity() < input.len() {
            out.reserve(input.len() - out.capacity());
        }
        // Extend to the required length with zeros so we can index it directly.
        out.resize(input.len(), 0.0_f32);
        self.process_chunk(input, out.as_mut_slice())
    }

    /// Update the target gain based on current loudness measurement.
    fn update_target_gain(&mut self) {
        let metrics = self.meter.metrics();

        if metrics.integrated_lufs.is_finite() {
            let target_lufs = self.config.standard.target_lufs();
            let gain_db = target_lufs - metrics.integrated_lufs;

            // Clamp gain to safe range
            let gain_db_clamped = gain_db.clamp(-30.0, 30.0);

            self.target_gain = db_to_linear(gain_db_clamped);
        }
    }

    /// Flush the lookahead buffer.
    pub fn flush(&mut self, output: &mut Vec<f32>) {
        while !self.lookahead_buffer.is_empty() {
            for _ch in 0..self.config.channels {
                if let Some(sample) = self.lookahead_buffer.pop_front() {
                    output.push((sample * self.current_gain) as f32);
                }
            }
        }

        // Apply limiter to flushed samples if enabled
        if let Some(ref mut limiter) = self.limiter {
            let start_idx = output.len().saturating_sub(self.lookahead_buffer.len());
            if start_idx < output.len() {
                let _ = limiter.process_f32_inplace(&mut output[start_idx..]);
            }
        }
    }

    /// Reset the normalizer state.
    ///
    /// The internal `scratch_out` buffer retains its allocation so that subsequent calls
    /// to `process_into` do not need to re-allocate.
    pub fn reset(&mut self) {
        self.meter.reset();
        if let Some(ref mut limiter) = self.limiter {
            limiter.reset();
        }
        self.lookahead_buffer.clear();
        self.current_gain = 1.0;
        self.target_gain = 1.0;
        self.samples_processed = 0;
        // Intentionally keep scratch_out capacity; clear length only.
        self.scratch_out.clear();
    }

    /// Get the current gain in dB.
    pub fn current_gain_db(&self) -> f64 {
        20.0 * self.current_gain.log10()
    }

    /// Get the target gain in dB.
    pub fn target_gain_db(&self) -> f64 {
        20.0 * self.target_gain.log10()
    }

    /// Get the latency in samples.
    pub fn latency_samples(&self) -> usize {
        self.config.lookahead_samples()
    }

    /// Get the latency in milliseconds.
    pub fn latency_ms(&self) -> f64 {
        self.config.lookahead_ms
    }

    /// Get the configuration.
    pub fn config(&self) -> &RealtimeConfig {
        &self.config
    }

    /// Get the number of samples processed.
    pub fn samples_processed(&self) -> usize {
        self.samples_processed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_realtime_config_validation() {
        let config = RealtimeConfig::new(Standard::EbuR128, 48000.0, 2);
        assert!(config.validate().is_ok());

        let bad_config = RealtimeConfig {
            buffer_size: 0,
            ..config
        };
        assert!(bad_config.validate().is_err());
    }

    #[test]
    fn test_realtime_normalizer_creation() {
        let config = RealtimeConfig::new(Standard::EbuR128, 48000.0, 2);
        let normalizer = RealtimeNormalizer::new(config);
        assert!(normalizer.is_ok());
    }

    #[test]
    fn test_low_latency_config() {
        let config = RealtimeConfig::low_latency(Standard::Spotify, 48000.0, 2);
        assert_eq!(config.buffer_size, 256);
        assert_eq!(config.lookahead_ms, 3.0);
    }

    #[test]
    fn test_latency_calculation() {
        let config = RealtimeConfig::new(Standard::EbuR128, 48000.0, 2);
        let normalizer = RealtimeNormalizer::new(config).expect("should succeed in test");

        assert_eq!(normalizer.latency_ms(), 10.0);
        assert_eq!(normalizer.latency_samples(), 480); // 10ms at 48kHz
    }
}
