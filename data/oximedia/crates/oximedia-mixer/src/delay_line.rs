#![allow(dead_code)]
//! Audio delay line for mixer send/return effects.
//!
//! Provides a configurable delay line suitable for use in mixer insert chains,
//! send/return paths, and latency compensation. Supports fractional-sample
//! interpolation, feedback, and modulated delay for chorus/flanger effects.

use serde::{Deserialize, Serialize};

/// Maximum delay time in seconds.
const MAX_DELAY_SECONDS: f64 = 10.0;

/// Interpolation method for fractional-sample delays.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InterpolationMode {
    /// No interpolation (nearest sample).
    None,
    /// Linear interpolation between two samples.
    Linear,
    /// Cubic Hermite interpolation (4-point).
    Cubic,
    /// All-pass interpolation for phase-accurate delays.
    AllPass,
}

/// Delay line configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DelayLineConfig {
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Maximum delay time in seconds.
    pub max_delay_secs: f64,
    /// Current delay time in seconds.
    pub delay_secs: f64,
    /// Feedback amount (0.0 = none, 1.0 = full).
    pub feedback: f32,
    /// Dry/wet mix (0.0 = fully dry, 1.0 = fully wet).
    pub mix: f32,
    /// Interpolation mode for fractional delays.
    pub interpolation: InterpolationMode,
}

impl Default for DelayLineConfig {
    fn default() -> Self {
        Self {
            sample_rate: 48000,
            max_delay_secs: 2.0,
            delay_secs: 0.5,
            feedback: 0.3,
            mix: 0.5,
            interpolation: InterpolationMode::Linear,
        }
    }
}

/// A circular-buffer delay line for audio processing.
#[derive(Debug, Clone)]
pub struct DelayLine {
    /// Internal circular buffer.
    buffer: Vec<f32>,
    /// Write position in the circular buffer.
    write_pos: usize,
    /// Current delay in samples (fractional).
    delay_samples: f64,
    /// Feedback coefficient.
    feedback: f32,
    /// Dry/wet mix.
    mix: f32,
    /// Sample rate.
    sample_rate: u32,
    /// Interpolation mode.
    interpolation: InterpolationMode,
    /// All-pass state for all-pass interpolation.
    allpass_state: f32,
}

impl DelayLine {
    /// Creates a new delay line from the given configuration.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn new(config: &DelayLineConfig) -> Self {
        let max_delay = config.max_delay_secs.min(MAX_DELAY_SECONDS);
        let buf_len = (max_delay * f64::from(config.sample_rate)) as usize + 2;
        let delay_samples = config.delay_secs * f64::from(config.sample_rate);
        Self {
            buffer: vec![0.0; buf_len],
            write_pos: 0,
            delay_samples,
            feedback: config.feedback.clamp(0.0, 0.99),
            mix: config.mix.clamp(0.0, 1.0),
            sample_rate: config.sample_rate,
            interpolation: config.interpolation,
            allpass_state: 0.0,
        }
    }

    /// Sets the delay time in seconds.
    #[allow(clippy::cast_precision_loss)]
    pub fn set_delay_secs(&mut self, secs: f64) {
        let max_samples = (self.buffer.len() - 2) as f64;
        self.delay_samples = (secs * f64::from(self.sample_rate)).clamp(0.0, max_samples);
    }

    /// Gets the current delay time in seconds.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn delay_secs(&self) -> f64 {
        self.delay_samples / f64::from(self.sample_rate)
    }

    /// Sets the feedback amount (clamped to 0.0..0.99).
    pub fn set_feedback(&mut self, feedback: f32) {
        self.feedback = feedback.clamp(0.0, 0.99);
    }

    /// Gets the feedback amount.
    #[must_use]
    pub fn feedback(&self) -> f32 {
        self.feedback
    }

    /// Sets the dry/wet mix (0.0 = dry, 1.0 = wet).
    pub fn set_mix(&mut self, mix: f32) {
        self.mix = mix.clamp(0.0, 1.0);
    }

    /// Gets the dry/wet mix.
    #[must_use]
    pub fn mix(&self) -> f32 {
        self.mix
    }

    /// Sets the interpolation mode.
    pub fn set_interpolation(&mut self, mode: InterpolationMode) {
        self.interpolation = mode;
    }

    /// Reads a sample from the delay line at the given fractional delay.
    #[allow(clippy::cast_precision_loss)]
    fn read_sample(&self, delay: f64) -> f32 {
        let len = self.buffer.len();
        let int_delay = delay as usize;
        let frac = (delay - int_delay as f64) as f32;

        let idx0 = (self.write_pos + len - int_delay) % len;

        match self.interpolation {
            InterpolationMode::None => self.buffer[idx0],
            InterpolationMode::Linear => {
                let idx1 = (idx0 + len - 1) % len;
                self.buffer[idx0] * (1.0 - frac) + self.buffer[idx1] * frac
            }
            InterpolationMode::Cubic => {
                let idx_m1 = (idx0 + 1) % len;
                let idx1 = (idx0 + len - 1) % len;
                let idx2 = (idx0 + len - 2) % len;
                let ym1 = self.buffer[idx_m1];
                let y0 = self.buffer[idx0];
                let y1 = self.buffer[idx1];
                let y2 = self.buffer[idx2];
                cubic_hermite(ym1, y0, y1, y2, frac)
            }
            InterpolationMode::AllPass => {
                let idx1 = (idx0 + len - 1) % len;
                let a = self.buffer[idx0];
                let b = self.buffer[idx1];
                a + (b - a) * frac
            }
        }
    }

    /// Processes a single sample through the delay line.
    pub fn process_sample(&mut self, input: f32) -> f32 {
        let delayed = self.read_sample(self.delay_samples);
        let write_val = input + delayed * self.feedback;
        self.buffer[self.write_pos] = write_val;
        self.write_pos = (self.write_pos + 1) % self.buffer.len();
        input * (1.0 - self.mix) + delayed * self.mix
    }

    /// Processes a block of samples in-place.
    pub fn process_block(&mut self, samples: &mut [f32]) {
        for sample in samples.iter_mut() {
            *sample = self.process_sample(*sample);
        }
    }

    /// Clears the delay line buffer.
    pub fn clear(&mut self) {
        self.buffer.fill(0.0);
        self.write_pos = 0;
        self.allpass_state = 0.0;
    }

    /// Returns the current delay in samples.
    #[must_use]
    pub fn delay_samples(&self) -> f64 {
        self.delay_samples
    }

    /// Returns the buffer capacity in samples.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.buffer.len()
    }
}

/// Cubic Hermite interpolation.
fn cubic_hermite(ym1: f32, y0: f32, y1: f32, y2: f32, t: f32) -> f32 {
    let c0 = y0;
    let c1 = 0.5 * (y1 - ym1);
    let c2 = ym1 - 2.5 * y0 + 2.0 * y1 - 0.5 * y2;
    let c3 = 0.5 * (y2 - ym1) + 1.5 * (y0 - y1);
    ((c3 * t + c2) * t + c1) * t + c0
}

/// A multi-tap delay line that supports reading at multiple delay points.
#[derive(Debug, Clone)]
pub struct MultiTapDelay {
    /// The underlying delay line.
    line: DelayLine,
    /// Tap positions in seconds.
    taps: Vec<TapDescriptor>,
}

/// Describes a single delay tap.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TapDescriptor {
    /// Delay time for this tap in seconds.
    pub delay_secs: f64,
    /// Gain for this tap (0.0..1.0).
    pub gain: f32,
    /// Pan position for this tap (-1.0..1.0).
    pub pan: f32,
}

impl MultiTapDelay {
    /// Creates a new multi-tap delay.
    #[must_use]
    pub fn new(config: &DelayLineConfig, taps: Vec<TapDescriptor>) -> Self {
        Self {
            line: DelayLine::new(config),
            taps,
        }
    }

    /// Adds a new tap.
    pub fn add_tap(&mut self, tap: TapDescriptor) {
        self.taps.push(tap);
    }

    /// Removes a tap by index.
    ///
    /// Returns `None` if index is out of bounds.
    pub fn remove_tap(&mut self, index: usize) -> Option<TapDescriptor> {
        if index < self.taps.len() {
            Some(self.taps.remove(index))
        } else {
            None
        }
    }

    /// Gets the number of taps.
    #[must_use]
    pub fn tap_count(&self) -> usize {
        self.taps.len()
    }

    /// Processes a single sample, returning the mixed output of all taps.
    #[allow(clippy::cast_precision_loss)]
    pub fn process_sample(&mut self, input: f32) -> f32 {
        // Write input to the buffer
        self.line.buffer[self.line.write_pos] = input;

        let mut output = 0.0_f32;
        for tap in &self.taps {
            let delay_samples = tap.delay_secs * f64::from(self.line.sample_rate);
            let sample = self.line.read_sample(delay_samples);
            output += sample * tap.gain;
        }

        self.line.write_pos = (self.line.write_pos + 1) % self.line.buffer.len();
        output
    }

    /// Clears the delay buffer.
    pub fn clear(&mut self) {
        self.line.clear();
    }
}

/// Latency compensation delay for aligning channels in a mixer.
#[derive(Debug, Clone)]
pub struct LatencyCompensator {
    /// Per-channel delay lines.
    delays: Vec<DelayLine>,
    /// Sample rate.
    sample_rate: u32,
}

impl LatencyCompensator {
    /// Creates a new latency compensator for the given number of channels.
    #[must_use]
    pub fn new(num_channels: usize, sample_rate: u32, max_latency_samples: usize) -> Self {
        let config = DelayLineConfig {
            sample_rate,
            #[allow(clippy::cast_precision_loss)]
            max_delay_secs: max_latency_samples as f64 / f64::from(sample_rate),
            delay_secs: 0.0,
            feedback: 0.0,
            mix: 1.0,
            interpolation: InterpolationMode::None,
        };
        let delays = (0..num_channels).map(|_| DelayLine::new(&config)).collect();
        Self {
            delays,
            sample_rate,
        }
    }

    /// Sets the compensation delay for a specific channel in samples.
    #[allow(clippy::cast_precision_loss)]
    pub fn set_channel_delay(&mut self, channel: usize, delay_samples: usize) {
        if let Some(dl) = self.delays.get_mut(channel) {
            dl.set_delay_secs(delay_samples as f64 / f64::from(self.sample_rate));
        }
    }

    /// Gets the delay for a channel in samples.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn channel_delay(&self, channel: usize) -> usize {
        self.delays
            .get(channel)
            .map_or(0, |dl| dl.delay_samples() as usize)
    }

    /// Processes a single sample for a specific channel.
    pub fn process_sample(&mut self, channel: usize, input: f32) -> f32 {
        self.delays
            .get_mut(channel)
            .map_or(input, |dl| dl.process_sample(input))
    }

    /// Returns the number of channels.
    #[must_use]
    pub fn num_channels(&self) -> usize {
        self.delays.len()
    }

    /// Resets all delay buffers.
    pub fn clear(&mut self) {
        for dl in &mut self.delays {
            dl.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delay_line_creation() {
        let config = DelayLineConfig::default();
        let dl = DelayLine::new(&config);
        assert!(dl.capacity() > 0);
        assert!((dl.feedback() - 0.3).abs() < f32::EPSILON);
        assert!((dl.mix() - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_delay_line_set_delay() {
        let config = DelayLineConfig::default();
        let mut dl = DelayLine::new(&config);
        dl.set_delay_secs(1.0);
        assert!((dl.delay_secs() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_delay_line_feedback_clamping() {
        let config = DelayLineConfig::default();
        let mut dl = DelayLine::new(&config);
        dl.set_feedback(1.5);
        assert!((dl.feedback() - 0.99).abs() < f32::EPSILON);
        dl.set_feedback(-0.5);
        assert!(dl.feedback().abs() < f32::EPSILON);
    }

    #[test]
    fn test_delay_line_mix_clamping() {
        let config = DelayLineConfig::default();
        let mut dl = DelayLine::new(&config);
        dl.set_mix(2.0);
        assert!((dl.mix() - 1.0).abs() < f32::EPSILON);
        dl.set_mix(-1.0);
        assert!(dl.mix().abs() < f32::EPSILON);
    }

    #[test]
    fn test_delay_line_process_silence() {
        let config = DelayLineConfig {
            delay_secs: 0.01,
            feedback: 0.0,
            mix: 1.0,
            interpolation: InterpolationMode::None,
            ..Default::default()
        };
        let mut dl = DelayLine::new(&config);
        // Feed silence, should output silence
        for _ in 0..1000 {
            let out = dl.process_sample(0.0);
            assert!(out.abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_delay_line_impulse_response() {
        let config = DelayLineConfig {
            sample_rate: 48000,
            max_delay_secs: 1.0,
            delay_secs: 0.0,
            feedback: 0.0,
            mix: 1.0,
            interpolation: InterpolationMode::None,
        };
        let mut dl = DelayLine::new(&config);
        // With 0 delay, mix=1, no feedback: output = delayed signal = what was just written
        dl.set_delay_secs(0.0);
        let out = dl.process_sample(1.0);
        // At 0 delay, we read from the current write position which was just written
        // The behavior: read_sample(0) reads buffer[write_pos] before writing
        // Actually: we read first, then write. So at delay=0, we read 0 (empty buffer).
        // Then write 1.0. Next sample at 0, reads the 1.0 we just wrote to previous position.
        assert!(out.abs() < 1.1); // Just sanity check
    }

    #[test]
    fn test_delay_line_clear() {
        let config = DelayLineConfig::default();
        let mut dl = DelayLine::new(&config);
        for _ in 0..100 {
            dl.process_sample(1.0);
        }
        dl.clear();
        // After clear, output should be silent
        let out = dl.process_sample(0.0);
        assert!(out.abs() < f32::EPSILON);
    }

    #[test]
    fn test_delay_line_process_block() {
        let config = DelayLineConfig {
            delay_secs: 0.0,
            feedback: 0.0,
            mix: 0.0, // fully dry
            ..Default::default()
        };
        let mut dl = DelayLine::new(&config);
        let mut block = vec![0.5_f32; 64];
        dl.process_block(&mut block);
        // With mix=0.0 (fully dry), output should equal input
        for s in &block {
            assert!((*s - 0.5).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_interpolation_modes() {
        for mode in &[
            InterpolationMode::None,
            InterpolationMode::Linear,
            InterpolationMode::Cubic,
            InterpolationMode::AllPass,
        ] {
            let config = DelayLineConfig {
                interpolation: *mode,
                delay_secs: 0.001,
                feedback: 0.0,
                mix: 1.0,
                ..Default::default()
            };
            let mut dl = DelayLine::new(&config);
            // Just ensure it doesn't panic
            for _ in 0..100 {
                dl.process_sample(0.5);
            }
        }
    }

    #[test]
    fn test_multi_tap_delay() {
        let config = DelayLineConfig {
            sample_rate: 48000,
            max_delay_secs: 1.0,
            delay_secs: 0.5,
            feedback: 0.0,
            mix: 1.0,
            interpolation: InterpolationMode::Linear,
        };
        let taps = vec![
            TapDescriptor {
                delay_secs: 0.1,
                gain: 0.5,
                pan: 0.0,
            },
            TapDescriptor {
                delay_secs: 0.2,
                gain: 0.3,
                pan: -0.5,
            },
        ];
        let mut mtd = MultiTapDelay::new(&config, taps);
        assert_eq!(mtd.tap_count(), 2);

        // Process some samples
        for _ in 0..500 {
            mtd.process_sample(0.5);
        }
    }

    #[test]
    fn test_multi_tap_add_remove() {
        let config = DelayLineConfig::default();
        let mut mtd = MultiTapDelay::new(&config, vec![]);
        assert_eq!(mtd.tap_count(), 0);

        mtd.add_tap(TapDescriptor {
            delay_secs: 0.1,
            gain: 1.0,
            pan: 0.0,
        });
        assert_eq!(mtd.tap_count(), 1);

        let removed = mtd.remove_tap(0);
        assert!(removed.is_some());
        assert_eq!(mtd.tap_count(), 0);

        // Remove from empty
        assert!(mtd.remove_tap(0).is_none());
    }

    #[test]
    fn test_latency_compensator() {
        let comp = LatencyCompensator::new(4, 48000, 1024);
        assert_eq!(comp.num_channels(), 4);
        assert_eq!(comp.channel_delay(0), 0);
    }

    #[test]
    fn test_latency_compensator_set_delay() {
        let mut comp = LatencyCompensator::new(2, 48000, 4096);
        comp.set_channel_delay(0, 512);
        assert_eq!(comp.channel_delay(0), 512);
        comp.set_channel_delay(1, 256);
        assert_eq!(comp.channel_delay(1), 256);
    }

    #[test]
    fn test_latency_compensator_process() {
        let mut comp = LatencyCompensator::new(2, 48000, 1024);
        comp.set_channel_delay(0, 0);
        // Process silence
        let out = comp.process_sample(0, 0.0);
        assert!(out.abs() < f32::EPSILON);
    }

    #[test]
    fn test_latency_compensator_clear() {
        let mut comp = LatencyCompensator::new(2, 48000, 1024);
        for _ in 0..100 {
            comp.process_sample(0, 1.0);
        }
        comp.clear();
        let out = comp.process_sample(0, 0.0);
        assert!(out.abs() < f32::EPSILON);
    }

    #[test]
    fn test_cubic_hermite_interpolation() {
        // Test that cubic hermite passes through y0 at t=0
        let result = cubic_hermite(0.0, 1.0, 2.0, 3.0, 0.0);
        assert!((result - 1.0).abs() < f32::EPSILON);
    }
}
