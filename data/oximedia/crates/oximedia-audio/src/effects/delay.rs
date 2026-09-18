//! Delay line utilities for modulation effects.
//!
//! This module provides specialized delay line implementations optimized
//! for chorus, flanger, and other modulation effects that require
//! fractional delay and interpolation.

#![forbid(unsafe_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use std::collections::VecDeque;

/// Interpolation method for fractional delay.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum InterpolationMode {
    /// No interpolation - nearest sample.
    None,
    /// Linear interpolation between two samples.
    #[default]
    Linear,
    /// Cubic interpolation using four samples.
    Cubic,
}

/// Fractional delay line with interpolation.
///
/// Supports reading at non-integer delay times using various interpolation
/// methods. Essential for smooth modulation effects.
#[derive(Clone, Debug)]
pub struct FractionalDelayLine {
    /// Circular buffer for samples.
    buffer: VecDeque<f64>,
    /// Maximum delay in samples.
    max_delay_samples: usize,
    /// Interpolation mode.
    interpolation: InterpolationMode,
}

impl FractionalDelayLine {
    /// Create a new fractional delay line.
    ///
    /// # Arguments
    ///
    /// * `max_delay_ms` - Maximum delay time in milliseconds
    /// * `sample_rate` - Sample rate in Hz
    /// * `interpolation` - Interpolation mode
    #[must_use]
    pub fn new(max_delay_ms: f64, sample_rate: f64, interpolation: InterpolationMode) -> Self {
        let max_delay_samples = ((max_delay_ms * 0.001 * sample_rate) as usize).max(4);
        let mut buffer = VecDeque::with_capacity(max_delay_samples + 4);

        // Initialize buffer with zeros
        for _ in 0..max_delay_samples {
            buffer.push_back(0.0);
        }

        Self {
            buffer,
            max_delay_samples,
            interpolation,
        }
    }

    /// Write a new sample and advance the delay line.
    pub fn write(&mut self, sample: f64) {
        self.buffer.push_back(sample);
        if self.buffer.len() > self.max_delay_samples {
            self.buffer.pop_front();
        }
    }

    /// Read a sample at a fractional delay time.
    ///
    /// # Arguments
    ///
    /// * `delay_samples` - Delay time in samples (can be fractional)
    ///
    /// # Returns
    ///
    /// Interpolated sample value
    #[must_use]
    pub fn read(&self, delay_samples: f64) -> f64 {
        if self.buffer.is_empty() {
            return 0.0;
        }

        let delay_clamped = delay_samples.clamp(0.0, (self.max_delay_samples - 1) as f64);

        match self.interpolation {
            InterpolationMode::None => self.read_nearest(delay_clamped),
            InterpolationMode::Linear => self.read_linear(delay_clamped),
            InterpolationMode::Cubic => self.read_cubic(delay_clamped),
        }
    }

    /// Read nearest sample (no interpolation).
    fn read_nearest(&self, delay_samples: f64) -> f64 {
        let index = delay_samples.round() as usize;
        self.read_at_index(index)
    }

    /// Read with linear interpolation.
    fn read_linear(&self, delay_samples: f64) -> f64 {
        let index = delay_samples.floor() as usize;
        let frac = delay_samples - delay_samples.floor();

        let sample1 = self.read_at_index(index);
        let sample2 = self.read_at_index(index + 1);

        sample1 + frac * (sample2 - sample1)
    }

    /// Read with cubic interpolation (Hermite).
    fn read_cubic(&self, delay_samples: f64) -> f64 {
        let index = delay_samples.floor() as usize;
        let frac = delay_samples - delay_samples.floor();

        let y0 = self.read_at_index(index.saturating_sub(1));
        let y1 = self.read_at_index(index);
        let y2 = self.read_at_index(index + 1);
        let y3 = self.read_at_index(index + 2);

        // Hermite interpolation
        let c0 = y1;
        let c1 = 0.5 * (y2 - y0);
        let c2 = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3;
        let c3 = 0.5 * (y3 - y0) + 1.5 * (y1 - y2);

        ((c3 * frac + c2) * frac + c1) * frac + c0
    }

    /// Read sample at specific index from the back of the buffer.
    fn read_at_index(&self, index: usize) -> f64 {
        if index >= self.buffer.len() {
            0.0
        } else {
            self.buffer[self.buffer.len() - 1 - index]
        }
    }

    /// Process sample with feedback.
    ///
    /// # Arguments
    ///
    /// * `input` - Input sample
    /// * `delay_samples` - Delay time in samples
    /// * `feedback` - Feedback amount (-1.0 to 1.0)
    pub fn process_with_feedback(&mut self, input: f64, delay_samples: f64, feedback: f64) -> f64 {
        let delayed = self.read(delay_samples);
        self.write(input + delayed * feedback);
        delayed
    }

    /// Reset the delay line.
    pub fn reset(&mut self) {
        self.buffer.clear();
        for _ in 0..self.max_delay_samples {
            self.buffer.push_back(0.0);
        }
    }

    /// Get maximum delay in samples.
    #[must_use]
    pub fn max_delay_samples(&self) -> usize {
        self.max_delay_samples
    }

    /// Set interpolation mode.
    pub fn set_interpolation(&mut self, interpolation: InterpolationMode) {
        self.interpolation = interpolation;
    }

    /// Get current interpolation mode.
    #[must_use]
    pub fn interpolation(&self) -> InterpolationMode {
        self.interpolation
    }
}

/// All-pass filter for phase shifting.
///
/// Used in phaser effects to create frequency-dependent phase shifts.
#[derive(Clone, Debug)]
pub struct AllPassFilter {
    /// Coefficient.
    coefficient: f64,
    /// Previous input sample.
    x1: f64,
    /// Previous output sample.
    y1: f64,
}

impl AllPassFilter {
    /// Create a new all-pass filter.
    ///
    /// # Arguments
    ///
    /// * `coefficient` - Filter coefficient (-1.0 to 1.0)
    #[must_use]
    pub fn new(coefficient: f64) -> Self {
        Self {
            coefficient: coefficient.clamp(-0.999, 0.999),
            x1: 0.0,
            y1: 0.0,
        }
    }

    /// Process one sample through the all-pass filter.
    pub fn process(&mut self, input: f64) -> f64 {
        let output = self.coefficient * input + self.x1 - self.coefficient * self.y1;
        self.x1 = input;
        self.y1 = output;
        output
    }

    /// Set the coefficient.
    pub fn set_coefficient(&mut self, coefficient: f64) {
        self.coefficient = coefficient.clamp(-0.999, 0.999);
    }

    /// Get the current coefficient.
    #[must_use]
    pub fn coefficient(&self) -> f64 {
        self.coefficient
    }

    /// Reset the filter state.
    pub fn reset(&mut self) {
        self.x1 = 0.0;
        self.y1 = 0.0;
    }
}

/// First-order all-pass filter for phaser stages.
///
/// Provides a simpler all-pass implementation optimized for phaser effects.
#[derive(Clone, Debug)]
pub struct FirstOrderAllPass {
    /// State variable.
    state: f64,
    /// Coefficient.
    a1: f64,
}

impl FirstOrderAllPass {
    /// Create a new first-order all-pass filter.
    ///
    /// # Arguments
    ///
    /// * `frequency` - Cutoff frequency in Hz
    /// * `sample_rate` - Sample rate in Hz
    #[must_use]
    pub fn new(frequency: f64, sample_rate: f64) -> Self {
        let a1 = Self::calculate_coefficient(frequency, sample_rate);
        Self { state: 0.0, a1 }
    }

    /// Calculate coefficient from frequency.
    fn calculate_coefficient(frequency: f64, sample_rate: f64) -> f64 {
        let tan_half = (std::f64::consts::PI * frequency / sample_rate).tan();
        (tan_half - 1.0) / (tan_half + 1.0)
    }

    /// Process one sample.
    pub fn process(&mut self, input: f64) -> f64 {
        let output = self.a1 * input + self.state;
        self.state = input - self.a1 * output;
        output
    }

    /// Update the frequency.
    pub fn set_frequency(&mut self, frequency: f64, sample_rate: f64) {
        self.a1 = Self::calculate_coefficient(frequency, sample_rate);
    }

    /// Reset the filter.
    pub fn reset(&mut self) {
        self.state = 0.0;
    }

    /// Get the current coefficient.
    #[must_use]
    pub fn coefficient(&self) -> f64 {
        self.a1
    }
}

/// Modulated delay line with built-in LFO modulation.
///
/// Combines fractional delay with automatic modulation for convenience.
#[derive(Clone, Debug)]
pub struct ModulatedDelayLine {
    /// Fractional delay line.
    delay_line: FractionalDelayLine,
    /// Base delay in samples.
    base_delay: f64,
    /// Modulation depth in samples.
    mod_depth: f64,
}

impl ModulatedDelayLine {
    /// Create a new modulated delay line.
    ///
    /// # Arguments
    ///
    /// * `max_delay_ms` - Maximum delay time in milliseconds
    /// * `sample_rate` - Sample rate in Hz
    #[must_use]
    pub fn new(max_delay_ms: f64, sample_rate: f64) -> Self {
        Self {
            delay_line: FractionalDelayLine::new(
                max_delay_ms,
                sample_rate,
                InterpolationMode::Linear,
            ),
            base_delay: 0.0,
            mod_depth: 0.0,
        }
    }

    /// Set base delay time.
    ///
    /// # Arguments
    ///
    /// * `delay_ms` - Base delay in milliseconds
    /// * `sample_rate` - Sample rate in Hz
    pub fn set_base_delay(&mut self, delay_ms: f64, sample_rate: f64) {
        self.base_delay = delay_ms * 0.001 * sample_rate;
    }

    /// Set modulation depth.
    ///
    /// # Arguments
    ///
    /// * `depth_ms` - Modulation depth in milliseconds
    /// * `sample_rate` - Sample rate in Hz
    pub fn set_mod_depth(&mut self, depth_ms: f64, sample_rate: f64) {
        self.mod_depth = depth_ms * 0.001 * sample_rate;
    }

    /// Process one sample with LFO modulation.
    ///
    /// # Arguments
    ///
    /// * `input` - Input sample
    /// * `lfo_value` - LFO value (-1.0 to 1.0)
    pub fn process(&mut self, input: f64, lfo_value: f64) -> f64 {
        let modulated_delay = self.base_delay + lfo_value * self.mod_depth;
        let output = self.delay_line.read(modulated_delay);
        self.delay_line.write(input);
        output
    }

    /// Process with feedback.
    ///
    /// # Arguments
    ///
    /// * `input` - Input sample
    /// * `lfo_value` - LFO value (-1.0 to 1.0)
    /// * `feedback` - Feedback amount
    pub fn process_with_feedback(&mut self, input: f64, lfo_value: f64, feedback: f64) -> f64 {
        let modulated_delay = self.base_delay + lfo_value * self.mod_depth;
        self.delay_line
            .process_with_feedback(input, modulated_delay, feedback)
    }

    /// Reset the delay line.
    pub fn reset(&mut self) {
        self.delay_line.reset();
    }

    /// Set interpolation mode.
    pub fn set_interpolation(&mut self, mode: InterpolationMode) {
        self.delay_line.set_interpolation(mode);
    }
}
