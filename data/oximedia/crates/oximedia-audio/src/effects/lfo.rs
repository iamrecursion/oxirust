//! Low-Frequency Oscillator (LFO) implementation.
//!
//! This module provides LFO generators with multiple waveform types
//! for modulating audio effects parameters.

#![forbid(unsafe_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use std::f64::consts::PI;

/// LFO waveform type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LfoWaveform {
    /// Sine wave - smooth, natural modulation.
    #[default]
    Sine,
    /// Triangle wave - linear rise and fall.
    Triangle,
    /// Sawtooth wave - linear rise, instant fall.
    Sawtooth,
    /// Square wave - instant switching between high and low.
    Square,
    /// Random wave - stepped random values.
    Random,
}

/// Low-Frequency Oscillator.
///
/// Generates periodic control signals for modulating effect parameters.
/// Supports multiple waveform types with configurable rate and phase.
#[derive(Clone, Debug)]
pub struct Lfo {
    /// Current phase position (0.0 to 1.0).
    phase: f64,
    /// Phase increment per sample.
    phase_increment: f64,
    /// Waveform type.
    waveform: LfoWaveform,
    /// Sample rate in Hz.
    sample_rate: f64,
    /// LFO rate in Hz.
    rate_hz: f64,
    /// Random state for random waveform.
    random_state: u64,
    /// Last random value.
    last_random: f64,
}

impl Lfo {
    /// Create a new LFO.
    ///
    /// # Arguments
    ///
    /// * `rate_hz` - LFO rate in Hz (typically 0.1 to 20.0)
    /// * `sample_rate` - Audio sample rate in Hz
    /// * `waveform` - Waveform type
    #[must_use]
    pub fn new(rate_hz: f64, sample_rate: f64, waveform: LfoWaveform) -> Self {
        let phase_increment = rate_hz / sample_rate;
        Self {
            phase: 0.0,
            phase_increment,
            waveform,
            sample_rate,
            rate_hz,
            random_state: 0x123456789ABCDEF0,
            last_random: 0.0,
        }
    }

    /// Set the LFO rate.
    ///
    /// # Arguments
    ///
    /// * `rate_hz` - LFO rate in Hz
    pub fn set_rate(&mut self, rate_hz: f64) {
        self.rate_hz = rate_hz;
        self.phase_increment = rate_hz / self.sample_rate;
    }

    /// Get the current LFO rate.
    #[must_use]
    pub fn rate(&self) -> f64 {
        self.rate_hz
    }

    /// Set the waveform type.
    pub fn set_waveform(&mut self, waveform: LfoWaveform) {
        self.waveform = waveform;
    }

    /// Get the current waveform type.
    #[must_use]
    pub fn waveform(&self) -> LfoWaveform {
        self.waveform
    }

    /// Set the phase offset.
    ///
    /// # Arguments
    ///
    /// * `phase` - Phase offset (0.0 to 1.0)
    pub fn set_phase(&mut self, phase: f64) {
        self.phase = phase.fract();
    }

    /// Get the current phase.
    #[must_use]
    pub fn phase(&self) -> f64 {
        self.phase
    }

    /// Reset the LFO to zero phase.
    pub fn reset(&mut self) {
        self.phase = 0.0;
        self.last_random = 0.0;
    }

    /// Generate next LFO sample value.
    ///
    /// Returns a value between -1.0 and 1.0.
    pub fn next(&mut self) -> f64 {
        let value = self.generate_waveform();

        // Advance phase
        self.phase += self.phase_increment;
        if self.phase >= 1.0 {
            self.phase -= 1.0;
        }

        value
    }

    /// Generate next unipolar LFO sample value.
    ///
    /// Returns a value between 0.0 and 1.0.
    pub fn next_unipolar(&mut self) -> f64 {
        (self.next() + 1.0) * 0.5
    }

    /// Generate waveform value at current phase.
    fn generate_waveform(&mut self) -> f64 {
        match self.waveform {
            LfoWaveform::Sine => self.sine(),
            LfoWaveform::Triangle => self.triangle(),
            LfoWaveform::Sawtooth => self.sawtooth(),
            LfoWaveform::Square => self.square(),
            LfoWaveform::Random => self.random(),
        }
    }

    /// Generate sine wave value.
    fn sine(&self) -> f64 {
        (self.phase * 2.0 * PI).sin()
    }

    /// Generate triangle wave value.
    fn triangle(&self) -> f64 {
        if self.phase < 0.5 {
            4.0 * self.phase - 1.0
        } else {
            3.0 - 4.0 * self.phase
        }
    }

    /// Generate sawtooth wave value.
    fn sawtooth(&self) -> f64 {
        2.0 * self.phase - 1.0
    }

    /// Generate square wave value.
    fn square(&self) -> f64 {
        if self.phase < 0.5 {
            1.0
        } else {
            -1.0
        }
    }

    /// Generate random wave value using simple PRNG.
    fn random(&mut self) -> f64 {
        // Generate new random value at phase wrap
        if self.phase < self.phase_increment {
            self.random_state = self.random_state.wrapping_mul(6364136223846793005);
            self.random_state = self.random_state.wrapping_add(1442695040888963407);
            let normalized = (self.random_state >> 32) as f64 / (u32::MAX as f64);
            self.last_random = normalized * 2.0 - 1.0;
        }
        self.last_random
    }

    /// Get value at current phase without advancing.
    #[must_use]
    pub fn peek(&self) -> f64 {
        match self.waveform {
            LfoWaveform::Sine => self.sine(),
            LfoWaveform::Triangle => self.triangle(),
            LfoWaveform::Sawtooth => self.sawtooth(),
            LfoWaveform::Square => self.square(),
            LfoWaveform::Random => self.last_random,
        }
    }
}

/// Stereo LFO with independent or linked channels.
///
/// Provides two LFOs with configurable phase offset for stereo effects.
#[derive(Clone, Debug)]
pub struct StereoLfo {
    /// Left channel LFO.
    left: Lfo,
    /// Right channel LFO.
    right: Lfo,
    /// Phase offset between channels (0.0 to 1.0).
    phase_offset: f64,
}

impl StereoLfo {
    /// Create a new stereo LFO.
    ///
    /// # Arguments
    ///
    /// * `rate_hz` - LFO rate in Hz
    /// * `sample_rate` - Audio sample rate in Hz
    /// * `waveform` - Waveform type
    /// * `phase_offset` - Phase offset between channels (0.0 to 1.0)
    ///                    0.0 = in phase, 0.5 = 180 degrees out of phase
    #[must_use]
    pub fn new(rate_hz: f64, sample_rate: f64, waveform: LfoWaveform, phase_offset: f64) -> Self {
        let left = Lfo::new(rate_hz, sample_rate, waveform);
        let mut right = Lfo::new(rate_hz, sample_rate, waveform);
        right.set_phase(phase_offset);

        Self {
            left,
            right,
            phase_offset,
        }
    }

    /// Set the LFO rate for both channels.
    pub fn set_rate(&mut self, rate_hz: f64) {
        self.left.set_rate(rate_hz);
        self.right.set_rate(rate_hz);
    }

    /// Set the waveform type for both channels.
    pub fn set_waveform(&mut self, waveform: LfoWaveform) {
        self.left.set_waveform(waveform);
        self.right.set_waveform(waveform);
    }

    /// Set the phase offset between channels.
    ///
    /// # Arguments
    ///
    /// * `offset` - Phase offset (0.0 to 1.0)
    pub fn set_phase_offset(&mut self, offset: f64) {
        self.phase_offset = offset;
        let left_phase = self.left.phase();
        self.right.set_phase((left_phase + offset).fract());
    }

    /// Get the current phase offset.
    #[must_use]
    pub fn phase_offset(&self) -> f64 {
        self.phase_offset
    }

    /// Reset both LFOs.
    pub fn reset(&mut self) {
        self.left.reset();
        self.right.reset();
        self.right.set_phase(self.phase_offset);
    }

    /// Generate next stereo LFO sample values.
    ///
    /// Returns (left, right) values between -1.0 and 1.0.
    pub fn next(&mut self) -> (f64, f64) {
        (self.left.next(), self.right.next())
    }

    /// Generate next unipolar stereo LFO sample values.
    ///
    /// Returns (left, right) values between 0.0 and 1.0.
    pub fn next_unipolar(&mut self) -> (f64, f64) {
        (self.left.next_unipolar(), self.right.next_unipolar())
    }

    /// Get reference to left channel LFO.
    #[must_use]
    pub fn left(&self) -> &Lfo {
        &self.left
    }

    /// Get reference to right channel LFO.
    #[must_use]
    pub fn right(&self) -> &Lfo {
        &self.right
    }

    /// Get mutable reference to left channel LFO.
    pub fn left_mut(&mut self) -> &mut Lfo {
        &mut self.left
    }

    /// Get mutable reference to right channel LFO.
    pub fn right_mut(&mut self) -> &mut Lfo {
        &mut self.right
    }
}

/// Parameter smoother for avoiding audio artifacts from parameter changes.
///
/// Uses a simple one-pole lowpass filter to smooth parameter transitions.
#[derive(Clone, Debug)]
pub struct ParameterSmoother {
    /// Current smoothed value.
    current: f64,
    /// Target value.
    target: f64,
    /// Smoothing coefficient (0.0 to 1.0).
    coefficient: f64,
}

impl ParameterSmoother {
    /// Create a new parameter smoother.
    ///
    /// # Arguments
    ///
    /// * `initial_value` - Initial parameter value
    /// * `smoothing_time_ms` - Smoothing time in milliseconds
    /// * `sample_rate` - Audio sample rate in Hz
    #[must_use]
    pub fn new(initial_value: f64, smoothing_time_ms: f64, sample_rate: f64) -> Self {
        let coefficient = Self::calculate_coefficient(smoothing_time_ms, sample_rate);
        Self {
            current: initial_value,
            target: initial_value,
            coefficient,
        }
    }

    /// Calculate smoothing coefficient from time constant.
    fn calculate_coefficient(time_ms: f64, sample_rate: f64) -> f64 {
        let time_samples = time_ms * 0.001 * sample_rate;
        (-1.0 / time_samples).exp()
    }

    /// Set the target value.
    pub fn set_target(&mut self, target: f64) {
        self.target = target;
    }

    /// Get the current smoothed value.
    #[must_use]
    pub fn current(&self) -> f64 {
        self.current
    }

    /// Get the target value.
    #[must_use]
    pub fn target(&self) -> f64 {
        self.target
    }

    /// Process one sample and return the smoothed value.
    pub fn next(&mut self) -> f64 {
        self.current = self.current * self.coefficient + self.target * (1.0 - self.coefficient);
        self.current
    }

    /// Reset to a specific value immediately.
    pub fn reset(&mut self, value: f64) {
        self.current = value;
        self.target = value;
    }

    /// Check if the smoother has reached the target (within epsilon).
    #[must_use]
    pub fn is_stable(&self) -> bool {
        (self.current - self.target).abs() < 1e-6
    }

    /// Set smoothing time.
    pub fn set_smoothing_time(&mut self, time_ms: f64, sample_rate: f64) {
        self.coefficient = Self::calculate_coefficient(time_ms, sample_rate);
    }
}
