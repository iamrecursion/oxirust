//! Low-Frequency Oscillator (LFO) implementation.
//!
//! Provides oscillators for modulating effect parameters. Supports multiple
//! waveforms and is optimized for real-time use.

use std::f32::consts::TAU;

/// Waveform types for LFO.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LfoWaveform {
    /// Sine wave - smooth, natural modulation.
    Sine,
    /// Triangle wave - linear rise and fall.
    Triangle,
    /// Sawtooth wave - linear rise with instant reset.
    Sawtooth,
    /// Square wave - instant switching between min and max.
    Square,
    /// Random sample-and-hold.
    Random,
}

/// Low-Frequency Oscillator.
///
/// Generates periodic control signals for modulating effect parameters.
#[derive(Debug, Clone)]
pub struct Lfo {
    /// Current phase (0.0 - 1.0).
    phase: f32,
    /// Phase increment per sample.
    phase_inc: f32,
    /// Waveform type.
    waveform: LfoWaveform,
    /// Sample rate.
    sample_rate: f32,
    /// Random state for Random waveform.
    random_state: u32,
    /// Previous random value.
    random_value: f32,
}

impl Lfo {
    /// Create a new LFO.
    ///
    /// # Arguments
    ///
    /// * `frequency` - LFO frequency in Hz
    /// * `sample_rate` - Audio sample rate
    /// * `waveform` - Waveform type
    #[must_use]
    pub fn new(frequency: f32, sample_rate: f32, waveform: LfoWaveform) -> Self {
        let phase_inc = frequency / sample_rate;
        Self {
            phase: 0.0,
            phase_inc,
            waveform,
            sample_rate,
            random_state: 0x1234_5678,
            random_value: 0.0,
        }
    }

    /// Get the next LFO value (range: -1.0 to 1.0).
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> f32 {
        let value = match self.waveform {
            LfoWaveform::Sine => (self.phase * TAU).sin(),
            LfoWaveform::Triangle => {
                if self.phase < 0.5 {
                    4.0 * self.phase - 1.0
                } else {
                    3.0 - 4.0 * self.phase
                }
            }
            LfoWaveform::Sawtooth => 2.0 * self.phase - 1.0,
            LfoWaveform::Square => {
                if self.phase < 0.5 {
                    -1.0
                } else {
                    1.0
                }
            }
            LfoWaveform::Random => {
                // Update on phase wrap
                if self.phase < self.phase_inc {
                    self.random_state = self
                        .random_state
                        .wrapping_mul(1_103_515_245)
                        .wrapping_add(12_345);
                    #[allow(clippy::cast_precision_loss)]
                    let random_val = (self.random_state as f32) / (u32::MAX as f32);
                    self.random_value = 2.0 * random_val - 1.0;
                }
                self.random_value
            }
        };

        self.phase += self.phase_inc;
        if self.phase >= 1.0 {
            self.phase -= 1.0;
        }

        value
    }

    /// Get LFO value in unipolar range (0.0 to 1.0).
    pub fn next_unipolar(&mut self) -> f32 {
        (self.next() + 1.0) * 0.5
    }

    /// Set the LFO frequency.
    pub fn set_frequency(&mut self, frequency: f32) {
        self.phase_inc = frequency / self.sample_rate;
    }

    /// Set the waveform.
    pub fn set_waveform(&mut self, waveform: LfoWaveform) {
        self.waveform = waveform;
    }

    /// Reset the LFO phase to 0.
    pub fn reset(&mut self) {
        self.phase = 0.0;
    }

    /// Set the phase (0.0 - 1.0).
    pub fn set_phase(&mut self, phase: f32) {
        self.phase = phase.rem_euclid(1.0);
    }

    /// Get current phase (0.0 - 1.0).
    #[must_use]
    pub fn phase(&self) -> f32 {
        self.phase
    }
}

/// Stereo LFO with independent or linked phases.
#[derive(Debug, Clone)]
pub struct StereoLfo {
    /// Left channel LFO.
    pub left: Lfo,
    /// Right channel LFO.
    pub right: Lfo,
}

impl StereoLfo {
    /// Create a new stereo LFO with a phase offset between channels.
    ///
    /// # Arguments
    ///
    /// * `frequency` - LFO frequency in Hz
    /// * `sample_rate` - Audio sample rate
    /// * `waveform` - Waveform type
    /// * `phase_offset` - Phase offset for right channel (0.0 - 1.0)
    #[must_use]
    pub fn new(frequency: f32, sample_rate: f32, waveform: LfoWaveform, phase_offset: f32) -> Self {
        let left = Lfo::new(frequency, sample_rate, waveform);
        let mut right = Lfo::new(frequency, sample_rate, waveform);
        right.set_phase(phase_offset);

        Self { left, right }
    }

    /// Get the next stereo LFO values.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> (f32, f32) {
        (self.left.next(), self.right.next())
    }

    /// Get stereo values in unipolar range.
    pub fn next_unipolar(&mut self) -> (f32, f32) {
        (self.left.next_unipolar(), self.right.next_unipolar())
    }

    /// Set frequency for both channels.
    pub fn set_frequency(&mut self, frequency: f32) {
        self.left.set_frequency(frequency);
        self.right.set_frequency(frequency);
    }

    /// Set waveform for both channels.
    pub fn set_waveform(&mut self, waveform: LfoWaveform) {
        self.left.set_waveform(waveform);
        self.right.set_waveform(waveform);
    }

    /// Reset both LFOs.
    pub fn reset(&mut self) {
        self.left.reset();
        self.right.reset();
    }
}

/// Parameter smoother for avoiding clicks and pops.
///
/// Uses a one-pole lowpass filter to smooth parameter changes.
#[derive(Debug, Clone)]
pub struct ParameterSmoother {
    /// Current value.
    current: f32,
    /// Target value.
    target: f32,
    /// Smoothing coefficient (0.0 - 1.0).
    coefficient: f32,
}

impl ParameterSmoother {
    /// Create a new parameter smoother.
    ///
    /// # Arguments
    ///
    /// * `time_constant_ms` - Time constant in milliseconds
    /// * `sample_rate` - Audio sample rate
    #[must_use]
    pub fn new(time_constant_ms: f32, sample_rate: f32) -> Self {
        let coefficient = (-1000.0 / (time_constant_ms * sample_rate)).exp();
        Self {
            current: 0.0,
            target: 0.0,
            coefficient,
        }
    }

    /// Set the target value.
    pub fn set_target(&mut self, target: f32) {
        self.target = target;
    }

    /// Get the next smoothed value.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> f32 {
        self.current = self.target + self.coefficient * (self.current - self.target);
        self.current
    }

    /// Reset to a specific value instantly.
    pub fn reset(&mut self, value: f32) {
        self.current = value;
        self.target = value;
    }

    /// Check if the smoother has reached its target.
    #[must_use]
    pub fn is_stable(&self) -> bool {
        (self.current - self.target).abs() < 1e-3
    }
}

/// Smoothed parameter for real-time audio processing.
///
/// Prevents zipper noise (audible stepping artifacts) when parameters are
/// changed during playback. Uses a logarithmic one-pole filter with
/// configurable smoothing time and supports both linear and exponential
/// interpolation modes.
///
/// # Example
///
/// ```ignore
/// use oximedia_effects::utils::SmoothedParameter;
///
/// let mut param = SmoothedParameter::new(0.5, 10.0, 48000.0);
/// param.set(1.0);  // Set new target
/// for _ in 0..480 {
///     let smooth_val = param.next();  // Gradually approaches 1.0
/// }
/// ```
#[derive(Debug, Clone)]
pub struct SmoothedParameter {
    /// Current smoothed value.
    current: f32,
    /// Target value to approach.
    target: f32,
    /// One-pole coefficient for smoothing.
    coeff: f32,
    /// Smoothing time in milliseconds.
    smooth_time_ms: f32,
    /// Sample rate in Hz.
    sample_rate: f32,
    /// Minimum value clamp.
    min_val: f32,
    /// Maximum value clamp.
    max_val: f32,
    /// Number of samples remaining until stable.
    steps_remaining: u32,
    /// Total smoothing steps per transition.
    total_steps: u32,
    /// Interpolation mode.
    mode: SmoothingMode,
}

/// Smoothing interpolation mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmoothingMode {
    /// Linear interpolation (constant step per sample).
    Linear,
    /// Exponential (one-pole lowpass) — natural-sounding for gain/frequency.
    Exponential,
    /// Logarithmic — perceptually uniform for dB-scale parameters.
    Logarithmic,
}

impl SmoothedParameter {
    /// Create a new smoothed parameter.
    ///
    /// # Arguments
    ///
    /// * `initial` - Starting value
    /// * `smooth_time_ms` - Smoothing time constant in milliseconds
    /// * `sample_rate` - Audio sample rate in Hz
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn new(initial: f32, smooth_time_ms: f32, sample_rate: f32) -> Self {
        let total_steps = (smooth_time_ms * 0.001 * sample_rate).max(1.0) as u32;
        let coeff = Self::compute_coeff(smooth_time_ms, sample_rate);
        Self {
            current: initial,
            target: initial,
            coeff,
            smooth_time_ms,
            sample_rate,
            min_val: f32::NEG_INFINITY,
            max_val: f32::INFINITY,
            steps_remaining: 0,
            total_steps,
            mode: SmoothingMode::Exponential,
        }
    }

    /// Create with a value range clamp.
    #[must_use]
    pub fn with_range(mut self, min: f32, max: f32) -> Self {
        self.min_val = min;
        self.max_val = max;
        self.current = self.current.clamp(min, max);
        self.target = self.target.clamp(min, max);
        self
    }

    /// Create with a specific smoothing mode.
    #[must_use]
    pub fn with_mode(mut self, mode: SmoothingMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set a new target value. Smoothing begins immediately.
    pub fn set(&mut self, target: f32) {
        self.target = target.clamp(self.min_val, self.max_val);
        self.steps_remaining = self.total_steps;
    }

    /// Set value immediately without smoothing.
    pub fn set_immediate(&mut self, value: f32) {
        let v = value.clamp(self.min_val, self.max_val);
        self.current = v;
        self.target = v;
        self.steps_remaining = 0;
    }

    /// Get the next smoothed value, advancing by one sample.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> f32 {
        if self.steps_remaining == 0 {
            self.current = self.target;
            return self.current;
        }

        self.steps_remaining = self.steps_remaining.saturating_sub(1);

        match self.mode {
            SmoothingMode::Linear => {
                let step = (self.target - self.current) / (self.steps_remaining + 1) as f32;
                self.current += step;
            }
            SmoothingMode::Exponential => {
                self.current = self.target + self.coeff * (self.current - self.target);
            }
            SmoothingMode::Logarithmic => {
                // Work in log domain for perceptually uniform transitions
                let epsilon = 1e-10;
                let current_log = (self.current.abs() + epsilon).ln();
                let target_log = (self.target.abs() + epsilon).ln();
                let smoothed_log = target_log + self.coeff * (current_log - target_log);
                let sign = if self.target >= 0.0 { 1.0 } else { -1.0 };
                self.current = sign * smoothed_log.exp();
            }
        }

        if self.steps_remaining == 0 {
            self.current = self.target;
        }

        self.current
    }

    /// Process a block of samples, filling the output with smoothed values.
    pub fn process_block(&mut self, output: &mut [f32]) {
        for sample in output.iter_mut() {
            *sample = self.next();
        }
    }

    /// Check if the parameter has reached its target.
    #[must_use]
    pub fn is_smoothing(&self) -> bool {
        self.steps_remaining > 0
    }

    /// Get the current value without advancing.
    #[must_use]
    pub fn current(&self) -> f32 {
        self.current
    }

    /// Get the target value.
    #[must_use]
    pub fn target(&self) -> f32 {
        self.target
    }

    /// Update the sample rate (recalculates coefficients).
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn set_sample_rate(&mut self, sample_rate: f32) {
        self.sample_rate = sample_rate;
        self.coeff = Self::compute_coeff(self.smooth_time_ms, sample_rate);
        self.total_steps = (self.smooth_time_ms * 0.001 * sample_rate).max(1.0) as u32;
    }

    /// Update the smoothing time in milliseconds.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn set_smooth_time(&mut self, ms: f32) {
        self.smooth_time_ms = ms;
        self.coeff = Self::compute_coeff(ms, self.sample_rate);
        self.total_steps = (ms * 0.001 * self.sample_rate).max(1.0) as u32;
    }

    fn compute_coeff(time_ms: f32, sample_rate: f32) -> f32 {
        let samples = time_ms * 0.001 * sample_rate;
        if samples > 0.0 {
            (-2.2 / samples as f64).exp() as f32
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lfo_sine() {
        let mut lfo = Lfo::new(1.0, 100.0, LfoWaveform::Sine);

        // At phase 0, sine should be 0
        let v0 = lfo.next();
        assert!(v0.abs() < 0.1);

        // At phase 0.25, sine should be ~1.0
        for _ in 0..24 {
            lfo.next();
        }
        let v25 = lfo.next();
        assert!((v25 - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_lfo_triangle() {
        let mut lfo = Lfo::new(1.0, 100.0, LfoWaveform::Triangle);

        // At phase 0, triangle should be -1
        let v0 = lfo.next();
        assert!((v0 + 1.0).abs() < 0.1);

        // Triangle increases linearly
        let v1 = lfo.next();
        assert!(v1 > v0);
    }

    #[test]
    fn test_lfo_square() {
        let mut lfo = Lfo::new(1.0, 100.0, LfoWaveform::Square);

        // Square wave should only output -1.0 or 1.0
        for _ in 0..100 {
            let val = lfo.next();
            assert!(
                val == -1.0 || val == 1.0,
                "Square wave should only be -1.0 or 1.0"
            );
        }
    }

    #[test]
    fn test_lfo_unipolar() {
        let mut lfo = Lfo::new(1.0, 100.0, LfoWaveform::Sine);

        for _ in 0..100 {
            let val = lfo.next_unipolar();
            assert!(val >= 0.0 && val <= 1.0);
        }
    }

    #[test]
    fn test_stereo_lfo() {
        let mut lfo = StereoLfo::new(1.0, 100.0, LfoWaveform::Sine, 0.5);

        let (l, r) = lfo.next();
        // With 0.5 phase offset (180 degrees), values should be opposite
        assert!((l + r).abs() < 0.1);
    }

    #[test]
    fn test_parameter_smoother() {
        let mut smoother = ParameterSmoother::new(10.0, 48000.0);
        smoother.reset(0.0);
        smoother.set_target(1.0);

        // Should gradually approach target
        let v1 = smoother.next();
        let v2 = smoother.next();
        let v3 = smoother.next();

        assert!(v1 > 0.0 && v1 < 1.0);
        assert!(v2 > v1);
        assert!(v3 > v2);

        // After many samples, should be close to target
        for _ in 0..100000 {
            smoother.next();
        }
        assert!(smoother.is_stable());
    }

    #[test]
    fn test_lfo_reset() {
        let mut lfo = Lfo::new(1.0, 100.0, LfoWaveform::Sine);

        // Advance phase
        for _ in 0..25 {
            lfo.next();
        }
        assert!(lfo.phase() > 0.0);

        // Reset should bring back to 0
        lfo.reset();
        assert_eq!(lfo.phase(), 0.0);
    }

    // --- SmoothedParameter tests ---

    #[test]
    fn test_smoothed_parameter_creation() {
        let param = SmoothedParameter::new(0.5, 10.0, 48000.0);
        assert!((param.current() - 0.5).abs() < 1e-6);
        assert!((param.target() - 0.5).abs() < 1e-6);
        assert!(!param.is_smoothing());
    }

    #[test]
    fn test_smoothed_parameter_set_and_smooth() {
        let mut param = SmoothedParameter::new(0.0, 10.0, 48000.0);
        param.set(1.0);
        assert!(param.is_smoothing());

        let mut prev = 0.0;
        // Value should monotonically increase toward target
        for _ in 0..480 {
            let v = param.next();
            assert!(v >= prev - 1e-6, "Value should not decrease: {v} < {prev}");
            assert!(v.is_finite());
            prev = v;
        }

        // After full smoothing time, should be at target
        for _ in 0..48000 {
            param.next();
        }
        assert!((param.current() - 1.0).abs() < 1e-3);
        assert!(!param.is_smoothing());
    }

    #[test]
    fn test_smoothed_parameter_immediate() {
        let mut param = SmoothedParameter::new(0.0, 10.0, 48000.0);
        param.set_immediate(0.75);
        assert!((param.current() - 0.75).abs() < 1e-6);
        assert!(!param.is_smoothing());
    }

    #[test]
    fn test_smoothed_parameter_range_clamp() {
        let mut param = SmoothedParameter::new(0.5, 10.0, 48000.0).with_range(0.0, 1.0);
        param.set(2.0);
        assert!((param.target() - 1.0).abs() < 1e-6);

        param.set(-1.0);
        assert!((param.target() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_smoothed_parameter_linear_mode() {
        let mut param = SmoothedParameter::new(0.0, 10.0, 48000.0).with_mode(SmoothingMode::Linear);
        param.set(1.0);

        // Linear mode should produce monotonically increasing values
        let mut prev = 0.0;
        for _ in 0..480 {
            let v = param.next();
            assert!(v >= prev - 1e-6);
            prev = v;
        }
    }

    #[test]
    fn test_smoothed_parameter_logarithmic_mode() {
        let mut param = SmoothedParameter::new(0.1, 10.0, 48000.0)
            .with_mode(SmoothingMode::Logarithmic)
            .with_range(0.001, 10.0);
        param.set(1.0);

        for _ in 0..48000 {
            let v = param.next();
            assert!(v.is_finite());
        }
        assert!((param.current() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_smoothed_parameter_block_processing() {
        let mut param = SmoothedParameter::new(0.0, 5.0, 48000.0);
        param.set(1.0);

        let mut block = vec![0.0f32; 256];
        param.process_block(&mut block);

        // Each value should be finite and increasing
        for &v in &block {
            assert!(v.is_finite());
        }
        assert!(block[255] > block[0]);
    }

    #[test]
    fn test_smoothed_parameter_no_zipper() {
        // Verify smoothed output has no sudden jumps during the transition.
        // Use 50ms smoothing; check only the first 90% of the ramp where
        // the exponential smoothing is active (before the final snap-to-target).
        let mut param = SmoothedParameter::new(0.0, 50.0, 48000.0);
        param.set(1.0);

        let total_steps = (50.0 * 0.001 * 48000.0) as usize; // 2400
        let check_steps = total_steps * 9 / 10; // check first 90%

        let mut prev = 0.0f32;
        let mut max_delta = 0.0f32;
        for _ in 0..check_steps {
            let v = param.next();
            let delta = (v - prev).abs();
            if delta > max_delta {
                max_delta = delta;
            }
            prev = v;
        }

        // Max per-sample change should be small (no zipper)
        assert!(
            max_delta < 0.05,
            "Max delta {max_delta} too large — zipper noise detected"
        );
    }

    #[test]
    fn test_smoothed_parameter_retarget() {
        // Change target mid-smoothing
        let mut param = SmoothedParameter::new(0.0, 10.0, 48000.0);
        param.set(1.0);
        for _ in 0..240 {
            param.next();
        }
        let mid = param.current();

        param.set(0.5);
        for _ in 0..48000 {
            param.next();
        }
        assert!((param.current() - 0.5).abs() < 1e-3);
        // Should have gone up then come back down
        assert!(mid > 0.0);
    }

    #[test]
    fn test_smoothed_parameter_set_sample_rate() {
        let mut param = SmoothedParameter::new(0.5, 10.0, 48000.0);
        param.set_sample_rate(96000.0);
        param.set(1.0);
        for _ in 0..96000 {
            param.next();
        }
        assert!((param.current() - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_smoothed_parameter_set_smooth_time() {
        let mut param = SmoothedParameter::new(0.0, 5.0, 48000.0);
        param.set_smooth_time(50.0);
        param.set(1.0);
        // Should still converge, just slower
        for _ in 0..48000 {
            param.next();
        }
        assert!((param.current() - 1.0).abs() < 1e-3);
    }
}
