//! Flanger effect — short delay with LFO-modulated sweep and feedback.
//!
//! A flanger creates a comb-filter effect by mixing a signal with a slightly
//! delayed copy of itself, where the delay time is continuously swept by a
//! low-frequency oscillator (LFO). The characteristic "jet plane" sound is
//! produced by the constructive and destructive interference between the
//! direct and delayed signals.
//!
//! # Example
//!
//! ```
//! use oximedia_effects::flanger::{FlangerConfig, Flanger};
//!
//! let config = FlangerConfig::default();
//! let mut flanger = Flanger::new(config, 48_000.0);
//!
//! let mut buffer = vec![0.0_f32; 512];
//! buffer[0] = 1.0; // impulse
//! for sample in buffer.iter_mut() {
//!     *sample = flanger.apply_sample(*sample);
//! }
//! ```

#![allow(dead_code)]

/// LFO waveform shape for the delay modulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LfoShape {
    /// Sine wave (smooth, natural-sounding).
    Sine,
    /// Triangle wave (linear sweep, slightly harsher).
    Triangle,
    /// Sawtooth wave (asymmetric sweep).
    Sawtooth,
}

/// Configuration for the [`Flanger`] effect.
#[derive(Debug, Clone)]
pub struct FlangerConfig {
    /// Minimum delay in milliseconds (typically 0.1 – 5.0 ms).
    pub min_delay_ms: f32,
    /// Maximum delay in milliseconds (typically 1.0 – 20.0 ms).
    pub max_delay_ms: f32,
    /// LFO rate in Hz (typically 0.1 – 10.0 Hz).
    pub rate_hz: f32,
    /// Feedback amount (–1.0 to +1.0). Positive = comb resonance.
    pub feedback: f32,
    /// Wet/dry mix (0.0 = dry only, 1.0 = wet only).
    pub mix: f32,
    /// LFO waveform shape.
    pub lfo_shape: LfoShape,
    /// If `true`, the feedback polarity is inverted (negative flange).
    pub invert: bool,
}

impl Default for FlangerConfig {
    fn default() -> Self {
        Self {
            min_delay_ms: 0.1,
            max_delay_ms: 7.0,
            rate_hz: 0.5,
            feedback: 0.7,
            mix: 0.5,
            lfo_shape: LfoShape::Sine,
            invert: false,
        }
    }
}

impl FlangerConfig {
    /// Classic slow flange preset.
    #[must_use]
    pub fn slow_flange() -> Self {
        Self {
            min_delay_ms: 0.1,
            max_delay_ms: 10.0,
            rate_hz: 0.25,
            feedback: 0.8,
            mix: 0.5,
            lfo_shape: LfoShape::Sine,
            invert: false,
        }
    }

    /// Fast jet-plane flange preset.
    #[must_use]
    pub fn jet_flange() -> Self {
        Self {
            min_delay_ms: 0.1,
            max_delay_ms: 5.0,
            rate_hz: 4.0,
            feedback: 0.85,
            mix: 0.6,
            lfo_shape: LfoShape::Sine,
            invert: false,
        }
    }
}

/// Internal per-channel state for [`Flanger`].
#[derive(Debug, Clone)]
pub struct FlangerState {
    /// Current LFO phase in radians.
    pub lfo_phase: f32,
    /// Current feedback sample.
    pub feedback_sample: f32,
}

impl FlangerState {
    /// Create a new zeroed channel state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            lfo_phase: 0.0,
            feedback_sample: 0.0,
        }
    }

    /// Reset this channel state.
    pub fn reset(&mut self) {
        self.lfo_phase = 0.0;
        self.feedback_sample = 0.0;
    }
}

impl Default for FlangerState {
    fn default() -> Self {
        Self::new()
    }
}

/// A flanger effect with configurable LFO and feedback.
pub struct Flanger {
    config: FlangerConfig,
    sample_rate: f32,
    /// Delay buffer (power-of-two for fast modulo).
    buffer: Vec<f32>,
    mask: usize,
    write_pos: usize,
    /// LFO phase accumulator (radians).
    lfo_phase: f32,
    lfo_inc: f32,
    /// Delay sweep range in samples.
    min_delay_samples: f32,
    max_delay_samples: f32,
    /// Feedback memory.
    feedback_memory: f32,
}

impl Flanger {
    /// Create a new flanger at the given sample rate.
    #[must_use]
    pub fn new(config: FlangerConfig, sample_rate: f32) -> Self {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let max_samples = (config.max_delay_ms * 0.001 * sample_rate * 2.0).ceil() as usize;
        let capacity = max_samples.next_power_of_two().max(4);

        let min_delay_samples = config.min_delay_ms * 0.001 * sample_rate;
        let max_delay_samples = config.max_delay_ms * 0.001 * sample_rate;
        let lfo_inc = 2.0 * std::f32::consts::PI * config.rate_hz / sample_rate;

        Self {
            config,
            sample_rate,
            buffer: vec![0.0; capacity],
            mask: capacity - 1,
            write_pos: 0,
            lfo_phase: 0.0,
            lfo_inc,
            min_delay_samples,
            max_delay_samples,
            feedback_memory: 0.0,
        }
    }

    /// Compute the current LFO value in [0, 1].
    fn lfo_value(&self) -> f32 {
        match self.config.lfo_shape {
            LfoShape::Sine => (self.lfo_phase.sin() + 1.0) * 0.5,
            LfoShape::Triangle => {
                let t = self.lfo_phase / (2.0 * std::f32::consts::PI);
                let t = t - t.floor();
                if t < 0.5 {
                    t * 2.0
                } else {
                    (1.0 - t) * 2.0
                }
            }
            LfoShape::Sawtooth => {
                let t = self.lfo_phase / (2.0 * std::f32::consts::PI);
                t - t.floor()
            }
        }
    }

    /// Read from the delay buffer using linear interpolation.
    #[allow(clippy::cast_precision_loss)]
    fn read_interpolated(&self, delay_samples: f32) -> f32 {
        let d = delay_samples.clamp(0.0, (self.buffer.len() - 1) as f32);
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let d_int = d as usize;
        let d_frac = d - d_int as f32;

        let idx0 = self.write_pos.wrapping_sub(d_int).wrapping_sub(1) & self.mask;
        let idx1 = self.write_pos.wrapping_sub(d_int).wrapping_sub(2) & self.mask;
        let s0 = self.buffer[idx0];
        let s1 = self.buffer[idx1];
        s0 + d_frac * (s1 - s0)
    }

    /// Process a single mono sample through the flanger.
    #[must_use]
    pub fn apply_sample(&mut self, input: f32) -> f32 {
        // Compute modulated delay
        let lfo = self.lfo_value();
        let delay_samples =
            self.min_delay_samples + lfo * (self.max_delay_samples - self.min_delay_samples);

        // Write input + feedback to buffer
        let fb = if self.config.invert {
            -self.config.feedback
        } else {
            self.config.feedback
        };
        self.buffer[self.write_pos] = input + self.feedback_memory * fb;
        self.write_pos = (self.write_pos + 1) & self.mask;

        // Read delayed sample
        let delayed = self.read_interpolated(delay_samples);
        self.feedback_memory = delayed;

        // Advance LFO
        self.lfo_phase += self.lfo_inc;
        if self.lfo_phase >= 2.0 * std::f32::consts::PI {
            self.lfo_phase -= 2.0 * std::f32::consts::PI;
        }

        // Mix dry and wet
        input * (1.0 - self.config.mix) + delayed * self.config.mix
    }

    /// Process a buffer of mono samples in-place.
    pub fn process(&mut self, buffer: &mut [f32]) {
        for s in buffer.iter_mut() {
            *s = self.apply_sample(*s);
        }
    }

    /// Process stereo samples in-place.
    pub fn process_stereo(&mut self, left: &mut [f32], right: &mut [f32]) {
        let len = left.len().min(right.len());
        for i in 0..len {
            left[i] = self.apply_sample(left[i]);
            right[i] = self.apply_sample(right[i]);
        }
    }

    /// Reset all internal state.
    pub fn reset(&mut self) {
        for s in &mut self.buffer {
            *s = 0.0;
        }
        self.write_pos = 0;
        self.lfo_phase = 0.0;
        self.feedback_memory = 0.0;
    }

    /// Update the LFO rate without re-creating the effect.
    pub fn set_rate_hz(&mut self, rate_hz: f32) {
        self.config.rate_hz = rate_hz.max(0.001);
        self.lfo_inc = 2.0 * std::f32::consts::PI * self.config.rate_hz / self.sample_rate;
    }

    /// Update the mix level.
    pub fn set_mix(&mut self, mix: f32) {
        self.config.mix = mix.clamp(0.0, 1.0);
    }

    /// Return the current LFO phase in radians.
    #[must_use]
    pub fn lfo_phase(&self) -> f32 {
        self.lfo_phase
    }

    /// Set the wet/dry mix ratio via the `AudioEffect` trait.
    ///
    /// Delegates to [`set_mix`](Self::set_mix) so the internal `config.mix` field
    /// is the single source of truth.
    pub fn set_wet_dry_mix(&mut self, wet: f32) {
        self.set_mix(wet);
    }

    /// Return the current wet level.
    #[must_use]
    pub fn wet_level(&self) -> f32 {
        self.config.mix
    }

    /// Return a snapshot of the internal state.
    #[must_use]
    pub fn state(&self) -> FlangerState {
        FlangerState {
            lfo_phase: self.lfo_phase,
            feedback_sample: self.feedback_memory,
        }
    }
}

impl crate::AudioEffect for Flanger {
    const EFFECT_ID: &'static str = "flanger";

    fn process_sample(&mut self, input: f32) -> f32 {
        self.apply_sample(input)
    }

    fn process_sample_stereo(&mut self, left: f32, right: f32) -> (f32, f32) {
        (self.apply_sample(left), self.apply_sample(right))
    }

    fn reset(&mut self) {
        self.reset();
    }

    fn set_sample_rate(&mut self, sample_rate: f32) {
        self.sample_rate = sample_rate;
        self.lfo_inc = 2.0 * std::f32::consts::PI * self.config.rate_hz / sample_rate;
        self.min_delay_samples = self.config.min_delay_ms * 0.001 * sample_rate;
        self.max_delay_samples = self.config.max_delay_ms * 0.001 * sample_rate;
    }

    fn set_wet_dry(&mut self, wet: f32) {
        self.set_mix(wet);
    }

    fn wet_dry(&self) -> f32 {
        self.config.mix
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_flanger() -> Flanger {
        Flanger::new(FlangerConfig::default(), 48_000.0)
    }

    #[test]
    fn test_flanger_new_buffer_power_of_two() {
        let f = make_flanger();
        assert!(f.buffer.len().is_power_of_two());
    }

    #[test]
    fn test_flanger_silence_in_silence_out_approx() {
        let mut f = make_flanger();
        // All zeros → output should remain near zero
        for _ in 0..512 {
            let out = f.apply_sample(0.0);
            assert!(out.abs() < 1e-6);
        }
    }

    #[test]
    fn test_flanger_apply_sample_no_panic() {
        let mut f = make_flanger();
        for i in 0..1024 {
            let _ = f.apply_sample((i as f32 * 0.01).sin());
        }
    }

    #[test]
    fn test_flanger_mix_zero_bypasses() {
        let config = FlangerConfig {
            mix: 0.0,
            feedback: 0.0,
            ..FlangerConfig::default()
        };
        let mut f = Flanger::new(config, 48_000.0);
        let out = f.apply_sample(0.5);
        assert!((out - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_flanger_lfo_phase_advances() {
        let mut f = make_flanger();
        let p0 = f.lfo_phase();
        let _ = f.apply_sample(0.0);
        let p1 = f.lfo_phase();
        assert!(p1 > p0);
    }

    #[test]
    fn test_flanger_reset_clears_buffer() {
        let mut f = make_flanger();
        for _ in 0..256 {
            let _ = f.apply_sample(1.0);
        }
        f.reset();
        for &v in f.buffer.iter() {
            assert_eq!(v, 0.0);
        }
        assert_eq!(f.write_pos, 0);
        assert_eq!(f.lfo_phase, 0.0);
    }

    #[test]
    fn test_flanger_set_rate_hz() {
        let mut f = make_flanger();
        f.set_rate_hz(2.0);
        assert!((f.config.rate_hz - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_flanger_set_mix() {
        let mut f = make_flanger();
        f.set_mix(0.8);
        assert!((f.config.mix - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_flanger_set_mix_clamped() {
        let mut f = make_flanger();
        f.set_mix(2.0);
        assert_eq!(f.config.mix, 1.0);
    }

    #[test]
    fn test_flanger_state_snapshot() {
        let mut f = make_flanger();
        let _ = f.apply_sample(0.5);
        let st = f.state();
        assert!(st.lfo_phase >= 0.0);
    }

    #[test]
    fn test_flanger_lfo_shape_triangle() {
        let config = FlangerConfig {
            lfo_shape: LfoShape::Triangle,
            ..FlangerConfig::default()
        };
        let mut f = Flanger::new(config, 48_000.0);
        for _ in 0..1024 {
            let _ = f.apply_sample(0.1);
        }
        // No panic and LFO continues
        assert!(f.lfo_phase() >= 0.0);
    }

    #[test]
    fn test_flanger_lfo_shape_sawtooth() {
        let config = FlangerConfig {
            lfo_shape: LfoShape::Sawtooth,
            ..FlangerConfig::default()
        };
        let mut f = Flanger::new(config, 48_000.0);
        for _ in 0..1024 {
            let _ = f.apply_sample(0.1);
        }
        assert!(f.lfo_phase() >= 0.0);
    }

    #[test]
    fn test_flanger_invert_flag() {
        let config_normal = FlangerConfig {
            invert: false,
            ..FlangerConfig::default()
        };
        let config_invert = FlangerConfig {
            invert: true,
            ..FlangerConfig::default()
        };
        let mut fn_ = Flanger::new(config_normal, 48_000.0);
        let mut fi = Flanger::new(config_invert, 48_000.0);
        // After one identical impulse they may diverge in later samples
        let _ = fn_.apply_sample(1.0);
        let _ = fi.apply_sample(1.0);
        // Just check no panic; detailed equality not guaranteed
    }

    #[test]
    fn test_flanger_process_buffer() {
        let mut f = make_flanger();
        let mut buf = vec![0.1_f32; 256];
        f.process(&mut buf);
        assert_eq!(buf.len(), 256);
    }

    #[test]
    fn test_flanger_slow_flange_preset() {
        let c = FlangerConfig::slow_flange();
        assert!(c.rate_hz < 1.0);
        assert!(c.max_delay_ms > 5.0);
    }

    #[test]
    fn test_flanger_jet_preset() {
        let c = FlangerConfig::jet_flange();
        assert!(c.rate_hz > 2.0);
    }

    #[test]
    fn test_flanger_state_default() {
        let st = FlangerState::default();
        assert_eq!(st.lfo_phase, 0.0);
        assert_eq!(st.feedback_sample, 0.0);
    }
}
