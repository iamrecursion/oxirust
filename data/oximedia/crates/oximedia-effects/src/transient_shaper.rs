#![allow(dead_code)]

//! Transient shaping effect for controlling attack and sustain.
//!
//! A transient shaper splits the audio signal into its transient (attack)
//! and sustain (body) components, then lets you independently boost or
//! cut each. This is useful for adding punch to drums, taming harsh
//! attacks on acoustic guitars, or reshaping the envelope of any sound.

/// Configuration for the transient shaper.
#[derive(Debug, Clone)]
pub struct TransientShaperConfig {
    /// Attack gain in dB (positive = boost transients, negative = soften).
    pub attack_db: f32,
    /// Sustain gain in dB (positive = boost body, negative = tighten).
    pub sustain_db: f32,
    /// Fast envelope time constant in ms (tracks transients).
    pub fast_time_ms: f32,
    /// Slow envelope time constant in ms (tracks sustain).
    pub slow_time_ms: f32,
    /// Output gain in dB.
    pub output_gain_db: f32,
    /// Sample rate in Hz.
    pub sample_rate: f32,
}

impl Default for TransientShaperConfig {
    fn default() -> Self {
        Self {
            attack_db: 0.0,
            sustain_db: 0.0,
            fast_time_ms: 1.0,
            slow_time_ms: 20.0,
            output_gain_db: 0.0,
            sample_rate: 48000.0,
        }
    }
}

impl TransientShaperConfig {
    /// Create a new config with the given sample rate.
    #[must_use]
    pub fn new(sample_rate: f32) -> Self {
        Self {
            sample_rate,
            ..Default::default()
        }
    }

    /// Set attack gain.
    #[must_use]
    pub fn with_attack(mut self, db: f32) -> Self {
        self.attack_db = db.clamp(-24.0, 24.0);
        self
    }

    /// Set sustain gain.
    #[must_use]
    pub fn with_sustain(mut self, db: f32) -> Self {
        self.sustain_db = db.clamp(-24.0, 24.0);
        self
    }

    /// Set output gain.
    #[must_use]
    pub fn with_output_gain(mut self, db: f32) -> Self {
        self.output_gain_db = db.clamp(-24.0, 24.0);
        self
    }

    /// Set fast envelope time.
    #[must_use]
    pub fn with_fast_time(mut self, ms: f32) -> Self {
        self.fast_time_ms = ms.clamp(0.1, 10.0);
        self
    }

    /// Set slow envelope time.
    #[must_use]
    pub fn with_slow_time(mut self, ms: f32) -> Self {
        self.slow_time_ms = ms.clamp(5.0, 200.0);
        self
    }
}

/// Convert dB to linear gain.
#[allow(clippy::cast_precision_loss)]
fn db_to_lin(db: f32) -> f32 {
    10.0f32.powf(db / 20.0)
}

/// Compute one-pole coefficient from time constant in ms.
#[allow(clippy::cast_precision_loss)]
fn coeff_from_ms(ms: f32, sr: f32) -> f32 {
    if ms <= 0.0 || sr <= 0.0 {
        return 0.0;
    }
    let n = ms * 0.001 * sr;
    (-1.0f32 / n).exp()
}

/// A single-channel envelope follower with configurable time constant.
#[derive(Debug, Clone)]
pub struct EnvelopeFollower {
    /// Current value.
    value: f32,
    /// Attack coefficient.
    attack_coeff: f32,
    /// Release coefficient.
    release_coeff: f32,
}

impl EnvelopeFollower {
    /// Create a new envelope follower.
    #[must_use]
    pub fn new(time_ms: f32, sample_rate: f32) -> Self {
        let c = coeff_from_ms(time_ms, sample_rate);
        Self {
            value: 0.0,
            attack_coeff: c,
            release_coeff: c,
        }
    }

    /// Process one sample (absolute value input).
    pub fn process(&mut self, input_abs: f32) -> f32 {
        if input_abs > self.value {
            self.value = self.attack_coeff * self.value + (1.0 - self.attack_coeff) * input_abs;
        } else {
            self.value = self.release_coeff * self.value + (1.0 - self.release_coeff) * input_abs;
        }
        self.value
    }

    /// Reset to zero.
    pub fn reset(&mut self) {
        self.value = 0.0;
    }

    /// Get current value.
    #[must_use]
    pub fn value(&self) -> f32 {
        self.value
    }
}

/// The transient shaper processor.
#[derive(Debug)]
pub struct TransientShaper {
    config: TransientShaperConfig,
    /// Fast envelope (tracks transients).
    fast_env: EnvelopeFollower,
    /// Slow envelope (tracks sustain).
    slow_env: EnvelopeFollower,
    /// Linear attack gain.
    attack_gain: f32,
    /// Linear sustain gain.
    sustain_gain: f32,
    /// Linear output gain.
    output_gain: f32,
}

impl TransientShaper {
    /// Create a new transient shaper.
    #[must_use]
    pub fn new(config: TransientShaperConfig) -> Self {
        let fast_env = EnvelopeFollower::new(config.fast_time_ms, config.sample_rate);
        let slow_env = EnvelopeFollower::new(config.slow_time_ms, config.sample_rate);
        let attack_gain = db_to_lin(config.attack_db);
        let sustain_gain = db_to_lin(config.sustain_db);
        let output_gain = db_to_lin(config.output_gain_db);
        Self {
            config,
            fast_env,
            slow_env,
            attack_gain,
            sustain_gain,
            output_gain,
        }
    }

    /// Process a single sample.
    pub fn process_sample(&mut self, input: f32) -> f32 {
        let abs_in = input.abs();
        let fast = self.fast_env.process(abs_in);
        let slow = self.slow_env.process(abs_in);

        // Transient component: difference between fast and slow envelopes
        let transient_diff = (fast - slow).max(0.0);
        // Sustain component: the slow envelope
        let sustain_level = slow;

        // Apply gains to components
        let transient_boost = transient_diff * (self.attack_gain - 1.0);
        let sustain_boost = sustain_level * (self.sustain_gain - 1.0);

        // Reconstruct: original + modifications scaled by sign
        let sign = if input >= 0.0 { 1.0 } else { -1.0 };
        let output = input + sign * (transient_boost + sustain_boost);
        output * self.output_gain
    }

    /// Process a buffer in-place.
    pub fn process(&mut self, buffer: &mut [f32]) {
        for sample in buffer.iter_mut() {
            *sample = self.process_sample(*sample);
        }
    }

    /// Process stereo buffers.
    pub fn process_stereo(&mut self, left: &mut [f32], right: &mut [f32]) {
        let len = left.len().min(right.len());
        for i in 0..len {
            left[i] = self.process_sample(left[i]);
            right[i] = self.process_sample(right[i]);
        }
    }

    /// Reset internal state.
    pub fn reset(&mut self) {
        self.fast_env.reset();
        self.slow_env.reset();
    }

    /// Update attack gain in dB.
    pub fn set_attack_db(&mut self, db: f32) {
        self.config.attack_db = db.clamp(-24.0, 24.0);
        self.attack_gain = db_to_lin(self.config.attack_db);
    }

    /// Update sustain gain in dB.
    pub fn set_sustain_db(&mut self, db: f32) {
        self.config.sustain_db = db.clamp(-24.0, 24.0);
        self.sustain_gain = db_to_lin(self.config.sustain_db);
    }

    /// Get the current fast envelope level.
    #[must_use]
    pub fn fast_envelope(&self) -> f32 {
        self.fast_env.value()
    }

    /// Get the current slow envelope level.
    #[must_use]
    pub fn slow_envelope(&self) -> f32 {
        self.slow_env.value()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_db_to_lin_zero() {
        assert!((db_to_lin(0.0) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_db_to_lin_positive() {
        let v = db_to_lin(6.0);
        assert!(v > 1.5 && v < 2.1);
    }

    #[test]
    fn test_db_to_lin_negative() {
        let v = db_to_lin(-6.0);
        assert!(v > 0.4 && v < 0.6);
    }

    #[test]
    fn test_coeff_from_ms() {
        let c = coeff_from_ms(10.0, 48000.0);
        assert!(c > 0.0 && c < 1.0);
    }

    #[test]
    fn test_coeff_from_ms_zero() {
        let c = coeff_from_ms(0.0, 48000.0);
        assert!((c - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_envelope_follower_tracks_up() {
        let mut env = EnvelopeFollower::new(1.0, 48000.0);
        for _ in 0..480 {
            env.process(1.0);
        }
        assert!(env.value() > 0.9);
    }

    #[test]
    fn test_envelope_follower_tracks_down() {
        let mut env = EnvelopeFollower::new(1.0, 48000.0);
        for _ in 0..480 {
            env.process(1.0);
        }
        for _ in 0..4800 {
            env.process(0.0);
        }
        assert!(env.value() < 0.1);
    }

    #[test]
    fn test_envelope_follower_reset() {
        let mut env = EnvelopeFollower::new(1.0, 48000.0);
        env.process(1.0);
        env.reset();
        assert!((env.value() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_shaper_passthrough() {
        let config = TransientShaperConfig {
            attack_db: 0.0,
            sustain_db: 0.0,
            output_gain_db: 0.0,
            ..TransientShaperConfig::new(48000.0)
        };
        let mut shaper = TransientShaper::new(config);
        // With 0 dB on everything, output should be close to input
        // after the envelope settles
        let mut buf = vec![0.5f32; 4800];
        shaper.process(&mut buf);
        // Last samples should be close to 0.5
        let last = buf[buf.len() - 1];
        assert!((last - 0.5).abs() < 0.15, "got {last}");
    }

    #[test]
    fn test_shaper_attack_boost() {
        let config = TransientShaperConfig::new(48000.0).with_attack(12.0);
        let mut shaper = TransientShaper::new(config);
        // Feed a transient: silence then impulse
        let silence = vec![0.0f32; 480];
        let mut impulse = vec![0.0f32; 480];
        impulse[0] = 1.0;
        // Process silence
        for &s in &silence {
            shaper.process_sample(s);
        }
        // Process impulse — first sample should be boosted
        let out = shaper.process_sample(impulse[0]);
        assert!(
            out.abs() >= 1.0,
            "boosted transient should be >= 1.0, got {out}"
        );
    }

    #[test]
    fn test_shaper_reset() {
        let mut shaper = TransientShaper::new(TransientShaperConfig::default());
        shaper.process_sample(1.0);
        shaper.reset();
        assert!((shaper.fast_envelope() - 0.0).abs() < 1e-10);
        assert!((shaper.slow_envelope() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_set_attack_db() {
        let mut shaper = TransientShaper::new(TransientShaperConfig::default());
        shaper.set_attack_db(6.0);
        assert!((shaper.config.attack_db - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_set_sustain_db() {
        let mut shaper = TransientShaper::new(TransientShaperConfig::default());
        shaper.set_sustain_db(-3.0);
        assert!((shaper.config.sustain_db - (-3.0)).abs() < 1e-5);
    }

    #[test]
    fn test_config_builder() {
        let cfg = TransientShaperConfig::new(44100.0)
            .with_attack(6.0)
            .with_sustain(-3.0)
            .with_output_gain(1.0)
            .with_fast_time(2.0)
            .with_slow_time(30.0);
        assert!((cfg.attack_db - 6.0).abs() < 1e-5);
        assert!((cfg.sustain_db - (-3.0)).abs() < 1e-5);
        assert!((cfg.output_gain_db - 1.0).abs() < 1e-5);
        assert!((cfg.fast_time_ms - 2.0).abs() < 1e-5);
        assert!((cfg.slow_time_ms - 30.0).abs() < 1e-5);
    }

    #[test]
    fn test_process_stereo() {
        let mut shaper = TransientShaper::new(TransientShaperConfig::default());
        let mut left = vec![0.3f32; 100];
        let mut right = vec![0.3f32; 100];
        shaper.process_stereo(&mut left, &mut right);
        // Should not crash, values should be finite
        for &s in &left {
            assert!(s.is_finite());
        }
        for &s in &right {
            assert!(s.is_finite());
        }
    }
}
