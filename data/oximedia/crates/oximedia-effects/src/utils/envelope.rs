//! Envelope follower and detection utilities.
//!
//! Provides envelope detection for dynamics processing, level metering,
//! and sidechain applications.

/// Envelope follower for tracking signal amplitude.
///
/// Uses separate attack and release times to follow the signal envelope.
#[derive(Debug, Clone)]
pub struct EnvelopeFollower {
    /// Current envelope value.
    envelope: f32,
    /// Attack coefficient.
    attack_coeff: f32,
    /// Release coefficient.
    release_coeff: f32,
    /// Sample rate.
    sample_rate: f32,
}

impl EnvelopeFollower {
    /// Create a new envelope follower.
    ///
    /// # Arguments
    ///
    /// * `attack_ms` - Attack time in milliseconds
    /// * `release_ms` - Release time in milliseconds
    /// * `sample_rate` - Audio sample rate
    #[must_use]
    pub fn new(attack_ms: f32, release_ms: f32, sample_rate: f32) -> Self {
        let attack_coeff = (-1000.0 / (attack_ms * sample_rate)).exp();
        let release_coeff = (-1000.0 / (release_ms * sample_rate)).exp();

        Self {
            envelope: 0.0,
            attack_coeff,
            release_coeff,
            sample_rate,
        }
    }

    /// Process a sample and return the envelope value.
    pub fn process(&mut self, input: f32) -> f32 {
        let input_abs = input.abs();

        if input_abs > self.envelope {
            // Attack
            self.envelope = input_abs + self.attack_coeff * (self.envelope - input_abs);
        } else {
            // Release
            self.envelope = input_abs + self.release_coeff * (self.envelope - input_abs);
        }

        self.envelope
    }

    /// Get current envelope value without processing.
    #[must_use]
    pub fn current(&self) -> f32 {
        self.envelope
    }

    /// Reset the envelope to zero.
    pub fn reset(&mut self) {
        self.envelope = 0.0;
    }

    /// Set attack time in milliseconds.
    pub fn set_attack(&mut self, attack_ms: f32) {
        self.attack_coeff = (-1000.0 / (attack_ms * self.sample_rate)).exp();
    }

    /// Set release time in milliseconds.
    pub fn set_release(&mut self, release_ms: f32) {
        self.release_coeff = (-1000.0 / (release_ms * self.sample_rate)).exp();
    }
}

/// RMS (Root Mean Square) envelope detector.
///
/// Provides a more accurate representation of perceived loudness compared
/// to peak detection.
#[derive(Debug, Clone)]
pub struct RmsDetector {
    /// Circular buffer for RMS calculation.
    buffer: Vec<f32>,
    /// Current write position.
    write_pos: usize,
    /// Sum of squares.
    sum_squares: f32,
    /// Window size.
    window_size: usize,
}

impl RmsDetector {
    /// Create a new RMS detector.
    ///
    /// # Arguments
    ///
    /// * `window_ms` - RMS window size in milliseconds
    /// * `sample_rate` - Audio sample rate
    #[must_use]
    pub fn new(window_ms: f32, sample_rate: f32) -> Self {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let window_size = ((window_ms * sample_rate / 1000.0) as usize).max(1);
        let buffer = vec![0.0; window_size];

        Self {
            buffer,
            write_pos: 0,
            sum_squares: 0.0,
            window_size,
        }
    }

    /// Process a sample and return the RMS value.
    pub fn process(&mut self, input: f32) -> f32 {
        // Remove old value from sum
        let old_value = self.buffer[self.write_pos];
        self.sum_squares -= old_value * old_value;

        // Add new value
        self.buffer[self.write_pos] = input;
        self.sum_squares += input * input;

        // Advance write position
        self.write_pos = (self.write_pos + 1) % self.window_size;

        // Return RMS value
        #[allow(clippy::cast_precision_loss)]
        let rms_val = (self.sum_squares / self.window_size as f32).sqrt();
        rms_val
    }

    /// Get current RMS value without processing.
    #[must_use]
    pub fn current(&self) -> f32 {
        #[allow(clippy::cast_precision_loss)]
        let rms_val = (self.sum_squares / self.window_size as f32).sqrt();
        rms_val
    }

    /// Reset the detector.
    pub fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.write_pos = 0;
        self.sum_squares = 0.0;
    }
}

/// Peak detector with hold and decay.
///
/// Tracks the peak level with configurable hold time and decay rate.
#[derive(Debug, Clone)]
pub struct PeakDetector {
    /// Current peak value.
    peak: f32,
    /// Hold counter.
    hold_counter: usize,
    /// Hold time in samples.
    hold_samples: usize,
    /// Decay coefficient.
    decay_coeff: f32,
}

impl PeakDetector {
    /// Create a new peak detector.
    ///
    /// # Arguments
    ///
    /// * `hold_ms` - Hold time in milliseconds
    /// * `decay_ms` - Decay time in milliseconds
    /// * `sample_rate` - Audio sample rate
    #[must_use]
    pub fn new(hold_ms: f32, decay_ms: f32, sample_rate: f32) -> Self {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let hold_samples = (hold_ms * sample_rate / 1000.0) as usize;
        let decay_coeff = (-1000.0 / (decay_ms * sample_rate)).exp();

        Self {
            peak: 0.0,
            hold_counter: 0,
            hold_samples,
            decay_coeff,
        }
    }

    /// Process a sample and return the peak value.
    pub fn process(&mut self, input: f32) -> f32 {
        let input_abs = input.abs();

        if input_abs >= self.peak {
            // New peak
            self.peak = input_abs;
            self.hold_counter = self.hold_samples;
        } else if self.hold_counter > 0 {
            // In hold period
            self.hold_counter -= 1;
        } else {
            // Decay
            self.peak *= self.decay_coeff;
        }

        self.peak
    }

    /// Get current peak value.
    #[must_use]
    pub fn current(&self) -> f32 {
        self.peak
    }

    /// Reset the detector.
    pub fn reset(&mut self) {
        self.peak = 0.0;
        self.hold_counter = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_envelope_follower_attack() {
        let mut env = EnvelopeFollower::new(1.0, 100.0, 48000.0);

        // Process increasing signal
        let _ = env.process(0.0);
        let v1 = env.process(1.0);
        let v2 = env.process(1.0);

        // Envelope should increase
        assert!(v1 > 0.0);
        assert!(v2 > v1);
    }

    #[test]
    fn test_envelope_follower_release() {
        let mut env = EnvelopeFollower::new(1.0, 100.0, 48000.0);

        // Set high envelope
        env.process(1.0);
        env.process(1.0);

        // Process silence
        let v1 = env.process(0.0);
        let v2 = env.process(0.0);
        let v3 = env.process(0.0);

        // Envelope should decrease slowly due to long release
        assert!(v1 < 1.0);
        assert!(v2 < v1);
        assert!(v3 < v2);
        assert!(v3 > 0.0); // But not instantly to zero
    }

    #[test]
    fn test_rms_detector() {
        let mut rms = RmsDetector::new(10.0, 48000.0);

        // Process silence
        assert_eq!(rms.process(0.0), 0.0);

        // Process constant signal
        for _ in 0..1000 {
            rms.process(1.0);
        }

        // RMS of constant 1.0 should be 1.0
        let rms_value = rms.current();
        assert!((rms_value - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_rms_sine_wave() {
        let mut rms = RmsDetector::new(10.0, 48000.0);

        // RMS of sine wave with amplitude 1.0 should be ~0.707
        use std::f32::consts::TAU;
        for i in 0..1000 {
            #[allow(clippy::cast_precision_loss)]
            let sample = (i as f32 * TAU / 100.0).sin();
            rms.process(sample);
        }

        let rms_value = rms.current();
        assert!((rms_value - 0.707).abs() < 0.1);
    }

    #[test]
    fn test_peak_detector() {
        let mut peak = PeakDetector::new(100.0, 1000.0, 48000.0);

        // Process peak
        peak.process(1.0);
        assert_eq!(peak.current(), 1.0);

        // Process silence - should hold
        for _ in 0..1000 {
            peak.process(0.0);
        }
        assert_eq!(peak.current(), 1.0); // Still holding

        // After hold period, should start decaying
        for _ in 0..10000 {
            peak.process(0.0);
        }
        assert!(peak.current() < 1.0);
    }

    #[test]
    fn test_envelope_reset() {
        let mut env = EnvelopeFollower::new(1.0, 100.0, 48000.0);
        env.process(1.0);
        assert!(env.current() > 0.0);

        env.reset();
        assert_eq!(env.current(), 0.0);
    }
}
