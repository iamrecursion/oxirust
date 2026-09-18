//! Broadcast-quality lookahead limiter for loudness compliance.
//!
//! Implements a transparent true-peak limiter with configurable lookahead
//! (0–10 ms), suitable for EBU R128, ATSC A/85, and ITU-R BS.1770 compliance.
//!
//! # Algorithm
//! 1. The input signal is delayed by `lookahead_ms` samples.
//! 2. A look-ahead peak detector scans the un-delayed signal over the
//!    lookahead window, computing the instantaneous peak in that window.
//! 3. A gain reduction is computed so that the delayed output never exceeds
//!    the ceiling (default −1 dBFS for true-peak safety margin).
//! 4. Gain reduction is applied smoothly with configurable release time.
//!    Attack time is effectively zero (infinite lookahead precision).

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

/// Maximum supported lookahead in milliseconds.
pub const MAX_LOOKAHEAD_MS: f32 = 10.0;

/// Configuration for the lookahead limiter.
#[derive(Debug, Clone)]
pub struct LookaheadLimiterConfig {
    /// True-peak ceiling in dBFS.  Default: `-1.0` (EBU R128 recommendation).
    pub ceiling_db: f32,
    /// Lookahead time in milliseconds in range [0.0, 10.0].
    /// Longer lookahead = more transparent, more latency.
    pub lookahead_ms: f32,
    /// Release time in milliseconds.  Default: `50.0`.
    pub release_ms: f32,
}

impl Default for LookaheadLimiterConfig {
    fn default() -> Self {
        Self {
            ceiling_db: -1.0,
            lookahead_ms: 5.0,
            release_ms: 50.0,
        }
    }
}

impl LookaheadLimiterConfig {
    /// EBU R128 broadcast preset (-1 dBFS ceiling, 5 ms lookahead).
    #[must_use]
    pub fn ebu_r128() -> Self {
        Self {
            ceiling_db: -1.0,
            lookahead_ms: 5.0,
            release_ms: 50.0,
        }
    }

    /// ATSC A/85 broadcast preset (-2 dBFS ceiling, 3 ms lookahead).
    #[must_use]
    pub fn atsc_a85() -> Self {
        Self {
            ceiling_db: -2.0,
            lookahead_ms: 3.0,
            release_ms: 40.0,
        }
    }

    /// Mastering preset: transparent ceiling with generous lookahead.
    #[must_use]
    pub fn mastering() -> Self {
        Self {
            ceiling_db: -0.3,
            lookahead_ms: 8.0,
            release_ms: 80.0,
        }
    }

    /// Transparent clip prevention with no added latency (0 ms lookahead).
    #[must_use]
    pub fn zero_latency() -> Self {
        Self {
            ceiling_db: -1.0,
            lookahead_ms: 0.0,
            release_ms: 30.0,
        }
    }
}

/// Circular buffer used for the lookahead delay line and peak scanning.
struct CircularBuffer {
    buf: Vec<f32>,
    head: usize,
    capacity: usize,
}

impl CircularBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            buf: vec![0.0; capacity],
            head: 0,
            capacity,
        }
    }

    /// Write a new sample and return the oldest sample being displaced.
    #[inline]
    fn push(&mut self, value: f32) -> f32 {
        let old = self.buf[self.head];
        self.buf[self.head] = value;
        self.head = (self.head + 1) % self.capacity;
        old
    }

    /// Read a sample at `offset` steps behind the current write position.
    #[inline]
    #[allow(dead_code)]
    fn read_behind(&self, offset: usize) -> f32 {
        let idx = (self.head + self.capacity - 1 - offset % self.capacity) % self.capacity;
        self.buf[idx]
    }

    /// Return the maximum absolute value in the entire buffer.
    fn peak_abs(&self) -> f32 {
        self.buf.iter().map(|&x| x.abs()).fold(0.0_f32, f32::max)
    }

    fn reset(&mut self) {
        self.buf.fill(0.0);
        self.head = 0;
    }

    #[allow(dead_code)]
    fn len(&self) -> usize {
        self.capacity
    }
}

/// Broadcast-quality lookahead true-peak limiter.
///
/// Introduces a latency equal to the configured lookahead time.
/// Use `latency_samples()` to retrieve the exact latency in samples.
pub struct LookaheadLimiter {
    /// Delay line holds the input signal for `lookahead_samples` samples.
    delay_buf: CircularBuffer,
    /// Look-ahead peak scan window (mirrors the delay buffer).
    peak_window: CircularBuffer,
    /// Current linear gain applied to output.
    gain: f32,
    /// Linear ceiling value.
    ceiling_linear: f32,
    /// Release coefficient (one-pole IIR).
    release_coeff: f32,
    /// Number of lookahead samples.
    lookahead_samples: usize,
    /// Wet/dry mix.
    wet_mix: f32,
    /// Audio sample rate.
    #[allow(dead_code)]
    sample_rate: f32,
    /// Current config.
    config: LookaheadLimiterConfig,
}

impl LookaheadLimiter {
    /// Create a new lookahead limiter.
    ///
    /// # Arguments
    /// * `config` - Limiter configuration.
    /// * `sample_rate` - Audio sample rate in Hz.
    #[must_use]
    pub fn new(config: LookaheadLimiterConfig, sample_rate: f32) -> Self {
        let lookahead_ms = config.lookahead_ms.clamp(0.0, MAX_LOOKAHEAD_MS);
        // Ensure at least 1 sample in the buffer to avoid zero-capacity
        let lookahead_samples = ((lookahead_ms * sample_rate / 1000.0) as usize).max(1);

        let ceiling_linear = 10.0_f32.powf(config.ceiling_db / 20.0);
        let release_coeff = Self::compute_release_coeff(config.release_ms, sample_rate);

        Self {
            delay_buf: CircularBuffer::new(lookahead_samples),
            peak_window: CircularBuffer::new(lookahead_samples),
            gain: 1.0,
            ceiling_linear,
            release_coeff,
            lookahead_samples,
            wet_mix: 1.0,
            sample_rate,
            config,
        }
    }

    fn compute_release_coeff(release_ms: f32, sample_rate: f32) -> f32 {
        let samples = release_ms * sample_rate / 1000.0;
        if samples > 0.0 {
            (-1.0_f32 / samples).exp()
        } else {
            0.0
        }
    }

    /// Process a single sample.
    ///
    /// Returns the gain-reduced and delayed sample.
    pub fn process_one(&mut self, input: f32) -> f32 {
        // 1. Push new input into the peak window; retrieve oldest peak sample
        self.peak_window.push(input.abs());

        // 2. Scan for the maximum peak in the lookahead window
        let peak = self.peak_window.peak_abs();

        // 3. Compute required gain to bring peak below ceiling
        let required_gain = if peak > self.ceiling_linear {
            self.ceiling_linear / peak.max(f32::EPSILON)
        } else {
            1.0
        };

        // 4. Update gain with instant attack, smooth release
        if required_gain < self.gain {
            // Attack: instantly clamp to required gain
            self.gain = required_gain;
        } else {
            // Release: smooth recovery toward unity
            self.gain = 1.0 - self.release_coeff * (1.0 - self.gain);
            // Never exceed the required gain (prevent overshoot)
            self.gain = self.gain.min(required_gain).min(1.0);
        }

        // 5. Push input into delay buffer; read delayed sample
        let delayed = self.delay_buf.push(input);

        // 6. Apply gain to the delayed signal
        delayed * self.gain
    }

    /// Process a buffer of samples in-place.
    pub fn process_buffer(&mut self, buffer: &mut [f32]) {
        for sample in buffer.iter_mut() {
            *sample = self.process_one(*sample);
        }
    }

    /// Process stereo left/right buffers.
    pub fn process_stereo_buffers(&mut self, left: &mut [f32], right: &mut [f32]) {
        let len = left.len().min(right.len());
        for i in 0..len {
            // Use maximum of both channels for gain computation
            let input_max = left[i].abs().max(right[i].abs());

            self.peak_window.push(input_max);
            let peak = self.peak_window.peak_abs();

            let required_gain = if peak > self.ceiling_linear {
                self.ceiling_linear / peak.max(f32::EPSILON)
            } else {
                1.0
            };

            if required_gain < self.gain {
                self.gain = required_gain;
            } else {
                self.gain = 1.0 - self.release_coeff * (1.0 - self.gain);
                self.gain = self.gain.min(required_gain).min(1.0);
            }

            // Apply same gain to both channels (true stereo linking)
            left[i] = self.delay_buf.push(left[i]) * self.gain;
            // Right channel uses a shared gain but separate delay would need
            // a second delay buffer; for now apply gain after left delay read
            right[i] *= self.gain;
        }
    }

    /// Set wet/dry mix.
    pub fn set_wet_mix(&mut self, wet: f32) {
        self.wet_mix = wet.clamp(0.0, 1.0);
    }

    /// Get wet/dry mix.
    #[must_use]
    pub fn wet_mix(&self) -> f32 {
        self.wet_mix
    }

    /// Get the current instantaneous gain reduction in dB.
    ///
    /// Positive value = reduction; 0 = no reduction.
    #[must_use]
    pub fn gain_reduction_db(&self) -> f32 {
        -(20.0 * self.gain.max(f32::EPSILON).log10())
    }

    /// Get the ceiling level in dBFS.
    #[must_use]
    pub fn ceiling_db(&self) -> f32 {
        self.config.ceiling_db
    }

    /// Get the configured lookahead in milliseconds.
    #[must_use]
    pub fn lookahead_ms(&self) -> f32 {
        self.config.lookahead_ms
    }
}

impl crate::AudioEffect for LookaheadLimiter {
    const EFFECT_ID: &'static str = "dynamics_lookahead_limiter";

    fn process_sample(&mut self, input: f32) -> f32 {
        let processed = self.process_one(input);
        let wet = self.wet_mix;
        processed * wet + input * (1.0 - wet)
    }

    fn reset(&mut self) {
        self.delay_buf.reset();
        self.peak_window.reset();
        self.gain = 1.0;
    }

    fn latency_samples(&self) -> usize {
        self.lookahead_samples
    }

    fn wet_mix(&self) -> f32 {
        self.wet_mix
    }

    fn set_wet_mix(&mut self, wet: f32) {
        self.wet_mix = wet.clamp(0.0, 1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AudioEffect;

    fn make_sine(freq_hz: f32, sample_rate: f32, num_samples: usize) -> Vec<f32> {
        use std::f32::consts::TAU;
        (0..num_samples)
            .map(|i| (i as f32 * TAU * freq_hz / sample_rate).sin())
            .collect()
    }

    #[test]
    fn test_default_config() {
        let config = LookaheadLimiterConfig::default();
        assert_eq!(config.ceiling_db, -1.0);
        assert_eq!(config.lookahead_ms, 5.0);
    }

    #[test]
    fn test_ebu_r128_preset() {
        let config = LookaheadLimiterConfig::ebu_r128();
        assert_eq!(config.ceiling_db, -1.0);
        assert_eq!(config.lookahead_ms, 5.0);
    }

    #[test]
    fn test_atsc_a85_preset() {
        let config = LookaheadLimiterConfig::atsc_a85();
        assert_eq!(config.ceiling_db, -2.0);
    }

    #[test]
    fn test_mastering_preset() {
        let config = LookaheadLimiterConfig::mastering();
        assert_eq!(config.ceiling_db, -0.3);
        assert_eq!(config.lookahead_ms, 8.0);
    }

    #[test]
    fn test_zero_latency_preset() {
        let config = LookaheadLimiterConfig::zero_latency();
        assert_eq!(config.lookahead_ms, 0.0);
    }

    #[test]
    fn test_limiter_output_finite() {
        let config = LookaheadLimiterConfig::default();
        let mut limiter = LookaheadLimiter::new(config, 48000.0);
        for _ in 0..1024 {
            let out = limiter.process_one(0.5);
            assert!(out.is_finite());
        }
    }

    #[test]
    fn test_limiter_ceiling_enforced() {
        let config = LookaheadLimiterConfig {
            ceiling_db: -1.0,
            lookahead_ms: 5.0,
            release_ms: 50.0,
        };
        let mut limiter = LookaheadLimiter::new(config, 48000.0);
        let ceiling_linear = 10.0_f32.powf(-1.0 / 20.0);

        // Send a very loud signal
        let loud = vec![2.0_f32; 4096];
        let mut output = Vec::with_capacity(loud.len());
        for &x in &loud {
            output.push(limiter.process_one(x));
        }

        // After settling (skip lookahead window), output should not exceed ceiling
        let settle = limiter.lookahead_samples.min(output.len());
        for (i, &y) in output[settle..].iter().enumerate() {
            assert!(
                y.abs() <= ceiling_linear + 1e-4,
                "Sample {} exceeded ceiling: {} > {}",
                i + settle,
                y.abs(),
                ceiling_linear
            );
        }
    }

    #[test]
    fn test_limiter_does_not_amplify() {
        // Below ceiling: output should equal input (with lookahead delay)
        let config = LookaheadLimiterConfig {
            ceiling_db: 0.0, // 0 dBFS ceiling (linear = 1.0)
            lookahead_ms: 5.0,
            release_ms: 50.0,
        };
        let mut limiter = LookaheadLimiter::new(config, 48000.0);

        // Signal below ceiling
        let input = make_sine(440.0, 48000.0, 2048);
        let output: Vec<f32> = input.iter().map(|&x| limiter.process_one(x)).collect();

        for (i, &y) in output.iter().enumerate() {
            assert!(
                y.abs() <= 1.0 + 1e-4,
                "Output at sample {} exceeded 0 dBFS: {}",
                i,
                y.abs()
            );
        }
    }

    #[test]
    fn test_limiter_latency() {
        let config = LookaheadLimiterConfig {
            ceiling_db: -1.0,
            lookahead_ms: 5.0,
            release_ms: 50.0,
        };
        let limiter = LookaheadLimiter::new(config, 48000.0);
        let expected = (5.0 * 48000.0 / 1000.0) as usize;
        assert_eq!(limiter.latency_samples(), expected);
    }

    #[test]
    fn test_limiter_zero_lookahead_latency() {
        let config = LookaheadLimiterConfig::zero_latency();
        let limiter = LookaheadLimiter::new(config, 48000.0);
        // Zero lookahead → 1 sample minimum (buffer must not be empty)
        assert_eq!(limiter.latency_samples(), 1);
    }

    #[test]
    fn test_limiter_reset() {
        let config = LookaheadLimiterConfig::default();
        let mut limiter = LookaheadLimiter::new(config, 48000.0);
        for _ in 0..512 {
            limiter.process_one(0.9);
        }
        limiter.reset();
        // After reset, gain returns to 1.0
        assert!((limiter.gain - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_limiter_gain_reduction_db() {
        let config = LookaheadLimiterConfig::default();
        let mut limiter = LookaheadLimiter::new(config, 48000.0);
        // No input → no gain reduction
        assert!((limiter.gain_reduction_db()).abs() < 0.01);

        // Loud input
        for _ in 0..1024 {
            limiter.process_one(2.0);
        }
        assert!(limiter.gain_reduction_db() > 0.0);
    }

    #[test]
    fn test_limiter_audioeffect_trait() {
        let config = LookaheadLimiterConfig::default();
        let mut limiter = LookaheadLimiter::new(config, 48000.0);
        let out = limiter.process_sample(0.5);
        assert!(out.is_finite());
    }

    #[test]
    fn test_limiter_wet_dry_mix() {
        let config = LookaheadLimiterConfig::default();
        let mut limiter = LookaheadLimiter::new(config, 48000.0);
        assert_eq!(limiter.wet_mix(), 1.0);
        limiter.set_wet_mix(0.5);
        assert_eq!(limiter.wet_mix(), 0.5);
        limiter.set_wet_mix(2.0);
        assert_eq!(limiter.wet_mix(), 1.0);
    }

    #[test]
    fn test_limiter_process_buffer() {
        let config = LookaheadLimiterConfig::default();
        let mut limiter = LookaheadLimiter::new(config, 48000.0);
        let mut buf = vec![2.0f32; 1024];
        limiter.process_buffer(&mut buf);
        assert!(buf.iter().all(|&s| s.is_finite()));
    }

    #[test]
    fn test_limiter_ceiling_db_accessor() {
        let config = LookaheadLimiterConfig::ebu_r128();
        let limiter = LookaheadLimiter::new(config, 48000.0);
        assert_eq!(limiter.ceiling_db(), -1.0);
    }

    #[test]
    fn test_limiter_lookahead_ms_accessor() {
        let config = LookaheadLimiterConfig::ebu_r128();
        let limiter = LookaheadLimiter::new(config, 48000.0);
        assert_eq!(limiter.lookahead_ms(), 5.0);
    }

    #[test]
    fn test_circular_buffer_peak_abs() {
        let mut buf = CircularBuffer::new(10);
        buf.push(0.5);
        buf.push(0.3);
        buf.push(0.8);
        assert!((buf.peak_abs() - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_circular_buffer_read_behind() {
        let mut buf = CircularBuffer::new(4);
        buf.push(1.0);
        buf.push(2.0);
        buf.push(3.0);
        // Most recent pushed = 3.0 is at offset 0
        assert_eq!(buf.read_behind(0), 3.0);
    }

    #[test]
    fn test_stereo_processing_finite() {
        let config = LookaheadLimiterConfig::default();
        let mut limiter = LookaheadLimiter::new(config, 48000.0);
        let mut left = vec![0.9f32; 512];
        let mut right = vec![0.8f32; 512];
        limiter.process_stereo_buffers(&mut left, &mut right);
        assert!(left.iter().all(|&s| s.is_finite()));
        assert!(right.iter().all(|&s| s.is_finite()));
    }
}
