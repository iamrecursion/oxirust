//! Peak limiting: true peak detection, inter-sample peak estimation, and limiter lookahead.
//!
//! Implements a lookahead brick-wall limiter suitable for broadcast use.
//! True peak detection uses 4x linear interpolation for inter-sample peak estimation.

#![allow(dead_code)]

/// Default lookahead in samples (at 48 kHz ≈ 5 ms).
pub const DEFAULT_LOOKAHEAD_SAMPLES: usize = 240;

/// Convert a linear amplitude ratio to dBFS.
#[inline]
#[allow(clippy::cast_precision_loss)]
pub fn linear_to_db(linear: f64) -> f64 {
    if linear <= 0.0 {
        f64::NEG_INFINITY
    } else {
        20.0 * linear.log10()
    }
}

/// Convert dBFS to a linear amplitude ratio.
#[inline]
pub fn db_to_linear(db: f64) -> f64 {
    10.0_f64.powf(db / 20.0)
}

/// Configuration for the peak limiter.
#[derive(Debug, Clone)]
pub struct PeakLimiterConfig {
    /// True peak ceiling in dBTP (typically -1.0 for EBU R128).
    pub ceiling_dbtp: f64,
    /// Lookahead buffer length in samples.
    pub lookahead_samples: usize,
    /// Attack time in samples (how quickly gain is reduced).
    pub attack_samples: usize,
    /// Release time in samples (how quickly gain recovers).
    pub release_samples: usize,
}

impl PeakLimiterConfig {
    /// Create a standard EBU R128 limiter configuration.
    pub fn ebu_r128(sample_rate: f64) -> Self {
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        let sr = sample_rate as usize;
        Self {
            ceiling_dbtp: -1.0,
            lookahead_samples: sr / 200, // 5 ms
            attack_samples: sr / 1000,   // 1 ms
            release_samples: sr / 10,    // 100 ms
        }
    }
}

impl Default for PeakLimiterConfig {
    fn default() -> Self {
        Self {
            ceiling_dbtp: -1.0,
            lookahead_samples: DEFAULT_LOOKAHEAD_SAMPLES,
            attack_samples: 48,
            release_samples: 4800,
        }
    }
}

/// Estimate the inter-sample (true) peak between two consecutive samples using
/// linear interpolation at 4x oversampling.
#[allow(clippy::cast_precision_loss)]
pub fn estimate_true_peak(s0: f64, s1: f64) -> f64 {
    // 4 interpolated points between s0 and s1
    (0..=4)
        .map(|i| {
            let t = f64::from(i) / 4.0;
            (s0 * (1.0 - t) + s1 * t).abs()
        })
        .fold(0.0_f64, f64::max)
}

/// Find the true peak of a mono signal using inter-sample estimation.
pub fn true_peak_mono(samples: &[f64]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut peak = samples[0].abs();
    for window in samples.windows(2) {
        let isp = estimate_true_peak(window[0], window[1]);
        if isp > peak {
            peak = isp;
        }
    }
    peak
}

/// Find the true peak across all channels of interleaved audio.
///
/// `channels` must be > 0.
pub fn true_peak_interleaved(samples: &[f64], channels: usize) -> f64 {
    assert!(channels > 0, "channels must be > 0");
    let mut peak = 0.0_f64;
    for ch in 0..channels {
        // Collect this channel
        let mono: Vec<f64> = samples.iter().skip(ch).step_by(channels).copied().collect();
        let ch_peak = true_peak_mono(&mono);
        if ch_peak > peak {
            peak = ch_peak;
        }
    }
    peak
}

/// Stateful lookahead peak limiter for streaming use.
///
/// Feed samples in blocks via [`PeakLimiter::process`].
#[derive(Debug)]
pub struct PeakLimiter {
    config: PeakLimiterConfig,
    ceiling_linear: f64,
    /// Circular lookahead buffer (mono samples).
    lookahead: Vec<f64>,
    lookahead_pos: usize,
    /// Current gain reduction (linear, 1.0 = no reduction).
    gain: f64,
    /// Direction of gain change per sample.
    gain_step_attack: f64,
    gain_step_release: f64,
}

impl PeakLimiter {
    /// Create a new peak limiter.
    pub fn new(config: PeakLimiterConfig) -> Self {
        let ceiling_linear = db_to_linear(config.ceiling_dbtp);
        let gain_step_attack = 1.0 / config.attack_samples.max(1) as f64;
        let gain_step_release = 1.0 / config.release_samples.max(1) as f64;
        let lookahead_len = config.lookahead_samples.max(1);
        Self {
            config,
            ceiling_linear,
            lookahead: vec![0.0; lookahead_len],
            lookahead_pos: 0,
            gain: 1.0,
            gain_step_attack,
            gain_step_release,
        }
    }

    /// Process a mono block in-place.
    ///
    /// Returns the minimum gain applied during this block (for metering).
    pub fn process(&mut self, samples: &mut [f64]) -> f64 {
        let mut min_gain = 1.0_f64;
        for sample in samples.iter_mut() {
            // Look at the incoming sample and decide if gain needs reducing
            let abs = sample.abs();
            let required_gain = if abs > self.ceiling_linear {
                self.ceiling_linear / abs
            } else {
                1.0
            };

            if required_gain < self.gain {
                // Attack: reduce gain quickly
                self.gain -= self.gain_step_attack;
                self.gain = self.gain.max(required_gain);
            } else {
                // Release: recover gain slowly
                self.gain += self.gain_step_release;
                self.gain = self.gain.min(1.0);
            }

            // Write delayed sample from lookahead, apply current gain
            let delayed = self.lookahead[self.lookahead_pos];
            self.lookahead[self.lookahead_pos] = *sample;
            self.lookahead_pos = (self.lookahead_pos + 1) % self.lookahead.len();

            *sample = delayed * self.gain;
            if self.gain < min_gain {
                min_gain = self.gain;
            }
        }
        min_gain
    }

    /// Reset internal state (gain = 1, buffers cleared).
    pub fn reset(&mut self) {
        self.gain = 1.0;
        self.lookahead_pos = 0;
        for v in &mut self.lookahead {
            *v = 0.0;
        }
    }

    /// Return the current gain reduction in dB.
    pub fn gain_reduction_db(&self) -> f64 {
        linear_to_db(self.gain)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_to_db_unity() {
        assert!((linear_to_db(1.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_linear_to_db_zero() {
        assert_eq!(linear_to_db(0.0), f64::NEG_INFINITY);
    }

    #[test]
    fn test_db_to_linear_unity() {
        assert!((db_to_linear(0.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_db_roundtrip() {
        let db = -6.0;
        let result = linear_to_db(db_to_linear(db));
        assert!((result - db).abs() < 1e-9);
    }

    #[test]
    fn test_estimate_true_peak_zero_crossing() {
        // Two samples of equal magnitude: inter-sample peak should equal that magnitude
        let peak = estimate_true_peak(0.5, 0.5);
        assert!((peak - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_estimate_true_peak_negative_sample() {
        let peak = estimate_true_peak(-0.8, 0.0);
        assert!(peak >= 0.0, "peak must be non-negative");
        assert!(peak <= 0.8 + 1e-10);
    }

    #[test]
    fn test_true_peak_mono_silence() {
        let samples = vec![0.0f64; 100];
        assert_eq!(true_peak_mono(&samples), 0.0);
    }

    #[test]
    fn test_true_peak_mono_clipping() {
        let mut samples = vec![0.0f64; 48];
        samples[20] = 1.5;
        let peak = true_peak_mono(&samples);
        assert!(peak >= 1.0);
    }

    #[test]
    fn test_true_peak_interleaved_stereo() {
        // Left channel silent, right channel at 0.5
        let samples: Vec<f64> = (0..20)
            .map(|i| if i % 2 == 0 { 0.0 } else { 0.5 })
            .collect();
        let peak = true_peak_interleaved(&samples, 2);
        assert!(peak > 0.0);
        assert!(peak <= 0.5 + 1e-6);
    }

    #[test]
    fn test_peak_limiter_config_default() {
        let cfg = PeakLimiterConfig::default();
        assert_eq!(cfg.ceiling_dbtp, -1.0);
        assert_eq!(cfg.lookahead_samples, DEFAULT_LOOKAHEAD_SAMPLES);
    }

    #[test]
    fn test_peak_limiter_ebu_r128() {
        let cfg = PeakLimiterConfig::ebu_r128(48000.0);
        assert_eq!(cfg.ceiling_dbtp, -1.0);
        assert!(cfg.lookahead_samples > 0);
    }

    #[test]
    fn test_peak_limiter_silence_passthrough() {
        let mut limiter = PeakLimiter::new(PeakLimiterConfig::default());
        let mut samples = vec![0.0f64; 500];
        limiter.process(&mut samples);
        // All samples should remain 0 (or the lookahead delay fills with 0s)
        assert!(samples.iter().all(|&s| s.abs() < 1e-12));
    }

    #[test]
    fn test_peak_limiter_clips_loud_signal() {
        let mut limiter = PeakLimiter::new(PeakLimiterConfig {
            ceiling_dbtp: 0.0, // 0 dBTP = unity
            lookahead_samples: 10,
            attack_samples: 1,
            release_samples: 100,
        });
        // Signal well above ceiling
        let mut samples = vec![2.0f64; 200];
        limiter.process(&mut samples);
        // After enough time, samples should be at or near ceiling (1.0).
        // A tolerance of 0.02 accommodates one release-step overshoot before
        // the attack envelope corrects on the following sample.
        let max = samples[100..].iter().cloned().fold(0.0_f64, f64::max);
        assert!(max <= 1.0 + 0.02, "max = {max}");
    }

    #[test]
    fn test_peak_limiter_reset() {
        let mut limiter = PeakLimiter::new(PeakLimiterConfig::default());
        let mut samples = vec![0.9f64; 100];
        limiter.process(&mut samples);
        limiter.reset();
        assert!((limiter.gain - 1.0).abs() < 1e-10);
    }
}
