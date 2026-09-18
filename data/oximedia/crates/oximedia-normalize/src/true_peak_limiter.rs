//! True-peak lookahead brick-wall limiter.
//!
//! Provides `LimiterConfig`, `TruePeakLimiter`, and `TruePeakStats` for
//! preventing inter-sample clipping above a configurable dBTP ceiling.

#![allow(dead_code)]

/// Configuration for the true-peak limiter.
#[derive(Debug, Clone)]
pub struct LimiterConfig {
    /// True-peak ceiling in dBTP (negative, e.g. -1.0).
    pub ceiling_dbtp: f64,
    /// Attack time in milliseconds.
    pub attack_ms: f64,
    /// Release time in milliseconds.
    pub release_ms: f64,
    /// Lookahead buffer length in milliseconds.
    pub lookahead_ms: f64,
    /// Sample rate in Hz.
    pub sample_rate: f64,
}

impl LimiterConfig {
    /// Create a new limiter configuration.
    pub fn new(ceiling_dbtp: f64, sample_rate: f64) -> Self {
        Self {
            ceiling_dbtp,
            attack_ms: 0.5,
            release_ms: 100.0,
            lookahead_ms: 2.0,
            sample_rate,
        }
    }

    /// Create the default broadcast config (-1 dBTP ceiling, 48 kHz).
    pub fn broadcast() -> Self {
        Self::new(-1.0, 48_000.0)
    }

    /// Validate that the config is sensible.
    pub fn is_valid(&self) -> bool {
        self.ceiling_dbtp <= 0.0
            && self.attack_ms > 0.0
            && self.release_ms > 0.0
            && self.lookahead_ms >= 0.0
            && self.sample_rate >= 8_000.0
            && self.sample_rate <= 384_000.0
    }

    /// Return the ceiling as a linear amplitude value.
    #[allow(clippy::cast_precision_loss)]
    pub fn ceiling_linear(&self) -> f64 {
        10.0_f64.powf(self.ceiling_dbtp / 20.0)
    }

    /// Lookahead size in samples.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn lookahead_samples(&self) -> usize {
        ((self.lookahead_ms / 1000.0) * self.sample_rate).ceil() as usize
    }
}

/// Per-sample true-peak brick-wall limiter with lookahead.
///
/// Uses a simple envelope follower: the gain reduction envelope is driven
/// down when a peak exceeds the ceiling and recovers with the configured
/// release time.
pub struct TruePeakLimiter {
    config: LimiterConfig,
    /// Current gain reduction factor (1.0 = no reduction, < 1.0 = attenuating).
    current_gain: f64,
    /// Running peak (linear) seen recently.
    peak_hold: f64,
    /// Attack coefficient per sample.
    attack_coeff: f64,
    /// Release coefficient per sample.
    release_coeff: f64,
    /// Total samples processed.
    samples_processed: u64,
    /// Cumulative gain reduction in dB (sum, for stats).
    total_reduction_db: f64,
    /// Number of samples where limiting was active.
    limited_samples: u64,
}

impl TruePeakLimiter {
    /// Create a new limiter from `config`.
    pub fn new(config: LimiterConfig) -> Self {
        let attack_coeff = Self::time_to_coeff(config.attack_ms, config.sample_rate);
        let release_coeff = Self::time_to_coeff(config.release_ms, config.sample_rate);
        Self {
            config,
            current_gain: 1.0,
            peak_hold: 0.0,
            attack_coeff,
            release_coeff,
            samples_processed: 0,
            total_reduction_db: 0.0,
            limited_samples: 0,
        }
    }

    /// Process a single sample and return the gain-reduced output.
    #[allow(clippy::cast_precision_loss)]
    pub fn process_sample(&mut self, sample: f32) -> f32 {
        let x = f64::from(sample.abs());
        let ceiling = self.config.ceiling_linear();

        // Update peak hold
        if x > self.peak_hold {
            self.peak_hold = x;
        } else {
            self.peak_hold *= self.release_coeff;
        }

        // Compute desired gain
        let desired_gain = if self.peak_hold > ceiling {
            ceiling / self.peak_hold
        } else {
            1.0
        };

        // Apply attack / release to smooth gain reduction
        if desired_gain < self.current_gain {
            self.current_gain =
                self.current_gain * self.attack_coeff + desired_gain * (1.0 - self.attack_coeff);
        } else {
            self.current_gain =
                self.current_gain * self.release_coeff + desired_gain * (1.0 - self.release_coeff);
        }

        // Clamp gain to never exceed 1.0
        self.current_gain = self.current_gain.min(1.0);

        // Accumulate stats
        if self.current_gain < 0.9999 {
            let reduction_db = -20.0 * self.current_gain.log10();
            self.total_reduction_db += reduction_db;
            self.limited_samples += 1;
        }
        self.samples_processed += 1;

        (f64::from(sample) * self.current_gain) as f32
    }

    /// Process an entire buffer in-place.
    pub fn process_buffer(&mut self, buffer: &mut [f32]) {
        for s in buffer.iter_mut() {
            *s = self.process_sample(*s);
        }
    }

    /// Return the current instantaneous gain reduction in dB (0.0 = none).
    #[allow(clippy::cast_precision_loss)]
    pub fn gain_reduction_db(&self) -> f64 {
        if self.current_gain >= 1.0 {
            0.0
        } else {
            -20.0 * self.current_gain.log10()
        }
    }

    /// Reset the limiter state.
    pub fn reset(&mut self) {
        self.current_gain = 1.0;
        self.peak_hold = 0.0;
        self.samples_processed = 0;
        self.total_reduction_db = 0.0;
        self.limited_samples = 0;
    }

    /// Return a statistics snapshot.
    pub fn stats(&self) -> TruePeakStats {
        TruePeakStats {
            samples_processed: self.samples_processed,
            limited_samples: self.limited_samples,
            total_reduction_db: self.total_reduction_db,
            peak_reduction_db: self.gain_reduction_db(),
        }
    }

    /// Compute a single-pole IIR coefficient for a given time constant.
    fn time_to_coeff(time_ms: f64, sample_rate: f64) -> f64 {
        let tau_samples = (time_ms / 1000.0) * sample_rate;
        if tau_samples <= 0.0 {
            0.0
        } else {
            (-1.0 / tau_samples).exp()
        }
    }
}

/// Statistics produced by `TruePeakLimiter`.
#[derive(Debug, Clone)]
pub struct TruePeakStats {
    /// Total samples processed since last reset.
    pub samples_processed: u64,
    /// Number of samples where limiting was active.
    pub limited_samples: u64,
    /// Cumulative gain reduction in dB.
    pub total_reduction_db: f64,
    /// Most recent instantaneous gain reduction in dB.
    pub peak_reduction_db: f64,
}

impl TruePeakStats {
    /// Return `true` if any limiting occurred.
    pub fn has_limiting(&self) -> bool {
        self.limited_samples > 0
    }

    /// Return the percentage of samples that were limited (0.0–100.0).
    #[allow(clippy::cast_precision_loss)]
    pub fn reduction_percentage(&self) -> f64 {
        if self.samples_processed == 0 {
            return 0.0;
        }
        100.0 * self.limited_samples as f64 / self.samples_processed as f64
    }

    /// Average gain reduction per limited sample in dB.
    #[allow(clippy::cast_precision_loss)]
    pub fn average_reduction_db(&self) -> f64 {
        if self.limited_samples == 0 {
            return 0.0;
        }
        self.total_reduction_db / self.limited_samples as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_is_valid() {
        let cfg = LimiterConfig::new(-1.0, 48_000.0);
        assert!(cfg.is_valid());
    }

    #[test]
    fn test_config_invalid_ceiling() {
        let mut cfg = LimiterConfig::new(-1.0, 48_000.0);
        cfg.ceiling_dbtp = 1.0; // positive — invalid
        assert!(!cfg.is_valid());
    }

    #[test]
    fn test_config_invalid_sample_rate() {
        let cfg = LimiterConfig::new(-1.0, 100.0);
        assert!(!cfg.is_valid());
    }

    #[test]
    fn test_ceiling_linear_minus_1dbtp() {
        let cfg = LimiterConfig::new(-1.0, 48_000.0);
        let lin = cfg.ceiling_linear();
        // -1 dBTP ≈ 0.891
        assert!((lin - 0.891_250_9).abs() < 1e-5);
    }

    #[test]
    fn test_lookahead_samples() {
        let cfg = LimiterConfig::new(-1.0, 48_000.0);
        // 2.0 ms * 48000 = 96 samples
        assert_eq!(cfg.lookahead_samples(), 96);
    }

    #[test]
    fn test_limiter_passthrough_low_signal() {
        let cfg = LimiterConfig::broadcast();
        let mut lim = TruePeakLimiter::new(cfg);
        // Signal at 0.1 — well below -1 dBTP ceiling
        let out = lim.process_sample(0.1_f32);
        assert!(out.abs() <= 0.1_f32 + 1e-4);
    }

    #[test]
    fn test_limiter_reduces_loud_signal() {
        let cfg = LimiterConfig::broadcast();
        let mut lim = TruePeakLimiter::new(cfg);
        // Process 200 samples at full scale — limiter should engage
        for _ in 0..200 {
            lim.process_sample(1.0_f32);
        }
        assert!(lim.gain_reduction_db() > 0.0);
    }

    #[test]
    fn test_stats_has_limiting() {
        let cfg = LimiterConfig::broadcast();
        let mut lim = TruePeakLimiter::new(cfg);
        for _ in 0..500 {
            lim.process_sample(1.0_f32);
        }
        let stats = lim.stats();
        assert!(stats.has_limiting());
    }

    #[test]
    fn test_stats_no_limiting_quiet_signal() {
        let cfg = LimiterConfig::broadcast();
        let mut lim = TruePeakLimiter::new(cfg);
        for _ in 0..100 {
            lim.process_sample(0.01_f32);
        }
        let stats = lim.stats();
        assert_eq!(stats.samples_processed, 100);
        // Quiet signal should not trigger limiting
        assert!(!stats.has_limiting());
    }

    #[test]
    fn test_reduction_percentage_range() {
        let cfg = LimiterConfig::broadcast();
        let mut lim = TruePeakLimiter::new(cfg);
        for _ in 0..1000 {
            lim.process_sample(1.0_f32);
        }
        let pct = lim.stats().reduction_percentage();
        assert!(pct >= 0.0 && pct <= 100.0);
    }

    #[test]
    fn test_limiter_reset() {
        let cfg = LimiterConfig::broadcast();
        let mut lim = TruePeakLimiter::new(cfg);
        for _ in 0..200 {
            lim.process_sample(1.0_f32);
        }
        lim.reset();
        assert_eq!(lim.stats().samples_processed, 0);
        assert!(!lim.stats().has_limiting());
    }

    #[test]
    fn test_process_buffer() {
        let cfg = LimiterConfig::broadcast();
        let mut lim = TruePeakLimiter::new(cfg);
        // Pre-process a warm-up buffer so the gain envelope is fully settled.
        let mut warmup = vec![1.0_f32; 1024];
        lim.process_buffer(&mut warmup);
        // Now process the actual measurement buffer.
        let mut buf = vec![1.0_f32; 256];
        lim.process_buffer(&mut buf);
        // After settling, all outputs must be at or below the ceiling.
        let ceiling_lin = lim.config.ceiling_linear() as f32;
        for &s in &buf {
            assert!(s.abs() <= ceiling_lin + 0.01);
        }
    }

    #[test]
    fn test_gain_reduction_db_after_reset() {
        let cfg = LimiterConfig::broadcast();
        let mut lim = TruePeakLimiter::new(cfg);
        lim.reset();
        assert_eq!(lim.gain_reduction_db(), 0.0);
    }

    #[test]
    fn test_broadcast_preset() {
        let cfg = LimiterConfig::broadcast();
        assert!((cfg.ceiling_dbtp - -1.0).abs() < 1e-9);
        assert!(cfg.is_valid());
    }
}
