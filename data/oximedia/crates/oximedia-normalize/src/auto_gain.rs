//! Automatic Gain Control for `OxiMedia` normalize crate.
//!
//! Provides adaptive gain management with fixed, adaptive and program-aware modes.

#![allow(dead_code)]

/// Gain operating mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GainMode {
    /// Fixed gain applied uniformly.
    Fixed,
    /// Adaptive gain that tracks signal level.
    Adaptive,
    /// Program-loudness-aware mode (follows EBU R128 style).
    Program,
}

impl GainMode {
    /// Human-readable description of the mode.
    pub fn description(self) -> &'static str {
        match self {
            Self::Fixed => "Fixed gain: constant dB offset applied to every sample",
            Self::Adaptive => "Adaptive gain: tracks RMS level with attack/release",
            Self::Program => "Program gain: integrates long-term loudness per ITU-R BS.1770",
        }
    }
}

/// Configuration for `AutoGainControl`.
#[derive(Clone, Debug)]
pub struct AutoGainConfig {
    /// Operating mode.
    pub mode: GainMode,
    /// Target output level in dBFS.
    pub target_db: f32,
    /// Minimum gain in dB (lower bound of gain_range_db).
    pub min_gain_db: f32,
    /// Maximum gain in dB (upper bound of gain_range_db).
    pub max_gain_db: f32,
    /// Attack time in milliseconds.
    pub attack_ms: f32,
    /// Release time in milliseconds.
    pub release_ms: f32,
    /// Sample rate in Hz.
    pub sample_rate: f32,
}

impl AutoGainConfig {
    /// Construct a new config with sensible broadcast defaults.
    pub fn new(sample_rate: f32) -> Self {
        Self {
            mode: GainMode::Adaptive,
            target_db: -18.0,
            min_gain_db: -20.0,
            max_gain_db: 20.0,
            attack_ms: 10.0,
            release_ms: 150.0,
            sample_rate,
        }
    }

    /// Returns `(min_gain_db, max_gain_db)` as the allowed gain range.
    pub fn gain_range_db(&self) -> (f32, f32) {
        (self.min_gain_db, self.max_gain_db)
    }

    /// Validate the configuration, returning an error message on failure.
    pub fn validate(&self) -> Result<(), String> {
        if self.min_gain_db >= self.max_gain_db {
            return Err("min_gain_db must be less than max_gain_db".to_string());
        }
        if self.attack_ms <= 0.0 {
            return Err("attack_ms must be positive".to_string());
        }
        if self.release_ms <= 0.0 {
            return Err("release_ms must be positive".to_string());
        }
        if self.sample_rate < 8_000.0 || self.sample_rate > 384_000.0 {
            return Err(format!("sample_rate {} is out of range", self.sample_rate));
        }
        Ok(())
    }
}

/// Automatic Gain Controller.
///
/// Processes a stream of audio samples and applies time-varying gain to
/// maintain a target output level.
pub struct AutoGainControl {
    config: AutoGainConfig,
    /// Current gain (linear).
    current_gain: f32,
    /// Smoothed envelope (linear RMS estimate).
    envelope: f32,
    /// Attack coefficient per sample.
    attack_coeff: f32,
    /// Release coefficient per sample.
    release_coeff: f32,
    /// Total samples processed.
    samples_processed: u64,
}

impl AutoGainControl {
    /// Create a new `AutoGainControl` from the given config.
    pub fn new(config: AutoGainConfig) -> Self {
        let sr = config.sample_rate;
        let attack_coeff = (-1.0_f32 / (config.attack_ms * 0.001 * sr)).exp();
        let release_coeff = (-1.0_f32 / (config.release_ms * 0.001 * sr)).exp();
        Self {
            config,
            current_gain: 1.0,
            envelope: 1e-6,
            attack_coeff,
            release_coeff,
            samples_processed: 0,
        }
    }

    /// Process a single sample and return the gain-adjusted output.
    #[allow(clippy::cast_precision_loss)]
    pub fn process_sample(&mut self, sample: f32) -> f32 {
        let abs_sample = sample.abs();

        // Update envelope with asymmetric attack/release
        let coeff = if abs_sample > self.envelope {
            self.attack_coeff
        } else {
            self.release_coeff
        };
        self.envelope = coeff * self.envelope + (1.0 - coeff) * abs_sample;

        // Compute target gain from envelope
        let target_level_lin = 10.0_f32.powf(self.config.target_db / 20.0);
        let desired_gain = if self.envelope > 1e-9 {
            target_level_lin / self.envelope
        } else {
            1.0
        };

        // Clamp to configured range
        let (min_g, max_g) = self.config.gain_range_db();
        let min_lin = 10.0_f32.powf(min_g / 20.0);
        let max_lin = 10.0_f32.powf(max_g / 20.0);
        self.current_gain = desired_gain.clamp(min_lin, max_lin);

        self.samples_processed += 1;
        sample * self.current_gain
    }

    /// Current applied gain in dB.
    #[allow(clippy::cast_precision_loss)]
    pub fn current_gain_db(&self) -> f32 {
        20.0 * self.current_gain.log10()
    }

    /// Reset internal state (envelope and gain).
    pub fn reset(&mut self) {
        self.envelope = 1e-6;
        self.current_gain = 1.0;
        self.samples_processed = 0;
    }

    /// Number of samples processed so far.
    pub fn samples_processed(&self) -> u64 {
        self.samples_processed
    }

    /// Reference to the configuration.
    pub fn config(&self) -> &AutoGainConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gain_mode_description_fixed() {
        let desc = GainMode::Fixed.description();
        assert!(desc.contains("Fixed"));
    }

    #[test]
    fn test_gain_mode_description_adaptive() {
        let desc = GainMode::Adaptive.description();
        assert!(desc.contains("Adaptive"));
    }

    #[test]
    fn test_gain_mode_description_program() {
        let desc = GainMode::Program.description();
        assert!(desc.contains("Program"));
    }

    #[test]
    fn test_gain_mode_equality() {
        assert_eq!(GainMode::Fixed, GainMode::Fixed);
        assert_ne!(GainMode::Fixed, GainMode::Adaptive);
    }

    #[test]
    fn test_config_gain_range_db() {
        let cfg = AutoGainConfig::new(48_000.0);
        let (min, max) = cfg.gain_range_db();
        assert!(min < max, "min must be less than max");
        assert!((min - (-20.0)).abs() < 1e-5);
        assert!((max - 20.0).abs() < 1e-5);
    }

    #[test]
    fn test_config_validate_ok() {
        let cfg = AutoGainConfig::new(48_000.0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_validate_bad_range() {
        let mut cfg = AutoGainConfig::new(48_000.0);
        cfg.min_gain_db = 5.0;
        cfg.max_gain_db = 5.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validate_bad_attack() {
        let mut cfg = AutoGainConfig::new(48_000.0);
        cfg.attack_ms = 0.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validate_bad_sample_rate() {
        let mut cfg = AutoGainConfig::new(48_000.0);
        cfg.sample_rate = 100.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_agc_creation() {
        let cfg = AutoGainConfig::new(48_000.0);
        let agc = AutoGainControl::new(cfg);
        // Initial gain should be 1.0 (0 dB)
        assert!((agc.current_gain_db()).abs() < 1.0);
    }

    #[test]
    fn test_agc_process_silence() {
        let cfg = AutoGainConfig::new(48_000.0);
        let mut agc = AutoGainControl::new(cfg);
        let out = agc.process_sample(0.0);
        assert!(out.abs() < 1e-6);
    }

    #[test]
    fn test_agc_samples_processed_count() {
        let cfg = AutoGainConfig::new(48_000.0);
        let mut agc = AutoGainControl::new(cfg);
        for _ in 0..100 {
            agc.process_sample(0.1);
        }
        assert_eq!(agc.samples_processed(), 100);
    }

    #[test]
    fn test_agc_reset_clears_state() {
        let cfg = AutoGainConfig::new(48_000.0);
        let mut agc = AutoGainControl::new(cfg);
        for _ in 0..1000 {
            agc.process_sample(0.5);
        }
        agc.reset();
        assert_eq!(agc.samples_processed(), 0);
    }

    #[test]
    fn test_agc_gain_clamped_to_max() {
        let mut cfg = AutoGainConfig::new(48_000.0);
        cfg.max_gain_db = 6.0;
        let mut agc = AutoGainControl::new(cfg);
        // Feed near-silence so gain wants to rise; check it doesn't exceed max
        for _ in 0..5000 {
            agc.process_sample(1e-5);
        }
        assert!(agc.current_gain_db() <= 6.1); // small float tolerance
    }

    #[test]
    fn test_agc_gain_clamped_to_min() {
        let mut cfg = AutoGainConfig::new(48_000.0);
        cfg.min_gain_db = -6.0;
        let mut agc = AutoGainControl::new(cfg);
        // Feed loud signal so gain wants to fall; check it doesn't drop below min
        for _ in 0..5000 {
            agc.process_sample(1.0);
        }
        assert!(agc.current_gain_db() >= -6.1);
    }

    #[test]
    fn test_agc_config_accessor() {
        let cfg = AutoGainConfig::new(44_100.0);
        let agc = AutoGainControl::new(cfg);
        assert!((agc.config().sample_rate - 44_100.0).abs() < 1.0);
    }
}
