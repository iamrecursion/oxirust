//! Sidechain routing for dynamics processing in the `OxiMedia` mixer.
//!
//! Sidechaining allows a compressor, gate, or other dynamic processor to
//! use a *different* signal for level detection than the signal being
//! processed.  Common uses include ducking, de-essing, and pumping effects.

#![allow(dead_code)]

/// Where the sidechain signal is sourced from.
#[derive(Debug, Clone, PartialEq)]
pub enum SidechainSource {
    /// Use the channel's own signal (standard dynamics behaviour).
    Internal,
    /// Use an external hardware or software input.
    External,
    /// Use a specific mix bus identified by its ID.
    Bus(u32),
}

impl SidechainSource {
    /// Returns `true` if the source is external to the current channel.
    #[must_use]
    pub fn is_external(&self) -> bool {
        matches!(self, Self::External | Self::Bus(_))
    }
}

/// Configuration for sidechain filtering and monitoring.
#[derive(Debug, Clone)]
pub struct SidechainConfig {
    /// Source of the sidechain signal.
    pub source: SidechainSource,
    /// High-pass filter cutoff frequency in Hz (0 = bypass).
    pub hp_freq_hz: f32,
    /// Low-pass filter cutoff frequency in Hz (0 = bypass).
    pub lp_freq_hz: f32,
    /// When `true`, the sidechain signal is routed to the monitor output so
    /// the engineer can hear what the detector is reacting to.
    pub listen_mode: bool,
}

impl SidechainConfig {
    /// Create a default sidechain configuration (internal source, no filters).
    #[must_use]
    pub fn default() -> Self {
        Self {
            source: SidechainSource::Internal,
            hp_freq_hz: 0.0,
            lp_freq_hz: 0.0,
            listen_mode: false,
        }
    }

    /// Create a bandpass-filtered sidechain configuration using the internal source.
    ///
    /// `low` sets the high-pass frequency and `high` sets the low-pass frequency.
    #[must_use]
    pub fn bandpass_config(low: f32, high: f32) -> Self {
        Self {
            source: SidechainSource::Internal,
            hp_freq_hz: low.max(0.0),
            lp_freq_hz: high.max(0.0),
            listen_mode: false,
        }
    }

    /// Returns `true` if a high-pass filter is active.
    #[must_use]
    pub fn has_hp_filter(&self) -> bool {
        self.hp_freq_hz > 0.0
    }

    /// Returns `true` if a low-pass filter is active.
    #[must_use]
    pub fn has_lp_filter(&self) -> bool {
        self.lp_freq_hz > 0.0
    }

    /// Returns `true` if both HP and LP filters form a bandpass filter.
    #[must_use]
    pub fn is_bandpass(&self) -> bool {
        self.has_hp_filter() && self.has_lp_filter()
    }
}

/// Envelope follower that tracks the peak/RMS of the sidechain signal.
///
/// Uses a simple one-pole IIR filter for attack and release.
#[derive(Debug, Clone)]
pub struct SidechainDetector {
    /// Sidechain configuration.
    pub config: SidechainConfig,
    /// Current envelope level (0.0–1.0+).
    envelope: f32,
}

impl SidechainDetector {
    /// Create a new detector with the given configuration.
    #[must_use]
    pub fn new(config: SidechainConfig) -> Self {
        Self {
            config,
            envelope: 0.0,
        }
    }

    /// Feed one sample into the envelope follower.
    ///
    /// * `sample`  – the sidechain sample (absolute value is used internally)
    /// * `attack`  – attack coefficient (0.0–1.0; higher = faster)
    /// * `release` – release coefficient (0.0–1.0; higher = faster)
    ///
    /// Returns the updated envelope value.
    pub fn process(&mut self, sample: f32, attack: f32, release: f32) -> f32 {
        let level = sample.abs();
        let attack = attack.clamp(0.0, 1.0);
        let release = release.clamp(0.0, 1.0);

        if level > self.envelope {
            // Attack: envelope follows the rising signal
            self.envelope += attack * (level - self.envelope);
        } else {
            // Release: envelope decays toward the falling signal
            self.envelope += release * (level - self.envelope);
        }

        self.envelope
    }

    /// Return the current envelope level without advancing it.
    #[must_use]
    pub fn current_level(&self) -> f32 {
        self.envelope
    }

    /// Reset the envelope to zero (e.g. after a silence or scene change).
    pub fn reset(&mut self) {
        self.envelope = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- SidechainSource ----

    #[test]
    fn test_internal_is_not_external() {
        assert!(!SidechainSource::Internal.is_external());
    }

    #[test]
    fn test_external_is_external() {
        assert!(SidechainSource::External.is_external());
    }

    #[test]
    fn test_bus_is_external() {
        assert!(SidechainSource::Bus(3).is_external());
    }

    #[test]
    fn test_bus_id_is_preserved() {
        match SidechainSource::Bus(42) {
            SidechainSource::Bus(id) => assert_eq!(id, 42),
            _ => panic!("expected Bus variant"),
        }
    }

    // ---- SidechainConfig ----

    #[test]
    fn test_default_config_internal_source() {
        let cfg = SidechainConfig::default();
        assert_eq!(cfg.source, SidechainSource::Internal);
    }

    #[test]
    fn test_default_config_no_filters() {
        let cfg = SidechainConfig::default();
        assert!(!cfg.has_hp_filter());
        assert!(!cfg.has_lp_filter());
    }

    #[test]
    fn test_bandpass_config_sets_freqs() {
        let cfg = SidechainConfig::bandpass_config(200.0, 8000.0);
        assert!((cfg.hp_freq_hz - 200.0).abs() < f32::EPSILON);
        assert!((cfg.lp_freq_hz - 8000.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_bandpass_config_is_bandpass() {
        let cfg = SidechainConfig::bandpass_config(100.0, 5000.0);
        assert!(cfg.is_bandpass());
    }

    #[test]
    fn test_default_config_is_not_bandpass() {
        let cfg = SidechainConfig::default();
        assert!(!cfg.is_bandpass());
    }

    #[test]
    fn test_bandpass_config_negative_clamped_to_zero() {
        let cfg = SidechainConfig::bandpass_config(-100.0, -50.0);
        assert!(!cfg.has_hp_filter());
        assert!(!cfg.has_lp_filter());
    }

    // ---- SidechainDetector ----

    #[test]
    fn test_new_detector_envelope_is_zero() {
        let detector = SidechainDetector::new(SidechainConfig::default());
        assert!(detector.current_level().abs() < f32::EPSILON);
    }

    #[test]
    fn test_process_rises_on_attack() {
        let mut det = SidechainDetector::new(SidechainConfig::default());
        let level = det.process(1.0, 1.0, 0.0); // instant attack
        assert!((level - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_process_falls_on_release() {
        let mut det = SidechainDetector::new(SidechainConfig::default());
        // Fill envelope to 1.0
        det.process(1.0, 1.0, 0.0);
        // Release with instant release coefficient
        let level = det.process(0.0, 0.0, 1.0);
        assert!(level.abs() < f32::EPSILON);
    }

    #[test]
    fn test_current_level_does_not_change_envelope() {
        let mut det = SidechainDetector::new(SidechainConfig::default());
        det.process(0.5, 1.0, 0.0);
        let before = det.current_level();
        let after = det.current_level();
        assert!((before - after).abs() < f32::EPSILON);
    }

    #[test]
    fn test_process_uses_absolute_value() {
        let mut det_pos = SidechainDetector::new(SidechainConfig::default());
        let mut det_neg = SidechainDetector::new(SidechainConfig::default());
        det_pos.process(0.7, 1.0, 0.0);
        det_neg.process(-0.7, 1.0, 0.0);
        assert!((det_pos.current_level() - det_neg.current_level()).abs() < f32::EPSILON);
    }

    #[test]
    fn test_reset_clears_envelope() {
        let mut det = SidechainDetector::new(SidechainConfig::default());
        det.process(1.0, 1.0, 0.0);
        det.reset();
        assert!(det.current_level().abs() < f32::EPSILON);
    }

    #[test]
    fn test_attack_coefficient_partial() {
        let mut det = SidechainDetector::new(SidechainConfig::default());
        // With attack=0.5 and sample=1.0, envelope should be 0.5 after one sample
        let level = det.process(1.0, 0.5, 0.0);
        assert!((level - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_partial_release() {
        let mut det = SidechainDetector::new(SidechainConfig::default());
        det.process(1.0, 1.0, 0.0); // envelope = 1.0
        let level = det.process(0.0, 0.0, 0.5); // release halfway
        assert!((level - 0.5).abs() < 1e-6);
    }
}
