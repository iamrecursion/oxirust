#![allow(dead_code)]

//! Sidechain ducking effect for automated audio mixing.
//!
//! Ducking automatically lowers the level of a music or background track when
//! a voiceover or dialogue signal is detected. This module provides a
//! level-follower / envelope approach suitable for podcast production,
//! broadcast, and live-streaming scenarios.
//!
//! Supports both explicit sidechain buffer input (`process_buffers`) and
//! stereo sidechain pairs (`process_stereo_sidechain`) for true stereo ducking.

/// Configuration for the ducking effect.
#[derive(Debug, Clone)]
pub struct DuckingConfig {
    /// Threshold in dB below which the sidechain is considered inactive.
    pub threshold_db: f32,
    /// Amount of gain reduction in dB when ducking is active.
    pub duck_amount_db: f32,
    /// Attack time in milliseconds (how fast ducking engages).
    pub attack_ms: f32,
    /// Release time in milliseconds (how fast ducking releases).
    pub release_ms: f32,
    /// Hold time in milliseconds (minimum time ducking stays engaged).
    pub hold_ms: f32,
    /// Sample rate in Hz.
    pub sample_rate: f32,
}

impl Default for DuckingConfig {
    fn default() -> Self {
        Self {
            threshold_db: -30.0,
            duck_amount_db: -12.0,
            attack_ms: 10.0,
            release_ms: 200.0,
            hold_ms: 50.0,
            sample_rate: 48000.0,
        }
    }
}

impl DuckingConfig {
    /// Create a new ducking config.
    #[must_use]
    pub fn new(sample_rate: f32) -> Self {
        Self {
            sample_rate,
            ..Default::default()
        }
    }

    /// Set threshold.
    #[must_use]
    pub fn with_threshold(mut self, db: f32) -> Self {
        self.threshold_db = db;
        self
    }

    /// Set duck amount.
    #[must_use]
    pub fn with_duck_amount(mut self, db: f32) -> Self {
        self.duck_amount_db = db.min(0.0);
        self
    }

    /// Set attack time.
    #[must_use]
    pub fn with_attack(mut self, ms: f32) -> Self {
        self.attack_ms = ms.max(0.1);
        self
    }

    /// Set release time.
    #[must_use]
    pub fn with_release(mut self, ms: f32) -> Self {
        self.release_ms = ms.max(1.0);
        self
    }

    /// Set hold time.
    #[must_use]
    pub fn with_hold(mut self, ms: f32) -> Self {
        self.hold_ms = ms.max(0.0);
        self
    }
}

/// Convert decibels to linear gain.
#[allow(clippy::cast_precision_loss)]
fn db_to_linear(db: f32) -> f32 {
    10.0f32.powf(db / 20.0)
}

/// Convert linear gain to decibels.
#[allow(clippy::cast_precision_loss)]
fn linear_to_db(lin: f32) -> f32 {
    if lin <= 0.0 {
        -120.0
    } else {
        20.0 * lin.log10()
    }
}

/// Compute a one-pole smoothing coefficient from a time constant in ms.
#[allow(clippy::cast_precision_loss)]
fn time_constant(ms: f32, sample_rate: f32) -> f32 {
    if ms <= 0.0 || sample_rate <= 0.0 {
        return 1.0;
    }
    let samples = ms * 0.001 * sample_rate;
    (-1.0f32 / samples).exp()
}

/// Sidechain input mode for the ducker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SidechainMode {
    /// Use peak level of the sidechain signal.
    Peak,
    /// Use RMS level of the sidechain signal (smoother response).
    Rms,
    /// Use the maximum of left and right sidechain channels (stereo-aware).
    StereoMax,
}

/// Real-time sidechain ducker.
#[derive(Debug)]
pub struct Ducker {
    config: DuckingConfig,
    /// Current envelope follower value (linear).
    envelope: f32,
    /// Current gain reduction (linear, 0.0..1.0).
    gain: f32,
    /// Smoothing coefficient for attack.
    attack_coeff: f32,
    /// Smoothing coefficient for release.
    release_coeff: f32,
    /// Threshold in linear domain.
    threshold_linear: f32,
    /// Duck amount as linear gain multiplier.
    duck_gain: f32,
    /// Hold counter in samples.
    hold_counter: u32,
    /// Hold duration in samples.
    hold_samples: u32,
    /// Sidechain detection mode.
    sidechain_mode: SidechainMode,
    /// RMS accumulator for RMS mode.
    rms_accumulator: f32,
    /// RMS window counter.
    rms_count: u32,
    /// RMS window size in samples.
    rms_window: u32,
    /// Last computed RMS level.
    rms_level: f32,
}

impl Ducker {
    /// Create a new ducker with the given config.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    #[must_use]
    pub fn new(config: DuckingConfig) -> Self {
        let attack_coeff = time_constant(config.attack_ms, config.sample_rate);
        let release_coeff = time_constant(config.release_ms, config.sample_rate);
        let threshold_linear = db_to_linear(config.threshold_db);
        let duck_gain = db_to_linear(config.duck_amount_db);
        let hold_samples = (config.hold_ms * 0.001 * config.sample_rate) as u32;
        // RMS window: ~3ms for responsive metering
        let rms_window = (config.sample_rate * 0.003).max(1.0) as u32;
        Self {
            config,
            envelope: 0.0,
            gain: 1.0,
            attack_coeff,
            release_coeff,
            threshold_linear,
            duck_gain,
            hold_counter: 0,
            hold_samples,
            sidechain_mode: SidechainMode::Peak,
            rms_accumulator: 0.0,
            rms_count: 0,
            rms_window,
            rms_level: 0.0,
        }
    }

    /// Set the sidechain detection mode.
    pub fn set_sidechain_mode(&mut self, mode: SidechainMode) {
        self.sidechain_mode = mode;
    }

    /// Get the current sidechain mode.
    #[must_use]
    pub fn sidechain_mode(&self) -> SidechainMode {
        self.sidechain_mode
    }

    /// Compute the detection level from a raw sidechain absolute value.
    fn detect_level(&mut self, sidechain_abs: f32) -> f32 {
        match self.sidechain_mode {
            SidechainMode::Peak | SidechainMode::StereoMax => sidechain_abs,
            SidechainMode::Rms => {
                self.rms_accumulator += sidechain_abs * sidechain_abs;
                self.rms_count += 1;
                if self.rms_count >= self.rms_window {
                    self.rms_level = (self.rms_accumulator / self.rms_window as f32).sqrt();
                    self.rms_accumulator = 0.0;
                    self.rms_count = 0;
                }
                self.rms_level
            }
        }
    }

    /// Process one sample: given a sidechain level, return the gain to apply to the music track.
    pub fn process_sample(&mut self, sidechain_abs: f32) -> f32 {
        let level = self.detect_level(sidechain_abs);

        // Envelope follower (peak)
        if level > self.envelope {
            self.envelope = self.attack_coeff * self.envelope + (1.0 - self.attack_coeff) * level;
        } else {
            self.envelope = self.release_coeff * self.envelope + (1.0 - self.release_coeff) * level;
        }

        // Determine target gain
        let target = if self.envelope > self.threshold_linear {
            self.hold_counter = self.hold_samples;
            self.duck_gain
        } else if self.hold_counter > 0 {
            self.hold_counter -= 1;
            self.duck_gain
        } else {
            1.0
        };

        // Smooth gain transition
        if target < self.gain {
            self.gain = self.attack_coeff * self.gain + (1.0 - self.attack_coeff) * target;
        } else {
            self.gain = self.release_coeff * self.gain + (1.0 - self.release_coeff) * target;
        }

        self.gain
    }

    /// Apply ducking to a music buffer given a sidechain buffer (same length).
    pub fn process_buffers(&mut self, music: &mut [f32], sidechain: &[f32]) {
        let len = music.len().min(sidechain.len());
        for i in 0..len {
            let sc_abs = sidechain[i].abs();
            let gain = self.process_sample(sc_abs);
            music[i] *= gain;
        }
    }

    /// Apply ducking to stereo music using a stereo sidechain.
    ///
    /// Uses `StereoMax` detection: the louder of the two sidechain channels
    /// drives the ducking. Both music channels receive the same gain reduction
    /// to preserve stereo image.
    pub fn process_stereo_sidechain(
        &mut self,
        music_l: &mut [f32],
        music_r: &mut [f32],
        sidechain_l: &[f32],
        sidechain_r: &[f32],
    ) {
        let len = music_l
            .len()
            .min(music_r.len())
            .min(sidechain_l.len())
            .min(sidechain_r.len());

        for i in 0..len {
            // Take the louder of the two sidechain channels
            let sc_abs = sidechain_l[i].abs().max(sidechain_r[i].abs());
            let gain = self.process_sample(sc_abs);
            music_l[i] *= gain;
            music_r[i] *= gain;
        }
    }

    /// Reset internal state.
    pub fn reset(&mut self) {
        self.envelope = 0.0;
        self.gain = 1.0;
        self.hold_counter = 0;
        self.rms_accumulator = 0.0;
        self.rms_count = 0;
        self.rms_level = 0.0;
    }

    /// Get the current envelope value.
    #[must_use]
    pub fn envelope(&self) -> f32 {
        self.envelope
    }

    /// Get the current gain (1.0 = no ducking, lower = ducking active).
    #[must_use]
    pub fn current_gain(&self) -> f32 {
        self.gain
    }

    /// Get the current gain in dB.
    #[must_use]
    pub fn current_gain_db(&self) -> f32 {
        linear_to_db(self.gain)
    }

    /// Update sample rate (recalculates coefficients).
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub fn set_sample_rate(&mut self, sr: f32) {
        self.config.sample_rate = sr;
        self.attack_coeff = time_constant(self.config.attack_ms, sr);
        self.release_coeff = time_constant(self.config.release_ms, sr);
        self.hold_samples = (self.config.hold_ms * 0.001 * sr) as u32;
        self.rms_window = (sr * 0.003).max(1.0) as u32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_db_to_linear_zero() {
        let v = db_to_linear(0.0);
        assert!((v - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_db_to_linear_minus6() {
        let v = db_to_linear(-6.0206);
        assert!((v - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_linear_to_db_one() {
        let v = linear_to_db(1.0);
        assert!(v.abs() < 1e-5);
    }

    #[test]
    fn test_linear_to_db_zero() {
        let v = linear_to_db(0.0);
        assert_eq!(v, -120.0);
    }

    #[test]
    fn test_time_constant_positive() {
        let c = time_constant(10.0, 48000.0);
        assert!(c > 0.0 && c < 1.0);
    }

    #[test]
    fn test_time_constant_zero_ms() {
        let c = time_constant(0.0, 48000.0);
        assert!((c - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_ducker_initial_state() {
        let ducker = Ducker::new(DuckingConfig::default());
        assert!((ducker.current_gain() - 1.0).abs() < 1e-5);
        assert!((ducker.envelope() - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_ducker_no_sidechain() {
        let mut ducker = Ducker::new(DuckingConfig::default());
        // Feed silence on sidechain — gain should stay near 1.0
        for _ in 0..1000 {
            let g = ducker.process_sample(0.0);
            assert!(g > 0.99);
        }
    }

    #[test]
    fn test_ducker_with_loud_sidechain() {
        let config = DuckingConfig {
            threshold_db: -30.0,
            duck_amount_db: -12.0,
            attack_ms: 1.0,
            release_ms: 50.0,
            hold_ms: 0.0,
            sample_rate: 48000.0,
        };
        let mut ducker = Ducker::new(config);
        // Feed loud sidechain
        for _ in 0..4800 {
            ducker.process_sample(0.9);
        }
        // Gain should have decreased
        assert!(ducker.current_gain() < 0.5);
    }

    #[test]
    fn test_ducker_release() {
        let config = DuckingConfig {
            threshold_db: -30.0,
            duck_amount_db: -12.0,
            attack_ms: 1.0,
            release_ms: 10.0,
            hold_ms: 0.0,
            sample_rate: 48000.0,
        };
        let mut ducker = Ducker::new(config);
        // Engage ducking
        for _ in 0..4800 {
            ducker.process_sample(0.9);
        }
        let ducked_gain = ducker.current_gain();
        // Release
        for _ in 0..48000 {
            ducker.process_sample(0.0);
        }
        assert!(ducker.current_gain() > ducked_gain);
    }

    #[test]
    fn test_process_buffers() {
        let mut ducker = Ducker::new(DuckingConfig::default());
        let mut music = vec![1.0f32; 100];
        let sidechain = vec![0.0f32; 100];
        ducker.process_buffers(&mut music, &sidechain);
        // No sidechain: music should be approximately unchanged
        for &s in &music {
            assert!(s > 0.99);
        }
    }

    #[test]
    fn test_reset() {
        let mut ducker = Ducker::new(DuckingConfig::default());
        for _ in 0..1000 {
            ducker.process_sample(0.9);
        }
        ducker.reset();
        assert!((ducker.current_gain() - 1.0).abs() < 1e-5);
        assert!((ducker.envelope() - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_config_builder() {
        let cfg = DuckingConfig::new(44100.0)
            .with_threshold(-20.0)
            .with_duck_amount(-6.0)
            .with_attack(5.0)
            .with_release(100.0)
            .with_hold(30.0);
        assert!((cfg.sample_rate - 44100.0).abs() < 1e-5);
        assert!((cfg.threshold_db - (-20.0)).abs() < 1e-5);
        assert!((cfg.duck_amount_db - (-6.0)).abs() < 1e-5);
        assert!((cfg.attack_ms - 5.0).abs() < 1e-5);
        assert!((cfg.release_ms - 100.0).abs() < 1e-5);
        assert!((cfg.hold_ms - 30.0).abs() < 1e-5);
    }

    #[test]
    fn test_set_sample_rate() {
        let mut ducker = Ducker::new(DuckingConfig::new(48000.0));
        ducker.set_sample_rate(96000.0);
        assert!((ducker.config.sample_rate - 96000.0).abs() < 1e-5);
    }

    // --- Sidechain mode tests ---

    #[test]
    fn test_sidechain_mode_rms() {
        let config = DuckingConfig {
            threshold_db: -30.0,
            duck_amount_db: -12.0,
            attack_ms: 1.0,
            release_ms: 50.0,
            hold_ms: 0.0,
            sample_rate: 48000.0,
        };
        let mut ducker = Ducker::new(config);
        ducker.set_sidechain_mode(SidechainMode::Rms);
        assert_eq!(ducker.sidechain_mode(), SidechainMode::Rms);

        // Feed loud sidechain — should still duck
        for _ in 0..4800 {
            ducker.process_sample(0.9);
        }
        assert!(ducker.current_gain() < 0.5);
    }

    #[test]
    fn test_stereo_sidechain_ducking() {
        let config = DuckingConfig {
            threshold_db: -30.0,
            duck_amount_db: -12.0,
            attack_ms: 1.0,
            release_ms: 50.0,
            hold_ms: 0.0,
            sample_rate: 48000.0,
        };
        let mut ducker = Ducker::new(config);

        let mut music_l = vec![1.0f32; 4800];
        let mut music_r = vec![1.0f32; 4800];
        let sc_l = vec![0.9f32; 4800]; // Loud left sidechain
        let sc_r = vec![0.0f32; 4800]; // Silent right sidechain

        ducker.process_stereo_sidechain(&mut music_l, &mut music_r, &sc_l, &sc_r);

        // Both channels should be ducked (stereo max uses loud left channel)
        let last_l = music_l[4799];
        let last_r = music_r[4799];
        assert!(last_l < 0.5, "Left should be ducked, got {last_l}");
        assert!(last_r < 0.5, "Right should be ducked, got {last_r}");
        // Both channels get same gain to preserve stereo image
        assert!(
            (last_l - last_r).abs() < 1e-5,
            "Stereo image should be preserved"
        );
    }

    #[test]
    fn test_stereo_sidechain_no_trigger() {
        let config = DuckingConfig {
            threshold_db: -30.0,
            duck_amount_db: -12.0,
            attack_ms: 1.0,
            release_ms: 50.0,
            hold_ms: 0.0,
            sample_rate: 48000.0,
        };
        let mut ducker = Ducker::new(config);

        let mut music_l = vec![1.0f32; 100];
        let mut music_r = vec![1.0f32; 100];
        let sc_l = vec![0.0f32; 100];
        let sc_r = vec![0.0f32; 100];

        ducker.process_stereo_sidechain(&mut music_l, &mut music_r, &sc_l, &sc_r);

        // Silent sidechain: no ducking
        for &s in &music_l {
            assert!(s > 0.99);
        }
        for &s in &music_r {
            assert!(s > 0.99);
        }
    }
}
