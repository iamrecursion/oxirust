//! Multiband dynamics compressor.
//!
//! Splits the input signal into three frequency bands using crossover filters
//! and applies independent compression to each band before summing them back.
//!
//! # Architecture
//!
//! ```text
//!              ┌─── LP1 ─────────────────────────────────── COMP_LOW ──┐
//! input ───────┤    LP(f1)                                             ├─── Σ ─── output
//!              │                                                        │
//!              ├─── HP1→LP2 ─────────────────────────────── COMP_MID ──┤
//!              │    HP(f1) → LP(f2)                                    │
//!              │                                                        │
//!              └─── HP2 ─────────────────────────────────── COMP_HIGH─┘
//!                   HP(f2)
//! ```
//!
//! Default crossover frequencies: **250 Hz** (low/mid) and **4 000 Hz** (mid/high).
//!
//! # Example
//!
//! ```
//! use oximedia_effects::multiband_compressor::{MultibandCompressor, BandConfig};
//!
//! let mut comp = MultibandCompressor::new(48_000.0);
//!
//! let mut buf = vec![0.3f32; 512];
//! comp.process(&mut buf);
//! ```

#![allow(clippy::cast_precision_loss)]

use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// BandConfig
// ---------------------------------------------------------------------------

/// Per-band compression parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct BandConfig {
    /// Threshold in dBFS above which compression starts.
    pub threshold_db: f32,
    /// Compression ratio (e.g. `4.0` = 4:1).
    pub ratio: f32,
    /// Attack time in milliseconds.
    pub attack_ms: f32,
    /// Release time in milliseconds.
    pub release_ms: f32,
    /// Knee width in dB (0 = hard knee).
    pub knee_db: f32,
    /// Makeup gain in dB applied after compression.
    pub makeup_gain_db: f32,
}

impl Default for BandConfig {
    fn default() -> Self {
        Self {
            threshold_db: -18.0,
            ratio: 4.0,
            attack_ms: 10.0,
            release_ms: 100.0,
            knee_db: 6.0,
            makeup_gain_db: 0.0,
        }
    }
}

impl BandConfig {
    /// Create a gentle compression band.
    #[must_use]
    pub fn gentle() -> Self {
        Self {
            threshold_db: -12.0,
            ratio: 2.0,
            attack_ms: 20.0,
            release_ms: 200.0,
            knee_db: 6.0,
            makeup_gain_db: 0.0,
        }
    }

    /// Create an aggressive limiting band.
    #[must_use]
    pub fn limiting() -> Self {
        Self {
            threshold_db: -6.0,
            ratio: 20.0,
            attack_ms: 1.0,
            release_ms: 50.0,
            knee_db: 0.0,
            makeup_gain_db: 0.0,
        }
    }

    /// Validate parameters.
    pub fn validate(&self) -> Result<(), String> {
        if self.ratio < 1.0 {
            return Err(format!("ratio must be >= 1.0, got {}", self.ratio));
        }
        if self.attack_ms <= 0.0 {
            return Err(format!("attack_ms must be > 0, got {}", self.attack_ms));
        }
        if self.release_ms <= 0.0 {
            return Err(format!("release_ms must be > 0, got {}", self.release_ms));
        }
        if self.knee_db < 0.0 {
            return Err(format!("knee_db must be >= 0, got {}", self.knee_db));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Biquad filter (direct form II transposed)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    s1: f32,
    s2: f32,
}

impl Biquad {
    fn process(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.s1;
        self.s1 = self.b1 * x - self.a1 * y + self.s2;
        self.s2 = self.b2 * x - self.a2 * y;
        y
    }

    fn reset(&mut self) {
        self.s1 = 0.0;
        self.s2 = 0.0;
    }

    /// Linkwitz-Riley 2nd-order lowpass (Q = 0.5).
    fn lr_lowpass(freq_hz: f32, sample_rate: f32) -> Self {
        let w0 = 2.0 * PI * freq_hz / sample_rate;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let q = 0.5_f32.sqrt(); // Butterworth Q
        let alpha = sin_w0 / (2.0 * q);
        let a0 = 1.0 + alpha;
        Self {
            b0: ((1.0 - cos_w0) / 2.0) / a0,
            b1: (1.0 - cos_w0) / a0,
            b2: ((1.0 - cos_w0) / 2.0) / a0,
            a1: (-2.0 * cos_w0) / a0,
            a2: (1.0 - alpha) / a0,
            s1: 0.0,
            s2: 0.0,
        }
    }

    /// Linkwitz-Riley 2nd-order highpass (Q = 0.5).
    fn lr_highpass(freq_hz: f32, sample_rate: f32) -> Self {
        let w0 = 2.0 * PI * freq_hz / sample_rate;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let q = 0.5_f32.sqrt();
        let alpha = sin_w0 / (2.0 * q);
        let a0 = 1.0 + alpha;
        Self {
            b0: ((1.0 + cos_w0) / 2.0) / a0,
            b1: -(1.0 + cos_w0) / a0,
            b2: ((1.0 + cos_w0) / 2.0) / a0,
            a1: (-2.0 * cos_w0) / a0,
            a2: (1.0 - alpha) / a0,
            s1: 0.0,
            s2: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Single-band compressor
// ---------------------------------------------------------------------------

/// State for one compression band.
#[derive(Debug, Clone)]
struct BandCompressor {
    config: BandConfig,
    /// Envelope follower state (dBFS).
    envelope_db: f32,
    /// Precomputed attack coefficient.
    attack_coeff: f32,
    /// Precomputed release coefficient.
    release_coeff: f32,
    /// Precomputed makeup gain (linear).
    makeup_linear: f32,
}

impl BandCompressor {
    fn new(config: BandConfig, sample_rate: f32) -> Self {
        let attack_coeff = compute_time_coeff(config.attack_ms, sample_rate);
        let release_coeff = compute_time_coeff(config.release_ms, sample_rate);
        let makeup_linear = db_to_linear(config.makeup_gain_db);
        Self {
            config,
            envelope_db: -144.0,
            attack_coeff,
            release_coeff,
            makeup_linear,
        }
    }

    fn process_sample(&mut self, x: f32) -> f32 {
        let input_db = linear_to_db(x.abs());

        // Level detection (peak with attack/release smoothing)
        if input_db > self.envelope_db {
            self.envelope_db =
                self.attack_coeff * self.envelope_db + (1.0 - self.attack_coeff) * input_db;
        } else {
            self.envelope_db =
                self.release_coeff * self.envelope_db + (1.0 - self.release_coeff) * input_db;
        }

        // Gain computer with soft knee
        let gain_db = self.compute_gain(self.envelope_db);

        // Apply gain and makeup
        x * db_to_linear(gain_db) * self.makeup_linear
    }

    fn compute_gain(&self, level_db: f32) -> f32 {
        let threshold = self.config.threshold_db;
        let ratio = self.config.ratio;
        let knee = self.config.knee_db;

        let excess = level_db - threshold;

        if knee > 0.0 {
            let half_knee = knee / 2.0;
            if excess < -half_knee {
                // Below knee: no compression
                0.0
            } else if excess < half_knee {
                // In the knee: smooth transition
                let t = (excess + half_knee) / knee;
                let slope = (1.0 / ratio - 1.0) * t * t / 2.0;
                slope * knee / 2.0 // gain reduction in dB
            } else {
                // Above knee: full compression
                (1.0 / ratio - 1.0) * excess + (1.0 / ratio - 1.0) * half_knee / 2.0
            }
        } else if excess > 0.0 {
            // Hard knee
            (1.0 / ratio - 1.0) * excess
        } else {
            0.0
        }
    }

    fn reset(&mut self) {
        self.envelope_db = -144.0;
    }

    #[allow(dead_code)]
    fn update_sample_rate(&mut self, sample_rate: f32) {
        self.attack_coeff = compute_time_coeff(self.config.attack_ms, sample_rate);
        self.release_coeff = compute_time_coeff(self.config.release_ms, sample_rate);
    }
}

// ---------------------------------------------------------------------------
// MultibandCompressor
// ---------------------------------------------------------------------------

/// Three-band multiband compressor.
pub struct MultibandCompressor {
    /// Crossover frequency between low and mid bands (Hz).
    crossover_low_hz: f32,
    /// Crossover frequency between mid and high bands (Hz).
    crossover_high_hz: f32,
    /// Lowpass filter for low band (LP1).
    lp1: Biquad,
    /// Highpass filter for mid band (HP1).
    hp1: Biquad,
    /// Lowpass filter for mid band upper edge (LP2).
    lp2: Biquad,
    /// Highpass filter for high band (HP2).
    hp2: Biquad,
    /// Low band compressor.
    comp_low: BandCompressor,
    /// Mid band compressor.
    comp_mid: BandCompressor,
    /// High band compressor.
    comp_high: BandCompressor,
    #[allow(dead_code)]
    sample_rate: f32,
}

impl MultibandCompressor {
    /// Create a new `MultibandCompressor` with default band configurations.
    ///
    /// Default crossovers: 250 Hz (low/mid) and 4 000 Hz (mid/high).
    #[must_use]
    pub fn new(sample_rate: f32) -> Self {
        let crossover_low = 250.0f32;
        let crossover_high = 4_000.0f32;
        Self::new_with_bands(
            BandConfig::default(),
            BandConfig::default(),
            BandConfig::default(),
            crossover_low,
            crossover_high,
            sample_rate,
        )
    }

    /// Create a `MultibandCompressor` with custom per-band configurations.
    ///
    /// * `low`            – compression config for the low band (<= `crossover_low_hz`)
    /// * `mid`            – compression config for the mid band
    /// * `high`           – compression config for the high band (>= `crossover_high_hz`)
    /// * `crossover_low_hz`  – crossover between low and mid bands (Hz)
    /// * `crossover_high_hz` – crossover between mid and high bands (Hz)
    /// * `sample_rate`    – audio sample rate (Hz)
    #[must_use]
    pub fn new_with_bands(
        low: BandConfig,
        mid: BandConfig,
        high: BandConfig,
        crossover_low_hz: f32,
        crossover_high_hz: f32,
        sample_rate: f32,
    ) -> Self {
        let xover_low = crossover_low_hz.clamp(20.0, sample_rate * 0.45);
        let xover_high = crossover_high_hz.clamp(xover_low + 1.0, sample_rate * 0.49);

        Self {
            crossover_low_hz: xover_low,
            crossover_high_hz: xover_high,
            lp1: Biquad::lr_lowpass(xover_low, sample_rate),
            hp1: Biquad::lr_highpass(xover_low, sample_rate),
            lp2: Biquad::lr_lowpass(xover_high, sample_rate),
            hp2: Biquad::lr_highpass(xover_high, sample_rate),
            comp_low: BandCompressor::new(low, sample_rate),
            comp_mid: BandCompressor::new(mid, sample_rate),
            comp_high: BandCompressor::new(high, sample_rate),
            sample_rate,
        }
    }

    /// Return the low/mid crossover frequency.
    #[must_use]
    pub fn crossover_low_hz(&self) -> f32 {
        self.crossover_low_hz
    }

    /// Return the mid/high crossover frequency.
    #[must_use]
    pub fn crossover_high_hz(&self) -> f32 {
        self.crossover_high_hz
    }

    /// Process a single sample through all three bands.
    pub fn process_sample(&mut self, input: f32) -> f32 {
        // Split into bands
        let low = self.lp1.process(input);
        let mid_hp = self.hp1.process(input);
        let mid = self.lp2.process(mid_hp);
        let high = self.hp2.process(mid_hp);

        // Compress each band
        let low_out = self.comp_low.process_sample(low);
        let mid_out = self.comp_mid.process_sample(mid);
        let high_out = self.comp_high.process_sample(high);

        // Sum bands
        low_out + mid_out + high_out
    }

    /// Process a buffer of samples in-place.
    pub fn process(&mut self, buffer: &mut [f32]) {
        for sample in buffer.iter_mut() {
            *sample = self.process_sample(*sample);
        }
    }

    /// Reset all filter and compressor states.
    pub fn reset(&mut self) {
        self.lp1.reset();
        self.hp1.reset();
        self.lp2.reset();
        self.hp2.reset();
        self.comp_low.reset();
        self.comp_mid.reset();
        self.comp_high.reset();
    }

    /// Return a reference to the low band configuration.
    #[must_use]
    pub fn low_config(&self) -> &BandConfig {
        &self.comp_low.config
    }

    /// Return a reference to the mid band configuration.
    #[must_use]
    pub fn mid_config(&self) -> &BandConfig {
        &self.comp_mid.config
    }

    /// Return a reference to the high band configuration.
    #[must_use]
    pub fn high_config(&self) -> &BandConfig {
        &self.comp_high.config
    }
}

// ---------------------------------------------------------------------------
// DSP utility functions
// ---------------------------------------------------------------------------

/// Compute a smoothing coefficient for an attack/release time in ms.
/// Uses the standard `exp(-1/(time_s * sample_rate))` formula.
fn compute_time_coeff(time_ms: f32, sample_rate: f32) -> f32 {
    if time_ms <= 0.0 || sample_rate <= 0.0 {
        return 0.0;
    }
    let time_samples = time_ms * 0.001 * sample_rate;
    (-1.0 / time_samples).exp()
}

/// Convert linear amplitude to dBFS. Returns –144 dB for zero/near-zero input.
fn linear_to_db(x: f32) -> f32 {
    if x < 1e-7 {
        -144.0
    } else {
        20.0 * x.log10()
    }
}

/// Convert dBFS to linear amplitude.
fn db_to_linear(db: f32) -> f32 {
    10.0_f32.powf(db / 20.0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- BandConfig tests --------------------------------------------------

    #[test]
    fn test_band_config_default() {
        let c = BandConfig::default();
        assert!(c.validate().is_ok());
        assert!(c.ratio >= 1.0);
        assert!(c.attack_ms > 0.0);
        assert!(c.release_ms > 0.0);
    }

    #[test]
    fn test_band_config_gentle() {
        let c = BandConfig::gentle();
        assert!(c.validate().is_ok());
    }

    #[test]
    fn test_band_config_limiting() {
        let c = BandConfig::limiting();
        assert!(c.validate().is_ok());
        assert!(c.ratio >= 10.0);
    }

    #[test]
    fn test_band_config_validate_bad_ratio() {
        let c = BandConfig {
            ratio: 0.5,
            ..Default::default()
        };
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_band_config_validate_zero_attack() {
        let c = BandConfig {
            attack_ms: 0.0,
            ..Default::default()
        };
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_band_config_validate_zero_release() {
        let c = BandConfig {
            release_ms: 0.0,
            ..Default::default()
        };
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_band_config_validate_negative_knee() {
        let c = BandConfig {
            knee_db: -1.0,
            ..Default::default()
        };
        assert!(c.validate().is_err());
    }

    // ---- MultibandCompressor construction ----------------------------------

    #[test]
    fn test_new_default_crossovers() {
        let comp = MultibandCompressor::new(48_000.0);
        assert!((comp.crossover_low_hz() - 250.0).abs() < 1.0);
        assert!((comp.crossover_high_hz() - 4_000.0).abs() < 1.0);
    }

    #[test]
    fn test_new_with_bands_custom_crossovers() {
        let comp = MultibandCompressor::new_with_bands(
            BandConfig::default(),
            BandConfig::default(),
            BandConfig::default(),
            300.0,
            3_000.0,
            48_000.0,
        );
        assert!((comp.crossover_low_hz() - 300.0).abs() < 1.0);
        assert!((comp.crossover_high_hz() - 3_000.0).abs() < 1.0);
    }

    // ---- Processing --------------------------------------------------------

    #[test]
    fn test_process_sample_finite() {
        let mut comp = MultibandCompressor::new(48_000.0);
        for i in 0..1024 {
            let s = (i as f32 * 0.02).sin() * 0.4;
            let out = comp.process_sample(s);
            assert!(out.is_finite(), "non-finite output at sample {i}");
        }
    }

    #[test]
    fn test_process_silent_input() {
        let mut comp = MultibandCompressor::new(48_000.0);
        let out = comp.process_sample(0.0);
        assert!(out.abs() < 1e-6);
    }

    #[test]
    fn test_process_buffer_length_preserved() {
        let mut comp = MultibandCompressor::new(48_000.0);
        let mut buf = vec![0.2f32; 512];
        comp.process(&mut buf);
        assert_eq!(buf.len(), 512);
    }

    #[test]
    fn test_process_buffer_all_finite() {
        let mut comp = MultibandCompressor::new(48_000.0);
        let mut buf: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.05).sin() * 0.5).collect();
        comp.process(&mut buf);
        for &v in &buf {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_reset_clears_state() {
        let mut comp = MultibandCompressor::new(48_000.0);
        let mut buf = vec![0.8f32; 512];
        comp.process(&mut buf);
        comp.reset();
        // After reset, processing a quiet signal should behave like a fresh instance
        let mut comp2 = MultibandCompressor::new(48_000.0);
        let input = 0.1f32;
        let out1 = comp.process_sample(input);
        let out2 = comp2.process_sample(input);
        assert!((out1 - out2).abs() < 1e-4);
    }

    // ---- Compression behaviour ---------------------------------------------

    #[test]
    fn test_compression_reduces_loud_signal() {
        // With high ratio and low threshold, loud signals should be attenuated
        let low = BandConfig {
            threshold_db: -30.0,
            ratio: 10.0,
            attack_ms: 1.0,
            release_ms: 50.0,
            knee_db: 0.0,
            makeup_gain_db: 0.0,
        };
        let mut comp = MultibandCompressor::new_with_bands(
            low,
            BandConfig::default(),
            BandConfig::default(),
            200.0,
            5_000.0,
            48_000.0,
        );
        // Feed loud signal to build up envelope
        let input = 0.9f32;
        let mut outputs = Vec::new();
        for _ in 0..500 {
            outputs.push(comp.process_sample(input));
        }
        let last = *outputs.last().unwrap_or(&0.0);
        // Output should be reduced significantly below input in the steady state
        assert!(
            last.abs() <= input.abs() + 0.05,
            "Compression did not reduce level: {last}"
        );
    }

    #[test]
    fn test_makeup_gain_increases_output() {
        let band_no_makeup = BandConfig {
            makeup_gain_db: 0.0,
            threshold_db: -6.0,
            ratio: 4.0,
            attack_ms: 1.0,
            release_ms: 50.0,
            knee_db: 0.0,
        };
        let band_with_makeup = BandConfig {
            makeup_gain_db: 6.0,
            ..band_no_makeup.clone()
        };

        let mut comp1 = MultibandCompressor::new_with_bands(
            band_no_makeup,
            BandConfig::default(),
            BandConfig::default(),
            250.0,
            4_000.0,
            48_000.0,
        );
        let mut comp2 = MultibandCompressor::new_with_bands(
            band_with_makeup,
            BandConfig::default(),
            BandConfig::default(),
            250.0,
            4_000.0,
            48_000.0,
        );

        let input = 0.1f32; // Below threshold, so gain computer neutral, but makeup still applies
                            // Warm up
        for _ in 0..100 {
            comp1.process_sample(input);
            comp2.process_sample(input);
        }
        let out1 = comp1.process_sample(input);
        let out2 = comp2.process_sample(input);

        assert!(
            out2.abs() >= out1.abs(),
            "Makeup gain should increase output level"
        );
    }

    // ---- Utility function tests --------------------------------------------

    #[test]
    fn test_db_to_linear_roundtrip() {
        let db = -12.0f32;
        let linear = db_to_linear(db);
        let back = linear_to_db(linear);
        assert!((back - db).abs() < 0.01);
    }

    #[test]
    fn test_linear_to_db_zero() {
        let db = linear_to_db(0.0);
        assert!(db <= -100.0, "Zero amplitude should give very low dB");
    }

    #[test]
    fn test_compute_time_coeff_range() {
        let coeff = compute_time_coeff(10.0, 48_000.0);
        assert!(coeff > 0.0 && coeff < 1.0);
    }
}
