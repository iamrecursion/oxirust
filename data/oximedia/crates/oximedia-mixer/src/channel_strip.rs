#![allow(dead_code)]
//! Channel strip processing chain for the mixer.
//!
//! A channel strip represents the complete signal processing chain for a single
//! mixer channel, including input gain, high-pass filter, EQ, dynamics, insert
//! effects, fader, pan, and output routing. This module models the typical
//! analog console channel strip in a digital workflow.

use serde::{Deserialize, Serialize};

/// Processing stage within a channel strip.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StripStage {
    /// Input gain/trim stage.
    InputGain,
    /// High-pass filter stage.
    HighPass,
    /// Equalizer stage.
    Equalizer,
    /// Dynamics (compressor/gate) stage.
    Dynamics,
    /// Insert effect slot.
    Insert,
    /// Channel fader stage.
    Fader,
    /// Pan control stage.
    Pan,
    /// Output/routing stage.
    Output,
}

/// High-pass filter configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HighPassFilter {
    /// Whether the filter is enabled.
    pub enabled: bool,
    /// Cutoff frequency in Hz.
    pub frequency: f32,
    /// Filter slope in dB/octave (typically 6, 12, 18, or 24).
    pub slope_db_per_octave: u32,
    /// Internal state.
    prev_output: f32,
    /// Previous input sample for filtering.
    prev_input: f32,
}

impl Default for HighPassFilter {
    fn default() -> Self {
        Self {
            enabled: false,
            frequency: 80.0,
            slope_db_per_octave: 12,
            prev_output: 0.0,
            prev_input: 0.0,
        }
    }
}

impl HighPassFilter {
    /// Creates a new high-pass filter with the given cutoff frequency.
    #[must_use]
    pub fn new(frequency: f32) -> Self {
        Self {
            enabled: true,
            frequency: frequency.clamp(20.0, 20000.0),
            ..Default::default()
        }
    }

    /// Processes a single sample through the high-pass filter.
    #[allow(clippy::cast_precision_loss)]
    pub fn process(&mut self, input: f32, sample_rate: u32) -> f32 {
        if !self.enabled {
            return input;
        }
        let rc = 1.0 / (2.0 * std::f32::consts::PI * self.frequency);
        let dt = 1.0 / sample_rate as f32;
        let alpha = rc / (rc + dt);
        let output = alpha * (self.prev_output + input - self.prev_input);
        self.prev_input = input;
        self.prev_output = output;
        output
    }

    /// Resets the filter state.
    pub fn reset(&mut self) {
        self.prev_output = 0.0;
        self.prev_input = 0.0;
    }
}

/// Parametric EQ band for the channel strip.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StripEqBand {
    /// Whether this band is enabled.
    pub enabled: bool,
    /// Center frequency in Hz.
    pub frequency: f32,
    /// Gain in dB (-18.0 to +18.0).
    pub gain_db: f32,
    /// Q factor (bandwidth).
    pub q: f32,
}

impl Default for StripEqBand {
    fn default() -> Self {
        Self {
            enabled: true,
            frequency: 1000.0,
            gain_db: 0.0,
            q: 1.0,
        }
    }
}

/// EQ section with four parametric bands.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StripEq {
    /// Whether the entire EQ section is enabled.
    pub enabled: bool,
    /// Low band (typically shelf or bell, 20-500 Hz).
    pub low: StripEqBand,
    /// Low-mid band (200-2000 Hz).
    pub low_mid: StripEqBand,
    /// High-mid band (1000-8000 Hz).
    pub high_mid: StripEqBand,
    /// High band (2000-20000 Hz).
    pub high: StripEqBand,
}

impl Default for StripEq {
    fn default() -> Self {
        Self {
            enabled: true,
            low: StripEqBand {
                frequency: 100.0,
                ..Default::default()
            },
            low_mid: StripEqBand {
                frequency: 500.0,
                ..Default::default()
            },
            high_mid: StripEqBand {
                frequency: 3000.0,
                ..Default::default()
            },
            high: StripEqBand {
                frequency: 10000.0,
                ..Default::default()
            },
        }
    }
}

impl StripEq {
    /// Applies the EQ to a sample (simplified single-sample processing).
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn process(&self, input: f32) -> f32 {
        if !self.enabled {
            return input;
        }
        let mut output = input;
        for band in &[&self.low, &self.low_mid, &self.high_mid, &self.high] {
            if band.enabled && band.gain_db.abs() > 0.01 {
                let gain_linear = 10.0_f32.powf(band.gain_db / 20.0);
                output *= gain_linear;
            }
        }
        output
    }
}

/// Dynamics section (simplified compressor + gate).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StripDynamics {
    /// Whether the dynamics section is enabled.
    pub enabled: bool,
    /// Compressor threshold in dB.
    pub comp_threshold_db: f32,
    /// Compressor ratio (e.g., 4.0 = 4:1).
    pub comp_ratio: f32,
    /// Compressor attack time in milliseconds.
    pub comp_attack_ms: f32,
    /// Compressor release time in milliseconds.
    pub comp_release_ms: f32,
    /// Gate threshold in dB.
    pub gate_threshold_db: f32,
    /// Whether the gate is enabled.
    pub gate_enabled: bool,
    /// Internal envelope follower state.
    envelope: f32,
}

impl Default for StripDynamics {
    fn default() -> Self {
        Self {
            enabled: false,
            comp_threshold_db: -20.0,
            comp_ratio: 4.0,
            comp_attack_ms: 10.0,
            comp_release_ms: 100.0,
            gate_threshold_db: -60.0,
            gate_enabled: false,
            envelope: 0.0,
        }
    }
}

impl StripDynamics {
    /// Processes a sample through the dynamics chain.
    #[allow(clippy::cast_precision_loss)]
    pub fn process(&mut self, input: f32, sample_rate: u32) -> f32 {
        if !self.enabled {
            return input;
        }
        let level_db = if input.abs() > 1e-10 {
            20.0 * input.abs().log10()
        } else {
            -100.0
        };

        // Gate
        if self.gate_enabled && level_db < self.gate_threshold_db {
            return 0.0;
        }

        // Compressor
        if level_db > self.comp_threshold_db {
            let over = level_db - self.comp_threshold_db;
            let compressed_over = over / self.comp_ratio;
            let target_db = self.comp_threshold_db + compressed_over;
            let gain_db = target_db - level_db;

            // Smooth envelope
            let attack_coeff = (-1.0 / (self.comp_attack_ms * 0.001 * sample_rate as f32)).exp();
            let release_coeff = (-1.0 / (self.comp_release_ms * 0.001 * sample_rate as f32)).exp();
            let coeff = if gain_db < self.envelope {
                attack_coeff
            } else {
                release_coeff
            };
            self.envelope = coeff * self.envelope + (1.0 - coeff) * gain_db;

            let gain_linear = 10.0_f32.powf(self.envelope / 20.0);
            return input * gain_linear;
        }

        input
    }

    /// Resets the dynamics state.
    pub fn reset(&mut self) {
        self.envelope = 0.0;
    }
}

/// Complete channel strip configuration and processing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelStrip {
    /// Channel strip name.
    pub name: String,
    /// Input gain/trim in dB.
    pub input_gain_db: f32,
    /// Phase invert.
    pub phase_invert: bool,
    /// High-pass filter.
    pub high_pass: HighPassFilter,
    /// Parametric EQ.
    pub eq: StripEq,
    /// Dynamics (compressor + gate).
    pub dynamics: StripDynamics,
    /// Fader level (0.0 = -inf, 1.0 = unity).
    pub fader: f32,
    /// Pan position (-1.0 = left, 0.0 = center, 1.0 = right).
    pub pan: f32,
    /// Mute state.
    pub mute: bool,
    /// Solo state.
    pub solo: bool,
    /// Sample rate.
    sample_rate: u32,
    /// Processing stage order (for reordering).
    stage_order: Vec<StripStage>,
}

impl ChannelStrip {
    /// Creates a new channel strip with the given name and sample rate.
    #[must_use]
    pub fn new(name: &str, sample_rate: u32) -> Self {
        Self {
            name: name.to_string(),
            input_gain_db: 0.0,
            phase_invert: false,
            high_pass: HighPassFilter::default(),
            eq: StripEq::default(),
            dynamics: StripDynamics::default(),
            fader: 1.0,
            pan: 0.0,
            mute: false,
            solo: false,
            sample_rate,
            stage_order: vec![
                StripStage::InputGain,
                StripStage::HighPass,
                StripStage::Equalizer,
                StripStage::Dynamics,
                StripStage::Fader,
                StripStage::Pan,
            ],
        }
    }

    /// Processes a mono sample through the entire channel strip.
    pub fn process_sample(&mut self, input: f32) -> f32 {
        if self.mute {
            return 0.0;
        }

        let mut sample = input;

        // Input gain
        let gain_linear = 10.0_f32.powf(self.input_gain_db / 20.0);
        sample *= gain_linear;

        // Phase invert
        if self.phase_invert {
            sample = -sample;
        }

        // High-pass filter
        sample = self.high_pass.process(sample, self.sample_rate);

        // EQ
        sample = self.eq.process(sample);

        // Dynamics
        sample = self.dynamics.process(sample, self.sample_rate);

        // Fader
        sample *= self.fader;

        sample
    }

    /// Processes a mono sample and returns stereo (left, right) after panning.
    pub fn process_sample_stereo(&mut self, input: f32) -> (f32, f32) {
        let mono = self.process_sample(input);
        let pan_norm = (self.pan + 1.0) * 0.5; // 0..1
        let left = mono * (1.0 - pan_norm).sqrt();
        let right = mono * pan_norm.sqrt();
        (left, right)
    }

    /// Processes a block of mono samples in-place.
    pub fn process_block(&mut self, samples: &mut [f32]) {
        for s in samples.iter_mut() {
            *s = self.process_sample(*s);
        }
    }

    /// Sets the input gain in dB.
    pub fn set_input_gain_db(&mut self, gain_db: f32) {
        self.input_gain_db = gain_db.clamp(-60.0, 24.0);
    }

    /// Gets the input gain in dB.
    #[must_use]
    pub fn input_gain_db(&self) -> f32 {
        self.input_gain_db
    }

    /// Sets the fader level (0.0 = -inf, 1.0 = unity, >1.0 = boost).
    pub fn set_fader(&mut self, level: f32) {
        self.fader = level.clamp(0.0, 2.0);
    }

    /// Gets the fader level.
    #[must_use]
    pub fn fader(&self) -> f32 {
        self.fader
    }

    /// Sets the pan position (-1.0 left, 0.0 center, 1.0 right).
    pub fn set_pan(&mut self, pan: f32) {
        self.pan = pan.clamp(-1.0, 1.0);
    }

    /// Gets the pan position.
    #[must_use]
    pub fn pan(&self) -> f32 {
        self.pan
    }

    /// Toggles mute state, returns new state.
    pub fn toggle_mute(&mut self) -> bool {
        self.mute = !self.mute;
        self.mute
    }

    /// Toggles solo state, returns new state.
    pub fn toggle_solo(&mut self) -> bool {
        self.solo = !self.solo;
        self.solo
    }

    /// Resets all processing state (filters, dynamics envelope, etc.).
    pub fn reset(&mut self) {
        self.high_pass.reset();
        self.dynamics.reset();
    }

    /// Gets the current stage order.
    #[must_use]
    pub fn stage_order(&self) -> &[StripStage] {
        &self.stage_order
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_strip_creation() {
        let strip = ChannelStrip::new("Vocal", 48000);
        assert_eq!(strip.name, "Vocal");
        assert!((strip.fader - 1.0).abs() < f32::EPSILON);
        assert!(strip.pan.abs() < f32::EPSILON);
        assert!(!strip.mute);
        assert!(!strip.solo);
    }

    #[test]
    fn test_channel_strip_mute() {
        let mut strip = ChannelStrip::new("Test", 48000);
        let out = strip.process_sample(1.0);
        assert!(out.abs() > 0.0);
        strip.toggle_mute();
        assert!(strip.mute);
        let out = strip.process_sample(1.0);
        assert!(out.abs() < f32::EPSILON);
    }

    #[test]
    fn test_channel_strip_phase_invert() {
        let mut strip = ChannelStrip::new("Test", 48000);
        strip.phase_invert = true;
        let out = strip.process_sample(1.0);
        assert!(out < 0.0);
    }

    #[test]
    fn test_channel_strip_pan_center() {
        let mut strip = ChannelStrip::new("Test", 48000);
        strip.set_pan(0.0);
        let (left, right) = strip.process_sample_stereo(1.0);
        assert!((left - right).abs() < 0.01);
    }

    #[test]
    fn test_channel_strip_pan_left() {
        let mut strip = ChannelStrip::new("Test", 48000);
        strip.set_pan(-1.0);
        let (left, right) = strip.process_sample_stereo(1.0);
        assert!(left > right);
        assert!(right.abs() < 0.01);
    }

    #[test]
    fn test_channel_strip_pan_right() {
        let mut strip = ChannelStrip::new("Test", 48000);
        strip.set_pan(1.0);
        let (left, right) = strip.process_sample_stereo(1.0);
        assert!(right > left);
        assert!(left.abs() < 0.01);
    }

    #[test]
    fn test_channel_strip_input_gain() {
        let mut strip = ChannelStrip::new("Test", 48000);
        strip.set_input_gain_db(6.0);
        assert!((strip.input_gain_db() - 6.0).abs() < f32::EPSILON);
        let out = strip.process_sample(0.5);
        // +6 dB ~ 2x gain, so output should be roughly 1.0
        assert!(out > 0.9 && out < 1.1);
    }

    #[test]
    fn test_channel_strip_fader() {
        let mut strip = ChannelStrip::new("Test", 48000);
        strip.set_fader(0.5);
        assert!((strip.fader() - 0.5).abs() < f32::EPSILON);
        let out = strip.process_sample(1.0);
        assert!((out - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_channel_strip_fader_clamping() {
        let mut strip = ChannelStrip::new("Test", 48000);
        strip.set_fader(5.0);
        assert!((strip.fader() - 2.0).abs() < f32::EPSILON);
        strip.set_fader(-1.0);
        assert!(strip.fader().abs() < f32::EPSILON);
    }

    #[test]
    fn test_high_pass_filter() {
        let mut hpf = HighPassFilter::new(1000.0);
        assert!(hpf.enabled);
        assert!((hpf.frequency - 1000.0).abs() < f32::EPSILON);
        // Process some samples
        for _ in 0..100 {
            hpf.process(0.5, 48000);
        }
        hpf.reset();
    }

    #[test]
    fn test_strip_eq_bypass() {
        let mut eq = StripEq::default();
        eq.enabled = false;
        let result = eq.process(0.5);
        assert!((result - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_strip_dynamics_bypass() {
        let mut dyn_proc = StripDynamics::default();
        dyn_proc.enabled = false;
        let result = dyn_proc.process(0.5, 48000);
        assert!((result - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_channel_strip_process_block() {
        let mut strip = ChannelStrip::new("Test", 48000);
        strip.set_fader(0.5);
        let mut block = vec![1.0_f32; 64];
        strip.process_block(&mut block);
        for s in &block {
            assert!((*s - 0.5).abs() < 0.01);
        }
    }

    #[test]
    fn test_channel_strip_reset() {
        let mut strip = ChannelStrip::new("Test", 48000);
        strip.high_pass = HighPassFilter::new(100.0);
        strip.dynamics.enabled = true;
        for _ in 0..100 {
            strip.process_sample(0.5);
        }
        strip.reset();
        // After reset, internal states should be zeroed
    }

    #[test]
    fn test_channel_strip_toggle() {
        let mut strip = ChannelStrip::new("Test", 48000);
        assert!(!strip.mute);
        assert!(strip.toggle_mute());
        assert!(strip.mute);
        assert!(!strip.toggle_mute());

        assert!(!strip.solo);
        assert!(strip.toggle_solo());
        assert!(strip.solo);
    }

    #[test]
    fn test_stage_order() {
        let strip = ChannelStrip::new("Test", 48000);
        let order = strip.stage_order();
        assert!(!order.is_empty());
        assert_eq!(order[0], StripStage::InputGain);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Full parametric channel strip (biquad EQ + dynamics + constant-power pan)
// ═══════════════════════════════════════════════════════════════════════════

/// Full parametric channel strip with biquad EQ, gate, compressor, and pan.
///
/// This submodule provides the higher-fidelity channel strip types that operate
/// on blocks of samples with proper per-sample stateful DSP (biquad filters,
/// envelope followers).
pub mod parametric {
    use std::f32::consts::PI;

    // ─────────────────────────────────────────────────────── EqBandType ───

    /// Filter type for a [`ParametricEqBand`].
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum EqBandType {
        /// Bell/peaking EQ.
        Peaking,
        /// Low-frequency shelving filter.
        LowShelf,
        /// High-frequency shelving filter.
        HighShelf,
        /// 2nd-order Butterworth low-pass.
        LowPass,
        /// 2nd-order Butterworth high-pass.
        HighPass,
        /// Notch (band-reject) filter.
        Notch,
        /// Band-pass filter (constant skirt gain).
        BandPass,
        /// All-pass phase rotator.
        AllPass,
    }

    // ─────────────────────────────────────────────────── ParametricEqBand ───

    /// Stateful biquad EQ band using Direct Form II transposed.
    ///
    /// Coefficients are computed once in [`new`](ParametricEqBand::new) using
    /// the Audio EQ Cookbook bilinear-transform formulas.  Call
    /// [`process_sample`](ParametricEqBand::process_sample) per sample.
    #[derive(Debug, Clone)]
    pub struct ParametricEqBand {
        /// Band is active.
        pub enabled: bool,
        /// Centre / corner frequency in Hz.
        pub frequency_hz: f32,
        /// Gain in dB (used by Peaking, LowShelf, HighShelf).
        pub gain_db: f32,
        /// Quality factor (0.1 … 10.0).
        pub q: f32,
        /// Filter type.
        pub band_type: EqBandType,
        // Biquad coefficients (normalised: a0 = 1)
        b0: f32,
        b1: f32,
        b2: f32,
        a1: f32,
        a2: f32,
        // Direct Form II state
        x1: f32,
        x2: f32,
        y1: f32,
        y2: f32,
    }

    impl ParametricEqBand {
        /// Create a new band and compute its biquad coefficients.
        #[must_use]
        pub fn new(
            freq_hz: f32,
            gain_db: f32,
            q: f32,
            band_type: EqBandType,
            sample_rate: u32,
        ) -> Self {
            let mut band = Self {
                enabled: true,
                frequency_hz: freq_hz,
                gain_db,
                q: q.clamp(0.1, 10.0),
                band_type,
                b0: 1.0,
                b1: 0.0,
                b2: 0.0,
                a1: 0.0,
                a2: 0.0,
                x1: 0.0,
                x2: 0.0,
                y1: 0.0,
                y2: 0.0,
            };
            band.compute_coefficients(sample_rate);
            band
        }

        /// Recompute biquad coefficients (call after changing parameters).
        pub fn compute_coefficients(&mut self, sample_rate: u32) {
            #[allow(clippy::cast_precision_loss)]
            let fs = sample_rate as f32;
            let f0 = self.frequency_hz.clamp(1.0, fs * 0.499);
            let q = self.q.clamp(0.001, 100.0);
            let a_lin = 10.0_f32.powf(self.gain_db / 40.0); // sqrt(10^(dB/20))
            let w0 = 2.0 * PI * f0 / fs;
            let cos_w0 = w0.cos();
            let sin_w0 = w0.sin();
            let alpha = sin_w0 / (2.0 * q);

            let (b0, b1, b2, a0, a1, a2) = match self.band_type {
                EqBandType::Peaking => {
                    let b0 = 1.0 + alpha * a_lin;
                    let b1 = -2.0 * cos_w0;
                    let b2 = 1.0 - alpha * a_lin;
                    let a0 = 1.0 + alpha / a_lin;
                    let a1 = -2.0 * cos_w0;
                    let a2 = 1.0 - alpha / a_lin;
                    (b0, b1, b2, a0, a1, a2)
                }
                EqBandType::LowShelf => {
                    let s = 2.0_f32.sqrt() * alpha / a_lin; // shelf-slope term
                    let b0 = a_lin * ((a_lin + 1.0) - (a_lin - 1.0) * cos_w0 + s);
                    let b1 = 2.0 * a_lin * ((a_lin - 1.0) - (a_lin + 1.0) * cos_w0);
                    let b2 = a_lin * ((a_lin + 1.0) - (a_lin - 1.0) * cos_w0 - s);
                    let a0 = (a_lin + 1.0) + (a_lin - 1.0) * cos_w0 + s;
                    let a1 = -2.0 * ((a_lin - 1.0) + (a_lin + 1.0) * cos_w0);
                    let a2 = (a_lin + 1.0) + (a_lin - 1.0) * cos_w0 - s;
                    (b0, b1, b2, a0, a1, a2)
                }
                EqBandType::HighShelf => {
                    let s = 2.0_f32.sqrt() * alpha / a_lin;
                    let b0 = a_lin * ((a_lin + 1.0) + (a_lin - 1.0) * cos_w0 + s);
                    let b1 = -2.0 * a_lin * ((a_lin - 1.0) + (a_lin + 1.0) * cos_w0);
                    let b2 = a_lin * ((a_lin + 1.0) + (a_lin - 1.0) * cos_w0 - s);
                    let a0 = (a_lin + 1.0) - (a_lin - 1.0) * cos_w0 + s;
                    let a1 = 2.0 * ((a_lin - 1.0) - (a_lin + 1.0) * cos_w0);
                    let a2 = (a_lin + 1.0) - (a_lin - 1.0) * cos_w0 - s;
                    (b0, b1, b2, a0, a1, a2)
                }
                EqBandType::LowPass => {
                    let b0 = (1.0 - cos_w0) / 2.0;
                    let b1 = 1.0 - cos_w0;
                    let b2 = (1.0 - cos_w0) / 2.0;
                    let a0 = 1.0 + alpha;
                    let a1 = -2.0 * cos_w0;
                    let a2 = 1.0 - alpha;
                    (b0, b1, b2, a0, a1, a2)
                }
                EqBandType::HighPass => {
                    let b0 = (1.0 + cos_w0) / 2.0;
                    let b1 = -(1.0 + cos_w0);
                    let b2 = (1.0 + cos_w0) / 2.0;
                    let a0 = 1.0 + alpha;
                    let a1 = -2.0 * cos_w0;
                    let a2 = 1.0 - alpha;
                    (b0, b1, b2, a0, a1, a2)
                }
                EqBandType::Notch => {
                    let b0 = 1.0;
                    let b1 = -2.0 * cos_w0;
                    let b2 = 1.0;
                    let a0 = 1.0 + alpha;
                    let a1 = -2.0 * cos_w0;
                    let a2 = 1.0 - alpha;
                    (b0, b1, b2, a0, a1, a2)
                }
                EqBandType::BandPass => {
                    // Constant skirt gain, peak gain = Q
                    let b0 = sin_w0 / 2.0;
                    let b1 = 0.0;
                    let b2 = -(sin_w0 / 2.0);
                    let a0 = 1.0 + alpha;
                    let a1 = -2.0 * cos_w0;
                    let a2 = 1.0 - alpha;
                    (b0, b1, b2, a0, a1, a2)
                }
                EqBandType::AllPass => {
                    let b0 = 1.0 - alpha;
                    let b1 = -2.0 * cos_w0;
                    let b2 = 1.0 + alpha;
                    let a0 = 1.0 + alpha;
                    let a1 = -2.0 * cos_w0;
                    let a2 = 1.0 - alpha;
                    (b0, b1, b2, a0, a1, a2)
                }
            };

            // Normalise by a0
            let inv_a0 = if a0.abs() > 1e-10 { 1.0 / a0 } else { 1.0 };
            self.b0 = b0 * inv_a0;
            self.b1 = b1 * inv_a0;
            self.b2 = b2 * inv_a0;
            self.a1 = a1 * inv_a0;
            self.a2 = a2 * inv_a0;
        }

        /// Process a single sample through the biquad (Direct Form II).
        pub fn process_sample(&mut self, x: f32) -> f32 {
            if !self.enabled {
                return x;
            }
            let y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2
                - self.a1 * self.y1
                - self.a2 * self.y2;
            self.x2 = self.x1;
            self.x1 = x;
            self.y2 = self.y1;
            self.y1 = y;
            y
        }

        /// Reset filter state without changing coefficients.
        pub fn reset(&mut self) {
            self.x1 = 0.0;
            self.x2 = 0.0;
            self.y1 = 0.0;
            self.y2 = 0.0;
        }
    }

    // ────────────────────────────────────────────── FullChannelStrip ───

    /// Full DAW-style channel strip: trim → polarity → EQ → gate →
    /// compressor → fader → output (with optional stereo panning).
    ///
    /// Use `process_mono` for mono processing and `process_stereo` to get
    /// a constant-power stereo pair.
    #[derive(Debug, Clone)]
    pub struct ChannelStrip {
        /// Strip identifier.
        pub id: String,
        /// Input trim in dB.
        pub gain_db: f32,
        /// Invert signal polarity.
        pub polarity_flip: bool,
        /// Up to 8 parametric EQ bands.
        pub eq_bands: Vec<ParametricEqBand>,
        /// Gate open threshold in dB.
        pub gate_threshold_db: f32,
        /// Gate enabled flag.
        pub gate_enabled: bool,
        /// Compressor threshold in dB.
        pub compressor_threshold_db: f32,
        /// Compressor ratio (e.g., 4.0 = 4:1).
        pub compressor_ratio: f32,
        /// Compressor attack in ms.
        pub compressor_attack_ms: f32,
        /// Compressor release in ms.
        pub compressor_release_ms: f32,
        /// Compressor enabled flag.
        pub compressor_enabled: bool,
        /// Pan position: −1.0 (full left) … +1.0 (full right).
        pub pan: f32,
        /// Channel fader in dB.
        pub fader_db: f32,
        /// Mute flag.
        pub muted: bool,
        sample_rate: u32,
        /// Compressor envelope follower (linear).
        comp_env: f32,
        /// Gate envelope follower (linear).
        gate_env: f32,
    }

    impl ChannelStrip {
        /// Create a new channel strip with neutral (bypass) defaults.
        #[must_use]
        pub fn new(id: &str, sample_rate: u32) -> Self {
            Self {
                id: id.to_string(),
                gain_db: 0.0,
                polarity_flip: false,
                eq_bands: Vec::new(),
                gate_threshold_db: -60.0,
                gate_enabled: false,
                compressor_threshold_db: -20.0,
                compressor_ratio: 4.0,
                compressor_attack_ms: 10.0,
                compressor_release_ms: 100.0,
                compressor_enabled: false,
                pan: 0.0,
                fader_db: 0.0,
                muted: false,
                sample_rate,
                comp_env: 0.0,
                gate_env: 0.0,
            }
        }

        /// Add an EQ band (up to 8 are recommended; no hard limit enforced).
        pub fn add_eq_band(&mut self, band: ParametricEqBand) {
            self.eq_bands.push(band);
        }

        /// Process a block of mono samples through the full chain.
        ///
        /// Chain: input trim → polarity → EQ → gate → compressor → fader → output.
        #[must_use]
        pub fn process_mono(&mut self, samples: &[f32]) -> Vec<f32> {
            if self.muted {
                return vec![0.0_f32; samples.len()];
            }

            #[allow(clippy::cast_precision_loss)]
            let sr = self.sample_rate as f32;

            // Pre-compute time constants for envelope followers
            let attack_coeff_comp = Self::time_coeff(self.compressor_attack_ms, sr);
            let release_coeff_comp = Self::time_coeff(self.compressor_release_ms, sr);
            // Gate uses fixed 1 ms attack / 50 ms release
            let attack_coeff_gate = Self::time_coeff(1.0, sr);
            let release_coeff_gate = Self::time_coeff(50.0, sr);

            let trim_gain = 10.0_f32.powf(self.gain_db / 20.0);
            let fader_gain = 10.0_f32.powf(self.fader_db / 20.0);
            let polarity = if self.polarity_flip { -1.0_f32 } else { 1.0 };

            let mut out = Vec::with_capacity(samples.len());

            for &x in samples {
                // 1. Input trim
                let mut s = x * trim_gain;
                // 2. Polarity
                s *= polarity;
                // 3. EQ
                for band in &mut self.eq_bands {
                    s = band.process_sample(s);
                }
                // 4. Gate
                if self.gate_enabled {
                    let level = s.abs();
                    // Envelope follower on level
                    let coeff = if level > self.gate_env {
                        attack_coeff_gate
                    } else {
                        release_coeff_gate
                    };
                    self.gate_env = coeff * self.gate_env + (1.0 - coeff) * level;
                    let env_db = if self.gate_env > 1e-10 {
                        20.0 * self.gate_env.log10()
                    } else {
                        -120.0
                    };
                    if env_db < self.gate_threshold_db {
                        s *= 10.0_f32.powf(-80.0 / 20.0); // −80 dB
                    }
                }
                // 5. Compressor
                if self.compressor_enabled {
                    let level = s.abs();
                    let level_db = if level > 1e-10 {
                        20.0 * level.log10()
                    } else {
                        -120.0
                    };
                    // Gain reduction
                    let gr_db = if level_db > self.compressor_threshold_db {
                        let over = level_db - self.compressor_threshold_db;
                        over * (1.0 - 1.0 / self.compressor_ratio.max(1.0))
                    } else {
                        0.0
                    };
                    // Smooth with envelope follower
                    let coeff = if gr_db > self.comp_env {
                        attack_coeff_comp
                    } else {
                        release_coeff_comp
                    };
                    self.comp_env = coeff * self.comp_env + (1.0 - coeff) * gr_db;
                    let gain_linear = 10.0_f32.powf(-self.comp_env / 20.0);
                    s *= gain_linear;
                }
                // 6. Fader
                s *= fader_gain;
                out.push(s);
            }

            out
        }

        /// Process stereo pair through the mono chain then apply constant-power pan.
        ///
        /// Pan angles: `L = cos((pan+1)·π/4)`,  `R = sin((pan+1)·π/4)`.
        /// The two input channels are summed to mono before processing.
        #[must_use]
        pub fn process_stereo(&mut self, l: &[f32], r: &[f32]) -> (Vec<f32>, Vec<f32>) {
            let len = l.len().min(r.len());
            let mono: Vec<f32> = l[..len]
                .iter()
                .zip(r[..len].iter())
                .map(|(&a, &b)| (a + b) * 0.5)
                .collect();

            let processed = self.process_mono(&mono);

            let pan_norm = (self.pan + 1.0) * PI / 4.0;
            let left_gain = pan_norm.cos();
            let right_gain = pan_norm.sin();

            let left: Vec<f32> = processed.iter().map(|&s| s * left_gain).collect();
            let right: Vec<f32> = processed.iter().map(|&s| s * right_gain).collect();
            (left, right)
        }

        /// One-pole smoothing coefficient for a given time constant.
        fn time_coeff(ms: f32, sample_rate: f32) -> f32 {
            let tau = ms * 0.001 * sample_rate;
            if tau < 1.0 {
                0.0
            } else {
                (-1.0_f32 / tau).exp()
            }
        }
    }

    // ──────────────────────────────────────────────────────────── tests ───

    #[cfg(test)]
    mod tests {
        use super::*;

        fn sine_block(freq_hz: f32, sr: u32, n: usize) -> Vec<f32> {
            #[allow(clippy::cast_precision_loss)]
            let fs = sr as f32;
            (0..n)
                .map(|i| {
                    #[allow(clippy::cast_precision_loss)]
                    (2.0 * std::f32::consts::PI * freq_hz * i as f32 / fs).sin()
                })
                .collect()
        }

        #[test]
        fn test_channel_strip_new_defaults() {
            let cs = ChannelStrip::new("vox", 48_000);
            assert_eq!(cs.id, "vox");
            assert!((cs.gain_db).abs() < f32::EPSILON);
            assert!((cs.fader_db).abs() < f32::EPSILON);
            assert!(!cs.muted);
            assert!(!cs.compressor_enabled);
            assert!(!cs.gate_enabled);
        }

        #[test]
        fn test_channel_strip_mute() {
            let mut cs = ChannelStrip::new("test", 48_000);
            cs.muted = true;
            let out = cs.process_mono(&[0.5_f32; 16]);
            assert!(out.iter().all(|&s| s.abs() < f32::EPSILON));
        }

        #[test]
        fn test_channel_strip_unity_passthrough() {
            let mut cs = ChannelStrip::new("test", 48_000);
            let input = vec![0.3_f32; 8];
            let out = cs.process_mono(&input);
            for (&i, o) in input.iter().zip(out.iter()) {
                assert!((i - o).abs() < 1e-5, "expected {i} got {o}");
            }
        }

        #[test]
        fn test_channel_strip_polarity_flip() {
            let mut cs = ChannelStrip::new("test", 48_000);
            cs.polarity_flip = true;
            let out = cs.process_mono(&[1.0_f32; 4]);
            assert!(out.iter().all(|&s| (s - (-1.0)).abs() < 1e-5));
        }

        #[test]
        fn test_channel_strip_fader_6db() {
            let mut cs = ChannelStrip::new("test", 48_000);
            cs.fader_db = 6.0206; // ≈ 2×
            let out = cs.process_mono(&[0.5_f32; 4]);
            for &s in &out {
                assert!((s - 1.0).abs() < 0.01, "expected ~1.0, got {s}");
            }
        }

        #[test]
        fn test_channel_strip_trim_minus6db() {
            let mut cs = ChannelStrip::new("test", 48_000);
            cs.gain_db = -6.0206;
            let out = cs.process_mono(&[1.0_f32; 4]);
            for &s in &out {
                assert!((s - 0.5).abs() < 0.01, "expected ~0.5, got {s}");
            }
        }

        #[test]
        fn test_eq_band_peaking() {
            let band = ParametricEqBand::new(1_000.0, 6.0, 1.0, EqBandType::Peaking, 48_000);
            assert!(band.enabled);
            // Rough sanity: at DC (0 Hz) a peaking filter with positive gain
            // should return a value close to input (DC not boosted by band-pass).
        }

        #[test]
        fn test_eq_band_lowpass_attenuates_highs() {
            let mut band = ParametricEqBand::new(200.0, 0.0, 0.707, EqBandType::LowPass, 48_000);
            // Process a 10 kHz sine — should be heavily attenuated
            let high = sine_block(10_000.0, 48_000, 512);
            // Warm up
            let mut out = 0.0_f32;
            for &s in &high {
                out = band.process_sample(s);
            }
            assert!(
                out.abs() < 0.1,
                "LPF @200 Hz should attenuate 10 kHz; got {out}"
            );
        }

        #[test]
        fn test_eq_band_highpass_attenuates_lows() {
            let mut band = ParametricEqBand::new(8_000.0, 0.0, 0.707, EqBandType::HighPass, 48_000);
            let low = sine_block(100.0, 48_000, 512);
            let mut out = 0.0_f32;
            for &s in &low {
                out = band.process_sample(s);
            }
            assert!(
                out.abs() < 0.1,
                "HPF @8 kHz should attenuate 100 Hz; got {out}"
            );
        }

        #[test]
        fn test_eq_band_reset() {
            let mut band = ParametricEqBand::new(1_000.0, 0.0, 1.0, EqBandType::Notch, 48_000);
            for _ in 0..100 {
                band.process_sample(0.5);
            }
            band.reset();
            assert_eq!(band.x1, 0.0);
            assert_eq!(band.y1, 0.0);
        }

        #[test]
        fn test_channel_strip_eq_in_chain() {
            let mut cs = ChannelStrip::new("test", 48_000);
            // Add a LPF at 200 Hz — high-frequency content should be removed
            cs.add_eq_band(ParametricEqBand::new(
                200.0,
                0.0,
                0.707,
                EqBandType::LowPass,
                48_000,
            ));
            let high = sine_block(10_000.0, 48_000, 512);
            let out = cs.process_mono(&high);
            let energy: f32 = out.iter().map(|&s| s * s).sum::<f32>() / out.len() as f32;
            assert!(energy < 0.01, "expected low energy after LPF, got {energy}");
        }

        #[test]
        fn test_channel_strip_stereo_pan_center() {
            let mut cs = ChannelStrip::new("test", 48_000);
            cs.pan = 0.0;
            let l = vec![1.0_f32; 8];
            let r = vec![1.0_f32; 8];
            let (ol, or_) = cs.process_stereo(&l, &r);
            // At center pan L ≈ R ≈ cos(π/4) = 0.7071
            for (&lv, &rv) in ol.iter().zip(or_.iter()) {
                assert!(
                    (lv - rv).abs() < 1e-5,
                    "L={lv} R={rv} should be equal at center pan"
                );
            }
        }

        #[test]
        fn test_channel_strip_stereo_pan_right() {
            let mut cs = ChannelStrip::new("test", 48_000);
            cs.pan = 1.0; // full right
            let l = vec![1.0_f32; 4];
            let r = vec![1.0_f32; 4];
            let (ol, or_) = cs.process_stereo(&l, &r);
            for (&lv, &rv) in ol.iter().zip(or_.iter()) {
                assert!(rv > lv, "R={rv} should exceed L={lv} at pan=1.0");
            }
        }

        #[test]
        fn test_channel_strip_compressor_reduces_level() {
            let mut cs = ChannelStrip::new("test", 48_000);
            cs.compressor_enabled = true;
            cs.compressor_threshold_db = -20.0;
            cs.compressor_ratio = 4.0;
            cs.compressor_attack_ms = 1.0;
            cs.compressor_release_ms = 50.0;
            // Input well above threshold
            let loud = vec![1.0_f32; 2048];
            let out = cs.process_mono(&loud);
            // The last sample should be reduced
            let last = *out.last().unwrap_or(&0.0);
            assert!(last < 0.9, "compressor should reduce level; got {last}");
        }

        #[test]
        fn test_channel_strip_gate_silences_below_threshold() {
            let mut cs = ChannelStrip::new("test", 48_000);
            cs.gate_enabled = true;
            cs.gate_threshold_db = -10.0; // gate closes for signals below -10 dB
                                          // Very quiet signal (-40 dB ≈ 0.01)
            let quiet = vec![0.01_f32; 512];
            let out = cs.process_mono(&quiet);
            // After settling, output should be very small
            let last = *out.last().unwrap_or(&0.5);
            assert!(
                last.abs() < 0.01,
                "gate should suppress quiet signal; got {last}"
            );
        }
    }
}
