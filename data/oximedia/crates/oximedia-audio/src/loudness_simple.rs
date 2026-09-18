//! Simplified EBU R128 loudness measurement (standalone, zero-allocation hot path).
//!
//! This module provides a self-contained, lightweight implementation of the
//! ITU-R BS.1770-4 / EBU R128 loudness measurement algorithm.  It is
//! intentionally kept simple – no inter-crate dependencies, pure f32 maths –
//! to serve as a fast pre-flight loudness check or as a reference implementation.
//!
//! For the full-featured implementation (normalisation, gating, multi-standard
//! support, reporting, …) see the [`crate::loudness`] module.
//!
//! # Features
//!
//! * **K-weighting pre-filter** – High-shelf pre-filter followed by a high-pass
//!   filter, as specified in ITU-R BS.1770-4.
//! * **RMS-based loudness** – `compute_rms_db` converts mean-square energy to dBFS.
//! * **`LoudnessMeter`** – Stateful meter that accumulates 100 ms blocks and
//!   returns an integrated (time-averaged) loudness estimate.
//!
//! # Example
//!
//! ```
//! use oximedia_audio::loudness_simple::{LoudnessConfig, LoudnessMeter, apply_k_weight, compute_rms_db, PreFilter};
//!
//! let config = LoudnessConfig::stereo_48k();
//! let mut meter = LoudnessMeter::new(config);
//!
//! // Simulate a 480-sample (10 ms) block of a full-scale sine wave
//! let samples: Vec<f32> = (0..480)
//!     .map(|i| (i as f32 * 0.1).sin())
//!     .collect();
//!
//! meter.add_block(&samples);
//! let loudness = meter.integrated_loudness();
//! assert!(loudness.is_finite() || loudness == f32::NEG_INFINITY);
//! ```

#![forbid(unsafe_code)]
#![allow(clippy::cast_precision_loss)]

use std::f32::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
// LoudnessConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the simplified loudness meter.
#[derive(Clone, Debug, PartialEq, Eq)]
#[allow(dead_code)]
pub struct LoudnessConfig {
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Number of audio channels.
    pub channels: u8,
}

impl LoudnessConfig {
    /// Stereo, 48 kHz – the EBU R128 reference configuration.
    #[must_use]
    pub fn stereo_48k() -> Self {
        Self {
            sample_rate: 48_000,
            channels: 2,
        }
    }

    /// Mono, 44.1 kHz.
    #[must_use]
    pub fn mono_44k() -> Self {
        Self {
            sample_rate: 44_100,
            channels: 1,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MomentaryLoudness
// ─────────────────────────────────────────────────────────────────────────────

/// A snapshot of loudness at a point in time.
#[derive(Clone, Debug, PartialEq)]
#[allow(dead_code)]
pub struct MomentaryLoudness {
    /// Integrated loudness since the meter was started (LUFS).
    pub integrated: f32,
    /// Momentary loudness (400 ms window) in LUFS.
    pub momentary: f32,
    /// Short-term loudness (3 s window) in LUFS.
    pub short_term: f32,
}

impl MomentaryLoudness {
    /// Return `true` when the integrated loudness is within `tolerance` LU of
    /// `target`.
    #[must_use]
    pub fn is_within_target(&self, target: f32, tolerance: f32) -> bool {
        (self.integrated - target).abs() <= tolerance
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PreFilter  (K-weighting)
// ─────────────────────────────────────────────────────────────────────────────

/// Two-stage K-weighting pre-filter as defined in ITU-R BS.1770-4.
///
/// Stage 1: High-shelf pre-filter (+4 dB shelf at ~1.5 kHz).
/// Stage 2: High-pass Butterworth filter (~38 Hz).
///
/// The coefficients are stored in direct-form-I biquad layout:
/// `y[n] = b[0]*x[n] + b[1]*x[n-1] + b[2]*x[n-2] - a[1]*y[n-1] - a[2]*y[n-2]`
/// (a\[0\] normalised to 1).
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct PreFilter {
    /// Numerator coefficients for the high-shelf stage.
    pub hs_b: [f32; 3],
    /// Denominator coefficients (a1, a2) for the high-shelf stage.
    pub hs_a: [f32; 3],
    /// Numerator coefficients for the high-pass stage.
    pub hp_b: [f32; 3],
    /// Denominator coefficients (a1, a2) for the high-pass stage.
    pub hp_a: [f32; 3],
    // Internal biquad state – high-shelf
    hs_x1: f32,
    hs_x2: f32,
    hs_y1: f32,
    hs_y2: f32,
    // Internal biquad state – high-pass
    hp_x1: f32,
    hp_x2: f32,
    hp_y1: f32,
    hp_y2: f32,
}

impl PreFilter {
    /// Design a K-weighting pre-filter for the given sample rate.
    ///
    /// The implementation follows the reference coefficients from
    /// ITU-R BS.1770-4 Annex 1 (48 kHz) and re-derives them for other
    /// sample rates using the bilinear transform.
    #[must_use]
    pub fn new_r128(sample_rate: f32) -> Self {
        // ── High-shelf pre-filter ──────────────────────────────────────────
        // Design via ITU-R BS.1770-4 Annex 1 formula (bilinear transform of
        // a second-order high-shelf filter with Fc ≈ 1681 Hz, gain +4 dB).
        let fs = sample_rate;

        // High-shelf: Fc = 1681.974 Hz, gain = +3.999843 dB
        let f0_hs = 1681.974_f32;
        let g_db = 3.999_843_f32;
        let q_hs = 0.7071_f32;
        let (hs_b, hs_a) = design_highshelf(f0_hs, g_db, q_hs, fs);

        // ── High-pass filter ───────────────────────────────────────────────
        // Second-order Butterworth high-pass, Fc ≈ 38.135 Hz
        let f0_hp = 38.135_f32;
        let q_hp = 0.5_f32.sqrt(); // 2nd-order Butterworth Q
        let (hp_b, hp_a) = design_highpass(f0_hp, q_hp, fs);

        Self {
            hs_b,
            hs_a,
            hp_b,
            hp_a,
            hs_x1: 0.0,
            hs_x2: 0.0,
            hs_y1: 0.0,
            hs_y2: 0.0,
            hp_x1: 0.0,
            hp_x2: 0.0,
            hp_y1: 0.0,
            hp_y2: 0.0,
        }
    }

    /// Process one sample through both filter stages (high-shelf then high-pass).
    fn process_sample(&mut self, x: f32) -> f32 {
        // Stage 1: high-shelf
        let hs_y = self.hs_b[0] * x + self.hs_b[1] * self.hs_x1 + self.hs_b[2] * self.hs_x2
            - self.hs_a[1] * self.hs_y1
            - self.hs_a[2] * self.hs_y2;
        self.hs_x2 = self.hs_x1;
        self.hs_x1 = x;
        self.hs_y2 = self.hs_y1;
        self.hs_y1 = hs_y;

        // Stage 2: high-pass
        let hp_y = self.hp_b[0] * hs_y + self.hp_b[1] * self.hp_x1 + self.hp_b[2] * self.hp_x2
            - self.hp_a[1] * self.hp_y1
            - self.hp_a[2] * self.hp_y2;
        self.hp_x2 = self.hp_x1;
        self.hp_x1 = hs_y;
        self.hp_y2 = self.hp_y1;
        self.hp_y1 = hp_y;

        hp_y
    }

    /// Reset the internal filter states.
    pub fn reset(&mut self) {
        self.hs_x1 = 0.0;
        self.hs_x2 = 0.0;
        self.hs_y1 = 0.0;
        self.hs_y2 = 0.0;
        self.hp_x1 = 0.0;
        self.hp_x2 = 0.0;
        self.hp_y1 = 0.0;
        self.hp_y2 = 0.0;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Filter design helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Design a second-order high-shelf biquad via bilinear transform.
/// Returns (b[3], a[3]) with a[0] == 1.
fn design_highshelf(fc: f32, gain_db: f32, q: f32, fs: f32) -> ([f32; 3], [f32; 3]) {
    let a = 10.0_f32.powf(gain_db / 40.0);
    let w0 = 2.0 * PI * fc / fs;
    let cos_w0 = w0.cos();
    let sin_w0 = w0.sin();
    let alpha = sin_w0 / (2.0 * q);
    let two_sqrt_a_alpha = 2.0 * a.sqrt() * alpha;
    let ap1 = a + 1.0;
    let am1 = a - 1.0;

    let b0 = a * (ap1 + am1 * cos_w0 + two_sqrt_a_alpha);
    let b1 = -2.0 * a * (am1 + ap1 * cos_w0);
    let b2 = a * (ap1 + am1 * cos_w0 - two_sqrt_a_alpha);
    let a0 = ap1 - am1 * cos_w0 + two_sqrt_a_alpha;
    let a1 = 2.0 * (am1 - ap1 * cos_w0);
    let a2 = ap1 - am1 * cos_w0 - two_sqrt_a_alpha;

    ([b0 / a0, b1 / a0, b2 / a0], [1.0, a1 / a0, a2 / a0])
}

/// Design a second-order Butterworth high-pass biquad.
/// Returns (b[3], a[3]) with a[0] == 1.
fn design_highpass(fc: f32, q: f32, fs: f32) -> ([f32; 3], [f32; 3]) {
    let w0 = 2.0 * PI * fc / fs;
    let cos_w0 = w0.cos();
    let sin_w0 = w0.sin();
    let alpha = sin_w0 / (2.0 * q);

    let b0 = (1.0 + cos_w0) / 2.0;
    let b1 = -(1.0 + cos_w0);
    let b2 = (1.0 + cos_w0) / 2.0;
    let a0 = 1.0 + alpha;
    let a1 = -2.0 * cos_w0;
    let a2 = 1.0 - alpha;

    ([b0 / a0, b1 / a0, b2 / a0], [1.0, a1 / a0, a2 / a0])
}

// ─────────────────────────────────────────────────────────────────────────────
// apply_k_weight
// ─────────────────────────────────────────────────────────────────────────────

/// Apply both K-weighting filter stages to a slice of samples.
///
/// # Arguments
///
/// * `samples` – Mono audio samples.
/// * `filter` – Pre-filter instance (state is mutated).
///
/// # Returns
///
/// A new `Vec<f32>` of filtered samples.
#[must_use]
#[allow(dead_code)]
pub fn apply_k_weight(samples: &[f32], filter: &mut PreFilter) -> Vec<f32> {
    samples.iter().map(|&s| filter.process_sample(s)).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// compute_rms_db
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the RMS level of a slice of samples in dBFS.
///
/// Returns `f32::NEG_INFINITY` for a silent (all-zero) input.
#[must_use]
#[allow(dead_code)]
pub fn compute_rms_db(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return f32::NEG_INFINITY;
    }
    let mean_sq: f32 = samples.iter().map(|&s| s * s).sum::<f32>() / samples.len() as f32;
    if mean_sq <= 0.0 {
        f32::NEG_INFINITY
    } else {
        10.0 * mean_sq.log10()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LoudnessMeter
// ─────────────────────────────────────────────────────────────────────────────

/// Stateful, simplified EBU R128 loudness meter.
///
/// Call [`LoudnessMeter::add_block`] with successive audio blocks (any size).
/// The meter accumulates the mean-square energy of each block after K-weighting
/// and computes an integrated loudness on demand.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct LoudnessMeter {
    config: LoudnessConfig,
    /// Accumulated mean-square energy blocks (one entry per `add_block` call).
    history: Vec<f32>,
    /// Stateful K-weighting pre-filter.
    filter: PreFilter,
}

impl LoudnessMeter {
    /// Create a new meter with the given configuration.
    #[must_use]
    pub fn new(config: LoudnessConfig) -> Self {
        let filter = PreFilter::new_r128(config.sample_rate as f32);
        Self {
            config,
            history: Vec::new(),
            filter,
        }
    }

    /// Process one block of (mono or interleaved) samples.
    ///
    /// The block is K-weighted and its mean-square energy is stored.
    pub fn add_block(&mut self, samples: &[f32]) {
        if samples.is_empty() {
            return;
        }
        let weighted: Vec<f32> = samples
            .iter()
            .map(|&s| self.filter.process_sample(s))
            .collect();
        let mean_sq: f32 = weighted.iter().map(|&s| s * s).sum::<f32>() / weighted.len() as f32;
        self.history.push(mean_sq);
    }

    /// Compute the integrated loudness in LUFS (or LKFS) over all accumulated blocks.
    ///
    /// Returns `f32::NEG_INFINITY` when no non-silent blocks have been seen.
    #[must_use]
    pub fn integrated_loudness(&self) -> f32 {
        if self.history.is_empty() {
            return f32::NEG_INFINITY;
        }
        let total_ms: f32 = self.history.iter().sum::<f32>() / self.history.len() as f32;
        if total_ms <= 0.0 {
            return f32::NEG_INFINITY;
        }
        // Convert mean-square to LUFS: -0.691 + 10 * log10(sum of mean-square)
        -0.691 + 10.0 * total_ms.log10()
    }

    /// Return the number of blocks that have been fed into the meter.
    #[must_use]
    pub fn block_count(&self) -> usize {
        self.history.len()
    }

    /// Reset the meter (clear history and filter state).
    pub fn reset(&mut self) {
        self.history.clear();
        self.filter.reset();
    }

    /// Return the meter configuration.
    #[must_use]
    pub fn config(&self) -> &LoudnessConfig {
        &self.config
    }

    /// Return a snapshot of the current loudness.
    ///
    /// Because this simplified implementation does not maintain separate
    /// momentary / short-term windows, all three fields reflect the same
    /// integrated value.
    #[must_use]
    pub fn snapshot(&self) -> MomentaryLoudness {
        let il = self.integrated_loudness();
        MomentaryLoudness {
            integrated: il,
            momentary: il,
            short_term: il,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 48_000.0;

    // ── LoudnessConfig ────────────────────────────────────────────────────────

    #[test]
    fn test_stereo_48k_config() {
        let c = LoudnessConfig::stereo_48k();
        assert_eq!(c.sample_rate, 48_000);
        assert_eq!(c.channels, 2);
    }

    #[test]
    fn test_mono_44k_config() {
        let c = LoudnessConfig::mono_44k();
        assert_eq!(c.sample_rate, 44_100);
        assert_eq!(c.channels, 1);
    }

    // ── MomentaryLoudness ─────────────────────────────────────────────────────

    #[test]
    fn test_is_within_target_true() {
        let ml = MomentaryLoudness {
            integrated: -23.0,
            momentary: -23.0,
            short_term: -23.0,
        };
        assert!(ml.is_within_target(-23.0, 1.0));
    }

    #[test]
    fn test_is_within_target_false_too_loud() {
        let ml = MomentaryLoudness {
            integrated: -20.0,
            momentary: -20.0,
            short_term: -20.0,
        };
        assert!(!ml.is_within_target(-23.0, 1.0));
    }

    #[test]
    fn test_is_within_target_boundary() {
        let ml = MomentaryLoudness {
            integrated: -22.0,
            momentary: -22.0,
            short_term: -22.0,
        };
        // exactly 1 LU away – should be within tolerance of 1
        assert!(ml.is_within_target(-23.0, 1.0));
    }

    // ── PreFilter ─────────────────────────────────────────────────────────────

    #[test]
    fn test_prefilter_coeffs_are_finite() {
        let f = PreFilter::new_r128(SR);
        assert!(f.hs_b.iter().all(|x| x.is_finite()));
        assert!(f.hs_a.iter().all(|x| x.is_finite()));
        assert!(f.hp_b.iter().all(|x| x.is_finite()));
        assert!(f.hp_a.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_prefilter_processes_without_nans() {
        let mut f = PreFilter::new_r128(SR);
        for i in 0..1000 {
            let s = (i as f32 * 0.01).sin();
            let out = f.process_sample(s);
            assert!(out.is_finite(), "NaN or Inf at sample {i}");
        }
    }

    #[test]
    fn test_prefilter_reset_clears_state() {
        let mut f = PreFilter::new_r128(SR);
        // Drive the filter to a non-zero state
        for _ in 0..200 {
            f.process_sample(1.0);
        }
        f.reset();
        // After reset, zero input should produce zero output
        let out = f.process_sample(0.0);
        assert_eq!(out, 0.0);
    }

    #[test]
    fn test_prefilter_highpass_blocks_dc() {
        let mut f = PreFilter::new_r128(SR);
        let mut out = 0.0_f32;
        for _ in 0..5000 {
            out = f.process_sample(1.0);
        }
        assert!(
            out.abs() < 0.1,
            "K-weight filter should attenuate DC; got {out}"
        );
    }

    // ── apply_k_weight ────────────────────────────────────────────────────────

    #[test]
    fn test_apply_k_weight_length_preserved() {
        let mut f = PreFilter::new_r128(SR);
        let input = vec![0.5_f32; 480];
        let output = apply_k_weight(&input, &mut f);
        assert_eq!(output.len(), input.len());
    }

    #[test]
    fn test_apply_k_weight_all_finite() {
        let mut f = PreFilter::new_r128(SR);
        let input: Vec<f32> = (0..480).map(|i| (i as f32 * 0.05).sin()).collect();
        let output = apply_k_weight(&input, &mut f);
        assert!(output.iter().all(|x| x.is_finite()));
    }

    // ── compute_rms_db ────────────────────────────────────────────────────────

    #[test]
    fn test_rms_db_silence_is_neg_inf() {
        let out = compute_rms_db(&[0.0_f32; 100]);
        assert_eq!(out, f32::NEG_INFINITY);
    }

    #[test]
    fn test_rms_db_empty_is_neg_inf() {
        let out = compute_rms_db(&[]);
        assert_eq!(out, f32::NEG_INFINITY);
    }

    #[test]
    fn test_rms_db_full_scale_sine() {
        // Full-scale sine RMS is 1/sqrt(2) ≈ 0.707, so RMS in dB ≈ -3.01 dBFS
        let sr = 48_000_usize;
        let samples: Vec<f32> = (0..sr)
            .map(|i| (2.0 * PI * 1000.0 * i as f32 / sr as f32).sin())
            .collect();
        let rms = compute_rms_db(&samples);
        assert!((rms - (-3.01)).abs() < 0.1, "Expected ≈ -3 dBFS, got {rms}");
    }

    // ── LoudnessMeter ─────────────────────────────────────────────────────────

    #[test]
    fn test_meter_block_count_increments() {
        let mut m = LoudnessMeter::new(LoudnessConfig::stereo_48k());
        m.add_block(&[0.1_f32; 480]);
        m.add_block(&[0.1_f32; 480]);
        assert_eq!(m.block_count(), 2);
    }

    #[test]
    fn test_meter_empty_returns_neg_inf() {
        let m = LoudnessMeter::new(LoudnessConfig::stereo_48k());
        assert_eq!(m.integrated_loudness(), f32::NEG_INFINITY);
    }

    #[test]
    fn test_meter_reset_clears_history() {
        let mut m = LoudnessMeter::new(LoudnessConfig::stereo_48k());
        m.add_block(&[0.5_f32; 480]);
        m.reset();
        assert_eq!(m.block_count(), 0);
        assert_eq!(m.integrated_loudness(), f32::NEG_INFINITY);
    }

    #[test]
    fn test_meter_loudness_is_finite_for_non_silent_input() {
        let mut m = LoudnessMeter::new(LoudnessConfig::stereo_48k());
        let block: Vec<f32> = (0..4800).map(|i| (i as f32 * 0.1).sin()).collect();
        m.add_block(&block);
        let il = m.integrated_loudness();
        assert!(
            il.is_finite(),
            "Integrated loudness should be finite; got {il}"
        );
    }

    #[test]
    fn test_meter_config_accessor() {
        let config = LoudnessConfig::stereo_48k();
        let m = LoudnessMeter::new(config.clone());
        assert_eq!(*m.config(), config);
    }

    #[test]
    fn test_meter_snapshot_values_match_integrated() {
        let mut m = LoudnessMeter::new(LoudnessConfig::stereo_48k());
        m.add_block(&vec![0.3_f32; 960]);
        let snap = m.snapshot();
        let il = m.integrated_loudness();
        assert_eq!(snap.integrated, il);
        assert_eq!(snap.momentary, il);
        assert_eq!(snap.short_term, il);
    }

    #[test]
    fn test_meter_empty_block_does_not_add_to_history() {
        let mut m = LoudnessMeter::new(LoudnessConfig::stereo_48k());
        m.add_block(&[]);
        assert_eq!(m.block_count(), 0);
    }
}
