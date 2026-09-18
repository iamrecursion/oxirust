//! EBU R128 / ITU-R BS.1770-4 loudness normalisation for use as a `RestoreChain` step.
//!
//! This module implements an integrated loudness meter and gain normaliser that
//! conforms to EBU R128 / ITU-R BS.1770-4:
//!
//! 1. **Pre-filter** (K-weighting stage 1): high-shelf biquad that models the
//!    acoustic transfer function of the human head.
//! 2. **High-pass filter** (K-weighting stage 2): second-order Butterworth HP at 38 Hz.
//! 3. **Mean square** power measurement in 400 ms blocks with 75% overlap (100 ms hop).
//! 4. **Gating**: blocks with power < –70 LKFS absolute gate are excluded; of the
//!    remaining blocks, blocks < –10 dB relative to the un-gated average are also
//!    excluded (relative gate).
//! 5. The integrated loudness `L_I` is computed from the gated block set.
//! 6. A linear gain `G = 10^((target_lkfs − L_I) / 20)` is applied.
//!
//! # References
//! * EBU Tech 3341 (EBU R128)
//! * ITU-R BS.1770-4
//!
//! # Example
//! ```
//! use oximedia_restore::loudness_normalization::{LoudnessNormalizer, LoudnessNormalizerConfig};
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let config = LoudnessNormalizerConfig::default(); // target = −23 LKFS
//! let mut normalizer = LoudnessNormalizer::new(config, 48000);
//! let input = vec![0.5_f32; 48000];
//! let output = normalizer.process(&input)?;
//! assert_eq!(output.len(), input.len());
//! # Ok(())
//! # }
//! ```

use crate::error::{RestoreError, RestoreResult};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the EBU R128 loudness normaliser.
#[derive(Debug, Clone)]
pub struct LoudnessNormalizerConfig {
    /// Target integrated loudness in LKFS (default: −23.0 LKFS per EBU R128).
    pub target_lkfs: f32,
    /// Maximum gain to apply in dB (prevents extreme amplification on very quiet content).
    pub max_gain_db: f32,
    /// Maximum attenuation to apply in dB (positive value; prevents over-reduction).
    pub max_attenuation_db: f32,
    /// Block duration in seconds (EBU R128: 0.4 s).
    pub block_duration_s: f32,
    /// Block hop in seconds (EBU R128: 0.1 s).
    pub block_hop_s: f32,
}

impl Default for LoudnessNormalizerConfig {
    fn default() -> Self {
        Self {
            target_lkfs: -23.0,
            max_gain_db: 30.0,
            max_attenuation_db: 30.0,
            block_duration_s: 0.4,
            block_hop_s: 0.1,
        }
    }
}

// ---------------------------------------------------------------------------
// Biquad state (direct form II transposed)
// ---------------------------------------------------------------------------

/// Second-order IIR filter using direct form II transposed.
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
    fn new(b0: f32, b1: f32, b2: f32, a1: f32, a2: f32) -> Self {
        Self {
            b0,
            b1,
            b2,
            a1,
            a2,
            s1: 0.0,
            s2: 0.0,
        }
    }

    #[inline]
    fn process_sample(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.s1;
        self.s1 = self.b1 * x - self.a1 * y + self.s2;
        self.s2 = self.b2 * x - self.a2 * y;
        y
    }

    fn process(&mut self, samples: &[f32]) -> Vec<f32> {
        samples.iter().map(|&s| self.process_sample(s)).collect()
    }

    #[allow(dead_code)]
    fn reset(&mut self) {
        self.s1 = 0.0;
        self.s2 = 0.0;
    }
}

// ---------------------------------------------------------------------------
// K-weighting filter design (EBU R128 Annex 1)
// ---------------------------------------------------------------------------

/// Compute K-weighting pre-filter coefficients for a given sample rate.
///
/// Stage 1 (high-shelf): compensates head transfer function.
/// Coefficients from ITU-R BS.1770-4, Table 1.
#[allow(clippy::cast_precision_loss)]
fn kw_prefilter(sample_rate: u32) -> Biquad {
    // Coefficients are specified for 48 kHz; warp for other sample rates using
    // the bilinear transform relationship.
    let fs = sample_rate as f64;

    // Reference: EBU Tech 3341, Table 1 (48 kHz values pre-warped for any rate).
    // These values come from the standard at 48 kHz and are re-derived here:
    let db = 3.999_843_853_973_347_f64;
    let f0 = 1_681.974_450_955_533_f64;
    let q = 0.717_897_987_600_710_f64;

    let k = (std::f64::consts::PI * f0 / fs).tan();
    let vb = 10.0_f64.powf(db / 20.0);
    let norm = 1.0 + k / q + k * k;
    let b0 = (vb + vb / q * k + k * k) / norm;
    let b1 = 2.0 * (k * k - vb) / norm;
    let b2 = (vb - vb / q * k + k * k) / norm;
    let a1 = 2.0 * (k * k - 1.0) / norm;
    let a2 = (1.0 - k / q + k * k) / norm;

    Biquad::new(b0 as f32, b1 as f32, b2 as f32, a1 as f32, a2 as f32)
}

/// Compute K-weighting high-pass filter coefficients.
///
/// Stage 2: second-order Butterworth HP at 38.13547 Hz.
#[allow(clippy::cast_precision_loss)]
fn kw_highpass(sample_rate: u32) -> Biquad {
    let fs = sample_rate as f64;
    let f0 = 38.135_470_876_008_7_f64;
    let k = (std::f64::consts::PI * f0 / fs).tan();
    let norm = k * k + std::f64::consts::SQRT_2 * k + 1.0;
    let b0 = 1.0 / norm;
    let b1 = -2.0 / norm;
    let b2 = 1.0 / norm;
    let a1 = 2.0 * (k * k - 1.0) / norm;
    let a2 = (k * k - std::f64::consts::SQRT_2 * k + 1.0) / norm;

    Biquad::new(b0 as f32, b1 as f32, b2 as f32, a1 as f32, a2 as f32)
}

// ---------------------------------------------------------------------------
// Loudness measurement
// ---------------------------------------------------------------------------

/// Measure integrated loudness from mono K-weighted samples using EBU R128 gating.
///
/// Returns the integrated loudness in LKFS (= LUFS for mono).
/// Returns `None` when there are insufficient gated blocks to form a valid measurement.
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn measure_integrated_loudness(
    kw_samples: &[f32],
    sample_rate: u32,
    block_duration_s: f32,
    block_hop_s: f32,
) -> Option<f32> {
    let block_len = (block_duration_s * sample_rate as f32).round() as usize;
    let hop_len = (block_hop_s * sample_rate as f32).round() as usize;

    if block_len == 0 || hop_len == 0 || kw_samples.len() < block_len {
        return None;
    }

    // Collect mean-square power for each block
    let mut block_powers: Vec<f32> = Vec::new();
    let mut pos = 0;
    while pos + block_len <= kw_samples.len() {
        let block = &kw_samples[pos..pos + block_len];
        let ms: f32 = block.iter().map(|&s| s * s).sum::<f32>() / block_len as f32;
        block_powers.push(ms);
        pos += hop_len;
    }

    if block_powers.is_empty() {
        return None;
    }

    // Absolute gate: –70 LKFS → power threshold = 10^((-70+0.691)/10)
    // (the +0.691 term converts LKFS→ power correctly)
    let abs_gate_power = 10.0_f32.powf((-70.0_f32 + 0.691) / 10.0);

    let above_abs: Vec<f32> = block_powers
        .iter()
        .copied()
        .filter(|&p| p > abs_gate_power)
        .collect();

    if above_abs.is_empty() {
        return None;
    }

    // Un-gated average (used for relative gate)
    let ungated_mean: f32 = above_abs.iter().sum::<f32>() / above_abs.len() as f32;

    // Relative gate: –10 dB relative to ungated mean
    let rel_gate_power = ungated_mean * 10.0_f32.powf(-10.0 / 10.0);

    let above_rel: Vec<f32> = above_abs
        .into_iter()
        .filter(|&p| p > rel_gate_power)
        .collect();

    if above_rel.is_empty() {
        return None;
    }

    let gated_mean: f32 = above_rel.iter().sum::<f32>() / above_rel.len() as f32;

    if gated_mean <= 0.0 {
        return None;
    }

    // Integrated loudness: L_I = −0.691 + 10·log10(gated_mean)
    let lkfs = -0.691 + 10.0 * gated_mean.log10();
    Some(lkfs)
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Result of a loudness normalisation pass.
#[derive(Debug, Clone)]
pub struct LoudnessNormalizationResult {
    /// Measured integrated loudness before normalisation (LKFS).
    /// `None` when content is too quiet / too short to measure.
    pub measured_lkfs: Option<f32>,
    /// Gain applied in dB (positive = amplification, negative = attenuation).
    pub applied_gain_db: f32,
    /// Linear gain multiplier actually applied.
    pub linear_gain: f32,
}

/// EBU R128 loudness normaliser.
///
/// Measures the integrated loudness of the input and applies a linear gain to
/// reach the configured target.  Can be used stand-alone or as a
/// [`RestorationStep::LoudnessNormalization`](crate::RestorationStep) inside a
/// [`RestoreChain`](crate::RestoreChain).
#[derive(Debug, Clone)]
pub struct LoudnessNormalizer {
    config: LoudnessNormalizerConfig,
    sample_rate: u32,
}

impl LoudnessNormalizer {
    /// Create a new loudness normaliser.
    ///
    /// # Arguments
    ///
    /// * `config` - Normaliser configuration
    /// * `sample_rate` - Sample rate of audio to process in Hz
    #[must_use]
    pub fn new(config: LoudnessNormalizerConfig, sample_rate: u32) -> Self {
        Self {
            config,
            sample_rate,
        }
    }

    /// Process mono audio samples: measure integrated loudness and apply gain.
    ///
    /// # Arguments
    ///
    /// * `samples` - Input mono samples
    ///
    /// # Returns
    ///
    /// Gain-adjusted samples and the measurement result.
    pub fn process_with_result(
        &self,
        samples: &[f32],
    ) -> RestoreResult<(Vec<f32>, LoudnessNormalizationResult)> {
        if samples.is_empty() {
            return Ok((
                Vec::new(),
                LoudnessNormalizationResult {
                    measured_lkfs: None,
                    applied_gain_db: 0.0,
                    linear_gain: 1.0,
                },
            ));
        }

        // Apply K-weighting
        let kw_samples = self.apply_kweighting(samples)?;

        // Measure integrated loudness
        let measured = measure_integrated_loudness(
            &kw_samples,
            self.sample_rate,
            self.config.block_duration_s,
            self.config.block_hop_s,
        );

        let (gain_db, linear_gain) = match measured {
            Some(lkfs) => {
                let raw_gain_db = self.config.target_lkfs - lkfs;
                let clamped_db =
                    raw_gain_db.clamp(-self.config.max_attenuation_db, self.config.max_gain_db);
                let lin = 10.0_f32.powf(clamped_db / 20.0);
                (clamped_db, lin)
            }
            None => (0.0, 1.0),
        };

        let output: Vec<f32> = samples.iter().map(|&s| s * linear_gain).collect();

        Ok((
            output,
            LoudnessNormalizationResult {
                measured_lkfs: measured,
                applied_gain_db: gain_db,
                linear_gain,
            },
        ))
    }

    /// Process mono audio samples and return the gain-adjusted output.
    ///
    /// This is the convenience variant used by `RestoreChain`.
    pub fn process(&self, samples: &[f32]) -> RestoreResult<Vec<f32>> {
        let (output, _) = self.process_with_result(samples)?;
        Ok(output)
    }

    /// Apply both stages of K-weighting to the samples.
    fn apply_kweighting(&self, samples: &[f32]) -> RestoreResult<Vec<f32>> {
        if self.sample_rate == 0 {
            return Err(RestoreError::InvalidParameter(
                "sample_rate must be > 0".to_string(),
            ));
        }

        let mut stage1 = kw_prefilter(self.sample_rate);
        let mut stage2 = kw_highpass(self.sample_rate);

        let after_stage1 = stage1.process(samples);
        Ok(stage2.process(&after_stage1))
    }

    /// Reset internal filter state (if any is accumulated).
    ///
    /// The current implementation is stateless between `process` calls, so this
    /// is a no-op provided for API symmetry.
    pub fn reset(&mut self) {
        // Stateless — nothing to reset.
    }

    /// Return the configured target integrated loudness in LKFS.
    #[must_use]
    pub fn target_lkfs(&self) -> f32 {
        self.config.target_lkfs
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn sine_wave(freq_hz: f32, sample_rate: u32, duration_s: f32, amplitude: f32) -> Vec<f32> {
        let n = (sample_rate as f32 * duration_s) as usize;
        (0..n)
            .map(|i| amplitude * (2.0 * PI * freq_hz * i as f32 / sample_rate as f32).sin())
            .collect()
    }

    #[test]
    fn test_config_default() {
        let cfg = LoudnessNormalizerConfig::default();
        assert!((cfg.target_lkfs - (-23.0)).abs() < 1e-5);
        assert!(cfg.max_gain_db > 0.0);
    }

    #[test]
    fn test_process_empty_input() {
        let normalizer = LoudnessNormalizer::new(LoudnessNormalizerConfig::default(), 48000);
        let result = normalizer.process(&[]).expect("should succeed");
        assert!(result.is_empty());
    }

    #[test]
    fn test_process_does_not_change_length() {
        let normalizer = LoudnessNormalizer::new(LoudnessNormalizerConfig::default(), 48000);
        let input = sine_wave(1000.0, 48000, 2.0, 0.5);
        let output = normalizer.process(&input).expect("should succeed");
        assert_eq!(output.len(), input.len());
    }

    #[test]
    fn test_process_with_result_returns_measurement() {
        let normalizer = LoudnessNormalizer::new(LoudnessNormalizerConfig::default(), 48000);
        let input = sine_wave(1000.0, 48000, 5.0, 0.5);
        let (_output, result) = normalizer
            .process_with_result(&input)
            .expect("should succeed");
        // A 5-second 1 kHz sine at 0.5 amplitude should produce a valid measurement
        assert!(
            result.measured_lkfs.is_some(),
            "expected a loudness measurement"
        );
        assert!(result.linear_gain > 0.0);
        assert!(result.linear_gain.is_finite());
    }

    #[test]
    fn test_gain_applied_correctly() {
        // Very loud signal should be attenuated toward target.
        let normalizer = LoudnessNormalizer::new(LoudnessNormalizerConfig::default(), 48000);
        let input = sine_wave(1000.0, 48000, 5.0, 0.99); // near full scale
        let (output, result) = normalizer
            .process_with_result(&input)
            .expect("should succeed");

        if let Some(measured) = result.measured_lkfs {
            if measured > -23.0 {
                // Signal louder than target — should be attenuated
                let rms_in: f32 =
                    (input.iter().map(|&s| s * s).sum::<f32>() / input.len() as f32).sqrt();
                let rms_out: f32 =
                    (output.iter().map(|&s| s * s).sum::<f32>() / output.len() as f32).sqrt();
                assert!(rms_out < rms_in, "loud input should be attenuated");
            }
        }
    }

    #[test]
    fn test_gain_clamped_to_max() {
        // Extremely quiet signal should not be amplified beyond max_gain_db
        let cfg = LoudnessNormalizerConfig {
            max_gain_db: 6.0,
            ..Default::default()
        };
        let normalizer = LoudnessNormalizer::new(cfg, 48000);
        let input = sine_wave(1000.0, 48000, 5.0, 0.0001); // very quiet
        let (_output, result) = normalizer
            .process_with_result(&input)
            .expect("should succeed");

        let max_lin = 10.0_f32.powf(6.0 / 20.0);
        assert!(
            result.linear_gain <= max_lin + 1e-4,
            "gain exceeded max: {} > {}",
            result.linear_gain,
            max_lin
        );
    }

    #[test]
    fn test_kweighting_produces_same_length() {
        let normalizer = LoudnessNormalizer::new(LoudnessNormalizerConfig::default(), 44100);
        let input = sine_wave(440.0, 44100, 1.0, 0.5);
        let kw = normalizer.apply_kweighting(&input).expect("should succeed");
        assert_eq!(kw.len(), input.len());
    }

    #[test]
    fn test_measure_integrated_loudness_silence_returns_none() {
        let silence = vec![0.0_f32; 48000];
        let result = measure_integrated_loudness(&silence, 48000, 0.4, 0.1);
        assert!(result.is_none(), "silence should yield None");
    }

    #[test]
    fn test_measure_integrated_loudness_sine_is_finite() {
        let samples = sine_wave(1000.0, 48000, 10.0, 0.5);
        let result = measure_integrated_loudness(&samples, 48000, 0.4, 0.1);
        if let Some(lkfs) = result {
            assert!(lkfs.is_finite(), "lkfs should be finite, got {lkfs}");
            assert!(lkfs < 0.0, "lkfs should be negative for < 0 dBFS signal");
        }
    }

    #[test]
    fn test_biquad_reset() {
        let mut bq = kw_prefilter(48000);
        let samples = vec![1.0_f32; 100];
        let _ = bq.process(&samples);
        bq.reset();
        assert_eq!(bq.s1, 0.0);
        assert_eq!(bq.s2, 0.0);
    }
}
