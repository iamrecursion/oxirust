//! Watermark strength analysis and adaptive strength control.
//!
//! This module provides tools for analysing the optimal embedding strength of a
//! watermark given a host signal, as well as adaptive algorithms that adjust
//! strength on a per-frame basis to balance imperceptibility and robustness.
//!
//! The [`WatermarkStrengthTuner`] uses binary search to find the maximum
//! embedding strength that keeps the perceptual PSNR above a caller-supplied
//! threshold.

#![allow(dead_code)]

use std::collections::VecDeque;

// ──────────────────────────────────────────────────────────────────────────────
// WatermarkEmbedder trait (local definition for strength tuning)
// ──────────────────────────────────────────────────────────────────────────────

/// Minimal embedding interface required by [`WatermarkStrengthTuner`].
///
/// Implementors take a signal and a payload and return the watermarked signal.
pub trait WatermarkEmbedder {
    /// Embed `payload` into `original` at the given `strength`, returning the
    /// watermarked signal.
    ///
    /// # Errors
    ///
    /// Returns a boxed error if embedding fails (e.g. signal too short).
    fn embed_with_strength(
        &self,
        original: &[f32],
        payload: &[u8],
        strength: f32,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>>;
}

// ──────────────────────────────────────────────────────────────────────────────
// PSNR helper
// ──────────────────────────────────────────────────────────────────────────────

/// Compute PSNR between `original` and `watermarked` f32 signals normalised
/// to `[−1, 1]`.
///
/// PSNR is defined as `20 × log₁₀(1 / RMSE)` where RMSE is the root-mean-
/// square error between the two signals.  Returns `f64::INFINITY` when the
/// signals are identical (RMSE = 0).  Returns `f64::NEG_INFINITY` when the
/// slice is empty.
#[must_use]
pub fn compute_psnr_f32(original: &[f32], watermarked: &[f32]) -> f64 {
    let n = original.len().min(watermarked.len());
    if n == 0 {
        return f64::NEG_INFINITY;
    }
    #[allow(clippy::cast_precision_loss)]
    let mse: f64 = original
        .iter()
        .zip(watermarked.iter())
        .map(|(&o, &w)| {
            let d = f64::from(o) - f64::from(w);
            d * d
        })
        .sum::<f64>()
        / n as f64;

    if mse == 0.0 {
        return f64::INFINITY;
    }
    20.0 * mse.sqrt().recip().log10()
}

// ──────────────────────────────────────────────────────────────────────────────
// Strength tuner
// ──────────────────────────────────────────────────────────────────────────────

/// Finds the maximum watermark embedding strength that keeps the signal PSNR
/// above a caller-specified threshold via binary search.
///
/// # Algorithm
///
/// Binary search over `[lo, hi]` where `lo = 0.01` and `hi = 1.0`.  At each
/// iteration the midpoint strength is tried; if the resulting PSNR is still
/// above `target_psnr_db` the midpoint becomes the new lower bound, otherwise
/// it becomes the new upper bound.  The search runs for at most
/// [`WatermarkStrengthTuner::MAX_ITERATIONS`] iterations.
pub struct WatermarkStrengthTuner;

impl WatermarkStrengthTuner {
    /// Maximum number of binary-search iterations.
    pub const MAX_ITERATIONS: usize = 30;

    /// Find the maximum strength in `[0.01, 1.0]` such that embedding
    /// `payload` into `original` produces PSNR ≥ `target_psnr_db`.
    ///
    /// Returns the best strength found, or an error if even the minimum
    /// strength `0.01` fails to produce a valid watermarked signal.
    ///
    /// # Errors
    ///
    /// Returns a boxed error if:
    /// * The embedder returns an error for the minimum strength.
    /// * `target_psnr_db` is non-finite.
    pub fn find_optimal_strength(
        embedder: &dyn WatermarkEmbedder,
        original: &[f32],
        payload: &[u8],
        target_psnr_db: f64,
    ) -> Result<f32, Box<dyn std::error::Error + Send + Sync>> {
        if !target_psnr_db.is_finite() {
            return Err("target_psnr_db must be finite".into());
        }

        let mut lo: f32 = 0.01;
        let mut hi: f32 = 1.0;

        // Check that the minimum strength is at least feasible.
        let wm_lo = embedder.embed_with_strength(original, payload, lo)?;
        let psnr_lo = compute_psnr_f32(original, &wm_lo);
        if psnr_lo < target_psnr_db {
            // Even the minimum is too strong (very tight threshold). Return lo.
            return Ok(lo);
        }

        // `lo` passed the PSNR check — use it as the starting best.
        let mut best = lo;

        for _ in 0..Self::MAX_ITERATIONS {
            let mid = lo + (hi - lo) * 0.5;
            if (hi - lo) < 1e-6 {
                break;
            }

            let wm_mid = match embedder.embed_with_strength(original, payload, mid) {
                Ok(wm) => wm,
                Err(_) => {
                    // Embedder rejected mid — strength is too high.
                    hi = mid;
                    continue;
                }
            };

            let psnr = compute_psnr_f32(original, &wm_mid);
            if psnr >= target_psnr_db {
                // Strength `mid` is acceptable — try higher.
                best = mid;
                lo = mid;
            } else {
                // PSNR too low — reduce strength.
                hi = mid;
            }
        }

        Ok(best)
    }
}

// ---------------------------------------------------------------------------
// Strength profile
// ---------------------------------------------------------------------------

/// A per-frame strength profile describing how much modification each segment
/// of the audio can tolerate.
#[derive(Debug, Clone)]
pub struct StrengthProfile {
    /// Strength value for each frame in [0.0, 1.0].
    pub values: Vec<f64>,
    /// Frame size that was used for analysis.
    pub frame_size: usize,
    /// Hop size between frames.
    pub hop_size: usize,
}

impl StrengthProfile {
    /// Return the average strength across all frames.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn mean(&self) -> f64 {
        if self.values.is_empty() {
            return 0.0;
        }
        self.values.iter().sum::<f64>() / self.values.len() as f64
    }

    /// Return the minimum strength across all frames.
    pub fn min(&self) -> f64 {
        self.values.iter().copied().fold(f64::INFINITY, f64::min)
    }

    /// Return the maximum strength across all frames.
    pub fn max(&self) -> f64 {
        self.values
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Return the strength at a specific frame index.
    #[must_use]
    pub fn at(&self, frame_index: usize) -> Option<f64> {
        self.values.get(frame_index).copied()
    }

    /// Number of frames.
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether the profile is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Analyser configuration
// ---------------------------------------------------------------------------

/// Configuration for the strength analyser.
#[derive(Debug, Clone)]
pub struct StrengthAnalyserConfig {
    /// Frame size for analysis windows.
    pub frame_size: usize,
    /// Hop size between frames.
    pub hop_size: usize,
    /// Target minimum SNR in dB after embedding.
    pub target_snr_db: f64,
    /// Maximum allowed strength (cap).
    pub max_strength: f64,
    /// Minimum allowed strength (floor).
    pub min_strength: f64,
    /// Smoothing window (number of frames for temporal smoothing).
    pub smoothing_window: usize,
}

impl Default for StrengthAnalyserConfig {
    fn default() -> Self {
        Self {
            frame_size: 2048,
            hop_size: 1024,
            target_snr_db: 30.0,
            max_strength: 0.3,
            min_strength: 0.001,
            smoothing_window: 5,
        }
    }
}

// ---------------------------------------------------------------------------
// Analyser
// ---------------------------------------------------------------------------

/// Analyses audio to produce an adaptive strength profile.
#[derive(Debug, Clone)]
pub struct StrengthAnalyser {
    /// Analyser configuration.
    pub config: StrengthAnalyserConfig,
}

impl StrengthAnalyser {
    /// Create a new analyser with the given configuration.
    #[must_use]
    pub fn new(config: StrengthAnalyserConfig) -> Self {
        Self { config }
    }

    /// Analyse a signal and produce a per-frame strength profile.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn analyse(&self, samples: &[f64]) -> StrengthProfile {
        let num_frames = if samples.len() >= self.config.frame_size {
            (samples.len() - self.config.frame_size) / self.config.hop_size + 1
        } else {
            0
        };

        let mut raw = Vec::with_capacity(num_frames);
        for f in 0..num_frames {
            let start = f * self.config.hop_size;
            let end = start + self.config.frame_size;
            let frame = &samples[start..end];
            let rms = frame_rms(frame);
            // Strength proportional to RMS: louder frames can hide more
            let target_noise = rms / db_to_linear(self.config.target_snr_db);
            let strength = target_noise.clamp(self.config.min_strength, self.config.max_strength);
            raw.push(strength);
        }

        let smoothed = temporal_smooth(&raw, self.config.smoothing_window);

        StrengthProfile {
            values: smoothed,
            frame_size: self.config.frame_size,
            hop_size: self.config.hop_size,
        }
    }

    /// Quick estimate of the optimal uniform strength for the whole signal.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn optimal_uniform(&self, samples: &[f64]) -> f64 {
        let rms = frame_rms(samples);
        let target_noise = rms / db_to_linear(self.config.target_snr_db);
        target_noise.clamp(self.config.min_strength, self.config.max_strength)
    }
}

// ---------------------------------------------------------------------------
// Strength validator
// ---------------------------------------------------------------------------

/// Validation result for a given embedding strength.
#[derive(Debug, Clone)]
pub struct StrengthValidation {
    /// Whether the strength passes the quality threshold.
    pub passes: bool,
    /// Estimated SNR after embedding with this strength.
    pub estimated_snr_db: f64,
    /// Suggested strength if current one doesn't pass.
    pub suggested_strength: f64,
    /// Per-frame quality estimates.
    pub frame_snrs: Vec<f64>,
}

/// Validate that a given strength will maintain quality targets.
#[allow(clippy::cast_precision_loss)]
pub fn validate_strength(
    samples: &[f64],
    strength: f64,
    target_snr_db: f64,
    frame_size: usize,
    hop_size: usize,
) -> StrengthValidation {
    let num_frames = if samples.len() >= frame_size {
        (samples.len() - frame_size) / hop_size + 1
    } else {
        0
    };

    let mut frame_snrs = Vec::with_capacity(num_frames);
    for f in 0..num_frames {
        let start = f * hop_size;
        let end = start + frame_size;
        let frame = &samples[start..end];
        let rms = frame_rms(frame);
        let noise_rms = strength;
        let snr = if noise_rms > 0.0 {
            20.0 * (rms / noise_rms).log10()
        } else {
            f64::INFINITY
        };
        frame_snrs.push(snr);
    }

    let min_snr = frame_snrs.iter().copied().fold(f64::INFINITY, f64::min);
    let passes = min_snr >= target_snr_db || frame_snrs.is_empty();

    // Suggest a strength that would achieve the target for the quietest frame
    let min_rms = (0..num_frames)
        .map(|f| {
            let start = f * hop_size;
            let end = start + frame_size;
            frame_rms(&samples[start..end])
        })
        .fold(f64::INFINITY, f64::min);
    let suggested = if min_rms.is_finite() && min_rms > 0.0 {
        min_rms / db_to_linear(target_snr_db)
    } else {
        0.001
    };

    StrengthValidation {
        passes,
        estimated_snr_db: min_snr,
        suggested_strength: suggested,
        frame_snrs,
    }
}

// ---------------------------------------------------------------------------
// Envelope follower
// ---------------------------------------------------------------------------

/// Real-time strength envelope follower that tracks audio level and outputs
/// a smoothed strength value.
#[derive(Debug, Clone)]
pub struct StrengthEnvelopeFollower {
    /// Attack time constant (samples).
    pub attack: f64,
    /// Release time constant (samples).
    pub release: f64,
    /// Current envelope value.
    envelope: f64,
    /// Minimum output strength.
    pub min_strength: f64,
    /// Maximum output strength.
    pub max_strength: f64,
    /// SNR target in dB.
    pub target_snr_db: f64,
}

impl StrengthEnvelopeFollower {
    /// Create a new envelope follower.
    #[must_use]
    pub fn new(attack: f64, release: f64, target_snr_db: f64) -> Self {
        Self {
            attack,
            release,
            envelope: 0.0,
            min_strength: 0.001,
            max_strength: 0.3,
            target_snr_db,
        }
    }

    /// Process one sample and return the recommended strength.
    pub fn process(&mut self, sample: f64) -> f64 {
        let abs_val = sample.abs();
        if abs_val > self.envelope {
            self.envelope += self.attack * (abs_val - self.envelope);
        } else {
            self.envelope += self.release * (abs_val - self.envelope);
        }
        let strength = self.envelope / db_to_linear(self.target_snr_db);
        strength.clamp(self.min_strength, self.max_strength)
    }

    /// Reset the envelope state.
    pub fn reset(&mut self) {
        self.envelope = 0.0;
    }

    /// Get the current envelope level.
    #[must_use]
    pub fn current_envelope(&self) -> f64 {
        self.envelope
    }
}

// ---------------------------------------------------------------------------
// Strength histogram
// ---------------------------------------------------------------------------

/// Histogram of strength values across frames.
#[derive(Debug, Clone)]
pub struct StrengthHistogram {
    /// Bin counts.
    pub bins: Vec<usize>,
    /// Bin edges (len = `bins.len()` + 1).
    pub edges: Vec<f64>,
}

impl StrengthHistogram {
    /// Build a histogram from a strength profile.
    #[must_use]
    pub fn from_profile(profile: &StrengthProfile, num_bins: usize) -> Self {
        let num_bins = num_bins.max(1);
        let lo = profile.min().min(0.0);
        let hi = profile.max().max(lo + 1e-12);
        let bin_width = (hi - lo) / num_bins as f64;

        let mut bins = vec![0usize; num_bins];
        let mut edges = Vec::with_capacity(num_bins + 1);
        for i in 0..=num_bins {
            edges.push(lo + i as f64 * bin_width);
        }

        for &v in &profile.values {
            let idx = ((v - lo) / bin_width) as usize;
            let idx = idx.min(num_bins - 1);
            bins[idx] += 1;
        }

        Self { bins, edges }
    }

    /// Return the bin with the most counts (mode).
    #[must_use]
    pub fn mode_bin(&self) -> usize {
        self.bins
            .iter()
            .enumerate()
            .max_by_key(|(_, &c)| c)
            .map_or(0, |(i, _)| i)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// RMS of a frame.
#[allow(clippy::cast_precision_loss)]
fn frame_rms(frame: &[f64]) -> f64 {
    if frame.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = frame.iter().map(|s| s * s).sum();
    (sum_sq / frame.len() as f64).sqrt()
}

/// Convert decibels to linear amplitude ratio.
fn db_to_linear(db: f64) -> f64 {
    10.0f64.powf(db / 20.0)
}

/// Temporal smoothing via moving average.
fn temporal_smooth(values: &[f64], window: usize) -> Vec<f64> {
    if window <= 1 || values.is_empty() {
        return values.to_vec();
    }
    let mut result = Vec::with_capacity(values.len());
    let mut buf: VecDeque<f64> = VecDeque::with_capacity(window);
    let mut running_sum = 0.0f64;

    for &v in values {
        buf.push_back(v);
        running_sum += v;
        if buf.len() > window {
            running_sum -= buf.pop_front().unwrap_or(0.0);
        }
        #[allow(clippy::cast_precision_loss)]
        let avg = running_sum / buf.len() as f64;
        result.push(avg);
    }
    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = StrengthAnalyserConfig::default();
        assert_eq!(cfg.frame_size, 2048);
        assert_eq!(cfg.hop_size, 1024);
        assert!((cfg.target_snr_db - 30.0).abs() < 1e-9);
    }

    #[test]
    fn test_frame_rms_silence() {
        let frame = vec![0.0f64; 512];
        assert!(frame_rms(&frame).abs() < 1e-12);
    }

    #[test]
    fn test_frame_rms_constant() {
        let frame = vec![0.5f64; 1000];
        let rms = frame_rms(&frame);
        assert!((rms - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_db_to_linear() {
        assert!((db_to_linear(0.0) - 1.0).abs() < 1e-9);
        assert!((db_to_linear(20.0) - 10.0).abs() < 1e-6);
        assert!((db_to_linear(40.0) - 100.0).abs() < 1e-4);
    }

    #[test]
    fn test_temporal_smooth_passthrough() {
        let vals = vec![1.0, 2.0, 3.0, 4.0];
        let smoothed = temporal_smooth(&vals, 1);
        assert_eq!(smoothed, vals);
    }

    #[test]
    fn test_temporal_smooth_window() {
        let vals = vec![0.0, 0.0, 10.0, 0.0, 0.0];
        let smoothed = temporal_smooth(&vals, 3);
        // After the spike, smoothing should spread it
        assert!(smoothed[2] > smoothed[0]);
    }

    #[test]
    fn test_analyser_produces_profile() {
        let config = StrengthAnalyserConfig {
            frame_size: 256,
            hop_size: 128,
            smoothing_window: 3,
            ..Default::default()
        };
        let analyser = StrengthAnalyser::new(config);
        let signal: Vec<f64> = (0..4096).map(|i| (i as f64 * 0.01).sin()).collect();
        let profile = analyser.analyse(&signal);
        assert!(!profile.is_empty());
        assert!(profile.len() > 10);
        for &v in &profile.values {
            assert!(v >= 0.001); // min_strength
            assert!(v <= 0.3); // max_strength
        }
    }

    #[test]
    fn test_optimal_uniform() {
        let analyser = StrengthAnalyser::new(StrengthAnalyserConfig::default());
        let signal = vec![0.5f64; 4096];
        let strength = analyser.optimal_uniform(&signal);
        assert!(strength > 0.0);
        assert!(strength <= 0.3);
    }

    #[test]
    fn test_validate_strength_passes() {
        let signal: Vec<f64> = vec![1.0; 4096];
        let v = validate_strength(&signal, 0.001, 30.0, 256, 128);
        assert!(v.passes);
        assert!(v.estimated_snr_db > 30.0);
    }

    #[test]
    fn test_validate_strength_fails() {
        let signal: Vec<f64> = vec![0.001; 4096];
        let v = validate_strength(&signal, 0.5, 60.0, 256, 128);
        assert!(!v.passes);
        assert!(v.suggested_strength < 0.5);
    }

    #[test]
    fn test_envelope_follower() {
        let mut follower = StrengthEnvelopeFollower::new(0.1, 0.01, 30.0);
        let s1 = follower.process(0.8);
        assert!(s1 > 0.0);
        let s2 = follower.process(0.0);
        // Release is slow, so strength should still be positive
        assert!(s2 > 0.0);
        follower.reset();
        assert!(follower.current_envelope().abs() < 1e-12);
    }

    #[test]
    fn test_strength_profile_stats() {
        let profile = StrengthProfile {
            values: vec![0.1, 0.2, 0.3, 0.4, 0.5],
            frame_size: 256,
            hop_size: 128,
        };
        assert!((profile.mean() - 0.3).abs() < 1e-9);
        assert!((profile.min() - 0.1).abs() < 1e-9);
        assert!((profile.max() - 0.5).abs() < 1e-9);
        assert_eq!(profile.at(2), Some(0.3));
        assert_eq!(profile.at(99), None);
        assert_eq!(profile.len(), 5);
        assert!(!profile.is_empty());
    }

    #[test]
    fn test_histogram_from_profile() {
        let profile = StrengthProfile {
            values: vec![0.1, 0.1, 0.1, 0.5, 0.5],
            frame_size: 256,
            hop_size: 128,
        };
        let hist = StrengthHistogram::from_profile(&profile, 4);
        assert_eq!(hist.bins.len(), 4);
        assert_eq!(hist.edges.len(), 5);
        let total: usize = hist.bins.iter().sum();
        assert_eq!(total, 5);
        // Mode should be the first bin (contains three 0.1 values)
        assert_eq!(hist.mode_bin(), 0);
    }

    #[test]
    fn test_empty_signal() {
        let analyser = StrengthAnalyser::new(StrengthAnalyserConfig::default());
        let profile = analyser.analyse(&[]);
        assert!(profile.is_empty());
    }

    // ── Item 3: WatermarkStrengthTuner ───────────────────────────────────────

    /// Trivial embedder that adds `strength * 0.1` to every sample.
    struct TrivialEmbedder;

    impl WatermarkEmbedder for TrivialEmbedder {
        fn embed_with_strength(
            &self,
            original: &[f32],
            _payload: &[u8],
            strength: f32,
        ) -> Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
            // Add a constant offset proportional to strength.
            let wm: Vec<f32> = original.iter().map(|&s| s + strength * 0.1).collect();
            Ok(wm)
        }
    }

    #[test]
    fn test_strength_tuner_converges() {
        // Signal: constant 0.5, embedder adds noise proportional to strength.
        let original = vec![0.5f32; 1024];
        let payload = b"test";
        // Target PSNR of 20 dB; tuner should converge to some valid strength.
        let strength = WatermarkStrengthTuner::find_optimal_strength(
            &TrivialEmbedder,
            &original,
            payload,
            20.0,
        )
        .expect("tuner should converge in test");
        assert!(
            strength >= 0.01,
            "strength should be at least minimum, got {strength}"
        );
        assert!(
            strength <= 1.0,
            "strength should be at most maximum, got {strength}"
        );

        // Verify that the returned strength actually achieves the target.
        let wm = TrivialEmbedder
            .embed_with_strength(&original, payload, strength)
            .expect("embed should succeed in test");
        let psnr = compute_psnr_f32(&original, &wm);
        assert!(
            psnr >= 20.0,
            "PSNR {psnr:.2} dB should be >= target 20.0 dB"
        );
    }

    #[test]
    fn test_strength_tuner_high_psnr_low_strength() {
        // With a very high PSNR requirement (60 dB), the tuner should return a
        // small strength value because any large modification lowers PSNR.
        let original = vec![0.5f32; 1024];
        let payload = b"x";
        let strength = WatermarkStrengthTuner::find_optimal_strength(
            &TrivialEmbedder,
            &original,
            payload,
            60.0,
        )
        .expect("tuner should succeed in test");
        assert!(
            strength < 0.5,
            "high PSNR requirement should force low strength, got {strength}"
        );
    }

    #[test]
    fn test_compute_psnr_f32_identical() {
        let signal = vec![0.5f32; 256];
        let psnr = compute_psnr_f32(&signal, &signal);
        assert!(psnr.is_infinite() && psnr.is_sign_positive());
    }

    #[test]
    fn test_compute_psnr_f32_empty() {
        let psnr = compute_psnr_f32(&[], &[]);
        assert!(psnr.is_infinite() && psnr.is_sign_negative());
    }
}
