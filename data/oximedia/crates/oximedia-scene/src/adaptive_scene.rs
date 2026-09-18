//! Adaptive scene detection with complexity histogram analysis.
//!
//! This module provides an adaptive scene detector that measures content complexity
//! from colour/intensity histogram spread and variance, then dynamically adjusts the
//! detection threshold according to a sliding window of recent frame complexity scores.
//!
//! # Algorithm
//!
//! For each frame the complexity score is computed as a weighted combination of:
//!
//! * **Shannon entropy** of the normalised intensity histogram — measures how spread
//!   out the pixel-intensity distribution is.  A flat (uniform) histogram has maximum
//!   entropy (~8 bits for 256 bins); a narrow spike has near-zero entropy.
//! * **Variance** of the normalised histogram — captures how much the bin counts
//!   deviate from their mean, complementing entropy in heterogeneous distributions.
//! * **Spread** (inter-quartile range of occupied bins) — penalises thin-tailed
//!   distributions even when a few outlier bins inflate entropy.
//!
//! The adaptive threshold is then interpolated between `min_threshold` and
//! `max_threshold` proportionally to the mean complexity across the sliding window:
//!
//! ```text
//! threshold = min_threshold + mean_complexity * (max_threshold - min_threshold)
//! ```
//!
//! A high-complexity scene (many distinct tones, rich texture) warrants a *higher*
//! threshold so that minor content variation is not spuriously flagged as a cut.
//! A low-complexity scene (uniform background, fade-to-black) benefits from a
//! *lower* threshold so that subtle boundary signals are not missed.

use crate::error::{SceneError, SceneResult};

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the adaptive scene detector.
///
/// All threshold fields are in \[0, 1\].
#[derive(Debug, Clone, PartialEq)]
pub struct AdaptiveConfig {
    /// Base threshold used when the complexity window is empty.
    pub base_threshold: f32,
    /// Lower bound on the adaptive threshold (used for simple/dark content).
    pub min_threshold: f32,
    /// Upper bound on the adaptive threshold (used for complex/rich content).
    pub max_threshold: f32,
    /// Number of recent frames considered when computing the mean complexity.
    pub window_size: usize,
    /// Weight of the entropy component in the complexity score (0–1).
    pub entropy_weight: f32,
    /// Weight of the variance component in the complexity score (0–1).
    pub variance_weight: f32,
    /// Weight of the spread (IQR) component in the complexity score (0–1).
    pub spread_weight: f32,
    /// Number of histogram bins used when computing complexity from raw pixels.
    pub histogram_bins: usize,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            base_threshold: 0.35,
            min_threshold: 0.15,
            max_threshold: 0.55,
            window_size: 30,
            entropy_weight: 0.5,
            variance_weight: 0.3,
            spread_weight: 0.2,
            histogram_bins: 256,
        }
    }
}

impl AdaptiveConfig {
    /// Validate configuration; returns `Err` if any field is out of range.
    pub fn validate(&self) -> SceneResult<()> {
        if self.min_threshold > self.max_threshold {
            return Err(SceneError::InvalidParameter(format!(
                "min_threshold ({}) must be ≤ max_threshold ({})",
                self.min_threshold, self.max_threshold
            )));
        }
        if !(0.0..=1.0).contains(&self.base_threshold) {
            return Err(SceneError::InvalidParameter(format!(
                "base_threshold {} out of [0, 1]",
                self.base_threshold
            )));
        }
        if !(0.0..=1.0).contains(&self.min_threshold) {
            return Err(SceneError::InvalidParameter(format!(
                "min_threshold {} out of [0, 1]",
                self.min_threshold
            )));
        }
        if !(0.0..=1.0).contains(&self.max_threshold) {
            return Err(SceneError::InvalidParameter(format!(
                "max_threshold {} out of [0, 1]",
                self.max_threshold
            )));
        }
        if self.window_size == 0 {
            return Err(SceneError::InvalidParameter(
                "window_size must be > 0".into(),
            ));
        }
        if self.histogram_bins < 2 {
            return Err(SceneError::InvalidParameter(
                "histogram_bins must be ≥ 2".into(),
            ));
        }
        let weight_sum = self.entropy_weight + self.variance_weight + self.spread_weight;
        if (weight_sum - 1.0).abs() > 0.01 {
            return Err(SceneError::InvalidParameter(format!(
                "entropy_weight + variance_weight + spread_weight must sum to 1.0, got {weight_sum}"
            )));
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Complexity score
// ─────────────────────────────────────────────────────────────────────────────

/// Normalised complexity score for a single frame in \[0, 1\].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComplexityScore {
    /// Shannon entropy component (normalised to \[0, 1\]).
    pub entropy: f32,
    /// Histogram variance component (normalised to \[0, 1\]).
    pub variance: f32,
    /// Histogram spread (IQR) component (normalised to \[0, 1\]).
    pub spread: f32,
    /// Weighted sum of the three components.
    pub overall: f32,
}

/// Compute the [`ComplexityScore`] from a pre-built intensity histogram.
///
/// `histogram` must be a non-empty slice of non-negative counts; it need not
/// be normalised beforehand.
///
/// `config` drives the per-component weights and the bin count.
///
/// # Errors
///
/// Returns [`SceneError::InsufficientData`] if `histogram` is empty.
pub fn compute_complexity_from_histogram(
    histogram: &[u64],
    config: &AdaptiveConfig,
) -> SceneResult<ComplexityScore> {
    if histogram.is_empty() {
        return Err(SceneError::InsufficientData(
            "histogram must not be empty".into(),
        ));
    }

    let total: u64 = histogram.iter().sum();
    if total == 0 {
        // All-zero image → zero complexity
        return Ok(ComplexityScore {
            entropy: 0.0,
            variance: 0.0,
            spread: 0.0,
            overall: 0.0,
        });
    }

    let n = histogram.len() as f64;
    let total_f = total as f64;

    // Normalise to probability distribution
    let probs: Vec<f64> = histogram.iter().map(|&c| c as f64 / total_f).collect();

    // --- Shannon entropy ---
    let raw_entropy: f64 = probs
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum();
    // Maximum possible entropy for `n` bins is ln(n)
    let max_entropy = n.ln();
    let norm_entropy = if max_entropy > 0.0 {
        (raw_entropy / max_entropy).clamp(0.0, 1.0)
    } else {
        0.0
    };

    // --- Histogram variance ---
    let mean_prob = 1.0 / n; // for a uniform distribution
    let raw_variance: f64 = probs.iter().map(|&p| (p - mean_prob).powi(2)).sum::<f64>() / n;
    // Maximum variance happens when all mass is in one bin: p=1, rest=0
    // max_var = (1 - 1/n)^2/n + (n-1)*(0 - 1/n)^2/n = (1/n)(1 - 1/n)
    let max_variance = (1.0 / n) * (1.0 - 1.0 / n);
    // High variance → *low* complexity (concentrated distribution)
    // Invert so that spread distributions score high
    let norm_variance = if max_variance > 0.0 {
        (1.0 - (raw_variance / max_variance).clamp(0.0, 1.0)) as f64
    } else {
        0.0
    };

    // --- Spread: IQR of occupied bins ---
    // Compute the 25th and 75th percentile bin indices weighted by probability.
    let spread_score = compute_iqr_spread(&probs);

    let overall = (norm_entropy as f32 * config.entropy_weight)
        + (norm_variance as f32 * config.variance_weight)
        + (spread_score * config.spread_weight);

    Ok(ComplexityScore {
        entropy: norm_entropy as f32,
        variance: norm_variance as f32,
        spread: spread_score,
        overall: overall.clamp(0.0, 1.0),
    })
}

/// Compute the IQR spread of a normalised probability histogram.
///
/// Returns a value in \[0, 1\] where 1 = maximum spread (uniform distribution).
fn compute_iqr_spread(probs: &[f64]) -> f32 {
    let n = probs.len();
    if n < 4 {
        return 0.5;
    }

    // Build cumulative distribution
    let mut cdf = Vec::with_capacity(n);
    let mut cumulative = 0.0f64;
    for &p in probs {
        cumulative += p;
        cdf.push(cumulative);
    }

    // Find the bin index where CDF first crosses 0.25 and 0.75
    let q1 = cdf.iter().position(|&c| c >= 0.25).unwrap_or(0) as f64;
    let q3 = cdf.iter().position(|&c| c >= 0.75).unwrap_or(n - 1) as f64;

    let iqr = q3 - q1;
    // Normalise by the maximum possible IQR which is n/2
    ((iqr / (n as f64 / 2.0)).clamp(0.0, 1.0)) as f32
}

/// Build an intensity histogram from raw pixel bytes.
///
/// `pixels` should be grayscale (one byte per pixel) or, for colour frames,
/// the caller should pass the luminance channel.
///
/// `bins` must be ≥ 2.
pub fn build_intensity_histogram(pixels: &[u8], bins: usize) -> Vec<u64> {
    assert!(bins >= 2, "bins must be ≥ 2");
    let mut hist = vec![0u64; bins];
    let scale = bins as f32 / 256.0;
    for &px in pixels {
        let idx = ((px as f32 * scale) as usize).min(bins - 1);
        hist[idx] += 1;
    }
    hist
}

// ─────────────────────────────────────────────────────────────────────────────
// Adaptive detector
// ─────────────────────────────────────────────────────────────────────────────

/// Frame difference descriptor fed to the adaptive detector.
#[derive(Debug, Clone)]
pub struct FrameDiff {
    /// Sequential frame index (zero-based).
    pub frame_index: u64,
    /// Normalised inter-frame difference in \[0, 1\].
    pub diff: f32,
    /// Precomputed complexity score for the *current* frame (optional).
    ///
    /// If `None` the complexity is not used to adjust the threshold for this
    /// particular frame, and only the sliding-window history is applied.
    pub complexity: Option<ComplexityScore>,
}

impl FrameDiff {
    /// Create a new frame-diff descriptor without a complexity score.
    #[must_use]
    pub fn new(frame_index: u64, diff: f32) -> Self {
        Self {
            frame_index,
            diff: diff.clamp(0.0, 1.0),
            complexity: None,
        }
    }

    /// Create a new frame-diff descriptor with a precomputed complexity score.
    #[must_use]
    pub fn with_complexity(frame_index: u64, diff: f32, complexity: ComplexityScore) -> Self {
        Self {
            frame_index,
            diff: diff.clamp(0.0, 1.0),
            complexity: Some(complexity),
        }
    }
}

/// A detected scene cut produced by [`AdaptiveSceneDetector`].
#[derive(Debug, Clone)]
pub struct AdaptiveSceneCut {
    /// Frame index where the cut was detected.
    pub frame_index: u64,
    /// Normalised inter-frame difference that triggered the cut.
    pub diff: f32,
    /// The adaptive threshold that was in effect at this frame.
    pub threshold_used: f32,
    /// Complexity score of the frame (if available).
    pub complexity: Option<ComplexityScore>,
}

/// Internal sliding-window entry.
#[derive(Debug, Clone, Copy)]
struct WindowEntry {
    complexity: f32,
}

/// Adaptive scene detector that dynamically adjusts its detection threshold
/// based on the content complexity of recent frames.
///
/// # Usage
///
/// ```
/// use oximedia_scene::adaptive_scene::{AdaptiveConfig, AdaptiveSceneDetector, FrameDiff};
///
/// let mut detector = AdaptiveSceneDetector::new(AdaptiveConfig::default());
/// detector.push_frame(FrameDiff::new(0, 0.05));
/// detector.push_frame(FrameDiff::new(1, 0.04));
/// detector.push_frame(FrameDiff::new(2, 0.80)); // big jump → scene cut
/// let cuts = detector.detect();
/// assert!(!cuts.is_empty());
/// ```
pub struct AdaptiveSceneDetector {
    config: AdaptiveConfig,
    /// Accumulated frame descriptors (in insertion order).
    frames: Vec<FrameDiff>,
    /// Sliding window of recent per-frame complexity scores.
    complexity_window: std::collections::VecDeque<WindowEntry>,
}

impl AdaptiveSceneDetector {
    /// Create a new detector with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if `config.validate()` fails.
    pub fn new(config: AdaptiveConfig) -> Self {
        Self {
            config,
            frames: Vec::new(),
            complexity_window: std::collections::VecDeque::new(),
        }
    }

    /// Replace the current configuration.
    ///
    /// Existing frame data is preserved; only the detection parameters change.
    ///
    /// # Errors
    ///
    /// Returns an error if `config.validate()` fails.
    pub fn configure(&mut self, config: AdaptiveConfig) -> SceneResult<()> {
        config.validate()?;
        self.config = config;
        Ok(())
    }

    /// Return a reference to the current configuration.
    #[must_use]
    pub const fn config(&self) -> &AdaptiveConfig {
        &self.config
    }

    /// Add a frame descriptor to the detector.
    ///
    /// The complexity score (if present) is added to the sliding window.
    pub fn push_frame(&mut self, frame: FrameDiff) {
        // Update sliding window
        if let Some(score) = frame.complexity {
            self.complexity_window.push_back(WindowEntry {
                complexity: score.overall,
            });
            while self.complexity_window.len() > self.config.window_size {
                self.complexity_window.pop_front();
            }
        }
        self.frames.push(frame);
    }

    /// Compute the current adaptive threshold from the sliding-window mean.
    ///
    /// Falls back to `config.base_threshold` when the window is empty.
    #[must_use]
    pub fn current_threshold(&self) -> f32 {
        self.threshold_for_window_mean(self.window_mean())
    }

    /// Return the mean complexity across the current window.
    fn window_mean(&self) -> Option<f32> {
        if self.complexity_window.is_empty() {
            None
        } else {
            let sum: f32 = self.complexity_window.iter().map(|e| e.complexity).sum();
            Some(sum / self.complexity_window.len() as f32)
        }
    }

    /// Map a mean complexity (or `None`) to a concrete threshold value.
    fn threshold_for_window_mean(&self, mean: Option<f32>) -> f32 {
        match mean {
            None => self.config.base_threshold,
            Some(m) => {
                let m = m.clamp(0.0, 1.0);
                self.config.min_threshold
                    + m * (self.config.max_threshold - self.config.min_threshold)
            }
        }
    }

    /// Detect scene cuts in all accumulated frames and return the cuts found.
    ///
    /// The threshold for each frame is computed from the complexity window
    /// **as it would have been at that point in time** (i.e. the window is
    /// replayed in frame-insertion order).
    #[must_use]
    pub fn detect(&self) -> Vec<AdaptiveSceneCut> {
        let mut cuts = Vec::new();
        let mut window: std::collections::VecDeque<f32> = std::collections::VecDeque::new();

        for frame in &self.frames {
            // Compute threshold from the window state *before* this frame
            let mean = if window.is_empty() {
                None
            } else {
                let sum: f32 = window.iter().sum();
                Some(sum / window.len() as f32)
            };
            let threshold = self.threshold_for_window_mean(mean);

            if frame.diff >= threshold {
                cuts.push(AdaptiveSceneCut {
                    frame_index: frame.frame_index,
                    diff: frame.diff,
                    threshold_used: threshold,
                    complexity: frame.complexity,
                });
            }

            // Update window for subsequent frames
            if let Some(score) = frame.complexity {
                window.push_back(score.overall);
                while window.len() > self.config.window_size {
                    window.pop_front();
                }
            }
        }

        cuts
    }

    /// Detect scene cuts while also supplying raw pixel data for complexity.
    ///
    /// Each element of `frame_data` contains:
    /// * `frame_index` — sequential frame number
    /// * `diff` — normalised inter-frame difference (\[0, 1\])
    /// * `pixels` — grayscale (luminance) pixel bytes for the frame
    ///
    /// The detector automatically builds a histogram from `pixels` and
    /// computes a [`ComplexityScore`] for each frame.
    ///
    /// # Errors
    ///
    /// Returns an error if histogram complexity computation fails.
    pub fn detect_with_pixels(
        &mut self,
        frame_data: &[(u64, f32, &[u8])],
    ) -> SceneResult<Vec<AdaptiveSceneCut>> {
        self.reset();

        for &(frame_index, diff, pixels) in frame_data {
            let histogram = build_intensity_histogram(pixels, self.config.histogram_bins);
            let complexity = compute_complexity_from_histogram(&histogram, &self.config)?;
            self.push_frame(FrameDiff::with_complexity(frame_index, diff, complexity));
        }

        Ok(self.detect())
    }

    /// Remove all accumulated frame data and reset the complexity window.
    pub fn reset(&mut self) {
        self.frames.clear();
        self.complexity_window.clear();
    }

    /// Return the number of frames currently held by the detector.
    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }
}

impl Default for AdaptiveSceneDetector {
    fn default() -> Self {
        Self::new(AdaptiveConfig::default())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Rolling adaptive scene detector (mean + k·σ threshold, flash suppression)
// ─────────────────────────────────────────────────────────────────────────────

/// A scene-change event produced by [`RollingSceneDetector`].
#[derive(Debug, Clone, PartialEq)]
pub struct SceneChange {
    /// Zero-based index of the frame where the cut was detected.
    pub frame_idx: u64,
    /// Confidence in [0, 1]: how far above threshold the difference was.
    /// `confidence = (diff - threshold) / threshold`, clamped to [0, 1].
    pub confidence: f64,
    /// `true` when the detector suspects this is a brief flash rather than a
    /// genuine scene boundary (the frame difference is very large — more than
    /// 2× the adaptive threshold — and the frame history looks low-energy).
    pub is_flash: bool,
}

/// Adaptive scene detector using a rolling mean + k·σ threshold.
///
/// # Algorithm
///
/// A rolling window of recent inter-frame difference scores is maintained.
/// After each call to [`RollingSceneDetector::push_frame_diff`]:
///
/// 1. The adaptive threshold is recomputed as `mean + k × stddev` of the
///    window (falls back to 0.3 when fewer than two samples are present).
/// 2. If the new difference exceeds the threshold *and* at least
///    `min_scene_duration` frames have elapsed since the last cut, a
///    [`SceneChange`] is emitted.
/// 3. Flash suppression: if the difference is ≥ 2× the threshold the cut
///    is marked `is_flash = true`, indicating a transient brightness spike.
/// 4. The difference is appended to the history (oldest entry evicted when the
///    window is full).
///
/// # Example
///
/// ```
/// use oximedia_scene::adaptive_scene::{RollingSceneDetector, SceneChange};
///
/// let mut det = RollingSceneDetector::new(30, 2.0, 5);
///
/// // Feed stable frames
/// for _ in 0..30 {
///     det.push_frame_diff(0.05);
/// }
///
/// // A hard cut
/// let cut = det.push_frame_diff(0.80);
/// assert!(cut.is_some());
/// ```
pub struct RollingSceneDetector {
    /// Number of entries kept in the rolling history.
    window_size: usize,
    /// `threshold = mean + k_factor × stddev`.
    k_factor: f64,
    /// Minimum number of frames between consecutive scene cuts.
    min_scene_duration: u32,
    /// Rolling history of recent inter-frame difference scores.
    history: std::collections::VecDeque<f64>,
    /// Monotonically increasing frame index (incremented on every push).
    frame_idx: u64,
    /// Frame index of the most-recently emitted scene cut.
    last_cut_frame: Option<u64>,
}

impl RollingSceneDetector {
    /// Create a new detector.
    ///
    /// # Parameters
    ///
    /// * `window_size` — number of recent differences used to estimate the
    ///   rolling mean and standard deviation.  Must be ≥ 2; clamped to 2 if
    ///   smaller.
    /// * `k_factor` — multiplier for the standard deviation term.  Typical
    ///   values: 1.5 (sensitive) to 3.0 (conservative).
    /// * `min_scene_duration` — minimum number of frames that must have passed
    ///   since the last cut before a new one can be emitted.
    pub fn new(window_size: usize, k_factor: f64, min_scene_duration: u32) -> Self {
        let window_size = window_size.max(2);
        Self {
            window_size,
            k_factor,
            min_scene_duration,
            history: std::collections::VecDeque::with_capacity(window_size),
            frame_idx: 0,
            last_cut_frame: None,
        }
    }

    /// Submit a new inter-frame difference score and get back a [`SceneChange`]
    /// if a cut was detected.
    ///
    /// The `diff` value should be in [0, 1] (e.g. normalised mean absolute
    /// pixel difference), though values outside that range are accepted.
    pub fn push_frame_diff(&mut self, diff: f64) -> Option<SceneChange> {
        let threshold = self.adaptive_threshold();
        let idx = self.frame_idx;
        self.frame_idx += 1;

        let result = if diff > threshold {
            // Honour min_scene_duration
            let frames_since_last = match self.last_cut_frame {
                None => u64::MAX,
                Some(last) => idx.saturating_sub(last),
            };

            if frames_since_last >= self.min_scene_duration as u64 {
                // Flash suppression: very high difference (≥ 2× threshold)
                // relative to a low-history baseline is considered a flash.
                let is_flash = diff >= 2.0 * threshold && self.history_mean() < threshold;

                // Confidence: normalised distance above threshold.
                let confidence = if threshold > 0.0 {
                    ((diff - threshold) / threshold).clamp(0.0, 1.0)
                } else {
                    1.0
                };

                self.last_cut_frame = Some(idx);
                Some(SceneChange {
                    frame_idx: idx,
                    confidence,
                    is_flash,
                })
            } else {
                None
            }
        } else {
            None
        };

        // Update rolling history AFTER computing the threshold for this frame.
        self.history.push_back(diff);
        if self.history.len() > self.window_size {
            self.history.pop_front();
        }

        result
    }

    /// Current adaptive threshold: `mean + k_factor × stddev` of the history.
    ///
    /// Returns 0.3 when fewer than two samples are available.
    pub fn adaptive_threshold(&self) -> f64 {
        let n = self.history.len();
        if n < 2 {
            return 0.3;
        }

        let mean = self.history_mean();
        let variance = self
            .history
            .iter()
            .map(|&x| {
                let d = x - mean;
                d * d
            })
            .sum::<f64>()
            / n as f64;
        let stddev = variance.sqrt();

        (mean + self.k_factor * stddev).max(0.0)
    }

    // ── helpers ──────────────────────────────────────────────────────────────

    fn history_mean(&self) -> f64 {
        let n = self.history.len();
        if n == 0 {
            return 0.0;
        }
        self.history.iter().sum::<f64>() / n as f64
    }

    /// Return the number of frames that have been submitted so far.
    #[must_use]
    pub fn frame_count(&self) -> u64 {
        self.frame_idx
    }

    /// Reset all state (history, frame counter, last cut).
    pub fn reset(&mut self) {
        self.history.clear();
        self.frame_idx = 0;
        self.last_cut_frame = None;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ─────────────────────────────────────────────────────────────

    /// Build a synthetic grayscale frame that is uniformly distributed across
    /// all 256 levels (maximum complexity / entropy).
    fn uniform_pixels(n_pixels: usize) -> Vec<u8> {
        (0..n_pixels).map(|i| (i % 256) as u8).collect()
    }

    /// Build a synthetic frame whose pixels are all a single constant value
    /// (minimum complexity).
    fn constant_pixels(value: u8, n_pixels: usize) -> Vec<u8> {
        vec![value; n_pixels]
    }

    // ── AdaptiveConfig tests ─────────────────────────────────────────────────

    #[test]
    fn test_config_default_is_valid() {
        assert!(AdaptiveConfig::default().validate().is_ok());
    }

    #[test]
    fn test_config_min_greater_than_max_is_invalid() {
        let cfg = AdaptiveConfig {
            min_threshold: 0.6,
            max_threshold: 0.4,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_base_threshold_out_of_range() {
        let cfg = AdaptiveConfig {
            base_threshold: 1.5,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_zero_window_size_is_invalid() {
        let cfg = AdaptiveConfig {
            window_size: 0,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_bad_weights_is_invalid() {
        let cfg = AdaptiveConfig {
            entropy_weight: 0.5,
            variance_weight: 0.5,
            spread_weight: 0.5, // sums to 1.5
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    // ── complexity computation ────────────────────────────────────────────────

    #[test]
    fn test_uniform_histogram_has_high_entropy() {
        // A perfectly uniform histogram should yield entropy ≈ 1.
        let hist: Vec<u64> = vec![100; 256];
        let config = AdaptiveConfig::default();
        let score = compute_complexity_from_histogram(&hist, &config)
            .expect("uniform histogram should compute");
        assert!(
            score.entropy > 0.99,
            "uniform histogram entropy should be ≈ 1, got {}",
            score.entropy
        );
    }

    #[test]
    fn test_single_bin_histogram_has_zero_entropy() {
        let mut hist = vec![0u64; 256];
        hist[128] = 1000;
        let config = AdaptiveConfig::default();
        let score = compute_complexity_from_histogram(&hist, &config)
            .expect("single-bin histogram should compute");
        assert!(
            score.entropy < 0.01,
            "single-bin histogram entropy should be ≈ 0, got {}",
            score.entropy
        );
    }

    #[test]
    fn test_empty_histogram_returns_error() {
        let config = AdaptiveConfig::default();
        assert!(compute_complexity_from_histogram(&[], &config).is_err());
    }

    #[test]
    fn test_all_zero_histogram_returns_zero_complexity() {
        let hist = vec![0u64; 64];
        let config = AdaptiveConfig::default();
        let score = compute_complexity_from_histogram(&hist, &config)
            .expect("zero histogram should compute");
        assert_eq!(score.overall, 0.0);
    }

    #[test]
    fn test_uniform_pixels_give_higher_complexity_than_constant() {
        let cfg = AdaptiveConfig::default();
        let bins = cfg.histogram_bins;

        let uniform_hist = build_intensity_histogram(&uniform_pixels(4096), bins);
        let constant_hist = build_intensity_histogram(&constant_pixels(128, 4096), bins);

        let uniform_score = compute_complexity_from_histogram(&uniform_hist, &cfg)
            .expect("uniform histogram should compute");
        let constant_score = compute_complexity_from_histogram(&constant_hist, &cfg)
            .expect("constant histogram should compute");

        assert!(
            uniform_score.overall > constant_score.overall,
            "uniform frame (score={}) should be more complex than constant frame (score={})",
            uniform_score.overall,
            constant_score.overall
        );
    }

    // ── build_intensity_histogram ─────────────────────────────────────────────

    #[test]
    fn test_intensity_histogram_bin_count() {
        let pixels = uniform_pixels(512);
        let hist = build_intensity_histogram(&pixels, 64);
        assert_eq!(hist.len(), 64);
    }

    #[test]
    fn test_intensity_histogram_pixel_sum_equals_input_len() {
        let pixels = uniform_pixels(1024);
        let hist = build_intensity_histogram(&pixels, 256);
        let total: u64 = hist.iter().sum();
        assert_eq!(total, 1024);
    }

    // ── AdaptiveSceneDetector ─────────────────────────────────────────────────

    #[test]
    fn test_detector_default_no_cuts_on_stable_sequence() {
        let mut det = AdaptiveSceneDetector::default();
        for i in 0..20u64 {
            det.push_frame(FrameDiff::new(i, 0.02));
        }
        assert!(
            det.detect().is_empty(),
            "stable sequence should produce no cuts"
        );
    }

    #[test]
    fn test_detector_detects_hard_cut() {
        let mut det = AdaptiveSceneDetector::default();
        for i in 0..10u64 {
            det.push_frame(FrameDiff::new(i, 0.04));
        }
        det.push_frame(FrameDiff::new(10, 0.95)); // obvious hard cut
        for i in 11..20u64 {
            det.push_frame(FrameDiff::new(i, 0.03));
        }
        let cuts = det.detect();
        assert_eq!(cuts.len(), 1);
        assert_eq!(cuts[0].frame_index, 10);
    }

    #[test]
    fn test_detector_reset_clears_state() {
        let mut det = AdaptiveSceneDetector::default();
        det.push_frame(FrameDiff::new(0, 0.9));
        det.reset();
        assert_eq!(det.frame_count(), 0);
        assert!(det.detect().is_empty());
    }

    #[test]
    fn test_configure_updates_thresholds() {
        let mut det = AdaptiveSceneDetector::default();
        let new_cfg = AdaptiveConfig {
            base_threshold: 0.1,
            min_threshold: 0.05,
            max_threshold: 0.2,
            ..AdaptiveConfig::default()
        };
        det.configure(new_cfg.clone())
            .expect("valid config should apply");
        assert_eq!(det.config().base_threshold, 0.1);
        assert_eq!(det.config().min_threshold, 0.05);
        assert_eq!(det.config().max_threshold, 0.2);
    }

    #[test]
    fn test_configure_with_invalid_config_returns_error() {
        let mut det = AdaptiveSceneDetector::default();
        let bad_cfg = AdaptiveConfig {
            min_threshold: 0.8,
            max_threshold: 0.2, // invalid: min > max
            ..AdaptiveConfig::default()
        };
        assert!(det.configure(bad_cfg).is_err());
    }

    #[test]
    fn test_high_complexity_window_raises_threshold() {
        // Force the window to contain high-complexity scores, then
        // verify that the adaptive threshold is closer to max_threshold.
        let cfg = AdaptiveConfig {
            min_threshold: 0.1,
            max_threshold: 0.6,
            window_size: 5,
            ..AdaptiveConfig::default()
        };
        let mut det = AdaptiveSceneDetector::new(cfg);

        // Push frames with high complexity (score ≈ 1.0)
        let high_score = ComplexityScore {
            entropy: 1.0,
            variance: 1.0,
            spread: 1.0,
            overall: 1.0,
        };
        for i in 0..5u64 {
            det.push_frame(FrameDiff::with_complexity(i, 0.01, high_score));
        }

        let threshold = det.current_threshold();
        assert!(
            threshold > 0.4,
            "high-complexity window should push threshold toward max (0.6), got {threshold}"
        );
    }

    #[test]
    fn test_low_complexity_window_lowers_threshold() {
        let cfg = AdaptiveConfig {
            min_threshold: 0.1,
            max_threshold: 0.6,
            window_size: 5,
            ..AdaptiveConfig::default()
        };
        let mut det = AdaptiveSceneDetector::new(cfg);

        // Push frames with near-zero complexity
        let low_score = ComplexityScore {
            entropy: 0.0,
            variance: 0.0,
            spread: 0.0,
            overall: 0.0,
        };
        for i in 0..5u64 {
            det.push_frame(FrameDiff::with_complexity(i, 0.01, low_score));
        }

        let threshold = det.current_threshold();
        assert!(
            threshold < 0.2,
            "low-complexity window should keep threshold near min (0.1), got {threshold}"
        );
    }

    #[test]
    fn test_detect_with_pixels_finds_cut_in_pixel_data() {
        let cfg = AdaptiveConfig {
            base_threshold: 0.35,
            min_threshold: 0.15,
            max_threshold: 0.55,
            window_size: 10,
            ..AdaptiveConfig::default()
        };
        let mut det = AdaptiveSceneDetector::new(cfg);

        // Build frame data: small diffs then a large jump
        let n_pixels = 512;
        let uniform = uniform_pixels(n_pixels);
        let constant = constant_pixels(0, n_pixels);

        let frame_data: Vec<(u64, f32, &[u8])> = vec![
            (0, 0.02, &uniform),
            (1, 0.03, &uniform),
            (2, 0.02, &uniform),
            (3, 0.90, &constant), // hard cut
            (4, 0.02, &constant),
        ];

        let cuts = det
            .detect_with_pixels(&frame_data)
            .expect("detection should succeed");
        assert!(!cuts.is_empty(), "should detect at least one cut");
        assert!(
            cuts.iter().any(|c| c.frame_index == 3),
            "cut should be at frame 3"
        );
    }

    #[test]
    fn test_multiple_cuts_detected_in_order() {
        let mut det = AdaptiveSceneDetector::default();
        // Interleave stable frames with big jumps
        for i in 0..30u64 {
            let diff = if i == 5 || i == 15 || i == 25 {
                0.95
            } else {
                0.02
            };
            det.push_frame(FrameDiff::new(i, diff));
        }
        let cuts = det.detect();
        assert_eq!(cuts.len(), 3);
        assert_eq!(cuts[0].frame_index, 5);
        assert_eq!(cuts[1].frame_index, 15);
        assert_eq!(cuts[2].frame_index, 25);
    }

    #[test]
    fn test_sliding_window_respects_window_size() {
        let cfg = AdaptiveConfig {
            window_size: 3,
            ..AdaptiveConfig::default()
        };
        let mut det = AdaptiveSceneDetector::new(cfg);

        // Flood the window with high-complexity scores
        let high = ComplexityScore {
            entropy: 1.0,
            variance: 1.0,
            spread: 1.0,
            overall: 1.0,
        };
        for i in 0..10u64 {
            det.push_frame(FrameDiff::with_complexity(i, 0.01, high));
        }
        // Internal window should be capped at window_size=3
        assert_eq!(det.complexity_window.len(), 3);
    }

    #[test]
    fn test_threshold_used_is_stored_in_cut() {
        let mut det = AdaptiveSceneDetector::default();
        det.push_frame(FrameDiff::new(0, 0.95));
        let cuts = det.detect();
        assert!(!cuts.is_empty());
        // threshold_used should be base_threshold since window was empty
        assert!(
            (cuts[0].threshold_used - det.config().base_threshold).abs() < 1e-6,
            "threshold_used should equal base_threshold when window is empty, got {}",
            cuts[0].threshold_used
        );
    }

    // ── RollingSceneDetector tests ────────────────────────────────────────────

    #[test]
    fn test_rolling_empty_history_returns_default_threshold() {
        let det = RollingSceneDetector::new(30, 2.0, 5);
        // With no history, adaptive_threshold must return 0.3 (the fallback).
        let t = det.adaptive_threshold();
        assert!(
            (t - 0.3).abs() < 1e-9,
            "empty history should return 0.3, got {t}"
        );
    }

    #[test]
    fn test_rolling_below_threshold_returns_none() {
        // With variable but bounded diffs (range 0.05..0.15), the adaptive
        // threshold (mean + 2*stddev) should be above ~0.17. Pushing a value
        // within the same range should not trigger a cut.
        let mut det = RollingSceneDetector::new(10, 2.0, 1);
        // Fill the window with values in [0.05, 0.15] so stddev > 0
        for i in 0..10u64 {
            det.push_frame_diff(0.05 + (i as f64) * 0.01);
        }
        // Push another in-range value — should be below threshold
        assert!(
            det.push_frame_diff(0.10).is_none(),
            "value within range should not trigger a cut"
        );
    }

    #[test]
    fn test_rolling_hard_cut_detected() {
        let mut det = RollingSceneDetector::new(10, 2.0, 1);
        // Seed window with very small, stable diffs
        for _ in 0..10 {
            det.push_frame_diff(0.02);
        }
        // Large jump → scene cut
        let cut = det.push_frame_diff(0.90);
        assert!(cut.is_some(), "large diff should trigger a cut");
        let c = cut.expect("large diff should produce a cut result");
        assert_eq!(c.frame_idx, 10, "cut should be at frame 10 (0-indexed)");
    }

    #[test]
    fn test_rolling_confidence_in_unit_range() {
        let mut det = RollingSceneDetector::new(10, 2.0, 1);
        for _ in 0..10 {
            det.push_frame_diff(0.02);
        }
        let cut = det.push_frame_diff(0.90).expect("should detect cut");
        assert!(
            (0.0..=1.0).contains(&cut.confidence),
            "confidence must be in [0,1], got {}",
            cut.confidence
        );
    }

    #[test]
    fn test_rolling_min_scene_duration_suppresses_rapid_cuts() {
        let mut det = RollingSceneDetector::new(30, 0.5, 10); // min 10 frames between cuts
                                                              // Seed window
        for _ in 0..30 {
            det.push_frame_diff(0.05);
        }
        // First cut
        let cut1 = det.push_frame_diff(0.95);
        assert!(cut1.is_some(), "first cut should be detected");
        // Immediately another large diff — should be suppressed
        let cut2 = det.push_frame_diff(0.95);
        assert!(
            cut2.is_none(),
            "cut within min_scene_duration should be suppressed"
        );
    }

    #[test]
    fn test_rolling_min_scene_duration_allows_cut_after_cooldown() {
        let mut det = RollingSceneDetector::new(30, 0.5, 5);
        for _ in 0..30 {
            det.push_frame_diff(0.05);
        }
        let cut1 = det.push_frame_diff(0.95);
        assert!(cut1.is_some());
        // Feed 5 quiet frames to satisfy min_scene_duration
        for _ in 0..5 {
            det.push_frame_diff(0.05);
        }
        // Now another large diff should be allowed
        let cut2 = det.push_frame_diff(0.95);
        assert!(cut2.is_some(), "cut after cooldown should be detected");
    }

    #[test]
    fn test_rolling_flash_suppression_marks_is_flash() {
        // A very large spike (≥ 2× threshold) against a quiet background → flash
        let mut det = RollingSceneDetector::new(20, 2.0, 1);
        // Seed with very low, stable diffs so mean << threshold
        for _ in 0..20 {
            det.push_frame_diff(0.01);
        }
        // Extreme spike ≥ 2× threshold
        let threshold = det.adaptive_threshold();
        let spike = threshold * 2.5;
        let cut = det.push_frame_diff(spike).expect("should detect cut");
        assert!(
            cut.is_flash,
            "extreme spike against quiet baseline should be flagged as flash (threshold={threshold:.4}, spike={spike:.4})"
        );
    }

    #[test]
    fn test_rolling_normal_cut_is_not_flash() {
        let mut det = RollingSceneDetector::new(20, 1.5, 1);
        // Fill with somewhat variable diffs to push up the mean
        for i in 0..20 {
            det.push_frame_diff(0.1 + (i as f64) * 0.005);
        }
        // A cut just above threshold but < 2× threshold
        let threshold = det.adaptive_threshold();
        let diff = threshold * 1.5; // 1.5× — above threshold, below 2×
        let cut = det.push_frame_diff(diff).expect("should detect cut");
        assert!(
            !cut.is_flash,
            "moderate cut above baseline should not be flagged as flash (threshold={threshold:.4}, diff={diff:.4})"
        );
    }

    #[test]
    fn test_rolling_mean_stddev_threshold_calculation() {
        let mut det = RollingSceneDetector::new(4, 1.0, 1);
        // Push exactly [0.1, 0.2, 0.3, 0.4]
        det.push_frame_diff(0.1);
        det.push_frame_diff(0.2);
        det.push_frame_diff(0.3);
        det.push_frame_diff(0.4);
        // mean = 0.25
        // deviations: -0.15, -0.05, +0.05, +0.15
        // squared:    0.0225, 0.0025, 0.0025, 0.0225  → sum = 0.05
        // population variance = 0.05 / 4 = 0.0125
        // stddev = sqrt(0.0125) ≈ 0.111803...
        // threshold = 0.25 + 1.0 * 0.111803... ≈ 0.361803...
        let t = det.adaptive_threshold();
        let expected = 0.25_f64 + (0.0125_f64).sqrt();
        assert!(
            (t - expected).abs() < 1e-9,
            "expected threshold {expected:.9}, got {t:.9}"
        );
    }

    #[test]
    fn test_rolling_window_eviction_respects_window_size() {
        let mut det = RollingSceneDetector::new(5, 2.0, 1);
        for _ in 0..20 {
            det.push_frame_diff(0.1);
        }
        assert_eq!(
            det.history.len(),
            5,
            "history deque should be capped at window_size"
        );
    }

    #[test]
    fn test_rolling_reset_clears_state() {
        let mut det = RollingSceneDetector::new(10, 2.0, 1);
        for _ in 0..10 {
            det.push_frame_diff(0.05);
        }
        det.push_frame_diff(0.90);
        det.reset();
        assert_eq!(det.frame_count(), 0);
        assert!(det.history.is_empty());
        assert!(
            (det.adaptive_threshold() - 0.3).abs() < 1e-9,
            "threshold should revert to 0.3 after reset"
        );
    }

    #[test]
    fn test_rolling_frame_idx_monotonically_increases() {
        let mut det = RollingSceneDetector::new(10, 2.0, 1);
        for i in 0..5u64 {
            det.push_frame_diff(0.05);
            assert_eq!(det.frame_count(), i + 1);
        }
    }
}
