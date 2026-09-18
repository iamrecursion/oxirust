//! Automatic thumbnail selection from video frames.
//!
//! Selects the best frames for use as video thumbnails based on multiple
//! quality and aesthetic criteria:
//!
//! - **Sharpness**: Reject blurry frames (Laplacian variance)
//! - **Brightness**: Prefer well-exposed frames, reject too dark/bright
//! - **Contrast**: Favor frames with good tonal range
//! - **Face presence**: Boost frames containing faces
//! - **Color diversity**: Prefer visually interesting frames
//! - **Rule of thirds**: Reward compositionally balanced images
//! - **Motion blur**: Penalize frames captured during fast motion
//! - **Temporal spread**: Prefer thumbnails from different parts of the video
//!
//! # Example
//!
//! ```
//! use oximedia_auto::auto_thumbnail::{ThumbnailSelector, ThumbnailConfig};
//!
//! let config = ThumbnailConfig::default();
//! let selector = ThumbnailSelector::new(config);
//! ```

#![allow(dead_code)]

use crate::error::{AutoError, AutoResult};
use crate::scoring::SceneFeatures;
use oximedia_core::Timestamp;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Weights controlling how much each quality metric contributes to the
/// overall thumbnail score.
#[derive(Debug, Clone)]
pub struct ThumbnailWeights {
    /// Weight for sharpness score.
    pub sharpness: f64,
    /// Weight for brightness score (penalizes too dark / too bright).
    pub brightness: f64,
    /// Weight for contrast score.
    pub contrast: f64,
    /// Weight for face presence score.
    pub face_presence: f64,
    /// Weight for color diversity score.
    pub color_diversity: f64,
    /// Weight for rule-of-thirds composition score.
    pub composition: f64,
    /// Weight for motion blur penalty (higher = more penalty).
    pub motion_blur_penalty: f64,
}

impl Default for ThumbnailWeights {
    fn default() -> Self {
        Self {
            sharpness: 1.5,
            brightness: 1.0,
            contrast: 1.0,
            face_presence: 1.3,
            color_diversity: 0.8,
            composition: 0.7,
            motion_blur_penalty: 1.2,
        }
    }
}

/// Configuration for the thumbnail selector.
#[derive(Debug, Clone)]
pub struct ThumbnailConfig {
    /// Number of candidate thumbnails to return.
    pub num_candidates: usize,
    /// Minimum sharpness score (0.0-1.0) to be considered.
    pub min_sharpness: f64,
    /// Acceptable brightness range [min, max] in 0.0-1.0.
    pub brightness_range: (f64, f64),
    /// Minimum temporal distance (ms) between selected thumbnails.
    pub min_temporal_gap_ms: i64,
    /// Scoring weights.
    pub weights: ThumbnailWeights,
    /// Skip the first N percent of the video (often intro/slate).
    pub skip_start_pct: f64,
    /// Skip the last N percent of the video (often credits).
    pub skip_end_pct: f64,
    /// Sampling interval: evaluate every Nth frame for efficiency.
    pub sample_every_n: usize,
}

impl Default for ThumbnailConfig {
    fn default() -> Self {
        Self {
            num_candidates: 5,
            min_sharpness: 0.15,
            brightness_range: (0.20, 0.85),
            min_temporal_gap_ms: 5_000,
            weights: ThumbnailWeights::default(),
            skip_start_pct: 0.05,
            skip_end_pct: 0.05,
            sample_every_n: 1,
        }
    }
}

impl ThumbnailConfig {
    /// Create a new default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of candidate thumbnails.
    #[must_use]
    pub const fn with_num_candidates(mut self, n: usize) -> Self {
        self.num_candidates = n;
        self
    }

    /// Set the minimum sharpness threshold.
    #[must_use]
    pub fn with_min_sharpness(mut self, s: f64) -> Self {
        self.min_sharpness = s.clamp(0.0, 1.0);
        self
    }

    /// Set the sampling interval.
    #[must_use]
    pub const fn with_sample_every_n(mut self, n: usize) -> Self {
        self.sample_every_n = n;
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> AutoResult<()> {
        if self.num_candidates == 0 {
            return Err(AutoError::invalid_parameter(
                "num_candidates",
                "must be at least 1",
            ));
        }
        if !(0.0..=1.0).contains(&self.min_sharpness) {
            return Err(AutoError::InvalidThreshold {
                threshold: self.min_sharpness,
                min: 0.0,
                max: 1.0,
            });
        }
        if self.brightness_range.0 >= self.brightness_range.1 {
            return Err(AutoError::invalid_parameter(
                "brightness_range",
                "min must be less than max",
            ));
        }
        if self.sample_every_n == 0 {
            return Err(AutoError::invalid_parameter(
                "sample_every_n",
                "must be at least 1",
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Frame analysis
// ---------------------------------------------------------------------------

/// Per-frame quality metrics used for thumbnail scoring.
#[derive(Debug, Clone)]
pub struct FrameQuality {
    /// Frame index within the video.
    pub frame_index: usize,
    /// Presentation timestamp.
    pub timestamp: Timestamp,
    /// Laplacian-variance sharpness (0.0-1.0).
    pub sharpness: f64,
    /// Mean brightness (0.0-1.0).
    pub brightness: f64,
    /// Contrast (standard deviation of luminance, 0.0-1.0).
    pub contrast: f64,
    /// Number of faces detected.
    pub face_count: usize,
    /// Total face coverage area (0.0-1.0).
    pub face_coverage: f64,
    /// Color diversity / saturation spread (0.0-1.0).
    pub color_diversity: f64,
    /// Estimated motion blur amount (0.0-1.0, higher = more blur).
    pub motion_blur: f64,
    /// Composition score (rule of thirds, 0.0-1.0).
    pub composition: f64,
}

impl FrameQuality {
    /// Create a new frame quality record.
    #[must_use]
    pub fn new(frame_index: usize, timestamp: Timestamp) -> Self {
        Self {
            frame_index,
            timestamp,
            sharpness: 0.0,
            brightness: 0.5,
            contrast: 0.5,
            face_count: 0,
            face_coverage: 0.0,
            color_diversity: 0.5,
            motion_blur: 0.0,
            composition: 0.5,
        }
    }

    /// Build a `FrameQuality` from existing `SceneFeatures`.
    #[must_use]
    pub fn from_scene_features(
        frame_index: usize,
        timestamp: Timestamp,
        features: &SceneFeatures,
    ) -> Self {
        Self {
            frame_index,
            timestamp,
            sharpness: features.sharpness,
            brightness: features.brightness_mean,
            contrast: features.contrast,
            face_count: features.face_count,
            face_coverage: features.face_coverage,
            color_diversity: features.color_diversity,
            motion_blur: features.motion_intensity.min(1.0),
            composition: features.edge_density, // proxy
        }
    }
}

// ---------------------------------------------------------------------------
// Thumbnail candidate
// ---------------------------------------------------------------------------

/// A scored thumbnail candidate.
#[derive(Debug, Clone)]
pub struct ThumbnailCandidate {
    /// Frame index.
    pub frame_index: usize,
    /// Presentation timestamp.
    pub timestamp: Timestamp,
    /// Overall composite thumbnail score (higher is better).
    pub score: f64,
    /// Underlying quality metrics.
    pub quality: FrameQuality,
    /// Human-readable reason this frame was selected.
    pub reason: String,
}

impl ThumbnailCandidate {
    /// Duration from the start of the video (ms).
    #[must_use]
    pub fn offset_ms(&self) -> i64 {
        self.timestamp.pts
    }
}

// ---------------------------------------------------------------------------
// Scoring helpers
// ---------------------------------------------------------------------------

/// Score a single frame for thumbnail quality.
fn score_frame(quality: &FrameQuality, weights: &ThumbnailWeights) -> f64 {
    let mut score = 0.0;
    let mut total_weight = 0.0;

    // Sharpness
    score += quality.sharpness * weights.sharpness;
    total_weight += weights.sharpness;

    // Brightness: penalize both extremes; ideal around 0.45-0.65
    let brightness_score = 1.0 - (quality.brightness - 0.55).abs() * 2.5;
    score += brightness_score.clamp(0.0, 1.0) * weights.brightness;
    total_weight += weights.brightness;

    // Contrast
    score += quality.contrast * weights.contrast;
    total_weight += weights.contrast;

    // Face presence
    let face_score = if quality.face_count > 0 {
        (quality.face_coverage * 2.0).min(1.0)
    } else {
        0.0
    };
    score += face_score * weights.face_presence;
    total_weight += weights.face_presence;

    // Color diversity
    score += quality.color_diversity * weights.color_diversity;
    total_weight += weights.color_diversity;

    // Composition
    score += quality.composition * weights.composition;
    total_weight += weights.composition;

    // Motion blur penalty (subtractive)
    score -= quality.motion_blur * weights.motion_blur_penalty;
    total_weight += weights.motion_blur_penalty;

    if total_weight > 0.0 {
        (score / total_weight).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Build a human-readable reason string for the candidate.
fn build_reason(quality: &FrameQuality) -> String {
    let mut parts = Vec::new();

    if quality.sharpness > 0.6 {
        parts.push("sharp".to_string());
    }
    if quality.face_count > 0 {
        parts.push(format!(
            "{} face{}",
            quality.face_count,
            if quality.face_count == 1 { "" } else { "s" }
        ));
    }
    if quality.color_diversity > 0.5 {
        parts.push("colorful".to_string());
    }
    if quality.contrast > 0.5 {
        parts.push("good contrast".to_string());
    }

    if parts.is_empty() {
        "balanced quality".to_string()
    } else {
        parts.join(", ")
    }
}

// ---------------------------------------------------------------------------
// ThumbnailSelector
// ---------------------------------------------------------------------------

/// Selects the best frames for use as video thumbnails.
pub struct ThumbnailSelector {
    config: ThumbnailConfig,
}

impl ThumbnailSelector {
    /// Create a new selector with the given configuration.
    #[must_use]
    pub fn new(config: ThumbnailConfig) -> Self {
        Self { config }
    }

    /// Select the best thumbnail candidates from a list of frame quality
    /// records.
    ///
    /// The returned list is sorted by descending score and limited to
    /// `config.num_candidates` entries.
    ///
    /// # Errors
    ///
    /// Returns an error if configuration is invalid or no frames pass the
    /// quality filters.
    pub fn select(&self, frames: &[FrameQuality]) -> AutoResult<Vec<ThumbnailCandidate>> {
        self.config.validate()?;

        if frames.is_empty() {
            return Err(AutoError::insufficient_data(
                "No frames provided for thumbnail selection",
            ));
        }

        // Determine the usable temporal range
        let total_duration_ms = frames.last().map(|f| f.timestamp.pts).unwrap_or(0).max(1);
        let start_cutoff_ms = (total_duration_ms as f64 * self.config.skip_start_pct) as i64;
        let end_cutoff_ms =
            total_duration_ms - (total_duration_ms as f64 * self.config.skip_end_pct) as i64;

        // Score and filter all frames
        let mut scored: Vec<(f64, &FrameQuality)> = frames
            .iter()
            .step_by(self.config.sample_every_n.max(1))
            .filter(|fq| {
                // Temporal range filter
                fq.timestamp.pts >= start_cutoff_ms && fq.timestamp.pts <= end_cutoff_ms
            })
            .filter(|fq| {
                // Minimum quality filters
                fq.sharpness >= self.config.min_sharpness
                    && fq.brightness >= self.config.brightness_range.0
                    && fq.brightness <= self.config.brightness_range.1
            })
            .map(|fq| {
                let s = score_frame(fq, &self.config.weights);
                (s, fq)
            })
            .collect();

        if scored.is_empty() {
            return Err(AutoError::insufficient_data(
                "No frames passed quality filters for thumbnail selection",
            ));
        }

        // Sort by score descending
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Greedily select candidates enforcing minimum temporal gap
        let mut selected: Vec<ThumbnailCandidate> = Vec::new();

        for (score, fq) in &scored {
            if selected.len() >= self.config.num_candidates {
                break;
            }

            // Enforce temporal spread
            let too_close = selected.iter().any(|existing| {
                (existing.timestamp.pts - fq.timestamp.pts).abs() < self.config.min_temporal_gap_ms
            });
            if too_close {
                continue;
            }

            selected.push(ThumbnailCandidate {
                frame_index: fq.frame_index,
                timestamp: fq.timestamp,
                score: *score,
                quality: (*fq).clone(),
                reason: build_reason(fq),
            });
        }

        if selected.is_empty() {
            // Fall back: return the single best frame regardless of gap
            if let Some((score, fq)) = scored.first() {
                selected.push(ThumbnailCandidate {
                    frame_index: fq.frame_index,
                    timestamp: fq.timestamp,
                    score: *score,
                    quality: (*fq).clone(),
                    reason: build_reason(fq),
                });
            }
        }

        Ok(selected)
    }

    /// Convenience: select a single best thumbnail.
    ///
    /// # Errors
    ///
    /// Returns an error if selection fails.
    pub fn select_best(&self, frames: &[FrameQuality]) -> AutoResult<ThumbnailCandidate> {
        let mut cfg = self.config.clone();
        cfg.num_candidates = 1;
        let selector = Self::new(cfg);
        let mut candidates = selector.select(frames)?;
        candidates
            .pop()
            .ok_or_else(|| AutoError::insufficient_data("No thumbnail candidate found"))
    }

    /// Select thumbnails from `SceneFeatures` data with timestamps.
    ///
    /// # Errors
    ///
    /// Returns an error if selection fails.
    pub fn select_from_scene_features(
        &self,
        scene_data: &[(usize, Timestamp, SceneFeatures)],
    ) -> AutoResult<Vec<ThumbnailCandidate>> {
        let qualities: Vec<FrameQuality> = scene_data
            .iter()
            .map(|(idx, ts, feat)| FrameQuality::from_scene_features(*idx, *ts, feat))
            .collect();
        self.select(&qualities)
    }

    /// Get the current configuration.
    #[must_use]
    pub const fn config(&self) -> &ThumbnailConfig {
        &self.config
    }
}

impl Default for ThumbnailSelector {
    fn default() -> Self {
        Self::new(ThumbnailConfig::default())
    }
}

// ---------------------------------------------------------------------------
// Convenience functions
// ---------------------------------------------------------------------------

/// Analyze a grayscale frame buffer and return quality metrics.
///
/// `frame` is a row-major luma buffer of `width * height` bytes.
pub fn analyze_frame_quality(
    frame: &[u8],
    width: u32,
    height: u32,
    frame_index: usize,
    timestamp: Timestamp,
) -> FrameQuality {
    let n = (width as usize) * (height as usize);
    if frame.len() < n || n == 0 {
        return FrameQuality::new(frame_index, timestamp);
    }

    let pixels = &frame[..n];

    // Mean brightness
    let sum: u64 = pixels.iter().map(|&p| p as u64).sum();
    let brightness = sum as f64 / (n as f64 * 255.0);

    // Contrast (stddev of luminance)
    let mean = sum as f64 / n as f64;
    let variance: f64 = pixels
        .iter()
        .map(|&p| (p as f64 - mean).powi(2))
        .sum::<f64>()
        / n as f64;
    let contrast = (variance.sqrt() / 128.0).min(1.0);

    // Sharpness via Laplacian variance
    let sharpness = compute_laplacian_variance(pixels, width as usize, height as usize);

    FrameQuality {
        frame_index,
        timestamp,
        sharpness,
        brightness,
        contrast,
        face_count: 0,
        face_coverage: 0.0,
        color_diversity: contrast * 0.8, // rough proxy
        motion_blur: 0.0,
        composition: 0.5,
    }
}

/// Compute Laplacian variance as a sharpness metric (0.0-1.0).
fn compute_laplacian_variance(pixels: &[u8], w: usize, h: usize) -> f64 {
    if w < 3 || h < 3 {
        return 0.0;
    }

    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut count = 0usize;

    for y in 1..(h - 1) {
        for x in 1..(w - 1) {
            let idx = y * w + x;
            let c = pixels[idx] as f64;
            let n = pixels[(y - 1) * w + x] as f64;
            let s = pixels[(y + 1) * w + x] as f64;
            let west = pixels[y * w + (x - 1)] as f64;
            let e = pixels[y * w + (x + 1)] as f64;
            let lap = (n + s + west + e - 4.0 * c).abs();
            sum += lap;
            sum_sq += lap * lap;
            count += 1;
        }
    }

    if count == 0 {
        return 0.0;
    }

    let mean = sum / count as f64;
    let variance = (sum_sq / count as f64) - mean * mean;
    // Normalize: typical Laplacian variance for sharp images is 500-2000+
    (variance / 2000.0).clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_core::Rational;

    fn ts(ms: i64) -> Timestamp {
        Timestamp::new(ms, Rational::new(1, 1000))
    }

    fn make_quality(index: usize, ms: i64, sharpness: f64, brightness: f64) -> FrameQuality {
        FrameQuality {
            frame_index: index,
            timestamp: ts(ms),
            sharpness,
            brightness,
            contrast: 0.5,
            face_count: 0,
            face_coverage: 0.0,
            color_diversity: 0.5,
            motion_blur: 0.0,
            composition: 0.5,
        }
    }

    #[test]
    fn test_config_default_valid() {
        let cfg = ThumbnailConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_zero_candidates_invalid() {
        let cfg = ThumbnailConfig::default().with_num_candidates(0);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_builder() {
        let cfg = ThumbnailConfig::default()
            .with_num_candidates(3)
            .with_min_sharpness(0.3)
            .with_sample_every_n(2);
        assert_eq!(cfg.num_candidates, 3);
        assert!((cfg.min_sharpness - 0.3).abs() < 1e-9);
        assert_eq!(cfg.sample_every_n, 2);
    }

    #[test]
    fn test_select_empty_frames_error() {
        let selector = ThumbnailSelector::default();
        let result = selector.select(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_select_single_frame() {
        let frames = vec![make_quality(0, 5000, 0.5, 0.5)];
        let mut cfg = ThumbnailConfig::default()
            .with_num_candidates(1)
            .with_min_sharpness(0.1);
        cfg.skip_start_pct = 0.0;
        cfg.skip_end_pct = 0.0;
        let selector = ThumbnailSelector::new(cfg);
        let result = selector.select(&frames);
        assert!(result.is_ok());
        let candidates = result.expect("should succeed");
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].frame_index, 0);
    }

    #[test]
    fn test_select_prefers_sharp_frames() {
        let frames = vec![
            make_quality(0, 5000, 0.2, 0.5),
            make_quality(1, 15000, 0.9, 0.5),
            make_quality(2, 25000, 0.3, 0.5),
        ];
        let mut cfg = ThumbnailConfig::default()
            .with_num_candidates(1)
            .with_min_sharpness(0.1);
        cfg.skip_start_pct = 0.0;
        cfg.skip_end_pct = 0.0;
        let selector = ThumbnailSelector::new(cfg);
        let candidates = selector.select(&frames).expect("should succeed");
        assert_eq!(candidates[0].frame_index, 1, "sharpest frame should win");
    }

    #[test]
    fn test_select_rejects_too_dark() {
        let frames = vec![
            make_quality(0, 5000, 0.5, 0.05), // too dark
            make_quality(1, 15000, 0.5, 0.5), // good
        ];
        let mut cfg = ThumbnailConfig::default()
            .with_num_candidates(2)
            .with_min_sharpness(0.1);
        cfg.skip_start_pct = 0.0;
        cfg.skip_end_pct = 0.0;
        let selector = ThumbnailSelector::new(cfg);
        let candidates = selector.select(&frames).expect("should succeed");
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].frame_index, 1);
    }

    #[test]
    fn test_select_rejects_too_bright() {
        let frames = vec![
            make_quality(0, 5000, 0.5, 0.95), // too bright
            make_quality(1, 15000, 0.5, 0.5), // good
        ];
        let mut cfg = ThumbnailConfig::default()
            .with_num_candidates(2)
            .with_min_sharpness(0.1);
        cfg.skip_start_pct = 0.0;
        cfg.skip_end_pct = 0.0;
        let selector = ThumbnailSelector::new(cfg);
        let candidates = selector.select(&frames).expect("should succeed");
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].frame_index, 1);
    }

    #[test]
    fn test_select_enforces_temporal_gap() {
        let frames = vec![
            make_quality(0, 5000, 0.8, 0.5),
            make_quality(1, 6000, 0.9, 0.5),  // 1s gap - too close
            make_quality(2, 15000, 0.7, 0.5), // 10s gap - ok
        ];
        let mut cfg = ThumbnailConfig::default()
            .with_num_candidates(3)
            .with_min_sharpness(0.1);
        cfg.min_temporal_gap_ms = 5000;
        let selector = ThumbnailSelector::new(cfg);
        let candidates = selector.select(&frames).expect("should succeed");
        // Should pick at most 2 (frames 1 and 2 are >5s apart; frame 0 is too close to 1)
        assert!(candidates.len() <= 2);
    }

    #[test]
    fn test_select_skips_intro_outro() {
        // Video is 100s long; skip first/last 5%
        let frames: Vec<FrameQuality> = (0..100)
            .map(|i| make_quality(i, i as i64 * 1000, 0.5, 0.5))
            .collect();
        let mut cfg = ThumbnailConfig::default()
            .with_num_candidates(1)
            .with_min_sharpness(0.1);
        cfg.skip_start_pct = 0.10;
        cfg.skip_end_pct = 0.10;
        cfg.min_temporal_gap_ms = 0;
        let selector = ThumbnailSelector::new(cfg);
        let candidates = selector.select(&frames).expect("should succeed");
        let selected_ms = candidates[0].timestamp.pts;
        assert!(selected_ms >= 10_000, "should skip intro: ts={selected_ms}");
        assert!(selected_ms <= 90_000, "should skip outro: ts={selected_ms}");
    }

    #[test]
    fn test_select_best_returns_one() {
        let frames = vec![
            make_quality(0, 5000, 0.3, 0.5),
            make_quality(1, 15000, 0.8, 0.5),
        ];
        let mut cfg = ThumbnailConfig::default().with_min_sharpness(0.1);
        cfg.skip_start_pct = 0.0;
        cfg.skip_end_pct = 0.0;
        let selector = ThumbnailSelector::new(cfg);
        let best = selector.select_best(&frames).expect("should succeed");
        assert_eq!(best.frame_index, 1);
    }

    #[test]
    fn test_score_frame_motion_blur_penalty() {
        let mut fq = make_quality(0, 1000, 0.5, 0.5);
        let weights = ThumbnailWeights::default();
        let clean_score = score_frame(&fq, &weights);
        fq.motion_blur = 0.9;
        let blurry_score = score_frame(&fq, &weights);
        assert!(
            blurry_score < clean_score,
            "motion blur should reduce score: clean={clean_score} blurry={blurry_score}"
        );
    }

    #[test]
    fn test_score_frame_face_boost() {
        let mut fq = make_quality(0, 1000, 0.5, 0.5);
        let weights = ThumbnailWeights::default();
        let no_face_score = score_frame(&fq, &weights);
        fq.face_count = 1;
        fq.face_coverage = 0.3;
        let face_score = score_frame(&fq, &weights);
        assert!(
            face_score > no_face_score,
            "face should boost score: no_face={no_face_score} face={face_score}"
        );
    }

    #[test]
    fn test_analyze_frame_quality_uniform() {
        let frame = vec![128u8; 100 * 100];
        let fq = analyze_frame_quality(&frame, 100, 100, 0, ts(0));
        assert!((fq.brightness - 128.0 / 255.0).abs() < 0.01);
        // Uniform → low sharpness
        assert!(fq.sharpness < 0.1);
    }

    #[test]
    fn test_analyze_frame_quality_edge_pattern() {
        // Use a pattern with gradual edges rather than extreme checkerboard
        // so that Laplacian variance is clearly positive
        let mut frame = vec![0u8; 50 * 50];
        for y in 0..50usize {
            for x in 0..50usize {
                // Create alternating vertical stripes of width 5
                frame[y * 50 + x] = if (x / 5) % 2 == 0 { 200 } else { 50 };
            }
        }
        let fq = analyze_frame_quality(&frame, 50, 50, 0, ts(0));
        // Stripe edges produce Laplacian response → sharpness > 0
        assert!(
            fq.sharpness > 0.0,
            "sharpness should be positive: {}",
            fq.sharpness
        );
    }

    #[test]
    fn test_analyze_frame_quality_empty_buffer() {
        let fq = analyze_frame_quality(&[], 10, 10, 0, ts(0));
        assert!((fq.brightness - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_from_scene_features() {
        let mut feat = SceneFeatures::default();
        feat.sharpness = 0.7;
        feat.brightness_mean = 0.6;
        feat.contrast = 0.5;
        feat.face_count = 2;
        feat.face_coverage = 0.3;
        let fq = FrameQuality::from_scene_features(42, ts(10_000), &feat);
        assert_eq!(fq.frame_index, 42);
        assert!((fq.sharpness - 0.7).abs() < 1e-9);
        assert_eq!(fq.face_count, 2);
    }

    #[test]
    fn test_select_from_scene_features() {
        let data: Vec<(usize, Timestamp, SceneFeatures)> = (0..5)
            .map(|i| {
                let mut f = SceneFeatures::default();
                f.sharpness = 0.3 + i as f64 * 0.1;
                f.brightness_mean = 0.5;
                f.contrast = 0.5;
                (i, ts(i as i64 * 10_000), f)
            })
            .collect();
        let cfg = ThumbnailConfig::default()
            .with_num_candidates(2)
            .with_min_sharpness(0.1);
        let selector = ThumbnailSelector::new(cfg);
        let candidates = selector
            .select_from_scene_features(&data)
            .expect("should succeed");
        assert!(!candidates.is_empty());
        assert!(candidates.len() <= 2);
    }

    #[test]
    fn test_build_reason_sharp_face() {
        let mut fq = make_quality(0, 0, 0.8, 0.5);
        fq.face_count = 2;
        let reason = build_reason(&fq);
        assert!(reason.contains("sharp"));
        assert!(reason.contains("2 faces"));
    }

    #[test]
    fn test_build_reason_balanced() {
        let fq = make_quality(0, 0, 0.2, 0.5);
        let reason = build_reason(&fq);
        assert_eq!(reason, "balanced quality");
    }

    #[test]
    fn test_thumbnail_candidate_offset_ms() {
        let c = ThumbnailCandidate {
            frame_index: 0,
            timestamp: ts(42_000),
            score: 0.8,
            quality: make_quality(0, 42_000, 0.5, 0.5),
            reason: String::new(),
        };
        assert_eq!(c.offset_ms(), 42_000);
    }

    #[test]
    fn test_laplacian_variance_tiny_image() {
        let pixels = vec![0u8; 4]; // 2x2
        let v = compute_laplacian_variance(&pixels, 2, 2);
        assert!((v - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_all_frames_filtered_fallback() {
        // All frames below sharpness threshold except one
        let frames = vec![
            make_quality(0, 5000, 0.01, 0.5),
            make_quality(1, 15000, 0.01, 0.5),
        ];
        let cfg = ThumbnailConfig::default()
            .with_num_candidates(3)
            .with_min_sharpness(0.9);
        let selector = ThumbnailSelector::new(cfg);
        let result = selector.select(&frames);
        assert!(
            result.is_err(),
            "should fail when all frames are too blurry"
        );
    }
}
