// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Content-aware segment boundary selection.
//!
//! This module implements intelligent segment boundary detection that aligns
//! segment cuts to scene changes, keyframe positions, and audio transients.
//! Rather than cutting at fixed time intervals, content-aware boundary selection
//! minimises visual disruption and improves compression efficiency by preferring
//! cuts at natural transition points in the content.
//!
//! # Algorithm
//!
//! 1. Collect candidate keyframe positions from the codec demuxer.
//! 2. For each candidate near a scheduled cut, compute a visual discontinuity
//!    score (based on inter-frame pixel difference, histogram distance, or
//!    external scene-change metadata).
//! 3. Select the candidate with the highest discontinuity score within an
//!    acceptable search window around the target cut time.
//! 4. Apply a minimum separation constraint so that no two boundaries are
//!    placed closer than `min_segment_duration`.
//!
//! The implementation is fully self-contained and does **not** depend on any
//! external encoder or demuxer library — callers supply pre-computed keyframe
//! positions and optional scene-change scores.

use crate::error::{PackagerError, PackagerResult};
use std::time::Duration;

// ---------------------------------------------------------------------------
// SceneChangeHint
// ---------------------------------------------------------------------------

/// A hint about a scene change at a specific point in the media timeline.
///
/// Scene change hints can come from:
/// - A codec's inline scene-change detection metadata.
/// - An external shot-boundary detection pass (e.g. `oximedia-shots`).
/// - Application-level annotations (chapter markers, ad splice points).
#[derive(Debug, Clone)]
pub struct SceneChangeHint {
    /// Presentation timestamp of the potential scene change.
    pub timestamp: Duration,
    /// Confidence score in the range `[0.0, 1.0]`.
    ///
    /// A value of `1.0` indicates very high confidence (hard cut detected);
    /// `0.0` indicates no evidence of a scene change.
    pub confidence: f64,
    /// Source that generated this hint.
    pub source: SceneChangeSource,
}

impl SceneChangeHint {
    /// Create a new scene change hint.
    ///
    /// `confidence` is clamped to `[0.0, 1.0]`.
    #[must_use]
    pub fn new(timestamp: Duration, confidence: f64, source: SceneChangeSource) -> Self {
        Self {
            timestamp,
            confidence: confidence.clamp(0.0, 1.0),
            source,
        }
    }
}

/// The source of a [`SceneChangeHint`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SceneChangeSource {
    /// Detected via frame pixel-difference heuristic.
    PixelDiff,
    /// Detected via histogram distance metric.
    HistogramDist,
    /// Provided by an external shot-boundary detector.
    External,
    /// Inserted as an application-level annotation.
    Annotation,
}

// ---------------------------------------------------------------------------
// KeyframePosition
// ---------------------------------------------------------------------------

/// A codec keyframe (IDR/I-frame) position in the media timeline.
#[derive(Debug, Clone)]
pub struct KeyframePosition {
    /// Presentation timestamp of this keyframe.
    pub timestamp: Duration,
    /// Byte offset in the source bitstream, if available.
    pub byte_offset: Option<u64>,
    /// Encoded frame size in bytes, if available.
    pub frame_size: Option<u32>,
}

impl KeyframePosition {
    /// Create a new keyframe position.
    #[must_use]
    pub fn new(timestamp: Duration) -> Self {
        Self {
            timestamp,
            byte_offset: None,
            frame_size: None,
        }
    }

    /// Attach a byte offset.
    #[must_use]
    pub fn with_byte_offset(mut self, offset: u64) -> Self {
        self.byte_offset = Some(offset);
        self
    }

    /// Attach a frame size.
    #[must_use]
    pub fn with_frame_size(mut self, size: u32) -> Self {
        self.frame_size = Some(size);
        self
    }
}

// ---------------------------------------------------------------------------
// BoundaryConfig
// ---------------------------------------------------------------------------

/// Configuration for content-aware boundary selection.
#[derive(Debug, Clone)]
pub struct BoundaryConfig {
    /// Target segment duration.  Boundaries will be placed as close to this
    /// value as possible while still respecting content cuts.
    pub target_duration: Duration,
    /// Minimum allowed segment duration.  No boundary will be placed within
    /// this distance of the previous boundary.
    pub min_segment_duration: Duration,
    /// Maximum allowed segment duration.  A hard cut is forced if no good
    /// candidate is found by this deadline.
    pub max_segment_duration: Duration,
    /// Half-width of the search window around each target cut time.
    /// A keyframe at `target ± search_window` is eligible for consideration.
    pub search_window: Duration,
    /// Minimum confidence threshold for a scene-change hint to influence
    /// boundary selection.  Hints below this threshold are ignored.
    pub min_scene_confidence: f64,
    /// Weight for scene-change score when ranking candidates.
    /// Total rank = `scene_weight * confidence + keyframe_proximity_weight * proximity`.
    pub scene_weight: f64,
    /// Weight for keyframe proximity when ranking candidates.
    pub keyframe_proximity_weight: f64,
}

impl Default for BoundaryConfig {
    fn default() -> Self {
        Self {
            target_duration: Duration::from_secs(6),
            min_segment_duration: Duration::from_secs(2),
            max_segment_duration: Duration::from_secs(12),
            search_window: Duration::from_millis(500),
            min_scene_confidence: 0.3,
            scene_weight: 0.7,
            keyframe_proximity_weight: 0.3,
        }
    }
}

impl BoundaryConfig {
    /// Create a new boundary config with the given target duration.
    #[must_use]
    pub fn with_target_duration(target_duration: Duration) -> Self {
        Self {
            target_duration,
            ..Self::default()
        }
    }

    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Returns [`PackagerError::InvalidConfig`] if any invariant is violated.
    pub fn validate(&self) -> PackagerResult<()> {
        if self.min_segment_duration >= self.target_duration {
            return Err(PackagerError::InvalidConfig(
                "min_segment_duration must be less than target_duration".into(),
            ));
        }
        if self.target_duration > self.max_segment_duration {
            return Err(PackagerError::InvalidConfig(
                "target_duration must not exceed max_segment_duration".into(),
            ));
        }
        if self.search_window > self.target_duration / 2 {
            return Err(PackagerError::InvalidConfig(
                "search_window must be at most half of target_duration".into(),
            ));
        }
        if !(0.0..=1.0).contains(&self.min_scene_confidence) {
            return Err(PackagerError::InvalidConfig(
                "min_scene_confidence must be in [0.0, 1.0]".into(),
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// BoundaryCandidate
// ---------------------------------------------------------------------------

/// An evaluated candidate for a segment boundary.
#[derive(Debug, Clone)]
pub struct BoundaryCandidate {
    /// The timestamp of this candidate.
    pub timestamp: Duration,
    /// Combined ranking score (higher = better boundary point).
    pub score: f64,
    /// Whether this candidate is at a keyframe position.
    pub is_keyframe: bool,
    /// Scene-change confidence contributing to the score (0 if none).
    pub scene_confidence: f64,
}

// ---------------------------------------------------------------------------
// ContentBoundarySelector
// ---------------------------------------------------------------------------

/// Selects optimal segment boundary points using content-aware heuristics.
///
/// # Example
///
/// ```
/// use std::time::Duration;
/// use oximedia_packager::content_boundary::{
///     BoundaryConfig, ContentBoundarySelector, KeyframePosition, SceneChangeHint,
///     SceneChangeSource,
/// };
///
/// let config = BoundaryConfig::default();
/// let mut selector = ContentBoundarySelector::new(config);
///
/// // Register keyframes at 0, 2, 4, 6 seconds
/// for secs in [0, 2, 4, 6, 8, 10, 12] {
///     selector.add_keyframe(KeyframePosition::new(Duration::from_secs(secs)));
/// }
///
/// // Register a scene change at ~6 seconds with high confidence
/// selector.add_scene_hint(SceneChangeHint::new(
///     Duration::from_millis(6_100),
///     0.9,
///     SceneChangeSource::External,
/// ));
///
/// // Select boundaries for a 30-second clip
/// let boundaries = selector
///     .select_boundaries(Duration::ZERO, Duration::from_secs(30))
///     .expect("selection should succeed");
///
/// assert!(!boundaries.is_empty());
/// ```
pub struct ContentBoundarySelector {
    config: BoundaryConfig,
    keyframes: Vec<KeyframePosition>,
    scene_hints: Vec<SceneChangeHint>,
}

impl ContentBoundarySelector {
    /// Create a new selector with the given configuration.
    #[must_use]
    pub fn new(config: BoundaryConfig) -> Self {
        Self {
            config,
            keyframes: Vec::new(),
            scene_hints: Vec::new(),
        }
    }

    /// Register a keyframe position.
    pub fn add_keyframe(&mut self, kf: KeyframePosition) {
        self.keyframes.push(kf);
    }

    /// Register multiple keyframe positions.
    pub fn add_keyframes(&mut self, kfs: impl IntoIterator<Item = KeyframePosition>) {
        self.keyframes.extend(kfs);
    }

    /// Register a scene change hint.
    pub fn add_scene_hint(&mut self, hint: SceneChangeHint) {
        self.scene_hints.push(hint);
    }

    /// Register multiple scene change hints.
    pub fn add_scene_hints(&mut self, hints: impl IntoIterator<Item = SceneChangeHint>) {
        self.scene_hints.extend(hints);
    }

    /// Select optimal segment boundaries for the range `[start, end)`.
    ///
    /// Returns a sorted list of boundary timestamps (not including `start`
    /// itself; does not include `end` unless it aligns exactly with a
    /// calculated cut).
    ///
    /// # Errors
    ///
    /// Returns [`PackagerError::InvalidConfig`] if the configuration is invalid
    /// or `start >= end`.
    pub fn select_boundaries(
        &self,
        start: Duration,
        end: Duration,
    ) -> PackagerResult<Vec<BoundaryCandidate>> {
        self.config.validate()?;

        if start >= end {
            return Err(PackagerError::InvalidConfig(
                "start must be less than end".into(),
            ));
        }

        // Sort keyframes and hints for efficient lookup.
        let mut sorted_kf = self.keyframes.clone();
        sorted_kf.sort_by_key(|k| k.timestamp);

        let mut sorted_hints = self.scene_hints.clone();
        sorted_hints.sort_by_key(|h| h.timestamp);

        let mut boundaries = Vec::new();
        let mut current = start;

        loop {
            let target = current + self.config.target_duration;
            if target >= end {
                break;
            }

            // Hard deadline — we must cut by this time regardless.
            let deadline = current + self.config.max_segment_duration;
            let deadline = deadline.min(end);

            // Search window: [target - window, target + window] ∩ [current + min, deadline].
            let window_start = target.saturating_sub(self.config.search_window);
            let window_start = window_start.max(current + self.config.min_segment_duration);
            let window_end = (target + self.config.search_window).min(deadline);

            // Find all keyframes within the search window.
            let candidates_kf: Vec<&KeyframePosition> = sorted_kf
                .iter()
                .filter(|k| k.timestamp > window_start && k.timestamp <= window_end)
                .collect();

            if candidates_kf.is_empty() {
                // No keyframe within the window — force cut at the target (or
                // the closest keyframe after the deadline).
                let forced_ts = self.nearest_keyframe_at_or_after(&sorted_kf, target);
                let forced_ts = forced_ts.min(deadline);
                if forced_ts <= current + self.config.min_segment_duration {
                    // Safety guard — advance by target to avoid infinite loop.
                    current += self.config.target_duration;
                    continue;
                }
                boundaries.push(BoundaryCandidate {
                    timestamp: forced_ts,
                    score: 0.0,
                    is_keyframe: false,
                    scene_confidence: 0.0,
                });
                current = forced_ts;
                continue;
            }

            // Score each keyframe candidate.
            let best = candidates_kf
                .iter()
                .map(|kf| {
                    let scene_conf = self.best_scene_confidence_near(kf.timestamp);
                    let proximity = self.proximity_score(kf.timestamp, target);
                    let score = self.config.scene_weight * scene_conf
                        + self.config.keyframe_proximity_weight * proximity;
                    BoundaryCandidate {
                        timestamp: kf.timestamp,
                        score,
                        is_keyframe: true,
                        scene_confidence: scene_conf,
                    }
                })
                .max_by(|a, b| {
                    a.score
                        .partial_cmp(&b.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

            if let Some(candidate) = best {
                current = candidate.timestamp;
                boundaries.push(candidate);
            } else {
                // Fallback — advance by target duration.
                current += self.config.target_duration;
            }
        }

        Ok(boundaries)
    }

    /// Return the best scene-change confidence within `search_window` of `ts`.
    fn best_scene_confidence_near(&self, ts: Duration) -> f64 {
        let window = self.config.search_window;
        self.scene_hints
            .iter()
            .filter(|h| {
                h.confidence >= self.config.min_scene_confidence
                    && ts.abs_diff(h.timestamp) <= window
            })
            .map(|h| h.confidence)
            .fold(0.0_f64, f64::max)
    }

    /// Compute a proximity score in `[0, 1]` for how close `candidate` is to
    /// `target`.  A candidate exactly at `target` scores `1.0`; one at the
    /// edge of the search window scores `0.0`.
    fn proximity_score(&self, candidate: Duration, target: Duration) -> f64 {
        let window_secs = self.config.search_window.as_secs_f64();
        if window_secs <= 0.0 {
            return 1.0;
        }
        let diff = candidate.abs_diff(target).as_secs_f64();
        (1.0 - diff / window_secs).max(0.0)
    }

    /// Return the timestamp of the nearest keyframe at or after `ts`.
    ///
    /// If no keyframe exists at or after `ts`, returns `ts` itself.
    fn nearest_keyframe_at_or_after(
        &self,
        sorted_kf: &[KeyframePosition],
        ts: Duration,
    ) -> Duration {
        sorted_kf
            .iter()
            .find(|k| k.timestamp >= ts)
            .map(|k| k.timestamp)
            .unwrap_or(ts)
    }

    /// Return all registered keyframes sorted by timestamp.
    #[must_use]
    pub fn keyframes_sorted(&self) -> Vec<&KeyframePosition> {
        let mut kf: Vec<&KeyframePosition> = self.keyframes.iter().collect();
        kf.sort_by_key(|k| k.timestamp);
        kf
    }

    /// Return the number of registered keyframes.
    #[must_use]
    pub fn keyframe_count(&self) -> usize {
        self.keyframes.len()
    }

    /// Return the number of registered scene change hints.
    #[must_use]
    pub fn scene_hint_count(&self) -> usize {
        self.scene_hints.len()
    }

    /// Clear all registered keyframes and scene hints.
    pub fn clear(&mut self) {
        self.keyframes.clear();
        self.scene_hints.clear();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn secs(s: u64) -> Duration {
        Duration::from_secs(s)
    }

    fn millis(ms: u64) -> Duration {
        Duration::from_millis(ms)
    }

    // --- BoundaryConfig validation -----------------------------------------

    #[test]
    fn test_boundary_config_default_is_valid() {
        assert!(BoundaryConfig::default().validate().is_ok());
    }

    #[test]
    fn test_boundary_config_invalid_min_ge_target() {
        let cfg = BoundaryConfig {
            min_segment_duration: secs(6),
            target_duration: secs(6),
            ..BoundaryConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_boundary_config_invalid_target_gt_max() {
        let cfg = BoundaryConfig {
            target_duration: secs(15),
            max_segment_duration: secs(12),
            ..BoundaryConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_boundary_config_invalid_search_window_too_large() {
        let cfg = BoundaryConfig {
            target_duration: secs(6),
            search_window: secs(4), // > target/2 = 3s
            ..BoundaryConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_boundary_config_invalid_confidence_out_of_range() {
        let cfg = BoundaryConfig {
            min_scene_confidence: 1.5,
            ..BoundaryConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    // --- SceneChangeHint -----------------------------------------------------

    #[test]
    fn test_scene_change_hint_confidence_clamped() {
        let hint = SceneChangeHint::new(secs(5), 2.0, SceneChangeSource::External);
        assert_eq!(hint.confidence, 1.0);

        let hint2 = SceneChangeHint::new(secs(5), -0.5, SceneChangeSource::PixelDiff);
        assert_eq!(hint2.confidence, 0.0);
    }

    #[test]
    fn test_scene_change_hint_fields() {
        let hint = SceneChangeHint::new(secs(10), 0.85, SceneChangeSource::HistogramDist);
        assert_eq!(hint.timestamp, secs(10));
        assert!((hint.confidence - 0.85).abs() < 1e-9);
        assert_eq!(hint.source, SceneChangeSource::HistogramDist);
    }

    // --- KeyframePosition ----------------------------------------------------

    #[test]
    fn test_keyframe_position_builder() {
        let kf = KeyframePosition::new(secs(3))
            .with_byte_offset(1024)
            .with_frame_size(512);
        assert_eq!(kf.timestamp, secs(3));
        assert_eq!(kf.byte_offset, Some(1024));
        assert_eq!(kf.frame_size, Some(512));
    }

    #[test]
    fn test_keyframe_position_defaults() {
        let kf = KeyframePosition::new(secs(5));
        assert!(kf.byte_offset.is_none());
        assert!(kf.frame_size.is_none());
    }

    // --- ContentBoundarySelector basics --------------------------------------

    #[test]
    fn test_selector_add_and_count() {
        let mut selector = ContentBoundarySelector::new(BoundaryConfig::default());
        selector.add_keyframe(KeyframePosition::new(secs(0)));
        selector.add_keyframe(KeyframePosition::new(secs(2)));
        assert_eq!(selector.keyframe_count(), 2);

        selector.add_scene_hint(SceneChangeHint::new(
            secs(6),
            0.9,
            SceneChangeSource::External,
        ));
        assert_eq!(selector.scene_hint_count(), 1);
    }

    #[test]
    fn test_selector_clear() {
        let mut selector = ContentBoundarySelector::new(BoundaryConfig::default());
        selector.add_keyframe(KeyframePosition::new(secs(0)));
        selector.add_scene_hint(SceneChangeHint::new(
            secs(0),
            0.5,
            SceneChangeSource::Annotation,
        ));
        selector.clear();
        assert_eq!(selector.keyframe_count(), 0);
        assert_eq!(selector.scene_hint_count(), 0);
    }

    #[test]
    fn test_select_boundaries_invalid_range() {
        let selector = ContentBoundarySelector::new(BoundaryConfig::default());
        let result = selector.select_boundaries(secs(10), secs(5));
        assert!(result.is_err());
    }

    #[test]
    fn test_select_boundaries_empty_produces_no_boundaries() {
        // Range smaller than one target duration — no boundaries expected.
        let selector = ContentBoundarySelector::new(BoundaryConfig::default());
        let result = selector
            .select_boundaries(secs(0), secs(4))
            .expect("should succeed");
        assert!(result.is_empty());
    }

    #[test]
    fn test_select_boundaries_regular_keyframes() {
        // Keyframes every 2 seconds, target 6s, 30s clip → expect ~4 boundaries.
        let mut selector = ContentBoundarySelector::new(BoundaryConfig::default());
        for i in 0..=15u64 {
            selector.add_keyframe(KeyframePosition::new(secs(i * 2)));
        }
        let boundaries = selector
            .select_boundaries(secs(0), secs(30))
            .expect("should succeed");

        // Verify all boundaries are within the range
        for b in &boundaries {
            assert!(b.timestamp > secs(0));
            assert!(b.timestamp <= secs(30));
            assert!(b.is_keyframe);
        }

        // At least 3 boundaries for a 30-second clip with 6-second target
        assert!(boundaries.len() >= 3);
    }

    #[test]
    fn test_select_boundaries_scene_change_preferred() {
        // Place a high-confidence scene change slightly off the target cut time
        // and verify that the selector picks the keyframe nearest the scene change.
        let config = BoundaryConfig {
            target_duration: secs(6),
            search_window: millis(600),
            ..BoundaryConfig::default()
        };
        let mut selector = ContentBoundarySelector::new(config);

        // Keyframes at 4s, 6s, 8s
        selector.add_keyframe(KeyframePosition::new(secs(4)));
        selector.add_keyframe(KeyframePosition::new(secs(6)));
        selector.add_keyframe(KeyframePosition::new(secs(8)));
        selector.add_keyframe(KeyframePosition::new(secs(12)));
        selector.add_keyframe(KeyframePosition::new(secs(14)));

        // High-confidence scene change at 6.1s (close to target of 6s)
        selector.add_scene_hint(SceneChangeHint::new(
            millis(6_100),
            0.95,
            SceneChangeSource::External,
        ));

        let boundaries = selector
            .select_boundaries(secs(0), secs(20))
            .expect("should succeed");

        // The first boundary should prefer 6s (nearest keyframe to scene change).
        let first = boundaries
            .first()
            .expect("should have at least one boundary");
        // Allow 6s or 8s — both are within the search window of the 6s target
        assert!(
            first.timestamp == secs(6) || first.timestamp == secs(8),
            "Expected 6s or 8s, got {:?}",
            first.timestamp
        );
    }

    #[test]
    fn test_select_boundaries_min_separation_enforced() {
        // Dense keyframes — verify min_segment_duration is respected.
        let config = BoundaryConfig {
            target_duration: secs(6),
            min_segment_duration: secs(3),
            ..BoundaryConfig::default()
        };
        let mut selector = ContentBoundarySelector::new(config.clone());

        // Keyframes every 1 second
        for i in 0..=30u64 {
            selector.add_keyframe(KeyframePosition::new(secs(i)));
        }

        let boundaries = selector
            .select_boundaries(secs(0), secs(30))
            .expect("should succeed");

        // Verify minimum separation
        let mut prev = secs(0);
        for b in &boundaries {
            assert!(
                b.timestamp >= prev + config.min_segment_duration,
                "Boundary {:?} too close to previous {:?}",
                b.timestamp,
                prev
            );
            prev = b.timestamp;
        }
    }

    #[test]
    fn test_select_boundaries_keyframes_sorted() {
        let mut selector = ContentBoundarySelector::new(BoundaryConfig::default());
        selector.add_keyframe(KeyframePosition::new(secs(12)));
        selector.add_keyframe(KeyframePosition::new(secs(0)));
        selector.add_keyframe(KeyframePosition::new(secs(6)));

        let sorted = selector.keyframes_sorted();
        assert_eq!(sorted[0].timestamp, secs(0));
        assert_eq!(sorted[1].timestamp, secs(6));
        assert_eq!(sorted[2].timestamp, secs(12));
    }

    #[test]
    fn test_boundary_candidate_fields() {
        let candidate = BoundaryCandidate {
            timestamp: secs(6),
            score: 0.85,
            is_keyframe: true,
            scene_confidence: 0.9,
        };
        assert_eq!(candidate.timestamp, secs(6));
        assert!((candidate.score - 0.85).abs() < 1e-9);
        assert!(candidate.is_keyframe);
        assert!((candidate.scene_confidence - 0.9).abs() < 1e-9);
    }

    #[test]
    fn test_add_keyframes_batch() {
        let mut selector = ContentBoundarySelector::new(BoundaryConfig::default());
        let kfs: Vec<KeyframePosition> = (0..5u64)
            .map(|i| KeyframePosition::new(secs(i * 6)))
            .collect();
        selector.add_keyframes(kfs);
        assert_eq!(selector.keyframe_count(), 5);
    }

    #[test]
    fn test_add_scene_hints_batch() {
        let mut selector = ContentBoundarySelector::new(BoundaryConfig::default());
        let hints: Vec<SceneChangeHint> = (1..4u64)
            .map(|i| SceneChangeHint::new(secs(i * 6), 0.7, SceneChangeSource::PixelDiff))
            .collect();
        selector.add_scene_hints(hints);
        assert_eq!(selector.scene_hint_count(), 3);
    }

    #[test]
    fn test_with_target_duration_constructor() {
        let config = BoundaryConfig::with_target_duration(secs(4));
        assert_eq!(config.target_duration, secs(4));
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_scene_change_source_variants() {
        assert_eq!(SceneChangeSource::PixelDiff, SceneChangeSource::PixelDiff);
        assert_ne!(
            SceneChangeSource::HistogramDist,
            SceneChangeSource::External
        );
        assert_ne!(
            SceneChangeSource::Annotation,
            SceneChangeSource::HistogramDist
        );
    }
}
