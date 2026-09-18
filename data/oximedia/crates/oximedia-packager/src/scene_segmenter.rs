// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Scene-aligned segmentation for content-aware packaging.
//!
//! The [`SceneAlignedSegmenter`] aligns segment boundaries to scene changes
//! and keyframes for better compression quality. Rather than cutting blindly
//! at fixed intervals, it considers:
//!
//! 1. **Scene change confidence** — boundaries near detected scene changes
//!    produce cleaner segment boundaries and improve encoder efficiency.
//!
//! 2. **Keyframe alignment** — segments always start at keyframes, avoiding
//!    the need for open-GOP decoding at segment boundaries.
//!
//! 3. **Duration constraints** — minimum / maximum segment duration limits
//!    prevent pathologically short or long segments.
//!
//! This builds on top of [`ContentBoundarySelector`]
//! to provide a higher-level, easier-to-use API specifically designed for
//! packaging workflows.

use crate::content_boundary::{
    BoundaryCandidate, BoundaryConfig, ContentBoundarySelector, KeyframePosition, SceneChangeHint,
    SceneChangeSource,
};
use crate::error::{PackagerError, PackagerResult};
use std::time::Duration;

// ---------------------------------------------------------------------------
// SceneAlignedConfig
// ---------------------------------------------------------------------------

/// Configuration for the scene-aligned segmenter.
#[derive(Debug, Clone)]
pub struct SceneAlignedConfig {
    /// Target segment duration.
    pub target_duration: Duration,
    /// Minimum segment duration (hard lower bound).
    pub min_duration: Duration,
    /// Maximum segment duration (hard upper bound).
    pub max_duration: Duration,
    /// Half-width of the search window around each target cut point.
    pub search_window: Duration,
    /// Minimum scene-change confidence to influence boundaries.
    pub min_confidence: f64,
    /// Weight for scene-change signal vs. proximity to target.
    /// Range: 0.0 (ignore scene changes) to 1.0 (only scene changes).
    pub scene_weight: f64,
    /// Whether to prefer keyframe-aligned boundaries exclusively.
    pub keyframe_only: bool,
    /// Look-ahead buffer size: how many future keyframes to consider.
    pub lookahead_keyframes: usize,
}

impl Default for SceneAlignedConfig {
    fn default() -> Self {
        Self {
            target_duration: Duration::from_secs(6),
            min_duration: Duration::from_secs(2),
            max_duration: Duration::from_secs(12),
            search_window: Duration::from_millis(500),
            min_confidence: 0.3,
            scene_weight: 0.7,
            keyframe_only: true,
            lookahead_keyframes: 10,
        }
    }
}

impl SceneAlignedConfig {
    /// Create a config with a specific target duration.
    #[must_use]
    pub fn with_target(target: Duration) -> Self {
        Self {
            target_duration: target,
            ..Self::default()
        }
    }

    /// Set the search window.
    #[must_use]
    pub fn with_search_window(mut self, window: Duration) -> Self {
        self.search_window = window;
        self
    }

    /// Set the scene weight.
    #[must_use]
    pub fn with_scene_weight(mut self, weight: f64) -> Self {
        self.scene_weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if any invariant is violated.
    pub fn validate(&self) -> PackagerResult<()> {
        if self.min_duration >= self.target_duration {
            return Err(PackagerError::InvalidConfig(
                "min_duration must be less than target_duration".into(),
            ));
        }
        if self.target_duration > self.max_duration {
            return Err(PackagerError::InvalidConfig(
                "target_duration must not exceed max_duration".into(),
            ));
        }
        if self.search_window > self.target_duration / 2 {
            return Err(PackagerError::InvalidConfig(
                "search_window must be at most half of target_duration".into(),
            ));
        }
        if !(0.0..=1.0).contains(&self.min_confidence) {
            return Err(PackagerError::InvalidConfig(
                "min_confidence must be in [0.0, 1.0]".into(),
            ));
        }
        Ok(())
    }

    /// Convert to a [`BoundaryConfig`] for the underlying selector.
    #[must_use]
    pub fn to_boundary_config(&self) -> BoundaryConfig {
        BoundaryConfig {
            target_duration: self.target_duration,
            min_segment_duration: self.min_duration,
            max_segment_duration: self.max_duration,
            search_window: self.search_window,
            min_scene_confidence: self.min_confidence,
            scene_weight: self.scene_weight,
            keyframe_proximity_weight: 1.0 - self.scene_weight,
        }
    }
}

// ---------------------------------------------------------------------------
// SegmentBoundary
// ---------------------------------------------------------------------------

/// A selected segment boundary point.
#[derive(Debug, Clone)]
pub struct SegmentBoundary {
    /// Timestamp of this boundary.
    pub timestamp: Duration,
    /// Combined ranking score.
    pub score: f64,
    /// Whether this boundary falls on a keyframe.
    pub is_keyframe: bool,
    /// Scene-change confidence at this point.
    pub scene_confidence: f64,
    /// Segment duration (from this boundary to the next, or end of stream).
    pub segment_duration: Option<Duration>,
}

impl From<BoundaryCandidate> for SegmentBoundary {
    fn from(c: BoundaryCandidate) -> Self {
        Self {
            timestamp: c.timestamp,
            score: c.score,
            is_keyframe: c.is_keyframe,
            scene_confidence: c.scene_confidence,
            segment_duration: None,
        }
    }
}

// ---------------------------------------------------------------------------
// SceneAlignedSegmenter
// ---------------------------------------------------------------------------

/// Content-aware segmenter that aligns boundaries to scene changes and
/// keyframes for optimal compression quality.
///
/// # Example
///
/// ```
/// use std::time::Duration;
/// use oximedia_packager::scene_segmenter::{SceneAlignedConfig, SceneAlignedSegmenter};
///
/// let config = SceneAlignedConfig::default();
/// let mut segmenter = SceneAlignedSegmenter::new(config);
///
/// // Register keyframes at regular intervals
/// for i in 0..=15u64 {
///     segmenter.add_keyframe(Duration::from_secs(i * 2), None);
/// }
///
/// // Register a scene change
/// segmenter.add_scene_change(Duration::from_millis(6_100), 0.9);
///
/// // Compute boundaries for a 30-second clip
/// let boundaries = segmenter
///     .compute_boundaries(Duration::ZERO, Duration::from_secs(30))
///     .expect("should succeed");
///
/// assert!(!boundaries.is_empty());
/// ```
pub struct SceneAlignedSegmenter {
    config: SceneAlignedConfig,
    keyframes: Vec<KeyframePosition>,
    scene_hints: Vec<SceneChangeHint>,
}

impl SceneAlignedSegmenter {
    /// Create a new scene-aligned segmenter.
    #[must_use]
    pub fn new(config: SceneAlignedConfig) -> Self {
        Self {
            config,
            keyframes: Vec::new(),
            scene_hints: Vec::new(),
        }
    }

    /// Register a keyframe at the given timestamp.
    ///
    /// `byte_offset` is optional metadata for byte-range addressing.
    pub fn add_keyframe(&mut self, timestamp: Duration, byte_offset: Option<u64>) {
        let mut kf = KeyframePosition::new(timestamp);
        if let Some(offset) = byte_offset {
            kf = kf.with_byte_offset(offset);
        }
        self.keyframes.push(kf);
    }

    /// Register multiple keyframes at once.
    pub fn add_keyframes(&mut self, timestamps: impl IntoIterator<Item = Duration>) {
        for ts in timestamps {
            self.keyframes.push(KeyframePosition::new(ts));
        }
    }

    /// Register a scene change at the given timestamp with confidence.
    pub fn add_scene_change(&mut self, timestamp: Duration, confidence: f64) {
        self.scene_hints.push(SceneChangeHint::new(
            timestamp,
            confidence,
            SceneChangeSource::External,
        ));
    }

    /// Register a scene change from a specific source.
    pub fn add_scene_change_with_source(
        &mut self,
        timestamp: Duration,
        confidence: f64,
        source: SceneChangeSource,
    ) {
        self.scene_hints
            .push(SceneChangeHint::new(timestamp, confidence, source));
    }

    /// Register multiple scene changes at once.
    pub fn add_scene_changes(&mut self, hints: impl IntoIterator<Item = (Duration, f64)>) {
        for (ts, conf) in hints {
            self.add_scene_change(ts, conf);
        }
    }

    /// Return the number of registered keyframes.
    #[must_use]
    pub fn keyframe_count(&self) -> usize {
        self.keyframes.len()
    }

    /// Return the number of registered scene changes.
    #[must_use]
    pub fn scene_change_count(&self) -> usize {
        self.scene_hints.len()
    }

    /// Clear all registered keyframes and scene changes.
    pub fn clear(&mut self) {
        self.keyframes.clear();
        self.scene_hints.clear();
    }

    /// Compute optimal segment boundaries for the range `[start, end)`.
    ///
    /// Returns boundaries sorted by timestamp. Segment durations are
    /// filled in for each boundary (time from that boundary to the next
    /// boundary or `end`).
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid or `start >= end`.
    pub fn compute_boundaries(
        &self,
        start: Duration,
        end: Duration,
    ) -> PackagerResult<Vec<SegmentBoundary>> {
        self.config.validate()?;

        if start >= end {
            return Err(PackagerError::InvalidConfig(
                "start must be less than end".into(),
            ));
        }

        let boundary_config = self.config.to_boundary_config();
        let mut selector = ContentBoundarySelector::new(boundary_config);

        // Feed keyframes
        selector.add_keyframes(self.keyframes.iter().cloned());

        // Feed scene hints
        selector.add_scene_hints(self.scene_hints.iter().cloned());

        // Select raw boundaries
        let candidates = selector.select_boundaries(start, end)?;

        // Convert to SegmentBoundary and compute durations
        let mut boundaries: Vec<SegmentBoundary> =
            candidates.into_iter().map(SegmentBoundary::from).collect();

        // Fill in segment durations
        for i in 0..boundaries.len() {
            let next_ts = if i + 1 < boundaries.len() {
                boundaries[i + 1].timestamp
            } else {
                end
            };
            let duration = next_ts.saturating_sub(boundaries[i].timestamp);
            boundaries[i].segment_duration = Some(duration);
        }

        Ok(boundaries)
    }

    /// Compute segment count for the given range.
    ///
    /// Returns the number of segments (boundaries + 1 for the initial segment).
    ///
    /// # Errors
    ///
    /// Returns an error on invalid configuration.
    pub fn segment_count(&self, start: Duration, end: Duration) -> PackagerResult<usize> {
        let boundaries = self.compute_boundaries(start, end)?;
        Ok(boundaries.len() + 1) // +1 for the first segment before the first boundary
    }

    /// Return the average segment duration for the given range.
    ///
    /// # Errors
    ///
    /// Returns an error on invalid configuration.
    pub fn average_segment_duration(
        &self,
        start: Duration,
        end: Duration,
    ) -> PackagerResult<Duration> {
        let count = self.segment_count(start, end)?;
        if count == 0 {
            return Ok(Duration::ZERO);
        }
        let total = end.saturating_sub(start);
        Ok(total / count as u32)
    }

    /// Get the current configuration.
    #[must_use]
    pub fn config(&self) -> &SceneAlignedConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// ContentAwareSegmenter
// ---------------------------------------------------------------------------

/// A lightweight entry pairing a keyframe PTS with a scene-cut flag.
#[derive(Debug, Clone)]
pub struct KeyframeEntry {
    /// Presentation timestamp of this keyframe.
    pub pts: Duration,
    /// Whether this keyframe is a detected scene cut (hard cut / IDR at scene change).
    pub is_scene_cut: bool,
}

/// Simple content-aware segmenter that places boundaries at the keyframe
/// nearest to each target cut time, with preference for scene-cut keyframes.
///
/// Unlike [`SceneAlignedSegmenter`] (which uses a weighted confidence model
/// via [`ContentBoundarySelector`]),
/// `ContentAwareSegmenter` uses a direct binary scene-cut flag sourced from
/// codec metadata (e.g. IDR-at-scene-change) rather than a floating-point
/// confidence score.  This makes it a good fit for workflows where the codec
/// or a fast pre-pass provides reliable keyframe/scene-cut annotations.
///
/// # Algorithm
///
/// 1. Collect [`KeyframeEntry`] positions (PTS + `is_scene_cut` flag).
/// 2. For each target boundary time `t` (spaced by `target_duration`):
///    a. Search all keyframes within `[t − target/2, t + target/2]` (search window),
///       further constrained to `[prev + min_duration, prev + max_duration]`.
///    b. Among candidates, prefer scene-cut keyframes; break ties by proximity to `t`.
///    c. If no keyframe found in the window, fall back to the nearest keyframe
///       at or after `t`, clamped to the hard deadline (`prev + max_duration`).
/// 3. Return boundaries sorted by timestamp.
///
/// # Example
///
/// ```
/// use std::time::Duration;
/// use oximedia_packager::scene_segmenter::ContentAwareSegmenter;
///
/// let mut seg = ContentAwareSegmenter::new(Duration::from_secs(6));
/// for i in 0..=10u64 {
///     seg.add_keyframe(Duration::from_secs(i * 3), i % 2 == 0);
/// }
/// let boundaries = seg
///     .select_boundaries(Duration::ZERO, Duration::from_secs(30))
///     .expect("should succeed");
/// assert!(!boundaries.is_empty());
/// ```
pub struct ContentAwareSegmenter {
    keyframes: Vec<KeyframeEntry>,
    target_duration: Duration,
    min_duration: Duration,
    max_duration: Duration,
}

impl ContentAwareSegmenter {
    /// Create a new segmenter with the given target segment duration.
    ///
    /// Defaults:
    /// - `min_duration` = `target_duration / 3`
    /// - `max_duration` = `target_duration * 2`
    #[must_use]
    pub fn new(target_duration: Duration) -> Self {
        let min_duration = target_duration / 3;
        let max_duration = target_duration * 2;
        Self {
            keyframes: Vec::new(),
            target_duration,
            min_duration,
            max_duration,
        }
    }

    /// Override the minimum segment duration.
    #[must_use]
    pub fn with_min_duration(mut self, min: Duration) -> Self {
        self.min_duration = min;
        self
    }

    /// Override the maximum segment duration.
    #[must_use]
    pub fn with_max_duration(mut self, max: Duration) -> Self {
        self.max_duration = max;
        self
    }

    /// Register a keyframe at the given presentation timestamp.
    ///
    /// `is_scene_cut` should be `true` when the codec or a pre-pass has
    /// identified this frame as a scene change (hard cut).
    pub fn add_keyframe(&mut self, pts: Duration, is_scene_cut: bool) {
        self.keyframes.push(KeyframeEntry { pts, is_scene_cut });
    }

    /// Register multiple keyframes from an iterator of `(pts, is_scene_cut)` pairs.
    pub fn add_keyframes(&mut self, entries: impl IntoIterator<Item = (Duration, bool)>) {
        for (pts, sc) in entries {
            self.add_keyframe(pts, sc);
        }
    }

    /// Return the number of registered keyframes.
    #[must_use]
    pub fn keyframe_count(&self) -> usize {
        self.keyframes.len()
    }

    /// Return the configured target duration.
    #[must_use]
    pub fn target_duration(&self) -> Duration {
        self.target_duration
    }

    /// Clear all registered keyframes.
    pub fn clear(&mut self) {
        self.keyframes.clear();
    }

    /// Select segment boundaries for the range `[start, end)`.
    ///
    /// Returns an ordered list of [`SegmentBoundary`] values (not including
    /// `start`; the final segment ends at `end`).
    ///
    /// # Errors
    ///
    /// Returns [`PackagerError::InvalidConfig`] if `start >= end`.
    pub fn select_boundaries(
        &self,
        start: Duration,
        end: Duration,
    ) -> PackagerResult<Vec<SegmentBoundary>> {
        if start >= end {
            return Err(PackagerError::InvalidConfig(
                "start must be less than end".into(),
            ));
        }

        // Sort keyframes by PTS for efficient searching.
        let mut sorted: Vec<&KeyframeEntry> = self.keyframes.iter().collect();
        sorted.sort_by_key(|k| k.pts);

        let mut boundaries: Vec<SegmentBoundary> = Vec::new();
        let mut prev = start;

        loop {
            let target = prev + self.target_duration;
            if target >= end {
                break;
            }

            // Hard deadline: we must cut before exceeding max_duration.
            let deadline = (prev + self.max_duration).min(end);

            // Search window around target, constrained by [prev+min, deadline].
            let half = self.target_duration / 2;
            let win_start = target.saturating_sub(half).max(prev + self.min_duration);
            let win_end = (target + half).min(deadline);

            // Candidates within the search window.
            let candidates: Vec<&KeyframeEntry> = sorted
                .iter()
                .copied()
                .filter(|k| k.pts > win_start && k.pts <= win_end)
                .collect();

            let chosen_ts = if candidates.is_empty() {
                // Fallback: nearest keyframe at or after target, clamped to deadline.
                let fallback = sorted
                    .iter()
                    .find(|k| k.pts >= target)
                    .map(|k| k.pts)
                    .unwrap_or(target);
                fallback.min(deadline)
            } else {
                // Among candidates prefer scene cuts; break ties by proximity to target.
                let best = candidates
                    .iter()
                    .max_by(|a, b| {
                        let score_a = Self::candidate_score(a, target);
                        let score_b = Self::candidate_score(b, target);
                        score_a
                            .partial_cmp(&score_b)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .copied();

                match best {
                    Some(entry) => entry.pts,
                    None => target.min(deadline),
                }
            };

            // Guard against infinite loop: chosen_ts must advance past prev.
            if chosen_ts <= prev {
                prev += self.target_duration;
                continue;
            }

            let is_scene_cut = sorted.iter().any(|k| k.pts == chosen_ts && k.is_scene_cut);

            boundaries.push(SegmentBoundary {
                timestamp: chosen_ts,
                score: if is_scene_cut { 1.0 } else { 0.5 },
                is_keyframe: true,
                scene_confidence: if is_scene_cut { 1.0 } else { 0.0 },
                segment_duration: None,
            });
            prev = chosen_ts;
        }

        // Fill segment durations.
        for i in 0..boundaries.len() {
            let next_ts = if i + 1 < boundaries.len() {
                boundaries[i + 1].timestamp
            } else {
                end
            };
            boundaries[i].segment_duration = Some(next_ts.saturating_sub(boundaries[i].timestamp));
        }

        Ok(boundaries)
    }

    /// Compute a selection score for a keyframe candidate.
    ///
    /// Score = 1.0 for scene cuts + proximity bonus, purely proximity otherwise.
    fn candidate_score(entry: &KeyframeEntry, target: Duration) -> f64 {
        // Proximity: 1.0 at exact target, approaching 0 further away.
        // We use a generous window so the scale doesn't matter much.
        let diff_secs = entry.pts.abs_diff(target).as_secs_f64();
        let proximity = 1.0 / (1.0 + diff_secs);

        if entry.is_scene_cut {
            // Scene cuts get a large bonus so they always win over regular keyframes.
            10.0 + proximity
        } else {
            proximity
        }
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

    // --- SceneAlignedConfig -------------------------------------------------

    #[test]
    fn test_config_default_is_valid() {
        assert!(SceneAlignedConfig::default().validate().is_ok());
    }

    #[test]
    fn test_config_with_target() {
        let c = SceneAlignedConfig::with_target(secs(4));
        assert_eq!(c.target_duration, secs(4));
        assert!(c.validate().is_ok());
    }

    #[test]
    fn test_config_with_search_window() {
        let c = SceneAlignedConfig::default().with_search_window(millis(300));
        assert_eq!(c.search_window, millis(300));
    }

    #[test]
    fn test_config_with_scene_weight() {
        let c = SceneAlignedConfig::default().with_scene_weight(0.9);
        assert!((c.scene_weight - 0.9).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_scene_weight_clamped() {
        let c = SceneAlignedConfig::default().with_scene_weight(1.5);
        assert!((c.scene_weight - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_validate_min_ge_target() {
        let mut c = SceneAlignedConfig::default();
        c.min_duration = secs(6);
        c.target_duration = secs(6);
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_config_validate_target_gt_max() {
        let mut c = SceneAlignedConfig::default();
        c.target_duration = secs(15);
        c.max_duration = secs(12);
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_config_validate_window_too_large() {
        let mut c = SceneAlignedConfig::default();
        c.search_window = secs(4);
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_config_validate_confidence_out_of_range() {
        let mut c = SceneAlignedConfig::default();
        c.min_confidence = 1.5;
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_config_to_boundary_config() {
        let c = SceneAlignedConfig::default();
        let bc = c.to_boundary_config();
        assert_eq!(bc.target_duration, c.target_duration);
        assert_eq!(bc.min_segment_duration, c.min_duration);
        assert_eq!(bc.max_segment_duration, c.max_duration);
    }

    // --- SceneAlignedSegmenter basics ---------------------------------------

    #[test]
    fn test_segmenter_new() {
        let s = SceneAlignedSegmenter::new(SceneAlignedConfig::default());
        assert_eq!(s.keyframe_count(), 0);
        assert_eq!(s.scene_change_count(), 0);
    }

    #[test]
    fn test_segmenter_add_keyframe() {
        let mut s = SceneAlignedSegmenter::new(SceneAlignedConfig::default());
        s.add_keyframe(secs(0), None);
        s.add_keyframe(secs(2), Some(1024));
        assert_eq!(s.keyframe_count(), 2);
    }

    #[test]
    fn test_segmenter_add_keyframes_batch() {
        let mut s = SceneAlignedSegmenter::new(SceneAlignedConfig::default());
        s.add_keyframes((0..5u64).map(|i| secs(i * 2)));
        assert_eq!(s.keyframe_count(), 5);
    }

    #[test]
    fn test_segmenter_add_scene_change() {
        let mut s = SceneAlignedSegmenter::new(SceneAlignedConfig::default());
        s.add_scene_change(secs(6), 0.9);
        assert_eq!(s.scene_change_count(), 1);
    }

    #[test]
    fn test_segmenter_add_scene_changes_batch() {
        let mut s = SceneAlignedSegmenter::new(SceneAlignedConfig::default());
        s.add_scene_changes(vec![(secs(6), 0.9), (secs(12), 0.8)]);
        assert_eq!(s.scene_change_count(), 2);
    }

    #[test]
    fn test_segmenter_clear() {
        let mut s = SceneAlignedSegmenter::new(SceneAlignedConfig::default());
        s.add_keyframe(secs(0), None);
        s.add_scene_change(secs(6), 0.9);
        s.clear();
        assert_eq!(s.keyframe_count(), 0);
        assert_eq!(s.scene_change_count(), 0);
    }

    // --- compute_boundaries -------------------------------------------------

    #[test]
    fn test_compute_boundaries_invalid_range() {
        let s = SceneAlignedSegmenter::new(SceneAlignedConfig::default());
        assert!(s.compute_boundaries(secs(10), secs(5)).is_err());
    }

    #[test]
    fn test_compute_boundaries_short_range() {
        let s = SceneAlignedSegmenter::new(SceneAlignedConfig::default());
        let b = s
            .compute_boundaries(secs(0), secs(4))
            .expect("should succeed");
        assert!(b.is_empty()); // shorter than target → no boundaries
    }

    #[test]
    fn test_compute_boundaries_regular_keyframes() {
        let mut s = SceneAlignedSegmenter::new(SceneAlignedConfig::default());
        for i in 0..=15u64 {
            s.add_keyframe(secs(i * 2), None);
        }
        let boundaries = s
            .compute_boundaries(secs(0), secs(30))
            .expect("should succeed");

        assert!(!boundaries.is_empty());
        assert!(boundaries.len() >= 3); // 30s / 6s target = ~5 segments = ~4 boundaries

        // All boundaries should have segment durations filled in
        for b in &boundaries {
            assert!(b.segment_duration.is_some());
        }
    }

    #[test]
    fn test_compute_boundaries_scene_changes_influence() {
        let mut s = SceneAlignedSegmenter::new(SceneAlignedConfig::default());

        // Keyframes every 2 seconds
        for i in 0..=15u64 {
            s.add_keyframe(secs(i * 2), None);
        }

        // High-confidence scene change near 6s
        s.add_scene_change(millis(6_100), 0.95);

        let boundaries = s
            .compute_boundaries(secs(0), secs(30))
            .expect("should succeed");
        assert!(!boundaries.is_empty());

        // First boundary should be near the scene change (6s or 8s)
        let first = &boundaries[0];
        assert!(
            first.timestamp == secs(6) || first.timestamp == secs(8),
            "Expected 6s or 8s, got {:?}",
            first.timestamp
        );
    }

    #[test]
    fn test_compute_boundaries_segment_durations_sum() {
        let mut s = SceneAlignedSegmenter::new(SceneAlignedConfig::default());
        for i in 0..=15u64 {
            s.add_keyframe(secs(i * 2), None);
        }

        let boundaries = s
            .compute_boundaries(secs(0), secs(30))
            .expect("should succeed");

        // First segment: from 0 to first boundary
        // Then each boundary's segment_duration
        // Total should sum to 30s (approximately)
        if let Some(first) = boundaries.first() {
            let mut total = first.timestamp; // first segment
            for b in &boundaries {
                if let Some(d) = b.segment_duration {
                    total += d;
                }
            }
            // Should be close to 30s (within the search window tolerance)
            assert!(
                total >= secs(28) && total <= secs(32),
                "Total duration {total:?} should be close to 30s"
            );
        }
    }

    // --- segment_count / average_duration -----------------------------------

    #[test]
    fn test_segment_count() {
        let mut s = SceneAlignedSegmenter::new(SceneAlignedConfig::default());
        for i in 0..=15u64 {
            s.add_keyframe(secs(i * 2), None);
        }
        let count = s.segment_count(secs(0), secs(30)).expect("should succeed");
        assert!(
            count >= 4 && count <= 8,
            "segment count {count} out of expected range"
        );
    }

    #[test]
    fn test_average_segment_duration() {
        let mut s = SceneAlignedSegmenter::new(SceneAlignedConfig::default());
        for i in 0..=15u64 {
            s.add_keyframe(secs(i * 2), None);
        }
        let avg = s
            .average_segment_duration(secs(0), secs(30))
            .expect("should succeed");
        // Should be roughly around 6s
        assert!(
            avg >= secs(3) && avg <= secs(10),
            "Average duration {avg:?} out of expected range"
        );
    }

    // --- SegmentBoundary ----------------------------------------------------

    #[test]
    fn test_segment_boundary_from_candidate() {
        let candidate = BoundaryCandidate {
            timestamp: secs(6),
            score: 0.85,
            is_keyframe: true,
            scene_confidence: 0.9,
        };
        let b: SegmentBoundary = candidate.into();
        assert_eq!(b.timestamp, secs(6));
        assert!((b.score - 0.85).abs() < 1e-9);
        assert!(b.is_keyframe);
        assert!(b.segment_duration.is_none());
    }

    #[test]
    fn test_segmenter_config_accessor() {
        let config = SceneAlignedConfig::with_target(secs(4));
        let s = SceneAlignedSegmenter::new(config);
        assert_eq!(s.config().target_duration, secs(4));
    }

    #[test]
    fn test_add_scene_change_with_source() {
        let mut s = SceneAlignedSegmenter::new(SceneAlignedConfig::default());
        s.add_scene_change_with_source(secs(6), 0.9, SceneChangeSource::HistogramDist);
        assert_eq!(s.scene_change_count(), 1);
    }

    // --- ContentAwareSegmenter tests ----------------------------------------

    #[test]
    fn test_content_aware_new_defaults() {
        let s = ContentAwareSegmenter::new(secs(6));
        assert_eq!(s.target_duration(), secs(6));
        assert_eq!(s.keyframe_count(), 0);
    }

    #[test]
    fn test_content_aware_add_keyframe() {
        let mut s = ContentAwareSegmenter::new(secs(6));
        s.add_keyframe(secs(0), false);
        s.add_keyframe(secs(6), true);
        assert_eq!(s.keyframe_count(), 2);
    }

    #[test]
    fn test_content_aware_start_ge_end_error() {
        let s = ContentAwareSegmenter::new(secs(6));
        assert!(s.select_boundaries(secs(10), secs(5)).is_err());
        assert!(s.select_boundaries(secs(5), secs(5)).is_err());
    }

    #[test]
    fn test_content_aware_no_keyframes_returns_empty() {
        let s = ContentAwareSegmenter::new(secs(6));
        // Range shorter than target — no boundaries expected
        let b = s
            .select_boundaries(secs(0), secs(4))
            .expect("should succeed");
        assert!(b.is_empty());
    }

    #[test]
    fn test_content_aware_single_keyframe_boundary() {
        let mut s = ContentAwareSegmenter::new(secs(6));
        s.add_keyframe(secs(0), false);
        s.add_keyframe(secs(6), false);
        s.add_keyframe(secs(12), false);

        let boundaries = s
            .select_boundaries(secs(0), secs(12))
            .expect("should succeed");
        assert!(!boundaries.is_empty());
        // First boundary should be near 6s
        let first = &boundaries[0];
        assert_eq!(first.timestamp, secs(6));
    }

    #[test]
    fn test_content_aware_scene_cut_preferred() {
        // Two candidate keyframes near the 6s target: 5s (regular) and 7s (scene cut).
        // The scene cut at 7s should be preferred even though 5s is slightly closer.
        let mut s = ContentAwareSegmenter::new(secs(6));
        s.add_keyframe(secs(0), false);
        s.add_keyframe(secs(5), false); // 1s before target
        s.add_keyframe(secs(7), true); // 1s after target — scene cut
        s.add_keyframe(secs(14), false);

        let boundaries = s
            .select_boundaries(secs(0), secs(14))
            .expect("should succeed");
        assert!(!boundaries.is_empty());
        // The first boundary should prefer the scene-cut keyframe at 7s
        assert_eq!(
            boundaries[0].timestamp,
            secs(7),
            "scene cut should be preferred"
        );
    }

    #[test]
    fn test_content_aware_nearest_keyframe_fallback() {
        // No keyframe in the search window — should fall back to nearest at/after target.
        let mut s = ContentAwareSegmenter::new(secs(6));
        s.add_keyframe(secs(0), false);
        // Next keyframe at 9s, well outside default window of ±3s from 6s target
        s.add_keyframe(secs(9), false);
        s.add_keyframe(secs(18), false);

        let boundaries = s
            .select_boundaries(secs(0), secs(18))
            .expect("should succeed");
        assert!(!boundaries.is_empty());
        // The first boundary must be a valid keyframe position
        assert!(boundaries[0].timestamp > secs(0));
        assert!(boundaries[0].timestamp <= secs(18));
    }

    #[test]
    fn test_content_aware_ordered_output() {
        let mut s = ContentAwareSegmenter::new(secs(6));
        for i in 0..=20u64 {
            s.add_keyframe(secs(i * 2), i % 3 == 0);
        }
        let boundaries = s
            .select_boundaries(secs(0), secs(40))
            .expect("should succeed");

        // Verify boundaries are strictly ordered
        for w in boundaries.windows(2) {
            assert!(
                w[0].timestamp < w[1].timestamp,
                "boundaries must be strictly ascending"
            );
        }
    }

    #[test]
    fn test_content_aware_multi_segment_clip() {
        let mut s = ContentAwareSegmenter::new(secs(6));
        for i in 0..=30u64 {
            s.add_keyframe(secs(i * 2), i % 4 == 0);
        }
        let boundaries = s
            .select_boundaries(secs(0), secs(60))
            .expect("should succeed");

        // 60s / 6s = ~10 segments → ~9 boundaries
        assert!(
            boundaries.len() >= 5,
            "expected several boundaries, got {}",
            boundaries.len()
        );
    }
}
