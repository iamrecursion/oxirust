#![allow(dead_code)]
//! Intelligent segment merging for auto-assembled clips.
//!
//! After the auto-editor has identified candidate segments (highlights,
//! scored scenes, cut points) there are often very short or adjacent segments
//! that should be merged for better pacing and to avoid jarring micro-cuts.
//! This module provides configurable merge strategies.

use std::fmt;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A time range representing one segment.
#[derive(Debug, Clone, PartialEq)]
pub struct Segment {
    /// Start time in milliseconds.
    pub start_ms: i64,
    /// End time in milliseconds.
    pub end_ms: i64,
    /// Optional importance / score (0..1).
    pub score: f64,
    /// Optional label for the segment.
    pub label: String,
}

impl Segment {
    /// Create a new segment.
    pub fn new(start_ms: i64, end_ms: i64) -> Self {
        Self {
            start_ms,
            end_ms,
            score: 0.5,
            label: String::new(),
        }
    }

    /// Set the score.
    pub fn with_score(mut self, score: f64) -> Self {
        self.score = score;
        self
    }

    /// Set the label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = label.into();
        self
    }

    /// Duration in milliseconds.
    pub fn duration_ms(&self) -> i64 {
        self.end_ms - self.start_ms
    }

    /// Whether this segment overlaps another.
    pub fn overlaps(&self, other: &Self) -> bool {
        self.start_ms < other.end_ms && other.start_ms < self.end_ms
    }

    /// Gap between the end of this segment and the start of another.
    /// Negative values mean overlap.
    pub fn gap_ms(&self, other: &Self) -> i64 {
        other.start_ms - self.end_ms
    }

    /// Merge with another segment, keeping the wider time range.
    pub fn merge_with(&self, other: &Self) -> Self {
        let start = self.start_ms.min(other.start_ms);
        let end = self.end_ms.max(other.end_ms);
        let score = self.score.max(other.score);
        let label = if self.label.is_empty() {
            other.label.clone()
        } else {
            self.label.clone()
        };
        Self {
            start_ms: start,
            end_ms: end,
            score,
            label,
        }
    }
}

impl fmt::Display for Segment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{} - {} ms, score={:.2}]",
            self.start_ms, self.end_ms, self.score
        )
    }
}

/// Strategy used when deciding whether to merge two adjacent segments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeStrategy {
    /// Merge if the gap between segments is smaller than a threshold.
    GapThreshold,
    /// Merge if merged duration would stay under a maximum.
    MaxDuration,
    /// Merge both gap + max-duration rules together.
    Combined,
}

/// Configuration for the segment merger.
#[derive(Debug, Clone)]
pub struct SegmentMergeConfig {
    /// Strategy to apply.
    pub strategy: MergeStrategy,
    /// Maximum gap (ms) below which segments are merged.
    pub max_gap_ms: i64,
    /// Maximum resulting segment duration (ms). 0 = unlimited.
    pub max_merged_duration_ms: i64,
    /// Minimum segment duration after merge (short segments get merged).
    pub min_segment_duration_ms: i64,
    /// Whether to merge overlapping segments unconditionally.
    pub merge_overlaps: bool,
}

impl Default for SegmentMergeConfig {
    fn default() -> Self {
        Self {
            strategy: MergeStrategy::Combined,
            max_gap_ms: 500,
            max_merged_duration_ms: 30_000,
            min_segment_duration_ms: 1000,
            merge_overlaps: true,
        }
    }
}

impl SegmentMergeConfig {
    /// Create a new config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the merge strategy.
    pub fn with_strategy(mut self, s: MergeStrategy) -> Self {
        self.strategy = s;
        self
    }

    /// Set the max gap.
    pub fn with_max_gap_ms(mut self, ms: i64) -> Self {
        self.max_gap_ms = ms;
        self
    }

    /// Set the max merged duration.
    pub fn with_max_merged_duration_ms(mut self, ms: i64) -> Self {
        self.max_merged_duration_ms = ms;
        self
    }

    /// Set the minimum segment duration.
    pub fn with_min_segment_duration_ms(mut self, ms: i64) -> Self {
        self.min_segment_duration_ms = ms;
        self
    }
}

/// Result summary for a merge operation.
#[derive(Debug, Clone)]
pub struct MergeResult {
    /// Segments after merging.
    pub segments: Vec<Segment>,
    /// Number of merge operations performed.
    pub merges_performed: usize,
    /// Number of segments removed (too short after merging).
    pub segments_removed: usize,
}

// ---------------------------------------------------------------------------
// Merger
// ---------------------------------------------------------------------------

/// Performs intelligent segment merging.
#[derive(Debug, Clone)]
pub struct SegmentMerger {
    /// Configuration.
    config: SegmentMergeConfig,
}

impl SegmentMerger {
    /// Create a new merger.
    pub fn new(config: SegmentMergeConfig) -> Self {
        Self { config }
    }

    /// Merge a list of segments according to the configured strategy.
    ///
    /// Segments are sorted by start time before processing.
    pub fn merge(&self, segments: &[Segment]) -> MergeResult {
        if segments.is_empty() {
            return MergeResult {
                segments: Vec::new(),
                merges_performed: 0,
                segments_removed: 0,
            };
        }

        let mut sorted: Vec<Segment> = segments.to_vec();
        sorted.sort_by_key(|s| s.start_ms);

        let mut merged: Vec<Segment> = Vec::new();
        let mut merge_count: usize = 0;

        for seg in sorted {
            if let Some(last) = merged.last_mut() {
                if self.should_merge(last, &seg) {
                    *last = last.merge_with(&seg);
                    merge_count += 1;
                    continue;
                }
            }
            merged.push(seg);
        }

        // Remove segments that are too short
        let before_filter = merged.len();
        merged.retain(|s| s.duration_ms() >= self.config.min_segment_duration_ms);
        let segments_removed = before_filter - merged.len();

        MergeResult {
            segments: merged,
            merges_performed: merge_count,
            segments_removed,
        }
    }

    fn should_merge(&self, a: &Segment, b: &Segment) -> bool {
        if self.config.merge_overlaps && a.overlaps(b) {
            return true;
        }
        let gap = a.gap_ms(b);
        let merged_dur = b.end_ms - a.start_ms;

        match self.config.strategy {
            MergeStrategy::GapThreshold => gap <= self.config.max_gap_ms,
            MergeStrategy::MaxDuration => {
                self.config.max_merged_duration_ms <= 0
                    || merged_dur <= self.config.max_merged_duration_ms
            }
            MergeStrategy::Combined => {
                gap <= self.config.max_gap_ms
                    && (self.config.max_merged_duration_ms <= 0
                        || merged_dur <= self.config.max_merged_duration_ms)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(start: i64, end: i64) -> Segment {
        Segment::new(start, end)
    }

    #[test]
    fn test_segment_duration() {
        let s = seg(1000, 3000);
        assert_eq!(s.duration_ms(), 2000);
    }

    #[test]
    fn test_segment_overlaps() {
        assert!(seg(0, 1000).overlaps(&seg(500, 1500)));
        assert!(!seg(0, 1000).overlaps(&seg(1000, 2000)));
    }

    #[test]
    fn test_segment_gap() {
        assert_eq!(seg(0, 1000).gap_ms(&seg(1200, 2000)), 200);
        assert_eq!(seg(0, 1000).gap_ms(&seg(800, 2000)), -200);
    }

    #[test]
    fn test_merge_with() {
        let m = seg(0, 1000).merge_with(&seg(800, 2000));
        assert_eq!(m.start_ms, 0);
        assert_eq!(m.end_ms, 2000);
    }

    #[test]
    fn test_merge_empty() {
        let merger = SegmentMerger::new(SegmentMergeConfig::default());
        let result = merger.merge(&[]);
        assert!(result.segments.is_empty());
        assert_eq!(result.merges_performed, 0);
    }

    #[test]
    fn test_merge_single() {
        let merger = SegmentMerger::new(SegmentMergeConfig::default());
        let result = merger.merge(&[seg(0, 5000)]);
        assert_eq!(result.segments.len(), 1);
        assert_eq!(result.merges_performed, 0);
    }

    #[test]
    fn test_merge_adjacent_within_gap() {
        let cfg = SegmentMergeConfig::default().with_max_gap_ms(500);
        let merger = SegmentMerger::new(cfg);
        let result = merger.merge(&[seg(0, 2000), seg(2300, 5000)]);
        assert_eq!(result.segments.len(), 1);
        assert_eq!(result.merges_performed, 1);
    }

    #[test]
    fn test_no_merge_large_gap() {
        let cfg = SegmentMergeConfig::default().with_max_gap_ms(100);
        let merger = SegmentMerger::new(cfg);
        let result = merger.merge(&[seg(0, 2000), seg(3000, 5000)]);
        assert_eq!(result.segments.len(), 2);
        assert_eq!(result.merges_performed, 0);
    }

    #[test]
    fn test_merge_overlapping() {
        let merger = SegmentMerger::new(SegmentMergeConfig::default());
        let result = merger.merge(&[seg(0, 3000), seg(2000, 5000)]);
        assert_eq!(result.segments.len(), 1);
        assert_eq!(result.segments[0].start_ms, 0);
        assert_eq!(result.segments[0].end_ms, 5000);
    }

    #[test]
    fn test_max_duration_prevents_merge() {
        let cfg = SegmentMergeConfig::default()
            .with_strategy(MergeStrategy::Combined)
            .with_max_gap_ms(1000)
            .with_max_merged_duration_ms(3000);
        let merger = SegmentMerger::new(cfg);
        let result = merger.merge(&[seg(0, 2000), seg(2500, 6000)]);
        // Merged duration would be 6000 > 3000, so no merge
        assert_eq!(result.segments.len(), 2);
    }

    #[test]
    fn test_short_segments_removed() {
        let cfg = SegmentMergeConfig::default().with_min_segment_duration_ms(2000);
        let merger = SegmentMerger::new(cfg);
        let result = merger.merge(&[seg(0, 500), seg(10000, 15000)]);
        // First segment is too short (500ms < 2000ms)
        assert_eq!(result.segments.len(), 1);
        assert_eq!(result.segments[0].start_ms, 10000);
        assert_eq!(result.segments_removed, 1);
    }

    #[test]
    fn test_segment_display() {
        let s = seg(0, 1000).with_score(0.75);
        let display = format!("{s}");
        assert!(display.contains("0 - 1000 ms"));
        assert!(display.contains("0.75"));
    }

    #[test]
    fn test_merge_preserves_order() {
        let merger = SegmentMerger::new(
            SegmentMergeConfig::default()
                .with_max_gap_ms(0)
                .with_min_segment_duration_ms(0),
        );
        let segs = vec![seg(5000, 6000), seg(0, 1000), seg(2000, 3000)];
        let result = merger.merge(&segs);
        for w in result.segments.windows(2) {
            assert!(w[0].start_ms <= w[1].start_ms);
        }
    }

    #[test]
    fn test_config_builder() {
        let cfg = SegmentMergeConfig::new()
            .with_strategy(MergeStrategy::GapThreshold)
            .with_max_gap_ms(200)
            .with_max_merged_duration_ms(10_000)
            .with_min_segment_duration_ms(500);
        assert_eq!(cfg.strategy, MergeStrategy::GapThreshold);
        assert_eq!(cfg.max_gap_ms, 200);
        assert_eq!(cfg.max_merged_duration_ms, 10_000);
        assert_eq!(cfg.min_segment_duration_ms, 500);
    }
}
