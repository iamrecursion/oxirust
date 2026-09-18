//! Incremental / dirty-region rendering for the timeline editor.
//!
//! Tracks which frame ranges are "dirty" (need re-rendering) and exposes an
//! API to query, merge, and clear those regions.  Rendering itself is
//! delegated to the caller via the returned dirty region list.

use oximedia_core::Rational;

use crate::error::EditResult;
use crate::render::RenderConfig;
use crate::timeline::Timeline;
use std::sync::Arc;

// ─────────────────────────────────────────────────────────────────────────────
// DirtyRegion
// ─────────────────────────────────────────────────────────────────────────────

/// A contiguous range of frames that require re-rendering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DirtyRegion {
    /// First dirty frame (inclusive).
    pub start_frame: u64,
    /// Last dirty frame (exclusive).
    pub end_frame: u64,
}

impl DirtyRegion {
    /// Create a new dirty region.
    ///
    /// If `end_frame <= start_frame` the region is normalised to a single-frame
    /// region `[start_frame, start_frame + 1)`.
    #[must_use]
    pub fn new(start_frame: u64, end_frame: u64) -> Self {
        let end_frame = end_frame.max(start_frame + 1);
        Self {
            start_frame,
            end_frame,
        }
    }

    /// Returns `true` when the two regions overlap or are adjacent.
    #[must_use]
    pub fn overlaps(&self, other: &DirtyRegion) -> bool {
        // Adjacent regions (end == other.start) are also merged
        self.start_frame <= other.end_frame && other.start_frame <= self.end_frame
    }

    /// Merge two regions into a bounding region covering both.
    #[must_use]
    pub fn merge(&self, other: &DirtyRegion) -> DirtyRegion {
        DirtyRegion {
            start_frame: self.start_frame.min(other.start_frame),
            end_frame: self.end_frame.max(other.end_frame),
        }
    }

    /// Number of frames covered by this region.
    #[must_use]
    pub fn frame_count(&self) -> u64 {
        self.end_frame.saturating_sub(self.start_frame)
    }

    /// Returns `true` when `frame` falls inside this region.
    #[must_use]
    pub fn contains(&self, frame: u64) -> bool {
        frame >= self.start_frame && frame < self.end_frame
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// IncrementalRenderer
// ─────────────────────────────────────────────────────────────────────────────

/// Tracks dirty frame ranges and drives incremental re-renders.
///
/// Overlapping or adjacent dirty regions are automatically coalesced on every
/// `mark_dirty` call to keep the list compact.
pub struct IncrementalRenderer {
    /// Active dirty regions (always sorted by `start_frame`, non-overlapping).
    dirty_regions: Vec<DirtyRegion>,
    /// Render configuration used when rendering is triggered.
    pub config: RenderConfig,
    /// Frame rate of the associated timeline.
    pub frame_rate: Rational,
}

impl IncrementalRenderer {
    /// Create a new incremental renderer.
    #[must_use]
    pub fn new(config: RenderConfig, frame_rate: Rational) -> Self {
        Self {
            dirty_regions: Vec::new(),
            config,
            frame_rate,
        }
    }

    /// Mark the range `[start_frame, end_frame)` as dirty.
    ///
    /// The new region is inserted and then the list is coalesced so that it
    /// always consists of non-overlapping, sorted regions.
    pub fn mark_dirty(&mut self, start_frame: u64, end_frame: u64) {
        self.dirty_regions
            .push(DirtyRegion::new(start_frame, end_frame));
        self.coalesce();
    }

    /// Mark every frame in `[0, total_frames)` as dirty.
    pub fn mark_all_dirty(&mut self, total_frames: u64) {
        if total_frames == 0 {
            return;
        }
        self.dirty_regions = vec![DirtyRegion::new(0, total_frames)];
    }

    /// Returns `true` if `frame` falls inside any dirty region.
    #[must_use]
    pub fn is_dirty(&self, frame: u64) -> bool {
        self.dirty_regions.iter().any(|r| r.contains(frame))
    }

    /// Returns a slice of the current dirty regions (sorted, non-overlapping).
    #[must_use]
    pub fn get_dirty_regions(&self) -> &[DirtyRegion] {
        &self.dirty_regions
    }

    /// Total number of dirty frames across all regions.
    #[must_use]
    pub fn dirty_frame_count(&self) -> u64 {
        self.dirty_regions
            .iter()
            .map(DirtyRegion::frame_count)
            .sum()
    }

    /// Clear all dirty regions.
    pub fn clear_dirty(&mut self) {
        self.dirty_regions.clear();
    }

    /// Returns the number of dirty frames that need rendering, then clears
    /// the dirty list (signalling that rendering has been requested).
    ///
    /// The caller is expected to iterate `get_dirty_regions` *before* calling
    /// this method in order to render the correct frames.
    ///
    /// The `_timeline` parameter is accepted for API consistency and future use.
    pub fn render_incremental(&mut self, _timeline: &Arc<Timeline>) -> EditResult<usize> {
        let count = self.dirty_frame_count() as usize;
        self.clear_dirty();
        Ok(count)
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    /// Sort and merge overlapping/adjacent dirty regions.
    fn coalesce(&mut self) {
        if self.dirty_regions.len() <= 1 {
            return;
        }

        // Sort by start frame
        self.dirty_regions.sort_by_key(|r| r.start_frame);

        let mut merged: Vec<DirtyRegion> = Vec::with_capacity(self.dirty_regions.len());
        for region in &self.dirty_regions {
            if let Some(last) = merged.last_mut() {
                if last.end_frame >= region.start_frame {
                    // Overlapping or adjacent — extend the last region
                    last.end_frame = last.end_frame.max(region.end_frame);
                    continue;
                }
            }
            merged.push(*region);
        }

        self.dirty_regions = merged;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_core::Rational;

    fn renderer() -> IncrementalRenderer {
        IncrementalRenderer::new(RenderConfig::default(), Rational::new(30, 1))
    }

    #[test]
    fn test_mark_dirty_and_is_dirty() {
        let mut r = renderer();
        assert!(!r.is_dirty(10));
        r.mark_dirty(5, 20);
        assert!(r.is_dirty(5));
        assert!(r.is_dirty(10));
        assert!(r.is_dirty(19));
        assert!(!r.is_dirty(20));
        assert!(!r.is_dirty(4));
    }

    #[test]
    fn test_clear_dirty() {
        let mut r = renderer();
        r.mark_dirty(0, 100);
        assert_eq!(r.dirty_frame_count(), 100);
        r.clear_dirty();
        assert_eq!(r.dirty_frame_count(), 0);
        assert!(r.get_dirty_regions().is_empty());
    }

    #[test]
    fn test_merge_overlapping_regions() {
        let mut r = renderer();
        r.mark_dirty(0, 50);
        r.mark_dirty(30, 80);
        // Should coalesce into [0, 80)
        assert_eq!(r.get_dirty_regions().len(), 1);
        assert_eq!(r.get_dirty_regions()[0].start_frame, 0);
        assert_eq!(r.get_dirty_regions()[0].end_frame, 80);
        assert_eq!(r.dirty_frame_count(), 80);
    }

    #[test]
    fn test_merge_adjacent_regions() {
        let mut r = renderer();
        r.mark_dirty(0, 10);
        r.mark_dirty(10, 20);
        // Adjacent regions should merge
        assert_eq!(r.get_dirty_regions().len(), 1);
        assert_eq!(r.get_dirty_regions()[0].end_frame, 20);
    }

    #[test]
    fn test_non_overlapping_regions_stay_separate() {
        let mut r = renderer();
        r.mark_dirty(0, 10);
        r.mark_dirty(20, 30);
        assert_eq!(r.get_dirty_regions().len(), 2);
        assert_eq!(r.dirty_frame_count(), 20);
    }

    #[test]
    fn test_mark_all_dirty() {
        let mut r = renderer();
        r.mark_dirty(5, 10);
        r.mark_all_dirty(1000);
        assert_eq!(r.get_dirty_regions().len(), 1);
        assert_eq!(r.dirty_frame_count(), 1000);
    }

    #[test]
    fn test_render_incremental_clears_dirty() {
        let mut r = renderer();
        r.mark_dirty(0, 60);
        let timeline = std::sync::Arc::new(crate::timeline::Timeline::default());
        let count = r
            .render_incremental(&timeline)
            .expect("render_incremental ok");
        assert_eq!(count, 60);
        assert!(r.get_dirty_regions().is_empty());
    }

    #[test]
    fn test_dirty_region_frame_count() {
        let region = DirtyRegion::new(10, 50);
        assert_eq!(region.frame_count(), 40);
    }

    #[test]
    fn test_dirty_region_merge() {
        let a = DirtyRegion::new(0, 10);
        let b = DirtyRegion::new(5, 20);
        let merged = a.merge(&b);
        assert_eq!(merged.start_frame, 0);
        assert_eq!(merged.end_frame, 20);
    }
}
