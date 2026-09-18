//! Partial conform: process only a selected subset of clips in a batch.
//!
//! The standard `BatchProcessor` always processes *all* clips in every job.
//! In large episodic workflows it is common to re-conform only the clips that
//! changed since the last session (e.g. after a colour-grade revision or a
//! re-cut that touched only scenes 7–12).  Scanning and re-linking every clip
//! would waste significant I/O.
//!
//! [`PartialConformSelector`] encodes the selection predicate and
//! [`PartialBatchProcessor`] applies it to filter a `BatchJob`'s clip list
//! before processing.
//!
//! # Selection modes
//!
//! | Variant | Keeps clip if … |
//! |---------|-----------------|
//! | `All` | always (default — same as full conform) |
//! | `ByEventNumbers(set)` | clip's `event_number` metadata is in `set` |
//! | `ByReelNames(set)` | clip's `reel` metadata is in `set` |
//! | `ByClipIds(set)` | `clip.id` is in `set` |
//! | `Custom(fn)` | caller-supplied predicate returns `true` |
//!
//! # Example
//!
//! ```no_run
//! use oximedia_conform::partial_conform::{PartialBatchProcessor, PartialConformSelector};
//! use std::collections::HashSet;
//!
//! // Re-conform only event numbers 7, 8, 9
//! let selector = PartialConformSelector::ByEventNumbers(
//!     vec![7, 8, 9].into_iter().collect()
//! );
//! let processor = PartialBatchProcessor::new(selector);
//! let selected = processor.filter_clips(&[
//!     // ClipReference objects from an EDL import
//! ]);
//! assert!(selected.len() <= 3);
//! ```

use crate::error::ConformResult;
use crate::types::ClipReference;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

// ─────────────────────────────────────────────────────────────────────────────
// PartialConformSelector
// ─────────────────────────────────────────────────────────────────────────────

/// Predicate that selects which clips to include in a partial conform.
#[derive(Debug, Clone)]
pub enum PartialConformSelector {
    /// Include all clips (equivalent to a full conform).
    All,
    /// Include clips whose `event_number` metadata field matches any value in the set.
    ByEventNumbers(HashSet<u32>),
    /// Include clips whose `reel` metadata field is in the set.
    ByReelNames(HashSet<String>),
    /// Include clips whose `id` is in the set.
    ByClipIds(HashSet<String>),
    /// Include clips that match *all* of the supplied selectors (AND conjunction).
    All2(Box<PartialConformSelector>, Box<PartialConformSelector>),
    /// Include clips that match *any* of the supplied selectors (OR disjunction).
    Any2(Box<PartialConformSelector>, Box<PartialConformSelector>),
}

impl PartialConformSelector {
    /// Returns `true` if the given clip should be included.
    #[must_use]
    pub fn matches(&self, clip: &ClipReference) -> bool {
        match self {
            Self::All => true,
            Self::ByEventNumbers(nums) => clip
                .metadata
                .get("event_number")
                .and_then(|s| s.parse::<u32>().ok())
                .map_or(false, |n| nums.contains(&n)),
            Self::ByReelNames(reels) => clip
                .metadata
                .get("reel")
                .map_or(false, |r| reels.contains(r.as_str())),
            Self::ByClipIds(ids) => ids.contains(clip.id.as_str()),
            Self::All2(a, b) => a.matches(clip) && b.matches(clip),
            Self::Any2(a, b) => a.matches(clip) || b.matches(clip),
        }
    }

    /// Build a selector that accepts any clip whose reel is in the slice.
    #[must_use]
    pub fn by_reels(reels: &[&str]) -> Self {
        Self::ByReelNames(reels.iter().map(|s| s.to_string()).collect())
    }

    /// Build a selector that accepts any clip whose event number is in the slice.
    #[must_use]
    pub fn by_events(numbers: &[u32]) -> Self {
        Self::ByEventNumbers(numbers.iter().copied().collect())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PartialConformStats
// ─────────────────────────────────────────────────────────────────────────────

/// Statistics produced by a partial conform run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartialConformStats {
    /// Total clips in the full input list.
    pub total_clips: usize,
    /// Clips selected for this partial conform.
    pub selected_clips: usize,
    /// Clips skipped (not selected).
    pub skipped_clips: usize,
}

impl PartialConformStats {
    /// Compute stats from a full clip list and a filtered subset.
    #[must_use]
    pub fn compute(total: usize, selected: usize) -> Self {
        Self {
            total_clips: total,
            selected_clips: selected,
            skipped_clips: total.saturating_sub(selected),
        }
    }

    /// Fraction of clips selected: `selected / total` in [0.0, 1.0].
    #[must_use]
    pub fn selection_fraction(&self) -> f64 {
        if self.total_clips == 0 {
            0.0
        } else {
            self.selected_clips as f64 / self.total_clips as f64
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PartialBatchProcessor
// ─────────────────────────────────────────────────────────────────────────────

/// Wraps a [`PartialConformSelector`] and exposes filtering helpers.
///
/// The processor is stateless: it can be shared and reused across multiple
/// calls to [`filter_clips`].  The actual conform session creation and
/// execution is the responsibility of the caller.
///
/// [`filter_clips`]: PartialBatchProcessor::filter_clips
pub struct PartialBatchProcessor {
    /// The selection predicate.
    pub selector: PartialConformSelector,
}

impl PartialBatchProcessor {
    /// Create a new processor with the given selector.
    #[must_use]
    pub fn new(selector: PartialConformSelector) -> Self {
        Self { selector }
    }

    /// Return only the clips that satisfy the selector.
    #[must_use]
    pub fn filter_clips<'a>(&self, clips: &'a [ClipReference]) -> Vec<&'a ClipReference> {
        clips.iter().filter(|c| self.selector.matches(c)).collect()
    }

    /// Return owned clones of the selected clips.
    #[must_use]
    pub fn select_clips(&self, clips: &[ClipReference]) -> Vec<ClipReference> {
        clips
            .iter()
            .filter(|c| self.selector.matches(c))
            .cloned()
            .collect()
    }

    /// Return the stats for a filter operation without doing the actual filtering.
    #[must_use]
    pub fn stats(&self, clips: &[ClipReference]) -> PartialConformStats {
        let selected = clips.iter().filter(|c| self.selector.matches(c)).count();
        PartialConformStats::compute(clips.len(), selected)
    }

    /// Split clips into (selected, skipped) in a single pass.
    ///
    /// This avoids iterating the list twice when both partitions are needed.
    #[must_use]
    pub fn partition_clips(
        &self,
        clips: Vec<ClipReference>,
    ) -> (Vec<ClipReference>, Vec<ClipReference>) {
        clips.into_iter().partition(|c| self.selector.matches(c))
    }

    /// Build a `PartialConformPlan` that records the selection decisions.
    ///
    /// # Errors
    ///
    /// Always succeeds; returns `ConformResult` for API uniformity.
    pub fn build_plan(&self, clips: &[ClipReference]) -> ConformResult<PartialConformPlan> {
        let mut selected = Vec::new();
        let mut skipped = Vec::new();

        for clip in clips {
            if self.selector.matches(clip) {
                selected.push(clip.id.clone());
            } else {
                skipped.push(clip.id.clone());
            }
        }

        let stats = PartialConformStats::compute(clips.len(), selected.len());

        Ok(PartialConformPlan {
            selected_ids: selected,
            skipped_ids: skipped,
            stats,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PartialConformPlan
// ─────────────────────────────────────────────────────────────────────────────

/// A resolved partial conform plan: clip IDs that will be processed vs. skipped.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartialConformPlan {
    /// IDs of clips selected for conforming.
    pub selected_ids: Vec<String>,
    /// IDs of clips that were skipped.
    pub skipped_ids: Vec<String>,
    /// Summary statistics.
    pub stats: PartialConformStats,
}

impl PartialConformPlan {
    /// Returns `true` if the plan selects at least one clip.
    #[must_use]
    pub fn has_work(&self) -> bool {
        !self.selected_ids.is_empty()
    }

    /// Returns `true` if any clips were skipped.
    #[must_use]
    pub fn has_skipped(&self) -> bool {
        !self.skipped_ids.is_empty()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{FrameRate, Timecode, TrackType};
    use std::collections::HashMap;

    fn make_clip(id: &str, reel: &str, event: u32) -> ClipReference {
        let mut metadata = HashMap::new();
        metadata.insert("reel".to_string(), reel.to_string());
        metadata.insert("event_number".to_string(), event.to_string());
        let tc = Timecode::new(0, 0, 0, 0);
        ClipReference {
            id: id.to_string(),
            source_file: None,
            source_in: tc,
            source_out: tc,
            record_in: tc,
            record_out: tc,
            track: TrackType::Video,
            fps: FrameRate::Fps25,
            metadata,
        }
    }

    #[test]
    fn test_selector_all_matches_everything() {
        let clips = vec![make_clip("c1", "R1", 1), make_clip("c2", "R2", 2)];
        let proc = PartialBatchProcessor::new(PartialConformSelector::All);
        assert_eq!(proc.filter_clips(&clips).len(), 2);
    }

    #[test]
    fn test_selector_by_event_numbers() {
        let clips = vec![
            make_clip("c1", "R1", 1),
            make_clip("c2", "R2", 2),
            make_clip("c3", "R3", 3),
        ];
        let sel = PartialConformSelector::by_events(&[1, 3]);
        let proc = PartialBatchProcessor::new(sel);
        let selected = proc.filter_clips(&clips);
        assert_eq!(selected.len(), 2);
        assert!(selected.iter().any(|c| c.id == "c1"));
        assert!(selected.iter().any(|c| c.id == "c3"));
    }

    #[test]
    fn test_selector_by_reel_names() {
        let clips = vec![
            make_clip("c1", "REEL_A", 1),
            make_clip("c2", "REEL_B", 2),
            make_clip("c3", "REEL_A", 3),
        ];
        let sel = PartialConformSelector::by_reels(&["REEL_A"]);
        let proc = PartialBatchProcessor::new(sel);
        let selected = proc.filter_clips(&clips);
        assert_eq!(selected.len(), 2);
    }

    #[test]
    fn test_selector_by_clip_ids() {
        let clips = vec![
            make_clip("event_1", "R1", 1),
            make_clip("event_2", "R2", 2),
            make_clip("event_3", "R3", 3),
        ];
        let ids: HashSet<String> = ["event_1", "event_3"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let sel = PartialConformSelector::ByClipIds(ids);
        let proc = PartialBatchProcessor::new(sel);
        let selected = proc.filter_clips(&clips);
        assert_eq!(selected.len(), 2);
    }

    #[test]
    fn test_selector_any2_disjunction() {
        let clips = vec![
            make_clip("c1", "REEL_A", 1),
            make_clip("c2", "REEL_B", 2),
            make_clip("c3", "REEL_C", 3),
        ];
        let sel = PartialConformSelector::Any2(
            Box::new(PartialConformSelector::by_reels(&["REEL_A"])),
            Box::new(PartialConformSelector::by_events(&[2])),
        );
        let proc = PartialBatchProcessor::new(sel);
        let selected = proc.filter_clips(&clips);
        // c1 (REEL_A) + c2 (event 2) — but not c3
        assert_eq!(selected.len(), 2);
    }

    #[test]
    fn test_selector_all2_conjunction() {
        let clips = vec![
            make_clip("c1", "REEL_A", 1),
            make_clip("c2", "REEL_A", 2),
            make_clip("c3", "REEL_B", 1),
        ];
        let sel = PartialConformSelector::All2(
            Box::new(PartialConformSelector::by_reels(&["REEL_A"])),
            Box::new(PartialConformSelector::by_events(&[1])),
        );
        let proc = PartialBatchProcessor::new(sel);
        let selected = proc.filter_clips(&clips);
        // Only c1 has REEL_A AND event 1
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].id, "c1");
    }

    #[test]
    fn test_stats_computation() {
        let clips = vec![
            make_clip("c1", "R1", 1),
            make_clip("c2", "R2", 2),
            make_clip("c3", "R3", 3),
            make_clip("c4", "R4", 4),
        ];
        let sel = PartialConformSelector::by_events(&[1, 2]);
        let proc = PartialBatchProcessor::new(sel);
        let stats = proc.stats(&clips);
        assert_eq!(stats.total_clips, 4);
        assert_eq!(stats.selected_clips, 2);
        assert_eq!(stats.skipped_clips, 2);
        assert!((stats.selection_fraction() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_partition_clips() {
        let clips = vec![
            make_clip("c1", "R1", 1),
            make_clip("c2", "R2", 2),
            make_clip("c3", "R3", 3),
        ];
        let sel = PartialConformSelector::by_events(&[2]);
        let proc = PartialBatchProcessor::new(sel);
        let (selected, skipped) = proc.partition_clips(clips);
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].id, "c2");
        assert_eq!(skipped.len(), 2);
    }

    #[test]
    fn test_build_plan() {
        let clips = vec![make_clip("c1", "R1", 1), make_clip("c2", "R2", 2)];
        let sel = PartialConformSelector::by_events(&[1]);
        let proc = PartialBatchProcessor::new(sel);
        let plan = proc.build_plan(&clips).expect("plan should build");
        assert_eq!(plan.selected_ids, vec!["c1"]);
        assert_eq!(plan.skipped_ids, vec!["c2"]);
        assert!(plan.has_work());
        assert!(plan.has_skipped());
    }

    #[test]
    fn test_empty_clip_list() {
        let clips: Vec<ClipReference> = vec![];
        let sel = PartialConformSelector::All;
        let proc = PartialBatchProcessor::new(sel);
        assert!(proc.filter_clips(&clips).is_empty());
        let stats = proc.stats(&clips);
        assert_eq!(stats.total_clips, 0);
        assert_eq!(stats.selection_fraction(), 0.0);
    }
}
