//! EDL optimization and consolidation.
//!
//! This module provides tools for optimizing and consolidating EDL data,
//! including merging adjacent clips, removing duplicates, and sorting by
//! timeline position.

/// Options controlling EDL optimization behavior.
#[derive(Debug, Clone)]
pub struct OptimizeOptions {
    /// Merge adjacent clips from the same source.
    pub merge_adjacent: bool,
    /// Remove duplicate clips.
    pub remove_duplicates: bool,
    /// Sort clips by timeline (record_in) position.
    pub sort_by_timeline: bool,
    /// Consolidate clips from the same source file.
    pub consolidate_sources: bool,
}

impl OptimizeOptions {
    /// Create default optimization options (all enabled).
    #[must_use]
    pub fn all() -> Self {
        Self {
            merge_adjacent: true,
            remove_duplicates: true,
            sort_by_timeline: true,
            consolidate_sources: true,
        }
    }

    /// Create options with no optimizations enabled.
    #[must_use]
    pub fn none() -> Self {
        Self {
            merge_adjacent: false,
            remove_duplicates: false,
            sort_by_timeline: false,
            consolidate_sources: false,
        }
    }
}

impl Default for OptimizeOptions {
    fn default() -> Self {
        Self::all()
    }
}

/// Statistics describing the result of an optimization pass.
#[derive(Debug, Clone, Default)]
pub struct OptimizeStats {
    /// Number of clips before optimization.
    pub original_clips: usize,
    /// Number of clips after optimization.
    pub optimized_clips: usize,
    /// Number of clips merged with adjacent clips.
    pub merged_count: usize,
    /// Number of duplicate clips removed.
    pub removed_count: usize,
}

impl OptimizeStats {
    /// Create a new statistics report.
    #[must_use]
    pub fn new(original: usize) -> Self {
        Self {
            original_clips: original,
            optimized_clips: original,
            merged_count: 0,
            removed_count: 0,
        }
    }

    /// Calculate the reduction percentage.
    #[must_use]
    pub fn reduction_percent(&self) -> f64 {
        if self.original_clips == 0 {
            return 0.0;
        }
        let saved = self.original_clips.saturating_sub(self.optimized_clips);
        (saved as f64 / self.original_clips as f64) * 100.0
    }
}

/// A simplified clip entry used for optimization operations.
///
/// Uses frame counts for precise comparisons and merging.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EdlClipEntry {
    /// Unique event ID.
    pub id: u32,
    /// Source reel/file name.
    pub source_name: String,
    /// Record (timeline) in point, in frames.
    pub record_in: u64,
    /// Record (timeline) out point, in frames.
    pub record_out: u64,
    /// Source in point, in frames.
    pub source_in: u64,
    /// Source out point, in frames.
    pub source_out: u64,
}

impl EdlClipEntry {
    /// Create a new clip entry.
    #[allow(dead_code)]
    #[must_use]
    pub fn new(
        id: u32,
        source_name: impl Into<String>,
        record_in: u64,
        record_out: u64,
        source_in: u64,
        source_out: u64,
    ) -> Self {
        Self {
            id,
            source_name: source_name.into(),
            record_in,
            record_out,
            source_in,
            source_out,
        }
    }

    /// Duration of this clip in frames.
    #[must_use]
    pub fn duration(&self) -> u64 {
        self.record_out.saturating_sub(self.record_in)
    }

    /// Check whether two clips are adjacent on the timeline AND the source.
    ///
    /// Two clips are adjacent when:
    /// - They share the same source name.
    /// - `self.record_out == other.record_in` (back-to-back on timeline).
    /// - `self.source_out == other.source_in` (continuous in source).
    #[must_use]
    pub fn is_adjacent_to(&self, other: &Self) -> bool {
        self.source_name == other.source_name
            && self.record_out == other.record_in
            && self.source_out == other.source_in
    }
}

/// Run the full optimization pipeline on a list of clip entries.
///
/// The operations are applied in the following order (each enabled step
/// operates on the result of the previous one):
///
/// 1. Sort by timeline (if `opts.sort_by_timeline`)
/// 2. Remove duplicates (if `opts.remove_duplicates`)
/// 3. Merge adjacent clips (if `opts.merge_adjacent`)
///
/// Returns an [`OptimizeStats`] describing what changed.
pub fn optimize_edl(clips: &mut Vec<EdlClipEntry>, opts: &OptimizeOptions) -> OptimizeStats {
    let original = clips.len();
    let mut stats = OptimizeStats::new(original);

    if opts.sort_by_timeline {
        sort_by_timeline(clips);
    }

    if opts.remove_duplicates {
        let before = clips.len();
        let removed = remove_duplicate_clips(clips);
        stats.removed_count += removed;
        let after = clips.len();
        debug_assert_eq!(before - after, removed);
    }

    if opts.merge_adjacent {
        let before = clips.len();
        merge_adjacent_clips(clips);
        let merged = before - clips.len();
        stats.merged_count += merged;
    }

    stats.optimized_clips = clips.len();
    stats
}

/// Sort clips by their record-in (timeline) position, ascending.
pub fn sort_by_timeline(clips: &mut Vec<EdlClipEntry>) {
    clips.sort_by_key(|c| (c.record_in, c.source_name.clone()));
}

/// Remove clips that are exact duplicates of another clip in the list.
///
/// A duplicate is a clip where `id`, `source_name`, `record_in`,
/// `record_out`, `source_in`, and `source_out` all match an earlier entry.
///
/// Returns the number of duplicates removed.
pub fn remove_duplicate_clips(clips: &mut Vec<EdlClipEntry>) -> usize {
    let before = clips.len();
    let mut seen: std::collections::HashSet<(String, u64, u64, u64, u64)> =
        std::collections::HashSet::new();

    clips.retain(|c| {
        let key = (
            c.source_name.clone(),
            c.record_in,
            c.record_out,
            c.source_in,
            c.source_out,
        );
        seen.insert(key)
    });

    before - clips.len()
}

/// Consolidate events in an [`crate::Edl`] by merging adjacent cuts from the same reel.
///
/// This operates directly on the main `Edl` structure, converting events to
/// clip entries, running the optimization pipeline, and converting back.
/// The EDL events are replaced in-place.
///
/// Only consecutive `Cut` events with the same reel, where the first event's
/// `record_out` equals the next event's `record_in` AND the source timecodes
/// are contiguous, will be merged.
///
/// Returns an `OptimizeStats` describing the consolidation results.
pub fn consolidate_edl_events(edl: &mut crate::Edl, opts: &OptimizeOptions) -> OptimizeStats {
    use crate::event::{EditType, EdlEvent};
    use crate::timecode::EdlTimecode;

    if edl.events.is_empty() {
        return OptimizeStats::new(0);
    }

    let original_count = edl.events.len();

    // Keep a reference to the original events for non-merged properties
    let original_events = edl.events.clone();

    // Build a map of event index -> original event for properties preservation
    // We only consolidate Cut events; non-Cut events pass through untouched.
    let mut cut_clips: Vec<(usize, EdlClipEntry)> = Vec::new();
    let mut non_cut_indices: Vec<usize> = Vec::new();

    for (i, event) in edl.events.iter().enumerate() {
        if event.edit_type == EditType::Cut {
            cut_clips.push((
                i,
                EdlClipEntry {
                    id: event.number,
                    source_name: event.reel.clone(),
                    record_in: event.record_in.to_frames(),
                    record_out: event.record_out.to_frames(),
                    source_in: event.source_in.to_frames(),
                    source_out: event.source_out.to_frames(),
                },
            ));
        } else {
            non_cut_indices.push(i);
        }
    }

    let mut cut_clip_entries: Vec<EdlClipEntry> =
        cut_clips.iter().map(|(_, c)| c.clone()).collect();
    let stats = optimize_edl(&mut cut_clip_entries, opts);

    // Rebuild the event list from optimized clips + non-cut events
    let mut new_events: Vec<EdlEvent> = Vec::new();

    // Convert optimized clips back to events
    for clip in &cut_clip_entries {
        // Find the original event that corresponds to the beginning of this clip
        // (the merged clip retains the id of the first contributing event)
        let template = original_events
            .iter()
            .find(|e| e.number == clip.id)
            .unwrap_or(&original_events[0]);

        // Reconstruct timecodes from frames
        let frame_rate = edl.frame_rate;
        let record_in = match EdlTimecode::from_frames(clip.record_in, frame_rate) {
            Ok(tc) => tc,
            Err(_) => continue,
        };
        let record_out = match EdlTimecode::from_frames(clip.record_out, frame_rate) {
            Ok(tc) => tc,
            Err(_) => continue,
        };
        let source_in = match EdlTimecode::from_frames(clip.source_in, frame_rate) {
            Ok(tc) => tc,
            Err(_) => continue,
        };
        let source_out = match EdlTimecode::from_frames(clip.source_out, frame_rate) {
            Ok(tc) => tc,
            Err(_) => continue,
        };

        let mut new_event = EdlEvent::new(
            clip.id,
            clip.source_name.clone(),
            template.track.clone(),
            EditType::Cut,
            source_in,
            source_out,
            record_in,
            record_out,
        );

        // Preserve clip name from the first contributing event
        new_event.clip_name = template.clip_name.clone();
        new_event.comments = template.comments.clone();

        new_events.push(new_event);
    }

    // Add non-cut events back
    for &idx in &non_cut_indices {
        new_events.push(original_events[idx].clone());
    }

    // Sort by record_in
    new_events.sort_by_key(|e| e.record_in.to_frames());

    // Renumber
    for (i, event) in new_events.iter_mut().enumerate() {
        event.number = (i + 1) as u32;
    }

    edl.events = new_events;

    OptimizeStats {
        original_clips: original_count,
        optimized_clips: edl.events.len(),
        merged_count: stats.merged_count,
        removed_count: stats.removed_count,
    }
}

/// Merge adjacent clips that share the same source and are contiguous.
///
/// When two consecutive clips satisfy [`EdlClipEntry::is_adjacent_to`],
/// the first clip is extended to cover the range of both, and the second
/// is removed.  The process repeats until no more merges are possible.
pub fn merge_adjacent_clips(clips: &mut Vec<EdlClipEntry>) {
    let mut i = 0;
    while i + 1 < clips.len() {
        // Clone just the fields we need to avoid borrow-checker conflicts.
        let current_record_out = clips[i].record_out;
        let current_source_out = clips[i].source_out;
        let current_source_name = clips[i].source_name.clone();

        let next = &clips[i + 1];
        let can_merge = current_source_name == next.source_name
            && current_record_out == next.record_in
            && current_source_out == next.source_in;

        if can_merge {
            let next_record_out = clips[i + 1].record_out;
            let next_source_out = clips[i + 1].source_out;
            clips[i].record_out = next_record_out;
            clips[i].source_out = next_source_out;
            clips.remove(i + 1);
            // Don't advance i — try to merge the (now extended) clip with the
            // next one too.
        } else {
            i += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_clip(
        id: u32,
        source: &str,
        rec_in: u64,
        rec_out: u64,
        src_in: u64,
        src_out: u64,
    ) -> EdlClipEntry {
        EdlClipEntry::new(id, source, rec_in, rec_out, src_in, src_out)
    }

    // --- OptimizeOptions ---

    #[test]
    fn test_optimize_options_all() {
        let opts = OptimizeOptions::all();
        assert!(opts.merge_adjacent);
        assert!(opts.remove_duplicates);
        assert!(opts.sort_by_timeline);
        assert!(opts.consolidate_sources);
    }

    #[test]
    fn test_optimize_options_none() {
        let opts = OptimizeOptions::none();
        assert!(!opts.merge_adjacent);
        assert!(!opts.remove_duplicates);
        assert!(!opts.sort_by_timeline);
        assert!(!opts.consolidate_sources);
    }

    #[test]
    fn test_optimize_options_default() {
        let opts = OptimizeOptions::default();
        assert!(opts.merge_adjacent);
    }

    // --- OptimizeStats ---

    #[test]
    fn test_optimize_stats_reduction_percent_zero() {
        let stats = OptimizeStats::new(0);
        assert_eq!(stats.reduction_percent(), 0.0);
    }

    #[test]
    fn test_optimize_stats_reduction_percent_half() {
        let mut stats = OptimizeStats::new(10);
        stats.optimized_clips = 5;
        let pct = stats.reduction_percent();
        assert!((pct - 50.0).abs() < f64::EPSILON);
    }

    // --- EdlClipEntry ---

    #[test]
    fn test_clip_duration() {
        let clip = make_clip(1, "A001", 0, 50, 0, 50);
        assert_eq!(clip.duration(), 50);
    }

    #[test]
    fn test_clip_is_adjacent_to_true() {
        let a = make_clip(1, "A001", 0, 50, 100, 150);
        let b = make_clip(2, "A001", 50, 100, 150, 200);
        assert!(a.is_adjacent_to(&b));
    }

    #[test]
    fn test_clip_is_adjacent_to_false_different_source() {
        let a = make_clip(1, "A001", 0, 50, 0, 50);
        let b = make_clip(2, "B001", 50, 100, 50, 100);
        assert!(!a.is_adjacent_to(&b));
    }

    #[test]
    fn test_clip_is_adjacent_to_false_gap() {
        let a = make_clip(1, "A001", 0, 50, 0, 50);
        let b = make_clip(2, "A001", 60, 110, 50, 100);
        assert!(!a.is_adjacent_to(&b));
    }

    // --- sort_by_timeline ---

    #[test]
    fn test_sort_by_timeline() {
        let mut clips = vec![
            make_clip(2, "A001", 100, 150, 0, 50),
            make_clip(1, "A001", 0, 50, 0, 50),
            make_clip(3, "A001", 50, 100, 0, 50),
        ];
        sort_by_timeline(&mut clips);
        assert_eq!(clips[0].record_in, 0);
        assert_eq!(clips[1].record_in, 50);
        assert_eq!(clips[2].record_in, 100);
    }

    // --- remove_duplicate_clips ---

    #[test]
    fn test_remove_duplicate_clips_no_dupes() {
        let mut clips = vec![
            make_clip(1, "A001", 0, 50, 0, 50),
            make_clip(2, "A001", 50, 100, 50, 100),
        ];
        let removed = remove_duplicate_clips(&mut clips);
        assert_eq!(removed, 0);
        assert_eq!(clips.len(), 2);
    }

    #[test]
    fn test_remove_duplicate_clips_with_dupes() {
        let mut clips = vec![
            make_clip(1, "A001", 0, 50, 0, 50),
            make_clip(2, "A001", 0, 50, 0, 50), // exact duplicate of clip 1
            make_clip(3, "A001", 50, 100, 50, 100),
        ];
        let removed = remove_duplicate_clips(&mut clips);
        assert_eq!(removed, 1);
        assert_eq!(clips.len(), 2);
    }

    // --- merge_adjacent_clips ---

    #[test]
    fn test_merge_adjacent_clips_basic() {
        let mut clips = vec![
            make_clip(1, "A001", 0, 50, 100, 150),
            make_clip(2, "A001", 50, 100, 150, 200),
        ];
        merge_adjacent_clips(&mut clips);
        assert_eq!(clips.len(), 1);
        assert_eq!(clips[0].record_out, 100);
        assert_eq!(clips[0].source_out, 200);
    }

    #[test]
    fn test_merge_adjacent_clips_chain() {
        let mut clips = vec![
            make_clip(1, "A001", 0, 25, 0, 25),
            make_clip(2, "A001", 25, 50, 25, 50),
            make_clip(3, "A001", 50, 75, 50, 75),
        ];
        merge_adjacent_clips(&mut clips);
        assert_eq!(clips.len(), 1);
        assert_eq!(clips[0].record_out, 75);
    }

    #[test]
    fn test_merge_adjacent_clips_non_adjacent() {
        let mut clips = vec![
            make_clip(1, "A001", 0, 50, 0, 50),
            make_clip(2, "B001", 50, 100, 0, 50), // different source
        ];
        merge_adjacent_clips(&mut clips);
        assert_eq!(clips.len(), 2);
    }

    // --- optimize_edl ---

    #[test]
    fn test_optimize_edl_full() {
        let mut clips = vec![
            make_clip(3, "A001", 100, 150, 100, 150), // out of order
            make_clip(1, "A001", 0, 50, 0, 50),
            make_clip(2, "A001", 50, 100, 50, 100),
            make_clip(1, "A001", 0, 50, 0, 50), // duplicate
        ];
        let opts = OptimizeOptions::all();
        let stats = optimize_edl(&mut clips, &opts);

        // After dedup + sort + merge: 3 unique → 3 adjacent → 1 merged clip
        assert_eq!(clips.len(), 1);
        assert_eq!(clips[0].record_out, 150);
        assert!(stats.optimized_clips < stats.original_clips);
        assert!(stats.reduction_percent() > 0.0);
    }

    #[test]
    fn test_optimize_edl_empty() {
        let mut clips: Vec<EdlClipEntry> = vec![];
        let stats = optimize_edl(&mut clips, &OptimizeOptions::all());
        assert_eq!(stats.original_clips, 0);
        assert_eq!(stats.optimized_clips, 0);
    }

    // ── consolidate_edl_events tests ──

    mod consolidate_tests {
        use super::super::*;
        use crate::event::{EditType, EdlEvent, TrackType};
        use crate::timecode::{EdlFrameRate, EdlTimecode};
        use crate::{Edl, EdlFormat};

        fn make_cut_event(num: u32, reel: &str, sec_in: u8, sec_out: u8) -> EdlEvent {
            let fr = EdlFrameRate::Fps25;
            EdlEvent::new(
                num,
                reel.to_string(),
                TrackType::Video,
                EditType::Cut,
                EdlTimecode::new(1, 0, sec_in, 0, fr).expect("failed to create"),
                EdlTimecode::new(1, 0, sec_out, 0, fr).expect("failed to create"),
                EdlTimecode::new(1, 0, sec_in, 0, fr).expect("failed to create"),
                EdlTimecode::new(1, 0, sec_out, 0, fr).expect("failed to create"),
            )
        }

        fn make_edl(events: Vec<EdlEvent>) -> Edl {
            let mut edl = Edl::new(EdlFormat::Cmx3600);
            edl.set_frame_rate(EdlFrameRate::Fps25);
            for e in events {
                edl.events.push(e);
            }
            edl
        }

        #[test]
        fn test_consolidate_empty_edl() {
            let mut edl = make_edl(vec![]);
            let stats = consolidate_edl_events(&mut edl, &OptimizeOptions::all());
            assert_eq!(stats.original_clips, 0);
            assert_eq!(stats.optimized_clips, 0);
        }

        #[test]
        fn test_consolidate_single_event() {
            let mut edl = make_edl(vec![make_cut_event(1, "R1", 0, 5)]);
            let stats = consolidate_edl_events(&mut edl, &OptimizeOptions::all());
            assert_eq!(stats.original_clips, 1);
            assert_eq!(stats.optimized_clips, 1);
            assert_eq!(edl.events.len(), 1);
        }

        #[test]
        fn test_consolidate_adjacent_same_reel() {
            let mut edl = make_edl(vec![
                make_cut_event(1, "R1", 0, 5),
                make_cut_event(2, "R1", 5, 10),
            ]);
            let stats = consolidate_edl_events(&mut edl, &OptimizeOptions::all());
            assert_eq!(stats.original_clips, 2);
            assert_eq!(stats.optimized_clips, 1);
            assert_eq!(edl.events.len(), 1);
            // Merged event should span 0..10
            assert_eq!(edl.events[0].record_in.seconds(), 0);
            assert_eq!(edl.events[0].record_out.seconds(), 10);
        }

        #[test]
        fn test_consolidate_chain_of_three() {
            let mut edl = make_edl(vec![
                make_cut_event(1, "R1", 0, 5),
                make_cut_event(2, "R1", 5, 10),
                make_cut_event(3, "R1", 10, 15),
            ]);
            let stats = consolidate_edl_events(&mut edl, &OptimizeOptions::all());
            assert_eq!(stats.optimized_clips, 1);
            assert_eq!(edl.events[0].record_out.seconds(), 15);
        }

        #[test]
        fn test_consolidate_different_reels_not_merged() {
            let mut edl = make_edl(vec![
                make_cut_event(1, "R1", 0, 5),
                make_cut_event(2, "R2", 5, 10),
            ]);
            let stats = consolidate_edl_events(&mut edl, &OptimizeOptions::all());
            assert_eq!(stats.optimized_clips, 2);
            assert_eq!(edl.events.len(), 2);
        }

        #[test]
        fn test_consolidate_non_adjacent_not_merged() {
            let mut edl = make_edl(vec![
                make_cut_event(1, "R1", 0, 5),
                make_cut_event(2, "R1", 10, 15), // gap at 5..10
            ]);
            let stats = consolidate_edl_events(&mut edl, &OptimizeOptions::all());
            assert_eq!(stats.optimized_clips, 2);
        }

        #[test]
        fn test_consolidate_preserves_non_cut_events() {
            let fr = EdlFrameRate::Fps25;
            let tc3 = EdlTimecode::new(1, 0, 5, 0, fr).expect("failed to create");
            let tc4 = EdlTimecode::new(1, 0, 10, 0, fr).expect("failed to create");

            let mut dissolve = EdlEvent::new(
                2,
                "R1".to_string(),
                TrackType::Video,
                EditType::Dissolve,
                tc3,
                tc4,
                tc3,
                tc4,
            );
            dissolve.set_transition_duration(15);

            let mut edl = make_edl(vec![make_cut_event(1, "R1", 0, 5), dissolve]);
            let _stats = consolidate_edl_events(&mut edl, &OptimizeOptions::all());
            // Both events should remain (dissolve is not a Cut, so not consolidated)
            assert_eq!(edl.events.len(), 2);
        }

        #[test]
        fn test_consolidate_renumbers_correctly() {
            let mut edl = make_edl(vec![
                make_cut_event(10, "R1", 0, 5),
                make_cut_event(20, "R1", 5, 10),
                make_cut_event(30, "R2", 15, 20),
            ]);
            consolidate_edl_events(&mut edl, &OptimizeOptions::all());
            // After consolidation: R1 merged -> 1 event, R2 -> 1 event = 2 total
            assert_eq!(edl.events.len(), 2);
            assert_eq!(edl.events[0].number, 1);
            assert_eq!(edl.events[1].number, 2);
        }

        #[test]
        fn test_consolidate_preserves_clip_name() {
            let mut evt = make_cut_event(1, "R1", 0, 5);
            evt.clip_name = Some("my_clip.mov".to_string());
            let mut edl = make_edl(vec![evt, make_cut_event(2, "R1", 5, 10)]);
            consolidate_edl_events(&mut edl, &OptimizeOptions::all());
            assert_eq!(edl.events[0].clip_name, Some("my_clip.mov".to_string()));
        }

        #[test]
        fn test_consolidate_with_duplicates() {
            let mut edl = make_edl(vec![
                make_cut_event(1, "R1", 0, 5),
                make_cut_event(2, "R1", 0, 5), // duplicate
                make_cut_event(3, "R1", 5, 10),
            ]);
            let stats = consolidate_edl_events(&mut edl, &OptimizeOptions::all());
            // After dedup + merge: should be 1 event spanning 0..10
            assert_eq!(edl.events.len(), 1);
            assert!(stats.removed_count > 0 || stats.merged_count > 0);
        }
    }
}
