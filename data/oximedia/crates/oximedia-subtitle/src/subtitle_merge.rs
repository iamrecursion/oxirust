//! Multi-track subtitle merging utilities.
//!
//! Provides tools for combining subtitle tracks from different sources,
//! detecting overlaps, and resolving conflicts between entries.

#![allow(dead_code)]

/// Strategy used when merging subtitle tracks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeStrategy {
    /// Prefer entries from the first (primary) track on conflict.
    PreferFirst,
    /// Prefer entries from the last added track on conflict.
    PreferLast,
    /// Keep all entries; mark conflicting ones.
    KeepAll,
    /// Drop entries that overlap with any existing entry.
    DropOnConflict,
}

impl MergeStrategy {
    /// Return a human-readable name for the strategy.
    pub fn name(&self) -> &'static str {
        match self {
            MergeStrategy::PreferFirst => "prefer_first",
            MergeStrategy::PreferLast => "prefer_last",
            MergeStrategy::KeepAll => "keep_all",
            MergeStrategy::DropOnConflict => "drop_on_conflict",
        }
    }
}

/// A subtitle entry used for merging operations.
#[derive(Debug, Clone)]
pub struct SubtitleEntry {
    /// Track index this entry originated from.
    pub track: usize,
    /// Start time in milliseconds.
    pub start_ms: i64,
    /// End time in milliseconds.
    pub end_ms: i64,
    /// Text content.
    pub text: String,
    /// Whether this entry is flagged as conflicting.
    pub conflicted: bool,
}

impl SubtitleEntry {
    /// Create a new `SubtitleEntry`.
    pub fn new(track: usize, start_ms: i64, end_ms: i64, text: impl Into<String>) -> Self {
        Self {
            track,
            start_ms,
            end_ms,
            text: text.into(),
            conflicted: false,
        }
    }

    /// Returns `true` if this entry's time range overlaps with `other`.
    pub fn overlaps(&self, other: &SubtitleEntry) -> bool {
        self.start_ms < other.end_ms && self.end_ms > other.start_ms
    }

    /// Return the duration in milliseconds.
    pub fn duration_ms(&self) -> i64 {
        self.end_ms - self.start_ms
    }

    /// Mark this entry as conflicted.
    pub fn mark_conflicted(&mut self) {
        self.conflicted = true;
    }
}

/// Merger that combines multiple subtitle tracks into one.
#[derive(Debug)]
pub struct SubtitleMerger {
    strategy: MergeStrategy,
    tracks: Vec<Vec<SubtitleEntry>>,
}

impl SubtitleMerger {
    /// Create a new `SubtitleMerger` with the given strategy.
    pub fn new(strategy: MergeStrategy) -> Self {
        Self {
            strategy,
            tracks: Vec::new(),
        }
    }

    /// Add a track of subtitle entries.
    pub fn add_track(&mut self, entries: Vec<SubtitleEntry>) {
        self.tracks.push(entries);
    }

    /// Return the number of tracks.
    pub fn track_count(&self) -> usize {
        self.tracks.len()
    }

    /// Merge all tracks according to the configured strategy.
    pub fn merge(&self) -> MergeResult {
        let mut merged: Vec<SubtitleEntry> = Vec::new();
        let mut conflicts: usize = 0;

        for (track_idx, track) in self.tracks.iter().enumerate() {
            for entry in track {
                let mut new_entry = entry.clone();
                new_entry.track = track_idx;

                let overlap_count = merged.iter().filter(|e| e.overlaps(&new_entry)).count();

                if overlap_count > 0 {
                    conflicts += 1;
                    match self.strategy {
                        MergeStrategy::PreferFirst => {
                            // Skip this entry — existing entries take priority
                            continue;
                        }
                        MergeStrategy::PreferLast => {
                            // Remove overlapping entries, add new one
                            merged.retain(|e| !e.overlaps(&new_entry));
                            merged.push(new_entry);
                        }
                        MergeStrategy::KeepAll => {
                            new_entry.mark_conflicted();
                            merged.push(new_entry);
                        }
                        MergeStrategy::DropOnConflict => {
                            // Drop the incoming entry
                            continue;
                        }
                    }
                } else {
                    merged.push(new_entry);
                }
            }
        }

        // Sort by start time
        merged.sort_by_key(|e| e.start_ms);

        MergeResult {
            entries: merged,
            conflict_count: conflicts,
        }
    }
}

/// The result of a subtitle merge operation.
#[derive(Debug)]
pub struct MergeResult {
    /// Merged and sorted subtitle entries.
    pub entries: Vec<SubtitleEntry>,
    /// Number of conflicts detected during merge.
    pub conflict_count: usize,
}

impl MergeResult {
    /// Return the total number of entries.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Return the number of conflicts.
    pub fn conflict_count(&self) -> usize {
        self.conflict_count
    }

    /// Return entries that were flagged as conflicting.
    pub fn conflicted_entries(&self) -> Vec<&SubtitleEntry> {
        self.entries.iter().filter(|e| e.conflicted).collect()
    }

    /// Return the total duration span of the merged result (ms).
    pub fn span_ms(&self) -> i64 {
        let start = self.entries.first().map(|e| e.start_ms).unwrap_or(0);
        let end = self.entries.last().map(|e| e.end_ms).unwrap_or(0);
        end - start
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(track: usize, start: i64, end: i64, text: &str) -> SubtitleEntry {
        SubtitleEntry::new(track, start, end, text)
    }

    #[test]
    fn test_merge_strategy_name() {
        assert_eq!(MergeStrategy::PreferFirst.name(), "prefer_first");
        assert_eq!(MergeStrategy::PreferLast.name(), "prefer_last");
        assert_eq!(MergeStrategy::KeepAll.name(), "keep_all");
        assert_eq!(MergeStrategy::DropOnConflict.name(), "drop_on_conflict");
    }

    #[test]
    fn test_subtitle_entry_overlaps_true() {
        let a = make_entry(0, 1000, 4000, "A");
        let b = make_entry(1, 3000, 6000, "B");
        assert!(a.overlaps(&b));
        assert!(b.overlaps(&a));
    }

    #[test]
    fn test_subtitle_entry_overlaps_false() {
        let a = make_entry(0, 1000, 3000, "A");
        let b = make_entry(1, 3000, 5000, "B");
        assert!(!a.overlaps(&b));
    }

    #[test]
    fn test_subtitle_entry_duration_ms() {
        let e = make_entry(0, 1000, 4500, "Hi");
        assert_eq!(e.duration_ms(), 3500);
    }

    #[test]
    fn test_subtitle_entry_mark_conflicted() {
        let mut e = make_entry(0, 0, 1000, "X");
        assert!(!e.conflicted);
        e.mark_conflicted();
        assert!(e.conflicted);
    }

    #[test]
    fn test_merger_add_track() {
        let mut merger = SubtitleMerger::new(MergeStrategy::PreferFirst);
        merger.add_track(vec![make_entry(0, 0, 1000, "A")]);
        assert_eq!(merger.track_count(), 1);
    }

    #[test]
    fn test_merge_no_overlap() {
        let mut merger = SubtitleMerger::new(MergeStrategy::PreferFirst);
        merger.add_track(vec![make_entry(0, 0, 1000, "A")]);
        merger.add_track(vec![make_entry(1, 2000, 3000, "B")]);
        let result = merger.merge();
        assert_eq!(result.entry_count(), 2);
        assert_eq!(result.conflict_count(), 0);
    }

    #[test]
    fn test_merge_prefer_first_drops_conflict() {
        let mut merger = SubtitleMerger::new(MergeStrategy::PreferFirst);
        merger.add_track(vec![make_entry(0, 0, 3000, "First")]);
        merger.add_track(vec![make_entry(1, 1000, 4000, "Conflict")]);
        let result = merger.merge();
        assert_eq!(result.entry_count(), 1);
        assert_eq!(result.entries[0].text, "First");
    }

    #[test]
    fn test_merge_prefer_last_replaces() {
        let mut merger = SubtitleMerger::new(MergeStrategy::PreferLast);
        merger.add_track(vec![make_entry(0, 0, 3000, "First")]);
        merger.add_track(vec![make_entry(1, 1000, 4000, "Last")]);
        let result = merger.merge();
        assert_eq!(result.entry_count(), 1);
        assert_eq!(result.entries[0].text, "Last");
    }

    #[test]
    fn test_merge_keep_all_marks_conflicts() {
        let mut merger = SubtitleMerger::new(MergeStrategy::KeepAll);
        merger.add_track(vec![make_entry(0, 0, 3000, "A")]);
        merger.add_track(vec![make_entry(1, 1000, 4000, "B")]);
        let result = merger.merge();
        assert_eq!(result.entry_count(), 2);
        assert_eq!(result.conflict_count(), 1);
        assert_eq!(result.conflicted_entries().len(), 1);
    }

    #[test]
    fn test_merge_drop_on_conflict() {
        let mut merger = SubtitleMerger::new(MergeStrategy::DropOnConflict);
        merger.add_track(vec![make_entry(0, 0, 3000, "A")]);
        merger.add_track(vec![make_entry(1, 1000, 4000, "B")]);
        let result = merger.merge();
        assert_eq!(result.entry_count(), 1);
        assert_eq!(result.entries[0].text, "A");
    }

    #[test]
    fn test_merge_result_span_ms() {
        let mut merger = SubtitleMerger::new(MergeStrategy::KeepAll);
        merger.add_track(vec![
            make_entry(0, 1000, 3000, "A"),
            make_entry(0, 5000, 8000, "B"),
        ]);
        let result = merger.merge();
        assert_eq!(result.span_ms(), 7000);
    }

    #[test]
    fn test_merge_sorted_output() {
        let mut merger = SubtitleMerger::new(MergeStrategy::KeepAll);
        merger.add_track(vec![
            make_entry(0, 5000, 7000, "Late"),
            make_entry(0, 1000, 3000, "Early"),
        ]);
        let result = merger.merge();
        assert_eq!(result.entries[0].start_ms, 1000);
        assert_eq!(result.entries[1].start_ms, 5000);
    }
}
