#![allow(dead_code)]
//! Clip merging and concatenation utilities.
//!
//! This module provides tools for combining multiple clips into single unified clips,
//! including support for different merge strategies (concatenation, interleaving),
//! gap handling, and metadata consolidation. It is useful for assembling rough cuts,
//! combining multicam angles, and creating highlight reels.

use std::collections::HashMap;

/// Unique identifier for a merge operation.
pub type MergeId = u64;

/// Strategy for merging clips.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeStrategy {
    /// Concatenate clips sequentially end-to-end.
    Concatenate,
    /// Interleave clips by alternating between sources.
    Interleave,
    /// Overlap clips using crossfade transitions.
    Crossfade,
    /// Stack clips in parallel tracks.
    Stack,
}

/// Gap handling policy when merging clips.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GapPolicy {
    /// Remove gaps entirely (no dead time).
    RemoveGaps,
    /// Preserve original timing gaps between clips.
    PreserveGaps,
    /// Fill gaps with black/silence.
    FillBlack,
    /// Limit gaps to a maximum duration in milliseconds.
    MaxGap(u64),
}

/// A source clip entry for merging.
#[derive(Debug, Clone)]
pub struct MergeEntry {
    /// Identifier or name of the source clip.
    pub clip_name: String,
    /// Start time in milliseconds within the source clip.
    pub source_in: u64,
    /// End time in milliseconds within the source clip.
    pub source_out: u64,
    /// Track assignment (for stack merges).
    pub track: u32,
    /// Optional transition duration in milliseconds to the next clip.
    pub transition_ms: u64,
}

impl MergeEntry {
    /// Creates a new merge entry.
    pub fn new(clip_name: &str, source_in: u64, source_out: u64) -> Self {
        Self {
            clip_name: clip_name.to_string(),
            source_in,
            source_out,
            track: 0,
            transition_ms: 0,
        }
    }

    /// Returns the duration of this entry in milliseconds.
    pub fn duration_ms(&self) -> u64 {
        self.source_out.saturating_sub(self.source_in)
    }

    /// Sets the track assignment.
    pub fn with_track(mut self, track: u32) -> Self {
        self.track = track;
        self
    }

    /// Sets the transition duration.
    pub fn with_transition(mut self, ms: u64) -> Self {
        self.transition_ms = ms;
        self
    }
}

/// Metadata consolidation policy for merged clips.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetadataPolicy {
    /// Keep metadata from the first clip only.
    KeepFirst,
    /// Keep metadata from the last clip only.
    KeepLast,
    /// Merge all metadata (union of keywords, etc.).
    MergeAll,
    /// Discard all source metadata.
    Discard,
}

/// Configuration for a merge operation.
#[derive(Debug, Clone)]
pub struct MergeConfig {
    /// Strategy to use for merging.
    pub strategy: MergeStrategy,
    /// How to handle gaps between clips.
    pub gap_policy: GapPolicy,
    /// How to consolidate metadata.
    pub metadata_policy: MetadataPolicy,
    /// Name for the output merged clip.
    pub output_name: String,
    /// Default crossfade duration in milliseconds (for Crossfade strategy).
    pub default_crossfade_ms: u64,
}

impl Default for MergeConfig {
    fn default() -> Self {
        Self {
            strategy: MergeStrategy::Concatenate,
            gap_policy: GapPolicy::RemoveGaps,
            metadata_policy: MetadataPolicy::MergeAll,
            output_name: String::from("merged_clip"),
            default_crossfade_ms: 500,
        }
    }
}

/// Result of a merge operation.
#[derive(Debug, Clone)]
pub struct MergeResult {
    /// Unique ID assigned to this merge.
    pub merge_id: MergeId,
    /// Total duration of the merged output in milliseconds.
    pub total_duration_ms: u64,
    /// Number of source clips merged.
    pub source_count: usize,
    /// Number of transitions applied.
    pub transitions_applied: usize,
    /// Consolidated keywords from merged clips.
    pub merged_keywords: Vec<String>,
    /// Warnings generated during merge.
    pub warnings: Vec<String>,
}

/// The clip merger engine.
#[derive(Debug)]
pub struct ClipMerger {
    /// Merge configuration.
    config: MergeConfig,
    /// Source entries to merge.
    entries: Vec<MergeEntry>,
    /// Next merge ID.
    next_id: MergeId,
    /// Source clip keywords for consolidation.
    source_keywords: HashMap<String, Vec<String>>,
}

impl ClipMerger {
    /// Creates a new clip merger with the given configuration.
    pub fn new(config: MergeConfig) -> Self {
        Self {
            config,
            entries: Vec::new(),
            next_id: 1,
            source_keywords: HashMap::new(),
        }
    }

    /// Creates a merger with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(MergeConfig::default())
    }

    /// Adds a merge entry.
    pub fn add_entry(&mut self, entry: MergeEntry) {
        self.entries.push(entry);
    }

    /// Adds keywords associated with a source clip.
    pub fn add_keywords(&mut self, clip_name: &str, keywords: Vec<String>) {
        self.source_keywords.insert(clip_name.to_string(), keywords);
    }

    /// Returns the number of entries queued.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Clears all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.source_keywords.clear();
    }

    /// Executes the merge and returns the result.
    pub fn execute(&mut self) -> MergeResult {
        let merge_id = self.next_id;
        self.next_id += 1;

        let source_count = self.entries.len();
        let mut total_duration_ms = 0u64;
        let mut transitions_applied = 0usize;
        let mut warnings = Vec::new();

        match self.config.strategy {
            MergeStrategy::Concatenate => {
                for entry in &self.entries {
                    total_duration_ms += entry.duration_ms();
                }
            }
            MergeStrategy::Crossfade => {
                for (i, entry) in self.entries.iter().enumerate() {
                    total_duration_ms += entry.duration_ms();
                    if i > 0 {
                        let xfade = if entry.transition_ms > 0 {
                            entry.transition_ms
                        } else {
                            self.config.default_crossfade_ms
                        };
                        total_duration_ms = total_duration_ms.saturating_sub(xfade);
                        transitions_applied += 1;
                    }
                }
            }
            MergeStrategy::Interleave => {
                for entry in &self.entries {
                    total_duration_ms += entry.duration_ms();
                }
                if self.entries.len() > 1 {
                    warnings.push("Interleave mode: clips will alternate".to_string());
                }
            }
            MergeStrategy::Stack => {
                total_duration_ms = self
                    .entries
                    .iter()
                    .map(|e| e.duration_ms())
                    .max()
                    .unwrap_or(0);
            }
        }

        // Handle gap policy
        if let GapPolicy::MaxGap(max_ms) = self.config.gap_policy {
            if max_ms == 0 {
                warnings.push("MaxGap(0) is equivalent to RemoveGaps".to_string());
            }
        }

        // Consolidate keywords
        let merged_keywords = self.consolidate_keywords();

        // Validate
        for entry in &self.entries {
            if entry.duration_ms() == 0 {
                warnings.push(format!("Clip '{}' has zero duration", entry.clip_name));
            }
        }

        MergeResult {
            merge_id,
            total_duration_ms,
            source_count,
            transitions_applied,
            merged_keywords,
            warnings,
        }
    }

    /// Consolidates keywords according to the metadata policy.
    fn consolidate_keywords(&self) -> Vec<String> {
        match self.config.metadata_policy {
            MetadataPolicy::Discard => Vec::new(),
            MetadataPolicy::KeepFirst => {
                if let Some(entry) = self.entries.first() {
                    self.source_keywords
                        .get(&entry.clip_name)
                        .cloned()
                        .unwrap_or_default()
                } else {
                    Vec::new()
                }
            }
            MetadataPolicy::KeepLast => {
                if let Some(entry) = self.entries.last() {
                    self.source_keywords
                        .get(&entry.clip_name)
                        .cloned()
                        .unwrap_or_default()
                } else {
                    Vec::new()
                }
            }
            MetadataPolicy::MergeAll => {
                let mut all: Vec<String> = Vec::new();
                for entry in &self.entries {
                    if let Some(kws) = self.source_keywords.get(&entry.clip_name) {
                        for kw in kws {
                            if !all.contains(kw) {
                                all.push(kw.clone());
                            }
                        }
                    }
                }
                all
            }
        }
    }

    /// Validates that all entries have valid time ranges.
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        for (i, entry) in self.entries.iter().enumerate() {
            if entry.source_in > entry.source_out {
                errors.push(format!(
                    "Entry {}: in ({}) > out ({})",
                    i, entry.source_in, entry.source_out
                ));
            }
        }
        errors
    }
}

/// Computes the total duration of a list of merge entries when concatenated.
pub fn compute_concat_duration(entries: &[MergeEntry]) -> u64 {
    entries.iter().map(|e| e.duration_ms()).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_entry_new() {
        let e = MergeEntry::new("clip1", 0, 5000);
        assert_eq!(e.clip_name, "clip1");
        assert_eq!(e.source_in, 0);
        assert_eq!(e.source_out, 5000);
        assert_eq!(e.track, 0);
    }

    #[test]
    fn test_merge_entry_duration() {
        let e = MergeEntry::new("clip1", 1000, 4000);
        assert_eq!(e.duration_ms(), 3000);
    }

    #[test]
    fn test_merge_entry_zero_duration() {
        let e = MergeEntry::new("clip1", 5000, 5000);
        assert_eq!(e.duration_ms(), 0);
    }

    #[test]
    fn test_merge_entry_with_track() {
        let e = MergeEntry::new("clip1", 0, 1000).with_track(3);
        assert_eq!(e.track, 3);
    }

    #[test]
    fn test_merge_entry_with_transition() {
        let e = MergeEntry::new("clip1", 0, 1000).with_transition(250);
        assert_eq!(e.transition_ms, 250);
    }

    #[test]
    fn test_clip_merger_concatenate() {
        let mut merger = ClipMerger::with_defaults();
        merger.add_entry(MergeEntry::new("a", 0, 2000));
        merger.add_entry(MergeEntry::new("b", 0, 3000));
        let result = merger.execute();
        assert_eq!(result.total_duration_ms, 5000);
        assert_eq!(result.source_count, 2);
    }

    #[test]
    fn test_clip_merger_crossfade() {
        let config = MergeConfig {
            strategy: MergeStrategy::Crossfade,
            default_crossfade_ms: 500,
            ..Default::default()
        };
        let mut merger = ClipMerger::new(config);
        merger.add_entry(MergeEntry::new("a", 0, 3000));
        merger.add_entry(MergeEntry::new("b", 0, 3000));
        let result = merger.execute();
        // 3000 + 3000 - 500 = 5500
        assert_eq!(result.total_duration_ms, 5500);
        assert_eq!(result.transitions_applied, 1);
    }

    #[test]
    fn test_clip_merger_stack() {
        let config = MergeConfig {
            strategy: MergeStrategy::Stack,
            ..Default::default()
        };
        let mut merger = ClipMerger::new(config);
        merger.add_entry(MergeEntry::new("a", 0, 2000).with_track(0));
        merger.add_entry(MergeEntry::new("b", 0, 5000).with_track(1));
        let result = merger.execute();
        assert_eq!(result.total_duration_ms, 5000);
    }

    #[test]
    fn test_clip_merger_keywords_merge_all() {
        let mut merger = ClipMerger::with_defaults();
        merger.add_entry(MergeEntry::new("a", 0, 1000));
        merger.add_entry(MergeEntry::new("b", 0, 1000));
        merger.add_keywords("a", vec!["sports".to_string(), "outdoor".to_string()]);
        merger.add_keywords("b", vec!["outdoor".to_string(), "sunny".to_string()]);
        let result = merger.execute();
        assert_eq!(result.merged_keywords.len(), 3);
        assert!(result.merged_keywords.contains(&"sports".to_string()));
        assert!(result.merged_keywords.contains(&"outdoor".to_string()));
        assert!(result.merged_keywords.contains(&"sunny".to_string()));
    }

    #[test]
    fn test_clip_merger_keywords_discard() {
        let config = MergeConfig {
            metadata_policy: MetadataPolicy::Discard,
            ..Default::default()
        };
        let mut merger = ClipMerger::new(config);
        merger.add_entry(MergeEntry::new("a", 0, 1000));
        merger.add_keywords("a", vec!["tag".to_string()]);
        let result = merger.execute();
        assert!(result.merged_keywords.is_empty());
    }

    #[test]
    fn test_clip_merger_validate() {
        let mut merger = ClipMerger::with_defaults();
        merger.add_entry(MergeEntry::new("bad", 5000, 1000)); // invalid
        merger.add_entry(MergeEntry::new("good", 0, 3000));
        let errors = merger.validate();
        assert_eq!(errors.len(), 1);
    }

    #[test]
    fn test_clip_merger_clear() {
        let mut merger = ClipMerger::with_defaults();
        merger.add_entry(MergeEntry::new("a", 0, 1000));
        assert_eq!(merger.entry_count(), 1);
        merger.clear();
        assert_eq!(merger.entry_count(), 0);
    }

    #[test]
    fn test_clip_merger_empty() {
        let mut merger = ClipMerger::with_defaults();
        let result = merger.execute();
        assert_eq!(result.total_duration_ms, 0);
        assert_eq!(result.source_count, 0);
    }

    #[test]
    fn test_compute_concat_duration() {
        let entries = vec![
            MergeEntry::new("a", 0, 2000),
            MergeEntry::new("b", 500, 1500),
        ];
        assert_eq!(compute_concat_duration(&entries), 3000);
    }

    #[test]
    fn test_merge_id_increments() {
        let mut merger = ClipMerger::with_defaults();
        merger.add_entry(MergeEntry::new("a", 0, 1000));
        let r1 = merger.execute();
        merger.add_entry(MergeEntry::new("b", 0, 1000));
        let r2 = merger.execute();
        assert_eq!(r1.merge_id, 1);
        assert_eq!(r2.merge_id, 2);
    }
}
