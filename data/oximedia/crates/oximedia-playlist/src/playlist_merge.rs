#![allow(dead_code)]

//! Playlist merging engine for combining multiple playlists.
//!
//! This module provides strategies for merging two or more playlists
//! into a single unified playlist: concatenation, interleaving,
//! deduplication, and priority-based merging.

use std::collections::HashSet;

/// Strategy for merging playlists.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MergeStrategy {
    /// Concatenate all playlists in order (playlist A items, then B, etc.).
    Concatenate,
    /// Interleave items from each playlist in round-robin fashion.
    Interleave,
    /// Merge by priority, taking higher-priority items first.
    PriorityMerge,
    /// Merge and deduplicate, keeping the first occurrence of each track.
    DeduplicateFirst,
    /// Merge and deduplicate, keeping the last occurrence of each track.
    DeduplicateLast,
    /// Merge by scheduled time, producing a time-sorted result.
    TimeSorted,
}

/// A track entry used during merge operations.
#[derive(Debug, Clone, PartialEq)]
pub struct MergeTrack {
    /// Unique track identifier.
    pub id: String,
    /// Display title.
    pub title: String,
    /// Duration in milliseconds.
    pub duration_ms: u64,
    /// Priority (higher = more important).
    pub priority: u32,
    /// Scheduled start time in epoch milliseconds (0 = unscheduled).
    pub scheduled_at: u64,
    /// Source playlist name for provenance tracking.
    pub source_playlist: String,
}

impl MergeTrack {
    /// Create a new merge track.
    pub fn new(id: &str, title: &str) -> Self {
        Self {
            id: id.to_string(),
            title: title.to_string(),
            duration_ms: 0,
            priority: 0,
            scheduled_at: 0,
            source_playlist: String::new(),
        }
    }

    /// Set the duration in milliseconds.
    pub fn with_duration_ms(mut self, ms: u64) -> Self {
        self.duration_ms = ms;
        self
    }

    /// Set the priority.
    pub fn with_priority(mut self, p: u32) -> Self {
        self.priority = p;
        self
    }

    /// Set the scheduled start time.
    pub fn with_scheduled_at(mut self, ts: u64) -> Self {
        self.scheduled_at = ts;
        self
    }

    /// Set the source playlist name.
    pub fn with_source(mut self, source: &str) -> Self {
        self.source_playlist = source.to_string();
        self
    }
}

/// Result of a merge operation.
#[derive(Debug, Clone)]
pub struct MergeResult {
    /// The merged tracks.
    pub tracks: Vec<MergeTrack>,
    /// Number of duplicates removed (if dedup strategy used).
    pub duplicates_removed: usize,
    /// Total duration of the merged playlist in milliseconds.
    pub total_duration_ms: u64,
    /// Number of source playlists that contributed.
    pub source_count: usize,
}

/// Engine that merges multiple playlists using configurable strategies.
#[derive(Debug)]
pub struct PlaylistMergeEngine {
    /// The merge strategy to use.
    strategy: MergeStrategy,
    /// Maximum total duration in milliseconds (0 = unlimited).
    max_duration_ms: u64,
}

impl PlaylistMergeEngine {
    /// Create a new merge engine with the given strategy.
    pub fn new(strategy: MergeStrategy) -> Self {
        Self {
            strategy,
            max_duration_ms: 0,
        }
    }

    /// Set a maximum total duration for the merged playlist.
    pub fn with_max_duration_ms(mut self, max_ms: u64) -> Self {
        self.max_duration_ms = max_ms;
        self
    }

    /// Return the current merge strategy.
    pub fn strategy(&self) -> &MergeStrategy {
        &self.strategy
    }

    /// Merge multiple playlists into one.
    pub fn merge(&self, playlists: &[Vec<MergeTrack>]) -> MergeResult {
        let source_count = playlists.len();
        let (tracks, duplicates_removed) = match &self.strategy {
            MergeStrategy::Concatenate => (Self::concatenate(playlists), 0),
            MergeStrategy::Interleave => (Self::interleave(playlists), 0),
            MergeStrategy::PriorityMerge => (Self::priority_merge(playlists), 0),
            MergeStrategy::DeduplicateFirst => Self::dedup_first(playlists),
            MergeStrategy::DeduplicateLast => Self::dedup_last(playlists),
            MergeStrategy::TimeSorted => (Self::time_sorted(playlists), 0),
        };

        let tracks = self.apply_max_duration(tracks);
        let total_duration_ms = tracks.iter().map(|t| t.duration_ms).sum();

        MergeResult {
            tracks,
            duplicates_removed,
            total_duration_ms,
            source_count,
        }
    }

    /// Concatenate all playlists sequentially.
    fn concatenate(playlists: &[Vec<MergeTrack>]) -> Vec<MergeTrack> {
        playlists.iter().flat_map(|p| p.iter().cloned()).collect()
    }

    /// Interleave tracks from all playlists in round-robin order.
    fn interleave(playlists: &[Vec<MergeTrack>]) -> Vec<MergeTrack> {
        let max_len = playlists.iter().map(std::vec::Vec::len).max().unwrap_or(0);
        let mut result = Vec::new();
        for i in 0..max_len {
            for playlist in playlists {
                if let Some(track) = playlist.get(i) {
                    result.push(track.clone());
                }
            }
        }
        result
    }

    /// Merge all tracks then sort by priority descending.
    fn priority_merge(playlists: &[Vec<MergeTrack>]) -> Vec<MergeTrack> {
        let mut all: Vec<MergeTrack> = playlists.iter().flat_map(|p| p.iter().cloned()).collect();
        all.sort_by(|a, b| b.priority.cmp(&a.priority));
        all
    }

    /// Concatenate and deduplicate, keeping the first occurrence.
    fn dedup_first(playlists: &[Vec<MergeTrack>]) -> (Vec<MergeTrack>, usize) {
        let all: Vec<MergeTrack> = playlists.iter().flat_map(|p| p.iter().cloned()).collect();
        let total = all.len();
        let mut seen = HashSet::new();
        let mut result = Vec::new();
        for track in all {
            if seen.insert(track.id.clone()) {
                result.push(track);
            }
        }
        let removed = total - result.len();
        (result, removed)
    }

    /// Concatenate and deduplicate, keeping the last occurrence.
    fn dedup_last(playlists: &[Vec<MergeTrack>]) -> (Vec<MergeTrack>, usize) {
        let all: Vec<MergeTrack> = playlists.iter().flat_map(|p| p.iter().cloned()).collect();
        let total = all.len();
        let mut seen = HashSet::new();
        let mut result = Vec::new();
        // Iterate in reverse, then reverse the result
        for track in all.into_iter().rev() {
            if seen.insert(track.id.clone()) {
                result.push(track);
            }
        }
        result.reverse();
        let removed = total - result.len();
        (result, removed)
    }

    /// Concatenate and sort by scheduled time ascending.
    fn time_sorted(playlists: &[Vec<MergeTrack>]) -> Vec<MergeTrack> {
        let mut all: Vec<MergeTrack> = playlists.iter().flat_map(|p| p.iter().cloned()).collect();
        all.sort_by_key(|t| t.scheduled_at);
        all
    }

    /// Trim the track list if it exceeds the maximum duration.
    fn apply_max_duration(&self, tracks: Vec<MergeTrack>) -> Vec<MergeTrack> {
        if self.max_duration_ms == 0 {
            return tracks;
        }
        let mut total = 0u64;
        let mut result = Vec::new();
        for track in tracks {
            if total + track.duration_ms > self.max_duration_ms {
                break;
            }
            total += track.duration_ms;
            result.push(track);
        }
        result
    }

    /// Count distinct track ids across all playlists.
    pub fn distinct_track_count(playlists: &[Vec<MergeTrack>]) -> usize {
        let ids: HashSet<&str> = playlists
            .iter()
            .flat_map(|p| p.iter().map(|t| t.id.as_str()))
            .collect();
        ids.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn playlist_a() -> Vec<MergeTrack> {
        vec![
            MergeTrack::new("1", "Track One")
                .with_duration_ms(3000)
                .with_priority(5)
                .with_scheduled_at(1000)
                .with_source("A"),
            MergeTrack::new("2", "Track Two")
                .with_duration_ms(4000)
                .with_priority(3)
                .with_scheduled_at(5000)
                .with_source("A"),
        ]
    }

    fn playlist_b() -> Vec<MergeTrack> {
        vec![
            MergeTrack::new("3", "Track Three")
                .with_duration_ms(2000)
                .with_priority(8)
                .with_scheduled_at(2000)
                .with_source("B"),
            MergeTrack::new("2", "Track Two B")
                .with_duration_ms(4500)
                .with_priority(1)
                .with_scheduled_at(6000)
                .with_source("B"),
        ]
    }

    #[test]
    fn test_concatenate() {
        let engine = PlaylistMergeEngine::new(MergeStrategy::Concatenate);
        let result = engine.merge(&[playlist_a(), playlist_b()]);
        assert_eq!(result.tracks.len(), 4);
        assert_eq!(result.tracks[0].id, "1");
        assert_eq!(result.tracks[2].id, "3");
    }

    #[test]
    fn test_interleave() {
        let engine = PlaylistMergeEngine::new(MergeStrategy::Interleave);
        let result = engine.merge(&[playlist_a(), playlist_b()]);
        assert_eq!(result.tracks.len(), 4);
        // Round-robin: a[0], b[0], a[1], b[1]
        assert_eq!(result.tracks[0].id, "1");
        assert_eq!(result.tracks[1].id, "3");
        assert_eq!(result.tracks[2].id, "2");
    }

    #[test]
    fn test_priority_merge() {
        let engine = PlaylistMergeEngine::new(MergeStrategy::PriorityMerge);
        let result = engine.merge(&[playlist_a(), playlist_b()]);
        assert_eq!(result.tracks[0].priority, 8);
        assert_eq!(result.tracks[1].priority, 5);
    }

    #[test]
    fn test_dedup_first() {
        let engine = PlaylistMergeEngine::new(MergeStrategy::DeduplicateFirst);
        let result = engine.merge(&[playlist_a(), playlist_b()]);
        // Track "2" appears in both; first occurrence is from A
        assert_eq!(result.tracks.len(), 3);
        assert_eq!(result.duplicates_removed, 1);
        let track_2 = result
            .tracks
            .iter()
            .find(|t| t.id == "2")
            .expect("should succeed in test");
        assert_eq!(track_2.source_playlist, "A");
    }

    #[test]
    fn test_dedup_last() {
        let engine = PlaylistMergeEngine::new(MergeStrategy::DeduplicateLast);
        let result = engine.merge(&[playlist_a(), playlist_b()]);
        assert_eq!(result.tracks.len(), 3);
        assert_eq!(result.duplicates_removed, 1);
        let track_2 = result
            .tracks
            .iter()
            .find(|t| t.id == "2")
            .expect("should succeed in test");
        assert_eq!(track_2.source_playlist, "B");
    }

    #[test]
    fn test_time_sorted() {
        let engine = PlaylistMergeEngine::new(MergeStrategy::TimeSorted);
        let result = engine.merge(&[playlist_a(), playlist_b()]);
        for w in result.tracks.windows(2) {
            assert!(w[0].scheduled_at <= w[1].scheduled_at);
        }
    }

    #[test]
    fn test_max_duration() {
        let engine =
            PlaylistMergeEngine::new(MergeStrategy::Concatenate).with_max_duration_ms(5000);
        let result = engine.merge(&[playlist_a(), playlist_b()]);
        // Track One (3000) + Track Two (4000) => 7000 > 5000, so only first track
        assert_eq!(result.tracks.len(), 1);
        assert!(result.total_duration_ms <= 5000);
    }

    #[test]
    fn test_empty_playlists() {
        let engine = PlaylistMergeEngine::new(MergeStrategy::Concatenate);
        let result = engine.merge(&[]);
        assert!(result.tracks.is_empty());
        assert_eq!(result.total_duration_ms, 0);
    }

    #[test]
    fn test_single_playlist() {
        let engine = PlaylistMergeEngine::new(MergeStrategy::Concatenate);
        let result = engine.merge(&[playlist_a()]);
        assert_eq!(result.tracks.len(), 2);
        assert_eq!(result.source_count, 1);
    }

    #[test]
    fn test_total_duration() {
        let engine = PlaylistMergeEngine::new(MergeStrategy::Concatenate);
        let result = engine.merge(&[playlist_a()]);
        assert_eq!(result.total_duration_ms, 7000);
    }

    #[test]
    fn test_distinct_track_count() {
        let count = PlaylistMergeEngine::distinct_track_count(&[playlist_a(), playlist_b()]);
        // ids: 1, 2, 3 => 3 distinct
        assert_eq!(count, 3);
    }

    #[test]
    fn test_strategy_accessor() {
        let engine = PlaylistMergeEngine::new(MergeStrategy::Interleave);
        assert_eq!(*engine.strategy(), MergeStrategy::Interleave);
    }

    #[test]
    fn test_merge_result_source_count() {
        let engine = PlaylistMergeEngine::new(MergeStrategy::Concatenate);
        let result = engine.merge(&[playlist_a(), playlist_b()]);
        assert_eq!(result.source_count, 2);
    }

    #[test]
    fn test_interleave_uneven_lengths() {
        let short = vec![MergeTrack::new("x", "X").with_duration_ms(100)];
        let long = vec![
            MergeTrack::new("a", "A").with_duration_ms(100),
            MergeTrack::new("b", "B").with_duration_ms(200),
            MergeTrack::new("c", "C").with_duration_ms(300),
        ];
        let engine = PlaylistMergeEngine::new(MergeStrategy::Interleave);
        let result = engine.merge(&[short, long]);
        assert_eq!(result.tracks.len(), 4);
    }
}
