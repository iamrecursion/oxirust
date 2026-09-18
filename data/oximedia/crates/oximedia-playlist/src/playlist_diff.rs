#![allow(dead_code)]

//! Playlist diffing engine for comparing two playlists and detecting changes.
//!
//! This module computes structural differences between playlist versions,
//! producing a list of change operations (add, remove, move, modify)
//! that can be applied to transform one playlist into another.

use std::collections::{HashMap, HashSet};

/// The kind of change detected between two playlists.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiffKind {
    /// A track was added.
    Added,
    /// A track was removed.
    Removed,
    /// A track was moved to a different position.
    Moved,
    /// A track's metadata (title, duration, etc.) was modified in place.
    Modified,
    /// No change.
    Unchanged,
}

/// Represents a single item in a playlist snapshot.
#[derive(Debug, Clone, PartialEq)]
pub struct DiffEntry {
    /// Unique identifier for the track.
    pub id: String,
    /// Display title.
    pub title: String,
    /// Duration in milliseconds.
    pub duration_ms: u64,
    /// Position index in the playlist.
    pub position: usize,
    /// Optional metadata hash for quick equality checks.
    pub metadata_hash: u64,
}

impl DiffEntry {
    /// Create a new diff entry.
    pub fn new(id: &str, title: &str, duration_ms: u64, position: usize) -> Self {
        Self {
            id: id.to_string(),
            title: title.to_string(),
            duration_ms,
            position,
            metadata_hash: 0,
        }
    }

    /// Set the metadata hash value.
    pub fn with_metadata_hash(mut self, hash: u64) -> Self {
        self.metadata_hash = hash;
        self
    }
}

/// A single change record between two playlist versions.
#[derive(Debug, Clone, PartialEq)]
pub struct DiffChange {
    /// The kind of change.
    pub kind: DiffKind,
    /// Track id affected.
    pub track_id: String,
    /// Position in the old playlist (if applicable).
    pub old_position: Option<usize>,
    /// Position in the new playlist (if applicable).
    pub new_position: Option<usize>,
    /// Human-readable description of the change.
    pub description: String,
}

/// Summary statistics for a playlist diff.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiffSummary {
    /// Number of tracks added.
    pub added: usize,
    /// Number of tracks removed.
    pub removed: usize,
    /// Number of tracks moved.
    pub moved: usize,
    /// Number of tracks modified in place.
    pub modified: usize,
    /// Number of tracks unchanged.
    pub unchanged: usize,
    /// Total number of changes.
    pub total_changes: usize,
}

/// Engine that computes diffs between two playlist snapshots.
#[derive(Debug)]
pub struct PlaylistDiffEngine;

impl PlaylistDiffEngine {
    /// Compute the diff between an old and new playlist version.
    pub fn diff(old: &[DiffEntry], new: &[DiffEntry]) -> Vec<DiffChange> {
        let mut changes = Vec::new();

        let old_map: HashMap<&str, &DiffEntry> = old.iter().map(|e| (e.id.as_str(), e)).collect();
        let new_map: HashMap<&str, &DiffEntry> = new.iter().map(|e| (e.id.as_str(), e)).collect();

        let old_ids: HashSet<&str> = old.iter().map(|e| e.id.as_str()).collect();
        let new_ids: HashSet<&str> = new.iter().map(|e| e.id.as_str()).collect();

        // Removed tracks
        for &id in old_ids.difference(&new_ids) {
            if let Some(entry) = old_map.get(id) {
                changes.push(DiffChange {
                    kind: DiffKind::Removed,
                    track_id: id.to_string(),
                    old_position: Some(entry.position),
                    new_position: None,
                    description: format!(
                        "Track '{}' removed from position {}",
                        entry.title, entry.position
                    ),
                });
            }
        }

        // Added tracks
        for &id in new_ids.difference(&old_ids) {
            if let Some(entry) = new_map.get(id) {
                changes.push(DiffChange {
                    kind: DiffKind::Added,
                    track_id: id.to_string(),
                    old_position: None,
                    new_position: Some(entry.position),
                    description: format!(
                        "Track '{}' added at position {}",
                        entry.title, entry.position
                    ),
                });
            }
        }

        // Tracks in both — check for moves and modifications
        for &id in old_ids.intersection(&new_ids) {
            let old_entry = old_map[id];
            let new_entry = new_map[id];

            let position_changed = old_entry.position != new_entry.position;
            let content_changed = old_entry.title != new_entry.title
                || old_entry.duration_ms != new_entry.duration_ms
                || (old_entry.metadata_hash != 0
                    && new_entry.metadata_hash != 0
                    && old_entry.metadata_hash != new_entry.metadata_hash);

            if content_changed {
                changes.push(DiffChange {
                    kind: DiffKind::Modified,
                    track_id: id.to_string(),
                    old_position: Some(old_entry.position),
                    new_position: Some(new_entry.position),
                    description: format!("Track '{}' modified", old_entry.title),
                });
            } else if position_changed {
                changes.push(DiffChange {
                    kind: DiffKind::Moved,
                    track_id: id.to_string(),
                    old_position: Some(old_entry.position),
                    new_position: Some(new_entry.position),
                    description: format!(
                        "Track '{}' moved from {} to {}",
                        old_entry.title, old_entry.position, new_entry.position
                    ),
                });
            }
        }

        // Sort by new_position (or old_position for removals)
        changes.sort_by_key(|c| c.new_position.or(c.old_position).unwrap_or(usize::MAX));
        changes
    }

    /// Produce a summary of a diff result.
    pub fn summarize(changes: &[DiffChange]) -> DiffSummary {
        let mut added = 0;
        let mut removed = 0;
        let mut moved = 0;
        let mut modified = 0;
        let unchanged = 0;

        for c in changes {
            match c.kind {
                DiffKind::Added => added += 1,
                DiffKind::Removed => removed += 1,
                DiffKind::Moved => moved += 1,
                DiffKind::Modified => modified += 1,
                DiffKind::Unchanged => {}
            }
        }

        let total_changes = added + removed + moved + modified;

        DiffSummary {
            added,
            removed,
            moved,
            modified,
            unchanged,
            total_changes,
        }
    }

    /// Return true if the two playlists are identical.
    pub fn is_identical(old: &[DiffEntry], new: &[DiffEntry]) -> bool {
        if old.len() != new.len() {
            return false;
        }
        for (a, b) in old.iter().zip(new.iter()) {
            if a.id != b.id
                || a.position != b.position
                || a.title != b.title
                || a.duration_ms != b.duration_ms
            {
                return false;
            }
        }
        true
    }

    /// Compute a simple similarity ratio (0.0 to 1.0) between two playlists.
    #[allow(clippy::cast_precision_loss)]
    pub fn similarity(old: &[DiffEntry], new: &[DiffEntry]) -> f64 {
        let old_ids: HashSet<&str> = old.iter().map(|e| e.id.as_str()).collect();
        let new_ids: HashSet<&str> = new.iter().map(|e| e.id.as_str()).collect();
        let union_count = old_ids.union(&new_ids).count();
        if union_count == 0 {
            return 1.0;
        }
        let intersection_count = old_ids.intersection(&new_ids).count();
        intersection_count as f64 / union_count as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn old_playlist() -> Vec<DiffEntry> {
        vec![
            DiffEntry::new("a", "Song A", 3000, 0),
            DiffEntry::new("b", "Song B", 4000, 1),
            DiffEntry::new("c", "Song C", 2500, 2),
        ]
    }

    fn new_playlist_with_add_remove() -> Vec<DiffEntry> {
        vec![
            DiffEntry::new("a", "Song A", 3000, 0),
            DiffEntry::new("d", "Song D", 5000, 1),
            DiffEntry::new("c", "Song C", 2500, 2),
        ]
    }

    #[test]
    fn test_identical_playlists() {
        let old = old_playlist();
        let new = old_playlist();
        assert!(PlaylistDiffEngine::is_identical(&old, &new));
    }

    #[test]
    fn test_different_length_not_identical() {
        let old = old_playlist();
        let new = vec![DiffEntry::new("a", "Song A", 3000, 0)];
        assert!(!PlaylistDiffEngine::is_identical(&old, &new));
    }

    #[test]
    fn test_add_and_remove() {
        let old = old_playlist();
        let new = new_playlist_with_add_remove();
        let changes = PlaylistDiffEngine::diff(&old, &new);
        let summary = PlaylistDiffEngine::summarize(&changes);
        assert_eq!(summary.added, 1);
        assert_eq!(summary.removed, 1);
    }

    #[test]
    fn test_moved_track() {
        let old = old_playlist();
        let new = vec![
            DiffEntry::new("c", "Song C", 2500, 0),
            DiffEntry::new("a", "Song A", 3000, 1),
            DiffEntry::new("b", "Song B", 4000, 2),
        ];
        let changes = PlaylistDiffEngine::diff(&old, &new);
        let summary = PlaylistDiffEngine::summarize(&changes);
        assert_eq!(summary.moved, 3);
    }

    #[test]
    fn test_modified_track() {
        let old = old_playlist();
        let new = vec![
            DiffEntry::new("a", "Song A Remix", 3000, 0),
            DiffEntry::new("b", "Song B", 4000, 1),
            DiffEntry::new("c", "Song C", 2500, 2),
        ];
        let changes = PlaylistDiffEngine::diff(&old, &new);
        let summary = PlaylistDiffEngine::summarize(&changes);
        assert_eq!(summary.modified, 1);
    }

    #[test]
    fn test_modified_duration() {
        let old = old_playlist();
        let new = vec![
            DiffEntry::new("a", "Song A", 9999, 0),
            DiffEntry::new("b", "Song B", 4000, 1),
            DiffEntry::new("c", "Song C", 2500, 2),
        ];
        let changes = PlaylistDiffEngine::diff(&old, &new);
        let has_modified = changes.iter().any(|c| c.kind == DiffKind::Modified);
        assert!(has_modified);
    }

    #[test]
    fn test_empty_diff() {
        let changes = PlaylistDiffEngine::diff(&[], &[]);
        assert!(changes.is_empty());
    }

    #[test]
    fn test_all_added() {
        let new = old_playlist();
        let changes = PlaylistDiffEngine::diff(&[], &new);
        let summary = PlaylistDiffEngine::summarize(&changes);
        assert_eq!(summary.added, 3);
        assert_eq!(summary.removed, 0);
    }

    #[test]
    fn test_all_removed() {
        let old = old_playlist();
        let changes = PlaylistDiffEngine::diff(&old, &[]);
        let summary = PlaylistDiffEngine::summarize(&changes);
        assert_eq!(summary.removed, 3);
        assert_eq!(summary.added, 0);
    }

    #[test]
    fn test_similarity_identical() {
        let a = old_playlist();
        let b = old_playlist();
        let sim = PlaylistDiffEngine::similarity(&a, &b);
        assert!((sim - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_similarity_disjoint() {
        let a = vec![DiffEntry::new("x", "X", 100, 0)];
        let b = vec![DiffEntry::new("y", "Y", 200, 0)];
        let sim = PlaylistDiffEngine::similarity(&a, &b);
        assert!((sim - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_similarity_partial() {
        let a = old_playlist();
        let b = new_playlist_with_add_remove();
        let sim = PlaylistDiffEngine::similarity(&a, &b);
        // a,b,c vs a,d,c => intersection {a,c}=2, union {a,b,c,d}=4 => 0.5
        assert!((sim - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_similarity_empty() {
        let sim = PlaylistDiffEngine::similarity(&[], &[]);
        assert!((sim - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_diff_change_description() {
        let old = old_playlist();
        let new = new_playlist_with_add_remove();
        let changes = PlaylistDiffEngine::diff(&old, &new);
        for c in &changes {
            assert!(!c.description.is_empty());
        }
    }

    #[test]
    fn test_metadata_hash_change() {
        let old = vec![DiffEntry::new("a", "Song A", 3000, 0).with_metadata_hash(111)];
        let new = vec![DiffEntry::new("a", "Song A", 3000, 0).with_metadata_hash(222)];
        let changes = PlaylistDiffEngine::diff(&old, &new);
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].kind, DiffKind::Modified);
    }

    #[test]
    fn test_summary_total() {
        let old = old_playlist();
        let new = new_playlist_with_add_remove();
        let changes = PlaylistDiffEngine::diff(&old, &new);
        let summary = PlaylistDiffEngine::summarize(&changes);
        assert_eq!(
            summary.total_changes,
            summary.added + summary.removed + summary.moved + summary.modified
        );
    }
}
