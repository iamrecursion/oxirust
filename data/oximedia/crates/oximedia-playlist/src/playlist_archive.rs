#![allow(dead_code)]
//! Playlist archiving, versioning, and retrieval.
//!
//! This module provides the ability to snapshot a playlist at a point in time,
//! store it in an in-memory archive, and later retrieve or compare versions.
//! Archived playlists are immutable and carry a monotonically increasing
//! version number together with an optional human-readable tag.

use std::collections::BTreeMap;
use std::time::{Duration, SystemTime};

// ---------------------------------------------------------------------------
// Archived item
// ---------------------------------------------------------------------------

/// A lightweight representation of a single playlist entry stored inside an
/// archived snapshot.
#[derive(Debug, Clone, PartialEq)]
pub struct ArchivedEntry {
    /// Asset identifier (file name or media ID).
    pub asset_id: String,
    /// Duration of the item.
    pub duration: Duration,
    /// Position index within the playlist.
    pub position: usize,
}

impl ArchivedEntry {
    /// Create a new archived entry.
    pub fn new(asset_id: impl Into<String>, duration: Duration, position: usize) -> Self {
        Self {
            asset_id: asset_id.into(),
            duration,
            position,
        }
    }
}

// ---------------------------------------------------------------------------
// Snapshot
// ---------------------------------------------------------------------------

/// An immutable snapshot of a playlist at a specific version.
#[derive(Debug, Clone)]
pub struct PlaylistSnapshot {
    /// Monotonically increasing version number.
    pub version: u64,
    /// Optional human-readable tag (e.g. "pre-live-switch").
    pub tag: Option<String>,
    /// Timestamp when the snapshot was taken.
    pub created_at: SystemTime,
    /// Ordered list of entries.
    entries: Vec<ArchivedEntry>,
}

impl PlaylistSnapshot {
    /// Create a new snapshot from a list of entries.
    pub fn new(version: u64, entries: Vec<ArchivedEntry>) -> Self {
        Self {
            version,
            tag: None,
            created_at: SystemTime::now(),
            entries,
        }
    }

    /// Attach a tag to this snapshot.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tag = Some(tag.into());
        self
    }

    /// Return the number of entries.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Return a reference to the entries slice.
    pub fn entries(&self) -> &[ArchivedEntry] {
        &self.entries
    }

    /// Total duration across all entries.
    pub fn total_duration(&self) -> Duration {
        self.entries.iter().map(|e| e.duration).sum()
    }

    /// Return the list of asset IDs in order.
    pub fn asset_ids(&self) -> Vec<&str> {
        self.entries.iter().map(|e| e.asset_id.as_str()).collect()
    }
}

// ---------------------------------------------------------------------------
// Diff
// ---------------------------------------------------------------------------

/// Describes a single difference between two snapshots.
#[derive(Debug, Clone, PartialEq)]
pub enum DiffEntry {
    /// An entry was added at the given position.
    Added(ArchivedEntry),
    /// An entry was removed from the given position.
    Removed(ArchivedEntry),
    /// An entry was moved from one position to another.
    Moved {
        /// The asset that moved.
        asset_id: String,
        /// Old position.
        from: usize,
        /// New position.
        to: usize,
    },
}

/// Result of comparing two snapshots.
#[derive(Debug, Clone)]
pub struct SnapshotDiff {
    /// Version of the older snapshot.
    pub from_version: u64,
    /// Version of the newer snapshot.
    pub to_version: u64,
    /// Individual differences.
    pub changes: Vec<DiffEntry>,
}

impl SnapshotDiff {
    /// Returns `true` when there are no differences.
    pub fn is_empty(&self) -> bool {
        self.changes.is_empty()
    }

    /// Number of changes.
    pub fn change_count(&self) -> usize {
        self.changes.len()
    }
}

// ---------------------------------------------------------------------------
// Archive
// ---------------------------------------------------------------------------

/// In-memory archive that stores multiple versioned playlist snapshots.
#[derive(Debug, Clone)]
pub struct PlaylistArchive {
    /// Name / identifier for this archive.
    pub name: String,
    /// Snapshots keyed by version number.
    snapshots: BTreeMap<u64, PlaylistSnapshot>,
    /// Next version to assign.
    next_version: u64,
    /// Maximum number of snapshots to retain (0 = unlimited).
    max_snapshots: usize,
}

impl PlaylistArchive {
    /// Create a new archive.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            snapshots: BTreeMap::new(),
            next_version: 1,
            max_snapshots: 0,
        }
    }

    /// Set a maximum snapshot retention count.
    pub fn with_max_snapshots(mut self, max: usize) -> Self {
        self.max_snapshots = max;
        self
    }

    /// Take a snapshot from the given entries and store it.
    /// Returns the assigned version number.
    pub fn snapshot(&mut self, entries: Vec<ArchivedEntry>) -> u64 {
        let version = self.next_version;
        self.next_version += 1;
        let snap = PlaylistSnapshot::new(version, entries);
        self.snapshots.insert(version, snap);
        self.enforce_retention();
        version
    }

    /// Take a snapshot with an explicit tag.
    pub fn snapshot_tagged(&mut self, entries: Vec<ArchivedEntry>, tag: impl Into<String>) -> u64 {
        let version = self.next_version;
        self.next_version += 1;
        let snap = PlaylistSnapshot::new(version, entries).with_tag(tag);
        self.snapshots.insert(version, snap);
        self.enforce_retention();
        version
    }

    /// Retrieve a snapshot by version number.
    pub fn get(&self, version: u64) -> Option<&PlaylistSnapshot> {
        self.snapshots.get(&version)
    }

    /// Retrieve the latest snapshot.
    pub fn latest(&self) -> Option<&PlaylistSnapshot> {
        self.snapshots.values().next_back()
    }

    /// Number of stored snapshots.
    pub fn snapshot_count(&self) -> usize {
        self.snapshots.len()
    }

    /// List all stored version numbers.
    pub fn versions(&self) -> Vec<u64> {
        self.snapshots.keys().copied().collect()
    }

    /// Delete a snapshot.
    pub fn delete(&mut self, version: u64) -> bool {
        self.snapshots.remove(&version).is_some()
    }

    /// Compute a simple diff between two versions.
    pub fn diff(&self, from: u64, to: u64) -> Option<SnapshotDiff> {
        let snap_from = self.snapshots.get(&from)?;
        let snap_to = self.snapshots.get(&to)?;
        let mut changes = Vec::new();

        let old_ids: Vec<&str> = snap_from.asset_ids().into_iter().collect();
        let new_ids: Vec<&str> = snap_to.asset_ids().into_iter().collect();

        // Detect removals
        for (i, entry) in snap_from.entries().iter().enumerate() {
            if !new_ids.contains(&entry.asset_id.as_str()) {
                changes.push(DiffEntry::Removed(snap_from.entries()[i].clone()));
            }
        }

        // Detect additions
        for (i, entry) in snap_to.entries().iter().enumerate() {
            if !old_ids.contains(&entry.asset_id.as_str()) {
                changes.push(DiffEntry::Added(snap_to.entries()[i].clone()));
            }
        }

        Some(SnapshotDiff {
            from_version: from,
            to_version: to,
            changes,
        })
    }

    // Internal: trim oldest snapshots when over the retention limit.
    fn enforce_retention(&mut self) {
        if self.max_snapshots == 0 {
            return;
        }
        while self.snapshots.len() > self.max_snapshots {
            if let Some(&oldest) = self.snapshots.keys().next() {
                self.snapshots.remove(&oldest);
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

    fn sample_entries(ids: &[&str]) -> Vec<ArchivedEntry> {
        ids.iter()
            .enumerate()
            .map(|(i, id)| ArchivedEntry::new(*id, Duration::from_secs(60), i))
            .collect()
    }

    #[test]
    fn test_snapshot_creation() {
        let entries = sample_entries(&["a", "b", "c"]);
        let snap = PlaylistSnapshot::new(1, entries);
        assert_eq!(snap.version, 1);
        assert_eq!(snap.entry_count(), 3);
        assert!(snap.tag.is_none());
    }

    #[test]
    fn test_snapshot_with_tag() {
        let snap = PlaylistSnapshot::new(1, vec![]).with_tag("release");
        assert_eq!(snap.tag.as_deref(), Some("release"));
    }

    #[test]
    fn test_snapshot_total_duration() {
        let entries = sample_entries(&["a", "b"]);
        let snap = PlaylistSnapshot::new(1, entries);
        assert_eq!(snap.total_duration(), Duration::from_secs(120));
    }

    #[test]
    fn test_snapshot_asset_ids() {
        let entries = sample_entries(&["x", "y"]);
        let snap = PlaylistSnapshot::new(1, entries);
        assert_eq!(snap.asset_ids(), vec!["x", "y"]);
    }

    #[test]
    fn test_archive_snapshot_increments_version() {
        let mut archive = PlaylistArchive::new("test");
        let v1 = archive.snapshot(sample_entries(&["a"]));
        let v2 = archive.snapshot(sample_entries(&["b"]));
        assert_eq!(v1, 1);
        assert_eq!(v2, 2);
        assert_eq!(archive.snapshot_count(), 2);
    }

    #[test]
    fn test_archive_get_latest() {
        let mut archive = PlaylistArchive::new("test");
        archive.snapshot(sample_entries(&["a"]));
        archive.snapshot(sample_entries(&["b"]));
        assert_eq!(archive.latest().expect("should succeed in test").version, 2);
    }

    #[test]
    fn test_archive_delete() {
        let mut archive = PlaylistArchive::new("test");
        archive.snapshot(sample_entries(&["a"]));
        assert!(archive.delete(1));
        assert!(!archive.delete(1));
        assert_eq!(archive.snapshot_count(), 0);
    }

    #[test]
    fn test_archive_versions_list() {
        let mut archive = PlaylistArchive::new("test");
        archive.snapshot(sample_entries(&["a"]));
        archive.snapshot(sample_entries(&["b"]));
        assert_eq!(archive.versions(), vec![1, 2]);
    }

    #[test]
    fn test_archive_retention_limit() {
        let mut archive = PlaylistArchive::new("test").with_max_snapshots(2);
        archive.snapshot(sample_entries(&["a"]));
        archive.snapshot(sample_entries(&["b"]));
        archive.snapshot(sample_entries(&["c"]));
        // Oldest should have been removed
        assert_eq!(archive.snapshot_count(), 2);
        assert!(archive.get(1).is_none());
        assert!(archive.get(2).is_some());
        assert!(archive.get(3).is_some());
    }

    #[test]
    fn test_diff_no_changes() {
        let mut archive = PlaylistArchive::new("test");
        let entries = sample_entries(&["a", "b"]);
        archive.snapshot(entries.clone());
        archive.snapshot(entries);
        let diff = archive.diff(1, 2).expect("should succeed in test");
        assert!(diff.is_empty());
    }

    #[test]
    fn test_diff_detects_addition() {
        let mut archive = PlaylistArchive::new("test");
        archive.snapshot(sample_entries(&["a"]));
        archive.snapshot(sample_entries(&["a", "b"]));
        let diff = archive.diff(1, 2).expect("should succeed in test");
        assert_eq!(diff.change_count(), 1);
        match &diff.changes[0] {
            DiffEntry::Added(e) => assert_eq!(e.asset_id, "b"),
            _ => panic!("expected Added"),
        }
    }

    #[test]
    fn test_diff_detects_removal() {
        let mut archive = PlaylistArchive::new("test");
        archive.snapshot(sample_entries(&["a", "b"]));
        archive.snapshot(sample_entries(&["a"]));
        let diff = archive.diff(1, 2).expect("should succeed in test");
        assert_eq!(diff.change_count(), 1);
        match &diff.changes[0] {
            DiffEntry::Removed(e) => assert_eq!(e.asset_id, "b"),
            _ => panic!("expected Removed"),
        }
    }

    #[test]
    fn test_archived_entry_new() {
        let entry = ArchivedEntry::new("clip.mxf", Duration::from_secs(30), 0);
        assert_eq!(entry.asset_id, "clip.mxf");
        assert_eq!(entry.duration, Duration::from_secs(30));
        assert_eq!(entry.position, 0);
    }
}
