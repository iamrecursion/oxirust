//! EDL change tracking with version history.
//!
//! This module maintains a linear version history of an EDL, recording
//! snapshots and change descriptions so that users can review, compare,
//! and roll back EDL modifications.

#![allow(dead_code)]

use crate::edl_changelist::{diff_events, Changelist};
use crate::event::EdlEvent;
use crate::timecode::EdlFrameRate;

/// A single snapshot of the EDL state.
#[derive(Debug, Clone)]
pub struct EdlSnapshot {
    /// Version number (1-indexed, monotonically increasing).
    pub version: u32,
    /// Human-readable description of the change.
    pub description: String,
    /// Timestamp of the snapshot (seconds since epoch, or relative).
    pub timestamp: u64,
    /// Title at this version.
    pub title: Option<String>,
    /// Frame rate at this version.
    pub frame_rate: EdlFrameRate,
    /// Cloned events at this version.
    pub events: Vec<EdlEvent>,
}

/// Version history container for an EDL.
#[derive(Debug)]
pub struct VersionHistory {
    /// All snapshots, ordered by version number.
    snapshots: Vec<EdlSnapshot>,
    /// Next version number to assign.
    next_version: u32,
    /// Maximum number of snapshots to keep (0 = unlimited).
    max_snapshots: usize,
}

impl VersionHistory {
    /// Create a new empty version history.
    #[must_use]
    pub fn new() -> Self {
        Self {
            snapshots: Vec::new(),
            next_version: 1,
            max_snapshots: 0,
        }
    }

    /// Create a version history with a maximum snapshot limit.
    #[must_use]
    pub fn with_max_snapshots(max: usize) -> Self {
        Self {
            snapshots: Vec::new(),
            next_version: 1,
            max_snapshots: max,
        }
    }

    /// Record a new snapshot.
    pub fn record(
        &mut self,
        description: impl Into<String>,
        timestamp: u64,
        title: Option<String>,
        frame_rate: EdlFrameRate,
        events: Vec<EdlEvent>,
    ) -> u32 {
        let version = self.next_version;
        self.next_version += 1;

        self.snapshots.push(EdlSnapshot {
            version,
            description: description.into(),
            timestamp,
            title,
            frame_rate,
            events,
        });

        // Enforce max snapshots limit (keep most recent)
        if self.max_snapshots > 0 && self.snapshots.len() > self.max_snapshots {
            let excess = self.snapshots.len() - self.max_snapshots;
            self.snapshots.drain(..excess);
        }

        version
    }

    /// Get total number of snapshots.
    #[must_use]
    pub fn snapshot_count(&self) -> usize {
        self.snapshots.len()
    }

    /// Check if history is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.snapshots.is_empty()
    }

    /// Get the latest snapshot.
    #[must_use]
    pub fn latest(&self) -> Option<&EdlSnapshot> {
        self.snapshots.last()
    }

    /// Get a snapshot by version number.
    #[must_use]
    pub fn get_version(&self, version: u32) -> Option<&EdlSnapshot> {
        self.snapshots.iter().find(|s| s.version == version)
    }

    /// Get all version numbers.
    #[must_use]
    pub fn versions(&self) -> Vec<u32> {
        self.snapshots.iter().map(|s| s.version).collect()
    }

    /// Get a summary of all snapshots (version, description, event count).
    #[must_use]
    pub fn summary(&self) -> Vec<VersionSummary> {
        self.snapshots
            .iter()
            .map(|s| VersionSummary {
                version: s.version,
                description: s.description.clone(),
                timestamp: s.timestamp,
                event_count: s.events.len(),
            })
            .collect()
    }

    /// Compare two versions and produce a changelist.
    ///
    /// Returns `None` if either version is not found.
    #[must_use]
    pub fn diff(&self, version_a: u32, version_b: u32) -> Option<Changelist> {
        let a = self.get_version(version_a)?;
        let b = self.get_version(version_b)?;
        Some(diff_events(&a.events, &b.events))
    }

    /// Get the events from a specific version.
    #[must_use]
    pub fn events_at(&self, version: u32) -> Option<&[EdlEvent]> {
        self.get_version(version).map(|s| s.events.as_slice())
    }

    /// Get the latest version number (or 0 if no snapshots).
    #[must_use]
    pub fn latest_version(&self) -> u32 {
        self.snapshots.last().map_or(0, |s| s.version)
    }

    /// Clear all history.
    pub fn clear(&mut self) {
        self.snapshots.clear();
        self.next_version = 1;
    }
}

impl Default for VersionHistory {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary information about a version.
#[derive(Debug, Clone)]
pub struct VersionSummary {
    /// Version number.
    pub version: u32,
    /// Description.
    pub description: String,
    /// Timestamp.
    pub timestamp: u64,
    /// Number of events.
    pub event_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::{EditType, TrackType};
    use crate::timecode::EdlTimecode;

    fn make_tc(h: u8, m: u8, s: u8, f: u8) -> EdlTimecode {
        EdlTimecode::new(h, m, s, f, EdlFrameRate::Fps25).expect("failed to create")
    }

    fn make_event(num: u32, reel: &str) -> EdlEvent {
        let tc1 = make_tc(1, 0, 0, 0);
        let tc2 = make_tc(1, 0, 5, 0);
        EdlEvent::new(
            num,
            reel.to_string(),
            TrackType::Video,
            EditType::Cut,
            tc1,
            tc2,
            tc1,
            tc2,
        )
    }

    #[test]
    fn test_version_history_new() {
        let vh = VersionHistory::new();
        assert!(vh.is_empty());
        assert_eq!(vh.snapshot_count(), 0);
        assert_eq!(vh.latest_version(), 0);
    }

    #[test]
    fn test_record_snapshot() {
        let mut vh = VersionHistory::new();
        let v = vh.record(
            "Initial version",
            1000,
            Some("Test EDL".to_string()),
            EdlFrameRate::Fps25,
            vec![make_event(1, "A001")],
        );
        assert_eq!(v, 1);
        assert_eq!(vh.snapshot_count(), 1);
        assert_eq!(vh.latest_version(), 1);
    }

    #[test]
    fn test_record_multiple_snapshots() {
        let mut vh = VersionHistory::new();
        vh.record(
            "v1",
            1000,
            None,
            EdlFrameRate::Fps25,
            vec![make_event(1, "A001")],
        );
        vh.record(
            "v2",
            2000,
            None,
            EdlFrameRate::Fps25,
            vec![make_event(1, "A001"), make_event(2, "A002")],
        );
        vh.record(
            "v3",
            3000,
            None,
            EdlFrameRate::Fps25,
            vec![make_event(1, "B001")],
        );

        assert_eq!(vh.snapshot_count(), 3);
        assert_eq!(vh.latest_version(), 3);
        assert_eq!(vh.versions(), vec![1, 2, 3]);
    }

    #[test]
    fn test_get_version() {
        let mut vh = VersionHistory::new();
        vh.record(
            "v1",
            1000,
            None,
            EdlFrameRate::Fps25,
            vec![make_event(1, "A001")],
        );
        vh.record(
            "v2",
            2000,
            None,
            EdlFrameRate::Fps25,
            vec![make_event(1, "B001")],
        );

        let snap = vh.get_version(1).expect("version 1 should exist");
        assert_eq!(snap.events[0].reel, "A001");

        let snap2 = vh.get_version(2).expect("version 2 should exist");
        assert_eq!(snap2.events[0].reel, "B001");

        assert!(vh.get_version(99).is_none());
    }

    #[test]
    fn test_diff_versions() {
        let mut vh = VersionHistory::new();
        vh.record(
            "v1",
            1000,
            None,
            EdlFrameRate::Fps25,
            vec![make_event(1, "A001")],
        );
        vh.record(
            "v2",
            2000,
            None,
            EdlFrameRate::Fps25,
            vec![make_event(1, "A001"), make_event(2, "A002")],
        );

        let cl = vh.diff(1, 2).expect("diff should succeed");
        assert_eq!(cl.added_count(), 1);
        assert_eq!(cl.removed_count(), 0);
    }

    #[test]
    fn test_diff_nonexistent_version() {
        let vh = VersionHistory::new();
        assert!(vh.diff(1, 2).is_none());
    }

    #[test]
    fn test_events_at() {
        let mut vh = VersionHistory::new();
        vh.record(
            "v1",
            1000,
            None,
            EdlFrameRate::Fps25,
            vec![make_event(1, "A001")],
        );
        let events = vh.events_at(1).expect("events should exist");
        assert_eq!(events.len(), 1);
        assert!(vh.events_at(99).is_none());
    }

    #[test]
    fn test_latest() {
        let mut vh = VersionHistory::new();
        assert!(vh.latest().is_none());

        vh.record("first", 1000, None, EdlFrameRate::Fps25, vec![]);
        vh.record("second", 2000, None, EdlFrameRate::Fps25, vec![]);

        let latest = vh.latest().expect("latest should exist");
        assert_eq!(latest.description, "second");
        assert_eq!(latest.version, 2);
    }

    #[test]
    fn test_summary() {
        let mut vh = VersionHistory::new();
        vh.record(
            "Initial",
            1000,
            None,
            EdlFrameRate::Fps25,
            vec![make_event(1, "A001")],
        );
        vh.record(
            "Added clip",
            2000,
            None,
            EdlFrameRate::Fps25,
            vec![make_event(1, "A001"), make_event(2, "A002")],
        );

        let summaries = vh.summary();
        assert_eq!(summaries.len(), 2);
        assert_eq!(summaries[0].event_count, 1);
        assert_eq!(summaries[1].event_count, 2);
        assert_eq!(summaries[0].description, "Initial");
    }

    #[test]
    fn test_max_snapshots() {
        let mut vh = VersionHistory::with_max_snapshots(3);
        for i in 0..5 {
            vh.record(
                format!("v{}", i + 1),
                (i as u64 + 1) * 1000,
                None,
                EdlFrameRate::Fps25,
                vec![],
            );
        }
        assert_eq!(vh.snapshot_count(), 3);
        // Oldest two should be gone
        assert!(vh.get_version(1).is_none());
        assert!(vh.get_version(2).is_none());
        assert!(vh.get_version(3).is_some());
        assert!(vh.get_version(5).is_some());
    }

    #[test]
    fn test_clear() {
        let mut vh = VersionHistory::new();
        vh.record("v1", 1000, None, EdlFrameRate::Fps25, vec![]);
        vh.record("v2", 2000, None, EdlFrameRate::Fps25, vec![]);
        vh.clear();
        assert!(vh.is_empty());
        assert_eq!(vh.latest_version(), 0);

        // After clear, new versions start at 1
        let v = vh.record("fresh", 3000, None, EdlFrameRate::Fps25, vec![]);
        assert_eq!(v, 1);
    }

    #[test]
    fn test_default() {
        let vh = VersionHistory::default();
        assert!(vh.is_empty());
    }
}
