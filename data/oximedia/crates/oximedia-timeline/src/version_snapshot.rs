//! Timeline version snapshots (undo/history checkpoints).
//!
//! `VersionSnapshot` stores a lightweight description of a timeline state so
//! that operations can be undone or compared.  Full media data is never
//! duplicated; only structural metadata (track list, clip positions, etc.) is
//! serialised into the snapshot.

#![allow(dead_code)]

/// A record of a single clip's position in the timeline at snapshot time.
#[derive(Debug, Clone, PartialEq)]
pub struct ClipRecord {
    /// Clip identifier.
    pub clip_id: u64,
    /// Track this clip lives on.
    pub track_id: u32,
    /// Start frame on the outer timeline.
    pub start_frame: u64,
    /// End frame on the outer timeline (exclusive).
    pub end_frame: u64,
}

impl ClipRecord {
    /// Duration in frames.
    #[must_use]
    pub fn duration(&self) -> u64 {
        self.end_frame.saturating_sub(self.start_frame)
    }
}

/// A lightweight description of a track at snapshot time.
#[derive(Debug, Clone, PartialEq)]
pub struct TrackRecord {
    /// Track identifier.
    pub track_id: u32,
    /// Human-readable name.
    pub name: String,
    /// Whether the track was locked when the snapshot was taken.
    pub locked: bool,
    /// Whether the track was muted.
    pub muted: bool,
}

/// A saved snapshot of timeline structure.
#[derive(Debug, Clone)]
pub struct VersionSnapshot {
    /// Auto-incrementing snapshot number (1-based).
    pub version: u64,
    /// Human-readable label (e.g. operation that created this snapshot).
    pub label: String,
    /// Tracks as they were at snapshot time.
    pub tracks: Vec<TrackRecord>,
    /// Clip positions as they were at snapshot time.
    pub clips: Vec<ClipRecord>,
}

impl VersionSnapshot {
    /// Create a new snapshot.
    #[must_use]
    pub fn new(
        version: u64,
        label: impl Into<String>,
        tracks: Vec<TrackRecord>,
        clips: Vec<ClipRecord>,
    ) -> Self {
        Self {
            version,
            label: label.into(),
            tracks,
            clips,
        }
    }

    /// Total number of clips captured.
    #[must_use]
    pub fn clip_count(&self) -> usize {
        self.clips.len()
    }

    /// Total number of tracks captured.
    #[must_use]
    pub fn track_count(&self) -> usize {
        self.tracks.len()
    }

    /// Find a clip record by clip ID.
    #[must_use]
    pub fn find_clip(&self, clip_id: u64) -> Option<&ClipRecord> {
        self.clips.iter().find(|c| c.clip_id == clip_id)
    }

    /// Clips that belong to a specific track.
    #[must_use]
    pub fn clips_on_track(&self, track_id: u32) -> Vec<&ClipRecord> {
        self.clips
            .iter()
            .filter(|c| c.track_id == track_id)
            .collect()
    }

    /// Returns `true` when the snapshot has no clips and no tracks.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.clips.is_empty() && self.tracks.is_empty()
    }
}

/// Manages a stack of version snapshots supporting undo/redo.
#[derive(Debug)]
pub struct SnapshotHistory {
    snapshots: Vec<VersionSnapshot>,
    /// Index of the current position in history (points to the *applied*
    /// snapshot, or `None` if history is empty).
    cursor: Option<usize>,
    next_version: u64,
    /// Maximum snapshots to retain.
    max_depth: usize,
}

impl SnapshotHistory {
    /// Create a new history with the given maximum depth.
    #[must_use]
    pub fn new(max_depth: usize) -> Self {
        Self {
            snapshots: Vec::new(),
            cursor: None,
            next_version: 1,
            max_depth: max_depth.max(1),
        }
    }

    /// Push a new snapshot onto the history.  Any snapshots *after* the
    /// current cursor are discarded (redo branch is lost).
    pub fn push(
        &mut self,
        label: impl Into<String>,
        tracks: Vec<TrackRecord>,
        clips: Vec<ClipRecord>,
    ) -> u64 {
        // Trim redo branch.
        if let Some(idx) = self.cursor {
            self.snapshots.truncate(idx + 1);
        }

        let version = self.next_version;
        self.next_version += 1;
        self.snapshots
            .push(VersionSnapshot::new(version, label, tracks, clips));

        // Enforce max depth.
        if self.snapshots.len() > self.max_depth {
            self.snapshots.remove(0);
        }

        self.cursor = Some(self.snapshots.len().saturating_sub(1));
        version
    }

    /// Move the cursor one step back and return the snapshot to restore.
    pub fn undo(&mut self) -> Option<&VersionSnapshot> {
        let idx = self.cursor?;
        if idx == 0 {
            return None;
        }
        self.cursor = Some(idx - 1);
        self.snapshots.get(idx - 1)
    }

    /// Move the cursor one step forward and return the snapshot to apply.
    pub fn redo(&mut self) -> Option<&VersionSnapshot> {
        let idx = self.cursor?;
        let next = idx + 1;
        if next >= self.snapshots.len() {
            return None;
        }
        self.cursor = Some(next);
        self.snapshots.get(next)
    }

    /// Current snapshot (i.e. the applied state).
    #[must_use]
    pub fn current(&self) -> Option<&VersionSnapshot> {
        self.snapshots.get(self.cursor?)
    }

    /// Total snapshots stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.snapshots.len()
    }

    /// Returns `true` when there are no snapshots.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.snapshots.is_empty()
    }

    /// Returns `true` if an undo step is available.
    #[must_use]
    pub fn can_undo(&self) -> bool {
        self.cursor.is_some_and(|idx| idx > 0)
    }

    /// Returns `true` if a redo step is available.
    #[must_use]
    pub fn can_redo(&self) -> bool {
        self.cursor
            .is_some_and(|idx| idx + 1 < self.snapshots.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_clip(clip_id: u64, track_id: u32, start: u64, end: u64) -> ClipRecord {
        ClipRecord {
            clip_id,
            track_id,
            start_frame: start,
            end_frame: end,
        }
    }

    fn make_track(id: u32, name: &str) -> TrackRecord {
        TrackRecord {
            track_id: id,
            name: name.into(),
            locked: false,
            muted: false,
        }
    }

    #[test]
    fn clip_record_duration() {
        let c = make_clip(1, 0, 10, 25);
        assert_eq!(c.duration(), 15);
    }

    #[test]
    fn clip_record_duration_zero_when_inverted() {
        let c = make_clip(1, 0, 25, 10);
        assert_eq!(c.duration(), 0);
    }

    #[test]
    fn snapshot_clip_count() {
        let snap = VersionSnapshot::new(1, "init", vec![], vec![make_clip(1, 0, 0, 10)]);
        assert_eq!(snap.clip_count(), 1);
    }

    #[test]
    fn snapshot_track_count() {
        let snap = VersionSnapshot::new(1, "init", vec![make_track(0, "V1")], vec![]);
        assert_eq!(snap.track_count(), 1);
    }

    #[test]
    fn snapshot_find_clip() {
        let snap = VersionSnapshot::new(1, "s", vec![], vec![make_clip(42, 0, 0, 5)]);
        assert!(snap.find_clip(42).is_some());
        assert!(snap.find_clip(99).is_none());
    }

    #[test]
    fn snapshot_clips_on_track() {
        let clips = vec![
            make_clip(1, 0, 0, 5),
            make_clip(2, 1, 5, 10),
            make_clip(3, 0, 10, 15),
        ];
        let snap = VersionSnapshot::new(1, "s", vec![], clips);
        assert_eq!(snap.clips_on_track(0).len(), 2);
        assert_eq!(snap.clips_on_track(1).len(), 1);
    }

    #[test]
    fn snapshot_is_empty() {
        let snap = VersionSnapshot::new(1, "s", vec![], vec![]);
        assert!(snap.is_empty());
    }

    #[test]
    fn history_push_increments_version() {
        let mut h = SnapshotHistory::new(10);
        let v1 = h.push("a", vec![], vec![]);
        let v2 = h.push("b", vec![], vec![]);
        assert_eq!(v2, v1 + 1);
    }

    #[test]
    fn history_undo_redo_basic() {
        let mut h = SnapshotHistory::new(10);
        h.push("v1", vec![], vec![make_clip(1, 0, 0, 10)]);
        h.push("v2", vec![], vec![make_clip(1, 0, 0, 20)]);
        let undone = h.undo().expect("should succeed in test");
        assert_eq!(undone.label, "v1");
        let redone = h.redo().expect("should succeed in test");
        assert_eq!(redone.label, "v2");
    }

    #[test]
    fn history_undo_at_start_returns_none() {
        let mut h = SnapshotHistory::new(10);
        h.push("v1", vec![], vec![]);
        assert!(h.undo().is_none()); // already at oldest
    }

    #[test]
    fn history_redo_at_end_returns_none() {
        let mut h = SnapshotHistory::new(10);
        h.push("v1", vec![], vec![]);
        assert!(h.redo().is_none());
    }

    #[test]
    fn history_max_depth_enforced() {
        let mut h = SnapshotHistory::new(3);
        h.push("a", vec![], vec![]);
        h.push("b", vec![], vec![]);
        h.push("c", vec![], vec![]);
        h.push("d", vec![], vec![]);
        assert_eq!(h.len(), 3);
    }

    #[test]
    fn history_can_undo_can_redo() {
        let mut h = SnapshotHistory::new(10);
        assert!(!h.can_undo());
        assert!(!h.can_redo());
        h.push("a", vec![], vec![]);
        h.push("b", vec![], vec![]);
        assert!(h.can_undo());
        h.undo();
        assert!(h.can_redo());
    }
}
