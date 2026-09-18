//! Track locking and protection system.
//!
//! Provides mechanisms to lock tracks and clips against accidental
//! modifications during editing. Supports full lock, partial lock
//! (position-only, content-only), and clip-level pinning.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::too_many_arguments)]

use std::collections::{HashMap, HashSet};

/// Level of protection applied to a track or clip.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LockLevel {
    /// Fully unlocked; all operations allowed.
    Unlocked,
    /// Position is locked; clip cannot be moved or trimmed but effects/volume can change.
    PositionLocked,
    /// Content is locked; effects/volume cannot change but clip can be moved.
    ContentLocked,
    /// Fully locked; no modifications allowed.
    FullyLocked,
}

impl LockLevel {
    /// Whether position-changing operations are blocked.
    #[must_use]
    pub fn blocks_position(&self) -> bool {
        matches!(self, Self::PositionLocked | Self::FullyLocked)
    }

    /// Whether content-changing operations are blocked.
    #[must_use]
    pub fn blocks_content(&self) -> bool {
        matches!(self, Self::ContentLocked | Self::FullyLocked)
    }

    /// Whether any operations are blocked.
    #[must_use]
    pub fn is_locked(&self) -> bool {
        !matches!(self, Self::Unlocked)
    }
}

/// A lock applied to a specific track.
#[derive(Debug, Clone)]
pub struct TrackLock {
    /// Track index.
    pub track_index: u32,
    /// Lock level.
    pub level: LockLevel,
    /// Who set this lock (user ID, system, etc.).
    pub locked_by: String,
    /// Reason / note.
    pub reason: String,
}

impl TrackLock {
    /// Create a new track lock.
    #[must_use]
    pub fn new(track_index: u32, level: LockLevel, locked_by: &str) -> Self {
        Self {
            track_index,
            level,
            locked_by: locked_by.to_string(),
            reason: String::new(),
        }
    }

    /// Attach a reason.
    #[must_use]
    pub fn with_reason(mut self, reason: &str) -> Self {
        self.reason = reason.to_string();
        self
    }
}

/// A lock applied to a specific clip.
#[derive(Debug, Clone)]
pub struct ClipLock {
    /// Clip identifier.
    pub clip_id: u64,
    /// Lock level.
    pub level: LockLevel,
    /// Who set this lock.
    pub locked_by: String,
}

impl ClipLock {
    /// Create a new clip lock.
    #[must_use]
    pub fn new(clip_id: u64, level: LockLevel, locked_by: &str) -> Self {
        Self {
            clip_id,
            level,
            locked_by: locked_by.to_string(),
        }
    }
}

/// Result of checking whether an operation is permitted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LockCheckResult {
    /// Operation is allowed.
    Allowed,
    /// Operation is blocked by a track lock.
    BlockedByTrack {
        /// Track index.
        track: u32,
        /// Lock level.
        level: LockLevel,
    },
    /// Operation is blocked by a clip lock.
    BlockedByClip {
        /// Clip ID.
        clip_id: u64,
        /// Lock level.
        level: LockLevel,
    },
}

impl LockCheckResult {
    /// Whether the operation is allowed.
    #[must_use]
    pub fn is_allowed(&self) -> bool {
        matches!(self, Self::Allowed)
    }
}

/// Kind of editing operation being checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationKind {
    /// Moving or trimming a clip (position-changing).
    Move,
    /// Changing effects, volume, or content properties.
    ContentEdit,
    /// Deleting a clip.
    Delete,
    /// Adding a new clip to the track.
    Add,
}

impl OperationKind {
    /// Whether this operation changes position.
    #[must_use]
    pub fn is_position_change(&self) -> bool {
        matches!(self, Self::Move | Self::Delete)
    }

    /// Whether this operation changes content.
    #[must_use]
    pub fn is_content_change(&self) -> bool {
        matches!(self, Self::ContentEdit | Self::Delete)
    }
}

/// Manager for track and clip locks.
#[derive(Debug, Clone, Default)]
pub struct LockManager {
    /// Track locks keyed by track index.
    track_locks: HashMap<u32, TrackLock>,
    /// Clip locks keyed by clip ID.
    clip_locks: HashMap<u64, ClipLock>,
    /// Pinned clips (cannot be moved by ripple operations).
    pinned_clips: HashSet<u64>,
}

impl LockManager {
    /// Create a new lock manager.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Lock a track.
    pub fn lock_track(&mut self, lock: TrackLock) {
        self.track_locks.insert(lock.track_index, lock);
    }

    /// Unlock a track.
    pub fn unlock_track(&mut self, track_index: u32) -> bool {
        self.track_locks.remove(&track_index).is_some()
    }

    /// Get the lock for a track.
    #[must_use]
    pub fn get_track_lock(&self, track_index: u32) -> Option<&TrackLock> {
        self.track_locks.get(&track_index)
    }

    /// Lock a clip.
    pub fn lock_clip(&mut self, lock: ClipLock) {
        self.clip_locks.insert(lock.clip_id, lock);
    }

    /// Unlock a clip.
    pub fn unlock_clip(&mut self, clip_id: u64) -> bool {
        self.clip_locks.remove(&clip_id).is_some()
    }

    /// Get the lock for a clip.
    #[must_use]
    pub fn get_clip_lock(&self, clip_id: u64) -> Option<&ClipLock> {
        self.clip_locks.get(&clip_id)
    }

    /// Pin a clip (prevent ripple operations from moving it).
    pub fn pin_clip(&mut self, clip_id: u64) {
        self.pinned_clips.insert(clip_id);
    }

    /// Unpin a clip.
    pub fn unpin_clip(&mut self, clip_id: u64) -> bool {
        self.pinned_clips.remove(&clip_id)
    }

    /// Check if a clip is pinned.
    #[must_use]
    pub fn is_pinned(&self, clip_id: u64) -> bool {
        self.pinned_clips.contains(&clip_id)
    }

    /// Check whether an operation is permitted on a clip on a given track.
    #[must_use]
    pub fn check(&self, track_index: u32, clip_id: u64, op: OperationKind) -> LockCheckResult {
        // Check track lock first.
        if let Some(tl) = self.track_locks.get(&track_index) {
            if Self::does_lock_block(tl.level, op) {
                return LockCheckResult::BlockedByTrack {
                    track: track_index,
                    level: tl.level,
                };
            }
        }
        // Check clip lock.
        if let Some(cl) = self.clip_locks.get(&clip_id) {
            if Self::does_lock_block(cl.level, op) {
                return LockCheckResult::BlockedByClip {
                    clip_id,
                    level: cl.level,
                };
            }
        }
        LockCheckResult::Allowed
    }

    /// Determine if a lock level blocks a given operation kind.
    fn does_lock_block(level: LockLevel, op: OperationKind) -> bool {
        match level {
            LockLevel::Unlocked => false,
            LockLevel::FullyLocked => true,
            LockLevel::PositionLocked => op.is_position_change(),
            LockLevel::ContentLocked => op.is_content_change(),
        }
    }

    /// Count of locked tracks.
    #[must_use]
    pub fn locked_track_count(&self) -> usize {
        self.track_locks.len()
    }

    /// Count of locked clips.
    #[must_use]
    pub fn locked_clip_count(&self) -> usize {
        self.clip_locks.len()
    }

    /// Count of pinned clips.
    #[must_use]
    pub fn pinned_clip_count(&self) -> usize {
        self.pinned_clips.len()
    }

    /// Clear all locks and pins.
    pub fn clear_all(&mut self) {
        self.track_locks.clear();
        self.clip_locks.clear();
        self.pinned_clips.clear();
    }

    /// List all locked track indices.
    #[must_use]
    pub fn locked_tracks(&self) -> Vec<u32> {
        self.track_locks.keys().copied().collect()
    }

    /// List all locked clip IDs.
    #[must_use]
    pub fn locked_clips(&self) -> Vec<u64> {
        self.clip_locks.keys().copied().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lock_level_blocks_position() {
        assert!(!LockLevel::Unlocked.blocks_position());
        assert!(LockLevel::PositionLocked.blocks_position());
        assert!(!LockLevel::ContentLocked.blocks_position());
        assert!(LockLevel::FullyLocked.blocks_position());
    }

    #[test]
    fn test_lock_level_blocks_content() {
        assert!(!LockLevel::Unlocked.blocks_content());
        assert!(!LockLevel::PositionLocked.blocks_content());
        assert!(LockLevel::ContentLocked.blocks_content());
        assert!(LockLevel::FullyLocked.blocks_content());
    }

    #[test]
    fn test_lock_level_is_locked() {
        assert!(!LockLevel::Unlocked.is_locked());
        assert!(LockLevel::PositionLocked.is_locked());
        assert!(LockLevel::ContentLocked.is_locked());
        assert!(LockLevel::FullyLocked.is_locked());
    }

    #[test]
    fn test_track_lock_new() {
        let tl = TrackLock::new(0, LockLevel::FullyLocked, "alice").with_reason("final mix");
        assert_eq!(tl.track_index, 0);
        assert_eq!(tl.locked_by, "alice");
        assert_eq!(tl.reason, "final mix");
    }

    #[test]
    fn test_clip_lock_new() {
        let cl = ClipLock::new(42, LockLevel::ContentLocked, "bob");
        assert_eq!(cl.clip_id, 42);
        assert_eq!(cl.level, LockLevel::ContentLocked);
    }

    #[test]
    fn test_manager_lock_unlock_track() {
        let mut mgr = LockManager::new();
        mgr.lock_track(TrackLock::new(0, LockLevel::FullyLocked, "u"));
        assert_eq!(mgr.locked_track_count(), 1);
        assert!(mgr.get_track_lock(0).is_some());
        assert!(mgr.unlock_track(0));
        assert_eq!(mgr.locked_track_count(), 0);
    }

    #[test]
    fn test_manager_lock_unlock_clip() {
        let mut mgr = LockManager::new();
        mgr.lock_clip(ClipLock::new(10, LockLevel::PositionLocked, "u"));
        assert_eq!(mgr.locked_clip_count(), 1);
        assert!(mgr.get_clip_lock(10).is_some());
        assert!(mgr.unlock_clip(10));
        assert_eq!(mgr.locked_clip_count(), 0);
    }

    #[test]
    fn test_manager_pin_unpin() {
        let mut mgr = LockManager::new();
        mgr.pin_clip(5);
        assert!(mgr.is_pinned(5));
        assert_eq!(mgr.pinned_clip_count(), 1);
        assert!(mgr.unpin_clip(5));
        assert!(!mgr.is_pinned(5));
    }

    #[test]
    fn test_check_fully_locked_track() {
        let mut mgr = LockManager::new();
        mgr.lock_track(TrackLock::new(0, LockLevel::FullyLocked, "u"));
        let r = mgr.check(0, 1, OperationKind::Move);
        assert!(!r.is_allowed());
        assert!(matches!(
            r,
            LockCheckResult::BlockedByTrack { track: 0, .. }
        ));
    }

    #[test]
    fn test_check_position_locked_allows_content_edit() {
        let mut mgr = LockManager::new();
        mgr.lock_track(TrackLock::new(0, LockLevel::PositionLocked, "u"));
        // Content edit should be allowed.
        assert!(mgr.check(0, 1, OperationKind::ContentEdit).is_allowed());
        // Move should be blocked.
        assert!(!mgr.check(0, 1, OperationKind::Move).is_allowed());
    }

    #[test]
    fn test_check_content_locked_allows_move() {
        let mut mgr = LockManager::new();
        mgr.lock_track(TrackLock::new(0, LockLevel::ContentLocked, "u"));
        assert!(mgr.check(0, 1, OperationKind::Move).is_allowed());
        assert!(!mgr.check(0, 1, OperationKind::ContentEdit).is_allowed());
    }

    #[test]
    fn test_check_clip_lock_overrides() {
        let mut mgr = LockManager::new();
        // Track unlocked, but clip is fully locked.
        mgr.lock_clip(ClipLock::new(42, LockLevel::FullyLocked, "u"));
        let r = mgr.check(0, 42, OperationKind::Add);
        assert!(!r.is_allowed());
        assert!(matches!(
            r,
            LockCheckResult::BlockedByClip { clip_id: 42, .. }
        ));
    }

    #[test]
    fn test_check_unlocked() {
        let mgr = LockManager::new();
        assert!(mgr.check(0, 1, OperationKind::Move).is_allowed());
        assert!(mgr.check(0, 1, OperationKind::Delete).is_allowed());
    }

    #[test]
    fn test_clear_all() {
        let mut mgr = LockManager::new();
        mgr.lock_track(TrackLock::new(0, LockLevel::FullyLocked, "u"));
        mgr.lock_clip(ClipLock::new(1, LockLevel::FullyLocked, "u"));
        mgr.pin_clip(2);
        mgr.clear_all();
        assert_eq!(mgr.locked_track_count(), 0);
        assert_eq!(mgr.locked_clip_count(), 0);
        assert_eq!(mgr.pinned_clip_count(), 0);
    }

    #[test]
    fn test_operation_kind_classification() {
        assert!(OperationKind::Move.is_position_change());
        assert!(OperationKind::Delete.is_position_change());
        assert!(!OperationKind::ContentEdit.is_position_change());
        assert!(!OperationKind::Add.is_position_change());

        assert!(OperationKind::ContentEdit.is_content_change());
        assert!(OperationKind::Delete.is_content_change());
        assert!(!OperationKind::Move.is_content_change());
    }
}
