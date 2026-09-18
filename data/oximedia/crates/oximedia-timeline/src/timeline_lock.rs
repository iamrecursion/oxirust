#![allow(dead_code)]
//! Timeline locking: prevent edits to ranges, tracks, or entire timelines.

/// Determines which parts of the timeline a lock covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LockScope {
    /// Locks a specific frame range on one track.
    Range,
    /// Locks an entire track (all frames).
    Track,
    /// Locks the entire timeline (all tracks, all frames).
    Everything,
}

impl LockScope {
    /// Returns `true` if this scope affects a given track.
    /// `Everything` and `Track` always do; `Range` depends on whether the range overlaps.
    #[must_use]
    pub fn affects_track(&self) -> bool {
        matches!(self, LockScope::Track | LockScope::Everything)
    }

    /// Human-readable description.
    #[must_use]
    pub fn description(&self) -> &'static str {
        match self {
            LockScope::Range => "Frame range locked",
            LockScope::Track => "Track locked",
            LockScope::Everything => "Everything locked",
        }
    }
}

/// A single lock entry covering a range of frames on an optional track.
#[derive(Debug, Clone)]
pub struct TimelineLock {
    /// Unique lock identifier.
    pub id: u64,
    /// Scope of this lock.
    pub scope: LockScope,
    /// Track index (relevant for `Range` and `Track` scopes; ignored for `Everything`).
    pub track_index: Option<usize>,
    /// Inclusive start frame.
    pub start_frame: i64,
    /// Exclusive end frame (`i64::MAX` for track/everything locks).
    pub end_frame: i64,
    /// Optional human-readable reason for the lock.
    pub reason: Option<String>,
}

impl TimelineLock {
    /// Creates a new lock.
    #[must_use]
    pub fn new(
        id: u64,
        scope: LockScope,
        track_index: Option<usize>,
        start_frame: i64,
        end_frame: i64,
    ) -> Self {
        Self {
            id,
            scope,
            track_index,
            start_frame,
            end_frame,
            reason: None,
        }
    }

    /// Attaches a reason string.
    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = Some(reason.into());
        self
    }

    /// Returns `true` if `frame` falls within the locked range `[start_frame, end_frame)`.
    #[must_use]
    pub fn is_locked_at(&self, frame: i64) -> bool {
        match self.scope {
            LockScope::Everything => true,
            LockScope::Track => true,
            LockScope::Range => frame >= self.start_frame && frame < self.end_frame,
        }
    }
}

/// Manages a set of timeline locks.
#[derive(Debug, Default)]
pub struct LockManager {
    locks: Vec<TimelineLock>,
    next_id: u64,
}

impl LockManager {
    /// Creates a new, empty lock manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            locks: Vec::new(),
            next_id: 1,
        }
    }

    /// Locks a specific frame range on `track_index`.
    /// Returns the new lock id.
    pub fn lock_range(&mut self, track_index: usize, start_frame: i64, end_frame: i64) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.locks.push(TimelineLock::new(
            id,
            LockScope::Range,
            Some(track_index),
            start_frame,
            end_frame,
        ));
        id
    }

    /// Locks an entire track.
    /// Returns the new lock id.
    pub fn lock_track(&mut self, track_index: usize) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.locks.push(TimelineLock::new(
            id,
            LockScope::Track,
            Some(track_index),
            0,
            i64::MAX,
        ));
        id
    }

    /// Locks the entire timeline.
    pub fn lock_all(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.locks.push(TimelineLock::new(
            id,
            LockScope::Everything,
            None,
            0,
            i64::MAX,
        ));
        id
    }

    /// Removes the lock with the given id. Returns `true` if a lock was removed.
    pub fn unlock(&mut self, id: u64) -> bool {
        let before = self.locks.len();
        self.locks.retain(|l| l.id != id);
        self.locks.len() < before
    }

    /// Returns all locks that cover `(track_index, frame)`.
    #[must_use]
    pub fn locked_regions(&self, track_index: usize, frame: i64) -> Vec<&TimelineLock> {
        self.locks
            .iter()
            .filter(|l| match l.scope {
                LockScope::Everything => l.is_locked_at(frame),
                LockScope::Track => l.track_index == Some(track_index),
                LockScope::Range => l.track_index == Some(track_index) && l.is_locked_at(frame),
            })
            .collect()
    }

    /// Returns `true` if any lock covers `(track_index, frame)`.
    #[must_use]
    pub fn is_locked(&self, track_index: usize, frame: i64) -> bool {
        !self.locked_regions(track_index, frame).is_empty()
    }

    /// Total number of active locks.
    #[must_use]
    pub fn count(&self) -> usize {
        self.locks.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_track_scope_affects_track() {
        assert!(LockScope::Track.affects_track());
    }

    #[test]
    fn test_everything_scope_affects_track() {
        assert!(LockScope::Everything.affects_track());
    }

    #[test]
    fn test_range_scope_not_blanket() {
        assert!(!LockScope::Range.affects_track());
    }

    #[test]
    fn test_scope_descriptions() {
        assert!(LockScope::Range.description().contains("range"));
        assert!(LockScope::Track.description().contains("Track"));
        assert!(LockScope::Everything.description().contains("Everything"));
    }

    #[test]
    fn test_timeline_lock_is_locked_at_range() {
        let lock = TimelineLock::new(1, LockScope::Range, Some(0), 50, 100);
        assert!(lock.is_locked_at(75));
        assert!(!lock.is_locked_at(25));
        assert!(!lock.is_locked_at(100)); // end is exclusive
    }

    #[test]
    fn test_timeline_lock_everything_always_locked() {
        let lock = TimelineLock::new(1, LockScope::Everything, None, 0, i64::MAX);
        assert!(lock.is_locked_at(0));
        assert!(lock.is_locked_at(999_999));
    }

    #[test]
    fn test_timeline_lock_with_reason() {
        let lock = TimelineLock::new(1, LockScope::Track, Some(0), 0, i64::MAX)
            .with_reason("Review in progress");
        assert_eq!(lock.reason.as_deref(), Some("Review in progress"));
    }

    #[test]
    fn test_manager_lock_range_and_query() {
        let mut mgr = LockManager::new();
        let id = mgr.lock_range(0, 100, 200);
        assert_eq!(id, 1);
        assert!(mgr.is_locked(0, 150));
        assert!(!mgr.is_locked(0, 50));
    }

    #[test]
    fn test_manager_lock_track() {
        let mut mgr = LockManager::new();
        mgr.lock_track(2);
        // Any frame on track 2 should be locked
        assert!(mgr.is_locked(2, 0));
        assert!(mgr.is_locked(2, 999_999));
        // Track 3 not locked
        assert!(!mgr.is_locked(3, 0));
    }

    #[test]
    fn test_manager_unlock() {
        let mut mgr = LockManager::new();
        let id = mgr.lock_range(0, 0, 100);
        assert!(mgr.is_locked(0, 50));
        assert!(mgr.unlock(id));
        assert!(!mgr.is_locked(0, 50));
    }

    #[test]
    fn test_manager_unlock_nonexistent_is_false() {
        let mut mgr = LockManager::new();
        assert!(!mgr.unlock(42));
    }

    #[test]
    fn test_manager_lock_all() {
        let mut mgr = LockManager::new();
        mgr.lock_all();
        assert!(mgr.is_locked(0, 0));
        assert!(mgr.is_locked(999, 50_000));
    }

    #[test]
    fn test_manager_count() {
        let mut mgr = LockManager::new();
        mgr.lock_range(0, 0, 100);
        mgr.lock_track(1);
        assert_eq!(mgr.count(), 2);
    }
}
