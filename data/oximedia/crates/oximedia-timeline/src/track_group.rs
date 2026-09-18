//! Track group management for the timeline.
//!
//! `TrackGroup` collects related tracks (e.g. a video track together with its
//! audio stems) so that operations such as lock, mute or solo can be applied
//! atomically to the whole group.

#![allow(dead_code)]

/// The visibility / audibility state applied to a group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupState {
    /// All tracks in the group are active.
    Active,
    /// All tracks are muted (audio silenced / video hidden).
    Muted,
    /// All tracks are locked (no edits allowed).
    Locked,
    /// All tracks are both muted and locked.
    MutedAndLocked,
}

impl GroupState {
    /// Returns `true` when audio and video from this group are suppressed.
    #[must_use]
    pub fn is_muted(self) -> bool {
        matches!(self, Self::Muted | Self::MutedAndLocked)
    }

    /// Returns `true` when no edits are permitted.
    #[must_use]
    pub fn is_locked(self) -> bool {
        matches!(self, Self::Locked | Self::MutedAndLocked)
    }
}

/// A named group of track IDs.
#[derive(Debug, Clone)]
pub struct TrackGroup {
    /// Unique identifier for this group.
    pub id: u32,
    /// Human-readable label.
    pub name: String,
    /// IDs of tracks belonging to this group.
    pub track_ids: Vec<u32>,
    /// Current state of the group.
    pub state: GroupState,
    /// Colour hint for the UI (e.g. `"#FF5733"`).
    pub color: Option<String>,
}

impl TrackGroup {
    /// Create a new, empty track group.
    #[must_use]
    pub fn new(id: u32, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            track_ids: Vec::new(),
            state: GroupState::Active,
            color: None,
        }
    }

    /// Add a track to the group. Returns `false` if the track is already a
    /// member.
    pub fn add_track(&mut self, track_id: u32) -> bool {
        if self.track_ids.contains(&track_id) {
            return false;
        }
        self.track_ids.push(track_id);
        true
    }

    /// Remove a track from the group. Returns `true` if the track was found
    /// and removed.
    pub fn remove_track(&mut self, track_id: u32) -> bool {
        let before = self.track_ids.len();
        self.track_ids.retain(|&id| id != track_id);
        self.track_ids.len() < before
    }

    /// Returns `true` when the given track belongs to this group.
    #[must_use]
    pub fn contains(&self, track_id: u32) -> bool {
        self.track_ids.contains(&track_id)
    }

    /// Number of tracks in the group.
    #[must_use]
    pub fn len(&self) -> usize {
        self.track_ids.len()
    }

    /// Returns `true` when the group has no member tracks.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.track_ids.is_empty()
    }

    /// Apply a new state to the group.
    pub fn set_state(&mut self, state: GroupState) {
        self.state = state;
    }

    /// Returns `true` when editing is permitted for this group.
    #[must_use]
    pub fn is_editable(&self) -> bool {
        !self.state.is_locked()
    }
}

/// Registry that manages all track groups in a timeline.
#[derive(Debug, Default)]
pub struct TrackGroupRegistry {
    groups: Vec<TrackGroup>,
    next_id: u32,
}

impl TrackGroupRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new group with an auto-assigned ID and return that ID.
    pub fn create_group(&mut self, name: impl Into<String>) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        self.groups.push(TrackGroup::new(id, name));
        id
    }

    /// Look up a group by ID.
    #[must_use]
    pub fn find(&self, id: u32) -> Option<&TrackGroup> {
        self.groups.iter().find(|g| g.id == id)
    }

    /// Look up a group by ID (mutable).
    pub fn find_mut(&mut self, id: u32) -> Option<&mut TrackGroup> {
        self.groups.iter_mut().find(|g| g.id == id)
    }

    /// Find which group (if any) a track belongs to.
    #[must_use]
    pub fn group_of_track(&self, track_id: u32) -> Option<&TrackGroup> {
        self.groups.iter().find(|g| g.contains(track_id))
    }

    /// Remove a group entirely. Returns `true` on success.
    pub fn remove_group(&mut self, id: u32) -> bool {
        let before = self.groups.len();
        self.groups.retain(|g| g.id != id);
        self.groups.len() < before
    }

    /// Return all groups whose state satisfies the predicate.
    #[must_use]
    pub fn groups_with_state(&self, state: GroupState) -> Vec<&TrackGroup> {
        self.groups.iter().filter(|g| g.state == state).collect()
    }

    /// Total number of groups.
    #[must_use]
    pub fn count(&self) -> usize {
        self.groups.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn group_state_is_muted() {
        assert!(GroupState::Muted.is_muted());
        assert!(GroupState::MutedAndLocked.is_muted());
        assert!(!GroupState::Active.is_muted());
        assert!(!GroupState::Locked.is_muted());
    }

    #[test]
    fn group_state_is_locked() {
        assert!(GroupState::Locked.is_locked());
        assert!(GroupState::MutedAndLocked.is_locked());
        assert!(!GroupState::Active.is_locked());
        assert!(!GroupState::Muted.is_locked());
    }

    #[test]
    fn new_group_is_empty() {
        let g = TrackGroup::new(0, "Test");
        assert!(g.is_empty());
        assert_eq!(g.len(), 0);
    }

    #[test]
    fn add_track_returns_false_on_duplicate() {
        let mut g = TrackGroup::new(0, "G");
        assert!(g.add_track(10));
        assert!(!g.add_track(10));
        assert_eq!(g.len(), 1);
    }

    #[test]
    fn remove_track_returns_true_on_success() {
        let mut g = TrackGroup::new(0, "G");
        g.add_track(5);
        assert!(g.remove_track(5));
        assert!(g.is_empty());
    }

    #[test]
    fn remove_track_returns_false_when_absent() {
        let mut g = TrackGroup::new(0, "G");
        assert!(!g.remove_track(99));
    }

    #[test]
    fn contains_reflects_membership() {
        let mut g = TrackGroup::new(0, "G");
        g.add_track(7);
        assert!(g.contains(7));
        assert!(!g.contains(8));
    }

    #[test]
    fn set_state_changes_state() {
        let mut g = TrackGroup::new(0, "G");
        g.set_state(GroupState::Locked);
        assert!(g.state.is_locked());
        assert!(!g.is_editable());
    }

    #[test]
    fn active_group_is_editable() {
        let g = TrackGroup::new(0, "G");
        assert!(g.is_editable());
    }

    #[test]
    fn registry_create_group_increments_id() {
        let mut reg = TrackGroupRegistry::new();
        let id0 = reg.create_group("A");
        let id1 = reg.create_group("B");
        assert_eq!(id1, id0 + 1);
    }

    #[test]
    fn registry_find_returns_none_for_unknown_id() {
        let reg = TrackGroupRegistry::new();
        assert!(reg.find(42).is_none());
    }

    #[test]
    fn registry_group_of_track() {
        let mut reg = TrackGroupRegistry::new();
        let gid = reg.create_group("G");
        reg.find_mut(gid)
            .expect("should succeed in test")
            .add_track(3);
        assert_eq!(
            reg.group_of_track(3).expect("should succeed in test").id,
            gid
        );
        assert!(reg.group_of_track(99).is_none());
    }

    #[test]
    fn registry_remove_group() {
        let mut reg = TrackGroupRegistry::new();
        let gid = reg.create_group("X");
        assert_eq!(reg.count(), 1);
        assert!(reg.remove_group(gid));
        assert_eq!(reg.count(), 0);
    }

    #[test]
    fn registry_groups_with_state() {
        let mut reg = TrackGroupRegistry::new();
        let gid = reg.create_group("M");
        reg.find_mut(gid)
            .expect("should succeed in test")
            .set_state(GroupState::Muted);
        let muted = reg.groups_with_state(GroupState::Muted);
        assert_eq!(muted.len(), 1);
        let locked = reg.groups_with_state(GroupState::Locked);
        assert!(locked.is_empty());
    }
}
