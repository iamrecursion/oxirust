//! Multi-track edit operations.
//!
//! Provides track management including locking, visibility, muting, and soloing
//! for video, audio, title, and effect tracks.

#![allow(dead_code)]
#![allow(clippy::module_name_repetitions)]

/// The type of media carried by a track.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackType {
    /// A video track.
    Video,
    /// An audio track.
    Audio,
    /// A title (graphics/subtitle) track.
    Title,
    /// An effect (adjustment) track.
    Effect,
}

impl TrackType {
    /// Returns `true` when the track carries both audio and video content
    /// (i.e., it is a Video or Audio track).
    #[must_use]
    pub fn is_av(self) -> bool {
        matches!(self, TrackType::Video | TrackType::Audio)
    }
}

/// Per-track edit-lock state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrackLock {
    /// Unique track identifier.
    pub track_id: u32,
    /// When `true`, edits to this track are blocked.
    pub locked: bool,
    /// When `true`, sync-lock prevents this track from drifting during ripple edits.
    pub sync_locked: bool,
}

impl TrackLock {
    /// Create a new unlocked `TrackLock` for `track_id`.
    #[must_use]
    pub fn new(track_id: u32) -> Self {
        Self {
            track_id,
            locked: false,
            sync_locked: false,
        }
    }

    /// Returns `true` when the track can be edited (not hard-locked).
    #[must_use]
    pub fn can_edit(&self) -> bool {
        !self.locked
    }
}

/// Per-track visibility and monitoring state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrackVisibility {
    /// Unique track identifier.
    pub track_id: u32,
    /// Whether the track content is visible in the viewer.
    pub visible: bool,
    /// When `true`, only solo tracks are rendered.
    pub solo: bool,
    /// When `true`, audio output from this track is silenced.
    pub muted: bool,
}

impl TrackVisibility {
    /// Create a new `TrackVisibility` with sensible defaults (visible, not solo, not muted).
    #[must_use]
    pub fn new(track_id: u32) -> Self {
        Self {
            track_id,
            visible: true,
            solo: false,
            muted: false,
        }
    }

    /// Returns `true` when this track's audio is audible.
    ///
    /// A track is audible when:
    /// - it is not muted, **and**
    /// - either no track is soloed, or this track is the soloed one.
    #[must_use]
    pub fn is_audible(&self, any_solo: bool) -> bool {
        if self.muted {
            return false;
        }
        if any_solo {
            return self.solo;
        }
        true
    }
}

/// Complete multi-track configuration for a timeline.
#[derive(Debug, Default)]
pub struct MultitrackConfig {
    /// Ordered list of `(track_id, TrackType)` pairs.
    pub tracks: Vec<(u32, TrackType)>,
    /// Lock states, one entry per track.
    pub locks: Vec<TrackLock>,
    /// Visibility states, one entry per track.
    pub visibility: Vec<TrackVisibility>,
}

impl MultitrackConfig {
    /// Create an empty configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a new track and initialise its lock/visibility state.
    pub fn add_track(&mut self, track_id: u32, track_type: TrackType) {
        self.tracks.push((track_id, track_type));
        self.locks.push(TrackLock::new(track_id));
        self.visibility.push(TrackVisibility::new(track_id));
    }

    /// Lock or unlock a track by ID.  Does nothing if the track is not found.
    pub fn lock_track(&mut self, track_id: u32, locked: bool) {
        if let Some(lock) = self.locks.iter_mut().find(|l| l.track_id == track_id) {
            lock.locked = locked;
        }
    }

    /// Mute or unmute a track by ID.  Does nothing if the track is not found.
    pub fn mute_track(&mut self, track_id: u32, muted: bool) {
        if let Some(vis) = self.visibility.iter_mut().find(|v| v.track_id == track_id) {
            vis.muted = muted;
        }
    }

    /// Solo or un-solo a track by ID.  Does nothing if the track is not found.
    pub fn solo_track(&mut self, track_id: u32, solo: bool) {
        if let Some(vis) = self.visibility.iter_mut().find(|v| v.track_id == track_id) {
            vis.solo = solo;
        }
    }

    /// Return the IDs of all tracks whose audio is currently audible.
    #[must_use]
    pub fn audible_tracks(&self) -> Vec<u32> {
        let any_solo = self.visibility.iter().any(|v| v.solo);
        self.visibility
            .iter()
            .filter(|v| v.is_audible(any_solo))
            .map(|v| v.track_id)
            .collect()
    }

    /// Return the total number of tracks.
    #[must_use]
    pub fn track_count(&self) -> usize {
        self.tracks.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ----- TrackType tests -----

    #[test]
    fn test_track_type_video_is_av() {
        assert!(TrackType::Video.is_av());
    }

    #[test]
    fn test_track_type_audio_is_av() {
        assert!(TrackType::Audio.is_av());
    }

    #[test]
    fn test_track_type_title_not_av() {
        assert!(!TrackType::Title.is_av());
    }

    #[test]
    fn test_track_type_effect_not_av() {
        assert!(!TrackType::Effect.is_av());
    }

    // ----- TrackLock tests -----

    #[test]
    fn test_track_lock_new_unlocked() {
        let lock = TrackLock::new(1);
        assert!(!lock.locked);
        assert!(lock.can_edit());
    }

    #[test]
    fn test_track_lock_locked_cannot_edit() {
        let mut lock = TrackLock::new(2);
        lock.locked = true;
        assert!(!lock.can_edit());
    }

    // ----- TrackVisibility tests -----

    #[test]
    fn test_track_visibility_defaults() {
        let vis = TrackVisibility::new(1);
        assert!(vis.visible);
        assert!(!vis.solo);
        assert!(!vis.muted);
    }

    #[test]
    fn test_is_audible_no_solo_not_muted() {
        let vis = TrackVisibility::new(1);
        assert!(vis.is_audible(false));
    }

    #[test]
    fn test_is_audible_muted() {
        let mut vis = TrackVisibility::new(1);
        vis.muted = true;
        assert!(!vis.is_audible(false));
    }

    #[test]
    fn test_is_audible_solo_mode_not_soloed() {
        let vis = TrackVisibility::new(1); // solo = false
        assert!(!vis.is_audible(true));
    }

    #[test]
    fn test_is_audible_solo_mode_is_soloed() {
        let mut vis = TrackVisibility::new(1);
        vis.solo = true;
        assert!(vis.is_audible(true));
    }

    // ----- MultitrackConfig tests -----

    #[test]
    fn test_multitrack_add_track() {
        let mut cfg = MultitrackConfig::new();
        cfg.add_track(1, TrackType::Video);
        cfg.add_track(2, TrackType::Audio);
        assert_eq!(cfg.track_count(), 2);
    }

    #[test]
    fn test_multitrack_lock_track() {
        let mut cfg = MultitrackConfig::new();
        cfg.add_track(1, TrackType::Video);
        cfg.lock_track(1, true);
        assert!(!cfg.locks[0].can_edit());
    }

    #[test]
    fn test_multitrack_mute_track() {
        let mut cfg = MultitrackConfig::new();
        cfg.add_track(1, TrackType::Audio);
        cfg.mute_track(1, true);
        let audible = cfg.audible_tracks();
        assert!(!audible.contains(&1));
    }

    #[test]
    fn test_multitrack_solo_track() {
        let mut cfg = MultitrackConfig::new();
        cfg.add_track(1, TrackType::Audio);
        cfg.add_track(2, TrackType::Audio);
        cfg.solo_track(1, true);
        let audible = cfg.audible_tracks();
        // Only track 1 should be audible because it's the soloed track
        assert!(audible.contains(&1));
        assert!(!audible.contains(&2));
    }

    #[test]
    fn test_multitrack_audible_tracks_all_active() {
        let mut cfg = MultitrackConfig::new();
        cfg.add_track(1, TrackType::Audio);
        cfg.add_track(2, TrackType::Audio);
        let audible = cfg.audible_tracks();
        assert_eq!(audible.len(), 2);
    }
}
