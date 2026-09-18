//! Track management for timeline.

use serde::{Deserialize, Serialize};
use std::fmt;
use uuid::Uuid;

use crate::clip::{Clip, ClipId};
use crate::effects::EffectStack;
use crate::error::{TimelineError, TimelineResult};
use crate::types::Position;

/// Unique identifier for a track.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TrackId(Uuid);

impl TrackId {
    /// Creates a new random track ID.
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Creates a track ID from a UUID.
    #[must_use]
    pub const fn from_uuid(uuid: Uuid) -> Self {
        Self(uuid)
    }

    /// Returns the underlying UUID.
    #[must_use]
    pub const fn as_uuid(&self) -> &Uuid {
        &self.0
    }
}

impl Default for TrackId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for TrackId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Type of track.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TrackType {
    /// Video track.
    Video,
    /// Audio track.
    Audio,
    /// Subtitle track.
    Subtitle,
}

impl TrackType {
    /// Checks if this is a video track.
    #[must_use]
    pub const fn is_video(self) -> bool {
        matches!(self, Self::Video)
    }

    /// Checks if this is an audio track.
    #[must_use]
    pub const fn is_audio(self) -> bool {
        matches!(self, Self::Audio)
    }

    /// Checks if this is a subtitle track.
    #[must_use]
    pub const fn is_subtitle(self) -> bool {
        matches!(self, Self::Subtitle)
    }
}

impl fmt::Display for TrackType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Video => write!(f, "Video"),
            Self::Audio => write!(f, "Audio"),
            Self::Subtitle => write!(f, "Subtitle"),
        }
    }
}

/// A track in the timeline.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Track {
    /// Unique identifier for this track.
    pub id: TrackId,
    /// Name of the track.
    pub name: String,
    /// Type of track.
    pub track_type: TrackType,
    /// Clips in this track (sorted by timeline position).
    pub clips: Vec<Clip>,
    /// Track-level effects.
    pub effects: EffectStack,
    /// Whether track is muted (audio only).
    pub muted: bool,
    /// Whether track is hidden (video only).
    pub hidden: bool,
    /// Whether track is locked.
    pub locked: bool,
    /// Whether track is solo (audio only).
    pub solo: bool,
    /// Track volume (0.0-1.0, audio only).
    pub volume: f32,
    /// Track pan (-1.0 to 1.0, audio only).
    pub pan: f32,
    /// Z-index for video compositing (video only).
    pub z_index: i32,
    /// Track height in UI (pixels).
    pub height: u32,
}

impl Track {
    /// Creates a new track.
    #[must_use]
    pub fn new(name: String, track_type: TrackType) -> Self {
        Self {
            id: TrackId::new(),
            name,
            track_type,
            clips: Vec::new(),
            effects: EffectStack::new(),
            muted: false,
            hidden: false,
            locked: false,
            solo: false,
            volume: 1.0,
            pan: 0.0,
            z_index: 0,
            height: 100,
        }
    }

    /// Adds a clip to the track.
    ///
    /// # Errors
    ///
    /// Returns error if track is locked or clip overlaps with existing clips.
    pub fn add_clip(&mut self, clip: Clip) -> TimelineResult<()> {
        if self.locked {
            return Err(TimelineError::TrackLocked(self.id));
        }

        // Check for overlaps
        for existing in &self.clips {
            if clip.overlaps(existing.timeline_in, existing.timeline_out()) {
                return Err(TimelineError::ClipOverlap(clip.timeline_in.value()));
            }
        }

        self.clips.push(clip);
        self.sort_clips();
        Ok(())
    }

    /// Removes a clip by ID.
    ///
    /// # Errors
    ///
    /// Returns error if track is locked or clip not found.
    pub fn remove_clip(&mut self, clip_id: ClipId) -> TimelineResult<Clip> {
        if self.locked {
            return Err(TimelineError::TrackLocked(self.id));
        }

        let index = self
            .clips
            .iter()
            .position(|c| c.id == clip_id)
            .ok_or(TimelineError::ClipNotFound(clip_id))?;

        Ok(self.clips.remove(index))
    }

    /// Gets a clip by ID.
    #[must_use]
    pub fn get_clip(&self, clip_id: ClipId) -> Option<&Clip> {
        self.clips.iter().find(|c| c.id == clip_id)
    }

    /// Gets a mutable reference to a clip by ID.
    pub fn get_clip_mut(&mut self, clip_id: ClipId) -> Option<&mut Clip> {
        self.clips.iter_mut().find(|c| c.id == clip_id)
    }

    /// Gets the clip at a given position.
    #[must_use]
    pub fn clip_at_position(&self, position: Position) -> Option<&Clip> {
        self.clips
            .iter()
            .find(|clip| clip.contains_position(position))
    }

    /// Gets all clips that overlap a time range.
    #[must_use]
    pub fn clips_in_range(&self, start: Position, end: Position) -> Vec<&Clip> {
        self.clips
            .iter()
            .filter(|clip| clip.overlaps(start, end))
            .collect()
    }

    /// Sorts clips by timeline position.
    fn sort_clips(&mut self) {
        self.clips.sort_by_key(|clip| clip.timeline_in.value());
    }

    /// Locks the track.
    pub fn lock(&mut self) {
        self.locked = true;
    }

    /// Unlocks the track.
    pub fn unlock(&mut self) {
        self.locked = false;
    }

    /// Mutes the track (audio only).
    pub fn mute(&mut self) {
        if self.track_type.is_audio() {
            self.muted = true;
        }
    }

    /// Unmutes the track (audio only).
    pub fn unmute(&mut self) {
        if self.track_type.is_audio() {
            self.muted = false;
        }
    }

    /// Solos the track (audio only).
    pub fn solo(&mut self) {
        if self.track_type.is_audio() {
            self.solo = true;
        }
    }

    /// Unsolos the track (audio only).
    pub fn unsolo(&mut self) {
        if self.track_type.is_audio() {
            self.solo = false;
        }
    }

    /// Hides the track (video only).
    pub fn hide(&mut self) {
        if self.track_type.is_video() {
            self.hidden = true;
        }
    }

    /// Shows the track (video only).
    pub fn show(&mut self) {
        if self.track_type.is_video() {
            self.hidden = false;
        }
    }

    /// Sets track volume (audio only).
    ///
    /// # Errors
    ///
    /// Returns error if volume is not between 0.0 and 1.0.
    pub fn set_volume(&mut self, volume: f32) -> TimelineResult<()> {
        if !(0.0..=1.0).contains(&volume) {
            return Err(TimelineError::Other(format!(
                "Invalid volume: {volume} (must be 0.0-1.0)"
            )));
        }
        if self.track_type.is_audio() {
            self.volume = volume;
        }
        Ok(())
    }

    /// Sets track pan (audio only).
    ///
    /// # Errors
    ///
    /// Returns error if pan is not between -1.0 and 1.0.
    pub fn set_pan(&mut self, pan: f32) -> TimelineResult<()> {
        if !(-1.0..=1.0).contains(&pan) {
            return Err(TimelineError::Other(format!(
                "Invalid pan: {pan} (must be -1.0 to 1.0)"
            )));
        }
        if self.track_type.is_audio() {
            self.pan = pan;
        }
        Ok(())
    }

    /// Sets z-index (video only).
    pub fn set_z_index(&mut self, z_index: i32) {
        if self.track_type.is_video() {
            self.z_index = z_index;
        }
    }

    /// Returns the total number of clips.
    #[must_use]
    pub fn clip_count(&self) -> usize {
        self.clips.len()
    }

    /// Checks if track is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.clips.is_empty()
    }

    /// Clears all clips from the track.
    ///
    /// # Errors
    ///
    /// Returns error if track is locked.
    pub fn clear(&mut self) -> TimelineResult<()> {
        if self.locked {
            return Err(TimelineError::TrackLocked(self.id));
        }
        self.clips.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clip::MediaSource;

    fn create_test_track() -> Track {
        Track::new("Test Track".to_string(), TrackType::Video)
    }

    fn create_test_clip(timeline_in: i64, duration: i64) -> Clip {
        Clip::new(
            format!("Clip at {timeline_in}"),
            MediaSource::black(),
            Position::new(0),
            Position::new(duration),
            Position::new(timeline_in),
        )
        .expect("should succeed in test")
    }

    #[test]
    fn test_track_id_creation() {
        let id1 = TrackId::new();
        let id2 = TrackId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_track_type() {
        assert!(TrackType::Video.is_video());
        assert!(!TrackType::Video.is_audio());
        assert!(TrackType::Audio.is_audio());
        assert!(!TrackType::Audio.is_video());
    }

    #[test]
    fn test_track_creation() {
        let track = create_test_track();
        assert_eq!(track.name, "Test Track");
        assert_eq!(track.track_type, TrackType::Video);
        assert!(!track.locked);
        assert!(!track.muted);
        assert!(!track.hidden);
    }

    #[test]
    fn test_track_add_clip() {
        let mut track = create_test_track();
        let clip = create_test_clip(0, 100);
        assert!(track.add_clip(clip).is_ok());
        assert_eq!(track.clip_count(), 1);
    }

    #[test]
    fn test_track_add_overlapping_clip() {
        let mut track = create_test_track();
        track
            .add_clip(create_test_clip(0, 100))
            .expect("should succeed in test");
        let result = track.add_clip(create_test_clip(50, 100));
        assert!(result.is_err());
    }

    #[test]
    fn test_track_add_clip_to_locked_track() {
        let mut track = create_test_track();
        track.lock();
        let result = track.add_clip(create_test_clip(0, 100));
        assert!(result.is_err());
    }

    #[test]
    fn test_track_remove_clip() {
        let mut track = create_test_track();
        let clip = create_test_clip(0, 100);
        let clip_id = clip.id;
        track.add_clip(clip).expect("should succeed in test");
        assert!(track.remove_clip(clip_id).is_ok());
        assert_eq!(track.clip_count(), 0);
    }

    #[test]
    fn test_track_get_clip() {
        let mut track = create_test_track();
        let clip = create_test_clip(0, 100);
        let clip_id = clip.id;
        track.add_clip(clip).expect("should succeed in test");
        assert!(track.get_clip(clip_id).is_some());
    }

    #[test]
    fn test_track_clip_at_position() {
        let mut track = create_test_track();
        track
            .add_clip(create_test_clip(0, 100))
            .expect("should succeed in test");
        track
            .add_clip(create_test_clip(200, 100))
            .expect("should succeed in test");

        assert!(track.clip_at_position(Position::new(50)).is_some());
        assert!(track.clip_at_position(Position::new(250)).is_some());
        assert!(track.clip_at_position(Position::new(150)).is_none());
    }

    #[test]
    fn test_track_clips_in_range() {
        let mut track = create_test_track();
        track
            .add_clip(create_test_clip(0, 100))
            .expect("should succeed in test");
        track
            .add_clip(create_test_clip(200, 100))
            .expect("should succeed in test");
        track
            .add_clip(create_test_clip(400, 100))
            .expect("should succeed in test");

        let clips = track.clips_in_range(Position::new(50), Position::new(250));
        assert_eq!(clips.len(), 2);
    }

    #[test]
    fn test_track_lock_unlock() {
        let mut track = create_test_track();
        assert!(!track.locked);
        track.lock();
        assert!(track.locked);
        track.unlock();
        assert!(!track.locked);
    }

    #[test]
    fn test_track_mute_unmute() {
        let mut track = Track::new("Audio Track".to_string(), TrackType::Audio);
        assert!(!track.muted);
        track.mute();
        assert!(track.muted);
        track.unmute();
        assert!(!track.muted);
    }

    #[test]
    fn test_track_mute_video_track() {
        let mut track = create_test_track(); // Video track
        track.mute();
        assert!(!track.muted); // Should not mute video track
    }

    #[test]
    fn test_track_solo() {
        let mut track = Track::new("Audio Track".to_string(), TrackType::Audio);
        assert!(!track.solo);
        track.solo();
        assert!(track.solo);
        track.unsolo();
        assert!(!track.solo);
    }

    #[test]
    fn test_track_hide_show() {
        let mut track = create_test_track();
        assert!(!track.hidden);
        track.hide();
        assert!(track.hidden);
        track.show();
        assert!(!track.hidden);
    }

    #[test]
    fn test_track_set_volume() {
        let mut track = Track::new("Audio Track".to_string(), TrackType::Audio);
        assert!(track.set_volume(0.5).is_ok());
        assert!((track.volume - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_track_set_invalid_volume() {
        let mut track = Track::new("Audio Track".to_string(), TrackType::Audio);
        assert!(track.set_volume(1.5).is_err());
        assert!(track.set_volume(-0.5).is_err());
    }

    #[test]
    fn test_track_set_pan() {
        let mut track = Track::new("Audio Track".to_string(), TrackType::Audio);
        assert!(track.set_pan(0.5).is_ok());
        assert!((track.pan - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_track_set_z_index() {
        let mut track = create_test_track();
        track.set_z_index(5);
        assert_eq!(track.z_index, 5);
    }

    #[test]
    fn test_track_is_empty() {
        let mut track = create_test_track();
        assert!(track.is_empty());
        track
            .add_clip(create_test_clip(0, 100))
            .expect("should succeed in test");
        assert!(!track.is_empty());
    }

    #[test]
    fn test_track_clear() {
        let mut track = create_test_track();
        track
            .add_clip(create_test_clip(0, 100))
            .expect("should succeed in test");
        track
            .add_clip(create_test_clip(200, 100))
            .expect("should succeed in test");
        assert!(track.clear().is_ok());
        assert!(track.is_empty());
    }

    #[test]
    fn test_track_sorting() {
        let mut track = create_test_track();
        track
            .add_clip(create_test_clip(200, 100))
            .expect("should succeed in test");
        track
            .add_clip(create_test_clip(0, 100))
            .expect("should succeed in test");
        track
            .add_clip(create_test_clip(100, 100))
            .expect("should succeed in test");

        // Clips should be sorted by timeline position
        assert_eq!(track.clips[0].timeline_in.value(), 0);
        assert_eq!(track.clips[1].timeline_in.value(), 100);
        assert_eq!(track.clips[2].timeline_in.value(), 200);
    }
}
