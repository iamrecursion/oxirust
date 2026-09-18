//! Track and timeline structures.

use crate::timeline::clip::TimelineClip;
use crate::timeline::transition::Transition;
use crate::types::FrameRate;
use serde::{Deserialize, Serialize};

/// Track kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrackKind {
    /// Video track.
    Video,
    /// Audio track.
    Audio,
}

/// A track in the timeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Track {
    /// Track ID.
    pub id: String,
    /// Track kind.
    pub kind: TrackKind,
    /// Track name.
    pub name: Option<String>,
    /// Clips on this track.
    pub clips: Vec<TimelineClip>,
    /// Transitions on this track.
    pub transitions: Vec<Transition>,
}

impl Track {
    /// Create a new track.
    #[must_use]
    pub fn new(id: String, kind: TrackKind) -> Self {
        Self {
            id,
            kind,
            name: None,
            clips: Vec::new(),
            transitions: Vec::new(),
        }
    }

    /// Add a clip to the track.
    pub fn add_clip(&mut self, clip: TimelineClip) {
        self.clips.push(clip);
    }

    /// Add a transition to the track.
    pub fn add_transition(&mut self, transition: Transition) {
        self.transitions.push(transition);
    }

    /// Get the total duration in frames.
    #[must_use]
    pub fn duration_frames(&self) -> u64 {
        self.clips
            .iter()
            .map(super::clip::TimelineClip::duration_frames)
            .sum()
    }

    /// Sort clips by timeline position.
    pub fn sort_clips(&mut self) {
        self.clips.sort_by_key(|c| c.timeline_in.to_frames(c.fps));
    }
}

/// A complete timeline with multiple tracks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Timeline {
    /// Timeline name.
    pub name: String,
    /// Frame rate.
    pub fps: FrameRate,
    /// Video tracks.
    pub video_tracks: Vec<Track>,
    /// Audio tracks.
    pub audio_tracks: Vec<Track>,
}

impl Timeline {
    /// Create a new timeline.
    #[must_use]
    pub fn new(name: String, fps: FrameRate) -> Self {
        Self {
            name,
            fps,
            video_tracks: Vec::new(),
            audio_tracks: Vec::new(),
        }
    }

    /// Add a video track.
    pub fn add_video_track(&mut self, track: Track) {
        self.video_tracks.push(track);
    }

    /// Add an audio track.
    pub fn add_audio_track(&mut self, track: Track) {
        self.audio_tracks.push(track);
    }

    /// Get the total duration in frames.
    #[must_use]
    pub fn duration_frames(&self) -> u64 {
        let video_duration = self
            .video_tracks
            .iter()
            .map(Track::duration_frames)
            .max()
            .unwrap_or(0);
        let audio_duration = self
            .audio_tracks
            .iter()
            .map(Track::duration_frames)
            .max()
            .unwrap_or(0);
        video_duration.max(audio_duration)
    }

    /// Get the total duration in seconds.
    #[must_use]
    pub fn duration_seconds(&self) -> f64 {
        self.duration_frames() as f64 / self.fps.as_f64()
    }

    /// Get the number of tracks.
    #[must_use]
    pub fn track_count(&self) -> usize {
        self.video_tracks.len() + self.audio_tracks.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use crate::types::Timecode;

    #[test]
    fn test_track_creation() {
        let track = Track::new("V1".to_string(), TrackKind::Video);
        assert_eq!(track.id, "V1");
        assert_eq!(track.kind, TrackKind::Video);
        assert_eq!(track.clips.len(), 0);
    }

    #[test]
    fn test_timeline_creation() {
        let timeline = Timeline::new("My Timeline".to_string(), FrameRate::Fps25);
        assert_eq!(timeline.name, "My Timeline");
        assert_eq!(timeline.track_count(), 0);
    }

    #[test]
    fn test_add_tracks() {
        let mut timeline = Timeline::new("Test".to_string(), FrameRate::Fps25);
        let v_track = Track::new("V1".to_string(), TrackKind::Video);
        let a_track = Track::new("A1".to_string(), TrackKind::Audio);

        timeline.add_video_track(v_track);
        timeline.add_audio_track(a_track);

        assert_eq!(timeline.video_tracks.len(), 1);
        assert_eq!(timeline.audio_tracks.len(), 1);
        assert_eq!(timeline.track_count(), 2);
    }
}
