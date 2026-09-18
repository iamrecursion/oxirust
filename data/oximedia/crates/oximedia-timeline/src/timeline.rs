//! Main timeline structure and operations.

use oximedia_core::Rational;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::HashMap;
use uuid::Uuid;

use crate::clip::{Clip, ClipId};
use crate::error::{TimelineError, TimelineResult};
use crate::marker::MarkerCollection;
use crate::metadata::Metadata;
use crate::timecode::TimecodeFormat;
use crate::track::{Track, TrackId, TrackType};
use crate::transition::Transition;
use crate::types::{Duration, Position};

/// Serializable frame rate wrapper.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct RationalSerde {
    num: i64,
    den: i64,
}

impl From<Rational> for RationalSerde {
    fn from(r: Rational) -> Self {
        Self {
            num: r.num,
            den: r.den,
        }
    }
}

impl From<RationalSerde> for Rational {
    fn from(r: RationalSerde) -> Self {
        Rational::new(r.num, r.den)
    }
}

/// Main timeline structure.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Timeline {
    /// Unique identifier for this timeline.
    pub id: Uuid,
    /// Name of the timeline/project.
    pub name: String,
    /// Frame rate as rational number.
    #[serde(
        serialize_with = "serialize_rational",
        deserialize_with = "deserialize_rational"
    )]
    pub frame_rate: Rational,
    /// Audio sample rate (Hz).
    pub sample_rate: u32,
    /// Timeline duration.
    pub duration: Duration,
    /// Timecode format.
    pub timecode_format: TimecodeFormat,
    /// Video tracks (ordered from bottom to top).
    pub video_tracks: Vec<Track>,
    /// Audio tracks.
    pub audio_tracks: Vec<Track>,
    /// Subtitle tracks.
    pub subtitle_tracks: Vec<Track>,
    /// Timeline markers.
    pub markers: MarkerCollection,
    /// Timeline metadata.
    pub metadata: Metadata,
    /// Transitions between clips.
    pub transitions: HashMap<ClipId, Transition>,
    /// Current playhead position.
    pub playhead: Position,
    /// In point for sequence.
    pub in_point: Option<Position>,
    /// Out point for sequence.
    pub out_point: Option<Position>,
}

impl Timeline {
    /// Creates a new timeline.
    ///
    /// # Errors
    ///
    /// Returns error if frame rate or sample rate is invalid.
    pub fn new(
        name: impl Into<String>,
        frame_rate: Rational,
        sample_rate: u32,
    ) -> TimelineResult<Self> {
        // Validate frame rate
        if frame_rate.num <= 0 || frame_rate.den <= 0 {
            return Err(TimelineError::InvalidFrameRate(
                frame_rate.num,
                frame_rate.den,
            ));
        }

        // Validate sample rate
        if sample_rate == 0 {
            return Err(TimelineError::InvalidSampleRate(sample_rate));
        }

        let timecode_format = TimecodeFormat::from_frame_rate(frame_rate);

        Ok(Self {
            id: Uuid::new_v4(),
            name: name.into(),
            frame_rate,
            sample_rate,
            duration: Duration::zero(),
            timecode_format,
            video_tracks: Vec::new(),
            audio_tracks: Vec::new(),
            subtitle_tracks: Vec::new(),
            markers: MarkerCollection::new(),
            metadata: Metadata::new(),
            transitions: HashMap::new(),
            playhead: Position::zero(),
            in_point: None,
            out_point: None,
        })
    }

    /// Adds a video track.
    ///
    /// # Errors
    ///
    /// Returns error on failure (currently infallible).
    pub fn add_video_track(&mut self, name: impl Into<String>) -> TimelineResult<TrackId> {
        let track = Track::new(name.into(), TrackType::Video);
        let track_id = track.id;
        self.video_tracks.push(track);
        self.update_track_z_indices();
        Ok(track_id)
    }

    /// Adds an audio track.
    ///
    /// # Errors
    ///
    /// Returns error on failure (currently infallible).
    pub fn add_audio_track(&mut self, name: impl Into<String>) -> TimelineResult<TrackId> {
        let track = Track::new(name.into(), TrackType::Audio);
        let track_id = track.id;
        self.audio_tracks.push(track);
        Ok(track_id)
    }

    /// Adds a subtitle track.
    ///
    /// # Errors
    ///
    /// Returns error on failure (currently infallible).
    pub fn add_subtitle_track(&mut self, name: impl Into<String>) -> TimelineResult<TrackId> {
        let track = Track::new(name.into(), TrackType::Subtitle);
        let track_id = track.id;
        self.subtitle_tracks.push(track);
        Ok(track_id)
    }

    /// Removes a track by ID.
    ///
    /// # Errors
    ///
    /// Returns error if track not found.
    pub fn remove_track(&mut self, track_id: TrackId) -> TimelineResult<Track> {
        // Try video tracks
        if let Some(index) = self.video_tracks.iter().position(|t| t.id == track_id) {
            let track = self.video_tracks.remove(index);
            self.update_track_z_indices();
            return Ok(track);
        }

        // Try audio tracks
        if let Some(index) = self.audio_tracks.iter().position(|t| t.id == track_id) {
            return Ok(self.audio_tracks.remove(index));
        }

        // Try subtitle tracks
        if let Some(index) = self.subtitle_tracks.iter().position(|t| t.id == track_id) {
            return Ok(self.subtitle_tracks.remove(index));
        }

        Err(TimelineError::TrackNotFound(track_id))
    }

    /// Gets a track by ID.
    #[must_use]
    pub fn get_track(&self, track_id: TrackId) -> Option<&Track> {
        self.video_tracks
            .iter()
            .find(|t| t.id == track_id)
            .or_else(|| self.audio_tracks.iter().find(|t| t.id == track_id))
            .or_else(|| self.subtitle_tracks.iter().find(|t| t.id == track_id))
    }

    /// Gets a mutable reference to a track by ID.
    pub fn get_track_mut(&mut self, track_id: TrackId) -> Option<&mut Track> {
        if let Some(track) = self.video_tracks.iter_mut().find(|t| t.id == track_id) {
            return Some(track);
        }
        if let Some(track) = self.audio_tracks.iter_mut().find(|t| t.id == track_id) {
            return Some(track);
        }
        self.subtitle_tracks.iter_mut().find(|t| t.id == track_id)
    }

    /// Adds a clip to a track.
    ///
    /// # Errors
    ///
    /// Returns error if track not found or track is locked.
    pub fn add_clip(&mut self, track_id: TrackId, clip: Clip) -> TimelineResult<()> {
        let track = self
            .get_track_mut(track_id)
            .ok_or(TimelineError::TrackNotFound(track_id))?;
        track.add_clip(clip)?;
        self.update_duration();
        Ok(())
    }

    /// Removes a clip from a track.
    ///
    /// # Errors
    ///
    /// Returns error if track or clip not found.
    pub fn remove_clip(&mut self, track_id: TrackId, clip_id: ClipId) -> TimelineResult<Clip> {
        let track = self
            .get_track_mut(track_id)
            .ok_or(TimelineError::TrackNotFound(track_id))?;
        let clip = track.remove_clip(clip_id)?;
        self.transitions.remove(&clip_id);
        self.update_duration();
        Ok(clip)
    }

    /// Adds a transition to a clip.
    ///
    /// # Errors
    ///
    /// Returns error if clip not found or transition is invalid.
    pub fn add_transition(
        &mut self,
        clip_id: ClipId,
        transition: Transition,
    ) -> TimelineResult<()> {
        // Validate that clip exists
        if self.find_clip(clip_id).is_none() {
            return Err(TimelineError::ClipNotFound(clip_id));
        }
        self.transitions.insert(clip_id, transition);
        Ok(())
    }

    /// Removes a transition from a clip.
    pub fn remove_transition(&mut self, clip_id: ClipId) -> Option<Transition> {
        self.transitions.remove(&clip_id)
    }

    /// Gets a transition for a clip.
    #[must_use]
    pub fn get_transition(&self, clip_id: ClipId) -> Option<&Transition> {
        self.transitions.get(&clip_id)
    }

    /// Finds a clip by ID across all tracks.
    #[must_use]
    pub fn find_clip(&self, clip_id: ClipId) -> Option<(&Track, &Clip)> {
        for track in self
            .video_tracks
            .iter()
            .chain(&self.audio_tracks)
            .chain(&self.subtitle_tracks)
        {
            if let Some(clip) = track.get_clip(clip_id) {
                return Some((track, clip));
            }
        }
        None
    }

    /// Gets a mutable reference to a clip by ID.
    /// Returns the clip without the track reference to avoid borrow checker issues.
    pub fn get_clip_mut(&mut self, clip_id: ClipId) -> Option<&mut Clip> {
        for track in self
            .video_tracks
            .iter_mut()
            .chain(&mut self.audio_tracks)
            .chain(&mut self.subtitle_tracks)
        {
            if let Some(clip) = track.get_clip_mut(clip_id) {
                return Some(clip);
            }
        }
        None
    }

    /// Sets the playhead position.
    pub fn set_playhead(&mut self, position: Position) {
        self.playhead = position;
    }

    /// Gets the current playhead position.
    #[must_use]
    pub const fn playhead(&self) -> Position {
        self.playhead
    }

    /// Sets the sequence in point.
    pub fn set_in_point(&mut self, position: Position) {
        self.in_point = Some(position);
    }

    /// Sets the sequence out point.
    pub fn set_out_point(&mut self, position: Position) {
        self.out_point = Some(position);
    }

    /// Clears the in/out points.
    pub fn clear_in_out(&mut self) {
        self.in_point = None;
        self.out_point = None;
    }

    /// Gets the sequence in point.
    #[must_use]
    pub const fn in_point(&self) -> Option<Position> {
        self.in_point
    }

    /// Gets the sequence out point.
    #[must_use]
    pub const fn out_point(&self) -> Option<Position> {
        self.out_point
    }

    /// Updates the timeline duration based on all clips.
    pub fn update_duration(&mut self) {
        let mut max_duration = 0_i64;

        for track in self
            .video_tracks
            .iter()
            .chain(&self.audio_tracks)
            .chain(&self.subtitle_tracks)
        {
            for clip in &track.clips {
                let clip_end = clip.timeline_out().value();
                if clip_end > max_duration {
                    max_duration = clip_end;
                }
            }
        }

        self.duration = Duration::new(max_duration);
    }

    /// Updates z-indices for video tracks.
    fn update_track_z_indices(&mut self) {
        for (index, track) in self.video_tracks.iter_mut().enumerate() {
            track.z_index = index as i32;
        }
    }

    /// Gets total number of tracks.
    #[must_use]
    pub fn track_count(&self) -> usize {
        self.video_tracks.len() + self.audio_tracks.len() + self.subtitle_tracks.len()
    }

    /// Gets total number of clips.
    #[must_use]
    pub fn clip_count(&self) -> usize {
        self.video_tracks
            .iter()
            .chain(&self.audio_tracks)
            .chain(&self.subtitle_tracks)
            .map(super::track::Track::clip_count)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clip::MediaSource;

    fn create_test_timeline() -> Timeline {
        Timeline::new("Test Timeline", Rational::new(24, 1), 48000).expect("should succeed in test")
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
    fn test_timeline_creation() {
        let timeline = create_test_timeline();
        assert_eq!(timeline.name, "Test Timeline");
        assert_eq!(timeline.frame_rate, Rational::new(24, 1));
        assert_eq!(timeline.sample_rate, 48000);
    }

    #[test]
    fn test_timeline_invalid_frame_rate() {
        assert!(Timeline::new("Test", Rational::new(0, 1), 48000).is_err());
        // Use direct struct construction to bypass Rational::new's panic on zero denominator
        assert!(Timeline::new("Test", Rational { num: 24, den: 0 }, 48000).is_err());
    }

    #[test]
    fn test_timeline_invalid_sample_rate() {
        assert!(Timeline::new("Test", Rational::new(24, 1), 0).is_err());
    }

    #[test]
    fn test_add_video_track() {
        let mut timeline = create_test_timeline();
        let track_id = timeline
            .add_video_track("V1")
            .expect("should succeed in test");
        assert_eq!(timeline.video_tracks.len(), 1);
        assert!(timeline.get_track(track_id).is_some());
    }

    #[test]
    fn test_add_audio_track() {
        let mut timeline = create_test_timeline();
        let track_id = timeline
            .add_audio_track("A1")
            .expect("should succeed in test");
        assert_eq!(timeline.audio_tracks.len(), 1);
        assert!(timeline.get_track(track_id).is_some());
    }

    #[test]
    fn test_remove_track() {
        let mut timeline = create_test_timeline();
        let track_id = timeline
            .add_video_track("V1")
            .expect("should succeed in test");
        assert!(timeline.remove_track(track_id).is_ok());
        assert_eq!(timeline.video_tracks.len(), 0);
    }

    #[test]
    fn test_add_clip_to_track() {
        let mut timeline = create_test_timeline();
        let track_id = timeline
            .add_video_track("V1")
            .expect("should succeed in test");
        let clip = create_test_clip(0, 100);
        assert!(timeline.add_clip(track_id, clip).is_ok());
    }

    #[test]
    fn test_remove_clip_from_track() {
        let mut timeline = create_test_timeline();
        let track_id = timeline
            .add_video_track("V1")
            .expect("should succeed in test");
        let clip = create_test_clip(0, 100);
        let clip_id = clip.id;
        timeline
            .add_clip(track_id, clip)
            .expect("should succeed in test");
        assert!(timeline.remove_clip(track_id, clip_id).is_ok());
    }

    #[test]
    fn test_find_clip() {
        let mut timeline = create_test_timeline();
        let track_id = timeline
            .add_video_track("V1")
            .expect("should succeed in test");
        let clip = create_test_clip(0, 100);
        let clip_id = clip.id;
        timeline
            .add_clip(track_id, clip)
            .expect("should succeed in test");
        assert!(timeline.find_clip(clip_id).is_some());
    }

    #[test]
    fn test_playhead() {
        let mut timeline = create_test_timeline();
        assert_eq!(timeline.playhead().value(), 0);
        timeline.set_playhead(Position::new(100));
        assert_eq!(timeline.playhead().value(), 100);
    }

    #[test]
    fn test_in_out_points() {
        let mut timeline = create_test_timeline();
        assert!(timeline.in_point().is_none());
        assert!(timeline.out_point().is_none());

        timeline.set_in_point(Position::new(100));
        timeline.set_out_point(Position::new(200));
        assert_eq!(timeline.in_point(), Some(Position::new(100)));
        assert_eq!(timeline.out_point(), Some(Position::new(200)));

        timeline.clear_in_out();
        assert!(timeline.in_point().is_none());
        assert!(timeline.out_point().is_none());
    }

    #[test]
    fn test_duration_update() {
        let mut timeline = create_test_timeline();
        let track_id = timeline
            .add_video_track("V1")
            .expect("should succeed in test");
        timeline
            .add_clip(track_id, create_test_clip(0, 100))
            .expect("should succeed in test");
        assert_eq!(timeline.duration.value(), 100);

        timeline
            .add_clip(track_id, create_test_clip(200, 100))
            .expect("should succeed in test");
        assert_eq!(timeline.duration.value(), 300);
    }

    #[test]
    fn test_track_count() {
        let mut timeline = create_test_timeline();
        timeline
            .add_video_track("V1")
            .expect("should succeed in test");
        timeline
            .add_video_track("V2")
            .expect("should succeed in test");
        timeline
            .add_audio_track("A1")
            .expect("should succeed in test");
        assert_eq!(timeline.track_count(), 3);
    }

    #[test]
    fn test_clip_count() {
        let mut timeline = create_test_timeline();
        let v_track = timeline
            .add_video_track("V1")
            .expect("should succeed in test");
        let a_track = timeline
            .add_audio_track("A1")
            .expect("should succeed in test");
        timeline
            .add_clip(v_track, create_test_clip(0, 100))
            .expect("should succeed in test");
        timeline
            .add_clip(a_track, create_test_clip(0, 100))
            .expect("should succeed in test");
        assert_eq!(timeline.clip_count(), 2);
    }

    #[test]
    fn test_add_transition() {
        let mut timeline = create_test_timeline();
        let track_id = timeline
            .add_video_track("V1")
            .expect("should succeed in test");
        let clip = create_test_clip(0, 100);
        let clip_id = clip.id;
        timeline
            .add_clip(track_id, clip)
            .expect("should succeed in test");

        let transition = Transition::dissolve(Duration::new(24));
        assert!(timeline.add_transition(clip_id, transition).is_ok());
        assert!(timeline.get_transition(clip_id).is_some());
    }

    #[test]
    fn test_remove_transition() {
        let mut timeline = create_test_timeline();
        let track_id = timeline
            .add_video_track("V1")
            .expect("should succeed in test");
        let clip = create_test_clip(0, 100);
        let clip_id = clip.id;
        timeline
            .add_clip(track_id, clip)
            .expect("should succeed in test");

        let transition = Transition::dissolve(Duration::new(24));
        timeline
            .add_transition(clip_id, transition)
            .expect("should succeed in test");
        assert!(timeline.remove_transition(clip_id).is_some());
        assert!(timeline.get_transition(clip_id).is_none());
    }

    #[test]
    fn test_z_index_update() {
        let mut timeline = create_test_timeline();
        timeline
            .add_video_track("V1")
            .expect("should succeed in test");
        timeline
            .add_video_track("V2")
            .expect("should succeed in test");
        timeline
            .add_video_track("V3")
            .expect("should succeed in test");

        assert_eq!(timeline.video_tracks[0].z_index, 0);
        assert_eq!(timeline.video_tracks[1].z_index, 1);
        assert_eq!(timeline.video_tracks[2].z_index, 2);
    }
}

/// Serialize Rational as a struct with num and den fields.
fn serialize_rational<S>(rational: &Rational, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let rs = RationalSerde::from(*rational);
    rs.serialize(serializer)
}

/// Deserialize Rational from a struct with num and den fields.
fn deserialize_rational<'de, D>(deserializer: D) -> Result<Rational, D::Error>
where
    D: Deserializer<'de>,
{
    let rs = RationalSerde::deserialize(deserializer)?;
    Ok(Rational::from(rs))
}
