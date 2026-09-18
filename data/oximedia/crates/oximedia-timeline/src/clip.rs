//! Clip management for timeline.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::PathBuf;
use uuid::Uuid;

use crate::effects::EffectStack;
use crate::keyframe::Keyframe;
use crate::types::{Duration, Position, Speed};

/// Unique identifier for a clip.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ClipId(Uuid);

impl ClipId {
    /// Creates a new random clip ID.
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Creates a clip ID from a UUID.
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

impl Default for ClipId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for ClipId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Media source for a clip.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum MediaSource {
    /// File-based media.
    File {
        /// Path to the media file.
        path: PathBuf,
        /// Stream index for video (if multi-stream file).
        video_stream: Option<usize>,
        /// Stream index for audio (if multi-stream file).
        audio_stream: Option<usize>,
    },
    /// Sequence-based media (image sequence).
    Sequence {
        /// Pattern for image sequence (e.g., "frame_%04d.png").
        pattern: String,
        /// Starting frame number.
        start: usize,
        /// Ending frame number.
        end: usize,
    },
    /// Nested sequence reference.
    SequenceReference {
        /// ID of the referenced sequence.
        sequence_id: Uuid,
    },
    /// Color/solid generator.
    Color {
        /// RGBA color values (0.0-1.0).
        rgba: [f32; 4],
    },
    /// Bars and tone generator.
    BarsAndTone,
    /// Black frames.
    Black,
}

impl MediaSource {
    /// Creates a file-based media source.
    #[must_use]
    pub fn file(path: PathBuf) -> Self {
        Self::File {
            path,
            video_stream: None,
            audio_stream: None,
        }
    }

    /// Creates a file-based media source with stream indices.
    #[must_use]
    pub fn file_with_streams(
        path: PathBuf,
        video_stream: Option<usize>,
        audio_stream: Option<usize>,
    ) -> Self {
        Self::File {
            path,
            video_stream,
            audio_stream,
        }
    }

    /// Creates an image sequence source.
    #[must_use]
    pub fn sequence(pattern: String, start: usize, end: usize) -> Self {
        Self::Sequence {
            pattern,
            start,
            end,
        }
    }

    /// Creates a nested sequence reference.
    #[must_use]
    pub fn sequence_ref(sequence_id: Uuid) -> Self {
        Self::SequenceReference { sequence_id }
    }

    /// Creates a color generator.
    #[must_use]
    pub fn color(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self::Color { rgba: [r, g, b, a] }
    }

    /// Creates a bars and tone generator.
    #[must_use]
    pub fn bars_and_tone() -> Self {
        Self::BarsAndTone
    }

    /// Creates a black frames generator.
    #[must_use]
    pub fn black() -> Self {
        Self::Black
    }

    /// Checks if this is a file-based source.
    #[must_use]
    pub const fn is_file(&self) -> bool {
        matches!(self, Self::File { .. })
    }

    /// Checks if this is a generated source.
    #[must_use]
    pub const fn is_generated(&self) -> bool {
        matches!(self, Self::Color { .. } | Self::BarsAndTone | Self::Black)
    }

    /// Checks if this is a sequence reference.
    #[must_use]
    pub const fn is_sequence_ref(&self) -> bool {
        matches!(self, Self::SequenceReference { .. })
    }
}

/// A clip in the timeline.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Clip {
    /// Unique identifier for this clip.
    pub id: ClipId,
    /// Name of the clip.
    pub name: String,
    /// Media source for this clip.
    pub source: MediaSource,
    /// In-point in source media (source timecode).
    pub source_in: Position,
    /// Out-point in source media (source timecode).
    pub source_out: Position,
    /// Position on timeline (destination timecode).
    pub timeline_in: Position,
    /// Playback speed multiplier.
    pub speed: Speed,
    /// Whether clip is reversed.
    pub reversed: bool,
    /// Effects applied to this clip.
    pub effects: EffectStack,
    /// Keyframes for animated properties.
    pub keyframes: Vec<Keyframe>,
    /// Whether clip is enabled.
    pub enabled: bool,
    /// Whether clip is locked.
    pub locked: bool,
    /// Custom metadata.
    pub metadata: std::collections::HashMap<String, String>,
}

impl Clip {
    /// Creates a new clip.
    ///
    /// # Errors
    ///
    /// Returns error if speed is invalid.
    pub fn new(
        name: String,
        source: MediaSource,
        source_in: Position,
        source_out: Position,
        timeline_in: Position,
    ) -> crate::error::TimelineResult<Self> {
        Ok(Self {
            id: ClipId::new(),
            name,
            source,
            source_in,
            source_out,
            timeline_in,
            speed: Speed::normal(),
            reversed: false,
            effects: EffectStack::new(),
            keyframes: Vec::new(),
            enabled: true,
            locked: false,
            metadata: std::collections::HashMap::new(),
        })
    }

    /// Returns the source duration (out - in).
    #[must_use]
    pub fn source_duration(&self) -> Duration {
        Duration::new(self.source_out.value() - self.source_in.value())
    }

    /// Returns the timeline duration (accounting for speed).
    #[must_use]
    pub fn timeline_duration(&self) -> Duration {
        self.speed.apply_to_duration(self.source_duration())
    }

    /// Returns the timeline out position.
    #[must_use]
    pub fn timeline_out(&self) -> Position {
        self.timeline_in + self.timeline_duration()
    }

    /// Checks if a position is within this clip.
    #[must_use]
    pub fn contains_position(&self, position: Position) -> bool {
        position >= self.timeline_in && position < self.timeline_out()
    }

    /// Checks if this clip overlaps with another time range.
    #[must_use]
    pub fn overlaps(&self, start: Position, end: Position) -> bool {
        self.timeline_in < end && self.timeline_out() > start
    }

    /// Sets the playback speed.
    ///
    /// # Errors
    ///
    /// Returns error if speed is invalid.
    pub fn set_speed(&mut self, speed: Speed) -> crate::error::TimelineResult<()> {
        self.speed = speed;
        Ok(())
    }

    /// Reverses the clip playback.
    pub fn reverse(&mut self) {
        self.reversed = !self.reversed;
    }

    /// Trims the start of the clip.
    pub fn trim_start(&mut self, new_in: Position) {
        let old_in = self.timeline_in;
        self.timeline_in = new_in;
        let offset = new_in.value() - old_in.value();
        self.source_in = Position::new(self.source_in.value() + offset);
    }

    /// Trims the end of the clip.
    pub fn trim_end(&mut self, new_out: Position) {
        let new_duration = Duration::new(new_out.value() - self.timeline_in.value());
        let source_duration = self.speed.apply_to_duration(new_duration);
        self.source_out = Position::new(self.source_in.value() + source_duration.value());
    }

    /// Splits the clip at a position.
    ///
    /// # Errors
    ///
    /// Returns error if position is not within clip.
    pub fn split_at(&self, position: Position) -> crate::error::TimelineResult<(Self, Self)> {
        if !self.contains_position(position) {
            return Err(crate::error::TimelineError::InvalidPosition(format!(
                "Position {position} not in clip range"
            )));
        }

        let offset = Duration::new(position.value() - self.timeline_in.value());
        let source_offset = Duration::new((offset.value() as f64 * self.speed.value()) as i64);

        let mut left = self.clone();
        left.id = ClipId::new();
        left.source_out = Position::new(self.source_in.value() + source_offset.value());

        let mut right = self.clone();
        right.id = ClipId::new();
        right.timeline_in = position;
        right.source_in = Position::new(self.source_in.value() + source_offset.value());

        Ok((left, right))
    }

    /// Moves the clip to a new timeline position.
    pub fn move_to(&mut self, new_position: Position) {
        self.timeline_in = new_position;
    }

    /// Adds metadata to the clip.
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    /// Gets metadata value.
    #[must_use]
    pub fn get_metadata(&self, key: &str) -> Option<&String> {
        self.metadata.get(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_clip() -> Clip {
        Clip::new(
            "Test Clip".to_string(),
            MediaSource::black(),
            Position::new(0),
            Position::new(100),
            Position::new(0),
        )
        .expect("should succeed in test")
    }

    #[test]
    fn test_clip_id_creation() {
        let id1 = ClipId::new();
        let id2 = ClipId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_media_source_file() {
        let source = MediaSource::file(PathBuf::from("/path/to/file.mp4"));
        assert!(source.is_file());
        assert!(!source.is_generated());
    }

    #[test]
    fn test_media_source_color() {
        let source = MediaSource::color(1.0, 0.0, 0.0, 1.0);
        assert!(source.is_generated());
        assert!(!source.is_file());
    }

    #[test]
    fn test_clip_creation() {
        let clip = create_test_clip();
        assert_eq!(clip.name, "Test Clip");
        assert_eq!(clip.source_in.value(), 0);
        assert_eq!(clip.source_out.value(), 100);
        assert!(clip.enabled);
        assert!(!clip.locked);
    }

    #[test]
    fn test_clip_source_duration() {
        let clip = create_test_clip();
        assert_eq!(clip.source_duration().value(), 100);
    }

    #[test]
    fn test_clip_timeline_duration() {
        let clip = create_test_clip();
        assert_eq!(clip.timeline_duration().value(), 100);
    }

    #[test]
    fn test_clip_timeline_duration_with_speed() {
        let mut clip = create_test_clip();
        clip.set_speed(Speed::new(2.0).expect("should succeed in test"))
            .expect("should succeed in test");
        assert_eq!(clip.timeline_duration().value(), 50);
    }

    #[test]
    fn test_clip_timeline_out() {
        let clip = create_test_clip();
        assert_eq!(clip.timeline_out().value(), 100);
    }

    #[test]
    fn test_clip_contains_position() {
        let clip = create_test_clip();
        assert!(clip.contains_position(Position::new(50)));
        assert!(clip.contains_position(Position::new(0)));
        assert!(!clip.contains_position(Position::new(100)));
        assert!(!clip.contains_position(Position::new(150)));
    }

    #[test]
    fn test_clip_overlaps() {
        let clip = create_test_clip();
        assert!(clip.overlaps(Position::new(50), Position::new(150)));
        assert!(clip.overlaps(Position::new(0), Position::new(50)));
        assert!(!clip.overlaps(Position::new(100), Position::new(200)));
        assert!(!clip.overlaps(Position::new(200), Position::new(300)));
    }

    #[test]
    fn test_clip_reverse() {
        let mut clip = create_test_clip();
        assert!(!clip.reversed);
        clip.reverse();
        assert!(clip.reversed);
        clip.reverse();
        assert!(!clip.reversed);
    }

    #[test]
    fn test_clip_split() {
        let clip = create_test_clip();
        let (left, right) = clip
            .split_at(Position::new(50))
            .expect("should succeed in test");

        assert_eq!(left.timeline_in.value(), 0);
        assert_eq!(left.source_out.value(), 50);

        assert_eq!(right.timeline_in.value(), 50);
        assert_eq!(right.source_in.value(), 50);
    }

    #[test]
    fn test_clip_split_invalid_position() {
        let clip = create_test_clip();
        assert!(clip.split_at(Position::new(150)).is_err());
    }

    #[test]
    fn test_clip_move_to() {
        let mut clip = create_test_clip();
        clip.move_to(Position::new(100));
        assert_eq!(clip.timeline_in.value(), 100);
        assert_eq!(clip.timeline_out().value(), 200);
    }

    #[test]
    fn test_clip_metadata() {
        let mut clip = create_test_clip();
        clip.add_metadata("key1".to_string(), "value1".to_string());
        assert_eq!(clip.get_metadata("key1"), Some(&"value1".to_string()));
        assert_eq!(clip.get_metadata("key2"), None);
    }

    #[test]
    fn test_clip_trim_start() {
        let mut clip = create_test_clip();
        clip.trim_start(Position::new(10));
        assert_eq!(clip.timeline_in.value(), 10);
        assert_eq!(clip.source_in.value(), 10);
    }

    #[test]
    fn test_clip_trim_end() {
        let mut clip = create_test_clip();
        clip.trim_end(Position::new(50));
        assert_eq!(clip.timeline_out().value(), 50);
        assert_eq!(clip.source_out.value(), 50);
    }
}
