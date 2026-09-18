//! Timeline clip representation.

use crate::types::{ClipMatch, FrameRate, Timecode};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// A clip in the timeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineClip {
    /// Unique clip ID.
    pub id: String,
    /// Source media file path.
    pub source_path: PathBuf,
    /// Source in point (timecode or frames).
    pub source_in: Timecode,
    /// Source out point.
    pub source_out: Timecode,
    /// Timeline in point.
    pub timeline_in: Timecode,
    /// Timeline out point.
    pub timeline_out: Timecode,
    /// Frame rate.
    pub fps: FrameRate,
    /// Clip name.
    pub name: Option<String>,
    /// Match score (0.0 - 1.0).
    pub match_score: f64,
}

impl TimelineClip {
    /// Create a timeline clip from a clip match.
    #[must_use]
    pub fn from_match(clip_match: &ClipMatch) -> Self {
        Self {
            id: clip_match.clip.id.clone(),
            source_path: clip_match.media.path.clone(),
            source_in: clip_match.clip.source_in,
            source_out: clip_match.clip.source_out,
            timeline_in: clip_match.clip.record_in,
            timeline_out: clip_match.clip.record_out,
            fps: clip_match.clip.fps,
            name: clip_match.clip.source_file.clone(),
            match_score: clip_match.score,
        }
    }

    /// Get the duration in frames.
    #[must_use]
    pub fn duration_frames(&self) -> u64 {
        self.timeline_out.to_frames(self.fps) - self.timeline_in.to_frames(self.fps)
    }

    /// Get the duration in seconds.
    #[must_use]
    pub fn duration_seconds(&self) -> f64 {
        self.duration_frames() as f64 / self.fps.as_f64()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ClipReference, MatchMethod, MediaFile, TrackType};
    use std::path::PathBuf;

    #[test]
    fn test_timeline_clip_from_match() {
        let clip_ref = ClipReference {
            id: "test".to_string(),
            source_file: Some("test.mov".to_string()),
            source_in: Timecode::new(1, 0, 0, 0),
            source_out: Timecode::new(1, 0, 10, 0),
            record_in: Timecode::new(1, 0, 0, 0),
            record_out: Timecode::new(1, 0, 10, 0),
            track: TrackType::Video,
            fps: FrameRate::Fps25,
            metadata: std::collections::HashMap::new(),
        };

        let clip_match = ClipMatch {
            clip: clip_ref,
            media: MediaFile::new(PathBuf::from("/test/file.mov")),
            score: 1.0,
            method: MatchMethod::ExactFilename,
            details: String::new(),
        };

        let timeline_clip = TimelineClip::from_match(&clip_match);
        assert_eq!(timeline_clip.id, "test");
        assert!((timeline_clip.match_score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_clip_duration() {
        let clip_ref = ClipReference {
            id: "test".to_string(),
            source_file: Some("test.mov".to_string()),
            source_in: Timecode::new(1, 0, 0, 0),
            source_out: Timecode::new(1, 0, 10, 0),
            record_in: Timecode::new(1, 0, 0, 0),
            record_out: Timecode::new(1, 0, 10, 0),
            track: TrackType::Video,
            fps: FrameRate::Fps25,
            metadata: std::collections::HashMap::new(),
        };

        let clip_match = ClipMatch {
            clip: clip_ref,
            media: MediaFile::new(PathBuf::from("/test/file.mov")),
            score: 1.0,
            method: MatchMethod::ExactFilename,
            details: String::new(),
        };

        let timeline_clip = TimelineClip::from_match(&clip_match);
        assert_eq!(timeline_clip.duration_frames(), 250); // 10 seconds * 25 fps
        assert!((timeline_clip.duration_seconds() - 10.0).abs() < 0.1);
    }
}
