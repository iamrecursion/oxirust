//! Timeline clip-sequence management.
//!
//! Provides `SequenceSettings`, `TrackKind`, `SequenceTrack`, `SequenceClip`,
//! and `ClipSequence` — a self-contained, frame-accurate arrangement of clips
//! across multiple tracks.

/// Basic settings that characterise a clip sequence.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SequenceSettings {
    /// Frames per second (e.g. 23.976, 25.0, 29.97).
    pub frame_rate: f64,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Audio sample rate in Hz (e.g. 48000).
    pub sample_rate: u32,
    /// Number of audio channels.
    pub channels: u8,
}

impl SequenceSettings {
    /// Returns `true` if the sequence is HD (width >= 1280 and height >= 720).
    #[must_use]
    pub fn is_hd(&self) -> bool {
        self.width >= 1280 && self.height >= 720
    }
}

impl Default for SequenceSettings {
    fn default() -> Self {
        Self {
            frame_rate: 25.0,
            width: 1920,
            height: 1080,
            sample_rate: 48_000,
            channels: 2,
        }
    }
}

/// The kind of media carried by a track.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum TrackKind {
    /// Video frames.
    Video,
    /// Audio samples.
    Audio,
    /// Subtitle/caption data.
    Subtitle,
    /// Visual effects and compositing.
    Effect,
}

impl TrackKind {
    /// Returns `true` for track types that carry primary media (video or audio).
    #[must_use]
    pub fn is_media(&self) -> bool {
        matches!(self, Self::Video | Self::Audio)
    }
}

/// A single track within a `ClipSequence`.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SequenceTrack {
    /// Unique track identifier within the sequence.
    pub id: u32,
    /// Human-readable label.
    pub name: String,
    /// Kind of media this track carries.
    pub track_kind: TrackKind,
    /// Whether this track is locked against edits.
    pub locked: bool,
    /// Whether this track is muted during playback.
    pub muted: bool,
}

impl SequenceTrack {
    /// Returns `true` if the track is neither locked nor muted — i.e. it is
    /// available for both editing and playback.
    #[must_use]
    pub fn is_available(&self) -> bool {
        !self.locked && !self.muted
    }
}

/// A clip placed on a specific track within a `ClipSequence`.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SequenceClip {
    /// Unique clip identifier.
    pub id: u64,
    /// ID of the track this clip belongs to.
    pub track_id: u32,
    /// Frame position on the timeline where the clip starts.
    pub start_frame: u64,
    /// Number of frames the clip occupies.
    pub duration_frames: u32,
    /// Identifier of the source media asset.
    pub source_id: String,
    /// Playback speed multiplier (1.0 = normal, 0.5 = half, -1.0 = reversed).
    pub speed: f32,
}

impl SequenceClip {
    /// Returns the exclusive end frame of this clip on the timeline.
    #[must_use]
    pub fn end_frame(&self) -> u64 {
        self.start_frame + u64::from(self.duration_frames)
    }

    /// Returns `true` if the clip plays in reverse (speed < 0.0).
    #[must_use]
    pub fn is_reversed(&self) -> bool {
        self.speed < 0.0
    }

    /// Returns `true` if the clip is playing at less than half speed.
    #[must_use]
    pub fn is_slow_motion(&self) -> bool {
        self.speed.abs() < 0.5 && self.speed != 0.0
    }
}

/// A complete arrangement of tracks and clips representing an editing sequence.
#[derive(Debug, Default)]
#[allow(dead_code)]
pub struct ClipSequence {
    /// Sequence-wide settings.
    pub settings: SequenceSettings,
    /// Ordered list of tracks.
    pub tracks: Vec<SequenceTrack>,
    /// All clips placed on the timeline.
    pub clips: Vec<SequenceClip>,
}

impl ClipSequence {
    /// Create a new `ClipSequence` with the given settings.
    #[must_use]
    pub fn new(settings: SequenceSettings) -> Self {
        Self {
            settings,
            tracks: Vec::new(),
            clips: Vec::new(),
        }
    }

    /// Append a track to the sequence.
    pub fn add_track(&mut self, track: SequenceTrack) {
        self.tracks.push(track);
    }

    /// Place a clip on the timeline.
    pub fn add_clip(&mut self, clip: SequenceClip) {
        self.clips.push(clip);
    }

    /// Return all clips that belong to the given track.
    #[must_use]
    pub fn clips_on_track(&self, track_id: u32) -> Vec<&SequenceClip> {
        self.clips
            .iter()
            .filter(|c| c.track_id == track_id)
            .collect()
    }

    /// Return the total duration of the sequence in frames (end frame of the
    /// last clip across all tracks), or 0 if there are no clips.
    #[must_use]
    pub fn duration_frames(&self) -> u64 {
        self.clips
            .iter()
            .map(SequenceClip::end_frame)
            .max()
            .unwrap_or(0)
    }

    /// Look up a clip by its unique identifier.
    #[must_use]
    pub fn find_clip(&self, id: u64) -> Option<&SequenceClip> {
        self.clips.iter().find(|c| c.id == id)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_settings() -> SequenceSettings {
        SequenceSettings::default()
    }

    fn make_track(id: u32, kind: TrackKind) -> SequenceTrack {
        SequenceTrack {
            id,
            name: format!("Track {id}"),
            track_kind: kind,
            locked: false,
            muted: false,
        }
    }

    fn make_clip(id: u64, track_id: u32, start: u64, dur: u32, speed: f32) -> SequenceClip {
        SequenceClip {
            id,
            track_id,
            start_frame: start,
            duration_frames: dur,
            source_id: format!("src-{id}"),
            speed,
        }
    }

    // --- SequenceSettings ---

    #[test]
    fn test_settings_is_hd_default() {
        assert!(default_settings().is_hd());
    }

    #[test]
    fn test_settings_is_hd_false_for_sd() {
        let s = SequenceSettings {
            width: 720,
            height: 576,
            ..Default::default()
        };
        assert!(!s.is_hd());
    }

    #[test]
    fn test_settings_hd_boundary_1280x720() {
        let s = SequenceSettings {
            width: 1280,
            height: 720,
            ..Default::default()
        };
        assert!(s.is_hd());
    }

    // --- TrackKind ---

    #[test]
    fn test_track_kind_video_is_media() {
        assert!(TrackKind::Video.is_media());
    }

    #[test]
    fn test_track_kind_audio_is_media() {
        assert!(TrackKind::Audio.is_media());
    }

    #[test]
    fn test_track_kind_subtitle_not_media() {
        assert!(!TrackKind::Subtitle.is_media());
    }

    #[test]
    fn test_track_kind_effect_not_media() {
        assert!(!TrackKind::Effect.is_media());
    }

    // --- SequenceTrack ---

    #[test]
    fn test_track_is_available_when_not_locked_or_muted() {
        let t = make_track(1, TrackKind::Video);
        assert!(t.is_available());
    }

    #[test]
    fn test_track_not_available_when_locked() {
        let mut t = make_track(2, TrackKind::Audio);
        t.locked = true;
        assert!(!t.is_available());
    }

    #[test]
    fn test_track_not_available_when_muted() {
        let mut t = make_track(3, TrackKind::Video);
        t.muted = true;
        assert!(!t.is_available());
    }

    // --- SequenceClip ---

    #[test]
    fn test_clip_end_frame() {
        let c = make_clip(1, 1, 100, 50, 1.0);
        assert_eq!(c.end_frame(), 150);
    }

    #[test]
    fn test_clip_is_reversed() {
        let c = make_clip(2, 1, 0, 100, -1.0);
        assert!(c.is_reversed());
    }

    #[test]
    fn test_clip_not_reversed_at_normal_speed() {
        let c = make_clip(3, 1, 0, 100, 1.0);
        assert!(!c.is_reversed());
    }

    #[test]
    fn test_clip_is_slow_motion_below_half_speed() {
        let c = make_clip(4, 1, 0, 200, 0.25);
        assert!(c.is_slow_motion());
    }

    #[test]
    fn test_clip_not_slow_motion_at_half_speed() {
        let c = make_clip(5, 1, 0, 200, 0.5);
        assert!(!c.is_slow_motion());
    }

    // --- ClipSequence ---

    #[test]
    fn test_sequence_duration_frames_empty() {
        let seq = ClipSequence::new(default_settings());
        assert_eq!(seq.duration_frames(), 0);
    }

    #[test]
    fn test_sequence_duration_frames_max_end() {
        let mut seq = ClipSequence::new(default_settings());
        seq.add_track(make_track(1, TrackKind::Video));
        seq.add_clip(make_clip(1, 1, 0, 100, 1.0));
        seq.add_clip(make_clip(2, 1, 50, 200, 1.0)); // ends at 250
        assert_eq!(seq.duration_frames(), 250);
    }

    #[test]
    fn test_sequence_clips_on_track() {
        let mut seq = ClipSequence::new(default_settings());
        seq.add_track(make_track(1, TrackKind::Video));
        seq.add_track(make_track(2, TrackKind::Audio));
        seq.add_clip(make_clip(1, 1, 0, 100, 1.0));
        seq.add_clip(make_clip(2, 2, 0, 100, 1.0));
        seq.add_clip(make_clip(3, 1, 100, 50, 1.0));
        assert_eq!(seq.clips_on_track(1).len(), 2);
        assert_eq!(seq.clips_on_track(2).len(), 1);
    }

    #[test]
    fn test_sequence_find_clip_existing() {
        let mut seq = ClipSequence::new(default_settings());
        seq.add_clip(make_clip(42, 1, 0, 60, 1.0));
        assert!(seq.find_clip(42).is_some());
    }

    #[test]
    fn test_sequence_find_clip_nonexistent() {
        let seq = ClipSequence::new(default_settings());
        assert!(seq.find_clip(999).is_none());
    }
}
