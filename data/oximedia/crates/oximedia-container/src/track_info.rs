#![allow(dead_code)]
//! High-level track type and collection helpers.
//!
//! Provides `TrackType`, `TrackInfo`, and `TrackCollection` for querying
//! the media tracks present in a container without full demuxing.

/// The broad category of a media track.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TrackType {
    /// A video track carrying compressed picture data.
    Video,
    /// An audio track carrying compressed audio data.
    Audio,
    /// A subtitle / closed-caption track.
    Subtitle,
    /// A generic data track (e.g. telemetry, chapters).
    Data,
}

impl TrackType {
    /// Returns `true` if the track is Audio or Video (i.e. an A/V track).
    #[must_use]
    pub fn is_a_v(&self) -> bool {
        matches!(self, Self::Video | Self::Audio)
    }

    /// Returns `true` for video tracks.
    #[must_use]
    pub fn is_video(&self) -> bool {
        matches!(self, Self::Video)
    }

    /// Returns `true` for audio tracks.
    #[must_use]
    pub fn is_audio(&self) -> bool {
        matches!(self, Self::Audio)
    }
}

/// Metadata describing a single media track.
#[derive(Debug, Clone, PartialEq)]
pub struct TrackInfo {
    /// Zero-based track index in the container.
    pub index: usize,
    /// Track type.
    pub track_type: TrackType,
    /// Codec identifier string (e.g. `"avc1"`, `"opus"`).
    pub codec: String,
    /// Optional human-readable language tag (e.g. `"eng"`, `"jpn"`).
    pub language: Option<String>,
    /// Whether the track is currently enabled for playback.
    pub enabled: bool,
    /// Bit rate in bits per second, if known.
    pub bitrate_bps: Option<u64>,
}

impl TrackInfo {
    /// Creates a new `TrackInfo`.
    #[must_use]
    pub fn new(index: usize, track_type: TrackType, codec: impl Into<String>) -> Self {
        Self {
            index,
            track_type,
            codec: codec.into(),
            language: None,
            enabled: true,
            bitrate_bps: None,
        }
    }

    /// Builder: set the language tag.
    #[must_use]
    pub fn with_language(mut self, lang: impl Into<String>) -> Self {
        self.language = Some(lang.into());
        self
    }

    /// Builder: set the enabled flag.
    #[must_use]
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Builder: set the bit rate.
    #[must_use]
    pub fn with_bitrate(mut self, bps: u64) -> Self {
        self.bitrate_bps = Some(bps);
        self
    }

    /// Returns `true` when this track is enabled.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Returns `true` when this track carries video.
    #[must_use]
    pub fn is_video(&self) -> bool {
        self.track_type.is_video()
    }

    /// Returns `true` when this track carries audio.
    #[must_use]
    pub fn is_audio(&self) -> bool {
        self.track_type.is_audio()
    }
}

/// A collection of `TrackInfo` entries for a single container file.
#[derive(Debug, Clone, Default)]
pub struct TrackCollection {
    tracks: Vec<TrackInfo>,
}

impl TrackCollection {
    /// Creates an empty `TrackCollection`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a track to the collection.
    pub fn add(&mut self, track: TrackInfo) {
        self.tracks.push(track);
    }

    /// Returns the total number of tracks.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tracks.len()
    }

    /// Returns `true` when the collection contains no tracks.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tracks.is_empty()
    }

    /// Returns only the video tracks.
    #[must_use]
    pub fn video_tracks(&self) -> Vec<&TrackInfo> {
        self.tracks.iter().filter(|t| t.is_video()).collect()
    }

    /// Returns only the audio tracks.
    #[must_use]
    pub fn audio_tracks(&self) -> Vec<&TrackInfo> {
        self.tracks.iter().filter(|t| t.is_audio()).collect()
    }

    /// Returns only subtitle tracks.
    #[must_use]
    pub fn subtitle_tracks(&self) -> Vec<&TrackInfo> {
        self.tracks
            .iter()
            .filter(|t| t.track_type == TrackType::Subtitle)
            .collect()
    }

    /// Returns tracks that are currently enabled.
    #[must_use]
    pub fn enabled_tracks(&self) -> Vec<&TrackInfo> {
        self.tracks.iter().filter(|t| t.is_enabled()).collect()
    }

    /// Looks up a track by its index (O(n)).
    #[must_use]
    pub fn by_index(&self, index: usize) -> Option<&TrackInfo> {
        self.tracks.iter().find(|t| t.index == index)
    }
}

// ─── Unit tests ───────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    // 1. is_a_v for Video
    #[test]
    fn test_is_a_v_video() {
        assert!(TrackType::Video.is_a_v());
    }

    // 2. is_a_v for Audio
    #[test]
    fn test_is_a_v_audio() {
        assert!(TrackType::Audio.is_a_v());
    }

    // 3. is_a_v false for Subtitle
    #[test]
    fn test_is_a_v_subtitle_false() {
        assert!(!TrackType::Subtitle.is_a_v());
    }

    // 4. is_a_v false for Data
    #[test]
    fn test_is_a_v_data_false() {
        assert!(!TrackType::Data.is_a_v());
    }

    // 5. TrackInfo default enabled
    #[test]
    fn test_track_info_default_enabled() {
        let t = TrackInfo::new(0, TrackType::Video, "avc1");
        assert!(t.is_enabled());
    }

    // 6. TrackInfo with_enabled false
    #[test]
    fn test_track_info_disabled() {
        let t = TrackInfo::new(1, TrackType::Audio, "opus").with_enabled(false);
        assert!(!t.is_enabled());
    }

    // 7. TrackInfo with_language
    #[test]
    fn test_track_info_language() {
        let t = TrackInfo::new(0, TrackType::Audio, "aac").with_language("eng");
        assert_eq!(t.language.as_deref(), Some("eng"));
    }

    // 8. TrackInfo with_bitrate
    #[test]
    fn test_track_info_bitrate() {
        let t = TrackInfo::new(0, TrackType::Video, "hevc").with_bitrate(4_000_000);
        assert_eq!(t.bitrate_bps, Some(4_000_000));
    }

    // 9. Empty collection
    #[test]
    fn test_collection_empty() {
        let c = TrackCollection::new();
        assert!(c.is_empty());
        assert_eq!(c.len(), 0);
    }

    // 10. add increases len
    #[test]
    fn test_collection_add() {
        let mut c = TrackCollection::new();
        c.add(TrackInfo::new(0, TrackType::Video, "vp9"));
        assert_eq!(c.len(), 1);
    }

    // 11. video_tracks filters correctly
    #[test]
    fn test_video_tracks() {
        let mut c = TrackCollection::new();
        c.add(TrackInfo::new(0, TrackType::Video, "vp9"));
        c.add(TrackInfo::new(1, TrackType::Audio, "opus"));
        assert_eq!(c.video_tracks().len(), 1);
        assert_eq!(c.audio_tracks().len(), 1);
    }

    // 12. subtitle_tracks
    #[test]
    fn test_subtitle_tracks() {
        let mut c = TrackCollection::new();
        c.add(TrackInfo::new(0, TrackType::Video, "h264"));
        c.add(TrackInfo::new(1, TrackType::Subtitle, "srt"));
        assert_eq!(c.subtitle_tracks().len(), 1);
    }

    // 13. by_index lookup
    #[test]
    fn test_by_index() {
        let mut c = TrackCollection::new();
        c.add(TrackInfo::new(0, TrackType::Video, "av1"));
        c.add(TrackInfo::new(1, TrackType::Audio, "flac"));
        assert_eq!(c.by_index(1).map(|t| t.codec.as_str()), Some("flac"));
    }

    // 14. enabled_tracks filters disabled ones
    #[test]
    fn test_enabled_tracks_filter() {
        let mut c = TrackCollection::new();
        c.add(TrackInfo::new(0, TrackType::Video, "h264").with_enabled(true));
        c.add(TrackInfo::new(1, TrackType::Audio, "aac").with_enabled(false));
        assert_eq!(c.enabled_tracks().len(), 1);
    }
}
