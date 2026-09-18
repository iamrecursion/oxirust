//! Multivariant playlist management for adaptive streaming.
//!
//! Provides types for managing video renditions, audio renditions, and
//! multivariant (master) playlists used in HLS and DASH.

#![allow(dead_code)]

/// A video rendition representing a single quality level.
#[derive(Debug, Clone, PartialEq)]
pub struct VideoRendition {
    /// Unique identifier for this rendition.
    pub id: String,
    /// Bandwidth in bits per second.
    pub bandwidth_bps: u32,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Frame rate in frames per second.
    pub frame_rate: f32,
    /// Codec string (e.g., `"av01.0.04M.08"`).
    pub codecs: String,
}

impl VideoRendition {
    /// Create a new video rendition.
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        bandwidth_bps: u32,
        width: u32,
        height: u32,
        frame_rate: f32,
        codecs: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            bandwidth_bps,
            width,
            height,
            frame_rate,
            codecs: codecs.into(),
        }
    }

    /// Return a human-readable resolution name based on the frame height.
    ///
    /// Examples: `"2160p"`, `"1080p"`, `"720p"`, `"480p"`, `"360p"`, `"240p"`.
    #[must_use]
    pub fn resolution_name(&self) -> String {
        format!("{}p", self.height)
    }

    /// Return `true` if this rendition is considered HD (720p or above).
    #[must_use]
    pub fn is_hd(&self) -> bool {
        self.height >= 720
    }
}

/// An audio rendition representing a single audio track option.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioRendition {
    /// Unique identifier for this rendition.
    pub id: String,
    /// Bandwidth in bits per second.
    pub bandwidth_bps: u32,
    /// Number of audio channels.
    pub channels: u8,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// BCP-47 language tag (e.g., `"en"`, `"fr"`).
    pub language: String,
    /// Codec string (e.g., `"opus"`).
    pub codecs: String,
}

impl AudioRendition {
    /// Create a new audio rendition.
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        bandwidth_bps: u32,
        channels: u8,
        sample_rate: u32,
        language: impl Into<String>,
        codecs: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            bandwidth_bps,
            channels,
            sample_rate,
            language: language.into(),
            codecs: codecs.into(),
        }
    }

    /// Return `true` if this rendition has more than 2 channels (surround sound).
    #[must_use]
    pub fn is_surround(&self) -> bool {
        self.channels > 2
    }
}

/// A multivariant (master) playlist aggregating all available renditions.
#[derive(Debug, Clone, Default)]
pub struct MultivariantPlaylist {
    /// Available video renditions, ordered by bandwidth ascending.
    pub video: Vec<VideoRendition>,
    /// Available audio renditions.
    pub audio: Vec<AudioRendition>,
    /// Subtitle track identifiers.
    pub subtitles: Vec<String>,
}

impl MultivariantPlaylist {
    /// Create an empty multivariant playlist.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a video rendition.  Renditions are kept sorted by bandwidth (ascending).
    pub fn add_video(&mut self, rendition: VideoRendition) {
        self.video.push(rendition);
        self.video.sort_by_key(|r| r.bandwidth_bps);
    }

    /// Add an audio rendition.
    pub fn add_audio(&mut self, rendition: AudioRendition) {
        self.audio.push(rendition);
    }

    /// Return a reference to the default (lowest-bandwidth) video rendition, if any.
    #[must_use]
    pub fn default_video(&self) -> Option<&VideoRendition> {
        self.video.first()
    }

    /// Return the `(min, max)` bandwidth range across all video renditions.
    ///
    /// Returns `(0, 0)` when there are no video renditions.
    #[must_use]
    pub fn bandwidth_range(&self) -> (u32, u32) {
        if self.video.is_empty() {
            return (0, 0);
        }
        let min = self
            .video
            .iter()
            .map(|r| r.bandwidth_bps)
            .min()
            .unwrap_or(0);
        let max = self
            .video
            .iter()
            .map(|r| r.bandwidth_bps)
            .max()
            .unwrap_or(0);
        (min, max)
    }

    /// Return `true` if there is at least one audio rendition.
    #[must_use]
    pub fn has_audio_groups(&self) -> bool {
        !self.audio.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- VideoRendition tests ---

    #[test]
    fn test_video_rendition_resolution_name_1080p() {
        let r = VideoRendition::new("v1", 5_000_000, 1920, 1080, 30.0, "av1");
        assert_eq!(r.resolution_name(), "1080p");
    }

    #[test]
    fn test_video_rendition_resolution_name_720p() {
        let r = VideoRendition::new("v2", 3_000_000, 1280, 720, 30.0, "av1");
        assert_eq!(r.resolution_name(), "720p");
    }

    #[test]
    fn test_video_rendition_resolution_name_480p() {
        let r = VideoRendition::new("v3", 1_500_000, 854, 480, 30.0, "av1");
        assert_eq!(r.resolution_name(), "480p");
    }

    #[test]
    fn test_video_rendition_is_hd_true_for_720p() {
        let r = VideoRendition::new("v2", 3_000_000, 1280, 720, 30.0, "av1");
        assert!(r.is_hd());
    }

    #[test]
    fn test_video_rendition_is_hd_true_for_1080p() {
        let r = VideoRendition::new("v1", 5_000_000, 1920, 1080, 30.0, "av1");
        assert!(r.is_hd());
    }

    #[test]
    fn test_video_rendition_is_hd_false_for_480p() {
        let r = VideoRendition::new("v3", 1_500_000, 854, 480, 30.0, "av1");
        assert!(!r.is_hd());
    }

    #[test]
    fn test_video_rendition_resolution_name_2160p() {
        let r = VideoRendition::new("v4", 15_000_000, 3840, 2160, 24.0, "av1");
        assert_eq!(r.resolution_name(), "2160p");
    }

    // --- AudioRendition tests ---

    #[test]
    fn test_audio_rendition_is_surround_false_for_stereo() {
        let a = AudioRendition::new("a1", 128_000, 2, 48_000, "en", "opus");
        assert!(!a.is_surround());
    }

    #[test]
    fn test_audio_rendition_is_surround_true_for_5_1() {
        let a = AudioRendition::new("a2", 384_000, 6, 48_000, "en", "opus");
        assert!(a.is_surround());
    }

    #[test]
    fn test_audio_rendition_is_surround_false_for_mono() {
        let a = AudioRendition::new("a3", 64_000, 1, 44_100, "en", "opus");
        assert!(!a.is_surround());
    }

    // --- MultivariantPlaylist tests ---

    #[test]
    fn test_playlist_empty_default_video() {
        let playlist = MultivariantPlaylist::new();
        assert!(playlist.default_video().is_none());
    }

    #[test]
    fn test_playlist_add_video_and_get_default() {
        let mut playlist = MultivariantPlaylist::new();
        playlist.add_video(VideoRendition::new(
            "v1", 5_000_000, 1920, 1080, 30.0, "av1",
        ));
        playlist.add_video(VideoRendition::new("v2", 1_500_000, 854, 480, 30.0, "av1"));
        // default should be lowest bandwidth (480p)
        let default = playlist.default_video().expect("should succeed in test");
        assert_eq!(default.id, "v2");
    }

    #[test]
    fn test_playlist_bandwidth_range_empty() {
        let playlist = MultivariantPlaylist::new();
        assert_eq!(playlist.bandwidth_range(), (0, 0));
    }

    #[test]
    fn test_playlist_bandwidth_range() {
        let mut playlist = MultivariantPlaylist::new();
        playlist.add_video(VideoRendition::new(
            "v1", 5_000_000, 1920, 1080, 30.0, "av1",
        ));
        playlist.add_video(VideoRendition::new("v2", 1_500_000, 854, 480, 30.0, "av1"));
        playlist.add_video(VideoRendition::new("v3", 3_000_000, 1280, 720, 30.0, "av1"));
        let (min, max) = playlist.bandwidth_range();
        assert_eq!(min, 1_500_000);
        assert_eq!(max, 5_000_000);
    }

    #[test]
    fn test_playlist_has_audio_groups_false_when_empty() {
        let playlist = MultivariantPlaylist::new();
        assert!(!playlist.has_audio_groups());
    }

    #[test]
    fn test_playlist_has_audio_groups_true_after_add() {
        let mut playlist = MultivariantPlaylist::new();
        playlist.add_audio(AudioRendition::new("a1", 128_000, 2, 48_000, "en", "opus"));
        assert!(playlist.has_audio_groups());
    }

    #[test]
    fn test_playlist_video_sorted_by_bandwidth() {
        let mut playlist = MultivariantPlaylist::new();
        playlist.add_video(VideoRendition::new(
            "v1", 5_000_000, 1920, 1080, 30.0, "av1",
        ));
        playlist.add_video(VideoRendition::new("v2", 1_500_000, 854, 480, 30.0, "av1"));
        playlist.add_video(VideoRendition::new("v3", 3_000_000, 1280, 720, 30.0, "av1"));
        let bandwidths: Vec<u32> = playlist.video.iter().map(|r| r.bandwidth_bps).collect();
        assert_eq!(bandwidths, vec![1_500_000, 3_000_000, 5_000_000]);
    }
}
