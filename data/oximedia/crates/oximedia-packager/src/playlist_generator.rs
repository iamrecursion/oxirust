#![allow(dead_code)]
//! Playlist generation utilities for adaptive streaming.
//!
//! This module provides tools for generating HLS media playlists and
//! DASH period definitions, including segment listing, variant stream
//! assembly, and playlist serialization helpers.

use std::fmt;
use std::time::Duration;

/// A media segment entry in a playlist.
#[derive(Debug, Clone)]
pub struct PlaylistSegment {
    /// Segment index.
    pub index: u64,
    /// Segment duration.
    pub duration: Duration,
    /// Relative URI for the segment file.
    pub uri: String,
    /// Optional byte range offset.
    pub byte_range_offset: Option<u64>,
    /// Optional byte range length.
    pub byte_range_length: Option<u64>,
    /// Whether this segment has a discontinuity before it.
    pub discontinuity: bool,
}

impl PlaylistSegment {
    /// Create a new playlist segment.
    #[must_use]
    pub fn new(index: u64, duration: Duration, uri: impl Into<String>) -> Self {
        Self {
            index,
            duration,
            uri: uri.into(),
            byte_range_offset: None,
            byte_range_length: None,
            discontinuity: false,
        }
    }

    /// Set byte range for this segment.
    #[must_use]
    pub fn with_byte_range(mut self, offset: u64, length: u64) -> Self {
        self.byte_range_offset = Some(offset);
        self.byte_range_length = Some(length);
        self
    }

    /// Mark this segment as having a discontinuity.
    #[must_use]
    pub fn with_discontinuity(mut self) -> Self {
        self.discontinuity = true;
        self
    }

    /// Check if this segment has a byte range.
    #[must_use]
    pub fn has_byte_range(&self) -> bool {
        self.byte_range_offset.is_some() && self.byte_range_length.is_some()
    }
}

/// A variant stream definition for master/multivariant playlists.
#[derive(Debug, Clone)]
pub struct VariantStream {
    /// Bandwidth in bits per second.
    pub bandwidth: u64,
    /// Average bandwidth in bits per second.
    pub average_bandwidth: Option<u64>,
    /// Resolution width.
    pub width: u32,
    /// Resolution height.
    pub height: u32,
    /// Codec string (e.g., "av01.0.08M.08").
    pub codecs: String,
    /// Frame rate.
    pub frame_rate: Option<f64>,
    /// URI to the media playlist.
    pub uri: String,
}

impl VariantStream {
    /// Create a new variant stream.
    #[must_use]
    pub fn new(
        bandwidth: u64,
        width: u32,
        height: u32,
        codecs: impl Into<String>,
        uri: impl Into<String>,
    ) -> Self {
        Self {
            bandwidth,
            average_bandwidth: None,
            width,
            height,
            codecs: codecs.into(),
            frame_rate: None,
            uri: uri.into(),
        }
    }

    /// Set average bandwidth.
    #[must_use]
    pub fn with_average_bandwidth(mut self, avg: u64) -> Self {
        self.average_bandwidth = Some(avg);
        self
    }

    /// Set frame rate.
    #[must_use]
    pub fn with_frame_rate(mut self, fps: f64) -> Self {
        self.frame_rate = Some(fps);
        self
    }

    /// Format the resolution as "`WIDTHxHEIGHT`".
    #[must_use]
    pub fn resolution_string(&self) -> String {
        format!("{}x{}", self.width, self.height)
    }
}

/// Type of playlist being generated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaylistType {
    /// Live playlist (no ENDLIST tag).
    Live,
    /// Event playlist (appends but no removal).
    Event,
    /// VOD playlist (complete, static).
    Vod,
}

impl fmt::Display for PlaylistType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Live => write!(f, "LIVE"),
            Self::Event => write!(f, "EVENT"),
            Self::Vod => write!(f, "VOD"),
        }
    }
}

/// Media playlist builder for HLS-style playlists.
pub struct MediaPlaylistBuilder {
    /// HLS version.
    version: u8,
    /// Target duration (maximum segment duration, rounded up).
    target_duration: u32,
    /// Media sequence number.
    media_sequence: u64,
    /// Playlist type.
    playlist_type: Option<PlaylistType>,
    /// Segments in the playlist.
    segments: Vec<PlaylistSegment>,
    /// Whether the playlist is complete (add ENDLIST).
    is_ended: bool,
}

impl MediaPlaylistBuilder {
    /// Create a new media playlist builder.
    #[must_use]
    pub fn new(target_duration: u32) -> Self {
        Self {
            version: 7,
            target_duration,
            media_sequence: 0,
            playlist_type: None,
            segments: Vec::new(),
            is_ended: false,
        }
    }

    /// Set the HLS version.
    #[must_use]
    pub fn version(mut self, version: u8) -> Self {
        self.version = version;
        self
    }

    /// Set the media sequence number.
    #[must_use]
    pub fn media_sequence(mut self, seq: u64) -> Self {
        self.media_sequence = seq;
        self
    }

    /// Set the playlist type.
    #[must_use]
    pub fn playlist_type(mut self, pt: PlaylistType) -> Self {
        self.playlist_type = Some(pt);
        self
    }

    /// Add a segment to the playlist.
    #[must_use]
    pub fn segment(mut self, seg: PlaylistSegment) -> Self {
        self.segments.push(seg);
        self
    }

    /// Add multiple segments.
    #[must_use]
    pub fn segments(mut self, segs: impl IntoIterator<Item = PlaylistSegment>) -> Self {
        self.segments.extend(segs);
        self
    }

    /// Mark the playlist as ended (VOD or finished live).
    #[must_use]
    pub fn ended(mut self) -> Self {
        self.is_ended = true;
        self
    }

    /// Build the playlist as an M3U8 string.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn build(&self) -> String {
        let mut output = String::new();

        output.push_str("#EXTM3U\n");
        output.push_str(&format!("#EXT-X-VERSION:{}\n", self.version));
        output.push_str(&format!("#EXT-X-TARGETDURATION:{}\n", self.target_duration));
        output.push_str(&format!("#EXT-X-MEDIA-SEQUENCE:{}\n", self.media_sequence));

        if let Some(pt) = &self.playlist_type {
            output.push_str(&format!("#EXT-X-PLAYLIST-TYPE:{pt}\n"));
        }

        for seg in &self.segments {
            if seg.discontinuity {
                output.push_str("#EXT-X-DISCONTINUITY\n");
            }

            let duration_secs = seg.duration.as_secs_f64();
            output.push_str(&format!("#EXTINF:{duration_secs:.3},\n"));

            if let (Some(offset), Some(length)) = (seg.byte_range_offset, seg.byte_range_length) {
                output.push_str(&format!("#EXT-X-BYTERANGE:{length}@{offset}\n"));
            }

            output.push_str(&seg.uri);
            output.push('\n');
        }

        if self.is_ended {
            output.push_str("#EXT-X-ENDLIST\n");
        }

        output
    }

    /// Get the number of segments.
    #[must_use]
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Compute total duration of all segments.
    #[must_use]
    pub fn total_duration(&self) -> Duration {
        self.segments.iter().map(|s| s.duration).sum()
    }
}

impl Default for MediaPlaylistBuilder {
    fn default() -> Self {
        Self::new(6)
    }
}

/// Master playlist builder for HLS multivariant playlists.
pub struct MasterPlaylistBuilder {
    /// HLS version.
    version: u8,
    /// Variant streams.
    variants: Vec<VariantStream>,
}

impl MasterPlaylistBuilder {
    /// Create a new master playlist builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            version: 7,
            variants: Vec::new(),
        }
    }

    /// Set the HLS version.
    #[must_use]
    pub fn version(mut self, version: u8) -> Self {
        self.version = version;
        self
    }

    /// Add a variant stream.
    #[must_use]
    pub fn variant(mut self, v: VariantStream) -> Self {
        self.variants.push(v);
        self
    }

    /// Build the master playlist as an M3U8 string.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn build(&self) -> String {
        let mut output = String::new();

        output.push_str("#EXTM3U\n");
        output.push_str(&format!("#EXT-X-VERSION:{}\n", self.version));

        for v in &self.variants {
            let mut attrs = format!(
                "BANDWIDTH={},RESOLUTION={},CODECS=\"{}\"",
                v.bandwidth,
                v.resolution_string(),
                v.codecs
            );

            if let Some(avg) = v.average_bandwidth {
                attrs = format!("{attrs},AVERAGE-BANDWIDTH={avg}");
            }

            if let Some(fps) = v.frame_rate {
                attrs = format!("{attrs},FRAME-RATE={fps:.3}");
            }

            output.push_str(&format!("#EXT-X-STREAM-INF:{attrs}\n"));
            output.push_str(&v.uri);
            output.push('\n');
        }

        output
    }

    /// Get the number of variant streams.
    #[must_use]
    pub fn variant_count(&self) -> usize {
        self.variants.len()
    }
}

impl Default for MasterPlaylistBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_playlist_segment_basic() {
        let seg = PlaylistSegment::new(0, Duration::from_secs(6), "seg0.m4s");
        assert_eq!(seg.index, 0);
        assert_eq!(seg.uri, "seg0.m4s");
        assert!(!seg.has_byte_range());
        assert!(!seg.discontinuity);
    }

    #[test]
    fn test_playlist_segment_byte_range() {
        let seg =
            PlaylistSegment::new(0, Duration::from_secs(6), "seg0.m4s").with_byte_range(0, 1024);
        assert!(seg.has_byte_range());
        assert_eq!(seg.byte_range_offset, Some(0));
        assert_eq!(seg.byte_range_length, Some(1024));
    }

    #[test]
    fn test_playlist_segment_discontinuity() {
        let seg = PlaylistSegment::new(0, Duration::from_secs(6), "seg0.m4s").with_discontinuity();
        assert!(seg.discontinuity);
    }

    #[test]
    fn test_variant_stream_resolution() {
        let v = VariantStream::new(5_000_000, 1920, 1080, "av01.0.08M.08", "1080p.m3u8");
        assert_eq!(v.resolution_string(), "1920x1080");
    }

    #[test]
    fn test_variant_stream_with_extras() {
        let v = VariantStream::new(5_000_000, 1920, 1080, "av01.0.08M.08", "1080p.m3u8")
            .with_average_bandwidth(4_500_000)
            .with_frame_rate(29.97);
        assert_eq!(v.average_bandwidth, Some(4_500_000));
        assert!((v.frame_rate.expect("should succeed in test") - 29.97).abs() < f64::EPSILON);
    }

    #[test]
    fn test_playlist_type_display() {
        assert_eq!(PlaylistType::Live.to_string(), "LIVE");
        assert_eq!(PlaylistType::Event.to_string(), "EVENT");
        assert_eq!(PlaylistType::Vod.to_string(), "VOD");
    }

    #[test]
    fn test_media_playlist_basic() {
        let playlist = MediaPlaylistBuilder::new(6)
            .media_sequence(0)
            .playlist_type(PlaylistType::Vod)
            .segment(PlaylistSegment::new(0, Duration::from_secs(6), "seg0.m4s"))
            .segment(PlaylistSegment::new(1, Duration::from_secs(6), "seg1.m4s"))
            .ended()
            .build();

        assert!(playlist.contains("#EXTM3U"));
        assert!(playlist.contains("#EXT-X-TARGETDURATION:6"));
        assert!(playlist.contains("#EXT-X-MEDIA-SEQUENCE:0"));
        assert!(playlist.contains("#EXT-X-PLAYLIST-TYPE:VOD"));
        assert!(playlist.contains("seg0.m4s"));
        assert!(playlist.contains("seg1.m4s"));
        assert!(playlist.contains("#EXT-X-ENDLIST"));
    }

    #[test]
    fn test_media_playlist_live_no_endlist() {
        let playlist = MediaPlaylistBuilder::new(6)
            .segment(PlaylistSegment::new(0, Duration::from_secs(6), "seg0.m4s"))
            .build();

        assert!(!playlist.contains("#EXT-X-ENDLIST"));
    }

    #[test]
    fn test_media_playlist_segment_count() {
        let builder = MediaPlaylistBuilder::new(6)
            .segment(PlaylistSegment::new(0, Duration::from_secs(6), "s0.m4s"))
            .segment(PlaylistSegment::new(1, Duration::from_secs(6), "s1.m4s"));
        assert_eq!(builder.segment_count(), 2);
    }

    #[test]
    fn test_media_playlist_total_duration() {
        let builder = MediaPlaylistBuilder::new(6)
            .segment(PlaylistSegment::new(0, Duration::from_secs(6), "s0.m4s"))
            .segment(PlaylistSegment::new(1, Duration::from_secs(4), "s1.m4s"));
        assert_eq!(builder.total_duration(), Duration::from_secs(10));
    }

    #[test]
    fn test_media_playlist_discontinuity() {
        let playlist = MediaPlaylistBuilder::new(6)
            .segment(PlaylistSegment::new(0, Duration::from_secs(6), "s0.m4s"))
            .segment(PlaylistSegment::new(1, Duration::from_secs(6), "s1.m4s").with_discontinuity())
            .build();

        assert!(playlist.contains("#EXT-X-DISCONTINUITY"));
    }

    #[test]
    fn test_media_playlist_byte_range() {
        let playlist = MediaPlaylistBuilder::new(6)
            .segment(
                PlaylistSegment::new(0, Duration::from_secs(6), "combined.m4s")
                    .with_byte_range(0, 65536),
            )
            .build();

        assert!(playlist.contains("#EXT-X-BYTERANGE:65536@0"));
    }

    #[test]
    fn test_master_playlist() {
        let playlist = MasterPlaylistBuilder::new()
            .variant(VariantStream::new(
                5_000_000,
                1920,
                1080,
                "av01.0.08M.08",
                "1080p/playlist.m3u8",
            ))
            .variant(VariantStream::new(
                3_000_000,
                1280,
                720,
                "av01.0.05M.08",
                "720p/playlist.m3u8",
            ))
            .build();

        assert!(playlist.contains("#EXTM3U"));
        assert!(playlist.contains("#EXT-X-STREAM-INF:"));
        assert!(playlist.contains("1920x1080"));
        assert!(playlist.contains("1280x720"));
        assert!(playlist.contains("1080p/playlist.m3u8"));
        assert!(playlist.contains("720p/playlist.m3u8"));
    }

    #[test]
    fn test_master_playlist_variant_count() {
        let builder = MasterPlaylistBuilder::new()
            .variant(VariantStream::new(
                5_000_000,
                1920,
                1080,
                "av1",
                "1080.m3u8",
            ))
            .variant(VariantStream::new(3_000_000, 1280, 720, "av1", "720.m3u8"));
        assert_eq!(builder.variant_count(), 2);
    }

    #[test]
    fn test_default_builders() {
        let media = MediaPlaylistBuilder::default();
        assert_eq!(media.segment_count(), 0);

        let master = MasterPlaylistBuilder::default();
        assert_eq!(master.variant_count(), 0);
    }
}
