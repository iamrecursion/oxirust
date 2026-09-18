//! HLS playlist builders.

use crate::hls::{MasterPlaylist, MediaPlaylist, Segment, StreamInf, VariantStream};
use std::time::Duration;

/// Master playlist builder.
pub struct MasterPlaylistBuilder {
    playlist: MasterPlaylist,
}

impl MasterPlaylistBuilder {
    /// Creates a new master playlist builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            playlist: MasterPlaylist::new(),
        }
    }

    /// Sets HLS version.
    #[must_use]
    pub fn version(mut self, version: u8) -> Self {
        self.playlist.version = version;
        self
    }

    /// Enables independent segments.
    #[must_use]
    pub fn independent_segments(mut self) -> Self {
        self.playlist.independent_segments = true;
        self
    }

    /// Adds a variant stream.
    #[must_use]
    pub fn add_variant(
        mut self,
        bandwidth: u64,
        resolution: (u32, u32),
        codecs: impl Into<String>,
        uri: impl Into<String>,
    ) -> Self {
        let stream_inf = StreamInf {
            bandwidth,
            average_bandwidth: Some(bandwidth),
            codecs: Some(codecs.into()),
            resolution: Some(resolution),
            frame_rate: Some(30.0),
            hdcp_level: None,
            audio: None,
            video: None,
            subtitles: None,
            closed_captions: None,
        };

        self.playlist.variants.push(VariantStream {
            stream_inf,
            uri: uri.into(),
        });

        self
    }

    /// Builds the master playlist.
    #[must_use]
    pub fn build(self) -> MasterPlaylist {
        self.playlist
    }
}

impl Default for MasterPlaylistBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Media playlist builder.
pub struct MediaPlaylistBuilder {
    playlist: MediaPlaylist,
}

impl MediaPlaylistBuilder {
    /// Creates a new media playlist builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            playlist: MediaPlaylist::new(),
        }
    }

    /// Sets HLS version.
    #[must_use]
    pub fn version(mut self, version: u8) -> Self {
        self.playlist.version = version;
        self
    }

    /// Sets target duration.
    #[must_use]
    pub fn target_duration(mut self, duration: u64) -> Self {
        self.playlist.target_duration = duration;
        self
    }

    /// Sets media sequence.
    #[must_use]
    pub fn media_sequence(mut self, sequence: u64) -> Self {
        self.playlist.media_sequence = sequence;
        self
    }

    /// Sets playlist type.
    #[must_use]
    pub fn playlist_type(mut self, ptype: crate::hls::PlaylistType) -> Self {
        self.playlist.playlist_type = Some(ptype);
        self
    }

    /// Adds a segment.
    #[must_use]
    pub fn add_segment(mut self, duration: Duration, uri: impl Into<String>) -> Self {
        self.playlist
            .segments
            .push(Segment::new(duration, uri.into()));
        self
    }

    /// Marks playlist as ended.
    #[must_use]
    pub fn ended(mut self) -> Self {
        self.playlist.ended = true;
        self
    }

    /// Builds the media playlist.
    #[must_use]
    pub fn build(self) -> MediaPlaylist {
        self.playlist
    }
}

impl Default for MediaPlaylistBuilder {
    fn default() -> Self {
        Self::new()
    }
}
