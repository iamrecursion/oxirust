// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Thumbnail / trick-play track generation for adaptive streaming.
//!
//! This module provides:
//!
//! - **HLS I-frame playlists** (`EXT-X-I-FRAMES-ONLY`): media playlists
//!   containing only I-frames (keyframes), allowing players to display
//!   thumbnails or perform fast-forward/rewind scrubbing.
//!
//! - **DASH thumbnail adaptation sets**: `<AdaptationSet>` elements
//!   containing only trick-play representations with reduced resolution
//!   thumbnails on a tiled grid.
//!
//! # HLS I-Frame Playlist
//!
//! Per RFC 8216 §4.3.3.6, an I-frame playlist is referenced from the
//! multivariant playlist via `EXT-X-I-FRAME-STREAM-INF` and uses
//! `EXT-X-BYTERANGE` to address individual I-frames within the full
//! media segments.
//!
//! # DASH Thumbnail Adaptation Set
//!
//! DASH-IF IOP §6.2.6 defines an essential property
//! `http://dashif.org/guidelines/thumbnail_tile` with a value of
//! `<columns>x<rows>` describing how thumbnails are laid out in a tile
//! sprite image.

use crate::error::{PackagerError, PackagerResult};
use std::time::Duration;

// ---------------------------------------------------------------------------
// IFrameEntry
// ---------------------------------------------------------------------------

/// A single I-frame (keyframe) entry for an I-frame playlist.
#[derive(Debug, Clone)]
pub struct IFrameEntry {
    /// Duration of this I-frame's display period (time until next I-frame).
    pub duration: Duration,
    /// Byte offset of this I-frame within the segment/container file.
    pub byte_offset: u64,
    /// Byte length of this I-frame's encoded data.
    pub byte_length: u64,
    /// URI of the segment file (or container file for byte-range addressing).
    pub uri: String,
    /// Decode timestamp in timescale ticks.
    pub decode_time: u64,
}

impl IFrameEntry {
    /// Create a new I-frame entry.
    #[must_use]
    pub fn new(
        duration: Duration,
        byte_offset: u64,
        byte_length: u64,
        uri: impl Into<String>,
    ) -> Self {
        Self {
            duration,
            byte_offset,
            byte_length,
            uri: uri.into(),
            decode_time: 0,
        }
    }

    /// Set the decode timestamp.
    #[must_use]
    pub fn with_decode_time(mut self, decode_time: u64) -> Self {
        self.decode_time = decode_time;
        self
    }

    /// HLS `EXT-X-BYTERANGE` attribute string.
    #[must_use]
    pub fn hls_byterange(&self) -> String {
        format!("{}@{}", self.byte_length, self.byte_offset)
    }
}

// ---------------------------------------------------------------------------
// IFramePlaylist
// ---------------------------------------------------------------------------

/// An HLS I-frame-only playlist (`EXT-X-I-FRAMES-ONLY`).
///
/// This playlist type enables trick-play features (fast forward, rewind,
/// thumbnail scrubbing) by listing only the keyframes from the media.
#[derive(Debug, Clone)]
pub struct IFramePlaylist {
    /// Target duration (maximum I-frame display duration).
    pub target_duration: Duration,
    /// Bandwidth in bits per second for the I-frame stream.
    pub bandwidth: u64,
    /// Video codec string (e.g. `"av01.0.08M.08"`).
    pub codecs: String,
    /// Resolution width.
    pub width: u32,
    /// Resolution height.
    pub height: u32,
    /// I-frame entries.
    entries: Vec<IFrameEntry>,
    /// URI for the `EXT-X-MAP` init segment.
    pub init_uri: Option<String>,
    /// Byte range for the init segment (if byte-range addressed).
    pub init_byterange: Option<(u64, u64)>,
}

impl IFramePlaylist {
    /// Create a new I-frame playlist.
    #[must_use]
    pub fn new(bandwidth: u64, codecs: impl Into<String>, width: u32, height: u32) -> Self {
        Self {
            target_duration: Duration::ZERO,
            bandwidth,
            codecs: codecs.into(),
            width,
            height,
            entries: Vec::new(),
            init_uri: None,
            init_byterange: None,
        }
    }

    /// Set the init segment URI.
    #[must_use]
    pub fn with_init_uri(mut self, uri: impl Into<String>) -> Self {
        self.init_uri = Some(uri.into());
        self
    }

    /// Set the init segment byte range (for single-file addressing).
    #[must_use]
    pub fn with_init_byterange(mut self, offset: u64, length: u64) -> Self {
        self.init_byterange = Some((offset, length));
        self
    }

    /// Add an I-frame entry.
    pub fn add_entry(&mut self, entry: IFrameEntry) {
        // Update target duration if this entry is longer
        if entry.duration > self.target_duration {
            self.target_duration = entry.duration;
        }
        self.entries.push(entry);
    }

    /// Return all entries.
    #[must_use]
    pub fn entries(&self) -> &[IFrameEntry] {
        &self.entries
    }

    /// Return the number of I-frame entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return `true` if no entries have been added.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Resolution string (e.g. `"1920x1080"`).
    #[must_use]
    pub fn resolution_string(&self) -> String {
        format!("{}x{}", self.width, self.height)
    }

    /// Render the I-frame playlist as an M3U8 string.
    ///
    /// Produces a complete `EXT-X-I-FRAMES-ONLY` media playlist.
    #[must_use]
    pub fn to_m3u8(&self) -> String {
        let mut out = String::new();
        out.push_str("#EXTM3U\n");
        out.push_str("#EXT-X-VERSION:6\n");
        let target_secs = self.target_duration.as_secs().max(1);
        out.push_str(&format!("#EXT-X-TARGETDURATION:{target_secs}\n"));
        out.push_str("#EXT-X-I-FRAMES-ONLY\n");

        // Init segment map
        if let Some(init_uri) = &self.init_uri {
            if let Some((offset, length)) = self.init_byterange {
                out.push_str(&format!(
                    "#EXT-X-MAP:URI=\"{init_uri}\",BYTERANGE=\"{length}@{offset}\"\n"
                ));
            } else {
                out.push_str(&format!("#EXT-X-MAP:URI=\"{init_uri}\"\n"));
            }
        }

        for entry in &self.entries {
            let secs = entry.duration.as_secs_f64();
            out.push_str(&format!("#EXTINF:{secs:.6},\n"));
            out.push_str(&format!("#EXT-X-BYTERANGE:{}\n", entry.hls_byterange()));
            out.push_str(&entry.uri);
            out.push('\n');
        }

        out.push_str("#EXT-X-ENDLIST\n");
        out
    }

    /// Render an `EXT-X-I-FRAME-STREAM-INF` tag for the multivariant playlist.
    ///
    /// `playlist_uri` is the relative URI of this I-frame playlist file.
    #[must_use]
    pub fn to_stream_inf_tag(&self, playlist_uri: &str) -> String {
        format!(
            "#EXT-X-I-FRAME-STREAM-INF:BANDWIDTH={},CODECS=\"{}\",RESOLUTION={},URI=\"{}\"",
            self.bandwidth,
            self.codecs,
            self.resolution_string(),
            playlist_uri
        )
    }

    /// Validate the playlist.
    ///
    /// # Errors
    ///
    /// Returns an error if the playlist is empty or bandwidth is zero.
    pub fn validate(&self) -> PackagerResult<()> {
        if self.bandwidth == 0 {
            return Err(PackagerError::InvalidConfig(
                "I-frame playlist bandwidth must not be zero".into(),
            ));
        }
        if self.width == 0 || self.height == 0 {
            return Err(PackagerError::InvalidConfig(
                "I-frame playlist resolution must not be zero".into(),
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ThumbnailTile
// ---------------------------------------------------------------------------

/// Configuration for a DASH thumbnail tile grid.
///
/// Thumbnails are arranged in a sprite image of `columns x rows` tiles.
/// Each tile is a small video frame at reduced resolution.
#[derive(Debug, Clone)]
pub struct ThumbnailTile {
    /// Number of columns in the tile grid.
    pub columns: u32,
    /// Number of rows in the tile grid.
    pub rows: u32,
    /// Width of each individual thumbnail in pixels.
    pub thumb_width: u32,
    /// Height of each individual thumbnail in pixels.
    pub thumb_height: u32,
    /// Interval between thumbnails.
    pub interval: Duration,
}

impl ThumbnailTile {
    /// Create a new thumbnail tile configuration.
    #[must_use]
    pub fn new(columns: u32, rows: u32, thumb_width: u32, thumb_height: u32) -> Self {
        Self {
            columns,
            rows,
            thumb_width,
            thumb_height,
            interval: Duration::from_secs(10),
        }
    }

    /// Set the thumbnail interval.
    #[must_use]
    pub fn with_interval(mut self, interval: Duration) -> Self {
        self.interval = interval;
        self
    }

    /// Total number of thumbnails per tile image.
    #[must_use]
    pub fn thumbnails_per_tile(&self) -> u32 {
        self.columns * self.rows
    }

    /// Total tile image width in pixels.
    #[must_use]
    pub fn tile_width(&self) -> u32 {
        self.columns * self.thumb_width
    }

    /// Total tile image height in pixels.
    #[must_use]
    pub fn tile_height(&self) -> u32 {
        self.rows * self.thumb_height
    }

    /// DASH-IF thumbnail tile property value: `"<columns>x<rows>"`.
    #[must_use]
    pub fn tile_property_value(&self) -> String {
        format!("{}x{}", self.columns, self.rows)
    }

    /// Validate the tile configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are zero.
    pub fn validate(&self) -> PackagerResult<()> {
        if self.columns == 0 || self.rows == 0 {
            return Err(PackagerError::InvalidConfig(
                "Thumbnail tile columns and rows must be greater than zero".into(),
            ));
        }
        if self.thumb_width == 0 || self.thumb_height == 0 {
            return Err(PackagerError::InvalidConfig(
                "Thumbnail dimensions must be greater than zero".into(),
            ));
        }
        if self.interval.is_zero() {
            return Err(PackagerError::InvalidConfig(
                "Thumbnail interval must be greater than zero".into(),
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ThumbnailTrackEntry
// ---------------------------------------------------------------------------

/// A single thumbnail entry referencing a specific region of a tile image.
#[derive(Debug, Clone)]
pub struct ThumbnailTrackEntry {
    /// Presentation time of this thumbnail.
    pub time: Duration,
    /// Duration this thumbnail covers.
    pub duration: Duration,
    /// URI of the tile sprite image.
    pub tile_uri: String,
    /// Column index (0-based) within the tile.
    pub column: u32,
    /// Row index (0-based) within the tile.
    pub row: u32,
}

impl ThumbnailTrackEntry {
    /// Create a new thumbnail entry.
    #[must_use]
    pub fn new(
        time: Duration,
        duration: Duration,
        tile_uri: impl Into<String>,
        column: u32,
        row: u32,
    ) -> Self {
        Self {
            time,
            duration,
            tile_uri: tile_uri.into(),
            column,
            row,
        }
    }

    /// Spatial fragment identifier for DASH Media Fragments URI:
    /// `xywh=pixel,<x>,<y>,<w>,<h>`.
    #[must_use]
    pub fn spatial_fragment(&self, thumb_width: u32, thumb_height: u32) -> String {
        let x = self.column * thumb_width;
        let y = self.row * thumb_height;
        format!("xywh=pixel,{},{},{},{}", x, y, thumb_width, thumb_height)
    }
}

// ---------------------------------------------------------------------------
// ThumbnailTrack
// ---------------------------------------------------------------------------

/// A thumbnail / trick-play track that can be rendered as either an
/// HLS I-frame playlist or a DASH thumbnail adaptation set.
#[derive(Debug, Clone)]
pub struct ThumbnailTrack {
    /// Tile configuration.
    pub tile: ThumbnailTile,
    /// Bandwidth in bits per second.
    pub bandwidth: u64,
    /// Video codec string.
    pub codecs: String,
    /// Thumbnail entries.
    entries: Vec<ThumbnailTrackEntry>,
}

impl ThumbnailTrack {
    /// Create a new thumbnail track.
    #[must_use]
    pub fn new(tile: ThumbnailTile, bandwidth: u64, codecs: impl Into<String>) -> Self {
        Self {
            tile,
            bandwidth,
            codecs: codecs.into(),
            entries: Vec::new(),
        }
    }

    /// Add a thumbnail entry.
    pub fn add_entry(&mut self, entry: ThumbnailTrackEntry) {
        self.entries.push(entry);
    }

    /// Return all entries.
    #[must_use]
    pub fn entries(&self) -> &[ThumbnailTrackEntry] {
        &self.entries
    }

    /// Number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the track is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Compute the number of tile images needed.
    #[must_use]
    pub fn tile_count(&self) -> usize {
        let per = self.tile.thumbnails_per_tile() as usize;
        if per == 0 {
            return 0;
        }
        (self.entries.len() + per - 1) / per
    }

    /// Generate a DASH `<AdaptationSet>` XML fragment for thumbnails.
    ///
    /// Uses the DASH-IF thumbnail tile essential property.
    #[must_use]
    pub fn to_dash_adaptation_set(&self) -> String {
        let mut xml = String::new();
        xml.push_str(&format!(
            r#"<AdaptationSet mimeType="image/jpeg" contentType="image">"#
        ));
        xml.push_str(&format!(
            r#"<EssentialProperty schemeIdUri="http://dashif.org/guidelines/thumbnail_tile" value="{}"/>"#,
            self.tile.tile_property_value()
        ));
        xml.push_str(&format!(
            r#"<Representation id="thumbnails" bandwidth="{}" width="{}" height="{}">"#,
            self.bandwidth,
            self.tile.tile_width(),
            self.tile.tile_height()
        ));

        // Segment list of tile images
        let per = self.tile.thumbnails_per_tile() as usize;
        if per > 0 {
            xml.push_str("<SegmentList>");
            let chunk_count = self.tile_count();
            for i in 0..chunk_count {
                let start_idx = i * per;
                if let Some(entry) = self.entries.get(start_idx) {
                    xml.push_str(&format!(r#"<SegmentURL media="{}"/>"#, entry.tile_uri));
                }
            }
            xml.push_str("</SegmentList>");
        }

        xml.push_str("</Representation>");
        xml.push_str("</AdaptationSet>");
        xml
    }

    /// Validate the thumbnail track.
    ///
    /// # Errors
    ///
    /// Returns an error if the tile config is invalid.
    pub fn validate(&self) -> PackagerResult<()> {
        self.tile.validate()?;
        if self.bandwidth == 0 {
            return Err(PackagerError::InvalidConfig(
                "Thumbnail track bandwidth must not be zero".into(),
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// IFramePlaylistBuilder
// ---------------------------------------------------------------------------

/// Builder for constructing I-frame playlists from a stream of keyframe
/// positions and byte ranges.
pub struct IFramePlaylistBuilder {
    bandwidth: u64,
    codecs: String,
    width: u32,
    height: u32,
    init_uri: Option<String>,
    init_byterange: Option<(u64, u64)>,
    entries: Vec<IFrameEntry>,
}

impl IFramePlaylistBuilder {
    /// Create a new builder.
    #[must_use]
    pub fn new(bandwidth: u64, codecs: impl Into<String>, width: u32, height: u32) -> Self {
        Self {
            bandwidth,
            codecs: codecs.into(),
            width,
            height,
            init_uri: None,
            init_byterange: None,
            entries: Vec::new(),
        }
    }

    /// Set init segment URI.
    #[must_use]
    pub fn init_uri(mut self, uri: impl Into<String>) -> Self {
        self.init_uri = Some(uri.into());
        self
    }

    /// Set init segment byte range.
    #[must_use]
    pub fn init_byterange(mut self, offset: u64, length: u64) -> Self {
        self.init_byterange = Some((offset, length));
        self
    }

    /// Add an I-frame entry.
    #[must_use]
    pub fn entry(mut self, entry: IFrameEntry) -> Self {
        self.entries.push(entry);
        self
    }

    /// Build the I-frame playlist.
    #[must_use]
    pub fn build(self) -> IFramePlaylist {
        let mut playlist =
            IFramePlaylist::new(self.bandwidth, self.codecs, self.width, self.height);
        if let Some(uri) = self.init_uri {
            playlist.init_uri = Some(uri);
        }
        playlist.init_byterange = self.init_byterange;
        for entry in self.entries {
            playlist.add_entry(entry);
        }
        playlist
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn dur(s: u64) -> Duration {
        Duration::from_secs(s)
    }

    fn dur_ms(ms: u64) -> Duration {
        Duration::from_millis(ms)
    }

    // --- IFrameEntry --------------------------------------------------------

    #[test]
    fn test_iframe_entry_new() {
        let e = IFrameEntry::new(dur(6), 1024, 512, "seg0.m4s");
        assert_eq!(e.duration, dur(6));
        assert_eq!(e.byte_offset, 1024);
        assert_eq!(e.byte_length, 512);
        assert_eq!(e.uri, "seg0.m4s");
        assert_eq!(e.decode_time, 0);
    }

    #[test]
    fn test_iframe_entry_with_decode_time() {
        let e = IFrameEntry::new(dur(6), 0, 100, "seg.m4s").with_decode_time(540_000);
        assert_eq!(e.decode_time, 540_000);
    }

    #[test]
    fn test_iframe_entry_hls_byterange() {
        let e = IFrameEntry::new(dur(6), 256, 1024, "seg.m4s");
        assert_eq!(e.hls_byterange(), "1024@256");
    }

    // --- IFramePlaylist -----------------------------------------------------

    #[test]
    fn test_iframe_playlist_new() {
        let p = IFramePlaylist::new(500_000, "av01.0.08M.08", 1920, 1080);
        assert_eq!(p.bandwidth, 500_000);
        assert_eq!(p.width, 1920);
        assert_eq!(p.height, 1080);
        assert!(p.is_empty());
    }

    #[test]
    fn test_iframe_playlist_add_entry() {
        let mut p = IFramePlaylist::new(500_000, "av01.0.08M.08", 1920, 1080);
        p.add_entry(IFrameEntry::new(dur(6), 0, 512, "seg0.m4s"));
        p.add_entry(IFrameEntry::new(dur(4), 512, 256, "seg0.m4s"));
        assert_eq!(p.len(), 2);
        assert_eq!(p.target_duration, dur(6));
    }

    #[test]
    fn test_iframe_playlist_to_m3u8() {
        let mut p =
            IFramePlaylist::new(500_000, "av01.0.08M.08", 1920, 1080).with_init_uri("init.mp4");
        p.add_entry(IFrameEntry::new(dur(6), 256, 1024, "video.mp4"));
        p.add_entry(IFrameEntry::new(dur(6), 1280, 900, "video.mp4"));

        let m3u8 = p.to_m3u8();
        assert!(m3u8.contains("#EXTM3U"));
        assert!(m3u8.contains("#EXT-X-I-FRAMES-ONLY"));
        assert!(m3u8.contains("#EXT-X-MAP:URI=\"init.mp4\""));
        assert!(m3u8.contains("#EXT-X-BYTERANGE:1024@256"));
        assert!(m3u8.contains("#EXT-X-BYTERANGE:900@1280"));
        assert!(m3u8.contains("#EXT-X-ENDLIST"));
    }

    #[test]
    fn test_iframe_playlist_to_m3u8_with_init_byterange() {
        let mut p = IFramePlaylist::new(500_000, "av01.0.08M.08", 1920, 1080)
            .with_init_uri("video.mp4")
            .with_init_byterange(0, 512);
        p.add_entry(IFrameEntry::new(dur(6), 512, 1024, "video.mp4"));

        let m3u8 = p.to_m3u8();
        assert!(m3u8.contains("BYTERANGE=\"512@0\""));
    }

    #[test]
    fn test_iframe_playlist_stream_inf_tag() {
        let p = IFramePlaylist::new(500_000, "av01.0.08M.08", 1920, 1080);
        let tag = p.to_stream_inf_tag("iframe.m3u8");
        assert!(tag.contains("EXT-X-I-FRAME-STREAM-INF"));
        assert!(tag.contains("BANDWIDTH=500000"));
        assert!(tag.contains("CODECS=\"av01.0.08M.08\""));
        assert!(tag.contains("RESOLUTION=1920x1080"));
        assert!(tag.contains("URI=\"iframe.m3u8\""));
    }

    #[test]
    fn test_iframe_playlist_validate_ok() {
        let p = IFramePlaylist::new(500_000, "av01.0.08M.08", 1920, 1080);
        assert!(p.validate().is_ok());
    }

    #[test]
    fn test_iframe_playlist_validate_zero_bandwidth() {
        let p = IFramePlaylist::new(0, "av01.0.08M.08", 1920, 1080);
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_iframe_playlist_validate_zero_resolution() {
        let p = IFramePlaylist::new(500_000, "av01.0.08M.08", 0, 1080);
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_iframe_playlist_resolution_string() {
        let p = IFramePlaylist::new(500_000, "av01.0.08M.08", 1280, 720);
        assert_eq!(p.resolution_string(), "1280x720");
    }

    // --- ThumbnailTile ------------------------------------------------------

    #[test]
    fn test_thumbnail_tile_new() {
        let t = ThumbnailTile::new(10, 10, 160, 90);
        assert_eq!(t.columns, 10);
        assert_eq!(t.rows, 10);
        assert_eq!(t.thumbnails_per_tile(), 100);
        assert_eq!(t.tile_width(), 1600);
        assert_eq!(t.tile_height(), 900);
    }

    #[test]
    fn test_thumbnail_tile_property_value() {
        let t = ThumbnailTile::new(5, 4, 160, 90);
        assert_eq!(t.tile_property_value(), "5x4");
    }

    #[test]
    fn test_thumbnail_tile_with_interval() {
        let t = ThumbnailTile::new(10, 10, 160, 90).with_interval(dur(5));
        assert_eq!(t.interval, dur(5));
    }

    #[test]
    fn test_thumbnail_tile_validate_ok() {
        let t = ThumbnailTile::new(10, 10, 160, 90);
        assert!(t.validate().is_ok());
    }

    #[test]
    fn test_thumbnail_tile_validate_zero_columns() {
        let t = ThumbnailTile::new(0, 10, 160, 90);
        assert!(t.validate().is_err());
    }

    #[test]
    fn test_thumbnail_tile_validate_zero_thumb_width() {
        let t = ThumbnailTile::new(10, 10, 0, 90);
        assert!(t.validate().is_err());
    }

    #[test]
    fn test_thumbnail_tile_validate_zero_interval() {
        let t = ThumbnailTile::new(10, 10, 160, 90).with_interval(Duration::ZERO);
        assert!(t.validate().is_err());
    }

    // --- ThumbnailTrackEntry ------------------------------------------------

    #[test]
    fn test_thumbnail_entry_new() {
        let e = ThumbnailTrackEntry::new(dur(0), dur(10), "tile0.jpg", 0, 0);
        assert_eq!(e.time, dur(0));
        assert_eq!(e.tile_uri, "tile0.jpg");
        assert_eq!(e.column, 0);
        assert_eq!(e.row, 0);
    }

    #[test]
    fn test_thumbnail_entry_spatial_fragment() {
        let e = ThumbnailTrackEntry::new(dur(10), dur(10), "tile0.jpg", 3, 2);
        let frag = e.spatial_fragment(160, 90);
        assert_eq!(frag, "xywh=pixel,480,180,160,90");
    }

    // --- ThumbnailTrack -----------------------------------------------------

    #[test]
    fn test_thumbnail_track_new() {
        let tile = ThumbnailTile::new(10, 10, 160, 90);
        let t = ThumbnailTrack::new(tile, 50_000, "jpeg");
        assert!(t.is_empty());
        assert_eq!(t.bandwidth, 50_000);
    }

    #[test]
    fn test_thumbnail_track_add_entries() {
        let tile = ThumbnailTile::new(5, 4, 160, 90);
        let mut t = ThumbnailTrack::new(tile, 50_000, "jpeg");
        for i in 0..25u32 {
            t.add_entry(ThumbnailTrackEntry::new(
                dur_ms(i as u64 * 10_000),
                dur(10),
                format!("tile{}.jpg", i / 20),
                i % 5,
                (i / 5) % 4,
            ));
        }
        assert_eq!(t.len(), 25);
        assert_eq!(t.tile_count(), 2); // ceil(25 / 20) = 2
    }

    #[test]
    fn test_thumbnail_track_to_dash_adaptation_set() {
        let tile = ThumbnailTile::new(5, 4, 160, 90);
        let mut t = ThumbnailTrack::new(tile, 50_000, "jpeg");
        t.add_entry(ThumbnailTrackEntry::new(dur(0), dur(10), "tile0.jpg", 0, 0));

        let xml = t.to_dash_adaptation_set();
        assert!(xml.contains("AdaptationSet"));
        assert!(xml.contains("image/jpeg"));
        assert!(xml.contains("thumbnail_tile"));
        assert!(xml.contains("5x4"));
        assert!(xml.contains("tile0.jpg"));
    }

    #[test]
    fn test_thumbnail_track_validate_ok() {
        let tile = ThumbnailTile::new(10, 10, 160, 90);
        let t = ThumbnailTrack::new(tile, 50_000, "jpeg");
        assert!(t.validate().is_ok());
    }

    #[test]
    fn test_thumbnail_track_validate_zero_bandwidth() {
        let tile = ThumbnailTile::new(10, 10, 160, 90);
        let t = ThumbnailTrack::new(tile, 0, "jpeg");
        assert!(t.validate().is_err());
    }

    // --- IFramePlaylistBuilder ----------------------------------------------

    #[test]
    fn test_iframe_playlist_builder() {
        let p = IFramePlaylistBuilder::new(500_000, "av01.0.08M.08", 1920, 1080)
            .init_uri("init.mp4")
            .entry(IFrameEntry::new(dur(6), 256, 1024, "video.mp4"))
            .entry(IFrameEntry::new(dur(6), 1280, 900, "video.mp4"))
            .build();

        assert_eq!(p.len(), 2);
        assert!(p.init_uri.is_some());
        assert_eq!(p.target_duration, dur(6));
    }

    #[test]
    fn test_iframe_playlist_builder_with_byterange() {
        let p = IFramePlaylistBuilder::new(500_000, "av01.0.08M.08", 1920, 1080)
            .init_uri("video.mp4")
            .init_byterange(0, 512)
            .build();

        assert_eq!(p.init_byterange, Some((0, 512)));
    }
}
