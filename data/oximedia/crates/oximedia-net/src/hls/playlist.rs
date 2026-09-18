//! HLS playlist parsing and writing.
//!
//! This module provides types for parsing and generating M3U8 playlists,
//! including both master playlists and media playlists.

#![allow(dead_code)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::similar_names)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::range_plus_one)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::comparison_chain)]
#![allow(clippy::unused_self)]
#![allow(clippy::trivially_copy_pass_by_ref)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::redundant_closure_for_method_calls)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::should_implement_trait)]
#![allow(clippy::items_after_statements)]
#![allow(clippy::if_not_else)]
#![allow(clippy::format_push_string)]
#![allow(clippy::single_match_else)]
#![allow(clippy::redundant_slicing)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::map_unwrap_or)]
#![allow(clippy::derivable_impls)]
#![allow(clippy::assigning_clones)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::format_collect)]
#![allow(clippy::useless_conversion)]
#![allow(clippy::unused_async)]
#![allow(clippy::identity_op)]
use crate::error::{NetError, NetResult};
use std::collections::HashMap;
use std::str::FromStr;
use std::time::Duration;

/// HLS playlist tag types.
#[derive(Debug, Clone, PartialEq)]
pub enum PlaylistTag {
    /// `#EXTM3U` - File header.
    ExtM3U,
    /// `#EXT-X-VERSION` - Protocol version.
    ExtXVersion(u8),
    /// `#EXT-X-TARGETDURATION` - Maximum segment duration.
    ExtXTargetDuration(u64),
    /// `#EXT-X-MEDIA-SEQUENCE` - First segment sequence number.
    ExtXMediaSequence(u64),
    /// `#EXT-X-DISCONTINUITY-SEQUENCE` - Discontinuity sequence.
    ExtXDiscontinuitySequence(u64),
    /// `#EXT-X-ENDLIST` - End of playlist marker.
    ExtXEndList,
    /// `#EXT-X-PLAYLIST-TYPE` - VOD or EVENT.
    ExtXPlaylistType(PlaylistType),
    /// `#EXTINF` - Segment duration and optional title.
    ExtInf(f64, Option<String>),
    /// `#EXT-X-BYTERANGE` - Byte range for segment.
    ExtXByteRange(u64, Option<u64>),
    /// `#EXT-X-DISCONTINUITY` - Discontinuity marker.
    ExtXDiscontinuity,
    /// `#EXT-X-KEY` - Encryption key info.
    ExtXKey(KeyInfo),
    /// `#EXT-X-MAP` - Initialization segment.
    ExtXMap(MapInfo),
    /// `#EXT-X-STREAM-INF` - Variant stream info.
    ExtXStreamInf(StreamInf),
    /// `#EXT-X-MEDIA` - Rendition.
    ExtXMedia(MediaInfo),
    /// `#EXT-X-INDEPENDENT-SEGMENTS` - Independent segments flag.
    ExtXIndependentSegments,
    /// `#EXT-X-START` - Preferred start point.
    ExtXStart(f64, Option<bool>),
    /// Unknown or unsupported tag.
    Unknown(String, Option<String>),
}

/// Playlist type (VOD or EVENT).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaylistType {
    /// Video on demand - complete playlist.
    Vod,
    /// Event - playlist may grow.
    Event,
}

impl FromStr for PlaylistType {
    type Err = NetError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "VOD" => Ok(Self::Vod),
            "EVENT" => Ok(Self::Event),
            _ => Err(NetError::parse(0, format!("Unknown playlist type: {s}"))),
        }
    }
}

/// Media type for renditions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaType {
    /// Audio rendition.
    Audio,
    /// Video rendition.
    Video,
    /// Subtitle rendition.
    Subtitles,
    /// Closed captions.
    ClosedCaptions,
}

impl FromStr for MediaType {
    type Err = NetError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "AUDIO" => Ok(Self::Audio),
            "VIDEO" => Ok(Self::Video),
            "SUBTITLES" => Ok(Self::Subtitles),
            "CLOSED-CAPTIONS" => Ok(Self::ClosedCaptions),
            _ => Err(NetError::parse(0, format!("Unknown media type: {s}"))),
        }
    }
}

/// Encryption key information.
#[derive(Debug, Clone, PartialEq)]
pub struct KeyInfo {
    /// Encryption method (NONE, AES-128, SAMPLE-AES).
    pub method: String,
    /// URI for key file.
    pub uri: Option<String>,
    /// Initialization vector.
    pub iv: Option<Vec<u8>>,
    /// Key format.
    pub keyformat: Option<String>,
    /// Key format versions.
    pub keyformatversions: Option<String>,
}

impl Default for KeyInfo {
    fn default() -> Self {
        Self {
            method: "NONE".to_string(),
            uri: None,
            iv: None,
            keyformat: None,
            keyformatversions: None,
        }
    }
}

/// Initialization segment information.
#[derive(Debug, Clone, PartialEq)]
pub struct MapInfo {
    /// URI of the initialization segment.
    pub uri: String,
    /// Byte range within the resource.
    pub byterange: Option<(u64, Option<u64>)>,
}

/// Stream info for variant streams.
#[derive(Debug, Clone, PartialEq)]
pub struct StreamInf {
    /// Peak bandwidth in bits per second.
    pub bandwidth: u64,
    /// Average bandwidth in bits per second.
    pub average_bandwidth: Option<u64>,
    /// Codec string.
    pub codecs: Option<String>,
    /// Resolution (width x height).
    pub resolution: Option<(u32, u32)>,
    /// Frame rate.
    pub frame_rate: Option<f64>,
    /// HDCP level.
    pub hdcp_level: Option<String>,
    /// Audio group ID.
    pub audio: Option<String>,
    /// Video group ID.
    pub video: Option<String>,
    /// Subtitles group ID.
    pub subtitles: Option<String>,
    /// Closed captions group ID.
    pub closed_captions: Option<String>,
}

impl Default for StreamInf {
    fn default() -> Self {
        Self {
            bandwidth: 0,
            average_bandwidth: None,
            codecs: None,
            resolution: None,
            frame_rate: None,
            hdcp_level: None,
            audio: None,
            video: None,
            subtitles: None,
            closed_captions: None,
        }
    }
}

/// Media rendition information.
#[derive(Debug, Clone, PartialEq)]
pub struct MediaInfo {
    /// Media type.
    pub media_type: MediaType,
    /// URI for the rendition playlist.
    pub uri: Option<String>,
    /// Group ID.
    pub group_id: String,
    /// Language tag.
    pub language: Option<String>,
    /// Associated language.
    pub assoc_language: Option<String>,
    /// Human-readable name.
    pub name: String,
    /// Default rendition flag.
    pub default: bool,
    /// Autoselect flag.
    pub autoselect: bool,
    /// Forced rendition flag.
    pub forced: bool,
    /// Instream ID for closed captions.
    pub instream_id: Option<String>,
    /// Characteristics.
    pub characteristics: Option<String>,
    /// Channels (audio).
    pub channels: Option<String>,
}

impl Default for MediaInfo {
    fn default() -> Self {
        Self {
            media_type: MediaType::Audio,
            uri: None,
            group_id: String::new(),
            language: None,
            assoc_language: None,
            name: String::new(),
            default: false,
            autoselect: false,
            forced: false,
            instream_id: None,
            characteristics: None,
            channels: None,
        }
    }
}

/// A variant stream in a master playlist.
#[derive(Debug, Clone)]
pub struct VariantStream {
    /// Stream info.
    pub stream_inf: StreamInf,
    /// URI of the media playlist.
    pub uri: String,
}

/// A segment in a media playlist.
#[derive(Debug, Clone)]
pub struct Segment {
    /// Segment duration.
    pub duration: Duration,
    /// Segment title/description.
    pub title: Option<String>,
    /// Segment URI.
    pub uri: String,
    /// Byte range within the resource.
    pub byte_range: Option<(u64, Option<u64>)>,
    /// Discontinuity flag.
    pub discontinuity: bool,
    /// Encryption key.
    pub key: Option<KeyInfo>,
    /// Initialization map.
    pub map: Option<MapInfo>,
    /// Program date/time.
    pub program_date_time: Option<String>,
}

impl Segment {
    /// Creates a new segment with the given duration and URI.
    #[must_use]
    pub fn new(duration: Duration, uri: impl Into<String>) -> Self {
        Self {
            duration,
            title: None,
            uri: uri.into(),
            byte_range: None,
            discontinuity: false,
            key: None,
            map: None,
            program_date_time: None,
        }
    }

    /// Sets the byte range for this segment.
    #[must_use]
    pub fn with_byte_range(mut self, length: u64, offset: Option<u64>) -> Self {
        self.byte_range = Some((length, offset));
        self
    }

    /// Sets the discontinuity flag.
    #[must_use]
    pub fn with_discontinuity(mut self) -> Self {
        self.discontinuity = true;
        self
    }

    /// Returns true if this segment has a byte range.
    #[must_use]
    pub const fn has_byte_range(&self) -> bool {
        self.byte_range.is_some()
    }
}

/// HLS master playlist containing variant streams.
#[derive(Debug, Clone, Default)]
pub struct MasterPlaylist {
    /// HLS version.
    pub version: u8,
    /// Independent segments flag.
    pub independent_segments: bool,
    /// Start time offset.
    pub start: Option<(f64, bool)>,
    /// Variant streams.
    pub variants: Vec<VariantStream>,
    /// Media renditions.
    pub media: Vec<MediaInfo>,
    /// Base URI for resolving relative URLs.
    pub base_uri: Option<String>,
}

impl MasterPlaylist {
    /// Creates a new empty master playlist.
    #[must_use]
    pub fn new() -> Self {
        Self {
            version: 3,
            ..Default::default()
        }
    }

    /// Parses a master playlist from M3U8 data.
    ///
    /// # Errors
    ///
    /// Returns an error if the playlist is malformed.
    pub fn parse(data: &str) -> NetResult<Self> {
        let mut playlist = Self::new();
        let mut lines = data.lines().peekable();
        let mut current_stream_inf: Option<StreamInf> = None;

        // Check for EXTM3U header
        match lines.next() {
            Some(line) if line.trim() == "#EXTM3U" => {}
            _ => return Err(NetError::playlist("Missing #EXTM3U header")),
        }

        for line in lines {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            if line.starts_with("#EXT-X-VERSION:") {
                if let Some(ver) = line.strip_prefix("#EXT-X-VERSION:") {
                    playlist.version = ver.parse().unwrap_or(3);
                }
            } else if line.starts_with("#EXT-X-INDEPENDENT-SEGMENTS") {
                playlist.independent_segments = true;
            } else if line.starts_with("#EXT-X-STREAM-INF:") {
                current_stream_inf = Some(parse_stream_inf(line)?);
            } else if line.starts_with("#EXT-X-MEDIA:") {
                playlist.media.push(parse_media_info(line)?);
            } else if line.starts_with("#EXT-X-START:") {
                playlist.start = parse_start_tag(line)?;
            } else if !line.starts_with('#') {
                // URI line
                if let Some(stream_inf) = current_stream_inf.take() {
                    playlist.variants.push(VariantStream {
                        stream_inf,
                        uri: line.to_string(),
                    });
                }
            }
        }

        Ok(playlist)
    }

    /// Returns the variant stream best suited for the given bandwidth.
    #[must_use]
    pub fn best_variant_for_bandwidth(&self, bandwidth: u64) -> Option<&VariantStream> {
        let mut best: Option<&VariantStream> = None;
        for variant in &self.variants {
            if variant.stream_inf.bandwidth <= bandwidth {
                match best {
                    Some(b) if b.stream_inf.bandwidth >= variant.stream_inf.bandwidth => {}
                    _ => best = Some(variant),
                }
            }
        }
        // If no variant fits, return the lowest bandwidth
        best.or_else(|| self.variants.iter().min_by_key(|v| v.stream_inf.bandwidth))
    }

    /// Returns all variants sorted by bandwidth (ascending).
    #[must_use]
    pub fn variants_by_bandwidth(&self) -> Vec<&VariantStream> {
        let mut sorted: Vec<_> = self.variants.iter().collect();
        sorted.sort_by_key(|v| v.stream_inf.bandwidth);
        sorted
    }

    /// Writes the playlist to M3U8 format.
    #[must_use]
    pub fn to_m3u8(&self) -> String {
        let mut out = String::new();
        out.push_str("#EXTM3U\n");
        out.push_str(&format!("#EXT-X-VERSION:{}\n", self.version));

        if self.independent_segments {
            out.push_str("#EXT-X-INDEPENDENT-SEGMENTS\n");
        }

        // Write media renditions
        for media in &self.media {
            out.push_str(&format_media_info(media));
        }

        // Write variants
        for variant in &self.variants {
            out.push_str(&format_stream_inf(&variant.stream_inf));
            out.push_str(&variant.uri);
            out.push('\n');
        }

        out
    }
}

/// HLS media playlist containing segments.
#[derive(Debug, Clone, Default)]
pub struct MediaPlaylist {
    /// HLS version.
    pub version: u8,
    /// Target segment duration.
    pub target_duration: u64,
    /// Sequence number of first segment.
    pub media_sequence: u64,
    /// Discontinuity sequence.
    pub discontinuity_sequence: u64,
    /// Playlist type (VOD or EVENT).
    pub playlist_type: Option<PlaylistType>,
    /// Segments in the playlist.
    pub segments: Vec<Segment>,
    /// End list flag.
    pub ended: bool,
    /// Current encryption key.
    pub current_key: Option<KeyInfo>,
    /// Current initialization map.
    pub current_map: Option<MapInfo>,
    /// Base URI for resolving relative URLs.
    pub base_uri: Option<String>,
}

impl MediaPlaylist {
    /// Creates a new empty media playlist.
    #[must_use]
    pub fn new() -> Self {
        Self {
            version: 3,
            target_duration: 10,
            ..Default::default()
        }
    }

    /// Parses a media playlist from M3U8 data.
    ///
    /// # Errors
    ///
    /// Returns an error if the playlist is malformed.
    pub fn parse(data: &str) -> NetResult<Self> {
        let mut playlist = Self::new();
        let mut lines = data.lines().peekable();
        let mut current_extinf: Option<(f64, Option<String>)> = None;
        let mut current_byte_range: Option<(u64, Option<u64>)> = None;
        let mut pending_discontinuity = false;
        let mut last_byte_range_end: Option<u64> = None;

        // Check for EXTM3U header
        match lines.next() {
            Some(line) if line.trim() == "#EXTM3U" => {}
            _ => return Err(NetError::playlist("Missing #EXTM3U header")),
        }

        for line in lines {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            if line.starts_with("#EXT-X-VERSION:") {
                if let Some(ver) = line.strip_prefix("#EXT-X-VERSION:") {
                    playlist.version = ver.parse().unwrap_or(3);
                }
            } else if line.starts_with("#EXT-X-TARGETDURATION:") {
                if let Some(dur) = line.strip_prefix("#EXT-X-TARGETDURATION:") {
                    playlist.target_duration = dur.parse().unwrap_or(10);
                }
            } else if line.starts_with("#EXT-X-MEDIA-SEQUENCE:") {
                if let Some(seq) = line.strip_prefix("#EXT-X-MEDIA-SEQUENCE:") {
                    playlist.media_sequence = seq.parse().unwrap_or(0);
                }
            } else if line.starts_with("#EXT-X-DISCONTINUITY-SEQUENCE:") {
                if let Some(seq) = line.strip_prefix("#EXT-X-DISCONTINUITY-SEQUENCE:") {
                    playlist.discontinuity_sequence = seq.parse().unwrap_or(0);
                }
            } else if line.starts_with("#EXT-X-PLAYLIST-TYPE:") {
                if let Some(ptype) = line.strip_prefix("#EXT-X-PLAYLIST-TYPE:") {
                    playlist.playlist_type = ptype.parse().ok();
                }
            } else if line == "#EXT-X-ENDLIST" {
                playlist.ended = true;
            } else if line == "#EXT-X-DISCONTINUITY" {
                pending_discontinuity = true;
            } else if line.starts_with("#EXTINF:") {
                current_extinf = parse_extinf(line)?;
            } else if line.starts_with("#EXT-X-BYTERANGE:") {
                current_byte_range = parse_byte_range(line, last_byte_range_end)?;
            } else if line.starts_with("#EXT-X-KEY:") {
                playlist.current_key = Some(parse_key_info(line)?);
            } else if line.starts_with("#EXT-X-MAP:") {
                playlist.current_map = Some(parse_map_info(line)?);
            } else if !line.starts_with('#') {
                // URI line - create segment
                if let Some((duration, title)) = current_extinf.take() {
                    let mut segment =
                        Segment::new(Duration::from_secs_f64(duration), line.to_string());
                    segment.title = title;
                    segment.discontinuity = pending_discontinuity;
                    segment.key = playlist.current_key.clone();
                    segment.map = playlist.current_map.clone();

                    if let Some((length, offset)) = current_byte_range.take() {
                        segment.byte_range = Some((length, offset));
                        last_byte_range_end = Some(offset.unwrap_or(0) + length);
                    } else {
                        last_byte_range_end = None;
                    }

                    playlist.segments.push(segment);
                    pending_discontinuity = false;
                }
            }
        }

        Ok(playlist)
    }

    /// Returns the total duration of all segments.
    #[must_use]
    pub fn total_duration(&self) -> Duration {
        self.segments.iter().map(|s| s.duration).sum()
    }

    /// Returns the number of segments.
    #[must_use]
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Returns true if this is a VOD playlist.
    #[must_use]
    pub fn is_vod(&self) -> bool {
        self.ended || matches!(self.playlist_type, Some(PlaylistType::Vod))
    }

    /// Returns true if this is a live playlist.
    #[must_use]
    pub fn is_live(&self) -> bool {
        !self.ended && !matches!(self.playlist_type, Some(PlaylistType::Vod))
    }

    /// Writes the playlist to M3U8 format.
    #[must_use]
    pub fn to_m3u8(&self) -> String {
        let mut out = String::new();
        out.push_str("#EXTM3U\n");
        out.push_str(&format!("#EXT-X-VERSION:{}\n", self.version));
        out.push_str(&format!("#EXT-X-TARGETDURATION:{}\n", self.target_duration));
        out.push_str(&format!("#EXT-X-MEDIA-SEQUENCE:{}\n", self.media_sequence));

        if let Some(ref ptype) = self.playlist_type {
            let type_str = match ptype {
                PlaylistType::Vod => "VOD",
                PlaylistType::Event => "EVENT",
            };
            out.push_str(&format!("#EXT-X-PLAYLIST-TYPE:{type_str}\n"));
        }

        let mut last_key: Option<&KeyInfo> = None;
        let mut last_map: Option<&MapInfo> = None;

        for segment in &self.segments {
            // Write key if changed
            if segment.key.as_ref() != last_key {
                if let Some(ref key) = segment.key {
                    out.push_str(&format_key_info(key));
                }
                last_key = segment.key.as_ref();
            }

            // Write map if changed
            if segment.map.as_ref() != last_map {
                if let Some(ref map) = segment.map {
                    out.push_str(&format_map_info(map));
                }
                last_map = segment.map.as_ref();
            }

            if segment.discontinuity {
                out.push_str("#EXT-X-DISCONTINUITY\n");
            }

            let duration = segment.duration.as_secs_f64();
            match &segment.title {
                Some(title) => out.push_str(&format!("#EXTINF:{duration:.3},{title}\n")),
                None => out.push_str(&format!("#EXTINF:{duration:.3},\n")),
            }

            if let Some((length, offset)) = segment.byte_range {
                match offset {
                    Some(off) => out.push_str(&format!("#EXT-X-BYTERANGE:{length}@{off}\n")),
                    None => out.push_str(&format!("#EXT-X-BYTERANGE:{length}\n")),
                }
            }

            out.push_str(&segment.uri);
            out.push('\n');
        }

        if self.ended {
            out.push_str("#EXT-X-ENDLIST\n");
        }

        out
    }
}

// Helper functions for parsing

fn parse_attributes(line: &str) -> HashMap<String, String> {
    let mut attrs = HashMap::new();
    let mut remaining = line;

    while !remaining.is_empty() {
        // Find key
        let Some(eq_pos) = remaining.find('=') else {
            break;
        };
        let key = remaining[..eq_pos].trim().to_string();
        remaining = &remaining[eq_pos + 1..];

        // Parse value (handle quoted strings)
        let value = if remaining.starts_with('"') {
            remaining = &remaining[1..];
            if let Some(quote_end) = remaining.find('"') {
                let val = remaining[..quote_end].to_string();
                remaining = &remaining[quote_end + 1..];
                val
            } else {
                remaining.to_string()
            }
        } else if let Some(comma_pos) = remaining.find(',') {
            let val = remaining[..comma_pos].to_string();
            remaining = &remaining[comma_pos..];
            val
        } else {
            let val = remaining.to_string();
            remaining = "";
            val
        };

        attrs.insert(key, value);

        // Skip comma
        remaining = remaining.trim_start_matches(',').trim_start();
    }

    attrs
}

fn parse_stream_inf(line: &str) -> NetResult<StreamInf> {
    let content = line
        .strip_prefix("#EXT-X-STREAM-INF:")
        .ok_or_else(|| NetError::parse(0, "Invalid STREAM-INF tag"))?;

    let attrs = parse_attributes(content);
    let mut info = StreamInf::default();

    if let Some(bw) = attrs.get("BANDWIDTH") {
        info.bandwidth = bw.parse().unwrap_or(0);
    }
    if let Some(avg) = attrs.get("AVERAGE-BANDWIDTH") {
        info.average_bandwidth = avg.parse().ok();
    }
    if let Some(codecs) = attrs.get("CODECS") {
        info.codecs = Some(codecs.clone());
    }
    if let Some(res) = attrs.get("RESOLUTION") {
        if let Some((w, h)) = res.split_once('x') {
            if let (Ok(width), Ok(height)) = (w.parse(), h.parse()) {
                info.resolution = Some((width, height));
            }
        }
    }
    if let Some(fr) = attrs.get("FRAME-RATE") {
        info.frame_rate = fr.parse().ok();
    }
    if let Some(audio) = attrs.get("AUDIO") {
        info.audio = Some(audio.clone());
    }
    if let Some(video) = attrs.get("VIDEO") {
        info.video = Some(video.clone());
    }
    if let Some(subs) = attrs.get("SUBTITLES") {
        info.subtitles = Some(subs.clone());
    }
    if let Some(cc) = attrs.get("CLOSED-CAPTIONS") {
        info.closed_captions = Some(cc.clone());
    }

    Ok(info)
}

fn parse_media_info(line: &str) -> NetResult<MediaInfo> {
    let content = line
        .strip_prefix("#EXT-X-MEDIA:")
        .ok_or_else(|| NetError::parse(0, "Invalid MEDIA tag"))?;

    let attrs = parse_attributes(content);
    let mut info = MediaInfo::default();

    if let Some(mtype) = attrs.get("TYPE") {
        info.media_type = mtype.parse()?;
    }
    if let Some(uri) = attrs.get("URI") {
        info.uri = Some(uri.clone());
    }
    if let Some(gid) = attrs.get("GROUP-ID") {
        info.group_id = gid.clone();
    }
    if let Some(lang) = attrs.get("LANGUAGE") {
        info.language = Some(lang.clone());
    }
    if let Some(name) = attrs.get("NAME") {
        info.name = name.clone();
    }
    if let Some(def) = attrs.get("DEFAULT") {
        info.default = def == "YES";
    }
    if let Some(auto) = attrs.get("AUTOSELECT") {
        info.autoselect = auto == "YES";
    }
    if let Some(forced) = attrs.get("FORCED") {
        info.forced = forced == "YES";
    }
    if let Some(channels) = attrs.get("CHANNELS") {
        info.channels = Some(channels.clone());
    }

    Ok(info)
}

fn parse_start_tag(line: &str) -> NetResult<Option<(f64, bool)>> {
    let content = line
        .strip_prefix("#EXT-X-START:")
        .ok_or_else(|| NetError::parse(0, "Invalid START tag"))?;

    let attrs = parse_attributes(content);
    if let Some(offset) = attrs.get("TIME-OFFSET") {
        let time: f64 = offset
            .parse()
            .map_err(|_| NetError::parse(0, "Invalid TIME-OFFSET"))?;
        let precise = attrs.get("PRECISE").is_some_and(|v| v == "YES");
        Ok(Some((time, precise)))
    } else {
        Ok(None)
    }
}

fn parse_extinf(line: &str) -> NetResult<Option<(f64, Option<String>)>> {
    let content = line
        .strip_prefix("#EXTINF:")
        .ok_or_else(|| NetError::parse(0, "Invalid EXTINF tag"))?;

    let (duration_str, title) = match content.find(',') {
        Some(pos) => {
            let title = content[pos + 1..].trim();
            (
                &content[..pos],
                if title.is_empty() {
                    None
                } else {
                    Some(title.to_string())
                },
            )
        }
        None => (content, None),
    };

    let duration: f64 = duration_str
        .parse()
        .map_err(|_| NetError::parse(0, "Invalid EXTINF duration"))?;
    Ok(Some((duration, title)))
}

fn parse_byte_range(line: &str, last_end: Option<u64>) -> NetResult<Option<(u64, Option<u64>)>> {
    let content = line
        .strip_prefix("#EXT-X-BYTERANGE:")
        .ok_or_else(|| NetError::parse(0, "Invalid BYTERANGE tag"))?;

    let (length_str, offset) = match content.find('@') {
        Some(pos) => {
            let off: u64 = content[pos + 1..]
                .parse()
                .map_err(|_| NetError::parse(0, "Invalid byte range offset"))?;
            (&content[..pos], Some(off))
        }
        None => (content, last_end),
    };

    let length: u64 = length_str
        .parse()
        .map_err(|_| NetError::parse(0, "Invalid byte range length"))?;
    Ok(Some((length, offset)))
}

fn parse_key_info(line: &str) -> NetResult<KeyInfo> {
    let content = line
        .strip_prefix("#EXT-X-KEY:")
        .ok_or_else(|| NetError::parse(0, "Invalid KEY tag"))?;

    let attrs = parse_attributes(content);
    let mut info = KeyInfo::default();

    if let Some(method) = attrs.get("METHOD") {
        info.method = method.clone();
    }
    if let Some(uri) = attrs.get("URI") {
        info.uri = Some(uri.clone());
    }
    if let Some(iv) = attrs.get("IV") {
        // Parse hex IV (0x prefix)
        let hex = iv.strip_prefix("0x").unwrap_or(iv);
        if let Ok(bytes) = hex::decode(hex) {
            info.iv = Some(bytes);
        } else {
            // Fallback: store as bytes
            info.iv = Some(hex.as_bytes().to_vec());
        }
    }
    if let Some(fmt) = attrs.get("KEYFORMAT") {
        info.keyformat = Some(fmt.clone());
    }
    if let Some(vers) = attrs.get("KEYFORMATVERSIONS") {
        info.keyformatversions = Some(vers.clone());
    }

    Ok(info)
}

fn parse_map_info(line: &str) -> NetResult<MapInfo> {
    let content = line
        .strip_prefix("#EXT-X-MAP:")
        .ok_or_else(|| NetError::parse(0, "Invalid MAP tag"))?;

    let attrs = parse_attributes(content);
    let uri = attrs
        .get("URI")
        .ok_or_else(|| NetError::parse(0, "MAP tag missing URI"))?
        .clone();

    let byterange = if let Some(br) = attrs.get("BYTERANGE") {
        let (length_str, offset) = match br.find('@') {
            Some(pos) => {
                let off: u64 = br[pos + 1..]
                    .parse()
                    .map_err(|_| NetError::parse(0, "Invalid MAP byte range"))?;
                (&br[..pos], Some(off))
            }
            None => (br.as_str(), None),
        };
        let length: u64 = length_str
            .parse()
            .map_err(|_| NetError::parse(0, "Invalid MAP byte range length"))?;
        Some((length, offset))
    } else {
        None
    };

    Ok(MapInfo { uri, byterange })
}

// Helper functions for formatting

fn format_stream_inf(info: &StreamInf) -> String {
    let mut out = format!("#EXT-X-STREAM-INF:BANDWIDTH={}", info.bandwidth);

    if let Some(avg) = info.average_bandwidth {
        out.push_str(&format!(",AVERAGE-BANDWIDTH={avg}"));
    }
    if let Some(ref codecs) = info.codecs {
        out.push_str(&format!(",CODECS=\"{codecs}\""));
    }
    if let Some((w, h)) = info.resolution {
        out.push_str(&format!(",RESOLUTION={w}x{h}"));
    }
    if let Some(fr) = info.frame_rate {
        out.push_str(&format!(",FRAME-RATE={fr:.3}"));
    }
    if let Some(ref audio) = info.audio {
        out.push_str(&format!(",AUDIO=\"{audio}\""));
    }
    if let Some(ref subs) = info.subtitles {
        out.push_str(&format!(",SUBTITLES=\"{subs}\""));
    }

    out.push('\n');
    out
}

fn format_media_info(info: &MediaInfo) -> String {
    let type_str = match info.media_type {
        MediaType::Audio => "AUDIO",
        MediaType::Video => "VIDEO",
        MediaType::Subtitles => "SUBTITLES",
        MediaType::ClosedCaptions => "CLOSED-CAPTIONS",
    };

    let mut out = format!(
        "#EXT-X-MEDIA:TYPE={type_str},GROUP-ID=\"{}\",NAME=\"{}\"",
        info.group_id, info.name
    );

    if let Some(ref uri) = info.uri {
        out.push_str(&format!(",URI=\"{uri}\""));
    }
    if let Some(ref lang) = info.language {
        out.push_str(&format!(",LANGUAGE=\"{lang}\""));
    }
    if info.default {
        out.push_str(",DEFAULT=YES");
    }
    if info.autoselect {
        out.push_str(",AUTOSELECT=YES");
    }

    out.push('\n');
    out
}

fn format_key_info(info: &KeyInfo) -> String {
    let mut out = format!("#EXT-X-KEY:METHOD={}", info.method);

    if let Some(ref uri) = info.uri {
        out.push_str(&format!(",URI=\"{uri}\""));
    }
    if let Some(ref iv) = info.iv {
        let hex_iv: String = iv.iter().map(|b| format!("{b:02X}")).collect();
        out.push_str(&format!(",IV=0x{hex_iv}"));
    }

    out.push('\n');
    out
}

fn format_map_info(info: &MapInfo) -> String {
    let mut out = format!("#EXT-X-MAP:URI=\"{}\"", info.uri);

    if let Some((length, offset)) = info.byterange {
        match offset {
            Some(off) => out.push_str(&format!(",BYTERANGE=\"{length}@{off}\"")),
            None => out.push_str(&format!(",BYTERANGE=\"{length}\"")),
        }
    }

    out.push('\n');
    out
}

// Simple hex decode for IV parsing
mod hex {
    pub fn decode(s: &str) -> Result<Vec<u8>, ()> {
        let mut bytes = Vec::with_capacity(s.len() / 2);
        let mut chars = s.chars();

        while let (Some(h), Some(l)) = (chars.next(), chars.next()) {
            let high = h.to_digit(16).ok_or(())?;
            let low = l.to_digit(16).ok_or(())?;
            bytes.push((high * 16 + low) as u8);
        }

        Ok(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_master_playlist() {
        let data = r#"#EXTM3U
#EXT-X-VERSION:4
#EXT-X-INDEPENDENT-SEGMENTS
#EXT-X-STREAM-INF:BANDWIDTH=1500000,RESOLUTION=1280x720,CODECS="avc1.4d401f,mp4a.40.2"
720p.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=3000000,RESOLUTION=1920x1080
1080p.m3u8
"#;

        let playlist = MasterPlaylist::parse(data).expect("should succeed in test");
        assert_eq!(playlist.version, 4);
        assert!(playlist.independent_segments);
        assert_eq!(playlist.variants.len(), 2);

        let v720 = &playlist.variants[0];
        assert_eq!(v720.stream_inf.bandwidth, 1_500_000);
        assert_eq!(v720.stream_inf.resolution, Some((1280, 720)));
        assert_eq!(v720.uri, "720p.m3u8");

        let v1080 = &playlist.variants[1];
        assert_eq!(v1080.stream_inf.bandwidth, 3_000_000);
        assert_eq!(v1080.stream_inf.resolution, Some((1920, 1080)));
    }

    #[test]
    fn test_parse_media_playlist() {
        let data = r#"#EXTM3U
#EXT-X-VERSION:3
#EXT-X-TARGETDURATION:10
#EXT-X-MEDIA-SEQUENCE:0
#EXT-X-PLAYLIST-TYPE:VOD
#EXTINF:9.009,
segment0.ts
#EXTINF:9.009,
segment1.ts
#EXT-X-DISCONTINUITY
#EXTINF:9.009,
segment2.ts
#EXT-X-ENDLIST
"#;

        let playlist = MediaPlaylist::parse(data).expect("should succeed in test");
        assert_eq!(playlist.version, 3);
        assert_eq!(playlist.target_duration, 10);
        assert_eq!(playlist.media_sequence, 0);
        assert!(playlist.is_vod());
        assert!(playlist.ended);
        assert_eq!(playlist.segments.len(), 3);

        assert!(!playlist.segments[0].discontinuity);
        assert!(!playlist.segments[1].discontinuity);
        assert!(playlist.segments[2].discontinuity);
    }

    #[test]
    fn test_parse_byte_range() {
        let data = r#"#EXTM3U
#EXT-X-VERSION:4
#EXT-X-TARGETDURATION:10
#EXTINF:10.0,
#EXT-X-BYTERANGE:1000@0
video.mp4
#EXTINF:10.0,
#EXT-X-BYTERANGE:1000@1000
video.mp4
"#;

        let playlist = MediaPlaylist::parse(data).expect("should succeed in test");
        assert_eq!(playlist.segments.len(), 2);
        assert_eq!(playlist.segments[0].byte_range, Some((1000, Some(0))));
        assert_eq!(playlist.segments[1].byte_range, Some((1000, Some(1000))));
    }

    #[test]
    fn test_best_variant_for_bandwidth() {
        let data = r#"#EXTM3U
#EXT-X-STREAM-INF:BANDWIDTH=500000
240p.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=1500000
720p.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=3000000
1080p.m3u8
"#;

        let playlist = MasterPlaylist::parse(data).expect("should succeed in test");

        // Should pick 720p for 2Mbps bandwidth
        let best = playlist
            .best_variant_for_bandwidth(2_000_000)
            .expect("should succeed in test");
        assert_eq!(best.stream_inf.bandwidth, 1_500_000);

        // Should pick lowest for very low bandwidth
        let best = playlist
            .best_variant_for_bandwidth(100_000)
            .expect("should succeed in test");
        assert_eq!(best.stream_inf.bandwidth, 500_000);

        // Should pick highest that fits
        let best = playlist
            .best_variant_for_bandwidth(5_000_000)
            .expect("should succeed in test");
        assert_eq!(best.stream_inf.bandwidth, 3_000_000);
    }

    #[test]
    fn test_roundtrip_master_playlist() {
        let mut playlist = MasterPlaylist::new();
        playlist.version = 4;
        playlist.independent_segments = true;
        playlist.variants.push(VariantStream {
            stream_inf: StreamInf {
                bandwidth: 1_500_000,
                resolution: Some((1280, 720)),
                ..Default::default()
            },
            uri: "720p.m3u8".to_string(),
        });

        let m3u8 = playlist.to_m3u8();
        let parsed = MasterPlaylist::parse(&m3u8).expect("should succeed in test");

        assert_eq!(parsed.version, 4);
        assert!(parsed.independent_segments);
        assert_eq!(parsed.variants.len(), 1);
        assert_eq!(parsed.variants[0].stream_inf.bandwidth, 1_500_000);
    }

    #[test]
    fn test_roundtrip_media_playlist() {
        let mut playlist = MediaPlaylist::new();
        playlist.version = 3;
        playlist.target_duration = 10;
        playlist.ended = true;
        playlist.segments.push(Segment::new(
            Duration::from_secs_f64(9.5),
            "seg0.ts".to_string(),
        ));
        playlist
            .segments
            .push(Segment::new(Duration::from_secs_f64(10.0), "seg1.ts").with_discontinuity());

        let m3u8 = playlist.to_m3u8();
        let parsed = MediaPlaylist::parse(&m3u8).expect("should succeed in test");

        assert_eq!(parsed.segments.len(), 2);
        assert!(!parsed.segments[0].discontinuity);
        assert!(parsed.segments[1].discontinuity);
        assert!(parsed.ended);
    }

    #[test]
    fn test_segment_builder() {
        let seg = Segment::new(Duration::from_secs(10), "test.ts")
            .with_byte_range(1000, Some(500))
            .with_discontinuity();

        assert!(seg.has_byte_range());
        assert_eq!(seg.byte_range, Some((1000, Some(500))));
        assert!(seg.discontinuity);
    }

    #[test]
    fn test_total_duration() {
        let mut playlist = MediaPlaylist::new();
        playlist.segments.push(Segment::new(
            Duration::from_secs_f64(10.0),
            "seg0.ts".to_string(),
        ));
        playlist.segments.push(Segment::new(
            Duration::from_secs_f64(9.5),
            "seg1.ts".to_string(),
        ));

        let total = playlist.total_duration();
        assert!((total.as_secs_f64() - 19.5).abs() < 0.001);
    }

    #[test]
    fn test_hls_m3u8_format_compliance() {
        let mut playlist = MediaPlaylist::new();
        playlist.version = 3;
        playlist.target_duration = 10;
        playlist.media_sequence = 0;
        playlist.ended = true;
        playlist.segments.push(Segment::new(
            Duration::from_secs_f64(9.5),
            "seg0.ts".to_string(),
        ));
        playlist.segments.push(Segment::new(
            Duration::from_secs_f64(10.0),
            "seg1.ts".to_string(),
        ));
        playlist.segments.push(Segment::new(
            Duration::from_secs_f64(8.8),
            "seg2.ts".to_string(),
        ));

        let m3u8 = playlist.to_m3u8();
        let lines: Vec<&str> = m3u8.lines().collect();

        // 1. First line is exactly "#EXTM3U"
        assert_eq!(
            lines.first().copied(),
            Some("#EXTM3U"),
            "first line must be #EXTM3U"
        );

        // 2. Contains "#EXT-X-VERSION:N" where N >= 3
        let version_line = lines
            .iter()
            .find(|l| l.starts_with("#EXT-X-VERSION:"))
            .expect("#EXT-X-VERSION tag must be present");
        let version_num: u8 = version_line
            .strip_prefix("#EXT-X-VERSION:")
            .expect("prefix present")
            .parse()
            .expect("version must be numeric");
        assert!(
            version_num >= 3,
            "#EXT-X-VERSION must be >= 3, got {version_num}"
        );

        // 3. Contains "#EXT-X-TARGETDURATION:N"
        assert!(
            lines
                .iter()
                .any(|l| l.starts_with("#EXT-X-TARGETDURATION:")),
            "#EXT-X-TARGETDURATION tag must be present"
        );

        // 4. Contains "#EXT-X-MEDIA-SEQUENCE:0"
        assert!(
            lines.iter().any(|l| *l == "#EXT-X-MEDIA-SEQUENCE:0"),
            "#EXT-X-MEDIA-SEQUENCE:0 must be present"
        );

        // 5. Each segment entry has "#EXTINF:<duration>," followed immediately by a URI line
        let mut i = 0usize;
        let mut extinf_count = 0usize;
        while i < lines.len() {
            if lines[i].starts_with("#EXTINF:") {
                assert!(
                    lines[i].contains(','),
                    "#EXTINF line must contain a comma: {}",
                    lines[i]
                );
                assert!(
                    i + 1 < lines.len(),
                    "#EXTINF at line {i} must be followed by a URI line"
                );
                let uri_line = lines[i + 1];
                assert!(
                    !uri_line.is_empty() && !uri_line.starts_with('#'),
                    "URI line after #EXTINF must be non-empty and not a tag, got: {uri_line}"
                );
                extinf_count += 1;
                i += 2;
            } else {
                i += 1;
            }
        }
        assert_eq!(extinf_count, 3, "expected 3 segments, found {extinf_count}");
    }
}
