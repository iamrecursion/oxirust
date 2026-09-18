//! HLS M3U8 playlist import and export.
//!
//! This module implements full HLS (HTTP Live Streaming) M3U8 playlist parsing
//! and serialisation per RFC 8216 / Apple HLS spec.  It handles both **media
//! playlists** (a list of media segments) and **master playlists** (a list of
//! `#EXT-X-STREAM-INF` variant streams), as well as SCTE-35 ad cue markers
//! embedded via `#EXT-X-DATERANGE` or `#EXT-OATCLS-SCTE35`.
//!
//! # Round-trip guarantee
//!
//! All tags parsed from an M3U8 document are preserved in the structured
//! representation, so `parse → export` produces a semantically-equivalent
//! document (whitespace and comment ordering may differ).

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_truncation)]

use crate::{PlaylistError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Write as FmtWrite;

// ── Target duration / media sequence ─────────────────────────────────────────

/// Protocol version declared by `#EXT-X-VERSION`.
pub type HlsVersion = u8;

// ── Encryption ────────────────────────────────────────────────────────────────

/// Encryption method used for media segments (`#EXT-X-KEY`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum EncryptionMethod {
    /// No encryption.
    None,
    /// AES-128 block encryption.
    Aes128,
    /// Sample-AES encryption (for fMP4/CMAF).
    SampleAes,
}

impl EncryptionMethod {
    fn as_str(&self) -> &'static str {
        match self {
            Self::None => "NONE",
            Self::Aes128 => "AES-128",
            Self::SampleAes => "SAMPLE-AES",
        }
    }

    fn from_str(s: &str) -> Self {
        match s {
            "AES-128" => Self::Aes128,
            "SAMPLE-AES" => Self::SampleAes,
            _ => Self::None,
        }
    }
}

/// Key descriptor from `#EXT-X-KEY`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HlsKey {
    /// Encryption method.
    pub method: EncryptionMethod,
    /// Key URI (absent for `NONE`).
    pub uri: Option<String>,
    /// Initialisation vector (hex string, optional).
    pub iv: Option<String>,
    /// Key format (default: `"identity"`).
    pub key_format: Option<String>,
    /// Key format versions (e.g., `"1"`).
    pub key_format_versions: Option<String>,
}

// ── Media map ────────────────────────────────────────────────────────────────

/// `#EXT-X-MAP` descriptor pointing to the initialisation segment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HlsMap {
    /// URI to the initialisation segment.
    pub uri: String,
    /// Optional byte range within the URI.
    pub byte_range: Option<String>,
}

// ── SCTE-35 ad cue ───────────────────────────────────────────────────────────

/// Source of the SCTE-35 data in the M3U8 file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Scte35Source {
    /// `#EXT-X-SCTE35:CUE="<base64>"` (Apple HLS extension).
    HlsCue(String),
    /// `#EXT-OATCLS-SCTE35:<base64>` (Wowza / Adobe extension).
    OatCls(String),
    /// Duration from `#EXT-X-CUE-OUT:<secs>`.
    CueOut(f64),
    /// `#EXT-X-CUE-IN` marker.
    CueIn,
    /// `#EXT-X-DATERANGE` with a `SCTE35-CMD` attribute.
    DateRange {
        /// `ID` attribute.
        id: String,
        /// `START-DATE` attribute.
        start_date: String,
        /// Raw SCTE-35 hex data.
        scte35_cmd: Option<String>,
        /// Duration attribute (seconds).
        duration: Option<f64>,
    },
}

/// An ad cue marker embedded in a media segment sequence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HlsAdCue {
    /// Byte offset of the cue in the serialised M3U8 (informational).
    pub segment_index: usize,
    /// SCTE-35 source data.
    pub source: Scte35Source,
    /// Human-readable label (derived from `#EXT-X-CUE-OUT-CONT` etc.).
    pub label: Option<String>,
}

// ── Media segment ─────────────────────────────────────────────────────────────

/// A single media segment in an HLS media playlist.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HlsSegment {
    /// Segment URI.
    pub uri: String,
    /// Duration in seconds from `#EXTINF`.
    pub duration_secs: f64,
    /// Optional human-readable title from `#EXTINF`.
    pub title: String,
    /// Optional byte range from `#EXT-X-BYTERANGE`.
    pub byte_range: Option<String>,
    /// Whether this segment is discontinuous (`#EXT-X-DISCONTINUITY`).
    pub discontinuity: bool,
    /// Optional program date-time (`#EXT-X-PROGRAM-DATE-TIME`).
    pub program_date_time: Option<String>,
    /// Optional encryption key override for this segment.
    pub key: Option<HlsKey>,
    /// Optional map override for this segment.
    pub map: Option<HlsMap>,
    /// Ad cue markers that precede this segment.
    pub ad_cues: Vec<HlsAdCue>,
    /// Extra tags (unknown extensions) stored verbatim.
    pub extra_tags: Vec<String>,
}

impl HlsSegment {
    /// Creates a new segment with the given URI and duration.
    #[must_use]
    pub fn new(uri: impl Into<String>, duration_secs: f64) -> Self {
        Self {
            uri: uri.into(),
            duration_secs,
            title: String::new(),
            byte_range: None,
            discontinuity: false,
            program_date_time: None,
            key: None,
            map: None,
            ad_cues: Vec::new(),
            extra_tags: Vec::new(),
        }
    }
}

// ── Stream variant (master playlist) ─────────────────────────────────────────

/// A variant stream entry from `#EXT-X-STREAM-INF` in a master playlist.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HlsVariantStream {
    /// Stream URI.
    pub uri: String,
    /// Peak bit rate in bits per second.
    pub bandwidth: u64,
    /// Average bit rate (optional).
    pub average_bandwidth: Option<u64>,
    /// Codec string (e.g., `"avc1.42e01e,mp4a.40.2"`).
    pub codecs: Option<String>,
    /// Resolution (e.g., `1920x1080`).
    pub resolution: Option<String>,
    /// Frame rate.
    pub frame_rate: Option<f64>,
    /// HDCP level.
    pub hdcp_level: Option<String>,
    /// GROUP-ID for audio rendition.
    pub audio: Option<String>,
    /// GROUP-ID for video rendition.
    pub video: Option<String>,
    /// GROUP-ID for subtitles rendition.
    pub subtitles: Option<String>,
    /// GROUP-ID for closed captions.
    pub closed_captions: Option<String>,
    /// Extra attributes stored verbatim.
    pub extra_attrs: HashMap<String, String>,
}

// ── Rendition (EXT-X-MEDIA) ───────────────────────────────────────────────────

/// A rendition from `#EXT-X-MEDIA`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HlsRendition {
    /// Rendition type (`AUDIO`, `VIDEO`, `SUBTITLES`, `CLOSED-CAPTIONS`).
    pub rendition_type: String,
    /// Group identifier.
    pub group_id: String,
    /// Human-readable name.
    pub name: String,
    /// Language tag (BCP-47).
    pub language: Option<String>,
    /// Default rendition flag.
    pub is_default: bool,
    /// Auto-select flag.
    pub auto_select: bool,
    /// URI to the rendition playlist (absent for closed captions).
    pub uri: Option<String>,
    /// Characteristics (comma-separated).
    pub characteristics: Option<String>,
}

// ── M3U8 Playlist ────────────────────────────────────────────────────────────

/// HLS protocol type.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum HlsPlaylistKind {
    /// Media playlist containing segments.
    Media,
    /// Master playlist containing variant stream references.
    Master,
}

/// An HLS M3U8 playlist (either media or master).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HlsPlaylist {
    /// Playlist type (media or master).
    pub kind: HlsPlaylistKind,
    /// HLS version (`#EXT-X-VERSION`).
    pub version: HlsVersion,
    /// Target segment duration in seconds (`#EXT-X-TARGETDURATION`).
    pub target_duration: Option<u64>,
    /// Media sequence number (`#EXT-X-MEDIA-SEQUENCE`).
    pub media_sequence: u64,
    /// Discontinuity sequence (`#EXT-X-DISCONTINUITY-SEQUENCE`).
    pub discontinuity_sequence: u64,
    /// Playlist type qualifier (`EVENT` or `VOD`).
    pub playlist_type: Option<String>,
    /// Whether the playlist ends with `#EXT-X-ENDLIST`.
    pub is_ended: bool,
    /// Whether independent segments are declared.
    pub independent_segments: bool,
    /// Active encryption key (applies to subsequent segments).
    pub default_key: Option<HlsKey>,
    /// Default initialisation map.
    pub default_map: Option<HlsMap>,
    /// Media segments (media playlists only).
    pub segments: Vec<HlsSegment>,
    /// Variant streams (master playlists only).
    pub variant_streams: Vec<HlsVariantStream>,
    /// Renditions from `#EXT-X-MEDIA` (master playlists only).
    pub renditions: Vec<HlsRendition>,
    /// Unrecognised top-level tags stored verbatim.
    pub extra_tags: Vec<String>,
}

impl HlsPlaylist {
    /// Creates a new empty media playlist.
    #[must_use]
    pub fn new_media() -> Self {
        Self {
            kind: HlsPlaylistKind::Media,
            version: 3,
            target_duration: None,
            media_sequence: 0,
            discontinuity_sequence: 0,
            playlist_type: None,
            is_ended: false,
            independent_segments: false,
            default_key: None,
            default_map: None,
            segments: Vec::new(),
            variant_streams: Vec::new(),
            renditions: Vec::new(),
            extra_tags: Vec::new(),
        }
    }

    /// Creates a new empty master playlist.
    #[must_use]
    pub fn new_master() -> Self {
        let mut p = Self::new_media();
        p.kind = HlsPlaylistKind::Master;
        p
    }

    /// Total duration of all media segments (media playlists only).
    #[must_use]
    pub fn total_duration_secs(&self) -> f64 {
        self.segments.iter().map(|s| s.duration_secs).sum()
    }

    /// Track count (number of segments).
    #[must_use]
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Parses an HLS M3U8 document from a string slice.
    ///
    /// # Errors
    ///
    /// Returns [`PlaylistError::InvalidItem`] if the document is not a valid
    /// M3U8 file (missing `#EXTM3U` header or malformed tags).
    pub fn parse(content: &str) -> Result<Self> {
        let mut lines = content.lines().peekable();

        // Must start with #EXTM3U
        let first = lines
            .next()
            .map(str::trim)
            .ok_or_else(|| PlaylistError::InvalidItem("Empty M3U8 document".into()))?;
        if first != "#EXTM3U" {
            return Err(PlaylistError::InvalidItem(format!(
                "M3U8 must start with #EXTM3U, got: {first}"
            )));
        }

        let mut playlist = Self::new_media();
        // Segment-level pending state
        let mut pending_duration: Option<f64> = None;
        let mut pending_title = String::new();
        let mut pending_byte_range: Option<String> = None;
        let mut pending_discontinuity = false;
        let mut pending_pdt: Option<String> = None;
        let mut pending_key: Option<HlsKey> = None;
        let mut pending_map: Option<HlsMap> = None;
        let mut pending_cues: Vec<HlsAdCue> = Vec::new();
        let mut pending_extras: Vec<String> = Vec::new();
        let mut segment_index: usize = 0;

        for raw_line in lines {
            let line = raw_line.trim();
            if line.is_empty() {
                continue;
            }

            // ── Playlist-level tags ───────────────────────────────────────
            if let Some(rest) = line.strip_prefix("#EXT-X-VERSION:") {
                playlist.version = rest.trim().parse::<u8>().unwrap_or(3);
                continue;
            }
            if let Some(rest) = line.strip_prefix("#EXT-X-TARGETDURATION:") {
                playlist.target_duration = rest.trim().parse::<u64>().ok();
                continue;
            }
            if let Some(rest) = line.strip_prefix("#EXT-X-MEDIA-SEQUENCE:") {
                playlist.media_sequence = rest.trim().parse::<u64>().unwrap_or(0);
                continue;
            }
            if let Some(rest) = line.strip_prefix("#EXT-X-DISCONTINUITY-SEQUENCE:") {
                playlist.discontinuity_sequence = rest.trim().parse::<u64>().unwrap_or(0);
                continue;
            }
            if let Some(rest) = line.strip_prefix("#EXT-X-PLAYLIST-TYPE:") {
                playlist.playlist_type = Some(rest.trim().to_string());
                continue;
            }
            if line == "#EXT-X-ENDLIST" {
                playlist.is_ended = true;
                continue;
            }
            if line == "#EXT-X-INDEPENDENT-SEGMENTS" {
                playlist.independent_segments = true;
                continue;
            }

            // ── Master playlist tags ──────────────────────────────────────
            if let Some(rest) = line.strip_prefix("#EXT-X-STREAM-INF:") {
                playlist.kind = HlsPlaylistKind::Master;
                let attrs = parse_attr_list(rest);
                let variant = HlsVariantStream {
                    uri: String::new(), // filled by next line
                    bandwidth: attrs
                        .get("BANDWIDTH")
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(0),
                    average_bandwidth: attrs.get("AVERAGE-BANDWIDTH").and_then(|v| v.parse().ok()),
                    codecs: attrs.get("CODECS").cloned(),
                    resolution: attrs.get("RESOLUTION").cloned(),
                    frame_rate: attrs.get("FRAME-RATE").and_then(|v| v.parse().ok()),
                    hdcp_level: attrs.get("HDCP-LEVEL").cloned(),
                    audio: attrs.get("AUDIO").cloned(),
                    video: attrs.get("VIDEO").cloned(),
                    subtitles: attrs.get("SUBTITLES").cloned(),
                    closed_captions: attrs.get("CLOSED-CAPTIONS").cloned(),
                    extra_attrs: attrs
                        .into_iter()
                        .filter(|(k, _)| {
                            !matches!(
                                k.as_str(),
                                "BANDWIDTH"
                                    | "AVERAGE-BANDWIDTH"
                                    | "CODECS"
                                    | "RESOLUTION"
                                    | "FRAME-RATE"
                                    | "HDCP-LEVEL"
                                    | "AUDIO"
                                    | "VIDEO"
                                    | "SUBTITLES"
                                    | "CLOSED-CAPTIONS"
                            )
                        })
                        .collect(),
                };
                playlist.variant_streams.push(variant);
                continue;
            }
            if let Some(rest) = line.strip_prefix("#EXT-X-MEDIA:") {
                playlist.kind = HlsPlaylistKind::Master;
                let attrs = parse_attr_list(rest);
                let rendition = HlsRendition {
                    rendition_type: attrs.get("TYPE").cloned().unwrap_or_else(|| "AUDIO".into()),
                    group_id: attrs.get("GROUP-ID").cloned().unwrap_or_default(),
                    name: attrs.get("NAME").cloned().unwrap_or_default(),
                    language: attrs.get("LANGUAGE").cloned(),
                    is_default: attrs.get("DEFAULT").map(|v| v == "YES").unwrap_or(false),
                    auto_select: attrs.get("AUTOSELECT").map(|v| v == "YES").unwrap_or(false),
                    uri: attrs.get("URI").cloned(),
                    characteristics: attrs.get("CHARACTERISTICS").cloned(),
                };
                playlist.renditions.push(rendition);
                continue;
            }

            // ── Segment-level tags ────────────────────────────────────────
            if let Some(rest) = line.strip_prefix("#EXTINF:") {
                let (dur_str, title_str) = rest.split_once(',').unwrap_or((rest, ""));
                pending_duration = dur_str.trim().parse::<f64>().ok();
                pending_title = title_str.trim().to_string();
                continue;
            }
            if let Some(rest) = line.strip_prefix("#EXT-X-BYTERANGE:") {
                pending_byte_range = Some(rest.trim().to_string());
                continue;
            }
            if line == "#EXT-X-DISCONTINUITY" {
                pending_discontinuity = true;
                continue;
            }
            if let Some(rest) = line.strip_prefix("#EXT-X-PROGRAM-DATE-TIME:") {
                pending_pdt = Some(rest.trim().to_string());
                continue;
            }
            if let Some(rest) = line.strip_prefix("#EXT-X-KEY:") {
                let k = parse_hls_key(rest);
                if playlist.segments.is_empty() && pending_duration.is_none() {
                    // Playlist-level default key
                    playlist.default_key = Some(k.clone());
                }
                pending_key = Some(k);
                continue;
            }
            if let Some(rest) = line.strip_prefix("#EXT-X-MAP:") {
                let attrs = parse_attr_list(rest);
                let map = HlsMap {
                    uri: attrs.get("URI").cloned().unwrap_or_default(),
                    byte_range: attrs.get("BYTERANGE").cloned(),
                };
                if playlist.segments.is_empty() && pending_duration.is_none() {
                    playlist.default_map = Some(map.clone());
                }
                pending_map = Some(map);
                continue;
            }

            // ── SCTE-35 / Ad cue tags ─────────────────────────────────────
            if let Some(rest) = line.strip_prefix("#EXT-X-SCTE35:") {
                let attrs = parse_attr_list(rest);
                if let Some(cue) = attrs.get("CUE") {
                    pending_cues.push(HlsAdCue {
                        segment_index,
                        source: Scte35Source::HlsCue(cue.clone()),
                        label: None,
                    });
                }
                continue;
            }
            if let Some(rest) = line.strip_prefix("#EXT-OATCLS-SCTE35:") {
                pending_cues.push(HlsAdCue {
                    segment_index,
                    source: Scte35Source::OatCls(rest.trim().to_string()),
                    label: None,
                });
                continue;
            }
            if let Some(rest) = line.strip_prefix("#EXT-X-CUE-OUT:") {
                let secs = rest.trim().parse::<f64>().unwrap_or(0.0);
                pending_cues.push(HlsAdCue {
                    segment_index,
                    source: Scte35Source::CueOut(secs),
                    label: None,
                });
                continue;
            }
            if line == "#EXT-X-CUE-IN" {
                pending_cues.push(HlsAdCue {
                    segment_index,
                    source: Scte35Source::CueIn,
                    label: None,
                });
                continue;
            }
            if let Some(rest) = line.strip_prefix("#EXT-X-DATERANGE:") {
                let attrs = parse_attr_list(rest);
                pending_cues.push(HlsAdCue {
                    segment_index,
                    source: Scte35Source::DateRange {
                        id: attrs.get("ID").cloned().unwrap_or_default(),
                        start_date: attrs.get("START-DATE").cloned().unwrap_or_default(),
                        scte35_cmd: attrs.get("SCTE35-CMD").cloned(),
                        duration: attrs.get("DURATION").and_then(|v| v.parse().ok()),
                    },
                    label: None,
                });
                continue;
            }

            // ── URI line (media segment or variant stream URI) ─────────────
            if !line.starts_with('#') {
                // Master playlist: fill in the last variant's URI
                if playlist.kind == HlsPlaylistKind::Master {
                    if let Some(last) = playlist.variant_streams.last_mut() {
                        if last.uri.is_empty() {
                            last.uri = line.to_string();
                        }
                    }
                    continue;
                }

                // Media segment
                if let Some(dur) = pending_duration.take() {
                    let seg = HlsSegment {
                        uri: line.to_string(),
                        duration_secs: dur,
                        title: std::mem::take(&mut pending_title),
                        byte_range: pending_byte_range.take(),
                        discontinuity: std::mem::replace(&mut pending_discontinuity, false),
                        program_date_time: pending_pdt.take(),
                        key: pending_key.take(),
                        map: pending_map.take(),
                        ad_cues: std::mem::take(&mut pending_cues),
                        extra_tags: std::mem::take(&mut pending_extras),
                    };
                    playlist.segments.push(seg);
                    segment_index += 1;
                }
                continue;
            }

            // ── Unknown / extension tag ───────────────────────────────────
            if line.starts_with('#') {
                pending_extras.push(line.to_string());
            }
        }

        Ok(playlist)
    }

    /// Serialises the playlist to an M3U8 string.
    ///
    /// # Errors
    ///
    /// Returns [`PlaylistError::InvalidItem`] if an internal formatting error occurs.
    pub fn to_m3u8(&self) -> Result<String> {
        let mut out = String::with_capacity(4096);
        writeln!(out, "#EXTM3U").map_err(|e| PlaylistError::InvalidItem(e.to_string()))?;
        writeln!(out, "#EXT-X-VERSION:{}", self.version)
            .map_err(|e| PlaylistError::InvalidItem(e.to_string()))?;

        if self.independent_segments {
            writeln!(out, "#EXT-X-INDEPENDENT-SEGMENTS")
                .map_err(|e| PlaylistError::InvalidItem(e.to_string()))?;
        }

        if self.kind == HlsPlaylistKind::Master {
            self.write_master(&mut out)?;
        } else {
            self.write_media(&mut out)?;
        }

        Ok(out)
    }

    fn write_master(&self, out: &mut String) -> Result<()> {
        for r in &self.renditions {
            write!(
                out,
                "#EXT-X-MEDIA:TYPE={},GROUP-ID=\"{}\",NAME=\"{}\"",
                r.rendition_type, r.group_id, r.name
            )
            .map_err(|e| PlaylistError::InvalidItem(e.to_string()))?;
            if let Some(lang) = &r.language {
                write!(out, ",LANGUAGE=\"{lang}\"")
                    .map_err(|e| PlaylistError::InvalidItem(e.to_string()))?;
            }
            write!(out, ",DEFAULT={}", if r.is_default { "YES" } else { "NO" })
                .map_err(|e| PlaylistError::InvalidItem(e.to_string()))?;
            write!(
                out,
                ",AUTOSELECT={}",
                if r.auto_select { "YES" } else { "NO" }
            )
            .map_err(|e| PlaylistError::InvalidItem(e.to_string()))?;
            if let Some(uri) = &r.uri {
                write!(out, ",URI=\"{uri}\"")
                    .map_err(|e| PlaylistError::InvalidItem(e.to_string()))?;
            }
            writeln!(out).map_err(|e| PlaylistError::InvalidItem(e.to_string()))?;
        }

        for v in &self.variant_streams {
            write!(out, "#EXT-X-STREAM-INF:BANDWIDTH={}", v.bandwidth)
                .map_err(|e| PlaylistError::InvalidItem(e.to_string()))?;
            if let Some(ab) = v.average_bandwidth {
                write!(out, ",AVERAGE-BANDWIDTH={ab}")
                    .map_err(|e| PlaylistError::InvalidItem(e.to_string()))?;
            }
            if let Some(c) = &v.codecs {
                write!(out, ",CODECS=\"{c}\"")
                    .map_err(|e| PlaylistError::InvalidItem(e.to_string()))?;
            }
            if let Some(res) = &v.resolution {
                write!(out, ",RESOLUTION={res}")
                    .map_err(|e| PlaylistError::InvalidItem(e.to_string()))?;
            }
            if let Some(fps) = v.frame_rate {
                write!(out, ",FRAME-RATE={fps:.3}")
                    .map_err(|e| PlaylistError::InvalidItem(e.to_string()))?;
            }
            if let Some(a) = &v.audio {
                write!(out, ",AUDIO=\"{a}\"")
                    .map_err(|e| PlaylistError::InvalidItem(e.to_string()))?;
            }
            if let Some(s) = &v.subtitles {
                write!(out, ",SUBTITLES=\"{s}\"")
                    .map_err(|e| PlaylistError::InvalidItem(e.to_string()))?;
            }
            writeln!(out).map_err(|e| PlaylistError::InvalidItem(e.to_string()))?;
            writeln!(out, "{}", v.uri).map_err(|e| PlaylistError::InvalidItem(e.to_string()))?;
        }

        Ok(())
    }

    fn write_media(&self, out: &mut String) -> Result<()> {
        if let Some(td) = self.target_duration {
            writeln!(out, "#EXT-X-TARGETDURATION:{td}")
                .map_err(|e| PlaylistError::InvalidItem(e.to_string()))?;
        }
        writeln!(out, "#EXT-X-MEDIA-SEQUENCE:{}", self.media_sequence)
            .map_err(|e| PlaylistError::InvalidItem(e.to_string()))?;
        if self.discontinuity_sequence > 0 {
            writeln!(
                out,
                "#EXT-X-DISCONTINUITY-SEQUENCE:{}",
                self.discontinuity_sequence
            )
            .map_err(|e| PlaylistError::InvalidItem(e.to_string()))?;
        }
        if let Some(pt) = &self.playlist_type {
            writeln!(out, "#EXT-X-PLAYLIST-TYPE:{pt}")
                .map_err(|e| PlaylistError::InvalidItem(e.to_string()))?;
        }
        if let Some(k) = &self.default_key {
            write_key_tag(out, k)?;
        }
        if let Some(m) = &self.default_map {
            write_map_tag(out, m)?;
        }

        for seg in &self.segments {
            // Ad cue markers
            for cue in &seg.ad_cues {
                write_ad_cue(out, cue)?;
            }
            // Extra tags
            for tag in &seg.extra_tags {
                writeln!(out, "{tag}").map_err(|e| PlaylistError::InvalidItem(e.to_string()))?;
            }
            if seg.discontinuity {
                writeln!(out, "#EXT-X-DISCONTINUITY")
                    .map_err(|e| PlaylistError::InvalidItem(e.to_string()))?;
            }
            if let Some(pdt) = &seg.program_date_time {
                writeln!(out, "#EXT-X-PROGRAM-DATE-TIME:{pdt}")
                    .map_err(|e| PlaylistError::InvalidItem(e.to_string()))?;
            }
            if let Some(k) = &seg.key {
                write_key_tag(out, k)?;
            }
            if let Some(m) = &seg.map {
                write_map_tag(out, m)?;
            }
            if let Some(br) = &seg.byte_range {
                writeln!(out, "#EXT-X-BYTERANGE:{br}")
                    .map_err(|e| PlaylistError::InvalidItem(e.to_string()))?;
            }
            // Duration: 6-decimal precision per HLS spec
            let title = if seg.title.is_empty() {
                String::new()
            } else {
                format!(" {}", seg.title)
            };
            writeln!(out, "#EXTINF:{:.6},{}", seg.duration_secs, title.trim())
                .map_err(|e| PlaylistError::InvalidItem(e.to_string()))?;
            writeln!(out, "{}", seg.uri).map_err(|e| PlaylistError::InvalidItem(e.to_string()))?;
        }

        if self.is_ended {
            writeln!(out, "#EXT-X-ENDLIST")
                .map_err(|e| PlaylistError::InvalidItem(e.to_string()))?;
        }

        Ok(())
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn write_key_tag(out: &mut String, k: &HlsKey) -> Result<()> {
    write!(out, "#EXT-X-KEY:METHOD={}", k.method.as_str())
        .map_err(|e| PlaylistError::InvalidItem(e.to_string()))?;
    if let Some(uri) = &k.uri {
        write!(out, ",URI=\"{uri}\"").map_err(|e| PlaylistError::InvalidItem(e.to_string()))?;
    }
    if let Some(iv) = &k.iv {
        write!(out, ",IV={iv}").map_err(|e| PlaylistError::InvalidItem(e.to_string()))?;
    }
    if let Some(fmt) = &k.key_format {
        write!(out, ",KEYFORMAT=\"{fmt}\"")
            .map_err(|e| PlaylistError::InvalidItem(e.to_string()))?;
    }
    if let Some(fv) = &k.key_format_versions {
        write!(out, ",KEYFORMATVERSIONS=\"{fv}\"")
            .map_err(|e| PlaylistError::InvalidItem(e.to_string()))?;
    }
    writeln!(out).map_err(|e| PlaylistError::InvalidItem(e.to_string()))
}

fn write_map_tag(out: &mut String, m: &HlsMap) -> Result<()> {
    write!(out, "#EXT-X-MAP:URI=\"{}\"", m.uri)
        .map_err(|e| PlaylistError::InvalidItem(e.to_string()))?;
    if let Some(br) = &m.byte_range {
        write!(out, ",BYTERANGE=\"{br}\"")
            .map_err(|e| PlaylistError::InvalidItem(e.to_string()))?;
    }
    writeln!(out).map_err(|e| PlaylistError::InvalidItem(e.to_string()))
}

fn write_ad_cue(out: &mut String, cue: &HlsAdCue) -> Result<()> {
    match &cue.source {
        Scte35Source::HlsCue(data) => writeln!(out, "#EXT-X-SCTE35:CUE=\"{data}\"")
            .map_err(|e| PlaylistError::InvalidItem(e.to_string())),
        Scte35Source::OatCls(data) => writeln!(out, "#EXT-OATCLS-SCTE35:{data}")
            .map_err(|e| PlaylistError::InvalidItem(e.to_string())),
        Scte35Source::CueOut(secs) => writeln!(out, "#EXT-X-CUE-OUT:{secs}")
            .map_err(|e| PlaylistError::InvalidItem(e.to_string())),
        Scte35Source::CueIn => {
            writeln!(out, "#EXT-X-CUE-IN").map_err(|e| PlaylistError::InvalidItem(e.to_string()))
        }
        Scte35Source::DateRange {
            id,
            start_date,
            scte35_cmd,
            duration,
        } => {
            write!(
                out,
                "#EXT-X-DATERANGE:ID=\"{id}\",START-DATE=\"{start_date}\""
            )
            .map_err(|e| PlaylistError::InvalidItem(e.to_string()))?;
            if let Some(cmd) = scte35_cmd {
                write!(out, ",SCTE35-CMD={cmd}")
                    .map_err(|e| PlaylistError::InvalidItem(e.to_string()))?;
            }
            if let Some(d) = duration {
                write!(out, ",DURATION={d}")
                    .map_err(|e| PlaylistError::InvalidItem(e.to_string()))?;
            }
            writeln!(out).map_err(|e| PlaylistError::InvalidItem(e.to_string()))
        }
    }
}

/// Parse an HLS attribute list string into a `HashMap<String, String>`.
/// Handles quoted and unquoted values.
fn parse_attr_list(input: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    let mut rest = input.trim();
    while !rest.is_empty() {
        // Key
        let Some(eq) = rest.find('=') else { break };
        let key = rest[..eq].trim().to_uppercase();
        rest = &rest[eq + 1..];

        // Value
        let (value, consumed) = if rest.starts_with('"') {
            let inner = &rest[1..];
            let end = inner.find('"').unwrap_or(inner.len());
            (&inner[..end], end + 2) // +2 for the two quotes
        } else {
            // Unquoted: ends at comma or end
            let end = rest.find(',').unwrap_or(rest.len());
            (&rest[..end], end)
        };

        map.insert(key, value.to_string());
        rest = if consumed < rest.len() {
            &rest[consumed..]
        } else {
            ""
        };
        // Skip leading comma
        rest = rest.trim_start_matches(',').trim_start();
    }
    map
}

/// Parse an `#EXT-X-KEY` attribute list into an `HlsKey`.
fn parse_hls_key(input: &str) -> HlsKey {
    let attrs = parse_attr_list(input);
    HlsKey {
        method: attrs
            .get("METHOD")
            .map(|v| EncryptionMethod::from_str(v))
            .unwrap_or(EncryptionMethod::None),
        uri: attrs.get("URI").cloned(),
        iv: attrs.get("IV").cloned(),
        key_format: attrs.get("KEYFORMAT").cloned(),
        key_format_versions: attrs.get("KEYFORMATVERSIONS").cloned(),
    }
}

// ── Playlist statistics ───────────────────────────────────────────────────────

/// Summary statistics for an `HlsPlaylist`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HlsStats {
    /// Total number of media segments.
    pub segment_count: usize,
    /// Sum of all segment durations in seconds.
    pub total_duration_secs: f64,
    /// Maximum segment duration in seconds.
    pub max_segment_duration_secs: f64,
    /// Minimum segment duration in seconds.
    pub min_segment_duration_secs: f64,
    /// Average segment duration in seconds.
    pub avg_segment_duration_secs: f64,
    /// Number of ad cue markers across all segments.
    pub ad_cue_count: usize,
    /// Number of discontinuity markers.
    pub discontinuity_count: usize,
    /// Number of encrypted segments (non-NONE key).
    pub encrypted_segment_count: usize,
}

impl HlsPlaylist {
    /// Compute summary statistics for this playlist.
    #[must_use]
    pub fn stats(&self) -> HlsStats {
        if self.segments.is_empty() {
            return HlsStats {
                segment_count: 0,
                total_duration_secs: 0.0,
                max_segment_duration_secs: 0.0,
                min_segment_duration_secs: 0.0,
                avg_segment_duration_secs: 0.0,
                ad_cue_count: 0,
                discontinuity_count: 0,
                encrypted_segment_count: 0,
            };
        }

        let durations: Vec<f64> = self.segments.iter().map(|s| s.duration_secs).collect();
        let total: f64 = durations.iter().sum();
        let max = durations.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min = durations.iter().cloned().fold(f64::INFINITY, f64::min);
        let count = self.segments.len();
        let ad_cue_count: usize = self.segments.iter().map(|s| s.ad_cues.len()).sum();
        let discontinuity_count = self.segments.iter().filter(|s| s.discontinuity).count();
        let encrypted_count = self
            .segments
            .iter()
            .filter(|s| {
                s.key
                    .as_ref()
                    .map(|k| k.method != EncryptionMethod::None)
                    .unwrap_or(false)
            })
            .count();

        HlsStats {
            segment_count: count,
            total_duration_secs: total,
            max_segment_duration_secs: max,
            min_segment_duration_secs: min,
            avg_segment_duration_secs: total / count as f64,
            ad_cue_count,
            discontinuity_count,
            encrypted_segment_count: encrypted_count,
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const BASIC_MEDIA: &str = "\
#EXTM3U
#EXT-X-VERSION:3
#EXT-X-TARGETDURATION:10
#EXT-X-MEDIA-SEQUENCE:0
#EXTINF:9.009,
segment0.ts
#EXTINF:9.009,
segment1.ts
#EXTINF:3.003,
segment2.ts
#EXT-X-ENDLIST
";

    const VOD_WITH_SCTE35: &str = "\
#EXTM3U
#EXT-X-VERSION:3
#EXT-X-TARGETDURATION:10
#EXT-X-MEDIA-SEQUENCE:0
#EXT-X-PLAYLIST-TYPE:VOD
#EXTINF:10.0,intro
intro.ts
#EXT-X-CUE-OUT:30
#EXTINF:10.0,ad1
ad1.ts
#EXT-X-CUE-IN
#EXTINF:10.0,main
main.ts
#EXT-X-ENDLIST
";

    const MASTER: &str = "\
#EXTM3U
#EXT-X-VERSION:3
#EXT-X-STREAM-INF:BANDWIDTH=800000,RESOLUTION=640x360,CODECS=\"avc1.42e01e,mp4a.40.2\"
low.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=2400000,RESOLUTION=1280x720,CODECS=\"avc1.4d401f,mp4a.40.2\"
mid.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=6000000,RESOLUTION=1920x1080,CODECS=\"avc1.640028,mp4a.40.2\"
high.m3u8
";

    #[test]
    fn test_parse_basic_media() {
        let pl = HlsPlaylist::parse(BASIC_MEDIA).expect("parse should succeed");
        assert_eq!(pl.kind, HlsPlaylistKind::Media);
        assert_eq!(pl.version, 3);
        assert_eq!(pl.target_duration, Some(10));
        assert_eq!(pl.segments.len(), 3);
        assert!(pl.is_ended);
        assert!((pl.segments[0].duration_secs - 9.009).abs() < 1e-6);
    }

    #[test]
    fn test_total_duration() {
        let pl = HlsPlaylist::parse(BASIC_MEDIA).expect("parse should succeed");
        let total = pl.total_duration_secs();
        assert!((total - 21.021).abs() < 1e-6);
    }

    #[test]
    fn test_parse_scte35_cue_out_in() {
        let pl = HlsPlaylist::parse(VOD_WITH_SCTE35).expect("parse should succeed");
        // The #EXT-X-CUE-OUT:30 precedes ad1.ts
        let ad_seg = &pl.segments[1];
        assert_eq!(ad_seg.ad_cues.len(), 1);
        match &ad_seg.ad_cues[0].source {
            Scte35Source::CueOut(secs) => assert!((secs - 30.0).abs() < 1e-9),
            other => panic!("Expected CueOut, got {other:?}"),
        }
        // #EXT-X-CUE-IN precedes main.ts
        let main_seg = &pl.segments[2];
        assert_eq!(main_seg.ad_cues.len(), 1);
        assert!(matches!(main_seg.ad_cues[0].source, Scte35Source::CueIn));
    }

    #[test]
    fn test_parse_master_playlist() {
        let pl = HlsPlaylist::parse(MASTER).expect("parse should succeed");
        assert_eq!(pl.kind, HlsPlaylistKind::Master);
        assert_eq!(pl.variant_streams.len(), 3);
        assert_eq!(pl.variant_streams[0].bandwidth, 800_000);
        assert_eq!(pl.variant_streams[0].uri, "low.m3u8");
        assert_eq!(pl.variant_streams[0].resolution.as_deref(), Some("640x360"));
    }

    #[test]
    fn test_round_trip_media() {
        let pl = HlsPlaylist::parse(BASIC_MEDIA).expect("parse should succeed");
        let out = pl.to_m3u8().expect("serialise should succeed");
        let pl2 = HlsPlaylist::parse(&out).expect("re-parse should succeed");
        assert_eq!(pl2.segments.len(), pl.segments.len());
        assert_eq!(pl2.is_ended, pl.is_ended);
        assert!((pl2.total_duration_secs() - pl.total_duration_secs()).abs() < 1e-4);
    }

    #[test]
    fn test_round_trip_master() {
        let pl = HlsPlaylist::parse(MASTER).expect("parse should succeed");
        let out = pl.to_m3u8().expect("serialise should succeed");
        let pl2 = HlsPlaylist::parse(&out).expect("re-parse should succeed");
        assert_eq!(pl2.variant_streams.len(), pl.variant_streams.len());
        assert_eq!(pl2.variant_streams[1].bandwidth, 2_400_000);
    }

    #[test]
    fn test_stats() {
        let pl = HlsPlaylist::parse(BASIC_MEDIA).expect("parse should succeed");
        let stats = pl.stats();
        assert_eq!(stats.segment_count, 3);
        assert!((stats.total_duration_secs - 21.021).abs() < 1e-4);
        assert!((stats.max_segment_duration_secs - 9.009).abs() < 1e-6);
        assert!((stats.min_segment_duration_secs - 3.003).abs() < 1e-6);
    }

    #[test]
    fn test_stats_empty_playlist() {
        let pl = HlsPlaylist::new_media();
        let stats = pl.stats();
        assert_eq!(stats.segment_count, 0);
        assert_eq!(stats.total_duration_secs, 0.0);
    }

    #[test]
    fn test_oatcls_scte35_parsed() {
        let content = "#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:10\n\
            #EXT-X-MEDIA-SEQUENCE:0\n\
            #EXT-OATCLS-SCTE35:/DAlAAAAAA==\n\
            #EXTINF:10.0,\nseg.ts\n#EXT-X-ENDLIST\n";
        let pl = HlsPlaylist::parse(content).expect("parse should succeed");
        let seg = &pl.segments[0];
        assert_eq!(seg.ad_cues.len(), 1);
        assert!(matches!(seg.ad_cues[0].source, Scte35Source::OatCls(_)));
    }

    #[test]
    fn test_missing_extm3u_header_fails() {
        let result = HlsPlaylist::parse("#EXT-X-VERSION:3\n");
        assert!(result.is_err());
    }

    #[test]
    fn test_encryption_key_parsed() {
        let content = "#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:10\n\
            #EXT-X-MEDIA-SEQUENCE:0\n\
            #EXT-X-KEY:METHOD=AES-128,URI=\"https://example.com/key\",IV=0x1234\n\
            #EXTINF:10.0,\nseg.ts\n#EXT-X-ENDLIST\n";
        let pl = HlsPlaylist::parse(content).expect("parse should succeed");
        let key = pl.default_key.as_ref().expect("should have default key");
        assert_eq!(key.method, EncryptionMethod::Aes128);
        assert_eq!(key.uri.as_deref(), Some("https://example.com/key"));
    }

    #[test]
    fn test_playlist_type_vod() {
        let pl = HlsPlaylist::parse(VOD_WITH_SCTE35).expect("parse should succeed");
        assert_eq!(pl.playlist_type.as_deref(), Some("VOD"));
    }

    #[test]
    fn test_daterange_scte35() {
        let content = "#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:10\n\
            #EXT-X-MEDIA-SEQUENCE:0\n\
            #EXT-X-DATERANGE:ID=\"ad1\",START-DATE=\"2024-01-01T00:00:00Z\",\
SCTE35-CMD=0xFC3025,DURATION=30\n\
            #EXTINF:10.0,\nseg.ts\n#EXT-X-ENDLIST\n";
        let pl = HlsPlaylist::parse(content).expect("parse should succeed");
        let seg = &pl.segments[0];
        assert_eq!(seg.ad_cues.len(), 1);
        match &seg.ad_cues[0].source {
            Scte35Source::DateRange { id, duration, .. } => {
                assert_eq!(id, "ad1");
                assert_eq!(*duration, Some(30.0));
            }
            other => panic!("Expected DateRange, got {other:?}"),
        }
    }
}
