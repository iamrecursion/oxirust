//! HLS playlist generation (M3U8).

use crate::config::BitrateEntry;
use crate::encryption::EncryptionHandler;
use crate::error::{PackagerError, PackagerResult};
use crate::manifest::{CodecStringBuilder, DurationFormatter};
use crate::segment::SegmentInfo;
use std::fmt::Write as FmtWrite;
use std::time::Duration;

/// HLS playlist type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaylistType {
    /// Master playlist (multi-variant).
    Master,
    /// Media playlist (single variant).
    Media,
}

/// Media type for HLS.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaType {
    /// Video.
    Video,
    /// Audio.
    Audio,
    /// Subtitles.
    Subtitles,
}

impl MediaType {
    /// Convert to HLS media type string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::Video => "VIDEO",
            Self::Audio => "AUDIO",
            Self::Subtitles => "SUBTITLES",
        }
    }
}

/// HLS variant stream information.
#[derive(Debug, Clone)]
pub struct VariantStream {
    /// Bandwidth in bits per second.
    pub bandwidth: u32,
    /// Average bandwidth.
    pub average_bandwidth: Option<u32>,
    /// Codec string.
    pub codecs: String,
    /// Resolution (width x height).
    pub resolution: Option<(u32, u32)>,
    /// Frame rate.
    pub frame_rate: Option<f64>,
    /// URI to media playlist.
    pub uri: String,
    /// Audio group ID.
    pub audio: Option<String>,
    /// Subtitles group ID.
    pub subtitles: Option<String>,
}

impl VariantStream {
    /// Create a new variant stream.
    #[must_use]
    pub fn new(bandwidth: u32, codecs: String, uri: String) -> Self {
        Self {
            bandwidth,
            average_bandwidth: None,
            codecs,
            resolution: None,
            frame_rate: None,
            uri,
            audio: None,
            subtitles: None,
        }
    }

    /// Set resolution.
    #[must_use]
    pub fn with_resolution(mut self, width: u32, height: u32) -> Self {
        self.resolution = Some((width, height));
        self
    }

    /// Set frame rate.
    #[must_use]
    pub fn with_frame_rate(mut self, fps: f64) -> Self {
        self.frame_rate = Some(fps);
        self
    }

    /// Set audio group.
    #[must_use]
    pub fn with_audio(mut self, audio: String) -> Self {
        self.audio = Some(audio);
        self
    }

    /// Set average bandwidth.
    #[must_use]
    pub fn with_average_bandwidth(mut self, avg: u32) -> Self {
        self.average_bandwidth = Some(avg);
        self
    }
}

/// HLS media group (for alternate audio/subtitles).
#[derive(Debug, Clone)]
pub struct MediaGroup {
    /// Media type.
    pub media_type: MediaType,
    /// Group ID.
    pub group_id: String,
    /// Media name.
    pub name: String,
    /// Language.
    pub language: Option<String>,
    /// Is default.
    pub is_default: bool,
    /// Auto-select.
    pub autoselect: bool,
    /// URI to media playlist.
    pub uri: Option<String>,
}

impl MediaGroup {
    /// Create a new media group.
    #[must_use]
    pub fn new(media_type: MediaType, group_id: String, name: String) -> Self {
        Self {
            media_type,
            group_id,
            name,
            language: None,
            is_default: false,
            autoselect: false,
            uri: None,
        }
    }

    /// Set language.
    #[must_use]
    pub fn with_language(mut self, lang: String) -> Self {
        self.language = Some(lang);
        self
    }

    /// Set as default.
    #[must_use]
    pub fn as_default(mut self) -> Self {
        self.is_default = true;
        self
    }

    /// Set URI.
    #[must_use]
    pub fn with_uri(mut self, uri: String) -> Self {
        self.uri = Some(uri);
        self
    }
}

/// Master playlist builder.
pub struct MasterPlaylistBuilder {
    variants: Vec<VariantStream>,
    media_groups: Vec<MediaGroup>,
    version: u32,
    independent_segments: bool,
}

impl MasterPlaylistBuilder {
    /// Create a new master playlist builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            variants: Vec::new(),
            media_groups: Vec::new(),
            version: 6,
            independent_segments: true,
        }
    }

    /// Add a variant stream.
    pub fn add_variant(&mut self, variant: VariantStream) {
        self.variants.push(variant);
    }

    /// Add a media group.
    pub fn add_media_group(&mut self, group: MediaGroup) {
        self.media_groups.push(group);
    }

    /// Set HLS version.
    #[must_use]
    pub fn with_version(mut self, version: u32) -> Self {
        self.version = version;
        self
    }

    /// Build the master playlist.
    pub fn build(&self) -> PackagerResult<String> {
        let mut output = String::new();

        // Header
        writeln!(output, "#EXTM3U")
            .map_err(|e| PackagerError::manifest_failed(format!("Write error: {e}")))?;
        writeln!(output, "#EXT-X-VERSION:{}", self.version)
            .map_err(|e| PackagerError::manifest_failed(format!("Write error: {e}")))?;

        if self.independent_segments {
            writeln!(output, "#EXT-X-INDEPENDENT-SEGMENTS")
                .map_err(|e| PackagerError::manifest_failed(format!("Write error: {e}")))?;
        }

        // Media groups
        for group in &self.media_groups {
            self.write_media_group(&mut output, group)?;
        }

        // Variant streams (sorted by bandwidth)
        let mut sorted_variants = self.variants.clone();
        sorted_variants.sort_by_key(|v| v.bandwidth);

        for variant in &sorted_variants {
            self.write_variant_stream(&mut output, variant)?;
        }

        Ok(output)
    }

    /// Write media group tag.
    fn write_media_group(&self, output: &mut String, group: &MediaGroup) -> PackagerResult<()> {
        write!(
            output,
            "#EXT-X-MEDIA:TYPE={},GROUP-ID=\"{}\",NAME=\"{}\"",
            group.media_type.as_str(),
            group.group_id,
            group.name
        )
        .map_err(|e| PackagerError::manifest_failed(format!("Write error: {e}")))?;

        if let Some(lang) = &group.language {
            write!(output, ",LANGUAGE=\"{lang}\"")
                .map_err(|e| PackagerError::manifest_failed(format!("Write error: {e}")))?;
        }

        if group.is_default {
            write!(output, ",DEFAULT=YES")
                .map_err(|e| PackagerError::manifest_failed(format!("Write error: {e}")))?;
        }

        if group.autoselect {
            write!(output, ",AUTOSELECT=YES")
                .map_err(|e| PackagerError::manifest_failed(format!("Write error: {e}")))?;
        }

        if let Some(uri) = &group.uri {
            write!(output, ",URI=\"{uri}\"")
                .map_err(|e| PackagerError::manifest_failed(format!("Write error: {e}")))?;
        }

        writeln!(output)
            .map_err(|e| PackagerError::manifest_failed(format!("Write error: {e}")))?;

        Ok(())
    }

    /// Write variant stream tag.
    fn write_variant_stream(
        &self,
        output: &mut String,
        variant: &VariantStream,
    ) -> PackagerResult<()> {
        write!(
            output,
            "#EXT-X-STREAM-INF:BANDWIDTH={},CODECS=\"{}\"",
            variant.bandwidth, variant.codecs
        )
        .map_err(|e| PackagerError::manifest_failed(format!("Write error: {e}")))?;

        if let Some(avg) = variant.average_bandwidth {
            write!(output, ",AVERAGE-BANDWIDTH={avg}")
                .map_err(|e| PackagerError::manifest_failed(format!("Write error: {e}")))?;
        }

        if let Some((width, height)) = variant.resolution {
            write!(output, ",RESOLUTION={width}x{height}")
                .map_err(|e| PackagerError::manifest_failed(format!("Write error: {e}")))?;
        }

        if let Some(fps) = variant.frame_rate {
            write!(output, ",FRAME-RATE={fps:.3}")
                .map_err(|e| PackagerError::manifest_failed(format!("Write error: {e}")))?;
        }

        if let Some(audio) = &variant.audio {
            write!(output, ",AUDIO=\"{audio}\"")
                .map_err(|e| PackagerError::manifest_failed(format!("Write error: {e}")))?;
        }

        if let Some(subtitles) = &variant.subtitles {
            write!(output, ",SUBTITLES=\"{subtitles}\"")
                .map_err(|e| PackagerError::manifest_failed(format!("Write error: {e}")))?;
        }

        writeln!(output)
            .map_err(|e| PackagerError::manifest_failed(format!("Write error: {e}")))?;
        writeln!(output, "{}", variant.uri)
            .map_err(|e| PackagerError::manifest_failed(format!("Write error: {e}")))?;

        Ok(())
    }
}

impl Default for MasterPlaylistBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Media playlist builder.
pub struct MediaPlaylistBuilder {
    segments: Vec<SegmentInfo>,
    version: u32,
    target_duration: Duration,
    media_sequence: u64,
    encryption_handler: Option<EncryptionHandler>,
    playlist_type: Option<String>,
    end_list: bool,
}

impl MediaPlaylistBuilder {
    /// Create a new media playlist builder.
    #[must_use]
    pub fn new(target_duration: Duration) -> Self {
        Self {
            segments: Vec::new(),
            version: 6,
            target_duration,
            media_sequence: 0,
            encryption_handler: None,
            playlist_type: None,
            end_list: false,
        }
    }

    /// Add a segment.
    pub fn add_segment(&mut self, segment: SegmentInfo) {
        self.segments.push(segment);
    }

    /// Set media sequence number.
    #[must_use]
    pub fn with_media_sequence(mut self, seq: u64) -> Self {
        self.media_sequence = seq;
        self
    }

    /// Set encryption handler.
    #[must_use]
    pub fn with_encryption(mut self, handler: EncryptionHandler) -> Self {
        self.encryption_handler = Some(handler);
        self
    }

    /// Set playlist type (VOD or EVENT).
    #[must_use]
    pub fn with_playlist_type(mut self, playlist_type: String) -> Self {
        self.playlist_type = Some(playlist_type);
        self
    }

    /// Mark playlist as ended.
    #[must_use]
    pub fn with_end_list(mut self) -> Self {
        self.end_list = true;
        self
    }

    /// Build the media playlist.
    pub fn build(&self) -> PackagerResult<String> {
        let mut output = String::new();

        // Header
        writeln!(output, "#EXTM3U")
            .map_err(|e| PackagerError::manifest_failed(format!("Write error: {e}")))?;
        writeln!(output, "#EXT-X-VERSION:{}", self.version)
            .map_err(|e| PackagerError::manifest_failed(format!("Write error: {e}")))?;

        // Target duration (rounded up)
        let target_secs = self.target_duration.as_secs() + 1;
        writeln!(output, "#EXT-X-TARGETDURATION:{target_secs}")
            .map_err(|e| PackagerError::manifest_failed(format!("Write error: {e}")))?;

        // Media sequence
        writeln!(output, "#EXT-X-MEDIA-SEQUENCE:{}", self.media_sequence)
            .map_err(|e| PackagerError::manifest_failed(format!("Write error: {e}")))?;

        // Playlist type
        if let Some(ptype) = &self.playlist_type {
            writeln!(output, "#EXT-X-PLAYLIST-TYPE:{ptype}")
                .map_err(|e| PackagerError::manifest_failed(format!("Write error: {e}")))?;
        }

        // Encryption
        if let Some(handler) = &self.encryption_handler {
            if handler.is_enabled() {
                let key_tag = handler.generate_hls_key_tag()?;
                writeln!(output, "{key_tag}")
                    .map_err(|e| PackagerError::manifest_failed(format!("Write error: {e}")))?;
            }
        }

        // Segments
        for segment in &self.segments {
            self.write_segment(&mut output, segment)?;
        }

        // End list marker
        if self.end_list {
            writeln!(output, "#EXT-X-ENDLIST")
                .map_err(|e| PackagerError::manifest_failed(format!("Write error: {e}")))?;
        }

        Ok(output)
    }

    /// Write segment information.
    fn write_segment(&self, output: &mut String, segment: &SegmentInfo) -> PackagerResult<()> {
        let duration_str = DurationFormatter::format_hls_duration(segment.duration);

        writeln!(output, "#EXTINF:{duration_str},")
            .map_err(|e| PackagerError::manifest_failed(format!("Write error: {e}")))?;
        writeln!(output, "{}", segment.path)
            .map_err(|e| PackagerError::manifest_failed(format!("Write error: {e}")))?;

        Ok(())
    }

    /// Calculate target duration from segments.
    #[must_use]
    pub fn calculate_target_duration(segments: &[SegmentInfo]) -> Duration {
        segments
            .iter()
            .map(|s| s.duration)
            .max()
            .unwrap_or(Duration::from_secs(6))
    }
}

/// Generate HLS variant stream from bitrate entry.
pub fn variant_from_bitrate_entry(
    entry: &BitrateEntry,
    uri: String,
) -> PackagerResult<VariantStream> {
    let codec_str = match entry.codec.as_str() {
        "av1" => CodecStringBuilder::av1(0, 4, 8),
        "vp9" => CodecStringBuilder::vp9(0, 40, 8),
        "vp8" => CodecStringBuilder::vp8(),
        _ => {
            return Err(PackagerError::unsupported_codec(format!(
                "Unsupported codec: {}",
                entry.codec
            )))
        }
    };

    let mut variant = VariantStream::new(entry.bitrate, codec_str, uri)
        .with_resolution(entry.width, entry.height);

    if let Some(fps) = entry.framerate {
        variant = variant.with_frame_rate(fps);
    }

    Ok(variant)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_master_playlist_generation() {
        let mut builder = MasterPlaylistBuilder::new();

        let variant = VariantStream::new(
            1_000_000,
            "av01.0.04M.08".to_string(),
            "variant_1.m3u8".to_string(),
        )
        .with_resolution(1280, 720);

        builder.add_variant(variant);

        let playlist = builder.build().expect("should succeed in test");

        assert!(playlist.contains("#EXTM3U"));
        assert!(playlist.contains("#EXT-X-VERSION:6"));
        assert!(playlist.contains("BANDWIDTH=1000000"));
        assert!(playlist.contains("variant_1.m3u8"));
    }

    #[test]
    fn test_media_playlist_generation() {
        let builder = MediaPlaylistBuilder::new(Duration::from_secs(6)).with_end_list();

        let playlist = builder.build().expect("should succeed in test");

        assert!(playlist.contains("#EXTM3U"));
        assert!(playlist.contains("#EXT-X-TARGETDURATION:7"));
        assert!(playlist.contains("#EXT-X-ENDLIST"));
    }

    #[test]
    fn test_variant_from_bitrate_entry() {
        let entry = BitrateEntry::new(1_000_000, 1280, 720, "av1");
        let variant = variant_from_bitrate_entry(&entry, "variant.m3u8".to_string())
            .expect("should succeed in test");

        assert_eq!(variant.bandwidth, 1_000_000);
        assert_eq!(variant.resolution, Some((1280, 720)));
    }
}
