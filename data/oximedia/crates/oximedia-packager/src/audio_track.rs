// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Multi-audio track support for adaptive streaming packaging.
//!
//! This module provides `AudioTrackDescriptor` with language/accessibility
//! labels, and generates multi-audio variant entries for HLS and DASH
//! manifests.
//!
//! # HLS Multi-Audio
//!
//! HLS uses `EXT-X-MEDIA` tags with `TYPE=AUDIO` in the multivariant playlist
//! to declare alternate audio renditions. Each `EXT-X-STREAM-INF` then
//! references an `AUDIO` group containing the available languages.
//!
//! # DASH Multi-Audio
//!
//! DASH uses separate `<AdaptationSet>` elements for each audio track,
//! distinguished by `@lang` and `<Accessibility>` descriptors.

use crate::error::{PackagerError, PackagerResult};
use std::time::Duration;

// ---------------------------------------------------------------------------
// AudioRole
// ---------------------------------------------------------------------------

/// The role of an audio track, per DASH Role scheme and HLS CHARACTERISTICS.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AudioRole {
    /// Main dialogue track.
    Main,
    /// Alternate language dubbing.
    Dub,
    /// Audio description for visually impaired (AD).
    Description,
    /// Commentary track.
    Commentary,
    /// Emergency information.
    Emergency,
    /// Supplementary audio (e.g. director's commentary).
    Supplementary,
}

impl AudioRole {
    /// DASH `<Role>` scheme value.
    #[must_use]
    pub fn dash_role_value(self) -> &'static str {
        match self {
            Self::Main => "main",
            Self::Dub => "dub",
            Self::Description => "description",
            Self::Commentary => "commentary",
            Self::Emergency => "emergency",
            Self::Supplementary => "supplementary",
        }
    }

    /// HLS `CHARACTERISTICS` attribute value (from RFC 8216).
    #[must_use]
    pub fn hls_characteristics(self) -> Option<&'static str> {
        match self {
            Self::Description => Some("public.accessibility.describes-video"),
            Self::Commentary => Some("public.accessibility.transcribes-spoken-dialog"),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// AudioCodecInfo
// ---------------------------------------------------------------------------

/// Audio codec information for manifest generation.
#[derive(Debug, Clone)]
pub struct AudioCodecInfo {
    /// Codec identifier string for manifests (e.g. `"opus"`, `"flac"`, `"vorbis"`).
    pub codecs_string: String,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Number of audio channels.
    pub channels: u32,
}

impl AudioCodecInfo {
    /// Create a new audio codec info.
    #[must_use]
    pub fn new(codecs_string: impl Into<String>, sample_rate: u32, channels: u32) -> Self {
        Self {
            codecs_string: codecs_string.into(),
            sample_rate,
            channels,
        }
    }

    /// Create Opus codec info.
    #[must_use]
    pub fn opus(sample_rate: u32, channels: u32) -> Self {
        Self::new("opus", sample_rate, channels)
    }

    /// Create FLAC codec info.
    #[must_use]
    pub fn flac(sample_rate: u32, channels: u32) -> Self {
        Self::new("fLaC", sample_rate, channels)
    }

    /// Create Vorbis codec info.
    #[must_use]
    pub fn vorbis(sample_rate: u32, channels: u32) -> Self {
        Self::new("vorbis", sample_rate, channels)
    }
}

// ---------------------------------------------------------------------------
// AudioTrackDescriptor
// ---------------------------------------------------------------------------

/// Describes a single audio track for multi-audio packaging.
///
/// Each descriptor maps to:
/// - One HLS `EXT-X-MEDIA` tag (TYPE=AUDIO)
/// - One DASH `<AdaptationSet>` with `@lang` and `<Role>`
#[derive(Debug, Clone)]
pub struct AudioTrackDescriptor {
    /// Unique identifier (used in GROUP-ID and @id).
    pub id: String,
    /// BCP 47 language tag (e.g. `"en"`, `"ja"`, `"fr-CA"`).
    pub language: String,
    /// Human-readable label (e.g. `"English"`, `"Japanese"`).
    pub label: String,
    /// Codec information.
    pub codec: AudioCodecInfo,
    /// Bitrate in bits per second.
    pub bitrate: u64,
    /// Role of this audio track.
    pub role: AudioRole,
    /// Whether this is the default audio selection.
    pub is_default: bool,
    /// Whether this track should auto-select.
    pub autoselect: bool,
    /// URI of the media playlist for this audio track.
    pub playlist_uri: String,
    /// Init segment URI (for fMP4/CMAF).
    pub init_uri: Option<String>,
    /// Segment duration.
    pub segment_duration: Duration,
    /// Accessibility labels (for hearing impaired, etc.).
    pub accessibility_labels: Vec<String>,
}

impl AudioTrackDescriptor {
    /// Create a new audio track descriptor.
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        language: impl Into<String>,
        label: impl Into<String>,
        codec: AudioCodecInfo,
        bitrate: u64,
    ) -> Self {
        Self {
            id: id.into(),
            language: language.into(),
            label: label.into(),
            codec,
            bitrate,
            role: AudioRole::Main,
            is_default: false,
            autoselect: true,
            playlist_uri: String::new(),
            init_uri: None,
            segment_duration: Duration::from_secs(6),
            accessibility_labels: Vec::new(),
        }
    }

    /// Set the track role.
    #[must_use]
    pub fn with_role(mut self, role: AudioRole) -> Self {
        self.role = role;
        self
    }

    /// Mark as default selection.
    #[must_use]
    pub fn as_default(mut self) -> Self {
        self.is_default = true;
        self
    }

    /// Set the playlist URI.
    #[must_use]
    pub fn with_playlist_uri(mut self, uri: impl Into<String>) -> Self {
        self.playlist_uri = uri.into();
        self
    }

    /// Set the init segment URI.
    #[must_use]
    pub fn with_init_uri(mut self, uri: impl Into<String>) -> Self {
        self.init_uri = Some(uri.into());
        self
    }

    /// Add an accessibility label.
    #[must_use]
    pub fn with_accessibility(mut self, label: impl Into<String>) -> Self {
        self.accessibility_labels.push(label.into());
        self
    }

    /// Render as an HLS `EXT-X-MEDIA` tag for the multivariant playlist.
    #[must_use]
    pub fn to_hls_media_tag(&self, group_id: &str) -> String {
        let mut attrs = Vec::new();
        attrs.push("TYPE=AUDIO".to_string());
        attrs.push(format!("GROUP-ID=\"{group_id}\""));
        attrs.push(format!("NAME=\"{}\"", self.label));
        attrs.push(format!("LANGUAGE=\"{}\"", self.language));

        if self.is_default {
            attrs.push("DEFAULT=YES".to_string());
        } else {
            attrs.push("DEFAULT=NO".to_string());
        }

        if self.autoselect {
            attrs.push("AUTOSELECT=YES".to_string());
        }

        if let Some(chars) = self.role.hls_characteristics() {
            attrs.push(format!("CHARACTERISTICS=\"{chars}\""));
        } else if !self.accessibility_labels.is_empty() {
            attrs.push(format!(
                "CHARACTERISTICS=\"{}\"",
                self.accessibility_labels.join(",")
            ));
        }

        attrs.push(format!("CHANNELS=\"{}\"", self.codec.channels));

        if !self.playlist_uri.is_empty() {
            attrs.push(format!("URI=\"{}\"", self.playlist_uri));
        }

        format!("#EXT-X-MEDIA:{}", attrs.join(","))
    }

    /// Render as a DASH `<AdaptationSet>` XML fragment.
    #[must_use]
    pub fn to_dash_adaptation_set(&self) -> String {
        let mut xml = String::new();
        xml.push_str(&format!(
            r#"<AdaptationSet id="{}" contentType="audio" lang="{}" mimeType="audio/mp4">"#,
            self.id, self.language
        ));
        xml.push_str(&format!(
            r#"<Role schemeIdUri="urn:mpeg:dash:role:2011" value="{}"/>"#,
            self.role.dash_role_value()
        ));

        for label in &self.accessibility_labels {
            xml.push_str(&format!(
                r#"<Accessibility schemeIdUri="urn:mpeg:dash:role:2011" value="{label}"/>"#
            ));
        }

        xml.push_str(&format!(
            r#"<Representation id="{}_rep" bandwidth="{}" codecs="{}" audioSamplingRate="{}"><AudioChannelConfiguration schemeIdUri="urn:mpeg:dash:23003:3:audio_channel_configuration:2011" value="{}"/></Representation>"#,
            self.id,
            self.bitrate,
            self.codec.codecs_string,
            self.codec.sample_rate,
            self.codec.channels
        ));

        xml.push_str("</AdaptationSet>");
        xml
    }

    /// Validate the descriptor.
    ///
    /// # Errors
    ///
    /// Returns an error if required fields are missing or invalid.
    pub fn validate(&self) -> PackagerResult<()> {
        if self.id.is_empty() {
            return Err(PackagerError::InvalidConfig(
                "Audio track ID must not be empty".into(),
            ));
        }
        if self.language.is_empty() {
            return Err(PackagerError::InvalidConfig(
                "Audio track language must not be empty".into(),
            ));
        }
        if self.bitrate == 0 {
            return Err(PackagerError::InvalidConfig(
                "Audio track bitrate must not be zero".into(),
            ));
        }
        if self.codec.sample_rate == 0 {
            return Err(PackagerError::InvalidConfig(
                "Audio track sample rate must not be zero".into(),
            ));
        }
        if self.codec.channels == 0 {
            return Err(PackagerError::InvalidConfig(
                "Audio track channel count must not be zero".into(),
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// MultiAudioSet
// ---------------------------------------------------------------------------

/// A collection of audio tracks forming a multi-audio presentation.
///
/// Used to generate the audio portion of HLS multivariant playlists
/// and DASH MPD documents.
#[derive(Debug, Clone, Default)]
pub struct MultiAudioSet {
    /// Audio group ID (used in HLS GROUP-ID).
    pub group_id: String,
    /// Audio track descriptors.
    tracks: Vec<AudioTrackDescriptor>,
}

impl MultiAudioSet {
    /// Create a new multi-audio set with the given group ID.
    #[must_use]
    pub fn new(group_id: impl Into<String>) -> Self {
        Self {
            group_id: group_id.into(),
            tracks: Vec::new(),
        }
    }

    /// Add an audio track.
    pub fn add_track(&mut self, track: AudioTrackDescriptor) {
        self.tracks.push(track);
    }

    /// Return all tracks.
    #[must_use]
    pub fn tracks(&self) -> &[AudioTrackDescriptor] {
        &self.tracks
    }

    /// Number of tracks.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tracks.len()
    }

    /// Whether the set is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tracks.is_empty()
    }

    /// Return the default track, if one is set.
    #[must_use]
    pub fn default_track(&self) -> Option<&AudioTrackDescriptor> {
        self.tracks.iter().find(|t| t.is_default)
    }

    /// Return tracks for a given language.
    #[must_use]
    pub fn tracks_for_language(&self, lang: &str) -> Vec<&AudioTrackDescriptor> {
        self.tracks.iter().filter(|t| t.language == lang).collect()
    }

    /// Return all unique languages in the set.
    #[must_use]
    pub fn languages(&self) -> Vec<&str> {
        let mut langs: Vec<&str> = self.tracks.iter().map(|t| t.language.as_str()).collect();
        langs.sort();
        langs.dedup();
        langs
    }

    /// Render all tracks as HLS `EXT-X-MEDIA` tags.
    #[must_use]
    pub fn to_hls_media_tags(&self) -> String {
        let mut out = String::new();
        for track in &self.tracks {
            out.push_str(&track.to_hls_media_tag(&self.group_id));
            out.push('\n');
        }
        out
    }

    /// Render all tracks as DASH `<AdaptationSet>` XML fragments.
    #[must_use]
    pub fn to_dash_adaptation_sets(&self) -> String {
        let mut out = String::new();
        for track in &self.tracks {
            out.push_str(&track.to_dash_adaptation_set());
            out.push('\n');
        }
        out
    }

    /// Validate all tracks in the set.
    ///
    /// # Errors
    ///
    /// Returns the first validation error encountered.
    pub fn validate(&self) -> PackagerResult<()> {
        if self.group_id.is_empty() {
            return Err(PackagerError::InvalidConfig(
                "Audio group ID must not be empty".into(),
            ));
        }
        // Check for duplicate IDs
        for (i, track) in self.tracks.iter().enumerate() {
            track.validate()?;
            for other in &self.tracks[i + 1..] {
                if track.id == other.id {
                    return Err(PackagerError::InvalidConfig(format!(
                        "Duplicate audio track ID: {}",
                        track.id
                    )));
                }
            }
        }
        // Check at most one default
        let default_count = self.tracks.iter().filter(|t| t.is_default).count();
        if default_count > 1 {
            return Err(PackagerError::InvalidConfig(
                "At most one audio track can be the default".into(),
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn opus_codec() -> AudioCodecInfo {
        AudioCodecInfo::opus(48_000, 2)
    }

    fn flac_codec() -> AudioCodecInfo {
        AudioCodecInfo::flac(44_100, 2)
    }

    // --- AudioRole ----------------------------------------------------------

    #[test]
    fn test_audio_role_dash_values() {
        assert_eq!(AudioRole::Main.dash_role_value(), "main");
        assert_eq!(AudioRole::Dub.dash_role_value(), "dub");
        assert_eq!(AudioRole::Description.dash_role_value(), "description");
        assert_eq!(AudioRole::Commentary.dash_role_value(), "commentary");
        assert_eq!(AudioRole::Emergency.dash_role_value(), "emergency");
        assert_eq!(AudioRole::Supplementary.dash_role_value(), "supplementary");
    }

    #[test]
    fn test_audio_role_hls_characteristics() {
        assert!(AudioRole::Description.hls_characteristics().is_some());
        assert!(AudioRole::Commentary.hls_characteristics().is_some());
        assert!(AudioRole::Main.hls_characteristics().is_none());
        assert!(AudioRole::Dub.hls_characteristics().is_none());
    }

    // --- AudioCodecInfo -----------------------------------------------------

    #[test]
    fn test_opus_codec() {
        let c = AudioCodecInfo::opus(48_000, 2);
        assert_eq!(c.codecs_string, "opus");
        assert_eq!(c.sample_rate, 48_000);
        assert_eq!(c.channels, 2);
    }

    #[test]
    fn test_flac_codec() {
        let c = AudioCodecInfo::flac(96_000, 6);
        assert_eq!(c.codecs_string, "fLaC");
        assert_eq!(c.sample_rate, 96_000);
        assert_eq!(c.channels, 6);
    }

    #[test]
    fn test_vorbis_codec() {
        let c = AudioCodecInfo::vorbis(44_100, 2);
        assert_eq!(c.codecs_string, "vorbis");
    }

    // --- AudioTrackDescriptor -----------------------------------------------

    #[test]
    fn test_audio_track_new() {
        let t = AudioTrackDescriptor::new("en-main", "en", "English", opus_codec(), 128_000);
        assert_eq!(t.id, "en-main");
        assert_eq!(t.language, "en");
        assert_eq!(t.label, "English");
        assert_eq!(t.bitrate, 128_000);
        assert_eq!(t.role, AudioRole::Main);
        assert!(!t.is_default);
    }

    #[test]
    fn test_audio_track_with_role() {
        let t = AudioTrackDescriptor::new("ad-en", "en", "Audio Description", opus_codec(), 64_000)
            .with_role(AudioRole::Description);
        assert_eq!(t.role, AudioRole::Description);
    }

    #[test]
    fn test_audio_track_as_default() {
        let t =
            AudioTrackDescriptor::new("en", "en", "English", opus_codec(), 128_000).as_default();
        assert!(t.is_default);
    }

    #[test]
    fn test_audio_track_hls_media_tag() {
        let t = AudioTrackDescriptor::new("en-main", "en", "English", opus_codec(), 128_000)
            .as_default()
            .with_playlist_uri("audio/en/index.m3u8");

        let tag = t.to_hls_media_tag("audio-group");
        assert!(tag.contains("EXT-X-MEDIA"));
        assert!(tag.contains("TYPE=AUDIO"));
        assert!(tag.contains("GROUP-ID=\"audio-group\""));
        assert!(tag.contains("NAME=\"English\""));
        assert!(tag.contains("LANGUAGE=\"en\""));
        assert!(tag.contains("DEFAULT=YES"));
        assert!(tag.contains("CHANNELS=\"2\""));
        assert!(tag.contains("URI=\"audio/en/index.m3u8\""));
    }

    #[test]
    fn test_audio_track_hls_media_tag_non_default() {
        let t = AudioTrackDescriptor::new("ja", "ja", "Japanese", opus_codec(), 128_000)
            .with_playlist_uri("audio/ja/index.m3u8");

        let tag = t.to_hls_media_tag("audio-group");
        assert!(tag.contains("DEFAULT=NO"));
        assert!(tag.contains("LANGUAGE=\"ja\""));
    }

    #[test]
    fn test_audio_track_hls_with_accessibility() {
        let t = AudioTrackDescriptor::new("ad", "en", "Audio Description", opus_codec(), 64_000)
            .with_role(AudioRole::Description)
            .with_playlist_uri("audio/ad/index.m3u8");

        let tag = t.to_hls_media_tag("audio-group");
        assert!(tag.contains("CHARACTERISTICS=\"public.accessibility.describes-video\""));
    }

    #[test]
    fn test_audio_track_dash_adaptation_set() {
        let t = AudioTrackDescriptor::new("en-main", "en", "English", opus_codec(), 128_000);

        let xml = t.to_dash_adaptation_set();
        assert!(xml.contains("<AdaptationSet"));
        assert!(xml.contains("lang=\"en\""));
        assert!(xml.contains("contentType=\"audio\""));
        assert!(xml.contains("value=\"main\""));
        assert!(xml.contains("bandwidth=\"128000\""));
        assert!(xml.contains("codecs=\"opus\""));
        assert!(xml.contains("audioSamplingRate=\"48000\""));
        assert!(xml.contains("</AdaptationSet>"));
    }

    #[test]
    fn test_audio_track_validate_ok() {
        let t = AudioTrackDescriptor::new("en", "en", "English", opus_codec(), 128_000);
        assert!(t.validate().is_ok());
    }

    #[test]
    fn test_audio_track_validate_empty_id() {
        let t = AudioTrackDescriptor::new("", "en", "English", opus_codec(), 128_000);
        assert!(t.validate().is_err());
    }

    #[test]
    fn test_audio_track_validate_empty_language() {
        let t = AudioTrackDescriptor::new("en", "", "English", opus_codec(), 128_000);
        assert!(t.validate().is_err());
    }

    #[test]
    fn test_audio_track_validate_zero_bitrate() {
        let t = AudioTrackDescriptor::new("en", "en", "English", opus_codec(), 0);
        assert!(t.validate().is_err());
    }

    #[test]
    fn test_audio_track_validate_zero_sample_rate() {
        let c = AudioCodecInfo::new("opus", 0, 2);
        let t = AudioTrackDescriptor::new("en", "en", "English", c, 128_000);
        assert!(t.validate().is_err());
    }

    #[test]
    fn test_audio_track_validate_zero_channels() {
        let c = AudioCodecInfo::new("opus", 48_000, 0);
        let t = AudioTrackDescriptor::new("en", "en", "English", c, 128_000);
        assert!(t.validate().is_err());
    }

    // --- MultiAudioSet ------------------------------------------------------

    #[test]
    fn test_multi_audio_set_new() {
        let set = MultiAudioSet::new("audio-group");
        assert_eq!(set.group_id, "audio-group");
        assert!(set.is_empty());
    }

    #[test]
    fn test_multi_audio_set_add_tracks() {
        let mut set = MultiAudioSet::new("audio-group");
        set.add_track(
            AudioTrackDescriptor::new("en", "en", "English", opus_codec(), 128_000)
                .as_default()
                .with_playlist_uri("audio/en/index.m3u8"),
        );
        set.add_track(
            AudioTrackDescriptor::new("ja", "ja", "Japanese", opus_codec(), 128_000)
                .with_playlist_uri("audio/ja/index.m3u8"),
        );
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_multi_audio_set_default_track() {
        let mut set = MultiAudioSet::new("audio-group");
        set.add_track(
            AudioTrackDescriptor::new("en", "en", "English", opus_codec(), 128_000).as_default(),
        );
        set.add_track(AudioTrackDescriptor::new(
            "ja",
            "ja",
            "Japanese",
            opus_codec(),
            128_000,
        ));
        let def = set.default_track();
        assert!(def.is_some());
        assert_eq!(def.map(|t| t.id.as_str()), Some("en"));
    }

    #[test]
    fn test_multi_audio_set_languages() {
        let mut set = MultiAudioSet::new("audio-group");
        set.add_track(AudioTrackDescriptor::new(
            "en",
            "en",
            "English",
            opus_codec(),
            128_000,
        ));
        set.add_track(AudioTrackDescriptor::new(
            "ja",
            "ja",
            "Japanese",
            opus_codec(),
            128_000,
        ));
        set.add_track(
            AudioTrackDescriptor::new("en-ad", "en", "English AD", opus_codec(), 64_000)
                .with_role(AudioRole::Description),
        );

        let langs = set.languages();
        assert_eq!(langs, vec!["en", "ja"]);
    }

    #[test]
    fn test_multi_audio_set_tracks_for_language() {
        let mut set = MultiAudioSet::new("audio-group");
        set.add_track(AudioTrackDescriptor::new(
            "en",
            "en",
            "English",
            opus_codec(),
            128_000,
        ));
        set.add_track(AudioTrackDescriptor::new(
            "ja",
            "ja",
            "Japanese",
            opus_codec(),
            128_000,
        ));

        let en = set.tracks_for_language("en");
        assert_eq!(en.len(), 1);
        assert_eq!(en[0].id, "en");
    }

    #[test]
    fn test_multi_audio_set_to_hls_media_tags() {
        let mut set = MultiAudioSet::new("audio-group");
        set.add_track(
            AudioTrackDescriptor::new("en", "en", "English", opus_codec(), 128_000)
                .as_default()
                .with_playlist_uri("audio/en.m3u8"),
        );
        set.add_track(
            AudioTrackDescriptor::new("ja", "ja", "Japanese", opus_codec(), 128_000)
                .with_playlist_uri("audio/ja.m3u8"),
        );

        let tags = set.to_hls_media_tags();
        let lines: Vec<&str> = tags.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("LANGUAGE=\"en\""));
        assert!(lines[1].contains("LANGUAGE=\"ja\""));
    }

    #[test]
    fn test_multi_audio_set_to_dash_adaptation_sets() {
        let mut set = MultiAudioSet::new("audio-group");
        set.add_track(AudioTrackDescriptor::new(
            "en",
            "en",
            "English",
            opus_codec(),
            128_000,
        ));
        set.add_track(AudioTrackDescriptor::new(
            "ja",
            "ja",
            "Japanese",
            flac_codec(),
            256_000,
        ));

        let xml = set.to_dash_adaptation_sets();
        assert!(xml.contains("lang=\"en\""));
        assert!(xml.contains("lang=\"ja\""));
        assert!(xml.contains("codecs=\"opus\""));
        assert!(xml.contains("codecs=\"fLaC\""));
    }

    #[test]
    fn test_multi_audio_set_validate_ok() {
        let mut set = MultiAudioSet::new("audio-group");
        set.add_track(
            AudioTrackDescriptor::new("en", "en", "English", opus_codec(), 128_000).as_default(),
        );
        assert!(set.validate().is_ok());
    }

    #[test]
    fn test_multi_audio_set_validate_empty_group_id() {
        let set = MultiAudioSet::new("");
        assert!(set.validate().is_err());
    }

    #[test]
    fn test_multi_audio_set_validate_duplicate_ids() {
        let mut set = MultiAudioSet::new("audio-group");
        set.add_track(AudioTrackDescriptor::new(
            "en",
            "en",
            "English",
            opus_codec(),
            128_000,
        ));
        set.add_track(AudioTrackDescriptor::new(
            "en",
            "en",
            "English 2",
            opus_codec(),
            64_000,
        ));
        assert!(set.validate().is_err());
    }

    #[test]
    fn test_multi_audio_set_validate_multiple_defaults() {
        let mut set = MultiAudioSet::new("audio-group");
        set.add_track(
            AudioTrackDescriptor::new("en", "en", "English", opus_codec(), 128_000).as_default(),
        );
        set.add_track(
            AudioTrackDescriptor::new("ja", "ja", "Japanese", opus_codec(), 128_000).as_default(),
        );
        assert!(set.validate().is_err());
    }

    #[test]
    fn test_audio_track_with_init_uri() {
        let t = AudioTrackDescriptor::new("en", "en", "English", opus_codec(), 128_000)
            .with_init_uri("audio/en/init.mp4");
        assert_eq!(t.init_uri, Some("audio/en/init.mp4".to_string()));
    }

    #[test]
    fn test_audio_track_with_custom_accessibility() {
        let t = AudioTrackDescriptor::new("en-hi", "en", "Hearing Impaired", opus_codec(), 128_000)
            .with_accessibility("public.accessibility.transcribes-spoken-dialog");
        assert_eq!(t.accessibility_labels.len(), 1);
    }
}
