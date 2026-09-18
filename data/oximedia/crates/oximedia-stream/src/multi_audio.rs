//! Multi-audio track management for adaptive streaming manifests.
//!
//! Supports language selection, accessibility tracks, and HLS `#EXT-X-MEDIA`
//! tag generation for multi-language and multi-channel audio renditions.

use crate::StreamError;

// ─── Audio Codec Identifiers ─────────────────────────────────────────────────

/// Codec identifier for an audio track.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AudioCodecId {
    /// AAC-LC (ISO 14496-3).
    AacLc,
    /// HE-AAC v1 (SBR).
    HeAacV1,
    /// HE-AAC v2 (SBR + PS).
    HeAacV2,
    /// Opus (RFC 7845).
    Opus,
    /// AC-3 / Dolby Digital.
    Ac3,
    /// E-AC-3 / Dolby Digital Plus.
    Eac3,
    /// Free-form codec string (e.g. `"mp4a.40.29"`).
    Other(String),
}

impl AudioCodecId {
    /// Return the RFC 6381 codec string for use in HLS manifests.
    pub fn codec_string(&self) -> String {
        match self {
            Self::AacLc => "mp4a.40.2".to_string(),
            Self::HeAacV1 => "mp4a.40.5".to_string(),
            Self::HeAacV2 => "mp4a.40.29".to_string(),
            Self::Opus => "opus".to_string(),
            Self::Ac3 => "ac-3".to_string(),
            Self::Eac3 => "ec-3".to_string(),
            Self::Other(s) => s.clone(),
        }
    }
}

// ─── Audio Track ─────────────────────────────────────────────────────────────

/// Describes a single audio rendition within an adaptive stream.
#[derive(Debug, Clone)]
pub struct AudioTrack {
    /// Unique identifier for this track (used in `GROUP-ID`).
    pub id: String,
    /// BCP-47 language tag (e.g. `"en"`, `"fr-CA"`).
    pub language: String,
    /// Human-readable name shown in player UI (e.g. `"English"`).
    pub name: String,
    /// Number of audio channels.
    pub channels: u8,
    /// Audio codec in use.
    pub codec: AudioCodecId,
    /// Whether this is the default rendition when no preference is set.
    pub default: bool,
    /// Whether a player may automatically select this track.
    pub autoselect: bool,
    /// Optional URI for the rendition's media playlist.
    pub uri: Option<String>,
    /// Whether this track is for accessibility purposes (e.g. audio description).
    pub accessibility: bool,
    /// Group identifier used in `#EXT-X-MEDIA GROUP-ID`.
    pub group_id: String,
}

impl AudioTrack {
    /// Create a new audio track with sensible defaults.
    pub fn new(
        id: impl Into<String>,
        language: impl Into<String>,
        name: impl Into<String>,
    ) -> Self {
        let id = id.into();
        Self {
            group_id: "audio".to_string(),
            id,
            language: language.into(),
            name: name.into(),
            channels: 2,
            codec: AudioCodecId::AacLc,
            default: false,
            autoselect: true,
            uri: None,
            accessibility: false,
        }
    }

    /// Render this track as a single `#EXT-X-MEDIA` line.
    pub fn to_ext_x_media_line(&self) -> String {
        let mut line = format!(
            "#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID=\"{}\",LANGUAGE=\"{}\",NAME=\"{}\"",
            self.group_id, self.language, self.name
        );
        line.push_str(if self.default {
            ",DEFAULT=YES"
        } else {
            ",DEFAULT=NO"
        });
        line.push_str(if self.autoselect {
            ",AUTOSELECT=YES"
        } else {
            ",AUTOSELECT=NO"
        });
        line.push_str(&format!(",CHANNELS=\"{}\"", self.channels));
        line.push_str(&format!(",CODECS=\"{}\"", self.codec.codec_string()));
        if self.accessibility {
            line.push_str(",CHARACTERISTICS=\"public.accessibility.describes-video\"");
        }
        if let Some(uri) = &self.uri {
            line.push_str(&format!(",URI=\"{}\"", uri));
        }
        line
    }
}

// ─── Audio Track Manager ─────────────────────────────────────────────────────

/// Manages a collection of audio tracks for an adaptive streaming presentation.
#[derive(Debug, Default)]
pub struct AudioTrackManager {
    tracks: Vec<AudioTrack>,
}

impl AudioTrackManager {
    /// Create an empty manager.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a track to the manager.
    ///
    /// Returns an error if a track with the same `id` already exists.
    pub fn add_track(&mut self, track: AudioTrack) -> Result<(), StreamError> {
        if self.tracks.iter().any(|t| t.id == track.id) {
            return Err(StreamError::Generic(format!(
                "audio track with id '{}' already exists",
                track.id
            )));
        }
        self.tracks.push(track);
        Ok(())
    }

    /// Remove a track by `id`.
    ///
    /// Returns `true` if the track was found and removed.
    pub fn remove_track(&mut self, id: &str) -> bool {
        let before = self.tracks.len();
        self.tracks.retain(|t| t.id != id);
        self.tracks.len() < before
    }

    /// Select the first track matching `language` (BCP-47 prefix match).
    ///
    /// For example, `"fr"` matches `"fr"` and `"fr-CA"`.
    pub fn select_track(&self, language: &str) -> Option<&AudioTrack> {
        // Exact match first
        if let Some(t) = self.tracks.iter().find(|t| t.language == language) {
            return Some(t);
        }
        // Prefix match (e.g. "en" matches "en-US")
        self.tracks
            .iter()
            .find(|t| t.language.starts_with(language))
    }

    /// Return the default track, if one is marked.
    pub fn default_track(&self) -> Option<&AudioTrack> {
        self.tracks.iter().find(|t| t.default)
    }

    /// Return all tracks in this manager.
    pub fn tracks(&self) -> &[AudioTrack] {
        &self.tracks
    }

    /// Return the number of tracks.
    pub fn len(&self) -> usize {
        self.tracks.len()
    }

    /// Return `true` if no tracks are registered.
    pub fn is_empty(&self) -> bool {
        self.tracks.is_empty()
    }

    /// Generate the full `#EXT-X-MEDIA` block for inclusion in an HLS master
    /// playlist.
    pub fn to_hls_ext_x_media(&self) -> String {
        self.tracks
            .iter()
            .map(|t| t.to_ext_x_media_line())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Set the default track by `id`, clearing any previous default.
    ///
    /// Returns an error if no track with that `id` exists.
    pub fn set_default(&mut self, id: &str) -> Result<(), StreamError> {
        if !self.tracks.iter().any(|t| t.id == id) {
            return Err(StreamError::Generic(format!(
                "no track with id '{}' found",
                id
            )));
        }
        for track in &mut self.tracks {
            track.default = track.id == id;
        }
        Ok(())
    }

    /// Return tracks filtered by group_id.
    pub fn tracks_in_group<'a>(&'a self, group_id: &str) -> Vec<&'a AudioTrack> {
        self.tracks
            .iter()
            .filter(|t| t.group_id == group_id)
            .collect()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn english_track() -> AudioTrack {
        let mut t = AudioTrack::new("en-main", "en", "English");
        t.default = true;
        t.uri = Some("audio_en.m3u8".to_string());
        t
    }

    fn french_track() -> AudioTrack {
        let mut t = AudioTrack::new("fr-main", "fr-CA", "Français");
        t.channels = 2;
        t
    }

    #[test]
    fn test_add_track_and_count() {
        let mut mgr = AudioTrackManager::new();
        mgr.add_track(english_track()).expect("add en");
        mgr.add_track(french_track()).expect("add fr");
        assert_eq!(mgr.len(), 2);
    }

    #[test]
    fn test_duplicate_id_rejected() {
        let mut mgr = AudioTrackManager::new();
        mgr.add_track(english_track()).expect("first add");
        let err = mgr.add_track(english_track());
        assert!(err.is_err(), "duplicate id must be rejected");
    }

    #[test]
    fn test_select_exact_language() {
        let mut mgr = AudioTrackManager::new();
        mgr.add_track(english_track()).expect("add");
        mgr.add_track(french_track()).expect("add");
        let track = mgr.select_track("fr-CA").expect("should find fr-CA");
        assert_eq!(track.id, "fr-main");
    }

    #[test]
    fn test_select_prefix_language() {
        let mut mgr = AudioTrackManager::new();
        mgr.add_track(french_track()).expect("add");
        // "fr" prefix should match "fr-CA"
        let track = mgr.select_track("fr").expect("prefix match");
        assert_eq!(track.id, "fr-main");
    }

    #[test]
    fn test_select_missing_language_returns_none() {
        let mgr = AudioTrackManager::new();
        assert!(mgr.select_track("ja").is_none());
    }

    #[test]
    fn test_default_track() {
        let mut mgr = AudioTrackManager::new();
        mgr.add_track(english_track()).expect("add");
        mgr.add_track(french_track()).expect("add");
        let d = mgr.default_track().expect("should have default");
        assert_eq!(d.id, "en-main");
    }

    #[test]
    fn test_set_default() {
        let mut mgr = AudioTrackManager::new();
        mgr.add_track(english_track()).expect("add");
        mgr.add_track(french_track()).expect("add");
        mgr.set_default("fr-main").expect("set default");
        let d = mgr.default_track().expect("default");
        assert_eq!(d.id, "fr-main");
        // old default should be cleared
        let en = mgr.select_track("en").expect("en");
        assert!(!en.default);
    }

    #[test]
    fn test_ext_x_media_contains_type_audio() {
        let mut mgr = AudioTrackManager::new();
        mgr.add_track(english_track()).expect("add");
        let block = mgr.to_hls_ext_x_media();
        assert!(block.contains("TYPE=AUDIO"));
    }

    #[test]
    fn test_ext_x_media_default_yes() {
        let track = english_track(); // default=true
        let line = track.to_ext_x_media_line();
        assert!(line.contains("DEFAULT=YES"));
    }

    #[test]
    fn test_ext_x_media_uri_included() {
        let track = english_track();
        let line = track.to_ext_x_media_line();
        assert!(line.contains("URI=\"audio_en.m3u8\""));
    }

    #[test]
    fn test_remove_track() {
        let mut mgr = AudioTrackManager::new();
        mgr.add_track(english_track()).expect("add");
        assert!(mgr.remove_track("en-main"));
        assert_eq!(mgr.len(), 0);
    }

    #[test]
    fn test_codec_strings() {
        assert_eq!(AudioCodecId::AacLc.codec_string(), "mp4a.40.2");
        assert_eq!(AudioCodecId::Opus.codec_string(), "opus");
        assert_eq!(AudioCodecId::Eac3.codec_string(), "ec-3");
    }
}
