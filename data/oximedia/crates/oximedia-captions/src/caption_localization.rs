//! Multi-language caption localization management.
//!
//! This module provides structures and utilities for managing multiple caption tracks
//! across different languages for a single video asset. It handles:
//! - Storing and retrieving caption tracks by language code
//! - Fallback language resolution (e.g. `"en-GB"` → `"en"`)
//! - Marking a primary/default track
//! - Exporting a manifest of available languages
//!
//! # Usage
//! ```rust
//! use oximedia_captions::caption_localization::{LocalizationSet, LocalizationConfig};
//! use oximedia_captions::caption_localization::LocalizedTrack;
//!
//! let mut set = LocalizationSet::new(LocalizationConfig::default());
//! let track = LocalizedTrack::new("en".to_string(), "English".to_string());
//! set.add_track(track);
//! assert!(set.has_language("en"));
//! ```

#![allow(dead_code)]

use std::collections::HashMap;

// ── LocalizedTrack ────────────────────────────────────────────────────────────

/// A caption track associated with a BCP-47 language tag.
#[derive(Debug, Clone)]
pub struct LocalizedTrack {
    /// BCP-47 language tag (e.g. `"en"`, `"fr-CA"`, `"zh-Hant-TW"`).
    pub language_tag: String,
    /// Human-readable language name (e.g. `"English"`, `"French (Canada)"`).
    pub display_name: String,
    /// Whether this is the primary/default track for this language family.
    pub is_default: bool,
    /// Whether this track is SDH (Subtitles for Deaf and Hard-of-Hearing).
    pub is_sdh: bool,
    /// Whether this track contains forced narrative subtitles only.
    pub is_forced: bool,
    /// Optional label for the track (e.g. `"Commentary"`, `"Original"`).
    pub label: Option<String>,
    /// Opaque caption data (serialized format, e.g. SRT or VTT bytes).
    pub data: Vec<u8>,
    /// MIME type of the `data` field (e.g. `"text/vtt"`, `"application/ttml+xml"`).
    pub mime_type: String,
}

impl LocalizedTrack {
    /// Create a new empty localized track.
    #[must_use]
    pub fn new(language_tag: String, display_name: String) -> Self {
        Self {
            language_tag,
            display_name,
            is_default: false,
            is_sdh: false,
            is_forced: false,
            label: None,
            data: Vec::new(),
            mime_type: "text/vtt".to_string(),
        }
    }

    /// Builder: mark this track as default.
    #[must_use]
    pub fn as_default(mut self) -> Self {
        self.is_default = true;
        self
    }

    /// Builder: mark this track as SDH.
    #[must_use]
    pub fn as_sdh(mut self) -> Self {
        self.is_sdh = true;
        self
    }

    /// Builder: mark this track as forced narrative.
    #[must_use]
    pub fn as_forced(mut self) -> Self {
        self.is_forced = true;
        self
    }

    /// Builder: set caption data.
    #[must_use]
    pub fn with_data(mut self, data: Vec<u8>, mime_type: impl Into<String>) -> Self {
        self.data = data;
        self.mime_type = mime_type.into();
        self
    }

    /// Builder: set optional label.
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Extract the primary language subtag (e.g. `"en"` from `"en-GB"`).
    #[must_use]
    pub fn primary_subtag(&self) -> &str {
        self.language_tag
            .split('-')
            .next()
            .unwrap_or(&self.language_tag)
    }

    /// Whether the track has content.
    #[must_use]
    pub fn has_content(&self) -> bool {
        !self.data.is_empty()
    }
}

// ── LocalizationConfig ────────────────────────────────────────────────────────

/// Configuration for localization set behaviour.
#[derive(Debug, Clone)]
pub struct LocalizationConfig {
    /// Whether to allow duplicate language tags (e.g. two `"en"` tracks).
    pub allow_duplicates: bool,
    /// Fallback resolution: when a specific variant is not found, try the primary subtag.
    pub use_fallback: bool,
    /// Maximum number of tracks allowed (0 = unlimited).
    pub max_tracks: usize,
}

impl Default for LocalizationConfig {
    fn default() -> Self {
        Self {
            allow_duplicates: false,
            use_fallback: true,
            max_tracks: 0,
        }
    }
}

// ── LanguageManifest ─────────────────────────────────────────────────────────

/// A lightweight manifest entry describing one available caption track.
#[derive(Debug, Clone, PartialEq)]
pub struct ManifestEntry {
    /// BCP-47 language tag.
    pub language_tag: String,
    /// Human-readable name.
    pub display_name: String,
    /// MIME type.
    pub mime_type: String,
    /// Whether this is the default track.
    pub is_default: bool,
    /// Whether SDH.
    pub is_sdh: bool,
    /// Whether forced narrative.
    pub is_forced: bool,
    /// Optional label.
    pub label: Option<String>,
}

// ── LocalizationSet ───────────────────────────────────────────────────────────

/// Manages a set of localized caption tracks for a single video asset.
#[derive(Debug, Clone)]
pub struct LocalizationSet {
    config: LocalizationConfig,
    /// Tracks stored by normalized language tag.
    tracks: Vec<LocalizedTrack>,
    /// Index: language_tag → track indices in `tracks`.
    index: HashMap<String, Vec<usize>>,
}

impl LocalizationSet {
    /// Create a new empty localization set.
    #[must_use]
    pub fn new(config: LocalizationConfig) -> Self {
        Self {
            config,
            tracks: Vec::new(),
            index: HashMap::new(),
        }
    }

    /// Add a track to the set.
    ///
    /// If `allow_duplicates` is `false` and a track with the same language tag
    /// already exists, the existing track is replaced.
    pub fn add_track(&mut self, track: LocalizedTrack) {
        let tag = track.language_tag.to_lowercase();

        if !self.config.allow_duplicates {
            // Remove any existing track with the same tag
            if let Some(indices) = self.index.get(&tag) {
                let idx = indices[0];
                self.tracks.remove(idx);
                self.rebuild_index();
            }
        }

        let idx = self.tracks.len();
        self.tracks.push(track);
        self.index.entry(tag).or_default().push(idx);
    }

    /// Remove a track by language tag.
    ///
    /// Returns `true` if a track was removed.
    pub fn remove_track(&mut self, language_tag: &str) -> bool {
        let tag = language_tag.to_lowercase();
        if self.index.contains_key(&tag) {
            self.tracks.retain(|t| t.language_tag.to_lowercase() != tag);
            self.rebuild_index();
            true
        } else {
            false
        }
    }

    /// Retrieve the first track matching `language_tag`.
    ///
    /// If `use_fallback` is enabled and no exact match is found, the primary
    /// language subtag is tried (e.g. `"en-GB"` → `"en"`).
    #[must_use]
    pub fn get_track(&self, language_tag: &str) -> Option<&LocalizedTrack> {
        let tag = language_tag.to_lowercase();
        if let Some(indices) = self.index.get(&tag) {
            return self.tracks.get(indices[0]);
        }
        if self.config.use_fallback {
            let primary = tag.split('-').next().unwrap_or(&tag).to_string();
            if primary != tag {
                if let Some(indices) = self.index.get(&primary) {
                    return self.tracks.get(indices[0]);
                }
            }
        }
        None
    }

    /// Get the default track (first track marked `is_default`, or the first track overall).
    #[must_use]
    pub fn default_track(&self) -> Option<&LocalizedTrack> {
        self.tracks
            .iter()
            .find(|t| t.is_default)
            .or_else(|| self.tracks.first())
    }

    /// Check whether a language is available.
    #[must_use]
    pub fn has_language(&self, language_tag: &str) -> bool {
        self.get_track(language_tag).is_some()
    }

    /// All language tags currently in the set.
    #[must_use]
    pub fn language_tags(&self) -> Vec<&str> {
        self.tracks
            .iter()
            .map(|t| t.language_tag.as_str())
            .collect()
    }

    /// Total number of tracks.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tracks.len()
    }

    /// Whether the set is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tracks.is_empty()
    }

    /// Export a manifest of all available tracks (lightweight metadata only).
    #[must_use]
    pub fn manifest(&self) -> Vec<ManifestEntry> {
        self.tracks
            .iter()
            .map(|t| ManifestEntry {
                language_tag: t.language_tag.clone(),
                display_name: t.display_name.clone(),
                mime_type: t.mime_type.clone(),
                is_default: t.is_default,
                is_sdh: t.is_sdh,
                is_forced: t.is_forced,
                label: t.label.clone(),
            })
            .collect()
    }

    /// Set the default track by language tag.
    ///
    /// Clears `is_default` from all other tracks.
    pub fn set_default(&mut self, language_tag: &str) -> bool {
        let tag = language_tag.to_lowercase();
        let found = self
            .tracks
            .iter()
            .any(|t| t.language_tag.to_lowercase() == tag);
        if found {
            for track in &mut self.tracks {
                track.is_default = track.language_tag.to_lowercase() == tag;
            }
        }
        found
    }

    /// Rebuild the language → index mapping after structural changes.
    fn rebuild_index(&mut self) {
        self.index.clear();
        for (i, track) in self.tracks.iter().enumerate() {
            self.index
                .entry(track.language_tag.to_lowercase())
                .or_default()
                .push(i);
        }
    }

    /// Iterate over all tracks in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = &LocalizedTrack> {
        self.tracks.iter()
    }
}

impl Default for LocalizationSet {
    fn default() -> Self {
        Self::new(LocalizationConfig::default())
    }
}

// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_set() -> LocalizationSet {
        let mut set = LocalizationSet::default();
        set.add_track(LocalizedTrack::new("en".to_string(), "English".to_string()).as_default());
        set.add_track(LocalizedTrack::new("fr".to_string(), "French".to_string()));
        set.add_track(LocalizedTrack::new("de".to_string(), "German".to_string()));
        set
    }

    #[test]
    fn test_has_language() {
        let set = make_set();
        assert!(set.has_language("en"));
        assert!(set.has_language("fr"));
        assert!(!set.has_language("ja"));
    }

    #[test]
    fn test_get_track_exact() {
        let set = make_set();
        let t = set.get_track("fr");
        assert!(t.is_some());
        assert_eq!(
            t.expect("value should be present should succeed")
                .display_name,
            "French"
        );
    }

    #[test]
    fn test_get_track_fallback() {
        let mut set = LocalizationSet::default();
        set.add_track(LocalizedTrack::new("en".to_string(), "English".to_string()));
        // "en-GB" should fall back to "en"
        let t = set.get_track("en-GB");
        assert!(t.is_some());
        assert_eq!(
            t.expect("value should be present should succeed")
                .language_tag,
            "en"
        );
    }

    #[test]
    fn test_get_track_no_fallback() {
        let mut set = LocalizationSet::new(LocalizationConfig {
            use_fallback: false,
            ..Default::default()
        });
        set.add_track(LocalizedTrack::new("en".to_string(), "English".to_string()));
        assert!(set.get_track("en-GB").is_none());
    }

    #[test]
    fn test_default_track_is_first_marked() {
        let set = make_set();
        let def = set.default_track();
        assert!(def.is_some());
        assert_eq!(
            def.expect("value should be present should succeed")
                .language_tag,
            "en"
        );
    }

    #[test]
    fn test_default_track_falls_back_to_first() {
        let mut set = LocalizationSet::default();
        set.add_track(LocalizedTrack::new("de".to_string(), "German".to_string()));
        let def = set.default_track();
        assert!(def.is_some());
        assert_eq!(
            def.expect("value should be present should succeed")
                .language_tag,
            "de"
        );
    }

    #[test]
    fn test_set_default() {
        let mut set = make_set();
        set.set_default("fr");
        assert!(
            set.get_track("fr")
                .expect("get track should succeed")
                .is_default
        );
        assert!(
            !set.get_track("en")
                .expect("get track should succeed")
                .is_default
        );
    }

    #[test]
    fn test_remove_track() {
        let mut set = make_set();
        assert_eq!(set.len(), 3);
        let removed = set.remove_track("fr");
        assert!(removed);
        assert_eq!(set.len(), 2);
        assert!(!set.has_language("fr"));
    }

    #[test]
    fn test_remove_nonexistent_track() {
        let mut set = make_set();
        let removed = set.remove_track("ja");
        assert!(!removed);
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn test_no_duplicate_by_default() {
        let mut set = LocalizationSet::default();
        set.add_track(LocalizedTrack::new("en".to_string(), "English".to_string()));
        set.add_track(LocalizedTrack::new(
            "en".to_string(),
            "English US".to_string(),
        ));
        assert_eq!(set.len(), 1);
        assert_eq!(
            set.get_track("en")
                .expect("get track should succeed")
                .display_name,
            "English US"
        );
    }

    #[test]
    fn test_allow_duplicates() {
        let mut set = LocalizationSet::new(LocalizationConfig {
            allow_duplicates: true,
            ..Default::default()
        });
        set.add_track(LocalizedTrack::new("en".to_string(), "English".to_string()));
        set.add_track(LocalizedTrack::new(
            "en".to_string(),
            "English SDH".to_string(),
        ));
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_manifest_entries() {
        let set = make_set();
        let manifest = set.manifest();
        assert_eq!(manifest.len(), 3);
        let en = manifest.iter().find(|e| e.language_tag == "en");
        assert!(en.is_some());
        assert!(
            en.expect("value should be present should succeed")
                .is_default
        );
    }

    #[test]
    fn test_language_tags() {
        let set = make_set();
        let tags = set.language_tags();
        assert!(tags.contains(&"en"));
        assert!(tags.contains(&"fr"));
        assert!(tags.contains(&"de"));
    }

    #[test]
    fn test_track_builder_methods() {
        let track = LocalizedTrack::new("en".to_string(), "English".to_string())
            .as_default()
            .as_sdh()
            .as_forced()
            .with_label("Commentary")
            .with_data(b"WEBVTT\n".to_vec(), "text/vtt");
        assert!(track.is_default);
        assert!(track.is_sdh);
        assert!(track.is_forced);
        assert_eq!(track.label.as_deref(), Some("Commentary"));
        assert!(track.has_content());
        assert_eq!(track.mime_type, "text/vtt");
    }

    #[test]
    fn test_primary_subtag() {
        let track =
            LocalizedTrack::new("zh-Hant-TW".to_string(), "Traditional Chinese".to_string());
        assert_eq!(track.primary_subtag(), "zh");
    }

    #[test]
    fn test_is_empty_and_len() {
        let mut set = LocalizationSet::default();
        assert!(set.is_empty());
        set.add_track(LocalizedTrack::new("en".to_string(), "English".to_string()));
        assert!(!set.is_empty());
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn test_iter() {
        let set = make_set();
        let count = set.iter().count();
        assert_eq!(count, 3);
    }
}
