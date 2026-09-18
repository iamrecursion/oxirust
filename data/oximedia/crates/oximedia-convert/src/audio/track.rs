// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Audio track selection and management.

use crate::{ConversionError, Result};
use std::path::Path;

/// Selector for audio tracks in media files.
#[derive(Debug, Clone)]
pub struct AudioTrackSelector {
    prefer_language: Option<String>,
    prefer_channels: Option<u32>,
}

impl AudioTrackSelector {
    /// Create a new audio track selector.
    #[must_use]
    pub fn new() -> Self {
        Self {
            prefer_language: None,
            prefer_channels: None,
        }
    }

    /// Prefer tracks with specific language.
    pub fn with_language<S: Into<String>>(mut self, language: S) -> Self {
        self.prefer_language = Some(language.into());
        self
    }

    /// Prefer tracks with specific number of channels.
    #[must_use]
    pub fn with_channels(mut self, channels: u32) -> Self {
        self.prefer_channels = Some(channels);
        self
    }

    /// List all audio tracks in a media file.
    pub fn list_tracks<P: AsRef<Path>>(&self, input: P) -> Result<Vec<AudioTrackInfo>> {
        let _input = input.as_ref();

        // Placeholder for actual track detection
        // In a real implementation, this would use oximedia-core
        Ok(Vec::new())
    }

    /// Select the best audio track based on preferences.
    pub fn select_best<P: AsRef<Path>>(&self, input: P) -> Result<AudioTrackInfo> {
        let tracks = self.list_tracks(input)?;

        if tracks.is_empty() {
            return Err(ConversionError::InvalidInput(
                "No audio tracks found".to_string(),
            ));
        }

        // Apply selection logic
        let mut best_track = &tracks[0];

        for track in &tracks {
            // Prefer matching language
            if let Some(ref lang) = self.prefer_language {
                if track.language.as_ref() == Some(lang) {
                    best_track = track;
                    break;
                }
            }

            // Prefer matching channels
            if let Some(channels) = self.prefer_channels {
                if track.channels == channels {
                    best_track = track;
                }
            }

            // Prefer default track
            if track.is_default && !best_track.is_default {
                best_track = track;
            }
        }

        Ok(best_track.clone())
    }

    /// Select track by index.
    pub fn select_by_index<P: AsRef<Path>>(
        &self,
        input: P,
        index: usize,
    ) -> Result<AudioTrackInfo> {
        let tracks = self.list_tracks(input)?;

        tracks
            .into_iter()
            .nth(index)
            .ok_or_else(|| ConversionError::InvalidInput(format!("Track index {index} not found")))
    }

    /// Select track by language.
    pub fn select_by_language<P: AsRef<Path>>(
        &self,
        input: P,
        language: &str,
    ) -> Result<AudioTrackInfo> {
        let tracks = self.list_tracks(input)?;

        tracks
            .into_iter()
            .find(|t| t.language.as_deref() == Some(language))
            .ok_or_else(|| {
                ConversionError::InvalidInput(format!("No track found for language: {language}"))
            })
    }

    /// Prefer stereo tracks.
    #[must_use]
    pub fn prefer_stereo(self) -> Self {
        self.with_channels(2)
    }

    /// Prefer surround sound tracks.
    #[must_use]
    pub fn prefer_surround(self) -> Self {
        self.with_channels(6)
    }
}

impl Default for AudioTrackSelector {
    fn default() -> Self {
        Self::new()
    }
}

/// Information about an audio track.
#[derive(Debug, Clone)]
pub struct AudioTrackInfo {
    /// Track index
    pub index: usize,
    /// Codec name
    pub codec: String,
    /// Language code (e.g., "eng", "spa")
    pub language: Option<String>,
    /// Track title/description
    pub title: Option<String>,
    /// Number of channels
    pub channels: u32,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Bitrate in bits per second
    pub bitrate: Option<u64>,
    /// Whether this is the default track
    pub is_default: bool,
}

impl AudioTrackInfo {
    /// Create a new audio track info.
    #[must_use]
    pub fn new(index: usize, codec: String, channels: u32, sample_rate: u32) -> Self {
        Self {
            index,
            codec,
            language: None,
            title: None,
            channels,
            sample_rate,
            bitrate: None,
            is_default: false,
        }
    }

    /// Check if this is a stereo track.
    #[must_use]
    pub fn is_stereo(&self) -> bool {
        self.channels == 2
    }

    /// Check if this is a mono track.
    #[must_use]
    pub fn is_mono(&self) -> bool {
        self.channels == 1
    }

    /// Check if this is a surround sound track.
    #[must_use]
    pub fn is_surround(&self) -> bool {
        self.channels >= 6
    }

    /// Get the channel layout name.
    #[must_use]
    pub fn channel_layout(&self) -> &'static str {
        match self.channels {
            1 => "mono",
            2 => "stereo",
            6 => "5.1",
            8 => "7.1",
            _ => "unknown",
        }
    }

    /// Get a description of this track.
    #[must_use]
    pub fn description(&self) -> String {
        let mut parts = vec![
            format!("Track {}", self.index),
            format!("{} {}", self.codec, self.channel_layout()),
        ];

        if let Some(lang) = &self.language {
            parts.push(format!("Language: {lang}"));
        }

        if let Some(title) = &self.title {
            parts.push(format!("\"{title}\""));
        }

        if let Some(bitrate) = self.bitrate {
            parts.push(format!("{} kbps", bitrate / 1000));
        }

        if self.is_default {
            parts.push("(default)".to_string());
        }

        parts.join(" - ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_selector_creation() {
        let selector = AudioTrackSelector::new();
        assert!(selector.prefer_language.is_none());
        assert!(selector.prefer_channels.is_none());
    }

    #[test]
    fn test_selector_preferences() {
        let selector = AudioTrackSelector::new()
            .with_language("eng")
            .with_channels(2);

        assert_eq!(selector.prefer_language, Some("eng".to_string()));
        assert_eq!(selector.prefer_channels, Some(2));
    }

    #[test]
    fn test_track_info_creation() {
        let track = AudioTrackInfo::new(0, "aac".to_string(), 2, 48000);

        assert_eq!(track.index, 0);
        assert_eq!(track.codec, "aac");
        assert_eq!(track.channels, 2);
        assert_eq!(track.sample_rate, 48000);
    }

    #[test]
    fn test_track_channel_checks() {
        let stereo = AudioTrackInfo::new(0, "aac".to_string(), 2, 48000);
        assert!(stereo.is_stereo());
        assert!(!stereo.is_mono());
        assert!(!stereo.is_surround());

        let mono = AudioTrackInfo::new(0, "aac".to_string(), 1, 48000);
        assert!(mono.is_mono());
        assert!(!mono.is_stereo());

        let surround = AudioTrackInfo::new(0, "aac".to_string(), 6, 48000);
        assert!(surround.is_surround());
        assert!(!surround.is_stereo());
    }

    #[test]
    fn test_channel_layout() {
        assert_eq!(
            AudioTrackInfo::new(0, "aac".to_string(), 1, 48000).channel_layout(),
            "mono"
        );
        assert_eq!(
            AudioTrackInfo::new(0, "aac".to_string(), 2, 48000).channel_layout(),
            "stereo"
        );
        assert_eq!(
            AudioTrackInfo::new(0, "aac".to_string(), 6, 48000).channel_layout(),
            "5.1"
        );
    }

    #[test]
    fn test_track_description() {
        let track = AudioTrackInfo {
            index: 0,
            codec: "aac".to_string(),
            language: Some("eng".to_string()),
            title: Some("English".to_string()),
            channels: 2,
            sample_rate: 48000,
            bitrate: Some(192_000),
            is_default: true,
        };

        let desc = track.description();
        assert!(desc.contains("Track 0"));
        assert!(desc.contains("aac"));
        assert!(desc.contains("stereo"));
        assert!(desc.contains("eng"));
        assert!(desc.contains("English"));
        assert!(desc.contains("192 kbps"));
        assert!(desc.contains("default"));
    }

    #[test]
    fn test_convenience_methods() {
        let selector = AudioTrackSelector::new().prefer_stereo();
        assert_eq!(selector.prefer_channels, Some(2));

        let selector = AudioTrackSelector::new().prefer_surround();
        assert_eq!(selector.prefer_channels, Some(6));
    }
}
