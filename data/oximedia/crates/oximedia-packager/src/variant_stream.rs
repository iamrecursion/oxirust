#![allow(dead_code)]
//! Variant stream management for multi-bitrate adaptive packaging.
//!
//! This module provides structures for tracking variant streams (renditions)
//! in an adaptive bitrate ladder, including video, audio, and subtitle
//! alternate renditions used in HLS and DASH packaging.

use crate::config::SegmentFormat;
use crate::error::{PackagerError, PackagerResult};

/// Codec identifier for a variant stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamCodec {
    /// AV1 video codec.
    Av1,
    /// VP9 video codec.
    Vp9,
    /// VP8 video codec.
    Vp8,
    /// Opus audio codec.
    Opus,
    /// Vorbis audio codec.
    Vorbis,
    /// FLAC audio codec.
    Flac,
    /// WebVTT subtitle format.
    WebVtt,
}

impl StreamCodec {
    /// MIME codecs string for HLS/DASH manifests.
    #[must_use]
    pub const fn codecs_string(&self) -> &'static str {
        match self {
            Self::Av1 => "av01.0.08M.08",
            Self::Vp9 => "vp09.00.31.08",
            Self::Vp8 => "vp8",
            Self::Opus => "opus",
            Self::Vorbis => "vorbis",
            Self::Flac => "flac",
            Self::WebVtt => "wvtt",
        }
    }

    /// Whether this codec represents video.
    #[must_use]
    pub const fn is_video(&self) -> bool {
        matches!(self, Self::Av1 | Self::Vp9 | Self::Vp8)
    }

    /// Whether this codec represents audio.
    #[must_use]
    pub const fn is_audio(&self) -> bool {
        matches!(self, Self::Opus | Self::Vorbis | Self::Flac)
    }

    /// Whether this codec represents subtitles.
    #[must_use]
    pub const fn is_subtitle(&self) -> bool {
        matches!(self, Self::WebVtt)
    }
}

/// A single variant stream in the adaptive ladder.
#[derive(Debug, Clone)]
pub struct VariantStream {
    /// Unique identifier for this variant.
    pub id: String,
    /// Video codec (if video variant).
    pub video_codec: Option<StreamCodec>,
    /// Audio codec (if audio variant or muxed).
    pub audio_codec: Option<StreamCodec>,
    /// Width in pixels (video only).
    pub width: Option<u32>,
    /// Height in pixels (video only).
    pub height: Option<u32>,
    /// Frame rate (video only).
    pub frame_rate: Option<f64>,
    /// Peak video bitrate in bits/s.
    pub video_bitrate: u64,
    /// Audio bitrate in bits/s.
    pub audio_bitrate: u64,
    /// Segment format for this variant.
    pub segment_format: SegmentFormat,
    /// Language tag (BCP 47) for audio/subtitle variants.
    pub language: Option<String>,
    /// Whether this variant is the default selection.
    pub is_default: bool,
}

impl VariantStream {
    /// Create a new video variant.
    #[must_use]
    pub fn video(id: &str, codec: StreamCodec, width: u32, height: u32, bitrate: u64) -> Self {
        Self {
            id: id.to_string(),
            video_codec: Some(codec),
            audio_codec: None,
            width: Some(width),
            height: Some(height),
            frame_rate: None,
            video_bitrate: bitrate,
            audio_bitrate: 0,
            segment_format: SegmentFormat::Fmp4,
            language: None,
            is_default: false,
        }
    }

    /// Create a new audio-only variant.
    #[must_use]
    pub fn audio(id: &str, codec: StreamCodec, bitrate: u64, language: &str) -> Self {
        Self {
            id: id.to_string(),
            video_codec: None,
            audio_codec: Some(codec),
            width: None,
            height: None,
            frame_rate: None,
            video_bitrate: 0,
            audio_bitrate: bitrate,
            segment_format: SegmentFormat::Fmp4,
            language: Some(language.to_string()),
            is_default: false,
        }
    }

    /// Set this variant as the default.
    #[must_use]
    pub fn as_default(mut self) -> Self {
        self.is_default = true;
        self
    }

    /// Set the frame rate.
    #[must_use]
    pub fn with_frame_rate(mut self, fps: f64) -> Self {
        self.frame_rate = Some(fps);
        self
    }

    /// Total bandwidth for this variant.
    #[must_use]
    pub fn total_bandwidth(&self) -> u64 {
        self.video_bitrate + self.audio_bitrate
    }

    /// Build the combined codecs string for manifests.
    #[must_use]
    pub fn combined_codecs(&self) -> String {
        let mut parts = Vec::new();
        if let Some(vc) = &self.video_codec {
            parts.push(vc.codecs_string().to_string());
        }
        if let Some(ac) = &self.audio_codec {
            parts.push(ac.codecs_string().to_string());
        }
        parts.join(",")
    }

    /// Resolution string (e.g. "1920x1080").
    #[must_use]
    pub fn resolution_string(&self) -> Option<String> {
        match (self.width, self.height) {
            (Some(w), Some(h)) => Some(format!("{w}x{h}")),
            _ => None,
        }
    }

    /// Validate the variant stream.
    ///
    /// # Errors
    ///
    /// Returns an error if required fields are missing.
    pub fn validate(&self) -> PackagerResult<()> {
        if self.id.is_empty() {
            return Err(PackagerError::InvalidConfig(
                "Variant stream ID must not be empty".into(),
            ));
        }
        if self.video_codec.is_none() && self.audio_codec.is_none() {
            return Err(PackagerError::InvalidConfig(
                "Variant must have at least one codec".into(),
            ));
        }
        if self.video_codec.is_some() && (self.width.is_none() || self.height.is_none()) {
            return Err(PackagerError::InvalidConfig(
                "Video variant must specify width and height".into(),
            ));
        }
        Ok(())
    }
}

/// A collection of variant streams forming an adaptive set.
#[derive(Debug, Clone)]
pub struct VariantSet {
    /// The variant streams.
    pub variants: Vec<VariantStream>,
}

impl VariantSet {
    /// Create a new empty variant set.
    #[must_use]
    pub fn new() -> Self {
        Self {
            variants: Vec::new(),
        }
    }

    /// Add a variant stream.
    pub fn add(&mut self, variant: VariantStream) {
        self.variants.push(variant);
    }

    /// Number of variants.
    #[must_use]
    pub fn len(&self) -> usize {
        self.variants.len()
    }

    /// Whether the set is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.variants.is_empty()
    }

    /// Get all video variants, sorted by bandwidth (ascending).
    #[must_use]
    pub fn video_variants(&self) -> Vec<&VariantStream> {
        let mut vids: Vec<&VariantStream> = self
            .variants
            .iter()
            .filter(|v| v.video_codec.is_some())
            .collect();
        vids.sort_by_key(|v| v.video_bitrate);
        vids
    }

    /// Get all audio variants.
    #[must_use]
    pub fn audio_variants(&self) -> Vec<&VariantStream> {
        self.variants
            .iter()
            .filter(|v| v.video_codec.is_none() && v.audio_codec.is_some())
            .collect()
    }

    /// Validate all variants.
    ///
    /// # Errors
    ///
    /// Returns the first validation error encountered.
    pub fn validate(&self) -> PackagerResult<()> {
        if self.variants.is_empty() {
            return Err(PackagerError::InvalidConfig(
                "Variant set must have at least one variant".into(),
            ));
        }
        for v in &self.variants {
            v.validate()?;
        }
        Ok(())
    }
}

impl Default for VariantSet {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_codec_properties() {
        assert!(StreamCodec::Av1.is_video());
        assert!(!StreamCodec::Av1.is_audio());
        assert!(StreamCodec::Opus.is_audio());
        assert!(StreamCodec::WebVtt.is_subtitle());
    }

    #[test]
    fn test_codecs_string() {
        assert_eq!(StreamCodec::Av1.codecs_string(), "av01.0.08M.08");
        assert_eq!(StreamCodec::Opus.codecs_string(), "opus");
    }

    #[test]
    fn test_video_variant_creation() {
        let v = VariantStream::video("v1", StreamCodec::Av1, 1920, 1080, 5_000_000);
        assert_eq!(v.width, Some(1920));
        assert_eq!(v.height, Some(1080));
        assert!(v.validate().is_ok());
    }

    #[test]
    fn test_audio_variant_creation() {
        let v = VariantStream::audio("a1", StreamCodec::Opus, 128_000, "en");
        assert_eq!(v.language, Some("en".to_string()));
        assert!(v.validate().is_ok());
    }

    #[test]
    fn test_variant_total_bandwidth() {
        let mut v = VariantStream::video("v1", StreamCodec::Vp9, 1280, 720, 3_000_000);
        v.audio_bitrate = 128_000;
        assert_eq!(v.total_bandwidth(), 3_128_000);
    }

    #[test]
    fn test_combined_codecs() {
        let mut v = VariantStream::video("v1", StreamCodec::Av1, 1920, 1080, 5_000_000);
        v.audio_codec = Some(StreamCodec::Opus);
        let codecs = v.combined_codecs();
        assert!(codecs.contains("av01"));
        assert!(codecs.contains("opus"));
    }

    #[test]
    fn test_resolution_string() {
        let v = VariantStream::video("v1", StreamCodec::Av1, 1920, 1080, 5_000_000);
        assert_eq!(v.resolution_string(), Some("1920x1080".to_string()));
    }

    #[test]
    fn test_audio_variant_no_resolution() {
        let v = VariantStream::audio("a1", StreamCodec::Opus, 128_000, "en");
        assert_eq!(v.resolution_string(), None);
    }

    #[test]
    fn test_variant_validate_empty_id() {
        let v = VariantStream::video("", StreamCodec::Av1, 1920, 1080, 5_000_000);
        assert!(v.validate().is_err());
    }

    #[test]
    fn test_variant_validate_no_codec() {
        let v = VariantStream {
            id: "x".to_string(),
            video_codec: None,
            audio_codec: None,
            width: None,
            height: None,
            frame_rate: None,
            video_bitrate: 0,
            audio_bitrate: 0,
            segment_format: SegmentFormat::Fmp4,
            language: None,
            is_default: false,
        };
        assert!(v.validate().is_err());
    }

    #[test]
    fn test_variant_set() {
        let mut set = VariantSet::new();
        set.add(VariantStream::video(
            "v1",
            StreamCodec::Av1,
            1920,
            1080,
            5_000_000,
        ));
        set.add(VariantStream::video(
            "v2",
            StreamCodec::Av1,
            1280,
            720,
            3_000_000,
        ));
        assert_eq!(set.len(), 2);
        assert!(set.validate().is_ok());
    }

    #[test]
    fn test_variant_set_video_sorted() {
        let mut set = VariantSet::new();
        set.add(VariantStream::video(
            "hi",
            StreamCodec::Av1,
            1920,
            1080,
            5_000_000,
        ));
        set.add(VariantStream::video(
            "lo",
            StreamCodec::Av1,
            640,
            360,
            500_000,
        ));
        let vids = set.video_variants();
        assert!(vids[0].video_bitrate < vids[1].video_bitrate);
    }

    #[test]
    fn test_variant_set_empty_validation() {
        let set = VariantSet::new();
        assert!(set.validate().is_err());
    }

    #[test]
    fn test_default_variant() {
        let v = VariantStream::video("v1", StreamCodec::Av1, 1920, 1080, 5_000_000).as_default();
        assert!(v.is_default);
    }
}
