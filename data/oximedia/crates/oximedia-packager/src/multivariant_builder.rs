// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Multi-variant playlist generation from [`VariantStream`] / [`VariantSet`].
//!
//! This module bridges the abstract `variant_stream::VariantSet` and
//! `variant_stream::VariantStream` types to the concrete HLS and DASH
//! manifest builders, providing a single entry-point for generating both
//! HLS master playlists and DASH MPD adaptation sets from a shared variant
//! description.
//!
//! # Design
//!
//! ```text
//!                  VariantSet
//!                 /          \
//!   HlsMultivariantBuilder  DashAdaptationSetBuilder
//!           |                          |
//!  HLS M3U8 master playlist    DASH AdaptationSet XML
//! ```
//!
//! # Example — HLS
//!
//! ```
//! use oximedia_packager::multivariant_builder::HlsMultivariantBuilder;
//! use oximedia_packager::variant_stream::{StreamCodec, VariantSet, VariantStream};
//!
//! let mut set = VariantSet::new();
//! set.add(VariantStream::video("1080p", StreamCodec::Av1, 1920, 1080, 5_000_000).as_default());
//! set.add(VariantStream::video("720p", StreamCodec::Av1, 1280, 720, 3_000_000));
//! set.add(VariantStream::audio("audio-en", StreamCodec::Opus, 128_000, "en").as_default());
//!
//! let playlist = HlsMultivariantBuilder::from_variant_set(&set, "segments")
//!     .build()
//!     .expect("should succeed");
//!
//! assert!(playlist.contains("#EXT-X-STREAM-INF"));
//! assert!(playlist.contains("CODECS="));
//! ```
//!
//! # Example — DASH
//!
//! ```
//! use oximedia_packager::multivariant_builder::DashAdaptationSetBuilder;
//! use oximedia_packager::variant_stream::{StreamCodec, VariantSet, VariantStream};
//!
//! let mut set = VariantSet::new();
//! set.add(VariantStream::video("v1", StreamCodec::Vp9, 1920, 1080, 5_000_000));
//! set.add(VariantStream::video("v2", StreamCodec::Vp9, 1280, 720, 2_500_000));
//!
//! let xml = DashAdaptationSetBuilder::from_variant_set(&set, 90_000)
//!     .build()
//!     .expect("should succeed");
//!
//! assert!(xml.contains("AdaptationSet"));
//! assert!(xml.contains("Representation"));
//! ```

use crate::error::{PackagerError, PackagerResult};
use crate::variant_stream::{StreamCodec, VariantSet, VariantStream};

// ---------------------------------------------------------------------------
// Codec string helpers
// ---------------------------------------------------------------------------

/// Convert a [`StreamCodec`] to an RFC-compliant codec string.
#[must_use]
pub fn codec_string(codec: &StreamCodec) -> &'static str {
    codec.codecs_string()
}

/// Build the combined `CODECS` attribute from video and audio codecs in a
/// `VariantStream`, e.g. `"av01.0.08M.08,opus"`.
#[must_use]
pub fn codecs_attr(variant: &VariantStream) -> String {
    let mut parts: Vec<String> = Vec::new();
    if let Some(vc) = &variant.video_codec {
        parts.push(vc.codecs_string().to_string());
    }
    if let Some(ac) = &variant.audio_codec {
        parts.push(ac.codecs_string().to_string());
    }
    parts.join(",")
}

// ---------------------------------------------------------------------------
// HlsMultivariantBuilder
// ---------------------------------------------------------------------------

/// Generates an HLS master (multivariant) playlist from a [`VariantSet`].
///
/// The resulting M3U8 playlist includes:
/// - `#EXT-X-MEDIA` entries for audio renditions.
/// - `#EXT-X-STREAM-INF` entries for video renditions (sorted highest to
///   lowest bandwidth, per Apple's recommendation).
pub struct HlsMultivariantBuilder<'a> {
    set: &'a VariantSet,
    base_uri: String,
    version: u32,
    independent_segments: bool,
    audio_group_id: String,
}

impl<'a> HlsMultivariantBuilder<'a> {
    /// Create a new builder.
    ///
    /// `base_uri` is the prefix for media playlist URIs (e.g. `"segments"` or
    /// `""` for the same directory).
    #[must_use]
    pub fn from_variant_set(set: &'a VariantSet, base_uri: impl Into<String>) -> Self {
        Self {
            set,
            base_uri: base_uri.into(),
            version: 7,
            independent_segments: true,
            audio_group_id: "audio".to_string(),
        }
    }

    /// Set the HLS version (default: 7).
    #[must_use]
    pub fn with_version(mut self, v: u32) -> Self {
        self.version = v;
        self
    }

    /// Set the audio group ID used in `EXT-X-MEDIA` and `AUDIO` attributes.
    #[must_use]
    pub fn with_audio_group_id(mut self, id: impl Into<String>) -> Self {
        self.audio_group_id = id.into();
        self
    }

    /// Whether to emit `#EXT-X-INDEPENDENT-SEGMENTS` (default: `true`).
    #[must_use]
    pub fn independent_segments(mut self, enabled: bool) -> Self {
        self.independent_segments = enabled;
        self
    }

    /// Build the master playlist as an M3U8 string.
    ///
    /// # Errors
    ///
    /// Returns an error if the variant set fails validation or if a codec is
    /// unsupported.
    pub fn build(&self) -> PackagerResult<String> {
        self.set.validate()?;

        let mut out = String::new();
        out.push_str("#EXTM3U\n");
        out.push_str(&format!("#EXT-X-VERSION:{}\n", self.version));

        if self.independent_segments {
            out.push_str("#EXT-X-INDEPENDENT-SEGMENTS\n");
        }

        // --- EXT-X-MEDIA for audio renditions --------------------------------
        let audio_variants = self.set.audio_variants();
        let has_audio_group = !audio_variants.is_empty();

        for av in &audio_variants {
            self.write_ext_x_media(&mut out, av, &self.audio_group_id)?;
        }

        // --- EXT-X-STREAM-INF for video variants (highest → lowest bw) ------
        let mut video_variants = self.set.video_variants();
        video_variants.sort_by(|a, b| b.video_bitrate.cmp(&a.video_bitrate));

        for vv in &video_variants {
            self.write_stream_inf(&mut out, vv, has_audio_group)?;
        }

        Ok(out)
    }

    fn media_playlist_uri(&self, variant: &VariantStream) -> String {
        if self.base_uri.is_empty() {
            format!("{}.m3u8", variant.id)
        } else {
            format!("{}/{}.m3u8", self.base_uri, variant.id)
        }
    }

    fn write_ext_x_media(
        &self,
        out: &mut String,
        av: &VariantStream,
        group_id: &str,
    ) -> PackagerResult<()> {
        let codec = av.audio_codec.as_ref().ok_or_else(|| {
            PackagerError::InvalidConfig("audio variant has no audio codec".into())
        })?;

        let lang = av.language.as_deref().unwrap_or("und");
        let is_default = if av.is_default { "YES" } else { "NO" };
        let uri = self.media_playlist_uri(av);

        out.push_str(&format!(
            "#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID=\"{group_id}\",NAME=\"{name}\",LANGUAGE=\"{lang}\",CODECS=\"{codecs}\",DEFAULT={is_default},AUTOSELECT=YES,URI=\"{uri}\"\n",
            name = av.id,
            codecs = codec.codecs_string(),
        ));
        Ok(())
    }

    fn write_stream_inf(
        &self,
        out: &mut String,
        vv: &VariantStream,
        has_audio_group: bool,
    ) -> PackagerResult<()> {
        let codecs = codecs_attr(vv);
        let bandwidth = vv.total_bandwidth();

        let mut attrs = format!("BANDWIDTH={bandwidth},CODECS=\"{codecs}\"");

        if let (Some(w), Some(h)) = (vv.width, vv.height) {
            attrs.push_str(&format!(",RESOLUTION={w}x{h}"));
        }

        if let Some(fps) = vv.frame_rate {
            attrs.push_str(&format!(",FRAME-RATE={fps:.3}"));
        }

        if has_audio_group {
            attrs.push_str(&format!(",AUDIO=\"{}\"", self.audio_group_id));
        }

        out.push_str(&format!("#EXT-X-STREAM-INF:{attrs}\n"));
        out.push_str(&self.media_playlist_uri(vv));
        out.push('\n');

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// DashAdaptationSetBuilder
// ---------------------------------------------------------------------------

/// Generates a DASH `<AdaptationSet>` XML fragment from a [`VariantSet`].
///
/// Only video variants are included; call this separately for video and audio
/// adaptation sets.
pub struct DashAdaptationSetBuilder<'a> {
    set: &'a VariantSet,
    timescale: u32,
    segment_template: Option<DashSegmentTemplate>,
    adaptation_set_id: u32,
    content_type: DashContentType,
}

/// Content type for a DASH adaptation set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DashContentType {
    /// Video adaptation set.
    Video,
    /// Audio adaptation set.
    Audio,
}

impl DashContentType {
    /// Returns the `contentType` string for the MPD.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Video => "video",
            Self::Audio => "audio",
        }
    }

    /// Returns a MIME type string for this content type.
    #[must_use]
    pub fn mime_type(self) -> &'static str {
        match self {
            Self::Video => "video/mp4",
            Self::Audio => "audio/mp4",
        }
    }
}

/// Segment template for DASH manifest.
#[derive(Debug, Clone)]
pub struct DashSegmentTemplate {
    /// Initialization segment URL template.
    pub initialization: String,
    /// Media segment URL template.
    pub media: String,
    /// Start number.
    pub start_number: u32,
    /// Segment duration in timescale ticks.
    pub duration: u64,
}

impl DashSegmentTemplate {
    /// Create a new segment template.
    #[must_use]
    pub fn new(
        initialization: impl Into<String>,
        media: impl Into<String>,
        start_number: u32,
        duration: u64,
    ) -> Self {
        Self {
            initialization: initialization.into(),
            media: media.into(),
            start_number,
            duration,
        }
    }
}

impl<'a> DashAdaptationSetBuilder<'a> {
    /// Create a new builder for **video** variants from the set.
    ///
    /// `timescale` is the media timescale (ticks per second).
    #[must_use]
    pub fn from_variant_set(set: &'a VariantSet, timescale: u32) -> Self {
        Self {
            set,
            timescale,
            segment_template: None,
            adaptation_set_id: 1,
            content_type: DashContentType::Video,
        }
    }

    /// Override the content type (default: video).
    #[must_use]
    pub fn content_type(mut self, ct: DashContentType) -> Self {
        self.content_type = ct;
        self
    }

    /// Set the adaptation set ID.
    #[must_use]
    pub fn with_id(mut self, id: u32) -> Self {
        self.adaptation_set_id = id;
        self
    }

    /// Set a segment template.
    #[must_use]
    pub fn with_segment_template(mut self, tmpl: DashSegmentTemplate) -> Self {
        self.segment_template = Some(tmpl);
        self
    }

    /// Build the `<AdaptationSet>` XML string.
    ///
    /// # Errors
    ///
    /// Returns an error if the variant set is invalid or contains no
    /// applicable variants.
    pub fn build(&self) -> PackagerResult<String> {
        self.set.validate()?;

        let variants: Vec<&VariantStream> = match self.content_type {
            DashContentType::Video => self.set.video_variants(),
            DashContentType::Audio => self.set.audio_variants(),
        };

        if variants.is_empty() {
            return Err(PackagerError::InvalidConfig(format!(
                "no {} variants in the VariantSet",
                self.content_type.as_str()
            )));
        }

        let mut out = String::new();
        out.push_str(&format!(
            "<AdaptationSet id=\"{}\" contentType=\"{}\" mimeType=\"{}\">",
            self.adaptation_set_id,
            self.content_type.as_str(),
            self.content_type.mime_type(),
        ));

        // Segment template (shared)
        if let Some(tmpl) = &self.segment_template {
            out.push_str(&format!(
                "<SegmentTemplate initialization=\"{}\" media=\"{}\" startNumber=\"{}\" timescale=\"{}\" duration=\"{}\"/>",
                tmpl.initialization,
                tmpl.media,
                tmpl.start_number,
                self.timescale,
                tmpl.duration,
            ));
        }

        // Representations — sorted by bandwidth ascending
        let mut sorted = variants;
        sorted.sort_by_key(|v: &&VariantStream| v.total_bandwidth());

        for v in sorted {
            self.write_representation(&mut out, v)?;
        }

        out.push_str("</AdaptationSet>");
        Ok(out)
    }

    fn write_representation(&self, out: &mut String, v: &VariantStream) -> PackagerResult<()> {
        let codecs = codecs_attr(v);
        let bandwidth = v.total_bandwidth();

        let mut attrs = format!(
            "id=\"{}\" bandwidth=\"{bandwidth}\" codecs=\"{codecs}\"",
            v.id,
        );

        if let (Some(w), Some(h)) = (v.width, v.height) {
            attrs.push_str(&format!(" width=\"{w}\" height=\"{h}\""));
        }

        if let Some(fps) = v.frame_rate {
            attrs.push_str(&format!(" frameRate=\"{fps:.3}\""));
        }

        if let Some(lang) = &v.language {
            attrs.push_str(&format!(" lang=\"{lang}\""));
        }

        out.push_str(&format!("<Representation {attrs}/>"));
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// MultivariantPlaylistBuilder
// ---------------------------------------------------------------------------

/// Convenience builder that generates an HLS master (multi-variant) playlist
/// directly from a [`PackagerConfig`](crate::config::PackagerConfig) that
/// carries an explicit [`VariantSet`].
///
/// This is a thin wrapper around [`HlsMultivariantBuilder`] that removes the
/// need to manually extract the `VariantSet` from the config.
///
/// # Example
///
/// ```
/// use oximedia_packager::config::PackagerConfig;
/// use oximedia_packager::multivariant_builder::MultivariantPlaylistBuilder;
/// use oximedia_packager::variant_stream::{StreamCodec, VariantSet, VariantStream};
///
/// let mut vs = VariantSet::new();
/// vs.add(VariantStream::video("1080p", StreamCodec::Av1, 1920, 1080, 5_000_000));
/// vs.add(VariantStream::video("720p",  StreamCodec::Av1, 1280,  720, 3_000_000));
///
/// let config = PackagerConfig::new().with_variant_set(vs);
/// let playlist = MultivariantPlaylistBuilder::new(&config, "segments")
///     .build_hls()
///     .expect("should succeed");
///
/// assert!(playlist.contains("#EXT-X-STREAM-INF"));
/// assert!(playlist.contains("BANDWIDTH=5000000"));
/// ```
pub struct MultivariantPlaylistBuilder<'a> {
    config: &'a crate::config::PackagerConfig,
    base_uri: String,
}

impl<'a> MultivariantPlaylistBuilder<'a> {
    /// Create a new builder backed by the given packager config.
    ///
    /// `base_uri` is the prefix applied to per-variant playlist URIs
    /// (e.g. `"segments"` or `""` for the same directory).
    #[must_use]
    pub fn new(config: &'a crate::config::PackagerConfig, base_uri: impl Into<String>) -> Self {
        Self {
            config,
            base_uri: base_uri.into(),
        }
    }

    /// Build the HLS master playlist string.
    ///
    /// # Errors
    ///
    /// Returns [`PackagerError::InvalidConfig`] if:
    /// - `config.variant_set` is `None`.
    /// - The variant set fails its own validation.
    pub fn build_hls(&self) -> PackagerResult<String> {
        let vs = self.config.variant_set.as_ref().ok_or_else(|| {
            PackagerError::InvalidConfig(
                "no variant_set in PackagerConfig; call with_variant_set() first".into(),
            )
        })?;

        HlsMultivariantBuilder::from_variant_set(vs, &self.base_uri).build()
    }
}

// ---------------------------------------------------------------------------
// Conversion helpers
// ---------------------------------------------------------------------------

/// Convert a [`VariantSet`] to a sorted list of (bandwidth, codec_string,
/// resolution) tuples, useful for testing.
#[must_use]
pub fn variant_summary(set: &VariantSet) -> Vec<(u64, String, Option<String>)> {
    let mut result: Vec<(u64, String, Option<String>)> = set
        .variants
        .iter()
        .map(|v: &VariantStream| (v.total_bandwidth(), codecs_attr(v), v.resolution_string()))
        .collect();
    result.sort_by_key(|(bw, _, _)| *bw);
    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::variant_stream::{StreamCodec, VariantSet, VariantStream};

    fn make_av1_set() -> VariantSet {
        let mut set = VariantSet::new();
        set.add(
            VariantStream::video("1080p", StreamCodec::Av1, 1920, 1080, 5_000_000).as_default(),
        );
        set.add(VariantStream::video(
            "720p",
            StreamCodec::Av1,
            1280,
            720,
            3_000_000,
        ));
        set.add(VariantStream::video(
            "480p",
            StreamCodec::Av1,
            854,
            480,
            1_500_000,
        ));
        set
    }

    fn make_set_with_audio() -> VariantSet {
        let mut set = make_av1_set();
        set.add(VariantStream::audio("audio-en", StreamCodec::Opus, 128_000, "en").as_default());
        set.add(VariantStream::audio(
            "audio-fr",
            StreamCodec::Opus,
            128_000,
            "fr",
        ));
        set
    }

    // --- codec_string / codecs_attr -----------------------------------------

    #[test]
    fn test_codec_string_av1() {
        assert_eq!(codec_string(&StreamCodec::Av1), "av01.0.08M.08");
    }

    #[test]
    fn test_codec_string_opus() {
        assert_eq!(codec_string(&StreamCodec::Opus), "opus");
    }

    #[test]
    fn test_codecs_attr_video_only() {
        let v = VariantStream::video("v1", StreamCodec::Vp9, 1280, 720, 3_000_000);
        let attr = codecs_attr(&v);
        assert_eq!(attr, "vp09.00.31.08");
    }

    #[test]
    fn test_codecs_attr_video_and_audio() {
        let mut v = VariantStream::video("v1", StreamCodec::Av1, 1920, 1080, 5_000_000);
        v.audio_codec = Some(StreamCodec::Opus);
        let attr = codecs_attr(&v);
        assert!(attr.contains("av01"));
        assert!(attr.contains("opus"));
        assert!(attr.contains(','));
    }

    #[test]
    fn test_codecs_attr_audio_only() {
        let a = VariantStream::audio("a1", StreamCodec::Opus, 128_000, "en");
        let attr = codecs_attr(&a);
        assert_eq!(attr, "opus");
    }

    // --- HlsMultivariantBuilder ---------------------------------------------

    #[test]
    fn test_hls_builder_header() {
        let set = make_av1_set();
        let playlist = HlsMultivariantBuilder::from_variant_set(&set, "segs")
            .build()
            .expect("should succeed");

        assert!(playlist.starts_with("#EXTM3U\n"));
        assert!(playlist.contains("#EXT-X-VERSION:7"));
        assert!(playlist.contains("#EXT-X-INDEPENDENT-SEGMENTS"));
    }

    #[test]
    fn test_hls_builder_stream_inf_entries() {
        let set = make_av1_set();
        let playlist = HlsMultivariantBuilder::from_variant_set(&set, "segs")
            .build()
            .expect("should succeed");

        // Three video variants → three EXT-X-STREAM-INF entries
        assert_eq!(playlist.matches("#EXT-X-STREAM-INF").count(), 3);
    }

    #[test]
    fn test_hls_builder_sorted_highest_to_lowest() {
        let set = make_av1_set();
        let playlist = HlsMultivariantBuilder::from_variant_set(&set, "segs")
            .build()
            .expect("should succeed");

        // Find BANDWIDTH values in order
        let bandwidths: Vec<u64> = playlist
            .lines()
            .filter(|l| l.starts_with("#EXT-X-STREAM-INF"))
            .filter_map(|l| {
                l.split("BANDWIDTH=")
                    .nth(1)
                    .and_then(|s| s.split(',').next())
                    .and_then(|s| s.parse::<u64>().ok())
            })
            .collect();

        // Should be sorted highest → lowest
        assert!(bandwidths.windows(2).all(|w| w[0] >= w[1]));
    }

    #[test]
    fn test_hls_builder_resolution_in_stream_inf() {
        let set = make_av1_set();
        let playlist = HlsMultivariantBuilder::from_variant_set(&set, "segs")
            .build()
            .expect("should succeed");

        assert!(playlist.contains("RESOLUTION=1920x1080"));
        assert!(playlist.contains("RESOLUTION=1280x720"));
        assert!(playlist.contains("RESOLUTION=854x480"));
    }

    #[test]
    fn test_hls_builder_codec_in_stream_inf() {
        let set = make_av1_set();
        let playlist = HlsMultivariantBuilder::from_variant_set(&set, "segs")
            .build()
            .expect("should succeed");

        assert!(playlist.contains("CODECS=\"av01"));
    }

    #[test]
    fn test_hls_builder_uris() {
        let set = make_av1_set();
        let playlist = HlsMultivariantBuilder::from_variant_set(&set, "segs")
            .build()
            .expect("should succeed");

        assert!(playlist.contains("segs/1080p.m3u8"));
        assert!(playlist.contains("segs/720p.m3u8"));
        assert!(playlist.contains("segs/480p.m3u8"));
    }

    #[test]
    fn test_hls_builder_empty_base_uri() {
        let set = make_av1_set();
        let playlist = HlsMultivariantBuilder::from_variant_set(&set, "")
            .build()
            .expect("should succeed");

        assert!(playlist.contains("1080p.m3u8"));
    }

    #[test]
    fn test_hls_builder_with_audio_group() {
        let set = make_set_with_audio();
        let playlist = HlsMultivariantBuilder::from_variant_set(&set, "segs")
            .build()
            .expect("should succeed");

        // Audio EXT-X-MEDIA entries
        assert!(playlist.contains("EXT-X-MEDIA:TYPE=AUDIO"));
        assert!(playlist.contains("GROUP-ID=\"audio\""));
        assert!(playlist.contains("LANGUAGE=\"en\""));
        assert!(playlist.contains("LANGUAGE=\"fr\""));

        // Video streams reference the audio group
        assert!(playlist.contains("AUDIO=\"audio\""));
    }

    #[test]
    fn test_hls_builder_custom_audio_group_id() {
        let set = make_set_with_audio();
        let playlist = HlsMultivariantBuilder::from_variant_set(&set, "segs")
            .with_audio_group_id("primary-audio")
            .build()
            .expect("should succeed");

        assert!(playlist.contains("GROUP-ID=\"primary-audio\""));
        assert!(playlist.contains("AUDIO=\"primary-audio\""));
    }

    #[test]
    fn test_hls_builder_version_override() {
        let set = make_av1_set();
        let playlist = HlsMultivariantBuilder::from_variant_set(&set, "segs")
            .with_version(6)
            .build()
            .expect("should succeed");

        assert!(playlist.contains("#EXT-X-VERSION:6"));
    }

    #[test]
    fn test_hls_builder_no_independent_segments() {
        let set = make_av1_set();
        let playlist = HlsMultivariantBuilder::from_variant_set(&set, "segs")
            .independent_segments(false)
            .build()
            .expect("should succeed");

        assert!(!playlist.contains("#EXT-X-INDEPENDENT-SEGMENTS"));
    }

    #[test]
    fn test_hls_builder_frame_rate_emitted() {
        let mut set = VariantSet::new();
        set.add(
            VariantStream::video("v1", StreamCodec::Av1, 1920, 1080, 5_000_000)
                .with_frame_rate(29.97),
        );
        let playlist = HlsMultivariantBuilder::from_variant_set(&set, "segs")
            .build()
            .expect("should succeed");
        assert!(playlist.contains("FRAME-RATE=29.970"));
    }

    // --- DashAdaptationSetBuilder -------------------------------------------

    #[test]
    fn test_dash_builder_adaptation_set_element() {
        let set = make_av1_set();
        let xml = DashAdaptationSetBuilder::from_variant_set(&set, 90_000)
            .build()
            .expect("should succeed");

        assert!(xml.starts_with("<AdaptationSet"));
        assert!(xml.ends_with("</AdaptationSet>"));
        assert!(xml.contains("contentType=\"video\""));
        assert!(xml.contains("mimeType=\"video/mp4\""));
    }

    #[test]
    fn test_dash_builder_representation_count() {
        let set = make_av1_set();
        let xml = DashAdaptationSetBuilder::from_variant_set(&set, 90_000)
            .build()
            .expect("should succeed");

        assert_eq!(xml.matches("<Representation").count(), 3);
    }

    #[test]
    fn test_dash_builder_representation_bandwidth() {
        let set = make_av1_set();
        let xml = DashAdaptationSetBuilder::from_variant_set(&set, 90_000)
            .build()
            .expect("should succeed");

        assert!(xml.contains("bandwidth=\"5000000\""));
        assert!(xml.contains("bandwidth=\"3000000\""));
        assert!(xml.contains("bandwidth=\"1500000\""));
    }

    #[test]
    fn test_dash_builder_representation_dimensions() {
        let set = make_av1_set();
        let xml = DashAdaptationSetBuilder::from_variant_set(&set, 90_000)
            .build()
            .expect("should succeed");

        assert!(xml.contains("width=\"1920\" height=\"1080\""));
        assert!(xml.contains("width=\"1280\" height=\"720\""));
    }

    #[test]
    fn test_dash_builder_codecs() {
        let set = make_av1_set();
        let xml = DashAdaptationSetBuilder::from_variant_set(&set, 90_000)
            .build()
            .expect("should succeed");

        assert!(xml.contains("codecs=\"av01"));
    }

    #[test]
    fn test_dash_builder_with_segment_template() {
        let set = make_av1_set();
        let tmpl = DashSegmentTemplate::new(
            "init-$RepresentationID$.mp4",
            "seg-$Number$.mp4",
            1,
            540_000,
        );
        let xml = DashAdaptationSetBuilder::from_variant_set(&set, 90_000)
            .with_segment_template(tmpl)
            .build()
            .expect("should succeed");

        assert!(xml.contains("SegmentTemplate"));
        assert!(xml.contains("init-$RepresentationID$.mp4"));
        assert!(xml.contains("timescale=\"90000\""));
    }

    #[test]
    fn test_dash_builder_audio_content_type() {
        let mut set = VariantSet::new();
        // Need at least one audio variant in the set to pass validate()
        // BUT validate() only checks there's at least one variant total.
        set.add(VariantStream::audio("a1", StreamCodec::Opus, 128_000, "en"));
        let xml = DashAdaptationSetBuilder::from_variant_set(&set, 48_000)
            .content_type(DashContentType::Audio)
            .build()
            .expect("should succeed");

        assert!(xml.contains("contentType=\"audio\""));
        assert!(xml.contains("mimeType=\"audio/mp4\""));
    }

    #[test]
    fn test_dash_builder_no_video_variants_in_audio_set_fails() {
        // A video-only set asking for audio should fail
        let set = make_av1_set();
        let result = DashAdaptationSetBuilder::from_variant_set(&set, 48_000)
            .content_type(DashContentType::Audio)
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_dash_builder_representation_id() {
        let set = make_av1_set();
        let xml = DashAdaptationSetBuilder::from_variant_set(&set, 90_000)
            .build()
            .expect("should succeed");

        assert!(xml.contains("id=\"1080p\""));
        assert!(xml.contains("id=\"720p\""));
        assert!(xml.contains("id=\"480p\""));
    }

    #[test]
    fn test_dash_builder_custom_id() {
        let set = make_av1_set();
        let xml = DashAdaptationSetBuilder::from_variant_set(&set, 90_000)
            .with_id(2)
            .build()
            .expect("should succeed");

        assert!(xml.contains("id=\"2\""));
    }

    // --- variant_summary ----------------------------------------------------

    #[test]
    fn test_variant_summary_sorted_by_bandwidth() {
        let set = make_av1_set();
        let summary = variant_summary(&set);
        // All video variants
        let bws: Vec<u64> = summary.iter().map(|(bw, _, _)| *bw).collect();
        assert!(bws.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn test_variant_summary_resolution() {
        let set = make_av1_set();
        let summary = variant_summary(&set);
        let resolutions: Vec<Option<String>> = summary.into_iter().map(|(_, _, r)| r).collect();
        assert!(resolutions
            .iter()
            .any(|r| r.as_deref() == Some("1920x1080")));
    }

    // --- DashContentType ----------------------------------------------------

    #[test]
    fn test_content_type_video_str() {
        assert_eq!(DashContentType::Video.as_str(), "video");
        assert_eq!(DashContentType::Video.mime_type(), "video/mp4");
    }

    #[test]
    fn test_content_type_audio_str() {
        assert_eq!(DashContentType::Audio.as_str(), "audio");
        assert_eq!(DashContentType::Audio.mime_type(), "audio/mp4");
    }

    // --- MultivariantPlaylistBuilder ----------------------------------------

    use crate::config::PackagerConfig;

    fn make_config_with_variants() -> PackagerConfig {
        let mut vs = VariantSet::new();
        vs.add(VariantStream::video(
            "1080p",
            StreamCodec::Av1,
            1920,
            1080,
            5_000_000,
        ));
        vs.add(VariantStream::video(
            "720p",
            StreamCodec::Av1,
            1280,
            720,
            3_000_000,
        ));
        vs.add(VariantStream::video(
            "480p",
            StreamCodec::Av1,
            854,
            480,
            1_500_000,
        ));
        PackagerConfig::new().with_variant_set(vs)
    }

    #[test]
    fn test_multivariant_builder_no_variant_set_returns_error() {
        let config = PackagerConfig::new(); // no variant_set
        let result = MultivariantPlaylistBuilder::new(&config, "segs").build_hls();
        assert!(result.is_err(), "should error when variant_set is None");
    }

    #[test]
    fn test_multivariant_builder_three_stream_inf_entries() {
        let config = make_config_with_variants();
        let playlist = MultivariantPlaylistBuilder::new(&config, "segs")
            .build_hls()
            .expect("should succeed");

        assert_eq!(
            playlist.matches("#EXT-X-STREAM-INF").count(),
            3,
            "should have 3 EXT-X-STREAM-INF entries"
        );
    }

    #[test]
    fn test_multivariant_builder_bandwidth_values() {
        let config = make_config_with_variants();
        let playlist = MultivariantPlaylistBuilder::new(&config, "segs")
            .build_hls()
            .expect("should succeed");

        assert!(playlist.contains("BANDWIDTH=5000000"));
        assert!(playlist.contains("BANDWIDTH=3000000"));
        assert!(playlist.contains("BANDWIDTH=1500000"));
    }

    #[test]
    fn test_multivariant_builder_resolution_attributes() {
        let config = make_config_with_variants();
        let playlist = MultivariantPlaylistBuilder::new(&config, "segs")
            .build_hls()
            .expect("should succeed");

        assert!(playlist.contains("RESOLUTION=1920x1080"));
        assert!(playlist.contains("RESOLUTION=1280x720"));
        assert!(playlist.contains("RESOLUTION=854x480"));
    }

    #[test]
    fn test_multivariant_builder_codecs_attribute() {
        let config = make_config_with_variants();
        let playlist = MultivariantPlaylistBuilder::new(&config, "segs")
            .build_hls()
            .expect("should succeed");

        assert!(
            playlist.contains("CODECS=\"av01"),
            "CODECS attr should reference av1"
        );
    }

    #[test]
    fn test_multivariant_builder_uri_pattern() {
        let config = make_config_with_variants();
        let playlist = MultivariantPlaylistBuilder::new(&config, "segments")
            .build_hls()
            .expect("should succeed");

        assert!(playlist.contains("segments/1080p.m3u8"));
        assert!(playlist.contains("segments/720p.m3u8"));
        assert!(playlist.contains("segments/480p.m3u8"));
    }

    #[test]
    fn test_multivariant_builder_extm3u_header() {
        let config = make_config_with_variants();
        let playlist = MultivariantPlaylistBuilder::new(&config, "")
            .build_hls()
            .expect("should succeed");

        assert!(playlist.starts_with("#EXTM3U\n"), "must start with EXTM3U");
    }
}
