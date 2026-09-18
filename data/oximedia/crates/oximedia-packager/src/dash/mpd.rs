//! DASH MPD (Media Presentation Description) generation.

use crate::config::BitrateEntry;
use crate::error::{PackagerError, PackagerResult};
use crate::manifest::{CodecStringBuilder, DurationFormatter};
use chrono::Utc;
use quick_xml::events::{BytesDecl, BytesEnd, BytesStart, BytesText, Event};
use quick_xml::Writer;
use std::io::Cursor;
use std::time::Duration;

/// MPD type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MpdType {
    /// Static MPD (VOD).
    Static,
    /// Dynamic MPD (live).
    Dynamic,
}

impl MpdType {
    /// Convert to string for MPD.
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::Static => "static",
            Self::Dynamic => "dynamic",
        }
    }
}

/// DASH profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DashProfile {
    /// Live profile.
    Live,
    /// On-demand profile.
    OnDemand,
    /// Main profile.
    Main,
}

impl DashProfile {
    /// Convert to URN string.
    #[must_use]
    pub fn as_urn(&self) -> &str {
        match self {
            Self::Live => "urn:mpeg:dash:profile:isoff-live:2011",
            Self::OnDemand => "urn:mpeg:dash:profile:isoff-on-demand:2011",
            Self::Main => "urn:mpeg:dash:profile:isoff-main:2011",
        }
    }
}

/// Representation in an adaptation set.
#[derive(Debug, Clone)]
pub struct Representation {
    /// Representation ID.
    pub id: String,
    /// Bandwidth in bits per second.
    pub bandwidth: u32,
    /// Codec string.
    pub codecs: String,
    /// Width (for video).
    pub width: Option<u32>,
    /// Height (for video).
    pub height: Option<u32>,
    /// Frame rate (for video).
    pub frame_rate: Option<String>,
    /// Audio sampling rate.
    pub audio_sampling_rate: Option<u32>,
    /// Segment template.
    pub segment_template: Option<SegmentTemplate>,
    /// Base URL.
    pub base_url: Option<String>,
}

impl Representation {
    /// Create a new representation.
    #[must_use]
    pub fn new(id: String, bandwidth: u32, codecs: String) -> Self {
        Self {
            id,
            bandwidth,
            codecs,
            width: None,
            height: None,
            frame_rate: None,
            audio_sampling_rate: None,
            segment_template: None,
            base_url: None,
        }
    }

    /// Set dimensions.
    #[must_use]
    pub fn with_dimensions(mut self, width: u32, height: u32) -> Self {
        self.width = Some(width);
        self.height = Some(height);
        self
    }

    /// Set frame rate.
    #[must_use]
    pub fn with_frame_rate(mut self, fps: f64) -> Self {
        self.frame_rate = Some(format!("{fps:.3}"));
        self
    }

    /// Set segment template.
    #[must_use]
    pub fn with_segment_template(mut self, template: SegmentTemplate) -> Self {
        self.segment_template = Some(template);
        self
    }
}

/// Segment template for DASH.
#[derive(Debug, Clone)]
pub struct SegmentTemplate {
    /// Initialization segment URL template.
    pub initialization: String,
    /// Media segment URL template.
    pub media: String,
    /// Segment duration (in timescale units).
    pub duration: u64,
    /// Timescale.
    pub timescale: u32,
    /// Start number.
    pub start_number: u32,
}

impl SegmentTemplate {
    /// Create a new segment template.
    #[must_use]
    pub fn new(initialization: String, media: String, duration: u64, timescale: u32) -> Self {
        Self {
            initialization,
            media,
            duration,
            timescale,
            start_number: 1,
        }
    }

    /// Set start number.
    #[must_use]
    pub fn with_start_number(mut self, start: u32) -> Self {
        self.start_number = start;
        self
    }
}

/// Adaptation set in MPD.
#[derive(Debug, Clone)]
pub struct AdaptationSet {
    /// Adaptation set ID.
    pub id: u32,
    /// Content type (video, audio, text).
    pub content_type: String,
    /// MIME type.
    pub mime_type: String,
    /// Segment alignment flag.
    pub segment_alignment: bool,
    /// Representations in this set.
    pub representations: Vec<Representation>,
    /// Language (for audio/subtitle).
    pub lang: Option<String>,
}

impl AdaptationSet {
    /// Create a new adaptation set.
    #[must_use]
    pub fn new(id: u32, content_type: String, mime_type: String) -> Self {
        Self {
            id,
            content_type,
            mime_type,
            segment_alignment: true,
            representations: Vec::new(),
            lang: None,
        }
    }

    /// Add a representation.
    pub fn add_representation(&mut self, representation: Representation) {
        self.representations.push(representation);
    }

    /// Set language.
    #[must_use]
    pub fn with_language(mut self, lang: String) -> Self {
        self.lang = Some(lang);
        self
    }
}

/// Period in MPD.
#[derive(Debug, Clone)]
pub struct Period {
    /// Period ID.
    pub id: String,
    /// Duration.
    pub duration: Option<Duration>,
    /// Adaptation sets.
    pub adaptation_sets: Vec<AdaptationSet>,
}

impl Period {
    /// Create a new period.
    #[must_use]
    pub fn new(id: String) -> Self {
        Self {
            id,
            duration: None,
            adaptation_sets: Vec::new(),
        }
    }

    /// Set duration.
    #[must_use]
    pub fn with_duration(mut self, duration: Duration) -> Self {
        self.duration = Some(duration);
        self
    }

    /// Add an adaptation set.
    pub fn add_adaptation_set(&mut self, set: AdaptationSet) {
        self.adaptation_sets.push(set);
    }
}

/// MPD builder for generating DASH manifests.
pub struct MpdBuilder {
    mpd_type: MpdType,
    profile: DashProfile,
    min_buffer_time: Duration,
    media_presentation_duration: Option<Duration>,
    periods: Vec<Period>,
    base_url: Option<String>,
}

impl MpdBuilder {
    /// Create a new MPD builder.
    #[must_use]
    pub fn new(mpd_type: MpdType, profile: DashProfile) -> Self {
        Self {
            mpd_type,
            profile,
            min_buffer_time: Duration::from_secs(2),
            media_presentation_duration: None,
            periods: Vec::new(),
            base_url: None,
        }
    }

    /// Set minimum buffer time.
    #[must_use]
    pub fn with_min_buffer_time(mut self, duration: Duration) -> Self {
        self.min_buffer_time = duration;
        self
    }

    /// Set media presentation duration.
    #[must_use]
    pub fn with_duration(mut self, duration: Duration) -> Self {
        self.media_presentation_duration = Some(duration);
        self
    }

    /// Set base URL.
    #[must_use]
    pub fn with_base_url(mut self, url: String) -> Self {
        self.base_url = Some(url);
        self
    }

    /// Add a period.
    pub fn add_period(&mut self, period: Period) {
        self.periods.push(period);
    }

    /// Build the MPD XML.
    pub fn build(&self) -> PackagerResult<String> {
        let mut writer = Writer::new_with_indent(Cursor::new(Vec::new()), b' ', 2);

        // XML declaration
        writer
            .write_event(Event::Decl(BytesDecl::new("1.0", Some("UTF-8"), None)))
            .map_err(|e| PackagerError::manifest_failed(format!("XML write error: {e}")))?;

        // MPD element
        let mut mpd_elem = BytesStart::new("MPD");
        mpd_elem.push_attribute(("xmlns", "urn:mpeg:dash:schema:mpd:2011"));
        mpd_elem.push_attribute(("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance"));
        mpd_elem.push_attribute((
            "xsi:schemaLocation",
            "urn:mpeg:dash:schema:mpd:2011 DASH-MPD.xsd",
        ));
        mpd_elem.push_attribute(("type", self.mpd_type.as_str()));
        mpd_elem.push_attribute(("profiles", self.profile.as_urn()));

        // Min buffer time
        let min_buffer = DurationFormatter::format_iso8601_duration(self.min_buffer_time);
        mpd_elem.push_attribute(("minBufferTime", min_buffer.as_str()));

        // Media presentation duration (for static MPD)
        if let Some(duration) = self.media_presentation_duration {
            let duration_str = DurationFormatter::format_iso8601_duration(duration);
            mpd_elem.push_attribute(("mediaPresentationDuration", duration_str.as_str()));
        }

        // Publish time (for dynamic MPD)
        if self.mpd_type == MpdType::Dynamic {
            let now = Utc::now().to_rfc3339();
            mpd_elem.push_attribute(("publishTime", now.as_str()));
        }

        writer
            .write_event(Event::Start(mpd_elem))
            .map_err(|e| PackagerError::manifest_failed(format!("XML write error: {e}")))?;

        // Base URL
        if let Some(base) = &self.base_url {
            writer
                .write_event(Event::Start(BytesStart::new("BaseURL")))
                .map_err(|e| PackagerError::manifest_failed(format!("XML write error: {e}")))?;
            writer
                .write_event(Event::Text(BytesText::new(base)))
                .map_err(|e| PackagerError::manifest_failed(format!("XML write error: {e}")))?;
            writer
                .write_event(Event::End(BytesEnd::new("BaseURL")))
                .map_err(|e| PackagerError::manifest_failed(format!("XML write error: {e}")))?;
        }

        // Periods
        for period in &self.periods {
            self.write_period(&mut writer, period)?;
        }

        // Close MPD
        writer
            .write_event(Event::End(BytesEnd::new("MPD")))
            .map_err(|e| PackagerError::manifest_failed(format!("XML write error: {e}")))?;

        let xml_bytes = writer.into_inner().into_inner();
        String::from_utf8(xml_bytes)
            .map_err(|e| PackagerError::manifest_failed(format!("UTF-8 error: {e}")))
    }

    /// Write a period element.
    fn write_period<W: std::io::Write>(
        &self,
        writer: &mut Writer<W>,
        period: &Period,
    ) -> PackagerResult<()> {
        let mut period_elem = BytesStart::new("Period");
        period_elem.push_attribute(("id", period.id.as_str()));

        if let Some(duration) = period.duration {
            let duration_str = DurationFormatter::format_iso8601_duration(duration);
            period_elem.push_attribute(("duration", duration_str.as_str()));
        }

        writer
            .write_event(Event::Start(period_elem))
            .map_err(|e| PackagerError::manifest_failed(format!("XML write error: {e}")))?;

        // Adaptation sets
        for set in &period.adaptation_sets {
            self.write_adaptation_set(writer, set)?;
        }

        writer
            .write_event(Event::End(BytesEnd::new("Period")))
            .map_err(|e| PackagerError::manifest_failed(format!("XML write error: {e}")))?;

        Ok(())
    }

    /// Write an adaptation set element.
    fn write_adaptation_set<W: std::io::Write>(
        &self,
        writer: &mut Writer<W>,
        set: &AdaptationSet,
    ) -> PackagerResult<()> {
        let mut set_elem = BytesStart::new("AdaptationSet");
        set_elem.push_attribute(("id", set.id.to_string().as_str()));
        set_elem.push_attribute(("contentType", set.content_type.as_str()));
        set_elem.push_attribute(("mimeType", set.mime_type.as_str()));

        if set.segment_alignment {
            set_elem.push_attribute(("segmentAlignment", "true"));
        }

        if let Some(lang) = &set.lang {
            set_elem.push_attribute(("lang", lang.as_str()));
        }

        writer
            .write_event(Event::Start(set_elem))
            .map_err(|e| PackagerError::manifest_failed(format!("XML write error: {e}")))?;

        // Representations
        for repr in &set.representations {
            self.write_representation(writer, repr)?;
        }

        writer
            .write_event(Event::End(BytesEnd::new("AdaptationSet")))
            .map_err(|e| PackagerError::manifest_failed(format!("XML write error: {e}")))?;

        Ok(())
    }

    /// Write a representation element.
    fn write_representation<W: std::io::Write>(
        &self,
        writer: &mut Writer<W>,
        repr: &Representation,
    ) -> PackagerResult<()> {
        let mut repr_elem = BytesStart::new("Representation");
        repr_elem.push_attribute(("id", repr.id.as_str()));
        repr_elem.push_attribute(("bandwidth", repr.bandwidth.to_string().as_str()));
        repr_elem.push_attribute(("codecs", repr.codecs.as_str()));

        if let Some(width) = repr.width {
            repr_elem.push_attribute(("width", width.to_string().as_str()));
        }

        if let Some(height) = repr.height {
            repr_elem.push_attribute(("height", height.to_string().as_str()));
        }

        if let Some(fps) = &repr.frame_rate {
            repr_elem.push_attribute(("frameRate", fps.as_str()));
        }

        if let Some(sample_rate) = repr.audio_sampling_rate {
            repr_elem.push_attribute(("audioSamplingRate", sample_rate.to_string().as_str()));
        }

        writer
            .write_event(Event::Start(repr_elem))
            .map_err(|e| PackagerError::manifest_failed(format!("XML write error: {e}")))?;

        // Base URL
        if let Some(base_url) = &repr.base_url {
            writer
                .write_event(Event::Start(BytesStart::new("BaseURL")))
                .map_err(|e| PackagerError::manifest_failed(format!("XML write error: {e}")))?;
            writer
                .write_event(Event::Text(BytesText::new(base_url)))
                .map_err(|e| PackagerError::manifest_failed(format!("XML write error: {e}")))?;
            writer
                .write_event(Event::End(BytesEnd::new("BaseURL")))
                .map_err(|e| PackagerError::manifest_failed(format!("XML write error: {e}")))?;
        }

        // Segment template
        if let Some(template) = &repr.segment_template {
            self.write_segment_template(writer, template)?;
        }

        writer
            .write_event(Event::End(BytesEnd::new("Representation")))
            .map_err(|e| PackagerError::manifest_failed(format!("XML write error: {e}")))?;

        Ok(())
    }

    /// Write segment template element.
    fn write_segment_template<W: std::io::Write>(
        &self,
        writer: &mut Writer<W>,
        template: &SegmentTemplate,
    ) -> PackagerResult<()> {
        let mut template_elem = BytesStart::new("SegmentTemplate");
        template_elem.push_attribute(("initialization", template.initialization.as_str()));
        template_elem.push_attribute(("media", template.media.as_str()));
        template_elem.push_attribute(("duration", template.duration.to_string().as_str()));
        template_elem.push_attribute(("timescale", template.timescale.to_string().as_str()));
        template_elem.push_attribute(("startNumber", template.start_number.to_string().as_str()));

        writer
            .write_event(Event::Empty(template_elem))
            .map_err(|e| PackagerError::manifest_failed(format!("XML write error: {e}")))?;

        Ok(())
    }
}

/// Create representation from bitrate entry.
pub fn representation_from_bitrate_entry(
    entry: &BitrateEntry,
    id: String,
) -> PackagerResult<Representation> {
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

    let mut repr = Representation::new(id, entry.bitrate, codec_str)
        .with_dimensions(entry.width, entry.height);

    if let Some(fps) = entry.framerate {
        repr = repr.with_frame_rate(fps);
    }

    Ok(repr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mpd_type_conversion() {
        assert_eq!(MpdType::Static.as_str(), "static");
        assert_eq!(MpdType::Dynamic.as_str(), "dynamic");
    }

    #[test]
    fn test_dash_profile_urn() {
        assert!(DashProfile::OnDemand.as_urn().contains("on-demand"));
    }

    #[test]
    fn test_representation_creation() {
        let repr =
            Representation::new("video1".to_string(), 1_000_000, "av01.0.04M.08".to_string());

        assert_eq!(repr.id, "video1");
        assert_eq!(repr.bandwidth, 1_000_000);
    }

    #[test]
    fn test_mpd_builder() {
        let mut builder = MpdBuilder::new(MpdType::Static, DashProfile::OnDemand);

        let mut period = Period::new("0".to_string());
        let mut adaptation_set =
            AdaptationSet::new(0, "video".to_string(), "video/mp4".to_string());

        let repr = Representation::new("1".to_string(), 1_000_000, "av01.0.04M.08".to_string());

        adaptation_set.add_representation(repr);
        period.add_adaptation_set(adaptation_set);
        builder.add_period(period);

        let mpd = builder.build().expect("should succeed in test");

        assert!(mpd.contains("<MPD"));
        assert!(mpd.contains("type=\"static\""));
        assert!(mpd.contains("Representation"));
    }
}
