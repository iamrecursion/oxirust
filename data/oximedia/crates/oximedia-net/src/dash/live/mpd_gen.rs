//! Dynamic MPD (Media Presentation Description) generation for DASH live streaming.
//!
//! This module provides functionality to generate and update dynamic MPDs
//! for live DASH streaming, including SegmentTimeline, UTCTiming, and
//! multi-period support.

#![allow(dead_code)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::too_many_arguments)]

use crate::dash::mpd::{
    format_iso8601_duration, AdaptationSet, Mpd, MpdType, Period, Representation, SegmentTemplate,
    SegmentTimeline,
};
use std::time::{Duration, SystemTime};

/// Dynamic MPD generator for live streaming.
///
/// This structure manages the generation of dynamic MPDs, including
/// timeline updates, period management, and manifest refresh.
#[derive(Debug)]
pub struct DynamicMpdGenerator {
    /// Base MPD template.
    mpd: Mpd,
    /// Current period ID counter.
    period_id_counter: u32,
    /// MPD update counter.
    publish_counter: u64,
}

impl DynamicMpdGenerator {
    /// Creates a new dynamic MPD generator.
    ///
    /// # Arguments
    ///
    /// * `config` - MPD configuration
    #[must_use]
    pub fn new(config: MpdConfig) -> Self {
        let mut mpd = Mpd::new();
        mpd.mpd_type = MpdType::Dynamic;
        mpd.min_buffer_time = config.min_buffer_time;
        mpd.suggested_presentation_delay = Some(config.suggested_presentation_delay);
        mpd.time_shift_buffer_depth = Some(config.time_shift_buffer_depth);
        mpd.availability_start_time = Some(Self::format_time(config.availability_start_time));
        mpd.minimum_update_period = Some(config.minimum_update_period);
        mpd.profiles = vec!["urn:mpeg:dash:profile:isoff-live:2011".to_string()];

        Self {
            mpd,
            period_id_counter: 0,
            publish_counter: 0,
        }
    }

    /// Adds a period to the MPD.
    ///
    /// # Arguments
    ///
    /// * `start_time` - Period start time (None for first period)
    /// * `duration` - Period duration (None for open-ended)
    ///
    /// # Returns
    ///
    /// The period ID
    pub fn add_period(
        &mut self,
        start_time: Option<Duration>,
        duration: Option<Duration>,
    ) -> String {
        let period_id = format!("period_{}", self.period_id_counter);
        self.period_id_counter += 1;

        let mut period = Period::new();
        period.id = Some(period_id.clone());
        period.start = start_time;
        period.duration = duration;

        self.mpd.periods.push(period);

        period_id
    }

    /// Adds an adaptation set to a period.
    ///
    /// # Arguments
    ///
    /// * `period_id` - Period identifier
    /// * `adaptation_set` - The adaptation set to add
    pub fn add_adaptation_set(&mut self, period_id: &str, adaptation_set: AdaptationSet) {
        if let Some(period) = self
            .mpd
            .periods
            .iter_mut()
            .find(|p| p.id.as_deref() == Some(period_id))
        {
            period.adaptation_sets.push(adaptation_set);
        }
    }

    /// Updates the segment timeline for a representation.
    ///
    /// # Arguments
    ///
    /// * `period_id` - Period identifier
    /// * `adaptation_set_id` - Adaptation set ID
    /// * `representation_id` - Representation ID
    /// * `timeline` - New timeline
    pub fn update_timeline(
        &mut self,
        period_id: &str,
        adaptation_set_id: u32,
        representation_id: &str,
        timeline: SegmentTimeline,
    ) {
        if let Some(period) = self
            .mpd
            .periods
            .iter_mut()
            .find(|p| p.id.as_deref() == Some(period_id))
        {
            if let Some(adaptation_set) = period
                .adaptation_sets
                .iter_mut()
                .find(|a| a.id == Some(adaptation_set_id))
            {
                if let Some(representation) = adaptation_set
                    .representations
                    .iter_mut()
                    .find(|r| r.id == representation_id)
                {
                    if let Some(ref mut template) = representation.segment_template {
                        template.segment_timeline = Some(timeline);
                    }
                }
            }
        }
    }

    /// Generates the MPD XML.
    ///
    /// # Returns
    ///
    /// The MPD as XML string
    #[must_use]
    pub fn generate_xml(&mut self) -> String {
        self.publish_counter += 1;
        self.mpd.publish_time = Some(Self::format_time(SystemTime::now()));

        self.build_xml()
    }

    /// Adds a UTCTiming element.
    ///
    /// # Arguments
    ///
    /// * `scheme` - Timing scheme (e.g., "urn:mpeg:dash:utc:http-xsdate:2014")
    /// * `value` - Timing source URL or value
    pub fn add_utc_timing(&mut self, scheme: String, value: String) {
        // Store in base URLs for now (proper implementation would extend Mpd struct)
        let timing_descriptor = format!("utc-timing:{}:{}", scheme, value);
        self.mpd.base_urls.push(timing_descriptor);
    }

    /// Sets the availability start time.
    pub fn set_availability_start_time(&mut self, time: SystemTime) {
        self.mpd.availability_start_time = Some(Self::format_time(time));
    }

    /// Returns the current MPD.
    #[must_use]
    pub fn mpd(&self) -> &Mpd {
        &self.mpd
    }

    /// Returns the current period.
    #[must_use]
    pub fn current_period(&self) -> Option<&Period> {
        self.mpd.periods.last()
    }

    /// Returns a mutable reference to the current period.
    pub fn current_period_mut(&mut self) -> Option<&mut Period> {
        self.mpd.periods.last_mut()
    }

    /// Formats a system time as ISO 8601.
    fn format_time(time: SystemTime) -> String {
        super::timeline::TimelineManager::format_system_time(time)
    }

    /// Builds the MPD XML.
    fn build_xml(&self) -> String {
        let mut xml = String::new();

        // XML declaration
        xml.push_str(r#"<?xml version="1.0" encoding="UTF-8"?>"#);
        xml.push('\n');

        // MPD element
        xml.push_str(r#"<MPD xmlns="urn:mpeg:dash:schema:mpd:2011""#);
        xml.push_str(r#" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance""#);
        xml.push_str(r#" xsi:schemaLocation="urn:mpeg:dash:schema:mpd:2011 DASH-MPD.xsd""#);
        xml.push_str(r#" type="dynamic""#);

        // Profiles
        if !self.mpd.profiles.is_empty() {
            xml.push_str(r#" profiles=""#);
            xml.push_str(&self.mpd.profiles.join(","));
            xml.push('"');
        }

        // Min buffer time
        xml.push_str(r#" minBufferTime=""#);
        xml.push_str(&format_iso8601_duration(self.mpd.min_buffer_time));
        xml.push('"');

        // Availability start time
        if let Some(ref ast) = self.mpd.availability_start_time {
            xml.push_str(r#" availabilityStartTime=""#);
            xml.push_str(ast);
            xml.push('"');
        }

        // Publish time
        if let Some(ref pt) = self.mpd.publish_time {
            xml.push_str(r#" publishTime=""#);
            xml.push_str(pt);
            xml.push('"');
        }

        // Minimum update period
        if let Some(mup) = self.mpd.minimum_update_period {
            xml.push_str(r#" minimumUpdatePeriod=""#);
            xml.push_str(&format_iso8601_duration(mup));
            xml.push('"');
        }

        // Time shift buffer depth
        if let Some(tsbd) = self.mpd.time_shift_buffer_depth {
            xml.push_str(r#" timeShiftBufferDepth=""#);
            xml.push_str(&format_iso8601_duration(tsbd));
            xml.push('"');
        }

        // Suggested presentation delay
        if let Some(spd) = self.mpd.suggested_presentation_delay {
            xml.push_str(r#" suggestedPresentationDelay=""#);
            xml.push_str(&format_iso8601_duration(spd));
            xml.push('"');
        }

        xml.push_str(">\n");

        // UTCTiming elements (extracted from base_urls for simplicity)
        for base_url in &self.mpd.base_urls {
            if base_url.starts_with("utc-timing:") {
                let parts: Vec<&str> = base_url.splitn(3, ':').collect();
                if parts.len() == 3 {
                    xml.push_str("  <UTCTiming schemeIdUri=\"");
                    xml.push_str(parts[1]);
                    xml.push_str("\" value=\"");
                    xml.push_str(parts[2]);
                    xml.push_str("\"/>\n");
                }
            }
        }

        // Periods
        for period in &self.mpd.periods {
            self.build_period_xml(&mut xml, period);
        }

        xml.push_str("</MPD>\n");

        xml
    }

    /// Builds XML for a period.
    fn build_period_xml(&self, xml: &mut String, period: &Period) {
        xml.push_str("  <Period");

        if let Some(ref id) = period.id {
            xml.push_str(r#" id=""#);
            xml.push_str(id);
            xml.push('"');
        }

        if let Some(start) = period.start {
            xml.push_str(r#" start=""#);
            xml.push_str(&format_iso8601_duration(start));
            xml.push('"');
        }

        if let Some(duration) = period.duration {
            xml.push_str(r#" duration=""#);
            xml.push_str(&format_iso8601_duration(duration));
            xml.push('"');
        }

        xml.push_str(">\n");

        // Adaptation sets
        for adaptation_set in &period.adaptation_sets {
            self.build_adaptation_set_xml(xml, adaptation_set);
        }

        xml.push_str("  </Period>\n");
    }

    /// Builds XML for an adaptation set.
    fn build_adaptation_set_xml(&self, xml: &mut String, adaptation_set: &AdaptationSet) {
        xml.push_str("    <AdaptationSet");

        if let Some(id) = adaptation_set.id {
            xml.push_str(&format!(r#" id="{id}""#));
        }

        if let Some(ref content_type) = adaptation_set.content_type {
            xml.push_str(r#" contentType=""#);
            xml.push_str(content_type);
            xml.push('"');
        }

        if let Some(ref mime_type) = adaptation_set.mime_type {
            xml.push_str(r#" mimeType=""#);
            xml.push_str(mime_type);
            xml.push('"');
        }

        if adaptation_set.segment_alignment {
            xml.push_str(r#" segmentAlignment="true""#);
        }

        xml.push_str(">\n");

        // Segment template at adaptation set level
        if let Some(ref template) = adaptation_set.segment_template {
            self.build_segment_template_xml(xml, template, "      ");
        }

        // Representations
        for representation in &adaptation_set.representations {
            self.build_representation_xml(xml, representation);
        }

        xml.push_str("    </AdaptationSet>\n");
    }

    /// Builds XML for a representation.
    fn build_representation_xml(&self, xml: &mut String, representation: &Representation) {
        xml.push_str("      <Representation");

        xml.push_str(r#" id=""#);
        xml.push_str(&representation.id);
        xml.push('"');

        xml.push_str(&format!(r#" bandwidth="{}""#, representation.bandwidth));

        if let Some(ref codecs) = representation.codecs {
            xml.push_str(r#" codecs=""#);
            xml.push_str(codecs);
            xml.push('"');
        }

        if let Some(width) = representation.width {
            xml.push_str(&format!(r#" width="{width}""#));
        }

        if let Some(height) = representation.height {
            xml.push_str(&format!(r#" height="{height}""#));
        }

        xml.push_str(">\n");

        // Segment template at representation level
        if let Some(ref template) = representation.segment_template {
            self.build_segment_template_xml(xml, template, "        ");
        }

        xml.push_str("      </Representation>\n");
    }

    /// Builds XML for a segment template.
    fn build_segment_template_xml(
        &self,
        xml: &mut String,
        template: &SegmentTemplate,
        indent: &str,
    ) {
        xml.push_str(indent);
        xml.push_str("<SegmentTemplate");

        xml.push_str(&format!(r#" timescale="{}""#, template.timescale));

        if let Some(ref media) = template.media {
            xml.push_str(r#" media=""#);
            xml.push_str(media);
            xml.push('"');
        }

        if let Some(ref initialization) = template.initialization {
            xml.push_str(r#" initialization=""#);
            xml.push_str(initialization);
            xml.push('"');
        }

        xml.push_str(&format!(r#" startNumber="{}""#, template.start_number));

        if let Some(pto) = template.presentation_time_offset {
            xml.push_str(&format!(r#" presentationTimeOffset="{pto}""#));
        }

        if template.segment_timeline.is_some() {
            xml.push_str(">\n");

            // Segment timeline
            if let Some(ref timeline) = template.segment_timeline {
                self.build_timeline_xml(xml, timeline, &format!("{indent}  "));
            }

            xml.push_str(indent);
            xml.push_str("</SegmentTemplate>\n");
        } else {
            if let Some(duration) = template.duration {
                xml.push_str(&format!(r#" duration="{duration}""#));
            }
            xml.push_str("/>\n");
        }
    }

    /// Builds XML for a segment timeline.
    fn build_timeline_xml(&self, xml: &mut String, timeline: &SegmentTimeline, indent: &str) {
        xml.push_str(indent);
        xml.push_str("<SegmentTimeline>\n");

        for entry in &timeline.entries {
            xml.push_str(indent);
            xml.push_str("  <S");

            if let Some(t) = entry.start {
                xml.push_str(&format!(r#" t="{t}""#));
            }

            xml.push_str(&format!(r#" d="{}""#, entry.duration));

            if entry.repeat != 0 {
                xml.push_str(&format!(r#" r="{}""#, entry.repeat));
            }

            xml.push_str("/>\n");
        }

        xml.push_str(indent);
        xml.push_str("</SegmentTimeline>\n");
    }
}

/// MPD configuration for live streaming.
#[derive(Debug, Clone)]
pub struct MpdConfig {
    /// Minimum buffer time.
    pub min_buffer_time: Duration,
    /// Suggested presentation delay.
    pub suggested_presentation_delay: Duration,
    /// Time shift buffer depth (DVR window).
    pub time_shift_buffer_depth: Duration,
    /// Availability start time.
    pub availability_start_time: SystemTime,
    /// Minimum update period.
    pub minimum_update_period: Duration,
}

impl Default for MpdConfig {
    fn default() -> Self {
        Self {
            min_buffer_time: Duration::from_secs(2),
            suggested_presentation_delay: Duration::from_secs(6),
            time_shift_buffer_depth: Duration::from_secs(60),
            availability_start_time: SystemTime::now(),
            minimum_update_period: Duration::from_secs(2),
        }
    }
}

/// Builder for creating adaptation sets.
#[derive(Debug)]
pub struct AdaptationSetBuilder {
    adaptation_set: AdaptationSet,
}

impl AdaptationSetBuilder {
    /// Creates a new builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            adaptation_set: AdaptationSet::new(),
        }
    }

    /// Sets the ID.
    #[must_use]
    pub fn id(mut self, id: u32) -> Self {
        self.adaptation_set.id = Some(id);
        self
    }

    /// Sets the content type.
    #[must_use]
    pub fn content_type(mut self, content_type: impl Into<String>) -> Self {
        self.adaptation_set.content_type = Some(content_type.into());
        self
    }

    /// Sets the MIME type.
    #[must_use]
    pub fn mime_type(mut self, mime_type: impl Into<String>) -> Self {
        self.adaptation_set.mime_type = Some(mime_type.into());
        self
    }

    /// Enables segment alignment.
    #[must_use]
    pub fn segment_alignment(mut self, enabled: bool) -> Self {
        self.adaptation_set.segment_alignment = enabled;
        self
    }

    /// Sets the segment template.
    #[must_use]
    pub fn segment_template(mut self, template: SegmentTemplate) -> Self {
        self.adaptation_set.segment_template = Some(template);
        self
    }

    /// Adds a representation.
    #[must_use]
    pub fn representation(mut self, representation: Representation) -> Self {
        self.adaptation_set.representations.push(representation);
        self
    }

    /// Builds the adaptation set.
    #[must_use]
    pub fn build(self) -> AdaptationSet {
        self.adaptation_set
    }
}

impl Default for AdaptationSetBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for creating representations.
#[derive(Debug)]
pub struct RepresentationBuilder {
    representation: Representation,
}

impl RepresentationBuilder {
    /// Creates a new builder.
    #[must_use]
    pub fn new(id: impl Into<String>, bandwidth: u64) -> Self {
        Self {
            representation: Representation::new(id, bandwidth),
        }
    }

    /// Sets the codecs.
    #[must_use]
    pub fn codecs(mut self, codecs: impl Into<String>) -> Self {
        self.representation.codecs = Some(codecs.into());
        self
    }

    /// Sets the resolution.
    #[must_use]
    pub fn resolution(mut self, width: u32, height: u32) -> Self {
        self.representation.width = Some(width);
        self.representation.height = Some(height);
        self
    }

    /// Sets the segment template.
    #[must_use]
    pub fn segment_template(mut self, template: SegmentTemplate) -> Self {
        self.representation.segment_template = Some(template);
        self
    }

    /// Builds the representation.
    #[must_use]
    pub fn build(self) -> Representation {
        self.representation
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mpd_config_default() {
        let config = MpdConfig::default();
        assert_eq!(config.min_buffer_time, Duration::from_secs(2));
        assert_eq!(config.suggested_presentation_delay, Duration::from_secs(6));
    }

    #[test]
    fn test_dynamic_mpd_generator_creation() {
        let config = MpdConfig::default();
        let generator = DynamicMpdGenerator::new(config);

        assert!(generator.mpd().is_live());
        assert_eq!(generator.mpd().periods.len(), 0);
    }

    #[test]
    fn test_add_period() {
        let config = MpdConfig::default();
        let mut generator = DynamicMpdGenerator::new(config);

        let period_id = generator.add_period(None, None);
        assert_eq!(period_id, "period_0");
        assert_eq!(generator.mpd().periods.len(), 1);

        let period_id2 = generator.add_period(Some(Duration::from_secs(60)), None);
        assert_eq!(period_id2, "period_1");
        assert_eq!(generator.mpd().periods.len(), 2);
    }

    #[test]
    fn test_generate_xml() {
        let config = MpdConfig::default();
        let mut generator = DynamicMpdGenerator::new(config);

        let xml = generator.generate_xml();
        assert!(xml.contains("<?xml"));
        assert!(xml.contains("<MPD"));
        assert!(xml.contains("type=\"dynamic\""));
    }

    #[test]
    fn test_adaptation_set_builder() {
        let template = SegmentTemplate::new(90000);

        let adaptation_set = AdaptationSetBuilder::new()
            .id(0)
            .content_type("video")
            .mime_type("video/mp4")
            .segment_alignment(true)
            .segment_template(template)
            .build();

        assert_eq!(adaptation_set.id, Some(0));
        assert_eq!(adaptation_set.content_type, Some("video".to_string()));
        assert!(adaptation_set.segment_alignment);
    }

    #[test]
    fn test_representation_builder() {
        let template = SegmentTemplate::new(90000);

        let representation = RepresentationBuilder::new("720p", 1_500_000)
            .codecs("avc1.4d401f")
            .resolution(1280, 720)
            .segment_template(template)
            .build();

        assert_eq!(representation.id, "720p");
        assert_eq!(representation.bandwidth, 1_500_000);
        assert_eq!(representation.width, Some(1280));
        assert_eq!(representation.height, Some(720));
    }

    #[test]
    fn test_utc_timing() {
        let config = MpdConfig::default();
        let mut generator = DynamicMpdGenerator::new(config);

        generator.add_utc_timing(
            "urn:mpeg:dash:utc:http-xsdate:2014".to_string(),
            "https://time.example.com/utc".to_string(),
        );

        let xml = generator.generate_xml();
        assert!(xml.contains("UTCTiming"));
    }

    #[test]
    fn test_mpd_generation_structural_validation() {
        // Structural validation: checks required MPEG-DASH elements and attributes.
        // Not validated against the official DASH-MPD XML Schema (would require an external XSD validator).
        let mut generator = DynamicMpdGenerator::new(MpdConfig::default());

        let period_id = generator.add_period(None, None);

        let template = SegmentTemplate::new(90000)
            .with_media("seg_$Number$.m4s")
            .with_initialization("init.mp4");

        let representation = RepresentationBuilder::new("720p", 1_500_000)
            .codecs("avc1.4d401f")
            .resolution(1280, 720)
            .build();

        let adaptation_set = AdaptationSetBuilder::new()
            .id(0)
            .content_type("video")
            .mime_type("video/mp4")
            .segment_alignment(true)
            .segment_template(template)
            .representation(representation)
            .build();

        generator.add_adaptation_set(&period_id, adaptation_set);

        let xml = generator.generate_xml();

        assert!(xml.contains("<?xml"), "must have XML declaration");
        assert!(xml.contains("<MPD"), "must have MPD element");
        assert!(xml.contains("type=\"dynamic\""), "must be dynamic type");
        assert!(
            xml.contains("profiles=\"urn:mpeg:dash:profile:isoff-live:2011\""),
            "must have DASH live profile"
        );
        assert!(xml.contains("minBufferTime="), "must have minBufferTime");
        assert!(
            xml.contains("availabilityStartTime="),
            "must have availabilityStartTime"
        );
        assert!(xml.contains("<Period"), "must have Period element");
        assert!(
            xml.contains("<AdaptationSet"),
            "must have AdaptationSet element"
        );
        assert!(
            xml.contains("contentType=\"video\""),
            "must have contentType"
        );
        assert!(xml.contains("mimeType=\"video/mp4\""), "must have mimeType");
        assert!(
            xml.contains("segmentAlignment=\"true\""),
            "must have segmentAlignment"
        );
        assert!(
            xml.contains("<SegmentTemplate"),
            "must have SegmentTemplate"
        );
        assert!(xml.contains("timescale=\"90000\""), "must have timescale");
        assert!(
            xml.contains("media=\"seg_$Number$.m4s\""),
            "must have media template"
        );
        assert!(
            xml.contains("initialization=\"init.mp4\""),
            "must have initialization"
        );
        assert!(xml.contains("<Representation"), "must have Representation");
        assert!(xml.contains("id=\"720p\""), "must have representation id");
        assert!(xml.contains("bandwidth=\"1500000\""), "must have bandwidth");
        assert!(xml.contains("codecs=\"avc1.4d401f\""), "must have codecs");
        assert!(xml.contains("width=\"1280\""), "must have width");
        assert!(xml.contains("height=\"720\""), "must have height");
        assert!(xml.contains("</MPD>"), "must have closing MPD tag");

        // Well-formedness: exactly one MPD root element
        assert_eq!(xml.matches("<MPD").count(), 1, "must have exactly one <MPD");
        assert_eq!(
            xml.matches("</MPD>").count(),
            1,
            "must have exactly one </MPD>"
        );
    }
}
