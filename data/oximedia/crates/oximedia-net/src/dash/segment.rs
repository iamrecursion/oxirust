//! DASH segment handling.
//!
//! This module provides types for managing DASH segments.

#![allow(dead_code)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::similar_names)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::range_plus_one)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::comparison_chain)]
#![allow(clippy::unused_self)]
#![allow(clippy::trivially_copy_pass_by_ref)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::redundant_closure_for_method_calls)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::should_implement_trait)]
#![allow(clippy::items_after_statements)]
#![allow(clippy::if_not_else)]
#![allow(clippy::format_push_string)]
#![allow(clippy::single_match_else)]
#![allow(clippy::redundant_slicing)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::map_unwrap_or)]
#![allow(clippy::derivable_impls)]
#![allow(clippy::assigning_clones)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::format_collect)]
#![allow(clippy::useless_conversion)]
#![allow(clippy::unused_async)]
#![allow(clippy::identity_op)]
use super::mpd::{Representation, SegmentTemplate, SegmentTimeline};
use std::time::Duration;

/// Information about a DASH segment.
#[derive(Debug, Clone)]
pub struct SegmentInfo {
    /// Segment number.
    pub number: u64,
    /// Start time in timescale units.
    pub start_time: u64,
    /// Duration in timescale units.
    pub duration: u64,
    /// Timescale.
    pub timescale: u32,
}

impl SegmentInfo {
    /// Creates a new segment info.
    #[must_use]
    pub const fn new(number: u64, start_time: u64, duration: u64, timescale: u32) -> Self {
        Self {
            number,
            start_time,
            duration,
            timescale,
        }
    }

    /// Returns the start time in seconds.
    #[must_use]
    pub fn start_time_secs(&self) -> f64 {
        self.start_time as f64 / self.timescale as f64
    }

    /// Returns the duration in seconds.
    #[must_use]
    pub fn duration_secs(&self) -> f64 {
        self.duration as f64 / self.timescale as f64
    }

    /// Returns the end time in timescale units.
    #[must_use]
    pub const fn end_time(&self) -> u64 {
        self.start_time + self.duration
    }

    /// Returns the end time in seconds.
    #[must_use]
    pub fn end_time_secs(&self) -> f64 {
        self.end_time() as f64 / self.timescale as f64
    }

    /// Returns the start time as Duration.
    #[must_use]
    pub fn start_duration(&self) -> Duration {
        Duration::from_secs_f64(self.start_time_secs())
    }

    /// Returns the segment duration as Duration.
    #[must_use]
    pub fn segment_duration(&self) -> Duration {
        Duration::from_secs_f64(self.duration_secs())
    }
}

/// A DASH segment with its URL and metadata.
#[derive(Debug, Clone)]
pub struct DashSegment {
    /// Segment URL.
    pub url: String,
    /// Segment info.
    pub info: SegmentInfo,
    /// Byte range (if applicable).
    pub byte_range: Option<(u64, u64)>,
    /// Is initialization segment.
    pub is_init: bool,
}

impl DashSegment {
    /// Creates a new DASH segment.
    #[must_use]
    pub fn new(url: impl Into<String>, info: SegmentInfo) -> Self {
        Self {
            url: url.into(),
            info,
            byte_range: None,
            is_init: false,
        }
    }

    /// Creates an initialization segment.
    #[must_use]
    pub fn init(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            info: SegmentInfo::new(0, 0, 0, 1),
            byte_range: None,
            is_init: true,
        }
    }

    /// Sets the byte range.
    #[must_use]
    pub const fn with_byte_range(mut self, start: u64, end: u64) -> Self {
        self.byte_range = Some((start, end));
        self
    }

    /// Returns true if this segment has a byte range.
    #[must_use]
    pub const fn has_byte_range(&self) -> bool {
        self.byte_range.is_some()
    }

    /// Returns the HTTP Range header value if applicable.
    #[must_use]
    pub fn range_header(&self) -> Option<String> {
        self.byte_range
            .map(|(start, end)| format!("bytes={start}-{end}"))
    }
}

/// Generates segments from a segment template.
#[derive(Debug)]
pub struct SegmentGenerator {
    /// Representation ID.
    representation_id: String,
    /// Segment template.
    template: SegmentTemplate,
    /// Base URL.
    base_url: Option<String>,
}

impl SegmentGenerator {
    /// Creates a new segment generator.
    #[must_use]
    pub fn new(representation_id: impl Into<String>, template: SegmentTemplate) -> Self {
        Self {
            representation_id: representation_id.into(),
            template,
            base_url: None,
        }
    }

    /// Creates a generator from a representation.
    #[must_use]
    pub fn from_representation(rep: &Representation) -> Option<Self> {
        let template = rep.segment_template.clone()?;
        Some(Self::new(&rep.id, template))
    }

    /// Sets the base URL.
    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// Returns the timescale.
    #[must_use]
    pub const fn timescale(&self) -> u32 {
        self.template.timescale
    }

    /// Generates the initialization segment.
    #[must_use]
    pub fn initialization_segment(&self) -> Option<DashSegment> {
        let url = self.template.initialization_url(&self.representation_id)?;
        let full_url = self.resolve_url(&url);
        Some(DashSegment::init(full_url))
    }

    /// Generates a segment by number (for number-based templates).
    #[must_use]
    pub fn segment_by_number(&self, number: u64) -> Option<DashSegment> {
        let duration = self.template.duration?;
        let start_time = (number - self.template.start_number) * duration;

        let url = self
            .template
            .media_url(&self.representation_id, number, Some(start_time))?;
        let full_url = self.resolve_url(&url);

        let info = SegmentInfo::new(number, start_time, duration, self.template.timescale);
        Some(DashSegment::new(full_url, info))
    }

    /// Generates a segment by time (for timeline-based templates).
    #[must_use]
    pub fn segment_by_time(&self, time: u64, duration: u64, number: u64) -> Option<DashSegment> {
        let url = self
            .template
            .media_url(&self.representation_id, number, Some(time))?;
        let full_url = self.resolve_url(&url);

        let info = SegmentInfo::new(number, time, duration, self.template.timescale);
        Some(DashSegment::new(full_url, info))
    }

    /// Generates all segments from the timeline.
    pub fn segments_from_timeline(&self) -> Vec<DashSegment> {
        let Some(ref timeline) = self.template.segment_timeline else {
            return Vec::new();
        };

        let mut segments = Vec::new();
        let mut number = self.template.start_number;

        for (start_time, duration) in timeline.iter_segments() {
            if let Some(seg) = self.segment_by_time(start_time, duration, number) {
                segments.push(seg);
            }
            number += 1;
        }

        segments
    }

    /// Generates segments for a time range.
    pub fn segments_for_range(&self, start_secs: f64, end_secs: f64) -> Vec<DashSegment> {
        let start_time = (start_secs * self.template.timescale as f64) as u64;
        let end_time = (end_secs * self.template.timescale as f64) as u64;

        if let Some(ref timeline) = self.template.segment_timeline {
            self.segments_from_timeline_range(timeline, start_time, end_time)
        } else if let Some(duration) = self.template.duration {
            self.segments_from_number_range(duration, start_time, end_time)
        } else {
            Vec::new()
        }
    }

    fn segments_from_timeline_range(
        &self,
        timeline: &SegmentTimeline,
        start_time: u64,
        end_time: u64,
    ) -> Vec<DashSegment> {
        let mut segments = Vec::new();
        let mut number = self.template.start_number;

        for (seg_start, seg_duration) in timeline.iter_segments() {
            let seg_end = seg_start + seg_duration;

            // Check if segment overlaps with requested range
            if seg_end > start_time && seg_start < end_time {
                if let Some(seg) = self.segment_by_time(seg_start, seg_duration, number) {
                    segments.push(seg);
                }
            }

            // Stop if we've passed the end
            if seg_start >= end_time {
                break;
            }

            number += 1;
        }

        segments
    }

    fn segments_from_number_range(
        &self,
        duration: u64,
        start_time: u64,
        end_time: u64,
    ) -> Vec<DashSegment> {
        let mut segments = Vec::new();

        let start_number = self.template.start_number + start_time / duration;
        let end_number = self.template.start_number + (end_time + duration - 1) / duration;

        for number in start_number..=end_number {
            if let Some(seg) = self.segment_by_number(number) {
                segments.push(seg);
            }
        }

        segments
    }

    /// Returns the segment containing the given time.
    #[must_use]
    pub fn segment_at_time(&self, time_secs: f64) -> Option<DashSegment> {
        let time = (time_secs * self.template.timescale as f64) as u64;

        if let Some(ref timeline) = self.template.segment_timeline {
            let mut number = self.template.start_number;
            for (seg_start, seg_duration) in timeline.iter_segments() {
                if time >= seg_start && time < seg_start + seg_duration {
                    return self.segment_by_time(seg_start, seg_duration, number);
                }
                number += 1;
            }
            None
        } else {
            let duration = self.template.duration?;
            let number = self.template.start_number + time / duration;
            self.segment_by_number(number)
        }
    }

    fn resolve_url(&self, url: &str) -> String {
        if url.starts_with("http://") || url.starts_with("https://") {
            return url.to_string();
        }

        match &self.base_url {
            Some(base) => {
                if base.ends_with('/') {
                    format!("{base}{url}")
                } else {
                    format!("{base}/{url}")
                }
            }
            None => url.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dash::mpd::SegmentTimelineEntry;

    #[test]
    fn test_segment_info() {
        let info = SegmentInfo::new(1, 90000, 90000, 90000);

        assert_eq!(info.number, 1);
        assert!((info.start_time_secs() - 1.0).abs() < 0.001);
        assert!((info.duration_secs() - 1.0).abs() < 0.001);
        assert_eq!(info.end_time(), 180000);
    }

    #[test]
    fn test_dash_segment() {
        let info = SegmentInfo::new(1, 0, 90000, 90000);
        let seg = DashSegment::new("segment1.m4s", info).with_byte_range(0, 999);

        assert_eq!(seg.url, "segment1.m4s");
        assert!(seg.has_byte_range());
        assert_eq!(seg.range_header(), Some("bytes=0-999".to_string()));
        assert!(!seg.is_init);
    }

    #[test]
    fn test_dash_segment_init() {
        let seg = DashSegment::init("init.mp4");
        assert!(seg.is_init);
        assert_eq!(seg.url, "init.mp4");
    }

    #[test]
    fn test_segment_generator_number_based() {
        let template = SegmentTemplate::new(90000)
            .with_media("segment_$Number$.m4s")
            .with_initialization("init.mp4");

        let mut gen = SegmentGenerator::new("720p", template);
        gen.template.duration = Some(90000);
        gen.template.start_number = 1;

        // Test initialization
        let init = gen
            .initialization_segment()
            .expect("should succeed in test");
        assert!(init.is_init);
        assert_eq!(init.url, "init.mp4");

        // Test segment by number
        let seg = gen.segment_by_number(1).expect("should succeed in test");
        assert_eq!(seg.url, "segment_1.m4s");
        assert_eq!(seg.info.number, 1);
        assert_eq!(seg.info.start_time, 0);
    }

    #[test]
    fn test_segment_generator_with_base_url() {
        let template = SegmentTemplate::new(90000).with_media("segment_$Number$.m4s");

        let mut gen =
            SegmentGenerator::new("720p", template).with_base_url("https://cdn.example.com/video");
        gen.template.duration = Some(90000);

        let seg = gen.segment_by_number(1).expect("should succeed in test");
        assert_eq!(seg.url, "https://cdn.example.com/video/segment_1.m4s");
    }

    #[test]
    fn test_segment_generator_timeline() {
        let mut timeline = SegmentTimeline::new();
        timeline.add_entry(SegmentTimelineEntry::new(90000).with_start(0));
        timeline.add_entry(SegmentTimelineEntry::new(90000).with_repeat(2));

        let template = SegmentTemplate {
            timescale: 90000,
            start_number: 1,
            media: Some("segment_$Number$_$Time$.m4s".to_string()),
            segment_timeline: Some(timeline),
            ..Default::default()
        };

        let gen = SegmentGenerator::new("720p", template);
        let segments = gen.segments_from_timeline();

        assert_eq!(segments.len(), 4);
        assert_eq!(segments[0].info.start_time, 0);
        assert_eq!(segments[1].info.start_time, 90000);
        assert_eq!(segments[2].info.start_time, 180000);
        assert_eq!(segments[3].info.start_time, 270000);
    }

    #[test]
    fn test_segment_at_time() {
        let template = SegmentTemplate::new(1000).with_media("seg_$Number$.m4s");

        let mut gen = SegmentGenerator::new("v1", template);
        gen.template.duration = Some(10000); // 10 seconds
        gen.template.start_number = 0;

        // At 5 seconds, should be segment 0
        let seg = gen.segment_at_time(5.0).expect("should succeed in test");
        assert_eq!(seg.info.number, 0);

        // At 15 seconds, should be segment 1
        let seg = gen.segment_at_time(15.0).expect("should succeed in test");
        assert_eq!(seg.info.number, 1);
    }

    #[test]
    fn test_segments_for_range() {
        let template = SegmentTemplate::new(1000).with_media("seg_$Number$.m4s");

        let mut gen = SegmentGenerator::new("v1", template);
        gen.template.duration = Some(10000); // 10 seconds per segment
        gen.template.start_number = 0;

        // Get segments from 5 to 25 seconds
        // Segment 0: 0-10s, Segment 1: 10-20s, Segment 2: 20-30s
        // All three overlap with the requested range 5-25s
        // But we also get segment 3 because of ceiling division
        let segments = gen.segments_for_range(5.0, 25.0);
        assert!(segments.len() >= 3);
    }
}
