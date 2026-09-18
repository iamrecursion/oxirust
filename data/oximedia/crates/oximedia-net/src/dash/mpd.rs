//! DASH MPD (Media Presentation Description) parsing.
//!
//! This module provides types for parsing and representing MPEG-DASH manifests.

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
use crate::error::{NetError, NetResult};
use std::time::Duration;

/// MPD presentation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MpdType {
    /// Static (VOD) presentation.
    #[default]
    Static,
    /// Dynamic (live) presentation.
    Dynamic,
}

impl MpdType {
    /// Parses from string.
    #[must_use]
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "dynamic" => Self::Dynamic,
            _ => Self::Static,
        }
    }

    /// Returns true if this is a live presentation.
    #[must_use]
    pub const fn is_live(&self) -> bool {
        matches!(self, Self::Dynamic)
    }
}

/// URL type for base URLs and initialization segments.
#[derive(Debug, Clone, Default)]
pub struct UrlType {
    /// URL source.
    pub source_url: Option<String>,
    /// Byte range.
    pub range: Option<(u64, u64)>,
}

impl UrlType {
    /// Creates a new URL type.
    #[must_use]
    pub fn new(source_url: impl Into<String>) -> Self {
        Self {
            source_url: Some(source_url.into()),
            range: None,
        }
    }

    /// Sets the byte range.
    #[must_use]
    pub const fn with_range(mut self, start: u64, end: u64) -> Self {
        self.range = Some((start, end));
        self
    }
}

/// Program information element.
#[derive(Debug, Clone, Default)]
pub struct ProgramInformation {
    /// Language.
    pub lang: Option<String>,
    /// More information URL.
    pub more_info_url: Option<String>,
    /// Title.
    pub title: Option<String>,
    /// Source.
    pub source: Option<String>,
    /// Copyright.
    pub copyright: Option<String>,
}

/// Generic descriptor element.
#[derive(Debug, Clone)]
pub struct Descriptor {
    /// Scheme ID URI.
    pub scheme_id_uri: String,
    /// Value.
    pub value: Option<String>,
    /// ID.
    pub id: Option<String>,
}

impl Descriptor {
    /// Creates a new descriptor.
    #[must_use]
    pub fn new(scheme_id_uri: impl Into<String>) -> Self {
        Self {
            scheme_id_uri: scheme_id_uri.into(),
            value: None,
            id: None,
        }
    }
}

/// Content protection element.
#[derive(Debug, Clone)]
pub struct ContentProtection {
    /// Scheme ID URI.
    pub scheme_id_uri: String,
    /// Value.
    pub value: Option<String>,
    /// Default key ID.
    pub default_kid: Option<String>,
    /// PSSH data (base64).
    pub pssh: Option<String>,
}

impl ContentProtection {
    /// Creates a new content protection element.
    #[must_use]
    pub fn new(scheme_id_uri: impl Into<String>) -> Self {
        Self {
            scheme_id_uri: scheme_id_uri.into(),
            value: None,
            default_kid: None,
            pssh: None,
        }
    }

    /// Returns true if this is Widevine DRM.
    #[must_use]
    pub fn is_widevine(&self) -> bool {
        self.scheme_id_uri
            .contains("edef8ba9-79d6-4ace-a3c8-27dcd51d21ed")
    }

    /// Returns true if this is PlayReady DRM.
    #[must_use]
    pub fn is_playready(&self) -> bool {
        self.scheme_id_uri
            .contains("9a04f079-9840-4286-ab92-e65be0885f95")
    }
}

/// Segment timeline entry (S element).
#[derive(Debug, Clone, Copy)]
pub struct SegmentTimelineEntry {
    /// Start time (t attribute).
    pub start: Option<u64>,
    /// Duration (d attribute).
    pub duration: u64,
    /// Repeat count (r attribute, -1 means repeat until end).
    pub repeat: i32,
}

impl SegmentTimelineEntry {
    /// Creates a new timeline entry.
    #[must_use]
    pub const fn new(duration: u64) -> Self {
        Self {
            start: None,
            duration,
            repeat: 0,
        }
    }

    /// Sets the start time.
    #[must_use]
    pub const fn with_start(mut self, start: u64) -> Self {
        self.start = Some(start);
        self
    }

    /// Sets the repeat count.
    #[must_use]
    pub const fn with_repeat(mut self, repeat: i32) -> Self {
        self.repeat = repeat;
        self
    }

    /// Returns the number of segments this entry represents.
    #[must_use]
    pub const fn segment_count(&self) -> u32 {
        if self.repeat < 0 {
            u32::MAX
        } else {
            (self.repeat + 1) as u32
        }
    }
}

/// Segment timeline (SegmentTimeline element).
#[derive(Debug, Clone, Default)]
pub struct SegmentTimeline {
    /// Timeline entries.
    pub entries: Vec<SegmentTimelineEntry>,
}

impl SegmentTimeline {
    /// Creates a new empty timeline.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Adds an entry to the timeline.
    pub fn add_entry(&mut self, entry: SegmentTimelineEntry) {
        self.entries.push(entry);
    }

    /// Returns the total duration in timescale units.
    #[must_use]
    pub fn total_duration(&self) -> u64 {
        let mut total = 0u64;
        for entry in &self.entries {
            let count = if entry.repeat < 0 {
                1
            } else {
                (entry.repeat + 1) as u64
            };
            total += entry.duration * count;
        }
        total
    }

    /// Iterates over all segment start times and durations.
    pub fn iter_segments(&self) -> impl Iterator<Item = (u64, u64)> + '_ {
        SegmentTimelineIterator {
            entries: &self.entries,
            entry_idx: 0,
            repeat_idx: 0,
            current_time: 0,
        }
    }
}

struct SegmentTimelineIterator<'a> {
    entries: &'a [SegmentTimelineEntry],
    entry_idx: usize,
    repeat_idx: i32,
    current_time: u64,
}

impl Iterator for SegmentTimelineIterator<'_> {
    type Item = (u64, u64);

    fn next(&mut self) -> Option<Self::Item> {
        while self.entry_idx < self.entries.len() {
            let entry = &self.entries[self.entry_idx];

            // Set start time from entry if this is the first segment of this entry
            if self.repeat_idx == 0 {
                if let Some(start) = entry.start {
                    self.current_time = start;
                }
            }

            if entry.repeat < 0 || self.repeat_idx <= entry.repeat {
                let start = self.current_time;
                let duration = entry.duration;
                self.current_time += duration;
                self.repeat_idx += 1;

                if entry.repeat >= 0 && self.repeat_idx > entry.repeat {
                    self.entry_idx += 1;
                    self.repeat_idx = 0;
                }

                return Some((start, duration));
            }

            self.entry_idx += 1;
            self.repeat_idx = 0;
        }
        None
    }
}

/// Segment base information.
#[derive(Debug, Clone, Default)]
pub struct SegmentBase {
    /// Timescale.
    pub timescale: Option<u32>,
    /// Presentation time offset.
    pub presentation_time_offset: Option<u64>,
    /// Index range.
    pub index_range: Option<(u64, u64)>,
    /// Initialization URL.
    pub initialization: Option<UrlType>,
    /// Representation index URL.
    pub representation_index: Option<UrlType>,
}

/// Segment list element.
#[derive(Debug, Clone, Default)]
pub struct SegmentList {
    /// Timescale.
    pub timescale: Option<u32>,
    /// Duration per segment.
    pub duration: Option<u64>,
    /// Start number.
    pub start_number: Option<u64>,
    /// Initialization URL.
    pub initialization: Option<UrlType>,
    /// Segment URLs.
    pub segment_urls: Vec<UrlType>,
}

/// Segment template element.
#[derive(Debug, Clone, Default)]
pub struct SegmentTemplate {
    /// Timescale.
    pub timescale: u32,
    /// Duration per segment.
    pub duration: Option<u64>,
    /// Start number.
    pub start_number: u64,
    /// Presentation time offset.
    pub presentation_time_offset: Option<u64>,
    /// Media URL template.
    pub media: Option<String>,
    /// Initialization URL template.
    pub initialization: Option<String>,
    /// Segment timeline.
    pub segment_timeline: Option<SegmentTimeline>,
}

impl SegmentTemplate {
    /// Creates a new segment template.
    #[must_use]
    pub fn new(timescale: u32) -> Self {
        Self {
            timescale,
            start_number: 1,
            ..Default::default()
        }
    }

    /// Sets the media template.
    #[must_use]
    pub fn with_media(mut self, media: impl Into<String>) -> Self {
        self.media = Some(media.into());
        self
    }

    /// Sets the initialization template.
    #[must_use]
    pub fn with_initialization(mut self, init: impl Into<String>) -> Self {
        self.initialization = Some(init.into());
        self
    }

    /// Generates a media URL for a given segment.
    #[must_use]
    pub fn media_url(
        &self,
        representation_id: &str,
        number: u64,
        time: Option<u64>,
    ) -> Option<String> {
        let template = self.media.as_ref()?;
        let url = substitute_template(template, representation_id, number, time, self.timescale);
        Some(url)
    }

    /// Generates an initialization URL.
    #[must_use]
    pub fn initialization_url(&self, representation_id: &str) -> Option<String> {
        let template = self.initialization.as_ref()?;
        let url = substitute_template(template, representation_id, 0, None, self.timescale);
        Some(url)
    }

    /// Returns the segment duration in seconds.
    #[must_use]
    pub fn segment_duration_secs(&self) -> Option<f64> {
        self.duration.map(|d| d as f64 / self.timescale as f64)
    }
}

/// Substitutes template variables.
fn substitute_template(
    template: &str,
    representation_id: &str,
    number: u64,
    time: Option<u64>,
    bandwidth: u32,
) -> String {
    let mut result = template.to_string();

    // Simple substitution (without format specifiers)
    result = result.replace("$RepresentationID$", representation_id);
    result = result.replace("$Number$", &number.to_string());
    result = result.replace("$Bandwidth$", &bandwidth.to_string());

    if let Some(t) = time {
        result = result.replace("$Time$", &t.to_string());
    }

    // Handle format specifiers like $Number%05d$
    result = substitute_with_format(&result, "Number", number);
    result = substitute_with_format(&result, "Time", time.unwrap_or(0));

    result
}

/// Maximum zero-pad width honoured in a `$Number%0Nd$` style template.
///
/// Real segment templates never need more than a couple of digits. Capping the
/// width stops a hostile MPD from requesting e.g. `%0999999999d$` and forcing a
/// multi-gigabyte formatted string.
const MAX_TEMPLATE_WIDTH: usize = 32;

fn substitute_with_format(s: &str, var: &str, value: u64) -> String {
    let pattern = format!("${var}%");
    let mut result = s.to_string();
    let mut search_start = 0;

    // `search_start` strictly increases on every iteration, so this always
    // terminates. The previous version hung forever on a standard
    // `$Number%0Nd$` template: an off-by-one made `format_spec` include the
    // trailing `d`, so the width parsed to 0, the rebuilt `full_pattern` never
    // matched `result`, and the cursor was never advanced.
    while let Some(rel) = result[search_start..].find(&pattern) {
        let abs_start = search_start + rel;
        let after_pattern = abs_start + pattern.len();
        if let Some(rel_end) = result[after_pattern..].find("d$") {
            // Width digits sit between '%' and 'd', e.g. "05" in "%05d$".
            let format_spec = &result[after_pattern..after_pattern + rel_end];
            let zero_pad = format_spec.starts_with('0');
            let width = format_spec
                .trim_start_matches('0')
                .parse::<usize>()
                .unwrap_or(0)
                .min(MAX_TEMPLATE_WIDTH);

            let formatted = if zero_pad && width > 0 {
                format!("{value:0>width$}")
            } else {
                format!("{value:>width$}")
            };

            // Replace exactly this `$Var%<spec>d$` occurrence and continue
            // scanning past the substituted text.
            let spec_end = after_pattern + rel_end + 2; // include the "d$"
            result.replace_range(abs_start..spec_end, &formatted);
            search_start = abs_start + formatted.len();
        } else {
            // No `d$` terminator for this '%': skip past it and keep scanning.
            search_start = after_pattern;
        }
    }

    result
}

/// Content component element.
#[derive(Debug, Clone, Default)]
pub struct ContentComponent {
    /// ID.
    pub id: Option<String>,
    /// Content type.
    pub content_type: Option<String>,
    /// Language.
    pub lang: Option<String>,
}

/// Representation element.
#[derive(Debug, Clone, Default)]
pub struct Representation {
    /// ID.
    pub id: String,
    /// Bandwidth in bits per second.
    pub bandwidth: u64,
    /// Width (video).
    pub width: Option<u32>,
    /// Height (video).
    pub height: Option<u32>,
    /// Frame rate.
    pub frame_rate: Option<String>,
    /// Sample rate (audio).
    pub audio_sampling_rate: Option<u32>,
    /// Codecs string.
    pub codecs: Option<String>,
    /// MIME type.
    pub mime_type: Option<String>,
    /// Segment base.
    pub segment_base: Option<SegmentBase>,
    /// Segment list.
    pub segment_list: Option<SegmentList>,
    /// Segment template.
    pub segment_template: Option<SegmentTemplate>,
    /// Base URLs.
    pub base_urls: Vec<String>,
    /// Content protection.
    pub content_protection: Vec<ContentProtection>,
}

impl Representation {
    /// Creates a new representation.
    #[must_use]
    pub fn new(id: impl Into<String>, bandwidth: u64) -> Self {
        Self {
            id: id.into(),
            bandwidth,
            ..Default::default()
        }
    }

    /// Returns the resolution if available.
    #[must_use]
    pub fn resolution(&self) -> Option<(u32, u32)> {
        match (self.width, self.height) {
            (Some(w), Some(h)) => Some((w, h)),
            _ => None,
        }
    }

    /// Returns true if this is a video representation.
    #[must_use]
    pub fn is_video(&self) -> bool {
        self.mime_type
            .as_ref()
            .is_some_and(|m| m.starts_with("video/"))
            || self.width.is_some()
    }

    /// Returns true if this is an audio representation.
    #[must_use]
    pub fn is_audio(&self) -> bool {
        self.mime_type
            .as_ref()
            .is_some_and(|m| m.starts_with("audio/"))
            || self.audio_sampling_rate.is_some()
    }
}

/// Adaptation set element.
#[derive(Debug, Clone, Default)]
pub struct AdaptationSet {
    /// ID.
    pub id: Option<u32>,
    /// Group.
    pub group: Option<u32>,
    /// Content type.
    pub content_type: Option<String>,
    /// Language.
    pub lang: Option<String>,
    /// MIME type.
    pub mime_type: Option<String>,
    /// Codecs.
    pub codecs: Option<String>,
    /// Width (video).
    pub width: Option<u32>,
    /// Height (video).
    pub height: Option<u32>,
    /// Frame rate.
    pub frame_rate: Option<String>,
    /// Audio sampling rate.
    pub audio_sampling_rate: Option<u32>,
    /// Segment alignment.
    pub segment_alignment: bool,
    /// Subsegment alignment.
    pub subsegment_alignment: bool,
    /// Bitstream switching.
    pub bitstream_switching: bool,
    /// Segment base.
    pub segment_base: Option<SegmentBase>,
    /// Segment list.
    pub segment_list: Option<SegmentList>,
    /// Segment template.
    pub segment_template: Option<SegmentTemplate>,
    /// Content components.
    pub content_components: Vec<ContentComponent>,
    /// Representations.
    pub representations: Vec<Representation>,
    /// Content protection.
    pub content_protection: Vec<ContentProtection>,
    /// Accessibility descriptors.
    pub accessibility: Vec<Descriptor>,
    /// Role descriptors.
    pub role: Vec<Descriptor>,
}

impl AdaptationSet {
    /// Creates a new adaptation set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns true if this is a video adaptation set.
    #[must_use]
    pub fn is_video(&self) -> bool {
        self.content_type.as_deref() == Some("video")
            || self
                .mime_type
                .as_ref()
                .is_some_and(|m| m.starts_with("video/"))
            || self.representations.iter().any(Representation::is_video)
    }

    /// Returns true if this is an audio adaptation set.
    #[must_use]
    pub fn is_audio(&self) -> bool {
        self.content_type.as_deref() == Some("audio")
            || self
                .mime_type
                .as_ref()
                .is_some_and(|m| m.starts_with("audio/"))
            || self.representations.iter().any(Representation::is_audio)
    }

    /// Returns true if this is a text/subtitle adaptation set.
    #[must_use]
    pub fn is_text(&self) -> bool {
        self.content_type.as_deref() == Some("text")
            || self
                .mime_type
                .as_ref()
                .is_some_and(|m| m.starts_with("text/"))
    }

    /// Returns representations sorted by bandwidth.
    #[must_use]
    pub fn representations_by_bandwidth(&self) -> Vec<&Representation> {
        let mut reps: Vec<_> = self.representations.iter().collect();
        reps.sort_by_key(|r| r.bandwidth);
        reps
    }
}

/// Period element.
#[derive(Debug, Clone, Default)]
pub struct Period {
    /// ID.
    pub id: Option<String>,
    /// Start time.
    pub start: Option<Duration>,
    /// Duration.
    pub duration: Option<Duration>,
    /// Bitstream switching.
    pub bitstream_switching: bool,
    /// Segment base.
    pub segment_base: Option<SegmentBase>,
    /// Segment list.
    pub segment_list: Option<SegmentList>,
    /// Segment template.
    pub segment_template: Option<SegmentTemplate>,
    /// Adaptation sets.
    pub adaptation_sets: Vec<AdaptationSet>,
    /// Base URLs.
    pub base_urls: Vec<String>,
}

impl Period {
    /// Creates a new period.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns video adaptation sets.
    #[must_use]
    pub fn video_adaptation_sets(&self) -> Vec<&AdaptationSet> {
        self.adaptation_sets
            .iter()
            .filter(|a| a.is_video())
            .collect()
    }

    /// Returns audio adaptation sets.
    #[must_use]
    pub fn audio_adaptation_sets(&self) -> Vec<&AdaptationSet> {
        self.adaptation_sets
            .iter()
            .filter(|a| a.is_audio())
            .collect()
    }

    /// Returns text/subtitle adaptation sets.
    #[must_use]
    pub fn text_adaptation_sets(&self) -> Vec<&AdaptationSet> {
        self.adaptation_sets
            .iter()
            .filter(|a| a.is_text())
            .collect()
    }
}

/// Media Presentation Description (MPD).
#[derive(Debug, Clone, Default)]
pub struct Mpd {
    /// MPD type (static/dynamic).
    pub mpd_type: MpdType,
    /// Minimum buffer time.
    pub min_buffer_time: Duration,
    /// Media presentation duration.
    pub media_presentation_duration: Option<Duration>,
    /// Availability start time (ISO 8601).
    pub availability_start_time: Option<String>,
    /// Availability end time (ISO 8601).
    pub availability_end_time: Option<String>,
    /// Publish time (ISO 8601).
    pub publish_time: Option<String>,
    /// Minimum update period.
    pub minimum_update_period: Option<Duration>,
    /// Suggested presentation delay.
    pub suggested_presentation_delay: Option<Duration>,
    /// Time shift buffer depth.
    pub time_shift_buffer_depth: Option<Duration>,
    /// Profiles.
    pub profiles: Vec<String>,
    /// Base URLs.
    pub base_urls: Vec<String>,
    /// Program information.
    pub program_information: Option<ProgramInformation>,
    /// Periods.
    pub periods: Vec<Period>,
}

impl Mpd {
    /// Creates a new empty MPD.
    #[must_use]
    pub fn new() -> Self {
        Self {
            min_buffer_time: Duration::from_secs(2),
            ..Default::default()
        }
    }

    /// Parses an MPD from XML string.
    ///
    /// # Errors
    ///
    /// Returns an error if the XML is malformed.
    ///
    /// Note: This is a skeleton implementation. Full XML parsing would
    /// require an XML library like quick-xml.
    pub fn parse(xml: &str) -> NetResult<Self> {
        // Basic validation
        if !xml.contains("<MPD") {
            return Err(NetError::parse(0, "Missing MPD root element"));
        }

        let mut mpd = Self::new();

        // Parse type attribute
        if let Some(type_value) = extract_attribute(xml, "MPD", "type") {
            mpd.mpd_type = MpdType::from_str(&type_value);
        }

        // Parse minBufferTime
        if let Some(mbt) = extract_attribute(xml, "MPD", "minBufferTime") {
            if let Some(dur) = parse_iso8601_duration(&mbt) {
                mpd.min_buffer_time = dur;
            }
        }

        // Parse mediaPresentationDuration
        if let Some(mpd_dur) = extract_attribute(xml, "MPD", "mediaPresentationDuration") {
            mpd.media_presentation_duration = parse_iso8601_duration(&mpd_dur);
        }

        // Parse profiles
        if let Some(profiles) = extract_attribute(xml, "MPD", "profiles") {
            mpd.profiles = profiles.split(',').map(|s| s.trim().to_string()).collect();
        }

        Ok(mpd)
    }

    /// Returns true if this is a live presentation.
    #[must_use]
    pub const fn is_live(&self) -> bool {
        self.mpd_type.is_live()
    }

    /// Returns the total duration if known.
    #[must_use]
    pub fn duration(&self) -> Option<Duration> {
        self.media_presentation_duration
    }

    /// Returns the first period.
    #[must_use]
    pub fn first_period(&self) -> Option<&Period> {
        self.periods.first()
    }
}

/// Extracts an attribute value from a simple XML element (basic implementation).
fn extract_attribute(xml: &str, element: &str, attr: &str) -> Option<String> {
    let element_start = xml.find(&format!("<{element}"))?;
    let element_end = xml[element_start..].find('>')? + element_start;
    let element_str = &xml[element_start..element_end];

    let attr_pattern = format!("{attr}=\"");
    let attr_start = element_str.find(&attr_pattern)? + attr_pattern.len();
    let attr_end = element_str[attr_start..].find('"')? + attr_start;

    Some(element_str[attr_start..attr_end].to_string())
}

/// Parses an ISO 8601 duration string (e.g., "PT10S", "PT1H30M").
///
/// Supports: P[nY][nM][nD][T[nH][nM][nS]]
#[must_use]
pub fn parse_iso8601_duration(s: &str) -> Option<Duration> {
    let s = s.trim();
    if !s.starts_with('P') {
        return None;
    }

    let s = &s[1..];
    let mut total_secs = 0.0;
    let mut in_time = false;
    let mut num_str = String::new();

    for ch in s.chars() {
        match ch {
            'T' => in_time = true,
            'Y' if !in_time => {
                if let Ok(n) = num_str.parse::<f64>() {
                    total_secs += n * 365.25 * 24.0 * 60.0 * 60.0;
                }
                num_str.clear();
            }
            'M' if !in_time => {
                if let Ok(n) = num_str.parse::<f64>() {
                    total_secs += n * 30.0 * 24.0 * 60.0 * 60.0;
                }
                num_str.clear();
            }
            'D' => {
                if let Ok(n) = num_str.parse::<f64>() {
                    total_secs += n * 24.0 * 60.0 * 60.0;
                }
                num_str.clear();
            }
            'H' => {
                if let Ok(n) = num_str.parse::<f64>() {
                    total_secs += n * 60.0 * 60.0;
                }
                num_str.clear();
            }
            'M' if in_time => {
                if let Ok(n) = num_str.parse::<f64>() {
                    total_secs += n * 60.0;
                }
                num_str.clear();
            }
            'S' => {
                if let Ok(n) = num_str.parse::<f64>() {
                    total_secs += n;
                }
                num_str.clear();
            }
            c if c.is_ascii_digit() || c == '.' => num_str.push(c),
            _ => {}
        }
    }

    if total_secs > 0.0 {
        Some(Duration::from_secs_f64(total_secs))
    } else {
        None
    }
}

/// Formats a duration as ISO 8601.
#[must_use]
#[allow(dead_code)]
pub fn format_iso8601_duration(dur: Duration) -> String {
    let total_secs = dur.as_secs_f64();

    if total_secs < 60.0 {
        return format!("PT{total_secs:.3}S");
    }

    let hours = (total_secs / 3600.0).floor() as u64;
    let minutes = ((total_secs % 3600.0) / 60.0).floor() as u64;
    let seconds = total_secs % 60.0;

    let mut result = String::from("PT");
    if hours > 0 {
        result.push_str(&format!("{hours}H"));
    }
    if minutes > 0 {
        result.push_str(&format!("{minutes}M"));
    }
    if seconds > 0.0 || result == "PT" {
        result.push_str(&format!("{seconds:.3}S"));
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mpd_type() {
        assert!(!MpdType::Static.is_live());
        assert!(MpdType::Dynamic.is_live());
        assert_eq!(MpdType::from_str("dynamic"), MpdType::Dynamic);
        assert_eq!(MpdType::from_str("static"), MpdType::Static);
    }

    #[test]
    fn test_parse_iso8601_duration() {
        assert_eq!(
            parse_iso8601_duration("PT10S"),
            Some(Duration::from_secs(10))
        );
        assert_eq!(
            parse_iso8601_duration("PT1M30S"),
            Some(Duration::from_secs(90))
        );
        assert_eq!(
            parse_iso8601_duration("PT1H"),
            Some(Duration::from_secs(3600))
        );
        assert_eq!(
            parse_iso8601_duration("PT1H30M45S"),
            Some(Duration::from_secs(5445))
        );
        assert_eq!(
            parse_iso8601_duration("P1D"),
            Some(Duration::from_secs(86400))
        );

        // Fractional seconds
        let dur = parse_iso8601_duration("PT10.5S").expect("should succeed in test");
        assert!((dur.as_secs_f64() - 10.5).abs() < 0.001);
    }

    #[test]
    fn test_format_iso8601_duration() {
        assert_eq!(
            format_iso8601_duration(Duration::from_secs(10)),
            "PT10.000S"
        );
        assert_eq!(
            format_iso8601_duration(Duration::from_secs(90)),
            "PT1M30.000S"
        );
        // 3600 seconds = 1 hour with 0 minutes and 0 seconds
        assert_eq!(format_iso8601_duration(Duration::from_secs(3600)), "PT1H");
    }

    #[test]
    fn test_segment_template() {
        let template = SegmentTemplate::new(90000)
            .with_media("video_$RepresentationID$_$Number$.m4s")
            .with_initialization("video_$RepresentationID$_init.mp4");

        let media_url = template
            .media_url("720p", 1, None)
            .expect("should succeed in test");
        assert_eq!(media_url, "video_720p_1.m4s");

        let init_url = template
            .initialization_url("720p")
            .expect("should succeed in test");
        assert_eq!(init_url, "video_720p_init.mp4");
    }

    #[test]
    fn test_segment_template_with_time() {
        let template = SegmentTemplate::new(90000).with_media("segment_$Time$.m4s");

        let url = template
            .media_url("v1", 1, Some(900000))
            .expect("should succeed in test");
        assert_eq!(url, "segment_900000.m4s");
    }

    #[test]
    fn test_segment_timeline() {
        let mut timeline = SegmentTimeline::new();
        timeline.add_entry(SegmentTimelineEntry::new(90000).with_start(0));
        timeline.add_entry(SegmentTimelineEntry::new(90000).with_repeat(2));

        let segments: Vec<_> = timeline.iter_segments().collect();
        assert_eq!(segments.len(), 4);
        assert_eq!(segments[0], (0, 90000));
        assert_eq!(segments[1], (90000, 90000));
    }

    #[test]
    fn test_representation() {
        let rep = Representation::new("720p", 1_500_000);
        assert_eq!(rep.id, "720p");
        assert_eq!(rep.bandwidth, 1_500_000);
    }

    #[test]
    fn test_adaptation_set_type() {
        let mut video_as = AdaptationSet::new();
        video_as.content_type = Some("video".to_string());
        assert!(video_as.is_video());
        assert!(!video_as.is_audio());

        let mut audio_as = AdaptationSet::new();
        audio_as.mime_type = Some("audio/mp4".to_string());
        assert!(audio_as.is_audio());
        assert!(!audio_as.is_video());
    }

    #[test]
    fn test_mpd_parse_basic() {
        let xml = r#"<?xml version="1.0"?>
            <MPD type="static" minBufferTime="PT2S" mediaPresentationDuration="PT1H30M">
            </MPD>"#;

        let mpd = Mpd::parse(xml).expect("should succeed in test");
        assert_eq!(mpd.mpd_type, MpdType::Static);
        assert_eq!(mpd.min_buffer_time, Duration::from_secs(2));
        assert_eq!(
            mpd.media_presentation_duration,
            Some(Duration::from_secs(5400))
        );
    }

    #[test]
    fn test_mpd_parse_live() {
        let xml = r#"<MPD type="dynamic" minBufferTime="PT4S"></MPD>"#;

        let mpd = Mpd::parse(xml).expect("should succeed in test");
        assert!(mpd.is_live());
        assert_eq!(mpd.min_buffer_time, Duration::from_secs(4));
    }

    #[test]
    fn test_url_type() {
        let url = UrlType::new("init.mp4").with_range(0, 999);
        assert_eq!(url.source_url, Some("init.mp4".to_string()));
        assert_eq!(url.range, Some((0, 999)));
    }

    #[test]
    fn test_content_protection() {
        let cp = ContentProtection::new("urn:uuid:edef8ba9-79d6-4ace-a3c8-27dcd51d21ed");
        assert!(cp.is_widevine());
        assert!(!cp.is_playready());
    }

    #[test]
    fn segment_template_number_format_spec_terminates() {
        // A standard `$Number%0Nd$` template previously hung the worker forever.
        let template = SegmentTemplate::new(90000).with_media("seg_$Number%05d$.m4s");
        let url = template.media_url("v1", 42, None).expect("must terminate");
        assert_eq!(url, "seg_00042.m4s");
    }

    #[test]
    fn substitute_with_format_basic_and_termination() {
        assert_eq!(
            substitute_with_format("$Number%05d$", "Number", 42),
            "00042"
        );
        // Space padding (no leading zero in the spec).
        assert_eq!(substitute_with_format("$Number%5d$", "Number", 42), "   42");
        // No format spec: returned unchanged (and must terminate).
        assert_eq!(substitute_with_format("plain", "Number", 1), "plain");
        // Dangling '%' with no `d$` terminator must not loop forever.
        assert_eq!(
            substitute_with_format("$Number%zzz", "Number", 1),
            "$Number%zzz"
        );
    }

    #[test]
    fn substitute_with_format_width_is_capped() {
        // A hostile width must not allocate a multi-gigabyte string.
        let out = substitute_with_format("$Number%0999999999d$", "Number", 7);
        assert!(out.len() <= MAX_TEMPLATE_WIDTH);
    }
}
