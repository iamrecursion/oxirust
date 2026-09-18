//! MPD (Media Presentation Description) builder.
//!
//! Generates DASH manifest files.

use chrono::{DateTime, Utc};
use std::time::Duration;

/// Representation (quality variant).
#[derive(Debug, Clone)]
pub struct Representation {
    /// Representation ID.
    pub id: String,

    /// Bandwidth in bits per second.
    pub bandwidth: u64,

    /// Width in pixels.
    pub width: u32,

    /// Height in pixels.
    pub height: u32,

    /// Codec string.
    pub codecs: String,

    /// Initialization segment URL.
    pub init_url: String,

    /// Media segment template.
    pub media_template: String,
}

/// MPD builder.
pub struct MpdBuilder {
    /// MPD type (static or dynamic).
    mpd_type: String,

    /// Availability start time (for live).
    availability_start_time: Option<DateTime<Utc>>,

    /// Minimum buffer time.
    min_buffer_time: Duration,

    /// Suggested presentation delay.
    suggested_presentation_delay: Option<Duration>,

    /// Time shift buffer depth (DVR window).
    time_shift_buffer_depth: Option<Duration>,

    /// Video representations.
    video_representations: Vec<Representation>,

    /// Audio representations.
    audio_representations: Vec<Representation>,

    /// Segment duration.
    segment_duration: Duration,

    /// Low latency configuration.
    low_latency: Option<Duration>,
}

impl MpdBuilder {
    /// Creates a new MPD builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            mpd_type: "static".to_string(),
            availability_start_time: None,
            min_buffer_time: Duration::from_secs(2),
            suggested_presentation_delay: None,
            time_shift_buffer_depth: None,
            video_representations: Vec::new(),
            audio_representations: Vec::new(),
            segment_duration: Duration::from_secs(2),
            low_latency: None,
        }
    }

    /// Sets MPD type to live/dynamic.
    #[must_use]
    pub fn live(mut self) -> Self {
        self.mpd_type = "dynamic".to_string();
        self.availability_start_time = Some(Utc::now());
        self
    }

    /// Sets availability start time.
    #[must_use]
    pub fn availability_start_time(mut self, time: DateTime<Utc>) -> Self {
        self.availability_start_time = Some(time);
        self
    }

    /// Sets minimum buffer time.
    #[must_use]
    pub const fn min_buffer_time(mut self, duration: Duration) -> Self {
        self.min_buffer_time = duration;
        self
    }

    /// Sets suggested presentation delay.
    #[must_use]
    pub fn suggested_presentation_delay(mut self, duration: Duration) -> Self {
        self.suggested_presentation_delay = Some(duration);
        self
    }

    /// Sets time shift buffer depth (DVR window).
    #[must_use]
    pub fn time_shift_buffer_depth(mut self, duration: Duration) -> Self {
        self.time_shift_buffer_depth = Some(duration);
        self
    }

    /// Sets segment duration.
    #[must_use]
    pub const fn segment_duration(mut self, duration: Duration) -> Self {
        self.segment_duration = duration;
        self
    }

    /// Enables low latency with chunk duration.
    #[must_use]
    pub fn low_latency(mut self, chunk_duration: Duration) -> Self {
        self.low_latency = Some(chunk_duration);
        self
    }

    /// Adds a video representation.
    #[must_use]
    pub fn add_video_representation(
        mut self,
        id: String,
        bandwidth: u64,
        width: u32,
        height: u32,
        codecs: String,
    ) -> Self {
        self.video_representations.push(Representation {
            id: id.clone(),
            bandwidth,
            width,
            height,
            codecs,
            init_url: format!("init_video_{id}.mp4"),
            media_template: format!("video_$Number$_{id}.m4s"),
        });
        self
    }

    /// Adds an audio representation.
    #[must_use]
    pub fn add_audio_representation(mut self, id: String, bandwidth: u64, codecs: String) -> Self {
        self.audio_representations.push(Representation {
            id: id.clone(),
            bandwidth,
            width: 0,
            height: 0,
            codecs,
            init_url: format!("init_audio_{id}.mp4"),
            media_template: format!("audio_$Number$_{id}.m4s"),
        });
        self
    }

    /// Builds the MPD XML.
    #[must_use]
    pub fn build(self) -> String {
        let mut mpd = String::new();

        mpd.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        mpd.push_str("<MPD xmlns=\"urn:mpeg:dash:schema:mpd:2011\" ");
        mpd.push_str("xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" ");
        mpd.push_str("xsi:schemaLocation=\"urn:mpeg:dash:schema:mpd:2011 http://standards.iso.org/ittf/PubliclyAvailableStandards/MPEG-DASH_schema_files/DASH-MPD.xsd\" ");
        mpd.push_str(&format!("type=\"{}\" ", self.mpd_type));

        if let Some(start_time) = self.availability_start_time {
            mpd.push_str(&format!(
                "availabilityStartTime=\"{}\" ",
                start_time.to_rfc3339()
            ));
        }

        mpd.push_str(&format!(
            "minBufferTime=\"PT{}S\" ",
            self.min_buffer_time.as_secs()
        ));

        if let Some(delay) = self.suggested_presentation_delay {
            mpd.push_str(&format!(
                "suggestedPresentationDelay=\"PT{}S\" ",
                delay.as_secs()
            ));
        }

        if let Some(depth) = self.time_shift_buffer_depth {
            mpd.push_str(&format!("timeShiftBufferDepth=\"PT{}S\" ", depth.as_secs()));
        }

        mpd.push_str("profiles=\"urn:mpeg:dash:profile:isoff-live:2011\">\n");

        // Period
        mpd.push_str("  <Period>\n");

        // Video adaptation set
        if !self.video_representations.is_empty() {
            mpd.push_str("    <AdaptationSet mimeType=\"video/mp4\" ");
            mpd.push_str("segmentAlignment=\"true\" ");
            mpd.push_str("startWithSAP=\"1\">\n");

            for repr in &self.video_representations {
                mpd.push_str(&format!("      <Representation id=\"{}\" ", repr.id));
                mpd.push_str(&format!("bandwidth=\"{}\" ", repr.bandwidth));
                mpd.push_str(&format!("width=\"{}\" ", repr.width));
                mpd.push_str(&format!("height=\"{}\" ", repr.height));
                mpd.push_str(&format!("codecs=\"{}\">\n", repr.codecs));

                // Segment template
                mpd.push_str("        <SegmentTemplate ");
                mpd.push_str(&format!(
                    "timescale=\"1000\" duration=\"{}\" ",
                    self.segment_duration.as_millis()
                ));
                mpd.push_str(&format!("initialization=\"{}\" ", repr.init_url));
                mpd.push_str(&format!("media=\"{}\" ", repr.media_template));
                mpd.push_str("startNumber=\"1\"/>\n");

                mpd.push_str("      </Representation>\n");
            }

            mpd.push_str("    </AdaptationSet>\n");
        }

        // Audio adaptation set
        if !self.audio_representations.is_empty() {
            mpd.push_str("    <AdaptationSet mimeType=\"audio/mp4\" ");
            mpd.push_str("segmentAlignment=\"true\" ");
            mpd.push_str("startWithSAP=\"1\">\n");

            for repr in &self.audio_representations {
                mpd.push_str(&format!("      <Representation id=\"{}\" ", repr.id));
                mpd.push_str(&format!("bandwidth=\"{}\" ", repr.bandwidth));
                mpd.push_str(&format!("codecs=\"{}\">\n", repr.codecs));

                // Segment template
                mpd.push_str("        <SegmentTemplate ");
                mpd.push_str(&format!(
                    "timescale=\"1000\" duration=\"{}\" ",
                    self.segment_duration.as_millis()
                ));
                mpd.push_str(&format!("initialization=\"{}\" ", repr.init_url));
                mpd.push_str(&format!("media=\"{}\" ", repr.media_template));
                mpd.push_str("startNumber=\"1\"/>\n");

                mpd.push_str("      </Representation>\n");
            }

            mpd.push_str("    </AdaptationSet>\n");
        }

        mpd.push_str("  </Period>\n");
        mpd.push_str("</MPD>\n");

        mpd
    }
}

impl Default for MpdBuilder {
    fn default() -> Self {
        Self::new()
    }
}
