//! DASH MPD (Media Presentation Description) manifest generation.

use crate::dash::DashConfig;
use crate::error::ServerResult;
use parking_lot::RwLock;
use std::collections::VecDeque;

/// Period in MPD.
#[derive(Debug, Clone)]
pub struct Period {
    /// Period ID.
    pub id: String,

    /// Start time (optional).
    pub start: Option<f64>,

    /// Duration (optional).
    pub duration: Option<f64>,

    /// Adaptation sets.
    pub adaptation_sets: Vec<AdaptationSet>,
}

/// Adaptation set (audio or video).
#[derive(Debug, Clone)]
pub struct AdaptationSet {
    /// ID.
    pub id: u32,

    /// Content type (video/audio).
    pub content_type: String,

    /// MIME type.
    pub mime_type: String,

    /// Codecs.
    pub codecs: String,

    /// Representations.
    pub representations: Vec<Representation>,
}

/// Representation (quality level).
#[derive(Debug, Clone)]
pub struct Representation {
    /// ID.
    pub id: String,

    /// Bandwidth in bits per second.
    pub bandwidth: u64,

    /// Width (for video).
    pub width: Option<u32>,

    /// Height (for video).
    pub height: Option<u32>,

    /// Frame rate (for video).
    pub frame_rate: Option<f64>,

    /// Initialization segment.
    pub initialization: String,

    /// Media segments.
    pub media: String,

    /// Start number.
    pub start_number: u64,
}

/// Segment timeline entry.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct SegmentTimelineEntry {
    /// Segment number.
    number: u64,

    /// Duration.
    duration: f64,

    /// Start time.
    time: f64,
}

/// MPD generator.
#[allow(dead_code)]
pub struct MpdGenerator {
    /// Configuration.
    config: DashConfig,

    /// Periods.
    periods: RwLock<Vec<Period>>,

    /// Segment timeline.
    timeline: RwLock<VecDeque<SegmentTimelineEntry>>,

    /// Current time.
    current_time: RwLock<f64>,
}

impl MpdGenerator {
    /// Creates a new MPD generator.
    #[must_use]
    pub fn new(config: DashConfig) -> Self {
        Self {
            config,
            periods: RwLock::new(Vec::new()),
            timeline: RwLock::new(VecDeque::new()),
            current_time: RwLock::new(0.0),
        }
    }

    /// Adds a segment to the timeline.
    pub fn add_segment(&self, number: u64, duration: f64) -> ServerResult<()> {
        let time = *self.current_time.read();

        let entry = SegmentTimelineEntry {
            number,
            duration,
            time,
        };

        let mut timeline = self.timeline.write();
        timeline.push_back(entry);

        // Remove old segments based on time shift buffer depth
        if let Some(buffer_depth) = self.config.time_shift_buffer_depth {
            let buffer_secs = buffer_depth.as_secs_f64();
            while !timeline.is_empty() {
                if let Some(first) = timeline.front() {
                    if time - first.time > buffer_secs {
                        timeline.pop_front();
                    } else {
                        break;
                    }
                }
            }
        }

        *self.current_time.write() = time + duration;

        Ok(())
    }

    /// Generates the MPD XML.
    #[must_use]
    pub fn generate(&self) -> String {
        let mut mpd = String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        mpd.push_str("<MPD xmlns=\"urn:mpeg:dash:schema:mpd:2011\" ");
        mpd.push_str("type=\"dynamic\" ");
        mpd.push_str(&format!(
            "minBufferTime=\"PT{}S\" ",
            self.config.min_buffer_time.as_secs()
        ));

        if let Some(depth) = self.config.time_shift_buffer_depth {
            mpd.push_str(&format!("timeShiftBufferDepth=\"PT{}S\" ", depth.as_secs()));
        }

        mpd.push_str(&format!(
            "suggestedPresentationDelay=\"PT{}S\" ",
            self.config.suggested_presentation_delay.as_secs()
        ));

        mpd.push_str("profiles=\"urn:mpeg:dash:profile:isoff-live:2011\">\n");

        // Add Period
        mpd.push_str("  <Period id=\"0\">\n");

        // Add video adaptation set
        mpd.push_str("    <AdaptationSet id=\"1\" contentType=\"video\" ");
        mpd.push_str("mimeType=\"video/mp4\" codecs=\"av01.0.05M.08\">\n");

        // Add representation
        mpd.push_str("      <Representation id=\"1080p\" bandwidth=\"4500000\" ");
        mpd.push_str("width=\"1920\" height=\"1080\" frameRate=\"30\">\n");

        mpd.push_str("        <SegmentTemplate timescale=\"1000\" ");
        mpd.push_str("initialization=\"init.mp4\" ");
        mpd.push_str("media=\"segment$Number$.m4s\" startNumber=\"1\">\n");

        // Add segment timeline
        mpd.push_str("          <SegmentTimeline>\n");

        let timeline = self.timeline.read();
        for entry in timeline.iter() {
            mpd.push_str(&format!(
                "            <S t=\"{}\" d=\"{}\" />\n",
                (entry.time * 1000.0) as u64,
                (entry.duration * 1000.0) as u64
            ));
        }

        mpd.push_str("          </SegmentTimeline>\n");
        mpd.push_str("        </SegmentTemplate>\n");
        mpd.push_str("      </Representation>\n");
        mpd.push_str("    </AdaptationSet>\n");

        mpd.push_str("  </Period>\n");
        mpd.push_str("</MPD>\n");

        mpd
    }
}
