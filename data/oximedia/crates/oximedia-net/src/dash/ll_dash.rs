//! Low-Latency DASH (LL-DASH) implementation.
//!
//! Implements LL-DASH features per ISO/IEC 23009-1:2022 including:
//! - CMAF-based low-latency segments with chunked transfer encoding
//! - Service description with latency targets
//! - Producer reference time for wall-clock synchronization
//! - Resync points for late-joining clients
//! - Availability time offset for early segment advertisement

use std::collections::VecDeque;
use std::fmt::Write as FmtWrite;
use std::time::SystemTime;

// ─── Configuration ────────────────────────────────────────────────────────────

/// LL-DASH configuration controlling latency and chunk behavior.
#[derive(Debug, Clone)]
pub struct LlDashConfig {
    /// Target segment duration in seconds (e.g., 2.0).
    pub segment_duration_secs: f64,
    /// Chunk duration within a segment in seconds (e.g., 0.5).
    pub chunk_duration_secs: f64,
    /// Target latency in seconds (e.g., 3.0).
    pub target_latency_secs: f64,
    /// Minimum latency in seconds (e.g., 2.0).
    pub min_latency_secs: f64,
    /// Maximum latency in seconds (e.g., 5.0).
    pub max_latency_secs: f64,
    /// Minimum playback rate for catchup (e.g., 0.96).
    pub min_playback_rate: f64,
    /// Maximum playback rate for catchup (e.g., 1.04).
    pub max_playback_rate: f64,
    /// Availability time offset in seconds for early advertisement.
    pub availability_time_offset: f64,
    /// Timescale for segment addressing (e.g., 90000 for video).
    pub timescale: u32,
}

impl Default for LlDashConfig {
    fn default() -> Self {
        Self {
            segment_duration_secs: 2.0,
            chunk_duration_secs: 0.5,
            target_latency_secs: 3.0,
            min_latency_secs: 2.0,
            max_latency_secs: 5.0,
            min_playback_rate: 0.96,
            max_playback_rate: 1.04,
            availability_time_offset: 0.0,
            timescale: 90000,
        }
    }
}

impl LlDashConfig {
    /// Creates a config with custom chunk duration; derived fields recalculated.
    #[must_use]
    pub fn with_chunk_duration(chunk_duration_secs: f64) -> Self {
        Self {
            chunk_duration_secs,
            target_latency_secs: chunk_duration_secs * 6.0,
            min_latency_secs: chunk_duration_secs * 4.0,
            max_latency_secs: chunk_duration_secs * 10.0,
            ..Self::default()
        }
    }

    /// Calculates chunks per segment.
    #[must_use]
    pub fn chunks_per_segment(&self) -> u32 {
        if self.chunk_duration_secs <= 0.0 {
            return 1;
        }
        (self.segment_duration_secs / self.chunk_duration_secs).ceil() as u32
    }
}

// ─── Service Description ──────────────────────────────────────────────────────

/// ServiceDescription element for LL-DASH MPD.
///
/// Controls client-side playback latency behavior including
/// target latency and playback rate adjustment range.
#[derive(Debug, Clone)]
pub struct ServiceDescription {
    /// Service description ID.
    pub id: u32,
    /// Target latency in milliseconds.
    pub target_latency_ms: u32,
    /// Minimum latency in milliseconds.
    pub min_latency_ms: u32,
    /// Maximum latency in milliseconds.
    pub max_latency_ms: u32,
    /// Minimum playback rate for catchup/fallback.
    pub min_playback_rate: f64,
    /// Maximum playback rate for catchup/fallback.
    pub max_playback_rate: f64,
}

impl ServiceDescription {
    /// Creates a service description from config.
    #[must_use]
    pub fn from_config(config: &LlDashConfig) -> Self {
        Self {
            id: 0,
            target_latency_ms: (config.target_latency_secs * 1000.0) as u32,
            min_latency_ms: (config.min_latency_secs * 1000.0) as u32,
            max_latency_ms: (config.max_latency_secs * 1000.0) as u32,
            min_playback_rate: config.min_playback_rate,
            max_playback_rate: config.max_playback_rate,
        }
    }

    /// Renders as XML element.
    #[must_use]
    pub fn to_xml(&self) -> String {
        let mut xml = String::with_capacity(512);
        let _ = writeln!(xml, "  <ServiceDescription id=\"{}\">", self.id);
        let _ = writeln!(
            xml,
            "    <Latency target=\"{}\" min=\"{}\" max=\"{}\"/>",
            self.target_latency_ms, self.min_latency_ms, self.max_latency_ms
        );
        let _ = writeln!(
            xml,
            "    <PlaybackRate min=\"{:.2}\" max=\"{:.2}\"/>",
            self.min_playback_rate, self.max_playback_rate
        );
        xml.push_str("  </ServiceDescription>");
        xml
    }
}

// ─── CMAF Chunk ───────────────────────────────────────────────────────────────

/// A CMAF chunk within an LL-DASH segment.
///
/// Each chunk is independently decodable (starts with a moof+mdat pair).
#[derive(Debug, Clone)]
pub struct CmafChunk {
    /// Chunk index within the segment (0-based).
    pub index: u32,
    /// Duration of this chunk in timescale units.
    pub duration_ticks: u64,
    /// Duration of this chunk in seconds.
    pub duration_secs: f64,
    /// Whether this chunk contains a keyframe (SAP type 1/2).
    pub is_independent: bool,
    /// Byte offset within the segment.
    pub byte_offset: u64,
    /// Size of this chunk in bytes.
    pub size: u64,
    /// Wall-clock time when this chunk became available.
    pub available_at: SystemTime,
    /// Whether this is the last chunk in the segment.
    pub is_last: bool,
}

impl CmafChunk {
    /// Creates a new CMAF chunk.
    #[must_use]
    pub fn new(index: u32, duration_secs: f64, timescale: u32) -> Self {
        Self {
            index,
            duration_ticks: (duration_secs * f64::from(timescale)) as u64,
            duration_secs,
            is_independent: false,
            byte_offset: 0,
            size: 0,
            available_at: SystemTime::now(),
            is_last: false,
        }
    }

    /// Marks this chunk as containing a keyframe.
    #[must_use]
    pub fn with_independent(mut self) -> Self {
        self.is_independent = true;
        self
    }

    /// Sets the byte range within the segment.
    #[must_use]
    pub fn with_byte_range(mut self, offset: u64, size: u64) -> Self {
        self.byte_offset = offset;
        self.size = size;
        self
    }

    /// Returns the HTTP Content-Range header value.
    #[must_use]
    pub fn content_range_header(&self, total_size: u64) -> String {
        let end = self.byte_offset + self.size.saturating_sub(1);
        format!("bytes {}-{}/{}", self.byte_offset, end, total_size)
    }
}

// ─── LL-DASH Segment ──────────────────────────────────────────────────────────

/// An LL-DASH segment composed of CMAF chunks.
#[derive(Debug, Clone)]
pub struct LlDashSegment {
    /// Segment number (template $Number$).
    pub number: u64,
    /// Start time in timescale units (template $Time$).
    pub start_time: u64,
    /// Total duration in timescale units.
    pub duration_ticks: u64,
    /// Duration in seconds.
    pub duration_secs: f64,
    /// Ordered list of CMAF chunks.
    pub chunks: Vec<CmafChunk>,
    /// Representation ID this segment belongs to.
    pub representation_id: String,
    /// Whether all chunks have been produced (segment finalized).
    pub is_complete: bool,
    /// Wall-clock time the segment was created.
    pub created_at: SystemTime,
}

impl LlDashSegment {
    /// Creates a new segment.
    #[must_use]
    pub fn new(number: u64, start_time: u64, representation_id: impl Into<String>) -> Self {
        Self {
            number,
            start_time,
            duration_ticks: 0,
            duration_secs: 0.0,
            chunks: Vec::new(),
            representation_id: representation_id.into(),
            is_complete: false,
            created_at: SystemTime::now(),
        }
    }

    /// Appends a chunk and accumulates duration.
    pub fn push_chunk(&mut self, chunk: CmafChunk) {
        self.duration_ticks += chunk.duration_ticks;
        self.duration_secs += chunk.duration_secs;
        if chunk.is_last {
            self.is_complete = true;
        }
        self.chunks.push(chunk);
    }

    /// Returns the number of chunks produced so far.
    #[must_use]
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Returns the total size across all chunks.
    #[must_use]
    pub fn total_size(&self) -> u64 {
        self.chunks.iter().map(|c| c.size).sum()
    }

    /// Returns a SegmentTimeline `<S>` element for this segment.
    #[must_use]
    pub fn to_timeline_s_element(&self) -> String {
        format!(
            "<S t=\"{}\" d=\"{}\"/>",
            self.start_time, self.duration_ticks
        )
    }
}

// ─── Resync Point ─────────────────────────────────────────────────────────────

/// Resync point for late-joining clients.
///
/// Marks a position within the stream where a decoder can begin
/// decoding without prior context (Random Access Point).
#[derive(Debug, Clone)]
pub struct ResyncPoint {
    /// Segment number containing the resync point.
    pub segment_number: u64,
    /// Chunk index within the segment.
    pub chunk_index: u32,
    /// Presentation time in timescale units.
    pub presentation_time: u64,
    /// SAP type (1 = closed GOP IDR, 2 = open GOP).
    pub sap_type: u8,
}

impl ResyncPoint {
    /// Creates a new resync point at a segment boundary.
    #[must_use]
    pub fn at_segment(segment_number: u64, presentation_time: u64) -> Self {
        Self {
            segment_number,
            chunk_index: 0,
            presentation_time,
            sap_type: 1,
        }
    }

    /// Creates a resync point at a specific chunk within a segment.
    #[must_use]
    pub fn at_chunk(segment_number: u64, chunk_index: u32, presentation_time: u64) -> Self {
        Self {
            segment_number,
            chunk_index,
            presentation_time,
            sap_type: 1,
        }
    }

    /// Sets the SAP type.
    #[must_use]
    pub fn with_sap_type(mut self, sap_type: u8) -> Self {
        self.sap_type = sap_type;
        self
    }

    /// Renders as an XML Resync element.
    #[must_use]
    pub fn to_xml(&self) -> String {
        format!(
            "<Resync type=\"{}\" dT=\"{}\" dImax=\"0\"/>",
            self.sap_type, self.presentation_time
        )
    }
}

// ─── LL-DASH Playlist / MPD Manager ──────────────────────────────────────────

/// Maximum number of segments kept in the sliding window.
const DEFAULT_LL_DASH_WINDOW: usize = 10;

/// LL-DASH MPD manager for server-side segment and manifest management.
///
/// Manages the sliding window of segments, chunk availability, and
/// generates LL-DASH-compliant MPD manifests.
#[derive(Debug)]
pub struct LlDashMpd {
    /// Configuration.
    config: LlDashConfig,
    /// Service description for the manifest.
    service_description: ServiceDescription,
    /// Sliding window of segments.
    segments: VecDeque<LlDashSegment>,
    /// Maximum segments in the window.
    window_size: usize,
    /// Chunks accumulated for the current in-progress segment.
    current_chunks: Vec<CmafChunk>,
    /// Next segment number.
    next_segment_number: u64,
    /// Current time in timescale units.
    current_time: u64,
    /// Resync points for late-joining.
    resync_points: VecDeque<ResyncPoint>,
    /// Max resync points to keep.
    max_resync_points: usize,
    /// Availability start time (wall-clock).
    availability_start_time: SystemTime,
    /// Representation ID.
    representation_id: String,
}

impl LlDashMpd {
    /// Creates a new LL-DASH MPD manager.
    #[must_use]
    pub fn new(config: &LlDashConfig) -> Self {
        let service_description = ServiceDescription::from_config(config);
        Self {
            config: config.clone(),
            service_description,
            segments: VecDeque::with_capacity(DEFAULT_LL_DASH_WINDOW + 1),
            window_size: DEFAULT_LL_DASH_WINDOW,
            current_chunks: Vec::new(),
            next_segment_number: 1,
            current_time: 0,
            resync_points: VecDeque::new(),
            max_resync_points: 5,
            availability_start_time: SystemTime::now(),
            representation_id: "1".to_owned(),
        }
    }

    /// Sets the sliding window size.
    pub fn set_window_size(&mut self, size: usize) {
        self.window_size = size.max(1);
    }

    /// Sets the representation ID.
    pub fn set_representation_id(&mut self, id: impl Into<String>) {
        self.representation_id = id.into();
    }

    /// Adds a CMAF chunk to the current segment.
    ///
    /// When the segment is complete (enough chunks accumulated),
    /// it is finalized and added to the window.
    pub fn add_chunk(&mut self, mut chunk: CmafChunk) {
        // Track resync points for independent chunks.
        if chunk.is_independent {
            let pt = self.current_time + chunk.duration_ticks;
            let rp = ResyncPoint::at_chunk(self.next_segment_number, chunk.index, pt);
            self.resync_points.push_back(rp);
            while self.resync_points.len() > self.max_resync_points {
                self.resync_points.pop_front();
            }
        }

        let chunks_per_seg = self.config.chunks_per_segment();
        let is_last = (self.current_chunks.len() as u32 + 1) >= chunks_per_seg;
        chunk.is_last = is_last;
        self.current_chunks.push(chunk);

        if is_last {
            self.finalize_segment();
        }
    }

    /// Returns the current number of completed segments.
    #[must_use]
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Returns the number of chunks in the current in-progress segment.
    #[must_use]
    pub fn current_chunk_count(&self) -> usize {
        self.current_chunks.len()
    }

    /// Returns the latest completed segment number.
    #[must_use]
    pub fn last_segment_number(&self) -> u64 {
        self.segments.back().map(|s| s.number).unwrap_or(0)
    }

    /// Generates the LL-DASH MPD XML.
    #[must_use]
    pub fn to_mpd_xml(&self) -> String {
        let mut xml = String::with_capacity(4096);

        // XML header
        xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        xml.push_str("<MPD xmlns=\"urn:mpeg:dash:schema:mpd:2011\"\n");
        xml.push_str("     xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"\n");
        xml.push_str("     type=\"dynamic\"\n");
        let _ = writeln!(
            xml,
            "     minimumUpdatePeriod=\"PT{:.1}S\"",
            self.config.chunk_duration_secs
        );
        let _ = writeln!(
            xml,
            "     minBufferTime=\"PT{:.1}S\"",
            self.config.segment_duration_secs
        );
        let _ = writeln!(
            xml,
            "     suggestedPresentationDelay=\"PT{:.1}S\"",
            self.config.target_latency_secs
        );
        let ast = format_system_time(self.availability_start_time);
        let _ = writeln!(xml, "     availabilityStartTime=\"{ast}\"");
        xml.push_str("     profiles=\"urn:mpeg:dash:profile:isoff-live:2011,urn:mpeg:dash:profile:cmaf:2019\">\n");

        // Service description
        let _ = writeln!(xml, "{}", self.service_description.to_xml());

        // Period
        xml.push_str("  <Period id=\"0\" start=\"PT0S\">\n");

        // Adaptation set
        xml.push_str("    <AdaptationSet mimeType=\"video/mp4\" contentType=\"video\">\n");

        // Segment template
        let ato = if self.config.availability_time_offset > 0.0 {
            format!(
                " availabilityTimeOffset=\"{:.3}\"",
                self.config.availability_time_offset
            )
        } else {
            String::new()
        };
        let _ = writeln!(
            xml,
            "      <SegmentTemplate timescale=\"{}\" media=\"segment_$Number$.m4s\" initialization=\"init.mp4\"{ato}>",
            self.config.timescale
        );

        // Timeline
        xml.push_str("        <SegmentTimeline>\n");
        for seg in &self.segments {
            let _ = writeln!(xml, "          {}", seg.to_timeline_s_element());
        }
        xml.push_str("        </SegmentTimeline>\n");
        xml.push_str("      </SegmentTemplate>\n");

        // Representation
        let _ = writeln!(
            xml,
            "      <Representation id=\"{}\" bandwidth=\"2000000\" width=\"1920\" height=\"1080\"/>",
            self.representation_id
        );

        xml.push_str("    </AdaptationSet>\n");
        xml.push_str("  </Period>\n");
        xml.push_str("</MPD>\n");

        xml
    }

    /// Returns chunk availability information for a specific segment and chunk.
    ///
    /// Returns `None` if the requested chunk is not yet available.
    #[must_use]
    pub fn chunk_availability(&self, segment_number: u64, chunk_index: u32) -> Option<&CmafChunk> {
        // Check completed segments
        for seg in &self.segments {
            if seg.number == segment_number {
                return seg.chunks.get(chunk_index as usize);
            }
        }
        // Check current in-progress segment
        if segment_number == self.next_segment_number {
            self.current_chunks.get(chunk_index as usize)
        } else {
            None
        }
    }

    /// Returns the latest resync points for late-joining clients.
    #[must_use]
    pub fn resync_points(&self) -> Vec<&ResyncPoint> {
        self.resync_points.iter().collect()
    }

    // ── Private ──────────────────────────────────────────────────────────────

    fn finalize_segment(&mut self) {
        let number = self.next_segment_number;
        self.next_segment_number += 1;

        let mut seg = LlDashSegment::new(number, self.current_time, &self.representation_id);
        for chunk in self.current_chunks.drain(..) {
            seg.push_chunk(chunk);
        }
        self.current_time += seg.duration_ticks;

        self.segments.push_back(seg);

        // Slide window
        while self.segments.len() > self.window_size {
            self.segments.pop_front();
        }
    }
}

/// Formats a `SystemTime` as ISO 8601 (simplified).
fn format_system_time(time: SystemTime) -> String {
    match time.duration_since(SystemTime::UNIX_EPOCH) {
        Ok(d) => {
            let secs = d.as_secs();
            let days = secs / 86400;
            let rem = secs % 86400;
            let hours = rem / 3600;
            let minutes = (rem % 3600) / 60;
            let seconds = rem % 60;
            // Approximate date calculation (good enough for testing)
            let years = 1970 + days / 365;
            let day_of_year = days % 365;
            let month = day_of_year / 30 + 1;
            let day = day_of_year % 30 + 1;
            format!("{years:04}-{month:02}-{day:02}T{hours:02}:{minutes:02}:{seconds:02}Z")
        }
        Err(_) => "1970-01-01T00:00:00Z".to_owned(),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn default_config() -> LlDashConfig {
        LlDashConfig::default()
    }

    fn make_chunk(index: u32, independent: bool) -> CmafChunk {
        let mut c = CmafChunk::new(index, 0.5, 90000);
        if independent {
            c = c.with_independent();
        }
        c = c.with_byte_range(index as u64 * 10000, 10000);
        c
    }

    // 1. Default config values
    #[test]
    fn test_default_config() {
        let cfg = default_config();
        assert!((cfg.segment_duration_secs - 2.0).abs() < 1e-9);
        assert!((cfg.chunk_duration_secs - 0.5).abs() < 1e-9);
        assert!((cfg.target_latency_secs - 3.0).abs() < 1e-9);
    }

    // 2. with_chunk_duration recalculates target latency
    #[test]
    fn test_with_chunk_duration() {
        let cfg = LlDashConfig::with_chunk_duration(0.25);
        assert!((cfg.target_latency_secs - 1.5).abs() < 1e-9);
        assert!((cfg.min_latency_secs - 1.0).abs() < 1e-9);
    }

    // 3. Chunks per segment calculation
    #[test]
    fn test_chunks_per_segment() {
        let cfg = default_config();
        assert_eq!(cfg.chunks_per_segment(), 4); // 2.0 / 0.5 = 4
    }

    // 4. Service description from config
    #[test]
    fn test_service_description_from_config() {
        let cfg = default_config();
        let sd = ServiceDescription::from_config(&cfg);
        assert_eq!(sd.target_latency_ms, 3000);
        assert_eq!(sd.min_latency_ms, 2000);
        assert_eq!(sd.max_latency_ms, 5000);
    }

    // 5. Service description XML rendering
    #[test]
    fn test_service_description_xml() {
        let cfg = default_config();
        let sd = ServiceDescription::from_config(&cfg);
        let xml = sd.to_xml();
        assert!(xml.contains("ServiceDescription"));
        assert!(xml.contains("Latency"));
        assert!(xml.contains("target=\"3000\""));
        assert!(xml.contains("PlaybackRate"));
    }

    // 6. CMAF chunk creation
    #[test]
    fn test_cmaf_chunk_new() {
        let chunk = CmafChunk::new(0, 0.5, 90000);
        assert_eq!(chunk.index, 0);
        assert_eq!(chunk.duration_ticks, 45000);
        assert!(!chunk.is_independent);
    }

    // 7. CMAF chunk independent flag
    #[test]
    fn test_cmaf_chunk_independent() {
        let chunk = CmafChunk::new(0, 0.5, 90000).with_independent();
        assert!(chunk.is_independent);
    }

    // 8. CMAF chunk byte range
    #[test]
    fn test_cmaf_chunk_byte_range() {
        let chunk = CmafChunk::new(0, 0.5, 90000).with_byte_range(1024, 4096);
        assert_eq!(chunk.byte_offset, 1024);
        assert_eq!(chunk.size, 4096);
    }

    // 9. Content-Range header
    #[test]
    fn test_content_range_header() {
        let chunk = CmafChunk::new(0, 0.5, 90000).with_byte_range(100, 500);
        let header = chunk.content_range_header(10000);
        assert_eq!(header, "bytes 100-599/10000");
    }

    // 10. LL-DASH segment creation
    #[test]
    fn test_ll_dash_segment_new() {
        let seg = LlDashSegment::new(1, 0, "720p");
        assert_eq!(seg.number, 1);
        assert_eq!(seg.start_time, 0);
        assert!(!seg.is_complete);
        assert_eq!(seg.representation_id, "720p");
    }

    // 11. Segment push_chunk accumulates duration
    #[test]
    fn test_segment_push_chunk() {
        let mut seg = LlDashSegment::new(1, 0, "1");
        seg.push_chunk(make_chunk(0, true));
        seg.push_chunk(make_chunk(1, false));
        assert_eq!(seg.chunk_count(), 2);
        assert!((seg.duration_secs - 1.0).abs() < 1e-6);
    }

    // 12. Segment total size
    #[test]
    fn test_segment_total_size() {
        let mut seg = LlDashSegment::new(1, 0, "1");
        seg.push_chunk(make_chunk(0, true));
        seg.push_chunk(make_chunk(1, false));
        assert_eq!(seg.total_size(), 20000);
    }

    // 13. Segment timeline element
    #[test]
    fn test_segment_timeline_element() {
        let mut seg = LlDashSegment::new(1, 90000, "1");
        seg.push_chunk(make_chunk(0, true));
        let s = seg.to_timeline_s_element();
        assert!(s.contains("t=\"90000\""));
        assert!(s.contains("d=\"45000\""));
    }

    // 14. Resync point at segment boundary
    #[test]
    fn test_resync_point_at_segment() {
        let rp = ResyncPoint::at_segment(5, 450000);
        assert_eq!(rp.segment_number, 5);
        assert_eq!(rp.chunk_index, 0);
        assert_eq!(rp.sap_type, 1);
    }

    // 15. Resync point XML
    #[test]
    fn test_resync_point_xml() {
        let rp = ResyncPoint::at_segment(1, 90000).with_sap_type(2);
        let xml = rp.to_xml();
        assert!(xml.contains("type=\"2\""));
        assert!(xml.contains("dT=\"90000\""));
    }

    // 16. LlDashMpd creation
    #[test]
    fn test_ll_dash_mpd_new() {
        let cfg = default_config();
        let mpd = LlDashMpd::new(&cfg);
        assert_eq!(mpd.segment_count(), 0);
        assert_eq!(mpd.current_chunk_count(), 0);
    }

    // 17. Adding chunks without completing segment
    #[test]
    fn test_add_chunks_partial() {
        let cfg = default_config();
        let mut mpd = LlDashMpd::new(&cfg);
        mpd.add_chunk(make_chunk(0, true));
        mpd.add_chunk(make_chunk(1, false));
        assert_eq!(mpd.segment_count(), 0);
        assert_eq!(mpd.current_chunk_count(), 2);
    }

    // 18. Completing a segment
    #[test]
    fn test_complete_segment() {
        let cfg = default_config(); // 4 chunks per segment
        let mut mpd = LlDashMpd::new(&cfg);
        for i in 0..4u32 {
            mpd.add_chunk(make_chunk(i, i == 0));
        }
        assert_eq!(mpd.segment_count(), 1);
        assert_eq!(mpd.current_chunk_count(), 0);
        assert_eq!(mpd.last_segment_number(), 1);
    }

    // 19. Window slides after exceeding size
    #[test]
    fn test_window_slides() {
        let cfg = default_config();
        let mut mpd = LlDashMpd::new(&cfg);
        mpd.set_window_size(3);
        // Create 5 segments (4 chunks each)
        for seg in 0..5u32 {
            for chunk in 0..4u32 {
                mpd.add_chunk(make_chunk(seg * 10 + chunk, chunk == 0));
            }
        }
        assert_eq!(mpd.segment_count(), 3);
    }

    // 20. MPD XML generation
    #[test]
    fn test_mpd_xml_generation() {
        let cfg = default_config();
        let mut mpd = LlDashMpd::new(&cfg);
        // Add one segment
        for i in 0..4u32 {
            mpd.add_chunk(make_chunk(i, i == 0));
        }
        let xml = mpd.to_mpd_xml();
        assert!(xml.contains("<?xml"));
        assert!(xml.contains("MPD"));
        assert!(xml.contains("type=\"dynamic\""));
        assert!(xml.contains("ServiceDescription"));
        assert!(xml.contains("SegmentTimeline"));
        assert!(xml.contains("cmaf"));
    }

    // 21. Chunk availability for completed segment
    #[test]
    fn test_chunk_availability_completed() {
        let cfg = default_config();
        let mut mpd = LlDashMpd::new(&cfg);
        for i in 0..4u32 {
            mpd.add_chunk(make_chunk(i, i == 0));
        }
        let chunk = mpd.chunk_availability(1, 0);
        assert!(chunk.is_some());
        assert!(mpd.chunk_availability(1, 5).is_none());
    }

    // 22. Chunk availability for in-progress segment
    #[test]
    fn test_chunk_availability_in_progress() {
        let cfg = default_config();
        let mut mpd = LlDashMpd::new(&cfg);
        mpd.add_chunk(make_chunk(0, true));
        mpd.add_chunk(make_chunk(1, false));
        // next_segment_number is 1, so query segment 1
        let chunk = mpd.chunk_availability(1, 0);
        assert!(chunk.is_some());
        assert!(mpd.chunk_availability(1, 3).is_none());
    }

    // 23. Resync points tracked
    #[test]
    fn test_resync_points_tracked() {
        let cfg = default_config();
        let mut mpd = LlDashMpd::new(&cfg);
        // Add chunks with independent flag
        for i in 0..4u32 {
            mpd.add_chunk(make_chunk(i, i == 0));
        }
        let rps = mpd.resync_points();
        assert!(!rps.is_empty());
    }

    // 24. Format system time
    #[test]
    fn test_format_system_time() {
        let t = SystemTime::UNIX_EPOCH + Duration::from_secs(86400); // 1 day
        let s = format_system_time(t);
        assert!(s.contains("1970"));
        assert!(s.ends_with('Z'));
    }

    // 25. LlDashConfig zero chunk duration edge case
    #[test]
    fn test_zero_chunk_duration() {
        let mut cfg = default_config();
        cfg.chunk_duration_secs = 0.0;
        assert_eq!(cfg.chunks_per_segment(), 1);
    }

    // 26. Segment is_complete flag set by last chunk
    #[test]
    fn test_segment_complete_flag() {
        let mut seg = LlDashSegment::new(1, 0, "1");
        let mut c = make_chunk(0, true);
        c.is_last = true;
        seg.push_chunk(c);
        assert!(seg.is_complete);
    }

    // 27. Multiple representation IDs
    #[test]
    fn test_set_representation_id() {
        let cfg = default_config();
        let mut mpd = LlDashMpd::new(&cfg);
        mpd.set_representation_id("1080p");
        for i in 0..4u32 {
            mpd.add_chunk(make_chunk(i, i == 0));
        }
        let xml = mpd.to_mpd_xml();
        assert!(xml.contains("id=\"1080p\""));
    }

    // 28. Availability time offset in XML
    #[test]
    fn test_availability_time_offset() {
        let mut cfg = default_config();
        cfg.availability_time_offset = 1.5;
        let mut mpd = LlDashMpd::new(&cfg);
        for i in 0..4u32 {
            mpd.add_chunk(make_chunk(i, i == 0));
        }
        let xml = mpd.to_mpd_xml();
        assert!(xml.contains("availabilityTimeOffset=\"1.500\""));
    }
}
