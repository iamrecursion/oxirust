//! LL-DASH chunked-transfer encoder.
//!
//! [`LlDashChunker`] wraps an ongoing DASH segment as a CMAF "chunked"
//! delivery stream.  Each [`DashChunk`] corresponds to one
//! `moof`+`mdat` box pair that is independently parseable.  A matching
//! `availabilityTimeOffset` is computed and written into the MPD
//! [`SegmentTemplate`] so clients can fetch chunks before the full
//! segment is available.
//!
//! # Design
//!
//! ```text
//!   push_frame() ─► [accumulator] ─► flush_chunk() ─► DashChunk ready
//!                                                 └─► DashChunkEvent broadcast
//! ```

use crate::error::{NetError, NetResult};
use std::collections::VecDeque;
use std::time::{Duration, SystemTime};

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for [`LlDashChunker`].
#[derive(Debug, Clone)]
pub struct LlDashChunkerConfig {
    /// Target chunk duration in seconds (typically 0.5–1.0 s).
    pub chunk_duration_secs: f64,
    /// Number of chunks per complete segment.
    pub chunks_per_segment: u32,
    /// Timescale for ticks (default 90 000 for video).
    pub timescale: u32,
    /// Representation ID embedded in MPD `<Representation>`.
    pub representation_id: String,
    /// Bandwidth in bits per second for MPD advertising.
    pub bandwidth_bps: u64,
    /// `availabilityTimeOffset` value in seconds.
    ///
    /// Set to `chunk_duration_secs` to advertise availability one
    /// chunk before segment end.  Set to `segment_duration - chunk_duration`
    /// for maximum early advertisement.
    pub availability_time_offset: f64,
    /// Maximum segments to retain in the sliding window.
    pub window_size: usize,
}

impl Default for LlDashChunkerConfig {
    fn default() -> Self {
        let chunk = 0.5_f64;
        let chunks_per_seg = 4u32;
        let seg_dur = chunk * f64::from(chunks_per_seg);
        Self {
            chunk_duration_secs: chunk,
            chunks_per_segment: chunks_per_seg,
            timescale: 90_000,
            representation_id: "1".to_owned(),
            bandwidth_bps: 2_000_000,
            availability_time_offset: seg_dur - chunk, // one chunk before end
            window_size: 10,
        }
    }
}

impl LlDashChunkerConfig {
    /// Creates a config with the given chunk duration and number of chunks per segment.
    #[must_use]
    pub fn new(chunk_duration_secs: f64, chunks_per_segment: u32) -> Self {
        let seg_dur = chunk_duration_secs * f64::from(chunks_per_segment);
        Self {
            chunk_duration_secs,
            chunks_per_segment,
            availability_time_offset: seg_dur - chunk_duration_secs,
            ..Self::default()
        }
    }

    /// Returns the segment duration in seconds.
    #[must_use]
    pub fn segment_duration_secs(&self) -> f64 {
        self.chunk_duration_secs * f64::from(self.chunks_per_segment)
    }

    /// Returns duration as timescale ticks.
    #[must_use]
    pub fn chunk_duration_ticks(&self) -> u64 {
        (self.chunk_duration_secs * f64::from(self.timescale)) as u64
    }
}

// ─── Chunk ────────────────────────────────────────────────────────────────────

/// A single CMAF chunk within an LL-DASH segment.
///
/// Each chunk is a `moof`+`mdat` pair suitable for chunked HTTP transfer.
#[derive(Debug, Clone)]
pub struct DashChunk {
    /// Segment number this chunk belongs to.
    pub segment_number: u64,
    /// Chunk index within the segment (0-based).
    pub chunk_index: u32,
    /// Start time of this chunk in timescale units.
    pub start_time_ticks: u64,
    /// Duration of this chunk in timescale units.
    pub duration_ticks: u64,
    /// Duration in seconds.
    pub duration_secs: f64,
    /// Whether this chunk contains a keyframe (SAP type 1).
    pub is_independent: bool,
    /// Whether this is the last chunk in its segment.
    pub is_last: bool,
    /// Raw payload bytes (simulated CMAF envelope).
    pub data: Vec<u8>,
    /// Wall-clock time this chunk was produced.
    pub produced_at: SystemTime,
    /// Byte range within the parent segment for byte-range delivery.
    pub byte_offset: u64,
}

impl DashChunk {
    /// Returns an HTTP `Content-Range` header value for this chunk.
    #[must_use]
    pub fn content_range(&self, total_segment_bytes: u64) -> String {
        let end = self.byte_offset + self.data.len() as u64;
        format!("bytes {}-{}/{}", self.byte_offset, end.saturating_sub(1), total_segment_bytes)
    }

    /// Returns the URL template substitution for this chunk's segment.
    #[must_use]
    pub fn segment_url_number(&self) -> String {
        self.segment_number.to_string()
    }
}

// ─── Segment Record ───────────────────────────────────────────────────────────

/// A completed LL-DASH segment.
#[derive(Debug, Clone)]
pub struct CompletedDashSegment {
    /// Segment number.
    pub number: u64,
    /// Segment start time in timescale ticks.
    pub start_time_ticks: u64,
    /// Total duration in timescale ticks.
    pub duration_ticks: u64,
    /// Total duration in seconds.
    pub duration_secs: f64,
    /// All chunks that make up this segment.
    pub chunks: Vec<DashChunk>,
    /// Finalization time.
    pub finalized_at: SystemTime,
}

impl CompletedDashSegment {
    /// Returns the `<S>` SegmentTimeline element for this segment.
    #[must_use]
    pub fn to_timeline_s(&self) -> String {
        format!(
            "<S t=\"{}\" d=\"{}\"/>",
            self.start_time_ticks, self.duration_ticks
        )
    }

    /// Returns the total byte size across all chunks.
    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        self.chunks.iter().map(|c| c.data.len() as u64).sum()
    }
}

// ─── MPD Fragment Generator ───────────────────────────────────────────────────

/// Generates the `<SegmentTemplate>` XML fragment with `availabilityTimeOffset`.
#[must_use]
pub fn segment_template_xml(config: &LlDashChunkerConfig) -> String {
    format!(
        "<SegmentTemplate timescale=\"{ts}\" \
         media=\"chunk_$Number$_$Time$.m4s\" \
         initialization=\"init.mp4\" \
         availabilityTimeOffset=\"{ato:.3}\">\n",
        ts = config.timescale,
        ato = config.availability_time_offset,
    )
}

/// Generates a minimal LL-DASH MPD with `ServiceDescription` and
/// `availabilityTimeOffset` using the sliding window of completed segments.
#[must_use]
pub fn generate_ll_dash_mpd(
    config: &LlDashChunkerConfig,
    segments: &VecDeque<CompletedDashSegment>,
    availability_start: SystemTime,
) -> String {
    let ast = format_system_time(availability_start);
    let seg_dur = config.segment_duration_secs();
    let update_period = config.chunk_duration_secs;

    let mut xml = String::with_capacity(2048);
    xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    xml.push_str("<MPD xmlns=\"urn:mpeg:dash:schema:mpd:2011\"\n");
    xml.push_str("     type=\"dynamic\"\n");
    xml.push_str(&format!("     minimumUpdatePeriod=\"PT{update_period:.3}S\"\n"));
    xml.push_str(&format!("     minBufferTime=\"PT{seg_dur:.1}S\"\n"));
    xml.push_str(&format!("     availabilityStartTime=\"{ast}\"\n"));
    xml.push_str("     profiles=\"urn:mpeg:dash:profile:isoff-live:2011,urn:mpeg:dash:profile:cmaf:2019\">\n");

    // ServiceDescription for LL-DASH latency hints
    xml.push_str("  <ServiceDescription id=\"0\">\n");
    xml.push_str(&format!(
        "    <Latency target=\"{}\" min=\"{}\" max=\"{}\"/>\n",
        (seg_dur * 1000.0) as u32,
        (seg_dur * 1000.0 * 0.5) as u32,
        (seg_dur * 1000.0 * 2.0) as u32,
    ));
    xml.push_str("    <PlaybackRate min=\"0.96\" max=\"1.04\"/>\n");
    xml.push_str("  </ServiceDescription>\n");

    xml.push_str("  <Period id=\"0\" start=\"PT0S\">\n");
    xml.push_str("    <AdaptationSet mimeType=\"video/mp4\" contentType=\"video\">\n");
    xml.push_str(&segment_template_xml(config));
    xml.push_str("        <SegmentTimeline>\n");
    for seg in segments {
        xml.push_str(&format!("          {}\n", seg.to_timeline_s()));
    }
    xml.push_str("        </SegmentTimeline>\n");
    xml.push_str("      </SegmentTemplate>\n");
    xml.push_str(&format!(
        "      <Representation id=\"{}\" bandwidth=\"{}\" width=\"1920\" height=\"1080\"/>\n",
        config.representation_id, config.bandwidth_bps
    ));
    xml.push_str("    </AdaptationSet>\n");
    xml.push_str("  </Period>\n");
    xml.push_str("</MPD>\n");
    xml
}

fn format_system_time(t: SystemTime) -> String {
    match t.duration_since(SystemTime::UNIX_EPOCH) {
        Ok(d) => {
            let secs = d.as_secs();
            let days = secs / 86400;
            let rem = secs % 86400;
            let h = rem / 3600;
            let m = (rem % 3600) / 60;
            let s = rem % 60;
            let y = 1970 + days / 365;
            let doy = days % 365;
            let mo = doy / 30 + 1;
            let day = doy % 30 + 1;
            format!("{y:04}-{mo:02}-{day:02}T{h:02}:{m:02}:{s:02}Z")
        }
        Err(_) => "1970-01-01T00:00:00Z".to_owned(),
    }
}

// ─── Accumulator ─────────────────────────────────────────────────────────────

#[derive(Debug, Default)]
struct ChunkAccumulator {
    data: Vec<u8>,
    duration_ms: u64,
    has_keyframe: bool,
    frame_count: u32,
}

impl ChunkAccumulator {
    fn push(&mut self, data: &[u8], duration_ms: u64, is_keyframe: bool) {
        self.data.extend_from_slice(data);
        self.duration_ms += duration_ms;
        self.frame_count += 1;
        if is_keyframe {
            self.has_keyframe = true;
        }
    }

    fn reset(&mut self) {
        self.data.clear();
        self.duration_ms = 0;
        self.has_keyframe = false;
        self.frame_count = 0;
    }

    fn is_empty(&self) -> bool {
        self.frame_count == 0
    }
}

// ─── LlDashChunker ───────────────────────────────────────────────────────────

/// LL-DASH chunked-transfer encoder.
///
/// Accepts raw encoded frames and slices them into CMAF chunks, with
/// each chunk available for chunked HTTP delivery before the segment ends.
///
/// The `availabilityTimeOffset` in the generated MPD tells clients that
/// chunks are available before the segment's nominal availability time,
/// enabling sub-segment latency.
pub struct LlDashChunker {
    config: LlDashChunkerConfig,
    /// Current segment number (1-based as per DASH convention).
    segment_number: u64,
    /// Chunk index within the current segment.
    chunk_index: u32,
    /// Current time in timescale ticks.
    current_time_ticks: u64,
    /// Byte offset within the current segment.
    current_byte_offset: u64,
    /// Frame accumulator for the current chunk.
    accumulator: ChunkAccumulator,
    /// Chunks accumulated for the current (incomplete) segment.
    current_segment_chunks: Vec<DashChunk>,
    /// Ready chunks for consumption.
    ready_chunks: VecDeque<DashChunk>,
    /// Completed segment sliding window.
    completed: VecDeque<CompletedDashSegment>,
    /// Availability start time for MPD.
    availability_start: SystemTime,
    /// PTS of the last frame for duration computation.
    last_pts_ms: Option<u64>,
    /// Default frame duration in ms.
    default_frame_duration_ms: u64,
}

impl LlDashChunker {
    /// Creates a new chunker with the given configuration.
    #[must_use]
    pub fn new(config: LlDashChunkerConfig) -> Self {
        Self {
            config,
            segment_number: 1,
            chunk_index: 0,
            current_time_ticks: 0,
            current_byte_offset: 0,
            accumulator: ChunkAccumulator::default(),
            current_segment_chunks: Vec::new(),
            ready_chunks: VecDeque::new(),
            completed: VecDeque::new(),
            availability_start: SystemTime::now(),
            last_pts_ms: None,
            default_frame_duration_ms: 33,
        }
    }

    /// Creates a chunker with default configuration.
    #[must_use]
    pub fn default_chunker() -> Self {
        Self::new(LlDashChunkerConfig::default())
    }

    /// Returns the current segment number.
    #[must_use]
    pub fn current_segment_number(&self) -> u64 {
        self.segment_number
    }

    /// Returns the current chunk index within the active segment.
    #[must_use]
    pub fn current_chunk_index(&self) -> u32 {
        self.chunk_index
    }

    /// Returns the number of completed segments in the window.
    #[must_use]
    pub fn completed_segment_count(&self) -> usize {
        self.completed.len()
    }

    /// Returns the number of ready chunks.
    #[must_use]
    pub fn ready_chunk_count(&self) -> usize {
        self.ready_chunks.len()
    }

    /// Pushes a raw encoded frame into the chunker.
    pub fn push_frame(&mut self, data: &[u8], pts_ms: u64, is_keyframe: bool) {
        let dur_ms = match self.last_pts_ms {
            Some(prev) => pts_ms.saturating_sub(prev).max(1),
            None => self.default_frame_duration_ms,
        };
        self.last_pts_ms = Some(pts_ms);

        // Flush on keyframe if accumulator has data
        if is_keyframe && !self.accumulator.is_empty() {
            self.flush_chunk();
        }

        self.accumulator.push(data, dur_ms, is_keyframe);

        // Flush when chunk duration threshold reached
        let target_ms = (self.config.chunk_duration_secs * 1000.0) as u64;
        if self.accumulator.duration_ms >= target_ms {
            self.flush_chunk();
        }
    }

    /// Forces the current accumulator into a chunk.
    pub fn flush(&mut self) {
        if !self.accumulator.is_empty() {
            self.flush_chunk();
        }
    }

    /// Drains all ready chunks.
    pub fn drain_chunks(&mut self) -> Vec<DashChunk> {
        self.ready_chunks.drain(..).collect()
    }

    /// Takes the next completed segment from the window.
    pub fn take_completed_segment(&mut self) -> Option<CompletedDashSegment> {
        self.completed.pop_front()
    }

    /// Returns all completed segments in the sliding window.
    #[must_use]
    pub fn completed_segments(&self) -> &VecDeque<CompletedDashSegment> {
        &self.completed
    }

    /// Generates the current MPD XML.
    #[must_use]
    pub fn generate_mpd(&self) -> String {
        generate_ll_dash_mpd(&self.config, &self.completed, self.availability_start)
    }

    /// Returns the `availabilityTimeOffset` value from config.
    #[must_use]
    pub fn availability_time_offset(&self) -> f64 {
        self.config.availability_time_offset
    }

    // ── Private ───────────────────────────────────────────────────────────────

    fn flush_chunk(&mut self) {
        if self.accumulator.is_empty() {
            return;
        }

        let dur_ticks = (self.accumulator.duration_ms as f64
            / 1000.0
            * f64::from(self.config.timescale)) as u64;
        let dur_secs = self.accumulator.duration_ms as f64 / 1000.0;
        let is_last = self.chunk_index + 1 >= self.config.chunks_per_segment;
        let byte_offset = self.current_byte_offset;

        let chunk = DashChunk {
            segment_number: self.segment_number,
            chunk_index: self.chunk_index,
            start_time_ticks: self.current_time_ticks,
            duration_ticks: dur_ticks,
            duration_secs: dur_secs,
            is_independent: self.accumulator.has_keyframe,
            is_last,
            data: std::mem::take(&mut self.accumulator.data),
            produced_at: SystemTime::now(),
            byte_offset,
        };

        let chunk_size = chunk.data.len() as u64;
        self.current_byte_offset += chunk_size;
        self.current_time_ticks += dur_ticks;
        self.accumulator.reset();
        self.chunk_index += 1;
        self.current_segment_chunks.push(chunk.clone());
        self.ready_chunks.push_back(chunk);

        if is_last {
            self.finalize_segment();
        }
    }

    fn finalize_segment(&mut self) {
        let total_dur_ticks: u64 = self.current_segment_chunks.iter().map(|c| c.duration_ticks).sum();
        let total_dur_secs: f64 = self.current_segment_chunks.iter().map(|c| c.duration_secs).sum();
        let start = self
            .current_segment_chunks
            .first()
            .map(|c| c.start_time_ticks)
            .unwrap_or(0);

        let seg = CompletedDashSegment {
            number: self.segment_number,
            start_time_ticks: start,
            duration_ticks: total_dur_ticks,
            duration_secs: total_dur_secs,
            chunks: std::mem::take(&mut self.current_segment_chunks),
            finalized_at: SystemTime::now(),
        };

        self.completed.push_back(seg);
        while self.completed.len() > self.config.window_size {
            self.completed.pop_front();
        }

        self.segment_number += 1;
        self.chunk_index = 0;
        self.current_byte_offset = 0;
    }
}

impl std::fmt::Debug for LlDashChunker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlDashChunker")
            .field("segment_number", &self.segment_number)
            .field("chunk_index", &self.chunk_index)
            .field("ready_chunks", &self.ready_chunks.len())
            .finish()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_chunker() -> LlDashChunker {
        LlDashChunker::new(LlDashChunkerConfig::new(0.1, 3))
    }

    fn push_frames(chunker: &mut LlDashChunker, count: usize) {
        for i in 0..count {
            let pts = i as u64 * 33;
            chunker.push_frame(&[0u8; 512], pts, i == 0);
        }
    }

    // 1. Config defaults
    #[test]
    fn test_config_default() {
        let cfg = LlDashChunkerConfig::default();
        assert!((cfg.chunk_duration_secs - 0.5).abs() < 1e-9);
        assert_eq!(cfg.chunks_per_segment, 4);
    }

    // 2. Config segment duration
    #[test]
    fn test_config_segment_duration() {
        let cfg = LlDashChunkerConfig::new(0.5, 4);
        assert!((cfg.segment_duration_secs() - 2.0).abs() < 1e-9);
    }

    // 3. Config availability_time_offset
    #[test]
    fn test_config_ato() {
        let cfg = LlDashChunkerConfig::new(0.5, 4);
        // ato = segment_dur - chunk_dur = 2.0 - 0.5 = 1.5
        assert!((cfg.availability_time_offset - 1.5).abs() < 1e-9);
    }

    // 4. Config chunk duration ticks
    #[test]
    fn test_chunk_duration_ticks() {
        let cfg = LlDashChunkerConfig::default();
        let ticks = cfg.chunk_duration_ticks();
        assert_eq!(ticks, (0.5 * 90_000.0) as u64);
    }

    // 5. Chunker initial state
    #[test]
    fn test_chunker_initial_state() {
        let c = default_chunker();
        assert_eq!(c.current_segment_number(), 1);
        assert_eq!(c.current_chunk_index(), 0);
        assert_eq!(c.ready_chunk_count(), 0);
    }

    // 6. Push frame accumulates without producing chunk yet
    #[test]
    fn test_push_frame_no_immediate_chunk() {
        let mut c = default_chunker();
        c.push_frame(&[0u8; 512], 0, true);
        assert_eq!(c.ready_chunk_count(), 0);
    }

    // 7. Enough frames produce a chunk
    #[test]
    fn test_frames_produce_chunk() {
        let mut c = default_chunker(); // chunk_dur = 100 ms
        for i in 0..5u64 {
            c.push_frame(&[0u8; 256], i * 33, i == 0);
        }
        let chunks = c.drain_chunks();
        assert!(!chunks.is_empty());
    }

    // 8. Keyframe forces chunk boundary
    #[test]
    fn test_keyframe_forces_chunk_boundary() {
        let mut c = LlDashChunker::new(LlDashChunkerConfig::new(5.0, 2)); // long chunks
        c.push_frame(&[0u8; 256], 0, false);
        c.push_frame(&[0u8; 256], 33, true); // keyframe → flush previous
        let chunks = c.drain_chunks();
        assert_eq!(chunks.len(), 1);
    }

    // 9. Independent flag set on keyframe chunks
    #[test]
    fn test_independent_flag() {
        let mut c = LlDashChunker::new(LlDashChunkerConfig::new(5.0, 2));
        c.push_frame(&[0u8; 256], 0, true); // keyframe in accumulator
        c.push_frame(&[0u8; 256], 33, false);
        c.push_frame(&[0u8; 256], 66, true); // flush previous
        let chunks = c.drain_chunks();
        if let Some(first) = chunks.first() {
            assert!(first.is_independent);
        }
    }

    // 10. Chunk carries correct segment number
    #[test]
    fn test_chunk_segment_number() {
        let mut c = default_chunker();
        push_frames(&mut c, 5);
        let chunks = c.drain_chunks();
        for ch in &chunks {
            assert_eq!(ch.segment_number, 1);
        }
    }

    // 11. Chunk indices are sequential within a segment
    #[test]
    fn test_chunk_indices_sequential() {
        let mut c = LlDashChunker::new(LlDashChunkerConfig::new(0.1, 5));
        for i in 0..8u64 {
            c.push_frame(&[0u8; 256], i * 33, i % 3 == 0);
        }
        let chunks = c.drain_chunks();
        for (expected, ch) in chunks.iter().enumerate() {
            assert_eq!(ch.chunk_index, expected as u32);
        }
    }

    // 12. Full segment produced after chunks_per_segment chunks
    #[test]
    fn test_full_segment_produced() {
        let mut c = LlDashChunker::new(LlDashChunkerConfig::new(0.1, 3));
        for i in 0..12u64 {
            c.push_frame(&[0u8; 256], i * 20, i % 4 == 0);
        }
        let seg = c.take_completed_segment();
        assert!(seg.is_some());
        let s = seg.expect("should have segment");
        assert_eq!(s.number, 1);
        assert_eq!(s.chunks.len(), 3);
    }

    // 13. Segment number increments after completion
    #[test]
    fn test_segment_number_increments() {
        let mut c = LlDashChunker::new(LlDashChunkerConfig::new(0.1, 2));
        for i in 0..12u64 {
            c.push_frame(&[0u8; 256], i * 20, i % 3 == 0);
        }
        assert!(c.current_segment_number() >= 2);
    }

    // 14. Window size limits completed segments
    #[test]
    fn test_window_size_limit() {
        let mut cfg = LlDashChunkerConfig::new(0.1, 2);
        cfg.window_size = 2;
        let mut c = LlDashChunker::new(cfg);
        for i in 0..20u64 {
            c.push_frame(&[0u8; 256], i * 20, i % 3 == 0);
        }
        assert!(c.completed_segment_count() <= 2);
    }

    // 15. Flush produces remaining accumulator as chunk
    #[test]
    fn test_explicit_flush() {
        let mut c = default_chunker();
        c.push_frame(&[0u8; 256], 0, true);
        c.flush();
        assert!(c.ready_chunk_count() > 0 || c.current_chunk_index() > 0);
    }

    // 16. Content-Range header format
    #[test]
    fn test_content_range_header() {
        let chunk = DashChunk {
            segment_number: 1,
            chunk_index: 0,
            start_time_ticks: 0,
            duration_ticks: 45000,
            duration_secs: 0.5,
            is_independent: true,
            is_last: false,
            data: vec![0u8; 1024],
            produced_at: SystemTime::now(),
            byte_offset: 0,
        };
        let range = chunk.content_range(10240);
        assert!(range.starts_with("bytes 0-"));
        assert!(range.contains("/10240"));
    }

    // 17. Segment URL number
    #[test]
    fn test_segment_url_number() {
        let chunk = DashChunk {
            segment_number: 42,
            chunk_index: 0,
            start_time_ticks: 0,
            duration_ticks: 0,
            duration_secs: 0.0,
            is_independent: false,
            is_last: false,
            data: vec![],
            produced_at: SystemTime::now(),
            byte_offset: 0,
        };
        assert_eq!(chunk.segment_url_number(), "42");
    }

    // 18. Timeline `<S>` element format
    #[test]
    fn test_timeline_s_element() {
        let seg = CompletedDashSegment {
            number: 1,
            start_time_ticks: 90000,
            duration_ticks: 180000,
            duration_secs: 2.0,
            chunks: vec![],
            finalized_at: SystemTime::now(),
        };
        let s = seg.to_timeline_s();
        assert!(s.contains("t=\"90000\""));
        assert!(s.contains("d=\"180000\""));
    }

    // 19. Segment total_bytes
    #[test]
    fn test_segment_total_bytes() {
        let chunk = DashChunk {
            segment_number: 1,
            chunk_index: 0,
            start_time_ticks: 0,
            duration_ticks: 45000,
            duration_secs: 0.5,
            is_independent: true,
            is_last: true,
            data: vec![0u8; 2048],
            produced_at: SystemTime::now(),
            byte_offset: 0,
        };
        let seg = CompletedDashSegment {
            number: 1,
            start_time_ticks: 0,
            duration_ticks: 45000,
            duration_secs: 0.5,
            chunks: vec![chunk],
            finalized_at: SystemTime::now(),
        };
        assert_eq!(seg.total_bytes(), 2048);
    }

    // 20. MPD XML contains availabilityTimeOffset
    #[test]
    fn test_mpd_contains_ato() {
        let mut cfg = LlDashChunkerConfig::default();
        cfg.availability_time_offset = 1.5;
        let mut c = LlDashChunker::new(cfg);
        for i in 0..8u64 {
            c.push_frame(&[0u8; 256], i * 20, i % 4 == 0);
        }
        let mpd = c.generate_mpd();
        assert!(mpd.contains("availabilityTimeOffset=\"1.500\""));
    }

    // 21. MPD XML is well-formed (contains key elements)
    #[test]
    fn test_mpd_xml_wellformed() {
        let mut c = default_chunker();
        for i in 0..8u64 {
            c.push_frame(&[0u8; 256], i * 20, i % 4 == 0);
        }
        let mpd = c.generate_mpd();
        assert!(mpd.contains("<?xml"));
        assert!(mpd.contains("MPD"));
        assert!(mpd.contains("type=\"dynamic\""));
        assert!(mpd.contains("ServiceDescription"));
        assert!(mpd.contains("SegmentTimeline"));
    }

    // 22. segment_template_xml contains timescale and ATO
    #[test]
    fn test_segment_template_xml() {
        let mut cfg = LlDashChunkerConfig::default();
        cfg.availability_time_offset = 0.5;
        let xml = segment_template_xml(&cfg);
        assert!(xml.contains("timescale=\"90000\""));
        assert!(xml.contains("availabilityTimeOffset=\"0.500\""));
    }

    // 23. Debug format available
    #[test]
    fn test_debug_format() {
        let c = default_chunker();
        let dbg = format!("{c:?}");
        assert!(dbg.contains("LlDashChunker"));
    }

    // 24. Chunk byte offsets increase monotonically within segment
    #[test]
    fn test_byte_offset_monotonic() {
        let mut c = LlDashChunker::new(LlDashChunkerConfig::new(0.1, 5));
        for i in 0..6u64 {
            c.push_frame(&[1u8; 512], i * 20, i % 2 == 0);
        }
        let chunks = c.drain_chunks();
        if chunks.len() > 1 {
            for w in chunks.windows(2) {
                assert!(
                    w[1].byte_offset >= w[0].byte_offset,
                    "byte offsets should increase"
                );
            }
        }
    }

    // 25. Chunker availability_time_offset accessor
    #[test]
    fn test_ato_accessor() {
        let mut cfg = LlDashChunkerConfig::default();
        cfg.availability_time_offset = 1.23;
        let c = LlDashChunker::new(cfg);
        assert!((c.availability_time_offset() - 1.23).abs() < 1e-9);
    }
}
