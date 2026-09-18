//! Low-Latency DASH (LL-DASH) with CMAF chunked transfer encoding.
//!
//! Implements the DASH-IF Low Latency specification with:
//! - Chunked Transfer Encoding for partial CMAF segments
//! - Availability timeline management
//! - MPD update scheduling for sub-second latency
//! - Availability start time and presentation delay tracking

use crate::StreamError;
use std::collections::VecDeque;

// ─── LL-DASH Configuration ────────────────────────────────────────────────────

/// Configuration for Low-Latency DASH operation.
#[derive(Debug, Clone)]
pub struct LlDashConfig {
    /// Segment duration in milliseconds (e.g. 2000 for 2-second segments).
    pub segment_duration_ms: u32,
    /// CMAF chunk duration in milliseconds (e.g. 200 for 200 ms chunks).
    pub chunk_duration_ms: u32,
    /// Target end-to-end latency in milliseconds.
    pub target_latency_ms: u32,
    /// MPD update period in milliseconds (how often the manifest is refreshed).
    pub mpd_update_period_ms: u32,
    /// Number of segments to retain in the live sliding window.
    pub live_window_segments: usize,
    /// Suggested presentation delay in milliseconds.
    pub suggested_presentation_delay_ms: u32,
}

impl Default for LlDashConfig {
    fn default() -> Self {
        Self {
            segment_duration_ms: 2000,
            chunk_duration_ms: 200,
            target_latency_ms: 1500,
            mpd_update_period_ms: 500,
            live_window_segments: 5,
            suggested_presentation_delay_ms: 3000,
        }
    }
}

// ─── CMAF Chunk ───────────────────────────────────────────────────────────────

/// A CMAF chunk within a segment, ready for chunked transfer.
///
/// Each chunk corresponds to a single `moof+mdat` pair in the CMAF container
/// and is independently decodable when the track initialization segment has
/// been received.
#[derive(Debug, Clone)]
pub struct LlDashChunk {
    /// Segment sequence number this chunk belongs to.
    pub segment_sequence: u64,
    /// Zero-based index of this chunk within the parent segment.
    pub chunk_index: u32,
    /// Total number of chunks expected in this segment.
    pub total_chunks: u32,
    /// Base media decode time for this chunk (in the representation timescale).
    pub base_media_decode_time: u64,
    /// Duration of this chunk in timescale units.
    pub duration: u64,
    /// Whether this chunk is independently decodable (starts on an I-frame).
    pub independent: bool,
    /// Encoded chunk payload bytes (moof + mdat).
    pub data: Vec<u8>,
}

impl LlDashChunk {
    /// Create a new LL-DASH chunk.
    pub fn new(
        segment_sequence: u64,
        chunk_index: u32,
        total_chunks: u32,
        base_media_decode_time: u64,
        duration: u64,
        independent: bool,
        data: Vec<u8>,
    ) -> Self {
        Self {
            segment_sequence,
            chunk_index,
            total_chunks,
            base_media_decode_time,
            duration,
            independent,
            data,
        }
    }

    /// Return `true` if this is the last chunk in the segment.
    pub fn is_last(&self) -> bool {
        self.chunk_index + 1 >= self.total_chunks
    }

    /// Return the byte length of this chunk.
    pub fn byte_len(&self) -> usize {
        self.data.len()
    }

    /// Format a `Content-Type` header value for chunked CMAF delivery.
    pub fn content_type() -> &'static str {
        "video/mp4; codecs=\"av01\""
    }

    /// Format HTTP chunked-transfer headers for this chunk.
    ///
    /// Returns a header string suitable for prefixing the binary payload.
    /// Each chunk uses `Transfer-Encoding: chunked` with the size in hex.
    pub fn chunk_header(&self) -> String {
        format!("{:x}\r\n", self.data.len())
    }

    /// Return the chunk terminator for HTTP chunked transfer encoding.
    pub fn chunk_terminator() -> &'static str {
        "\r\n"
    }
}

// ─── LL-DASH Segment ─────────────────────────────────────────────────────────

/// A fully assembled LL-DASH segment, comprised of one or more chunks.
#[derive(Debug, Clone)]
pub struct LlDashSegment {
    /// Sequence number (1-based, matching $Number$ in DASH template).
    pub sequence_number: u64,
    /// Start time in representation timescale units.
    pub start_time: u64,
    /// Duration in representation timescale units.
    pub duration: u64,
    /// Assembled chunk data.
    pub chunks: Vec<LlDashChunk>,
    /// Whether all chunks have been received (segment is complete).
    pub complete: bool,
}

impl LlDashSegment {
    /// Create an empty in-progress segment.
    pub fn new(sequence_number: u64, start_time: u64, duration: u64) -> Self {
        Self {
            sequence_number,
            start_time,
            duration,
            chunks: Vec::new(),
            complete: false,
        }
    }

    /// Add a chunk to this segment.
    pub fn add_chunk(&mut self, chunk: LlDashChunk) {
        let is_last = chunk.is_last();
        self.chunks.push(chunk);
        if is_last {
            self.complete = true;
        }
    }

    /// Total number of bytes in all chunks.
    pub fn total_bytes(&self) -> usize {
        self.chunks.iter().map(|c| c.data.len()).sum()
    }

    /// Number of chunks received so far.
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }
}

// ─── LL-DASH Timeline ─────────────────────────────────────────────────────────

/// Manages the live sliding window and MPD timeline for LL-DASH.
///
/// Tracks all in-progress and completed segments, generates MPD patch data,
/// and maintains the availability time offset for chunked delivery.
#[derive(Debug)]
pub struct LlDashTimeline {
    config: LlDashConfig,
    /// Completed segments in the live window (sliding).
    completed_segments: VecDeque<LlDashSegment>,
    /// The currently accumulating segment (if any).
    current_segment: Option<LlDashSegment>,
    /// Timescale for this representation (ticks per second).
    timescale: u32,
    /// Sequence counter for the next segment.
    next_sequence: u64,
    /// MPD publish time counter (monotonically increasing).
    publish_time: u64,
}

impl LlDashTimeline {
    /// Create a new timeline with the given configuration and timescale.
    pub fn new(config: LlDashConfig, timescale: u32) -> Self {
        Self {
            config,
            completed_segments: VecDeque::new(),
            current_segment: None,
            timescale: timescale.max(1),
            next_sequence: 1,
            publish_time: 0,
        }
    }

    /// Segment duration in timescale units.
    pub fn segment_duration_ticks(&self) -> u64 {
        self.config.segment_duration_ms as u64 * self.timescale as u64 / 1000
    }

    /// Chunk duration in timescale units.
    pub fn chunk_duration_ticks(&self) -> u64 {
        self.config.chunk_duration_ms as u64 * self.timescale as u64 / 1000
    }

    /// Begin a new segment at the given base media decode time.
    ///
    /// Returns an error if a segment is already in progress.
    pub fn begin_segment(&mut self, base_media_decode_time: u64) -> Result<(), StreamError> {
        if self.current_segment.is_some() {
            return Err(StreamError::Generic(
                "segment already in progress; call finalize_segment first".to_string(),
            ));
        }
        let seq = self.next_sequence;
        self.next_sequence += 1;
        self.current_segment = Some(LlDashSegment::new(
            seq,
            base_media_decode_time,
            self.segment_duration_ticks(),
        ));
        Ok(())
    }

    /// Add a chunk to the current in-progress segment.
    ///
    /// Returns an error if no segment is in progress.
    pub fn add_chunk(&mut self, chunk: LlDashChunk) -> Result<(), StreamError> {
        match &mut self.current_segment {
            Some(seg) => {
                seg.add_chunk(chunk);
                Ok(())
            }
            None => Err(StreamError::Generic(
                "no segment in progress; call begin_segment first".to_string(),
            )),
        }
    }

    /// Finalize the current segment and move it to the completed window.
    ///
    /// Returns the finalized segment or an error if no segment was in progress.
    pub fn finalize_segment(&mut self) -> Result<LlDashSegment, StreamError> {
        let seg = self
            .current_segment
            .take()
            .ok_or_else(|| StreamError::Generic("no segment in progress".to_string()))?;
        self.completed_segments.push_back(seg.clone());
        // Evict old segments beyond the live window.
        while self.completed_segments.len() > self.config.live_window_segments {
            self.completed_segments.pop_front();
        }
        self.publish_time += 1;
        Ok(seg)
    }

    /// Number of complete segments currently in the live window.
    pub fn segment_count(&self) -> usize {
        self.completed_segments.len()
    }

    /// Whether a segment is currently being assembled.
    pub fn has_current_segment(&self) -> bool {
        self.current_segment.is_some()
    }

    /// Return a reference to all completed segments.
    pub fn segments(&self) -> &VecDeque<LlDashSegment> {
        &self.completed_segments
    }

    /// Return the current publish time counter.
    pub fn publish_time(&self) -> u64 {
        self.publish_time
    }

    /// Generate an LL-DASH MPD XML document for the current timeline state.
    ///
    /// The output uses the DASH-IF live profile with chunked CMAF addressing
    /// and an `availabilityTimeOffset` equal to the chunk duration minus
    /// the segment duration (expressed in the same unit as `@duration`).
    ///
    /// # Parameters
    ///
    /// - `base_url`: The base URL prepended to all segment URIs.
    /// - `codec_string`: The codec string for the `Representation@codecs` attribute.
    /// - `bandwidth_bps`: Representation bandwidth in bits per second.
    pub fn generate_mpd(&self, base_url: &str, codec_string: &str, bandwidth_bps: u64) -> String {
        let seg_dur = self.segment_duration_ticks();
        let chunk_dur = self.chunk_duration_ticks();

        // availabilityTimeOffset = segmentDuration - chunkDuration (in seconds)
        let ato_secs = if seg_dur > chunk_dur {
            (seg_dur - chunk_dur) as f64 / self.timescale as f64
        } else {
            0.0
        };

        let target_latency_secs = self.config.target_latency_ms as f64 / 1000.0;
        let suggested_delay_secs = self.config.suggested_presentation_delay_ms as f64 / 1000.0;
        let update_period_secs = self.config.mpd_update_period_ms as f64 / 1000.0;

        let mut out = String::with_capacity(2048);
        out.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        out.push_str("<MPD xmlns=\"urn:mpeg:dash:schema:mpd:2011\"\n");
        out.push_str("     xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"\n");
        out.push_str("     xsi:schemaLocation=\"urn:mpeg:dash:schema:mpd:2011 DASH-MPD.xsd\"\n");
        out.push_str("     profiles=\"urn:mpeg:dash:profile:isoff-live:2011\"\n");
        out.push_str("     type=\"dynamic\"\n");
        out.push_str(&format!(
            "     minBufferTime=\"PT{:.3}S\"\n",
            target_latency_secs
        ));
        out.push_str(&format!(
            "     suggestedPresentationDelay=\"PT{:.3}S\"\n",
            suggested_delay_secs
        ));
        out.push_str(&format!(
            "     minimumUpdatePeriod=\"PT{:.3}S\"\n",
            update_period_secs
        ));
        out.push_str("     availabilityStartTime=\"1970-01-01T00:00:00Z\"\n");
        out.push_str(">\n");
        out.push_str("  <Period start=\"PT0S\" id=\"period-0\">\n");
        out.push_str("    <AdaptationSet mimeType=\"video/mp4\" segmentAlignment=\"true\" startWithSAP=\"1\">\n");
        out.push_str(&format!(
            "      <Representation id=\"v0\" bandwidth=\"{}\" codecs=\"{}\">\n",
            bandwidth_bps, codec_string
        ));

        if !base_url.is_empty() {
            out.push_str(&format!(
                "        <BaseURL>{}</BaseURL>\n",
                xml_escape_ll(base_url)
            ));
        }

        out.push_str("        <SegmentTemplate\n");
        out.push_str(&format!("          timescale=\"{}\"\n", self.timescale));
        out.push_str(&format!("          duration=\"{}\"\n", seg_dur));
        out.push_str("          initialization=\"init.mp4\"\n");
        out.push_str("          media=\"seg$Number$.m4s\"\n");
        out.push_str("          startNumber=\"1\"\n");
        out.push_str(&format!(
            "          availabilityTimeOffset=\"{:.3}\"\n",
            ato_secs
        ));
        out.push_str("          availabilityTimeComplete=\"false\"\n");
        out.push_str("        />\n");

        out.push_str("      </Representation>\n");
        out.push_str("    </AdaptationSet>\n");
        out.push_str("  </Period>\n");
        out.push_str("</MPD>\n");
        out
    }

    /// Render the current MPD asynchronously into an [`tokio::io::AsyncWrite`] sink.
    ///
    /// Calls [`Self::generate_mpd`] internally and streams the result via
    /// `write_all`. Enabled only when the `async` Cargo feature is on.
    ///
    /// # Errors
    ///
    /// Returns [`StreamError::IoError`] if the underlying writer fails.
    #[cfg(feature = "async")]
    pub async fn write_async<W: tokio::io::AsyncWrite + Unpin>(
        &self,
        base_url: &str,
        codec_string: &str,
        bandwidth_bps: u64,
        w: &mut W,
    ) -> Result<(), StreamError> {
        use tokio::io::AsyncWriteExt;
        let s = self.generate_mpd(base_url, codec_string, bandwidth_bps);
        w.write_all(s.as_bytes())
            .await
            .map_err(|e| StreamError::IoError(format!("ll-dash write: {e}")))?;
        Ok(())
    }
}

/// Escape the five XML predefined entities.
fn xml_escape_ll(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            other => out.push(other),
        }
    }
    out
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config() -> LlDashConfig {
        LlDashConfig {
            segment_duration_ms: 2000,
            chunk_duration_ms: 200,
            target_latency_ms: 1500,
            mpd_update_period_ms: 500,
            live_window_segments: 5,
            suggested_presentation_delay_ms: 3000,
        }
    }

    fn make_timeline() -> LlDashTimeline {
        LlDashTimeline::new(make_config(), 90_000)
    }

    fn make_chunk(seg_seq: u64, idx: u32, total: u32, independent: bool) -> LlDashChunk {
        LlDashChunk::new(
            seg_seq,
            idx,
            total,
            idx as u64 * 18_000,
            18_000,
            independent,
            vec![0xAB; 100],
        )
    }

    // ── LlDashConfig ─────────────────────────────────────────────────────────

    #[test]
    fn test_default_config() {
        let cfg = LlDashConfig::default();
        assert_eq!(cfg.segment_duration_ms, 2000);
        assert_eq!(cfg.chunk_duration_ms, 200);
        assert_eq!(cfg.target_latency_ms, 1500);
        assert_eq!(cfg.live_window_segments, 5);
    }

    // ── LlDashChunk ──────────────────────────────────────────────────────────

    #[test]
    fn test_chunk_is_last() {
        let c = make_chunk(1, 9, 10, false);
        assert!(c.is_last());
        let c2 = make_chunk(1, 8, 10, false);
        assert!(!c2.is_last());
    }

    #[test]
    fn test_chunk_byte_len() {
        let c = make_chunk(1, 0, 5, true);
        assert_eq!(c.byte_len(), 100);
    }

    #[test]
    fn test_chunk_header_format() {
        let c = make_chunk(1, 0, 5, true);
        let header = c.chunk_header();
        // 100 in hex = "64"
        assert!(header.starts_with("64"), "header={header:?}");
        assert!(header.ends_with("\r\n"));
    }

    // ── LlDashSegment ────────────────────────────────────────────────────────

    #[test]
    fn test_segment_add_chunk_completes() {
        let mut seg = LlDashSegment::new(1, 0, 180_000);
        assert!(!seg.complete);
        seg.add_chunk(make_chunk(1, 0, 1, true)); // total=1, so is_last=true
        assert!(seg.complete);
        assert_eq!(seg.chunk_count(), 1);
        assert_eq!(seg.total_bytes(), 100);
    }

    #[test]
    fn test_segment_not_complete_until_last_chunk() {
        let mut seg = LlDashSegment::new(1, 0, 180_000);
        seg.add_chunk(make_chunk(1, 0, 3, true));
        assert!(!seg.complete, "only first chunk of 3 added");
        seg.add_chunk(make_chunk(1, 1, 3, false));
        assert!(!seg.complete, "only second chunk of 3 added");
        seg.add_chunk(make_chunk(1, 2, 3, false));
        assert!(seg.complete, "all chunks added");
    }

    // ── LlDashTimeline ───────────────────────────────────────────────────────

    #[test]
    fn test_timeline_begin_and_add_chunk() {
        let mut tl = make_timeline();
        tl.begin_segment(0).expect("begin_segment");
        assert!(tl.has_current_segment());
        tl.add_chunk(make_chunk(1, 0, 10, true)).expect("add_chunk");
    }

    #[test]
    fn test_timeline_begin_segment_twice_errors() {
        let mut tl = make_timeline();
        tl.begin_segment(0).expect("first begin");
        assert!(tl.begin_segment(0).is_err(), "second begin should fail");
    }

    #[test]
    fn test_timeline_add_chunk_without_segment_errors() {
        let mut tl = make_timeline();
        let err = tl.add_chunk(make_chunk(1, 0, 1, true));
        assert!(err.is_err());
    }

    #[test]
    fn test_timeline_finalize_segment() {
        let mut tl = make_timeline();
        tl.begin_segment(0).expect("begin");
        for i in 0..3u32 {
            tl.add_chunk(make_chunk(1, i, 3, i == 0)).expect("add");
        }
        let seg = tl.finalize_segment().expect("finalize");
        assert!(seg.complete);
        assert_eq!(seg.chunk_count(), 3);
        assert_eq!(tl.segment_count(), 1);
    }

    #[test]
    fn test_timeline_live_window_eviction() {
        let mut cfg = make_config();
        cfg.live_window_segments = 3;
        let mut tl = LlDashTimeline::new(cfg, 90_000);
        for seq in 0..5u64 {
            tl.begin_segment(seq * 180_000).expect("begin");
            tl.add_chunk(make_chunk(seq + 1, 0, 1, true)).expect("add");
            tl.finalize_segment().expect("finalize");
        }
        assert_eq!(tl.segment_count(), 3, "window should cap at 3");
    }

    #[test]
    fn test_timeline_sequence_increments() {
        let mut tl = make_timeline();
        for i in 0..3u64 {
            tl.begin_segment(i * 180_000).expect("begin");
            tl.add_chunk(make_chunk(i + 1, 0, 1, true)).expect("add");
            let seg = tl.finalize_segment().expect("finalize");
            assert_eq!(seg.sequence_number, i + 1);
        }
    }

    #[test]
    fn test_segment_duration_ticks() {
        let tl = make_timeline();
        // 2000 ms * 90_000 / 1000 = 180_000
        assert_eq!(tl.segment_duration_ticks(), 180_000);
    }

    #[test]
    fn test_chunk_duration_ticks() {
        let tl = make_timeline();
        // 200 ms * 90_000 / 1000 = 18_000
        assert_eq!(tl.chunk_duration_ticks(), 18_000);
    }

    // ── generate_mpd ─────────────────────────────────────────────────────────

    #[test]
    fn test_generate_mpd_basic() {
        let tl = make_timeline();
        let mpd = tl.generate_mpd("https://cdn.example.com/live/", "av01.0.04M.08", 2_500_000);
        assert!(mpd.contains("<?xml"), "mpd={mpd}");
        assert!(mpd.contains("<MPD"), "mpd={mpd}");
        assert!(mpd.contains("type=\"dynamic\""), "mpd={mpd}");
        assert!(mpd.contains("availabilityTimeOffset="), "mpd={mpd}");
        assert!(
            mpd.contains("availabilityTimeComplete=\"false\""),
            "mpd={mpd}"
        );
    }

    #[test]
    fn test_generate_mpd_contains_codec() {
        let tl = make_timeline();
        let mpd = tl.generate_mpd("", "av01.0.04M.08", 5_000_000);
        assert!(mpd.contains("av01.0.04M.08"), "mpd={mpd}");
    }

    #[test]
    fn test_generate_mpd_contains_segment_template() {
        let tl = make_timeline();
        let mpd = tl.generate_mpd("", "av01", 1_000_000);
        assert!(mpd.contains("<SegmentTemplate"), "mpd={mpd}");
        assert!(mpd.contains("timescale=\"90000\""), "mpd={mpd}");
        assert!(mpd.contains("duration=\"180000\""), "mpd={mpd}");
    }

    #[test]
    fn test_availability_time_offset_calculated() {
        let tl = make_timeline();
        // ato = (180000 - 18000) / 90000 = 1.8 seconds
        let mpd = tl.generate_mpd("", "av01", 1_000_000);
        assert!(
            mpd.contains("availabilityTimeOffset=\"1.800\""),
            "mpd={mpd}"
        );
    }

    #[test]
    fn test_xml_escape_in_base_url() {
        let tl = make_timeline();
        let mpd = tl.generate_mpd("https://cdn.example.com/live&v=1/", "av01", 1_000_000);
        assert!(
            mpd.contains("&amp;"),
            "ampersand should be escaped; mpd={mpd}"
        );
    }
}
