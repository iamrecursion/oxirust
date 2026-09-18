//! Low-Latency HLS (LL-HLS) support per RFC 8216bis / Apple LL-HLS spec.
//!
//! Provides partial segment management, preload hints, blocking playlist reload
//! support, and playlist generation for sub-second live streaming latency.

use crate::StreamError;
use std::collections::VecDeque;

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for Low-Latency HLS operation.
#[derive(Debug, Clone)]
pub struct LlHlsConfig {
    /// Target duration of each full segment in milliseconds.
    pub segment_duration_ms: u32,
    /// Target duration of each partial segment in milliseconds.
    pub part_duration_ms: u32,
    /// Number of full segments to hold back from the live edge for clients that
    /// do not support LL-HLS (HOLD-BACK in EXT-X-SERVER-CONTROL).
    pub holdback_segments: u32,
    /// Target end-to-end latency in milliseconds (used to derive HOLD-BACK).
    pub target_latency_ms: u32,
    /// Whether to support playlist delta updates (`_HLS_skip` query parameter).
    pub playlist_delta_updates: bool,
    /// Maximum number of partial segments to retain per full segment.
    pub max_parts_per_segment: u32,
    /// Number of full segments to retain in the playlist.
    pub playlist_window_segments: usize,
    /// CAN-SKIP-UNTIL value in seconds for EXT-X-SERVER-CONTROL.
    pub can_skip_until: f64,
    /// PART-HOLD-BACK value in seconds for EXT-X-SERVER-CONTROL.
    pub part_hold_back: f64,
}

impl Default for LlHlsConfig {
    fn default() -> Self {
        Self {
            segment_duration_ms: 6000,
            part_duration_ms: 200,
            holdback_segments: 3,
            target_latency_ms: 1000,
            playlist_delta_updates: true,
            max_parts_per_segment: 25,
            playlist_window_segments: 5,
            can_skip_until: 6.0,
            part_hold_back: 0.6,
        }
    }
}

// ─── Partial Segment ─────────────────────────────────────────────────────────

/// A partial segment within an LL-HLS stream.
///
/// Partial segments are sub-divisions of a full segment, allowing clients to
/// begin playback before the full segment is complete.
#[derive(Debug, Clone)]
pub struct PartialSegment {
    /// Full-segment sequence number this partial belongs to.
    pub sequence: u32,
    /// Zero-based index of this partial within the parent segment.
    pub part_index: u32,
    /// Duration of this partial segment in milliseconds.
    pub duration_ms: u32,
    /// Whether this partial is independently decodable (starts on a keyframe).
    pub independent: bool,
    /// Raw media data for this partial segment.
    pub data: Vec<u8>,
    /// Optional URI override; if `None`, a URI is auto-generated.
    pub uri: Option<String>,
    /// Optional byte range within the segment file: `(start_byte, length)`.
    pub byte_range: Option<(u64, u64)>,
}

impl PartialSegment {
    /// Create a new partial segment.
    pub fn new(
        sequence: u32,
        part_index: u32,
        duration_ms: u32,
        independent: bool,
        data: Vec<u8>,
        byte_range: Option<(u64, u64)>,
    ) -> Self {
        Self {
            sequence,
            part_index,
            duration_ms,
            independent,
            data,
            uri: None,
            byte_range,
        }
    }

    /// Return the auto-generated URI for this partial segment.
    pub fn auto_uri(&self) -> String {
        format!("seg{}_part{}.mp4", self.sequence, self.part_index)
    }

    /// Return the effective URI (custom or auto-generated).
    pub fn effective_uri(&self) -> String {
        self.uri.clone().unwrap_or_else(|| self.auto_uri())
    }

    /// Duration as floating-point seconds.
    pub fn duration_secs(&self) -> f64 {
        self.duration_ms as f64 / 1000.0
    }
}

// ─── LlHlsSegment ─────────────────────────────────────────────────────────────

/// A full segment assembled from one or more partial segments, as tracked by
/// the LL-HLS layer.
#[derive(Debug, Clone)]
pub struct LlHlsSegment {
    /// Partial segments that make up this full segment.
    pub parts: Vec<PartialSegment>,
    /// Concatenated raw data for the full segment (may be empty until finalised).
    pub full_data: Vec<u8>,
    /// Media sequence number for this segment.
    pub sequence_num: u64,
    /// Total duration in milliseconds.
    pub duration_ms: u32,
}

impl LlHlsSegment {
    /// Create a new, initially empty full segment.
    pub fn new(sequence_num: u64) -> Self {
        Self {
            parts: Vec::new(),
            full_data: Vec::new(),
            sequence_num,
            duration_ms: 0,
        }
    }

    /// Add a partial segment, accumulating its data and duration.
    pub fn add_part(&mut self, part: PartialSegment) {
        self.duration_ms += part.duration_ms;
        self.full_data.extend_from_slice(&part.data);
        self.parts.push(part);
    }

    /// Return the number of partial segments.
    pub fn part_count(&self) -> usize {
        self.parts.len()
    }

    /// Return the total duration in milliseconds (sum of parts).
    pub fn total_duration_ms(&self) -> u32 {
        self.parts.iter().map(|p| p.duration_ms).sum()
    }

    /// Return `true` if the segment has at least one part and its full_data is
    /// non-empty (i.e. it has been finalised).
    pub fn is_complete(&self) -> bool {
        !self.parts.is_empty() && !self.full_data.is_empty()
    }
}

// ─── HintType ─────────────────────────────────────────────────────────────────

/// The type of preload hint — either a partial segment or an initialization map.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HintType {
    /// Hint for the next partial segment (`TYPE=PART`).
    Part,
    /// Hint for the media initialization section (`TYPE=MAP`).
    Map,
}

impl HintType {
    /// Return the string representation used in the HLS playlist tag.
    pub fn as_str(&self) -> &'static str {
        match self {
            HintType::Part => "PART",
            HintType::Map => "MAP",
        }
    }
}

// ─── Preload Hint ─────────────────────────────────────────────────────────────

/// Hint to clients about the next partial segment or map that will become
/// available, emitted as `#EXT-X-PRELOAD-HINT` in the playlist.
#[derive(Debug, Clone)]
pub struct PreloadHint {
    /// The hint type (PART or MAP).
    pub hint_type: HintType,
    /// URI of the upcoming resource.
    pub uri: String,
    /// Optional byte-range start offset within the URI.
    pub byte_range_start: Option<u64>,
    /// Optional byte-range length.
    pub byte_range_length: Option<u64>,
}

impl PreloadHint {
    /// Create a new preload hint.
    pub fn new(
        hint_type: HintType,
        uri: String,
        byte_range_start: Option<u64>,
        byte_range_length: Option<u64>,
    ) -> Self {
        Self {
            hint_type,
            uri,
            byte_range_start,
            byte_range_length,
        }
    }

    /// Render this hint as an `#EXT-X-PRELOAD-HINT` playlist line.
    pub fn to_tag(&self) -> String {
        let mut tag = format!(
            "#EXT-X-PRELOAD-HINT:TYPE={},URI=\"{}\"",
            self.hint_type.as_str(),
            self.uri
        );
        if let Some(start) = self.byte_range_start {
            tag.push_str(&format!(",BYTERANGE-START={}", start));
        }
        if let Some(len) = self.byte_range_length {
            tag.push_str(&format!(",BYTERANGE-LENGTH={}", len));
        }
        tag
    }
}

// ─── Blocking Playlist Reload ─────────────────────────────────────────────────

/// A blocking playlist reload request as defined in RFC 8216bis §6.2.5.
///
/// The client sends `_HLS_msn=N` (and optionally `_HLS_part=M`) query
/// parameters; the server holds the response until the requested media
/// sequence number (and optional part) is available.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockingReloadRequest {
    /// The requested media sequence number.
    pub msn: u64,
    /// The optional requested part index within the segment.
    pub part: Option<u32>,
}

impl BlockingReloadRequest {
    /// Create a new blocking reload request.
    pub fn new(msn: u64, part: Option<u32>) -> Self {
        Self { msn, part }
    }
}

/// Tracks the current published state of an LL-HLS playlist for the purpose of
/// answering blocking reload requests.
#[derive(Debug, Clone)]
pub struct LlHlsPlaylistState {
    /// The current (latest completed or in-progress) media sequence number.
    pub current_msn: u64,
    /// The index of the most recently published partial segment in the current segment.
    pub current_part: u32,
    /// Total number of parts in the current segment (used for bounds checking).
    pub total_parts_in_segment: u32,
}

impl LlHlsPlaylistState {
    /// Create a new playlist state.
    pub fn new(current_msn: u64, current_part: u32, total_parts_in_segment: u32) -> Self {
        Self {
            current_msn,
            current_part,
            total_parts_in_segment,
        }
    }

    /// Return `true` if this state can fulfill the given blocking reload request.
    ///
    /// A request is fulfillable when:
    /// - `req.msn < self.current_msn`, OR
    /// - `req.msn == self.current_msn` AND `req.part <= self.current_part`
    ///   (or `req.part` is `None`, meaning any part at that MSN suffices).
    pub fn can_fulfill_request(&self, req: &BlockingReloadRequest) -> bool {
        if req.msn < self.current_msn {
            return true;
        }
        if req.msn == self.current_msn {
            return req.part.map_or(true, |p| p <= self.current_part);
        }
        false
    }

    /// Advance to the next partial segment within the current full segment.
    pub fn advance_part(&mut self) {
        self.current_part += 1;
    }

    /// Advance to the next full segment, resetting the part counter to 0.
    pub fn advance_segment(&mut self) {
        self.current_msn += 1;
        self.current_part = 0;
    }
}

// ─── Internal full-segment record ────────────────────────────────────────────

/// An internal representation of a completed full segment.
#[derive(Debug, Clone)]
struct FullSegment {
    sequence: u64,
    duration_ms: u32,
    uri: String,
    parts: Vec<PartialSegment>,
}

impl FullSegment {
    fn duration_secs(&self) -> f64 {
        self.duration_ms as f64 / 1000.0
    }
}

// ─── LL-HLS Playlist ─────────────────────────────────────────────────────────

/// Manages state for an LL-HLS media playlist and generates RFC 8216bis output.
#[derive(Debug)]
pub struct LlHlsPlaylist {
    config: LlHlsConfig,
    /// Completed full segments (rolling window).
    full_segments: VecDeque<FullSegment>,
    /// Partial segments accumulating for the *current* (incomplete) full segment.
    pending_parts: Vec<PartialSegment>,
    /// Sequence number for the next full segment.
    next_sequence: u64,
    /// Media sequence counter (oldest full segment still in playlist).
    media_sequence: u64,
    /// Stream target duration in seconds (max full-segment duration seen).
    target_duration_secs: u32,
}

impl LlHlsPlaylist {
    /// Create a new LL-HLS playlist with the given configuration.
    pub fn new(config: LlHlsConfig) -> Self {
        Self {
            config,
            full_segments: VecDeque::new(),
            pending_parts: Vec::new(),
            next_sequence: 0,
            media_sequence: 0,
            target_duration_secs: 2,
        }
    }

    /// Add a partial segment to the playlist.
    ///
    /// When enough partial segments accumulate to form a full segment, the full
    /// segment is finalised automatically.
    pub fn add_partial_segment(&mut self, part: PartialSegment) {
        self.pending_parts.push(part);
    }

    /// Finalise the current partial segments into a full segment.
    ///
    /// Call this when the encoder signals the end of a segment boundary.
    /// Returns an error if there are no pending parts to finalise.
    pub fn finalize_segment(&mut self) -> Result<(), StreamError> {
        if self.pending_parts.is_empty() {
            return Err(StreamError::Generic(
                "no pending partial segments to finalize".to_string(),
            ));
        }
        let total_ms: u32 = self.pending_parts.iter().map(|p| p.duration_ms).sum();
        let seq = self.next_sequence;
        let uri = format!("seg{}.mp4", seq);
        let seg = FullSegment {
            sequence: seq,
            duration_ms: total_ms,
            uri,
            parts: std::mem::take(&mut self.pending_parts),
        };
        let seg_secs = (total_ms + 999) / 1000; // ceiling seconds
        if seg_secs > self.target_duration_secs {
            self.target_duration_secs = seg_secs;
        }
        self.full_segments.push_back(seg);
        self.next_sequence += 1;

        // Evict old segments beyond the window.
        while self.full_segments.len() > self.config.playlist_window_segments {
            self.full_segments.pop_front();
            self.media_sequence += 1;
        }
        Ok(())
    }

    /// Build and return the current media playlist as an M3U8 string.
    ///
    /// Includes `#EXT-X-PART` tags for all retained partial segments,
    /// `#EXT-X-PART-INF`, `#EXT-X-SERVER-CONTROL`, and a `#EXT-X-PRELOAD-HINT`
    /// for the next expected partial segment.
    pub fn generate_media_playlist(&self) -> String {
        let mut out = String::with_capacity(4096);

        // Header
        out.push_str("#EXTM3U\n");
        out.push_str("#EXT-X-VERSION:9\n");
        out.push_str(&format!(
            "#EXT-X-TARGETDURATION:{}\n",
            self.target_duration_secs
        ));
        out.push_str(&format!("#EXT-X-MEDIA-SEQUENCE:{}\n", self.media_sequence));

        // LL-HLS server control
        let part_hold_back = self.config.part_hold_back;
        let hold_back = self.config.target_latency_ms as f64 / 1000.0;
        let skip_part = if self.config.playlist_delta_updates {
            format!(",CAN-SKIP-UNTIL={:.1}", self.config.can_skip_until)
        } else {
            String::new()
        };
        out.push_str(&format!(
            "#EXT-X-SERVER-CONTROL:CAN-BLOCK-RELOAD=YES,PART-HOLD-BACK={:.3},HOLD-BACK={:.3}{}\n",
            part_hold_back, hold_back, skip_part,
        ));

        // Part target duration
        let part_target = self.config.part_duration_ms as f64 / 1000.0;
        out.push_str(&format!("#EXT-X-PART-INF:PART-TARGET={:.3}\n", part_target));

        out.push('\n');

        // Full segments with their parts
        for seg in &self.full_segments {
            // Emit #EXT-X-PART for each part within this segment
            for part in &seg.parts {
                out.push_str(&self.format_part_tag(part));
            }
            // Full segment
            out.push_str(&format!("#EXTINF:{:.3},\n", seg.duration_secs()));
            out.push_str(&seg.uri);
            out.push('\n');
        }

        // Pending (in-progress) parts
        for part in &self.pending_parts {
            out.push_str(&self.format_part_tag(part));
        }

        // Preload hint for next partial segment
        let hint = self.preload_hint_full();
        out.push_str(&hint.to_tag());
        out.push('\n');

        out
    }

    /// Format a single `#EXT-X-PART` tag line for a partial segment.
    fn format_part_tag(&self, part: &PartialSegment) -> String {
        let indep = if part.independent {
            ",INDEPENDENT=YES"
        } else {
            ""
        };
        let byte_range = if let Some((start, len)) = part.byte_range {
            format!(",BYTERANGE=\"{}@{}\"", len, start)
        } else {
            String::new()
        };
        format!(
            "#EXT-X-PART:DURATION={:.3},URI=\"{}\"{}{}\n",
            part.duration_secs(),
            part.effective_uri(),
            indep,
            byte_range,
        )
    }

    /// Return a reference to the configuration.
    pub fn config(&self) -> &LlHlsConfig {
        &self.config
    }

    /// Return the number of complete full segments currently in the playlist.
    pub fn segment_count(&self) -> usize {
        self.full_segments.len()
    }

    /// Return the number of pending partial segments for the current segment.
    pub fn pending_part_count(&self) -> usize {
        self.pending_parts.len()
    }

    /// Return the current media sequence number.
    pub fn media_sequence(&self) -> u64 {
        self.media_sequence
    }

    /// Return the next full-segment sequence number.
    pub fn next_sequence(&self) -> u64 {
        self.next_sequence
    }

    /// Generate the preload hint for the next expected partial segment.
    ///
    /// The legacy form — kept for API compatibility. Prefer `preload_hint_full`.
    pub fn preload_hint(&self) -> PreloadHint {
        self.preload_hint_full()
    }

    /// Generate the full [`PreloadHint`] (with [`HintType`]) for the next
    /// expected partial segment.
    pub fn preload_hint_full(&self) -> PreloadHint {
        let next_seq = self.next_sequence;
        let next_part = self.pending_parts.len() as u32;
        PreloadHint::new(
            HintType::Part,
            format!("seg{}_part{}.mp4", next_seq, next_part),
            None,
            None,
        )
    }

    /// Return the current blocking-reload state of this playlist.
    ///
    /// The state reflects the latest media sequence number and the number of
    /// pending parts published so far within that segment.
    pub fn blocking_reload_state(&self) -> LlHlsPlaylistState {
        let current_msn = if self.next_sequence > 0 {
            self.next_sequence - 1
        } else {
            0
        };
        let current_part = self.pending_parts.len() as u32;
        LlHlsPlaylistState::new(current_msn, current_part, current_part)
    }

    /// Render the current playlist asynchronously into an [`tokio::io::AsyncWrite`] sink.
    ///
    /// Equivalent to [`Self::generate_media_playlist`] followed by an async
    /// `write_all`. Enabled only when the `async` Cargo feature is on.
    ///
    /// # Errors
    ///
    /// Returns [`StreamError::IoError`] if the underlying writer fails.
    #[cfg(feature = "async")]
    pub async fn write_async<W: tokio::io::AsyncWrite + Unpin>(
        &self,
        w: &mut W,
    ) -> Result<(), StreamError> {
        use tokio::io::AsyncWriteExt;
        let s = self.generate_media_playlist();
        w.write_all(s.as_bytes())
            .await
            .map_err(|e| StreamError::IoError(format!("ll-hls write: {e}")))?;
        Ok(())
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: create a partial segment without a byte range.
    fn make_part(seq: u32, idx: u32, independent: bool) -> PartialSegment {
        PartialSegment::new(seq, idx, 200, independent, vec![0xAA, 0xBB], None)
    }

    // Helper: create a partial segment with a byte range.
    fn make_part_with_range(
        seq: u32,
        idx: u32,
        independent: bool,
        byte_range: (u64, u64),
    ) -> PartialSegment {
        PartialSegment::new(
            seq,
            idx,
            200,
            independent,
            vec![0xAA, 0xBB],
            Some(byte_range),
        )
    }

    // ── LL-HLS core tests (≥15) ────────────────────────────────────────────

    #[test]
    fn test_config_defaults() {
        let cfg = LlHlsConfig::default();
        assert_eq!(cfg.part_duration_ms, 200);
        assert_eq!(cfg.segment_duration_ms, 6000);
        assert_eq!(cfg.holdback_segments, 3);
        assert_eq!(cfg.target_latency_ms, 1000);
        assert!(cfg.playlist_delta_updates);
        assert_eq!(cfg.playlist_window_segments, 5);
        assert!((cfg.can_skip_until - 6.0).abs() < 1e-9);
        assert!((cfg.part_hold_back - 0.6).abs() < 1e-9);
    }

    #[test]
    fn test_partial_segment_auto_uri() {
        let p = make_part(3, 1, true);
        assert_eq!(p.auto_uri(), "seg3_part1.mp4");
    }

    #[test]
    fn test_partial_segment_custom_uri() {
        let mut p = make_part(0, 0, false);
        p.uri = Some("custom/path.mp4".to_string());
        assert_eq!(p.effective_uri(), "custom/path.mp4");
    }

    #[test]
    fn test_partial_segment_duration_secs() {
        let p = make_part(0, 0, false);
        assert!((p.duration_secs() - 0.200).abs() < 1e-9);
    }

    #[test]
    fn test_add_and_pending_count() {
        let mut pl = LlHlsPlaylist::new(LlHlsConfig::default());
        pl.add_partial_segment(make_part(0, 0, true));
        pl.add_partial_segment(make_part(0, 1, false));
        assert_eq!(pl.pending_part_count(), 2);
    }

    #[test]
    fn test_finalize_segment_increments_sequence() {
        let mut pl = LlHlsPlaylist::new(LlHlsConfig::default());
        pl.add_partial_segment(make_part(0, 0, true));
        pl.finalize_segment().expect("finalize should succeed");
        assert_eq!(pl.next_sequence(), 1);
        assert_eq!(pl.segment_count(), 1);
        assert_eq!(pl.pending_part_count(), 0);
    }

    #[test]
    fn test_finalize_empty_returns_error() {
        let mut pl = LlHlsPlaylist::new(LlHlsConfig::default());
        assert!(pl.finalize_segment().is_err());
    }

    #[test]
    fn test_playlist_window_eviction() {
        let mut cfg = LlHlsConfig::default();
        cfg.playlist_window_segments = 3;
        let mut pl = LlHlsPlaylist::new(cfg);
        for i in 0..5u32 {
            pl.add_partial_segment(make_part(i, 0, true));
            pl.finalize_segment().expect("finalize");
        }
        assert_eq!(pl.segment_count(), 3);
        assert_eq!(pl.media_sequence(), 2);
    }

    #[test]
    fn test_generate_media_playlist_header() {
        let mut pl = LlHlsPlaylist::new(LlHlsConfig::default());
        pl.add_partial_segment(make_part(0, 0, true));
        pl.finalize_segment().expect("finalize");
        let m3u8 = pl.generate_media_playlist();
        assert!(m3u8.starts_with("#EXTM3U\n"));
        assert!(m3u8.contains("#EXT-X-VERSION:9"));
        assert!(m3u8.contains("#EXT-X-PART-INF:PART-TARGET="));
        assert!(m3u8.contains("#EXT-X-SERVER-CONTROL:"));
    }

    #[test]
    fn test_generate_playlist_contains_ext_x_part() {
        let mut pl = LlHlsPlaylist::new(LlHlsConfig::default());
        pl.add_partial_segment(make_part(0, 0, true));
        pl.add_partial_segment(make_part(0, 1, false));
        pl.finalize_segment().expect("finalize");
        let m3u8 = pl.generate_media_playlist();
        assert!(m3u8.contains("#EXT-X-PART:"), "should have EXT-X-PART tags");
    }

    #[test]
    fn test_generate_playlist_contains_preload_hint() {
        let pl = LlHlsPlaylist::new(LlHlsConfig::default());
        let m3u8 = pl.generate_media_playlist();
        assert!(m3u8.contains("#EXT-X-PRELOAD-HINT:TYPE=PART,URI="));
    }

    #[test]
    fn test_preload_hint_uri_format() {
        let mut pl = LlHlsPlaylist::new(LlHlsConfig::default());
        pl.add_partial_segment(make_part(0, 0, true));
        let hint = pl.preload_hint();
        assert_eq!(hint.hint_type, HintType::Part);
        // next_sequence=0, pending len=1 → next part index = 1
        assert_eq!(hint.uri, "seg0_part1.mp4");
    }

    #[test]
    fn test_independent_flag_in_playlist() {
        let mut pl = LlHlsPlaylist::new(LlHlsConfig::default());
        pl.add_partial_segment(make_part(0, 0, true));
        pl.add_partial_segment(make_part(0, 1, false));
        pl.finalize_segment().expect("finalize");
        let m3u8 = pl.generate_media_playlist();
        assert!(
            m3u8.contains("INDEPENDENT=YES"),
            "independent part should be flagged"
        );
    }

    #[test]
    fn test_delta_updates_server_control() {
        let mut cfg = LlHlsConfig::default();
        cfg.playlist_delta_updates = true;
        let pl = LlHlsPlaylist::new(cfg);
        let m3u8 = pl.generate_media_playlist();
        assert!(m3u8.contains("CAN-SKIP-UNTIL=6.0"));
    }

    #[test]
    fn test_no_delta_updates_server_control() {
        let mut cfg = LlHlsConfig::default();
        cfg.playlist_delta_updates = false;
        let pl = LlHlsPlaylist::new(cfg);
        let m3u8 = pl.generate_media_playlist();
        assert!(
            !m3u8.contains("CAN-SKIP-UNTIL"),
            "should not include CAN-SKIP-UNTIL"
        );
    }

    #[test]
    fn test_ll_hls_segment_new() {
        let seg = LlHlsSegment::new(42);
        assert_eq!(seg.sequence_num, 42);
        assert_eq!(seg.duration_ms, 0);
        assert!(seg.parts.is_empty());
        assert!(seg.full_data.is_empty());
    }

    #[test]
    fn test_ll_hls_segment_part_count() {
        let mut seg = LlHlsSegment::new(0);
        seg.add_part(make_part(0, 0, true));
        seg.add_part(make_part(0, 1, false));
        assert_eq!(seg.part_count(), 2);
    }

    #[test]
    fn test_ll_hls_segment_total_duration() {
        let mut seg = LlHlsSegment::new(0);
        seg.add_part(make_part(0, 0, true)); // 200 ms
        seg.add_part(make_part(0, 1, false)); // 200 ms
        assert_eq!(seg.total_duration_ms(), 400);
        assert_eq!(seg.duration_ms, 400);
    }

    #[test]
    fn test_ll_hls_segment_is_complete() {
        let mut seg = LlHlsSegment::new(0);
        assert!(!seg.is_complete());
        seg.add_part(make_part(0, 0, true));
        // full_data populated by add_part
        assert!(seg.is_complete());
    }

    #[test]
    fn test_partial_segment_byte_range_in_playlist() {
        let mut pl = LlHlsPlaylist::new(LlHlsConfig::default());
        pl.add_partial_segment(make_part_with_range(0, 0, true, (0, 512)));
        pl.finalize_segment().expect("finalize");
        let m3u8 = pl.generate_media_playlist();
        assert!(
            m3u8.contains("BYTERANGE="),
            "should emit BYTERANGE attribute"
        );
    }

    #[test]
    fn test_config_can_skip_until_in_playlist() {
        let mut cfg = LlHlsConfig::default();
        cfg.can_skip_until = 12.0;
        cfg.playlist_delta_updates = true;
        let pl = LlHlsPlaylist::new(cfg);
        let m3u8 = pl.generate_media_playlist();
        assert!(
            m3u8.contains("CAN-SKIP-UNTIL=12.0"),
            "should reflect custom can_skip_until"
        );
    }

    #[test]
    fn test_part_hold_back_in_playlist() {
        let mut cfg = LlHlsConfig::default();
        cfg.part_hold_back = 0.9;
        let pl = LlHlsPlaylist::new(cfg);
        let m3u8 = pl.generate_media_playlist();
        assert!(
            m3u8.contains("PART-HOLD-BACK=0.900"),
            "should reflect custom part_hold_back"
        );
    }

    // ── PreloadHint tests (≥8) ─────────────────────────────────────────────

    #[test]
    fn test_hint_type_part_str() {
        assert_eq!(HintType::Part.as_str(), "PART");
    }

    #[test]
    fn test_hint_type_map_str() {
        assert_eq!(HintType::Map.as_str(), "MAP");
    }

    #[test]
    fn test_preload_hint_construction() {
        let hint = PreloadHint::new(HintType::Part, "seg0_part0.mp4".to_string(), None, None);
        assert_eq!(hint.hint_type, HintType::Part);
        assert_eq!(hint.uri, "seg0_part0.mp4");
        assert!(hint.byte_range_start.is_none());
        assert!(hint.byte_range_length.is_none());
    }

    #[test]
    fn test_preload_hint_no_byte_range() {
        let hint = PreloadHint::new(HintType::Part, "seg0_part1.mp4".to_string(), None, None);
        let tag = hint.to_tag();
        assert!(!tag.contains("BYTERANGE-START"));
        assert!(!tag.contains("BYTERANGE-LENGTH"));
    }

    #[test]
    fn test_preload_hint_with_byte_range() {
        let hint = PreloadHint::new(
            HintType::Part,
            "seg0_part2.mp4".to_string(),
            Some(1024),
            Some(512),
        );
        let tag = hint.to_tag();
        assert!(tag.contains("BYTERANGE-START=1024"));
        assert!(tag.contains("BYTERANGE-LENGTH=512"));
    }

    #[test]
    fn test_preload_hint_in_playlist_type_part() {
        let pl = LlHlsPlaylist::new(LlHlsConfig::default());
        let m3u8 = pl.generate_media_playlist();
        assert!(m3u8.contains("TYPE=PART"));
    }

    #[test]
    fn test_preload_hint_map_type() {
        let hint = PreloadHint::new(HintType::Map, "init.mp4".to_string(), None, None);
        assert_eq!(hint.hint_type.as_str(), "MAP");
        let tag = hint.to_tag();
        assert!(tag.contains("TYPE=MAP"));
        assert!(tag.contains("URI=\"init.mp4\""));
    }

    #[test]
    fn test_preload_hint_byte_range_length_none_by_default() {
        let hint = PreloadHint::new(HintType::Part, "seg0_part0.mp4".to_string(), Some(0), None);
        assert!(hint.byte_range_length.is_none());
        let tag = hint.to_tag();
        assert!(tag.contains("BYTERANGE-START=0"));
        assert!(!tag.contains("BYTERANGE-LENGTH"));
    }

    #[test]
    fn test_preload_hint_to_tag_format() {
        let hint = PreloadHint::new(HintType::Part, "seg5_part2.mp4".to_string(), None, None);
        let tag = hint.to_tag();
        assert_eq!(tag, "#EXT-X-PRELOAD-HINT:TYPE=PART,URI=\"seg5_part2.mp4\"");
    }

    // ── BlockingReloadRequest tests (≥8) ──────────────────────────────────

    #[test]
    fn test_blocking_reload_can_fulfill_past_msn() {
        let state = LlHlsPlaylistState::new(5, 3, 10);
        let req = BlockingReloadRequest::new(4, None);
        assert!(state.can_fulfill_request(&req));
    }

    #[test]
    fn test_blocking_reload_cannot_fulfill_future_msn() {
        let state = LlHlsPlaylistState::new(5, 3, 10);
        let req = BlockingReloadRequest::new(6, None);
        assert!(!state.can_fulfill_request(&req));
    }

    #[test]
    fn test_blocking_reload_same_msn_no_part() {
        let state = LlHlsPlaylistState::new(5, 3, 10);
        let req = BlockingReloadRequest::new(5, None);
        assert!(state.can_fulfill_request(&req));
    }

    #[test]
    fn test_blocking_reload_same_msn_past_part() {
        let state = LlHlsPlaylistState::new(5, 3, 10);
        let req = BlockingReloadRequest::new(5, Some(2));
        assert!(state.can_fulfill_request(&req));
    }

    #[test]
    fn test_blocking_reload_same_msn_exact_part() {
        let state = LlHlsPlaylistState::new(5, 3, 10);
        let req = BlockingReloadRequest::new(5, Some(3));
        assert!(state.can_fulfill_request(&req));
    }

    #[test]
    fn test_blocking_reload_same_msn_future_part() {
        let state = LlHlsPlaylistState::new(5, 3, 10);
        let req = BlockingReloadRequest::new(5, Some(4));
        assert!(!state.can_fulfill_request(&req));
    }

    #[test]
    fn test_blocking_reload_advance_part() {
        let mut state = LlHlsPlaylistState::new(5, 3, 10);
        state.advance_part();
        assert_eq!(state.current_part, 4);
        assert_eq!(state.current_msn, 5);
    }

    #[test]
    fn test_blocking_reload_advance_segment() {
        let mut state = LlHlsPlaylistState::new(5, 3, 10);
        state.advance_segment();
        assert_eq!(state.current_msn, 6);
        assert_eq!(state.current_part, 0);
    }

    #[test]
    fn test_blocking_reload_request_new() {
        let req = BlockingReloadRequest::new(10, Some(2));
        assert_eq!(req.msn, 10);
        assert_eq!(req.part, Some(2));
    }

    #[test]
    fn test_playlist_state_from_playlist() {
        let mut pl = LlHlsPlaylist::new(LlHlsConfig::default());
        pl.add_partial_segment(make_part(0, 0, true));
        pl.finalize_segment().expect("finalize");
        // After finalize, next_sequence=1, pending is empty
        let state = pl.blocking_reload_state();
        // current_msn = next_sequence - 1 = 0
        assert_eq!(state.current_msn, 0);
        assert_eq!(state.current_part, 0);
    }

    #[test]
    fn test_playlist_state_with_pending_parts() {
        let mut pl = LlHlsPlaylist::new(LlHlsConfig::default());
        pl.add_partial_segment(make_part(0, 0, true));
        pl.finalize_segment().expect("finalize");
        // Add 2 pending parts to segment 1
        pl.add_partial_segment(make_part(1, 0, true));
        pl.add_partial_segment(make_part(1, 1, false));
        let state = pl.blocking_reload_state();
        // current_msn = next_sequence - 1 = 0 (next_sequence became 1 after finalize)
        assert_eq!(state.current_msn, 0);
        // current_part reflects pending parts count = 2
        assert_eq!(state.current_part, 2);
    }
}
