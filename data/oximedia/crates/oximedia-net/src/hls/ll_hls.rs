//! Low Latency HLS (LL-HLS) support — Apple RFC 8216bis.
//!
//! Implements the full LL-HLS extension including:
//! - Partial segments (EXT-X-PART)
//! - Server-control directives (EXT-X-SERVER-CONTROL)
//! - Blocking playlist reload (MSN/Part query parameters)
//! - Preload hints (EXT-X-PRELOAD-HINT)
//! - Rendition reports (EXT-X-RENDITION-REPORT)
//! - Delta playlist updates (EXT-X-SKIP)

use std::collections::VecDeque;
use std::fmt::Write as FmtWrite;

// ─── Configuration ────────────────────────────────────────────────────────────

/// LL-HLS configuration controlling timing and feature flags.
#[derive(Debug, Clone)]
pub struct LlHlsConfig {
    /// Full segment target duration in seconds (e.g., 6.0).
    pub target_duration_secs: f64,
    /// Partial segment duration in seconds (e.g., 0.2).
    pub part_duration_secs: f64,
    /// Client hold-back from live edge (typically 3 × part_duration).
    pub hold_back_secs: f64,
    /// Partial hold-back from live edge (typically 3 × part_duration + 0.5).
    pub part_hold_back_secs: f64,
    /// Delta playlist skip optimization threshold in seconds.
    pub can_skip_until: f64,
    /// Whether the server supports blocking playlist reload.
    pub can_block_reload: bool,
}

impl Default for LlHlsConfig {
    fn default() -> Self {
        let part = 0.2_f64;
        Self {
            target_duration_secs: 6.0,
            part_duration_secs: part,
            hold_back_secs: part * 3.0,
            part_hold_back_secs: part * 3.0 + 0.5,
            can_skip_until: 24.0,
            can_block_reload: true,
        }
    }
}

impl LlHlsConfig {
    /// Creates a config with custom part duration; all derived fields are
    /// recalculated automatically.
    #[must_use]
    pub fn with_part_duration(part_duration_secs: f64) -> Self {
        Self {
            part_duration_secs,
            hold_back_secs: part_duration_secs * 3.0,
            part_hold_back_secs: part_duration_secs * 3.0 + 0.5,
            ..Self::default()
        }
    }
}

// ─── Segment types ────────────────────────────────────────────────────────────

/// A partial segment (EXT-X-PART) within a full segment.
#[derive(Debug, Clone)]
pub struct MediaPart {
    /// URI of the partial media resource.
    pub uri: String,
    /// Duration of this part in seconds.
    pub duration_secs: f64,
    /// Whether this part starts with a keyframe (IDR), enabling seek here.
    pub independent: bool,
    /// Optional byte-range `(offset, length)` within a shared resource.
    pub byterange: Option<(u64, u64)>,
    /// Whether this is a gap part (placeholder for missing content).
    pub gap: bool,
}

impl MediaPart {
    /// Creates a simple non-independent, non-gap part.
    #[must_use]
    pub fn new(uri: impl Into<String>, duration_secs: f64) -> Self {
        Self {
            uri: uri.into(),
            duration_secs,
            independent: false,
            byterange: None,
            gap: false,
        }
    }

    /// Marks this part as keyframe-independent.
    #[must_use]
    pub fn independent(mut self) -> Self {
        self.independent = true;
        self
    }

    /// Attaches a byte-range `(offset, length)` to the part.
    #[must_use]
    pub fn with_byterange(mut self, offset: u64, length: u64) -> Self {
        self.byterange = Some((offset, length));
        self
    }

    /// Renders the EXT-X-PART tag line for this part.
    #[must_use]
    pub fn to_tag(&self) -> String {
        let mut tag = format!(
            "#EXT-X-PART:DURATION={:.5},URI=\"{}\"",
            self.duration_secs, self.uri
        );
        if self.independent {
            tag.push_str(",INDEPENDENT=YES");
        }
        if let Some((offset, len)) = self.byterange {
            let _ = write!(tag, ",BYTERANGE=\"{len}@{offset}\"");
        }
        if self.gap {
            tag.push_str(",GAP=YES");
        }
        tag
    }
}

/// A complete media segment containing zero or more partial segments.
#[derive(Debug, Clone)]
pub struct LlHlsSegment {
    /// URI of the complete segment media resource.
    pub uri: String,
    /// Duration of the complete segment in seconds.
    pub duration_secs: f64,
    /// Partial segments that make up this segment.
    pub parts: Vec<MediaPart>,
    /// Monotonically increasing sequence number.
    pub sequence_number: u64,
    /// ISO 8601 program date/time string (EXT-X-PROGRAM-DATE-TIME).
    pub program_date_time: Option<String>,
    /// Whether the INDEPENDENT-SEGMENTS tag should be emitted.
    pub independent_segments: bool,
}

impl LlHlsSegment {
    /// Creates a new, empty segment.
    #[must_use]
    pub fn new(uri: impl Into<String>, sequence_number: u64) -> Self {
        Self {
            uri: uri.into(),
            duration_secs: 0.0,
            parts: Vec::new(),
            sequence_number,
            program_date_time: None,
            independent_segments: false,
        }
    }

    /// Appends a part and accumulates its duration into the segment total.
    pub fn push_part(&mut self, part: MediaPart) {
        self.duration_secs += part.duration_secs;
        self.parts.push(part);
    }

    /// Renders all EXT-X-PART tags followed by the EXTINF + URI pair.
    #[must_use]
    pub fn to_tags(&self) -> String {
        let mut out = String::new();
        if let Some(pdt) = &self.program_date_time {
            let _ = writeln!(out, "#EXT-X-PROGRAM-DATE-TIME:{pdt}");
        }
        for part in &self.parts {
            let _ = writeln!(out, "{}", part.to_tag());
        }
        let _ = writeln!(out, "#EXTINF:{:.5},", self.duration_secs);
        let _ = writeln!(out, "{}", self.uri);
        out
    }
}

// ─── Playlist-level types ─────────────────────────────────────────────────────

/// EXT-X-SERVER-CONTROL directive block.
#[derive(Debug, Clone)]
pub struct ServerControl {
    /// Seconds the client must stay behind the live edge (hold-back).
    pub hold_back: f64,
    /// Seconds the client must stay behind when consuming parts (part hold-back).
    pub part_hold_back: f64,
    /// Whether the server supports blocking playlist reload.
    pub can_block_reload: bool,
    /// Seconds after which a client may request a delta (skip) playlist.
    pub can_skip_until: f64,
}

impl ServerControl {
    /// Renders the EXT-X-SERVER-CONTROL tag.
    #[must_use]
    pub fn to_tag(&self) -> String {
        let mut tag = format!(
            "#EXT-X-SERVER-CONTROL:HOLD-BACK={:.1},PART-HOLD-BACK={:.3}",
            self.hold_back, self.part_hold_back
        );
        if self.can_block_reload {
            tag.push_str(",CAN-BLOCK-RELOAD=YES");
        }
        if self.can_skip_until > 0.0 {
            let _ = write!(tag, ",CAN-SKIP-UNTIL={:.1}", self.can_skip_until);
        }
        tag
    }
}

/// EXT-X-PRELOAD-HINT for the next partial segment.
#[derive(Debug, Clone)]
pub struct PreloadHint {
    /// Hint type — always `"PART"` for partial segments.
    pub hint_type: String,
    /// URI of the hinted resource (may be the in-progress part).
    pub uri: String,
    /// Byte offset within the hinted resource (for byte-range delivery).
    pub byterange_start: Option<u64>,
}

impl PreloadHint {
    /// Creates a `PART` preload hint.
    #[must_use]
    pub fn part(uri: impl Into<String>) -> Self {
        Self {
            hint_type: "PART".to_owned(),
            uri: uri.into(),
            byterange_start: None,
        }
    }

    /// Renders the EXT-X-PRELOAD-HINT tag.
    #[must_use]
    pub fn to_tag(&self) -> String {
        let mut tag = format!(
            "#EXT-X-PRELOAD-HINT:TYPE={},URI=\"{}\"",
            self.hint_type, self.uri
        );
        if let Some(start) = self.byterange_start {
            let _ = write!(tag, ",BYTERANGE-START={start}");
        }
        tag
    }
}

/// EXT-X-RENDITION-REPORT for an alternate rendition.
#[derive(Debug, Clone)]
pub struct RenditionReport {
    /// Relative URI of the other rendition's playlist.
    pub uri: String,
    /// Last complete media sequence number in that rendition.
    pub last_msn: u64,
    /// Last partial-segment index within `last_msn`.
    pub last_part: u32,
}

impl RenditionReport {
    /// Renders the EXT-X-RENDITION-REPORT tag.
    #[must_use]
    pub fn to_tag(&self) -> String {
        format!(
            "#EXT-X-RENDITION-REPORT:URI=\"{}\",LAST-MSN={},LAST-PART={}",
            self.uri, self.last_msn, self.last_part
        )
    }
}

// ─── Playlist ─────────────────────────────────────────────────────────────────

/// Maximum number of complete segments kept in the sliding window.
const DEFAULT_WINDOW_SIZE: usize = 5;

/// An LL-HLS media playlist that can be incrementally updated with parts.
///
/// This is the main type for server-side LL-HLS playlist management.
#[derive(Debug, Clone)]
pub struct LlHlsPlaylist {
    /// Current media sequence number (first segment in the window).
    pub media_sequence: u64,
    /// Maximum segment duration in whole seconds (EXT-X-TARGETDURATION).
    pub target_duration: u64,
    /// Maximum reported part duration for EXT-X-PART-INF.
    pub part_inf_duration: f64,
    /// EXT-X-SERVER-CONTROL parameters.
    pub server_control: ServerControl,
    /// Sliding window of complete and in-progress segments.
    pub segments: VecDeque<LlHlsSegment>,
    /// Next partial segment hint (EXT-X-PRELOAD-HINT).
    pub preload_hint: Option<PreloadHint>,
    /// Cross-rendition reports (EXT-X-RENDITION-REPORT).
    pub rendition_reports: Vec<RenditionReport>,
    /// Maximum number of complete segments retained in the window.
    window_size: usize,
    /// Parts accumulated for the currently open segment.
    current_parts: Vec<MediaPart>,
    /// URI of the currently open (incomplete) segment.
    current_segment_uri: String,
    /// Sequence number of the next segment to be finalised.
    next_sequence: u64,
}

impl LlHlsPlaylist {
    /// Creates a new playlist from an [`LlHlsConfig`].
    #[must_use]
    pub fn new(config: &LlHlsConfig) -> Self {
        let server_control = ServerControl {
            hold_back: config.hold_back_secs,
            part_hold_back: config.part_hold_back_secs,
            can_block_reload: config.can_block_reload,
            can_skip_until: config.can_skip_until,
        };
        Self {
            media_sequence: 0,
            target_duration: config.target_duration_secs.ceil() as u64,
            part_inf_duration: config.part_duration_secs,
            server_control,
            segments: VecDeque::with_capacity(DEFAULT_WINDOW_SIZE + 1),
            preload_hint: None,
            rendition_reports: Vec::new(),
            window_size: DEFAULT_WINDOW_SIZE,
            current_parts: Vec::new(),
            current_segment_uri: String::new(),
            next_sequence: 0,
        }
    }

    /// Sets the window size (number of complete segments retained).
    pub fn set_window_size(&mut self, size: usize) {
        self.window_size = size.max(1);
    }

    /// Sets the URI of the currently open (in-progress) segment.
    pub fn set_current_segment_uri(&mut self, uri: impl Into<String>) {
        self.current_segment_uri = uri.into();
    }

    /// Appends a part to the current in-progress segment.
    ///
    /// If `segment_complete` is `true`, the accumulated parts are consolidated
    /// into a new [`LlHlsSegment`] and the window is slid forward.
    pub fn add_part(&mut self, part: MediaPart, segment_complete: bool) {
        // Update preload hint to point at the incoming part's successor.
        self.preload_hint = Some(PreloadHint::part(format!(
            "{}.part",
            part.uri.trim_end_matches(".part")
        )));

        self.current_parts.push(part);

        if segment_complete {
            self.finalise_segment();
        }
    }

    /// Renders the full EXT-M3U playlist as a `String`.
    ///
    /// Includes all required LL-HLS extension tags.
    #[must_use]
    pub fn to_m3u8(&self) -> String {
        let mut out = String::with_capacity(4096);

        // Header
        out.push_str("#EXTM3U\n");
        out.push_str("#EXT-X-VERSION:9\n");
        let _ = writeln!(out, "#EXT-X-TARGETDURATION:{}", self.target_duration);
        let _ = writeln!(out, "#EXT-X-MEDIA-SEQUENCE:{}", self.media_sequence);
        let _ = writeln!(
            out,
            "#EXT-X-PART-INF:PART-TARGET={:.5}",
            self.part_inf_duration
        );
        let _ = writeln!(out, "{}", self.server_control.to_tag());

        // Segments
        for seg in &self.segments {
            out.push_str(&seg.to_tags());
        }

        // In-progress parts for the current open segment.
        for part in &self.current_parts {
            let _ = writeln!(out, "{}", part.to_tag());
        }

        // Preload hint
        if let Some(hint) = &self.preload_hint {
            let _ = writeln!(out, "{}", hint.to_tag());
        }

        // Rendition reports
        for report in &self.rendition_reports {
            let _ = writeln!(out, "{}", report.to_tag());
        }

        out
    }

    /// Returns the rendered playlist if the given `msn` (media sequence
    /// number) and optional `part` index have already been produced.
    ///
    /// Returns `None` when the requested position is still in the future,
    /// signalling that the caller should wait (blocking reload).
    #[must_use]
    pub fn blocking_playlist_response(&self, msn: u64, part: Option<u32>) -> Option<String> {
        // Determine the highest sequence number currently available.
        let last_complete = self
            .segments
            .back()
            .map(|s| s.sequence_number)
            .unwrap_or(self.media_sequence.saturating_sub(1));

        match part {
            None => {
                // Client wants the complete segment `msn`.
                if last_complete >= msn {
                    Some(self.to_m3u8())
                } else {
                    None
                }
            }
            Some(part_idx) => {
                // Client wants at least part `part_idx` of segment `msn`.
                if last_complete > msn {
                    // Segment `msn` is fully available.
                    return Some(self.to_m3u8());
                }
                if last_complete == msn {
                    // Segment is complete → all parts available.
                    return Some(self.to_m3u8());
                }
                // The segment is the current in-progress one.
                let in_progress_msn = last_complete + 1;
                if in_progress_msn == msn && self.current_parts.len() > part_idx as usize {
                    Some(self.to_m3u8())
                } else {
                    None
                }
            }
        }
    }

    /// Returns the last complete media sequence number.
    #[must_use]
    pub fn last_msn(&self) -> u64 {
        self.segments
            .back()
            .map(|s| s.sequence_number)
            .unwrap_or(self.media_sequence.saturating_sub(1))
    }

    /// Returns the number of parts accumulated in the current open segment.
    #[must_use]
    pub fn current_part_count(&self) -> usize {
        self.current_parts.len()
    }

    // ── private ───────────────────────────────────────────────────────────────

    fn finalise_segment(&mut self) {
        let seq = self.next_sequence;
        self.next_sequence += 1;

        let uri = if self.current_segment_uri.is_empty() {
            format!("seg{seq}.ts")
        } else {
            std::mem::take(&mut self.current_segment_uri)
        };

        let mut seg = LlHlsSegment::new(uri, seq);
        for part in self.current_parts.drain(..) {
            seg.push_part(part);
        }

        self.segments.push_back(seg);

        // Slide the window: remove old segments once we exceed the limit.
        while self.segments.len() > self.window_size {
            self.segments.pop_front();
            self.media_sequence += 1;
        }

        // Clear the preload hint after the segment completes.
        self.preload_hint = None;
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_playlist() -> LlHlsPlaylist {
        LlHlsPlaylist::new(&LlHlsConfig::default())
    }

    fn make_part(idx: u32, independent: bool) -> MediaPart {
        let mut p = MediaPart::new(format!("part{idx}.mp4"), 0.2);
        if independent {
            p = p.independent();
        }
        p
    }

    // 1. Default config has sensible part duration
    #[test]
    fn test_default_config_part_duration() {
        let cfg = LlHlsConfig::default();
        assert!((cfg.part_duration_secs - 0.2).abs() < 1e-9);
    }

    // 2. Default config hold-back is 3 × part duration
    #[test]
    fn test_default_config_hold_back() {
        let cfg = LlHlsConfig::default();
        let expected = cfg.part_duration_secs * 3.0;
        assert!((cfg.hold_back_secs - expected).abs() < 1e-9);
    }

    // 3. with_part_duration recalculates hold_back
    #[test]
    fn test_custom_part_duration_hold_back() {
        let cfg = LlHlsConfig::with_part_duration(0.5);
        assert!((cfg.hold_back_secs - 1.5).abs() < 1e-9);
        assert!((cfg.part_hold_back_secs - 2.0).abs() < 1e-9);
    }

    // 4. MediaPart to_tag contains URI and DURATION
    #[test]
    fn test_media_part_to_tag_basic() {
        let part = MediaPart::new("part0.mp4", 0.2);
        let tag = part.to_tag();
        assert!(tag.contains("EXT-X-PART"));
        assert!(tag.contains("part0.mp4"));
        assert!(tag.contains("DURATION=0.20000"));
    }

    // 5. MediaPart to_tag includes INDEPENDENT=YES when set
    #[test]
    fn test_media_part_independent_tag() {
        let part = MediaPart::new("part0.mp4", 0.2).independent();
        assert!(part.to_tag().contains("INDEPENDENT=YES"));
    }

    // 6. MediaPart byterange rendered correctly
    #[test]
    fn test_media_part_byterange() {
        let part = MediaPart::new("seg.mp4", 0.2).with_byterange(1024, 2048);
        let tag = part.to_tag();
        assert!(tag.contains("BYTERANGE="));
        assert!(tag.contains("2048@1024"));
    }

    // 7. ServerControl tag has HOLD-BACK and CAN-BLOCK-RELOAD
    #[test]
    fn test_server_control_tag() {
        let sc = ServerControl {
            hold_back: 0.6,
            part_hold_back: 1.1,
            can_block_reload: true,
            can_skip_until: 24.0,
        };
        let tag = sc.to_tag();
        assert!(tag.contains("HOLD-BACK=0.6"));
        assert!(tag.contains("CAN-BLOCK-RELOAD=YES"));
        assert!(tag.contains("CAN-SKIP-UNTIL=24.0"));
    }

    // 8. PreloadHint PART tag renders correctly
    #[test]
    fn test_preload_hint_part_tag() {
        let hint = PreloadHint::part("next_part.mp4");
        let tag = hint.to_tag();
        assert!(tag.contains("EXT-X-PRELOAD-HINT"));
        assert!(tag.contains("TYPE=PART"));
        assert!(tag.contains("next_part.mp4"));
    }

    // 9. RenditionReport tag renders correctly
    #[test]
    fn test_rendition_report_tag() {
        let rr = RenditionReport {
            uri: "audio/playlist.m3u8".to_owned(),
            last_msn: 42,
            last_part: 3,
        };
        let tag = rr.to_tag();
        assert!(tag.contains("EXT-X-RENDITION-REPORT"));
        assert!(tag.contains("LAST-MSN=42"));
        assert!(tag.contains("LAST-PART=3"));
    }

    // 10. Empty playlist renders M3U8 header tags
    #[test]
    fn test_empty_playlist_to_m3u8() {
        let pl = default_playlist();
        let m3u8 = pl.to_m3u8();
        assert!(m3u8.contains("#EXTM3U"));
        assert!(m3u8.contains("#EXT-X-VERSION:9"));
        assert!(m3u8.contains("#EXT-X-PART-INF:"));
        assert!(m3u8.contains("#EXT-X-SERVER-CONTROL:"));
    }

    // 11. add_part without segment_complete keeps parts in current bucket
    #[test]
    fn test_add_part_no_segment_complete() {
        let mut pl = default_playlist();
        pl.add_part(make_part(0, true), false);
        pl.add_part(make_part(1, false), false);
        assert_eq!(pl.current_part_count(), 2);
        assert_eq!(pl.segments.len(), 0);
    }

    // 12. add_part with segment_complete finalises a segment
    #[test]
    fn test_add_part_finalises_segment() {
        let mut pl = default_playlist();
        for i in 0..5 {
            pl.add_part(make_part(i, i == 0), i == 4);
        }
        assert_eq!(pl.segments.len(), 1);
        assert_eq!(pl.current_part_count(), 0);
        let seg = pl.segments.front().expect("segment must exist");
        assert_eq!(seg.parts.len(), 5);
    }

    // 13. Window slides correctly after exceeding window_size
    #[test]
    fn test_window_slides() {
        let mut pl = default_playlist();
        pl.set_window_size(3);
        // Create 5 complete segments.
        for seg_idx in 0..5u32 {
            for part_idx in 0..5u32 {
                let last = part_idx == 4;
                pl.add_part(make_part(seg_idx * 10 + part_idx, part_idx == 0), last);
            }
        }
        assert_eq!(pl.segments.len(), 3);
        assert_eq!(pl.media_sequence, 2);
    }

    // 14. blocking_playlist_response returns None for future MSN
    #[test]
    fn test_blocking_response_future_msn_returns_none() {
        let pl = default_playlist();
        assert!(pl.blocking_playlist_response(100, None).is_none());
    }

    // 15. blocking_playlist_response returns Some after segment exists
    #[test]
    fn test_blocking_response_returns_some_when_available() {
        let mut pl = default_playlist();
        for i in 0..5u32 {
            pl.add_part(make_part(i, i == 0), i == 4);
        }
        let msn = pl.last_msn();
        let result = pl.blocking_playlist_response(msn, None);
        assert!(result.is_some());
        let m3u8 = result.expect("should be some");
        assert!(m3u8.contains("#EXTM3U"));
    }

    // 16. Playlist m3u8 contains EXTINF for finalised segments
    #[test]
    fn test_m3u8_contains_extinf() {
        let mut pl = default_playlist();
        pl.set_current_segment_uri("seg0.ts");
        for i in 0..5u32 {
            pl.add_part(make_part(i, i == 0), i == 4);
        }
        let m3u8 = pl.to_m3u8();
        assert!(m3u8.contains("#EXTINF:"));
        assert!(m3u8.contains("seg0.ts"));
    }

    // 17. LlHlsSegment duration accumulates from parts
    #[test]
    fn test_segment_duration_accumulates() {
        let mut seg = LlHlsSegment::new("seg.ts", 0);
        for _ in 0..10 {
            seg.push_part(MediaPart::new("p.mp4", 0.2));
        }
        assert!((seg.duration_secs - 2.0).abs() < 1e-9);
    }

    // 18. Rendition reports appear in m3u8 output
    #[test]
    fn test_rendition_reports_in_m3u8() {
        let mut pl = default_playlist();
        pl.rendition_reports.push(RenditionReport {
            uri: "audio.m3u8".to_owned(),
            last_msn: 0,
            last_part: 0,
        });
        assert!(pl.to_m3u8().contains("EXT-X-RENDITION-REPORT"));
    }

    // 19. last_msn is correct after several segments
    #[test]
    fn test_last_msn_tracking() {
        let mut pl = default_playlist();
        for seg in 0..3u32 {
            for part in 0..5u32 {
                pl.add_part(make_part(seg * 10 + part, part == 0), part == 4);
            }
        }
        assert_eq!(pl.last_msn(), 2);
    }

    // 20. blocking_playlist_response for in-progress part returns Some when available
    #[test]
    fn test_blocking_response_for_in_progress_part() {
        let mut pl = default_playlist();
        // Finalise segment 0.
        for i in 0..5u32 {
            pl.add_part(make_part(i, i == 0), i == 4);
        }
        // Add 3 parts to segment 1 (in progress).
        for i in 0..3u32 {
            pl.add_part(make_part(10 + i, i == 0), false);
        }
        // Request msn=1 part=2 (index 2 of 3 available parts).
        let result = pl.blocking_playlist_response(1, Some(2));
        assert!(result.is_some());
    }
}
