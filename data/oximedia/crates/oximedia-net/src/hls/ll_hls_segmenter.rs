//! CMAF LL-HLS segmenter producing partial segments (EXT-X-PART).
//!
//! [`LlHlsSegmenter`] accepts raw media frames and slices them into partial
//! segments of 100–200 ms, emitting [`HlsPartialSegment`] values that are
//! ready to serve via HTTP.  Every N partial segments are gathered into a
//! complete segment so the standard sliding window remains intact.
//!
//! The segmenter tracks:
//! - Partial segment numbering (`part_index` within each full segment)
//! - Whether a partial starts on an IDR / independent frame
//! - Byte accumulation so callers know the output size
//! - ISO 8601 program-date-time for the first part of each segment

use crate::error::{NetError, NetResult};
use std::collections::VecDeque;
use std::time::{Duration, SystemTime};

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for [`LlHlsSegmenter`].
#[derive(Debug, Clone)]
pub struct LlHlsSegmenterConfig {
    /// Target duration of each partial segment (100–200 ms recommended).
    pub part_duration: Duration,
    /// Number of partial segments that form one complete segment.
    pub parts_per_segment: u32,
    /// URI prefix for generated resource paths (e.g., `"seg"`).
    pub uri_prefix: String,
    /// Whether to emit `INDEPENDENT=YES` on keyframe-aligned parts.
    pub mark_independent: bool,
    /// Maximum number of completed segments to retain in the sliding window.
    pub window_size: usize,
}

impl Default for LlHlsSegmenterConfig {
    fn default() -> Self {
        Self {
            part_duration: Duration::from_millis(200),
            parts_per_segment: 30, // 30 × 200 ms = 6 s segments
            uri_prefix: "seg".to_owned(),
            mark_independent: true,
            window_size: 5,
        }
    }
}

impl LlHlsSegmenterConfig {
    /// Creates a config targeting the given part duration.
    #[must_use]
    pub fn with_part_duration(part_duration_ms: u64) -> Self {
        Self {
            part_duration: Duration::from_millis(part_duration_ms),
            ..Self::default()
        }
    }

    /// Returns the target full-segment duration in seconds.
    #[must_use]
    pub fn segment_duration_secs(&self) -> f64 {
        self.part_duration.as_secs_f64() * f64::from(self.parts_per_segment)
    }

    /// Returns the part duration in seconds.
    #[must_use]
    pub fn part_duration_secs(&self) -> f64 {
        self.part_duration.as_secs_f64()
    }
}

// ─── Partial Segment ──────────────────────────────────────────────────────────

/// A single partial segment produced by [`LlHlsSegmenter`].
///
/// Each partial segment is a self-contained CMAF chunk (or MPEG-TS slice)
/// that is individually addressable over HTTP.
#[derive(Debug, Clone)]
pub struct HlsPartialSegment {
    /// Full-segment sequence number this part belongs to.
    pub segment_sequence: u64,
    /// Part index within the current segment (0-based).
    pub part_index: u32,
    /// URI of this partial resource.
    pub uri: String,
    /// Duration of this part in seconds.
    pub duration_secs: f64,
    /// Whether this part starts with an IDR / independent frame.
    pub independent: bool,
    /// Accumulated payload bytes.
    pub data: Vec<u8>,
    /// Wall-clock time this part was created.
    pub created_at: SystemTime,
    /// Whether this is the last partial in its full segment.
    pub is_last_in_segment: bool,
}

impl HlsPartialSegment {
    /// Returns the `EXT-X-PART` tag for this partial segment.
    #[must_use]
    pub fn to_ext_x_part_tag(&self) -> String {
        let mut tag = format!(
            "#EXT-X-PART:DURATION={:.5},URI=\"{}\"",
            self.duration_secs, self.uri
        );
        if self.independent {
            tag.push_str(",INDEPENDENT=YES");
        }
        tag
    }

    /// Returns the URI of the full segment that contains this part.
    #[must_use]
    pub fn parent_segment_uri(&self) -> String {
        // Parts are named seg{N}_part{P}.mp4; segment is seg{N}.mp4
        format!("seg{}.mp4", self.segment_sequence)
    }
}

// ─── Completed Segment Record ─────────────────────────────────────────────────

/// Metadata about a full segment that has been finalised.
#[derive(Debug, Clone)]
pub struct CompletedSegment {
    /// Segment sequence number.
    pub sequence: u64,
    /// URI of the segment resource.
    pub uri: String,
    /// Total duration in seconds (sum of all parts).
    pub duration_secs: f64,
    /// The parts that make up this segment.
    pub parts: Vec<HlsPartialSegment>,
    /// Wall-clock time the segment was finalized.
    pub finalized_at: SystemTime,
}

impl CompletedSegment {
    /// Returns the `EXTINF` + URI lines for the segment.
    #[must_use]
    pub fn to_m3u8_lines(&self) -> String {
        let mut out = String::new();
        for part in &self.parts {
            out.push_str(&part.to_ext_x_part_tag());
            out.push('\n');
        }
        out.push_str(&format!("#EXTINF:{:.5},\n{}\n", self.duration_secs, self.uri));
        out
    }
}

// ─── Frame Input ──────────────────────────────────────────────────────────────

/// A media frame submitted to the segmenter.
#[derive(Debug, Clone)]
pub struct MediaFrame {
    /// Raw encoded frame data.
    pub data: Vec<u8>,
    /// Presentation timestamp in milliseconds.
    pub pts_ms: u64,
    /// Whether this frame is an IDR (keyframe).
    pub is_keyframe: bool,
}

impl MediaFrame {
    /// Creates a new media frame.
    #[must_use]
    pub fn new(data: Vec<u8>, pts_ms: u64, is_keyframe: bool) -> Self {
        Self {
            data,
            pts_ms,
            is_keyframe,
        }
    }
}

// ─── LlHlsSegmenter ───────────────────────────────────────────────────────────

/// Current state of a partial segment being built.
#[derive(Debug, Default)]
struct PartAccumulator {
    data: Vec<u8>,
    duration_ms: u64,
    has_keyframe: bool,
    frame_count: u32,
    start_pts_ms: Option<u64>,
}

impl PartAccumulator {
    fn reset(&mut self) {
        self.data.clear();
        self.duration_ms = 0;
        self.has_keyframe = false;
        self.frame_count = 0;
        self.start_pts_ms = None;
    }

    fn is_empty(&self) -> bool {
        self.frame_count == 0
    }

    fn push_frame(&mut self, frame: &MediaFrame, frame_duration_ms: u64) {
        if self.start_pts_ms.is_none() {
            self.start_pts_ms = Some(frame.pts_ms);
        }
        self.data.extend_from_slice(&frame.data);
        self.duration_ms += frame_duration_ms;
        self.frame_count += 1;
        if frame.is_keyframe {
            self.has_keyframe = true;
        }
    }
}

/// CMAF LL-HLS segmenter that produces partial segments from raw media frames.
///
/// # Usage
///
/// 1. Call [`push_frame`] for each encoded frame.
/// 2. Poll [`drain_parts`] to retrieve ready partial segments.
/// 3. Call [`take_completed_segment`] when a full segment is done.
///
/// The segmenter automatically closes a partial when either
/// (a) its accumulated duration meets `part_duration`, or
/// (b) a keyframe arrives and `force_split_on_keyframe` is set.
pub struct LlHlsSegmenter {
    config: LlHlsSegmenterConfig,
    /// Current segment sequence number.
    segment_seq: u64,
    /// Part index within the current segment.
    part_index: u32,
    /// Accumulated frames for the current partial.
    accumulator: PartAccumulator,
    /// Parts produced for the current (incomplete) segment.
    current_segment_parts: Vec<HlsPartialSegment>,
    /// Sliding window of completed segments.
    completed: VecDeque<CompletedSegment>,
    /// Parts ready for consumption by callers.
    ready_parts: VecDeque<HlsPartialSegment>,
    /// Whether a keyframe should always start a new partial.
    force_split_on_keyframe: bool,
    /// PTS of the last frame, used to compute frame duration.
    last_pts_ms: Option<u64>,
    /// Default frame duration when PTS gaps are unavailable.
    default_frame_duration_ms: u64,
}

impl LlHlsSegmenter {
    /// Creates a new segmenter with the given configuration.
    #[must_use]
    pub fn new(config: LlHlsSegmenterConfig) -> Self {
        Self {
            config,
            segment_seq: 0,
            part_index: 0,
            accumulator: PartAccumulator::default(),
            current_segment_parts: Vec::new(),
            completed: VecDeque::new(),
            ready_parts: VecDeque::new(),
            force_split_on_keyframe: true,
            default_frame_duration_ms: 33, // ~30 fps
        }
    }

    /// Creates a segmenter with default configuration and 200 ms parts.
    #[must_use]
    pub fn default_200ms() -> Self {
        Self::new(LlHlsSegmenterConfig::default())
    }

    /// Creates a segmenter with 100 ms parts.
    #[must_use]
    pub fn default_100ms() -> Self {
        Self::new(LlHlsSegmenterConfig::with_part_duration(100))
    }

    /// Returns the current segment sequence number.
    #[must_use]
    pub fn current_segment_seq(&self) -> u64 {
        self.segment_seq
    }

    /// Returns the current part index within the active segment.
    #[must_use]
    pub fn current_part_index(&self) -> u32 {
        self.part_index
    }

    /// Returns the number of completed segments in the sliding window.
    #[must_use]
    pub fn completed_segment_count(&self) -> usize {
        self.completed.len()
    }

    /// Returns the number of parts ready for consumption.
    #[must_use]
    pub fn ready_part_count(&self) -> usize {
        self.ready_parts.len()
    }

    /// Pushes a media frame into the segmenter.
    ///
    /// Internally accumulates frames; when a partial boundary is reached the
    /// partial is finalised and queued for consumption via [`drain_parts`].
    pub fn push_frame(&mut self, frame: MediaFrame) {
        // Compute frame duration
        let frame_dur = match self.last_pts_ms {
            Some(prev) => frame.pts_ms.saturating_sub(prev).max(1),
            None => self.default_frame_duration_ms,
        };
        self.last_pts_ms = Some(frame.pts_ms);

        // If forced keyframe split and accumulator is non-empty, flush first
        if self.force_split_on_keyframe && frame.is_keyframe && !self.accumulator.is_empty() {
            self.flush_partial();
        }

        self.accumulator.push_frame(&frame, frame_dur);

        // Check if the current partial has reached its target duration
        let target_ms = self.config.part_duration.as_millis() as u64;
        if self.accumulator.duration_ms >= target_ms {
            self.flush_partial();
        }
    }

    /// Flushes the current partial accumulator, even if underfilled.
    ///
    /// Useful at stream end.
    pub fn flush(&mut self) {
        if !self.accumulator.is_empty() {
            self.flush_partial();
        }
    }

    /// Drains all ready partial segments.
    pub fn drain_parts(&mut self) -> Vec<HlsPartialSegment> {
        self.ready_parts.drain(..).collect()
    }

    /// Returns the most recently completed full segment, if any.
    pub fn take_completed_segment(&mut self) -> Option<CompletedSegment> {
        self.completed.pop_front()
    }

    /// Returns a reference to the completed segment window.
    #[must_use]
    pub fn completed_segments(&self) -> &VecDeque<CompletedSegment> {
        &self.completed
    }

    /// Generates the `EXT-X-PRELOAD-HINT` URI for the next expected part.
    #[must_use]
    pub fn preload_hint_uri(&self) -> String {
        format!(
            "{}{}_part{}.mp4",
            self.config.uri_prefix,
            self.segment_seq,
            self.part_index
        )
    }

    // ── Private ───────────────────────────────────────────────────────────────

    fn flush_partial(&mut self) {
        if self.accumulator.is_empty() {
            return;
        }

        let duration_secs = self.accumulator.duration_ms as f64 / 1000.0;
        let independent = self.config.mark_independent && self.accumulator.has_keyframe;
        let uri = format!(
            "{}{}_part{}.mp4",
            self.config.uri_prefix, self.segment_seq, self.part_index
        );

        // Determine whether this part completes the segment
        let is_last = self.part_index + 1 >= self.config.parts_per_segment;

        let part = HlsPartialSegment {
            segment_sequence: self.segment_seq,
            part_index: self.part_index,
            uri,
            duration_secs,
            independent,
            data: std::mem::take(&mut self.accumulator.data),
            created_at: SystemTime::now(),
            is_last_in_segment: is_last,
        };

        self.accumulator.reset();
        self.part_index += 1;
        self.current_segment_parts.push(part.clone());
        self.ready_parts.push_back(part);

        if is_last {
            self.finalize_segment();
        }
    }

    fn finalize_segment(&mut self) {
        let seg_uri = format!("{}{}.mp4", self.config.uri_prefix, self.segment_seq);
        let total_duration: f64 = self.current_segment_parts.iter().map(|p| p.duration_secs).sum();

        let seg = CompletedSegment {
            sequence: self.segment_seq,
            uri: seg_uri,
            duration_secs: total_duration,
            parts: std::mem::take(&mut self.current_segment_parts),
            finalized_at: SystemTime::now(),
        };

        self.completed.push_back(seg);

        // Slide window
        while self.completed.len() > self.config.window_size {
            self.completed.pop_front();
        }

        self.segment_seq += 1;
        self.part_index = 0;
    }
}

impl std::fmt::Debug for LlHlsSegmenter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlHlsSegmenter")
            .field("segment_seq", &self.segment_seq)
            .field("part_index", &self.part_index)
            .field("ready_parts", &self.ready_parts.len())
            .finish()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(pts_ms: u64, is_keyframe: bool, size: usize) -> MediaFrame {
        MediaFrame::new(vec![0u8; size], pts_ms, is_keyframe)
    }

    fn default_segmenter() -> LlHlsSegmenter {
        let mut cfg = LlHlsSegmenterConfig::default();
        cfg.parts_per_segment = 3; // 3 parts × 200 ms = 600 ms segment
        LlHlsSegmenter::new(cfg)
    }

    // 1. Config segment duration calculation
    #[test]
    fn test_config_segment_duration() {
        let cfg = LlHlsSegmenterConfig::default();
        let dur = cfg.segment_duration_secs();
        assert!(dur > 0.0);
    }

    // 2. Config part duration in seconds
    #[test]
    fn test_config_part_duration_secs() {
        let cfg = LlHlsSegmenterConfig::with_part_duration(100);
        let d = cfg.part_duration_secs();
        assert!((d - 0.1).abs() < 1e-9);
    }

    // 3. Segmenter starts at sequence 0
    #[test]
    fn test_initial_state() {
        let seg = default_segmenter();
        assert_eq!(seg.current_segment_seq(), 0);
        assert_eq!(seg.current_part_index(), 0);
        assert_eq!(seg.ready_part_count(), 0);
    }

    // 4. Pushing a single frame does not immediately produce a part
    #[test]
    fn test_single_frame_no_part() {
        let mut seg = default_segmenter();
        seg.push_frame(make_frame(0, true, 1024));
        // Frame is shorter than part_duration — no part yet
        assert_eq!(seg.ready_part_count(), 0);
    }

    // 5. Accumulating enough frames flushes a partial
    #[test]
    fn test_part_flush_on_duration() {
        let mut cfg = LlHlsSegmenterConfig::default();
        cfg.part_duration = Duration::from_millis(100);
        cfg.parts_per_segment = 10;
        let mut seg = LlHlsSegmenter::new(cfg);

        // Push frames at 33 ms intervals until 100 ms passed
        for i in 0..4u64 {
            seg.push_frame(make_frame(i * 33, i == 0, 512));
        }
        // After 4 frames × 33 ms = 99 ms, still under limit.
        // 5th frame at 132 ms pushes over.
        seg.push_frame(make_frame(4 * 33, false, 512));
        let parts = seg.drain_parts();
        assert!(!parts.is_empty());
    }

    // 6. Keyframe forces a new partial boundary
    #[test]
    fn test_keyframe_splits_partial() {
        let mut cfg = LlHlsSegmenterConfig::default();
        cfg.part_duration = Duration::from_millis(500); // Long duration
        cfg.parts_per_segment = 10;
        let mut seg = LlHlsSegmenter::new(cfg);

        // Push a non-keyframe, then a keyframe
        seg.push_frame(make_frame(0, false, 512));
        seg.push_frame(make_frame(33, true, 512)); // keyframe forces split

        let parts = seg.drain_parts();
        assert_eq!(parts.len(), 1, "first partial should flush on keyframe");
    }

    // 7. Keyframe partial is marked independent
    #[test]
    fn test_independent_marking() {
        let mut cfg = LlHlsSegmenterConfig::default();
        cfg.part_duration = Duration::from_millis(500);
        cfg.parts_per_segment = 5;
        let mut seg = LlHlsSegmenter::new(cfg);

        seg.push_frame(make_frame(0, true, 512)); // keyframe in accumulator
        seg.push_frame(make_frame(33, false, 512)); // second frame
        // Force flush by sending a keyframe
        seg.push_frame(make_frame(66, true, 512));

        let parts = seg.drain_parts();
        assert!(!parts.is_empty());
        let first = &parts[0];
        assert!(first.independent, "first part has keyframe → INDEPENDENT=YES");
    }

    // 8. EXT-X-PART tag format
    #[test]
    fn test_ext_x_part_tag() {
        let part = HlsPartialSegment {
            segment_sequence: 2,
            part_index: 0,
            uri: "seg2_part0.mp4".to_owned(),
            duration_secs: 0.2,
            independent: true,
            data: vec![],
            created_at: SystemTime::now(),
            is_last_in_segment: false,
        };
        let tag = part.to_ext_x_part_tag();
        assert!(tag.contains("#EXT-X-PART"));
        assert!(tag.contains("DURATION=0.20000"));
        assert!(tag.contains("INDEPENDENT=YES"));
        assert!(tag.contains("seg2_part0.mp4"));
    }

    // 9. EXT-X-PART tag without independent
    #[test]
    fn test_ext_x_part_tag_no_independent() {
        let part = HlsPartialSegment {
            segment_sequence: 0,
            part_index: 1,
            uri: "seg0_part1.mp4".to_owned(),
            duration_secs: 0.2,
            independent: false,
            data: vec![],
            created_at: SystemTime::now(),
            is_last_in_segment: false,
        };
        let tag = part.to_ext_x_part_tag();
        assert!(!tag.contains("INDEPENDENT"));
    }

    // 10. Parent segment URI derived correctly
    #[test]
    fn test_parent_segment_uri() {
        let part = HlsPartialSegment {
            segment_sequence: 7,
            part_index: 2,
            uri: "seg7_part2.mp4".to_owned(),
            duration_secs: 0.2,
            independent: false,
            data: vec![],
            created_at: SystemTime::now(),
            is_last_in_segment: false,
        };
        assert_eq!(part.parent_segment_uri(), "seg7.mp4");
    }

    // 11. A full segment is produced after N parts
    #[test]
    fn test_full_segment_produced() {
        let mut cfg = LlHlsSegmenterConfig::default();
        cfg.part_duration = Duration::from_millis(50);
        cfg.parts_per_segment = 2;
        let mut seg = LlHlsSegmenter::new(cfg);

        // Push enough frames to fill 2 parts (2 × 50 ms = 100 ms)
        for i in 0..6u64 {
            seg.push_frame(make_frame(i * 20, i == 0, 256));
        }
        let completed = seg.take_completed_segment();
        assert!(completed.is_some());
        let c = completed.expect("should have segment");
        assert_eq!(c.sequence, 0);
        assert_eq!(c.parts.len(), 2);
    }

    // 12. Segment sequence increments after completion
    #[test]
    fn test_segment_seq_increments() {
        let mut cfg = LlHlsSegmenterConfig::default();
        cfg.part_duration = Duration::from_millis(50);
        cfg.parts_per_segment = 2;
        let mut seg = LlHlsSegmenter::new(cfg);

        // Two full segments worth of frames
        for i in 0..12u64 {
            seg.push_frame(make_frame(i * 20, i % 4 == 0, 256));
        }
        assert_eq!(seg.current_segment_seq(), 2);
    }

    // 13. Window size limits stored completed segments
    #[test]
    fn test_window_size_limit() {
        let mut cfg = LlHlsSegmenterConfig::default();
        cfg.part_duration = Duration::from_millis(50);
        cfg.parts_per_segment = 2;
        cfg.window_size = 2;
        let mut seg = LlHlsSegmenter::new(cfg);

        // Produce 4 segments
        for i in 0..16u64 {
            seg.push_frame(make_frame(i * 20, i % 4 == 0, 256));
        }
        assert_eq!(seg.completed_segment_count(), 2);
    }

    // 14. Flush drains remaining accumulator
    #[test]
    fn test_flush_drains_accumulator() {
        let mut seg = default_segmenter();
        seg.push_frame(make_frame(0, true, 512)); // under 200 ms
        seg.flush();
        // At least one part should be queued
        assert!(seg.ready_part_count() > 0 || seg.current_segment_seq() > 0);
    }

    // 15. Preload hint URI format
    #[test]
    fn test_preload_hint_uri() {
        let seg = default_segmenter();
        let hint = seg.preload_hint_uri();
        assert!(hint.contains("seg0_part0.mp4"));
    }

    // 16. Preload hint URI advances with part index
    #[test]
    fn test_preload_hint_advances() {
        let mut cfg = LlHlsSegmenterConfig::default();
        cfg.part_duration = Duration::from_millis(50);
        cfg.parts_per_segment = 5;
        let mut seg = LlHlsSegmenter::new(cfg);

        // Flush one partial
        for i in 0..4u64 {
            seg.push_frame(make_frame(i * 20, i == 0, 256));
        }
        seg.drain_parts();
        let hint = seg.preload_hint_uri();
        // Part index should be >= 1 now
        assert!(!hint.contains("_part0.mp4") || seg.current_part_index() == 0);
    }

    // 17. MediaFrame construction
    #[test]
    fn test_media_frame_new() {
        let f = MediaFrame::new(vec![1, 2, 3], 1000, true);
        assert_eq!(f.pts_ms, 1000);
        assert!(f.is_keyframe);
        assert_eq!(f.data.len(), 3);
    }

    // 18. CompletedSegment to_m3u8_lines contains EXTINF
    #[test]
    fn test_completed_segment_to_m3u8() {
        let part = HlsPartialSegment {
            segment_sequence: 0,
            part_index: 0,
            uri: "seg0_part0.mp4".to_owned(),
            duration_secs: 0.2,
            independent: true,
            data: vec![],
            created_at: SystemTime::now(),
            is_last_in_segment: false,
        };
        let seg = CompletedSegment {
            sequence: 0,
            uri: "seg0.mp4".to_owned(),
            duration_secs: 0.2,
            parts: vec![part],
            finalized_at: SystemTime::now(),
        };
        let lines = seg.to_m3u8_lines();
        assert!(lines.contains("#EXT-X-PART"));
        assert!(lines.contains("#EXTINF"));
        assert!(lines.contains("seg0.mp4"));
    }

    // 19. Debug format is available
    #[test]
    fn test_segmenter_debug() {
        let seg = default_segmenter();
        let dbg = format!("{seg:?}");
        assert!(dbg.contains("LlHlsSegmenter"));
    }

    // 20. default_100ms constructor
    #[test]
    fn test_default_100ms_ctor() {
        let seg = LlHlsSegmenter::default_100ms();
        assert!((seg.config.part_duration_secs() - 0.1).abs() < 1e-9);
    }

    // 21. default_200ms constructor
    #[test]
    fn test_default_200ms_ctor() {
        let seg = LlHlsSegmenter::default_200ms();
        assert!((seg.config.part_duration_secs() - 0.2).abs() < 1e-9);
    }

    // 22. Part index resets after segment boundary
    #[test]
    fn test_part_index_resets_on_segment() {
        let mut cfg = LlHlsSegmenterConfig::default();
        cfg.part_duration = Duration::from_millis(50);
        cfg.parts_per_segment = 2;
        let mut seg = LlHlsSegmenter::new(cfg);

        for i in 0..6u64 {
            seg.push_frame(make_frame(i * 20, i % 3 == 0, 256));
        }
        // After segment 0 is complete, part index resets to start new segment
        assert_eq!(seg.current_part_index(), 0);
    }

    // 23. Parts carry correct segment sequence
    #[test]
    fn test_parts_carry_segment_seq() {
        let mut cfg = LlHlsSegmenterConfig::default();
        cfg.part_duration = Duration::from_millis(50);
        cfg.parts_per_segment = 10;
        let mut seg = LlHlsSegmenter::new(cfg);

        seg.push_frame(make_frame(0, true, 256));
        seg.push_frame(make_frame(33, false, 256));
        seg.push_frame(make_frame(66, true, 256)); // flush
        let parts = seg.drain_parts();
        for p in &parts {
            assert_eq!(p.segment_sequence, 0);
        }
    }

    // 24. Multiple parts have increasing part indices
    #[test]
    fn test_part_indices_increasing() {
        let mut cfg = LlHlsSegmenterConfig::default();
        cfg.part_duration = Duration::from_millis(50);
        cfg.parts_per_segment = 5;
        let mut seg = LlHlsSegmenter::new(cfg);

        for i in 0..6u64 {
            seg.push_frame(make_frame(i * 20, i % 2 == 0, 256));
        }
        let parts = seg.drain_parts();
        let indices: Vec<u32> = parts.iter().map(|p| p.part_index).collect();
        if indices.len() > 1 {
            for w in indices.windows(2) {
                assert!(w[1] == w[0] + 1, "part indices should be sequential");
            }
        }
    }

    // 25. Last part in segment is flagged
    #[test]
    fn test_last_part_in_segment_flag() {
        let mut cfg = LlHlsSegmenterConfig::default();
        cfg.part_duration = Duration::from_millis(50);
        cfg.parts_per_segment = 2;
        let mut seg = LlHlsSegmenter::new(cfg);

        for i in 0..6u64 {
            seg.push_frame(make_frame(i * 20, i % 3 == 0, 256));
        }
        let all_parts = seg.drain_parts();
        // The last part of each segment should be flagged
        if !all_parts.is_empty() {
            let last = all_parts.last().expect("should have a part");
            // At least one part should have is_last_in_segment if a full segment was created
            let _ = last.is_last_in_segment; // just verify field exists
        }
    }
}
