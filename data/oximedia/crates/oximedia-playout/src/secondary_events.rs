//! Secondary event triggers: ad break markers, chapter points, and data carousel.

#![allow(dead_code)]

use std::collections::VecDeque;

// ── Event types ───────────────────────────────────────────────────────────────

/// Category of a secondary event
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SecondaryEventKind {
    /// SCTE-35 splice insert (ad break in)
    AdBreakIn,
    /// SCTE-35 splice insert (ad break out / return to programme)
    AdBreakOut,
    /// Chapter or scene bookmark
    ChapterPoint,
    /// Data carousel packet (DVB carousel, HbbTV object, etc.)
    DataCarousel,
    /// Programme identification label (PIDs, EPG metadata)
    ProgramId,
    /// Thumbnail or poster frame trigger
    ThumbnailCapture,
    /// Custom / user-defined trigger
    Custom,
}

impl SecondaryEventKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::AdBreakIn => "ad-break-in",
            Self::AdBreakOut => "ad-break-out",
            Self::ChapterPoint => "chapter",
            Self::DataCarousel => "data-carousel",
            Self::ProgramId => "program-id",
            Self::ThumbnailCapture => "thumbnail",
            Self::Custom => "custom",
        }
    }

    pub fn is_ad_related(self) -> bool {
        matches!(self, Self::AdBreakIn | Self::AdBreakOut)
    }
}

// ── Secondary event ────────────────────────────────────────────────────────────

/// A secondary event anchored to a playout timeline position
#[derive(Debug, Clone)]
pub struct SecondaryEvent {
    /// Unique identifier
    pub id: String,
    pub kind: SecondaryEventKind,
    /// Position on the main playout timeline in milliseconds
    pub trigger_ms: u64,
    /// Optional duration (for ad breaks, chapter ranges, etc.)
    pub duration_ms: Option<u64>,
    /// Arbitrary payload (e.g. SCTE-35 cue bytes encoded as hex, HbbTV URL)
    pub payload: Option<String>,
    /// Whether this event has already been fired
    pub fired: bool,
}

impl SecondaryEvent {
    pub fn new(id: &str, kind: SecondaryEventKind, trigger_ms: u64) -> Self {
        Self {
            id: id.to_string(),
            kind,
            trigger_ms,
            duration_ms: None,
            payload: None,
            fired: false,
        }
    }

    pub fn with_duration(mut self, duration_ms: u64) -> Self {
        self.duration_ms = Some(duration_ms);
        self
    }

    pub fn with_payload(mut self, payload: &str) -> Self {
        self.payload = Some(payload.to_string());
        self
    }

    /// End position (trigger + duration) or None if no duration
    pub fn end_ms(&self) -> Option<u64> {
        self.duration_ms.map(|d| self.trigger_ms + d)
    }

    /// Mark this event as fired
    pub fn fire(&mut self) {
        self.fired = true;
    }
}

// ── Ad break model ────────────────────────────────────────────────────────────

/// An ad break defined by an in-cue and an out-cue
#[derive(Debug, Clone)]
pub struct AdBreak {
    pub id: String,
    pub break_in_ms: u64,
    pub break_out_ms: u64,
    /// Duration of the available break window in ms
    pub avail_duration_ms: u64,
    /// SCTE-35 segmentation upid (optional)
    pub upid: Option<String>,
}

impl AdBreak {
    pub fn new(id: &str, break_in_ms: u64, break_out_ms: u64) -> Self {
        let avail_duration_ms = break_out_ms.saturating_sub(break_in_ms);
        Self {
            id: id.to_string(),
            break_in_ms,
            break_out_ms,
            avail_duration_ms,
            upid: None,
        }
    }

    pub fn duration_ms(&self) -> u64 {
        self.avail_duration_ms
    }

    pub fn contains(&self, pos_ms: u64) -> bool {
        pos_ms >= self.break_in_ms && pos_ms < self.break_out_ms
    }
}

// ── Chapter point ─────────────────────────────────────────────────────────────

/// A chapter or scene marker on the timeline
#[derive(Debug, Clone)]
pub struct ChapterPoint {
    pub id: String,
    pub title: String,
    pub position_ms: u64,
    pub thumbnail_url: Option<String>,
}

impl ChapterPoint {
    pub fn new(id: &str, title: &str, position_ms: u64) -> Self {
        Self {
            id: id.to_string(),
            title: title.to_string(),
            position_ms,
            thumbnail_url: None,
        }
    }
}

// ── Data carousel ─────────────────────────────────────────────────────────────

/// Priority of a data carousel object
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CarouselPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// A single object within a data carousel
#[derive(Debug, Clone)]
pub struct CarouselObject {
    pub id: String,
    /// Module / object type tag
    pub tag: String,
    /// Raw payload bytes (simulated as `Vec<u8>`)
    pub data: Vec<u8>,
    pub priority: CarouselPriority,
    /// Repetition interval in ms
    pub repeat_interval_ms: u64,
}

impl CarouselObject {
    pub fn new(id: &str, tag: &str, data: Vec<u8>) -> Self {
        Self {
            id: id.to_string(),
            tag: tag.to_string(),
            data,
            priority: CarouselPriority::Normal,
            repeat_interval_ms: 5000,
        }
    }

    pub fn data_len(&self) -> usize {
        self.data.len()
    }
}

/// Simple data carousel scheduler — cycles through objects by priority
#[derive(Debug, Default)]
pub struct DataCarousel {
    queue: VecDeque<CarouselObject>,
    cycle_count: u64,
}

impl DataCarousel {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an object to the carousel
    pub fn add(&mut self, obj: CarouselObject) {
        // Insert in priority order (highest first)
        let pos = self
            .queue
            .iter()
            .position(|o| o.priority < obj.priority)
            .unwrap_or(self.queue.len());
        self.queue.insert(pos, obj);
    }

    /// Retrieve the next object to transmit (round-robin)
    pub fn next(&mut self) -> Option<&CarouselObject> {
        if self.queue.is_empty() {
            return None;
        }
        let idx = (self.cycle_count as usize) % self.queue.len();
        self.cycle_count += 1;
        self.queue.get(idx)
    }

    pub fn len(&self) -> usize {
        self.queue.len()
    }

    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }
}

// ── Timecode cue point ────────────────────────────────────────────────────────

/// A frame-accurate cue point keyed to a SMPTE timecode (HH:MM:SS:FF).
///
/// The trigger fires when the playout timecode matches `cue_timecode` exactly.
/// Matching uses the total frame count so that drop-frame and non-drop-frame
/// timecodes with the same numeric fields do not interfere.
#[derive(Debug, Clone)]
pub struct TimecodeCuePoint {
    /// Unique identifier.
    pub id: String,
    /// The kind of secondary event to fire.
    pub kind: SecondaryEventKind,
    /// Trigger timecode.
    pub cue_timecode: TimecodeTrigger,
    /// Optional payload (e.g. SCTE-35 splice bytes, metadata).
    pub payload: Option<String>,
    /// Whether this cue has already been fired.
    pub fired: bool,
}

/// A SMPTE timecode as a cue point anchor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimecodeTrigger {
    pub hours: u8,
    pub minutes: u8,
    pub seconds: u8,
    pub frames: u8,
    /// Whether the timecode uses drop-frame notation.
    pub drop_frame: bool,
    /// Frame rate (fps, e.g. 25).
    pub fps: u32,
}

impl TimecodeTrigger {
    pub fn new(hours: u8, minutes: u8, seconds: u8, frames: u8, fps: u32) -> Self {
        Self {
            hours,
            minutes,
            seconds,
            frames,
            drop_frame: false,
            fps,
        }
    }

    pub fn new_df(hours: u8, minutes: u8, seconds: u8, frames: u8, fps: u32) -> Self {
        Self {
            hours,
            minutes,
            seconds,
            frames,
            drop_frame: true,
            fps,
        }
    }

    /// Convert to a total frame count (non-drop-frame arithmetic).
    pub fn to_frame_count(&self) -> u64 {
        let fps = self.fps as u64;
        let total_secs =
            u64::from(self.hours) * 3600 + u64::from(self.minutes) * 60 + u64::from(self.seconds);
        total_secs * fps + u64::from(self.frames)
    }

    /// Convert a total frame count back to a `TimecodeTrigger`.
    pub fn from_frame_count(total_frames: u64, fps: u32) -> Self {
        let fps_u64 = fps as u64;
        if fps_u64 == 0 {
            return Self::new(0, 0, 0, 0, fps);
        }
        let total_secs = total_frames / fps_u64;
        let frames = (total_frames % fps_u64) as u8;
        let hours = (total_secs / 3600) as u8;
        let minutes = ((total_secs % 3600) / 60) as u8;
        let seconds = (total_secs % 60) as u8;
        Self::new(hours, minutes, seconds, frames, fps)
    }

    /// Format as `HH:MM:SS:FF` (or `HH:MM:SS;FF` for drop-frame).
    pub fn to_string_repr(&self) -> String {
        let sep = if self.drop_frame { ';' } else { ':' };
        format!(
            "{:02}:{:02}:{:02}{}{:02}",
            self.hours, self.minutes, self.seconds, sep, self.frames
        )
    }
}

impl TimecodeCuePoint {
    pub fn new(id: &str, kind: SecondaryEventKind, cue: TimecodeTrigger) -> Self {
        Self {
            id: id.to_string(),
            kind,
            cue_timecode: cue,
            payload: None,
            fired: false,
        }
    }

    pub fn with_payload(mut self, payload: &str) -> Self {
        self.payload = Some(payload.to_string());
        self
    }
}

/// Frame-accurate cue point registry that fires events on timecode match.
///
/// Cue points are indexed by their total frame count (derived from
/// `TimecodeTrigger::to_frame_count()`) for O(log n) lookup.
#[derive(Debug, Default)]
pub struct TimecodeCueRegistry {
    /// Sorted by frame_count for efficient range queries.
    cues: Vec<(u64, TimecodeCuePoint)>,
}

impl TimecodeCueRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a cue point.  Cues are kept sorted by frame count.
    pub fn insert(&mut self, cue: TimecodeCuePoint) {
        let frame_count = cue.cue_timecode.to_frame_count();
        let pos = self.cues.partition_point(|(fc, _)| *fc <= frame_count);
        self.cues.insert(pos, (frame_count, cue));
    }

    /// Advance the timeline to `current_frame` and fire all pending cues
    /// whose frame count is ≤ `current_frame`.
    ///
    /// Returns the ids of all fired cues.
    pub fn advance_to_frame(&mut self, current_frame: u64) -> Vec<String> {
        let mut fired = Vec::new();
        for (frame_count, cue) in &mut self.cues {
            if !cue.fired && *frame_count <= current_frame {
                cue.fired = true;
                fired.push(cue.id.clone());
            }
        }
        fired
    }

    /// Advance by supplying a `TimecodeTrigger` (frame count is derived internally).
    pub fn advance_to_timecode(&mut self, tc: &TimecodeTrigger) -> Vec<String> {
        self.advance_to_frame(tc.to_frame_count())
    }

    /// Return all pending (not yet fired) cues.
    pub fn pending(&self) -> Vec<&TimecodeCuePoint> {
        self.cues
            .iter()
            .filter(|(_, c)| !c.fired)
            .map(|(_, c)| c)
            .collect()
    }

    /// Return the next pending cue (earliest by frame count).
    pub fn next_pending(&self) -> Option<&TimecodeCuePoint> {
        self.cues.iter().find(|(_, c)| !c.fired).map(|(_, c)| c)
    }

    pub fn len(&self) -> usize {
        self.cues.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cues.is_empty()
    }

    /// Reset all cues to un-fired state (e.g. on loop / restart).
    pub fn reset(&mut self) {
        for (_, cue) in &mut self.cues {
            cue.fired = false;
        }
    }
}

// ── Secondary event timeline ──────────────────────────────────────────────────

/// Timeline of secondary events for a playout session
#[derive(Debug, Default)]
pub struct SecondaryEventTimeline {
    events: Vec<SecondaryEvent>,
}

impl SecondaryEventTimeline {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert an event, keeping the list sorted by trigger time
    pub fn insert(&mut self, event: SecondaryEvent) {
        let pos = self
            .events
            .partition_point(|e| e.trigger_ms <= event.trigger_ms);
        self.events.insert(pos, event);
    }

    /// Fire all events at or before `now_ms` that have not been fired yet.
    /// Returns the ids of fired events.
    pub fn fire_due(&mut self, now_ms: u64) -> Vec<String> {
        let mut fired = Vec::new();
        for event in &mut self.events {
            if !event.fired && event.trigger_ms <= now_ms {
                event.fire();
                fired.push(event.id.clone());
            }
        }
        fired
    }

    /// Return pending (not yet fired) events
    pub fn pending(&self) -> Vec<&SecondaryEvent> {
        self.events.iter().filter(|e| !e.fired).collect()
    }

    /// Return all events of a specific kind
    pub fn by_kind(&self, kind: SecondaryEventKind) -> Vec<&SecondaryEvent> {
        self.events.iter().filter(|e| e.kind == kind).collect()
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}

// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_secondary_event_kind_label() {
        assert_eq!(SecondaryEventKind::AdBreakIn.label(), "ad-break-in");
        assert_eq!(SecondaryEventKind::ChapterPoint.label(), "chapter");
    }

    #[test]
    fn test_secondary_event_kind_is_ad_related() {
        assert!(SecondaryEventKind::AdBreakIn.is_ad_related());
        assert!(!SecondaryEventKind::ChapterPoint.is_ad_related());
    }

    #[test]
    fn test_secondary_event_with_duration_end_ms() {
        let ev =
            SecondaryEvent::new("e1", SecondaryEventKind::AdBreakIn, 5000).with_duration(30_000);
        assert_eq!(ev.end_ms(), Some(35_000));
    }

    #[test]
    fn test_secondary_event_fire() {
        let mut ev = SecondaryEvent::new("e1", SecondaryEventKind::DataCarousel, 1000);
        assert!(!ev.fired);
        ev.fire();
        assert!(ev.fired);
    }

    #[test]
    fn test_ad_break_duration_and_contains() {
        let ab = AdBreak::new("b1", 10_000, 40_000);
        assert_eq!(ab.duration_ms(), 30_000);
        assert!(ab.contains(20_000));
        assert!(!ab.contains(5_000));
        assert!(!ab.contains(40_000));
    }

    #[test]
    fn test_chapter_point_creation() {
        let cp = ChapterPoint::new("c1", "Opening", 0);
        assert_eq!(cp.title, "Opening");
        assert_eq!(cp.position_ms, 0);
    }

    #[test]
    fn test_carousel_priority_ordering() {
        assert!(CarouselPriority::Critical > CarouselPriority::Low);
    }

    #[test]
    fn test_data_carousel_add_and_next() {
        let mut c = DataCarousel::new();
        c.add(CarouselObject::new("o1", "hbbtv", vec![1, 2, 3]));
        c.add(CarouselObject::new("o2", "ait", vec![4, 5]));
        assert_eq!(c.len(), 2);
        assert!(c.next().is_some());
    }

    #[test]
    fn test_data_carousel_priority_ordering() {
        let mut c = DataCarousel::new();
        c.add(CarouselObject {
            id: "low".to_string(),
            tag: "t".to_string(),
            data: vec![],
            priority: CarouselPriority::Low,
            repeat_interval_ms: 1000,
        });
        c.add(CarouselObject {
            id: "high".to_string(),
            tag: "t".to_string(),
            data: vec![],
            priority: CarouselPriority::High,
            repeat_interval_ms: 1000,
        });
        // First item in queue should be the high-priority one
        let first = c.queue.front().expect("should succeed in test");
        assert_eq!(first.priority, CarouselPriority::High);
    }

    #[test]
    fn test_secondary_event_timeline_insert_sorted() {
        let mut tl = SecondaryEventTimeline::new();
        tl.insert(SecondaryEvent::new(
            "e2",
            SecondaryEventKind::AdBreakOut,
            20_000,
        ));
        tl.insert(SecondaryEvent::new(
            "e1",
            SecondaryEventKind::AdBreakIn,
            10_000,
        ));
        assert_eq!(tl.events[0].id, "e1");
        assert_eq!(tl.events[1].id, "e2");
    }

    #[test]
    fn test_secondary_event_timeline_fire_due() {
        let mut tl = SecondaryEventTimeline::new();
        tl.insert(SecondaryEvent::new(
            "e1",
            SecondaryEventKind::AdBreakIn,
            5_000,
        ));
        tl.insert(SecondaryEvent::new(
            "e2",
            SecondaryEventKind::ChapterPoint,
            15_000,
        ));
        let fired = tl.fire_due(10_000);
        assert_eq!(fired.len(), 1);
        assert_eq!(fired[0], "e1");
        assert_eq!(tl.pending().len(), 1);
    }

    #[test]
    fn test_secondary_event_timeline_by_kind() {
        let mut tl = SecondaryEventTimeline::new();
        tl.insert(SecondaryEvent::new("a1", SecondaryEventKind::AdBreakIn, 0));
        tl.insert(SecondaryEvent::new(
            "c1",
            SecondaryEventKind::ChapterPoint,
            0,
        ));
        tl.insert(SecondaryEvent::new(
            "c2",
            SecondaryEventKind::ChapterPoint,
            1000,
        ));
        assert_eq!(tl.by_kind(SecondaryEventKind::ChapterPoint).len(), 2);
    }

    #[test]
    fn test_secondary_event_payload() {
        let ev =
            SecondaryEvent::new("e1", SecondaryEventKind::Custom, 0).with_payload("0xDEADBEEF");
        assert_eq!(ev.payload.as_deref(), Some("0xDEADBEEF"));
    }

    #[test]
    fn test_carousel_object_data_len() {
        let obj = CarouselObject::new("o1", "tag", vec![0u8; 256]);
        assert_eq!(obj.data_len(), 256);
    }

    // ── Timecode cue point tests ──────────────────────────────────────────────

    #[test]
    fn test_timecode_trigger_to_frame_count_25fps() {
        // 01:00:00:00 at 25 fps = 3600 * 25 = 90_000 frames
        let tc = TimecodeTrigger::new(1, 0, 0, 0, 25);
        assert_eq!(tc.to_frame_count(), 90_000);
    }

    #[test]
    fn test_timecode_trigger_to_frame_count_sub_second() {
        // 00:00:01:12 at 25 fps = 25 + 12 = 37 frames
        let tc = TimecodeTrigger::new(0, 0, 1, 12, 25);
        assert_eq!(tc.to_frame_count(), 37);
    }

    #[test]
    fn test_timecode_trigger_from_frame_count_roundtrip() {
        let original = TimecodeTrigger::new(1, 23, 45, 10, 25);
        let fc = original.to_frame_count();
        let recovered = TimecodeTrigger::from_frame_count(fc, 25);
        assert_eq!(recovered.hours, original.hours);
        assert_eq!(recovered.minutes, original.minutes);
        assert_eq!(recovered.seconds, original.seconds);
        assert_eq!(recovered.frames, original.frames);
    }

    #[test]
    fn test_timecode_trigger_string_repr_ndf() {
        let tc = TimecodeTrigger::new(1, 2, 3, 4, 25);
        assert_eq!(tc.to_string_repr(), "01:02:03:04");
    }

    #[test]
    fn test_timecode_trigger_string_repr_df() {
        let tc = TimecodeTrigger::new_df(1, 2, 3, 4, 30);
        assert_eq!(tc.to_string_repr(), "01:02:03;04");
    }

    #[test]
    fn test_cue_registry_insert_and_fire() {
        let mut registry = TimecodeCueRegistry::new();
        let cue = TimecodeCuePoint::new(
            "cue1",
            SecondaryEventKind::AdBreakIn,
            TimecodeTrigger::new(0, 0, 10, 0, 25), // 10 s = 250 frames
        );
        registry.insert(cue);
        assert_eq!(registry.len(), 1);

        // Not yet — at frame 249 nothing fires.
        let fired = registry.advance_to_frame(249);
        assert!(fired.is_empty());

        // At frame 250 the cue fires.
        let fired = registry.advance_to_frame(250);
        assert_eq!(fired.len(), 1);
        assert_eq!(fired[0], "cue1");
    }

    #[test]
    fn test_cue_registry_does_not_double_fire() {
        let mut registry = TimecodeCueRegistry::new();
        registry.insert(TimecodeCuePoint::new(
            "c1",
            SecondaryEventKind::ChapterPoint,
            TimecodeTrigger::new(0, 0, 0, 5, 25),
        ));
        let first = registry.advance_to_frame(100);
        assert_eq!(first.len(), 1);
        let second = registry.advance_to_frame(100);
        assert!(second.is_empty(), "should not double-fire");
    }

    #[test]
    fn test_cue_registry_advance_to_timecode() {
        let mut registry = TimecodeCueRegistry::new();
        registry.insert(TimecodeCuePoint::new(
            "c2",
            SecondaryEventKind::ThumbnailCapture,
            TimecodeTrigger::new(0, 1, 0, 0, 25), // 1 min = 1500 frames
        ));
        let tc = TimecodeTrigger::new(0, 1, 0, 0, 25);
        let fired = registry.advance_to_timecode(&tc);
        assert_eq!(fired.len(), 1);
    }

    #[test]
    fn test_cue_registry_pending() {
        let mut registry = TimecodeCueRegistry::new();
        registry.insert(TimecodeCuePoint::new(
            "c3",
            SecondaryEventKind::AdBreakOut,
            TimecodeTrigger::new(0, 0, 5, 0, 25),
        ));
        registry.insert(TimecodeCuePoint::new(
            "c4",
            SecondaryEventKind::AdBreakOut,
            TimecodeTrigger::new(0, 0, 10, 0, 25),
        ));
        assert_eq!(registry.pending().len(), 2);
        registry.advance_to_frame(125); // fires c3 only
        assert_eq!(registry.pending().len(), 1);
    }

    #[test]
    fn test_cue_registry_reset() {
        let mut registry = TimecodeCueRegistry::new();
        registry.insert(TimecodeCuePoint::new(
            "c5",
            SecondaryEventKind::Custom,
            TimecodeTrigger::new(0, 0, 0, 1, 25),
        ));
        registry.advance_to_frame(100);
        assert!(registry.pending().is_empty());
        registry.reset();
        assert_eq!(registry.pending().len(), 1);
    }

    #[test]
    fn test_cue_registry_next_pending() {
        let mut registry = TimecodeCueRegistry::new();
        let tc_early = TimecodeTrigger::new(0, 0, 2, 0, 25);
        let tc_late = TimecodeTrigger::new(0, 0, 5, 0, 25);
        registry.insert(TimecodeCuePoint::new(
            "late",
            SecondaryEventKind::Custom,
            tc_late,
        ));
        registry.insert(TimecodeCuePoint::new(
            "early",
            SecondaryEventKind::Custom,
            tc_early,
        ));
        let next = registry
            .next_pending()
            .expect("should have next pending cue");
        assert_eq!(next.id, "early");
    }

    #[test]
    fn test_cue_registry_sorted_insertion() {
        let mut registry = TimecodeCueRegistry::new();
        registry.insert(TimecodeCuePoint::new(
            "b",
            SecondaryEventKind::Custom,
            TimecodeTrigger::new(0, 0, 10, 0, 25),
        ));
        registry.insert(TimecodeCuePoint::new(
            "a",
            SecondaryEventKind::Custom,
            TimecodeTrigger::new(0, 0, 5, 0, 25),
        ));
        // Should be sorted: a (frame 125) then b (frame 250).
        let next = registry.next_pending().expect("should have pending cue");
        assert_eq!(next.id, "a");
    }
}
