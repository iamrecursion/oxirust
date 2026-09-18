#![allow(dead_code)]
//! Event-triggered timecode capture for marking in/out points and cue triggers.
//!
//! The [`TimecodeEventCapture`] struct accumulates [`TimecodeEvent`] records
//! as they are triggered during a session.  Events carry a captured timecode,
//! a label, and an optional payload so they can be used for:
//!
//! - Marking edit in/out points during a live or offline session.
//! - Recording cue triggers (e.g. lighting, effects, playback).
//! - Logging any user-defined production note alongside a precise timecode.
//!
//! # Example
//!
//! ```rust,no_run
//! use oximedia_timecode::{FrameRate, Timecode, timecode_event::TimecodeEventCapture};
//!
//! let mut capture = TimecodeEventCapture::new();
//! let tc = Timecode::new(1, 0, 0, 0, FrameRate::Fps25).expect("valid");
//! capture.mark_in(tc);
//! let out_tc = Timecode::new(1, 0, 30, 0, FrameRate::Fps25).expect("valid");
//! capture.mark_out(out_tc);
//! ```

use crate::Timecode;

// ---------------------------------------------------------------------------
// Event kind
// ---------------------------------------------------------------------------

/// The kind of a captured timecode event.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum EventKind {
    /// Edit mark-in point.
    MarkIn,
    /// Edit mark-out point.
    MarkOut,
    /// Generic cue trigger (carry-along label in [`TimecodeEvent::label`]).
    Cue,
    /// User-defined / arbitrary event.
    Custom(String),
}

impl std::fmt::Display for EventKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EventKind::MarkIn => write!(f, "MARK_IN"),
            EventKind::MarkOut => write!(f, "MARK_OUT"),
            EventKind::Cue => write!(f, "CUE"),
            EventKind::Custom(s) => write!(f, "CUSTOM:{}", s),
        }
    }
}

// ---------------------------------------------------------------------------
// Event record
// ---------------------------------------------------------------------------

/// A single timecode event record.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TimecodeEvent {
    /// The timecode at which this event was captured.
    pub timecode: Timecode,
    /// The kind of event.
    pub kind: EventKind,
    /// Optional human-readable label (e.g., "Scene 3 Slate").
    pub label: String,
    /// Optional arbitrary payload (e.g., JSON-encoded metadata).
    pub payload: Option<String>,
}

impl TimecodeEvent {
    /// Create a new event with the given kind and an empty label.
    pub fn new(timecode: Timecode, kind: EventKind) -> Self {
        Self {
            timecode,
            kind,
            label: String::new(),
            payload: None,
        }
    }

    /// Attach a human-readable label to this event.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = label.into();
        self
    }

    /// Attach an arbitrary payload string to this event.
    pub fn with_payload(mut self, payload: impl Into<String>) -> Self {
        self.payload = Some(payload.into());
        self
    }
}

impl std::fmt::Display for TimecodeEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {} {}", self.kind, self.timecode, self.label)
    }
}

// ---------------------------------------------------------------------------
// Edit range (in/out pair)
// ---------------------------------------------------------------------------

/// An in/out range captured from mark-in and mark-out events.
#[derive(Debug, Clone)]
pub struct EditRange {
    /// Mark-in timecode.
    pub mark_in: Timecode,
    /// Mark-out timecode.
    pub mark_out: Timecode,
}

impl EditRange {
    /// Duration of the edit range in frames.
    ///
    /// Returns 0 if mark_out is at or before mark_in.
    pub fn duration_frames(&self) -> u64 {
        let fi = self.mark_in.to_frames();
        let fo = self.mark_out.to_frames();
        fo.saturating_sub(fi)
    }

    /// Duration of the edit range in seconds (approximate for pull-down rates).
    pub fn duration_seconds(&self) -> f64 {
        self.mark_out.to_seconds_f64() - self.mark_in.to_seconds_f64()
    }
}

// ---------------------------------------------------------------------------
// Capture controller
// ---------------------------------------------------------------------------

/// Accumulates timecode events recorded during a production session.
///
/// Thread safety: [`TimecodeEventCapture`] is not `Sync`; wrap in a `Mutex`
/// or `RwLock` if sharing across threads.
#[derive(Debug, Default)]
pub struct TimecodeEventCapture {
    /// All captured events, in the order they were recorded.
    events: Vec<TimecodeEvent>,
    /// The most recent mark-in timecode, if any (used by [`close_range`]).
    pending_mark_in: Option<Timecode>,
}

impl TimecodeEventCapture {
    /// Create an empty capture session.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a mark-in event at `tc`.
    ///
    /// Also stores `tc` as the pending mark-in so that a subsequent
    /// [`mark_out`](Self::mark_out) can construct an [`EditRange`].
    pub fn mark_in(&mut self, tc: Timecode) {
        self.pending_mark_in = Some(tc);
        self.events.push(TimecodeEvent::new(tc, EventKind::MarkIn));
    }

    /// Record a mark-out event at `tc`.
    ///
    /// Returns the completed [`EditRange`] if a pending mark-in exists.
    pub fn mark_out(&mut self, tc: Timecode) -> Option<EditRange> {
        self.events.push(TimecodeEvent::new(tc, EventKind::MarkOut));
        self.pending_mark_in.take().map(|mark_in| EditRange {
            mark_in,
            mark_out: tc,
        })
    }

    /// Record a cue trigger at `tc` with an optional label.
    pub fn cue(&mut self, tc: Timecode, label: impl Into<String>) {
        self.events
            .push(TimecodeEvent::new(tc, EventKind::Cue).with_label(label));
    }

    /// Record a custom event at `tc`.
    pub fn custom(
        &mut self,
        tc: Timecode,
        name: impl Into<String>,
        label: impl Into<String>,
        payload: Option<String>,
    ) {
        let mut ev = TimecodeEvent::new(tc, EventKind::Custom(name.into())).with_label(label);
        if let Some(p) = payload {
            ev = ev.with_payload(p);
        }
        self.events.push(ev);
    }

    /// Return a slice of all recorded events.
    pub fn events(&self) -> &[TimecodeEvent] {
        &self.events
    }

    /// Return only the mark-in events.
    pub fn mark_ins(&self) -> Vec<&TimecodeEvent> {
        self.events
            .iter()
            .filter(|e| e.kind == EventKind::MarkIn)
            .collect()
    }

    /// Return only the mark-out events.
    pub fn mark_outs(&self) -> Vec<&TimecodeEvent> {
        self.events
            .iter()
            .filter(|e| e.kind == EventKind::MarkOut)
            .collect()
    }

    /// Reconstruct all completed in/out ranges from the event log.
    ///
    /// Pairs mark-in events with the next mark-out event that follows them.
    pub fn edit_ranges(&self) -> Vec<EditRange> {
        let mut ranges = Vec::new();
        let mut pending: Option<Timecode> = None;

        for ev in &self.events {
            match &ev.kind {
                EventKind::MarkIn => {
                    pending = Some(ev.timecode);
                }
                EventKind::MarkOut => {
                    if let Some(mark_in) = pending.take() {
                        ranges.push(EditRange {
                            mark_in,
                            mark_out: ev.timecode,
                        });
                    }
                }
                _ => {}
            }
        }

        ranges
    }

    /// Clear all events and reset pending mark-in state.
    pub fn clear(&mut self) {
        self.events.clear();
        self.pending_mark_in = None;
    }

    /// Number of recorded events.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Whether no events have been recorded.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FrameRate;

    fn tc(h: u8, m: u8, s: u8, f: u8) -> Timecode {
        Timecode::new(h, m, s, f, FrameRate::Fps25).expect("valid timecode")
    }

    #[test]
    fn test_mark_in_records_event() {
        let mut cap = TimecodeEventCapture::new();
        cap.mark_in(tc(0, 0, 1, 0));
        assert_eq!(cap.len(), 1);
        assert_eq!(cap.events()[0].kind, EventKind::MarkIn);
    }

    #[test]
    fn test_mark_out_returns_range() {
        let mut cap = TimecodeEventCapture::new();
        cap.mark_in(tc(0, 0, 1, 0));
        let range = cap.mark_out(tc(0, 0, 5, 0));
        assert!(range.is_some());
        let r = range.expect("should have range");
        assert_eq!(r.duration_frames(), 4 * 25);
    }

    #[test]
    fn test_mark_out_without_mark_in_returns_none() {
        let mut cap = TimecodeEventCapture::new();
        let range = cap.mark_out(tc(0, 0, 5, 0));
        assert!(range.is_none());
    }

    #[test]
    fn test_cue_event() {
        let mut cap = TimecodeEventCapture::new();
        cap.cue(tc(0, 1, 0, 0), "Scene 1");
        assert_eq!(cap.len(), 1);
        assert_eq!(cap.events()[0].kind, EventKind::Cue);
        assert_eq!(cap.events()[0].label, "Scene 1");
    }

    #[test]
    fn test_custom_event() {
        let mut cap = TimecodeEventCapture::new();
        cap.custom(
            tc(0, 2, 0, 0),
            "FLASH",
            "Harding flash detected",
            Some("{\"severity\":\"high\"}".into()),
        );
        assert_eq!(cap.len(), 1);
        assert!(matches!(&cap.events()[0].kind, EventKind::Custom(n) if n == "FLASH"));
        assert!(cap.events()[0].payload.is_some());
    }

    #[test]
    fn test_edit_ranges_reconstruction() {
        let mut cap = TimecodeEventCapture::new();
        cap.mark_in(tc(0, 0, 1, 0));
        cap.mark_out(tc(0, 0, 5, 0));
        cap.mark_in(tc(0, 1, 0, 0));
        cap.mark_out(tc(0, 1, 30, 0));

        let ranges = cap.edit_ranges();
        assert_eq!(ranges.len(), 2);
        assert_eq!(ranges[0].duration_frames(), 4 * 25);
        assert_eq!(ranges[1].duration_frames(), 30 * 25);
    }

    #[test]
    fn test_mark_ins_filter() {
        let mut cap = TimecodeEventCapture::new();
        cap.mark_in(tc(0, 0, 1, 0));
        cap.cue(tc(0, 0, 2, 0), "cue");
        cap.mark_out(tc(0, 0, 5, 0));
        assert_eq!(cap.mark_ins().len(), 1);
        assert_eq!(cap.mark_outs().len(), 1);
    }

    #[test]
    fn test_clear_resets_state() {
        let mut cap = TimecodeEventCapture::new();
        cap.mark_in(tc(0, 0, 1, 0));
        cap.clear();
        assert!(cap.is_empty());
        // After clear, mark_out should return None (no pending mark_in)
        let range = cap.mark_out(tc(0, 0, 5, 0));
        assert!(range.is_none());
    }

    #[test]
    fn test_duration_seconds() {
        let r = EditRange {
            mark_in: tc(0, 0, 0, 0),
            mark_out: tc(0, 0, 4, 0),
        };
        assert!((r.duration_seconds() - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_event_display() {
        let ev = TimecodeEvent::new(tc(1, 2, 3, 4), EventKind::Cue).with_label("test");
        let s = ev.to_string();
        assert!(s.contains("CUE"));
        assert!(s.contains("01:02:03:04"));
    }

    #[test]
    fn test_event_kind_display() {
        assert_eq!(EventKind::MarkIn.to_string(), "MARK_IN");
        assert_eq!(EventKind::MarkOut.to_string(), "MARK_OUT");
        assert_eq!(EventKind::Cue.to_string(), "CUE");
        assert_eq!(EventKind::Custom("FOO".into()).to_string(), "CUSTOM:FOO");
    }
}
