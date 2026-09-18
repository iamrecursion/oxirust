//! Session recording and playback for collaborative edit history replay.
//!
//! Records a time-ordered log of collaboration operations during a live session
//! and supports replaying them at arbitrary speeds, including step-by-step
//! inspection and range-based extraction.

#![allow(dead_code)]

use crate::operation_log::Operation;
use std::collections::VecDeque;

// ─────────────────────────────────────────────────────────────────────────────
// RecordedEvent
// ─────────────────────────────────────────────────────────────────────────────

/// The kind of event that was captured in the recording.
#[derive(Debug, Clone, PartialEq)]
pub enum RecordedEventKind {
    /// A document operation was applied.
    Operation(Operation),
    /// A user joined the session.
    UserJoined {
        user_id: String,
        display_name: String,
    },
    /// A user left the session.
    UserLeft { user_id: String },
    /// A cursor moved to a new timeline position.
    CursorMoved {
        user_id: String,
        frame: u64,
        track: u32,
    },
    /// A snapshot was committed.
    SnapshotCommitted {
        snapshot_id: u64,
        description: String,
    },
    /// A lock was acquired on a resource.
    LockAcquired { user_id: String, resource: String },
    /// A lock was released.
    LockReleased { user_id: String, resource: String },
    /// An annotation was added.
    AnnotationAdded { user_id: String, annotation_id: u64 },
}

/// A single event captured in the session recording.
#[derive(Debug, Clone)]
pub struct RecordedEvent {
    /// Sequential event number (monotonically increasing from 0).
    pub seq: u64,
    /// Wall-clock timestamp in milliseconds since the Unix epoch.
    pub timestamp_ms: u64,
    /// The event payload.
    pub kind: RecordedEventKind,
}

impl RecordedEvent {
    fn new(seq: u64, timestamp_ms: u64, kind: RecordedEventKind) -> Self {
        Self {
            seq,
            timestamp_ms,
            kind,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SessionRecorder
// ─────────────────────────────────────────────────────────────────────────────

/// Records collaboration session events for later replay.
#[derive(Debug)]
pub struct SessionRecorder {
    /// All captured events in chronological order.
    pub events: Vec<RecordedEvent>,
    /// Sequential counter for event IDs.
    next_seq: u64,
    /// Whether recording is currently active.
    pub is_recording: bool,
    /// Session identifier.
    pub session_id: String,
    /// When the recording started (epoch milliseconds, 0 if not started).
    pub started_at_ms: u64,
}

impl SessionRecorder {
    /// Create a new recorder for the given session.
    #[must_use]
    pub fn new(session_id: impl Into<String>) -> Self {
        Self {
            events: Vec::new(),
            next_seq: 0,
            is_recording: false,
            session_id: session_id.into(),
            started_at_ms: 0,
        }
    }

    /// Start recording.
    pub fn start(&mut self, now_ms: u64) {
        self.is_recording = true;
        self.started_at_ms = now_ms;
    }

    /// Stop recording.
    pub fn stop(&mut self) {
        self.is_recording = false;
    }

    /// Record an event at the given timestamp.  Returns the assigned sequence
    /// number, or `None` if recording is paused.
    pub fn record(&mut self, timestamp_ms: u64, kind: RecordedEventKind) -> Option<u64> {
        if !self.is_recording {
            return None;
        }
        let seq = self.next_seq;
        self.next_seq += 1;
        self.events
            .push(RecordedEvent::new(seq, timestamp_ms, kind));
        Some(seq)
    }

    /// Convenience: record an operation event.
    pub fn record_operation(&mut self, op: Operation, timestamp_ms: u64) -> Option<u64> {
        self.record(timestamp_ms, RecordedEventKind::Operation(op))
    }

    /// Convenience: record a user-joined event.
    pub fn record_user_joined(
        &mut self,
        user_id: impl Into<String>,
        display_name: impl Into<String>,
        timestamp_ms: u64,
    ) -> Option<u64> {
        self.record(
            timestamp_ms,
            RecordedEventKind::UserJoined {
                user_id: user_id.into(),
                display_name: display_name.into(),
            },
        )
    }

    /// Convenience: record a cursor-moved event.
    pub fn record_cursor(
        &mut self,
        user_id: impl Into<String>,
        frame: u64,
        track: u32,
        timestamp_ms: u64,
    ) -> Option<u64> {
        self.record(
            timestamp_ms,
            RecordedEventKind::CursorMoved {
                user_id: user_id.into(),
                frame,
                track,
            },
        )
    }

    /// Total number of recorded events.
    #[must_use]
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Total recording duration in milliseconds.
    ///
    /// Returns 0 for an empty or single-event recording.
    #[must_use]
    pub fn duration_ms(&self) -> u64 {
        match (self.events.first(), self.events.last()) {
            (Some(first), Some(last)) => last.timestamp_ms.saturating_sub(first.timestamp_ms),
            _ => 0,
        }
    }

    /// Extract a sub-range of events by timestamp.
    #[must_use]
    pub fn events_in_range(&self, start_ms: u64, end_ms: u64) -> Vec<&RecordedEvent> {
        self.events
            .iter()
            .filter(|e| e.timestamp_ms >= start_ms && e.timestamp_ms <= end_ms)
            .collect()
    }

    /// Extract only operation events, suitable for replaying document state.
    #[must_use]
    pub fn operation_events(&self) -> Vec<&RecordedEvent> {
        self.events
            .iter()
            .filter(|e| matches!(e.kind, RecordedEventKind::Operation(_)))
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PlaybackSession
// ─────────────────────────────────────────────────────────────────────────────

/// Playback speed multiplier.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlaybackSpeed(pub f64);

impl PlaybackSpeed {
    /// Real-time (1×).
    pub const REALTIME: Self = Self(1.0);
    /// Double speed (2×).
    pub const FAST: Self = Self(2.0);
    /// Half speed (0.5×).
    pub const SLOW: Self = Self(0.5);

    /// Return the underlying multiplier.
    #[must_use]
    pub fn factor(self) -> f64 {
        self.0
    }

    /// Scale a duration by the playback speed.
    #[must_use]
    pub fn scale_ms(self, real_ms: u64) -> u64 {
        if self.0 <= 0.0 {
            return 0;
        }
        (real_ms as f64 / self.0) as u64
    }
}

impl std::fmt::Display for PlaybackSpeed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.2}x", self.0)
    }
}

/// A playback session that streams events from a `SessionRecorder` at a
/// configurable speed.
///
/// The session maintains a read cursor and exposes events one by one via
/// [`PlaybackSession::next_event`].
#[derive(Debug)]
pub struct PlaybackSession<'r> {
    recorder: &'r SessionRecorder,
    /// Index of the next event to emit.
    cursor: usize,
    /// Playback speed.
    pub speed: PlaybackSpeed,
    /// Virtual "now" in recording-time milliseconds.  Advanced each call to
    /// `advance_to`.
    pub virtual_time_ms: u64,
}

impl<'r> PlaybackSession<'r> {
    /// Create a playback session from a recorder.
    #[must_use]
    pub fn new(recorder: &'r SessionRecorder, speed: PlaybackSpeed) -> Self {
        let start = recorder.events.first().map(|e| e.timestamp_ms).unwrap_or(0);
        Self {
            recorder,
            cursor: 0,
            speed,
            virtual_time_ms: start,
        }
    }

    /// Advance the virtual clock by `real_elapsed_ms` wall-clock milliseconds
    /// and return all events that fall within the newly elapsed range.
    pub fn advance(&mut self, real_elapsed_ms: u64) -> Vec<&'r RecordedEvent> {
        let delta = self.speed.scale_ms(real_elapsed_ms);
        let new_virtual_time = self
            .virtual_time_ms
            .saturating_add((real_elapsed_ms as f64 * self.speed.factor()) as u64);
        let prev_vt = self.virtual_time_ms;
        self.virtual_time_ms = new_virtual_time;
        let _ = delta; // used above

        let mut emitted = Vec::new();
        while self.cursor < self.recorder.events.len() {
            let ev = &self.recorder.events[self.cursor];
            if ev.timestamp_ms
                >= self.recorder.started_at_ms
                    + prev_vt.saturating_sub(
                        self.recorder
                            .events
                            .first()
                            .map(|e| e.timestamp_ms)
                            .unwrap_or(0),
                    )
                && ev.timestamp_ms
                    <= self
                        .recorder
                        .events
                        .first()
                        .map(|e| e.timestamp_ms)
                        .unwrap_or(0)
                        + self.virtual_time_ms.saturating_sub(
                            self.recorder
                                .events
                                .first()
                                .map(|e| e.timestamp_ms)
                                .unwrap_or(0),
                        )
            {
                emitted.push(ev);
                self.cursor += 1;
            } else {
                break;
            }
        }
        emitted
    }

    /// Return the next event in the recording without time-gating, or `None`
    /// when all events have been consumed (step-by-step mode).
    pub fn next_event(&mut self) -> Option<&'r RecordedEvent> {
        if self.cursor < self.recorder.events.len() {
            let ev = &self.recorder.events[self.cursor];
            self.cursor += 1;
            self.virtual_time_ms = ev.timestamp_ms;
            Some(ev)
        } else {
            None
        }
    }

    /// Seek to a specific sequence number.  The cursor will be positioned
    /// just before the event with `seq >= target_seq`.
    pub fn seek_to_seq(&mut self, target_seq: u64) {
        self.cursor = self
            .recorder
            .events
            .iter()
            .position(|e| e.seq >= target_seq)
            .unwrap_or(self.recorder.events.len());
        if let Some(ev) = self.recorder.events.get(self.cursor) {
            self.virtual_time_ms = ev.timestamp_ms;
        }
    }

    /// Whether all events have been consumed.
    #[must_use]
    pub fn is_finished(&self) -> bool {
        self.cursor >= self.recorder.events.len()
    }

    /// Number of events remaining.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.recorder.events.len().saturating_sub(self.cursor)
    }

    /// Collect all remaining events and mark the playback as finished.
    pub fn drain_remaining(&mut self) -> Vec<&'r RecordedEvent> {
        let events: Vec<&RecordedEvent> = self.recorder.events[self.cursor..].iter().collect();
        self.cursor = self.recorder.events.len();
        events
    }

    /// Replay all operation events and return the resulting `Vec<f32>` state.
    ///
    /// Non-operation events are silently skipped.  If any operation returns an
    /// error it is also skipped (permissive replay).
    pub fn replay_to_state(&self, initial: Vec<f32>) -> Vec<f32> {
        let mut state = initial;
        for ev in &self.recorder.events[..self.cursor] {
            if let RecordedEventKind::Operation(op) = &ev.kind {
                let _ = crate::operation_log::apply(&mut state, op);
            }
        }
        state
    }
}

/// Collect all operations from a recording into a `VecDeque` for batch
/// processing (e.g. re-applying to a fresh document).
#[must_use]
pub fn extract_operations(recorder: &SessionRecorder) -> VecDeque<Operation> {
    recorder
        .events
        .iter()
        .filter_map(|e| {
            if let RecordedEventKind::Operation(op) = &e.kind {
                Some(op.clone())
            } else {
                None
            }
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operation_log::{OpType, Operation};

    fn make_op(id: u64, index: usize, value: f32) -> Operation {
        Operation::new(id, 1, 0, "test", OpType::Insert { index, value })
    }

    #[test]
    fn test_recorder_start_stop() {
        let mut rec = SessionRecorder::new("session-1");
        assert!(!rec.is_recording);
        rec.start(0);
        assert!(rec.is_recording);
        rec.stop();
        assert!(!rec.is_recording);
    }

    #[test]
    fn test_recorder_record_while_stopped_returns_none() {
        let mut rec = SessionRecorder::new("s1");
        // not started
        let result = rec.record(
            1_000,
            RecordedEventKind::UserLeft {
                user_id: "u1".to_string(),
            },
        );
        assert!(result.is_none());
        assert_eq!(rec.event_count(), 0);
    }

    #[test]
    fn test_recorder_record_sequence_numbers() {
        let mut rec = SessionRecorder::new("s1");
        rec.start(0);
        let s1 = rec
            .record_user_joined("u1", "Alice", 100)
            .expect("should record");
        let s2 = rec
            .record_user_joined("u2", "Bob", 200)
            .expect("should record");
        assert_eq!(s1, 0);
        assert_eq!(s2, 1);
        assert_eq!(rec.event_count(), 2);
    }

    #[test]
    fn test_recorder_duration_ms() {
        let mut rec = SessionRecorder::new("s1");
        rec.start(0);
        rec.record_operation(make_op(1, 0, 1.0), 1_000);
        rec.record_operation(make_op(2, 1, 2.0), 5_000);
        assert_eq!(rec.duration_ms(), 4_000);
    }

    #[test]
    fn test_recorder_events_in_range() {
        let mut rec = SessionRecorder::new("s1");
        rec.start(0);
        rec.record_operation(make_op(1, 0, 1.0), 1_000);
        rec.record_operation(make_op(2, 1, 2.0), 3_000);
        rec.record_operation(make_op(3, 2, 3.0), 6_000);
        let range = rec.events_in_range(1_000, 3_000);
        assert_eq!(range.len(), 2);
    }

    #[test]
    fn test_recorder_operation_events_filter() {
        let mut rec = SessionRecorder::new("s1");
        rec.start(0);
        rec.record_operation(make_op(1, 0, 1.0), 1_000);
        rec.record_user_joined("u1", "Alice", 1_500);
        rec.record_operation(make_op(2, 1, 2.0), 2_000);
        let ops = rec.operation_events();
        assert_eq!(ops.len(), 2);
    }

    #[test]
    fn test_extract_operations() {
        let mut rec = SessionRecorder::new("s1");
        rec.start(0);
        rec.record_operation(make_op(1, 0, 1.0), 1_000);
        rec.record_user_joined("u1", "Alice", 1_500);
        rec.record_operation(make_op(2, 1, 2.0), 2_000);
        let ops = extract_operations(&rec);
        assert_eq!(ops.len(), 2);
    }

    #[test]
    fn test_playback_step_by_step() {
        let mut rec = SessionRecorder::new("s1");
        rec.start(0);
        rec.record_operation(make_op(1, 0, 1.0), 1_000);
        rec.record_operation(make_op(2, 1, 2.0), 2_000);
        rec.record_operation(make_op(3, 2, 3.0), 3_000);

        let mut playback = PlaybackSession::new(&rec, PlaybackSpeed::REALTIME);
        assert!(!playback.is_finished());
        assert_eq!(playback.remaining(), 3);

        let ev1 = playback.next_event().expect("first event");
        assert_eq!(ev1.seq, 0);

        let ev2 = playback.next_event().expect("second event");
        assert_eq!(ev2.seq, 1);

        assert_eq!(playback.remaining(), 1);
    }

    #[test]
    fn test_playback_drain_remaining() {
        let mut rec = SessionRecorder::new("s1");
        rec.start(0);
        for i in 0u64..5 {
            rec.record_operation(make_op(i, 0, i as f32), i * 1_000);
        }
        let mut playback = PlaybackSession::new(&rec, PlaybackSpeed::REALTIME);
        let drained = playback.drain_remaining();
        assert_eq!(drained.len(), 5);
        assert!(playback.is_finished());
    }

    #[test]
    fn test_playback_seek_to_seq() {
        let mut rec = SessionRecorder::new("s1");
        rec.start(0);
        for i in 0u64..10 {
            rec.record_operation(make_op(i, 0, i as f32), i * 500);
        }
        let mut playback = PlaybackSession::new(&rec, PlaybackSpeed::REALTIME);
        playback.seek_to_seq(5);
        let ev = playback.next_event().expect("event after seek");
        assert_eq!(ev.seq, 5);
    }

    #[test]
    fn test_playback_replay_to_state() {
        let mut rec = SessionRecorder::new("s1");
        rec.start(0);
        rec.record_operation(
            Operation::new(
                1,
                1,
                0,
                "t",
                OpType::Insert {
                    index: 0,
                    value: 10.0,
                },
            ),
            1_000,
        );
        rec.record_operation(
            Operation::new(
                2,
                1,
                0,
                "t",
                OpType::Insert {
                    index: 1,
                    value: 20.0,
                },
            ),
            2_000,
        );

        let mut playback = PlaybackSession::new(&rec, PlaybackSpeed::REALTIME);
        // Consume both events so cursor = 2.
        playback.next_event();
        playback.next_event();

        let state = playback.replay_to_state(Vec::new());
        assert_eq!(state, vec![10.0, 20.0]);
    }

    #[test]
    fn test_playback_speed_scale() {
        assert_eq!(PlaybackSpeed::FAST.scale_ms(1_000), 500); // 1000 / 2
        assert_eq!(PlaybackSpeed::SLOW.scale_ms(1_000), 2_000); // 1000 / 0.5
        assert_eq!(PlaybackSpeed::REALTIME.scale_ms(1_000), 1_000);
    }

    #[test]
    fn test_playback_speed_display() {
        assert_eq!(PlaybackSpeed::REALTIME.to_string(), "1.00x");
        assert_eq!(PlaybackSpeed::FAST.to_string(), "2.00x");
    }

    #[test]
    fn test_cursor_event_recording() {
        let mut rec = SessionRecorder::new("s1");
        rec.start(0);
        rec.record_cursor("u1", 240, 2, 1_000);
        assert_eq!(rec.event_count(), 1);
        let ev = &rec.events[0];
        assert!(matches!(
            ev.kind,
            RecordedEventKind::CursorMoved { ref user_id, frame: 240, track: 2 }
            if user_id == "u1"
        ));
    }
}
