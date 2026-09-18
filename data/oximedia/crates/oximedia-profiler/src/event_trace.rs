//! Event tracing for profiler-level diagnostic recording.
//!
//! Captures structured trace events tagged with severity levels, enabling
//! post-hoc filtering and analysis of pipeline execution.
//!
//! # Zero-allocation design
//!
//! [`TraceEvent`] is `Copy`; heap strings have been replaced with:
//!
//! - `component: u32` — index into a [`StringTable`] (cold intern path).
//! - `message: [u8; 64]` + `msg_len: u8` — inline UTF-8 storage; messages
//!   longer than 64 bytes are silently truncated at a UTF-8 character
//!   boundary with a `…` marker.
//! - `level: u8` — packed [`TraceLevel`] as an integer.
//!
//! The hot `emit` path therefore performs **no heap allocation**.
//!
//! The event buffer is an [`EventRingBuffer<TraceEvent, N>`] with `N = 4096`,
//! so the oldest event is overwritten when the buffer fills without any
//! dynamic allocation.
//!
//! ## Honest caveats
//!
//! Two paths are *not* zero-allocation, by design, and neither is the hot
//! recording path:
//!
//! - **First-sight component interning.** [`StringTable::intern`] inserts into
//!   a `HashMap` the very first time a given `&'static str` component name is
//!   seen, which may allocate. Every subsequent `emit` for that component is a
//!   pure lookup. With a fixed, bounded set of component names the cost is
//!   amortized to zero; the steady state is allocation-free.
//! - **Draining for reports.** [`EventTrace::drain`] hands back an owned
//!   `Vec<TraceEvent>` (the cold reporting path). Callers that want to avoid
//!   even that allocation can reuse a buffer via
//!   [`EventTrace::drain_into`].

use crate::event_ring_buffer::EventRingBuffer;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// TraceLevel
// ---------------------------------------------------------------------------

/// Severity level of a trace event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum TraceLevel {
    /// Very fine-grained diagnostic information.
    Trace,
    /// General informational message.
    Info,
    /// Potentially problematic situation.
    Warn,
    /// Error that did not halt execution.
    Error,
    /// Critical failure.
    Critical,
}

impl TraceLevel {
    /// Returns a short string label for the level.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Trace => "TRACE",
            Self::Info => "INFO",
            Self::Warn => "WARN",
            Self::Error => "ERROR",
            Self::Critical => "CRIT",
        }
    }

    /// Returns `true` if this level is at or above `other`.
    #[must_use]
    pub fn is_at_least(&self, other: TraceLevel) -> bool {
        *self >= other
    }

    /// Pack as `u8`.
    #[inline]
    fn as_u8(self) -> u8 {
        match self {
            Self::Trace => 0,
            Self::Info => 1,
            Self::Warn => 2,
            Self::Error => 3,
            Self::Critical => 4,
        }
    }

    /// Unpack from `u8` (unknown values map to `Trace`).
    #[inline]
    fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::Info,
            2 => Self::Warn,
            3 => Self::Error,
            4 => Self::Critical,
            _ => Self::Trace,
        }
    }
}

// ---------------------------------------------------------------------------
// Inline message storage
// ---------------------------------------------------------------------------

/// Maximum number of bytes stored inline in a [`TraceEvent`] message.
///
/// Messages longer than this are truncated at the nearest UTF-8 character
/// boundary and a `…` suffix is inserted.
const MSG_CAP: usize = 64;

/// Encode `s` into a `[u8; MSG_CAP]` + length byte.
///
/// The function is allocation-free: it copies bytes directly into the array.
fn encode_message(s: &str) -> ([u8; MSG_CAP], u8) {
    let bytes = s.as_bytes();
    if bytes.len() <= MSG_CAP {
        let mut buf = [0u8; MSG_CAP];
        buf[..bytes.len()].copy_from_slice(bytes);
        (buf, bytes.len() as u8)
    } else {
        // Truncate to the longest UTF-8-valid prefix that fits the suffix `…`
        // (3 bytes) within MSG_CAP.
        let suffix = b"\xe2\x80\xa6"; // UTF-8 for U+2026 HORIZONTAL ELLIPSIS
        let max_body = MSG_CAP - suffix.len();
        // Walk back from max_body to find a UTF-8 character boundary.
        let mut end = max_body;
        while end > 0 && (bytes[end] & 0xC0) == 0x80 {
            end -= 1;
        }
        let mut buf = [0u8; MSG_CAP];
        buf[..end].copy_from_slice(&bytes[..end]);
        buf[end..end + suffix.len()].copy_from_slice(suffix);
        let len = end + suffix.len();
        (buf, len as u8)
    }
}

/// Decode an inline message back to a `String`.
fn decode_message(buf: &[u8; MSG_CAP], len: u8) -> String {
    let slice = &buf[..len as usize];
    // SAFETY: encode_message guarantees the bytes form valid UTF-8 (we only
    // cut at char boundaries).  We use from_utf8_lossy as a belt-and-
    // suspenders fallback.
    String::from_utf8_lossy(slice).into_owned()
}

// ---------------------------------------------------------------------------
// TraceEvent (Copy)
// ---------------------------------------------------------------------------

/// A single trace event stored in the ring buffer.
///
/// This type is `Copy` so that it can live in the zero-allocation
/// [`EventRingBuffer`] without heap indirection.
#[derive(Clone, Copy)]
pub struct TraceEvent {
    /// Wall-clock offset from the tracing session start (nanoseconds).
    pub offset_ns: u64,
    /// Packed [`TraceLevel`] value.
    level_u8: u8,
    /// Index into the [`StringTable`] that maps to the component name.
    pub component_id: u32,
    /// Inline message bytes.
    msg_buf: [u8; MSG_CAP],
    /// Valid byte length within `msg_buf`.
    msg_len: u8,
}

// Manual Debug so we don't print raw byte arrays.
impl std::fmt::Debug for TraceEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TraceEvent")
            .field("offset_ns", &self.offset_ns)
            .field("level", &self.level())
            .field("component_id", &self.component_id)
            .field("message", &self.message())
            .finish()
    }
}

impl Default for TraceEvent {
    fn default() -> Self {
        Self {
            offset_ns: 0,
            level_u8: 0,
            component_id: 0,
            msg_buf: [0u8; MSG_CAP],
            msg_len: 0,
        }
    }
}

impl TraceEvent {
    /// Creates a new trace event from raw parts.
    #[must_use]
    pub fn new(offset_ns: u64, level: TraceLevel, component_id: u32, message: &str) -> Self {
        let (msg_buf, msg_len) = encode_message(message);
        Self {
            offset_ns,
            level_u8: level.as_u8(),
            component_id,
            msg_buf,
            msg_len,
        }
    }

    /// Returns the [`TraceLevel`] of this event.
    #[must_use]
    #[inline]
    pub fn level(&self) -> TraceLevel {
        TraceLevel::from_u8(self.level_u8)
    }

    /// Returns the message as a `String` (decodes inline storage).
    #[must_use]
    pub fn message(&self) -> String {
        decode_message(&self.msg_buf, self.msg_len)
    }
}

// ---------------------------------------------------------------------------
// StringTable
// ---------------------------------------------------------------------------

/// Cold intern table mapping `&'static str` component names to `u32` ids.
///
/// All string interning happens on the **cold** path (`emit`/`start_session`);
/// the hot path only touches `u32` ids.
#[derive(Debug, Default)]
pub struct StringTable {
    map: HashMap<&'static str, u32>,
    next_id: u32,
}

impl StringTable {
    /// Creates an empty string table.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Interns `name`, returning its stable `u32` id.
    ///
    /// This method allocates only the first time a given string is seen.
    pub fn intern(&mut self, name: &'static str) -> u32 {
        if let Some(&id) = self.map.get(name) {
            return id;
        }
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
        self.map.insert(name, id);
        id
    }

    /// Returns the `u32` id for `name`, or `u32::MAX` if not yet interned.
    #[must_use]
    pub fn lookup(&self, name: &'static str) -> Option<u32> {
        self.map.get(name).copied()
    }

    /// Looks up the `&'static str` for a given `id`, if any.
    #[must_use]
    pub fn resolve(&self, id: u32) -> Option<&'static str> {
        self.map.iter().find(|(_, &v)| v == id).map(|(k, _)| *k)
    }
}

// ---------------------------------------------------------------------------
// Ring-buffer capacity
// ---------------------------------------------------------------------------

/// Compile-time capacity of the internal ring buffer.
///
/// 4 096 events: large enough for most profiling sessions, small enough to
/// keep the `EventTrace` stack-allocatable (each `TraceEvent` is ~72 bytes →
/// ~288 KiB total, kept on the heap inside `Box`).
const RING_CAP: usize = 4096;

// ---------------------------------------------------------------------------
// EventTrace
// ---------------------------------------------------------------------------

/// Trace event recorder backed by a zero-allocation ring buffer.
///
/// The hot [`emit`](Self::emit) path:
/// 1. Checks the level filter (`u8` compare).
/// 2. Computes the offset from `session_start`.
/// 3. Encodes the message into inline storage (byte copy, no alloc).
/// 4. Pushes a `Copy` event into the ring buffer (array write, no alloc).
///
/// The cold intern path (string → `u32`) only runs when a new component name
/// is seen for the first time.
#[derive(Debug)]
pub struct EventTrace {
    /// Zero-allocation ring buffer.
    events: Box<EventRingBuffer<TraceEvent, RING_CAP>>,
    /// Minimum level that will be recorded (events below this are dropped).
    min_level: TraceLevel,
    /// Session start instant used to compute event offsets.
    session_start: Option<Instant>,
    /// Component name → id intern table (cold path).
    string_table: StringTable,
}

impl EventTrace {
    /// Creates a new `EventTrace` with the given minimum level.
    ///
    /// The `capacity` parameter is accepted for API compatibility but the
    /// actual ring buffer size is fixed at `RING_CAP`.  Values larger than
    /// `RING_CAP` are silently clamped.
    #[must_use]
    pub fn new(_capacity: usize, min_level: TraceLevel) -> Self {
        Self {
            events: Box::new(EventRingBuffer::new()),
            min_level,
            session_start: None,
            string_table: StringTable::new(),
        }
    }

    /// Creates an `EventTrace` with `Trace`-level recording (captures
    /// everything).
    #[must_use]
    pub fn verbose(_capacity: usize) -> Self {
        Self::new(_capacity, TraceLevel::Trace)
    }

    /// Starts a new tracing session, resetting the clock and buffer.
    pub fn start_session(&mut self) {
        self.session_start = Some(Instant::now());
        self.events.clear();
    }

    /// Returns `true` if a session has been started.
    #[must_use]
    pub fn has_session(&self) -> bool {
        self.session_start.is_some()
    }

    /// Returns the current session duration, if a session is active.
    #[must_use]
    pub fn session_duration(&self) -> Option<Duration> {
        self.session_start.map(|t| t.elapsed())
    }

    /// Emits an event into the trace buffer.
    ///
    /// Events below `min_level` are silently dropped.  When the buffer reaches
    /// capacity the oldest event is overwritten (ring semantics — no
    /// allocation).
    ///
    /// `component` must be a `&'static str` so it can be interned with zero
    /// heap allocation after the first occurrence.
    pub fn emit(&mut self, level: TraceLevel, component: &'static str, message: &str) {
        if level < self.min_level {
            return;
        }
        let offset_ns = self
            .session_start
            .map(|t| t.elapsed().as_nanos() as u64)
            .unwrap_or(0);
        let component_id = self.string_table.intern(component);
        let event = TraceEvent::new(offset_ns, level, component_id, message);
        self.events.push(event);
    }

    /// Inserts a pre-built event, respecting the level filter.
    pub fn push(&mut self, event: TraceEvent) {
        if event.level() < self.min_level {
            return;
        }
        self.events.push(event);
    }

    /// Returns all events at or above `level`.
    #[must_use]
    pub fn filter_by_level(&self, level: TraceLevel) -> Vec<TraceEvent> {
        self.events.iter().filter(|e| e.level() >= level).collect()
    }

    /// Returns all events from the given component (by static name).
    #[must_use]
    pub fn filter_by_component(&self, component: &'static str) -> Vec<TraceEvent> {
        match self.string_table.lookup(component) {
            None => Vec::new(),
            Some(id) => self
                .events
                .iter()
                .filter(|e| e.component_id == id)
                .collect(),
        }
    }

    /// Returns events whose offset falls within `[start_ns, end_ns)`.
    #[must_use]
    pub fn filter_by_time(&self, start_ns: u64, end_ns: u64) -> Vec<TraceEvent> {
        self.events
            .iter()
            .filter(|e| e.offset_ns >= start_ns && e.offset_ns < end_ns)
            .collect()
    }

    /// Returns a reference to the underlying ring buffer.
    #[must_use]
    pub fn events(&self) -> &EventRingBuffer<TraceEvent, RING_CAP> {
        &self.events
    }

    /// Returns the number of events currently stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Returns the fixed capacity of the underlying ring buffer (`RING_CAP`).
    ///
    /// This value is **invariant** for the lifetime of the `EventTrace`: the
    /// backing store is a fixed-size array, so recording can never trigger a
    /// reallocation or grow the buffer.  Use it to assert (in tests or asserts)
    /// that the hot recording path stays allocation-free.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.events.capacity()
    }

    /// Drains all recorded events into a freshly-allocated `Vec` (oldest first)
    /// and clears the buffer.
    ///
    /// This is the **cold reporting path**, not the hot recording path: the
    /// returned `Vec` is heap-allocated.  [`emit`](Self::emit) itself remains
    /// zero-allocation; only draining/reporting allocates.  Use
    /// [`drain_into`](Self::drain_into) to reuse a caller-owned buffer and avoid
    /// even that allocation.
    #[must_use]
    pub fn drain(&mut self) -> Vec<TraceEvent> {
        let mut out = Vec::with_capacity(self.events.len());
        self.events.drain_into(&mut out);
        out
    }

    /// Drains all recorded events into a caller-provided `Vec` (oldest first)
    /// and clears the buffer.
    ///
    /// Reusing the same `out` buffer across drains (e.g. `out.clear()` then
    /// `drain_into(&mut out)`) lets the reporting path avoid per-drain
    /// allocation once `out` has grown to its steady-state size.
    pub fn drain_into(&mut self, out: &mut Vec<TraceEvent>) {
        self.events.drain_into(out);
    }

    /// Returns `true` if no events are stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Returns the configured minimum recording level.
    #[must_use]
    pub fn min_level(&self) -> TraceLevel {
        self.min_level
    }

    /// Clears all stored events without resetting the session clock.
    pub fn clear(&mut self) {
        self.events.clear();
    }

    /// Returns a reference to the internal string table.
    #[must_use]
    pub fn string_table(&self) -> &StringTable {
        &self.string_table
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn trace_with_events() -> EventTrace {
        let mut t = EventTrace::verbose(64);
        t.start_session();
        t.emit(TraceLevel::Trace, "codec", "frame decoded");
        t.emit(TraceLevel::Info, "pipeline", "stage started");
        t.emit(TraceLevel::Warn, "codec", "pts discontinuity");
        t.emit(TraceLevel::Error, "io", "read timeout");
        t
    }

    #[test]
    fn test_new_empty() {
        let t = EventTrace::new(32, TraceLevel::Info);
        assert!(t.is_empty());
        assert_eq!(t.len(), 0);
    }

    #[test]
    fn test_emit_respects_min_level() {
        let mut t = EventTrace::new(32, TraceLevel::Warn);
        t.start_session();
        t.emit(TraceLevel::Trace, "c", "msg");
        t.emit(TraceLevel::Info, "c", "msg");
        assert_eq!(t.len(), 0);
        t.emit(TraceLevel::Warn, "c", "warning");
        assert_eq!(t.len(), 1);
    }

    #[test]
    fn test_capacity_evicts_oldest() {
        // Use a small custom capacity (but internally RING_CAP=4096;
        // we verify wrapping with 4097 events).
        let mut t = EventTrace::verbose(4096);
        t.start_session();
        // Push RING_CAP + 1 events — oldest should be overwritten.
        for i in 0u64..=(RING_CAP as u64) {
            t.emit(TraceLevel::Info, "c", &format!("event-{i}"));
        }
        assert_eq!(t.len(), RING_CAP);
        // The first event ("event-0") must no longer be in the buffer.
        let msgs: Vec<String> = t.events().iter().map(|e| e.message()).collect();
        assert!(
            !msgs.contains(&"event-0".to_string()),
            "event-0 should have been evicted; buffer: {:?}",
            &msgs[..4]
        );
    }

    #[test]
    fn test_filter_by_level() {
        let t = trace_with_events();
        let errors = t.filter_by_level(TraceLevel::Error);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].level(), TraceLevel::Error);
    }

    #[test]
    fn test_filter_by_level_warn_and_above() {
        let t = trace_with_events();
        let high = t.filter_by_level(TraceLevel::Warn);
        assert_eq!(high.len(), 2); // Warn + Error
    }

    #[test]
    fn test_filter_by_component() {
        let t = trace_with_events();
        let codec = t.filter_by_component("codec");
        assert_eq!(codec.len(), 2);
    }

    #[test]
    fn test_filter_by_time_all() {
        let t = trace_with_events();
        let all = t.filter_by_time(0, u64::MAX);
        assert_eq!(all.len(), 4);
    }

    #[test]
    fn test_filter_by_time_empty_range() {
        let t = trace_with_events();
        let none = t.filter_by_time(u64::MAX - 1, u64::MAX);
        assert!(none.is_empty());
    }

    #[test]
    fn test_session_has_session() {
        let mut t = EventTrace::verbose(32);
        assert!(!t.has_session());
        t.start_session();
        assert!(t.has_session());
    }

    #[test]
    fn test_session_duration_some_after_start() {
        let mut t = EventTrace::verbose(32);
        t.start_session();
        assert!(t.session_duration().is_some());
    }

    #[test]
    fn test_clear_resets_events() {
        let mut t = trace_with_events();
        t.clear();
        assert!(t.is_empty());
        assert!(t.has_session()); // session clock survives clear
    }

    #[test]
    fn test_trace_level_ordering() {
        assert!(TraceLevel::Critical > TraceLevel::Error);
        assert!(TraceLevel::Error > TraceLevel::Warn);
        assert!(TraceLevel::Warn > TraceLevel::Info);
        assert!(TraceLevel::Info > TraceLevel::Trace);
    }

    #[test]
    fn test_trace_level_is_at_least() {
        assert!(TraceLevel::Error.is_at_least(TraceLevel::Warn));
        assert!(!TraceLevel::Info.is_at_least(TraceLevel::Warn));
    }

    #[test]
    fn test_trace_level_labels() {
        assert_eq!(TraceLevel::Trace.label(), "TRACE");
        assert_eq!(TraceLevel::Critical.label(), "CRIT");
    }

    #[test]
    fn test_min_level_accessor() {
        let t = EventTrace::new(32, TraceLevel::Warn);
        assert_eq!(t.min_level(), TraceLevel::Warn);
    }

    // -----------------------------------------------------------------------
    // Sub-item 30 new tests
    // -----------------------------------------------------------------------

    /// Emit 100 events, verify ring wraps (oldest overwritten), all emitted
    /// values readable in order modulo ring size.
    #[test]
    fn test_event_trace_zero_alloc_smoke() {
        let mut t = EventTrace::verbose(4096);
        t.start_session();

        // Push a modest number of events well within ring capacity.
        for i in 0u64..100 {
            t.emit(TraceLevel::Info, "smoke", &format!("msg-{i}"));
        }
        assert_eq!(t.len(), 100, "100 events must be stored");

        // All 100 must be recoverable (no wrapping yet).
        let msgs: Vec<String> = t.events().iter().map(|e| e.message()).collect();
        for i in 0u64..100 {
            assert!(
                msgs.contains(&format!("msg-{i}")),
                "msg-{i} not found in buffer"
            );
        }

        // Now overflow the ring by RING_CAP + 50 more events.
        for i in 100u64..((RING_CAP as u64) + 50) {
            t.emit(TraceLevel::Info, "smoke", &format!("msg-{i}"));
        }
        // Buffer must be at capacity.
        assert_eq!(t.len(), RING_CAP);
        // The very first event ("msg-0") must have been evicted.
        let msgs2: Vec<String> = t.events().iter().map(|e| e.message()).collect();
        assert!(
            !msgs2.contains(&"msg-0".to_string()),
            "msg-0 should be evicted after overflow"
        );
    }

    /// Emit an event with a known component/message, read it back, verify
    /// fields are unchanged (Copy roundtrip).
    #[test]
    fn test_event_trace_copy_roundtrip() {
        let mut t = EventTrace::verbose(4096);
        t.start_session();
        t.emit(TraceLevel::Warn, "roundtrip", "hello world");

        assert_eq!(t.len(), 1);
        let ev = t.events().iter().next().expect("must have one event");
        assert_eq!(ev.level(), TraceLevel::Warn);
        assert_eq!(ev.message(), "hello world");
        // The component_id must be 0 (first interned string).
        assert_eq!(ev.component_id, 0);
        // Resolve back to the original name.
        assert_eq!(
            t.string_table().resolve(0),
            Some("roundtrip"),
            "string table round-trip failed"
        );
    }

    /// The reported capacity matches the compile-time `RING_CAP` and ignores
    /// the (legacy) `capacity` constructor argument.
    #[test]
    fn test_capacity_accessor_matches_ring_cap() {
        let t = EventTrace::new(7, TraceLevel::Info);
        assert_eq!(t.capacity(), RING_CAP);
    }

    /// Recording must never grow the backing store: capture `capacity()` before
    /// and after a burst that exceeds the ring capacity and assert it is
    /// unchanged, and that `len()` never exceeds `capacity()`.
    ///
    /// The message is a fixed `&str` and the component is pre-interned before
    /// the measured burst, so every `emit` in the loop walks the fully
    /// zero-allocation path (level check → byte copy → array write).
    #[test]
    fn test_capacity_unchanged_across_burst() {
        let mut t = EventTrace::verbose(RING_CAP);
        t.start_session();

        // Pre-warm the intern table so the burst itself never interns (and so
        // the HashMap never grows during measurement).
        t.emit(TraceLevel::Info, "burst", "warmup");

        let cap_before = t.capacity();
        assert_eq!(cap_before, RING_CAP);

        // Emit well beyond capacity with a constant message (no per-iteration
        // allocation anywhere in the loop body).
        for _ in 0..(RING_CAP * 2 + 17) {
            t.emit(TraceLevel::Info, "burst", "steady-state event");
            assert!(
                t.len() <= t.capacity(),
                "len ({}) must never exceed capacity ({})",
                t.len(),
                t.capacity()
            );
        }

        assert_eq!(
            t.capacity(),
            cap_before,
            "capacity must be invariant across a burst (no reallocation)"
        );
        assert_eq!(t.len(), RING_CAP, "buffer must be saturated, not grown");
        // Only one component was ever interned, regardless of event count.
        assert_eq!(t.string_table().resolve(0), Some("burst"));
    }

    /// `drain` returns events oldest-first and empties the buffer.
    #[test]
    fn test_drain_returns_events_in_order_and_clears() {
        let mut t = EventTrace::verbose(RING_CAP);
        t.start_session();
        t.emit(TraceLevel::Info, "drain", "first");
        t.emit(TraceLevel::Warn, "drain", "second");
        t.emit(TraceLevel::Error, "drain", "third");

        let drained = t.drain();
        let messages: Vec<String> = drained.iter().map(TraceEvent::message).collect();
        assert_eq!(messages, vec!["first", "second", "third"]);
        assert_eq!(drained[0].level(), TraceLevel::Info);
        assert_eq!(drained[2].level(), TraceLevel::Error);

        // Drain consumes: the buffer is now empty but the session clock and the
        // capacity survive.
        assert!(t.is_empty());
        assert_eq!(t.len(), 0);
        assert!(t.has_session());
        assert_eq!(t.capacity(), RING_CAP);
    }

    /// `drain` after wrap-around yields only the surviving (newest) events in
    /// insertion order.
    #[test]
    fn test_drain_after_wrap_yields_survivors() {
        let mut t = EventTrace::verbose(RING_CAP);
        t.start_session();
        for i in 0u64..((RING_CAP as u64) + 3) {
            t.emit(TraceLevel::Info, "wrap", &format!("e{i}"));
        }
        let drained = t.drain();
        assert_eq!(drained.len(), RING_CAP);
        // Oldest three (e0, e1, e2) were overwritten; the first survivor is e3.
        assert_eq!(drained[0].message(), "e3");
        assert_eq!(
            drained[RING_CAP - 1].message(),
            format!("e{}", RING_CAP + 2)
        );
        assert!(t.is_empty());
    }

    /// `drain_into` reuses a caller-owned buffer; once it reaches steady-state
    /// size, repeated drains do not reallocate it.
    #[test]
    fn test_drain_into_reuses_buffer() {
        let mut t = EventTrace::verbose(RING_CAP);
        t.start_session();

        let mut sink: Vec<TraceEvent> = Vec::with_capacity(8);
        let reserved = sink.capacity();

        for round in 0..3u64 {
            sink.clear();
            for i in 0..4u64 {
                t.emit(TraceLevel::Info, "reuse", &format!("r{round}-{i}"));
            }
            t.drain_into(&mut sink);
            assert_eq!(sink.len(), 4);
            assert_eq!(sink[0].message(), format!("r{round}-0"));
            assert!(t.is_empty(), "drain_into must clear the trace");
        }

        // The pre-sized sink never needed to grow.
        assert_eq!(sink.capacity(), reserved, "drain sink must not reallocate");
    }
}
