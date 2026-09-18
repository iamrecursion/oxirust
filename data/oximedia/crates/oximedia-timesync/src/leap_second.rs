//! Leap-second event modelling and lookup table.

#![allow(dead_code)]

/// The type of a leap-second insertion or deletion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeapSecondType {
    /// A positive leap second is inserted (23:59:60 occurs).
    Positive,
    /// A negative leap second is deleted (23:59:58 is immediately followed by 00:00:00).
    Negative,
}

impl LeapSecondType {
    /// Returns `+1` or `-1` to apply to a TAI-UTC offset.
    #[must_use]
    pub fn delta(&self) -> i32 {
        match self {
            Self::Positive => 1,
            Self::Negative => -1,
        }
    }
}

/// A single leap-second event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LeapSecondEvent {
    /// Unix timestamp (seconds since 1970-01-01 00:00:00 UTC) at which the leap occurs.
    pub unix_timestamp: i64,
    /// Type of leap second.
    pub kind: LeapSecondType,
    /// TAI-UTC difference **after** this leap second is applied.
    pub tai_utc_offset: i32,
}

impl LeapSecondEvent {
    /// Create a new event.
    #[must_use]
    pub fn new(unix_timestamp: i64, kind: LeapSecondType, tai_utc_offset: i32) -> Self {
        Self {
            unix_timestamp,
            kind,
            tai_utc_offset,
        }
    }

    /// `true` when the given Unix timestamp falls within one second of this leap second event.
    #[must_use]
    pub fn affects_timestamp(&self, ts: i64) -> bool {
        (ts - self.unix_timestamp).unsigned_abs() <= 1
    }
}

/// A lookup table of known historical and scheduled leap-second events.
#[derive(Debug, Default)]
pub struct LeapSecondTable {
    events: Vec<LeapSecondEvent>,
}

impl LeapSecondTable {
    /// Create an empty table.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a table pre-populated with a selection of historical leap seconds
    /// (TAI-UTC offsets based on IERS bulletins through 2016-12-31).
    #[must_use]
    pub fn with_historical() -> Self {
        let mut t = Self::new();
        // A representative subset; offsets are cumulative TAI-UTC.
        let entries: &[(i64, LeapSecondType, i32)] = &[
            (63072000, LeapSecondType::Positive, 10),   // 1972-07-01
            (78796800, LeapSecondType::Positive, 11),   // 1973-01-01
            (94694400, LeapSecondType::Positive, 12),   // 1974-01-01
            (1136073600, LeapSecondType::Positive, 33), // 2006-01-01
            (1341100800, LeapSecondType::Positive, 35), // 2012-07-01
            (1435708800, LeapSecondType::Positive, 36), // 2015-07-01
            (1483228800, LeapSecondType::Positive, 37), // 2017-01-01
        ];
        for &(ts, kind, offset) in entries {
            t.add_event(LeapSecondEvent::new(ts, kind, offset));
        }
        t
    }

    /// Add an event to the table, keeping it sorted by timestamp.
    pub fn add_event(&mut self, event: LeapSecondEvent) {
        let pos = self
            .events
            .binary_search_by_key(&event.unix_timestamp, |e| e.unix_timestamp)
            .unwrap_or_else(|i| i);
        self.events.insert(pos, event);
    }

    /// `true` when the given Unix timestamp coincides with a leap-second event
    /// (within ±1 second).
    #[must_use]
    pub fn is_leap_second_at(&self, ts: i64) -> bool {
        self.events.iter().any(|e| e.affects_timestamp(ts))
    }

    /// Return the next leap-second event strictly after the given Unix timestamp,
    /// or `None` if no future event is known.
    #[must_use]
    pub fn next_leap(&self, after_ts: i64) -> Option<&LeapSecondEvent> {
        self.events.iter().find(|e| e.unix_timestamp > after_ts)
    }

    /// Return the TAI-UTC offset valid at a given Unix timestamp.
    ///
    /// Returns `None` when the timestamp precedes all known events.
    #[must_use]
    pub fn tai_utc_at(&self, ts: i64) -> Option<i32> {
        // Find the last event that is ≤ ts.
        self.events
            .iter()
            .rev()
            .find(|e| e.unix_timestamp <= ts)
            .map(|e| e.tai_utc_offset)
    }

    /// Number of events in the table.
    #[must_use]
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// `true` when the table contains no events.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_leap_second_type_positive_delta() {
        assert_eq!(LeapSecondType::Positive.delta(), 1);
    }

    #[test]
    fn test_leap_second_type_negative_delta() {
        assert_eq!(LeapSecondType::Negative.delta(), -1);
    }

    #[test]
    fn test_event_affects_timestamp_exact() {
        let e = LeapSecondEvent::new(1_000_000, LeapSecondType::Positive, 37);
        assert!(e.affects_timestamp(1_000_000));
    }

    #[test]
    fn test_event_affects_timestamp_one_second_after() {
        let e = LeapSecondEvent::new(1_000_000, LeapSecondType::Positive, 37);
        assert!(e.affects_timestamp(1_000_001));
    }

    #[test]
    fn test_event_does_not_affect_distant_timestamp() {
        let e = LeapSecondEvent::new(1_000_000, LeapSecondType::Positive, 37);
        assert!(!e.affects_timestamp(2_000_000));
    }

    #[test]
    fn test_empty_table() {
        let t = LeapSecondTable::new();
        assert!(t.is_empty());
        assert_eq!(t.len(), 0);
    }

    #[test]
    fn test_add_event_increments_len() {
        let mut t = LeapSecondTable::new();
        t.add_event(LeapSecondEvent::new(100, LeapSecondType::Positive, 10));
        assert_eq!(t.len(), 1);
    }

    #[test]
    fn test_add_events_sorted() {
        let mut t = LeapSecondTable::new();
        t.add_event(LeapSecondEvent::new(300, LeapSecondType::Positive, 12));
        t.add_event(LeapSecondEvent::new(100, LeapSecondType::Positive, 10));
        t.add_event(LeapSecondEvent::new(200, LeapSecondType::Positive, 11));
        assert_eq!(t.events[0].unix_timestamp, 100);
        assert_eq!(t.events[1].unix_timestamp, 200);
        assert_eq!(t.events[2].unix_timestamp, 300);
    }

    #[test]
    fn test_is_leap_second_at_known_event() {
        let mut t = LeapSecondTable::new();
        t.add_event(LeapSecondEvent::new(500, LeapSecondType::Positive, 15));
        assert!(t.is_leap_second_at(500));
    }

    #[test]
    fn test_is_not_leap_second_at_unrelated_ts() {
        let mut t = LeapSecondTable::new();
        t.add_event(LeapSecondEvent::new(500, LeapSecondType::Positive, 15));
        assert!(!t.is_leap_second_at(10_000));
    }

    #[test]
    fn test_next_leap_returns_future_event() {
        let mut t = LeapSecondTable::new();
        t.add_event(LeapSecondEvent::new(100, LeapSecondType::Positive, 10));
        t.add_event(LeapSecondEvent::new(200, LeapSecondType::Positive, 11));
        let next = t.next_leap(150);
        assert!(next.is_some());
        assert_eq!(next.expect("should succeed in test").unix_timestamp, 200);
    }

    #[test]
    fn test_next_leap_none_when_past_all_events() {
        let mut t = LeapSecondTable::new();
        t.add_event(LeapSecondEvent::new(100, LeapSecondType::Positive, 10));
        assert!(t.next_leap(200).is_none());
    }

    #[test]
    fn test_tai_utc_at_before_all_events() {
        let t = LeapSecondTable::with_historical();
        assert!(t.tai_utc_at(0).is_none());
    }

    #[test]
    fn test_tai_utc_at_after_last_known_event() {
        let t = LeapSecondTable::with_historical();
        // 2020-01-01 should still show offset 37 (no new leap since 2017).
        let ts_2020: i64 = 1_577_836_800;
        assert_eq!(t.tai_utc_at(ts_2020), Some(37));
    }

    #[test]
    fn test_historical_table_populated() {
        let t = LeapSecondTable::with_historical();
        assert!(!t.is_empty());
    }
}
