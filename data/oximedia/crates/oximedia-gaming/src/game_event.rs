//! Game event logging and querying for streaming sessions.
//!
//! Captures structured events (kills, deaths, objectives, etc.) that occur
//! during a game and provides time-windowed query methods for highlight
//! detection and stream overlay display.

#![allow(dead_code)]

use std::time::{Duration, Instant};

/// Taxonomy of in-game events that can be logged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GameEventType {
    /// A player elimination.
    Kill,
    /// The local player was eliminated.
    Death,
    /// An assist on an elimination.
    Assist,
    /// An objective was captured or completed.
    ObjectiveCaptured,
    /// A multi-kill streak (value carries the streak count).
    MultiKill(u8),
    /// Match started.
    MatchStart,
    /// Match ended.
    MatchEnd,
    /// An in-game achievement was unlocked.
    Achievement,
    /// Level-up or rank promotion.
    LevelUp,
    /// Damage milestone crossed.
    DamageMilestone,
}

impl GameEventType {
    /// Human-readable label for the event type.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Kill => "Kill",
            Self::Death => "Death",
            Self::Assist => "Assist",
            Self::ObjectiveCaptured => "Objective Captured",
            Self::MultiKill(_) => "Multi-Kill",
            Self::MatchStart => "Match Start",
            Self::MatchEnd => "Match End",
            Self::Achievement => "Achievement",
            Self::LevelUp => "Level Up",
            Self::DamageMilestone => "Damage Milestone",
        }
    }

    /// Whether this event is typically considered a highlight moment.
    #[must_use]
    pub fn is_highlight(self) -> bool {
        matches!(
            self,
            Self::MultiKill(_) | Self::Achievement | Self::LevelUp | Self::ObjectiveCaptured
        )
    }
}

/// A single captured game event.
#[derive(Debug, Clone)]
pub struct GameEvent {
    /// The type of event.
    pub event_type: GameEventType,
    /// Wall-clock timestamp when the event was recorded.
    pub timestamp: Instant,
    /// Optional description or player name associated with the event.
    pub description: Option<String>,
    /// Optional numeric value (e.g. damage dealt, score delta).
    pub value: Option<i64>,
}

impl GameEvent {
    /// Create a new event with the current timestamp.
    #[must_use]
    pub fn new(event_type: GameEventType) -> Self {
        Self {
            event_type,
            timestamp: Instant::now(),
            description: None,
            value: None,
        }
    }

    /// Attach a human-readable description.
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Attach a numeric value to the event.
    #[must_use]
    pub fn with_value(mut self, value: i64) -> Self {
        self.value = Some(value);
        self
    }

    /// Whether this event qualifies as a highlight.
    #[must_use]
    pub fn is_highlight(&self) -> bool {
        self.event_type.is_highlight()
    }
}

/// An append-only, queryable log of [`GameEvent`]s.
pub struct GameEventLog {
    events: Vec<GameEvent>,
    /// Maximum number of events stored before the oldest are evicted.
    capacity: usize,
}

impl GameEventLog {
    /// Create a new event log with the given maximum capacity.
    ///
    /// # Panics
    ///
    /// Panics if `capacity` is zero.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "capacity must be at least 1");
        Self {
            events: Vec::with_capacity(capacity.min(1024)),
            capacity,
        }
    }

    /// Append an event to the log, evicting the oldest entry when full.
    pub fn push(&mut self, event: GameEvent) {
        if self.events.len() == self.capacity {
            self.events.remove(0);
        }
        self.events.push(event);
    }

    /// Total number of events currently stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Whether the log contains no events.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Return references to the `n` most-recently added events.
    /// Returns fewer than `n` events if the log is smaller.
    #[must_use]
    pub fn recent(&self, n: usize) -> &[GameEvent] {
        let start = self.events.len().saturating_sub(n);
        &self.events[start..]
    }

    /// All events recorded within the last `window` duration.
    #[must_use]
    pub fn within_window(&self, window: Duration) -> Vec<&GameEvent> {
        let now = Instant::now();
        self.events
            .iter()
            .filter(|e| now.duration_since(e.timestamp) <= window)
            .collect()
    }

    /// All highlight events in the log.
    #[must_use]
    pub fn highlights(&self) -> Vec<&GameEvent> {
        self.events.iter().filter(|e| e.is_highlight()).collect()
    }

    /// All events matching a given type (including `MultiKill` regardless of count).
    #[must_use]
    pub fn by_type(&self, event_type: GameEventType) -> Vec<&GameEvent> {
        self.events
            .iter()
            .filter(|e| {
                // For MultiKill, match any streak count.
                matches!(
                    (e.event_type, event_type),
                    (GameEventType::MultiKill(_), GameEventType::MultiKill(_))
                ) || e.event_type == event_type
            })
            .collect()
    }

    /// Count events of a particular type.
    #[must_use]
    pub fn count(&self, event_type: GameEventType) -> usize {
        self.by_type(event_type).len()
    }

    /// Clear all events.
    pub fn clear(&mut self) {
        self.events.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- GameEventType ---

    #[test]
    fn test_event_type_labels() {
        assert_eq!(GameEventType::Kill.label(), "Kill");
        assert_eq!(GameEventType::Death.label(), "Death");
        assert_eq!(GameEventType::MultiKill(3).label(), "Multi-Kill");
        assert_eq!(GameEventType::MatchStart.label(), "Match Start");
    }

    #[test]
    fn test_is_highlight() {
        assert!(GameEventType::MultiKill(2).is_highlight());
        assert!(GameEventType::Achievement.is_highlight());
        assert!(GameEventType::LevelUp.is_highlight());
        assert!(GameEventType::ObjectiveCaptured.is_highlight());
        assert!(!GameEventType::Kill.is_highlight());
        assert!(!GameEventType::Death.is_highlight());
    }

    // --- GameEvent ---

    #[test]
    fn test_event_construction() {
        let e = GameEvent::new(GameEventType::Kill)
            .with_description("Headshot")
            .with_value(100);
        assert_eq!(e.event_type, GameEventType::Kill);
        assert_eq!(e.description.as_deref(), Some("Headshot"));
        assert_eq!(e.value, Some(100));
    }

    #[test]
    fn test_event_is_highlight_delegates() {
        let e = GameEvent::new(GameEventType::Achievement);
        assert!(e.is_highlight());
        let e2 = GameEvent::new(GameEventType::Death);
        assert!(!e2.is_highlight());
    }

    // --- GameEventLog ---

    #[test]
    fn test_log_empty_initially() {
        let log = GameEventLog::new(100);
        assert!(log.is_empty());
        assert_eq!(log.len(), 0);
    }

    #[test]
    fn test_log_push_and_len() {
        let mut log = GameEventLog::new(10);
        log.push(GameEvent::new(GameEventType::Kill));
        log.push(GameEvent::new(GameEventType::Death));
        assert_eq!(log.len(), 2);
    }

    #[test]
    fn test_log_eviction_when_full() {
        let mut log = GameEventLog::new(3);
        log.push(GameEvent::new(GameEventType::MatchStart));
        log.push(GameEvent::new(GameEventType::Kill));
        log.push(GameEvent::new(GameEventType::Kill));
        // Adding a fourth should evict the first (MatchStart).
        log.push(GameEvent::new(GameEventType::Death));
        assert_eq!(log.len(), 3);
        // The oldest should now be Kill, not MatchStart.
        assert_eq!(log.recent(3)[0].event_type, GameEventType::Kill);
    }

    #[test]
    fn test_recent_fewer_than_n() {
        let mut log = GameEventLog::new(10);
        log.push(GameEvent::new(GameEventType::Kill));
        let r = log.recent(5);
        assert_eq!(r.len(), 1);
    }

    #[test]
    fn test_recent_exactly_n() {
        let mut log = GameEventLog::new(10);
        for _ in 0..4 {
            log.push(GameEvent::new(GameEventType::Assist));
        }
        assert_eq!(log.recent(4).len(), 4);
    }

    #[test]
    fn test_highlights() {
        let mut log = GameEventLog::new(20);
        log.push(GameEvent::new(GameEventType::Kill));
        log.push(GameEvent::new(GameEventType::MultiKill(3)));
        log.push(GameEvent::new(GameEventType::Achievement));
        log.push(GameEvent::new(GameEventType::Death));
        let hl = log.highlights();
        assert_eq!(hl.len(), 2);
    }

    #[test]
    fn test_count_by_type() {
        let mut log = GameEventLog::new(20);
        log.push(GameEvent::new(GameEventType::Kill));
        log.push(GameEvent::new(GameEventType::Kill));
        log.push(GameEvent::new(GameEventType::Death));
        assert_eq!(log.count(GameEventType::Kill), 2);
        assert_eq!(log.count(GameEventType::Death), 1);
        assert_eq!(log.count(GameEventType::Assist), 0);
    }

    #[test]
    fn test_clear() {
        let mut log = GameEventLog::new(10);
        log.push(GameEvent::new(GameEventType::Kill));
        log.clear();
        assert!(log.is_empty());
    }

    #[test]
    fn test_within_window_includes_recent() {
        let mut log = GameEventLog::new(20);
        log.push(GameEvent::new(GameEventType::Kill));
        let in_window = log.within_window(Duration::from_secs(5));
        assert_eq!(in_window.len(), 1);
    }
}
