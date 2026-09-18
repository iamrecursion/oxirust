#![allow(dead_code)]
//! Game event timeline tracking for `OxiMedia`.
//!
//! Tracks discrete game events (kills, deaths, objectives, milestones) along a
//! wall-clock timeline, providing windowed queries and milestone filtering.

use std::time::{Duration, Instant};

/// Classification of a game event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GameEventType {
    /// Player scored a kill
    Kill,
    /// Player was eliminated
    Death,
    /// Match began
    MatchStart,
    /// Match ended
    MatchEnd,
    /// Team captured an objective
    ObjectiveCaptured,
    /// Player achieved a high-value multi-kill
    MultiKill,
    /// Player levelled up
    LevelUp,
    /// Achievement unlocked
    AchievementUnlocked,
    /// Custom application-defined event
    Custom,
}

impl GameEventType {
    /// Return `true` for events considered milestones worth highlighting.
    #[must_use]
    pub fn is_milestone(self) -> bool {
        matches!(
            self,
            GameEventType::MultiKill
                | GameEventType::AchievementUnlocked
                | GameEventType::ObjectiveCaptured
                | GameEventType::MatchEnd
        )
    }

    /// Human-readable label for the event type.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            GameEventType::Kill => "kill",
            GameEventType::Death => "death",
            GameEventType::MatchStart => "match_start",
            GameEventType::MatchEnd => "match_end",
            GameEventType::ObjectiveCaptured => "objective_captured",
            GameEventType::MultiKill => "multi_kill",
            GameEventType::LevelUp => "level_up",
            GameEventType::AchievementUnlocked => "achievement_unlocked",
            GameEventType::Custom => "custom",
        }
    }
}

/// A single timestamped game event.
#[derive(Debug, Clone)]
pub struct GameEvent {
    /// Kind of event.
    pub event_type: GameEventType,
    /// Wall-clock instant at which the event occurred.
    pub occurred_at: Instant,
    /// Optional free-form description.
    pub description: Option<String>,
    /// Arbitrary integer payload (score delta, kill streak count, etc.).
    pub value: i64,
}

impl GameEvent {
    /// Create a new event occurring right now.
    #[must_use]
    pub fn new(event_type: GameEventType, value: i64) -> Self {
        Self {
            event_type,
            occurred_at: Instant::now(),
            description: None,
            value,
        }
    }

    /// Create an event with an explicit instant.
    #[must_use]
    pub fn with_instant(event_type: GameEventType, value: i64, occurred_at: Instant) -> Self {
        Self {
            event_type,
            occurred_at,
            description: None,
            value,
        }
    }

    /// Attach a description.
    #[must_use]
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Age of the event in milliseconds relative to `now`.
    #[must_use]
    pub fn age_ms(&self, now: Instant) -> u64 {
        now.saturating_duration_since(self.occurred_at).as_millis() as u64
    }

    /// Return `true` if the event is a milestone.
    #[must_use]
    pub fn is_milestone(&self) -> bool {
        self.event_type.is_milestone()
    }
}

/// Ordered sequence of game events.
#[derive(Debug, Default)]
pub struct EventTimeline {
    events: Vec<GameEvent>,
}

impl EventTimeline {
    /// Create an empty timeline.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append an event to the timeline.
    pub fn add_event(&mut self, event: GameEvent) {
        self.events.push(event);
    }

    /// Total number of events recorded.
    #[must_use]
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Return `true` when no events have been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Return all events whose `occurred_at` falls within the half-open
    /// interval `[now - window, now)`.
    #[must_use]
    pub fn events_in_window(&self, now: Instant, window: Duration) -> Vec<&GameEvent> {
        let cutoff = now.checked_sub(window).unwrap_or(now);
        self.events
            .iter()
            .filter(|e| e.occurred_at >= cutoff && e.occurred_at < now)
            .collect()
    }

    /// Return only milestone events from the full timeline.
    #[must_use]
    pub fn milestone_events(&self) -> Vec<&GameEvent> {
        self.events.iter().filter(|e| e.is_milestone()).collect()
    }

    /// Return all events of a specific type.
    #[must_use]
    pub fn events_of_type(&self, event_type: GameEventType) -> Vec<&GameEvent> {
        self.events
            .iter()
            .filter(|e| e.event_type == event_type)
            .collect()
    }

    /// Sum of `value` across all events.
    #[must_use]
    pub fn total_value(&self) -> i64 {
        self.events.iter().map(|e| e.value).sum()
    }

    /// Clear all recorded events.
    pub fn clear(&mut self) {
        self.events.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    fn now() -> Instant {
        Instant::now()
    }

    #[test]
    fn test_event_type_milestone_classification() {
        assert!(GameEventType::MultiKill.is_milestone());
        assert!(GameEventType::AchievementUnlocked.is_milestone());
        assert!(GameEventType::ObjectiveCaptured.is_milestone());
        assert!(GameEventType::MatchEnd.is_milestone());

        assert!(!GameEventType::Kill.is_milestone());
        assert!(!GameEventType::Death.is_milestone());
        assert!(!GameEventType::MatchStart.is_milestone());
        assert!(!GameEventType::LevelUp.is_milestone());
        assert!(!GameEventType::Custom.is_milestone());
    }

    #[test]
    fn test_event_type_labels() {
        assert_eq!(GameEventType::Kill.label(), "kill");
        assert_eq!(GameEventType::Death.label(), "death");
        assert_eq!(GameEventType::MultiKill.label(), "multi_kill");
        assert_eq!(GameEventType::MatchStart.label(), "match_start");
        assert_eq!(GameEventType::MatchEnd.label(), "match_end");
    }

    #[test]
    fn test_game_event_new() {
        let before = Instant::now();
        let event = GameEvent::new(GameEventType::Kill, 10);
        let after = Instant::now();

        assert_eq!(event.event_type, GameEventType::Kill);
        assert_eq!(event.value, 10);
        assert!(event.occurred_at >= before && event.occurred_at <= after);
    }

    #[test]
    fn test_game_event_age_ms() {
        let base = Instant::now();
        let event = GameEvent::with_instant(GameEventType::Kill, 1, base);
        let later = base + Duration::from_millis(500);
        let age = event.age_ms(later);
        assert_eq!(age, 500);
    }

    #[test]
    fn test_game_event_age_ms_zero_for_future() {
        // If `now` is before `occurred_at`, saturating subtraction yields 0.
        let base = Instant::now();
        let future = base + Duration::from_millis(100);
        let event = GameEvent::with_instant(GameEventType::Kill, 1, future);
        assert_eq!(event.age_ms(base), 0);
    }

    #[test]
    fn test_game_event_is_milestone() {
        let milestone = GameEvent::new(GameEventType::MultiKill, 5);
        let ordinary = GameEvent::new(GameEventType::Kill, 1);
        assert!(milestone.is_milestone());
        assert!(!ordinary.is_milestone());
    }

    #[test]
    fn test_game_event_with_description() {
        let event = GameEvent::new(GameEventType::Custom, 0).with_description("boss defeated");
        assert_eq!(event.description.as_deref(), Some("boss defeated"));
    }

    #[test]
    fn test_timeline_add_and_len() {
        let mut tl = EventTimeline::new();
        assert!(tl.is_empty());
        tl.add_event(GameEvent::new(GameEventType::Kill, 1));
        tl.add_event(GameEvent::new(GameEventType::Death, 0));
        assert_eq!(tl.len(), 2);
        assert!(!tl.is_empty());
    }

    #[test]
    fn test_timeline_events_in_window() {
        let base = Instant::now();
        let mut tl = EventTimeline::new();

        // 2 seconds ago — inside a 5-second window
        tl.add_event(GameEvent::with_instant(
            GameEventType::Kill,
            1,
            base - Duration::from_secs(2),
        ));
        // 10 seconds ago — outside the window
        tl.add_event(GameEvent::with_instant(
            GameEventType::Death,
            0,
            base - Duration::from_secs(10),
        ));

        let now = base + Duration::from_millis(1); // slightly after base
        let window = Duration::from_secs(5);
        let results = tl.events_in_window(now, window);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].event_type, GameEventType::Kill);
    }

    #[test]
    fn test_timeline_milestone_events() {
        let mut tl = EventTimeline::new();
        tl.add_event(GameEvent::new(GameEventType::Kill, 1));
        tl.add_event(GameEvent::new(GameEventType::MultiKill, 3));
        tl.add_event(GameEvent::new(GameEventType::AchievementUnlocked, 0));
        tl.add_event(GameEvent::new(GameEventType::Death, 0));

        let milestones = tl.milestone_events();
        assert_eq!(milestones.len(), 2);
    }

    #[test]
    fn test_timeline_total_value() {
        let mut tl = EventTimeline::new();
        tl.add_event(GameEvent::new(GameEventType::Kill, 10));
        tl.add_event(GameEvent::new(GameEventType::Kill, 20));
        tl.add_event(GameEvent::new(GameEventType::Death, -5));
        assert_eq!(tl.total_value(), 25);
    }

    #[test]
    fn test_timeline_events_of_type() {
        let mut tl = EventTimeline::new();
        tl.add_event(GameEvent::new(GameEventType::Kill, 1));
        tl.add_event(GameEvent::new(GameEventType::Kill, 2));
        tl.add_event(GameEvent::new(GameEventType::Death, 0));

        let kills = tl.events_of_type(GameEventType::Kill);
        assert_eq!(kills.len(), 2);
    }

    #[test]
    fn test_timeline_clear() {
        let mut tl = EventTimeline::new();
        tl.add_event(GameEvent::new(GameEventType::Kill, 1));
        tl.clear();
        assert!(tl.is_empty());
    }

    #[test]
    fn test_with_instant_constructor() {
        let t = now() - Duration::from_secs(1);
        let event = GameEvent::with_instant(GameEventType::LevelUp, 99, t);
        assert_eq!(event.event_type, GameEventType::LevelUp);
        assert_eq!(event.value, 99);
        assert_eq!(event.occurred_at, t);
    }
}
