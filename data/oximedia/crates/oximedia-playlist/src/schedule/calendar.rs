//! Calendar-based scheduling.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// A calendar event for scheduling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalendarEvent {
    /// Unique identifier.
    pub id: String,

    /// Event title.
    pub title: String,

    /// Start time.
    pub start_time: DateTime<Utc>,

    /// End time.
    pub end_time: DateTime<Utc>,

    /// Playlist ID to play.
    pub playlist_id: String,

    /// Event description.
    pub description: Option<String>,

    /// Event tags.
    pub tags: Vec<String>,

    /// Whether this event is enabled.
    pub enabled: bool,
}

impl CalendarEvent {
    /// Creates a new calendar event.
    #[must_use]
    pub fn new<S: Into<String>>(
        title: S,
        playlist_id: S,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
    ) -> Self {
        Self {
            id: generate_id(),
            title: title.into(),
            start_time,
            end_time,
            playlist_id: playlist_id.into(),
            description: None,
            tags: Vec::new(),
            enabled: true,
        }
    }

    /// Checks if this event is active at the given time.
    #[must_use]
    pub fn is_active_at(&self, time: &DateTime<Utc>) -> bool {
        self.enabled && time >= &self.start_time && time < &self.end_time
    }

    /// Returns the duration of this event.
    #[must_use]
    pub fn duration(&self) -> chrono::Duration {
        self.end_time - self.start_time
    }
}

/// Calendar-based schedule.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CalendarSchedule {
    /// Events in the calendar.
    pub events: Vec<CalendarEvent>,
}

impl CalendarSchedule {
    /// Creates a new calendar schedule.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds an event to the calendar.
    pub fn add_event(&mut self, event: CalendarEvent) {
        self.events.push(event);
        self.sort_events();
    }

    /// Removes an event by ID.
    pub fn remove_event(&mut self, event_id: &str) {
        self.events.retain(|e| e.id != event_id);
    }

    /// Gets events active at a specific time.
    #[must_use]
    pub fn get_active_events(&self, time: &DateTime<Utc>) -> Vec<&CalendarEvent> {
        self.events
            .iter()
            .filter(|e| e.is_active_at(time))
            .collect()
    }

    /// Gets events in a time range.
    #[must_use]
    pub fn get_events_in_range(
        &self,
        start: &DateTime<Utc>,
        end: &DateTime<Utc>,
    ) -> Vec<&CalendarEvent> {
        self.events
            .iter()
            .filter(|e| e.enabled && e.start_time < *end && e.end_time > *start)
            .collect()
    }

    /// Sorts events by start time.
    fn sort_events(&mut self) {
        self.events.sort_by(|a, b| a.start_time.cmp(&b.start_time));
    }

    /// Returns the number of events.
    #[must_use]
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Returns true if there are no events.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}

fn generate_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("event_{timestamp}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    #[test]
    fn test_calendar_event() {
        let now = Utc::now();
        let later = now + Duration::hours(1);
        let event = CalendarEvent::new("Test Event", "playlist_1", now, later);

        assert_eq!(event.title, "Test Event");
        assert!(event.is_active_at(&now));
        assert!(!event.is_active_at(&later));
    }

    #[test]
    fn test_calendar_schedule() {
        let mut schedule = CalendarSchedule::new();
        let now = Utc::now();
        let later = now + Duration::hours(1);

        let event = CalendarEvent::new("Test", "playlist_1", now, later);
        schedule.add_event(event);

        assert_eq!(schedule.len(), 1);
        assert_eq!(schedule.get_active_events(&now).len(), 1);
    }
}
