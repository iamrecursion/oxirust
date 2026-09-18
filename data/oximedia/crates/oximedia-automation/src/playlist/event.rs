//! Playlist event handling.

use serde::{Deserialize, Serialize};
use std::time::SystemTime;

/// Playlist event type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventType {
    /// Item started playing
    ItemStart,
    /// Item finished playing
    ItemEnd,
    /// Pre-roll started
    PrerollStart,
    /// Pre-roll completed
    PrerollEnd,
    /// Playlist completed
    PlaylistComplete,
    /// Error occurred
    Error,
}

/// Playlist event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlaylistEvent {
    /// Event type
    pub event_type: EventType,
    /// Associated item ID
    pub item_id: String,
    /// Event timestamp
    pub timestamp: SystemTime,
}

impl PlaylistEvent {
    /// Create a new playlist event.
    pub fn new(event_type: EventType, item_id: String) -> Self {
        Self {
            event_type,
            item_id,
            timestamp: SystemTime::now(),
        }
    }

    /// Create item start event.
    pub fn item_start(item_id: String) -> Self {
        Self::new(EventType::ItemStart, item_id)
    }

    /// Create item end event.
    pub fn item_end(item_id: String) -> Self {
        Self::new(EventType::ItemEnd, item_id)
    }

    /// Create pre-roll start event.
    pub fn preroll_start(item_id: String) -> Self {
        Self::new(EventType::PrerollStart, item_id)
    }

    /// Create pre-roll end event.
    pub fn preroll_end(item_id: String) -> Self {
        Self::new(EventType::PrerollEnd, item_id)
    }

    /// Create playlist complete event.
    pub fn playlist_complete() -> Self {
        Self::new(EventType::PlaylistComplete, String::new())
    }

    /// Create error event.
    pub fn error(item_id: String) -> Self {
        Self::new(EventType::Error, item_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_creation() {
        let event = PlaylistEvent::item_start("test_item".to_string());
        assert_eq!(event.event_type, EventType::ItemStart);
        assert_eq!(event.item_id, "test_item");
    }

    #[test]
    fn test_event_types() {
        let start = PlaylistEvent::item_start("test".to_string());
        let end = PlaylistEvent::item_end("test".to_string());

        assert_eq!(start.event_type, EventType::ItemStart);
        assert_eq!(end.event_type, EventType::ItemEnd);
        assert_ne!(start.event_type, end.event_type);
    }
}
