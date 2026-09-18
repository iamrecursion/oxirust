//! Real-time metadata tracking during playback.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Playback event for metadata tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PlaybackEvent {
    /// Item started playing.
    ItemStarted {
        /// Item ID.
        item_id: String,
        /// Start time.
        timestamp: DateTime<Utc>,
    },

    /// Item finished playing.
    ItemFinished {
        /// Item ID.
        item_id: String,
        /// End time.
        timestamp: DateTime<Utc>,
        /// Whether it completed successfully.
        success: bool,
    },

    /// Item paused.
    ItemPaused {
        /// Item ID.
        item_id: String,
        /// Pause time.
        timestamp: DateTime<Utc>,
    },

    /// Item resumed.
    ItemResumed {
        /// Item ID.
        item_id: String,
        /// Resume time.
        timestamp: DateTime<Utc>,
    },

    /// Error occurred.
    Error {
        /// Item ID.
        item_id: String,
        /// Error message.
        error: String,
        /// Error time.
        timestamp: DateTime<Utc>,
    },
}

/// Tracked item information.
#[derive(Debug, Clone)]
struct TrackedItem {
    item_id: String,
    start_time: Option<DateTime<Utc>>,
    end_time: Option<DateTime<Utc>>,
    pause_count: u32,
    error_count: u32,
    events: Vec<PlaybackEvent>,
}

/// Metadata tracker for real-time playback tracking.
pub struct MetadataTracker {
    tracked_items: Arc<RwLock<HashMap<String, TrackedItem>>>,
    event_history: Arc<RwLock<Vec<PlaybackEvent>>>,
    max_history: usize,
}

impl MetadataTracker {
    /// Creates a new metadata tracker.
    #[must_use]
    pub fn new() -> Self {
        Self::with_max_history(1000)
    }

    /// Creates a new metadata tracker with a maximum history size.
    #[must_use]
    pub fn with_max_history(max_history: usize) -> Self {
        Self {
            tracked_items: Arc::new(RwLock::new(HashMap::new())),
            event_history: Arc::new(RwLock::new(Vec::new())),
            max_history,
        }
    }

    /// Records a playback event.
    pub fn record_event(&self, event: PlaybackEvent) {
        // Update tracked item
        if let Ok(mut items) = self.tracked_items.write() {
            match &event {
                PlaybackEvent::ItemStarted { item_id, timestamp } => {
                    items.insert(
                        item_id.clone(),
                        TrackedItem {
                            item_id: item_id.clone(),
                            start_time: Some(*timestamp),
                            end_time: None,
                            pause_count: 0,
                            error_count: 0,
                            events: vec![event.clone()],
                        },
                    );
                }
                PlaybackEvent::ItemFinished {
                    item_id, timestamp, ..
                } => {
                    if let Some(item) = items.get_mut(item_id) {
                        item.end_time = Some(*timestamp);
                        item.events.push(event.clone());
                    }
                }
                PlaybackEvent::ItemPaused { item_id, .. } => {
                    if let Some(item) = items.get_mut(item_id) {
                        item.pause_count += 1;
                        item.events.push(event.clone());
                    }
                }
                PlaybackEvent::ItemResumed { item_id, .. } => {
                    if let Some(item) = items.get_mut(item_id) {
                        item.events.push(event.clone());
                    }
                }
                PlaybackEvent::Error { item_id, .. } => {
                    if let Some(item) = items.get_mut(item_id) {
                        item.error_count += 1;
                        item.events.push(event.clone());
                    }
                }
            }
        }

        // Add to history
        if let Ok(mut history) = self.event_history.write() {
            history.push(event);

            // Trim history if it exceeds max size
            if history.len() > self.max_history {
                history.remove(0);
            }
        }
    }

    /// Gets all events for a specific item.
    #[must_use]
    pub fn get_item_events(&self, item_id: &str) -> Vec<PlaybackEvent> {
        self.tracked_items
            .read()
            .ok()
            .and_then(|items| items.get(item_id).map(|item| item.events.clone()))
            .unwrap_or_default()
    }

    /// Gets the full event history.
    #[must_use]
    pub fn get_event_history(&self) -> Vec<PlaybackEvent> {
        self.event_history
            .read()
            .map(|h| h.clone())
            .unwrap_or_default()
    }

    /// Gets statistics for a specific item.
    #[must_use]
    pub fn get_item_stats(&self, item_id: &str) -> Option<ItemStats> {
        self.tracked_items.read().ok().and_then(|items| {
            items.get(item_id).map(|item| ItemStats {
                item_id: item.item_id.clone(),
                start_time: item.start_time,
                end_time: item.end_time,
                pause_count: item.pause_count,
                error_count: item.error_count,
                event_count: item.events.len(),
            })
        })
    }

    /// Clears all tracking data.
    pub fn clear(&self) {
        if let Ok(mut items) = self.tracked_items.write() {
            items.clear();
        }
        if let Ok(mut history) = self.event_history.write() {
            history.clear();
        }
    }

    /// Returns the number of tracked items.
    #[must_use]
    pub fn tracked_item_count(&self) -> usize {
        self.tracked_items
            .read()
            .map(|items| items.len())
            .unwrap_or(0)
    }

    /// Returns the event history count.
    #[must_use]
    pub fn event_count(&self) -> usize {
        self.event_history.read().map(|h| h.len()).unwrap_or(0)
    }
}

impl Default for MetadataTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for a tracked item.
#[derive(Debug, Clone)]
pub struct ItemStats {
    /// Item ID.
    pub item_id: String,
    /// Start time.
    pub start_time: Option<DateTime<Utc>>,
    /// End time.
    pub end_time: Option<DateTime<Utc>>,
    /// Number of pauses.
    pub pause_count: u32,
    /// Number of errors.
    pub error_count: u32,
    /// Total number of events.
    pub event_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metadata_tracker() {
        let tracker = MetadataTracker::new();
        let now = Utc::now();

        tracker.record_event(PlaybackEvent::ItemStarted {
            item_id: "item1".to_string(),
            timestamp: now,
        });

        tracker.record_event(PlaybackEvent::ItemPaused {
            item_id: "item1".to_string(),
            timestamp: now,
        });

        let events = tracker.get_item_events("item1");
        assert_eq!(events.len(), 2);

        let stats = tracker
            .get_item_stats("item1")
            .expect("should succeed in test");
        assert_eq!(stats.pause_count, 1);
        assert_eq!(stats.error_count, 0);
    }

    #[test]
    fn test_event_history() {
        let tracker = MetadataTracker::with_max_history(2);
        let now = Utc::now();

        tracker.record_event(PlaybackEvent::ItemStarted {
            item_id: "item1".to_string(),
            timestamp: now,
        });

        tracker.record_event(PlaybackEvent::ItemStarted {
            item_id: "item2".to_string(),
            timestamp: now,
        });

        tracker.record_event(PlaybackEvent::ItemStarted {
            item_id: "item3".to_string(),
            timestamp: now,
        });

        let history = tracker.get_event_history();
        // Should only keep last 2 events
        assert_eq!(history.len(), 2);
    }
}
