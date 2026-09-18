//! View history tracking.

use crate::error::RecommendResult;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// View event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViewEvent {
    /// User ID
    pub user_id: Uuid,
    /// Content ID
    pub content_id: Uuid,
    /// Session ID
    pub session_id: Uuid,
    /// Watch time (milliseconds)
    pub watch_time_ms: i64,
    /// Completed
    pub completed: bool,
    /// Timestamp
    pub timestamp: i64,
    /// Device type
    pub device: Option<String>,
    /// Playback quality
    pub quality: Option<String>,
}

impl ViewEvent {
    /// Create a new view event
    #[must_use]
    pub fn new(user_id: Uuid, content_id: Uuid, watch_time_ms: i64, completed: bool) -> Self {
        Self {
            user_id,
            content_id,
            session_id: Uuid::new_v4(),
            watch_time_ms,
            completed,
            timestamp: chrono::Utc::now().timestamp(),
            device: None,
            quality: None,
        }
    }

    /// Calculate completion rate (0-1)
    #[must_use]
    pub fn completion_rate(&self, total_duration_ms: i64) -> f32 {
        if total_duration_ms == 0 {
            return 0.0;
        }
        (self.watch_time_ms as f32 / total_duration_ms as f32).min(1.0)
    }
}

/// History tracker
pub struct HistoryTracker {
    /// View events by user
    user_history: HashMap<Uuid, Vec<ViewEvent>>,
    /// View events by content
    content_views: HashMap<Uuid, Vec<ViewEvent>>,
    /// Maximum history size per user
    max_history_size: usize,
}

impl HistoryTracker {
    /// Create a new history tracker
    #[must_use]
    pub fn new() -> Self {
        Self {
            user_history: HashMap::new(),
            content_views: HashMap::new(),
            max_history_size: 1000,
        }
    }

    /// Set maximum history size
    pub fn set_max_history_size(&mut self, size: usize) {
        self.max_history_size = size;
    }

    /// Record a view event
    ///
    /// # Errors
    ///
    /// Returns an error if recording fails
    pub fn record_view(
        &mut self,
        user_id: Uuid,
        content_id: Uuid,
        watch_time_ms: i64,
        completed: bool,
    ) -> RecommendResult<()> {
        let event = ViewEvent::new(user_id, content_id, watch_time_ms, completed);

        // Add to user history
        let user_events = self.user_history.entry(user_id).or_default();
        user_events.push(event.clone());

        // Trim if exceeds max size
        if user_events.len() > self.max_history_size {
            user_events.drain(0..user_events.len() - self.max_history_size);
        }

        // Add to content views
        self.content_views
            .entry(content_id)
            .or_default()
            .push(event);

        Ok(())
    }

    /// Get user's view history
    #[must_use]
    pub fn get_user_history(&self, user_id: Uuid) -> Vec<ViewEvent> {
        self.user_history.get(&user_id).cloned().unwrap_or_default()
    }

    /// Get content's view history
    #[must_use]
    pub fn get_content_views(&self, content_id: Uuid) -> Vec<ViewEvent> {
        self.content_views
            .get(&content_id)
            .cloned()
            .unwrap_or_default()
    }

    /// Get user's recently viewed content
    #[must_use]
    pub fn get_recently_viewed(&self, user_id: Uuid, limit: usize) -> Vec<Uuid> {
        let events = self.get_user_history(user_id);
        let mut content_ids: Vec<Uuid> = events.iter().rev().map(|e| e.content_id).collect();
        content_ids.dedup();
        content_ids.truncate(limit);
        content_ids
    }

    /// Get user's completed content
    #[must_use]
    pub fn get_completed_content(&self, user_id: Uuid) -> Vec<Uuid> {
        self.get_user_history(user_id)
            .into_iter()
            .filter(|e| e.completed)
            .map(|e| e.content_id)
            .collect()
    }

    /// Get user's watch time for content
    #[must_use]
    pub fn get_total_watch_time(&self, user_id: Uuid, content_id: Uuid) -> i64 {
        self.get_user_history(user_id)
            .into_iter()
            .filter(|e| e.content_id == content_id)
            .map(|e| e.watch_time_ms)
            .sum()
    }

    /// Check if user has viewed content
    #[must_use]
    pub fn has_viewed(&self, user_id: Uuid, content_id: Uuid) -> bool {
        self.get_user_history(user_id)
            .iter()
            .any(|e| e.content_id == content_id)
    }

    /// Get view count for content
    #[must_use]
    pub fn get_view_count(&self, content_id: Uuid) -> usize {
        self.content_views.get(&content_id).map_or(0, Vec::len)
    }

    /// Get unique viewer count for content
    #[must_use]
    pub fn get_unique_viewers(&self, content_id: Uuid) -> usize {
        let events = self.content_views.get(&content_id);
        let Some(events) = events else {
            return 0;
        };

        let unique_users: std::collections::HashSet<Uuid> =
            events.iter().map(|e| e.user_id).collect();
        unique_users.len()
    }

    /// Get average watch time for content
    #[must_use]
    pub fn get_avg_watch_time(&self, content_id: Uuid) -> i64 {
        let events = self.content_views.get(&content_id);
        let Some(events) = events else {
            return 0;
        };

        if events.is_empty() {
            return 0;
        }

        let total: i64 = events.iter().map(|e| e.watch_time_ms).sum();
        total / events.len() as i64
    }

    /// Get completion rate for content
    #[must_use]
    pub fn get_completion_rate(&self, content_id: Uuid) -> f32 {
        let events = self.content_views.get(&content_id);
        let Some(events) = events else {
            return 0.0;
        };

        if events.is_empty() {
            return 0.0;
        }

        let completed = events.iter().filter(|e| e.completed).count();
        completed as f32 / events.len() as f32
    }
}

impl Default for HistoryTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// View statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViewStatistics {
    /// Total views
    pub total_views: usize,
    /// Unique viewers
    pub unique_viewers: usize,
    /// Average watch time (milliseconds)
    pub avg_watch_time_ms: i64,
    /// Completion rate
    pub completion_rate: f32,
    /// Total watch time (milliseconds)
    pub total_watch_time_ms: i64,
}

impl ViewStatistics {
    /// Calculate statistics from view events
    #[must_use]
    pub fn from_events(events: &[ViewEvent]) -> Self {
        if events.is_empty() {
            return Self::default();
        }

        let total_views = events.len();
        let unique_viewers: std::collections::HashSet<Uuid> =
            events.iter().map(|e| e.user_id).collect();
        let total_watch_time_ms: i64 = events.iter().map(|e| e.watch_time_ms).sum();
        let avg_watch_time_ms = total_watch_time_ms / total_views as i64;
        let completed = events.iter().filter(|e| e.completed).count();
        let completion_rate = completed as f32 / total_views as f32;

        Self {
            total_views,
            unique_viewers: unique_viewers.len(),
            avg_watch_time_ms,
            completion_rate,
            total_watch_time_ms,
        }
    }
}

impl Default for ViewStatistics {
    fn default() -> Self {
        Self {
            total_views: 0,
            unique_viewers: 0,
            avg_watch_time_ms: 0,
            completion_rate: 0.0,
            total_watch_time_ms: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_view_event_creation() {
        let user_id = Uuid::new_v4();
        let content_id = Uuid::new_v4();
        let event = ViewEvent::new(user_id, content_id, 60000, true);

        assert_eq!(event.user_id, user_id);
        assert_eq!(event.content_id, content_id);
        assert!(event.completed);
    }

    #[test]
    fn test_completion_rate() {
        let event = ViewEvent::new(Uuid::new_v4(), Uuid::new_v4(), 30000, false);
        let rate = event.completion_rate(60000);
        assert!((rate - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_history_tracker() {
        let mut tracker = HistoryTracker::new();
        let user_id = Uuid::new_v4();
        let content_id = Uuid::new_v4();

        tracker
            .record_view(user_id, content_id, 60000, true)
            .expect("should succeed in test");

        assert!(tracker.has_viewed(user_id, content_id));
        assert_eq!(tracker.get_view_count(content_id), 1);
    }

    #[test]
    fn test_recently_viewed() {
        let mut tracker = HistoryTracker::new();
        let user_id = Uuid::new_v4();
        let content1 = Uuid::new_v4();
        let content2 = Uuid::new_v4();

        tracker
            .record_view(user_id, content1, 60000, true)
            .expect("should succeed in test");
        tracker
            .record_view(user_id, content2, 60000, false)
            .expect("should succeed in test");

        let recent = tracker.get_recently_viewed(user_id, 10);
        assert_eq!(recent.len(), 2);
        assert_eq!(recent[0], content2); // Most recent first
    }

    #[test]
    fn test_view_statistics() {
        let user1 = Uuid::new_v4();
        let user2 = Uuid::new_v4();
        let content = Uuid::new_v4();

        let events = vec![
            ViewEvent::new(user1, content, 60000, true),
            ViewEvent::new(user2, content, 30000, false),
            ViewEvent::new(user1, content, 60000, true),
        ];

        let stats = ViewStatistics::from_events(&events);
        assert_eq!(stats.total_views, 3);
        assert_eq!(stats.unique_viewers, 2);
    }
}
