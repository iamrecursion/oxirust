//! Session tracking and analysis.

use super::track::ViewEvent;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Viewing session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViewingSession {
    /// Session ID
    pub session_id: Uuid,
    /// User ID
    pub user_id: Uuid,
    /// Events in this session
    pub events: Vec<ViewEvent>,
    /// Session start time
    pub start_time: i64,
    /// Session end time
    pub end_time: i64,
    /// Session duration (seconds)
    pub duration_seconds: i64,
}

impl ViewingSession {
    /// Create a new viewing session
    #[must_use]
    pub fn new(session_id: Uuid, user_id: Uuid) -> Self {
        let now = chrono::Utc::now().timestamp();
        Self {
            session_id,
            user_id,
            events: Vec::new(),
            start_time: now,
            end_time: now,
            duration_seconds: 0,
        }
    }

    /// Add an event to the session
    pub fn add_event(&mut self, event: ViewEvent) {
        self.events.push(event);
        self.update_times();
    }

    /// Update session times
    fn update_times(&mut self) {
        if self.events.is_empty() {
            return;
        }

        self.start_time = self.events.first().map_or(self.start_time, |e| e.timestamp);
        self.end_time = self.events.last().map_or(self.end_time, |e| e.timestamp);
        self.duration_seconds = self.end_time - self.start_time;
    }

    /// Get total watch time in session
    #[must_use]
    pub fn total_watch_time(&self) -> i64 {
        self.events.iter().map(|e| e.watch_time_ms).sum()
    }

    /// Get number of items watched
    #[must_use]
    pub fn items_watched(&self) -> usize {
        let unique_content: std::collections::HashSet<Uuid> =
            self.events.iter().map(|e| e.content_id).collect();
        unique_content.len()
    }

    /// Get completion rate for session
    #[must_use]
    pub fn completion_rate(&self) -> f32 {
        if self.events.is_empty() {
            return 0.0;
        }

        let completed = self.events.iter().filter(|e| e.completed).count();
        completed as f32 / self.events.len() as f32
    }
}

/// Session tracker
pub struct SessionTracker {
    /// Active sessions
    active_sessions: std::collections::HashMap<Uuid, ViewingSession>,
    /// Completed sessions
    completed_sessions: Vec<ViewingSession>,
    /// Session timeout (seconds)
    session_timeout: i64,
}

impl SessionTracker {
    /// Create a new session tracker
    #[must_use]
    pub fn new(session_timeout: i64) -> Self {
        Self {
            active_sessions: std::collections::HashMap::new(),
            completed_sessions: Vec::new(),
            session_timeout,
        }
    }

    /// Add a view event to appropriate session
    pub fn add_event(&mut self, event: ViewEvent) {
        // Find or create session
        let session_id = event.session_id;

        let session = self
            .active_sessions
            .entry(session_id)
            .or_insert_with(|| ViewingSession::new(session_id, event.user_id));

        session.add_event(event);

        // Check for timeout
        self.check_timeouts();
    }

    /// Check for timed-out sessions
    fn check_timeouts(&mut self) {
        let now = chrono::Utc::now().timestamp();
        let timeout = self.session_timeout;

        let timed_out: Vec<Uuid> = self
            .active_sessions
            .iter()
            .filter(|(_, session)| now - session.end_time > timeout)
            .map(|(id, _)| *id)
            .collect();

        for session_id in timed_out {
            if let Some(session) = self.active_sessions.remove(&session_id) {
                self.completed_sessions.push(session);
            }
        }
    }

    /// Get active session for user
    #[must_use]
    pub fn get_active_session(&self, session_id: Uuid) -> Option<&ViewingSession> {
        self.active_sessions.get(&session_id)
    }

    /// Get user's completed sessions
    #[must_use]
    pub fn get_user_sessions(&self, user_id: Uuid) -> Vec<&ViewingSession> {
        self.completed_sessions
            .iter()
            .filter(|s| s.user_id == user_id)
            .collect()
    }

    /// Get session statistics
    #[must_use]
    pub fn get_session_stats(&self, user_id: Uuid) -> SessionStatistics {
        let sessions = self.get_user_sessions(user_id);

        if sessions.is_empty() {
            return SessionStatistics::default();
        }

        let total_sessions = sessions.len();
        let total_duration: i64 = sessions.iter().map(|s| s.duration_seconds).sum();
        let avg_duration = total_duration / total_sessions as i64;

        let total_items: usize = sessions.iter().map(|s| s.items_watched()).sum();
        let avg_items_per_session = total_items as f32 / total_sessions as f32;

        let total_watch_time: i64 = sessions.iter().map(|s| s.total_watch_time()).sum();
        let avg_watch_time = total_watch_time / total_sessions as i64;

        SessionStatistics {
            total_sessions,
            avg_duration_seconds: avg_duration,
            avg_items_per_session,
            avg_watch_time_ms: avg_watch_time,
        }
    }

    /// Close all active sessions
    pub fn close_all_sessions(&mut self) {
        let session_ids: Vec<Uuid> = self.active_sessions.keys().copied().collect();

        for session_id in session_ids {
            if let Some(session) = self.active_sessions.remove(&session_id) {
                self.completed_sessions.push(session);
            }
        }
    }
}

impl Default for SessionTracker {
    fn default() -> Self {
        Self::new(1800) // 30 minute default timeout
    }
}

/// Session statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionStatistics {
    /// Total number of sessions
    pub total_sessions: usize,
    /// Average session duration (seconds)
    pub avg_duration_seconds: i64,
    /// Average items per session
    pub avg_items_per_session: f32,
    /// Average watch time per session (milliseconds)
    pub avg_watch_time_ms: i64,
}

impl Default for SessionStatistics {
    fn default() -> Self {
        Self {
            total_sessions: 0,
            avg_duration_seconds: 0,
            avg_items_per_session: 0.0,
            avg_watch_time_ms: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_viewing_session_creation() {
        let session_id = Uuid::new_v4();
        let user_id = Uuid::new_v4();
        let session = ViewingSession::new(session_id, user_id);

        assert_eq!(session.session_id, session_id);
        assert_eq!(session.user_id, user_id);
        assert_eq!(session.events.len(), 0);
    }

    #[test]
    fn test_session_add_event() {
        let session_id = Uuid::new_v4();
        let user_id = Uuid::new_v4();
        let mut session = ViewingSession::new(session_id, user_id);

        let event = ViewEvent::new(user_id, Uuid::new_v4(), 60000, true);
        session.add_event(event);

        assert_eq!(session.events.len(), 1);
    }

    #[test]
    fn test_session_tracker() {
        let mut tracker = SessionTracker::new(1800);
        let session_id = Uuid::new_v4();
        let user_id = Uuid::new_v4();

        let mut event = ViewEvent::new(user_id, Uuid::new_v4(), 60000, true);
        event.session_id = session_id;

        tracker.add_event(event);

        assert!(tracker.get_active_session(session_id).is_some());
    }

    #[test]
    fn test_session_statistics() {
        let tracker = SessionTracker::new(1800);
        let user_id = Uuid::new_v4();
        let stats = tracker.get_session_stats(user_id);

        assert_eq!(stats.total_sessions, 0);
    }
}
