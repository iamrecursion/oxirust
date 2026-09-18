//! Transcode session tracking with lifecycle management.
//!
//! Provides `SessionState`, `TranscodeSession`, and `TranscodeSessionManager`
//! for monitoring active and completed transcode operations.

#![allow(dead_code)]

use std::collections::HashMap;

/// Lifecycle state of a transcode session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionState {
    /// Session has been created but not yet started.
    Pending,
    /// Transcoding is currently in progress.
    Running,
    /// Transcoding finished successfully.
    Completed,
    /// Transcoding failed with an error.
    Failed,
    /// Session was cancelled by the user.
    Cancelled,
}

impl SessionState {
    /// Returns `true` if the session is currently active (Running).
    #[must_use]
    pub fn is_active(self) -> bool {
        self == SessionState::Running
    }

    /// Returns `true` if the session has reached a terminal state.
    #[must_use]
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            SessionState::Completed | SessionState::Failed | SessionState::Cancelled
        )
    }

    /// Returns a short label for display purposes.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            SessionState::Pending => "pending",
            SessionState::Running => "running",
            SessionState::Completed => "completed",
            SessionState::Failed => "failed",
            SessionState::Cancelled => "cancelled",
        }
    }
}

/// A transcode session representing one input-to-output operation.
#[derive(Debug, Clone)]
pub struct TranscodeSession {
    /// Unique session identifier.
    pub id: u64,
    /// Input file path.
    pub input_path: String,
    /// Output file path.
    pub output_path: String,
    /// Current state of the session.
    pub state: SessionState,
    /// Start time in milliseconds since epoch (0 if not started).
    pub start_ms: u64,
    /// End time in milliseconds since epoch (0 if not finished).
    pub end_ms: u64,
    /// Progress as a value in `[0.0, 1.0]`.
    progress: f64,
    /// Total duration of the input in milliseconds.
    pub total_duration_ms: u64,
}

impl TranscodeSession {
    /// Creates a new session in the `Pending` state.
    pub fn new(
        id: u64,
        input_path: impl Into<String>,
        output_path: impl Into<String>,
        total_duration_ms: u64,
    ) -> Self {
        Self {
            id,
            input_path: input_path.into(),
            output_path: output_path.into(),
            state: SessionState::Pending,
            start_ms: 0,
            end_ms: 0,
            progress: 0.0,
            total_duration_ms,
        }
    }

    /// Marks the session as started at the given timestamp.
    pub fn start(&mut self, now_ms: u64) {
        self.state = SessionState::Running;
        self.start_ms = now_ms;
    }

    /// Updates the progress (clamped to [0.0, 1.0]).
    pub fn set_progress(&mut self, pct: f64) {
        self.progress = pct.clamp(0.0, 1.0);
    }

    /// Marks the session as completed at the given timestamp.
    pub fn complete(&mut self, now_ms: u64) {
        self.state = SessionState::Completed;
        self.end_ms = now_ms;
        self.progress = 1.0;
    }

    /// Marks the session as failed at the given timestamp.
    pub fn fail(&mut self, now_ms: u64) {
        self.state = SessionState::Failed;
        self.end_ms = now_ms;
    }

    /// Marks the session as cancelled at the given timestamp.
    pub fn cancel(&mut self, now_ms: u64) {
        self.state = SessionState::Cancelled;
        self.end_ms = now_ms;
    }

    /// Returns elapsed time in milliseconds (0 if not started, uses `end_ms` if terminal).
    #[must_use]
    pub fn elapsed_ms(&self, now_ms: u64) -> u64 {
        if self.start_ms == 0 {
            return 0;
        }
        let end = if self.state.is_terminal() {
            self.end_ms
        } else {
            now_ms
        };
        end.saturating_sub(self.start_ms)
    }

    /// Returns progress as a percentage (0–100).
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn progress_pct(&self) -> f64 {
        self.progress * 100.0
    }

    /// Returns `true` if the session is currently running.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.state.is_active()
    }
}

/// Manages a collection of transcode sessions.
#[derive(Debug, Default)]
pub struct TranscodeSessionManager {
    sessions: HashMap<u64, TranscodeSession>,
    next_id: u64,
}

impl TranscodeSessionManager {
    /// Creates a new, empty manager.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new session and registers it. Returns the session ID.
    pub fn create(
        &mut self,
        input_path: impl Into<String>,
        output_path: impl Into<String>,
        total_duration_ms: u64,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        let session = TranscodeSession::new(id, input_path, output_path, total_duration_ms);
        self.sessions.insert(id, session);
        id
    }

    /// Returns a reference to the session with the given ID, if it exists.
    #[must_use]
    pub fn get(&self, id: u64) -> Option<&TranscodeSession> {
        self.sessions.get(&id)
    }

    /// Returns a mutable reference to the session with the given ID.
    pub fn get_mut(&mut self, id: u64) -> Option<&mut TranscodeSession> {
        self.sessions.get_mut(&id)
    }

    /// Returns the number of currently active (Running) sessions.
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.sessions.values().filter(|s| s.is_active()).count()
    }

    /// Returns the total number of sessions tracked.
    #[must_use]
    pub fn total_count(&self) -> usize {
        self.sessions.len()
    }

    /// Removes a session by ID. Returns `true` if it existed.
    pub fn remove(&mut self, id: u64) -> bool {
        self.sessions.remove(&id).is_some()
    }

    /// Returns IDs of all sessions in a given state.
    #[must_use]
    pub fn sessions_in_state(&self, state: SessionState) -> Vec<u64> {
        self.sessions
            .values()
            .filter(|s| s.state == state)
            .map(|s| s.id)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_is_active_only_running() {
        assert!(SessionState::Running.is_active());
        assert!(!SessionState::Pending.is_active());
        assert!(!SessionState::Completed.is_active());
        assert!(!SessionState::Failed.is_active());
        assert!(!SessionState::Cancelled.is_active());
    }

    #[test]
    fn test_state_is_terminal() {
        assert!(SessionState::Completed.is_terminal());
        assert!(SessionState::Failed.is_terminal());
        assert!(SessionState::Cancelled.is_terminal());
        assert!(!SessionState::Pending.is_terminal());
        assert!(!SessionState::Running.is_terminal());
    }

    #[test]
    fn test_state_labels() {
        assert_eq!(SessionState::Running.label(), "running");
        assert_eq!(SessionState::Pending.label(), "pending");
        assert_eq!(SessionState::Completed.label(), "completed");
    }

    #[test]
    fn test_session_initial_state_pending() {
        let s = TranscodeSession::new(0, "in.mp4", "out.mp4", 60_000);
        assert_eq!(s.state, SessionState::Pending);
        assert_eq!(s.elapsed_ms(1000), 0);
    }

    #[test]
    fn test_session_start_sets_running() {
        let mut s = TranscodeSession::new(0, "in", "out", 60_000);
        s.start(1000);
        assert!(s.is_active());
        assert_eq!(s.state, SessionState::Running);
    }

    #[test]
    fn test_session_elapsed_ms_while_running() {
        let mut s = TranscodeSession::new(0, "in", "out", 60_000);
        s.start(1000);
        assert_eq!(s.elapsed_ms(4000), 3000);
    }

    #[test]
    fn test_session_elapsed_ms_after_complete() {
        let mut s = TranscodeSession::new(0, "in", "out", 60_000);
        s.start(1000);
        s.complete(5000);
        // Should use end_ms regardless of now_ms
        assert_eq!(s.elapsed_ms(9999), 4000);
    }

    #[test]
    fn test_session_progress_pct_clamped() {
        let mut s = TranscodeSession::new(0, "in", "out", 60_000);
        s.set_progress(1.5);
        assert!((s.progress_pct() - 100.0).abs() < f64::EPSILON);
        s.set_progress(-0.5);
        assert!((s.progress_pct()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_session_complete_sets_progress_full() {
        let mut s = TranscodeSession::new(0, "in", "out", 60_000);
        s.start(0);
        s.complete(1000);
        assert!((s.progress_pct() - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_session_fail() {
        let mut s = TranscodeSession::new(0, "in", "out", 60_000);
        s.start(0);
        s.fail(500);
        assert_eq!(s.state, SessionState::Failed);
        assert!(s.state.is_terminal());
    }

    #[test]
    fn test_manager_create_and_get() {
        let mut mgr = TranscodeSessionManager::new();
        let id = mgr.create("in.mp4", "out.mp4", 60_000);
        let s = mgr.get(id).expect("should succeed in test");
        assert_eq!(s.id, id);
        assert_eq!(s.state, SessionState::Pending);
    }

    #[test]
    fn test_manager_active_count() {
        let mut mgr = TranscodeSessionManager::new();
        let id1 = mgr.create("a", "b", 1000);
        let id2 = mgr.create("c", "d", 1000);
        mgr.get_mut(id1).expect("should succeed in test").start(0);
        assert_eq!(mgr.active_count(), 1);
        mgr.get_mut(id2).expect("should succeed in test").start(0);
        assert_eq!(mgr.active_count(), 2);
    }

    #[test]
    fn test_manager_remove() {
        let mut mgr = TranscodeSessionManager::new();
        let id = mgr.create("in", "out", 1000);
        assert!(mgr.remove(id));
        assert!(!mgr.remove(id));
        assert!(mgr.get(id).is_none());
    }

    #[test]
    fn test_manager_sessions_in_state() {
        let mut mgr = TranscodeSessionManager::new();
        let id = mgr.create("in", "out", 1000);
        mgr.get_mut(id).expect("should succeed in test").start(0);
        mgr.get_mut(id)
            .expect("should succeed in test")
            .complete(100);
        let completed = mgr.sessions_in_state(SessionState::Completed);
        assert!(completed.contains(&id));
    }
}
