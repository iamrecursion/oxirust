#![allow(dead_code)]
//! Network session lifecycle tracking for streaming connections.
//!
//! Provides [`SessionTracker`] that maintains a registry of active streaming
//! sessions, assigns unique identifiers, and tracks per-session metrics such as
//! bytes transferred, duration, and state transitions.

use std::collections::HashMap;
use std::fmt;

/// Unique identifier for a tracked session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SessionId(u64);

impl SessionId {
    /// Creates a session ID from a raw integer.
    pub const fn from_raw(id: u64) -> Self {
        Self(id)
    }

    /// Returns the raw integer value.
    pub const fn raw(self) -> u64 {
        self.0
    }
}

impl fmt::Display for SessionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "session-{}", self.0)
    }
}

/// Lifecycle state of a streaming session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SessionState {
    /// Connection established but no media flowing yet.
    Connecting,
    /// Actively streaming media.
    Active,
    /// Temporarily paused (e.g. buffering).
    Paused,
    /// Reconnecting after transient failure.
    Reconnecting,
    /// Gracefully closed.
    Closed,
    /// Terminated due to an error.
    Failed,
}

impl fmt::Display for SessionState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::Connecting => "Connecting",
            Self::Active => "Active",
            Self::Paused => "Paused",
            Self::Reconnecting => "Reconnecting",
            Self::Closed => "Closed",
            Self::Failed => "Failed",
        };
        f.write_str(label)
    }
}

impl SessionState {
    /// Returns `true` if the session is still alive (not closed or failed).
    pub fn is_alive(self) -> bool {
        !matches!(self, Self::Closed | Self::Failed)
    }
}

/// Per-session statistics.
#[derive(Debug, Clone, PartialEq)]
pub struct SessionStats {
    /// Total bytes sent by this session.
    pub bytes_sent: u64,
    /// Total bytes received by this session.
    pub bytes_received: u64,
    /// Duration the session has been alive, in milliseconds.
    pub duration_ms: u64,
    /// Number of state transitions since creation.
    pub transition_count: u32,
}

impl SessionStats {
    /// Returns total bytes transferred (sent + received).
    pub const fn total_bytes(&self) -> u64 {
        self.bytes_sent + self.bytes_received
    }

    /// Returns the effective throughput in kilobits per second.
    #[allow(clippy::cast_precision_loss)]
    pub fn throughput_kbps(&self) -> f64 {
        if self.duration_ms == 0 {
            return 0.0;
        }
        (self.total_bytes() as f64 * 8.0) / (self.duration_ms as f64)
    }
}

/// Internal record for a tracked session.
#[derive(Debug, Clone)]
struct SessionRecord {
    /// Unique session identifier.
    id: SessionId,
    /// Current state.
    state: SessionState,
    /// Descriptive label (e.g. URI or peer address).
    label: String,
    /// Cumulative stats.
    stats: SessionStats,
}

/// Registry that tracks active and recently-closed sessions.
#[derive(Debug)]
pub struct SessionTracker {
    sessions: HashMap<SessionId, SessionRecord>,
    next_id: u64,
}

impl SessionTracker {
    /// Creates a new, empty tracker.
    pub fn new() -> Self {
        Self {
            sessions: HashMap::new(),
            next_id: 1,
        }
    }

    /// Opens a new session with the given label and returns its [`SessionId`].
    pub fn open(&mut self, label: impl Into<String>) -> SessionId {
        let id = SessionId(self.next_id);
        self.next_id += 1;
        let record = SessionRecord {
            id,
            state: SessionState::Connecting,
            label: label.into(),
            stats: SessionStats {
                bytes_sent: 0,
                bytes_received: 0,
                duration_ms: 0,
                transition_count: 0,
            },
        };
        self.sessions.insert(id, record);
        id
    }

    /// Returns the current state of a session, or `None` if the ID is unknown.
    pub fn state(&self, id: SessionId) -> Option<SessionState> {
        self.sessions.get(&id).map(|r| r.state)
    }

    /// Transitions a session to a new state.
    ///
    /// Returns `true` if the transition was applied, `false` if the session
    /// was not found.
    pub fn transition(&mut self, id: SessionId, new_state: SessionState) -> bool {
        if let Some(rec) = self.sessions.get_mut(&id) {
            rec.state = new_state;
            rec.stats.transition_count += 1;
            true
        } else {
            false
        }
    }

    /// Records bytes sent on a session.
    pub fn record_sent(&mut self, id: SessionId, bytes: u64) {
        if let Some(rec) = self.sessions.get_mut(&id) {
            rec.stats.bytes_sent += bytes;
        }
    }

    /// Records bytes received on a session.
    pub fn record_received(&mut self, id: SessionId, bytes: u64) {
        if let Some(rec) = self.sessions.get_mut(&id) {
            rec.stats.bytes_received += bytes;
        }
    }

    /// Updates the duration field for a session.
    pub fn update_duration(&mut self, id: SessionId, duration_ms: u64) {
        if let Some(rec) = self.sessions.get_mut(&id) {
            rec.stats.duration_ms = duration_ms;
        }
    }

    /// Returns a snapshot of the statistics for a session.
    pub fn stats(&self, id: SessionId) -> Option<SessionStats> {
        self.sessions.get(&id).map(|r| r.stats.clone())
    }

    /// Returns the label for a session.
    pub fn label(&self, id: SessionId) -> Option<&str> {
        self.sessions.get(&id).map(|r| r.label.as_str())
    }

    /// Returns the total number of tracked sessions (all states).
    pub fn total_sessions(&self) -> usize {
        self.sessions.len()
    }

    /// Returns the number of sessions that are currently alive.
    pub fn active_count(&self) -> usize {
        self.sessions
            .values()
            .filter(|r| r.state.is_alive())
            .count()
    }

    /// Removes all closed/failed sessions from the registry and returns how
    /// many were purged.
    pub fn purge_dead(&mut self) -> usize {
        let before = self.sessions.len();
        self.sessions.retain(|_, r| r.state.is_alive());
        before - self.sessions.len()
    }

    /// Returns all session IDs currently tracked.
    pub fn session_ids(&self) -> Vec<SessionId> {
        self.sessions.keys().copied().collect()
    }
}

impl Default for SessionTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Unit tests ───────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    // 1. new tracker is empty
    #[test]
    fn test_new_tracker_empty() {
        let t = SessionTracker::new();
        assert_eq!(t.total_sessions(), 0);
        assert_eq!(t.active_count(), 0);
    }

    // 2. open creates session in Connecting state
    #[test]
    fn test_open_creates_connecting() {
        let mut t = SessionTracker::new();
        let id = t.open("rtmp://live");
        assert_eq!(t.state(id), Some(SessionState::Connecting));
    }

    // 3. sequential IDs
    #[test]
    fn test_sequential_ids() {
        let mut t = SessionTracker::new();
        let a = t.open("a");
        let b = t.open("b");
        assert_eq!(b.raw() - a.raw(), 1);
    }

    // 4. transition updates state
    #[test]
    fn test_transition() {
        let mut t = SessionTracker::new();
        let id = t.open("test");
        assert!(t.transition(id, SessionState::Active));
        assert_eq!(t.state(id), Some(SessionState::Active));
    }

    // 5. transition unknown id returns false
    #[test]
    fn test_transition_unknown() {
        let mut t = SessionTracker::new();
        assert!(!t.transition(SessionId::from_raw(999), SessionState::Active));
    }

    // 6. record_sent accumulates
    #[test]
    fn test_record_sent() {
        let mut t = SessionTracker::new();
        let id = t.open("test");
        t.record_sent(id, 1000);
        t.record_sent(id, 2000);
        let s = t.stats(id).expect("should succeed in test");
        assert_eq!(s.bytes_sent, 3000);
    }

    // 7. record_received accumulates
    #[test]
    fn test_record_received() {
        let mut t = SessionTracker::new();
        let id = t.open("test");
        t.record_received(id, 500);
        let s = t.stats(id).expect("should succeed in test");
        assert_eq!(s.bytes_received, 500);
    }

    // 8. total_bytes combines sent + received
    #[test]
    fn test_total_bytes() {
        let s = SessionStats {
            bytes_sent: 100,
            bytes_received: 200,
            duration_ms: 1000,
            transition_count: 0,
        };
        assert_eq!(s.total_bytes(), 300);
    }

    // 9. throughput_kbps calculation
    #[test]
    fn test_throughput_kbps() {
        let s = SessionStats {
            bytes_sent: 1000,
            bytes_received: 0,
            duration_ms: 1000,
            transition_count: 0,
        };
        assert!((s.throughput_kbps() - 8.0).abs() < 1e-9);
    }

    // 10. throughput zero duration
    #[test]
    fn test_throughput_zero_duration() {
        let s = SessionStats {
            bytes_sent: 1000,
            bytes_received: 0,
            duration_ms: 0,
            transition_count: 0,
        };
        assert_eq!(s.throughput_kbps(), 0.0);
    }

    // 11. active_count filters dead
    #[test]
    fn test_active_count() {
        let mut t = SessionTracker::new();
        let a = t.open("a");
        let _b = t.open("b");
        t.transition(a, SessionState::Closed);
        assert_eq!(t.active_count(), 1);
    }

    // 12. purge_dead removes closed/failed
    #[test]
    fn test_purge_dead() {
        let mut t = SessionTracker::new();
        let a = t.open("a");
        let _b = t.open("b");
        t.transition(a, SessionState::Failed);
        let purged = t.purge_dead();
        assert_eq!(purged, 1);
        assert_eq!(t.total_sessions(), 1);
    }

    // 13. label retrieval
    #[test]
    fn test_label() {
        let mut t = SessionTracker::new();
        let id = t.open("srt://host:9000");
        assert_eq!(t.label(id), Some("srt://host:9000"));
    }

    // 14. SessionId display
    #[test]
    fn test_session_id_display() {
        let id = SessionId::from_raw(42);
        assert_eq!(format!("{id}"), "session-42");
    }

    // 15. SessionState display
    #[test]
    fn test_session_state_display() {
        assert_eq!(format!("{}", SessionState::Active), "Active");
        assert_eq!(format!("{}", SessionState::Reconnecting), "Reconnecting");
    }

    // 16. SessionState is_alive
    #[test]
    fn test_session_state_is_alive() {
        assert!(SessionState::Active.is_alive());
        assert!(SessionState::Paused.is_alive());
        assert!(!SessionState::Closed.is_alive());
        assert!(!SessionState::Failed.is_alive());
    }
}
