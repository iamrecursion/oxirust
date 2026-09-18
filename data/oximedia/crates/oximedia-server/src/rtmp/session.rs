//! RTMP session management.

use oximedia_net::rtmp::StreamMetadata;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use uuid::Uuid;

/// RTMP session state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionState {
    /// Connecting.
    Connecting,
    /// Handshake in progress.
    Handshake,
    /// Connected and ready.
    Connected,
    /// Publishing stream.
    Publishing,
    /// Playing stream.
    Playing,
    /// Disconnecting.
    Disconnecting,
    /// Disconnected.
    Disconnected,
}

/// RTMP ingest session.
#[derive(Debug, Clone)]
pub struct IngestSession {
    /// Session ID.
    pub id: Uuid,

    /// Client address.
    pub client_addr: std::net::SocketAddr,

    /// Application name.
    pub app_name: String,

    /// Stream key.
    pub stream_key: String,

    /// Session state.
    pub state: SessionState,

    /// Session metadata.
    pub metadata: Option<StreamMetadata>,

    /// Connection time.
    pub connected_at: Instant,

    /// Last activity time.
    pub last_activity: Instant,

    /// Total bytes sent.
    pub bytes_sent: u64,

    /// Total bytes received.
    pub bytes_received: u64,

    /// Total packets sent.
    pub packets_sent: u64,

    /// Total packets received.
    pub packets_received: u64,
}

impl IngestSession {
    /// Creates a new session.
    #[must_use]
    pub fn new(client_addr: std::net::SocketAddr) -> Self {
        let now = Instant::now();
        Self {
            id: Uuid::new_v4(),
            client_addr,
            app_name: String::new(),
            stream_key: String::new(),
            state: SessionState::Connecting,
            metadata: None,
            connected_at: now,
            last_activity: now,
            bytes_sent: 0,
            bytes_received: 0,
            packets_sent: 0,
            packets_received: 0,
        }
    }

    /// Updates the state.
    pub fn set_state(&mut self, state: SessionState) {
        self.state = state;
        self.last_activity = Instant::now();
    }

    /// Sets stream info.
    pub fn set_stream(&mut self, app_name: String, stream_key: String) {
        self.app_name = app_name;
        self.stream_key = stream_key;
        self.last_activity = Instant::now();
    }

    /// Records bytes sent.
    pub fn record_sent(&mut self, bytes: u64) {
        self.bytes_sent += bytes;
        self.packets_sent += 1;
        self.last_activity = Instant::now();
    }

    /// Records bytes received.
    pub fn record_received(&mut self, bytes: u64) {
        self.bytes_received += bytes;
        self.packets_received += 1;
        self.last_activity = Instant::now();
    }

    /// Gets session duration.
    #[must_use]
    pub fn duration(&self) -> Duration {
        self.last_activity.duration_since(self.connected_at)
    }

    /// Gets idle duration.
    #[must_use]
    pub fn idle_duration(&self) -> Duration {
        Instant::now().duration_since(self.last_activity)
    }

    /// Checks if session is active.
    #[must_use]
    pub fn is_active(&self) -> bool {
        matches!(
            self.state,
            SessionState::Connected | SessionState::Publishing | SessionState::Playing
        )
    }
}

/// Session manager.
pub struct SessionManager {
    /// Active sessions.
    sessions: RwLock<HashMap<Uuid, IngestSession>>,

    /// Session timeout.
    timeout: Duration,
}

impl SessionManager {
    /// Creates a new session manager.
    #[must_use]
    pub fn new(timeout: Duration) -> Self {
        Self {
            sessions: RwLock::new(HashMap::new()),
            timeout,
        }
    }

    /// Creates a new session.
    pub fn create_session(&self, client_addr: std::net::SocketAddr) -> Uuid {
        let session = IngestSession::new(client_addr);
        let id = session.id;

        let mut sessions = self.sessions.write();
        sessions.insert(id, session);

        id
    }

    /// Gets a session.
    #[must_use]
    pub fn get_session(&self, id: Uuid) -> Option<IngestSession> {
        let sessions = self.sessions.read();
        sessions.get(&id).cloned()
    }

    /// Updates a session.
    pub fn update_session<F>(&self, id: Uuid, f: F)
    where
        F: FnOnce(&mut IngestSession),
    {
        let mut sessions = self.sessions.write();
        if let Some(session) = sessions.get_mut(&id) {
            f(session);
        }
    }

    /// Removes a session.
    pub fn remove_session(&self, id: Uuid) {
        let mut sessions = self.sessions.write();
        sessions.remove(&id);
    }

    /// Lists all sessions.
    #[must_use]
    pub fn list_sessions(&self) -> Vec<IngestSession> {
        let sessions = self.sessions.read();
        sessions.values().cloned().collect()
    }

    /// Cleans up timed out sessions.
    pub fn cleanup_timeout_sessions(&self) -> usize {
        let mut sessions = self.sessions.write();
        let now = Instant::now();
        let timeout = self.timeout;

        let timed_out: Vec<Uuid> = sessions
            .iter()
            .filter(|(_, s)| now.duration_since(s.last_activity) > timeout)
            .map(|(id, _)| *id)
            .collect();

        let count = timed_out.len();
        for id in timed_out {
            sessions.remove(&id);
        }

        count
    }

    /// Gets session count.
    #[must_use]
    pub fn session_count(&self) -> usize {
        let sessions = self.sessions.read();
        sessions.len()
    }

    /// Gets active session count.
    #[must_use]
    pub fn active_session_count(&self) -> usize {
        let sessions = self.sessions.read();
        sessions.values().filter(|s| s.is_active()).count()
    }
}

impl Default for SessionManager {
    fn default() -> Self {
        Self::new(Duration::from_secs(30))
    }
}
