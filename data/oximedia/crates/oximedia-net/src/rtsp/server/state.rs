//! RTSP server configuration and per-session state.

use std::time::{Duration, Instant};

/// RTSP server configuration.
#[derive(Debug, Clone)]
pub struct RtspServerConfig {
    /// TCP address to bind (e.g. `"0.0.0.0:554"`).
    pub bind_address: String,
    /// How long a session lives without any request before it expires.
    pub session_timeout: Duration,
    /// Maximum number of concurrent connections accepted.
    pub max_connections: usize,
}

impl Default for RtspServerConfig {
    fn default() -> Self {
        Self {
            bind_address: "0.0.0.0:554".to_string(),
            session_timeout: Duration::from_secs(60),
            max_connections: 100,
        }
    }
}

/// Lifecycle states of an RTSP session (RFC 2326 §A.1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RtspSessionState {
    /// Session created but no media stream established yet.
    Init,
    /// SETUP has been issued; stream is configured but not streaming.
    Ready,
    /// PLAY has been issued; media is flowing.
    Playing,
    /// PAUSE has been issued; stream paused but session still alive.
    Paused,
}

/// An active RTSP session.
///
/// Tracks the session ID, state machine position, mounted path, channel
/// assignments, and the inactivity deadline.
pub struct RtspSession {
    /// Unique session identifier echoed in `Session:` headers.
    pub id: String,
    /// Current RFC 2326 state.
    pub state: RtspSessionState,
    /// The mount-point path this session is subscribed to.
    pub mount_path: String,
    /// RTP interleaved channel ID negotiated during SETUP.
    pub channel_id: u8,
    /// Wall-clock deadline; past this point the session is considered expired.
    pub expires_at: Instant,
    /// Session timeout duration (used to refresh the deadline).
    pub timeout: Duration,
}

impl RtspSession {
    /// Create a new session in the `Init` state.
    #[must_use]
    pub fn new(id: String, mount_path: String, channel_id: u8, timeout: Duration) -> Self {
        Self {
            expires_at: Instant::now() + timeout,
            id,
            state: RtspSessionState::Init,
            mount_path,
            channel_id,
            timeout,
        }
    }

    /// Push the expiry deadline forward by `timeout`.
    pub fn refresh(&mut self) {
        self.expires_at = Instant::now() + self.timeout;
    }

    /// True if the session has not received any request within the timeout window.
    #[must_use]
    pub fn is_expired(&self) -> bool {
        Instant::now() > self.expires_at
    }
}
