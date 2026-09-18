//! NDI connection state tracking and lifecycle events.
#![allow(dead_code)]

use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// ReconnectPolicy — exponential backoff parameters
// ---------------------------------------------------------------------------

/// Policy controlling automatic reconnection behaviour after a disconnect.
///
/// The delay for attempt `n` (0-indexed) is:
///
/// ```text
/// delay = min(base_delay_ms * 2^n, max_delay_ms)
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReconnectPolicy {
    /// Maximum number of reconnection attempts before giving up.
    /// `0` means reconnection is disabled.
    pub max_attempts: u32,
    /// Base delay in milliseconds for the first reconnect attempt.
    pub base_delay_ms: u64,
    /// Maximum delay cap in milliseconds.
    pub max_delay_ms: u64,
}

impl ReconnectPolicy {
    /// Create a new reconnect policy.
    pub fn new(max_attempts: u32, base_delay_ms: u64, max_delay_ms: u64) -> Self {
        Self {
            max_attempts,
            base_delay_ms,
            max_delay_ms,
        }
    }

    /// Compute the delay (in milliseconds) for reconnect attempt `n` (0-indexed).
    ///
    /// Returns `min(base_delay_ms * 2^n, max_delay_ms)`.  Uses saturating
    /// arithmetic to avoid overflow on large `n`.
    pub fn delay_for_attempt(&self, n: u32) -> u64 {
        // 2^n saturates at u64::MAX when n >= 64; saturating_mul protects us.
        let shift = n.min(63);
        let multiplier: u64 = 1u64 << shift;
        let delay = self.base_delay_ms.saturating_mul(multiplier);
        delay.min(self.max_delay_ms)
    }

    /// Whether a reconnect attempt `n` (0-indexed) should still be made.
    pub fn should_attempt(&self, n: u32) -> bool {
        self.max_attempts > 0 && n < self.max_attempts
    }
}

impl Default for ReconnectPolicy {
    fn default() -> Self {
        // 5 attempts, starting at 250 ms, capped at 30 s.
        Self::new(5, 250, 30_000)
    }
}

/// Events that can occur during the lifecycle of an NDI connection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConnectionEvent {
    /// The TCP/UDP connection was successfully established.
    Connected,
    /// Connection was cleanly closed by the remote peer.
    Disconnected,
    /// A network or protocol error occurred.
    Error(String),
    /// Connection attempt timed out.
    Timeout,
    /// The connection was intentionally paused (e.g. bandwidth throttle).
    Paused,
    /// A previously paused connection has resumed.
    Resumed,
    /// Authentication succeeded.
    Authenticated,
    /// Authentication failed.
    AuthFailed(String),
}

impl ConnectionEvent {
    /// Returns `true` if this event represents an error condition.
    pub fn is_error(&self) -> bool {
        matches!(
            self,
            ConnectionEvent::Error(_) | ConnectionEvent::Timeout | ConnectionEvent::AuthFailed(_)
        )
    }

    /// A short string identifier for logging.
    pub fn kind(&self) -> &'static str {
        match self {
            ConnectionEvent::Connected => "connected",
            ConnectionEvent::Disconnected => "disconnected",
            ConnectionEvent::Error(_) => "error",
            ConnectionEvent::Timeout => "timeout",
            ConnectionEvent::Paused => "paused",
            ConnectionEvent::Resumed => "resumed",
            ConnectionEvent::Authenticated => "authenticated",
            ConnectionEvent::AuthFailed(_) => "auth_failed",
        }
    }
}

/// High-level states an NDI connection can be in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NdiConnectionState {
    /// Not yet attempted.
    Idle,
    /// Connection is being established.
    Connecting,
    /// Fully connected and streaming.
    Streaming,
    /// Connection is live but momentarily paused.
    Paused,
    /// A recoverable error occurred; reconnect will be attempted.
    Recovering,
    /// Actively attempting to reconnect; carries the attempt number (0-indexed).
    Reconnecting(u32),
    /// Terminal failure — no further reconnect attempts.
    Failed,
    /// The connection was gracefully closed.
    Closed,
}

impl NdiConnectionState {
    /// Returns `true` for states that represent normal, active operation.
    pub fn is_healthy(&self) -> bool {
        matches!(
            self,
            NdiConnectionState::Streaming | NdiConnectionState::Paused
        )
    }

    /// Returns `true` for terminal states (no further state transitions expected).
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            NdiConnectionState::Failed | NdiConnectionState::Closed
        )
    }

    /// Returns `true` if the state represents an in-progress reconnect attempt.
    pub fn is_reconnecting(&self) -> bool {
        matches!(self, NdiConnectionState::Reconnecting(_))
    }

    /// If reconnecting, returns the current attempt number; otherwise `None`.
    pub fn reconnect_attempt(&self) -> Option<u32> {
        match self {
            NdiConnectionState::Reconnecting(n) => Some(*n),
            _ => None,
        }
    }
}

/// An entry in the connection state history.
#[derive(Debug, Clone)]
struct StateEntry {
    state: NdiConnectionState,
    entered_at: Instant,
}

/// Tracks state transitions for an NDI connection, recording durations.
#[derive(Debug)]
pub struct ConnectionStateTracker {
    current: StateEntry,
    history: Vec<StateEntry>,
    /// Maximum history entries to retain.
    max_history: usize,
}

impl ConnectionStateTracker {
    /// Create a new tracker starting in the `Idle` state.
    pub fn new() -> Self {
        Self {
            current: StateEntry {
                state: NdiConnectionState::Idle,
                entered_at: Instant::now(),
            },
            history: Vec::new(),
            max_history: 64,
        }
    }

    /// Create a tracker with a custom history capacity.
    pub fn with_max_history(max_history: usize) -> Self {
        Self {
            current: StateEntry {
                state: NdiConnectionState::Idle,
                entered_at: Instant::now(),
            },
            history: Vec::new(),
            max_history,
        }
    }

    /// Returns the current connection state.
    pub fn state(&self) -> NdiConnectionState {
        self.current.state
    }

    /// Transition to a new state. Returns the previous state.
    pub fn transition(&mut self, new_state: NdiConnectionState) -> NdiConnectionState {
        let old = self.current.state;
        let old_entry = StateEntry {
            state: old,
            entered_at: self.current.entered_at,
        };
        // Push old entry to history, trimming if needed.
        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push(old_entry);
        self.current = StateEntry {
            state: new_state,
            entered_at: Instant::now(),
        };
        old
    }

    /// Milliseconds spent in the current state.
    pub fn state_duration_ms(&self) -> u64 {
        self.current.entered_at.elapsed().as_millis() as u64
    }

    /// Duration spent in the current state.
    pub fn state_duration(&self) -> Duration {
        self.current.entered_at.elapsed()
    }

    /// Apply a `ConnectionEvent` and automatically transition to the appropriate state.
    ///
    /// When a `Disconnected` or error event arrives while a `ReconnectPolicy`
    /// is active (see `ConnectionStateMachine`), the state machine enters
    /// `Reconnecting(0)` instead of `Closed`.
    pub fn apply_event(&mut self, event: &ConnectionEvent) {
        let next = match event {
            ConnectionEvent::Connected | ConnectionEvent::Authenticated => {
                NdiConnectionState::Connecting
            }
            ConnectionEvent::Disconnected => NdiConnectionState::Closed,
            ConnectionEvent::Error(_) => NdiConnectionState::Recovering,
            ConnectionEvent::Timeout => NdiConnectionState::Recovering,
            ConnectionEvent::Paused => NdiConnectionState::Paused,
            ConnectionEvent::Resumed => NdiConnectionState::Streaming,
            ConnectionEvent::AuthFailed(_) => NdiConnectionState::Failed,
        };
        self.transition(next);
    }

    /// Apply a `ConnectionEvent` using the supplied `ReconnectPolicy`.
    ///
    /// On `Disconnected` or `Error`/`Timeout` events the tracker enters the
    /// `Reconnecting(attempt)` state rather than `Closed`/`Recovering`,
    /// incrementing the attempt counter each time.  Once
    /// `policy.max_attempts` is exhausted the tracker moves to `Failed`.
    pub fn apply_event_with_policy(&mut self, event: &ConnectionEvent, policy: &ReconnectPolicy) {
        let current = self.state();
        let next = match event {
            ConnectionEvent::Connected | ConnectionEvent::Authenticated => {
                NdiConnectionState::Connecting
            }
            ConnectionEvent::Paused => NdiConnectionState::Paused,
            ConnectionEvent::Resumed => NdiConnectionState::Streaming,
            ConnectionEvent::AuthFailed(_) => NdiConnectionState::Failed,
            // Disconnected / errors enter Reconnecting state machine.
            ConnectionEvent::Disconnected
            | ConnectionEvent::Error(_)
            | ConnectionEvent::Timeout => {
                // Determine the next attempt number.
                let attempt = match current {
                    NdiConnectionState::Reconnecting(n) => n + 1,
                    _ => 0,
                };
                if policy.should_attempt(attempt) {
                    NdiConnectionState::Reconnecting(attempt)
                } else {
                    NdiConnectionState::Failed
                }
            }
        };
        self.transition(next);
    }

    /// Transition from `Reconnecting` to `Streaming` on successful reconnect.
    ///
    /// Should be called by the reconnect logic after a connection is
    /// re-established.  Has no effect if the current state is not
    /// `Reconnecting`.
    pub fn on_reconnect_success(&mut self) {
        if self.current.state.is_reconnecting() {
            self.transition(NdiConnectionState::Streaming);
        }
    }

    /// Returns `true` if the current state is healthy.
    pub fn is_healthy(&self) -> bool {
        self.current.state.is_healthy()
    }

    /// Number of state transitions recorded in history.
    pub fn transition_count(&self) -> usize {
        self.history.len()
    }

    /// Returns the history of previous states (oldest first).
    pub fn history(&self) -> impl Iterator<Item = NdiConnectionState> + '_ {
        self.history.iter().map(|e| e.state)
    }
}

impl Default for ConnectionStateTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_connection_event_is_error_true() {
        assert!(ConnectionEvent::Error("oops".into()).is_error());
        assert!(ConnectionEvent::Timeout.is_error());
        assert!(ConnectionEvent::AuthFailed("bad".into()).is_error());
    }

    #[test]
    fn test_connection_event_is_error_false() {
        assert!(!ConnectionEvent::Connected.is_error());
        assert!(!ConnectionEvent::Disconnected.is_error());
        assert!(!ConnectionEvent::Paused.is_error());
        assert!(!ConnectionEvent::Resumed.is_error());
        assert!(!ConnectionEvent::Authenticated.is_error());
    }

    #[test]
    fn test_connection_event_kind() {
        assert_eq!(ConnectionEvent::Connected.kind(), "connected");
        assert_eq!(ConnectionEvent::Error("x".into()).kind(), "error");
        assert_eq!(ConnectionEvent::Timeout.kind(), "timeout");
    }

    #[test]
    fn test_ndi_connection_state_is_healthy() {
        assert!(NdiConnectionState::Streaming.is_healthy());
        assert!(NdiConnectionState::Paused.is_healthy());
        assert!(!NdiConnectionState::Idle.is_healthy());
        assert!(!NdiConnectionState::Failed.is_healthy());
    }

    #[test]
    fn test_ndi_connection_state_is_terminal() {
        assert!(NdiConnectionState::Failed.is_terminal());
        assert!(NdiConnectionState::Closed.is_terminal());
        assert!(!NdiConnectionState::Streaming.is_terminal());
    }

    #[test]
    fn test_tracker_initial_state() {
        let tracker = ConnectionStateTracker::new();
        assert_eq!(tracker.state(), NdiConnectionState::Idle);
    }

    #[test]
    fn test_tracker_transition() {
        let mut tracker = ConnectionStateTracker::new();
        let prev = tracker.transition(NdiConnectionState::Connecting);
        assert_eq!(prev, NdiConnectionState::Idle);
        assert_eq!(tracker.state(), NdiConnectionState::Connecting);
    }

    #[test]
    fn test_tracker_transition_count() {
        let mut tracker = ConnectionStateTracker::new();
        tracker.transition(NdiConnectionState::Connecting);
        tracker.transition(NdiConnectionState::Streaming);
        assert_eq!(tracker.transition_count(), 2);
    }

    #[test]
    fn test_tracker_state_duration_ms_non_negative() {
        let tracker = ConnectionStateTracker::new();
        // Should be >= 0 and very small (just created).
        assert!(tracker.state_duration_ms() < 1000);
    }

    #[test]
    fn test_tracker_apply_event_error_goes_recovering() {
        let mut tracker = ConnectionStateTracker::new();
        tracker.apply_event(&ConnectionEvent::Error("net fail".into()));
        assert_eq!(tracker.state(), NdiConnectionState::Recovering);
    }

    #[test]
    fn test_tracker_apply_event_auth_failed_goes_failed() {
        let mut tracker = ConnectionStateTracker::new();
        tracker.apply_event(&ConnectionEvent::AuthFailed("bad token".into()));
        assert_eq!(tracker.state(), NdiConnectionState::Failed);
    }

    #[test]
    fn test_tracker_apply_event_resumed_goes_streaming() {
        let mut tracker = ConnectionStateTracker::new();
        tracker.apply_event(&ConnectionEvent::Resumed);
        assert_eq!(tracker.state(), NdiConnectionState::Streaming);
    }

    #[test]
    fn test_tracker_is_healthy_after_stream() {
        let mut tracker = ConnectionStateTracker::new();
        tracker.apply_event(&ConnectionEvent::Resumed);
        assert!(tracker.is_healthy());
    }

    #[test]
    fn test_tracker_history_iteration() {
        let mut tracker = ConnectionStateTracker::new();
        tracker.transition(NdiConnectionState::Connecting);
        tracker.transition(NdiConnectionState::Streaming);
        let hist: Vec<_> = tracker.history().collect();
        assert_eq!(hist[0], NdiConnectionState::Idle);
        assert_eq!(hist[1], NdiConnectionState::Connecting);
    }

    #[test]
    fn test_tracker_max_history_trimmed() {
        let mut tracker = ConnectionStateTracker::with_max_history(3);
        for _ in 0..5 {
            tracker.transition(NdiConnectionState::Connecting);
            tracker.transition(NdiConnectionState::Streaming);
        }
        assert!(tracker.transition_count() <= 3);
    }

    // -----------------------------------------------------------------------
    // ReconnectPolicy tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_reconnect_backoff_doubling() {
        let policy = ReconnectPolicy::new(10, 100, 60_000);
        // Attempt 0: 100 * 2^0 = 100 ms
        assert_eq!(policy.delay_for_attempt(0), 100);
        // Attempt 1: 100 * 2^1 = 200 ms
        assert_eq!(policy.delay_for_attempt(1), 200);
        // Attempt 2: 100 * 2^2 = 400 ms
        assert_eq!(policy.delay_for_attempt(2), 400);
        // Attempt 3: 100 * 2^3 = 800 ms
        assert_eq!(policy.delay_for_attempt(3), 800);
        // Attempt 4: 100 * 2^4 = 1600 ms
        assert_eq!(policy.delay_for_attempt(4), 1600);
    }

    #[test]
    fn test_reconnect_backoff_capped() {
        let policy = ReconnectPolicy::new(10, 1000, 5000);
        // After a few doublings it should be capped at max_delay_ms.
        assert_eq!(policy.delay_for_attempt(10), 5000);
        assert_eq!(policy.delay_for_attempt(100), 5000);
    }

    #[test]
    fn test_reconnect_policy_should_attempt() {
        let policy = ReconnectPolicy::new(3, 100, 10_000);
        assert!(policy.should_attempt(0));
        assert!(policy.should_attempt(1));
        assert!(policy.should_attempt(2));
        assert!(!policy.should_attempt(3)); // exhausted
    }

    #[test]
    fn test_reconnect_policy_disabled() {
        let policy = ReconnectPolicy::new(0, 100, 10_000);
        assert!(!policy.should_attempt(0));
    }

    #[test]
    fn test_connection_state_reconnecting_variant() {
        let state = NdiConnectionState::Reconnecting(2);
        assert!(state.is_reconnecting());
        assert_eq!(state.reconnect_attempt(), Some(2));
        assert!(!state.is_healthy());
        assert!(!state.is_terminal());
    }

    #[test]
    fn test_apply_event_with_policy_enters_reconnecting() {
        let mut tracker = ConnectionStateTracker::new();
        tracker.transition(NdiConnectionState::Streaming);
        let policy = ReconnectPolicy::new(5, 100, 30_000);
        tracker.apply_event_with_policy(&ConnectionEvent::Disconnected, &policy);
        assert_eq!(tracker.state(), NdiConnectionState::Reconnecting(0));
    }

    #[test]
    fn test_apply_event_with_policy_increments_attempt() {
        let mut tracker = ConnectionStateTracker::new();
        tracker.transition(NdiConnectionState::Streaming);
        let policy = ReconnectPolicy::new(5, 100, 30_000);
        // First disconnect → Reconnecting(0)
        tracker.apply_event_with_policy(&ConnectionEvent::Disconnected, &policy);
        assert_eq!(tracker.state(), NdiConnectionState::Reconnecting(0));
        // Another failure while reconnecting → Reconnecting(1)
        tracker.apply_event_with_policy(&ConnectionEvent::Error("timeout".into()), &policy);
        assert_eq!(tracker.state(), NdiConnectionState::Reconnecting(1));
    }

    #[test]
    fn test_apply_event_with_policy_exceeds_max_goes_failed() {
        let mut tracker = ConnectionStateTracker::new();
        tracker.transition(NdiConnectionState::Streaming);
        let policy = ReconnectPolicy::new(2, 100, 30_000);
        // Attempt 0
        tracker.apply_event_with_policy(&ConnectionEvent::Disconnected, &policy);
        assert_eq!(tracker.state(), NdiConnectionState::Reconnecting(0));
        // Attempt 1
        tracker.apply_event_with_policy(&ConnectionEvent::Disconnected, &policy);
        assert_eq!(tracker.state(), NdiConnectionState::Reconnecting(1));
        // Attempt 2 — exceeds max_attempts (2), should go Failed
        tracker.apply_event_with_policy(&ConnectionEvent::Disconnected, &policy);
        assert_eq!(tracker.state(), NdiConnectionState::Failed);
    }

    #[test]
    fn test_on_reconnect_success() {
        let mut tracker = ConnectionStateTracker::new();
        tracker.transition(NdiConnectionState::Reconnecting(1));
        tracker.on_reconnect_success();
        assert_eq!(tracker.state(), NdiConnectionState::Streaming);
    }
}
