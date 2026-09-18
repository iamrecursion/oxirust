//! WebSocket connection manager.

use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::broadcast;

/// WebSocket message types.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Message {
    /// Transcoding job progress update
    TranscodeProgress {
        /// Job ID
        job_id: String,
        /// Progress percentage (0-100)
        progress: f64,
        /// Current status
        status: String,
    },
    /// Media processing update
    MediaProcessing {
        /// Media ID
        media_id: String,
        /// Processing status
        status: String,
    },
    /// Upload progress update
    UploadProgress {
        /// Upload ID
        upload_id: String,
        /// Progress percentage (0-100)
        progress: f64,
    },
    /// General notification
    Notification {
        /// Notification message
        message: String,
        /// Severity level
        level: String,
    },
    /// Ping/pong for keep-alive
    Ping,
    /// Pong response
    Pong,
}

/// WebSocket connection manager.
#[derive(Clone)]
pub struct WebSocketManager {
    /// User connections: user_id -> broadcast channel
    connections: Arc<DashMap<String, broadcast::Sender<Message>>>,
}

impl WebSocketManager {
    /// Creates a new WebSocket manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            connections: Arc::new(DashMap::new()),
        }
    }

    /// Registers a new connection for a user.
    #[must_use]
    pub fn register(&self, user_id: String) -> broadcast::Receiver<Message> {
        let (tx, rx) = broadcast::channel(100);
        self.connections.insert(user_id, tx);
        rx
    }

    /// Unregisters a connection.
    pub fn unregister(&self, user_id: &str) {
        self.connections.remove(user_id);
    }

    /// Sends a message to a specific user.
    pub fn send_to_user(&self, user_id: &str, message: Message) {
        if let Some(tx) = self.connections.get(user_id) {
            tx.send(message).ok();
        }
    }

    /// Broadcasts a message to all connected users.
    pub fn broadcast(&self, message: Message) {
        for entry in self.connections.iter() {
            entry.value().send(message.clone()).ok();
        }
    }

    /// Gets the number of active connections.
    #[must_use]
    pub fn connection_count(&self) -> usize {
        self.connections.len()
    }
}

impl Default for WebSocketManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn progress(job: &str, pct: f64) -> Message {
        Message::TranscodeProgress {
            job_id: job.to_string(),
            progress: pct,
            status: "running".to_string(),
        }
    }

    #[tokio::test]
    async fn test_connection_lifecycle_connect_push_disconnect() {
        let mgr = WebSocketManager::new();
        assert_eq!(mgr.connection_count(), 0);

        // ── Connect + subscribe ───────────────────────────────────────────
        let mut rx = mgr.register("alice".to_string());
        assert_eq!(mgr.connection_count(), 1);

        // ── Server-pushed event reaches the subscriber ────────────────────
        mgr.send_to_user("alice", progress("job-1", 42.0));
        let received = rx.recv().await.expect("alice should receive the push");
        match received {
            Message::TranscodeProgress {
                job_id, progress, ..
            } => {
                assert_eq!(job_id, "job-1");
                assert!((progress - 42.0).abs() < f64::EPSILON);
            }
            other => panic!("unexpected message variant: {other:?}"),
        }

        // ── Disconnect ────────────────────────────────────────────────────
        mgr.unregister("alice");
        assert_eq!(mgr.connection_count(), 0);

        // After disconnect, sending to the (now absent) user is a no-op and the
        // dropped sender closes the channel, so further recv() yields Closed.
        mgr.send_to_user("alice", progress("job-1", 100.0));
        drop(mgr);
        assert!(
            rx.recv().await.is_err(),
            "channel must be closed after unregister"
        );
    }

    #[tokio::test]
    async fn test_events_are_filtered_per_subscription() {
        let mgr = WebSocketManager::new();
        let mut alice = mgr.register("alice".to_string());
        let mut bob = mgr.register("bob".to_string());
        assert_eq!(mgr.connection_count(), 2);

        // A targeted push to Alice must NOT be delivered to Bob.
        mgr.send_to_user("alice", progress("a-job", 10.0));

        let got = alice.recv().await.expect("alice receives her event");
        match got {
            Message::TranscodeProgress { job_id, .. } => assert_eq!(job_id, "a-job"),
            other => panic!("unexpected: {other:?}"),
        }

        // Bob's channel has nothing queued — try_recv reports Empty, not a value.
        assert!(
            matches!(bob.try_recv(), Err(broadcast::error::TryRecvError::Empty)),
            "bob must not receive alice's targeted event"
        );
    }

    #[tokio::test]
    async fn test_broadcast_reaches_all_subscribers() {
        let mgr = WebSocketManager::new();
        let mut a = mgr.register("a".to_string());
        let mut b = mgr.register("b".to_string());

        mgr.broadcast(Message::Notification {
            message: "maintenance".to_string(),
            level: "info".to_string(),
        });

        for rx in [&mut a, &mut b] {
            match rx.recv().await.expect("each subscriber gets the broadcast") {
                Message::Notification { message, level } => {
                    assert_eq!(message, "maintenance");
                    assert_eq!(level, "info");
                }
                other => panic!("unexpected: {other:?}"),
            }
        }
    }

    #[tokio::test]
    async fn test_send_to_unknown_user_is_noop() {
        let mgr = WebSocketManager::new();
        // No connections registered — must not panic.
        mgr.send_to_user("ghost", progress("x", 1.0));
        assert_eq!(mgr.connection_count(), 0);
    }
}
