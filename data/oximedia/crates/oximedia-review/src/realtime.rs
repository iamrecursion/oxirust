//! Real-time collaboration features.

use crate::{error::ReviewResult, SessionId, User};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

pub mod cursor;
pub mod presence;
pub mod sync;

pub use cursor::{CursorPosition, UserCursor};
pub use presence::{PresenceStatus, UserPresence};
pub use sync::SyncMessage;

/// Real-time event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RealtimeEvent {
    /// User joined the session.
    UserJoined(User),
    /// User left the session.
    UserLeft(String),
    /// Comment added.
    CommentAdded {
        /// Comment ID.
        comment_id: crate::CommentId,
        /// User who added the comment.
        user_id: String,
    },
    /// Drawing added.
    DrawingAdded {
        /// Drawing ID.
        drawing_id: crate::DrawingId,
        /// User who added the drawing.
        user_id: String,
    },
    /// Cursor moved.
    CursorMoved {
        /// User ID.
        user_id: String,
        /// Frame number.
        frame: i64,
        /// Position.
        position: CursorPosition,
    },
    /// Presence updated.
    PresenceUpdated {
        /// User ID.
        user_id: String,
        /// New status.
        status: PresenceStatus,
    },
    /// Delta-based annotation synchronisation.
    ///
    /// The `payload` is the JSON encoding of a
    /// [`crate::realtime_delta::DeltaMessage`] (incremental deltas, a full
    /// snapshot, or a resync signal). `seq` is the broadcaster sequence number
    /// this message advances the receiver to (0 for snapshot/resync).
    AnnotationDelta {
        /// Sequence number this message advances the receiver to.
        seq: u64,
        /// JSON-encoded [`crate::realtime_delta::DeltaMessage`].
        payload: String,
    },
}

/// Real-time session.
pub struct RealtimeSession {
    session_id: SessionId,
    active_users: Vec<User>,
}

impl RealtimeSession {
    /// Create a new real-time session.
    ///
    /// On non-WASM targets this also registers a broadcast channel so that
    /// [`broadcast_event`] and [`subscribe`] can be used for the session.
    #[must_use]
    pub fn new(session_id: SessionId) -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let (tx, _rx) = tokio::sync::broadcast::channel::<RealtimeEvent>(64);
            let key = session_id.to_string();
            if let Ok(mut registry) = REALTIME_REGISTRY
                .get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
                .lock()
            {
                registry.insert(key, tx);
            }
        }
        Self {
            session_id,
            active_users: Vec::new(),
        }
    }

    /// Add a user to the session.
    pub fn add_user(&mut self, user: User) {
        if !self.active_users.iter().any(|u| u.id == user.id) {
            self.active_users.push(user);
        }
    }

    /// Remove a user from the session.
    pub fn remove_user(&mut self, user_id: &str) {
        self.active_users.retain(|u| u.id != user_id);
    }

    /// Get all active users.
    #[must_use]
    pub fn active_users(&self) -> &[User] {
        &self.active_users
    }

    /// Count active users.
    #[must_use]
    pub fn user_count(&self) -> usize {
        self.active_users.len()
    }

    /// Get session ID.
    #[must_use]
    pub fn session_id(&self) -> SessionId {
        self.session_id
    }
}

/// Process-local registry mapping session IDs to broadcast senders.
///
/// Only available on non-WASM targets.
#[cfg(not(target_arch = "wasm32"))]
static REALTIME_REGISTRY: std::sync::OnceLock<
    std::sync::Mutex<
        std::collections::HashMap<String, tokio::sync::broadcast::Sender<RealtimeEvent>>,
    >,
> = std::sync::OnceLock::new();

/// Broadcast an event to all subscribers of the given session.
///
/// On non-WASM targets this fans the event out through the process-local
/// `REALTIME_REGISTRY`.  Subscribers that have fallen behind (lagged) are
/// silently skipped — their receivers will receive a
/// [`tokio::sync::broadcast::error::RecvError::Lagged`] on their next poll.
///
/// # Errors
///
/// Returns `ReviewError::InvalidConfig` when the session has not been
/// registered (i.e. [`RealtimeSession::new`] was never called for this ID).
/// On WASM targets the call is a no-op and always succeeds.
pub async fn broadcast_event(session_id: SessionId, event: RealtimeEvent) -> ReviewResult<()> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        use crate::error::ReviewError;

        let key = session_id.to_string();
        let registry = REALTIME_REGISTRY
            .get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));

        let sender = registry
            .lock()
            .map_err(|_| ReviewError::InvalidConfig("registry lock poisoned".into()))
            .and_then(|guard| {
                guard.get(&key).cloned().ok_or_else(|| {
                    ReviewError::InvalidConfig(format!("session '{key}' not registered"))
                })
            })?;

        // `send` returns the number of receivers that got the event, or an
        // error if there are *no* receivers at all.  We treat zero-receiver
        // sessions as success (events are fire-and-forget when nobody is
        // subscribed).
        let _ = sender.send(event);
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (session_id, event);
    }
    Ok(())
}

/// Broadcast the incremental annotation delta between `prev` and `curr`.
///
/// This is the delta-based replacement for full-state annotation broadcast: it
/// diffs the two annotation sets (attributing changes to `author`), wraps the
/// resulting operations in a [`crate::realtime_delta::DeltaMessage::Deltas`],
/// and fans it out through the same process-local registry used by
/// [`broadcast_event`].
///
/// If the diff cannot be serialised (a corrupt annotation), the bridge falls
/// back to sending a full [`crate::realtime_delta::DeltaMessage::Snapshot`] of
/// `curr` so receivers still converge — never dropping the update silently.
///
/// Returns the assigned sequence number (the number of deltas emitted; `0`
/// when there was no change and nothing was sent).
///
/// # Errors
///
/// Returns `ReviewError::InvalidConfig` when the session has not been
/// registered, or `ReviewError::Serialization` if even the snapshot fallback
/// fails to encode.
#[cfg(not(target_arch = "wasm32"))]
pub async fn broadcast_annotation_delta(
    session_id: SessionId,
    prev: &[crate::drawing::Annotation],
    curr: &[crate::drawing::Annotation],
    author: &str,
) -> ReviewResult<crate::realtime_delta::Seq> {
    use crate::realtime_delta::{try_diff_annotations, DeltaEntry, DeltaMessage};

    // Strict diff: if serialisation fails, fall back to a full snapshot so the
    // receiver still converges (never silently drop the change).
    let message = match try_diff_annotations(prev, curr, author) {
        Ok(deltas) => {
            if deltas.is_empty() {
                // Nothing changed — no traffic, seq stays 0.
                return Ok(0);
            }
            let now = chrono::Utc::now();
            let entries: Vec<DeltaEntry> = deltas
                .into_iter()
                .enumerate()
                .map(|(i, delta)| DeltaEntry {
                    seq: (i as u64) + 1,
                    timestamp: now,
                    delta,
                })
                .collect();
            DeltaMessage::Deltas {
                base_seq: 0,
                entries,
            }
        }
        Err(_) => DeltaMessage::Snapshot {
            seq: 0,
            annotations: curr.to_vec(),
        },
    };

    let seq = match &message {
        DeltaMessage::Deltas { entries, .. } => entries.len() as u64,
        _ => 0,
    };
    let payload = serde_json::to_string(&message)?;
    broadcast_event(session_id, RealtimeEvent::AnnotationDelta { seq, payload }).await?;
    Ok(seq)
}

/// Broadcast a full annotation snapshot for the session.
///
/// This is the back-compatible full-state path: it wraps `annotations` in a
/// [`crate::realtime_delta::DeltaMessage::Snapshot`] and fans it out. Use it
/// when a client first joins, after a resync, or whenever the incremental diff
/// is not applicable.
///
/// # Errors
///
/// Returns `ReviewError::InvalidConfig` when the session is not registered, or
/// `ReviewError::Serialization` if the snapshot cannot be encoded.
#[cfg(not(target_arch = "wasm32"))]
pub async fn broadcast_annotation_snapshot(
    session_id: SessionId,
    annotations: &[crate::drawing::Annotation],
) -> ReviewResult<()> {
    use crate::realtime_delta::DeltaMessage;

    let message = DeltaMessage::Snapshot {
        seq: 0,
        annotations: annotations.to_vec(),
    };
    let payload = serde_json::to_string(&message)?;
    broadcast_event(
        session_id,
        RealtimeEvent::AnnotationDelta { seq: 0, payload },
    )
    .await
}

/// Subscribe to events for the given session.
///
/// Returns `Some(receiver)` when the session exists in the registry, or
/// `None` when it has not been registered.  Only available on non-WASM
/// targets.
#[cfg(not(target_arch = "wasm32"))]
pub fn subscribe(session_id: SessionId) -> Option<tokio::sync::broadcast::Receiver<RealtimeEvent>> {
    let key = session_id.to_string();
    let registry =
        REALTIME_REGISTRY.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    registry
        .lock()
        .ok()
        .and_then(|guard| guard.get(&key).map(|tx| tx.subscribe()))
}

/// Activity log entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivityLogEntry {
    /// Entry ID.
    pub id: String,
    /// Session ID.
    pub session_id: SessionId,
    /// User who performed the activity.
    pub user_id: String,
    /// Activity type.
    pub activity_type: String,
    /// Activity details.
    pub details: String,
    /// Timestamp.
    pub timestamp: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::UserRole;

    fn create_test_user(id: &str) -> User {
        User {
            id: id.to_string(),
            name: format!("User {}", id),
            email: format!("{}@example.com", id),
            role: UserRole::Reviewer,
        }
    }

    #[test]
    fn test_realtime_session_creation() {
        let session_id = SessionId::new();
        let session = RealtimeSession::new(session_id);
        assert_eq!(session.user_count(), 0);
    }

    #[test]
    fn test_realtime_session_add_user() {
        let session_id = SessionId::new();
        let mut session = RealtimeSession::new(session_id);

        let user = create_test_user("user1");
        session.add_user(user);

        assert_eq!(session.user_count(), 1);
    }

    #[test]
    fn test_realtime_session_remove_user() {
        let session_id = SessionId::new();
        let mut session = RealtimeSession::new(session_id);

        let user = create_test_user("user1");
        session.add_user(user);
        assert_eq!(session.user_count(), 1);

        session.remove_user("user1");
        assert_eq!(session.user_count(), 0);
    }

    #[test]
    fn test_realtime_session_duplicate_user() {
        let session_id = SessionId::new();
        let mut session = RealtimeSession::new(session_id);

        let user = create_test_user("user1");
        session.add_user(user.clone());
        session.add_user(user);

        assert_eq!(session.user_count(), 1);
    }

    #[tokio::test]
    async fn test_broadcast_event() {
        let session_id = SessionId::new();
        let _session = RealtimeSession::new(session_id);
        let event = RealtimeEvent::UserJoined(create_test_user("user1"));

        let result = broadcast_event(session_id, event).await;
        assert!(result.is_ok());
    }

    /// Broadcast delivers to a subscriber that called `subscribe()`.
    #[cfg(not(target_arch = "wasm32"))]
    #[tokio::test]
    async fn test_broadcast_delivers_to_subscriber() {
        let session_id = SessionId::new();
        let _session = RealtimeSession::new(session_id);

        let mut rx = subscribe(session_id).expect("session should be registered");

        let event = RealtimeEvent::UserJoined(create_test_user("alice"));
        broadcast_event(session_id, event).await.unwrap();

        let received = rx.recv().await.expect("should receive event");
        match received {
            RealtimeEvent::UserJoined(u) => assert_eq!(u.id, "alice"),
            other => panic!("unexpected event variant: {other:?}"),
        }
    }

    /// A lagged receiver should not cause the broadcaster to panic.
    #[cfg(not(target_arch = "wasm32"))]
    #[tokio::test]
    async fn test_broadcast_survives_lagged_receiver() {
        let session_id = SessionId::new();
        let _session = RealtimeSession::new(session_id);

        // Keep a subscriber alive but never drain it — it will lag after 64
        // messages (the channel capacity).
        let lagging_rx = subscribe(session_id).expect("session should be registered");

        // Send 70 events — 6 more than the channel capacity.
        for i in 0u64..70 {
            let event = RealtimeEvent::UserLeft(format!("user-{i}"));
            broadcast_event(session_id, event)
                .await
                .expect("broadcast must not error even with lagging receiver");
        }

        // The lagged receiver will get RecvError::Lagged on its next recv() —
        // verify this is handled gracefully (no panic path in the library).
        let mut rx = lagging_rx;
        match rx.recv().await {
            Ok(_) | Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {}
            Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                // channel was closed — also acceptable
            }
        }
    }

    /// `broadcast_annotation_delta` delivers an `AnnotationDelta` event whose
    /// payload decodes to a `DeltaMessage::Deltas`, and applying it reconstructs
    /// the new annotation.
    #[cfg(not(target_arch = "wasm32"))]
    #[tokio::test]
    async fn test_broadcast_annotation_delta_delivers() {
        use crate::drawing::{
            annotation::Annotation, Circle, Color, Drawing, DrawingTool, Point, Shape, StrokeStyle,
        };
        use crate::realtime_delta::{apply_message, DeltaMessage};
        use crate::DrawingId;
        use std::collections::HashMap;

        let session_id = SessionId::new();
        let _session = RealtimeSession::new(session_id);
        let mut rx = subscribe(session_id).expect("session should be registered");

        let id = DrawingId::new();
        let drawing = Drawing {
            id,
            session_id,
            frame: 42,
            tool: DrawingTool::Circle,
            shape: Shape::Circle(Circle::new(Point::new(0.5, 0.5), 0.2)),
            style: StrokeStyle::solid(Color::red(), 2.0),
            author: "alice".to_string(),
        };
        let new_ann = Annotation::new(drawing);

        // Diff from empty → {new_ann}: a single Add.
        let seq =
            broadcast_annotation_delta(session_id, &[], std::slice::from_ref(&new_ann), "alice")
                .await
                .expect("broadcast delta");
        assert_eq!(seq, 1, "one Add delta emitted");

        let received = rx.recv().await.expect("should receive delta event");
        let payload = match received {
            RealtimeEvent::AnnotationDelta { seq: s, payload } => {
                assert_eq!(s, 1);
                payload
            }
            other => panic!("unexpected event variant: {other:?}"),
        };

        let msg: DeltaMessage = serde_json::from_str(&payload).expect("decode message");
        assert!(matches!(msg, DeltaMessage::Deltas { .. }));

        let mut state: HashMap<DrawingId, Annotation> = HashMap::new();
        apply_message(&mut state, &msg).expect("apply received message");
        assert!(
            state.contains_key(&id),
            "applied state contains new annotation"
        );
    }

    /// A no-op diff (prev == curr) emits nothing and returns seq 0.
    #[cfg(not(target_arch = "wasm32"))]
    #[tokio::test]
    async fn test_broadcast_annotation_delta_no_change_is_noop() {
        use crate::drawing::{
            annotation::Annotation, Circle, Color, Drawing, DrawingTool, Point, Shape, StrokeStyle,
        };
        use crate::DrawingId;

        let session_id = SessionId::new();
        let _session = RealtimeSession::new(session_id);

        let id = DrawingId::new();
        let drawing = Drawing {
            id,
            session_id,
            frame: 1,
            tool: DrawingTool::Circle,
            shape: Shape::Circle(Circle::new(Point::new(0.5, 0.5), 0.2)),
            style: StrokeStyle::solid(Color::red(), 2.0),
            author: "alice".to_string(),
        };
        let ann = Annotation::new(drawing);
        let state = std::slice::from_ref(&ann);

        let seq = broadcast_annotation_delta(session_id, state, state, "alice")
            .await
            .expect("broadcast no-op");
        assert_eq!(seq, 0, "no change → no delta, seq 0");
    }

    /// `broadcast_annotation_snapshot` delivers a `Snapshot` message.
    #[cfg(not(target_arch = "wasm32"))]
    #[tokio::test]
    async fn test_broadcast_annotation_snapshot_delivers() {
        use crate::drawing::{
            annotation::Annotation, Circle, Color, Drawing, DrawingTool, Point, Shape, StrokeStyle,
        };
        use crate::realtime_delta::DeltaMessage;
        use crate::DrawingId;

        let session_id = SessionId::new();
        let _session = RealtimeSession::new(session_id);
        let mut rx = subscribe(session_id).expect("session should be registered");

        let id = DrawingId::new();
        let drawing = Drawing {
            id,
            session_id,
            frame: 7,
            tool: DrawingTool::Circle,
            shape: Shape::Circle(Circle::new(Point::new(0.5, 0.5), 0.2)),
            style: StrokeStyle::solid(Color::red(), 2.0),
            author: "alice".to_string(),
        };
        let ann = Annotation::new(drawing);

        broadcast_annotation_snapshot(session_id, &[ann])
            .await
            .expect("broadcast snapshot");

        let received = rx.recv().await.expect("should receive snapshot event");
        let payload = match received {
            RealtimeEvent::AnnotationDelta { seq, payload } => {
                assert_eq!(seq, 0, "snapshot carries seq 0");
                payload
            }
            other => panic!("unexpected event variant: {other:?}"),
        };
        let msg: DeltaMessage = serde_json::from_str(&payload).expect("decode message");
        assert!(matches!(msg, DeltaMessage::Snapshot { .. }));
    }
}
