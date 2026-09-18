//! WebSocket real-time updates.

use crate::{AutomationError, Result};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::{broadcast, mpsc, RwLock};
use tracing::{debug, info, warn};

/// WebSocket message type.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WebSocketMessage {
    /// Status update
    #[serde(rename = "status")]
    Status {
        /// Status data payload
        data: serde_json::Value,
    },

    /// Event notification
    #[serde(rename = "event")]
    Event {
        /// Event name
        event: String,
        /// Event data payload
        data: serde_json::Value,
    },

    /// Metrics update
    #[serde(rename = "metrics")]
    Metrics {
        /// Metrics data payload
        data: serde_json::Value,
    },

    /// Alert notification
    #[serde(rename = "alert")]
    Alert {
        /// Alert severity level
        severity: String,
        /// Alert message text
        message: String,
    },

    /// Ping
    #[serde(rename = "ping")]
    Ping,

    /// Pong
    #[serde(rename = "pong")]
    Pong,
}

/// WebSocket connection.
pub struct WebSocketConnection {
    id: String,
    tx: mpsc::UnboundedSender<String>,
}

impl WebSocketConnection {
    /// Create a new WebSocket connection.
    pub fn new(id: String, tx: mpsc::UnboundedSender<String>) -> Self {
        Self { id, tx }
    }

    /// Send message to connection.
    pub fn send(&self, message: String) -> Result<()> {
        self.tx
            .send(message)
            .map_err(|_| AutomationError::RemoteControl("Failed to send message".to_string()))?;
        Ok(())
    }

    /// Get connection ID.
    pub fn id(&self) -> &str {
        &self.id
    }
}

/// WebSocket handler.
pub struct WebSocketHandler {
    connections: Arc<RwLock<Vec<WebSocketConnection>>>,
}

impl WebSocketHandler {
    /// Create a new WebSocket handler.
    pub fn new() -> Self {
        info!("Creating WebSocket handler");

        Self {
            connections: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Add a new connection.
    pub async fn add_connection(&self, id: String, tx: mpsc::UnboundedSender<String>) {
        debug!("Adding WebSocket connection: {}", id);

        let connection = WebSocketConnection::new(id, tx);

        let mut connections = self.connections.write().await;
        connections.push(connection);
    }

    /// Remove a connection.
    pub async fn remove_connection(&self, id: &str) {
        debug!("Removing WebSocket connection: {}", id);

        let mut connections = self.connections.write().await;
        connections.retain(|conn| conn.id() != id);
    }

    /// Broadcast message to all connections.
    pub async fn broadcast(&self, message: String) -> Result<()> {
        debug!(
            "Broadcasting message to {} connections",
            self.connections.read().await.len()
        );

        let connections = self.connections.read().await;

        for connection in connections.iter() {
            if let Err(e) = connection.send(message.clone()) {
                debug!("Failed to send to connection {}: {}", connection.id(), e);
            }
        }

        Ok(())
    }

    /// Send message to specific connection.
    pub async fn send_to(&self, id: &str, message: String) -> Result<()> {
        let connections = self.connections.read().await;

        for connection in connections.iter() {
            if connection.id() == id {
                return connection.send(message);
            }
        }

        Err(AutomationError::NotFound(format!("Connection {id}")))
    }

    /// Get number of active connections.
    pub async fn connection_count(&self) -> usize {
        self.connections.read().await.len()
    }

    /// Broadcast status update.
    pub async fn broadcast_status(&self, data: serde_json::Value) -> Result<()> {
        let message = WebSocketMessage::Status { data };
        let json = serde_json::to_string(&message)?;
        self.broadcast(json).await
    }

    /// Broadcast event notification.
    pub async fn broadcast_event(&self, event: String, data: serde_json::Value) -> Result<()> {
        let message = WebSocketMessage::Event { event, data };
        let json = serde_json::to_string(&message)?;
        self.broadcast(json).await
    }

    /// Broadcast metrics update.
    pub async fn broadcast_metrics(&self, data: serde_json::Value) -> Result<()> {
        let message = WebSocketMessage::Metrics { data };
        let json = serde_json::to_string(&message)?;
        self.broadcast(json).await
    }

    /// Broadcast alert.
    pub async fn broadcast_alert(&self, severity: String, message: String) -> Result<()> {
        let msg = WebSocketMessage::Alert { severity, message };
        let json = serde_json::to_string(&msg)?;
        self.broadcast(json).await
    }
}

impl Default for WebSocketHandler {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Real-time event streaming via tokio broadcast channel
// ─────────────────────────────────────────────────────────────────────────────

/// The default broadcast channel capacity (number of messages buffered before
/// lagging receivers are dropped).
const BROADCAST_CAPACITY: usize = 256;

/// An automation event that can be streamed to WebSocket dashboard clients.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutomationEvent {
    /// Monotonically increasing sequence number.
    pub sequence: u64,
    /// Event kind (e.g. `"playout_start"`, `"failover_triggered"`).
    pub kind: String,
    /// Channel identifier, if the event is channel-scoped.
    pub channel_id: Option<String>,
    /// JSON payload carrying event-specific data.
    pub payload: serde_json::Value,
    /// UNIX timestamp in milliseconds when the event occurred.
    pub timestamp_ms: i64,
}

impl AutomationEvent {
    /// Create a new automation event with the given kind and payload.
    pub fn new(kind: impl Into<String>, payload: serde_json::Value) -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);
        Self {
            sequence: 0, // set by the stream publisher
            kind: kind.into(),
            channel_id: None,
            payload,
            timestamp_ms,
        }
    }

    /// Attach a channel ID to the event.
    pub fn with_channel(mut self, channel_id: impl Into<String>) -> Self {
        self.channel_id = Some(channel_id.into());
        self
    }
}

/// Real-time WebSocket event stream using a tokio broadcast channel.
///
/// The `WebSocketEventStream` acts as the publisher side of a fan-out bus.
/// Each WebSocket client obtains a [`broadcast::Receiver`] by calling
/// [`subscribe`][Self::subscribe].  Published events are cloned and delivered
/// to every active subscriber; lagging receivers (e.g. a slow browser tab)
/// are silently dropped per the tokio broadcast semantics.
///
/// # Example
///
/// ```rust,no_run
/// use oximedia_automation::remote::websocket::{WebSocketEventStream, AutomationEvent};
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let stream = WebSocketEventStream::new(256);
///
/// // In a WebSocket handler task:
/// let mut rx = stream.subscribe();
/// tokio::spawn(async move {
///     loop {
///         match rx.recv().await {
///             Ok(event) => println!("Got event: {}", event.kind),
///             Err(_) => break,
///         }
///     }
/// });
///
/// // Publish an event from the automation engine:
/// stream.publish(AutomationEvent::new("playout_start", serde_json::json!({"clip": "news.mp4"}))).await?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct WebSocketEventStream {
    sender: Arc<broadcast::Sender<AutomationEvent>>,
    /// Shared sequence counter — incremented atomically per publish.
    sequence: Arc<std::sync::atomic::AtomicU64>,
}

impl WebSocketEventStream {
    /// Create a new event stream with the given broadcast channel `capacity`.
    ///
    /// A larger capacity reduces the chance of lagging receivers being dropped
    /// at the cost of additional memory.
    pub fn new(capacity: usize) -> Self {
        let (sender, _) = broadcast::channel(capacity.max(1));
        info!("Created WebSocketEventStream with capacity {}", capacity);
        Self {
            sender: Arc::new(sender),
            sequence: Arc::new(std::sync::atomic::AtomicU64::new(1)),
        }
    }

    /// Create a new event stream with the default capacity of
    /// `BROADCAST_CAPACITY` (256 messages).
    pub fn with_default_capacity() -> Self {
        Self::new(BROADCAST_CAPACITY)
    }

    /// Subscribe to the event stream.
    ///
    /// The returned [`broadcast::Receiver`] will receive all events published
    /// after this call.  Dropping the receiver unsubscribes the client.
    pub fn subscribe(&self) -> broadcast::Receiver<AutomationEvent> {
        self.sender.subscribe()
    }

    /// Publish an automation event to all current subscribers.
    ///
    /// The event's `sequence` field is assigned automatically.  If there are
    /// no active subscribers the event is silently discarded.
    ///
    /// # Errors
    ///
    /// Returns [`AutomationError::RemoteControl`] if the event cannot be
    /// serialised to JSON for logging purposes.
    pub async fn publish(&self, mut event: AutomationEvent) -> Result<()> {
        event.sequence = self
            .sequence
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let subscriber_count = self.sender.receiver_count();
        debug!(
            "Publishing automation event '{}' (seq={}) to {} subscriber(s)",
            event.kind, event.sequence, subscriber_count
        );

        // `send` returns Err only when there are no receivers — that is not
        // an application error, so we log a debug message and move on.
        if let Err(_e) = self.sender.send(event) {
            debug!("No active WebSocket subscribers; event discarded");
        }
        Ok(())
    }

    /// Publish a simple named event with an arbitrary JSON payload.
    pub async fn emit(&self, kind: impl Into<String>, payload: serde_json::Value) -> Result<()> {
        self.publish(AutomationEvent::new(kind, payload)).await
    }

    /// Publish a channel-scoped event.
    pub async fn emit_channel(
        &self,
        kind: impl Into<String>,
        channel_id: impl Into<String>,
        payload: serde_json::Value,
    ) -> Result<()> {
        let event = AutomationEvent::new(kind, payload).with_channel(channel_id);
        self.publish(event).await
    }

    /// Returns the number of active subscribers.
    pub fn subscriber_count(&self) -> usize {
        self.sender.receiver_count()
    }

    /// Broadcast a formatted alert event to all subscribers.
    pub async fn emit_alert(
        &self,
        severity: impl Into<String>,
        message: impl Into<String>,
    ) -> Result<()> {
        let severity = severity.into();
        let message = message.into();
        warn!("Broadcasting alert event [{}]: {}", severity, message);
        self.emit(
            "alert",
            serde_json::json!({ "severity": severity, "message": message }),
        )
        .await
    }
}

impl std::fmt::Debug for WebSocketEventStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WebSocketEventStream")
            .field("subscriber_count", &self.sender.receiver_count())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_websocket_handler() {
        let handler = WebSocketHandler::new();
        assert_eq!(handler.connection_count().await, 0);

        let (tx, _rx) = mpsc::unbounded_channel();
        handler.add_connection("conn1".to_string(), tx).await;

        assert_eq!(handler.connection_count().await, 1);

        handler.remove_connection("conn1").await;
        assert_eq!(handler.connection_count().await, 0);
    }

    #[tokio::test]
    async fn test_broadcast() {
        let handler = WebSocketHandler::new();

        let (tx, mut rx) = mpsc::unbounded_channel();
        handler.add_connection("conn1".to_string(), tx).await;

        handler
            .broadcast("test message".to_string())
            .await
            .expect("operation should succeed");

        let received = rx.recv().await.expect("recv should succeed");
        assert_eq!(received, "test message");
    }

    #[test]
    fn test_websocket_message_serialization() {
        let message = WebSocketMessage::Status {
            data: serde_json::json!({"status": "ok"}),
        };

        let json = serde_json::to_string(&message).expect("to_string should succeed");
        assert!(json.contains("\"type\":\"status\""));
    }

    // ── WebSocketEventStream tests ──────────────────────────────────────────

    #[tokio::test]
    async fn test_event_stream_publish_and_receive() {
        let stream = WebSocketEventStream::new(16);
        let mut rx = stream.subscribe();

        stream
            .emit("playout_start", serde_json::json!({"clip": "news.mp4"}))
            .await
            .expect("emit should succeed");

        let event = rx.recv().await.expect("Should receive event");
        assert_eq!(event.kind, "playout_start");
        assert!(event.sequence > 0, "sequence should be assigned");
    }

    #[tokio::test]
    async fn test_event_stream_channel_scoped() {
        let stream = WebSocketEventStream::new(16);
        let mut rx = stream.subscribe();

        stream
            .emit_channel(
                "failover_triggered",
                "CH1",
                serde_json::json!({"reason": "health_check_failed"}),
            )
            .await
            .expect("emit_channel should succeed");

        let event = rx.recv().await.expect("Should receive event");
        assert_eq!(event.kind, "failover_triggered");
        assert_eq!(event.channel_id.as_deref(), Some("CH1"));
    }

    #[tokio::test]
    async fn test_event_stream_no_subscribers_no_error() {
        let stream = WebSocketEventStream::new(16);
        // Publish without any subscriber should not return an error
        stream
            .emit("heartbeat", serde_json::json!({}))
            .await
            .expect("emit with no subscribers should not fail");
    }

    #[tokio::test]
    async fn test_event_stream_multiple_subscribers() {
        let stream = WebSocketEventStream::new(32);
        let mut rx1 = stream.subscribe();
        let mut rx2 = stream.subscribe();

        stream
            .emit("alert", serde_json::json!({"level": "critical"}))
            .await
            .expect("emit should succeed");

        let ev1 = rx1.recv().await.expect("rx1 should receive");
        let ev2 = rx2.recv().await.expect("rx2 should receive");
        assert_eq!(ev1.kind, ev2.kind);
        assert_eq!(ev1.sequence, ev2.sequence);
    }

    #[tokio::test]
    async fn test_event_stream_subscriber_count() {
        let stream = WebSocketEventStream::new(16);
        assert_eq!(stream.subscriber_count(), 0);
        let _rx1 = stream.subscribe();
        let _rx2 = stream.subscribe();
        assert_eq!(stream.subscriber_count(), 2);
    }

    #[tokio::test]
    async fn test_event_stream_alert() {
        let stream = WebSocketEventStream::new(16);
        let mut rx = stream.subscribe();

        stream
            .emit_alert("warning", "Primary source degraded")
            .await
            .expect("emit_alert should succeed");

        let event = rx.recv().await.expect("Should receive alert event");
        assert_eq!(event.kind, "alert");
        let payload = event.payload.as_object().expect("payload should be object");
        assert_eq!(payload["severity"], "warning");
    }
}
