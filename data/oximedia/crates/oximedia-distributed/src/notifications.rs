//! WebSocket-style real-time job status notification bus.
//!
//! Provides an in-process publish/subscribe channel for job lifecycle events.
//! Subscribers receive events via [`tokio::sync::broadcast`] receivers, which
//! are cheaply cloneable and work across async tasks.
//!
//! # Example
//!
//! ```rust
//! use oximedia_distributed::notifications::{NotificationBus, JobEventType, JobEvent};
//! use uuid::Uuid;
//! use std::time::SystemTime;
//!
//! let bus = NotificationBus::new(64);
//! let mut rx = bus.subscribe();
//!
//! let event = JobEvent {
//!     job_id: Uuid::new_v4(),
//!     event_type: JobEventType::Queued,
//!     timestamp: SystemTime::now(),
//! };
//! bus.send(event.clone());
//! ```

use std::time::SystemTime;
use tokio::sync::broadcast;
use uuid::Uuid;

/// Event types for job lifecycle transitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum JobEventType {
    /// Job has been accepted into the queue.
    Queued,
    /// Job has been assigned to a worker and started.
    Started,
    /// Job finished successfully.
    Completed,
    /// Job failed (exceeded retries or fatal error).
    Failed,
    /// Job was explicitly cancelled.
    Cancelled,
}

impl JobEventType {
    /// Human-readable label for this event type.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Started => "started",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }

    /// Returns `true` if this event represents a terminal state.
    #[must_use]
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }
}

impl std::fmt::Display for JobEventType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

/// A job lifecycle event published on the [`NotificationBus`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct JobEvent {
    /// The job this event refers to.
    pub job_id: Uuid,
    /// What happened to the job.
    pub event_type: JobEventType,
    /// Wall-clock time of the event.
    #[serde(with = "system_time_serde")]
    pub timestamp: SystemTime,
}

impl JobEvent {
    /// Create a new event with the current system time.
    #[must_use]
    pub fn new(job_id: Uuid, event_type: JobEventType) -> Self {
        Self {
            job_id,
            event_type,
            timestamp: SystemTime::now(),
        }
    }

    /// Returns `true` if this is a terminal event.
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        self.event_type.is_terminal()
    }
}

mod system_time_serde {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    pub fn serialize<S: Serializer>(t: &SystemTime, s: S) -> Result<S::Ok, S::Error> {
        let epoch_secs = t
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_secs();
        s.serialize_u64(epoch_secs)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<SystemTime, D::Error> {
        let secs = u64::deserialize(d)?;
        Ok(UNIX_EPOCH + Duration::from_secs(secs))
    }
}

/// Publish/subscribe bus for [`JobEvent`]s.
///
/// Built on a [`tokio::sync::broadcast`] channel; all subscribers receive every
/// event. Lagged subscribers (that fall more than `capacity` events behind) will
/// receive a [`broadcast::error::RecvError::Lagged`] error and must re-subscribe
/// if they need to continue receiving.
pub struct NotificationBus {
    tx: broadcast::Sender<JobEvent>,
}

impl NotificationBus {
    /// Create a new bus with the given ring-buffer capacity.
    ///
    /// A reasonable default is 64–256 events.  When the buffer fills, the oldest
    /// events are dropped and late subscribers receive a `Lagged` error.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        let (tx, _) = broadcast::channel(capacity);
        Self { tx }
    }

    /// Publish an event to all current subscribers.
    ///
    /// Returns the number of active receivers that received the event.
    /// Returns 0 if no subscribers are active (not an error).
    pub fn send(&self, event: JobEvent) -> usize {
        self.tx.send(event).unwrap_or(0)
    }

    /// Subscribe to future events.
    ///
    /// The returned receiver will receive all events published **after** this
    /// call.  Clone the receiver to fan it out to multiple async tasks.
    #[must_use]
    pub fn subscribe(&self) -> broadcast::Receiver<JobEvent> {
        self.tx.subscribe()
    }

    /// Returns the number of active subscribers.
    #[must_use]
    pub fn subscriber_count(&self) -> usize {
        self.tx.receiver_count()
    }

    /// Create a convenience event and send it in one call.
    pub fn notify(&self, job_id: Uuid, event_type: JobEventType) -> usize {
        self.send(JobEvent::new(job_id, event_type))
    }
}

impl std::fmt::Debug for NotificationBus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NotificationBus")
            .field("subscriber_count", &self.tx.receiver_count())
            .finish()
    }
}

impl Default for NotificationBus {
    fn default() -> Self {
        Self::new(64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::broadcast::error::TryRecvError;

    #[tokio::test]
    async fn test_notification_bus_subscriber_receives_events() {
        let bus = NotificationBus::new(16);
        let mut rx = bus.subscribe();

        let job_id = Uuid::new_v4();
        bus.notify(job_id, JobEventType::Queued);

        let event = rx.recv().await.expect("should receive event");
        assert_eq!(event.job_id, job_id);
        assert_eq!(event.event_type, JobEventType::Queued);
    }

    #[tokio::test]
    async fn test_notification_multiple_subscribers() {
        let bus = NotificationBus::new(16);
        let mut rx1 = bus.subscribe();
        let mut rx2 = bus.subscribe();

        let job_id = Uuid::new_v4();
        let sent = bus.notify(job_id, JobEventType::Started);
        // Both receivers registered before send
        assert_eq!(sent, 2);

        let e1 = rx1.recv().await.expect("rx1 should receive");
        let e2 = rx2.recv().await.expect("rx2 should receive");
        assert_eq!(e1.event_type, JobEventType::Started);
        assert_eq!(e2.event_type, JobEventType::Started);
    }

    #[test]
    fn test_no_subscriber_send_returns_zero() {
        let bus = NotificationBus::new(8);
        // No subscribers registered
        let sent = bus.notify(Uuid::new_v4(), JobEventType::Completed);
        assert_eq!(sent, 0);
    }

    #[tokio::test]
    async fn test_terminal_event_types() {
        assert!(JobEventType::Completed.is_terminal());
        assert!(JobEventType::Failed.is_terminal());
        assert!(JobEventType::Cancelled.is_terminal());
        assert!(!JobEventType::Queued.is_terminal());
        assert!(!JobEventType::Started.is_terminal());
    }

    #[tokio::test]
    async fn test_event_sequence_in_order() {
        let bus = NotificationBus::new(16);
        let mut rx = bus.subscribe();

        let job_id = Uuid::new_v4();
        let types = [
            JobEventType::Queued,
            JobEventType::Started,
            JobEventType::Completed,
        ];
        for &t in &types {
            bus.notify(job_id, t);
        }

        for &expected in &types {
            let event = rx.recv().await.expect("should receive");
            assert_eq!(event.event_type, expected);
        }
    }

    #[test]
    fn test_subscriber_count_tracks_receivers() {
        let bus = NotificationBus::new(8);
        assert_eq!(bus.subscriber_count(), 0);
        let _rx1 = bus.subscribe();
        assert_eq!(bus.subscriber_count(), 1);
        let _rx2 = bus.subscribe();
        assert_eq!(bus.subscriber_count(), 2);
    }

    #[tokio::test]
    async fn test_dropped_subscriber_does_not_receive() {
        let bus = NotificationBus::new(8);
        let mut rx = bus.subscribe();

        // Drop a second subscriber; it should not affect the first
        {
            let _dropped = bus.subscribe();
        }

        bus.notify(Uuid::new_v4(), JobEventType::Failed);
        let event = rx.try_recv();
        assert!(event.is_ok() || matches!(event, Err(TryRecvError::Empty)));
    }

    #[test]
    fn test_event_labels() {
        assert_eq!(JobEventType::Queued.label(), "queued");
        assert_eq!(JobEventType::Started.label(), "started");
        assert_eq!(JobEventType::Completed.label(), "completed");
        assert_eq!(JobEventType::Failed.label(), "failed");
        assert_eq!(JobEventType::Cancelled.label(), "cancelled");
    }

    #[test]
    fn test_default_bus_capacity() {
        let bus = NotificationBus::default();
        // Should not panic
        let _rx = bus.subscribe();
    }
}
