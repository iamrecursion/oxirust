#![allow(dead_code)]
//! Notification subsystem for playlist events.
//!
//! This module provides an in-process pub/sub mechanism that fires notifications
//! when important playlist lifecycle events occur (item added, removed, started,
//! ended, error, etc.). Subscribers register callbacks and receive typed events
//! that they can log, forward to external systems, or act upon.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime};

// ---------------------------------------------------------------------------
// Event kinds
// ---------------------------------------------------------------------------

/// Kinds of playlist events that can trigger notifications.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlaylistEventKind {
    /// A new item was added to the playlist.
    ItemAdded,
    /// An item was removed from the playlist.
    ItemRemoved,
    /// Playback of an item started.
    ItemStarted,
    /// Playback of an item ended normally.
    ItemEnded,
    /// An error occurred during playback.
    PlaybackError,
    /// The playlist was loaded or reloaded.
    PlaylistLoaded,
    /// The playlist was cleared.
    PlaylistCleared,
    /// The playlist loop restarted from the beginning.
    LoopRestarted,
    /// A schedule conflict was detected.
    ScheduleConflict,
    /// Failover to backup content was triggered.
    FailoverTriggered,
}

/// A playlist notification event carrying context about what happened.
#[derive(Debug, Clone)]
pub struct PlaylistEvent {
    /// The kind of event.
    pub kind: PlaylistEventKind,
    /// Human-readable message.
    pub message: String,
    /// Identifier of the affected playlist (if applicable).
    pub playlist_id: Option<String>,
    /// Identifier of the affected item (if applicable).
    pub item_id: Option<String>,
    /// Timestamp when the event was created.
    pub timestamp: SystemTime,
    /// Severity level (0 = informational, 1 = warning, 2 = error).
    pub severity: u8,
}

impl PlaylistEvent {
    /// Create a new event.
    pub fn new(kind: PlaylistEventKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            playlist_id: None,
            item_id: None,
            timestamp: SystemTime::now(),
            severity: 0,
        }
    }

    /// Attach a playlist identifier.
    pub fn with_playlist_id(mut self, id: impl Into<String>) -> Self {
        self.playlist_id = Some(id.into());
        self
    }

    /// Attach an item identifier.
    pub fn with_item_id(mut self, id: impl Into<String>) -> Self {
        self.item_id = Some(id.into());
        self
    }

    /// Set severity level.
    pub fn with_severity(mut self, level: u8) -> Self {
        self.severity = level;
        self
    }
}

// ---------------------------------------------------------------------------
// Subscriber
// ---------------------------------------------------------------------------

/// Unique identifier for a subscriber registration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SubscriberId(u64);

impl std::fmt::Display for SubscriberId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "sub-{}", self.0)
    }
}

/// Static counter for generating unique subscriber IDs.
static NEXT_SUBSCRIBER_ID: AtomicU64 = AtomicU64::new(1);

/// A registered subscriber that collects events matching its filter.
#[derive(Debug, Clone)]
pub struct EventSubscriber {
    /// Unique identifier.
    pub id: SubscriberId,
    /// Human-readable label.
    pub label: String,
    /// Set of event kinds this subscriber is interested in.
    /// Empty means "all events".
    pub filter: Vec<PlaylistEventKind>,
    /// Minimum severity to receive (0 = all).
    pub min_severity: u8,
    /// Collected events (acts as a simple mailbox).
    inbox: Vec<PlaylistEvent>,
}

impl EventSubscriber {
    /// Create a new subscriber that receives all events.
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            id: SubscriberId(NEXT_SUBSCRIBER_ID.fetch_add(1, Ordering::Relaxed)),
            label: label.into(),
            filter: Vec::new(),
            min_severity: 0,
            inbox: Vec::new(),
        }
    }

    /// Restrict this subscriber to specific event kinds.
    pub fn with_filter(mut self, kinds: Vec<PlaylistEventKind>) -> Self {
        self.filter = kinds;
        self
    }

    /// Only receive events at or above the given severity.
    pub fn with_min_severity(mut self, level: u8) -> Self {
        self.min_severity = level;
        self
    }

    /// Check whether this subscriber accepts the given event.
    pub fn accepts(&self, event: &PlaylistEvent) -> bool {
        if event.severity < self.min_severity {
            return false;
        }
        if self.filter.is_empty() {
            return true;
        }
        self.filter.contains(&event.kind)
    }

    /// Deliver an event to this subscriber's inbox.
    pub fn deliver(&mut self, event: PlaylistEvent) {
        if self.accepts(&event) {
            self.inbox.push(event);
        }
    }

    /// Drain all events from the inbox.
    pub fn drain(&mut self) -> Vec<PlaylistEvent> {
        std::mem::take(&mut self.inbox)
    }

    /// Number of pending events.
    pub fn pending_count(&self) -> usize {
        self.inbox.len()
    }
}

// ---------------------------------------------------------------------------
// Notification hub
// ---------------------------------------------------------------------------

/// Central hub that dispatches playlist events to subscribers.
#[derive(Debug, Clone)]
pub struct NotificationHub {
    /// Registered subscribers keyed by ID.
    subscribers: HashMap<SubscriberId, EventSubscriber>,
    /// Rolling event history (bounded).
    history: Vec<PlaylistEvent>,
    /// Maximum history length.
    max_history: usize,
    /// Total events dispatched.
    total_dispatched: u64,
}

impl NotificationHub {
    /// Create a new notification hub.
    pub fn new() -> Self {
        Self {
            subscribers: HashMap::new(),
            history: Vec::new(),
            max_history: 1000,
            total_dispatched: 0,
        }
    }

    /// Set the maximum event history length.
    pub fn with_max_history(mut self, max: usize) -> Self {
        self.max_history = max;
        self
    }

    /// Register a subscriber and return its ID.
    pub fn subscribe(&mut self, sub: EventSubscriber) -> SubscriberId {
        let id = sub.id;
        self.subscribers.insert(id, sub);
        id
    }

    /// Unregister a subscriber.
    pub fn unsubscribe(&mut self, id: SubscriberId) -> bool {
        self.subscribers.remove(&id).is_some()
    }

    /// Number of active subscribers.
    pub fn subscriber_count(&self) -> usize {
        self.subscribers.len()
    }

    /// Dispatch an event to all matching subscribers.
    pub fn dispatch(&mut self, event: PlaylistEvent) {
        self.total_dispatched += 1;
        for sub in self.subscribers.values_mut() {
            sub.deliver(event.clone());
        }
        self.history.push(event);
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }
    }

    /// Shorthand to dispatch a simple informational event.
    pub fn notify(&mut self, kind: PlaylistEventKind, message: impl Into<String>) {
        self.dispatch(PlaylistEvent::new(kind, message));
    }

    /// Total events dispatched since creation.
    pub fn total_dispatched(&self) -> u64 {
        self.total_dispatched
    }

    /// Drain events from a specific subscriber's inbox.
    pub fn drain_subscriber(&mut self, id: SubscriberId) -> Vec<PlaylistEvent> {
        self.subscribers
            .get_mut(&id)
            .map(|s| s.drain())
            .unwrap_or_default()
    }

    /// Return the most recent `n` events from the history.
    pub fn recent_history(&self, n: usize) -> &[PlaylistEvent] {
        let start = self.history.len().saturating_sub(n);
        &self.history[start..]
    }

    /// Count events in history matching the given kind.
    pub fn count_by_kind(&self, kind: PlaylistEventKind) -> usize {
        self.history.iter().filter(|e| e.kind == kind).count()
    }

    /// Return the elapsed time since the last event, if any.
    pub fn time_since_last_event(&self) -> Option<Duration> {
        self.history.last().and_then(|e| e.timestamp.elapsed().ok())
    }
}

impl Default for NotificationHub {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_creation() {
        let event = PlaylistEvent::new(PlaylistEventKind::ItemAdded, "added clip");
        assert_eq!(event.kind, PlaylistEventKind::ItemAdded);
        assert_eq!(event.message, "added clip");
        assert_eq!(event.severity, 0);
    }

    #[test]
    fn test_event_builder() {
        let event = PlaylistEvent::new(PlaylistEventKind::PlaybackError, "fail")
            .with_playlist_id("pl1")
            .with_item_id("item42")
            .with_severity(2);
        assert_eq!(event.playlist_id.as_deref(), Some("pl1"));
        assert_eq!(event.item_id.as_deref(), Some("item42"));
        assert_eq!(event.severity, 2);
    }

    #[test]
    fn test_subscriber_accepts_all() {
        let sub = EventSubscriber::new("all");
        let event = PlaylistEvent::new(PlaylistEventKind::ItemStarted, "x");
        assert!(sub.accepts(&event));
    }

    #[test]
    fn test_subscriber_filter() {
        let sub =
            EventSubscriber::new("errors_only").with_filter(vec![PlaylistEventKind::PlaybackError]);
        let ok_event = PlaylistEvent::new(PlaylistEventKind::ItemStarted, "ok");
        let err_event = PlaylistEvent::new(PlaylistEventKind::PlaybackError, "err");
        assert!(!sub.accepts(&ok_event));
        assert!(sub.accepts(&err_event));
    }

    #[test]
    fn test_subscriber_severity_filter() {
        let sub = EventSubscriber::new("warn+").with_min_severity(1);
        let info = PlaylistEvent::new(PlaylistEventKind::ItemAdded, "lo");
        let warn = PlaylistEvent::new(PlaylistEventKind::ItemAdded, "hi").with_severity(1);
        assert!(!sub.accepts(&info));
        assert!(sub.accepts(&warn));
    }

    #[test]
    fn test_subscriber_deliver_and_drain() {
        let mut sub = EventSubscriber::new("x");
        sub.deliver(PlaylistEvent::new(PlaylistEventKind::ItemAdded, "a"));
        sub.deliver(PlaylistEvent::new(PlaylistEventKind::ItemRemoved, "b"));
        assert_eq!(sub.pending_count(), 2);
        let events = sub.drain();
        assert_eq!(events.len(), 2);
        assert_eq!(sub.pending_count(), 0);
    }

    #[test]
    fn test_hub_subscribe_unsubscribe() {
        let mut hub = NotificationHub::new();
        let id = hub.subscribe(EventSubscriber::new("s1"));
        assert_eq!(hub.subscriber_count(), 1);
        assert!(hub.unsubscribe(id));
        assert_eq!(hub.subscriber_count(), 0);
        assert!(!hub.unsubscribe(id)); // already removed
    }

    #[test]
    fn test_hub_dispatch_reaches_subscribers() {
        let mut hub = NotificationHub::new();
        let id = hub.subscribe(EventSubscriber::new("s1"));
        hub.dispatch(PlaylistEvent::new(PlaylistEventKind::ItemStarted, "go"));
        let events = hub.drain_subscriber(id);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].kind, PlaylistEventKind::ItemStarted);
    }

    #[test]
    fn test_hub_notify_shorthand() {
        let mut hub = NotificationHub::new();
        let id = hub.subscribe(EventSubscriber::new("s"));
        hub.notify(PlaylistEventKind::LoopRestarted, "loop");
        assert_eq!(hub.total_dispatched(), 1);
        assert_eq!(hub.drain_subscriber(id).len(), 1);
    }

    #[test]
    fn test_hub_history_capped() {
        let mut hub = NotificationHub::new().with_max_history(3);
        for i in 0..5 {
            hub.notify(PlaylistEventKind::ItemAdded, format!("e{i}"));
        }
        assert_eq!(hub.recent_history(10).len(), 3);
    }

    #[test]
    fn test_hub_count_by_kind() {
        let mut hub = NotificationHub::new();
        hub.notify(PlaylistEventKind::ItemAdded, "a");
        hub.notify(PlaylistEventKind::ItemAdded, "b");
        hub.notify(PlaylistEventKind::ItemRemoved, "c");
        assert_eq!(hub.count_by_kind(PlaylistEventKind::ItemAdded), 2);
        assert_eq!(hub.count_by_kind(PlaylistEventKind::ItemRemoved), 1);
    }

    #[test]
    fn test_hub_default() {
        let hub = NotificationHub::default();
        assert_eq!(hub.subscriber_count(), 0);
        assert_eq!(hub.total_dispatched(), 0);
    }

    #[test]
    fn test_subscriber_id_display() {
        let sub = EventSubscriber::new("test");
        let display = format!("{}", sub.id);
        assert!(display.starts_with("sub-"));
    }
}
