//! Publish/subscribe event bus for workflow task communication.
//!
//! Provides a typed, thread-safe event bus that enables decoupled
//! communication between workflow tasks. Tasks can publish events
//! by topic, and other tasks can subscribe to topics with optional
//! filters. Supports event history, topic-based routing, and
//! dead letter tracking for undelivered events.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Unique identifier for a subscription.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SubscriptionId(u64);

impl SubscriptionId {
    /// Get the raw numeric ID.
    #[must_use]
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for SubscriptionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "sub-{}", self.0)
    }
}

/// An event published on the bus.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusEvent {
    /// Event topic (e.g. "task.completed", "workflow.error").
    pub topic: String,
    /// Source identifier (task ID, workflow ID, etc.).
    pub source: String,
    /// Event payload as JSON value.
    pub payload: serde_json::Value,
    /// Timestamp in milliseconds since epoch.
    pub timestamp_ms: u64,
    /// Optional correlation ID for tracing related events.
    pub correlation_id: Option<String>,
    /// Event metadata.
    pub metadata: HashMap<String, String>,
}

impl BusEvent {
    /// Create a new event.
    #[must_use]
    pub fn new(
        topic: impl Into<String>,
        source: impl Into<String>,
        payload: serde_json::Value,
        timestamp_ms: u64,
    ) -> Self {
        Self {
            topic: topic.into(),
            source: source.into(),
            payload,
            timestamp_ms,
            correlation_id: None,
            metadata: HashMap::new(),
        }
    }

    /// Set correlation ID.
    #[must_use]
    pub fn with_correlation_id(mut self, id: impl Into<String>) -> Self {
        self.correlation_id = Some(id.into());
        self
    }

    /// Add metadata.
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Filter criteria for subscriptions.
#[derive(Debug, Clone)]
pub struct EventFilter {
    /// Required source prefix (if set, event source must start with this).
    pub source_prefix: Option<String>,
    /// Required metadata keys and values (all must match).
    pub metadata_match: HashMap<String, String>,
    /// Minimum payload field value (for numeric comparisons).
    /// Format: ("field_name", minimum_value).
    pub min_values: Vec<(String, f64)>,
}

impl Default for EventFilter {
    fn default() -> Self {
        Self {
            source_prefix: None,
            metadata_match: HashMap::new(),
            min_values: Vec::new(),
        }
    }
}

impl EventFilter {
    /// Create an empty filter (matches everything).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Filter by source prefix.
    #[must_use]
    pub fn with_source_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.source_prefix = Some(prefix.into());
        self
    }

    /// Require a metadata key-value match.
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata_match.insert(key.into(), value.into());
        self
    }

    /// Require a minimum value for a payload field.
    #[must_use]
    pub fn with_min_value(mut self, field: impl Into<String>, min: f64) -> Self {
        self.min_values.push((field.into(), min));
        self
    }

    /// Check if an event matches this filter.
    #[must_use]
    pub fn matches(&self, event: &BusEvent) -> bool {
        // Check source prefix
        if let Some(ref prefix) = self.source_prefix {
            if !event.source.starts_with(prefix) {
                return false;
            }
        }

        // Check metadata
        for (key, expected) in &self.metadata_match {
            match event.metadata.get(key) {
                Some(actual) if actual == expected => {}
                _ => return false,
            }
        }

        // Check minimum values in payload
        for (field, min_val) in &self.min_values {
            if let Some(val) = event.payload.get(field).and_then(|v| v.as_f64()) {
                if val < *min_val {
                    return false;
                }
            } else {
                return false; // field not found or not a number
            }
        }

        true
    }
}

/// Internal subscription record.
#[derive(Debug, Clone)]
struct Subscription {
    id: SubscriptionId,
    topic_pattern: String,
    filter: EventFilter,
    /// Delivered events for this subscription.
    delivered: Vec<BusEvent>,
    /// Maximum events to retain (0 = unlimited).
    max_retained: usize,
    /// Whether this subscription is active.
    active: bool,
}

impl Subscription {
    /// Check if the subscription's topic pattern matches the event topic.
    fn topic_matches(&self, topic: &str) -> bool {
        if self.topic_pattern == "*" {
            return true;
        }
        if self.topic_pattern.ends_with(".*") {
            let prefix = &self.topic_pattern[..self.topic_pattern.len() - 2];
            return topic.starts_with(prefix);
        }
        self.topic_pattern == topic
    }
}

/// Dead letter entry for events that had no subscribers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeadLetterEntry {
    /// The undelivered event.
    pub event: BusEvent,
    /// Reason it was undelivered.
    pub reason: String,
}

/// Thread-safe event bus for publish/subscribe communication.
#[derive(Debug, Clone)]
pub struct EventBus {
    inner: Arc<Mutex<EventBusInner>>,
}

#[derive(Debug)]
struct EventBusInner {
    subscriptions: Vec<Subscription>,
    next_sub_id: u64,
    /// All published events (for replay/history).
    event_history: Vec<BusEvent>,
    /// Events with no matching subscribers.
    dead_letters: Vec<DeadLetterEntry>,
    /// Maximum history size (0 = unlimited).
    max_history: usize,
    /// Maximum dead letter size.
    max_dead_letters: usize,
    /// Total events published.
    total_published: u64,
    /// Total deliveries across all subscriptions.
    total_delivered: u64,
}

/// Statistics for the event bus.
#[derive(Debug, Clone)]
pub struct EventBusStats {
    /// Number of active subscriptions.
    pub active_subscriptions: usize,
    /// Total subscriptions (including inactive).
    pub total_subscriptions: usize,
    /// Total events published.
    pub total_published: u64,
    /// Total deliveries across all subscriptions.
    pub total_delivered: u64,
    /// Events in history.
    pub history_size: usize,
    /// Dead letter count.
    pub dead_letter_count: usize,
}

/// Configuration for the event bus.
#[derive(Debug, Clone)]
pub struct EventBusConfig {
    /// Maximum event history size (0 = unlimited).
    pub max_history: usize,
    /// Maximum dead letter queue size.
    pub max_dead_letters: usize,
    /// Default max retained events per subscription.
    pub default_max_retained: usize,
}

impl Default for EventBusConfig {
    fn default() -> Self {
        Self {
            max_history: 10_000,
            max_dead_letters: 1_000,
            default_max_retained: 100,
        }
    }
}

impl EventBus {
    /// Create a new event bus with default config.
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(EventBusConfig::default())
    }

    /// Create a new event bus with custom config.
    #[must_use]
    pub fn with_config(config: EventBusConfig) -> Self {
        Self {
            inner: Arc::new(Mutex::new(EventBusInner {
                subscriptions: Vec::new(),
                next_sub_id: 1,
                event_history: Vec::new(),
                dead_letters: Vec::new(),
                max_history: config.max_history,
                max_dead_letters: config.max_dead_letters,
                total_published: 0,
                total_delivered: 0,
            })),
        }
    }

    /// Subscribe to a topic pattern.
    ///
    /// Topic patterns:
    /// - Exact: `"task.completed"` matches only `"task.completed"`
    /// - Wildcard: `"task.*"` matches `"task.completed"`, `"task.failed"`, etc.
    /// - Catch-all: `"*"` matches everything
    ///
    /// Returns a subscription ID that can be used to retrieve delivered events
    /// or unsubscribe.
    pub fn subscribe(
        &self,
        topic_pattern: impl Into<String>,
        filter: EventFilter,
    ) -> SubscriptionId {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let id = SubscriptionId(inner.next_sub_id);
        inner.next_sub_id += 1;

        inner.subscriptions.push(Subscription {
            id,
            topic_pattern: topic_pattern.into(),
            filter,
            delivered: Vec::new(),
            max_retained: 100,
            active: true,
        });

        id
    }

    /// Subscribe with a custom max retained events count.
    pub fn subscribe_with_retention(
        &self,
        topic_pattern: impl Into<String>,
        filter: EventFilter,
        max_retained: usize,
    ) -> SubscriptionId {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let id = SubscriptionId(inner.next_sub_id);
        inner.next_sub_id += 1;

        inner.subscriptions.push(Subscription {
            id,
            topic_pattern: topic_pattern.into(),
            filter,
            delivered: Vec::new(),
            max_retained,
            active: true,
        });

        id
    }

    /// Unsubscribe (marks subscription as inactive).
    pub fn unsubscribe(&self, sub_id: SubscriptionId) -> bool {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(sub) = inner.subscriptions.iter_mut().find(|s| s.id == sub_id) {
            sub.active = false;
            true
        } else {
            false
        }
    }

    /// Publish an event to the bus.
    ///
    /// Returns the number of subscriptions that received the event.
    pub fn publish(&self, event: BusEvent) -> usize {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        inner.total_published += 1;
        let mut delivery_count = 0usize;

        // Deliver to matching active subscriptions
        for sub in inner.subscriptions.iter_mut() {
            if !sub.active {
                continue;
            }
            if !sub.topic_matches(&event.topic) {
                continue;
            }
            if !sub.filter.matches(&event) {
                continue;
            }

            sub.delivered.push(event.clone());
            delivery_count += 1;

            // Trim retained events if over limit
            if sub.max_retained > 0 && sub.delivered.len() > sub.max_retained {
                let excess = sub.delivered.len() - sub.max_retained;
                sub.delivered.drain(..excess);
            }
        }

        inner.total_delivered += delivery_count as u64;

        // Track dead letters
        if delivery_count == 0 {
            inner.dead_letters.push(DeadLetterEntry {
                event: event.clone(),
                reason: "no matching subscribers".to_string(),
            });
            if inner.max_dead_letters > 0 && inner.dead_letters.len() > inner.max_dead_letters {
                inner.dead_letters.remove(0);
            }
        }

        // Add to history
        inner.event_history.push(event);
        if inner.max_history > 0 && inner.event_history.len() > inner.max_history {
            inner.event_history.remove(0);
        }

        delivery_count
    }

    /// Get events delivered to a subscription.
    #[must_use]
    pub fn get_events(&self, sub_id: SubscriptionId) -> Vec<BusEvent> {
        let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        inner
            .subscriptions
            .iter()
            .find(|s| s.id == sub_id)
            .map_or_else(Vec::new, |s| s.delivered.clone())
    }

    /// Drain (consume) events from a subscription, returning and clearing them.
    pub fn drain_events(&self, sub_id: SubscriptionId) -> Vec<BusEvent> {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        inner
            .subscriptions
            .iter_mut()
            .find(|s| s.id == sub_id)
            .map_or_else(Vec::new, |s| std::mem::take(&mut s.delivered))
    }

    /// Get dead letters.
    #[must_use]
    pub fn dead_letters(&self) -> Vec<DeadLetterEntry> {
        let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        inner.dead_letters.clone()
    }

    /// Get event history.
    #[must_use]
    pub fn event_history(&self) -> Vec<BusEvent> {
        let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        inner.event_history.clone()
    }

    /// Get events from history matching a specific topic.
    #[must_use]
    pub fn events_by_topic(&self, topic: &str) -> Vec<BusEvent> {
        let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        inner
            .event_history
            .iter()
            .filter(|e| e.topic == topic)
            .cloned()
            .collect()
    }

    /// Get bus statistics.
    #[must_use]
    pub fn stats(&self) -> EventBusStats {
        let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        EventBusStats {
            active_subscriptions: inner.subscriptions.iter().filter(|s| s.active).count(),
            total_subscriptions: inner.subscriptions.len(),
            total_published: inner.total_published,
            total_delivered: inner.total_delivered,
            history_size: inner.event_history.len(),
            dead_letter_count: inner.dead_letters.len(),
        }
    }

    /// Clear all history, dead letters, and delivered events.
    pub fn clear(&self) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        inner.event_history.clear();
        inner.dead_letters.clear();
        for sub in &mut inner.subscriptions {
            sub.delivered.clear();
        }
    }

    /// Replay events from history to a new subscription.
    ///
    /// Returns the number of events replayed.
    pub fn replay_to(&self, sub_id: SubscriptionId) -> usize {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());

        // Find the subscription index
        let sub_idx = match inner.subscriptions.iter().position(|s| s.id == sub_id) {
            Some(idx) => idx,
            None => return 0,
        };

        // Clone history to avoid borrow conflicts
        let history = inner.event_history.clone();
        let mut count = 0;

        for event in &history {
            let sub = &inner.subscriptions[sub_idx];
            if sub.active && sub.topic_matches(&event.topic) && sub.filter.matches(event) {
                inner.subscriptions[sub_idx].delivered.push(event.clone());
                count += 1;
            }
        }

        count
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_event(topic: &str, source: &str) -> BusEvent {
        BusEvent::new(topic, source, serde_json::json!({}), 1000)
    }

    // --- BusEvent ---

    #[test]
    fn test_event_creation() {
        let event = make_event("task.done", "task-1");
        assert_eq!(event.topic, "task.done");
        assert_eq!(event.source, "task-1");
        assert_eq!(event.timestamp_ms, 1000);
        assert!(event.correlation_id.is_none());
    }

    #[test]
    fn test_event_with_correlation() {
        let event = make_event("task.done", "task-1").with_correlation_id("corr-123");
        assert_eq!(event.correlation_id, Some("corr-123".to_string()));
    }

    #[test]
    fn test_event_with_metadata() {
        let event = make_event("task.done", "task-1").with_metadata("workflow", "wf-1");
        assert_eq!(event.metadata.get("workflow"), Some(&"wf-1".to_string()));
    }

    // --- EventFilter ---

    #[test]
    fn test_empty_filter_matches_all() {
        let filter = EventFilter::new();
        let event = make_event("any.topic", "any-source");
        assert!(filter.matches(&event));
    }

    #[test]
    fn test_source_prefix_filter() {
        let filter = EventFilter::new().with_source_prefix("task-");
        assert!(filter.matches(&make_event("x", "task-1")));
        assert!(!filter.matches(&make_event("x", "workflow-1")));
    }

    #[test]
    fn test_metadata_filter() {
        let filter = EventFilter::new().with_metadata("env", "production");
        let event = make_event("x", "y").with_metadata("env", "production");
        assert!(filter.matches(&event));

        let event2 = make_event("x", "y").with_metadata("env", "staging");
        assert!(!filter.matches(&event2));
    }

    #[test]
    fn test_min_value_filter() {
        let filter = EventFilter::new().with_min_value("quality", 90.0);
        let event = BusEvent::new("x", "y", serde_json::json!({"quality": 95.0}), 0);
        assert!(filter.matches(&event));

        let event2 = BusEvent::new("x", "y", serde_json::json!({"quality": 80.0}), 0);
        assert!(!filter.matches(&event2));
    }

    #[test]
    fn test_min_value_missing_field() {
        let filter = EventFilter::new().with_min_value("quality", 90.0);
        let event = BusEvent::new("x", "y", serde_json::json!({"other": 95.0}), 0);
        assert!(!filter.matches(&event));
    }

    // --- EventBus core ---

    #[test]
    fn test_bus_subscribe_and_publish() {
        let bus = EventBus::new();
        let sub = bus.subscribe("task.done", EventFilter::new());
        let delivered = bus.publish(make_event("task.done", "task-1"));
        assert_eq!(delivered, 1);

        let events = bus.get_events(sub);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].source, "task-1");
    }

    #[test]
    fn test_bus_no_match() {
        let bus = EventBus::new();
        bus.subscribe("task.done", EventFilter::new());
        let delivered = bus.publish(make_event("workflow.started", "wf-1"));
        assert_eq!(delivered, 0);
    }

    #[test]
    fn test_bus_wildcard_topic() {
        let bus = EventBus::new();
        let sub = bus.subscribe("task.*", EventFilter::new());
        bus.publish(make_event("task.done", "t1"));
        bus.publish(make_event("task.failed", "t2"));
        bus.publish(make_event("workflow.started", "wf1"));

        let events = bus.get_events(sub);
        assert_eq!(events.len(), 2);
    }

    #[test]
    fn test_bus_catch_all() {
        let bus = EventBus::new();
        let sub = bus.subscribe("*", EventFilter::new());
        bus.publish(make_event("task.done", "t1"));
        bus.publish(make_event("workflow.started", "wf1"));

        let events = bus.get_events(sub);
        assert_eq!(events.len(), 2);
    }

    #[test]
    fn test_bus_multiple_subscribers() {
        let bus = EventBus::new();
        let sub1 = bus.subscribe("task.done", EventFilter::new());
        let sub2 = bus.subscribe("task.done", EventFilter::new());
        let sub3 = bus.subscribe("task.failed", EventFilter::new());

        let delivered = bus.publish(make_event("task.done", "t1"));
        assert_eq!(delivered, 2);
        assert_eq!(bus.get_events(sub1).len(), 1);
        assert_eq!(bus.get_events(sub2).len(), 1);
        assert_eq!(bus.get_events(sub3).len(), 0);
    }

    #[test]
    fn test_bus_unsubscribe() {
        let bus = EventBus::new();
        let sub = bus.subscribe("task.done", EventFilter::new());

        bus.publish(make_event("task.done", "t1"));
        assert_eq!(bus.get_events(sub).len(), 1);

        assert!(bus.unsubscribe(sub));
        bus.publish(make_event("task.done", "t2"));
        // No new events after unsubscribe
        assert_eq!(bus.get_events(sub).len(), 1);
    }

    #[test]
    fn test_bus_unsubscribe_nonexistent() {
        let bus = EventBus::new();
        assert!(!bus.unsubscribe(SubscriptionId(999)));
    }

    #[test]
    fn test_bus_drain_events() {
        let bus = EventBus::new();
        let sub = bus.subscribe("task.done", EventFilter::new());
        bus.publish(make_event("task.done", "t1"));
        bus.publish(make_event("task.done", "t2"));

        let drained = bus.drain_events(sub);
        assert_eq!(drained.len(), 2);
        assert_eq!(bus.get_events(sub).len(), 0);
    }

    // --- Dead letters ---

    #[test]
    fn test_dead_letters() {
        let bus = EventBus::new();
        // No subscribers, so event goes to dead letters
        bus.publish(make_event("orphan.event", "src"));
        let dead = bus.dead_letters();
        assert_eq!(dead.len(), 1);
        assert_eq!(dead[0].event.topic, "orphan.event");
    }

    #[test]
    fn test_dead_letters_not_for_delivered() {
        let bus = EventBus::new();
        bus.subscribe("task.done", EventFilter::new());
        bus.publish(make_event("task.done", "t1"));
        assert!(bus.dead_letters().is_empty());
    }

    // --- History ---

    #[test]
    fn test_event_history() {
        let bus = EventBus::new();
        bus.publish(make_event("a", "s1"));
        bus.publish(make_event("b", "s2"));

        let history = bus.event_history();
        assert_eq!(history.len(), 2);
    }

    #[test]
    fn test_events_by_topic() {
        let bus = EventBus::new();
        bus.publish(make_event("task.done", "t1"));
        bus.publish(make_event("task.failed", "t2"));
        bus.publish(make_event("task.done", "t3"));

        let done_events = bus.events_by_topic("task.done");
        assert_eq!(done_events.len(), 2);
    }

    // --- Stats ---

    #[test]
    fn test_stats() {
        let bus = EventBus::new();
        let _sub1 = bus.subscribe("task.done", EventFilter::new());
        let sub2 = bus.subscribe("task.done", EventFilter::new());
        bus.unsubscribe(sub2);

        bus.publish(make_event("task.done", "t1"));
        bus.publish(make_event("orphan", "s1"));

        let stats = bus.stats();
        assert_eq!(stats.active_subscriptions, 1);
        assert_eq!(stats.total_subscriptions, 2);
        assert_eq!(stats.total_published, 2);
        assert_eq!(stats.total_delivered, 1);
        assert_eq!(stats.history_size, 2);
        assert_eq!(stats.dead_letter_count, 1);
    }

    // --- Retention limits ---

    #[test]
    fn test_retention_limit() {
        let bus = EventBus::new();
        let sub = bus.subscribe_with_retention("task.done", EventFilter::new(), 3);

        for i in 0..5 {
            bus.publish(make_event("task.done", &format!("t{i}")));
        }

        let events = bus.get_events(sub);
        assert_eq!(events.len(), 3);
        // Should have the latest 3
        assert_eq!(events[0].source, "t2");
        assert_eq!(events[2].source, "t4");
    }

    // --- Clear ---

    #[test]
    fn test_clear() {
        let bus = EventBus::new();
        let sub = bus.subscribe("task.done", EventFilter::new());
        bus.publish(make_event("task.done", "t1"));
        bus.publish(make_event("orphan", "s1"));

        bus.clear();
        assert!(bus.event_history().is_empty());
        assert!(bus.dead_letters().is_empty());
        assert!(bus.get_events(sub).is_empty());
    }

    // --- Replay ---

    #[test]
    fn test_replay_to_subscription() {
        let bus = EventBus::new();

        // Publish before subscribing
        bus.publish(make_event("task.done", "t1"));
        bus.publish(make_event("task.done", "t2"));
        bus.publish(make_event("workflow.started", "wf1"));

        // Late subscriber
        let sub = bus.subscribe("task.done", EventFilter::new());
        assert_eq!(bus.get_events(sub).len(), 0);

        // Replay history
        let replayed = bus.replay_to(sub);
        assert_eq!(replayed, 2);
        assert_eq!(bus.get_events(sub).len(), 2);
    }

    // --- Filter combined with topic ---

    #[test]
    fn test_topic_and_filter_combined() {
        let bus = EventBus::new();
        let filter = EventFilter::new().with_source_prefix("encoder-");
        let sub = bus.subscribe("task.done", filter);

        bus.publish(make_event("task.done", "encoder-1"));
        bus.publish(make_event("task.done", "decoder-1"));

        let events = bus.get_events(sub);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].source, "encoder-1");
    }

    #[test]
    fn test_subscription_id_display() {
        let id = SubscriptionId(42);
        assert_eq!(id.to_string(), "sub-42");
        assert_eq!(id.as_u64(), 42);
    }

    // --- Thread safety ---

    #[test]
    fn test_bus_is_clone_and_send() {
        let bus = EventBus::new();
        let bus2 = bus.clone();
        let sub = bus.subscribe("test", EventFilter::new());
        bus2.publish(make_event("test", "src"));
        assert_eq!(bus.get_events(sub).len(), 1);
    }
}
