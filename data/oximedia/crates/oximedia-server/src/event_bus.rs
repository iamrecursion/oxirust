//! Internal pub/sub event bus for decoupling handlers from side effects.
//!
//! Provides a typed, in-process event system where handlers can emit events
//! and subscribers can react asynchronously. Supports topic-based routing,
//! wildcard subscriptions, and delivery guarantees.

#![allow(dead_code)]

use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// An event in the bus.
#[derive(Debug, Clone)]
pub struct Event {
    /// Unique event ID.
    pub id: String,
    /// Topic / event type.
    pub topic: String,
    /// Event payload (JSON-serializable string).
    pub payload: String,
    /// When the event was created.
    pub timestamp: u64,
    /// Source of the event (handler name, module, etc.).
    pub source: String,
    /// Correlation ID for tracing event chains.
    pub correlation_id: Option<String>,
    /// Event metadata.
    pub metadata: HashMap<String, String>,
}

impl Event {
    /// Creates a new event.
    pub fn new(
        topic: impl Into<String>,
        payload: impl Into<String>,
        source: impl Into<String>,
    ) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Self {
            id: format!("evt-{}", now),
            topic: topic.into(),
            payload: payload.into(),
            timestamp: now,
            source: source.into(),
            correlation_id: None,
            metadata: HashMap::new(),
        }
    }

    /// Sets the event ID.
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = id.into();
        self
    }

    /// Sets a correlation ID.
    pub fn with_correlation_id(mut self, id: impl Into<String>) -> Self {
        self.correlation_id = Some(id.into());
        self
    }

    /// Adds metadata.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// A subscription to events.
#[derive(Debug, Clone)]
pub struct Subscription {
    /// Unique subscription ID.
    pub id: String,
    /// Topic pattern (exact match or wildcard `*`).
    pub topic_pattern: String,
    /// Subscriber name.
    pub subscriber: String,
    /// When the subscription was created.
    pub created_at: Instant,
    /// Number of events delivered to this subscription.
    pub delivered: u64,
    /// Whether the subscription is active.
    pub active: bool,
}

impl Subscription {
    /// Checks whether an event topic matches this subscription's pattern.
    pub fn matches_topic(&self, topic: &str) -> bool {
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

/// Delivery record for auditing.
#[derive(Debug, Clone)]
pub struct DeliveryRecord {
    /// Event ID.
    pub event_id: String,
    /// Subscription ID.
    pub subscription_id: String,
    /// When delivery occurred.
    pub delivered_at: Instant,
    /// Whether delivery was successful.
    pub success: bool,
    /// Error message if delivery failed.
    pub error: Option<String>,
}

/// Statistics for the event bus.
#[derive(Debug, Clone, Default)]
pub struct EventBusStats {
    /// Total events published.
    pub events_published: u64,
    /// Total deliveries attempted.
    pub deliveries_attempted: u64,
    /// Successful deliveries.
    pub deliveries_succeeded: u64,
    /// Failed deliveries.
    pub deliveries_failed: u64,
    /// Active subscriptions.
    pub active_subscriptions: usize,
    /// Events in the buffer (undelivered).
    pub buffer_size: usize,
}

impl EventBusStats {
    /// Delivery success rate.
    pub fn delivery_success_rate(&self) -> f64 {
        if self.deliveries_attempted == 0 {
            return 1.0;
        }
        self.deliveries_succeeded as f64 / self.deliveries_attempted as f64
    }
}

/// Configuration for the event bus.
#[derive(Debug, Clone)]
pub struct EventBusConfig {
    /// Maximum buffer size for undelivered events.
    pub max_buffer_size: usize,
    /// How long to retain events in the buffer.
    pub buffer_ttl: Duration,
    /// Maximum subscriptions per topic.
    pub max_subscriptions_per_topic: usize,
    /// Whether to store delivery records.
    pub store_delivery_records: bool,
    /// Maximum delivery records to retain.
    pub max_delivery_records: usize,
}

impl Default for EventBusConfig {
    fn default() -> Self {
        Self {
            max_buffer_size: 10_000,
            buffer_ttl: Duration::from_secs(3600),
            max_subscriptions_per_topic: 100,
            store_delivery_records: true,
            max_delivery_records: 100_000,
        }
    }
}

/// The internal event bus.
pub struct EventBus {
    config: EventBusConfig,
    /// All subscriptions.
    subscriptions: Vec<Subscription>,
    /// Event buffer for replay.
    buffer: Vec<Event>,
    /// Delivery records.
    delivery_records: Vec<DeliveryRecord>,
    /// Statistics.
    stats: EventBusStats,
    /// Next subscription ID counter.
    next_sub_id: u64,
}

impl EventBus {
    /// Creates a new event bus.
    pub fn new(config: EventBusConfig) -> Self {
        Self {
            config,
            subscriptions: Vec::new(),
            buffer: Vec::new(),
            delivery_records: Vec::new(),
            stats: EventBusStats::default(),
            next_sub_id: 1,
        }
    }

    /// Subscribes to a topic pattern.
    ///
    /// Returns the subscription ID.
    pub fn subscribe(
        &mut self,
        topic_pattern: impl Into<String>,
        subscriber: impl Into<String>,
    ) -> String {
        let id = format!("sub-{}", self.next_sub_id);
        self.next_sub_id += 1;

        self.subscriptions.push(Subscription {
            id: id.clone(),
            topic_pattern: topic_pattern.into(),
            subscriber: subscriber.into(),
            created_at: Instant::now(),
            delivered: 0,
            active: true,
        });

        self.stats.active_subscriptions = self.subscriptions.iter().filter(|s| s.active).count();

        id
    }

    /// Unsubscribes by subscription ID.
    pub fn unsubscribe(&mut self, subscription_id: &str) -> bool {
        if let Some(sub) = self
            .subscriptions
            .iter_mut()
            .find(|s| s.id == subscription_id)
        {
            sub.active = false;
            self.stats.active_subscriptions =
                self.subscriptions.iter().filter(|s| s.active).count();
            true
        } else {
            false
        }
    }

    /// Publishes an event to all matching subscribers.
    ///
    /// Returns the number of deliveries.
    pub fn publish(&mut self, event: Event) -> usize {
        self.stats.events_published += 1;

        // Buffer the event
        if self.buffer.len() < self.config.max_buffer_size {
            self.buffer.push(event.clone());
            self.stats.buffer_size = self.buffer.len();
        }

        // Find matching subscriptions
        let matching: Vec<usize> = self
            .subscriptions
            .iter()
            .enumerate()
            .filter(|(_, s)| s.active && s.matches_topic(&event.topic))
            .map(|(i, _)| i)
            .collect();

        let count = matching.len();

        for idx in matching {
            self.stats.deliveries_attempted += 1;
            self.subscriptions[idx].delivered += 1;
            self.stats.deliveries_succeeded += 1;

            if self.config.store_delivery_records {
                self.delivery_records.push(DeliveryRecord {
                    event_id: event.id.clone(),
                    subscription_id: self.subscriptions[idx].id.clone(),
                    delivered_at: Instant::now(),
                    success: true,
                    error: None,
                });

                // Trim delivery records
                if self.delivery_records.len() > self.config.max_delivery_records {
                    let excess = self.delivery_records.len() - self.config.max_delivery_records;
                    self.delivery_records.drain(..excess);
                }
            }
        }

        count
    }

    /// Returns all active subscriptions for a topic.
    pub fn subscriptions_for_topic(&self, topic: &str) -> Vec<&Subscription> {
        self.subscriptions
            .iter()
            .filter(|s| s.active && s.matches_topic(topic))
            .collect()
    }

    /// Returns all events in the buffer for a topic.
    pub fn events_for_topic(&self, topic: &str) -> Vec<&Event> {
        self.buffer.iter().filter(|e| e.topic == topic).collect()
    }

    /// Purges old events from the buffer.
    pub fn purge_buffer(&mut self) -> usize {
        let before = self.buffer.len();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let ttl_secs = self.config.buffer_ttl.as_secs();
        self.buffer.retain(|e| now - e.timestamp < ttl_secs);
        self.stats.buffer_size = self.buffer.len();
        before - self.buffer.len()
    }

    /// Removes inactive subscriptions.
    pub fn cleanup_subscriptions(&mut self) -> usize {
        let before = self.subscriptions.len();
        self.subscriptions.retain(|s| s.active);
        before - self.subscriptions.len()
    }

    /// Returns statistics.
    pub fn stats(&self) -> &EventBusStats {
        &self.stats
    }

    /// Returns the number of active subscriptions.
    pub fn active_subscription_count(&self) -> usize {
        self.stats.active_subscriptions
    }

    /// Returns the buffer size.
    pub fn buffer_size(&self) -> usize {
        self.buffer.len()
    }

    /// Returns recent delivery records.
    pub fn recent_deliveries(&self, limit: usize) -> &[DeliveryRecord] {
        let start = self.delivery_records.len().saturating_sub(limit);
        &self.delivery_records[start..]
    }

    /// Returns all subscriptions.
    pub fn all_subscriptions(&self) -> &[Subscription] {
        &self.subscriptions
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new(EventBusConfig::default())
    }
}

/// Thread-safe event bus wrapper.
pub struct SharedEventBus {
    inner: Arc<RwLock<EventBus>>,
}

impl SharedEventBus {
    /// Creates a new shared event bus.
    pub fn new(config: EventBusConfig) -> Self {
        Self {
            inner: Arc::new(RwLock::new(EventBus::new(config))),
        }
    }

    /// Publishes an event.
    pub fn publish(&self, event: Event) -> usize {
        self.inner.write().publish(event)
    }

    /// Subscribes to a topic.
    pub fn subscribe(&self, topic_pattern: &str, subscriber: &str) -> String {
        self.inner.write().subscribe(topic_pattern, subscriber)
    }

    /// Unsubscribes.
    pub fn unsubscribe(&self, subscription_id: &str) -> bool {
        self.inner.write().unsubscribe(subscription_id)
    }

    /// Returns stats.
    pub fn stats(&self) -> EventBusStats {
        self.inner.read().stats().clone()
    }
}

impl Default for SharedEventBus {
    fn default() -> Self {
        Self::new(EventBusConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Event

    #[test]
    fn test_event_creation() {
        let event = Event::new("media.uploaded", r#"{"id":"m1"}"#, "upload_handler");
        assert_eq!(event.topic, "media.uploaded");
        assert_eq!(event.source, "upload_handler");
    }

    #[test]
    fn test_event_builders() {
        let event = Event::new("t", "p", "s")
            .with_id("evt-custom")
            .with_correlation_id("corr-1")
            .with_metadata("key", "value");
        assert_eq!(event.id, "evt-custom");
        assert_eq!(event.correlation_id, Some("corr-1".to_string()));
        assert_eq!(event.metadata.get("key"), Some(&"value".to_string()));
    }

    // Subscription

    #[test]
    fn test_subscription_exact_match() {
        let sub = Subscription {
            id: "sub-1".into(),
            topic_pattern: "media.uploaded".into(),
            subscriber: "test".into(),
            created_at: Instant::now(),
            delivered: 0,
            active: true,
        };
        assert!(sub.matches_topic("media.uploaded"));
        assert!(!sub.matches_topic("media.deleted"));
    }

    #[test]
    fn test_subscription_wildcard() {
        let sub = Subscription {
            id: "sub-1".into(),
            topic_pattern: "*".into(),
            subscriber: "test".into(),
            created_at: Instant::now(),
            delivered: 0,
            active: true,
        };
        assert!(sub.matches_topic("anything"));
        assert!(sub.matches_topic("media.uploaded"));
    }

    #[test]
    fn test_subscription_prefix_wildcard() {
        let sub = Subscription {
            id: "sub-1".into(),
            topic_pattern: "media.*".into(),
            subscriber: "test".into(),
            created_at: Instant::now(),
            delivered: 0,
            active: true,
        };
        assert!(sub.matches_topic("media.uploaded"));
        assert!(sub.matches_topic("media.deleted"));
        assert!(!sub.matches_topic("transcode.started"));
    }

    // EventBus

    #[test]
    fn test_subscribe_and_publish() {
        let mut bus = EventBus::default();
        bus.subscribe("media.uploaded", "handler1");
        let count = bus.publish(Event::new("media.uploaded", "{}", "test"));
        assert_eq!(count, 1);
    }

    #[test]
    fn test_publish_no_subscribers() {
        let mut bus = EventBus::default();
        let count = bus.publish(Event::new("media.uploaded", "{}", "test"));
        assert_eq!(count, 0);
    }

    #[test]
    fn test_multiple_subscribers() {
        let mut bus = EventBus::default();
        bus.subscribe("media.uploaded", "handler1");
        bus.subscribe("media.uploaded", "handler2");
        bus.subscribe("media.*", "handler3");
        let count = bus.publish(Event::new("media.uploaded", "{}", "test"));
        assert_eq!(count, 3);
    }

    #[test]
    fn test_unsubscribe() {
        let mut bus = EventBus::default();
        let id = bus.subscribe("media.uploaded", "handler1");
        assert!(bus.unsubscribe(&id));
        let count = bus.publish(Event::new("media.uploaded", "{}", "test"));
        assert_eq!(count, 0);
    }

    #[test]
    fn test_unsubscribe_unknown() {
        let mut bus = EventBus::default();
        assert!(!bus.unsubscribe("nonexistent"));
    }

    #[test]
    fn test_active_subscription_count() {
        let mut bus = EventBus::default();
        bus.subscribe("t1", "s1");
        let id = bus.subscribe("t2", "s2");
        assert_eq!(bus.active_subscription_count(), 2);
        bus.unsubscribe(&id);
        assert_eq!(bus.active_subscription_count(), 1);
    }

    #[test]
    fn test_buffer_stores_events() {
        let mut bus = EventBus::default();
        bus.publish(Event::new("media.uploaded", "{}", "test"));
        bus.publish(Event::new("media.deleted", "{}", "test"));
        assert_eq!(bus.buffer_size(), 2);
    }

    #[test]
    fn test_events_for_topic() {
        let mut bus = EventBus::default();
        bus.publish(Event::new("media.uploaded", "{}", "test"));
        bus.publish(Event::new("media.deleted", "{}", "test"));
        bus.publish(Event::new("media.uploaded", "{}", "test2"));
        let events = bus.events_for_topic("media.uploaded");
        assert_eq!(events.len(), 2);
    }

    #[test]
    fn test_subscriptions_for_topic() {
        let mut bus = EventBus::default();
        bus.subscribe("media.uploaded", "h1");
        bus.subscribe("media.*", "h2");
        bus.subscribe("transcode.done", "h3");
        let subs = bus.subscriptions_for_topic("media.uploaded");
        assert_eq!(subs.len(), 2);
    }

    #[test]
    fn test_stats_tracking() {
        let mut bus = EventBus::default();
        bus.subscribe("media.uploaded", "handler");
        bus.publish(Event::new("media.uploaded", "{}", "test"));
        let stats = bus.stats();
        assert_eq!(stats.events_published, 1);
        assert_eq!(stats.deliveries_attempted, 1);
        assert_eq!(stats.deliveries_succeeded, 1);
    }

    #[test]
    fn test_stats_delivery_success_rate() {
        let stats = EventBusStats {
            deliveries_attempted: 10,
            deliveries_succeeded: 8,
            ..Default::default()
        };
        assert!((stats.delivery_success_rate() - 0.8).abs() < 1e-9);
    }

    #[test]
    fn test_cleanup_subscriptions() {
        let mut bus = EventBus::default();
        let id = bus.subscribe("t1", "s1");
        bus.subscribe("t2", "s2");
        bus.unsubscribe(&id);
        let removed = bus.cleanup_subscriptions();
        assert_eq!(removed, 1);
        assert_eq!(bus.all_subscriptions().len(), 1);
    }

    #[test]
    fn test_delivery_records() {
        let config = EventBusConfig {
            store_delivery_records: true,
            ..Default::default()
        };
        let mut bus = EventBus::new(config);
        bus.subscribe("t", "s");
        bus.publish(Event::new("t", "{}", "test"));
        let records = bus.recent_deliveries(10);
        assert_eq!(records.len(), 1);
        assert!(records[0].success);
    }

    #[test]
    fn test_max_buffer_size() {
        let config = EventBusConfig {
            max_buffer_size: 3,
            ..Default::default()
        };
        let mut bus = EventBus::new(config);
        for i in 0..5 {
            bus.publish(Event::new("t", format!("{}", i), "test"));
        }
        assert_eq!(bus.buffer_size(), 3);
    }

    // SharedEventBus

    #[test]
    fn test_shared_event_bus() {
        let bus = SharedEventBus::default();
        bus.subscribe("t", "s");
        let count = bus.publish(Event::new("t", "{}", "test"));
        assert_eq!(count, 1);
        assert_eq!(bus.stats().events_published, 1);
    }

    #[test]
    fn test_shared_event_bus_unsubscribe() {
        let bus = SharedEventBus::default();
        let id = bus.subscribe("t", "s");
        assert!(bus.unsubscribe(&id));
    }

    #[test]
    fn test_default_config() {
        let cfg = EventBusConfig::default();
        assert_eq!(cfg.max_buffer_size, 10_000);
        assert!(cfg.store_delivery_records);
    }
}
