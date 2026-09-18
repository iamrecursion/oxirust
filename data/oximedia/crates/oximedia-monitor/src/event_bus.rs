//! Event bus for the monitoring system.
//!
//! Provides a publish/subscribe mechanism for monitoring events with
//! filtering capabilities and history management.

/// Category of a monitoring event.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub enum EventCategory {
    /// System-level events (CPU, memory, disk).
    System,
    /// Network-related events.
    Network,
    /// Media processing events.
    Media,
    /// Job lifecycle events.
    Job,
    /// Alert events.
    Alert,
    /// Heartbeat / liveness events.
    Heartbeat,
}

impl EventCategory {
    /// Returns `true` if this category is considered critical.
    #[must_use]
    pub fn is_critical(&self) -> bool {
        matches!(self, Self::Alert)
    }
}

/// A single monitoring event.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct MonitorEvent {
    /// Unique identifier.
    pub id: u64,
    /// Event category.
    pub category: EventCategory,
    /// Source component that generated the event.
    pub source: String,
    /// Human-readable message.
    pub message: String,
    /// Unix epoch timestamp (seconds).
    pub timestamp_epoch: u64,
    /// Severity level (0 = lowest, 10 = highest).
    pub severity: u8,
}

impl MonitorEvent {
    /// Create a new monitoring event.
    ///
    /// The `id` field is set to `0` and should be updated by the [`EventBus`]
    /// upon publishing.
    pub fn new(
        category: EventCategory,
        source: impl Into<String>,
        message: impl Into<String>,
        epoch: u64,
    ) -> Self {
        Self {
            id: 0,
            category,
            source: source.into(),
            message: message.into(),
            timestamp_epoch: epoch,
            severity: 5,
        }
    }

    /// Returns `true` when the event severity is critical (>= 8).
    #[must_use]
    pub fn is_critical(&self) -> bool {
        self.severity >= 8
    }
}

/// Filter for querying events from the bus.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct EventFilter {
    /// Only include events whose category is in this list.
    /// An empty list means "accept all categories".
    pub categories: Vec<EventCategory>,
    /// Minimum severity required for an event to pass.
    pub min_severity: u8,
    /// Optional substring that must appear in the event's `source` field.
    pub source_pattern: Option<String>,
}

impl EventFilter {
    /// Create a new filter with default (permissive) settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns `true` if the given event matches this filter.
    #[must_use]
    pub fn matches(&self, event: &MonitorEvent) -> bool {
        if event.severity < self.min_severity {
            return false;
        }

        if !self.categories.is_empty() && !self.categories.contains(&event.category) {
            return false;
        }

        if let Some(ref pattern) = self.source_pattern {
            if !event.source.contains(pattern.as_str()) {
                return false;
            }
        }

        true
    }
}

/// In-memory event bus with history management.
#[derive(Debug)]
#[allow(dead_code)]
pub struct EventBus {
    events: Vec<MonitorEvent>,
    next_id: u64,
    max_history: usize,
}

impl EventBus {
    /// Create a new event bus with the given history capacity.
    #[must_use]
    pub fn new(max_history: usize) -> Self {
        Self {
            events: Vec::new(),
            next_id: 1,
            max_history,
        }
    }

    /// Publish an event.  The event's `id` will be set automatically.
    /// When the history is full, the oldest event is dropped.
    pub fn publish(&mut self, mut event: MonitorEvent) {
        event.id = self.next_id;
        self.next_id += 1;
        self.events.push(event);

        while self.events.len() > self.max_history {
            self.events.remove(0);
        }
    }

    /// Query events matching the supplied filter.
    #[must_use]
    pub fn query(&self, filter: &EventFilter) -> Vec<&MonitorEvent> {
        self.events.iter().filter(|e| filter.matches(e)).collect()
    }

    /// Return all events whose severity is critical (>= 8).
    #[must_use]
    pub fn critical_events(&self) -> Vec<&MonitorEvent> {
        self.events.iter().filter(|e| e.is_critical()).collect()
    }

    /// Return the total number of events currently in history.
    #[must_use]
    pub fn event_count(&self) -> usize {
        self.events.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_event(cat: EventCategory, source: &str, msg: &str, epoch: u64) -> MonitorEvent {
        MonitorEvent::new(cat, source, msg, epoch)
    }

    // ------------------------------------------------------------------ //
    // EventCategory
    // ------------------------------------------------------------------ //

    #[test]
    fn test_category_alert_is_critical() {
        assert!(EventCategory::Alert.is_critical());
    }

    #[test]
    fn test_category_system_not_critical() {
        assert!(!EventCategory::System.is_critical());
    }

    #[test]
    fn test_category_network_not_critical() {
        assert!(!EventCategory::Network.is_critical());
    }

    #[test]
    fn test_category_media_not_critical() {
        assert!(!EventCategory::Media.is_critical());
    }

    #[test]
    fn test_category_job_not_critical() {
        assert!(!EventCategory::Job.is_critical());
    }

    #[test]
    fn test_category_heartbeat_not_critical() {
        assert!(!EventCategory::Heartbeat.is_critical());
    }

    // ------------------------------------------------------------------ //
    // MonitorEvent
    // ------------------------------------------------------------------ //

    #[test]
    fn test_event_new_defaults() {
        let e = make_event(EventCategory::System, "cpu", "high usage", 1_000);
        assert_eq!(e.id, 0);
        assert_eq!(e.source, "cpu");
        assert_eq!(e.message, "high usage");
        assert_eq!(e.timestamp_epoch, 1_000);
        assert_eq!(e.severity, 5);
    }

    #[test]
    fn test_event_is_critical_high_severity() {
        let mut e = make_event(EventCategory::Alert, "src", "msg", 0);
        e.severity = 8;
        assert!(e.is_critical());
    }

    #[test]
    fn test_event_is_critical_low_severity() {
        let mut e = make_event(EventCategory::Alert, "src", "msg", 0);
        e.severity = 7;
        assert!(!e.is_critical());
    }

    // ------------------------------------------------------------------ //
    // EventFilter
    // ------------------------------------------------------------------ //

    #[test]
    fn test_filter_default_accepts_all() {
        let filter = EventFilter::new();
        let e = make_event(EventCategory::System, "cpu", "msg", 0);
        assert!(filter.matches(&e));
    }

    #[test]
    fn test_filter_min_severity_excludes() {
        let filter = EventFilter {
            min_severity: 7,
            ..EventFilter::default()
        };
        let e = make_event(EventCategory::System, "cpu", "msg", 0);
        // default severity is 5
        assert!(!filter.matches(&e));
    }

    #[test]
    fn test_filter_category_excludes() {
        let filter = EventFilter {
            categories: vec![EventCategory::Network],
            ..EventFilter::default()
        };
        let e = make_event(EventCategory::System, "cpu", "msg", 0);
        assert!(!filter.matches(&e));
    }

    #[test]
    fn test_filter_source_pattern_matches() {
        let filter = EventFilter {
            source_pattern: Some("encoder".to_string()),
            ..EventFilter::default()
        };
        let e = make_event(EventCategory::Media, "encoder-1", "msg", 0);
        assert!(filter.matches(&e));
    }

    #[test]
    fn test_filter_source_pattern_excludes() {
        let filter = EventFilter {
            source_pattern: Some("encoder".to_string()),
            ..EventFilter::default()
        };
        let e = make_event(EventCategory::Media, "decoder-1", "msg", 0);
        assert!(!filter.matches(&e));
    }

    // ------------------------------------------------------------------ //
    // EventBus
    // ------------------------------------------------------------------ //

    #[test]
    fn test_bus_publish_increments_id() {
        let mut bus = EventBus::new(10);
        bus.publish(make_event(EventCategory::System, "s", "m", 0));
        bus.publish(make_event(EventCategory::System, "s", "m", 0));
        assert_eq!(bus.events[0].id, 1);
        assert_eq!(bus.events[1].id, 2);
    }

    #[test]
    fn test_bus_max_history_respected() {
        let mut bus = EventBus::new(3);
        for _ in 0..5 {
            bus.publish(make_event(EventCategory::System, "s", "m", 0));
        }
        assert_eq!(bus.event_count(), 3);
    }

    #[test]
    fn test_bus_query_with_filter() {
        let mut bus = EventBus::new(100);
        bus.publish(make_event(EventCategory::System, "cpu", "ok", 0));
        bus.publish(make_event(EventCategory::Alert, "cpu", "critical", 0));

        let filter = EventFilter {
            categories: vec![EventCategory::Alert],
            ..EventFilter::default()
        };
        let results = bus.query(&filter);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].message, "critical");
    }

    #[test]
    fn test_bus_critical_events() {
        let mut bus = EventBus::new(100);
        let mut e = make_event(EventCategory::Alert, "src", "crisis", 0);
        e.severity = 9;
        bus.publish(e);
        bus.publish(make_event(EventCategory::System, "cpu", "normal", 0));

        let crits = bus.critical_events();
        assert_eq!(crits.len(), 1);
        assert_eq!(crits[0].message, "crisis");
    }

    #[test]
    fn test_bus_empty_query_returns_all() {
        let mut bus = EventBus::new(100);
        bus.publish(make_event(EventCategory::System, "s", "m", 0));
        bus.publish(make_event(EventCategory::Network, "n", "m", 0));

        let filter = EventFilter::new();
        assert_eq!(bus.query(&filter).len(), 2);
    }
}
