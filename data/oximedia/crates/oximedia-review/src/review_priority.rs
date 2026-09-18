#![allow(dead_code)]
//! Priority management for review items and sessions.
//!
//! Provides a system for assigning, escalating, and sorting review items
//! by priority level, including urgency-based auto-escalation and
//! weighted priority scoring.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Priority level for a review item.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PriorityLevel {
    /// Lowest priority, handled when convenient.
    Low,
    /// Normal priority, standard turnaround.
    Normal,
    /// High priority, should be reviewed promptly.
    High,
    /// Critical priority, blocks release.
    Critical,
    /// Emergency priority, requires immediate attention.
    Emergency,
}

impl PriorityLevel {
    /// Return the numeric weight for this priority level.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn weight(&self) -> f64 {
        match self {
            Self::Low => 1.0,
            Self::Normal => 2.0,
            Self::High => 4.0,
            Self::Critical => 8.0,
            Self::Emergency => 16.0,
        }
    }

    /// Return the maximum allowed age in hours before escalation.
    #[must_use]
    pub fn max_age_hours(&self) -> u64 {
        match self {
            Self::Low => 168,     // 1 week
            Self::Normal => 72,   // 3 days
            Self::High => 24,     // 1 day
            Self::Critical => 4,  // 4 hours
            Self::Emergency => 1, // 1 hour
        }
    }

    /// Escalate to the next higher priority level, if possible.
    #[must_use]
    pub fn escalate(&self) -> Self {
        match self {
            Self::Low => Self::Normal,
            Self::Normal => Self::High,
            Self::High => Self::Critical,
            Self::Critical => Self::Emergency,
            Self::Emergency => Self::Emergency,
        }
    }

    /// De-escalate to the next lower priority level, if possible.
    #[must_use]
    pub fn deescalate(&self) -> Self {
        match self {
            Self::Low => Self::Low,
            Self::Normal => Self::Low,
            Self::High => Self::Normal,
            Self::Critical => Self::High,
            Self::Emergency => Self::Critical,
        }
    }

    /// Check whether this priority is at or above the given threshold.
    #[must_use]
    pub fn is_at_least(&self, threshold: Self) -> bool {
        self.weight() >= threshold.weight()
    }
}

impl PartialOrd for PriorityLevel {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PriorityLevel {
    fn cmp(&self, other: &Self) -> Ordering {
        #[allow(clippy::cast_precision_loss)]
        self.weight()
            .partial_cmp(&other.weight())
            .unwrap_or(Ordering::Equal)
    }
}

/// A review item with an associated priority.
#[derive(Debug, Clone)]
pub struct PrioritizedItem {
    /// Unique identifier for this item.
    pub item_id: String,
    /// Current priority level.
    pub priority: PriorityLevel,
    /// Age of the item in hours since creation.
    pub age_hours: u64,
    /// Number of times this item has been escalated.
    pub escalation_count: u32,
    /// Optional label for grouping.
    pub label: Option<String>,
}

impl PrioritizedItem {
    /// Create a new prioritized item.
    #[must_use]
    pub fn new(item_id: impl Into<String>, priority: PriorityLevel) -> Self {
        Self {
            item_id: item_id.into(),
            priority,
            age_hours: 0,
            escalation_count: 0,
            label: None,
        }
    }

    /// Set the age of the item in hours.
    #[must_use]
    pub fn with_age(mut self, hours: u64) -> Self {
        self.age_hours = hours;
        self
    }

    /// Set an optional label.
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Compute the effective priority score, factoring in age-based urgency.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn effective_score(&self) -> f64 {
        let base = self.priority.weight();
        let max_age = self.priority.max_age_hours() as f64;
        let age = self.age_hours as f64;
        let urgency_factor = if max_age > 0.0 {
            (age / max_age).min(2.0)
        } else {
            1.0
        };
        base * (1.0 + urgency_factor)
    }

    /// Check whether this item should be auto-escalated based on its age.
    #[must_use]
    pub fn should_escalate(&self) -> bool {
        self.age_hours > self.priority.max_age_hours()
    }

    /// Escalate this item if it is overdue.
    pub fn auto_escalate(&mut self) {
        if self.should_escalate() {
            self.priority = self.priority.escalate();
            self.escalation_count += 1;
        }
    }
}

impl PartialEq for PrioritizedItem {
    fn eq(&self, other: &Self) -> bool {
        self.item_id == other.item_id
    }
}

impl Eq for PrioritizedItem {}

impl PartialOrd for PrioritizedItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PrioritizedItem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.effective_score()
            .partial_cmp(&other.effective_score())
            .unwrap_or(Ordering::Equal)
    }
}

/// A priority queue for review items, returning the highest-priority item first.
#[derive(Debug)]
pub struct PriorityQueue {
    /// Internal heap storage.
    heap: BinaryHeap<PrioritizedItem>,
    /// Maximum number of items allowed in the queue.
    capacity: usize,
}

impl PriorityQueue {
    /// Create a new priority queue with the given capacity.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            heap: BinaryHeap::with_capacity(capacity),
            capacity,
        }
    }

    /// Push an item onto the queue.
    ///
    /// Returns `false` if the queue is at capacity.
    pub fn push(&mut self, item: PrioritizedItem) -> bool {
        if self.heap.len() >= self.capacity {
            return false;
        }
        self.heap.push(item);
        true
    }

    /// Pop the highest-priority item from the queue.
    pub fn pop(&mut self) -> Option<PrioritizedItem> {
        self.heap.pop()
    }

    /// Peek at the highest-priority item without removing it.
    #[must_use]
    pub fn peek(&self) -> Option<&PrioritizedItem> {
        self.heap.peek()
    }

    /// Return the number of items in the queue.
    #[must_use]
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// Check whether the queue is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Auto-escalate all items in the queue that are overdue.
    pub fn escalate_overdue(&mut self) {
        let items: Vec<_> = std::mem::take(&mut self.heap).into_vec();
        for mut item in items {
            item.auto_escalate();
            self.heap.push(item);
        }
    }

    /// Drain all items from the queue in priority order.
    pub fn drain_sorted(&mut self) -> Vec<PrioritizedItem> {
        let mut result = Vec::with_capacity(self.heap.len());
        while let Some(item) = self.heap.pop() {
            result.push(item);
        }
        result
    }
}

/// Configuration for priority-based escalation policies.
#[derive(Debug, Clone)]
pub struct EscalationPolicy {
    /// Whether automatic escalation is enabled.
    pub auto_escalate: bool,
    /// Maximum number of escalations before alerting.
    pub max_escalations: u32,
    /// Multiplier applied to the age threshold when computing urgency.
    pub urgency_multiplier: f64,
}

impl Default for EscalationPolicy {
    fn default() -> Self {
        Self {
            auto_escalate: true,
            max_escalations: 3,
            urgency_multiplier: 1.5,
        }
    }
}

impl EscalationPolicy {
    /// Create a new escalation policy.
    #[must_use]
    pub fn new(auto_escalate: bool, max_escalations: u32, urgency_multiplier: f64) -> Self {
        Self {
            auto_escalate,
            max_escalations,
            urgency_multiplier,
        }
    }

    /// Check whether the given item has exceeded the maximum escalation count.
    #[must_use]
    pub fn exceeded_max(&self, item: &PrioritizedItem) -> bool {
        item.escalation_count >= self.max_escalations
    }

    /// Compute the adjusted score for an item using the urgency multiplier.
    #[must_use]
    pub fn adjusted_score(&self, item: &PrioritizedItem) -> f64 {
        item.effective_score() * self.urgency_multiplier
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_priority_level_weight_ordering() {
        assert!(PriorityLevel::Low.weight() < PriorityLevel::Normal.weight());
        assert!(PriorityLevel::Normal.weight() < PriorityLevel::High.weight());
        assert!(PriorityLevel::High.weight() < PriorityLevel::Critical.weight());
        assert!(PriorityLevel::Critical.weight() < PriorityLevel::Emergency.weight());
    }

    #[test]
    fn test_priority_level_escalation() {
        assert_eq!(PriorityLevel::Low.escalate(), PriorityLevel::Normal);
        assert_eq!(PriorityLevel::Normal.escalate(), PriorityLevel::High);
        assert_eq!(PriorityLevel::High.escalate(), PriorityLevel::Critical);
        assert_eq!(PriorityLevel::Critical.escalate(), PriorityLevel::Emergency);
        assert_eq!(
            PriorityLevel::Emergency.escalate(),
            PriorityLevel::Emergency
        );
    }

    #[test]
    fn test_priority_level_deescalation() {
        assert_eq!(PriorityLevel::Low.deescalate(), PriorityLevel::Low);
        assert_eq!(PriorityLevel::Normal.deescalate(), PriorityLevel::Low);
        assert_eq!(PriorityLevel::High.deescalate(), PriorityLevel::Normal);
        assert_eq!(PriorityLevel::Critical.deescalate(), PriorityLevel::High);
        assert_eq!(
            PriorityLevel::Emergency.deescalate(),
            PriorityLevel::Critical
        );
    }

    #[test]
    fn test_priority_is_at_least() {
        assert!(PriorityLevel::Emergency.is_at_least(PriorityLevel::Low));
        assert!(PriorityLevel::High.is_at_least(PriorityLevel::High));
        assert!(!PriorityLevel::Low.is_at_least(PriorityLevel::Normal));
    }

    #[test]
    fn test_prioritized_item_creation() {
        let item = PrioritizedItem::new("item-1", PriorityLevel::High)
            .with_age(10)
            .with_label("urgent");
        assert_eq!(item.item_id, "item-1");
        assert_eq!(item.priority, PriorityLevel::High);
        assert_eq!(item.age_hours, 10);
        assert_eq!(item.label, Some("urgent".to_string()));
    }

    #[test]
    fn test_effective_score_increases_with_age() {
        let young = PrioritizedItem::new("a", PriorityLevel::Normal).with_age(0);
        let old = PrioritizedItem::new("b", PriorityLevel::Normal).with_age(100);
        assert!(old.effective_score() > young.effective_score());
    }

    #[test]
    fn test_should_escalate() {
        let fresh = PrioritizedItem::new("a", PriorityLevel::Normal).with_age(10);
        assert!(!fresh.should_escalate());
        let stale = PrioritizedItem::new("b", PriorityLevel::Normal).with_age(200);
        assert!(stale.should_escalate());
    }

    #[test]
    fn test_auto_escalate() {
        let mut item = PrioritizedItem::new("a", PriorityLevel::Low).with_age(200);
        assert!(item.should_escalate());
        item.auto_escalate();
        assert_eq!(item.priority, PriorityLevel::Normal);
        assert_eq!(item.escalation_count, 1);
    }

    #[test]
    fn test_priority_queue_push_pop() {
        let mut q = PriorityQueue::new(10);
        q.push(PrioritizedItem::new("low", PriorityLevel::Low));
        q.push(PrioritizedItem::new("high", PriorityLevel::High));
        q.push(PrioritizedItem::new("crit", PriorityLevel::Critical));

        let top = q.pop().expect("should succeed in test");
        assert_eq!(top.item_id, "crit");
    }

    #[test]
    fn test_priority_queue_capacity() {
        let mut q = PriorityQueue::new(2);
        assert!(q.push(PrioritizedItem::new("a", PriorityLevel::Low)));
        assert!(q.push(PrioritizedItem::new("b", PriorityLevel::Low)));
        assert!(!q.push(PrioritizedItem::new("c", PriorityLevel::Low)));
        assert_eq!(q.len(), 2);
    }

    #[test]
    fn test_priority_queue_drain_sorted() {
        let mut q = PriorityQueue::new(10);
        q.push(PrioritizedItem::new("low", PriorityLevel::Low));
        q.push(PrioritizedItem::new("high", PriorityLevel::High));
        let drained = q.drain_sorted();
        assert_eq!(drained.len(), 2);
        assert_eq!(drained[0].item_id, "high");
        assert_eq!(drained[1].item_id, "low");
        assert!(q.is_empty());
    }

    #[test]
    fn test_escalation_policy_default() {
        let policy = EscalationPolicy::default();
        assert!(policy.auto_escalate);
        assert_eq!(policy.max_escalations, 3);
        assert!((policy.urgency_multiplier - 1.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_escalation_policy_exceeded_max() {
        let policy = EscalationPolicy::new(true, 2, 1.0);
        let mut item = PrioritizedItem::new("x", PriorityLevel::Low);
        item.escalation_count = 3;
        assert!(policy.exceeded_max(&item));
    }

    #[test]
    fn test_escalation_policy_adjusted_score() {
        let policy = EscalationPolicy::new(true, 3, 2.0);
        let item = PrioritizedItem::new("x", PriorityLevel::Normal);
        let base = item.effective_score();
        let adjusted = policy.adjusted_score(&item);
        assert!((adjusted - base * 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_priority_queue_escalate_overdue() {
        let mut q = PriorityQueue::new(10);
        q.push(PrioritizedItem::new("stale", PriorityLevel::Low).with_age(500));
        q.push(PrioritizedItem::new("fresh", PriorityLevel::High).with_age(1));
        q.escalate_overdue();
        let top = q.pop().expect("should succeed in test");
        // The stale item should have been escalated to Normal
        assert_eq!(top.item_id, "stale");
    }
}
