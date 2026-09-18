#![allow(dead_code)]
//! Priority-based playlist scheduling.
//!
//! Provides [`PriorityLevel`] and [`PriorityScheduler`] for ordering
//! playlist items by their editorial or operational urgency.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Priority level
// ---------------------------------------------------------------------------

/// Editorial or operational priority level for a playlist item.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PriorityLevel {
    /// Filler content — lowest priority.
    Filler = 0,
    /// Standard scheduled content.
    Standard = 1,
    /// Important content (e.g., peak-hour programming).
    Important = 2,
    /// Breaking news or emergency interrupts.
    Breaking = 3,
}

impl PriorityLevel {
    /// Returns a human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Filler => "filler",
            Self::Standard => "standard",
            Self::Important => "important",
            Self::Breaking => "breaking",
        }
    }

    /// Returns `true` if this level may interrupt currently playing content.
    pub fn can_interrupt(&self) -> bool {
        matches!(self, Self::Breaking)
    }

    /// Returns the numeric rank (higher = more urgent).
    pub fn rank(&self) -> u8 {
        *self as u8
    }
}

impl PartialOrd for PriorityLevel {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PriorityLevel {
    fn cmp(&self, other: &Self) -> Ordering {
        self.rank().cmp(&other.rank())
    }
}

// ---------------------------------------------------------------------------
// Priority item
// ---------------------------------------------------------------------------

/// A playlist item annotated with a scheduling priority.
#[derive(Debug, Clone)]
pub struct PriorityItem {
    /// Unique item identifier.
    pub id: u64,
    /// URI or path to the media.
    pub uri: String,
    /// Scheduling priority.
    pub priority: PriorityLevel,
    /// Optional media duration.
    pub duration: Option<Duration>,
    /// Sequence number used as tiebreaker (lower = scheduled earlier).
    pub sequence: u64,
}

impl PriorityItem {
    /// Creates a new priority item.
    pub fn new(id: u64, uri: impl Into<String>, priority: PriorityLevel) -> Self {
        Self {
            id,
            uri: uri.into(),
            priority,
            duration: None,
            sequence: id,
        }
    }

    /// Attaches a duration.
    pub fn with_duration(mut self, dur: Duration) -> Self {
        self.duration = Some(dur);
        self
    }

    /// Overrides the tiebreaker sequence number.
    pub fn with_sequence(mut self, seq: u64) -> Self {
        self.sequence = seq;
        self
    }
}

// Implement Ord/PartialOrd so BinaryHeap returns highest priority first.
// For equal priority, lower sequence wins (higher value in comparison means
// it was inserted first).
impl PartialEq for PriorityItem {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.sequence == other.sequence
    }
}

impl Eq for PriorityItem {}

impl PartialOrd for PriorityItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PriorityItem {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher priority first; then lower sequence number first (invert).
        self.priority
            .cmp(&other.priority)
            .then(other.sequence.cmp(&self.sequence))
    }
}

// ---------------------------------------------------------------------------
// PriorityScheduler
// ---------------------------------------------------------------------------

/// Schedules playlist items by priority, using a max-heap internally.
#[derive(Debug, Default)]
pub struct PriorityScheduler {
    heap: BinaryHeap<PriorityItem>,
    next_seq: u64,
}

impl PriorityScheduler {
    /// Creates an empty scheduler.
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts an item, assigning it an auto-incremented sequence number
    /// as a tiebreaker.
    pub fn push(&mut self, mut item: PriorityItem) {
        item.sequence = self.next_seq;
        self.next_seq += 1;
        self.heap.push(item);
    }

    /// Removes and returns the next item to be played (highest priority,
    /// earliest insertion as tiebreaker).
    pub fn pop(&mut self) -> Option<PriorityItem> {
        self.heap.pop()
    }

    /// Returns a reference to the next item without removing it.
    pub fn peek(&self) -> Option<&PriorityItem> {
        self.heap.peek()
    }

    /// Returns the number of items queued.
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// Returns `true` if no items are queued.
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Removes all items whose priority is below `min`.
    pub fn drain_below(&mut self, min: PriorityLevel) {
        let remaining: Vec<PriorityItem> = self
            .heap
            .drain()
            .filter(|item| item.priority >= min)
            .collect();
        self.heap.extend(remaining);
    }

    /// Returns a snapshot of all items sorted by priority (does not mutate).
    pub fn sorted_snapshot(&self) -> Vec<&PriorityItem> {
        let mut v: Vec<&PriorityItem> = self.heap.iter().collect();
        v.sort_unstable_by(|a, b| b.cmp(a));
        v
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_priority_level_rank_order() {
        assert!(PriorityLevel::Breaking > PriorityLevel::Important);
        assert!(PriorityLevel::Important > PriorityLevel::Standard);
        assert!(PriorityLevel::Standard > PriorityLevel::Filler);
    }

    #[test]
    fn test_priority_level_label() {
        assert_eq!(PriorityLevel::Filler.label(), "filler");
        assert_eq!(PriorityLevel::Standard.label(), "standard");
        assert_eq!(PriorityLevel::Important.label(), "important");
        assert_eq!(PriorityLevel::Breaking.label(), "breaking");
    }

    #[test]
    fn test_can_interrupt_only_breaking() {
        assert!(!PriorityLevel::Filler.can_interrupt());
        assert!(!PriorityLevel::Standard.can_interrupt());
        assert!(!PriorityLevel::Important.can_interrupt());
        assert!(PriorityLevel::Breaking.can_interrupt());
    }

    #[test]
    fn test_scheduler_pop_highest_priority_first() {
        let mut sched = PriorityScheduler::new();
        sched.push(PriorityItem::new(1, "filler.mp4", PriorityLevel::Filler));
        sched.push(PriorityItem::new(
            2,
            "breaking.mp4",
            PriorityLevel::Breaking,
        ));
        sched.push(PriorityItem::new(3, "std.mp4", PriorityLevel::Standard));

        let first = sched.pop().expect("should succeed in test");
        assert_eq!(first.priority, PriorityLevel::Breaking);
    }

    #[test]
    fn test_scheduler_fifo_tiebreaker() {
        let mut sched = PriorityScheduler::new();
        sched.push(PriorityItem::new(1, "a.mp4", PriorityLevel::Standard));
        sched.push(PriorityItem::new(2, "b.mp4", PriorityLevel::Standard));

        // Both Standard — first inserted should come out first
        let first = sched.pop().expect("should succeed in test");
        assert_eq!(first.uri, "a.mp4");
    }

    #[test]
    fn test_scheduler_len_and_empty() {
        let mut sched = PriorityScheduler::new();
        assert!(sched.is_empty());
        sched.push(PriorityItem::new(1, "x.mp4", PriorityLevel::Standard));
        assert_eq!(sched.len(), 1);
        assert!(!sched.is_empty());
    }

    #[test]
    fn test_scheduler_peek_does_not_remove() {
        let mut sched = PriorityScheduler::new();
        sched.push(PriorityItem::new(1, "x.mp4", PriorityLevel::Important));
        assert!(sched.peek().is_some());
        assert_eq!(sched.len(), 1);
    }

    #[test]
    fn test_scheduler_pop_empty_returns_none() {
        let mut sched = PriorityScheduler::new();
        assert!(sched.pop().is_none());
    }

    #[test]
    fn test_drain_below_removes_low_priority() {
        let mut sched = PriorityScheduler::new();
        sched.push(PriorityItem::new(1, "filler.mp4", PriorityLevel::Filler));
        sched.push(PriorityItem::new(2, "std.mp4", PriorityLevel::Standard));
        sched.push(PriorityItem::new(3, "imp.mp4", PriorityLevel::Important));

        sched.drain_below(PriorityLevel::Standard);
        assert_eq!(sched.len(), 2);
    }

    #[test]
    fn test_priority_item_with_duration() {
        let item = PriorityItem::new(7, "v.mp4", PriorityLevel::Standard)
            .with_duration(Duration::from_secs(120));
        assert_eq!(item.duration, Some(Duration::from_secs(120)));
    }

    #[test]
    fn test_sorted_snapshot_order() {
        let mut sched = PriorityScheduler::new();
        sched.push(PriorityItem::new(1, "filler.mp4", PriorityLevel::Filler));
        sched.push(PriorityItem::new(
            2,
            "breaking.mp4",
            PriorityLevel::Breaking,
        ));
        sched.push(PriorityItem::new(3, "std.mp4", PriorityLevel::Standard));

        let snap = sched.sorted_snapshot();
        assert_eq!(snap[0].priority, PriorityLevel::Breaking);
    }

    #[test]
    fn test_priority_rank_values() {
        assert_eq!(PriorityLevel::Filler.rank(), 0);
        assert_eq!(PriorityLevel::Standard.rank(), 1);
        assert_eq!(PriorityLevel::Important.rank(), 2);
        assert_eq!(PriorityLevel::Breaking.rank(), 3);
    }

    #[test]
    fn test_multiple_breaking_fifo() {
        let mut sched = PriorityScheduler::new();
        sched.push(PriorityItem::new(
            10,
            "first_break.mp4",
            PriorityLevel::Breaking,
        ));
        sched.push(PriorityItem::new(
            11,
            "second_break.mp4",
            PriorityLevel::Breaking,
        ));

        let first = sched.pop().expect("should succeed in test");
        assert_eq!(first.uri, "first_break.mp4");
    }

    #[test]
    fn test_drain_below_filler_removes_nothing_when_all_above() {
        let mut sched = PriorityScheduler::new();
        sched.push(PriorityItem::new(1, "s.mp4", PriorityLevel::Standard));
        sched.push(PriorityItem::new(2, "i.mp4", PriorityLevel::Important));
        sched.drain_below(PriorityLevel::Filler);
        assert_eq!(sched.len(), 2);
    }
}
