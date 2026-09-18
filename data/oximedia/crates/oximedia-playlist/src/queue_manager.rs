#![allow(dead_code)]
//! Queue manager for ordered media playback queues.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// A single entry in a play queue.
#[derive(Debug, Clone)]
pub struct QueueEntry {
    /// Unique identifier for the entry.
    pub id: u64,
    /// URI or path of the media item.
    pub uri: String,
    /// Optional display title.
    pub title: Option<String>,
    /// Optional duration of the media.
    pub duration: Option<Duration>,
    /// Time at which this entry was enqueued.
    pub enqueued_at: Instant,
    /// Optional TTL after which the entry is considered expired.
    pub ttl: Option<Duration>,
}

impl QueueEntry {
    /// Creates a new queue entry.
    pub fn new(id: u64, uri: impl Into<String>) -> Self {
        Self {
            id,
            uri: uri.into(),
            title: None,
            duration: None,
            enqueued_at: Instant::now(),
            ttl: None,
        }
    }

    /// Attaches a title.
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Attaches a duration.
    pub fn with_duration(mut self, dur: Duration) -> Self {
        self.duration = Some(dur);
        self
    }

    /// Attaches a time-to-live for the entry.
    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.ttl = Some(ttl);
        self
    }

    /// Returns true if the entry has exceeded its TTL.
    pub fn is_expired(&self) -> bool {
        if let Some(ttl) = self.ttl {
            self.enqueued_at.elapsed() >= ttl
        } else {
            false
        }
    }

    /// Returns the age of the entry since it was enqueued.
    pub fn age(&self) -> Duration {
        self.enqueued_at.elapsed()
    }
}

/// Priority level for queue entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum QueuePriority {
    /// Low priority — plays after normal entries.
    Low = 0,
    /// Normal priority.
    #[default]
    Normal = 1,
    /// High priority — plays before normal entries.
    High = 2,
    /// Urgent — plays immediately at front.
    Urgent = 3,
}

/// Ordered playback queue supporting FIFO dequeue with priority insertion.
#[derive(Debug, Default)]
pub struct PlayQueue {
    entries: VecDeque<QueueEntry>,
    next_id: u64,
    max_size: Option<usize>,
}

impl PlayQueue {
    /// Creates an unbounded play queue.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a play queue with a maximum capacity.
    pub fn with_capacity(max: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            next_id: 0,
            max_size: Some(max),
        }
    }

    /// Adds an entry to the back of the queue.
    ///
    /// Returns `None` if the queue is at capacity.
    pub fn enqueue(&mut self, uri: impl Into<String>) -> Option<u64> {
        if let Some(max) = self.max_size {
            if self.entries.len() >= max {
                return None;
            }
        }
        let id = self.next_id;
        self.next_id += 1;
        self.entries.push_back(QueueEntry::new(id, uri));
        Some(id)
    }

    /// Adds an entry at the front (high-priority insertion).
    pub fn enqueue_front(&mut self, uri: impl Into<String>) -> Option<u64> {
        if let Some(max) = self.max_size {
            if self.entries.len() >= max {
                return None;
            }
        }
        let id = self.next_id;
        self.next_id += 1;
        self.entries.push_front(QueueEntry::new(id, uri));
        Some(id)
    }

    /// Removes and returns the entry at the front of the queue.
    pub fn dequeue_front(&mut self) -> Option<QueueEntry> {
        self.entries.pop_front()
    }

    /// Returns a reference to the entry at the front without removing it.
    pub fn peek(&self) -> Option<&QueueEntry> {
        self.entries.front()
    }

    /// Returns the number of entries currently in the queue.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns true if the queue has no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Removes all expired entries from the queue and returns the count removed.
    pub fn purge_expired(&mut self) -> usize {
        let before = self.entries.len();
        self.entries.retain(|e| !e.is_expired());
        before - self.entries.len()
    }

    /// Clears all entries from the queue.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Returns an iterator over the entries without consuming the queue.
    pub fn iter(&self) -> impl Iterator<Item = &QueueEntry> {
        self.entries.iter()
    }

    /// Returns the maximum allowed size, if any.
    pub fn max_size(&self) -> Option<usize> {
        self.max_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enqueue_and_len() {
        let mut q = PlayQueue::new();
        q.enqueue("a.mp4");
        q.enqueue("b.mp4");
        assert_eq!(q.len(), 2);
    }

    #[test]
    fn test_is_empty_initially() {
        let q = PlayQueue::new();
        assert!(q.is_empty());
    }

    #[test]
    fn test_dequeue_front_order() {
        let mut q = PlayQueue::new();
        q.enqueue("first.mp4");
        q.enqueue("second.mp4");
        let entry = q.dequeue_front().expect("should succeed in test");
        assert_eq!(entry.uri, "first.mp4");
    }

    #[test]
    fn test_dequeue_front_empty() {
        let mut q = PlayQueue::new();
        assert!(q.dequeue_front().is_none());
    }

    #[test]
    fn test_peek_does_not_remove() {
        let mut q = PlayQueue::new();
        q.enqueue("item.mp4");
        let _ = q.peek();
        assert_eq!(q.len(), 1);
    }

    #[test]
    fn test_peek_returns_front() {
        let mut q = PlayQueue::new();
        q.enqueue("alpha.mp4");
        q.enqueue("beta.mp4");
        let front = q.peek().expect("should succeed in test");
        assert_eq!(front.uri, "alpha.mp4");
    }

    #[test]
    fn test_enqueue_front_priority() {
        let mut q = PlayQueue::new();
        q.enqueue("normal.mp4");
        q.enqueue_front("urgent.mp4");
        let front = q.dequeue_front().expect("should succeed in test");
        assert_eq!(front.uri, "urgent.mp4");
    }

    #[test]
    fn test_capacity_limit() {
        let mut q = PlayQueue::with_capacity(2);
        assert!(q.enqueue("a.mp4").is_some());
        assert!(q.enqueue("b.mp4").is_some());
        assert!(q.enqueue("c.mp4").is_none());
    }

    #[test]
    fn test_clear() {
        let mut q = PlayQueue::new();
        q.enqueue("a.mp4");
        q.enqueue("b.mp4");
        q.clear();
        assert!(q.is_empty());
    }

    #[test]
    fn test_entry_not_expired_without_ttl() {
        let entry = QueueEntry::new(1, "x.mp4");
        assert!(!entry.is_expired());
    }

    #[test]
    fn test_entry_not_expired_with_long_ttl() {
        let entry = QueueEntry::new(1, "x.mp4").with_ttl(Duration::from_secs(3600));
        assert!(!entry.is_expired());
    }

    #[test]
    fn test_queue_entry_with_title() {
        let entry = QueueEntry::new(42, "vid.mp4").with_title("My Video");
        assert_eq!(entry.title.as_deref(), Some("My Video"));
    }

    #[test]
    fn test_unique_ids_assigned() {
        let mut q = PlayQueue::new();
        let id1 = q.enqueue("a.mp4").expect("should succeed in test");
        let id2 = q.enqueue("b.mp4").expect("should succeed in test");
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_purge_expired_no_ttl() {
        let mut q = PlayQueue::new();
        q.enqueue("a.mp4");
        q.enqueue("b.mp4");
        let removed = q.purge_expired();
        assert_eq!(removed, 0);
        assert_eq!(q.len(), 2);
    }

    #[test]
    fn test_max_size_is_stored() {
        let q = PlayQueue::with_capacity(5);
        assert_eq!(q.max_size(), Some(5));
    }

    #[test]
    fn test_iter_length() {
        let mut q = PlayQueue::new();
        q.enqueue("a.mp4");
        q.enqueue("b.mp4");
        q.enqueue("c.mp4");
        assert_eq!(q.iter().count(), 3);
    }
}
