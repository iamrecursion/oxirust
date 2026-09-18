//! Render queue for managing ordered, prioritised render requests.
//!
//! Provides `RenderPriority`, `RenderRequest`, and `RenderQueue` for
//! tracking and scheduling frame render jobs within a timeline pipeline.

#![allow(dead_code)]

use std::time::{Duration, Instant};

/// Priority levels for render requests.
///
/// Higher numeric value = higher priority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RenderPriority {
    /// Background pre-fetch rendering (lowest).
    Background,
    /// Normal batch rendering.
    Normal,
    /// User-initiated rendering (interactive).
    Interactive,
    /// Real-time playback rendering (highest).
    Realtime,
}

impl RenderPriority {
    /// Returns the numeric priority value (0 = lowest, 3 = highest).
    #[must_use]
    pub const fn value(&self) -> u8 {
        match self {
            Self::Background => 0,
            Self::Normal => 1,
            Self::Interactive => 2,
            Self::Realtime => 3,
        }
    }

    /// Returns `true` if this priority is at least `Interactive`.
    #[must_use]
    pub const fn is_high_priority(&self) -> bool {
        matches!(self, Self::Interactive | Self::Realtime)
    }

    /// Returns a human-readable label.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Background => "background",
            Self::Normal => "normal",
            Self::Interactive => "interactive",
            Self::Realtime => "realtime",
        }
    }
}

/// A single render request for one or more frames.
#[derive(Debug, Clone)]
pub struct RenderRequest {
    /// Unique request identifier.
    pub id: u64,
    /// The first frame to render (inclusive).
    pub start_frame: u64,
    /// The last frame to render (inclusive).
    pub end_frame: u64,
    /// Priority of this request.
    pub priority: RenderPriority,
    /// When the request was submitted.
    pub submitted_at: Instant,
    /// Optional deadline; if `Some` and now > deadline, the request is expired.
    pub deadline: Option<Instant>,
}

impl RenderRequest {
    /// Creates a new `RenderRequest` submitted at the current instant.
    #[must_use]
    pub fn new(id: u64, start_frame: u64, end_frame: u64, priority: RenderPriority) -> Self {
        Self {
            id,
            start_frame,
            end_frame,
            priority,
            submitted_at: Instant::now(),
            deadline: None,
        }
    }

    /// Creates a `RenderRequest` with an explicit deadline.
    #[must_use]
    pub fn with_deadline(
        id: u64,
        start_frame: u64,
        end_frame: u64,
        priority: RenderPriority,
        deadline: Instant,
    ) -> Self {
        Self {
            id,
            start_frame,
            end_frame,
            priority,
            submitted_at: Instant::now(),
            deadline: Some(deadline),
        }
    }

    /// Returns `true` if this request has passed its deadline.
    ///
    /// A request with no deadline is never considered expired.
    #[must_use]
    pub fn is_expired(&self) -> bool {
        self.deadline.is_some_and(|dl| Instant::now() > dl)
    }

    /// Returns the number of frames covered by this request.
    #[must_use]
    pub fn frame_count(&self) -> u64 {
        self.end_frame.saturating_sub(self.start_frame) + 1
    }

    /// Returns how long this request has been waiting since submission.
    #[must_use]
    pub fn age(&self) -> Duration {
        self.submitted_at.elapsed()
    }
}

/// A priority queue of `RenderRequest` entries.
///
/// `next_request` always returns the highest-priority, oldest request first.
/// Within the same priority tier, FIFO ordering is used.
#[derive(Debug, Default)]
pub struct RenderQueue {
    requests: Vec<RenderRequest>,
    next_id: u64,
}

impl RenderQueue {
    /// Creates a new, empty `RenderQueue`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            requests: Vec::new(),
            next_id: 1,
        }
    }

    /// Submits a new render request to the queue.
    ///
    /// Automatically assigns a unique ID and inserts in priority order
    /// (highest priority first; FIFO within priority).
    pub fn submit(&mut self, start_frame: u64, end_frame: u64, priority: RenderPriority) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        let req = RenderRequest::new(id, start_frame, end_frame, priority);
        self.requests.push(req);
        // Sort descending by priority value, then ascending by submitted_at (FIFO).
        self.requests.sort_by(|a, b| {
            b.priority
                .value()
                .cmp(&a.priority.value())
                .then(a.submitted_at.cmp(&b.submitted_at))
        });
        id
    }

    /// Removes and returns the next request to render (highest priority, FIFO).
    ///
    /// Returns `None` if the queue is empty.
    pub fn next_request(&mut self) -> Option<RenderRequest> {
        if self.requests.is_empty() {
            None
        } else {
            Some(self.requests.remove(0))
        }
    }

    /// Returns the number of pending (not yet consumed) requests.
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.requests.len()
    }

    /// Returns `true` if there are no pending requests.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }

    /// Removes all expired requests from the queue and returns the count removed.
    pub fn purge_expired(&mut self) -> usize {
        let before = self.requests.len();
        self.requests.retain(|r| !r.is_expired());
        before - self.requests.len()
    }

    /// Cancels the request with the given `id`, returning `true` if found.
    pub fn cancel(&mut self, id: u64) -> bool {
        let before = self.requests.len();
        self.requests.retain(|r| r.id != id);
        self.requests.len() < before
    }

    /// Returns a view of all pending requests in priority order.
    #[must_use]
    pub fn peek_all(&self) -> &[RenderRequest] {
        &self.requests
    }

    /// Returns the priority of the next request without removing it.
    #[must_use]
    pub fn next_priority(&self) -> Option<RenderPriority> {
        self.requests.first().map(|r| r.priority)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_priority_values() {
        assert!(RenderPriority::Realtime.value() > RenderPriority::Interactive.value());
        assert!(RenderPriority::Interactive.value() > RenderPriority::Normal.value());
        assert!(RenderPriority::Normal.value() > RenderPriority::Background.value());
    }

    #[test]
    fn test_priority_is_high() {
        assert!(RenderPriority::Interactive.is_high_priority());
        assert!(RenderPriority::Realtime.is_high_priority());
        assert!(!RenderPriority::Normal.is_high_priority());
        assert!(!RenderPriority::Background.is_high_priority());
    }

    #[test]
    fn test_priority_labels() {
        assert_eq!(RenderPriority::Background.label(), "background");
        assert_eq!(RenderPriority::Normal.label(), "normal");
        assert_eq!(RenderPriority::Interactive.label(), "interactive");
        assert_eq!(RenderPriority::Realtime.label(), "realtime");
    }

    #[test]
    fn test_render_request_frame_count() {
        let req = RenderRequest::new(1, 0, 23, RenderPriority::Normal);
        assert_eq!(req.frame_count(), 24);
    }

    #[test]
    fn test_render_request_single_frame() {
        let req = RenderRequest::new(1, 100, 100, RenderPriority::Normal);
        assert_eq!(req.frame_count(), 1);
    }

    #[test]
    fn test_render_request_not_expired_by_default() {
        let req = RenderRequest::new(1, 0, 23, RenderPriority::Normal);
        assert!(!req.is_expired());
    }

    #[test]
    fn test_render_request_expired_past_deadline() {
        let past = Instant::now() - Duration::from_secs(10);
        let req = RenderRequest::with_deadline(1, 0, 23, RenderPriority::Normal, past);
        assert!(req.is_expired());
    }

    #[test]
    fn test_render_request_not_expired_future_deadline() {
        let future = Instant::now() + Duration::from_secs(60);
        let req = RenderRequest::with_deadline(1, 0, 23, RenderPriority::Normal, future);
        assert!(!req.is_expired());
    }

    #[test]
    fn test_render_queue_submit_assigns_id() {
        let mut q = RenderQueue::new();
        let id = q.submit(0, 23, RenderPriority::Normal);
        assert_eq!(id, 1);
        let id2 = q.submit(24, 47, RenderPriority::Normal);
        assert_eq!(id2, 2);
    }

    #[test]
    fn test_render_queue_pending_count() {
        let mut q = RenderQueue::new();
        assert_eq!(q.pending_count(), 0);
        q.submit(0, 23, RenderPriority::Normal);
        assert_eq!(q.pending_count(), 1);
    }

    #[test]
    fn test_render_queue_next_request_priority_order() {
        let mut q = RenderQueue::new();
        q.submit(0, 23, RenderPriority::Background);
        q.submit(24, 47, RenderPriority::Realtime);
        let req = q.next_request().expect("should succeed in test");
        assert_eq!(req.priority, RenderPriority::Realtime);
    }

    #[test]
    fn test_render_queue_empty_next_is_none() {
        let mut q = RenderQueue::new();
        assert!(q.next_request().is_none());
    }

    #[test]
    fn test_render_queue_cancel() {
        let mut q = RenderQueue::new();
        let id = q.submit(0, 23, RenderPriority::Normal);
        assert!(q.cancel(id));
        assert!(q.is_empty());
        assert!(!q.cancel(999));
    }

    #[test]
    fn test_render_queue_purge_expired() {
        let mut q = RenderQueue::new();
        // Inject an already-expired request manually
        let past = Instant::now() - Duration::from_secs(10);
        let expired = RenderRequest::with_deadline(99, 0, 10, RenderPriority::Normal, past);
        q.requests.push(expired);
        q.submit(100, 110, RenderPriority::Normal);
        let removed = q.purge_expired();
        assert_eq!(removed, 1);
        assert_eq!(q.pending_count(), 1);
    }

    #[test]
    fn test_render_queue_next_priority() {
        let mut q = RenderQueue::new();
        q.submit(0, 23, RenderPriority::Interactive);
        assert_eq!(q.next_priority(), Some(RenderPriority::Interactive));
    }

    #[test]
    fn test_render_queue_peek_all() {
        let mut q = RenderQueue::new();
        q.submit(0, 23, RenderPriority::Normal);
        q.submit(24, 47, RenderPriority::Background);
        assert_eq!(q.peek_all().len(), 2);
    }

    #[test]
    fn test_render_queue_is_empty() {
        let q = RenderQueue::new();
        assert!(q.is_empty());
    }
}
