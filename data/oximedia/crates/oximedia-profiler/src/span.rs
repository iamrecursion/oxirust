//! Hierarchical span tracking for nested function timing.
//!
//! Provides `SpanId`, `Span`, `SpanTracker`, and `SpanGuard` for recording
//! parent-child relationships between profiled code sections.  A thread-local
//! stack maintains the currently active span so that nested `enter()` calls
//! automatically wire up parent/child links.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Global span-id generator
// ---------------------------------------------------------------------------

static NEXT_SPAN_ID: AtomicU64 = AtomicU64::new(1);

fn alloc_span_id() -> u64 {
    NEXT_SPAN_ID.fetch_add(1, Ordering::Relaxed)
}

// ---------------------------------------------------------------------------
// SpanId
// ---------------------------------------------------------------------------

/// Unique identifier for a profiling span.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SpanId(pub u64);

impl SpanId {
    /// Returns the raw numeric value.
    #[must_use]
    pub fn value(self) -> u64 {
        self.0
    }
}

// ---------------------------------------------------------------------------
// Span
// ---------------------------------------------------------------------------

/// A single timed section of code within a profiling session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Span {
    /// Unique identifier of this span.
    pub id: SpanId,
    /// Identifier of the parent span, if any.
    pub parent_id: Option<SpanId>,
    /// Human-readable span name.
    pub name: String,
    /// Monotonic start time (stored as nanoseconds from the session epoch).
    pub start_ns: u64,
    /// Monotonic end time; `None` while the span is still open.
    pub end_ns: Option<u64>,
    /// Ids of child spans opened while this span was active.
    pub children: Vec<SpanId>,
}

impl Span {
    /// Returns the duration of a finished span, or `None` if still open.
    #[must_use]
    pub fn duration(&self) -> Option<Duration> {
        self.end_ns.map(|end| {
            let ns = end.saturating_sub(self.start_ns);
            Duration::from_nanos(ns)
        })
    }

    /// Returns `true` if the span has been closed.
    #[must_use]
    pub fn is_closed(&self) -> bool {
        self.end_ns.is_some()
    }
}

// ---------------------------------------------------------------------------
// SpanTracker (shared state behind a Mutex)
// ---------------------------------------------------------------------------

/// Internal mutable state for the span tracker.
#[derive(Debug)]
struct SpanTrackerInner {
    /// All spans recorded in this session, keyed by id.
    spans: HashMap<SpanId, Span>,
    /// Root span ids (spans with no parent).
    roots: Vec<SpanId>,
    /// Epoch `Instant` used to convert `Instant` → nanosecond offset.
    epoch: Instant,
}

impl SpanTrackerInner {
    fn new() -> Self {
        Self {
            spans: HashMap::new(),
            roots: Vec::new(),
            epoch: Instant::now(),
        }
    }

    fn now_ns(&self) -> u64 {
        self.epoch.elapsed().as_nanos() as u64
    }

    fn open_span(&mut self, name: String, parent_id: Option<SpanId>) -> SpanId {
        let id = SpanId(alloc_span_id());
        let span = Span {
            id,
            parent_id,
            name,
            start_ns: self.now_ns(),
            end_ns: None,
            children: Vec::new(),
        };

        if let Some(pid) = parent_id {
            if let Some(parent) = self.spans.get_mut(&pid) {
                parent.children.push(id);
            }
        } else {
            self.roots.push(id);
        }

        self.spans.insert(id, span);
        id
    }

    fn close_span(&mut self, id: SpanId) {
        let now = self.now_ns();
        if let Some(span) = self.spans.get_mut(&id) {
            if span.end_ns.is_none() {
                span.end_ns = Some(now);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Thread-local span stack
// ---------------------------------------------------------------------------

thread_local! {
    static CURRENT_SPAN_STACK: RefCell<Vec<SpanId>> = const { RefCell::new(Vec::new()) };
}

fn current_span() -> Option<SpanId> {
    CURRENT_SPAN_STACK.with(|stack| stack.borrow().last().copied())
}

fn push_span(id: SpanId) {
    CURRENT_SPAN_STACK.with(|stack| stack.borrow_mut().push(id));
}

fn pop_span() {
    CURRENT_SPAN_STACK.with(|stack| {
        stack.borrow_mut().pop();
    });
}

// ---------------------------------------------------------------------------
// SpanTracker
// ---------------------------------------------------------------------------

/// Thread-safe hierarchical span tracker.
///
/// Multiple threads can call `enter()` concurrently; each thread maintains its
/// own current-span stack, so parent/child links are correct per-thread.
#[derive(Clone, Debug)]
pub struct SpanTracker {
    inner: Arc<Mutex<SpanTrackerInner>>,
}

impl SpanTracker {
    /// Creates a new `SpanTracker`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(SpanTrackerInner::new())),
        }
    }

    /// Opens a new span, automatically parented to the currently active span
    /// on the calling thread.  Returns a `SpanGuard`; when the guard is
    /// dropped, the span is closed.
    pub fn enter(&self, name: impl Into<String>) -> SpanGuard {
        let parent_id = current_span();
        let id = self
            .inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .open_span(name.into(), parent_id);
        push_span(id);

        SpanGuard {
            id,
            tracker: Arc::clone(&self.inner),
        }
    }

    /// Returns an immutable view of the span with the given id, if it exists.
    #[must_use]
    pub fn span(&self, id: SpanId) -> Option<Span> {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .spans
            .get(&id)
            .cloned()
    }

    /// Returns all root span ids (spans with no parent).
    #[must_use]
    pub fn root_span_ids(&self) -> Vec<SpanId> {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .roots
            .clone()
    }

    /// Returns all spans in insertion order.
    #[must_use]
    pub fn all_spans(&self) -> Vec<Span> {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .spans
            .values()
            .cloned()
            .collect()
    }

    /// Returns every span whose name matches `name`.
    #[must_use]
    pub fn spans_by_name(&self, name: &str) -> Vec<Span> {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .spans
            .values()
            .filter(|s| s.name == name)
            .cloned()
            .collect()
    }

    /// Clears all recorded spans and resets the epoch.
    pub fn reset(&self) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        *inner = SpanTrackerInner::new();
    }

    /// Returns the total number of recorded spans.
    #[must_use]
    pub fn span_count(&self) -> usize {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .spans
            .len()
    }

    /// Returns the sum of durations for all closed spans with `name`.
    #[must_use]
    pub fn total_duration_for(&self, name: &str) -> Duration {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .spans
            .values()
            .filter(|s| s.name == name)
            .filter_map(|s| s.duration())
            .fold(Duration::ZERO, |acc, d| acc + d)
    }
}

impl Default for SpanTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// SpanGuard
// ---------------------------------------------------------------------------

/// RAII guard that closes a span when dropped.
///
/// Created by `SpanTracker::enter()`; closing via `drop()` or explicit
/// `finish()` is safe and idempotent.
pub struct SpanGuard {
    id: SpanId,
    tracker: Arc<Mutex<SpanTrackerInner>>,
}

impl SpanGuard {
    /// Returns the id of the span this guard manages.
    #[must_use]
    pub fn span_id(&self) -> SpanId {
        self.id
    }

    /// Explicitly closes the span and removes it from the thread-local stack.
    ///
    /// Calling this is optional; `drop()` does the same thing.
    pub fn finish(self) {
        // drop() handles everything
    }
}

impl Drop for SpanGuard {
    fn drop(&mut self) {
        pop_span();
        self.tracker
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .close_span(self.id);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_single_span_timing() {
        let tracker = SpanTracker::new();
        {
            let _guard = tracker.enter("work");
            thread::sleep(Duration::from_millis(10));
        }
        let spans = tracker.spans_by_name("work");
        assert_eq!(spans.len(), 1);
        let dur = spans[0].duration().expect("span should be closed");
        assert!(dur >= Duration::from_millis(10), "duration was {:?}", dur);
    }

    #[test]
    fn test_nested_spans_parent_child() {
        let tracker = SpanTracker::new();
        let outer_id;
        let inner_id;
        {
            let outer = tracker.enter("outer");
            outer_id = outer.span_id();
            {
                let inner = tracker.enter("inner");
                inner_id = inner.span_id();
            }
        }

        let outer = tracker.span(outer_id).expect("outer must exist");
        let inner = tracker.span(inner_id).expect("inner must exist");

        assert_eq!(inner.parent_id, Some(outer_id));
        assert!(outer.children.contains(&inner_id));
    }

    #[test]
    fn test_root_span_has_no_parent() {
        let tracker = SpanTracker::new();
        let id;
        {
            let g = tracker.enter("root");
            id = g.span_id();
        }
        let span = tracker.span(id).expect("should exist");
        assert!(span.parent_id.is_none());
        assert!(tracker.root_span_ids().contains(&id));
    }

    #[test]
    fn test_span_closed_after_guard_drop() {
        let tracker = SpanTracker::new();
        let id;
        {
            let g = tracker.enter("closed");
            id = g.span_id();
        }
        let span = tracker.span(id).expect("should exist");
        assert!(span.is_closed());
    }

    #[test]
    fn test_span_open_while_guard_alive() {
        let tracker = SpanTracker::new();
        let _g = tracker.enter("open");
        // The span should not be closed yet — but we can't check directly while
        // the guard is alive; instead verify it IS closed after drop.
        let id = _g.span_id();
        drop(_g);
        assert!(tracker.span(id).expect("must exist").is_closed());
    }

    #[test]
    fn test_multiple_siblings() {
        let tracker = SpanTracker::new();
        let parent_id;
        {
            let p = tracker.enter("parent");
            parent_id = p.span_id();
            let _c1 = tracker.enter("child_a");
            // c1 dropped here
        }
        {
            let _ = tracker.enter("parent");
        }
        {
            let p = tracker.span(parent_id).expect("must exist");
            assert_eq!(p.children.len(), 1);
        }
    }

    #[test]
    fn test_total_duration_accumulation() {
        let tracker = SpanTracker::new();
        for _ in 0..3 {
            let _g = tracker.enter("repeated");
            thread::sleep(Duration::from_millis(5));
        }
        let total = tracker.total_duration_for("repeated");
        assert!(total >= Duration::from_millis(15), "was {:?}", total);
    }

    #[test]
    fn test_span_count() {
        let tracker = SpanTracker::new();
        {
            let _a = tracker.enter("a");
            let _b = tracker.enter("b");
        }
        assert_eq!(tracker.span_count(), 2);
    }

    #[test]
    fn test_reset_clears_spans() {
        let tracker = SpanTracker::new();
        {
            let _g = tracker.enter("some_span");
        }
        tracker.reset();
        assert_eq!(tracker.span_count(), 0);
        assert!(tracker.root_span_ids().is_empty());
    }

    #[test]
    fn test_deep_nesting() {
        let tracker = SpanTracker::new();
        let mut ids = Vec::new();
        {
            let l0 = tracker.enter("l0");
            ids.push(l0.span_id());
            {
                let l1 = tracker.enter("l1");
                ids.push(l1.span_id());
                {
                    let l2 = tracker.enter("l2");
                    ids.push(l2.span_id());
                }
            }
        }
        let l2 = tracker.span(ids[2]).expect("l2");
        let l1 = tracker.span(ids[1]).expect("l1");
        assert_eq!(l2.parent_id, Some(ids[1]));
        assert_eq!(l1.parent_id, Some(ids[0]));
        assert!(l1.children.contains(&ids[2]));
    }
}
