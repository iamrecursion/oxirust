//! Time-based search for video events.
//!
//! Provides an interval-tree backed index over [`TemporalEvent`]s for
//! efficient overlap queries, plus a high-level [`TemporalSearcher`].

use std::collections::HashSet;

// ──────────────────────────────────────────────────────────────────────────────
// TimeRange
// ──────────────────────────────────────────────────────────────────────────────

/// A closed time interval expressed in milliseconds from the start of a media
/// asset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeRange {
    /// Start of the range in milliseconds (inclusive).
    pub start_ms: u64,
    /// End of the range in milliseconds (inclusive).
    pub end_ms: u64,
}

impl TimeRange {
    /// Creates a new `TimeRange`.
    ///
    /// # Panics
    ///
    /// Panics in debug builds when `start_ms > end_ms`.
    #[must_use]
    pub fn new(start_ms: u64, end_ms: u64) -> Self {
        debug_assert!(start_ms <= end_ms, "start_ms must be <= end_ms");
        Self { start_ms, end_ms }
    }

    /// Returns `true` when `t` is within this range (both endpoints inclusive).
    #[must_use]
    pub fn contains(&self, t: u64) -> bool {
        t >= self.start_ms && t <= self.end_ms
    }

    /// Returns `true` when this range overlaps with `other`.
    #[must_use]
    pub fn overlaps(&self, other: &Self) -> bool {
        self.start_ms <= other.end_ms && other.start_ms <= self.end_ms
    }

    /// Returns the duration in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> u64 {
        self.end_ms - self.start_ms
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// TemporalEvent
// ──────────────────────────────────────────────────────────────────────────────

/// A timed event within a media asset.
#[derive(Debug, Clone)]
pub struct TemporalEvent {
    /// Unique event identifier.
    pub id: u64,
    /// ID of the media asset that this event belongs to.
    pub media_id: u64,
    /// Time interval of the event within the media asset.
    pub time_range: TimeRange,
    /// Descriptive tags associated with the event.
    pub tags: Vec<String>,
    /// Relevance score (application-defined).
    pub score: f32,
}

impl TemporalEvent {
    /// Creates a new event.
    #[must_use]
    pub fn new(
        id: u64,
        media_id: u64,
        time_range: TimeRange,
        tags: Vec<String>,
        score: f32,
    ) -> Self {
        Self {
            id,
            media_id,
            time_range,
            tags,
            score,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// TemporalIndex  (simple sorted interval store)
// ──────────────────────────────────────────────────────────────────────────────

/// A simple interval index sorted by `start_ms`.
///
/// Queries use a binary search to find the first potential overlap and then
/// scan forward until no more overlaps are possible.  This is O(n) in the
/// worst case but fast for most real-world queries where results are sparse.
pub struct TemporalIndex {
    events: Vec<TemporalEvent>,
    /// Kept sorted so that `query_range` can binary-search the start.
    sorted: bool,
}

impl TemporalIndex {
    /// Creates a new, empty index.
    #[must_use]
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            sorted: true,
        }
    }

    /// Inserts a `TemporalEvent`.
    pub fn insert(&mut self, event: TemporalEvent) {
        self.events.push(event);
        self.sorted = false;
    }

    /// Returns the number of stored events.
    #[must_use]
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Returns `true` when the index has no events.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Ensures the internal list is sorted by start time.
    fn ensure_sorted(&mut self) {
        if !self.sorted {
            self.events
                .sort_by_key(|e| (e.time_range.start_ms, e.time_range.end_ms));
            self.sorted = true;
        }
    }

    /// Returns all events that overlap the half-open query window
    /// `[start_ms, end_ms]`.
    pub fn query_range(&mut self, start_ms: u64, end_ms: u64) -> Vec<&TemporalEvent> {
        self.ensure_sorted();

        // Binary search for the first event whose start could still overlap.
        // An event at position i overlaps if `events[i].start_ms <= end_ms`.
        let first = self
            .events
            .partition_point(|e| e.time_range.start_ms > end_ms);
        // `partition_point` gives us the count of events that START after `end_ms`;
        // those cannot overlap. We want the first event whose start <= end_ms.
        // Let's do it correctly: scan from 0 up to the last event that could start
        // before or at end_ms.
        let last_possible = self
            .events
            .partition_point(|e| e.time_range.start_ms <= end_ms);

        // Within events[0..last_possible], keep those whose end >= start_ms.
        let _ = first; // suppress unused warning
        self.events[..last_possible]
            .iter()
            .filter(|e| e.time_range.end_ms >= start_ms)
            .collect()
    }
}

impl Default for TemporalIndex {
    fn default() -> Self {
        Self::new()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// TemporalSearchQuery
// ──────────────────────────────────────────────────────────────────────────────

/// Query parameters for [`TemporalSearcher`].
#[derive(Debug, Clone, Default)]
pub struct TemporalSearchQuery {
    /// Optional time-range filter.  When set, only events that overlap this
    /// range are returned.
    pub range: Option<TimeRange>,
    /// Optional minimum duration filter in milliseconds.
    pub min_duration_ms: Option<u64>,
    /// Tag filter – events must contain ALL of these tags to be returned.
    /// An empty vec means "no tag filter".
    pub tags: Vec<String>,
}

impl TemporalSearchQuery {
    /// Creates an empty query (matches everything).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Restricts to events overlapping `range`.
    #[must_use]
    pub fn with_range(mut self, range: TimeRange) -> Self {
        self.range = Some(range);
        self
    }

    /// Restricts to events whose duration is at least `min_ms` milliseconds.
    #[must_use]
    pub fn with_min_duration(mut self, min_ms: u64) -> Self {
        self.min_duration_ms = Some(min_ms);
        self
    }

    /// Restricts to events that carry ALL of the supplied tags.
    #[must_use]
    pub fn with_tags(mut self, tags: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.tags = tags.into_iter().map(Into::into).collect();
        self
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// TemporalSearcher
// ──────────────────────────────────────────────────────────────────────────────

/// High-level temporal searcher that wraps a [`TemporalIndex`].
pub struct TemporalSearcher {
    index: TemporalIndex,
}

impl TemporalSearcher {
    /// Creates a new searcher with an empty index.
    #[must_use]
    pub fn new() -> Self {
        Self {
            index: TemporalIndex::new(),
        }
    }

    /// Inserts an event into the underlying index.
    pub fn insert(&mut self, event: TemporalEvent) {
        self.index.insert(event);
    }

    /// Executes a temporal search.
    ///
    /// Results are sorted by descending `score` then by `start_ms`.
    #[must_use]
    pub fn search(&mut self, query: &TemporalSearchQuery) -> Vec<TemporalEvent> {
        // Determine candidate set.
        let candidates: Vec<&TemporalEvent> = match query.range {
            Some(r) => self.index.query_range(r.start_ms, r.end_ms),
            None => self.index.events.iter().collect(),
        };

        // Build required tag set.
        let required: HashSet<&str> = query.tags.iter().map(String::as_str).collect();

        let mut results: Vec<TemporalEvent> = candidates
            .into_iter()
            .filter(|e| {
                // Duration filter.
                if let Some(min_dur) = query.min_duration_ms {
                    if e.time_range.duration_ms() < min_dur {
                        return false;
                    }
                }
                // Tag filter: every required tag must be present.
                if !required.is_empty() {
                    let event_tags: HashSet<&str> = e.tags.iter().map(String::as_str).collect();
                    if !required.is_subset(&event_tags) {
                        return false;
                    }
                }
                true
            })
            .cloned()
            .collect();

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.time_range.start_ms.cmp(&b.time_range.start_ms))
        });

        results
    }
}

impl Default for TemporalSearcher {
    fn default() -> Self {
        Self::new()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(id: u64, start: u64, end: u64, tags: &[&str], score: f32) -> TemporalEvent {
        TemporalEvent::new(
            id,
            1,
            TimeRange::new(start, end),
            tags.iter().map(|s| s.to_string()).collect(),
            score,
        )
    }

    // ── TimeRange ──

    #[test]
    fn test_time_range_contains() {
        let r = TimeRange::new(1000, 5000);
        assert!(r.contains(1000));
        assert!(r.contains(3000));
        assert!(r.contains(5000));
        assert!(!r.contains(999));
        assert!(!r.contains(5001));
    }

    #[test]
    fn test_time_range_overlaps_partial() {
        let a = TimeRange::new(0, 3000);
        let b = TimeRange::new(2000, 6000);
        assert!(a.overlaps(&b));
        assert!(b.overlaps(&a));
    }

    #[test]
    fn test_time_range_overlaps_adjacent() {
        let a = TimeRange::new(0, 1000);
        let b = TimeRange::new(1000, 2000);
        assert!(a.overlaps(&b));
    }

    #[test]
    fn test_time_range_no_overlap() {
        let a = TimeRange::new(0, 999);
        let b = TimeRange::new(1000, 2000);
        assert!(!a.overlaps(&b));
    }

    #[test]
    fn test_time_range_duration() {
        let r = TimeRange::new(1000, 4000);
        assert_eq!(r.duration_ms(), 3000);
    }

    // ── TemporalIndex ──

    #[test]
    fn test_index_query_range_basic() {
        let mut idx = TemporalIndex::new();
        idx.insert(ev(1, 0, 2000, &[], 1.0));
        idx.insert(ev(2, 3000, 5000, &[], 1.0));
        idx.insert(ev(3, 1000, 4000, &[], 1.0));

        let results = idx.query_range(500, 1500);
        let ids: Vec<u64> = results.iter().map(|e| e.id).collect();
        assert!(ids.contains(&1), "Event 1 should overlap [500,1500]");
        assert!(ids.contains(&3), "Event 3 should overlap [500,1500]");
        assert!(!ids.contains(&2), "Event 2 should not overlap [500,1500]");
    }

    #[test]
    fn test_index_empty() {
        let mut idx = TemporalIndex::new();
        assert!(idx.is_empty());
        let results = idx.query_range(0, 10000);
        assert!(results.is_empty());
    }

    // ── TemporalSearcher ──

    #[test]
    fn test_searcher_range_filter() {
        let mut s = TemporalSearcher::new();
        s.insert(ev(1, 0, 1000, &["action"], 0.9));
        s.insert(ev(2, 5000, 8000, &["drama"], 0.8));

        let q = TemporalSearchQuery::new().with_range(TimeRange::new(0, 2000));
        let results = s.search(&q);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 1);
    }

    #[test]
    fn test_searcher_tag_filter() {
        let mut s = TemporalSearcher::new();
        s.insert(ev(1, 0, 1000, &["action", "hero"], 0.9));
        s.insert(ev(2, 2000, 3000, &["drama"], 0.8));
        s.insert(ev(3, 4000, 5000, &["action"], 0.7));

        let q = TemporalSearchQuery::new().with_tags(["action", "hero"]);
        let results = s.search(&q);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 1);
    }

    #[test]
    fn test_searcher_min_duration_filter() {
        let mut s = TemporalSearcher::new();
        s.insert(ev(1, 0, 500, &[], 1.0)); // 500 ms
        s.insert(ev(2, 0, 2000, &[], 1.0)); // 2000 ms

        let q = TemporalSearchQuery::new().with_min_duration(1000);
        let results = s.search(&q);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 2);
    }

    #[test]
    fn test_searcher_score_ordering() {
        let mut s = TemporalSearcher::new();
        s.insert(ev(1, 0, 1000, &[], 0.3));
        s.insert(ev(2, 0, 1000, &[], 0.9));
        s.insert(ev(3, 0, 1000, &[], 0.6));

        let q = TemporalSearchQuery::new();
        let results = s.search(&q);
        assert_eq!(results[0].id, 2);
        assert_eq!(results[1].id, 3);
        assert_eq!(results[2].id, 1);
    }

    #[test]
    fn test_searcher_no_filter_returns_all() {
        let mut s = TemporalSearcher::new();
        for i in 0..5 {
            s.insert(ev(i, i * 1000, (i + 1) * 1000, &[], 1.0));
        }
        let results = s.search(&TemporalSearchQuery::new());
        assert_eq!(results.len(), 5);
    }
}
