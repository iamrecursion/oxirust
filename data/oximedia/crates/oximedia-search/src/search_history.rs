#![allow(dead_code)]

//! Search history tracking and analytics.
//!
//! This module records past search queries, maintains per-user history,
//! supports history-based autocompletion, and provides aggregate analytics
//! on search patterns over time.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Unique identifier for a user or session.
pub type UserId = u64;

/// A single recorded search event.
#[derive(Debug, Clone)]
pub struct HistoryEntry {
    /// The raw query text.
    pub query: String,
    /// Unix-epoch timestamp (seconds) when the search was executed.
    pub timestamp: u64,
    /// Number of results returned.
    pub result_count: usize,
    /// User / session that issued the query.
    pub user_id: UserId,
}

/// Aggregated statistics for a particular query string.
#[derive(Debug, Clone, Default)]
pub struct QueryStats {
    /// How many times this exact query was executed.
    pub count: usize,
    /// Timestamp of the earliest execution.
    pub first_seen: u64,
    /// Timestamp of the most recent execution.
    pub last_seen: u64,
    /// Average result count across all executions.
    pub avg_results: f64,
}

/// Container holding time-windowed search frequency data.
#[derive(Debug, Clone)]
pub struct FrequencyBucket {
    /// Start of the time window (epoch seconds).
    pub window_start: u64,
    /// End of the time window (epoch seconds).
    pub window_end: u64,
    /// Number of queries within this window.
    pub count: usize,
}

/// In-memory search history store.
#[derive(Debug, Default)]
pub struct SearchHistory {
    /// Per-user list of history entries.
    entries: HashMap<UserId, Vec<HistoryEntry>>,
    /// Global query statistics keyed by normalised query text.
    stats: HashMap<String, QueryStats>,
}

impl SearchHistory {
    /// Create a new empty history store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a new search event.
    pub fn record(&mut self, entry: HistoryEntry) {
        let key = entry.query.to_lowercase();

        // Update per-query stats.
        let qs = self.stats.entry(key).or_default();
        let prev_total = qs.avg_results * qs.count as f64;
        qs.count += 1;
        if qs.first_seen == 0 || entry.timestamp < qs.first_seen {
            qs.first_seen = entry.timestamp;
        }
        if entry.timestamp > qs.last_seen {
            qs.last_seen = entry.timestamp;
        }
        qs.avg_results = (prev_total + entry.result_count as f64) / qs.count as f64;

        // Store in per-user list.
        self.entries.entry(entry.user_id).or_default().push(entry);
    }

    /// Return the N most recent queries for a given user (newest first).
    pub fn recent(&self, user_id: UserId, n: usize) -> Vec<&HistoryEntry> {
        match self.entries.get(&user_id) {
            Some(list) => list.iter().rev().take(n).collect(),
            None => Vec::new(),
        }
    }

    /// Return all entries for a user within a time range (inclusive).
    pub fn range(&self, user_id: UserId, from: u64, to: u64) -> Vec<&HistoryEntry> {
        match self.entries.get(&user_id) {
            Some(list) => list
                .iter()
                .filter(|e| e.timestamp >= from && e.timestamp <= to)
                .collect(),
            None => Vec::new(),
        }
    }

    /// Return entries across all users whose query prefix-matches the given text.
    pub fn autocomplete(&self, prefix: &str, limit: usize) -> Vec<String> {
        let lower = prefix.to_lowercase();
        let mut results: Vec<(&String, &QueryStats)> = self
            .stats
            .iter()
            .filter(|(k, _)| k.starts_with(&lower))
            .collect();
        results.sort_by(|a, b| b.1.count.cmp(&a.1.count));
        results
            .into_iter()
            .take(limit)
            .map(|(k, _)| k.clone())
            .collect()
    }

    /// Return the top-N most-executed queries globally.
    pub fn top_queries(&self, n: usize) -> Vec<(String, usize)> {
        let mut sorted: Vec<_> = self.stats.iter().collect();
        sorted.sort_by(|a, b| b.1.count.cmp(&a.1.count));
        sorted
            .into_iter()
            .take(n)
            .map(|(k, v)| (k.clone(), v.count))
            .collect()
    }

    /// Compute frequency buckets for a given user within a window.
    pub fn frequency_buckets(
        &self,
        user_id: UserId,
        from: u64,
        to: u64,
        bucket_size: u64,
    ) -> Vec<FrequencyBucket> {
        let entries = self.range(user_id, from, to);
        let mut buckets = Vec::new();
        let mut start = from;
        while start < to {
            let end = (start + bucket_size).min(to);
            let count = entries
                .iter()
                .filter(|e| e.timestamp >= start && e.timestamp < end)
                .count();
            buckets.push(FrequencyBucket {
                window_start: start,
                window_end: end,
                count,
            });
            start = end;
        }
        buckets
    }

    /// Total number of entries stored across all users.
    pub fn total_entries(&self) -> usize {
        self.entries.values().map(|v| v.len()).sum()
    }

    /// Number of distinct users that have history.
    pub fn user_count(&self) -> usize {
        self.entries.len()
    }

    /// Number of distinct normalised queries tracked.
    pub fn distinct_queries(&self) -> usize {
        self.stats.len()
    }

    /// Clear all history for a specific user.
    pub fn clear_user(&mut self, user_id: UserId) {
        self.entries.remove(&user_id);
    }

    /// Clear the entire history store.
    pub fn clear_all(&mut self) {
        self.entries.clear();
        self.stats.clear();
    }

    /// Get stats for a specific query string.
    pub fn query_stats(&self, query: &str) -> Option<&QueryStats> {
        self.stats.get(&query.to_lowercase())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(query: &str, ts: u64, results: usize, uid: UserId) -> HistoryEntry {
        HistoryEntry {
            query: query.to_string(),
            timestamp: ts,
            result_count: results,
            user_id: uid,
        }
    }

    #[test]
    fn test_new_is_empty() {
        let h = SearchHistory::new();
        assert_eq!(h.total_entries(), 0);
        assert_eq!(h.user_count(), 0);
        assert_eq!(h.distinct_queries(), 0);
    }

    #[test]
    fn test_record_single() {
        let mut h = SearchHistory::new();
        h.record(make_entry("hello", 100, 5, 1));
        assert_eq!(h.total_entries(), 1);
        assert_eq!(h.user_count(), 1);
        assert_eq!(h.distinct_queries(), 1);
    }

    #[test]
    fn test_recent_ordering() {
        let mut h = SearchHistory::new();
        h.record(make_entry("a", 1, 1, 1));
        h.record(make_entry("b", 2, 2, 1));
        h.record(make_entry("c", 3, 3, 1));
        let recent = h.recent(1, 2);
        assert_eq!(recent.len(), 2);
        assert_eq!(recent[0].query, "c");
        assert_eq!(recent[1].query, "b");
    }

    #[test]
    fn test_recent_nonexistent_user() {
        let h = SearchHistory::new();
        let recent = h.recent(42, 10);
        assert!(recent.is_empty());
    }

    #[test]
    fn test_range_filter() {
        let mut h = SearchHistory::new();
        h.record(make_entry("a", 10, 1, 1));
        h.record(make_entry("b", 20, 2, 1));
        h.record(make_entry("c", 30, 3, 1));
        let r = h.range(1, 15, 25);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].query, "b");
    }

    #[test]
    fn test_autocomplete_prefix() {
        let mut h = SearchHistory::new();
        h.record(make_entry("sunset", 1, 1, 1));
        h.record(make_entry("sunrise", 2, 1, 1));
        h.record(make_entry("moon", 3, 1, 1));
        let ac = h.autocomplete("sun", 10);
        assert_eq!(ac.len(), 2);
        assert!(ac.contains(&"sunset".to_string()));
        assert!(ac.contains(&"sunrise".to_string()));
    }

    #[test]
    fn test_autocomplete_limit() {
        let mut h = SearchHistory::new();
        for i in 0..10 {
            h.record(make_entry(&format!("sun{i}"), 1, 1, 1));
        }
        let ac = h.autocomplete("sun", 3);
        assert_eq!(ac.len(), 3);
    }

    #[test]
    fn test_top_queries() {
        let mut h = SearchHistory::new();
        for _ in 0..5 {
            h.record(make_entry("popular", 1, 1, 1));
        }
        h.record(make_entry("rare", 1, 1, 2));
        let top = h.top_queries(1);
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].0, "popular");
        assert_eq!(top[0].1, 5);
    }

    #[test]
    fn test_frequency_buckets() {
        let mut h = SearchHistory::new();
        h.record(make_entry("a", 5, 1, 1));
        h.record(make_entry("b", 15, 1, 1));
        h.record(make_entry("c", 25, 1, 1));
        let buckets = h.frequency_buckets(1, 0, 30, 10);
        assert_eq!(buckets.len(), 3);
        assert_eq!(buckets[0].count, 1);
        assert_eq!(buckets[1].count, 1);
        assert_eq!(buckets[2].count, 1);
    }

    #[test]
    fn test_clear_user() {
        let mut h = SearchHistory::new();
        h.record(make_entry("a", 1, 1, 1));
        h.record(make_entry("b", 2, 1, 2));
        h.clear_user(1);
        assert_eq!(h.total_entries(), 1);
        assert_eq!(h.user_count(), 1);
    }

    #[test]
    fn test_clear_all() {
        let mut h = SearchHistory::new();
        h.record(make_entry("a", 1, 1, 1));
        h.record(make_entry("b", 2, 1, 2));
        h.clear_all();
        assert_eq!(h.total_entries(), 0);
        assert_eq!(h.distinct_queries(), 0);
    }

    #[test]
    fn test_query_stats() {
        let mut h = SearchHistory::new();
        h.record(make_entry("Demo", 10, 3, 1));
        h.record(make_entry("demo", 20, 7, 2));
        let qs = h.query_stats("DEMO").expect("should succeed in test");
        assert_eq!(qs.count, 2);
        assert_eq!(qs.first_seen, 10);
        assert_eq!(qs.last_seen, 20);
        assert!((qs.avg_results - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_case_insensitive_normalisation() {
        let mut h = SearchHistory::new();
        h.record(make_entry("FOO", 1, 1, 1));
        h.record(make_entry("foo", 2, 1, 1));
        h.record(make_entry("Foo", 3, 1, 1));
        assert_eq!(h.distinct_queries(), 1);
    }
}
