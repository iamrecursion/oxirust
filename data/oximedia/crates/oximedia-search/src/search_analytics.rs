#![allow(dead_code)]
//! Search query analytics with click-through tracking for relevance tuning.
//!
//! Extends basic query logging with click-through events, session tracking,
//! and click-through rate (CTR) computation to enable data-driven relevance
//! tuning and search quality measurement.
//!
//! # Click-through tracking
//!
//! After a user executes a search and views the results, every result they
//! click on is recorded as a [`ClickEvent`]. This allows computing:
//!
//! - **Click-through rate (CTR)**: fraction of queries where at least one
//!   result was clicked.
//! - **Mean reciprocal rank (MRR)**: average of 1/rank for the first clicked
//!   result, measuring how high relevant results appear.
//! - **Per-query CTR**: how often each query leads to clicks vs. abandonment.
//! - **Position bias analysis**: which result positions get the most clicks.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use uuid::Uuid;

/// A single recorded search event.
#[derive(Debug, Clone)]
pub struct QueryEvent {
    /// The search query string.
    pub query: String,
    /// Unix timestamp (seconds) when the query was executed.
    pub timestamp_secs: u64,
    /// Number of results returned.
    pub results: usize,
    /// Query execution duration in milliseconds.
    pub duration_ms: u64,
    /// Unique session identifier linking this query to subsequent clicks.
    pub session_id: Option<Uuid>,
}

impl QueryEvent {
    /// Create a new `QueryEvent` with the current timestamp.
    pub fn new(query: impl Into<String>, results: usize, duration_ms: u64) -> Self {
        let timestamp_secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        Self {
            query: query.into(),
            timestamp_secs,
            results,
            duration_ms,
            session_id: None,
        }
    }

    /// Create a `QueryEvent` with an explicit timestamp (useful for tests).
    pub fn with_timestamp(
        query: impl Into<String>,
        results: usize,
        duration_ms: u64,
        timestamp_secs: u64,
    ) -> Self {
        Self {
            query: query.into(),
            timestamp_secs,
            results,
            duration_ms,
            session_id: None,
        }
    }

    /// Create a `QueryEvent` with a session ID for click-through tracking.
    pub fn with_session(
        query: impl Into<String>,
        results: usize,
        duration_ms: u64,
        timestamp_secs: u64,
        session_id: Uuid,
    ) -> Self {
        Self {
            query: query.into(),
            timestamp_secs,
            results,
            duration_ms,
            session_id: Some(session_id),
        }
    }

    /// Returns the number of results returned for this query.
    #[must_use]
    pub fn result_count(&self) -> usize {
        self.results
    }

    /// Returns `true` if the query returned no results.
    #[must_use]
    pub fn is_zero_result(&self) -> bool {
        self.results == 0
    }
}

/// A click event recording that a user clicked on a specific search result.
#[derive(Debug, Clone)]
pub struct ClickEvent {
    /// Session ID linking this click to the originating query.
    pub session_id: Uuid,
    /// The asset that was clicked.
    pub asset_id: Uuid,
    /// 0-based position of the clicked result in the result list.
    pub position: usize,
    /// Unix timestamp (seconds) when the click occurred.
    pub timestamp_secs: u64,
    /// Time spent viewing the result in milliseconds (dwell time), if known.
    pub dwell_time_ms: Option<u64>,
}

impl ClickEvent {
    /// Create a new click event.
    pub fn new(session_id: Uuid, asset_id: Uuid, position: usize, timestamp_secs: u64) -> Self {
        Self {
            session_id,
            asset_id,
            position,
            timestamp_secs,
            dwell_time_ms: None,
        }
    }

    /// Create a click event with dwell time.
    pub fn with_dwell(
        session_id: Uuid,
        asset_id: Uuid,
        position: usize,
        timestamp_secs: u64,
        dwell_time_ms: u64,
    ) -> Self {
        Self {
            session_id,
            asset_id,
            position,
            timestamp_secs,
            dwell_time_ms: Some(dwell_time_ms),
        }
    }

    /// Returns `true` if this is a "long click" (dwell > 30 seconds),
    /// commonly used as a signal of user satisfaction.
    #[must_use]
    pub fn is_long_click(&self) -> bool {
        self.dwell_time_ms.map_or(false, |d| d >= 30_000)
    }
}

/// Per-query click-through statistics.
#[derive(Debug, Clone, Default)]
pub struct QueryClickStats {
    /// Total number of times this query was executed.
    pub impressions: usize,
    /// Number of sessions where at least one result was clicked.
    pub sessions_with_clicks: usize,
    /// Total clicks across all sessions for this query.
    pub total_clicks: usize,
    /// Sum of 1/rank for first clicked position (for MRR computation).
    pub reciprocal_rank_sum: f64,
}

impl QueryClickStats {
    /// Click-through rate: fraction of sessions with at least one click.
    #[must_use]
    pub fn ctr(&self) -> f64 {
        if self.impressions == 0 {
            0.0
        } else {
            self.sessions_with_clicks as f64 / self.impressions as f64
        }
    }

    /// Mean reciprocal rank (MRR): average of 1/(position+1) for first clicks.
    #[must_use]
    pub fn mrr(&self) -> f64 {
        if self.sessions_with_clicks == 0 {
            0.0
        } else {
            self.reciprocal_rank_sum / self.sessions_with_clicks as f64
        }
    }
}

/// Position-level click statistics.
#[derive(Debug, Clone, Default)]
pub struct PositionStats {
    /// Number of times a result at this position was shown.
    pub impressions: usize,
    /// Number of clicks on results at this position.
    pub clicks: usize,
}

impl PositionStats {
    /// CTR for this position.
    #[must_use]
    pub fn ctr(&self) -> f64 {
        if self.impressions == 0 {
            0.0
        } else {
            self.clicks as f64 / self.impressions as f64
        }
    }
}

/// Aggregated analytics over recorded search events with click-through tracking.
#[derive(Debug, Default)]
pub struct SearchAnalytics {
    events: Vec<QueryEvent>,
    clicks: Vec<ClickEvent>,
    /// Sessions that have received at least one click, keyed by session_id.
    session_clicks: HashMap<Uuid, Vec<ClickEvent>>,
}

impl SearchAnalytics {
    /// Create a new, empty analytics tracker.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a search event.
    pub fn record_query(&mut self, event: QueryEvent) {
        self.events.push(event);
    }

    /// Record a click event.
    pub fn record_click(&mut self, click: ClickEvent) {
        self.session_clicks
            .entry(click.session_id)
            .or_default()
            .push(click.clone());
        self.clicks.push(click);
    }

    /// Returns total number of recorded queries.
    #[must_use]
    pub fn total_queries(&self) -> usize {
        self.events.len()
    }

    /// Returns total number of recorded clicks.
    #[must_use]
    pub fn total_clicks(&self) -> usize {
        self.clicks.len()
    }

    /// Returns the `n` most popular query strings by occurrence count.
    #[must_use]
    pub fn popular_queries(&self, n: usize) -> Vec<(String, usize)> {
        let mut counts: HashMap<&str, usize> = HashMap::new();
        for e in &self.events {
            *counts.entry(e.query.as_str()).or_insert(0) += 1;
        }
        let mut sorted: Vec<(String, usize)> = counts
            .into_iter()
            .map(|(q, c)| (q.to_string(), c))
            .collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        sorted.truncate(n);
        sorted
    }

    /// Returns the `n` most common queries that returned zero results.
    #[must_use]
    pub fn zero_result_queries(&self, n: usize) -> Vec<(String, usize)> {
        let mut counts: HashMap<&str, usize> = HashMap::new();
        for e in &self.events {
            if e.is_zero_result() {
                *counts.entry(e.query.as_str()).or_insert(0) += 1;
            }
        }
        let mut sorted: Vec<(String, usize)> = counts
            .into_iter()
            .map(|(q, c)| (q.to_string(), c))
            .collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        sorted.truncate(n);
        sorted
    }

    /// Returns the average result count across all recorded events.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn avg_result_count(&self) -> f64 {
        if self.events.is_empty() {
            return 0.0;
        }
        let total: usize = self.events.iter().map(|e| e.results).sum();
        total as f64 / self.events.len() as f64
    }

    /// Returns the average query duration in milliseconds.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn avg_duration_ms(&self) -> f64 {
        if self.events.is_empty() {
            return 0.0;
        }
        let total: u64 = self.events.iter().map(|e| e.duration_ms).sum();
        total as f64 / self.events.len() as f64
    }

    /// Compute the overall click-through rate across all queries with sessions.
    ///
    /// CTR = (sessions with at least one click) / (total sessions).
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn overall_ctr(&self) -> f64 {
        let sessions_with_queries: HashMap<Uuid, bool> = self
            .events
            .iter()
            .filter_map(|e| e.session_id.map(|sid| (sid, false)))
            .collect();

        if sessions_with_queries.is_empty() {
            return 0.0;
        }

        let sessions_with_clicks = sessions_with_queries
            .keys()
            .filter(|sid| self.session_clicks.contains_key(sid))
            .count();

        sessions_with_clicks as f64 / sessions_with_queries.len() as f64
    }

    /// Compute per-query click-through statistics.
    ///
    /// Returns a map from normalized query string to [`QueryClickStats`].
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn per_query_click_stats(&self) -> HashMap<String, QueryClickStats> {
        let mut stats: HashMap<String, QueryClickStats> = HashMap::new();

        // Count impressions per query
        for event in &self.events {
            let key = event.query.to_lowercase();
            stats.entry(key).or_default().impressions += 1;
        }

        // Build session->query mapping
        let mut session_query: HashMap<Uuid, String> = HashMap::new();
        for event in &self.events {
            if let Some(sid) = event.session_id {
                session_query.insert(sid, event.query.to_lowercase());
            }
        }

        // Count clicks per query via sessions
        for (session_id, clicks) in &self.session_clicks {
            if let Some(query) = session_query.get(session_id) {
                let qs = stats.entry(query.clone()).or_default();
                qs.sessions_with_clicks += 1;
                qs.total_clicks += clicks.len();

                // MRR: use the first click's position (lowest position = highest rank)
                if let Some(first_click) = clicks.iter().min_by_key(|c| c.position) {
                    qs.reciprocal_rank_sum += 1.0 / (first_click.position as f64 + 1.0);
                }
            }
        }

        stats
    }

    /// Compute click statistics per result position (0-based).
    ///
    /// Returns a vector of [`PositionStats`] for positions 0..max_position.
    #[must_use]
    pub fn position_click_stats(&self, max_positions: usize) -> Vec<PositionStats> {
        let mut stats = vec![PositionStats::default(); max_positions];

        // Every query with results contributes an impression at each position
        // up to min(result_count, max_positions)
        for event in &self.events {
            let n = event.results.min(max_positions);
            for pos in 0..n {
                stats[pos].impressions += 1;
            }
        }

        // Clicks contribute to the respective position
        for click in &self.clicks {
            if click.position < max_positions {
                stats[click.position].clicks += 1;
            }
        }

        stats
    }

    /// Returns the `n` queries with the lowest CTR (most abandoned queries).
    ///
    /// Only includes queries with at least `min_impressions` to avoid noise.
    #[must_use]
    pub fn lowest_ctr_queries(&self, n: usize, min_impressions: usize) -> Vec<(String, f64)> {
        let stats = self.per_query_click_stats();
        let mut sorted: Vec<(String, f64)> = stats
            .into_iter()
            .filter(|(_, qs)| qs.impressions >= min_impressions)
            .map(|(q, qs)| (q, qs.ctr()))
            .collect();
        sorted.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        sorted.truncate(n);
        sorted
    }

    /// Returns the `n` most-clicked asset IDs across all queries.
    #[must_use]
    pub fn most_clicked_assets(&self, n: usize) -> Vec<(Uuid, usize)> {
        let mut counts: HashMap<Uuid, usize> = HashMap::new();
        for click in &self.clicks {
            *counts.entry(click.asset_id).or_insert(0) += 1;
        }
        let mut sorted: Vec<(Uuid, usize)> = counts.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        sorted.truncate(n);
        sorted
    }

    /// Average dwell time across all clicks that have dwell data.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn avg_dwell_time_ms(&self) -> f64 {
        let (total, count) = self.clicks.iter().fold((0u64, 0usize), |(sum, cnt), c| {
            if let Some(dwell) = c.dwell_time_ms {
                (sum + dwell, cnt + 1)
            } else {
                (sum, cnt)
            }
        });
        if count == 0 {
            0.0
        } else {
            total as f64 / count as f64
        }
    }

    /// Fraction of clicks that are "long clicks" (dwell >= 30s).
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn long_click_rate(&self) -> f64 {
        let with_dwell: Vec<&ClickEvent> = self
            .clicks
            .iter()
            .filter(|c| c.dwell_time_ms.is_some())
            .collect();
        if with_dwell.is_empty() {
            return 0.0;
        }
        let long_count = with_dwell.iter().filter(|c| c.is_long_click()).count();
        long_count as f64 / with_dwell.len() as f64
    }

    /// Clear all recorded events and clicks.
    pub fn clear(&mut self) {
        self.events.clear();
        self.clicks.clear();
        self.session_clicks.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_event(query: &str, results: usize) -> QueryEvent {
        QueryEvent::with_timestamp(query, results, 10, 1_000_000)
    }

    fn make_session_event(query: &str, results: usize, session_id: Uuid) -> QueryEvent {
        QueryEvent::with_session(query, results, 10, 1_000_000, session_id)
    }

    #[test]
    fn test_query_event_result_count() {
        let e = make_event("video", 42);
        assert_eq!(e.result_count(), 42);
    }

    #[test]
    fn test_query_event_is_zero_result_true() {
        let e = make_event("xyzqrs", 0);
        assert!(e.is_zero_result());
    }

    #[test]
    fn test_query_event_is_zero_result_false() {
        let e = make_event("video", 5);
        assert!(!e.is_zero_result());
    }

    #[test]
    fn test_analytics_empty() {
        let a = SearchAnalytics::new();
        assert_eq!(a.total_queries(), 0);
        assert_eq!(a.total_clicks(), 0);
    }

    #[test]
    fn test_record_query_increments_count() {
        let mut a = SearchAnalytics::new();
        a.record_query(make_event("video", 10));
        assert_eq!(a.total_queries(), 1);
    }

    #[test]
    fn test_popular_queries_ordering() {
        let mut a = SearchAnalytics::new();
        a.record_query(make_event("audio", 5));
        a.record_query(make_event("video", 5));
        a.record_query(make_event("video", 3));
        let popular = a.popular_queries(2);
        assert_eq!(popular[0].0, "video");
        assert_eq!(popular[0].1, 2);
    }

    #[test]
    fn test_popular_queries_limited() {
        let mut a = SearchAnalytics::new();
        for i in 0..10 {
            a.record_query(make_event(&format!("q{}", i), 1));
        }
        assert_eq!(a.popular_queries(3).len(), 3);
    }

    #[test]
    fn test_zero_result_queries() {
        let mut a = SearchAnalytics::new();
        a.record_query(make_event("missing", 0));
        a.record_query(make_event("missing", 0));
        a.record_query(make_event("video", 10));
        let zero = a.zero_result_queries(5);
        assert_eq!(zero.len(), 1);
        assert_eq!(zero[0].0, "missing");
        assert_eq!(zero[0].1, 2);
    }

    #[test]
    fn test_avg_result_count() {
        let mut a = SearchAnalytics::new();
        a.record_query(make_event("a", 10));
        a.record_query(make_event("b", 20));
        assert!((a.avg_result_count() - 15.0).abs() < 1e-9);
    }

    #[test]
    fn test_avg_result_count_empty() {
        let a = SearchAnalytics::new();
        assert_eq!(a.avg_result_count(), 0.0);
    }

    #[test]
    fn test_avg_duration_ms() {
        let mut a = SearchAnalytics::new();
        a.record_query(QueryEvent::with_timestamp("a", 1, 100, 0));
        a.record_query(QueryEvent::with_timestamp("b", 1, 200, 0));
        assert!((a.avg_duration_ms() - 150.0).abs() < 1e-9);
    }

    #[test]
    fn test_clear() {
        let mut a = SearchAnalytics::new();
        a.record_query(make_event("x", 1));
        a.record_click(ClickEvent::new(Uuid::new_v4(), Uuid::new_v4(), 0, 100));
        a.clear();
        assert_eq!(a.total_queries(), 0);
        assert_eq!(a.total_clicks(), 0);
    }

    #[test]
    fn test_popular_queries_empty() {
        let a = SearchAnalytics::new();
        assert!(a.popular_queries(5).is_empty());
    }

    #[test]
    fn test_zero_result_queries_none() {
        let mut a = SearchAnalytics::new();
        a.record_query(make_event("video", 10));
        assert!(a.zero_result_queries(5).is_empty());
    }

    // ── Click-through tracking tests ──

    #[test]
    fn test_record_click() {
        let mut a = SearchAnalytics::new();
        let sid = Uuid::new_v4();
        let aid = Uuid::new_v4();
        a.record_click(ClickEvent::new(sid, aid, 0, 1_000_001));
        assert_eq!(a.total_clicks(), 1);
    }

    #[test]
    fn test_click_event_long_click() {
        let sid = Uuid::new_v4();
        let aid = Uuid::new_v4();
        let short = ClickEvent::with_dwell(sid, aid, 0, 100, 5_000);
        assert!(!short.is_long_click());
        let long = ClickEvent::with_dwell(sid, aid, 0, 100, 60_000);
        assert!(long.is_long_click());
        let no_dwell = ClickEvent::new(sid, aid, 0, 100);
        assert!(!no_dwell.is_long_click());
    }

    #[test]
    fn test_overall_ctr() {
        let mut a = SearchAnalytics::new();
        let s1 = Uuid::new_v4();
        let s2 = Uuid::new_v4();
        let s3 = Uuid::new_v4();

        // 3 sessions, only 2 get clicks
        a.record_query(make_session_event("video", 10, s1));
        a.record_query(make_session_event("audio", 5, s2));
        a.record_query(make_session_event("codec", 3, s3));

        a.record_click(ClickEvent::new(s1, Uuid::new_v4(), 0, 100));
        a.record_click(ClickEvent::new(s2, Uuid::new_v4(), 1, 100));

        let ctr = a.overall_ctr();
        // 2 out of 3 sessions clicked
        assert!((ctr - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_overall_ctr_no_sessions() {
        let mut a = SearchAnalytics::new();
        // Events without session IDs
        a.record_query(make_event("video", 10));
        assert_eq!(a.overall_ctr(), 0.0);
    }

    #[test]
    fn test_per_query_click_stats() {
        let mut a = SearchAnalytics::new();
        let s1 = Uuid::new_v4();
        let s2 = Uuid::new_v4();
        let s3 = Uuid::new_v4();

        // "video" queried 3 times, clicked in 2 sessions
        a.record_query(make_session_event("video", 10, s1));
        a.record_query(make_session_event("video", 10, s2));
        a.record_query(make_session_event("video", 10, s3));

        a.record_click(ClickEvent::new(s1, Uuid::new_v4(), 0, 100)); // position 0
        a.record_click(ClickEvent::new(s1, Uuid::new_v4(), 2, 101)); // another click in s1
        a.record_click(ClickEvent::new(s2, Uuid::new_v4(), 3, 100)); // position 3

        let stats = a.per_query_click_stats();
        let video_stats = stats.get("video").expect("video should have stats");
        assert_eq!(video_stats.impressions, 3);
        assert_eq!(video_stats.sessions_with_clicks, 2);
        assert_eq!(video_stats.total_clicks, 3);
        // CTR = 2/3
        assert!((video_stats.ctr() - 2.0 / 3.0).abs() < 1e-9);
        // MRR: s1 first click at pos 0 -> 1/1 = 1.0, s2 first click at pos 3 -> 1/4 = 0.25
        // MRR = (1.0 + 0.25) / 2 = 0.625
        assert!((video_stats.mrr() - 0.625).abs() < 1e-9);
    }

    #[test]
    fn test_per_query_click_stats_empty() {
        let a = SearchAnalytics::new();
        let stats = a.per_query_click_stats();
        assert!(stats.is_empty());
    }

    #[test]
    fn test_query_click_stats_no_clicks() {
        let qs = QueryClickStats::default();
        assert_eq!(qs.ctr(), 0.0);
        assert_eq!(qs.mrr(), 0.0);
    }

    #[test]
    fn test_position_click_stats() {
        let mut a = SearchAnalytics::new();
        let s1 = Uuid::new_v4();
        let s2 = Uuid::new_v4();

        a.record_query(make_session_event("video", 5, s1));
        a.record_query(make_session_event("audio", 3, s2));

        a.record_click(ClickEvent::new(s1, Uuid::new_v4(), 0, 100));
        a.record_click(ClickEvent::new(s1, Uuid::new_v4(), 2, 101));
        a.record_click(ClickEvent::new(s2, Uuid::new_v4(), 0, 100));

        let pos_stats = a.position_click_stats(5);
        assert_eq!(pos_stats.len(), 5);
        // Position 0: 2 impressions (both queries had >= 1 result), 2 clicks
        assert_eq!(pos_stats[0].impressions, 2);
        assert_eq!(pos_stats[0].clicks, 2);
        assert!((pos_stats[0].ctr() - 1.0).abs() < 1e-9);

        // Position 2: 2 impressions (video has 5 results, audio has 3), 1 click
        assert_eq!(pos_stats[2].impressions, 2);
        assert_eq!(pos_stats[2].clicks, 1);
    }

    #[test]
    fn test_position_stats_ctr() {
        let ps = PositionStats {
            impressions: 100,
            clicks: 25,
        };
        assert!((ps.ctr() - 0.25).abs() < 1e-9);
    }

    #[test]
    fn test_position_stats_ctr_zero() {
        let ps = PositionStats::default();
        assert_eq!(ps.ctr(), 0.0);
    }

    #[test]
    fn test_lowest_ctr_queries() {
        let mut a = SearchAnalytics::new();
        let s1 = Uuid::new_v4();
        let s2 = Uuid::new_v4();
        let s3 = Uuid::new_v4();
        let s4 = Uuid::new_v4();

        // "abandoned" queried 3 times, never clicked
        a.record_query(make_session_event("abandoned", 10, s1));
        a.record_query(make_session_event("abandoned", 10, s2));
        a.record_query(make_session_event("abandoned", 10, s3));

        // "popular" queried 2 times, clicked both
        a.record_query(make_session_event("popular", 10, s4));
        a.record_click(ClickEvent::new(s4, Uuid::new_v4(), 0, 100));

        let lowest = a.lowest_ctr_queries(5, 2);
        assert!(!lowest.is_empty());
        assert_eq!(lowest[0].0, "abandoned");
        assert!((lowest[0].1).abs() < 1e-9); // CTR = 0
    }

    #[test]
    fn test_most_clicked_assets() {
        let mut a = SearchAnalytics::new();
        let asset1 = Uuid::new_v4();
        let asset2 = Uuid::new_v4();

        a.record_click(ClickEvent::new(Uuid::new_v4(), asset1, 0, 100));
        a.record_click(ClickEvent::new(Uuid::new_v4(), asset1, 0, 101));
        a.record_click(ClickEvent::new(Uuid::new_v4(), asset1, 0, 102));
        a.record_click(ClickEvent::new(Uuid::new_v4(), asset2, 0, 103));

        let top = a.most_clicked_assets(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, asset1);
        assert_eq!(top[0].1, 3);
        assert_eq!(top[1].0, asset2);
        assert_eq!(top[1].1, 1);
    }

    #[test]
    fn test_avg_dwell_time() {
        let mut a = SearchAnalytics::new();
        a.record_click(ClickEvent::with_dwell(
            Uuid::new_v4(),
            Uuid::new_v4(),
            0,
            100,
            10_000,
        ));
        a.record_click(ClickEvent::with_dwell(
            Uuid::new_v4(),
            Uuid::new_v4(),
            0,
            101,
            20_000,
        ));
        // Click without dwell should be excluded from average
        a.record_click(ClickEvent::new(Uuid::new_v4(), Uuid::new_v4(), 0, 102));

        assert!((a.avg_dwell_time_ms() - 15_000.0).abs() < 1e-9);
    }

    #[test]
    fn test_avg_dwell_time_empty() {
        let a = SearchAnalytics::new();
        assert_eq!(a.avg_dwell_time_ms(), 0.0);
    }

    #[test]
    fn test_long_click_rate() {
        let mut a = SearchAnalytics::new();
        a.record_click(ClickEvent::with_dwell(
            Uuid::new_v4(),
            Uuid::new_v4(),
            0,
            100,
            5_000,
        )); // short
        a.record_click(ClickEvent::with_dwell(
            Uuid::new_v4(),
            Uuid::new_v4(),
            0,
            101,
            60_000,
        )); // long
        a.record_click(ClickEvent::with_dwell(
            Uuid::new_v4(),
            Uuid::new_v4(),
            0,
            102,
            45_000,
        )); // long
            // No-dwell click excluded from rate
        a.record_click(ClickEvent::new(Uuid::new_v4(), Uuid::new_v4(), 0, 103));

        // 2 long out of 3 with dwell data
        assert!((a.long_click_rate() - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_long_click_rate_empty() {
        let a = SearchAnalytics::new();
        assert_eq!(a.long_click_rate(), 0.0);
    }

    #[test]
    fn test_query_event_session_constructor() {
        let sid = Uuid::new_v4();
        let e = QueryEvent::with_session("test", 5, 10, 1000, sid);
        assert_eq!(e.session_id, Some(sid));
        assert_eq!(e.query, "test");
    }
}
