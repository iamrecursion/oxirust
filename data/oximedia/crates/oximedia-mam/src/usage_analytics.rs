#![allow(dead_code)]
//! Asset usage analytics and access pattern tracking.
//!
//! Collects and analyses asset access events to produce per-asset usage
//! statistics, popularity rankings, and access trend summaries useful for
//! capacity planning and content strategy.

use std::collections::HashMap;

/// Kinds of access events recorded against an asset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AccessKind {
    /// The asset was viewed / previewed.
    View,
    /// The asset was downloaded.
    Download,
    /// The asset metadata was edited.
    Edit,
    /// The asset was shared or exported.
    Share,
    /// The asset was streamed.
    Stream,
    /// The asset was used in a project.
    ProjectUse,
}

impl AccessKind {
    /// Return a static label string for the access kind.
    pub fn label(self) -> &'static str {
        match self {
            Self::View => "view",
            Self::Download => "download",
            Self::Edit => "edit",
            Self::Share => "share",
            Self::Stream => "stream",
            Self::ProjectUse => "project_use",
        }
    }
}

/// A single recorded access event.
#[derive(Debug, Clone)]
pub struct AccessEvent {
    /// Asset identifier.
    pub asset_id: String,
    /// User who performed the access.
    pub user_id: String,
    /// Kind of access.
    pub kind: AccessKind,
    /// Unix-epoch timestamp (seconds).
    pub timestamp: u64,
}

impl AccessEvent {
    /// Create a new access event.
    pub fn new(
        asset_id: impl Into<String>,
        user_id: impl Into<String>,
        kind: AccessKind,
        timestamp: u64,
    ) -> Self {
        Self {
            asset_id: asset_id.into(),
            user_id: user_id.into(),
            kind,
            timestamp,
        }
    }
}

/// Per-asset usage statistics.
#[derive(Debug, Clone, PartialEq)]
pub struct AssetUsageStats {
    /// Total number of access events.
    pub total_accesses: u64,
    /// Breakdown by access kind.
    pub by_kind: HashMap<AccessKind, u64>,
    /// Number of unique users who accessed this asset.
    pub unique_users: u64,
    /// Timestamp of the first known access.
    pub first_access: Option<u64>,
    /// Timestamp of the most recent access.
    pub last_access: Option<u64>,
}

impl Default for AssetUsageStats {
    fn default() -> Self {
        Self {
            total_accesses: 0,
            by_kind: HashMap::new(),
            unique_users: 0,
            first_access: None,
            last_access: None,
        }
    }
}

/// Popularity ranking entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PopularityEntry {
    /// Asset identifier.
    pub asset_id: String,
    /// Total number of accesses used for ranking.
    pub access_count: u64,
}

/// Aggregate summary over a collection of events.
#[derive(Debug, Clone)]
pub struct AnalyticsSummary {
    /// Total events processed.
    pub total_events: u64,
    /// Number of distinct assets referenced.
    pub distinct_assets: u64,
    /// Number of distinct users.
    pub distinct_users: u64,
    /// Most common access kind (if any).
    pub top_kind: Option<AccessKind>,
}

/// In-memory usage analytics tracker.
#[derive(Debug)]
pub struct UsageAnalyticsTracker {
    /// All recorded events.
    events: Vec<AccessEvent>,
}

impl Default for UsageAnalyticsTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl UsageAnalyticsTracker {
    /// Create a new empty tracker.
    pub fn new() -> Self {
        Self { events: Vec::new() }
    }

    /// Record an access event.
    pub fn record(&mut self, event: AccessEvent) {
        self.events.push(event);
    }

    /// Return the total number of recorded events.
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Compute usage statistics for a given asset ID.
    pub fn stats_for(&self, asset_id: &str) -> AssetUsageStats {
        let relevant: Vec<&AccessEvent> = self
            .events
            .iter()
            .filter(|e| e.asset_id == asset_id)
            .collect();

        if relevant.is_empty() {
            return AssetUsageStats::default();
        }

        let mut by_kind: HashMap<AccessKind, u64> = HashMap::new();
        let mut users = std::collections::HashSet::new();
        let mut first = u64::MAX;
        let mut last = 0u64;

        for ev in &relevant {
            *by_kind.entry(ev.kind).or_insert(0) += 1;
            users.insert(ev.user_id.clone());
            if ev.timestamp < first {
                first = ev.timestamp;
            }
            if ev.timestamp > last {
                last = ev.timestamp;
            }
        }

        AssetUsageStats {
            total_accesses: relevant.len() as u64,
            by_kind,
            unique_users: users.len() as u64,
            first_access: Some(first),
            last_access: Some(last),
        }
    }

    /// Return the top-N most accessed assets.
    pub fn top_assets(&self, n: usize) -> Vec<PopularityEntry> {
        let mut counts: HashMap<&str, u64> = HashMap::new();
        for ev in &self.events {
            *counts.entry(&ev.asset_id).or_insert(0) += 1;
        }
        let mut entries: Vec<PopularityEntry> = counts
            .into_iter()
            .map(|(id, count)| PopularityEntry {
                asset_id: id.to_string(),
                access_count: count,
            })
            .collect();
        entries.sort_by(|a, b| b.access_count.cmp(&a.access_count));
        entries.truncate(n);
        entries
    }

    /// Return events filtered to a time window `[from, to]` (inclusive).
    pub fn events_in_range(&self, from: u64, to: u64) -> Vec<&AccessEvent> {
        self.events
            .iter()
            .filter(|e| e.timestamp >= from && e.timestamp <= to)
            .collect()
    }

    /// Produce a high-level analytics summary over all recorded events.
    pub fn summary(&self) -> AnalyticsSummary {
        let mut assets = std::collections::HashSet::new();
        let mut users = std::collections::HashSet::new();
        let mut kind_counts: HashMap<AccessKind, u64> = HashMap::new();

        for ev in &self.events {
            assets.insert(ev.asset_id.clone());
            users.insert(ev.user_id.clone());
            *kind_counts.entry(ev.kind).or_insert(0) += 1;
        }

        let top_kind = kind_counts
            .into_iter()
            .max_by_key(|(_, c)| *c)
            .map(|(k, _)| k);

        AnalyticsSummary {
            total_events: self.events.len() as u64,
            distinct_assets: assets.len() as u64,
            distinct_users: users.len() as u64,
            top_kind,
        }
    }

    /// Clear all recorded events.
    pub fn clear(&mut self) {
        self.events.clear();
    }

    /// Count events for a specific access kind.
    pub fn count_by_kind(&self, kind: AccessKind) -> u64 {
        self.events.iter().filter(|e| e.kind == kind).count() as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_event(asset: &str, user: &str, kind: AccessKind, ts: u64) -> AccessEvent {
        AccessEvent::new(asset, user, kind, ts)
    }

    #[test]
    fn test_access_kind_label() {
        assert_eq!(AccessKind::View.label(), "view");
        assert_eq!(AccessKind::Download.label(), "download");
        assert_eq!(AccessKind::Edit.label(), "edit");
        assert_eq!(AccessKind::Share.label(), "share");
        assert_eq!(AccessKind::Stream.label(), "stream");
        assert_eq!(AccessKind::ProjectUse.label(), "project_use");
    }

    #[test]
    fn test_tracker_empty() {
        let tracker = UsageAnalyticsTracker::new();
        assert_eq!(tracker.event_count(), 0);
        let stats = tracker.stats_for("nonexistent");
        assert_eq!(stats.total_accesses, 0);
    }

    #[test]
    fn test_record_and_count() {
        let mut tracker = UsageAnalyticsTracker::new();
        tracker.record(make_event("a1", "u1", AccessKind::View, 100));
        tracker.record(make_event("a1", "u2", AccessKind::Download, 200));
        assert_eq!(tracker.event_count(), 2);
    }

    #[test]
    fn test_stats_for_asset() {
        let mut tracker = UsageAnalyticsTracker::new();
        tracker.record(make_event("a1", "u1", AccessKind::View, 100));
        tracker.record(make_event("a1", "u2", AccessKind::View, 200));
        tracker.record(make_event("a1", "u1", AccessKind::Download, 300));
        tracker.record(make_event("a2", "u1", AccessKind::View, 150));

        let stats = tracker.stats_for("a1");
        assert_eq!(stats.total_accesses, 3);
        assert_eq!(stats.unique_users, 2);
        assert_eq!(stats.first_access, Some(100));
        assert_eq!(stats.last_access, Some(300));
        assert_eq!(
            *stats
                .by_kind
                .get(&AccessKind::View)
                .expect("should succeed in test"),
            2
        );
        assert_eq!(
            *stats
                .by_kind
                .get(&AccessKind::Download)
                .expect("should succeed in test"),
            1
        );
    }

    #[test]
    fn test_top_assets() {
        let mut tracker = UsageAnalyticsTracker::new();
        for i in 0..5 {
            tracker.record(make_event("popular", "u1", AccessKind::View, i));
        }
        for i in 0..2 {
            tracker.record(make_event("less", "u1", AccessKind::View, 10 + i));
        }
        tracker.record(make_event("rare", "u1", AccessKind::View, 20));

        let top = tracker.top_assets(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].asset_id, "popular");
        assert_eq!(top[0].access_count, 5);
        assert_eq!(top[1].asset_id, "less");
        assert_eq!(top[1].access_count, 2);
    }

    #[test]
    fn test_events_in_range() {
        let mut tracker = UsageAnalyticsTracker::new();
        tracker.record(make_event("a1", "u1", AccessKind::View, 10));
        tracker.record(make_event("a1", "u1", AccessKind::View, 50));
        tracker.record(make_event("a1", "u1", AccessKind::View, 100));

        let range = tracker.events_in_range(20, 80);
        assert_eq!(range.len(), 1);
        assert_eq!(range[0].timestamp, 50);
    }

    #[test]
    fn test_summary() {
        let mut tracker = UsageAnalyticsTracker::new();
        tracker.record(make_event("a1", "u1", AccessKind::View, 1));
        tracker.record(make_event("a2", "u2", AccessKind::View, 2));
        tracker.record(make_event("a1", "u1", AccessKind::Download, 3));

        let summary = tracker.summary();
        assert_eq!(summary.total_events, 3);
        assert_eq!(summary.distinct_assets, 2);
        assert_eq!(summary.distinct_users, 2);
        assert_eq!(summary.top_kind, Some(AccessKind::View));
    }

    #[test]
    fn test_clear() {
        let mut tracker = UsageAnalyticsTracker::new();
        tracker.record(make_event("a1", "u1", AccessKind::View, 1));
        assert_eq!(tracker.event_count(), 1);
        tracker.clear();
        assert_eq!(tracker.event_count(), 0);
    }

    #[test]
    fn test_count_by_kind() {
        let mut tracker = UsageAnalyticsTracker::new();
        tracker.record(make_event("a1", "u1", AccessKind::View, 1));
        tracker.record(make_event("a1", "u1", AccessKind::View, 2));
        tracker.record(make_event("a1", "u1", AccessKind::Download, 3));

        assert_eq!(tracker.count_by_kind(AccessKind::View), 2);
        assert_eq!(tracker.count_by_kind(AccessKind::Download), 1);
        assert_eq!(tracker.count_by_kind(AccessKind::Edit), 0);
    }

    #[test]
    fn test_default_trait() {
        let tracker = UsageAnalyticsTracker::default();
        assert_eq!(tracker.event_count(), 0);
    }

    #[test]
    fn test_popularity_entry_equality() {
        let a = PopularityEntry {
            asset_id: "x".to_string(),
            access_count: 10,
        };
        let b = PopularityEntry {
            asset_id: "x".to_string(),
            access_count: 10,
        };
        assert_eq!(a, b);
    }

    #[test]
    fn test_stats_default() {
        let stats = AssetUsageStats::default();
        assert_eq!(stats.total_accesses, 0);
        assert!(stats.first_access.is_none());
        assert!(stats.last_access.is_none());
        assert_eq!(stats.unique_users, 0);
    }
}
