//! Play history tracking: recently played, skip tracking, and completion rates.
//!
//! This module provides a richer history layer on top of raw play events,
//! including per-track statistics, skip detection, and completion rate queries.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;

// ── play record ───────────────────────────────────────────────────────────────

/// How a playback session ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlayOutcome {
    /// Track was played to completion.
    Completed,
    /// Track was skipped before completion.
    Skipped,
    /// Playback was paused mid-way and never resumed.
    Abandoned,
}

/// A single entry in the play history log.
#[derive(Debug, Clone)]
pub struct HistoryRecord {
    /// Track identifier.
    pub track_id: u64,
    /// Unix timestamp (seconds) when playback started.
    pub started_at: u64,
    /// Duration played in seconds.
    pub played_seconds: f64,
    /// Total track duration in seconds.
    pub total_seconds: f64,
    /// How playback ended.
    pub outcome: PlayOutcome,
}

impl HistoryRecord {
    /// Create a new history record.
    #[must_use]
    pub fn new(
        track_id: u64,
        started_at: u64,
        played_seconds: f64,
        total_seconds: f64,
        outcome: PlayOutcome,
    ) -> Self {
        Self {
            track_id,
            started_at,
            played_seconds,
            total_seconds,
            outcome,
        }
    }

    /// Fraction of the track that was played (0.0 – 1.0).
    #[must_use]
    pub fn completion_fraction(&self) -> f64 {
        if self.total_seconds < f64::EPSILON {
            return 0.0;
        }
        (self.played_seconds / self.total_seconds).clamp(0.0, 1.0)
    }

    /// Return `true` if the track was completed.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.outcome == PlayOutcome::Completed
    }

    /// Return `true` if the track was skipped.
    #[must_use]
    pub fn is_skip(&self) -> bool {
        self.outcome == PlayOutcome::Skipped
    }
}

// ── per-track stats ───────────────────────────────────────────────────────────

/// Aggregated statistics for a single track.
#[derive(Debug, Clone, Default)]
pub struct TrackStats {
    /// Total number of plays (including skips/abandoned).
    pub play_count: u32,
    /// Number of completed plays.
    pub complete_count: u32,
    /// Number of skips.
    pub skip_count: u32,
    /// Number of abandoned plays.
    pub abandon_count: u32,
    /// Cumulative seconds played across all sessions.
    pub total_played_seconds: f64,
    /// Unix timestamp of the most recent play start.
    pub last_played_at: u64,
}

impl TrackStats {
    /// Completion rate (completed / total plays). Returns 0.0 if never played.
    #[must_use]
    pub fn completion_rate(&self) -> f64 {
        if self.play_count == 0 {
            return 0.0;
        }
        self.complete_count as f64 / self.play_count as f64
    }

    /// Skip rate (skips / total plays). Returns 0.0 if never played.
    #[must_use]
    pub fn skip_rate(&self) -> f64 {
        if self.play_count == 0 {
            return 0.0;
        }
        self.skip_count as f64 / self.play_count as f64
    }

    /// Mean seconds played per session.
    #[must_use]
    pub fn mean_played_seconds(&self) -> f64 {
        if self.play_count == 0 {
            return 0.0;
        }
        self.total_played_seconds / self.play_count as f64
    }
}

// ── history store ─────────────────────────────────────────────────────────────

/// Central store for all play history.
#[derive(Debug, Default)]
pub struct HistoryStore {
    /// All records in insertion order.
    records: Vec<HistoryRecord>,
    /// Per-track aggregated stats (kept up to date on each insert).
    stats: HashMap<u64, TrackStats>,
    /// Maximum number of records to keep (0 = unlimited).
    capacity: usize,
}

impl HistoryStore {
    /// Create a new, unlimited history store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a history store with a maximum capacity (oldest evicted first).
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            capacity,
            ..Self::default()
        }
    }

    /// Record a play event.
    pub fn record(&mut self, rec: HistoryRecord) {
        // Update stats.
        let stats = self.stats.entry(rec.track_id).or_default();
        stats.play_count += 1;
        stats.total_played_seconds += rec.played_seconds;
        if rec.started_at > stats.last_played_at {
            stats.last_played_at = rec.started_at;
        }
        match rec.outcome {
            PlayOutcome::Completed => stats.complete_count += 1,
            PlayOutcome::Skipped => stats.skip_count += 1,
            PlayOutcome::Abandoned => stats.abandon_count += 1,
        }
        self.records.push(rec);
        // Enforce capacity.
        if self.capacity > 0 && self.records.len() > self.capacity {
            self.records.remove(0);
        }
    }

    /// Return stats for a specific track, or `None` if never played.
    #[must_use]
    pub fn track_stats(&self, track_id: u64) -> Option<&TrackStats> {
        self.stats.get(&track_id)
    }

    /// Return the `n` most recently started tracks (deduplicated by track ID).
    #[must_use]
    pub fn recently_played(&self, n: usize) -> Vec<u64> {
        let mut seen = std::collections::HashSet::new();
        let mut result = Vec::new();
        for rec in self.records.iter().rev() {
            if seen.insert(rec.track_id) {
                result.push(rec.track_id);
                if result.len() >= n {
                    break;
                }
            }
        }
        result
    }

    /// Return the top `n` tracks by total play count.
    #[must_use]
    pub fn most_played(&self, n: usize) -> Vec<(u64, u32)> {
        let mut counts: Vec<(u64, u32)> = self
            .stats
            .iter()
            .map(|(&id, s)| (id, s.play_count))
            .collect();
        counts.sort_by(|a, b| b.1.cmp(&a.1));
        counts.truncate(n);
        counts
    }

    /// Return the top `n` most-skipped tracks (by skip count).
    #[must_use]
    pub fn most_skipped(&self, n: usize) -> Vec<(u64, u32)> {
        let mut skips: Vec<(u64, u32)> = self
            .stats
            .iter()
            .map(|(&id, s)| (id, s.skip_count))
            .collect();
        skips.sort_by(|a, b| b.1.cmp(&a.1));
        skips.truncate(n);
        skips
    }

    /// Total number of records stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Return `true` if no records have been stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Overall completion rate across all plays.
    #[must_use]
    pub fn global_completion_rate(&self) -> f64 {
        if self.records.is_empty() {
            return 0.0;
        }
        let completed = self.records.iter().filter(|r| r.is_complete()).count();
        completed as f64 / self.records.len() as f64
    }

    /// Return all records for tracks played within the given epoch window.
    #[must_use]
    pub fn records_in_window(&self, from_epoch: u64, to_epoch: u64) -> Vec<&HistoryRecord> {
        self.records
            .iter()
            .filter(|r| r.started_at >= from_epoch && r.started_at <= to_epoch)
            .collect()
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_record(
        track_id: u64,
        started_at: u64,
        played: f64,
        total: f64,
        outcome: PlayOutcome,
    ) -> HistoryRecord {
        HistoryRecord::new(track_id, started_at, played, total, outcome)
    }

    #[test]
    fn test_record_completion_fraction() {
        let r = make_record(1, 0, 120.0, 200.0, PlayOutcome::Completed);
        assert!((r.completion_fraction() - 0.6).abs() < 1e-9);
    }

    #[test]
    fn test_record_zero_duration() {
        let r = make_record(1, 0, 0.0, 0.0, PlayOutcome::Completed);
        assert_eq!(r.completion_fraction(), 0.0);
    }

    #[test]
    fn test_record_is_complete() {
        assert!(make_record(1, 0, 100.0, 100.0, PlayOutcome::Completed).is_complete());
        assert!(!make_record(1, 0, 50.0, 100.0, PlayOutcome::Skipped).is_complete());
    }

    #[test]
    fn test_record_is_skip() {
        assert!(make_record(1, 0, 30.0, 100.0, PlayOutcome::Skipped).is_skip());
        assert!(!make_record(1, 0, 100.0, 100.0, PlayOutcome::Completed).is_skip());
    }

    #[test]
    fn test_store_record_and_stats() {
        let mut store = HistoryStore::new();
        store.record(make_record(10, 100, 180.0, 200.0, PlayOutcome::Completed));
        store.record(make_record(10, 200, 30.0, 200.0, PlayOutcome::Skipped));
        let stats = store.track_stats(10).expect("should succeed in test");
        assert_eq!(stats.play_count, 2);
        assert_eq!(stats.complete_count, 1);
        assert_eq!(stats.skip_count, 1);
        assert_eq!(stats.last_played_at, 200);
    }

    #[test]
    fn test_store_unknown_track() {
        let store = HistoryStore::new();
        assert!(store.track_stats(999).is_none());
    }

    #[test]
    fn test_store_recently_played() {
        let mut store = HistoryStore::new();
        for ts in [1u64, 2, 3] {
            store.record(make_record(
                ts,
                ts * 100,
                60.0,
                60.0,
                PlayOutcome::Completed,
            ));
        }
        let recent = store.recently_played(2);
        assert_eq!(recent.len(), 2);
        assert_eq!(recent[0], 3); // most recent first
    }

    #[test]
    fn test_store_recently_played_dedup() {
        let mut store = HistoryStore::new();
        store.record(make_record(1, 100, 60.0, 60.0, PlayOutcome::Completed));
        store.record(make_record(1, 200, 60.0, 60.0, PlayOutcome::Completed));
        store.record(make_record(2, 300, 60.0, 60.0, PlayOutcome::Completed));
        let recent = store.recently_played(10);
        assert_eq!(recent.len(), 2); // 1 and 2, not 1 twice
    }

    #[test]
    fn test_store_most_played() {
        let mut store = HistoryStore::new();
        for _ in 0..3 {
            store.record(make_record(1, 0, 60.0, 60.0, PlayOutcome::Completed));
        }
        store.record(make_record(2, 0, 60.0, 60.0, PlayOutcome::Completed));
        let top = store.most_played(1);
        assert_eq!(top[0].0, 1);
        assert_eq!(top[0].1, 3);
    }

    #[test]
    fn test_store_most_skipped() {
        let mut store = HistoryStore::new();
        for _ in 0..4 {
            store.record(make_record(5, 0, 10.0, 60.0, PlayOutcome::Skipped));
        }
        store.record(make_record(6, 0, 10.0, 60.0, PlayOutcome::Skipped));
        let top = store.most_skipped(1);
        assert_eq!(top[0].0, 5);
        assert_eq!(top[0].1, 4);
    }

    #[test]
    fn test_store_capacity_eviction() {
        let mut store = HistoryStore::with_capacity(3);
        for i in 0u64..5 {
            store.record(make_record(i, i * 100, 60.0, 60.0, PlayOutcome::Completed));
        }
        assert_eq!(store.len(), 3);
    }

    #[test]
    fn test_store_global_completion_rate() {
        let mut store = HistoryStore::new();
        store.record(make_record(1, 0, 60.0, 60.0, PlayOutcome::Completed));
        store.record(make_record(2, 0, 10.0, 60.0, PlayOutcome::Skipped));
        assert!((store.global_completion_rate() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_store_empty_completion_rate() {
        let store = HistoryStore::new();
        assert_eq!(store.global_completion_rate(), 0.0);
    }

    #[test]
    fn test_store_records_in_window() {
        let mut store = HistoryStore::new();
        store.record(make_record(1, 1000, 60.0, 60.0, PlayOutcome::Completed));
        store.record(make_record(2, 2000, 60.0, 60.0, PlayOutcome::Completed));
        store.record(make_record(3, 3000, 60.0, 60.0, PlayOutcome::Completed));
        let win = store.records_in_window(1500, 2500);
        assert_eq!(win.len(), 1);
        assert_eq!(win[0].track_id, 2);
    }

    #[test]
    fn test_track_stats_rates() {
        let mut store = HistoryStore::new();
        for _ in 0..6 {
            store.record(make_record(7, 0, 60.0, 60.0, PlayOutcome::Completed));
        }
        for _ in 0..4 {
            store.record(make_record(7, 0, 10.0, 60.0, PlayOutcome::Skipped));
        }
        let s = store.track_stats(7).expect("should succeed in test");
        assert!((s.completion_rate() - 0.6).abs() < 1e-9);
        assert!((s.skip_rate() - 0.4).abs() < 1e-9);
    }
}
