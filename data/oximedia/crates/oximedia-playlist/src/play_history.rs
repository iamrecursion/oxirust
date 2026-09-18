//! Play history and recently-played tracking.
//!
//! Records play events per track and provides queries for play counts,
//! recently played tracks, and most-played tracks.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

const SECONDS_PER_DAY: u64 = 86_400;

/// A single play event recorded in the history.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlayEvent {
    /// The track that was played.
    pub track_id: u64,
    /// Unix epoch timestamp of when playback started (seconds).
    pub timestamp_epoch: u64,
    /// Whether the track was played to completion.
    pub completed: bool,
    /// Cumulative play count for this track at the time of recording.
    pub play_count: u32,
}

impl PlayEvent {
    /// Returns `true` when the track was played to completion.
    pub fn is_full_play(&self) -> bool {
        self.completed
    }

    /// Returns how many days ago this event occurred relative to `now` (epoch seconds).
    pub fn age_days(&self, now: u64) -> f32 {
        let elapsed_secs = now.saturating_sub(self.timestamp_epoch);
        elapsed_secs as f32 / SECONDS_PER_DAY as f32
    }
}

/// In-memory play history with a capped number of entries.
#[derive(Debug, Clone)]
pub struct PlayHistory {
    /// Recorded events, ordered from oldest to newest.
    pub events: Vec<PlayEvent>,
    /// Maximum number of events retained.
    pub max_entries: usize,
}

impl PlayHistory {
    /// Creates an empty history with the given capacity cap.
    pub fn new(max_entries: usize) -> Self {
        Self {
            events: Vec::new(),
            max_entries,
        }
    }

    /// Records a play event for `track_id`.
    ///
    /// When `max_entries` is exceeded the oldest entry is evicted.
    pub fn record_play(&mut self, track_id: u64, epoch: u64, completed: bool) {
        let count = self.play_count(track_id) + 1;
        self.events.push(PlayEvent {
            track_id,
            timestamp_epoch: epoch,
            completed,
            play_count: count,
        });
        if self.max_entries > 0 && self.events.len() > self.max_entries {
            self.events.remove(0);
        }
    }

    /// Records a batch of play events in one vectorized pass.
    ///
    /// Each tuple is `(track_id, epoch, completed)`.  This is **equivalent to
    /// calling [`record_play`](Self::record_play) for each tuple, in order**,
    /// with respect to the appended [`PlayEvent`]s and their `play_count`
    /// fields — but it is more efficient: the running per-track counts are
    /// seeded in a single `O(n)` pass over the existing events, all `N`
    /// events are appended, and any capacity overflow is evicted **once** at
    /// the end (rather than incrementally after every push).
    ///
    /// # Equivalence boundary
    ///
    /// Strict element-by-element equality with `N` sequential `record_play`
    /// calls holds when the history is **unbounded** (`max_entries == 0`) or
    /// when `max_entries >= events.len() + plays.len()` (no eviction occurs).
    /// When eviction *does* occur, the final retained window — its length and
    /// the retained `track_id`s in order — is identical to the sequential
    /// path; only the `play_count` recorded on events that are later evicted
    /// may differ, because the sequential path shrinks the counting base as it
    /// evicts mid-sequence while this method evicts in a single final step.
    pub fn batch_record(&mut self, plays: &[(u64, u64, bool)]) {
        use std::collections::HashMap;
        let mut counts: HashMap<u64, u32> = HashMap::new();
        for e in &self.events {
            *counts.entry(e.track_id).or_insert(0) += 1;
        }
        self.events.reserve(plays.len());
        for &(track_id, epoch, completed) in plays {
            let c = counts.entry(track_id).or_insert(0);
            *c += 1;
            self.events.push(PlayEvent {
                track_id,
                timestamp_epoch: epoch,
                completed,
                play_count: *c,
            });
        }
        if self.max_entries > 0 && self.events.len() > self.max_entries {
            let overflow = self.events.len() - self.max_entries;
            self.events.drain(0..overflow);
        }
    }

    /// Returns the total number of times `track_id` appears in the history.
    pub fn play_count(&self, track_id: u64) -> u32 {
        self.events
            .iter()
            .filter(|e| e.track_id == track_id)
            .count() as u32
    }

    /// Returns up to `limit` track ids from the most recently played events
    /// (newest first, de-duplicated).
    pub fn recently_played(&self, limit: usize) -> Vec<u64> {
        let mut seen = std::collections::HashSet::new();
        let mut result = Vec::new();
        for event in self.events.iter().rev() {
            if seen.insert(event.track_id) {
                result.push(event.track_id);
                if result.len() >= limit {
                    break;
                }
            }
        }
        result
    }

    /// Returns up to `limit` `(track_id, play_count)` pairs sorted by play
    /// count descending.
    pub fn most_played(&self, limit: usize) -> Vec<(u64, u32)> {
        let mut counts: std::collections::HashMap<u64, u32> = std::collections::HashMap::new();
        for event in &self.events {
            *counts.entry(event.track_id).or_insert(0) += 1;
        }
        let mut sorted: Vec<(u64, u32)> = counts.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        sorted.truncate(limit);
        sorted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- PlayEvent ---

    fn make_event(track_id: u64, completed: bool, epoch: u64) -> PlayEvent {
        PlayEvent {
            track_id,
            timestamp_epoch: epoch,
            completed,
            play_count: 1,
        }
    }

    #[test]
    fn test_is_full_play_true() {
        let e = make_event(1, true, 1_000_000);
        assert!(e.is_full_play());
    }

    #[test]
    fn test_is_full_play_false() {
        let e = make_event(1, false, 1_000_000);
        assert!(!e.is_full_play());
    }

    #[test]
    fn test_age_days_one_day() {
        let e = make_event(1, true, 0);
        let now = SECONDS_PER_DAY;
        let age = e.age_days(now);
        assert!((age - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_age_days_zero() {
        let e = make_event(1, true, 1_000);
        let age = e.age_days(1_000);
        assert_eq!(age, 0.0);
    }

    #[test]
    fn test_age_days_future_now_saturates() {
        // now < timestamp should not panic
        let e = make_event(1, true, 5_000);
        let age = e.age_days(1_000);
        assert_eq!(age, 0.0);
    }

    // --- PlayHistory ---

    fn make_history() -> PlayHistory {
        let mut h = PlayHistory::new(100);
        h.record_play(10, 1_000, true);
        h.record_play(20, 2_000, false);
        h.record_play(10, 3_000, true);
        h.record_play(30, 4_000, true);
        h.record_play(20, 5_000, true);
        h.record_play(10, 6_000, false);
        h
    }

    #[test]
    fn test_play_count_track() {
        let h = make_history();
        assert_eq!(h.play_count(10), 3);
    }

    #[test]
    fn test_play_count_other_track() {
        let h = make_history();
        assert_eq!(h.play_count(20), 2);
    }

    #[test]
    fn test_play_count_unknown_track() {
        let h = make_history();
        assert_eq!(h.play_count(99), 0);
    }

    #[test]
    fn test_recently_played_order() {
        let h = make_history();
        // Last played order: 10 (epoch 6000), 20 (5000), 30 (4000)
        let recent = h.recently_played(3);
        assert_eq!(recent[0], 10);
        assert_eq!(recent[1], 20);
        assert_eq!(recent[2], 30);
    }

    #[test]
    fn test_recently_played_deduplication() {
        let h = make_history();
        // track 10 played 3 times but should appear once
        let recent = h.recently_played(10);
        let count_10 = recent.iter().filter(|&&id| id == 10).count();
        assert_eq!(count_10, 1);
    }

    #[test]
    fn test_recently_played_limit() {
        let h = make_history();
        let recent = h.recently_played(1);
        assert_eq!(recent.len(), 1);
    }

    #[test]
    fn test_most_played_order() {
        let h = make_history();
        // track 10 → 3 plays, 20 → 2 plays, 30 → 1 play
        let top = h.most_played(3);
        assert_eq!(top[0].0, 10);
        assert_eq!(top[0].1, 3);
        assert_eq!(top[1].0, 20);
        assert_eq!(top[1].1, 2);
    }

    #[test]
    fn test_most_played_limit() {
        let h = make_history();
        let top = h.most_played(2);
        assert_eq!(top.len(), 2);
    }

    #[test]
    fn test_max_entries_eviction() {
        let mut h = PlayHistory::new(3);
        h.record_play(1, 100, true);
        h.record_play(2, 200, true);
        h.record_play(3, 300, true);
        h.record_play(4, 400, true);
        // oldest (track 1 at epoch 100) should be evicted
        assert_eq!(h.events.len(), 3);
        assert!(h.events.iter().all(|e| e.track_id != 1));
    }

    #[test]
    fn test_record_play_increments_count() {
        let mut h = PlayHistory::new(100);
        h.record_play(42, 1_000, true);
        h.record_play(42, 2_000, true);
        assert_eq!(h.play_count(42), 2);
    }

    // --- batch_record ---

    /// The six plays exercise repeated track ids (10 ×3, 20 ×2, 30 ×1) so the
    /// per-track `play_count` progression is non-trivial.
    const BATCH_PLAYS: [(u64, u64, bool); 6] = [
        (10, 1_000, true),
        (20, 2_000, false),
        (10, 3_000, true),
        (30, 4_000, true),
        (20, 5_000, true),
        (10, 6_000, false),
    ];

    #[test]
    fn test_batch_record_equals_n_single_writes_unbounded() {
        // Unbounded history (max_entries == 0): no eviction can occur, so the
        // batch path must produce byte-for-byte identical events to N
        // sequential record_play calls (including every play_count field).
        let mut sequential = PlayHistory::new(0);
        for &(track_id, epoch, completed) in &BATCH_PLAYS {
            sequential.record_play(track_id, epoch, completed);
        }

        let mut batched = PlayHistory::new(0);
        batched.batch_record(&BATCH_PLAYS);

        // PlayEvent derives PartialEq, so we compare element-wise directly.
        assert_eq!(sequential.events, batched.events);
        // Spot-check that play_count actually progressed (1,1,2,1,2,3).
        let counts: Vec<u32> = batched.events.iter().map(|e| e.play_count).collect();
        assert_eq!(counts, vec![1, 1, 2, 1, 2, 3]);
    }

    #[test]
    fn test_batch_record_eviction_window_matches() {
        // Capped history (cap == 3): record_play evicts incrementally while
        // batch_record evicts once at the end. The EQUIVALENCE BOUNDARY says
        // the final retained WINDOW (length + retained track_ids/order) is
        // identical, even though mid-sequence play_count assignment may differ.
        let plays: [(u64, u64, bool); 5] = [
            (1, 100, true),
            (2, 200, true),
            (3, 300, true),
            (1, 400, true),
            (2, 500, true),
        ];

        let mut sequential = PlayHistory::new(3);
        for &(track_id, epoch, completed) in &plays {
            sequential.record_play(track_id, epoch, completed);
        }

        let mut batched = PlayHistory::new(3);
        batched.batch_record(&plays);

        // Both retain exactly the last `cap` events.
        assert_eq!(sequential.events.len(), 3);
        assert_eq!(batched.events.len(), 3);

        // The retained window — track_ids in order and their timestamps — is
        // identical (the last 3 plays: 3@300, 1@400, 2@500).
        let seq_window: Vec<(u64, u64)> = sequential
            .events
            .iter()
            .map(|e| (e.track_id, e.timestamp_epoch))
            .collect();
        let batch_window: Vec<(u64, u64)> = batched
            .events
            .iter()
            .map(|e| (e.track_id, e.timestamp_epoch))
            .collect();
        assert_eq!(seq_window, batch_window);
        assert_eq!(batch_window, vec![(3, 300), (1, 400), (2, 500)]);
    }

    #[test]
    fn test_batch_record_empty_is_noop() {
        // An empty batch on a non-empty history leaves it untouched, and on a
        // fresh history produces no events.
        let mut seeded = make_history();
        let before = seeded.events.clone();
        seeded.batch_record(&[]);
        assert_eq!(seeded.events, before);

        let mut fresh = PlayHistory::new(10);
        fresh.batch_record(&[]);
        assert!(fresh.events.is_empty());
    }

    #[test]
    fn test_batch_record_appends_to_existing_events() {
        // batch_record must seed running counts from the *existing* events,
        // not start from zero. Pre-seed track 10 once, then batch two more
        // plays of track 10 → counts must continue 2, 3 (not 1, 2).
        let mut h = PlayHistory::new(0);
        h.record_play(10, 500, true);
        h.batch_record(&[(10, 600, true), (10, 700, false)]);

        let counts: Vec<u32> = h.events.iter().map(|e| e.play_count).collect();
        assert_eq!(counts, vec![1, 2, 3]);
        assert_eq!(h.play_count(10), 3);
    }
}
