//! LRU-style angle score cache for the automatic switcher.
//!
//! Avoids redundant face/motion detection score computation for camera angles
//! whose frame content has not changed since the last evaluation.

use std::collections::HashMap;

/// Cache mapping `(angle_index, frame_number)` → computed score.
///
/// Capacity is bounded by `max_entries`; when the cache is full, `evict_oldest`
/// removes the oldest 10 % of entries (insertion order via sequential serial).
#[derive(Debug)]
pub struct AngleScoreCache {
    /// Stored scores keyed by (angle_index, frame_number).
    cache: HashMap<(usize, u64), (f32, u64)>,
    /// Maximum number of entries before eviction is needed.
    max_entries: usize,
    /// Monotonically increasing counter used to track insertion order.
    serial: u64,
    /// Total cache-hit count (for hit-rate reporting).
    hits: u64,
    /// Total cache-miss count (for hit-rate reporting).
    misses: u64,
}

impl AngleScoreCache {
    /// Create a new cache with the given capacity.
    #[must_use]
    pub fn new(max_entries: usize) -> Self {
        Self {
            cache: HashMap::new(),
            max_entries: max_entries.max(1),
            serial: 0,
            hits: 0,
            misses: 0,
        }
    }

    /// Look up a previously computed score.
    ///
    /// Returns `Some(score)` on a cache hit, `None` on a miss.
    /// Updates internal hit/miss counters.
    pub fn get(&mut self, angle_idx: usize, frame: u64) -> Option<f32> {
        if let Some(&(score, _serial)) = self.cache.get(&(angle_idx, frame)) {
            self.hits += 1;
            Some(score)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Store a computed score.  Evicts the oldest 10 % of entries when the
    /// cache is at capacity.
    pub fn insert(&mut self, angle_idx: usize, frame: u64, score: f32) {
        if self.cache.len() >= self.max_entries && !self.cache.contains_key(&(angle_idx, frame)) {
            self.evict_oldest();
        }
        self.serial += 1;
        self.cache.insert((angle_idx, frame), (score, self.serial));
    }

    /// Remove all cached scores for the given angle index.
    pub fn invalidate_angle(&mut self, angle_idx: usize) {
        self.cache.retain(|&(a, _), _| a != angle_idx);
    }

    /// Cache hit rate in the range [0.0, 1.0].
    ///
    /// Returns 0.0 if no lookups have been made yet.
    #[must_use]
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Number of entries currently stored in the cache.
    #[must_use]
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Returns `true` if the cache holds no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Evict the oldest 10 % of entries (by insertion serial number).
    ///
    /// Removes at least one entry so that `insert` can always make progress.
    pub fn evict_oldest(&mut self) {
        if self.cache.is_empty() {
            return;
        }

        // Determine how many entries to remove (at least 1, at most 10 %).
        let evict_count = ((self.cache.len() / 10) + 1).min(self.cache.len());

        // Collect serials in ascending order and pick the `evict_count` smallest.
        let mut serials: Vec<((usize, u64), u64)> = self
            .cache
            .iter()
            .map(|(&key, &(_score, serial))| (key, serial))
            .collect();

        serials.sort_unstable_by_key(|&(_, s)| s);

        for (key, _) in serials.into_iter().take(evict_count) {
            self.cache.remove(&key);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_angle_score_cache_hit() {
        let mut cache = AngleScoreCache::new(64);
        cache.insert(0, 100, 0.75);
        // First call: cache miss (not yet inserted at the time of first get).
        // Re-verify after insert:
        let result = cache.get(0, 100);
        assert_eq!(result, Some(0.75));
        // Second call with same key → hit.
        let result2 = cache.get(0, 100);
        assert_eq!(result2, Some(0.75));
    }

    #[test]
    fn test_angle_score_cache_miss() {
        let mut cache = AngleScoreCache::new(64);
        cache.insert(0, 100, 0.5);
        // Different frame number → miss.
        assert_eq!(cache.get(0, 101), None);
    }

    #[test]
    fn test_angle_score_cache_invalidate() {
        let mut cache = AngleScoreCache::new(64);
        cache.insert(0, 100, 0.9);
        cache.insert(1, 100, 0.8);
        cache.invalidate_angle(0);
        // Angle 0 entry should be gone.
        assert_eq!(cache.get(0, 100), None);
        // Angle 1 entry should still be present.
        assert_eq!(cache.get(1, 100), Some(0.8));
    }

    #[test]
    fn test_angle_score_cache_evict() {
        // Small capacity to trigger eviction easily.
        let mut cache = AngleScoreCache::new(10);
        for i in 0..10_u64 {
            cache.insert(0, i, i as f32 * 0.1);
        }
        let before = cache.len();
        assert_eq!(before, 10);
        // Inserting an eleventh entry must trigger eviction.
        cache.insert(0, 99, 0.99);
        assert!(
            cache.len() < before,
            "evict_oldest should have reduced cache size; len={}",
            cache.len()
        );
    }

    #[test]
    fn test_hit_rate_tracking() {
        let mut cache = AngleScoreCache::new(64);
        cache.insert(0, 1, 0.5);
        cache.get(0, 1); // hit
        cache.get(0, 2); // miss
        let rate = cache.hit_rate();
        assert!(
            (rate - 0.5).abs() < 1e-9,
            "hit_rate should be 0.5, got {rate}"
        );
    }
}
