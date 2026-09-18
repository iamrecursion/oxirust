//! Advanced eviction policy implementations.
//!
//! This module provides standalone data structures that implement various cache
//! eviction strategies, decoupled from any specific cache backend:
//!
//! - [`EvictionPolicy`] — discriminated union of all supported policies.
//! - [`FrequencyCounter`] — windowed frequency estimator with decay.
//! - [`LfuEvictionTracker`] — O(1) amortised LFU tracking via frequency buckets.
//! - [`TinyLfuAdmission`] — admission gate used by the TinyLFU policy.
//! - [`ArcTracker`] — Adaptive Replacement Cache ghost-list tracking.

use std::collections::{BTreeMap, HashMap, VecDeque};

use crate::bloom_filter::{BloomFilter, CountingBloomFilter};

// ── EvictionPolicy ────────────────────────────────────────────────────────────

/// Discriminated union of all eviction strategies understood by this crate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvictionPolicy {
    /// Least Recently Used — evict the entry that was accessed longest ago.
    Lru,
    /// Least Frequently Used — evict the entry with the fewest accesses.
    Lfu,
    /// First In, First Out — evict the oldest inserted entry.
    Fifo,
    /// Random — evict a uniformly random entry.
    Random,
    /// TinyLFU — frequency-based admission combined with approximate LFU
    /// tracking via a small Count-Min sketch.
    TinyLfu,
    /// Adaptive Replacement Cache — self-tuning balance between recency
    /// (T1/B1 ghost) and frequency (T2/B2 ghost) lists.
    ArcCache,
}

// ── FrequencyCounter ──────────────────────────────────────────────────────────

/// Windowed frequency counter with periodic exponential decay.
///
/// Maintains per-key hit counts and a sliding window so that old popularity
/// does not permanently inflate a key's frequency estimate (frequency
/// inflation problem).  Calling [`decay_all`] halves every counter, which
/// approximates the behaviour of a sliding window.
///
/// [`decay_all`]: FrequencyCounter::decay_all
#[derive(Debug, Clone)]
pub struct FrequencyCounter {
    /// Map from key hash to hit count.
    counters: HashMap<u64, u64>,
    /// Target window size; when `total_increments` exceeds this, the caller
    /// should invoke `decay_all`.
    window_size: usize,
    /// Running total of increment calls since the last decay.
    total_increments: usize,
}

impl FrequencyCounter {
    /// Create a new `FrequencyCounter` with the given window size.
    pub fn new(window_size: usize) -> Self {
        Self {
            counters: HashMap::new(),
            window_size: window_size.max(1),
            total_increments: 0,
        }
    }

    /// Increment the count for `key`.
    ///
    /// Automatically triggers [`decay_all`] when `total_increments` reaches
    /// `window_size`.
    ///
    /// [`decay_all`]: FrequencyCounter::decay_all
    pub fn increment(&mut self, key: u64) {
        *self.counters.entry(key).or_insert(0) += 1;
        self.total_increments += 1;
        if self.total_increments >= self.window_size {
            self.decay_all();
        }
    }

    /// Return the current frequency estimate for `key`.
    pub fn frequency(&self, key: u64) -> u64 {
        self.counters.get(&key).copied().unwrap_or(0)
    }

    /// Halve all counters (right-shift by 1) to prevent frequency inflation.
    ///
    /// Counters that fall to zero are removed to keep memory bounded.
    pub fn decay_all(&mut self) {
        self.counters.retain(|_, count| {
            *count >>= 1;
            *count > 0
        });
        self.total_increments = 0;
    }

    /// Return the number of distinct keys currently tracked.
    pub fn tracked_keys(&self) -> usize {
        self.counters.len()
    }

    /// Reset all state.
    pub fn clear(&mut self) {
        self.counters.clear();
        self.total_increments = 0;
    }
}

// ── LfuEvictionTracker ────────────────────────────────────────────────────────

/// O(1) amortised LFU eviction tracker.
///
/// Maintains a set of frequency buckets, each holding a FIFO queue of keys at
/// that frequency.  Promotion moves a key from `freq[f]` to `freq[f+1]`; the
/// victim is always the oldest key in the lowest-frequency bucket.
#[derive(Debug, Clone)]
pub struct LfuEvictionTracker {
    /// `frequency → FIFO queue of keys at that frequency`.
    freq_buckets: BTreeMap<u64, VecDeque<u64>>,
    /// `key → current frequency`.
    key_freq: HashMap<u64, u64>,
}

impl LfuEvictionTracker {
    /// Create an empty tracker.
    pub fn new() -> Self {
        Self {
            freq_buckets: BTreeMap::new(),
            key_freq: HashMap::new(),
        }
    }

    /// Insert a brand-new `key` at frequency 1.
    ///
    /// If `key` already exists its frequency is not reset; call [`promote`]
    /// instead.
    ///
    /// [`promote`]: LfuEvictionTracker::promote
    pub fn insert(&mut self, key: u64) {
        if self.key_freq.contains_key(&key) {
            return;
        }
        self.key_freq.insert(key, 1);
        self.freq_buckets.entry(1).or_default().push_back(key);
    }

    /// Record an access to `key`, moving it from `freq[f]` to `freq[f+1]`.
    ///
    /// No-op if `key` is not tracked.
    pub fn promote(&mut self, key: u64) {
        let old_freq = match self.key_freq.get_mut(&key) {
            Some(f) => *f,
            None => return,
        };
        let new_freq = old_freq.saturating_add(1);

        // Remove from old bucket.
        if let Some(bucket) = self.freq_buckets.get_mut(&old_freq) {
            bucket.retain(|&k| k != key);
            if bucket.is_empty() {
                self.freq_buckets.remove(&old_freq);
            }
        }

        // Insert into new bucket.
        self.key_freq.insert(key, new_freq);
        self.freq_buckets
            .entry(new_freq)
            .or_default()
            .push_back(key);
    }

    /// Remove and return the key with the lowest frequency (tie-broken by
    /// oldest insertion into that frequency bucket — i.e. FIFO within the
    /// bucket).
    ///
    /// Returns `None` if no keys are tracked.
    pub fn evict(&mut self) -> Option<u64> {
        // BTreeMap is ordered: first entry is the minimum frequency.
        let (&min_freq, _) = self.freq_buckets.iter().next()?;
        let victim = self
            .freq_buckets
            .get_mut(&min_freq)
            .and_then(|q| q.pop_front())?;

        // Clean up empty bucket.
        if let Some(bucket) = self.freq_buckets.get(&min_freq) {
            if bucket.is_empty() {
                self.freq_buckets.remove(&min_freq);
            }
        }

        self.key_freq.remove(&victim);
        Some(victim)
    }

    /// Remove a specific `key` from the tracker (e.g. on explicit deletion).
    ///
    /// Returns `true` if the key was present.
    pub fn remove(&mut self, key: u64) -> bool {
        if let Some(freq) = self.key_freq.remove(&key) {
            if let Some(bucket) = self.freq_buckets.get_mut(&freq) {
                bucket.retain(|&k| k != key);
                if bucket.is_empty() {
                    self.freq_buckets.remove(&freq);
                }
            }
            true
        } else {
            false
        }
    }

    /// Return the current frequency of `key`, or `0` if not tracked.
    pub fn frequency(&self, key: u64) -> u64 {
        self.key_freq.get(&key).copied().unwrap_or(0)
    }

    /// Return the number of tracked keys.
    pub fn len(&self) -> usize {
        self.key_freq.len()
    }

    /// Return `true` when no keys are tracked.
    pub fn is_empty(&self) -> bool {
        self.key_freq.is_empty()
    }
}

impl Default for LfuEvictionTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ── TinyLfuAdmission ──────────────────────────────────────────────────────────

/// TinyLFU admission gate.
///
/// Combines a standard [`BloomFilter`] (to track whether a candidate has been
/// seen before) with a [`CountingBloomFilter`] (to estimate its recent
/// frequency).  A new item is admitted into the main cache only when its
/// frequency estimate exceeds that of the item being evicted.
pub struct TinyLfuAdmission {
    /// Doorkeeper: a standard Bloom filter that tracks *whether* an item has
    /// previously been seen.  Items only get a counter in `window_counter`
    /// after passing through the doorkeeper once — approximating a two-phase
    /// admission.
    bloom: BloomFilter,
    /// Approximate frequency counter using 4-bit counting Bloom filter.
    window_counter: CountingBloomFilter,
    /// FNV-1a–based frequency map for the last `window_size` operations.
    freq_counter: FrequencyCounter,
}

impl TinyLfuAdmission {
    /// Create a new admission gate sized for `expected_items` keys.
    pub fn new(expected_items: usize) -> Self {
        let window_size = expected_items * 4; // standard W = 4 × capacity
        Self {
            bloom: BloomFilter::new(expected_items.max(1), 0.01),
            window_counter: CountingBloomFilter::new(expected_items.max(1), 0.01),
            freq_counter: FrequencyCounter::new(window_size.max(1)),
        }
    }

    /// Record an access to `candidate_key`.
    ///
    /// On the first encounter (not in doorkeeper) the key is added to the
    /// bloom filter.  On subsequent encounters a counter is incremented.
    pub fn record_access(&mut self, candidate_key: u64) {
        let key_bytes = candidate_key.to_le_bytes();
        if !self.bloom.contains(&key_bytes) {
            self.bloom.insert(&key_bytes);
        } else {
            self.window_counter.insert(&key_bytes);
        }
        self.freq_counter.increment(candidate_key);
    }

    /// Decide whether `candidate_key` should be admitted in place of an entry
    /// whose eviction frequency estimate is `evicted_freq`.
    ///
    /// Returns `true` (admit) when the candidate's estimated frequency exceeds
    /// the evicted entry's frequency.
    pub fn should_admit(&mut self, candidate_key: u64, evicted_freq: u64) -> bool {
        self.record_access(candidate_key);
        let candidate_freq = self.freq_counter.frequency(candidate_key);
        candidate_freq > evicted_freq
    }

    /// Return the estimated frequency for `key`.
    pub fn estimated_frequency(&self, key: u64) -> u64 {
        self.freq_counter.frequency(key)
    }

    /// Decay all counters (delegate to the underlying [`FrequencyCounter`]).
    pub fn decay(&mut self) {
        self.freq_counter.decay_all();
    }
}

// ── ArcTracker ────────────────────────────────────────────────────────────────

/// Adaptive Replacement Cache ghost-list tracker.
///
/// ARC maintains four lists:
///
/// | List | Meaning |
/// |------|---------|
/// | T1   | Recency list — recently accessed **once** |
/// | T2   | Frequency list — accessed **at least twice** |
/// | B1   | Ghost entries for T1 evictions |
/// | B2   | Ghost entries for T2 evictions |
///
/// The tuning parameter `p` represents the target size of T1.  It adapts
/// based on ghost-list hits:
///
/// - Hit in B1 → favour recency; increase `p`.
/// - Hit in B2 → favour frequency; decrease `p`.
///
/// This struct tracks only the *sizes* of the four lists and `p`; the actual
/// key storage is left to the caller.
#[derive(Debug, Clone)]
pub struct ArcTracker {
    /// Current size of T1 (recency list).
    pub t1_size: usize,
    /// Current size of T2 (frequency list).
    pub t2_size: usize,
    /// Current size of B1 (T1 ghost list).
    pub b1_size: usize,
    /// Current size of B2 (T2 ghost list).
    pub b2_size: usize,
    /// ARC parameter: target size for T1.  Adaptively updated.
    pub p: usize,
    /// Maximum total capacity (T1 + T2 ≤ capacity).
    capacity: usize,
}

impl ArcTracker {
    /// Create a new `ArcTracker` with `capacity` total slots and `p` starting
    /// at 0 (fully favour frequency until evidence of recency value arrives).
    pub fn new(capacity: usize) -> Self {
        Self {
            t1_size: 0,
            t2_size: 0,
            b1_size: 0,
            b2_size: 0,
            p: 0,
            capacity,
        }
    }

    // ── Adaptation ────────────────────────────────────────────────────────────

    /// Increase `p` when there is a hit in B1 (recency ghost): the filter
    /// previously evicted something recency-accessed that turned out to be
    /// useful.
    ///
    /// Increase amount: `max(b2_size / b1_size, 1)`, capped so `p ≤ capacity`.
    pub fn adapt_on_hit_b1(&mut self) {
        let delta = self.b2_size.checked_div(self.b1_size).unwrap_or(0).max(1);
        self.p = self.p.saturating_add(delta).min(self.capacity);
    }

    /// Decrease `p` when there is a hit in B2 (frequency ghost): the filter
    /// previously evicted something frequently accessed that turned out to be
    /// useful.
    ///
    /// Decrease amount: `max(b1_size / b2_size, 1)`, floored at 0.
    pub fn adapt_on_hit_b2(&mut self) {
        let delta = self.b1_size.checked_div(self.b2_size).unwrap_or(0).max(1);
        self.p = self.p.saturating_sub(delta);
    }

    // ── Size bookkeeping helpers ───────────────────────────────────────────────

    /// Called when an item is admitted to T1.
    pub fn on_admit_t1(&mut self) {
        self.t1_size += 1;
    }

    /// Called when an item is promoted from T1 to T2 (second access).
    pub fn on_promote_t1_to_t2(&mut self) {
        if self.t1_size > 0 {
            self.t1_size -= 1;
        }
        self.t2_size += 1;
    }

    /// Called when a T1 entry is evicted and becomes a B1 ghost.
    pub fn on_evict_t1(&mut self) {
        if self.t1_size > 0 {
            self.t1_size -= 1;
        }
        self.b1_size += 1;
    }

    /// Called when a T2 entry is evicted and becomes a B2 ghost.
    pub fn on_evict_t2(&mut self) {
        if self.t2_size > 0 {
            self.t2_size -= 1;
        }
        self.b2_size += 1;
    }

    /// Called when a B1 ghost entry is reclaimed (evicted from ghost list).
    pub fn on_remove_b1_ghost(&mut self) {
        if self.b1_size > 0 {
            self.b1_size -= 1;
        }
    }

    /// Called when a B2 ghost entry is reclaimed (evicted from ghost list).
    pub fn on_remove_b2_ghost(&mut self) {
        if self.b2_size > 0 {
            self.b2_size -= 1;
        }
    }

    // ── Queries ───────────────────────────────────────────────────────────────

    /// Total live entries (T1 + T2).
    pub fn live_size(&self) -> usize {
        self.t1_size + self.t2_size
    }

    /// Total ghost entries (B1 + B2).
    pub fn ghost_size(&self) -> usize {
        self.b1_size + self.b2_size
    }

    /// Return `true` when the live cache has reached `capacity`.
    pub fn is_full(&self) -> bool {
        self.live_size() >= self.capacity
    }

    /// Decide which live list to evict from (T1 or T2) under the ARC policy.
    ///
    /// Returns `true` to evict from T1, `false` to evict from T2.
    ///
    /// Prefers T1 when `t1_size > p` (T1 has exceeded its target) or when T2
    /// is empty.
    pub fn should_evict_t1(&self) -> bool {
        (self.t1_size > 0) && (self.t1_size > self.p || self.t2_size == 0)
    }

    /// Maximum capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── FrequencyCounter ──────────────────────────────────────────────────────

    #[test]
    fn test_frequency_counter_increment() {
        let mut fc = FrequencyCounter::new(100);
        fc.increment(42);
        fc.increment(42);
        fc.increment(42);
        assert_eq!(fc.frequency(42), 3);
    }

    #[test]
    fn test_frequency_counter_absent_key() {
        let fc = FrequencyCounter::new(50);
        assert_eq!(fc.frequency(999), 0);
    }

    #[test]
    fn test_frequency_counter_decay() {
        let mut fc = FrequencyCounter::new(100);
        fc.increment(1);
        fc.increment(1);
        fc.increment(1);
        fc.increment(1); // freq[1] = 4
        fc.decay_all();
        assert_eq!(fc.frequency(1), 2); // 4 >> 1 = 2
    }

    #[test]
    fn test_frequency_counter_decay_removes_zero() {
        let mut fc = FrequencyCounter::new(100);
        fc.increment(77); // freq = 1
        fc.decay_all(); // 1 >> 1 = 0 → removed
        assert_eq!(fc.frequency(77), 0);
        assert_eq!(fc.tracked_keys(), 0);
    }

    #[test]
    fn test_frequency_counter_auto_decay_on_window_fill() {
        let window = 8usize;
        let mut fc = FrequencyCounter::new(window);
        // Fill the window exactly.
        for _ in 0..window {
            fc.increment(1);
        }
        // Auto-decay should have halved the counter.
        // freq was 8 before decay; after one decay: 4.
        assert_eq!(fc.frequency(1), 4);
    }

    #[test]
    fn test_frequency_counter_clear() {
        let mut fc = FrequencyCounter::new(50);
        fc.increment(10);
        fc.increment(20);
        fc.clear();
        assert_eq!(fc.tracked_keys(), 0);
        assert_eq!(fc.frequency(10), 0);
    }

    // ── LfuEvictionTracker ────────────────────────────────────────────────────

    #[test]
    fn test_lfu_insert_and_frequency() {
        let mut tracker = LfuEvictionTracker::new();
        tracker.insert(100);
        assert_eq!(tracker.frequency(100), 1);
    }

    #[test]
    fn test_lfu_promote() {
        let mut tracker = LfuEvictionTracker::new();
        tracker.insert(1);
        tracker.insert(2);
        tracker.promote(1);
        // key 1 is at freq 2; key 2 is at freq 1 → evict key 2
        let victim = tracker.evict();
        assert_eq!(victim, Some(2));
    }

    #[test]
    fn test_lfu_evict_lowest_frequency() {
        let mut tracker = LfuEvictionTracker::new();
        tracker.insert(10);
        tracker.insert(20);
        tracker.insert(30);
        // Promote 10 and 20 to freq 2; 30 stays at freq 1.
        tracker.promote(10);
        tracker.promote(20);
        let victim = tracker.evict();
        assert_eq!(victim, Some(30), "key 30 has lowest frequency");
    }

    #[test]
    fn test_lfu_evict_fifo_within_bucket() {
        let mut tracker = LfuEvictionTracker::new();
        // All keys at the same frequency; oldest (inserted first) should be
        // evicted first.
        tracker.insert(1);
        tracker.insert(2);
        tracker.insert(3);
        assert_eq!(tracker.evict(), Some(1));
        assert_eq!(tracker.evict(), Some(2));
        assert_eq!(tracker.evict(), Some(3));
    }

    #[test]
    fn test_lfu_evict_empty() {
        let mut tracker = LfuEvictionTracker::new();
        assert!(tracker.evict().is_none());
    }

    #[test]
    fn test_lfu_remove() {
        let mut tracker = LfuEvictionTracker::new();
        tracker.insert(55);
        assert!(tracker.remove(55));
        assert_eq!(tracker.frequency(55), 0);
        assert!(tracker.is_empty());
    }

    #[test]
    fn test_lfu_remove_absent() {
        let mut tracker = LfuEvictionTracker::new();
        assert!(!tracker.remove(999));
    }

    #[test]
    fn test_lfu_len_and_is_empty() {
        let mut tracker = LfuEvictionTracker::new();
        assert!(tracker.is_empty());
        tracker.insert(1);
        tracker.insert(2);
        assert_eq!(tracker.len(), 2);
        tracker.evict();
        assert_eq!(tracker.len(), 1);
    }

    #[test]
    fn test_lfu_insert_duplicate_no_reset() {
        let mut tracker = LfuEvictionTracker::new();
        tracker.insert(7);
        tracker.promote(7);
        tracker.promote(7);
        // Calling insert again on an existing key should be a no-op.
        tracker.insert(7);
        assert_eq!(tracker.frequency(7), 3);
    }

    // ── TinyLfuAdmission ──────────────────────────────────────────────────────

    #[test]
    fn test_tinylfu_should_admit_popular() {
        let mut gate = TinyLfuAdmission::new(100);
        let key = 42u64;
        // Record many accesses → high frequency.
        for _ in 0..20 {
            gate.record_access(key);
        }
        // Evicted entry has frequency 1 → popular key should be admitted.
        assert!(gate.should_admit(key, 1));
    }

    #[test]
    fn test_tinylfu_should_not_admit_cold() {
        let mut gate = TinyLfuAdmission::new(100);
        let candidate = 99u64;
        // Do not record any accesses to the candidate — frequency is 0.
        // Evicted entry had frequency 10 → cold candidate should NOT be admitted.
        let freq = gate.estimated_frequency(candidate);
        assert_eq!(freq, 0);
        // Direct check: freq 0 < evicted_freq 10 → not admitted.
        assert!(!gate.should_admit(candidate, 10));
    }

    #[test]
    fn test_tinylfu_decay() {
        let mut gate = TinyLfuAdmission::new(100);
        for _ in 0..8 {
            gate.record_access(1);
        }
        let before = gate.estimated_frequency(1);
        gate.decay();
        let after = gate.estimated_frequency(1);
        assert!(after <= before, "decay should not increase frequency");
    }

    // ── ArcTracker ────────────────────────────────────────────────────────────

    #[test]
    fn test_arc_initial_state() {
        let arc = ArcTracker::new(100);
        assert_eq!(arc.t1_size, 0);
        assert_eq!(arc.t2_size, 0);
        assert_eq!(arc.b1_size, 0);
        assert_eq!(arc.b2_size, 0);
        assert_eq!(arc.p, 0);
        assert_eq!(arc.capacity(), 100);
    }

    #[test]
    fn test_arc_admit_t1() {
        let mut arc = ArcTracker::new(10);
        arc.on_admit_t1();
        arc.on_admit_t1();
        assert_eq!(arc.t1_size, 2);
        assert_eq!(arc.live_size(), 2);
    }

    #[test]
    fn test_arc_promote_t1_to_t2() {
        let mut arc = ArcTracker::new(10);
        arc.on_admit_t1();
        arc.on_promote_t1_to_t2();
        assert_eq!(arc.t1_size, 0);
        assert_eq!(arc.t2_size, 1);
    }

    #[test]
    fn test_arc_evict_t1_ghost() {
        let mut arc = ArcTracker::new(10);
        arc.on_admit_t1();
        arc.on_evict_t1();
        assert_eq!(arc.t1_size, 0);
        assert_eq!(arc.b1_size, 1);
        assert_eq!(arc.ghost_size(), 1);
    }

    #[test]
    fn test_arc_evict_t2_ghost() {
        let mut arc = ArcTracker::new(10);
        arc.on_admit_t1();
        arc.on_promote_t1_to_t2();
        arc.on_evict_t2();
        assert_eq!(arc.t2_size, 0);
        assert_eq!(arc.b2_size, 1);
    }

    #[test]
    fn test_arc_adapt_on_hit_b1_increases_p() {
        let mut arc = ArcTracker::new(100);
        arc.b1_size = 5;
        arc.b2_size = 10;
        let p_before = arc.p;
        arc.adapt_on_hit_b1();
        assert!(arc.p > p_before, "p should increase on B1 hit");
    }

    #[test]
    fn test_arc_adapt_on_hit_b2_decreases_p() {
        let mut arc = ArcTracker::new(100);
        arc.p = 50;
        arc.b1_size = 10;
        arc.b2_size = 5;
        arc.adapt_on_hit_b2();
        assert!(arc.p < 50, "p should decrease on B2 hit");
    }

    #[test]
    fn test_arc_p_capped_at_capacity() {
        let mut arc = ArcTracker::new(10);
        arc.p = 9;
        arc.b1_size = 1;
        arc.b2_size = 100;
        arc.adapt_on_hit_b1();
        assert!(arc.p <= 10, "p must not exceed capacity");
    }

    #[test]
    fn test_arc_p_floor_at_zero() {
        let mut arc = ArcTracker::new(10);
        arc.p = 0;
        arc.b2_size = 5;
        arc.b1_size = 1;
        arc.adapt_on_hit_b2();
        assert_eq!(arc.p, 0, "p must not go below zero");
    }

    #[test]
    fn test_arc_is_full() {
        let mut arc = ArcTracker::new(3);
        arc.on_admit_t1();
        arc.on_admit_t1();
        assert!(!arc.is_full());
        arc.on_admit_t1();
        assert!(arc.is_full());
    }

    #[test]
    fn test_arc_should_evict_t1_when_over_target() {
        let mut arc = ArcTracker::new(10);
        arc.t1_size = 5;
        arc.t2_size = 3;
        arc.p = 2; // T1 target is 2 but actual is 5 → evict T1
        assert!(arc.should_evict_t1());
    }

    #[test]
    fn test_arc_should_evict_t2_when_t1_at_target() {
        let mut arc = ArcTracker::new(10);
        arc.t1_size = 2;
        arc.t2_size = 4;
        arc.p = 3; // T1 ≤ p → prefer evicting T2
        assert!(!arc.should_evict_t1());
    }

    #[test]
    fn test_arc_remove_b1_ghost() {
        let mut arc = ArcTracker::new(10);
        arc.b1_size = 3;
        arc.on_remove_b1_ghost();
        assert_eq!(arc.b1_size, 2);
    }

    #[test]
    fn test_arc_remove_b2_ghost() {
        let mut arc = ArcTracker::new(10);
        arc.b2_size = 5;
        arc.on_remove_b2_ghost();
        assert_eq!(arc.b2_size, 4);
    }

    // ── EvictionPolicy enum ───────────────────────────────────────────────────

    #[test]
    fn test_eviction_policy_equality() {
        assert_eq!(EvictionPolicy::Lru, EvictionPolicy::Lru);
        assert_ne!(EvictionPolicy::Lru, EvictionPolicy::Lfu);
        assert_ne!(EvictionPolicy::TinyLfu, EvictionPolicy::ArcCache);
    }

    #[test]
    fn test_eviction_policy_clone() {
        let p = EvictionPolicy::ArcCache;
        let q = p.clone();
        assert_eq!(p, q);
    }
}
