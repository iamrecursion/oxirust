//! Higher-level inference memoization cache keyed by `(expr_fingerprint, input_hash)`.
//!
//! This module provides a memoization layer distinct from the lower-level `TensorCache`
//! in `cache.rs`. Entries are keyed by a `MemoKey` that combines an expression fingerprint
//! (derived from a `TLExpr`) with an optional hash over input values.

use std::collections::{HashMap, VecDeque};
use std::marker::PhantomData;
use std::time::{Duration, Instant};

// ─────────────────────────────────────────────────────────────────────────────
// MemoKey
// ─────────────────────────────────────────────────────────────────────────────

/// Key for memoized results.
///
/// Combines an expression fingerprint (structural identity of a `TLExpr`) with
/// a hash over concrete input values so that two calls sharing the same graph
/// shape but with different data produce distinct cache keys.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MemoKey {
    /// Structural fingerprint of the expression graph.
    pub expr_fingerprint: u64,
    /// Hash of the concrete input values (0 when inputs are not considered).
    pub input_hash: u64,
}

impl MemoKey {
    /// Construct a `MemoKey` directly from its two components.
    pub fn new(expr_fingerprint: u64, input_hash: u64) -> Self {
        Self {
            expr_fingerprint,
            input_hash,
        }
    }

    /// Build a key from a `TLExpr`, setting `input_hash` to 0.
    pub fn from_expr(expr: &tensorlogic_ir::TLExpr) -> Self {
        let fp = tensorlogic_ir::expr_fingerprint(expr);
        Self::new(fp, 0)
    }

    /// Build a key from a `TLExpr` plus a pre-computed input hash.
    pub fn from_expr_and_hash(expr: &tensorlogic_ir::TLExpr, input_hash: u64) -> Self {
        let fp = tensorlogic_ir::expr_fingerprint(expr);
        Self::new(fp, input_hash)
    }

    /// Compute an FNV-1a hash over a slice of `f64` values.
    ///
    /// The result can be supplied as `input_hash` when constructing a `MemoKey`
    /// so that semantically identical graphs but with different numeric inputs
    /// map to different cache entries.
    pub fn hash_inputs(inputs: &[f64]) -> u64 {
        // FNV-1a 64-bit: offset basis = 14695981039346656037, prime = 1099511628211
        let mut state: u64 = 14_695_981_039_346_656_037;
        for &v in inputs {
            // XOR the low byte first (FNV-1a byte-at-a-time over the 8 bytes of the f64)
            let bits = v.to_bits();
            for byte_idx in 0..8u64 {
                let byte = (bits >> (byte_idx * 8)) & 0xFF;
                state ^= byte;
                state = state.wrapping_mul(1_099_511_628_211);
            }
        }
        state
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MemoEvictionPolicy
// ─────────────────────────────────────────────────────────────────────────────

/// Eviction policy for the memo cache.
///
/// Distinct from the `EvictionPolicy` in `cache.rs` — this enum adds a
/// time-to-live variant and uses `Lru`/`Fifo` naming conventions to avoid
/// any confusion with the lower-level type.
#[derive(Debug, Clone)]
pub enum MemoEvictionPolicy {
    /// Evict the entry that has not been accessed for the longest time.
    Lru,
    /// Evict the entry that was inserted first (oldest by insertion order).
    Fifo,
    /// Evict entries older than the given `Duration`; when no such entry
    /// exists, fall back to oldest-inserted.
    Ttl(Duration),
}

// ─────────────────────────────────────────────────────────────────────────────
// MemoConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for a `MemoCache`.
#[derive(Debug, Clone)]
pub struct MemoConfig {
    /// Maximum number of entries the cache is allowed to hold.
    pub max_entries: usize,
    /// Optional time-to-live applied uniformly to every entry on access.
    ///
    /// Entries that are still within `ttl` are not considered expired by the
    /// `Ttl` eviction policy; however, setting this field does not automatically
    /// enable TTL eviction — use `MemoEvictionPolicy::Ttl` for that.
    pub ttl: Option<Duration>,
    /// The eviction policy used when the cache is at capacity.
    pub eviction: MemoEvictionPolicy,
}

impl Default for MemoConfig {
    fn default() -> Self {
        Self {
            max_entries: 1024,
            ttl: None,
            eviction: MemoEvictionPolicy::Lru,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MemoStats
// ─────────────────────────────────────────────────────────────────────────────

/// Runtime statistics for a `MemoCache`.
///
/// Distinct from `CacheStats` in `cache.rs` — these counters use `u64` (rather
/// than `usize`) and track an extra dimension: entries that were found but had
/// already expired.
#[derive(Debug, Clone, Default)]
pub struct MemoStats {
    /// Number of lookups that returned a cached value.
    pub hits: u64,
    /// Number of lookups that found no entry at all.
    pub misses: u64,
    /// Number of entries that were evicted to make room for new ones.
    pub evictions: u64,
    /// Number of lookups that found an entry but it had expired (TTL).
    pub expired_on_access: u64,
    /// Number of entries currently stored in the cache.
    pub current_entries: usize,
}

impl MemoStats {
    /// Fraction of total lookups (hit + miss + expired) that were hits.
    ///
    /// Returns 0.0 when no lookups have been made.
    pub fn hit_rate(&self) -> f64 {
        let total = self.total_lookups();
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Total number of cache lookups (hits + misses + expired accesses).
    pub fn total_lookups(&self) -> u64 {
        self.hits + self.misses + self.expired_on_access
    }

    /// Human-readable one-line summary.
    pub fn summary(&self) -> String {
        format!(
            "MemoCache: entries={} hits={} misses={} expired={} evictions={} hit_rate={:.1}%",
            self.current_entries,
            self.hits,
            self.misses,
            self.expired_on_access,
            self.evictions,
            self.hit_rate() * 100.0,
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MemoLookupResult
// ─────────────────────────────────────────────────────────────────────────────

/// The outcome of a `MemoCache::get` call.
#[derive(Debug, Clone)]
pub enum MemoLookupResult<V> {
    /// A valid, unexpired entry was found; the value is returned.
    Hit(V),
    /// No entry exists for the key.
    Miss,
    /// An entry existed but had exceeded its TTL; it has been removed.
    Expired,
}

// ─────────────────────────────────────────────────────────────────────────────
// MemoEntry  (private)
// ─────────────────────────────────────────────────────────────────────────────

/// Internal storage record for a single cached value.
#[derive(Debug, Clone)]
struct MemoEntry<V> {
    value: V,
    inserted_at: Instant,
    last_accessed: Instant,
    access_count: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// MemoCache
// ─────────────────────────────────────────────────────────────────────────────

/// A memoization cache for inference results keyed by `MemoKey`.
///
/// ## Eviction
///
/// When `insert` is called and the cache has reached `config.max_entries`, one
/// entry is evicted according to `config.eviction`:
///
/// - `Lru`  — the entry with the oldest `last_accessed` timestamp.
/// - `Fifo` — the entry at the front of the insertion-order deque.
/// - `Ttl`  — any expired entry first; if none, the front of the insertion
///   deque (i.e., FIFO fall-back).
///
/// ## TTL on access
///
/// If `config.ttl` is `Some(d)`, entries are checked on every `get` call.  An
/// entry whose `inserted_at` is older than `d` will be removed and
/// `MemoLookupResult::Expired` returned instead of `Hit`.
pub struct MemoCache<V: Clone> {
    entries: HashMap<MemoKey, MemoEntry<V>>,
    /// Tracks insertion / recency order.  Back = most-recently used/inserted.
    insertion_order: VecDeque<MemoKey>,
    config: MemoConfig,
    stats: MemoStats,
}

impl<V: Clone + std::fmt::Debug> MemoCache<V> {
    // ── Construction ─────────────────────────────────────────────────────────

    /// Create a new cache with the given configuration.
    pub fn new(config: MemoConfig) -> Self {
        let max = config.max_entries;
        Self {
            entries: HashMap::with_capacity(max.min(1024)),
            insertion_order: VecDeque::with_capacity(max.min(1024)),
            config,
            stats: MemoStats::default(),
        }
    }

    /// Create a cache with `MemoConfig::default()`.
    pub fn with_default() -> Self {
        Self::new(MemoConfig::default())
    }

    /// Create a cache with LRU eviction and a custom capacity.
    pub fn with_max_entries(max: usize) -> Self {
        Self::new(MemoConfig {
            max_entries: max,
            ..MemoConfig::default()
        })
    }

    // ── Core operations ───────────────────────────────────────────────────────

    /// Look up `key` in the cache.
    ///
    /// Returns:
    /// - `Hit(v)` — the key was present, fresh, and `v` is a clone of the value.
    /// - `Miss`   — the key was not found.
    /// - `Expired`— the key was found but had exceeded its TTL; it has been
    ///   removed from the cache.
    pub fn get(&mut self, key: &MemoKey) -> MemoLookupResult<V> {
        // Check existence first without taking a mutable borrow.
        if !self.entries.contains_key(key) {
            self.stats.misses += 1;
            return MemoLookupResult::Miss;
        }

        // The entry exists — check TTL.
        if self.is_expired_by_key(key) {
            self.entries.remove(key);
            self.insertion_order.retain(|k| k != key);
            self.stats.current_entries = self.entries.len();
            self.stats.expired_on_access += 1;
            return MemoLookupResult::Expired;
        }

        // Fresh hit — update access metadata.
        if let Some(entry) = self.entries.get_mut(key) {
            entry.last_accessed = Instant::now();
            entry.access_count += 1;
            let value = entry.value.clone();
            // Move to back of deque for LRU tracking.
            self.update_lru(key);
            self.stats.hits += 1;
            MemoLookupResult::Hit(value)
        } else {
            // Should be unreachable given the contains_key check above.
            self.stats.misses += 1;
            MemoLookupResult::Miss
        }
    }

    /// Insert `value` under `key`.
    ///
    /// If the cache is already at capacity, one entry is evicted first
    /// according to `config.eviction`.
    pub fn insert(&mut self, key: MemoKey, value: V) {
        // If already present, update in-place without changing insertion order.
        if self.entries.contains_key(&key) {
            if let Some(entry) = self.entries.get_mut(&key) {
                entry.value = value;
                entry.last_accessed = Instant::now();
                entry.access_count += 1;
            }
            return;
        }

        // Evict if at capacity.
        if self.entries.len() >= self.config.max_entries {
            self.evict_one();
        }

        let now = Instant::now();
        let entry = MemoEntry {
            value,
            inserted_at: now,
            last_accessed: now,
            access_count: 1,
        };

        self.entries.insert(key.clone(), entry);
        self.insertion_order.push_back(key);
        self.stats.current_entries = self.entries.len();
    }

    /// Remove the entry for `key`, returning `true` if it was present.
    pub fn invalidate(&mut self, key: &MemoKey) -> bool {
        let removed = self.entries.remove(key).is_some();
        if removed {
            self.insertion_order.retain(|k| k != key);
            self.stats.current_entries = self.entries.len();
        }
        removed
    }

    /// Remove all entries and reset the entry count in stats.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.insertion_order.clear();
        self.stats.current_entries = 0;
    }

    /// Snapshot of the current statistics.
    pub fn stats(&self) -> &MemoStats {
        &self.stats
    }

    /// Number of entries currently stored.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Return `true` if the entry for `key` has exceeded its TTL.
    ///
    /// The TTL is taken from `config.ttl` (if set) or, when the eviction policy
    /// is `Ttl(d)`, from `d`.  If neither is configured the entry is never
    /// considered expired by this method.
    fn is_expired_by_key(&self, key: &MemoKey) -> bool {
        let ttl = match (&self.config.ttl, &self.config.eviction) {
            (Some(d), _) => Some(*d),
            (None, MemoEvictionPolicy::Ttl(d)) => Some(*d),
            _ => None,
        };
        if let Some(duration) = ttl {
            if let Some(entry) = self.entries.get(key) {
                return entry.inserted_at.elapsed() > duration;
            }
        }
        false
    }

    /// Return `true` if `entry` has exceeded its TTL (used during eviction).
    fn is_expired(&self, entry: &MemoEntry<V>) -> bool {
        let ttl = match (&self.config.ttl, &self.config.eviction) {
            (Some(d), _) => Some(*d),
            (None, MemoEvictionPolicy::Ttl(d)) => Some(*d),
            _ => None,
        };
        ttl.map(|d| entry.inserted_at.elapsed() > d)
            .unwrap_or(false)
    }

    /// Remove one entry according to the configured eviction policy.
    fn evict_one(&mut self) {
        let key_to_remove = match &self.config.eviction {
            MemoEvictionPolicy::Lru => self.find_lru_key(),
            MemoEvictionPolicy::Fifo => self.find_fifo_key(),
            MemoEvictionPolicy::Ttl(_) => {
                // Prefer any expired entry; fall back to FIFO.
                self.find_expired_key().or_else(|| self.find_fifo_key())
            }
        };

        if let Some(key) = key_to_remove {
            self.entries.remove(&key);
            self.insertion_order.retain(|k| k != &key);
            self.stats.evictions += 1;
            self.stats.current_entries = self.entries.len();
        }
    }

    /// Return the key of the least-recently-used entry.
    ///
    /// `insertion_order` is kept as an LRU deque: on every access the key is
    /// moved to the back, so the front is always the LRU entry.  Using the
    /// deque is O(1) and avoids `Instant` precision issues that could cause
    /// ties when entries are inserted or accessed within the same nanosecond.
    fn find_lru_key(&self) -> Option<MemoKey> {
        self.insertion_order.front().cloned()
    }

    /// Return the key at the front of the insertion-order deque.
    fn find_fifo_key(&self) -> Option<MemoKey> {
        self.insertion_order.front().cloned()
    }

    /// Return any key whose entry has exceeded the TTL.
    fn find_expired_key(&self) -> Option<MemoKey> {
        self.entries
            .iter()
            .find(|(_, e)| self.is_expired(e))
            .map(|(k, _)| k.clone())
    }

    /// Move `key` to the back of `insertion_order` to mark it as recently used.
    fn update_lru(&mut self, key: &MemoKey) {
        // Only bother if the eviction policy cares about recency.
        if matches!(self.config.eviction, MemoEvictionPolicy::Lru) {
            if let Some(pos) = self.insertion_order.iter().position(|k| k == key) {
                self.insertion_order.remove(pos);
                self.insertion_order.push_back(key.clone());
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience type alias
// ─────────────────────────────────────────────────────────────────────────────

/// Type alias for the primary use-case: caching `ArrayD<f64>` inference results.
pub type ExprMemoCache = MemoCache<ndarray::ArrayD<f64>>;

// ─────────────────────────────────────────────────────────────────────────────
// MemoCacheBuilder
// ─────────────────────────────────────────────────────────────────────────────

/// Builder for `MemoCache`, following the COOLJAPAN builder pattern.
pub struct MemoCacheBuilder<V: Clone + std::fmt::Debug> {
    config: MemoConfig,
    _phantom: PhantomData<V>,
}

impl<V: Clone + std::fmt::Debug> MemoCacheBuilder<V> {
    /// Start building with default configuration.
    pub fn new() -> Self {
        Self {
            config: MemoConfig::default(),
            _phantom: PhantomData,
        }
    }

    /// Set the maximum number of entries.
    pub fn max_entries(mut self, max: usize) -> Self {
        self.config.max_entries = max;
        self
    }

    /// Enable per-entry time-to-live.
    pub fn ttl(mut self, duration: Duration) -> Self {
        self.config.ttl = Some(duration);
        self
    }

    /// Set the eviction policy.
    pub fn eviction(mut self, policy: MemoEvictionPolicy) -> Self {
        self.config.eviction = policy;
        self
    }

    /// Consume the builder and produce a `MemoCache`.
    pub fn build(self) -> MemoCache<V> {
        MemoCache::new(self.config)
    }
}

impl<V: Clone + std::fmt::Debug> Default for MemoCacheBuilder<V> {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use tensorlogic_ir::{TLExpr, Term};

    // ── helpers ───────────────────────────────────────────────────────────────

    fn make_expr_a() -> TLExpr {
        TLExpr::pred("foo", vec![Term::var("x")])
    }

    fn make_expr_b() -> TLExpr {
        TLExpr::pred("bar", vec![Term::var("y")])
    }

    // ── MemoKey tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_memo_key_equality() {
        let k1 = MemoKey::new(42, 99);
        let k2 = MemoKey::new(42, 99);
        let k3 = MemoKey::new(42, 100);
        assert_eq!(k1, k2);
        assert_ne!(k1, k3);
    }

    #[test]
    fn test_memo_key_from_expr() {
        let expr = make_expr_a();
        let key = MemoKey::from_expr(&expr);
        assert_eq!(key.input_hash, 0);
        // Fingerprint should be deterministic
        let key2 = MemoKey::from_expr(&expr);
        assert_eq!(key.expr_fingerprint, key2.expr_fingerprint);
    }

    #[test]
    fn test_memo_key_hash_inputs_consistent() {
        let inputs = vec![1.0_f64, 2.0, 3.0];
        let h1 = MemoKey::hash_inputs(&inputs);
        let h2 = MemoKey::hash_inputs(&inputs);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_memo_key_hash_inputs_different() {
        let h1 = MemoKey::hash_inputs(&[1.0, 2.0, 3.0]);
        let h2 = MemoKey::hash_inputs(&[1.0, 2.0, 4.0]);
        assert_ne!(h1, h2);
    }

    // ── MemoCache basic tests ─────────────────────────────────────────────────

    #[test]
    fn test_memo_cache_miss_on_empty() {
        let mut cache: MemoCache<i32> = MemoCache::with_default();
        let key = MemoKey::new(1, 0);
        assert!(matches!(cache.get(&key), MemoLookupResult::Miss));
        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn test_memo_cache_hit_after_insert() {
        let mut cache: MemoCache<i32> = MemoCache::with_default();
        let key = MemoKey::new(7, 0);
        cache.insert(key.clone(), 42);
        assert!(matches!(cache.get(&key), MemoLookupResult::Hit(42)));
        assert_eq!(cache.stats().hits, 1);
    }

    #[test]
    fn test_memo_cache_hit_rate_zero_initially() {
        let cache: MemoCache<i32> = MemoCache::with_default();
        assert_eq!(cache.stats().hit_rate(), 0.0);
    }

    #[test]
    fn test_memo_cache_hit_rate_after_hit() {
        let mut cache: MemoCache<i32> = MemoCache::with_default();
        let key = MemoKey::new(1, 0);
        cache.insert(key.clone(), 10);
        cache.get(&key); // hit
        cache.get(&MemoKey::new(2, 0)); // miss
        let rate = cache.stats().hit_rate();
        assert!((rate - 0.5).abs() < 1e-9, "expected 0.5, got {rate}");
    }

    // ── Eviction policy tests ─────────────────────────────────────────────────

    #[test]
    fn test_memo_cache_lru_evicts_oldest_access() {
        // max_entries = 2, insert k1,k2 then access k1, then insert k3 → k2 evicted
        let mut cache: MemoCache<i32> = MemoCache::new(MemoConfig {
            max_entries: 2,
            ttl: None,
            eviction: MemoEvictionPolicy::Lru,
        });

        let k1 = MemoKey::new(1, 0);
        let k2 = MemoKey::new(2, 0);
        let k3 = MemoKey::new(3, 0);

        cache.insert(k1.clone(), 1);
        cache.insert(k2.clone(), 2);
        // Access k1 so k2 becomes LRU
        cache.get(&k1);
        // Inserting k3 must evict k2
        cache.insert(k3.clone(), 3);

        assert!(matches!(cache.get(&k1), MemoLookupResult::Hit(1)));
        assert!(matches!(cache.get(&k2), MemoLookupResult::Miss));
        assert!(matches!(cache.get(&k3), MemoLookupResult::Hit(3)));
        assert!(cache.stats().evictions >= 1);
    }

    #[test]
    fn test_memo_cache_fifo_evicts_first_inserted() {
        let mut cache: MemoCache<i32> = MemoCache::new(MemoConfig {
            max_entries: 2,
            ttl: None,
            eviction: MemoEvictionPolicy::Fifo,
        });

        let k1 = MemoKey::new(1, 0);
        let k2 = MemoKey::new(2, 0);
        let k3 = MemoKey::new(3, 0);

        cache.insert(k1.clone(), 10);
        cache.insert(k2.clone(), 20);
        // Accessing k1 should NOT change eviction order for FIFO
        cache.get(&k1);
        cache.insert(k3.clone(), 30); // k1 is the oldest insert → evicted

        assert!(matches!(cache.get(&k1), MemoLookupResult::Miss));
        assert!(matches!(cache.get(&k2), MemoLookupResult::Hit(20)));
        assert!(matches!(cache.get(&k3), MemoLookupResult::Hit(30)));
    }

    #[test]
    fn test_memo_cache_ttl_expires_entry() {
        let ttl = Duration::from_millis(10);
        let mut cache: MemoCache<i32> = MemoCache::new(MemoConfig {
            max_entries: 16,
            ttl: Some(ttl),
            eviction: MemoEvictionPolicy::Ttl(ttl),
        });

        let key = MemoKey::new(99, 0);
        cache.insert(key.clone(), 55);

        // Should still be fresh immediately.
        assert!(matches!(cache.get(&key), MemoLookupResult::Hit(55)));

        // Wait for expiry.
        thread::sleep(Duration::from_millis(20));

        assert!(matches!(cache.get(&key), MemoLookupResult::Expired));
        assert_eq!(cache.stats().expired_on_access, 1);
    }

    // ── Invalidation & clear tests ────────────────────────────────────────────

    #[test]
    fn test_memo_cache_invalidate_key() {
        let mut cache: MemoCache<i32> = MemoCache::with_default();
        let key = MemoKey::new(5, 0);
        cache.insert(key.clone(), 77);
        assert!(cache.invalidate(&key));
        assert!(!cache.invalidate(&key)); // already gone
        assert!(matches!(cache.get(&key), MemoLookupResult::Miss));
    }

    #[test]
    fn test_memo_cache_clear() {
        let mut cache: MemoCache<i32> = MemoCache::with_default();
        cache.insert(MemoKey::new(1, 0), 1);
        cache.insert(MemoKey::new(2, 0), 2);
        assert_eq!(cache.len(), 2);
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.stats().current_entries, 0);
    }

    // ── Length / empty tests ──────────────────────────────────────────────────

    #[test]
    fn test_memo_cache_len() {
        let mut cache: MemoCache<i32> = MemoCache::with_default();
        assert_eq!(cache.len(), 0);
        cache.insert(MemoKey::new(1, 0), 10);
        assert_eq!(cache.len(), 1);
        cache.insert(MemoKey::new(2, 0), 20);
        assert_eq!(cache.len(), 2);
    }

    // ── Stats tests ───────────────────────────────────────────────────────────

    #[test]
    fn test_memo_stats_total_lookups() {
        let mut cache: MemoCache<i32> = MemoCache::with_default();
        let key = MemoKey::new(1, 0);
        cache.insert(key.clone(), 1);
        cache.get(&key); // hit
        cache.get(&MemoKey::new(99, 0)); // miss
        assert_eq!(cache.stats().total_lookups(), 2);
    }

    #[test]
    fn test_memo_stats_summary_nonempty() {
        let mut cache: MemoCache<i32> = MemoCache::with_default();
        cache.insert(MemoKey::new(1, 0), 1);
        cache.get(&MemoKey::new(1, 0));
        let summary = cache.stats().summary();
        assert!(summary.contains("MemoCache"));
        assert!(summary.contains("hits=1"));
    }

    // ── MemoLookupResult variant tests ────────────────────────────────────────

    #[test]
    fn test_memo_lookup_result_variants() {
        // Ensure all three variants can be constructed and pattern-matched.
        let hit: MemoLookupResult<i32> = MemoLookupResult::Hit(42);
        let miss: MemoLookupResult<i32> = MemoLookupResult::Miss;
        let expired: MemoLookupResult<i32> = MemoLookupResult::Expired;

        assert!(matches!(hit, MemoLookupResult::Hit(42)));
        assert!(matches!(miss, MemoLookupResult::Miss));
        assert!(matches!(expired, MemoLookupResult::Expired));
    }

    // ── Builder tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_memo_cache_builder_default() {
        let cache: MemoCache<i32> = MemoCacheBuilder::new().build();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_memo_cache_builder_custom_config() {
        let cache: MemoCache<i32> = MemoCacheBuilder::new()
            .max_entries(8)
            .ttl(Duration::from_secs(60))
            .eviction(MemoEvictionPolicy::Fifo)
            .build();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    // ── Type alias test ───────────────────────────────────────────────────────

    #[test]
    fn test_expr_memo_cache_type_alias() {
        use ndarray::ArrayD;
        let mut cache: ExprMemoCache = MemoCache::with_default();
        let key = MemoKey::from_expr(&make_expr_a());
        let arr = ArrayD::<f64>::zeros(ndarray::IxDyn(&[2, 3]));
        cache.insert(key.clone(), arr.clone());
        assert!(matches!(cache.get(&key), MemoLookupResult::Hit(_)));
    }

    // ── Expr-based key tests ──────────────────────────────────────────────────

    #[test]
    fn test_memo_key_from_expr_different_exprs() {
        let ka = MemoKey::from_expr(&make_expr_a());
        let kb = MemoKey::from_expr(&make_expr_b());
        // Different literal values → different fingerprints
        assert_ne!(ka.expr_fingerprint, kb.expr_fingerprint);
    }

    #[test]
    fn test_memo_key_from_expr_and_hash() {
        let expr = make_expr_a();
        let h = MemoKey::hash_inputs(&[1.0, 2.0]);
        let key = MemoKey::from_expr_and_hash(&expr, h);
        assert_eq!(key.input_hash, h);
        assert_ne!(key.input_hash, 0);
    }
}
