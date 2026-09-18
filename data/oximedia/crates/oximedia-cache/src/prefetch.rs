//! Prefetch module: pre-loading sequential media segments based on access patterns.
//!
//! Provides [`PrefetchStrategy`] variants and a [`Prefetcher`] that warms a
//! backing cache by predicting which keys will be accessed next.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

// ── Cache trait ───────────────────────────────────────────────────────────────

/// Minimal read/write interface required by the prefetcher.
///
/// Implementors must be `Send + Sync` so the prefetcher can issue async-style
/// warm operations from background contexts.
pub trait Cache: Send + Sync {
    /// Return `true` if `key` is currently present in the cache.
    fn contains(&self, key: &str) -> bool;

    /// Insert `(key, value)` into the cache.
    fn insert(&self, key: String, value: Vec<u8>);

    /// Return the value for `key`, or `None` if absent.
    fn get(&self, key: &str) -> Option<Vec<u8>>;
}

// ── PrefetchStrategy ─────────────────────────────────────────────────────────

/// Strategy controlling which keys the prefetcher will pre-warm.
#[derive(Debug, Clone)]
pub enum PrefetchStrategy {
    /// Assume keys are named like `"segment-NNN"`.
    ///
    /// On access to key `"segment-N"`, pre-fetch `"segment-(N+1)"` through
    /// `"segment-(N+lookahead)"`.
    Sequential {
        /// Number of keys to prefetch ahead of the current position.
        lookahead: usize,
    },

    /// Follow a fixed ordered access pattern.
    ///
    /// On access to key `K`, the next key in the pattern list is pre-warmed.
    /// When the end of the list is reached, the pattern wraps around.
    AccessPattern(Vec<String>),
}

impl PrefetchStrategy {
    /// Given the `current_key`, predict the next keys to prefetch.
    ///
    /// Returns an empty vec when no prediction is possible.
    pub fn predict_next(&self, current_key: &str) -> Vec<String> {
        match self {
            PrefetchStrategy::Sequential { lookahead } => {
                predict_sequential(current_key, *lookahead)
            }
            PrefetchStrategy::AccessPattern(pattern) => {
                predict_access_pattern(current_key, pattern)
            }
        }
    }
}

// ── Sequential prediction helper ─────────────────────────────────────────────

/// Extract the numeric suffix from `"<prefix><number>"` patterns.
///
/// Returns `(prefix, number)` when a trailing decimal sequence is found and
/// the character immediately before the digits is a recognised segment
/// separator (`-`, `_`, or `/`).  This prevents treating embedded digits in
/// file extensions (e.g. the `8` in `manifest.m3u8`) as sequential counters.
fn split_numeric_suffix(key: &str) -> Option<(&str, u64)> {
    // Find the last run of ASCII digits.
    let digits_start = key
        .char_indices()
        .rev()
        .take_while(|(_, c)| c.is_ascii_digit())
        .last()
        .map(|(i, _)| i);

    match digits_start {
        Some(idx) if idx < key.len() => {
            let prefix = &key[..idx];
            // The character immediately before the digit run must be a
            // recognised segment separator.  An empty prefix (all-digit key)
            // or a prefix that ends with an alphanumeric or dot character
            // indicates the digits are part of a name/extension, not a counter.
            let separator_ok = prefix
                .chars()
                .next_back()
                .map_or(false, |c| matches!(c, '-' | '_' | '/'));
            if !separator_ok {
                return None;
            }
            let num_str = &key[idx..];
            num_str.parse::<u64>().ok().map(|n| (prefix, n))
        }
        _ => None,
    }
}

fn predict_sequential(current_key: &str, lookahead: usize) -> Vec<String> {
    if lookahead == 0 {
        return Vec::new();
    }
    match split_numeric_suffix(current_key) {
        Some((prefix, n)) => (1..=lookahead as u64)
            .map(|offset| {
                // Preserve zero-padding width of the original number.
                let width = current_key.len() - prefix.len();
                if width > 1 {
                    format!("{prefix}{:0>width$}", n + offset, width = width)
                } else {
                    format!("{prefix}{}", n + offset)
                }
            })
            .collect(),
        None => Vec::new(),
    }
}

fn predict_access_pattern(current_key: &str, pattern: &[String]) -> Vec<String> {
    if pattern.is_empty() {
        return Vec::new();
    }
    // Find the current key in the pattern, then return the next entry.
    pattern
        .iter()
        .position(|k| k == current_key)
        .map(|idx| {
            let next_idx = (idx + 1) % pattern.len();
            vec![pattern[next_idx].clone()]
        })
        .unwrap_or_default()
}

// ── Prefetcher ────────────────────────────────────────────────────────────────

/// A prefetcher that warms a [`Cache`] based on the configured [`PrefetchStrategy`].
///
/// Call `trigger_prefetch` each time a key is accessed; the prefetcher will
/// synchronously insert placeholder entries for predicted future keys that are
/// not already present.
///
/// # Thread safety
///
/// `Prefetcher` is `Clone` and thread-safe when the inner cache is `Send +
/// Sync`.  The pending queue is protected by a `Mutex`.
pub struct Prefetcher {
    /// Strategy driving key prediction.
    pub strategy: PrefetchStrategy,
    /// Backing cache to warm.
    cache: Arc<dyn Cache>,
    /// FIFO queue of pending prefetch requests (for deferred processing).
    pending: Mutex<VecDeque<String>>,
    /// Maximum pending queue depth before oldest entries are dropped.
    max_pending: usize,
    /// Loader function that produces the value bytes for a given key.
    /// Defaults to producing an empty `Vec<u8>` (placeholder).
    loader: Arc<dyn Fn(&str) -> Vec<u8> + Send + Sync>,
}

impl std::fmt::Debug for Prefetcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Prefetcher")
            .field("strategy", &self.strategy)
            .field("max_pending", &self.max_pending)
            .finish()
    }
}

impl Prefetcher {
    /// Create a new `Prefetcher` with the given strategy and cache.
    ///
    /// Uses a no-op loader that inserts empty-byte placeholder entries.
    pub fn new(strategy: PrefetchStrategy, cache: Arc<dyn Cache>) -> Self {
        Self {
            strategy,
            cache,
            pending: Mutex::new(VecDeque::new()),
            max_pending: 256,
            loader: Arc::new(|_key| Vec::new()),
        }
    }

    /// Create a `Prefetcher` with a custom value loader function.
    ///
    /// The `loader` receives the key and returns the bytes that should be
    /// stored in the cache for that key (e.g. reads from disk or network).
    pub fn with_loader<F>(strategy: PrefetchStrategy, cache: Arc<dyn Cache>, loader: F) -> Self
    where
        F: Fn(&str) -> Vec<u8> + Send + Sync + 'static,
    {
        Self {
            strategy,
            cache,
            pending: Mutex::new(VecDeque::new()),
            max_pending: 256,
            loader: Arc::new(loader),
        }
    }

    /// Set the maximum number of pending prefetch requests.
    pub fn with_max_pending(mut self, max: usize) -> Self {
        self.max_pending = max.max(1);
        self
    }

    /// Trigger a prefetch based on `current_key`.
    ///
    /// Predicts the next keys using the configured strategy and immediately
    /// inserts them into the cache via the loader if they are not already
    /// present.  Predictions that cannot be determined (e.g. key does not
    /// match the expected pattern) are silently ignored.
    pub fn trigger_prefetch(&self, current_key: &str) {
        let predicted = self.strategy.predict_next(current_key);
        for key in predicted {
            if !self.cache.contains(&key) {
                let value = (self.loader)(&key);
                self.cache.insert(key.clone(), value);
                // Enqueue in the pending list for potential async drain.
                if let Ok(mut q) = self.pending.lock() {
                    if q.len() >= self.max_pending {
                        q.pop_front();
                    }
                    q.push_back(key);
                }
            }
        }
    }

    /// Return the number of keys currently in the pending queue.
    pub fn pending_count(&self) -> usize {
        self.pending.lock().map(|q| q.len()).unwrap_or(0)
    }

    /// Drain the pending queue and return all queued keys.
    ///
    /// This can be used by a background worker to post-process prefetch
    /// completions.
    pub fn drain_pending(&self) -> Vec<String> {
        self.pending
            .lock()
            .map(|mut q| q.drain(..).collect())
            .unwrap_or_default()
    }

    /// Return a reference to the underlying cache.
    pub fn cache(&self) -> &Arc<dyn Cache> {
        &self.cache
    }
}

// ── Simple in-memory cache impl for tests ────────────────────────────────────

/// Minimal in-memory [`Cache`] implementation backed by a `Mutex<HashMap>`.
///
/// Used primarily in tests and examples.
pub struct MemoryCache {
    store: Mutex<std::collections::HashMap<String, Vec<u8>>>,
}

impl MemoryCache {
    /// Create a new empty `MemoryCache`.
    pub fn new() -> Self {
        Self {
            store: Mutex::new(std::collections::HashMap::new()),
        }
    }
}

impl Default for MemoryCache {
    fn default() -> Self {
        Self::new()
    }
}

impl Cache for MemoryCache {
    fn contains(&self, key: &str) -> bool {
        self.store
            .lock()
            .map(|m| m.contains_key(key))
            .unwrap_or(false)
    }

    fn insert(&self, key: String, value: Vec<u8>) {
        if let Ok(mut m) = self.store.lock() {
            m.insert(key, value);
        }
    }

    fn get(&self, key: &str) -> Option<Vec<u8>> {
        self.store.lock().ok().and_then(|m| m.get(key).cloned())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    fn make_cache() -> Arc<MemoryCache> {
        Arc::new(MemoryCache::new())
    }

    // 1. predict_next for sequential with numeric suffix
    #[test]
    fn test_sequential_predict_basic() {
        let strategy = PrefetchStrategy::Sequential { lookahead: 3 };
        let next = strategy.predict_next("segment-005");
        assert_eq!(next, vec!["segment-006", "segment-007", "segment-008"]);
    }

    // 2. predict_next with zero lookahead returns empty
    #[test]
    fn test_sequential_predict_zero_lookahead() {
        let strategy = PrefetchStrategy::Sequential { lookahead: 0 };
        assert!(strategy.predict_next("seg-1").is_empty());
    }

    // 3. predict_next on non-numeric key returns empty
    #[test]
    fn test_sequential_predict_non_numeric() {
        let strategy = PrefetchStrategy::Sequential { lookahead: 2 };
        assert!(strategy.predict_next("manifest.m3u8").is_empty());
    }

    // 4. AccessPattern predict next key
    #[test]
    fn test_access_pattern_predict_next() {
        let keys = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let strategy = PrefetchStrategy::AccessPattern(keys);
        let next = strategy.predict_next("b");
        assert_eq!(next, vec!["c"]);
    }

    // 5. AccessPattern wraps around at end
    #[test]
    fn test_access_pattern_wrap_around() {
        let keys = vec!["x".to_string(), "y".to_string(), "z".to_string()];
        let strategy = PrefetchStrategy::AccessPattern(keys);
        let next = strategy.predict_next("z");
        assert_eq!(next, vec!["x"]);
    }

    // 6. AccessPattern returns empty for unknown current_key
    #[test]
    fn test_access_pattern_unknown_key() {
        let keys = vec!["a".to_string(), "b".to_string()];
        let strategy = PrefetchStrategy::AccessPattern(keys);
        assert!(strategy.predict_next("unknown").is_empty());
    }

    // 7. trigger_prefetch warms cache sequentially
    #[test]
    fn test_trigger_prefetch_sequential() {
        let cache = make_cache();
        let prefetcher = Prefetcher::new(
            PrefetchStrategy::Sequential { lookahead: 2 },
            Arc::clone(&cache) as Arc<dyn Cache>,
        );
        prefetcher.trigger_prefetch("seg-010");
        assert!(cache.contains("seg-011"), "seg-011 should be prefetched");
        assert!(cache.contains("seg-012"), "seg-012 should be prefetched");
        assert!(
            !cache.contains("seg-013"),
            "seg-013 should NOT be prefetched"
        );
    }

    // 8. trigger_prefetch does not overwrite existing cached entry
    #[test]
    fn test_trigger_prefetch_no_overwrite() {
        let cache = make_cache();
        // Pre-populate with known value.
        cache.insert("seg-002".to_string(), vec![0xAB]);
        let prefetcher = Prefetcher::new(
            PrefetchStrategy::Sequential { lookahead: 2 },
            Arc::clone(&cache) as Arc<dyn Cache>,
        );
        prefetcher.trigger_prefetch("seg-001");
        // seg-002 should retain its original value.
        assert_eq!(
            cache.get("seg-002"),
            Some(vec![0xAB]),
            "existing entry should not be overwritten"
        );
    }

    // 9. Custom loader produces correct values
    #[test]
    fn test_custom_loader() {
        let cache = make_cache();
        let prefetcher = Prefetcher::with_loader(
            PrefetchStrategy::Sequential { lookahead: 1 },
            Arc::clone(&cache) as Arc<dyn Cache>,
            |key| format!("data-for-{key}").into_bytes(),
        );
        prefetcher.trigger_prefetch("chunk-004");
        let val = cache
            .get("chunk-005")
            .expect("chunk-005 should be in cache");
        assert_eq!(val, b"data-for-chunk-005");
    }

    // 10. pending_count and drain_pending
    #[test]
    fn test_pending_queue() {
        let cache = make_cache();
        let prefetcher = Prefetcher::new(
            PrefetchStrategy::Sequential { lookahead: 3 },
            Arc::clone(&cache) as Arc<dyn Cache>,
        );
        prefetcher.trigger_prefetch("frame-100");
        // 3 keys should be queued (101, 102, 103).
        assert_eq!(prefetcher.pending_count(), 3);
        let drained = prefetcher.drain_pending();
        assert_eq!(drained.len(), 3);
        assert_eq!(prefetcher.pending_count(), 0);
    }

    // 11. max_pending limits queue depth (oldest dropped)
    #[test]
    fn test_max_pending_limit() {
        let cache = make_cache();
        let prefetcher = Prefetcher::new(
            PrefetchStrategy::Sequential { lookahead: 5 },
            Arc::clone(&cache) as Arc<dyn Cache>,
        )
        .with_max_pending(3);
        // Prefetch from key with lookahead=5, but max_pending=3.
        prefetcher.trigger_prefetch("v-000");
        assert!(
            prefetcher.pending_count() <= 3,
            "pending should not exceed max_pending=3"
        );
    }

    // 12. AccessPattern prefetcher warms next key
    #[test]
    fn test_trigger_prefetch_access_pattern() {
        let cache = make_cache();
        let keys = vec![
            "intro".to_string(),
            "main".to_string(),
            "credits".to_string(),
        ];
        let prefetcher = Prefetcher::new(
            PrefetchStrategy::AccessPattern(keys),
            Arc::clone(&cache) as Arc<dyn Cache>,
        );
        prefetcher.trigger_prefetch("intro");
        assert!(cache.contains("main"), "main should be prefetched");
        assert!(
            !cache.contains("credits"),
            "credits should NOT be prefetched yet"
        );
    }

    // 13. Concurrent trigger_prefetch is safe
    #[test]
    fn test_concurrent_trigger_prefetch() {
        let cache = Arc::new(MemoryCache::new());
        let prefetcher = Arc::new(Prefetcher::new(
            PrefetchStrategy::Sequential { lookahead: 1 },
            Arc::clone(&cache) as Arc<dyn Cache>,
        ));
        let threads: Vec<_> = (0..4)
            .map(|i| {
                let p = Arc::clone(&prefetcher);
                thread::spawn(move || {
                    for j in 0..25u32 {
                        p.trigger_prefetch(&format!("seg-{}", i * 100 + j));
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().expect("thread panicked");
        }
        // No assertion except no panic; cache should have entries.
        // The prefetcher should have processed without data races.
    }

    // 14. Zero-padded segment numbering
    #[test]
    fn test_sequential_zero_padded() {
        let strategy = PrefetchStrategy::Sequential { lookahead: 2 };
        let next = strategy.predict_next("segment-099");
        assert_eq!(next, vec!["segment-100", "segment-101"]);
    }
}
