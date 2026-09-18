//! Cache metrics module: hit/miss rates, latency tracking, eviction counters.
//!
//! Provides [`CacheMetrics`] with atomic counters safe for multi-threaded
//! access, and a [`CacheMetricsSnapshot`] for point-in-time reporting.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

// ── CacheMetrics ─────────────────────────────────────────────────────────────

/// Atomic cache metrics counters.
///
/// All fields use `AtomicU64` with `Relaxed` ordering for maximum throughput;
/// consistency is guaranteed only when you call `snapshot`.
///
/// # Example
/// ```rust
/// use oximedia_cache::cache_metrics::CacheMetrics;
/// let m = CacheMetrics::new();
/// m.record_hit(500);
/// m.record_miss(1_200);
/// let s = m.snapshot();
/// assert!((s.hit_rate - 0.5).abs() < 1e-9);
/// ```
pub struct CacheMetrics {
    /// Total number of cache hits.
    hits: AtomicU64,
    /// Total number of cache misses.
    misses: AtomicU64,
    /// Total number of evictions.
    evictions: AtomicU64,
    /// Accumulated hit latency in nanoseconds.
    total_hit_latency_ns: AtomicU64,
    /// Accumulated miss latency in nanoseconds.
    total_miss_latency_ns: AtomicU64,
    /// Total number of latency samples (hits + misses with latency recorded).
    latency_samples: AtomicU64,
}

impl std::fmt::Debug for CacheMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CacheMetrics")
            .field("hits", &self.hits.load(Ordering::Relaxed))
            .field("misses", &self.misses.load(Ordering::Relaxed))
            .field("evictions", &self.evictions.load(Ordering::Relaxed))
            .finish()
    }
}

impl Default for CacheMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl CacheMetrics {
    /// Create a new zeroed `CacheMetrics` instance.
    pub fn new() -> Self {
        Self {
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            total_hit_latency_ns: AtomicU64::new(0),
            total_miss_latency_ns: AtomicU64::new(0),
            latency_samples: AtomicU64::new(0),
        }
    }

    /// Create a new `CacheMetrics` wrapped in an `Arc` for sharing across threads.
    pub fn new_shared() -> Arc<Self> {
        Arc::new(Self::new())
    }

    /// Record a cache hit with the given lookup latency in nanoseconds.
    ///
    /// Increments the hit counter and accumulates latency for average
    /// latency calculation.
    pub fn record_hit(&self, latency_ns: u64) {
        self.hits.fetch_add(1, Ordering::Relaxed);
        self.total_hit_latency_ns
            .fetch_add(latency_ns, Ordering::Relaxed);
        self.latency_samples.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a cache miss with the given lookup latency in nanoseconds.
    ///
    /// Increments the miss counter and accumulates latency for average
    /// latency calculation.
    pub fn record_miss(&self, latency_ns: u64) {
        self.misses.fetch_add(1, Ordering::Relaxed);
        self.total_miss_latency_ns
            .fetch_add(latency_ns, Ordering::Relaxed);
        self.latency_samples.fetch_add(1, Ordering::Relaxed);
    }

    /// Record one cache eviction.
    pub fn record_eviction(&self) {
        self.evictions.fetch_add(1, Ordering::Relaxed);
    }

    /// Return the current hit rate as a fraction in `[0.0, 1.0]`.
    ///
    /// Returns `0.0` when no lookups have been recorded yet.
    pub fn hit_rate(&self) -> f64 {
        let h = self.hits.load(Ordering::Relaxed);
        let m = self.misses.load(Ordering::Relaxed);
        let total = h + m;
        if total == 0 {
            0.0
        } else {
            h as f64 / total as f64
        }
    }

    /// Return the current miss rate as a fraction in `[0.0, 1.0]`.
    ///
    /// Returns `0.0` when no lookups have been recorded yet.
    pub fn miss_rate(&self) -> f64 {
        let h = self.hits.load(Ordering::Relaxed);
        let m = self.misses.load(Ordering::Relaxed);
        let total = h + m;
        if total == 0 {
            0.0
        } else {
            m as f64 / total as f64
        }
    }

    /// Return the average lookup latency in nanoseconds across all recorded
    /// operations (hits and misses combined).
    ///
    /// Returns `0.0` when no operations have been recorded.
    pub fn avg_latency_ns(&self) -> f64 {
        let samples = self.latency_samples.load(Ordering::Relaxed);
        if samples == 0 {
            return 0.0;
        }
        let total_lat = self.total_hit_latency_ns.load(Ordering::Relaxed)
            + self.total_miss_latency_ns.load(Ordering::Relaxed);
        total_lat as f64 / samples as f64
    }

    /// Return the total number of recorded hits.
    pub fn total_hits(&self) -> u64 {
        self.hits.load(Ordering::Relaxed)
    }

    /// Return the total number of recorded misses.
    pub fn total_misses(&self) -> u64 {
        self.misses.load(Ordering::Relaxed)
    }

    /// Return the total number of recorded evictions.
    pub fn total_evictions(&self) -> u64 {
        self.evictions.load(Ordering::Relaxed)
    }

    /// Return the eviction rate: `evictions / (hits + misses)`.
    ///
    /// Returns `0.0` when no lookups have been recorded yet.
    pub fn eviction_rate(&self) -> f64 {
        let h = self.hits.load(Ordering::Relaxed);
        let m = self.misses.load(Ordering::Relaxed);
        let evictions = self.evictions.load(Ordering::Relaxed);
        let total = h + m;
        if total == 0 {
            0.0
        } else {
            evictions as f64 / total as f64
        }
    }

    /// Reset all counters to zero.
    pub fn reset(&self) {
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
        self.evictions.store(0, Ordering::Relaxed);
        self.total_hit_latency_ns.store(0, Ordering::Relaxed);
        self.total_miss_latency_ns.store(0, Ordering::Relaxed);
        self.latency_samples.store(0, Ordering::Relaxed);
    }

    /// Capture an immutable point-in-time snapshot of the current metrics.
    pub fn snapshot(&self) -> CacheMetricsSnapshot {
        let total_hits = self.hits.load(Ordering::Relaxed);
        let total_misses = self.misses.load(Ordering::Relaxed);
        let total_evictions = self.evictions.load(Ordering::Relaxed);
        let total_lat = self.total_hit_latency_ns.load(Ordering::Relaxed)
            + self.total_miss_latency_ns.load(Ordering::Relaxed);
        let samples = self.latency_samples.load(Ordering::Relaxed);

        let total_ops = total_hits + total_misses;
        let hit_rate = if total_ops == 0 {
            0.0
        } else {
            total_hits as f64 / total_ops as f64
        };
        let miss_rate = if total_ops == 0 {
            0.0
        } else {
            total_misses as f64 / total_ops as f64
        };
        let avg_latency_ns = if samples == 0 {
            0.0
        } else {
            total_lat as f64 / samples as f64
        };
        let eviction_rate = if total_ops == 0 {
            0.0
        } else {
            total_evictions as f64 / total_ops as f64
        };

        CacheMetricsSnapshot {
            hit_rate,
            miss_rate,
            total_hits,
            total_misses,
            total_evictions,
            eviction_rate,
            avg_latency_ns,
        }
    }
}

// ── CacheMetricsSnapshot ─────────────────────────────────────────────────────

/// Immutable point-in-time snapshot of cache metrics.
///
/// Obtained by calling [`CacheMetrics::snapshot`].
#[derive(Debug, Clone)]
pub struct CacheMetricsSnapshot {
    /// Fraction of lookups that resulted in a hit: `hits / (hits + misses)`.
    pub hit_rate: f64,
    /// Fraction of lookups that resulted in a miss: `misses / (hits + misses)`.
    pub miss_rate: f64,
    /// Total cache hits at snapshot time.
    pub total_hits: u64,
    /// Total cache misses at snapshot time.
    pub total_misses: u64,
    /// Total evictions at snapshot time.
    pub total_evictions: u64,
    /// Eviction rate: `evictions / (hits + misses)`.
    pub eviction_rate: f64,
    /// Average lookup latency in nanoseconds.
    pub avg_latency_ns: f64,
}

impl CacheMetricsSnapshot {
    /// Return `true` when the hit rate exceeds `threshold` (e.g. `0.80`).
    pub fn is_hit_rate_above(&self, threshold: f64) -> bool {
        self.hit_rate > threshold
    }

    /// Return the total number of operations (hits + misses).
    pub fn total_ops(&self) -> u64 {
        self.total_hits + self.total_misses
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    // 1. New metrics start at zero
    #[test]
    fn test_new_metrics_zeroed() {
        let m = CacheMetrics::new();
        assert_eq!(m.total_hits(), 0);
        assert_eq!(m.total_misses(), 0);
        assert_eq!(m.total_evictions(), 0);
        assert_eq!(m.hit_rate(), 0.0);
        assert_eq!(m.miss_rate(), 0.0);
        assert_eq!(m.avg_latency_ns(), 0.0);
    }

    // 2. record_hit increments hits
    #[test]
    fn test_record_hit() {
        let m = CacheMetrics::new();
        m.record_hit(100);
        m.record_hit(200);
        assert_eq!(m.total_hits(), 2);
        assert_eq!(m.total_misses(), 0);
    }

    // 3. record_miss increments misses
    #[test]
    fn test_record_miss() {
        let m = CacheMetrics::new();
        m.record_miss(500);
        assert_eq!(m.total_misses(), 1);
        assert_eq!(m.total_hits(), 0);
    }

    // 4. record_eviction increments evictions
    #[test]
    fn test_record_eviction() {
        let m = CacheMetrics::new();
        m.record_eviction();
        m.record_eviction();
        m.record_eviction();
        assert_eq!(m.total_evictions(), 3);
    }

    // 5. hit_rate with equal hits and misses
    #[test]
    fn test_hit_rate_equal() {
        let m = CacheMetrics::new();
        for _ in 0..50 {
            m.record_hit(10);
        }
        for _ in 0..50 {
            m.record_miss(10);
        }
        assert!((m.hit_rate() - 0.5).abs() < 1e-9);
    }

    // 6. miss_rate is complement of hit_rate
    #[test]
    fn test_miss_rate_complement() {
        let m = CacheMetrics::new();
        m.record_hit(10);
        m.record_hit(10);
        m.record_hit(10);
        m.record_miss(10);
        let hr = m.hit_rate();
        let mr = m.miss_rate();
        assert!((hr + mr - 1.0).abs() < 1e-9, "hit+miss should equal 1.0");
    }

    // 7. avg_latency_ns calculation
    #[test]
    fn test_avg_latency_ns() {
        let m = CacheMetrics::new();
        m.record_hit(100);
        m.record_hit(300);
        m.record_miss(200);
        // (100 + 300 + 200) / 3 = 200.0
        let avg = m.avg_latency_ns();
        assert!((avg - 200.0).abs() < 1e-9, "expected 200ns avg, got {avg}");
    }

    // 8. snapshot captures consistent values
    #[test]
    fn test_snapshot_consistency() {
        let m = CacheMetrics::new();
        for i in 0u64..10 {
            m.record_hit(i * 10);
        }
        for _ in 0..5 {
            m.record_miss(50);
        }
        m.record_eviction();
        let s = m.snapshot();
        assert_eq!(s.total_hits, 10);
        assert_eq!(s.total_misses, 5);
        assert_eq!(s.total_evictions, 1);
        assert!((s.hit_rate - 10.0 / 15.0).abs() < 1e-9);
        assert!((s.miss_rate - 5.0 / 15.0).abs() < 1e-9);
        assert!((s.hit_rate + s.miss_rate - 1.0).abs() < 1e-9);
    }

    // 9. eviction_rate calculation
    #[test]
    fn test_eviction_rate() {
        let m = CacheMetrics::new();
        m.record_hit(10);
        m.record_miss(10);
        m.record_eviction();
        // 1 eviction / 2 ops = 0.5
        assert!((m.eviction_rate() - 0.5).abs() < 1e-9);
    }

    // 10. reset clears all counters
    #[test]
    fn test_reset() {
        let m = CacheMetrics::new();
        m.record_hit(1000);
        m.record_miss(2000);
        m.record_eviction();
        m.reset();
        assert_eq!(m.total_hits(), 0);
        assert_eq!(m.total_misses(), 0);
        assert_eq!(m.total_evictions(), 0);
        assert_eq!(m.avg_latency_ns(), 0.0);
        assert_eq!(m.hit_rate(), 0.0);
    }

    // 11. is_hit_rate_above threshold check
    #[test]
    fn test_snapshot_is_hit_rate_above() {
        let m = CacheMetrics::new();
        for _ in 0..90 {
            m.record_hit(10);
        }
        for _ in 0..10 {
            m.record_miss(10);
        }
        let s = m.snapshot();
        assert!(s.is_hit_rate_above(0.80));
        assert!(!s.is_hit_rate_above(0.95));
    }

    // 12. snapshot total_ops helper
    #[test]
    fn test_snapshot_total_ops() {
        let m = CacheMetrics::new();
        m.record_hit(1);
        m.record_hit(1);
        m.record_miss(1);
        let s = m.snapshot();
        assert_eq!(s.total_ops(), 3);
    }

    // 13. Concurrent recording from multiple threads
    #[test]
    fn test_concurrent_recording() {
        let m = Arc::new(CacheMetrics::new());
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let m2 = Arc::clone(&m);
                thread::spawn(move || {
                    for _ in 0..1000 {
                        m2.record_hit(50);
                        m2.record_miss(100);
                        m2.record_eviction();
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().expect("thread panicked");
        }
        assert_eq!(m.total_hits(), 8 * 1000);
        assert_eq!(m.total_misses(), 8 * 1000);
        assert_eq!(m.total_evictions(), 8 * 1000);
    }

    // 14. hit_rate and miss_rate are 0 before any ops
    #[test]
    fn test_zero_ops_rates() {
        let m = CacheMetrics::new();
        assert_eq!(m.hit_rate(), 0.0);
        assert_eq!(m.miss_rate(), 0.0);
        assert_eq!(m.eviction_rate(), 0.0);
    }

    // 15. new_shared creates Arc-wrapped metrics
    #[test]
    fn test_new_shared() {
        let m = CacheMetrics::new_shared();
        m.record_hit(1);
        assert_eq!(m.total_hits(), 1);
    }
}
