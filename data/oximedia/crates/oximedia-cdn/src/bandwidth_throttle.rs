//! Per-client and per-origin bandwidth rate limiting for CDN delivery.
//!
//! # Overview
//!
//! `BandwidthThrottle` tracks byte-level rate-limiting for individual
//! clients or origins using a **token-bucket** algorithm.  Each bucket
//! refills at a configurable rate and has a configurable burst capacity.
//! A central `ThrottleRegistry` manages named buckets and provides a
//! high-level API for recording traffic and checking whether a sender
//! should be throttled.
//!
//! # Token-bucket algorithm
//!
//! - Each bucket has `capacity` tokens (bytes).
//! - Tokens refill at `rate_bytes_per_sec` per elapsed second.
//! - A request for `n` bytes succeeds if `tokens >= n`; tokens are
//!   decremented by `n` on success.
//! - If `tokens < n`, the request is *throttled* and tokens are **not**
//!   consumed (caller should retry after `deficit / rate` seconds).
//!
//! # Thread safety
//!
//! Each `TokenBucket` is wrapped in an `Arc<Mutex<…>>` inside the registry
//! so that concurrent writers contend per-bucket rather than globally.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use thiserror::Error;

// ─── Errors ───────────────────────────────────────────────────────────────────

/// Errors that can arise from bandwidth throttle operations.
#[derive(Debug, Error, PartialEq)]
pub enum ThrottleError {
    /// The requested transfer is too large to ever fit in the bucket
    /// (exceeds burst capacity).
    #[error(
        "request size {requested} bytes exceeds bucket burst capacity {capacity} bytes for '{id}'"
    )]
    ExceedsBurstCapacity {
        /// Client/origin identifier.
        id: String,
        /// Requested bytes.
        requested: u64,
        /// Maximum burst capacity.
        capacity: u64,
    },
    /// A Mutex was poisoned.
    #[error("internal lock poisoned for bucket '{0}'")]
    LockPoisoned(String),
}

// ─── ThrottleConfig ───────────────────────────────────────────────────────────

/// Configuration for a token-bucket bandwidth limiter.
#[derive(Debug, Clone)]
pub struct ThrottleConfig {
    /// Sustained rate in bytes per second.
    pub rate_bytes_per_sec: u64,
    /// Maximum burst capacity in bytes (initial token count).
    pub burst_capacity: u64,
    /// If `true`, throttle checks that exceed burst capacity return an error
    /// instead of queuing.
    pub reject_oversized: bool,
}

impl ThrottleConfig {
    /// Create a config with equal rate and burst capacity.
    pub fn new(rate_bytes_per_sec: u64) -> Self {
        Self {
            rate_bytes_per_sec,
            burst_capacity: rate_bytes_per_sec, // 1-second burst
            reject_oversized: true,
        }
    }

    /// Set the burst capacity to a multiple of the rate.
    pub fn with_burst_multiplier(mut self, multiplier: u64) -> Self {
        self.burst_capacity = self.rate_bytes_per_sec.saturating_mul(multiplier);
        self
    }

    /// Explicitly set the burst capacity in bytes.
    pub fn with_burst_capacity(mut self, capacity: u64) -> Self {
        self.burst_capacity = capacity;
        self
    }

    /// Disable rejection of requests exceeding burst capacity (they simply
    /// block the bucket until it refills).
    pub fn allow_oversized(mut self) -> Self {
        self.reject_oversized = false;
        self
    }
}

impl Default for ThrottleConfig {
    fn default() -> Self {
        // 10 MiB/s sustained, 10 MiB burst.
        Self::new(10 * 1024 * 1024)
    }
}

// ─── ThrottleResult ───────────────────────────────────────────────────────────

/// Outcome of a [`TokenBucket::consume`] call.
#[derive(Debug, Clone, PartialEq)]
pub enum ThrottleResult {
    /// Request was allowed; `tokens_remaining` reflects the bucket level after
    /// deducting the requested bytes.
    Allowed {
        /// Remaining tokens in the bucket after this request.
        tokens_remaining: u64,
    },
    /// Request was throttled; the bucket does not have enough tokens.
    Throttled {
        /// How many tokens are currently available.
        tokens_available: u64,
        /// How many bytes were requested.
        requested: u64,
        /// Estimated wait time until `requested` bytes become available.
        wait_duration: Duration,
    },
}

impl ThrottleResult {
    /// Returns `true` if the request was allowed.
    pub fn is_allowed(&self) -> bool {
        matches!(self, Self::Allowed { .. })
    }

    /// Returns `true` if the request was throttled.
    pub fn is_throttled(&self) -> bool {
        matches!(self, Self::Throttled { .. })
    }
}

// ─── TokenBucket ─────────────────────────────────────────────────────────────

/// A single token-bucket for one client or origin.
#[derive(Debug)]
pub struct TokenBucket {
    /// Identifier (client IP, origin URL, etc.).
    pub id: String,
    /// Configuration snapshot.
    pub config: ThrottleConfig,
    /// Current token count (in bytes).
    tokens: u64,
    /// Timestamp of the last refill operation.
    last_refill: Instant,
    /// Total bytes allowed since creation.
    total_bytes_allowed: u64,
    /// Total bytes throttled (denied) since creation.
    total_bytes_throttled: u64,
    /// Number of requests allowed.
    requests_allowed: u64,
    /// Number of requests throttled.
    requests_throttled: u64,
}

impl TokenBucket {
    /// Create a new full bucket (tokens == burst_capacity).
    pub fn new(id: impl Into<String>, config: ThrottleConfig) -> Self {
        let tokens = config.burst_capacity;
        Self {
            id: id.into(),
            config,
            tokens,
            last_refill: Instant::now(),
            total_bytes_allowed: 0,
            total_bytes_throttled: 0,
            requests_allowed: 0,
            requests_throttled: 0,
        }
    }

    /// Create a bucket with an explicit `now` (for testing).
    pub fn new_at(id: impl Into<String>, config: ThrottleConfig, now: Instant) -> Self {
        let tokens = config.burst_capacity;
        Self {
            id: id.into(),
            config,
            tokens,
            last_refill: now,
            total_bytes_allowed: 0,
            total_bytes_throttled: 0,
            requests_allowed: 0,
            requests_throttled: 0,
        }
    }

    /// Refill tokens based on elapsed time since the last refill.
    ///
    /// Call this before any `consume` to ensure the bucket is up to date.
    pub fn refill(&mut self) {
        self.refill_at(Instant::now());
    }

    /// Refill with an explicit `now` (for deterministic tests).
    pub fn refill_at(&mut self, now: Instant) {
        let elapsed = now.saturating_duration_since(self.last_refill);
        let new_tokens = (elapsed.as_secs_f64() * self.config.rate_bytes_per_sec as f64) as u64;
        if new_tokens > 0 {
            self.tokens = self
                .tokens
                .saturating_add(new_tokens)
                .min(self.config.burst_capacity);
            self.last_refill = now;
        }
    }

    /// Attempt to consume `bytes` from the bucket.
    ///
    /// Returns [`ThrottleResult::Allowed`] if there are enough tokens,
    /// [`ThrottleResult::Throttled`] otherwise.
    ///
    /// Returns [`ThrottleError::ExceedsBurstCapacity`] if `bytes` exceeds the
    /// burst capacity and `config.reject_oversized` is `true`.
    pub fn consume(&mut self, bytes: u64) -> Result<ThrottleResult, ThrottleError> {
        self.consume_at(bytes, Instant::now())
    }

    /// Consume with an explicit `now` (for deterministic tests).
    pub fn consume_at(
        &mut self,
        bytes: u64,
        now: Instant,
    ) -> Result<ThrottleResult, ThrottleError> {
        self.refill_at(now);

        if self.config.reject_oversized && bytes > self.config.burst_capacity {
            return Err(ThrottleError::ExceedsBurstCapacity {
                id: self.id.clone(),
                requested: bytes,
                capacity: self.config.burst_capacity,
            });
        }

        if self.tokens >= bytes {
            self.tokens -= bytes;
            self.total_bytes_allowed += bytes;
            self.requests_allowed += 1;
            Ok(ThrottleResult::Allowed {
                tokens_remaining: self.tokens,
            })
        } else {
            let deficit = bytes - self.tokens;
            let wait_secs = deficit as f64 / self.config.rate_bytes_per_sec as f64;
            let wait_duration = Duration::from_secs_f64(wait_secs);
            self.total_bytes_throttled += bytes;
            self.requests_throttled += 1;
            Ok(ThrottleResult::Throttled {
                tokens_available: self.tokens,
                requested: bytes,
                wait_duration,
            })
        }
    }

    /// Current token count without refilling.
    pub fn tokens(&self) -> u64 {
        self.tokens
    }

    /// Set tokens directly (for testing or administrative overrides).
    pub fn set_tokens(&mut self, tokens: u64) {
        self.tokens = tokens.min(self.config.burst_capacity);
    }

    /// Total bytes successfully transmitted through this bucket.
    pub fn total_bytes_allowed(&self) -> u64 {
        self.total_bytes_allowed
    }

    /// Total bytes that were throttled (denied).
    pub fn total_bytes_throttled(&self) -> u64 {
        self.total_bytes_throttled
    }

    /// Number of allowed requests.
    pub fn requests_allowed(&self) -> u64 {
        self.requests_allowed
    }

    /// Number of throttled requests.
    pub fn requests_throttled(&self) -> u64 {
        self.requests_throttled
    }

    /// Throttle ratio: `throttled / (allowed + throttled)`.
    pub fn throttle_ratio(&self) -> f64 {
        let total = self.requests_allowed + self.requests_throttled;
        if total == 0 {
            0.0
        } else {
            self.requests_throttled as f64 / total as f64
        }
    }

    /// Estimated time until `bytes` tokens will be available.
    pub fn wait_time_for(&self, bytes: u64) -> Duration {
        if self.tokens >= bytes {
            Duration::ZERO
        } else {
            let deficit = bytes - self.tokens;
            let secs = deficit as f64 / self.config.rate_bytes_per_sec as f64;
            Duration::from_secs_f64(secs)
        }
    }
}

// ─── ThrottleRegistry ────────────────────────────────────────────────────────

/// Thread-safe registry mapping identifiers to [`TokenBucket`]s.
///
/// Buckets are created on first access using the `default_config`.
/// A single `Mutex<HashMap>` protects the bucket map; individual bucket
/// operations do NOT hold the map lock (buckets are wrapped in their own
/// `Arc<Mutex<TokenBucket>>`).
pub struct ThrottleRegistry {
    buckets: Mutex<HashMap<String, Arc<Mutex<TokenBucket>>>>,
    /// Config applied to buckets that are auto-created on first access.
    pub default_config: ThrottleConfig,
}

impl ThrottleRegistry {
    /// Create an empty registry with the given default configuration.
    pub fn new(default_config: ThrottleConfig) -> Self {
        Self {
            buckets: Mutex::new(HashMap::new()),
            default_config,
        }
    }

    /// Register or replace a bucket with a specific config for `id`.
    pub fn register(&self, id: impl Into<String>, config: ThrottleConfig) {
        let id = id.into();
        if let Ok(mut guard) = self.buckets.lock() {
            let bucket = TokenBucket::new(id.clone(), config);
            guard.insert(id, Arc::new(Mutex::new(bucket)));
        }
    }

    /// Get (or auto-create) the bucket for `id`.
    pub fn get_or_create(&self, id: &str) -> Option<Arc<Mutex<TokenBucket>>> {
        let mut guard = self.buckets.lock().ok()?;
        if !guard.contains_key(id) {
            let bucket = TokenBucket::new(id, self.default_config.clone());
            guard.insert(id.to_string(), Arc::new(Mutex::new(bucket)));
        }
        guard.get(id).cloned()
    }

    /// Check whether `id` is currently throttled for `bytes`.
    ///
    /// Auto-creates the bucket if it does not exist.
    pub fn check(&self, id: &str, bytes: u64) -> Result<ThrottleResult, ThrottleError> {
        let arc = self
            .get_or_create(id)
            .ok_or_else(|| ThrottleError::LockPoisoned(id.to_string()))?;
        let mut bucket = arc
            .lock()
            .map_err(|_| ThrottleError::LockPoisoned(id.to_string()))?;
        bucket.consume(bytes)
    }

    /// Remove a bucket by `id`.  Returns `true` if a bucket was removed.
    pub fn remove(&self, id: &str) -> bool {
        self.buckets
            .lock()
            .map(|mut g| g.remove(id).is_some())
            .unwrap_or(false)
    }

    /// Number of registered buckets.
    pub fn len(&self) -> usize {
        self.buckets.lock().map(|g| g.len()).unwrap_or(0)
    }

    /// Returns `true` if the registry has no buckets.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Snapshot throttle statistics for all registered buckets.
    pub fn snapshot_all(&self) -> Vec<BucketSnapshot> {
        let Ok(guard) = self.buckets.lock() else {
            return Vec::new();
        };
        guard
            .values()
            .filter_map(|arc| arc.lock().ok().map(|b| BucketSnapshot::from(&*b)))
            .collect()
    }
}

impl Default for ThrottleRegistry {
    fn default() -> Self {
        Self::new(ThrottleConfig::default())
    }
}

// ─── BucketSnapshot ───────────────────────────────────────────────────────────

/// Immutable statistics snapshot for a single token bucket.
#[derive(Debug, Clone)]
pub struct BucketSnapshot {
    /// Bucket identifier.
    pub id: String,
    /// Current token count.
    pub tokens: u64,
    /// Burst capacity.
    pub capacity: u64,
    /// Sustained rate (bytes/s).
    pub rate_bytes_per_sec: u64,
    /// Total bytes allowed.
    pub total_bytes_allowed: u64,
    /// Total bytes throttled.
    pub total_bytes_throttled: u64,
    /// Requests allowed.
    pub requests_allowed: u64,
    /// Requests throttled.
    pub requests_throttled: u64,
}

impl From<&TokenBucket> for BucketSnapshot {
    fn from(b: &TokenBucket) -> Self {
        Self {
            id: b.id.clone(),
            tokens: b.tokens,
            capacity: b.config.burst_capacity,
            rate_bytes_per_sec: b.config.rate_bytes_per_sec,
            total_bytes_allowed: b.total_bytes_allowed,
            total_bytes_throttled: b.total_bytes_throttled,
            requests_allowed: b.requests_allowed,
            requests_throttled: b.requests_throttled,
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn make_config(rate: u64, burst: u64) -> ThrottleConfig {
        ThrottleConfig::new(rate).with_burst_capacity(burst)
    }

    // 1. New bucket starts full (tokens == burst_capacity).
    #[test]
    fn test_new_bucket_starts_full() {
        let config = make_config(1000, 5000);
        let b = TokenBucket::new("client1", config);
        assert_eq!(b.tokens(), 5000);
    }

    // 2. Consume within capacity → Allowed.
    #[test]
    fn test_consume_within_capacity() {
        let config = make_config(1000, 5000);
        let now = Instant::now();
        let mut b = TokenBucket::new_at("c", config, now);
        let result = b.consume_at(1000, now).expect("no bucket error");
        assert!(result.is_allowed());
        assert_eq!(b.tokens(), 4000);
    }

    // 3. Consume exceeding tokens → Throttled.
    #[test]
    fn test_consume_exceeding_tokens() {
        let config = make_config(1000, 1000);
        let now = Instant::now();
        let mut b = TokenBucket::new_at("c", config, now);
        // Drain the bucket first.
        b.consume_at(1000, now).expect("ok");
        // Now request more.
        let result = b.consume_at(1, now).expect("no bucket error");
        assert!(result.is_throttled());
        if let ThrottleResult::Throttled {
            tokens_available,
            requested,
            ..
        } = result
        {
            assert_eq!(tokens_available, 0);
            assert_eq!(requested, 1);
        } else {
            panic!("expected Throttled");
        }
    }

    // 4. Throttled request does not deduct tokens.
    #[test]
    fn test_throttled_does_not_deduct_tokens() {
        let config = make_config(1000, 500);
        let now = Instant::now();
        let mut b = TokenBucket::new_at("c", config, now);
        b.set_tokens(0);
        let before = b.tokens();
        b.consume_at(100, now).expect("no bucket err");
        assert_eq!(b.tokens(), before); // tokens unchanged
    }

    // 5. Refill adds tokens proportional to elapsed time.
    #[test]
    fn test_refill_adds_tokens() {
        let config = make_config(1000, 5000); // 1000 bytes/s
        let now = Instant::now();
        let mut b = TokenBucket::new_at("c", config, now);
        b.set_tokens(0); // drain
                         // Simulate 2 seconds passing.
        let later = now + Duration::from_secs(2);
        b.refill_at(later);
        // Should have 2000 tokens (2s * 1000 bytes/s).
        assert_eq!(b.tokens(), 2000);
    }

    // 6. Refill caps at burst_capacity.
    #[test]
    fn test_refill_caps_at_burst() {
        let config = make_config(1000, 3000);
        let now = Instant::now();
        let mut b = TokenBucket::new_at("c", config, now);
        b.set_tokens(2500);
        // After 5 seconds (5000 tokens), should cap at 3000.
        let later = now + Duration::from_secs(5);
        b.refill_at(later);
        assert_eq!(b.tokens(), 3000);
    }

    // 7. ExceedsBurstCapacity error when oversized request and reject_oversized=true.
    #[test]
    fn test_exceeds_burst_capacity_error() {
        let config = make_config(1000, 500);
        let now = Instant::now();
        let mut b = TokenBucket::new_at("client_x", config, now);
        let err = b.consume_at(600, now).unwrap_err();
        assert!(
            matches!(
                err,
                ThrottleError::ExceedsBurstCapacity {
                    requested: 600,
                    capacity: 500,
                    ..
                }
            ),
            "err={err:?}"
        );
    }

    // 8. allow_oversized allows large requests when enough tokens exist.
    #[test]
    fn test_allow_oversized_when_enough_tokens() {
        let config = make_config(1000, 500).allow_oversized();
        let now = Instant::now();
        let mut b = TokenBucket::new_at("c", config.clone(), now);
        // The burst capacity is 500, but we allow_oversized and tokens == capacity.
        // A request of 501 exceeds capacity but should not return an error.
        let result = b
            .consume_at(501, now)
            .expect("no error with allow_oversized");
        // 501 > 500 tokens → Throttled (not an error).
        assert!(result.is_throttled());
    }

    // 9. wait_time_for returns zero when tokens sufficient.
    #[test]
    fn test_wait_time_zero_when_sufficient() {
        let config = make_config(1000, 5000);
        let b = TokenBucket::new("c", config);
        assert_eq!(b.wait_time_for(1000), Duration::ZERO);
    }

    // 10. wait_time_for calculates deficit correctly.
    #[test]
    fn test_wait_time_calculated_correctly() {
        let config = make_config(1000, 5000); // 1000 bytes/s
        let now = Instant::now();
        let mut b = TokenBucket::new_at("c", config, now);
        b.set_tokens(0);
        // Need 2000 bytes at 1000 bytes/s → 2 seconds.
        let wait = b.wait_time_for(2000);
        let secs = wait.as_secs_f64();
        assert!((secs - 2.0).abs() < 0.001, "secs={secs}");
    }

    // 11. Total bytes allowed accumulates correctly.
    #[test]
    fn test_total_bytes_allowed_accumulates() {
        let config = make_config(1000, 5000);
        let now = Instant::now();
        let mut b = TokenBucket::new_at("c", config, now);
        b.consume_at(100, now).expect("ok");
        b.consume_at(200, now).expect("ok");
        assert_eq!(b.total_bytes_allowed(), 300);
        assert_eq!(b.requests_allowed(), 2);
    }

    // 12. Total bytes throttled accumulates correctly.
    #[test]
    fn test_total_bytes_throttled_accumulates() {
        let config = make_config(1000, 200);
        let now = Instant::now();
        let mut b = TokenBucket::new_at("c", config, now);
        b.set_tokens(0);
        b.consume_at(100, now).expect("ok");
        b.consume_at(50, now).expect("ok");
        assert_eq!(b.total_bytes_throttled(), 150);
        assert_eq!(b.requests_throttled(), 2);
    }

    // 13. throttle_ratio is 0 when no requests.
    #[test]
    fn test_throttle_ratio_no_requests() {
        let b = TokenBucket::new("c", ThrottleConfig::default());
        assert!((b.throttle_ratio() - 0.0).abs() < 1e-10);
    }

    // 14. throttle_ratio with mixed results.
    #[test]
    fn test_throttle_ratio_mixed() {
        let config = make_config(1000, 2000);
        let now = Instant::now();
        let mut b = TokenBucket::new_at("c", config, now);
        b.consume_at(1000, now).expect("ok"); // allowed
        b.consume_at(1000, now).expect("ok"); // allowed
        b.set_tokens(0);
        b.consume_at(1, now).expect("ok"); // throttled
                                           // 1 throttled out of 3 total → ratio = 1/3.
        let ratio = b.throttle_ratio();
        assert!((ratio - 1.0 / 3.0).abs() < 1e-10, "ratio={ratio}");
    }

    // ── ThrottleRegistry ──────────────────────────────────────────────────

    // 15. Registry starts empty.
    #[test]
    fn test_registry_starts_empty() {
        let reg = ThrottleRegistry::new(ThrottleConfig::default());
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
    }

    // 16. check auto-creates bucket on first access.
    #[test]
    fn test_check_auto_creates_bucket() {
        let reg = ThrottleRegistry::new(ThrottleConfig::new(100_000).with_burst_capacity(100_000));
        let result = reg.check("client-1", 1000).expect("no lock error");
        assert!(result.is_allowed());
        assert_eq!(reg.len(), 1);
    }

    // 17. register overrides default config.
    #[test]
    fn test_register_override() {
        let reg = ThrottleRegistry::new(ThrottleConfig::new(100_000));
        reg.register(
            "vip-client",
            ThrottleConfig::new(1_000_000).with_burst_capacity(5_000_000),
        );
        let arc = reg.get_or_create("vip-client").expect("bucket");
        let b = arc.lock().expect("lock");
        assert_eq!(b.config.rate_bytes_per_sec, 1_000_000);
    }

    // 18. check throttles after tokens exhausted.
    #[test]
    fn test_check_throttles() {
        let reg = ThrottleRegistry::new(ThrottleConfig::new(1000).with_burst_capacity(500));
        // First call drains the bucket.
        reg.check("slow-client", 500).expect("ok");
        // Second call should throttle.
        let result = reg.check("slow-client", 1).expect("ok");
        assert!(result.is_throttled(), "result={result:?}");
    }

    // 19. remove deletes the bucket.
    #[test]
    fn test_remove_bucket() {
        let reg = ThrottleRegistry::new(ThrottleConfig::default());
        reg.check("c", 0).expect("ok");
        assert!(!reg.is_empty());
        assert!(reg.remove("c"));
        assert!(reg.is_empty());
        assert!(!reg.remove("c")); // already gone
    }

    // 20. snapshot_all returns one entry per bucket.
    #[test]
    fn test_snapshot_all() {
        let reg = ThrottleRegistry::new(ThrottleConfig::new(1000).with_burst_capacity(1000));
        reg.check("a", 100).expect("ok");
        reg.check("b", 200).expect("ok");
        let snaps = reg.snapshot_all();
        assert_eq!(snaps.len(), 2);
    }

    // 21. BucketSnapshot from TokenBucket.
    #[test]
    fn test_bucket_snapshot() {
        let config = make_config(500, 2000);
        let now = Instant::now();
        let mut b = TokenBucket::new_at("snap-test", config, now);
        b.consume_at(300, now).expect("ok");
        let snap = BucketSnapshot::from(&b);
        assert_eq!(snap.id, "snap-test");
        assert_eq!(snap.tokens, 1700);
        assert_eq!(snap.capacity, 2000);
        assert_eq!(snap.rate_bytes_per_sec, 500);
        assert_eq!(snap.total_bytes_allowed, 300);
        assert_eq!(snap.requests_allowed, 1);
    }

    // 22. ThrottleConfig::default is 10 MiB/s.
    #[test]
    fn test_throttle_config_default() {
        let cfg = ThrottleConfig::default();
        assert_eq!(cfg.rate_bytes_per_sec, 10 * 1024 * 1024);
        assert_eq!(cfg.burst_capacity, 10 * 1024 * 1024);
    }

    // 23. with_burst_multiplier sets burst as multiple of rate.
    #[test]
    fn test_burst_multiplier() {
        let cfg = ThrottleConfig::new(1000).with_burst_multiplier(5);
        assert_eq!(cfg.burst_capacity, 5000);
    }

    // 24. Consecutive refills do not double-count tokens.
    #[test]
    fn test_no_double_refill() {
        let config = make_config(1000, 5000);
        let now = Instant::now();
        let mut b = TokenBucket::new_at("c", config, now);
        b.set_tokens(0);
        let t1 = now + Duration::from_secs(1);
        b.refill_at(t1);
        let t2 = t1 + Duration::from_millis(500);
        b.refill_at(t2);
        // After 1s: 1000 tokens; after another 0.5s: +500 → 1500.
        assert!(
            b.tokens() >= 1400 && b.tokens() <= 1600,
            "tokens={}",
            b.tokens()
        );
    }

    // 25. Multiple clients do not interfere.
    #[test]
    fn test_multiple_clients_independent() {
        let reg = ThrottleRegistry::new(ThrottleConfig::new(1000).with_burst_capacity(1000));
        // Client A uses 1000 bytes.
        let r_a = reg.check("client-a", 1000).expect("ok");
        assert!(r_a.is_allowed());
        // Client B has a fresh bucket and can also use 1000 bytes.
        let r_b = reg.check("client-b", 1000).expect("ok");
        assert!(r_b.is_allowed());
        // Client A is now throttled.
        let r_a2 = reg.check("client-a", 1).expect("ok");
        assert!(r_a2.is_throttled());
    }

    // 26. Throttled wait_duration is non-zero.
    #[test]
    fn test_throttled_wait_duration_non_zero() {
        let config = make_config(1000, 500);
        let now = Instant::now();
        let mut b = TokenBucket::new_at("c", config, now);
        b.set_tokens(0);
        let result = b.consume_at(500, now).expect("ok");
        if let ThrottleResult::Throttled { wait_duration, .. } = result {
            assert!(wait_duration > Duration::ZERO);
        } else {
            panic!("expected Throttled");
        }
    }

    // 27. ThrottleConfig::new sets reject_oversized = true.
    #[test]
    fn test_config_reject_oversized_default() {
        let cfg = ThrottleConfig::new(1000);
        assert!(cfg.reject_oversized);
        let cfg2 = cfg.allow_oversized();
        assert!(!cfg2.reject_oversized);
    }

    // 28. get_or_create returns same bucket on repeated calls.
    #[test]
    fn test_get_or_create_same_bucket() {
        let reg = ThrottleRegistry::new(ThrottleConfig::new(1000).with_burst_capacity(1000));
        reg.check("c", 100).expect("ok"); // creates and consumes 100
        let arc = reg.get_or_create("c").expect("bucket");
        let b = arc.lock().expect("lock");
        // Should have 900 tokens remaining (not re-created).
        assert_eq!(b.tokens(), 900);
    }

    // 29. set_tokens clamps to burst capacity.
    #[test]
    fn test_set_tokens_clamps() {
        let config = make_config(100, 500);
        let mut b = TokenBucket::new("c", config);
        b.set_tokens(99999);
        assert_eq!(b.tokens(), 500); // clamped to burst_capacity
    }

    // 30. ThrottleResult::is_allowed / is_throttled are mutually exclusive.
    #[test]
    fn test_throttle_result_flags_exclusive() {
        let allowed = ThrottleResult::Allowed {
            tokens_remaining: 100,
        };
        assert!(allowed.is_allowed());
        assert!(!allowed.is_throttled());

        let throttled = ThrottleResult::Throttled {
            tokens_available: 0,
            requested: 100,
            wait_duration: Duration::from_secs(1),
        };
        assert!(!throttled.is_allowed());
        assert!(throttled.is_throttled());
    }
}
