//! Rate limiting primitives for batch processing pipelines.
//!
//! Provides a token-bucket implementation and a concurrency limiter that
//! can be composed to enforce both throughput and parallelism constraints.
//! Also provides a per-user rate limiter (`UserRateLimiter`) to enforce
//! per-user submission quotas and concurrency caps.

#![allow(dead_code)]

use parking_lot::Mutex;
use std::collections::HashMap;

/// Configuration for rate and concurrency limiting.
#[derive(Debug, Clone)]
pub struct ThrottleConfig {
    /// Maximum number of items to process concurrently.
    pub max_concurrent: usize,
    /// Sustained throughput limit (items per second).
    pub rate_per_second: f64,
    /// Additional tokens that may be consumed in a single burst.
    pub burst_allowance: usize,
}

impl ThrottleConfig {
    /// Default configuration (8 concurrent, 10 rps, burst 5).
    #[must_use]
    pub fn default() -> Self {
        Self {
            max_concurrent: 8,
            rate_per_second: 10.0,
            burst_allowance: 5,
        }
    }

    /// Conservative configuration (2 concurrent, 2 rps, no burst).
    #[must_use]
    pub fn conservative() -> Self {
        Self {
            max_concurrent: 2,
            rate_per_second: 2.0,
            burst_allowance: 0,
        }
    }
}

/// A token-bucket throttle driven by caller-supplied wall-clock timestamps.
///
/// The caller is responsible for providing monotonic `now_ms` values.
#[derive(Debug, Clone)]
pub struct TokenBucketThrottle {
    /// Current number of available tokens (fractional).
    pub tokens: f64,
    /// Maximum number of tokens (bucket capacity).
    pub max_tokens: f64,
    /// Tokens added per millisecond.
    pub refill_rate: f64,
    /// Timestamp of the last refill (milliseconds).
    pub last_refill_ms: u64,
}

impl TokenBucketThrottle {
    /// Create a new token bucket.
    ///
    /// * `max_tokens` — bucket capacity.
    /// * `rate_per_second` — sustained token replenishment rate.
    /// * `now_ms` — current timestamp used to initialise the refill clock.
    #[must_use]
    pub fn new(max_tokens: f64, rate_per_second: f64, now_ms: u64) -> Self {
        Self {
            tokens: max_tokens,
            max_tokens,
            refill_rate: rate_per_second / 1_000.0,
            last_refill_ms: now_ms,
        }
    }

    /// Refill the bucket based on the elapsed time since the last refill.
    pub fn refill(&mut self, now_ms: u64) {
        let elapsed = now_ms.saturating_sub(self.last_refill_ms) as f64;
        self.tokens = (self.tokens + elapsed * self.refill_rate).min(self.max_tokens);
        self.last_refill_ms = now_ms;
    }

    /// Try to consume one token.
    ///
    /// Returns `true` when a token was successfully consumed, `false` when the
    /// bucket is empty (the caller should back off).
    pub fn try_consume(&mut self, now_ms: u64) -> bool {
        self.refill(now_ms);
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }

    /// Current number of available tokens.
    #[must_use]
    pub fn available_tokens(&self) -> f64 {
        self.tokens
    }
}

/// Limits the number of simultaneously active items.
#[derive(Debug, Clone)]
pub struct ConcurrencyLimiter {
    /// Number of currently active (acquired) slots.
    pub active: usize,
    /// Maximum permitted concurrent slots.
    pub max_concurrent: usize,
}

impl ConcurrencyLimiter {
    /// Create a new limiter.
    #[must_use]
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            active: 0,
            max_concurrent,
        }
    }

    /// Attempt to acquire a concurrency slot.
    ///
    /// Returns `true` on success. The caller **must** call [`release`](Self::release)
    /// when the work item finishes.
    pub fn try_acquire(&mut self) -> bool {
        if self.active < self.max_concurrent {
            self.active += 1;
            true
        } else {
            false
        }
    }

    /// Release a previously acquired slot.
    pub fn release(&mut self) {
        self.active = self.active.saturating_sub(1);
    }

    /// Current utilisation as a fraction in `[0.0, 1.0]`.
    ///
    /// Returns `0.0` when `max_concurrent` is zero.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn utilization(&self) -> f64 {
        if self.max_concurrent == 0 {
            return 0.0;
        }
        self.active as f64 / self.max_concurrent as f64
    }
}

/// Accumulated statistics for a throttle session.
#[derive(Debug, Clone, Default)]
pub struct ThrottleStats {
    /// Number of requests that were allowed through.
    pub accepted: u64,
    /// Number of requests that were rejected (bucket empty).
    pub rejected: u64,
    /// Total milliseconds spent waiting across all accepted requests.
    pub total_wait_ms: u64,
}

impl ThrottleStats {
    /// Fraction of requests that were rejected (`[0.0, 1.0]`).
    ///
    /// Returns `0.0` when no requests have been made.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn rejection_rate(&self) -> f64 {
        let total = self.accepted + self.rejected;
        if total == 0 {
            return 0.0;
        }
        self.rejected as f64 / total as f64
    }

    /// Average wait time per accepted request in milliseconds.
    ///
    /// Returns `0.0` when no requests have been accepted.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn avg_wait_ms(&self) -> f64 {
        if self.accepted == 0 {
            return 0.0;
        }
        self.total_wait_ms as f64 / self.accepted as f64
    }
}

// ---------------------------------------------------------------------------
// Per-user rate limiting
// ---------------------------------------------------------------------------

/// Configuration for per-user submission quotas and concurrency caps.
#[derive(Debug, Clone)]
pub struct UserLimitConfig {
    /// Maximum number of jobs a single user may have concurrently active.
    pub max_concurrent_jobs: usize,
    /// Maximum number of job submissions allowed within `window_secs`.
    pub max_submissions_per_window: usize,
    /// Length of the sliding submission-count window in seconds.
    pub window_secs: u64,
}

impl UserLimitConfig {
    /// Lenient defaults: 10 concurrent jobs, 100 submissions per minute.
    #[must_use]
    pub fn lenient() -> Self {
        Self {
            max_concurrent_jobs: 10,
            max_submissions_per_window: 100,
            window_secs: 60,
        }
    }

    /// Strict defaults: 2 concurrent jobs, 10 submissions per minute.
    #[must_use]
    pub fn strict() -> Self {
        Self {
            max_concurrent_jobs: 2,
            max_submissions_per_window: 10,
            window_secs: 60,
        }
    }
}

/// Internal per-user accounting state.
#[derive(Debug, Default)]
struct UserState {
    /// Number of currently active (in-progress) jobs for this user.
    active_jobs: usize,
    /// Timestamps (in seconds) of recent submissions within the current window.
    submission_times: Vec<u64>,
}

impl UserState {
    /// Purge submission timestamps that have fallen outside `window_secs`.
    fn evict_stale(&mut self, now_secs: u64, window_secs: u64) {
        let cutoff = now_secs.saturating_sub(window_secs);
        self.submission_times.retain(|&t| t >= cutoff);
    }
}

/// Enforces per-user job submission quotas and concurrency limits.
///
/// All state is protected by a `parking_lot::Mutex` so this type is `Send +
/// Sync` and can be shared across async tasks without additional wrapping.
pub struct UserRateLimiter {
    config: UserLimitConfig,
    users: Mutex<HashMap<String, UserState>>,
}

impl UserRateLimiter {
    /// Create a new limiter with the given configuration.
    #[must_use]
    pub fn new(config: UserLimitConfig) -> Self {
        Self {
            config,
            users: Mutex::new(HashMap::new()),
        }
    }

    /// Attempt to record a new job submission for `user_id` at `now_secs`.
    ///
    /// Returns `true` when both limits are satisfied and the submission is
    /// recorded.  Returns `false` when either the per-window submission quota
    /// or the concurrency cap would be exceeded.
    pub fn try_submit_at(&self, user_id: &str, now_secs: u64) -> bool {
        let mut guard = self.users.lock();
        let state = guard.entry(user_id.to_string()).or_default();

        // Evict stale entries from the sliding window.
        state.evict_stale(now_secs, self.config.window_secs);

        // Check both limits before recording.
        if state.submission_times.len() >= self.config.max_submissions_per_window {
            return false;
        }
        if state.active_jobs >= self.config.max_concurrent_jobs {
            return false;
        }

        state.submission_times.push(now_secs);
        state.active_jobs += 1;
        true
    }

    /// Signal that one active job for `user_id` has finished (success or failure).
    ///
    /// Silently does nothing when called for an unknown user or when `active_jobs`
    /// is already zero (idempotent / safe to call from error paths).
    pub fn release(&self, user_id: &str) {
        let mut guard = self.users.lock();
        if let Some(state) = guard.get_mut(user_id) {
            state.active_jobs = state.active_jobs.saturating_sub(1);
        }
    }

    /// Return the number of currently active jobs for `user_id`.
    ///
    /// Returns `0` for unknown users.
    #[must_use]
    pub fn active_jobs(&self, user_id: &str) -> usize {
        self.users.lock().get(user_id).map_or(0, |s| s.active_jobs)
    }

    /// Return the number of submissions recorded for `user_id` in the current window.
    ///
    /// The window is evaluated relative to `now_secs`.
    #[must_use]
    pub fn window_submissions(&self, user_id: &str) -> usize {
        self.users
            .lock()
            .get(user_id)
            .map_or(0, |s| s.submission_times.len())
    }

    /// Remove all recorded state for `user_id`.  Primarily useful in tests.
    pub fn reset_user(&self, user_id: &str) {
        self.users.lock().remove(user_id);
    }

    /// Return the number of distinct users currently tracked.
    #[must_use]
    pub fn tracked_user_count(&self) -> usize {
        self.users.lock().len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---------- ThrottleConfig ----------

    #[test]
    fn test_throttle_config_default_fields() {
        let cfg = ThrottleConfig::default();
        assert_eq!(cfg.max_concurrent, 8);
        assert!(cfg.rate_per_second > 0.0);
        assert_eq!(cfg.burst_allowance, 5);
    }

    #[test]
    fn test_throttle_config_conservative_more_restrictive() {
        let c = ThrottleConfig::conservative();
        let d = ThrottleConfig::default();
        assert!(c.max_concurrent < d.max_concurrent);
        assert!(c.rate_per_second < d.rate_per_second);
    }

    // ---------- TokenBucketThrottle ----------

    #[test]
    fn test_token_bucket_starts_full() {
        let tb = TokenBucketThrottle::new(10.0, 1.0, 0);
        assert!((tb.available_tokens() - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_token_bucket_consume_decrements() {
        let mut tb = TokenBucketThrottle::new(10.0, 0.0, 0);
        assert!(tb.try_consume(0));
        assert!((tb.available_tokens() - 9.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_token_bucket_empty_rejects() {
        let mut tb = TokenBucketThrottle::new(1.0, 0.0, 0);
        assert!(tb.try_consume(0));
        assert!(!tb.try_consume(0)); // bucket now empty
    }

    #[test]
    fn test_token_bucket_refill_over_time() {
        // 1 token/second = 1 token/1000 ms
        let mut tb = TokenBucketThrottle::new(10.0, 1.0, 0);
        // Drain the bucket
        for _ in 0..10 {
            tb.try_consume(0);
        }
        assert!(tb.available_tokens() < 1.0);
        // Advance 2000 ms → 2 new tokens
        tb.refill(2_000);
        assert!(tb.available_tokens() >= 2.0);
    }

    #[test]
    fn test_token_bucket_does_not_exceed_max() {
        let mut tb = TokenBucketThrottle::new(5.0, 100.0, 0);
        tb.refill(10_000); // would give 1_000_000 tokens without the cap
        assert!((tb.available_tokens() - 5.0).abs() < f64::EPSILON);
    }

    // ---------- ConcurrencyLimiter ----------

    #[test]
    fn test_concurrency_limiter_acquire_success() {
        let mut cl = ConcurrencyLimiter::new(3);
        assert!(cl.try_acquire());
        assert_eq!(cl.active, 1);
    }

    #[test]
    fn test_concurrency_limiter_blocks_at_max() {
        let mut cl = ConcurrencyLimiter::new(2);
        cl.try_acquire();
        cl.try_acquire();
        assert!(!cl.try_acquire());
    }

    #[test]
    fn test_concurrency_limiter_release() {
        let mut cl = ConcurrencyLimiter::new(1);
        cl.try_acquire();
        cl.release();
        assert!(cl.try_acquire()); // slot is free again
    }

    #[test]
    fn test_concurrency_limiter_utilization() {
        let mut cl = ConcurrencyLimiter::new(4);
        cl.try_acquire();
        cl.try_acquire();
        let u = cl.utilization();
        assert!((u - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_concurrency_limiter_zero_max_utilization() {
        let cl = ConcurrencyLimiter::new(0);
        assert!((cl.utilization() - 0.0).abs() < f64::EPSILON);
    }

    // ---------- ThrottleStats ----------

    #[test]
    fn test_throttle_stats_rejection_rate_no_requests() {
        let s = ThrottleStats::default();
        assert!((s.rejection_rate() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_throttle_stats_rejection_rate_half() {
        let s = ThrottleStats {
            accepted: 5,
            rejected: 5,
            total_wait_ms: 0,
        };
        assert!((s.rejection_rate() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_throttle_stats_avg_wait_no_accepted() {
        let s = ThrottleStats {
            accepted: 0,
            rejected: 3,
            total_wait_ms: 100,
        };
        assert!((s.avg_wait_ms() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_throttle_stats_avg_wait() {
        let s = ThrottleStats {
            accepted: 4,
            rejected: 0,
            total_wait_ms: 200,
        };
        assert!((s.avg_wait_ms() - 50.0).abs() < f64::EPSILON);
    }

    // ---------- UserRateLimiter ----------

    fn make_limiter(
        max_concurrent: usize,
        max_per_window: usize,
        window_secs: u64,
    ) -> UserRateLimiter {
        UserRateLimiter::new(UserLimitConfig {
            max_concurrent_jobs: max_concurrent,
            max_submissions_per_window: max_per_window,
            window_secs,
        })
    }

    #[test]
    fn test_user_rate_limiter_first_submission_allowed() {
        let lim = make_limiter(5, 10, 60);
        assert!(lim.try_submit_at("alice", 0));
        assert_eq!(lim.active_jobs("alice"), 1);
        assert_eq!(lim.window_submissions("alice"), 1);
    }

    #[test]
    fn test_user_rate_limiter_concurrent_cap_blocks() {
        let lim = make_limiter(2, 100, 60);
        assert!(lim.try_submit_at("bob", 0));
        assert!(lim.try_submit_at("bob", 0));
        // Third submission exceeds max_concurrent_jobs=2
        assert!(!lim.try_submit_at("bob", 0));
    }

    #[test]
    fn test_user_rate_limiter_window_quota_blocks() {
        let lim = make_limiter(100, 3, 60);
        assert!(lim.try_submit_at("carol", 0));
        assert!(lim.try_submit_at("carol", 0));
        assert!(lim.try_submit_at("carol", 0));
        // Fourth exceeds max_submissions_per_window=3
        assert!(!lim.try_submit_at("carol", 0));
    }

    #[test]
    fn test_user_rate_limiter_release_allows_new_submission() {
        let lim = make_limiter(1, 100, 60);
        assert!(lim.try_submit_at("dave", 0));
        assert!(!lim.try_submit_at("dave", 0)); // at cap
        lim.release("dave");
        assert!(lim.try_submit_at("dave", 1)); // slot freed
    }

    #[test]
    fn test_user_rate_limiter_sliding_window_evicts_old() {
        // Window of 10 seconds, quota of 2 per window
        let lim = make_limiter(100, 2, 10);
        assert!(lim.try_submit_at("eve", 0));
        assert!(lim.try_submit_at("eve", 0));
        // At quota inside the window
        assert!(!lim.try_submit_at("eve", 5));
        // Advance past the window — old entries should be evicted
        assert!(lim.try_submit_at("eve", 15));
    }

    #[test]
    fn test_user_rate_limiter_release_unknown_user_is_safe() {
        let lim = make_limiter(5, 10, 60);
        // Releasing a user that was never tracked must not panic or error
        lim.release("unknown");
        assert_eq!(lim.active_jobs("unknown"), 0);
    }

    #[test]
    fn test_user_rate_limiter_active_jobs_unknown_user_is_zero() {
        let lim = make_limiter(5, 10, 60);
        assert_eq!(lim.active_jobs("nobody"), 0);
    }

    #[test]
    fn test_user_rate_limiter_window_submissions_unknown_user_is_zero() {
        let lim = make_limiter(5, 10, 60);
        assert_eq!(lim.window_submissions("nobody"), 0);
    }

    #[test]
    fn test_user_rate_limiter_reset_clears_state() {
        let lim = make_limiter(5, 10, 60);
        lim.try_submit_at("frank", 0);
        assert_eq!(lim.active_jobs("frank"), 1);
        lim.reset_user("frank");
        assert_eq!(lim.active_jobs("frank"), 0);
        assert_eq!(lim.tracked_user_count(), 0);
    }

    #[test]
    fn test_user_rate_limiter_tracked_user_count() {
        let lim = make_limiter(5, 10, 60);
        assert_eq!(lim.tracked_user_count(), 0);
        lim.try_submit_at("grace", 0);
        lim.try_submit_at("henry", 0);
        assert_eq!(lim.tracked_user_count(), 2);
    }

    #[test]
    fn test_user_rate_limiter_independent_users() {
        let lim = make_limiter(1, 100, 60);
        // Different users should not interfere with each other's concurrency cap
        assert!(lim.try_submit_at("user_a", 0));
        assert!(!lim.try_submit_at("user_a", 0)); // user_a at cap
        assert!(lim.try_submit_at("user_b", 0)); // user_b unaffected
    }

    #[test]
    fn test_user_limit_config_lenient_gt_strict() {
        let lenient = UserLimitConfig::lenient();
        let strict = UserLimitConfig::strict();
        assert!(lenient.max_concurrent_jobs > strict.max_concurrent_jobs);
        assert!(lenient.max_submissions_per_window > strict.max_submissions_per_window);
    }
}
