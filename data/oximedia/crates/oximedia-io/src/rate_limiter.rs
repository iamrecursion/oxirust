//! I/O rate limiting for bandwidth control.
//!
//! Provides a token-bucket rate limiter, a per-stream rate limiting manager,
//! a sliding-window bandwidth tracker, and a bidirectional rate limiter with
//! separate read and write bandwidth limits.

#![allow(dead_code)]

use std::collections::HashMap;

// ──────────────────────────────────────────────────────────────────────────────
// RateLimit
// ──────────────────────────────────────────────────────────────────────────────

/// Describes the rate limit for a single I/O stream
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RateLimit {
    /// Sustained rate in bytes per second
    pub bytes_per_sec: u64,
    /// Maximum burst size in bytes (tokens that can accumulate)
    pub burst_bytes: u64,
}

impl RateLimit {
    /// Create a simple limit with no extra burst (burst == `bytes_per_sec`)
    #[must_use]
    pub fn new(bytes_per_sec: u64) -> Self {
        Self {
            bytes_per_sec,
            burst_bytes: bytes_per_sec,
        }
    }

    /// Create a limit with an explicit burst allowance
    #[must_use]
    pub fn with_burst(bytes_per_sec: u64, burst_bytes: u64) -> Self {
        Self {
            bytes_per_sec,
            burst_bytes,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// TokenBucket
// ──────────────────────────────────────────────────────────────────────────────

/// Token-bucket rate limiter
///
/// Tokens accumulate at `refill_rate` tokens/ms up to `capacity`.
/// Each byte consumed removes one token.
#[derive(Debug, Clone)]
pub struct TokenBucket {
    /// Maximum number of tokens (== `burst_bytes`)
    pub capacity: f64,
    /// Current token count
    pub tokens: f64,
    /// Token refill rate in tokens per millisecond
    pub refill_rate: f64,
    /// Timestamp (ms) of the last refill
    pub last_refill_ms: u64,
}

impl TokenBucket {
    /// Create a new bucket from a `RateLimit`.
    ///
    /// `now_ms` should be the current time in milliseconds (e.g. from a
    /// monotonic clock or test fixture).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn new(limit: RateLimit, now_ms: u64) -> Self {
        let refill_rate = limit.bytes_per_sec as f64 / 1000.0; // per ms
        Self {
            capacity: limit.burst_bytes as f64,
            tokens: limit.burst_bytes as f64, // start full
            refill_rate,
            last_refill_ms: now_ms,
        }
    }

    /// Attempt to consume `bytes` tokens at time `now_ms`.
    ///
    /// Refills the bucket first, then checks availability.
    /// Returns `true` if the tokens were consumed, `false` if insufficient.
    #[allow(clippy::cast_precision_loss)]
    pub fn try_consume(&mut self, bytes: u64, now_ms: u64) -> bool {
        // Refill
        if now_ms > self.last_refill_ms {
            let elapsed_ms = (now_ms - self.last_refill_ms) as f64;
            self.tokens = (self.tokens + elapsed_ms * self.refill_rate).min(self.capacity);
            self.last_refill_ms = now_ms;
        }

        if self.tokens >= bytes as f64 {
            self.tokens -= bytes as f64;
            true
        } else {
            false
        }
    }

    /// Estimate how many milliseconds until `bytes` tokens are available
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub fn wait_ms_for(&self, bytes: u64) -> u64 {
        let needed = bytes as f64 - self.tokens;
        if needed <= 0.0 || self.refill_rate <= 0.0 {
            return 0;
        }
        (needed / self.refill_rate).ceil() as u64
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// RateLimitResult
// ──────────────────────────────────────────────────────────────────────────────

/// Result returned by `RateLimiter::check_and_consume`
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RateLimitResult {
    /// The bytes were consumed; the caller may proceed immediately
    Allowed,
    /// The bucket is insufficient; caller should wait `wait_ms` milliseconds
    Throttled(u64),
}

impl RateLimitResult {
    /// Returns `true` if the operation is allowed
    #[must_use]
    pub fn is_allowed(&self) -> bool {
        matches!(self, Self::Allowed)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// RateLimiter
// ──────────────────────────────────────────────────────────────────────────────

/// Per-stream rate limiter managing multiple `TokenBucket`s
pub struct RateLimiter {
    buckets: HashMap<String, TokenBucket>,
}

impl RateLimiter {
    /// Create a new, empty rate limiter
    #[must_use]
    pub fn new() -> Self {
        Self {
            buckets: HashMap::new(),
        }
    }

    /// Add or replace a rate limit for the stream identified by `id`
    pub fn add_stream(&mut self, id: &str, limit: RateLimit, now_ms: u64) {
        let bucket = TokenBucket::new(limit, now_ms);
        self.buckets.insert(id.to_string(), bucket);
    }

    /// Attempt to consume `bytes` from the bucket for stream `id`.
    ///
    /// Returns `Throttled(wait_ms)` if the bucket is insufficient, or
    /// `Allowed` if the bytes were successfully consumed.
    ///
    /// If no bucket is registered for `id`, this always returns `Allowed`.
    pub fn check_and_consume(&mut self, id: &str, bytes: u64, now_ms: u64) -> RateLimitResult {
        let Some(bucket) = self.buckets.get_mut(id) else {
            return RateLimitResult::Allowed;
        };

        if bucket.try_consume(bytes, now_ms) {
            RateLimitResult::Allowed
        } else {
            let wait_ms = bucket.wait_ms_for(bytes);
            RateLimitResult::Throttled(wait_ms)
        }
    }

    /// Remove the rate limit for the stream identified by `id`
    pub fn remove_stream(&mut self, id: &str) {
        self.buckets.remove(id);
    }

    /// Returns the number of registered streams
    #[must_use]
    pub fn stream_count(&self) -> usize {
        self.buckets.len()
    }
}

impl Default for RateLimiter {
    fn default() -> Self {
        Self::new()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// BandwidthTracker — sliding window
// ──────────────────────────────────────────────────────────────────────────────

/// Tracks bandwidth with a sliding time window
///
/// Observations older than `window_ms` milliseconds are discarded when
/// querying `current_bps`.
pub struct BandwidthTracker {
    /// Sliding window duration in milliseconds
    window_ms: u64,
    /// Observations stored as (`timestamp_ms`, bytes)
    observations: Vec<(u64, u64)>,
}

impl BandwidthTracker {
    /// Create a tracker with the given sliding window duration
    #[must_use]
    pub fn new(window_ms: u64) -> Self {
        Self {
            window_ms,
            observations: Vec::new(),
        }
    }

    /// Record that `bytes` were transferred at time `now_ms`
    pub fn record(&mut self, bytes: u64, now_ms: u64) {
        self.observations.push((now_ms, bytes));
        // Prune old observations eagerly to avoid unbounded growth
        self.prune(now_ms);
    }

    /// Compute the current bandwidth in bytes per second.
    ///
    /// Returns 0 if no observations are within the window.
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub fn current_bps(&self, now_ms: u64) -> u64 {
        let cutoff = now_ms.saturating_sub(self.window_ms);
        let total_bytes: u64 = self
            .observations
            .iter()
            .filter(|(ts, _)| *ts >= cutoff)
            .map(|(_, bytes)| bytes)
            .sum();

        if self.window_ms == 0 {
            return 0;
        }

        // Convert window from ms to seconds for bytes/sec calculation
        let window_sec = self.window_ms as f64 / 1000.0;
        (total_bytes as f64 / window_sec) as u64
    }

    /// Total bytes recorded in the current window
    #[must_use]
    pub fn total_bytes_in_window(&self, now_ms: u64) -> u64 {
        let cutoff = now_ms.saturating_sub(self.window_ms);
        self.observations
            .iter()
            .filter(|(ts, _)| *ts >= cutoff)
            .map(|(_, bytes)| bytes)
            .sum()
    }

    fn prune(&mut self, now_ms: u64) {
        let cutoff = now_ms.saturating_sub(self.window_ms);
        self.observations.retain(|(ts, _)| *ts >= cutoff);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// IoDirection — read vs write
// ──────────────────────────────────────────────────────────────────────────────

/// Direction of an I/O operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IoDirection {
    /// Data flowing from source to consumer (read).
    Read,
    /// Data flowing from producer to sink (write).
    Write,
}

// ──────────────────────────────────────────────────────────────────────────────
// DirectionalLimits — separate read/write rate limits
// ──────────────────────────────────────────────────────────────────────────────

/// Independent bandwidth limits for read and write directions.
#[derive(Debug, Clone, Copy)]
pub struct DirectionalLimits {
    /// Maximum sustained bandwidth for read operations (bytes per second).
    pub read_bytes_per_sec: u64,
    /// Maximum burst for read operations (bytes).
    pub read_burst_bytes: u64,
    /// Maximum sustained bandwidth for write operations (bytes per second).
    pub write_bytes_per_sec: u64,
    /// Maximum burst for write operations (bytes).
    pub write_burst_bytes: u64,
}

impl DirectionalLimits {
    /// Create symmetric limits where read and write have the same rate.
    #[must_use]
    pub fn symmetric(bytes_per_sec: u64) -> Self {
        Self {
            read_bytes_per_sec: bytes_per_sec,
            read_burst_bytes: bytes_per_sec,
            write_bytes_per_sec: bytes_per_sec,
            write_burst_bytes: bytes_per_sec,
        }
    }

    /// Create asymmetric limits (common in consumer networking: slow upload, fast download).
    #[must_use]
    pub fn asymmetric(read_bps: u64, write_bps: u64) -> Self {
        Self {
            read_bytes_per_sec: read_bps,
            read_burst_bytes: read_bps,
            write_bytes_per_sec: write_bps,
            write_burst_bytes: write_bps,
        }
    }

    /// Set burst sizes explicitly.
    #[must_use]
    pub fn with_burst(mut self, read_burst: u64, write_burst: u64) -> Self {
        self.read_burst_bytes = read_burst;
        self.write_burst_bytes = write_burst;
        self
    }

    /// Convert to a `RateLimit` for the given direction.
    #[must_use]
    pub fn for_direction(&self, direction: IoDirection) -> RateLimit {
        match direction {
            IoDirection::Read => {
                RateLimit::with_burst(self.read_bytes_per_sec, self.read_burst_bytes)
            }
            IoDirection::Write => {
                RateLimit::with_burst(self.write_bytes_per_sec, self.write_burst_bytes)
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// DirectionalRateLimiter
// ──────────────────────────────────────────────────────────────────────────────

/// A rate limiter that maintains separate token buckets for read and write.
///
/// Useful for I/O layers where upload and download bandwidth must be
/// controlled independently (e.g. network media streaming, cloud storage).
pub struct DirectionalRateLimiter {
    read_bucket: TokenBucket,
    write_bucket: TokenBucket,
    config: DirectionalLimits,
}

impl DirectionalRateLimiter {
    /// Create a new directional limiter with the given limits.
    ///
    /// `now_ms` is the current time in milliseconds.
    #[must_use]
    pub fn new(limits: DirectionalLimits, now_ms: u64) -> Self {
        let read_limit = limits.for_direction(IoDirection::Read);
        let write_limit = limits.for_direction(IoDirection::Write);
        Self {
            read_bucket: TokenBucket::new(read_limit, now_ms),
            write_bucket: TokenBucket::new(write_limit, now_ms),
            config: limits,
        }
    }

    /// Attempt to consume `bytes` from the appropriate bucket for `direction`.
    ///
    /// Returns `Allowed` if tokens were consumed, `Throttled(wait_ms)` otherwise.
    pub fn check_and_consume(
        &mut self,
        direction: IoDirection,
        bytes: u64,
        now_ms: u64,
    ) -> RateLimitResult {
        let bucket = match direction {
            IoDirection::Read => &mut self.read_bucket,
            IoDirection::Write => &mut self.write_bucket,
        };
        if bucket.try_consume(bytes, now_ms) {
            RateLimitResult::Allowed
        } else {
            let wait_ms = bucket.wait_ms_for(bytes);
            RateLimitResult::Throttled(wait_ms)
        }
    }

    /// Return an estimate of how long until `bytes` tokens are available in `direction`.
    #[must_use]
    pub fn wait_ms_for(&self, direction: IoDirection, bytes: u64) -> u64 {
        match direction {
            IoDirection::Read => self.read_bucket.wait_ms_for(bytes),
            IoDirection::Write => self.write_bucket.wait_ms_for(bytes),
        }
    }

    /// Return the configured directional limits.
    #[must_use]
    pub fn config(&self) -> &DirectionalLimits {
        &self.config
    }

    /// Return current token count for the given direction.
    #[must_use]
    pub fn available_tokens(&self, direction: IoDirection) -> f64 {
        match direction {
            IoDirection::Read => self.read_bucket.tokens,
            IoDirection::Write => self.write_bucket.tokens,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// DirectionalBandwidthTracker
// ──────────────────────────────────────────────────────────────────────────────

/// Tracks separate read and write bandwidth with sliding time windows.
pub struct DirectionalBandwidthTracker {
    read_tracker: BandwidthTracker,
    write_tracker: BandwidthTracker,
}

impl DirectionalBandwidthTracker {
    /// Create a new tracker with the given window duration for both directions.
    #[must_use]
    pub fn new(window_ms: u64) -> Self {
        Self {
            read_tracker: BandwidthTracker::new(window_ms),
            write_tracker: BandwidthTracker::new(window_ms),
        }
    }

    /// Record a transfer of `bytes` in `direction` at time `now_ms`.
    pub fn record(&mut self, direction: IoDirection, bytes: u64, now_ms: u64) {
        match direction {
            IoDirection::Read => self.read_tracker.record(bytes, now_ms),
            IoDirection::Write => self.write_tracker.record(bytes, now_ms),
        }
    }

    /// Current bandwidth in bytes per second for the given direction.
    #[must_use]
    pub fn current_bps(&self, direction: IoDirection, now_ms: u64) -> u64 {
        match direction {
            IoDirection::Read => self.read_tracker.current_bps(now_ms),
            IoDirection::Write => self.write_tracker.current_bps(now_ms),
        }
    }

    /// Total bytes transferred in the window for the given direction.
    #[must_use]
    pub fn total_bytes_in_window(&self, direction: IoDirection, now_ms: u64) -> u64 {
        match direction {
            IoDirection::Read => self.read_tracker.total_bytes_in_window(now_ms),
            IoDirection::Write => self.write_tracker.total_bytes_in_window(now_ms),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── TokenBucket ────────────────────────────────────────────────────────────

    #[test]
    fn test_token_bucket_starts_full() {
        let limit = RateLimit::with_burst(1000, 5000);
        let bucket = TokenBucket::new(limit, 0);
        assert!((bucket.tokens - 5000.0).abs() < f64::EPSILON);
        assert!((bucket.capacity - 5000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_token_bucket_consume_success() {
        let limit = RateLimit::new(1_000_000);
        let mut bucket = TokenBucket::new(limit, 0);
        assert!(bucket.try_consume(500_000, 0));
        assert!((bucket.tokens - 500_000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_token_bucket_consume_fail_insufficient() {
        let limit = RateLimit::new(1000);
        let mut bucket = TokenBucket::new(limit, 0);
        // Immediately consume everything first
        bucket.try_consume(1000, 0);
        // No tokens left, should fail
        assert!(!bucket.try_consume(1, 0));
    }

    #[test]
    fn test_token_bucket_refill_over_time() {
        let limit = RateLimit::new(1000); // 1 token/ms
        let mut bucket = TokenBucket::new(limit, 0);
        bucket.try_consume(1000, 0); // drain
                                     // After 500 ms, 500 tokens should have refilled
        assert!(bucket.try_consume(500, 500));
    }

    #[test]
    fn test_token_bucket_capped_at_capacity() {
        let limit = RateLimit::new(100);
        let mut bucket = TokenBucket::new(limit, 0);
        // Pass 100 seconds — tokens should NOT exceed capacity
        bucket.try_consume(0, 100_000);
        assert!(bucket.tokens <= bucket.capacity + f64::EPSILON);
    }

    #[test]
    fn test_token_bucket_wait_ms() {
        let limit = RateLimit::new(1000); // 1 token/ms
        let mut bucket = TokenBucket::new(limit, 0);
        bucket.try_consume(1000, 0); // drain
        let wait = bucket.wait_ms_for(500);
        assert!(wait >= 500);
    }

    // ── RateLimitResult ────────────────────────────────────────────────────────

    #[test]
    fn test_rate_limit_result_is_allowed() {
        assert!(RateLimitResult::Allowed.is_allowed());
        assert!(!RateLimitResult::Throttled(100).is_allowed());
    }

    // ── RateLimiter ────────────────────────────────────────────────────────────

    #[test]
    fn test_rate_limiter_no_stream_always_allowed() {
        let mut rl = RateLimiter::new();
        assert_eq!(
            rl.check_and_consume("unknown", 1_000_000, 0),
            RateLimitResult::Allowed
        );
    }

    #[test]
    fn test_rate_limiter_stream_allowed() {
        let mut rl = RateLimiter::new();
        rl.add_stream("s1", RateLimit::new(10_000), 0);
        assert_eq!(
            rl.check_and_consume("s1", 5_000, 0),
            RateLimitResult::Allowed
        );
    }

    #[test]
    fn test_rate_limiter_stream_throttled() {
        let mut rl = RateLimiter::new();
        rl.add_stream("s2", RateLimit::new(100), 0);
        rl.check_and_consume("s2", 100, 0); // drain
        let result = rl.check_and_consume("s2", 50, 0);
        assert!(matches!(result, RateLimitResult::Throttled(_)));
    }

    #[test]
    fn test_rate_limiter_remove_stream() {
        let mut rl = RateLimiter::new();
        rl.add_stream("s3", RateLimit::new(100), 0);
        rl.remove_stream("s3");
        assert_eq!(rl.stream_count(), 0);
        // Now the stream is gone; should be allowed
        assert_eq!(
            rl.check_and_consume("s3", 99999, 0),
            RateLimitResult::Allowed
        );
    }

    // ── BandwidthTracker ───────────────────────────────────────────────────────

    #[test]
    fn test_bandwidth_tracker_empty() {
        let tracker = BandwidthTracker::new(1000);
        assert_eq!(tracker.current_bps(0), 0);
    }

    #[test]
    fn test_bandwidth_tracker_single_observation() {
        let mut tracker = BandwidthTracker::new(1000); // 1-second window
        tracker.record(500_000, 500); // 500 KB at t=500ms
                                      // 500 KB / 1 s = 500_000 bps
        assert_eq!(tracker.current_bps(1000), 500_000);
    }

    #[test]
    fn test_bandwidth_tracker_old_observations_pruned() {
        let mut tracker = BandwidthTracker::new(1000);
        tracker.record(1_000_000, 0); // at t=0
                                      // At t=2000, the window covers [1000, 2000]; observation at 0 is outside
        assert_eq!(tracker.current_bps(2000), 0);
    }

    #[test]
    fn test_bandwidth_tracker_total_bytes_in_window() {
        let mut tracker = BandwidthTracker::new(2000);
        tracker.record(100, 0);
        tracker.record(200, 1000);
        tracker.record(400, 2000);
        // At now=2000 window covers [0, 2000]; all three are included
        assert_eq!(tracker.total_bytes_in_window(2000), 700);
    }

    // ── DirectionalLimits ─────────────────────────────────────────────────────

    #[test]
    fn test_directional_limits_symmetric() {
        let limits = DirectionalLimits::symmetric(1_000_000);
        assert_eq!(limits.read_bytes_per_sec, 1_000_000);
        assert_eq!(limits.write_bytes_per_sec, 1_000_000);
        assert_eq!(limits.read_burst_bytes, 1_000_000);
        assert_eq!(limits.write_burst_bytes, 1_000_000);
    }

    #[test]
    fn test_directional_limits_asymmetric() {
        let limits = DirectionalLimits::asymmetric(10_000_000, 1_000_000);
        assert_eq!(limits.read_bytes_per_sec, 10_000_000);
        assert_eq!(limits.write_bytes_per_sec, 1_000_000);
    }

    #[test]
    fn test_directional_limits_for_direction() {
        let limits = DirectionalLimits::asymmetric(2000, 1000);
        let read_limit = limits.for_direction(IoDirection::Read);
        assert_eq!(read_limit.bytes_per_sec, 2000);
        let write_limit = limits.for_direction(IoDirection::Write);
        assert_eq!(write_limit.bytes_per_sec, 1000);
    }

    // ── DirectionalRateLimiter ────────────────────────────────────────────────

    #[test]
    fn test_directional_rate_limiter_read_allowed() {
        let limits = DirectionalLimits::asymmetric(10_000, 1_000);
        let mut rl = DirectionalRateLimiter::new(limits, 0);
        assert_eq!(
            rl.check_and_consume(IoDirection::Read, 5_000, 0),
            RateLimitResult::Allowed
        );
    }

    #[test]
    fn test_directional_rate_limiter_write_throttled() {
        let limits = DirectionalLimits::asymmetric(10_000, 500);
        let mut rl = DirectionalRateLimiter::new(limits, 0);
        // Drain write bucket
        rl.check_and_consume(IoDirection::Write, 500, 0);
        let result = rl.check_and_consume(IoDirection::Write, 100, 0);
        assert!(matches!(result, RateLimitResult::Throttled(_)));
    }

    #[test]
    fn test_directional_rate_limiter_read_and_write_independent() {
        let limits = DirectionalLimits::asymmetric(10_000, 500);
        let mut rl = DirectionalRateLimiter::new(limits, 0);
        // Drain write bucket completely
        rl.check_and_consume(IoDirection::Write, 500, 0);
        // Write is throttled
        assert!(matches!(
            rl.check_and_consume(IoDirection::Write, 1, 0),
            RateLimitResult::Throttled(_)
        ));
        // Read is still allowed (independent bucket, 10_000 burst)
        assert_eq!(
            rl.check_and_consume(IoDirection::Read, 5_000, 0),
            RateLimitResult::Allowed
        );
    }

    #[test]
    fn test_directional_rate_limiter_wait_ms() {
        let limits = DirectionalLimits::symmetric(1000); // 1 token/ms
        let mut rl = DirectionalRateLimiter::new(limits, 0);
        rl.check_and_consume(IoDirection::Write, 1000, 0); // drain write
        let wait = rl.wait_ms_for(IoDirection::Write, 500);
        assert!(wait >= 500);
        // Read has full tokens
        assert_eq!(rl.wait_ms_for(IoDirection::Read, 500), 0);
    }

    #[test]
    fn test_directional_rate_limiter_available_tokens() {
        let limits = DirectionalLimits::symmetric(1000);
        let rl = DirectionalRateLimiter::new(limits, 0);
        // Both start full
        assert!((rl.available_tokens(IoDirection::Read) - 1000.0).abs() < f64::EPSILON);
        assert!((rl.available_tokens(IoDirection::Write) - 1000.0).abs() < f64::EPSILON);
    }

    // ── DirectionalBandwidthTracker ───────────────────────────────────────────

    #[test]
    fn test_directional_bandwidth_tracker_separate() {
        let mut tracker = DirectionalBandwidthTracker::new(1000);
        tracker.record(IoDirection::Read, 500_000, 500);
        tracker.record(IoDirection::Write, 100_000, 500);
        // Read: 500 KB / 1 s
        assert_eq!(tracker.current_bps(IoDirection::Read, 1000), 500_000);
        // Write: 100 KB / 1 s
        assert_eq!(tracker.current_bps(IoDirection::Write, 1000), 100_000);
    }

    #[test]
    fn test_directional_bandwidth_tracker_total_bytes() {
        let mut tracker = DirectionalBandwidthTracker::new(2000);
        tracker.record(IoDirection::Read, 100, 0);
        tracker.record(IoDirection::Write, 200, 500);
        assert_eq!(tracker.total_bytes_in_window(IoDirection::Read, 2000), 100);
        assert_eq!(tracker.total_bytes_in_window(IoDirection::Write, 2000), 200);
    }
}
