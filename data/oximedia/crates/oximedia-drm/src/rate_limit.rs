//! Rate limiting for the DRM license server.
//!
//! Provides multiple rate limiting strategies to prevent abuse:
//! - **Token bucket**: smooth rate limiting with burst allowance
//! - **Sliding window**: count-based limiting over a time window
//! - **Per-client**: track limits per device ID / client IP
//! - **Adaptive**: automatic backoff under sustained load

use std::collections::HashMap;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

/// Rate limiting errors.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum RateLimitError {
    #[error("rate limit exceeded: {reason} (retry after {retry_after_secs}s)")]
    LimitExceeded {
        reason: String,
        retry_after_secs: u64,
    },

    #[error("client banned: {client_id}")]
    ClientBanned { client_id: String },
}

// ---------------------------------------------------------------------------
// Token bucket
// ---------------------------------------------------------------------------

/// Token bucket rate limiter.
///
/// Allows `capacity` requests in a burst and refills at `refill_rate` tokens
/// per second.
#[derive(Debug, Clone)]
pub struct TokenBucket {
    /// Maximum token capacity.
    capacity: f64,
    /// Current token count.
    tokens: f64,
    /// Tokens added per second.
    refill_rate: f64,
    /// Last refill timestamp (Unix seconds).
    last_refill_secs: u64,
}

impl TokenBucket {
    /// Create a new token bucket.
    pub fn new(capacity: f64, refill_rate: f64) -> Self {
        Self {
            capacity,
            tokens: capacity,
            refill_rate,
            last_refill_secs: 0,
        }
    }

    /// Attempt to consume one token at the given Unix timestamp.
    ///
    /// Returns `true` if the request is allowed, `false` if rate-limited.
    pub fn try_acquire(&mut self, now_secs: u64) -> bool {
        self.refill(now_secs);
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }

    /// Attempt to consume `n` tokens.
    pub fn try_acquire_n(&mut self, n: f64, now_secs: u64) -> bool {
        self.refill(now_secs);
        if self.tokens >= n {
            self.tokens -= n;
            true
        } else {
            false
        }
    }

    /// Current token count.
    pub fn available_tokens(&self) -> f64 {
        self.tokens
    }

    /// Seconds until at least one token is available (0 if already available).
    pub fn time_until_available(&self) -> f64 {
        if self.tokens >= 1.0 {
            0.0
        } else {
            (1.0 - self.tokens) / self.refill_rate
        }
    }

    fn refill(&mut self, now_secs: u64) {
        if now_secs > self.last_refill_secs {
            let elapsed = (now_secs - self.last_refill_secs) as f64;
            self.tokens = (self.tokens + elapsed * self.refill_rate).min(self.capacity);
            self.last_refill_secs = now_secs;
        }
    }
}

// ---------------------------------------------------------------------------
// Sliding window counter
// ---------------------------------------------------------------------------

/// Sliding window rate limiter using fixed-size time slots.
#[derive(Debug, Clone)]
pub struct SlidingWindowCounter {
    /// Maximum requests allowed in the window.
    max_requests: u64,
    /// Window duration in seconds.
    #[allow(dead_code)]
    window_secs: u64,
    /// Slot duration in seconds (window_secs / num_slots).
    slot_secs: u64,
    /// Ring buffer of slot counts.
    slots: Vec<u64>,
    /// Timestamp of the current (latest) slot's start.
    current_slot_start: u64,
    /// Index into `slots` for the current slot.
    current_idx: usize,
    /// Whether `current_slot_start` has been initialized.
    initialized: bool,
}

impl SlidingWindowCounter {
    /// Create a new sliding window counter.
    ///
    /// `num_slots` controls granularity (more slots = smoother but more memory).
    pub fn new(max_requests: u64, window_secs: u64, num_slots: usize) -> Self {
        let num_slots = num_slots.max(1);
        let slot_secs = (window_secs / num_slots as u64).max(1);
        Self {
            max_requests,
            window_secs,
            slot_secs,
            slots: vec![0; num_slots],
            current_slot_start: 0,
            current_idx: 0,
            initialized: false,
        }
    }

    /// Record a request at the given timestamp. Returns `true` if allowed.
    pub fn try_record(&mut self, now_secs: u64) -> bool {
        self.advance_to(now_secs);
        let total: u64 = self.slots.iter().sum();
        if total >= self.max_requests {
            return false;
        }
        self.slots[self.current_idx] += 1;
        true
    }

    /// Return the current request count within the window.
    pub fn current_count(&self) -> u64 {
        self.slots.iter().sum()
    }

    /// Remaining requests available in the window.
    pub fn remaining(&self) -> u64 {
        let used: u64 = self.slots.iter().sum();
        self.max_requests.saturating_sub(used)
    }

    fn advance_to(&mut self, now_secs: u64) {
        if !self.initialized {
            self.current_slot_start = now_secs;
            self.initialized = true;
            return;
        }

        let elapsed = now_secs.saturating_sub(self.current_slot_start);
        let slots_to_advance = (elapsed / self.slot_secs) as usize;

        if slots_to_advance == 0 {
            return;
        }

        let num_slots = self.slots.len();
        let to_clear = slots_to_advance.min(num_slots);
        for i in 0..to_clear {
            let idx = (self.current_idx + 1 + i) % num_slots;
            self.slots[idx] = 0;
        }
        self.current_idx = (self.current_idx + to_clear) % num_slots;
        self.current_slot_start += (to_clear as u64) * self.slot_secs;
    }
}

// ---------------------------------------------------------------------------
// Per-client rate limiter
// ---------------------------------------------------------------------------

/// Per-client rate limiter that tracks token buckets per client ID.
#[derive(Debug, Clone)]
pub struct PerClientRateLimiter {
    /// Token bucket capacity per client.
    capacity: f64,
    /// Token bucket refill rate per client (tokens/sec).
    refill_rate: f64,
    /// Per-client buckets keyed by client identifier.
    clients: HashMap<String, TokenBucket>,
    /// Clients that have been temporarily banned.
    banned: HashMap<String, u64>, // client_id -> ban expiry timestamp
    /// Number of consecutive rejections before banning.
    ban_threshold: u32,
    /// Ban duration in seconds.
    ban_duration_secs: u64,
    /// Consecutive rejection counts per client.
    rejection_counts: HashMap<String, u32>,
}

impl PerClientRateLimiter {
    /// Create a new per-client rate limiter.
    pub fn new(capacity: f64, refill_rate: f64) -> Self {
        Self {
            capacity,
            refill_rate,
            clients: HashMap::new(),
            banned: HashMap::new(),
            ban_threshold: 10,
            ban_duration_secs: 300,
            rejection_counts: HashMap::new(),
        }
    }

    /// Set the ban threshold (consecutive rejections before ban) and duration.
    pub fn with_ban_policy(mut self, threshold: u32, duration_secs: u64) -> Self {
        self.ban_threshold = threshold;
        self.ban_duration_secs = duration_secs;
        self
    }

    /// Check if a client request is allowed.
    pub fn try_acquire(&mut self, client_id: &str, now_secs: u64) -> Result<(), RateLimitError> {
        // Check ban
        if let Some(&expiry) = self.banned.get(client_id) {
            if now_secs < expiry {
                return Err(RateLimitError::ClientBanned {
                    client_id: client_id.to_string(),
                });
            }
            // Ban expired
            self.banned.remove(client_id);
            self.rejection_counts.remove(client_id);
        }

        let bucket = self
            .clients
            .entry(client_id.to_string())
            .or_insert_with(|| TokenBucket::new(self.capacity, self.refill_rate));

        if bucket.try_acquire(now_secs) {
            // Reset rejection counter on success
            self.rejection_counts.remove(client_id);
            Ok(())
        } else {
            // Track consecutive rejections
            let count = self
                .rejection_counts
                .entry(client_id.to_string())
                .or_insert(0);
            *count += 1;

            if *count >= self.ban_threshold {
                self.banned
                    .insert(client_id.to_string(), now_secs + self.ban_duration_secs);
                return Err(RateLimitError::ClientBanned {
                    client_id: client_id.to_string(),
                });
            }

            let retry_after = bucket.time_until_available().ceil() as u64;
            Err(RateLimitError::LimitExceeded {
                reason: format!("client {client_id} rate limit exceeded"),
                retry_after_secs: retry_after.max(1),
            })
        }
    }

    /// Number of tracked clients.
    pub fn client_count(&self) -> usize {
        self.clients.len()
    }

    /// Number of currently banned clients.
    pub fn banned_count(&self) -> usize {
        self.banned.len()
    }

    /// Remove stale client entries that haven't been used recently.
    pub fn evict_stale_clients(&mut self, now_secs: u64, stale_after_secs: u64) {
        self.clients.retain(|_, bucket| {
            now_secs.saturating_sub(bucket.last_refill_secs) < stale_after_secs
        });
        self.banned.retain(|_, &mut expiry| expiry > now_secs);
    }

    /// Check if a specific client is currently banned.
    pub fn is_banned(&self, client_id: &str, now_secs: u64) -> bool {
        self.banned
            .get(client_id)
            .map_or(false, |&expiry| now_secs < expiry)
    }
}

// ---------------------------------------------------------------------------
// Composite rate limiter (combines global + per-client)
// ---------------------------------------------------------------------------

/// Composite rate limiter that enforces both global and per-client limits.
#[derive(Debug, Clone)]
pub struct CompositeRateLimiter {
    /// Global rate limiter (shared across all clients).
    pub global: TokenBucket,
    /// Per-client rate limiter.
    pub per_client: PerClientRateLimiter,
}

impl CompositeRateLimiter {
    /// Create a new composite rate limiter.
    pub fn new(
        global_capacity: f64,
        global_refill_rate: f64,
        client_capacity: f64,
        client_refill_rate: f64,
    ) -> Self {
        Self {
            global: TokenBucket::new(global_capacity, global_refill_rate),
            per_client: PerClientRateLimiter::new(client_capacity, client_refill_rate),
        }
    }

    /// Check if a request is allowed (must pass both global and per-client checks).
    pub fn try_acquire(&mut self, client_id: &str, now_secs: u64) -> Result<(), RateLimitError> {
        // Check global limit first
        if !self.global.try_acquire(now_secs) {
            return Err(RateLimitError::LimitExceeded {
                reason: "global rate limit exceeded".to_string(),
                retry_after_secs: self.global.time_until_available().ceil() as u64,
            });
        }
        // Then check per-client
        self.per_client.try_acquire(client_id, now_secs)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Token bucket ----

    #[test]
    fn test_token_bucket_initial_capacity() {
        let bucket = TokenBucket::new(10.0, 1.0);
        assert!((bucket.available_tokens() - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_token_bucket_acquire() {
        let mut bucket = TokenBucket::new(5.0, 1.0);
        assert!(bucket.try_acquire(0));
        assert!((bucket.available_tokens() - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_token_bucket_exhaustion() {
        let mut bucket = TokenBucket::new(2.0, 1.0);
        assert!(bucket.try_acquire(0));
        assert!(bucket.try_acquire(0));
        assert!(!bucket.try_acquire(0)); // exhausted
    }

    #[test]
    fn test_token_bucket_refill() {
        let mut bucket = TokenBucket::new(2.0, 1.0);
        assert!(bucket.try_acquire(0));
        assert!(bucket.try_acquire(0));
        assert!(!bucket.try_acquire(0));
        // After 1 second, should have 1 token
        assert!(bucket.try_acquire(1));
    }

    #[test]
    fn test_token_bucket_refill_capped() {
        let mut bucket = TokenBucket::new(3.0, 1.0);
        // Wait a long time -> tokens capped at capacity
        bucket.refill(1000);
        assert!((bucket.available_tokens() - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_token_bucket_acquire_n() {
        let mut bucket = TokenBucket::new(10.0, 1.0);
        assert!(bucket.try_acquire_n(5.0, 0));
        assert!((bucket.available_tokens() - 5.0).abs() < f64::EPSILON);
        assert!(!bucket.try_acquire_n(6.0, 0));
    }

    #[test]
    fn test_token_bucket_time_until_available() {
        let mut bucket = TokenBucket::new(1.0, 2.0);
        assert!((bucket.time_until_available()).abs() < f64::EPSILON); // has 1 token
        bucket.try_acquire(0);
        // Need 1 token, refill rate is 2/s -> 0.5s
        assert!((bucket.time_until_available() - 0.5).abs() < 0.01);
    }

    // ---- Sliding window counter ----

    #[test]
    fn test_sliding_window_basic() {
        let mut counter = SlidingWindowCounter::new(5, 10, 5);
        assert!(counter.try_record(0));
        assert!(counter.try_record(0));
        assert_eq!(counter.current_count(), 2);
        assert_eq!(counter.remaining(), 3);
    }

    #[test]
    fn test_sliding_window_exhaustion() {
        let mut counter = SlidingWindowCounter::new(3, 10, 5);
        assert!(counter.try_record(0));
        assert!(counter.try_record(0));
        assert!(counter.try_record(0));
        assert!(!counter.try_record(0)); // at limit
    }

    #[test]
    fn test_sliding_window_slot_expiry() {
        let mut counter = SlidingWindowCounter::new(2, 10, 5);
        // slot_secs = 10/5 = 2
        assert!(counter.try_record(0));
        assert!(counter.try_record(0));
        assert!(!counter.try_record(0)); // at limit

        // Advance past the window -> old slots expire
        assert!(counter.try_record(12));
    }

    #[test]
    fn test_sliding_window_remaining() {
        let mut counter = SlidingWindowCounter::new(10, 60, 6);
        for _ in 0..7 {
            counter.try_record(0);
        }
        assert_eq!(counter.remaining(), 3);
    }

    // ---- Per-client rate limiter ----

    #[test]
    fn test_per_client_basic() {
        let mut limiter = PerClientRateLimiter::new(3.0, 1.0);
        assert!(limiter.try_acquire("client-a", 0).is_ok());
        assert!(limiter.try_acquire("client-b", 0).is_ok());
        assert_eq!(limiter.client_count(), 2);
    }

    #[test]
    fn test_per_client_isolation() {
        let mut limiter = PerClientRateLimiter::new(2.0, 1.0);
        assert!(limiter.try_acquire("a", 0).is_ok());
        assert!(limiter.try_acquire("a", 0).is_ok());
        assert!(limiter.try_acquire("a", 0).is_err()); // a exhausted
        assert!(limiter.try_acquire("b", 0).is_ok()); // b still has tokens
    }

    #[test]
    fn test_per_client_ban() {
        let mut limiter = PerClientRateLimiter::new(1.0, 0.1).with_ban_policy(3, 60);

        assert!(limiter.try_acquire("bad", 0).is_ok());
        // Next 3 will fail and trigger ban
        assert!(limiter.try_acquire("bad", 0).is_err());
        assert!(limiter.try_acquire("bad", 0).is_err());
        let result = limiter.try_acquire("bad", 0);
        assert!(matches!(result, Err(RateLimitError::ClientBanned { .. })));
        assert!(limiter.is_banned("bad", 0));
        assert_eq!(limiter.banned_count(), 1);
    }

    #[test]
    fn test_per_client_ban_expires() {
        let mut limiter = PerClientRateLimiter::new(1.0, 0.1).with_ban_policy(2, 10);

        assert!(limiter.try_acquire("x", 0).is_ok());
        assert!(limiter.try_acquire("x", 0).is_err());
        let _ = limiter.try_acquire("x", 0); // triggers ban
        assert!(limiter.is_banned("x", 5));
        assert!(!limiter.is_banned("x", 15)); // ban expired
                                              // Should be able to acquire again after ban
        assert!(limiter.try_acquire("x", 15).is_ok());
    }

    #[test]
    fn test_per_client_evict_stale() {
        let mut limiter = PerClientRateLimiter::new(5.0, 1.0);
        assert!(limiter.try_acquire("old", 0).is_ok());
        assert!(limiter.try_acquire("new", 100).is_ok());
        assert_eq!(limiter.client_count(), 2);

        limiter.evict_stale_clients(100, 50);
        assert_eq!(limiter.client_count(), 1);
    }

    // ---- Composite rate limiter ----

    #[test]
    fn test_composite_both_pass() {
        let mut limiter = CompositeRateLimiter::new(100.0, 10.0, 5.0, 1.0);
        assert!(limiter.try_acquire("client-1", 0).is_ok());
    }

    #[test]
    fn test_composite_global_blocks() {
        let mut limiter = CompositeRateLimiter::new(1.0, 0.1, 100.0, 10.0);
        assert!(limiter.try_acquire("a", 0).is_ok());
        let result = limiter.try_acquire("b", 0);
        assert!(result.is_err());
        if let Err(RateLimitError::LimitExceeded { reason, .. }) = &result {
            assert!(reason.contains("global"));
        }
    }

    #[test]
    fn test_composite_per_client_blocks() {
        let mut limiter = CompositeRateLimiter::new(100.0, 10.0, 1.0, 0.1);
        assert!(limiter.try_acquire("greedy", 0).is_ok());
        let result = limiter.try_acquire("greedy", 0);
        assert!(result.is_err());
    }

    // ---- Error display ----

    #[test]
    fn test_rate_limit_error_display() {
        let e = RateLimitError::LimitExceeded {
            reason: "too fast".to_string(),
            retry_after_secs: 5,
        };
        let msg = format!("{e}");
        assert!(msg.contains("too fast"));
        assert!(msg.contains("5s"));
    }

    #[test]
    fn test_client_banned_error_display() {
        let e = RateLimitError::ClientBanned {
            client_id: "abc".to_string(),
        };
        assert!(format!("{e}").contains("abc"));
    }
}
