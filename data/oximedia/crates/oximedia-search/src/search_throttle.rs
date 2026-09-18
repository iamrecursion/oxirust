#![allow(dead_code)]

//! Search request throttling and rate limiting.
//!
//! This module provides configurable rate limiters for search queries.
//! It supports per-user token-bucket rate limiting, global throughput caps,
//! and burst allowance logic.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Configuration for a token-bucket rate limiter.
#[derive(Debug, Clone)]
pub struct ThrottleConfig {
    /// Maximum number of tokens (burst capacity).
    pub capacity: u64,
    /// Tokens replenished per second.
    pub refill_rate: f64,
    /// Initial tokens granted to new users.
    pub initial_tokens: u64,
}

impl Default for ThrottleConfig {
    fn default() -> Self {
        Self {
            capacity: 100,
            refill_rate: 10.0,
            initial_tokens: 100,
        }
    }
}

/// Internal state for one rate-limit bucket.
#[derive(Debug, Clone)]
struct Bucket {
    /// Current available tokens.
    tokens: f64,
    /// Timestamp (seconds) of the last refill calculation.
    last_refill: f64,
    /// Total requests attempted (including denied).
    total_attempts: u64,
    /// Total requests that were allowed.
    total_allowed: u64,
}

/// Outcome of trying to consume a token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ThrottleDecision {
    /// The request is allowed.
    Allowed,
    /// The request is denied; the caller should wait.
    Denied,
}

/// Per-user rate limiter backed by token buckets.
#[derive(Debug)]
pub struct SearchThrottle {
    /// Configuration.
    config: ThrottleConfig,
    /// Per-user buckets.
    buckets: HashMap<u64, Bucket>,
    /// Global bucket (user id 0 is reserved for global).
    global: Bucket,
}

/// Summary statistics of throttle state.
#[derive(Debug, Clone)]
pub struct ThrottleStats {
    /// Number of tracked users.
    pub tracked_users: usize,
    /// Total attempts across all users.
    pub total_attempts: u64,
    /// Total allowed across all users.
    pub total_allowed: u64,
    /// Total denied across all users.
    pub total_denied: u64,
}

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

impl SearchThrottle {
    /// Create a new throttle with the given configuration.
    pub fn new(config: ThrottleConfig) -> Self {
        let global = Bucket {
            tokens: config.initial_tokens as f64,
            last_refill: 0.0,
            total_attempts: 0,
            total_allowed: 0,
        };
        Self {
            config,
            buckets: HashMap::new(),
            global,
        }
    }

    /// Create a throttle with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(ThrottleConfig::default())
    }

    /// Try to consume one token for the given user at the given time.
    pub fn try_acquire(&mut self, user_id: u64, now_secs: f64) -> ThrottleDecision {
        // Refill & check per-user bucket.
        let bucket = self.buckets.entry(user_id).or_insert_with(|| Bucket {
            tokens: self.config.initial_tokens as f64,
            last_refill: now_secs,
            total_attempts: 0,
            total_allowed: 0,
        });

        Self::refill(
            bucket,
            now_secs,
            self.config.refill_rate,
            self.config.capacity,
        );
        bucket.total_attempts += 1;

        if bucket.tokens < 1.0 {
            return ThrottleDecision::Denied;
        }

        bucket.tokens -= 1.0;
        bucket.total_allowed += 1;

        // Also check global bucket.
        Self::refill(
            &mut self.global,
            now_secs,
            self.config.refill_rate,
            self.config.capacity,
        );
        self.global.total_attempts += 1;

        if self.global.tokens < 1.0 {
            return ThrottleDecision::Denied;
        }

        self.global.tokens -= 1.0;
        self.global.total_allowed += 1;

        ThrottleDecision::Allowed
    }

    /// Peek at the number of available tokens for a user without consuming.
    pub fn available_tokens(&self, user_id: u64) -> f64 {
        self.buckets
            .get(&user_id)
            .map_or(self.config.initial_tokens as f64, |b| b.tokens)
    }

    /// Estimated seconds until one token becomes available for a user.
    pub fn wait_estimate(&self, user_id: u64) -> f64 {
        let tokens = self.available_tokens(user_id);
        if tokens >= 1.0 {
            return 0.0;
        }
        let deficit = 1.0 - tokens;
        deficit / self.config.refill_rate
    }

    /// Return aggregate statistics.
    pub fn stats(&self) -> ThrottleStats {
        let total_attempts: u64 = self.buckets.values().map(|b| b.total_attempts).sum();
        let total_allowed: u64 = self.buckets.values().map(|b| b.total_allowed).sum();
        ThrottleStats {
            tracked_users: self.buckets.len(),
            total_attempts,
            total_allowed,
            total_denied: total_attempts.saturating_sub(total_allowed),
        }
    }

    /// Reset the bucket for a specific user.
    pub fn reset_user(&mut self, user_id: u64) {
        self.buckets.remove(&user_id);
    }

    /// Reset all buckets.
    pub fn reset_all(&mut self) {
        self.buckets.clear();
        self.global = Bucket {
            tokens: self.config.initial_tokens as f64,
            last_refill: 0.0,
            total_attempts: 0,
            total_allowed: 0,
        };
    }

    /// Number of tracked user buckets.
    pub fn tracked_users(&self) -> usize {
        self.buckets.len()
    }

    /// Reference to the current configuration.
    pub fn config(&self) -> &ThrottleConfig {
        &self.config
    }

    // Internal refill logic.
    fn refill(bucket: &mut Bucket, now_secs: f64, rate: f64, capacity: u64) {
        let elapsed = now_secs - bucket.last_refill;
        if elapsed > 0.0 {
            bucket.tokens = (bucket.tokens + elapsed * rate).min(capacity as f64);
            bucket.last_refill = now_secs;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = ThrottleConfig::default();
        assert_eq!(cfg.capacity, 100);
        assert!((cfg.refill_rate - 10.0).abs() < f64::EPSILON);
        assert_eq!(cfg.initial_tokens, 100);
    }

    #[test]
    fn test_with_defaults() {
        let t = SearchThrottle::with_defaults();
        assert_eq!(t.tracked_users(), 0);
    }

    #[test]
    fn test_acquire_allowed() {
        let mut t = SearchThrottle::with_defaults();
        let d = t.try_acquire(1, 0.0);
        assert_eq!(d, ThrottleDecision::Allowed);
    }

    #[test]
    fn test_acquire_exhaust_tokens() {
        let config = ThrottleConfig {
            capacity: 2,
            refill_rate: 0.0, // no refill
            initial_tokens: 2,
        };
        let mut t = SearchThrottle::new(config);
        assert_eq!(t.try_acquire(1, 0.0), ThrottleDecision::Allowed);
        assert_eq!(t.try_acquire(1, 0.0), ThrottleDecision::Allowed);
        assert_eq!(t.try_acquire(1, 0.0), ThrottleDecision::Denied);
    }

    #[test]
    fn test_refill_restores_tokens() {
        let config = ThrottleConfig {
            capacity: 5,
            refill_rate: 1.0,
            initial_tokens: 1,
        };
        let mut t = SearchThrottle::new(config);
        // Exhaust the single token.
        assert_eq!(t.try_acquire(1, 0.0), ThrottleDecision::Allowed);
        assert_eq!(t.try_acquire(1, 0.0), ThrottleDecision::Denied);
        // After 2 seconds, 2 tokens should be refilled.
        assert_eq!(t.try_acquire(1, 2.0), ThrottleDecision::Allowed);
    }

    #[test]
    fn test_available_tokens_new_user() {
        let t = SearchThrottle::with_defaults();
        assert!((t.available_tokens(99) - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_available_tokens_after_use() {
        let mut t = SearchThrottle::with_defaults();
        t.try_acquire(1, 0.0);
        assert!((t.available_tokens(1) - 99.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_wait_estimate_zero() {
        let t = SearchThrottle::with_defaults();
        assert!((t.wait_estimate(1) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_wait_estimate_positive() {
        let config = ThrottleConfig {
            capacity: 1,
            refill_rate: 2.0,
            initial_tokens: 0,
        };
        let t = SearchThrottle::new(config);
        // 0 tokens, need 1, rate 2/sec → wait 0.5 sec
        assert!((t.wait_estimate(42) - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_stats() {
        let mut t = SearchThrottle::with_defaults();
        t.try_acquire(1, 0.0);
        t.try_acquire(2, 0.0);
        let s = t.stats();
        assert_eq!(s.tracked_users, 2);
        assert_eq!(s.total_attempts, 2);
        assert_eq!(s.total_allowed, 2);
        assert_eq!(s.total_denied, 0);
    }

    #[test]
    fn test_reset_user() {
        let mut t = SearchThrottle::with_defaults();
        t.try_acquire(1, 0.0);
        assert_eq!(t.tracked_users(), 1);
        t.reset_user(1);
        assert_eq!(t.tracked_users(), 0);
    }

    #[test]
    fn test_reset_all() {
        let mut t = SearchThrottle::with_defaults();
        t.try_acquire(1, 0.0);
        t.try_acquire(2, 0.0);
        t.reset_all();
        assert_eq!(t.tracked_users(), 0);
    }

    #[test]
    fn test_global_bucket_cap() {
        let config = ThrottleConfig {
            capacity: 2,
            refill_rate: 0.0,
            initial_tokens: 2,
        };
        let mut t = SearchThrottle::new(config);
        // Two different users each take 1 from global
        assert_eq!(t.try_acquire(1, 0.0), ThrottleDecision::Allowed);
        assert_eq!(t.try_acquire(2, 0.0), ThrottleDecision::Allowed);
        // Global is now exhausted even though user 3 has tokens
        assert_eq!(t.try_acquire(3, 0.0), ThrottleDecision::Denied);
    }
}
