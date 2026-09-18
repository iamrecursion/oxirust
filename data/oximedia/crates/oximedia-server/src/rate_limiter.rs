//! Token-bucket rate limiter with per-client tracking.
//!
//! Provides `RateLimiter` (single bucket) and `PerClientRateLimiter`
//! (HashMap-backed, one bucket per client).
//!
//! # Example
//!
//! ```rust
//! use oximedia_server::rate_limiter::{TokenBucketConfig, RateLimiter};
//!
//! let config = TokenBucketConfig { capacity: 10, refill_rate: 1.0, refill_interval_ms: 1000 };
//! let mut limiter = RateLimiter::new(config);
//! assert!(limiter.try_consume(1).is_ok());
//! ```

#![allow(dead_code)]

use std::collections::HashMap;

// ── Configuration ─────────────────────────────────────────────────────────────

/// Configuration parameters for a token-bucket rate limiter.
#[derive(Debug, Clone)]
pub struct TokenBucketConfig {
    /// Maximum token capacity (burst ceiling).
    pub capacity: u32,
    /// Number of tokens added per refill interval.
    pub refill_rate: f32,
    /// Refill interval in milliseconds.
    pub refill_interval_ms: u64,
}

impl TokenBucketConfig {
    /// Creates a new config with the given parameters.
    pub fn new(capacity: u32, refill_rate: f32, refill_interval_ms: u64) -> Self {
        Self {
            capacity,
            refill_rate,
            refill_interval_ms,
        }
    }

    /// Tokens-per-millisecond refill rate.
    pub fn tokens_per_ms(&self) -> f64 {
        if self.refill_interval_ms == 0 {
            return 0.0;
        }
        f64::from(self.refill_rate) / self.refill_interval_ms as f64
    }
}

impl Default for TokenBucketConfig {
    fn default() -> Self {
        Self {
            capacity: 100,
            refill_rate: 10.0,
            refill_interval_ms: 1_000,
        }
    }
}

// ── Error type ────────────────────────────────────────────────────────────────

/// Error returned when a rate-limit check fails.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RateLimitError {
    /// The bucket does not have enough tokens.
    InsufficientTokens {
        /// Tokens currently available (integer floor).
        available: u32,
        /// Tokens requested.
        requested: u32,
    },
    /// The requested token count exceeds the bucket capacity.
    ExceedsCapacity {
        /// Bucket capacity.
        capacity: u32,
        /// Tokens requested.
        requested: u32,
    },
    /// Zero-token consumption is not meaningful.
    ZeroRequest,
}

impl std::fmt::Display for RateLimitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InsufficientTokens {
                available,
                requested,
            } => write!(
                f,
                "rate limit: need {requested} tokens but only {available} available"
            ),
            Self::ExceedsCapacity {
                capacity,
                requested,
            } => write!(
                f,
                "rate limit: request for {requested} tokens exceeds capacity {capacity}"
            ),
            Self::ZeroRequest => write!(f, "rate limit: zero-token request is invalid"),
        }
    }
}

impl std::error::Error for RateLimitError {}

// ── Single-bucket rate limiter ────────────────────────────────────────────────

/// A single token-bucket rate limiter.
///
/// Tokens accumulate at the configured `refill_rate` per `refill_interval_ms`.
/// Bursts up to `capacity` are allowed.
#[derive(Debug, Clone)]
pub struct RateLimiter {
    config: TokenBucketConfig,
    /// Current token level (fractional for smooth refill).
    tokens: f64,
    /// Last refill timestamp in milliseconds.
    last_refill_ms: u64,
}

impl RateLimiter {
    /// Creates a new rate limiter filled to capacity.
    pub fn new(config: TokenBucketConfig) -> Self {
        let tokens = f64::from(config.capacity);
        Self {
            config,
            tokens,
            last_refill_ms: 0,
        }
    }

    /// Returns the number of whole tokens currently available.
    pub fn available_tokens(&self) -> u32 {
        self.tokens.floor() as u32
    }

    /// Refills the bucket based on elapsed time since `last_refill_ms`.
    pub fn refill(&mut self, now_ms: u64) {
        if now_ms > self.last_refill_ms {
            let elapsed = (now_ms - self.last_refill_ms) as f64;
            let gained = elapsed * self.config.tokens_per_ms();
            self.tokens = (self.tokens + gained).min(f64::from(self.config.capacity));
            self.last_refill_ms = now_ms;
        }
    }

    /// Attempts to consume `tokens` from the bucket.
    ///
    /// Does NOT refill before consuming — call [`refill`][Self::refill] first
    /// if you want time-based replenishment.
    ///
    /// # Errors
    ///
    /// - [`RateLimitError::ZeroRequest`] if `tokens == 0`.
    /// - [`RateLimitError::ExceedsCapacity`] if `tokens > capacity`.
    /// - [`RateLimitError::InsufficientTokens`] if not enough tokens available.
    pub fn try_consume(&mut self, tokens: u32) -> Result<(), RateLimitError> {
        if tokens == 0 {
            return Err(RateLimitError::ZeroRequest);
        }
        if tokens > self.config.capacity {
            return Err(RateLimitError::ExceedsCapacity {
                capacity: self.config.capacity,
                requested: tokens,
            });
        }
        let requested = f64::from(tokens);
        if self.tokens >= requested {
            self.tokens -= requested;
            Ok(())
        } else {
            Err(RateLimitError::InsufficientTokens {
                available: self.available_tokens(),
                requested: tokens,
            })
        }
    }

    /// Refills at `now_ms` then tries to consume `tokens`.
    pub fn try_consume_at(&mut self, tokens: u32, now_ms: u64) -> Result<(), RateLimitError> {
        self.refill(now_ms);
        self.try_consume(tokens)
    }

    /// Returns the configuration.
    pub fn config(&self) -> &TokenBucketConfig {
        &self.config
    }

    /// Returns the exact (fractional) token count.
    pub fn tokens_exact(&self) -> f64 {
        self.tokens
    }

    /// Resets the bucket to full capacity.
    pub fn reset(&mut self) {
        self.tokens = f64::from(self.config.capacity);
        self.last_refill_ms = 0;
    }
}

// ── Per-client rate limiter ───────────────────────────────────────────────────

/// A client identifier (typically an IP address or user ID string).
pub type ClientId = String;

/// Per-client rate limiter backed by a `HashMap<ClientId, RateLimiter>`.
///
/// A new bucket (filled to capacity) is created on first access for any client.
#[derive(Debug)]
pub struct PerClientRateLimiter {
    default_config: TokenBucketConfig,
    buckets: HashMap<ClientId, RateLimiter>,
}

impl PerClientRateLimiter {
    /// Creates a new per-client limiter with the given default configuration.
    pub fn new(default_config: TokenBucketConfig) -> Self {
        Self {
            default_config,
            buckets: HashMap::new(),
        }
    }

    /// Returns (or creates) the bucket for `client_id`.
    pub fn get_or_create(&mut self, client_id: &ClientId) -> &mut RateLimiter {
        let config = self.default_config.clone();
        self.buckets
            .entry(client_id.clone())
            .or_insert_with(|| RateLimiter::new(config))
    }

    /// Refills the client's bucket at `now_ms`, then attempts to consume `tokens`.
    ///
    /// Creates a new full bucket if the client has not been seen before.
    pub fn try_consume(
        &mut self,
        client_id: &ClientId,
        tokens: u32,
        now_ms: u64,
    ) -> Result<(), RateLimitError> {
        let config = self.default_config.clone();
        let bucket = self
            .buckets
            .entry(client_id.clone())
            .or_insert_with(|| RateLimiter::new(config));
        bucket.refill(now_ms);
        bucket.try_consume(tokens)
    }

    /// Removes the bucket for `client_id`. Returns `true` if one existed.
    pub fn remove(&mut self, client_id: &ClientId) -> bool {
        self.buckets.remove(client_id).is_some()
    }

    /// Returns the number of tracked clients.
    pub fn client_count(&self) -> usize {
        self.buckets.len()
    }

    /// Removes all clients whose buckets are full (no consumption has occurred).
    ///
    /// This is a lightweight idle cleanup: clients that have never consumed
    /// tokens and are back at capacity are safe to evict.
    pub fn cleanup_empty(&mut self) {
        let capacity_cutoff: Vec<ClientId> = self
            .buckets
            .iter()
            .filter(|(_, b)| b.available_tokens() >= b.config().capacity)
            .map(|(id, _)| id.clone())
            .collect();
        for id in capacity_cutoff {
            self.buckets.remove(&id);
        }
    }

    /// Returns a reference to the default config.
    pub fn default_config(&self) -> &TokenBucketConfig {
        &self.default_config
    }

    /// Returns an iterator over all tracked client IDs.
    pub fn client_ids(&self) -> impl Iterator<Item = &ClientId> {
        self.buckets.keys()
    }

    /// Resets all buckets to full capacity without removing them.
    pub fn reset_all(&mut self) {
        for bucket in self.buckets.values_mut() {
            bucket.reset();
        }
    }

    /// Returns available tokens for a client (0 if client not yet tracked).
    pub fn available_for(&self, client_id: &ClientId) -> u32 {
        self.buckets
            .get(client_id)
            .map_or(self.default_config.capacity, |b| b.available_tokens())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> TokenBucketConfig {
        TokenBucketConfig {
            capacity: 10,
            refill_rate: 1.0,
            refill_interval_ms: 1_000,
        }
    }

    // ── TokenBucketConfig ────────────────────────────────────────────────────

    #[test]
    fn test_config_default_values() {
        let c = TokenBucketConfig::default();
        assert_eq!(c.capacity, 100);
        assert!(c.refill_rate > 0.0);
        assert!(c.refill_interval_ms > 0);
    }

    #[test]
    fn test_config_tokens_per_ms() {
        let c = TokenBucketConfig {
            capacity: 100,
            refill_rate: 10.0,
            refill_interval_ms: 1_000,
        };
        assert!((c.tokens_per_ms() - 0.01).abs() < 1e-9);
    }

    #[test]
    fn test_config_tokens_per_ms_zero_interval() {
        let c = TokenBucketConfig {
            capacity: 10,
            refill_rate: 5.0,
            refill_interval_ms: 0,
        };
        assert_eq!(c.tokens_per_ms(), 0.0);
    }

    // ── RateLimiter – basic consumption ──────────────────────────────────────

    #[test]
    fn test_new_limiter_starts_full() {
        let limiter = RateLimiter::new(default_config());
        assert_eq!(limiter.available_tokens(), 10);
    }

    #[test]
    fn test_try_consume_single_token_ok() {
        let mut limiter = RateLimiter::new(default_config());
        assert!(limiter.try_consume(1).is_ok());
        assert_eq!(limiter.available_tokens(), 9);
    }

    #[test]
    fn test_try_consume_multiple_tokens_ok() {
        let mut limiter = RateLimiter::new(default_config());
        assert!(limiter.try_consume(5).is_ok());
        assert_eq!(limiter.available_tokens(), 5);
    }

    #[test]
    fn test_try_consume_all_tokens_ok() {
        let mut limiter = RateLimiter::new(default_config());
        assert!(limiter.try_consume(10).is_ok());
        assert_eq!(limiter.available_tokens(), 0);
    }

    #[test]
    fn test_try_consume_fails_when_insufficient() {
        let mut limiter = RateLimiter::new(default_config());
        limiter.try_consume(10).expect("first consume");
        let err = limiter.try_consume(1).unwrap_err();
        assert!(matches!(
            err,
            RateLimitError::InsufficientTokens {
                available: 0,
                requested: 1
            }
        ));
    }

    #[test]
    fn test_try_consume_zero_returns_error() {
        let mut limiter = RateLimiter::new(default_config());
        let err = limiter.try_consume(0).unwrap_err();
        assert_eq!(err, RateLimitError::ZeroRequest);
    }

    #[test]
    fn test_try_consume_exceeds_capacity_returns_error() {
        let mut limiter = RateLimiter::new(default_config());
        let err = limiter.try_consume(11).unwrap_err();
        assert!(matches!(
            err,
            RateLimitError::ExceedsCapacity {
                capacity: 10,
                requested: 11
            }
        ));
    }

    // ── RateLimiter – refill ──────────────────────────────────────────────────

    #[test]
    fn test_refill_adds_tokens_over_time() {
        // 1 token per ms (rate=1, interval=1ms)
        let config = TokenBucketConfig {
            capacity: 10,
            refill_rate: 1.0,
            refill_interval_ms: 1,
        };
        let mut limiter = RateLimiter::new(config);
        limiter.try_consume(10).expect("drain");
        limiter.refill(5); // +5 tokens
        assert_eq!(limiter.available_tokens(), 5);
    }

    #[test]
    fn test_refill_caps_at_capacity() {
        let config = TokenBucketConfig {
            capacity: 5,
            refill_rate: 1.0,
            refill_interval_ms: 1,
        };
        let mut limiter = RateLimiter::new(config);
        limiter.refill(1_000_000); // massive time delta
        assert_eq!(limiter.available_tokens(), 5); // capped at capacity
    }

    #[test]
    fn test_try_consume_at_refills_then_consumes() {
        let config = TokenBucketConfig {
            capacity: 10,
            refill_rate: 1.0,
            refill_interval_ms: 1,
        };
        let mut limiter = RateLimiter::new(config);
        limiter.try_consume(10).expect("drain");
        // Simulate 3 ms passing → 3 tokens refilled
        assert!(limiter.try_consume_at(3, 3).is_ok());
    }

    #[test]
    fn test_reset_restores_full_capacity() {
        let mut limiter = RateLimiter::new(default_config());
        limiter.try_consume(10).expect("drain");
        assert_eq!(limiter.available_tokens(), 0);
        limiter.reset();
        assert_eq!(limiter.available_tokens(), 10);
    }

    // ── PerClientRateLimiter ─────────────────────────────────────────────────

    #[test]
    fn test_per_client_creates_new_bucket_on_first_access() {
        let mut pcrl = PerClientRateLimiter::new(default_config());
        assert_eq!(pcrl.client_count(), 0);
        let id = "client-1".to_string();
        pcrl.get_or_create(&id);
        assert_eq!(pcrl.client_count(), 1);
    }

    #[test]
    fn test_per_client_try_consume_ok() {
        let mut pcrl = PerClientRateLimiter::new(default_config());
        let id = "user-a".to_string();
        assert!(pcrl.try_consume(&id, 1, 0).is_ok());
    }

    #[test]
    fn test_per_client_clients_are_isolated() {
        let config = TokenBucketConfig {
            capacity: 2,
            refill_rate: 0.0,
            refill_interval_ms: 1_000,
        };
        let mut pcrl = PerClientRateLimiter::new(config);
        let a = "alice".to_string();
        let b = "bob".to_string();
        pcrl.try_consume(&a, 2, 0).expect("alice drain");
        // alice is exhausted; bob is unaffected
        assert!(pcrl.try_consume(&b, 1, 0).is_ok());
        assert!(pcrl.try_consume(&a, 1, 0).is_err());
    }

    #[test]
    fn test_per_client_remove() {
        let mut pcrl = PerClientRateLimiter::new(default_config());
        let id = "temp".to_string();
        pcrl.get_or_create(&id);
        assert!(pcrl.remove(&id));
        assert!(!pcrl.remove(&id)); // already gone
        assert_eq!(pcrl.client_count(), 0);
    }

    #[test]
    fn test_per_client_cleanup_empty_removes_full_buckets() {
        let config = TokenBucketConfig {
            capacity: 5,
            refill_rate: 0.0,
            refill_interval_ms: 1_000,
        };
        let mut pcrl = PerClientRateLimiter::new(config);
        let active = "active".to_string();
        let idle = "idle".to_string();
        pcrl.get_or_create(&idle); // never consumed → full
        pcrl.try_consume(&active, 1, 0).expect("consume");
        pcrl.cleanup_empty();
        // idle bucket (full) should be removed; active bucket stays
        assert_eq!(pcrl.client_count(), 1);
        assert!(pcrl.buckets.contains_key(&active));
    }

    #[test]
    fn test_per_client_available_for_unknown_client_returns_capacity() {
        let pcrl = PerClientRateLimiter::new(default_config());
        let id = "unknown".to_string();
        assert_eq!(pcrl.available_for(&id), 10);
    }

    #[test]
    fn test_per_client_reset_all() {
        let config = TokenBucketConfig {
            capacity: 5,
            refill_rate: 0.0,
            refill_interval_ms: 1_000,
        };
        let mut pcrl = PerClientRateLimiter::new(config);
        let id = "user".to_string();
        pcrl.try_consume(&id, 5, 0).expect("drain");
        assert_eq!(pcrl.available_for(&id), 0);
        pcrl.reset_all();
        assert_eq!(pcrl.available_for(&id), 5);
    }

    #[test]
    fn test_per_client_refill_via_try_consume_at() {
        let config = TokenBucketConfig {
            capacity: 10,
            refill_rate: 1.0,
            refill_interval_ms: 1, // 1 token/ms
        };
        let mut pcrl = PerClientRateLimiter::new(config);
        let id = "refill-user".to_string();
        pcrl.try_consume(&id, 10, 0).expect("drain");
        assert!(pcrl.try_consume(&id, 5, 0).is_err());
        // After 5 ms: +5 tokens
        assert!(pcrl.try_consume(&id, 5, 5).is_ok());
    }

    #[test]
    fn test_error_display_insufficient() {
        let e = RateLimitError::InsufficientTokens {
            available: 2,
            requested: 5,
        };
        let s = e.to_string();
        assert!(s.contains("5"));
        assert!(s.contains("2"));
    }

    #[test]
    fn test_error_display_exceeds_capacity() {
        let e = RateLimitError::ExceedsCapacity {
            capacity: 10,
            requested: 20,
        };
        let s = e.to_string();
        assert!(s.contains("20"));
        assert!(s.contains("10"));
    }

    #[test]
    fn test_error_display_zero_request() {
        let e = RateLimitError::ZeroRequest;
        assert!(!e.to_string().is_empty());
    }
}
