//! Login rate limiting for access control in `OxiMedia`.
//!
//! Provides tracking of login attempts and enforcement of rate limits
//! to prevent brute-force authentication attacks.

#![allow(dead_code)]

use std::collections::HashMap;

/// A single recorded login attempt.
#[derive(Debug, Clone)]
pub struct LoginAttempt {
    /// Username or identifier for the attempt.
    pub username: String,
    /// Timestamp in milliseconds since epoch.
    pub timestamp_ms: u64,
    /// Whether the attempt was successful.
    pub success: bool,
}

impl LoginAttempt {
    /// Create a new login attempt record.
    #[must_use]
    pub fn new(username: impl Into<String>, timestamp_ms: u64, success: bool) -> Self {
        Self {
            username: username.into(),
            timestamp_ms,
            success,
        }
    }

    /// Returns `true` if the attempt falls within `window_ms` of `now_ms`.
    #[must_use]
    pub fn is_recent(&self, now_ms: u64, window_ms: u64) -> bool {
        now_ms.saturating_sub(self.timestamp_ms) < window_ms
    }
}

/// Configuration for the login rate limiter.
#[derive(Debug, Clone)]
pub struct LoginRateConfig {
    /// Maximum number of failed attempts allowed per window.
    pub max_failures: u32,
    /// Length of the rolling window in milliseconds.
    pub window_ms: u64,
    /// How long a blocked user stays blocked (ms).
    pub lockout_ms: u64,
}

impl LoginRateConfig {
    /// Create a new config.
    #[must_use]
    pub fn new(max_failures: u32, window_ms: u64, lockout_ms: u64) -> Self {
        Self {
            max_failures,
            window_ms,
            lockout_ms,
        }
    }

    /// Returns the maximum failures allowed per window.
    #[must_use]
    pub fn max_per_window(&self) -> u32 {
        self.max_failures
    }
}

impl Default for LoginRateConfig {
    fn default() -> Self {
        Self::new(5, 60_000, 300_000)
    }
}

/// Tracks login attempts and enforces rate limits per username.
#[derive(Debug, Default)]
pub struct LoginRateLimiter {
    config: LoginRateConfig,
    /// All recorded attempts, keyed by username.
    history: HashMap<String, Vec<LoginAttempt>>,
    /// Explicit lockout end times (ms), keyed by username.
    lockouts: HashMap<String, u64>,
}

impl LoginRateLimiter {
    /// Create a limiter with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: LoginRateConfig::default(),
            history: HashMap::new(),
            lockouts: HashMap::new(),
        }
    }

    /// Create a limiter with the given configuration.
    #[must_use]
    pub fn with_config(config: LoginRateConfig) -> Self {
        Self {
            config,
            history: HashMap::new(),
            lockouts: HashMap::new(),
        }
    }

    /// Record a login attempt for a user at the given timestamp.
    /// Applies lockout if failure threshold is exceeded.
    pub fn record_attempt(&mut self, attempt: LoginAttempt, now_ms: u64) {
        let username = attempt.username.clone();
        let entry = self.history.entry(username.clone()).or_default();
        entry.push(attempt);

        // Check if we should impose a lockout
        let failures = self.window_count(&username, now_ms, false);
        if failures >= self.config.max_failures {
            self.lockouts
                .insert(username, now_ms + self.config.lockout_ms);
        }
    }

    /// Returns `true` if the user is currently blocked.
    #[must_use]
    pub fn is_blocked(&self, username: &str, now_ms: u64) -> bool {
        if let Some(&until) = self.lockouts.get(username) {
            if now_ms < until {
                return true;
            }
        }
        false
    }

    /// Count attempts for `username` within the rolling window.
    /// If `successes_only` is `false`, counts only failures.
    #[must_use]
    pub fn window_count(&self, username: &str, now_ms: u64, successes_only: bool) -> u32 {
        let Some(attempts) = self.history.get(username) else {
            return 0;
        };
        attempts
            .iter()
            .filter(|a| {
                a.is_recent(now_ms, self.config.window_ms)
                    && (successes_only == a.success || (!successes_only && !a.success))
            })
            .count() as u32
    }

    /// Return total recorded attempts for a user (all time).
    #[must_use]
    pub fn total_attempts(&self, username: &str) -> usize {
        self.history.get(username).map_or(0, Vec::len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn attempt(user: &str, ts: u64, ok: bool) -> LoginAttempt {
        LoginAttempt::new(user, ts, ok)
    }

    #[test]
    fn test_is_recent_within_window() {
        let a = attempt("alice", 1000, false);
        assert!(a.is_recent(1500, 1000));
    }

    #[test]
    fn test_is_recent_outside_window() {
        let a = attempt("alice", 1000, false);
        assert!(!a.is_recent(2001, 1000));
    }

    #[test]
    fn test_is_recent_exact_boundary() {
        let a = attempt("alice", 1000, false);
        // difference == window_ms is NOT recent (< not <=)
        assert!(!a.is_recent(2000, 1000));
    }

    #[test]
    fn test_max_per_window_default() {
        let cfg = LoginRateConfig::default();
        assert_eq!(cfg.max_per_window(), 5);
    }

    #[test]
    fn test_max_per_window_custom() {
        let cfg = LoginRateConfig::new(3, 30_000, 120_000);
        assert_eq!(cfg.max_per_window(), 3);
    }

    #[test]
    fn test_no_block_initially() {
        let limiter = LoginRateLimiter::new();
        assert!(!limiter.is_blocked("bob", 0));
    }

    #[test]
    fn test_window_count_empty() {
        let limiter = LoginRateLimiter::new();
        assert_eq!(limiter.window_count("bob", 0, false), 0);
    }

    #[test]
    fn test_record_failure_increments_count() {
        let mut limiter = LoginRateLimiter::new();
        limiter.record_attempt(attempt("carol", 100, false), 200);
        limiter.record_attempt(attempt("carol", 150, false), 200);
        assert_eq!(limiter.window_count("carol", 200, false), 2);
    }

    #[test]
    fn test_success_not_counted_as_failure() {
        let mut limiter = LoginRateLimiter::new();
        limiter.record_attempt(attempt("carol", 100, true), 200);
        assert_eq!(limiter.window_count("carol", 200, false), 0);
    }

    #[test]
    fn test_lockout_after_max_failures() {
        let cfg = LoginRateConfig::new(3, 60_000, 300_000);
        let mut limiter = LoginRateLimiter::with_config(cfg);
        let now = 10_000u64;
        for _ in 0..3 {
            limiter.record_attempt(attempt("dave", now - 100, false), now);
        }
        assert!(limiter.is_blocked("dave", now));
    }

    #[test]
    fn test_lockout_expires() {
        let cfg = LoginRateConfig::new(3, 60_000, 1_000);
        let mut limiter = LoginRateLimiter::with_config(cfg);
        let now = 10_000u64;
        for _ in 0..3 {
            limiter.record_attempt(attempt("eve", now - 100, false), now);
        }
        assert!(limiter.is_blocked("eve", now));
        assert!(!limiter.is_blocked("eve", now + 2_000));
    }

    #[test]
    fn test_total_attempts_counts_all() {
        let mut limiter = LoginRateLimiter::new();
        limiter.record_attempt(attempt("frank", 100, true), 200);
        limiter.record_attempt(attempt("frank", 150, false), 200);
        assert_eq!(limiter.total_attempts("frank"), 2);
    }

    #[test]
    fn test_different_users_isolated() {
        let mut limiter = LoginRateLimiter::new();
        limiter.record_attempt(attempt("grace", 100, false), 200);
        assert_eq!(limiter.window_count("heidi", 200, false), 0);
    }

    #[test]
    fn test_with_config_constructor() {
        let cfg = LoginRateConfig::new(10, 5_000, 60_000);
        let limiter = LoginRateLimiter::with_config(cfg);
        assert!(!limiter.is_blocked("ivan", 0));
    }
}
