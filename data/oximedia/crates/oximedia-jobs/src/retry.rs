// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Retry logic for failed jobs.
//!
//! Provides three retry strategies (Fixed, ExponentialBackoff, Linear) plus a
//! `RetryState` tracker that records failure history and computes the next
//! retry window.  A `CircuitBreaker` is also included to prevent retry storms
//! when a downstream service is consistently failing.

use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// RetryStrategy
// ---------------------------------------------------------------------------

/// Strategy that determines whether and when to retry a failed job.
#[derive(Debug, Clone)]
pub enum RetryStrategy {
    /// No retries – fail immediately.
    NoRetry,

    /// Retry up to `max_retries` times with a constant `delay` between each.
    Fixed { delay: Duration, max_retries: u32 },

    /// Retry with exponentially increasing delays.
    ///
    /// `delay(n) = min(initial_delay * multiplier^n, max_delay)`
    ExponentialBackoff {
        initial_delay: Duration,
        max_delay: Duration,
        multiplier: f64,
        max_retries: u32,
    },

    /// Retry with linearly increasing delays.
    ///
    /// `delay(n) = initial_delay + increment * n`
    Linear {
        initial_delay: Duration,
        increment: Duration,
        max_retries: u32,
    },
}

impl RetryStrategy {
    /// Returns `true` if another attempt should be made after `attempt` failures.
    ///
    /// `attempt` is 0-indexed: 0 means "this is the result of the first try".
    #[must_use]
    pub fn should_retry(&self, attempt: u32) -> bool {
        match self {
            Self::NoRetry => false,
            Self::Fixed { max_retries, .. }
            | Self::ExponentialBackoff { max_retries, .. }
            | Self::Linear { max_retries, .. } => attempt < *max_retries,
        }
    }

    /// How long to wait before attempt number `attempt` (0-indexed).
    ///
    /// Returns `Duration::ZERO` for `NoRetry` or when `attempt == 0`.
    #[must_use]
    pub fn delay_for_attempt(&self, attempt: u32) -> Duration {
        match self {
            Self::NoRetry => Duration::ZERO,
            Self::Fixed { delay, .. } => *delay,
            Self::ExponentialBackoff {
                initial_delay,
                max_delay,
                multiplier,
                ..
            } => {
                // delay = initial * multiplier^attempt, capped at max_delay
                let factor = multiplier.powi(attempt as i32);
                let secs = initial_delay.as_secs_f64() * factor;
                let computed = Duration::from_secs_f64(secs);
                computed.min(*max_delay)
            }
            Self::Linear {
                initial_delay,
                increment,
                ..
            } => *initial_delay + *increment * attempt,
        }
    }

    /// Maximum number of retries configured.  Returns 0 for `NoRetry`.
    #[must_use]
    pub fn max_retries(&self) -> u32 {
        match self {
            Self::NoRetry => 0,
            Self::Fixed { max_retries, .. }
            | Self::ExponentialBackoff { max_retries, .. }
            | Self::Linear { max_retries, .. } => *max_retries,
        }
    }

    /// Human-readable strategy name.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::NoRetry => "NoRetry",
            Self::Fixed { .. } => "Fixed",
            Self::ExponentialBackoff { .. } => "ExponentialBackoff",
            Self::Linear { .. } => "Linear",
        }
    }
}

// ---------------------------------------------------------------------------
// RetryState
// ---------------------------------------------------------------------------

/// Tracks retry history for a single job execution.
#[derive(Debug, Clone)]
pub struct RetryState {
    /// Number of attempts made so far (incremented on each `record_failure`).
    pub attempt: u32,
    /// The retry strategy governing this job.
    pub strategy: RetryStrategy,
    /// Error message from the most recent failure.
    pub last_error: Option<String>,
    /// When the next retry should be attempted (`None` if no retry is pending).
    pub next_retry_at: Option<Instant>,
    /// Cumulative wait time across all retries.
    pub total_delay: Duration,
}

impl RetryState {
    /// Create a new `RetryState` with zero attempts.
    #[must_use]
    pub fn new(strategy: RetryStrategy) -> Self {
        Self {
            attempt: 0,
            strategy,
            last_error: None,
            next_retry_at: None,
            total_delay: Duration::ZERO,
        }
    }

    /// Record a failure and schedule the next retry if the strategy allows it.
    ///
    /// Returns `true` if a retry has been scheduled (i.e. the caller should try
    /// again after `next_retry_at`), or `false` if max retries have been
    /// exhausted.
    pub fn record_failure(&mut self, error: &str) -> bool {
        self.last_error = Some(error.to_string());

        if self.strategy.should_retry(self.attempt) {
            let delay = self.strategy.delay_for_attempt(self.attempt);
            self.total_delay += delay;
            self.next_retry_at = Some(Instant::now() + delay);
            self.attempt += 1;
            true
        } else {
            self.attempt += 1;
            self.next_retry_at = None;
            false
        }
    }

    /// Returns `true` if the scheduled retry window has passed and the job
    /// should now be re-executed.
    #[must_use]
    pub fn should_retry_now(&self) -> bool {
        self.next_retry_at
            .map(|t| Instant::now() >= t)
            .unwrap_or(false)
    }

    /// Reset the attempt counter and clear retry scheduling (e.g. after a
    /// successful execution or manual intervention).
    pub fn reset(&mut self) {
        self.attempt = 0;
        self.last_error = None;
        self.next_retry_at = None;
        self.total_delay = Duration::ZERO;
    }
}

// ---------------------------------------------------------------------------
// CircuitBreaker
// ---------------------------------------------------------------------------

/// State of the circuit breaker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Normal operation – requests pass through.
    Closed,
    /// The circuit has tripped – all requests are blocked.
    Open,
    /// The circuit is testing recovery – a limited number of requests are allowed.
    HalfOpen,
}

/// Circuit breaker that prevents retry storms.
///
/// The state machine is:
/// ```text
/// Closed --[failures >= threshold]--> Open
/// Open   --[open_duration elapsed]--> HalfOpen
/// HalfOpen --[success_threshold successes]--> Closed
/// HalfOpen --[any failure]----------> Open
/// ```
pub struct CircuitBreaker {
    state: CircuitState,
    failure_count: u32,
    /// Number of consecutive failures needed to trip the circuit.
    failure_threshold: u32,
    success_count: u32,
    /// Number of consecutive successes needed in HalfOpen to close the circuit.
    success_threshold: u32,
    last_failure: Option<Instant>,
    /// How long the circuit stays Open before transitioning to HalfOpen.
    open_duration: Duration,
}

impl CircuitBreaker {
    /// Create a new circuit breaker.
    ///
    /// * `failure_threshold` – consecutive failures before opening (default: 5).
    /// * `open_duration` – how long to stay Open before probing recovery.
    #[must_use]
    pub fn new(failure_threshold: u32, open_duration: Duration) -> Self {
        Self {
            state: CircuitState::Closed,
            failure_count: 0,
            failure_threshold,
            success_count: 0,
            success_threshold: 1,
            last_failure: None,
            open_duration,
        }
    }

    /// Record a successful call.
    pub fn record_success(&mut self) {
        match self.state {
            CircuitState::HalfOpen => {
                self.success_count += 1;
                if self.success_count >= self.success_threshold {
                    self.state = CircuitState::Closed;
                    self.failure_count = 0;
                    self.success_count = 0;
                }
            }
            CircuitState::Closed => {
                // Reset the failure counter on success
                self.failure_count = 0;
            }
            CircuitState::Open => {
                // Shouldn't happen normally, but handle gracefully
            }
        }
    }

    /// Record a failed call.
    pub fn record_failure(&mut self) {
        self.last_failure = Some(Instant::now());
        self.success_count = 0;

        match self.state {
            CircuitState::Closed => {
                self.failure_count += 1;
                if self.failure_count >= self.failure_threshold {
                    self.state = CircuitState::Open;
                }
            }
            CircuitState::HalfOpen => {
                // Any failure in HalfOpen re-opens the circuit
                self.failure_count += 1;
                self.state = CircuitState::Open;
            }
            CircuitState::Open => {
                self.failure_count += 1;
            }
        }
    }

    /// Current circuit state.
    #[must_use]
    pub fn state(&self) -> CircuitState {
        self.state
    }

    /// Returns `true` when the circuit is Open (all calls should be blocked).
    #[must_use]
    pub fn is_open(&self) -> bool {
        self.state == CircuitState::Open
    }

    /// Cumulative failure count since the last reset or successful close.
    #[must_use]
    pub fn failure_count(&self) -> u32 {
        self.failure_count
    }

    /// Hard-reset: return to `Closed` and clear all counters.
    pub fn reset(&mut self) {
        self.state = CircuitState::Closed;
        self.failure_count = 0;
        self.success_count = 0;
        self.last_failure = None;
    }

    /// Check whether the circuit should transition from `Open` to `HalfOpen`.
    ///
    /// Call this periodically (e.g. before every blocked request) to allow
    /// recovery probing after `open_duration` has elapsed.
    pub fn check_recovery(&mut self) {
        if self.state == CircuitState::Open {
            if let Some(last) = self.last_failure {
                if last.elapsed() >= self.open_duration {
                    self.state = CircuitState::HalfOpen;
                    self.success_count = 0;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    // ------------------------------------------------------------------
    // RetryStrategy
    // ------------------------------------------------------------------

    #[test]
    fn test_no_retry_should_never_retry() {
        let s = RetryStrategy::NoRetry;
        assert!(!s.should_retry(0));
        assert!(!s.should_retry(100));
    }

    #[test]
    fn test_fixed_should_retry() {
        let s = RetryStrategy::Fixed {
            delay: Duration::from_secs(1),
            max_retries: 3,
        };
        assert!(s.should_retry(0));
        assert!(s.should_retry(1));
        assert!(s.should_retry(2));
        // At attempt == max_retries we are already past the limit
        assert!(!s.should_retry(3));
        assert!(!s.should_retry(4));
    }

    #[test]
    fn test_fixed_delay_is_constant() {
        let delay = Duration::from_secs(5);
        let s = RetryStrategy::Fixed {
            delay,
            max_retries: 3,
        };
        assert_eq!(s.delay_for_attempt(0), delay);
        assert_eq!(s.delay_for_attempt(1), delay);
        assert_eq!(s.delay_for_attempt(10), delay);
    }

    #[test]
    fn test_exponential_backoff_should_retry() {
        let s = RetryStrategy::ExponentialBackoff {
            initial_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(60),
            multiplier: 2.0,
            max_retries: 4,
        };
        assert!(s.should_retry(0));
        assert!(s.should_retry(3));
        assert!(!s.should_retry(4));
    }

    #[test]
    fn test_exponential_backoff_delay_doubles() {
        let s = RetryStrategy::ExponentialBackoff {
            initial_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(3600),
            multiplier: 2.0,
            max_retries: 5,
        };
        let d0 = s.delay_for_attempt(0); // 1s * 2^0 = 1s
        let d1 = s.delay_for_attempt(1); // 1s * 2^1 = 2s
        let d2 = s.delay_for_attempt(2); // 1s * 2^2 = 4s
        assert_eq!(d0, Duration::from_secs(1));
        assert_eq!(d1, Duration::from_secs(2));
        assert_eq!(d2, Duration::from_secs(4));
        // Doubling holds
        assert_eq!(d1, d0 * 2);
        assert_eq!(d2, d1 * 2);
    }

    #[test]
    fn test_exponential_backoff_max_delay_cap() {
        let s = RetryStrategy::ExponentialBackoff {
            initial_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(10),
            multiplier: 2.0,
            max_retries: 10,
        };
        // After many doublings the delay must stay at max_delay
        assert!(s.delay_for_attempt(20) <= Duration::from_secs(10));
    }

    #[test]
    fn test_linear_should_retry() {
        let s = RetryStrategy::Linear {
            initial_delay: Duration::from_secs(1),
            increment: Duration::from_secs(1),
            max_retries: 3,
        };
        assert!(s.should_retry(0));
        assert!(s.should_retry(2));
        assert!(!s.should_retry(3));
    }

    #[test]
    fn test_linear_delay_increases() {
        let s = RetryStrategy::Linear {
            initial_delay: Duration::from_secs(2),
            increment: Duration::from_secs(3),
            max_retries: 5,
        };
        assert_eq!(s.delay_for_attempt(0), Duration::from_secs(2)); // 2 + 3*0 = 2
        assert_eq!(s.delay_for_attempt(1), Duration::from_secs(5)); // 2 + 3*1 = 5
        assert_eq!(s.delay_for_attempt(2), Duration::from_secs(8)); // 2 + 3*2 = 8
    }

    #[test]
    fn test_strategy_max_retries() {
        assert_eq!(RetryStrategy::NoRetry.max_retries(), 0);
        assert_eq!(
            RetryStrategy::Fixed {
                delay: Duration::ZERO,
                max_retries: 7
            }
            .max_retries(),
            7
        );
        assert_eq!(
            RetryStrategy::ExponentialBackoff {
                initial_delay: Duration::ZERO,
                max_delay: Duration::ZERO,
                multiplier: 2.0,
                max_retries: 5,
            }
            .max_retries(),
            5
        );
    }

    #[test]
    fn test_strategy_name() {
        assert_eq!(RetryStrategy::NoRetry.name(), "NoRetry");
        assert_eq!(
            RetryStrategy::Fixed {
                delay: Duration::ZERO,
                max_retries: 1
            }
            .name(),
            "Fixed"
        );
        assert_eq!(
            RetryStrategy::ExponentialBackoff {
                initial_delay: Duration::ZERO,
                max_delay: Duration::ZERO,
                multiplier: 2.0,
                max_retries: 1,
            }
            .name(),
            "ExponentialBackoff"
        );
        assert_eq!(
            RetryStrategy::Linear {
                initial_delay: Duration::ZERO,
                increment: Duration::ZERO,
                max_retries: 1,
            }
            .name(),
            "Linear"
        );
    }

    // ------------------------------------------------------------------
    // RetryState
    // ------------------------------------------------------------------

    #[test]
    fn test_retry_state_new() {
        let rs = RetryState::new(RetryStrategy::NoRetry);
        assert_eq!(rs.attempt, 0);
        assert!(rs.last_error.is_none());
        assert!(rs.next_retry_at.is_none());
        assert_eq!(rs.total_delay, Duration::ZERO);
    }

    #[test]
    fn test_retry_state_record_failure_increments_attempt() {
        let mut rs = RetryState::new(RetryStrategy::Fixed {
            delay: Duration::from_millis(1),
            max_retries: 3,
        });
        let will_retry = rs.record_failure("error A");
        assert!(will_retry);
        assert_eq!(rs.attempt, 1);
        assert_eq!(rs.last_error.as_deref(), Some("error A"));
    }

    #[test]
    fn test_retry_state_returns_false_at_max_retries() {
        let mut rs = RetryState::new(RetryStrategy::Fixed {
            delay: Duration::from_millis(1),
            max_retries: 2,
        });
        assert!(rs.record_failure("e1")); // attempt 0 -> 1
        assert!(rs.record_failure("e2")); // attempt 1 -> 2
                                          // At attempt == 2 == max_retries, should_retry(2) is false
        let should = rs.record_failure("e3");
        assert!(!should);
        assert_eq!(rs.attempt, 3);
    }

    #[test]
    fn test_retry_state_no_retry_strategy() {
        let mut rs = RetryState::new(RetryStrategy::NoRetry);
        assert!(!rs.record_failure("boom"));
        assert!(rs.next_retry_at.is_none());
    }

    #[test]
    fn test_retry_state_should_retry_now() {
        let mut rs = RetryState::new(RetryStrategy::Fixed {
            delay: Duration::from_nanos(1), // near-zero delay
            max_retries: 5,
        });
        rs.record_failure("err");
        // After 1 ns the retry window should have passed
        std::thread::sleep(Duration::from_millis(5));
        assert!(rs.should_retry_now());
    }

    #[test]
    fn test_retry_state_reset() {
        let mut rs = RetryState::new(RetryStrategy::Fixed {
            delay: Duration::from_secs(1),
            max_retries: 3,
        });
        rs.record_failure("first");
        rs.reset();
        assert_eq!(rs.attempt, 0);
        assert!(rs.last_error.is_none());
        assert!(rs.next_retry_at.is_none());
        assert_eq!(rs.total_delay, Duration::ZERO);
    }

    // ------------------------------------------------------------------
    // CircuitBreaker
    // ------------------------------------------------------------------

    #[test]
    fn test_circuit_breaker_starts_closed() {
        let cb = CircuitBreaker::new(5, Duration::from_secs(60));
        assert_eq!(cb.state(), CircuitState::Closed);
        assert!(!cb.is_open());
        assert_eq!(cb.failure_count(), 0);
    }

    #[test]
    fn test_circuit_breaker_opens_after_threshold_failures() {
        let mut cb = CircuitBreaker::new(3, Duration::from_secs(60));
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Closed);
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Closed);
        cb.record_failure(); // 3rd failure reaches threshold
        assert_eq!(cb.state(), CircuitState::Open);
        assert!(cb.is_open());
    }

    #[test]
    fn test_circuit_breaker_half_open_after_duration() {
        let mut cb = CircuitBreaker::new(1, Duration::from_nanos(1));
        cb.record_failure(); // opens immediately
        assert_eq!(cb.state(), CircuitState::Open);

        std::thread::sleep(Duration::from_millis(5));
        cb.check_recovery();
        assert_eq!(cb.state(), CircuitState::HalfOpen);
    }

    #[test]
    fn test_circuit_breaker_closes_after_successes_in_half_open() {
        let mut cb = CircuitBreaker::new(1, Duration::from_nanos(1));
        cb.record_failure();
        std::thread::sleep(Duration::from_millis(5));
        cb.check_recovery(); // -> HalfOpen

        cb.record_success(); // success_threshold == 1 -> Closed
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn test_circuit_breaker_re_opens_on_failure_in_half_open() {
        let mut cb = CircuitBreaker::new(1, Duration::from_nanos(1));
        cb.record_failure();
        std::thread::sleep(Duration::from_millis(5));
        cb.check_recovery(); // -> HalfOpen

        cb.record_failure(); // any failure re-opens
        assert_eq!(cb.state(), CircuitState::Open);
    }

    #[test]
    fn test_circuit_breaker_reset() {
        let mut cb = CircuitBreaker::new(2, Duration::from_secs(60));
        cb.record_failure();
        cb.record_failure(); // -> Open
        assert!(cb.is_open());

        cb.reset();
        assert_eq!(cb.state(), CircuitState::Closed);
        assert_eq!(cb.failure_count(), 0);
        assert!(!cb.is_open());
    }

    #[test]
    fn test_circuit_breaker_success_resets_failure_count_in_closed() {
        let mut cb = CircuitBreaker::new(3, Duration::from_secs(60));
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.failure_count(), 2);
        // A success while Closed resets the streak
        cb.record_success();
        assert_eq!(cb.failure_count(), 0);
    }
}
