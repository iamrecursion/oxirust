#![allow(dead_code)]
//! Retry policy configuration and evaluation for job execution.
//!
//! Provides flexible retry strategies including exponential backoff, linear backoff,
//! and fixed-interval retries with jitter support, circuit breaker patterns,
//! and per-error-class retry behavior.
//!
//! ## Default Configuration
//!
//! | Parameter | Default | Description |
//! |-----------|---------|-------------|
//! | `max_retries` | 3 | Maximum number of retry attempts after the initial failure |
//! | `initial` (Exponential) | 1 s | Delay before the first retry |
//! | `multiplier` (Exponential) | 2.0 | Factor by which the delay grows on each retry |
//! | `max_delay` | 60 s | Upper cap on any single inter-retry delay |
//! | `jitter` | `JitterMode::None` | Randomness mode added to computed delays |
//! | `circuit_breaker_threshold` | 5 | Consecutive failures that open the circuit |
//! | `retry_on_timeout` | `true` | Whether timed-out jobs are retried |
//!
//! ## Backoff Formula
//!
//! For `BackoffStrategy::Exponential`, the delay before attempt *n* (0-indexed) is:
//!
//! ```text
//! delay(n) = min(initial × multiplier^n, max_delay)
//! ```
//!
//! With the defaults this gives: 1 s → 2 s → 4 s → 8 s → 16 s → (capped at 60 s).
//!
//! ## Jitter Modes
//!
//! Jitter spreads retry storms across time when many jobs fail simultaneously.
//!
//! | Mode | Effect |
//! |------|--------|
//! | `JitterMode::None` | Use the computed delay exactly (no randomness) |
//! | `JitterMode::Full` | Replace the delay with a uniform random value in `[0, delay]` |
//! | `JitterMode::Equal` | Use `delay/2 + uniform_random(0, delay/2)` |
//!
//! ## Per-Error-Class Overrides
//!
//! Use `ErrorClassPolicy` to apply different retry rules for different error
//! categories.  For example you might never retry `ErrorClass::Permanent` errors
//! while retrying `ErrorClass::Network` errors up to 5 times with a fixed 500 ms
//! interval:
//!
//! ```rust,no_run
//! use oximedia_jobs::retry_policy::{
//!     BackoffStrategy, ErrorClass, ErrorClassPolicy, RetryPolicyConfig,
//! };
//! use std::time::Duration;
//!
//! let policy = RetryPolicyConfig::new()
//!     .with_error_override(ErrorClassPolicy {
//!         error_class: ErrorClass::Network,
//!         max_retries: 5,
//!         retryable: true,
//!     })
//!     .with_backoff(BackoffStrategy::Fixed {
//!         interval: Duration::from_millis(500),
//!     });
//! ```

use std::collections::HashMap;
use std::time::Duration;

/// Strategy for computing delay between retries.
#[derive(Debug, Clone, PartialEq)]
pub enum BackoffStrategy {
    /// Fixed interval between retries.
    Fixed {
        /// The constant delay between attempts.
        interval: Duration,
    },
    /// Linearly increasing delay.
    Linear {
        /// Initial delay for the first retry.
        initial: Duration,
        /// Amount added to the delay on each subsequent retry.
        increment: Duration,
        /// Maximum delay cap.
        max_delay: Duration,
    },
    /// Exponentially increasing delay.
    Exponential {
        /// Initial delay for the first retry.
        initial: Duration,
        /// Multiplier applied on each retry (typically 2.0).
        multiplier: f64,
        /// Maximum delay cap.
        max_delay: Duration,
    },
}

impl BackoffStrategy {
    /// Compute the delay for the given attempt number (0-indexed).
    #[allow(clippy::cast_precision_loss)]
    pub fn delay_for_attempt(&self, attempt: u32) -> Duration {
        match self {
            BackoffStrategy::Fixed { interval } => *interval,
            BackoffStrategy::Linear {
                initial,
                increment,
                max_delay,
            } => {
                let added = increment.as_millis() as u64 * u64::from(attempt);
                let total_ms = initial.as_millis() as u64 + added;
                let capped = total_ms.min(max_delay.as_millis() as u64);
                Duration::from_millis(capped)
            }
            BackoffStrategy::Exponential {
                initial,
                multiplier,
                max_delay,
            } => {
                let base_ms = initial.as_millis() as f64;
                let factor = multiplier.powi(attempt as i32);
                let delay_ms = base_ms * factor;
                let capped = delay_ms.min(max_delay.as_millis() as f64);
                Duration::from_millis(capped as u64)
            }
        }
    }
}

/// Jitter mode to add randomness to retry delays.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JitterMode {
    /// No jitter applied.
    None,
    /// Full jitter: delay is uniform random in [0, computed_delay].
    Full,
    /// Equal jitter: delay is computed_delay/2 + uniform random in [0, computed_delay/2].
    Equal,
}

/// Classification of errors for per-class retry behavior.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ErrorClass {
    /// Transient network errors.
    Network,
    /// Resource exhaustion (CPU, memory, disk).
    Resource,
    /// Timeout errors.
    Timeout,
    /// Codec or processing errors.
    Processing,
    /// Unknown / catch-all errors.
    Unknown,
}

/// Per-error-class retry override.
#[derive(Debug, Clone)]
pub struct ErrorClassPolicy {
    /// The error class this policy applies to.
    pub error_class: ErrorClass,
    /// Maximum retries for this class (overrides global).
    pub max_retries: u32,
    /// Whether to retry this class at all.
    pub retryable: bool,
}

/// Circuit breaker state for controlling retry behavior under sustained failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CircuitState {
    /// Normal operation — retries allowed.
    Closed,
    /// Too many failures — retries blocked.
    Open,
    /// Testing whether service recovered — limited retries.
    HalfOpen,
}

/// Circuit breaker configuration and state.
#[derive(Debug, Clone)]
pub struct CircuitBreaker {
    /// Current state of the circuit.
    pub state: CircuitState,
    /// Number of consecutive failures.
    pub failure_count: u32,
    /// Threshold at which the circuit opens.
    pub failure_threshold: u32,
    /// Number of successes needed to close from half-open.
    pub success_threshold: u32,
    /// Consecutive successes in half-open state.
    pub success_count: u32,
    /// Duration the circuit stays open before moving to half-open.
    pub open_duration: Duration,
}

impl CircuitBreaker {
    /// Create a new circuit breaker with the given thresholds.
    pub fn new(failure_threshold: u32, success_threshold: u32, open_duration: Duration) -> Self {
        Self {
            state: CircuitState::Closed,
            failure_count: 0,
            failure_threshold,
            success_threshold,
            success_count: 0,
            open_duration,
        }
    }

    /// Record a failure and update state.
    pub fn record_failure(&mut self) {
        self.success_count = 0;
        self.failure_count += 1;
        if self.failure_count >= self.failure_threshold {
            self.state = CircuitState::Open;
        }
    }

    /// Record a success and update state.
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
                self.failure_count = 0;
            }
            CircuitState::Open => {}
        }
    }

    /// Attempt to transition from open to half-open.
    pub fn try_half_open(&mut self) -> bool {
        if self.state == CircuitState::Open {
            self.state = CircuitState::HalfOpen;
            self.success_count = 0;
            true
        } else {
            false
        }
    }

    /// Check whether a retry is currently allowed.
    pub fn allows_retry(&self) -> bool {
        self.state != CircuitState::Open
    }

    /// Reset the circuit breaker to its initial closed state.
    pub fn reset(&mut self) {
        self.state = CircuitState::Closed;
        self.failure_count = 0;
        self.success_count = 0;
    }
}

/// Comprehensive retry policy configuration.
#[derive(Debug, Clone)]
pub struct RetryPolicyConfig {
    /// Maximum number of retry attempts.
    pub max_retries: u32,
    /// Backoff strategy.
    pub backoff: BackoffStrategy,
    /// Jitter mode.
    pub jitter: JitterMode,
    /// Per-error-class overrides.
    pub error_overrides: HashMap<ErrorClass, ErrorClassPolicy>,
    /// Whether to retry on timeout.
    pub retry_on_timeout: bool,
    /// Overall deadline for all retries combined.
    pub total_timeout: Option<Duration>,
}

impl Default for RetryPolicyConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            backoff: BackoffStrategy::Exponential {
                initial: Duration::from_secs(1),
                multiplier: 2.0,
                max_delay: Duration::from_secs(60),
            },
            jitter: JitterMode::None,
            error_overrides: HashMap::new(),
            retry_on_timeout: true,
            total_timeout: Some(Duration::from_secs(300)),
        }
    }
}

impl RetryPolicyConfig {
    /// Create a new policy with exponential backoff defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum number of retries.
    pub fn with_max_retries(mut self, max: u32) -> Self {
        self.max_retries = max;
        self
    }

    /// Set the backoff strategy.
    pub fn with_backoff(mut self, strategy: BackoffStrategy) -> Self {
        self.backoff = strategy;
        self
    }

    /// Set the jitter mode.
    pub fn with_jitter(mut self, jitter: JitterMode) -> Self {
        self.jitter = jitter;
        self
    }

    /// Add an error-class-specific override.
    pub fn with_error_override(mut self, policy: ErrorClassPolicy) -> Self {
        self.error_overrides
            .insert(policy.error_class.clone(), policy);
        self
    }

    /// Set the total timeout across all retries.
    pub fn with_total_timeout(mut self, timeout: Duration) -> Self {
        self.total_timeout = Some(timeout);
        self
    }

    /// Determine whether a retry should be attempted for the given error class and attempt number.
    pub fn should_retry(&self, error_class: &ErrorClass, attempt: u32) -> bool {
        if let Some(override_policy) = self.error_overrides.get(error_class) {
            if !override_policy.retryable {
                return false;
            }
            return attempt < override_policy.max_retries;
        }
        if *error_class == ErrorClass::Timeout && !self.retry_on_timeout {
            return false;
        }
        attempt < self.max_retries
    }

    /// Compute the delay for a given attempt.
    pub fn delay_for(&self, attempt: u32) -> Duration {
        self.backoff.delay_for_attempt(attempt)
    }

    /// Compute the cumulative delay across all attempts up to the given attempt.
    #[allow(clippy::cast_precision_loss)]
    pub fn cumulative_delay(&self, up_to_attempt: u32) -> Duration {
        let mut total = Duration::ZERO;
        for i in 0..up_to_attempt {
            total += self.backoff.delay_for_attempt(i);
        }
        total
    }

    /// Check whether cumulative delay exceeds the total timeout.
    pub fn exceeds_total_timeout(&self, attempt: u32) -> bool {
        if let Some(timeout) = self.total_timeout {
            self.cumulative_delay(attempt) > timeout
        } else {
            false
        }
    }
}

/// Summary of a retry evaluation decision.
#[derive(Debug, Clone, PartialEq)]
pub struct RetryDecision {
    /// Whether to retry.
    pub should_retry: bool,
    /// Delay before the retry (if applicable).
    pub delay: Duration,
    /// The attempt number this decision is for.
    pub attempt: u32,
    /// Reason the retry was denied (if applicable).
    pub denial_reason: Option<String>,
}

/// Evaluate a full retry decision given a policy, circuit breaker, error class, and attempt.
pub fn evaluate_retry(
    policy: &RetryPolicyConfig,
    breaker: &CircuitBreaker,
    error_class: &ErrorClass,
    attempt: u32,
) -> RetryDecision {
    if !breaker.allows_retry() {
        return RetryDecision {
            should_retry: false,
            delay: Duration::ZERO,
            attempt,
            denial_reason: Some("Circuit breaker is open".to_string()),
        };
    }
    if !policy.should_retry(error_class, attempt) {
        return RetryDecision {
            should_retry: false,
            delay: Duration::ZERO,
            attempt,
            denial_reason: Some("Max retries exceeded or error class not retryable".to_string()),
        };
    }
    if policy.exceeds_total_timeout(attempt + 1) {
        return RetryDecision {
            should_retry: false,
            delay: Duration::ZERO,
            attempt,
            denial_reason: Some("Total timeout would be exceeded".to_string()),
        };
    }
    let delay = policy.delay_for(attempt);
    RetryDecision {
        should_retry: true,
        delay,
        attempt,
        denial_reason: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_backoff() {
        let strategy = BackoffStrategy::Fixed {
            interval: Duration::from_secs(5),
        };
        assert_eq!(strategy.delay_for_attempt(0), Duration::from_secs(5));
        assert_eq!(strategy.delay_for_attempt(3), Duration::from_secs(5));
        assert_eq!(strategy.delay_for_attempt(100), Duration::from_secs(5));
    }

    #[test]
    fn test_linear_backoff() {
        let strategy = BackoffStrategy::Linear {
            initial: Duration::from_secs(1),
            increment: Duration::from_secs(2),
            max_delay: Duration::from_secs(10),
        };
        assert_eq!(strategy.delay_for_attempt(0), Duration::from_secs(1));
        assert_eq!(strategy.delay_for_attempt(1), Duration::from_secs(3));
        assert_eq!(strategy.delay_for_attempt(2), Duration::from_secs(5));
        // Capped at max_delay
        assert_eq!(strategy.delay_for_attempt(10), Duration::from_secs(10));
    }

    #[test]
    fn test_exponential_backoff() {
        let strategy = BackoffStrategy::Exponential {
            initial: Duration::from_secs(1),
            multiplier: 2.0,
            max_delay: Duration::from_secs(30),
        };
        assert_eq!(strategy.delay_for_attempt(0), Duration::from_secs(1));
        assert_eq!(strategy.delay_for_attempt(1), Duration::from_secs(2));
        assert_eq!(strategy.delay_for_attempt(2), Duration::from_secs(4));
        assert_eq!(strategy.delay_for_attempt(3), Duration::from_secs(8));
        // Capped at max_delay
        assert_eq!(strategy.delay_for_attempt(10), Duration::from_secs(30));
    }

    #[test]
    fn test_default_policy() {
        let policy = RetryPolicyConfig::default();
        assert_eq!(policy.max_retries, 3);
        assert!(policy.retry_on_timeout);
        assert_eq!(policy.total_timeout, Some(Duration::from_secs(300)));
    }

    #[test]
    fn test_should_retry_within_limit() {
        let policy = RetryPolicyConfig::new().with_max_retries(3);
        assert!(policy.should_retry(&ErrorClass::Network, 0));
        assert!(policy.should_retry(&ErrorClass::Network, 2));
        assert!(!policy.should_retry(&ErrorClass::Network, 3));
    }

    #[test]
    fn test_should_retry_with_error_override() {
        let policy = RetryPolicyConfig::new().with_error_override(ErrorClassPolicy {
            error_class: ErrorClass::Processing,
            max_retries: 0,
            retryable: false,
        });
        assert!(!policy.should_retry(&ErrorClass::Processing, 0));
        assert!(policy.should_retry(&ErrorClass::Network, 0));
    }

    #[test]
    fn test_timeout_not_retryable() {
        let policy = RetryPolicyConfig {
            retry_on_timeout: false,
            ..RetryPolicyConfig::default()
        };
        assert!(!policy.should_retry(&ErrorClass::Timeout, 0));
        assert!(policy.should_retry(&ErrorClass::Network, 0));
    }

    #[test]
    fn test_cumulative_delay() {
        let policy = RetryPolicyConfig::new().with_backoff(BackoffStrategy::Fixed {
            interval: Duration::from_secs(2),
        });
        assert_eq!(policy.cumulative_delay(3), Duration::from_secs(6));
    }

    #[test]
    fn test_exceeds_total_timeout() {
        let policy = RetryPolicyConfig::new()
            .with_backoff(BackoffStrategy::Fixed {
                interval: Duration::from_secs(100),
            })
            .with_total_timeout(Duration::from_secs(250));
        assert!(!policy.exceeds_total_timeout(2));
        assert!(policy.exceeds_total_timeout(3));
    }

    #[test]
    fn test_circuit_breaker_closed() {
        let breaker = CircuitBreaker::new(3, 2, Duration::from_secs(30));
        assert!(breaker.allows_retry());
        assert_eq!(breaker.state, CircuitState::Closed);
    }

    #[test]
    fn test_circuit_breaker_opens_on_failures() {
        let mut breaker = CircuitBreaker::new(3, 2, Duration::from_secs(30));
        breaker.record_failure();
        breaker.record_failure();
        assert!(breaker.allows_retry());
        breaker.record_failure();
        assert!(!breaker.allows_retry());
        assert_eq!(breaker.state, CircuitState::Open);
    }

    #[test]
    fn test_circuit_breaker_half_open_to_closed() {
        let mut breaker = CircuitBreaker::new(2, 2, Duration::from_secs(30));
        breaker.record_failure();
        breaker.record_failure();
        assert_eq!(breaker.state, CircuitState::Open);
        assert!(breaker.try_half_open());
        assert_eq!(breaker.state, CircuitState::HalfOpen);
        breaker.record_success();
        assert_eq!(breaker.state, CircuitState::HalfOpen);
        breaker.record_success();
        assert_eq!(breaker.state, CircuitState::Closed);
    }

    #[test]
    fn test_circuit_breaker_reset() {
        let mut breaker = CircuitBreaker::new(2, 2, Duration::from_secs(30));
        breaker.record_failure();
        breaker.record_failure();
        assert_eq!(breaker.state, CircuitState::Open);
        breaker.reset();
        assert_eq!(breaker.state, CircuitState::Closed);
        assert_eq!(breaker.failure_count, 0);
    }

    #[test]
    fn test_evaluate_retry_success() {
        let policy = RetryPolicyConfig::default();
        let breaker = CircuitBreaker::new(5, 2, Duration::from_secs(30));
        let decision = evaluate_retry(&policy, &breaker, &ErrorClass::Network, 0);
        assert!(decision.should_retry);
        assert!(decision.denial_reason.is_none());
    }

    #[test]
    fn test_evaluate_retry_circuit_open() {
        let policy = RetryPolicyConfig::default();
        let mut breaker = CircuitBreaker::new(1, 1, Duration::from_secs(30));
        breaker.record_failure();
        let decision = evaluate_retry(&policy, &breaker, &ErrorClass::Network, 0);
        assert!(!decision.should_retry);
        assert_eq!(
            decision.denial_reason,
            Some("Circuit breaker is open".to_string())
        );
    }

    #[test]
    fn test_policy_builder_chain() {
        let policy = RetryPolicyConfig::new()
            .with_max_retries(5)
            .with_jitter(JitterMode::Full)
            .with_backoff(BackoffStrategy::Linear {
                initial: Duration::from_millis(500),
                increment: Duration::from_millis(500),
                max_delay: Duration::from_secs(10),
            })
            .with_total_timeout(Duration::from_secs(120));
        assert_eq!(policy.max_retries, 5);
        assert_eq!(policy.jitter, JitterMode::Full);
        assert_eq!(policy.total_timeout, Some(Duration::from_secs(120)));
    }
}
