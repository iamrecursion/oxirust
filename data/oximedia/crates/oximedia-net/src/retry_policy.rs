#![allow(dead_code)]
//! Retry policies for network operations.
//!
//! Provides configurable retry behaviour with exponential back-off, jitter,
//! and per-error-class decisions. The main entry point is [`RetryPolicy`].

use std::time::Duration;

/// Classification of an error for retry decision purposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorClass {
    /// Transient error — safe to retry (e.g. timeout, 503).
    Transient,
    /// Permanent error — retrying will not help (e.g. 404, 400).
    Permanent,
    /// Rate-limited — should back off longer before retrying.
    RateLimited,
    /// Connection refused — peer is not listening.
    ConnectionRefused,
    /// Unknown classification; defaults to transient handling.
    Unknown,
}

impl ErrorClass {
    /// Returns `true` if this class of error should be retried.
    #[must_use]
    pub fn should_retry(self) -> bool {
        matches!(
            self,
            Self::Transient | Self::RateLimited | Self::ConnectionRefused | Self::Unknown
        )
    }

    /// Returns a multiplier applied to the base delay for this error class.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn delay_multiplier(self) -> f64 {
        match self {
            Self::RateLimited => 3.0,
            Self::ConnectionRefused => 2.0,
            Self::Transient | Self::Unknown => 1.0,
            Self::Permanent => 0.0,
        }
    }
}

/// Back-off strategy used between retry attempts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackoffStrategy {
    /// Constant delay between retries.
    Constant,
    /// Exponential back-off (delay doubles each attempt).
    Exponential,
    /// Linear back-off (delay increases by `base_delay` each attempt).
    Linear,
}

/// Configuration for a retry policy.
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    /// Maximum number of retry attempts (excluding the initial try).
    pub max_retries: u32,
    /// Base delay before the first retry.
    pub base_delay: Duration,
    /// Maximum delay cap.
    pub max_delay: Duration,
    /// Back-off strategy.
    pub strategy: BackoffStrategy,
    /// Jitter factor in `[0.0, 1.0]`. 0 means no jitter.
    pub jitter_factor: f64,
    /// Whether to retry on permanent errors (usually `false`).
    pub retry_permanent: bool,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay: Duration::from_millis(500),
            max_delay: Duration::from_secs(30),
            strategy: BackoffStrategy::Exponential,
            jitter_factor: 0.25,
            retry_permanent: false,
        }
    }
}

impl RetryPolicy {
    /// Creates a new policy with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the maximum number of retries.
    #[must_use]
    pub const fn with_max_retries(mut self, n: u32) -> Self {
        self.max_retries = n;
        self
    }

    /// Sets the base delay.
    #[must_use]
    pub const fn with_base_delay(mut self, d: Duration) -> Self {
        self.base_delay = d;
        self
    }

    /// Sets the maximum delay cap.
    #[must_use]
    pub const fn with_max_delay(mut self, d: Duration) -> Self {
        self.max_delay = d;
        self
    }

    /// Sets the back-off strategy.
    #[must_use]
    pub const fn with_strategy(mut self, s: BackoffStrategy) -> Self {
        self.strategy = s;
        self
    }

    /// Sets the jitter factor (clamped to `[0.0, 1.0]`).
    #[must_use]
    pub fn with_jitter(mut self, factor: f64) -> Self {
        self.jitter_factor = factor.clamp(0.0, 1.0);
        self
    }

    /// Decides whether to retry the `attempt`-th failure for the given error class.
    ///
    /// `attempt` is 1-based (1 = first failure).
    #[must_use]
    pub fn should_retry(&self, attempt: u32, error_class: ErrorClass) -> bool {
        if attempt > self.max_retries {
            return false;
        }
        if error_class == ErrorClass::Permanent && !self.retry_permanent {
            return false;
        }
        error_class.should_retry()
    }

    /// Computes the delay before the `attempt`-th retry (1-based).
    ///
    /// Does **not** include jitter — call [`apply_jitter`](Self::apply_jitter) separately
    /// if randomisation is desired.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn compute_delay(&self, attempt: u32, error_class: ErrorClass) -> Duration {
        let base_ms = self.base_delay.as_millis() as f64;
        let multiplier = error_class.delay_multiplier();
        let attempt_idx = attempt.saturating_sub(1) as f64;

        let raw_ms = match self.strategy {
            BackoffStrategy::Constant => base_ms,
            BackoffStrategy::Exponential => base_ms * 2.0_f64.powf(attempt_idx),
            BackoffStrategy::Linear => base_ms * (1.0 + attempt_idx),
        };

        let adjusted = raw_ms * multiplier;
        let max_ms = self.max_delay.as_millis() as f64;
        let clamped = adjusted.min(max_ms);

        Duration::from_millis(clamped as u64)
    }

    /// Applies jitter to a delay by scaling it within `[delay * (1 - jitter), delay]`.
    ///
    /// `random_01` should be a uniform random value in `[0.0, 1.0]`.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn apply_jitter(&self, delay: Duration, random_01: f64) -> Duration {
        if self.jitter_factor <= 0.0 {
            return delay;
        }
        let ms = delay.as_millis() as f64;
        let jitter_range = ms * self.jitter_factor;
        let jittered = ms - jitter_range + (random_01.clamp(0.0, 1.0) * jitter_range);
        Duration::from_millis(jittered.max(0.0) as u64)
    }
}

/// Outcome of a retry decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RetryOutcome {
    /// Retry after waiting the specified delay.
    RetryAfter(Duration),
    /// Do not retry; give up.
    GiveUp,
}

/// Tracks the state of an ongoing retry sequence.
#[derive(Debug, Clone)]
pub struct RetryState {
    /// The policy governing retries.
    policy: RetryPolicy,
    /// Current attempt number (starts at 0; first failure = attempt 1).
    attempt: u32,
    /// Total time spent waiting so far.
    total_wait: Duration,
}

impl RetryState {
    /// Creates a new retry state from a policy.
    #[must_use]
    pub fn new(policy: RetryPolicy) -> Self {
        Self {
            policy,
            attempt: 0,
            total_wait: Duration::ZERO,
        }
    }

    /// Returns the current attempt number.
    #[must_use]
    pub const fn attempt(&self) -> u32 {
        self.attempt
    }

    /// Returns the total accumulated wait time.
    #[must_use]
    pub const fn total_wait(&self) -> Duration {
        self.total_wait
    }

    /// Records a failure and returns the retry decision.
    pub fn record_failure(&mut self, error_class: ErrorClass) -> RetryOutcome {
        self.attempt += 1;
        if !self.policy.should_retry(self.attempt, error_class) {
            return RetryOutcome::GiveUp;
        }
        let delay = self.policy.compute_delay(self.attempt, error_class);
        self.total_wait += delay;
        RetryOutcome::RetryAfter(delay)
    }

    /// Resets the retry state so the sequence can start fresh.
    pub fn reset(&mut self) {
        self.attempt = 0;
        self.total_wait = Duration::ZERO;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_class_should_retry() {
        assert!(ErrorClass::Transient.should_retry());
        assert!(ErrorClass::RateLimited.should_retry());
        assert!(ErrorClass::ConnectionRefused.should_retry());
        assert!(ErrorClass::Unknown.should_retry());
        assert!(!ErrorClass::Permanent.should_retry());
    }

    #[test]
    fn test_error_class_delay_multiplier() {
        assert!((ErrorClass::Transient.delay_multiplier() - 1.0).abs() < f64::EPSILON);
        assert!((ErrorClass::RateLimited.delay_multiplier() - 3.0).abs() < f64::EPSILON);
        assert!((ErrorClass::ConnectionRefused.delay_multiplier() - 2.0).abs() < f64::EPSILON);
        assert!((ErrorClass::Permanent.delay_multiplier()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_default_policy() {
        let p = RetryPolicy::default();
        assert_eq!(p.max_retries, 3);
        assert_eq!(p.base_delay, Duration::from_millis(500));
        assert_eq!(p.strategy, BackoffStrategy::Exponential);
    }

    #[test]
    fn test_should_retry_within_limit() {
        let p = RetryPolicy::new().with_max_retries(2);
        assert!(p.should_retry(1, ErrorClass::Transient));
        assert!(p.should_retry(2, ErrorClass::Transient));
        assert!(!p.should_retry(3, ErrorClass::Transient));
    }

    #[test]
    fn test_should_not_retry_permanent() {
        let p = RetryPolicy::new();
        assert!(!p.should_retry(1, ErrorClass::Permanent));
    }

    #[test]
    fn test_exponential_backoff_delays() {
        let p = RetryPolicy::new()
            .with_base_delay(Duration::from_millis(100))
            .with_max_delay(Duration::from_secs(60))
            .with_strategy(BackoffStrategy::Exponential);

        let d1 = p.compute_delay(1, ErrorClass::Transient);
        let d2 = p.compute_delay(2, ErrorClass::Transient);
        let d3 = p.compute_delay(3, ErrorClass::Transient);

        assert_eq!(d1, Duration::from_millis(100));
        assert_eq!(d2, Duration::from_millis(200));
        assert_eq!(d3, Duration::from_millis(400));
    }

    #[test]
    fn test_linear_backoff_delays() {
        let p = RetryPolicy::new()
            .with_base_delay(Duration::from_millis(100))
            .with_strategy(BackoffStrategy::Linear);

        assert_eq!(
            p.compute_delay(1, ErrorClass::Transient),
            Duration::from_millis(100)
        );
        assert_eq!(
            p.compute_delay(2, ErrorClass::Transient),
            Duration::from_millis(200)
        );
        assert_eq!(
            p.compute_delay(3, ErrorClass::Transient),
            Duration::from_millis(300)
        );
    }

    #[test]
    fn test_constant_backoff() {
        let p = RetryPolicy::new()
            .with_base_delay(Duration::from_millis(250))
            .with_strategy(BackoffStrategy::Constant);

        assert_eq!(
            p.compute_delay(1, ErrorClass::Transient),
            Duration::from_millis(250)
        );
        assert_eq!(
            p.compute_delay(5, ErrorClass::Transient),
            Duration::from_millis(250)
        );
    }

    #[test]
    fn test_delay_capped_at_max() {
        let p = RetryPolicy::new()
            .with_base_delay(Duration::from_secs(10))
            .with_max_delay(Duration::from_secs(15))
            .with_strategy(BackoffStrategy::Exponential);

        // attempt 3: 10 * 2^2 = 40s, capped to 15s
        let d = p.compute_delay(3, ErrorClass::Transient);
        assert_eq!(d, Duration::from_secs(15));
    }

    #[test]
    fn test_rate_limited_extra_delay() {
        let p = RetryPolicy::new()
            .with_base_delay(Duration::from_millis(100))
            .with_max_delay(Duration::from_secs(60))
            .with_strategy(BackoffStrategy::Constant);

        let d = p.compute_delay(1, ErrorClass::RateLimited);
        assert_eq!(d, Duration::from_millis(300)); // 100 * 3.0
    }

    #[test]
    fn test_jitter_zero_factor() {
        let p = RetryPolicy::new().with_jitter(0.0);
        let delay = Duration::from_millis(1000);
        let result = p.apply_jitter(delay, 0.5);
        assert_eq!(result, delay);
    }

    #[test]
    fn test_jitter_reduces_delay() {
        let p = RetryPolicy::new().with_jitter(0.5);
        let delay = Duration::from_millis(1000);
        // random_01 = 0 → delay * (1 - 0.5) = 500
        let result = p.apply_jitter(delay, 0.0);
        assert_eq!(result, Duration::from_millis(500));
        // random_01 = 1 → delay * 1.0 = 1000
        let result_max = p.apply_jitter(delay, 1.0);
        assert_eq!(result_max, Duration::from_millis(1000));
    }

    #[test]
    fn test_retry_state_give_up() {
        let policy = RetryPolicy::new().with_max_retries(1);
        let mut state = RetryState::new(policy);
        let r1 = state.record_failure(ErrorClass::Transient);
        assert!(matches!(r1, RetryOutcome::RetryAfter(_)));
        let r2 = state.record_failure(ErrorClass::Transient);
        assert_eq!(r2, RetryOutcome::GiveUp);
    }

    #[test]
    fn test_retry_state_reset() {
        let policy = RetryPolicy::new().with_max_retries(1);
        let mut state = RetryState::new(policy);
        state.record_failure(ErrorClass::Transient);
        state.record_failure(ErrorClass::Transient);
        assert_eq!(state.attempt(), 2);

        state.reset();
        assert_eq!(state.attempt(), 0);
        assert_eq!(state.total_wait(), Duration::ZERO);
    }

    #[test]
    fn test_retry_state_accumulates_wait() {
        let policy = RetryPolicy::new()
            .with_base_delay(Duration::from_millis(100))
            .with_max_retries(3)
            .with_strategy(BackoffStrategy::Constant);
        let mut state = RetryState::new(policy);

        state.record_failure(ErrorClass::Transient);
        assert_eq!(state.total_wait(), Duration::from_millis(100));

        state.record_failure(ErrorClass::Transient);
        assert_eq!(state.total_wait(), Duration::from_millis(200));
    }

    #[test]
    fn test_builder_chaining() {
        let p = RetryPolicy::new()
            .with_max_retries(5)
            .with_base_delay(Duration::from_millis(200))
            .with_max_delay(Duration::from_secs(10))
            .with_strategy(BackoffStrategy::Linear)
            .with_jitter(0.3);
        assert_eq!(p.max_retries, 5);
        assert_eq!(p.base_delay, Duration::from_millis(200));
        assert_eq!(p.max_delay, Duration::from_secs(10));
        assert_eq!(p.strategy, BackoffStrategy::Linear);
        assert!((p.jitter_factor - 0.3).abs() < f64::EPSILON);
    }
}
