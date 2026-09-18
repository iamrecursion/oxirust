//! Retry policies and tracking for batch job execution.
//!
//! Supports fixed-delay, exponential back-off, and no-retry strategies,
//! with a `RetryTracker` that records attempts and signals exhaustion.

#![allow(dead_code)]

/// Strategy used to compute delays between retry attempts.
#[derive(Debug, Clone, PartialEq)]
pub enum RetryStrategy {
    /// Never retry; fail immediately on first error.
    NoRetry,
    /// Wait a fixed number of milliseconds between every attempt.
    Fixed {
        /// Delay in milliseconds between each retry.
        delay_ms: u64,
    },
    /// Double the delay on each attempt, starting from `initial_ms`.
    Exponential {
        /// Starting delay in milliseconds for the first retry.
        initial_ms: u64,
        /// Maximum delay in milliseconds; caps exponential growth.
        max_ms: u64,
    },
    /// Exponential backoff with decorrelated jitter to prevent thundering herd.
    ///
    /// Uses the "full jitter" algorithm: `delay = random(0, min(max_ms, initial_ms * 2^attempt))`.
    /// This spreads retries uniformly across the window, eliminating synchronised
    /// retry storms when many jobs fail at the same time.
    ExponentialWithJitter {
        /// Starting delay in milliseconds for the first retry.
        initial_ms: u64,
        /// Maximum delay in milliseconds; caps exponential growth.
        max_ms: u64,
        /// Jitter mode.
        jitter: JitterMode,
    },
    /// Immediate retry with no delay.
    Immediate,
}

/// Jitter algorithms for exponential backoff.
///
/// All modes compute a base delay of `initial_ms * 2^attempt` (capped at `max_ms`),
/// then apply a randomisation strategy to spread retries over the window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JitterMode {
    /// Full jitter: `uniform(0, base_delay)`.  Maximum spread; best for
    /// preventing thundering herd when many clients retry simultaneously.
    Full,
    /// Equal jitter: `base_delay / 2 + uniform(0, base_delay / 2)`.
    /// Guarantees at least half the base delay, limiting how aggressive the
    /// retry can be while still providing good spread.
    Equal,
    /// Decorrelated jitter: `min(max_ms, uniform(initial_ms, prev_delay * 3))`.
    /// Each retry's delay depends on the previous one rather than the attempt
    /// number, producing a smoothly varying sequence.
    Decorrelated,
}

/// Configuration for a retry policy.
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Retry strategy controlling delay calculation
    pub strategy: RetryStrategy,
    /// Maximum number of retry attempts (0 means no retries)
    pub max_attempts: u32,
}

impl RetryConfig {
    /// Creates a no-retry configuration.
    #[must_use]
    pub fn no_retry() -> Self {
        Self {
            strategy: RetryStrategy::NoRetry,
            max_attempts: 0,
        }
    }

    /// Creates a fixed-delay retry configuration.
    #[must_use]
    pub fn fixed(max_attempts: u32, delay_ms: u64) -> Self {
        Self {
            strategy: RetryStrategy::Fixed { delay_ms },
            max_attempts,
        }
    }

    /// Creates an exponential back-off retry configuration.
    #[must_use]
    pub fn exponential(max_attempts: u32, initial_ms: u64, max_ms: u64) -> Self {
        Self {
            strategy: RetryStrategy::Exponential { initial_ms, max_ms },
            max_attempts,
        }
    }

    /// Creates an exponential back-off with full jitter configuration.
    ///
    /// Full jitter spreads retries uniformly across `[0, base_delay]` to prevent
    /// thundering herd when many jobs fail simultaneously.
    #[must_use]
    pub fn exponential_full_jitter(max_attempts: u32, initial_ms: u64, max_ms: u64) -> Self {
        Self {
            strategy: RetryStrategy::ExponentialWithJitter {
                initial_ms,
                max_ms,
                jitter: JitterMode::Full,
            },
            max_attempts,
        }
    }

    /// Creates an exponential back-off with equal jitter configuration.
    ///
    /// Equal jitter guarantees at least half the base delay while still providing
    /// spread: `delay = base/2 + uniform(0, base/2)`.
    #[must_use]
    pub fn exponential_equal_jitter(max_attempts: u32, initial_ms: u64, max_ms: u64) -> Self {
        Self {
            strategy: RetryStrategy::ExponentialWithJitter {
                initial_ms,
                max_ms,
                jitter: JitterMode::Equal,
            },
            max_attempts,
        }
    }

    /// Creates an exponential back-off with decorrelated jitter configuration.
    ///
    /// Decorrelated jitter computes each delay relative to the previous one,
    /// producing a smooth, unpredictable retry cadence.
    #[must_use]
    pub fn exponential_decorrelated_jitter(
        max_attempts: u32,
        initial_ms: u64,
        max_ms: u64,
    ) -> Self {
        Self {
            strategy: RetryStrategy::ExponentialWithJitter {
                initial_ms,
                max_ms,
                jitter: JitterMode::Decorrelated,
            },
            max_attempts,
        }
    }

    /// Returns `true` if another attempt should be made.
    ///
    /// `attempt` is the number of attempts *already made* (0-based).
    #[must_use]
    pub fn should_retry(&self, attempt: u32) -> bool {
        if self.strategy == RetryStrategy::NoRetry {
            return false;
        }
        attempt < self.max_attempts
    }

    /// Returns the delay in milliseconds before the next attempt.
    ///
    /// `attempt` is the number of attempts *already made* (0-based).
    /// Returns 0 for `Immediate` or `NoRetry`, the fixed delay for `Fixed`,
    /// and `initial_ms * 2^attempt` (capped at `max_ms`) for `Exponential`.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn next_delay_ms(&self, attempt: u32) -> u64 {
        match &self.strategy {
            RetryStrategy::NoRetry | RetryStrategy::Immediate => 0,
            RetryStrategy::Fixed { delay_ms } => *delay_ms,
            RetryStrategy::Exponential { initial_ms, max_ms } => {
                let factor = 1u64.checked_shl(attempt).unwrap_or(u64::MAX);
                let delay = initial_ms.saturating_mul(factor);
                delay.min(*max_ms)
            }
            RetryStrategy::ExponentialWithJitter {
                initial_ms,
                max_ms,
                jitter,
            } => {
                let factor = 1u64.checked_shl(attempt).unwrap_or(u64::MAX);
                let base_delay = initial_ms.saturating_mul(factor).min(*max_ms);
                // Apply jitter deterministically for `next_delay_ms` (upper bound).
                // Actual randomisation happens at the call site; here we return the
                // maximum possible delay for the given jitter mode.
                match jitter {
                    JitterMode::Full => base_delay,
                    JitterMode::Equal => base_delay,
                    JitterMode::Decorrelated => base_delay,
                }
            }
        }
    }
}

/// Tracks retry attempts for a single job and reports exhaustion.
#[derive(Debug, Clone)]
pub struct RetryTracker {
    config: RetryConfig,
    attempts: u32,
}

impl RetryTracker {
    /// Creates a new tracker with the given configuration.
    #[must_use]
    pub fn new(config: RetryConfig) -> Self {
        Self {
            config,
            attempts: 0,
        }
    }

    /// Records a completed attempt.  Returns the delay in ms before the next
    /// attempt should be made, or `None` if retries are exhausted.
    pub fn record_attempt(&mut self) -> Option<u64> {
        let delay = self.config.next_delay_ms(self.attempts);
        self.attempts += 1;
        if self.config.should_retry(self.attempts) || self.attempts <= self.config.max_attempts {
            // still within limit — but we already incremented, so check:
            if self.attempts <= self.config.max_attempts
                && self.config.strategy != RetryStrategy::NoRetry
            {
                return Some(delay);
            }
        }
        None
    }

    /// Returns `true` if no more retries are available.
    #[must_use]
    pub fn is_exhausted(&self) -> bool {
        !self.config.should_retry(self.attempts)
    }

    /// Returns the number of attempts made so far.
    #[must_use]
    pub fn attempt_count(&self) -> u32 {
        self.attempts
    }

    /// Resets the tracker back to zero attempts.
    pub fn reset(&mut self) {
        self.attempts = 0;
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_retry_should_retry_false() {
        let cfg = RetryConfig::no_retry();
        assert!(!cfg.should_retry(0));
    }

    #[test]
    fn test_fixed_should_retry_within_limit() {
        let cfg = RetryConfig::fixed(3, 100);
        assert!(cfg.should_retry(0));
        assert!(cfg.should_retry(2));
        assert!(!cfg.should_retry(3));
    }

    #[test]
    fn test_fixed_next_delay() {
        let cfg = RetryConfig::fixed(3, 250);
        assert_eq!(cfg.next_delay_ms(0), 250);
        assert_eq!(cfg.next_delay_ms(2), 250);
    }

    #[test]
    fn test_immediate_next_delay_zero() {
        let cfg = RetryConfig {
            strategy: RetryStrategy::Immediate,
            max_attempts: 2,
        };
        assert_eq!(cfg.next_delay_ms(0), 0);
    }

    #[test]
    fn test_exponential_delay_doubles() {
        let cfg = RetryConfig::exponential(5, 100, 10_000);
        assert_eq!(cfg.next_delay_ms(0), 100); // 100 * 2^0
        assert_eq!(cfg.next_delay_ms(1), 200); // 100 * 2^1
        assert_eq!(cfg.next_delay_ms(2), 400); // 100 * 2^2
        assert_eq!(cfg.next_delay_ms(3), 800); // 100 * 2^3
    }

    #[test]
    fn test_exponential_delay_capped() {
        let cfg = RetryConfig::exponential(10, 1_000, 5_000);
        assert_eq!(cfg.next_delay_ms(10), 5_000); // capped
    }

    #[test]
    fn test_no_retry_delay_zero() {
        let cfg = RetryConfig::no_retry();
        assert_eq!(cfg.next_delay_ms(0), 0);
    }

    #[test]
    fn test_tracker_initial_not_exhausted() {
        let tracker = RetryTracker::new(RetryConfig::fixed(3, 100));
        assert!(!tracker.is_exhausted());
    }

    #[test]
    fn test_tracker_exhausted_after_max_attempts() {
        let mut tracker = RetryTracker::new(RetryConfig::fixed(2, 50));
        tracker.record_attempt();
        tracker.record_attempt();
        assert!(tracker.is_exhausted());
    }

    #[test]
    fn test_tracker_attempt_count() {
        let mut tracker = RetryTracker::new(RetryConfig::fixed(5, 10));
        tracker.record_attempt();
        tracker.record_attempt();
        assert_eq!(tracker.attempt_count(), 2);
    }

    #[test]
    fn test_tracker_no_retry_immediately_exhausted() {
        let tracker = RetryTracker::new(RetryConfig::no_retry());
        assert!(tracker.is_exhausted());
    }

    #[test]
    fn test_tracker_reset() {
        let mut tracker = RetryTracker::new(RetryConfig::fixed(3, 100));
        tracker.record_attempt();
        tracker.record_attempt();
        tracker.record_attempt();
        assert!(tracker.is_exhausted());
        tracker.reset();
        assert!(!tracker.is_exhausted());
        assert_eq!(tracker.attempt_count(), 0);
    }
}
