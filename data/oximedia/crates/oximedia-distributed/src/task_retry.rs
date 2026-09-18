#![allow(dead_code)]
//! Task retry logic with configurable backoff strategies for distributed jobs.
//!
//! Provides exponential backoff, linear backoff, and constant-interval retry
//! policies. Tracks attempt history and enforces maximum retry limits.

use std::fmt;
use std::time::Duration;

/// Backoff strategy for retries.
#[derive(Debug, Clone, PartialEq)]
pub enum BackoffStrategy {
    /// Constant delay between retries.
    Constant {
        /// The fixed delay duration.
        delay: Duration,
    },
    /// Linear increase: delay = base + attempt * step.
    Linear {
        /// Base delay for the first retry.
        base: Duration,
        /// Additive step per attempt.
        step: Duration,
    },
    /// Exponential increase: delay = base * multiplier^attempt, capped at max.
    Exponential {
        /// Base delay for the first retry.
        base: Duration,
        /// Multiplier per attempt.
        multiplier: f64,
        /// Maximum delay cap.
        max_delay: Duration,
    },
}

impl fmt::Display for BackoffStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Constant { delay } => write!(f, "Constant({delay:?})"),
            Self::Linear { base, step } => write!(f, "Linear(base={base:?}, step={step:?})"),
            Self::Exponential {
                base, multiplier, ..
            } => write!(f, "Exponential(base={base:?}, mult={multiplier:.1})"),
        }
    }
}

/// Configuration for retry behavior.
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    /// Maximum number of retry attempts (0 = no retries).
    pub max_retries: u32,
    /// Backoff strategy.
    pub backoff: BackoffStrategy,
    /// Whether to add jitter to computed delays.
    pub jitter: bool,
    /// Maximum jitter percentage (0..100).
    pub jitter_percent: u32,
    /// Set of error codes that are retryable (empty = all errors are retryable).
    pub retryable_codes: Vec<String>,
}

impl RetryPolicy {
    /// Create a new retry policy with exponential backoff defaults.
    #[must_use]
    pub fn new(max_retries: u32) -> Self {
        Self {
            max_retries,
            backoff: BackoffStrategy::Exponential {
                base: Duration::from_millis(100),
                multiplier: 2.0,
                max_delay: Duration::from_secs(30),
            },
            jitter: false,
            jitter_percent: 20,
            retryable_codes: Vec::new(),
        }
    }

    /// Set the backoff strategy.
    #[must_use]
    pub fn with_backoff(mut self, backoff: BackoffStrategy) -> Self {
        self.backoff = backoff;
        self
    }

    /// Enable jitter with the given percentage.
    #[must_use]
    pub fn with_jitter(mut self, percent: u32) -> Self {
        self.jitter = true;
        self.jitter_percent = percent.min(100);
        self
    }

    /// Add a retryable error code.
    pub fn add_retryable_code(&mut self, code: &str) {
        self.retryable_codes.push(code.to_string());
    }

    /// Check if an error code is retryable.
    #[must_use]
    pub fn is_retryable(&self, code: &str) -> bool {
        if self.retryable_codes.is_empty() {
            return true; // all errors retryable by default
        }
        self.retryable_codes.iter().any(|c| c == code)
    }

    /// Compute the delay for the given attempt number (0-based).
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn compute_delay(&self, attempt: u32) -> Duration {
        match &self.backoff {
            BackoffStrategy::Constant { delay } => *delay,
            BackoffStrategy::Linear { base, step } => *base + *step * attempt,
            BackoffStrategy::Exponential {
                base,
                multiplier,
                max_delay,
            } => {
                let base_ms = base.as_millis() as f64;
                let computed = base_ms * multiplier.powi(attempt as i32);
                let capped = computed.min(max_delay.as_millis() as f64);
                Duration::from_millis(capped as u64)
            }
        }
    }

    /// Check if another retry attempt is allowed.
    #[must_use]
    pub fn can_retry(&self, attempts_so_far: u32) -> bool {
        attempts_so_far < self.max_retries
    }
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self::new(3)
    }
}

/// Outcome of a single task execution attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttemptOutcome {
    /// The attempt succeeded.
    Success,
    /// The attempt failed with the given error code and message.
    Failed {
        /// Error code.
        code: String,
        /// Error message.
        message: String,
    },
    /// The attempt timed out.
    Timeout,
    /// The attempt was cancelled.
    Cancelled,
}

impl fmt::Display for AttemptOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Success => write!(f, "Success"),
            Self::Failed { code, message } => write!(f, "Failed({code}): {message}"),
            Self::Timeout => write!(f, "Timeout"),
            Self::Cancelled => write!(f, "Cancelled"),
        }
    }
}

/// Record of a single retry attempt.
#[derive(Debug, Clone)]
pub struct AttemptRecord {
    /// Attempt number (0-based).
    pub attempt: u32,
    /// Outcome of this attempt.
    pub outcome: AttemptOutcome,
    /// Duration of this attempt.
    pub duration_ms: u64,
    /// The backoff delay that was waited before this attempt (0 for first attempt).
    pub delay_before_ms: u64,
}

impl AttemptRecord {
    /// Create a new attempt record.
    #[must_use]
    pub fn new(
        attempt: u32,
        outcome: AttemptOutcome,
        duration_ms: u64,
        delay_before_ms: u64,
    ) -> Self {
        Self {
            attempt,
            outcome,
            duration_ms,
            delay_before_ms,
        }
    }

    /// Whether this attempt succeeded.
    #[must_use]
    pub fn is_success(&self) -> bool {
        self.outcome == AttemptOutcome::Success
    }
}

/// Tracks the retry state for a single task.
#[derive(Debug, Clone)]
pub struct RetryTracker {
    /// The task identifier.
    pub task_id: String,
    /// Retry policy in use.
    pub policy: RetryPolicy,
    /// History of attempts.
    pub history: Vec<AttemptRecord>,
    /// Whether the task has been exhausted (no more retries).
    pub exhausted: bool,
    /// Whether the task ultimately succeeded.
    pub succeeded: bool,
}

impl RetryTracker {
    /// Create a new retry tracker for a task.
    #[must_use]
    pub fn new(task_id: &str, policy: RetryPolicy) -> Self {
        Self {
            task_id: task_id.to_string(),
            policy,
            history: Vec::new(),
            exhausted: false,
            succeeded: false,
        }
    }

    /// Record an attempt and determine the next action.
    pub fn record_attempt(&mut self, outcome: AttemptOutcome, duration_ms: u64) -> RetryDecision {
        let attempt_num = self.history.len() as u32;
        let delay_before = if attempt_num == 0 {
            0
        } else {
            self.policy.compute_delay(attempt_num - 1).as_millis() as u64
        };

        self.history.push(AttemptRecord::new(
            attempt_num,
            outcome.clone(),
            duration_ms,
            delay_before,
        ));

        if outcome == AttemptOutcome::Success {
            self.succeeded = true;
            return RetryDecision::Done;
        }

        if outcome == AttemptOutcome::Cancelled {
            self.exhausted = true;
            return RetryDecision::Abort("Cancelled by user".to_string());
        }

        // Check retryable
        if let AttemptOutcome::Failed { ref code, .. } = outcome {
            if !self.policy.is_retryable(code) {
                self.exhausted = true;
                return RetryDecision::Abort(format!("Error code '{code}' is not retryable"));
            }
        }

        // Check if we can retry
        if self.policy.can_retry(attempt_num + 1) {
            let delay = self.policy.compute_delay(attempt_num);
            RetryDecision::RetryAfter(delay)
        } else {
            self.exhausted = true;
            RetryDecision::Exhausted
        }
    }

    /// Number of attempts made.
    #[must_use]
    pub fn attempt_count(&self) -> usize {
        self.history.len()
    }

    /// Total duration across all attempts (excluding delays).
    #[must_use]
    pub fn total_attempt_duration_ms(&self) -> u64 {
        self.history.iter().map(|r| r.duration_ms).sum()
    }

    /// Total delay time waited across all attempts.
    #[must_use]
    pub fn total_delay_ms(&self) -> u64 {
        self.history.iter().map(|r| r.delay_before_ms).sum()
    }
}

/// Decision returned after recording an attempt.
#[derive(Debug, Clone, PartialEq)]
pub enum RetryDecision {
    /// Task completed successfully, no retry needed.
    Done,
    /// Retry after the given delay.
    RetryAfter(Duration),
    /// All retries exhausted, task permanently failed.
    Exhausted,
    /// Task aborted due to non-retryable error or cancellation.
    Abort(String),
}

impl fmt::Display for RetryDecision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Done => write!(f, "Done"),
            Self::RetryAfter(d) => write!(f, "RetryAfter({d:?})"),
            Self::Exhausted => write!(f, "Exhausted"),
            Self::Abort(reason) => write!(f, "Abort({reason})"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_backoff() {
        let policy = RetryPolicy::new(3).with_backoff(BackoffStrategy::Constant {
            delay: Duration::from_millis(500),
        });
        assert_eq!(policy.compute_delay(0), Duration::from_millis(500));
        assert_eq!(policy.compute_delay(1), Duration::from_millis(500));
        assert_eq!(policy.compute_delay(5), Duration::from_millis(500));
    }

    #[test]
    fn test_linear_backoff() {
        let policy = RetryPolicy::new(5).with_backoff(BackoffStrategy::Linear {
            base: Duration::from_millis(100),
            step: Duration::from_millis(200),
        });
        assert_eq!(policy.compute_delay(0), Duration::from_millis(100));
        assert_eq!(policy.compute_delay(1), Duration::from_millis(300));
        assert_eq!(policy.compute_delay(2), Duration::from_millis(500));
    }

    #[test]
    fn test_exponential_backoff() {
        let policy = RetryPolicy::new(5).with_backoff(BackoffStrategy::Exponential {
            base: Duration::from_millis(100),
            multiplier: 2.0,
            max_delay: Duration::from_secs(10),
        });
        assert_eq!(policy.compute_delay(0), Duration::from_millis(100));
        assert_eq!(policy.compute_delay(1), Duration::from_millis(200));
        assert_eq!(policy.compute_delay(2), Duration::from_millis(400));
        assert_eq!(policy.compute_delay(3), Duration::from_millis(800));
    }

    #[test]
    fn test_exponential_backoff_cap() {
        let policy = RetryPolicy::new(10).with_backoff(BackoffStrategy::Exponential {
            base: Duration::from_secs(1),
            multiplier: 3.0,
            max_delay: Duration::from_secs(5),
        });
        // 1000 * 3^5 = 243000 ms, but capped at 5000ms
        assert_eq!(policy.compute_delay(5), Duration::from_secs(5));
    }

    #[test]
    fn test_can_retry() {
        let policy = RetryPolicy::new(3);
        assert!(policy.can_retry(0));
        assert!(policy.can_retry(1));
        assert!(policy.can_retry(2));
        assert!(!policy.can_retry(3));
        assert!(!policy.can_retry(4));
    }

    #[test]
    fn test_retryable_codes_default_all() {
        let policy = RetryPolicy::new(3);
        assert!(policy.is_retryable("any_code"));
        assert!(policy.is_retryable("another_code"));
    }

    #[test]
    fn test_retryable_codes_specific() {
        let mut policy = RetryPolicy::new(3);
        policy.add_retryable_code("TIMEOUT");
        policy.add_retryable_code("UNAVAILABLE");
        assert!(policy.is_retryable("TIMEOUT"));
        assert!(policy.is_retryable("UNAVAILABLE"));
        assert!(!policy.is_retryable("PERMISSION_DENIED"));
    }

    #[test]
    fn test_retry_tracker_success_first_attempt() {
        let policy = RetryPolicy::new(3);
        let mut tracker = RetryTracker::new("task-1", policy);
        let decision = tracker.record_attempt(AttemptOutcome::Success, 100);
        assert_eq!(decision, RetryDecision::Done);
        assert!(tracker.succeeded);
        assert!(!tracker.exhausted);
        assert_eq!(tracker.attempt_count(), 1);
    }

    #[test]
    fn test_retry_tracker_fail_then_succeed() {
        let policy = RetryPolicy::new(3);
        let mut tracker = RetryTracker::new("task-2", policy);

        let d1 = tracker.record_attempt(
            AttemptOutcome::Failed {
                code: "ERR".to_string(),
                message: "boom".to_string(),
            },
            50,
        );
        assert!(matches!(d1, RetryDecision::RetryAfter(_)));

        let d2 = tracker.record_attempt(AttemptOutcome::Success, 80);
        assert_eq!(d2, RetryDecision::Done);
        assert!(tracker.succeeded);
        assert_eq!(tracker.attempt_count(), 2);
    }

    #[test]
    fn test_retry_tracker_exhausted() {
        let policy = RetryPolicy::new(2);
        let mut tracker = RetryTracker::new("task-3", policy);

        let fail = AttemptOutcome::Failed {
            code: "ERR".to_string(),
            message: "fail".to_string(),
        };
        tracker.record_attempt(fail.clone(), 10);
        tracker.record_attempt(fail.clone(), 10);
        let d = tracker.record_attempt(fail, 10);
        assert_eq!(d, RetryDecision::Exhausted);
        assert!(tracker.exhausted);
        assert!(!tracker.succeeded);
    }

    #[test]
    fn test_retry_tracker_cancelled() {
        let policy = RetryPolicy::new(5);
        let mut tracker = RetryTracker::new("task-4", policy);
        let d = tracker.record_attempt(AttemptOutcome::Cancelled, 0);
        assert!(matches!(d, RetryDecision::Abort(_)));
        assert!(tracker.exhausted);
    }

    #[test]
    fn test_retry_tracker_non_retryable_code() {
        let mut policy = RetryPolicy::new(5);
        policy.add_retryable_code("TIMEOUT");

        let mut tracker = RetryTracker::new("task-5", policy);
        let d = tracker.record_attempt(
            AttemptOutcome::Failed {
                code: "PERMISSION_DENIED".to_string(),
                message: "no access".to_string(),
            },
            20,
        );
        assert!(matches!(d, RetryDecision::Abort(_)));
    }

    #[test]
    fn test_retry_tracker_total_durations() {
        let policy = RetryPolicy::new(3);
        let mut tracker = RetryTracker::new("task-6", policy);
        tracker.record_attempt(
            AttemptOutcome::Failed {
                code: "E".to_string(),
                message: "".to_string(),
            },
            100,
        );
        tracker.record_attempt(AttemptOutcome::Success, 200);
        assert_eq!(tracker.total_attempt_duration_ms(), 300);
    }

    #[test]
    fn test_attempt_outcome_display() {
        assert_eq!(AttemptOutcome::Success.to_string(), "Success");
        assert_eq!(AttemptOutcome::Timeout.to_string(), "Timeout");
        assert_eq!(AttemptOutcome::Cancelled.to_string(), "Cancelled");
    }

    #[test]
    fn test_backoff_strategy_display() {
        let c = BackoffStrategy::Constant {
            delay: Duration::from_millis(100),
        };
        assert!(c.to_string().contains("Constant"));

        let l = BackoffStrategy::Linear {
            base: Duration::from_millis(50),
            step: Duration::from_millis(100),
        };
        assert!(l.to_string().contains("Linear"));
    }

    #[test]
    fn test_retry_decision_display() {
        assert_eq!(RetryDecision::Done.to_string(), "Done");
        assert_eq!(RetryDecision::Exhausted.to_string(), "Exhausted");
    }

    #[test]
    fn test_retry_policy_default() {
        let policy = RetryPolicy::default();
        assert_eq!(policy.max_retries, 3);
    }

    #[test]
    fn test_attempt_record_is_success() {
        let success = AttemptRecord::new(0, AttemptOutcome::Success, 50, 0);
        assert!(success.is_success());
        let fail = AttemptRecord::new(0, AttemptOutcome::Timeout, 50, 0);
        assert!(!fail.is_success());
    }

    #[test]
    fn test_jitter_config() {
        let policy = RetryPolicy::new(3).with_jitter(30);
        assert!(policy.jitter);
        assert_eq!(policy.jitter_percent, 30);
    }

    #[test]
    fn test_jitter_capped_at_100() {
        let policy = RetryPolicy::new(3).with_jitter(200);
        assert_eq!(policy.jitter_percent, 100);
    }
}
