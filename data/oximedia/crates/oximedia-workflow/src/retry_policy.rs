#![allow(dead_code)]
//! Retry policy engine for workflow task execution.
//!
//! Provides configurable retry strategies for failed workflow tasks including
//! fixed delay, exponential back-off, linear back-off, and custom schedules.
//! Includes jitter support, max attempt limits, and retry budgets.

use std::collections::HashMap;
use std::time::Duration;

/// Strategy used to calculate delays between retries.
#[derive(Debug, Clone, PartialEq)]
pub enum RetryStrategy {
    /// Fixed delay between retries.
    Fixed {
        /// The constant delay.
        delay: Duration,
    },
    /// Exponential back-off: `base_delay * multiplier^attempt`.
    Exponential {
        /// Initial delay.
        base_delay: Duration,
        /// Multiplier applied per attempt.
        multiplier: f64,
        /// Maximum delay cap.
        max_delay: Duration,
    },
    /// Linear back-off: `base_delay + increment * attempt`.
    Linear {
        /// Initial delay.
        base_delay: Duration,
        /// Increment per attempt.
        increment: Duration,
        /// Maximum delay cap.
        max_delay: Duration,
    },
    /// A predefined schedule of delays.
    Custom {
        /// List of delays for each retry attempt.
        delays: Vec<Duration>,
    },
    /// No retries at all.
    NoRetry,
}

/// Jitter mode applied to computed delays.
#[derive(Debug, Clone, PartialEq)]
pub enum JitterMode {
    /// No jitter.
    None,
    /// Full jitter: random value in `[0, delay]`.
    Full,
    /// Equal jitter: `delay/2 + random(0, delay/2)`.
    Equal,
    /// Decorrelated jitter based on previous delay.
    Decorrelated,
}

/// Configuration for a retry policy.
#[derive(Debug, Clone)]
pub struct RetryPolicyConfig {
    /// Name of this policy.
    pub name: String,
    /// The retry strategy.
    pub strategy: RetryStrategy,
    /// Maximum number of retry attempts (0 = no retries).
    pub max_attempts: u32,
    /// Jitter mode.
    pub jitter: JitterMode,
    /// Set of error codes that should be retried. Empty means all errors.
    pub retryable_errors: Vec<String>,
    /// Set of error codes that should never be retried.
    pub non_retryable_errors: Vec<String>,
    /// Maximum total time spent retrying.
    pub total_timeout: Option<Duration>,
    /// Metadata.
    pub metadata: HashMap<String, String>,
}

impl RetryPolicyConfig {
    /// Create a new retry policy configuration.
    pub fn new(name: impl Into<String>, strategy: RetryStrategy, max_attempts: u32) -> Self {
        Self {
            name: name.into(),
            strategy,
            max_attempts,
            jitter: JitterMode::None,
            retryable_errors: Vec::new(),
            non_retryable_errors: Vec::new(),
            total_timeout: None,
            metadata: HashMap::new(),
        }
    }

    /// Set the jitter mode.
    #[must_use]
    pub fn with_jitter(mut self, jitter: JitterMode) -> Self {
        self.jitter = jitter;
        self
    }

    /// Add a retryable error code.
    pub fn add_retryable_error(mut self, code: impl Into<String>) -> Self {
        self.retryable_errors.push(code.into());
        self
    }

    /// Add a non-retryable error code.
    pub fn add_non_retryable_error(mut self, code: impl Into<String>) -> Self {
        self.non_retryable_errors.push(code.into());
        self
    }

    /// Set the total timeout.
    #[must_use]
    pub fn with_total_timeout(mut self, timeout: Duration) -> Self {
        self.total_timeout = Some(timeout);
        self
    }

    /// Set metadata.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Create a default fixed-delay policy.
    pub fn fixed(name: impl Into<String>, delay: Duration, max_attempts: u32) -> Self {
        Self::new(name, RetryStrategy::Fixed { delay }, max_attempts)
    }

    /// Create a default exponential back-off policy.
    pub fn exponential(
        name: impl Into<String>,
        base_delay: Duration,
        multiplier: f64,
        max_delay: Duration,
        max_attempts: u32,
    ) -> Self {
        Self::new(
            name,
            RetryStrategy::Exponential {
                base_delay,
                multiplier,
                max_delay,
            },
            max_attempts,
        )
    }
}

/// Tracks the state of retry attempts for a single task.
#[derive(Debug, Clone)]
pub struct RetryState {
    /// The retry policy configuration.
    pub config: RetryPolicyConfig,
    /// Current attempt number (starts at 0).
    pub current_attempt: u32,
    /// History of delays that were applied.
    pub delay_history: Vec<Duration>,
    /// Total time spent waiting so far.
    pub total_wait_time: Duration,
    /// Error messages from each failed attempt.
    pub error_history: Vec<String>,
    /// Whether retries are exhausted.
    pub exhausted: bool,
}

impl RetryState {
    /// Create a new retry state from a config.
    #[must_use]
    pub fn new(config: RetryPolicyConfig) -> Self {
        Self {
            config,
            current_attempt: 0,
            delay_history: Vec::new(),
            total_wait_time: Duration::ZERO,
            error_history: Vec::new(),
            exhausted: false,
        }
    }

    /// Check whether the given error code is retryable under this policy.
    #[must_use]
    pub fn is_retryable(&self, error_code: &str) -> bool {
        if self
            .config
            .non_retryable_errors
            .contains(&error_code.to_string())
        {
            return false;
        }
        if self.config.retryable_errors.is_empty() {
            return true;
        }
        self.config
            .retryable_errors
            .contains(&error_code.to_string())
    }

    /// Check whether another retry attempt is possible.
    #[must_use]
    pub fn can_retry(&self) -> bool {
        if self.exhausted {
            return false;
        }
        if self.current_attempt >= self.config.max_attempts {
            return false;
        }
        if let Some(timeout) = self.config.total_timeout {
            if self.total_wait_time >= timeout {
                return false;
            }
        }
        true
    }

    /// Calculate the delay for the next retry attempt.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn next_delay(&self) -> Duration {
        let attempt = self.current_attempt;
        match &self.config.strategy {
            RetryStrategy::Fixed { delay } => *delay,
            RetryStrategy::Exponential {
                base_delay,
                multiplier,
                max_delay,
            } => {
                let base_ms = base_delay.as_millis() as f64;
                let computed = base_ms * multiplier.powi(attempt as i32);
                let capped = computed.min(max_delay.as_millis() as f64);
                Duration::from_millis(capped as u64)
            }
            RetryStrategy::Linear {
                base_delay,
                increment,
                max_delay,
            } => {
                let base_ms = base_delay.as_millis() as u64;
                let inc_ms = increment.as_millis() as u64;
                let computed = base_ms.saturating_add(inc_ms.saturating_mul(u64::from(attempt)));
                let capped = computed.min(max_delay.as_millis() as u64);
                Duration::from_millis(capped)
            }
            RetryStrategy::Custom { delays } => {
                if (attempt as usize) < delays.len() {
                    delays[attempt as usize]
                } else {
                    delays.last().copied().unwrap_or(Duration::ZERO)
                }
            }
            RetryStrategy::NoRetry => Duration::ZERO,
        }
    }

    /// Apply jitter to a computed delay.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn apply_jitter(&self, delay: Duration, random_factor: f64) -> Duration {
        let ms = delay.as_millis() as f64;
        let factor = random_factor.clamp(0.0, 1.0);
        let jittered = match &self.config.jitter {
            JitterMode::None => ms,
            JitterMode::Full => ms * factor,
            JitterMode::Equal => (ms / 2.0) + (ms / 2.0) * factor,
            JitterMode::Decorrelated => {
                let prev = self
                    .delay_history
                    .last()
                    .map_or(ms, |d| d.as_millis() as f64);
                let upper = prev * 3.0;
                ms.max(upper * factor)
            }
        };
        Duration::from_millis(jittered as u64)
    }

    /// Record a failed attempt and advance the state.
    pub fn record_attempt(&mut self, error_message: impl Into<String>) {
        let delay = self.next_delay();
        self.delay_history.push(delay);
        self.total_wait_time += delay;
        self.error_history.push(error_message.into());
        self.current_attempt += 1;
        if !self.can_retry() {
            self.exhausted = true;
        }
    }

    /// Return the number of attempts made so far.
    #[must_use]
    pub fn attempts_made(&self) -> u32 {
        self.current_attempt
    }

    /// Return remaining attempts.
    #[must_use]
    pub fn remaining_attempts(&self) -> u32 {
        self.config
            .max_attempts
            .saturating_sub(self.current_attempt)
    }

    /// Reset the retry state for reuse.
    pub fn reset(&mut self) {
        self.current_attempt = 0;
        self.delay_history.clear();
        self.total_wait_time = Duration::ZERO;
        self.error_history.clear();
        self.exhausted = false;
    }
}

/// Registry of named retry policies.
#[derive(Debug)]
pub struct RetryPolicyRegistry {
    /// Policies indexed by name.
    policies: HashMap<String, RetryPolicyConfig>,
}

impl Default for RetryPolicyRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl RetryPolicyRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            policies: HashMap::new(),
        }
    }

    /// Register a policy.
    pub fn register(&mut self, config: RetryPolicyConfig) {
        self.policies.insert(config.name.clone(), config);
    }

    /// Get a policy by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&RetryPolicyConfig> {
        self.policies.get(name)
    }

    /// Create a new retry state for a named policy.
    #[must_use]
    pub fn create_state(&self, policy_name: &str) -> Option<RetryState> {
        self.policies
            .get(policy_name)
            .map(|c| RetryState::new(c.clone()))
    }

    /// Return the number of registered policies.
    #[must_use]
    pub fn count(&self) -> usize {
        self.policies.len()
    }

    /// List all registered policy names.
    #[must_use]
    pub fn policy_names(&self) -> Vec<&str> {
        self.policies
            .keys()
            .map(std::string::String::as_str)
            .collect()
    }

    /// Remove a policy.
    pub fn remove(&mut self, name: &str) -> Option<RetryPolicyConfig> {
        self.policies.remove(name)
    }

    /// Create a registry pre-loaded with common default policies.
    #[must_use]
    pub fn with_defaults() -> Self {
        let mut registry = Self::new();
        registry.register(RetryPolicyConfig::fixed(
            "quick-retry",
            Duration::from_secs(1),
            3,
        ));
        registry.register(RetryPolicyConfig::exponential(
            "exponential-backoff",
            Duration::from_secs(1),
            2.0,
            Duration::from_secs(60),
            5,
        ));
        registry.register(RetryPolicyConfig::new(
            "no-retry",
            RetryStrategy::NoRetry,
            0,
        ));
        registry
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_delay() {
        let config = RetryPolicyConfig::fixed("test", Duration::from_secs(5), 3);
        let state = RetryState::new(config);
        assert_eq!(state.next_delay(), Duration::from_secs(5));
    }

    #[test]
    fn test_exponential_delay() {
        let config = RetryPolicyConfig::exponential(
            "exp",
            Duration::from_millis(100),
            2.0,
            Duration::from_secs(10),
            5,
        );
        let mut state = RetryState::new(config);
        assert_eq!(state.next_delay(), Duration::from_millis(100)); // 100 * 2^0
        state.current_attempt = 1;
        assert_eq!(state.next_delay(), Duration::from_millis(200)); // 100 * 2^1
        state.current_attempt = 2;
        assert_eq!(state.next_delay(), Duration::from_millis(400)); // 100 * 2^2
    }

    #[test]
    fn test_exponential_capped() {
        let config = RetryPolicyConfig::exponential(
            "exp-cap",
            Duration::from_secs(1),
            10.0,
            Duration::from_secs(5),
            10,
        );
        let mut state = RetryState::new(config);
        state.current_attempt = 5;
        assert_eq!(state.next_delay(), Duration::from_secs(5)); // capped
    }

    #[test]
    fn test_linear_delay() {
        let config = RetryPolicyConfig::new(
            "linear",
            RetryStrategy::Linear {
                base_delay: Duration::from_millis(100),
                increment: Duration::from_millis(50),
                max_delay: Duration::from_secs(1),
            },
            5,
        );
        let mut state = RetryState::new(config);
        assert_eq!(state.next_delay(), Duration::from_millis(100));
        state.current_attempt = 2;
        assert_eq!(state.next_delay(), Duration::from_millis(200));
    }

    #[test]
    fn test_custom_delays() {
        let config = RetryPolicyConfig::new(
            "custom",
            RetryStrategy::Custom {
                delays: vec![
                    Duration::from_secs(1),
                    Duration::from_secs(5),
                    Duration::from_secs(30),
                ],
            },
            3,
        );
        let mut state = RetryState::new(config);
        assert_eq!(state.next_delay(), Duration::from_secs(1));
        state.current_attempt = 1;
        assert_eq!(state.next_delay(), Duration::from_secs(5));
        state.current_attempt = 2;
        assert_eq!(state.next_delay(), Duration::from_secs(30));
    }

    #[test]
    fn test_no_retry() {
        let config = RetryPolicyConfig::new("none", RetryStrategy::NoRetry, 0);
        let state = RetryState::new(config);
        assert!(!state.can_retry());
    }

    #[test]
    fn test_can_retry_limits() {
        let config = RetryPolicyConfig::fixed("test", Duration::from_secs(1), 2);
        let mut state = RetryState::new(config);
        assert!(state.can_retry());
        state.record_attempt("error 1");
        assert!(state.can_retry());
        state.record_attempt("error 2");
        assert!(!state.can_retry());
        assert!(state.exhausted);
    }

    #[test]
    fn test_total_timeout_limit() {
        let config = RetryPolicyConfig::fixed("test", Duration::from_secs(10), 100)
            .with_total_timeout(Duration::from_secs(5));
        let mut state = RetryState::new(config);
        state.record_attempt("fail");
        // total_wait_time is now 10s which exceeds the 5s timeout
        assert!(!state.can_retry());
    }

    #[test]
    fn test_retryable_errors() {
        let config = RetryPolicyConfig::fixed("test", Duration::from_secs(1), 3)
            .add_retryable_error("TIMEOUT")
            .add_retryable_error("CONN_RESET");
        let state = RetryState::new(config);
        assert!(state.is_retryable("TIMEOUT"));
        assert!(state.is_retryable("CONN_RESET"));
        assert!(!state.is_retryable("PERMISSION_DENIED"));
    }

    #[test]
    fn test_non_retryable_errors() {
        let config = RetryPolicyConfig::fixed("test", Duration::from_secs(1), 3)
            .add_non_retryable_error("FATAL");
        let state = RetryState::new(config);
        assert!(!state.is_retryable("FATAL"));
        assert!(state.is_retryable("ANYTHING_ELSE"));
    }

    #[test]
    fn test_record_attempt_tracks_history() {
        let config = RetryPolicyConfig::fixed("test", Duration::from_secs(1), 5);
        let mut state = RetryState::new(config);
        state.record_attempt("error A");
        state.record_attempt("error B");
        assert_eq!(state.attempts_made(), 2);
        assert_eq!(state.remaining_attempts(), 3);
        assert_eq!(state.error_history.len(), 2);
        assert_eq!(state.delay_history.len(), 2);
    }

    #[test]
    fn test_reset_state() {
        let config = RetryPolicyConfig::fixed("test", Duration::from_secs(1), 3);
        let mut state = RetryState::new(config);
        state.record_attempt("err");
        state.record_attempt("err");
        state.reset();
        assert_eq!(state.attempts_made(), 0);
        assert!(state.can_retry());
        assert!(!state.exhausted);
    }

    #[test]
    fn test_jitter_none() {
        let config = RetryPolicyConfig::fixed("test", Duration::from_secs(2), 3);
        let state = RetryState::new(config);
        let delay = state.apply_jitter(Duration::from_secs(2), 0.5);
        assert_eq!(delay, Duration::from_secs(2));
    }

    #[test]
    fn test_jitter_full() {
        let config = RetryPolicyConfig::fixed("test", Duration::from_secs(1), 3)
            .with_jitter(JitterMode::Full);
        let state = RetryState::new(config);
        let delay = state.apply_jitter(Duration::from_secs(1), 0.5);
        assert_eq!(delay, Duration::from_millis(500));
    }

    #[test]
    fn test_jitter_equal() {
        let config = RetryPolicyConfig::fixed("test", Duration::from_secs(1), 3)
            .with_jitter(JitterMode::Equal);
        let state = RetryState::new(config);
        let delay = state.apply_jitter(Duration::from_secs(1), 0.5);
        // 500 + 500*0.5 = 750
        assert_eq!(delay, Duration::from_millis(750));
    }

    #[test]
    fn test_registry_basic() {
        let mut registry = RetryPolicyRegistry::new();
        let config = RetryPolicyConfig::fixed("my-policy", Duration::from_secs(1), 3);
        registry.register(config);
        assert_eq!(registry.count(), 1);
        assert!(registry.get("my-policy").is_some());
        assert!(registry.get("unknown").is_none());
    }

    #[test]
    fn test_registry_create_state() {
        let mut registry = RetryPolicyRegistry::new();
        registry.register(RetryPolicyConfig::fixed("p", Duration::from_secs(1), 2));
        let state = registry.create_state("p");
        assert!(state.is_some());
        let s = state.expect("should succeed in test");
        assert_eq!(s.config.max_attempts, 2);
    }

    #[test]
    fn test_registry_with_defaults() {
        let registry = RetryPolicyRegistry::with_defaults();
        assert!(registry.get("quick-retry").is_some());
        assert!(registry.get("exponential-backoff").is_some());
        assert!(registry.get("no-retry").is_some());
        assert_eq!(registry.count(), 3);
    }

    #[test]
    fn test_registry_remove() {
        let mut registry = RetryPolicyRegistry::with_defaults();
        let removed = registry.remove("quick-retry");
        assert!(removed.is_some());
        assert!(registry.get("quick-retry").is_none());
    }

    #[test]
    fn test_config_metadata() {
        let config =
            RetryPolicyConfig::fixed("m", Duration::from_secs(1), 1).with_metadata("team", "media");
        assert_eq!(
            config.metadata.get("team").map(|s| s.as_str()),
            Some("media")
        );
    }

    #[test]
    fn test_default_registry() {
        let registry = RetryPolicyRegistry::default();
        assert_eq!(registry.count(), 0);
    }
}

// ============================================================================
// Simple flat API: ExponentialRetryPolicy / RetryDecision / RetryPolicyState
// ============================================================================

/// Decision returned by [`RetryPolicyState::next_delay`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RetryDecision {
    /// Retry after the given delay.
    Retry(Duration),
    /// No more retries — give up.
    GiveUp,
}

/// A simple, self-contained exponential backoff policy.
///
/// Unlike the richer [`RetryPolicyConfig`] / [`RetryState`] API this struct
/// uses a flat set of fields that are easy to construct from configuration
/// files or environment variables.
#[derive(Debug, Clone)]
pub struct ExponentialRetryPolicy {
    /// Maximum number of retry attempts (0 = no retries).
    pub max_attempts: u32,
    /// Delay before the first retry, in milliseconds.
    pub initial_delay_ms: u64,
    /// Multiplicative factor applied to the delay after each attempt.
    pub multiplier: f64,
    /// Hard cap on the delay, in milliseconds.
    pub max_delay_ms: u64,
    /// Whether to apply ±50 % splitmix64 jitter to each computed delay.
    pub jitter: bool,
}

impl ExponentialRetryPolicy {
    /// Create a new policy with explicit parameters.
    #[must_use]
    pub fn new(
        max_attempts: u32,
        initial_delay_ms: u64,
        multiplier: f64,
        max_delay_ms: u64,
        jitter: bool,
    ) -> Self {
        Self {
            max_attempts,
            initial_delay_ms,
            multiplier,
            max_delay_ms,
            jitter,
        }
    }

    /// Sensible defaults: 3 attempts, 100 ms base, ×2, 30 s cap, with jitter.
    #[must_use]
    pub fn default_policy() -> Self {
        Self::new(3, 100, 2.0, 30_000, true)
    }

    /// Policy that never retries.
    #[must_use]
    pub fn no_retry() -> Self {
        Self::new(0, 0, 1.0, 0, false)
    }

    /// Fixed-delay policy.
    #[must_use]
    pub fn fixed(delay_ms: u64, max_attempts: u32) -> Self {
        Self::new(max_attempts, delay_ms, 1.0, delay_ms, false)
    }

    /// Create a [`RetryPolicyState`] that tracks progress against this policy.
    #[must_use]
    pub fn start(&self) -> RetryPolicyState {
        RetryPolicyState::new(self.clone())
    }
}

impl Default for ExponentialRetryPolicy {
    fn default() -> Self {
        Self::default_policy()
    }
}

// ---------------------------------------------------------------------------
// splitmix64 PRNG (no rand crate dependency)
// ---------------------------------------------------------------------------

#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Return a pseudo-random f64 in `[0.0, 1.0)` using splitmix64.
#[inline]
fn splitmix64_f64(state: &mut u64) -> f64 {
    let raw = splitmix64(state);
    // Use the top 53 bits for a double in [0, 1).
    (raw >> 11) as f64 / (1u64 << 53) as f64
}

// ---------------------------------------------------------------------------
// RetryPolicyState
// ---------------------------------------------------------------------------

/// Tracks retry progress for an [`ExponentialRetryPolicy`].
pub struct RetryPolicyState {
    policy: ExponentialRetryPolicy,
    attempt: u32,
    last_delay_ms: u64,
    seed: u64,
}

impl RetryPolicyState {
    /// Create a new state.  The PRNG seed is derived from the current wall
    /// clock so that different instances produce different jitter sequences.
    #[must_use]
    pub fn new(policy: ExponentialRetryPolicy) -> Self {
        // Seed from system time nanos; fall back to a fixed value.
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos() as u64)
            .unwrap_or(0xdead_beef_cafe_babe);
        Self {
            policy,
            attempt: 0,
            last_delay_ms: 0,
            seed,
        }
    }

    /// Compute and return the delay for the next retry attempt, or
    /// [`RetryDecision::GiveUp`] if the attempt budget is exhausted.
    ///
    /// This method advances the internal attempt counter on each call.
    #[allow(clippy::cast_precision_loss)]
    pub fn next_delay(&mut self) -> RetryDecision {
        if self.attempt >= self.policy.max_attempts {
            return RetryDecision::GiveUp;
        }

        // Exponential base delay.
        let base = self.policy.initial_delay_ms as f64;
        let computed = base * self.policy.multiplier.powi(self.attempt as i32);
        let mut delay_ms = computed.min(self.policy.max_delay_ms as f64) as u64;

        // Optional jitter: scale delay by a factor in [0.5, 1.5].
        if self.policy.jitter && delay_ms > 0 {
            let r = splitmix64_f64(&mut self.seed); // [0, 1)
            let factor = 0.5 + r; // [0.5, 1.5)
            delay_ms = ((delay_ms as f64 * factor) as u64).min(self.policy.max_delay_ms);
        }

        self.last_delay_ms = delay_ms;
        self.attempt += 1;

        RetryDecision::Retry(Duration::from_millis(delay_ms))
    }

    /// Reset the attempt counter and last-delay record so the policy can be
    /// reused from scratch.
    pub fn reset(&mut self) {
        self.attempt = 0;
        self.last_delay_ms = 0;
    }

    /// Return the number of attempts consumed so far.
    #[must_use]
    pub fn attempts_used(&self) -> u32 {
        self.attempt
    }
}

// ---------------------------------------------------------------------------
// Tests for the new API
// ---------------------------------------------------------------------------

#[cfg(test)]
mod exponential_tests {
    use super::*;

    #[test]
    fn no_retry_gives_up_immediately() {
        let mut state = ExponentialRetryPolicy::no_retry().start();
        assert_eq!(state.next_delay(), RetryDecision::GiveUp);
    }

    #[test]
    fn fixed_policy_same_delay() {
        let mut state = ExponentialRetryPolicy::fixed(200, 3).start();
        for _ in 0..3 {
            match state.next_delay() {
                RetryDecision::Retry(d) => assert_eq!(d.as_millis(), 200),
                RetryDecision::GiveUp => panic!("should not give up yet"),
            }
        }
        assert_eq!(state.next_delay(), RetryDecision::GiveUp);
    }

    #[test]
    fn exponential_increases_correctly() {
        // multiplier=2, base=100ms, no jitter → 100, 200, 400
        let policy = ExponentialRetryPolicy::new(3, 100, 2.0, 10_000, false);
        let mut state = policy.start();
        let d0 = state.next_delay();
        let d1 = state.next_delay();
        let d2 = state.next_delay();
        assert_eq!(d0, RetryDecision::Retry(Duration::from_millis(100)));
        assert_eq!(d1, RetryDecision::Retry(Duration::from_millis(200)));
        assert_eq!(d2, RetryDecision::Retry(Duration::from_millis(400)));
        assert_eq!(state.next_delay(), RetryDecision::GiveUp);
    }

    #[test]
    fn max_delay_cap_honored() {
        let policy = ExponentialRetryPolicy::new(5, 1000, 10.0, 3000, false);
        let mut state = policy.start();
        for _ in 0..5 {
            if let RetryDecision::Retry(d) = state.next_delay() {
                assert!(d.as_millis() <= 3000, "delay exceeds cap: {d:?}");
            }
        }
    }

    #[test]
    fn jitter_within_expected_range() {
        let policy = ExponentialRetryPolicy::new(10, 1000, 1.0, 10_000, true);
        let mut state = policy.start();
        for _ in 0..10 {
            if let RetryDecision::Retry(d) = state.next_delay() {
                let ms = d.as_millis() as f64;
                // With factor in [0.5, 1.5], delay should be in [500, 1500].
                assert!(
                    ms >= 500.0 && ms <= 1500.0,
                    "jittered delay {ms} out of range"
                );
            }
        }
    }

    #[test]
    fn reset_restarts_from_zero() {
        let policy = ExponentialRetryPolicy::new(2, 100, 2.0, 10_000, false);
        let mut state = policy.start();
        state.next_delay();
        state.next_delay();
        assert_eq!(state.next_delay(), RetryDecision::GiveUp);
        state.reset();
        assert_eq!(state.attempts_used(), 0);
        // After reset, first delay should be base delay again.
        assert_eq!(
            state.next_delay(),
            RetryDecision::Retry(Duration::from_millis(100))
        );
    }

    #[test]
    fn multiplier_one_acts_like_fixed() {
        let policy = ExponentialRetryPolicy::new(4, 500, 1.0, 10_000, false);
        let mut state = policy.start();
        for _ in 0..4 {
            assert_eq!(
                state.next_delay(),
                RetryDecision::Retry(Duration::from_millis(500))
            );
        }
        assert_eq!(state.next_delay(), RetryDecision::GiveUp);
    }

    #[test]
    fn attempts_used_tracks_correctly() {
        let policy = ExponentialRetryPolicy::new(3, 100, 2.0, 10_000, false);
        let mut state = policy.start();
        assert_eq!(state.attempts_used(), 0);
        state.next_delay();
        assert_eq!(state.attempts_used(), 1);
        state.next_delay();
        assert_eq!(state.attempts_used(), 2);
    }

    #[test]
    fn default_policy_has_sane_values() {
        let p = ExponentialRetryPolicy::default_policy();
        assert_eq!(p.max_attempts, 3);
        assert_eq!(p.initial_delay_ms, 100);
        assert!((p.multiplier - 2.0).abs() < f64::EPSILON);
        assert_eq!(p.max_delay_ms, 30_000);
        assert!(p.jitter);
    }

    #[test]
    fn zero_initial_delay_no_jitter() {
        let policy = ExponentialRetryPolicy::new(2, 0, 2.0, 0, true);
        let mut state = policy.start();
        assert_eq!(
            state.next_delay(),
            RetryDecision::Retry(Duration::from_millis(0))
        );
    }

    #[test]
    fn give_up_after_all_attempts_exhausted() {
        let policy = ExponentialRetryPolicy::new(1, 50, 1.0, 1000, false);
        let mut state = policy.start();
        assert!(matches!(state.next_delay(), RetryDecision::Retry(_)));
        assert_eq!(state.next_delay(), RetryDecision::GiveUp);
        // Additional calls still return GiveUp.
        assert_eq!(state.next_delay(), RetryDecision::GiveUp);
    }

    #[test]
    fn splitmix64_produces_different_values() {
        let mut seed = 12345u64;
        let a = splitmix64(&mut seed);
        let b = splitmix64(&mut seed);
        assert_ne!(a, b);
    }

    #[test]
    fn splitmix64_f64_in_unit_range() {
        let mut seed = 0xabcd_ef01_2345_6789u64;
        for _ in 0..1000 {
            let v = splitmix64_f64(&mut seed);
            assert!(v >= 0.0 && v < 1.0, "out of range: {v}");
        }
    }
}
