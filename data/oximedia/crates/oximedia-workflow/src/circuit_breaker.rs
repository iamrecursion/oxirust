//! Circuit breaker pattern for workflow retry protection.
//!
//! Implements a three-state circuit breaker (Closed, Open, HalfOpen) that
//! tracks consecutive failures across workflows and prevents retry storms.
//! When failures exceed a threshold, the breaker opens and rejects calls
//! for a configurable cooldown period. After cooldown, a limited number
//! of probe calls are allowed (half-open) to test recovery.

use std::collections::HashMap;
use std::time::Duration;

/// State of the circuit breaker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Normal operation: requests pass through. Failures are counted.
    Closed,
    /// Breaker tripped: all requests are rejected until cooldown expires.
    Open,
    /// Cooldown expired: a limited number of probe requests are allowed.
    HalfOpen,
}

impl std::fmt::Display for CircuitState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Closed => write!(f, "Closed"),
            Self::Open => write!(f, "Open"),
            Self::HalfOpen => write!(f, "HalfOpen"),
        }
    }
}

/// Configuration for a circuit breaker instance.
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Number of consecutive failures before the breaker opens.
    pub failure_threshold: u32,
    /// Duration the breaker stays open before transitioning to half-open.
    pub cooldown_period: Duration,
    /// Number of probe requests allowed in half-open state.
    pub half_open_max_probes: u32,
    /// Minimum number of requests before the breaker can trip.
    pub minimum_request_count: u32,
    /// Optional failure rate threshold (0.0..=1.0). If set, the breaker
    /// opens when the failure rate exceeds this value AND consecutive
    /// failures >= `failure_threshold`.
    pub failure_rate_threshold: Option<f64>,
    /// Human-readable name for logging.
    pub name: String,
}

impl CircuitBreakerConfig {
    /// Create a new configuration with sensible defaults.
    #[must_use]
    pub fn new(name: impl Into<String>, failure_threshold: u32, cooldown_period: Duration) -> Self {
        Self {
            failure_threshold,
            cooldown_period,
            half_open_max_probes: 1,
            minimum_request_count: 1,
            failure_rate_threshold: None,
            name: name.into(),
        }
    }

    /// Set number of probes allowed in half-open state.
    #[must_use]
    pub fn with_half_open_probes(mut self, probes: u32) -> Self {
        self.half_open_max_probes = probes;
        self
    }

    /// Set minimum request count before the breaker can trip.
    #[must_use]
    pub fn with_minimum_requests(mut self, count: u32) -> Self {
        self.minimum_request_count = count;
        self
    }

    /// Set failure rate threshold (0.0 to 1.0).
    #[must_use]
    pub fn with_failure_rate_threshold(mut self, rate: f64) -> Self {
        self.failure_rate_threshold = Some(rate.clamp(0.0, 1.0));
        self
    }
}

/// Metrics tracked by the circuit breaker.
#[derive(Debug, Clone, Default)]
pub struct CircuitBreakerMetrics {
    /// Total requests attempted.
    pub total_requests: u64,
    /// Total successful requests.
    pub total_successes: u64,
    /// Total failed requests.
    pub total_failures: u64,
    /// Total requests rejected due to open breaker.
    pub total_rejected: u64,
    /// Current consecutive failure count.
    pub consecutive_failures: u32,
    /// Number of times the breaker has tripped (closed -> open).
    pub trip_count: u64,
    /// Number of times the breaker has recovered (half-open -> closed).
    pub recovery_count: u64,
}

impl CircuitBreakerMetrics {
    /// Current failure rate as a fraction (0.0..=1.0).
    #[must_use]
    pub fn failure_rate(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            self.total_failures as f64 / self.total_requests as f64
        }
    }
}

/// A circuit breaker instance managing a single execution context.
#[derive(Debug, Clone)]
pub struct CircuitBreaker {
    config: CircuitBreakerConfig,
    state: CircuitState,
    metrics: CircuitBreakerMetrics,
    /// Timestamp (ms) when the breaker last opened.
    opened_at_ms: Option<u64>,
    /// Number of probes attempted in current half-open window.
    half_open_probes: u32,
}

/// Result of checking whether a request is allowed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CircuitDecision {
    /// Request is allowed to proceed.
    Allow,
    /// Request is rejected because the breaker is open.
    Reject {
        /// Estimated time remaining before half-open in milliseconds.
        remaining_cooldown_ms: u64,
    },
}

impl CircuitBreaker {
    /// Create a new circuit breaker with the given configuration.
    #[must_use]
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            config,
            state: CircuitState::Closed,
            metrics: CircuitBreakerMetrics::default(),
            opened_at_ms: None,
            half_open_probes: 0,
        }
    }

    /// Current state.
    #[must_use]
    pub fn state(&self) -> CircuitState {
        self.state
    }

    /// Reference to metrics.
    #[must_use]
    pub fn metrics(&self) -> &CircuitBreakerMetrics {
        &self.metrics
    }

    /// Configuration reference.
    #[must_use]
    pub fn config(&self) -> &CircuitBreakerConfig {
        &self.config
    }

    /// Check whether a request is allowed at the given wall-clock time.
    ///
    /// This call transitions Open -> HalfOpen when the cooldown expires.
    #[must_use]
    pub fn check(&mut self, now_ms: u64) -> CircuitDecision {
        match self.state {
            CircuitState::Closed => CircuitDecision::Allow,
            CircuitState::Open => {
                let opened = self.opened_at_ms.unwrap_or(now_ms);
                let elapsed_ms = now_ms.saturating_sub(opened);
                let cooldown_ms = self.config.cooldown_period.as_millis() as u64;

                if elapsed_ms >= cooldown_ms {
                    // Transition to half-open
                    self.state = CircuitState::HalfOpen;
                    self.half_open_probes = 0;
                    CircuitDecision::Allow
                } else {
                    self.metrics.total_rejected += 1;
                    CircuitDecision::Reject {
                        remaining_cooldown_ms: cooldown_ms.saturating_sub(elapsed_ms),
                    }
                }
            }
            CircuitState::HalfOpen => {
                if self.half_open_probes < self.config.half_open_max_probes {
                    CircuitDecision::Allow
                } else {
                    self.metrics.total_rejected += 1;
                    CircuitDecision::Reject {
                        remaining_cooldown_ms: 0,
                    }
                }
            }
        }
    }

    /// Record a successful request.
    pub fn record_success(&mut self) {
        self.metrics.total_requests += 1;
        self.metrics.total_successes += 1;
        self.metrics.consecutive_failures = 0;

        match self.state {
            CircuitState::HalfOpen => {
                self.half_open_probes += 1;
                if self.half_open_probes >= self.config.half_open_max_probes {
                    // All probes succeeded: recover
                    self.state = CircuitState::Closed;
                    self.opened_at_ms = None;
                    self.metrics.recovery_count += 1;
                }
            }
            CircuitState::Open => {
                // Shouldn't happen in normal flow, but handle gracefully
            }
            CircuitState::Closed => {}
        }
    }

    /// Record a failed request at the given time.
    pub fn record_failure(&mut self, now_ms: u64) {
        self.metrics.total_requests += 1;
        self.metrics.total_failures += 1;
        self.metrics.consecutive_failures += 1;

        match self.state {
            CircuitState::Closed => {
                if self.should_trip() {
                    self.trip(now_ms);
                }
            }
            CircuitState::HalfOpen => {
                // Any failure in half-open immediately re-opens
                self.half_open_probes += 1;
                self.trip(now_ms);
            }
            CircuitState::Open => {}
        }
    }

    /// Force-reset the breaker to Closed state. Clears consecutive failures.
    pub fn reset(&mut self) {
        self.state = CircuitState::Closed;
        self.opened_at_ms = None;
        self.half_open_probes = 0;
        self.metrics.consecutive_failures = 0;
    }

    /// Check whether the breaker should trip based on current metrics.
    fn should_trip(&self) -> bool {
        if self.metrics.total_requests < u64::from(self.config.minimum_request_count) {
            return false;
        }

        if self.metrics.consecutive_failures < self.config.failure_threshold {
            return false;
        }

        // If a failure rate threshold is configured, also check that
        if let Some(rate_threshold) = self.config.failure_rate_threshold {
            if self.metrics.failure_rate() < rate_threshold {
                return false;
            }
        }

        true
    }

    /// Transition to Open state.
    fn trip(&mut self, now_ms: u64) {
        self.state = CircuitState::Open;
        self.opened_at_ms = Some(now_ms);
        self.half_open_probes = 0;
        self.metrics.trip_count += 1;
    }
}

/// Registry of named circuit breakers for managing multiple task types.
#[derive(Debug, Default)]
pub struct CircuitBreakerRegistry {
    breakers: HashMap<String, CircuitBreaker>,
}

impl CircuitBreakerRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new circuit breaker. Overwrites any existing one with the same name.
    pub fn register(&mut self, breaker: CircuitBreaker) {
        let name = breaker.config.name.clone();
        self.breakers.insert(name, breaker);
    }

    /// Get a mutable reference to a breaker by name.
    pub fn get_mut(&mut self, name: &str) -> Option<&mut CircuitBreaker> {
        self.breakers.get_mut(name)
    }

    /// Get an immutable reference to a breaker by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&CircuitBreaker> {
        self.breakers.get(name)
    }

    /// Check whether a named breaker allows a request.
    ///
    /// Returns `Allow` if the breaker does not exist (fail-open).
    #[must_use]
    pub fn check(&mut self, name: &str, now_ms: u64) -> CircuitDecision {
        match self.breakers.get_mut(name) {
            Some(breaker) => breaker.check(now_ms),
            None => CircuitDecision::Allow,
        }
    }

    /// Record a success on a named breaker.
    pub fn record_success(&mut self, name: &str) {
        if let Some(b) = self.breakers.get_mut(name) {
            b.record_success();
        }
    }

    /// Record a failure on a named breaker.
    pub fn record_failure(&mut self, name: &str, now_ms: u64) {
        if let Some(b) = self.breakers.get_mut(name) {
            b.record_failure(now_ms);
        }
    }

    /// Number of registered breakers.
    #[must_use]
    pub fn count(&self) -> usize {
        self.breakers.len()
    }

    /// List all registered breaker names.
    #[must_use]
    pub fn names(&self) -> Vec<&str> {
        self.breakers.keys().map(String::as_str).collect()
    }

    /// Remove a breaker by name.
    pub fn remove(&mut self, name: &str) -> Option<CircuitBreaker> {
        self.breakers.remove(name)
    }

    /// Get a summary of all breakers' states.
    #[must_use]
    pub fn summary(&self) -> Vec<CircuitBreakerSummary> {
        self.breakers
            .values()
            .map(|b| CircuitBreakerSummary {
                name: b.config.name.clone(),
                state: b.state,
                consecutive_failures: b.metrics.consecutive_failures,
                total_requests: b.metrics.total_requests,
                failure_rate: b.metrics.failure_rate(),
                trip_count: b.metrics.trip_count,
            })
            .collect()
    }
}

/// Summary of a circuit breaker's state for reporting.
#[derive(Debug, Clone)]
pub struct CircuitBreakerSummary {
    /// Breaker name.
    pub name: String,
    /// Current state.
    pub state: CircuitState,
    /// Current consecutive failure count.
    pub consecutive_failures: u32,
    /// Total requests processed.
    pub total_requests: u64,
    /// Current failure rate.
    pub failure_rate: f64,
    /// Number of times tripped.
    pub trip_count: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(threshold: u32, cooldown_ms: u64) -> CircuitBreakerConfig {
        CircuitBreakerConfig::new(
            "test-breaker",
            threshold,
            Duration::from_millis(cooldown_ms),
        )
    }

    // --- CircuitState ---

    #[test]
    fn test_circuit_state_display() {
        assert_eq!(CircuitState::Closed.to_string(), "Closed");
        assert_eq!(CircuitState::Open.to_string(), "Open");
        assert_eq!(CircuitState::HalfOpen.to_string(), "HalfOpen");
    }

    // --- CircuitBreakerConfig ---

    #[test]
    fn test_config_creation() {
        let config = make_config(5, 10_000);
        assert_eq!(config.failure_threshold, 5);
        assert_eq!(config.cooldown_period, Duration::from_secs(10));
        assert_eq!(config.half_open_max_probes, 1);
        assert_eq!(config.minimum_request_count, 1);
        assert!(config.failure_rate_threshold.is_none());
    }

    #[test]
    fn test_config_with_builder_methods() {
        let config = make_config(3, 5000)
            .with_half_open_probes(3)
            .with_minimum_requests(10)
            .with_failure_rate_threshold(0.5);
        assert_eq!(config.half_open_max_probes, 3);
        assert_eq!(config.minimum_request_count, 10);
        assert_eq!(config.failure_rate_threshold, Some(0.5));
    }

    #[test]
    fn test_failure_rate_threshold_clamp() {
        let config = make_config(3, 5000).with_failure_rate_threshold(1.5);
        assert_eq!(config.failure_rate_threshold, Some(1.0));
    }

    // --- CircuitBreakerMetrics ---

    #[test]
    fn test_metrics_failure_rate_zero_requests() {
        let metrics = CircuitBreakerMetrics::default();
        assert!((metrics.failure_rate() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_metrics_failure_rate_computed() {
        let metrics = CircuitBreakerMetrics {
            total_requests: 10,
            total_failures: 3,
            ..Default::default()
        };
        assert!((metrics.failure_rate() - 0.3).abs() < f64::EPSILON);
    }

    // --- CircuitBreaker core ---

    #[test]
    fn test_starts_closed() {
        let cb = CircuitBreaker::new(make_config(3, 5000));
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn test_closed_allows_requests() {
        let mut cb = CircuitBreaker::new(make_config(3, 5000));
        assert_eq!(cb.check(0), CircuitDecision::Allow);
    }

    #[test]
    fn test_trips_after_threshold_failures() {
        let mut cb = CircuitBreaker::new(make_config(3, 5000));
        cb.record_failure(100);
        assert_eq!(cb.state(), CircuitState::Closed);
        cb.record_failure(200);
        assert_eq!(cb.state(), CircuitState::Closed);
        cb.record_failure(300);
        assert_eq!(cb.state(), CircuitState::Open);
        assert_eq!(cb.metrics().trip_count, 1);
    }

    #[test]
    fn test_open_rejects_requests() {
        let mut cb = CircuitBreaker::new(make_config(2, 5000));
        cb.record_failure(100);
        cb.record_failure(200);
        assert_eq!(cb.state(), CircuitState::Open);

        match cb.check(300) {
            CircuitDecision::Reject {
                remaining_cooldown_ms,
            } => {
                assert!(remaining_cooldown_ms > 0);
            }
            CircuitDecision::Allow => panic!("expected reject"),
        }
    }

    #[test]
    fn test_transitions_to_half_open_after_cooldown() {
        let mut cb = CircuitBreaker::new(make_config(2, 1000));
        cb.record_failure(0);
        cb.record_failure(100);
        assert_eq!(cb.state(), CircuitState::Open);

        // Before cooldown
        assert!(matches!(cb.check(500), CircuitDecision::Reject { .. }));

        // After cooldown
        assert_eq!(cb.check(1200), CircuitDecision::Allow);
        assert_eq!(cb.state(), CircuitState::HalfOpen);
    }

    #[test]
    fn test_half_open_success_recovers() {
        let mut cb = CircuitBreaker::new(make_config(2, 1000));
        cb.record_failure(0);
        cb.record_failure(100);

        // Move to half-open
        assert_eq!(cb.check(1200), CircuitDecision::Allow);
        assert_eq!(cb.state(), CircuitState::HalfOpen);

        // Record success -> recovers to Closed
        cb.record_success();
        assert_eq!(cb.state(), CircuitState::Closed);
        assert_eq!(cb.metrics().recovery_count, 1);
    }

    #[test]
    fn test_half_open_failure_re_opens() {
        let mut cb = CircuitBreaker::new(make_config(2, 1000));
        cb.record_failure(0);
        cb.record_failure(100);

        // Move to half-open
        assert_eq!(cb.check(1200), CircuitDecision::Allow);

        // Failure in half-open re-opens
        cb.record_failure(1300);
        assert_eq!(cb.state(), CircuitState::Open);
        assert_eq!(cb.metrics().trip_count, 2);
    }

    #[test]
    fn test_half_open_max_probes() {
        let config = make_config(2, 1000).with_half_open_probes(2);
        let mut cb = CircuitBreaker::new(config);
        cb.record_failure(0);
        cb.record_failure(100);

        // Move to half-open
        assert_eq!(cb.check(1200), CircuitDecision::Allow);
        assert_eq!(cb.state(), CircuitState::HalfOpen);

        // First probe passes
        assert_eq!(cb.check(1300), CircuitDecision::Allow);

        // Second probe: still needs one more success to close
        // But third call should be rejected (max 2 probes)
        cb.record_success(); // probe 1 success
        cb.record_success(); // probe 2 success -> recovers
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn test_success_resets_consecutive_failures() {
        let mut cb = CircuitBreaker::new(make_config(3, 5000));
        cb.record_failure(100);
        cb.record_failure(200);
        assert_eq!(cb.metrics().consecutive_failures, 2);

        cb.record_success();
        assert_eq!(cb.metrics().consecutive_failures, 0);

        // Now need 3 more failures to trip
        cb.record_failure(300);
        cb.record_failure(400);
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn test_minimum_request_count() {
        let config = make_config(2, 5000).with_minimum_requests(5);
        let mut cb = CircuitBreaker::new(config);

        // Fail 2 times, but total requests < 5 so no trip
        cb.record_failure(100);
        cb.record_failure(200);
        assert_eq!(cb.state(), CircuitState::Closed);

        // Need more requests to reach minimum
        cb.record_success();
        cb.record_success();
        cb.record_failure(300); // now at 5 total, 1 consecutive -> no trip
        cb.record_failure(400); // 6 total, 2 consecutive -> trip!
        assert_eq!(cb.state(), CircuitState::Open);
    }

    #[test]
    fn test_failure_rate_threshold() {
        let config = make_config(2, 5000)
            .with_minimum_requests(1)
            .with_failure_rate_threshold(0.5);
        let mut cb = CircuitBreaker::new(config);

        // Many successes, then 2 failures -> rate < 0.5 so no trip
        for _ in 0..10 {
            cb.record_success();
        }
        cb.record_failure(100);
        cb.record_failure(200);
        // failure_rate = 2/12 ≈ 0.167 < 0.5 -> no trip
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn test_reset() {
        let mut cb = CircuitBreaker::new(make_config(2, 5000));
        cb.record_failure(0);
        cb.record_failure(100);
        assert_eq!(cb.state(), CircuitState::Open);

        cb.reset();
        assert_eq!(cb.state(), CircuitState::Closed);
        assert_eq!(cb.metrics().consecutive_failures, 0);
    }

    // --- CircuitBreakerRegistry ---

    #[test]
    fn test_registry_register_and_get() {
        let mut registry = CircuitBreakerRegistry::new();
        registry.register(CircuitBreaker::new(make_config(3, 5000)));
        assert_eq!(registry.count(), 1);
        assert!(registry.get("test-breaker").is_some());
    }

    #[test]
    fn test_registry_check_unknown_allows() {
        let mut registry = CircuitBreakerRegistry::new();
        assert_eq!(registry.check("nonexistent", 0), CircuitDecision::Allow);
    }

    #[test]
    fn test_registry_record_success_failure() {
        let mut registry = CircuitBreakerRegistry::new();
        registry.register(CircuitBreaker::new(make_config(2, 5000)));

        registry.record_failure("test-breaker", 100);
        registry.record_failure("test-breaker", 200);

        let breaker = registry.get("test-breaker").expect("should exist");
        assert_eq!(breaker.state(), CircuitState::Open);

        // Success on unknown breaker should not panic
        registry.record_success("nonexistent");
    }

    #[test]
    fn test_registry_remove() {
        let mut registry = CircuitBreakerRegistry::new();
        registry.register(CircuitBreaker::new(make_config(3, 5000)));
        let removed = registry.remove("test-breaker");
        assert!(removed.is_some());
        assert_eq!(registry.count(), 0);
    }

    #[test]
    fn test_registry_names() {
        let mut registry = CircuitBreakerRegistry::new();
        registry.register(CircuitBreaker::new(CircuitBreakerConfig::new(
            "alpha",
            3,
            Duration::from_secs(5),
        )));
        registry.register(CircuitBreaker::new(CircuitBreakerConfig::new(
            "beta",
            3,
            Duration::from_secs(5),
        )));

        let mut names = registry.names();
        names.sort();
        assert_eq!(names, vec!["alpha", "beta"]);
    }

    #[test]
    fn test_registry_summary() {
        let mut registry = CircuitBreakerRegistry::new();
        let mut cb = CircuitBreaker::new(make_config(2, 5000));
        cb.record_failure(0);
        cb.record_failure(100);
        registry.register(cb);

        let summary = registry.summary();
        assert_eq!(summary.len(), 1);
        assert_eq!(summary[0].state, CircuitState::Open);
        assert_eq!(summary[0].trip_count, 1);
    }

    #[test]
    fn test_full_lifecycle() {
        let mut cb = CircuitBreaker::new(make_config(3, 2000).with_half_open_probes(1));

        // Phase 1: Closed, accumulate failures
        cb.record_success();
        cb.record_failure(100);
        cb.record_failure(200);
        cb.record_failure(300);
        assert_eq!(cb.state(), CircuitState::Open);

        // Phase 2: Open, requests rejected
        assert!(matches!(cb.check(500), CircuitDecision::Reject { .. }));
        assert!(matches!(cb.check(1000), CircuitDecision::Reject { .. }));

        // Phase 3: Cooldown expires -> HalfOpen
        assert_eq!(cb.check(2500), CircuitDecision::Allow);
        assert_eq!(cb.state(), CircuitState::HalfOpen);

        // Phase 4: Probe succeeds -> recovery
        cb.record_success();
        assert_eq!(cb.state(), CircuitState::Closed);
        assert_eq!(cb.metrics().recovery_count, 1);
        assert_eq!(cb.metrics().trip_count, 1);
        assert_eq!(cb.metrics().total_requests, 5);
    }

    #[test]
    fn test_rejected_counter() {
        let mut cb = CircuitBreaker::new(make_config(1, 5000));
        cb.record_failure(0);
        assert_eq!(cb.state(), CircuitState::Open);

        // Multiple rejections
        let _ = cb.check(100);
        let _ = cb.check(200);
        let _ = cb.check(300);
        assert_eq!(cb.metrics().total_rejected, 3);
    }
}
