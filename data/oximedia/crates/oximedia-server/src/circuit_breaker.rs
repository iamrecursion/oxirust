#![allow(dead_code)]
//! Circuit breaker pattern for service resilience.
//!
//! Implements a three-state circuit breaker (Closed, Open, Half-Open)
//! that protects downstream services from cascading failures by
//! short-circuiting calls once a failure threshold is reached.

use std::time::{Duration, Instant};

/// The three states of the circuit breaker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CircuitState {
    /// Normal operation – requests pass through.
    Closed,
    /// Failures exceeded threshold – requests are immediately rejected.
    Open,
    /// Trial period – a limited number of requests are allowed through.
    HalfOpen,
}

impl CircuitState {
    /// Returns `true` when requests should be allowed.
    pub fn allows_request(&self) -> bool {
        matches!(self, Self::Closed | Self::HalfOpen)
    }

    /// Human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Closed => "Closed",
            Self::Open => "Open",
            Self::HalfOpen => "Half-Open",
        }
    }
}

/// Configuration for the circuit breaker.
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Number of consecutive failures before the circuit opens.
    pub failure_threshold: u32,
    /// Duration the circuit stays open before transitioning to half-open.
    pub open_duration: Duration,
    /// Number of trial successes required to close the circuit from half-open.
    pub half_open_successes: u32,
    /// Optional name for identification in logs.
    pub name: String,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            open_duration: Duration::from_secs(30),
            half_open_successes: 2,
            name: "default".to_string(),
        }
    }
}

impl CircuitBreakerConfig {
    /// Creates a config with the given name.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Sets the failure threshold.
    #[must_use]
    pub fn with_failure_threshold(mut self, threshold: u32) -> Self {
        self.failure_threshold = threshold;
        self
    }

    /// Sets the open duration.
    #[must_use]
    pub fn with_open_duration(mut self, duration: Duration) -> Self {
        self.open_duration = duration;
        self
    }

    /// Sets the half-open success count.
    #[must_use]
    pub fn with_half_open_successes(mut self, count: u32) -> Self {
        self.half_open_successes = count;
        self
    }
}

/// Statistics tracked by the circuit breaker.
#[derive(Debug, Clone)]
pub struct CircuitBreakerStats {
    /// Total requests attempted.
    pub total_requests: u64,
    /// Successful requests.
    pub successful: u64,
    /// Failed requests.
    pub failed: u64,
    /// Requests rejected while open.
    pub rejected: u64,
    /// Number of times the circuit tripped (closed -> open).
    pub trips: u64,
    /// Number of times the circuit recovered (half-open -> closed).
    pub recoveries: u64,
}

impl CircuitBreakerStats {
    /// Creates zeroed stats.
    pub fn new() -> Self {
        Self {
            total_requests: 0,
            successful: 0,
            failed: 0,
            rejected: 0,
            trips: 0,
            recoveries: 0,
        }
    }

    /// Success rate as a fraction (0.0 – 1.0).
    #[allow(clippy::cast_precision_loss)]
    pub fn success_rate(&self) -> f64 {
        let attempted = self.successful + self.failed;
        if attempted == 0 {
            return 1.0;
        }
        self.successful as f64 / attempted as f64
    }

    /// Rejection rate.
    #[allow(clippy::cast_precision_loss)]
    pub fn rejection_rate(&self) -> f64 {
        if self.total_requests == 0 {
            return 0.0;
        }
        self.rejected as f64 / self.total_requests as f64
    }
}

impl Default for CircuitBreakerStats {
    fn default() -> Self {
        Self::new()
    }
}

/// The outcome of a request guarded by the circuit breaker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestOutcome {
    /// The request succeeded.
    Success,
    /// The request failed.
    Failure,
}

/// A circuit breaker instance.
pub struct CircuitBreaker {
    /// Configuration.
    config: CircuitBreakerConfig,
    /// Current state.
    state: CircuitState,
    /// Consecutive failure count (in closed state).
    consecutive_failures: u32,
    /// Consecutive success count (in half-open state).
    half_open_successes: u32,
    /// When the circuit was last opened.
    opened_at: Option<Instant>,
    /// Accumulated statistics.
    stats: CircuitBreakerStats,
}

impl CircuitBreaker {
    /// Creates a new circuit breaker in the Closed state.
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            config,
            state: CircuitState::Closed,
            consecutive_failures: 0,
            half_open_successes: 0,
            opened_at: None,
            stats: CircuitBreakerStats::new(),
        }
    }

    /// Returns the current state, potentially transitioning from Open to Half-Open.
    pub fn state(&mut self) -> CircuitState {
        self.maybe_transition_to_half_open();
        self.state
    }

    /// Checks whether a request should be allowed.
    /// Returns `true` if allowed, `false` if rejected.
    pub fn allow_request(&mut self) -> bool {
        self.maybe_transition_to_half_open();
        self.stats.total_requests += 1;
        if self.state == CircuitState::Open {
            self.stats.rejected += 1;
            return false;
        }
        true
    }

    /// Records the outcome of a request.
    pub fn record(&mut self, outcome: RequestOutcome) {
        match outcome {
            RequestOutcome::Success => self.on_success(),
            RequestOutcome::Failure => self.on_failure(),
        }
    }

    /// Returns the config name.
    pub fn name(&self) -> &str {
        &self.config.name
    }

    /// Returns current stats.
    pub fn stats(&self) -> &CircuitBreakerStats {
        &self.stats
    }

    /// Manually resets the circuit breaker to Closed.
    pub fn reset(&mut self) {
        self.state = CircuitState::Closed;
        self.consecutive_failures = 0;
        self.half_open_successes = 0;
        self.opened_at = None;
    }

    /// Manually trips the circuit breaker to Open.
    pub fn trip(&mut self) {
        self.state = CircuitState::Open;
        self.opened_at = Some(Instant::now());
        self.stats.trips += 1;
    }

    // ── Internal helpers ──

    fn on_success(&mut self) {
        self.stats.successful += 1;
        match self.state {
            CircuitState::Closed => {
                self.consecutive_failures = 0;
            }
            CircuitState::HalfOpen => {
                self.half_open_successes += 1;
                if self.half_open_successes >= self.config.half_open_successes {
                    self.state = CircuitState::Closed;
                    self.consecutive_failures = 0;
                    self.half_open_successes = 0;
                    self.stats.recoveries += 1;
                }
            }
            CircuitState::Open => { /* shouldn't happen, but ignore */ }
        }
    }

    fn on_failure(&mut self) {
        self.stats.failed += 1;
        match self.state {
            CircuitState::Closed => {
                self.consecutive_failures += 1;
                if self.consecutive_failures >= self.config.failure_threshold {
                    self.trip();
                }
            }
            CircuitState::HalfOpen => {
                // Any failure in half-open immediately re-opens
                self.trip();
                self.half_open_successes = 0;
            }
            CircuitState::Open => { /* already open */ }
        }
    }

    fn maybe_transition_to_half_open(&mut self) {
        if self.state == CircuitState::Open {
            if let Some(opened) = self.opened_at {
                if opened.elapsed() >= self.config.open_duration {
                    self.state = CircuitState::HalfOpen;
                    self.half_open_successes = 0;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_breaker() -> CircuitBreaker {
        CircuitBreaker::new(
            CircuitBreakerConfig::default()
                .with_failure_threshold(3)
                .with_open_duration(Duration::from_millis(50))
                .with_half_open_successes(2),
        )
    }

    #[test]
    fn test_initial_state_is_closed() {
        let mut cb = default_breaker();
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn test_allows_request_when_closed() {
        let mut cb = default_breaker();
        assert!(cb.allow_request());
    }

    #[test]
    fn test_trips_after_threshold_failures() {
        let mut cb = default_breaker();
        for _ in 0..3 {
            cb.allow_request();
            cb.record(RequestOutcome::Failure);
        }
        assert_eq!(cb.state(), CircuitState::Open);
        assert_eq!(cb.stats().trips, 1);
    }

    #[test]
    fn test_rejects_when_open() {
        let mut cb = default_breaker();
        cb.trip();
        assert!(!cb.allow_request());
        assert_eq!(cb.stats().rejected, 1);
    }

    #[test]
    fn test_transitions_to_half_open_after_timeout() {
        let mut cb = CircuitBreaker::new(
            CircuitBreakerConfig::default()
                .with_failure_threshold(1)
                .with_open_duration(Duration::from_millis(1)),
        );
        cb.allow_request();
        cb.record(RequestOutcome::Failure);
        assert_eq!(cb.state, CircuitState::Open);
        std::thread::sleep(Duration::from_millis(5));
        assert_eq!(cb.state(), CircuitState::HalfOpen);
    }

    #[test]
    fn test_half_open_recovers_on_successes() {
        let mut cb = CircuitBreaker::new(
            CircuitBreakerConfig::default()
                .with_failure_threshold(1)
                .with_open_duration(Duration::from_millis(1))
                .with_half_open_successes(2),
        );
        cb.allow_request();
        cb.record(RequestOutcome::Failure);
        std::thread::sleep(Duration::from_millis(5));
        let _ = cb.state(); // transition to half-open

        cb.record(RequestOutcome::Success);
        cb.record(RequestOutcome::Success);
        assert_eq!(cb.state(), CircuitState::Closed);
        assert_eq!(cb.stats().recoveries, 1);
    }

    #[test]
    fn test_half_open_reopens_on_failure() {
        let mut cb = CircuitBreaker::new(
            CircuitBreakerConfig::default()
                .with_failure_threshold(1)
                .with_open_duration(Duration::from_millis(1))
                .with_half_open_successes(3),
        );
        cb.allow_request();
        cb.record(RequestOutcome::Failure);
        std::thread::sleep(Duration::from_millis(5));
        let _ = cb.state(); // half-open
        cb.record(RequestOutcome::Failure);
        assert_eq!(cb.state, CircuitState::Open);
    }

    #[test]
    fn test_success_resets_failure_count() {
        let mut cb = default_breaker();
        cb.allow_request();
        cb.record(RequestOutcome::Failure);
        cb.allow_request();
        cb.record(RequestOutcome::Failure);
        cb.allow_request();
        cb.record(RequestOutcome::Success);
        // Counter was reset; 1 more failure shouldn't trip
        cb.allow_request();
        cb.record(RequestOutcome::Failure);
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn test_manual_reset() {
        let mut cb = default_breaker();
        cb.trip();
        assert_eq!(cb.state, CircuitState::Open);
        cb.reset();
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn test_stats_success_rate() {
        let mut stats = CircuitBreakerStats::new();
        stats.successful = 9;
        stats.failed = 1;
        assert!((stats.success_rate() - 0.9).abs() < 1e-9);
    }

    #[test]
    fn test_stats_rejection_rate() {
        let mut stats = CircuitBreakerStats::new();
        stats.total_requests = 100;
        stats.rejected = 25;
        assert!((stats.rejection_rate() - 0.25).abs() < 1e-9);
    }

    #[test]
    fn test_circuit_state_labels() {
        assert_eq!(CircuitState::Closed.label(), "Closed");
        assert_eq!(CircuitState::Open.label(), "Open");
        assert_eq!(CircuitState::HalfOpen.label(), "Half-Open");
    }

    #[test]
    fn test_config_builder() {
        let cfg = CircuitBreakerConfig::default()
            .with_name("my-service")
            .with_failure_threshold(10)
            .with_open_duration(Duration::from_secs(60))
            .with_half_open_successes(5);
        assert_eq!(cfg.name, "my-service");
        assert_eq!(cfg.failure_threshold, 10);
        assert_eq!(cfg.half_open_successes, 5);
    }
}
