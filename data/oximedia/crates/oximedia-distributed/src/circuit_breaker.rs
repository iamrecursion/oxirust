#![allow(dead_code)]
//! Circuit breaker pattern for distributed systems fault isolation.
//!
//! Implements the classic three-state circuit breaker (Closed, Open, Half-Open)
//! to prevent cascading failures when a downstream service or node becomes
//! unavailable. Tracks failure rates and automatically transitions between
//! states based on configurable thresholds.

use std::fmt;
use std::time::Duration;

/// The three states of a circuit breaker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Normal operation: requests pass through. Failures are counted.
    Closed,
    /// Tripped: requests are immediately rejected without calling the downstream.
    Open,
    /// Probe state: a limited number of requests are allowed through to test recovery.
    HalfOpen,
}

impl fmt::Display for CircuitState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Closed => write!(f, "Closed"),
            Self::Open => write!(f, "Open"),
            Self::HalfOpen => write!(f, "Half-Open"),
        }
    }
}

/// Configuration for the circuit breaker.
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Number of consecutive failures before opening the circuit.
    pub failure_threshold: u32,
    /// Duration the circuit stays open before transitioning to half-open.
    pub open_duration: Duration,
    /// Number of successful probes in half-open state before closing.
    pub success_threshold: u32,
    /// Maximum requests allowed through during half-open state.
    pub half_open_max_requests: u32,
    /// Optional failure rate threshold (0.0..1.0). If set, the circuit opens
    /// when the failure rate exceeds this value over the sliding window.
    pub failure_rate_threshold: Option<f64>,
    /// Sliding window size for failure rate calculation.
    pub window_size: u32,
}

impl CircuitBreakerConfig {
    /// Create a new configuration with sensible defaults.
    #[must_use]
    pub fn new() -> Self {
        Self {
            failure_threshold: 5,
            open_duration: Duration::from_secs(30),
            success_threshold: 3,
            half_open_max_requests: 3,
            failure_rate_threshold: None,
            window_size: 20,
        }
    }

    /// Set the failure threshold.
    #[must_use]
    pub fn with_failure_threshold(mut self, threshold: u32) -> Self {
        self.failure_threshold = threshold;
        self
    }

    /// Set the open duration.
    #[must_use]
    pub fn with_open_duration(mut self, duration: Duration) -> Self {
        self.open_duration = duration;
        self
    }

    /// Set the success threshold for half-open to closed transition.
    #[must_use]
    pub fn with_success_threshold(mut self, threshold: u32) -> Self {
        self.success_threshold = threshold;
        self
    }

    /// Enable failure rate based tripping.
    #[must_use]
    pub fn with_failure_rate(mut self, rate: f64, window: u32) -> Self {
        self.failure_rate_threshold = Some(rate.clamp(0.0, 1.0));
        self.window_size = window;
        self
    }
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Event types emitted by the circuit breaker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CircuitEvent {
    /// Circuit opened (tripped) due to failures.
    Opened {
        /// The reason for opening.
        reason: String,
    },
    /// Circuit transitioned to half-open for probing.
    HalfOpened,
    /// Circuit closed (recovered).
    Closed,
    /// A request was rejected by the open circuit.
    Rejected,
    /// A probe request succeeded in half-open state.
    ProbeSuccess,
    /// A probe request failed in half-open state.
    ProbeFailure,
}

impl fmt::Display for CircuitEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Opened { reason } => write!(f, "Circuit Opened: {reason}"),
            Self::HalfOpened => write!(f, "Circuit Half-Opened"),
            Self::Closed => write!(f, "Circuit Closed"),
            Self::Rejected => write!(f, "Request Rejected"),
            Self::ProbeSuccess => write!(f, "Probe Success"),
            Self::ProbeFailure => write!(f, "Probe Failure"),
        }
    }
}

/// Sliding window of recent request outcomes for failure rate calculation.
#[derive(Debug, Clone)]
pub struct SlidingWindow {
    /// Ring buffer of outcomes (true = success, false = failure).
    outcomes: Vec<bool>,
    /// Current write position.
    position: usize,
    /// Total entries written.
    total_written: u64,
}

impl SlidingWindow {
    /// Create a new sliding window of the given size.
    #[must_use]
    pub fn new(size: u32) -> Self {
        Self {
            outcomes: vec![true; size as usize],
            position: 0,
            total_written: 0,
        }
    }

    /// Record a success.
    pub fn record_success(&mut self) {
        self.record(true);
    }

    /// Record a failure.
    pub fn record_failure(&mut self) {
        self.record(false);
    }

    /// Internal record.
    fn record(&mut self, success: bool) {
        if self.outcomes.is_empty() {
            return;
        }
        self.outcomes[self.position] = success;
        self.position = (self.position + 1) % self.outcomes.len();
        self.total_written += 1;
    }

    /// Compute the failure rate (0.0..1.0).
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn failure_rate(&self) -> f64 {
        if self.outcomes.is_empty() || self.total_written == 0 {
            return 0.0;
        }
        let effective_len = (self.total_written as usize).min(self.outcomes.len());
        let failures = self.outcomes[..effective_len]
            .iter()
            .filter(|&&s| !s)
            .count();
        failures as f64 / effective_len as f64
    }

    /// Number of entries in the window.
    #[must_use]
    pub fn size(&self) -> usize {
        self.outcomes.len()
    }

    /// Reset all entries to success.
    pub fn reset(&mut self) {
        for o in &mut self.outcomes {
            *o = true;
        }
        self.position = 0;
        self.total_written = 0;
    }
}

/// The circuit breaker state machine.
#[derive(Debug, Clone)]
pub struct CircuitBreaker {
    /// The name/identifier for this circuit breaker.
    pub name: String,
    /// Current state.
    state: CircuitState,
    /// Configuration.
    config: CircuitBreakerConfig,
    /// Consecutive failure count (in Closed state).
    consecutive_failures: u32,
    /// Consecutive success count (in `HalfOpen` state).
    consecutive_successes: u32,
    /// Requests sent during half-open.
    half_open_requests: u32,
    /// Sliding window for rate-based detection.
    window: SlidingWindow,
    /// Event log.
    events: Vec<CircuitEvent>,
    /// Timestamp (as a simple monotonic counter) when the circuit opened.
    opened_at_tick: u64,
    /// Current tick counter (simulated time for testability).
    current_tick: u64,
}

impl CircuitBreaker {
    /// Create a new circuit breaker.
    #[must_use]
    pub fn new(name: &str, config: CircuitBreakerConfig) -> Self {
        let window_size = config.window_size;
        Self {
            name: name.to_string(),
            state: CircuitState::Closed,
            config,
            consecutive_failures: 0,
            consecutive_successes: 0,
            half_open_requests: 0,
            window: SlidingWindow::new(window_size),
            events: Vec::new(),
            opened_at_tick: 0,
            current_tick: 0,
        }
    }

    /// Get the current state.
    #[must_use]
    pub fn state(&self) -> CircuitState {
        self.state
    }

    /// Advance the internal tick by the given amount (milliseconds).
    pub fn advance_tick(&mut self, millis: u64) {
        self.current_tick += millis;
        // Check if we should transition from Open to HalfOpen
        if self.state == CircuitState::Open {
            let elapsed = self.current_tick - self.opened_at_tick;
            if elapsed >= self.config.open_duration.as_millis() as u64 {
                self.transition_to(CircuitState::HalfOpen);
            }
        }
    }

    /// Check if a request should be allowed through.
    pub fn allow_request(&mut self) -> bool {
        match self.state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                self.emit(CircuitEvent::Rejected);
                false
            }
            CircuitState::HalfOpen => {
                if self.half_open_requests < self.config.half_open_max_requests {
                    self.half_open_requests += 1;
                    true
                } else {
                    self.emit(CircuitEvent::Rejected);
                    false
                }
            }
        }
    }

    /// Record a successful request outcome.
    pub fn record_success(&mut self) {
        self.window.record_success();
        match self.state {
            CircuitState::Closed => {
                self.consecutive_failures = 0;
            }
            CircuitState::HalfOpen => {
                self.consecutive_successes += 1;
                self.emit(CircuitEvent::ProbeSuccess);
                if self.consecutive_successes >= self.config.success_threshold {
                    self.transition_to(CircuitState::Closed);
                }
            }
            CircuitState::Open => {}
        }
    }

    /// Record a failed request outcome.
    pub fn record_failure(&mut self) {
        self.window.record_failure();
        match self.state {
            CircuitState::Closed => {
                self.consecutive_failures += 1;
                if self.consecutive_failures >= self.config.failure_threshold {
                    self.trip("Consecutive failure threshold exceeded".to_string());
                } else if let Some(rate_threshold) = self.config.failure_rate_threshold {
                    if self.window.failure_rate() > rate_threshold {
                        self.trip(format!(
                            "Failure rate {:.2}% exceeded threshold {:.2}%",
                            self.window.failure_rate() * 100.0,
                            rate_threshold * 100.0,
                        ));
                    }
                }
            }
            CircuitState::HalfOpen => {
                self.emit(CircuitEvent::ProbeFailure);
                self.trip("Probe failed in half-open state".to_string());
            }
            CircuitState::Open => {}
        }
    }

    /// Force-open the circuit.
    pub fn trip(&mut self, reason: String) {
        self.emit(CircuitEvent::Opened { reason });
        self.state = CircuitState::Open;
        self.opened_at_tick = self.current_tick;
        self.consecutive_failures = 0;
        self.consecutive_successes = 0;
        self.half_open_requests = 0;
    }

    /// Force-close (reset) the circuit.
    pub fn reset(&mut self) {
        self.transition_to(CircuitState::Closed);
        self.window.reset();
    }

    /// Get the event history.
    #[must_use]
    pub fn events(&self) -> &[CircuitEvent] {
        &self.events
    }

    /// Get the number of consecutive failures.
    #[must_use]
    pub fn consecutive_failures(&self) -> u32 {
        self.consecutive_failures
    }

    /// Get the current failure rate from the sliding window.
    #[must_use]
    pub fn failure_rate(&self) -> f64 {
        self.window.failure_rate()
    }

    /// Internal transition helper.
    fn transition_to(&mut self, new_state: CircuitState) {
        self.state = new_state;
        match new_state {
            CircuitState::Closed => {
                self.consecutive_failures = 0;
                self.consecutive_successes = 0;
                self.half_open_requests = 0;
                self.emit(CircuitEvent::Closed);
            }
            CircuitState::HalfOpen => {
                self.consecutive_successes = 0;
                self.half_open_requests = 0;
                self.emit(CircuitEvent::HalfOpened);
            }
            CircuitState::Open => {
                // Handled by trip()
            }
        }
    }

    /// Emit an event.
    fn emit(&mut self, event: CircuitEvent) {
        self.events.push(event);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_state_display() {
        assert_eq!(CircuitState::Closed.to_string(), "Closed");
        assert_eq!(CircuitState::Open.to_string(), "Open");
        assert_eq!(CircuitState::HalfOpen.to_string(), "Half-Open");
    }

    #[test]
    fn test_config_defaults() {
        let config = CircuitBreakerConfig::default();
        assert_eq!(config.failure_threshold, 5);
        assert_eq!(config.success_threshold, 3);
        assert_eq!(config.open_duration, Duration::from_secs(30));
    }

    #[test]
    fn test_config_builder() {
        let config = CircuitBreakerConfig::new()
            .with_failure_threshold(10)
            .with_open_duration(Duration::from_secs(60))
            .with_success_threshold(5);
        assert_eq!(config.failure_threshold, 10);
        assert_eq!(config.open_duration, Duration::from_secs(60));
        assert_eq!(config.success_threshold, 5);
    }

    #[test]
    fn test_circuit_starts_closed() {
        let cb = CircuitBreaker::new("test", CircuitBreakerConfig::default());
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn test_closed_allows_requests() {
        let mut cb = CircuitBreaker::new("test", CircuitBreakerConfig::default());
        assert!(cb.allow_request());
    }

    #[test]
    fn test_consecutive_failures_trip() {
        let config = CircuitBreakerConfig::new().with_failure_threshold(3);
        let mut cb = CircuitBreaker::new("test", config);

        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Closed);
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Closed);
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
    }

    #[test]
    fn test_open_rejects_requests() {
        let config = CircuitBreakerConfig::new().with_failure_threshold(1);
        let mut cb = CircuitBreaker::new("test", config);
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
        assert!(!cb.allow_request());
    }

    #[test]
    fn test_open_to_half_open_transition() {
        let config = CircuitBreakerConfig::new()
            .with_failure_threshold(1)
            .with_open_duration(Duration::from_secs(1));
        let mut cb = CircuitBreaker::new("test", config);
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);

        cb.advance_tick(999);
        assert_eq!(cb.state(), CircuitState::Open);

        cb.advance_tick(1);
        assert_eq!(cb.state(), CircuitState::HalfOpen);
    }

    #[test]
    fn test_half_open_allows_limited_requests() {
        let config = CircuitBreakerConfig::new()
            .with_failure_threshold(1)
            .with_open_duration(Duration::from_millis(100))
            .with_success_threshold(2);
        let mut cb = CircuitBreaker::new("test", config);
        cb.record_failure();
        cb.advance_tick(100);
        assert_eq!(cb.state(), CircuitState::HalfOpen);

        // Should allow up to half_open_max_requests
        assert!(cb.allow_request());
    }

    #[test]
    fn test_half_open_success_closes() {
        let config = CircuitBreakerConfig::new()
            .with_failure_threshold(1)
            .with_open_duration(Duration::from_millis(100))
            .with_success_threshold(2);
        let mut cb = CircuitBreaker::new("test", config);
        cb.record_failure();
        cb.advance_tick(100);
        assert_eq!(cb.state(), CircuitState::HalfOpen);

        cb.record_success();
        assert_eq!(cb.state(), CircuitState::HalfOpen);
        cb.record_success();
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn test_half_open_failure_reopens() {
        let config = CircuitBreakerConfig::new()
            .with_failure_threshold(1)
            .with_open_duration(Duration::from_millis(100))
            .with_success_threshold(3);
        let mut cb = CircuitBreaker::new("test", config);
        cb.record_failure();
        cb.advance_tick(100);
        assert_eq!(cb.state(), CircuitState::HalfOpen);

        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
    }

    #[test]
    fn test_success_resets_failure_count() {
        let config = CircuitBreakerConfig::new().with_failure_threshold(3);
        let mut cb = CircuitBreaker::new("test", config);
        cb.record_failure();
        cb.record_failure();
        cb.record_success(); // should reset counter
        cb.record_failure();
        cb.record_failure();
        // Only 2 consecutive failures, not 3
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn test_force_reset() {
        let config = CircuitBreakerConfig::new().with_failure_threshold(1);
        let mut cb = CircuitBreaker::new("test", config);
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
        cb.reset();
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn test_events_emitted() {
        let config = CircuitBreakerConfig::new().with_failure_threshold(1);
        let mut cb = CircuitBreaker::new("test", config);
        cb.record_failure();
        let events = cb.events();
        assert!(!events.is_empty());
        assert!(matches!(events.last(), Some(CircuitEvent::Opened { .. })));
    }

    #[test]
    fn test_sliding_window_failure_rate() {
        let mut w = SlidingWindow::new(10);
        // Record 3 failures and 7 successes
        for _ in 0..7 {
            w.record_success();
        }
        for _ in 0..3 {
            w.record_failure();
        }
        let rate = w.failure_rate();
        assert!((rate - 0.3).abs() < 0.01);
    }

    #[test]
    fn test_sliding_window_empty() {
        let w = SlidingWindow::new(10);
        assert_eq!(w.failure_rate(), 0.0);
    }

    #[test]
    fn test_sliding_window_reset() {
        let mut w = SlidingWindow::new(5);
        w.record_failure();
        w.record_failure();
        w.reset();
        assert_eq!(w.failure_rate(), 0.0);
    }

    #[test]
    fn test_failure_rate_threshold() {
        let config = CircuitBreakerConfig::new()
            .with_failure_threshold(100) // high threshold so only rate matters
            .with_failure_rate(0.5, 10);
        let mut cb = CircuitBreaker::new("test", config);
        // Fill window with failures to exceed 50% rate
        for _ in 0..10 {
            cb.record_failure();
            if cb.state() == CircuitState::Open {
                break;
            }
        }
        assert_eq!(cb.state(), CircuitState::Open);
    }

    #[test]
    fn test_circuit_event_display() {
        let ev = CircuitEvent::Opened {
            reason: "test".to_string(),
        };
        assert!(ev.to_string().contains("test"));
        assert_eq!(CircuitEvent::Closed.to_string(), "Circuit Closed");
    }
}
