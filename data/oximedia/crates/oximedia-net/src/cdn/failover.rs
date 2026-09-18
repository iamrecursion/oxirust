//! Failover management and circuit breaker implementation.
//!
//! This module provides automatic failover with circuit breaker pattern,
//! exponential backoff, and graceful degradation.

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use std::time::SystemTime;

/// Circuit breaker state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CircuitState {
    /// Circuit is closed, requests flow normally.
    Closed,
    /// Circuit is open, requests are blocked.
    Open,
    /// Circuit is half-open, testing if service recovered.
    HalfOpen,
}

impl CircuitState {
    /// Returns true if requests should be allowed.
    #[must_use]
    pub const fn allows_requests(&self) -> bool {
        matches!(self, Self::Closed | Self::HalfOpen)
    }

    /// Returns the state name.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Closed => "Closed",
            Self::Open => "Open",
            Self::HalfOpen => "Half-Open",
        }
    }
}

/// Circuit breaker configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    /// Failure threshold before opening circuit.
    pub failure_threshold: u32,
    /// Success threshold to close circuit from half-open.
    pub success_threshold: u32,
    /// Timeout before attempting recovery.
    pub timeout: Duration,
    /// Half-open request limit.
    pub half_open_requests: u32,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            success_threshold: 2,
            timeout: Duration::from_secs(60),
            half_open_requests: 3,
        }
    }
}

/// Circuit breaker for a single provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreaker {
    /// Provider ID.
    pub provider_id: String,
    /// Current state.
    pub state: CircuitState,
    /// Configuration.
    pub config: CircuitBreakerConfig,
    /// Consecutive failure count.
    pub consecutive_failures: u32,
    /// Consecutive success count.
    pub consecutive_successes: u32,
    /// Total failures.
    pub total_failures: u64,
    /// Total successes.
    pub total_successes: u64,
    /// Time when circuit opened.
    pub opened_at: Option<SystemTime>,
    /// Time when last state change occurred.
    pub last_state_change: SystemTime,
    /// Half-open request count.
    pub half_open_request_count: u32,
}

impl CircuitBreaker {
    /// Creates a new circuit breaker.
    #[must_use]
    pub fn new(provider_id: String) -> Self {
        Self::with_config(provider_id, CircuitBreakerConfig::default())
    }

    /// Creates a circuit breaker with custom configuration.
    #[must_use]
    pub fn with_config(provider_id: String, config: CircuitBreakerConfig) -> Self {
        Self {
            provider_id,
            state: CircuitState::Closed,
            config,
            consecutive_failures: 0,
            consecutive_successes: 0,
            total_failures: 0,
            total_successes: 0,
            opened_at: None,
            last_state_change: SystemTime::now(),
            half_open_request_count: 0,
        }
    }

    /// Checks if a request is allowed through the circuit breaker.
    #[must_use]
    pub fn allow_request(&mut self) -> bool {
        match self.state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                // Check if timeout has elapsed
                if let Some(opened_at) = self.opened_at {
                    if let Ok(elapsed) = opened_at.elapsed() {
                        if elapsed >= self.config.timeout {
                            self.transition_to_half_open();
                            return true;
                        }
                    }
                }
                false
            }
            CircuitState::HalfOpen => {
                // Allow limited requests in half-open state
                self.half_open_request_count < self.config.half_open_requests
            }
        }
    }

    /// Records a successful request.
    pub fn record_success(&mut self) {
        self.total_successes += 1;
        self.consecutive_successes += 1;
        self.consecutive_failures = 0;

        match self.state {
            CircuitState::HalfOpen => {
                self.half_open_request_count += 1;
                if self.consecutive_successes >= self.config.success_threshold {
                    self.transition_to_closed();
                }
            }
            CircuitState::Open => {
                // Should not happen, but handle gracefully
                self.transition_to_half_open();
            }
            CircuitState::Closed => {
                // Normal operation
            }
        }
    }

    /// Records a failed request.
    pub fn record_failure(&mut self) {
        self.total_failures += 1;
        self.consecutive_failures += 1;
        self.consecutive_successes = 0;

        match self.state {
            CircuitState::Closed => {
                if self.consecutive_failures >= self.config.failure_threshold {
                    self.transition_to_open();
                }
            }
            CircuitState::HalfOpen => {
                // Any failure in half-open state reopens circuit
                self.transition_to_open();
            }
            CircuitState::Open => {
                // Already open
            }
        }
    }

    /// Transitions to closed state.
    fn transition_to_closed(&mut self) {
        self.state = CircuitState::Closed;
        self.consecutive_failures = 0;
        self.half_open_request_count = 0;
        self.opened_at = None;
        self.last_state_change = SystemTime::now();
    }

    /// Transitions to open state.
    fn transition_to_open(&mut self) {
        self.state = CircuitState::Open;
        self.opened_at = Some(SystemTime::now());
        self.last_state_change = SystemTime::now();
        self.half_open_request_count = 0;
    }

    /// Transitions to half-open state.
    fn transition_to_half_open(&mut self) {
        self.state = CircuitState::HalfOpen;
        self.consecutive_successes = 0;
        self.consecutive_failures = 0;
        self.half_open_request_count = 0;
        self.last_state_change = SystemTime::now();
    }

    /// Manually resets the circuit breaker.
    pub fn reset(&mut self) {
        self.transition_to_closed();
        self.consecutive_failures = 0;
        self.consecutive_successes = 0;
    }

    /// Gets the failure rate.
    #[must_use]
    pub fn failure_rate(&self) -> f64 {
        let total = self.total_failures + self.total_successes;
        if total == 0 {
            0.0
        } else {
            self.total_failures as f64 / total as f64
        }
    }

    /// Checks if the circuit is open.
    #[must_use]
    pub const fn is_open(&self) -> bool {
        matches!(self.state, CircuitState::Open)
    }

    /// Checks if the circuit is closed.
    #[must_use]
    pub const fn is_closed(&self) -> bool {
        matches!(self.state, CircuitState::Closed)
    }

    /// Checks if the circuit is half-open.
    #[must_use]
    pub const fn is_half_open(&self) -> bool {
        matches!(self.state, CircuitState::HalfOpen)
    }
}

/// Exponential backoff configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackoffConfig {
    /// Initial backoff duration.
    pub initial_interval: Duration,
    /// Maximum backoff duration.
    pub max_interval: Duration,
    /// Multiplier for each retry.
    pub multiplier: f64,
    /// Maximum number of retries.
    pub max_retries: u32,
}

impl Default for BackoffConfig {
    fn default() -> Self {
        Self {
            initial_interval: Duration::from_millis(100),
            max_interval: Duration::from_secs(60),
            multiplier: 2.0,
            max_retries: 5,
        }
    }
}

/// Exponential backoff state.
#[derive(Debug, Clone)]
pub struct BackoffState {
    /// Configuration.
    config: BackoffConfig,
    /// Current retry attempt.
    attempt: u32,
    /// Next backoff duration.
    next_interval: Duration,
}

impl BackoffState {
    /// Creates a new backoff state.
    #[must_use]
    pub fn new(config: BackoffConfig) -> Self {
        let next_interval = config.initial_interval;
        Self {
            config,
            attempt: 0,
            next_interval,
        }
    }

    /// Gets the next backoff duration.
    #[must_use]
    pub fn next_backoff(&mut self) -> Option<Duration> {
        if self.attempt >= self.config.max_retries {
            return None;
        }

        let current = self.next_interval;
        self.attempt += 1;

        // Calculate next interval with exponential backoff
        let next_ms = (current.as_millis() as f64 * self.config.multiplier) as u64;
        self.next_interval = Duration::from_millis(next_ms).min(self.config.max_interval);

        Some(current)
    }

    /// Resets the backoff state.
    pub fn reset(&mut self) {
        self.attempt = 0;
        self.next_interval = self.config.initial_interval;
    }

    /// Gets the current attempt number.
    #[must_use]
    pub const fn attempt(&self) -> u32 {
        self.attempt
    }
}

/// Fallback chain configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FallbackChain {
    /// Primary provider ID.
    pub primary: String,
    /// Fallback provider IDs in order of preference.
    pub fallbacks: Vec<String>,
    /// Current active provider index.
    current_index: usize,
}

impl FallbackChain {
    /// Creates a new fallback chain.
    #[must_use]
    pub fn new(primary: String, fallbacks: Vec<String>) -> Self {
        Self {
            primary,
            fallbacks,
            current_index: 0,
        }
    }

    /// Gets the current active provider.
    #[must_use]
    pub fn current_provider(&self) -> &str {
        if self.current_index == 0 {
            &self.primary
        } else {
            &self.fallbacks[self.current_index - 1]
        }
    }

    /// Moves to the next fallback provider.
    pub fn next_fallback(&mut self) -> Option<&str> {
        if self.current_index < self.fallbacks.len() {
            self.current_index += 1;
            Some(self.current_provider())
        } else {
            None
        }
    }

    /// Resets to the primary provider.
    pub fn reset(&mut self) {
        self.current_index = 0;
    }

    /// Checks if on primary provider.
    #[must_use]
    pub const fn is_primary(&self) -> bool {
        self.current_index == 0
    }

    /// Gets all providers in the chain.
    #[must_use]
    pub fn all_providers(&self) -> Vec<&str> {
        let mut providers = vec![self.primary.as_str()];
        providers.extend(self.fallbacks.iter().map(String::as_str));
        providers
    }
}

/// Graceful degradation configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DegradationConfig {
    /// Enable graceful degradation.
    pub enabled: bool,
    /// Minimum quality level (0-100).
    pub min_quality: u8,
    /// Reduce quality on error.
    pub reduce_quality_on_error: bool,
    /// Quality reduction step.
    pub quality_step: u8,
}

impl Default for DegradationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_quality: 30,
            reduce_quality_on_error: true,
            quality_step: 10,
        }
    }
}

/// Failover manager state.
struct FailoverState {
    /// Circuit breakers per provider.
    circuit_breakers: HashMap<String, CircuitBreaker>,
    /// Backoff states per provider.
    backoff_states: HashMap<String, BackoffState>,
    /// Fallback chains.
    fallback_chains: HashMap<String, FallbackChain>,
    /// Manual overrides (provider_id -> enabled).
    manual_overrides: HashMap<String, bool>,
}

/// Failover manager for CDN providers.
pub struct FailoverManager {
    /// Failure threshold.
    failure_threshold: u32,
    /// Circuit breaker timeout.
    circuit_timeout: Duration,
    /// Internal state.
    state: Arc<RwLock<FailoverState>>,
    /// Backoff configuration.
    backoff_config: BackoffConfig,
}

impl FailoverManager {
    /// Creates a new failover manager.
    #[must_use]
    pub fn new(failure_threshold: u32, circuit_timeout: Duration) -> Self {
        let state = FailoverState {
            circuit_breakers: HashMap::new(),
            backoff_states: HashMap::new(),
            fallback_chains: HashMap::new(),
            manual_overrides: HashMap::new(),
        };

        Self {
            failure_threshold,
            circuit_timeout,
            state: Arc::new(RwLock::new(state)),
            backoff_config: BackoffConfig::default(),
        }
    }

    /// Records a successful request.
    pub fn record_success(&self, provider_id: &str) {
        let mut state = self.state.write();

        // Update circuit breaker
        let breaker = state
            .circuit_breakers
            .entry(provider_id.to_string())
            .or_insert_with(|| {
                CircuitBreaker::with_config(
                    provider_id.to_string(),
                    CircuitBreakerConfig {
                        failure_threshold: self.failure_threshold,
                        timeout: self.circuit_timeout,
                        ..Default::default()
                    },
                )
            });
        breaker.record_success();

        // Reset backoff
        if let Some(backoff) = state.backoff_states.get_mut(provider_id) {
            backoff.reset();
        }
    }

    /// Records a failed request.
    pub fn record_failure(&self, provider_id: &str) {
        let mut state = self.state.write();

        // Update circuit breaker
        let breaker = state
            .circuit_breakers
            .entry(provider_id.to_string())
            .or_insert_with(|| {
                CircuitBreaker::with_config(
                    provider_id.to_string(),
                    CircuitBreakerConfig {
                        failure_threshold: self.failure_threshold,
                        timeout: self.circuit_timeout,
                        ..Default::default()
                    },
                )
            });
        breaker.record_failure();

        // Update backoff
        let backoff = state
            .backoff_states
            .entry(provider_id.to_string())
            .or_insert_with(|| BackoffState::new(self.backoff_config.clone()));
        let _next_backoff = backoff.next_backoff();
    }

    /// Checks if a provider is available (circuit not open).
    #[must_use]
    pub fn is_available(&self, provider_id: &str) -> bool {
        let state = self.state.read();

        // Check manual override
        if let Some(&enabled) = state.manual_overrides.get(provider_id) {
            if !enabled {
                return false;
            }
        }

        // Check circuit breaker
        if let Some(breaker) = state.circuit_breakers.get(provider_id) {
            !breaker.is_open()
        } else {
            true
        }
    }

    /// Checks if a circuit breaker is open.
    #[must_use]
    pub fn is_open(&self, provider_id: &str) -> bool {
        self.state
            .read()
            .circuit_breakers
            .get(provider_id)
            .map_or(false, CircuitBreaker::is_open)
    }

    /// Gets the circuit breaker state.
    #[must_use]
    pub fn get_circuit_state(&self, provider_id: &str) -> Option<CircuitState> {
        self.state
            .read()
            .circuit_breakers
            .get(provider_id)
            .map(|b| b.state)
    }

    /// Manually opens a circuit breaker.
    pub fn open_circuit(&self, provider_id: &str) {
        let mut state = self.state.write();
        state
            .manual_overrides
            .insert(provider_id.to_string(), false);
    }

    /// Manually closes a circuit breaker.
    pub fn close_circuit(&self, provider_id: &str) {
        let mut state = self.state.write();
        state.manual_overrides.insert(provider_id.to_string(), true);
        if let Some(breaker) = state.circuit_breakers.get_mut(provider_id) {
            breaker.reset();
        }
    }

    /// Resets all circuit breakers.
    pub fn reset_all(&self) {
        let mut state = self.state.write();
        for breaker in state.circuit_breakers.values_mut() {
            breaker.reset();
        }
        state.manual_overrides.clear();
    }

    /// Gets the circuit breaker for a provider.
    #[must_use]
    pub fn get_circuit_breaker(&self, provider_id: &str) -> Option<CircuitBreaker> {
        self.state.read().circuit_breakers.get(provider_id).cloned()
    }

    /// Adds a fallback chain.
    pub fn add_fallback_chain(&self, primary: String, fallbacks: Vec<String>) {
        let mut state = self.state.write();
        let chain = FallbackChain::new(primary.clone(), fallbacks);
        state.fallback_chains.insert(primary, chain);
    }

    /// Gets the next fallback provider.
    pub fn get_next_fallback(&self, primary_id: &str) -> Option<String> {
        let mut state = self.state.write();
        state
            .fallback_chains
            .get_mut(primary_id)
            .and_then(|chain| chain.next_fallback().map(String::from))
    }

    /// Resets fallback chain to primary.
    pub fn reset_fallback_chain(&self, primary_id: &str) {
        let mut state = self.state.write();
        if let Some(chain) = state.fallback_chains.get_mut(primary_id) {
            chain.reset();
        }
    }

    /// Gets all circuit breaker states.
    #[must_use]
    pub fn get_all_states(&self) -> HashMap<String, CircuitState> {
        self.state
            .read()
            .circuit_breakers
            .iter()
            .map(|(id, breaker)| (id.clone(), breaker.state))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_state() {
        assert!(CircuitState::Closed.allows_requests());
        assert!(CircuitState::HalfOpen.allows_requests());
        assert!(!CircuitState::Open.allows_requests());
    }

    #[test]
    fn test_circuit_breaker_creation() {
        let breaker = CircuitBreaker::new("provider-1".to_string());
        assert_eq!(breaker.provider_id, "provider-1");
        assert_eq!(breaker.state, CircuitState::Closed);
        assert!(breaker.is_closed());
    }

    #[test]
    fn test_circuit_breaker_failure() {
        let mut breaker = CircuitBreaker::new("provider-1".to_string());

        // Record failures up to threshold
        for _ in 0..5 {
            breaker.record_failure();
        }

        assert!(breaker.is_open());
        assert!(!breaker.allow_request());
    }

    #[test]
    fn test_circuit_breaker_success() {
        let mut breaker = CircuitBreaker::new("provider-1".to_string());

        breaker.record_success();
        assert_eq!(breaker.consecutive_successes, 1);
        assert_eq!(breaker.total_successes, 1);
        assert!(breaker.is_closed());
    }

    #[test]
    fn test_circuit_breaker_reset() {
        let mut breaker = CircuitBreaker::new("provider-1".to_string());

        for _ in 0..5 {
            breaker.record_failure();
        }
        assert!(breaker.is_open());

        breaker.reset();
        assert!(breaker.is_closed());
        assert_eq!(breaker.consecutive_failures, 0);
    }

    #[test]
    fn test_backoff_state() {
        let mut backoff = BackoffState::new(BackoffConfig::default());

        let first = backoff.next_backoff();
        assert!(first.is_some());
        assert_eq!(backoff.attempt(), 1);

        let second = backoff.next_backoff();
        assert!(second.is_some());
        assert!(second.expect("should succeed in test") > first.expect("should succeed in test"));
    }

    #[test]
    fn test_backoff_reset() {
        let mut backoff = BackoffState::new(BackoffConfig::default());

        let _first = backoff.next_backoff();
        assert_eq!(backoff.attempt(), 1);

        backoff.reset();
        assert_eq!(backoff.attempt(), 0);
    }

    #[test]
    fn test_fallback_chain() {
        let mut chain = FallbackChain::new(
            "primary".to_string(),
            vec!["fallback1".to_string(), "fallback2".to_string()],
        );

        assert_eq!(chain.current_provider(), "primary");
        assert!(chain.is_primary());

        chain.next_fallback();
        assert_eq!(chain.current_provider(), "fallback1");
        assert!(!chain.is_primary());

        chain.next_fallback();
        assert_eq!(chain.current_provider(), "fallback2");

        chain.reset();
        assert_eq!(chain.current_provider(), "primary");
        assert!(chain.is_primary());
    }

    #[test]
    fn test_failover_manager() {
        let manager = FailoverManager::new(3, Duration::from_secs(60));

        manager.record_success("provider-1");
        assert!(manager.is_available("provider-1"));

        for _ in 0..3 {
            manager.record_failure("provider-1");
        }

        assert!(!manager.is_available("provider-1"));
        assert!(manager.is_open("provider-1"));
    }

    #[test]
    fn test_failover_manager_manual_override() {
        let manager = FailoverManager::new(3, Duration::from_secs(60));

        manager.open_circuit("provider-1");
        assert!(!manager.is_available("provider-1"));

        manager.close_circuit("provider-1");
        assert!(manager.is_available("provider-1"));
    }

    #[test]
    fn test_failover_manager_reset() {
        let manager = FailoverManager::new(3, Duration::from_secs(60));

        for _ in 0..3 {
            manager.record_failure("provider-1");
        }
        assert!(!manager.is_available("provider-1"));

        manager.reset_all();
        assert!(manager.is_available("provider-1"));
    }

    #[test]
    fn test_cdn_failover_primary_timeout_switches_to_secondary() {
        let manager = FailoverManager::new(3, Duration::from_secs(60));

        manager.add_fallback_chain("cdn-primary".to_string(), vec!["cdn-secondary".to_string()]);

        // Both providers initially available (no circuit breaker state yet)
        assert!(
            manager.is_available("cdn-primary"),
            "primary should be initially available"
        );
        assert!(
            manager.is_available("cdn-secondary"),
            "secondary should be initially available"
        );

        // Simulate 3 failures on primary — reaches failure_threshold
        manager.record_failure("cdn-primary");
        manager.record_failure("cdn-primary");
        manager.record_failure("cdn-primary");

        // Primary circuit should now be open
        assert!(
            !manager.is_available("cdn-primary"),
            "primary should be unavailable after threshold failures"
        );
        assert!(
            manager.is_open("cdn-primary"),
            "primary circuit should be open"
        );

        // Next fallback from primary chain should be the secondary
        let next = manager.get_next_fallback("cdn-primary");
        assert!(next.is_some(), "should have a fallback available");
        assert_eq!(
            next.as_deref(),
            Some("cdn-secondary"),
            "fallback should be cdn-secondary"
        );

        // Secondary was never touched — still available
        assert!(
            manager.is_available("cdn-secondary"),
            "secondary should still be available"
        );
    }

    #[test]
    fn test_cdn_failover_secondary_chain_exhaustion() {
        let manager = FailoverManager::new(2, Duration::from_secs(60));

        manager.add_fallback_chain("p".to_string(), vec!["s1".to_string(), "s2".to_string()]);

        // Record 2 failures on primary — opens circuit on "p"
        manager.record_failure("p");
        manager.record_failure("p");
        assert!(
            !manager.is_available("p"),
            "primary should be unavailable after 2 failures"
        );

        // First fallback from "p" chain → s1
        let next1 = manager.get_next_fallback("p");
        assert_eq!(next1.as_deref(), Some("s1"), "first fallback should be s1");

        // Record 2 failures on s1 — opens circuit on "s1"
        manager.record_failure("s1");
        manager.record_failure("s1");
        assert!(
            !manager.is_available("s1"),
            "s1 should be unavailable after 2 failures"
        );

        // Second fallback from "p" chain → s2
        let next2 = manager.get_next_fallback("p");
        assert_eq!(next2.as_deref(), Some("s2"), "second fallback should be s2");

        // Chain exhausted — no more fallbacks
        let next3 = manager.get_next_fallback("p");
        assert!(next3.is_none(), "chain should be exhausted after s2");

        // s2 was never failed — still available
        assert!(
            manager.is_available("s2"),
            "s2 should still be available (never failed)"
        );
    }
}
