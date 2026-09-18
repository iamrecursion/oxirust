//! Circuit breaker pattern for error handling

use crate::error::{CoreResult as Result, ErrorContext};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// State of the circuit breaker
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Circuit is functioning normally
    Closed,
    /// Circuit is temporarily disabled
    Open,
    /// Circuit is testing if it should reopen
    HalfOpen,
}

/// Status of circuit breaker
#[derive(Debug, Clone)]
pub struct CircuitBreakerStatus {
    pub state: CircuitState,
    pub failure_count: usize,
    pub success_count: usize,
    pub last_state_change: Instant,
}

/// Fallback strategy for failed operations
#[derive(Debug, Clone)]
pub enum FallbackStrategy {
    /// Return default value
    Default,
    /// Use cached value
    Cache,
    /// Execute alternative function
    Alternative,
    /// Fail fast
    FailFast,
}

/// Retry policy configuration
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    pub max_retries: usize,
    pub initial_delay: Duration,
    pub max_delay: Duration,
    pub exponential_base: f64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            exponential_base: 2.0,
        }
    }
}

/// Circuit breaker for fault tolerance
pub struct CircuitBreaker {
    /// Current state
    state: Arc<Mutex<CircuitState>>,
    /// Failure count
    failure_count: AtomicUsize,
    /// Success count in half-open state
    success_count: AtomicUsize,
    /// Timestamp of last state change
    last_state_change: Arc<Mutex<Instant>>,
    /// Configuration
    config: CircuitBreakerConfig,
}

/// Configuration for circuit breaker
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Threshold for opening circuit
    pub failure_threshold: usize,
    /// Threshold for closing circuit from half-open
    pub success_threshold: usize,
    /// Timeout before attempting recovery
    pub timeout: Duration,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            success_threshold: 3,
            timeout: Duration::from_secs(30),
        }
    }
}

impl CircuitBreaker {
    /// Create new circuit breaker
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            state: Arc::new(Mutex::new(CircuitState::Closed)),
            failure_count: AtomicUsize::new(0),
            success_count: AtomicUsize::new(0),
            last_state_change: Arc::new(Mutex::new(Instant::now())),
            config,
        }
    }

    /// Get current state
    pub fn state(&self) -> CircuitState {
        *self.state.lock().expect("Operation failed")
    }

    /// Record success
    pub fn record_success(&self) {
        let mut state = self.state.lock().expect("Operation failed");
        match *state {
            CircuitState::HalfOpen => {
                let count = self.success_count.fetch_add(1, Ordering::SeqCst) + 1;
                if count >= self.config.success_threshold {
                    *state = CircuitState::Closed;
                    self.failure_count.store(0, Ordering::SeqCst);
                    self.success_count.store(0, Ordering::SeqCst);
                    *self.last_state_change.lock().expect("Operation failed") = Instant::now();
                }
            }
            CircuitState::Closed => {
                self.failure_count.store(0, Ordering::SeqCst);
            }
            CircuitState::Open => {}
        }
    }

    /// Record failure
    pub fn record_failure(&self) {
        let mut state = self.state.lock().expect("Operation failed");
        match *state {
            CircuitState::Closed => {
                let count = self.failure_count.fetch_add(1, Ordering::SeqCst) + 1;
                if count >= self.config.failure_threshold {
                    *state = CircuitState::Open;
                    *self.last_state_change.lock().expect("Operation failed") = Instant::now();
                }
            }
            CircuitState::HalfOpen => {
                *state = CircuitState::Open;
                self.failure_count.store(0, Ordering::SeqCst);
                self.success_count.store(0, Ordering::SeqCst);
                *self.last_state_change.lock().expect("Operation failed") = Instant::now();
            }
            CircuitState::Open => {}
        }
    }

    /// Check if should transition from open to half-open
    pub fn check_state(&self) {
        let mut state = self.state.lock().expect("Operation failed");
        if *state == CircuitState::Open {
            let elapsed = self
                .last_state_change
                .lock()
                .expect("Operation failed")
                .elapsed();
            if elapsed >= self.config.timeout {
                *state = CircuitState::HalfOpen;
                self.success_count.store(0, Ordering::SeqCst);
                *self.last_state_change.lock().expect("Operation failed") = Instant::now();
            }
        }
    }

    /// Check if circuit allows request
    pub fn is_allowed(&self) -> bool {
        self.check_state();
        let state = self.state();
        state == CircuitState::Closed || state == CircuitState::HalfOpen
    }

    /// Get current status
    pub fn status(&self) -> CircuitBreakerStatus {
        CircuitBreakerStatus {
            state: self.state(),
            failure_count: self.failure_count.load(Ordering::SeqCst),
            success_count: self.success_count.load(Ordering::SeqCst),
            last_state_change: *self.last_state_change.lock().expect("Operation failed"),
        }
    }

    /// Execute function with circuit breaker protection
    pub fn execute<F, T>(&self, f: F) -> Result<T>
    where
        F: FnOnce() -> Result<T>,
    {
        use crate::error::CoreError;

        // Check if we can execute
        if !self.is_allowed() {
            return Err(CoreError::ValueError(ErrorContext::new(
                "Circuit breaker is open",
            )));
        }

        // Execute the function
        match f() {
            Ok(result) => {
                self.record_success();
                Ok(result)
            }
            Err(e) => {
                self.record_failure();
                Err(e)
            }
        }
    }
}

/// Executor with retry capabilities
pub struct RetryExecutor {
    policy: RetryPolicy,
}

impl RetryExecutor {
    /// Create new retry executor
    pub fn new(policy: RetryPolicy) -> Self {
        Self { policy }
    }

    /// Execute function with retries
    pub fn execute<F, T>(&self, mut f: F) -> Result<T>
    where
        F: FnMut() -> Result<T>,
    {
        let mut last_error = None;
        for _ in 0..self.policy.max_retries {
            match f() {
                Ok(result) => return Ok(result),
                Err(e) => last_error = Some(e),
            }
        }
        Err(last_error.expect("Operation failed"))
    }
}

/// Resilient executor with circuit breaker and fallback
pub struct ResilientExecutor {
    circuit_breaker: CircuitBreaker,
    retry_executor: RetryExecutor,
    fallback_strategy: FallbackStrategy,
}

impl ResilientExecutor {
    /// Create new resilient executor
    pub fn new(
        circuit_breaker: CircuitBreaker,
        retry_executor: RetryExecutor,
        fallback_strategy: FallbackStrategy,
    ) -> Self {
        Self {
            circuit_breaker,
            retry_executor,
            fallback_strategy,
        }
    }

    /// Execute function with resilience
    pub fn execute<F, T>(&self, f: F) -> Result<T>
    where
        F: FnMut() -> Result<T>,
    {
        if !self.circuit_breaker.is_allowed() {
            return Err(crate::error::CoreError::ValueError(ErrorContext::new(
                "Circuit breaker is open",
            )));
        }

        match self.retry_executor.execute(f) {
            Ok(result) => {
                self.circuit_breaker.record_success();
                Ok(result)
            }
            Err(e) => {
                self.circuit_breaker.record_failure();
                Err(e)
            }
        }
    }
}

// Global registry of circuit breakers
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::RwLock;

static CIRCUIT_BREAKERS: Lazy<RwLock<HashMap<String, Arc<CircuitBreaker>>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

/// Get a circuit breaker by name
pub fn get_circuitbreaker(name: &str) -> Option<Arc<CircuitBreaker>> {
    CIRCUIT_BREAKERS
        .read()
        .expect("Operation failed")
        .get(name)
        .cloned()
}

/// List all circuit breakers
pub fn list_circuitbreakers() -> Vec<String> {
    CIRCUIT_BREAKERS
        .read()
        .expect("Operation failed")
        .keys()
        .cloned()
        .collect()
}
