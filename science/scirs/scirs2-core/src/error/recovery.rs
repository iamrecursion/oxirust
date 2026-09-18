//! Advanced error recovery and resilience mechanisms for ``SciRS2``
//!
//! This module provides sophisticated error handling patterns including:
//! - Automatic retry mechanisms with backoff strategies
//! - Circuit breaker patterns for fault tolerance
//! - Error recovery hints and suggestions
//! - Graceful degradation strategies
//! - Error aggregation and reporting

use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::error::{CoreError, CoreResult, ErrorContext};

/// Recovery strategy for handling errors
#[derive(Debug, Clone)]
pub enum RecoveryStrategy {
    /// Fail immediately without retry
    FailFast,
    /// Retry with exponential backoff
    ExponentialBackoff {
        max_attempts: usize,
        initialdelay: Duration,
        maxdelay: Duration,
        multiplier: f64,
    },
    /// Retry with linear backoff
    LinearBackoff {
        max_attempts: usize,
        delay: Duration,
    },
    /// Retry with custom backoff strategy
    CustomBackoff {
        max_attempts: usize,
        delays: Vec<Duration>,
    },
    /// Circuit breaker pattern
    CircuitBreaker {
        failure_threshold: usize,
        timeout: Duration,
        recoverytimeout: Duration,
    },
    /// Fallback to alternative implementation
    Fallback,
    /// Graceful degradation with reduced functionality
    GracefulDegradation,
}

impl Default for RecoveryStrategy {
    fn default() -> Self {
        Self::ExponentialBackoff {
            max_attempts: 3,
            initialdelay: Duration::from_millis(100),
            maxdelay: Duration::from_secs(5),
            multiplier: 2.0,
        }
    }
}

/// Recovery hint providing actionable suggestions for error resolution
#[derive(Debug, Clone)]
pub struct RecoveryHint {
    /// Short description of the suggested action
    pub action: String,
    /// Detailed explanation of why this action might help
    pub explanation: String,
    /// Code examples or specific steps to take
    pub examples: Vec<String>,
    /// Confidence level that this hint will resolve the issue (0.0 to 1.0)
    pub confidence: f64,
}

impl RecoveryHint {
    /// Create a new recovery hint
    pub fn new<S: Into<String>>(action: S, explanation: S, confidence: f64) -> Self {
        Self {
            action: action.into(),
            explanation: explanation.into(),
            examples: Vec::new(),
            confidence: confidence.clamp(0.0, 1.0),
        }
    }

    /// Add an example to the recovery hint
    pub fn with_example<S: Into<String>>(mut self, example: S) -> Self {
        self.examples.push(example.into());
        self
    }

    /// Add multiple examples to the recovery hint
    pub fn with_examples<I, S>(mut self, examples: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.examples.extend(examples.into_iter().map(|s| s.into()));
        self
    }
}

impl fmt::Display for RecoveryHint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "💡 {} (confidence: {:.1}%)",
            self.action,
            self.confidence * 100.0
        )?;
        writeln!(f, "   {}", self.explanation)?;

        if !self.examples.is_empty() {
            writeln!(f, "   Examples:")?;
            for (i, example) in self.examples.iter().enumerate() {
                writeln!(f, "   {}. {}", i + 1, example)?;
            }
        }

        Ok(())
    }
}

/// Enhanced error with recovery capabilities
#[derive(Debug, Clone)]
pub struct RecoverableError {
    /// The original error
    pub error: CoreError,
    /// Suggested recovery strategy
    pub strategy: RecoveryStrategy,
    /// Recovery hints for the user
    pub hints: Vec<RecoveryHint>,
    /// Whether this error is retryable
    pub retryable: bool,
    /// Error severity level
    pub severity: ErrorSeverity,
    /// Additional metadata about the error
    pub metadata: std::collections::HashMap<String, String>,
}

/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorSeverity {
    /// Informational - operation succeeded with warnings
    Info,
    /// Warning - operation succeeded but with issues
    Warning,
    /// Error - operation failed but system is stable
    Error,
    /// Critical - operation failed and system stability affected
    Critical,
    /// Fatal - operation failed and system cannot continue
    Fatal,
}

impl fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Info => write!(f, "INFO"),
            Self::Warning => write!(f, "WARN"),
            Self::Error => write!(f, "ERROR"),
            Self::Critical => write!(f, "CRITICAL"),
            Self::Fatal => write!(f, "FATAL"),
        }
    }
}

impl RecoverableError {
    /// Create a new recoverable error
    pub fn error(error: CoreError) -> Self {
        let (strategy, hints, retryable, severity) = Self::analyzeerror(&error);

        Self {
            error,
            strategy,
            hints,
            retryable,
            severity,
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Create a recoverable error with custom strategy
    pub fn with_strategy(mut self, strategy: RecoveryStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Add a recovery hint
    pub fn with_hint(mut self, hint: RecoveryHint) -> Self {
        self.hints.push(hint);
        self
    }

    /// Add metadata to the error
    pub fn with_metadata<K, V>(mut self, key: K, value: V) -> Self
    where
        K: Into<String>,
        V: Into<String>,
    {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Set error severity
    pub fn with_severity(mut self, severity: ErrorSeverity) -> Self {
        self.severity = severity;
        self
    }

    /// Analyze an error and suggest recovery strategies
    fn analyzeerror(
        error: &CoreError,
    ) -> (RecoveryStrategy, Vec<RecoveryHint>, bool, ErrorSeverity) {
        match error {
            CoreError::DomainError(_) => (
                RecoveryStrategy::FailFast,
                vec![
                    RecoveryHint::new(
                        "Check input values",
                        "Domain errors usually indicate input values are outside the valid range",
                        0.9,
                    ).with_examples(vec![
                        "Ensure all inputs are finite (not NaN or infinity)",
                        "Check that array indices are within bounds",
                        "Verify parameter ranges match function requirements",
                    ]),
                ],
                false,
                ErrorSeverity::Error,
            ),

            CoreError::ConvergenceError(_) => (
                RecoveryStrategy::ExponentialBackoff {
                    max_attempts: 5,
                    initialdelay: Duration::from_millis(500),
                    maxdelay: Duration::from_secs(10),
                    multiplier: 1.5,
                },
                vec![
                    RecoveryHint::new(
                        "Adjust convergence parameters",
                        "Convergence failures often indicate numerical instability or poor initial conditions",
                        0.8,
                    ).with_examples(vec![
                        "Increase maximum iteration count",
                        "Decrease convergence tolerance",
                        "Try different initial values or starting points",
                        "Use a more robust algorithm variant",
                    ]),
                    RecoveryHint::new(
                        "Check problem conditioning",
                        "Poor problem conditioning can prevent convergence",
                        0.7,
                    ).with_examples(vec![
                        "Scale input data to similar ranges",
                        "Add regularization to improve conditioning",
                        "Use preconditioning techniques",
                    ]),
                ],
                true,
                ErrorSeverity::Warning,
            ),

            CoreError::MemoryError(_) => (
                RecoveryStrategy::GracefulDegradation,
                vec![
                    RecoveryHint::new(
                        "Reduce memory usage",
                        "Memory errors indicate insufficient resources for the requested operation",
                        0.9,
                    ).with_examples(vec![
                        "Process data in smaller chunks",
                        "Use out-of-core algorithms",
                        "Reduce precision (e.g., f32 instead of f64)",
                        "Free unused variables before large operations",
                    ]),
                    RecoveryHint::new(
                        "Use streaming algorithms",
                        "Stream processing can handle larger datasets with limited memory",
                        0.8,
                    ).with_examples(vec![
                        "Enable chunked processing with scirs2_core::memory_efficient",
                        "Use iterative algorithms instead of direct methods",
                        "Consider using memory-mapped files for large arrays",
                    ]),
                ],
                false,
                ErrorSeverity::Critical,
            ),

            CoreError::TimeoutError(_) => (
                RecoveryStrategy::LinearBackoff {
                    max_attempts: 3,
                    delay: Duration::from_secs(1),
                },
                vec![
                    RecoveryHint::new(
                        "Increase timeout duration",
                        "Timeout errors may indicate the operation needs more time",
                        0.7,
                    ).with_examples(vec![
                        "Set a larger timeout value",
                        "Use asynchronous operations with progress tracking",
                        "Break large operations into smaller parts",
                    ]),
                ],
                true,
                ErrorSeverity::Warning,
            ),

            CoreError::ShapeError(_) | CoreError::DimensionError(_) => (
                RecoveryStrategy::FailFast,
                vec![
                    RecoveryHint::new(
                        "Verify array dimensions",
                        "Shape/dimension errors indicate incompatible array sizes",
                        0.95,
                    ).with_examples(vec![
                        "Check input array shapes with .shape() method",
                        "Ensure matrix dimensions are compatible for operations",
                        "Use broadcasting or reshaping to make arrays compatible",
                        "Transpose arrays if needed (e.g., .t() for transpose)",
                    ]),
                ],
                false,
                ErrorSeverity::Error,
            ),

            CoreError::NotImplementedError(_) => (
                RecoveryStrategy::Fallback,
                vec![
                    RecoveryHint::new(
                        "Use alternative implementation",
                        "This feature is not yet implemented in `SciRS2`",
                        0.8,
                    ).with_examples(vec![
                        "Check if a similar function exists in another module",
                        "Use a more basic implementation as a workaround",
                        "Consider using external libraries for this functionality",
                    ]),
                ],
                false,
                ErrorSeverity::Info,
            ),

            CoreError::IoError(_) => (
                RecoveryStrategy::ExponentialBackoff {
                    max_attempts: 3,
                    initialdelay: Duration::from_millis(200),
                    maxdelay: Duration::from_secs(2),
                    multiplier: 2.0,
                },
                vec![
                    RecoveryHint::new(
                        "Check file permissions and paths",
                        "I/O errors often indicate file system issues",
                        0.8,
                    ).with_examples(vec![
                        "Verify file paths exist and are accessible",
                        "Check read/write permissions",
                        "Ensure sufficient disk space is available",
                        "Try absolute paths instead of relative paths",
                    ]),
                ],
                true,
                ErrorSeverity::Warning,
            ), _ => (
                RecoveryStrategy::default(),
                vec![
                    RecoveryHint::new(
                        "Check error details",
                        "Review the specific error message for more information",
                        0.5,
                    ),
                ],
                true,
                ErrorSeverity::Error,
            ),
        }
    }

    /// Get a user-friendly error report with recovery suggestions
    pub fn recovery_report(&self) -> String {
        let mut report = String::new();

        report.push_str(&format!("🚨 {} Error: {}\n\n", self.severity, self.error));

        if self.retryable {
            report.push_str("✅ This error may be retryable\n");
        } else {
            report.push_str("❌ This error is not retryable\n");
        }

        report.push_str(&format!("🔧 Suggested strategy: {:?}\n\n", self.strategy));

        if !self.hints.is_empty() {
            report.push_str("🔍 Recovery suggestions:\n");
            for (i, hint) in self.hints.iter().enumerate() {
                report.push_str(&format!("{}. {}\n", i + 1, hint));
            }
        }

        if !self.metadata.is_empty() {
            report.push_str("\n📋 Additional information:\n");
            for (key, value) in &self.metadata {
                report.push_str(&format!("   {key}: {value}\n"));
            }
        }

        report
    }
}

impl fmt::Display for RecoverableError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.recovery_report())
    }
}

impl std::error::Error for RecoverableError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

/// Circuit breaker for handling repeated failures
#[derive(Debug)]
pub struct CircuitBreaker {
    failure_threshold: usize,
    #[allow(dead_code)]
    timeout: Duration,
    recoverytimeout: Duration,
    state: Arc<Mutex<CircuitBreakerState>>,
}

#[derive(Debug)]
struct CircuitBreakerState {
    failure_count: usize,
    last_failure_time: Option<Instant>,
    state: CircuitState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

impl CircuitBreaker {
    /// Create a new circuit breaker
    pub fn timeout(failure_threshold: usize, timeout: Duration, recoverytimeout: Duration) -> Self {
        Self {
            failure_threshold,
            timeout,
            recoverytimeout,
            state: Arc::new(Mutex::new(CircuitBreakerState {
                failure_count: 0,
                last_failure_time: None,
                state: CircuitState::Closed,
            })),
        }
    }

    /// Create a new circuit breaker (alias for timeout method)
    pub fn new(failure_threshold: usize, timeout: Duration, recoverytimeout: Duration) -> Self {
        Self::timeout(failure_threshold, timeout, recoverytimeout)
    }

    /// Execute a function with circuit breaker protection
    pub fn execute<F, T>(&self, f: F) -> CoreResult<T>
    where
        F: FnOnce() -> CoreResult<T>,
    {
        // Check if circuit should allow execution
        if !self.should_allow_execution() {
            return Err(CoreError::ComputationError(ErrorContext::new(
                "Circuit breaker is open - too many recent failures",
            )));
        }

        // Execute the function
        match f() {
            Ok(result) => {
                self.on_success();
                Ok(result)
            }
            Err(err) => {
                self.on_failure();
                Err(err)
            }
        }
    }

    fn should_allow_execution(&self) -> bool {
        let mut state = self.state.lock().expect("Operation failed");

        match state.state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                if let Some(last_failure) = state.last_failure_time {
                    if last_failure.elapsed() >= self.recoverytimeout {
                        state.state = CircuitState::HalfOpen;
                        true
                    } else {
                        false
                    }
                } else {
                    true
                }
            }
            CircuitState::HalfOpen => true,
        }
    }

    fn on_success(&self) {
        let mut state = self.state.lock().expect("Operation failed");
        state.failure_count = 0;
        state.state = CircuitState::Closed;
    }

    fn on_failure(&self) {
        let mut state = self.state.lock().expect("Operation failed");
        state.failure_count += 1;
        state.last_failure_time = Some(Instant::now());

        if state.failure_count >= self.failure_threshold {
            state.state = CircuitState::Open;
        }
    }

    /// Get current circuit breaker status
    pub fn status(&self) -> CircuitBreakerStatus {
        let state = self.state.lock().expect("Operation failed");
        CircuitBreakerStatus {
            state: state.state,
            failure_count: state.failure_count,
            failure_threshold: self.failure_threshold,
        }
    }
}

/// Circuit breaker status information
#[derive(Debug, Clone)]
pub struct CircuitBreakerStatus {
    pub state: CircuitState,
    pub failure_count: usize,
    pub failure_threshold: usize,
}

impl fmt::Display for CircuitBreakerStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Circuit breaker: {state:?} ({failure_count}/{failure_threshold} failures)",
            state = self.state,
            failure_count = self.failure_count,
            failure_threshold = self.failure_threshold
        )
    }
}

/// Retry executor with configurable backoff strategies
#[derive(Debug)]
pub struct RetryExecutor {
    strategy: RecoveryStrategy,
}

impl RetryExecutor {
    /// Create a new retry executor with the given strategy
    pub fn strategy(strategy: RecoveryStrategy) -> Self {
        Self { strategy }
    }

    /// Create a new retry executor (alias for strategy method)
    pub fn new(strategy: RecoveryStrategy) -> Self {
        Self::strategy(strategy)
    }

    /// Execute a function with retry logic
    pub fn execute<F, T>(&self, mut f: F) -> CoreResult<T>
    where
        F: FnMut() -> CoreResult<T>,
    {
        match &self.strategy {
            RecoveryStrategy::FailFast => f(),

            RecoveryStrategy::ExponentialBackoff {
                max_attempts,
                initialdelay,
                maxdelay,
                multiplier,
            } => {
                let mut delay = *initialdelay;
                let mut lasterror = None;

                for attempt in 0..*max_attempts {
                    match f() {
                        Ok(result) => return Ok(result),
                        Err(err) => {
                            lasterror = Some(err);

                            if attempt < max_attempts - 1 {
                                std::thread::sleep(delay);
                                delay = std::cmp::min(
                                    Duration::from_nanos(
                                        (delay.as_nanos() as f64 * multiplier) as u64,
                                    ),
                                    *maxdelay,
                                );
                            }
                        }
                    }
                }

                Err(lasterror.expect("Operation failed"))
            }

            RecoveryStrategy::LinearBackoff {
                max_attempts,
                delay,
            } => {
                let mut lasterror = None;

                for attempt in 0..*max_attempts {
                    match f() {
                        Ok(result) => return Ok(result),
                        Err(err) => {
                            lasterror = Some(err);

                            if attempt < max_attempts - 1 {
                                std::thread::sleep(*delay);
                            }
                        }
                    }
                }

                Err(lasterror.expect("Operation failed"))
            }

            RecoveryStrategy::CustomBackoff {
                max_attempts,
                delays,
            } => {
                let mut lasterror = None;

                for attempt in 0..*max_attempts {
                    match f() {
                        Ok(result) => return Ok(result),
                        Err(err) => {
                            lasterror = Some(err);

                            if attempt < max_attempts - 1 {
                                if let Some(&delay) = delays.get(attempt) {
                                    std::thread::sleep(delay);
                                }
                            }
                        }
                    }
                }

                Err(lasterror.expect("Operation failed"))
            }

            _ => f(), // Other strategies not applicable for retry
        }
    }
}

/// Error aggregator for collecting multiple errors
#[derive(Debug, Default)]
pub struct ErrorAggregator {
    errors: Vec<RecoverableError>,
    maxerrors: Option<usize>,
}

impl ErrorAggregator {
    /// Create a new error aggregator
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new error aggregator with maximum error limit
    pub fn errors(maxerrors: usize) -> Self {
        Self {
            errors: Vec::new(),
            maxerrors: Some(maxerrors),
        }
    }

    /// Add an error to the aggregator
    pub fn adderror(&mut self, error: RecoverableError) {
        if let Some(max) = self.maxerrors {
            if self.errors.len() >= max {
                return; // Ignore additional errors
            }
        }

        self.errors.push(error);
    }

    /// Add a simple error to the aggregator
    pub fn add_simpleerror(&mut self, error: CoreError) {
        self.adderror(RecoverableError::error(error));
    }

    /// Check if there are any errors
    pub fn haserrors(&self) -> bool {
        !self.errors.is_empty()
    }

    /// Get the number of errors
    pub fn error_count(&self) -> usize {
        self.errors.len()
    }

    /// Get all errors
    pub fn errors_2(&self) -> &[RecoverableError] {
        &self.errors
    }

    /// Get the most severe error
    pub fn most_severeerror(&self) -> Option<&RecoverableError> {
        self.errors.iter().max_by_key(|err| err.severity)
    }

    /// Get a summary of all errors
    pub fn summary(&self) -> String {
        if self.errors.is_empty() {
            return "No errors".to_string();
        }

        let mut summary = format!("Collected {count} error(s):\n", count = self.errors.len());

        for (i, error) in self.errors.iter().enumerate() {
            summary.push_str(&format!(
                "{num}. [{severity}] {error}\n",
                num = i + 1,
                severity = error.severity,
                error = error.error
            ));
        }

        if let Some(most_severe) = self.most_severeerror() {
            summary.push_str(&format!(
                "\nMost severe: {error}\n",
                error = most_severe.error
            ));
        }

        summary
    }

    /// Convert to a single error if there are any errors
    pub fn into_result<T>(self, successvalue: T) -> Result<T, Box<RecoverableError>> {
        if let Some(most_severe) = self.errors.into_iter().max_by_key(|err| err.severity) {
            Err(Box::new(most_severe))
        } else {
            Ok(successvalue)
        }
    }

    /// Clear all collected errors
    pub fn clear(&mut self) {
        self.errors.clear();
    }
}

impl fmt::Display for ErrorAggregator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.summary())
    }
}

/// Convenience function to create common recovery hints
pub mod hints {
    use super::RecoveryHint;

    /// Create a hint for checking input values
    pub fn check_inputs() -> RecoveryHint {
        RecoveryHint::new(
            "Verify input parameters",
            "Invalid inputs are a common source of errors",
            0.8,
        )
        .with_examples(vec![
            "Check for NaN or infinite values",
            "Ensure arrays have the correct shape",
            "Verify parameter ranges",
        ])
    }

    /// Create a hint for numerical stability issues
    pub fn numerical_stability() -> RecoveryHint {
        RecoveryHint::new(
            "Improve numerical stability",
            "Numerical instability can cause computation failures",
            0.7,
        )
        .with_examples(vec![
            "Scale input data to similar ranges",
            "Use higher precision arithmetic",
            "Add regularization or conditioning",
        ])
    }

    /// Create a hint for memory optimization
    pub fn memory_optimization() -> RecoveryHint {
        RecoveryHint::new(
            "Optimize memory usage",
            "Large datasets may require memory-efficient approaches",
            0.9,
        )
        .with_examples(vec![
            "Process data in chunks",
            "Use streaming algorithms",
            "Reduce precision if appropriate",
        ])
    }

    /// Create a hint for algorithm selection
    pub fn algorithm_selection() -> RecoveryHint {
        RecoveryHint::new(
            "Try alternative algorithms",
            "Different algorithms may work better for your specific problem",
            0.6,
        )
        .with_examples(vec![
            "Use iterative instead of direct methods",
            "Try a more robust variant",
            "Consider approximate methods",
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ErrorContext;

    #[test]
    fn test_recovery_hint_creation() {
        let hint = RecoveryHint::new("Test action", "Test explanation", 0.8)
            .with_example("Example 1")
            .with_example("Example 2");

        assert_eq!(hint.action, "Test action");
        assert_eq!(hint.confidence, 0.8);
        assert_eq!(hint.examples.len(), 2);
    }

    #[test]
    fn test_recoverableerror_analysis() {
        let domainerror = CoreError::DomainError(ErrorContext::new("Test domain error"));
        let recoverable = RecoverableError::error(domainerror);

        assert!(!recoverable.retryable);
        assert_eq!(recoverable.severity, ErrorSeverity::Error);
        assert!(!recoverable.hints.is_empty());
    }

    #[test]
    fn test_circuitbreaker() {
        let cb = CircuitBreaker::new(2, Duration::from_millis(100), Duration::from_millis(500));

        // First failure
        let result: std::result::Result<(), CoreError> =
            cb.execute(|| Err(CoreError::ComputationError(ErrorContext::new("Test error"))));
        assert!(result.is_err());

        // Second failure - should trigger circuit open
        let result: std::result::Result<(), CoreError> =
            cb.execute(|| Err(CoreError::ComputationError(ErrorContext::new("Test error"))));
        assert!(result.is_err());

        // Third attempt - should be blocked by circuit breaker
        let result = cb.execute(|| Ok(()));
        assert!(result.is_err());

        let status = cb.status();
        assert_eq!(status.state, CircuitState::Open);
    }

    #[test]
    fn test_recovery_retry_executor() {
        let executor = RetryExecutor::new(RecoveryStrategy::LinearBackoff {
            max_attempts: 3,
            delay: Duration::from_millis(1),
        });

        let mut attempt_count = 0;
        let result = executor.execute(|| {
            attempt_count += 1;
            if attempt_count < 3 {
                Err(CoreError::ComputationError(ErrorContext::new("Test error")))
            } else {
                Ok(42)
            }
        });

        assert_eq!(result.expect("Operation failed"), 42);
        assert_eq!(attempt_count, 3);
    }

    #[test]
    fn testerror_aggregator() {
        let mut aggregator = ErrorAggregator::new();

        aggregator.add_simpleerror(CoreError::ValueError(ErrorContext::new("Error 1")));
        aggregator.add_simpleerror(CoreError::DomainError(ErrorContext::new("Error 2")));

        assert_eq!(aggregator.error_count(), 2);
        assert!(aggregator.haserrors());

        let summary = aggregator.summary();
        assert!(summary.contains("Collected 2 error(s)"));
    }
}
