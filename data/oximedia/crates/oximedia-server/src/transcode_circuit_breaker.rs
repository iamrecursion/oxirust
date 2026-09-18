//! Circuit breaker integration with the transcoding backend.
//!
//! Wraps the `CircuitBreaker` to protect the transcoding pipeline from
//! cascading failures. When the transcoding backend begins failing beyond
//! a configurable threshold, subsequent requests are short-circuited until
//! the backend recovers.

#![allow(dead_code)]

use crate::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig, CircuitState, RequestOutcome};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Error returned when the circuit breaker rejects a transcoding request.
#[derive(Debug, Clone)]
pub struct TranscodeCircuitOpenError {
    /// Name of the backend whose circuit is open.
    pub backend: String,
    /// When the circuit was opened.
    pub opened_at: Instant,
    /// How long until the circuit transitions to half-open.
    pub retry_after: Duration,
}

impl std::fmt::Display for TranscodeCircuitOpenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Transcode backend '{}' circuit is open; retry after {:?}",
            self.backend, self.retry_after
        )
    }
}

impl std::error::Error for TranscodeCircuitOpenError {}

/// Outcome of a transcoding operation for circuit breaker recording.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TranscodeOutcome {
    /// Transcoding completed successfully.
    Success,
    /// Transcoding failed with a retriable error (counts toward threshold).
    TransientFailure,
    /// Transcoding failed with a permanent error (does NOT trip the circuit).
    PermanentFailure,
    /// Transcoding timed out (counts toward threshold).
    Timeout,
}

impl TranscodeOutcome {
    /// Whether this outcome should be counted as a circuit breaker failure.
    fn is_circuit_failure(self) -> bool {
        matches!(self, Self::TransientFailure | Self::Timeout)
    }
}

/// Configuration for the transcoding circuit breaker.
#[derive(Debug, Clone)]
pub struct TranscodeCircuitBreakerConfig {
    /// Base circuit breaker configuration.
    pub circuit: CircuitBreakerConfig,
    /// Timeout for individual transcode operations.
    pub operation_timeout: Duration,
    /// Whether to track per-format circuit breakers.
    pub per_format_tracking: bool,
    /// Cooldown period before retrying after circuit opens.
    pub cooldown: Duration,
}

impl Default for TranscodeCircuitBreakerConfig {
    fn default() -> Self {
        Self {
            circuit: CircuitBreakerConfig::default()
                .with_name("transcode")
                .with_failure_threshold(5)
                .with_open_duration(Duration::from_secs(30))
                .with_half_open_successes(2),
            operation_timeout: Duration::from_secs(300),
            per_format_tracking: true,
            cooldown: Duration::from_secs(10),
        }
    }
}

/// Per-format failure statistics.
#[derive(Debug, Clone)]
pub struct FormatStats {
    /// Total attempts for this format.
    pub total: u64,
    /// Successful completions.
    pub successes: u64,
    /// Transient failures.
    pub transient_failures: u64,
    /// Permanent failures.
    pub permanent_failures: u64,
    /// Timeouts.
    pub timeouts: u64,
    /// Last failure timestamp.
    pub last_failure: Option<Instant>,
    /// Last success timestamp.
    pub last_success: Option<Instant>,
}

impl FormatStats {
    fn new() -> Self {
        Self {
            total: 0,
            successes: 0,
            transient_failures: 0,
            permanent_failures: 0,
            timeouts: 0,
            last_failure: None,
            last_success: None,
        }
    }

    /// Success rate as a fraction 0.0 to 1.0.
    pub fn success_rate(&self) -> f64 {
        if self.total == 0 {
            return 1.0;
        }
        self.successes as f64 / self.total as f64
    }

    /// Average failure rate for transient + timeout failures.
    pub fn transient_failure_rate(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        (self.transient_failures + self.timeouts) as f64 / self.total as f64
    }
}

/// Overall health status of the transcoding backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TranscodeHealthStatus {
    /// Backend is healthy and accepting requests.
    Healthy,
    /// Backend is degraded (circuit is half-open).
    Degraded,
    /// Backend is unavailable (circuit is open).
    Unavailable,
}

impl TranscodeHealthStatus {
    /// Human-readable label.
    pub fn label(self) -> &'static str {
        match self {
            Self::Healthy => "healthy",
            Self::Degraded => "degraded",
            Self::Unavailable => "unavailable",
        }
    }
}

/// A transcoding-aware circuit breaker that integrates with the transcode engine.
///
/// Manages a global circuit breaker plus optional per-format breakers.
pub struct TranscodeCircuitBreaker {
    config: TranscodeCircuitBreakerConfig,
    /// Global circuit breaker.
    global: Arc<RwLock<CircuitBreaker>>,
    /// Per-format circuit breakers (format name -> breaker).
    per_format: Arc<RwLock<HashMap<String, CircuitBreaker>>>,
    /// Per-format statistics.
    format_stats: Arc<RwLock<HashMap<String, FormatStats>>>,
    /// Timestamp of the last global trip.
    last_trip: Arc<RwLock<Option<Instant>>>,
}

impl TranscodeCircuitBreaker {
    /// Creates a new transcoding circuit breaker.
    pub fn new(config: TranscodeCircuitBreakerConfig) -> Self {
        let global = CircuitBreaker::new(config.circuit.clone());
        Self {
            config,
            global: Arc::new(RwLock::new(global)),
            per_format: Arc::new(RwLock::new(HashMap::new())),
            format_stats: Arc::new(RwLock::new(HashMap::new())),
            last_trip: Arc::new(RwLock::new(None)),
        }
    }

    /// Checks whether a transcode request should be allowed.
    ///
    /// Returns `Ok(())` if the request is allowed, or an error if the
    /// circuit is open.
    pub fn check_allowed(&self, format: Option<&str>) -> Result<(), TranscodeCircuitOpenError> {
        // Check global circuit first
        {
            let mut global = self.global.write();
            if !global.allow_request() {
                let retry_after = self.config.circuit.open_duration;
                return Err(TranscodeCircuitOpenError {
                    backend: self.config.circuit.name.clone(),
                    opened_at: self.last_trip.read().unwrap_or(Instant::now()),
                    retry_after,
                });
            }
        }

        // Check per-format circuit if enabled
        if self.config.per_format_tracking {
            if let Some(fmt) = format {
                let mut per_format = self.per_format.write();
                let breaker = per_format.entry(fmt.to_string()).or_insert_with(|| {
                    let cfg = self
                        .config
                        .circuit
                        .clone()
                        .with_name(format!("transcode-{}", fmt));
                    CircuitBreaker::new(cfg)
                });
                if !breaker.allow_request() {
                    return Err(TranscodeCircuitOpenError {
                        backend: format!("transcode-{}", fmt),
                        opened_at: self.last_trip.read().unwrap_or(Instant::now()),
                        retry_after: self.config.circuit.open_duration,
                    });
                }
            }
        }

        Ok(())
    }

    /// Records the outcome of a transcoding operation.
    pub fn record_outcome(&self, format: Option<&str>, outcome: TranscodeOutcome) {
        let cb_outcome = if outcome.is_circuit_failure() {
            RequestOutcome::Failure
        } else {
            RequestOutcome::Success
        };

        // Record on global breaker
        {
            let mut global = self.global.write();
            let state_before = global.state();
            global.record(cb_outcome);
            let state_after = global.state();

            // Track trip events
            if state_before != CircuitState::Open && state_after == CircuitState::Open {
                *self.last_trip.write() = Some(Instant::now());
            }
        }

        // Record per-format
        if self.config.per_format_tracking {
            if let Some(fmt) = format {
                // Update per-format circuit breaker
                {
                    let mut per_format = self.per_format.write();
                    if let Some(breaker) = per_format.get_mut(fmt) {
                        breaker.record(cb_outcome);
                    }
                }

                // Update per-format stats
                let mut stats = self.format_stats.write();
                let fs = stats
                    .entry(fmt.to_string())
                    .or_insert_with(FormatStats::new);
                fs.total += 1;
                match outcome {
                    TranscodeOutcome::Success => {
                        fs.successes += 1;
                        fs.last_success = Some(Instant::now());
                    }
                    TranscodeOutcome::TransientFailure => {
                        fs.transient_failures += 1;
                        fs.last_failure = Some(Instant::now());
                    }
                    TranscodeOutcome::PermanentFailure => {
                        fs.permanent_failures += 1;
                        fs.last_failure = Some(Instant::now());
                    }
                    TranscodeOutcome::Timeout => {
                        fs.timeouts += 1;
                        fs.last_failure = Some(Instant::now());
                    }
                }
            }
        }
    }

    /// Returns the current health status of the transcoding backend.
    pub fn health_status(&self) -> TranscodeHealthStatus {
        let mut global = self.global.write();
        match global.state() {
            CircuitState::Closed => TranscodeHealthStatus::Healthy,
            CircuitState::HalfOpen => TranscodeHealthStatus::Degraded,
            CircuitState::Open => TranscodeHealthStatus::Unavailable,
        }
    }

    /// Returns the global circuit state.
    pub fn global_state(&self) -> CircuitState {
        self.global.write().state()
    }

    /// Returns the circuit state for a specific format.
    pub fn format_state(&self, format: &str) -> Option<CircuitState> {
        let mut per_format = self.per_format.write();
        per_format.get_mut(format).map(|b| b.state())
    }

    /// Returns statistics for a specific format.
    pub fn format_stats(&self, format: &str) -> Option<FormatStats> {
        self.format_stats.read().get(format).cloned()
    }

    /// Returns statistics for all formats.
    pub fn all_format_stats(&self) -> HashMap<String, FormatStats> {
        self.format_stats.read().clone()
    }

    /// Manually resets the global circuit breaker.
    pub fn reset_global(&self) {
        self.global.write().reset();
    }

    /// Manually resets a per-format circuit breaker.
    pub fn reset_format(&self, format: &str) -> bool {
        let mut per_format = self.per_format.write();
        if let Some(breaker) = per_format.get_mut(format) {
            breaker.reset();
            true
        } else {
            false
        }
    }

    /// Returns the configured operation timeout.
    pub fn operation_timeout(&self) -> Duration {
        self.config.operation_timeout
    }

    /// Returns the number of tracked formats.
    pub fn tracked_format_count(&self) -> usize {
        self.per_format.read().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> TranscodeCircuitBreakerConfig {
        TranscodeCircuitBreakerConfig {
            circuit: CircuitBreakerConfig::default()
                .with_name("test-transcode")
                .with_failure_threshold(3)
                .with_open_duration(Duration::from_millis(50))
                .with_half_open_successes(2),
            operation_timeout: Duration::from_secs(60),
            per_format_tracking: true,
            cooldown: Duration::from_millis(10),
        }
    }

    #[test]
    fn test_allows_request_when_healthy() {
        let cb = TranscodeCircuitBreaker::new(default_config());
        assert!(cb.check_allowed(None).is_ok());
        assert!(cb.check_allowed(Some("av1")).is_ok());
    }

    #[test]
    fn test_trips_after_failures() {
        let cb = TranscodeCircuitBreaker::new(default_config());
        for _ in 0..3 {
            let _ = cb.check_allowed(None);
            cb.record_outcome(None, TranscodeOutcome::TransientFailure);
        }
        assert!(cb.check_allowed(None).is_err());
    }

    #[test]
    fn test_timeout_counts_as_failure() {
        let cb = TranscodeCircuitBreaker::new(default_config());
        for _ in 0..3 {
            let _ = cb.check_allowed(None);
            cb.record_outcome(None, TranscodeOutcome::Timeout);
        }
        assert!(cb.check_allowed(None).is_err());
    }

    #[test]
    fn test_permanent_failure_does_not_trip() {
        let cb = TranscodeCircuitBreaker::new(default_config());
        for _ in 0..10 {
            let _ = cb.check_allowed(None);
            cb.record_outcome(None, TranscodeOutcome::PermanentFailure);
        }
        assert!(cb.check_allowed(None).is_ok());
    }

    #[test]
    fn test_success_resets_failure_count() {
        let cb = TranscodeCircuitBreaker::new(default_config());
        // 2 failures, then success
        let _ = cb.check_allowed(None);
        cb.record_outcome(None, TranscodeOutcome::TransientFailure);
        let _ = cb.check_allowed(None);
        cb.record_outcome(None, TranscodeOutcome::TransientFailure);
        let _ = cb.check_allowed(None);
        cb.record_outcome(None, TranscodeOutcome::Success);
        // 1 more failure should not trip (threshold=3)
        let _ = cb.check_allowed(None);
        cb.record_outcome(None, TranscodeOutcome::TransientFailure);
        assert!(cb.check_allowed(None).is_ok());
    }

    #[test]
    fn test_per_format_tracking() {
        let cb = TranscodeCircuitBreaker::new(default_config());
        // Trip the "vp9" format
        for _ in 0..3 {
            let _ = cb.check_allowed(Some("vp9"));
            cb.record_outcome(Some("vp9"), TranscodeOutcome::TransientFailure);
        }
        // vp9 should be blocked
        assert!(cb.check_allowed(Some("vp9")).is_err());
        // av1 should still work (global also tripped though)
        // Reset global to test per-format isolation
        cb.reset_global();
        assert!(cb.check_allowed(Some("av1")).is_ok());
    }

    #[test]
    fn test_health_status_healthy() {
        let cb = TranscodeCircuitBreaker::new(default_config());
        assert_eq!(cb.health_status(), TranscodeHealthStatus::Healthy);
    }

    #[test]
    fn test_health_status_unavailable() {
        let cb = TranscodeCircuitBreaker::new(default_config());
        for _ in 0..3 {
            let _ = cb.check_allowed(None);
            cb.record_outcome(None, TranscodeOutcome::TransientFailure);
        }
        assert_eq!(cb.health_status(), TranscodeHealthStatus::Unavailable);
    }

    #[test]
    fn test_health_status_degraded() {
        let config = TranscodeCircuitBreakerConfig {
            circuit: CircuitBreakerConfig::default()
                .with_failure_threshold(1)
                .with_open_duration(Duration::from_millis(1))
                .with_half_open_successes(2),
            ..default_config()
        };
        let cb = TranscodeCircuitBreaker::new(config);
        let _ = cb.check_allowed(None);
        cb.record_outcome(None, TranscodeOutcome::TransientFailure);
        std::thread::sleep(Duration::from_millis(5));
        assert_eq!(cb.health_status(), TranscodeHealthStatus::Degraded);
    }

    #[test]
    fn test_format_stats_tracked() {
        let cb = TranscodeCircuitBreaker::new(default_config());
        let _ = cb.check_allowed(Some("av1"));
        cb.record_outcome(Some("av1"), TranscodeOutcome::Success);
        let _ = cb.check_allowed(Some("av1"));
        cb.record_outcome(Some("av1"), TranscodeOutcome::TransientFailure);

        let stats = cb.format_stats("av1").expect("stats should exist");
        assert_eq!(stats.total, 2);
        assert_eq!(stats.successes, 1);
        assert_eq!(stats.transient_failures, 1);
        assert!((stats.success_rate() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_all_format_stats() {
        let cb = TranscodeCircuitBreaker::new(default_config());
        let _ = cb.check_allowed(Some("av1"));
        cb.record_outcome(Some("av1"), TranscodeOutcome::Success);
        let _ = cb.check_allowed(Some("vp9"));
        cb.record_outcome(Some("vp9"), TranscodeOutcome::Success);
        assert_eq!(cb.all_format_stats().len(), 2);
    }

    #[test]
    fn test_reset_global() {
        let cb = TranscodeCircuitBreaker::new(default_config());
        for _ in 0..3 {
            let _ = cb.check_allowed(None);
            cb.record_outcome(None, TranscodeOutcome::TransientFailure);
        }
        assert!(cb.check_allowed(None).is_err());
        cb.reset_global();
        assert!(cb.check_allowed(None).is_ok());
    }

    #[test]
    fn test_reset_format() {
        let cb = TranscodeCircuitBreaker::new(default_config());
        // Create and trip format
        for _ in 0..3 {
            let _ = cb.check_allowed(Some("vp9"));
            cb.record_outcome(Some("vp9"), TranscodeOutcome::TransientFailure);
        }
        assert!(cb.reset_format("vp9"));
        assert!(!cb.reset_format("unknown"));
    }

    #[test]
    fn test_tracked_format_count() {
        let cb = TranscodeCircuitBreaker::new(default_config());
        assert_eq!(cb.tracked_format_count(), 0);
        let _ = cb.check_allowed(Some("av1"));
        assert_eq!(cb.tracked_format_count(), 1);
        let _ = cb.check_allowed(Some("vp9"));
        assert_eq!(cb.tracked_format_count(), 2);
    }

    #[test]
    fn test_operation_timeout() {
        let cb = TranscodeCircuitBreaker::new(default_config());
        assert_eq!(cb.operation_timeout(), Duration::from_secs(60));
    }

    #[test]
    fn test_transcode_circuit_open_error_display() {
        let err = TranscodeCircuitOpenError {
            backend: "test".to_string(),
            opened_at: Instant::now(),
            retry_after: Duration::from_secs(30),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("test"));
        assert!(msg.contains("circuit is open"));
    }

    #[test]
    fn test_transcode_outcome_is_circuit_failure() {
        assert!(TranscodeOutcome::TransientFailure.is_circuit_failure());
        assert!(TranscodeOutcome::Timeout.is_circuit_failure());
        assert!(!TranscodeOutcome::Success.is_circuit_failure());
        assert!(!TranscodeOutcome::PermanentFailure.is_circuit_failure());
    }

    #[test]
    fn test_health_status_labels() {
        assert_eq!(TranscodeHealthStatus::Healthy.label(), "healthy");
        assert_eq!(TranscodeHealthStatus::Degraded.label(), "degraded");
        assert_eq!(TranscodeHealthStatus::Unavailable.label(), "unavailable");
    }

    #[test]
    fn test_format_stats_transient_failure_rate() {
        let mut fs = FormatStats::new();
        fs.total = 10;
        fs.transient_failures = 2;
        fs.timeouts = 1;
        assert!((fs.transient_failure_rate() - 0.3).abs() < 1e-9);
    }

    #[test]
    fn test_format_stats_empty() {
        let fs = FormatStats::new();
        assert!((fs.success_rate() - 1.0).abs() < 1e-9);
        assert!((fs.transient_failure_rate() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_global_state() {
        let cb = TranscodeCircuitBreaker::new(default_config());
        assert_eq!(cb.global_state(), CircuitState::Closed);
    }

    #[test]
    fn test_format_state() {
        let cb = TranscodeCircuitBreaker::new(default_config());
        assert!(cb.format_state("av1").is_none());
        let _ = cb.check_allowed(Some("av1"));
        assert_eq!(cb.format_state("av1"), Some(CircuitState::Closed));
    }

    #[test]
    fn test_default_config_values() {
        let cfg = TranscodeCircuitBreakerConfig::default();
        assert_eq!(cfg.circuit.failure_threshold, 5);
        assert_eq!(cfg.operation_timeout, Duration::from_secs(300));
        assert!(cfg.per_format_tracking);
    }
}
