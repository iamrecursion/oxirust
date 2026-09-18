//! Health check and readiness monitoring for the OxiMedia server.
//!
//! Provides liveness/readiness probes, dependency health checks,
//! and structured JSON response formatting for orchestrators such as
//! Kubernetes, Consul, and load balancers.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Outcome of a single health check
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckStatus {
    /// The component is healthy
    Healthy,
    /// The component is degraded but still functional
    Degraded,
    /// The component is unhealthy
    Unhealthy,
    /// The check timed out
    Timeout,
    /// The check was skipped (e.g., optional dependency not configured)
    Skipped,
}

impl CheckStatus {
    /// Returns the HTTP status code appropriate for this check outcome
    pub fn http_status(&self) -> u16 {
        match self {
            CheckStatus::Healthy | CheckStatus::Skipped | CheckStatus::Degraded => 200,
            CheckStatus::Unhealthy | CheckStatus::Timeout => 503,
        }
    }

    /// Returns the display label for this status
    pub fn label(&self) -> &'static str {
        match self {
            CheckStatus::Healthy => "healthy",
            CheckStatus::Degraded => "degraded",
            CheckStatus::Unhealthy => "unhealthy",
            CheckStatus::Timeout => "timeout",
            CheckStatus::Skipped => "skipped",
        }
    }

    /// Returns true if this status is considered a passing check
    pub fn is_passing(&self) -> bool {
        matches!(
            self,
            CheckStatus::Healthy | CheckStatus::Degraded | CheckStatus::Skipped
        )
    }
}

/// Result of an individual dependency check
#[derive(Debug, Clone)]
pub struct CheckResult {
    /// Name of the component or dependency
    pub name: String,
    /// Health status
    pub status: CheckStatus,
    /// Optional detail message
    pub message: Option<String>,
    /// Latency of the check
    pub latency: Duration,
    /// Arbitrary metadata (e.g., version, connection count)
    pub metadata: HashMap<String, String>,
}

impl CheckResult {
    /// Creates a healthy result with no message
    pub fn healthy(name: impl Into<String>, latency: Duration) -> Self {
        Self {
            name: name.into(),
            status: CheckStatus::Healthy,
            message: None,
            latency,
            metadata: HashMap::new(),
        }
    }

    /// Creates an unhealthy result with an error message
    pub fn unhealthy(
        name: impl Into<String>,
        latency: Duration,
        message: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            status: CheckStatus::Unhealthy,
            message: Some(message.into()),
            latency,
            metadata: HashMap::new(),
        }
    }

    /// Creates a degraded result
    pub fn degraded(
        name: impl Into<String>,
        latency: Duration,
        message: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            status: CheckStatus::Degraded,
            message: Some(message.into()),
            latency,
            metadata: HashMap::new(),
        }
    }

    /// Creates a skipped result
    pub fn skipped(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status: CheckStatus::Skipped,
            message: None,
            latency: Duration::ZERO,
            metadata: HashMap::new(),
        }
    }

    /// Attaches metadata to the result
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Aggregated health response
#[derive(Debug, Clone)]
pub struct HealthResponse {
    /// Overall status derived from all checks
    pub status: CheckStatus,
    /// Server version string
    pub version: String,
    /// Individual check results
    pub checks: Vec<CheckResult>,
    /// Time taken to run all checks
    pub total_latency: Duration,
    /// When the response was generated (uptime reference)
    pub generated_at: Instant,
}

impl HealthResponse {
    /// Creates a new health response from a list of check results
    pub fn new(
        version: impl Into<String>,
        checks: Vec<CheckResult>,
        total_latency: Duration,
    ) -> Self {
        let status = Self::aggregate_status(&checks);
        Self {
            status,
            version: version.into(),
            checks,
            total_latency,
            generated_at: Instant::now(),
        }
    }

    /// Computes the aggregate status from individual results:
    /// - Any `Unhealthy` or `Timeout` → `Unhealthy`
    /// - Any `Degraded` → `Degraded`
    /// - All `Healthy` / `Skipped` → `Healthy`
    fn aggregate_status(checks: &[CheckResult]) -> CheckStatus {
        let has_unhealthy = checks
            .iter()
            .any(|c| matches!(c.status, CheckStatus::Unhealthy | CheckStatus::Timeout));
        if has_unhealthy {
            return CheckStatus::Unhealthy;
        }
        let has_degraded = checks.iter().any(|c| c.status == CheckStatus::Degraded);
        if has_degraded {
            return CheckStatus::Degraded;
        }
        CheckStatus::Healthy
    }

    /// Returns the appropriate HTTP status code
    pub fn http_status(&self) -> u16 {
        self.status.http_status()
    }

    /// Converts the response to a JSON-like string (no external deps)
    pub fn to_json(&self) -> String {
        let checks_json: Vec<String> = self
            .checks
            .iter()
            .map(|c| {
                let msg = c
                    .message
                    .as_deref()
                    .map(|m| format!(r#","message":"{m}""#))
                    .unwrap_or_default();
                let meta: Vec<String> = c
                    .metadata
                    .iter()
                    .map(|(k, v)| format!(r#""{k}":"{v}""#))
                    .collect();
                let meta_str = if meta.is_empty() {
                    String::new()
                } else {
                    format!(r#","metadata":{{{}}}"#, meta.join(","))
                };
                format!(
                    r#"{{"name":"{}","status":"{}","latency_ms":{:.2}{msg}{meta_str}}}"#,
                    c.name,
                    c.status.label(),
                    c.latency.as_secs_f64() * 1000.0,
                )
            })
            .collect();

        format!(
            r#"{{"status":"{}","version":"{}","total_latency_ms":{:.2},"checks":[{}]}}"#,
            self.status.label(),
            self.version,
            self.total_latency.as_secs_f64() * 1000.0,
            checks_json.join(","),
        )
    }
}

/// Configuration for a dependency health check
#[derive(Debug, Clone)]
pub struct CheckConfig {
    /// Name of the dependency
    pub name: String,
    /// Timeout for the check
    pub timeout: Duration,
    /// Whether the dependency is required (affects aggregate status)
    pub required: bool,
}

impl CheckConfig {
    /// Creates a required check config
    pub fn required(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            timeout: Duration::from_secs(5),
            required: true,
        }
    }

    /// Creates an optional check config
    pub fn optional(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            timeout: Duration::from_secs(5),
            required: false,
        }
    }

    /// Sets a custom timeout
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

/// Mock dependency simulator for testing checks
#[derive(Debug, Clone)]
pub struct MockDependency {
    /// Dependency name
    pub name: String,
    /// Whether the dependency should report healthy
    pub healthy: bool,
    /// Simulated latency
    pub latency: Duration,
    /// Optional detail message
    pub message: Option<String>,
}

impl MockDependency {
    /// Creates a healthy mock dependency
    pub fn healthy(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            healthy: true,
            latency: Duration::from_millis(1),
            message: None,
        }
    }

    /// Creates an unhealthy mock dependency
    pub fn unhealthy(name: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            healthy: false,
            latency: Duration::from_millis(10),
            message: Some(message.into()),
        }
    }

    /// Runs the mock check and returns a CheckResult
    pub fn check(&self) -> CheckResult {
        if self.healthy {
            CheckResult::healthy(&self.name, self.latency)
        } else {
            CheckResult::unhealthy(
                &self.name,
                self.latency,
                self.message
                    .clone()
                    .unwrap_or_else(|| "unhealthy".to_string()),
            )
        }
    }
}

/// Liveness probe — reports whether the process itself is alive
#[derive(Debug, Clone)]
pub struct LivenessProbe {
    /// Server version
    pub version: String,
}

impl LivenessProbe {
    /// Creates a new liveness probe
    pub fn new(version: impl Into<String>) -> Self {
        Self {
            version: version.into(),
        }
    }

    /// Runs the liveness check (always returns healthy for a live process)
    pub fn check(&self) -> HealthResponse {
        let result = CheckResult::healthy("process", Duration::from_nanos(1));
        HealthResponse::new(&self.version, vec![result], Duration::from_nanos(1))
    }
}

/// Readiness probe — reports whether the server is ready to accept traffic
pub struct ReadinessProbe {
    /// Server version
    pub version: String,
    /// List of dependency checks to run
    pub dependencies: Vec<MockDependency>,
}

impl ReadinessProbe {
    /// Creates a new readiness probe
    pub fn new(version: impl Into<String>) -> Self {
        Self {
            version: version.into(),
            dependencies: Vec::new(),
        }
    }

    /// Adds a dependency check
    #[must_use]
    pub fn add_dependency(mut self, dep: MockDependency) -> Self {
        self.dependencies.push(dep);
        self
    }

    /// Runs all dependency checks and returns an aggregated health response
    pub fn check(&self) -> HealthResponse {
        let start = Instant::now();
        let checks: Vec<CheckResult> = self.dependencies.iter().map(|d| d.check()).collect();
        let total = start.elapsed();
        HealthResponse::new(&self.version, checks, total)
    }
}

// ── Deep Health Checks ───────────────────────────────────────────────────────

/// Category of a deep health check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CheckCategory {
    /// Database connectivity and query latency.
    Database,
    /// Filesystem / media storage availability.
    Storage,
    /// Transcoding worker availability.
    TranscodeWorker,
    /// External dependency (CDN, cloud storage, etc.).
    ExternalService,
    /// Memory / resource utilization.
    Resources,
}

impl CheckCategory {
    /// Returns the display label for this category.
    pub fn label(&self) -> &'static str {
        match self {
            CheckCategory::Database => "database",
            CheckCategory::Storage => "storage",
            CheckCategory::TranscodeWorker => "transcode_worker",
            CheckCategory::ExternalService => "external_service",
            CheckCategory::Resources => "resources",
        }
    }
}

/// Configuration for a deep health check probe.
#[derive(Debug, Clone)]
pub struct DeepCheckConfig {
    /// The check category.
    pub category: CheckCategory,
    /// Human-readable name.
    pub name: String,
    /// Maximum acceptable latency before marking as degraded.
    pub degraded_threshold: Duration,
    /// Maximum acceptable latency before marking as unhealthy (timeout).
    pub unhealthy_threshold: Duration,
    /// Whether this check is critical (affects overall readiness).
    pub critical: bool,
}

impl DeepCheckConfig {
    /// Creates a new deep check config.
    pub fn new(category: CheckCategory, name: impl Into<String>) -> Self {
        Self {
            category,
            name: name.into(),
            degraded_threshold: Duration::from_millis(500),
            unhealthy_threshold: Duration::from_secs(5),
            critical: true,
        }
    }

    /// Sets the degraded latency threshold.
    #[must_use]
    pub fn with_degraded_threshold(mut self, threshold: Duration) -> Self {
        self.degraded_threshold = threshold;
        self
    }

    /// Sets the unhealthy latency threshold.
    #[must_use]
    pub fn with_unhealthy_threshold(mut self, threshold: Duration) -> Self {
        self.unhealthy_threshold = threshold;
        self
    }

    /// Marks this check as non-critical (will not affect overall readiness).
    #[must_use]
    pub fn non_critical(mut self) -> Self {
        self.critical = false;
        self
    }
}

/// Simulated deep health check that evaluates latency and availability.
#[derive(Debug, Clone)]
pub struct DeepHealthCheck {
    /// Configuration for this check.
    config: DeepCheckConfig,
    /// Whether this check's target is currently reachable.
    reachable: bool,
    /// Simulated latency for the check.
    latency: Duration,
    /// Optional error message when unreachable.
    error_message: Option<String>,
}

impl DeepHealthCheck {
    /// Creates a new deep health check.
    pub fn new(config: DeepCheckConfig) -> Self {
        Self {
            config,
            reachable: true,
            latency: Duration::from_millis(1),
            error_message: None,
        }
    }

    /// Simulates setting the check result (for testing / offline evaluation).
    pub fn set_state(&mut self, reachable: bool, latency: Duration, error: Option<String>) {
        self.reachable = reachable;
        self.latency = latency;
        self.error_message = error;
    }

    /// Runs the check and returns a `CheckResult`.
    pub fn run(&self) -> CheckResult {
        if !self.reachable {
            return CheckResult::unhealthy(
                &self.config.name,
                self.latency,
                self.error_message
                    .clone()
                    .unwrap_or_else(|| "unreachable".to_string()),
            )
            .with_metadata("category", self.config.category.label())
            .with_metadata(
                "critical",
                if self.config.critical {
                    "true"
                } else {
                    "false"
                },
            );
        }

        if self.latency >= self.config.unhealthy_threshold {
            CheckResult::unhealthy(
                &self.config.name,
                self.latency,
                format!(
                    "latency {}ms exceeds unhealthy threshold {}ms",
                    self.latency.as_millis(),
                    self.config.unhealthy_threshold.as_millis()
                ),
            )
            .with_metadata("category", self.config.category.label())
        } else if self.latency >= self.config.degraded_threshold {
            CheckResult::degraded(
                &self.config.name,
                self.latency,
                format!(
                    "latency {}ms exceeds degraded threshold {}ms",
                    self.latency.as_millis(),
                    self.config.degraded_threshold.as_millis()
                ),
            )
            .with_metadata("category", self.config.category.label())
        } else {
            CheckResult::healthy(&self.config.name, self.latency)
                .with_metadata("category", self.config.category.label())
                .with_metadata("latency_ms", format!("{}", self.latency.as_millis()))
        }
    }

    /// Returns the check category.
    pub fn category(&self) -> CheckCategory {
        self.config.category
    }

    /// Returns whether this check is critical.
    pub fn is_critical(&self) -> bool {
        self.config.critical
    }
}

/// A comprehensive deep health probe that runs multiple categorized checks
/// and produces an aggregated result with separate readiness/liveness outcomes.
pub struct DeepHealthProbe {
    /// Server version string.
    pub version: String,
    /// Registered deep health checks.
    checks: Vec<DeepHealthCheck>,
}

impl DeepHealthProbe {
    /// Creates a new deep health probe.
    pub fn new(version: impl Into<String>) -> Self {
        Self {
            version: version.into(),
            checks: Vec::new(),
        }
    }

    /// Registers a deep health check.
    #[must_use]
    pub fn add_check(mut self, check: DeepHealthCheck) -> Self {
        self.checks.push(check);
        self
    }

    /// Adds a mutable reference to a check.
    pub fn add_check_mut(&mut self, check: DeepHealthCheck) {
        self.checks.push(check);
    }

    /// Returns the number of registered checks.
    pub fn check_count(&self) -> usize {
        self.checks.len()
    }

    /// Runs all registered checks and returns an aggregated health response.
    pub fn run_all(&self) -> DeepHealthResponse {
        let start = Instant::now();
        let results: Vec<CheckResult> = self.checks.iter().map(|c| c.run()).collect();
        let total_latency = start.elapsed();

        // Compute overall status considering criticality
        let overall_status = self.compute_overall_status(&results);

        // Compute readiness (only critical checks affect readiness)
        let critical_results: Vec<&CheckResult> = self
            .checks
            .iter()
            .zip(results.iter())
            .filter(|(check, _)| check.is_critical())
            .map(|(_, result)| result)
            .collect();
        let readiness_status = HealthResponse::aggregate_status_static(&critical_results);

        DeepHealthResponse {
            health: HealthResponse::new(&self.version, results, total_latency),
            overall_status,
            readiness_status,
            liveness_status: CheckStatus::Healthy, // Always healthy if we're running
        }
    }

    fn compute_overall_status(&self, results: &[CheckResult]) -> CheckStatus {
        let critical_unhealthy = self
            .checks
            .iter()
            .zip(results.iter())
            .any(|(check, result)| {
                check.is_critical()
                    && matches!(result.status, CheckStatus::Unhealthy | CheckStatus::Timeout)
            });
        if critical_unhealthy {
            return CheckStatus::Unhealthy;
        }

        let any_unhealthy = results
            .iter()
            .any(|r| matches!(r.status, CheckStatus::Unhealthy | CheckStatus::Timeout));
        if any_unhealthy {
            return CheckStatus::Degraded; // Non-critical unhealthy → degraded
        }

        let any_degraded = results.iter().any(|r| r.status == CheckStatus::Degraded);
        if any_degraded {
            return CheckStatus::Degraded;
        }

        CheckStatus::Healthy
    }
}

impl HealthResponse {
    /// Static version of aggregate_status that works with references.
    fn aggregate_status_static(checks: &[&CheckResult]) -> CheckStatus {
        let has_unhealthy = checks
            .iter()
            .any(|c| matches!(c.status, CheckStatus::Unhealthy | CheckStatus::Timeout));
        if has_unhealthy {
            return CheckStatus::Unhealthy;
        }
        let has_degraded = checks.iter().any(|c| c.status == CheckStatus::Degraded);
        if has_degraded {
            return CheckStatus::Degraded;
        }
        CheckStatus::Healthy
    }
}

/// Extended health response with readiness and liveness decomposition.
#[derive(Debug)]
pub struct DeepHealthResponse {
    /// The underlying health response with all check results.
    pub health: HealthResponse,
    /// Overall status (considering criticality weighting).
    pub overall_status: CheckStatus,
    /// Readiness status (only critical checks considered).
    pub readiness_status: CheckStatus,
    /// Liveness status (always healthy if the process is running).
    pub liveness_status: CheckStatus,
}

impl DeepHealthResponse {
    /// Returns `true` if the service is ready to accept traffic.
    pub fn is_ready(&self) -> bool {
        self.readiness_status.is_passing()
    }

    /// Returns `true` if the service is alive.
    pub fn is_alive(&self) -> bool {
        self.liveness_status.is_passing()
    }

    /// Returns the HTTP status code for the readiness probe.
    pub fn readiness_http_status(&self) -> u16 {
        self.readiness_status.http_status()
    }

    /// Returns checks that are unhealthy.
    pub fn unhealthy_checks(&self) -> Vec<&CheckResult> {
        self.health
            .checks
            .iter()
            .filter(|c| matches!(c.status, CheckStatus::Unhealthy | CheckStatus::Timeout))
            .collect()
    }

    /// Returns checks that are degraded.
    pub fn degraded_checks(&self) -> Vec<&CheckResult> {
        self.health
            .checks
            .iter()
            .filter(|c| c.status == CheckStatus::Degraded)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_status_http_status() {
        assert_eq!(CheckStatus::Healthy.http_status(), 200);
        assert_eq!(CheckStatus::Degraded.http_status(), 200);
        assert_eq!(CheckStatus::Unhealthy.http_status(), 503);
        assert_eq!(CheckStatus::Timeout.http_status(), 503);
        assert_eq!(CheckStatus::Skipped.http_status(), 200);
    }

    #[test]
    fn test_check_status_label() {
        assert_eq!(CheckStatus::Healthy.label(), "healthy");
        assert_eq!(CheckStatus::Degraded.label(), "degraded");
        assert_eq!(CheckStatus::Unhealthy.label(), "unhealthy");
        assert_eq!(CheckStatus::Timeout.label(), "timeout");
        assert_eq!(CheckStatus::Skipped.label(), "skipped");
    }

    #[test]
    fn test_check_status_is_passing() {
        assert!(CheckStatus::Healthy.is_passing());
        assert!(CheckStatus::Degraded.is_passing());
        assert!(CheckStatus::Skipped.is_passing());
        assert!(!CheckStatus::Unhealthy.is_passing());
        assert!(!CheckStatus::Timeout.is_passing());
    }

    #[test]
    fn test_check_result_healthy() {
        let r = CheckResult::healthy("database", Duration::from_millis(5));
        assert_eq!(r.status, CheckStatus::Healthy);
        assert!(r.message.is_none());
    }

    #[test]
    fn test_check_result_unhealthy() {
        let r = CheckResult::unhealthy("redis", Duration::from_secs(3), "connection refused");
        assert_eq!(r.status, CheckStatus::Unhealthy);
        assert_eq!(r.message.as_deref(), Some("connection refused"));
    }

    #[test]
    fn test_check_result_degraded() {
        let r = CheckResult::degraded("cache", Duration::from_millis(200), "high latency");
        assert_eq!(r.status, CheckStatus::Degraded);
    }

    #[test]
    fn test_check_result_skipped() {
        let r = CheckResult::skipped("optional-service");
        assert_eq!(r.status, CheckStatus::Skipped);
        assert_eq!(r.latency, Duration::ZERO);
    }

    #[test]
    fn test_check_result_with_metadata() {
        let r = CheckResult::healthy("db", Duration::from_millis(2))
            .with_metadata("version", "14.1")
            .with_metadata("connections", "42");
        assert_eq!(r.metadata.get("version").map(String::as_str), Some("14.1"));
        assert_eq!(
            r.metadata.get("connections").map(String::as_str),
            Some("42")
        );
    }

    #[test]
    fn test_health_response_all_healthy() {
        let checks = vec![
            CheckResult::healthy("db", Duration::from_millis(2)),
            CheckResult::healthy("cache", Duration::from_millis(1)),
        ];
        let resp = HealthResponse::new("1.0.0", checks, Duration::from_millis(3));
        assert_eq!(resp.status, CheckStatus::Healthy);
        assert_eq!(resp.http_status(), 200);
    }

    #[test]
    fn test_health_response_degraded() {
        let checks = vec![
            CheckResult::healthy("db", Duration::from_millis(2)),
            CheckResult::degraded("cache", Duration::from_millis(100), "slow"),
        ];
        let resp = HealthResponse::new("1.0.0", checks, Duration::from_millis(102));
        assert_eq!(resp.status, CheckStatus::Degraded);
        assert_eq!(resp.http_status(), 200);
    }

    #[test]
    fn test_health_response_unhealthy() {
        let checks = vec![
            CheckResult::healthy("db", Duration::from_millis(2)),
            CheckResult::unhealthy("queue", Duration::from_secs(5), "timeout"),
        ];
        let resp = HealthResponse::new("1.0.0", checks, Duration::from_millis(5002));
        assert_eq!(resp.status, CheckStatus::Unhealthy);
        assert_eq!(resp.http_status(), 503);
    }

    #[test]
    fn test_health_response_to_json_contains_status() {
        let checks = vec![CheckResult::healthy("db", Duration::from_millis(1))];
        let resp = HealthResponse::new("2.0.0", checks, Duration::from_millis(1));
        let json = resp.to_json();
        assert!(json.contains(r#""status":"healthy""#));
        assert!(json.contains(r#""version":"2.0.0""#));
        assert!(json.contains("db"));
    }

    #[test]
    fn test_liveness_probe_always_healthy() {
        let probe = LivenessProbe::new("1.2.3");
        let resp = probe.check();
        assert_eq!(resp.status, CheckStatus::Healthy);
        assert_eq!(resp.version, "1.2.3");
    }

    #[test]
    fn test_readiness_probe_all_healthy() {
        let probe = ReadinessProbe::new("1.0.0")
            .add_dependency(MockDependency::healthy("db"))
            .add_dependency(MockDependency::healthy("cache"));
        let resp = probe.check();
        assert_eq!(resp.status, CheckStatus::Healthy);
        assert_eq!(resp.checks.len(), 2);
    }

    #[test]
    fn test_readiness_probe_one_unhealthy() {
        let probe = ReadinessProbe::new("1.0.0")
            .add_dependency(MockDependency::healthy("db"))
            .add_dependency(MockDependency::unhealthy("redis", "ECONNREFUSED"));
        let resp = probe.check();
        assert_eq!(resp.status, CheckStatus::Unhealthy);
        assert_eq!(resp.http_status(), 503);
    }

    #[test]
    fn test_check_config_required() {
        let cfg = CheckConfig::required("database");
        assert!(cfg.required);
        assert_eq!(cfg.timeout, Duration::from_secs(5));
    }

    #[test]
    fn test_check_config_optional_with_timeout() {
        let cfg = CheckConfig::optional("analytics").with_timeout(Duration::from_secs(2));
        assert!(!cfg.required);
        assert_eq!(cfg.timeout, Duration::from_secs(2));
    }

    // ── Deep Health Check Tests ─────────────────────────────────────────────

    #[test]
    fn test_check_category_labels() {
        assert_eq!(CheckCategory::Database.label(), "database");
        assert_eq!(CheckCategory::Storage.label(), "storage");
        assert_eq!(CheckCategory::TranscodeWorker.label(), "transcode_worker");
        assert_eq!(CheckCategory::ExternalService.label(), "external_service");
        assert_eq!(CheckCategory::Resources.label(), "resources");
    }

    #[test]
    fn test_deep_check_config_defaults() {
        let cfg = DeepCheckConfig::new(CheckCategory::Database, "postgres");
        assert!(cfg.critical);
        assert_eq!(cfg.degraded_threshold, Duration::from_millis(500));
        assert_eq!(cfg.unhealthy_threshold, Duration::from_secs(5));
    }

    #[test]
    fn test_deep_check_config_builder() {
        let cfg = DeepCheckConfig::new(CheckCategory::Storage, "media-storage")
            .with_degraded_threshold(Duration::from_millis(200))
            .with_unhealthy_threshold(Duration::from_secs(2))
            .non_critical();
        assert!(!cfg.critical);
        assert_eq!(cfg.degraded_threshold, Duration::from_millis(200));
        assert_eq!(cfg.unhealthy_threshold, Duration::from_secs(2));
    }

    #[test]
    fn test_deep_health_check_healthy() {
        let cfg = DeepCheckConfig::new(CheckCategory::Database, "sqlite");
        let check = DeepHealthCheck::new(cfg);
        let result = check.run();
        assert_eq!(result.status, CheckStatus::Healthy);
        assert_eq!(
            result.metadata.get("category").map(String::as_str),
            Some("database")
        );
    }

    #[test]
    fn test_deep_health_check_degraded_latency() {
        let cfg = DeepCheckConfig::new(CheckCategory::Database, "sqlite")
            .with_degraded_threshold(Duration::from_millis(100));
        let mut check = DeepHealthCheck::new(cfg);
        check.set_state(true, Duration::from_millis(200), None);
        let result = check.run();
        assert_eq!(result.status, CheckStatus::Degraded);
        assert!(result.message.is_some());
    }

    #[test]
    fn test_deep_health_check_unhealthy_latency() {
        let cfg = DeepCheckConfig::new(CheckCategory::Database, "sqlite")
            .with_unhealthy_threshold(Duration::from_secs(1));
        let mut check = DeepHealthCheck::new(cfg);
        check.set_state(true, Duration::from_secs(2), None);
        let result = check.run();
        assert_eq!(result.status, CheckStatus::Unhealthy);
    }

    #[test]
    fn test_deep_health_check_unreachable() {
        let cfg = DeepCheckConfig::new(CheckCategory::Storage, "s3");
        let mut check = DeepHealthCheck::new(cfg);
        check.set_state(
            false,
            Duration::from_millis(10),
            Some("connection refused".to_string()),
        );
        let result = check.run();
        assert_eq!(result.status, CheckStatus::Unhealthy);
        assert_eq!(result.message.as_deref(), Some("connection refused"));
        assert_eq!(
            result.metadata.get("critical").map(String::as_str),
            Some("true")
        );
    }

    #[test]
    fn test_deep_health_check_non_critical_unreachable() {
        let cfg = DeepCheckConfig::new(CheckCategory::ExternalService, "analytics").non_critical();
        let mut check = DeepHealthCheck::new(cfg);
        check.set_state(false, Duration::from_millis(5), None);
        let result = check.run();
        assert_eq!(result.status, CheckStatus::Unhealthy);
        assert_eq!(
            result.metadata.get("critical").map(String::as_str),
            Some("false")
        );
    }

    #[test]
    fn test_deep_health_probe_all_healthy() {
        let probe = DeepHealthProbe::new("1.0.0")
            .add_check(DeepHealthCheck::new(DeepCheckConfig::new(
                CheckCategory::Database,
                "sqlite",
            )))
            .add_check(DeepHealthCheck::new(DeepCheckConfig::new(
                CheckCategory::Storage,
                "local-fs",
            )));
        let resp = probe.run_all();
        assert_eq!(resp.overall_status, CheckStatus::Healthy);
        assert_eq!(resp.readiness_status, CheckStatus::Healthy);
        assert!(resp.is_ready());
        assert!(resp.is_alive());
        assert_eq!(resp.readiness_http_status(), 200);
    }

    #[test]
    fn test_deep_health_probe_critical_unhealthy() {
        let mut db_check =
            DeepHealthCheck::new(DeepCheckConfig::new(CheckCategory::Database, "sqlite"));
        db_check.set_state(false, Duration::from_millis(5), Some("db down".to_string()));

        let probe =
            DeepHealthProbe::new("1.0.0")
                .add_check(db_check)
                .add_check(DeepHealthCheck::new(DeepCheckConfig::new(
                    CheckCategory::Storage,
                    "local-fs",
                )));
        let resp = probe.run_all();
        assert_eq!(resp.overall_status, CheckStatus::Unhealthy);
        assert_eq!(resp.readiness_status, CheckStatus::Unhealthy);
        assert!(!resp.is_ready());
        assert_eq!(resp.readiness_http_status(), 503);
        assert_eq!(resp.unhealthy_checks().len(), 1);
    }

    #[test]
    fn test_deep_health_probe_non_critical_unhealthy_degrades() {
        let mut analytics_check = DeepHealthCheck::new(
            DeepCheckConfig::new(CheckCategory::ExternalService, "analytics").non_critical(),
        );
        analytics_check.set_state(false, Duration::from_millis(5), None);

        let probe = DeepHealthProbe::new("1.0.0")
            .add_check(DeepHealthCheck::new(DeepCheckConfig::new(
                CheckCategory::Database,
                "sqlite",
            )))
            .add_check(analytics_check);
        let resp = probe.run_all();
        // Non-critical unhealthy → overall degraded, but readiness still healthy
        assert_eq!(resp.overall_status, CheckStatus::Degraded);
        assert_eq!(resp.readiness_status, CheckStatus::Healthy);
        assert!(resp.is_ready());
    }

    #[test]
    fn test_deep_health_probe_degraded_checks() {
        let mut slow_check = DeepHealthCheck::new(
            DeepCheckConfig::new(CheckCategory::Database, "sqlite")
                .with_degraded_threshold(Duration::from_millis(50)),
        );
        slow_check.set_state(true, Duration::from_millis(100), None);

        let probe = DeepHealthProbe::new("1.0.0").add_check(slow_check);
        let resp = probe.run_all();
        assert_eq!(resp.overall_status, CheckStatus::Degraded);
        assert_eq!(resp.degraded_checks().len(), 1);
    }

    #[test]
    fn test_deep_health_probe_check_count() {
        let probe = DeepHealthProbe::new("1.0.0")
            .add_check(DeepHealthCheck::new(DeepCheckConfig::new(
                CheckCategory::Database,
                "db",
            )))
            .add_check(DeepHealthCheck::new(DeepCheckConfig::new(
                CheckCategory::Storage,
                "fs",
            )))
            .add_check(DeepHealthCheck::new(DeepCheckConfig::new(
                CheckCategory::TranscodeWorker,
                "workers",
            )));
        assert_eq!(probe.check_count(), 3);
    }

    #[test]
    fn test_deep_health_probe_add_check_mut() {
        let mut probe = DeepHealthProbe::new("1.0.0");
        probe.add_check_mut(DeepHealthCheck::new(DeepCheckConfig::new(
            CheckCategory::Database,
            "db",
        )));
        assert_eq!(probe.check_count(), 1);
    }

    #[test]
    fn test_deep_health_response_always_alive() {
        // Even when everything is down, liveness should be healthy
        let mut db_check =
            DeepHealthCheck::new(DeepCheckConfig::new(CheckCategory::Database, "db"));
        db_check.set_state(false, Duration::from_millis(5), None);
        let probe = DeepHealthProbe::new("1.0.0").add_check(db_check);
        let resp = probe.run_all();
        assert!(resp.is_alive());
        assert_eq!(resp.liveness_status, CheckStatus::Healthy);
    }

    #[test]
    fn test_deep_health_check_category_accessors() {
        let check = DeepHealthCheck::new(
            DeepCheckConfig::new(CheckCategory::TranscodeWorker, "ffmpeg-pool").non_critical(),
        );
        assert_eq!(check.category(), CheckCategory::TranscodeWorker);
        assert!(!check.is_critical());
    }

    #[test]
    fn test_deep_health_probe_mixed_statuses() {
        let mut db = DeepHealthCheck::new(DeepCheckConfig::new(CheckCategory::Database, "db"));
        db.set_state(true, Duration::from_millis(1), None);

        let mut storage = DeepHealthCheck::new(
            DeepCheckConfig::new(CheckCategory::Storage, "storage")
                .with_degraded_threshold(Duration::from_millis(10)),
        );
        storage.set_state(true, Duration::from_millis(50), None);

        let mut cdn = DeepHealthCheck::new(
            DeepCheckConfig::new(CheckCategory::ExternalService, "cdn").non_critical(),
        );
        cdn.set_state(false, Duration::from_millis(5), Some("timeout".to_string()));

        let probe = DeepHealthProbe::new("1.0.0")
            .add_check(db)
            .add_check(storage)
            .add_check(cdn);

        let resp = probe.run_all();
        // Critical storage degraded + non-critical CDN unhealthy → degraded overall
        assert_eq!(resp.overall_status, CheckStatus::Degraded);
        // Readiness: critical checks only → storage degraded → readiness degraded
        assert_eq!(resp.readiness_status, CheckStatus::Degraded);
        assert!(resp.is_ready()); // degraded is still passing
        assert_eq!(resp.unhealthy_checks().len(), 1); // only CDN
        assert_eq!(resp.degraded_checks().len(), 1); // only storage
    }
}
