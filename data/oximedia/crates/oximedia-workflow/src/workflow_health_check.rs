//! Workflow engine health check module.
//!
//! Provides periodic health validation for the workflow engine, checking
//! queue depth, stuck tasks, resource pool saturation, and general engine
//! responsiveness. Produces structured health reports suitable for
//! monitoring dashboards and alerting systems.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Overall health status of the engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    /// All checks pass; engine is operating normally.
    Healthy,
    /// Some checks indicate degraded performance but engine is functional.
    Degraded,
    /// Critical issues detected; engine may not be functioning correctly.
    Unhealthy,
}

impl HealthStatus {
    /// Combine two statuses, taking the worse one.
    #[must_use]
    pub fn combine(self, other: Self) -> Self {
        match (self, other) {
            (Self::Unhealthy, _) | (_, Self::Unhealthy) => Self::Unhealthy,
            (Self::Degraded, _) | (_, Self::Degraded) => Self::Degraded,
            _ => Self::Healthy,
        }
    }
}

impl std::fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Healthy => write!(f, "healthy"),
            Self::Degraded => write!(f, "degraded"),
            Self::Unhealthy => write!(f, "unhealthy"),
        }
    }
}

/// Result of an individual health check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckResult {
    /// Name of the check.
    pub name: String,
    /// Status.
    pub status: HealthStatus,
    /// Human-readable message.
    pub message: String,
    /// Numeric value associated with the check (for thresholds).
    pub value: Option<f64>,
    /// Duration the check took to run.
    pub duration: Duration,
}

impl CheckResult {
    /// Create a healthy check result.
    #[must_use]
    pub fn healthy(name: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status: HealthStatus::Healthy,
            message: message.into(),
            value: None,
            duration: Duration::ZERO,
        }
    }

    /// Create a degraded check result.
    #[must_use]
    pub fn degraded(name: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status: HealthStatus::Degraded,
            message: message.into(),
            value: None,
            duration: Duration::ZERO,
        }
    }

    /// Create an unhealthy check result.
    #[must_use]
    pub fn unhealthy(name: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status: HealthStatus::Unhealthy,
            message: message.into(),
            value: None,
            duration: Duration::ZERO,
        }
    }

    /// Set numeric value.
    #[must_use]
    pub fn with_value(mut self, value: f64) -> Self {
        self.value = Some(value);
        self
    }

    /// Set duration.
    #[must_use]
    pub fn with_duration(mut self, duration: Duration) -> Self {
        self.duration = duration;
        self
    }
}

/// Complete health report for the workflow engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthReport {
    /// Overall engine status (worst of all checks).
    pub status: HealthStatus,
    /// Individual check results.
    pub checks: Vec<CheckResult>,
    /// When the report was generated (millis since epoch).
    pub generated_at_ms: u64,
    /// Total time to run all checks.
    pub total_duration: Duration,
    /// Engine version or build info.
    pub engine_info: HashMap<String, String>,
}

impl HealthReport {
    /// Count checks by status.
    #[must_use]
    pub fn count_by_status(&self) -> (usize, usize, usize) {
        let healthy = self
            .checks
            .iter()
            .filter(|c| c.status == HealthStatus::Healthy)
            .count();
        let degraded = self
            .checks
            .iter()
            .filter(|c| c.status == HealthStatus::Degraded)
            .count();
        let unhealthy = self
            .checks
            .iter()
            .filter(|c| c.status == HealthStatus::Unhealthy)
            .count();
        (healthy, degraded, unhealthy)
    }

    /// Get all failing checks.
    #[must_use]
    pub fn failing_checks(&self) -> Vec<&CheckResult> {
        self.checks
            .iter()
            .filter(|c| c.status != HealthStatus::Healthy)
            .collect()
    }
}

/// Thresholds for health checks.
#[derive(Debug, Clone)]
pub struct HealthThresholds {
    /// Queue depth above which status becomes Degraded.
    pub queue_depth_warn: usize,
    /// Queue depth above which status becomes Unhealthy.
    pub queue_depth_critical: usize,
    /// Number of stuck tasks before Degraded.
    pub stuck_tasks_warn: usize,
    /// Number of stuck tasks before Unhealthy.
    pub stuck_tasks_critical: usize,
    /// Resource pool utilisation (0.0..1.0) above which Degraded.
    pub resource_util_warn: f64,
    /// Resource pool utilisation above which Unhealthy.
    pub resource_util_critical: f64,
    /// Failed workflow rate (0.0..1.0) above which Degraded.
    pub failure_rate_warn: f64,
    /// Failed workflow rate above which Unhealthy.
    pub failure_rate_critical: f64,
    /// Task age in seconds before it's considered stuck.
    pub stuck_task_age_secs: u64,
    /// Maximum active workflows before Degraded.
    pub active_workflows_warn: usize,
    /// Maximum active workflows before Unhealthy.
    pub active_workflows_critical: usize,
}

impl Default for HealthThresholds {
    fn default() -> Self {
        Self {
            queue_depth_warn: 100,
            queue_depth_critical: 1000,
            stuck_tasks_warn: 5,
            stuck_tasks_critical: 20,
            resource_util_warn: 0.8,
            resource_util_critical: 0.95,
            failure_rate_warn: 0.1,
            failure_rate_critical: 0.5,
            stuck_task_age_secs: 3600,
            active_workflows_warn: 50,
            active_workflows_critical: 200,
        }
    }
}

/// Input metrics gathered from the engine for health evaluation.
#[derive(Debug, Clone, Default)]
pub struct EngineMetrics {
    /// Current queue depth (number of pending tasks).
    pub queue_depth: usize,
    /// Number of tasks that have been running longer than the stuck threshold.
    pub stuck_tasks: usize,
    /// Resource pool average utilisation (0.0..1.0).
    pub resource_utilisation: f64,
    /// Number of active workflows.
    pub active_workflows: usize,
    /// Total workflows (active + completed + failed).
    pub total_workflows: u64,
    /// Total failed workflows.
    pub failed_workflows: u64,
    /// Whether the persistence layer is responsive.
    pub persistence_ok: bool,
    /// Whether the scheduler is running.
    pub scheduler_running: bool,
    /// Number of registered resource types.
    pub resource_pool_count: usize,
    /// Custom metrics.
    pub custom: HashMap<String, f64>,
}

/// The health checker engine.
/// The health checker engine.
///
/// Contains closures for custom checks, so `Debug` is implemented manually.
pub struct HealthChecker {
    thresholds: HealthThresholds,
    /// Custom check functions (name -> check fn).
    custom_checks: Vec<(
        String,
        Box<dyn Fn(&EngineMetrics) -> CheckResult + Send + Sync>,
    )>,
    /// Engine info to include in reports.
    engine_info: HashMap<String, String>,
}

impl std::fmt::Debug for HealthChecker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HealthChecker")
            .field("thresholds", &self.thresholds)
            .field("custom_checks_count", &self.custom_checks.len())
            .field("engine_info", &self.engine_info)
            .finish()
    }
}

impl HealthChecker {
    /// Create a health checker with default thresholds.
    #[must_use]
    pub fn new() -> Self {
        Self {
            thresholds: HealthThresholds::default(),
            custom_checks: Vec::new(),
            engine_info: HashMap::new(),
        }
    }

    /// Create with custom thresholds.
    #[must_use]
    pub fn with_thresholds(thresholds: HealthThresholds) -> Self {
        Self {
            thresholds,
            custom_checks: Vec::new(),
            engine_info: HashMap::new(),
        }
    }

    /// Set engine info metadata.
    pub fn set_engine_info(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.engine_info.insert(key.into(), value.into());
    }

    /// Register a custom health check.
    pub fn add_custom_check(
        &mut self,
        name: impl Into<String>,
        check: impl Fn(&EngineMetrics) -> CheckResult + Send + Sync + 'static,
    ) {
        self.custom_checks.push((name.into(), Box::new(check)));
    }

    /// Run all health checks and produce a report.
    #[must_use]
    pub fn check(&self, metrics: &EngineMetrics, now_ms: u64) -> HealthReport {
        let start = std::time::Instant::now();
        let mut checks = Vec::new();

        checks.push(self.check_queue_depth(metrics));
        checks.push(self.check_stuck_tasks(metrics));
        checks.push(self.check_resource_utilisation(metrics));
        checks.push(self.check_failure_rate(metrics));
        checks.push(self.check_persistence(metrics));
        checks.push(self.check_scheduler(metrics));
        checks.push(self.check_active_workflows(metrics));

        // Run custom checks
        for (name, check_fn) in &self.custom_checks {
            let mut result = check_fn(metrics);
            if result.name.is_empty() {
                result.name = name.clone();
            }
            checks.push(result);
        }

        let overall = checks
            .iter()
            .fold(HealthStatus::Healthy, |acc, c| acc.combine(c.status));

        HealthReport {
            status: overall,
            checks,
            generated_at_ms: now_ms,
            total_duration: start.elapsed(),
            engine_info: self.engine_info.clone(),
        }
    }

    fn check_queue_depth(&self, metrics: &EngineMetrics) -> CheckResult {
        let depth = metrics.queue_depth;
        if depth >= self.thresholds.queue_depth_critical {
            CheckResult::unhealthy(
                "queue_depth",
                format!("queue depth {depth} exceeds critical threshold"),
            )
            .with_value(depth as f64)
        } else if depth >= self.thresholds.queue_depth_warn {
            CheckResult::degraded(
                "queue_depth",
                format!("queue depth {depth} exceeds warning threshold"),
            )
            .with_value(depth as f64)
        } else {
            CheckResult::healthy("queue_depth", format!("queue depth {depth} is normal"))
                .with_value(depth as f64)
        }
    }

    fn check_stuck_tasks(&self, metrics: &EngineMetrics) -> CheckResult {
        let stuck = metrics.stuck_tasks;
        if stuck >= self.thresholds.stuck_tasks_critical {
            CheckResult::unhealthy(
                "stuck_tasks",
                format!("{stuck} stuck tasks exceeds critical threshold"),
            )
            .with_value(stuck as f64)
        } else if stuck >= self.thresholds.stuck_tasks_warn {
            CheckResult::degraded(
                "stuck_tasks",
                format!("{stuck} stuck tasks exceeds warning threshold"),
            )
            .with_value(stuck as f64)
        } else {
            CheckResult::healthy("stuck_tasks", format!("{stuck} stuck tasks"))
                .with_value(stuck as f64)
        }
    }

    #[allow(clippy::cast_precision_loss)]
    fn check_resource_utilisation(&self, metrics: &EngineMetrics) -> CheckResult {
        let util = metrics.resource_utilisation;
        if util >= self.thresholds.resource_util_critical {
            CheckResult::unhealthy(
                "resource_utilisation",
                format!("resource utilisation {util:.1}% exceeds critical threshold"),
            )
            .with_value(util)
        } else if util >= self.thresholds.resource_util_warn {
            CheckResult::degraded(
                "resource_utilisation",
                format!("resource utilisation {util:.1}% exceeds warning threshold"),
            )
            .with_value(util)
        } else {
            CheckResult::healthy(
                "resource_utilisation",
                format!("resource utilisation {util:.1}% is normal"),
            )
            .with_value(util)
        }
    }

    #[allow(clippy::cast_precision_loss)]
    fn check_failure_rate(&self, metrics: &EngineMetrics) -> CheckResult {
        if metrics.total_workflows == 0 {
            return CheckResult::healthy("failure_rate", "no workflows processed yet")
                .with_value(0.0);
        }

        let rate = metrics.failed_workflows as f64 / metrics.total_workflows as f64;

        if rate >= self.thresholds.failure_rate_critical {
            CheckResult::unhealthy(
                "failure_rate",
                format!(
                    "failure rate {:.1}% exceeds critical threshold",
                    rate * 100.0
                ),
            )
            .with_value(rate)
        } else if rate >= self.thresholds.failure_rate_warn {
            CheckResult::degraded(
                "failure_rate",
                format!(
                    "failure rate {:.1}% exceeds warning threshold",
                    rate * 100.0
                ),
            )
            .with_value(rate)
        } else {
            CheckResult::healthy(
                "failure_rate",
                format!("failure rate {:.1}% is normal", rate * 100.0),
            )
            .with_value(rate)
        }
    }

    fn check_persistence(&self, metrics: &EngineMetrics) -> CheckResult {
        if metrics.persistence_ok {
            CheckResult::healthy("persistence", "persistence layer is responsive")
        } else {
            CheckResult::unhealthy("persistence", "persistence layer is not responding")
        }
    }

    fn check_scheduler(&self, metrics: &EngineMetrics) -> CheckResult {
        if metrics.scheduler_running {
            CheckResult::healthy("scheduler", "scheduler is running")
        } else {
            CheckResult::degraded("scheduler", "scheduler is not running")
        }
    }

    fn check_active_workflows(&self, metrics: &EngineMetrics) -> CheckResult {
        let active = metrics.active_workflows;
        if active >= self.thresholds.active_workflows_critical {
            CheckResult::unhealthy(
                "active_workflows",
                format!("{active} active workflows exceeds critical threshold"),
            )
            .with_value(active as f64)
        } else if active >= self.thresholds.active_workflows_warn {
            CheckResult::degraded(
                "active_workflows",
                format!("{active} active workflows exceeds warning threshold"),
            )
            .with_value(active as f64)
        } else {
            CheckResult::healthy("active_workflows", format!("{active} active workflows"))
                .with_value(active as f64)
        }
    }
}

impl Default for HealthChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// Format a health report as a human-readable string.
#[must_use]
pub fn format_health_report(report: &HealthReport) -> String {
    let mut lines = Vec::new();
    lines.push(format!("Engine Health: {}", report.status));
    lines.push(format!(
        "Generated: {} ms, took {:?}",
        report.generated_at_ms, report.total_duration
    ));

    let (healthy, degraded, unhealthy) = report.count_by_status();
    lines.push(format!(
        "Checks: {healthy} healthy, {degraded} degraded, {unhealthy} unhealthy"
    ));

    for check in &report.checks {
        let value_str = check
            .value
            .map(|v| format!(" (value: {v:.2})"))
            .unwrap_or_default();
        lines.push(format!(
            "  [{}] {}: {}{}",
            check.status, check.name, check.message, value_str
        ));
    }

    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_metrics() -> EngineMetrics {
        EngineMetrics {
            queue_depth: 10,
            stuck_tasks: 0,
            resource_utilisation: 0.3,
            active_workflows: 5,
            total_workflows: 100,
            failed_workflows: 2,
            persistence_ok: true,
            scheduler_running: true,
            resource_pool_count: 3,
            custom: HashMap::new(),
        }
    }

    // --- HealthStatus ---

    #[test]
    fn test_health_status_combine() {
        assert_eq!(
            HealthStatus::Healthy.combine(HealthStatus::Healthy),
            HealthStatus::Healthy
        );
        assert_eq!(
            HealthStatus::Healthy.combine(HealthStatus::Degraded),
            HealthStatus::Degraded
        );
        assert_eq!(
            HealthStatus::Degraded.combine(HealthStatus::Unhealthy),
            HealthStatus::Unhealthy
        );
        assert_eq!(
            HealthStatus::Unhealthy.combine(HealthStatus::Healthy),
            HealthStatus::Unhealthy
        );
    }

    #[test]
    fn test_health_status_display() {
        assert_eq!(HealthStatus::Healthy.to_string(), "healthy");
        assert_eq!(HealthStatus::Degraded.to_string(), "degraded");
        assert_eq!(HealthStatus::Unhealthy.to_string(), "unhealthy");
    }

    // --- CheckResult ---

    #[test]
    fn test_check_result_constructors() {
        let h = CheckResult::healthy("test", "ok");
        assert_eq!(h.status, HealthStatus::Healthy);

        let d = CheckResult::degraded("test", "warn");
        assert_eq!(d.status, HealthStatus::Degraded);

        let u = CheckResult::unhealthy("test", "fail");
        assert_eq!(u.status, HealthStatus::Unhealthy);
    }

    #[test]
    fn test_check_result_with_value() {
        let r = CheckResult::healthy("test", "ok").with_value(42.0);
        assert_eq!(r.value, Some(42.0));
    }

    // --- HealthChecker all healthy ---

    #[test]
    fn test_all_healthy() {
        let checker = HealthChecker::new();
        let metrics = default_metrics();
        let report = checker.check(&metrics, 1000);

        assert_eq!(report.status, HealthStatus::Healthy);
        let (healthy, degraded, unhealthy) = report.count_by_status();
        assert!(healthy >= 7); // 7 built-in checks
        assert_eq!(degraded, 0);
        assert_eq!(unhealthy, 0);
    }

    // --- Queue depth ---

    #[test]
    fn test_queue_depth_degraded() {
        let checker = HealthChecker::new();
        let mut metrics = default_metrics();
        metrics.queue_depth = 150;

        let report = checker.check(&metrics, 1000);
        let qd = report
            .checks
            .iter()
            .find(|c| c.name == "queue_depth")
            .expect("find");
        assert_eq!(qd.status, HealthStatus::Degraded);
    }

    #[test]
    fn test_queue_depth_unhealthy() {
        let checker = HealthChecker::new();
        let mut metrics = default_metrics();
        metrics.queue_depth = 1500;

        let report = checker.check(&metrics, 1000);
        let qd = report
            .checks
            .iter()
            .find(|c| c.name == "queue_depth")
            .expect("find");
        assert_eq!(qd.status, HealthStatus::Unhealthy);
    }

    // --- Stuck tasks ---

    #[test]
    fn test_stuck_tasks_degraded() {
        let checker = HealthChecker::new();
        let mut metrics = default_metrics();
        metrics.stuck_tasks = 10;

        let report = checker.check(&metrics, 1000);
        let st = report
            .checks
            .iter()
            .find(|c| c.name == "stuck_tasks")
            .expect("find");
        assert_eq!(st.status, HealthStatus::Degraded);
    }

    #[test]
    fn test_stuck_tasks_unhealthy() {
        let checker = HealthChecker::new();
        let mut metrics = default_metrics();
        metrics.stuck_tasks = 25;

        let report = checker.check(&metrics, 1000);
        let st = report
            .checks
            .iter()
            .find(|c| c.name == "stuck_tasks")
            .expect("find");
        assert_eq!(st.status, HealthStatus::Unhealthy);
    }

    // --- Resource utilisation ---

    #[test]
    fn test_resource_util_degraded() {
        let checker = HealthChecker::new();
        let mut metrics = default_metrics();
        metrics.resource_utilisation = 0.85;

        let report = checker.check(&metrics, 1000);
        let ru = report
            .checks
            .iter()
            .find(|c| c.name == "resource_utilisation")
            .expect("find");
        assert_eq!(ru.status, HealthStatus::Degraded);
    }

    #[test]
    fn test_resource_util_unhealthy() {
        let checker = HealthChecker::new();
        let mut metrics = default_metrics();
        metrics.resource_utilisation = 0.98;

        let report = checker.check(&metrics, 1000);
        let ru = report
            .checks
            .iter()
            .find(|c| c.name == "resource_utilisation")
            .expect("find");
        assert_eq!(ru.status, HealthStatus::Unhealthy);
    }

    // --- Failure rate ---

    #[test]
    fn test_failure_rate_degraded() {
        let checker = HealthChecker::new();
        let mut metrics = default_metrics();
        metrics.failed_workflows = 15;
        metrics.total_workflows = 100;

        let report = checker.check(&metrics, 1000);
        let fr = report
            .checks
            .iter()
            .find(|c| c.name == "failure_rate")
            .expect("find");
        assert_eq!(fr.status, HealthStatus::Degraded);
    }

    #[test]
    fn test_failure_rate_zero_workflows() {
        let checker = HealthChecker::new();
        let mut metrics = default_metrics();
        metrics.total_workflows = 0;
        metrics.failed_workflows = 0;

        let report = checker.check(&metrics, 1000);
        let fr = report
            .checks
            .iter()
            .find(|c| c.name == "failure_rate")
            .expect("find");
        assert_eq!(fr.status, HealthStatus::Healthy);
    }

    // --- Persistence ---

    #[test]
    fn test_persistence_unhealthy() {
        let checker = HealthChecker::new();
        let mut metrics = default_metrics();
        metrics.persistence_ok = false;

        let report = checker.check(&metrics, 1000);
        let p = report
            .checks
            .iter()
            .find(|c| c.name == "persistence")
            .expect("find");
        assert_eq!(p.status, HealthStatus::Unhealthy);
    }

    // --- Scheduler ---

    #[test]
    fn test_scheduler_degraded() {
        let checker = HealthChecker::new();
        let mut metrics = default_metrics();
        metrics.scheduler_running = false;

        let report = checker.check(&metrics, 1000);
        let s = report
            .checks
            .iter()
            .find(|c| c.name == "scheduler")
            .expect("find");
        assert_eq!(s.status, HealthStatus::Degraded);
    }

    // --- Active workflows ---

    #[test]
    fn test_active_workflows_degraded() {
        let checker = HealthChecker::new();
        let mut metrics = default_metrics();
        metrics.active_workflows = 60;

        let report = checker.check(&metrics, 1000);
        let aw = report
            .checks
            .iter()
            .find(|c| c.name == "active_workflows")
            .expect("find");
        assert_eq!(aw.status, HealthStatus::Degraded);
    }

    // --- Custom check ---

    #[test]
    fn test_custom_check() {
        let mut checker = HealthChecker::new();
        checker.add_custom_check("disk_space", |_metrics| {
            CheckResult::degraded("disk_space", "disk usage above 80%")
        });

        let metrics = default_metrics();
        let report = checker.check(&metrics, 1000);
        let ds = report
            .checks
            .iter()
            .find(|c| c.name == "disk_space")
            .expect("find");
        assert_eq!(ds.status, HealthStatus::Degraded);
    }

    // --- Custom thresholds ---

    #[test]
    fn test_custom_thresholds() {
        let thresholds = HealthThresholds {
            queue_depth_warn: 5,
            queue_depth_critical: 10,
            ..Default::default()
        };
        let checker = HealthChecker::with_thresholds(thresholds);
        let mut metrics = default_metrics();
        metrics.queue_depth = 7;

        let report = checker.check(&metrics, 1000);
        let qd = report
            .checks
            .iter()
            .find(|c| c.name == "queue_depth")
            .expect("find");
        assert_eq!(qd.status, HealthStatus::Degraded);
    }

    // --- Overall status ---

    #[test]
    fn test_overall_status_worst_wins() {
        let checker = HealthChecker::new();
        let mut metrics = default_metrics();
        metrics.persistence_ok = false; // unhealthy

        let report = checker.check(&metrics, 1000);
        assert_eq!(report.status, HealthStatus::Unhealthy);
    }

    // --- Report utilities ---

    #[test]
    fn test_failing_checks() {
        let checker = HealthChecker::new();
        let mut metrics = default_metrics();
        metrics.queue_depth = 200;

        let report = checker.check(&metrics, 1000);
        let failing = report.failing_checks();
        assert!(!failing.is_empty());
        assert!(failing.iter().any(|c| c.name == "queue_depth"));
    }

    #[test]
    fn test_format_health_report() {
        let checker = HealthChecker::new();
        let metrics = default_metrics();
        let report = checker.check(&metrics, 1000);
        let text = format_health_report(&report);

        assert!(text.contains("Engine Health: healthy"));
        assert!(text.contains("queue_depth"));
    }

    // --- Engine info ---

    #[test]
    fn test_engine_info() {
        let mut checker = HealthChecker::new();
        checker.set_engine_info("version", "0.1.2");

        let metrics = default_metrics();
        let report = checker.check(&metrics, 1000);
        assert_eq!(
            report.engine_info.get("version"),
            Some(&"0.1.2".to_string())
        );
    }

    #[test]
    fn test_count_by_status() {
        let checker = HealthChecker::new();
        let metrics = default_metrics();
        let report = checker.check(&metrics, 1000);
        let (h, d, u) = report.count_by_status();
        assert_eq!(h + d + u, report.checks.len());
    }
}
