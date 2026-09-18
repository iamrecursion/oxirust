#![allow(dead_code)]
//! Health probe system for monitoring automation subsystem vitals.
//!
//! Provides lightweight liveness, readiness, and performance probes for
//! each automation subsystem (playlist engine, device controllers, signal
//! router, failover manager, etc.). Aggregates individual probe results
//! into an overall system health status with severity levels.

use std::collections::HashMap;
use std::fmt;
use std::time::Duration;

/// Health status of a single probe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProbeStatus {
    /// Probe passed, subsystem is healthy.
    Healthy,
    /// Probe detected degraded performance.
    Degraded,
    /// Probe detected a failure.
    Unhealthy,
    /// Probe has not been evaluated yet.
    Unknown,
}

impl fmt::Display for ProbeStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Healthy => write!(f, "Healthy"),
            Self::Degraded => write!(f, "Degraded"),
            Self::Unhealthy => write!(f, "Unhealthy"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

impl ProbeStatus {
    /// Numeric severity (0 = healthy, higher = worse).
    pub fn severity(self) -> u8 {
        match self {
            Self::Healthy => 0,
            Self::Unknown => 1,
            Self::Degraded => 2,
            Self::Unhealthy => 3,
        }
    }

    /// Whether the status indicates the subsystem is operational.
    pub fn is_operational(self) -> bool {
        matches!(self, Self::Healthy | Self::Degraded)
    }
}

/// Type of health probe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProbeType {
    /// Liveness: is the subsystem alive and responsive?
    Liveness,
    /// Readiness: is the subsystem ready to accept work?
    Readiness,
    /// Performance: is the subsystem meeting latency/throughput targets?
    Performance,
}

impl fmt::Display for ProbeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Liveness => write!(f, "Liveness"),
            Self::Readiness => write!(f, "Readiness"),
            Self::Performance => write!(f, "Performance"),
        }
    }
}

/// Result from a single probe evaluation.
#[derive(Debug, Clone)]
pub struct ProbeResult {
    /// Name of the subsystem probed.
    pub subsystem: String,
    /// Type of probe.
    pub probe_type: ProbeType,
    /// Resulting status.
    pub status: ProbeStatus,
    /// Optional message with details.
    pub message: Option<String>,
    /// Response time of the probe itself.
    pub response_time: Duration,
    /// Timestamp of the probe (epoch millis).
    pub timestamp_ms: i64,
}

impl ProbeResult {
    /// Create a healthy probe result.
    pub fn healthy(subsystem: &str, probe_type: ProbeType, response_time: Duration) -> Self {
        Self {
            subsystem: subsystem.to_string(),
            probe_type,
            status: ProbeStatus::Healthy,
            message: None,
            response_time,
            timestamp_ms: 0,
        }
    }

    /// Create a degraded probe result with a message.
    pub fn degraded(
        subsystem: &str,
        probe_type: ProbeType,
        message: &str,
        response_time: Duration,
    ) -> Self {
        Self {
            subsystem: subsystem.to_string(),
            probe_type,
            status: ProbeStatus::Degraded,
            message: Some(message.to_string()),
            response_time,
            timestamp_ms: 0,
        }
    }

    /// Create an unhealthy probe result with a message.
    pub fn unhealthy(
        subsystem: &str,
        probe_type: ProbeType,
        message: &str,
        response_time: Duration,
    ) -> Self {
        Self {
            subsystem: subsystem.to_string(),
            probe_type,
            status: ProbeStatus::Unhealthy,
            message: Some(message.to_string()),
            response_time,
            timestamp_ms: 0,
        }
    }
}

/// Thresholds for performance probes.
#[derive(Debug, Clone)]
pub struct PerformanceThresholds {
    /// Maximum acceptable response time before "degraded".
    pub degraded_threshold: Duration,
    /// Maximum acceptable response time before "unhealthy".
    pub unhealthy_threshold: Duration,
    /// Maximum consecutive failures before marking unhealthy.
    pub max_consecutive_failures: u32,
}

impl Default for PerformanceThresholds {
    fn default() -> Self {
        Self {
            degraded_threshold: Duration::from_millis(100),
            unhealthy_threshold: Duration::from_millis(500),
            max_consecutive_failures: 3,
        }
    }
}

/// Configuration for a health probe.
#[derive(Debug, Clone)]
pub struct ProbeConfig {
    /// Subsystem name.
    pub subsystem: String,
    /// Probe type.
    pub probe_type: ProbeType,
    /// Interval between probe evaluations.
    pub interval: Duration,
    /// Timeout for probe evaluation.
    pub timeout: Duration,
    /// Performance thresholds (for performance probes).
    pub thresholds: PerformanceThresholds,
    /// Whether this probe is enabled.
    pub enabled: bool,
}

impl ProbeConfig {
    /// Create a default liveness probe config.
    pub fn liveness(subsystem: &str) -> Self {
        Self {
            subsystem: subsystem.to_string(),
            probe_type: ProbeType::Liveness,
            interval: Duration::from_secs(10),
            timeout: Duration::from_secs(5),
            thresholds: PerformanceThresholds::default(),
            enabled: true,
        }
    }

    /// Create a default readiness probe config.
    pub fn readiness(subsystem: &str) -> Self {
        Self {
            subsystem: subsystem.to_string(),
            probe_type: ProbeType::Readiness,
            interval: Duration::from_secs(5),
            timeout: Duration::from_secs(3),
            thresholds: PerformanceThresholds::default(),
            enabled: true,
        }
    }

    /// Create a default performance probe config.
    pub fn performance(subsystem: &str) -> Self {
        Self {
            subsystem: subsystem.to_string(),
            probe_type: ProbeType::Performance,
            interval: Duration::from_secs(30),
            timeout: Duration::from_secs(10),
            thresholds: PerformanceThresholds::default(),
            enabled: true,
        }
    }
}

/// Tracks the state of a single probe over time.
#[derive(Debug)]
pub struct ProbeTracker {
    /// Configuration for this probe.
    pub config: ProbeConfig,
    /// Latest result.
    pub latest: Option<ProbeResult>,
    /// Consecutive failure count.
    pub consecutive_failures: u32,
    /// Total evaluations.
    pub total_evaluations: u64,
    /// Total failures.
    pub total_failures: u64,
}

impl ProbeTracker {
    /// Create a new probe tracker.
    pub fn new(config: ProbeConfig) -> Self {
        Self {
            config,
            latest: None,
            consecutive_failures: 0,
            total_evaluations: 0,
            total_failures: 0,
        }
    }

    /// Record a probe result.
    pub fn record(&mut self, result: ProbeResult) {
        self.total_evaluations += 1;
        if result.status == ProbeStatus::Unhealthy {
            self.consecutive_failures += 1;
            self.total_failures += 1;
        } else {
            self.consecutive_failures = 0;
        }
        self.latest = Some(result);
    }

    /// Current status of this probe.
    pub fn status(&self) -> ProbeStatus {
        self.latest
            .as_ref()
            .map_or(ProbeStatus::Unknown, |r| r.status)
    }

    /// Failure rate (0.0-1.0).
    #[allow(clippy::cast_precision_loss)]
    pub fn failure_rate(&self) -> f64 {
        if self.total_evaluations == 0 {
            return 0.0;
        }
        self.total_failures as f64 / self.total_evaluations as f64
    }

    /// Whether the probe has exceeded the consecutive failure threshold.
    pub fn is_flapping(&self) -> bool {
        self.consecutive_failures >= self.config.thresholds.max_consecutive_failures
    }
}

/// Aggregate health report across all probes.
#[derive(Debug, Clone)]
pub struct HealthReport {
    /// Overall system status (worst of all probes).
    pub overall: ProbeStatus,
    /// Per-subsystem status summaries.
    pub subsystems: HashMap<String, ProbeStatus>,
    /// Total probes evaluated.
    pub total_probes: usize,
    /// Number of healthy probes.
    pub healthy_count: usize,
    /// Number of degraded probes.
    pub degraded_count: usize,
    /// Number of unhealthy probes.
    pub unhealthy_count: usize,
}

/// Health probe manager that aggregates probes across subsystems.
#[derive(Debug)]
pub struct HealthProbeManager {
    /// All registered probe trackers keyed by "subsystem::probe_type".
    probes: HashMap<String, ProbeTracker>,
}

impl HealthProbeManager {
    /// Create a new health probe manager.
    pub fn new() -> Self {
        Self {
            probes: HashMap::new(),
        }
    }

    /// Register a probe.
    pub fn register(&mut self, config: ProbeConfig) {
        let key = format!("{}::{}", config.subsystem, config.probe_type);
        self.probes.insert(key, ProbeTracker::new(config));
    }

    /// Record a probe result.
    pub fn record(&mut self, result: ProbeResult) {
        let key = format!("{}::{}", result.subsystem, result.probe_type);
        if let Some(tracker) = self.probes.get_mut(&key) {
            tracker.record(result);
        }
    }

    /// Get the status of a specific probe.
    pub fn probe_status(&self, subsystem: &str, probe_type: ProbeType) -> ProbeStatus {
        let key = format!("{subsystem}::{probe_type}");
        self.probes
            .get(&key)
            .map_or(ProbeStatus::Unknown, ProbeTracker::status)
    }

    /// Number of registered probes.
    pub fn probe_count(&self) -> usize {
        self.probes.len()
    }

    /// Generate a health report.
    pub fn report(&self) -> HealthReport {
        let mut overall = ProbeStatus::Healthy;
        let mut subsystems: HashMap<String, ProbeStatus> = HashMap::new();
        let mut healthy = 0usize;
        let mut degraded = 0usize;
        let mut unhealthy = 0usize;

        for tracker in self.probes.values() {
            let status = tracker.status();
            match status {
                ProbeStatus::Healthy => healthy += 1,
                ProbeStatus::Degraded => degraded += 1,
                ProbeStatus::Unhealthy => unhealthy += 1,
                ProbeStatus::Unknown => {}
            }
            if status.severity() > overall.severity() {
                overall = status;
            }
            let sub_entry = subsystems
                .entry(tracker.config.subsystem.clone())
                .or_insert(ProbeStatus::Healthy);
            if status.severity() > sub_entry.severity() {
                *sub_entry = status;
            }
        }

        HealthReport {
            overall,
            subsystems,
            total_probes: self.probes.len(),
            healthy_count: healthy,
            degraded_count: degraded,
            unhealthy_count: unhealthy,
        }
    }

    /// List all subsystems that are unhealthy.
    pub fn unhealthy_subsystems(&self) -> Vec<String> {
        let report = self.report();
        report
            .subsystems
            .iter()
            .filter(|(_, &status)| status == ProbeStatus::Unhealthy)
            .map(|(name, _)| name.clone())
            .collect()
    }

    /// Check if all probes are healthy.
    pub fn all_healthy(&self) -> bool {
        self.probes
            .values()
            .all(|t| t.status() == ProbeStatus::Healthy || t.status() == ProbeStatus::Unknown)
    }
}

impl Default for HealthProbeManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_probe_status_severity_ordering() {
        assert!(ProbeStatus::Unhealthy.severity() > ProbeStatus::Degraded.severity());
        assert!(ProbeStatus::Degraded.severity() > ProbeStatus::Unknown.severity());
        assert!(ProbeStatus::Unknown.severity() > ProbeStatus::Healthy.severity());
    }

    #[test]
    fn test_probe_status_display() {
        assert_eq!(ProbeStatus::Healthy.to_string(), "Healthy");
        assert_eq!(ProbeStatus::Unhealthy.to_string(), "Unhealthy");
    }

    #[test]
    fn test_probe_status_is_operational() {
        assert!(ProbeStatus::Healthy.is_operational());
        assert!(ProbeStatus::Degraded.is_operational());
        assert!(!ProbeStatus::Unhealthy.is_operational());
        assert!(!ProbeStatus::Unknown.is_operational());
    }

    #[test]
    fn test_probe_type_display() {
        assert_eq!(ProbeType::Liveness.to_string(), "Liveness");
        assert_eq!(ProbeType::Readiness.to_string(), "Readiness");
        assert_eq!(ProbeType::Performance.to_string(), "Performance");
    }

    #[test]
    fn test_probe_result_constructors() {
        let h = ProbeResult::healthy("playlist", ProbeType::Liveness, Duration::from_millis(5));
        assert_eq!(h.status, ProbeStatus::Healthy);
        assert!(h.message.is_none());

        let d = ProbeResult::degraded(
            "router",
            ProbeType::Performance,
            "slow",
            Duration::from_millis(150),
        );
        assert_eq!(d.status, ProbeStatus::Degraded);
        assert_eq!(d.message.as_deref(), Some("slow"));

        let u = ProbeResult::unhealthy(
            "device",
            ProbeType::Readiness,
            "offline",
            Duration::from_millis(0),
        );
        assert_eq!(u.status, ProbeStatus::Unhealthy);
    }

    #[test]
    fn test_probe_tracker_record_healthy() {
        let config = ProbeConfig::liveness("engine");
        let mut tracker = ProbeTracker::new(config);
        assert_eq!(tracker.status(), ProbeStatus::Unknown);

        tracker.record(ProbeResult::healthy(
            "engine",
            ProbeType::Liveness,
            Duration::from_millis(2),
        ));
        assert_eq!(tracker.status(), ProbeStatus::Healthy);
        assert_eq!(tracker.total_evaluations, 1);
        assert_eq!(tracker.consecutive_failures, 0);
    }

    #[test]
    fn test_probe_tracker_consecutive_failures() {
        let config = ProbeConfig::liveness("engine");
        let mut tracker = ProbeTracker::new(config);

        for _ in 0..3 {
            tracker.record(ProbeResult::unhealthy(
                "engine",
                ProbeType::Liveness,
                "down",
                Duration::from_millis(0),
            ));
        }
        assert_eq!(tracker.consecutive_failures, 3);
        assert!(tracker.is_flapping());
    }

    #[test]
    fn test_probe_tracker_failure_rate() {
        let config = ProbeConfig::liveness("test");
        let mut tracker = ProbeTracker::new(config);
        tracker.record(ProbeResult::healthy(
            "test",
            ProbeType::Liveness,
            Duration::from_millis(1),
        ));
        tracker.record(ProbeResult::unhealthy(
            "test",
            ProbeType::Liveness,
            "fail",
            Duration::from_millis(1),
        ));
        let rate = tracker.failure_rate();
        assert!((rate - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_health_probe_manager_register_and_status() {
        let mut mgr = HealthProbeManager::new();
        mgr.register(ProbeConfig::liveness("playlist"));
        assert_eq!(mgr.probe_count(), 1);
        assert_eq!(
            mgr.probe_status("playlist", ProbeType::Liveness),
            ProbeStatus::Unknown
        );
    }

    #[test]
    fn test_health_probe_manager_record_and_report() {
        let mut mgr = HealthProbeManager::new();
        mgr.register(ProbeConfig::liveness("playlist"));
        mgr.register(ProbeConfig::readiness("router"));

        mgr.record(ProbeResult::healthy(
            "playlist",
            ProbeType::Liveness,
            Duration::from_millis(2),
        ));
        mgr.record(ProbeResult::degraded(
            "router",
            ProbeType::Readiness,
            "slow",
            Duration::from_millis(120),
        ));

        let report = mgr.report();
        assert_eq!(report.overall, ProbeStatus::Degraded);
        assert_eq!(report.healthy_count, 1);
        assert_eq!(report.degraded_count, 1);
    }

    #[test]
    fn test_health_probe_manager_all_healthy() {
        let mut mgr = HealthProbeManager::new();
        mgr.register(ProbeConfig::liveness("a"));
        mgr.register(ProbeConfig::liveness("b"));

        mgr.record(ProbeResult::healthy(
            "a",
            ProbeType::Liveness,
            Duration::from_millis(1),
        ));
        mgr.record(ProbeResult::healthy(
            "b",
            ProbeType::Liveness,
            Duration::from_millis(1),
        ));
        assert!(mgr.all_healthy());
    }

    #[test]
    fn test_health_probe_manager_unhealthy_subsystems() {
        let mut mgr = HealthProbeManager::new();
        mgr.register(ProbeConfig::liveness("good"));
        mgr.register(ProbeConfig::liveness("bad"));

        mgr.record(ProbeResult::healthy(
            "good",
            ProbeType::Liveness,
            Duration::from_millis(1),
        ));
        mgr.record(ProbeResult::unhealthy(
            "bad",
            ProbeType::Liveness,
            "down",
            Duration::from_millis(0),
        ));

        let bad = mgr.unhealthy_subsystems();
        assert_eq!(bad.len(), 1);
        assert_eq!(bad[0], "bad");
    }

    #[test]
    fn test_performance_thresholds_default() {
        let t = PerformanceThresholds::default();
        assert_eq!(t.degraded_threshold, Duration::from_millis(100));
        assert_eq!(t.unhealthy_threshold, Duration::from_millis(500));
        assert_eq!(t.max_consecutive_failures, 3);
    }
}
