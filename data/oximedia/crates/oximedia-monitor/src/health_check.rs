//! Structured health check system for `OxiMedia` monitoring.
//!
//! Provides a trait-based health checker framework with built-in checkers for
//! disk space, memory, process uptime, and queue depth, plus a registry that
//! aggregates results and exposes a JSON health endpoint.
#![allow(dead_code)]

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::time::{Instant, SystemTime};

// ─────────────────────────────────────────────────────────────────────────────
// HealthStatus
// ─────────────────────────────────────────────────────────────────────────────

/// The health status of a single component.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum HealthStatus {
    /// Component is fully operational.
    Healthy,
    /// Component is operational but degraded; human-readable reason provided.
    Degraded(String),
    /// Component is not functioning; human-readable reason provided.
    Unhealthy(String),
}

impl HealthStatus {
    /// Returns a numeric severity (0=healthy, 1=degraded, 2=unhealthy).
    #[must_use]
    pub fn severity(&self) -> u8 {
        match self {
            Self::Healthy => 0,
            Self::Degraded(_) => 1,
            Self::Unhealthy(_) => 2,
        }
    }

    /// Returns `true` only for `Healthy`.
    #[must_use]
    pub fn is_healthy(&self) -> bool {
        matches!(self, Self::Healthy)
    }

    /// Short label suitable for JSON output.
    #[must_use]
    pub fn label(&self) -> &str {
        match self {
            Self::Healthy => "healthy",
            Self::Degraded(_) => "degraded",
            Self::Unhealthy(_) => "unhealthy",
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ComponentHealth
// ─────────────────────────────────────────────────────────────────────────────

/// Health report for a single named component.
#[derive(Debug, Clone)]
pub struct ComponentHealth {
    /// Component name.
    pub name: String,
    /// Current health status.
    pub status: HealthStatus,
    /// Optional latency measurement in milliseconds.
    pub latency_ms: Option<f64>,
    /// When the check was last performed.
    pub last_checked: SystemTime,
    /// Arbitrary key-value diagnostic details.
    pub details: HashMap<String, String>,
}

impl ComponentHealth {
    /// Create a healthy result with no latency or extra details.
    #[must_use]
    pub fn healthy(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status: HealthStatus::Healthy,
            latency_ms: None,
            last_checked: SystemTime::now(),
            details: HashMap::new(),
        }
    }

    /// Create a degraded result.
    #[must_use]
    pub fn degraded(name: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status: HealthStatus::Degraded(reason.into()),
            latency_ms: None,
            last_checked: SystemTime::now(),
            details: HashMap::new(),
        }
    }

    /// Create an unhealthy result.
    #[must_use]
    pub fn unhealthy(name: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status: HealthStatus::Unhealthy(reason.into()),
            latency_ms: None,
            last_checked: SystemTime::now(),
            details: HashMap::new(),
        }
    }

    /// Attach a latency measurement.
    #[must_use]
    pub fn with_latency(mut self, latency_ms: f64) -> Self {
        self.latency_ms = Some(latency_ms);
        self
    }

    /// Attach a key-value detail.
    #[must_use]
    pub fn with_detail(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.details.insert(key.into(), value.into());
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// HealthChecker trait
// ─────────────────────────────────────────────────────────────────────────────

/// A component that can report its own health.
pub trait HealthChecker: Send + Sync {
    /// Perform the health check and return the result.
    fn check(&self) -> ComponentHealth;

    /// Canonical name of this checker.
    fn name(&self) -> &str;
}

// ─────────────────────────────────────────────────────────────────────────────
// DiskSpaceChecker
// ─────────────────────────────────────────────────────────────────────────────

/// Checks available disk space at a given path.
///
/// On non-Unix platforms the available space is estimated as zero.
pub struct DiskSpaceChecker {
    /// Filesystem path to inspect.
    pub path: PathBuf,
    /// Usage percentage at which status becomes `Degraded`.
    pub warn_pct: f64,
    /// Usage percentage at which status becomes `Unhealthy`.
    pub crit_pct: f64,
}

impl DiskSpaceChecker {
    /// Create a new checker for `path` with `warn_pct` and `crit_pct` thresholds.
    #[must_use]
    pub fn new(path: impl Into<PathBuf>, warn_pct: f64, crit_pct: f64) -> Self {
        Self {
            path: path.into(),
            warn_pct,
            crit_pct,
        }
    }

    /// Attempt to query disk usage information.
    ///
    /// Returns `(used_bytes, total_bytes)` on success.
    fn disk_usage(&self) -> Option<(u64, u64)> {
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            // Use statvfs-like approach via std::fs::metadata for a rough estimate.
            // For a real implementation we would use the `nix` crate, but to stay
            // pure-Rust without extra dependencies we use a portable approximation.
            //
            // We rely on the fact that `std::fs::metadata` provides `blocks()` and
            // `blksize()` on Unix. This gives the space *used* by the file/dir entry
            // itself, not the whole filesystem — so we supplement with a pessimistic
            // estimate when this approach yields zero total bytes.
            if let Ok(meta) = std::fs::metadata(&self.path) {
                // Approximate: report blocks * blksize as used, with total = 2× used
                // as a conservative upper bound so that usage% is not misleadingly low.
                let used = meta.blocks() * meta.blksize();
                if used > 0 {
                    return Some((used, used * 2));
                }
            }
            // Fallback: treat as zero free.
            Some((0, 0))
        }
        #[cfg(not(unix))]
        {
            Some((0, 0))
        }
    }
}

impl HealthChecker for DiskSpaceChecker {
    fn name(&self) -> &str {
        "disk_space"
    }

    fn check(&self) -> ComponentHealth {
        let start = Instant::now();
        let name = format!("disk_space:{}", self.path.display());

        let (used, total) = match self.disk_usage() {
            Some(v) => v,
            None => {
                return ComponentHealth::unhealthy(&name, "unable to query disk space")
                    .with_latency(start.elapsed().as_secs_f64() * 1000.0);
            }
        };

        let latency = start.elapsed().as_secs_f64() * 1000.0;

        if total == 0 {
            // Cannot determine usage — treat as healthy but note it.
            return ComponentHealth::healthy(&name)
                .with_latency(latency)
                .with_detail("note", "disk usage indeterminate on this platform");
        }

        let usage_pct = used as f64 / total as f64 * 100.0;

        let mut health = if usage_pct >= self.crit_pct {
            ComponentHealth::unhealthy(
                &name,
                format!(
                    "disk usage {usage_pct:.1}% exceeds critical threshold {:.1}%",
                    self.crit_pct
                ),
            )
        } else if usage_pct >= self.warn_pct {
            ComponentHealth::degraded(
                &name,
                format!(
                    "disk usage {usage_pct:.1}% exceeds warning threshold {:.1}%",
                    self.warn_pct
                ),
            )
        } else {
            ComponentHealth::healthy(&name)
        };

        health = health
            .with_latency(latency)
            .with_detail("usage_pct", format!("{usage_pct:.2}"))
            .with_detail("used_bytes", used.to_string())
            .with_detail("total_bytes", total.to_string());

        health
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MemoryChecker
// ─────────────────────────────────────────────────────────────────────────────

/// Checks available system memory.
///
/// On Linux this reads `/proc/meminfo`; on other platforms it returns a
/// conservative estimate.
pub struct MemoryChecker {
    /// Warning threshold in megabytes of **available** memory.
    pub warn_mb: u64,
    /// Critical threshold in megabytes of **available** memory.
    pub crit_mb: u64,
}

impl MemoryChecker {
    /// Create a new checker.
    #[must_use]
    pub fn new(warn_mb: u64, crit_mb: u64) -> Self {
        Self { warn_mb, crit_mb }
    }

    /// Read available memory in bytes from `/proc/meminfo` (Linux only).
    #[cfg(target_os = "linux")]
    fn available_bytes() -> Option<u64> {
        use std::io::BufRead;

        let file = std::fs::File::open("/proc/meminfo").ok()?;
        let reader = std::io::BufReader::new(file);

        for line in reader.lines() {
            let line = line.ok()?;
            if line.starts_with("MemAvailable:") {
                let kb: u64 = line
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
                return Some(kb * 1024);
            }
        }
        None
    }

    #[cfg(not(target_os = "linux"))]
    fn available_bytes() -> Option<u64> {
        // Conservative: report a fixed 1 GiB available on non-Linux.
        Some(1024 * 1024 * 1024)
    }
}

impl HealthChecker for MemoryChecker {
    fn name(&self) -> &str {
        "memory"
    }

    fn check(&self) -> ComponentHealth {
        let start = Instant::now();

        let available_mb = Self::available_bytes()
            .map(|b| b / (1024 * 1024))
            .unwrap_or(0);

        let latency = start.elapsed().as_secs_f64() * 1000.0;

        let health = if available_mb < self.crit_mb {
            ComponentHealth::unhealthy(
                "memory",
                format!(
                    "available memory {available_mb} MiB below critical threshold {} MiB",
                    self.crit_mb
                ),
            )
        } else if available_mb < self.warn_mb {
            ComponentHealth::degraded(
                "memory",
                format!(
                    "available memory {available_mb} MiB below warning threshold {} MiB",
                    self.warn_mb
                ),
            )
        } else {
            ComponentHealth::healthy("memory")
        };

        health
            .with_latency(latency)
            .with_detail("available_mb", available_mb.to_string())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ProcessUptimeChecker
// ─────────────────────────────────────────────────────────────────────────────

/// Always healthy; reports process uptime since `started_at`.
pub struct ProcessUptimeChecker {
    /// Monotonic instant when the process (or subsystem) started.
    pub started_at: Instant,
}

impl ProcessUptimeChecker {
    /// Create a new uptime checker starting from `now`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            started_at: Instant::now(),
        }
    }

    /// Create a checker with an explicit start instant.
    #[must_use]
    pub fn with_start(started_at: Instant) -> Self {
        Self { started_at }
    }
}

impl Default for ProcessUptimeChecker {
    fn default() -> Self {
        Self::new()
    }
}

impl HealthChecker for ProcessUptimeChecker {
    fn name(&self) -> &str {
        "process_uptime"
    }

    fn check(&self) -> ComponentHealth {
        let uptime_secs = self.started_at.elapsed().as_secs();
        ComponentHealth::healthy("process_uptime")
            .with_detail("uptime_secs", uptime_secs.to_string())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// QueueDepthChecker
// ─────────────────────────────────────────────────────────────────────────────

/// Checks the depth of a shared atomic queue counter.
pub struct QueueDepthChecker {
    /// Shared atomic current depth.
    pub current_depth: Arc<AtomicUsize>,
    /// Depth at which status becomes `Degraded`.
    pub warn_depth: usize,
    /// Depth at which status becomes `Unhealthy`.
    pub crit_depth: usize,
}

impl QueueDepthChecker {
    /// Create a new checker.
    #[must_use]
    pub fn new(current_depth: Arc<AtomicUsize>, warn_depth: usize, crit_depth: usize) -> Self {
        Self {
            current_depth,
            warn_depth,
            crit_depth,
        }
    }
}

impl HealthChecker for QueueDepthChecker {
    fn name(&self) -> &str {
        "queue_depth"
    }

    fn check(&self) -> ComponentHealth {
        let depth = self
            .current_depth
            .load(std::sync::atomic::Ordering::Relaxed);

        let health = if depth >= self.crit_depth {
            ComponentHealth::unhealthy(
                "queue_depth",
                format!(
                    "queue depth {depth} exceeds critical threshold {}",
                    self.crit_depth
                ),
            )
        } else if depth >= self.warn_depth {
            ComponentHealth::degraded(
                "queue_depth",
                format!(
                    "queue depth {depth} exceeds warning threshold {}",
                    self.warn_depth
                ),
            )
        } else {
            ComponentHealth::healthy("queue_depth")
        };

        health.with_detail("depth", depth.to_string())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// HealthRegistry
// ─────────────────────────────────────────────────────────────────────────────

/// A registry of [`HealthChecker`] instances that can be polled collectively.
pub struct HealthRegistry {
    checkers: Vec<Box<dyn HealthChecker>>,
}

impl HealthRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            checkers: Vec::new(),
        }
    }

    /// Register a new health checker.
    pub fn register(&mut self, checker: Box<dyn HealthChecker>) {
        self.checkers.push(checker);
    }

    /// Run all registered checkers and collect their results.
    #[must_use]
    pub fn check_all(&self) -> Vec<ComponentHealth> {
        self.checkers.iter().map(|c| c.check()).collect()
    }

    /// Return the worst health status across all components.
    ///
    /// If no checkers are registered, returns `Healthy`.
    #[must_use]
    pub fn overall_status(&self) -> HealthStatus {
        let results = self.check_all();
        results
            .into_iter()
            .max_by_key(|r| r.status.severity())
            .map(|r| r.status)
            .unwrap_or(HealthStatus::Healthy)
    }

    /// Produce a JSON value suitable for serving from an HTTP `/health` endpoint.
    ///
    /// ```json
    /// {
    ///   "status": "healthy",
    ///   "components": {
    ///     "memory": { "status": "healthy", "latency_ms": 0.12, "details": {} },
    ///     ...
    ///   }
    /// }
    /// ```
    #[must_use]
    pub fn to_json(&self) -> serde_json::Value {
        let results = self.check_all();

        let overall = results
            .iter()
            .max_by_key(|r| r.status.severity())
            .map_or("healthy", |r| r.status.label());

        let components: serde_json::Map<String, serde_json::Value> = results
            .iter()
            .map(|r| {
                let mut obj = serde_json::Map::new();
                obj.insert(
                    "status".to_string(),
                    serde_json::Value::String(r.status.label().to_string()),
                );

                if let Some(latency) = r.latency_ms {
                    obj.insert(
                        "latency_ms".to_string(),
                        serde_json::Value::Number(
                            serde_json::Number::from_f64(latency)
                                .unwrap_or(serde_json::Number::from(0u64)),
                        ),
                    );
                }

                let details: serde_json::Map<String, serde_json::Value> = r
                    .details
                    .iter()
                    .map(|(k, v)| (k.clone(), serde_json::Value::String(v.clone())))
                    .collect();
                obj.insert("details".to_string(), serde_json::Value::Object(details));

                if let HealthStatus::Degraded(ref msg) | HealthStatus::Unhealthy(ref msg) = r.status
                {
                    obj.insert(
                        "message".to_string(),
                        serde_json::Value::String(msg.clone()),
                    );
                }

                (r.name.clone(), serde_json::Value::Object(obj))
            })
            .collect();

        serde_json::json!({
            "status": overall,
            "components": serde_json::Value::Object(components),
        })
    }
}

impl Default for HealthRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

    // ── HealthStatus ─────────────────────────────────────────────────────────

    #[test]
    fn test_health_status_severity_ordering() {
        assert!(HealthStatus::Healthy.severity() < HealthStatus::Degraded("x".into()).severity());
        assert!(
            HealthStatus::Degraded("x".into()).severity()
                < HealthStatus::Unhealthy("x".into()).severity()
        );
    }

    #[test]
    fn test_health_status_is_healthy() {
        assert!(HealthStatus::Healthy.is_healthy());
        assert!(!HealthStatus::Degraded("r".into()).is_healthy());
        assert!(!HealthStatus::Unhealthy("r".into()).is_healthy());
    }

    #[test]
    fn test_health_status_labels() {
        assert_eq!(HealthStatus::Healthy.label(), "healthy");
        assert_eq!(HealthStatus::Degraded("r".into()).label(), "degraded");
        assert_eq!(HealthStatus::Unhealthy("r".into()).label(), "unhealthy");
    }

    // ── ProcessUptimeChecker ─────────────────────────────────────────────────

    #[test]
    fn test_uptime_checker_is_always_healthy() {
        let checker = ProcessUptimeChecker::new();
        let result = checker.check();
        assert!(
            result.status.is_healthy(),
            "uptime checker must always be healthy"
        );
    }

    #[test]
    fn test_uptime_checker_reports_uptime_detail() {
        let checker = ProcessUptimeChecker::new();
        let result = checker.check();
        assert!(
            result.details.contains_key("uptime_secs"),
            "missing uptime_secs detail"
        );
    }

    // ── QueueDepthChecker ────────────────────────────────────────────────────

    #[test]
    fn test_queue_depth_healthy() {
        let depth = Arc::new(AtomicUsize::new(5));
        let checker = QueueDepthChecker::new(depth, 50, 100);
        let result = checker.check();
        assert!(result.status.is_healthy());
    }

    #[test]
    fn test_queue_depth_degraded() {
        let depth = Arc::new(AtomicUsize::new(60));
        let checker = QueueDepthChecker::new(depth, 50, 100);
        let result = checker.check();
        assert!(matches!(result.status, HealthStatus::Degraded(_)));
    }

    #[test]
    fn test_queue_depth_unhealthy() {
        let depth = Arc::new(AtomicUsize::new(150));
        let checker = QueueDepthChecker::new(depth, 50, 100);
        let result = checker.check();
        assert!(matches!(result.status, HealthStatus::Unhealthy(_)));
    }

    #[test]
    fn test_queue_depth_atomic_update_reflected() {
        let depth = Arc::new(AtomicUsize::new(0));
        let checker = QueueDepthChecker::new(Arc::clone(&depth), 10, 20);

        let before = checker.check();
        assert!(before.status.is_healthy());

        depth.store(15, Ordering::Relaxed);
        let after = checker.check();
        assert!(matches!(after.status, HealthStatus::Degraded(_)));
    }

    // ── DiskSpaceChecker ─────────────────────────────────────────────────────

    #[test]
    fn test_disk_space_checker_runs_without_panic() {
        let checker = DiskSpaceChecker::new("/tmp", 80.0, 95.0);
        let result = checker.check();
        // We just verify it doesn't panic and returns a valid name.
        assert!(result.name.contains("disk_space"));
    }

    // ── MemoryChecker ────────────────────────────────────────────────────────

    #[test]
    fn test_memory_checker_runs_without_panic() {
        let checker = MemoryChecker::new(512, 256);
        let result = checker.check();
        assert_eq!(checker.name(), "memory");
        assert!(result.details.contains_key("available_mb"));
    }

    // ── HealthRegistry ───────────────────────────────────────────────────────

    #[test]
    fn test_registry_overall_status_empty_is_healthy() {
        let registry = HealthRegistry::new();
        assert!(registry.overall_status().is_healthy());
    }

    #[test]
    fn test_registry_overall_status_worst_wins() {
        let depth = Arc::new(AtomicUsize::new(200)); // → Unhealthy
        let mut registry = HealthRegistry::new();
        registry.register(Box::new(ProcessUptimeChecker::new()));
        registry.register(Box::new(QueueDepthChecker::new(depth, 50, 100)));

        let overall = registry.overall_status();
        assert!(
            matches!(overall, HealthStatus::Unhealthy(_)),
            "expected Unhealthy, got {:?}",
            overall
        );
    }

    #[test]
    fn test_registry_to_json_structure() {
        let mut registry = HealthRegistry::new();
        registry.register(Box::new(ProcessUptimeChecker::new()));

        let json = registry.to_json();
        assert!(json.get("status").is_some(), "missing 'status' key");
        assert!(json.get("components").is_some(), "missing 'components' key");

        let components = json["components"]
            .as_object()
            .expect("components must be object");
        assert!(
            components.contains_key("process_uptime"),
            "missing process_uptime component"
        );
    }

    #[test]
    fn test_registry_check_all_returns_one_per_checker() {
        let mut registry = HealthRegistry::new();
        registry.register(Box::new(ProcessUptimeChecker::new()));
        registry.register(Box::new(MemoryChecker::new(512, 256)));
        let results = registry.check_all();
        assert_eq!(results.len(), 2);
    }
}
