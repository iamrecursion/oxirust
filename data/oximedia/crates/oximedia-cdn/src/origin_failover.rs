//! Origin server failover and load-balancing for CDN configurations.
//!
//! Provides thread-safe [`OriginServer`] (behind [`Arc`]) and [`OriginPool`]
//! with multiple selection strategies: [`OriginStrategy::Priority`],
//! [`OriginStrategy::WeightedRoundRobin`], [`OriginStrategy::ResponseTimeBased`],
//! and [`OriginStrategy::LeastConnections`].
//!
//! # Health tracking
//!
//! Each [`OriginServer`] maintains atomic counters so it can be shared across
//! threads without an outer lock.  After three consecutive failures the server
//! is marked unhealthy; a single success restores it.  Response latency is
//! tracked via EWMA with α = 0.3.
//!
//! # Health checker
//!
//! [`HealthChecker`] wraps a pool and can perform a simulated health probe
//! round, marking servers healthy or unhealthy based on a caller-supplied
//! oracle function.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use thiserror::Error;

// ─── Error ────────────────────────────────────────────────────────────────────

/// Errors produced by origin-pool operations.
#[derive(Debug, Error)]
pub enum OriginError {
    /// No healthy origins are available.
    #[error("no healthy origins available in pool")]
    NoHealthyOrigins,
    /// An origin with the given ID was not found.
    #[error("origin '{0}' not found")]
    NotFound(String),
    /// A Mutex was poisoned.
    #[error("internal lock poisoned")]
    LockPoisoned,
}

// ─── OriginStrategy ───────────────────────────────────────────────────────────

/// Strategy used by [`OriginPool`] to select the next origin server.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OriginStrategy {
    /// Use the healthy origin with the lowest `priority` number (0 = primary).
    Priority,
    /// Weighted round-robin across healthy origins (higher weight → more traffic).
    WeightedRoundRobin,
    /// Select the healthy origin with the lowest EWMA response time.
    ResponseTimeBased,
    /// Approximate least-connections using active connection count.
    LeastConnections,
}

// ─── OriginServer ─────────────────────────────────────────────────────────────

/// A single upstream / origin server with lock-free health tracking.
pub struct OriginServer {
    /// Unique identifier.
    pub id: String,
    /// Base URL of the origin (e.g. `"https://origin1.example.com"`).
    pub url: String,
    /// Weight used for weighted round-robin (higher = more traffic).
    pub weight: u32,
    /// Priority tier: 0 = primary, higher numbers = fallback tiers.
    pub priority: u8,

    // ── Atomic health state ────────────────────────────────────────────────
    /// Whether the server is currently healthy.
    pub healthy: AtomicBool,
    /// Number of consecutive failures since last healthy state.
    consecutive_failures: AtomicU32,
    /// Number of consecutive successes since last failure.
    consecutive_successes: AtomicU32,
    /// Mark unhealthy after this many consecutive failures.
    pub failure_threshold: u32,
    /// Mark healthy again after this many consecutive successes.
    pub recovery_threshold: u32,

    // ── Connection tracking ────────────────────────────────────────────────
    /// Current number of open connections (approximate).
    pub active_connections: AtomicU32,

    // ── Latency EWMA ──────────────────────────────────────────────────────
    /// Exponentially-smoothed response time in milliseconds (α = 0.3).
    pub ewma_response_ms: Mutex<f64>,

    // ── Request timeout ───────────────────────────────────────────────────
    /// Request timeout in milliseconds.
    pub timeout_ms: u64,

    // ── Total request counter ─────────────────────────────────────────────
    total_requests: AtomicU64,
    total_failures: AtomicU64,
}

impl std::fmt::Debug for OriginServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OriginServer")
            .field("id", &self.id)
            .field("url", &self.url)
            .field("weight", &self.weight)
            .field("priority", &self.priority)
            .field("healthy", &self.healthy.load(Ordering::Relaxed))
            .field(
                "consecutive_failures",
                &self.consecutive_failures.load(Ordering::Relaxed),
            )
            .field(
                "consecutive_successes",
                &self.consecutive_successes.load(Ordering::Relaxed),
            )
            .field(
                "active_connections",
                &self.active_connections.load(Ordering::Relaxed),
            )
            .field("timeout_ms", &self.timeout_ms)
            .finish()
    }
}

impl OriginServer {
    /// Create a new origin server with sensible defaults.
    ///
    /// Starts healthy with EWMA = 100 ms, failure_threshold = 3, recovery_threshold = 1.
    pub fn new(id: &str, url: &str, weight: u32, priority: u8) -> Self {
        Self {
            id: id.to_string(),
            url: url.to_string(),
            weight,
            priority,
            healthy: AtomicBool::new(true),
            consecutive_failures: AtomicU32::new(0),
            consecutive_successes: AtomicU32::new(0),
            failure_threshold: 3,
            recovery_threshold: 1,
            active_connections: AtomicU32::new(0),
            ewma_response_ms: Mutex::new(100.0),
            timeout_ms: 5_000,
            total_requests: AtomicU64::new(0),
            total_failures: AtomicU64::new(0),
        }
    }

    /// Returns `true` if the server is currently marked healthy.
    pub fn is_healthy(&self) -> bool {
        self.healthy.load(Ordering::Acquire)
    }

    /// Atomically increment the active-connection count.
    pub fn connect(&self) {
        self.active_connections.fetch_add(1, Ordering::Relaxed);
    }

    /// Atomically decrement the active-connection count (floor 0).
    pub fn disconnect(&self) {
        self.active_connections
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
                Some(v.saturating_sub(1))
            })
            .ok();
    }

    /// Record a failed request.
    ///
    /// Increments the consecutive failure counter and marks the server
    /// unhealthy once `failure_threshold` is reached.
    pub fn record_failure(&self) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.total_failures.fetch_add(1, Ordering::Relaxed);
        self.consecutive_successes.store(0, Ordering::Relaxed);
        let failures = self.consecutive_failures.fetch_add(1, Ordering::Relaxed) + 1;
        if failures >= self.failure_threshold {
            self.healthy.store(false, Ordering::Release);
        }
    }

    /// Record a successful response.
    ///
    /// - Updates EWMA: `new = 0.3 * sample + 0.7 * old`.
    /// - Resets the failure counter.
    /// - Marks the server healthy after `recovery_threshold` consecutive
    ///   successes.
    pub fn record_response_time(&self, sample_ms: f64) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.consecutive_failures.store(0, Ordering::Relaxed);
        let successes = self.consecutive_successes.fetch_add(1, Ordering::Relaxed) + 1;
        if successes >= self.recovery_threshold {
            self.healthy.store(true, Ordering::Release);
        }
        // Update EWMA — propagate lock-poison as a no-op to avoid unwrap.
        if let Ok(mut guard) = self.ewma_response_ms.lock() {
            *guard = 0.3 * sample_ms + 0.7 * (*guard);
        }
    }

    /// Read the current EWMA response time in milliseconds.
    pub fn ewma_ms(&self) -> f64 {
        self.ewma_response_ms.lock().map(|g| *g).unwrap_or(100.0)
    }

    /// Total requests observed (success + failure).
    pub fn total_requests(&self) -> u64 {
        self.total_requests.load(Ordering::Relaxed)
    }

    /// Total failures observed.
    pub fn total_failures(&self) -> u64 {
        self.total_failures.load(Ordering::Relaxed)
    }

    /// Current consecutive failure count.
    pub fn consecutive_failures(&self) -> u32 {
        self.consecutive_failures.load(Ordering::Relaxed)
    }

    /// Current consecutive success count.
    pub fn consecutive_successes(&self) -> u32 {
        self.consecutive_successes.load(Ordering::Relaxed)
    }
}

// ─── OriginPool ───────────────────────────────────────────────────────────────

/// A pool of [`OriginServer`]s (held as `Arc`) with a shared selection strategy.
pub struct OriginPool {
    servers: Vec<Arc<OriginServer>>,
    strategy: OriginStrategy,
    /// Internal round-robin cursor (protected by a Mutex for interior mutability).
    rr_index: Mutex<usize>,
}

impl OriginPool {
    /// Create an empty pool with the given selection strategy.
    pub fn new(strategy: OriginStrategy) -> Self {
        Self {
            servers: Vec::new(),
            strategy,
            rr_index: Mutex::new(0),
        }
    }

    /// Add a server to the pool.  The pool takes shared ownership via [`Arc`].
    pub fn add_server(&mut self, server: Arc<OriginServer>) {
        self.servers.push(server);
    }

    /// Add a server by value (wraps it in an [`Arc`] automatically).
    pub fn add_server_owned(&mut self, server: OriginServer) {
        self.servers.push(Arc::new(server));
    }

    /// Select the best server according to the pool's strategy.
    ///
    /// Returns `None` if no healthy server is available.
    pub fn select(&self) -> Option<Arc<OriginServer>> {
        match &self.strategy {
            OriginStrategy::Priority => self.select_priority(),
            OriginStrategy::WeightedRoundRobin => self.select_weighted_rr(),
            OriginStrategy::ResponseTimeBased => self.select_response_time(),
            OriginStrategy::LeastConnections => self.select_least_connections(),
        }
    }

    // ── Priority ──────────────────────────────────────────────────────────

    fn select_priority(&self) -> Option<Arc<OriginServer>> {
        self.servers
            .iter()
            .filter(|s| s.is_healthy())
            .min_by_key(|s| s.priority)
            .cloned()
    }

    // ── Weighted round-robin ──────────────────────────────────────────────

    fn select_weighted_rr(&self) -> Option<Arc<OriginServer>> {
        let healthy: Vec<&Arc<OriginServer>> = self
            .servers
            .iter()
            .filter(|s| s.is_healthy() && s.weight > 0)
            .collect();
        if healthy.is_empty() {
            return None;
        }
        let total_weight: u32 = healthy.iter().map(|s| s.weight).sum();
        if total_weight == 0 {
            return None;
        }
        let mut idx = match self.rr_index.lock() {
            Ok(g) => g,
            Err(_) => return healthy.last().map(|s| Arc::clone(s)),
        };
        *idx = (*idx + 1) % total_weight as usize;
        let target = *idx as u32;
        let mut cumulative = 0u32;
        for s in &healthy {
            cumulative += s.weight;
            if target < cumulative {
                return Some(Arc::clone(s));
            }
        }
        healthy.last().map(|s| Arc::clone(s))
    }

    // ── Response-time based ───────────────────────────────────────────────

    fn select_response_time(&self) -> Option<Arc<OriginServer>> {
        self.servers
            .iter()
            .filter(|s| s.is_healthy())
            .min_by(|a, b| {
                a.ewma_ms()
                    .partial_cmp(&b.ewma_ms())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned()
    }

    // ── Least connections ─────────────────────────────────────────────────

    fn select_least_connections(&self) -> Option<Arc<OriginServer>> {
        self.servers
            .iter()
            .filter(|s| s.is_healthy())
            .min_by_key(|s| s.active_connections.load(Ordering::Relaxed))
            .cloned()
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    /// Number of currently healthy servers.
    pub fn healthy_count(&self) -> usize {
        self.servers.iter().filter(|s| s.is_healthy()).count()
    }

    /// `true` if every server is unhealthy.
    pub fn all_failed(&self) -> bool {
        !self.servers.iter().any(|s| s.is_healthy())
    }

    /// Reset all servers to healthy with zeroed failure/success counters.
    pub fn reset_all(&self) {
        for s in &self.servers {
            s.healthy.store(true, Ordering::Release);
            s.consecutive_failures.store(0, Ordering::Relaxed);
            s.consecutive_successes.store(0, Ordering::Relaxed);
        }
    }

    /// Borrow the underlying server slice.
    pub fn servers(&self) -> &[Arc<OriginServer>] {
        &self.servers
    }

    /// Find a server by ID.
    pub fn get_server(&self, id: &str) -> Option<Arc<OriginServer>> {
        self.servers.iter().find(|s| s.id == id).cloned()
    }
}

// ─── Health check configuration ───────────────────────────────────────────────

/// Protocol used for health-check probes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HealthCheckProtocol {
    /// HTTP(S) GET probe — expects a 2xx response code.
    Http {
        /// Path appended to the origin URL (e.g. `"/healthz"`).
        path: String,
        /// Expected HTTP status code range (inclusive lower, exclusive upper).
        expected_status_min: u16,
        /// Upper bound (exclusive) of the expected HTTP status code range.
        expected_status_max: u16,
        /// Optional `Host` header override.
        host_header: Option<String>,
    },
    /// TCP connect probe — succeeds if a TCP connection can be established
    /// within the configured timeout.
    Tcp {
        /// Port to probe (overrides the port in the origin URL if set).
        port: Option<u16>,
    },
}

/// Result of a single health-check probe against one origin server.
#[derive(Debug, Clone)]
pub struct HealthCheckProbe {
    /// The origin server ID that was probed.
    pub server_id: String,
    /// Whether the probe was successful.
    pub healthy: bool,
    /// Measured latency in milliseconds (0.0 if the probe failed).
    pub latency_ms: f64,
    /// Human-readable reason (empty on success).
    pub reason: String,
}

/// Per-origin health-check configuration.
#[derive(Debug, Clone)]
pub struct HealthCheckConfig {
    /// Protocol and parameters for the probe.
    pub protocol: HealthCheckProtocol,
    /// Interval between consecutive probes for this origin.
    pub interval: Duration,
    /// Per-probe timeout.
    pub timeout: Duration,
    /// Number of consecutive failures before marking unhealthy.
    pub failure_threshold: u32,
    /// Number of consecutive successes before marking healthy.
    pub recovery_threshold: u32,
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            protocol: HealthCheckProtocol::Http {
                path: "/healthz".to_string(),
                expected_status_min: 200,
                expected_status_max: 300,
                host_header: None,
            },
            interval: Duration::from_secs(30),
            timeout: Duration::from_secs(5),
            failure_threshold: 3,
            recovery_threshold: 1,
        }
    }
}

impl HealthCheckConfig {
    /// Create a TCP-only health-check config.
    pub fn tcp(port: Option<u16>) -> Self {
        Self {
            protocol: HealthCheckProtocol::Tcp { port },
            ..Self::default()
        }
    }

    /// Create an HTTP health-check config with a custom path.
    pub fn http(path: impl Into<String>) -> Self {
        Self {
            protocol: HealthCheckProtocol::Http {
                path: path.into(),
                expected_status_min: 200,
                expected_status_max: 300,
                host_header: None,
            },
            ..Self::default()
        }
    }

    /// Set the probe interval.
    pub fn with_interval(mut self, interval: Duration) -> Self {
        self.interval = interval;
        self
    }

    /// Set the probe timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set the failure threshold.
    pub fn with_failure_threshold(mut self, threshold: u32) -> Self {
        self.failure_threshold = threshold;
        self
    }

    /// Set the recovery threshold.
    pub fn with_recovery_threshold(mut self, threshold: u32) -> Self {
        self.recovery_threshold = threshold;
        self
    }
}

// ─── HealthChecker ────────────────────────────────────────────────────────────

/// Drives periodic (simulated) health checks against an [`OriginPool`].
///
/// Supports configurable per-origin health check protocols (HTTP or TCP).
/// No real HTTP calls are made — the caller supplies an oracle function that
/// returns a `(healthy: bool, response_ms: f64)` tuple for each server URL,
/// or uses the configurable probe system with [`HealthCheckConfig`].
pub struct HealthChecker {
    pool: Arc<OriginPool>,
    /// Minimum interval between full check rounds.
    pub check_interval: Duration,
    /// Timestamp of the last check round.
    last_check: Mutex<Option<Instant>>,
    /// Per-origin health check configurations, keyed by server ID.
    configs: Mutex<std::collections::HashMap<String, HealthCheckConfig>>,
    /// Default health check configuration for origins without a specific config.
    pub default_config: HealthCheckConfig,
}

impl HealthChecker {
    /// Wrap `pool` in a checker with the given interval.
    pub fn new(pool: Arc<OriginPool>, check_interval: Duration) -> Self {
        Self {
            pool,
            check_interval,
            last_check: Mutex::new(None),
            configs: Mutex::new(std::collections::HashMap::new()),
            default_config: HealthCheckConfig::default(),
        }
    }

    /// Register a per-origin health-check configuration.
    pub fn set_config(&self, server_id: &str, config: HealthCheckConfig) {
        if let Ok(mut guard) = self.configs.lock() {
            guard.insert(server_id.to_string(), config);
        }
    }

    /// Retrieve the health-check configuration for a specific origin,
    /// falling back to `default_config` if none is registered.
    pub fn config_for(&self, server_id: &str) -> HealthCheckConfig {
        if let Ok(guard) = self.configs.lock() {
            if let Some(cfg) = guard.get(server_id) {
                return cfg.clone();
            }
        }
        self.default_config.clone()
    }

    /// Run health checks using the configurable probe system.
    ///
    /// `probe_fn` receives the server URL and its [`HealthCheckConfig`] and
    /// returns a [`HealthCheckProbe`] result. This allows callers to implement
    /// real HTTP or TCP checks while the checker drives the scheduling and
    /// state transitions.
    pub fn check_with_config<F>(&self, probe_fn: F) -> Vec<HealthCheckProbe>
    where
        F: Fn(&str, &HealthCheckConfig) -> HealthCheckProbe,
    {
        let mut results = Vec::new();
        for server in self.pool.servers() {
            let cfg = self.config_for(&server.id);
            let probe = probe_fn(&server.url, &cfg);
            if probe.healthy {
                server.record_response_time(probe.latency_ms);
            } else {
                server.record_failure();
            }
            results.push(probe);
        }
        if let Ok(mut last) = self.last_check.lock() {
            *last = Some(Instant::now());
        }
        results
    }

    /// Run one health-check round using `probe`.
    ///
    /// `probe(url) -> (is_healthy, response_ms)`.
    ///
    /// Each server in the pool is probed once; the result is applied via
    /// [`OriginServer::record_response_time`] or [`OriginServer::record_failure`].
    ///
    /// Returns the number of servers that were probed.
    pub fn check_now<F>(&self, probe: F) -> usize
    where
        F: Fn(&str) -> (bool, f64),
    {
        let mut checked = 0usize;
        for server in self.pool.servers() {
            let (ok, ms) = probe(&server.url);
            if ok {
                server.record_response_time(ms);
            } else {
                server.record_failure();
            }
            checked += 1;
        }
        if let Ok(mut last) = self.last_check.lock() {
            *last = Some(Instant::now());
        }
        checked
    }

    /// Returns `true` if `check_interval` has elapsed since the last check
    /// (or if no check has been run yet).
    pub fn is_due(&self) -> bool {
        match self.last_check.lock() {
            Err(_) => true,
            Ok(last) => match *last {
                None => true,
                Some(t) => t.elapsed() >= self.check_interval,
            },
        }
    }

    /// Run a check only if [`Self::is_due`] returns `true`.
    ///
    /// Returns the number of servers checked (0 if not due).
    pub fn check_if_due<F>(&self, probe: F) -> usize
    where
        F: Fn(&str) -> (bool, f64),
    {
        if self.is_due() {
            self.check_now(probe)
        } else {
            0
        }
    }

    /// Reference to the wrapped pool.
    pub fn pool(&self) -> &Arc<OriginPool> {
        &self.pool
    }
}

// ─── Circuit breaker ──────────────────────────────────────────────────────────

/// State of a circuit breaker protecting an origin server.
///
/// State transitions:
/// - **Closed** → **Open** when failure threshold is crossed.
/// - **Open** → **HalfOpen** after `open_duration` has elapsed (timeout recovery).
/// - **HalfOpen** → **Closed** on a successful probe; **HalfOpen** → **Open** on failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitBreakerState {
    /// Normal operation — requests flow through.
    Closed,
    /// Circuit is open — requests are rejected immediately.
    Open,
    /// A single probe request is allowed to test if the origin has recovered.
    HalfOpen,
}

/// A circuit breaker wrapping an [`OriginServer`].
///
/// Transitions between [`CircuitBreakerState`]s based on health results and
/// an elapsed timeout.  Thread-safety is achieved through atomics for state
/// transitions where possible; `Mutex<Instant>` protects the time-of-open
/// field.
pub struct CircuitBreaker {
    /// The origin server guarded by this breaker.
    pub server: Arc<OriginServer>,
    /// Duration to remain in [`CircuitBreakerState::Open`] before attempting
    /// a half-open probe.
    pub open_duration: Duration,
    state: Mutex<CircuitBreakerState>,
    opened_at: Mutex<Option<Instant>>,
}

impl CircuitBreaker {
    /// Create a new breaker for `server` with the given open-state timeout.
    pub fn new(server: Arc<OriginServer>, open_duration: Duration) -> Self {
        Self {
            server,
            open_duration,
            state: Mutex::new(CircuitBreakerState::Closed),
            opened_at: Mutex::new(None),
        }
    }

    /// Read the current circuit breaker state.
    pub fn state(&self) -> CircuitBreakerState {
        self.state
            .lock()
            .map(|g| *g)
            .unwrap_or(CircuitBreakerState::Open)
    }

    /// Transition **Open → HalfOpen** when `open_duration` has elapsed.
    ///
    /// Returns the new state.  If the breaker is already **Closed** or
    /// **HalfOpen** this is a no-op.
    pub fn check_health_and_maybe_recover(&self) -> CircuitBreakerState {
        let mut state_guard = match self.state.lock() {
            Ok(g) => g,
            Err(_) => return CircuitBreakerState::Open,
        };

        if *state_guard != CircuitBreakerState::Open {
            return *state_guard;
        }

        // Check if enough time has passed since the breaker opened.
        let opened_guard = match self.opened_at.lock() {
            Ok(g) => g,
            Err(_) => return CircuitBreakerState::Open,
        };
        let should_probe = match *opened_guard {
            None => false,
            Some(t) => t.elapsed() >= self.open_duration,
        };

        if should_probe {
            *state_guard = CircuitBreakerState::HalfOpen;
        }
        *state_guard
    }

    /// Record a successful probe result.
    ///
    /// - **HalfOpen** → **Closed**: origin has recovered.
    /// - **Closed**: no-op (already healthy).
    /// - **Open**: ignored (only probes from HalfOpen count).
    pub fn record_success(&self, latency_ms: f64) {
        let mut state_guard = match self.state.lock() {
            Ok(g) => g,
            Err(_) => return,
        };
        if *state_guard == CircuitBreakerState::HalfOpen {
            *state_guard = CircuitBreakerState::Closed;
            if let Ok(mut t) = self.opened_at.lock() {
                *t = None;
            }
        }
        // Update origin EWMA regardless.
        self.server.record_response_time(latency_ms);
    }

    /// Record a failed probe result.
    ///
    /// - **HalfOpen** → **Open**: probe failed; reset the open timer.
    /// - **Closed** → **Open** (if server failure threshold crossed).
    pub fn record_failure(&self) {
        self.server.record_failure();
        let mut state_guard = match self.state.lock() {
            Ok(g) => g,
            Err(_) => return,
        };
        let server_failed = !self.server.is_healthy();
        if *state_guard == CircuitBreakerState::HalfOpen || server_failed {
            *state_guard = CircuitBreakerState::Open;
            if let Ok(mut t) = self.opened_at.lock() {
                *t = Some(Instant::now());
            }
        }
    }

    /// Manually trip the breaker to **Open** (e.g. on external signal).
    pub fn trip(&self) {
        if let Ok(mut state) = self.state.lock() {
            *state = CircuitBreakerState::Open;
        }
        if let Ok(mut t) = self.opened_at.lock() {
            *t = Some(Instant::now());
        }
    }

    /// Reset to **Closed** (e.g. after manual operator intervention).
    pub fn reset(&self) {
        if let Ok(mut state) = self.state.lock() {
            *state = CircuitBreakerState::Closed;
        }
        if let Ok(mut t) = self.opened_at.lock() {
            *t = None;
        }
        self.server
            .healthy
            .store(true, std::sync::atomic::Ordering::Release);
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_server(id: &str, url: &str, weight: u32, priority: u8) -> Arc<OriginServer> {
        Arc::new(OriginServer::new(id, url, weight, priority))
    }

    // 1. Defaults
    #[test]
    fn test_origin_server_defaults() {
        let s = make_server("s1", "https://origin.example.com", 10, 0);
        assert_eq!(s.id, "s1");
        assert_eq!(s.weight, 10);
        assert_eq!(s.priority, 0);
        assert!(s.is_healthy());
        assert_eq!(s.consecutive_failures(), 0);
        assert!((s.ewma_ms() - 100.0).abs() < 1e-6);
    }

    // 2. record_failure marks unhealthy at threshold
    #[test]
    fn test_record_failure_marks_unhealthy() {
        let s = make_server("s", "http://x", 1, 0);
        s.record_failure();
        assert!(s.is_healthy());
        s.record_failure();
        assert!(s.is_healthy());
        s.record_failure(); // 3rd — crosses threshold
        assert!(!s.is_healthy());
    }

    // 3. record_failure resets consecutive_successes
    #[test]
    fn test_record_failure_resets_successes() {
        let s = make_server("s", "http://x", 1, 0);
        s.consecutive_successes.store(5, Ordering::Relaxed);
        s.record_failure();
        assert_eq!(s.consecutive_successes(), 0);
    }

    // 4. record_response_time restores health
    #[test]
    fn test_record_response_time_restores_health() {
        let s = make_server("s", "http://x", 1, 0);
        s.healthy.store(false, Ordering::Relaxed);
        s.record_response_time(50.0);
        assert!(s.is_healthy()); // recovery_threshold = 1
    }

    // 5. EWMA update: new = 0.3 * sample + 0.7 * old
    #[test]
    fn test_ewma_update() {
        let s = make_server("s", "http://x", 1, 0);
        // initial EWMA = 100.0
        s.record_response_time(200.0); // 0.3*200 + 0.7*100 = 130
        let ewma = s.ewma_ms();
        assert!((ewma - 130.0).abs() < 1e-4, "ewma={ewma}");
    }

    // 6. record_response_time resets failure counter
    #[test]
    fn test_record_success_resets_failures() {
        let s = make_server("s", "http://x", 1, 0);
        s.consecutive_failures.store(2, Ordering::Relaxed);
        s.record_response_time(50.0);
        assert_eq!(s.consecutive_failures(), 0);
    }

    // 7. connect / disconnect
    #[test]
    fn test_connect_disconnect() {
        let s = make_server("s", "http://x", 1, 0);
        s.connect();
        s.connect();
        assert_eq!(s.active_connections.load(Ordering::Relaxed), 2);
        s.disconnect();
        assert_eq!(s.active_connections.load(Ordering::Relaxed), 1);
        s.disconnect();
        s.disconnect(); // no underflow
        assert_eq!(s.active_connections.load(Ordering::Relaxed), 0);
    }

    // 8. Priority strategy selects lowest priority number
    #[test]
    fn test_priority_strategy_selects_lowest() {
        let mut pool = OriginPool::new(OriginStrategy::Priority);
        pool.add_server(make_server("primary", "http://primary", 1, 0));
        pool.add_server(make_server("fallback", "http://fallback", 1, 1));
        let sel = pool.select().expect("healthy server");
        assert_eq!(sel.url, "http://primary");
    }

    // 9. Priority strategy skips unhealthy
    #[test]
    fn test_priority_strategy_skips_unhealthy() {
        let mut pool = OriginPool::new(OriginStrategy::Priority);
        let primary = make_server("p", "http://primary", 1, 0);
        primary.healthy.store(false, Ordering::Relaxed);
        pool.add_server(primary);
        pool.add_server(make_server("fb", "http://fallback", 1, 1));
        let sel = pool.select().expect("fallback");
        assert_eq!(sel.url, "http://fallback");
    }

    // 10. WeightedRoundRobin distributes traffic
    #[test]
    fn test_weighted_rr_distributes() {
        let mut pool = OriginPool::new(OriginStrategy::WeightedRoundRobin);
        pool.add_server(make_server("a", "http://a", 1, 0));
        pool.add_server(make_server("b", "http://b", 1, 0));
        let mut seen_a = false;
        let mut seen_b = false;
        for _ in 0..10 {
            if let Some(s) = pool.select() {
                match s.url.as_str() {
                    "http://a" => seen_a = true,
                    "http://b" => seen_b = true,
                    _ => {}
                }
            }
        }
        assert!(seen_a && seen_b, "both servers should be selected");
    }

    // 11. ResponseTimeBased picks lowest EWMA
    #[test]
    fn test_response_time_based_picks_fastest() {
        let mut pool = OriginPool::new(OriginStrategy::ResponseTimeBased);
        let fast = make_server("fast", "http://fast", 1, 0);
        // lower EWMA initial = drive it down
        fast.record_response_time(10.0);
        fast.record_response_time(10.0);
        let slow = make_server("slow", "http://slow", 1, 0);
        slow.record_response_time(500.0);
        slow.record_response_time(500.0);
        pool.add_server(slow);
        pool.add_server(fast);
        let sel = pool.select().expect("server");
        assert_eq!(sel.url, "http://fast");
    }

    // 12. LeastConnections picks server with fewest active connections
    #[test]
    fn test_least_connections_picks_least_loaded() {
        let mut pool = OriginPool::new(OriginStrategy::LeastConnections);
        let busy = make_server("busy", "http://busy", 1, 0);
        busy.connect();
        busy.connect();
        busy.connect();
        let free = make_server("free", "http://free", 1, 0);
        pool.add_server(busy);
        pool.add_server(free);
        let sel = pool.select().expect("server");
        assert_eq!(sel.url, "http://free");
    }

    // 13. all_failed / healthy_count / reset_all
    #[test]
    fn test_all_failed_and_reset() {
        let mut pool = OriginPool::new(OriginStrategy::Priority);
        let s1 = make_server("s1", "http://s1", 1, 0);
        let s2 = make_server("s2", "http://s2", 1, 1);
        s1.healthy.store(false, Ordering::Relaxed);
        s2.healthy.store(false, Ordering::Relaxed);
        pool.add_server(s1);
        pool.add_server(s2);
        assert_eq!(pool.healthy_count(), 0);
        assert!(pool.all_failed());
        assert!(pool.select().is_none());
        pool.reset_all();
        assert_eq!(pool.healthy_count(), 2);
        assert!(!pool.all_failed());
    }

    // 14. get_server finds by ID
    #[test]
    fn test_get_server_by_id() {
        let mut pool = OriginPool::new(OriginStrategy::Priority);
        pool.add_server(make_server("alpha", "http://alpha", 1, 0));
        pool.add_server(make_server("beta", "http://beta", 1, 1));
        let found = pool.get_server("beta").expect("beta");
        assert_eq!(found.url, "http://beta");
        assert!(pool.get_server("gamma").is_none());
    }

    // 15. add_server_owned convenience method
    #[test]
    fn test_add_server_owned() {
        let mut pool = OriginPool::new(OriginStrategy::Priority);
        pool.add_server_owned(OriginServer::new("s", "http://s", 1, 0));
        assert_eq!(pool.servers().len(), 1);
    }

    // 16. total_requests and total_failures counters
    #[test]
    fn test_request_counters() {
        let s = make_server("s", "http://x", 1, 0);
        s.record_response_time(50.0);
        s.record_response_time(60.0);
        s.record_failure();
        assert_eq!(s.total_requests(), 3);
        assert_eq!(s.total_failures(), 1);
    }

    // 17. Multiple EWMA updates converge correctly
    #[test]
    fn test_ewma_convergence() {
        let s = make_server("s", "http://x", 1, 0);
        // Feed many samples of 50ms — EWMA should converge towards 50.
        for _ in 0..30 {
            s.record_response_time(50.0);
        }
        let ewma = s.ewma_ms();
        assert!(ewma < 60.0, "ewma={ewma} should be close to 50");
    }

    // 18. Unhealthy server after exactly failure_threshold failures
    #[test]
    fn test_exact_failure_threshold() {
        let s = Arc::new(OriginServer {
            failure_threshold: 5,
            ..OriginServer::new("s", "http://x", 1, 0)
        });
        for i in 0..4 {
            s.record_failure();
            assert!(
                s.is_healthy(),
                "should still be healthy after {} failures",
                i + 1
            );
        }
        s.record_failure();
        assert!(!s.is_healthy(), "should be unhealthy after 5 failures");
    }

    // 19. HealthChecker is_due logic
    #[test]
    fn test_health_checker_is_due() {
        let pool = Arc::new(OriginPool::new(OriginStrategy::Priority));
        let checker = HealthChecker::new(pool, Duration::from_secs(60));
        assert!(checker.is_due(), "should be due before first check");
        checker.check_now(|_| (true, 50.0));
        assert!(!checker.is_due(), "should not be due right after check");
    }

    // 20. HealthChecker check_now marks servers healthy/unhealthy
    #[test]
    fn test_health_checker_check_now() {
        let mut pool = OriginPool::new(OriginStrategy::Priority);
        let s1 = make_server("s1", "http://s1", 1, 0);
        // Lower s2's failure threshold to 1 so a single probe failure makes it unhealthy.
        let s2 = Arc::new(OriginServer {
            failure_threshold: 1,
            ..OriginServer::new("s2", "http://s2", 1, 1)
        });
        // Pre-fail s1 so it starts unhealthy.
        for _ in 0..3 {
            s1.record_failure();
        }
        pool.add_server(Arc::clone(&s1));
        pool.add_server(Arc::clone(&s2));
        let pool = Arc::new(pool);
        let checker = HealthChecker::new(Arc::clone(&pool), Duration::from_secs(30));

        let probed = checker.check_now(|url| {
            if url == "http://s1" {
                (true, 80.0) // s1 now succeeds → should recover
            } else {
                (false, 0.0) // s2 now fails
            }
        });
        assert_eq!(probed, 2);
        assert!(s1.is_healthy(), "s1 should be recovered");
        assert!(!s2.is_healthy(), "s2 should be unhealthy");
    }

    // 21. HealthChecker check_if_due skips when not due
    #[test]
    fn test_health_checker_check_if_due() {
        let mut pool = OriginPool::new(OriginStrategy::Priority);
        pool.add_server(make_server("s", "http://s", 1, 0));
        let pool = Arc::new(pool);
        let checker = HealthChecker::new(pool, Duration::from_secs(3600));
        // First call: due → checks.
        let first = checker.check_if_due(|_| (true, 50.0));
        assert_eq!(first, 1);
        // Second call: not due → skips.
        let second = checker.check_if_due(|_| (true, 50.0));
        assert_eq!(second, 0);
    }

    // 22. WeightedRoundRobin with unequal weights
    #[test]
    fn test_weighted_rr_unequal_weights() {
        let mut pool = OriginPool::new(OriginStrategy::WeightedRoundRobin);
        pool.add_server(make_server("heavy", "http://heavy", 3, 0));
        pool.add_server(make_server("light", "http://light", 1, 0));
        let mut heavy_count = 0u32;
        let mut light_count = 0u32;
        for _ in 0..100 {
            if let Some(s) = pool.select() {
                match s.url.as_str() {
                    "http://heavy" => heavy_count += 1,
                    "http://light" => light_count += 1,
                    _ => {}
                }
            }
        }
        assert!(heavy_count > light_count, "heavy should get more traffic");
    }

    // ── HealthCheckConfig ─────────────────────────────────────────────────

    // 23. Default health check config
    #[test]
    fn test_health_check_config_defaults() {
        let cfg = HealthCheckConfig::default();
        assert_eq!(cfg.failure_threshold, 3);
        assert_eq!(cfg.recovery_threshold, 1);
        assert_eq!(cfg.timeout, Duration::from_secs(5));
        assert_eq!(cfg.interval, Duration::from_secs(30));
        match &cfg.protocol {
            HealthCheckProtocol::Http {
                path,
                expected_status_min,
                expected_status_max,
                ..
            } => {
                assert_eq!(path, "/healthz");
                assert_eq!(*expected_status_min, 200);
                assert_eq!(*expected_status_max, 300);
            }
            _ => panic!("expected HTTP protocol"),
        }
    }

    // 24. TCP health check config
    #[test]
    fn test_health_check_config_tcp() {
        let cfg = HealthCheckConfig::tcp(Some(8080));
        match &cfg.protocol {
            HealthCheckProtocol::Tcp { port } => {
                assert_eq!(*port, Some(8080));
            }
            _ => panic!("expected TCP protocol"),
        }
    }

    // 25. HTTP health check config with custom path
    #[test]
    fn test_health_check_config_http() {
        let cfg = HealthCheckConfig::http("/status");
        match &cfg.protocol {
            HealthCheckProtocol::Http { path, .. } => {
                assert_eq!(path, "/status");
            }
            _ => panic!("expected HTTP protocol"),
        }
    }

    // 26. Builder methods
    #[test]
    fn test_health_check_config_builders() {
        let cfg = HealthCheckConfig::http("/alive")
            .with_interval(Duration::from_secs(10))
            .with_timeout(Duration::from_secs(2))
            .with_failure_threshold(5)
            .with_recovery_threshold(2);
        assert_eq!(cfg.interval, Duration::from_secs(10));
        assert_eq!(cfg.timeout, Duration::from_secs(2));
        assert_eq!(cfg.failure_threshold, 5);
        assert_eq!(cfg.recovery_threshold, 2);
    }

    // 27. HealthChecker set_config / config_for
    #[test]
    fn test_health_checker_per_origin_config() {
        let pool = Arc::new(OriginPool::new(OriginStrategy::Priority));
        let checker = HealthChecker::new(pool, Duration::from_secs(30));

        let tcp_cfg = HealthCheckConfig::tcp(Some(9090));
        checker.set_config("origin-1", tcp_cfg);

        let retrieved = checker.config_for("origin-1");
        match &retrieved.protocol {
            HealthCheckProtocol::Tcp { port } => {
                assert_eq!(*port, Some(9090));
            }
            _ => panic!("expected TCP config"),
        }

        // Unknown server falls back to default
        let fallback = checker.config_for("unknown");
        match &fallback.protocol {
            HealthCheckProtocol::Http { path, .. } => {
                assert_eq!(path, "/healthz");
            }
            _ => panic!("expected HTTP default"),
        }
    }

    // 28. HealthChecker check_with_config
    #[test]
    fn test_health_checker_check_with_config() {
        let mut pool = OriginPool::new(OriginStrategy::Priority);
        let s1 = make_server("s1", "http://s1", 1, 0);
        let s2 = make_server("s2", "http://s2", 1, 1);
        pool.add_server(Arc::clone(&s1));
        pool.add_server(Arc::clone(&s2));
        let pool = Arc::new(pool);

        let checker = HealthChecker::new(Arc::clone(&pool), Duration::from_secs(30));
        checker.set_config("s1", HealthCheckConfig::http("/ready"));
        checker.set_config("s2", HealthCheckConfig::tcp(Some(443)));

        let results = checker.check_with_config(|url, cfg| {
            let healthy = url == "http://s1";
            let latency = if healthy { 25.0 } else { 0.0 };
            let reason = if healthy {
                String::new()
            } else {
                match &cfg.protocol {
                    HealthCheckProtocol::Tcp { port } => {
                        format!("TCP connect failed on port {:?}", port)
                    }
                    HealthCheckProtocol::Http { path, .. } => {
                        format!("HTTP check failed at {path}")
                    }
                }
            };
            HealthCheckProbe {
                server_id: if url == "http://s1" {
                    "s1".to_string()
                } else {
                    "s2".to_string()
                },
                healthy,
                latency_ms: latency,
                reason,
            }
        });

        assert_eq!(results.len(), 2);
        assert!(results.iter().any(|r| r.server_id == "s1" && r.healthy));
        assert!(results.iter().any(|r| r.server_id == "s2" && !r.healthy));
    }

    // 29. HealthCheckProbe fields
    #[test]
    fn test_health_check_probe_fields() {
        let probe = HealthCheckProbe {
            server_id: "origin-1".to_string(),
            healthy: true,
            latency_ms: 42.5,
            reason: String::new(),
        };
        assert_eq!(probe.server_id, "origin-1");
        assert!(probe.healthy);
        assert!((probe.latency_ms - 42.5).abs() < 1e-10);
        assert!(probe.reason.is_empty());
    }

    // 30. check_with_config updates is_due
    #[test]
    fn test_check_with_config_updates_last_check() {
        let pool = Arc::new(OriginPool::new(OriginStrategy::Priority));
        let checker = HealthChecker::new(pool, Duration::from_secs(3600));
        assert!(checker.is_due());
        checker.check_with_config(|_, _| HealthCheckProbe {
            server_id: String::new(),
            healthy: true,
            latency_ms: 0.0,
            reason: String::new(),
        });
        assert!(!checker.is_due());
    }

    // ── CircuitBreaker ───────────────────────────────────────────────────────

    // 31. New circuit breaker starts Closed
    #[test]
    fn test_circuit_breaker_starts_closed() {
        let server = Arc::new(OriginServer::new("s", "http://s", 1, 0));
        let cb = CircuitBreaker::new(server, Duration::from_secs(30));
        assert_eq!(cb.state(), CircuitBreakerState::Closed);
    }

    // 32. trip() moves Closed → Open
    #[test]
    fn test_circuit_breaker_trip_opens() {
        let server = Arc::new(OriginServer::new("s", "http://s", 1, 0));
        let cb = CircuitBreaker::new(server, Duration::from_secs(30));
        cb.trip();
        assert_eq!(cb.state(), CircuitBreakerState::Open);
    }

    // 33. check_health_and_maybe_recover does not transition before timeout
    #[test]
    fn test_circuit_breaker_no_premature_recovery() {
        let server = Arc::new(OriginServer::new("s", "http://s", 1, 0));
        // Very long timeout so it won't elapse during the test.
        let cb = CircuitBreaker::new(server, Duration::from_secs(3600));
        cb.trip();
        assert_eq!(cb.state(), CircuitBreakerState::Open);
        let new_state = cb.check_health_and_maybe_recover();
        assert_eq!(new_state, CircuitBreakerState::Open);
    }

    // 34. check_health_and_maybe_recover transitions Open → HalfOpen after timeout
    #[test]
    fn test_circuit_breaker_recovers_to_half_open() {
        let server = Arc::new(OriginServer::new("s", "http://s", 1, 0));
        // Zero-duration timeout means it's always elapsed.
        let cb = CircuitBreaker::new(server, Duration::from_secs(0));
        cb.trip();
        // Tiny sleep to ensure elapsed > 0
        std::thread::sleep(Duration::from_millis(1));
        let new_state = cb.check_health_and_maybe_recover();
        assert_eq!(new_state, CircuitBreakerState::HalfOpen);
    }

    // 35. record_success on HalfOpen closes the breaker
    #[test]
    fn test_circuit_breaker_half_open_to_closed_on_success() {
        let server = Arc::new(OriginServer::new("s", "http://s", 1, 0));
        let cb = CircuitBreaker::new(server, Duration::from_secs(0));
        cb.trip();
        std::thread::sleep(Duration::from_millis(1));
        cb.check_health_and_maybe_recover(); // → HalfOpen
        cb.record_success(20.0);
        assert_eq!(cb.state(), CircuitBreakerState::Closed);
    }

    // 36. record_failure on HalfOpen re-opens the breaker
    #[test]
    fn test_circuit_breaker_half_open_to_open_on_failure() {
        let server = Arc::new(OriginServer::new("s", "http://s", 1, 0));
        let cb = CircuitBreaker::new(server, Duration::from_secs(0));
        cb.trip();
        std::thread::sleep(Duration::from_millis(1));
        cb.check_health_and_maybe_recover(); // → HalfOpen
                                             // Drive failures until server marks itself unhealthy
        for _ in 0..3 {
            cb.record_failure();
        }
        assert_eq!(cb.state(), CircuitBreakerState::Open);
    }

    // 37. reset() returns breaker to Closed and server to healthy
    #[test]
    fn test_circuit_breaker_reset() {
        let server = Arc::new(OriginServer::new("s", "http://s", 1, 0));
        let cb = CircuitBreaker::new(Arc::clone(&server), Duration::from_secs(30));
        cb.trip();
        assert_eq!(cb.state(), CircuitBreakerState::Open);
        cb.reset();
        assert_eq!(cb.state(), CircuitBreakerState::Closed);
        assert!(server.is_healthy());
    }
}

// ─── ConnectionPool ───────────────────────────────────────────────────────────

/// A keep-alive connection returned from / returned to [`ConnectionPool`].
pub struct PooledConn {
    /// Origin server identifier this connection belongs to.
    pub origin_id: String,
    /// Instant the connection was originally created.
    pub created_at: std::time::Instant,
}

impl PooledConn {
    fn new(origin_id: impl Into<String>) -> Self {
        Self {
            origin_id: origin_id.into(),
            created_at: std::time::Instant::now(),
        }
    }
}

/// Simulated keep-alive connection pool with idle-slot reuse.
///
/// Tracks how many connections were reused from the idle pool ([`reused`])
/// versus freshly created ([`created`]).
///
/// [`reused`]: ConnectionPool::reused
/// [`created`]: ConnectionPool::created
pub struct ConnectionPool {
    idle: Vec<PooledConn>,
    max_idle: usize,
    /// Number of connections reused from the idle pool.
    pub reused: u64,
    /// Number of new connections created (no idle slot matched).
    pub created: u64,
}

impl ConnectionPool {
    /// Create a new pool that retains at most `max_idle` idle connections.
    pub fn new(max_idle: usize) -> Self {
        Self {
            idle: Vec::new(),
            max_idle,
            reused: 0,
            created: 0,
        }
    }

    /// Acquire a connection for `origin_id`.
    ///
    /// If a matching idle connection exists it is popped (last-in, first-out)
    /// and `reused` is incremented.  Otherwise a new connection is created and
    /// `created` is incremented.
    pub fn acquire(&mut self, origin_id: &str) -> PooledConn {
        // Search from the back for efficiency (pop is O(1)).
        if let Some(pos) = self.idle.iter().rposition(|c| c.origin_id == origin_id) {
            let conn = self.idle.remove(pos);
            self.reused += 1;
            conn
        } else {
            self.created += 1;
            PooledConn::new(origin_id)
        }
    }

    /// Return a connection to the idle pool.
    ///
    /// If the pool is already at `max_idle` capacity the connection is dropped.
    pub fn release(&mut self, conn: PooledConn) {
        if self.idle.len() < self.max_idle {
            self.idle.push(conn);
        }
        // Exceeding capacity: conn is silently dropped (keep-alive close).
    }

    /// Number of connections currently in the idle pool.
    pub fn idle_len(&self) -> usize {
        self.idle.len()
    }
}

// ─── ConnectionPool tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod pool_tests {
    use super::*;

    // 5. acquire → release → acquire reuses the idle connection
    #[test]
    fn pool_reuses_idle_conn() {
        let mut pool = ConnectionPool::new(4);
        let conn = pool.acquire("origin-a");
        assert_eq!(pool.created, 1);
        assert_eq!(pool.reused, 0);

        pool.release(conn);

        let _conn2 = pool.acquire("origin-a");
        assert_eq!(pool.reused, 1, "second acquire should reuse idle conn");
        assert_eq!(pool.created, 1, "created should still be 1");
    }

    // 6. max_idle is respected — excess released connections are dropped
    #[test]
    fn pool_respects_max_idle() {
        let mut pool = ConnectionPool::new(1);
        let c1 = pool.acquire("origin-b");
        let c2 = pool.acquire("origin-b");

        pool.release(c1);
        pool.release(c2); // pool already full — should be dropped

        assert_eq!(pool.idle_len(), 1, "only 1 idle slot allowed");
    }

    // 7. released conn's origin_id matches what is acquired next
    #[test]
    fn pool_acquire_after_release_same_origin() {
        let mut pool = ConnectionPool::new(4);
        let conn = pool.acquire("origin-c");
        pool.release(conn);

        let reused = pool.acquire("origin-c");
        assert_eq!(
            reused.origin_id, "origin-c",
            "reused conn should have the same origin_id"
        );
    }
}
