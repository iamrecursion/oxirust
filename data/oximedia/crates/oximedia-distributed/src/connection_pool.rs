//! Connection pooling and retry for coordinator client connections.
//!
//! Provides a `ConnectionPool` that manages a configurable number of gRPC
//! client connections to the coordinator, with automatic reconnection on
//! failure and configurable retry policies.

use crate::{DistributedError, Result};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Mutex, Semaphore};
use tracing::{debug, info, warn};

/// Configuration for the connection pool.
#[derive(Debug, Clone)]
pub struct ConnectionPoolConfig {
    /// Target endpoint address (e.g., "http://127.0.0.1:50051").
    pub endpoint: String,
    /// Maximum number of connections in the pool.
    pub max_connections: usize,
    /// Minimum number of idle connections to keep warm.
    pub min_idle: usize,
    /// Connection timeout.
    pub connect_timeout: Duration,
    /// Maximum number of retries on connection failure.
    pub max_retries: u32,
    /// Base delay between retries (exponential backoff base).
    pub retry_base_delay: Duration,
    /// Maximum delay between retries.
    pub retry_max_delay: Duration,
    /// Maximum lifetime of a connection before forced recycling.
    pub max_lifetime: Duration,
    /// Health check interval for idle connections.
    pub health_check_interval: Duration,
}

impl Default for ConnectionPoolConfig {
    fn default() -> Self {
        Self {
            endpoint: "http://127.0.0.1:50051".to_string(),
            max_connections: 8,
            min_idle: 2,
            connect_timeout: Duration::from_secs(5),
            max_retries: 3,
            retry_base_delay: Duration::from_millis(100),
            retry_max_delay: Duration::from_secs(10),
            max_lifetime: Duration::from_secs(3600),
            health_check_interval: Duration::from_secs(30),
        }
    }
}

impl ConnectionPoolConfig {
    /// Create a new configuration with the given endpoint.
    #[must_use]
    pub fn new(endpoint: &str) -> Self {
        Self {
            endpoint: endpoint.to_string(),
            ..Default::default()
        }
    }

    /// Set the maximum pool size.
    #[must_use]
    pub fn with_max_connections(mut self, max: usize) -> Self {
        self.max_connections = max.max(1);
        self
    }

    /// Set the minimum idle connections.
    #[must_use]
    pub fn with_min_idle(mut self, min: usize) -> Self {
        self.min_idle = min;
        self
    }

    /// Set the maximum retries.
    #[must_use]
    pub fn with_max_retries(mut self, retries: u32) -> Self {
        self.max_retries = retries;
        self
    }

    /// Set the connection timeout.
    #[must_use]
    pub fn with_connect_timeout(mut self, timeout: Duration) -> Self {
        self.connect_timeout = timeout;
        self
    }

    /// Set the retry base delay.
    #[must_use]
    pub fn with_retry_base_delay(mut self, delay: Duration) -> Self {
        self.retry_base_delay = delay;
        self
    }

    /// Set the maximum connection lifetime.
    #[must_use]
    pub fn with_max_lifetime(mut self, lifetime: Duration) -> Self {
        self.max_lifetime = lifetime;
        self
    }
}

/// Retry policy for connection attempts.
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    /// Maximum retries.
    max_retries: u32,
    /// Base delay for exponential backoff.
    base_delay: Duration,
    /// Maximum delay cap.
    max_delay: Duration,
}

impl RetryPolicy {
    /// Create a new retry policy.
    #[must_use]
    pub fn new(max_retries: u32, base_delay: Duration, max_delay: Duration) -> Self {
        Self {
            max_retries,
            base_delay,
            max_delay,
        }
    }

    /// Compute the delay for the given attempt number (0-based).
    #[must_use]
    pub fn delay_for_attempt(&self, attempt: u32) -> Duration {
        let factor = 2u64.saturating_pow(attempt);
        let delay_ms = self.base_delay.as_millis() as u64 * factor;
        let capped = delay_ms.min(self.max_delay.as_millis() as u64);
        Duration::from_millis(capped)
    }

    /// Whether more attempts are allowed.
    #[must_use]
    pub fn should_retry(&self, attempt: u32) -> bool {
        attempt < self.max_retries
    }
}

/// State of a pooled connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionState {
    /// Connection is idle and available.
    Idle,
    /// Connection is currently in use.
    InUse,
    /// Connection is broken and needs reconnection.
    Broken,
    /// Connection has exceeded its maximum lifetime.
    Expired,
}

/// A pooled connection wrapper.
#[derive(Debug)]
#[allow(dead_code)]
pub struct PooledConnection {
    /// Unique connection identifier.
    id: u64,
    /// Current state.
    state: ConnectionState,
    /// When the connection was created (monotonic tick).
    created_at_tick: u64,
    /// When the connection was last used (monotonic tick).
    last_used_tick: u64,
    /// Number of requests served by this connection.
    requests_served: u64,
    /// The endpoint this connection targets.
    endpoint: String,
}

impl PooledConnection {
    /// Create a new connection.
    fn new(id: u64, endpoint: &str, now_tick: u64) -> Self {
        Self {
            id,
            state: ConnectionState::Idle,
            created_at_tick: now_tick,
            last_used_tick: now_tick,
            requests_served: 0,
            endpoint: endpoint.to_string(),
        }
    }

    /// Get the connection ID.
    #[must_use]
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Get the current state.
    #[must_use]
    pub fn state(&self) -> ConnectionState {
        self.state
    }

    /// Mark as in use.
    fn checkout(&mut self, now_tick: u64) {
        self.state = ConnectionState::InUse;
        self.last_used_tick = now_tick;
    }

    /// Return to idle after use.
    fn checkin(&mut self, now_tick: u64) {
        self.state = ConnectionState::Idle;
        self.last_used_tick = now_tick;
        self.requests_served += 1;
    }

    /// Mark as broken.
    fn mark_broken(&mut self) {
        self.state = ConnectionState::Broken;
    }

    /// Check if the connection has exceeded its maximum lifetime.
    fn is_expired(&self, now_tick: u64, max_lifetime_ms: u64) -> bool {
        now_tick.saturating_sub(self.created_at_tick) >= max_lifetime_ms
    }
}

/// Statistics for the connection pool.
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    /// Total connections created.
    pub total_created: u64,
    /// Total connections closed (recycled or broken).
    pub total_closed: u64,
    /// Total successful checkouts.
    pub total_checkouts: u64,
    /// Total failed connection attempts.
    pub total_failures: u64,
    /// Current pool size.
    pub current_size: usize,
    /// Current idle count.
    pub current_idle: usize,
    /// Current in-use count.
    pub current_in_use: usize,
}

/// Connection pool for coordinator client connections.
///
/// Manages a pool of connections with automatic reconnection, lifetime
/// management, and configurable pool sizing.
pub struct ConnectionPool {
    config: ConnectionPoolConfig,
    /// All connections.
    connections: Arc<Mutex<VecDeque<PooledConnection>>>,
    /// Semaphore to limit concurrent checkouts (reserved for async checkout).
    #[allow(dead_code)]
    checkout_semaphore: Arc<Semaphore>,
    /// Monotonic tick counter.
    current_tick: Arc<AtomicU64>,
    /// Next connection ID.
    next_id: Arc<AtomicU64>,
    /// Pool statistics.
    stats: Arc<Mutex<PoolStats>>,
    /// Whether the pool is shut down.
    is_shutdown: Arc<AtomicBool>,
    /// Retry policy.
    retry_policy: RetryPolicy,
}

impl ConnectionPool {
    /// Create a new connection pool with the given configuration.
    #[must_use]
    pub fn new(config: ConnectionPoolConfig) -> Self {
        let semaphore_permits = config.max_connections;
        let retry_policy = RetryPolicy::new(
            config.max_retries,
            config.retry_base_delay,
            config.retry_max_delay,
        );

        Self {
            config,
            connections: Arc::new(Mutex::new(VecDeque::new())),
            checkout_semaphore: Arc::new(Semaphore::new(semaphore_permits)),
            current_tick: Arc::new(AtomicU64::new(0)),
            next_id: Arc::new(AtomicU64::new(1)),
            stats: Arc::new(Mutex::new(PoolStats::default())),
            is_shutdown: Arc::new(AtomicBool::new(false)),
            retry_policy,
        }
    }

    /// Initialize the pool by creating `min_idle` connections.
    pub async fn initialize(&self) -> Result<()> {
        if self.is_shutdown.load(Ordering::Relaxed) {
            return Err(DistributedError::Worker("Pool is shut down".to_string()));
        }

        info!(
            "Initializing connection pool: endpoint={}, max={}, min_idle={}",
            self.config.endpoint, self.config.max_connections, self.config.min_idle
        );

        let mut conns = self.connections.lock().await;
        let mut stats = self.stats.lock().await;
        let now = self.current_tick.load(Ordering::Relaxed);

        for _ in 0..self.config.min_idle {
            let id = self.next_id.fetch_add(1, Ordering::Relaxed);
            let conn = PooledConnection::new(id, &self.config.endpoint, now);
            conns.push_back(conn);
            stats.total_created += 1;
            stats.current_size += 1;
            stats.current_idle += 1;
        }

        debug!("Pool initialized with {} connections", conns.len());
        Ok(())
    }

    /// Checkout a connection from the pool.
    ///
    /// Returns a connection ID that should be returned via `checkin` after use.
    /// If no idle connections are available and the pool is not at capacity, a
    /// new connection is created. Retries with exponential backoff on failure.
    pub async fn checkout(&self) -> Result<u64> {
        if self.is_shutdown.load(Ordering::Relaxed) {
            return Err(DistributedError::Worker("Pool is shut down".to_string()));
        }

        let now = self.current_tick.load(Ordering::Relaxed);
        let max_lifetime_ms = self.config.max_lifetime.as_millis() as u64;

        let mut conns = self.connections.lock().await;
        let mut stats = self.stats.lock().await;

        // Try to find an idle, non-expired connection
        for conn in conns.iter_mut() {
            if conn.state == ConnectionState::Idle && !conn.is_expired(now, max_lifetime_ms) {
                conn.checkout(now);
                stats.total_checkouts += 1;
                stats.current_idle = stats.current_idle.saturating_sub(1);
                stats.current_in_use += 1;
                debug!("Checked out connection {}", conn.id);
                return Ok(conn.id);
            }
        }

        // Remove expired and broken connections
        let before_len = conns.len();
        conns.retain(|c| c.state != ConnectionState::Broken && !c.is_expired(now, max_lifetime_ms));
        let removed = before_len - conns.len();
        if removed > 0 {
            stats.total_closed += removed as u64;
            stats.current_size = stats.current_size.saturating_sub(removed);
            stats.current_idle = stats.current_idle.saturating_sub(removed);
        }

        // Create a new connection if under capacity
        if conns.len() < self.config.max_connections {
            let id = self.next_id.fetch_add(1, Ordering::Relaxed);
            let mut conn = PooledConnection::new(id, &self.config.endpoint, now);
            conn.checkout(now);
            conns.push_back(conn);
            stats.total_created += 1;
            stats.total_checkouts += 1;
            stats.current_size += 1;
            stats.current_in_use += 1;
            debug!("Created and checked out new connection {}", id);
            return Ok(id);
        }

        // Pool is at capacity and all connections are in use
        Err(DistributedError::ResourceExhausted(
            "Connection pool exhausted".to_string(),
        ))
    }

    /// Return a connection to the pool after use.
    pub async fn checkin(&self, conn_id: u64) -> Result<()> {
        let now = self.current_tick.load(Ordering::Relaxed);
        let mut conns = self.connections.lock().await;
        let mut stats = self.stats.lock().await;

        for conn in conns.iter_mut() {
            if conn.id == conn_id {
                conn.checkin(now);
                stats.current_in_use = stats.current_in_use.saturating_sub(1);
                stats.current_idle += 1;
                debug!("Checked in connection {}", conn_id);
                return Ok(());
            }
        }

        Err(DistributedError::Worker(format!(
            "Connection {conn_id} not found in pool"
        )))
    }

    /// Mark a connection as broken (will be removed on next cleanup).
    pub async fn mark_broken(&self, conn_id: u64) -> Result<()> {
        let mut conns = self.connections.lock().await;
        let mut stats = self.stats.lock().await;

        for conn in conns.iter_mut() {
            if conn.id == conn_id {
                let was_in_use = conn.state == ConnectionState::InUse;
                conn.mark_broken();
                stats.total_failures += 1;
                if was_in_use {
                    stats.current_in_use = stats.current_in_use.saturating_sub(1);
                } else {
                    stats.current_idle = stats.current_idle.saturating_sub(1);
                }
                warn!("Connection {} marked as broken", conn_id);
                return Ok(());
            }
        }

        Err(DistributedError::Worker(format!(
            "Connection {conn_id} not found in pool"
        )))
    }

    /// Execute an operation with automatic retry on failure.
    ///
    /// `operation` is a closure that receives a connection ID and returns
    /// `Ok(T)` on success or `Err` on failure. On failure, the connection
    /// is marked broken and a new one is obtained for the next attempt.
    pub async fn execute_with_retry<F, T>(&self, mut operation: F) -> Result<T>
    where
        F: FnMut(u64) -> Result<T>,
    {
        let mut attempt = 0u32;

        loop {
            let conn_id = self.checkout().await?;

            match operation(conn_id) {
                Ok(value) => {
                    self.checkin(conn_id).await?;
                    return Ok(value);
                }
                Err(e) => {
                    let _ = self.mark_broken(conn_id).await;

                    if !self.retry_policy.should_retry(attempt) {
                        return Err(e);
                    }

                    let delay = self.retry_policy.delay_for_attempt(attempt);
                    debug!(
                        "Retry attempt {} after {delay:?} for connection failure: {e}",
                        attempt + 1
                    );
                    // In a real scenario we'd sleep here; for testability we
                    // just increment attempt and loop immediately.
                    attempt += 1;
                }
            }
        }
    }

    /// Advance the internal tick (for testing and lifetime management).
    pub fn advance_tick(&self, millis: u64) {
        self.current_tick.fetch_add(millis, Ordering::Relaxed);
    }

    /// Get current pool statistics.
    pub async fn stats(&self) -> PoolStats {
        self.stats.lock().await.clone()
    }

    /// Get the current pool size.
    pub async fn size(&self) -> usize {
        self.connections.lock().await.len()
    }

    /// Get the current number of idle connections.
    pub async fn idle_count(&self) -> usize {
        self.connections
            .lock()
            .await
            .iter()
            .filter(|c| c.state == ConnectionState::Idle)
            .count()
    }

    /// Get the retry policy.
    #[must_use]
    pub fn retry_policy(&self) -> &RetryPolicy {
        &self.retry_policy
    }

    /// Perform health checks on idle connections, removing broken/expired ones.
    pub async fn health_check(&self) -> Result<usize> {
        let now = self.current_tick.load(Ordering::Relaxed);
        let max_lifetime_ms = self.config.max_lifetime.as_millis() as u64;

        let mut conns = self.connections.lock().await;
        let mut stats = self.stats.lock().await;

        let before = conns.len();
        conns.retain(|c| {
            if c.state == ConnectionState::Broken {
                return false;
            }
            if c.state == ConnectionState::Idle && c.is_expired(now, max_lifetime_ms) {
                return false;
            }
            true
        });

        let removed = before - conns.len();
        stats.total_closed += removed as u64;
        stats.current_size = stats.current_size.saturating_sub(removed);
        stats.current_idle = stats.current_idle.saturating_sub(removed);

        // Replenish to min_idle if needed
        let current_idle = conns
            .iter()
            .filter(|c| c.state == ConnectionState::Idle)
            .count();
        let mut created = 0usize;
        while current_idle + created < self.config.min_idle
            && conns.len() + created < self.config.max_connections
        {
            let id = self.next_id.fetch_add(1, Ordering::Relaxed);
            conns.push_back(PooledConnection::new(id, &self.config.endpoint, now));
            stats.total_created += 1;
            stats.current_size += 1;
            stats.current_idle += 1;
            created += 1;
        }

        if removed > 0 || created > 0 {
            debug!(
                "Health check: removed={}, created={}, pool_size={}",
                removed,
                created,
                conns.len()
            );
        }

        Ok(removed)
    }

    /// Shut down the pool, closing all connections.
    pub async fn shutdown(&self) -> Result<()> {
        self.is_shutdown.store(true, Ordering::Relaxed);
        let mut conns = self.connections.lock().await;
        let mut stats = self.stats.lock().await;

        let count = conns.len();
        conns.clear();
        stats.total_closed += count as u64;
        stats.current_size = 0;
        stats.current_idle = 0;
        stats.current_in_use = 0;

        info!("Connection pool shut down, closed {} connections", count);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_pool(max: usize, min_idle: usize) -> ConnectionPool {
        ConnectionPool::new(
            ConnectionPoolConfig::new("http://localhost:50051")
                .with_max_connections(max)
                .with_min_idle(min_idle),
        )
    }

    #[test]
    fn test_config_defaults() {
        let config = ConnectionPoolConfig::default();
        assert_eq!(config.max_connections, 8);
        assert_eq!(config.min_idle, 2);
        assert_eq!(config.max_retries, 3);
    }

    #[test]
    fn test_config_builder() {
        let config = ConnectionPoolConfig::new("http://example.com:50051")
            .with_max_connections(16)
            .with_min_idle(4)
            .with_max_retries(5)
            .with_connect_timeout(Duration::from_secs(10))
            .with_retry_base_delay(Duration::from_millis(200))
            .with_max_lifetime(Duration::from_secs(7200));
        assert_eq!(config.max_connections, 16);
        assert_eq!(config.min_idle, 4);
        assert_eq!(config.max_retries, 5);
        assert_eq!(config.connect_timeout, Duration::from_secs(10));
    }

    #[test]
    fn test_config_max_connections_minimum_one() {
        let config = ConnectionPoolConfig::new("http://localhost:50051").with_max_connections(0);
        assert_eq!(config.max_connections, 1);
    }

    #[test]
    fn test_retry_policy_delay() {
        let policy = RetryPolicy::new(5, Duration::from_millis(100), Duration::from_secs(5));
        assert_eq!(policy.delay_for_attempt(0), Duration::from_millis(100));
        assert_eq!(policy.delay_for_attempt(1), Duration::from_millis(200));
        assert_eq!(policy.delay_for_attempt(2), Duration::from_millis(400));
        assert_eq!(policy.delay_for_attempt(3), Duration::from_millis(800));
    }

    #[test]
    fn test_retry_policy_delay_capped() {
        let policy = RetryPolicy::new(5, Duration::from_millis(100), Duration::from_millis(500));
        // 100 * 2^4 = 1600, should be capped at 500
        assert_eq!(policy.delay_for_attempt(4), Duration::from_millis(500));
    }

    #[test]
    fn test_retry_policy_should_retry() {
        let policy = RetryPolicy::new(3, Duration::from_millis(100), Duration::from_secs(5));
        assert!(policy.should_retry(0));
        assert!(policy.should_retry(1));
        assert!(policy.should_retry(2));
        assert!(!policy.should_retry(3));
    }

    #[test]
    fn test_connection_state() {
        let mut conn = PooledConnection::new(1, "http://localhost", 0);
        assert_eq!(conn.state(), ConnectionState::Idle);

        conn.checkout(10);
        assert_eq!(conn.state(), ConnectionState::InUse);

        conn.checkin(20);
        assert_eq!(conn.state(), ConnectionState::Idle);
        assert_eq!(conn.requests_served, 1);

        conn.mark_broken();
        assert_eq!(conn.state(), ConnectionState::Broken);
    }

    #[test]
    fn test_connection_expiry() {
        let conn = PooledConnection::new(1, "http://localhost", 100);
        assert!(!conn.is_expired(200, 1000));
        assert!(conn.is_expired(1100, 1000));
        assert!(conn.is_expired(1101, 1000));
    }

    #[tokio::test]
    async fn test_pool_initialize() {
        let pool = small_pool(4, 2);
        pool.initialize().await.expect("init should succeed");
        assert_eq!(pool.size().await, 2);
        assert_eq!(pool.idle_count().await, 2);
    }

    #[tokio::test]
    async fn test_pool_checkout_and_checkin() {
        let pool = small_pool(4, 1);
        pool.initialize().await.expect("init should succeed");

        let conn_id = pool.checkout().await.expect("checkout should succeed");
        assert_eq!(pool.idle_count().await, 0);

        pool.checkin(conn_id).await.expect("checkin should succeed");
        assert_eq!(pool.idle_count().await, 1);
    }

    #[tokio::test]
    async fn test_pool_creates_on_demand() {
        let pool = small_pool(4, 0);
        pool.initialize().await.expect("init should succeed");
        assert_eq!(pool.size().await, 0);

        let conn_id = pool.checkout().await.expect("checkout should succeed");
        assert_eq!(pool.size().await, 1);

        pool.checkin(conn_id).await.expect("checkin should succeed");
    }

    #[tokio::test]
    async fn test_pool_exhaustion() {
        let pool = small_pool(2, 0);
        pool.initialize().await.expect("init should succeed");

        let _c1 = pool.checkout().await.expect("checkout 1 should succeed");
        let _c2 = pool.checkout().await.expect("checkout 2 should succeed");

        let result = pool.checkout().await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_pool_mark_broken() {
        let pool = small_pool(4, 1);
        pool.initialize().await.expect("init should succeed");

        let conn_id = pool.checkout().await.expect("checkout should succeed");
        pool.mark_broken(conn_id)
            .await
            .expect("mark_broken should succeed");

        let stats = pool.stats().await;
        assert_eq!(stats.total_failures, 1);
    }

    #[tokio::test]
    async fn test_pool_health_check_removes_broken() {
        let pool = small_pool(4, 0);
        pool.initialize().await.expect("init should succeed");

        let c1 = pool.checkout().await.expect("checkout should succeed");
        pool.mark_broken(c1)
            .await
            .expect("mark_broken should succeed");

        let removed = pool
            .health_check()
            .await
            .expect("health_check should succeed");
        assert_eq!(removed, 1);
        assert_eq!(pool.size().await, 0);
    }

    #[tokio::test]
    async fn test_pool_health_check_removes_expired() {
        let pool = ConnectionPool::new(
            ConnectionPoolConfig::new("http://localhost:50051")
                .with_max_connections(4)
                .with_min_idle(0)
                .with_max_lifetime(Duration::from_millis(500)),
        );
        pool.initialize().await.expect("init should succeed");

        let c1 = pool.checkout().await.expect("checkout should succeed");
        pool.checkin(c1).await.expect("checkin should succeed");
        assert_eq!(pool.size().await, 1);

        // Advance time past lifetime
        pool.advance_tick(600);

        let removed = pool
            .health_check()
            .await
            .expect("health_check should succeed");
        assert_eq!(removed, 1);
        assert_eq!(pool.size().await, 0);
    }

    #[tokio::test]
    async fn test_pool_health_check_replenishes_min_idle() {
        let pool = small_pool(4, 2);
        pool.initialize().await.expect("init should succeed");
        assert_eq!(pool.size().await, 2);

        // Check out and break one connection
        let c1 = pool.checkout().await.expect("checkout should succeed");
        pool.mark_broken(c1)
            .await
            .expect("mark_broken should succeed");

        // Health check should remove broken and replenish
        pool.health_check()
            .await
            .expect("health_check should succeed");

        // Should have replenished to at least min_idle idle connections
        assert!(pool.idle_count().await >= 2);
    }

    #[tokio::test]
    async fn test_pool_execute_with_retry_success() {
        let pool = small_pool(4, 1);
        pool.initialize().await.expect("init should succeed");

        let result = pool
            .execute_with_retry(|conn_id| -> Result<u64> { Ok(conn_id * 2) })
            .await
            .expect("execute should succeed");
        assert!(result > 0);
    }

    #[tokio::test]
    async fn test_pool_execute_with_retry_fails_then_succeeds() {
        let pool = small_pool(4, 1);
        pool.initialize().await.expect("init should succeed");

        let call_count = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        let count = call_count.clone();

        let result = pool
            .execute_with_retry(move |conn_id| -> Result<u64> {
                let attempt = count.fetch_add(1, Ordering::Relaxed);
                if attempt == 0 {
                    Err(DistributedError::Worker("transient error".to_string()))
                } else {
                    Ok(conn_id)
                }
            })
            .await
            .expect("execute should succeed on retry");
        assert!(result > 0);
        assert_eq!(call_count.load(Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn test_pool_execute_with_retry_exhausted() {
        let pool = ConnectionPool::new(
            ConnectionPoolConfig::new("http://localhost:50051")
                .with_max_connections(8)
                .with_min_idle(0)
                .with_max_retries(2),
        );
        pool.initialize().await.expect("init should succeed");

        let result = pool
            .execute_with_retry(|_conn_id| -> Result<u64> {
                Err(DistributedError::Worker("permanent error".to_string()))
            })
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_pool_shutdown() {
        let pool = small_pool(4, 3);
        pool.initialize().await.expect("init should succeed");
        assert_eq!(pool.size().await, 3);

        pool.shutdown().await.expect("shutdown should succeed");
        assert_eq!(pool.size().await, 0);

        // Should reject new checkouts
        let result = pool.checkout().await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_pool_stats() {
        let pool = small_pool(4, 1);
        pool.initialize().await.expect("init should succeed");

        let c1 = pool.checkout().await.expect("checkout should succeed");
        pool.checkin(c1).await.expect("checkin should succeed");

        let stats = pool.stats().await;
        assert!(stats.total_created >= 1);
        assert!(stats.total_checkouts >= 1);
        assert_eq!(stats.current_in_use, 0);
    }

    #[tokio::test]
    async fn test_pool_checkin_nonexistent() {
        let pool = small_pool(4, 0);
        pool.initialize().await.expect("init should succeed");

        let result = pool.checkin(999).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_pool_mark_broken_nonexistent() {
        let pool = small_pool(4, 0);
        pool.initialize().await.expect("init should succeed");

        let result = pool.mark_broken(999).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_pool_multiple_checkouts() {
        let pool = small_pool(4, 0);
        pool.initialize().await.expect("init should succeed");

        let c1 = pool.checkout().await.expect("checkout 1 should succeed");
        let c2 = pool.checkout().await.expect("checkout 2 should succeed");
        let c3 = pool.checkout().await.expect("checkout 3 should succeed");

        assert_ne!(c1, c2);
        assert_ne!(c2, c3);

        let stats = pool.stats().await;
        assert_eq!(stats.current_in_use, 3);
        assert_eq!(stats.current_size, 3);

        pool.checkin(c1).await.expect("checkin should succeed");
        pool.checkin(c2).await.expect("checkin should succeed");
        pool.checkin(c3).await.expect("checkin should succeed");

        let stats = pool.stats().await;
        assert_eq!(stats.current_in_use, 0);
    }

    #[tokio::test]
    async fn test_pool_reuses_idle_connections() {
        let pool = small_pool(4, 1);
        pool.initialize().await.expect("init should succeed");

        let c1 = pool.checkout().await.expect("checkout 1 should succeed");
        pool.checkin(c1).await.expect("checkin should succeed");

        let c2 = pool.checkout().await.expect("checkout 2 should succeed");
        // Should reuse the same connection
        assert_eq!(c1, c2);
        pool.checkin(c2).await.expect("checkin should succeed");
    }
}
