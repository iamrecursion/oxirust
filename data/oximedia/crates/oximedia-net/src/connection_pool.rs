#![allow(dead_code)]
//! Connection pooling for persistent network connections.
//!
//! Provides a [`ConnectionPool`] that manages reusable connections keyed by
//! host/port pairs. Idle connections are reaped after a configurable timeout,
//! and the pool enforces per-host and global capacity limits.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// Unique identifier for a connection in the pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConnectionId(u64);

impl ConnectionId {
    /// Creates a new connection identifier from a raw value.
    #[must_use]
    pub const fn new(raw: u64) -> Self {
        Self(raw)
    }

    /// Returns the raw numeric identifier.
    #[must_use]
    pub const fn raw(self) -> u64 {
        self.0
    }
}

/// Health status of a pooled connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionHealth {
    /// Connection is healthy and ready for use.
    Healthy,
    /// Connection is degraded but may still work.
    Degraded,
    /// Connection has failed and should be removed.
    Failed,
    /// Connection health is unknown (not yet checked).
    Unknown,
}

impl ConnectionHealth {
    /// Returns `true` when the connection may be used.
    #[must_use]
    pub fn is_usable(self) -> bool {
        matches!(self, Self::Healthy | Self::Degraded | Self::Unknown)
    }
}

/// A single connection tracked by the pool.
#[derive(Debug, Clone)]
pub struct PooledConnection {
    /// Unique identifier for this connection.
    pub id: ConnectionId,
    /// Host this connection targets.
    pub host: String,
    /// Port number.
    pub port: u16,
    /// When the connection was created.
    pub created_at: Instant,
    /// When the connection was last used.
    pub last_used: Instant,
    /// Number of times this connection has been checked out.
    pub use_count: u64,
    /// Current health status.
    pub health: ConnectionHealth,
}

impl PooledConnection {
    /// Creates a new pooled connection.
    #[must_use]
    pub fn new(id: ConnectionId, host: String, port: u16) -> Self {
        let now = Instant::now();
        Self {
            id,
            host,
            port,
            created_at: now,
            last_used: now,
            use_count: 0,
            health: ConnectionHealth::Unknown,
        }
    }

    /// Returns the idle duration since last use.
    #[must_use]
    pub fn idle_duration(&self) -> Duration {
        self.last_used.elapsed()
    }

    /// Returns the total age of the connection.
    #[must_use]
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// Marks the connection as actively used.
    pub fn touch(&mut self) {
        self.last_used = Instant::now();
        self.use_count += 1;
    }
}

/// Configuration for the connection pool.
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Maximum total connections across all hosts.
    pub max_total: usize,
    /// Maximum connections per host.
    pub max_per_host: usize,
    /// How long an idle connection survives before reaping.
    pub idle_timeout: Duration,
    /// Maximum lifetime of any connection regardless of activity.
    pub max_lifetime: Duration,
    /// Minimum number of idle connections to keep per host.
    pub min_idle_per_host: usize,
    /// Whether to validate connections before handing them out.
    pub validate_on_checkout: bool,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            max_total: 100,
            max_per_host: 10,
            idle_timeout: Duration::from_secs(90),
            max_lifetime: Duration::from_secs(600),
            min_idle_per_host: 1,
            validate_on_checkout: true,
        }
    }
}

/// Pool statistics snapshot.
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    /// Total connections currently in the pool.
    pub total_connections: usize,
    /// Connections that are idle (available).
    pub idle_connections: usize,
    /// Connections that are checked out (in use).
    pub active_connections: usize,
    /// Total number of successful checkouts.
    pub total_checkouts: u64,
    /// Total number of connections reaped for being idle.
    pub total_reaped: u64,
    /// Total number of failed health checks.
    pub total_health_failures: u64,
    /// Number of distinct hosts with connections.
    pub host_count: usize,
}

/// A connection pool keyed by `(host, port)`.
#[derive(Debug)]
pub struct ConnectionPool {
    /// Configuration for the pool.
    config: PoolConfig,
    /// Idle connections per host key.
    idle: HashMap<String, VecDeque<PooledConnection>>,
    /// Number of active (checked-out) connections per host key.
    active_counts: HashMap<String, usize>,
    /// Next connection id to assign.
    next_id: u64,
    /// Total checkout counter.
    total_checkouts: u64,
    /// Total reaped counter.
    total_reaped: u64,
    /// Total health failure counter.
    total_health_failures: u64,
}

impl ConnectionPool {
    /// Creates a new pool with the given configuration.
    #[must_use]
    pub fn new(config: PoolConfig) -> Self {
        Self {
            config,
            idle: HashMap::new(),
            active_counts: HashMap::new(),
            next_id: 1,
            total_checkouts: 0,
            total_reaped: 0,
            total_health_failures: 0,
        }
    }

    /// Creates a pool with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(PoolConfig::default())
    }

    /// Returns the pool configuration.
    #[must_use]
    pub fn config(&self) -> &PoolConfig {
        &self.config
    }

    /// Builds a key from host and port.
    fn host_key(host: &str, port: u16) -> String {
        format!("{host}:{port}")
    }

    /// Returns the total number of connections (idle + active).
    #[must_use]
    pub fn total_connections(&self) -> usize {
        let idle: usize = self.idle.values().map(|q| q.len()).sum();
        let active: usize = self.active_counts.values().sum();
        idle + active
    }

    /// Returns a snapshot of pool statistics.
    #[must_use]
    pub fn stats(&self) -> PoolStats {
        let idle_connections: usize = self.idle.values().map(|q| q.len()).sum();
        let active_connections: usize = self.active_counts.values().sum();
        PoolStats {
            total_connections: idle_connections + active_connections,
            idle_connections,
            active_connections,
            total_checkouts: self.total_checkouts,
            total_reaped: self.total_reaped,
            total_health_failures: self.total_health_failures,
            host_count: self.idle.len(),
        }
    }

    /// Tries to check out an idle connection for the given host/port.
    ///
    /// Returns `None` if no idle connection is available for that target.
    pub fn checkout(&mut self, host: &str, port: u16) -> Option<PooledConnection> {
        let key = Self::host_key(host, port);
        let queue = self.idle.get_mut(&key)?;

        // Find a usable connection (pop from front = oldest first).
        while let Some(mut conn) = queue.pop_front() {
            if conn.health.is_usable() && conn.idle_duration() < self.config.idle_timeout {
                conn.touch();
                self.total_checkouts += 1;
                *self.active_counts.entry(key).or_insert(0) += 1;
                return Some(conn);
            }
            // Connection is stale or unhealthy — discard silently.
            self.total_reaped += 1;
        }

        None
    }

    /// Returns a connection to the pool after use.
    pub fn checkin(&mut self, mut conn: PooledConnection) {
        let key = Self::host_key(&conn.host, conn.port);

        // Decrement active count.
        if let Some(count) = self.active_counts.get_mut(&key) {
            *count = count.saturating_sub(1);
        }

        // Only return to idle pool if healthy and within limits.
        let queue = self.idle.entry(key).or_default();
        if conn.health.is_usable()
            && (self.config.max_lifetime == Duration::ZERO || conn.age() < self.config.max_lifetime)
            && queue.len() < self.config.max_per_host
        {
            conn.health = ConnectionHealth::Healthy;
            queue.push_back(conn);
        }
    }

    /// Creates a new connection and adds it to the active set.
    ///
    /// Returns `None` if the pool has reached its total capacity.
    pub fn create(&mut self, host: &str, port: u16) -> Option<PooledConnection> {
        if self.total_connections() >= self.config.max_total {
            return None;
        }
        let key = Self::host_key(host, port);
        let idle_count = self.idle.get(&key).map_or(0, |q| q.len());
        let active_count = self.active_counts.get(&key).copied().unwrap_or(0);
        if idle_count + active_count >= self.config.max_per_host {
            return None;
        }

        let id = ConnectionId::new(self.next_id);
        self.next_id += 1;
        let mut conn = PooledConnection::new(id, host.to_owned(), port);
        conn.touch();
        *self.active_counts.entry(key).or_insert(0) += 1;
        self.total_checkouts += 1;
        Some(conn)
    }

    /// Reaps all idle connections that have exceeded the idle timeout.
    ///
    /// Returns the number of connections removed.
    pub fn reap_idle(&mut self) -> usize {
        let timeout = self.config.idle_timeout;
        let min_idle = self.config.min_idle_per_host;
        let mut total_removed = 0;

        for queue in self.idle.values_mut() {
            let before = queue.len();
            let mut kept = VecDeque::new();
            for conn in queue.drain(..) {
                if kept.len() < min_idle || conn.idle_duration() < timeout {
                    kept.push_back(conn);
                } else {
                    total_removed += 1;
                }
            }
            *queue = kept;
            let _ = before; // suppress unused
        }
        self.total_reaped += total_removed as u64;
        total_removed
    }

    /// Reaps connections that have exceeded the maximum lifetime.
    ///
    /// Returns the number of connections removed.
    pub fn reap_expired(&mut self) -> usize {
        let max_lifetime = self.config.max_lifetime;
        let mut total_removed = 0;

        for queue in self.idle.values_mut() {
            let before = queue.len();
            queue.retain(|conn| conn.age() < max_lifetime);
            total_removed += before - queue.len();
        }
        self.total_reaped += total_removed as u64;
        total_removed
    }

    /// Marks a connection as failed, preventing it from being reused.
    pub fn mark_failed(&mut self, id: ConnectionId) {
        for queue in self.idle.values_mut() {
            for conn in queue.iter_mut() {
                if conn.id == id {
                    conn.health = ConnectionHealth::Failed;
                    self.total_health_failures += 1;
                    return;
                }
            }
        }
    }

    /// Returns the number of idle connections for a given host/port.
    #[must_use]
    pub fn idle_count(&self, host: &str, port: u16) -> usize {
        let key = Self::host_key(host, port);
        self.idle.get(&key).map_or(0, |q| q.len())
    }

    /// Returns the number of active (checked-out) connections for a host/port.
    #[must_use]
    pub fn active_count(&self, host: &str, port: u16) -> usize {
        let key = Self::host_key(host, port);
        self.active_counts.get(&key).copied().unwrap_or(0)
    }

    /// Removes all connections for a specific host/port.
    ///
    /// Returns the number of idle connections removed.
    pub fn remove_host(&mut self, host: &str, port: u16) -> usize {
        let key = Self::host_key(host, port);
        let removed = self.idle.remove(&key).map_or(0, |q| q.len());
        self.active_counts.remove(&key);
        removed
    }

    /// Clears all connections from the pool.
    pub fn clear(&mut self) {
        self.idle.clear();
        self.active_counts.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_pool() -> ConnectionPool {
        ConnectionPool::with_defaults()
    }

    #[test]
    fn test_connection_id_round_trip() {
        let id = ConnectionId::new(42);
        assert_eq!(id.raw(), 42);
    }

    #[test]
    fn test_connection_health_usable() {
        assert!(ConnectionHealth::Healthy.is_usable());
        assert!(ConnectionHealth::Degraded.is_usable());
        assert!(ConnectionHealth::Unknown.is_usable());
        assert!(!ConnectionHealth::Failed.is_usable());
    }

    #[test]
    fn test_pooled_connection_touch() {
        let mut conn = PooledConnection::new(ConnectionId::new(1), "host".into(), 80);
        assert_eq!(conn.use_count, 0);
        conn.touch();
        assert_eq!(conn.use_count, 1);
        conn.touch();
        assert_eq!(conn.use_count, 2);
    }

    #[test]
    fn test_pool_config_defaults() {
        let cfg = PoolConfig::default();
        assert_eq!(cfg.max_total, 100);
        assert_eq!(cfg.max_per_host, 10);
        assert!(cfg.validate_on_checkout);
    }

    #[test]
    fn test_create_and_checkout_flow() {
        let mut pool = default_pool();

        // Create a connection.
        let conn = pool
            .create("example.com", 443)
            .expect("should succeed in test");
        assert_eq!(conn.host, "example.com");
        assert_eq!(conn.port, 443);
        assert_eq!(pool.active_count("example.com", 443), 1);

        // Return it.
        pool.checkin(conn);
        assert_eq!(pool.idle_count("example.com", 443), 1);
        assert_eq!(pool.active_count("example.com", 443), 0);

        // Check it out again.
        let conn2 = pool
            .checkout("example.com", 443)
            .expect("should succeed in test");
        assert_eq!(conn2.use_count, 2); // touched on create + checkout
        assert_eq!(pool.idle_count("example.com", 443), 0);
    }

    #[test]
    fn test_checkout_empty_returns_none() {
        let mut pool = default_pool();
        assert!(pool.checkout("unknown.host", 80).is_none());
    }

    #[test]
    fn test_pool_stats() {
        let mut pool = default_pool();
        let conn = pool.create("a.com", 80).expect("should succeed in test");
        let _conn2 = pool.create("b.com", 80).expect("should succeed in test");

        pool.checkin(conn);

        let stats = pool.stats();
        assert_eq!(stats.total_connections, 2);
        assert_eq!(stats.idle_connections, 1);
        assert_eq!(stats.active_connections, 1);
        assert_eq!(stats.total_checkouts, 2);
    }

    #[test]
    fn test_per_host_limit() {
        let cfg = PoolConfig {
            max_per_host: 2,
            ..PoolConfig::default()
        };
        let mut pool = ConnectionPool::new(cfg);

        assert!(pool.create("h.com", 80).is_some());
        assert!(pool.create("h.com", 80).is_some());
        assert!(pool.create("h.com", 80).is_none()); // limit reached
    }

    #[test]
    fn test_total_limit() {
        let cfg = PoolConfig {
            max_total: 2,
            max_per_host: 5,
            ..PoolConfig::default()
        };
        let mut pool = ConnectionPool::new(cfg);

        assert!(pool.create("a.com", 80).is_some());
        assert!(pool.create("b.com", 80).is_some());
        assert!(pool.create("c.com", 80).is_none()); // total limit
    }

    #[test]
    fn test_mark_failed() {
        let mut pool = default_pool();
        let conn = pool.create("h.com", 80).expect("should succeed in test");
        let id = conn.id;
        pool.checkin(conn);
        pool.mark_failed(id);

        // Checkout should skip failed connection.
        assert!(pool.checkout("h.com", 80).is_none());
        assert_eq!(pool.stats().total_health_failures, 1);
    }

    #[test]
    fn test_remove_host() {
        let mut pool = default_pool();
        let conn = pool.create("rm.com", 443).expect("should succeed in test");
        pool.checkin(conn);
        assert_eq!(pool.idle_count("rm.com", 443), 1);

        let removed = pool.remove_host("rm.com", 443);
        assert_eq!(removed, 1);
        assert_eq!(pool.idle_count("rm.com", 443), 0);
    }

    #[test]
    fn test_clear_pool() {
        let mut pool = default_pool();
        let c1 = pool.create("a.com", 80).expect("should succeed in test");
        pool.checkin(c1);
        let _c2 = pool.create("b.com", 80);

        pool.clear();
        assert_eq!(pool.total_connections(), 0);
    }

    #[test]
    fn test_reap_expired_with_zero_lifetime() {
        let cfg = PoolConfig {
            max_lifetime: Duration::ZERO,
            ..PoolConfig::default()
        };
        let mut pool = ConnectionPool::new(cfg);
        let conn = pool.create("h.com", 80).expect("should succeed in test");
        pool.checkin(conn);

        // All connections are already past zero lifetime.
        let reaped = pool.reap_expired();
        assert_eq!(reaped, 1);
        assert_eq!(pool.idle_count("h.com", 80), 0);
    }

    #[test]
    fn test_pooled_connection_idle_duration() {
        let conn = PooledConnection::new(ConnectionId::new(1), "h.com".into(), 80);
        // Idle duration should be very small immediately after creation.
        assert!(conn.idle_duration() < Duration::from_secs(1));
        assert!(conn.age() < Duration::from_secs(1));
    }
}

// =============================================================================
// Generic ConnectionPool<T>
// =============================================================================

/// Configuration for the generic pool.
#[derive(Debug, Clone)]
pub struct GenericPoolConfig {
    /// Maximum total items (idle + active) in the pool.
    pub max_size: usize,
    /// Minimum idle items to retain (not currently enforced on eviction, used
    /// as a hint for pre-population).
    pub min_idle: usize,
    /// Maximum age of an idle item in milliseconds before it is evicted.
    /// `0` means evict immediately (useful in tests).
    pub max_lifetime_ms: u64,
}

impl Default for GenericPoolConfig {
    fn default() -> Self {
        Self {
            max_size: 16,
            min_idle: 1,
            max_lifetime_ms: 300_000,
        }
    }
}

impl GenericPoolConfig {
    /// Create a new config.
    #[must_use]
    pub const fn new(max_size: usize, min_idle: usize, max_lifetime_ms: u64) -> Self {
        Self {
            max_size,
            min_idle,
            max_lifetime_ms,
        }
    }
}

/// A slot held in the generic pool, pairing the value with its creation timestamp.
struct GenericPoolSlot<T> {
    value: T,
    /// Milliseconds since UNIX_EPOCH at creation.
    created_ms: u64,
}

/// Return the current time as milliseconds since UNIX_EPOCH, or 0 on error.
fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// A generic pool that manages reusable values of type `T`.
///
/// Items are obtained via [`GenericPool::try_acquire`] and returned via
/// [`GenericPool::release`]. Idle items older than `max_lifetime_ms` are
/// evicted lazily on the next [`try_acquire`][GenericPool::try_acquire] call.
///
/// # Example
///
/// ```rust
/// use oximedia_net::connection_pool::{GenericPool, GenericPoolConfig};
///
/// let mut pool: GenericPool<String> = GenericPool::new(GenericPoolConfig::default());
/// pool.add("conn-1".to_owned()).ok();
/// let item = pool.try_acquire();
/// assert!(item.is_some());
/// ```
pub struct GenericPool<T> {
    config: GenericPoolConfig,
    idle: VecDeque<GenericPoolSlot<T>>,
    active_count: usize,
}

impl<T> GenericPool<T> {
    /// Create a new pool with the given configuration.
    #[must_use]
    pub fn new(config: GenericPoolConfig) -> Self {
        Self {
            config,
            idle: VecDeque::new(),
            active_count: 0,
        }
    }

    /// Create a pool with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(GenericPoolConfig::default())
    }

    /// Return the pool configuration.
    #[must_use]
    pub fn config(&self) -> &GenericPoolConfig {
        &self.config
    }

    /// Total items in the pool (idle + active).
    #[must_use]
    pub fn total(&self) -> usize {
        self.idle.len() + self.active_count
    }

    /// Number of idle items available for acquisition.
    #[must_use]
    pub fn idle_count(&self) -> usize {
        self.idle.len()
    }

    /// Number of items currently checked out (active).
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.active_count
    }

    /// Add a pre-created value to the idle pool.
    ///
    /// Returns `Err(value)` if the pool is at capacity.
    pub fn add(&mut self, value: T) -> Result<(), T> {
        if self.total() >= self.config.max_size {
            return Err(value);
        }
        self.idle.push_back(GenericPoolSlot {
            value,
            created_ms: now_ms(),
        });
        Ok(())
    }

    /// Try to acquire an idle item from the pool.
    ///
    /// Evicts expired items first, then pops the oldest idle item.
    /// Returns `None` if no idle items are available.
    ///
    /// The caller **must** return the item via [`release`][GenericPool::release]
    /// when done.
    pub fn try_acquire(&mut self) -> Option<T> {
        self.evict_expired();
        if let Some(slot) = self.idle.pop_front() {
            self.active_count += 1;
            Some(slot.value)
        } else {
            None
        }
    }

    /// Return a previously acquired item back to the idle pool.
    ///
    /// If the pool is full, the item is silently dropped.
    pub fn release(&mut self, value: T) {
        self.active_count = self.active_count.saturating_sub(1);
        if self.idle.len() + self.active_count < self.config.max_size {
            self.idle.push_back(GenericPoolSlot {
                value,
                created_ms: now_ms(),
            });
        }
    }

    /// Remove all idle items that have exceeded `max_lifetime_ms`.
    ///
    /// If `max_lifetime_ms` is `0`, all idle items are evicted.
    pub fn evict_expired(&mut self) {
        let max_ms = self.config.max_lifetime_ms;
        let now = now_ms();
        self.idle
            .retain(|slot| max_ms > 0 && now.saturating_sub(slot.created_ms) < max_ms);
    }

    /// Drain all idle items, returning them as a `Vec`.
    ///
    /// Active items are NOT affected.
    pub fn drain(&mut self) -> Vec<T> {
        self.idle.drain(..).map(|s| s.value).collect()
    }

    /// Returns `true` when there are no idle or active items.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.idle.is_empty() && self.active_count == 0
    }
}

// =============================================================================
// GenericPool tests
// =============================================================================

#[cfg(test)]
mod generic_pool_tests {
    use super::*;

    fn make_pool(max_size: usize) -> GenericPool<i32> {
        GenericPool::new(GenericPoolConfig::new(max_size, 1, 300_000))
    }

    #[test]
    fn test_generic_pool_add_and_acquire() {
        let mut pool = make_pool(4);
        pool.add(42).expect("should add");
        let item = pool.try_acquire();
        assert_eq!(item, Some(42));
    }

    #[test]
    fn test_generic_pool_release_returns_to_idle() {
        let mut pool = make_pool(4);
        pool.add(10).expect("add");
        let item = pool.try_acquire().expect("acquire");
        assert_eq!(pool.idle_count(), 0);
        assert_eq!(pool.active_count(), 1);

        pool.release(item);
        assert_eq!(pool.idle_count(), 1);
        assert_eq!(pool.active_count(), 0);
    }

    #[test]
    fn test_generic_pool_max_size_respected() {
        let mut pool = make_pool(2);
        pool.add(1).expect("add 1");
        pool.add(2).expect("add 2");
        let result = pool.add(3);
        assert!(result.is_err(), "should reject when at capacity");
        assert_eq!(result.err(), Some(3));
    }

    #[test]
    fn test_generic_pool_drain() {
        let mut pool = make_pool(4);
        pool.add(1).expect("add 1");
        pool.add(2).expect("add 2");
        pool.add(3).expect("add 3");

        let drained = pool.drain();
        assert_eq!(drained.len(), 3);
        assert_eq!(pool.idle_count(), 0);
    }

    #[test]
    fn test_generic_pool_idle_count() {
        let mut pool = make_pool(4);
        assert_eq!(pool.idle_count(), 0);
        pool.add(1).expect("add");
        pool.add(2).expect("add");
        assert_eq!(pool.idle_count(), 2);
    }

    #[test]
    fn test_generic_pool_active_count_tracking() {
        let mut pool = make_pool(4);
        pool.add(7).expect("add");
        pool.add(8).expect("add");

        let _a = pool.try_acquire();
        assert_eq!(pool.active_count(), 1);
        let _b = pool.try_acquire();
        assert_eq!(pool.active_count(), 2);
        assert_eq!(pool.idle_count(), 0);
    }

    #[test]
    fn test_generic_pool_defaults() {
        let pool: GenericPool<String> = GenericPool::with_defaults();
        assert_eq!(pool.config().max_size, 16);
        assert_eq!(pool.config().min_idle, 1);
        assert_eq!(pool.config().max_lifetime_ms, 300_000);
        assert!(pool.is_empty());
    }

    #[test]
    fn test_generic_pool_add_overflow_returns_err() {
        let mut pool = make_pool(1);
        pool.add(100).expect("first add");
        let err = pool.add(200);
        assert!(err.is_err());
        assert_eq!(err.err(), Some(200));
    }

    #[test]
    fn test_generic_pool_config_fields() {
        let cfg = GenericPoolConfig::new(8, 2, 60_000);
        assert_eq!(cfg.max_size, 8);
        assert_eq!(cfg.min_idle, 2);
        assert_eq!(cfg.max_lifetime_ms, 60_000);
    }

    #[test]
    fn test_generic_pool_total() {
        let mut pool = make_pool(4);
        pool.add(1).expect("add");
        pool.add(2).expect("add");
        let _item = pool.try_acquire();
        // 1 idle + 1 active = 2
        assert_eq!(pool.total(), 2);
    }

    #[test]
    fn test_generic_pool_evict_expired_zero_lifetime() {
        // max_lifetime_ms = 0 → all idle items should be evicted immediately
        let mut pool: GenericPool<i32> = GenericPool::new(GenericPoolConfig::new(8, 1, 0));
        pool.add(1).expect("add");
        pool.add(2).expect("add");
        assert_eq!(pool.idle_count(), 2);

        pool.evict_expired();
        assert_eq!(pool.idle_count(), 0);
    }

    #[test]
    fn test_generic_pool_release_increments_idle() {
        let mut pool = make_pool(4);
        // Start with nothing
        assert_eq!(pool.idle_count(), 0);
        assert_eq!(pool.active_count(), 0);

        // Manually increment active by acquiring from a pool seeded via release
        pool.release(99); // release without prior acquire still adds to idle
        assert_eq!(pool.idle_count(), 1);
        assert_eq!(pool.active_count(), 0); // saturating_sub(0) = 0

        let item = pool.try_acquire().expect("acquire");
        assert_eq!(item, 99);
        assert_eq!(pool.idle_count(), 0);
        assert_eq!(pool.active_count(), 1);
    }

    #[test]
    fn test_generic_pool_is_empty() {
        let pool: GenericPool<u8> = GenericPool::with_defaults();
        assert!(pool.is_empty());
    }

    #[test]
    fn test_generic_pool_acquire_empty_returns_none() {
        let mut pool: GenericPool<u32> = GenericPool::with_defaults();
        assert!(pool.try_acquire().is_none());
    }
}
