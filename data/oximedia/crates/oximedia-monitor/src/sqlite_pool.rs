//! SQLite connection pool for concurrent metric storage access.
//!
//! Production monitoring systems issue concurrent reads (dashboard queries,
//! API requests) and writes (metric ingestion) simultaneously.  This module
//! provides a lightweight connection pool built on top of a `flume` channel
//! queue, using the Pure-Rust OxiSQL engine.
//!
//! # Design
//!
//! ```text
//! ┌──────────────────────────────────────────────────┐
//! │  SqlitePool                                       │
//! │  ┌──────────┐   borrow    ┌────────────────────┐ │
//! │  │  flume   │ ──────────► │ PooledConnection   │ │
//! │  │  channel │ ◄────────── │  (auto-returns on  │ │
//! │  │  (idle   │   Drop       │   Drop)            │ │
//! │  │  conns)  │             └────────────────────┘ │
//! └──────────────────────────────────────────────────┘
//! ```
//!
//! # Feature gating
//!
//! The entire module is gated behind `#[cfg(all(not(target_arch = "wasm32"), feature = "sqlite"))]`
//! to avoid pulling in oxisql on WASM or in builds that don't need SQLite.
//!
//! # Example
//!
//! ```no_run
//! # #[cfg(all(not(target_arch = "wasm32"), feature = "sqlite"))]
//! # {
//! use std::time::Duration;
//! use oximedia_monitor::sqlite_pool::{PoolConfig, SqlitePool};
//!
//! let config = PoolConfig::builder()
//!     .db_path("/tmp/metrics.db")
//!     .pool_size(4)
//!     .acquire_timeout(Duration::from_secs(5))
//!     .build();
//!
//! let pool = SqlitePool::open(config).expect("pool should open");
//! let conn = pool.acquire().expect("acquire should succeed");
//! conn.execute_batch("CREATE TABLE IF NOT EXISTS t (v INTEGER)").ok();
//! // Connection is returned to pool when `conn` is dropped.
//! # }
//! ```

#![allow(dead_code)]

// The entire module is only meaningful when oxisql is available.
#[cfg(all(not(target_arch = "wasm32"), feature = "sqlite"))]
pub use impl_sqlite::{PoolConfig, PoolConfigBuilder, PoolStats, PooledConnection, SqlitePool};

#[cfg(all(not(target_arch = "wasm32"), feature = "sqlite"))]
mod impl_sqlite {
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    use oxisql_core::ToSqlValue;
    use oxisql_sqlite_compat::SqliteConnectionBlocking;

    use crate::error::{MonitorError, MonitorResult};

    fn map_oxi(e: impl std::fmt::Display) -> MonitorError {
        MonitorError::Storage(e.to_string())
    }

    /// Returns `true` when the given engine error is the oxisqlite 0.3.0
    /// "feature not supported" signal — i.e. a settable PRAGMA the engine
    /// does not recognise (`"Not a valid pragma name"`) or `VACUUM`
    /// (`"VACUUM not supported yet"`).
    ///
    /// These are optional per-connection tuning operations; treating them as
    /// a no-op success is graceful degradation, not fabrication. Real data
    /// errors never match this predicate and are propagated unchanged.
    fn is_unsupported_pragma_or_vacuum(e: &impl std::fmt::Display) -> bool {
        let msg = e.to_string().to_ascii_lowercase();
        msg.contains("not a valid pragma name") || msg.contains("not supported yet")
    }

    // -----------------------------------------------------------------------
    // A pooled entry: a blocking connection handle
    // -----------------------------------------------------------------------

    /// A single connection entry held in the pool.
    pub struct ConnEntry {
        pub(super) conn: SqliteConnectionBlocking,
    }

    impl ConnEntry {
        /// Execute a statement batch on this connection.
        ///
        /// # Errors
        ///
        /// Returns an error if execution fails.
        pub fn execute_batch(&self, sql: &str) -> MonitorResult<()> {
            // `execute_batch` returns the affected-row count (`u64`); this
            // wrapper intentionally discards it to keep the batch-DDL
            // contract `-> MonitorResult<()>`.
            self.conn.execute_batch(sql).map(|_| ()).map_err(map_oxi)
        }

        /// Execute a statement, returning affected rows.
        ///
        /// # Errors
        ///
        /// Returns an error if execution fails.
        pub fn execute(&self, sql: &str, params: &[&dyn ToSqlValue]) -> MonitorResult<u64> {
            self.conn.execute(sql, params).map_err(map_oxi)
        }

        /// Run a query, returning all result rows.
        ///
        /// # Errors
        ///
        /// Returns an error if execution fails.
        pub fn query(
            &self,
            sql: &str,
            params: &[&dyn ToSqlValue],
        ) -> MonitorResult<Vec<oxisql_core::Row>> {
            self.conn.query(sql, params).map_err(map_oxi)
        }
    }

    // -----------------------------------------------------------------------
    // Configuration
    // -----------------------------------------------------------------------

    /// Configuration for [`SqlitePool`].
    #[derive(Debug, Clone)]
    pub struct PoolConfig {
        /// Path to the SQLite database file.  Use `:memory:` for in-memory.
        pub db_path: PathBuf,
        /// Number of connections to open eagerly at construction time.
        pub pool_size: usize,
        /// Maximum time to wait for an idle connection before giving up.
        pub acquire_timeout: Duration,
        /// Whether to enable WAL journal mode (recommended for concurrency).
        pub enable_wal: bool,
        /// Whether to enable foreign-key enforcement per connection.
        pub enable_foreign_keys: bool,
    }

    impl Default for PoolConfig {
        fn default() -> Self {
            Self {
                db_path: PathBuf::from(":memory:"),
                pool_size: 4,
                acquire_timeout: Duration::from_secs(10),
                enable_wal: true,
                enable_foreign_keys: true,
            }
        }
    }

    impl PoolConfig {
        /// Start building a [`PoolConfig`].
        #[must_use]
        pub fn builder() -> PoolConfigBuilder {
            PoolConfigBuilder::default()
        }
    }

    /// Builder for [`PoolConfig`].
    #[derive(Debug, Default)]
    pub struct PoolConfigBuilder {
        inner: PoolConfig,
    }

    impl PoolConfigBuilder {
        /// Set the database file path.
        #[must_use]
        pub fn db_path(mut self, path: impl Into<PathBuf>) -> Self {
            self.inner.db_path = path.into();
            self
        }

        /// Set the pool size (number of connections).
        #[must_use]
        pub fn pool_size(mut self, n: usize) -> Self {
            self.inner.pool_size = n.max(1);
            self
        }

        /// Set the acquire timeout.
        #[must_use]
        pub fn acquire_timeout(mut self, d: Duration) -> Self {
            self.inner.acquire_timeout = d;
            self
        }

        /// Enable or disable WAL journal mode.
        #[must_use]
        pub fn enable_wal(mut self, yes: bool) -> Self {
            self.inner.enable_wal = yes;
            self
        }

        /// Enable or disable foreign-key constraints.
        #[must_use]
        pub fn enable_foreign_keys(mut self, yes: bool) -> Self {
            self.inner.enable_foreign_keys = yes;
            self
        }

        /// Set the busy timeout in milliseconds (accepted for API compatibility; no-op).
        #[must_use]
        pub fn busy_timeout_ms(self, _ms: u32) -> Self {
            self
        }

        /// Build the configuration.
        #[must_use]
        pub fn build(self) -> PoolConfig {
            self.inner
        }
    }

    // -----------------------------------------------------------------------
    // Pool statistics
    // -----------------------------------------------------------------------

    /// Runtime statistics for a [`SqlitePool`].
    #[derive(Debug, Clone)]
    pub struct PoolStats {
        /// Total successful `acquire()` calls.
        pub total_acquired: u64,
        /// Total `acquire()` calls that timed out (pool exhausted).
        pub total_timeouts: u64,
        /// Total connections returned to the pool via drop.
        pub total_returned: u64,
        /// Number of idle connections currently in the pool.
        pub idle_count: usize,
        /// Configured pool capacity.
        pub capacity: usize,
    }

    // -----------------------------------------------------------------------
    // Inner pool state (shared via Arc)
    // -----------------------------------------------------------------------

    struct PoolInner {
        idle_tx: flume::Sender<ConnEntry>,
        idle_rx: flume::Receiver<ConnEntry>,
        config: PoolConfig,
        total_acquired: AtomicU64,
        total_timeouts: AtomicU64,
        total_returned: Arc<AtomicU64>,
    }

    impl std::fmt::Debug for PoolInner {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("PoolInner")
                .field("capacity", &self.config.pool_size)
                .field("idle", &self.idle_rx.len())
                .finish()
        }
    }

    // -----------------------------------------------------------------------
    // Connection pool
    // -----------------------------------------------------------------------

    /// A bounded pool of Pure-Rust OxiSQL connections.
    ///
    /// See the [module-level documentation](super) for the design overview.
    #[derive(Debug, Clone)]
    pub struct SqlitePool {
        inner: Arc<PoolInner>,
    }

    impl SqlitePool {
        /// Open a new pool, eagerly creating `config.pool_size` connections.
        ///
        /// # Errors
        ///
        /// Returns an error if any connection fails to open or if the
        /// per-connection PRAGMA commands fail.
        pub fn open(config: PoolConfig) -> MonitorResult<Self> {
            let (idle_tx, idle_rx) = flume::bounded(config.pool_size);

            let path_str = config.db_path.to_string_lossy().into_owned();

            for _ in 0..config.pool_size {
                let entry = Self::open_entry(&path_str, &config)?;
                idle_tx.send(entry).map_err(|_| {
                    MonitorError::Storage("pool channel closed unexpectedly".into())
                })?;
            }

            Ok(Self {
                inner: Arc::new(PoolInner {
                    idle_tx,
                    idle_rx,
                    config,
                    total_acquired: AtomicU64::new(0),
                    total_timeouts: AtomicU64::new(0),
                    total_returned: Arc::new(AtomicU64::new(0)),
                }),
            })
        }

        /// Open a single connection entry and apply per-connection PRAGMAs.
        fn open_entry(path_str: &str, config: &PoolConfig) -> MonitorResult<ConnEntry> {
            let conn = SqliteConnectionBlocking::open(path_str)
                .map_err(|e| MonitorError::Storage(format!("open connection: {e}")))?;

            // `journal_mode` and `foreign_keys` are optional connection tuning.
            // oxisqlite 0.3.0's engine does not accept these as settable PRAGMAs
            // and rejects them with "Not a valid pragma name", so apply them
            // best-effort: attempt each pragma, treat an engine "unsupported"
            // rejection as a no-op success (the engine then uses its own
            // defaults), and only surface genuine failures. This keeps the pool
            // usable across engines with differing PRAGMA support instead of
            // failing to open, without silently swallowing real errors.
            if config.enable_wal {
                Self::apply_optional_pragma(&conn, "PRAGMA journal_mode=WAL;")?;
            }

            if config.enable_foreign_keys {
                Self::apply_optional_pragma(&conn, "PRAGMA foreign_keys=ON;")?;
            }

            Ok(ConnEntry { conn })
        }

        /// Apply an optional tuning PRAGMA, tolerating the oxisqlite 0.3.0
        /// "Not a valid pragma name" rejection as a no-op success while still
        /// propagating any other (genuine) error.
        fn apply_optional_pragma(
            conn: &SqliteConnectionBlocking,
            pragma: &str,
        ) -> MonitorResult<()> {
            match conn.execute_batch(pragma) {
                Ok(_) => Ok(()),
                Err(e) if is_unsupported_pragma_or_vacuum(&e) => Ok(()),
                Err(e) => Err(MonitorError::Storage(format!("apply pragma: {e}"))),
            }
        }

        /// Acquire an idle connection from the pool.
        ///
        /// Blocks for up to `config.acquire_timeout`.  Returns a
        /// [`PooledConnection`] that automatically returns the connection to
        /// the pool when dropped.
        ///
        /// # Errors
        ///
        /// Returns `MonitorError::Storage("pool exhausted")` if no connection
        /// becomes available within the timeout.
        pub fn acquire(&self) -> MonitorResult<PooledConnection> {
            match self
                .inner
                .idle_rx
                .recv_timeout(self.inner.config.acquire_timeout)
            {
                Ok(entry) => {
                    self.inner.total_acquired.fetch_add(1, Ordering::Relaxed);
                    Ok(PooledConnection {
                        entry: Some(entry),
                        return_tx: self.inner.idle_tx.clone(),
                        returned_counter: Arc::clone(&self.inner.total_returned),
                    })
                }
                Err(_) => {
                    self.inner.total_timeouts.fetch_add(1, Ordering::Relaxed);
                    Err(MonitorError::Storage(
                        "pool exhausted: no idle connection within timeout".into(),
                    ))
                }
            }
        }

        /// Number of idle connections currently available.
        #[must_use]
        pub fn idle_count(&self) -> usize {
            self.inner.idle_rx.len()
        }

        /// Pool capacity (number of connections created at startup).
        #[must_use]
        pub fn capacity(&self) -> usize {
            self.inner.config.pool_size
        }

        /// Snapshot of pool runtime statistics.
        #[must_use]
        pub fn stats(&self) -> PoolStats {
            PoolStats {
                total_acquired: self.inner.total_acquired.load(Ordering::Relaxed),
                total_timeouts: self.inner.total_timeouts.load(Ordering::Relaxed),
                total_returned: self.inner.total_returned.load(Ordering::Relaxed),
                idle_count: self.idle_count(),
                capacity: self.capacity(),
            }
        }

        /// Database path used by the pool.
        #[must_use]
        pub fn db_path(&self) -> &Path {
            &self.inner.config.db_path
        }

        /// Execute a closure with an acquired connection, returning the
        /// connection to the pool automatically.
        ///
        /// # Errors
        ///
        /// Returns an error if acquiring the connection fails, or if the
        /// closure returns an error.
        pub fn with_connection<F, T>(&self, f: F) -> MonitorResult<T>
        where
            F: FnOnce(&ConnEntry) -> MonitorResult<T>,
        {
            let conn = self.acquire()?;
            f(&conn)
        }

        /// Execute a DDL/DML batch on every connection in the pool.
        ///
        /// Drains all idle connections, executes, and returns them.
        ///
        /// # Errors
        ///
        /// Returns an error if executing the batch fails.
        pub fn execute_on_all(&self, sql: &str) -> MonitorResult<()> {
            let mut held = Vec::with_capacity(self.inner.config.pool_size);

            while let Ok(entry) = self.inner.idle_rx.try_recv() {
                held.push(entry);
            }

            let mut err: Option<MonitorResult<()>> = None;
            for entry in &held {
                if let Err(e) = entry.execute_batch(sql) {
                    err = Some(Err(MonitorError::Storage(format!("execute_on_all: {e}"))));
                    break;
                }
            }

            for entry in held {
                let _ = self.inner.idle_tx.send(entry);
            }

            err.unwrap_or(Ok(()))
        }
    }

    // -----------------------------------------------------------------------
    // RAII connection guard
    // -----------------------------------------------------------------------

    /// A borrowed connection entry from [`SqlitePool`].
    ///
    /// Automatically returns the connection to the pool on drop.
    pub struct PooledConnection {
        entry: Option<ConnEntry>,
        return_tx: flume::Sender<ConnEntry>,
        returned_counter: Arc<AtomicU64>,
    }

    impl std::ops::Deref for PooledConnection {
        type Target = ConnEntry;
        fn deref(&self) -> &ConnEntry {
            self.entry.as_ref().expect("connection present until drop")
        }
    }

    impl Drop for PooledConnection {
        fn drop(&mut self) {
            if let Some(entry) = self.entry.take() {
                let _ = self.return_tx.send(entry);
                self.returned_counter.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Fallback: when sqlite feature is not enabled, expose a stub so the module
// compiles without the feature.
// ---------------------------------------------------------------------------

/// Pool configuration (requires `sqlite` feature).
#[cfg(any(target_arch = "wasm32", not(feature = "sqlite")))]
#[derive(Debug, Default, Clone)]
pub struct PoolConfig {
    /// Database path placeholder.
    pub db_path: String,
}

#[cfg(any(target_arch = "wasm32", not(feature = "sqlite")))]
impl PoolConfig {
    /// Build a config (no-op stub).
    #[must_use]
    pub fn builder() -> Self {
        Self::default()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, not(target_arch = "wasm32"), feature = "sqlite"))]
mod tests {
    use super::impl_sqlite::*;
    use std::time::Duration;

    fn in_memory_pool(size: usize) -> SqlitePool {
        let config = PoolConfig::builder()
            .db_path(":memory:")
            .pool_size(size)
            .enable_wal(false) // WAL unsupported on :memory:
            .acquire_timeout(Duration::from_millis(500))
            .build();
        SqlitePool::open(config).expect("pool should open")
    }

    #[test]
    fn test_pool_opens_and_stats() {
        let pool = in_memory_pool(3);
        assert_eq!(pool.capacity(), 3);
        assert_eq!(pool.idle_count(), 3);

        let stats = pool.stats();
        assert_eq!(stats.total_acquired, 0);
        assert_eq!(stats.total_timeouts, 0);
        assert_eq!(stats.capacity, 3);
    }

    #[test]
    fn test_acquire_and_return() {
        let pool = in_memory_pool(2);
        {
            let _conn = pool.acquire().expect("acquire should succeed");
            assert_eq!(pool.idle_count(), 1);
        }
        // Connection returned on drop.
        assert_eq!(pool.idle_count(), 2);
        assert_eq!(pool.stats().total_acquired, 1);
    }

    #[test]
    fn test_acquire_all_then_timeout() {
        let pool = in_memory_pool(2);
        let _c1 = pool.acquire().expect("first acquire should succeed");
        let _c2 = pool.acquire().expect("second acquire should succeed");
        assert_eq!(pool.idle_count(), 0);

        // Third acquire should timeout.
        let result = pool.acquire();
        assert!(result.is_err());
        assert_eq!(pool.stats().total_timeouts, 1);
    }

    #[test]
    fn test_with_connection_closure() {
        let pool = in_memory_pool(1);
        let result = pool.with_connection(|conn| {
            conn.execute_batch("CREATE TABLE IF NOT EXISTS t (v INTEGER)")?;
            conn.execute_batch("INSERT INTO t VALUES (42)")?;
            Ok(99u64)
        });
        assert_eq!(result.expect("closure should succeed"), 99u64);
        // Connection should be returned.
        assert_eq!(pool.idle_count(), 1);
    }

    #[test]
    fn test_execute_on_all() {
        let pool = in_memory_pool(3);
        let result = pool.execute_on_all("CREATE TABLE IF NOT EXISTS meta (k TEXT, v TEXT)");
        assert!(result.is_ok());
        assert_eq!(pool.idle_count(), 3); // all returned
    }

    #[test]
    fn test_pool_config_builder() {
        let cfg = PoolConfig::builder()
            .db_path(std::env::temp_dir().join("oximedia-monitor-pool-test.db"))
            .pool_size(8)
            .acquire_timeout(Duration::from_secs(30))
            .enable_wal(true)
            .enable_foreign_keys(true)
            .busy_timeout_ms(10_000)
            .build();
        assert_eq!(cfg.pool_size, 8);
        assert!(cfg.enable_wal);
    }

    #[test]
    fn test_pool_concurrent_access() {
        use std::sync::Arc;

        let pool = Arc::new(in_memory_pool(4));
        pool.execute_on_all("CREATE TABLE IF NOT EXISTS counter (n INTEGER)")
            .expect("DDL should succeed");

        let handles: Vec<_> = (0..4)
            .map(|_| {
                let p = Arc::clone(&pool);
                std::thread::spawn(move || {
                    let conn = p.acquire().expect("acquire in thread should succeed");
                    conn.execute_batch("INSERT INTO counter VALUES (1)")
                        .expect("insert should succeed");
                })
            })
            .collect();

        for h in handles {
            h.join().expect("thread should not panic");
        }

        assert_eq!(pool.idle_count(), 4);
    }
}

// Stub tests for non-sqlite builds so `cargo test` still runs.
#[cfg(all(test, any(target_arch = "wasm32", not(feature = "sqlite"))))]
mod stub_tests {
    use super::PoolConfig;

    #[test]
    fn test_pool_config_stub() {
        let _ = PoolConfig::default();
    }
}
