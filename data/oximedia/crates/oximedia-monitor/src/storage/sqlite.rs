//! `SQLite` storage for historical time series data.

use crate::error::MonitorResult;
use chrono::{DateTime, Utc};
use oxisql_core::{ToSqlValue, Value};
use oxisql_sqlite_compat::SqliteConnectionBlocking;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// Error helpers
// ---------------------------------------------------------------------------

fn map_oxi(e: impl std::fmt::Display) -> crate::error::MonitorError {
    crate::error::MonitorError::Storage(e.to_string())
}

/// Returns `true` when the given engine error is the oxisqlite 0.3.0
/// "feature not supported" signal — i.e. a settable PRAGMA the engine does
/// not recognise (`"Not a valid pragma name"`) or `VACUUM`
/// (`"VACUUM not supported yet"`).
///
/// These are optional tuning/maintenance operations; treating them as a no-op
/// success is graceful degradation, not fabrication. Real data errors
/// (constraint violations, type mismatches, missing tables, …) never match
/// this predicate and are propagated unchanged.
fn is_unsupported_pragma_or_vacuum(e: &impl std::fmt::Display) -> bool {
    let msg = e.to_string().to_ascii_lowercase();
    msg.contains("not a valid pragma name") || msg.contains("not supported yet")
}

/// Drive a blocking closure, optionally exiting the Tokio async context first.
///
/// `SqliteConnectionBlocking::execute / query / execute_batch` internally call
/// `tokio::runtime::Runtime::block_on` via `block_local`.  That call panics if
/// this thread is already inside a running Tokio runtime.  `block_in_place`
/// temporarily moves the thread out of the multi-thread scheduler, allowing
/// `block_on` to succeed.  When there is no current runtime (plain `#[test]`)
/// we call through directly.
///
/// Requires a `multi_thread` Tokio runtime when one is present — all async
/// test fixtures that exercise the storage path must use
/// `#[tokio::test(flavor = "multi_thread")]`.
fn run_blocking<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    if tokio::runtime::Handle::try_current().is_ok() {
        tokio::task::block_in_place(f)
    } else {
        f()
    }
}

// ---------------------------------------------------------------------------
// Column helpers
// ---------------------------------------------------------------------------

fn col_i64(row: &oxisql_core::Row, idx: usize) -> MonitorResult<i64> {
    match row.get_by_index(idx) {
        Some(Value::I64(n)) => Ok(*n),
        Some(other) => Err(crate::error::MonitorError::Storage(format!(
            "column {idx}: expected integer, got {}",
            other.type_name()
        ))),
        None => Err(crate::error::MonitorError::Storage(format!(
            "column {idx} missing from result row"
        ))),
    }
}

fn col_real(row: &oxisql_core::Row, idx: usize) -> MonitorResult<f64> {
    match row.get_by_index(idx) {
        Some(Value::F64(f)) => Ok(*f),
        Some(Value::I64(n)) => Ok(*n as f64),
        Some(other) => Err(crate::error::MonitorError::Storage(format!(
            "column {idx}: expected real, got {}",
            other.type_name()
        ))),
        None => Err(crate::error::MonitorError::Storage(format!(
            "column {idx} missing from result row"
        ))),
    }
}

fn col_text(row: &oxisql_core::Row, idx: usize) -> MonitorResult<String> {
    match row.get_by_index(idx) {
        Some(Value::Text(s)) => Ok(s.clone()),
        Some(other) => Err(crate::error::MonitorError::Storage(format!(
            "column {idx}: expected text, got {}",
            other.type_name()
        ))),
        None => Err(crate::error::MonitorError::Storage(format!(
            "column {idx} missing from result row"
        ))),
    }
}

fn col_opt_text(row: &oxisql_core::Row, idx: usize) -> MonitorResult<Option<String>> {
    match row.get_by_index(idx) {
        Some(Value::Null) | None => Ok(None),
        Some(Value::Text(s)) => Ok(Some(s.clone())),
        Some(other) => Err(crate::error::MonitorError::Storage(format!(
            "column {idx}: expected text or null, got {}",
            other.type_name()
        ))),
    }
}

// ---------------------------------------------------------------------------
// Inner state shared behind Arc<Mutex<…>>
// ---------------------------------------------------------------------------

struct Inner {
    conn: SqliteConnectionBlocking,
}

impl Inner {
    fn exec(&self, sql: &str, params: &[&dyn ToSqlValue]) -> MonitorResult<u64> {
        run_blocking(|| self.conn.execute(sql, params).map_err(map_oxi))
    }

    fn query(&self, sql: &str, params: &[&dyn ToSqlValue]) -> MonitorResult<Vec<oxisql_core::Row>> {
        run_blocking(|| self.conn.query(sql, params).map_err(map_oxi))
    }

    fn execute_batch(&self, sql: &str) -> MonitorResult<()> {
        run_blocking(|| self.conn.execute_batch(sql).map(|_| ()).map_err(map_oxi))
    }
}

/// A time series point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesPoint {
    /// Metric name.
    pub metric_name: String,
    /// Timestamp.
    pub timestamp: DateTime<Utc>,
    /// Value.
    pub value: f64,
    /// Labels (JSON encoded).
    pub labels: Option<String>,
}

/// `SQLite` storage for time series data.
#[derive(Clone)]
pub struct SqliteStorage {
    inner: Arc<Mutex<Inner>>,
}

impl SqliteStorage {
    /// Create a new `SQLite` storage.
    ///
    /// # Errors
    ///
    /// Returns an error if database initialization fails.
    pub fn new(path: impl AsRef<Path>) -> MonitorResult<Self> {
        // Create parent directory if it doesn't exist
        if let Some(parent) = path.as_ref().parent() {
            std::fs::create_dir_all(parent)?;
        }

        let path_str = path.as_ref().to_string_lossy().into_owned();
        let conn = SqliteConnectionBlocking::open(&path_str).map_err(map_oxi)?;

        let inner = Inner { conn };

        // Create tables and indices. Route through `Inner::execute_batch` so
        // the call is safe whether or not an active Tokio runtime is present.
        inner.execute_batch(
            "CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                value REAL NOT NULL,
                labels TEXT,
                UNIQUE(metric_name, timestamp, labels)
            );
            CREATE INDEX IF NOT EXISTS idx_metrics_name_time ON metrics(metric_name, timestamp);
            CREATE INDEX IF NOT EXISTS idx_metrics_time ON metrics(timestamp);
            CREATE TABLE IF NOT EXISTS metrics_1min (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                min_value REAL NOT NULL,
                max_value REAL NOT NULL,
                avg_value REAL NOT NULL,
                sum_value REAL NOT NULL,
                count INTEGER NOT NULL,
                labels TEXT,
                UNIQUE(metric_name, timestamp, labels)
            );
            CREATE TABLE IF NOT EXISTS metrics_1hour (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                min_value REAL NOT NULL,
                max_value REAL NOT NULL,
                avg_value REAL NOT NULL,
                sum_value REAL NOT NULL,
                count INTEGER NOT NULL,
                labels TEXT,
                UNIQUE(metric_name, timestamp, labels)
            );
            CREATE TABLE IF NOT EXISTS metrics_1day (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                min_value REAL NOT NULL,
                max_value REAL NOT NULL,
                avg_value REAL NOT NULL,
                sum_value REAL NOT NULL,
                count INTEGER NOT NULL,
                labels TEXT,
                UNIQUE(metric_name, timestamp, labels)
            );",
        )?;

        Ok(Self {
            inner: Arc::new(Mutex::new(inner)),
        })
    }

    fn with_inner<F, T>(&self, f: F) -> MonitorResult<T>
    where
        F: FnOnce(&Inner) -> MonitorResult<T>,
    {
        let guard = self.inner.lock().map_err(|_| {
            crate::error::MonitorError::Storage("SqliteStorage mutex poisoned".into())
        })?;
        f(&guard)
    }

    /// Insert a time series point.
    ///
    /// # Errors
    ///
    /// Returns an error if insertion fails.
    pub fn insert(&self, point: &TimeSeriesPoint) -> MonitorResult<()> {
        self.with_inner(|inner| {
            let ts = point.timestamp.timestamp();
            let labels_ref: Option<&str> = point.labels.as_deref();
            inner.exec(
                "INSERT OR REPLACE INTO metrics (metric_name, timestamp, value, labels) VALUES ($1, $2, $3, $4)",
                &[&point.metric_name, &ts, &point.value, &labels_ref],
            )?;
            Ok(())
        })
    }

    /// Insert multiple time series points sequentially.
    ///
    /// # Errors
    ///
    /// Returns an error if insertion fails.
    pub fn insert_batch(&self, points: &[TimeSeriesPoint]) -> MonitorResult<()> {
        if points.is_empty() {
            return Ok(());
        }
        self.with_inner(|inner| {
            for point in points {
                let ts = point.timestamp.timestamp();
                let labels_ref: Option<&str> = point.labels.as_deref();
                inner.exec(
                    "INSERT OR REPLACE INTO metrics (metric_name, timestamp, value, labels) VALUES ($1, $2, $3, $4)",
                    &[&point.metric_name, &ts, &point.value, &labels_ref],
                )?;
            }
            Ok(())
        })
    }

    /// Insert multiple downsampled aggregate rows into the 1-minute table.
    ///
    /// # Errors
    ///
    /// Returns an error if insertion fails.
    pub fn insert_1min_batch(&self, rows: &[AggregateRow]) -> MonitorResult<()> {
        self.insert_aggregate_batch("metrics_1min", rows)
    }

    /// Insert multiple downsampled aggregate rows into the 1-hour table.
    ///
    /// # Errors
    ///
    /// Returns an error if insertion fails.
    pub fn insert_1hour_batch(&self, rows: &[AggregateRow]) -> MonitorResult<()> {
        self.insert_aggregate_batch("metrics_1hour", rows)
    }

    /// Insert multiple downsampled aggregate rows into the 1-day table.
    ///
    /// # Errors
    ///
    /// Returns an error if insertion fails.
    pub fn insert_1day_batch(&self, rows: &[AggregateRow]) -> MonitorResult<()> {
        self.insert_aggregate_batch("metrics_1day", rows)
    }

    fn insert_aggregate_batch(&self, table: &str, rows: &[AggregateRow]) -> MonitorResult<()> {
        if rows.is_empty() {
            return Ok(());
        }
        self.with_inner(|inner| {
            let sql = format!(
                "INSERT OR REPLACE INTO {table}
                 (metric_name, timestamp, min_value, max_value, avg_value, sum_value, count, labels)
                 VALUES ($1, $2, $3, $4, $5, $6, $7, $8)"
            );
            for row in rows {
                let ts = row.timestamp.timestamp();
                let labels_ref: Option<&str> = row.labels.as_deref();
                inner.exec(
                    &sql,
                    &[
                        &row.metric_name,
                        &ts,
                        &row.min_value,
                        &row.max_value,
                        &row.avg_value,
                        &row.sum_value,
                        &row.count,
                        &labels_ref,
                    ],
                )?;
            }
            Ok(())
        })
    }

    /// Query time series points.
    ///
    /// # Errors
    ///
    /// Returns an error if query fails.
    pub fn query(
        &self,
        metric_name: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> MonitorResult<Vec<TimeSeriesPoint>> {
        self.with_inner(|inner| {
            let ts_start = start.timestamp();
            let ts_end = end.timestamp();
            let oxi_rows = inner.query(
                "SELECT metric_name, timestamp, value, labels FROM metrics WHERE metric_name = $1 AND timestamp >= $2 AND timestamp <= $3 ORDER BY timestamp ASC",
                &[&metric_name, &ts_start, &ts_end],
            )?;

            let mut points = Vec::with_capacity(oxi_rows.len());
            for row in &oxi_rows {
                let metric_name_v = col_text(row, 0)?;
                let ts_secs = col_i64(row, 1)?;
                let value = col_real(row, 2)?;
                let labels = col_opt_text(row, 3)?;

                let timestamp = DateTime::from_timestamp(ts_secs, 0).ok_or_else(|| {
                    crate::error::MonitorError::Storage(format!(
                        "timestamp value {ts_secs} is out of valid DateTime range"
                    ))
                })?;
                points.push(TimeSeriesPoint {
                    metric_name: metric_name_v,
                    timestamp,
                    value,
                    labels,
                });
            }
            Ok(points)
        })
    }

    /// Query aggregated data from 1-minute table.
    ///
    /// # Errors
    ///
    /// Returns an error if query fails.
    pub fn query_1min_aggregates(
        &self,
        metric_name: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> MonitorResult<Vec<AggregateRow>> {
        self.query_aggregate_table("metrics_1min", metric_name, start, end)
    }

    /// Query aggregated data from 1-hour table.
    ///
    /// # Errors
    ///
    /// Returns an error if query fails.
    pub fn query_1hour_aggregates(
        &self,
        metric_name: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> MonitorResult<Vec<AggregateRow>> {
        self.query_aggregate_table("metrics_1hour", metric_name, start, end)
    }

    /// Query aggregated data from 1-day table.
    ///
    /// # Errors
    ///
    /// Returns an error if query fails.
    pub fn query_1day_aggregates(
        &self,
        metric_name: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> MonitorResult<Vec<AggregateRow>> {
        self.query_aggregate_table("metrics_1day", metric_name, start, end)
    }

    fn query_aggregate_table(
        &self,
        table: &str,
        metric_name: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> MonitorResult<Vec<AggregateRow>> {
        self.with_inner(|inner| {
            let ts_start = start.timestamp();
            let ts_end = end.timestamp();
            let sql = format!(
                "SELECT metric_name, timestamp, min_value, max_value, avg_value, sum_value, count, labels FROM {table} WHERE metric_name = $1 AND timestamp >= $2 AND timestamp <= $3 ORDER BY timestamp ASC"
            );
            let oxi_rows = inner.query(&sql, &[&metric_name, &ts_start, &ts_end])?;

            let mut aggregates = Vec::with_capacity(oxi_rows.len());
            for row in &oxi_rows {
                let metric_name_v = col_text(row, 0)?;
                let ts_secs = col_i64(row, 1)?;
                let min_value = col_real(row, 2)?;
                let max_value = col_real(row, 3)?;
                let avg_value = col_real(row, 4)?;
                let sum_value = col_real(row, 5)?;
                let count = col_i64(row, 6)?;
                let labels = col_opt_text(row, 7)?;

                let timestamp = DateTime::from_timestamp(ts_secs, 0).ok_or_else(|| {
                    crate::error::MonitorError::Storage(format!(
                        "timestamp value {ts_secs} is out of valid DateTime range"
                    ))
                })?;
                aggregates.push(AggregateRow {
                    metric_name: metric_name_v,
                    timestamp,
                    min_value,
                    max_value,
                    avg_value,
                    sum_value,
                    count,
                    labels,
                });
            }
            Ok(aggregates)
        })
    }

    /// Delete old data points before the given timestamp.
    ///
    /// # Errors
    ///
    /// Returns an error if deletion fails.
    pub fn delete_before(&self, timestamp: DateTime<Utc>) -> MonitorResult<usize> {
        let ts = timestamp.timestamp();
        let affected = self
            .with_inner(|inner| inner.exec("DELETE FROM metrics WHERE timestamp < $1", &[&ts]))?;
        Ok(affected as usize)
    }

    /// Delete old rows from the 1-minute aggregate table.
    ///
    /// # Errors
    ///
    /// Returns an error if deletion fails.
    pub fn delete_1min_before(&self, timestamp: DateTime<Utc>) -> MonitorResult<usize> {
        self.delete_aggregate_before("metrics_1min", timestamp)
    }

    /// Delete old rows from the 1-hour aggregate table.
    ///
    /// # Errors
    ///
    /// Returns an error if deletion fails.
    pub fn delete_1hour_before(&self, timestamp: DateTime<Utc>) -> MonitorResult<usize> {
        self.delete_aggregate_before("metrics_1hour", timestamp)
    }

    /// Delete old rows from the 1-day aggregate table.
    ///
    /// # Errors
    ///
    /// Returns an error if deletion fails.
    pub fn delete_1day_before(&self, timestamp: DateTime<Utc>) -> MonitorResult<usize> {
        self.delete_aggregate_before("metrics_1day", timestamp)
    }

    fn delete_aggregate_before(
        &self,
        table: &str,
        timestamp: DateTime<Utc>,
    ) -> MonitorResult<usize> {
        let ts = timestamp.timestamp();
        let sql = format!("DELETE FROM {table} WHERE timestamp < $1");
        let affected = self.with_inner(|inner| inner.exec(&sql, &[&ts]))?;
        Ok(affected as usize)
    }

    /// Get the database size in bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if query fails.
    pub fn size(&self) -> MonitorResult<u64> {
        // Query page_count and page_size separately to avoid correlated subquery
        let page_count = self.with_inner(|inner| {
            let rows = inner.query("PRAGMA page_count", &[])?;
            match rows.first().and_then(|r| r.get_by_index(0)) {
                Some(Value::I64(n)) => Ok(*n),
                Some(Value::Null) | None => Ok(0_i64),
                Some(other) => Err(crate::error::MonitorError::Storage(format!(
                    "page_count: unexpected {}",
                    other.type_name()
                ))),
            }
        })?;
        let page_size = self.with_inner(|inner| {
            let rows = inner.query("PRAGMA page_size", &[])?;
            match rows.first().and_then(|r| r.get_by_index(0)) {
                Some(Value::I64(n)) => Ok(*n),
                Some(Value::Null) | None => Ok(4096_i64),
                Some(other) => Err(crate::error::MonitorError::Storage(format!(
                    "page_size: unexpected {}",
                    other.type_name()
                ))),
            }
        })?;
        Ok((page_count * page_size) as u64)
    }

    /// Vacuum the database to reclaim space.
    ///
    /// `VACUUM` is an optional maintenance operation. The oxisqlite 0.3.0
    /// engine does not implement it and rejects the statement with
    /// `"VACUUM not supported yet"`; in that case this method degrades
    /// gracefully to a no-op success rather than failing. Any other error
    /// (e.g. a genuine I/O failure) is propagated.
    ///
    /// # Errors
    ///
    /// Returns an error if vacuum fails for a reason other than the engine
    /// not supporting `VACUUM`.
    pub fn vacuum(&self) -> MonitorResult<()> {
        self.with_inner(
            |inner| match run_blocking(|| inner.conn.execute("VACUUM", &[])) {
                Ok(_) => Ok(()),
                Err(e) if is_unsupported_pragma_or_vacuum(&e) => Ok(()),
                Err(e) => Err(map_oxi(e)),
            },
        )
    }

    /// Get the count of metrics.
    ///
    /// # Errors
    ///
    /// Returns an error if query fails.
    pub fn count(&self) -> MonitorResult<usize> {
        self.with_inner(|inner| {
            let rows = inner.query("SELECT COUNT(*) FROM metrics", &[])?;
            match rows.first().and_then(|r| r.get_by_index(0)) {
                Some(Value::I64(n)) => Ok(*n as usize),
                Some(Value::Null) | None => Ok(0),
                Some(other) => Err(crate::error::MonitorError::Storage(format!(
                    "count: unexpected {}",
                    other.type_name()
                ))),
            }
        })
    }
}

/// Aggregated row from database.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregateRow {
    /// Metric name.
    pub metric_name: String,
    /// Timestamp.
    pub timestamp: DateTime<Utc>,
    /// Minimum value.
    pub min_value: f64,
    /// Maximum value.
    pub max_value: f64,
    /// Average value.
    pub avg_value: f64,
    /// Sum of values.
    pub sum_value: f64,
    /// Count of values.
    pub count: i64,
    /// Labels (JSON encoded).
    pub labels: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_sqlite_storage_creation() {
        let dir = tempdir().expect("failed to create temp dir");
        let db_path = dir.path().join("test.db");

        let _storage = SqliteStorage::new(&db_path).expect("failed to create");
        assert!(db_path.exists());
    }

    #[test]
    fn test_insert_and_query() {
        let dir = tempdir().expect("failed to create temp dir");
        let db_path = dir.path().join("test.db");
        let storage = SqliteStorage::new(&db_path).expect("failed to create");

        let now = Utc::now();

        let point = TimeSeriesPoint {
            metric_name: "cpu.usage".to_string(),
            timestamp: now,
            value: 42.5,
            labels: None,
        };

        storage.insert(&point).expect("failed to insert");

        let points = storage
            .query(
                "cpu.usage",
                now - chrono::Duration::seconds(10),
                now + chrono::Duration::seconds(10),
            )
            .expect("operation should succeed");

        assert_eq!(points.len(), 1);
        assert_eq!(points[0].value, 42.5);
    }

    #[test]
    fn test_insert_batch() {
        let dir = tempdir().expect("failed to create temp dir");
        let db_path = dir.path().join("test.db");
        let storage = SqliteStorage::new(&db_path).expect("failed to create");

        let now = Utc::now();

        let points: Vec<TimeSeriesPoint> = (0..10)
            .map(|i| TimeSeriesPoint {
                metric_name: "cpu.usage".to_string(),
                timestamp: now + chrono::Duration::seconds(i),
                value: i as f64,
                labels: None,
            })
            .collect();

        storage
            .insert_batch(&points)
            .expect("insert_batch should succeed");

        let queried = storage
            .query(
                "cpu.usage",
                now - chrono::Duration::seconds(10),
                now + chrono::Duration::seconds(20),
            )
            .expect("operation should succeed");

        assert_eq!(queried.len(), 10);
    }

    #[test]
    fn test_delete_before() {
        let dir = tempdir().expect("failed to create temp dir");
        let db_path = dir.path().join("test.db");
        let storage = SqliteStorage::new(&db_path).expect("failed to create");

        let now = Utc::now();

        let points: Vec<TimeSeriesPoint> = (0..10)
            .map(|i| TimeSeriesPoint {
                metric_name: "cpu.usage".to_string(),
                timestamp: now + chrono::Duration::seconds(i),
                value: i as f64,
                labels: None,
            })
            .collect();

        storage
            .insert_batch(&points)
            .expect("insert_batch should succeed");

        // Delete points before now + 5 seconds
        let deleted = storage
            .delete_before(now + chrono::Duration::seconds(5))
            .expect("operation should succeed");

        assert_eq!(deleted, 5);

        let remaining = storage.count().expect("count should succeed");
        assert_eq!(remaining, 5);
    }

    #[test]
    fn test_database_size() {
        let dir = tempdir().expect("failed to create temp dir");
        let db_path = dir.path().join("test.db");
        let storage = SqliteStorage::new(&db_path).expect("failed to create");

        let size = storage.size().expect("size should succeed");
        assert!(size > 0);
    }

    #[test]
    fn test_vacuum() {
        let dir = tempdir().expect("failed to create temp dir");
        let db_path = dir.path().join("test.db");
        let storage = SqliteStorage::new(&db_path).expect("failed to create");

        storage.vacuum().expect("vacuum should succeed");
    }

    #[test]
    fn test_insert_batch_empty_is_noop() {
        let dir = tempdir().expect("failed to create temp dir");
        let db_path = dir.path().join("test.db");
        let storage = SqliteStorage::new(&db_path).expect("failed to create");
        // Should not error on an empty slice.
        storage
            .insert_batch(&[])
            .expect("empty batch should succeed");
        let count = storage.count().expect("count should succeed");
        assert_eq!(count, 0);
    }

    #[test]
    fn test_insert_1min_batch() {
        let dir = tempdir().expect("failed to create temp dir");
        let db_path = dir.path().join("test.db");
        let storage = SqliteStorage::new(&db_path).expect("failed to create");

        let now = Utc::now();
        let rows: Vec<AggregateRow> = (0..5)
            .map(|i| AggregateRow {
                metric_name: "encoding.fps".to_string(),
                timestamp: now + chrono::Duration::seconds(i * 60),
                min_value: 28.0 + i as f64,
                max_value: 32.0 + i as f64,
                avg_value: 30.0 + i as f64,
                sum_value: (30.0 + i as f64) * 60.0,
                count: 60,
                labels: None,
            })
            .collect();

        storage
            .insert_1min_batch(&rows)
            .expect("insert_1min_batch should succeed");

        // Query and verify all 5 rows are present.
        let start = now - chrono::Duration::seconds(1);
        let end = now + chrono::Duration::seconds(5 * 60 + 1);
        let retrieved = storage
            .query_1min_aggregates("encoding.fps", start, end)
            .expect("query should succeed");

        assert_eq!(
            retrieved.len(),
            5,
            "expected 5 aggregate rows, got {}",
            retrieved.len()
        );
        // Verify ordering by timestamp ascending.
        for w in retrieved.windows(2) {
            assert!(
                w[0].timestamp <= w[1].timestamp,
                "rows should be ordered by timestamp"
            );
        }
    }

    #[test]
    fn test_insert_1min_batch_empty_is_noop() {
        let dir = tempdir().expect("failed to create temp dir");
        let db_path = dir.path().join("test.db");
        let storage = SqliteStorage::new(&db_path).expect("failed to create");
        storage
            .insert_1min_batch(&[])
            .expect("empty 1min batch should succeed");
    }

    #[test]
    fn test_insert_1hour_batch() {
        let dir = tempdir().expect("failed to create temp dir");
        let db_path = dir.path().join("test.db");
        let storage = SqliteStorage::new(&db_path).expect("failed to create");

        let now = Utc::now();
        let rows: Vec<AggregateRow> = (0..3)
            .map(|i| AggregateRow {
                metric_name: "cpu.usage".to_string(),
                timestamp: now + chrono::Duration::hours(i),
                min_value: 10.0,
                max_value: 90.0,
                avg_value: 50.0,
                sum_value: 50.0 * 3600.0,
                count: 3600,
                labels: None,
            })
            .collect();

        storage
            .insert_1hour_batch(&rows)
            .expect("insert_1hour_batch should succeed");

        let start = now - chrono::Duration::seconds(1);
        let end = now + chrono::Duration::hours(4);
        let retrieved = storage
            .query_1hour_aggregates("cpu.usage", start, end)
            .expect("query should succeed");

        assert_eq!(retrieved.len(), 3, "expected 3 hourly rows");
    }

    #[test]
    fn test_insert_1day_batch() {
        let dir = tempdir().expect("failed to create temp dir");
        let db_path = dir.path().join("test.db");
        let storage = SqliteStorage::new(&db_path).expect("failed to create");

        let now = Utc::now();
        let rows: Vec<AggregateRow> = (0..2)
            .map(|i| AggregateRow {
                metric_name: "memory.used".to_string(),
                timestamp: now + chrono::Duration::days(i),
                min_value: 4_000.0,
                max_value: 8_000.0,
                avg_value: 6_000.0,
                sum_value: 6_000.0 * 86_400.0,
                count: 86_400,
                labels: None,
            })
            .collect();

        storage
            .insert_1day_batch(&rows)
            .expect("insert_1day_batch should succeed");

        let start = now - chrono::Duration::seconds(1);
        let end = now + chrono::Duration::days(3);
        let retrieved = storage
            .query_1day_aggregates("memory.used", start, end)
            .expect("query should succeed");

        assert_eq!(retrieved.len(), 2, "expected 2 daily rows");
    }

    #[test]
    fn test_batch_write_multi_metric_single_transaction() {
        // Verifies that multiple different metrics can be inserted in one batch.
        let dir = tempdir().expect("failed to create temp dir");
        let db_path = dir.path().join("test.db");
        let storage = SqliteStorage::new(&db_path).expect("failed to create");

        let now = Utc::now();
        let points = vec![
            TimeSeriesPoint {
                metric_name: "cpu.usage".to_string(),
                timestamp: now,
                value: 55.0,
                labels: None,
            },
            TimeSeriesPoint {
                metric_name: "memory.used_mb".to_string(),
                timestamp: now,
                value: 4096.0,
                labels: None,
            },
            TimeSeriesPoint {
                metric_name: "encoding.fps".to_string(),
                timestamp: now,
                value: 29.97,
                labels: None,
            },
        ];

        storage
            .insert_batch(&points)
            .expect("multi-metric batch should succeed");

        let count = storage.count().expect("count should succeed");
        assert_eq!(count, 3, "all 3 metrics should be stored");
    }
}
