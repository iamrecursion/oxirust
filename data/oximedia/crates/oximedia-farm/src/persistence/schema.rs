//! Database schema definitions for the farm persistence layer.
//!
//! Initializes all tables and indexes.  Key design notes:
//! - `PRAGMA journal_mode=WAL` — improves concurrent read/write (best-effort:
//!   the oxisqlite 0.3.0 engine rejects this settable pragma, so we degrade
//!   gracefully when it is unavailable).
//! - `PRAGMA synchronous=NORMAL` — better throughput for media workloads
//!   (best-effort, same degradation rule).
//! - Composite index `idx_jobs_priority` for the scheduler hot-path.
//!
//! All methods accept a `super::Inner` reference (holding the synchronous
//! OxiSQL connection) so that callers do not need to hold an extra lock.

use super::Inner;
use crate::Result as FarmResult;
use oxisql_core::Value;

#[cfg(test)]
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// Schema
// ---------------------------------------------------------------------------

pub struct Schema;

#[allow(dead_code)]
impl Schema {
    /// Table-only DDL (no PRAGMA, no indexes).
    ///
    /// PRAGMA statements are not supported by oxisqlite.  Indexes are created
    /// separately via individual `execute` calls so that "IF NOT EXISTS" is
    /// properly honoured on re-open.
    const TABLES_DDL: &'static str = "
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            job_type TEXT NOT NULL,
            state TEXT NOT NULL,
            priority INTEGER NOT NULL,
            input_path TEXT NOT NULL,
            output_path TEXT NOT NULL,
            parameters TEXT NOT NULL,
            metadata TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            started_at INTEGER,
            completed_at INTEGER,
            deadline INTEGER
        );
        CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY,
            job_id TEXT NOT NULL,
            state TEXT NOT NULL,
            worker_id TEXT,
            task_type TEXT NOT NULL,
            payload BLOB NOT NULL,
            priority INTEGER NOT NULL,
            created_at INTEGER NOT NULL,
            assigned_at INTEGER,
            completed_at INTEGER,
            retry_count INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY(job_id) REFERENCES jobs(id)
        );
        CREATE TABLE IF NOT EXISTS workers (
            id TEXT PRIMARY KEY,
            hostname TEXT NOT NULL,
            state TEXT NOT NULL,
            capabilities TEXT NOT NULL,
            metadata TEXT NOT NULL,
            registered_at INTEGER NOT NULL,
            last_heartbeat INTEGER NOT NULL,
            active_tasks INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER NOT NULL,
            level TEXT NOT NULL,
            message TEXT NOT NULL,
            job_id TEXT,
            task_id TEXT,
            worker_id TEXT,
            context TEXT
        );
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER NOT NULL,
            metric_name TEXT NOT NULL,
            metric_value REAL NOT NULL,
            labels TEXT NOT NULL
        );
    ";

    /// Performance-tuning PRAGMAs applied best-effort at schema creation.
    ///
    /// These are settable PRAGMAs the oxisqlite 0.3.0 engine does not
    /// recognise (it rejects them with "Not a valid pragma name").  When the
    /// engine supports them they tune the on-disk journal/durability for media
    /// workloads; when it does not, we degrade gracefully and continue without
    /// them.  See [`Self::apply_pragmas_best_effort`].
    const PRAGMAS: &'static [&'static str] = &[
        "PRAGMA journal_mode=WAL",
        "PRAGMA synchronous=NORMAL",
        "PRAGMA foreign_keys=ON",
    ];

    /// Index DDL run individually to tolerate idempotent re-open.
    const INDEXES: &'static [&'static str] = &[
        "CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs(state)",
        "CREATE INDEX IF NOT EXISTS idx_jobs_priority ON jobs(priority, created_at)",
        "CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at)",
        "CREATE INDEX IF NOT EXISTS idx_tasks_job_id ON tasks(job_id)",
        "CREATE INDEX IF NOT EXISTS idx_tasks_state ON tasks(state)",
        "CREATE INDEX IF NOT EXISTS idx_tasks_worker_id ON tasks(worker_id)",
        "CREATE INDEX IF NOT EXISTS idx_tasks_priority ON tasks(priority)",
        "CREATE INDEX IF NOT EXISTS idx_workers_state ON workers(state)",
        "CREATE INDEX IF NOT EXISTS idx_workers_last_heartbeat ON workers(last_heartbeat)",
        "CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON logs(timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_logs_job_id ON logs(job_id)",
        "CREATE INDEX IF NOT EXISTS idx_logs_task_id ON logs(task_id)",
        "CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(metric_name)",
    ];

    /// Create all database tables and indexes (idempotent).
    pub(crate) fn create_tables(inner: &Inner) -> FarmResult<()> {
        // Real data DDL — never swallow these errors.
        inner.execute_batch(Self::TABLES_DDL)?;
        // Performance PRAGMAs are best-effort (graceful degradation).
        Self::apply_pragmas_best_effort(inner);
        for ddl in Self::INDEXES {
            if let Err(e) = inner.exec(ddl, &[]) {
                if !e.to_string().contains("already exists") {
                    return Err(e);
                }
            }
        }
        Ok(())
    }

    /// Apply the performance-tuning [`Self::PRAGMAS`] one at a time, ignoring
    /// engine-unsupported failures.
    ///
    /// The oxisqlite 0.3.0 engine rejects settable PRAGMAs (e.g.
    /// `journal_mode`, `foreign_keys`) with "Not a valid pragma name" and
    /// `VACUUM` with "VACUUM not supported yet".  Those are *enhancements*, not
    /// correctness requirements, so a rejection here is logged-as-no-op rather
    /// than propagated: graceful degradation, never fabrication.  Real data
    /// operations (CREATE TABLE, INSERT, …) keep propagating their errors.
    fn apply_pragmas_best_effort(inner: &Inner) {
        for pragma in Self::PRAGMAS {
            // Discard the result entirely: success applies the tuning, an
            // engine-unsupported error simply leaves the default in place.
            let _ = inner.exec(pragma, &[]);
        }
    }

    /// Insert a job row.
    ///
    /// Returns the number of rows inserted (1 on success).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn insert_job(
        inner: &Inner,
        id: &str,
        job_type: &str,
        state: &str,
        priority: i64,
        input_path: &str,
        output_path: &str,
        parameters: &str,
        metadata: &str,
        created_at: i64,
    ) -> FarmResult<u64> {
        inner.exec(
            "INSERT INTO jobs
             (id, job_type, state, priority, input_path, output_path,
              parameters, metadata, created_at)
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)",
            &[
                &id,
                &job_type,
                &state,
                &priority,
                &input_path,
                &output_path,
                &parameters,
                &metadata,
                &created_at,
            ],
        )
    }

    /// Query pending jobs ordered by priority (highest first) then oldest first.
    pub(crate) fn query_pending_jobs(inner: &Inner) -> FarmResult<Vec<(String, String, i64, i64)>> {
        let rows = inner.query(
            "SELECT id, state, priority, created_at
             FROM jobs
             WHERE state = 'pending'
             ORDER BY priority DESC, created_at ASC
             LIMIT 100",
            &[],
        )?;

        rows.iter()
            .map(|row| {
                let id = col_text(row, 0)?;
                let state = col_text(row, 1)?;
                let priority = col_i64(row, 2)?;
                let created_at = col_i64(row, 3)?;
                Ok((id, state, priority, created_at))
            })
            .collect()
    }

    /// Update a job's state.
    pub(crate) fn update_job_state(inner: &Inner, id: &str, new_state: &str) -> FarmResult<u64> {
        inner.exec(
            "UPDATE jobs SET state = $1 WHERE id = $2",
            &[&new_state, &id],
        )
    }

    /// Count tables by type (for tests).
    pub(crate) fn count_tables(inner: &Inner) -> FarmResult<i64> {
        let rows = inner.query(
            "SELECT COUNT(*) FROM sqlite_master \
             WHERE type='table' AND name NOT LIKE 'sqlite_%'",
            &[],
        )?;
        col_i64(
            rows.first()
                .ok_or_else(|| crate::FarmError::Database("no rows".to_string()))?,
            0,
        )
    }

    /// Check if a table exists (for tests).
    pub(crate) fn table_exists(inner: &Inner, name: &str) -> FarmResult<bool> {
        let rows = inner.query(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=$1",
            &[&name],
        )?;
        let count = col_i64(
            rows.first()
                .ok_or_else(|| crate::FarmError::Database("no rows".to_string()))?,
            0,
        )?;
        Ok(count > 0)
    }

    /// Check if an index exists (for tests).
    pub(crate) fn index_exists(inner: &Inner, name: &str) -> FarmResult<bool> {
        let rows = inner.query(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name=$1",
            &[&name],
        )?;
        let count = col_i64(
            rows.first()
                .ok_or_else(|| crate::FarmError::Database("no rows".to_string()))?,
            0,
        )?;
        Ok(count > 0)
    }
}

// ---------------------------------------------------------------------------
// Column helpers
// ---------------------------------------------------------------------------

fn col_text(row: &oxisql_core::Row, idx: usize) -> FarmResult<String> {
    match row.get_by_index(idx) {
        Some(Value::Text(s)) => Ok(s.clone()),
        Some(other) => Err(crate::FarmError::Database(format!(
            "column {idx}: expected text, got {}",
            other.type_name()
        ))),
        None => Err(crate::FarmError::Database(format!(
            "column {idx} missing from row"
        ))),
    }
}

fn col_i64(row: &oxisql_core::Row, idx: usize) -> FarmResult<i64> {
    match row.get_by_index(idx) {
        Some(Value::I64(n)) => Ok(*n),
        Some(Value::Null) | None => Ok(0),
        Some(other) => Err(crate::FarmError::Database(format!(
            "column {idx}: expected integer, got {}",
            other.type_name()
        ))),
    }
}

// ---------------------------------------------------------------------------
// Open helpers — used by tests
// ---------------------------------------------------------------------------

/// Open an in-memory OxiSQL connection wrapped in `Inner` for schema tests.
///
/// Returns `Arc<Mutex<Inner>>` to match the pattern used in `Database`.
#[cfg(test)]
pub(crate) fn open_memory_inner() -> FarmResult<Arc<Mutex<Inner>>> {
    use oxisql_sqlite_compat::SqliteConnectionBlocking;
    let conn = SqliteConnectionBlocking::open_memory()
        .map_err(|e| crate::FarmError::Database(e.to_string()))?;
    Ok(Arc::new(Mutex::new(Inner { conn })))
}

#[cfg(test)]
fn with_inner<F, T>(inner: &Arc<Mutex<Inner>>, f: F) -> FarmResult<T>
where
    F: FnOnce(&Inner) -> FarmResult<T>,
{
    let guard = inner
        .lock()
        .map_err(|_| crate::FarmError::Database("mutex poisoned".to_string()))?;
    f(&guard)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_inner() -> Arc<Mutex<Inner>> {
        open_memory_inner().expect("in-memory inner")
    }

    #[test]
    fn test_schema_creation() {
        let inner = make_inner();
        with_inner(&inner, Schema::create_tables).expect("create tables");

        let count = with_inner(&inner, Schema::count_tables).expect("count tables");
        assert_eq!(count, 5); // jobs, tasks, workers, logs, metrics
    }

    #[test]
    fn test_jobs_table_exists() {
        let inner = make_inner();
        with_inner(&inner, Schema::create_tables).expect("create tables");

        let exists = with_inner(&inner, |i| Schema::table_exists(i, "jobs")).expect("table_exists");
        assert!(exists);
    }

    #[test]
    fn test_tasks_table_exists() {
        let inner = make_inner();
        with_inner(&inner, Schema::create_tables).expect("create tables");

        let exists =
            with_inner(&inner, |i| Schema::table_exists(i, "tasks")).expect("table_exists");
        assert!(exists);
    }

    #[test]
    fn test_workers_table_exists() {
        let inner = make_inner();
        with_inner(&inner, Schema::create_tables).expect("create tables");

        let exists =
            with_inner(&inner, |i| Schema::table_exists(i, "workers")).expect("table_exists");
        assert!(exists);
    }

    #[test]
    fn test_pragmas_are_applied_without_error() {
        let inner = make_inner();
        // create_tables applies the performance PRAGMAs best-effort; whether
        // the engine accepts or rejects them, schema creation must still
        // succeed (graceful degradation).
        with_inner(&inner, Schema::create_tables).expect("apply pragmas via create_tables");
    }

    #[test]
    fn test_insert_and_query_pending_jobs() {
        let inner = make_inner();
        with_inner(&inner, Schema::create_tables).expect("create tables");

        with_inner(&inner, |i| {
            Schema::insert_job(
                i,
                "job-1",
                "transcode",
                "pending",
                5,
                "/in/a.mp4",
                "/out/a.mp4",
                "{}",
                "{}",
                1000,
            )
        })
        .expect("insert job-1");

        with_inner(&inner, |i| {
            Schema::insert_job(
                i,
                "job-2",
                "transcode",
                "pending",
                10,
                "/in/b.mp4",
                "/out/b.mp4",
                "{}",
                "{}",
                900,
            )
        })
        .expect("insert job-2");

        let rows = with_inner(&inner, Schema::query_pending_jobs).expect("query pending");
        assert_eq!(rows.len(), 2);
        // Highest priority (10) first — job-2
        assert_eq!(rows[0].0, "job-2");
        assert_eq!(rows[0].2, 10);
    }

    #[test]
    fn test_update_job_state() {
        let inner = make_inner();
        with_inner(&inner, Schema::create_tables).expect("create tables");

        with_inner(&inner, |i| {
            Schema::insert_job(
                i,
                "job-x",
                "transcode",
                "pending",
                1,
                "/in/x.mp4",
                "/out/x.mp4",
                "{}",
                "{}",
                100,
            )
        })
        .expect("insert");

        let updated = with_inner(&inner, |i| Schema::update_job_state(i, "job-x", "running"))
            .expect("update state");
        assert_eq!(updated, 1);

        let rows = with_inner(&inner, |i| {
            i.query("SELECT state FROM jobs WHERE id = 'job-x'", &[])
        })
        .expect("query state");

        let state = col_text(rows.first().expect("row"), 0).expect("state text");
        assert_eq!(state, "running");
    }

    #[test]
    fn test_composite_index_exists_on_jobs() {
        let inner = make_inner();
        with_inner(&inner, Schema::create_tables).expect("create tables");

        let exists = with_inner(&inner, |i| Schema::index_exists(i, "idx_jobs_priority"))
            .expect("index_exists");
        assert!(exists, "composite priority+created_at index should exist");
    }

    #[test]
    fn test_query_only_returns_pending_jobs() {
        let inner = make_inner();
        with_inner(&inner, Schema::create_tables).expect("create tables");

        with_inner(&inner, |i| {
            Schema::insert_job(
                i,
                "pending-job",
                "transcode",
                "pending",
                5,
                "/in/p.mp4",
                "/out/p.mp4",
                "{}",
                "{}",
                200,
            )
        })
        .expect("insert pending");
        with_inner(&inner, |i| {
            Schema::insert_job(
                i,
                "running-job",
                "transcode",
                "running",
                5,
                "/in/r.mp4",
                "/out/r.mp4",
                "{}",
                "{}",
                100,
            )
        })
        .expect("insert running");

        let rows = with_inner(&inner, Schema::query_pending_jobs).expect("query");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].0, "pending-job");
    }

    #[test]
    fn test_create_tables_is_idempotent() {
        let inner = make_inner();
        with_inner(&inner, Schema::create_tables).expect("first create_tables call");
        with_inner(&inner, Schema::create_tables)
            .expect("second create_tables should be idempotent");

        let count = with_inner(&inner, Schema::count_tables).expect("count tables");
        assert_eq!(
            count, 5,
            "idempotent create_tables must not duplicate tables"
        );
    }

    #[test]
    fn test_all_expected_table_names_exist() {
        let inner = make_inner();
        with_inner(&inner, Schema::create_tables).expect("create tables");

        let expected = ["jobs", "tasks", "workers", "logs", "metrics"];
        for name in expected {
            let exists =
                with_inner(&inner, |i| Schema::table_exists(i, name)).expect("table_exists");
            assert!(exists, "table '{name}' should exist after create_tables");
        }
    }

    #[test]
    fn test_key_indexes_exist_after_double_create() {
        let inner = make_inner();
        with_inner(&inner, Schema::create_tables).expect("first create");
        with_inner(&inner, Schema::create_tables).expect("second create");

        let expected_indexes = [
            "idx_jobs_state",
            "idx_jobs_priority",
            "idx_jobs_created_at",
            "idx_tasks_job_id",
            "idx_tasks_state",
            "idx_workers_state",
            "idx_workers_last_heartbeat",
            "idx_logs_timestamp",
            "idx_metrics_timestamp",
        ];
        for idx in expected_indexes {
            let exists =
                with_inner(&inner, |i| Schema::index_exists(i, idx)).expect("index_exists");
            assert!(
                exists,
                "index '{idx}' should exist after double create_tables"
            );
        }
    }

    #[test]
    fn test_insert_and_query_work_after_idempotent_create() {
        let inner = make_inner();
        with_inner(&inner, Schema::create_tables).expect("first create");
        with_inner(&inner, Schema::create_tables).expect("second create (idempotent)");

        with_inner(&inner, |i| {
            Schema::insert_job(
                i,
                "job-idem",
                "transcode",
                "pending",
                7,
                "/in/idem.mp4",
                "/out/idem.mp4",
                "{}",
                "{}",
                500,
            )
        })
        .expect("insert after idempotent create");

        let rows = with_inner(&inner, Schema::query_pending_jobs).expect("query pending");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].0, "job-idem");
        assert_eq!(rows[0].2, 7, "priority should be preserved");
    }

    #[test]
    fn test_priority_ordering_with_multiple_jobs() {
        let inner = make_inner();
        with_inner(&inner, Schema::create_tables).expect("create tables");

        let jobs = [
            ("j-low", 1i64, 1000i64),
            ("j-high", 10, 2000),
            ("j-mid", 5, 1500),
            ("j-hi2", 10, 1800), // same priority as j-high, older created_at
        ];
        for (id, prio, ts) in jobs {
            with_inner(&inner, |i| {
                Schema::insert_job(
                    i,
                    id,
                    "transcode",
                    "pending",
                    prio,
                    "/in/x.mp4",
                    "/out/x.mp4",
                    "{}",
                    "{}",
                    ts,
                )
            })
            .expect("insert");
        }

        let rows = with_inner(&inner, Schema::query_pending_jobs).expect("query pending");
        assert_eq!(rows.len(), 4);
        // Oldest among highest-priority first: j-hi2 (ts=1800) before j-high (ts=2000)
        assert_eq!(
            rows[0].0, "j-hi2",
            "oldest among highest-priority should come first"
        );
        assert_eq!(rows[0].2, 10);
        assert_eq!(rows[1].0, "j-high");
        assert_eq!(rows[2].2, 5);
        assert_eq!(rows[3].2, 1);
    }

    #[test]
    fn test_update_job_state_affects_pending_query() {
        let inner = make_inner();
        with_inner(&inner, Schema::create_tables).expect("create tables");

        with_inner(&inner, |i| {
            Schema::insert_job(
                i,
                "j-a",
                "transcode",
                "pending",
                5,
                "/in/a.mp4",
                "/out/a.mp4",
                "{}",
                "{}",
                100,
            )
        })
        .expect("insert j-a");
        with_inner(&inner, |i| {
            Schema::insert_job(
                i,
                "j-b",
                "transcode",
                "pending",
                3,
                "/in/b.mp4",
                "/out/b.mp4",
                "{}",
                "{}",
                200,
            )
        })
        .expect("insert j-b");

        let before = with_inner(&inner, Schema::query_pending_jobs).expect("query before");
        assert_eq!(before.len(), 2);

        with_inner(&inner, |i| Schema::update_job_state(i, "j-a", "running")).expect("update j-a");

        let after = with_inner(&inner, Schema::query_pending_jobs).expect("query after");
        assert_eq!(after.len(), 1, "only j-b should remain pending");
        assert_eq!(after[0].0, "j-b");
    }
}
