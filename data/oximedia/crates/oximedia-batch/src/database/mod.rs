//! Database persistence for batch processing.
//!
//! [`Database`] wraps an OxiSQL connection for SQLite access.  Use
//! [`Database::new`] with a file path for a file-backed database, or pass
//! `":memory:"` as the path for an ephemeral in-memory database for testing.

pub mod schema;

use crate::error::{BatchError, Result};
use crate::job::{BatchJob, BatchOperation, InputSpec, OutputSpec};
use crate::types::{JobId, JobState, Priority, RetryPolicy};
use chrono::Utc;
use oxisql_core::{ToSqlValue, Value};
use oxisql_sqlite_compat::SqliteConnectionBlocking;
use std::path::Path;
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// Error helpers
// ---------------------------------------------------------------------------

fn map_oxi(e: impl std::fmt::Display) -> BatchError {
    BatchError::DatabaseError(e.to_string())
}

fn col_text(row: &oxisql_core::Row, idx: usize) -> Result<String> {
    match row.get_by_index(idx) {
        Some(Value::Text(s)) => Ok(s.clone()),
        Some(other) => Err(BatchError::DatabaseError(format!(
            "column {idx}: expected text, got {}",
            other.type_name()
        ))),
        None => Err(BatchError::DatabaseError(format!(
            "column {idx} missing from row"
        ))),
    }
}

fn col_i64(row: &oxisql_core::Row, idx: usize) -> Result<i64> {
    match row.get_by_index(idx) {
        Some(Value::I64(n)) => Ok(*n),
        Some(Value::Null) | None => Ok(0),
        Some(other) => Err(BatchError::DatabaseError(format!(
            "column {idx}: expected integer, got {}",
            other.type_name()
        ))),
    }
}

fn col_opt_text(row: &oxisql_core::Row, idx: usize) -> Option<String> {
    match row.get_by_index(idx) {
        Some(Value::Text(s)) => Some(s.clone()),
        _ => None,
    }
}

fn col_real_or_none(row: &oxisql_core::Row, idx: usize) -> Option<f64> {
    match row.get_by_index(idx) {
        Some(Value::F64(f)) => Some(*f),
        Some(Value::I64(n)) => Some(*n as f64),
        Some(Value::Null) | None => None,
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Runtime guard helper
// ---------------------------------------------------------------------------

/// Drive a blocking closure, optionally exiting the Tokio async context first.
///
/// `SqliteConnectionBlocking::execute / query / execute_batch` internally call
/// `tokio::runtime::Runtime::block_on` via an internal `block_local` helper.
/// That call panics when this thread is already inside a running Tokio runtime.
/// `block_in_place` temporarily moves the thread out of the multi-thread
/// scheduler so that `block_on` can succeed.  When there is no current runtime
/// (plain `#[test]` contexts) we call through directly.
///
/// All async test fixtures that exercise the database layer must therefore use
/// `#[tokio::test(flavor = "multi_thread")]` so that `block_in_place` is
/// available.
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
// Shared inner state
// ---------------------------------------------------------------------------

struct Inner {
    conn: SqliteConnectionBlocking,
}

impl Inner {
    fn exec(&self, sql: &str, params: &[&dyn ToSqlValue]) -> Result<u64> {
        run_blocking(|| self.conn.execute(sql, params).map_err(map_oxi))
    }

    fn query(&self, sql: &str, params: &[&dyn ToSqlValue]) -> Result<Vec<oxisql_core::Row>> {
        run_blocking(|| self.conn.query(sql, params).map_err(map_oxi))
    }
}

// ---------------------------------------------------------------------------
// DatabasePool — configurable connection (compat shim, single connection)
// ---------------------------------------------------------------------------

/// A SQLite connection wrapper (single-connection; pool API for compatibility).
///
/// # Example
/// ```no_run
/// use std::path::Path;
/// use oximedia_batch::database::DatabasePool;
///
/// let pool = DatabasePool::new(Path::new("/tmp/batch.db"), 4).expect("pool creation failed");
/// let db = pool.into_database();
/// ```
pub struct DatabasePool {
    inner: Arc<Mutex<Inner>>,
}

impl DatabasePool {
    /// Create a new database backed by the SQLite file at *db_path*.
    ///
    /// # Errors
    ///
    /// Returns an error if max_connections is zero or the database fails to open.
    pub fn new(db_path: &Path, max_connections: u32) -> Result<Self> {
        if max_connections == 0 {
            return Err(BatchError::InvalidJobConfig(
                "max_connections must be >= 1".to_string(),
            ));
        }

        let path_str = db_path
            .to_str()
            .ok_or_else(|| {
                BatchError::InvalidJobConfig(
                    "database path contains non-UTF-8 characters".to_string(),
                )
            })?
            .to_string();

        let conn = SqliteConnectionBlocking::open(&path_str).map_err(map_oxi)?;

        let dp = Self {
            inner: Arc::new(Mutex::new(Inner { conn })),
        };
        dp.init_schema()?;
        Ok(dp)
    }

    fn with_inner<F, T>(&self, f: F) -> Result<T>
    where
        F: FnOnce(&Inner) -> Result<T>,
    {
        let guard = self
            .inner
            .lock()
            .map_err(|_| BatchError::DatabaseError("mutex poisoned".to_string()))?;
        f(&guard)
    }

    fn init_schema(&self) -> Result<()> {
        // SCHEMA_DDL is pure DDL (CREATE TABLE / CREATE INDEX) with no settable
        // PRAGMA or VACUUM statements, so it is issued in one batch. If settable
        // PRAGMAs (e.g. journal_mode, foreign_keys) or VACUUM were ever added,
        // they would need best-effort handling because the oxisqlite 0.3 engine
        // rejects them ("Not a valid pragma name" / "VACUUM not supported yet").
        self.with_inner(|inner| {
            run_blocking(|| {
                inner
                    .conn
                    .execute_batch(SCHEMA_DDL)
                    .map(|_| ())
                    .map_err(map_oxi)
            })
        })
    }

    /// Convert this pool into a [`Database`] that shares the same connection.
    #[must_use]
    pub fn into_database(self) -> Database {
        Database { inner: self.inner }
    }
}

const SCHEMA_DDL: &str = "
    CREATE TABLE IF NOT EXISTS jobs (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        operation TEXT NOT NULL,
        inputs TEXT,
        outputs TEXT,
        priority INTEGER NOT NULL,
        retry_policy TEXT,
        dependencies TEXT,
        schedule TEXT,
        metadata TEXT,
        created_at TEXT NOT NULL,
        started_at TEXT,
        completed_at TEXT,
        status TEXT NOT NULL,
        error TEXT
    );
    CREATE TABLE IF NOT EXISTS job_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        level TEXT NOT NULL,
        message TEXT NOT NULL,
        FOREIGN KEY(job_id) REFERENCES jobs(id)
    );
    CREATE TABLE IF NOT EXISTS job_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id TEXT NOT NULL,
        input_file TEXT NOT NULL,
        output_files TEXT,
        success INTEGER NOT NULL,
        error TEXT,
        duration REAL,
        FOREIGN KEY(job_id) REFERENCES jobs(id)
    );
    CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
    CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at);
    CREATE INDEX IF NOT EXISTS idx_job_logs_job_id ON job_logs(job_id);
";

// ---------------------------------------------------------------------------
// Database — main API
// ---------------------------------------------------------------------------

/// Database for job persistence
pub struct Database {
    inner: Arc<Mutex<Inner>>,
}

impl Database {
    /// Create a new database backed by the SQLite file at *path*.
    ///
    /// # Errors
    ///
    /// Returns an error if database initialization fails
    pub fn new(path: &str) -> Result<Self> {
        let conn = SqliteConnectionBlocking::open(path).map_err(map_oxi)?;

        let db = Self {
            inner: Arc::new(Mutex::new(Inner { conn })),
        };
        db.init_schema()?;
        Ok(db)
    }

    fn with_inner<F, T>(&self, f: F) -> Result<T>
    where
        F: FnOnce(&Inner) -> Result<T>,
    {
        let guard = self
            .inner
            .lock()
            .map_err(|_| BatchError::DatabaseError("mutex poisoned".to_string()))?;
        f(&guard)
    }

    fn init_schema(&self) -> Result<()> {
        // SCHEMA_DDL is pure DDL (CREATE TABLE / CREATE INDEX) with no settable
        // PRAGMA or VACUUM statements, so it is issued in one batch. If settable
        // PRAGMAs (e.g. journal_mode, foreign_keys) or VACUUM were ever added,
        // they would need best-effort handling because the oxisqlite 0.3 engine
        // rejects them ("Not a valid pragma name" / "VACUUM not supported yet").
        self.with_inner(|inner| {
            run_blocking(|| {
                inner
                    .conn
                    .execute_batch(SCHEMA_DDL)
                    .map(|_| ())
                    .map_err(map_oxi)
            })
        })
    }

    /// Save a job to the database
    ///
    /// # Errors
    ///
    /// Returns an error if saving fails
    pub fn save_job(&self, job: &BatchJob) -> Result<()> {
        let operation_json = serde_json::to_string(&job.operation)?;
        let inputs_json = serde_json::to_string(&job.inputs)?;
        let outputs_json = serde_json::to_string(&job.outputs)?;
        let retry_json = serde_json::to_string(&job.retry)?;
        let dependencies_json = serde_json::to_string(&job.dependencies)?;
        let schedule_json = serde_json::to_string(&job.schedule)?;
        let metadata_json = serde_json::to_string(&job.metadata)?;
        let priority_i = job.priority as i64;
        let now = Utc::now().to_rfc3339();

        self.with_inner(|inner| {
            inner.exec(
                "INSERT OR REPLACE INTO jobs (
                    id, name, operation, inputs, outputs, priority,
                    retry_policy, dependencies, schedule, metadata,
                    created_at, status
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)",
                &[
                    &job.id.as_str(),
                    &job.name.as_str(),
                    &operation_json,
                    &inputs_json,
                    &outputs_json,
                    &priority_i,
                    &retry_json,
                    &dependencies_json,
                    &schedule_json,
                    &metadata_json,
                    &now,
                    &"Queued",
                ],
            )?;
            Ok(())
        })
    }

    /// Update job status
    ///
    /// # Errors
    ///
    /// Returns an error if update fails
    pub fn update_job_status(&self, job_id: &JobId, status: JobState) -> Result<()> {
        let status_str = match status {
            JobState::Queued => "Queued",
            JobState::Running => "Running",
            JobState::Completed => "Completed",
            JobState::Failed => "Failed",
            JobState::Cancelled => "Cancelled",
            JobState::Pending => "Pending",
        };

        let now = Utc::now().to_rfc3339();

        self.with_inner(|inner| {
            match status {
                JobState::Running => {
                    inner.exec(
                        "UPDATE jobs SET status = $1, started_at = $2 WHERE id = $3",
                        &[&status_str, &now, &job_id.as_str()],
                    )?;
                }
                JobState::Completed | JobState::Failed | JobState::Cancelled => {
                    inner.exec(
                        "UPDATE jobs SET status = $1, completed_at = $2 WHERE id = $3",
                        &[&status_str, &now, &job_id.as_str()],
                    )?;
                }
                _ => {
                    inner.exec(
                        "UPDATE jobs SET status = $1 WHERE id = $2",
                        &[&status_str, &job_id.as_str()],
                    )?;
                }
            }
            Ok(())
        })
    }

    /// Log job error
    ///
    /// # Errors
    ///
    /// Returns an error if logging fails
    pub fn log_job_error(&self, job_id: &JobId, error: &str) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        self.with_inner(|inner| {
            inner.exec(
                "UPDATE jobs SET error = $1 WHERE id = $2",
                &[&error, &job_id.as_str()],
            )?;
            inner.exec(
                "INSERT INTO job_logs (job_id, timestamp, level, message) VALUES ($1, $2, $3, $4)",
                &[&job_id.as_str(), &now, &"ERROR", &error],
            )?;
            Ok(())
        })
    }

    /// Get job by ID
    ///
    /// # Errors
    ///
    /// Returns an error if the job is not found
    pub fn get_job(&self, job_id: &JobId) -> Result<BatchJob> {
        let rows = self.with_inner(|inner| {
            inner.query(
                "SELECT id, name, operation, inputs, outputs, priority, retry_policy FROM jobs WHERE id = $1",
                &[&job_id.as_str()],
            )
        })?;

        let row = rows
            .into_iter()
            .next()
            .ok_or_else(|| BatchError::JobNotFound(job_id.to_string()))?;

        let id = col_text(&row, 0)?;
        let name = col_text(&row, 1)?;
        let operation_json = col_text(&row, 2)?;
        let inputs_json = col_text(&row, 3)?;
        let outputs_json = col_text(&row, 4)?;
        let priority_i = col_i64(&row, 5)?;
        let retry_json = col_text(&row, 6)?;

        let operation: BatchOperation = serde_json::from_str(&operation_json)
            .map_err(|e| BatchError::DatabaseError(format!("operation deserialize: {e}")))?;
        let inputs: Vec<InputSpec> = serde_json::from_str(&inputs_json)
            .map_err(|e| BatchError::DatabaseError(format!("inputs deserialize: {e}")))?;
        let outputs: Vec<OutputSpec> = serde_json::from_str(&outputs_json)
            .map_err(|e| BatchError::DatabaseError(format!("outputs deserialize: {e}")))?;
        let retry: RetryPolicy = serde_json::from_str(&retry_json)
            .map_err(|e| BatchError::DatabaseError(format!("retry deserialize: {e}")))?;

        let priority_enum = match priority_i {
            0 => Priority::Low,
            2 => Priority::High,
            _ => Priority::Normal,
        };

        let mut job = BatchJob::new(name, operation);
        job.id = JobId::from_string(id);
        job.inputs = inputs;
        job.outputs = outputs;
        job.priority = priority_enum;
        job.retry = retry;

        Ok(job)
    }

    /// List all jobs
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails
    pub fn list_jobs(&self) -> Result<Vec<BatchJob>> {
        let rows = self.with_inner(|inner| {
            inner.query(
                "SELECT id, name, operation, inputs, outputs, priority, retry_policy FROM jobs ORDER BY created_at DESC",
                &[],
            )
        })?;

        rows.iter()
            .map(|row| {
                let id = col_text(row, 0)?;
                let name = col_text(row, 1)?;
                let operation_json = col_text(row, 2)?;
                let inputs_json = col_text(row, 3)?;
                let outputs_json = col_text(row, 4)?;
                let priority_i = col_i64(row, 5)?;
                let retry_json = col_text(row, 6)?;

                let operation: BatchOperation = serde_json::from_str(&operation_json)
                    .map_err(|e| BatchError::DatabaseError(format!("operation: {e}")))?;
                let inputs: Vec<InputSpec> = serde_json::from_str(&inputs_json)
                    .map_err(|e| BatchError::DatabaseError(format!("inputs: {e}")))?;
                let outputs: Vec<OutputSpec> = serde_json::from_str(&outputs_json)
                    .map_err(|e| BatchError::DatabaseError(format!("outputs: {e}")))?;
                let retry: RetryPolicy = serde_json::from_str(&retry_json)
                    .map_err(|e| BatchError::DatabaseError(format!("retry: {e}")))?;

                let priority_enum = match priority_i {
                    0 => Priority::Low,
                    2 => Priority::High,
                    _ => Priority::Normal,
                };

                let mut job = BatchJob::new(name, operation);
                job.id = JobId::from_string(id);
                job.inputs = inputs;
                job.outputs = outputs;
                job.priority = priority_enum;
                job.retry = retry;
                Ok(job)
            })
            .collect()
    }

    /// Get job statistics
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails
    pub fn get_statistics(&self) -> Result<JobStatistics> {
        fn extract_count(rows: &[oxisql_core::Row]) -> u64 {
            match rows.first().and_then(|r| r.get_by_index(0)) {
                Some(Value::I64(n)) if *n >= 0 => *n as u64,
                _ => 0,
            }
        }

        let total_rows = self.with_inner(|inner| inner.query("SELECT COUNT(*) FROM jobs", &[]))?;
        let total = extract_count(&total_rows);

        let queued_rows = self.with_inner(|inner| {
            inner.query("SELECT COUNT(*) FROM jobs WHERE status = $1", &[&"Queued"])
        })?;
        let running_rows = self.with_inner(|inner| {
            inner.query("SELECT COUNT(*) FROM jobs WHERE status = $1", &[&"Running"])
        })?;
        let completed_rows = self.with_inner(|inner| {
            inner.query(
                "SELECT COUNT(*) FROM jobs WHERE status = $1",
                &[&"Completed"],
            )
        })?;
        let failed_rows = self.with_inner(|inner| {
            inner.query("SELECT COUNT(*) FROM jobs WHERE status = $1", &[&"Failed"])
        })?;

        Ok(JobStatistics {
            total,
            queued: extract_count(&queued_rows),
            running: extract_count(&running_rows),
            completed: extract_count(&completed_rows),
            failed: extract_count(&failed_rows),
        })
    }

    /// Count jobs by status string
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails
    pub fn count_jobs_by_status(&self, status: &str) -> Result<u64> {
        let rows = self.with_inner(|inner| {
            inner.query("SELECT COUNT(*) FROM jobs WHERE status = $1", &[&status])
        })?;
        let count: u64 = match rows.first().and_then(|r| r.get_by_index(0)) {
            Some(Value::I64(n)) if *n >= 0 => *n as u64,
            _ => 0,
        };
        Ok(count)
    }

    /// Get total duration in seconds across all completed jobs
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails
    pub fn get_total_duration_secs(&self) -> Result<f64> {
        let rows = self.with_inner(|inner| {
            inner.query(
                "SELECT SUM(duration) FROM job_results WHERE success = 1",
                &[],
            )
        })?;
        Ok(rows
            .first()
            .and_then(|row| col_real_or_none(row, 0))
            .unwrap_or(0.0))
    }

    /// Get the status string for a job
    ///
    /// # Errors
    ///
    /// Returns an error if the job is not found
    pub fn get_job_status_string(&self, job_id: &crate::types::JobId) -> Result<String> {
        let rows = self.with_inner(|inner| {
            inner.query("SELECT status FROM jobs WHERE id = $1", &[&job_id.as_str()])
        })?;
        let row = rows
            .into_iter()
            .next()
            .ok_or_else(|| BatchError::JobNotFound(job_id.to_string()))?;
        col_text(&row, 0)
    }

    /// Get `started_at` timestamp for a job
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails
    pub fn get_job_started_at(&self, job_id: &crate::types::JobId) -> Result<Option<String>> {
        let rows = self.with_inner(|inner| {
            inner.query(
                "SELECT started_at FROM jobs WHERE id = $1",
                &[&job_id.as_str()],
            )
        })?;
        Ok(rows.first().and_then(|row| col_opt_text(row, 0)))
    }

    /// Get `completed_at` timestamp for a job
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails
    pub fn get_job_completed_at(&self, job_id: &crate::types::JobId) -> Result<Option<String>> {
        let rows = self.with_inner(|inner| {
            inner.query(
                "SELECT completed_at FROM jobs WHERE id = $1",
                &[&job_id.as_str()],
            )
        })?;
        Ok(rows.first().and_then(|row| col_opt_text(row, 0)))
    }

    /// Get duration in seconds for a job
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails
    pub fn get_job_duration_secs(&self, job_id: &crate::types::JobId) -> Result<Option<f64>> {
        let rows = self.with_inner(|inner| {
            inner.query(
                "SELECT SUM(duration) FROM job_results WHERE job_id = $1",
                &[&job_id.as_str()],
            )
        })?;
        Ok(rows.first().and_then(|row| col_real_or_none(row, 0)))
    }

    /// Get error message for a job
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails
    pub fn get_job_error(&self, job_id: &crate::types::JobId) -> Result<Option<String>> {
        let rows = self.with_inner(|inner| {
            inner.query("SELECT error FROM jobs WHERE id = $1", &[&job_id.as_str()])
        })?;
        Ok(rows.first().and_then(|row| col_opt_text(row, 0)))
    }
}

/// Job statistics
#[derive(Debug, Clone)]
pub struct JobStatistics {
    /// Total number of jobs
    pub total: u64,
    /// Queued jobs
    pub queued: u64,
    /// Running jobs
    pub running: u64,
    /// Completed jobs
    pub completed: u64,
    /// Failed jobs
    pub failed: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operations::FileOperation;
    use tempfile::NamedTempFile;

    #[test]
    fn test_database_creation() {
        let temp_file = NamedTempFile::new().expect("failed to create temp file");
        let db_path = temp_file
            .path()
            .to_str()
            .expect("path should be valid UTF-8");

        let result = Database::new(db_path);
        assert!(result.is_ok());
    }

    #[test]
    fn test_save_job() {
        let temp_file = NamedTempFile::new().expect("failed to create temp file");
        let db_path = temp_file
            .path()
            .to_str()
            .expect("path should be valid UTF-8");
        let db = Database::new(db_path).expect("failed to create database");

        let job = BatchJob::new(
            "test-job".to_string(),
            BatchOperation::FileOp {
                operation: FileOperation::Copy { overwrite: false },
            },
        );

        let result = db.save_job(&job);
        assert!(result.is_ok());
    }

    #[test]
    fn test_update_job_status() {
        let temp_file = NamedTempFile::new().expect("failed to create temp file");
        let db_path = temp_file
            .path()
            .to_str()
            .expect("path should be valid UTF-8");
        let db = Database::new(db_path).expect("failed to create database");

        let job = BatchJob::new(
            "test-job".to_string(),
            BatchOperation::FileOp {
                operation: FileOperation::Copy { overwrite: false },
            },
        );

        db.save_job(&job).expect("failed to save job");
        let result = db.update_job_status(&job.id, JobState::Running);
        assert!(result.is_ok());
    }

    #[test]
    fn test_get_job() {
        let temp_file = NamedTempFile::new().expect("failed to create temp file");
        let db_path = temp_file
            .path()
            .to_str()
            .expect("path should be valid UTF-8");
        let db = Database::new(db_path).expect("failed to create database");

        let job = BatchJob::new(
            "test-job".to_string(),
            BatchOperation::FileOp {
                operation: FileOperation::Copy { overwrite: false },
            },
        );

        db.save_job(&job).expect("failed to save job");
        let retrieved = db.get_job(&job.id);
        assert!(retrieved.is_ok());
        assert_eq!(
            retrieved.expect("retrieved should be valid").name,
            "test-job"
        );
    }

    #[test]
    fn test_list_jobs() {
        let temp_file = NamedTempFile::new().expect("failed to create temp file");
        let db_path = temp_file
            .path()
            .to_str()
            .expect("path should be valid UTF-8");
        let db = Database::new(db_path).expect("failed to create database");

        let job1 = BatchJob::new(
            "job1".to_string(),
            BatchOperation::FileOp {
                operation: FileOperation::Copy { overwrite: false },
            },
        );

        let job2 = BatchJob::new(
            "job2".to_string(),
            BatchOperation::FileOp {
                operation: FileOperation::Copy { overwrite: false },
            },
        );

        db.save_job(&job1).expect("failed to save job");
        db.save_job(&job2).expect("failed to save job");

        let jobs = db.list_jobs().expect("failed to list jobs");
        assert_eq!(jobs.len(), 2);
    }

    #[test]
    fn test_get_statistics() {
        let temp_file = NamedTempFile::new().expect("failed to create temp file");
        let db_path = temp_file
            .path()
            .to_str()
            .expect("path should be valid UTF-8");
        let db = Database::new(db_path).expect("failed to create database");

        let stats = db.get_statistics();
        assert!(stats.is_ok());

        let stats = stats.expect("stats should be valid");
        assert_eq!(stats.total, 0);
    }

    // -----------------------------------------------------------------------
    // DatabasePool tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_database_pool_new() {
        let temp_file = NamedTempFile::new().expect("temp file");
        let dp = DatabasePool::new(temp_file.path(), 4);
        assert!(
            dp.is_ok(),
            "DatabasePool::new should succeed: {:?}",
            dp.err()
        );
    }

    #[test]
    fn test_database_pool_rejects_zero_connections() {
        let temp_file = NamedTempFile::new().expect("temp file");
        let dp = DatabasePool::new(temp_file.path(), 0);
        assert!(dp.is_err(), "zero max_connections should be rejected");
    }

    #[test]
    fn test_database_pool_into_database() {
        let temp_file = NamedTempFile::new().expect("temp file");
        let dp = DatabasePool::new(temp_file.path(), 2).expect("pool");
        let db = dp.into_database();

        let stats = db.get_statistics().expect("stats");
        assert_eq!(stats.total, 0);
    }

    /// Verify basic sequential access works correctly.
    #[test]
    fn test_pool_sequential_access() {
        let temp_file = NamedTempFile::new().expect("temp file");
        let dp = DatabasePool::new(temp_file.path(), 4).expect("pool");
        let db = dp.into_database();

        const JOBS: usize = 40;
        for i in 0..JOBS {
            let job = BatchJob::new(
                format!("job-{i}"),
                BatchOperation::FileOp {
                    operation: FileOperation::Copy { overwrite: false },
                },
            );
            db.save_job(&job).expect("save_job should succeed");
        }

        let stats = db.get_statistics().expect("stats");
        assert_eq!(
            stats.total, JOBS as u64,
            "expected {} total jobs, got {}",
            JOBS, stats.total
        );
    }
}
