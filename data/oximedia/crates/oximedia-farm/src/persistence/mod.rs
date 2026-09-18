//! Persistence layer for the encoding farm.
//!
//! Provides SQLite-based persistent storage for:
//! - Jobs and their state
//! - Tasks and their assignments
//! - Worker registrations
//! - Job history and audit logs
//! - Metrics and statistics

mod schema;

use crate::{FarmError, JobId, JobState, JobType, Priority, Result, TaskId, TaskState, WorkerId};
use chrono::{DateTime, Utc};
use oxisql_core::{ToSqlValue, Value};
use oxisql_sqlite_compat::SqliteConnectionBlocking;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

pub use schema::Schema;

// ---------------------------------------------------------------------------
// Inner — shared connection state
// ---------------------------------------------------------------------------

pub(crate) struct Inner {
    /// Synchronous SQLite connection.
    ///
    /// `SqliteConnectionBlocking::open` / `open_memory` use an OS-thread
    /// detour (`block_static`) so they are safe inside any Tokio runtime.
    /// However, `execute`, `query`, and `execute_batch` use `block_local`,
    /// which builds a fresh `current_thread` runtime and calls `block_on`.
    /// Calling that from inside a running Tokio runtime panics.
    ///
    /// We work around this in `Inner::exec`, `Inner::query`, and
    /// `Inner::execute_batch` via the `run_blocking` helper below: when we
    /// detect an active Tokio runtime we first call
    /// `tokio::task::block_in_place` to temporarily exit the async context
    /// (requires a multi-thread runtime, which our `#[tokio::test]` fixtures
    /// already use), then let `block_local` succeed normally.  In plain
    /// `#[test]` contexts there is no current runtime and we call through
    /// directly.
    pub(crate) conn: SqliteConnectionBlocking,
}

/// Drive a blocking closure, optionally exiting the Tokio async context first.
///
/// `SqliteConnectionBlocking::execute / query / execute_batch` internally call
/// `tokio::runtime::Runtime::block_on` via `block_local`.  That call panics if
/// this thread is already inside a Tokio runtime.  `block_in_place` temporarily
/// moves the thread out of the multi-thread scheduler, allowing `block_on` to
/// succeed.  When there is no current runtime (plain `#[test]`) we call
/// directly.
///
/// Requires a `multi_thread` Tokio runtime when one is present — all async
/// test fixtures that exercise the persistence layer must use
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

impl Inner {
    pub(crate) fn exec(&self, sql: &str, params: &[&dyn ToSqlValue]) -> Result<u64> {
        run_blocking(|| {
            self.conn
                .execute(sql, params)
                .map_err(|e| FarmError::Database(e.to_string()))
        })
    }

    pub(crate) fn query(
        &self,
        sql: &str,
        params: &[&dyn ToSqlValue],
    ) -> Result<Vec<oxisql_core::Row>> {
        run_blocking(|| {
            self.conn
                .query(sql, params)
                .map_err(|e| FarmError::Database(e.to_string()))
        })
    }

    pub(crate) fn execute_batch(&self, sql: &str) -> Result<()> {
        // `execute_batch` returns the affected-row count (`u64`) under
        // oxisql 0.3.x; we discard it where `Result<()>` is expected.
        run_blocking(|| {
            self.conn
                .execute_batch(sql)
                .map(|_| ())
                .map_err(|e| FarmError::Database(e.to_string()))
        })
    }
}

// ---------------------------------------------------------------------------
// Column helpers
// ---------------------------------------------------------------------------

fn col_text(row: &oxisql_core::Row, idx: usize) -> Result<String> {
    match row.get_by_index(idx) {
        Some(Value::Text(s)) => Ok(s.clone()),
        Some(other) => Err(FarmError::Database(format!(
            "column {idx}: expected text, got {}",
            other.type_name()
        ))),
        None => Err(FarmError::Database(format!(
            "column {idx} missing from row"
        ))),
    }
}

fn col_i64(row: &oxisql_core::Row, idx: usize) -> Result<i64> {
    match row.get_by_index(idx) {
        Some(Value::I64(n)) => Ok(*n),
        Some(Value::Null) | None => Ok(0),
        Some(other) => Err(FarmError::Database(format!(
            "column {idx}: expected integer, got {}",
            other.type_name()
        ))),
    }
}

fn col_opt_i64(row: &oxisql_core::Row, idx: usize) -> Option<i64> {
    match row.get_by_index(idx) {
        Some(Value::I64(n)) => Some(*n),
        _ => None,
    }
}

fn col_opt_text(row: &oxisql_core::Row, idx: usize) -> Option<String> {
    match row.get_by_index(idx) {
        Some(Value::Text(s)) => Some(s.clone()),
        _ => None,
    }
}

fn col_opt_blob(row: &oxisql_core::Row, idx: usize) -> Option<Vec<u8>> {
    match row.get_by_index(idx) {
        Some(Value::Blob(b)) => Some(b.clone()),
        Some(Value::Text(s)) => Some(s.as_bytes().to_vec()),
        _ => None,
    }
}

fn col_count(rows: &[oxisql_core::Row]) -> u64 {
    match rows.first().and_then(|r| r.get_by_index(0)) {
        Some(Value::I64(n)) if *n >= 0 => *n as u64,
        _ => 0,
    }
}

fn ts_to_dt(ts: i64) -> Option<DateTime<Utc>> {
    DateTime::from_timestamp(ts, 0)
}

// ---------------------------------------------------------------------------
// Database
// ---------------------------------------------------------------------------

/// Compatibility type alias.
#[allow(dead_code)]
pub(crate) type DbPool = Arc<Mutex<Inner>>;

/// Database manager for farm persistence.
pub struct Database {
    inner: Arc<Mutex<Inner>>,
}

impl Database {
    fn with_inner<F, T>(&self, f: F) -> Result<T>
    where
        F: FnOnce(&Inner) -> Result<T>,
    {
        let guard = self
            .inner
            .lock()
            .map_err(|_| FarmError::Database("mutex poisoned".to_string()))?;
        f(&guard)
    }

    /// Create a new database connection backed by *path*.
    ///
    /// # Errors
    ///
    /// Returns an error if the database cannot be opened.
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_str = path
            .as_ref()
            .to_str()
            .ok_or_else(|| FarmError::Database("non-UTF-8 path".to_string()))?;

        let conn = SqliteConnectionBlocking::open(path_str)
            .map_err(|e| FarmError::Database(e.to_string()))?;

        let db = Self {
            inner: Arc::new(Mutex::new(Inner { conn })),
        };
        db.initialize_schema()?;
        Ok(db)
    }

    /// Create an in-memory database (for testing).
    ///
    /// # Errors
    ///
    /// Returns an error if the database cannot be created.
    pub fn in_memory() -> Result<Self> {
        let conn = SqliteConnectionBlocking::open_memory()
            .map_err(|e| FarmError::Database(e.to_string()))?;

        let db = Self {
            inner: Arc::new(Mutex::new(Inner { conn })),
        };
        db.initialize_schema()?;
        Ok(db)
    }

    fn initialize_schema(&self) -> Result<()> {
        self.with_inner(|inner| Schema::create_tables(inner))
    }

    /// Insert a new job.
    ///
    /// # Errors
    ///
    /// Returns an error if the insertion fails.
    pub fn insert_job(&self, job: &JobRecord) -> Result<()> {
        let id_s = job.id.to_string();
        let type_s = job.job_type.to_string();
        let state_s = job.state.to_string();
        let prio_i = i64::from(i32::from(job.priority));
        let params_s = serde_json::to_string(&job.parameters)?;
        let meta_s = serde_json::to_string(&job.metadata)?;
        let created_ts = job.created_at.timestamp();
        let deadline_ts: Option<i64> = job.deadline.map(|d| d.timestamp());

        self.with_inner(|inner| {
            inner.exec(
                "INSERT INTO jobs
                 (id, job_type, state, priority, input_path, output_path,
                  parameters, metadata, created_at, deadline)
                 VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)",
                &[
                    &id_s,
                    &type_s,
                    &state_s,
                    &prio_i,
                    &job.input_path.as_str(),
                    &job.output_path.as_str(),
                    &params_s,
                    &meta_s,
                    &created_ts,
                    &deadline_ts as &dyn ToSqlValue,
                ],
            )?;
            Ok(())
        })
    }

    /// Update job state.
    ///
    /// # Errors
    ///
    /// Returns an error if the job is not found or the update fails.
    pub fn update_job_state(&self, job_id: JobId, state: JobState) -> Result<()> {
        let id_s = job_id.to_string();
        let state_s = state.to_string();
        let now = Utc::now().timestamp();

        let updated = self.with_inner(|inner| match state {
            JobState::Running => inner.exec(
                "UPDATE jobs SET state = $1, started_at = $2 WHERE id = $3",
                &[&state_s, &now, &id_s],
            ),
            JobState::Completed
            | JobState::CompletedWithWarnings
            | JobState::Failed
            | JobState::Cancelled => inner.exec(
                "UPDATE jobs SET state = $1, completed_at = $2 WHERE id = $3",
                &[&state_s, &now, &id_s],
            ),
            _ => inner.exec(
                "UPDATE jobs SET state = $1 WHERE id = $2",
                &[&state_s, &id_s],
            ),
        })?;

        if updated == 0 {
            return Err(FarmError::NotFound(format!("Job {job_id} not found")));
        }
        Ok(())
    }

    /// Get job by ID.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn get_job(&self, job_id: JobId) -> Result<Option<JobRecord>> {
        let id_s = job_id.to_string();
        let rows = self.with_inner(|inner| {
            inner.query(
                "SELECT id, job_type, state, priority, input_path, output_path,
                 parameters, metadata, created_at, started_at, completed_at, deadline
                 FROM jobs WHERE id = $1",
                &[&id_s],
            )
        })?;

        if rows.is_empty() {
            return Ok(None);
        }
        let row = &rows[0];
        Ok(Some(row_to_job_record(row)?))
    }

    /// List jobs with optional filters.
    ///
    /// Dynamic query: each optional parameter appends a `$N` placeholder.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn list_jobs(
        &self,
        state_filter: Option<JobState>,
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> Result<Vec<JobRecord>> {
        // oxisqlite does not support parameterised LIMIT / OFFSET clauses, so
        // those values are inlined as integer literals.  `state_filter` comes
        // from a controlled enum so it is also safe to inline directly.
        let state_s: Option<String> = state_filter.map(|s| s.to_string());

        let mut sql = "SELECT id, job_type, state, priority, input_path, output_path,
                       parameters, metadata, created_at, started_at, completed_at, deadline
                       FROM jobs"
            .to_string();

        if let Some(ref st) = state_s {
            // The value originates from the JobState enum's Display impl —
            // safe to inline.  Use single-quoted SQL string literal.
            let escaped = st.replace('\'', "''");
            sql.push_str(&format!(" WHERE state = '{escaped}'"));
        }

        sql.push_str(" ORDER BY created_at DESC");

        if let Some(lim) = limit {
            sql.push_str(&format!(" LIMIT {lim}"));
        }

        if let Some(off) = offset {
            sql.push_str(&format!(" OFFSET {off}"));
        }

        let rows = self.with_inner(|inner| inner.query(&sql, &[]))?;
        rows.iter().map(row_to_job_record).collect()
    }

    /// Insert a new task.
    ///
    /// # Errors
    ///
    /// Returns an error if the insertion fails.
    pub fn insert_task(&self, task: &TaskRecord) -> Result<()> {
        let id_s = task.id.to_string();
        let job_id_s = task.job_id.to_string();
        let state_s = task.state.to_string();
        let worker_s: Option<String> = task.worker_id.as_ref().map(|w| w.to_string());
        let prio_i = i64::from(i32::from(task.priority));
        let created_ts = task.created_at.timestamp();
        let assigned_ts: Option<i64> = task.assigned_at.map(|d| d.timestamp());
        let retry_i = i64::from(task.retry_count);
        // Clone payload so we can hold the Vec in a named binding.
        let payload = task.payload.clone();

        self.with_inner(|inner| {
            inner.exec(
                "INSERT INTO tasks
                 (id, job_id, state, worker_id, task_type, payload,
                  priority, created_at, assigned_at, retry_count)
                 VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)",
                &[
                    &id_s,
                    &job_id_s,
                    &state_s,
                    &worker_s.as_deref() as &dyn ToSqlValue,
                    &task.task_type.as_str(),
                    &payload as &dyn ToSqlValue,
                    &prio_i,
                    &created_ts,
                    &assigned_ts as &dyn ToSqlValue,
                    &retry_i,
                ],
            )?;
            Ok(())
        })
    }

    /// Update task state.
    ///
    /// # Errors
    ///
    /// Returns an error if the update fails.
    pub fn update_task_state(
        &self,
        task_id: TaskId,
        state: TaskState,
        worker_id: Option<WorkerId>,
    ) -> Result<()> {
        let id_s = task_id.to_string();
        let state_s = state.to_string();
        let worker_s: Option<String> = worker_id.map(|w| w.to_string());
        let now = Utc::now().timestamp();
        let assigned_ts: Option<i64> = if state == TaskState::Assigned {
            Some(now)
        } else {
            None
        };

        self.with_inner(|inner| {
            inner.exec(
                "UPDATE tasks SET state = $1, worker_id = $2, assigned_at = $3 WHERE id = $4",
                &[
                    &state_s,
                    &worker_s.as_deref() as &dyn ToSqlValue,
                    &assigned_ts as &dyn ToSqlValue,
                    &id_s,
                ],
            )?;
            Ok(())
        })
    }

    /// Get pending tasks ordered by priority then creation time.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn get_pending_tasks(&self, limit: usize) -> Result<Vec<TaskRecord>> {
        // oxisqlite does not support parameterized LIMIT clauses, so we
        // inline the value directly.  The value is a caller-controlled usize
        // (not user input) so there is no injection risk.
        let sql = format!(
            "SELECT id, job_id, state, worker_id, task_type, payload, priority,
             created_at, assigned_at, retry_count
             FROM tasks
             WHERE state = 'Pending'
             ORDER BY priority DESC, created_at ASC
             LIMIT {limit}"
        );
        let rows = self.with_inner(|inner| inner.query(&sql, &[]))?;
        rows.iter().map(row_to_task_record).collect()
    }

    /// Get tasks for a specific job.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn get_job_tasks(&self, job_id: JobId) -> Result<Vec<TaskRecord>> {
        let id_s = job_id.to_string();
        let rows = self.with_inner(|inner| {
            inner.query(
                "SELECT id, job_id, state, worker_id, task_type, payload, priority,
                 created_at, assigned_at, retry_count
                 FROM tasks WHERE job_id = $1",
                &[&id_s],
            )
        })?;

        rows.iter().map(row_to_task_record).collect()
    }

    /// Increment task retry count; returns the new count.
    ///
    /// # Errors
    ///
    /// Returns an error if the update or select fails.
    pub fn increment_task_retry(&self, task_id: TaskId) -> Result<u32> {
        let id_s = task_id.to_string();
        self.with_inner(|inner| {
            inner.exec(
                "UPDATE tasks SET retry_count = retry_count + 1 WHERE id = $1",
                &[&id_s],
            )
        })?;

        let rows = self.with_inner(|inner| {
            inner.query("SELECT retry_count FROM tasks WHERE id = $1", &[&id_s])
        })?;

        let count = col_i64(
            rows.first()
                .ok_or_else(|| FarmError::NotFound(format!("task {task_id} not found")))?,
            0,
        )?;
        Ok(count as u32)
    }

    /// Get job statistics.
    ///
    /// # Errors
    ///
    /// Returns an error if the queries fail.
    pub fn get_job_stats(&self) -> Result<JobStats> {
        let count_state = |state: &str| -> Result<u64> {
            let rows = self.with_inner(|inner| {
                inner.query("SELECT COUNT(*) FROM jobs WHERE state = $1", &[&state])
            })?;
            Ok(col_count(&rows))
        };

        let total_rows = self.with_inner(|inner| inner.query("SELECT COUNT(*) FROM jobs", &[]))?;

        Ok(JobStats {
            total: col_count(&total_rows),
            pending: count_state("Pending")?,
            queued: count_state("Queued")?,
            running: count_state("Running")?,
            completed: count_state("Completed")?,
            failed: count_state("Failed")?,
        })
    }
}

// ---------------------------------------------------------------------------
// Row converters
// ---------------------------------------------------------------------------

fn row_to_job_record(row: &oxisql_core::Row) -> Result<JobRecord> {
    let id_s = col_text(row, 0)?;
    let job_type_s = col_text(row, 1)?;
    let state_s = col_text(row, 2)?;
    let priority_i = col_i64(row, 3)?;
    let input_path = col_text(row, 4)?;
    let output_path = col_text(row, 5)?;
    let params_s = col_text(row, 6)?;
    let meta_s = col_text(row, 7)?;
    let created_ts = col_i64(row, 8)?;
    let started_ts = col_opt_i64(row, 9);
    let completed_ts = col_opt_i64(row, 10);
    let deadline_ts = col_opt_i64(row, 11);

    let id_uuid = uuid::Uuid::parse_str(&id_s)
        .map_err(|e| FarmError::Database(format!("uuid parse: {e}")))?;

    let priority = Priority::try_from(priority_i as i32)
        .map_err(|_| FarmError::Database(format!("invalid priority: {priority_i}")))?;

    let parameters: HashMap<String, serde_json::Value> = serde_json::from_str(&params_s)
        .map_err(|e| FarmError::Database(format!("params json: {e}")))?;
    let metadata: HashMap<String, String> = serde_json::from_str(&meta_s)
        .map_err(|e| FarmError::Database(format!("metadata json: {e}")))?;

    let created_at = ts_to_dt(created_ts)
        .ok_or_else(|| FarmError::Database(format!("invalid timestamp: {created_ts}")))?;

    Ok(JobRecord {
        id: JobId::from_uuid(id_uuid),
        job_type: parse_job_type(&job_type_s),
        state: parse_job_state(&state_s),
        priority,
        input_path,
        output_path,
        parameters,
        metadata,
        created_at,
        started_at: started_ts.and_then(ts_to_dt),
        completed_at: completed_ts.and_then(ts_to_dt),
        deadline: deadline_ts.and_then(ts_to_dt),
    })
}

fn row_to_task_record(row: &oxisql_core::Row) -> Result<TaskRecord> {
    let id_s = col_text(row, 0)?;
    let job_id_s = col_text(row, 1)?;
    let state_s = col_text(row, 2)?;
    let worker_s = col_opt_text(row, 3);
    let task_type = col_text(row, 4)?;
    let payload = col_opt_blob(row, 5).unwrap_or_default();
    let priority_i = col_i64(row, 6)?;
    let created_ts = col_i64(row, 7)?;
    let assigned_ts = col_opt_i64(row, 8);
    let retry_i = col_i64(row, 9)?;

    let id_uuid = uuid::Uuid::parse_str(&id_s)
        .map_err(|e| FarmError::Database(format!("uuid parse id: {e}")))?;
    let job_id_uuid = uuid::Uuid::parse_str(&job_id_s)
        .map_err(|e| FarmError::Database(format!("uuid parse job_id: {e}")))?;

    let priority = Priority::try_from(priority_i as i32)
        .map_err(|_| FarmError::Database(format!("invalid priority: {priority_i}")))?;

    let created_at = ts_to_dt(created_ts)
        .ok_or_else(|| FarmError::Database(format!("invalid timestamp: {created_ts}")))?;

    Ok(TaskRecord {
        id: TaskId::from_uuid(id_uuid),
        job_id: JobId::from_uuid(job_id_uuid),
        state: parse_task_state(&state_s),
        worker_id: worker_s.map(WorkerId::new),
        task_type,
        payload,
        priority,
        created_at,
        assigned_at: assigned_ts.and_then(ts_to_dt),
        retry_count: retry_i as u32,
    })
}

// ---------------------------------------------------------------------------
// Records
// ---------------------------------------------------------------------------

/// Job record in the database.
#[derive(Debug, Clone)]
pub struct JobRecord {
    /// Job identifier.
    pub id: JobId,
    /// Job type.
    pub job_type: JobType,
    /// Current state.
    pub state: JobState,
    /// Scheduling priority.
    pub priority: Priority,
    /// Input file path.
    pub input_path: String,
    /// Output file path.
    pub output_path: String,
    /// JSON parameters map.
    pub parameters: HashMap<String, serde_json::Value>,
    /// Metadata key-value map.
    pub metadata: HashMap<String, String>,
    /// Creation time.
    pub created_at: DateTime<Utc>,
    /// Time the job started running.
    pub started_at: Option<DateTime<Utc>>,
    /// Time the job finished.
    pub completed_at: Option<DateTime<Utc>>,
    /// Optional deadline.
    pub deadline: Option<DateTime<Utc>>,
}

/// Task record in the database.
#[derive(Debug, Clone)]
pub struct TaskRecord {
    /// Task identifier.
    pub id: TaskId,
    /// Parent job identifier.
    pub job_id: JobId,
    /// Current state.
    pub state: TaskState,
    /// Assigned worker.
    pub worker_id: Option<WorkerId>,
    /// Task type string.
    pub task_type: String,
    /// Raw payload bytes.
    pub payload: Vec<u8>,
    /// Scheduling priority.
    pub priority: Priority,
    /// Creation time.
    pub created_at: DateTime<Utc>,
    /// Time the task was assigned.
    pub assigned_at: Option<DateTime<Utc>>,
    /// Number of retry attempts.
    pub retry_count: u32,
}

/// Aggregated job statistics.
#[derive(Debug, Clone)]
pub struct JobStats {
    /// Total jobs.
    pub total: u64,
    /// Pending jobs.
    pub pending: u64,
    /// Queued jobs.
    pub queued: u64,
    /// Running jobs.
    pub running: u64,
    /// Completed jobs.
    pub completed: u64,
    /// Failed jobs.
    pub failed: u64,
}

// ---------------------------------------------------------------------------
// Parse helpers
// ---------------------------------------------------------------------------

fn parse_job_type(s: &str) -> JobType {
    match s {
        "VideoTranscode" => JobType::VideoTranscode,
        "AudioTranscode" => JobType::AudioTranscode,
        "ThumbnailGeneration" => JobType::ThumbnailGeneration,
        "QcValidation" => JobType::QcValidation,
        "MediaAnalysis" => JobType::MediaAnalysis,
        "ContentFingerprinting" => JobType::ContentFingerprinting,
        "MultiOutputTranscode" => JobType::MultiOutputTranscode,
        _ => JobType::VideoTranscode,
    }
}

fn parse_job_state(s: &str) -> JobState {
    match s {
        "Pending" => JobState::Pending,
        "Queued" => JobState::Queued,
        "Running" => JobState::Running,
        "Completed" => JobState::Completed,
        "CompletedWithWarnings" => JobState::CompletedWithWarnings,
        "Failed" => JobState::Failed,
        "Cancelled" => JobState::Cancelled,
        "Paused" => JobState::Paused,
        _ => JobState::Pending,
    }
}

fn parse_task_state(s: &str) -> TaskState {
    match s {
        "Pending" => TaskState::Pending,
        "Assigned" => TaskState::Assigned,
        "Running" => TaskState::Running,
        "Completed" => TaskState::Completed,
        "Failed" => TaskState::Failed,
        _ => TaskState::Pending,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_database_creation() -> Result<()> {
        let db = Database::in_memory()?;
        let stats = db.get_job_stats()?;
        assert_eq!(stats.total, 0);
        Ok(())
    }

    #[test]
    fn test_job_insertion() -> Result<()> {
        let db = Database::in_memory()?;
        let job = JobRecord {
            id: JobId::new(),
            job_type: JobType::VideoTranscode,
            state: JobState::Pending,
            priority: Priority::Normal,
            input_path: "/input/test.mp4".to_string(),
            output_path: "/output/test.mp4".to_string(),
            parameters: HashMap::new(),
            metadata: HashMap::new(),
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            deadline: None,
        };

        db.insert_job(&job)?;
        let retrieved = db
            .get_job(job.id)?
            .ok_or_else(|| FarmError::NotFound(format!("Job {} not found", job.id)))?;
        assert_eq!(retrieved.id, job.id);
        assert_eq!(retrieved.state, JobState::Pending);
        Ok(())
    }

    #[test]
    fn test_job_state_update() -> Result<()> {
        let db = Database::in_memory()?;
        let job = JobRecord {
            id: JobId::new(),
            job_type: JobType::VideoTranscode,
            state: JobState::Pending,
            priority: Priority::Normal,
            input_path: "/input/test.mp4".to_string(),
            output_path: "/output/test.mp4".to_string(),
            parameters: HashMap::new(),
            metadata: HashMap::new(),
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            deadline: None,
        };

        db.insert_job(&job)?;
        db.update_job_state(job.id, JobState::Running)?;

        let retrieved = db
            .get_job(job.id)?
            .ok_or_else(|| FarmError::NotFound(format!("Job {} not found", job.id)))?;
        assert_eq!(retrieved.state, JobState::Running);
        assert!(retrieved.started_at.is_some());
        Ok(())
    }

    #[test]
    fn test_task_insertion() -> Result<()> {
        let db = Database::in_memory()?;

        let job_id = JobId::new();
        let job = JobRecord {
            id: job_id,
            job_type: JobType::VideoTranscode,
            priority: Priority::Normal,
            state: JobState::Pending,
            input_path: "/test/input.mp4".to_string(),
            output_path: "/test/output.mp4".to_string(),
            parameters: HashMap::new(),
            metadata: HashMap::new(),
            deadline: None,
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
        };
        db.insert_job(&job)?;

        let task = TaskRecord {
            id: TaskId::new(),
            job_id,
            state: TaskState::Pending,
            worker_id: None,
            task_type: "transcode".to_string(),
            payload: vec![1, 2, 3],
            priority: Priority::Normal,
            created_at: Utc::now(),
            assigned_at: None,
            retry_count: 0,
        };

        db.insert_task(&task)?;
        let tasks = db.get_pending_tasks(10)?;
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].id, task.id);
        Ok(())
    }

    #[test]
    fn test_task_priority_ordering() -> Result<()> {
        let db = Database::in_memory()?;

        let job_id = JobId::new();
        let job = JobRecord {
            id: job_id,
            job_type: JobType::VideoTranscode,
            priority: Priority::Normal,
            state: JobState::Pending,
            input_path: "/test/input.mp4".to_string(),
            output_path: "/test/output.mp4".to_string(),
            parameters: HashMap::new(),
            metadata: HashMap::new(),
            deadline: None,
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
        };
        db.insert_job(&job)?;

        let task_low = TaskRecord {
            id: TaskId::new(),
            job_id,
            state: TaskState::Pending,
            worker_id: None,
            task_type: "transcode".to_string(),
            payload: vec![],
            priority: Priority::Low,
            created_at: Utc::now(),
            assigned_at: None,
            retry_count: 0,
        };

        let task_high = TaskRecord {
            id: TaskId::new(),
            job_id,
            state: TaskState::Pending,
            worker_id: None,
            task_type: "transcode".to_string(),
            payload: vec![],
            priority: Priority::High,
            created_at: Utc::now(),
            assigned_at: None,
            retry_count: 0,
        };

        db.insert_task(&task_low)?;
        db.insert_task(&task_high)?;

        let tasks = db.get_pending_tasks(10)?;
        assert_eq!(tasks.len(), 2);
        assert_eq!(tasks[0].priority, Priority::High);
        assert_eq!(tasks[1].priority, Priority::Low);
        Ok(())
    }

    #[test]
    fn test_retry_count() -> Result<()> {
        let db = Database::in_memory()?;

        let job_id = JobId::new();
        let job = JobRecord {
            id: job_id,
            job_type: JobType::VideoTranscode,
            priority: Priority::Normal,
            state: JobState::Pending,
            input_path: "/test/input.mp4".to_string(),
            output_path: "/test/output.mp4".to_string(),
            parameters: HashMap::new(),
            metadata: HashMap::new(),
            deadline: None,
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
        };
        db.insert_job(&job)?;

        let task = TaskRecord {
            id: TaskId::new(),
            job_id,
            state: TaskState::Pending,
            worker_id: None,
            task_type: "transcode".to_string(),
            payload: vec![],
            priority: Priority::Normal,
            created_at: Utc::now(),
            assigned_at: None,
            retry_count: 0,
        };

        db.insert_task(&task)?;
        let count1 = db.increment_task_retry(task.id)?;
        assert_eq!(count1, 1);

        let count2 = db.increment_task_retry(task.id)?;
        assert_eq!(count2, 2);
        Ok(())
    }
}
