// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Database persistence for job queue.

use crate::job::{Job, JobStatus, Priority};
use chrono::{DateTime, Utc};
use oxisql_core::{ToSqlValue, Value};
use oxisql_sqlite_compat::SqliteConnectionBlocking;
use std::path::Path;
use std::sync::Mutex;
use thiserror::Error;
use uuid::Uuid;

/// Persistence errors
#[derive(Debug, Error)]
pub enum PersistenceError {
    /// Database error
    #[error("Database error: {0}")]
    Database(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Job not found
    #[error("Job not found: {0}")]
    JobNotFound(Uuid),
}

/// Map an OxiSQL error to [`PersistenceError`].
fn map_oxi(e: impl std::fmt::Display) -> PersistenceError {
    PersistenceError::Database(e.to_string())
}

/// Result type for persistence operations
pub type Result<T> = std::result::Result<T, PersistenceError>;

/// Drive a blocking closure safely from any calling context, including from
/// within a running Tokio async task.
///
/// `SqliteConnectionBlocking::execute/query/execute_batch` use an internal
/// `block_local` helper that builds a fresh `current_thread` Tokio runtime and
/// calls `block_on` on it. That panics when called from within an already-active
/// Tokio runtime ("`Cannot start a runtime from within a runtime`"), because
/// Tokio forbids nesting a blocking `block_on` call on the same thread that is
/// driving async tasks.
///
/// `tokio::task::block_in_place` solves this by temporarily moving the current
/// async task off the executor thread, giving the closure a clean synchronous
/// context to block in. It requires a `multi_thread` runtime; callers inside a
/// `current_thread` runtime must not call persistence operations directly from
/// async code (they must use `spawn_blocking` instead) — that is a usage error
/// and will be caught at runtime.
///
/// When called from a fully synchronous context (no active Tokio runtime),
/// the closure is invoked directly with no overhead.
fn run_blocking<F, T>(f: F) -> T
where
    F: FnOnce() -> T,
{
    match tokio::runtime::Handle::try_current() {
        Ok(_) => tokio::task::block_in_place(f),
        Err(_) => f(),
    }
}

/// Extract an `i64` from a result row at the given column index.
fn col_i64(row: &oxisql_core::Row, idx: usize) -> Result<i64> {
    match row.get_by_index(idx) {
        Some(Value::I64(n)) => Ok(*n),
        Some(other) => Err(PersistenceError::Database(format!(
            "column {idx}: expected integer, got {}",
            other.type_name()
        ))),
        None => Err(PersistenceError::Database(format!(
            "column {idx} missing from result row"
        ))),
    }
}

/// Extract a `String` from a result row at the given column index.
fn col_text(row: &oxisql_core::Row, idx: usize) -> Result<String> {
    match row.get_by_index(idx) {
        Some(Value::Text(s)) => Ok(s.clone()),
        Some(other) => Err(PersistenceError::Database(format!(
            "column {idx}: expected text, got {}",
            other.type_name()
        ))),
        None => Err(PersistenceError::Database(format!(
            "column {idx} missing from result row"
        ))),
    }
}

/// Extract an optional `String` from a result row (NULL or missing → `None`).
fn col_opt_text(row: &oxisql_core::Row, idx: usize) -> Result<Option<String>> {
    match row.get_by_index(idx) {
        Some(Value::Null) | None => Ok(None),
        Some(Value::Text(s)) => Ok(Some(s.clone())),
        Some(other) => Err(PersistenceError::Database(format!(
            "column {idx}: expected text or null, got {}",
            other.type_name()
        ))),
    }
}

/// Thread-safe inner state.
struct Inner {
    conn: SqliteConnectionBlocking,
}

impl Inner {
    fn exec(&self, sql: &str, params: &[&dyn ToSqlValue]) -> Result<u64> {
        run_blocking(|| self.conn.execute(sql, params)).map_err(map_oxi)
    }

    fn query(&self, sql: &str, params: &[&dyn ToSqlValue]) -> Result<Vec<oxisql_core::Row>> {
        run_blocking(|| self.conn.query(sql, params)).map_err(map_oxi)
    }
}

/// Job persistence layer using SQLite (Pure-Rust via OxiSQL).
///
/// Thread-safe: a `Mutex` serializes all database operations so that the
/// layer can be wrapped in `Arc` and shared across threads.
pub struct JobPersistence {
    inner: Mutex<Inner>,
}

impl JobPersistence {
    fn with_inner<F, T>(&self, f: F) -> Result<T>
    where
        F: FnOnce(&Inner) -> Result<T>,
    {
        let guard = self
            .inner
            .lock()
            .map_err(|_| PersistenceError::Database("mutex poisoned".to_string()))?;
        f(&guard)
    }

    /// Create a new persistence layer.
    ///
    /// # Errors
    ///
    /// Returns an error if database initialization fails
    pub fn new<P: AsRef<Path>>(db_path: P) -> Result<Self> {
        let path_str = db_path.as_ref().to_string_lossy().into_owned();
        let conn = SqliteConnectionBlocking::open(&path_str).map_err(map_oxi)?;
        let p = Self {
            inner: Mutex::new(Inner { conn }),
        };
        p.initialize_schema()?;
        Ok(p)
    }

    /// Create an in-memory persistence layer (for testing).
    ///
    /// # Errors
    ///
    /// Returns an error if database initialization fails
    pub fn in_memory() -> Result<Self> {
        let conn = SqliteConnectionBlocking::open_memory().map_err(map_oxi)?;
        let p = Self {
            inner: Mutex::new(Inner { conn }),
        };
        p.initialize_schema()?;
        Ok(p)
    }

    /// Helper: execute a statement (serialized via mutex).
    fn exec(&self, sql: &str, params: &[&dyn ToSqlValue]) -> Result<u64> {
        self.with_inner(|i| i.exec(sql, params))
    }

    /// Helper: query returning all rows (serialized via mutex).
    fn query(&self, sql: &str, params: &[&dyn ToSqlValue]) -> Result<Vec<oxisql_core::Row>> {
        self.with_inner(|i| i.query(sql, params))
    }

    /// Initialize database schema
    fn initialize_schema(&self) -> Result<()> {
        self.with_inner(|inner| {
            // Create table first. `execute_batch` returns the affected-row count
            // (`u64`); discard it with `.map(|_| ())` for the DDL `-> Result<()>`
            // contract.
            run_blocking(|| {
                inner.conn.execute_batch(
                    "CREATE TABLE IF NOT EXISTS jobs (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        priority INTEGER NOT NULL,
                        status TEXT NOT NULL,
                        payload_type TEXT NOT NULL,
                        payload_data TEXT NOT NULL,
                        retry_policy TEXT NOT NULL,
                        resource_quota TEXT NOT NULL,
                        condition_type TEXT NOT NULL,
                        condition_data TEXT NOT NULL,
                        dependencies TEXT NOT NULL,
                        tags TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        scheduled_at TEXT,
                        started_at TEXT,
                        completed_at TEXT,
                        deadline TEXT,
                        attempts INTEGER NOT NULL,
                        error TEXT,
                        progress INTEGER NOT NULL,
                        worker_id TEXT,
                        next_jobs TEXT NOT NULL
                    );",
                )
            })
            .map(|_| ())
            .map_err(map_oxi)?;
            // Create indexes individually — oxisqlite's execute_batch may not
            // properly honour IF NOT EXISTS on secondary DDL statements when the
            // index already exists, so we use separate execute calls and silently
            // ignore "already exists" errors for idempotent re-opens.
            let indexes = [
                "CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)",
                "CREATE INDEX IF NOT EXISTS idx_jobs_priority ON jobs(priority)",
                "CREATE INDEX IF NOT EXISTS idx_jobs_scheduled ON jobs(scheduled_at)",
                "CREATE INDEX IF NOT EXISTS idx_jobs_deadline ON jobs(deadline)",
            ];
            for ddl in &indexes {
                let result = run_blocking(|| inner.conn.execute(ddl, &[]));
                if let Err(e) = result {
                    let msg = e.to_string();
                    if !msg.contains("already exists") {
                        return Err(map_oxi(e));
                    }
                }
            }
            Ok(())
        })
    }

    /// Save a job to the database
    ///
    /// # Errors
    ///
    /// Returns an error if database operation fails
    pub fn save_job(&self, job: &Job) -> Result<()> {
        let payload_type = job.payload.job_type();
        let payload_data = serde_json::to_string(&job.payload)?;
        let retry_policy = serde_json::to_string(&job.retry_policy)?;
        let resource_quota = serde_json::to_string(&job.resource_quota)?;
        let condition_data = serde_json::to_string(&job.condition)?;
        let dependencies = serde_json::to_string(&job.dependencies)?;
        let tags = serde_json::to_string(&job.tags)?;
        let next_jobs = serde_json::to_string(&job.next_jobs)?;

        let id_s = job.id.to_string();
        let priority_i = job.priority as i64;
        let status_s = job.status.to_string();
        let created_at_s = job.created_at.to_rfc3339();
        let scheduled_at_s = job.scheduled_at.map(|dt| dt.to_rfc3339());
        let started_at_s = job.started_at.map(|dt| dt.to_rfc3339());
        let completed_at_s = job.completed_at.map(|dt| dt.to_rfc3339());
        let deadline_s = job.deadline.map(|dt| dt.to_rfc3339());
        let attempts_i = job.attempts as i64;
        let progress_i = job.progress as i64;

        let scheduled_at_ref: Option<&str> = scheduled_at_s.as_deref();
        let started_at_ref: Option<&str> = started_at_s.as_deref();
        let completed_at_ref: Option<&str> = completed_at_s.as_deref();
        let deadline_ref: Option<&str> = deadline_s.as_deref();
        let error_ref: Option<&str> = job.error.as_deref();
        let worker_id_ref: Option<&str> = job.worker_id.as_deref();

        self.exec(
            "INSERT OR REPLACE INTO jobs (
                id, name, priority, status, payload_type, payload_data,
                retry_policy, resource_quota, condition_type, condition_data,
                dependencies, tags, created_at, scheduled_at, started_at,
                completed_at, deadline, attempts, error, progress, worker_id, next_jobs
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22
            )",
            &[
                &id_s,
                &job.name,
                &priority_i,
                &status_s,
                &payload_type,
                &payload_data,
                &retry_policy,
                &resource_quota,
                &"condition",
                &condition_data,
                &dependencies,
                &tags,
                &created_at_s,
                &scheduled_at_ref,
                &started_at_ref,
                &completed_at_ref,
                &deadline_ref,
                &attempts_i,
                &error_ref,
                &progress_i,
                &worker_id_ref,
                &next_jobs,
            ],
        )?;
        Ok(())
    }

    /// Get a job by ID
    ///
    /// # Errors
    ///
    /// Returns an error if database operation fails or job is not found
    pub fn get_job(&self, id: Uuid) -> Result<Job> {
        let id_s = id.to_string();
        let rows = self.query("SELECT * FROM jobs WHERE id = $1", &[&id_s])?;
        let row = rows
            .into_iter()
            .next()
            .ok_or(PersistenceError::JobNotFound(id))?;
        Self::row_to_job(&row)
    }

    /// Delete a job by ID
    ///
    /// # Errors
    ///
    /// Returns an error if database operation fails
    pub fn delete_job(&self, id: Uuid) -> Result<()> {
        let id_s = id.to_string();
        self.exec("DELETE FROM jobs WHERE id = $1", &[&id_s])?;
        Ok(())
    }

    /// Save a batch of jobs — sequential executes (no transaction available in compat alpha).
    ///
    /// # Errors
    ///
    /// Returns an error if any serialization or database operation fails.
    pub fn save_jobs_batch(&self, jobs: &[Job]) -> Result<()> {
        for job in jobs {
            self.save_job(job)?;
        }
        Ok(())
    }

    /// Get jobs by status
    ///
    /// # Errors
    ///
    /// Returns an error if database operation fails
    pub fn get_jobs_by_status(&self, status: JobStatus) -> Result<Vec<Job>> {
        let status_s = status.to_string();
        let rows = self.query("SELECT * FROM jobs WHERE status = $1", &[&status_s])?;
        rows.iter().map(Self::row_to_job).collect()
    }

    /// Get pending jobs ordered by priority
    ///
    /// # Errors
    ///
    /// Returns an error if database operation fails
    pub fn get_pending_jobs(&self) -> Result<Vec<Job>> {
        let rows = self.query(
            "SELECT * FROM jobs WHERE status = 'pending' ORDER BY priority DESC, created_at ASC",
            &[],
        )?;
        rows.iter().map(Self::row_to_job).collect()
    }

    /// Get scheduled jobs that are ready to run
    ///
    /// # Errors
    ///
    /// Returns an error if database operation fails
    pub fn get_scheduled_jobs_ready(&self) -> Result<Vec<Job>> {
        let now = Utc::now().to_rfc3339();
        let rows = self.query(
            "SELECT * FROM jobs WHERE status = 'scheduled' AND scheduled_at <= $1 ORDER BY priority DESC, scheduled_at ASC",
            &[&now],
        )?;
        rows.iter().map(Self::row_to_job).collect()
    }

    /// Get jobs past their deadline
    ///
    /// # Errors
    ///
    /// Returns an error if database operation fails
    pub fn get_jobs_past_deadline(&self) -> Result<Vec<Job>> {
        let now = Utc::now().to_rfc3339();
        let rows = self.query(
            "SELECT * FROM jobs WHERE deadline IS NOT NULL AND deadline < $1 AND status NOT IN ('completed', 'failed', 'cancelled')",
            &[&now],
        )?;
        rows.iter().map(Self::row_to_job).collect()
    }

    /// Get all jobs
    ///
    /// # Errors
    ///
    /// Returns an error if database operation fails
    pub fn get_all_jobs(&self) -> Result<Vec<Job>> {
        let rows = self.query("SELECT * FROM jobs ORDER BY created_at DESC", &[])?;
        rows.iter().map(Self::row_to_job).collect()
    }

    /// Get jobs by tag
    ///
    /// # Errors
    ///
    /// Returns an error if database operation fails
    pub fn get_jobs_by_tag(&self, tag: &str) -> Result<Vec<Job>> {
        let pattern = format!("%\"{tag}\" %");
        let rows = self.query("SELECT * FROM jobs WHERE tags LIKE $1", &[&pattern])?;
        rows.iter().map(Self::row_to_job).collect()
    }

    /// Update job status
    ///
    /// # Errors
    ///
    /// Returns an error if database operation fails
    pub fn update_job_status(&self, id: Uuid, status: JobStatus) -> Result<()> {
        let status_s = status.to_string();
        let id_s = id.to_string();
        self.exec(
            "UPDATE jobs SET status = $1 WHERE id = $2",
            &[&status_s, &id_s],
        )?;
        Ok(())
    }

    /// Update job progress
    ///
    /// # Errors
    ///
    /// Returns an error if database operation fails
    pub fn update_job_progress(&self, id: Uuid, progress: u8) -> Result<()> {
        let progress_i = progress as i64;
        let id_s = id.to_string();
        self.exec(
            "UPDATE jobs SET progress = $1 WHERE id = $2",
            &[&progress_i, &id_s],
        )?;
        Ok(())
    }

    /// Get job count by status
    ///
    /// # Errors
    ///
    /// Returns an error if database operation fails
    pub fn count_jobs_by_status(&self, status: JobStatus) -> Result<usize> {
        let status_s = status.to_string();
        let rows = self.query("SELECT COUNT(*) FROM jobs WHERE status = $1", &[&status_s])?;
        match rows.first().and_then(|r| r.get_by_index(0)) {
            Some(Value::I64(n)) => Ok(*n as usize),
            Some(Value::Null) | None => Ok(0),
            Some(other) => Err(PersistenceError::Database(format!(
                "count_jobs_by_status: unexpected value {}",
                other.type_name()
            ))),
        }
    }

    /// Get total job count
    ///
    /// # Errors
    ///
    /// Returns an error if database operation fails
    pub fn count_jobs(&self) -> Result<usize> {
        let rows = self.query("SELECT COUNT(*) FROM jobs", &[])?;
        match rows.first().and_then(|r| r.get_by_index(0)) {
            Some(Value::I64(n)) => Ok(*n as usize),
            Some(Value::Null) | None => Ok(0),
            Some(other) => Err(PersistenceError::Database(format!(
                "count_jobs: unexpected value {}",
                other.type_name()
            ))),
        }
    }

    /// Clean up old completed jobs
    ///
    /// # Errors
    ///
    /// Returns an error if database operation fails
    pub fn cleanup_old_jobs(&self, days: i64) -> Result<usize> {
        let cutoff = (Utc::now() - chrono::Duration::days(days)).to_rfc3339();
        let affected = self.exec(
            "DELETE FROM jobs WHERE status IN ('completed', 'failed', 'cancelled') AND completed_at < $1",
            &[&cutoff],
        )?;
        Ok(affected as usize)
    }

    /// Convert a result row to a Job
    fn row_to_job(row: &oxisql_core::Row) -> Result<Job> {
        let id_s = col_text(row, 0)?;
        let payload_data = col_text(row, 5)?;
        let retry_policy_s = col_text(row, 6)?;
        let resource_quota_s = col_text(row, 7)?;
        let condition_data = col_text(row, 9)?;
        let dependencies_s = col_text(row, 10)?;
        let tags_s = col_text(row, 11)?;
        let created_at_s = col_text(row, 12)?;
        let scheduled_at_s = col_opt_text(row, 13)?;
        let started_at_s = col_opt_text(row, 14)?;
        let completed_at_s = col_opt_text(row, 15)?;
        let deadline_s = col_opt_text(row, 16)?;
        let next_jobs_s = col_text(row, 21)?;

        let priority_i = col_i64(row, 2)?;
        let priority = match priority_i {
            0 => Priority::Low,
            2 => Priority::High,
            _ => Priority::Normal,
        };

        let status_str = col_text(row, 3)?;
        let status = match status_str.as_str() {
            "running" => JobStatus::Running,
            "completed" => JobStatus::Completed,
            "failed" => JobStatus::Failed,
            "cancelled" => JobStatus::Cancelled,
            "waiting" => JobStatus::Waiting,
            "scheduled" => JobStatus::Scheduled,
            _ => JobStatus::Pending,
        };

        let attempts_i = col_i64(row, 17)?;
        let error_s = col_opt_text(row, 18)?;
        let progress_i = col_i64(row, 19)?;
        let worker_id_s = col_opt_text(row, 20)?;

        Ok(Job {
            id: Uuid::parse_str(&id_s).map_err(|e| {
                PersistenceError::Database(format!("invalid UUID in column 0: {e}"))
            })?,
            name: col_text(row, 1)?,
            priority,
            status,
            payload: serde_json::from_str(&payload_data)
                .map_err(|e| PersistenceError::Database(format!("payload_data column 5: {e}")))?,
            retry_policy: serde_json::from_str(&retry_policy_s)
                .map_err(|e| PersistenceError::Database(format!("retry_policy column 6: {e}")))?,
            resource_quota: serde_json::from_str(&resource_quota_s)
                .map_err(|e| PersistenceError::Database(format!("resource_quota column 7: {e}")))?,
            condition: serde_json::from_str(&condition_data)
                .map_err(|e| PersistenceError::Database(format!("condition column 9: {e}")))?,
            dependencies: serde_json::from_str(&dependencies_s)
                .map_err(|e| PersistenceError::Database(format!("dependencies column 10: {e}")))?,
            tags: serde_json::from_str(&tags_s)
                .map_err(|e| PersistenceError::Database(format!("tags column 11: {e}")))?,
            created_at: DateTime::parse_from_rfc3339(&created_at_s)
                .map(|dt| dt.with_timezone(&Utc))
                .map_err(|e| PersistenceError::Database(format!("created_at column 12: {e}")))?,
            scheduled_at: scheduled_at_s
                .as_ref()
                .map(|s| {
                    DateTime::parse_from_rfc3339(s)
                        .map(|dt| dt.with_timezone(&Utc))
                        .map_err(|e| {
                            PersistenceError::Database(format!("scheduled_at column 13: {e}"))
                        })
                })
                .transpose()?,
            started_at: started_at_s
                .as_ref()
                .map(|s| {
                    DateTime::parse_from_rfc3339(s)
                        .map(|dt| dt.with_timezone(&Utc))
                        .map_err(|e| {
                            PersistenceError::Database(format!("started_at column 14: {e}"))
                        })
                })
                .transpose()?,
            completed_at: completed_at_s
                .as_ref()
                .map(|s| {
                    DateTime::parse_from_rfc3339(s)
                        .map(|dt| dt.with_timezone(&Utc))
                        .map_err(|e| {
                            PersistenceError::Database(format!("completed_at column 15: {e}"))
                        })
                })
                .transpose()?,
            deadline: deadline_s
                .as_ref()
                .map(|s| {
                    DateTime::parse_from_rfc3339(s)
                        .map(|dt| dt.with_timezone(&Utc))
                        .map_err(|e| PersistenceError::Database(format!("deadline column 16: {e}")))
                })
                .transpose()?,
            attempts: attempts_i as u32,
            error: error_s,
            progress: progress_i as u8,
            worker_id: worker_id_s,
            next_jobs: serde_json::from_str(&next_jobs_s)
                .map_err(|e| PersistenceError::Database(format!("next_jobs column 21: {e}")))?,
        })
    }
}
