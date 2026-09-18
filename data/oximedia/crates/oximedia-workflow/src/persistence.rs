//! Workflow state persistence using `SQLite` (Pure-Rust via OxiSQL).

use crate::error::{Result, WorkflowError};
use crate::task::{Task, TaskId, TaskState};
use crate::workflow::{Workflow, WorkflowConfig, WorkflowId, WorkflowState};
use chrono::Utc;
use oxisql_core::{ToSqlValue, Value};
use oxisql_sqlite_compat::SqliteConnectionBlocking;
use std::path::Path;
use std::sync::{Arc, Mutex};
use tracing::{debug, info};

// ---------------------------------------------------------------------------
// Lazy deserialization helper
// ---------------------------------------------------------------------------

/// Wraps a raw JSON string and provides lazy on-demand deserialization of
/// [`WorkflowConfig`].  Parsing occurs only on the first call to
/// [`LazyWorkflowConfig::get_cloned`]; subsequent calls return the cached result.
///
/// The cache is stored inside a `Mutex<Option<WorkflowConfig>>` so the
/// struct can be shared safely and `get()` can take `&self`.
#[derive(Debug)]
pub struct LazyWorkflowConfig {
    raw_json: String,
    parsed: Mutex<Option<WorkflowConfig>>,
}

impl LazyWorkflowConfig {
    /// Construct a lazy wrapper around the raw JSON string. No parsing happens
    /// until [`Self::get_cloned`] is first called.
    #[must_use]
    pub fn new(raw_json: String) -> Self {
        Self {
            raw_json,
            parsed: Mutex::new(None),
        }
    }

    /// Return a clone of the parsed [`WorkflowConfig`], parsing lazily on
    /// first access and caching the result for subsequent calls.
    ///
    /// # Errors
    ///
    /// Returns an error if the stored JSON string is invalid or the internal
    /// mutex is poisoned.
    pub fn get_cloned(&self) -> Result<WorkflowConfig> {
        let mut guard = self
            .parsed
            .lock()
            .map_err(|_| WorkflowError::generic("LazyWorkflowConfig mutex poisoned"))?;
        if guard.is_none() {
            let config: WorkflowConfig =
                serde_json::from_str(&self.raw_json).map_err(WorkflowError::Serialization)?;
            *guard = Some(config);
        }
        // SAFETY: we just ensured the Option is Some.
        Ok(guard.as_ref().expect("just initialised above").clone())
    }

    /// Return a reference to the raw JSON string (always available without
    /// any parsing cost).
    #[must_use]
    pub fn raw(&self) -> &str {
        &self.raw_json
    }

    /// Return a clone of the raw JSON string.
    #[must_use]
    pub fn raw_json_owned(&self) -> String {
        self.raw_json.clone()
    }
}

// ---------------------------------------------------------------------------
// Error helpers
// ---------------------------------------------------------------------------

fn map_oxi(e: impl std::fmt::Display) -> WorkflowError {
    WorkflowError::Database(e.to_string())
}

fn col_text(row: &oxisql_core::Row, idx: usize) -> Result<String> {
    match row.get_by_index(idx) {
        Some(Value::Text(s)) => Ok(s.clone()),
        Some(other) => Err(WorkflowError::Database(format!(
            "column {idx}: expected text, got {}",
            other.type_name()
        ))),
        None => Err(WorkflowError::Database(format!(
            "column {idx} missing from result row"
        ))),
    }
}

fn col_opt_text(row: &oxisql_core::Row, idx: usize) -> Result<Option<String>> {
    match row.get_by_index(idx) {
        Some(Value::Null) | None => Ok(None),
        Some(Value::Text(s)) => Ok(Some(s.clone())),
        Some(other) => Err(WorkflowError::Database(format!(
            "column {idx}: expected text or null, got {}",
            other.type_name()
        ))),
    }
}

fn col_i64(row: &oxisql_core::Row, idx: usize) -> Result<i64> {
    match row.get_by_index(idx) {
        Some(Value::I64(n)) => Ok(*n),
        Some(other) => Err(WorkflowError::Database(format!(
            "column {idx}: expected integer, got {}",
            other.type_name()
        ))),
        None => Err(WorkflowError::Database(format!(
            "column {idx} missing from result row"
        ))),
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
        self.conn.execute(sql, params).map_err(map_oxi)
    }

    fn query(&self, sql: &str, params: &[&dyn ToSqlValue]) -> Result<Vec<oxisql_core::Row>> {
        self.conn.query(sql, params).map_err(map_oxi)
    }
}

// ---------------------------------------------------------------------------
// PersistenceManager
// ---------------------------------------------------------------------------

/// Workflow persistence manager.
pub struct PersistenceManager {
    inner: Arc<Mutex<Inner>>,
}

impl PersistenceManager {
    /// Create a new persistence manager.
    pub fn new(db_path: impl AsRef<Path>) -> Result<Self> {
        let path_str = db_path.as_ref().to_string_lossy().into_owned();
        let conn = SqliteConnectionBlocking::open(&path_str).map_err(map_oxi)?;
        let p = Self {
            inner: Arc::new(Mutex::new(Inner { conn })),
        };
        p.initialize_schema()?;
        Ok(p)
    }

    /// Create an in-memory persistence manager.
    pub fn in_memory() -> Result<Self> {
        let conn = SqliteConnectionBlocking::open_memory().map_err(map_oxi)?;
        let p = Self {
            inner: Arc::new(Mutex::new(Inner { conn })),
        };
        p.initialize_schema()?;
        Ok(p)
    }

    fn with_inner<F, T>(&self, f: F) -> Result<T>
    where
        F: FnOnce(&Inner) -> Result<T>,
    {
        let guard = self.inner.lock().map_err(|_| WorkflowError::LockPoisoned)?;
        f(&guard)
    }

    fn initialize_schema(&self) -> Result<()> {
        self.with_inner(|inner| {
            inner
                .conn
                .execute_batch(
                    r"
            CREATE TABLE IF NOT EXISTS workflows (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                state TEXT NOT NULL,
                config TEXT NOT NULL,
                metadata TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                workflow_id TEXT NOT NULL,
                name TEXT NOT NULL,
                task_type TEXT NOT NULL,
                state TEXT NOT NULL,
                priority INTEGER NOT NULL,
                retry_policy TEXT NOT NULL,
                timeout_secs INTEGER NOT NULL,
                dependencies TEXT,
                metadata TEXT,
                retry_count INTEGER NOT NULL DEFAULT 0,
                conditions TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (workflow_id) REFERENCES workflows(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                workflow_id TEXT NOT NULL,
                from_task TEXT NOT NULL,
                to_task TEXT NOT NULL,
                condition TEXT,
                FOREIGN KEY (workflow_id) REFERENCES workflows(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS task_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                workflow_id TEXT NOT NULL,
                status TEXT NOT NULL,
                data TEXT,
                error TEXT,
                duration_ms INTEGER NOT NULL,
                outputs TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (task_id) REFERENCES tasks(id) ON DELETE CASCADE,
                FOREIGN KEY (workflow_id) REFERENCES workflows(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS execution_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                workflow_id TEXT NOT NULL,
                state TEXT NOT NULL,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                duration_ms INTEGER,
                error TEXT,
                FOREIGN KEY (workflow_id) REFERENCES workflows(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_tasks_workflow ON tasks(workflow_id);
            CREATE INDEX IF NOT EXISTS idx_tasks_state ON tasks(state);
            CREATE INDEX IF NOT EXISTS idx_edges_workflow ON edges(workflow_id);
            CREATE INDEX IF NOT EXISTS idx_task_results_workflow ON task_results(workflow_id);
            CREATE INDEX IF NOT EXISTS idx_execution_history_workflow ON execution_history(workflow_id);
            ",
                )
                .map(|_| ())
                .map_err(map_oxi)?;
            Ok(())
        })?;
        info!("Database schema initialized");
        Ok(())
    }

    /// Save a workflow to the database.
    pub fn save_workflow(&self, workflow: &Workflow) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        let config_json = serde_json::to_string(&workflow.config)?;
        let metadata_json = serde_json::to_string(&workflow.metadata)?;
        let state_s = format!("{:?}", workflow.state);
        let id_s = workflow.id.to_string();

        self.with_inner(|inner| {
            inner.exec(
                "INSERT OR REPLACE INTO workflows (id, name, description, state, config, metadata, created_at, updated_at) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
                &[
                    &id_s,
                    &workflow.name.as_str(),
                    &workflow.description.as_str(),
                    &state_s,
                    &config_json,
                    &metadata_json,
                    &now,
                    &now,
                ],
            )?;
            Ok(())
        })?;

        // Save tasks
        for task in workflow.tasks.values() {
            self.save_task(workflow.id, task)?;
        }

        // Delete then re-insert edges
        let id_s = workflow.id.to_string();
        self.with_inner(|inner| {
            inner.exec("DELETE FROM edges WHERE workflow_id = $1", &[&id_s])?;
            for edge in &workflow.edges {
                let from_s = edge.from.to_string();
                let to_s = edge.to.to_string();
                inner.exec(
                    "INSERT INTO edges (workflow_id, from_task, to_task, condition) VALUES ($1, $2, $3, $4)",
                    &[
                        &id_s,
                        &from_s,
                        &to_s,
                        &edge.condition.as_deref(),
                    ],
                )?;
            }
            Ok(())
        })?;

        debug!("Saved workflow: {}", workflow.id);
        Ok(())
    }

    /// Load a workflow from the database.
    pub fn load_workflow(&self, workflow_id: WorkflowId) -> Result<Workflow> {
        let id_s = workflow_id.to_string();

        let rows = self.with_inner(|inner| {
            inner.query(
                "SELECT id, name, description, state, config, metadata FROM workflows WHERE id = $1",
                &[&id_s],
            )
        })?;

        let row = rows
            .into_iter()
            .next()
            .ok_or_else(|| WorkflowError::WorkflowNotFound(id_s.clone()))?;

        let config_json = col_text(&row, 4)?;
        let metadata_json = col_text(&row, 5)?;
        let state_str = col_text(&row, 3)?;

        let lazy_config = LazyWorkflowConfig::new(config_json);
        let config = lazy_config.get_cloned()?;
        let metadata = serde_json::from_str(&metadata_json)?;
        let state = self.parse_workflow_state(&state_str);

        let mut result = Workflow {
            id: workflow_id,
            name: col_text(&row, 1)?,
            description: col_opt_text(&row, 2)?.unwrap_or_default(),
            tasks: Default::default(),
            edges: Vec::new(),
            config,
            state,
            metadata,
        };

        // Load tasks
        let tasks = self.load_tasks(workflow_id)?;
        for task in tasks {
            result.tasks.insert(task.id, task);
        }

        // Load edges
        result.edges = self.load_edges(workflow_id)?;

        debug!("Loaded workflow: {}", workflow_id);
        Ok(result)
    }

    fn save_task(&self, workflow_id: WorkflowId, task: &Task) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        let task_type_json = serde_json::to_string(&task.task_type)?;
        let retry_policy_json = serde_json::to_string(&task.retry)?;
        let dependencies_json = serde_json::to_string(&task.dependencies)?;
        let metadata_json = serde_json::to_string(&task.metadata)?;
        let conditions_json = serde_json::to_string(&task.conditions)?;
        let state_s = format!("{:?}", task.state);
        let priority_i = task.priority as i64;
        let timeout_i = task.timeout.as_secs() as i64;
        let retry_count_i = task.retry_count as i64;
        let task_id_s = task.id.to_string();
        let workflow_id_s = workflow_id.to_string();

        self.with_inner(|inner| {
            inner.exec(
                "INSERT OR REPLACE INTO tasks
                 (id, workflow_id, name, task_type, state, priority, retry_policy, timeout_secs,
                  dependencies, metadata, retry_count, conditions, created_at, updated_at)
                 VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)",
                &[
                    &task_id_s,
                    &workflow_id_s,
                    &task.name.as_str(),
                    &task_type_json,
                    &state_s,
                    &priority_i,
                    &retry_policy_json,
                    &timeout_i,
                    &dependencies_json,
                    &metadata_json,
                    &retry_count_i,
                    &conditions_json,
                    &now,
                    &now,
                ],
            )?;
            Ok(())
        })
    }

    fn load_tasks(&self, workflow_id: WorkflowId) -> Result<Vec<Task>> {
        let id_s = workflow_id.to_string();
        let rows = self.with_inner(|inner| {
            inner.query(
                "SELECT id, name, task_type, state, priority, retry_policy, timeout_secs,
                         dependencies, metadata, retry_count, conditions
                  FROM tasks WHERE workflow_id = $1",
                &[&id_s],
            )
        })?;

        rows.iter().map(|row| self.task_from_row(row)).collect()
    }

    fn task_from_row(&self, row: &oxisql_core::Row) -> Result<Task> {
        let id_str = col_text(row, 0)?;
        let id = uuid::Uuid::parse_str(&id_str)
            .map_err(|e| WorkflowError::Database(format!("invalid task UUID: {e}")))?;

        let task_type_json = col_text(row, 2)?;
        let task_type: crate::task::TaskType = serde_json::from_str(&task_type_json)
            .map_err(|e| WorkflowError::Database(format!("task_type deserialize: {e}")))?;

        let state_str = col_text(row, 3)?;
        let state = self.parse_task_state(&state_str);

        let priority_int = col_i64(row, 4)?;
        let priority = self.parse_task_priority(priority_int as i32);

        let retry_policy_json = col_text(row, 5)?;
        let retry = serde_json::from_str(&retry_policy_json)
            .map_err(|e| WorkflowError::Database(format!("retry_policy deserialize: {e}")))?;

        let timeout_secs = col_i64(row, 6)?;
        let timeout = std::time::Duration::from_secs(u64::try_from(timeout_secs).unwrap_or(3600));

        let dependencies_json = col_text(row, 7)?;
        let dependencies = serde_json::from_str(&dependencies_json)
            .map_err(|e| WorkflowError::Database(format!("dependencies deserialize: {e}")))?;

        let metadata_json = col_text(row, 8)?;
        let metadata = serde_json::from_str(&metadata_json)
            .map_err(|e| WorkflowError::Database(format!("metadata deserialize: {e}")))?;

        let retry_count = col_i64(row, 9)? as u32;

        let conditions_json = col_text(row, 10)?;
        let conditions = serde_json::from_str(&conditions_json)
            .map_err(|e| WorkflowError::Database(format!("conditions deserialize: {e}")))?;

        Ok(Task {
            id: TaskId::from(id),
            name: col_text(row, 1)?,
            task_type,
            state,
            priority,
            retry,
            timeout,
            dependencies,
            metadata,
            retry_count,
            conditions,
        })
    }

    fn load_edges(&self, workflow_id: WorkflowId) -> Result<Vec<crate::workflow::Edge>> {
        let id_s = workflow_id.to_string();
        let rows = self.with_inner(|inner| {
            inner.query(
                "SELECT from_task, to_task, condition FROM edges WHERE workflow_id = $1",
                &[&id_s],
            )
        })?;

        rows.iter()
            .map(|row| {
                let from_str = col_text(row, 0)?;
                let to_str = col_text(row, 1)?;
                let condition = col_opt_text(row, 2)?;

                let from_uuid = uuid::Uuid::parse_str(&from_str)
                    .map_err(|e| WorkflowError::Database(format!("edge from_uuid: {e}")))?;
                let to_uuid = uuid::Uuid::parse_str(&to_str)
                    .map_err(|e| WorkflowError::Database(format!("edge to_uuid: {e}")))?;

                Ok(crate::workflow::Edge {
                    from: TaskId::from(from_uuid),
                    to: TaskId::from(to_uuid),
                    condition,
                })
            })
            .collect()
    }

    /// List all workflows.
    pub fn list_workflows(&self) -> Result<Vec<WorkflowId>> {
        let rows = self.with_inner(|inner| inner.query("SELECT id FROM workflows", &[]))?;

        rows.iter()
            .map(|row| {
                let id_str = col_text(row, 0)?;
                let uuid = uuid::Uuid::parse_str(&id_str)
                    .map_err(|e| WorkflowError::Database(format!("workflow UUID: {e}")))?;
                Ok(WorkflowId::from(uuid))
            })
            .collect()
    }

    /// Delete a workflow.
    pub fn delete_workflow(&self, workflow_id: WorkflowId) -> Result<()> {
        let id_s = workflow_id.to_string();
        self.with_inner(|inner| {
            inner.exec("DELETE FROM workflows WHERE id = $1", &[&id_s])?;
            Ok(())
        })?;
        debug!("Deleted workflow: {}", workflow_id);
        Ok(())
    }

    fn parse_workflow_state(&self, state_str: &str) -> WorkflowState {
        match state_str {
            "Created" => WorkflowState::Created,
            "Scheduled" => WorkflowState::Scheduled,
            "Running" => WorkflowState::Running,
            "Paused" => WorkflowState::Paused,
            "Completed" => WorkflowState::Completed,
            "Failed" => WorkflowState::Failed,
            "Cancelled" => WorkflowState::Cancelled,
            _ => WorkflowState::Created,
        }
    }

    fn parse_task_state(&self, state_str: &str) -> TaskState {
        match state_str {
            "Pending" => TaskState::Pending,
            "Queued" => TaskState::Queued,
            "Running" => TaskState::Running,
            "Completed" => TaskState::Completed,
            "Failed" => TaskState::Failed,
            "Cancelled" => TaskState::Cancelled,
            "Waiting" => TaskState::Waiting,
            "Retrying" => TaskState::Retrying,
            "Skipped" => TaskState::Skipped,
            _ => TaskState::Pending,
        }
    }

    fn parse_task_priority(&self, priority: i32) -> crate::task::TaskPriority {
        match priority {
            0 => crate::task::TaskPriority::Low,
            1 => crate::task::TaskPriority::Normal,
            2 => crate::task::TaskPriority::High,
            3 => crate::task::TaskPriority::Critical,
            _ => crate::task::TaskPriority::Normal,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::task::{Task, TaskType};
    use std::time::Duration;

    #[test]
    fn test_persistence_creation() {
        let persistence = PersistenceManager::in_memory();
        assert!(persistence.is_ok());
    }

    #[test]
    fn test_save_and_load_workflow() {
        let persistence = PersistenceManager::in_memory().expect("should succeed in test");
        let mut workflow = Workflow::new("test-workflow");

        let task = Task::new(
            "test-task",
            TaskType::Wait {
                duration: Duration::from_secs(10),
            },
        );
        workflow.add_task(task);

        persistence
            .save_workflow(&workflow)
            .expect("should succeed in test");
        let loaded = persistence
            .load_workflow(workflow.id)
            .expect("should succeed in test");

        assert_eq!(loaded.id, workflow.id);
        assert_eq!(loaded.name, workflow.name);
        assert_eq!(loaded.tasks.len(), 1);
    }

    #[test]
    fn test_list_workflows() {
        let persistence = PersistenceManager::in_memory().expect("should succeed in test");

        let workflow1 = Workflow::new("workflow1");
        let workflow2 = Workflow::new("workflow2");

        persistence
            .save_workflow(&workflow1)
            .expect("should succeed in test");
        persistence
            .save_workflow(&workflow2)
            .expect("should succeed in test");

        let workflows = persistence
            .list_workflows()
            .expect("should succeed in test");
        assert_eq!(workflows.len(), 2);
    }

    #[test]
    fn test_delete_workflow() {
        let persistence = PersistenceManager::in_memory().expect("should succeed in test");
        let workflow = Workflow::new("test-workflow");

        persistence
            .save_workflow(&workflow)
            .expect("should succeed in test");
        assert!(persistence.load_workflow(workflow.id).is_ok());

        persistence
            .delete_workflow(workflow.id)
            .expect("should succeed in test");
        assert!(persistence.load_workflow(workflow.id).is_err());
    }

    #[test]
    fn test_save_workflow_with_edges() {
        let persistence = PersistenceManager::in_memory().expect("should succeed in test");
        let mut workflow = Workflow::new("test-workflow");

        let task1 = Task::new(
            "task1",
            TaskType::Wait {
                duration: Duration::from_secs(1),
            },
        );
        let task2 = Task::new(
            "task2",
            TaskType::Wait {
                duration: Duration::from_secs(1),
            },
        );

        let id1 = workflow.add_task(task1);
        let id2 = workflow.add_task(task2);
        workflow.add_edge(id1, id2).expect("should succeed in test");

        persistence
            .save_workflow(&workflow)
            .expect("should succeed in test");
        let loaded = persistence
            .load_workflow(workflow.id)
            .expect("should succeed in test");

        assert_eq!(loaded.edges.len(), 1);
        assert_eq!(loaded.edges[0].from, id1);
        assert_eq!(loaded.edges[0].to, id2);
    }
}
