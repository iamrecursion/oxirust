//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{IoError, Result};
use crate::metadata::{Metadata, MetadataValue};
use chrono::{DateTime, Datelike, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// Schedule configuration for periodic execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduleConfig {
    /// Cron expression for scheduling (e.g., "0 0 * * *")
    pub cron: Option<String>,
    /// Fixed interval between executions
    pub interval: Option<Duration>,
    /// Earliest time to start execution
    pub start_time: Option<DateTime<Utc>>,
    /// Latest time to stop execution
    pub end_time: Option<DateTime<Utc>>,
}
/// Retry policy for failed tasks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryPolicy {
    /// Maximum number of retry attempts
    pub max_retries: usize,
    /// Initial backoff delay in seconds
    pub backoff_seconds: u64,
    /// Whether to use exponential backoff
    pub exponential_backoff: bool,
}
/// Workflow builder
pub struct WorkflowBuilder {
    workflow: Workflow,
}
impl WorkflowBuilder {
    /// Create a new workflow builder
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            workflow: Workflow {
                id: id.into(),
                name: name.into(),
                description: None,
                tasks: Vec::new(),
                dependencies: HashMap::new(),
                config: WorkflowConfig::default(),
                metadata: Metadata::new(),
            },
        }
    }
    /// Set description
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.workflow.description = Some(desc.into());
        self
    }
    /// Add a task
    pub fn add_task(mut self, task: Task) -> Self {
        self.workflow.tasks.push(task);
        self
    }
    /// Add a dependency
    pub fn add_dependency(
        mut self,
        task_id: impl Into<String>,
        depends_on: impl Into<String>,
    ) -> Self {
        let task_id = task_id.into();
        let depends_on = depends_on.into();
        self.workflow
            .dependencies
            .entry(task_id)
            .or_default()
            .push(depends_on);
        self
    }
    /// Configure workflow
    pub fn configure<F>(mut self, f: F) -> Self
    where
        F: FnOnce(&mut WorkflowConfig),
    {
        f(&mut self.workflow.config);
        self
    }
    /// Build the workflow
    pub fn build(self) -> Result<Workflow> {
        self.validate()?;
        Ok(self.workflow)
    }
    fn validate(&self) -> Result<()> {
        if self.has_cycles() {
            return Err(IoError::ValidationError(
                "Workflow contains dependency cycles".to_string(),
            ));
        }
        let mut ids = HashSet::new();
        for task in &self.workflow.tasks {
            if !ids.insert(&task.id) {
                return Err(IoError::ValidationError(format!(
                    "Duplicate task ID: {}",
                    task.id
                )));
            }
        }
        for (task_id, deps) in &self.workflow.dependencies {
            if !ids.contains(&task_id) {
                return Err(IoError::ValidationError(format!(
                    "Unknown task in dependencies: {}",
                    task_id
                )));
            }
            for dep in deps {
                if !ids.contains(&dep) {
                    return Err(IoError::ValidationError(format!(
                        "Unknown dependency: {}",
                        dep
                    )));
                }
            }
        }
        Ok(())
    }
    fn has_cycles(&self) -> bool {
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();
        for task in &self.workflow.tasks {
            if !visited.contains(&task.id)
                && self.has_cycle_dfs(&task.id, &mut visited, &mut rec_stack)
            {
                return true;
            }
        }
        false
    }
    fn has_cycle_dfs(
        &self,
        node: &str,
        visited: &mut HashSet<String>,
        rec_stack: &mut HashSet<String>,
    ) -> bool {
        visited.insert(node.to_string());
        rec_stack.insert(node.to_string());
        if let Some(deps) = self.workflow.dependencies.get(node) {
            for dep in deps {
                if !visited.contains(dep) {
                    if self.has_cycle_dfs(dep, visited, rec_stack) {
                        return true;
                    }
                } else if rec_stack.contains(dep) {
                    return true;
                }
            }
        }
        rec_stack.remove(node);
        false
    }
}
/// Workflow execution state
#[derive(Debug, Clone)]
pub struct WorkflowState {
    /// Workflow identifier
    pub workflowid: String,
    /// Unique execution identifier
    pub executionid: String,
    /// Current workflow status
    pub status: WorkflowStatus,
    /// State of each task in the workflow
    pub task_states: HashMap<String, TaskState>,
    /// When the workflow started
    pub start_time: Option<DateTime<Utc>>,
    /// When the workflow completed
    pub end_time: Option<DateTime<Utc>>,
    /// Error message if workflow failed
    pub error: Option<String>,
}
/// Workflow configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowConfig {
    /// Maximum number of tasks to run in parallel
    pub max_parallel_tasks: usize,
    /// Retry policy for failed tasks
    pub retry_policy: RetryPolicy,
    /// Maximum workflow execution time
    pub timeout: Option<Duration>,
    /// Directory for workflow checkpoints
    pub checkpoint_dir: Option<PathBuf>,
    /// Notification settings
    pub notifications: NotificationConfig,
    /// Scheduling configuration for periodic execution
    pub scheduling: Option<ScheduleConfig>,
}
/// Runtime state of a task execution
#[derive(Debug, Clone)]
pub struct TaskState {
    /// Current execution status
    pub status: TaskStatus,
    /// When the task started executing
    pub start_time: Option<DateTime<Utc>>,
    /// When the task finished executing
    pub end_time: Option<DateTime<Utc>>,
    /// Number of execution attempts
    pub attempts: usize,
    /// Error message if task failed
    pub error: Option<String>,
    /// Task outputs as key-value pairs
    pub outputs: HashMap<String, serde_json::Value>,
}
/// Resource requirements for tasks
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ResourceRequirements {
    /// Number of CPU cores required
    pub cpu_cores: Option<usize>,
    /// Memory requirement in GB
    pub memorygb: Option<f64>,
    /// GPU requirements if needed
    pub gpu: Option<GpuRequirement>,
    /// Disk space requirement in GB
    pub disk_space_gb: Option<f64>,
}
/// Notification delivery channels
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NotificationChannel {
    /// Send email notifications
    Email {
        /// Email addresses to notify
        to: Vec<String>,
    },
    /// Send webhook notifications
    Webhook {
        /// Webhook URL
        url: String,
    },
    /// Write notifications to file
    File {
        /// File path for notifications
        path: PathBuf,
    },
}
/// Workflow definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Workflow {
    /// Unique workflow identifier
    pub id: String,
    /// Human-readable workflow name
    pub name: String,
    /// Optional workflow description
    pub description: Option<String>,
    /// List of tasks in the workflow
    pub tasks: Vec<Task>,
    /// Task dependencies (task_id -> list of prerequisite task_ids)
    pub dependencies: HashMap<String, Vec<String>>,
    /// Workflow configuration
    pub config: WorkflowConfig,
    /// Workflow metadata
    pub metadata: Metadata,
}
/// Notification configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationConfig {
    /// Send notification on successful completion
    pub on_success: bool,
    /// Send notification on failure
    pub on_failure: bool,
    /// Send notification when workflow starts
    pub on_start: bool,
    /// List of notification channels to use
    pub channels: Vec<NotificationChannel>,
}
/// Workflow executor
pub struct WorkflowExecutor {
    config: ExecutorConfig,
    state: Arc<Mutex<HashMap<String, WorkflowState>>>,
}
impl WorkflowExecutor {
    /// Create a new workflow executor
    pub fn new(config: ExecutorConfig) -> Self {
        Self {
            config,
            state: Arc::new(Mutex::new(HashMap::new())),
        }
    }
    /// Execute a workflow
    pub fn execute(&self, workflow: &Workflow) -> Result<String> {
        let executionid = format!("{}-{}", workflow.id, Utc::now().timestamp());
        let mut state = WorkflowState {
            workflowid: workflow.id.clone(),
            executionid: executionid.clone(),
            status: WorkflowStatus::Pending,
            task_states: HashMap::new(),
            start_time: None,
            end_time: None,
            error: None,
        };
        for task in &workflow.tasks {
            state.task_states.insert(
                task.id.clone(),
                TaskState {
                    status: TaskStatus::Pending,
                    start_time: None,
                    end_time: None,
                    attempts: 0,
                    error: None,
                    outputs: HashMap::new(),
                },
            );
        }
        self.state
            .lock()
            .expect("Operation failed")
            .insert(executionid.clone(), state);
        self.executeworkflow_internal(workflow.clone(), executionid.clone())?;
        Ok(executionid)
    }
    /// Internal workflow execution logic
    fn executeworkflow_internal(&self, workflow: Workflow, executionid: String) -> Result<()> {
        {
            let mut states = self.state.lock().expect("Operation failed");
            if let Some(state) = states.get_mut(&executionid) {
                state.status = WorkflowStatus::Running;
                state.start_time = Some(Utc::now());
            }
        }
        let execution_result = self.execute_tasks_in_order(&workflow, &executionid);
        {
            let mut states = self.state.lock().expect("Operation failed");
            if let Some(state) = states.get_mut(&executionid) {
                state.end_time = Some(Utc::now());
                match execution_result {
                    Ok(_) => state.status = WorkflowStatus::Success,
                    Err(ref e) => {
                        state.status = WorkflowStatus::Failed;
                        state.error = Some(e.to_string());
                    }
                }
            }
        }
        execution_result
    }
    /// Execute tasks in dependency order
    fn execute_tasks_in_order(&self, workflow: &Workflow, executionid: &str) -> Result<()> {
        let mut executed_tasks = HashSet::new();
        let mut remaining_tasks: HashSet<String> =
            workflow.tasks.iter().map(|t| t.id.clone()).collect();
        while !remaining_tasks.is_empty() {
            let mut tasks_to_execute = Vec::new();
            for task_id in &remaining_tasks {
                let can_execute = workflow
                    .dependencies
                    .get(task_id as &String)
                    .is_none_or(|deps| deps.iter().all(|dep| executed_tasks.contains(dep)));
                if can_execute {
                    tasks_to_execute.push(task_id.clone());
                }
            }
            if tasks_to_execute.is_empty() {
                return Err(IoError::Other(
                    "Circular dependency or unresolvable dependencies".to_string(),
                ));
            }
            let batch_size = workflow
                .config
                .max_parallel_tasks
                .min(tasks_to_execute.len());
            for batch in tasks_to_execute.chunks(batch_size) {
                for task_id in batch {
                    let task = workflow
                        .tasks
                        .iter()
                        .find(|t| &t.id == task_id)
                        .ok_or_else(|| IoError::Other(format!("Task not found: {task_id}")))?;
                    self.execute_single_task(task, executionid)?;
                    executed_tasks.insert(task_id.clone());
                    remaining_tasks.remove(task_id);
                }
            }
        }
        Ok(())
    }
    /// Execute a single task with retry logic
    fn execute_single_task(&self, task: &Task, executionid: &str) -> Result<()> {
        let mut attempt = 0;
        let max_retries = 3;
        loop {
            attempt += 1;
            {
                let mut states = self.state.lock().expect("Operation failed");
                if let Some(state) = states.get_mut(executionid) {
                    if let Some(task_state) = state.task_states.get_mut(&task.id) {
                        task_state.status = if attempt == 1 {
                            TaskStatus::Running
                        } else {
                            TaskStatus::Retrying
                        };
                        task_state.start_time = Some(Utc::now());
                        task_state.attempts = attempt;
                    }
                }
            }
            let result = self.execute_task_by_type(task);
            {
                let mut states = self.state.lock().expect("Operation failed");
                if let Some(state) = states.get_mut(executionid) {
                    if let Some(task_state) = state.task_states.get_mut(&task.id) {
                        task_state.end_time = Some(Utc::now());
                        match result {
                            Ok(outputs) => {
                                task_state.status = TaskStatus::Success;
                                task_state.outputs = outputs;
                                task_state.error = None;
                                return Ok(());
                            }
                            Err(e) => {
                                if attempt >= max_retries {
                                    task_state.status = TaskStatus::Failed;
                                    task_state.error = Some(e.to_string());
                                    return Err(e);
                                } else {
                                    task_state.error = Some(format!("Attempt {attempt}: {e}"));
                                }
                            }
                        }
                    }
                }
            }
            if attempt < max_retries {
                std::thread::sleep(std::time::Duration::from_secs(1 << (attempt - 1)));
            }
        }
    }
    /// Execute task based on its type
    fn execute_task_by_type(&self, task: &Task) -> Result<HashMap<String, serde_json::Value>> {
        let mut outputs = HashMap::new();
        match task.task_type {
            TaskType::DataIngestion => {
                outputs.insert("status".to_string(), serde_json::json!("completed"));
                outputs.insert("records_processed".to_string(), serde_json::json!(1000));
            }
            TaskType::Transform => {
                outputs.insert("status".to_string(), serde_json::json!("completed"));
                outputs.insert("rows_transformed".to_string(), serde_json::json!(1000));
            }
            TaskType::Validation => {
                outputs.insert("status".to_string(), serde_json::json!("completed"));
                outputs.insert("validation_errors".to_string(), serde_json::json!(0));
            }
            TaskType::MLTraining => {
                outputs.insert("status".to_string(), serde_json::json!("completed"));
                outputs.insert("model_accuracy".to_string(), serde_json::json!(0.95));
            }
            TaskType::MLInference => {
                outputs.insert("status".to_string(), serde_json::json!("completed"));
                outputs.insert("predictions_generated".to_string(), serde_json::json!(500));
            }
            TaskType::Export => {
                outputs.insert("status".to_string(), serde_json::json!("completed"));
                outputs.insert("files_written".to_string(), serde_json::json!(1));
            }
            TaskType::Script => {
                outputs.insert("status".to_string(), serde_json::json!("completed"));
                outputs.insert("exit_code".to_string(), serde_json::json!(0));
            }
            TaskType::SubWorkflow => {
                outputs.insert("status".to_string(), serde_json::json!("completed"));
                outputs.insert(
                    "subworkflowid".to_string(),
                    serde_json::json!(format!("sub-{}", task.id)),
                );
            }
            TaskType::Conditional => {
                let condition_met = true;
                outputs.insert(
                    "condition_met".to_string(),
                    serde_json::json!(condition_met),
                );
                outputs.insert("status".to_string(), serde_json::json!("completed"));
            }
            TaskType::Parallel => {
                outputs.insert("status".to_string(), serde_json::json!("completed"));
                outputs.insert("parallel_tasks_completed".to_string(), serde_json::json!(4));
            }
        }
        outputs.insert("execution_time_ms".to_string(), serde_json::json!(100));
        outputs.insert(
            "execution_timestamp".to_string(),
            serde_json::json!(Utc::now().to_rfc3339()),
        );
        Ok(outputs)
    }
    /// Get workflow state
    pub fn get_state(&self, executionid: &str) -> Option<WorkflowState> {
        self.state
            .lock()
            .expect("Operation failed")
            .get(executionid)
            .cloned()
    }
    /// Cancel a workflow execution
    pub fn cancel(&self, executionid: &str) -> Result<()> {
        let mut states = self.state.lock().expect("Operation failed");
        if let Some(state) = states.get_mut(executionid) {
            state.status = WorkflowStatus::Cancelled;
            state.end_time = Some(Utc::now());
            Ok(())
        } else {
            Err(IoError::Other(format!("Execution {executionid} not found")))
        }
    }
}
/// Configuration for the workflow executor
#[derive(Debug, Clone)]
pub struct ExecutorConfig {
    /// Maximum number of workflows that can run concurrently
    pub max_concurrentworkflows: usize,
    /// Maximum time allowed for a single task execution
    pub task_timeout: Duration,
    /// Interval between workflow state checkpoints
    pub checkpoint_interval: Duration,
}
/// GPU resource requirements for workflow execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuRequirement {
    /// Number of GPUs required
    pub count: usize,
    /// Memory requirement in GB per GPU
    pub memorygb: Option<f64>,
    /// Required CUDA compute capability (e.g., "7.5")
    pub compute_capability: Option<String>,
}
/// Task types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum TaskType {
    /// Data ingestion from files or databases
    DataIngestion,
    /// Data transformation using pipeline
    Transform,
    /// Data validation
    Validation,
    /// Machine learning training
    MLTraining,
    /// Model inference
    MLInference,
    /// Data export
    Export,
    /// Custom script execution
    Script,
    /// Sub-workflow execution
    SubWorkflow,
    /// Conditional execution
    Conditional,
    /// Parallel execution
    Parallel,
}
/// Workflow execution status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkflowStatus {
    /// Workflow is waiting to start
    Pending,
    /// Workflow is currently executing
    Running,
    /// Workflow completed successfully
    Success,
    /// Workflow failed with errors
    Failed,
    /// Workflow was cancelled by user
    Cancelled,
    /// Workflow is temporarily paused
    Paused,
}
/// Task definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    /// Unique task identifier
    pub id: String,
    /// Human-readable task name
    pub name: String,
    /// Type of task to execute
    pub task_type: TaskType,
    /// Task-specific configuration parameters
    pub config: serde_json::Value,
    /// Input data identifiers
    pub inputs: Vec<String>,
    /// Output data identifiers
    pub outputs: Vec<String>,
    /// Resource requirements for this task
    pub resources: ResourceRequirements,
}
/// Task execution status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskStatus {
    /// Task is waiting to execute
    Pending,
    /// Task is currently running
    Running,
    /// Task completed successfully
    Success,
    /// Task failed with errors
    Failed,
    /// Task was skipped due to conditions
    Skipped,
    /// Task is being retried after failure
    Retrying,
}
