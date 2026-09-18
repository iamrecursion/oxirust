//! Builder pattern for workflow construction.

use crate::error::Result;
use crate::task::{RetryPolicy, Task, TaskId, TaskPriority, TaskType};
use crate::workflow::{Workflow, WorkflowConfig};
use std::collections::HashMap;
use std::time::Duration;

/// Workflow builder for ergonomic workflow construction.
pub struct WorkflowBuilder {
    workflow: Workflow,
    task_map: HashMap<String, TaskId>,
}

impl WorkflowBuilder {
    /// Create a new workflow builder.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            workflow: Workflow::new(name),
            task_map: HashMap::new(),
        }
    }

    /// Set workflow description.
    #[must_use]
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.workflow.description = description.into();
        self
    }

    /// Set workflow configuration.
    #[must_use]
    pub fn config(mut self, config: WorkflowConfig) -> Self {
        self.workflow.config = config;
        self
    }

    /// Set maximum concurrent tasks.
    #[must_use]
    pub fn max_concurrent_tasks(mut self, max: usize) -> Self {
        self.workflow.config.max_concurrent_tasks = max;
        self
    }

    /// Set global timeout.
    #[must_use]
    pub fn global_timeout(mut self, timeout: Duration) -> Self {
        self.workflow.config.global_timeout = Some(timeout);
        self
    }

    /// Enable fail-fast mode.
    #[must_use]
    pub fn fail_fast(mut self, enabled: bool) -> Self {
        self.workflow.config.fail_fast = enabled;
        self
    }

    /// Enable continue-on-error mode.
    #[must_use]
    pub fn continue_on_error(mut self, enabled: bool) -> Self {
        self.workflow.config.continue_on_error = enabled;
        self
    }

    /// Set a workflow variable.
    #[must_use]
    pub fn variable(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.workflow.config.variables.insert(key.into(), value);
        self
    }

    /// Add a task with a named reference.
    #[must_use]
    pub fn task(mut self, name: impl Into<String>, task: Task) -> Self {
        let name_str = name.into();
        let task_id = task.id;
        self.workflow.add_task(task);
        self.task_map.insert(name_str, task_id);
        self
    }

    /// Add a task with `TaskBuilder`.
    #[must_use]
    pub fn add_task(mut self, builder: TaskBuilder) -> Self {
        let (name, task) = builder.build();
        let task_id = task.id;
        self.workflow.add_task(task);
        if let Some(n) = name {
            self.task_map.insert(n, task_id);
        }
        self
    }

    /// Add a dependency between named tasks.
    pub fn depends_on(
        mut self,
        task: impl AsRef<str>,
        dependency: impl AsRef<str>,
    ) -> Result<Self> {
        let task_id = self
            .task_map
            .get(task.as_ref())
            .ok_or_else(|| crate::error::WorkflowError::TaskNotFound(task.as_ref().to_string()))?;

        let dep_id = self.task_map.get(dependency.as_ref()).ok_or_else(|| {
            crate::error::WorkflowError::TaskNotFound(dependency.as_ref().to_string())
        })?;

        self.workflow.add_edge(*dep_id, *task_id)?;
        Ok(self)
    }

    /// Add a conditional dependency.
    pub fn conditional_depends_on(
        mut self,
        task: impl AsRef<str>,
        dependency: impl AsRef<str>,
        condition: impl Into<String>,
    ) -> Result<Self> {
        let task_id = *self
            .task_map
            .get(task.as_ref())
            .ok_or_else(|| crate::error::WorkflowError::TaskNotFound(task.as_ref().to_string()))?;

        let dep_id = *self.task_map.get(dependency.as_ref()).ok_or_else(|| {
            crate::error::WorkflowError::TaskNotFound(dependency.as_ref().to_string())
        })?;

        self.workflow
            .add_conditional_edge(dep_id, task_id, condition.into())?;
        Ok(self)
    }

    /// Build the workflow.
    pub fn build(self) -> Result<Workflow> {
        self.workflow.validate()?;
        Ok(self.workflow)
    }
}

/// Task builder for ergonomic task construction.
pub struct TaskBuilder {
    name: Option<String>,
    task_name: String,
    task_type: TaskType,
    priority: TaskPriority,
    retry: RetryPolicy,
    timeout: Duration,
    metadata: HashMap<String, String>,
    conditions: Vec<String>,
}

impl TaskBuilder {
    /// Create a new task builder.
    #[must_use]
    pub fn new(task_name: impl Into<String>, task_type: TaskType) -> Self {
        Self {
            name: None,
            task_name: task_name.into(),
            task_type,
            priority: TaskPriority::Normal,
            retry: RetryPolicy::default(),
            timeout: Duration::from_secs(3600),
            metadata: HashMap::new(),
            conditions: Vec::new(),
        }
    }

    /// Set a named reference for this task.
    #[must_use]
    pub fn named(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set task priority.
    #[must_use]
    pub fn priority(mut self, priority: TaskPriority) -> Self {
        self.priority = priority;
        self
    }

    /// Set retry policy.
    #[must_use]
    pub fn retry(mut self, retry: RetryPolicy) -> Self {
        self.retry = retry;
        self
    }

    /// Set retry attempts.
    #[must_use]
    pub fn retry_attempts(mut self, attempts: u32) -> Self {
        self.retry.max_attempts = attempts;
        self
    }

    /// Set timeout.
    #[must_use]
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Add metadata.
    #[must_use]
    pub fn metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Add condition.
    #[must_use]
    pub fn condition(mut self, condition: impl Into<String>) -> Self {
        self.conditions.push(condition.into());
        self
    }

    /// Build the task.
    #[must_use]
    pub fn build(self) -> (Option<String>, Task) {
        let mut task = Task::new(self.task_name, self.task_type);
        task.priority = self.priority;
        task.retry = self.retry;
        task.timeout = self.timeout;
        task.metadata = self.metadata;
        task.conditions = self.conditions;

        (self.name, task)
    }
}

/// Transcode task builder.
pub struct TranscodeTaskBuilder {
    inner: TaskBuilder,
}

impl TranscodeTaskBuilder {
    /// Create a new transcode task builder.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        input: impl Into<std::path::PathBuf>,
        output: impl Into<std::path::PathBuf>,
        preset: impl Into<String>,
    ) -> Self {
        let task_type = TaskType::Transcode {
            input: input.into(),
            output: output.into(),
            preset: preset.into(),
            params: HashMap::new(),
        };

        Self {
            inner: TaskBuilder::new(name, task_type),
        }
    }

    /// Set task name reference.
    #[must_use]
    pub fn named(mut self, name: impl Into<String>) -> Self {
        self.inner = self.inner.named(name);
        self
    }

    /// Set priority.
    #[must_use]
    pub fn priority(mut self, priority: TaskPriority) -> Self {
        self.inner = self.inner.priority(priority);
        self
    }

    /// Set timeout.
    #[must_use]
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.inner = self.inner.timeout(timeout);
        self
    }

    /// Add parameter.
    #[must_use]
    pub fn param(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        if let TaskType::Transcode { ref mut params, .. } = self.inner.task_type {
            params.insert(key.into(), value);
        }
        self
    }

    /// Build the task.
    #[must_use]
    pub fn build(self) -> (Option<String>, Task) {
        self.inner.build()
    }
}

/// QC task builder.
pub struct QcTaskBuilder {
    inner: TaskBuilder,
}

impl QcTaskBuilder {
    /// Create a new QC task builder.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        input: impl Into<std::path::PathBuf>,
        profile: impl Into<String>,
    ) -> Self {
        let task_type = TaskType::QualityControl {
            input: input.into(),
            profile: profile.into(),
            rules: Vec::new(),
        };

        Self {
            inner: TaskBuilder::new(name, task_type),
        }
    }

    /// Set task name reference.
    #[must_use]
    pub fn named(mut self, name: impl Into<String>) -> Self {
        self.inner = self.inner.named(name);
        self
    }

    /// Add validation rule.
    #[must_use]
    pub fn rule(mut self, rule: impl Into<String>) -> Self {
        if let TaskType::QualityControl { ref mut rules, .. } = self.inner.task_type {
            rules.push(rule.into());
        }
        self
    }

    /// Build the task.
    #[must_use]
    pub fn build(self) -> (Option<String>, Task) {
        self.inner.build()
    }
}

/// Transfer task builder.
pub struct TransferTaskBuilder {
    inner: TaskBuilder,
}

impl TransferTaskBuilder {
    /// Create a new transfer task builder.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        source: impl Into<String>,
        destination: impl Into<String>,
        protocol: crate::task::TransferProtocol,
    ) -> Self {
        let task_type = TaskType::Transfer {
            source: source.into(),
            destination: destination.into(),
            protocol,
            options: HashMap::new(),
        };

        Self {
            inner: TaskBuilder::new(name, task_type),
        }
    }

    /// Set task name reference.
    #[must_use]
    pub fn named(mut self, name: impl Into<String>) -> Self {
        self.inner = self.inner.named(name);
        self
    }

    /// Add option.
    #[must_use]
    pub fn option(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        if let TaskType::Transfer {
            ref mut options, ..
        } = self.inner.task_type
        {
            options.insert(key.into(), value.into());
        }
        self
    }

    /// Build the task.
    #[must_use]
    pub fn build(self) -> (Option<String>, Task) {
        self.inner.build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workflow_builder() {
        let workflow = WorkflowBuilder::new("test-workflow")
            .description("Test workflow")
            .max_concurrent_tasks(4)
            .fail_fast(true)
            .build()
            .expect("should succeed in test");

        assert_eq!(workflow.name, "test-workflow");
        assert_eq!(workflow.description, "Test workflow");
        assert_eq!(workflow.config.max_concurrent_tasks, 4);
        assert!(workflow.config.fail_fast);
    }

    #[test]
    fn test_workflow_builder_with_tasks() {
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

        let workflow = WorkflowBuilder::new("test")
            .task("t1", task1)
            .task("t2", task2)
            .depends_on("t2", "t1")
            .expect("should succeed in test")
            .build()
            .expect("should succeed in test");

        assert_eq!(workflow.tasks.len(), 2);
        assert_eq!(workflow.edges.len(), 1);
    }

    #[test]
    fn test_task_builder() {
        let (name, task) = TaskBuilder::new(
            "test-task",
            TaskType::Wait {
                duration: Duration::from_secs(5),
            },
        )
        .named("my-task")
        .priority(TaskPriority::High)
        .timeout(Duration::from_secs(10))
        .metadata("key", "value")
        .condition("x > 5")
        .build();

        assert_eq!(name, Some("my-task".to_string()));
        assert_eq!(task.name, "test-task");
        assert_eq!(task.priority, TaskPriority::High);
        assert_eq!(task.timeout, Duration::from_secs(10));
        assert_eq!(task.metadata.get("key"), Some(&"value".to_string()));
        assert_eq!(task.conditions.len(), 1);
    }

    #[test]
    fn test_transcode_task_builder() {
        let (name, task) =
            TranscodeTaskBuilder::new("transcode", "/input.mp4", "/output.mp4", "h264")
                .named("my-transcode")
                .priority(TaskPriority::High)
                .param("bitrate", serde_json::json!(5000000))
                .build();

        assert_eq!(name, Some("my-transcode".to_string()));
        assert_eq!(task.name, "transcode");

        if let TaskType::Transcode { params, .. } = &task.task_type {
            assert_eq!(params.get("bitrate"), Some(&serde_json::json!(5000000)));
        } else {
            panic!("Wrong task type");
        }
    }

    #[test]
    fn test_qc_task_builder() {
        let (_, task) = QcTaskBuilder::new("qc", "/input.mp4", "broadcast")
            .rule("video_bitrate")
            .rule("audio_levels")
            .build();

        if let TaskType::QualityControl { rules, .. } = &task.task_type {
            assert_eq!(rules.len(), 2);
            assert!(rules.contains(&"video_bitrate".to_string()));
        } else {
            panic!("Wrong task type");
        }
    }

    #[test]
    fn test_transfer_task_builder() {
        let (_, task) = TransferTaskBuilder::new(
            "transfer",
            "/local/file.mp4",
            "s3://bucket/file.mp4",
            crate::task::TransferProtocol::S3,
        )
        .option("storage_class", "STANDARD")
        .build();

        if let TaskType::Transfer { options, .. } = &task.task_type {
            assert_eq!(options.get("storage_class"), Some(&"STANDARD".to_string()));
        } else {
            panic!("Wrong task type");
        }
    }

    #[test]
    fn test_workflow_builder_variables() {
        let workflow = WorkflowBuilder::new("test")
            .variable("input_dir", serde_json::json!("/inputs"))
            .variable("output_dir", serde_json::json!("/outputs"))
            .build()
            .expect("should succeed in test");

        assert_eq!(workflow.config.variables.len(), 2);
        assert_eq!(
            workflow.config.variables.get("input_dir"),
            Some(&serde_json::json!("/inputs"))
        );
    }
}
