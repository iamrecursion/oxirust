//! Task execution engine for automation workflows.
//!
//! Provides stateful task execution with a shared variable context,
//! pending-task queue, and simple synchronous runner.

#![allow(dead_code)]

use std::collections::{HashMap, VecDeque};

/// Status of an automation task.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskStatus {
    /// Waiting to be executed
    Pending,
    /// Currently executing
    Running,
    /// Successfully completed
    Completed,
    /// Execution failed
    Failed(String),
    /// Cancelled before execution
    Cancelled,
    /// Skipped due to a dependency not being met
    Skipped,
}

impl TaskStatus {
    /// Returns `true` if this status represents a terminal (non-runnable) state.
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Failed(_) | Self::Cancelled | Self::Skipped
        )
    }

    /// Returns `true` if execution succeeded.
    #[must_use]
    pub fn is_success(&self) -> bool {
        matches!(self, Self::Completed)
    }
}

/// Shared variable context passed to tasks during execution.
#[derive(Debug, Default, Clone)]
pub struct ExecutionContext {
    vars: HashMap<String, String>,
}

impl ExecutionContext {
    /// Creates a new empty context.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets a variable value.
    pub fn set_var(&mut self, key: &str, value: &str) {
        self.vars.insert(key.to_owned(), value.to_owned());
    }

    /// Gets a variable value, returning `None` if not set.
    #[must_use]
    pub fn get_var(&self, key: &str) -> Option<&str> {
        self.vars.get(key).map(String::as_str)
    }

    /// Returns the number of variables stored.
    #[must_use]
    pub fn var_count(&self) -> usize {
        self.vars.len()
    }
}

/// A single automation task.
#[derive(Debug, Clone)]
pub struct AutomationTask {
    /// Unique task identifier
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Script/command body to execute
    pub body: String,
}

impl AutomationTask {
    /// Creates a new task.
    #[must_use]
    pub fn new(id: &str, name: &str, body: &str) -> Self {
        Self {
            id: id.to_owned(),
            name: name.to_owned(),
            body: body.to_owned(),
        }
    }

    /// Executes the task against the given context and returns its final status.
    ///
    /// Current implementation is a simple stub: tasks whose body starts with
    /// `"fail:"` will return `Failed`, otherwise `Completed`.
    pub fn execute(&self, ctx: &mut ExecutionContext) -> TaskStatus {
        if self.body.starts_with("fail:") {
            TaskStatus::Failed(format!("Task '{}' body indicated failure", self.id))
        } else {
            // Record task execution in context for observability
            ctx.set_var(&format!("task.{}.status", self.id), "completed");
            TaskStatus::Completed
        }
    }
}

/// Executor that maintains a queue of pending tasks and runs them in order.
#[derive(Debug, Default)]
pub struct TaskExecutor {
    queue: VecDeque<AutomationTask>,
    results: Vec<(String, TaskStatus)>,
}

impl TaskExecutor {
    /// Creates a new executor.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Enqueues a task for execution.
    pub fn enqueue(&mut self, task: AutomationTask) {
        self.queue.push_back(task);
    }

    /// Returns the number of tasks waiting to be executed.
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.queue.len()
    }

    /// Runs all pending tasks against the given context in FIFO order.
    /// Returns a list of `(task_id, status)` pairs.
    pub fn run(&mut self, ctx: &mut ExecutionContext) -> Vec<(String, TaskStatus)> {
        let mut run_results = Vec::new();
        while let Some(task) = self.queue.pop_front() {
            let status = task.execute(ctx);
            run_results.push((task.id.clone(), status.clone()));
            self.results.push((task.id, status));
        }
        run_results
    }

    /// Returns all previously recorded results (from all `run` calls).
    #[must_use]
    pub fn all_results(&self) -> &[(String, TaskStatus)] {
        &self.results
    }

    /// Clears stored results.
    pub fn clear_results(&mut self) {
        self.results.clear();
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_status_pending_not_terminal() {
        assert!(!TaskStatus::Pending.is_terminal());
    }

    #[test]
    fn test_task_status_running_not_terminal() {
        assert!(!TaskStatus::Running.is_terminal());
    }

    #[test]
    fn test_task_status_completed_is_terminal() {
        assert!(TaskStatus::Completed.is_terminal());
    }

    #[test]
    fn test_task_status_failed_is_terminal() {
        assert!(TaskStatus::Failed("oops".to_owned()).is_terminal());
    }

    #[test]
    fn test_task_status_cancelled_is_terminal() {
        assert!(TaskStatus::Cancelled.is_terminal());
    }

    #[test]
    fn test_task_status_skipped_is_terminal() {
        assert!(TaskStatus::Skipped.is_terminal());
    }

    #[test]
    fn test_task_status_is_success() {
        assert!(TaskStatus::Completed.is_success());
        assert!(!TaskStatus::Failed("e".to_owned()).is_success());
    }

    #[test]
    fn test_execution_context_set_get() {
        let mut ctx = ExecutionContext::new();
        ctx.set_var("key", "value");
        assert_eq!(ctx.get_var("key"), Some("value"));
    }

    #[test]
    fn test_execution_context_missing_key() {
        let ctx = ExecutionContext::new();
        assert_eq!(ctx.get_var("nope"), None);
    }

    #[test]
    fn test_execution_context_var_count() {
        let mut ctx = ExecutionContext::new();
        ctx.set_var("a", "1");
        ctx.set_var("b", "2");
        assert_eq!(ctx.var_count(), 2);
    }

    #[test]
    fn test_automation_task_execute_success() {
        let task = AutomationTask::new("t1", "My Task", "echo hello");
        let mut ctx = ExecutionContext::new();
        let status = task.execute(&mut ctx);
        assert_eq!(status, TaskStatus::Completed);
        assert_eq!(ctx.get_var("task.t1.status"), Some("completed"));
    }

    #[test]
    fn test_automation_task_execute_fail() {
        let task = AutomationTask::new("t2", "Fail Task", "fail: something broke");
        let mut ctx = ExecutionContext::new();
        let status = task.execute(&mut ctx);
        assert!(matches!(status, TaskStatus::Failed(_)));
    }

    #[test]
    fn test_executor_pending_count() {
        let mut exec = TaskExecutor::new();
        exec.enqueue(AutomationTask::new("t1", "T1", "op"));
        exec.enqueue(AutomationTask::new("t2", "T2", "op"));
        assert_eq!(exec.pending_count(), 2);
    }

    #[test]
    fn test_executor_run_drains_queue() {
        let mut exec = TaskExecutor::new();
        exec.enqueue(AutomationTask::new("t1", "T1", "op1"));
        exec.enqueue(AutomationTask::new("t2", "T2", "op2"));
        let mut ctx = ExecutionContext::new();
        let results = exec.run(&mut ctx);
        assert_eq!(results.len(), 2);
        assert_eq!(exec.pending_count(), 0);
    }

    #[test]
    fn test_executor_all_results_accumulate() {
        let mut exec = TaskExecutor::new();
        exec.enqueue(AutomationTask::new("t1", "T1", "op"));
        let mut ctx = ExecutionContext::new();
        exec.run(&mut ctx);
        exec.enqueue(AutomationTask::new("t2", "T2", "op"));
        exec.run(&mut ctx);
        assert_eq!(exec.all_results().len(), 2);
    }

    #[test]
    fn test_executor_clear_results() {
        let mut exec = TaskExecutor::new();
        exec.enqueue(AutomationTask::new("t1", "T1", "op"));
        let mut ctx = ExecutionContext::new();
        exec.run(&mut ctx);
        exec.clear_results();
        assert!(exec.all_results().is_empty());
    }
}
