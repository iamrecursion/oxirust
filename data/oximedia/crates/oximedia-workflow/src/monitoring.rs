//! Workflow monitoring and metrics.

use crate::task::{TaskId, TaskState};
use crate::workflow::{WorkflowId, WorkflowState};
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Serde helpers for Arc<AtomicU64>
// ---------------------------------------------------------------------------

fn serialize_atomic<S: Serializer>(val: &Arc<AtomicU64>, ser: S) -> Result<S::Ok, S::Error> {
    ser.serialize_u64(val.load(Ordering::Relaxed))
}

fn deserialize_atomic<'de, D: Deserializer<'de>>(de: D) -> Result<Arc<AtomicU64>, D::Error> {
    let n = u64::deserialize(de)?;
    Ok(Arc::new(AtomicU64::new(n)))
}

fn default_atomic() -> Arc<AtomicU64> {
    Arc::new(AtomicU64::new(0))
}

/// Task execution metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskMetrics {
    /// Task identifier.
    pub task_id: TaskId,
    /// Task name.
    pub task_name: String,
    /// Current state.
    pub state: TaskState,
    /// Start time.
    pub start_time: Option<DateTime<Utc>>,
    /// End time.
    pub end_time: Option<DateTime<Utc>>,
    /// Execution duration.
    pub duration: Option<Duration>,
    /// Number of retry attempts.
    pub retry_count: u32,
    /// Error message if failed.
    pub error: Option<String>,
}

impl TaskMetrics {
    /// Create new task metrics.
    #[must_use]
    pub fn new(task_id: TaskId, task_name: String) -> Self {
        Self {
            task_id,
            task_name,
            state: TaskState::Pending,
            start_time: None,
            end_time: None,
            duration: None,
            retry_count: 0,
            error: None,
        }
    }

    /// Mark task as started.
    pub fn mark_started(&mut self) {
        self.state = TaskState::Running;
        self.start_time = Some(Utc::now());
    }

    /// Mark task as completed.
    pub fn mark_completed(&mut self) {
        self.state = TaskState::Completed;
        self.end_time = Some(Utc::now());
        if let Some(start) = self.start_time {
            self.duration = Some(Duration::from_millis(
                u64::try_from((Utc::now() - start).num_milliseconds()).unwrap_or(0),
            ));
        }
    }

    /// Mark task as failed.
    pub fn mark_failed(&mut self, error: String) {
        self.state = TaskState::Failed;
        self.end_time = Some(Utc::now());
        self.error = Some(error);
        if let Some(start) = self.start_time {
            self.duration = Some(Duration::from_millis(
                u64::try_from((Utc::now() - start).num_milliseconds()).unwrap_or(0),
            ));
        }
    }

    /// Increment retry count.
    pub fn increment_retry(&mut self) {
        self.retry_count += 1;
    }
}

/// Workflow execution metrics.
///
/// The `completed_tasks`, `failed_tasks`, and `running_tasks` counters are
/// backed by `Arc<AtomicU64>` so they can be incremented/decremented from
/// any thread without a mutable borrow. The `Arc` also allows sharing a
/// counter across cheap clones (each clone shares the same atomic).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowMetrics {
    /// Workflow identifier.
    pub workflow_id: WorkflowId,
    /// Workflow name.
    pub workflow_name: String,
    /// Current state.
    pub state: WorkflowState,
    /// Start time.
    pub start_time: Option<DateTime<Utc>>,
    /// End time.
    pub end_time: Option<DateTime<Utc>>,
    /// Total duration.
    pub duration: Option<Duration>,
    /// Task metrics.
    pub tasks: HashMap<TaskId, TaskMetrics>,
    /// Total task count.
    pub total_tasks: usize,
    /// Completed tasks count (atomic for lock-free concurrent updates).
    #[serde(
        serialize_with = "serialize_atomic",
        deserialize_with = "deserialize_atomic",
        default = "default_atomic"
    )]
    pub completed_tasks: Arc<AtomicU64>,
    /// Failed tasks count (atomic for lock-free concurrent updates).
    #[serde(
        serialize_with = "serialize_atomic",
        deserialize_with = "deserialize_atomic",
        default = "default_atomic"
    )]
    pub failed_tasks: Arc<AtomicU64>,
    /// Running tasks count (atomic for lock-free concurrent updates).
    #[serde(
        serialize_with = "serialize_atomic",
        deserialize_with = "deserialize_atomic",
        default = "default_atomic"
    )]
    pub running_tasks: Arc<AtomicU64>,
}

impl WorkflowMetrics {
    /// Create new workflow metrics.
    #[must_use]
    pub fn new(workflow_id: WorkflowId, workflow_name: String, total_tasks: usize) -> Self {
        Self {
            workflow_id,
            workflow_name,
            state: WorkflowState::Created,
            start_time: None,
            end_time: None,
            duration: None,
            tasks: HashMap::new(),
            total_tasks,
            completed_tasks: Arc::new(AtomicU64::new(0)),
            failed_tasks: Arc::new(AtomicU64::new(0)),
            running_tasks: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Read the completed tasks count.
    #[must_use]
    pub fn completed_tasks_count(&self) -> u64 {
        self.completed_tasks.load(Ordering::Relaxed)
    }

    /// Read the failed tasks count.
    #[must_use]
    pub fn failed_tasks_count(&self) -> u64 {
        self.failed_tasks.load(Ordering::Relaxed)
    }

    /// Read the running tasks count.
    #[must_use]
    pub fn running_tasks_count(&self) -> u64 {
        self.running_tasks.load(Ordering::Relaxed)
    }

    /// Mark workflow as started.
    pub fn mark_started(&mut self) {
        self.state = WorkflowState::Running;
        self.start_time = Some(Utc::now());
    }

    /// Mark workflow as completed.
    pub fn mark_completed(&mut self) {
        self.state = WorkflowState::Completed;
        self.end_time = Some(Utc::now());
        if let Some(start) = self.start_time {
            self.duration = Some(Duration::from_millis(
                u64::try_from((Utc::now() - start).num_milliseconds()).unwrap_or(0),
            ));
        }
    }

    /// Mark workflow as failed.
    pub fn mark_failed(&mut self) {
        self.state = WorkflowState::Failed;
        self.end_time = Some(Utc::now());
        if let Some(start) = self.start_time {
            self.duration = Some(Duration::from_millis(
                u64::try_from((Utc::now() - start).num_milliseconds()).unwrap_or(0),
            ));
        }
    }

    /// Update task metrics.
    pub fn update_task(&mut self, task_metrics: TaskMetrics) {
        // Update counters based on state change
        if let Some(old_metrics) = self.tasks.get(&task_metrics.task_id) {
            self.update_counters(old_metrics.state, false);
        }
        self.update_counters(task_metrics.state, true);

        self.tasks.insert(task_metrics.task_id, task_metrics);
    }

    fn update_counters(&self, state: TaskState, increment: bool) {
        match state {
            TaskState::Completed => {
                if increment {
                    self.completed_tasks.fetch_add(1, Ordering::Relaxed);
                } else {
                    // Saturating decrement — avoid underflow.
                    let _ = self.completed_tasks.fetch_update(
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                        |v| if v > 0 { Some(v - 1) } else { Some(0) },
                    );
                }
            }
            TaskState::Failed => {
                if increment {
                    self.failed_tasks.fetch_add(1, Ordering::Relaxed);
                } else {
                    let _ =
                        self.failed_tasks
                            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
                                if v > 0 {
                                    Some(v - 1)
                                } else {
                                    Some(0)
                                }
                            });
                }
            }
            TaskState::Running | TaskState::Retrying => {
                if increment {
                    self.running_tasks.fetch_add(1, Ordering::Relaxed);
                } else {
                    let _ = self.running_tasks.fetch_update(
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                        |v| if v > 0 { Some(v - 1) } else { Some(0) },
                    );
                }
            }
            _ => {}
        }
    }

    /// Get progress percentage.
    #[must_use]
    pub fn progress_percentage(&self) -> f64 {
        if self.total_tasks == 0 {
            return 100.0;
        }
        self.completed_tasks.load(Ordering::Relaxed) as f64 / self.total_tasks as f64 * 100.0
    }

    /// Get average task duration.
    #[must_use]
    pub fn average_task_duration(&self) -> Option<Duration> {
        let durations: Vec<_> = self.tasks.values().filter_map(|m| m.duration).collect();

        if durations.is_empty() {
            return None;
        }

        let total_ms: u64 = durations.iter().map(|d| d.as_millis() as u64).sum();
        Some(Duration::from_millis(total_ms / durations.len() as u64))
    }

    /// Get throughput (tasks per second).
    #[must_use]
    pub fn throughput(&self) -> f64 {
        let completed = self.completed_tasks.load(Ordering::Relaxed) as f64;
        if let (Some(start), Some(end)) = (self.start_time, self.end_time) {
            let duration_secs = (end - start).num_seconds().max(1);
            completed / duration_secs as f64
        } else if let Some(start) = self.start_time {
            let duration_secs = (Utc::now() - start).num_seconds().max(1);
            completed / duration_secs as f64
        } else {
            0.0
        }
    }
}

/// Monitoring service for tracking workflow execution.
pub struct MonitoringService {
    /// Active workflow metrics.
    workflows: Arc<DashMap<WorkflowId, WorkflowMetrics>>,
    /// Historical metrics (completed workflows).
    history: Arc<DashMap<WorkflowId, WorkflowMetrics>>,
}

impl MonitoringService {
    /// Create a new monitoring service.
    #[must_use]
    pub fn new() -> Self {
        Self {
            workflows: Arc::new(DashMap::new()),
            history: Arc::new(DashMap::new()),
        }
    }

    /// Start tracking a workflow.
    pub fn start_workflow(
        &self,
        workflow_id: WorkflowId,
        workflow_name: String,
        total_tasks: usize,
    ) {
        let mut metrics = WorkflowMetrics::new(workflow_id, workflow_name, total_tasks);
        metrics.mark_started();
        self.workflows.insert(workflow_id, metrics);
    }

    /// Update task progress.
    pub fn update_task(
        &self,
        workflow_id: WorkflowId,
        task_id: TaskId,
        task_name: String,
        state: TaskState,
        error: Option<String>,
    ) {
        if let Some(mut workflow_metrics) = self.workflows.get_mut(&workflow_id) {
            let mut task_metrics = workflow_metrics
                .tasks
                .get(&task_id)
                .cloned()
                .unwrap_or_else(|| TaskMetrics::new(task_id, task_name.clone()));

            match state {
                TaskState::Running => task_metrics.mark_started(),
                TaskState::Completed => task_metrics.mark_completed(),
                TaskState::Failed => {
                    if let Some(err) = error {
                        task_metrics.mark_failed(err);
                    } else {
                        task_metrics.mark_failed("Unknown error".to_string());
                    }
                }
                TaskState::Retrying => task_metrics.increment_retry(),
                _ => task_metrics.state = state,
            }

            workflow_metrics.update_task(task_metrics);
        }
    }

    /// Complete workflow tracking.
    pub fn complete_workflow(&self, workflow_id: WorkflowId, success: bool) {
        if let Some((_, mut metrics)) = self.workflows.remove(&workflow_id) {
            if success {
                metrics.mark_completed();
            } else {
                metrics.mark_failed();
            }
            self.history.insert(workflow_id, metrics);
        }
    }

    /// Get workflow metrics.
    #[must_use]
    pub fn get_workflow_metrics(&self, workflow_id: &WorkflowId) -> Option<WorkflowMetrics> {
        self.workflows
            .get(workflow_id)
            .map(|m| m.clone())
            .or_else(|| self.history.get(workflow_id).map(|m| m.clone()))
    }

    /// Get all active workflow metrics.
    #[must_use]
    pub fn get_active_workflows(&self) -> Vec<WorkflowMetrics> {
        self.workflows
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Get workflow history.
    #[must_use]
    pub fn get_history(&self, limit: Option<usize>) -> Vec<WorkflowMetrics> {
        let mut history: Vec<_> = self
            .history
            .iter()
            .map(|entry| entry.value().clone())
            .collect();

        // Sort by start time descending
        history.sort_by(|a, b| b.start_time.cmp(&a.start_time));

        if let Some(limit) = limit {
            history.truncate(limit);
        }

        history
    }

    /// Clear old history entries.
    pub fn clear_history(&self, older_than: DateTime<Utc>) {
        self.history.retain(|_, metrics| {
            if let Some(end_time) = metrics.end_time {
                end_time > older_than
            } else {
                true
            }
        });
    }

    /// Get system-wide statistics.
    #[must_use]
    pub fn get_statistics(&self) -> SystemStatistics {
        let active_workflows = self.workflows.len();
        let total_workflows = self.history.len() + active_workflows;

        let completed_workflows = self
            .history
            .iter()
            .filter(|entry| matches!(entry.value().state, WorkflowState::Completed))
            .count();

        let failed_workflows = self
            .history
            .iter()
            .filter(|entry| matches!(entry.value().state, WorkflowState::Failed))
            .count();

        let total_tasks_completed: usize = (self
            .workflows
            .iter()
            .map(|entry| entry.value().completed_tasks.load(Ordering::Relaxed))
            .sum::<u64>()
            + self
                .history
                .iter()
                .map(|entry| entry.value().completed_tasks.load(Ordering::Relaxed))
                .sum::<u64>()) as usize;

        let total_tasks_failed: usize = (self
            .workflows
            .iter()
            .map(|entry| entry.value().failed_tasks.load(Ordering::Relaxed))
            .sum::<u64>()
            + self
                .history
                .iter()
                .map(|entry| entry.value().failed_tasks.load(Ordering::Relaxed))
                .sum::<u64>()) as usize;

        SystemStatistics {
            active_workflows,
            total_workflows,
            completed_workflows,
            failed_workflows,
            total_tasks_completed,
            total_tasks_failed,
        }
    }
}

impl Default for MonitoringService {
    fn default() -> Self {
        Self::new()
    }
}

/// System-wide statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStatistics {
    /// Number of active workflows.
    pub active_workflows: usize,
    /// Total workflows (active + historical).
    pub total_workflows: usize,
    /// Completed workflows.
    pub completed_workflows: usize,
    /// Failed workflows.
    pub failed_workflows: usize,
    /// Total tasks completed.
    pub total_tasks_completed: usize,
    /// Total tasks failed.
    pub total_tasks_failed: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_metrics_creation() {
        let task_id = TaskId::new();
        let metrics = TaskMetrics::new(task_id, "test-task".to_string());
        assert_eq!(metrics.task_id, task_id);
        assert_eq!(metrics.task_name, "test-task");
        assert_eq!(metrics.state, TaskState::Pending);
    }

    #[test]
    fn test_task_metrics_lifecycle() {
        let task_id = TaskId::new();
        let mut metrics = TaskMetrics::new(task_id, "test-task".to_string());

        metrics.mark_started();
        assert_eq!(metrics.state, TaskState::Running);
        assert!(metrics.start_time.is_some());

        metrics.mark_completed();
        assert_eq!(metrics.state, TaskState::Completed);
        assert!(metrics.end_time.is_some());
        assert!(metrics.duration.is_some());
    }

    #[test]
    fn test_task_metrics_failure() {
        let task_id = TaskId::new();
        let mut metrics = TaskMetrics::new(task_id, "test-task".to_string());

        metrics.mark_started();
        metrics.mark_failed("Test error".to_string());

        assert_eq!(metrics.state, TaskState::Failed);
        assert_eq!(metrics.error, Some("Test error".to_string()));
    }

    #[test]
    fn test_workflow_metrics_creation() {
        let workflow_id = WorkflowId::new();
        let metrics = WorkflowMetrics::new(workflow_id, "test-workflow".to_string(), 5);

        assert_eq!(metrics.workflow_id, workflow_id);
        assert_eq!(metrics.total_tasks, 5);
        assert_eq!(metrics.completed_tasks_count(), 0);
    }

    #[test]
    fn test_workflow_metrics_progress() {
        let workflow_id = WorkflowId::new();
        let metrics = WorkflowMetrics::new(workflow_id, "test-workflow".to_string(), 10);

        assert_eq!(metrics.progress_percentage(), 0.0);

        metrics.completed_tasks.store(5, Ordering::Relaxed);
        assert_eq!(metrics.progress_percentage(), 50.0);

        metrics.completed_tasks.store(10, Ordering::Relaxed);
        assert_eq!(metrics.progress_percentage(), 100.0);
    }

    #[test]
    fn test_monitoring_service_creation() {
        let service = MonitoringService::new();
        assert_eq!(service.get_active_workflows().len(), 0);
    }

    #[test]
    fn test_monitoring_service_workflow_tracking() {
        let service = MonitoringService::new();
        let workflow_id = WorkflowId::new();

        service.start_workflow(workflow_id, "test-workflow".to_string(), 3);

        let metrics = service.get_workflow_metrics(&workflow_id);
        assert!(metrics.is_some());
        assert_eq!(
            metrics.expect("should succeed in test").state,
            WorkflowState::Running
        );
    }

    #[test]
    fn test_monitoring_service_task_updates() {
        let service = MonitoringService::new();
        let workflow_id = WorkflowId::new();
        let task_id = TaskId::new();

        service.start_workflow(workflow_id, "test-workflow".to_string(), 1);

        service.update_task(
            workflow_id,
            task_id,
            "task-1".to_string(),
            TaskState::Running,
            None,
        );

        let metrics = service
            .get_workflow_metrics(&workflow_id)
            .expect("should succeed in test");
        assert_eq!(metrics.running_tasks_count(), 1);

        service.update_task(
            workflow_id,
            task_id,
            "task-1".to_string(),
            TaskState::Completed,
            None,
        );

        let metrics = service
            .get_workflow_metrics(&workflow_id)
            .expect("should succeed in test");
        assert_eq!(metrics.completed_tasks_count(), 1);
        assert_eq!(metrics.running_tasks_count(), 0);
    }

    #[test]
    fn test_monitoring_service_completion() {
        let service = MonitoringService::new();
        let workflow_id = WorkflowId::new();

        service.start_workflow(workflow_id, "test-workflow".to_string(), 1);
        assert_eq!(service.get_active_workflows().len(), 1);

        service.complete_workflow(workflow_id, true);
        assert_eq!(service.get_active_workflows().len(), 0);

        let history = service.get_history(None);
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].state, WorkflowState::Completed);
    }

    #[test]
    fn test_statistics() {
        let service = MonitoringService::new();

        let wf1 = WorkflowId::new();
        let wf2 = WorkflowId::new();

        service.start_workflow(wf1, "workflow1".to_string(), 2);
        service.start_workflow(wf2, "workflow2".to_string(), 3);

        service.complete_workflow(wf1, true);
        service.complete_workflow(wf2, false);

        let stats = service.get_statistics();
        assert_eq!(stats.active_workflows, 0);
        assert_eq!(stats.total_workflows, 2);
        assert_eq!(stats.completed_workflows, 1);
        assert_eq!(stats.failed_workflows, 1);
    }
}
