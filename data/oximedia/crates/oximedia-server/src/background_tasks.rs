//! Background tasks module with persistent task queue for long-running operations.
//!
//! Provides a priority-based task queue, lifecycle management, progress tracking,
//! and retry logic for operations like transcoding, thumbnail generation,
//! and bulk media processing.

#![allow(dead_code)]

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Task priority levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TaskPriority {
    /// Critical tasks (system maintenance, urgent processing).
    Critical,
    /// High priority (user-initiated transcoding).
    High,
    /// Normal priority (standard operations).
    Normal,
    /// Low priority (background cleanup, optimization).
    Low,
}

impl TaskPriority {
    /// Numeric weight (higher = more important).
    pub fn weight(self) -> u8 {
        match self {
            Self::Critical => 4,
            Self::High => 3,
            Self::Normal => 2,
            Self::Low => 1,
        }
    }

    /// Human-readable label.
    pub fn label(self) -> &'static str {
        match self {
            Self::Critical => "critical",
            Self::High => "high",
            Self::Normal => "normal",
            Self::Low => "low",
        }
    }
}

impl Ord for TaskPriority {
    fn cmp(&self, other: &Self) -> Ordering {
        self.weight().cmp(&other.weight())
    }
}

impl PartialOrd for TaskPriority {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Task status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskStatus {
    /// Waiting in the queue.
    Queued,
    /// Currently being executed.
    Running,
    /// Completed successfully.
    Completed,
    /// Failed after all retries.
    Failed,
    /// Cancelled by user.
    Cancelled,
    /// Waiting for retry.
    RetryPending,
}

impl TaskStatus {
    /// Label.
    pub fn label(self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
            Self::RetryPending => "retry_pending",
        }
    }

    /// Whether the task is in a terminal state.
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }
}

/// Task type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskType {
    /// Transcode a media file.
    Transcode {
        media_id: String,
        target_codec: String,
    },
    /// Generate thumbnails.
    GenerateThumbnails { media_id: String, count: u32 },
    /// Analyze media metadata.
    AnalyzeMedia { media_id: String },
    /// Clean up expired uploads.
    CleanupUploads,
    /// Purge old cache entries.
    PurgeCache,
    /// Send a webhook notification.
    SendWebhook { url: String, payload: String },
    /// Custom task.
    Custom {
        name: String,
        params: HashMap<String, String>,
    },
}

impl TaskType {
    /// Label for the task type.
    pub fn label(&self) -> &str {
        match self {
            Self::Transcode { .. } => "transcode",
            Self::GenerateThumbnails { .. } => "generate_thumbnails",
            Self::AnalyzeMedia { .. } => "analyze_media",
            Self::CleanupUploads => "cleanup_uploads",
            Self::PurgeCache => "purge_cache",
            Self::SendWebhook { .. } => "send_webhook",
            Self::Custom { name, .. } => name.as_str(),
        }
    }
}

/// A background task.
#[derive(Debug, Clone)]
pub struct BackgroundTask {
    /// Unique task ID.
    pub id: String,
    /// Task type.
    pub task_type: TaskType,
    /// Priority.
    pub priority: TaskPriority,
    /// Current status.
    pub status: TaskStatus,
    /// Progress (0.0 to 1.0).
    pub progress: f64,
    /// When the task was created.
    pub created_at: u64,
    /// When the task started executing.
    pub started_at: Option<Instant>,
    /// When the task finished.
    pub finished_at: Option<Instant>,
    /// Number of attempts.
    pub attempts: u32,
    /// Maximum retry attempts.
    pub max_retries: u32,
    /// Error message.
    pub error: Option<String>,
    /// Output/result data.
    pub result: HashMap<String, String>,
    /// Owner user ID.
    pub owner: Option<String>,
}

impl BackgroundTask {
    /// Creates a new task.
    pub fn new(id: impl Into<String>, task_type: TaskType, priority: TaskPriority) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Self {
            id: id.into(),
            task_type,
            priority,
            status: TaskStatus::Queued,
            progress: 0.0,
            created_at: now,
            started_at: None,
            finished_at: None,
            attempts: 0,
            max_retries: 3,
            error: None,
            result: HashMap::new(),
            owner: None,
        }
    }

    /// Sets the owner.
    pub fn with_owner(mut self, owner: impl Into<String>) -> Self {
        self.owner = Some(owner.into());
        self
    }

    /// Sets max retries.
    pub fn with_max_retries(mut self, retries: u32) -> Self {
        self.max_retries = retries;
        self
    }

    /// Duration since creation.
    pub fn age(&self) -> Duration {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Duration::from_secs(now.saturating_sub(self.created_at))
    }

    /// Execution duration.
    pub fn execution_duration(&self) -> Duration {
        match (self.started_at, self.finished_at) {
            (Some(start), Some(end)) => end.duration_since(start),
            (Some(start), None) => start.elapsed(),
            _ => Duration::ZERO,
        }
    }

    /// Whether more retries are available.
    pub fn can_retry(&self) -> bool {
        self.attempts < self.max_retries
    }
}

/// Wrapper for BinaryHeap ordering.
#[derive(Debug)]
struct PrioritizedTask {
    task_id: String,
    priority: TaskPriority,
    created_at: u64,
}

impl PartialEq for PrioritizedTask {
    fn eq(&self, other: &Self) -> bool {
        self.task_id == other.task_id
    }
}

impl Eq for PrioritizedTask {}

impl Ord for PrioritizedTask {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority
            .cmp(&other.priority)
            .then_with(|| other.created_at.cmp(&self.created_at)) // older first at same priority
    }
}

impl PartialOrd for PrioritizedTask {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Statistics for the task queue.
#[derive(Debug, Clone, Default)]
pub struct TaskQueueStats {
    /// Total tasks submitted.
    pub total_submitted: u64,
    /// Currently queued.
    pub queued: usize,
    /// Currently running.
    pub running: usize,
    /// Completed.
    pub completed: u64,
    /// Failed.
    pub failed: u64,
    /// Cancelled.
    pub cancelled: u64,
    /// Total retries.
    pub retries: u64,
}

impl TaskQueueStats {
    /// Completion rate (completed / total non-cancelled).
    pub fn completion_rate(&self) -> f64 {
        let attempted = self.completed + self.failed;
        if attempted == 0 {
            return 1.0;
        }
        self.completed as f64 / attempted as f64
    }
}

/// The background task queue.
pub struct TaskQueue {
    /// All tasks by ID.
    tasks: HashMap<String, BackgroundTask>,
    /// Priority queue for pending tasks.
    queue: BinaryHeap<PrioritizedTask>,
    /// Maximum concurrent running tasks.
    max_concurrent: usize,
    /// Statistics.
    stats: TaskQueueStats,
    /// Next task ID counter.
    next_id: u64,
}

impl TaskQueue {
    /// Creates a new task queue.
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            tasks: HashMap::new(),
            queue: BinaryHeap::new(),
            max_concurrent,
            stats: TaskQueueStats::default(),
            next_id: 1,
        }
    }

    /// Generates a unique task ID.
    pub fn generate_id(&mut self) -> String {
        let id = format!("task-{}", self.next_id);
        self.next_id += 1;
        id
    }

    /// Submits a task to the queue.
    pub fn submit(&mut self, task: BackgroundTask) -> String {
        let id = task.id.clone();
        self.queue.push(PrioritizedTask {
            task_id: id.clone(),
            priority: task.priority,
            created_at: task.created_at,
        });
        self.tasks.insert(id.clone(), task);
        self.stats.total_submitted += 1;
        self.update_stats();
        id
    }

    /// Dequeues the next task to execute (if concurrency allows).
    pub fn dequeue(&mut self) -> Option<String> {
        let running = self
            .tasks
            .values()
            .filter(|t| t.status == TaskStatus::Running)
            .count();

        if running >= self.max_concurrent {
            return None;
        }

        while let Some(pt) = self.queue.pop() {
            if let Some(task) = self.tasks.get_mut(&pt.task_id) {
                if task.status == TaskStatus::Queued || task.status == TaskStatus::RetryPending {
                    task.status = TaskStatus::Running;
                    task.started_at = Some(Instant::now());
                    task.attempts += 1;
                    self.update_stats();
                    return Some(pt.task_id);
                }
            }
        }

        None
    }

    /// Gets a task by ID.
    pub fn get_task(&self, id: &str) -> Option<&BackgroundTask> {
        self.tasks.get(id)
    }

    /// Updates task progress.
    pub fn update_progress(&mut self, id: &str, progress: f64) -> bool {
        if let Some(task) = self.tasks.get_mut(id) {
            task.progress = progress.clamp(0.0, 1.0);
            true
        } else {
            false
        }
    }

    /// Completes a task.
    pub fn complete_task(&mut self, id: &str, result: HashMap<String, String>) -> bool {
        if let Some(task) = self.tasks.get_mut(id) {
            task.status = TaskStatus::Completed;
            task.progress = 1.0;
            task.finished_at = Some(Instant::now());
            task.result = result;
            self.stats.completed += 1;
            self.update_stats();
            true
        } else {
            false
        }
    }

    /// Fails a task (may retry).
    pub fn fail_task(&mut self, id: &str, error: &str) -> bool {
        if let Some(task) = self.tasks.get_mut(id) {
            task.error = Some(error.to_string());
            if task.can_retry() {
                task.status = TaskStatus::RetryPending;
                self.stats.retries += 1;
                // Re-enqueue
                self.queue.push(PrioritizedTask {
                    task_id: id.to_string(),
                    priority: task.priority,
                    created_at: task.created_at,
                });
            } else {
                task.status = TaskStatus::Failed;
                task.finished_at = Some(Instant::now());
                self.stats.failed += 1;
            }
            self.update_stats();
            true
        } else {
            false
        }
    }

    /// Cancels a task.
    pub fn cancel_task(&mut self, id: &str) -> bool {
        if let Some(task) = self.tasks.get_mut(id) {
            if !task.status.is_terminal() {
                task.status = TaskStatus::Cancelled;
                task.finished_at = Some(Instant::now());
                self.stats.cancelled += 1;
                self.update_stats();
                return true;
            }
        }
        false
    }

    /// Returns tasks by status.
    pub fn tasks_by_status(&self, status: TaskStatus) -> Vec<&BackgroundTask> {
        self.tasks.values().filter(|t| t.status == status).collect()
    }

    /// Returns tasks owned by a user.
    pub fn tasks_by_owner(&self, owner: &str) -> Vec<&BackgroundTask> {
        self.tasks
            .values()
            .filter(|t| t.owner.as_deref() == Some(owner))
            .collect()
    }

    /// Returns statistics.
    pub fn stats(&self) -> &TaskQueueStats {
        &self.stats
    }

    /// Returns the total number of tasks.
    pub fn total_tasks(&self) -> usize {
        self.tasks.len()
    }

    /// Removes completed/failed/cancelled tasks older than `max_age`.
    pub fn cleanup_old_tasks(&mut self, max_age: Duration) -> usize {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let max_age_secs = max_age.as_secs();

        let old_ids: Vec<String> = self
            .tasks
            .iter()
            .filter(|(_, t)| t.status.is_terminal() && (now - t.created_at) > max_age_secs)
            .map(|(id, _)| id.clone())
            .collect();

        let count = old_ids.len();
        for id in old_ids {
            self.tasks.remove(&id);
        }
        self.update_stats();
        count
    }

    fn update_stats(&mut self) {
        self.stats.queued = self
            .tasks
            .values()
            .filter(|t| t.status == TaskStatus::Queued || t.status == TaskStatus::RetryPending)
            .count();
        self.stats.running = self
            .tasks
            .values()
            .filter(|t| t.status == TaskStatus::Running)
            .count();
    }
}

impl Default for TaskQueue {
    fn default() -> Self {
        Self::new(4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_task(id: &str, task_type: TaskType, priority: TaskPriority) -> BackgroundTask {
        BackgroundTask::new(id, task_type, priority)
    }

    // TaskPriority

    #[test]
    fn test_priority_ordering() {
        assert!(TaskPriority::Critical > TaskPriority::High);
        assert!(TaskPriority::High > TaskPriority::Normal);
        assert!(TaskPriority::Normal > TaskPriority::Low);
    }

    #[test]
    fn test_priority_labels() {
        assert_eq!(TaskPriority::Critical.label(), "critical");
        assert_eq!(TaskPriority::Normal.label(), "normal");
    }

    // TaskStatus

    #[test]
    fn test_status_terminal() {
        assert!(!TaskStatus::Queued.is_terminal());
        assert!(!TaskStatus::Running.is_terminal());
        assert!(TaskStatus::Completed.is_terminal());
        assert!(TaskStatus::Failed.is_terminal());
        assert!(TaskStatus::Cancelled.is_terminal());
    }

    // TaskType

    #[test]
    fn test_task_type_labels() {
        assert_eq!(TaskType::CleanupUploads.label(), "cleanup_uploads");
        assert_eq!(
            TaskType::Transcode {
                media_id: "m1".into(),
                target_codec: "av1".into()
            }
            .label(),
            "transcode"
        );
    }

    // BackgroundTask

    #[test]
    fn test_task_creation() {
        let task = make_task("t1", TaskType::CleanupUploads, TaskPriority::Low);
        assert_eq!(task.id, "t1");
        assert_eq!(task.status, TaskStatus::Queued);
        assert!((task.progress - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_task_with_owner() {
        let task =
            make_task("t1", TaskType::CleanupUploads, TaskPriority::Low).with_owner("user-1");
        assert_eq!(task.owner, Some("user-1".to_string()));
    }

    #[test]
    fn test_task_can_retry() {
        let task = make_task("t1", TaskType::CleanupUploads, TaskPriority::Low);
        assert!(task.can_retry()); // max_retries=3, attempts=0
    }

    // TaskQueue

    #[test]
    fn test_submit_and_dequeue() {
        let mut q = TaskQueue::new(4);
        let task = make_task("t1", TaskType::CleanupUploads, TaskPriority::Normal);
        q.submit(task);
        assert_eq!(q.stats().queued, 1);

        let id = q.dequeue();
        assert_eq!(id, Some("t1".to_string()));
        assert_eq!(q.stats().running, 1);
    }

    #[test]
    fn test_priority_ordering_in_queue() {
        let mut q = TaskQueue::new(4);
        q.submit(make_task("low", TaskType::PurgeCache, TaskPriority::Low));
        q.submit(make_task(
            "high",
            TaskType::CleanupUploads,
            TaskPriority::High,
        ));
        q.submit(make_task(
            "crit",
            TaskType::AnalyzeMedia {
                media_id: "m1".into(),
            },
            TaskPriority::Critical,
        ));

        assert_eq!(q.dequeue(), Some("crit".to_string()));
        assert_eq!(q.dequeue(), Some("high".to_string()));
        assert_eq!(q.dequeue(), Some("low".to_string()));
    }

    #[test]
    fn test_max_concurrent_limit() {
        let mut q = TaskQueue::new(1);
        q.submit(make_task(
            "t1",
            TaskType::CleanupUploads,
            TaskPriority::Normal,
        ));
        q.submit(make_task("t2", TaskType::PurgeCache, TaskPriority::Normal));

        assert!(q.dequeue().is_some()); // t1 starts
        assert!(q.dequeue().is_none()); // t2 blocked (max_concurrent=1)
    }

    #[test]
    fn test_complete_task() {
        let mut q = TaskQueue::new(4);
        q.submit(make_task(
            "t1",
            TaskType::CleanupUploads,
            TaskPriority::Normal,
        ));
        q.dequeue();
        assert!(q.complete_task("t1", HashMap::new()));
        let task = q.get_task("t1").expect("should exist");
        assert_eq!(task.status, TaskStatus::Completed);
        assert!((task.progress - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_fail_task_with_retry() {
        let mut q = TaskQueue::new(4);
        q.submit(make_task(
            "t1",
            TaskType::CleanupUploads,
            TaskPriority::Normal,
        ));
        q.dequeue();
        assert!(q.fail_task("t1", "temporary error"));
        let task = q.get_task("t1").expect("should exist");
        assert_eq!(task.status, TaskStatus::RetryPending);
        assert_eq!(q.stats().retries, 1);

        // Should be able to dequeue again
        let id = q.dequeue();
        assert_eq!(id, Some("t1".to_string()));
    }

    #[test]
    fn test_fail_task_no_more_retries() {
        let mut q = TaskQueue::new(4);
        let task =
            make_task("t1", TaskType::CleanupUploads, TaskPriority::Normal).with_max_retries(1);
        q.submit(task);
        q.dequeue(); // attempt 1
        q.fail_task("t1", "err");
        q.dequeue(); // attempt 2 (retry)
        q.fail_task("t1", "err again");
        let task = q.get_task("t1").expect("should exist");
        assert_eq!(task.status, TaskStatus::Failed);
    }

    #[test]
    fn test_cancel_task() {
        let mut q = TaskQueue::new(4);
        q.submit(make_task(
            "t1",
            TaskType::CleanupUploads,
            TaskPriority::Normal,
        ));
        assert!(q.cancel_task("t1"));
        let task = q.get_task("t1").expect("should exist");
        assert_eq!(task.status, TaskStatus::Cancelled);
    }

    #[test]
    fn test_cancel_completed_task_fails() {
        let mut q = TaskQueue::new(4);
        q.submit(make_task(
            "t1",
            TaskType::CleanupUploads,
            TaskPriority::Normal,
        ));
        q.dequeue();
        q.complete_task("t1", HashMap::new());
        assert!(!q.cancel_task("t1"));
    }

    #[test]
    fn test_update_progress() {
        let mut q = TaskQueue::new(4);
        q.submit(make_task(
            "t1",
            TaskType::CleanupUploads,
            TaskPriority::Normal,
        ));
        assert!(q.update_progress("t1", 0.5));
        let task = q.get_task("t1").expect("should exist");
        assert!((task.progress - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_update_progress_clamped() {
        let mut q = TaskQueue::new(4);
        q.submit(make_task(
            "t1",
            TaskType::CleanupUploads,
            TaskPriority::Normal,
        ));
        q.update_progress("t1", 1.5);
        let task = q.get_task("t1").expect("should exist");
        assert!((task.progress - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_tasks_by_status() {
        let mut q = TaskQueue::new(4);
        q.submit(make_task(
            "t1",
            TaskType::CleanupUploads,
            TaskPriority::Normal,
        ));
        q.submit(make_task("t2", TaskType::PurgeCache, TaskPriority::Normal));
        q.dequeue(); // t1 running
        assert_eq!(q.tasks_by_status(TaskStatus::Running).len(), 1);
        assert_eq!(q.tasks_by_status(TaskStatus::Queued).len(), 1);
    }

    #[test]
    fn test_tasks_by_owner() {
        let mut q = TaskQueue::new(4);
        q.submit(
            make_task("t1", TaskType::CleanupUploads, TaskPriority::Normal).with_owner("alice"),
        );
        q.submit(make_task("t2", TaskType::PurgeCache, TaskPriority::Normal).with_owner("bob"));
        assert_eq!(q.tasks_by_owner("alice").len(), 1);
    }

    #[test]
    fn test_generate_id() {
        let mut q = TaskQueue::new(4);
        let id1 = q.generate_id();
        let id2 = q.generate_id();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_cleanup_old_tasks() {
        let mut q = TaskQueue::new(4);
        // Create a task with a created_at timestamp in the past
        let mut task = make_task("t1", TaskType::CleanupUploads, TaskPriority::Normal);
        task.created_at = task.created_at.saturating_sub(10); // 10 seconds ago
        q.submit(task);
        q.dequeue();
        q.complete_task("t1", HashMap::new());
        // Anything older than 5 seconds should be cleaned
        let cleaned = q.cleanup_old_tasks(Duration::from_secs(5));
        assert_eq!(cleaned, 1);
    }

    #[test]
    fn test_stats_completion_rate() {
        let mut stats = TaskQueueStats::default();
        stats.completed = 9;
        stats.failed = 1;
        assert!((stats.completion_rate() - 0.9).abs() < 1e-9);
    }

    #[test]
    fn test_total_tasks() {
        let mut q = TaskQueue::new(4);
        q.submit(make_task(
            "t1",
            TaskType::CleanupUploads,
            TaskPriority::Normal,
        ));
        q.submit(make_task("t2", TaskType::PurgeCache, TaskPriority::Normal));
        assert_eq!(q.total_tasks(), 2);
    }
}
