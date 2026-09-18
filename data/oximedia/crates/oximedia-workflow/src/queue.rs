//! Task queue implementation with priority support.

use crate::task::{Task, TaskId, TaskPriority, TaskState};
use dashmap::DashMap;
use std::collections::BinaryHeap;
use std::sync::Arc;
use tokio::sync::{Notify, RwLock};
use tracing::{debug, info};

/// Priority queue item.
#[derive(Debug, Clone)]
struct QueueItem {
    task: Task,
    priority: TaskPriority,
    sequence: u64,
}

impl PartialEq for QueueItem {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.sequence == other.sequence
    }
}

impl Eq for QueueItem {}

impl PartialOrd for QueueItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for QueueItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // First compare by priority (higher priority first)
        match self.priority.cmp(&other.priority) {
            std::cmp::Ordering::Equal => {
                // Then by sequence (lower sequence first for FIFO within same priority)
                other.sequence.cmp(&self.sequence)
            }
            ordering => ordering,
        }
    }
}

/// Task queue with priority support.
pub struct TaskQueue {
    /// Pending tasks ordered by priority.
    pending: Arc<RwLock<BinaryHeap<QueueItem>>>,
    /// Task lookup by ID.
    tasks: Arc<DashMap<TaskId, Task>>,
    /// Running tasks.
    running: Arc<DashMap<TaskId, Task>>,
    /// Sequence counter for FIFO ordering within same priority.
    sequence: Arc<RwLock<u64>>,
    /// Notification for new tasks.
    notify: Arc<Notify>,
    /// Maximum queue size (0 = unlimited).
    max_size: usize,
}

impl TaskQueue {
    /// Create a new task queue.
    #[must_use]
    pub fn new() -> Self {
        Self {
            pending: Arc::new(RwLock::new(BinaryHeap::new())),
            tasks: Arc::new(DashMap::new()),
            running: Arc::new(DashMap::new()),
            sequence: Arc::new(RwLock::new(0)),
            notify: Arc::new(Notify::new()),
            max_size: 0,
        }
    }

    /// Create a new task queue with maximum size.
    #[must_use]
    pub fn with_max_size(max_size: usize) -> Self {
        Self {
            pending: Arc::new(RwLock::new(BinaryHeap::new())),
            tasks: Arc::new(DashMap::new()),
            running: Arc::new(DashMap::new()),
            sequence: Arc::new(RwLock::new(0)),
            notify: Arc::new(Notify::new()),
            max_size,
        }
    }

    /// Enqueue a task.
    pub async fn enqueue(&self, mut task: Task) -> crate::error::Result<()> {
        // Check queue size limit
        if self.max_size > 0 && self.len().await >= self.max_size {
            return Err(crate::error::WorkflowError::ResourceLimitExceeded {
                resource: "queue_size".to_string(),
                limit: self.max_size.to_string(),
            });
        }

        task.set_state(TaskState::Queued)?;

        let priority = task.priority;
        let task_id = task.id;

        // Get sequence number
        let mut seq = self.sequence.write().await;
        let sequence = *seq;
        *seq += 1;
        drop(seq);

        // Add to tasks map
        self.tasks.insert(task_id, task.clone());

        // Add to pending queue
        let item = QueueItem {
            task,
            priority,
            sequence,
        };

        self.pending.write().await.push(item);

        debug!("Enqueued task {} with priority {:?}", task_id, priority);

        // Notify waiting consumers
        self.notify.notify_one();

        Ok(())
    }

    /// Dequeue the highest priority task.
    pub async fn dequeue(&self) -> Option<Task> {
        loop {
            // Try to get a task
            let mut pending = self.pending.write().await;
            if let Some(item) = pending.pop() {
                drop(pending);

                let mut task = item.task;
                let task_id = task.id;

                // Move to running
                if task.set_state(TaskState::Running).is_ok() {
                    self.running.insert(task_id, task.clone());
                    debug!("Dequeued task {}", task_id);
                    return Some(task);
                }

                // State transition failed, try next task
                continue;
            }

            drop(pending);

            // No tasks available, wait for notification
            self.notify.notified().await;
        }
    }

    /// Try to dequeue without blocking.
    pub async fn try_dequeue(&self) -> Option<Task> {
        let mut pending = self.pending.write().await;
        while let Some(item) = pending.pop() {
            let mut task = item.task;
            let task_id = task.id;

            if task.set_state(TaskState::Running).is_ok() {
                self.running.insert(task_id, task.clone());
                debug!("Dequeued task {}", task_id);
                return Some(task);
            }
        }

        None
    }

    /// Complete a task.
    pub async fn complete(&self, task_id: TaskId, success: bool) {
        if let Some((_, mut task)) = self.running.remove(&task_id) {
            let new_state = if success {
                TaskState::Completed
            } else {
                TaskState::Failed
            };

            if task.set_state(new_state).is_ok() {
                self.tasks.insert(task_id, task);
            }

            info!("Task {} completed with success={}", task_id, success);
        }
    }

    /// Get task by ID.
    #[must_use]
    pub fn get_task(&self, task_id: &TaskId) -> Option<Task> {
        self.tasks.get(task_id).map(|t| t.clone())
    }

    /// Get running task by ID.
    #[must_use]
    pub fn get_running_task(&self, task_id: &TaskId) -> Option<Task> {
        self.running.get(task_id).map(|t| t.clone())
    }

    /// Get queue length (pending + running).
    pub async fn len(&self) -> usize {
        let pending_count = self.pending.read().await.len();
        let running_count = self.running.len();
        pending_count + running_count
    }

    /// Check if queue is empty.
    pub async fn is_empty(&self) -> bool {
        self.len().await == 0
    }

    /// Get number of pending tasks.
    pub async fn pending_count(&self) -> usize {
        self.pending.read().await.len()
    }

    /// Get number of running tasks.
    #[must_use]
    pub fn running_count(&self) -> usize {
        self.running.len()
    }

    /// Clear all tasks.
    pub async fn clear(&self) {
        self.pending.write().await.clear();
        self.running.clear();
        self.tasks.clear();
        *self.sequence.write().await = 0;
        info!("Queue cleared");
    }

    /// Get all pending tasks (for inspection).
    pub async fn get_pending_tasks(&self) -> Vec<Task> {
        self.pending
            .read()
            .await
            .iter()
            .map(|item| item.task.clone())
            .collect()
    }

    /// Get all running tasks (for inspection).
    #[must_use]
    pub fn get_running_tasks(&self) -> Vec<Task> {
        self.running
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Remove a task from the queue (if not running).
    pub async fn remove(&self, task_id: TaskId) -> bool {
        if self.running.contains_key(&task_id) {
            return false;
        }

        self.tasks.remove(&task_id);

        // Rebuild pending queue without the task
        let mut pending = self.pending.write().await;
        let items: Vec<_> = pending
            .drain()
            .filter(|item| item.task.id != task_id)
            .collect();

        *pending = items.into_iter().collect();

        true
    }

    /// Requeue a failed task for retry.
    pub async fn requeue(&self, mut task: Task) -> crate::error::Result<()> {
        // Remove from running
        self.running.remove(&task.id);

        // Increment retry count
        task.increment_retry();

        // Check if should retry
        if !task.should_retry() {
            task.set_state(TaskState::Failed)?;
            self.tasks.insert(task.id, task);
            return Ok(());
        }

        // Requeue with delay
        task.set_state(TaskState::Retrying)?;
        let delay = task.retry_delay();

        debug!(
            "Requeueing task {} after {:?} (attempt {})",
            task.id,
            delay,
            task.retry_count + 1
        );

        let _task_id = task.id;
        let queue = self.clone();

        // Spawn delay and requeue
        tokio::spawn(async move {
            tokio::time::sleep(delay).await;
            let _ = queue.enqueue(task).await;
        });

        Ok(())
    }

    /// Get queue statistics.
    #[must_use]
    pub async fn statistics(&self) -> QueueStatistics {
        let pending = self.pending_count().await;
        let running = self.running_count();
        let total = self.tasks.len();

        // Count by priority
        let pending_tasks = self.get_pending_tasks().await;
        let mut low = 0;
        let mut normal = 0;
        let mut high = 0;
        let mut critical = 0;

        for task in pending_tasks {
            match task.priority {
                TaskPriority::Low => low += 1,
                TaskPriority::Normal => normal += 1,
                TaskPriority::High => high += 1,
                TaskPriority::Critical => critical += 1,
            }
        }

        QueueStatistics {
            total_tasks: total,
            pending_tasks: pending,
            running_tasks: running,
            low_priority: low,
            normal_priority: normal,
            high_priority: high,
            critical_priority: critical,
        }
    }
}

impl Clone for TaskQueue {
    fn clone(&self) -> Self {
        Self {
            pending: self.pending.clone(),
            tasks: self.tasks.clone(),
            running: self.running.clone(),
            sequence: self.sequence.clone(),
            notify: self.notify.clone(),
            max_size: self.max_size,
        }
    }
}

impl Default for TaskQueue {
    fn default() -> Self {
        Self::new()
    }
}

/// Queue statistics.
#[derive(Debug, Clone)]
pub struct QueueStatistics {
    /// Total tasks in queue.
    pub total_tasks: usize,
    /// Pending tasks.
    pub pending_tasks: usize,
    /// Running tasks.
    pub running_tasks: usize,
    /// Low priority tasks.
    pub low_priority: usize,
    /// Normal priority tasks.
    pub normal_priority: usize,
    /// High priority tasks.
    pub high_priority: usize,
    /// Critical priority tasks.
    pub critical_priority: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn create_test_task(name: &str, priority: TaskPriority) -> Task {
        Task::new(
            name,
            crate::task::TaskType::Wait {
                duration: Duration::from_secs(1),
            },
        )
        .with_priority(priority)
    }

    #[tokio::test]
    async fn test_queue_creation() {
        let queue = TaskQueue::new();
        assert_eq!(queue.pending_count().await, 0);
        assert_eq!(queue.running_count(), 0);
    }

    #[tokio::test]
    async fn test_enqueue_dequeue() {
        let queue = TaskQueue::new();
        let task = create_test_task("task1", TaskPriority::Normal);
        let task_id = task.id;

        queue.enqueue(task).await.expect("should succeed in test");
        assert_eq!(queue.pending_count().await, 1);

        let dequeued = queue.try_dequeue().await;
        assert!(dequeued.is_some());
        assert_eq!(dequeued.expect("should succeed in test").id, task_id);
        assert_eq!(queue.running_count(), 1);
    }

    #[tokio::test]
    async fn test_priority_ordering() {
        let queue = TaskQueue::new();

        let task_low = create_test_task("low", TaskPriority::Low);
        let task_high = create_test_task("high", TaskPriority::High);
        let task_normal = create_test_task("normal", TaskPriority::Normal);

        queue
            .enqueue(task_low)
            .await
            .expect("should succeed in test");
        queue
            .enqueue(task_high.clone())
            .await
            .expect("should succeed in test");
        queue
            .enqueue(task_normal)
            .await
            .expect("should succeed in test");

        let dequeued = queue.try_dequeue().await.expect("should succeed in test");
        assert_eq!(dequeued.id, task_high.id);
    }

    #[tokio::test]
    async fn test_fifo_within_priority() {
        let queue = TaskQueue::new();

        let task1 = create_test_task("task1", TaskPriority::Normal);
        let task2 = create_test_task("task2", TaskPriority::Normal);
        let task3 = create_test_task("task3", TaskPriority::Normal);

        let id1 = task1.id;

        queue.enqueue(task1).await.expect("should succeed in test");
        queue.enqueue(task2).await.expect("should succeed in test");
        queue.enqueue(task3).await.expect("should succeed in test");

        let dequeued = queue.try_dequeue().await.expect("should succeed in test");
        assert_eq!(dequeued.id, id1);
    }

    #[tokio::test]
    async fn test_complete_task() {
        let queue = TaskQueue::new();
        let task = create_test_task("task1", TaskPriority::Normal);
        let task_id = task.id;

        queue.enqueue(task).await.expect("should succeed in test");
        queue.try_dequeue().await;

        assert_eq!(queue.running_count(), 1);

        queue.complete(task_id, true).await;
        assert_eq!(queue.running_count(), 0);

        let completed_task = queue.get_task(&task_id).expect("should succeed in test");
        assert_eq!(completed_task.state, TaskState::Completed);
    }

    #[tokio::test]
    async fn test_queue_max_size() {
        let queue = TaskQueue::with_max_size(2);

        let task1 = create_test_task("task1", TaskPriority::Normal);
        let task2 = create_test_task("task2", TaskPriority::Normal);
        let task3 = create_test_task("task3", TaskPriority::Normal);

        assert!(queue.enqueue(task1).await.is_ok());
        assert!(queue.enqueue(task2).await.is_ok());
        assert!(queue.enqueue(task3).await.is_err());
    }

    #[tokio::test]
    async fn test_remove_task() {
        let queue = TaskQueue::new();
        let task = create_test_task("task1", TaskPriority::Normal);
        let task_id = task.id;

        queue.enqueue(task).await.expect("should succeed in test");
        assert_eq!(queue.pending_count().await, 1);

        let removed = queue.remove(task_id).await;
        assert!(removed);
        assert_eq!(queue.pending_count().await, 0);
    }

    #[tokio::test]
    async fn test_clear_queue() {
        let queue = TaskQueue::new();

        queue
            .enqueue(create_test_task("task1", TaskPriority::Normal))
            .await
            .expect("should succeed in test");
        queue
            .enqueue(create_test_task("task2", TaskPriority::Normal))
            .await
            .expect("should succeed in test");

        queue.clear().await;
        assert_eq!(queue.pending_count().await, 0);
    }

    #[tokio::test]
    async fn test_queue_statistics() {
        let queue = TaskQueue::new();

        queue
            .enqueue(create_test_task("low", TaskPriority::Low))
            .await
            .expect("should succeed in test");
        queue
            .enqueue(create_test_task("normal", TaskPriority::Normal))
            .await
            .expect("should succeed in test");
        queue
            .enqueue(create_test_task("high", TaskPriority::High))
            .await
            .expect("should succeed in test");

        let stats = queue.statistics().await;
        assert_eq!(stats.pending_tasks, 3);
        assert_eq!(stats.low_priority, 1);
        assert_eq!(stats.normal_priority, 1);
        assert_eq!(stats.high_priority, 1);
    }

    #[tokio::test]
    async fn test_requeue() {
        let queue = TaskQueue::new();
        let mut task = create_test_task("task1", TaskPriority::Normal);
        task.retry = crate::task::RetryPolicy {
            max_attempts: 3,
            initial_delay: Duration::from_millis(10),
            max_delay: Duration::from_secs(1),
            backoff_multiplier: 2.0,
            exponential_backoff: true,
        };

        let _task_id = task.id;

        queue.enqueue(task).await.expect("should succeed in test");
        let task = queue.try_dequeue().await.expect("should succeed in test");

        // Requeue for retry
        queue.requeue(task).await.expect("should succeed in test");

        // Wait for requeue delay — use a generous timeout so this test is
        // robust under concurrent workspace-level test load (initial_delay is
        // 10 ms; 500 ms gives 50× headroom for a loaded scheduler).
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Task should be back in queue
        assert!(queue.pending_count().await > 0 || queue.running_count() > 0);
    }
}
