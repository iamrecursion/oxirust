//! Task tracking and progress monitoring.

use crate::task::{Task, TaskStatus};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Task tracker for monitoring progress.
pub struct TaskTracker {
    tasks: HashMap<crate::TaskId, Task>,
}

impl TaskTracker {
    /// Create a new task tracker.
    #[must_use]
    pub fn new() -> Self {
        Self {
            tasks: HashMap::new(),
        }
    }

    /// Add a task to track.
    pub fn add_task(&mut self, task: Task) {
        self.tasks.insert(task.id, task);
    }

    /// Get task by ID.
    #[must_use]
    pub fn get_task(&self, id: crate::TaskId) -> Option<&Task> {
        self.tasks.get(&id)
    }

    /// Get all tasks.
    #[must_use]
    pub fn all_tasks(&self) -> Vec<&Task> {
        self.tasks.values().collect()
    }

    /// Get tasks by status.
    #[must_use]
    pub fn tasks_by_status(&self, status: TaskStatus) -> Vec<&Task> {
        self.tasks.values().filter(|t| t.status == status).collect()
    }

    /// Get overdue tasks.
    #[must_use]
    pub fn overdue_tasks(&self) -> Vec<&Task> {
        self.tasks.values().filter(|t| t.is_overdue()).collect()
    }

    /// Get task statistics.
    #[must_use]
    pub fn statistics(&self) -> TaskStatistics {
        TaskStatistics::from_tasks(self.tasks.values())
    }
}

impl Default for TaskTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Task statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TaskStatistics {
    /// Total tasks.
    pub total: usize,
    /// Open tasks.
    pub open: usize,
    /// In progress tasks.
    pub in_progress: usize,
    /// Completed tasks.
    pub completed: usize,
    /// Cancelled tasks.
    pub cancelled: usize,
    /// Blocked tasks.
    pub blocked: usize,
    /// Overdue tasks.
    pub overdue: usize,
}

impl TaskStatistics {
    /// Create statistics from an iterator of tasks.
    pub fn from_tasks<'a, I>(tasks: I) -> Self
    where
        I: Iterator<Item = &'a Task>,
    {
        let mut stats = Self::default();

        for task in tasks {
            stats.total += 1;
            match task.status {
                TaskStatus::Open => stats.open += 1,
                TaskStatus::InProgress => stats.in_progress += 1,
                TaskStatus::Completed => stats.completed += 1,
                TaskStatus::Cancelled => stats.cancelled += 1,
                TaskStatus::Blocked => stats.blocked += 1,
            }
            if task.is_overdue() {
                stats.overdue += 1;
            }
        }

        stats
    }

    /// Get completion percentage.
    #[must_use]
    pub fn completion_percentage(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }

        (self.completed as f64 / self.total as f64) * 100.0
    }

    /// Get active task count.
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.open + self.in_progress + self.blocked
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{SessionId, User, UserRole};
    use chrono::Utc;

    fn create_test_task(status: TaskStatus) -> Task {
        Task {
            id: crate::TaskId::new(),
            session_id: SessionId::new(),
            title: "Test".to_string(),
            description: None,
            assignee: User {
                id: "user1".to_string(),
                name: "User 1".to_string(),
                email: "user1@example.com".to_string(),
                role: UserRole::Reviewer,
            },
            creator: User {
                id: "user2".to_string(),
                name: "User 2".to_string(),
                email: "user2@example.com".to_string(),
                role: UserRole::Reviewer,
            },
            status,
            priority: crate::task::TaskPriority::Normal,
            deadline: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            completed_at: None,
        }
    }

    #[test]
    fn test_task_tracker() {
        let mut tracker = TaskTracker::new();

        let task1 = create_test_task(TaskStatus::Open);
        let task2 = create_test_task(TaskStatus::Completed);

        let task1_id = task1.id;

        tracker.add_task(task1);
        tracker.add_task(task2);

        assert_eq!(tracker.all_tasks().len(), 2);
        assert!(tracker.get_task(task1_id).is_some());
    }

    #[test]
    fn test_tasks_by_status() {
        let mut tracker = TaskTracker::new();

        tracker.add_task(create_test_task(TaskStatus::Open));
        tracker.add_task(create_test_task(TaskStatus::Open));
        tracker.add_task(create_test_task(TaskStatus::Completed));

        let open_tasks = tracker.tasks_by_status(TaskStatus::Open);
        assert_eq!(open_tasks.len(), 2);
    }

    #[test]
    fn test_task_statistics() {
        let tasks = vec![
            create_test_task(TaskStatus::Open),
            create_test_task(TaskStatus::InProgress),
            create_test_task(TaskStatus::Completed),
            create_test_task(TaskStatus::Completed),
        ];

        let stats = TaskStatistics::from_tasks(tasks.iter());

        assert_eq!(stats.total, 4);
        assert_eq!(stats.open, 1);
        assert_eq!(stats.in_progress, 1);
        assert_eq!(stats.completed, 2);
        assert!((stats.completion_percentage() - 50.0).abs() < 0.001);
    }
}
