//! Collaborative task tracking for post-production workflows.

#![allow(dead_code)]

/// Lifecycle status of a collaborative task.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskStatus {
    /// Task is created but not yet started.
    Open,
    /// Task is actively being worked on.
    InProgress,
    /// Task is submitted for review.
    InReview,
    /// Task has been approved and is complete.
    Approved,
    /// Task has been sent back due to issues.
    Rejected,
}

impl TaskStatus {
    /// Returns `true` for states that cannot transition further.
    pub fn is_terminal(&self) -> bool {
        matches!(self, TaskStatus::Approved | TaskStatus::Rejected)
    }

    /// Returns `true` when transitioning from `self` to `next` is a valid workflow step.
    ///
    /// Allowed transitions:
    /// - `Open` → `InProgress`, `Rejected`
    /// - `InProgress` → `InReview`, `Open`
    /// - `InReview` → `Approved`, `Rejected`, `InProgress`
    /// - `Approved` → (none)
    /// - `Rejected` → `Open`
    pub fn can_transition_to(&self, next: &TaskStatus) -> bool {
        match self {
            TaskStatus::Open => {
                matches!(next, TaskStatus::InProgress | TaskStatus::Rejected)
            }
            TaskStatus::InProgress => {
                matches!(next, TaskStatus::InReview | TaskStatus::Open)
            }
            TaskStatus::InReview => {
                matches!(
                    next,
                    TaskStatus::Approved | TaskStatus::Rejected | TaskStatus::InProgress
                )
            }
            TaskStatus::Approved => false,
            TaskStatus::Rejected => matches!(next, TaskStatus::Open),
        }
    }
}

/// Priority level for a task.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPriority {
    /// Low urgency.
    Low,
    /// Standard priority.
    Medium,
    /// High urgency.
    High,
    /// Drop-everything priority.
    Critical,
}

impl TaskPriority {
    /// Numeric priority level (higher number = higher priority).
    pub fn level(&self) -> u32 {
        match self {
            TaskPriority::Low => 1,
            TaskPriority::Medium => 2,
            TaskPriority::High => 3,
            TaskPriority::Critical => 4,
        }
    }
}

/// A single collaborative task.
pub struct CollabTask {
    /// Unique task identifier.
    pub id: u64,
    /// Short human-readable title.
    pub title: String,
    /// Username or display name of the assigned user.
    pub assignee: String,
    /// Current status.
    pub status: TaskStatus,
    /// Urgency level.
    pub priority: TaskPriority,
    /// Optional deadline as a Unix epoch (seconds). `None` means no deadline.
    pub due_epoch: Option<u64>,
}

impl CollabTask {
    /// Returns `true` when the task has a deadline that has passed.
    pub fn is_overdue(&self, now_epoch: u64) -> bool {
        match self.due_epoch {
            Some(due) => now_epoch > due && !self.status.is_terminal(),
            None => false,
        }
    }

    /// Attempt to transition this task to `next_status`.
    ///
    /// Returns `true` and mutates `status` when the transition is valid.
    /// Returns `false` without mutating when the transition is not allowed.
    pub fn transition(&mut self, next_status: &TaskStatus) -> bool {
        if self.status.can_transition_to(next_status) {
            self.status = next_status.clone();
            true
        } else {
            false
        }
    }
}

/// A board that manages multiple collaborative tasks.
pub struct TaskBoard {
    /// All tasks on this board.
    pub tasks: Vec<CollabTask>,
    /// Next available task id (auto-incremented).
    pub next_id: u64,
}

impl TaskBoard {
    /// Create an empty board.
    pub fn new() -> Self {
        Self {
            tasks: Vec::new(),
            next_id: 1,
        }
    }

    /// Add a new task and return its assigned id.
    pub fn add_task(
        &mut self,
        title: String,
        assignee: String,
        priority: TaskPriority,
        due_epoch: Option<u64>,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.tasks.push(CollabTask {
            id,
            title,
            assignee,
            status: TaskStatus::Open,
            priority,
            due_epoch,
        });
        id
    }

    /// Attempt to update a task's status by id.
    ///
    /// Returns `true` when the transition was accepted.
    pub fn update_status(&mut self, task_id: u64, next_status: &TaskStatus) -> bool {
        if let Some(task) = self.tasks.iter_mut().find(|t| t.id == task_id) {
            task.transition(next_status)
        } else {
            false
        }
    }

    /// Return references to all tasks assigned to `assignee`.
    pub fn tasks_by_assignee(&self, assignee: &str) -> Vec<&CollabTask> {
        self.tasks
            .iter()
            .filter(|t| t.assignee == assignee)
            .collect()
    }

    /// Return references to all tasks that are not in a terminal state.
    pub fn open_tasks(&self) -> Vec<&CollabTask> {
        self.tasks
            .iter()
            .filter(|t| !t.status.is_terminal())
            .collect()
    }

    /// Count of tasks in a terminal state (Approved or Rejected).
    pub fn completed_count(&self) -> usize {
        self.tasks.iter().filter(|t| t.status.is_terminal()).count()
    }
}

impl Default for TaskBoard {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- TaskStatus tests ----

    #[test]
    fn test_approved_is_terminal() {
        assert!(TaskStatus::Approved.is_terminal());
    }

    #[test]
    fn test_rejected_is_terminal() {
        assert!(TaskStatus::Rejected.is_terminal());
    }

    #[test]
    fn test_open_not_terminal() {
        assert!(!TaskStatus::Open.is_terminal());
    }

    #[test]
    fn test_open_can_transition_to_in_progress() {
        assert!(TaskStatus::Open.can_transition_to(&TaskStatus::InProgress));
    }

    #[test]
    fn test_approved_cannot_transition() {
        assert!(!TaskStatus::Approved.can_transition_to(&TaskStatus::Open));
        assert!(!TaskStatus::Approved.can_transition_to(&TaskStatus::Rejected));
    }

    #[test]
    fn test_in_review_can_transition_to_approved() {
        assert!(TaskStatus::InReview.can_transition_to(&TaskStatus::Approved));
    }

    #[test]
    fn test_rejected_can_reopen() {
        assert!(TaskStatus::Rejected.can_transition_to(&TaskStatus::Open));
    }

    // ---- TaskPriority tests ----

    #[test]
    fn test_priority_levels_ordered() {
        assert!(TaskPriority::Low.level() < TaskPriority::Medium.level());
        assert!(TaskPriority::Medium.level() < TaskPriority::High.level());
        assert!(TaskPriority::High.level() < TaskPriority::Critical.level());
    }

    #[test]
    fn test_critical_level_is_4() {
        assert_eq!(TaskPriority::Critical.level(), 4);
    }

    // ---- CollabTask tests ----

    #[test]
    fn test_is_overdue_with_past_deadline() {
        let task = CollabTask {
            id: 1,
            title: "test".to_string(),
            assignee: "alice".to_string(),
            status: TaskStatus::Open,
            priority: TaskPriority::High,
            due_epoch: Some(1000),
        };
        assert!(task.is_overdue(2000));
    }

    #[test]
    fn test_is_overdue_approved_task_is_not_overdue() {
        let task = CollabTask {
            id: 1,
            title: "test".to_string(),
            assignee: "alice".to_string(),
            status: TaskStatus::Approved,
            priority: TaskPriority::High,
            due_epoch: Some(1000),
        };
        assert!(!task.is_overdue(2000));
    }

    #[test]
    fn test_is_overdue_no_deadline() {
        let task = CollabTask {
            id: 1,
            title: "test".to_string(),
            assignee: "alice".to_string(),
            status: TaskStatus::Open,
            priority: TaskPriority::Low,
            due_epoch: None,
        };
        assert!(!task.is_overdue(9_999_999));
    }

    #[test]
    fn test_transition_valid_returns_true() {
        let mut task = CollabTask {
            id: 1,
            title: "test".to_string(),
            assignee: "bob".to_string(),
            status: TaskStatus::Open,
            priority: TaskPriority::Medium,
            due_epoch: None,
        };
        assert!(task.transition(&TaskStatus::InProgress));
        assert_eq!(task.status, TaskStatus::InProgress);
    }

    #[test]
    fn test_transition_invalid_returns_false() {
        let mut task = CollabTask {
            id: 1,
            title: "test".to_string(),
            assignee: "bob".to_string(),
            status: TaskStatus::Open,
            priority: TaskPriority::Medium,
            due_epoch: None,
        };
        // Open → Approved is not a valid direct transition
        assert!(!task.transition(&TaskStatus::Approved));
        assert_eq!(task.status, TaskStatus::Open);
    }

    // ---- TaskBoard tests ----

    #[test]
    fn test_add_task_increments_id() {
        let mut board = TaskBoard::new();
        let id1 = board.add_task(
            "Task A".to_string(),
            "alice".to_string(),
            TaskPriority::Low,
            None,
        );
        let id2 = board.add_task(
            "Task B".to_string(),
            "bob".to_string(),
            TaskPriority::High,
            None,
        );
        assert_eq!(id2, id1 + 1);
    }

    #[test]
    fn test_tasks_by_assignee() {
        let mut board = TaskBoard::new();
        board.add_task(
            "A".to_string(),
            "alice".to_string(),
            TaskPriority::Low,
            None,
        );
        board.add_task("B".to_string(), "bob".to_string(), TaskPriority::Low, None);
        board.add_task(
            "C".to_string(),
            "alice".to_string(),
            TaskPriority::Low,
            None,
        );
        assert_eq!(board.tasks_by_assignee("alice").len(), 2);
        assert_eq!(board.tasks_by_assignee("bob").len(), 1);
    }

    #[test]
    fn test_open_tasks_excludes_terminal() {
        let mut board = TaskBoard::new();
        let id = board.add_task(
            "Work".to_string(),
            "alice".to_string(),
            TaskPriority::High,
            None,
        );
        // Move to InProgress then InReview then Approved
        board.update_status(id, &TaskStatus::InProgress);
        board.update_status(id, &TaskStatus::InReview);
        board.update_status(id, &TaskStatus::Approved);
        assert!(board.open_tasks().is_empty());
    }

    #[test]
    fn test_completed_count() {
        let mut board = TaskBoard::new();
        let id = board.add_task(
            "Done".to_string(),
            "alice".to_string(),
            TaskPriority::Critical,
            None,
        );
        assert_eq!(board.completed_count(), 0);
        board.update_status(id, &TaskStatus::InProgress);
        board.update_status(id, &TaskStatus::InReview);
        board.update_status(id, &TaskStatus::Approved);
        assert_eq!(board.completed_count(), 1);
    }

    #[test]
    fn test_update_status_missing_task_returns_false() {
        let mut board = TaskBoard::new();
        assert!(!board.update_status(999, &TaskStatus::InProgress));
    }
}
