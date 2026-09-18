#![allow(dead_code)]
//! Task group management — group policies, task collections, and aggregate results.

/// Policy controlling how tasks within a group are executed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GroupPolicy {
    /// Tasks run one after another in order.
    Sequential,
    /// All tasks run concurrently.
    Parallel,
    /// Stop after the first task that succeeds; remaining tasks are skipped.
    FirstSuccess,
}

impl GroupPolicy {
    /// Human-readable description of this policy.
    #[must_use]
    pub fn description(&self) -> &'static str {
        match self {
            Self::Sequential => "Run tasks sequentially in order",
            Self::Parallel => "Run all tasks in parallel",
            Self::FirstSuccess => "Stop on first successful task",
        }
    }

    /// Returns `true` if tasks in this policy may run concurrently.
    #[must_use]
    pub fn allows_concurrency(&self) -> bool {
        matches!(self, Self::Parallel)
    }
}

/// A named task with an optional weight used for scheduling priority.
#[derive(Debug, Clone)]
pub struct Task {
    /// Unique name identifying the task within a group.
    pub name: String,
    /// Relative weight (higher → higher priority).
    pub weight: u32,
}

impl Task {
    /// Create a new task with the given name and default weight of 1.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            weight: 1,
        }
    }

    /// Create a task with an explicit weight.
    #[must_use]
    pub fn with_weight(name: impl Into<String>, weight: u32) -> Self {
        Self {
            name: name.into(),
            weight,
        }
    }
}

/// A collection of tasks governed by a single execution policy.
#[derive(Debug, Clone)]
pub struct TaskGroup {
    /// Name of this task group.
    pub name: String,
    /// Execution policy.
    pub policy: GroupPolicy,
    tasks: Vec<Task>,
}

impl TaskGroup {
    /// Create a new task group with the given name and policy.
    #[must_use]
    pub fn new(name: impl Into<String>, policy: GroupPolicy) -> Self {
        Self {
            name: name.into(),
            policy,
            tasks: Vec::new(),
        }
    }

    /// Add a task to the group.
    pub fn add_task(&mut self, task: Task) {
        self.tasks.push(task);
    }

    /// Return the number of tasks currently in the group.
    #[must_use]
    pub fn task_count(&self) -> usize {
        self.tasks.len()
    }

    /// Return an immutable slice of all tasks.
    #[must_use]
    pub fn tasks(&self) -> &[Task] {
        &self.tasks
    }

    /// Return the total combined weight of all tasks.
    #[must_use]
    pub fn total_weight(&self) -> u32 {
        self.tasks.iter().map(|t| t.weight).sum()
    }

    /// Return `true` if the group has no tasks.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tasks.is_empty()
    }
}

/// Outcome of a single task execution within a group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskOutcome {
    /// Task completed successfully.
    Success,
    /// Task failed with the given message.
    Failure(String),
    /// Task was skipped (e.g. due to `FirstSuccess` policy).
    Skipped,
}

impl TaskOutcome {
    /// Returns `true` if this outcome represents success.
    #[must_use]
    pub fn is_success(&self) -> bool {
        matches!(self, Self::Success)
    }

    /// Returns `true` if this outcome represents a failure.
    #[must_use]
    pub fn is_failure(&self) -> bool {
        matches!(self, Self::Failure(_))
    }
}

/// Aggregate result for all tasks within a group.
#[derive(Debug, Clone)]
pub struct GroupResult {
    /// Per-task outcomes in submission order.
    pub outcomes: Vec<(String, TaskOutcome)>,
}

impl GroupResult {
    /// Create an empty group result.
    #[must_use]
    pub fn new() -> Self {
        Self {
            outcomes: Vec::new(),
        }
    }

    /// Record the outcome of a named task.
    pub fn record(&mut self, task_name: impl Into<String>, outcome: TaskOutcome) {
        self.outcomes.push((task_name.into(), outcome));
    }

    /// Returns `true` if every recorded outcome is `Success`.
    #[must_use]
    pub fn all_success(&self) -> bool {
        !self.outcomes.is_empty()
            && self
                .outcomes
                .iter()
                .all(|(_, o)| matches!(o, TaskOutcome::Success))
    }

    /// Returns `true` if any recorded outcome is a failure.
    #[must_use]
    pub fn any_failure(&self) -> bool {
        self.outcomes
            .iter()
            .any(|(_, o)| matches!(o, TaskOutcome::Failure(_)))
    }

    /// Count of successful tasks.
    #[must_use]
    pub fn success_count(&self) -> usize {
        self.outcomes
            .iter()
            .filter(|(_, o)| matches!(o, TaskOutcome::Success))
            .count()
    }

    /// Count of failed tasks.
    #[must_use]
    pub fn failure_count(&self) -> usize {
        self.outcomes
            .iter()
            .filter(|(_, o)| matches!(o, TaskOutcome::Failure(_)))
            .count()
    }

    /// Count of skipped tasks.
    #[must_use]
    pub fn skipped_count(&self) -> usize {
        self.outcomes
            .iter()
            .filter(|(_, o)| matches!(o, TaskOutcome::Skipped))
            .count()
    }
}

impl Default for GroupResult {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_group_policy_description_sequential() {
        assert!(!GroupPolicy::Sequential.description().is_empty());
    }

    #[test]
    fn test_group_policy_description_parallel() {
        assert!(!GroupPolicy::Parallel.description().is_empty());
    }

    #[test]
    fn test_group_policy_description_first_success() {
        assert!(!GroupPolicy::FirstSuccess.description().is_empty());
    }

    #[test]
    fn test_group_policy_allows_concurrency() {
        assert!(GroupPolicy::Parallel.allows_concurrency());
        assert!(!GroupPolicy::Sequential.allows_concurrency());
        assert!(!GroupPolicy::FirstSuccess.allows_concurrency());
    }

    #[test]
    fn test_task_default_weight() {
        let t = Task::new("encode");
        assert_eq!(t.weight, 1);
        assert_eq!(t.name, "encode");
    }

    #[test]
    fn test_task_with_weight() {
        let t = Task::with_weight("upload", 5);
        assert_eq!(t.weight, 5);
    }

    #[test]
    fn test_task_group_add_and_count() {
        let mut g = TaskGroup::new("pipeline", GroupPolicy::Sequential);
        assert_eq!(g.task_count(), 0);
        assert!(g.is_empty());
        g.add_task(Task::new("step1"));
        g.add_task(Task::new("step2"));
        assert_eq!(g.task_count(), 2);
        assert!(!g.is_empty());
    }

    #[test]
    fn test_task_group_total_weight() {
        let mut g = TaskGroup::new("grp", GroupPolicy::Parallel);
        g.add_task(Task::with_weight("a", 3));
        g.add_task(Task::with_weight("b", 7));
        assert_eq!(g.total_weight(), 10);
    }

    #[test]
    fn test_task_group_tasks_slice() {
        let mut g = TaskGroup::new("grp", GroupPolicy::Parallel);
        g.add_task(Task::new("x"));
        assert_eq!(g.tasks().len(), 1);
        assert_eq!(g.tasks()[0].name, "x");
    }

    #[test]
    fn test_task_outcome_is_success() {
        assert!(TaskOutcome::Success.is_success());
        assert!(!TaskOutcome::Failure("err".to_string()).is_success());
        assert!(!TaskOutcome::Skipped.is_success());
    }

    #[test]
    fn test_task_outcome_is_failure() {
        assert!(TaskOutcome::Failure("oops".to_string()).is_failure());
        assert!(!TaskOutcome::Success.is_failure());
    }

    #[test]
    fn test_group_result_all_success_true() {
        let mut r = GroupResult::new();
        r.record("t1", TaskOutcome::Success);
        r.record("t2", TaskOutcome::Success);
        assert!(r.all_success());
    }

    #[test]
    fn test_group_result_all_success_false_on_failure() {
        let mut r = GroupResult::new();
        r.record("t1", TaskOutcome::Success);
        r.record("t2", TaskOutcome::Failure("fail".to_string()));
        assert!(!r.all_success());
    }

    #[test]
    fn test_group_result_all_success_false_when_empty() {
        let r = GroupResult::new();
        assert!(!r.all_success());
    }

    #[test]
    fn test_group_result_counts() {
        let mut r = GroupResult::new();
        r.record("a", TaskOutcome::Success);
        r.record("b", TaskOutcome::Failure("e".to_string()));
        r.record("c", TaskOutcome::Skipped);
        assert_eq!(r.success_count(), 1);
        assert_eq!(r.failure_count(), 1);
        assert_eq!(r.skipped_count(), 1);
        assert!(r.any_failure());
    }

    #[test]
    fn test_group_result_default() {
        let r = GroupResult::default();
        assert!(r.outcomes.is_empty());
    }
}
