//! Workflow definition and management.

use crate::error::{Result, WorkflowError};
use crate::task::{Task, TaskId};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

/// Unique workflow identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WorkflowId(Uuid);

impl WorkflowId {
    /// Create a new random workflow ID.
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Get the underlying UUID.
    #[must_use]
    pub const fn as_uuid(&self) -> &Uuid {
        &self.0
    }
}

impl Default for WorkflowId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for WorkflowId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<Uuid> for WorkflowId {
    fn from(uuid: Uuid) -> Self {
        Self(uuid)
    }
}

/// Workflow state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkflowState {
    /// Workflow is created but not started.
    Created,
    /// Workflow is scheduled to run.
    Scheduled,
    /// Workflow is running.
    Running,
    /// Workflow is paused.
    Paused,
    /// Workflow completed successfully.
    Completed,
    /// Workflow failed.
    Failed,
    /// Workflow was cancelled.
    Cancelled,
}

impl WorkflowState {
    /// Check if workflow is in a terminal state.
    #[must_use]
    pub const fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }

    /// Check if workflow is active.
    #[must_use]
    pub const fn is_active(&self) -> bool {
        matches!(self, Self::Running)
    }
}

/// Edge between tasks in the workflow DAG.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    /// Source task ID.
    pub from: TaskId,
    /// Target task ID.
    pub to: TaskId,
    /// Edge condition (optional).
    pub condition: Option<String>,
}

impl Edge {
    /// Create a new edge.
    #[must_use]
    pub const fn new(from: TaskId, to: TaskId) -> Self {
        Self {
            from,
            to,
            condition: None,
        }
    }

    /// Create an edge with a condition.
    #[must_use]
    pub fn with_condition(from: TaskId, to: TaskId, condition: String) -> Self {
        Self {
            from,
            to,
            condition: Some(condition),
        }
    }
}

/// Workflow configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowConfig {
    /// Maximum concurrent tasks.
    #[serde(default = "default_max_concurrent")]
    pub max_concurrent_tasks: usize,
    /// Global timeout for workflow.
    #[serde(default)]
    pub global_timeout: Option<std::time::Duration>,
    /// Whether to stop on first error.
    #[serde(default)]
    pub fail_fast: bool,
    /// Whether to continue on task failure.
    #[serde(default)]
    pub continue_on_error: bool,
    /// Variables available to all tasks.
    #[serde(default)]
    pub variables: HashMap<String, serde_json::Value>,
}

fn default_max_concurrent() -> usize {
    4
}

impl Default for WorkflowConfig {
    fn default() -> Self {
        Self {
            max_concurrent_tasks: default_max_concurrent(),
            global_timeout: None,
            fail_fast: false,
            continue_on_error: false,
            variables: HashMap::new(),
        }
    }
}

/// Workflow definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Workflow {
    /// Unique workflow identifier.
    pub id: WorkflowId,
    /// Workflow name.
    pub name: String,
    /// Workflow description.
    #[serde(default)]
    pub description: String,
    /// Tasks in the workflow.
    pub tasks: HashMap<TaskId, Task>,
    /// Edges defining task dependencies.
    pub edges: Vec<Edge>,
    /// Workflow configuration.
    #[serde(default)]
    pub config: WorkflowConfig,
    /// Current state.
    #[serde(default = "default_workflow_state")]
    pub state: WorkflowState,
    /// Workflow metadata.
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

fn default_workflow_state() -> WorkflowState {
    WorkflowState::Created
}

impl Workflow {
    /// Create a new workflow.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            id: WorkflowId::new(),
            name: name.into(),
            description: String::new(),
            tasks: HashMap::new(),
            edges: Vec::new(),
            config: WorkflowConfig::default(),
            state: WorkflowState::Created,
            metadata: HashMap::new(),
        }
    }

    /// Add a task to the workflow.
    pub fn add_task(&mut self, task: Task) -> TaskId {
        let task_id = task.id;
        self.tasks.insert(task_id, task);
        task_id
    }

    /// Add an edge between tasks.
    pub fn add_edge(&mut self, from: TaskId, to: TaskId) -> Result<()> {
        // Validate that both tasks exist
        if !self.tasks.contains_key(&from) {
            return Err(WorkflowError::TaskNotFound(from.to_string()));
        }
        if !self.tasks.contains_key(&to) {
            return Err(WorkflowError::TaskNotFound(to.to_string()));
        }

        self.edges.push(Edge::new(from, to));
        Ok(())
    }

    /// Add a conditional edge.
    pub fn add_conditional_edge(
        &mut self,
        from: TaskId,
        to: TaskId,
        condition: String,
    ) -> Result<()> {
        if !self.tasks.contains_key(&from) {
            return Err(WorkflowError::TaskNotFound(from.to_string()));
        }
        if !self.tasks.contains_key(&to) {
            return Err(WorkflowError::TaskNotFound(to.to_string()));
        }

        self.edges.push(Edge::with_condition(from, to, condition));
        Ok(())
    }

    /// Get task by ID.
    #[must_use]
    pub fn get_task(&self, task_id: &TaskId) -> Option<&Task> {
        self.tasks.get(task_id)
    }

    /// Get mutable task by ID.
    pub fn get_task_mut(&mut self, task_id: &TaskId) -> Option<&mut Task> {
        self.tasks.get_mut(task_id)
    }

    /// Get all tasks.
    #[must_use]
    pub fn tasks(&self) -> impl Iterator<Item = &Task> {
        self.tasks.values()
    }

    /// Get dependencies for a task.
    #[must_use]
    pub fn get_dependencies(&self, task_id: &TaskId) -> Vec<TaskId> {
        self.edges
            .iter()
            .filter(|e| &e.to == task_id)
            .map(|e| e.from)
            .collect()
    }

    /// Get dependents for a task.
    #[must_use]
    pub fn get_dependents(&self, task_id: &TaskId) -> Vec<TaskId> {
        self.edges
            .iter()
            .filter(|e| &e.from == task_id)
            .map(|e| e.to)
            .collect()
    }

    /// Get root tasks (tasks with no dependencies).
    #[must_use]
    pub fn get_root_tasks(&self) -> Vec<TaskId> {
        let has_incoming: HashSet<_> = self.edges.iter().map(|e| e.to).collect();
        self.tasks
            .keys()
            .filter(|id| !has_incoming.contains(id))
            .copied()
            .collect()
    }

    /// Get leaf tasks (tasks with no dependents).
    #[must_use]
    pub fn get_leaf_tasks(&self) -> Vec<TaskId> {
        let has_outgoing: HashSet<_> = self.edges.iter().map(|e| e.from).collect();
        self.tasks
            .keys()
            .filter(|id| !has_outgoing.contains(id))
            .copied()
            .collect()
    }

    /// Validate workflow structure.
    pub fn validate(&self) -> Result<()> {
        // Check for cycles
        if self.has_cycle() {
            return Err(WorkflowError::CycleDetected);
        }

        // Validate all edge references
        for edge in &self.edges {
            if !self.tasks.contains_key(&edge.from) {
                return Err(WorkflowError::TaskNotFound(edge.from.to_string()));
            }
            if !self.tasks.contains_key(&edge.to) {
                return Err(WorkflowError::TaskNotFound(edge.to.to_string()));
            }
        }

        Ok(())
    }

    /// Check if workflow has a cycle.
    #[must_use]
    pub fn has_cycle(&self) -> bool {
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();

        for task_id in self.tasks.keys() {
            if self.has_cycle_util(*task_id, &mut visited, &mut rec_stack) {
                return true;
            }
        }

        false
    }

    fn has_cycle_util(
        &self,
        task_id: TaskId,
        visited: &mut HashSet<TaskId>,
        rec_stack: &mut HashSet<TaskId>,
    ) -> bool {
        if rec_stack.contains(&task_id) {
            return true;
        }

        if visited.contains(&task_id) {
            return false;
        }

        visited.insert(task_id);
        rec_stack.insert(task_id);

        for dependent in self.get_dependents(&task_id) {
            if self.has_cycle_util(dependent, visited, rec_stack) {
                return true;
            }
        }

        rec_stack.remove(&task_id);
        false
    }

    /// Get topological order of tasks.
    pub fn topological_sort(&self) -> Result<Vec<TaskId>> {
        if self.has_cycle() {
            return Err(WorkflowError::CycleDetected);
        }

        let mut result = Vec::new();
        let mut visited = HashSet::new();

        for task_id in self.tasks.keys() {
            self.topological_sort_util(*task_id, &mut visited, &mut result);
        }

        result.reverse();
        Ok(result)
    }

    fn topological_sort_util(
        &self,
        task_id: TaskId,
        visited: &mut HashSet<TaskId>,
        result: &mut Vec<TaskId>,
    ) {
        if visited.contains(&task_id) {
            return;
        }

        visited.insert(task_id);

        for dependent in self.get_dependents(&task_id) {
            self.topological_sort_util(dependent, visited, result);
        }

        result.push(task_id);
    }

    /// Set workflow description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Set workflow configuration.
    #[must_use]
    pub fn with_config(mut self, config: WorkflowConfig) -> Self {
        self.config = config;
        self
    }

    /// Add metadata.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::task::TaskType;
    use std::time::Duration;

    fn create_test_task(name: &str) -> Task {
        Task::new(
            name,
            TaskType::Wait {
                duration: Duration::from_secs(1),
            },
        )
    }

    #[test]
    fn test_workflow_creation() {
        let workflow = Workflow::new("test-workflow");
        assert_eq!(workflow.name, "test-workflow");
        assert_eq!(workflow.state, WorkflowState::Created);
        assert!(workflow.tasks.is_empty());
    }

    #[test]
    fn test_add_task() {
        let mut workflow = Workflow::new("test");
        let task = create_test_task("task1");
        let task_id = workflow.add_task(task);
        assert_eq!(workflow.tasks.len(), 1);
        assert!(workflow.get_task(&task_id).is_some());
    }

    #[test]
    fn test_add_edge() {
        let mut workflow = Workflow::new("test");
        let task1 = create_test_task("task1");
        let task2 = create_test_task("task2");
        let id1 = workflow.add_task(task1);
        let id2 = workflow.add_task(task2);

        assert!(workflow.add_edge(id1, id2).is_ok());
        assert_eq!(workflow.edges.len(), 1);
    }

    #[test]
    fn test_add_edge_invalid_task() {
        let mut workflow = Workflow::new("test");
        let task1 = create_test_task("task1");
        let id1 = workflow.add_task(task1);
        let invalid_id = TaskId::new();

        assert!(workflow.add_edge(id1, invalid_id).is_err());
    }

    #[test]
    fn test_get_dependencies() {
        let mut workflow = Workflow::new("test");
        let task1 = create_test_task("task1");
        let task2 = create_test_task("task2");
        let task3 = create_test_task("task3");

        let id1 = workflow.add_task(task1);
        let id2 = workflow.add_task(task2);
        let id3 = workflow.add_task(task3);

        workflow.add_edge(id1, id3).expect("should succeed in test");
        workflow.add_edge(id2, id3).expect("should succeed in test");

        let deps = workflow.get_dependencies(&id3);
        assert_eq!(deps.len(), 2);
        assert!(deps.contains(&id1));
        assert!(deps.contains(&id2));
    }

    #[test]
    fn test_get_root_tasks() {
        let mut workflow = Workflow::new("test");
        let task1 = create_test_task("task1");
        let task2 = create_test_task("task2");
        let task3 = create_test_task("task3");

        let id1 = workflow.add_task(task1);
        let id2 = workflow.add_task(task2);
        let id3 = workflow.add_task(task3);

        workflow.add_edge(id1, id3).expect("should succeed in test");
        workflow.add_edge(id2, id3).expect("should succeed in test");

        let roots = workflow.get_root_tasks();
        assert_eq!(roots.len(), 2);
        assert!(roots.contains(&id1));
        assert!(roots.contains(&id2));
    }

    #[test]
    fn test_get_leaf_tasks() {
        let mut workflow = Workflow::new("test");
        let task1 = create_test_task("task1");
        let task2 = create_test_task("task2");
        let task3 = create_test_task("task3");

        let id1 = workflow.add_task(task1);
        let id2 = workflow.add_task(task2);
        let id3 = workflow.add_task(task3);

        workflow.add_edge(id1, id2).expect("should succeed in test");
        workflow.add_edge(id1, id3).expect("should succeed in test");

        let leaves = workflow.get_leaf_tasks();
        assert_eq!(leaves.len(), 2);
        assert!(leaves.contains(&id2));
        assert!(leaves.contains(&id3));
    }

    #[test]
    fn test_has_cycle_no_cycle() {
        let mut workflow = Workflow::new("test");
        let task1 = create_test_task("task1");
        let task2 = create_test_task("task2");
        let task3 = create_test_task("task3");

        let id1 = workflow.add_task(task1);
        let id2 = workflow.add_task(task2);
        let id3 = workflow.add_task(task3);

        workflow.add_edge(id1, id2).expect("should succeed in test");
        workflow.add_edge(id2, id3).expect("should succeed in test");

        assert!(!workflow.has_cycle());
    }

    #[test]
    fn test_has_cycle_with_cycle() {
        let mut workflow = Workflow::new("test");
        let task1 = create_test_task("task1");
        let task2 = create_test_task("task2");

        let id1 = workflow.add_task(task1);
        let id2 = workflow.add_task(task2);

        workflow.add_edge(id1, id2).expect("should succeed in test");
        workflow.add_edge(id2, id1).expect("should succeed in test");

        assert!(workflow.has_cycle());
    }

    #[test]
    fn test_validate_success() {
        let mut workflow = Workflow::new("test");
        let task1 = create_test_task("task1");
        let task2 = create_test_task("task2");

        let id1 = workflow.add_task(task1);
        let id2 = workflow.add_task(task2);
        workflow.add_edge(id1, id2).expect("should succeed in test");

        assert!(workflow.validate().is_ok());
    }

    #[test]
    fn test_validate_cycle() {
        let mut workflow = Workflow::new("test");
        let task1 = create_test_task("task1");
        let task2 = create_test_task("task2");

        let id1 = workflow.add_task(task1);
        let id2 = workflow.add_task(task2);
        workflow.add_edge(id1, id2).expect("should succeed in test");
        workflow.add_edge(id2, id1).expect("should succeed in test");

        assert!(workflow.validate().is_err());
    }

    #[test]
    fn test_topological_sort() {
        let mut workflow = Workflow::new("test");
        let task1 = create_test_task("task1");
        let task2 = create_test_task("task2");
        let task3 = create_test_task("task3");

        let id1 = workflow.add_task(task1);
        let id2 = workflow.add_task(task2);
        let id3 = workflow.add_task(task3);

        workflow.add_edge(id1, id2).expect("should succeed in test");
        workflow.add_edge(id2, id3).expect("should succeed in test");

        let sorted = workflow.topological_sort().expect("should succeed in test");
        assert_eq!(sorted.len(), 3);

        let pos1 = sorted
            .iter()
            .position(|&x| x == id1)
            .expect("should succeed in test");
        let pos2 = sorted
            .iter()
            .position(|&x| x == id2)
            .expect("should succeed in test");
        let pos3 = sorted
            .iter()
            .position(|&x| x == id3)
            .expect("should succeed in test");

        assert!(pos1 < pos2);
        assert!(pos2 < pos3);
    }

    #[test]
    fn test_workflow_state_terminal() {
        assert!(WorkflowState::Completed.is_terminal());
        assert!(WorkflowState::Failed.is_terminal());
        assert!(!WorkflowState::Running.is_terminal());
    }

    #[test]
    fn test_conditional_edge() {
        let mut workflow = Workflow::new("test");
        let task1 = create_test_task("task1");
        let task2 = create_test_task("task2");

        let id1 = workflow.add_task(task1);
        let id2 = workflow.add_task(task2);

        assert!(workflow
            .add_conditional_edge(id1, id2, "result == true".to_string())
            .is_ok());
        assert_eq!(workflow.edges.len(), 1);
        assert!(workflow.edges[0].condition.is_some());
    }
}
