//! Orchestration and coordination for distributed systems
//!
//! This module provides orchestration capabilities for managing distributed
//! workflows, task coordination, and resource allocation across cluster nodes.

use crate::error::{CoreError, CoreResult, ErrorContext};
use std::collections::{HashMap, VecDeque};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Task status in the orchestration system
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskStatus {
    /// Task is pending execution
    Pending,
    /// Task has been assigned to a node
    Assigned { nodeid: String },
    /// Task is currently running
    Running { nodeid: String, started_at: Instant },
    /// Task completed successfully
    Completed {
        nodeid: String,
        completed_at: Instant,
    },
    /// Task failed with an error
    Failed { nodeid: String, error: String },
    /// Task was cancelled
    Cancelled,
    /// Task timed out
    TimedOut,
}

/// Priority level for tasks
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPriority {
    /// Low priority task
    Low = 1,
    /// Normal priority task
    Normal = 2,
    /// High priority task
    High = 3,
    /// Critical priority task
    Critical = 4,
}

/// Task definition for orchestration
#[derive(Debug, Clone)]
pub struct Task {
    pub id: String,
    pub name: String,
    pub payload: Vec<u8>,
    pub priority: TaskPriority,
    pub timeout: Option<Duration>,
    pub dependencies: Vec<String>,
    pub retry_count: usize,
    pub maxretries: usize,
    pub created_at: Instant,
    pub status: TaskStatus,
}

impl Task {
    /// Create a new task
    pub fn new(id: String, name: String, payload: Vec<u8>) -> Self {
        Self {
            id,
            name,
            payload,
            priority: TaskPriority::Normal,
            timeout: Some(Duration::from_secs(300)), // 5 minutes default
            dependencies: Vec::new(),
            retry_count: 0,
            maxretries: 3,
            created_at: Instant::now(),
            status: TaskStatus::Pending,
        }
    }

    /// Set task priority
    pub fn with_priority(mut self, priority: TaskPriority) -> Self {
        self.priority = priority;
        self
    }

    /// Set task timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Add task dependencies
    pub fn with_dependencies(mut self, dependencies: Vec<String>) -> Self {
        self.dependencies = dependencies;
        self
    }

    /// Set maximum retry count
    pub fn retries(mut self, maxretries: usize) -> Self {
        self.maxretries = maxretries;
        self
    }

    /// Check if task can be retried
    pub fn can_retry(&self) -> bool {
        self.retry_count < self.maxretries
    }

    /// Increment retry count
    pub fn increment_retry(&mut self) {
        self.retry_count += 1;
    }

    /// Check if task has timed out
    pub fn has_timed_out(&self) -> bool {
        if let Some(timeout) = self.timeout {
            match &self.status {
                TaskStatus::Running { started_at, .. } => {
                    Instant::now().duration_since(*started_at) > timeout
                }
                _ => false,
            }
        } else {
            false
        }
    }
}

/// Workflow definition containing multiple tasks
#[derive(Debug)]
pub struct Workflow {
    pub id: String,
    pub name: String,
    pub tasks: HashMap<String, Task>,
    pub execution_order: Vec<String>,
    pub status: WorkflowStatus,
    pub created_at: Instant,
}

/// Workflow execution status
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkflowStatus {
    /// Workflow is pending execution
    Pending,
    /// Workflow is currently running
    Running,
    /// Workflow completed successfully
    Completed,
    /// Workflow failed
    Failed { error: String },
    /// Workflow was cancelled
    Cancelled,
}

impl Workflow {
    /// Create a new workflow
    pub fn workflow_id(id: String, name: String) -> Self {
        Self {
            id,
            name,
            tasks: HashMap::new(),
            execution_order: Vec::new(),
            status: WorkflowStatus::Pending,
            created_at: Instant::now(),
        }
    }

    /// Create a new workflow (alias for workflow_id)
    pub fn new(id: String, name: String) -> Self {
        Self::workflow_id(id, name)
    }

    /// Add a task to the workflow
    pub fn add_task(&mut self, task: Task) {
        let taskid = task.id.clone();
        self.tasks.insert(taskid.clone(), task);
        self.execution_order.push(taskid);
    }

    /// Get tasks that are ready to execute (dependencies satisfied)
    pub fn get_ready_tasks(&self) -> Vec<&Task> {
        self.tasks
            .values()
            .filter(|task| {
                matches!(task.status, TaskStatus::Pending)
                    && self.are_dependencies_satisfied(&task.id)
            })
            .collect()
    }

    fn are_dependencies_satisfied(&self, taskid: &str) -> bool {
        if let Some(task) = self.tasks.get(taskid) {
            task.dependencies.iter().all(|dep_id| {
                if let Some(dep_task) = self.tasks.get(dep_id) {
                    matches!(dep_task.status, TaskStatus::Completed { .. })
                } else {
                    false
                }
            })
        } else {
            false
        }
    }

    /// Check if workflow is complete
    pub fn is_complete(&self) -> bool {
        self.tasks.values().all(|task| {
            matches!(
                task.status,
                TaskStatus::Completed { .. } | TaskStatus::Failed { .. } | TaskStatus::Cancelled
            )
        })
    }

    /// Check if workflow has failed
    pub fn has_failed(&self) -> bool {
        self.tasks
            .values()
            .any(|task| matches!(task.status, TaskStatus::Failed { .. }))
    }
}

/// Node information for orchestration
#[derive(Debug, Clone)]
pub struct OrchestratorNode {
    pub nodeid: String,
    pub address: SocketAddr,
    pub capacity: usize,
    pub current_load: usize,
    pub capabilities: Vec<String>,
    pub last_heartbeat: Instant,
}

impl OrchestratorNode {
    /// Create a new orchestrator node
    pub fn id(nodeid: String, address: SocketAddr, capacity: usize) -> Self {
        Self {
            nodeid,
            address,
            capacity,
            current_load: 0,
            capabilities: Vec::new(),
            last_heartbeat: Instant::now(),
        }
    }

    /// Create a new orchestrator node (alias for id)
    pub fn new(nodeid: String, address: SocketAddr, capacity: usize) -> Self {
        Self::id(nodeid, address, capacity)
    }

    /// Check if node can accept more tasks
    pub fn can_accept_task(&self) -> bool {
        self.current_load < self.capacity
    }

    /// Update node heartbeat
    pub fn update_heartbeat(&mut self) {
        self.last_heartbeat = Instant::now();
    }

    /// Check if node is responsive
    pub fn is_responsive(&self, timeout: Duration) -> bool {
        Instant::now().duration_since(self.last_heartbeat) <= timeout
    }
}

/// Orchestration engine for managing distributed workflows
#[derive(Debug)]
pub struct OrchestrationEngine {
    workflows: Arc<Mutex<HashMap<String, Workflow>>>,
    nodes: Arc<Mutex<HashMap<String, OrchestratorNode>>>,
    task_queue: Arc<Mutex<VecDeque<String>>>, // Task IDs
    running_tasks: Arc<Mutex<HashMap<String, (String, Instant)>>>, // Task ID -> (Node ID, Start time)
    node_timeout: Duration,
}

impl OrchestrationEngine {
    /// Create a new orchestration engine
    pub fn new() -> Self {
        Self {
            workflows: Arc::new(Mutex::new(HashMap::new())),
            nodes: Arc::new(Mutex::new(HashMap::new())),
            task_queue: Arc::new(Mutex::new(VecDeque::new())),
            running_tasks: Arc::new(Mutex::new(HashMap::new())),
            node_timeout: Duration::from_secs(60),
        }
    }

    /// Register a node with the orchestrator
    pub fn register_node(&self, node: OrchestratorNode) -> CoreResult<()> {
        let mut nodes = self.nodes.lock().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new(
                "Failed to acquire nodes lock".to_string(),
            ))
        })?;
        nodes.insert(node.nodeid.clone(), node);
        Ok(())
    }

    /// Submit a workflow for execution
    pub fn submit_workflow(&self, workflow: Workflow) -> CoreResult<()> {
        let workflow_id = workflow.id.clone();

        let mut workflows = self.workflows.lock().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new(
                "Failed to acquire workflows lock".to_string(),
            ))
        })?;

        let mut task_queue = self.task_queue.lock().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new(
                "Failed to acquire task queue lock".to_string(),
            ))
        })?;

        // Add ready tasks to the queue
        for task in workflow.get_ready_tasks() {
            task_queue.push_back(task.id.clone());
        }

        workflows.insert(workflow_id, workflow);
        Ok(())
    }

    /// Submit a single task for execution
    pub fn submit_task(&self, task: Task) -> CoreResult<()> {
        let taskid = task.id.clone();

        // Create a single-task workflow
        let mut workflow = Workflow::new(format!("workflow_{taskid}"), task.name.to_string());
        workflow.add_task(task);

        self.submit_workflow(workflow)
    }

    /// Process the task queue and assign tasks to available nodes
    pub fn process_task_queue(&self) -> CoreResult<()> {
        let mut task_queue = self.task_queue.lock().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new(
                "Failed to acquire task queue lock".to_string(),
            ))
        })?;

        let mut nodes = self.nodes.lock().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new(
                "Failed to acquire nodes lock".to_string(),
            ))
        })?;

        let mut workflows = self.workflows.lock().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new(
                "Failed to acquire workflows lock".to_string(),
            ))
        })?;

        let mut running_tasks = self.running_tasks.lock().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new(
                "Failed to acquire running tasks lock".to_string(),
            ))
        })?;

        // Process tasks in priority order
        let mut tasks_to_assign = Vec::new();
        while let Some(taskid) = task_queue.pop_front() {
            tasks_to_assign.push(taskid);
        }

        // Sort by priority
        tasks_to_assign.sort_by(|a, b| {
            let priority_a = self
                .find_task_priority(a, &workflows)
                .unwrap_or(TaskPriority::Low);
            let priority_b = self
                .find_task_priority(b, &workflows)
                .unwrap_or(TaskPriority::Low);
            priority_b.cmp(&priority_a) // Higher priority first
        });

        for taskid in tasks_to_assign {
            // Find an available node
            if let Some(available_node) = nodes
                .values_mut()
                .filter(|node| node.can_accept_task() && node.is_responsive(self.node_timeout))
                .min_by_key(|node| node.current_load)
            {
                // Assign task to node
                if let Some(task) = self.find_task_mut(&taskid, &mut workflows) {
                    task.status = TaskStatus::Running {
                        nodeid: available_node.nodeid.clone(),
                        started_at: Instant::now(),
                    };

                    available_node.current_load += 1;
                    running_tasks.insert(taskid, (available_node.nodeid.clone(), Instant::now()));
                } else {
                    // Task not found, put it back in queue
                    task_queue.push_back(taskid);
                }
            } else {
                // No available nodes, put task back in queue
                task_queue.push_back(taskid);
            }
        }

        Ok(())
    }

    fn find_task_priority(
        &self,
        taskid: &str,
        workflows: &HashMap<String, Workflow>,
    ) -> Option<TaskPriority> {
        for workflow in workflows.values() {
            if let Some(task) = workflow.tasks.get(taskid) {
                return Some(task.priority);
            }
        }
        None
    }

    fn find_task_mut<'a>(
        &self,
        taskid: &str,
        workflows: &'a mut HashMap<String, Workflow>,
    ) -> Option<&'a mut Task> {
        for workflow in workflows.values_mut() {
            if let Some(task) = workflow.tasks.get_mut(taskid) {
                return Some(task);
            }
        }
        None
    }

    /// Mark a task as completed
    pub fn complete_task(&mut self, taskid: &str) -> CoreResult<()> {
        let mut workflows = self.workflows.lock().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new(
                "Failed to acquire workflows lock".to_string(),
            ))
        })?;

        let mut nodes = self.nodes.lock().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new(
                "Failed to acquire nodes lock".to_string(),
            ))
        })?;

        let mut running_tasks = self.running_tasks.lock().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new(
                "Failed to acquire running tasks lock".to_string(),
            ))
        })?;

        // Get the nodeid from running tasks
        let nodeid = running_tasks
            .get(taskid)
            .map(|(nodeid, _)| nodeid.clone())
            .unwrap_or_else(|| "unknown".to_string());

        // Update task status
        if let Some(task) = self.find_task_mut(taskid, &mut workflows) {
            task.status = TaskStatus::Completed {
                nodeid: nodeid.clone(),
                completed_at: Instant::now(),
            };
        }

        // Update node load
        if let Some(node) = nodes.get_mut(&nodeid) {
            node.current_load = node.current_load.saturating_sub(1);
        }

        // Remove from running tasks
        running_tasks.remove(taskid);

        // Add newly ready tasks to queue
        self.queue_ready_tasks(&workflows)?;

        Ok(())
    }

    fn queue_ready_tasks(&self, workflows: &HashMap<String, Workflow>) -> CoreResult<()> {
        let mut task_queue = self.task_queue.lock().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new(
                "Failed to acquire task queue lock".to_string(),
            ))
        })?;

        for workflow in workflows.values() {
            for task in workflow.get_ready_tasks() {
                if !task_queue.iter().any(|id| id == &task.id) {
                    task_queue.push_back(task.id.clone());
                }
            }
        }

        Ok(())
    }

    /// Get orchestration statistics
    pub fn get_statistics(&self) -> CoreResult<OrchestrationStats> {
        let workflows = self.workflows.lock().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new(
                "Failed to acquire workflows lock".to_string(),
            ))
        })?;

        let nodes = self.nodes.lock().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new(
                "Failed to acquire nodes lock".to_string(),
            ))
        })?;

        let task_queue = self.task_queue.lock().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new(
                "Failed to acquire task queue lock".to_string(),
            ))
        })?;

        let running_tasks = self.running_tasks.lock().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new(
                "Failed to acquire running tasks lock".to_string(),
            ))
        })?;

        let total_workflows = workflows.len();
        let pending_workflows = workflows
            .values()
            .filter(|w| matches!(w.status, WorkflowStatus::Pending))
            .count();
        let running_workflows = workflows
            .values()
            .filter(|w| matches!(w.status, WorkflowStatus::Running))
            .count();
        let completed_workflows = workflows
            .values()
            .filter(|w| matches!(w.status, WorkflowStatus::Completed))
            .count();

        let total_tasks: usize = workflows.values().map(|w| w.tasks.len()).sum();
        let pending_tasks = task_queue.len();
        let running_tasks_count = running_tasks.len();

        let total_nodes = nodes.len();
        let active_nodes = nodes
            .values()
            .filter(|n| n.is_responsive(self.node_timeout))
            .count();
        let total_capacity: usize = nodes.values().map(|n| n.capacity).sum();
        let current_load: usize = nodes.values().map(|n| n.current_load).sum();

        Ok(OrchestrationStats {
            total_workflows,
            pending_workflows,
            running_workflows,
            completed_workflows,
            total_tasks,
            pending_tasks,
            running_tasks: running_tasks_count,
            total_nodes,
            active_nodes,
            total_capacity,
            current_load,
        })
    }
}

impl Default for OrchestrationEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Orchestration statistics
#[derive(Debug)]
pub struct OrchestrationStats {
    pub total_workflows: usize,
    pub pending_workflows: usize,
    pub running_workflows: usize,
    pub completed_workflows: usize,
    pub total_tasks: usize,
    pub pending_tasks: usize,
    pub running_tasks: usize,
    pub total_nodes: usize,
    pub active_nodes: usize,
    pub total_capacity: usize,
    pub current_load: usize,
}

impl OrchestrationStats {
    /// Calculate capacity utilization percentage
    pub fn capacity_utilization(&self) -> f64 {
        if self.total_capacity == 0 {
            0.0
        } else {
            (self.current_load as f64 / self.total_capacity as f64) * 100.0
        }
    }

    /// Calculate node availability percentage
    pub fn node_availability(&self) -> f64 {
        if self.total_nodes == 0 {
            0.0
        } else {
            (self.active_nodes as f64 / self.total_nodes as f64) * 100.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr};

    #[test]
    fn test_task_creation() {
        let task = Task::new("task1".to_string(), "Test Task".to_string(), vec![1, 2, 3])
            .with_priority(TaskPriority::High)
            .with_timeout(Duration::from_secs(60));

        assert_eq!(task.id, "task1");
        assert_eq!(task.priority, TaskPriority::High);
        assert_eq!(task.timeout, Some(Duration::from_secs(60)));
        assert!(task.can_retry());
    }

    #[test]
    fn test_workflow_creation() {
        let mut workflow = Workflow::new("wf1".to_string(), "Test Workflow".to_string());
        let task = Task::new("task1".to_string(), "Test Task".to_string(), vec![1, 2, 3]);

        workflow.add_task(task);
        assert_eq!(workflow.tasks.len(), 1);
        assert_eq!(workflow.execution_order.len(), 1);

        let ready_tasks = workflow.get_ready_tasks();
        assert_eq!(ready_tasks.len(), 1);
    }

    #[test]
    fn test_orchestrator_node() {
        let address = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);
        let mut node = OrchestratorNode::new("node1".to_string(), address, 10);

        assert!(node.can_accept_task());
        assert!(node.is_responsive(Duration::from_secs(30)));

        node.current_load = 10;
        assert!(!node.can_accept_task());
    }

    #[test]
    fn test_orchestration_engine() {
        let engine = OrchestrationEngine::new();

        let address = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);
        let node = OrchestratorNode::new("node1".to_string(), address, 5);

        assert!(engine.register_node(node).is_ok());

        let task = Task::new("task1".to_string(), "Test Task".to_string(), vec![1, 2, 3]);
        assert!(engine.submit_task(task).is_ok());

        let stats = engine.get_statistics().expect("Operation failed");
        assert_eq!(stats.total_nodes, 1);
        assert_eq!(stats.total_workflows, 1);
    }

    #[test]
    fn test_orchestration_stats() {
        let stats = OrchestrationStats {
            total_workflows: 10,
            pending_workflows: 2,
            running_workflows: 3,
            completed_workflows: 5,
            total_tasks: 50,
            pending_tasks: 10,
            running_tasks: 15,
            total_nodes: 5,
            active_nodes: 4,
            total_capacity: 100,
            current_load: 75,
        };

        assert_eq!(stats.capacity_utilization(), 75.0);
        assert_eq!(stats.node_availability(), 80.0);
    }
}
