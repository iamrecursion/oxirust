//! Distributed task scheduler
//!
//! This module provides comprehensive task scheduling capabilities for
//! distributed computing, including priority-based scheduling, load
//! balancing, and fault-tolerant task execution coordination.

use super::cluster::{ComputeCapacity, DistributedTask, NodeInfo, ResourceRequirements, TaskId};
use crate::error::{CoreError, CoreResult, ErrorContext};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

/// Global distributed scheduler instance
static GLOBAL_SCHEDULER: std::sync::OnceLock<Arc<DistributedScheduler>> =
    std::sync::OnceLock::new();

/// Comprehensive distributed task scheduler
#[derive(Debug)]
pub struct DistributedScheduler {
    task_queue: Arc<Mutex<TaskQueue>>,
    execution_tracker: Arc<RwLock<ExecutionTracker>>,
    schedulingpolicies: Arc<RwLock<SchedulingPolicies>>,
    load_balancer: Arc<RwLock<LoadBalancer>>,
}

impl DistributedScheduler {
    /// Create new distributed scheduler
    pub fn new() -> CoreResult<Self> {
        Ok(Self {
            task_queue: Arc::new(Mutex::new(TaskQueue::new())),
            execution_tracker: Arc::new(RwLock::new(ExecutionTracker::new())),
            schedulingpolicies: Arc::new(RwLock::new(SchedulingPolicies::default())),
            load_balancer: Arc::new(RwLock::new(LoadBalancer::new())),
        })
    }

    /// Get global scheduler instance
    pub fn global() -> CoreResult<Arc<Self>> {
        Ok(GLOBAL_SCHEDULER
            .get_or_init(|| Arc::new(Self::new().expect("Operation failed")))
            .clone())
    }

    /// Submit task to scheduler
    pub fn submit_task(&self, task: DistributedTask) -> CoreResult<TaskId> {
        let mut queue = self.task_queue.lock().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new("Failed to acquire task queue lock"))
        })?;

        let taskid = task.taskid.clone();
        queue.enqueue(task)?;

        Ok(taskid)
    }

    /// Get pending task count
    pub fn get_pending_task_count(&self) -> CoreResult<usize> {
        let queue = self.task_queue.lock().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new("Failed to acquire task queue lock"))
        })?;

        Ok(queue.size())
    }

    /// Schedule next batch of tasks
    pub fn schedule_next(&self, availablenodes: &[NodeInfo]) -> CoreResult<Vec<TaskAssignment>> {
        let mut queue = self.task_queue.lock().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new("Failed to acquire task queue lock"))
        })?;

        let policies = self.schedulingpolicies.read().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new("Failed to acquire policies lock"))
        })?;

        let mut load_balancer = self.load_balancer.write().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new("Failed to acquire load balancer lock"))
        })?;

        // Schedule tasks based on policy
        let assignments = match policies.scheduling_algorithm {
            SchedulingAlgorithm::FirstComeFirstServe => {
                self.schedule_fcfs(&mut queue, availablenodes, &mut load_balancer)?
            }
            SchedulingAlgorithm::PriorityBased => {
                self.schedule_priority(&mut queue, availablenodes, &mut load_balancer)?
            }
            SchedulingAlgorithm::LoadBalanced => {
                self.schedule_load_balanced(&mut queue, availablenodes, &mut load_balancer)?
            }
            SchedulingAlgorithm::ResourceAware => {
                self.schedule_resource_aware(&mut queue, availablenodes, &mut load_balancer)?
            }
        };

        // Update execution tracker
        let mut tracker = self.execution_tracker.write().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new(
                "Failed to acquire execution tracker lock",
            ))
        })?;

        for assignment in &assignments {
            tracker.track_assignment(assignment.clone())?;
        }

        Ok(assignments)
    }

    fn schedule_fcfs(
        &self,
        queue: &mut TaskQueue,
        availablenodes: &[NodeInfo],
        load_balancer: &mut LoadBalancer,
    ) -> CoreResult<Vec<TaskAssignment>> {
        let mut assignments = Vec::new();

        while let Some(task) = queue.dequeue_next() {
            if let Some(node) = load_balancer.select_node_for_task(&task, availablenodes)? {
                assignments.push(TaskAssignment {
                    taskid: task.taskid.clone(),
                    nodeid: node.id.clone(),
                    assigned_at: Instant::now(),
                    estimated_duration: self.estimate_task_duration(&task, &node)?,
                });

                if assignments.len() >= 10 {
                    // Batch size limit
                    break;
                }
            } else {
                // No suitable node available, put task back
                queue.enqueue(task)?;
                break;
            }
        }

        Ok(assignments)
    }

    fn schedule_priority(
        &self,
        queue: &mut TaskQueue,
        availablenodes: &[NodeInfo],
        load_balancer: &mut LoadBalancer,
    ) -> CoreResult<Vec<TaskAssignment>> {
        let mut assignments = Vec::new();
        let mut scheduled_tasks = Vec::new();

        // Get tasks ordered by priority
        while let Some(task) = queue.dequeue_highest_priority() {
            if let Some(node) = load_balancer.select_node_for_task(&task, availablenodes)? {
                assignments.push(TaskAssignment {
                    taskid: task.taskid.clone(),
                    nodeid: node.id.clone(),
                    assigned_at: Instant::now(),
                    estimated_duration: self.estimate_task_duration(&task, &node)?,
                });

                if assignments.len() >= 10 {
                    // Batch size limit
                    break;
                }
            } else {
                // No suitable node available, save for later
                scheduled_tasks.push(task);
            }
        }

        // Put unscheduled tasks back
        for task in scheduled_tasks {
            queue.enqueue(task)?;
        }

        Ok(assignments)
    }

    fn schedule_load_balanced(
        &self,
        queue: &mut TaskQueue,
        availablenodes: &[NodeInfo],
        load_balancer: &mut LoadBalancer,
    ) -> CoreResult<Vec<TaskAssignment>> {
        let mut assignments = Vec::new();

        // Update load balancer with current node loads
        load_balancer.update_nodeloads(availablenodes)?;

        while let Some(task) = queue.dequeue_next() {
            if let Some(node) = load_balancer.select_least_loaded_node(&task, availablenodes)? {
                assignments.push(TaskAssignment {
                    taskid: task.taskid.clone(),
                    nodeid: node.id.clone(),
                    assigned_at: Instant::now(),
                    estimated_duration: self.estimate_task_duration(&task, &node)?,
                });

                // Update load balancer with new assignment
                load_balancer.record_assignment(&node.id, &task)?;

                if assignments.len() >= 10 {
                    // Batch size limit
                    break;
                }
            } else {
                queue.enqueue(task)?;
                break;
            }
        }

        Ok(assignments)
    }

    fn schedule_resource_aware(
        &self,
        queue: &mut TaskQueue,
        availablenodes: &[NodeInfo],
        load_balancer: &mut LoadBalancer,
    ) -> CoreResult<Vec<TaskAssignment>> {
        let mut assignments = Vec::new();

        while let Some(task) = queue.dequeue_next() {
            if let Some(node) = load_balancer.select_best_fit_node(&task, availablenodes)? {
                assignments.push(TaskAssignment {
                    taskid: task.taskid.clone(),
                    nodeid: node.id.clone(),
                    assigned_at: Instant::now(),
                    estimated_duration: self.estimate_task_duration(&task, &node)?,
                });

                if assignments.len() >= 10 {
                    // Batch size limit
                    break;
                }
            } else {
                queue.enqueue(task)?;
                break;
            }
        }

        Ok(assignments)
    }

    fn estimate_task_duration(
        &self,
        task: &DistributedTask,
        node: &NodeInfo,
    ) -> CoreResult<Duration> {
        // Simple estimation based on task requirements and node capabilities
        let cpu_factor =
            task.resource_requirements.cpu_cores as f64 / node.capabilities.cpu_cores as f64;
        let memory_factor =
            task.resource_requirements.memory_gb as f64 / node.capabilities.memory_gb as f64;

        let complexity_factor = cpu_factor.max(memory_factor);
        let base_duration = Duration::from_secs(60); // 1 minute base

        Ok(Duration::from_secs(
            (base_duration.as_secs() as f64 * complexity_factor) as u64,
        ))
    }
}

/// Task queue with priority support
#[derive(Debug)]
pub struct TaskQueue {
    priority_queue: BinaryHeap<PriorityTask>,
    fifo_queue: VecDeque<DistributedTask>,
    task_count: usize,
}

impl Default for TaskQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl TaskQueue {
    pub fn new() -> Self {
        Self {
            priority_queue: BinaryHeap::new(),
            fifo_queue: VecDeque::new(),
            task_count: 0,
        }
    }

    pub fn enqueue(&mut self, task: DistributedTask) -> CoreResult<()> {
        match task.priority {
            super::cluster::TaskPriority::Low | super::cluster::TaskPriority::Normal => {
                self.fifo_queue.push_back(task);
            }
            super::cluster::TaskPriority::High | super::cluster::TaskPriority::Critical => {
                self.priority_queue.push(PriorityTask {
                    task,
                    submitted_at: Instant::now(),
                });
            }
        }

        self.task_count += 1;
        Ok(())
    }

    pub fn dequeue_next(&mut self) -> Option<DistributedTask> {
        // Prefer high priority tasks
        if let Some(priority_task) = self.priority_queue.pop() {
            self.task_count -= 1;
            return Some(priority_task.task);
        }

        // Then FIFO tasks
        if let Some(task) = self.fifo_queue.pop_front() {
            self.task_count -= 1;
            return Some(task);
        }

        None
    }

    pub fn dequeue_highest_priority(&mut self) -> Option<DistributedTask> {
        if let Some(priority_task) = self.priority_queue.pop() {
            self.task_count -= 1;
            Some(priority_task.task)
        } else {
            None
        }
    }

    pub fn size(&self) -> usize {
        self.task_count
    }
}

/// Task wrapper for priority queue
#[derive(Debug)]
struct PriorityTask {
    task: DistributedTask,
    submitted_at: Instant,
}

impl PartialEq for PriorityTask {
    fn eq(&self, other: &Self) -> bool {
        self.task.priority == other.task.priority
    }
}

impl Eq for PriorityTask {}

impl PartialOrd for PriorityTask {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PriorityTask {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher priority first, then FIFO for same priority
        match self.task.priority.cmp(&other.task.priority) {
            Ordering::Equal => other.submitted_at.cmp(&self.submitted_at), // FIFO for same priority
            other => other,                                                // Higher priority first
        }
    }
}

/// Execution tracking and monitoring
#[derive(Debug)]
pub struct ExecutionTracker {
    active_assignments: HashMap<TaskId, TaskAssignment>,
    completed_tasks: VecDeque<CompletedTask>,
    failed_tasks: VecDeque<FailedTask>,
}

impl Default for ExecutionTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl ExecutionTracker {
    pub fn new() -> Self {
        Self {
            active_assignments: HashMap::new(),
            completed_tasks: VecDeque::with_capacity(1000),
            failed_tasks: VecDeque::with_capacity(1000),
        }
    }

    pub fn track_assignment(&mut self, assignment: TaskAssignment) -> CoreResult<()> {
        self.active_assignments
            .insert(assignment.taskid.clone(), assignment);
        Ok(())
    }

    pub fn mark_task_complete(
        &mut self,
        taskid: &TaskId,
        execution_time: Duration,
    ) -> CoreResult<()> {
        if let Some(assignment) = self.active_assignments.remove(taskid) {
            let completed_task = CompletedTask {
                taskid: taskid.clone(),
                nodeid: assignment.nodeid,
                execution_time,
                completed_at: Instant::now(),
            };

            self.completed_tasks.push_back(completed_task);

            // Maintain size limit
            while self.completed_tasks.len() > 1000 {
                self.completed_tasks.pop_front();
            }
        }

        Ok(())
    }

    pub fn mark_task_failed(&mut self, taskid: &TaskId, error: String) -> CoreResult<()> {
        if let Some(assignment) = self.active_assignments.remove(taskid) {
            let failed_task = FailedTask {
                taskid: taskid.clone(),
                nodeid: assignment.nodeid,
                error,
                failed_at: Instant::now(),
            };

            self.failed_tasks.push_back(failed_task);

            // Maintain size limit
            while self.failed_tasks.len() > 1000 {
                self.failed_tasks.pop_front();
            }
        }

        Ok(())
    }

    pub fn get_active_count(&self) -> usize {
        self.active_assignments.len()
    }
}

/// Load balancing for task distribution
#[derive(Debug)]
pub struct LoadBalancer {
    nodeloads: HashMap<String, NodeLoad>,
    balancing_strategy: LoadBalancingStrategy,
}

impl Default for LoadBalancer {
    fn default() -> Self {
        Self::new()
    }
}

impl LoadBalancer {
    pub fn new() -> Self {
        Self {
            nodeloads: HashMap::new(),
            balancing_strategy: LoadBalancingStrategy::LeastLoaded,
        }
    }

    pub fn select_node_for_task(
        &self,
        task: &DistributedTask,
        nodes: &[NodeInfo],
    ) -> CoreResult<Option<NodeInfo>> {
        match self.balancing_strategy {
            LoadBalancingStrategy::RoundRobin => self.select_round_robin(nodes),
            LoadBalancingStrategy::LeastLoaded => self.select_least_loaded(task, nodes),
            LoadBalancingStrategy::ResourceBased => self.select_resourcebased(task, nodes),
        }
    }

    pub fn select_least_loaded_node(
        &self,
        task: &DistributedTask,
        nodes: &[NodeInfo],
    ) -> CoreResult<Option<NodeInfo>> {
        self.select_least_loaded(task, nodes)
    }

    pub fn select_best_fit_node(
        &self,
        task: &DistributedTask,
        nodes: &[NodeInfo],
    ) -> CoreResult<Option<NodeInfo>> {
        self.select_resourcebased(task, nodes)
    }

    fn select_round_robin(&self, nodes: &[NodeInfo]) -> CoreResult<Option<NodeInfo>> {
        if nodes.is_empty() {
            return Ok(None);
        }

        // Simple round-robin selection
        let index = self.nodeloads.len() % nodes.len();
        Ok(Some(nodes[index].clone()))
    }

    fn select_least_loaded(
        &self,
        _task: &DistributedTask,
        nodes: &[NodeInfo],
    ) -> CoreResult<Option<NodeInfo>> {
        if nodes.is_empty() {
            return Ok(None);
        }

        // Select node with lowest current load
        let least_loaded = nodes.iter().min_by_key(|node| {
            self.nodeloads
                .get(&node.id)
                .map(|load| load.current_tasks)
                .unwrap_or(0)
        });

        Ok(least_loaded.cloned())
    }

    fn select_resourcebased(
        &self,
        task: &DistributedTask,
        nodes: &[NodeInfo],
    ) -> CoreResult<Option<NodeInfo>> {
        if nodes.is_empty() {
            return Ok(None);
        }

        // Select node that best fits the resource requirements
        let best_fit = nodes
            .iter()
            .filter(|node| self.can_satisfy_requirements(node, &task.resource_requirements))
            .min_by_key(|node| self.calculate_resource_waste(node, &task.resource_requirements));

        Ok(best_fit.cloned())
    }

    fn can_satisfy_requirements(
        &self,
        node: &NodeInfo,
        requirements: &ResourceRequirements,
    ) -> bool {
        let available = self.available_capacity(&node.id, &node.capabilities);

        available.cpu_cores >= requirements.cpu_cores
            && available.memory_gb >= requirements.memory_gb
            && available.gpu_count >= requirements.gpu_count
            && available.disk_space_gb >= requirements.disk_space_gb
    }

    fn calculate_resource_waste(
        &self,
        node: &NodeInfo,
        requirements: &ResourceRequirements,
    ) -> usize {
        let available = self.available_capacity(&node.id, &node.capabilities);

        let cpu_waste = available.cpu_cores.saturating_sub(requirements.cpu_cores);
        let memory_waste = available.memory_gb.saturating_sub(requirements.memory_gb);
        let gpu_waste = available.gpu_count.saturating_sub(requirements.gpu_count);
        let disk_waste = available
            .disk_space_gb
            .saturating_sub(requirements.disk_space_gb);

        cpu_waste + memory_waste + gpu_waste + disk_waste / 10 // Scale disk waste
    }

    fn available_capacity(
        &self,
        nodeid: &str,
        total_capacity: &super::cluster::NodeCapabilities,
    ) -> ComputeCapacity {
        let used = self
            .nodeloads
            .get(nodeid)
            .map(|load| &load.used_capacity)
            .cloned()
            .unwrap_or_default();

        ComputeCapacity {
            cpu_cores: total_capacity.cpu_cores.saturating_sub(used.cpu_cores),
            memory_gb: total_capacity.memory_gb.saturating_sub(used.memory_gb),
            gpu_count: total_capacity.gpu_count.saturating_sub(used.gpu_count),
            disk_space_gb: total_capacity
                .disk_space_gb
                .saturating_sub(used.disk_space_gb),
        }
    }

    pub fn update_nodeloads(&mut self, nodes: &[NodeInfo]) -> CoreResult<()> {
        // Initialize or update node loads
        for node in nodes {
            self.nodeloads
                .entry(node.id.clone())
                .or_insert_with(|| NodeLoad {
                    nodeid: node.id.clone(),
                    current_tasks: 0,
                    used_capacity: ComputeCapacity::default(),
                    last_updated: Instant::now(),
                });
        }

        Ok(())
    }

    pub fn record_assignment(&mut self, nodeid: &str, task: &DistributedTask) -> CoreResult<()> {
        if let Some(load) = self.nodeloads.get_mut(nodeid) {
            load.current_tasks += 1;
            load.used_capacity.cpu_cores += task.resource_requirements.cpu_cores;
            load.used_capacity.memory_gb += task.resource_requirements.memory_gb;
            load.used_capacity.gpu_count += task.resource_requirements.gpu_count;
            load.used_capacity.disk_space_gb += task.resource_requirements.disk_space_gb;
            load.last_updated = Instant::now();
        }

        Ok(())
    }
}

/// Node load tracking
#[derive(Debug, Clone)]
pub struct NodeLoad {
    pub nodeid: String,
    pub current_tasks: usize,
    pub used_capacity: ComputeCapacity,
    pub last_updated: Instant,
}

/// Scheduling policies and configuration
#[derive(Debug, Clone)]
pub struct SchedulingPolicies {
    pub scheduling_algorithm: SchedulingAlgorithm,
    pub load_balancing_strategy: LoadBalancingStrategy,
    pub batch_size: usize,
    pub scheduling_interval: Duration,
    pub priority_boost_threshold: Duration,
}

impl Default for SchedulingPolicies {
    fn default() -> Self {
        Self {
            scheduling_algorithm: SchedulingAlgorithm::PriorityBased,
            load_balancing_strategy: LoadBalancingStrategy::LeastLoaded,
            batch_size: 10,
            scheduling_interval: Duration::from_secs(5),
            priority_boost_threshold: Duration::from_secs(300), // 5 minutes
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulingAlgorithm {
    FirstComeFirstServe,
    PriorityBased,
    LoadBalanced,
    ResourceAware,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadBalancingStrategy {
    RoundRobin,
    LeastLoaded,
    ResourceBased,
}

/// Task assignment result
#[derive(Debug, Clone)]
pub struct TaskAssignment {
    pub taskid: TaskId,
    pub nodeid: String,
    pub assigned_at: Instant,
    pub estimated_duration: Duration,
}

/// Completed task record
#[derive(Debug, Clone)]
pub struct CompletedTask {
    pub taskid: TaskId,
    pub nodeid: String,
    pub execution_time: Duration,
    pub completed_at: Instant,
}

/// Failed task record
#[derive(Debug, Clone)]
pub struct FailedTask {
    pub taskid: TaskId,
    pub nodeid: String,
    pub error: String,
    pub failed_at: Instant,
}

/// Initialize distributed scheduler
#[allow(dead_code)]
pub fn initialize_distributed_scheduler() -> CoreResult<()> {
    let _scheduler = DistributedScheduler::global()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::{
        BackoffStrategy, ClusterTaskPriority, NodeCapabilities, NodeMetadata, NodeStatus, NodeType,
        RetryPolicy, TaskParameters, TaskType,
    };
    use std::net::{IpAddr, Ipv4Addr, SocketAddr};

    #[test]
    fn test_scheduler_creation() {
        let _scheduler = DistributedScheduler::new().expect("Operation failed");
        // Basic functionality test
    }

    #[test]
    fn test_task_queue() {
        let mut queue = TaskQueue::new();
        assert_eq!(queue.size(), 0);

        let task = create_test_task(ClusterTaskPriority::Normal);
        queue.enqueue(task).expect("Operation failed");
        assert_eq!(queue.size(), 1);

        let dequeued = queue.dequeue_next();
        assert!(dequeued.is_some());
        assert_eq!(queue.size(), 0);
    }

    #[test]
    fn test_priority_scheduling() {
        let mut queue = TaskQueue::new();

        // Add tasks with different priorities
        let low_task = create_test_task(ClusterTaskPriority::Low);
        let high_task = create_test_task(ClusterTaskPriority::High);

        queue.enqueue(low_task).expect("Operation failed");
        queue.enqueue(high_task).expect("Operation failed");

        // High priority task should come first
        let first = queue.dequeue_next().expect("Operation failed");
        assert_eq!(first.priority, ClusterTaskPriority::High);

        let second = queue.dequeue_next().expect("Operation failed");
        assert_eq!(second.priority, ClusterTaskPriority::Low);
    }

    #[test]
    fn test_load_balancer() {
        let balancer = LoadBalancer::new();
        let nodes = vec![create_test_node("node1"), create_test_node("node2")];
        let task = create_test_task(ClusterTaskPriority::Normal);

        let selected = balancer
            .select_node_for_task(&task, &nodes)
            .expect("Operation failed");
        assert!(selected.is_some());
    }

    fn create_test_task(priority: ClusterTaskPriority) -> DistributedTask {
        DistributedTask {
            taskid: TaskId::generate(),
            task_type: TaskType::Computation,
            resource_requirements: ResourceRequirements {
                cpu_cores: 2,
                memory_gb: 4,
                gpu_count: 0,
                disk_space_gb: 10,
                specialized_requirements: Vec::new(),
            },
            data_dependencies: Vec::new(),
            execution_parameters: TaskParameters {
                environment_variables: HashMap::new(),
                command_arguments: Vec::new(),
                timeout: None,
                retrypolicy: RetryPolicy {
                    max_attempts: 3,
                    backoff_strategy: BackoffStrategy::Fixed(Duration::from_secs(1)),
                },
            },
            priority,
        }
    }

    fn create_test_node(id: &str) -> NodeInfo {
        NodeInfo {
            id: id.to_string(),
            address: SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080),
            node_type: NodeType::Worker,
            capabilities: NodeCapabilities {
                cpu_cores: 8,
                memory_gb: 16,
                gpu_count: 1,
                disk_space_gb: 100,
                networkbandwidth_gbps: 1.0,
                specialized_units: Vec::new(),
            },
            status: NodeStatus::Healthy,
            last_seen: Instant::now(),
            metadata: NodeMetadata::default(),
        }
    }
}
