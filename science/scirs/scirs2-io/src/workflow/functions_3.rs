//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{IoError, Result};
use crate::metadata::{Metadata, MetadataValue};
use chrono::{DateTime, Datelike, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use super::types::{ResourceRequirements, Task, TaskStatus, TaskType, Workflow, WorkflowExecutor};

/// Event-driven workflows
pub mod events {
    use super::*;
    use crossbeam_channel::{Receiver, Sender};
    /// Event types that can trigger workflows
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub enum WorkflowEvent {
        FileCreated {
            path: String,
        },
        FileModified {
            path: String,
        },
        DataAvailable {
            source: String,
            timestamp: DateTime<Utc>,
        },
        ScheduledTime {
            workflowid: String,
        },
        ExternalTrigger {
            source: String,
            payload: serde_json::Value,
        },
        WorkflowCompleted {
            workflowid: String,
            executionid: String,
        },
        Custom {
            event_type: String,
            data: serde_json::Value,
        },
    }
    /// Event-driven workflow executor
    pub struct EventDrivenExecutor {
        event_rx: Receiver<WorkflowEvent>,
        event_tx: Sender<WorkflowEvent>,
        rules: Vec<EventRule>,
        executor: Arc<WorkflowExecutor>,
    }
    #[derive(Debug, Clone)]
    pub struct EventRule {
        pub id: String,
        pub event_pattern: EventPattern,
        pub workflowid: String,
        pub parameters: HashMap<String, serde_json::Value>,
    }
    #[derive(Debug, Clone)]
    pub enum EventPattern {
        FilePattern {
            path_regex: String,
        },
        SourcePattern {
            source: String,
        },
        EventTypePattern {
            event_type: String,
        },
        CompositePattern {
            patterns: Vec<EventPattern>,
            operator: LogicalOperator,
        },
    }
    #[derive(Debug, Clone)]
    pub enum LogicalOperator {
        And,
        Or,
        Not,
    }
    impl EventDrivenExecutor {
        pub fn new(executor: Arc<WorkflowExecutor>) -> Self {
            let (tx, rx) = crossbeam_channel::unbounded();
            Self {
                event_rx: rx,
                event_tx: tx,
                rules: Vec::new(),
                executor,
            }
        }
        /// Register an event rule
        pub fn register_rule(&mut self, rule: EventRule) {
            self.rules.push(rule);
        }
        /// Get event sender for external systems
        pub fn get_event_sender(&self) -> Sender<WorkflowEvent> {
            self.event_tx.clone()
        }
        /// Process events and trigger workflows
        pub fn process_events(&self, workflows: &HashMap<String, Workflow>) -> Result<()> {
            while let Ok(event) = self.event_rx.try_recv() {
                for rule in &self.rules {
                    if self.matches_pattern(&event, &rule.event_pattern) {
                        if let Some(workflow) = workflows.get(&rule.workflowid) {
                            let mut workflow = workflow.clone();
                            workflow.metadata.set(
                                "trigger_event",
                                MetadataValue::String(
                                    serde_json::to_string(&event).expect("Operation failed"),
                                ),
                            );
                            self.executor.execute(&workflow)?;
                        }
                    }
                }
            }
            Ok(())
        }
        #[allow(clippy::only_used_in_recursion)]
        fn matches_pattern(&self, event: &WorkflowEvent, pattern: &EventPattern) -> bool {
            match pattern {
                EventPattern::FilePattern { path_regex } => {
                    if let WorkflowEvent::FileCreated { path }
                    | WorkflowEvent::FileModified { path } = event
                    {
                        regex::Regex::new(path_regex)
                            .map(|re| re.is_match(path))
                            .unwrap_or(false)
                    } else {
                        false
                    }
                }
                EventPattern::SourcePattern { source } => match event {
                    WorkflowEvent::DataAvailable { source: s, .. } => s == source,
                    WorkflowEvent::ExternalTrigger { source: s, .. } => s == source,
                    WorkflowEvent::FileCreated { .. } => false,
                    WorkflowEvent::FileModified { .. } => false,
                    WorkflowEvent::ScheduledTime { .. } => false,
                    WorkflowEvent::WorkflowCompleted { .. } => false,
                    WorkflowEvent::Custom { .. } => false,
                },
                EventPattern::EventTypePattern { event_type } => {
                    if let WorkflowEvent::Custom { event_type: t, .. } = event {
                        t == event_type
                    } else {
                        false
                    }
                }
                EventPattern::CompositePattern { patterns, operator } => match operator {
                    LogicalOperator::And => patterns.iter().all(|p| self.matches_pattern(event, p)),
                    LogicalOperator::Or => patterns.iter().any(|p| self.matches_pattern(event, p)),
                    LogicalOperator::Not => {
                        !patterns.iter().any(|p| self.matches_pattern(event, p))
                    }
                },
            }
        }
    }
}
/// Workflow versioning and history
pub mod versioning {
    use super::*;
    /// Workflow version control
    pub struct WorkflowVersionControl {
        versions: HashMap<String, Vec<WorkflowVersion>>,
    }
    #[derive(Debug, Clone)]
    pub struct WorkflowVersion {
        pub version: String,
        pub workflow: Workflow,
        pub created_at: DateTime<Utc>,
        pub created_by: String,
        pub change_description: String,
        pub parent_version: Option<String>,
    }
    impl Default for WorkflowVersionControl {
        fn default() -> Self {
            Self::new()
        }
    }
    impl WorkflowVersionControl {
        pub fn new() -> Self {
            Self {
                versions: HashMap::new(),
            }
        }
        /// Create a new version
        pub fn create_version(
            &mut self,
            workflow: Workflow,
            created_by: impl Into<String>,
            description: impl Into<String>,
        ) -> String {
            let workflowid = workflow.id.clone();
            let versions = self.versions.entry(workflowid.clone()).or_default();
            let version_number = versions.len() + 1;
            let version = format!("v{version_number}.0.0");
            let parent_version = versions.last().map(|v| v.version.clone());
            versions.push(WorkflowVersion {
                version: version.clone(),
                workflow,
                created_at: Utc::now(),
                created_by: created_by.into(),
                change_description: description.into(),
                parent_version,
            });
            version
        }
        /// Get a specific version
        pub fn get_version(&self, workflowid: &str, version: &str) -> Option<&WorkflowVersion> {
            self.versions
                .get(workflowid)?
                .iter()
                .find(|v| v.version == version)
        }
        /// Get latest version
        pub fn get_latest(&self, workflowid: &str) -> Option<&WorkflowVersion> {
            self.versions.get(workflowid)?.last()
        }
        /// Get version history
        pub fn get_history(&self, workflowid: &str) -> Vec<&WorkflowVersion> {
            self.versions
                .get(workflowid)
                .map(|v| v.iter().collect())
                .unwrap_or_default()
        }
        /// Diff two versions
        pub fn diff(
            &self,
            workflowid: &str,
            version1: &str,
            version2: &str,
        ) -> Option<WorkflowDiff> {
            let v1 = self.get_version(workflowid, version1)?;
            let v2 = self.get_version(workflowid, version2)?;
            Some(WorkflowDiff {
                version1: version1.to_string(),
                version2: version2.to_string(),
                added_tasks: self.diff_tasks(&v1.workflow.tasks, &v2.workflow.tasks, true),
                removed_tasks: self.diff_tasks(&v1.workflow.tasks, &v2.workflow.tasks, false),
                modified_tasks: self.find_modified_tasks(&v1.workflow.tasks, &v2.workflow.tasks),
                dependency_changes: self
                    .diff_dependencies(&v1.workflow.dependencies, &v2.workflow.dependencies),
            })
        }
        fn diff_tasks(&self, tasks1: &[Task], tasks2: &[Task], added: bool) -> Vec<String> {
            let set1: HashSet<_> = tasks1.iter().map(|t| &t.id).collect();
            let set2: HashSet<_> = tasks2.iter().map(|t| &t.id).collect();
            if added {
                set2.difference(&set1).map(|id| (*id).clone()).collect()
            } else {
                set1.difference(&set2).map(|id| (*id).clone()).collect()
            }
        }
        fn find_modified_tasks(&self, tasks1: &[Task], tasks2: &[Task]) -> Vec<String> {
            let map1: HashMap<&String, &Task> = tasks1.iter().map(|t| (&t.id, t)).collect();
            let map2: HashMap<&String, &Task> = tasks2.iter().map(|t| (&t.id, t)).collect();
            let mut modified = Vec::new();
            for (id, task1) in map1 {
                if let Some(task2) = map2.get(id) {
                    if task1.name != task2.name || task1.config != task2.config {
                        modified.push(id.clone());
                    }
                }
            }
            modified
        }
        fn diff_dependencies(
            &self,
            deps1: &HashMap<String, Vec<String>>,
            deps2: &HashMap<String, Vec<String>>,
        ) -> Vec<DependencyChange> {
            let mut changes = Vec::new();
            let all_tasks: HashSet<_> = deps1.keys().chain(deps2.keys()).collect();
            for task in all_tasks {
                let deps1_set: HashSet<_> = deps1
                    .get(task)
                    .map(|d| d.iter().collect())
                    .unwrap_or_default();
                let deps2_set: HashSet<_> = deps2
                    .get(task)
                    .map(|d| d.iter().collect())
                    .unwrap_or_default();
                for added in deps2_set.difference(&deps1_set) {
                    changes.push(DependencyChange::Added {
                        task: (*task).clone(),
                        dependency: (*added).clone(),
                    });
                }
                for removed in deps1_set.difference(&deps2_set) {
                    changes.push(DependencyChange::Removed {
                        task: (*task).clone(),
                        dependency: (*removed).clone(),
                    });
                }
            }
            changes
        }
    }
    #[derive(Debug)]
    pub struct WorkflowDiff {
        pub version1: String,
        pub version2: String,
        pub added_tasks: Vec<String>,
        pub removed_tasks: Vec<String>,
        pub modified_tasks: Vec<String>,
        pub dependency_changes: Vec<DependencyChange>,
    }
    /// Dependency change event
    #[derive(Debug)]
    pub enum DependencyChange {
        /// Dependency was added
        Added {
            /// Task ID
            task: String,
            /// Dependency task ID
            dependency: String,
        },
        /// Dependency was removed
        Removed {
            /// Task ID
            task: String,
            /// Dependency task ID
            dependency: String,
        },
    }
}
/// Distributed execution support
pub mod distributed {
    use super::*;
    /// Distributed workflow executor
    pub struct DistributedExecutor {
        coordinator_url: String,
        worker_pool: WorkerPool,
        task_queue: Arc<Mutex<Vec<DistributedTask>>>,
    }
    /// Task in a distributed workflow
    #[derive(Debug, Clone)]
    pub struct DistributedTask {
        /// Task definition
        pub task: Task,
        /// Workflow identifier
        pub workflowid: String,
        /// Execution identifier
        pub executionid: String,
        /// Worker ID this task is assigned to
        pub assigned_worker: Option<String>,
        /// Current task status
        pub status: TaskStatus,
    }
    /// Pool of worker nodes
    pub struct WorkerPool {
        workers: Vec<WorkerNode>,
    }
    /// Worker node in the distributed system
    #[derive(Debug, Clone)]
    pub struct WorkerNode {
        /// Unique worker identifier
        pub id: String,
        /// Worker node URL
        pub url: String,
        /// Worker capabilities and resources
        pub capabilities: WorkerCapabilities,
        /// Current load (0.0-1.0)
        pub current_load: f64,
        /// Current worker status
        pub status: WorkerStatus,
    }
    /// Worker node capabilities and resources
    #[derive(Debug, Clone)]
    pub struct WorkerCapabilities {
        /// Number of CPU cores available
        pub cpu_cores: usize,
        /// Memory available in GB
        pub memorygb: f64,
        /// Whether GPU is available
        pub gpu_available: bool,
        /// Task types this worker can execute
        pub supported_task_types: Vec<TaskType>,
    }
    /// Status of a worker node
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum WorkerStatus {
        /// Worker is available for new tasks
        Available,
        /// Worker is currently busy
        Busy,
        /// Worker is offline
        Offline,
    }
    impl DistributedExecutor {
        /// Create a new distributed executor
        pub fn new(coordinator_url: impl Into<String>) -> Self {
            Self {
                coordinator_url: coordinator_url.into(),
                worker_pool: WorkerPool {
                    workers: Vec::new(),
                },
                task_queue: Arc::new(Mutex::new(Vec::new())),
            }
        }
        /// Register a worker node
        pub fn register_worker(&mut self, worker: WorkerNode) {
            self.worker_pool.workers.push(worker);
        }
        /// Schedule task to appropriate worker
        pub fn schedule_task(&self, task: DistributedTask) -> Result<String> {
            let worker = self.find_suitable_worker(&task)?;
            let mut queue = self.task_queue.lock().expect("Operation failed");
            let mut scheduled_task = task;
            scheduled_task.assigned_worker = Some(worker.id.clone());
            queue.push(scheduled_task);
            Ok(worker.id.clone())
        }
        fn find_suitable_worker(&self, task: &DistributedTask) -> Result<&WorkerNode> {
            let suitable_workers: Vec<_> = self
                .worker_pool
                .workers
                .iter()
                .filter(|w| {
                    w.status == WorkerStatus::Available
                        && w.capabilities
                            .supported_task_types
                            .contains(&task.task.task_type)
                        && self.meets_resource_requirements(w, &task.task.resources)
                })
                .collect();
            suitable_workers
                .into_iter()
                .min_by(|a, b| {
                    a.current_load
                        .partial_cmp(&b.current_load)
                        .expect("Operation failed")
                })
                .ok_or_else(|| IoError::Other("No suitable worker available".to_string()))
        }
        fn meets_resource_requirements(
            &self,
            worker: &WorkerNode,
            requirements: &ResourceRequirements,
        ) -> bool {
            if let Some(cpu) = requirements.cpu_cores {
                if worker.capabilities.cpu_cores < cpu {
                    return false;
                }
            }
            if let Some(memory) = requirements.memorygb {
                if worker.capabilities.memorygb < memory {
                    return false;
                }
            }
            if requirements.gpu.is_some() && !worker.capabilities.gpu_available {
                return false;
            }
            true
        }
    }
}
