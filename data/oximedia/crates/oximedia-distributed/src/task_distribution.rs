//! Intelligent task distribution across worker nodes.
//!
//! Provides affinity scoring, workload balancing, and task migration planning
//! for the distributed encoding cluster.

#![allow(dead_code)]

use crate::consensus::NodeId;

/// Describes the capabilities of a worker node.
#[derive(Debug, Clone)]
pub struct WorkerCapability {
    /// Number of CPU cores available.
    pub cpu_cores: u32,
    /// RAM in gigabytes.
    pub memory_gb: f32,
    /// GPU VRAM in gigabytes (0.0 if no GPU).
    pub gpu_vram_gb: f32,
    /// Network bandwidth in Mbps.
    pub network_mbps: u32,
    /// Arbitrary string tags (e.g., "av1", "gpu", "high-mem").
    pub tags: Vec<String>,
}

impl WorkerCapability {
    /// Create a new capability descriptor.
    #[must_use]
    pub fn new(
        cpu_cores: u32,
        memory_gb: f32,
        gpu_vram_gb: f32,
        network_mbps: u32,
        tags: Vec<String>,
    ) -> Self {
        Self {
            cpu_cores,
            memory_gb,
            gpu_vram_gb,
            network_mbps,
            tags,
        }
    }
}

/// Describes the resource requirements of a task.
#[derive(Debug, Clone)]
pub struct TaskRequirements {
    /// Minimum CPU cores needed.
    pub min_cpu_cores: u32,
    /// Minimum RAM in gigabytes needed.
    pub min_memory_gb: f32,
    /// Whether a GPU is required.
    pub requires_gpu: bool,
    /// Minimum GPU VRAM in gigabytes needed (only relevant if `requires_gpu`).
    pub min_gpu_vram_gb: f32,
    /// Tags that are preferred (not mandatory) on the worker.
    pub preferred_tags: Vec<String>,
}

impl TaskRequirements {
    /// Create a new task requirements descriptor.
    #[must_use]
    pub fn new(
        min_cpu_cores: u32,
        min_memory_gb: f32,
        requires_gpu: bool,
        min_gpu_vram_gb: f32,
        preferred_tags: Vec<String>,
    ) -> Self {
        Self {
            min_cpu_cores,
            min_memory_gb,
            requires_gpu,
            min_gpu_vram_gb,
            preferred_tags,
        }
    }
}

/// Computes affinity scores between workers and tasks.
pub struct AffinityScore;

impl AffinityScore {
    /// Compute a 0.0–1.0 affinity score.
    ///
    /// Scoring:
    /// - Returns 0.0 if mandatory minimums are not met.
    /// - Base score of 0.5 when all minimums are met.
    /// - +0.1 for each preferred tag match (capped so total <= 1.0).
    /// - +0.3 if GPU is required and available with sufficient VRAM.
    #[must_use]
    pub fn compute(capability: &WorkerCapability, requirements: &TaskRequirements) -> f32 {
        // Check mandatory minimums
        if capability.cpu_cores < requirements.min_cpu_cores {
            return 0.0;
        }
        if capability.memory_gb < requirements.min_memory_gb {
            return 0.0;
        }
        if requirements.requires_gpu && capability.gpu_vram_gb < requirements.min_gpu_vram_gb {
            return 0.0;
        }

        let mut score = 0.5_f32;

        // GPU bonus
        if requirements.requires_gpu && capability.gpu_vram_gb >= requirements.min_gpu_vram_gb {
            score += 0.3;
        }

        // Tag bonus: +0.1 per preferred tag found on the worker
        for tag in &requirements.preferred_tags {
            if capability.tags.contains(tag) {
                score += 0.1;
            }
        }

        score.min(1.0)
    }
}

/// Current status of a worker node.
#[derive(Debug, Clone)]
pub struct WorkerStatus {
    /// The node identifier.
    pub id: NodeId,
    /// The node's capabilities.
    pub capability: WorkerCapability,
    /// Current load as a fraction (0.0 = idle, 1.0 = fully loaded).
    pub load_pct: f32,
    /// Number of tasks currently assigned.
    pub task_count: u32,
}

impl WorkerStatus {
    /// Create a new worker status.
    #[must_use]
    pub fn new(id: NodeId, capability: WorkerCapability, load_pct: f32, task_count: u32) -> Self {
        Self {
            id,
            capability,
            load_pct,
            task_count,
        }
    }
}

/// Selects the best worker for a given task based on affinity and load.
pub struct WorkloadBalancer;

impl WorkloadBalancer {
    /// Assign a task to the worker with the highest effective score.
    ///
    /// Effective score = `affinity × (1 - load_pct)`.
    /// Returns `None` if no worker can satisfy the task requirements.
    #[must_use]
    pub fn assign_task(
        workers: &[WorkerStatus],
        requirements: &TaskRequirements,
    ) -> Option<NodeId> {
        workers
            .iter()
            .filter_map(|w| {
                let affinity = AffinityScore::compute(&w.capability, requirements);
                if affinity == 0.0 {
                    return None;
                }
                let effective = affinity * (1.0 - w.load_pct.clamp(0.0, 1.0));
                Some((w.id, effective))
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(id, _)| id)
    }
}

/// Reason why a task is being migrated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MigrationReason {
    /// Redistribute load across workers.
    LoadBalance,
    /// Source node has failed.
    NodeFailure,
    /// Source node has exhausted resources.
    ResourceExhausted,
    /// User explicitly requested the migration.
    UserRequest,
}

/// Describes a planned task migration.
#[derive(Debug, Clone)]
pub struct TaskMigration {
    /// Identifier of the task being migrated.
    pub task_id: String,
    /// Node the task is migrating from.
    pub from_node: NodeId,
    /// Node the task is migrating to.
    pub to_node: NodeId,
    /// Reason for the migration.
    pub reason: MigrationReason,
    /// Amount of state data to transfer in bytes.
    pub data_size_bytes: u64,
}

/// Plans task migrations to rebalance the cluster.
pub struct MigrationPlanner;

impl MigrationPlanner {
    /// Produce a list of migration recommendations.
    ///
    /// A migration is suggested when a worker's load exceeds the threshold
    /// and another worker's load is below `1.0 - threshold_imbalance`.
    #[must_use]
    pub fn plan(workers: &[WorkerStatus], threshold_imbalance: f32) -> Vec<TaskMigration> {
        let mut migrations = Vec::new();

        // Identify overloaded and underloaded workers
        let overloaded: Vec<&WorkerStatus> = workers
            .iter()
            .filter(|w| w.load_pct > threshold_imbalance)
            .collect();

        let underloaded: Vec<&WorkerStatus> = workers
            .iter()
            .filter(|w| w.load_pct < 1.0 - threshold_imbalance)
            .collect();

        for (idx, over) in overloaded.iter().enumerate() {
            if let Some(under) = underloaded.get(idx) {
                migrations.push(TaskMigration {
                    task_id: format!("task-from-{}", over.id.inner()),
                    from_node: over.id,
                    to_node: under.id,
                    reason: MigrationReason::LoadBalance,
                    data_size_bytes: 0,
                });
            }
        }

        migrations
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu_worker(id: u64, cores: u32, load: f32) -> WorkerStatus {
        WorkerStatus::new(
            NodeId::new(id),
            WorkerCapability::new(cores, 16.0, 0.0, 1000, vec![]),
            load,
            (load * 10.0) as u32,
        )
    }

    fn gpu_worker(id: u64, vram: f32, load: f32) -> WorkerStatus {
        WorkerStatus::new(
            NodeId::new(id),
            WorkerCapability::new(8, 32.0, vram, 10_000, vec!["gpu".to_string()]),
            load,
            (load * 10.0) as u32,
        )
    }

    fn basic_requirements() -> TaskRequirements {
        TaskRequirements::new(4, 8.0, false, 0.0, vec![])
    }

    fn gpu_requirements() -> TaskRequirements {
        TaskRequirements::new(4, 8.0, true, 8.0, vec!["gpu".to_string()])
    }

    #[test]
    fn test_affinity_meets_minimums() {
        let cap = WorkerCapability::new(8, 16.0, 0.0, 1000, vec![]);
        let req = basic_requirements();
        let score = AffinityScore::compute(&cap, &req);
        assert!((score - 0.5).abs() < 1e-5, "Expected 0.5, got {score}");
    }

    #[test]
    fn test_affinity_fails_cpu() {
        let cap = WorkerCapability::new(2, 16.0, 0.0, 1000, vec![]);
        let req = TaskRequirements::new(4, 8.0, false, 0.0, vec![]);
        assert_eq!(AffinityScore::compute(&cap, &req), 0.0);
    }

    #[test]
    fn test_affinity_fails_memory() {
        let cap = WorkerCapability::new(8, 4.0, 0.0, 1000, vec![]);
        let req = TaskRequirements::new(4, 8.0, false, 0.0, vec![]);
        assert_eq!(AffinityScore::compute(&cap, &req), 0.0);
    }

    #[test]
    fn test_affinity_gpu_bonus() {
        let cap = WorkerCapability::new(8, 32.0, 16.0, 10_000, vec!["gpu".to_string()]);
        let req = gpu_requirements();
        let score = AffinityScore::compute(&cap, &req);
        // 0.5 base + 0.3 gpu + 0.1 tag = 0.9
        assert!((score - 0.9).abs() < 1e-5, "Expected 0.9, got {score}");
    }

    #[test]
    fn test_affinity_tag_bonus() {
        let cap = WorkerCapability::new(
            8,
            16.0,
            0.0,
            1000,
            vec!["av1".to_string(), "fast".to_string()],
        );
        let req = TaskRequirements::new(
            4,
            8.0,
            false,
            0.0,
            vec!["av1".to_string(), "fast".to_string()],
        );
        let score = AffinityScore::compute(&cap, &req);
        // 0.5 + 0.1 + 0.1 = 0.7
        assert!((score - 0.7).abs() < 1e-5, "Expected 0.7, got {score}");
    }

    #[test]
    fn test_affinity_capped_at_one() {
        let tags: Vec<String> = (0..10).map(|i| format!("tag{i}")).collect();
        let cap = WorkerCapability::new(8, 16.0, 16.0, 10_000, tags.clone());
        let req = TaskRequirements::new(4, 8.0, true, 8.0, tags);
        let score = AffinityScore::compute(&cap, &req);
        assert!(score <= 1.0, "Score must not exceed 1.0");
    }

    #[test]
    fn test_workload_balancer_selects_best() {
        let workers = vec![
            cpu_worker(1, 8, 0.9), // heavily loaded
            cpu_worker(2, 8, 0.1), // lightly loaded
            cpu_worker(3, 8, 0.5),
        ];
        let req = basic_requirements();
        let assigned = WorkloadBalancer::assign_task(&workers, &req);
        assert_eq!(assigned, Some(NodeId::new(2)));
    }

    #[test]
    fn test_workload_balancer_no_capable_worker() {
        let workers = vec![WorkerStatus::new(
            NodeId::new(1),
            WorkerCapability::new(2, 4.0, 0.0, 1000, vec![]),
            0.0,
            0,
        )];
        let req = TaskRequirements::new(8, 32.0, false, 0.0, vec![]);
        let assigned = WorkloadBalancer::assign_task(&workers, &req);
        assert!(assigned.is_none());
    }

    #[test]
    fn test_workload_balancer_gpu_task() {
        let workers = vec![cpu_worker(1, 8, 0.1), gpu_worker(2, 16.0, 0.3)];
        let req = gpu_requirements();
        let assigned = WorkloadBalancer::assign_task(&workers, &req);
        assert_eq!(assigned, Some(NodeId::new(2)));
    }

    #[test]
    fn test_migration_planner_suggests_migrations() {
        let workers = vec![
            cpu_worker(1, 8, 0.9), // overloaded
            cpu_worker(2, 8, 0.1), // underloaded
        ];
        let migrations = MigrationPlanner::plan(&workers, 0.7);
        assert_eq!(migrations.len(), 1);
        assert_eq!(migrations[0].from_node, NodeId::new(1));
        assert_eq!(migrations[0].to_node, NodeId::new(2));
        assert_eq!(migrations[0].reason, MigrationReason::LoadBalance);
    }

    #[test]
    fn test_migration_planner_no_migrations_balanced() {
        let workers = vec![cpu_worker(1, 8, 0.5), cpu_worker(2, 8, 0.5)];
        let migrations = MigrationPlanner::plan(&workers, 0.7);
        assert!(migrations.is_empty());
    }
}
