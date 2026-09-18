//! Intelligent job distribution with data locality awareness.
//!
//! `JobDistributor` wraps a [`LoadBalancer`] and adds a locality layer:
//! when a job carries a `data_locality_hint` pointing to a storage server, the
//! distributor first tries to assign the job to a worker that is topologically
//! close to that storage server (as configured via [`JobDistributor::register_locality`]).
//! Only when no local worker has capacity does it fall back to the underlying
//! load-balancer.
//!
//! Batch distribution respects job priority — higher-priority jobs are assigned
//! before lower-priority ones.

use std::collections::HashMap;

use crate::load_balancer::{LoadBalancer, LoadBalancingStrategy};

/// A single farm job to be distributed to a worker.
#[derive(Debug, Clone)]
pub struct FarmJob {
    /// Globally unique job identifier.
    pub id: String,
    /// Semantic job type (e.g., "transcode", "qc", "thumbnail").
    pub job_type: String,
    /// Scheduling priority; higher values are dispatched first in batch mode.
    pub priority: u8,
    /// Paths of the input assets consumed by this job.
    pub input_paths: Vec<String>,
    /// Minimum CPU cores required to execute the job.
    pub required_cpu_cores: f32,
    /// Minimum RAM required to execute the job (mebibytes).
    pub required_memory_mb: u64,
    /// Expected wall-clock execution time (seconds).
    pub estimated_duration_secs: u64,
    /// Optional hint identifying which storage server or region holds the
    /// input data.  When set, the distributor will try to assign a worker
    /// that is close to this server.
    pub data_locality_hint: Option<String>,
    /// Whether the job requires a GPU worker.
    pub gpu_required: bool,
}

/// The result of a single distribution decision.
#[derive(Debug, Clone)]
pub struct DistributionDecision {
    /// The job that was assigned.
    pub job_id: String,
    /// The worker that will execute the job.
    pub assigned_worker: String,
    /// Composite score produced during selection (strategy-specific).
    pub score: f32,
    /// Human-readable explanation of why this worker was chosen.
    pub reason: String,
}

/// Distributes farm jobs to workers, combining locality awareness with a
/// configurable [`LoadBalancer`].
pub struct JobDistributor {
    /// The underlying load balancer used for non-locality decisions.
    pub load_balancer: LoadBalancer,
    /// Maps storage-server identifiers to the list of worker IDs that are
    /// network-adjacent to that server.
    pub locality_map: HashMap<String, Vec<String>>,
}

impl JobDistributor {
    /// Create a new distributor backed by a load balancer using `strategy`.
    #[must_use]
    pub fn new(strategy: LoadBalancingStrategy) -> Self {
        Self {
            load_balancer: LoadBalancer::new(strategy),
            locality_map: HashMap::new(),
        }
    }

    /// Declare which workers are co-located with (or close to) `storage_server`.
    pub fn register_locality(&mut self, storage_server: &str, worker_ids: Vec<String>) {
        self.locality_map
            .insert(storage_server.to_string(), worker_ids);
    }

    /// Distribute a single job to the most appropriate available worker.
    ///
    /// Decision priority:
    /// 1. If `data_locality_hint` is set and at least one registered local
    ///    worker has capacity → pick the least-loaded local worker (reason:
    ///    `"locality-preferred"`).
    /// 2. Delegate to the configured load balancer (reason: `"load-balanced"`).
    /// 3. If no worker is available at all, return `None`.
    pub fn distribute(&mut self, job: &FarmJob) -> Option<DistributionDecision> {
        // 1. Attempt locality-aware placement.
        if let Some(ref hint) = job.data_locality_hint {
            if let Some(local_ids) = self.locality_map.get(hint).cloned() {
                // Find the local worker with the fewest active jobs that has capacity.
                let local_available: Vec<&crate::load_balancer::WorkerCapacity> = self
                    .load_balancer
                    .available_workers()
                    .into_iter()
                    .filter(|w| local_ids.contains(&w.worker_id))
                    .collect();

                if let Some(best) = local_available.into_iter().min_by_key(|w| w.active_jobs) {
                    return Some(DistributionDecision {
                        job_id: job.id.clone(),
                        assigned_worker: best.worker_id.clone(),
                        score: 1.0,
                        reason: "locality-preferred".to_string(),
                    });
                }
            }
        }

        // 2. Fall back to configured load balancer.
        if let Some(worker_id) = self.load_balancer.select(
            &job.job_type,
            job.required_cpu_cores,
            job.required_memory_mb,
        ) {
            return Some(DistributionDecision {
                job_id: job.id.clone(),
                assigned_worker: worker_id,
                score: 0.5,
                reason: "load-balanced".to_string(),
            });
        }

        // 3. No worker available.
        None
    }

    /// Distribute a batch of jobs in descending priority order.
    ///
    /// After each successful placement the assigned worker's `active_jobs`
    /// count is incremented so that subsequent jobs in the same batch see an
    /// accurate view of remaining capacity.
    ///
    /// Jobs for which no worker is currently available are silently skipped —
    /// the returned `Vec` may be shorter than `jobs`.
    pub fn distribute_batch(&mut self, jobs: &[FarmJob]) -> Vec<DistributionDecision> {
        // Sort indices by priority descending (stable sort preserves original
        // order for equal priorities).
        let mut indices: Vec<usize> = (0..jobs.len()).collect();
        indices.sort_by(|&a, &b| jobs[b].priority.cmp(&jobs[a].priority));

        let mut decisions = Vec::with_capacity(jobs.len());
        for idx in indices {
            if let Some(decision) = self.distribute(&jobs[idx]) {
                // Increment the assigned worker's active_jobs so the next
                // iteration in this batch sees accurate capacity.
                let worker_id = decision.assigned_worker.clone();
                if let Some(w) = self
                    .load_balancer
                    .workers
                    .iter_mut()
                    .find(|w| w.worker_id == worker_id)
                {
                    w.active_jobs = w.active_jobs.saturating_add(1);
                }
                decisions.push(decision);
            }
        }
        decisions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::load_balancer::{LoadBalancingStrategy, WorkerCapacity};

    fn make_worker(id: &str, active: u32, max: u32) -> WorkerCapacity {
        WorkerCapacity {
            worker_id: id.to_string(),
            weight: 1,
            cpu_cores_available: 8.0,
            memory_mb_available: 16384,
            active_jobs: active,
            max_jobs: max,
            avg_latency_ms: 10.0,
            job_type_affinity: vec![],
            gpu_count: 0,
            gpu_memory_mb_available: 0,
            network_bandwidth_mbps: 1000.0,
            disk_io_mbps: 500.0,
        }
    }

    fn make_job(id: &str, priority: u8, locality: Option<&str>) -> FarmJob {
        FarmJob {
            id: id.to_string(),
            job_type: "transcode".to_string(),
            priority,
            input_paths: vec![],
            required_cpu_cores: 1.0,
            required_memory_mb: 512,
            estimated_duration_secs: 60,
            data_locality_hint: locality.map(|s| s.to_string()),
            gpu_required: false,
        }
    }

    fn distributor_with_workers(workers: Vec<WorkerCapacity>) -> JobDistributor {
        let mut d = JobDistributor::new(LoadBalancingStrategy::LeastConnections);
        for w in workers {
            d.load_balancer.add_worker(w);
        }
        d
    }

    // ── Basic distribution ────────────────────────────────────────────────────

    #[test]
    fn test_distribute_no_locality_uses_load_balancer() {
        let mut d =
            distributor_with_workers(vec![make_worker("w1", 0, 4), make_worker("w2", 2, 4)]);
        let job = make_job("j1", 5, None);
        let decision = d.distribute(&job).expect("should distribute");
        assert_eq!(decision.job_id, "j1");
        assert_eq!(decision.reason, "load-balanced");
        assert_eq!(decision.assigned_worker, "w1"); // least connections
    }

    #[test]
    fn test_distribute_with_locality_prefers_local_worker() {
        let mut d =
            distributor_with_workers(vec![make_worker("w1", 0, 4), make_worker("w2", 0, 4)]);
        d.register_locality("nas-01", vec!["w2".to_string()]);
        let job = make_job("j1", 5, Some("nas-01"));
        let decision = d.distribute(&job).expect("should distribute");
        assert_eq!(decision.assigned_worker, "w2");
        assert_eq!(decision.reason, "locality-preferred");
    }

    #[test]
    fn test_distribute_falls_back_when_local_worker_full() {
        let mut d = distributor_with_workers(vec![
            make_worker("w1", 0, 4),
            make_worker("w2", 4, 4), // full
        ]);
        d.register_locality("nas-01", vec!["w2".to_string()]);
        let job = make_job("j1", 5, Some("nas-01"));
        let decision = d.distribute(&job).expect("should fall back to w1");
        assert_eq!(decision.assigned_worker, "w1");
        assert_eq!(decision.reason, "load-balanced");
    }

    #[test]
    fn test_distribute_returns_none_when_all_full() {
        let mut d =
            distributor_with_workers(vec![make_worker("w1", 4, 4), make_worker("w2", 4, 4)]);
        let job = make_job("j1", 5, None);
        assert!(d.distribute(&job).is_none());
    }

    #[test]
    fn test_distribute_locality_picks_least_loaded_local() {
        let mut d = distributor_with_workers(vec![
            make_worker("w1", 3, 8), // local, busier
            make_worker("w2", 1, 8), // local, less busy
            make_worker("w3", 0, 8), // not local
        ]);
        d.register_locality("nas-01", vec!["w1".to_string(), "w2".to_string()]);
        let job = make_job("j1", 5, Some("nas-01"));
        let decision = d.distribute(&job).expect("should distribute");
        assert_eq!(decision.assigned_worker, "w2");
        assert_eq!(decision.reason, "locality-preferred");
    }

    // ── Batch distribution ────────────────────────────────────────────────────

    #[test]
    fn test_distribute_batch_respects_priority_order() {
        let mut d = distributor_with_workers(vec![
            make_worker("w1", 0, 1), // only one slot each
            make_worker("w2", 0, 1),
        ]);

        let jobs = vec![
            make_job("low", 1, None),
            make_job("high", 10, None),
            make_job("mid", 5, None),
        ];
        let decisions = d.distribute_batch(&jobs);
        // All three should be distributed (we have exactly 2 workers with 1 slot each).
        // high (10) then mid (5) get placed; low (1) finds no capacity.
        assert_eq!(decisions.len(), 2);
        assert_eq!(decisions[0].job_id, "high");
        assert_eq!(decisions[1].job_id, "mid");
    }

    #[test]
    fn test_distribute_batch_skips_jobs_with_no_worker() {
        let mut d = distributor_with_workers(vec![make_worker("w1", 0, 1)]);

        let jobs = vec![
            make_job("j1", 5, None),
            make_job("j2", 5, None), // no worker left
        ];
        let decisions = d.distribute_batch(&jobs);
        assert_eq!(decisions.len(), 1);
        assert_eq!(decisions[0].job_id, "j1");
    }

    #[test]
    fn test_distribute_batch_empty_returns_empty() {
        let mut d = distributor_with_workers(vec![make_worker("w1", 0, 4)]);
        let decisions = d.distribute_batch(&[]);
        assert!(decisions.is_empty());
    }

    #[test]
    fn test_distribute_batch_all_skip_when_overloaded() {
        let mut d =
            distributor_with_workers(vec![make_worker("w1", 4, 4), make_worker("w2", 4, 4)]);
        let jobs = vec![make_job("j1", 9, None), make_job("j2", 8, None)];
        let decisions = d.distribute_batch(&jobs);
        assert!(decisions.is_empty());
    }

    // ── Locality registration ─────────────────────────────────────────────────

    #[test]
    fn test_register_locality_multiple_servers() {
        let mut d = distributor_with_workers(vec![
            make_worker("w1", 0, 4),
            make_worker("w2", 0, 4),
            make_worker("w3", 0, 4),
        ]);
        d.register_locality("nas-01", vec!["w1".to_string()]);
        d.register_locality("nas-02", vec!["w2".to_string(), "w3".to_string()]);

        let j1 = make_job("j1", 5, Some("nas-01"));
        let d1 = d.distribute(&j1).expect("should distribute");
        assert_eq!(d1.assigned_worker, "w1");

        let j2 = make_job("j2", 5, Some("nas-02"));
        let d2 = d.distribute(&j2).expect("should distribute");
        assert!(d2.assigned_worker == "w2" || d2.assigned_worker == "w3");
    }

    #[test]
    fn test_distribute_unknown_locality_hint_falls_back() {
        let mut d = distributor_with_workers(vec![make_worker("w1", 0, 4)]);
        let job = make_job("j1", 5, Some("unknown-nas"));
        let decision = d.distribute(&job).expect("should fall back");
        assert_eq!(decision.reason, "load-balanced");
    }

    #[test]
    fn test_decision_fields_populated() {
        let mut d = distributor_with_workers(vec![make_worker("w1", 0, 4)]);
        let job = make_job("job-abc", 7, None);
        let decision = d.distribute(&job).expect("should distribute");
        assert_eq!(decision.job_id, "job-abc");
        assert!(!decision.assigned_worker.is_empty());
        assert!(!decision.reason.is_empty());
    }
}
